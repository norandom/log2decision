//! Contract tests of searching a store and reading its rows back (task 1.9;
//! requirements 5.2 to 5.6, 6.7 and 12).
//!
//! Synthetic logs with synthetic vectors (no encoder exists yet) and the mini fixture;
//! everything is written below the system temporary directory. Every result is compared
//! with a plain scan or a brute-force distance computation done here. The measurements
//! (heap against what was asked, HTTP CSIC 2010) are in `store_csic.rs`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    EncoderIdentity, InputAdapter, L2dError, Label, LabelledAdapter, LabelledFormat, LogRecord,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_store::{
    DISTANCE, Filter, GROUP, ID, INDEX_TEXT_FACTOR, LABEL, LogFlags, OCCURRENCES,
    PHRASE_BYTE_BYTES, QUERY, SEARCH_FIXED_BYTES, SOURCE, Store, StoreError, TEXT,
    TEXT_READ_FACTOR, TRUNCATED, VECTOR, VectorOrigin, hits_schema, log_id, rows_frame,
    rows_schema, stored_schema,
};
use rusqlite::{Connection, OpenFlags};

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const REVISION: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const OTHER_REVISION: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
const DIM: u16 = 8;
const FILE: &str = "a.log";

fn identity() -> EncoderIdentity {
    EncoderIdentity {
        name: "bytecnn".to_string(),
        revision: REVISION.to_string(),
        dimension: DIM,
    }
}

/// A project root in the system temporary directory, removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-search-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("the scratch root can be created");
        Self { root }
    }

    fn paths(&self) -> ProjectPaths {
        ProjectPaths::new(&self.root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A budget with a fixed free-memory reading, so the tests do not depend on the host.
fn budget_with(limit: Option<u64>, batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).expect("the budget is usable")
}

fn budget() -> MemoryBudget {
    budget_with(None, None)
}

fn record(text: &str, label: Label, group: Option<&str>, occurrences: u32) -> LogRecord {
    LogRecord {
        text: text.to_string(),
        label,
        group: group.map(str::to_string),
        first_position: 1,
        occurrences,
    }
}

fn logs_of(records: &[LogRecord]) -> LogsFrame {
    LogsFrame::from_records(records).expect("the records make a logs frame")
}

/// Good logs without a group, one per text.
fn plain(texts: &[&str]) -> LogsFrame {
    let records: Vec<LogRecord> = texts
        .iter()
        .map(|text| record(text, Label::Good, None, 1))
        .collect();
    logs_of(&records)
}

/// `rows` synthetic logs, every third one bad, over five groups.
fn synthetic_logs(rows: u32) -> LogsFrame {
    let records: Vec<LogRecord> = (0..rows)
        .map(|index| LogRecord {
            text: format!(
                "GET http://localhost:8080/shop/item.jsp?id={index}&q=value{}",
                index * 7
            ),
            label: if index % 3 == 0 {
                Label::Bad
            } else {
                Label::Good
            },
            group: Some(format!("/shop/{}", index % 5)),
            first_position: u64::from(index) + 1,
            occurrences: 1 + index % 4,
        })
        .collect();
    logs_of(&records)
}

/// The 169 distinct logs of the mini fixture.
fn fixture_logs() -> LogsFrame {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv");
    let (logs, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&path)
        .expect("the fixture is readable");
    assert_eq!(logs.len(), 169);
    logs
}

/// FNV-1a over the bytes of `text`: a fixed pseudo-random number per text.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A fixed unit vector for `text`.
fn unit_vector(text: &str) -> Vec<f32> {
    let mut state = fnv(text) | 1;
    let mut values: Vec<f32> = (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    for value in &mut values {
        *value /= norm;
    }
    values
}

/// A log longer than this many bytes counts as truncated in these tests.
const INPUT_BYTES: usize = 60;

/// The three flags of a text in these tests: made up from its length, so that the
/// three columns differ from each other.
fn flags_for(text: &str) -> LogFlags {
    LogFlags {
        truncated: text.len() > INPUT_BYTES,
        text_longer_than_input: text.len() > INPUT_BYTES / 2,
        empty: text.len().is_multiple_of(3),
    }
}

/// The rows frame for `logs` from `source`; `vector` gives the vector of a text.
fn rows_by(logs: &LogsFrame, source: &str, vector: impl Fn(&str) -> Vec<f32>) -> DataFrame {
    let vectors: Vec<f32> = logs.records().flat_map(|r| vector(&r.text)).collect();
    let flags: Vec<LogFlags> = logs.records().map(|r| flags_for(&r.text)).collect();
    rows_frame(logs, source, &vectors, DIM, &flags).expect("the logs make a rows frame")
}

fn rows(logs: &LogsFrame, source: &str) -> DataFrame {
    rows_by(logs, source, unit_vector)
}

/// A store named `s` that holds `logs` from `source`.
fn store_of(scratch: &Scratch, logs: &LogsFrame, source: &str) -> Store {
    let mut store = Store::create(
        &scratch.paths(),
        "s",
        VectorOrigin::without_release(&identity()),
    )
    .unwrap();
    store
        .add_with(
            &rows(logs, source),
            VectorOrigin::without_release(&identity()),
            &mut budget(),
        )
        .expect("the logs are added");
    store
}

fn contains(phrases: &[&str]) -> Filter {
    Filter {
        contains: phrases.iter().map(|phrase| phrase.to_string()).collect(),
        ..Filter::default()
    }
}

fn strings(frame: &DataFrame, column: &str) -> Vec<Option<String>> {
    frame
        .column(column)
        .unwrap_or_else(|error| panic!("column {column}: {error}"))
        .str()
        .unwrap()
        .iter()
        .map(|value| value.map(str::to_string))
        .collect()
}

fn texts(frame: &DataFrame) -> Vec<String> {
    strings(frame, TEXT).into_iter().flatten().collect()
}

fn ids(frame: &DataFrame) -> Vec<String> {
    strings(frame, ID).into_iter().flatten().collect()
}

fn vectors(frame: &DataFrame) -> Vec<Vec<f32>> {
    frame
        .column(VECTOR)
        .unwrap()
        .list()
        .unwrap()
        .amortized_iter()
        .map(|vector| {
            let vector = vector.expect("no vector is absent");
            vector.as_ref().f32().unwrap().iter().flatten().collect()
        })
        .collect()
}

fn distances(frame: &DataFrame) -> Vec<f32> {
    let column = frame.column(DISTANCE).unwrap().f32().unwrap();
    column.iter().flatten().collect()
}

fn queries(frame: &DataFrame) -> Vec<u32> {
    let column = frame.column(QUERY).unwrap().u32().unwrap();
    column.iter().flatten().collect()
}

/// The texts of `logs` that satisfy `keep`, in the order of the frame.
fn scan(logs: &LogsFrame, keep: impl Fn(&str) -> bool) -> Vec<String> {
    logs.records()
        .map(|record| record.text)
        .filter(|text| keep(text))
        .collect()
}

/// The texts a search for `filter` returns, without a limit.
fn found(store: &Store, filter: &Filter) -> Vec<String> {
    texts(
        &store
            .text_search(filter, usize::MAX)
            .unwrap_or_else(|error| panic!("{filter:?}: {error}")),
    )
}

/// Stored logs as the brute-force search takes them: text and vector, in the order they
/// were stored.
type Candidates = Vec<(String, Vec<f32>)>;

/// Squared L2 in double precision: the reference for the distances.
fn reference_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum()
}

/// The `k` nearest of `candidates` (text and vector, in insertion order) to `query` by
/// brute force: nearest first, earlier stored first among equals.
fn brute_force(candidates: &[(String, Vec<f32>)], query: &[f32], k: usize) -> Vec<(String, f64)> {
    let mut all: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(index, (_, vector))| (index, reference_distance(vector, query)))
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all.into_iter()
        .map(|(index, distance)| (candidates[index].0.clone(), distance))
        .collect()
}

fn with_vectors(logs: &LogsFrame) -> Vec<(String, Vec<f32>)> {
    logs.records()
        .map(|record| (record.text.clone(), unit_vector(&record.text)))
        .collect()
}

/// The hits of `query` are the `k` nearest of `candidates`, nearest first. Two logs
/// whose reference distances differ by less than the tolerance may come in either order.
fn assert_nearest(
    hits: &DataFrame,
    query_index: u32,
    candidates: &[(String, Vec<f32>)],
    query: &[f32],
    k: usize,
) {
    let expected = brute_force(candidates, query, k);
    let rows: Vec<usize> = queries(hits)
        .iter()
        .enumerate()
        .filter(|(_, q)| **q == query_index)
        .map(|(row, _)| row)
        .collect();
    assert_eq!(rows.len(), expected.len(), "query {query_index}");
    let all_texts = texts(hits);
    let all_distances = distances(hits);
    let mut seen = std::collections::HashSet::new();
    let mut previous = 0.0f32;
    for (row, (_, want)) in rows.iter().zip(&expected) {
        let text = &all_texts[*row];
        let got = all_distances[*row];
        assert!(seen.insert(text.clone()), "{text} is returned twice");
        assert!(got >= previous, "the distances of query {query_index} fall");
        previous = got;
        let vector = &candidates
            .iter()
            .find(|(candidate, _)| candidate == text)
            .unwrap_or_else(|| panic!("{text} is not a candidate"))
            .1;
        let own = reference_distance(vector, query);
        assert!(
            (f64::from(got) - own).abs() <= 1e-5,
            "query {query_index}, {text}: distance {got}, reference {own}"
        );
        assert!(
            (own - want).abs() <= 1e-5,
            "query {query_index}: {text} at {own} where the reference has {want}"
        );
    }
}
fn read_only(path: &Path) -> Connection {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("the store file opens read-only")
}

fn assert_fts_consistent(path: &Path) {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)
        .unwrap()
        .execute(
            "INSERT INTO logs_fts(logs_fts, rank) VALUES ('integrity-check', 1)",
            [],
        )
        .expect("the FTS5 index agrees with the logs table");
}

fn invalid_query(error: StoreError) -> String {
    match error {
        StoreError::InvalidQuery { detail } => detail,
        other => panic!("expected an invalid query, got {other:?}"),
    }
}

fn refusal(error: StoreError) -> String {
    match error {
        StoreError::Memory(L2dError::Refused { reason }) => reason,
        other => panic!("expected a memory refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// Text search (5.2)
// ---------------------------------------------------------------------------------------

/// A containment query returns the stored rows whose text contains the phrase, in the
/// order they were stored, with every column of the stored row.
#[test]
fn text_search_returns_the_stored_rows_that_contain_the_phrase() {
    let scratch = Scratch::new("contain");
    let logs = fixture_logs();
    let source = "tests/fixtures/mini-labelled.csv";
    let store = store_of(&scratch, &logs, source);

    for phrase in ["login", "select", "/shop", "id=", "../"] {
        let lower = phrase.to_ascii_lowercase();
        let expected = scan(&logs, |text| text.to_ascii_lowercase().contains(&lower));
        assert_eq!(found(&store, &contains(&[phrase])), expected, "{phrase}");
    }
    assert!(
        !found(&store, &contains(&["id="])).is_empty(),
        "the fixture has logs with a parameter"
    );

    // Every column, for all rows: an empty filter matches every log.
    let all = store.text_search(&Filter::default(), usize::MAX).unwrap();
    assert_eq!(**all.schema(), stored_schema());
    assert_eq!(all.height(), 169);
    let expected = rows(&logs, source);
    assert_eq!(
        ids(&all),
        scan(&logs, |_| true)
            .iter()
            .map(|text| log_id(source, text))
            .collect::<Vec<_>>()
    );
    for column in [TEXT, SOURCE, OCCURRENCES, LABEL, GROUP, TRUNCATED] {
        assert!(
            all.column(column)
                .unwrap()
                .equals_missing(expected.column(column).unwrap()),
            "column {column} came back changed"
        );
    }
    assert_eq!(vectors(&all), vectors(&expected));

    // The limit keeps the first rows in insertion order.
    let first = store.text_search(&Filter::default(), 5).unwrap();
    assert_eq!(texts(&first), texts(&all)[..5]);
    assert_eq!(
        store.text_search(&Filter::default(), 0).unwrap().height(),
        0
    );
    assert_eq!(
        **store
            .text_search(&contains(&["no log holds this"]), 10)
            .unwrap()
            .schema(),
        stored_schema()
    );
}

/// The stored schema is the rows frame with the id in front.
#[test]
fn the_stored_and_hits_frames_have_the_documented_schemas() {
    let list = |schema: &l2d_core::polars::prelude::Schema| -> Vec<String> {
        schema
            .iter()
            .map(|(name, dtype)| format!("{name} {dtype}"))
            .collect()
    };
    let mut expected = vec!["id str".to_string()];
    expected.extend(list(&rows_schema()));
    assert_eq!(list(&stored_schema()), expected);
    assert_eq!(
        list(&hits_schema()),
        [
            "query u32",
            "distance f32",
            "id str",
            "text str",
            "source str",
            "occurrences u32",
            "label str",
            "group str",
            "truncated bool",
            "text_longer_than_input bool",
            "empty bool"
        ]
    );
}

/// The index ignores case; a case-exact search is answered correctly all the same.
#[test]
fn text_search_ignores_case_unless_the_search_is_case_exact() {
    let scratch = Scratch::new("case");
    let logs = plain(&[
        "GET /Login?user=Admin",
        "get /login?user=admin",
        "GET /LOGIN?USER=ADMIN",
        "POST /logout",
        "ÄÖÜ im Pfad /straße",
        "äöü im pfad /STRASSE",
    ]);
    let store = store_of(&scratch, &logs, FILE);

    assert_eq!(
        found(&store, &contains(&["/LOGIN"])),
        [
            "GET /Login?user=Admin",
            "get /login?user=admin",
            "GET /LOGIN?USER=ADMIN"
        ]
    );
    let exact = |phrases: &[&str]| Filter {
        case_exact: true,
        ..contains(phrases)
    };
    assert_eq!(
        found(&store, &exact(&["/LOGIN"])),
        ["GET /LOGIN?USER=ADMIN"]
    );
    assert_eq!(
        found(&store, &exact(&["/Login"])),
        ["GET /Login?user=Admin"]
    );
    assert_eq!(found(&store, &exact(&["/lOGIN"])), [] as [&str; 0]);

    // Beyond ASCII the index folds case as well; it does not expand characters.
    assert_eq!(
        found(&store, &contains(&["äöü"])),
        ["ÄÖÜ im Pfad /straße", "äöü im pfad /STRASSE"]
    );
    assert_eq!(found(&store, &exact(&["äöü"])), ["äöü im pfad /STRASSE"]);
    assert_eq!(
        found(&store, &contains(&["straße"])),
        ["ÄÖÜ im Pfad /straße"]
    );
}

/// The trigram index cannot answer a phrase of one or two characters; a scan does, with
/// the same meaning, and the store says so.
#[test]
fn phrases_shorter_than_three_characters_are_answered_by_a_scan() {
    let scratch = Scratch::new("short");
    let logs = plain(&[
        "GET /a?x=1",
        "GET /b?X=2",
        "POST /c",
        "Ä",
        "ä und ö",
        "no question mark here",
    ]);
    let store = store_of(&scratch, &logs, FILE);

    assert_eq!(
        found(&store, &contains(&["?"])),
        ["GET /a?x=1", "GET /b?X=2"]
    );
    assert_eq!(
        found(&store, &contains(&["x="])),
        ["GET /a?x=1", "GET /b?X=2"]
    );
    assert_eq!(found(&store, &contains(&["ä"])), ["Ä", "ä und ö"]);
    let exact = |phrases: &[&str]| Filter {
        case_exact: true,
        ..contains(phrases)
    };
    assert_eq!(found(&store, &exact(&["x="])), ["GET /a?x=1"]);
    assert_eq!(found(&store, &exact(&["X="])), ["GET /b?X=2"]);
    assert_eq!(found(&store, &exact(&["Ä"])), ["Ä"]);
    // A short phrase beside a long one: the index narrows, the scan decides.
    assert_eq!(found(&store, &contains(&["GET", "=2"])), ["GET /b?X=2"]);

    let explained = store.explain(&contains(&["x="])).unwrap();
    assert!(!explained.text_index);
    assert_eq!(explained.scanned_phrases, ["x="]);
    let explained = store.explain(&contains(&["GET", "=2"])).unwrap();
    assert!(explained.text_index);
    assert_eq!(explained.scanned_phrases, ["=2"]);
    let explained = store.explain(&contains(&["GET /"])).unwrap();
    assert!(explained.text_index);
    assert!(explained.scanned_phrases.is_empty());
    assert!(
        explained.steps.iter().any(|step| step.contains("logs_fts")),
        "{:?}",
        explained.steps
    );
}

/// The text of a phrase is searched for as it is: quotes, the operators of the text
/// index and the metacharacters of SQL and of LIKE mean nothing.
#[test]
fn a_phrase_is_never_read_as_query_syntax() {
    let scratch = Scratch::new("syntax");
    let stored = [
        r#"GET /a?q="quoted" value"#,
        r#"say ""twice"" here"#,
        "cats AND dogs",
        "cats and birds",
        "red OR blue",
        "NOT here",
        "NEAR(cats dogs, 5)",
        "wild* card",
        "wildcat card",
        "text : value",
        "text: other",
        "^GET at the start",
        "a ^GET in the middle",
        "o'brien'; DROP TABLE logs;--",
        "100% sure",
        "100 percent sure",
        "under_score",
        "underXscore",
        r"back\slash \n",
        "(paren) {brace} [bracket]",
        "col:text -minus +plus",
        "tab\there",
        "plain log line",
    ];
    let logs = plain(&stored);
    let store = store_of(&scratch, &logs, FILE);
    let path = scratch.paths().store_path("s");

    let phrases = [
        "\"",
        "\"\"",
        "\"quoted\"",
        "\"\"twice\"\"",
        "q=\"",
        "AND",
        "cats AND dogs",
        "cats AND birds",
        "\"cats\" AND \"dogs\"",
        "OR",
        "red OR blue",
        "red OR green",
        "NOT",
        "NOT here",
        "NEAR",
        "NEAR(cats dogs, 5)",
        "NEAR(",
        "*",
        "wild*",
        "wild* card",
        "wil*",
        ":",
        "text :",
        "text : value",
        "text:",
        "^",
        "^GET",
        "^GET at",
        "'",
        "';",
        "'; DROP TABLE logs;--",
        "' OR '1'='1",
        "%",
        "100%",
        "0% s",
        "_",
        "under_score",
        "r_s",
        "\\",
        "\\n",
        "back\\slash",
        "(",
        "(paren)",
        "{brace}",
        "[bracket]",
        "-minus",
        "+plus",
        "col:text",
        "\t",
        "tab\there",
        "?1",
        ":name",
        "logs_fts",
        "*:*",
        "text MATCH",
    ];
    for case_exact in [false, true] {
        for phrase in phrases {
            let expected = scan(&logs, |text| {
                if case_exact {
                    text.contains(phrase)
                } else {
                    text.to_lowercase().contains(&phrase.to_lowercase())
                }
            });
            let filter = Filter {
                case_exact,
                ..contains(&[phrase])
            };
            assert_eq!(
                found(&store, &filter),
                expected,
                "phrase {phrase:?}, case exact {case_exact}"
            );
        }
    }
    // The cases that would differ if a phrase were read as syntax.
    assert_eq!(
        found(&store, &contains(&["cats AND birds"])),
        ["cats and birds"]
    );
    assert_eq!(found(&store, &contains(&["red OR green"])), [] as [&str; 0]);
    assert_eq!(found(&store, &contains(&["wild*"])), ["wild* card"]);
    assert_eq!(
        found(&store, &contains(&["^GET"])),
        ["^GET at the start", "a ^GET in the middle"]
    );
    assert_eq!(found(&store, &contains(&["100%"])), ["100% sure"]);
    assert_eq!(found(&store, &contains(&["under_score"])), ["under_score"]);

    // The same holds for the exact text, the source and for several phrases at once.
    for text in stored {
        let filter = Filter {
            equals: Some(text.to_string()),
            ..Filter::default()
        };
        assert_eq!(found(&store, &filter), [text], "equals {text:?}");
        let filter = Filter {
            source: Some(text.to_string()),
            ..Filter::default()
        };
        assert_eq!(found(&store, &filter), [] as [&str; 0], "source {text:?}");
    }
    assert_eq!(
        found(&store, &contains(&["\"", "AND", "OR"])),
        [] as [&str; 0]
    );
    assert_eq!(
        found(&store, &contains(&["'; DROP", "logs;--"])),
        ["o'brien'; DROP TABLE logs;--"]
    );

    // Nothing was changed by any of it.
    assert_eq!(store.len().unwrap(), stored.len() as u64);
    drop(store);
    assert_fts_consistent(&path);
    let tables: i64 = read_only(&path)
        .query_row(
            "SELECT count(*) FROM sqlite_master WHERE name IN ('meta', 'logs', 'logs_fts')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tables, 3);
}

/// The exact query: the whole text, byte for byte, whatever its length.
#[test]
fn the_exact_query_matches_the_whole_text_only() {
    let scratch = Scratch::new("equals");
    let long = format!("GET /long?{}", "abcdefghij".repeat(300));
    let logs = plain(&[
        "GET /a",
        "GET /a?x=1",
        "get /a",
        "ab",
        "abc",
        "",
        long.as_str(),
    ]);
    let store = store_of(&scratch, &logs, FILE);
    let equals = |text: &str| Filter {
        equals: Some(text.to_string()),
        ..Filter::default()
    };
    assert_eq!(found(&store, &equals("GET /a")), ["GET /a"]);
    assert_eq!(found(&store, &equals("get /a")), ["get /a"]);
    assert_eq!(found(&store, &equals("GET /A")), [] as [&str; 0]);
    assert_eq!(found(&store, &equals("ab")), ["ab"]);
    assert_eq!(found(&store, &equals("")), [""]);
    assert_eq!(found(&store, &equals(&long)), [long.as_str()]);
    assert_eq!(
        found(&store, &equals(&long[..long.len() - 1])),
        [] as [&str; 0]
    );
    // With a phrase as well: both must hold.
    let both = Filter {
        contains: vec!["x=".to_string()],
        ..equals("GET /a?x=1")
    };
    assert_eq!(found(&store, &both), ["GET /a?x=1"]);
    let both = Filter {
        contains: vec!["y=".to_string()],
        ..equals("GET /a?x=1")
    };
    assert_eq!(found(&store, &both), [] as [&str; 0]);

    assert!(store.explain(&equals("ab")).unwrap().scanned_phrases == ["ab"]);
    assert!(store.explain(&equals("GET /a")).unwrap().text_index);
}

/// Several phrases must all be contained; phrases that cannot be searched are named.
#[test]
fn several_phrases_must_all_hold_and_unusable_ones_are_refused() {
    let scratch = Scratch::new("phrases");
    let long = format!("GET /long?{}", "AbCdEfGhIj".repeat(300));
    let logs = plain(&["GET /a?x=1&y=2", "GET /a?x=1", "POST /a?y=2", long.as_str()]);
    let store = store_of(&scratch, &logs, FILE);

    assert_eq!(
        found(&store, &contains(&["x=1", "y=2"])),
        ["GET /a?x=1&y=2"]
    );
    assert_eq!(
        found(&store, &contains(&["x=1", "y=2", "POST"])),
        [] as [&str; 0]
    );

    let detail = invalid_query(store.text_search(&contains(&["x=1", ""]), 10).unwrap_err());
    assert!(detail.contains("empty"), "{detail}");

    // No stored text holds a NUL character, so nothing matches; the phrase is not cut
    // at the NUL.
    assert_eq!(found(&store, &contains(&["GET\0 /a"])), [] as [&str; 0]);
    assert_eq!(found(&store, &contains(&["\0"])), [] as [&str; 0]);

    // A long phrase is searched whatever its length, in both case modes: the text
    // index is given short pieces of it and the comparison decides.
    let phrase = &long[4..2_500];
    let exact = Filter {
        case_exact: true,
        ..contains(&[phrase])
    };
    assert_eq!(found(&store, &exact), [long.as_str()]);
    let other_case = phrase.to_lowercase();
    assert_eq!(
        found(
            &store,
            &Filter {
                case_exact: true,
                ..contains(&[&other_case])
            }
        ),
        [] as [&str; 0]
    );
    assert_eq!(found(&store, &contains(&[phrase])), [long.as_str()]);
    assert_eq!(found(&store, &contains(&[&other_case])), [long.as_str()]);
    // One character off anywhere, and nothing is found, although every piece given to
    // the index is still in the text.
    let mut off = phrase.to_string();
    off.replace_range(1_000..1_001, "#");
    assert_eq!(found(&store, &contains(&[&off])), [] as [&str; 0]);
    assert!(store.explain(&contains(&[&off])).unwrap().text_index);
}

/// The text index is given short pieces of the phrases: the first, the last and the
/// middle characters of a phrase, at most four pieces in one search, every phrase its
/// first piece before any its second; a phrase without a piece is scanned.
#[test]
fn the_text_index_is_given_short_pieces_of_the_phrases() {
    let scratch = Scratch::new("pieces");
    let store = store_of(&scratch, &plain(&["GET /a?x=1"]), FILE);
    let pieces = |filter: &Filter| {
        let explained = store.explain(filter).unwrap();
        assert_eq!(explained.text_index, !explained.index_pieces.is_empty());
        (explained.index_pieces, explained.scanned_phrases)
    };
    let none: [&str; 0] = [];
    assert_eq!(pieces(&contains(&["ab"])), (vec![], vec!["ab".to_string()]));
    assert_eq!(pieces(&contains(&["abc"])).0, ["abc"]);
    assert_eq!(pieces(&contains(&["abcd"])).0, ["abcd"]);
    assert_eq!(pieces(&contains(&["abcde"])).0, ["abcd", "bcde"]);
    assert_eq!(
        pieces(&contains(&["0123456789abcdefghij"])).0,
        ["0123", "ghij", "89ab"]
    );
    // Several phrases: first pieces first, four in all.
    assert_eq!(
        pieces(&contains(&["0123456789", "abcdefghij", "xy", "KLMNOPQRST"])),
        (
            vec![
                "0123".to_string(),
                "abcd".to_string(),
                "KLMN".to_string(),
                "6789".to_string()
            ],
            vec!["xy".to_string()]
        )
    );
    assert_eq!(
        pieces(&contains(&["aaaa1", "bbbb2", "cccc3", "dddd4", "eeee5"])),
        (
            vec![
                "aaaa".to_string(),
                "bbbb".to_string(),
                "cccc".to_string(),
                "dddd".to_string()
            ],
            vec!["eeee5".to_string()]
        )
    );
    // The exact text gives pieces as well, after the phrases.
    let filter = Filter {
        equals: Some("GET /a?x=1".to_string()),
        ..contains(&["x=1"])
    };
    assert_eq!(pieces(&filter).0, ["x=1", "GET ", "?x=1", " /a?"]);
    // Ignoring case, a piece is taken only from characters the index folds as the
    // comparison does (characters without case, basic Latin, Greek and Cyrillic);`n    // case-exactly from any.
    let cherokee = "\u{13A0}\u{13A1}\u{13A2}\u{13A3}\u{13A4}";
    assert_eq!(
        pieces(&contains(&[cherokee])),
        (vec![], vec![cherokee.to_string()])
    );
    let mixed = format!("{cherokee}path=x{cherokee}");
    assert_eq!(pieces(&contains(&[&mixed])).0, ["path", "th=x", "ath="]);
    let exact = Filter {
        case_exact: true,
        ..contains(&[cherokee])
    };
    assert_eq!(
        pieces(&exact).0,
        [
            "\u{13A0}\u{13A1}\u{13A2}\u{13A3}",
            "\u{13A1}\u{13A2}\u{13A3}\u{13A4}"
        ]
    );
    assert_eq!(pieces(&contains(&["äöüß"])).0, ["äöüß"]);
    // No piece is given twice, within a phrase or across phrases, whatever its case:
    // the phrase offers its next piece.
    assert_eq!(pieces(&contains(&["abcabcabc"])).0, ["abca", "cabc"]);
    assert_eq!(pieces(&contains(&["aaaaaaaaaaaaaaaa"])).0, ["aaaa"]);
    assert_eq!(
        pieces(&contains(&["abcaX", "abcaY"])),
        (
            vec!["abca".to_string(), "bcaY".to_string(), "bcaX".to_string()],
            vec![]
        )
    );
    assert_eq!(
        pieces(&contains(&["ABCD", "abcd"])),
        (vec!["ABCD".to_string()], vec![])
    );
    assert_eq!(
        pieces(&contains(&[
            "aaaa1", "aaaa2", "aaaa3", "aaaa4", "aaaa5", "aaaa6"
        ]))
        .0,
        ["aaaa", "aaa2", "aaa3", "aaa4"]
    );
    assert_eq!(pieces(&Filter::default()).0, none);
}

/// The lower-case form of the definition of "ignoring case", written out again here:
/// the Unicode lower-case mapping when it is one character, else the character.
fn lower(character: char) -> char {
    let mut mapped = character.to_lowercase();
    match (mapped.next(), mapped.next()) {
        (Some(one), None) => one,
        _ => character,
    }
}

fn lowered(text: &str) -> String {
    text.chars().map(lower).collect()
}

/// "Ignoring case" is one definition, whether the text index narrowed the logs (a
/// phrase of three or more characters) or every log was read (a shorter phrase): the
/// cases in which the index's own folding differs from it, in both directions.
#[test]
fn ignoring_case_means_the_same_with_and_without_the_text_index() {
    let scratch = Scratch::new("folding");
    let stored = [
        "σσσ sigma",
        "σσς final sigma",
        "ΣΣΣ capital sigma",
        "\u{13A0}\u{13A0}\u{13A0} cherokee capital",
        "\u{AB70}\u{AB70}\u{AB70} cherokee small",
        "İİİ dotted capital i",
        "i\u{307}i\u{307}i\u{307} i with a dot above",
        "iii plain i",
        "III plain capital i",
        "ßßß sharp s",
        "ẞẞẞ capital sharp s",
        "SSSSSS double s",
        "ſſſ long s",
        "sss s",
        "\u{212A}\u{212A}\u{212A} kelvin",
        "kkk k",
        "KKK capital k",
        "\u{212B}\u{212B}\u{212B} angstrom",
        "ååå a with ring",
        "ÅÅÅ capital a with ring",
        "ǅǅǅ dz titlecase",
        "ǆǆǆ dz small",
        "ǄǄǄ dz capital",
        "\u{10400}\u{10400}\u{10400} deseret capital",
        "\u{10428}\u{10428}\u{10428} deseret small",
        "ÄÄÄ a umlaut capital",
        "äää a umlaut",
        "x\u{FFFE}y end",
        "x\u{FFFF}y end",
        "x\u{FFFD}y end",
        "ééé e acute",
        "e\u{301}e\u{301}e\u{301} e with a combining acute",
        "ЖЖЖ cyrillic",
        "жжж cyrillic small",
    ];
    let logs = plain(&stored);
    let store = store_of(&scratch, &logs, FILE);

    // Every run of one to four characters of every stored text, as it is and with
    // each character in upper case where that is one character.
    let mut phrases: Vec<String> = Vec::new();
    for text in stored {
        let characters: Vec<char> = text.chars().collect();
        for length in 1..=4 {
            for window in characters.windows(length) {
                let phrase: String = window.iter().collect();
                let upper: String = window
                    .iter()
                    .map(|character| {
                        let mut mapped = character.to_uppercase();
                        match (mapped.next(), mapped.next()) {
                            (Some(one), None) => one,
                            _ => *character,
                        }
                    })
                    .collect();
                phrases.push(upper);
                phrases.push(phrase);
            }
        }
    }
    phrases.sort();
    phrases.dedup();
    assert!(phrases.len() > 400);
    let mut through_index = 0;
    for phrase in &phrases {
        let wanted = lowered(phrase);
        let expected = scan(&logs, |text| lowered(text).contains(&wanted));
        assert_eq!(
            found(&store, &contains(&[phrase])),
            expected,
            "phrase {phrase:?}"
        );
        through_index += usize::from(store.explain(&contains(&[phrase])).unwrap().text_index);
        // Case-exactly the index may be given any characters.
        let exact = Filter {
            case_exact: true,
            ..contains(&[phrase])
        };
        assert_eq!(
            found(&store, &exact),
            scan(&logs, |text| text.contains(phrase.as_str())),
            "phrase {phrase:?}, case-exact"
        );
    }
    assert!(
        through_index > 100,
        "{through_index} phrases used the index"
    );

    // The cases by name.
    let one = |phrase: &str| found(&store, &contains(&[phrase]));
    // The index folds final sigma to sigma; the definition does not.
    assert_eq!(one("σσς"), ["σσς final sigma"]);
    assert_eq!(one("σσσ"), ["σσσ sigma", "ΣΣΣ capital sigma"]);
    assert_eq!(
        one("σσ"),
        ["σσσ sigma", "σσς final sigma", "ΣΣΣ capital sigma"]
    );
    // The index does not know the Cherokee case pairs; the definition does, at every
    // length of the phrase.
    let cherokee = [
        "\u{13A0}\u{13A0}\u{13A0} cherokee capital",
        "\u{AB70}\u{AB70}\u{AB70} cherokee small",
    ];
    assert_eq!(one("\u{13A0}\u{13A0}"), cherokee);
    assert_eq!(one("\u{13A0}\u{13A0}\u{13A0}"), cherokee);
    assert_eq!(one("\u{AB70}\u{AB70}\u{AB70} "), cherokee);
    // The dotted capital i stays itself.
    assert_eq!(one("İİ"), ["İİİ dotted capital i"]);
    assert_eq!(one("İİİ"), ["İİİ dotted capital i"]);
    assert_eq!(
        one("i\u{307}i"),
        ["i\u{307}i\u{307}i\u{307} i with a dot above"]
    );
    assert_eq!(one("III"), ["iii plain i", "III plain capital i"]);
    // No expansion.
    assert_eq!(one("ßßß"), ["ßßß sharp s", "ẞẞẞ capital sharp s"]);
    assert_eq!(one("SSS"), ["SSSSSS double s", "sss s"]);
    assert_eq!(one("ſſſ"), ["ſſſ long s"]);
    assert_eq!(
        one("KKK"),
        ["\u{212A}\u{212A}\u{212A} kelvin", "kkk k", "KKK capital k"]
    );
    assert_eq!(
        one("ååå"),
        [
            "\u{212B}\u{212B}\u{212B} angstrom",
            "ååå a with ring",
            "ÅÅÅ capital a with ring"
        ]
    );
    assert_eq!(
        one("ǆǆǆ"),
        ["ǅǅǅ dz titlecase", "ǆǆǆ dz small", "ǄǄǄ dz capital"]
    );
    assert_eq!(
        one("\u{10428}\u{10428}\u{10428}"),
        [
            "\u{10400}\u{10400}\u{10400} deseret capital",
            "\u{10428}\u{10428}\u{10428} deseret small"
        ]
    );
    // The index maps U+FFFE and U+FFFF to U+FFFD; the definition keeps them apart.
    assert_eq!(one("x\u{FFFE}y"), ["x\u{FFFE}y end"]);
    assert_eq!(one("x\u{FFFF}y e"), ["x\u{FFFF}y end"]);
    assert_eq!(one("X\u{FFFD}Y"), ["x\u{FFFD}y end"]);
    // No normalisation.
    assert_eq!(one("ééé"), ["ééé e acute"]);
    assert_eq!(l2d_store::fold("ÄbΣİ\u{212A}"), "äbσİk");
}

/// Wherever the text index is given a piece of a phrase that ignores case, it leaves
/// every log that the definition accepts: checked for every character of Unicode that
/// has a case partner. For each such character a log of four of it is stored, and it
/// is searched for by four of each character with the same lower-case form.
#[test]
fn the_text_index_never_narrows_what_ignoring_case_accepts() {
    use std::collections::BTreeMap;
    // The characters by their lower-case form, where two or more share one.
    let mut classes: BTreeMap<char, Vec<char>> = BTreeMap::new();
    for character in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        if character != '\0' {
            classes.entry(lower(character)).or_default().push(character);
        }
    }
    classes.retain(|_, members| members.len() > 1);
    assert!(classes.len() > 1_300, "{} case classes", classes.len());
    let members: Vec<char> = classes.values().flatten().copied().collect();
    let texts: Vec<String> = members
        .iter()
        .map(|character| character.to_string().repeat(4))
        .collect();
    let scratch = Scratch::new("case-classes");
    let logs = plain(&texts.iter().map(String::as_str).collect::<Vec<_>>());
    let store = store_of(&scratch, &logs, FILE);
    assert_eq!(store.len().unwrap(), members.len() as u64);

    let mut through_index = 0usize;
    let mut scanned = 0usize;
    let mut misses = Vec::new();
    for class in classes.values() {
        let expected: Vec<String> = class
            .iter()
            .map(|character| character.to_string().repeat(4))
            .collect();
        for character in class {
            let phrase = character.to_string().repeat(4);
            let filter = contains(&[&phrase]);
            let explained = store.explain(&filter).unwrap();
            // The rule that decides whether the index gets a piece.
            assert_eq!(
                explained.text_index,
                l2d_store::index_folds_alike(*character),
                "U+{:04X}",
                u32::from(*character)
            );
            if !explained.text_index {
                // Without a piece the index has no say; every 50th is searched all
                // the same.
                scanned += 1;
                if !scanned.is_multiple_of(50) {
                    continue;
                }
            } else {
                through_index += 1;
            }
            let mut got = found(&store, &filter);
            got.sort();
            let mut wanted = expected.clone();
            wanted.sort();
            if got != wanted {
                misses.push(format!(
                    "U+{:04X} finds {} of the {} logs of its class",
                    u32::from(*character),
                    got.len(),
                    wanted.len()
                ));
            }
        }
    }
    println!(
        "{} case classes, {} characters: {through_index} searched through the index, \
         {scanned} by a scan",
        classes.len(),
        members.len()
    );
    assert!(misses.is_empty(), "{misses:#?}");
    assert!(through_index > 300 && scanned > 1_000);
}
// ---------------------------------------------------------------------------------------
// Label and source (5.5)
// ---------------------------------------------------------------------------------------

fn two_sources(scratch: &Scratch) -> (Store, LogsFrame) {
    let logs = synthetic_logs(60);
    let mut store = store_of(scratch, &logs, "first.log");
    store
        .add_with(
            &rows(&logs, "second.log"),
            VectorOrigin::without_release(&identity()),
            &mut budget(),
        )
        .unwrap();
    (store, logs)
}

/// Restrictions by label and by source, alone, together and with a phrase, and the SQL
/// index each of them uses.
#[test]
fn restrictions_by_label_and_source_use_the_sql_indexes() {
    let scratch = Scratch::new("restrict");
    let (store, logs) = two_sources(&scratch);
    let all = scan(&logs, |_| true);
    let bad: Vec<String> = logs
        .records()
        .filter(|record| record.label == Label::Bad)
        .map(|record| record.text)
        .collect();
    assert_eq!(bad.len(), 20);

    let by_label = Filter {
        label: Some(Label::Bad),
        ..Filter::default()
    };
    let frame = store.text_search(&by_label, usize::MAX).unwrap();
    assert_eq!(texts(&frame), [bad.clone(), bad.clone()].concat());
    assert!(
        strings(&frame, LABEL)
            .iter()
            .all(|l| l.as_deref() == Some("bad"))
    );

    let by_source = Filter {
        source: Some("second.log".to_string()),
        ..Filter::default()
    };
    let frame = store.text_search(&by_source, usize::MAX).unwrap();
    assert_eq!(texts(&frame), all);
    assert!(
        strings(&frame, SOURCE)
            .iter()
            .all(|s| s.as_deref() == Some("second.log"))
    );
    assert_eq!(
        ids(&frame),
        all.iter()
            .map(|text| log_id("second.log", text))
            .collect::<Vec<_>>()
    );

    let both = Filter {
        label: Some(Label::Bad),
        source: Some("first.log".to_string()),
        ..Filter::default()
    };
    assert_eq!(found(&store, &both), bad);
    let with_phrase = Filter {
        contains: vec!["id=3".to_string()],
        ..both.clone()
    };
    assert_eq!(
        found(&store, &with_phrase),
        scan(&logs, |text| text.contains("id=3"))
            .into_iter()
            .filter(|text| bad.contains(text))
            .collect::<Vec<_>>()
    );
    let unknown = Filter {
        label: Some(Label::Unknown),
        ..Filter::default()
    };
    assert_eq!(found(&store, &unknown), [] as [&str; 0]);
    // The source is compared whole and case-exactly.
    for source in ["second", "SECOND.LOG", "second.log "] {
        let filter = Filter {
            source: Some(source.to_string()),
            ..Filter::default()
        };
        assert_eq!(found(&store, &filter), [] as [&str; 0], "{source}");
    }

    // How SQLite finds the rows: through the index of the restriction, and without
    // sorting them afterwards (the rows come in insertion order as they are read).
    let steps = |filter: &Filter| store.explain(filter).unwrap().steps;
    let uses = |filter: &Filter, index: &str| {
        let steps = steps(filter);
        assert!(
            steps.iter().any(|step| step.contains(index)),
            "{filter:?}: {steps:?}"
        );
    };
    uses(&by_label, "USING INDEX logs_label");
    uses(&by_source, "USING INDEX logs_source");
    let plan = steps(&both);
    assert!(
        plan.iter()
            .any(|step| step.contains("USING INDEX logs_label")
                || step.contains("USING INDEX logs_source")),
        "{plan:?}"
    );
    for filter in [
        &Filter::default(),
        &by_label,
        &by_source,
        &both,
        &with_phrase,
        &contains(&["id=3"]),
        &contains(&["=3"]),
    ] {
        let plan = steps(filter);
        assert!(!plan.is_empty());
        assert!(
            !plan.iter().any(|step| step.contains("TEMP B-TREE")),
            "{filter:?} sorts its rows: {plan:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------
// Nearest (5.3, 5.4, 5.6, 6.7)
// ---------------------------------------------------------------------------------------

/// The nearest stored logs of each query vector, with their squared L2 distances,
/// against a brute-force computation in double precision.
#[test]
fn nearest_returns_the_exact_nearest_logs_with_their_distances() {
    let scratch = Scratch::new("nearest");
    let logs = synthetic_logs(500);
    let store = store_of(&scratch, &logs, FILE);
    let candidates = with_vectors(&logs);

    let query_vectors = [
        unit_vector("a log that is not stored"),
        unit_vector("another one"),
        candidates[123].1.clone(),
    ];
    let flat: Vec<f32> = query_vectors.concat();
    let found = store
        .nearest(
            &flat,
            VectorOrigin::without_release(&identity()),
            7,
            &Filter::default(),
        )
        .unwrap();
    assert_eq!(**found.hits.schema(), hits_schema());
    assert_eq!(found.hits.height(), 21);
    assert_eq!((found.queries, found.candidates), (3, 500));
    assert_eq!(
        queries(&found.hits),
        [[0u32; 7], [1; 7], [2; 7]].concat(),
        "hits are grouped by query, in the order of the queries"
    );
    for (index, query) in query_vectors.iter().enumerate() {
        assert_nearest(&found.hits, index as u32, &candidates, query, 7);
    }
    // A stored vector is its own nearest neighbour, at the distance 0 exactly.
    assert_eq!(texts(&found.hits)[14], candidates[123].0);
    assert_eq!(distances(&found.hits)[14], 0.0);

    // The other columns are those of the stored rows (the group for requirement 6.7).
    let all = store.text_search(&Filter::default(), usize::MAX).unwrap();
    let all_ids = ids(&all);
    for (row, id) in ids(&found.hits).iter().enumerate() {
        let stored = all_ids.iter().position(|other| other == id).unwrap();
        for column in [TEXT, SOURCE, LABEL, GROUP] {
            assert_eq!(
                strings(&found.hits, column)[row],
                strings(&all, column)[stored],
                "{column} of hit {row}"
            );
        }
    }
    assert!(strings(&found.hits, GROUP).iter().all(Option::is_some));

    // More asked for than stored: every log, nearest first.
    let every = store
        .nearest(
            &query_vectors[0],
            VectorOrigin::without_release(&identity()),
            10_000,
            &Filter::default(),
        )
        .unwrap();
    assert_eq!(every.hits.height(), 500);
    assert_nearest(&every.hits, 0, &candidates, &query_vectors[0], 500);
    // Distances are far from unit length as well.
    let far: Vec<f32> = query_vectors[1].iter().map(|v| v * 1000.0).collect();
    let found = store
        .nearest(
            &far,
            VectorOrigin::without_release(&identity()),
            3,
            &Filter::default(),
        )
        .unwrap();
    let expected = brute_force(&candidates, &far, 3);
    assert_eq!(
        texts(&found.hits),
        expected
            .iter()
            .map(|(text, _)| text.clone())
            .collect::<Vec<_>>()
    );
    let got = distances(&found.hits);
    for (got, (_, want)) in got.iter().zip(&expected) {
        assert!(
            (f64::from(*got) - want).abs() <= want * 1e-5,
            "{got} {want}"
        );
    }
}

/// Equal distances: the log stored first comes first.
#[test]
fn among_equal_distances_the_log_stored_first_comes_first() {
    let scratch = Scratch::new("ties");
    let logs = plain(&[
        "far away", "tie c", "tie a", "nearer", "tie b", "tie d", "tie e",
    ]);
    let vector = |text: &str| -> Vec<f32> {
        let mut values = vec![0.0f32; usize::from(DIM)];
        match text {
            "far away" => values[0] = -1.0,
            "nearer" => values[0] = 0.9,
            // Five logs with one and the same vector.
            _ => values[1] = 1.0,
        }
        values
    };
    let mut store = Store::create(
        &scratch.paths(),
        "s",
        VectorOrigin::without_release(&identity()),
    )
    .unwrap();
    store
        .add_with(
            &rows_by(&logs, FILE, vector),
            VectorOrigin::without_release(&identity()),
            &mut budget(),
        )
        .unwrap();

    let mut query = vec![0.0f32; usize::from(DIM)];
    query[0] = 1.0;
    let all = store
        .nearest(
            &query,
            VectorOrigin::without_release(&identity()),
            10,
            &Filter::default(),
        )
        .unwrap();
    assert_eq!(
        texts(&all.hits),
        [
            "nearer", "tie c", "tie a", "tie b", "tie d", "tie e", "far away"
        ]
    );
    let d = distances(&all.hits);
    assert!(d[1..6].iter().all(|distance| *distance == d[1]), "{d:?}");
    assert_eq!(d[6], 4.0);
    // The cut falls inside the tie: the first stored of the equals are kept.
    for (k, expected) in [
        (1, &["nearer"][..]),
        (2, &["nearer", "tie c"]),
        (3, &["nearer", "tie c", "tie a"]),
        (4, &["nearer", "tie c", "tie a", "tie b"]),
    ] {
        let found = store
            .nearest(
                &query,
                VectorOrigin::without_release(&identity()),
                k,
                &Filter::default(),
            )
            .unwrap();
        assert_eq!(texts(&found.hits), expected, "k = {k}");
    }
    // The same with one row per batch.
    let found = store
        .nearest_with(
            &query,
            VectorOrigin::without_release(&identity()),
            3,
            &Filter::default(),
            &mut budget_with(None, Some(1)),
        )
        .unwrap();
    assert_eq!(texts(&found.hits), ["nearer", "tie c", "tie a"]);
}

/// A text search and restrictions narrow the logs before the distances are ordered:
/// the nearest are the nearest of the matching logs, not the matching ones among the
/// nearest of all.
#[test]
fn a_text_filter_is_applied_before_the_distance_ordering() {
    let scratch = Scratch::new("combined");
    let (store, logs) = two_sources(&scratch);
    let query = unit_vector("the log searched for");

    let matching = |keep: &dyn Fn(&LogRecord) -> bool| -> Vec<(String, Vec<f32>)> {
        logs.records()
            .filter(|record| keep(record))
            .map(|record| (record.text.clone(), unit_vector(&record.text)))
            .collect()
    };
    let cases: Vec<(Filter, Candidates)> = vec![
        (
            Filter {
                source: Some("first.log".to_string()),
                ..contains(&["id=1"])
            },
            matching(&|record| record.text.contains("id=1")),
        ),
        (
            Filter {
                source: Some("first.log".to_string()),
                label: Some(Label::Bad),
                ..contains(&["ID=1"])
            },
            matching(&|record| record.text.contains("id=1") && record.label == Label::Bad),
        ),
        (
            Filter {
                source: Some("second.log".to_string()),
                case_exact: true,
                ..contains(&["=4", "value"])
            },
            matching(&|record| record.text.contains("=4")),
        ),
        (
            Filter {
                source: Some("first.log".to_string()),
                case_exact: true,
                ..contains(&["ID=1"])
            },
            Vec::new(),
        ),
    ];
    for (filter, candidates) in &cases {
        for k in [1, 3, 100] {
            let found = store
                .nearest(
                    &query,
                    VectorOrigin::without_release(&identity()),
                    k,
                    filter,
                )
                .unwrap();
            assert_eq!(found.candidates, candidates.len() as u64, "{filter:?}");
            assert_nearest(&found.hits, 0, candidates, &query, k);
        }
    }
    // The filter matters: the nearest of all logs is not among the matching ones.
    let (filter, candidates) = &cases[0];
    assert!(candidates.len() > 3 && candidates.len() < 60);
    let unfiltered = store
        .nearest(
            &query,
            VectorOrigin::without_release(&identity()),
            1,
            &Filter::default(),
        )
        .unwrap();
    let filtered = store
        .nearest(
            &query,
            VectorOrigin::without_release(&identity()),
            1,
            filter,
        )
        .unwrap();
    assert_ne!(texts(&unfiltered.hits), texts(&filtered.hits));

    // Without a source the same text of both sources is two logs, at one distance:
    // the one stored first comes first.
    let found = store
        .nearest(
            &query,
            VectorOrigin::without_release(&identity()),
            2,
            &Filter::default(),
        )
        .unwrap();
    assert_eq!(
        strings(&found.hits, SOURCE),
        [
            Some("first.log".to_string()),
            Some("second.log".to_string())
        ]
    );
    assert_eq!(distances(&found.hits)[0], distances(&found.hits)[1]);
}

/// Requirement 5.6: a vector of another encoder identity is not compared with the
/// store's vectors, and both identities are named.
#[test]
fn nearest_refuses_a_vector_of_another_identity_naming_both() {
    let scratch = Scratch::new("identity");
    let store = store_of(&scratch, &synthetic_logs(10), FILE);
    let query = unit_vector("q");
    let others = [
        EncoderIdentity {
            revision: OTHER_REVISION.to_string(),
            ..identity()
        },
        EncoderIdentity {
            name: "other".to_string(),
            ..identity()
        },
        EncoderIdentity {
            dimension: 4,
            ..identity()
        },
    ];
    for given in others {
        let length = usize::from(given.dimension);
        let error = store
            .nearest(
                &query[..length],
                VectorOrigin::without_release(&given),
                3,
                &Filter::default(),
            )
            .expect_err("another identity");
        assert_eq!(
            error,
            StoreError::IdentityMismatch {
                store: identity(),
                given: given.clone()
            }
        );
        let message = error.to_string();
        for part in [
            REVISION,
            given.revision.as_str(),
            "bytecnn",
            given.name.as_str(),
            &format!("dimension {}", given.dimension),
            "dimension 8",
        ] {
            assert!(message.contains(part), "{message} lacks {part}");
        }
        assert!(matches!(L2dError::from(error), L2dError::Refused { .. }));
    }
}

/// Queries that cannot be ones are named; nothing to find is an empty result.
#[test]
fn nearest_names_a_query_that_cannot_be_one() {
    let scratch = Scratch::new("invalid");
    let store = store_of(&scratch, &synthetic_logs(10), FILE);
    let query = unit_vector("q");

    let detail = invalid_query(
        store
            .nearest(
                &query[..7],
                VectorOrigin::without_release(&identity()),
                3,
                &Filter::default(),
            )
            .unwrap_err(),
    );
    assert!(detail.contains('7') && detail.contains('8'), "{detail}");
    let mut spoiled = [query.clone(), query.clone()].concat();
    spoiled[11] = f32::NAN;
    let detail = invalid_query(
        store
            .nearest(
                &spoiled,
                VectorOrigin::without_release(&identity()),
                3,
                &Filter::default(),
            )
            .unwrap_err(),
    );
    assert!(
        detail.contains("query 2") && detail.contains("value 4"),
        "{detail}"
    );
    let detail = invalid_query(
        store
            .nearest(
                &query,
                VectorOrigin::without_release(&identity()),
                3,
                &contains(&[""]),
            )
            .unwrap_err(),
    );
    assert!(detail.contains("empty"), "{detail}");
    assert!(matches!(
        L2dError::from(StoreError::InvalidQuery { detail }),
        L2dError::InvalidInput { .. }
    ));

    for (queries, k) in [(&query[..], 0usize), (&[][..], 3)] {
        let found = store
            .nearest(
                queries,
                VectorOrigin::without_release(&identity()),
                k,
                &Filter::default(),
            )
            .unwrap();
        assert_eq!(found.hits.height(), 0);
        assert_eq!(**found.hits.schema(), hits_schema());
    }
    let nothing = store
        .nearest(
            &query,
            VectorOrigin::without_release(&identity()),
            3,
            &contains(&["no log holds this"]),
        )
        .unwrap();
    assert_eq!((nothing.hits.height(), nothing.candidates), (0, 0));

    let empty = Store::create(
        &scratch.paths(),
        "empty",
        VectorOrigin::without_release(&identity()),
    )
    .unwrap();
    let found = empty
        .nearest(
            &query,
            VectorOrigin::without_release(&identity()),
            3,
            &Filter::default(),
        )
        .unwrap();
    assert_eq!((found.hits.height(), found.candidates), (0, 0));
    assert_eq!(
        empty.text_search(&Filter::default(), 5).unwrap().height(),
        0
    );
    assert_eq!(
        empty
            .vectors(
                VectorOrigin::without_release(&identity()),
                &Filter::default()
            )
            .unwrap()
            .height(),
        0
    );
}

// ---------------------------------------------------------------------------------------
// Export (the vectors for assessment and shift)
// ---------------------------------------------------------------------------------------

/// `vectors` returns every matching row with the vector as it was added, bit for bit.
#[test]
fn vectors_returns_the_matching_rows_with_their_vectors() {
    let scratch = Scratch::new("export");
    let (store, logs) = two_sources(&scratch);
    let added = rows(&logs, "first.log");

    let all = store
        .vectors(
            VectorOrigin::without_release(&identity()),
            &Filter::default(),
        )
        .unwrap();
    assert_eq!(**all.schema(), stored_schema());
    assert_eq!(all.height(), 120);
    assert_eq!(vectors(&all)[..60], vectors(&added));
    assert_eq!(vectors(&all)[60..], vectors(&added));

    let good_of_second = Filter {
        label: Some(Label::Good),
        source: Some("second.log".to_string()),
        ..Filter::default()
    };
    let frame = store
        .vectors(VectorOrigin::without_release(&identity()), &good_of_second)
        .unwrap();
    let expected: Vec<(String, Vec<f32>)> = logs
        .records()
        .filter(|record| record.label == Label::Good)
        .map(|record| (record.text.clone(), unit_vector(&record.text)))
        .collect();
    assert_eq!(frame.height(), 40);
    assert_eq!(
        texts(&frame),
        expected
            .iter()
            .map(|(text, _)| text.clone())
            .collect::<Vec<_>>()
    );
    let got = vectors(&frame);
    for (got, (text, want)) in got.iter().zip(&expected) {
        let bits = |vector: &[f32]| vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(got), bits(want), "{text}");
    }
    assert!(
        strings(&frame, LABEL)
            .iter()
            .all(|l| l.as_deref() == Some("good"))
    );
    assert_eq!(
        strings(&frame, GROUP),
        logs.records()
            .filter(|record| record.label == Label::Good)
            .map(|record| record.group)
            .collect::<Vec<_>>()
    );

    // With a phrase as well, and what was exported can be added to another store as it
    // is (the id column apart).
    let filter = Filter {
        source: Some("first.log".to_string()),
        ..contains(&["id=2"])
    };
    let frame = store
        .vectors(VectorOrigin::without_release(&identity()), &filter)
        .unwrap();
    assert_eq!(texts(&frame), scan(&logs, |text| text.contains("id=2")));
    let mut copy = Store::create(
        &scratch.paths(),
        "copy",
        VectorOrigin::without_release(&identity()),
    )
    .unwrap();
    let report = copy
        .add_with(
            &frame.drop(ID).unwrap(),
            VectorOrigin::without_release(&identity()),
            &mut budget(),
        )
        .unwrap();
    assert_eq!(report.added, frame.height() as u64);
    assert_eq!(
        ids(&copy
            .vectors(
                VectorOrigin::without_release(&identity()),
                &Filter::default()
            )
            .unwrap()),
        ids(&frame)
    );
}

// ---------------------------------------------------------------------------------------
// Memory (requirement 12)
// ---------------------------------------------------------------------------------------

/// Requirement 12.6: the hits are the same whatever the batch size, to the last bit of
/// every distance, and the plan reports the batches used.
#[test]
fn the_results_do_not_depend_on_the_batch_size() {
    let scratch = Scratch::new("batches");
    let logs = synthetic_logs(2_500);
    let store = store_of(&scratch, &logs, FILE);
    let flat: Vec<f32> = ["one", "two", "three", "four"]
        .iter()
        .flat_map(|text| unit_vector(text))
        .collect();
    let filter = contains(&["id=1"]);

    let run = |batch_rows: Option<u64>, filter: &Filter| {
        store
            .nearest_with(
                &flat,
                VectorOrigin::without_release(&identity()),
                9,
                filter,
                &mut budget_with(None, batch_rows),
            )
            .unwrap()
    };
    let reference = run(None, &Filter::default());
    assert_eq!(reference.hits.height(), 36);
    assert_eq!(reference.plan.as_ref().unwrap().max_rows_used, 65_536);
    let filtered = run(None, &filter);
    for batch_rows in [1u64, 7, 1_000, 1_024, 2_499, 2_500, 4_096] {
        let found = run(Some(batch_rows), &Filter::default());
        assert!(
            found.hits.equals_missing(&reference.hits),
            "batches of {batch_rows} rows give other hits"
        );
        let bits = |frame: &DataFrame| -> Vec<u32> {
            distances(frame).iter().map(|d| d.to_bits()).collect()
        };
        assert_eq!(bits(&found.hits), bits(&reference.hits));
        assert_eq!(found.candidates, 2_500);
        let plan = found.plan.as_ref().unwrap();
        assert_eq!(plan.max_rows_used, batch_rows);
        assert!(plan.overridden);
        assert!(
            run(Some(batch_rows), &filter)
                .hits
                .equals_missing(&filtered.hits)
        );
    }

    // Reading rows back does not depend on the budget either.
    let all = store
        .vectors(
            VectorOrigin::without_release(&identity()),
            &Filter::default(),
        )
        .unwrap();
    let small = store
        .vectors_with(
            VectorOrigin::without_release(&identity()),
            &Filter::default(),
            &mut budget_with(Some(64 * MIB), Some(3)),
        )
        .unwrap();
    assert!(small.equals_missing(&all));
}

/// Requirements 12.4, 12.5 and 12.9: a search holds its own share of the limit on top
/// of what the caller holds, is refused before it loads anything when that or the rows
/// to read do not fit, and hands the budget back as it was.
#[test]
fn a_search_that_does_not_fit_the_memory_limit_is_refused_before_loading() {
    let scratch = Scratch::new("memory");
    let logs = synthetic_logs(6_000);
    let store = store_of(&scratch, &logs, FILE);
    let query = unit_vector("q");

    // Under the fixed share nothing is read at all.
    let mut tight = budget_with(Some(32 * MIB + SEARCH_FIXED_BYTES - 1), None);
    let left = tight.left_bytes();
    for reason in [
        refusal(
            store
                .text_search_with(&Filter::default(), 10, &mut tight)
                .unwrap_err(),
        ),
        refusal(
            store
                .vectors_with(
                    VectorOrigin::without_release(&identity()),
                    &Filter::default(),
                    &mut tight,
                )
                .unwrap_err(),
        ),
        refusal(
            store
                .nearest_with(
                    &query,
                    VectorOrigin::without_release(&identity()),
                    3,
                    &Filter::default(),
                    &mut tight,
                )
                .unwrap_err(),
        ),
    ] {
        assert!(
            reason.contains(&SEARCH_FIXED_BYTES.to_string())
                && reason.contains(&left.to_string())
                && reason.contains(&(32 * MIB + SEARCH_FIXED_BYTES - 1).to_string()),
            "{reason}"
        );
    }
    assert_eq!(tight.left_bytes(), left);

    // The export needs every matching row at once: refused with the estimate and the
    // limit when it does not fit, done when it does.
    let room = 2 * MIB;
    let mut small = budget_with(Some(32 * MIB + SEARCH_FIXED_BYTES + room), None);
    let reason = refusal(
        store
            .vectors_with(
                VectorOrigin::without_release(&identity()),
                &Filter::default(),
                &mut small,
            )
            .unwrap_err(),
    );
    assert!(
        reason.contains("6000") && reason.contains("whole") && reason.contains("limit"),
        "{reason}"
    );
    assert_eq!(small.left_bytes(), SEARCH_FIXED_BYTES + room);
    // A narrower search fits the same limit (it holds its share for the phrase as
    // well), and so does a limit on the rows.
    let narrow = contains(&["id=29"]);
    let explained = store.explain(&narrow).unwrap();
    let longest = store.longest_text_bytes().unwrap();
    assert_eq!(
        longest,
        logs.records()
            .map(|record| record.text.len())
            .max()
            .unwrap() as u64
    );
    assert_eq!(explained.index_pieces, ["id=2", "d=29"]);
    assert_eq!(explained.longest_text_bytes, longest);
    // The fixed share, the texts read one at a time with the lower-case copy of the
    // phrase (5 bytes), and the eight characters given to the text index.
    let reading = TEXT_READ_FACTOR * longest + 5;
    assert_eq!(TEXT_READ_FACTOR, 3);
    let phrase_share = reading + 8 * PHRASE_BYTE_BYTES + INDEX_TEXT_FACTOR * 8 * longest;
    assert_eq!(explained.held_bytes, SEARCH_FIXED_BYTES + phrase_share);
    assert_eq!(
        store.explain(&Filter::default()).unwrap().held_bytes,
        SEARCH_FIXED_BYTES
    );
    let few = store
        .vectors_with(
            VectorOrigin::without_release(&identity()),
            &narrow,
            &mut small,
        )
        .unwrap();
    assert_eq!(few.height(), 111);
    let counted = store
        .matching_with(&narrow, usize::MAX, &mut small)
        .unwrap();
    assert_eq!(counted.rows, 111);
    assert!(counted.read_bytes >= few.estimated_size() as u64);
    assert!(counted.read_bytes < room - phrase_share);
    assert_eq!(
        store
            .text_search_with(&Filter::default(), 100, &mut small)
            .unwrap()
            .height(),
        100
    );
    let reason = refusal(
        store
            .text_search_with(&Filter::default(), 5_900, &mut small)
            .unwrap_err(),
    );
    assert!(reason.contains("5900"), "{reason}");
    assert_eq!(small.left_bytes(), SEARCH_FIXED_BYTES + room);

    // What the caller holds counts: the same limit, with a frame held, refuses.
    let mut holding = budget_with(Some(32 * MIB + SEARCH_FIXED_BYTES + room), None);
    holding.hold(room);
    assert!(holding.left_bytes() >= SEARCH_FIXED_BYTES);
    let reason = refusal(
        store
            .vectors_with(
                VectorOrigin::without_release(&identity()),
                &narrow,
                &mut holding,
            )
            .unwrap_err(),
    );
    for figure in [
        explained.held_bytes,
        SEARCH_FIXED_BYTES,
        longest,
        reading,
        8 * PHRASE_BYTE_BYTES + INDEX_TEXT_FACTOR * 8 * longest,
        32 * MIB + SEARCH_FIXED_BYTES + room,
    ] {
        assert!(
            reason.contains(&figure.to_string()),
            "{reason} lacks {figure}"
        );
    }
    // One byte short of the share is refused, the share itself is enough to count.
    let exactly = |bytes: u64| budget_with(Some(32 * MIB + bytes), None);
    refusal(
        store
            .matching_with(&narrow, usize::MAX, &mut exactly(explained.held_bytes - 1))
            .unwrap_err(),
    );
    assert_eq!(
        store
            .matching_with(&narrow, usize::MAX, &mut exactly(explained.held_bytes))
            .unwrap()
            .rows,
        111
    );
    refusal(
        store
            .vectors_with(
                VectorOrigin::without_release(&identity()),
                &Filter::default(),
                &mut holding,
            )
            .unwrap_err(),
    );
    assert_eq!(holding.left_bytes(), SEARCH_FIXED_BYTES);

    // The nearest search reads the vectors in batches and holds its best hits: with
    // many queries and many hits each, those do not fit.
    let many: Vec<f32> = (0..100).flat_map(|_| query.clone()).collect();
    let reason = refusal(
        store
            .nearest_with(
                &many,
                VectorOrigin::without_release(&identity()),
                3_000,
                &Filter::default(),
                &mut small,
            )
            .unwrap_err(),
    );
    assert!(
        reason.contains("100") && reason.contains("3000"),
        "{reason}"
    );
    let found = store
        .nearest_with(
            &many,
            VectorOrigin::without_release(&identity()),
            2,
            &Filter::default(),
            &mut small,
        )
        .unwrap();
    assert_eq!(found.hits.height(), 200);
    assert!(found.held_bytes >= SEARCH_FIXED_BYTES);
    assert_eq!(small.left_bytes(), SEARCH_FIXED_BYTES + room);
    assert_eq!(
        found.plan.unwrap().limit_bytes,
        32 * MIB + SEARCH_FIXED_BYTES + room
    );

    // Asking for no hit, or with no query, reads nothing and asks nothing of the
    // budget: it is answered even when nothing is left.
    holding.hold(holding.usable_bytes());
    assert_eq!(holding.left_bytes(), 0);
    for (queries, k) in [(&query[..], 0usize), (&[][..], 5)] {
        let found = store
            .nearest_with(
                queries,
                VectorOrigin::without_release(&identity()),
                k,
                &Filter::default(),
                &mut holding,
            )
            .expect("nothing is read, so nothing is refused");
        assert_eq!((found.hits.height(), found.candidates), (0, 0));
        assert_eq!((found.held_bytes, found.hits_bytes), (0, 0));
        assert!(found.plan.is_none());
    }
    refusal(
        store
            .nearest_with(
                &query,
                VectorOrigin::without_release(&identity()),
                1,
                &Filter::default(),
                &mut holding,
            )
            .unwrap_err(),
    );
}

/// Searching changes nothing: the file is the same byte for byte afterwards, and a
/// store that was searched takes adds as before.
#[test]
fn searching_leaves_the_store_as_it_was() {
    let scratch = Scratch::new("unchanged");
    let logs = synthetic_logs(200);
    let mut store = store_of(&scratch, &logs, FILE);
    let path = scratch.paths().store_path("s");
    let before = std::fs::read(&path).unwrap();

    let query = unit_vector("q");
    store
        .nearest(
            &query,
            VectorOrigin::without_release(&identity()),
            5,
            &contains(&["id=1"]),
        )
        .unwrap();
    store.text_search(&contains(&["=1"]), 10).unwrap();
    store
        .vectors(
            VectorOrigin::without_release(&identity()),
            &Filter::default(),
        )
        .unwrap();
    store.explain(&contains(&["id=1"])).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(
        std::fs::read_dir(scratch.paths().stores_dir())
            .unwrap()
            .count(),
        1
    );

    let report = store
        .add_with(
            &rows(&logs, "b.log"),
            VectorOrigin::without_release(&identity()),
            &mut budget(),
        )
        .unwrap();
    assert_eq!(report.added, 200);
    assert_eq!(
        store
            .vectors(
                VectorOrigin::without_release(&identity()),
                &Filter::default()
            )
            .unwrap()
            .height(),
        400
    );
    drop(store);
    assert_fts_consistent(&path);
}
