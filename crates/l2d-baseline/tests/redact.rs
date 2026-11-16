//! Redaction: proposed fragments and confirmed rules (requirements 16.4, 16.5, 16.12;
//! task 1.22).
//!
//! The fixture is `tests/fixtures/mini-labelled.csv` (169 logs read by the CSV adapter;
//! row 121 is skipped). Its requests all go to `localhost:8080`; the marked logs below
//! are the SQL injections found by a regular expression, as an analyst would mark them.
//! The expected counts are computed here from the texts, independently of the code
//! under test: the host and the endpoint paths of the fixture occur in a text only
//! where the code cuts them as fragments (each is followed by `?`, a space or the end),
//! so a plain substring test gives the number of logs they occur in.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use l2d_baseline::redact::{
    DEFAULT_PLACEHOLDER, FragmentKind, MIN_LOGS, RedactionInput, RedactionProposals,
    RedactionRules, RedactionSettings, candidate_fragments, propose_redactions,
};
use l2d_baseline::{BaselineScorer, BaselineSettings, CharHasher, HashingSettings, RoutingMode};
use l2d_core::marks::{MarkQuery, MarkSet, MarkStore};
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LinesAdapter,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths, RedactionKind, RedactionRule,
    ZipInputSettings,
};
use sha2::{Digest, Sha256};

const GIB: u64 = 1024 * 1024 * 1024;
const SQL_INJECTION: &str = r"(?i)union(\s|\+)+select";
const DROP_TABLE: &str = r"(?i)drop(\s|\+)+table";
const HOST: &str = "localhost:8080";
const ANADIR: &str = "/tienda1/publico/anadir.jsp";

/// A directory of its own in the system temp folder, removed when dropped.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("l2d-baseline-redact-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn project(&self) -> ProjectPaths {
        ProjectPaths::new(self.dir.join("project"))
    }

    /// An archive called `name` whose entries are `(entry path, lines)`, deflated.
    fn archive(&self, name: &str, entries: &[(&str, &[String])]) -> PathBuf {
        let path = self.dir.join(name);
        let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut writer = zip::ZipWriter::new(file);
        for (entry, lines) in entries {
            writer
                .start_file(*entry, zip::write::SimpleFileOptions::default())
                .unwrap();
            for line in *lines {
                writer.write_all(line.as_bytes()).unwrap();
                writer.write_all(b"\n").unwrap();
            }
        }
        writer.finish().unwrap().flush().unwrap();
        path
    }

    /// A text file of one log per line.
    fn lines(&self, name: &str, lines: &[String]) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv")
}

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB)).unwrap()
}

fn read_fixture() -> (LogsFrame, InputSummary, MemoryBudget) {
    let mut budget = budget();
    let (logs, summary, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .read_with(&fixture_path(), &mut budget)
        .unwrap();
    (logs, summary, budget)
}

/// Every text of the fixture's 200 data rows, in order (repeats included, and the text
/// of row 121, which has a field too few for the CSV adapter).
fn fixture_lines() -> Vec<String> {
    let text = std::fs::read_to_string(fixture_path()).unwrap();
    let mut records = first_fields(&text);
    records.remove(0);
    records
}

/// The first field of every CSV record (a small reader for the fixture's quoting; every
/// record of the fixture ends with a line feed).
fn first_fields(text: &str) -> Vec<String> {
    let mut records = Vec::new();
    let mut field = String::new();
    let mut first = true;
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                if first {
                    field.push('"');
                }
            }
            '"' => quoted = !quoted,
            ',' if !quoted => first = false,
            '\n' if !quoted => {
                records.push(std::mem::take(&mut field));
                first = true;
            }
            '\r' if !quoted => {}
            other => {
                if first {
                    field.push(other);
                }
            }
        }
    }
    records
}

fn texts(logs: &LogsFrame) -> Vec<String> {
    logs.records().map(|record| record.text).collect()
}

fn mark_fixture(pattern: &str) -> (MarkSet, LogsFrame, InputSummary, MemoryBudget) {
    let (logs, summary, budget) = read_fixture();
    let set = MarkSet::mark(
        "attacks",
        Label::Bad,
        MarkQuery::Regex {
            pattern: pattern.into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    (set, logs, summary, budget)
}

/// The texts of the marked and of the unmarked logs of `logs`.
fn split(set: &MarkSet, logs: &LogsFrame) -> (Vec<String>, Vec<String>) {
    let mut marked = Vec::new();
    let mut unmarked = Vec::new();
    for record in logs.records() {
        let position = record_position(logs, &record.text);
        if set.is_selected(position) {
            marked.push(record.text);
        } else {
            unmarked.push(record.text);
        }
    }
    (marked, unmarked)
}

fn record_position(logs: &LogsFrame, text: &str) -> u64 {
    let frame = logs.frame();
    let texts = frame.column(LogsFrame::TEXT).unwrap().str().unwrap();
    let positions = frame
        .column(LogsFrame::FIRST_POSITION)
        .unwrap()
        .u64()
        .unwrap();
    texts
        .into_no_null_iter()
        .zip(positions.into_no_null_iter())
        .find(|(candidate, _)| *candidate == text)
        .map(|(_, position)| position)
        .unwrap()
}

fn count_containing(texts: &[String], fragment: &str) -> u64 {
    texts.iter().filter(|text| text.contains(fragment)).count() as u64
}

fn propose_file(
    set: &MarkSet,
    logs: &LogsFrame,
    summary: &InputSummary,
    budget: &mut MemoryBudget,
    settings: &RedactionSettings,
) -> RedactionProposals {
    propose_redactions(
        set,
        RedactionInput::File { logs, summary },
        settings,
        budget,
    )
    .unwrap()
}

/// The proposals as `fragment -> (marked, unmarked)`, in their order.
fn table(proposals: &RedactionProposals) -> Vec<(String, u64, u64)> {
    proposals
        .proposals
        .iter()
        .map(|p| (p.fragment.clone(), p.marked, p.unmarked))
        .collect()
}

// ---------------------------------------------------------------------------------------
// Candidate fragments
// ---------------------------------------------------------------------------------------

#[test]
fn candidate_fragments_are_tokens_and_path_suffixes() {
    let settings = RedactionSettings::default();
    let text = "GET http://localhost:8080/tienda1/publico/anadir.jsp?id=7&nombre=Cava+Brut\
                &B1=Anadir+al+carrito -- ..";
    let fragments = candidate_fragments(text, &settings);
    // Sorted, without repeats.
    assert!(fragments.windows(2).all(|pair| pair[0] < pair[1]));
    for expected in [
        "GET",
        "http:",
        HOST,
        "tienda1",
        "publico",
        "anadir.jsp",
        ANADIR,
        "/publico/anadir.jsp",
        "/anadir.jsp",
        "id",
        "7",
        "nombre",
        "Cava+Brut",
        "B1",
        "Anadir+al+carrito",
    ] {
        assert!(
            fragments.contains(&expected),
            "{expected} missing from {fragments:?}"
        );
    }
    // No whole URL, no fragment that starts at the `//` of the scheme, nothing without a
    // letter or digit.
    for absent in [
        "http://localhost:8080/tienda1/publico/anadir.jsp",
        "//localhost:8080/tienda1/publico/anadir.jsp",
        "/localhost:8080/tienda1/publico/anadir.jsp",
        "--",
        "..",
        "id=7",
    ] {
        assert!(!fragments.contains(&absent), "{absent} in {fragments:?}");
    }
    assert_eq!(FragmentKind::of(ANADIR), FragmentKind::Path);
    assert_eq!(FragmentKind::of(HOST), FragmentKind::Token);

    // The bounds: longer fragments are left out, and only the first fragments of a log
    // in text order are taken.
    let short = RedactionSettings {
        max_fragment_bytes: 10,
        ..RedactionSettings::default()
    };
    let fragments = candidate_fragments(text, &short);
    assert!(fragments.iter().all(|fragment| fragment.len() <= 10));
    assert!(!fragments.contains(&ANADIR));
    assert!(fragments.contains(&"anadir.jsp"));
    let few = RedactionSettings {
        max_fragments_per_log: 3,
        ..RedactionSettings::default()
    };
    assert_eq!(candidate_fragments(text, &few), ["GET", "http:", HOST]);
    // Repeats count once; a multi-byte character stays inside its token.
    assert_eq!(candidate_fragments("ñu=ñu&ñu", &settings), ["ñu"]);
    assert!(candidate_fragments("", &settings).is_empty());
}

// ---------------------------------------------------------------------------------------
// Proposals
// ---------------------------------------------------------------------------------------

#[test]
fn on_the_fixture_the_url_path_and_host_are_proposed_and_a_payload_token_is_not() {
    for pattern in [SQL_INJECTION, DROP_TABLE] {
        let (set, logs, summary, mut budget) = mark_fixture(pattern);
        let (marked, unmarked) = split(&set, &logs);
        assert_eq!(marked.len() as u64, set.selected_count());
        let held = budget.held_bytes();
        let proposals = propose_file(
            &set,
            &logs,
            &summary,
            &mut budget,
            &RedactionSettings::default(),
        );
        // The budget holds afterwards what it held before.
        assert_eq!(budget.held_bytes(), held);
        assert_eq!(proposals.marked_logs, marked.len() as u64);
        assert_eq!(proposals.unmarked_logs, unmarked.len() as u64);
        assert_eq!(proposals.marked_logs + proposals.unmarked_logs, 169);
        assert!(proposals.min_marked >= MIN_LOGS);
        assert!(proposals.min_unmarked >= MIN_LOGS);

        // The host: in every log.
        let host = proposals.get(HOST).expect("the host is proposed");
        assert_eq!(host.kind, FragmentKind::Token);
        assert_eq!(
            (host.marked, host.unmarked),
            (marked.len() as u64, unmarked.len() as u64)
        );
        // The URL paths of the endpoints the marked logs went to, each with its counts.
        let mut endpoints = 0;
        for path in [
            ANADIR,
            "/tienda1/publico/registro.jsp",
            "/tienda1/publico/buscar.jsp",
            "/tienda1/publico/caracteristicas.jsp",
            "/tienda1/publico/autenticar.jsp",
            "/tienda1/index.jsp",
            "/tienda1/miembros/editar.jsp",
        ] {
            let counts = (
                count_containing(&marked, path),
                count_containing(&unmarked, path),
            );
            let qualifies = counts.0 >= proposals.min_marked && counts.1 >= proposals.min_unmarked;
            match proposals.get(path) {
                Some(proposal) => {
                    assert!(qualifies, "{path} {counts:?} {proposals:?}");
                    assert_eq!(proposal.kind, FragmentKind::Path);
                    assert_eq!((proposal.marked, proposal.unmarked), counts, "{path}");
                    endpoints += 1;
                }
                None => assert!(!qualifies, "{path} {counts:?} is not proposed"),
            }
        }
        // Both mark sets have two or more logs on `anadir.jsp` (and one each on most
        // other endpoints, which does not recur).
        assert!(endpoints >= 1, "{pattern}: {:?}", table(&proposals));
        assert!(proposals.get(ANADIR).is_some(), "{pattern}");
        // Payload tokens of the attack are frequent among the marked logs only.
        for proposal in &proposals.proposals {
            let upper = proposal.fragment.to_uppercase();
            for payload in ["UNION", "SELECT", "DROP", "TABLE"] {
                assert!(
                    !upper.contains(payload),
                    "{pattern}: payload token proposed: {proposal:?}"
                );
            }
            assert!(proposal.marked >= proposals.min_marked);
            assert!(proposal.unmarked >= proposals.min_unmarked);
        }
        // In order: most marked logs first, then most unmarked, then the longer fragment.
        assert!(proposals.proposals.windows(2).all(|pair| {
            (pair[1].marked, pair[1].unmarked, pair[1].fragment.len())
                <= (pair[0].marked, pair[0].unmarked, pair[0].fragment.len())
        }));
    }
}

#[test]
fn proposals_name_the_exact_counts_of_every_fragment_that_qualifies() {
    // Every candidate fragment counted by brute force over the logs: the proposals are
    // exactly those that reach both thresholds, with these counts.
    let (set, logs, summary, mut budget) = mark_fixture(SQL_INJECTION);
    let settings = RedactionSettings::default();
    let proposals = propose_file(&set, &logs, &summary, &mut budget, &settings);
    let (marked, unmarked) = split(&set, &logs);
    let mut counts: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for text in &marked {
        for fragment in candidate_fragments(text, &settings) {
            counts.entry(fragment.to_string()).or_default().0 += 1;
        }
    }
    for text in &unmarked {
        for fragment in candidate_fragments(text, &settings) {
            counts.entry(fragment.to_string()).or_default().1 += 1;
        }
    }
    let min_marked = MIN_LOGS.max((settings.min_marked_share * marked.len() as f64).ceil() as u64);
    let min_unmarked =
        MIN_LOGS.max((settings.min_unmarked_share * unmarked.len() as f64).ceil() as u64);
    assert_eq!(proposals.min_marked, min_marked);
    assert_eq!(proposals.min_unmarked, min_unmarked);
    let mut expected: Vec<(String, u64, u64)> = counts
        .into_iter()
        .filter(|(_, (m, u))| *m >= min_marked && *u >= min_unmarked)
        .map(|(fragment, (m, u))| (fragment, m, u))
        .collect();
    expected.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.2.cmp(&a.2))
            .then(b.0.len().cmp(&a.0.len()))
            .then(a.0.cmp(&b.0))
    });
    assert_eq!(table(&proposals), expected);
    assert!(!expected.is_empty());
}

#[test]
fn proposals_are_the_same_whatever_the_batch_size_and_the_entries_read_at_once() {
    let scratch = Scratch::new("batches");
    let lines = fixture_lines();
    assert_eq!(lines.len(), 200);
    let first = vec!["GET http://other.example/x?y=1".to_string()];
    let path = scratch.archive(
        "fixture.zip",
        &[("logs/first.log", &first), ("logs/fixture.log", &lines)],
    );
    let mut reference: Option<Vec<(String, u64, u64)>> = None;
    for max_batch_rows in [None, Some(7)] {
        for entries_at_once in [Some(1), Some(2)] {
            let settings = ZipInputSettings {
                entries: vec!["logs/*.log".into()],
                entries_at_once,
                ..ZipInputSettings::default()
            };
            let memory = MemorySettings {
                max_batch_rows,
                ..MemorySettings::default()
            };
            let mut budget = MemoryBudget::observe_with(&memory, || Ok(8 * GIB)).unwrap();
            let (set, _) = MarkSet::mark_archive(
                "attacks",
                Label::Bad,
                MarkQuery::Regex {
                    pattern: SQL_INJECTION.into(),
                },
                &path,
                &settings,
                &mut budget,
            )
            .unwrap();
            let held = budget.held_bytes();
            let proposals = propose_redactions(
                &set,
                RedactionInput::Archive {
                    path: &path,
                    settings: &settings,
                },
                &RedactionSettings::default(),
                &mut budget,
            )
            .unwrap();
            assert_eq!(budget.held_bytes(), held);
            assert!(proposals.get(HOST).is_some());
            assert!(proposals.get(ANADIR).is_some());
            assert!(proposals.archive.is_some());
            let got = table(&proposals);
            match &reference {
                None => reference = Some(got),
                Some(reference) => assert_eq!(
                    &got, reference,
                    "batch rows {max_batch_rows:?}, {entries_at_once:?} at once"
                ),
            }
        }
    }
    // The archive's logs are its distinct lines: the marked ones are the file's, the
    // host is in every distinct fixture line (row 121 among them, which the CSV adapter
    // skips) and not in the first entry's line.
    let (set, logs, summary, mut budget) = mark_fixture(SQL_INJECTION);
    let file = propose_file(
        &set,
        &logs,
        &summary,
        &mut budget,
        &RedactionSettings::default(),
    );
    let host = file.get(HOST).unwrap();
    let distinct: std::collections::BTreeSet<&String> = lines.iter().collect();
    let archive_host = reference
        .unwrap()
        .into_iter()
        .find(|(fragment, _, _)| fragment == HOST)
        .unwrap();
    assert_eq!(archive_host.1, host.marked);
    assert_eq!(archive_host.1 + archive_host.2, distinct.len() as u64);
}

/// A synthetic input whose rare fragments overflow small counters: 1,000 logs, every
/// other one marked, five endpoints, a user name and a number that differ in each log.
fn synthetic_lines() -> Vec<String> {
    (0..1000u64)
        .map(|i| {
            let attack = if i % 2 == 0 { "%27+OR+1%3D1" } else { "ok" };
            format!(
                "GET http://shop.example:9090/app/page{}.jsp?user=u{i}&t={}&q={attack}",
                i % 5,
                (i * 7919) % 100_003
            )
        })
        .collect()
}

fn mark_lines(path: &Path, budget: &mut MemoryBudget) -> (MarkSet, LogsFrame, InputSummary) {
    let (logs, summary, _) = LinesAdapter.read_with(path, budget).unwrap();
    let set = MarkSet::mark(
        "or-one",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "%27+OR+1%3D1".into(),
        },
        &logs,
        &summary,
        budget,
    )
    .unwrap();
    assert_eq!(set.count(), 500);
    (set, logs, summary)
}

#[test]
fn bounded_counters_give_the_exact_proposals_or_are_refused_with_the_figures() {
    let scratch = Scratch::new("counters");
    let path = scratch.lines("synthetic.log", &synthetic_lines());
    let mut budget = budget();
    let (set, logs, summary) = mark_lines(&path, &mut budget);

    let roomy = propose_file(
        &set,
        &logs,
        &summary,
        &mut budget,
        &RedactionSettings::default(),
    );
    assert_eq!(
        roomy.dropped, 0,
        "the default counters do not overflow here"
    );
    assert_eq!(roomy.min_marked, 25);
    let expected = table(&roomy);
    assert!(roomy.get("shop.example:9090").is_some());
    for page in 0..5 {
        let path = format!("/app/page{page}.jsp");
        let proposal = roomy.get(&path).expect("every endpoint path");
        assert_eq!((proposal.marked, proposal.unmarked), (100, 100), "{path}");
    }
    // Neither the attack nor the user names nor the numbers.
    assert!(roomy.get("%27+OR+1%3D1").is_none());
    assert!(roomy.get("u0").is_none());

    // Small counters drop counts of the rare fragments, but less than a proposal needs:
    // the proposals are the same.
    let small = RedactionSettings {
        counter_slots: 256,
        ..RedactionSettings::default()
    };
    let bounded = propose_file(&set, &logs, &summary, &mut budget, &small);
    assert!(bounded.dropped > 0, "the counters overflowed");
    assert!(bounded.dropped < bounded.min_marked);
    assert_eq!(table(&bounded), expected);
    assert!(bounded.counters_bytes < roomy.counters_bytes);

    // Counters too small to be sure of every fragment that qualifies: refused.
    let tiny = RedactionSettings {
        counter_slots: 4,
        ..RedactionSettings::default()
    };
    let error = propose_redactions(
        &set,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &tiny,
        &mut budget,
    )
    .unwrap_err();
    let L2dError::Refused { reason } = &error else {
        panic!("expected Refused, got {error:?}");
    };
    assert!(reason.contains("4 slots"), "{reason}");
    assert!(reason.contains("25 marked logs"), "{reason}");
}

#[test]
fn counters_that_do_not_fit_the_memory_limit_are_refused_before_counting() {
    let (set, logs, summary, _) = mark_fixture(SQL_INJECTION);
    let settings = MemorySettings {
        memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 1024 * 1024),
        ..MemorySettings::default()
    };
    let mut tight = MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).unwrap();
    let error = propose_redactions(
        &set,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &RedactionSettings::default(),
        &mut tight,
    )
    .unwrap_err();
    let L2dError::Refused { reason } = &error else {
        panic!("expected Refused, got {error:?}");
    };
    assert!(reason.contains("counters"), "{reason}");
    assert!(reason.contains("1.0 MiB (1048576 bytes)"), "{reason}");
    assert!(reason.contains("nothing was counted"), "{reason}");
    assert_eq!(tight.held_bytes(), 0);
    // Smaller counters fit.
    let small = RedactionSettings {
        counter_slots: 512,
        max_fragment_bytes: 64,
        ..RedactionSettings::default()
    };
    let proposals = propose_redactions(
        &set,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &small,
        &mut tight,
    )
    .unwrap();
    assert!(proposals.get(HOST).is_some());
    assert!(proposals.counters_bytes <= 1024 * 1024);
}

#[test]
fn settings_out_of_range_are_invalid_input() {
    let (set, logs, summary, mut budget) = mark_fixture(SQL_INJECTION);
    for settings in [
        RedactionSettings {
            min_marked_share: 0.0,
            ..RedactionSettings::default()
        },
        RedactionSettings {
            min_unmarked_share: 1.5,
            ..RedactionSettings::default()
        },
        RedactionSettings {
            min_marked_share: f64::NAN,
            ..RedactionSettings::default()
        },
        RedactionSettings {
            counter_slots: 0,
            ..RedactionSettings::default()
        },
        RedactionSettings {
            max_fragment_bytes: 0,
            ..RedactionSettings::default()
        },
        RedactionSettings {
            max_fragments_per_log: 0,
            ..RedactionSettings::default()
        },
    ] {
        let error = propose_redactions(
            &set,
            RedactionInput::File {
                logs: &logs,
                summary: &summary,
            },
            &settings,
            &mut budget,
        )
        .unwrap_err();
        assert!(matches!(error, L2dError::InvalidInput { .. }), "{error:?}");
    }
}

#[test]
fn an_input_that_is_not_the_mark_sets_is_refused() {
    let scratch = Scratch::new("changed");
    let (set, _, _, _) = mark_fixture(SQL_INJECTION);
    let mut text = std::fs::read_to_string(fixture_path()).unwrap();
    text = text.replacen("pagina=1,", "pagina=2,", 1);
    let changed = scratch.dir.join("changed.csv");
    std::fs::write(&changed, text).unwrap();
    let mut budget = budget();
    let (logs, summary, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .read_with(&changed, &mut budget)
        .unwrap();
    let error = propose_redactions(
        &set,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &RedactionSettings::default(),
        &mut budget,
    )
    .unwrap_err();
    let L2dError::Refused { reason } = &error else {
        panic!("expected Refused, got {error:?}");
    };
    assert!(reason.contains(&summary.content_sha256), "{reason}");
    assert!(
        reason.contains(&set.input().unwrap().content_sha256),
        "{reason}"
    );

    // A frame of the right input that lacks marked logs (the hash agrees, the logs do
    // not): refused, naming the marked logs found and expected.
    let (logs, summary, _) = read_fixture();
    let head = LogsFrame::new(logs.frame().head(Some(50))).unwrap();
    let error = propose_redactions(
        &set,
        RedactionInput::File {
            logs: &head,
            summary: &summary,
        },
        &RedactionSettings::default(),
        &mut budget,
    )
    .unwrap_err();
    assert!(
        matches!(&error, L2dError::Refused { reason }
            if reason.contains("marked logs") && reason.contains(&set.selected_count().to_string())),
        "{error:?}"
    );

    // A mark set of a file is refused for an archive, and one of an input for a store.
    let archive = scratch.archive("fixture.zip", &[("fixture.log", &fixture_lines())]);
    let error = propose_redactions(
        &set,
        RedactionInput::Archive {
            path: &archive,
            settings: &ZipInputSettings::default(),
        },
        &RedactionSettings::default(),
        &mut budget,
    )
    .unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");
    let store = FixtureStore::new(&texts(&read_fixture().0));
    let error = propose_redactions(
        &set,
        RedactionInput::Store(&store),
        &RedactionSettings::default(),
        &mut budget,
    )
    .unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");
}

// ---------------------------------------------------------------------------------------
// A store
// ---------------------------------------------------------------------------------------

/// The fixture's logs as a store would hand them out: ids are the SHA-256 of the text.
struct FixtureStore {
    rows: Vec<(String, String)>,
    hash: String,
}

impl FixtureStore {
    fn new(texts: &[String]) -> Self {
        let rows: Vec<(String, String)> = texts
            .iter()
            .map(|text| {
                (
                    format!("{:x}", Sha256::digest(text.as_bytes())),
                    text.clone(),
                )
            })
            .collect();
        let mut ids: Vec<&str> = rows.iter().map(|(id, _)| id.as_str()).collect();
        ids.sort_unstable();
        let hash = format!("{:x}", Sha256::digest(ids.concat().as_bytes()));
        Self { rows, hash }
    }
}

impl MarkStore for FixtureStore {
    fn store_name(&self) -> &str {
        "fixture"
    }

    fn content_sha256(&self, _budget: &mut MemoryBudget) -> Result<String, L2dError> {
        Ok(self.hash.clone())
    }

    fn phrase_ids(
        &self,
        phrase: &str,
        _id_bytes: u64,
        _budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&str),
    ) -> Result<(), L2dError> {
        for (id, text) in &self.rows {
            if text.contains(phrase) {
                each(id);
            }
        }
        Ok(())
    }

    fn text_batches(
        &self,
        _beside: u64,
        _per_row: u64,
        _budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        for (number, chunk) in self.rows.chunks(10).enumerate() {
            let seq: Vec<i64> = (0..chunk.len())
                .map(|row| (number * 10 + row) as i64)
                .collect();
            let ids: Vec<&str> = chunk.iter().map(|(id, _)| id.as_str()).collect();
            let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
            let frame = DataFrame::new(vec![
                Column::new("seq".into(), seq),
                Column::new("id".into(), ids),
                Column::new("text".into(), texts),
            ])
            .unwrap();
            each(&frame)?;
        }
        Ok(())
    }
}

#[test]
fn a_store_mark_set_gives_the_proposals_of_the_same_logs() {
    let (set, logs, summary, mut budget) = mark_fixture(SQL_INJECTION);
    let from_file = propose_file(
        &set,
        &logs,
        &summary,
        &mut budget,
        &RedactionSettings::default(),
    );
    let store = FixtureStore::new(&texts(&logs));
    let stored = MarkSet::mark_store(
        "attacks",
        Label::Bad,
        MarkQuery::Regex {
            pattern: SQL_INJECTION.into(),
        },
        &store,
        &mut budget,
    )
    .unwrap();
    assert_eq!(stored.count(), set.count());
    let held = budget.held_bytes();
    let from_store = propose_redactions(
        &stored,
        RedactionInput::Store(&store),
        &RedactionSettings::default(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(budget.held_bytes(), held);
    assert_eq!(table(&from_store), table(&from_file));
    assert!(from_store.archive.is_none());
    // A mark set of a store is refused for an input.
    let error = propose_redactions(
        &stored,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &RedactionSettings::default(),
        &mut budget,
    )
    .unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");
}

// ---------------------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------------------

fn literal(fragment: &str) -> RedactionRule {
    RedactionRule {
        kind: RedactionKind::Literal,
        fragment: fragment.into(),
        placeholder: DEFAULT_PLACEHOLDER.into(),
    }
}

fn rules(list: Vec<RedactionRule>) -> RedactionRules {
    RedactionRules::new(list, &budget()).unwrap()
}

#[test]
fn confirmed_rules_replace_their_fragments_by_the_placeholder() {
    let (set, logs, summary, mut budget) = mark_fixture(SQL_INJECTION);
    let proposals = propose_file(
        &set,
        &logs,
        &summary,
        &mut budget,
        &RedactionSettings::default(),
    );
    // The analyst confirms the host and one endpoint path (16.5).
    let confirmed = RedactionRules::confirmed(
        [proposals.get(HOST).unwrap(), proposals.get(ANADIR).unwrap()],
        &budget,
    )
    .unwrap();
    // Longer fragments first, so a fragment inside another is replaced after it.
    assert_eq!(confirmed.rules(), [literal(ANADIR), literal(HOST)]);
    let p = DEFAULT_PLACEHOLDER;
    assert_eq!(
        confirmed.apply("GET http://localhost:8080/tienda1/publico/anadir.jsp?id=7&x=y"),
        format!("GET http://{p}{p}?id=7&x=y")
    );
    // A text without the fragments is handed back as it is.
    let untouched = "GET http://other.example/tienda1/publico/anadir.jspx?id=7";
    assert!(matches!(
        confirmed.apply(untouched),
        std::borrow::Cow::Borrowed(_)
    ));

    // A literal is replaced only where it does not continue a token on either side.
    let tienda = rules(vec![literal("tienda1")]);
    assert_eq!(
        tienda.apply("/tienda1/a /tienda10/b xtienda1 tienda1.jsp tienda1"),
        format!("/{p}/a /tienda10/b xtienda1 tienda1.jsp {p}")
    );
    // An edited rule as a pattern, and one added by the analyst with its own
    // placeholder; an empty placeholder deletes.
    let edited = rules(vec![
        RedactionRule {
            kind: RedactionKind::Pattern,
            fragment: r"/tienda1/[a-z]+/[a-z]+\.jsp".into(),
            placeholder: "<path>".into(),
        },
        RedactionRule {
            kind: RedactionKind::Literal,
            fragment: "http://".into(),
            placeholder: String::new(),
        },
    ]);
    assert_eq!(
        edited.apply("GET http://localhost:8080/tienda1/publico/buscar.jsp?q=$1"),
        "GET localhost:8080<path>?q=$1"
    );
    // Rules apply in their order: the second sees what the first left.
    let ordered = rules(vec![literal(HOST), literal("localhost")]);
    assert_eq!(
        ordered.apply("localhost:8080 localhost"),
        format!("{p} {p}")
    );
    // A pattern that does not compile, an empty fragment or too many rules: refused as
    // the analyst's input.
    for bad in [
        vec![RedactionRule {
            kind: RedactionKind::Pattern,
            fragment: "(".into(),
            placeholder: String::new(),
        }],
        vec![literal("")],
        vec![literal("x"); l2d_core::marks::MAX_REDACTIONS + 1],
    ] {
        let error = RedactionRules::new(bad, &budget).unwrap_err();
        assert!(matches!(error, L2dError::InvalidInput { .. }), "{error:?}");
    }
}

#[test]
fn rules_are_saved_with_the_mark_set_and_loaded_back() {
    let scratch = Scratch::new("saved");
    let paths = scratch.project();
    let (mut set, _, _, budget) = mark_fixture(SQL_INJECTION);
    // Without rules the file is the format of task 1.21.
    let file = set.save(&paths, &budget).unwrap();
    let plain: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert_eq!(plain["format"], 1);
    assert!(plain.get("redactions").is_none());

    let confirmed = rules(vec![
        literal(ANADIR),
        literal(HOST),
        RedactionRule {
            kind: RedactionKind::Pattern,
            fragment: r"pagina=\d+".into(),
            placeholder: "pagina=N".into(),
        },
    ]);
    confirmed.save_with(&mut set, &paths, &budget).unwrap();
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert_eq!(saved["format"], 2);
    assert_eq!(saved["redactions"].as_array().unwrap().len(), 3);
    assert_eq!(saved["redactions"][0]["kind"], "literal");
    assert_eq!(saved["redactions"][0]["fragment"], ANADIR);
    assert_eq!(saved["redactions"][2]["kind"], "pattern");

    let loaded = MarkSet::load(&paths, "attacks", &budget).unwrap();
    assert_eq!(loaded, set);
    let again = RedactionRules::of_mark_set(&loaded, &budget).unwrap();
    assert_eq!(again.rules(), confirmed.rules());
    assert_eq!(again.content_sha256(), confirmed.content_sha256());

    // Other rules for the same marks replace them; the marks stay.
    let fewer = rules(vec![literal(HOST)]);
    fewer.save_with(&mut set, &paths, &budget).unwrap();
    let loaded = MarkSet::load(&paths, "attacks", &budget).unwrap();
    assert_eq!(loaded.redactions(), fewer.rules());
    assert_eq!(loaded.positions(), set.positions());

    // A file that says format 2 without rules, or format 1 with rules, is refused.
    let text = std::fs::read_to_string(&file).unwrap();
    std::fs::write(&file, text.replacen("\"format\": 2", "\"format\": 1", 1)).unwrap();
    let error = MarkSet::load(&paths, "attacks", &budget).unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");
}

#[test]
fn the_rules_hash_names_their_kind_fragment_placeholder_and_order() {
    let a = rules(vec![literal(ANADIR), literal(HOST)]);
    assert_eq!(a.content_sha256().len(), 64);
    assert_eq!(
        a.content_sha256(),
        rules(vec![literal(ANADIR), literal(HOST)]).content_sha256()
    );
    let mut differ = vec![
        rules(vec![literal(HOST), literal(ANADIR)]).content_sha256(),
        rules(vec![literal(ANADIR)]).content_sha256(),
        RedactionRules::none().content_sha256(),
        rules(vec![
            literal(ANADIR),
            RedactionRule {
                placeholder: "x".into(),
                ..literal(HOST)
            },
        ])
        .content_sha256(),
        rules(vec![
            literal(ANADIR),
            RedactionRule {
                kind: RedactionKind::Pattern,
                ..literal(HOST)
            },
        ])
        .content_sha256(),
    ];
    differ.push(a.content_sha256());
    let count = differ.len();
    differ.sort();
    differ.dedup();
    assert_eq!(differ.len(), count);
}

#[test]
fn rules_apply_the_same_at_fitting_and_at_scoring() {
    let scratch = Scratch::new("fit-score");
    let paths = scratch.project();
    let (mut set, logs, summary, mut budget) = mark_fixture(SQL_INJECTION);
    let proposals = propose_file(
        &set,
        &logs,
        &summary,
        &mut budget,
        &RedactionSettings::default(),
    );
    // Every proposal confirmed: what a fit sees.
    let at_fit = RedactionRules::confirmed(proposals.proposals.iter(), &budget).unwrap();
    assert!(at_fit.rules().len() >= 3);
    at_fit.save_with(&mut set, &paths, &budget).unwrap();
    // What scoring sees: the rules as saved with the mark set.
    let at_score =
        RedactionRules::of_mark_set(&MarkSet::load(&paths, "attacks", &budget).unwrap(), &budget)
            .unwrap();
    assert_eq!(at_score.content_sha256(), at_fit.content_sha256());

    let settings = BaselineSettings {
        buckets: 4096,
        ..BaselineSettings::default()
    };
    let hasher = CharHasher::new(settings.hashing()).unwrap();
    let redacted = at_fit.apply_frame(&logs).unwrap();
    assert_eq!(redacted.len(), logs.len());
    let scorer = BaselineScorer::fit_with(
        &redacted,
        std::slice::from_ref(&summary),
        &settings,
        &budget,
    )
    .unwrap();
    let scores = scorer.score(&redacted).unwrap();
    let fitted_ratios = scores.column("ratio").unwrap().f32().unwrap();
    let mut changed = 0;
    for ((record, fitted), redacted_text) in logs
        .records()
        .zip(fitted_ratios.into_no_null_iter())
        .zip(texts(&redacted))
    {
        let text = &record.text;
        let scored = at_score.apply(text);
        assert_eq!(scored, redacted_text);
        assert_eq!(
            hasher.row_redacted(text, &at_score),
            hasher.row_redacted(text, &at_fit)
        );
        assert_eq!(
            hasher.row_redacted(text, &at_score),
            hasher.row(&redacted_text)
        );
        assert_eq!(scorer.ratio(&scored).to_bits(), fitted.to_bits());
        // The fit took the labelled logs, by their redacted texts.
        if record.label != Label::Unknown {
            assert_eq!(scorer.was_fitted_on_text(&scored), Some(true));
        }
        if scored != *text {
            changed += 1;
        }
    }
    // Every fixture log names the host, so every one was redacted.
    assert_eq!(changed, logs.len());
}

#[test]
fn confirmed_rules_change_the_features_as_stated() {
    let hasher = CharHasher::default();
    let confirmed = rules(vec![
        literal("/tienda1/publico/anadir.jsp"),
        literal("/tienda1/publico/buscar.jsp"),
        literal(HOST),
    ]);
    let anadir = "GET http://localhost:8080/tienda1/publico/anadir.jsp?id=1+UNION+SELECT+1";
    let buscar = "GET http://localhost:8080/tienda1/publico/buscar.jsp?id=1+UNION+SELECT+1";
    // Unredacted, the endpoint tells the two apart; redacted, the features are the
    // same: what is left is the method, the placeholders and the payload.
    assert_ne!(hasher.row(anadir), hasher.row(buscar));
    assert_eq!(
        hasher.row_redacted(anadir, &confirmed),
        hasher.row_redacted(buscar, &confirmed)
    );
    let p = DEFAULT_PLACEHOLDER;
    let stated = format!("GET http://{p}{p}?id=1+UNION+SELECT+1");
    assert_eq!(hasher.row_redacted(anadir, &confirmed), hasher.row(&stated));
    assert_ne!(hasher.row_redacted(anadir, &confirmed), hasher.row(anadir));
    // No rules: the plain row.
    assert_eq!(
        hasher.row_redacted(anadir, &RedactionRules::none()),
        hasher.row(anadir)
    );
}

#[test]
fn the_routing_option_gives_the_same_features_as_its_rule() {
    let routing = RedactionRules::routing();
    assert_eq!(routing.rules().len(), 1);
    assert_eq!(routing.rules()[0].kind, RedactionKind::Pattern);
    assert_eq!(routing.rules()[0].placeholder, "");
    let mut samples = texts(&read_fixture().0);
    samples.extend(
        [
            "",
            "?",
            "\n",
            "GET",
            "GET http://localhost:8080/a.jsp",
            "GET http://localhost:8080/a.jsp?",
            "?x=1",
            "POST http://localhost:8080/a.jsp\nid=1&b=2",
            "POST http://localhost:8080/a.jsp\nid=1?b=2\nc",
            "GET /ñandú?q=ü",
            "GET /a?b\n?c",
        ]
        .map(String::from),
    );
    for buckets in [64, 65_536] {
        for ngram_max in [1, 3, 5] {
            let keep = CharHasher::new(HashingSettings {
                buckets,
                ngram_max,
                routing: RoutingMode::Keep,
            })
            .unwrap();
            let strip = CharHasher::new(HashingSettings {
                buckets,
                ngram_max,
                routing: RoutingMode::Strip,
            })
            .unwrap();
            for text in &samples {
                assert_eq!(
                    keep.row_redacted(text, &routing),
                    strip.row(text),
                    "{text:?} at {buckets} buckets, n-grams to {ngram_max}"
                );
            }
        }
    }
}
