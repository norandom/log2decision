//! Contract tests of the store (task 1.8; requirements 5.1, 5.6, 5.7, 6.7 and 12).
//!
//! Everything here runs on the mini fixture or on synthetic logs with synthetic vectors
//! (no encoder exists yet), and writes only below the system temporary directory.
//!
//! The search operations are task 1.9. Until then the tests read the database file
//! themselves, through their own SQLite connection, which also pins the schema as built.
//! The measurements on HTTP CSIC 2010 are in `store_csic.rs` and run on request.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use l2d_core::memory::BatchCause;
use l2d_core::polars::prelude::{
    Column, DataFrame, DataType, Float32Type, IntoColumn, ListBuilderTrait,
    ListPrimitiveChunkedBuilder, NamedFrom, Series,
};
use l2d_core::{
    EncoderIdentity, InputAdapter, L2dError, Label, LabelledAdapter, LabelledFormat, LogRecord,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_store::{
    ADD_FIXED_BYTES, ADD_TEXT_FACTOR, AddReport, IDLE_BYTES, MAX_NAME_BYTES, MAX_TEXT_BYTES,
    RELEASE_TEXT_BYTES, SCHEMA_VERSION, Store, StoreError, VECTOR, log_id, rows_frame, rows_schema,
};
use rusqlite::{Connection, OpenFlags};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const REVISION: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const OTHER_REVISION: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
const DIM: u16 = 8;
const SOURCE: &str = "tests/fixtures/mini-labelled.csv";

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
            "l2d-store-{}-{}-{name}",
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

    /// The file names in `stores/`, sorted.
    fn store_files(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.paths().stores_dir())
            .map(|entries| {
                entries
                    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
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

/// The 169 distinct logs of the mini fixture.
fn fixture_logs() -> LogsFrame {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv");
    let (logs, summary) = LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&path)
        .expect("the fixture is readable");
    assert_eq!(summary.distinct_texts, 169);
    assert_eq!(logs.len(), 169);
    logs
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

/// FNV-1a over the bytes of `text`: a fixed pseudo-random number per text.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A fixed unit vector for `text`; another `salt` gives another vector.
fn unit_vector(text: &str, salt: u64, dimension: u16) -> Vec<f32> {
    let mut state = fnv(text) ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut values: Vec<f32> = (0..dimension)
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

fn vectors_of(logs: &LogsFrame, salt: u64, dimension: u16) -> Vec<f32> {
    logs.records()
        .flat_map(|record| unit_vector(&record.text, salt, dimension))
        .collect()
}

/// A log longer than this many bytes counts as truncated in these tests.
const INPUT_BYTES: usize = 120;

fn truncated_of(logs: &LogsFrame) -> Vec<bool> {
    logs.records()
        .map(|record| record.text.len() > INPUT_BYTES)
        .collect()
}

/// The rows frame for `logs` from `source` with the vectors of `salt`.
fn rows(logs: &LogsFrame, source: &str, salt: u64) -> DataFrame {
    rows_frame(
        logs,
        source,
        &vectors_of(logs, salt, DIM),
        DIM,
        &truncated_of(logs),
    )
    .expect("the logs make a rows frame")
}

fn le_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// One row of the `logs` table, as the database file holds it.
#[derive(Debug, Clone, PartialEq)]
struct Raw {
    seq: i64,
    id: String,
    text: String,
    source: String,
    occurrences: i64,
    label: String,
    group_key: Option<String>,
    truncated: i64,
    vector: Vec<u8>,
    encoder_revision: String,
}

fn read_only(path: &Path) -> Connection {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("the store file opens read-only")
}

/// Every row of the `logs` table in insertion order.
fn dump(path: &Path) -> Vec<Raw> {
    let connection = read_only(path);
    let mut statement = connection
        .prepare(
            "SELECT seq, id, text, source, occurrences, label, group_key, truncated, vector, \
             encoder_revision FROM logs ORDER BY seq",
        )
        .expect("the logs table has the documented columns");
    statement
        .query_map([], |row| {
            Ok(Raw {
                seq: row.get(0)?,
                id: row.get(1)?,
                text: row.get(2)?,
                source: row.get(3)?,
                occurrences: row.get(4)?,
                label: row.get(5)?,
                group_key: row.get(6)?,
                truncated: row.get(7)?,
                vector: row.get(8)?,
                encoder_revision: row.get(9)?,
            })
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .expect("the rows are readable")
}

/// What `dump` must show after `logs` from `source` with the vectors of `salt` were
/// added to an empty store.
fn expected(logs: &LogsFrame, source: &str, salt: u64) -> Vec<Raw> {
    logs.records()
        .enumerate()
        .map(|(index, record)| Raw {
            seq: index as i64 + 1,
            id: log_id(source, &record.text),
            vector: le_bytes(&unit_vector(&record.text, salt, DIM)),
            truncated: i64::from(record.text.len() > INPUT_BYTES),
            text: record.text,
            source: source.to_string(),
            occurrences: i64::from(record.occurrences),
            label: record.label.as_str().to_string(),
            group_key: record.group,
            encoder_revision: REVISION.to_string(),
        })
        .collect()
}

/// The ids of the rows whose text contains `phrase`, found through the FTS5 index, in
/// insertion order.
fn fts_ids(path: &Path, phrase: &str) -> Vec<String> {
    let connection = read_only(path);
    let mut statement = connection
        .prepare(
            "SELECT logs.id FROM logs_fts JOIN logs ON logs.seq = logs_fts.rowid \
             WHERE logs_fts MATCH ?1 ORDER BY logs.seq",
        )
        .expect("the FTS5 table can be queried");
    let quoted = format!("\"{}\"", phrase.replace('"', "\"\""));
    statement
        .query_map([quoted], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .expect("the matches are readable")
}

/// The ids of the logs whose text contains `phrase`, by a plain scan of the frame.
///
/// The scan ignores ASCII case only. The index folds Unicode case (`ÄÖ` matches `äö`),
/// so the two agree only for phrases whose letters outside ASCII have one case in the
/// texts; the tests keep to such phrases.
fn scan_ids(logs: &LogsFrame, source: &str, phrase: &str) -> Vec<String> {
    let phrase = phrase.to_ascii_lowercase();
    logs.records()
        .filter(|record| record.text.to_ascii_lowercase().contains(&phrase))
        .map(|record| log_id(source, &record.text))
        .collect()
}

/// Asks FTS5 to compare its index with the `logs` table; fails when they disagree.
fn assert_fts_consistent(path: &Path) {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)
        .expect("the store file opens");
    connection
        .execute(
            "INSERT INTO logs_fts(logs_fts, rank) VALUES ('integrity-check', 1)",
            [],
        )
        .expect("the FTS5 index agrees with the logs table");
}

/// A `vector` column from rows that may be absent, ragged or hold absent values.
fn vector_column(rows: &[Option<Vec<Option<f32>>>]) -> Column {
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        VECTOR.into(),
        rows.len(),
        rows.len() * usize::from(DIM),
        DataType::Float32,
    );
    for row in rows {
        match row {
            Some(values) => builder
                .append_series(&Series::new("".into(), values.as_slice()))
                .expect("the values are 32-bit floats"),
            None => builder.append_null(),
        }
    }
    builder.finish().into_column()
}

/// `frame` with its `vector` column replaced by `vectors`.
fn with_vectors(mut frame: DataFrame, vectors: &[Option<Vec<Option<f32>>>]) -> DataFrame {
    frame
        .with_column(vector_column(vectors))
        .expect("the column replaces the vectors");
    frame
}

fn present(vector: Vec<f32>) -> Option<Vec<Option<f32>>> {
    Some(vector.into_iter().map(Some).collect())
}

/// Three hand-made logs.
fn three_logs() -> LogsFrame {
    logs_of(&[
        record("GET /a?x=1", Label::Good, Some("/a"), 2),
        record("GET /b?x=<script>", Label::Bad, None, 1),
        record("POST /c\nname=o'neil", Label::Unknown, Some("/c"), 5),
    ])
}

fn invalid_rows(error: StoreError) -> String {
    match error {
        StoreError::InvalidRows { detail } => detail,
        other => panic!("expected InvalidRows, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// The dependency
// ---------------------------------------------------------------------------------------

/// The design asks for bundled SQLite with FTS5. Release 0.32 of the binding has no
/// feature of that name; the bundled source is compiled with the extension. This fails
/// if a change of the dependency loses it or the `trigram` tokenizer (SQLite 3.34).
#[test]
fn the_bundled_sqlite_has_fts5_with_the_trigram_tokenizer() {
    let connection = Connection::open_in_memory().unwrap();
    let fts5: bool = connection
        .query_row(
            "SELECT sqlite_compileoption_used('ENABLE_FTS5')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(fts5, "the linked SQLite was built without FTS5");
    let version: String = connection
        .query_row("SELECT sqlite_version()", [], |row| row.get(0))
        .unwrap();
    let parts: Vec<u32> = version
        .split('.')
        .map(|part| part.parse().unwrap())
        .collect();
    assert!(
        (parts[0], parts[1]) >= (3, 34),
        "SQLite {version} has no trigram tokenizer"
    );
    connection
        .execute_batch("CREATE VIRTUAL TABLE probe USING fts5(text, tokenize = 'trigram')")
        .expect("the trigram tokenizer is available");
    // The SQLite that is linked is the one whose source ships with the binding, not a
    // system library that an environment variable of the build put in its place: the
    // version the linked library reports is the version of the bundled header.
    let bundled = std::ffi::CStr::from_bytes_with_nul(rusqlite::ffi::SQLITE_VERSION)
        .expect("the header's version is a C string")
        .to_str()
        .unwrap();
    assert_eq!(version, bundled);
    assert_eq!(
        bundled, "3.46.0",
        "the bundled SQLite changed: re-check the workspace manifest"
    );
}

// ---------------------------------------------------------------------------------------
// Create and open
// ---------------------------------------------------------------------------------------

#[test]
fn create_writes_one_database_file_with_the_identity_in_meta() {
    let scratch = Scratch::new("create");
    let paths = scratch.paths();
    let store = Store::create(&paths, "fixture", &identity()).expect("the store is created");

    let path = paths.stores_dir().join("fixture.db");
    assert_eq!(paths.store_path("fixture"), path);
    assert_eq!(store.path(), path);
    assert_eq!(store.name(), "fixture");
    assert_eq!(store.identity(), &identity());
    assert_eq!(store.tool_version(), env!("CARGO_PKG_VERSION"));
    assert_eq!(store.len().unwrap(), 0);
    assert!(store.is_empty().unwrap());
    let created_at = store.created_at().to_string();
    // UTC, to the second: 2026-11-12T09:30:00Z.
    let bytes = created_at.as_bytes();
    assert_eq!(bytes.len(), 20, "{created_at}");
    assert!(
        created_at.ends_with('Z') && bytes[10] == b'T',
        "{created_at}"
    );
    assert!(created_at.starts_with("20"), "{created_at}");
    drop(store);

    assert_eq!(scratch.store_files(), ["fixture.db"]);
    let connection = read_only(&path);
    let meta: (String, String, i64, String, String) = connection
        .query_row(
            "SELECT encoder_name, encoder_revision, encoder_dimension, created_at, tool_version \
             FROM meta",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .expect("meta holds exactly one row");
    assert_eq!(
        meta,
        (
            "bytecnn".to_string(),
            REVISION.to_string(),
            i64::from(DIM),
            created_at,
            env!("CARGO_PKG_VERSION").to_string()
        )
    );
    let version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, i64::from(SCHEMA_VERSION));
}

#[test]
fn the_schema_has_the_tables_columns_and_indexes_of_the_task() {
    let scratch = Scratch::new("schema");
    let paths = scratch.paths();
    drop(Store::create(&paths, "s", &identity()).unwrap());
    let connection = read_only(&paths.store_path("s"));

    let columns = |table: &str| -> Vec<String> {
        let mut statement = connection
            .prepare(&format!("SELECT name FROM pragma_table_info('{table}')"))
            .unwrap();
        statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    assert_eq!(
        columns("logs"),
        [
            "seq",
            "id",
            "text",
            "source",
            "occurrences",
            "label",
            "group_key",
            "truncated",
            "vector",
            "encoder_revision"
        ]
    );
    assert_eq!(
        columns("meta"),
        [
            "id",
            "encoder_name",
            "encoder_revision",
            "encoder_dimension",
            "created_at",
            "tool_version"
        ]
    );

    let fts: String = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'logs_fts'",
            [],
            |row| row.get(0),
        )
        .expect("the logs_fts table exists");
    let fts = fts.to_ascii_lowercase();
    assert!(fts.contains("using fts5"), "{fts}");
    assert!(fts.contains("trigram"), "{fts}");

    // Requirement 5.5: restrictions by label and by source are answered from indexes.
    let mut statement = connection
        .prepare(
            "SELECT info.name FROM pragma_index_list('logs') AS list, \
             pragma_index_info(list.name) AS info WHERE info.seqno = 0",
        )
        .unwrap();
    let indexed: Vec<String> = statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    for column in ["id", "label", "source"] {
        assert!(
            indexed.iter().any(|name| name == column),
            "no index leads with '{column}': {indexed:?}"
        );
    }
}

#[test]
fn create_refuses_a_name_that_exists_and_leaves_the_store_as_it_was() {
    let scratch = Scratch::new("exists");
    let paths = scratch.paths();
    let logs = three_logs();
    let mut store = Store::create(&paths, "kept", &identity()).unwrap();
    store
        .add_with(&rows(&logs, "a.log", 1), &identity(), &mut budget())
        .unwrap();
    drop(store);
    let before = dump(&paths.store_path("kept"));

    let other = EncoderIdentity {
        name: "other".to_string(),
        ..identity()
    };
    let error = Store::create(&paths, "kept", &other).expect_err("the name is taken");
    match &error {
        StoreError::Exists { name, at } => {
            assert_eq!(name, "kept");
            assert_eq!(at, &paths.store_path("kept"));
        }
        other => panic!("expected Exists, got {other:?}"),
    }
    assert!(matches!(L2dError::from(error), L2dError::Refused { .. }));
    assert_eq!(dump(&paths.store_path("kept")), before);
    let store = Store::open(&paths, "kept").unwrap();
    assert_eq!(store.identity(), &identity());
}

#[test]
fn open_names_a_missing_store_and_where_it_was_expected() {
    let scratch = Scratch::new("missing");
    let paths = scratch.paths();
    let error = Store::open(&paths, "absent").expect_err("there is no such store");
    assert_eq!(
        error,
        StoreError::Missing {
            name: "absent".to_string(),
            expected_at: paths.store_path("absent"),
        }
    );
    let message = error.to_string();
    assert!(message.contains("absent"), "{message}");
    assert!(
        message.contains(&paths.store_path("absent").display().to_string()),
        "{message}"
    );
    // Requirement 10.5, exit code 3.
    assert_eq!(
        L2dError::from(error),
        L2dError::Missing {
            item: "store absent".to_string(),
            expected_at: paths.store_path("absent"),
        }
    );
    // Looking for a store does not create anything.
    assert!(!paths.stores_dir().exists());
}

#[test]
fn names_that_are_not_plain_file_names_are_rejected() {
    let scratch = Scratch::new("names");
    let paths = scratch.paths();
    let too_long = "n".repeat(MAX_NAME_BYTES + 1);
    for name in [
        "", ".hidden", "a/b", "a\\b", "..", "a b", "a:b", "größe", &too_long,
    ] {
        for error in [
            Store::create(&paths, name, &identity()).expect_err("create refuses the name"),
            Store::open(&paths, name).expect_err("open refuses the name"),
        ] {
            assert!(
                matches!(error, StoreError::InvalidName { .. }),
                "name {name:?}: {error:?}"
            );
            assert!(matches!(
                L2dError::from(error),
                L2dError::InvalidInput { .. }
            ));
        }
    }
    assert!(!paths.stores_dir().exists(), "nothing was created");
    let longest = "n".repeat(MAX_NAME_BYTES);
    Store::create(&paths, &longest, &identity()).expect("a name of the longest length is fine");
    Store::create(&paths, "csic-2010_v1.a", &identity()).expect("the allowed characters");
}

#[test]
fn create_refuses_an_identity_that_cannot_be_one() {
    let scratch = Scratch::new("identity");
    let paths = scratch.paths();
    let unusable = [
        EncoderIdentity {
            name: String::new(),
            ..identity()
        },
        EncoderIdentity {
            revision: String::new(),
            ..identity()
        },
        EncoderIdentity {
            dimension: 0,
            ..identity()
        },
    ];
    for given in unusable {
        let error = Store::create(&paths, "s", &given).expect_err("the identity is unusable");
        assert!(
            matches!(error, StoreError::InvalidIdentity { .. }),
            "{given:?}: {error:?}"
        );
        assert!(matches!(
            L2dError::from(error),
            L2dError::InvalidInput { .. }
        ));
    }
    assert!(scratch.store_files().is_empty(), "nothing was created");
}

#[test]
fn open_refuses_a_file_that_is_not_a_store() {
    let scratch = Scratch::new("not-a-store");
    let paths = scratch.paths();
    std::fs::create_dir_all(paths.stores_dir()).unwrap();

    // Not a database at all.
    std::fs::write(paths.store_path("garbage"), [b'x'; 4096]).unwrap();
    // An empty file: what an interrupted creation leaves behind.
    std::fs::write(paths.store_path("empty"), b"").unwrap();
    // A database of some other program.
    let foreign = Connection::open(paths.store_path("foreign")).unwrap();
    foreign
        .execute_batch("CREATE TABLE logs (id TEXT); CREATE TABLE meta (x);")
        .unwrap();
    drop(foreign);
    // A store written by a later version of the tool.
    drop(Store::create(&paths, "newer", &identity()).unwrap());
    let newer = Connection::open(paths.store_path("newer")).unwrap();
    newer
        .pragma_update(None, "user_version", SCHEMA_VERSION + 1)
        .unwrap();
    drop(newer);

    for name in ["garbage", "empty", "foreign", "newer"] {
        let before = std::fs::read(paths.store_path(name)).unwrap();
        let error = Store::open(&paths, name).expect_err("the file is not a usable store");
        let StoreError::Failed { detail } = &error else {
            panic!("{name}: expected Failed, got {error:?}");
        };
        assert!(
            detail.contains(&paths.store_path(name).display().to_string()),
            "{name}: {detail}"
        );
        if name == "newer" {
            assert!(
                detail.contains(&(SCHEMA_VERSION + 1).to_string()),
                "{detail}"
            );
        }
        assert!(matches!(L2dError::from(error), L2dError::Runtime { .. }));
        assert_eq!(
            std::fs::read(paths.store_path(name)).unwrap(),
            before,
            "{name} was changed by the refused open"
        );
        // The name stays taken.
        assert!(matches!(
            Store::create(&paths, name, &identity()),
            Err(StoreError::Exists { .. })
        ));
    }
}

// ---------------------------------------------------------------------------------------
// Add, restart
// ---------------------------------------------------------------------------------------

#[test]
fn the_id_is_the_sha256_of_source_and_text() {
    let source = "data/x.log";
    let text = "GET /a?x=1";
    let mut hasher = Sha256::new();
    hasher.update((source.len() as u64).to_le_bytes());
    hasher.update(source.as_bytes());
    hasher.update(text.as_bytes());
    let expected: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(log_id(source, text), expected);
    assert_eq!(log_id(source, text).len(), 64);
    // The length prefix keeps source and text apart.
    assert_ne!(log_id("ab", "c"), log_id("a", "bc"));
    assert_ne!(log_id("a", "b"), log_id("b", "a"));
}

#[test]
fn the_rows_frame_has_the_documented_schema() {
    let logs = three_logs();
    let frame = rows(&logs, "a.log", 1);
    assert_eq!(**frame.schema(), rows_schema());
    let names: Vec<&str> = frame
        .get_column_names()
        .into_iter()
        .map(|name| name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "text",
            "source",
            "occurrences",
            "label",
            "group",
            "truncated",
            "vector"
        ]
    );
    assert_eq!(frame.height(), 3);

    let vectors = vectors_of(&logs, 1, DIM);
    let truncated = truncated_of(&logs);
    // One vector too few, one flag too few.
    let short = &vectors[..vectors.len() - usize::from(DIM)];
    let detail = invalid_rows(rows_frame(&logs, "a.log", short, DIM, &truncated).unwrap_err());
    assert!(detail.contains('3') && detail.contains('2'), "{detail}");
    let detail =
        invalid_rows(rows_frame(&logs, "a.log", &vectors, DIM, &truncated[..2]).unwrap_err());
    assert!(detail.contains('3') && detail.contains('2'), "{detail}");
    // No rows is a frame too.
    let none = rows_frame(&LogsFrame::empty(), "a.log", &[], DIM, &[]).unwrap();
    assert_eq!(none.height(), 0);
    assert_eq!(**none.schema(), rows_schema());
}

#[test]
fn added_logs_survive_a_restart_unchanged() {
    let scratch = Scratch::new("restart");
    let paths = scratch.paths();
    let logs = fixture_logs();

    let mut store = Store::create(&paths, "fixture", &identity()).unwrap();
    let created_at = store.created_at().to_string();
    let report: AddReport = store
        .add_with(&rows(&logs, SOURCE, 1), &identity(), &mut budget())
        .expect("the fixture is added");
    assert_eq!((report.rows, report.added, report.repeated), (169, 169, 0));
    assert_eq!(store.len().unwrap(), 169);
    drop(store);

    // At rest the store is the one file: no journal, no write-ahead log.
    assert_eq!(scratch.store_files(), ["fixture.db"]);

    let store = Store::open(&paths, "fixture").expect("the store opens again");
    assert_eq!(store.identity(), &identity());
    assert_eq!(store.created_at(), created_at);
    assert_eq!(store.tool_version(), env!("CARGO_PKG_VERSION"));
    assert_eq!(store.len().unwrap(), 169);
    assert!(!store.is_empty().unwrap());
    drop(store);

    let stored = dump(&paths.store_path("fixture"));
    assert_eq!(stored, expected(&logs, SOURCE, 1));
    // Requirement 5.1, spelled out for one row: text, source, occurrences, label, the
    // vector bit for bit, and the encoder revision.
    let first = logs.records().next().unwrap();
    assert_eq!(stored[0].text, first.text);
    assert_eq!(stored[0].source, SOURCE);
    assert_eq!(stored[0].label, "good");
    assert_eq!(stored[0].vector.len(), usize::from(DIM) * 4);
    assert_eq!(stored[0].encoder_revision, REVISION);
    assert!(
        stored.iter().any(|row| row.truncated == 1) && stored.iter().any(|row| row.truncated == 0)
    );
    assert!(stored.iter().any(|row| row.occurrences > 1));
    assert!(stored.iter().any(|row| row.label == "unknown"));
}

#[test]
fn the_default_add_observes_the_host_and_reports_its_plan() {
    let scratch = Scratch::new("default-add");
    let paths = scratch.paths();
    let logs = three_logs();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    let report = store
        .add(&rows(&logs, "a.log", 1), &identity())
        .expect("the host has memory for three rows");
    assert_eq!((report.rows, report.added, report.repeated), (3, 3, 0));
    // Requirement 12.3: free memory observed, limit applied, batch size chosen.
    assert!(report.plan.free_bytes > 0);
    assert!(report.plan.limit_bytes > 0);
    assert!(report.plan.bytes_per_row > 0);
    assert!(report.plan.batch_rows > 0);
    assert!(!report.plan.overridden);
    drop(store);
    assert_eq!(dump(&paths.store_path("s")), expected(&logs, "a.log", 1));
}

#[test]
fn adding_an_existing_id_adds_its_occurrences_and_changes_nothing_else() {
    let scratch = Scratch::new("repeat");
    let paths = scratch.paths();
    let first = three_logs();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    store
        .add_with(&rows(&first, "a.log", 1), &identity(), &mut budget())
        .unwrap();
    let before = dump(&paths.store_path("s"));

    // The same texts again from the same source, with other labels, groups, flags and
    // vectors (salt 2), and one text that is new. One text occurs twice in the frame.
    let again = logs_of(&[
        record("GET /b?x=<script>", Label::Good, Some("/other"), 10),
        record("GET /new", Label::Bad, None, 4),
        record("GET /a?x=1", Label::Bad, None, 7),
    ]);
    let mut frame = rows(&again, "a.log", 2);
    let twice = rows(
        &logs_of(&[record("GET /new", Label::Good, Some("/g"), 3)]),
        "a.log",
        3,
    );
    frame.vstack_mut(&twice).unwrap();
    let report = store
        .add_with(&frame, &identity(), &mut budget())
        .expect("repeated ids are not an error");
    assert_eq!((report.rows, report.added, report.repeated), (4, 1, 3));
    assert_eq!(store.len().unwrap(), 4);
    drop(store);

    let after = dump(&paths.store_path("s"));
    let mut wanted = before.clone();
    wanted[0].occurrences = 2 + 7;
    wanted[1].occurrences = 1 + 10;
    wanted.push(Raw {
        seq: 4,
        id: log_id("a.log", "GET /new"),
        text: "GET /new".to_string(),
        source: "a.log".to_string(),
        // 4 from its first row in the frame, 3 from its second.
        occurrences: 7,
        // Label, group, flag and vector are those of the first row that brought the id.
        label: "bad".to_string(),
        group_key: None,
        truncated: 0,
        vector: le_bytes(&unit_vector("GET /new", 2, DIM)),
        encoder_revision: REVISION.to_string(),
    });
    assert_eq!(after, wanted);
    assert_fts_consistent(&paths.store_path("s"));
    // The repeated text is in the text index once.
    assert_eq!(
        fts_ids(&paths.store_path("s"), "GET /new"),
        [log_id("a.log", "GET /new")]
    );
}

#[test]
fn the_same_text_from_another_source_is_another_log() {
    let scratch = Scratch::new("sources");
    let paths = scratch.paths();
    let logs = three_logs();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    store
        .add_with(&rows(&logs, "monday.log", 1), &identity(), &mut budget())
        .unwrap();
    let report = store
        .add_with(&rows(&logs, "tuesday.log", 1), &identity(), &mut budget())
        .unwrap();
    assert_eq!((report.rows, report.added, report.repeated), (3, 3, 0));
    drop(store);

    let stored = dump(&paths.store_path("s"));
    assert_eq!(stored.len(), 6);
    let mut wanted = expected(&logs, "monday.log", 1);
    wanted.extend(
        expected(&logs, "tuesday.log", 1)
            .into_iter()
            .map(|row| Raw {
                seq: row.seq + 3,
                ..row
            }),
    );
    assert_eq!(stored, wanted);
}

#[test]
fn an_empty_frame_adds_nothing() {
    let scratch = Scratch::new("empty-frame");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    let report = store
        .add_with(
            &rows(&LogsFrame::empty(), "a.log", 1),
            &identity(),
            &mut budget(),
        )
        .expect("no rows is not an error");
    assert_eq!((report.rows, report.added, report.repeated), (0, 0, 0));
    assert_eq!(store.len().unwrap(), 0);
}

// ---------------------------------------------------------------------------------------
// Encoder identity (requirement 5.6)
// ---------------------------------------------------------------------------------------

#[test]
fn vectors_of_another_identity_are_refused_with_both_identities_named() {
    let scratch = Scratch::new("mismatch");
    let paths = scratch.paths();
    let logs = three_logs();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    store
        .add_with(&rows(&logs, "a.log", 1), &identity(), &mut budget())
        .unwrap();
    drop(store);
    let before = dump(&paths.store_path("s"));

    let others = [
        EncoderIdentity {
            name: "wordcnn".to_string(),
            ..identity()
        },
        EncoderIdentity {
            revision: OTHER_REVISION.to_string(),
            ..identity()
        },
        EncoderIdentity {
            dimension: 16,
            ..identity()
        },
    ];
    // After a restart the identity comes from the file.
    let mut store = Store::open(&paths, "s").unwrap();
    for given in others {
        // Rows that would otherwise be accepted: only the identity differs.
        let more = rows(
            &logs_of(&[record("GET /new", Label::Bad, None, 1)]),
            "a.log",
            1,
        );
        let error = store
            .add_with(&more, &given, &mut budget())
            .expect_err("another identity is refused");
        assert_eq!(
            error,
            StoreError::IdentityMismatch {
                store: identity(),
                given: given.clone(),
            }
        );
        assert_eq!(store.require_identity(&given), Err(error.clone()));
        let message = error.to_string();
        for part in [
            "bytecnn",
            REVISION,
            &DIM.to_string(),
            given.name.as_str(),
            given.revision.as_str(),
            &given.dimension.to_string(),
        ] {
            assert!(message.contains(part), "'{part}' is not in: {message}");
        }
        // Requirement 5.6 through the tool-wide error: a refusal, exit code 4, with the
        // same text.
        match L2dError::from(error) {
            L2dError::Refused { reason } => assert_eq!(reason, message),
            other => panic!("expected Refused, got {other:?}"),
        }
    }
    assert_eq!(store.require_identity(&identity()), Ok(()));
    assert_eq!(store.len().unwrap(), 3);
    drop(store);
    assert_eq!(dump(&paths.store_path("s")), before);
}

// ---------------------------------------------------------------------------------------
// Text index
// ---------------------------------------------------------------------------------------

#[test]
fn fts5_queries_match_the_stored_text_also_after_a_restart() {
    let scratch = Scratch::new("fts");
    let paths = scratch.paths();
    let logs = fixture_logs();
    let mut store = Store::create(&paths, "fixture", &identity()).unwrap();
    store
        .add_with(&rows(&logs, SOURCE, 1), &identity(), &mut budget())
        .unwrap();
    drop(store);
    // A restart in between.
    drop(Store::open(&paths, "fixture").unwrap());
    let path = paths.store_path("fixture");

    let phrases = [
        "/tienda1/publico/buscar.jsp",
        "UNION SELECT",
        "<script>alert(\"xss\")</script>",
        "' OR '1'='1",
        "%27%3B+DROP+TABLE",
        "../../../../etc/passwd",
        "'; DROP TABLE usuarios; --",
        "login=javier&direccion=Calle+Mayor",
        "pwd",
        // The index ignores ASCII case.
        "union select",
    ];
    for phrase in phrases {
        let found = fts_ids(&path, phrase);
        assert!(!found.is_empty(), "'{phrase}' matches nothing");
        assert!(found.len() < 169, "'{phrase}' matches everything");
        assert_eq!(
            found,
            scan_ids(&logs, SOURCE, phrase),
            "'{phrase}': the index and a scan of the texts disagree"
        );
    }
    // A whole text is found exactly.
    let whole = logs.records().nth(100).unwrap().text;
    assert_eq!(fts_ids(&path, &whole), [log_id(SOURCE, &whole)]);
    // What is in no text matches nothing.
    assert!(fts_ids(&path, "no such text anywhere").is_empty());
    assert_fts_consistent(&path);
}

#[test]
fn the_text_index_follows_every_add() {
    let scratch = Scratch::new("fts-adds");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    let first = logs_of(&[
        record("GET /a?name=señor+núñez", Label::Good, None, 1),
        record("line one\nline two with a\ttab", Label::Unknown, None, 1),
    ]);
    store
        .add_with(&rows(&first, "a.log", 1), &identity(), &mut budget())
        .unwrap();
    let second = logs_of(&[record("GET /b?name=núñez", Label::Bad, None, 1)]);
    store
        .add_with(&rows(&second, "b.log", 1), &identity(), &mut budget())
        .unwrap();
    drop(store);
    let path = paths.store_path("s");

    assert_eq!(
        fts_ids(&path, "núñez"),
        [
            log_id("a.log", "GET /a?name=señor+núñez"),
            log_id("b.log", "GET /b?name=núñez")
        ]
    );
    assert_eq!(
        fts_ids(&path, "one\nline"),
        [log_id("a.log", "line one\nline two with a\ttab")]
    );
    assert_fts_consistent(&path);
}

// ---------------------------------------------------------------------------------------
// Rows that cannot be stored
// ---------------------------------------------------------------------------------------

#[test]
fn a_frame_with_another_schema_is_not_added() {
    let scratch = Scratch::new("schema-mismatch");
    let paths = scratch.paths();
    let logs = three_logs();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();

    // The logs frame itself: no source, flag or vector.
    let detail = invalid_rows(
        store
            .add_with(logs.frame(), &identity(), &mut budget())
            .unwrap_err(),
    );
    assert!(detail.contains("vector"), "{detail}");
    // A column of another type.
    let mut wrong = rows(&logs, "a.log", 1);
    wrong
        .with_column(Column::new("occurrences".into(), [1u64, 2, 3]))
        .unwrap();
    let detail = invalid_rows(
        store
            .add_with(&wrong, &identity(), &mut budget())
            .unwrap_err(),
    );
    assert!(detail.contains("occurrences"), "{detail}");
    // An absent value outside `group`.
    let mut absent = rows(&logs, "a.log", 1);
    absent
        .with_column(Column::new(
            "source".into(),
            [Some("a.log"), None, Some("a.log")],
        ))
        .unwrap();
    let detail = invalid_rows(
        store
            .add_with(&absent, &identity(), &mut budget())
            .unwrap_err(),
    );
    assert!(detail.contains("source"), "{detail}");
    // A label that is none of the three.
    let mut label = rows(&logs, "a.log", 1);
    label
        .with_column(Column::new("label".into(), ["good", "evil", "bad"]))
        .unwrap();
    let detail = invalid_rows(
        store
            .add_with(&label, &identity(), &mut budget())
            .unwrap_err(),
    );
    assert!(
        detail.contains("evil") && detail.contains("row 2"),
        "{detail}"
    );
    // No occurrence at all.
    let mut none = rows(&logs, "a.log", 1);
    none.with_column(Column::new("occurrences".into(), [1u32, 1, 0]))
        .unwrap();
    let detail = invalid_rows(
        store
            .add_with(&none, &identity(), &mut budget())
            .unwrap_err(),
    );
    assert!(detail.contains("row 3"), "{detail}");

    assert!(matches!(
        L2dError::from(StoreError::InvalidRows {
            detail: String::new()
        }),
        L2dError::Runtime { .. }
    ));
    assert_eq!(store.len().unwrap(), 0);
}

#[test]
fn a_vector_that_is_not_one_of_the_stores_dimension_is_not_added() {
    let scratch = Scratch::new("vectors");
    let paths = scratch.paths();
    let logs = three_logs();
    let good = |index: usize| {
        let text = logs.records().nth(index).unwrap().text;
        unit_vector(&text, 1, DIM)
    };
    let mut store = Store::create(&paths, "s", &identity()).unwrap();

    let cases: [(&str, Option<Vec<Option<f32>>>); 6] = [
        ("7 values", present(good(1)[..7].to_vec())),
        ("9 values", present([good(1), vec![0.0]].concat())),
        ("no vector", None),
        ("an absent value", {
            let mut values = present(good(1)).unwrap();
            values[3] = None;
            Some(values)
        }),
        ("not a number", {
            let mut values = good(1);
            values[0] = f32::NAN;
            present(values)
        }),
        ("infinite", {
            let mut values = good(1);
            values[7] = f32::INFINITY;
            present(values)
        }),
    ];
    for (what, bad) in cases {
        let frame = with_vectors(
            rows(&logs, "a.log", 1),
            &[present(good(0)), bad, present(good(2))],
        );
        // One row to a batch: the first row is already written when the second fails.
        let error = store
            .add_with(&frame, &identity(), &mut budget_with(None, Some(1)))
            .expect_err(what);
        let detail = invalid_rows(error);
        if what == "no vector" {
            // An absent value is found by the check of the columns, before any row.
            assert!(detail.contains("'vector'"), "{what}: {detail}");
        } else {
            assert!(detail.contains("row 2"), "{what}: {detail}");
        }
        // Nothing of the frame stays: an add is all or nothing.
        assert_eq!(store.len().unwrap(), 0, "{what}");
    }
    drop(store);
    assert!(dump(&paths.store_path("s")).is_empty());
    assert_fts_consistent(&paths.store_path("s"));

    // The database itself holds the rule, whatever writes to the file.
    let connection = Connection::open(paths.store_path("s")).unwrap();
    let insert = |label: &str, vector: Vec<u8>| {
        connection.execute(
            "INSERT INTO logs (id, text, source, occurrences, label, group_key, truncated, \
             vector, encoder_revision) VALUES ('x', 't', 's', 1, ?1, NULL, 0, ?2, 'r')",
            rusqlite::params![label, vector],
        )
    };
    insert("good", vec![0; 28]).expect_err("a vector of 7 values");
    insert("evil", vec![0; 32]).expect_err("an unknown label");
    insert("good", vec![0; 32]).expect("a row that keeps the rules");
}

#[test]
fn occurrences_beyond_32_bits_fail_and_leave_the_store_unchanged() {
    let scratch = Scratch::new("overflow");
    let paths = scratch.paths();
    let logs = logs_of(&[
        record("GET /a", Label::Good, None, 1),
        record("GET /b", Label::Good, None, u32::MAX),
    ]);
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    store
        .add_with(&rows(&logs, "a.log", 1), &identity(), &mut budget())
        .unwrap();
    let before = dump(&paths.store_path("s"));
    assert_eq!(before[1].occurrences, i64::from(u32::MAX));

    let error = store
        .add_with(&rows(&logs, "a.log", 1), &identity(), &mut budget())
        .expect_err("the count does not fit 32 bits");
    assert!(matches!(error, StoreError::Failed { .. }), "{error:?}");
    drop(store);
    // The first row of the failed add was not counted either.
    assert_eq!(dump(&paths.store_path("s")), before);
}

// ---------------------------------------------------------------------------------------
// Memory (requirement 12)
// ---------------------------------------------------------------------------------------

#[test]
fn the_stored_rows_do_not_depend_on_the_batch_size() {
    let logs = fixture_logs();
    let frame = rows(&logs, SOURCE, 1);
    let wanted = expected(&logs, SOURCE, 1);
    for batch_rows in [Some(1), Some(7), Some(64), Some(169), Some(1024), None] {
        let scratch = Scratch::new("batch-size");
        let paths = scratch.paths();
        let mut store = Store::create(&paths, "s", &identity()).unwrap();
        let mut budget = budget_with(None, batch_rows);
        let report = store.add_with(&frame, &identity(), &mut budget).unwrap();
        assert_eq!((report.rows, report.added, report.repeated), (169, 169, 0));
        let largest = batch_rows.unwrap_or(65_536);
        assert_eq!(report.plan.batch_rows, largest, "{batch_rows:?}");
        assert_eq!(report.plan.max_rows_used, largest, "{batch_rows:?}");
        assert_eq!(report.plan.overridden, batch_rows.is_some());
        assert!(report.plan.changes.is_empty());
        drop(store);
        assert_eq!(dump(&paths.store_path("s")), wanted, "{batch_rows:?}");
        assert_fts_consistent(&paths.store_path("s"));
        for phrase in ["UNION SELECT", "' OR '1'='1"] {
            assert_eq!(
                fts_ids(&paths.store_path("s"), phrase),
                scan_ids(&logs, SOURCE, phrase),
                "{batch_rows:?}"
            );
        }
    }
}

#[test]
fn free_memory_is_read_before_every_batch_and_the_batch_follows_it() {
    let logs = synthetic_logs(6_000);
    let frame = rows(&logs, "synthetic", 1);

    // Steady memory: the reference.
    let steady = Scratch::new("steady");
    let mut store = Store::create(&steady.paths(), "s", &identity()).unwrap();
    let report = store
        .add_with(&frame, &identity(), &mut budget_with(None, Some(2_048)))
        .unwrap();
    assert!(report.plan.changes.is_empty());
    drop(store);
    let wanted = dump(&steady.paths().store_path("s"));
    assert_eq!(wanted.len(), 6_000);

    // Free memory at the start 8 GiB, so the reserve is 2 GiB. Then, before the batches:
    // 8 GiB, 1 GiB (below the reserve: halve), 1 GiB (already the smallest), 8 GiB
    // (recovered: double).
    let readings = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&readings);
    let settings = MemorySettings {
        max_batch_rows: Some(2_048),
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&settings, move || {
        Ok(match counter.fetch_add(1, Ordering::Relaxed) {
            2 | 3 => GIB,
            _ => 8 * GIB,
        })
    })
    .unwrap();
    let moving = Scratch::new("moving");
    let mut store = Store::create(&moving.paths(), "s", &identity()).unwrap();
    let report = store.add_with(&frame, &identity(), &mut budget).unwrap();
    drop(store);

    // Once at the start and once before each of the four batches.
    assert_eq!(readings.load(Ordering::Relaxed), 5);
    let changes: Vec<_> = report
        .plan
        .changes
        .iter()
        .map(|change| {
            (
                change.at_row,
                change.from_rows,
                change.to_rows,
                change.cause,
                change.free_bytes,
            )
        })
        .collect();
    assert_eq!(
        changes,
        [
            (2_048, 2_048, 1_024, BatchCause::BelowReserve, GIB),
            (4_096, 1_024, 2_048, BatchCause::Recovered, 8 * GIB),
        ]
    );
    assert_eq!(report.plan.min_rows_used, 1_024);
    assert_eq!(report.plan.max_rows_used, 2_048);
    assert_eq!(
        (report.rows, report.added, report.repeated),
        (6_000, 6_000, 0)
    );
    // Requirement 12.6: the same store whatever the batches were.
    assert_eq!(dump(&moving.paths().store_path("s")), wanted);
    assert_fts_consistent(&moving.paths().store_path("s"));
}

#[test]
fn an_add_that_does_not_fit_the_memory_limit_is_refused_before_anything_is_written() {
    let scratch = Scratch::new("refused");
    let paths = scratch.paths();
    let logs = three_logs();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    let before = std::fs::read(paths.store_path("s")).unwrap();

    // 33 MiB: 32 MiB are the tool's own, 1 MiB is left, less than the store's own share.
    let limit = 33 * MIB;
    const { assert!(ADD_FIXED_BYTES > MIB) };
    let mut budget = budget_with(Some(limit), None);
    let left = budget.left_bytes();
    let error = store
        .add_with(&rows(&logs, "a.log", 1), &identity(), &mut budget)
        .expect_err("one mebibyte is not enough");
    let StoreError::Memory(inner) = &error else {
        panic!("expected Memory, got {error:?}");
    };
    let L2dError::Refused { reason } = inner else {
        panic!("expected a refusal, got {inner:?}");
    };
    // Requirement 12.4: the memory needed and the memory available.
    assert!(reason.contains(&limit.to_string()), "{reason}");
    assert!(reason.contains(&ADD_FIXED_BYTES.to_string()), "{reason}");
    assert!(reason.contains(&MIB.to_string()), "{reason}");
    assert_eq!(L2dError::from(error.clone()), inner.clone());
    assert_eq!(error.to_string(), inner.to_string());
    // The budget is handed back as it came.
    assert_eq!(budget.left_bytes(), left);
    assert_eq!(store.len().unwrap(), 0);
    drop(store);
    assert_eq!(std::fs::read(paths.store_path("s")).unwrap(), before);

    // The smallest limit that is enough for the store's share and these three rows.
    let mut store = Store::open(&paths, "s").unwrap();
    let mut enough = budget_with(Some(32 * MIB + ADD_FIXED_BYTES + MIB), None);
    store
        .add_with(&rows(&logs, "a.log", 1), &identity(), &mut enough)
        .expect("the store's share and a mebibyte for the rows");
}

#[test]
fn add_counts_its_own_share_on_top_of_what_the_caller_holds_and_hands_it_back() {
    let scratch = Scratch::new("held");
    let paths = scratch.paths();
    let logs = three_logs();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();

    // The caller holds a logs frame of 20 MiB within a limit of 64 MiB: 12 MiB are left,
    // less than the store's share.
    let mut budget = budget_with(Some(64 * MIB), None);
    budget.hold(20 * MIB);
    const {
        assert!(
            ADD_FIXED_BYTES > 12 * MIB,
            "the test assumes a larger share"
        )
    };
    let left = budget.left_bytes();
    let error = store
        .add_with(&rows(&logs, "a.log", 1), &identity(), &mut budget)
        .expect_err("what the caller holds counts");
    assert!(matches!(
        error,
        StoreError::Memory(L2dError::Refused { .. })
    ));
    assert_eq!(budget.left_bytes(), left);

    // With 1 MiB held there is room, and afterwards the caller's 1 MiB is held again.
    budget.hold(MIB);
    let left = budget.left_bytes();
    store
        .add_with(&rows(&logs, "a.log", 1), &identity(), &mut budget)
        .unwrap();
    assert_eq!(budget.left_bytes(), left);
    assert_eq!(store.len().unwrap(), 3);
}

/// What the reviewer's probe found: the budget refuses before the third batch. The rows
/// of the first two batches are rolled back, the budget is handed back, and the store
/// can be used.
#[test]
fn a_refusal_between_batches_rolls_the_add_back_and_hands_the_budget_back() {
    let scratch = Scratch::new("midway");
    let paths = scratch.paths();
    let logs = synthetic_logs(5_000);
    let frame = rows(&logs, "synthetic", 1);
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    let already = logs_of(&[record("already there", Label::Good, None, 1)]);
    store
        .add_with(&rows(&already, "a.log", 1), &identity(), &mut budget())
        .unwrap();
    let before = std::fs::read(paths.store_path("s")).unwrap();

    // 8 GiB free at the start and before the first two batches, then 1,000 bytes.
    let readings = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&readings);
    let settings = MemorySettings {
        max_batch_rows: Some(1_024),
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&settings, move || {
        Ok(match counter.fetch_add(1, Ordering::Relaxed) {
            0..=2 => 8 * GIB,
            _ => 1_000,
        })
    })
    .unwrap();
    budget.hold(12_345);
    let left = budget.left_bytes();

    let error = store
        .add_with(&frame, &identity(), &mut budget)
        .expect_err("1,000 free bytes are not enough for a batch");
    let StoreError::Memory(L2dError::Refused { reason }) = &error else {
        panic!("expected a memory refusal, got {error:?}");
    };
    // Requirement 12.4: needed, available, and how far the add had come.
    assert!(reason.contains("1000 bytes"), "{reason}");
    assert!(reason.contains("2048 rows"), "{reason}");
    // In the store's terms: its own share is named as such, not as rows that were read,
    // and the rows written are said to be rolled back.
    assert!(!reason.contains("rows read so far"), "{reason}");
    let own = ADD_FIXED_BYTES + ADD_TEXT_FACTOR * longest_text(&logs);
    assert!(
        reason.contains(&format!("{own} bytes are the store's own share")),
        "{reason}"
    );
    assert!(reason.contains("rolled back"), "{reason}");

    assert_eq!(
        budget.left_bytes(),
        left,
        "the caller's 12,345 bytes are held again"
    );
    assert_eq!(store.len().unwrap(), 1);
    drop(store);
    assert_eq!(std::fs::read(paths.store_path("s")).unwrap(), before);
    assert_fts_consistent(&paths.store_path("s"));

    let mut store = Store::open(&paths, "s").unwrap();
    let report = store
        .add_with(&frame, &identity(), &mut budget_with(None, Some(1_024)))
        .expect("with memory the same frame is added");
    assert_eq!((report.added, report.repeated), (5_000, 0));
    drop(store);
    assert_fts_consistent(&paths.store_path("s"));
}

fn longest_text(logs: &LogsFrame) -> u64 {
    logs.records()
        .map(|record| record.text.len() as u64)
        .max()
        .unwrap_or(0)
}

/// SQLite's text index reads a text up to its first NUL character: the rest would be
/// stored and hashed but could not be searched. Such a text is not stored.
#[test]
fn a_text_with_a_nul_character_is_not_added() {
    let scratch = Scratch::new("nul");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    for (text, at) in [("a\0bcdef\0ghi", 2), ("\0", 1), ("GET /a?x=1\0", 11)] {
        let logs = logs_of(&[
            record("GET /first", Label::Good, None, 1),
            record(text, Label::Bad, None, 1),
            record("GET /third", Label::Good, None, 1),
        ]);
        let error = store
            .add_with(
                &rows(&logs, "a.log", 1),
                &identity(),
                &mut budget_with(None, Some(1)),
            )
            .expect_err("a NUL in the text");
        let detail = invalid_rows(error);
        assert!(detail.contains("row 2"), "{text:?}: {detail}");
        assert!(detail.contains("NUL"), "{text:?}: {detail}");
        assert!(detail.contains(&format!("byte {at}")), "{text:?}: {detail}");
        assert_eq!(store.len().unwrap(), 0, "{text:?}");
    }
    drop(store);
    assert!(dump(&paths.store_path("s")).is_empty());
    assert_fts_consistent(&paths.store_path("s"));
}

/// The memory of the text index grows with the longest text. Above the longest length
/// it was measured for, a text is not stored.
#[test]
fn a_text_longer_than_the_longest_measured_is_not_added() {
    let scratch = Scratch::new("too-long");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    let before = std::fs::read(paths.store_path("s")).unwrap();
    let limit = usize::try_from(MAX_TEXT_BYTES).unwrap();

    let too_long = logs_of(&[
        record("GET /first", Label::Good, None, 1),
        record("GET /second", Label::Good, None, 1),
        record(&"a".repeat(limit + 1), Label::Bad, None, 1),
    ]);
    let mut budget = budget();
    let left = budget.left_bytes();
    let error = store
        .add_with(&rows(&too_long, "a.log", 1), &identity(), &mut budget)
        .expect_err("one byte too many");
    let detail = invalid_rows(error);
    assert!(detail.contains("row 3"), "{detail}");
    assert!(detail.contains(&(limit + 1).to_string()), "{detail}");
    assert!(
        detail.contains(&format!("at most {MAX_TEXT_BYTES} bytes")),
        "{detail}"
    );
    assert_eq!(budget.left_bytes(), left);
    // Refused before anything was written, the short rows included.
    assert_eq!(store.len().unwrap(), 0);
    drop(store);
    assert_eq!(std::fs::read(paths.store_path("s")).unwrap(), before);
}

/// The add's own share of the budget covers its longest text, wherever in the frame it
/// is, and an add whose share does not fit is refused before anything is written.
#[test]
fn the_longest_text_of_the_frame_decides_the_adds_own_share() {
    let scratch = Scratch::new("long-share");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    let before = std::fs::read(paths.store_path("s")).unwrap();

    const LONG: u64 = 100_000;
    let mut records: Vec<LogRecord> = (0..300)
        .map(|index| record(&format!("GET /short/{index}"), Label::Good, None, 1))
        .collect();
    // A repeating text: quick to index, and as long as any other of its length.
    records[200] = record(&"abcdefghij".repeat(10_000), Label::Bad, None, 1);
    let logs = logs_of(&records);
    let frame = rows(&logs, "a.log", 1);
    let text_share = ADD_TEXT_FACTOR * LONG;
    let own = ADD_FIXED_BYTES + text_share;
    assert!(text_share > MIB, "the test assumes a share that matters");

    // A limit that leaves one byte less than the add's own share.
    let mut short = budget_with(Some(32 * MIB + own - 1), None);
    let left = short.left_bytes();
    let error = store
        .add_with(&frame, &identity(), &mut short)
        .expect_err("one byte short");
    let StoreError::Memory(L2dError::Refused { reason }) = &error else {
        panic!("expected a memory refusal, got {error:?}");
    };
    for part in [
        format!("needs {own} bytes"),
        format!("{ADD_FIXED_BYTES} bytes for the database"),
        format!("{text_share} bytes for indexing the longest text"),
        format!("{LONG} bytes, row 201"),
        format!("{} bytes are left", own - 1),
        (32 * MIB + own - 1).to_string(),
    ] {
        assert!(reason.contains(&part), "'{part}' is not in: {reason}");
    }
    assert_eq!(short.left_bytes(), left);
    assert_eq!(store.len().unwrap(), 0);
    assert_eq!(std::fs::read(paths.store_path("s")).unwrap(), before);

    // With room for the share and the batch, the add happens and reports its share.
    let mut enough = budget_with(Some(32 * MIB + own + 4 * MIB), None);
    let report = store
        .add_with(&frame, &identity(), &mut enough)
        .expect("the share and four mebibytes for the rows");
    assert_eq!((report.added, report.repeated), (300, 0));
    assert_eq!(report.longest_text_bytes, LONG);
    assert_eq!(report.held_bytes, own);
    drop(store);
    assert_eq!(
        fts_ids(&paths.store_path("s"), "hijabc"),
        [log_id("a.log", &"abcdefghij".repeat(10_000))]
    );

    // Short texts: the share is the fixed one, to within a few kibibytes.
    let short_logs = three_logs();
    let mut store = Store::open(&paths, "s").unwrap();
    let report = store
        .add_with(&rows(&short_logs, "b.log", 1), &identity(), &mut budget())
        .unwrap();
    assert_eq!(report.longest_text_bytes, longest_text(&short_logs));
    assert_eq!(
        report.held_bytes,
        ADD_FIXED_BYTES + ADD_TEXT_FACTOR * longest_text(&short_logs)
    );
    assert!(report.held_bytes < ADD_FIXED_BYTES + 64 * 1024);
}

/// Up to `RELEASE_TEXT_BYTES` the store keeps its connection, and with it the table the
/// text index built for the longest text so far. Later adds on the same open store
/// count that text in their own share, until the store is opened anew.
#[test]
fn a_text_the_open_store_still_holds_a_table_for_counts_in_later_adds() {
    let scratch = Scratch::new("still-held");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    assert_eq!(store.retained_bytes(), IDLE_BYTES);

    const EARLIER: u64 = 10_000;
    const { assert!(EARLIER <= RELEASE_TEXT_BYTES) };
    let first = logs_of(&[
        record("GET /first", Label::Good, None, 1),
        record(&"abcdefghij".repeat(1_000), Label::Bad, None, 1),
    ]);
    let report = store
        .add_with(&rows(&first, "a.log", 1), &identity(), &mut budget())
        .unwrap();
    assert_eq!(report.longest_text_bytes, EARLIER);
    assert_eq!(
        report.held_bytes,
        ADD_FIXED_BYTES + ADD_TEXT_FACTOR * EARLIER
    );
    assert_eq!(store.retained_bytes(), IDLE_BYTES);

    // Short texts only, on the same open store: the earlier text still counts.
    let short = three_logs();
    let own = ADD_FIXED_BYTES + ADD_TEXT_FACTOR * EARLIER;
    let report = store
        .add_with(&rows(&short, "b.log", 1), &identity(), &mut budget())
        .unwrap();
    assert_eq!(report.longest_text_bytes, longest_text(&short));
    assert_eq!(report.held_bytes, own);

    // And it is named when the share does not fit.
    let mut tight = budget_with(Some(32 * MIB + own - 1), None);
    let error = store
        .add_with(&rows(&short, "c.log", 1), &identity(), &mut tight)
        .expect_err("one byte short");
    let StoreError::Memory(L2dError::Refused { reason }) = &error else {
        panic!("expected a memory refusal, got {error:?}");
    };
    assert!(reason.contains(&format!("needs {own} bytes")), "{reason}");
    assert!(
        reason.contains(&format!("{EARLIER} bytes, indexed by an earlier add")),
        "{reason}"
    );
    assert_eq!(store.len().unwrap(), 5);
    drop(store);

    // A store opened anew holds no such table.
    let mut store = Store::open(&paths, "s").unwrap();
    let report = store
        .add_with(&rows(&short, "c.log", 1), &identity(), &mut budget())
        .unwrap();
    assert_eq!(
        report.held_bytes,
        ADD_FIXED_BYTES + ADD_TEXT_FACTOR * longest_text(&short)
    );
}

/// Above `RELEASE_TEXT_BYTES` the add ends by opening the connection again, which is
/// the only way the text index gives its table back. The store goes on as before, and
/// later adds are not charged for the long text.
#[test]
fn after_a_text_above_the_release_threshold_the_store_goes_on_with_a_new_connection() {
    let scratch = Scratch::new("reopened");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity()).unwrap();
    let created_at = store.created_at().to_string();

    const LONG: u64 = 100_000;
    const { assert!(LONG > RELEASE_TEXT_BYTES) };
    let long = logs_of(&[
        record("GET /first", Label::Good, Some("/first"), 2),
        record(&"abcdefghij".repeat(10_000), Label::Bad, None, 1),
        record("GET /third", Label::Unknown, None, 3),
    ]);
    let report = store
        .add_with(&rows(&long, "a.log", 1), &identity(), &mut budget())
        .unwrap();
    assert_eq!((report.added, report.repeated), (3, 0));
    assert_eq!(report.held_bytes, ADD_FIXED_BYTES + ADD_TEXT_FACTOR * LONG);

    // The same store value, still usable in every way.
    assert_eq!(store.retained_bytes(), IDLE_BYTES);
    assert_eq!(store.len().unwrap(), 3);
    assert_eq!(store.identity(), &identity());
    assert_eq!(store.created_at(), created_at);
    assert_eq!(store.require_identity(&identity()), Ok(()));

    // A later add of short texts holds the fixed share and what its own texts need.
    let short = three_logs();
    let report = store
        .add_with(&rows(&short, "b.log", 1), &identity(), &mut budget())
        .unwrap();
    assert_eq!((report.added, report.repeated), (3, 0));
    assert_eq!(
        report.held_bytes,
        ADD_FIXED_BYTES + ADD_TEXT_FACTOR * longest_text(&short)
    );
    // Repeated ids are still found: the long text again, after the connection changed.
    let report = store
        .add_with(&rows(&long, "a.log", 2), &identity(), &mut budget())
        .unwrap();
    assert_eq!((report.added, report.repeated), (0, 3));

    // An add that fails after the long text was indexed releases as well, and the store
    // is as it was.
    let before = dump(&paths.store_path("s"));
    let spoiled = logs_of(&[
        record(&"klmnopqrst".repeat(10_000), Label::Bad, None, 1),
        record("GET /zero", Label::Good, None, 0),
    ]);
    let detail = invalid_rows(
        store
            .add_with(&rows(&spoiled, "c.log", 1), &identity(), &mut budget())
            .unwrap_err(),
    );
    assert!(detail.contains("row 2"), "{detail}");
    assert_eq!(store.retained_bytes(), IDLE_BYTES);
    assert_eq!(dump(&paths.store_path("s")), before);
    let report = store
        .add_with(&rows(&short, "d.log", 1), &identity(), &mut budget())
        .unwrap();
    assert_eq!(
        report.held_bytes,
        ADD_FIXED_BYTES + ADD_TEXT_FACTOR * longest_text(&short)
    );
    drop(store);

    let stored = dump(&paths.store_path("s"));
    assert_eq!(stored.len(), 9);
    assert_eq!(stored[..6], before[..]);
    // The occurrences of the long frame were added once more: 2 + 2, 1 + 1, 3 + 3.
    assert_eq!(
        stored[..3]
            .iter()
            .map(|row| row.occurrences)
            .collect::<Vec<_>>(),
        [4, 2, 6]
    );
    assert_eq!(
        fts_ids(&paths.store_path("s"), "hijabcdefghija"),
        [log_id("a.log", &"abcdefghij".repeat(10_000))]
    );
    assert!(fts_ids(&paths.store_path("s"), "klmnopqrst").is_empty());
    assert_fts_consistent(&paths.store_path("s"));
    assert_eq!(scratch.store_files(), ["s.db"]);
}
