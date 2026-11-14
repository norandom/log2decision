//! The AMD runtime release of a store's vectors (task 2.7; owner decision of 2026-11-13).
//!
//! The provider compiles the model per release of the AMD runtime and the vectors differ
//! between releases, so a store records the release beside the encoder identity when it
//! is created and refuses vectors made under another one, to store and to compare, naming
//! both. Vectors that were not made by the AMD runtime through this tool (the synthetic
//! vectors of the other test files) have no release; they and released vectors are kept
//! apart in both directions.
//!
//! The vectors here are synthetic and of dimension 3: nothing depends on an encoder.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_core::polars::prelude::DataFrame;
use l2d_core::{EncoderIdentity, L2dError, Label, LogRecord, LogsFrame, ProjectPaths};
use l2d_store::{Filter, SCHEMA_VERSION, Store, StoreError, VectorOrigin, rows_frame};
use rusqlite::{Connection, OpenFlags};

const DIM: u16 = 3;
const SOURCE: &str = "release-test";
const RELEASE: &str = "1.9.0";
const OTHER_RELEASE: &str = "1.10.0";

fn identity() -> EncoderIdentity {
    EncoderIdentity {
        name: "any-encoder".to_string(),
        revision: "00112233445566778899aabbccddeeff00112233".to_string(),
        dimension: DIM,
    }
}

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-store-release-{}-{}-{name}",
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

/// `count` logs with a vector each, the first text numbered `from`.
fn rows(from: u32, count: u32) -> DataFrame {
    let records: Vec<LogRecord> = (from..from + count)
        .map(|index| LogRecord {
            text: format!("GET /item?id={index}"),
            label: Label::Good,
            group: None,
            first_position: u64::from(index) + 1,
            occurrences: 1,
        })
        .collect();
    let logs = LogsFrame::from_records(&records).unwrap();
    let vectors: Vec<f32> = (from..from + count)
        .flat_map(|index| [index as f32, 1.0, -0.5])
        .collect();
    rows_frame(&logs, SOURCE, &vectors, DIM, &vec![false; count as usize]).unwrap()
}

fn released<'a>(identity: &'a EncoderIdentity, release: &'a str) -> VectorOrigin<'a> {
    VectorOrigin::made_under(identity, release)
}

fn file(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("the store file is readable")
}

fn read_only(path: &Path) -> Connection {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap()
}

fn mismatch(error: StoreError) -> (Option<String>, Option<String>) {
    match error {
        StoreError::ReleaseMismatch { store, given } => (store, given),
        other => panic!("expected a release mismatch, got: {other}"),
    }
}

#[test]
fn a_store_records_the_runtime_release_beside_the_encoder_identity() {
    let scratch = Scratch::new("records");
    let paths = scratch.paths();
    let id = identity();
    let store = Store::create(&paths, "s", released(&id, RELEASE)).unwrap();
    assert_eq!(store.runtime_release(), Some(RELEASE));
    assert_eq!(store.identity(), &id);
    drop(store);

    let connection = read_only(&paths.store_path("s"));
    let (name, release): (String, Option<String>) = connection
        .query_row(
            "SELECT encoder_name, runtime_release FROM meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("meta has the release beside the identity");
    assert_eq!(name, "any-encoder");
    assert_eq!(release.as_deref(), Some(RELEASE));
    let format: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(format, i64::from(SCHEMA_VERSION));
    assert_eq!(SCHEMA_VERSION, 2, "the release came with format 2");
    drop(connection);

    let again = Store::open(&paths, "s").unwrap();
    assert_eq!(again.runtime_release(), Some(RELEASE));
    assert_eq!(again.identity(), &id);
}

#[test]
fn a_store_created_from_an_identity_alone_records_no_release() {
    let scratch = Scratch::new("none");
    let paths = scratch.paths();
    let store = Store::create(&paths, "s", &identity()).unwrap();
    assert_eq!(store.runtime_release(), None);
    drop(store);
    let release: Option<String> = read_only(&paths.store_path("s"))
        .query_row("SELECT runtime_release FROM meta", [], |row| row.get(0))
        .unwrap();
    assert_eq!(release, None);
    assert_eq!(Store::open(&paths, "s").unwrap().runtime_release(), None);
}

#[test]
fn vectors_of_the_stores_release_are_added() {
    let scratch = Scratch::new("same");
    let id = identity();
    let mut store = Store::create(&scratch.paths(), "s", released(&id, RELEASE)).unwrap();
    let report = store.add(&rows(0, 5), released(&id, RELEASE)).unwrap();
    assert_eq!((report.added, report.repeated), (5, 0));
    let report = store.add(&rows(3, 4), released(&id, RELEASE)).unwrap();
    assert_eq!((report.added, report.repeated), (2, 2));
    assert_eq!(store.len().unwrap(), 7);
}

#[test]
fn vectors_made_under_another_release_are_refused_and_both_are_named() {
    let scratch = Scratch::new("other");
    let paths = scratch.paths();
    let id = identity();
    let mut store = Store::create(&paths, "s", released(&id, RELEASE)).unwrap();
    store.add(&rows(0, 5), released(&id, RELEASE)).unwrap();
    let before = file(&paths.store_path("s"));

    let error = store
        .add(&rows(5, 5), released(&id, OTHER_RELEASE))
        .expect_err("another release is refused");
    let message = error.to_string();
    assert!(
        message.contains(RELEASE) && message.contains(OTHER_RELEASE),
        "{message}"
    );
    assert!(message.contains("again"), "the way out is named: {message}");
    assert_eq!(
        mismatch(error.clone()),
        (Some(RELEASE.to_string()), Some(OTHER_RELEASE.to_string()))
    );
    assert!(
        matches!(L2dError::from(error), L2dError::Refused { .. }),
        "a refusal for the exit code"
    );
    assert_eq!(store.len().unwrap(), 5);
    assert_eq!(file(&paths.store_path("s")), before, "nothing was written");

    // The same check stands in front of the budget form and of a require.
    let error = store
        .require_origin(released(&id, OTHER_RELEASE))
        .expect_err("another release");
    assert!(matches!(error, StoreError::ReleaseMismatch { .. }));
    store.require_origin(released(&id, RELEASE)).unwrap();
}

#[test]
fn vectors_without_a_release_and_vectors_with_one_are_kept_apart() {
    let scratch = Scratch::new("apart");
    let paths = scratch.paths();
    let id = identity();

    let mut with = Store::create(&paths, "with", released(&id, RELEASE)).unwrap();
    let error = with
        .add(&rows(0, 2), &id)
        .expect_err("vectors of no known release into a store of a release");
    assert_eq!(mismatch(error.clone()), (Some(RELEASE.to_string()), None));
    assert!(error.to_string().contains(RELEASE), "{error}");
    assert_eq!(with.len().unwrap(), 0);

    let mut without = Store::create(&paths, "without", &id).unwrap();
    without.add(&rows(0, 2), &id).unwrap();
    let error = without
        .add(&rows(2, 2), released(&id, RELEASE))
        .expect_err("vectors of a release into a store without one");
    assert_eq!(mismatch(error), (None, Some(RELEASE.to_string())));
    assert_eq!(without.len().unwrap(), 2);
}

#[test]
fn the_encoder_identity_is_checked_before_the_release() {
    let scratch = Scratch::new("order");
    let id = identity();
    let mut store = Store::create(&scratch.paths(), "s", released(&id, RELEASE)).unwrap();
    let other = EncoderIdentity {
        dimension: DIM + 1,
        ..identity()
    };
    let error = store
        .add(&rows(0, 2), released(&other, OTHER_RELEASE))
        .expect_err("another encoder");
    assert!(
        matches!(error, StoreError::IdentityMismatch { .. }),
        "{error}"
    );
}

#[test]
fn query_vectors_of_another_release_are_not_compared() {
    let scratch = Scratch::new("nearest");
    let id = identity();
    let mut store = Store::create(&scratch.paths(), "s", released(&id, RELEASE)).unwrap();
    store.add(&rows(0, 5), released(&id, RELEASE)).unwrap();
    let query = [2.0f32, 1.0, -0.5];

    let near = store
        .nearest(&query, released(&id, RELEASE), 1, &Filter::default())
        .unwrap();
    assert_eq!(near.hits.height(), 1);

    let error = store
        .nearest(&query, released(&id, OTHER_RELEASE), 1, &Filter::default())
        .expect_err("a query vector of another release");
    assert_eq!(
        mismatch(error),
        (Some(RELEASE.to_string()), Some(OTHER_RELEASE.to_string()))
    );
    let error = store
        .nearest(&query, &id, 1, &Filter::default())
        .expect_err("a query vector of no known release");
    assert_eq!(mismatch(error), (Some(RELEASE.to_string()), None));
}

#[test]
fn a_release_that_cannot_be_one_is_refused_at_creation() {
    let scratch = Scratch::new("empty");
    let paths = scratch.paths();
    let id = identity();
    for unusable in ["", " ", "1.9.0\n", "1.9\0"] {
        let error =
            Store::create(&paths, "s", released(&id, unusable)).expect_err("an unusable release");
        assert!(
            matches!(error, StoreError::InvalidIdentity { .. }),
            "{unusable:?}: {error}"
        );
        assert!(!paths.store_path("s").exists(), "no file was left");
    }
}

#[test]
fn a_release_changed_under_an_open_store_stops_its_operations() {
    let scratch = Scratch::new("changed");
    let paths = scratch.paths();
    let id = identity();
    let mut store = Store::create(&paths, "s", released(&id, RELEASE)).unwrap();
    store.add(&rows(0, 3), released(&id, RELEASE)).unwrap();
    Connection::open(paths.store_path("s"))
        .unwrap()
        .execute("UPDATE meta SET runtime_release = ?1", [OTHER_RELEASE])
        .unwrap();

    let error = store
        .add(&rows(3, 3), released(&id, RELEASE))
        .expect_err("the file no longer describes the open store");
    let message = error.to_string();
    assert!(matches!(error, StoreError::Failed { .. }), "{message}");
    assert!(
        message.contains(RELEASE) && message.contains(OTHER_RELEASE),
        "{message}"
    );
    assert_eq!(store.len().unwrap(), 3);
}

/// The statements task 1.8 and 1.9 created a store with (format 1), as they were before
/// the release was recorded.
fn create_format_1(path: &Path, identity: &EncoderIdentity) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let connection = Connection::open(path).unwrap();
    connection
        .execute_batch(&format!(
            "CREATE TABLE meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                encoder_name TEXT NOT NULL,
                encoder_revision TEXT NOT NULL,
                encoder_dimension INTEGER NOT NULL CHECK (encoder_dimension BETWEEN 1 AND 65535),
                created_at TEXT NOT NULL,
                tool_version TEXT NOT NULL
            ) STRICT;
            CREATE TABLE logs (
                seq INTEGER PRIMARY KEY,
                id TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                occurrences INTEGER NOT NULL CHECK (occurrences BETWEEN 1 AND 4294967295),
                label TEXT NOT NULL CHECK (label IN ('good', 'bad', 'unknown')),
                group_key TEXT,
                truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
                vector BLOB NOT NULL CHECK (length(vector) = {}),
                encoder_revision TEXT NOT NULL
            ) STRICT;
            CREATE INDEX logs_label ON logs (label);
            CREATE INDEX logs_source ON logs (source);
            CREATE INDEX IF NOT EXISTS logs_text_bytes ON logs (octet_length(text));
            CREATE VIRTUAL TABLE logs_fts USING fts5(
                text, content = 'logs', content_rowid = 'seq', tokenize = 'trigram'
            );
            CREATE TRIGGER logs_fts_insert AFTER INSERT ON logs BEGIN
                INSERT INTO logs_fts (rowid, text) VALUES (new.seq, new.text);
            END;
            CREATE TRIGGER logs_fts_delete AFTER DELETE ON logs BEGIN
                INSERT INTO logs_fts (logs_fts, rowid, text) VALUES ('delete', old.seq, old.text);
            END;
            CREATE TRIGGER logs_fts_update AFTER UPDATE OF seq, text ON logs BEGIN
                INSERT INTO logs_fts (logs_fts, rowid, text) VALUES ('delete', old.seq, old.text);
                INSERT INTO logs_fts (rowid, text) VALUES (new.seq, new.text);
            END;
            PRAGMA application_id = {};
            PRAGMA user_version = 1;",
            4 * u32::from(identity.dimension),
            l2d_store::APPLICATION_ID
        ))
        .unwrap();
    connection
        .execute(
            "INSERT INTO meta (id, encoder_name, encoder_revision, encoder_dimension, \
             created_at, tool_version) VALUES (1, ?1, ?2, ?3, '2026-11-12T09:30:00Z', '0.1.0')",
            rusqlite::params![identity.name, identity.revision, identity.dimension],
        )
        .unwrap();
}

#[test]
fn a_store_written_before_the_release_was_recorded_is_a_store_without_a_release() {
    let scratch = Scratch::new("format1");
    let paths = scratch.paths();
    let id = identity();
    let path = paths.store_path("old");
    create_format_1(&path, &id);

    let mut store = Store::open(&paths, "old").expect("a format-1 store still opens");
    assert_eq!(store.runtime_release(), None);
    assert_eq!(store.identity(), &id);

    // What it could do before, it can do now: vectors of no known release.
    let report = store.add(&rows(0, 4), &id).unwrap();
    assert_eq!(report.added, 4);
    let near = store
        .nearest(&[1.0, 1.0, -0.5], &id, 2, &Filter::default())
        .unwrap();
    assert_eq!(near.hits.height(), 2);

    // Vectors of a release are refused: the file has no place for the release, and what
    // it holds was made under none that is known.
    let before = file(&path);
    let error = store
        .add(&rows(4, 4), released(&id, RELEASE))
        .expect_err("a format-1 store takes no vectors of a release");
    let message = error.to_string();
    assert_eq!(mismatch(error), (None, Some(RELEASE.to_string())));
    assert!(message.contains("again"), "{message}");
    assert_eq!(file(&path), before);
    drop(store);

    // The file is not migrated: it keeps its format and its columns.
    let connection = read_only(&path);
    let format: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(format, 1);
    let columns: i64 = connection
        .query_row(
            "SELECT count(*) FROM pragma_table_info('meta') WHERE name = 'runtime_release'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(columns, 0);
}

#[test]
fn a_file_of_format_2_without_the_release_column_is_not_opened() {
    let scratch = Scratch::new("format2-broken");
    let paths = scratch.paths();
    let path = paths.store_path("s");
    create_format_1(&path, &identity());
    Connection::open(&path)
        .unwrap()
        .pragma_update(None, "user_version", 2)
        .unwrap();
    let error = Store::open(&paths, "s").expect_err("format 2 has the release column");
    assert!(matches!(error, StoreError::Failed { .. }), "{error}");
    assert!(error.to_string().contains("runtime_release"), "{error}");
}
