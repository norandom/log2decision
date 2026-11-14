//! The database file of a store: its schema, how it is created and how it is checked
//! when it is opened.
//!
//! # Schema (format 2)
//!
//! `meta`, exactly one row:
//!
//! | Column | Type | Content |
//! |--------|------|---------|
//! | `id` | INTEGER | always 1 |
//! | `encoder_name`, `encoder_revision`, `encoder_dimension` | TEXT, TEXT, INTEGER | the one encoder identity of the store |
//! | `created_at` | TEXT | UTC, `YYYY-MM-DDTHH:MM:SSZ` |
//! | `tool_version` | TEXT | the version of the tool that created the store |
//! | `runtime_release` | TEXT, nullable | the release of the AMD runtime the vectors were made under; null for vectors that were not made by it through this tool |
//!
//! `logs`, one row per stored log:
//!
//! | Column | Type | Content |
//! |--------|------|---------|
//! | `seq` | INTEGER, primary key | order of insertion, from 1; the row id of the text index |
//! | `id` | TEXT, unique | [`crate::log_id`] of source and text |
//! | `text` | TEXT | the log text |
//! | `source` | TEXT, indexed | where the log came from |
//! | `occurrences` | INTEGER | 1 to 4,294,967,295 |
//! | `label` | TEXT, indexed | `good`, `bad` or `unknown` |
//! | `group_key` | TEXT, nullable | the group key |
//! | `truncated` | INTEGER | 0 or 1: the part of the text the encoder reads was cut |
//! | `text_longer_than_input` | INTEGER | 0 or 1: the whole text is longer than the encoder's input |
//! | `empty` | INTEGER | 0 or 1: no byte other than padding reached the model |
//! | `vector` | BLOB | `dimension` 32-bit floats, little-endian: `4 * dimension` bytes |
//! | `encoder_revision` | TEXT | the revision of the store's encoder |
//!
//! Both tables are `STRICT`, and the value rules above are `CHECK` constraints, the
//! vector length included: the file refuses a row that breaks them, whatever writes it.
//!
//! `logs_fts` is an FTS5 table over `logs.text` (`content = 'logs'`, so the text is
//! stored once) with the `trigram` tokenizer: a phrase of three or more characters
//! matches the texts that contain it, ignoring case, whatever characters it holds.
//! Triggers on `logs` keep it in step.
//!
//! `logs_text_bytes` is an index over `octet_length(text)`: the bytes of the longest
//! stored text are one look-up ([`crate::Store::longest_text_bytes`]), which every
//! search for a phrase needs to size its share of the memory limit. It was added with
//! task 1.9 without a new format number: a file without it is a valid store (the
//! look-up reads every row's length then), it gets the index with its next add, and
//! a version of the tool that does not know the index reads and writes a file that
//! has it, since SQLite keeps it up to date.
//!
//! The file carries the application id [`APPLICATION_ID`] and its format number
//! ([`SCHEMA_VERSION`]) in `PRAGMA user_version`.
//!
//! # Format 1
//!
//! Format 2 (task 2.7) added the column `runtime_release` to `meta` and the columns
//! `text_longer_than_input` and `empty` to `logs`, and nothing else. A file that says it
//! is of format 2 and lacks one of the three is not opened (the two flag columns were
//! added to the format before any store of real vectors was written in it, so there is
//! no third format for them).
//!
//! A file of format 1 ([`OLDEST_SCHEMA_VERSION`]), written before the release was
//! recorded, is still opened and stays usable as it was built (owner decision of
//! 2026-11-14): nothing is written to it when it is opened, it is searched by phrase,
//! and it takes and compares vectors without a release as before. It refuses vectors
//! made under any release, saying that it is of format 1, because nothing says under
//! which release, if any, its own were made. It is not migrated: it keeps its format
//! number and its columns. Its logs have the flag `truncated` only: the two other flags
//! are read back as absent values, which means unknown and not false, and those two
//! flags of rows added to it are not kept. A version of the tool that knows format 1
//! only does not open a file of format 2, so it cannot add vectors to one without the
//! check of the release.

use std::path::Path;
use std::time::Duration;

use l2d_core::EncoderIdentity;
use rusqlite::{Connection, OpenFlags};

/// The format of the database file a store is created with.
pub const SCHEMA_VERSION: u32 = 2;

/// The oldest format that is still opened (see the module description, "Format 1").
/// A file of a format outside these two is not opened.
pub const OLDEST_SCHEMA_VERSION: u32 = 1;

/// The longest runtime release a store records, in bytes.
pub const MAX_RELEASE_BYTES: usize = 64;

/// Marks a database file as a store of this tool (`PRAGMA application_id`): "L2DS".
pub const APPLICATION_ID: i32 = 0x4C32_4453;

/// The page cache a store's connection may fill (`PRAGMA cache_size`).
pub const CACHE_BYTES: u64 = 8 * 1024 * 1024;

/// How long a statement waits for another process that has the file locked.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Creates the index over the lengths of the texts, unless it is there.
pub(crate) const TEXT_BYTES_INDEX_SQL: &str =
    "CREATE INDEX IF NOT EXISTS logs_text_bytes ON logs (octet_length(text))";

/// The bytes of the longest stored text, 0 without a log. The expression is the one of
/// the index `logs_text_bytes`, so that SQLite answers from the index when it is there.
pub(crate) const LONGEST_TEXT_SQL: &str = "SELECT ifnull(max(octet_length(text)), 0) FROM logs";

/// What the `meta` table says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Meta {
    pub identity: EncoderIdentity,
    pub created_at: String,
    pub tool_version: String,
    /// The release of the AMD runtime the vectors were made under, if one is recorded.
    pub runtime_release: Option<String>,
}

/// The columns of `logs` that format 2 added, in the order they are written and read.
pub(crate) const FLAG_COLUMNS: [&str; 2] = ["text_longer_than_input", "empty"];

/// The statements that make an empty database a store for vectors of `dimension`.
fn schema_sql(dimension: u16) -> String {
    let vector_bytes = 4 * u32::from(dimension);
    format!(
        "CREATE TABLE meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            encoder_name TEXT NOT NULL,
            encoder_revision TEXT NOT NULL,
            encoder_dimension INTEGER NOT NULL CHECK (encoder_dimension BETWEEN 1 AND 65535),
            created_at TEXT NOT NULL,
            tool_version TEXT NOT NULL,
            runtime_release TEXT CHECK (runtime_release IS NULL OR
                length(runtime_release) BETWEEN 1 AND {MAX_RELEASE_BYTES})
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
            text_longer_than_input INTEGER NOT NULL
                CHECK (text_longer_than_input IN (0, 1)),
            empty INTEGER NOT NULL CHECK (empty IN (0, 1)),
            vector BLOB NOT NULL CHECK (length(vector) = {vector_bytes}),
            encoder_revision TEXT NOT NULL
        ) STRICT;
        CREATE INDEX logs_label ON logs (label);
        CREATE INDEX logs_source ON logs (source);
        {TEXT_BYTES_INDEX_SQL};
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
        PRAGMA application_id = {APPLICATION_ID};
        PRAGMA user_version = {SCHEMA_VERSION};"
    )
}

/// Opens the existing file `path` for reading and writing; never creates one.
fn connect(path: &Path) -> rusqlite::Result<Connection> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(BUSY_TIMEOUT)?;
    connection.pragma_update(None, "cache_size", -((CACHE_BYTES / 1024) as i64))?;
    Ok(connection)
}

/// Makes the empty file `path` a store for `identity` and `runtime_release`, in one
/// transaction: after a failure or an interruption the file holds no part of a store.
pub(crate) fn create(
    path: &Path,
    identity: &EncoderIdentity,
    runtime_release: Option<&str>,
    tool_version: &str,
) -> Result<(Connection, Meta, u32), String> {
    let run = || -> rusqlite::Result<(Connection, Meta, u32)> {
        let mut connection = connect(path)?;
        let transaction = connection.transaction()?;
        transaction.execute_batch(&schema_sql(identity.dimension))?;
        transaction.execute(
            "INSERT INTO meta (id, encoder_name, encoder_revision, encoder_dimension, \
             created_at, tool_version, runtime_release) \
             VALUES (1, ?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?4, ?5)",
            rusqlite::params![
                identity.name,
                identity.revision,
                identity.dimension,
                tool_version,
                runtime_release
            ],
        )?;
        transaction.commit()?;
        let meta = read_meta(&connection)?;
        Ok((connection, meta, SCHEMA_VERSION))
    };
    run().map_err(|error| error.to_string())
}

/// Opens the store file `path` after checking that it is one, of a format this version
/// reads and writes; the format is returned beside what `meta` says.
pub(crate) fn open(path: &Path) -> Result<(Connection, Meta, u32), String> {
    let connection = connect(path).map_err(|error| error.to_string())?;
    let pragma = |name: &str| -> Result<i64, String> {
        connection
            .query_row(&format!("PRAGMA {name}"), [], |row| row.get(0))
            .map_err(|error| error.to_string())
    };
    let application = pragma("application_id")?;
    if application != i64::from(APPLICATION_ID) {
        return Err(format!(
            "it is a database, but not a store of this tool (application id {application}, \
             expected {APPLICATION_ID})"
        ));
    }
    let version = pragma("user_version")?;
    if version < i64::from(OLDEST_SCHEMA_VERSION) || version > i64::from(SCHEMA_VERSION) {
        return Err(format!(
            "it has the store format {version}; this version of the tool reads and writes \
             the formats {OLDEST_SCHEMA_VERSION} to {SCHEMA_VERSION} only"
        ));
    }
    let tables: i64 = connection
        .query_row(
            "SELECT count(*) FROM sqlite_master \
             WHERE type = 'table' AND name IN ('meta', 'logs', 'logs_fts')",
            [],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if tables != 3 {
        return Err(format!(
            "{tables} of the tables meta, logs and logs_fts are there, not all three"
        ));
    }
    // The columns format 2 added are there in format 2 and in no older format.
    let newer = version == i64::from(SCHEMA_VERSION);
    let added = [("meta", "runtime_release")]
        .into_iter()
        .chain(FLAG_COLUMNS.map(|column| ("logs", column)));
    for (table, column) in added {
        let has = has_column(&connection, table, column).map_err(|error| error.to_string())?;
        if has != newer {
            return Err(format!(
                "it has the store format {version}, but its {table} table has {} column \
                 {column}; that column is what format {SCHEMA_VERSION} added",
                if has { "the" } else { "no" }
            ));
        }
    }
    let meta = read_meta(&connection).map_err(|error| format!("reading meta: {error}"))?;
    // The range was checked above.
    Ok((connection, meta, version as u32))
}

/// Whether `table` has the column `column`.
fn has_column(connection: &Connection, table: &str, column: &str) -> rusqlite::Result<bool> {
    connection.query_row(
        "SELECT count(*) FROM pragma_table_info(?1) WHERE name = ?2",
        [table, column],
        |row| row.get::<_, i64>(0).map(|columns| columns > 0),
    )
}

/// What the file says about the store now. A file of format 1 has no release.
pub(crate) fn read_meta(connection: &Connection) -> rusqlite::Result<Meta> {
    let release = if has_column(connection, "meta", "runtime_release")? {
        "runtime_release"
    } else {
        "NULL"
    };
    connection.query_row(
        &format!(
            "SELECT encoder_name, encoder_revision, encoder_dimension, created_at, \
             tool_version, {release} FROM meta WHERE id = 1"
        ),
        [],
        |row| {
            Ok(Meta {
                identity: EncoderIdentity {
                    name: row.get(0)?,
                    revision: row.get(1)?,
                    dimension: row.get(2)?,
                },
                created_at: row.get(3)?,
                tool_version: row.get(4)?,
                runtime_release: row.get(5)?,
            })
        },
    )
}
