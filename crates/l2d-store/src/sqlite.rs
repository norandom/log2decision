//! The database file of a store: its schema, how it is created and how it is checked
//! when it is opened.
//!
//! # Schema (format 1)
//!
//! `meta`, exactly one row:
//!
//! | Column | Type | Content |
//! |--------|------|---------|
//! | `id` | INTEGER | always 1 |
//! | `encoder_name`, `encoder_revision`, `encoder_dimension` | TEXT, TEXT, INTEGER | the one encoder identity of the store |
//! | `created_at` | TEXT | UTC, `YYYY-MM-DDTHH:MM:SSZ` |
//! | `tool_version` | TEXT | the version of the tool that created the store |
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
//! | `truncated` | INTEGER | 0 or 1 |
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
//! The file carries the application id [`APPLICATION_ID`] and its format number
//! ([`SCHEMA_VERSION`]) in `PRAGMA user_version`.

use std::path::Path;
use std::time::Duration;

use l2d_core::EncoderIdentity;
use rusqlite::{Connection, OpenFlags};

/// The format of the database file. A file of another format is not opened.
pub const SCHEMA_VERSION: u32 = 1;

/// Marks a database file as a store of this tool (`PRAGMA application_id`): "L2DS".
pub const APPLICATION_ID: i32 = 0x4C32_4453;

/// The page cache a store's connection may fill (`PRAGMA cache_size`).
pub const CACHE_BYTES: u64 = 8 * 1024 * 1024;

/// How long a statement waits for another process that has the file locked.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// What the `meta` table says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Meta {
    pub identity: EncoderIdentity,
    pub created_at: String,
    pub tool_version: String,
}

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
            vector BLOB NOT NULL CHECK (length(vector) = {vector_bytes}),
            encoder_revision TEXT NOT NULL
        ) STRICT;
        CREATE INDEX logs_label ON logs (label);
        CREATE INDEX logs_source ON logs (source);
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

/// Makes the empty file `path` a store for `identity`, in one transaction: after a
/// failure or an interruption the file holds no part of a store.
pub(crate) fn create(
    path: &Path,
    identity: &EncoderIdentity,
    tool_version: &str,
) -> Result<(Connection, Meta), String> {
    let run = || -> rusqlite::Result<(Connection, Meta)> {
        let mut connection = connect(path)?;
        let transaction = connection.transaction()?;
        transaction.execute_batch(&schema_sql(identity.dimension))?;
        transaction.execute(
            "INSERT INTO meta (id, encoder_name, encoder_revision, encoder_dimension, \
             created_at, tool_version) \
             VALUES (1, ?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?4)",
            rusqlite::params![
                identity.name,
                identity.revision,
                identity.dimension,
                tool_version
            ],
        )?;
        transaction.commit()?;
        let meta = read_meta(&connection)?;
        Ok((connection, meta))
    };
    run().map_err(|error| error.to_string())
}

/// Opens the store file `path` after checking that it is one, of this format.
pub(crate) fn open(path: &Path) -> Result<(Connection, Meta), String> {
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
    if version != i64::from(SCHEMA_VERSION) {
        return Err(format!(
            "it has the store format {version}; this version of the tool reads and writes \
             the format {SCHEMA_VERSION} only"
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
    let meta = read_meta(&connection).map_err(|error| format!("reading meta: {error}"))?;
    Ok((connection, meta))
}

fn read_meta(connection: &Connection) -> rusqlite::Result<Meta> {
    connection.query_row(
        "SELECT encoder_name, encoder_revision, encoder_dimension, created_at, tool_version \
         FROM meta WHERE id = 1",
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
            })
        },
    )
}
