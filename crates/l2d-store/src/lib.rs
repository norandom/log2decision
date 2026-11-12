//! The local store: logs with their labels and vectors in one SQLite database file,
//! `stores/<name>.db`, and its searches (requirements 5.1 to 5.7; tasks 1.8 and 1.9).
//!
//! A store holds the vectors of exactly one encoder identity, written to its `meta`
//! table when it is created. [`Store::add`] takes the identity of the encoder that
//! produced the vectors it is given and refuses another one, naming both
//! ([`StoreError::IdentityMismatch`]).
//!
//! Everything is synchronous and in this process: SQLite is compiled in, there is no
//! background service and no asynchronous runtime. At rest a store is its one file.
//!
//! # Rows
//!
//! Logs are handed to the store as a Polars frame, the store rows frame
//! ([`rows_schema`], built from a logs frame by [`rows_frame`]):
//!
//! | Column | Type | Null | Content |
//! |--------|------|------|---------|
//! | `text` | String | no | the log text |
//! | `source` | String | no | where the log came from |
//! | `occurrences` | UInt32 | no | how often the text occurred, at least 1 |
//! | `label` | String | no | `good`, `bad` or `unknown` |
//! | `group` | String | yes | the group key |
//! | `truncated` | Boolean | no | whether the encoder saw only the start of the text |
//! | `vector` | List(Float32) | no | the vector, exactly `dimension` finite values |
//!
//! The store derives the identity of a log itself: [`log_id`], the SHA-256 of source
//! and text. Adding a log whose id is stored already adds its occurrences to the stored
//! row and leaves text, label, group, flag and vector of that row as they are. The
//! encoder revision of a row is that of the store.
//!
//! An add is all or nothing: it is one database transaction, and a row that cannot be
//! stored, a refusal or a failure part-way leaves the store as it was.
//!
//! The schema of the database file is described in [`sqlite`].
//!
//! # Memory (requirement 12)
//!
//! [`Store::add_with`] walks the frame in batches sized by the caller's
//! [`MemoryBudget`] and reads the free memory again before each one. The Rust side
//! keeps nothing per row: a row is bound to the insert statement straight from the
//! frame's buffers. SQLite's side needs two things beside the caller's frame, and the
//! add holds both of the budget for as long as it runs (its own share,
//! [`AddReport::held_bytes`]):
//!
//! - the memory of the database connection, whatever the rows are
//!   ([`ADD_FIXED_BYTES`]);
//! - the memory the text index takes to index one text, which grows with the length of
//!   that text: [`ADD_TEXT_FACTOR`] bytes for every byte of the longest text of the
//!   frame. It is taken from the longest text of the whole frame, not of each batch,
//!   so that an add that cannot fit is refused before anything is written. A text
//!   longer than [`MAX_TEXT_BYTES`] is not stored at all.
//!
//! An add ends, whatever its outcome, by giving SQLite's memory back, so that an open
//! store keeps no more than [`IDLE_BYTES`] between adds ([`Store::retained_bytes`]):
//! the page cache is released, and when a text longer than [`RELEASE_TEXT_BYTES`] was
//! indexed the connection is closed and opened again, because the text index keeps a
//! table sized for the longest text it has indexed for as long as its connection
//! lives. Up to that length the table stays, and later adds on the same open store
//! count that earlier text in their own share.
//!
//! For each batch the budget is also asked for one further copy of its rows
//! ([`ADD_ROW_BYTES`] more per row), which is above what was measured. The frame itself
//! is the caller's and is not counted here: a caller that holds a frame passes the
//! budget it holds it under ([`Store::add`] does not see it), and a caller with more
//! logs than its budget allows in one frame adds them frame by frame.
//!
//! # Searching and reading rows back
//!
//! [`query`] holds the search by text, label and source and the reading of rows
//! ([`Store::text_search`], [`Store::vectors`], [`Filter`]); [`vector`] holds the exact
//! search by vector ([`Store::nearest`]), which refuses a query vector of another
//! encoder identity as [`Store::add`] does. Rows come back as Polars frames too
//! ([`stored_schema`], [`hits_schema`]). Searches change nothing in the store, hold
//! their own share of the caller's memory budget while they run, and leave an open
//! store at the level [`Store::retained_bytes`] reports.

pub mod query;
pub mod sqlite;
pub mod vector;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use l2d_core::polars::prelude::{
    ChunkFull, Column, DataFrame, DataType, Field, Float32Type, IntoColumn, ListBuilderTrait,
    ListPrimitiveChunkedBuilder, PolarsError, Schema, StringChunked,
};
use l2d_core::{
    BatchPlan, EncoderIdentity, L2dError, Label, LogsFrame, MemoryBudget, MemorySettings,
    ProjectPaths,
};
use rusqlite::{Connection, Statement};
use sha2::{Digest, Sha256};

pub use query::{
    DISTANCE, Explain, Filter, ID, INDEX_PIECE_CHARS, INDEX_TEXT_FACTOR, MAX_INDEX_PIECES,
    Matching, PHRASE_BYTE_BYTES, QUERY, READ_FIXED_BYTES, READ_ROW_BYTES, SEARCH_FIXED_BYTES,
    TEXT_READ_FACTOR, fold, fold_char, hits_schema, index_folds_alike, stored_schema,
};
pub use sqlite::{APPLICATION_ID, CACHE_BYTES, SCHEMA_VERSION};
pub use vector::{NEAREST_ROW_BYTES, Nearest};

/// The longest store name, in bytes.
pub const MAX_NAME_BYTES: usize = 64;

/// Column of the rows frame: the log text.
pub const TEXT: &str = LogsFrame::TEXT;
/// Column of the rows frame: where the log came from.
pub const SOURCE: &str = "source";
/// Column of the rows frame: how often the text occurred.
pub const OCCURRENCES: &str = LogsFrame::OCCURRENCES;
/// Column of the rows frame: `good`, `bad` or `unknown`.
pub const LABEL: &str = LogsFrame::LABEL;
/// Column of the rows frame: the group key.
pub const GROUP: &str = LogsFrame::GROUP;
/// Column of the rows frame: whether the encoder saw only the start of the text.
pub const TRUNCATED: &str = "truncated";
/// Column of the rows frame: the vector.
pub const VECTOR: &str = "vector";

/// What an add holds of the memory limit for as long as it runs, beside the caller's
/// frame and whatever its texts are: the page cache of the connection
/// ([`CACHE_BYTES`]) and what SQLite needs on top while a transaction is written.
///
/// Measured on the reference host (`tests/store_csic.rs`): adding the 25,608 distinct
/// logs of HTTP CSIC 2010 (the longest text has 891 bytes) with 128-dimension
/// vectors in one transaction, which writes a file of 45.7 MiB, SQLite's own heap
/// (`sqlite3_memory_highwater`) peaked at 8.5 MiB. It does not grow with the number of
/// rows, because SQLite writes changed pages to the file once the cache is full. It
/// does grow with the length of the longest text: see [`ADD_TEXT_FACTOR`].
pub const ADD_FIXED_BYTES: u64 = 16 * 1024 * 1024;

/// What an add holds of the memory limit, on top of [`ADD_FIXED_BYTES`], for every byte
/// of the longest text of the frame.
///
/// The text index (FTS5 with the `trigram` tokenizer) builds the entry of one text
/// whole in memory before it writes it, and that takes memory for every trigram of
/// the text that it has not met in that text before, on top of SQLite's copies of the
/// text. What one text took is given back before the next, so the longest text
/// decides, not the sum and not the average. The worst case is a text of single-byte
/// characters in which no trigram repeats; a log line, which repeats most of its
/// trigrams, takes a twentieth of that. The factor is set for the worst case, because
/// the store does not know which kind a text is: for long ordinary texts it asks for
/// far more than is used.
///
/// Measured on the reference host in release mode (`long_texts_measured` in
/// `tests/store_csic.rs`): the peak of SQLite's heap (`sqlite3_memory_highwater`) when
/// one long text is added among 200 short ones, in bytes per byte of the long text.
///
/// | Text | 64 KiB | 256 KiB | 1 MiB | 8 MiB |
/// |------|--------|---------|-------|-------|
/// | request lines, as in a log | 15 | 8 | 7 | 6 |
/// | random printable ASCII (90 symbols) | 151 | 107 | 42 | 12 |
/// | random two-byte characters | 93 | 88 | 80 | 76 |
/// | ASCII counting upwards, few trigrams repeat below 1 MiB | 170 | 157 | 114 | 22 |
///
/// These round lengths are not the worst case. The index doubles its table of slots
/// when the distinct trigrams of a text pass a power of two, so the heap steps up
/// just above one. Texts in which no trigram repeats at all (single-byte characters,
/// from a De Bruijn sequence), just below and just above such a step:
///
/// | Trigrams | 2^16 - 64 | 2^16 + 64 | 2^18 - 64 | 2^18 + 64 | 2^19 + 64 | 1,030,301 | 2^20 + 64 |
/// |----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
/// | heap, bytes per byte of text | 171 | 186 | 168 | 184 | 181 | 153 | 177 |
/// | heap as a part of the share held | 0.38 | 0.42 | 0.66 | 0.72 | 0.81 | 0.73 | 0.85 |
///
/// So the worst case is about 186 bytes of heap per byte of text, which the factor
/// exceeds by 3 %; the rest of the margin is in [`ADD_FIXED_BYTES`], of which a filled
/// page cache leaves 7.5 MiB. The most that was measured of a share held is 0.85 (a
/// text of 1,066,985 bytes with 2^20 + 64 trigrams, in a small database); with a
/// filled page cache that would be about 0.89. There are 1,030,301 trigrams of
/// single-byte characters that differ after case folding, so a longer text cannot
/// stay at this rate: random ASCII levels off once its 729,000 possible trigrams are
/// used up, and text of multi-byte characters, which does not level off, has one
/// trigram for two bytes or more (607 MiB for 8 MiB of text). Sixteen texts of 1 MiB
/// take what one takes and at most 7 MiB more; eight texts without a repeat of
/// 2^18 + 64 trigrams each use 0.77 of the share against 0.72 for one. That test
/// asserts, for every length up to [`MAX_TEXT_BYTES`] of the four kinds above and for
/// the texts without a repeat, that the add's own share covers SQLite's heap. The
/// columns for 8 MiB and for 2^20 + 64 trigrams (a text of 1,066,985 bytes) were
/// measured before the limit came down to 1 MiB; such texts are refused now.
pub const ADD_TEXT_FACTOR: u64 = 192;

/// The longest text a store takes, in bytes. A longer one is refused
/// ([`StoreError::InvalidRows`], naming the row and this limit) before anything is
/// written.
///
/// The limit is 1 MiB (owner decision of 2026-11-12; it was 8 MiB before). What a store
/// holds of the memory limit grows with its longest text, in adds and in searches, and
/// at 8 MiB that was out of proportion: an add held `ADD_TEXT_FACTOR * 8 MiB` = 1.5 GiB,
/// and a search for a phrase had to hold tens of mebibytes per character it gave the
/// text index (see [`query`]). At 1 MiB an add holds at most 16 MiB + 192 MiB = 208
/// MiB. 1 MiB is also the default longest record of the input layer.
///
/// Up to this length the memory of the text index is measured for every kind of text
/// in the table at [`ADD_TEXT_FACTOR`], and checked again by every run of that test.
/// The figures given there and elsewhere for longer texts were measured while such
/// texts were still stored.
pub const MAX_TEXT_BYTES: u64 = 1024 * 1024;

/// The most of SQLite's memory an open store keeps between adds, as long as every add
/// could give back what it took (see [`Store::retained_bytes`]).
///
/// Three things are in it, measured on the reference host in release mode
/// (`retained_memory_measured` in `tests/store_csic.rs`): the connection itself with
/// its schema and the statements of the text index, 0.09 to 0.13 MiB; no page cache,
/// which is released at the end of every add; and the table of the text index for the
/// longest text indexed since the connection was opened, which is at most 32 bytes per
/// byte of that text and so at most 0.5 MiB for a text of [`RELEASE_TEXT_BYTES`].
pub const IDLE_BYTES: u64 = 1024 * 1024;

/// The longest text after which a store keeps its connection. When an add has indexed
/// a longer one, the connection is closed and opened again at the end of the add.
///
/// The text index (FTS5) keeps a hash table in memory for the text it is indexing. Its
/// entries are freed when the text is written, but its table of slots is not: it has
/// grown to two to four slots of 8 bytes for every distinct trigram of the longest text
/// so far, and SQLite keeps it at that size until the connection is closed. Nothing
/// else gives it back (measured: `PRAGMA shrink_memory` releases the page cache only,
/// and releasing prepared statements releases nothing). After one text of 8 MiB of
/// two-byte characters (stored before the limit of 1 MiB) that table was 64 MiB; after
/// a text of 1 MiB in which no trigram
/// repeats, 32 MiB. Opening the connection again takes 2 to 3 ms.
pub const RELEASE_TEXT_BYTES: u64 = 16 * 1024;

/// What the table of the text index can keep per byte of the longest text indexed on a
/// connection: four slots of 8 bytes for each trigram, one trigram per byte at most.
pub const RETAINED_TEXT_FACTOR: u64 = 32;

/// The fixed part of the memory the budget is asked for per row of a batch; the other
/// part is the size of the row in the frame.
pub const ADD_ROW_BYTES: u64 = 256;

/// Why a store operation did not happen.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StoreError {
    /// The vectors given are from another encoder than the store's (requirement 5.6).
    #[error(
        "the store holds vectors of the encoder {}, but the vectors given are from the \
         encoder {}; vectors of different encoder identities are never stored together or \
         compared",
        describe(.store),
        describe(.given)
    )]
    IdentityMismatch {
        store: EncoderIdentity,
        given: EncoderIdentity,
    },
    /// There is no store of that name.
    #[error("missing store {name}: expected at {}", .expected_at.display())]
    Missing { name: String, expected_at: PathBuf },
    /// A store, or some other file, of that name is there already.
    #[error(
        "the store {name} cannot be created: {} exists already and was left unchanged",
        .at.display()
    )]
    Exists { name: String, at: PathBuf },
    /// The name cannot be a store name.
    #[error(
        "a store name must be 1 to {MAX_NAME_BYTES} letters, digits, '.', '_' or '-' and \
         must not start with a dot; '{name}' is not"
    )]
    InvalidName { name: String },
    /// The encoder identity cannot be one.
    #[error("unusable encoder identity: {detail}")]
    InvalidIdentity { detail: String },
    /// The frame is not a store rows frame, or one of its rows cannot be stored.
    #[error("not store rows: {detail}")]
    InvalidRows { detail: String },
    /// The search cannot be one: an empty phrase, a query vector of another length.
    #[error("not a search: {detail}")]
    InvalidQuery { detail: String },
    /// The memory budget does not allow the operation, or could not be observed.
    #[error("{0}")]
    Memory(L2dError),
    /// The database or the file system failed; the underlying message is included.
    #[error("the store failed: {detail}")]
    Failed { detail: String },
}

fn describe(identity: &EncoderIdentity) -> String {
    format!(
        "'{}' (revision {}, dimension {})",
        identity.name, identity.revision, identity.dimension
    )
}

impl From<StoreError> for L2dError {
    fn from(error: StoreError) -> Self {
        match error {
            StoreError::Missing { name, expected_at } => L2dError::Missing {
                item: format!("store {name}"),
                expected_at,
            },
            StoreError::IdentityMismatch { .. } | StoreError::Exists { .. } => L2dError::Refused {
                reason: error.to_string(),
            },
            StoreError::InvalidName { .. }
            | StoreError::InvalidIdentity { .. }
            | StoreError::InvalidQuery { .. } => L2dError::InvalidInput {
                detail: error.to_string(),
            },
            StoreError::Memory(inner) => inner,
            StoreError::InvalidRows { .. } | StoreError::Failed { .. } => L2dError::Runtime {
                detail: error.to_string(),
            },
        }
    }
}

/// What an add did.
#[derive(Debug, Clone, PartialEq)]
pub struct AddReport {
    /// The rows of the frame.
    pub rows: u64,
    /// The rows that became new stored logs.
    pub added: u64,
    /// The rows whose id was stored already (before the add, or by an earlier row of the
    /// frame): their occurrences were added to the stored log.
    pub repeated: u64,
    /// What the add held of the budget for as long as it ran, beside the batches:
    /// [`ADD_FIXED_BYTES`] and [`ADD_TEXT_FACTOR`] times `longest_text_bytes`, or times
    /// the bytes of a longer text that an earlier add indexed on the same open store
    /// and whose table the text index still held.
    pub held_bytes: u64,
    /// The bytes of the longest text of the frame.
    pub longest_text_bytes: u64,
    /// The free memory observed, the limit, the bytes per row and the batch sizes used
    /// (requirement 12.3, 12.8).
    pub plan: BatchPlan,
}

/// The identity of a log in a store: the SHA-256, in lower-case hexadecimal, of the
/// length of `source` in bytes (64 bits, little-endian), `source` and `text`.
///
/// The length keeps source and text apart, so that no two pairs share an id by
/// shifting characters from one to the other.
pub fn log_id(source: &str, text: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hasher = Sha256::new();
    hasher.update((source.len() as u64).to_le_bytes());
    hasher.update(source.as_bytes());
    hasher.update(text.as_bytes());
    let mut id = String::with_capacity(64);
    for byte in hasher.finalize() {
        id.push(char::from(HEX[usize::from(byte >> 4)]));
        id.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    id
}

/// The column names and types of the store rows frame, in order.
pub fn rows_schema() -> Schema {
    Schema::from_iter([
        Field::new(TEXT.into(), DataType::String),
        Field::new(SOURCE.into(), DataType::String),
        Field::new(OCCURRENCES.into(), DataType::UInt32),
        Field::new(LABEL.into(), DataType::String),
        Field::new(GROUP.into(), DataType::String),
        Field::new(TRUNCATED.into(), DataType::Boolean),
        Field::new(VECTOR.into(), DataType::List(Box::new(DataType::Float32))),
    ])
}

/// The store rows frame for `logs`, all from `source`.
///
/// `vectors` holds the vectors of the logs one after the other in the order of the
/// frame, `dimension` values each; `truncated` holds one flag per log. Other lengths are
/// [`StoreError::InvalidRows`]. The text, label, group and occurrence columns are
/// shared with `logs`, not copied.
pub fn rows_frame(
    logs: &LogsFrame,
    source: &str,
    vectors: &[f32],
    dimension: u16,
    truncated: &[bool],
) -> Result<DataFrame, StoreError> {
    let rows = logs.len();
    let width = usize::from(dimension);
    if vectors.len() != rows * width {
        return Err(StoreError::InvalidRows {
            detail: format!(
                "{rows} logs need {rows} vectors of dimension {dimension}, that is {} values; \
                 {} values were given ({} whole vectors)",
                rows * width,
                vectors.len(),
                vectors.len().checked_div(width).unwrap_or(0),
            ),
        });
    }
    if truncated.len() != rows {
        return Err(StoreError::InvalidRows {
            detail: format!(
                "{rows} logs need {rows} truncation flags; {} were given",
                truncated.len()
            ),
        });
    }
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        VECTOR.into(),
        rows,
        vectors.len(),
        DataType::Float32,
    );
    if width > 0 {
        for vector in vectors.chunks_exact(width) {
            builder.append_slice(vector);
        }
    }
    let column = |name: &str| logs.frame().column(name).cloned().map_err(|e| not_rows(&e));
    DataFrame::new(vec![
        column(LogsFrame::TEXT)?,
        StringChunked::full(SOURCE.into(), source, rows).into_column(),
        column(LogsFrame::OCCURRENCES)?,
        column(LogsFrame::LABEL)?,
        column(LogsFrame::GROUP)?,
        Column::new(TRUNCATED.into(), truncated),
        builder.finish().into_column(),
    ])
    .map_err(|error| not_rows(&error))
}

fn not_rows(error: &PolarsError) -> StoreError {
    StoreError::InvalidRows {
        detail: error.to_string(),
    }
}

/// Checks the columns of a rows frame: names, types, order, and no absent value outside
/// `group`. The values are checked row by row while they are written.
fn check_rows(rows: &DataFrame) -> Result<(), StoreError> {
    let expected = rows_schema();
    let found = rows.schema();
    if **found != expected {
        let list = |schema: &Schema| {
            schema
                .iter()
                .map(|(name, dtype)| format!("{name} {dtype}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(StoreError::InvalidRows {
            detail: format!(
                "expected the columns [{}], found [{}]",
                list(&expected),
                list(found)
            ),
        });
    }
    for column in rows.get_columns() {
        let absent = column.null_count();
        if absent > 0 && column.name().as_str() != GROUP {
            return Err(StoreError::InvalidRows {
                detail: format!("column '{}' holds {absent} absent values", column.name()),
            });
        }
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), StoreError> {
    let usable = !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if usable {
        Ok(())
    } else {
        Err(StoreError::InvalidName {
            name: name.to_string(),
        })
    }
}

fn validate_identity(identity: &EncoderIdentity) -> Result<(), StoreError> {
    let detail = if identity.name.is_empty() {
        "the encoder name is empty"
    } else if identity.revision.is_empty() {
        "the encoder revision is empty"
    } else if identity.dimension == 0 {
        "the dimension is 0; a vector has at least one value"
    } else {
        return Ok(());
    };
    Err(StoreError::InvalidIdentity {
        detail: detail.to_string(),
    })
}

fn failed(what: &str, path: &Path, detail: impl std::fmt::Display) -> StoreError {
    StoreError::Failed {
        detail: format!("{what} {}: {detail}", path.display()),
    }
}

/// One store: an open database file.
#[derive(Debug)]
pub struct Store {
    connection: Connection,
    name: String,
    path: PathBuf,
    meta: sqlite::Meta,
    /// The longest text indexed on this connection since it was opened: the text index
    /// keeps a table of that size (see [`RELEASE_TEXT_BYTES`]).
    indexed_text_bytes: u64,
}

impl Store {
    /// Creates the store `name` for vectors of the encoder `identity`, as
    /// `stores/<name>.db` below the project root.
    ///
    /// - A name that is not 1 to [`MAX_NAME_BYTES`] letters, digits, `.`, `_` or `-`, or
    ///   that starts with a dot, is [`StoreError::InvalidName`].
    /// - An identity with an empty name or revision or the dimension 0 is
    ///   [`StoreError::InvalidIdentity`].
    /// - When the file exists, whatever it holds, the result is [`StoreError::Exists`]
    ///   and the file is left unchanged.
    ///
    /// The name is taken by creating the empty file, which fails if another process got
    /// there first; tables and identity are then written in one transaction. After a
    /// failure the file is removed again. A process that is killed in between leaves an
    /// empty file, which [`Store::open`] reports as such.
    pub fn create(
        paths: &ProjectPaths,
        name: &str,
        identity: &EncoderIdentity,
    ) -> Result<Self, StoreError> {
        validate_name(name)?;
        validate_identity(identity)?;
        let directory = paths.stores_dir();
        let path = paths.store_path(name);
        fs::create_dir_all(&directory).map_err(|error| failed("creating", &directory, error))?;
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                return Err(StoreError::Exists {
                    name: name.to_string(),
                    at: path,
                });
            }
            Err(error) => return Err(failed("creating", &path, error)),
        }
        match sqlite::create(&path, identity, env!("CARGO_PKG_VERSION")) {
            Ok((connection, meta)) => Ok(Self {
                connection,
                name: name.to_string(),
                path,
                meta,
                indexed_text_bytes: 0,
            }),
            Err(detail) => {
                // Best effort: the failure itself is what the caller has to see.
                let _ = fs::remove_file(&path);
                Err(failed("creating the store", &path, detail))
            }
        }
    }

    /// Opens the store `name`.
    ///
    /// A file that is not there is [`StoreError::Missing`], naming the store and where
    /// it was expected. A file that is not a store of this tool and of this format
    /// ([`SCHEMA_VERSION`]) is [`StoreError::Failed`]; it is not changed.
    pub fn open(paths: &ProjectPaths, name: &str) -> Result<Self, StoreError> {
        validate_name(name)?;
        let path = paths.store_path(name);
        let length = match fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => metadata.len(),
            _ => {
                return Err(StoreError::Missing {
                    name: name.to_string(),
                    expected_at: path,
                });
            }
        };
        if length == 0 {
            return Err(failed(
                "opening the store",
                &path,
                "the file is empty: no store was completed there (an interrupted creation \
                 leaves such a file); remove it and create the store again",
            ));
        }
        let (connection, meta) =
            sqlite::open(&path).map_err(|detail| failed("opening the store", &path, detail))?;
        Ok(Self {
            connection,
            name: name.to_string(),
            path,
            meta,
            indexed_text_bytes: 0,
        })
    }

    /// The name the store was created or opened with.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The database file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The one encoder identity of the store's vectors.
    pub fn identity(&self) -> &EncoderIdentity {
        &self.meta.identity
    }

    /// When the store was created: UTC, `YYYY-MM-DDTHH:MM:SSZ`.
    pub fn created_at(&self) -> &str {
        &self.meta.created_at
    }

    /// The version of the tool that created the store.
    pub fn tool_version(&self) -> &str {
        &self.meta.tool_version
    }

    /// The most of SQLite's memory this open store keeps now, between adds: what a
    /// caller that keeps a store open counts against its memory limit.
    ///
    /// It is [`IDLE_BYTES`] at most, because every add ends by giving back what it took:
    /// the page cache always, and the table of the text index, by opening the
    /// connection again, when the add indexed a text longer than
    /// [`RELEASE_TEXT_BYTES`]. Only if that reopening failed (the add itself is not
    /// undone by it) is the figure larger, by [`RETAINED_TEXT_FACTOR`] bytes for each
    /// byte of the longest text; later adds then count that text in their own share and
    /// try again to give it back. The reopening also counts as failed when the file
    /// opens but no longer names this store's encoder and creation: the store then
    /// keeps its connection (every operation checks the file itself, see
    /// [`Store::add_with`]).
    ///
    /// Searching and reading rows back ([`query`], [`vector`]) end by releasing the
    /// page cache as well, so the figure holds after them too; a search counts what is
    /// retained above [`IDLE_BYTES`] in its own share.
    pub fn retained_bytes(&self) -> u64 {
        if self.indexed_text_bytes <= RELEASE_TEXT_BYTES {
            IDLE_BYTES
        } else {
            IDLE_BYTES + RETAINED_TEXT_FACTOR * self.indexed_text_bytes
        }
    }

    /// Gives back what an add took of SQLite's memory (see [`Store::retained_bytes`]).
    /// Failures are not reported: the outcome of the add stands, and what could not be
    /// given back stays counted.
    ///
    /// The file is read again when it is opened again, and it has to describe this
    /// store still: the same encoder identity, creation time and tool version. If it
    /// does not (another program wrote to it), the new connection is dropped and the
    /// old one is kept as after a reopening that failed. Nothing is remembered about
    /// it: every operation that stores or reads rows reads the file's description
    /// itself, in its own transaction ([`Store::require_same_file`]).
    fn release(&mut self) {
        if self.indexed_text_bytes > RELEASE_TEXT_BYTES
            && let Ok((connection, meta)) = sqlite::open(&self.path)
            && meta == self.meta
        {
            // The old connection is closed when it is dropped by the assignment.
            self.connection = connection;
            self.indexed_text_bytes = 0;
            return;
        }
        let _ = self.connection.execute_batch("PRAGMA shrink_memory");
    }

    /// [`StoreError::Failed`], naming both, unless the file still describes this open
    /// store: the encoder identity, the creation time and the tool version of its
    /// `meta` row, read through `connection` (inside the transaction of the operation
    /// that asks, so that what is checked is what is then written to or read).
    ///
    /// Another program can write to the file while the store is open. Without this
    /// check an add would store vectors under an identity the file no longer has, and a
    /// search would compare vectors with those of another encoder.
    pub(crate) fn require_same_file(&self, connection: &Connection) -> Result<(), StoreError> {
        let found = sqlite::read_meta(connection)
            .map_err(|error| failed("reading the description of the store", &self.path, error))?;
        if found == self.meta {
            return Ok(());
        }
        let say = |meta: &sqlite::Meta| {
            format!(
                "the encoder {}, created at {} by version {} of the tool",
                describe(&meta.identity),
                meta.created_at,
                meta.tool_version
            )
        };
        Err(failed(
            "using the open store",
            &self.path,
            format!(
                "the file changed under it: it was opened as a store of {}, and now says {}; \
                 nothing was stored or read through this open store; open it again",
                say(&self.meta),
                say(&found)
            ),
        ))
    }

    /// The bytes of the longest stored text; 0 for an empty store.
    ///
    /// The file keeps an index over the lengths of the texts (`logs_text_bytes`), so
    /// this is one look-up. A store written before that index existed answers by
    /// reading the length of every row, until its next add creates the index.
    pub fn longest_text_bytes(&self) -> Result<u64, StoreError> {
        self.connection
            .query_row(sqlite::LONGEST_TEXT_SQL, [], |row| row.get(0))
            .map_err(|error| failed("reading the longest text of", &self.path, error))
    }

    /// The number of stored logs.
    pub fn len(&self) -> Result<u64, StoreError> {
        self.connection
            .query_row("SELECT count(*) FROM logs", [], |row| row.get(0))
            .map_err(|error| failed("counting the logs of", &self.path, error))
    }

    /// Whether the store holds no log.
    pub fn is_empty(&self) -> Result<bool, StoreError> {
        self.len().map(|logs| logs == 0)
    }

    /// [`StoreError::IdentityMismatch`], naming both, unless `given` is the identity of
    /// the store's encoder: name, revision and dimension.
    pub fn require_identity(&self, given: &EncoderIdentity) -> Result<(), StoreError> {
        if *given == self.meta.identity {
            Ok(())
        } else {
            Err(StoreError::IdentityMismatch {
                store: self.meta.identity.clone(),
                given: given.clone(),
            })
        }
    }

    /// [`Store::add_with`] under a budget observed now with the default settings.
    ///
    /// That budget is fresh: nothing is held in it, so the frame `rows` itself and
    /// whatever else the caller has in memory are counted nowhere on this path. It is
    /// for callers whose frame is small. A caller that read its logs under a budget, or
    /// holds a frame of any size that matters, passes that budget to
    /// [`Store::add_with`], so that the frame counts against the same limit.
    pub fn add(
        &mut self,
        rows: &DataFrame,
        identity: &EncoderIdentity,
    ) -> Result<AddReport, StoreError> {
        let mut budget =
            MemoryBudget::observe(&MemorySettings::default()).map_err(StoreError::Memory)?;
        self.add_with(rows, identity, &mut budget)
    }

    /// Adds the logs of the store rows frame `rows` (see the crate description), whose
    /// vectors were produced by the encoder `identity`.
    ///
    /// - Another identity than the store's is [`StoreError::IdentityMismatch`].
    /// - A frame with other columns, or a row with an unknown label, no occurrence, or
    ///   a vector that is not `dimension` finite values, is [`StoreError::InvalidRows`];
    ///   the message names the row, counted from 1.
    /// - A log whose id is stored already has its occurrences added to the stored row;
    ///   nothing else of that row changes. A count beyond 32 bits is
    ///   [`StoreError::Failed`].
    /// - A text longer than [`MAX_TEXT_BYTES`] is [`StoreError::InvalidRows`], naming
    ///   the row and the limit, and so is a text that holds a NUL character.
    /// - When the budget leaves less than the add's own share ([`ADD_FIXED_BYTES`] and
    ///   [`ADD_TEXT_FACTOR`] times the bytes of the longest text of the frame, or of the
    ///   longest text this open store still holds a table for, if that is longer), or
    ///   not enough for the smallest batch, the add is refused ([`StoreError::Memory`])
    ///   with the memory needed and the memory left.
    ///
    /// - When the file no longer describes this open store (another program changed
    ///   its `meta` row: encoder identity, creation time or tool version), the add is
    ///   [`StoreError::Failed`], naming what the store was opened as and what the file
    ///   says now, and nothing is written. This is read inside the add's transaction,
    ///   before the first row, for every add.
    ///
    /// The frame is written in batches in row order; the free memory is read again
    /// before each batch. The stored rows do not depend on the batches. Whatever the
    /// outcome, the budget holds afterwards what it held before, the store has given
    /// back what the add took of SQLite's memory ([`Store::retained_bytes`]), and after
    /// anything but success the stored rows are as they were.
    pub fn add_with(
        &mut self,
        rows: &DataFrame,
        identity: &EncoderIdentity,
        budget: &mut MemoryBudget,
    ) -> Result<AddReport, StoreError> {
        self.require_identity(identity)?;
        check_rows(rows)?;
        let longest = longest_text(rows)?;
        if longest.bytes > MAX_TEXT_BYTES {
            return Err(StoreError::InvalidRows {
                detail: format!(
                    "row {}: the text has {} bytes; a stored text has at most {MAX_TEXT_BYTES} \
                     bytes, the longest for which the memory of the text index is known",
                    longest.row, longest.bytes
                ),
            });
        }

        // The text index still holds its table for the longest text it indexed on this
        // connection, so that text counts when it is longer than any of this frame.
        let counted = longest.bytes.max(self.indexed_text_bytes);
        let text_share = ADD_TEXT_FACTOR * counted;
        let own = ADD_FIXED_BYTES + text_share;
        let left = budget.left_bytes();
        if left < own {
            let which = if counted == longest.bytes {
                format!("{} bytes, row {}", longest.bytes, longest.row)
            } else {
                format!(
                    "{counted} bytes, indexed by an earlier add on this open store, whose \
                     table the text index still holds"
                )
            };
            return Err(StoreError::Memory(L2dError::Refused {
                reason: format!(
                    "adding to a store needs {own} bytes beside the rows: {ADD_FIXED_BYTES} \
                     bytes for the database itself and {text_share} bytes for indexing the \
                     longest text ({which}; {ADD_TEXT_FACTOR} bytes for each of its bytes); \
                     but {left} bytes are left of the memory limit of {} bytes; nothing was \
                     written",
                    budget.limit_bytes()
                ),
            }));
        }
        // What the caller holds already (its frames) stays counted during the add.
        let held = budget.usable_bytes() - left;
        budget.hold(held + own);
        let result = self.add_in_batches(rows, budget, own, longest.bytes);
        // Whatever the outcome: SQLite's memory is given back before the budget is.
        self.release();
        budget.hold(held);
        result
    }

    fn add_in_batches(
        &mut self,
        rows: &DataFrame,
        budget: &mut MemoryBudget,
        own: u64,
        longest_text_bytes: u64,
    ) -> Result<AddReport, StoreError> {
        let total = rows.height() as u64;
        let bytes_per_row = (rows.estimated_size() as u64).div_ceil(total.max(1)) + ADD_ROW_BYTES;
        let mut plan = budget
            .plan(total, bytes_per_row)
            .map_err(StoreError::Memory)?;
        let mut report = AddReport {
            rows: total,
            added: 0,
            repeated: 0,
            held_bytes: own,
            longest_text_bytes,
            plan: plan.clone(),
        };
        let path = self.path.clone();
        let fail = |error: rusqlite::Error| failed("adding to the store", &path, error);
        let dimension = usize::from(self.meta.identity.dimension);
        let revision = self.meta.identity.revision.clone();
        // Dropped without a commit, the transaction is rolled back: an add is all or
        // nothing.
        let transaction = self.connection.unchecked_transaction().map_err(fail)?;
        // Before anything is written: the file must still be this store's.
        self.require_same_file(&transaction)?;
        if total == 0 {
            return Ok(report);
        }
        // From here on texts are indexed, also by an add that fails later.
        self.indexed_text_bytes = self.indexed_text_bytes.max(longest_text_bytes);
        // A store written before the index over the text lengths existed gets it with
        // its first add; otherwise this does nothing.
        transaction
            .execute_batch(sqlite::TEXT_BYTES_INDEX_SQL)
            .map_err(fail)?;
        {
            let mut writer = Writer {
                insert: transaction
                    .prepare(
                        "INSERT INTO logs (id, text, source, occurrences, label, group_key, \
                         truncated, vector, encoder_revision) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) \
                         ON CONFLICT (id) DO NOTHING",
                    )
                    .map_err(fail)?,
                repeat: transaction
                    .prepare("UPDATE logs SET occurrences = occurrences + ?2 WHERE id = ?1")
                    .map_err(fail)?,
                dimension,
                revision: &revision,
                path: &path,
                blob: Vec::with_capacity(dimension * 4),
            };
            let mut done = 0u64;
            while done < total {
                let batch_rows = budget
                    .next_batch_rows(&mut plan, done)
                    .map_err(|error| refused_between_batches(error, own, done))?
                    .min(total - done);
                let batch = rows.slice(done as i64, batch_rows as usize);
                writer.write(&batch, done, &mut report)?;
                done += batch_rows;
            }
        }
        transaction.commit().map_err(fail)?;
        report.plan = plan;
        Ok(report)
    }
}

/// The longest text of a rows frame.
struct LongestText {
    /// Its length in bytes; 0 for a frame without rows.
    bytes: u64,
    /// Its row, counted from 1 (the first of that length); 0 for a frame without rows.
    row: u64,
}

fn longest_text(rows: &DataFrame) -> Result<LongestText, StoreError> {
    let texts = rows
        .column(TEXT)
        .and_then(|column| column.str())
        .map_err(|error| not_rows(&error))?;
    let mut longest = LongestText { bytes: 0, row: 0 };
    for (index, text) in texts.iter().enumerate() {
        let bytes = text.map_or(0, str::len) as u64;
        if bytes > longest.bytes || longest.row == 0 {
            longest = LongestText {
                bytes,
                row: index as u64 + 1,
            };
        }
    }
    Ok(longest)
}

/// The refusal of the memory budget before a batch that is not the first, in the
/// store's terms: the budget's own text speaks of rows read and held, but what an add
/// holds is its own share, and what it has written so far is rolled back.
fn refused_between_batches(error: L2dError, own: u64, rows_written: u64) -> StoreError {
    StoreError::Memory(match error {
        L2dError::Refused { reason } => L2dError::Refused {
            reason: format!(
                "{}; of what is held, {own} bytes are the store's own share for this add \
                 (the database and the indexing of its longest text) and the rest is what \
                 the caller held before; the {rows_written} rows written so far were rolled \
                 back and the store is as it was",
                reason.replace("is held by the rows read so far", "is held")
            ),
        },
        other => other,
    })
}

/// Writes rows through the two statements of an add.
struct Writer<'a> {
    insert: Statement<'a>,
    repeat: Statement<'a>,
    dimension: usize,
    revision: &'a str,
    path: &'a Path,
    /// The vector of the row being written, as it is stored; reused from row to row.
    blob: Vec<u8>,
}

impl Writer<'_> {
    /// Writes the rows of `batch`, whose first row is row `rows_before + 1` of the
    /// frame.
    fn write(
        &mut self,
        batch: &DataFrame,
        rows_before: u64,
        report: &mut AddReport,
    ) -> Result<(), StoreError> {
        let column = |name: &str| batch.column(name).map_err(|error| not_rows(&error));
        let texts = column(TEXT)?.str().map_err(|e| not_rows(&e))?;
        let sources = column(SOURCE)?.str().map_err(|e| not_rows(&e))?;
        let occurrences = column(OCCURRENCES)?.u32().map_err(|e| not_rows(&e))?;
        let labels = column(LABEL)?.str().map_err(|e| not_rows(&e))?;
        let groups = column(GROUP)?.str().map_err(|e| not_rows(&e))?;
        let truncated = column(TRUNCATED)?.bool().map_err(|e| not_rows(&e))?;
        let vectors = column(VECTOR)?.list().map_err(|e| not_rows(&e))?;

        let values = texts
            .iter()
            .zip(sources.iter())
            .zip(occurrences.iter())
            .zip(labels.iter())
            .zip(groups.iter())
            .zip(truncated.iter())
            .zip(vectors.amortized_iter());
        for (index, value) in values.enumerate() {
            let row = rows_before + index as u64 + 1;
            let ((((((text, source), occurrences), label), group), truncated), vector) = value;
            let invalid = |what: String| StoreError::InvalidRows {
                detail: format!("row {row}: {what}"),
            };
            let absent = |name: &str| invalid(format!("'{name}' is absent"));
            let text = text.ok_or_else(|| absent(TEXT))?;
            let source = source.ok_or_else(|| absent(SOURCE))?;
            let occurrences = occurrences.ok_or_else(|| absent(OCCURRENCES))?;
            let label = label.ok_or_else(|| absent(LABEL))?;
            let truncated = truncated.ok_or_else(|| absent(TRUNCATED))?;
            let vector = vector.ok_or_else(|| absent(VECTOR))?;

            if let Some(at) = text.find('\0') {
                return Err(invalid(format!(
                    "the text holds a NUL character (at byte {}); the text index reads a \
                     text up to its first NUL only, so what follows could not be searched",
                    at + 1
                )));
            }
            if Label::from_name(label).is_none() {
                return Err(invalid(format!(
                    "the label '{label}' is not good, bad or unknown"
                )));
            }
            if occurrences == 0 {
                return Err(invalid(
                    "the occurrence count is 0; a stored log occurred at least once".into(),
                ));
            }
            let vector = vector.as_ref().f32().map_err(|e| not_rows(&e))?;
            if vector.len() != self.dimension {
                return Err(invalid(format!(
                    "the vector has {} values; the store's encoder has the dimension {}",
                    vector.len(),
                    self.dimension
                )));
            }
            self.blob.clear();
            for (position, value) in vector.iter().enumerate() {
                match value {
                    Some(value) if value.is_finite() => {
                        self.blob.extend_from_slice(&value.to_le_bytes());
                    }
                    Some(value) => {
                        return Err(invalid(format!(
                            "value {} of the vector is {value}, not a finite number",
                            position + 1
                        )));
                    }
                    None => {
                        return Err(invalid(format!(
                            "value {} of the vector is absent",
                            position + 1
                        )));
                    }
                }
            }

            let id = log_id(source, text);
            let fail = |error: rusqlite::Error| {
                failed(
                    "adding to the store",
                    self.path,
                    format!("row {row}: {error}"),
                )
            };
            let inserted = self
                .insert
                .execute(rusqlite::params![
                    id,
                    text,
                    source,
                    occurrences,
                    label,
                    group,
                    truncated,
                    self.blob.as_slice(),
                    self.revision,
                ])
                .map_err(fail)?;
            if inserted == 1 {
                report.added += 1;
                continue;
            }
            let repeated = self
                .repeat
                .execute(rusqlite::params![id, occurrences])
                .map_err(fail)?;
            if repeated != 1 {
                return Err(failed(
                    "adding to the store",
                    self.path,
                    format!("row {row}: the id {id} was neither inserted nor found"),
                ));
            }
            report.repeated += 1;
        }
        Ok(())
    }
}
