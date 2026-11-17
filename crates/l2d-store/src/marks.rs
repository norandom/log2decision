//! What marking needs of a store (requirement 16.1 and 16.12; task 1.21, owner decision
//! of 2026-11-16): the store's content hash, the ids of the logs a phrase finds through
//! the text index, and the logs read in batches of `(seq, id, text)`. [`Store`]
//! implements [`MarkStore`] of `l2d-core` with them.
//!
//! # The content hash
//!
//! The SHA-256 over the ids of all stored logs (64 lower-case hexadecimal characters
//! each, nothing between them) in ascending order of the id. The same set of logs gives
//! the same hash whatever order they were stored in, and adding a log changes it. The
//! ids are read in the order of the unique index on `id` ([`CONTENT_HASH_SQL`]), so
//! nothing is sorted and nothing grows with the number of logs: the read holds the
//! search's fixed share only.
//!
//! # Phrases
//!
//! A phrase is searched byte for byte (`case_exact`) by the store's text search
//! ([`crate::query`]): the text index narrows the logs and every one left is compared.
//! Before the first id is handed out, the matches are counted and the caller's cost for
//! their ids is asked of the budget beside the search's share.
//!
//! # Batches
//!
//! [`Store::text_batches_with`] reads every stored log in the order stored
//! ([`TEXT_BATCH_SQL`]), within one read transaction, into frames of `seq` (Int64),
//! `id` and `text` (String). Beside the search's fixed share it holds
//! [`TEXT_READ_FACTOR`] times the longest stored text for SQLite's copy of the text
//! being read. A batch is cut before a further row of the longest text would no longer
//! fit what is left beside what the caller holds ([`READ_FIXED_BYTES`] for the frame,
//! [`READ_ROW_BYTES`], twice the text and the caller's `per_row` bytes for every row),
//! or at the budget's largest batch. When not even one row fits, the read is refused
//! with the figures.

use l2d_core::marks::MarkStore;
use l2d_core::polars::prelude::{Column, DataFrame, IntoColumn, StringChunkedBuilder};
use l2d_core::{L2dError, MemoryBudget};
use sha2::{Digest, Sha256};

use crate::query::{Compiled, Filter, ID, READ_FIXED_BYTES, READ_ROW_BYTES, TEXT_READ_FACTOR};
use crate::{Store, StoreError, TEXT, failed};

/// The statement that reads the ids for the content hash, in the order of the unique
/// index on `id`.
pub const CONTENT_HASH_SQL: &str = "SELECT id FROM logs ORDER BY id";

/// The statement that reads the logs in batches, in the order stored.
pub const TEXT_BATCH_SQL: &str = "SELECT seq, id, text FROM logs ORDER BY seq";

/// The column of a batch of texts with the row id of a log, its order of storing.
pub const SEQ: &str = "seq";

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(2 * bytes.len());
    for byte in bytes {
        text.push(char::from(HEX[usize::from(byte >> 4)]));
        text.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    text
}

/// The columns of one batch of texts while it is read.
struct TextBatch {
    seq: Vec<i64>,
    occurrences: Vec<u32>,
    id: StringChunkedBuilder,
    text: StringChunkedBuilder,
}

impl TextBatch {
    fn new() -> Self {
        Self {
            seq: Vec::new(),
            occurrences: Vec::new(),
            id: StringChunkedBuilder::new(ID.into(), 0),
            text: StringChunkedBuilder::new(TEXT.into(), 0),
        }
    }

    fn finish(self, counts: bool) -> Result<DataFrame, l2d_core::polars::prelude::PolarsError> {
        let mut columns = vec![
            Column::new(SEQ.into(), self.seq),
            self.id.finish().into_column(),
            self.text.finish().into_column(),
        ];
        if counts {
            columns.push(Column::new("occurrences".into(), self.occurrences));
        }
        DataFrame::new(columns)
    }
}

impl Store {
    /// The store's content hash (see the module description), read within `budget`:
    /// the search's fixed share is held while it runs, nothing else.
    pub fn content_sha256_with(&self, budget: &mut MemoryBudget) -> Result<String, StoreError> {
        let filter = Filter::default();
        let compiled = Compiled::new(&filter)?;
        self.search(&compiled, 0, "", budget, |_, _| {
            let read = || -> rusqlite::Result<String> {
                let mut statement = self.connection.prepare(CONTENT_HASH_SQL)?;
                let mut rows = statement.query([])?;
                let mut hasher = Sha256::new();
                while let Some(row) = rows.next()? {
                    hasher.update(row.get_ref(0)?.as_str()?.as_bytes());
                }
                Ok(hex(&hasher.finalize()))
            };
            read().map_err(|error| self.failure(error))
        })
    }

    /// Hands `each` the id of every stored log whose text contains `phrase` byte for
    /// byte, in the order stored, through the store's text search; returns how many.
    /// The matches are counted first and `id_bytes` for each is asked of the budget
    /// beside the search's share: [`StoreError::Memory`] with the figures when that is
    /// not left, before an id is handed out. An empty phrase is
    /// [`StoreError::InvalidQuery`].
    pub fn phrase_ids_with(
        &self,
        phrase: &str,
        id_bytes: u64,
        budget: &mut MemoryBudget,
        mut each: impl FnMut(&str),
    ) -> Result<u64, StoreError> {
        let filter = Filter {
            contains: vec![phrase.to_string()],
            case_exact: true,
            ..Filter::default()
        };
        let compiled = Compiled::new(&filter)?;
        self.search(&compiled, 0, "", budget, |budget, own| {
            let rows = self.measure(&compiled, usize::MAX)?.rows;
            let needed = rows.saturating_mul(id_bytes);
            let left = budget.left_bytes();
            if needed > left {
                return Err(StoreError::Memory(L2dError::Refused {
                    reason: format!(
                        "the ids of the {rows} logs of the store {} that contain the phrase \
                         need {needed} bytes ({id_bytes} each) beside the {own} bytes the \
                         search holds, but {left} bytes are left of the memory limit of {} \
                         bytes; no id was handed out",
                        self.name(),
                        budget.limit_bytes()
                    ),
                }));
            }
            let read = |each: &mut dyn FnMut(&str)| -> rusqlite::Result<u64> {
                let mut statement = self
                    .connection
                    .prepare(&compiled.sql("octet_length(logs.text), logs.text, logs.id"))?;
                let mut cursor = statement.query(compiled.parameters().as_slice())?;
                let mut read = 0u64;
                while let Some(row) = cursor.next()? {
                    if compiled.accepts(row, 0, 1)? {
                        each(row.get_ref(2)?.as_str()?);
                        read += 1;
                    }
                }
                Ok(read)
            };
            let read = read(&mut each).map_err(|error| self.failure(error))?;
            if read != rows {
                return Err(failed(
                    "searching the store",
                    self.path(),
                    format!(
                        "{rows} logs matched when they were counted, {read} when they were read"
                    ),
                ));
            }
            Ok(read)
        })
    }

    /// Hands `each` every stored log in batches (see the module description) and
    /// returns how many logs were read. `beside` is what the caller holds beside the
    /// read before the first batch, and `each` returns what it holds after each batch;
    /// `per_row` is what the caller may take for each row of a batch while it has the
    /// batch. An error of `each` ends the read and is returned as it is (in
    /// [`StoreError::Memory`]). The budget holds afterwards what it held before.
    pub fn text_batches_with(
        &self,
        beside: u64,
        per_row: u64,
        budget: &mut MemoryBudget,
        each: impl FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<u64, StoreError> {
        self.pattern_text_batches(false, beside, per_row, budget, each)
    }
    fn pattern_text_batches(
        &self,
        counts: bool,
        beside: u64,
        per_row: u64,
        budget: &mut MemoryBudget,
        mut each: impl FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<u64, StoreError> {
        let filter = Filter::default();
        let compiled = Compiled::new(&filter)?;
        self.search(&compiled, 0, "", budget, |budget, own| {
            // The maximum and the cursor must belong to the SAME read snapshot.
            // A writer can commit a longer text before BEGIN. Reserve its native
            // reader before preparing or stepping the text query, not afterwards.
            let longest = self.longest_text_bytes()?;
            let reading = TEXT_READ_FACTOR.saturating_mul(longest);
            budget
                .require_whole(
                    reading,
                    "reading stored texts from the same snapshot as the longest-text bound",
                )
                .map_err(StoreError::Memory)?;
            budget.hold(budget.held_bytes().saturating_add(reading));
            let own = own.saturating_add(reading);
            let left = budget.left_bytes();
            let limit = budget.limit_bytes();
            let max_rows = budget.max_batch_rows();
            let row = |text: u64| READ_ROW_BYTES + 2 * text + per_row + if counts { 16 } else { 0 };
            let refused = |beside: u64, done: u64| {
                StoreError::Memory(L2dError::Refused {
                    reason: format!(
                        "reading the store {} in batches needs {} bytes for a batch of one row \
                         of its longest text ({longest} bytes) beside the {beside} bytes held \
                         beside the read and the {own} bytes the read holds, but {left} bytes \
                         are left of the memory limit of {limit} bytes; stopped after {done} \
                         logs",
                        self.name(),
                        READ_FIXED_BYTES + row(longest)
                    ),
                })
            };
            let fail = |error| self.failure(error);
            let mut statement = self
                .connection
                .prepare(if counts {
                    "SELECT seq,id,text,occurrences FROM logs ORDER BY seq"
                } else {
                    TEXT_BATCH_SQL
                })
                .map_err(fail)?;
            let mut cursor = statement.query([]).map_err(fail)?;
            let mut beside = beside;
            let mut done = 0u64;
            let mut batch = TextBatch::new();
            let mut rows = 0u64;
            let mut bytes = READ_FIXED_BYTES;
            loop {
                let room = left.saturating_sub(beside);
                if rows == 0 && bytes + row(longest) > room {
                    return Err(refused(beside, done));
                }
                let full = rows > 0 && (rows == max_rows || bytes + row(longest) > room);
                let next = if full {
                    None
                } else {
                    cursor.next().map_err(fail)?
                };
                if let Some(next) = next {
                    let mut read = || -> rusqlite::Result<u64> {
                        let text = next.get_ref(2)?.as_str()?;
                        batch.seq.push(next.get(0)?);
                        if counts {
                            batch.occurrences.push(next.get(3)?);
                        }
                        batch.id.append_value(next.get_ref(1)?.as_str()?);
                        batch.text.append_value(text);
                        Ok(text.len() as u64)
                    };
                    bytes += row(read().map_err(fail)?);
                    rows += 1;
                    continue;
                }
                if rows == 0 {
                    return Ok(done);
                }
                let frame = std::mem::replace(&mut batch, TextBatch::new())
                    .finish(counts)
                    .map_err(|error| failed("reading the store", self.path(), error))?;
                beside = each(&frame).map_err(StoreError::Memory)?;
                drop(frame);
                done += rows;
                rows = 0;
                bytes = READ_FIXED_BYTES;
            }
        })
    }
}

impl MarkStore for Store {
    fn store_name(&self) -> &str {
        self.name()
    }

    fn content_sha256(&self, budget: &mut MemoryBudget) -> Result<String, L2dError> {
        Ok(self.content_sha256_with(budget)?)
    }

    fn phrase_ids(
        &self,
        phrase: &str,
        id_bytes: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&str),
    ) -> Result<(), L2dError> {
        self.phrase_ids_with(phrase, id_bytes, budget, each)?;
        Ok(())
    }

    fn text_batches(
        &self,
        beside: u64,
        per_row: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        self.text_batches_with(beside, per_row, budget, each)?;
        Ok(())
    }
}

impl l2d_core::selections::SelectionStore for Store {
    fn fts5_trigram_ids(
        &self,
        expression: &str,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&str) -> Result<(), L2dError>,
    ) -> Result<u64, L2dError> {
        if expression.is_empty() || expression.len() > 4096 {
            return Err(L2dError::InvalidInput {
                detail: "FTS5 expression requires 1 to 4096 bytes".into(),
            });
        }
        {
            // A prefix can expand to arbitrarily many vocabulary entries. Its
            // native readers cannot be bounded by the supplied expression length.
            // FTS5 quotes literal stars; doubled quotes leave this state unchanged.
            let mut quoted = false;
            for byte in expression.bytes() {
                if byte == b'"' {
                    quoted = !quoted;
                }
                if byte == b'*' && !quoted {
                    return Err(L2dError::Refused { reason: "FTS5 prefix expansion (*) outside quoted text is not supported within the bounded native-memory contract; use complete terms or quoted literal text".into() });
                }
            }
        }
        let filter = Filter::default();
        let compiled = Compiled::new(&filter)?;
        Ok(self.search(&compiled, 0, "", budget, |budget, _| {
            // self.search reserves the existing database/page-cache share and
            // opens the read transaction first. Determine the longest text in
            // that SAME snapshot, then reserve native MATCH readers before
            // preparing MATCH, counting rows, or invoking any callback.
            let longest = self.longest_text_bytes()?;
            let bytes = expression.len() as u64;
            let characters = expression.chars().count() as u64;
            // Charge every expression byte/character, including syntax, rather
            // than attempting to undercount parsed phrase/compound operands.
            let native = crate::query::PHRASE_BYTE_BYTES.saturating_mul(bytes)
                .saturating_add(crate::query::INDEX_TEXT_FACTOR.saturating_mul(characters).saturating_mul(longest));
            budget.require_whole(native, "FTS5 native trigram readers and longest-text position lists before MATCH execution").map_err(StoreError::Memory)?;
            budget.hold(budget.held_bytes().saturating_add(native));
            let fail = |e| self.failure(e);
            let total: u64 = self.connection.query_row("SELECT count(*) FROM logs", [], |r| r.get(0)).map_err(fail)?;
            let mut statement = self.connection.prepare("SELECT logs.id FROM logs_fts JOIN logs ON logs.seq = logs_fts.rowid WHERE logs_fts MATCH ?1").map_err(fail)?;
            let mut rows = statement.query([expression]).map_err(fail)?;
            while let Some(row) = rows.next().map_err(fail)? {
                each(row.get_ref(0).map_err(fail)?.as_str().map_err(|e| self.failure(e.into()))?).map_err(StoreError::Memory)?;
            }
            Ok(total)
        })?)
    }
}

impl l2d_core::patterns::PatternStore for Store {
    fn pattern_snapshot(&self, budget: &mut MemoryBudget) -> Result<String, L2dError> {
        let filter = Filter::default();
        let compiled = Compiled::new(&filter)?;
        Ok(self.search(&compiled, 0, "", budget, |_, _| {
            let mut query = self
                .connection
                .prepare("SELECT id,occurrences FROM logs ORDER BY id")
                .map_err(|e| self.failure(e))?;
            let mut rows = query.query([]).map_err(|e| self.failure(e))?;
            let mut hash = Sha256::new();
            while let Some(row) = rows.next().map_err(|e| self.failure(e))? {
                hash.update(
                    row.get_ref(0)
                        .map_err(|e| self.failure(e))?
                        .as_str()
                        .map_err(|e| self.failure(e.into()))?
                        .as_bytes(),
                );
                hash.update(
                    row.get::<_, u32>(1)
                        .map_err(|e| self.failure(e))?
                        .to_le_bytes(),
                );
            }
            Ok(hex(&hash.finalize()))
        })?)
    }
    fn pattern_batches(
        &self,
        beside: u64,
        per_row: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        self.pattern_text_batches(true, beside, per_row, budget, each)?;
        Ok(())
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::{LogFlags, VectorOrigin, rows_frame};
    use l2d_core::patterns::PatternStore;
    use l2d_core::{EncoderIdentity, Label, LogRecord, LogsFrame, MemorySettings, ProjectPaths};
    use std::path::PathBuf;
    fn identity() -> EncoderIdentity {
        EncoderIdentity {
            name: "pattern-snapshot".into(),
            revision: "a".repeat(64),
            dimension: 4,
        }
    }
    fn budget(usable: u64) -> MemoryBudget {
        MemoryBudget::observe_with(
            &MemorySettings {
                memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + usable),
                ..Default::default()
            },
            || Ok(8 * 1024 * 1024 * 1024),
        )
        .unwrap()
    }
    fn add(store: &mut Store, text: String) {
        let logs = LogsFrame::from_records(&[LogRecord {
            text,
            label: Label::Unknown,
            group: None,
            first_position: 1,
            occurrences: 1,
        }])
        .unwrap();
        let rows = rows_frame(&logs, "test", &[1., 0., 0., 0.], 4, &[LogFlags::default()]).unwrap();
        store
            .add_with(
                &rows,
                VectorOrigin::without_release(&identity()),
                &mut budget(512 * 1024 * 1024),
            )
            .unwrap();
    }
    #[test]
    fn longer_writer_child() {
        let Some(root) = std::env::var_os("L2D_PATTERN_LONGER_WRITER") else {
            return;
        };
        let mut store = Store::open(&ProjectPaths::new(root), "s").unwrap();
        add(&mut store, "x".repeat(1024 * 1024));
    }
    struct Trace {
        root: PathBuf,
        fired: bool,
        error: Option<String>,
    }
    // SAFETY: registered only for SQLITE_TRACE_STMT; SQLite supplies a valid SQL
    // C string and our live exclusive Trace pointer. No panic crosses this callback.
    unsafe extern "C" fn before_begin(
        _: u32,
        context: *mut std::ffi::c_void,
        _: *mut std::ffi::c_void,
        sql: *mut std::ffi::c_void,
    ) -> i32 {
        let trace = unsafe { &mut *context.cast::<Trace>() };
        let sql = unsafe { std::ffi::CStr::from_ptr(sql.cast()) }.to_bytes();
        if !trace.fired && sql.starts_with(b"BEGIN") {
            trace.fired = true;
            let result = std::env::current_exe().and_then(|exe| {
                std::process::Command::new(exe)
                    .args([
                        "--exact",
                        "marks::snapshot_tests::longer_writer_child",
                        "--nocapture",
                    ])
                    .env("L2D_PATTERN_LONGER_WRITER", &trace.root)
                    .output()
            });
            match result {
                Ok(output) if output.status.success() => {}
                Ok(output) => {
                    trace.error = Some(String::from_utf8_lossy(&output.stderr).into_owned())
                }
                Err(e) => trace.error = Some(e.to_string()),
            }
        }
        0
    }
    #[test]
    fn concurrent_longer_text_is_measured_inside_read_snapshot() {
        let root = std::env::temp_dir().join(format!("l2d-pattern-longer-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let paths = ProjectPaths::new(&root);
        let id = identity();
        let mut store = Store::create(&paths, "s", VectorOrigin::without_release(&id)).unwrap();
        add(&mut store, "short".into());
        let mut trace = Trace {
            root: root.clone(),
            fired: false,
            error: None,
        };
        // SAFETY: connection remains live, callback userdata is pinned on this stack
        // until the callback is unregistered immediately after the synchronous read.
        let native_before = unsafe {
            let rc = rusqlite::ffi::sqlite3_trace_v2(
                store.connection.handle(),
                rusqlite::ffi::SQLITE_TRACE_STMT as u32,
                Some(before_begin),
                (&mut trace as *mut Trace).cast(),
            );
            assert_eq!(rc, rusqlite::ffi::SQLITE_OK);
            rusqlite::ffi::sqlite3_memory_highwater(1);
            rusqlite::ffi::sqlite3_memory_used()
        };
        let mut seen = 0;
        let mut b = budget(17 * 1024 * 1024);
        let held = b.held_bytes();
        let result = store.pattern_batches(0, 0, &mut b, &mut |rows| {
            seen += rows.height();
            Ok(0)
        });
        // SAFETY: connection is still live; unregister before dropping Trace.
        let native_peak = unsafe {
            rusqlite::ffi::sqlite3_trace_v2(
                store.connection.handle(),
                0,
                None,
                std::ptr::null_mut(),
            );
            rusqlite::ffi::sqlite3_memory_highwater(0) - native_before
        };
        assert!(trace.fired, "writer must run exactly at BEGIN");
        assert!(trace.error.is_none(), "writer error: {:?}", trace.error);
        assert_eq!(store.longest_text_bytes().unwrap(), 1024 * 1024);
        println!(
            "concurrent longest-text read: native peak growth={native_peak}, callback rows={seen}, result={result:?}"
        );
        assert!(
            result.is_err(),
            "must refuse before reading an unadmitted long row"
        );
        assert_eq!(seen, 0);
        assert!(native_peak < 1024 * 1024);
        assert_eq!(b.held_bytes(), held);
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }
}
