//! Encoding logs into a store (task 2.7; requirements 4.1, 5.1, 5.6 and 12): the one
//! place where the logs frame of `l2d-core`, the encoder session of `l2d-encoder` and the
//! store of `l2d-store` meet. It lives here because this is the first crate that may
//! import both of the latter.
//!
//! [`encode_into_store`] takes the distinct texts of a logs frame, has the encoder turn
//! them into vectors batch by batch, and adds each batch to the store under the encoder's
//! identity and the release of the AMD runtime the vectors were made under
//! ([`VectorOrigin`]). The store keeps one vector per log.
//!
//! # NPU only
//!
//! The tool's one encoder is [`EncoderSession`], which opens on the NPU or not at all;
//! there is no other provider and no fallback. A caller opens the session first: when the
//! NPU cannot be used, `EncoderSession::open` says why and there is no encoder to call
//! this with, so nothing is stored. [`LogEncoder`] is the seam that lets everything
//! around the device (the batches, the rows, the refusals, the memory) be tested on a
//! host without it; the tests put the encoder crate's `ImageEncoder` around a formula
//! behind it.
//!
//! # Nothing here knows an encoder
//!
//! The dimension, the name, the revision and the release are what the encoder says they
//! are; the input mode and the input length are the encoder's own business and are seen
//! here only as the flags it reports per text. A store made by one encoder, or under one
//! release, refuses the vectors of another before a text is encoded.
//!
//! # What is stored of the flags
//!
//! The store keeps three flags per log ([`LogFlags`]), and each is what the encoder
//! reports for the text ([`TextInput`]): `truncated` (the part of the text the encoder
//! reads was cut), `text_longer_than_input` (the whole text is longer than the encoder's
//! input: the second flag of requirement 1.5) and `empty` (no byte other than padding
//! reached the model). They are also counted in the [`FillReport`].
//!
//! # A fill is all or nothing
//!
//! All batches of one call go into one transaction of the store ([`Fill`]): the vectors
//! are encoded and added batch by batch, and nothing of them is kept here after its
//! batch is added, but they become stored together, when the last batch is in. A failure
//! anywhere (the encoder, the memory budget, the store) rolls the whole fill back: the
//! store, its text index included, is as it was before, and encoding the same logs into
//! it again is safe. [`FillError`] says so.
//!
//! A text the store cannot hold (it has a NUL character, or more than
//! [`MAX_TEXT_BYTES`] bytes) is taken out before anything is encoded and reported by its
//! position in the source, as the input adapters report records they skip. It does not
//! end a batch: a batch is the next usable texts.
//!
//! # Memory (requirement 12)
//!
//! The logs frame is the caller's and is counted by the caller in the budget it passes.
//! On top of that, for as long as it runs, the fill counts:
//!
//! - what the open store keeps ([`Store::retained_bytes`]);
//! - the store's own share of an add ([`Store::add_share_bytes`]: the database, and the
//!   text index for the longest text). It is worked out from the longest text of the
//!   whole frame, checked before the first text is encoded, and held for the whole fill,
//!   between the adds too, because the store keeps that memory while its transaction is
//!   open;
//! - per batch, the bytes of its own rows in the logs frame (not the frame's average
//!   row) and, per row, [`fill_row_bytes`] without the frame's row: the vectors as the
//!   encoder returns them, the rows frame of the batch, and what the store asks per row
//!   while it writes.
//!
//! The number of rows of a batch comes from [`MemoryBudget::plan`] and
//! [`MemoryBudget::next_batch_rows`] (the free memory is read again before each one),
//! which reckon with the frame's average row. A batch whose own rows need more than is
//! left of the memory limit is halved until it fits, down to one row
//! ([`FillReport::batches_cut`]). Whether every row fits a batch of its own is known
//! from the frame and the limit alone, so it is checked before the first text is
//! encoded, and logs that fail it are refused there, naming the row. The smallest and
//! the largest batch in the plan of the report are those that were encoded and added.
//!
//! Whatever the outcome, the budget holds afterwards what it held before.

use l2d_core::polars::prelude::{DataFrame, PolarsError};
use l2d_core::{
    BatchPlan, EncoderIdentity, L2dError, LogsFrame, MemoryBudget, RunRecord, SkippedRecord,
};
use l2d_encoder::{EncoderError, EncoderSession, TextInput};
use l2d_store::{
    ADD_FIXED_BYTES, ADD_ROW_BYTES, ADD_TEXT_FACTOR, Fill, LogFlags, MAX_TEXT_BYTES, Store,
    StoreError, VectorOrigin, rows_frame,
};

/// The key of what was stored in [`RunRecord::settings`] ([`FillReport::record`]).
pub const STORE_KEY: &str = "store";

/// Bytes per row of a batch beside the vectors, the source and the logs frame's own row:
/// the encoder's flags of a text (32), the reference to the text (16), the three flags
/// as the store takes them (3), the row's place in the columns the rows frame adds
/// (offsets, the flags), the batch's own copy of the row's fixed-width values and string
/// views when texts were left out of it (64), the same again in what the store asks for,
/// and [`ADD_ROW_BYTES`].
pub const FILL_ROW_BYTES: u64 = 192 + ADD_ROW_BYTES;

/// What one row of a batch takes while it is encoded and added, for vectors of
/// `dimension` values, a source name of `source_bytes` bytes and logs whose frame takes
/// `logs_row_bytes` per row.
///
/// Three copies of the vector (the encoder's output, the column of the rows frame, and
/// the further copy of its rows the store asks the budget for), the source twice, the
/// frame's row once (the store's further copy; the frame itself is the caller's), and
/// [`FILL_ROW_BYTES`].
///
/// The plan of a fill is made with the average row of the frame for `logs_row_bytes`. A
/// batch is then held to the bytes of its own rows: the bytes its part of the frame has,
/// and this figure with 0 for `logs_row_bytes` for each of its rows.
pub fn fill_row_bytes(dimension: u16, source_bytes: u64, logs_row_bytes: u64) -> u64 {
    12 * u64::from(dimension) + 2 * source_bytes + logs_row_bytes + FILL_ROW_BYTES
}

/// The part of [`fill_row_bytes`] the fill itself holds while the store adds a batch:
/// two copies of the vector, the source once and 160 of the [`FILL_ROW_BYTES`]. The rest
/// is what the store asks the budget for per row.
fn own_row_bytes(dimension: u16, source_bytes: u64) -> u64 {
    8 * u64::from(dimension) + source_bytes + 160
}

/// Whatever turns texts into vectors for a store. The tool has one implementation,
/// [`EncoderSession`] on the NPU; the trait is the seam for tests without the device.
pub trait LogEncoder {
    /// Name, revision and dimension of the encoder: what the store is held to.
    fn identity(&self) -> &EncoderIdentity;

    /// The release of the AMD runtime the vectors are made under.
    fn runtime_release(&self) -> &str;

    /// Appends the vectors of the distinct `texts` to `vectors` (the identity's dimension
    /// values per text, in the order of the texts) and one [`TextInput`] per text to
    /// `flags`. On an error both are left as long as they were.
    fn encode_into(
        &mut self,
        texts: &[&str],
        vectors: &mut Vec<f32>,
        flags: &mut Vec<TextInput>,
    ) -> Result<(), EncoderError>;
}

impl LogEncoder for EncoderSession {
    fn identity(&self) -> &EncoderIdentity {
        &self.info().identity
    }

    fn runtime_release(&self) -> &str {
        &self.info().runtime_release
    }

    fn encode_into(
        &mut self,
        texts: &[&str],
        vectors: &mut Vec<f32>,
        flags: &mut Vec<TextInput>,
    ) -> Result<(), EncoderError> {
        EncoderSession::encode_into(self, texts, vectors, flags)
    }
}

/// What [`encode_into_store`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct FillReport {
    /// The rows of the logs frame: its distinct texts.
    pub logs: u64,
    /// The texts that were encoded and handed to the store: `logs` less the skipped.
    pub encoded: u64,
    /// The logs that became new stored logs.
    pub added: u64,
    /// The logs that were stored already: their occurrences were added.
    pub repeated: u64,
    /// The logs the store cannot hold, by their position in the source, with the reason.
    /// They were not encoded.
    pub skipped: Vec<SkippedRecord>,
    /// Of the encoded texts: those whose part that the encoder reads was cut
    /// ([`TextInput::truncated`]). Stored with the log, as the two that follow are.
    pub truncated: u64,
    /// Those that are longer than the encoder's input as a whole
    /// ([`TextInput::text_longer_than_input`]).
    pub longer_than_input: u64,
    /// Those whose input was empty ([`TextInput::empty`]). They share one vector.
    pub empty_inputs: u64,
    /// The batches: each was one call of the encoder and one add to the store, all of
    /// them in one transaction.
    pub adds: u64,
    /// The batches that were made smaller than the plan's size at that point because
    /// their own rows needed more than was left of the memory limit (their texts are
    /// longer than the frame's average).
    pub batches_cut: u64,
    /// The most the store held of the budget for itself during one of those adds
    /// (`AddReport::held_bytes`); 0 without an add.
    pub add_held_bytes: u64,
    /// The bytes of the longest text that was stored.
    pub longest_text_bytes: u64,
    /// The free memory observed, the limit, the bytes per row ([`fill_row_bytes`] with
    /// the frame's average row), the changes of the batch size the free memory caused,
    /// and the smallest and largest batch that were encoded and added (requirement 12.3,
    /// 12.8): the last batch and a batch that was cut count as what they were.
    pub plan: BatchPlan,
}

impl FillReport {
    /// Puts the batch plan (under `batch_plan`) and what was stored (under
    /// [`STORE_KEY`]: the store's name, encoder revision and runtime release, and the
    /// counts of this report) into the settings of a run record, beside what is there:
    /// the record a session's `finish` returns. The inputs are the caller's to add.
    pub fn record(&self, store: &Store, record: &mut RunRecord) -> Result<(), L2dError> {
        record.record_batch_plan(&self.plan)?;
        let stored = serde_json::json!({
            "name": store.name(),
            "encoder_revision": store.identity().revision,
            "runtime_release": store.runtime_release(),
            "logs": self.logs,
            "encoded": self.encoded,
            "added": self.added,
            "repeated": self.repeated,
            "skipped": self.skipped.len(),
            "truncated": self.truncated,
            "longer_than_input": self.longer_than_input,
            "empty_inputs": self.empty_inputs,
            "adds": self.adds,
            "batches_cut": self.batches_cut,
            "add_held_bytes": self.add_held_bytes,
            "longest_text_bytes": self.longest_text_bytes,
        });
        match &mut record.settings {
            serde_json::Value::Object(settings) => {
                settings.insert(STORE_KEY.to_string(), stored);
                Ok(())
            }
            // `record_batch_plan` made the settings an object or failed.
            _ => Err(L2dError::Runtime {
                detail: "the settings of the run record are not a JSON object".to_string(),
            }),
        }
    }
}

/// What stopped a fill.
#[derive(Debug, Clone, PartialEq)]
pub enum FillCause {
    /// The store: another encoder identity or runtime release, a failed add or commit.
    Store(StoreError),
    /// The encoder failed on a batch.
    Encoder(EncoderError),
    /// The memory budget does not allow it.
    Memory(L2dError),
}

/// Why [`encode_into_store`] stopped. Nothing of the logs is stored then: the fill is
/// one transaction, and the store is as it was before the call.
#[derive(Debug, Clone, PartialEq)]
pub struct FillError {
    /// Boxed, so that the error stays small beside the report.
    pub cause: Box<FillCause>,
    /// The rows of the logs frame, none of which was stored.
    pub logs: u64,
    /// The texts the encoder had turned into vectors when the fill stopped. That work
    /// is lost with the fill; 0 when the fill was refused before the encoder ran, or
    /// when its first batch failed.
    pub encoded: u64,
}

impl FillError {
    /// What the failure left, as the end of the message.
    fn outcome(&self) -> String {
        if self.encoded == 0 {
            format!(
                "; nothing was encoded and nothing of these {} logs was stored: the store is \
                 as it was",
                self.logs
            )
        } else {
            format!(
                "; the fill was rolled back: nothing of these {} logs was stored, although \
                 {} of them had been encoded; the store is as it was before, so the same \
                 logs can be encoded into it again",
                self.logs, self.encoded
            )
        }
    }
}

impl std::fmt::Display for FillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self.cause {
            FillCause::Store(error) => write!(f, "{error}")?,
            FillCause::Encoder(error) => write!(f, "{error}")?,
            FillCause::Memory(error) => write!(f, "{error}")?,
        }
        f.write_str(&self.outcome())
    }
}

impl std::error::Error for FillError {}

impl From<FillError> for L2dError {
    fn from(error: FillError) -> Self {
        let outcome = error.outcome();
        let cause = match *error.cause {
            FillCause::Store(cause) => L2dError::from(cause),
            FillCause::Encoder(cause) => L2dError::from(cause),
            FillCause::Memory(cause) => cause,
        };
        match cause {
            L2dError::Refused { reason } => L2dError::Refused {
                reason: reason + &outcome,
            },
            L2dError::Runtime { detail } => L2dError::Runtime {
                detail: detail + &outcome,
            },
            L2dError::InvalidInput { detail } => L2dError::InvalidInput {
                detail: detail + &outcome,
            },
            missing @ L2dError::Missing { .. } => missing,
        }
    }
}

fn not_logs(error: PolarsError) -> StoreError {
    StoreError::InvalidRows {
        detail: error.to_string(),
    }
}

/// The rows of a logs frame the store can hold, and the others.
struct Usable {
    /// The rows (from 0) of the texts the store cannot hold, ascending.
    unusable: Vec<usize>,
    skipped: Vec<SkippedRecord>,
    /// The bytes of the longest text among the others, and its row (from 1).
    longest_bytes: u64,
    longest_row: usize,
}

fn usable_rows(logs: &LogsFrame) -> Result<Usable, StoreError> {
    let frame = logs.frame();
    let texts = frame
        .column(LogsFrame::TEXT)
        .and_then(|column| column.str())
        .map_err(not_logs)?;
    let positions = frame
        .column(LogsFrame::FIRST_POSITION)
        .and_then(|column| column.u64())
        .map_err(not_logs)?;
    let mut usable = Usable {
        unusable: Vec::new(),
        skipped: Vec::new(),
        longest_bytes: 0,
        longest_row: 0,
    };
    for (row, (text, position)) in texts.iter().zip(positions.iter()).enumerate() {
        let text = text.unwrap_or_default();
        let bytes = text.len() as u64;
        let reason = if bytes > MAX_TEXT_BYTES {
            Some(format!(
                "the text has {bytes} bytes; a stored text has at most {MAX_TEXT_BYTES} bytes"
            ))
        } else {
            text.bytes().position(|byte| byte == 0).map(|at| {
                format!(
                    "the text holds a NUL character (at byte {at}); the text index of a store \
                     reads a text only up to it"
                )
            })
        };
        match reason {
            Some(reason) => {
                usable.unusable.push(row);
                usable.skipped.push(SkippedRecord {
                    position: position.unwrap_or_default(),
                    reason,
                });
            }
            None if bytes > usable.longest_bytes || usable.longest_row == 0 => {
                usable.longest_bytes = bytes;
                usable.longest_row = row + 1;
            }
            None => {}
        }
    }
    Ok(usable)
}

/// The usable rows of a logs frame, cut into batches by what their own bytes need.
struct Batches<'a> {
    frame: &'a DataFrame,
    /// The rows (from 0) that are left out, ascending.
    unusable: &'a [usize],
    /// What a row of a batch needs beside its bytes in the frame ([`fill_row_bytes`]
    /// without the frame's row).
    row_bytes: u64,
    /// What is left of the memory limit for the rows of a batch, beside everything the
    /// fill holds between batches. It is the same for every batch of a fill.
    left: u64,
}

/// One batch: the next usable rows from a row of the frame on.
struct Batch {
    /// Those rows of the logs frame, in order. Its columns share the frame's buffers.
    frame: DataFrame,
    rows: usize,
    /// The first row of the batch (from 0).
    first: usize,
    /// The row of the frame after the last row of the batch.
    next: usize,
}

/// The outcome of sizing a batch.
enum Fit {
    Fits(Batch),
    /// Even the first row alone needs more than is left: the row (from 0) and what it
    /// needs as a batch of its own.
    TooLarge {
        row: usize,
        need: u64,
    },
}

impl Batches<'_> {
    /// Whether `row` is left out.
    fn is_unusable(&self, row: usize) -> bool {
        self.unusable.binary_search(&row).is_ok()
    }

    /// The next `rows` usable rows from row `at` on, as one frame. The caller knows
    /// that so many are left.
    fn take(&self, at: usize, rows: usize) -> Result<Batch, StoreError> {
        let total = self.frame.height();
        let mut frame: Option<DataFrame> = None;
        let mut first = None;
        let mut row = at;
        let mut wanted = rows;
        while wanted > 0 && row < total {
            // The next row that is left out, at or after `row`.
            let skip = self.unusable.partition_point(|&unusable| unusable < row);
            let stop = self.unusable.get(skip).copied().unwrap_or(total);
            if stop == row {
                row += 1;
                continue;
            }
            let end = stop.min(row + wanted);
            let run = self.frame.slice(row as i64, end - row);
            first.get_or_insert(row);
            match &mut frame {
                None => frame = Some(run),
                Some(frame) => {
                    frame.vstack_mut(&run).map_err(not_logs)?;
                }
            }
            wanted -= end - row;
            row = end;
        }
        Ok(Batch {
            frame: frame.unwrap_or_else(|| self.frame.slice(at as i64, 0)),
            rows: rows - wanted,
            first: first.unwrap_or(at),
            next: row,
        })
    }

    /// What a batch needs of the budget while it is encoded and added: the bytes of its
    /// rows in the frame, and the fixed figure per row.
    fn need(&self, batch: &Batch) -> u64 {
        batch.frame.estimated_size() as u64 + batch.rows as u64 * self.row_bytes
    }

    /// The largest batch of at most `rows` usable rows from row `at` on that fits what
    /// is left: `rows`, halved until it fits, down to one row.
    fn fitting(&self, at: usize, rows: usize) -> Result<Fit, StoreError> {
        let mut rows = rows.max(1);
        loop {
            let batch = self.take(at, rows)?;
            let need = self.need(&batch);
            if need <= self.left {
                return Ok(Fit::Fits(batch));
            }
            if batch.rows <= 1 {
                return Ok(Fit::TooLarge {
                    row: batch.first,
                    need,
                });
            }
            rows = batch.rows / 2;
        }
    }

    /// The first row that does not fit a batch of its own, if there is one: found by
    /// cutting the `usable` rows into batches of at most `batch_rows` as a fill does.
    /// When every row is in a batch that fits, every row fits a batch of its own, since
    /// a batch needs no less than any part of it; so a fill that divides the rows
    /// otherwise (the free memory changed) is not refused by the limit either.
    fn first_too_large(
        &self,
        usable: u64,
        batch_rows: u64,
    ) -> Result<Option<(usize, u64)>, StoreError> {
        let mut at = 0usize;
        let mut done = 0u64;
        while done < usable {
            let rows = batch_rows.min(usable - done) as usize;
            match self.fitting(at, rows)? {
                Fit::Fits(batch) => {
                    at = batch.next;
                    done += batch.rows as u64;
                }
                Fit::TooLarge { row, need } => return Ok(Some((row, need))),
            }
        }
        Ok(None)
    }
}

/// Encodes the distinct texts of `logs` with `encoder` and adds them, with their vectors
/// and the three flags the encoder reports for each, to `store` as logs from `source`
/// (see the module description).
///
/// - A store of another encoder identity or of another runtime release than the
///   encoder's is refused, naming both, before a text is encoded
///   ([`StoreError::IdentityMismatch`], [`StoreError::ReleaseMismatch`]).
/// - When the budget does not leave the store's share of the add, the smallest batch of
///   average rows, or a batch of its own for every row, the fill is refused before a
///   text is encoded ([`FillCause::Memory`]).
/// - Texts the store cannot hold are left out and listed in [`FillReport::skipped`].
/// - A log that is stored already (the same source and text) has its occurrences added;
///   nothing else of it changes.
/// - The fill is one transaction of the store: a failure of the encoder, of the budget
///   or of an add in any batch leaves the store as it was before the call, and the
///   same logs can then be encoded into it again ([`FillError`]).
///
/// `logs` holds distinct texts (the input adapters see to that); the encoder is not
/// asked to deduplicate. `source` is stored as given and is part of a log's identity:
/// the caller settles one spelling of it. The stored rows do not depend on the batches.
pub fn encode_into_store<E: LogEncoder>(
    encoder: &mut E,
    store: &mut Store,
    logs: &LogsFrame,
    source: &str,
    budget: &mut MemoryBudget,
) -> Result<FillReport, FillError> {
    let held_before = budget.held_bytes();
    let mut encoded = 0u64;
    let result = fill(
        encoder,
        store,
        logs,
        source,
        budget,
        held_before,
        &mut encoded,
    );
    budget.hold(held_before);
    result.map_err(|cause| FillError {
        cause: Box::new(cause),
        logs: logs.len() as u64,
        encoded,
    })
}

/// The fill. `encoded` counts the texts the encoder has turned into vectors, for the
/// error of a fill that stops.
fn fill<E: LogEncoder>(
    encoder: &mut E,
    store: &mut Store,
    logs: &LogsFrame,
    source: &str,
    budget: &mut MemoryBudget,
    held_before: u64,
    encoded: &mut u64,
) -> Result<FillReport, FillCause> {
    let total = logs.len();
    let identity = encoder.identity().clone();
    let release = encoder.runtime_release().to_string();
    let origin = VectorOrigin::made_under(&identity, &release);
    store.require_origin(origin).map_err(FillCause::Store)?;

    let usable = usable_rows(logs).map_err(FillCause::Store)?;
    let rows = (total - usable.unusable.len()) as u64;
    let source_bytes = source.len() as u64;
    let logs_row_bytes = (logs.frame().estimated_size() as u64).div_ceil((total as u64).max(1));
    let bytes_per_row = fill_row_bytes(identity.dimension, source_bytes, logs_row_bytes);
    let own_row_bytes = own_row_bytes(identity.dimension, source_bytes);

    // What the fill holds between its batches, beside what the caller holds: what the
    // open store keeps, and the store's share of an add for the longest text of the
    // frame. The share is checked before the encoder runs: reading is charged far less
    // per byte than indexing, so logs that were read can still be too much to add.
    let retained = store.retained_bytes();
    let share = store.add_share_bytes(usable.longest_bytes);
    let base = held_before + retained;
    if rows > 0 {
        let left = budget.left_bytes();
        if left < retained + share {
            return Err(FillCause::Memory(L2dError::Refused {
                reason: format!(
                    "adding these logs to the store {} needs {share} bytes for the store \
                     itself ({ADD_FIXED_BYTES} bytes for the database and {ADD_TEXT_FACTOR} \
                     bytes for each byte of the longest text it has to index; the longest of \
                     these logs has {} bytes, row {}) beside the {retained} bytes the open \
                     store keeps, but {left} bytes are left of the memory limit of {} bytes \
                     ({held_before} bytes are held already); a lower limit for the length of \
                     a record makes the add fit",
                    store.name(),
                    usable.longest_bytes,
                    usable.longest_row,
                    budget.limit_bytes()
                ),
            }));
        }
        budget.hold(base + share);
    }
    let mut plan = budget
        .plan(rows, bytes_per_row)
        .map_err(FillCause::Memory)?;
    let mut report = FillReport {
        logs: total as u64,
        encoded: 0,
        added: 0,
        repeated: 0,
        skipped: usable.skipped,
        truncated: 0,
        longer_than_input: 0,
        empty_inputs: 0,
        adds: 0,
        batches_cut: 0,
        add_held_bytes: 0,
        longest_text_bytes: usable.longest_bytes,
        plan: plan.clone(),
    };
    if rows == 0 {
        return Ok(report);
    }

    let texts = logs
        .frame()
        .column(LogsFrame::TEXT)
        .and_then(|column| column.str())
        .map_err(|error| FillCause::Store(not_logs(error)))?;
    // The limit, less what is held between batches, is what a batch may take. It does
    // not change during the fill, so what can never fit is known now.
    let batches = Batches {
        frame: logs.frame(),
        unusable: &usable.unusable,
        row_bytes: fill_row_bytes(identity.dimension, source_bytes, 0),
        left: budget.left_bytes(),
    };
    let too_large = |row: usize, need: u64, budget: &MemoryBudget| {
        FillCause::Memory(L2dError::Refused {
            reason: format!(
                "the log in row {} (a text of {} bytes) needs {need} bytes while it is \
                 encoded and added as a batch of its own, but {} bytes are left of the memory \
                 limit of {} bytes: {held_before} bytes are held already, the open store \
                 keeps {retained} bytes and its share of an add is {share} bytes; a lower \
                 limit for the length of a record or a higher memory limit makes it fit",
                row + 1,
                texts.get(row).map_or(0, str::len),
                batches.left,
                budget.limit_bytes()
            ),
        })
    };
    if let Some((row, need)) = batches
        .first_too_large(rows, plan.batch_rows)
        .map_err(FillCause::Store)?
    {
        return Err(too_large(row, need, budget));
    }

    let dimension = usize::from(identity.dimension);
    // Reused from batch to batch, and no larger than the batch at hand.
    let mut batch_texts: Vec<&str> = Vec::new();
    let mut vectors: Vec<f32> = Vec::new();
    let mut inputs: Vec<TextInput> = Vec::new();
    let mut flags: Vec<LogFlags> = Vec::new();
    // The smallest and the largest batch that were encoded and added.
    let mut smallest = u64::MAX;
    let mut largest = 0u64;
    // One transaction for all batches: dropped without a commit, it is rolled back.
    let mut filling: Fill<'_> = store.begin_fill(origin).map_err(FillCause::Store)?;
    // The next row of the frame, and the rows encoded and added so far.
    let mut at = 0usize;
    let mut done = 0u64;
    while done < rows {
        // The store's share is held while the batch is sized: the add takes it next.
        budget.hold(base + share);
        let planned = budget
            .next_batch_rows(&mut plan, done)
            .map_err(FillCause::Memory)?
            .min(rows - done);
        let batch = match batches
            .fitting(at, planned as usize)
            .map_err(FillCause::Store)?
        {
            Fit::Fits(batch) => batch,
            // Not reached after the check above; refused all the same.
            Fit::TooLarge { row, need } => return Err(too_large(row, need, budget)),
        };
        let take = batch.rows;
        report.batches_cut += u64::from((take as u64) < planned);

        batch_texts.clear();
        vectors.clear();
        inputs.clear();
        flags.clear();
        if batch_texts.capacity() > take {
            // A smaller batch than before: what it does not need is given back, so that
            // what is held is what is counted.
            batch_texts.shrink_to(take);
            vectors.shrink_to(take * dimension);
            inputs.shrink_to(take);
            flags.shrink_to(take);
        }
        batch_texts.extend(
            (batch.first..batch.next)
                .filter(|row| !batches.is_unusable(*row))
                .map(|row| texts.get(row).unwrap_or_default()),
        );
        encoder
            .encode_into(&batch_texts, &mut vectors, &mut inputs)
            .map_err(FillCause::Encoder)?;
        if vectors.len() != take * dimension || inputs.len() != take {
            return Err(FillCause::Encoder(EncoderError::Failed {
                detail: format!(
                    "the encoder returned {} values and {} flags for {take} texts; its \
                     identity says {dimension} values per text",
                    vectors.len(),
                    inputs.len()
                ),
            }));
        }
        *encoded += take as u64;
        flags.extend(inputs.iter().map(|input| LogFlags {
            truncated: input.truncated,
            text_longer_than_input: input.text_longer_than_input,
            empty: input.empty,
        }));

        let frame = LogsFrame::new(batch.frame)
            .map_err(|error| StoreError::InvalidRows {
                detail: error.to_string(),
            })
            .and_then(|batch| rows_frame(&batch, source, &vectors, identity.dimension, &flags))
            .map_err(FillCause::Store)?;
        // What this batch holds here stays counted while the store adds it; the store
        // counts its own share itself.
        budget.hold(base + take as u64 * own_row_bytes);
        let added = filling.add_with(&frame, budget).map_err(FillCause::Store)?;

        report.added += added.added;
        report.repeated += added.repeated;
        report.adds += 1;
        report.add_held_bytes = report.add_held_bytes.max(added.held_bytes);
        report.encoded += take as u64;
        for input in &inputs {
            report.truncated += u64::from(input.truncated);
            report.longer_than_input += u64::from(input.text_longer_than_input);
            report.empty_inputs += u64::from(input.empty);
        }
        smallest = smallest.min(take as u64);
        largest = largest.max(take as u64);
        at = batch.next;
        done += take as u64;
    }
    // Every batch is in: now they are stored, together.
    filling.commit().map_err(FillCause::Store)?;
    // The batches the plan reports are the ones that were used.
    plan.min_rows_used = smallest;
    plan.max_rows_used = largest;
    report.plan = plan;
    Ok(report)
}
