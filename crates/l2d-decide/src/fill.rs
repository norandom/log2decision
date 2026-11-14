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
//! The store has one flag per log, `truncated`: it takes [`TextInput::truncated`] (the
//! part of the text the encoder reads was cut). The second flag of requirement 1.5
//! ([`TextInput::text_longer_than_input`]) and the empty inputs have no column in the
//! store; they are counted in the [`FillReport`].
//!
//! # Batches and what a failure leaves
//!
//! Each batch is one add, and an add is one transaction of the store. A failure in a
//! later batch therefore leaves the earlier batches stored; the error says how many logs
//! that is ([`FillError::stored_logs`]). Adding the same logs again adds their
//! occurrences a second time, as every repeated add does.
//!
//! A text the store cannot hold (it has a NUL character, or more than
//! [`MAX_TEXT_BYTES`] bytes) is taken out before anything is encoded and reported by its
//! position in the source, as the input adapters report records they skip.
//!
//! # Memory (requirement 12)
//!
//! The logs frame is the caller's and is counted by the caller in the budget it passes.
//! On top of that, for as long as it runs, the fill counts:
//!
//! - what the open store keeps ([`Store::retained_bytes`]);
//! - the store's own share of an add ([`Store::add_share_bytes`]: the database, and the
//!   text index for the longest text). It is worked out from the longest text of the
//!   whole frame and checked before the first text is encoded, so that logs whose add
//!   cannot fit are refused before the encoder runs;
//! - per row of a batch, [`fill_row_bytes`]: the vectors as the encoder returns them, the
//!   rows frame of the batch, and what the store asks per row while it writes.
//!
//! The batches come from [`MemoryBudget::plan`] and [`MemoryBudget::next_batch_rows`]
//! (the free memory is read again before each one); the plan is in the report. Whatever
//! the outcome, the budget holds afterwards what it held before.

use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    BatchPlan, EncoderIdentity, L2dError, LogsFrame, MemoryBudget, RunRecord, SkippedRecord,
};
use l2d_encoder::{EncoderError, EncoderSession, TextInput};
use l2d_store::{
    ADD_FIXED_BYTES, ADD_ROW_BYTES, ADD_TEXT_FACTOR, MAX_TEXT_BYTES, Store, StoreError,
    VectorOrigin, rows_frame,
};

/// The key of what was stored in [`RunRecord::settings`] ([`FillReport::record`]).
pub const STORE_KEY: &str = "store";

/// Bytes per row of a batch beside the vectors, the source and the logs frame's own row:
/// the encoder's flags of a text (32), the reference to the text (16), the row's place
/// in the columns the rows frame adds (offsets, the view of the source, the flag), the
/// same again in what the store asks for, and [`ADD_ROW_BYTES`].
pub const FILL_ROW_BYTES: u64 = 128 + ADD_ROW_BYTES;

/// What one row of a batch takes while it is encoded and added, for vectors of
/// `dimension` values, a source name of `source_bytes` bytes and logs whose frame takes
/// `logs_row_bytes` per row.
///
/// Three copies of the vector (the encoder's output, the column of the rows frame, and
/// the further copy of its rows the store asks the budget for), the source twice, the
/// frame's row once (the store's further copy; the frame itself is the caller's), and
/// [`FILL_ROW_BYTES`].
pub fn fill_row_bytes(dimension: u16, source_bytes: u64, logs_row_bytes: u64) -> u64 {
    12 * u64::from(dimension) + 2 * source_bytes + logs_row_bytes + FILL_ROW_BYTES
}

/// The part of [`fill_row_bytes`] the fill itself holds while the store adds a batch:
/// two copies of the vector, the source once and 96 of the [`FILL_ROW_BYTES`]. The rest
/// is what the store asks the budget for per row.
fn own_row_bytes(dimension: u16, source_bytes: u64) -> u64 {
    8 * u64::from(dimension) + source_bytes + 96
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
    /// ([`TextInput::truncated`], the flag that is stored with the log).
    pub truncated: u64,
    /// Those that are longer than the encoder's input as a whole
    /// ([`TextInput::text_longer_than_input`]; not stored).
    pub longer_than_input: u64,
    /// Those whose input was empty ([`TextInput::empty`]; not stored). They share one
    /// vector.
    pub empty_inputs: u64,
    /// The batches: each was one call of the encoder and one add to the store.
    pub adds: u64,
    /// The most the store held of the budget for itself during one of those adds
    /// (`AddReport::held_bytes`); 0 without an add.
    pub add_held_bytes: u64,
    /// The bytes of the longest text that was stored.
    pub longest_text_bytes: u64,
    /// The free memory observed, the limit, the bytes per row ([`fill_row_bytes`]) and
    /// the batch sizes used (requirement 12.3, 12.8).
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
    /// The store: another encoder identity or runtime release, a failed add.
    Store(StoreError),
    /// The encoder failed on a batch.
    Encoder(EncoderError),
    /// The memory budget does not allow it.
    Memory(L2dError),
}

/// Why [`encode_into_store`] stopped, and what it had stored by then.
#[derive(Debug, Clone, PartialEq)]
pub struct FillError {
    /// Boxed, so that the error stays small beside the report.
    pub cause: Box<FillCause>,
    /// The logs of the frame that earlier batches stored, and that stay stored.
    pub stored_logs: u64,
    /// The rows of the logs frame.
    pub logs: u64,
}

impl FillError {
    /// What the failure left in the store, as the end of the message.
    fn outcome(&self) -> String {
        if self.stored_logs == 0 {
            format!("; nothing of these {} logs was stored", self.logs)
        } else {
            format!(
                "; {} of the {} logs were stored by the batches before it and stay stored: \
                 adding the same logs again adds the occurrences of those a second time",
                self.stored_logs, self.logs
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
    let not_logs = |error: l2d_core::polars::prelude::PolarsError| StoreError::InvalidRows {
        detail: error.to_string(),
    };
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

/// Encodes the distinct texts of `logs` with `encoder` and adds them, with their vectors,
/// to `store` as logs from `source` (see the module description).
///
/// - A store of another encoder identity or of another runtime release than the
///   encoder's is refused, naming both, before a text is encoded
///   ([`StoreError::IdentityMismatch`], [`StoreError::ReleaseMismatch`]).
/// - When the budget does not leave the store's share of the add, or the smallest batch,
///   the fill is refused before a text is encoded ([`FillCause::Memory`]).
/// - Texts the store cannot hold are left out and listed in [`FillReport::skipped`].
/// - A log that is stored already (the same source and text) has its occurrences added;
///   nothing else of it changes.
/// - A failure of the encoder or of an add in a later batch leaves the earlier batches
///   stored ([`FillError::stored_logs`]).
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
    let result = fill(encoder, store, logs, source, budget, held_before);
    budget.hold(held_before);
    result
}

fn fill<E: LogEncoder>(
    encoder: &mut E,
    store: &mut Store,
    logs: &LogsFrame,
    source: &str,
    budget: &mut MemoryBudget,
    held_before: u64,
) -> Result<FillReport, FillError> {
    let total = logs.len();
    let stopped = |cause: FillCause, stored_logs: u64| FillError {
        cause: Box::new(cause),
        stored_logs,
        logs: total as u64,
    };
    let identity = encoder.identity().clone();
    let release = encoder.runtime_release().to_string();
    let origin = VectorOrigin::made_under(&identity, &release);
    store
        .require_origin(origin)
        .map_err(|error| stopped(FillCause::Store(error), 0))?;

    let usable = usable_rows(logs).map_err(|error| stopped(FillCause::Store(error), 0))?;
    let rows = (total - usable.unusable.len()) as u64;
    let source_bytes = source.len() as u64;
    let logs_row_bytes = (logs.frame().estimated_size() as u64).div_ceil((total as u64).max(1));
    let bytes_per_row = fill_row_bytes(identity.dimension, source_bytes, logs_row_bytes);
    let own_row_bytes = own_row_bytes(identity.dimension, source_bytes);
    let before_encoding = |error: L2dError| {
        stopped(
            FillCause::Memory(match error {
                L2dError::Refused { reason } => L2dError::Refused {
                    reason: format!("{reason}; nothing was encoded"),
                },
                other => other,
            }),
            0,
        )
    };

    // The store's share of an add, before the encoder runs: reading is charged far less
    // per byte than indexing, so logs that were read can still be too much to add.
    if rows > 0 {
        let retained = store.retained_bytes();
        let share = store.add_share_bytes(usable.longest_bytes);
        let left = budget.left_bytes();
        if left < retained + share {
            return Err(before_encoding(L2dError::Refused {
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
        budget.hold(held_before + retained + share);
    }
    let mut plan = budget.plan(rows, bytes_per_row).map_err(before_encoding)?;
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
        add_held_bytes: 0,
        longest_text_bytes: usable.longest_bytes,
        plan: plan.clone(),
    };

    let not_logs = |error: l2d_core::polars::prelude::PolarsError| StoreError::InvalidRows {
        detail: error.to_string(),
    };
    let texts = logs
        .frame()
        .column(LogsFrame::TEXT)
        .and_then(|column| column.str())
        .map_err(|error| stopped(FillCause::Store(not_logs(error)), 0))?;
    let dimension = usize::from(identity.dimension);
    // Reused from batch to batch; they grow to the largest batch.
    let mut batch_texts: Vec<&str> = Vec::new();
    let mut vectors: Vec<f32> = Vec::new();
    let mut flags: Vec<TextInput> = Vec::new();
    let mut truncated: Vec<bool> = Vec::new();
    let mut unusable = usable.unusable.iter().copied().peekable();
    // The next row of the frame, and the rows encoded and stored so far.
    let mut at = 0usize;
    let mut done = 0u64;
    while done < rows {
        while unusable.peek() == Some(&at) {
            unusable.next();
            at += 1;
        }
        // The store's share is held while the batch is sized: the add takes it next.
        let base = held_before + store.retained_bytes();
        budget.hold(base + store.add_share_bytes(usable.longest_bytes));
        let planned = budget
            .next_batch_rows(&mut plan, done)
            .map_err(|error| stopped(FillCause::Memory(error), done))?
            .min(rows - done);
        // A batch ends before the next text that is left out.
        let end = unusable.peek().copied().unwrap_or(total);
        let take = (planned as usize).min(end - at);

        batch_texts.clear();
        batch_texts.extend((at..at + take).map(|row| texts.get(row).unwrap_or_default()));
        vectors.clear();
        flags.clear();
        encoder
            .encode_into(&batch_texts, &mut vectors, &mut flags)
            .map_err(|error| stopped(FillCause::Encoder(error), done))?;
        if vectors.len() != take * dimension || flags.len() != take {
            return Err(stopped(
                FillCause::Encoder(EncoderError::Failed {
                    detail: format!(
                        "the encoder returned {} values and {} flags for {take} texts; its \
                         identity says {dimension} values per text",
                        vectors.len(),
                        flags.len()
                    ),
                }),
                done,
            ));
        }
        truncated.clear();
        truncated.extend(flags.iter().map(|input| input.truncated));

        let batch: DataFrame = logs.frame().slice(at as i64, take);
        let frame = LogsFrame::new(batch)
            .map_err(|error| StoreError::InvalidRows {
                detail: error.to_string(),
            })
            .and_then(|batch| rows_frame(&batch, source, &vectors, identity.dimension, &truncated))
            .map_err(|error| stopped(FillCause::Store(error), done))?;
        // What this batch holds here stays counted while the store adds it; the store
        // counts its own share itself.
        budget.hold(base + take as u64 * own_row_bytes);
        let added = store
            .add_with(&frame, origin, budget)
            .map_err(|error| stopped(FillCause::Store(error), done))?;

        report.added += added.added;
        report.repeated += added.repeated;
        report.adds += 1;
        report.add_held_bytes = report.add_held_bytes.max(added.held_bytes);
        report.encoded += take as u64;
        for input in &flags {
            report.truncated += u64::from(input.truncated);
            report.longer_than_input += u64::from(input.text_longer_than_input);
            report.empty_inputs += u64::from(input.empty);
        }
        at += take;
        done += take as u64;
    }
    report.plan = plan;
    Ok(report)
}
