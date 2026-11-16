//! Mark sets: the logs an analyst marks on their own input or store (requirements
//! 16.1, 16.3 and 16.12; task 1.21).
//!
//! A mark set is a named selection of logs of one input or one store, with a label the
//! analyst gives it (bad or good). It is the first step of the analyst's flow: mark,
//! group and de-select, redact, fit (design.md, "MarkSet, Redaction and the mark-set
//! score"). The marks come from the analyst's own query on their own logs, never from
//! the labels of a dataset.
//!
//! # Logs, positions and ids
//!
//! A mark set records the kind of source it was made on ([`MarkedSource`]) and names
//! its logs as that source does:
//!
//! - **An input** (a file or an archive): a log is a distinct text, named by its
//!   `first_position`, the position of the first occurrence of its text in the encoding
//!   the input already uses (the 1-based line or data-row index of a file,
//!   [`crate::input`]; the packed entry and line of an archive,
//!   [`crate::zip_input::ArchivePosition`]). The source's content hash is the input's
//!   ([`InputSummary::content_sha256`]); [`MarkSet::require_input`] refuses any other.
//! - **A store** (owner decision of 2026-11-16): a log is named by its id, the SHA-256
//!   of source and text the store gives every log (`l2d_store::log_id`). The store's
//!   content hash is the SHA-256 over the ids of all its logs in sorted order, so the
//!   same set of logs gives the same hash whatever order they were stored in;
//!   [`MarkSet::require_store`] refuses a store whose logs changed. The hash and the
//!   marks are read in two transactions, so marking takes the hash again afterwards and
//!   refuses a store that changed in between.
//!
//! For an archive the entry patterns it was read with are recorded as well, and
//! [`MarkSet::require_input`] refuses the same archive read with other patterns.
//!
//! A mark set of one kind is refused where the other is asked for, with a plain
//! message.
//!
//! # Queries
//!
//! - [`MarkQuery::Phrase`]: the logs whose text contains the phrase, byte for byte. A
//!   phrase is text: no character in it has a special meaning, and case is not
//!   ignored (a regular expression with `(?i)` does that). In a store the phrase goes
//!   through the store's text index ([`MarkStore::phrase_ids`]).
//! - [`MarkQuery::Regex`]: the logs whose text the regular expression matches anywhere
//!   (the syntax of the `regex` crate; no look-around, no back-references, so the time
//!   of a match grows with the text and not beyond). A store is read in batches for it
//!   ([`MarkStore::text_batches`]).
//! - [`MarkQuery::Positions`]: the logs at the positions given, of an input only. A
//!   position that names no log (a repeat of an earlier text, a skipped record, a
//!   position beyond the input) is refused, with the positions named.
//!
//! An input is marked in the batches it is read in: [`MarkSet::mark`] takes the logs
//! frame of a file the caller has read, [`MarkSet::mark_archive`] reads an archive
//! batch by batch ([`ZipInput::batches`]) and never holds it as a whole.
//! [`MarkSet::mark_store`] marks a store through [`MarkStore`], which `l2d-store`
//! implements (this crate cannot depend on it).
//!
//! # De-selection
//!
//! [`MarkSet::deselect`] (positions) and [`MarkSet::deselect_ids`] (ids) record single
//! logs the analyst takes out again (requirement 16.3). They stay in the mark set and in
//! its count, which is the count of the logs marked; [`MarkSet::selected`],
//! [`MarkSet::selected_ids`] and [`MarkSet::marked_in`] leave them out. Groups are
//! de-selected with task 2.8.
//!
//! # The file
//!
//! A mark set is saved as `<project>/marks/<name>.json` ([`ProjectPaths::mark_set_path`])
//! with the query, the label, the source (its kind, path or name, content hash, and the
//! entry patterns of an archive), the count, the marked positions or ids and the
//! de-selected ones. It is written to a temporary file beside it and renamed, so a file
//! is whole or absent. Saving over a file of the same name is allowed only for the same
//! marks (to record a de-selection or redaction rules); other marks under that name are
//! refused.
//!
//! The confirmed redaction rules ([`RedactionRule`], task 1.22) are kept in the same
//! file, in order. A file with rules has format [`FORMAT_WITH_REDACTIONS`]; one without
//! is format [`FORMAT`] and unchanged from task 1.21. What the rules do is
//! `l2d_baseline::redact`'s; this module keeps them and checks their shape.
//!
//! # Memory (requirements 12 and 15)
//!
//! What marking holds is counted against the memory budget, and the query's share is
//! asked for before anything is built:
//!
//! - the query: the phrase twice, the compiled regular expression
//!   ([`REGEX_HELD_BYTES`], with its size limits set below it), or the positions asked
//!   for ([`WANTED_POSITION_BYTES`] each);
//! - the marked positions, collected in chunks of [`CHUNK_POSITIONS`] so that no list
//!   is copied as it grows, and at the end copied once into the sorted list of the mark
//!   set (8 bytes each, beside the chunks);
//! - the marked ids of a store, [`MARKED_ID_BYTES`] each;
//! - saving: the file is written as a stream ([`SAVE_BUFFER_BYTES`]), and a file of the
//!   same name is read for the comparison only when [`LOAD_FACTOR`] times its size fits.
//!
//! For a file, the logs frame is the caller's and already counted as held; marking
//! takes what is left. For an archive the archive is opened first and the query's share
//! put beside what it then holds; the query is built only when that, with the fixed
//! memory of a batch, fits. The marked positions are held beside it
//! ([`ZipInput::hold_beside`], then [`ZipBatches::hold_beside`] after every batch) with
//! room for the positions of one more batch of the largest size, so the batches shrink
//! as they grow and the read is refused, with the figures, when they no longer fit. For
//! a store, the query's share is held in the budget while the store works; a phrase
//! search counts its matches first and asks for their ids before it hands out one; the
//! batches of a regular expression are cut so that each, with an id for each of its
//! rows, fits beside the ids marked so far.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use polars::prelude::{BooleanChunked, Column, DataFrame, NewChunkedArray};
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use crate::error::L2dError;
use crate::input::{PARSE_BATCH_BYTES, frame_error};
use crate::memory::{MemoryBudget, PROCESS_BASELINE_BYTES, exact};
use crate::paths::ProjectPaths;
use crate::types::{InputSummary, Label, LogsFrame};
use crate::zip_input::{ArchiveSummary, ZipBatches, ZipInput, ZipInputSettings};

/// The format number of a mark-set file without redaction rules.
pub const FORMAT: u32 = 1;

/// The format number of a mark-set file that holds redaction rules (task 1.22). A
/// reader that knows only [`FORMAT`] refuses such a file instead of passing over its
/// rules; a mark set without rules is still written as format 1, byte for byte as
/// before.
pub const FORMAT_WITH_REDACTIONS: u32 = 2;

/// The most redaction rules a mark set keeps.
pub const MAX_REDACTIONS: usize = 1024;

/// The longest placeholder of a redaction rule, in bytes.
pub const MAX_PLACEHOLDER_BYTES: usize = 64;

/// The kind a mark-set file names.
pub const KIND: &str = "mark-set";

/// The longest mark-set name, in bytes.
pub const MAX_NAME_BYTES: usize = 64;

/// The longest regular expression, in bytes.
pub const MAX_PATTERN_BYTES: usize = 64 * 1024;

/// The size limit of a compiled regular expression (`RegexBuilder::size_limit`).
pub const REGEX_SIZE_LIMIT: usize = 1024 * 1024;

/// The size limit of the lazy DFA's cache (`RegexBuilder::dfa_size_limit`).
pub const REGEX_DFA_SIZE_LIMIT: usize = 1024 * 1024;

/// What a compiled regular expression is counted as: its program and the caches of its
/// engines, each kept below the two limits above, several times over.
pub const REGEX_HELD_BYTES: u64 = 8 * 1024 * 1024;

/// What each position asked for by [`MarkQuery::Positions`] is counted as: the list in
/// the query, its sorted copy, and whether it was found.
pub const WANTED_POSITION_BYTES: u64 = 24;

/// How many marked positions one chunk holds.
pub const CHUNK_POSITIONS: usize = 4096;

const CHUNK_BYTES: u64 = 8 * CHUNK_POSITIONS as u64;

/// The length of a store's log id: 64 hexadecimal characters.
pub const ID_CHARS: usize = 64;

/// What each marked id of a store is counted as: its 64 characters, and its string in
/// the list three times over (the list doubles as it grows, and holds the old and the
/// new room for a moment).
pub const MARKED_ID_BYTES: u64 = ID_CHARS as u64 + 3 * size_of::<String>() as u64;

/// What loading a mark-set file is counted as, per byte of the file: the file, and the
/// lists parsed from it (a position takes at least two bytes in the file and eight in
/// memory, three times over while its list grows; an id 66 bytes in the file and 88 in
/// memory).
pub const LOAD_FACTOR: u64 = 16;

/// The buffer a mark set is written through, in bytes.
const SAVE_WRITE_BYTES: usize = 8 * 1024;

/// What saving a mark set is counted as, whatever its size: the write buffer and the
/// serialiser's own small state, twice over.
pub const SAVE_BUFFER_BYTES: u64 = 2 * SAVE_WRITE_BYTES as u64;

/// How many positions or ids a refusal names at most.
const NAMED_POSITIONS: usize = 10;

/// How much of a pattern or of its error a message shows, in bytes.
const SHOWN_BYTES: usize = 300;

/// Temporary files of saves in flight get distinct names.
static SAVES: AtomicU64 = AtomicU64::new(0);

/// How the analyst chose the logs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MarkQuery {
    /// The logs whose text contains the phrase, byte for byte.
    Phrase { phrase: String },
    /// The logs whose text the regular expression matches anywhere.
    Regex { pattern: String },
    /// The logs at these positions (first positions, in the input's encoding).
    Positions { positions: Vec<u64> },
}

/// The input a mark set was made on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkedInput {
    /// The path the input was read from, as given.
    pub path: String,
    /// The SHA-256 of the input file (of the archive file for an archive).
    pub content_sha256: String,
    /// For an archive, the entry patterns it was read with (empty: every entry); `None`
    /// for a file.
    pub archive_entries: Option<Vec<String>>,
}

/// The store a mark set was made on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkedStore {
    /// The store's name (`stores/<name>.db`).
    pub name: String,
    /// The SHA-256 over the ids of all its logs in sorted order.
    pub content_sha256: String,
}

/// What a mark set was made on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MarkedSource {
    Input(MarkedInput),
    Store(MarkedStore),
}

/// How a redaction rule finds what it replaces (task 1.22).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionKind {
    /// The fragment as text, where it does not continue a token on either side
    /// (`l2d_baseline::redact` states the rule).
    Literal,
    /// The fragment is a regular expression (the syntax of the `regex` crate).
    Pattern,
}

/// One confirmed, edited or added redaction (requirement 16.5): what is replaced, and
/// by what. The rules of a mark set are kept in order; `l2d_baseline::redact` applies
/// them and states what they do. This crate keeps them with the mark set and checks
/// only their shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionRule {
    pub kind: RedactionKind,
    /// The text or the pattern: 1 to [`MAX_PATTERN_BYTES`] bytes.
    pub fragment: String,
    /// What replaces it: at most [`MAX_PLACEHOLDER_BYTES`] bytes (empty deletes it).
    pub placeholder: String,
}

/// What marking needs of a store. `l2d-store` implements it for its `Store`; this
/// crate defines it because it cannot depend on that one (owner decision of
/// 2026-11-16).
///
/// Every method works within `budget` on top of what the caller holds in it, and gives
/// the budget back as it found it, whatever the outcome.
pub trait MarkStore {
    /// The store's name.
    fn store_name(&self) -> &str;

    /// The SHA-256, in lower-case hexadecimal, over the ids of all stored logs in
    /// sorted order, read as a stream: nothing grows with the number of logs.
    fn content_sha256(&self, budget: &mut MemoryBudget) -> Result<String, L2dError>;

    /// Hands `each` the id of every stored log whose text contains `phrase` byte for
    /// byte, found through the store's text index. Before it hands out the first, the
    /// store counts the matches and asks the budget for `id_bytes` for each beside what
    /// its search holds; [`L2dError::Refused`] with the figures when that does not fit.
    fn phrase_ids(
        &self,
        phrase: &str,
        id_bytes: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&str),
    ) -> Result<(), L2dError>;

    /// Hands `each` the stored logs in batches, in the order they were stored, as frames
    /// of `seq` (Int64), `id` and `text` (String). A batch is cut so that it, and
    /// `per_row` bytes for each of its rows, fit beside what the read holds and what the
    /// caller holds beside it: `beside` before the first batch, and after each batch
    /// what `each` returns. [`L2dError::Refused`] with the figures when not even one
    /// row of the longest text fits.
    fn text_batches(
        &self,
        beside: u64,
        per_row: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError>;
}

/// A named set of marked logs of one input or one store (see the module description).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkSet {
    name: String,
    label: Label,
    query: MarkQuery,
    source: MarkedSource,
    /// The marked positions of an input, ascending; empty for a store.
    positions: Vec<u64>,
    deselected: Vec<u64>,
    /// The marked ids of a store, ascending; empty for an input.
    ids: Vec<String>,
    deselected_ids: Vec<String>,
    /// The confirmed redaction rules, in order (task 1.22); empty until the analyst
    /// confirms some.
    redactions: Vec<RedactionRule>,
}

/// The file as it is written.
#[derive(Serialize)]
struct FileOut<'a> {
    format: u32,
    kind: &'static str,
    name: &'a str,
    label: Label,
    query: &'a MarkQuery,
    source: &'a MarkedSource,
    count: u64,
    positions: &'a [u64],
    deselected: &'a [u64],
    ids: &'a [String],
    deselected_ids: &'a [String],
    #[serde(skip_serializing_if = "<[RedactionRule]>::is_empty")]
    redactions: &'a [RedactionRule],
}

/// The file as it is read.
#[derive(Deserialize)]
struct FileIn {
    format: u32,
    kind: String,
    name: String,
    label: Label,
    query: MarkQuery,
    source: MarkedSource,
    count: u64,
    positions: Vec<u64>,
    deselected: Vec<u64>,
    ids: Vec<String>,
    deselected_ids: Vec<String>,
    #[serde(default)]
    redactions: Vec<RedactionRule>,
}

impl MarkSet {
    /// Marks the logs of a file the caller has read (`logs` and `summary` from an input
    /// adapter) by `query`, as the mark set `name` with `label`.
    ///
    /// The frame is the caller's and counted as held in `budget`; marking holds the
    /// query and the marked positions on top and is refused, with the figures, when they
    /// do not fit what `budget` leaves ([`L2dError::Refused`]).
    ///
    /// [`L2dError::InvalidInput`]: a name that is not 1 to [`MAX_NAME_BYTES`] letters,
    /// digits, `.`, `_` or `-` (not starting with a dot), a label other than bad or
    /// good, an empty phrase or pattern, a pattern that does not compile, no positions,
    /// or positions that name no log of the input.
    pub fn mark(
        name: &str,
        label: Label,
        query: MarkQuery,
        logs: &LogsFrame,
        summary: &InputSummary,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        check_name(name)?;
        check_label(label)?;
        let fixed = Matcher::check(&query)?;
        let left = budget.left_bytes();
        if fixed > left {
            return Err(query_refused(name, fixed, left, budget));
        }
        let mut matcher = Matcher::new(&query)?;
        let mut collected = Collected::new(left - fixed);
        collected
            .take(&mut matcher, logs)?
            .map_err(|needed| positions_refused(name, &collected, needed, fixed, left, budget))?;
        matcher.require_all_found()?;
        drop(matcher);
        let positions = collected.into_sorted().map_err(|(collected, needed)| {
            positions_refused(name, &collected, needed, fixed, left, budget)
        })?;
        Ok(Self::of_positions(
            name,
            label,
            query,
            MarkedInput {
                path: summary.source_path.to_string_lossy().into_owned(),
                content_sha256: summary.content_sha256.clone(),
                archive_entries: None,
            },
            positions,
        ))
    }

    /// Marks the logs of the archive at `path` by `query`, reading it in batches within
    /// `budget` with `settings`, as the mark set `name` with `label`. Returns the mark
    /// set and the archive's summary (with its entry reports, damaged entries among
    /// them, for the caller to report).
    ///
    /// The archive is never held as a whole: each batch is marked and let go. The
    /// deduplication of the archive stays in memory (no work directory is used); an
    /// archive whose distinct texts do not fit is refused as its read is
    /// (requirement 13.12). Errors as [`MarkSet::mark`] and as [`ZipInput::open`] and
    /// [`ZipInput::batches`]; a refusal of the read while positions are held says so.
    pub fn mark_archive(
        name: &str,
        label: Label,
        query: MarkQuery,
        path: &Path,
        settings: &ZipInputSettings,
        budget: &mut MemoryBudget,
    ) -> Result<(Self, ArchiveSummary), L2dError> {
        check_name(name)?;
        check_label(label)?;
        let fixed = Matcher::check(&query)?;
        let left = budget.left_bytes();
        if fixed > left {
            return Err(query_refused(name, fixed, left, budget));
        }
        // The archive is opened (its directory read and counted) and the query's share
        // put beside it before the query is built: a regular expression is compiled
        // only when it fits with the directory and the fixed memory of a batch.
        let max_rows = budget.max_batch_rows();
        let mut input = ZipInput::open(path, settings, budget)?;
        let mut collected = Collected::new(0);
        let mut beside = fixed + collected.reserve_for(max_rows);
        input.hold_beside(beside);
        let needed = input.held_bytes() + PARSE_BATCH_BYTES;
        if needed > budget.usable_bytes() {
            return Err(L2dError::Refused {
                reason: format!(
                    "marking `{name}` needs {} before it builds its query: {} for the query, \
                     {} for the positions of a first batch, {} that the archive holds once its \
                     directory is read, and {} for the fixed memory of a batch; but the memory \
                     limit of {} leaves {} beside the {} kept for the tool itself; nothing was \
                     marked",
                    exact(needed),
                    exact(fixed),
                    exact(beside - fixed),
                    exact(input.held_bytes() - beside),
                    exact(PARSE_BATCH_BYTES),
                    exact(budget.limit_bytes()),
                    exact(budget.usable_bytes()),
                    exact(PROCESS_BASELINE_BYTES)
                ),
            });
        }
        let mut matcher = Matcher::new(&query)?;
        {
            let mut batches: ZipBatches<'_> = input.batches(budget);
            while let Some(batch) = batches.next() {
                let batch = batch.map_err(|error| read_refused(error, name, &collected, beside))?;
                collected
                    .take(&mut matcher, &batch)?
                    .map_err(|needed| L2dError::Refused {
                        reason: format!(
                            "marking `{name}` took more marked positions from one batch of \
                             the archive than it had kept room for: {} needed, {} kept; \
                             nothing was saved",
                            exact(needed),
                            exact(collected.room)
                        ),
                    })?;
                drop(batch);
                beside = fixed + collected.reserve_for(max_rows);
                batches.hold_beside(beside);
            }
        }
        matcher.require_all_found()?;
        // The positions are copied into one sorted list while the archive is still
        // held: what the archive holds besides them stays counted.
        let archive_held = input.held_bytes().saturating_sub(beside);
        let usable = budget.usable_bytes();
        collected.room = usable.saturating_sub(archive_held + fixed);
        let positions = collected.into_sorted().map_err(|(collected, needed)| {
            positions_refused(
                name,
                &collected,
                needed,
                fixed,
                usable.saturating_sub(archive_held),
                budget,
            )
        })?;
        drop(matcher);
        let summary = input.finish();
        let set = Self::of_positions(
            name,
            label,
            query,
            MarkedInput {
                path: path.to_string_lossy().into_owned(),
                content_sha256: summary.input.content_sha256.clone(),
                archive_entries: Some(settings.entries.clone()),
            },
            positions,
        );
        Ok((set, summary))
    }

    /// Marks the logs of `store` by `query` (a phrase through its text index, a regular
    /// expression over its logs read in batches), as the mark set `name` with `label`.
    /// The logs are named by their ids, and the store's content hash is taken first.
    ///
    /// Errors as [`MarkSet::mark`]; positions are [`L2dError::InvalidInput`] here (they
    /// name the logs of an input). What the store refuses for memory, it refuses with
    /// the figures; the budget holds afterwards what it held before.
    pub fn mark_store(
        name: &str,
        label: Label,
        query: MarkQuery,
        store: &dyn MarkStore,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        check_name(name)?;
        check_label(label)?;
        if let MarkQuery::Positions { .. } = query {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "the logs of the store {} are marked by a phrase or a regular expression; \
                     positions name the logs of an input",
                    store.store_name()
                ),
            });
        }
        let fixed = Matcher::check(&query)?;
        let left = budget.left_bytes();
        if fixed > left {
            return Err(query_refused(name, fixed, left, budget));
        }
        let content_sha256 = store.content_sha256(budget)?;
        let held = budget.held_bytes();
        budget.hold(held + fixed);
        let marked = Self::store_ids(name, &query, store, budget);
        budget.hold(held);
        let mut ids = marked?;
        ids.sort_unstable();
        ids.dedup();
        // The hash and the marks are read in two transactions: the hash is taken again,
        // with the ids held, and a store whose logs changed in between is refused.
        budget.hold(held + MARKED_ID_BYTES * ids.len() as u64);
        let after = store.content_sha256(budget);
        budget.hold(held);
        let after = after?;
        if after != content_sha256 {
            return Err(L2dError::Refused {
                reason: format!(
                    "the store {} changed while `{name}` was marked: its content hash was {} \
                     before and is {} after; nothing was saved; mark it again",
                    store.store_name(),
                    content_sha256,
                    after
                ),
            });
        }
        Ok(Self {
            name: name.to_string(),
            label,
            query,
            source: MarkedSource::Store(MarkedStore {
                name: store.store_name().to_string(),
                content_sha256,
            }),
            positions: Vec::new(),
            deselected: Vec::new(),
            ids,
            deselected_ids: Vec::new(),
            redactions: Vec::new(),
        })
    }

    /// The ids of the logs of `store` that `query` marks, with the query's share held.
    fn store_ids(
        name: &str,
        query: &MarkQuery,
        store: &dyn MarkStore,
        budget: &mut MemoryBudget,
    ) -> Result<Vec<String>, L2dError> {
        let mut ids: Vec<String> = Vec::new();
        match query {
            MarkQuery::Phrase { phrase } => {
                store.phrase_ids(phrase, MARKED_ID_BYTES, budget, &mut |id| {
                    ids.push(id.to_string())
                })?;
            }
            MarkQuery::Regex { .. } => {
                let mut matcher = Matcher::new(query)?;
                let failed = |error| frame_error("marking the logs of a batch", &error);
                store
                    .text_batches(0, MARKED_ID_BYTES, budget, &mut |batch| {
                        let found = batch.column("id").and_then(Column::str).map_err(failed)?;
                        let texts = batch
                            .column(LogsFrame::TEXT)
                            .and_then(Column::str)
                            .map_err(failed)?;
                        for (id, text) in found.into_no_null_iter().zip(texts.into_no_null_iter()) {
                            if matcher.matches(0, text) {
                                ids.push(id.to_string());
                            }
                        }
                        Ok(MARKED_ID_BYTES * ids.len() as u64)
                    })
                    .map_err(|error| match error {
                        L2dError::Refused { reason } => L2dError::Refused {
                            reason: format!(
                                "marking `{name}` holds the ids of {} logs marked so far: \
                                 {reason}; nothing was saved",
                                ids.len()
                            ),
                        },
                        other => other,
                    })?;
            }
            MarkQuery::Positions { .. } => {}
        }
        Ok(ids)
    }

    fn of_positions(
        name: &str,
        label: Label,
        query: MarkQuery,
        input: MarkedInput,
        positions: Vec<u64>,
    ) -> Self {
        Self {
            name: name.to_string(),
            label,
            query,
            source: MarkedSource::Input(input),
            positions,
            deselected: Vec::new(),
            ids: Vec::new(),
            deselected_ids: Vec::new(),
            redactions: Vec::new(),
        }
    }

    /// The name of the mark set.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The label the analyst gave the marked logs: bad or good.
    pub fn label(&self) -> Label {
        self.label
    }

    /// The query the logs were marked by.
    pub fn query(&self) -> &MarkQuery {
        &self.query
    }

    /// What the logs were marked on.
    pub fn source(&self) -> &MarkedSource {
        &self.source
    }

    /// The input the logs were marked on; `None` for a mark set of a store.
    pub fn input(&self) -> Option<&MarkedInput> {
        match &self.source {
            MarkedSource::Input(input) => Some(input),
            MarkedSource::Store(_) => None,
        }
    }

    /// The store the logs were marked on; `None` for a mark set of an input.
    pub fn store(&self) -> Option<&MarkedStore> {
        match &self.source {
            MarkedSource::Store(store) => Some(store),
            MarkedSource::Input(_) => None,
        }
    }

    /// The marked positions of an input, ascending, de-selected ones included; empty
    /// for a mark set of a store.
    pub fn positions(&self) -> &[u64] {
        &self.positions
    }

    /// The marked ids of a store, ascending, de-selected ones included; empty for a
    /// mark set of an input.
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// How many logs were marked, de-selected ones included.
    pub fn count(&self) -> u64 {
        (self.positions.len() + self.ids.len()) as u64
    }

    /// The de-selected positions, ascending.
    pub fn deselected(&self) -> &[u64] {
        &self.deselected
    }

    /// The de-selected ids, ascending.
    pub fn deselected_ids(&self) -> &[String] {
        &self.deselected_ids
    }

    /// The marked positions that are not de-selected, ascending.
    pub fn selected(&self) -> impl Iterator<Item = u64> + '_ {
        self.positions
            .iter()
            .copied()
            .filter(|position| self.deselected.binary_search(position).is_err())
    }

    /// The marked ids that are not de-selected, ascending.
    pub fn selected_ids(&self) -> impl Iterator<Item = &str> + '_ {
        self.ids
            .iter()
            .filter(|id| self.deselected_ids.binary_search(id).is_err())
            .map(String::as_str)
    }

    /// How many marked logs are not de-selected.
    pub fn selected_count(&self) -> u64 {
        self.count() - (self.deselected.len() + self.deselected_ids.len()) as u64
    }

    /// Whether the log at `position` of the input is marked and not de-selected.
    pub fn is_selected(&self, position: u64) -> bool {
        self.positions.binary_search(&position).is_ok()
            && self.deselected.binary_search(&position).is_err()
    }

    /// Whether the stored log `id` is marked and not de-selected.
    pub fn is_selected_id(&self, id: &str) -> bool {
        let found = |ids: &[String]| {
            ids.binary_search_by(|marked| marked.as_str().cmp(id))
                .is_ok()
        };
        found(&self.ids) && !found(&self.deselected_ids)
    }

    /// The redaction rules kept with the mark set, in order (task 1.22); empty when none
    /// were confirmed.
    pub fn redactions(&self) -> &[RedactionRule] {
        &self.redactions
    }

    /// Keeps `rules` with the mark set in place of those it had, in their order
    /// (requirement 16.5); [`MarkSet::save`] then writes them with it. A mark set saved
    /// with rules is written as format [`FORMAT_WITH_REDACTIONS`]. Saving over the file
    /// of the same marks with other rules is allowed, as for de-selections.
    ///
    /// [`L2dError::InvalidInput`] for more than [`MAX_REDACTIONS`] rules, an empty or
    /// too long fragment, or a too long placeholder; nothing changes then. That a
    /// pattern compiles is checked where the rules are applied
    /// (`l2d_baseline::redact::RedactionRules`).
    pub fn set_redactions(&mut self, rules: Vec<RedactionRule>) -> Result<(), L2dError> {
        check_redactions(&rules)?;
        self.redactions = rules;
        Ok(())
    }

    /// De-selects single marked logs of an input by their positions (requirement
    /// 16.3): they stay in the mark set and are recorded as de-selected. De-selecting
    /// one again changes nothing.
    ///
    /// [`L2dError::InvalidInput`] naming the positions that are not marked, or for a
    /// mark set of a store; nothing is de-selected then.
    pub fn deselect(&mut self, positions: &[u64]) -> Result<(), L2dError> {
        if let MarkedSource::Store(store) = &self.source {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "the mark set `{}` was made on the store {} and names its logs by id, not \
                     by position; nothing was de-selected",
                    self.name, store.name
                ),
            });
        }
        let mut unmarked: Vec<u64> = positions
            .iter()
            .copied()
            .filter(|position| self.positions.binary_search(position).is_err())
            .collect();
        if !unmarked.is_empty() {
            unmarked.sort_unstable();
            unmarked.dedup();
            return Err(not_marked(&self.name, "position", &unmarked));
        }
        self.deselected.extend_from_slice(positions);
        self.deselected.sort_unstable();
        self.deselected.dedup();
        Ok(())
    }

    /// De-selects single marked logs of a store by their ids, as [`MarkSet::deselect`]
    /// does by positions. [`L2dError::InvalidInput`] naming the ids that are not
    /// marked, or for a mark set of an input; nothing is de-selected then.
    pub fn deselect_ids(&mut self, ids: &[String]) -> Result<(), L2dError> {
        if let MarkedSource::Input(input) = &self.source {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "the mark set `{}` was made on the input {} and names its logs by \
                     position, not by id; nothing was de-selected",
                    self.name, input.path
                ),
            });
        }
        let mut unmarked: Vec<&String> = ids
            .iter()
            .filter(|id| self.ids.binary_search(id).is_err())
            .collect();
        if !unmarked.is_empty() {
            unmarked.sort_unstable();
            unmarked.dedup();
            return Err(not_marked(&self.name, "id", &unmarked));
        }
        self.deselected_ids.extend_from_slice(ids);
        self.deselected_ids.sort_unstable();
        self.deselected_ids.dedup();
        Ok(())
    }

    /// Checks that `summary` is of the input the mark set was made on, read the same
    /// way: the same content hash, and for an archive the same entry patterns
    /// (`archive_entries`: the patterns it is read with now, `None` for a file).
    /// [`L2dError::Refused`] naming both otherwise: the positions of the mark set name
    /// other logs, or none, in another input, and in the same archive read with other
    /// patterns (the patterns decide which entries are read, so which logs are unmarked
    /// and where a text is first seen). Refused as well for a mark set of a store.
    pub fn require_input(
        &self,
        summary: &InputSummary,
        archive_entries: Option<&[String]>,
    ) -> Result<(), L2dError> {
        let input = match &self.source {
            MarkedSource::Input(input) => input,
            MarkedSource::Store(store) => {
                return Err(L2dError::Refused {
                    reason: format!(
                        "the mark set `{}` was made on the store {}, not on an input; its \
                         logs are named by their ids there and cannot be applied to the \
                         input {}",
                        self.name,
                        store.name,
                        summary.source_path.display()
                    ),
                });
            }
        };
        let read_as = |entries: Option<&[String]>| match entries {
            None => "a file".to_string(),
            Some([]) => "an archive with every entry".to_string(),
            Some(patterns) => format!("an archive with the entry patterns {}", patterns.join(", ")),
        };
        if summary.content_sha256 == input.content_sha256
            && input.archive_entries.as_deref() != archive_entries
        {
            return Err(L2dError::Refused {
                reason: format!(
                    "the mark set `{}` was made on {} read as {}; it is read now as {}, so \
                     its positions may name other logs; read it as before, or mark it again",
                    self.name,
                    input.path,
                    read_as(input.archive_entries.as_deref()),
                    read_as(archive_entries)
                ),
            });
        }
        if summary.content_sha256 == input.content_sha256 {
            return Ok(());
        }
        Err(L2dError::Refused {
            reason: format!(
                "the mark set `{}` was made on the input {} with the content hash {}; the \
                 input {} has the content hash {}, so its positions may name other logs; \
                 mark that input again",
                self.name,
                input.path,
                input.content_sha256,
                summary.source_path.display(),
                summary.content_sha256
            ),
        })
    }

    /// Checks that `store` holds the logs the mark set was made on, by its content hash
    /// (taken now, within `budget`). [`L2dError::Refused`] naming both hashes when the
    /// store's logs changed, and for a mark set of an input.
    pub fn require_store(
        &self,
        store: &dyn MarkStore,
        budget: &mut MemoryBudget,
    ) -> Result<(), L2dError> {
        let marked = match &self.source {
            MarkedSource::Store(marked) => marked,
            MarkedSource::Input(input) => {
                return Err(L2dError::Refused {
                    reason: format!(
                        "the mark set `{}` was made on the input {}, not on a store; its \
                         logs are named by their positions there and cannot be applied to \
                         the store {}",
                        self.name,
                        input.path,
                        store.store_name()
                    ),
                });
            }
        };
        let now = store.content_sha256(budget)?;
        if now == marked.content_sha256 {
            return Ok(());
        }
        Err(L2dError::Refused {
            reason: format!(
                "the mark set `{}` was made on the store {} when its logs had the content \
                 hash {}; the store {} now has the content hash {}: its logs changed, so \
                 the marks may not cover them; mark the store again",
                self.name,
                marked.name,
                marked.content_sha256,
                store.store_name(),
                now
            ),
        })
    }

    /// The rows of `batch` (a logs frame of the input, or one batch of it) that are
    /// marked and not de-selected, in the batch's order. It does not check the input:
    /// call [`MarkSet::require_input`] first. For a mark set of a store, which names no
    /// position, no row is marked.
    pub fn marked_in(&self, batch: &LogsFrame) -> Result<LogsFrame, L2dError> {
        let failed = |error| frame_error("selecting the marked logs of a batch", &error);
        let frame = batch.frame();
        let positions = frame
            .column(LogsFrame::FIRST_POSITION)
            .and_then(Column::u64)
            .map_err(failed)?;
        let keep: Vec<bool> = positions
            .into_no_null_iter()
            .map(|position| {
                self.positions.binary_search(&position).is_ok()
                    && self.deselected.binary_search(&position).is_err()
            })
            .collect();
        let kept = frame
            .filter(&BooleanChunked::from_slice("marked".into(), &keep))
            .map_err(failed)?;
        LogsFrame::new(kept)
    }

    /// The mark set as its file holds it.
    fn file_out(&self) -> FileOut<'_> {
        FileOut {
            format: if self.redactions.is_empty() {
                FORMAT
            } else {
                FORMAT_WITH_REDACTIONS
            },
            kind: KIND,
            name: &self.name,
            label: self.label,
            query: &self.query,
            source: &self.source,
            count: self.count(),
            positions: &self.positions,
            deselected: &self.deselected,
            ids: &self.ids,
            deselected_ids: &self.deselected_ids,
            redactions: &self.redactions,
        }
    }

    /// The bytes [`MarkSet::save`] writes for this mark set: equal mark sets give equal
    /// bytes. A score fitted from the mark set keeps them as its copy of the mark set
    /// (task 1.23). They take about the size of the file in memory.
    pub fn to_file_bytes(&self) -> Result<Vec<u8>, L2dError> {
        serde_json::to_vec_pretty(&self.file_out()).map_err(|error| L2dError::Runtime {
            detail: format!("describing the mark set `{}` failed: {error}", self.name),
        })
    }

    /// The mark set `name` from the bytes of its file, checked as [`MarkSet::load`]
    /// checks a file; `path` names where the bytes came from in a refusal. The caller
    /// counts the memory ([`LOAD_FACTOR`] times the bytes).
    pub fn from_file_bytes(bytes: &[u8], path: &Path, name: &str) -> Result<Self, L2dError> {
        parse_file(bytes, path, name)
    }

    /// Saves the mark set as `<project>/marks/<name>.json` and returns the path.
    ///
    /// The file is whole or absent. When a file of that name holds the same marks
    /// (name, label, query, source and the marked positions or ids), it is replaced,
    /// with the de-selections and redaction rules of this one; when it holds anything
    /// else, the save is
    /// [`L2dError::Refused`] and the file is left as it is.
    ///
    /// Memory: the file is written as a stream through a buffer of
    /// [`SAVE_BUFFER_BYTES`]; the comparison read of a file of the same name is counted
    /// as [`MarkSet::load`] counts it ([`LOAD_FACTOR`] times its size). Either is
    /// refused with the figures when it does not fit what `budget` leaves, and nothing
    /// is written then.
    pub fn save(&self, paths: &ProjectPaths, budget: &MemoryBudget) -> Result<PathBuf, L2dError> {
        check_name(&self.name)?;
        let dir = paths.marks_dir();
        let path = paths.mark_set_path(&self.name);
        let operation = format!("saving the mark set `{}`", self.name);
        if let Ok(metadata) = fs::metadata(&path) {
            budget.require_whole(metadata.len().saturating_mul(LOAD_FACTOR), &operation)?;
            let same = read_file(&path, &self.name).is_ok_and(|stored| stored.same_marks(self));
            if !same {
                return Err(L2dError::Refused {
                    reason: format!(
                        "{} holds another mark set called `{}` (another query, label, source \
                         or other marks); it was left unchanged; choose another name",
                        path.display(),
                        self.name
                    ),
                });
            }
        }
        budget.require_whole(SAVE_BUFFER_BYTES, &operation)?;
        fs::create_dir_all(&dir).map_err(|error| io_failure("creating", &dir, &error))?;
        let temporary = dir.join(format!(
            ".{}.json.saving-{}-{}",
            self.name,
            std::process::id(),
            SAVES.fetch_add(1, Ordering::Relaxed)
        ));
        let write = || -> Result<(), L2dError> {
            let file = fs::File::create(&temporary)
                .map_err(|error| io_failure("creating", &temporary, &error))?;
            let mut out = io::BufWriter::with_capacity(SAVE_WRITE_BYTES, file);
            serde_json::to_writer_pretty(&mut out, &self.file_out()).map_err(|error| {
                L2dError::Runtime {
                    detail: format!(
                        "writing the mark set `{}` to {} failed: {error}",
                        self.name,
                        temporary.display()
                    ),
                }
            })?;
            out.into_inner()
                .map_err(|error| io_failure("writing", &temporary, error.error()))?
                .sync_all()
                .map_err(|error| io_failure("writing", &temporary, &error))
        };
        let written = write().and_then(|()| {
            fs::rename(&temporary, &path).map_err(|error| io_failure("completing", &path, &error))
        });
        if written.is_err() {
            // Best effort: the failure itself is what the caller has to see.
            let _ = fs::remove_file(&temporary);
        }
        written.map(|()| path)
    }

    /// Loads the mark set `name` from `<project>/marks/<name>.json`.
    ///
    /// The file is read whole: it is counted as [`LOAD_FACTOR`] times its size and
    /// refused when that does not fit what `budget` leaves. A name that cannot be a
    /// mark set's is [`L2dError::InvalidInput`], a file that is not there
    /// [`L2dError::Missing`], and a file that is not a mark set this tool wrote (not
    /// readable, another format or kind, another name, a label other than bad or good,
    /// positions or ids out of order or of the other kind of source, a count that is not
    /// theirs, de-selections that are not marked) [`L2dError::Refused`].
    pub fn load(paths: &ProjectPaths, name: &str, budget: &MemoryBudget) -> Result<Self, L2dError> {
        check_name(name)?;
        let path = paths.mark_set_path(name);
        let metadata = match fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => {
                return Err(L2dError::Missing {
                    item: format!("mark set `{name}`"),
                    expected_at: path,
                });
            }
        };
        budget.require_whole(
            metadata.len().saturating_mul(LOAD_FACTOR),
            &format!("loading the mark set `{name}`"),
        )?;
        read_file(&path, name)
    }

    /// The names of the mark sets saved in the project, sorted; none when there is no
    /// `marks/` directory. Files that cannot be a mark set by their name are left out.
    pub fn list(paths: &ProjectPaths) -> Result<Vec<String>, L2dError> {
        let dir = paths.marks_dir();
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(io_failure("reading", &dir, &error)),
        };
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| io_failure("reading", &dir, &error))?;
            let file_name = entry.file_name();
            let Some(name) = file_name
                .to_str()
                .and_then(|file_name| file_name.strip_suffix(".json"))
            else {
                continue;
            };
            if check_name(name).is_ok() && entry.path().is_file() {
                names.push(name.to_string());
            }
        }
        names.sort();
        Ok(names)
    }

    /// Whether `other` holds the same marks: everything but the de-selections and the
    /// redaction rules.
    fn same_marks(&self, other: &Self) -> bool {
        self.name == other.name
            && self.label == other.label
            && self.query == other.query
            && self.source == other.source
            && self.positions == other.positions
            && self.ids == other.ids
    }
}

// --- Matching --------------------------------------------------------------------------

/// The query, ready to match.
enum Matcher {
    Phrase(String),
    Regex(Box<Regex>),
    Positions {
        /// Sorted, without repeats.
        wanted: Vec<u64>,
        found: Vec<bool>,
    },
}

impl Matcher {
    /// Checks `query` without building anything, and returns what its matcher will
    /// hold (see the module description), so that it can be refused before it is built.
    fn check(query: &MarkQuery) -> Result<u64, L2dError> {
        let invalid = |detail: String| Err(L2dError::InvalidInput { detail });
        match query {
            MarkQuery::Phrase { phrase } if phrase.is_empty() => {
                invalid("the phrase to mark by is empty".into())
            }
            MarkQuery::Phrase { phrase } => Ok(2 * phrase.len() as u64),
            MarkQuery::Regex { pattern } if pattern.is_empty() => {
                invalid("the regular expression to mark by is empty".into())
            }
            MarkQuery::Regex { pattern } if pattern.len() > MAX_PATTERN_BYTES => invalid(format!(
                "the regular expression to mark by has {} bytes; at most {MAX_PATTERN_BYTES} \
                 are taken",
                pattern.len()
            )),
            MarkQuery::Regex { .. } => Ok(REGEX_HELD_BYTES),
            MarkQuery::Positions { positions } if positions.is_empty() => {
                invalid("no positions to mark were given".into())
            }
            MarkQuery::Positions { positions } => {
                Ok(WANTED_POSITION_BYTES * positions.len() as u64)
            }
        }
    }

    /// The matcher of a query that passed [`Matcher::check`].
    fn new(query: &MarkQuery) -> Result<Self, L2dError> {
        let invalid = |detail: String| Err(L2dError::InvalidInput { detail });
        match query {
            MarkQuery::Phrase { phrase } => Ok(Self::Phrase(phrase.clone())),
            MarkQuery::Regex { pattern } => RegexBuilder::new(pattern)
                .size_limit(REGEX_SIZE_LIMIT)
                .dfa_size_limit(REGEX_DFA_SIZE_LIMIT)
                .build()
                .map(|regex| Self::Regex(Box::new(regex)))
                .or_else(|error| {
                    invalid(format!(
                        "the regular expression `{}` cannot be used: {}",
                        shortened(pattern),
                        shortened(&error.to_string())
                    ))
                }),
            MarkQuery::Positions { positions } => {
                let mut wanted = positions.clone();
                wanted.sort_unstable();
                wanted.dedup();
                wanted.shrink_to_fit();
                let found = vec![false; wanted.len()];
                Ok(Self::Positions { wanted, found })
            }
        }
    }

    fn matches(&mut self, position: u64, text: &str) -> bool {
        match self {
            Self::Phrase(phrase) => text.contains(phrase.as_str()),
            Self::Regex(regex) => regex.is_match(text),
            Self::Positions { wanted, found } => match wanted.binary_search(&position) {
                Ok(index) => {
                    found[index] = true;
                    true
                }
                Err(_) => false,
            },
        }
    }

    /// For positions: [`L2dError::InvalidInput`] naming those that named no log.
    fn require_all_found(&self) -> Result<(), L2dError> {
        let Self::Positions { wanted, found } = self else {
            return Ok(());
        };
        let missing: Vec<u64> = wanted
            .iter()
            .zip(found)
            .filter(|(_, found)| !**found)
            .map(|(position, _)| *position)
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(L2dError::InvalidInput {
            detail: format!(
                "{} {} no log of the input: {} (a log is named by the position where its \
                 text first occurs; a repeat, a skipped record or a position beyond the \
                 input names none); nothing was marked",
                missing.len(),
                if missing.len() == 1 {
                    "position names"
                } else {
                    "positions name"
                },
                named(&missing)
            ),
        })
    }
}

// --- The marked positions --------------------------------------------------------------

/// The marked positions while they are collected, in chunks of [`CHUNK_POSITIONS`].
struct Collected {
    chunks: Vec<Vec<u64>>,
    len: u64,
    /// The most the chunks may take, in bytes.
    room: u64,
}

/// What `chunks` full chunks take, with the list of them three times over (it doubles
/// as it grows, and holds the old and the new room for a moment).
fn chunk_bytes(chunks: u64) -> u64 {
    chunks * CHUNK_BYTES + 3 * size_of::<Vec<u64>>() as u64 * chunks.next_power_of_two().max(4)
}

impl Collected {
    fn new(room: u64) -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
            room,
        }
    }

    fn held_bytes(&self) -> u64 {
        chunk_bytes(self.chunks.len() as u64)
    }

    /// Sets the room to what is held now and what `rows` more positions can take, and
    /// returns it.
    fn reserve_for(&mut self, rows: u64) -> u64 {
        let chunks = self.chunks.len() as u64 + rows.div_ceil(CHUNK_POSITIONS as u64) + 1;
        self.room = chunk_bytes(chunks);
        self.room
    }

    /// Marks the rows of `batch` that `matcher` matches. The inner error is the room a
    /// new chunk needed when it did not fit.
    fn take(
        &mut self,
        matcher: &mut Matcher,
        batch: &LogsFrame,
    ) -> Result<Result<(), u64>, L2dError> {
        let failed = |error| frame_error("marking the logs of a batch", &error);
        let frame = batch.frame();
        let texts = frame
            .column(LogsFrame::TEXT)
            .and_then(Column::str)
            .map_err(failed)?;
        let positions = frame
            .column(LogsFrame::FIRST_POSITION)
            .and_then(Column::u64)
            .map_err(failed)?;
        for (text, position) in texts.into_no_null_iter().zip(positions.into_no_null_iter()) {
            if matcher.matches(position, text)
                && let Err(needed) = self.push(position)
            {
                return Ok(Err(needed));
            }
        }
        Ok(Ok(()))
    }

    fn push(&mut self, position: u64) -> Result<(), u64> {
        let full = self
            .chunks
            .last()
            .is_none_or(|chunk| chunk.len() == CHUNK_POSITIONS);
        if full {
            let needed = chunk_bytes(self.chunks.len() as u64 + 1);
            if needed > self.room {
                return Err(needed);
            }
            self.chunks.push(Vec::with_capacity(CHUNK_POSITIONS));
        }
        if let Some(chunk) = self.chunks.last_mut() {
            chunk.push(position);
        }
        self.len += 1;
        Ok(())
    }

    /// The positions in one sorted list without repeats: the chunks and the list
    /// together must fit the room; otherwise the chunks back, with what was needed.
    fn into_sorted(self) -> Result<Vec<u64>, (Self, u64)> {
        let needed = self.held_bytes() + 8 * self.len;
        if needed > self.room {
            return Err((self, needed));
        }
        let mut positions = Vec::with_capacity(self.len as usize);
        for chunk in self.chunks {
            positions.extend_from_slice(&chunk);
        }
        positions.sort_unstable();
        positions.dedup();
        Ok(positions)
    }
}

// --- Checks, messages and files --------------------------------------------------------

/// Checks a mark-set name: 1 to [`MAX_NAME_BYTES`] letters, digits, `.`, `_` and `-`,
/// not starting with a dot (the rule of the other saved things of the tool).
fn check_name(name: &str) -> Result<(), L2dError> {
    let usable = !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if usable {
        return Ok(());
    }
    Err(L2dError::InvalidInput {
        detail: format!(
            "a mark-set name must be 1 to {MAX_NAME_BYTES} letters, digits, '.', '_' or '-' \
             and must not start with a dot; got a name of {} bytes that is not",
            name.len()
        ),
    })
}

fn check_label(label: Label) -> Result<(), L2dError> {
    match label {
        Label::Bad | Label::Good => Ok(()),
        Label::Unknown => Err(L2dError::InvalidInput {
            detail: "the label of a mark set is bad or good, not unknown".into(),
        }),
    }
}

/// Whether `id` can be a store's log id: 64 lower-case hexadecimal characters.
fn is_id(id: &str) -> bool {
    id.len() == ID_CHARS
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// `text` up to [`SHOWN_BYTES`] bytes in a message, and how long it is when it is
/// longer.
fn shortened(text: &str) -> String {
    if text.len() <= SHOWN_BYTES {
        return text.to_string();
    }
    let mut end = SHOWN_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}... ({} bytes in all)", &text[..end], text.len())
}

/// Up to [`NAMED_POSITIONS`] items, and how many more there are.
fn named<T: ToString>(items: &[T]) -> String {
    let mut text = items
        .iter()
        .take(NAMED_POSITIONS)
        .map(T::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if items.len() > NAMED_POSITIONS {
        text.push_str(&format!(" and {} more", items.len() - NAMED_POSITIONS));
    }
    text
}

fn not_marked<T: ToString>(name: &str, what: &str, items: &[T]) -> L2dError {
    L2dError::InvalidInput {
        detail: format!(
            "{} {what}{} not marked in the mark set `{name}`: {}; nothing was de-selected",
            items.len(),
            if items.len() == 1 { " is" } else { "s are" },
            named(items)
        ),
    }
}

fn query_refused(name: &str, fixed: u64, left: u64, budget: &MemoryBudget) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "marking `{name}` needs {} for its query before it marks a log, but {} is left of \
             the memory limit of {}; nothing was marked",
            exact(fixed),
            exact(left),
            exact(budget.limit_bytes())
        ),
    }
}

fn positions_refused(
    name: &str,
    collected: &Collected,
    needed: u64,
    fixed: u64,
    left: u64,
    budget: &MemoryBudget,
) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "the marked positions of `{name}` do not fit: {} logs were marked so far, and \
             holding their positions needs {} beside {} for the query, but {} is left of the \
             memory limit of {}; nothing was saved",
            collected.len,
            exact(needed),
            exact(fixed),
            exact(left),
            exact(budget.limit_bytes())
        ),
    }
}

/// A refusal of the archive's read, with the marked positions named.
fn read_refused(error: L2dError, name: &str, collected: &Collected, beside: u64) -> L2dError {
    match error {
        L2dError::Refused { reason } => L2dError::Refused {
            reason: format!(
                "marking `{name}` holds beside the archive {} for its query and the marked \
                 positions of {} logs, with room for the next batch's: {reason}; nothing was \
                 saved",
                exact(beside),
                collected.len,
            ),
        },
        other => other,
    }
}

fn io_failure(doing: &str, path: &Path, error: &io::Error) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

fn not_a_mark_set(path: &Path, detail: impl std::fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "{} is not a mark set this tool can use: {detail}",
            path.display()
        ),
    }
}

/// Whether `items` ascend strictly and each is among `marked`.
fn ascending_within<T: Ord>(items: &[T], marked: &[T]) -> bool {
    items.windows(2).all(|pair| pair[0] < pair[1])
        && items.iter().all(|item| marked.binary_search(item).is_ok())
}

/// Reads and checks the file at `path`, expected to hold the mark set `name`.
fn read_file(path: &Path, name: &str) -> Result<MarkSet, L2dError> {
    let bytes = fs::read(path).map_err(|error| io_failure("reading", path, &error))?;
    parse_file(&bytes, path, name)
}

/// Checks the bytes of a mark-set file read from `path`, expected to hold the mark set
/// `name`.
fn parse_file(bytes: &[u8], path: &Path, name: &str) -> Result<MarkSet, L2dError> {
    let file: FileIn = serde_json::from_slice(bytes)
        .map_err(|error| not_a_mark_set(path, format!("it is not readable: {error}")))?;
    if (file.format != FORMAT && file.format != FORMAT_WITH_REDACTIONS) || file.kind != KIND {
        return Err(not_a_mark_set(
            path,
            format!(
                "it has format {} and kind '{}', expected {FORMAT} or \
                 {FORMAT_WITH_REDACTIONS} and '{KIND}'",
                file.format, file.kind
            ),
        ));
    }
    if (file.format == FORMAT_WITH_REDACTIONS) == file.redactions.is_empty() {
        return Err(not_a_mark_set(
            path,
            format!(
                "it has format {} with {} redaction rules; format {FORMAT} holds none and \
                 format {FORMAT_WITH_REDACTIONS} holds at least one",
                file.format,
                file.redactions.len()
            ),
        ));
    }
    check_redactions(&file.redactions).map_err(|error| match error {
        L2dError::InvalidInput { detail } => not_a_mark_set(path, detail),
        other => other,
    })?;
    if file.name != name {
        return Err(not_a_mark_set(
            path,
            format!("it names the mark set `{}`, not `{name}`", file.name),
        ));
    }
    check_label(file.label).map_err(|_| not_a_mark_set(path, "its label is not bad or good"))?;
    let of_store = matches!(file.source, MarkedSource::Store(_));
    if of_store && (!file.positions.is_empty() || !file.deselected.is_empty()) {
        return Err(not_a_mark_set(
            path,
            "it was made on a store but holds positions",
        ));
    }
    if !of_store && (!file.ids.is_empty() || !file.deselected_ids.is_empty()) {
        return Err(not_a_mark_set(
            path,
            "it was made on an input but holds ids",
        ));
    }
    if !file.positions.windows(2).all(|pair| pair[0] < pair[1])
        || !file.ids.windows(2).all(|pair| pair[0] < pair[1])
        || !file.ids.iter().all(|id| is_id(id))
    {
        return Err(not_a_mark_set(
            path,
            "its positions or ids are not in ascending order, or an id is not one",
        ));
    }
    let count = (file.positions.len() + file.ids.len()) as u64;
    if file.count != count {
        return Err(not_a_mark_set(
            path,
            format!(
                "its count is {} but it holds {count} marked logs",
                file.count
            ),
        ));
    }
    if !ascending_within(&file.deselected, &file.positions)
        || !ascending_within(&file.deselected_ids, &file.ids)
    {
        return Err(not_a_mark_set(
            path,
            "its de-selected logs are not marked ones in ascending order",
        ));
    }
    Ok(MarkSet {
        name: file.name,
        label: file.label,
        query: file.query,
        source: file.source,
        positions: file.positions,
        deselected: file.deselected,
        ids: file.ids,
        deselected_ids: file.deselected_ids,
        redactions: file.redactions,
    })
}

/// Checks the shape of redaction rules: at most [`MAX_REDACTIONS`], each with a
/// fragment of 1 to [`MAX_PATTERN_BYTES`] bytes and a placeholder of at most
/// [`MAX_PLACEHOLDER_BYTES`]. Whether a pattern compiles is checked where the rules are
/// applied (`l2d_baseline::redact`). [`L2dError::InvalidInput`] naming the first rule
/// that does not fit.
pub fn check_redactions(rules: &[RedactionRule]) -> Result<(), L2dError> {
    if rules.len() > MAX_REDACTIONS {
        return Err(L2dError::InvalidInput {
            detail: format!(
                "{} redaction rules were given; a mark set keeps at most {MAX_REDACTIONS}",
                rules.len()
            ),
        });
    }
    for (index, rule) in rules.iter().enumerate() {
        let number = index + 1;
        if rule.fragment.is_empty() || rule.fragment.len() > MAX_PATTERN_BYTES {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "redaction rule {number} has a fragment of {} bytes; it must have 1 to \
                     {MAX_PATTERN_BYTES}",
                    rule.fragment.len()
                ),
            });
        }
        if rule.placeholder.len() > MAX_PLACEHOLDER_BYTES {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "redaction rule {number} has a placeholder of {} bytes; it may have at \
                     most {MAX_PLACEHOLDER_BYTES}",
                    rule.placeholder.len()
                ),
            });
        }
    }
    Ok(())
}
