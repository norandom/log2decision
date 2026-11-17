//! Redaction: fragments proposed from a mark set, and the rules the analyst confirms
//! (requirements 16.4, 16.5 and 16.12; task 1.22).
//!
//! A score must find the same kind of log elsewhere, not the hosts and endpoints it was
//! fitted on (design.md, "MarkSet, Redaction and the mark-set score"). Redaction takes
//! what is specific to them out of the text before the features are hashed: the tool
//! proposes fragments ([`propose_redactions`]), the analyst confirms, edits, rejects or
//! adds ([`RedactionRules`]), and the confirmed rules are kept with the mark set.
//!
//! # Candidate fragments
//!
//! A text is cut into *words* at the separators: ASCII white space and control bytes,
//! and the punctuation `" ' ( ) , ; < = > ? [ \ ] ` { | } &`. Every other byte
//! belongs to a word, the bytes of a multi-byte character among them. A word is cut
//! further at `/` into *tokens*. The candidates of a text ([`candidate_fragments`]) are:
//!
//! - every token ([`FragmentKind::Token`]): a host with its port (`localhost:8080`), an
//!   address, a parameter name or value, a file name;
//! - every *path* ([`FragmentKind::Path`]): the run of tokens from a `/` to the end of
//!   its word, where that `/` neither follows nor precedes another `/`. In
//!   `http://localhost:8080/tienda1/publico/anadir.jsp` these are
//!   `/tienda1/publico/anadir.jsp`, `/publico/anadir.jsp` and `/anadir.jsp`; the `//`
//!   after the scheme starts none.
//!
//! A candidate needs a letter, a digit or a non-ASCII byte, and at most
//! [`RedactionSettings::max_fragment_bytes`] bytes. Of a log only the first
//! [`RedactionSettings::max_fragments_per_log`] candidates in text order are taken, and a
//! candidate that repeats in a log counts once for it.
//!
//! # Proposals
//!
//! A log is a distinct text of the input (or a stored log). The marked logs are those of
//! the mark set that are not de-selected; the unmarked logs are the logs of the same
//! input or store that the mark set does not mark. A de-selected log is on neither side:
//! the analyst took it out (requirement 16.3), and a score fitted from the mark set
//! keeps it out of fitting, neither marked nor of the other class (task 1.23); the
//! proposals count it apart ([`RedactionProposals::deselected_logs`]). A candidate is
//! proposed when it occurs in at least
//! `max(MIN_LOGS, ceil(min_marked_share x marked logs))` marked logs **and** in at least
//! `max(MIN_LOGS, ceil(min_unmarked_share x unmarked logs))` unmarked logs: it recurs
//! across the marked logs and is frequent in the others too, so it does not tell the
//! marked kind of log apart (host names, addresses, URL paths, fixed parameter names and
//! values). A token of the payload of an attack, frequent among the marked logs only, is
//! not proposed. Each proposal names the number of marked and of unmarked logs it occurs
//! in. Proposals are ordered by marked logs, then unmarked logs (both descending), then
//! by length (longest first) and bytes.
//!
//! # Counting: bounded counters, two reads
//!
//! The input is read twice, in the batches of its reader, and never held as a whole.
//!
//! 1. The candidates of the marked logs are counted in a bounded counter of
//!    [`RedactionSettings::counter_slots`] slots (the Misra-Gries summary): when it
//!    holds twice as many fragments as it has slots, the count of the slot-plus-first
//!    largest is taken off every count and the fragments left at zero are dropped. The
//!    total taken off, `dropped`, bounds what any fragment lost. When it is not below
//!    the marked logs a proposal needs, a fragment that qualifies may have been lost,
//!    and the proposals are refused with the figures; otherwise every fragment that
//!    qualifies is still counted, and every fragment whose count plus `dropped` reaches
//!    the threshold is a candidate.
//! 2. The candidates are counted exactly in the marked and in the unmarked logs, and
//!    those that reach both thresholds are proposed.
//!
//! The proposals are therefore those of an exact count, whatever the counter size that
//! was not refused, the batch size and the entries of an archive read at once: the
//! stream of logs is in the input's order of first occurrence either way.
//!
//! # Memory (requirements 12, 15 and 16.12)
//!
//! The counters are bounded, so what they can take is known before anything is read
//! ([`RedactionSettings::counters_bytes`]): both hash tables at twice the slots, as many
//! fragments of the largest length, the scratch of the selection and the candidates
//! moved between the reads, and the candidates of one log. That, and what the mark set
//! holds of its positions (8 bytes each) or ids (`MARKED_ID_BYTES` each), is held beside
//! the batches for the whole operation, and the operation is refused with the figures
//! before anything is counted when it does not fit what the budget leaves. A file's
//! logs frame is the caller's and counted as held already; an archive is read batch by
//! batch with that beside it; a store hands out batches cut to fit beside it.
//!
//! # Rules
//!
//! [`RedactionRules`] are ordered replacements by a placeholder token
//! ([`RedactionRule`] of `l2d-core`, kept with the mark set). Each rule applies to the
//! text the rules before it left:
//!
//! - a [`RedactionKind::Literal`] rule replaces every occurrence of its fragment, left to
//!   right, that does not continue a token on either side (a byte before a fragment
//!   that starts with a token byte, and a byte after one that ends with a token byte,
//!   must be a separator or `/`): `tienda1` is replaced in `/tienda1/` and not in
//!   `/tienda10/`;
//! - a [`RedactionKind::Pattern`] rule replaces every match of its regular expression
//!   (the syntax of the `regex` crate, with the size limits of mark sets); the
//!   placeholder is taken as it is, with no `$` expansion.
//!
//! Confirmed proposals become literal rules with [`DEFAULT_PLACEHOLDER`], the longest
//! first, so that a fragment inside another is replaced after it. The placeholder is the
//! ASCII control character SUB (substitute, `0x1A`): one byte, a separator, and not
//! found in log text. An empty placeholder deletes.
//!
//! The rules apply to the text before feature hashing ([`CharHasher::row_redacted`]),
//! before the routing option of requirement 2.6 splits it, and the same at fitting and
//! at scoring: a scorer fitted with rules (`BaselineScorer::fit_redacted_with`, task
//! 1.23) keeps them and hashes every text through [`CharHasher::row_redacted`], at
//! fitting and at scoring alike. The logs frame it is fitted on keeps the texts as they
//! were read, one row per distinct text: texts that are equal after redaction stay rows
//! of their own (nothing is merged, so no rule for occurrences or for opposite labels is
//! needed), and the hashes of the fitted texts are those of the texts as read. No logs
//! frame with redacted texts is ever built, so the uniqueness of the texts of a logs
//! frame holds. The routing option `strip` is itself a
//! rule ([`RedactionRules::routing`]): the plain feature rule after it gives the rows of
//! [`RoutingMode::Strip`](crate::RoutingMode::Strip) bit for bit. `isolate` hashes the
//! two parts into separate index ranges and has no rule of text.
//!
//! [`RedactionRules::content_sha256`] names the rules: their order, kinds, fragments and
//! placeholders. It is meant to be part of the revision of a score fitted from a mark
//! set (task 1.23).
//!
//! # Untrusted data
//!
//! Log text can hold attack payloads. It is cut and compared as bytes only, never
//! executed or used to build paths or commands; the patterns are the analyst's.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use l2d_core::input::PARSE_BATCH_BYTES;
use l2d_core::marks::{
    MARKED_ID_BYTES, MarkSet, MarkStore, REGEX_DFA_SIZE_LIMIT, REGEX_HELD_BYTES, REGEX_SIZE_LIMIT,
    check_redactions,
};
use l2d_core::memory::{PROCESS_BASELINE_BYTES, exact};
use l2d_core::polars::prelude::{Column, DataFrame, PolarsError};
use l2d_core::{
    ArchivePosition, ArchiveSummary, InputSummary, L2dError, LogsFrame, MemoryBudget, ProjectPaths,
    RedactionKind, RedactionRule, ZipInput, ZipInputSettings,
};
use regex::{NoExpand, Regex, RegexBuilder};
use sha2::{Digest, Sha256};

use crate::hashing::{CharHasher, SparseRow};

/// The placeholder of confirmed proposals: the ASCII control character SUB.
pub const DEFAULT_PLACEHOLDER: &str = "\u{1a}";

/// A proposal occurs in at least this share of the marked logs, by default.
pub const DEFAULT_MIN_MARKED_SHARE: f64 = 0.05;

/// A proposal occurs in at least this share of the unmarked logs, by default.
pub const DEFAULT_MIN_UNMARKED_SHARE: f64 = 0.01;

/// A proposal occurs in at least this many marked and this many unmarked logs, whatever
/// the shares: it recurs.
pub const MIN_LOGS: u64 = 2;

/// The slots of the bounded counter, by default.
pub const DEFAULT_COUNTER_SLOTS: usize = 32_768;

/// The most slots of the bounded counter.
pub const MAX_COUNTER_SLOTS: usize = 1 << 24;

/// The longest candidate fragment, in bytes, by default.
pub const DEFAULT_MAX_FRAGMENT_BYTES: usize = 128;

/// The longest candidate fragment that can be asked for, in bytes.
pub const MAX_FRAGMENT_BYTES: usize = 4096;

/// The candidates taken of one log, by default.
pub const DEFAULT_MAX_FRAGMENTS_PER_LOG: usize = 1024;

/// The most candidates of one log that can be asked for.
pub const MAX_FRAGMENTS_PER_LOG: usize = 1 << 20;

/// The rule of the routing option `strip` (requirement 2.6): everything up to and with
/// the first `?` or line feed, or the whole text when it has neither.
pub const ROUTING_PATTERN: &str = r"\A[^?\n]*(?:[?\n]|\z)";

/// What a fragment held in a counter is counted as beyond its bytes: the allocation's
/// rounding.
const FRAGMENT_OVERHEAD_BYTES: u64 = 16;

/// What one hash-table bucket takes beyond its entry: the control byte.
const BUCKET_CONTROL_BYTES: u64 = 1;

/// The settings of the proposals. All are the analyst's; the defaults are stated above.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedactionSettings {
    /// A proposal occurs in at least this share of the marked logs (above 0, at most 1).
    pub min_marked_share: f64,
    /// A proposal occurs in at least this share of the unmarked logs (above 0, at most 1).
    pub min_unmarked_share: f64,
    /// The slots of the bounded counter, 1 to [`MAX_COUNTER_SLOTS`].
    pub counter_slots: usize,
    /// The longest candidate fragment in bytes, 1 to [`MAX_FRAGMENT_BYTES`].
    pub max_fragment_bytes: usize,
    /// The candidates taken of one log, 1 to [`MAX_FRAGMENTS_PER_LOG`].
    pub max_fragments_per_log: usize,
}

impl Default for RedactionSettings {
    fn default() -> Self {
        Self {
            min_marked_share: DEFAULT_MIN_MARKED_SHARE,
            min_unmarked_share: DEFAULT_MIN_UNMARKED_SHARE,
            counter_slots: DEFAULT_COUNTER_SLOTS,
            max_fragment_bytes: DEFAULT_MAX_FRAGMENT_BYTES,
            max_fragments_per_log: DEFAULT_MAX_FRAGMENTS_PER_LOG,
        }
    }
}

impl RedactionSettings {
    /// [`L2dError::InvalidInput`] naming the first setting out of range.
    pub fn validate(&self) -> Result<(), L2dError> {
        let invalid = |detail: String| Err(L2dError::InvalidInput { detail });
        for (name, share) in [
            ("marked", self.min_marked_share),
            ("unmarked", self.min_unmarked_share),
        ] {
            if !(share > 0.0 && share <= 1.0) {
                return invalid(format!(
                    "the share of {name} logs a proposal needs must be above 0 and at most 1, \
                     not {share}"
                ));
            }
        }
        if !(1..=MAX_COUNTER_SLOTS).contains(&self.counter_slots) {
            return invalid(format!(
                "the counter slots must be 1 to {MAX_COUNTER_SLOTS}, not {}",
                self.counter_slots
            ));
        }
        if !(1..=MAX_FRAGMENT_BYTES).contains(&self.max_fragment_bytes) {
            return invalid(format!(
                "the longest fragment must be 1 to {MAX_FRAGMENT_BYTES} bytes, not {}",
                self.max_fragment_bytes
            ));
        }
        if !(1..=MAX_FRAGMENTS_PER_LOG).contains(&self.max_fragments_per_log) {
            return invalid(format!(
                "the fragments taken of one log must be 1 to {MAX_FRAGMENTS_PER_LOG}, not {}",
                self.max_fragments_per_log
            ));
        }
        Ok(())
    }

    /// What the counters can take at most, in bytes (module documentation, "Memory").
    pub fn counters_bytes(&self) -> u64 {
        let entries = 2 * self.counter_slots as u64;
        let first = table_bytes(entries, size_of::<(Box<str>, u64)>());
        let second = table_bytes(entries, size_of::<(Box<str>, Counts)>());
        let fragments = entries * (self.max_fragment_bytes as u64 + FRAGMENT_OVERHEAD_BYTES);
        let scratch = entries * size_of::<u64>() as u64;
        let moved = entries * size_of::<Box<str>>() as u64;
        let one_log = self.max_fragments_per_log as u64 * size_of::<(usize, usize)>() as u64;
        first + second + fragments + scratch + moved + one_log
    }

    /// The fewest marked and unmarked logs a proposal occurs in, for `marked` and
    /// `unmarked` logs.
    fn thresholds(&self, marked: u64, unmarked: u64) -> (u64, u64) {
        let at_least = |share: f64, logs: u64| MIN_LOGS.max((share * logs as f64).ceil() as u64);
        (
            at_least(self.min_marked_share, marked),
            at_least(self.min_unmarked_share, unmarked),
        )
    }
}

/// What a hash table of `entries` entries of `entry` bytes allocates.
fn table_bytes(entries: u64, entry: usize) -> u64 {
    let buckets = (entries * 8).div_ceil(7).next_power_of_two().max(4);
    buckets * (entry as u64 + BUCKET_CONTROL_BYTES) + 16
}

/// What kind of candidate a fragment is (module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentKind {
    /// A token: no `/` in it.
    Token,
    /// A path: from a `/` to the end of its word.
    Path,
}

impl FragmentKind {
    /// The kind of a candidate fragment: a path starts with `/`, a token never does.
    pub fn of(fragment: &str) -> Self {
        if fragment.starts_with('/') {
            Self::Path
        } else {
            Self::Token
        }
    }
}

/// One proposed redaction (16.4): the fragment and the number of marked and unmarked
/// logs it occurs in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactionProposal {
    pub fragment: String,
    pub kind: FragmentKind,
    pub marked: u64,
    pub unmarked: u64,
}

/// The proposals for a mark set, with what they were counted from.
#[derive(Debug, Clone)]
pub struct RedactionProposals {
    /// In order (module documentation, "Proposals").
    pub proposals: Vec<RedactionProposal>,
    /// The marked logs counted (the mark set's, less the de-selected ones).
    pub marked_logs: u64,
    /// The other logs of the input or store, less the de-selected ones.
    pub unmarked_logs: u64,
    /// The de-selected logs of the mark set: counted on neither side (16.3), as they are
    /// kept out of fitting (task 1.23).
    pub deselected_logs: u64,
    /// The fewest marked logs a proposal occurs in.
    pub min_marked: u64,
    /// The fewest unmarked logs a proposal occurs in.
    pub min_unmarked: u64,
    /// The settings used.
    pub settings: RedactionSettings,
    /// What the counters were counted as in the memory budget.
    pub counters_bytes: u64,
    /// What the bounded counter took off every count (0: it never overflowed).
    pub dropped: u64,
    /// The candidates counted exactly in the second read.
    pub candidates: u64,
    /// For an archive, the summary of its last read (with its entry reports, damaged
    /// entries among them, for the caller to report).
    pub archive: Option<ArchiveSummary>,
}

impl RedactionProposals {
    /// The proposal of `fragment`, if it is proposed.
    pub fn get(&self, fragment: &str) -> Option<&RedactionProposal> {
        self.proposals
            .iter()
            .find(|proposal| proposal.fragment == fragment)
    }
}

/// The logs a mark set is applied to: the input or store it was made on.
pub enum RedactionInput<'a> {
    /// A file its adapter has read: the logs frame (counted as held by the caller) and
    /// the summary.
    File {
        logs: &'a LogsFrame,
        summary: &'a InputSummary,
    },
    /// An archive, read in batches, with the settings (entry patterns among them) it was
    /// marked with.
    Archive {
        path: &'a Path,
        settings: &'a ZipInputSettings,
    },
    /// A store, read in batches.
    Store(&'a dyn MarkStore),
}

// --- Candidates -------------------------------------------------------------------------

/// Whether `byte` ends a word.
fn is_separator(byte: u8) -> bool {
    byte <= b' '
        || byte == 0x7f
        || matches!(
            byte,
            b'"' | b'\''
                | b'('
                | b')'
                | b','
                | b';'
                | b'<'
                | b'='
                | b'>'
                | b'?'
                | b'['
                | b'\\'
                | b']'
                | b'`'
                | b'{'
                | b'|'
                | b'}'
                | b'&'
        )
}

/// Whether `byte` belongs to a token: neither a separator nor `/`.
fn is_token_byte(byte: u8) -> bool {
    !is_separator(byte) && byte != b'/'
}

/// Whether `fragment` holds a letter, a digit or a non-ASCII byte.
fn has_content(fragment: &[u8]) -> bool {
    fragment
        .iter()
        .any(|byte| byte.is_ascii_alphanumeric() || *byte >= 0x80)
}

/// The candidates of `text` as byte ranges, sorted by their text and without repeats.
fn cut(text: &str, settings: &RedactionSettings, out: &mut Vec<(usize, usize)>) {
    out.clear();
    let bytes = text.as_bytes();
    // Adds `start..end` when it is a candidate; `false` once the log has all it takes.
    let take = |start: usize, end: usize, out: &mut Vec<(usize, usize)>| -> bool {
        if end - start > settings.max_fragment_bytes || !has_content(&bytes[start..end]) {
            return true;
        }
        if out.len() >= settings.max_fragments_per_log {
            return false;
        }
        out.push((start, end));
        true
    };
    let mut at = 0;
    'words: while at < bytes.len() {
        if is_separator(bytes[at]) {
            at += 1;
            continue;
        }
        let start = at;
        while at < bytes.len() && !is_separator(bytes[at]) {
            at += 1;
        }
        let end = at;
        let mut token = start;
        for slash in start..=end {
            if slash < end && bytes[slash] != b'/' {
                continue;
            }
            if slash > token && !take(token, slash, out) {
                break 'words;
            }
            if slash < end {
                let after_slash = slash > start && bytes[slash - 1] == b'/';
                let before_slash = slash + 1 < end && bytes[slash + 1] != b'/';
                if !after_slash && before_slash && !take(slash, end, out) {
                    break 'words;
                }
            }
            token = slash + 1;
        }
    }
    out.sort_unstable_by(|a, b| bytes[a.0..a.1].cmp(&bytes[b.0..b.1]));
    out.dedup_by(|a, b| bytes[a.0..a.1] == bytes[b.0..b.1]);
}

/// The candidate fragments of `text` under `settings` (module documentation, "Candidate
/// fragments"), sorted and without repeats.
pub fn candidate_fragments<'t>(text: &'t str, settings: &RedactionSettings) -> Vec<&'t str> {
    let mut ranges = Vec::new();
    cut(text, settings, &mut ranges);
    ranges
        .into_iter()
        .map(|(start, end)| &text[start..end])
        .collect()
}

// --- Counting ---------------------------------------------------------------------------

/// The marked and unmarked logs a candidate occurs in.
#[derive(Debug, Clone, Copy, Default)]
struct Counts {
    marked: u64,
    unmarked: u64,
}

/// The bounded counter of the first read (module documentation, "Counting").
struct Counter {
    table: HashMap<Box<str>, u64>,
    /// Fragments held at most: twice the slots.
    full: usize,
    slots: usize,
    dropped: u64,
    scratch: Vec<u64>,
}

impl Counter {
    fn new(slots: usize) -> Self {
        let full = 2 * slots;
        Self {
            table: HashMap::with_capacity(full),
            full,
            slots,
            dropped: 0,
            scratch: Vec::with_capacity(full),
        }
    }

    fn add(&mut self, fragment: &str) {
        if let Some(count) = self.table.get_mut(fragment) {
            *count += 1;
            return;
        }
        self.table.insert(fragment.into(), 1);
        if self.table.len() >= self.full {
            self.prune();
        }
    }

    /// Takes the count of the slot-plus-first largest off every count and drops the
    /// fragments left at zero: at most `slots` stay.
    fn prune(&mut self) {
        self.scratch.clear();
        self.scratch.extend(self.table.values().copied());
        let (_, cut, _) = self
            .scratch
            .select_nth_unstable_by(self.slots, |a, b| b.cmp(a));
        let cut = *cut;
        self.table.retain(|_, count| {
            *count -= cut.min(*count);
            *count > 0
        });
        self.dropped += cut;
    }
}

/// What the mark set holds of its logs, in bytes.
pub(crate) fn marks_bytes(marks: &MarkSet) -> u64 {
    8 * (marks.positions().len() + marks.deselected().len()) as u64
        + MARKED_ID_BYTES * (marks.id_count() + marks.deselected_id_count()) as u64
}

fn frame_failed(doing: &str, error: &PolarsError) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} failed: {error}"),
    }
}

/// One log as a reader of the input hands it out (task 1.23 reads the same way).
pub(crate) struct LogView<'a> {
    pub text: &'a str,
    /// The first position in an input; `None` for a store.
    pub position: Option<u64>,
    /// The id of a stored log; `None` for an input.
    pub id: Option<&'a str>,
    /// The group key the adapter gave the log (a file's `group` column).
    pub group: Option<&'a str>,
    /// The path of the archive entry the log was first read from.
    pub entry: Option<&'a str>,
    /// How often the text occurred (1 for a stored log).
    pub occurrences: u32,
    /// The store's sequence number of the log; `None` for an input.
    pub seq: Option<i64>,
}

/// What a mark set says of one log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MarkSide {
    /// Marked and not de-selected.
    Marked,
    /// Marked and de-selected: kept out of fitting and out of the proposals (16.3).
    Deselected,
    /// Not marked.
    Unmarked,
}

impl MarkSide {
    pub(crate) fn of(marks: &MarkSet, view: &LogView<'_>) -> Result<Self, L2dError> {
        Ok(match (view.position, view.id) {
            (Some(position), _) => {
                if marks.is_selected(position) {
                    MarkSide::Marked
                } else if marks.deselected().binary_search(&position).is_ok() {
                    MarkSide::Deselected
                } else {
                    MarkSide::Unmarked
                }
            }
            (None, Some(id)) => {
                if marks.is_selected_id(id)? {
                    MarkSide::Marked
                } else if marks.is_deselected_id(id)? {
                    MarkSide::Deselected
                } else {
                    MarkSide::Unmarked
                }
            }
            (None, None) => MarkSide::Unmarked,
        })
    }
}

/// Takes the logs a reader hands out.
pub(crate) trait LogSink {
    /// One log, in the input's order.
    fn log(&mut self, view: &LogView<'_>) -> Result<(), L2dError>;

    /// What the sink holds beside the batches now. Asked before the first batch and after
    /// every batch: an archive sizes its next batch with it, a store cuts it.
    fn beside(&mut self) -> u64;

    /// What a store is asked to charge for each row of a batch beside its own share,
    /// for what the sink keeps of the row (`MarkStore::text_batches`); 0 by default.
    fn per_row(&self) -> u64 {
        0
    }

    /// Told once an archive is open and checked, before the sink is asked what it holds
    /// beside the batches: the longest entry path of its directory and the most bytes a
    /// record may have. Nothing by default.
    fn archive_opened(&mut self, _longest_entry_path: u64, _max_record_bytes: u64) {}
}

/// Hands the logs of one logs frame (a file's, or a batch of an archive) to `sink`.
/// `entry_of` names the archive entry of a position.
fn frame_logs<'e>(
    frame: &DataFrame,
    sink: &mut dyn LogSink,
    entry_of: &dyn Fn(u64) -> Option<&'e str>,
) -> Result<(), L2dError> {
    let doing = "reading the logs of a batch";
    let column = |name: &str| {
        frame
            .column(name)
            .map_err(|error| frame_failed(doing, &error))
    };
    let texts = column(LogsFrame::TEXT)?
        .str()
        .map_err(|error| frame_failed(doing, &error))?;
    let positions = column(LogsFrame::FIRST_POSITION)?
        .u64()
        .map_err(|error| frame_failed(doing, &error))?;
    let groups = column(LogsFrame::GROUP)?
        .str()
        .map_err(|error| frame_failed(doing, &error))?;
    let occurrences = column(LogsFrame::OCCURRENCES)?
        .u32()
        .map_err(|error| frame_failed(doing, &error))?;
    for (((text, position), group), occurrences) in texts
        .into_no_null_iter()
        .zip(positions.into_no_null_iter())
        .zip(groups.iter())
        .zip(occurrences.into_no_null_iter())
    {
        sink.log(&LogView {
            text,
            position: Some(position),
            id: None,
            group,
            entry: entry_of(position),
            occurrences,
            seq: None,
        })?;
    }
    Ok(())
}

/// Hands every log of `input` to `sink`, in the input's order. `operation` names what
/// reads in a refusal; `held` is what the caller held before (an archive's reader sets
/// what is held itself, so it goes beside it). For an archive the archive is opened and
/// checked against the mark set first, and its summary returned.
pub(crate) fn read_logs(
    operation: &str,
    marks: &MarkSet,
    input: &RedactionInput<'_>,
    held: u64,
    budget: &mut MemoryBudget,
    sink: &mut dyn LogSink,
) -> Result<Option<ArchiveSummary>, L2dError> {
    let doing = "reading the logs of a batch";
    match input {
        RedactionInput::File { logs, .. } => {
            frame_logs(logs.frame(), sink, &|_| None)?;
            Ok(None)
        }
        RedactionInput::Archive { path, settings } => {
            let name = marks.name();
            let mut archive = ZipInput::open(path, settings, budget)?;
            let summary = InputSummary {
                source_path: path.to_path_buf(),
                records: 0,
                distinct_texts: 0,
                content_sha256: archive.content_sha256().to_string(),
                skipped: Vec::new(),
                label_conflicts: 0,
            };
            marks.require_input(&summary, Some(&settings.entries))?;
            sink.archive_opened(archive.longest_entry_path(), settings.max_record_bytes);
            let beside = held + sink.beside();
            archive.hold_beside(beside);
            let needed = archive.held_bytes() + PARSE_BATCH_BYTES;
            if needed > budget.usable_bytes() {
                return Err(L2dError::Refused {
                    reason: format!(
                        "{operation} for the mark set `{name}` needs {} before the first \
                         batch: {} beside the archive (what the operation and the mark set \
                         hold, room it keeps for what it takes of a batch, and what was held \
                         before), {} that the archive holds once its \
                         directory is read, and {} for the fixed memory of a batch; but the \
                         memory limit of {} leaves {} beside the {} kept for the tool itself; \
                         nothing was read",
                        exact(needed),
                        exact(beside),
                        exact(archive.held_bytes() - beside),
                        exact(PARSE_BATCH_BYTES),
                        exact(budget.limit_bytes()),
                        exact(budget.usable_bytes()),
                        exact(PROCESS_BASELINE_BYTES)
                    ),
                });
            }
            {
                let mut batches = archive.batches(budget);
                while let Some(batch) = batches.next() {
                    let batch = batch.map_err(|error| match error {
                        L2dError::Refused { reason } => L2dError::Refused {
                            reason: format!(
                                "{operation} for the mark set `{name}` holds {} beside the \
                                 archive (what the operation and the mark set hold, room it keeps \
                                 for what it takes of a batch, and what was held before): \
                                 {reason}",
                                exact(held + sink.beside())
                            ),
                        },
                        other => other,
                    })?;
                    let entries = &batches;
                    frame_logs(batch.frame(), sink, &|position| {
                        entries.entry_path(ArchivePosition::unpack(position).entry)
                    })?;
                    batches.hold_beside(held + sink.beside());
                }
            }
            Ok(Some(archive.finish()))
        }
        RedactionInput::Store(store) => {
            let beside = sink.beside();
            store.text_batches(beside, sink.per_row(), budget, &mut |batch| {
                let seqs = batch
                    .column("seq")
                    .and_then(Column::i64)
                    .map_err(|error| frame_failed(doing, &error))?;
                let ids = batch
                    .column("id")
                    .and_then(Column::str)
                    .map_err(|error| frame_failed(doing, &error))?;
                let texts = batch
                    .column(LogsFrame::TEXT)
                    .and_then(Column::str)
                    .map_err(|error| frame_failed(doing, &error))?;
                for ((seq, id), text) in seqs
                    .into_no_null_iter()
                    .zip(ids.into_no_null_iter())
                    .zip(texts.into_no_null_iter())
                {
                    sink.log(&LogView {
                        text,
                        position: None,
                        id: Some(id),
                        group: None,
                        entry: None,
                        occurrences: 1,
                        seq: Some(seq),
                    })?;
                }
                Ok(sink.beside())
            })?;
            Ok(None)
        }
    }
}

/// A sink that hands each log's text and side to a function, holding `beside`.
struct SideSink<'m, F: FnMut(&str, MarkSide)> {
    marks: &'m MarkSet,
    beside: u64,
    each: F,
}

impl<F: FnMut(&str, MarkSide)> LogSink for SideSink<'_, F> {
    fn log(&mut self, view: &LogView<'_>) -> Result<(), L2dError> {
        (self.each)(view.text, MarkSide::of(self.marks, view)?);
        Ok(())
    }

    fn beside(&mut self) -> u64 {
        self.beside
    }
}
/// Proposes redactions for the mark set `marks` on the input or store it was made on
/// (requirement 16.4; module documentation).
///
/// The input is checked first: a file by [`MarkSet::require_input`] on its summary, an
/// archive by its content hash and entry patterns when it is opened, a store by
/// [`MarkSet::require_store`] before the first read and after the second. A frame or
/// archive that does not hold the mark set's selected logs is refused.
///
/// - [`L2dError::InvalidInput`]: settings out of range.
/// - [`L2dError::Refused`]: another input or store, or a source of the other kind; the
///   counters and the marked logs do not fit what `budget` leaves (with the figures,
///   before anything is counted); the bounded counter dropped too much to be sure of
///   every fragment that qualifies (with the figures); an input that changed between
///   the two reads; and what the reader of an archive or store refuses.
///
/// The budget holds afterwards what it held before.
pub fn propose_redactions(
    marks: &MarkSet,
    input: RedactionInput<'_>,
    settings: &RedactionSettings,
    budget: &mut MemoryBudget,
) -> Result<RedactionProposals, L2dError> {
    let held = budget.held_bytes();
    let proposed = propose(marks, &input, settings, budget, held);
    budget.hold(held);
    proposed
}

fn propose(
    marks: &MarkSet,
    input: &RedactionInput<'_>,
    settings: &RedactionSettings,
    budget: &mut MemoryBudget,
    held: u64,
) -> Result<RedactionProposals, L2dError> {
    settings.validate()?;
    let name = marks.name();
    match input {
        RedactionInput::File { summary, .. } => marks.require_input(summary, None)?,
        RedactionInput::Store(store) => marks.require_store(*store, budget)?,
        RedactionInput::Archive { .. } => {}
    }
    let counters = settings.counters_bytes();
    let of_marks = marks_bytes(marks);
    let left = budget.left_bytes();
    if counters + of_marks > left {
        return Err(L2dError::Refused {
            reason: format!(
                "proposing redactions for the mark set `{name}` needs {} beside what is held: \
                 {} for its counters ({} slots, fragments of at most {} bytes) and {} for the \
                 marked logs it names; but the memory limit of {} leaves {} beside the {} held \
                 and the {} kept for the tool itself; nothing was counted (fewer counter \
                 slots or shorter fragments take less)",
                exact(counters + of_marks),
                exact(counters),
                settings.counter_slots,
                settings.max_fragment_bytes,
                exact(of_marks),
                exact(budget.limit_bytes()),
                exact(left),
                exact(held),
                exact(PROCESS_BASELINE_BYTES)
            ),
        });
    }
    // What is held beside the batches; an archive's reader also puts the caller's
    // holding beside them, since it sets what is held itself.
    let beside = counters + of_marks;
    let operation = "proposing redactions";

    // First read: the marked logs' candidates in the bounded counter. A de-selected log
    // is on neither side (16.3).
    let mut counter = Counter::new(settings.counter_slots);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(settings.max_fragments_per_log);
    let (mut marked_logs, mut unmarked_logs, mut deselected_logs) = (0u64, 0u64, 0u64);
    let mut first = SideSink {
        marks,
        beside,
        each: |text: &str, side: MarkSide| match side {
            MarkSide::Deselected => deselected_logs += 1,
            MarkSide::Unmarked => unmarked_logs += 1,
            MarkSide::Marked => {
                marked_logs += 1;
                cut(text, settings, &mut ranges);
                for &(start, end) in &ranges {
                    counter.add(&text[start..end]);
                }
            }
        },
    };
    let mut archive = read_logs(operation, marks, input, held, budget, &mut first)?;
    if marked_logs != marks.selected_count() {
        return Err(L2dError::Refused {
            reason: format!(
                "the mark set `{name}` names {} marked logs that are not de-selected, but {} \
                 of them are among the {} logs read; read the input as it was read when it \
                 was marked",
                marks.selected_count(),
                marked_logs,
                marked_logs + unmarked_logs
            ),
        });
    }
    let (min_marked, min_unmarked) = settings.thresholds(marked_logs, unmarked_logs);
    if counter.dropped >= min_marked {
        return Err(L2dError::Refused {
            reason: format!(
                "the counters of {} slots took up to {} off the count of a fragment while \
                 counting the {marked_logs} marked logs of `{name}`, which is not below the \
                 {min_marked} marked logs a proposal needs: a fragment that qualifies may \
                 have been lost; give the counters more slots (each takes about {} bytes) or \
                 ask for a higher share of marked logs",
                settings.counter_slots,
                counter.dropped,
                settings.counters_bytes() / settings.counter_slots as u64
            ),
        });
    }
    let dropped = counter.dropped;
    let candidates: Vec<Box<str>> = counter
        .table
        .drain()
        .filter(|(_, count)| count + dropped >= min_marked)
        .map(|(fragment, _)| fragment)
        .collect();
    drop(counter);
    let candidate_count = candidates.len() as u64;

    // Second read: the candidates counted exactly on both sides.
    let mut proposals = Vec::new();
    if !candidates.is_empty() {
        let mut counts: HashMap<Box<str>, Counts> = HashMap::with_capacity(candidates.len());
        counts.extend(
            candidates
                .into_iter()
                .map(|fragment| (fragment, Counts::default())),
        );
        let (mut marked_again, mut unmarked_again) = (0u64, 0u64);
        let mut second = SideSink {
            marks,
            beside,
            each: |text: &str, side: MarkSide| {
                let marked = match side {
                    MarkSide::Deselected => return,
                    MarkSide::Marked => true,
                    MarkSide::Unmarked => false,
                };
                if marked {
                    marked_again += 1;
                } else {
                    unmarked_again += 1;
                }
                cut(text, settings, &mut ranges);
                for &(start, end) in &ranges {
                    if let Some(count) = counts.get_mut(&text[start..end]) {
                        if marked {
                            count.marked += 1;
                        } else {
                            count.unmarked += 1;
                        }
                    }
                }
            },
        };
        archive = read_logs(operation, marks, input, held, budget, &mut second)?;
        if (marked_again, unmarked_again) != (marked_logs, unmarked_logs) {
            return Err(L2dError::Refused {
                reason: format!(
                    "the input of the mark set `{name}` changed between the two reads of the \
                     redaction proposals: {marked_logs} marked and {unmarked_logs} unmarked \
                     logs the first time, {marked_again} and {unmarked_again} the second"
                ),
            });
        }
        if let RedactionInput::Store(store) = input {
            marks.require_store(*store, budget)?;
        }
        proposals = counts
            .into_iter()
            .filter(|(_, count)| count.marked >= min_marked && count.unmarked >= min_unmarked)
            .map(|(fragment, count)| RedactionProposal {
                kind: FragmentKind::of(&fragment),
                fragment: fragment.into(),
                marked: count.marked,
                unmarked: count.unmarked,
            })
            .collect();
        proposals.sort_by(|a, b| {
            b.marked
                .cmp(&a.marked)
                .then(b.unmarked.cmp(&a.unmarked))
                .then(b.fragment.len().cmp(&a.fragment.len()))
                .then(a.fragment.cmp(&b.fragment))
        });
    }
    Ok(RedactionProposals {
        proposals,
        marked_logs,
        unmarked_logs,
        deselected_logs,
        min_marked,
        min_unmarked,
        settings: *settings,
        counters_bytes: counters,
        dropped,
        candidates: candidate_count,
        archive,
    })
}

// --- Rules ------------------------------------------------------------------------------

/// A rule ready to apply.
#[derive(Debug, Clone)]
enum Compiled {
    Literal,
    Pattern(Box<Regex>),
}

/// Ordered redaction rules, ready to apply (module documentation, "Rules").
#[derive(Debug, Clone)]
pub struct RedactionRules {
    rules: Vec<RedactionRule>,
    compiled: Vec<Compiled>,
}

/// What a rule is counted as in the memory budget.
fn rule_bytes(rule: &RedactionRule) -> u64 {
    let own = (size_of::<RedactionRule>() + rule.fragment.len() + rule.placeholder.len()) as u64;
    match rule.kind {
        RedactionKind::Literal => own,
        RedactionKind::Pattern => own + REGEX_HELD_BYTES,
    }
}

/// Rules are equal when they are the same rules in the same order; the compiled form
/// follows from them.
impl PartialEq for RedactionRules {
    fn eq(&self, other: &Self) -> bool {
        self.rules == other.rules
    }
}

fn compile(pattern: &str) -> Result<Regex, regex::Error> {
    RegexBuilder::new(pattern)
        .size_limit(REGEX_SIZE_LIMIT)
        .dfa_size_limit(REGEX_DFA_SIZE_LIMIT)
        .build()
}

impl RedactionRules {
    /// No rules: every text stays as it is.
    pub fn none() -> Self {
        Self {
            rules: Vec::new(),
            compiled: Vec::new(),
        }
    }

    /// The rules `rules`, in their order (confirmed, edited or added by the analyst,
    /// requirement 16.5).
    ///
    /// - [`L2dError::InvalidInput`]: more than `MAX_REDACTIONS` rules, an empty or too
    ///   long fragment, a too long placeholder, or a pattern that does not compile
    ///   within the size limits; the rule is named by its number.
    /// - [`L2dError::Refused`]: what the rules take ([`RedactionRules::held_bytes`])
    ///   does not fit what `budget` leaves; checked before a pattern is compiled.
    ///
    /// The budget is asked, not changed: the rules are held by whoever keeps them, and
    /// that owner counts [`RedactionRules::held_bytes`] in the budget for as long as it
    /// keeps them (a fit from a mark set holds them for the fit; a scorer counts them in
    /// its own [`held_bytes`](crate::BaselineScorer::held_bytes)).
    pub fn new(rules: Vec<RedactionRule>, budget: &MemoryBudget) -> Result<Self, L2dError> {
        check_redactions(&rules)?;
        let held: u64 = rules.iter().map(rule_bytes).sum();
        if held > budget.left_bytes() {
            return Err(L2dError::Refused {
                reason: format!(
                    "the {} redaction rules take {} ({} for each pattern), but the memory \
                     limit of {} leaves {} beside what is held; nothing was compiled",
                    rules.len(),
                    exact(held),
                    exact(REGEX_HELD_BYTES),
                    exact(budget.limit_bytes()),
                    exact(budget.left_bytes())
                ),
            });
        }
        Self::loaded(rules)
    }

    /// The rules of a saved scorer, compiled without asking a budget: the load counts
    /// them beforehand (`BaselineScorer::load_bytes`, which counts every pattern as
    /// `REGEX_HELD_BYTES`). Errors as [`RedactionRules::new`] for their shape.
    pub(crate) fn loaded(rules: Vec<RedactionRule>) -> Result<Self, L2dError> {
        check_redactions(&rules)?;
        let compiled = rules
            .iter()
            .enumerate()
            .map(|(index, rule)| match rule.kind {
                RedactionKind::Literal => Ok(Compiled::Literal),
                RedactionKind::Pattern => compile(&rule.fragment)
                    .map(|regex| Compiled::Pattern(Box::new(regex)))
                    .map_err(|error| L2dError::InvalidInput {
                        detail: format!(
                            "redaction rule {} is a pattern that cannot be used: {error}",
                            index + 1
                        ),
                    }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { rules, compiled })
    }

    /// The proposals the analyst confirmed, as literal rules with
    /// [`DEFAULT_PLACEHOLDER`]: the longest fragment first, then by bytes, each once.
    /// Errors as [`RedactionRules::new`].
    pub fn confirmed<'p>(
        proposals: impl IntoIterator<Item = &'p RedactionProposal>,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        let mut fragments: Vec<&str> = proposals
            .into_iter()
            .map(|proposal| proposal.fragment.as_str())
            .collect();
        fragments.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        fragments.dedup();
        Self::new(
            fragments
                .into_iter()
                .map(|fragment| RedactionRule {
                    kind: RedactionKind::Literal,
                    fragment: fragment.to_string(),
                    placeholder: DEFAULT_PLACEHOLDER.to_string(),
                })
                .collect(),
            budget,
        )
    }

    /// The rule of the routing option `strip` of requirement 2.6: [`ROUTING_PATTERN`]
    /// replaced by nothing. With it, the plain feature rule gives the rows of
    /// [`RoutingMode::Strip`](crate::RoutingMode::Strip).
    pub fn routing_rule() -> RedactionRule {
        RedactionRule {
            kind: RedactionKind::Pattern,
            fragment: ROUTING_PATTERN.to_string(),
            placeholder: String::new(),
        }
    }

    /// The rules of [`RedactionRules::routing_rule`] alone. It holds one compiled
    /// pattern ([`RedactionRules::held_bytes`]), checked against `budget` before it is
    /// compiled, as [`RedactionRules::new`] checks.
    pub fn routing(budget: &MemoryBudget) -> Result<Self, L2dError> {
        Self::new(vec![Self::routing_rule()], budget)
    }

    /// The rules kept with the mark set `marks`. Errors as [`RedactionRules::new`].
    pub fn of_mark_set(marks: &MarkSet, budget: &MemoryBudget) -> Result<Self, L2dError> {
        Self::new(marks.redactions().to_vec(), budget)
    }

    /// Keeps these rules with `marks` and saves it ([`MarkSet::save`]), so that they
    /// are part of the mark set from now on. On an error `marks` keeps the rules it had.
    pub fn save_with(
        &self,
        marks: &mut MarkSet,
        paths: &ProjectPaths,
        budget: &MemoryBudget,
    ) -> Result<PathBuf, L2dError> {
        let before = marks.redactions().to_vec();
        marks.set_redactions(self.rules.clone())?;
        let saved = marks.save(paths, budget);
        if saved.is_err() {
            marks.set_redactions(before)?;
        }
        saved
    }

    /// The rules, in order.
    pub fn rules(&self) -> &[RedactionRule] {
        &self.rules
    }

    /// Whether there are no rules.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The most bytes a text of `bytes` bytes can have after the rules, rule by rule: a
    /// literal rule whose placeholder is longer than its fragment replaces at most
    /// `bytes / fragment` occurrences; a pattern rule with a placeholder may match the
    /// empty text at every position, so it adds at most `(bytes + 1) x placeholder`; a
    /// rule whose placeholder is not longer than what it replaces adds nothing. Equal to
    /// `bytes` for no rules and for the rules of confirmed proposals (one-byte
    /// placeholders). Saturates.
    pub fn redacted_bound(&self, bytes: u64) -> u64 {
        self.rules.iter().fold(bytes, |bytes, rule| {
            let placeholder = rule.placeholder.len() as u64;
            match rule.kind {
                RedactionKind::Literal => {
                    let fragment = rule.fragment.len() as u64;
                    if placeholder <= fragment {
                        bytes
                    } else {
                        bytes.saturating_add(
                            (bytes / fragment).saturating_mul(placeholder - fragment),
                        )
                    }
                }
                RedactionKind::Pattern if placeholder == 0 => bytes,
                RedactionKind::Pattern => {
                    bytes.saturating_add(bytes.saturating_add(1).saturating_mul(placeholder))
                }
            }
        })
    }

    /// What the rules are counted as in the memory budget: their text, and
    /// `REGEX_HELD_BYTES` for every compiled pattern.
    pub fn held_bytes(&self) -> u64 {
        self.rules.iter().map(rule_bytes).sum()
    }

    /// `text` after every rule in order; borrowed when no rule changed it.
    pub fn apply<'t>(&self, text: &'t str) -> Cow<'t, str> {
        let mut current = Cow::Borrowed(text);
        for (rule, compiled) in self.rules.iter().zip(&self.compiled) {
            let replaced = match compiled {
                Compiled::Literal => replace_literal(&current, &rule.fragment, &rule.placeholder),
                Compiled::Pattern(regex) => {
                    match regex.replace_all(&current, NoExpand(&rule.placeholder)) {
                        Cow::Borrowed(_) => None,
                        Cow::Owned(replaced) => Some(replaced),
                    }
                }
            };
            if let Some(replaced) = replaced {
                current = Cow::Owned(replaced);
            }
        }
        current
    }

    /// The SHA-256, in lower-case hexadecimal, of the rules: a tag, the number of
    /// rules, and for each its kind (0 literal, 1 pattern), its fragment and its
    /// placeholder, each with its length (32-bit little-endian). Equal for equal rules
    /// in equal order, different when any of them differs.
    pub fn content_sha256(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"l2d-redaction-rules\0");
        hash.update((self.rules.len() as u32).to_le_bytes());
        for rule in &self.rules {
            hash.update([match rule.kind {
                RedactionKind::Literal => 0u8,
                RedactionKind::Pattern => 1u8,
            }]);
            for part in [&rule.fragment, &rule.placeholder] {
                hash.update((part.len() as u32).to_le_bytes());
                hash.update(part.as_bytes());
            }
        }
        format!("{:x}", hash.finalize())
    }
}

/// `text` with every occurrence of `fragment` that does not continue a token replaced
/// by `placeholder`; `None` when there is none.
fn replace_literal(text: &str, fragment: &str, placeholder: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let wanted = fragment.as_bytes();
    let (Some(&first), Some(&last)) = (wanted.first(), wanted.last()) else {
        return None;
    };
    let (open_start, open_end) = (is_token_byte(first), is_token_byte(last));
    let mut out: Option<String> = None;
    let mut copied = 0;
    let mut from = 0;
    while let Some(found) = text[from..].find(fragment) {
        let at = from + found;
        let end = at + wanted.len();
        let starts = !open_start || at == 0 || !is_token_byte(bytes[at - 1]);
        let ends = !open_end || end == bytes.len() || !is_token_byte(bytes[end]);
        if starts && ends {
            let replaced = out.get_or_insert_with(|| String::with_capacity(text.len()));
            replaced.push_str(&text[copied..at]);
            replaced.push_str(placeholder);
            copied = end;
            from = end;
        } else {
            from = at + text[at..].chars().next().map_or(1, char::len_utf8);
        }
    }
    out.map(|mut replaced| {
        replaced.push_str(&text[copied..]);
        replaced
    })
}

impl CharHasher {
    /// The feature row of `text` after `rules` (requirement 16.7): the rules apply to
    /// the whole text first, then the routing mode splits what they left, as
    /// [`CharHasher::row`] does. The same call at fitting and at scoring gives the same
    /// row for the same text.
    pub fn row_redacted(&self, text: &str, rules: &RedactionRules) -> SparseRow {
        self.row(&rules.apply(text))
    }
}
