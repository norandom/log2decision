//! A score fitted from a mark set (requirements 16.6, 16.7, 16.8, 16.3, 16.12, 6.1,
//! 6.6; task 1.23).
//!
//! The analyst's marks are the labels: [`fit_from_marks`] fits the baseline scorer of
//! requirement 2 on the logs a mark set marks, against the other class, with the mark
//! set's redaction rules applied to every text before it is hashed. The scorer it returns
//! keeps the rules and applies them first to every text it scores, and keeps the mark
//! sets it was fitted from ([`MarkSetOrigin`]); both are saved with it and are part of its
//! revision (`scorer.rs`, "Revision"). Labelled datasets play no part here: a CSIC or AIT
//! label can stand in for an analyst's marks only when a score's agnosticism is measured.
//!
//! # Which logs are fitted
//!
//! The input (or store) is read again, in its order, and each log is put on one side:
//!
//! - **marked**: marked by the mark set and not de-selected; it gets the mark set's label;
//! - **other class**: with a mark set of the other label, the logs it marks and does not
//!   de-select; without one, every log of the same input or store the mark set does not
//!   mark (the owner's decision of 2026-11-15). It gets the other label. Which of the two
//!   was used is stated ([`OtherClass`]) and saved;
//! - **left out**: a log de-selected in either mark set (16.3: kept out of fitting, and
//!   not counted as the other class either; the redaction proposals count it apart the
//!   same way), a log both mark sets mark (it has no label, as a text that occurred with
//!   both labels in a labelled input has none), and, with a mark set of the other label,
//!   a log neither marks. Each is counted ([`MarkFitCounts`]).
//!
//! The rows of the fit are the texts as they were read, one row per distinct text, in
//! the input's order. Texts that become equal under the rules stay rows of their own:
//! nothing is merged, so no rule for occurrence counts or for opposite labels is needed,
//! and the logs frame of the fit keeps texts that are unique, as every logs frame does.
//! The hashes of the fitted texts are those of the texts as read. A store can hold one
//! text from two sources; such a text is one row, its occurrences added, and left out
//! when its logs are on both sides.
//!
//! Below the counts of requirement 2.3 (at least 10 logs, at least 5 of each label) the
//! fit is refused with the marked, de-selected and other-class counts (16.8).
//!
//! # Group keys
//!
//! Each fitted log carries a group key for the held-out-group report of 16.9 (`l2d-eval`,
//! `markscore`): one default per kind of input, named in the output and settable
//! ([`GroupKeyRule`]): the group key the adapter gave a file's logs (HTTP CSIC 2010: the
//! URL path), the host of an archive entry (AIT: `gather/<host>/...`), none for a store;
//! or the first group of a pattern over the text.
//!
//! # Memory (16.12, requirements 12 and 15)
//!
//! The mark sets, the rules and a group-key pattern are counted first; the input is read
//! in the batches of its reader with what the fit has taken so far held beside them (an
//! archive sizes its next batch with it, a store cuts its batches to it), and the read is
//! refused with the figures when that no longer fits. Building the logs frame of the fit
//! is checked with `require_whole` before it is built, and the fit checks its own estimate
//! with the frame held. The budget holds afterwards what it held before; the logs frame
//! that is returned is the caller's to count ([`MarkFit::held_bytes`]).

use std::collections::HashMap;
use std::fmt;

use l2d_core::input::{PARSE_EXPANSION, PARSE_ROW_BYTES};
use l2d_core::marks::{
    MarkSet, MarkedSource, REGEX_DFA_SIZE_LIMIT, REGEX_HELD_BYTES, REGEX_SIZE_LIMIT,
};
use l2d_core::memory::exact;
use l2d_core::{ArchiveSummary, InputSummary, L2dError, Label, LogRecord, LogsFrame, MemoryBudget};
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use crate::redact::{
    LogSink, LogView, MarkSide, RedactionInput, RedactionRules, marks_bytes, read_logs,
};
use crate::scorer::{BaselineScorer, BaselineSettings, MIN_LABELLED, MIN_PER_LABEL, text_hash};

/// What a fitted log is counted as in memory beside its text and group key: its record
/// (72 bytes) twice over, for the growth of the list that holds the records, and the
/// slack of its strings.
const ROW_BYTES: u64 = 160;

/// What a stored log is counted as beside that: its entry in the table of texts met
/// (`by_text`), with the table's growth.
const STORE_ROW_BYTES: u64 = 48;

/// Of what a store charges a batch row for its text: twice the text
/// (`l2d_store`'s `Store::text_batches_with`, and the `MarkStore` contract).
const STORE_TEXT_FACTOR: f64 = 2.0;

/// The operation, as refusals name it.
const OPERATION: &str = "fitting a score from a mark set";

/// The operation, as the reader of an archive or store names it before the mark set.
const READING: &str = "fitting a score";

/// How the group key of a log is found (requirement 16.9: one default per kind of log,
/// named in the output, settable by the analyst).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupKeyRule {
    /// The group key the input adapter gave the log; for HTTP CSIC 2010 the URL path.
    /// The default for a file.
    InputGroup,
    /// The host of the archive entry the log was first read from: the directory after a
    /// leading `gather/` (the layout of the AIT Log Data Set), or else the first
    /// directory of the entry path; none for an entry at the top of the archive. The
    /// default for an archive.
    ArchiveHost,
    /// The first group of the first match of a regular expression over the text, or the
    /// whole match when the pattern has no group; none when it does not match.
    Pattern { pattern: String },
    /// No group key. The default for a store, whose batches carry no group keys.
    None,
}

impl GroupKeyRule {
    /// The default for the kind of `input`.
    pub fn default_for(input: &RedactionInput<'_>) -> Self {
        match input {
            RedactionInput::File { .. } => GroupKeyRule::InputGroup,
            RedactionInput::Archive { .. } => GroupKeyRule::ArchiveHost,
            RedactionInput::Store(_) => GroupKeyRule::None,
        }
    }

    /// The rule as the output names it.
    pub fn name(&self) -> String {
        match self {
            GroupKeyRule::InputGroup => {
                "the group key of the input (HTTP CSIC 2010: the URL path)".into()
            }
            GroupKeyRule::ArchiveHost => {
                "the host of the archive entry (AIT: the directory after gather/)".into()
            }
            GroupKeyRule::Pattern { pattern } => {
                format!("the first group of the pattern `{pattern}`, or its whole match")
            }
            GroupKeyRule::None => "no group key".into(),
        }
    }
}

/// The host of an archive entry path ([`GroupKeyRule::ArchiveHost`]).
pub fn archive_host(entry: &str) -> Option<&str> {
    let rest = entry.strip_prefix("gather/").unwrap_or(entry);
    let (host, _) = rest.split_once('/')?;
    (!host.is_empty()).then_some(host)
}

/// The settings of [`fit_from_marks`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MarkFitSettings {
    /// The baseline scorer's settings (requirement 2).
    pub baseline: BaselineSettings,
    /// The group key; `None` takes the default for the kind of input.
    pub group_key: Option<GroupKeyRule>,
}

/// What a score was fitted against (16.6), as the output states it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtherClass {
    /// The logs of the same input or store that the mark set does not mark.
    Unmarked,
    /// The analyst's mark set of the other label.
    MarkSet { name: String },
}

impl fmt::Display for OtherClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OtherClass::Unmarked => f.write_str("the unmarked logs of the same input or store"),
            OtherClass::MarkSet { name } => write!(f, "the mark set `{name}`"),
        }
    }
}

/// What a fit from a mark set took of the logs it read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MarkFitCounts {
    /// The logs read (distinct texts of the input, or stored logs).
    pub read: u64,
    /// Fitted with the mark set's label.
    pub marked: u64,
    /// Fitted with the other label.
    pub other: u64,
    /// Left out: de-selected in a mark set (16.3).
    pub deselected: u64,
    /// Left out: on both sides (marked by both mark sets, or a stored text whose logs
    /// are on both sides).
    pub conflicting: u64,
    /// Left out: marked by neither of two mark sets.
    pub not_marked: u64,
}

/// The mark sets a score was fitted from, as they were then, and what the fit took of
/// the logs. Saved with the score as copies of the mark-set files; their SHA-256 are
/// part of the score's revision.
#[derive(Debug, Clone)]
pub struct MarkSetOrigin {
    marks: MarkSet,
    marks_file: Vec<u8>,
    marks_sha256: String,
    other: Option<(MarkSet, Vec<u8>, String)>,
    counts: MarkFitCounts,
}

impl PartialEq for MarkSetOrigin {
    fn eq(&self, other: &Self) -> bool {
        // These immutable, validated JSON copies cover every mark-set field and are
        // already retained for the scorer's revision. Comparing them never reads disk.
        self.marks_file == other.marks_file
            && self.marks_sha256 == other.marks_sha256
            && self.other.as_ref().map(|(_, bytes, hash)| (bytes, hash))
                == other.other.as_ref().map(|(_, bytes, hash)| (bytes, hash))
            && self.counts == other.counts
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    l2d_core::input::content_sha256(bytes)
}

impl MarkSetOrigin {
    /// The origin of a score: the mark set and the bytes of its file, the mark set of the
    /// other label with its bytes when there was one, and the counts.
    pub(crate) fn new(
        marks: MarkSet,
        marks_file: Vec<u8>,
        other: Option<(MarkSet, Vec<u8>)>,
        counts: MarkFitCounts,
    ) -> Self {
        let marks_sha256 = sha256_hex(&marks_file);
        let other = other.map(|(set, bytes)| {
            let sha256 = sha256_hex(&bytes);
            (set, bytes, sha256)
        });
        Self {
            marks,
            marks_file,
            marks_sha256,
            other,
            counts,
        }
    }

    /// The mark set, with its de-selections and rules.
    pub fn marks(&self) -> &MarkSet {
        &self.marks
    }

    /// The mark set's file as it is kept with the score.
    pub fn marks_file(&self) -> &[u8] {
        &self.marks_file
    }

    /// SHA-256 of [`MarkSetOrigin::marks_file`].
    pub fn marks_sha256(&self) -> &str {
        &self.marks_sha256
    }

    /// The mark set of the other label; `None` when the score was fitted against the
    /// unmarked logs.
    pub fn other(&self) -> Option<&MarkSet> {
        self.other.as_ref().map(|(set, _, _)| set)
    }

    pub fn other_file(&self) -> Option<&[u8]> {
        self.other.as_ref().map(|(_, bytes, _)| bytes.as_slice())
    }

    pub fn other_sha256(&self) -> Option<&str> {
        self.other.as_ref().map(|(_, _, sha256)| sha256.as_str())
    }

    /// What the score was fitted against.
    pub fn other_class(&self) -> OtherClass {
        match self.other() {
            None => OtherClass::Unmarked,
            Some(other) => OtherClass::MarkSet {
                name: other.name().to_string(),
            },
        }
    }

    pub fn counts(&self) -> &MarkFitCounts {
        &self.counts
    }

    /// What the origin takes in memory: the mark sets twice (as parsed and as bytes).
    pub fn held_bytes(&self) -> u64 {
        let set = |set: &MarkSet, bytes: &[u8]| 2 * bytes.len() as u64 + marks_bytes(set);
        set(&self.marks, &self.marks_file)
            + self
                .other
                .as_ref()
                .map_or(0, |(other, bytes, _)| set(other, bytes))
    }
}

/// A score fitted from a mark set, with what it was fitted on.
#[derive(Debug, Clone)]
pub struct MarkFit {
    /// The score: a baseline scorer with the mark set's rules and its origin.
    pub scorer: BaselineScorer,
    /// The fitted logs as they were read, labelled by the marks (the mark set's label or
    /// the other), with their group keys; in the input's order. The held-out-group report
    /// works on it.
    pub logs: LogsFrame,
    pub counts: MarkFitCounts,
    pub other_class: OtherClass,
    /// The group key the logs carry.
    pub group_key: GroupKeyRule,
    /// For an archive, the summary of the read (with its entry reports).
    pub archive: Option<ArchiveSummary>,
    /// For an archive, the accounting of every batch handed out: what the fit held
    /// after copying it against what it had told the reader it held when the reader sized
    /// that batch. Empty for a file and a store.
    pub archive_batches: Vec<BatchRoom>,
}

/// The accounting of one batch of an archive (review round 2 of task 1.23).
///
/// The reader sizes batch `j` with what the fit told it after batch `j - 2` (after none
/// for the first two): `ZipBatches::hold_beside` takes effect for the batch after the
/// next. The fit refuses, with these figures, when its copies after batch `j` exceed
/// `reserved`; a completed fit has `held_after <= reserved` for every batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchRoom {
    /// 1 for the first batch handed out.
    pub batch: u64,
    /// The logs of the batch the fit took a copy of.
    pub rows: u64,
    /// What the fit held beside the archive when the reader sized the batch (mark
    /// sets, rules, pattern, copies so far, and the room it kept).
    pub reserved: u64,
    /// What the fit held after copying the batch.
    pub held_after: u64,
}

impl MarkFit {
    /// What [`MarkFit::logs`] takes in memory, for the caller to hold while it keeps it.
    pub fn held_bytes(&self) -> u64 {
        self.logs
            .records()
            .map(|record| {
                record.text.len() as u64
                    + record.group.as_ref().map_or(0, |group| group.len() as u64)
                    + ROW_BYTES
            })
            .sum()
    }
}

fn opposite(label: Label) -> Label {
    match label {
        Label::Bad => Label::Good,
        Label::Good => Label::Bad,
        Label::Unknown => Label::Unknown,
    }
}

/// Collects the fitted logs as the reader hands them out.
struct Collector<'m> {
    marks: &'m MarkSet,
    other: Option<&'m MarkSet>,
    rule: &'m GroupKeyRule,
    pattern: Option<&'m Regex>,
    /// What is held beside the collected logs (mark sets, rules, pattern).
    fixed: u64,
    rows: Vec<LogRecord>,
    rows_bytes: u64,
    /// What the budget left when the fit started: the ground the room for the copies of
    /// the next batches is taken from ([`Collector::beside`]).
    base: u64,
    /// How the room for the copies is kept ([`Room`]).
    room_rule: Room,
    /// What a store is asked to charge per row for the collector's copy of it.
    per_row: u64,
    counts: MarkFitCounts,
    marked_seen: u64,
    other_seen: u64,
    /// For a store: the row of each text, by its hash (a text can be stored twice).
    by_text: Option<HashMap<[u8; 16], usize>>,
    /// For a file, whose frame the caller holds and which is not read in batches: what
    /// the limit leaves for the collected logs. An archive and a store check them
    /// between their batches.
    room: Option<u64>,
    /// For an archive: the figures given to the reader so far (one before the first
    /// batch, one after each), the copies of the batch being read, and the accounting
    /// of the batches done ([`BatchRoom`]).
    archive: bool,
    besides: Vec<u64>,
    batch_rows: u64,
    batches: Vec<BatchRoom>,
}

impl Collector<'_> {
    /// For an archive, what the reader was told the fit holds when it sized the batch
    /// being read: the figure given after batch `j - 2` for batch `j`, the first figure
    /// for batches 1 and 2.
    fn reserved(&self) -> Option<u64> {
        if !self.archive || self.besides.is_empty() {
            return None;
        }
        let batch = self.besides.len();
        Some(self.besides[batch.max(2) - 2])
    }

    fn group_key(&self, view: &LogView<'_>) -> Option<String> {
        match self.rule {
            GroupKeyRule::InputGroup => view.group.map(str::to_string),
            GroupKeyRule::ArchiveHost => view.entry.and_then(archive_host).map(str::to_string),
            GroupKeyRule::Pattern { .. } => {
                let regex = self.pattern?;
                let captures = regex.captures(view.text)?;
                captures
                    .get(1)
                    .or_else(|| captures.get(0))
                    .map(|found| found.as_str().to_string())
            }
            GroupKeyRule::None => None,
        }
    }
}

impl LogSink for Collector<'_> {
    fn log(&mut self, view: &LogView<'_>) -> Result<(), L2dError> {
        self.counts.read += 1;
        let primary = MarkSide::of(self.marks, view)?;
        let second = self
            .other
            .map(|other| MarkSide::of(other, view))
            .transpose()?;
        if primary == MarkSide::Marked {
            self.marked_seen += 1;
        }
        if second == Some(MarkSide::Marked) {
            self.other_seen += 1;
        }
        let label = if primary == MarkSide::Deselected || second == Some(MarkSide::Deselected) {
            self.counts.deselected += 1;
            return Ok(());
        } else if primary == MarkSide::Marked && second == Some(MarkSide::Marked) {
            self.counts.conflicting += 1;
            return Ok(());
        } else if primary == MarkSide::Marked {
            self.marks.label()
        } else if second.is_none() || second == Some(MarkSide::Marked) {
            opposite(self.marks.label())
        } else {
            self.counts.not_marked += 1;
            return Ok(());
        };
        let marked = label == self.marks.label();

        if let Some(by_text) = &mut self.by_text {
            // A stored text met before: one row; left out when its logs disagree.
            if let Some(&row) = by_text.get(&text_hash(view.text)) {
                let record = &mut self.rows[row];
                record.occurrences = record.occurrences.saturating_add(view.occurrences);
                if record.label != label && record.label != Label::Unknown {
                    if record.label == self.marks.label() {
                        self.counts.marked -= 1;
                    } else {
                        self.counts.other -= 1;
                    }
                    self.counts.conflicting += 2;
                    record.label = Label::Unknown;
                } else if record.label == Label::Unknown {
                    self.counts.conflicting += 1;
                }
                return Ok(());
            }
            by_text.insert(text_hash(view.text), self.rows.len());
        }
        if marked {
            self.counts.marked += 1;
        } else {
            self.counts.other += 1;
        }
        let group = self.group_key(view);
        self.rows_bytes += view.text.len() as u64
            + group.as_ref().map_or(0, |group| group.len() as u64)
            + ROW_BYTES
            + if self.by_text.is_some() { 48 } else { 0 };
        if let Some(reserved) = self.reserved()
            && self.fixed + self.rows_bytes > reserved
        {
            return Err(L2dError::Refused {
                reason: format!(
                    "{OPERATION} `{}` holds {} after taking {} logs of batch {} of the archive, \
                     more than the {} it held beside the archive when the reader sized that \
                     batch; the room it kept for the copies of a batch was too small; nothing \
                     was fitted",
                    self.marks.name(),
                    exact(self.fixed + self.rows_bytes),
                    self.batch_rows + 1,
                    self.besides.len(),
                    exact(reserved)
                ),
            });
        }
        self.batch_rows += 1;
        if let Some(room) = self.room
            && self.fixed + self.rows_bytes > room
        {
            return Err(L2dError::Refused {
                reason: format!(
                    "{OPERATION} `{}` holds {} for the {} logs it has taken so far and {} \
                     for the mark sets, the rules and a group-key pattern, but the memory \
                     limit leaves {} beside the logs frame of the input; nothing was fitted",
                    self.marks.name(),
                    exact(self.rows_bytes),
                    self.rows.len() + 1,
                    exact(self.fixed),
                    exact(room)
                ),
            });
        }
        self.rows.push(LogRecord {
            text: view.text.to_string(),
            label,
            group,
            first_position: view
                .position
                .or_else(|| view.seq.map(|seq| seq as u64))
                .unwrap_or(0),
            occurrences: view.occurrences,
        });
        Ok(())
    }

    /// The mark sets, rules and pattern, the copies so far, and room for the copies the
    /// next batch can bring: the larger of the most one batch brought and the share of
    /// what is left that a batch cut to the rest can bring ([`copy_share`]). It is asked
    /// before the first batch too, so the first batch is planned with that room.
    fn beside(&mut self) -> u64 {
        if let Some(reserved) = self.reserved() {
            self.batches.push(BatchRoom {
                batch: self.besides.len() as u64,
                rows: self.batch_rows,
                reserved,
                held_after: self.fixed + self.rows_bytes,
            });
        }
        self.batch_rows = 0;
        let held = self.fixed + self.rows_bytes;
        let beside = held
            + self
                .room_rule
                .room(self.base, held, self.besides.last().copied());
        if !self.archive {
            return beside;
        }
        // What the reader was told never shrinks, so a batch sized later than the rule
        // below assumes (after a batch without new texts) has no more room.
        let beside = beside.max(self.besides.last().copied().unwrap_or(0));
        self.besides.push(beside);
        beside
    }

    fn archive_opened(&mut self, longest_entry_path: u64, max_record_bytes: u64) {
        if let Room::Archive {
            group_per_row,
            record_bytes,
            host,
            ..
        } = &mut self.room_rule
        {
            // A host is a part of an entry path.
            if *host {
                *group_per_row = longest_entry_path;
            }
            *record_bytes = max_record_bytes;
        }
    }

    fn per_row(&self) -> u64 {
        self.per_row
    }
}

/// How the collector keeps room for its copies of the batches to come (review rounds 1
/// and 2 of task 1.23). Notation: `base` is what the budget left when the fit began,
/// `H` what the collector holds now (mark sets, rules, pattern and copies), `B_k` the
/// figure it gives the reader after batch `k` (`B_0` before the first), `c` the most
/// copies one batch can bring per byte of the room it was sized in. A batch sized in
/// `room` bytes holds at most `room / PARSE_EXPANSION` bytes of text and
/// `room / PARSE_ROW_BYTES` rows (`ZipInput`'s batch rule, and the rule of its frames
/// from disk), plus the one record that crosses the limit; a store's at most `room / 2`
/// bytes of text. A copy takes its text, a group key as long as the text at most when
/// it comes from a pattern (`text_copies` 2, else 1) or as long as the longest entry
/// path at most when it is an archive host (`group_per_row`), and [`ROW_BYTES`].
#[derive(Debug, Clone, Copy)]
enum Room {
    /// A file is not read in batches; its copies are checked against what the limit
    /// leaves beside its frame.
    File,
    /// A store cuts every batch with the figure given after the batch before
    /// (`MarkStore::text_batches`) and is asked the per-row part as `per_row`:
    /// `c = text_copies / 2`, and keeping `R = c / (1 + c) x (base - H)` leaves the next
    /// batch at most `base - H - R`, whose copies take at most `c x (base - H - R) = R`.
    Store { text_copies: f64 },
    /// An archive sizes batch `j` with `B_(j-2)` (`B_0` for batches 1 and 2):
    /// `ZipBatches::hold_beside` reaches the budget when the next batch completes, not
    /// before it is sized. So when batch `k + 2` is copied the collector holds
    /// `H_k + copies(k+1) + copies(k+2)`, and the reader sized batch `k + 2` with `B_k`.
    /// That is within the limit when `R_k = B_k - H_k` covers both batches:
    ///
    /// - batch `k + 1` was sized with `B_(k-1)` (with `B_0` for `k = 0`), so
    ///   `copies(k+1) <= c x (base - B_(k-1)) + one record`;
    /// - batch `k + 2` is sized with `B_k = H_k + R_k`, so
    ///   `copies(k+2) <= c x (base - H_k - R_k) + one record`.
    ///
    /// For `k = 0` both are sized with `B_0`: `R_0 = 2c / (1 + 2c) x (base - H_0)`. For
    /// `k >= 1`: `R_k = c x ((base - B_(k-1)) + (base - H_k)) / (1 + c)`. Each adds two
    /// records' copies (`text_copies x record_bytes + ROW_BYTES + group_per_row`), and
    /// `B_k` never goes below `B_(k-1)` (a batch sized later, after one without new
    /// texts, then has no more room than assumed). The collector checks every batch
    /// against the figure it was sized with and refuses with the figures otherwise
    /// ([`BatchRoom`]).
    Archive {
        text_copies: f64,
        host: bool,
        group_per_row: u64,
        record_bytes: u64,
    },
}

impl Room {
    fn of(input: &RedactionInput<'_>, rule: &GroupKeyRule) -> Self {
        let text_copies = if matches!(rule, GroupKeyRule::Pattern { .. }) {
            2.0
        } else {
            1.0
        };
        match input {
            RedactionInput::File { .. } => Room::File,
            RedactionInput::Store(_) => Room::Store { text_copies },
            RedactionInput::Archive { settings, .. } => Room::Archive {
                text_copies,
                host: matches!(rule, GroupKeyRule::ArchiveHost),
                group_per_row: 0,
                record_bytes: settings.max_record_bytes,
            },
        }
    }

    /// The room to keep beside what is held (`held`), with `previous` the figure given
    /// after the batch before (`None` before the first batch).
    fn room(&self, base: u64, held: u64, previous: Option<u64>) -> u64 {
        let free = base.saturating_sub(held) as f64;
        match *self {
            Room::File => 0,
            Room::Store { text_copies } => {
                let c = text_copies / STORE_TEXT_FACTOR;
                (c / (1.0 + c) * free).ceil() as u64
            }
            Room::Archive {
                text_copies,
                group_per_row,
                record_bytes,
                ..
            } => {
                let per_row = (ROW_BYTES + group_per_row) as f64;
                let c = text_copies / PARSE_EXPANSION + per_row / PARSE_ROW_BYTES as f64;
                let record = text_copies * record_bytes as f64 + per_row;
                let room = match previous {
                    None => 2.0 * c / (1.0 + 2.0 * c) * free,
                    Some(previous) => {
                        let sized = base.saturating_sub(previous) as f64;
                        c * (sized + free) / (1.0 + c)
                    }
                };
                (room + 2.0 * record).ceil() as u64
            }
        }
    }
}

fn compile_group_pattern(pattern: &str) -> Result<Regex, L2dError> {
    RegexBuilder::new(pattern)
        .size_limit(REGEX_SIZE_LIMIT)
        .dfa_size_limit(REGEX_DFA_SIZE_LIMIT)
        .build()
        .map_err(|error| L2dError::InvalidInput {
            detail: format!("the group-key pattern `{pattern}` cannot be used: {error}"),
        })
}

/// Fits a score from the mark set `marks` on the input or store it was made on
/// (requirements 16.6, 16.7, 16.8; module documentation): the baseline scorer with the
/// mark set's redaction rules, on its marked logs after de-selection, against the logs
/// `other` marks when it is given (a mark set of the other label on the same input or
/// store) and else against the unmarked logs of the same input or store.
///
/// The input is checked first: a file by `MarkSet::require_input` on its summary, an
/// archive by its content hash and entry patterns when it is opened, a store by
/// `MarkSet::require_store` before and after the read.
///
/// - [`L2dError::InvalidInput`]: `other` has the same label; a group-key pattern that
///   cannot be used; settings out of range.
/// - [`L2dError::Refused`]: `other` was made on another input or store; another input,
///   or one read otherwise; the marked logs are not all among the logs read; fewer than
///   10 logs or fewer than 5 of a label after de-selection (16.8, with the counts); what
///   the fit holds does not fit the memory limit (with the figures).
///
/// The budget holds afterwards what it held before.
pub fn fit_from_marks(
    marks: &MarkSet,
    other: Option<&MarkSet>,
    input: RedactionInput<'_>,
    settings: &MarkFitSettings,
    budget: &mut MemoryBudget,
) -> Result<MarkFit, L2dError> {
    let guard = l2d_core::resource::ResourceGuard::tracked(
        "mark-set training",
        l2d_core::resource::busy_choice(),
        budget.usage(),
    )?;
    let held = budget.held_bytes();
    let fitted = fit(marks, other, &input, settings, budget, held);
    budget.hold(held);
    guard.finish()?;
    fitted
}

fn fit(
    marks: &MarkSet,
    other: Option<&MarkSet>,
    input: &RedactionInput<'_>,
    settings: &MarkFitSettings,
    budget: &mut MemoryBudget,
    held: u64,
) -> Result<MarkFit, L2dError> {
    let name = marks.name();
    if let Some(other) = other {
        if other.label() == marks.label() {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "the mark set `{}` has the label {} of the mark set `{name}`; a score is \
                     fitted against marks of the other label",
                    other.name(),
                    other.label().as_str()
                ),
            });
        }
        if other.source() != marks.source() {
            let describe = |source: &MarkedSource| match source {
                MarkedSource::Input(input) => {
                    format!(
                        "the input {} (SHA-256 {})",
                        input.path, input.content_sha256
                    )
                }
                MarkedSource::Store(store) => {
                    format!(
                        "the store {} (SHA-256 {})",
                        store.name, store.content_sha256
                    )
                }
            };
            return Err(L2dError::Refused {
                reason: format!(
                    "the mark set `{}` was made on {}, the mark set `{name}` on {}; a score \
                     is fitted against marks of the same logs",
                    other.name(),
                    describe(other.source()),
                    describe(marks.source())
                ),
            });
        }
    }
    match input {
        RedactionInput::File { summary, .. } => marks.require_input(summary, None)?,
        RedactionInput::Store(store) => marks.require_store(*store, budget)?,
        RedactionInput::Archive { .. } => {}
    }

    let rule = settings
        .group_key
        .clone()
        .unwrap_or_else(|| GroupKeyRule::default_for(input));
    let rules = RedactionRules::of_mark_set(marks, budget)?;
    let pattern_bytes = if matches!(rule, GroupKeyRule::Pattern { .. }) {
        REGEX_HELD_BYTES
    } else {
        0
    };
    let fixed =
        marks_bytes(marks) + other.map_or(0, marks_bytes) + rules.held_bytes() + pattern_bytes;
    if fixed > budget.left_bytes() {
        return Err(L2dError::Refused {
            reason: format!(
                "{OPERATION} `{name}` holds {} before it reads a log (the mark sets, the \
                 redaction rules and a group-key pattern), but the memory limit of {} leaves \
                 {}; nothing was read",
                exact(fixed),
                exact(budget.limit_bytes()),
                exact(budget.left_bytes())
            ),
        });
    }
    let pattern = match &rule {
        GroupKeyRule::Pattern { pattern } => Some(compile_group_pattern(pattern)?),
        _ => None,
    };

    let mut collector = Collector {
        marks,
        other,
        rule: &rule,
        pattern: pattern.as_ref(),
        fixed,
        rows: Vec::new(),
        rows_bytes: 0,
        base: budget.left_bytes(),
        room_rule: Room::of(input, &rule),
        per_row: if matches!(input, RedactionInput::Store(_)) {
            ROW_BYTES + STORE_ROW_BYTES
        } else {
            0
        },
        counts: MarkFitCounts::default(),
        marked_seen: 0,
        other_seen: 0,
        by_text: matches!(input, RedactionInput::Store(_)).then(HashMap::new),
        room: matches!(input, RedactionInput::File { .. }).then(|| budget.left_bytes()),
        archive: matches!(input, RedactionInput::Archive { .. }),
        besides: Vec::new(),
        batch_rows: 0,
        batches: Vec::new(),
    };
    // A file's frame is held by the caller already; an archive's reader and a store
    // put what the collector holds (`fixed` among it) and the room for the copies of
    // the next batch beside their batches, and a store charges `per_row` for each copy.
    let archive = read_logs(READING, marks, input, held, budget, &mut collector)?;
    if let RedactionInput::Store(store) = input {
        marks.require_store(*store, budget)?;
    }
    let Collector {
        rows,
        rows_bytes,
        counts,
        marked_seen,
        other_seen,
        batches: archive_batches,
        ..
    } = collector;
    for (set, seen) in [(Some(marks), marked_seen), (other, other_seen)] {
        let Some(set) = set else { continue };
        if seen != set.selected_count() {
            return Err(L2dError::Refused {
                reason: format!(
                    "the mark set `{}` names {} marked logs that are not de-selected, but {seen} \
                     of them are among the {} logs read; read the input as it was read when it \
                     was marked",
                    set.name(),
                    set.selected_count(),
                    counts.read
                ),
            });
        }
    }

    let other_class = match other {
        None => OtherClass::Unmarked,
        Some(other) => OtherClass::MarkSet {
            name: other.name().to_string(),
        },
    };
    if counts.marked < MIN_PER_LABEL
        || counts.other < MIN_PER_LABEL
        || counts.marked + counts.other < MIN_LABELLED
    {
        let conflicting = if counts.conflicting > 0 {
            format!(", {} on both sides", counts.conflicting)
        } else {
            String::new()
        };
        return Err(L2dError::Refused {
            reason: format!(
                "{OPERATION} `{name}` needs at least {MIN_LABELLED} labelled logs and at least \
                 {MIN_PER_LABEL} of each label (requirement 2.3); after de-selection it has {} \
                 marked logs ({} marked, {} de-selected{conflicting}) and {} logs of the other \
                 class ({other_class}); nothing was fitted",
                counts.marked,
                marks.count(),
                marks.deselected().len() + marks.deselected_id_count(),
                counts.other
            ),
        });
    }

    // The records are copied into the logs frame: both exist until the records go.
    budget.hold(held + fixed + rows_bytes);
    budget.require_whole(
        rows_bytes,
        "building the logs frame of a fit from a mark set",
    )?;
    let logs = LogsFrame::from_records(&rows)?;
    drop(rows);
    budget.hold(held + fixed + rows_bytes);

    let summary = match (input, &archive) {
        (RedactionInput::File { summary, .. }, _) => (*summary).clone(),
        (RedactionInput::Archive { .. }, Some(archive)) => archive.input.clone(),
        _ => {
            let store = marks.store().map_or_else(Default::default, |store| {
                (
                    format!("store:{}", store.name),
                    store.content_sha256.clone(),
                )
            });
            InputSummary {
                source_path: store.0.into(),
                records: counts.read,
                distinct_texts: counts.read,
                content_sha256: store.1,
                skipped: Vec::new(),
                label_conflicts: 0,
            }
        }
    };
    let scorer = BaselineScorer::fit_redacted_with(
        &logs,
        std::slice::from_ref(&summary),
        &settings.baseline,
        &rules,
        budget,
    )?;
    let origin = MarkSetOrigin::new(
        marks.clone(),
        marks.to_file_bytes()?,
        other
            .map(|other| other.to_file_bytes().map(|bytes| (other.clone(), bytes)))
            .transpose()?,
        counts,
    );
    Ok(MarkFit {
        scorer: scorer.with_origin(origin),
        logs,
        counts,
        other_class,
        group_key: rule,
        archive,
        archive_batches,
    })
}
