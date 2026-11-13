//! Conformal calibration: from a scorer's ratio to a label set with a stated error rate
//! (requirement 11).
//!
//! # Method
//!
//! Split (inductive) conformal prediction, label-conditional. It works on the ratio of
//! any scorer, so one calibration layer serves the baseline and the encoder-based
//! scorer, and the scorer is not fitted again.
//!
//! - **Calibration logs** are labelled logs the scorer was not fitted on, each with the
//!   ratio the scorer gives it.
//! - **Nonconformity** of a log for a label: `ratio` for good, `1 - ratio` for bad,
//!   computed in double precision from the 32-bit ratio. A higher value means the log
//!   fits the label worse.
//! - **P-value** of a new log for a label:
//!   `(1 + number of calibration logs of that label with nonconformity >= the new log's)
//!   / (n + 1)`, where `n` is the length of the list used. Ties count as greater or
//!   equal. Nothing is randomized.
//! - **Label set** at error rate `e`: every label whose p-value is greater than `e`:
//!   good, bad, both (the log is uncertain) or neither (the log is unlike the
//!   calibration logs of either label).
//! - **Guarantee**: for new logs that are exchangeable with the calibration logs of a
//!   list, the true label is missing from the set with probability at most `e`, for
//!   good logs and for bad logs separately (requirement 11.3). It says nothing about
//!   logs that do not resemble the calibration logs ([`EXCHANGEABILITY_NOTE`]).
//!
//! ## Lists (strata)
//!
//! Each label has a **pooled** list of all its calibration logs. A group that holds at
//! least [`CalibrationSettings::min_per_group`] calibration logs of a label gets a list
//! of its own for that label in addition (requirement 11.4). A new log is judged, per
//! label, on its group's list where there is one and on the pooled list otherwise; a
//! log without a group or of a group never seen always uses the pooled lists. Group
//! names are compared byte for byte. One log can therefore use its group's list for one
//! label and the pooled list for the other, and [`Calibrated`] names the list used for
//! each label.
//!
//! ## Supported error rates
//!
//! The smallest p-value a list of `n` logs can give is `1 / (n + 1)`, so a list can rule
//! a label out only at error rates of at least that. An error rate below the
//! `1 / (n + 1)` of the shortest list of the calibration is refused for every log with
//! [`CalibrationError::TooFewLogs`], which states the lengths of the shortest list of
//! each label and the smallest supported rate (requirement 11.5). A calibration without
//! logs of one of the labels supports no rate and is refused when it is fitted.
//!
//! ## Numbers
//!
//! P-values and error rates are compared as 32-bit floats: the p-value is computed in
//! double precision and rounded to `f32`, and that value is both what [`Calibrated`]
//! reports and what is compared with the error rate. A set can therefore be checked
//! from the reported numbers, and a rate typed as a decimal (`0.01`) is treated as the
//! fraction it stands for (`1 / 100`) at the limit.
//!
//! # The count
//!
//! Everything here is own code (owner decision of 2026-11-12, task 1.18). Each list is
//! kept once, sorted ascending, which is also the form it is saved and hashed in. The
//! number of calibration logs whose nonconformity is at least the new log's is the
//! length of the list less the number of values below the new one, found by binary
//! search; a decision therefore costs time in proportion to the logarithm of the
//! lengths of the two lists it uses. The p-value is
//! `(count as f64 + 1.0) / (n as f64 + 1.0)` rounded to `f32`.
//!
//! Task 1.13 took the count from the crate `conformal-prediction` 2.0.1, which scanned
//! a second copy of each list. Count and arithmetic are the same, so p-values, label
//! sets, revisions and saved files are those of task 1.13;
//! `tests/conformal_fixture.rs` compares them with files written by that code.
//!
//! # Memory
//!
//! Calibration needs all calibration points at once (requirement 12.5).
//! [`ConformalCalibrator::fit_with`] checks [`estimated_bytes`] with
//! [`MemoryBudget::require_whole`] before it allocates anything.
//!
//! A calibration, fitted or loaded, is not transient: it keeps its lists for as long as
//! it lives. [`ConformalCalibrator::held_bytes`] states how much (8 bytes per value of
//! every list, and a little per list and group), and an operation that uses a
//! calibration holds that figure in its memory budget (requirement 12.9); triage does.
//! [`ConformalCalibrator::load`] itself takes no budget: while it runs it takes about
//! twice what the calibration then keeps, up to three times for a calibration of many
//! small groups (measured in `tests/triage_memory.rs`).
//!
//! # Abstentions
//!
//! A log its scorer abstained on has no ratio and is not a calibration log:
//! [`calibration_points`] leaves it out. [`ConformalCalibrator::fit_baseline`] and
//! [`ConformalCalibrator::fit_scorer`] (any scorer under the [`Scorer`] contract) go
//! through it.
//!
//! # Saved form
//!
//! [`ConformalCalibrator::save`] writes `models/calibrations/<name>-<rev12>/` with two
//! files, following the conventions of the baseline scorer:
//!
//! - `scores.bin`: the nonconformity values of all lists, list after list, each list
//!   sorted ascending, as IEEE 754 doubles, little-endian, 8 bytes each.
//! - `calibration.json`: format number, kind, revision, tool version, the scorer's kind
//!   and revision, the settings, the number of good and bad calibration logs, and the
//!   lists in file order with label, group (`null` for the pooled list) and length.
//!
//! The lists are in a fixed order: the good lists, then the bad lists; within a label
//! the pooled list first, then the groups in byte order of their names.
//!
//! The directory is written under a temporary name beside its final place and renamed
//! when complete. Saving a calibration whose directory already holds the same revision
//! writes nothing.
//!
//! ## Revision
//!
//! The revision is the first 40 hexadecimal characters of the SHA-256 of these bytes
//! (all integers and bit patterns little-endian):
//!
//! ```text
//! "l2d-conformal-calibration/1\n"   28 bytes
//! scorer kind                       u8: 0 baseline, 1 neighbour
//! scorer revision                   u64 length, then the bytes
//! min_per_group                     u32
//! number of lists                   u64
//! per list, in file order:
//!   label                           u8: 0 good, 1 bad
//!   stratum                         u8: 0 pooled, 1 group
//!   group name                      u64 length, then the bytes (length 0 when pooled)
//!   length of the list              u64
//!   values                          f64 bit patterns (the list's bytes in scores.bin)
//! ```
//!
//! [`ConformalCalibrator::load`] checks the shape of the content, computes the revision
//! again and refuses a directory whose stored revision or name does not match. Equal
//! calibration logs give the same revision whatever their order.
//!
//! # Untrusted data
//!
//! Group names are data: they are compared and stored, never used as paths. The
//! calibration name is the only caller-supplied part of a path and is restricted to
//! letters, digits, `.`, `_` and `-`.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use l2d_baseline::BaselineScorer;
use l2d_baseline::scorer::{SCORES_ABSTAINED, SCORES_RATIO};
use l2d_core::input::content_sha256;
use l2d_core::polars::prelude::{
    Column, DataFrame, DataType, Field, IntoLazy, PolarsError, Schema, col, len,
};
use l2d_core::types::short_revision;
use l2d_core::{
    InputSummary, L2dError, Label, LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};
use serde::{Deserialize, Serialize};

use crate::ScorerKind;
use crate::scorer::{Scorer, check_scores};

/// The error rate used when the analyst names none (decision of 2026-11-11).
pub const DEFAULT_ERROR_RATE: f32 = 0.05;
/// How many calibration logs of a label a group needs for a list of its own.
pub const DEFAULT_MIN_PER_GROUP: u32 = 100;

/// Requirement 11.9: what every output that shows calibrated decisions states.
pub const EXCHANGEABILITY_NOTE: &str = "The error rate holds only for logs that resemble the \
     calibration logs (the same kind of logs, from the same sources, as the logs the \
     calibration was made on); `l2d shift` tests whether two sets of logs differ.";

/// Column of the calibration points frame: the scorer's ratio, Float32, never null.
pub const POINTS_RATIO: &str = "ratio";
/// Column of the calibration points frame: `good`, `bad` or `unknown`, never null.
pub const POINTS_LABEL: &str = "label";
/// Column of the calibration points frame: the group key, String, null without one.
pub const POINTS_GROUP: &str = "group";

/// The file of a calibration directory that describes it.
pub const MANIFEST_FILE: &str = "calibration.json";
/// The file of a calibration directory that holds the nonconformity lists.
pub const SCORES_FILE: &str = "scores.bin";
/// Longest calibration name, in bytes.
pub const MAX_NAME_BYTES: usize = 64;

const KIND: &str = "conformal-calibration";
const FORMAT: u32 = 1;
const REVISION_CHARS: usize = 40;
const REVISION_DOMAIN: &[u8] = b"l2d-conformal-calibration/1\n";
const OPERATION: &str = "calibrating";
const COUNT: &str = "logs";

/// Memory per calibration point, see [`estimated_bytes`].
const ESTIMATE_BYTES_PER_POINT: u64 = 200;
/// Memory per byte of group key, see [`estimated_bytes`].
const ESTIMATE_PER_GROUP_BYTE: u64 = 2;
/// Memory of a calibration whatever its size, see [`estimated_bytes`].
const ESTIMATE_FIXED_BYTES: u64 = 1024 * 1024;
/// Memory per row of the scores frame of [`ConformalCalibrator::fit_baseline`].
const SCORING_BYTES_PER_ROW: u64 = 48;
/// What one group costs in the index of the groups beside its name: the entry (the
/// name's 24 bytes and the 8 of the list's number) in a tree whose nodes are at least
/// half full, with the nodes above them. See [`ConformalCalibrator::held_bytes`].
pub const GROUP_INDEX_ENTRY_BYTES: usize = 96;

/// The memory a calibration of `points` rows, whose group keys take `group_bytes` bytes
/// in all, is estimated to need at its peak, in bytes:
/// `1 MiB + 200 * points + 2 * group_bytes`.
///
/// Per point that is 16 bytes kept (the value in the pooled list and in at most one
/// group list) and room for counting the logs per label and group in Polars, which also
/// copies each distinct group key and takes most of the peak. The figure was set in
/// task 1.13, when every value was kept twice more (32 bytes kept and 24 while a list
/// was built); task 1.18 left it as it was, so it now leaves more room.
/// `tests/conformal_csic.rs` asserts that the heap a calibration takes stays within it.
pub fn estimated_bytes(points: u64, group_bytes: u64) -> u64 {
    ESTIMATE_FIXED_BYTES
        .saturating_add(points.saturating_mul(ESTIMATE_BYTES_PER_POINT))
        .saturating_add(group_bytes.saturating_mul(ESTIMATE_PER_GROUP_BYTE))
}

/// The bytes the non-null strings of `column` take.
fn string_bytes(column: &Column) -> Result<u64, PolarsError> {
    Ok(column
        .str()?
        .iter()
        .flatten()
        .map(|text| text.len() as u64)
        .sum())
}

/// The labels a log cannot be ruled out of at the chosen error rate (requirement 11.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelSet {
    /// Only good is left: the log can be scored down at the stated error rate.
    Good,
    /// Only bad is left: the log belongs in the review ranking.
    Bad,
    /// Neither label can be ruled out: the log is uncertain.
    Both,
    /// Both labels are ruled out: the log is unlike the calibration logs.
    Neither,
}

impl LabelSet {
    /// The lower-case name used in frames and in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            LabelSet::Good => "good",
            LabelSet::Bad => "bad",
            LabelSet::Both => "both",
            LabelSet::Neither => "neither",
        }
    }

    /// Whether `label` is in the set. [`Label::Unknown`] is in no set.
    pub fn contains(self, label: Label) -> bool {
        match label {
            Label::Good => matches!(self, LabelSet::Good | LabelSet::Both),
            Label::Bad => matches!(self, LabelSet::Bad | LabelSet::Both),
            Label::Unknown => false,
        }
    }

    fn of(good: bool, bad: bool) -> Self {
        match (good, bad) {
            (true, false) => LabelSet::Good,
            (false, true) => LabelSet::Bad,
            (true, true) => LabelSet::Both,
            (false, false) => LabelSet::Neither,
        }
    }
}

/// Which list of calibration logs a p-value was computed on (requirement 11.4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stratum {
    /// All calibration logs of the label.
    Pooled,
    /// The calibration logs of the label in this group.
    Group(String),
}

impl fmt::Display for Stratum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stratum::Pooled => f.write_str("pooled"),
            Stratum::Group(group) => write!(f, "group {group}"),
        }
    }
}

/// A calibrated decision on one log.
///
/// `set` holds every label whose p-value is greater than `error_rate`, compared on the
/// values given here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibrated {
    pub set: LabelSet,
    /// P-value of the log for the label good.
    pub p_good: f32,
    /// P-value of the log for the label bad.
    pub p_bad: f32,
    /// The list `p_good` was computed on.
    pub good_stratum: Stratum,
    /// The list `p_bad` was computed on.
    pub bad_stratum: Stratum,
    /// The error rate the set was made for.
    pub error_rate: f32,
}

/// The settings of a calibration; part of its revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationSettings {
    /// How many calibration logs of a label a group needs for a list of its own; at
    /// least 1.
    pub min_per_group: u32,
}

impl Default for CalibrationSettings {
    fn default() -> Self {
        Self {
            min_per_group: DEFAULT_MIN_PER_GROUP,
        }
    }
}

/// One calibration log: the row view of the calibration points frame.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationPoint {
    /// The scorer's ratio for the log, `0.0..=1.0`.
    pub ratio: f32,
    pub label: Label,
    pub group: Option<String>,
}

/// One list of a calibration and how many calibration logs it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StratumCount {
    pub label: Label,
    pub stratum: Stratum,
    pub logs: u64,
}

/// Why a calibration or a calibrated decision was not made.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CalibrationError {
    /// Requirement 11.5. `good` and `bad` are the lengths of the shortest list of each
    /// label that the calibration uses (the pooled list, or a group's).
    #[error(
        "the calibration logs are too few for this error rate: the shortest lists hold \
         {good} good and {bad} bad calibration logs, which support error rates from \
         {smallest_supported} on"
    )]
    TooFewLogs {
        good: u64,
        bad: u64,
        smallest_supported: f32,
    },
    /// Requirement 11.1: the calibration logs are an input the scorer was fitted on.
    #[error(
        "the calibration logs are an input the scorer was fitted on (content SHA-256 \
         {content_sha256}); calibration needs logs the scorer has not seen"
    )]
    SameInputAsFitting { content_sha256: String },
    /// Requirement 11.1: the scorer does not name the inputs it was fitted on.
    #[error(
        "the scorer does not say which inputs it was fitted on, so it cannot be shown \
         that the calibration logs are not among them"
    )]
    UnknownFittingInputs,
    /// Requirement 11.1: the calibration logs come without the summary of their input.
    #[error(
        "the calibration logs come without the summary of the input they were read from, \
         so it cannot be shown that the scorer was not fitted on them"
    )]
    UnknownCalibrationInputs,
    /// The error rate is not a number above 0 and below 1.
    #[error("the error rate must be above 0 and below 1; got {error_rate}")]
    InvalidErrorRate { error_rate: f32 },
    /// The ratio to decide on is not a number in `0.0..=1.0`.
    #[error("a ratio lies between 0 and 1; got {ratio}")]
    InvalidRatio { ratio: f32 },
    /// The calibration points are not a calibration points frame.
    #[error("not a calibration points frame: {detail}")]
    InvalidPoints { detail: String },
    /// The settings cannot be used.
    #[error("unusable calibration settings: {detail}")]
    InvalidSettings { detail: String },
    /// The calibration was made for another scorer than the one whose ratios it is
    /// asked to judge.
    #[error(
        "the calibration was made for the {calibrated_kind} scorer of revision \
         {calibrated_revision}, not for the {kind} scorer of revision {revision}"
    )]
    OtherScorer {
        calibrated_kind: &'static str,
        calibrated_revision: String,
        kind: &'static str,
        revision: String,
    },
    /// The memory limit does not leave room for the calibration (requirement 12.5).
    #[error("{0}")]
    Memory(L2dError),
    /// The scorer failed on the calibration logs.
    #[error("{0}")]
    Scorer(L2dError),
    /// Polars failed, or the lists built from the points do not have the shape of a
    /// calibration.
    #[error("the conformal calibration failed: {detail}")]
    Internal { detail: String },
}

impl From<CalibrationError> for L2dError {
    fn from(error: CalibrationError) -> Self {
        match error {
            CalibrationError::Memory(inner) | CalibrationError::Scorer(inner) => inner,
            CalibrationError::TooFewLogs { .. }
            | CalibrationError::SameInputAsFitting { .. }
            | CalibrationError::UnknownFittingInputs
            | CalibrationError::UnknownCalibrationInputs
            | CalibrationError::OtherScorer { .. } => L2dError::Refused {
                reason: error.to_string(),
            },
            CalibrationError::InvalidErrorRate { .. }
            | CalibrationError::InvalidRatio { .. }
            | CalibrationError::InvalidPoints { .. }
            | CalibrationError::InvalidSettings { .. } => L2dError::InvalidInput {
                detail: error.to_string(),
            },
            CalibrationError::Internal { .. } => L2dError::Runtime {
                detail: error.to_string(),
            },
        }
    }
}

/// The column names and types of the calibration points frame (design, "Data frames").
pub fn points_schema() -> Schema {
    Schema::from_iter([
        Field::new(POINTS_RATIO.into(), DataType::Float32),
        Field::new(POINTS_LABEL.into(), DataType::String),
        Field::new(POINTS_GROUP.into(), DataType::String),
    ])
}

/// The calibration points frame of `points`, in their order.
pub fn points_frame(points: &[CalibrationPoint]) -> Result<DataFrame, L2dError> {
    let ratios: Vec<f32> = points.iter().map(|point| point.ratio).collect();
    let labels: Vec<&str> = points.iter().map(|point| point.label.as_str()).collect();
    let groups: Vec<Option<&str>> = points.iter().map(|point| point.group.as_deref()).collect();
    DataFrame::new(vec![
        Column::new(POINTS_RATIO.into(), ratios),
        Column::new(POINTS_LABEL.into(), labels),
        Column::new(POINTS_GROUP.into(), groups),
    ])
    .map_err(|error| L2dError::Runtime {
        detail: format!("building the calibration points frame failed: {error}"),
    })
}

/// The calibration points frame ([`points_schema`]) of `logs` and the scores frame a
/// scorer gave for them, row for row: the ratio from `scores`, label and group from
/// `logs`, in the order of the logs.
///
/// A log the scorer abstained on has no ratio and is not a calibration log: its row is
/// left out. The flag `abstained` decides, whatever the ratio column holds on that row.
/// Rows labelled `unknown` stay; [`ConformalCalibrator::fit_with`] ignores them.
///
/// A scores frame with another number of rows than `logs` is
/// [`CalibrationError::InvalidPoints`].
pub fn calibration_points(
    scores: &DataFrame,
    logs: &LogsFrame,
) -> Result<DataFrame, CalibrationError> {
    if scores.height() != logs.len() {
        return Err(CalibrationError::InvalidPoints {
            detail: format!(
                "the scores frame has {} rows for {} logs",
                scores.height(),
                logs.len()
            ),
        });
    }
    let polars = |error: PolarsError| internal("building the calibration points", error);
    let column = |frame: &DataFrame, from: &str, to: &str| {
        frame
            .column(from)
            .map(|column| column.clone().with_name(to.into()))
            .map_err(polars)
    };
    let points = DataFrame::new(vec![
        column(scores, SCORES_RATIO, POINTS_RATIO)?,
        column(logs.frame(), LogsFrame::LABEL, POINTS_LABEL)?,
        column(logs.frame(), LogsFrame::GROUP, POINTS_GROUP)?,
    ])
    .map_err(polars)?;
    let abstained = scores
        .column(SCORES_ABSTAINED)
        .and_then(Column::bool)
        .map_err(polars)?;
    if abstained.any() {
        points.filter(&!abstained).map_err(polars)
    } else {
        Ok(points)
    }
}

/// Refuses calibration logs the scorer was fitted on (requirement 11.1).
///
/// `inputs` are the summaries of the sources the calibration logs were read from. The
/// check compares content hashes of whole inputs: it refuses an input that is among the
/// scorer's fitting inputs, a scorer that names no fitting inputs, and calibration logs
/// without an input summary. It cannot tell single logs apart: when fitting logs and
/// calibration logs are parts of one input (the folds of an evaluation), the caller
/// keeps the parts disjoint and calls [`ConformalCalibrator::fit_with`] with the points.
pub fn require_held_out(
    scorer: &BaselineScorer,
    inputs: &[InputSummary],
) -> Result<(), CalibrationError> {
    if !scorer.has_known_fitting_inputs() {
        return Err(CalibrationError::UnknownFittingInputs);
    }
    if inputs.is_empty() {
        return Err(CalibrationError::UnknownCalibrationInputs);
    }
    match inputs
        .iter()
        .find(|input| scorer.was_fitted_on(&input.content_sha256))
    {
        Some(input) => Err(CalibrationError::SameInputAsFitting {
            content_sha256: input.content_sha256.clone(),
        }),
        None => Ok(()),
    }
}

/// Memory per calibration log that [`held_out_logs`] asks the budget for: the yes or no
/// per log and the rows of the logs frame that are kept (three text columns at 16
/// bytes a row, the position, the occurrences, and what Polars rounds up). The texts
/// themselves are shared with the frame given, not copied.
pub const HELD_OUT_ROW_BYTES: u64 = 96;

/// What a held-out report says when the scorer does not know the texts it was fitted
/// on (a scorer saved before task 1.11).
pub const WHOLE_FILE_NOTE: &str = "The scorer was saved without the hashes of the texts it \
     was fitted on, so only a calibration input that is identical, byte for byte, to an \
     input the scorer was fitted on is detected; the same logs in another file (reordered, \
     re-exported, a subset) are not. Fit the scorer again to get the check per log.";

/// What a held-out report says when the scorer's hashes cover only part of what it was
/// fitted on: a scorer saved before task 1.11 that was saved again after a new fit of
/// the same model on another input.
pub const PARTIAL_NOTE: &str = "The scorer's hashes of fitted texts cover only part of what \
     it was fitted on: they were added to a scorer saved without them, whose earlier fitting \
     inputs are known by their whole-file hash alone. Logs of those earlier inputs are \
     detected only when the calibration input is identical, byte for byte, to one of them. \
     Save the scorer under a new name after fitting it again to get the complete check.";

/// How the calibration logs were held to "not used to fit the scorer" (requirement
/// 11.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeldOutCheck {
    /// The whole-input check, and every log whose text the scorer was fitted on left
    /// out.
    PerLog,
    /// As [`HeldOutCheck::PerLog`], but the scorer's hashes cover only part of its
    /// fitting inputs ([`BaselineScorer::fitted_text_hashes_complete`] is false): logs
    /// of the other inputs are caught by the whole-input check alone.
    PerLogPartial,
    /// The whole-input check alone: the scorer does not know its fitted texts.
    WholeFile,
}

/// What [`held_out_logs`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HeldOut {
    pub check: HeldOutCheck,
    /// The calibration logs given.
    pub logs: u64,
    /// Those left out because the scorer was fitted on a log with the same text;
    /// always 0 with [`HeldOutCheck::WholeFile`].
    pub dropped: u64,
    /// Those that remain: `logs - dropped`.
    pub kept: u64,
}

impl HeldOut {
    /// What has to be said beside the result: [`WHOLE_FILE_NOTE`] when only an
    /// identical file is detected, [`PARTIAL_NOTE`] when the check per log covers only
    /// part of what the scorer was fitted on, nothing otherwise.
    pub fn note(&self) -> Option<&'static str> {
        match self.check {
            HeldOutCheck::PerLog => None,
            HeldOutCheck::PerLogPartial => Some(PARTIAL_NOTE),
            HeldOutCheck::WholeFile => Some(WHOLE_FILE_NOTE),
        }
    }
}

/// The calibration logs the baseline scorer was not fitted on (requirement 11.1; owner
/// decision of 2026-11-12, task 1.11).
///
/// First the whole-input check of [`require_held_out`], unchanged: an input that is a
/// fitting input, a scorer that names no fitting inputs, and logs without an input
/// summary are refused. Then, when the scorer knows the texts it was fitted on
/// ([`BaselineScorer::has_fitted_text_hashes`]), every log whose text is among them is
/// left out, whatever its label here; the rest is returned in its order with the
/// counts. A scorer that does not know them (saved before task 1.11) gets the logs back
/// as given, and the report says that only an identical file is detected
/// ([`HeldOut::note`]).
///
/// Nothing is refused for the number of logs left: the calibration refuses when no
/// good or no bad log remains, and [`ConformalCalibrator::require_error_rate`] when
/// they are too few for an error rate.
///
/// The budget is asked for [`HELD_OUT_ROW_BYTES`] per log before a row is copied; a
/// refusal is [`CalibrationError::Memory`]. The returned frame shares the texts of
/// `logs`; when nothing is left out it is `logs` itself.
pub fn held_out_logs(
    scorer: &BaselineScorer,
    logs: &LogsFrame,
    inputs: &[InputSummary],
    budget: &MemoryBudget,
) -> Result<(LogsFrame, HeldOut), CalibrationError> {
    require_held_out(scorer, inputs)?;
    let rows = logs.len() as u64;
    let unchanged = |check| {
        let report = HeldOut {
            check,
            logs: rows,
            dropped: 0,
            kept: rows,
        };
        (logs.clone(), report)
    };
    let per_log = match scorer.fitted_text_hashes_complete() {
        None => return Ok(unchanged(HeldOutCheck::WholeFile)),
        Some(true) => HeldOutCheck::PerLog,
        Some(false) => HeldOutCheck::PerLogPartial,
    };
    budget
        .require_whole(
            rows.saturating_mul(HELD_OUT_ROW_BYTES),
            "checking the calibration logs against the texts the scorer was fitted on",
        )
        .map_err(CalibrationError::Memory)?;
    let polars =
        |error: PolarsError| internal("leaving out the logs the scorer was fitted on", error);
    let texts = logs
        .frame()
        .column(LogsFrame::TEXT)
        .and_then(Column::str)
        .map_err(polars)?;
    let keep: Vec<bool> = texts
        .iter()
        .map(|text| !text.is_some_and(|text| scorer.was_fitted_on_text(text) == Some(true)))
        .collect();
    let kept = keep.iter().filter(|keep| **keep).count() as u64;
    if kept == rows {
        return Ok(unchanged(per_log));
    }
    let mask = Column::new("keep".into(), keep);
    let frame = mask
        .bool()
        .and_then(|mask| logs.frame().filter(mask))
        .map_err(polars)?;
    let frame =
        LogsFrame::new(frame).map_err(|error| internal("keeping the held-out logs", error))?;
    let report = HeldOut {
        check: per_log,
        logs: rows,
        dropped: rows - kept,
        kept,
    };
    Ok((frame, report))
}

/// The nonconformity of a log with this ratio for the label good (`bad == false`) or
/// bad. Adding zero turns a negative zero into zero, so equal ratios give equal bits.
fn nonconformity(ratio: f32, bad: bool) -> f64 {
    let ratio = f64::from(ratio) + 0.0;
    if bad { 1.0 - ratio } else { ratio }
}

fn usable_ratio(ratio: f32) -> bool {
    (0.0..=1.0).contains(&ratio)
}

/// The smallest error rate a list of `logs` calibration logs supports, as the 32-bit
/// value error rates are compared in.
fn smallest_rate(logs: u64) -> f32 {
    (1.0 / (logs as f64 + 1.0)) as f32
}

/// How many of `scores` are at least `candidate`; a score equal to the candidate
/// counts. `scores` is sorted ascending and holds no NaN, and `candidate` is not NaN
/// ([`ConformalCalibrator::assemble`] and [`usable_ratio`] see to both), so the values
/// below the candidate are a prefix of the list and its length is found by binary
/// search. The two zeros compare equal, as they do in a scan.
fn at_least(scores: &[f64], candidate: f64) -> usize {
    scores.len() - scores.partition_point(|score| *score < candidate)
}

/// The p-value of `candidate` on the sorted list `scores`:
/// `(count of scores >= candidate + 1) / (n + 1)`, in double precision.
///
/// The count and `n` are converted to `f64` before one is added to each, and the two
/// sums are divided: the operations, in the order, of the crate `conformal-prediction`
/// 2.0.1 that task 1.13 used, so the result has the same bits.
fn p_value_of(scores: &[f64], candidate: f64) -> f64 {
    (at_least(scores, candidate) as f64 + 1.0) / (scores.len() as f64 + 1.0)
}

fn internal(doing: &str, error: impl fmt::Display) -> CalibrationError {
    CalibrationError::Internal {
        detail: format!("{doing}: {error}"),
    }
}

/// One list of nonconformity values.
struct List {
    bad: bool,
    stratum: Stratum,
    /// Sorted ascending and not empty; what is saved, hashed and searched.
    scores: Vec<f64>,
}

impl List {
    fn label(&self) -> Label {
        if self.bad { Label::Bad } else { Label::Good }
    }

    fn logs(&self) -> u64 {
        self.scores.len() as u64
    }

    /// The p-value of a log with this ratio for the list's label, as the 32-bit value
    /// that is reported and compared with the error rate.
    fn p_value(&self, ratio: f32) -> f32 {
        p_value_of(&self.scores, nonconformity(ratio, self.bad)) as f32
    }
}

/// A scorer's ratio turned into label sets with a stated error rate. See the module
/// documentation.
pub struct ConformalCalibrator {
    scorer: ScorerKind,
    scorer_revision: String,
    settings: CalibrationSettings,
    /// In file order: good pooled, good groups, bad pooled, bad groups.
    lists: Vec<List>,
    /// Index of the pooled list of good and of bad.
    pooled: [usize; 2],
    /// Index of each group's list, for good and for bad.
    groups: [BTreeMap<String, usize>; 2],
    /// Length of the shortest list of good and of bad.
    shortest: [u64; 2],
    smallest_supported: f32,
    revision: String,
}

impl fmt::Debug for ConformalCalibrator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConformalCalibrator")
            .field("scorer", &self.scorer)
            .field("scorer_revision", &self.scorer_revision)
            .field("settings", &self.settings)
            .field("revision", &self.revision)
            .field("strata", &self.strata())
            .finish()
    }
}

/// Two calibrators are equal when they were made for the same scorer with the same
/// settings and hold the same lists.
impl PartialEq for ConformalCalibrator {
    fn eq(&self, other: &Self) -> bool {
        self.scorer == other.scorer
            && self.scorer_revision == other.scorer_revision
            && self.settings == other.settings
            && self.revision == other.revision
            && self.lists.len() == other.lists.len()
            && self.lists.iter().zip(&other.lists).all(|(own, theirs)| {
                own.bad == theirs.bad
                    && own.stratum == theirs.stratum
                    && own.scores.len() == theirs.scores.len()
                    && own
                        .scores
                        .iter()
                        .zip(&theirs.scores)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            })
    }
}

fn scorer_code(kind: ScorerKind) -> u8 {
    match kind {
        ScorerKind::Baseline => 0,
        ScorerKind::Neighbour => 1,
    }
}

fn score_bytes(scores: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(scores.len() * 8);
    for score in scores {
        bytes.extend_from_slice(&score.to_bits().to_le_bytes());
    }
    bytes
}

/// The revision of a calibration: see "Revision" in the module documentation.
fn revision_of(
    scorer: ScorerKind,
    scorer_revision: &str,
    settings: &CalibrationSettings,
    lists: &[List],
) -> String {
    let values: usize = lists.iter().map(|list| list.scores.len()).sum();
    let mut bytes = Vec::with_capacity(REVISION_DOMAIN.len() + 64 + values * 8 + lists.len() * 64);
    bytes.extend_from_slice(REVISION_DOMAIN);
    bytes.push(scorer_code(scorer));
    bytes.extend_from_slice(&(scorer_revision.len() as u64).to_le_bytes());
    bytes.extend_from_slice(scorer_revision.as_bytes());
    bytes.extend_from_slice(&settings.min_per_group.to_le_bytes());
    bytes.extend_from_slice(&(lists.len() as u64).to_le_bytes());
    for list in lists {
        bytes.push(u8::from(list.bad));
        match &list.stratum {
            Stratum::Pooled => {
                bytes.push(0);
                bytes.extend_from_slice(&0_u64.to_le_bytes());
            }
            Stratum::Group(group) => {
                bytes.push(1);
                bytes.extend_from_slice(&(group.len() as u64).to_le_bytes());
                bytes.extend_from_slice(group.as_bytes());
            }
        }
        bytes.extend_from_slice(&list.logs().to_le_bytes());
        for score in &list.scores {
            bytes.extend_from_slice(&score.to_bits().to_le_bytes());
        }
    }
    let mut revision = content_sha256(&bytes);
    revision.truncate(REVISION_CHARS);
    revision
}

#[derive(Debug, Serialize, Deserialize)]
struct ScorerFile {
    kind: String,
    revision: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ListFile {
    label: String,
    /// `null` for the pooled list.
    group: Option<String>,
    logs: u64,
}

/// The content of `calibration.json`.
#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    format: u32,
    kind: String,
    revision: String,
    tool_version: String,
    scorer: ScorerFile,
    settings: CalibrationSettings,
    good: u64,
    bad: u64,
    lists: Vec<ListFile>,
}

/// Checks a calibration name: 1 to [`MAX_NAME_BYTES`] characters out of letters, digits,
/// `.`, `_` and `-`, not starting with a dot (the rule of the baseline scorer).
fn validate_name(name: &str) -> Result<(), L2dError> {
    let usable = !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if usable {
        Ok(())
    } else {
        Err(L2dError::InvalidInput {
            detail: format!(
                "the calibration name must be 1 to {MAX_NAME_BYTES} letters, digits, '.', '_' \
                 or '-' and must not start with a dot; got a name of {} bytes that is not",
                name.len()
            ),
        })
    }
}

fn io_failure(doing: &str, path: &Path, error: &std::io::Error) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

fn not_a_calibration(dir: &Path, why: impl fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "{} does not hold a usable calibration: {why}",
            dir.display()
        ),
    }
}

/// Reads a file of a calibration directory; a file that is not there is
/// [`L2dError::Missing`].
fn read_calibration_file(dir: &Path, file: &str) -> Result<Vec<u8>, L2dError> {
    let path = dir.join(file);
    fs::read(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            L2dError::Missing {
                item: format!("calibration file {file}"),
                expected_at: path.clone(),
            }
        } else {
            io_failure("reading", &path, &error)
        }
    })
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), L2dError> {
    let failed = |error: std::io::Error| io_failure("writing", path, &error);
    let mut file = fs::File::create(path).map_err(failed)?;
    file.write_all(bytes).map_err(failed)?;
    file.sync_all().map_err(failed)
}

/// Makes temporary directory names of one process distinct.
static SAVES: AtomicU64 = AtomicU64::new(0);

/// Checks that `points` has the columns of [`points_schema`] and nulls only in `group`.
fn check_points(points: &DataFrame) -> Result<(), CalibrationError> {
    let expected = points_schema();
    let found = points.schema();
    if **found != expected {
        let describe = |schema: &Schema| {
            schema
                .iter()
                .map(|(name, dtype)| format!("{name} {dtype}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(CalibrationError::InvalidPoints {
            detail: format!(
                "expected the columns [{}], found [{}]",
                describe(&expected),
                describe(found)
            ),
        });
    }
    for name in [POINTS_RATIO, POINTS_LABEL] {
        let nulls = points
            .column(name)
            .map_err(|error| internal("reading the calibration points", error))?
            .null_count();
        if nulls > 0 {
            return Err(CalibrationError::InvalidPoints {
                detail: format!("column '{name}' holds {nulls} null values"),
            });
        }
    }
    Ok(())
}

impl ConformalCalibrator {
    /// Calibrates under a memory budget observed now with the default settings. See
    /// [`ConformalCalibrator::fit_with`].
    pub fn fit(
        points: &DataFrame,
        scorer: ScorerKind,
        scorer_revision: &str,
        settings: &CalibrationSettings,
    ) -> Result<Self, CalibrationError> {
        let budget =
            MemoryBudget::observe(&MemorySettings::default()).map_err(CalibrationError::Memory)?;
        Self::fit_with(points, scorer, scorer_revision, settings, &budget)
    }

    /// Calibrates the scorer of the given kind and revision on `points`, the
    /// calibration points frame ([`points_schema`]): one row per calibration log with
    /// the scorer's ratio, the label and the group key. Rows labelled `unknown` are not
    /// calibration logs and are ignored; the order of the rows does not matter.
    ///
    /// The caller has made sure that the scorer was not fitted on these logs
    /// ([`require_held_out`]; [`ConformalCalibrator::fit_baseline`] does both).
    ///
    /// - A frame with other columns, a null ratio or label, a label that is not one of
    ///   the three names, or a ratio outside `0.0..=1.0` is
    ///   [`CalibrationError::InvalidPoints`].
    /// - No good or no bad calibration log is [`CalibrationError::TooFewLogs`] with
    ///   the counts found and 1 as the smallest supported rate.
    /// - A calibration whose [`estimated_bytes`] exceed what `budget` leaves is
    ///   [`CalibrationError::Memory`], before anything is allocated.
    pub fn fit_with(
        points: &DataFrame,
        scorer: ScorerKind,
        scorer_revision: &str,
        settings: &CalibrationSettings,
        budget: &MemoryBudget,
    ) -> Result<Self, CalibrationError> {
        if settings.min_per_group == 0 {
            return Err(CalibrationError::InvalidSettings {
                detail: "a group needs at least 1 calibration log of a label for a list of \
                         its own; min_per_group is 0"
                    .into(),
            });
        }
        check_points(points)?;
        let polars = |error: PolarsError| internal("reading the calibration points", error);
        let group_bytes = points
            .column(POINTS_GROUP)
            .and_then(string_bytes)
            .map_err(polars)?;
        budget
            .require_whole(
                estimated_bytes(points.height() as u64, group_bytes),
                OPERATION,
            )
            .map_err(CalibrationError::Memory)?;

        // How many calibration logs each label has in each group, counted in Polars.
        let counts = points
            .clone()
            .lazy()
            .group_by_stable([col(POINTS_LABEL), col(POINTS_GROUP)])
            .agg([len().cast(DataType::UInt64).alias(COUNT)])
            .collect()
            .map_err(polars)?;
        let count_labels = counts
            .column(POINTS_LABEL)
            .and_then(Column::str)
            .map_err(polars)?;
        let count_groups = counts
            .column(POINTS_GROUP)
            .and_then(Column::str)
            .map_err(polars)?;
        let count_logs = counts.column(COUNT).and_then(Column::u64).map_err(polars)?;
        let mut totals = [0_u64; 2];
        let mut own: [BTreeMap<String, Vec<f64>>; 2] = [BTreeMap::new(), BTreeMap::new()];
        for ((label, group), logs) in count_labels
            .iter()
            .zip(count_groups.iter())
            .zip(count_logs.iter())
        {
            let (Some(label), Some(logs)) = (label, logs) else {
                return Err(internal(
                    "counting the calibration points",
                    "a count without a label",
                ));
            };
            let bad = match Label::from_name(label) {
                Some(Label::Good) => false,
                Some(Label::Bad) => true,
                Some(Label::Unknown) => continue,
                None => {
                    return Err(CalibrationError::InvalidPoints {
                        detail: format!("label '{label}' is not good, bad or unknown"),
                    });
                }
            };
            totals[usize::from(bad)] += logs;
            if let Some(group) = group
                && logs >= u64::from(settings.min_per_group)
            {
                own[usize::from(bad)].insert(group.to_owned(), Vec::with_capacity(logs as usize));
            }
        }
        drop(counts);

        // The values, list by list, in row order.
        let ratios = points
            .column(POINTS_RATIO)
            .and_then(Column::f32)
            .map_err(polars)?;
        let labels = points
            .column(POINTS_LABEL)
            .and_then(Column::str)
            .map_err(polars)?;
        let groups = points
            .column(POINTS_GROUP)
            .and_then(Column::str)
            .map_err(polars)?;
        let mut pooled = [
            Vec::with_capacity(totals[0] as usize),
            Vec::with_capacity(totals[1] as usize),
        ];
        for (row, ((ratio, label), group)) in ratios
            .iter()
            .zip(labels.iter())
            .zip(groups.iter())
            .enumerate()
        {
            let ratio = ratio.filter(|ratio| usable_ratio(*ratio)).ok_or_else(|| {
                CalibrationError::InvalidPoints {
                    detail: format!(
                        "row {row}: a ratio lies between 0 and 1; got {}",
                        ratio.map_or("null".to_owned(), |ratio| ratio.to_string())
                    ),
                }
            })?;
            let bad = match label.and_then(Label::from_name) {
                Some(Label::Good) => false,
                Some(Label::Bad) => true,
                _ => continue,
            };
            let value = nonconformity(ratio, bad);
            pooled[usize::from(bad)].push(value);
            if let Some(list) = group.and_then(|group| own[usize::from(bad)].get_mut(group)) {
                list.push(value);
            }
        }
        if totals[0] == 0 || totals[1] == 0 {
            return Err(CalibrationError::TooFewLogs {
                good: totals[0],
                bad: totals[1],
                smallest_supported: 1.0,
            });
        }

        let mut parts = Vec::with_capacity(2 + own[0].len() + own[1].len());
        for (bad, (pooled, own)) in [false, true].into_iter().zip(pooled.into_iter().zip(own)) {
            parts.push((bad, Stratum::Pooled, pooled));
            for (group, scores) in own {
                parts.push((bad, Stratum::Group(group), scores));
            }
        }
        for (_, _, scores) in &mut parts {
            scores.sort_unstable_by(f64::total_cmp);
        }
        Self::assemble(scorer, scorer_revision.to_owned(), *settings, parts)
            .map_err(|detail| internal("building the lists", detail))
    }

    /// Calibrates the baseline scorer on `logs`: checks that the scorer was not fitted
    /// on them ([`held_out_logs`] with `inputs`, the summaries of the sources `logs`
    /// was read from: the whole-input check, and since task 1.11 every log whose text
    /// the scorer was fitted on is left out), scores the rest and calibrates on those
    /// labelled good or bad. See [`ConformalCalibrator::fit_with`]. A caller that
    /// reports how many logs were left out calls [`held_out_logs`] itself and then
    /// [`ConformalCalibrator::fit_scorer`], as `l2d calibrate` does.
    pub fn fit_baseline(
        scorer: &BaselineScorer,
        logs: &LogsFrame,
        inputs: &[InputSummary],
        settings: &CalibrationSettings,
        budget: &MemoryBudget,
    ) -> Result<Self, CalibrationError> {
        let (logs, _) = held_out_logs(scorer, logs, inputs, budget)?;
        let scoring_bytes = (logs.len() as u64).saturating_mul(SCORING_BYTES_PER_ROW);
        Self::fit_scored(
            ScorerKind::Baseline,
            scorer.revision(),
            |logs| scorer.score(logs),
            scoring_bytes,
            &logs,
            settings,
            budget,
        )
    }

    /// Calibrates any scorer on `logs` through the [`Scorer`] contract: scores the
    /// rows, leaves out those the scorer abstained on ([`calibration_points`]) and
    /// calibrates on the rest that are labelled good or bad. See
    /// [`ConformalCalibrator::fit_with`].
    ///
    /// This is what [`ConformalCalibrator::fit_baseline`] does after its check of the
    /// inputs. The caller has made sure that the scorer was not fitted on these logs:
    /// [`require_held_out`] knows the baseline scorer only. The memory asked for is
    /// that of the calibration plus what the scorer says scoring all of `logs` at once
    /// costs ([`Scorer::cost`]).
    pub fn fit_scorer(
        scorer: &mut dyn Scorer,
        logs: &LogsFrame,
        settings: &CalibrationSettings,
        budget: &MemoryBudget,
    ) -> Result<Self, CalibrationError> {
        let kind = scorer.kind();
        let revision = scorer.revision().to_owned();
        let cost = scorer.cost(logs);
        let scoring_bytes = cost.fixed_bytes.saturating_add(
            (logs.len() as u64)
                .saturating_mul(cost.row_bytes.saturating_add(cost.evidence_row_bytes)),
        );
        Self::fit_scored(
            kind,
            &revision,
            |logs| scorer.score(logs),
            scoring_bytes,
            logs,
            settings,
            budget,
        )
    }

    /// The common path of [`ConformalCalibrator::fit_baseline`] and
    /// [`ConformalCalibrator::fit_scorer`]: the memory check, the scores of `score`,
    /// the calibration points without the abstentions, the calibration.
    fn fit_scored(
        kind: ScorerKind,
        revision: &str,
        score: impl FnOnce(&LogsFrame) -> Result<DataFrame, L2dError>,
        scoring_bytes: u64,
        logs: &LogsFrame,
        settings: &CalibrationSettings,
        budget: &MemoryBudget,
    ) -> Result<Self, CalibrationError> {
        let rows = logs.len() as u64;
        let group_bytes = logs
            .frame()
            .column(LogsFrame::GROUP)
            .and_then(string_bytes)
            .map_err(|error| internal("building the calibration points", error))?;
        budget
            .require_whole(
                estimated_bytes(rows, group_bytes).saturating_add(scoring_bytes),
                OPERATION,
            )
            .map_err(CalibrationError::Memory)?;

        let scores = score(logs).map_err(CalibrationError::Scorer)?;
        check_scores(&scores, logs, kind).map_err(|detail| {
            CalibrationError::Scorer(L2dError::Runtime {
                detail: format!(
                    "the {} scorer returned an unusable scores frame: {detail}",
                    kind.as_str()
                ),
            })
        })?;
        let points = calibration_points(&scores, logs)?;
        drop(scores);
        Self::fit_with(&points, kind, revision, settings, budget)
    }

    /// Builds a calibrator from its lists after checking their shape: the fixed order,
    /// one pooled list per label, no empty list, group lists of at least
    /// `min_per_group` and at most the pooled number of logs, and sorted values in
    /// `0.0..=1.0`.
    fn assemble(
        scorer: ScorerKind,
        scorer_revision: String,
        settings: CalibrationSettings,
        parts: Vec<(bool, Stratum, Vec<f64>)>,
    ) -> Result<Self, String> {
        if settings.min_per_group == 0 {
            return Err("min_per_group is 0".into());
        }
        let mut pooled = [None, None];
        let mut groups: [BTreeMap<String, usize>; 2] = [BTreeMap::new(), BTreeMap::new()];
        let mut shortest = [u64::MAX; 2];
        let mut previous: Option<(bool, &Stratum)> = None;
        for (index, (bad, stratum, scores)) in parts.iter().enumerate() {
            let label = usize::from(*bad);
            let name = format!(
                "list {index} ({} {stratum})",
                if *bad { "bad" } else { "good" }
            );
            let in_order = match (previous, stratum) {
                (None, Stratum::Pooled) => !*bad,
                (Some((before, _)), Stratum::Pooled) => !before && *bad,
                (Some((before, Stratum::Pooled)), Stratum::Group(_)) => before == *bad,
                (Some((before, Stratum::Group(earlier))), Stratum::Group(group)) => {
                    before == *bad && earlier.as_bytes() < group.as_bytes()
                }
                (None, Stratum::Group(_)) => false,
            };
            if !in_order {
                return Err(format!("{name} is out of order"));
            }
            previous = Some((*bad, stratum));
            if scores.is_empty() {
                return Err(format!("{name} is empty"));
            }
            if !scores.iter().all(|score| (0.0..=1.0).contains(score)) {
                return Err(format!("{name} holds a value outside 0 to 1"));
            }
            if !scores.windows(2).all(|pair| pair[0] <= pair[1]) {
                return Err(format!("{name} is not sorted"));
            }
            let logs = scores.len() as u64;
            match stratum {
                Stratum::Pooled => pooled[label] = Some(index),
                Stratum::Group(group) => {
                    let all = pooled[label].map_or(0, |pooled| parts[pooled].2.len() as u64);
                    if logs < u64::from(settings.min_per_group) || logs > all {
                        return Err(format!(
                            "{name} holds {logs} logs, outside the minimum per group of {} \
                             and the {all} logs of its label",
                            settings.min_per_group
                        ));
                    }
                    groups[label].insert(group.clone(), index);
                }
            }
            shortest[label] = shortest[label].min(logs);
        }
        let [Some(good), Some(bad)] = pooled else {
            return Err("the pooled list of a label is missing".into());
        };

        let lists = parts
            .into_iter()
            .map(|(bad, stratum, scores)| List {
                bad,
                stratum,
                scores,
            })
            .collect::<Vec<_>>();
        let revision = revision_of(scorer, &scorer_revision, &settings, &lists);
        Ok(Self {
            scorer,
            scorer_revision,
            settings,
            lists,
            pooled: [good, bad],
            groups,
            shortest,
            smallest_supported: smallest_rate(shortest[0].min(shortest[1])),
            revision,
        })
    }

    /// The kind of scorer the calibration was made for.
    pub fn scorer_kind(&self) -> ScorerKind {
        self.scorer
    }

    /// The revision of the scorer the calibration was made for.
    pub fn scorer_revision(&self) -> &str {
        &self.scorer_revision
    }

    /// The settings the calibration was made with.
    pub fn settings(&self) -> &CalibrationSettings {
        &self.settings
    }

    /// The revision: 40 hexadecimal characters, see the module documentation.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The number of good calibration logs.
    pub fn good(&self) -> u64 {
        self.lists[self.pooled[0]].logs()
    }

    /// The number of bad calibration logs.
    pub fn bad(&self) -> u64 {
        self.lists[self.pooled[1]].logs()
    }

    /// The lists in their fixed order with the number of calibration logs each holds.
    pub fn strata(&self) -> Vec<StratumCount> {
        self.lists
            .iter()
            .map(|list| StratumCount {
                label: list.label(),
                stratum: list.stratum.clone(),
                logs: list.logs(),
            })
            .collect()
    }

    /// The memory the calibration keeps for as long as it lives, fitted or loaded, in
    /// bytes (requirement 12.9): 8 bytes per value of every list (a log of a group with
    /// a list of its own is in two lists), and for each list its description and, for
    /// a group's list, the group's name twice (in the list and in the index of the
    /// groups) with [`GROUP_INDEX_ENTRY_BYTES`] for its index entry.
    ///
    /// An operation that uses a calibration holds this figure in its memory budget
    /// for as long as it runs; triage does ([`crate::triage`]).
    /// `tests/triage_memory.rs` measures it against the heap of a loaded calibration:
    /// the figure is never below what the heap shows and at most an eighth above (the
    /// most for a calibration of many small groups).
    pub fn held_bytes(&self) -> u64 {
        let lists: usize = self
            .lists
            .iter()
            .map(|list| {
                let name = match &list.stratum {
                    Stratum::Pooled => 0,
                    Stratum::Group(group) => group.len(),
                };
                list.scores.capacity() * size_of::<f64>() + name
            })
            .sum();
        let index: usize = self
            .groups
            .iter()
            .flat_map(BTreeMap::keys)
            .map(|group| group.len() + GROUP_INDEX_ENTRY_BYTES)
            .sum();
        (size_of::<Self>()
            + self.lists.capacity() * size_of::<List>()
            + lists
            + index
            + self.scorer_revision.len()
            + self.revision.len()) as u64
    }

    /// The smallest error rate the calibration supports: `1 / (n + 1)` of its shortest
    /// list.
    pub fn smallest_supported(&self) -> f32 {
        self.smallest_supported
    }

    /// Checks an error rate before it is used: [`CalibrationError::InvalidErrorRate`]
    /// unless it is above 0 and below 1, [`CalibrationError::TooFewLogs`] when it is
    /// below [`ConformalCalibrator::smallest_supported`] (requirement 11.5).
    pub fn require_error_rate(&self, error_rate: f32) -> Result<(), CalibrationError> {
        if !(error_rate > 0.0 && error_rate < 1.0) {
            return Err(CalibrationError::InvalidErrorRate { error_rate });
        }
        if error_rate < self.smallest_supported {
            return Err(CalibrationError::TooFewLogs {
                good: self.shortest[0],
                bad: self.shortest[1],
                smallest_supported: self.smallest_supported,
            });
        }
        Ok(())
    }

    /// Checks that ratios of the scorer of this kind and revision may be judged by this
    /// calibration: [`CalibrationError::OtherScorer`] when it was made for another.
    pub fn require_scorer(&self, kind: ScorerKind, revision: &str) -> Result<(), CalibrationError> {
        if kind == self.scorer && revision == self.scorer_revision {
            return Ok(());
        }
        Err(CalibrationError::OtherScorer {
            calibrated_kind: self.scorer.as_str(),
            calibrated_revision: self.scorer_revision.clone(),
            kind: kind.as_str(),
            revision: revision.to_owned(),
        })
    }

    /// The list of a label for a log of `group`: the group's where it has one.
    fn list(&self, bad: bool, group: Option<&str>) -> &List {
        let label = usize::from(bad);
        let index = group
            .and_then(|group| self.groups[label].get(group))
            .copied()
            .unwrap_or(self.pooled[label]);
        &self.lists[index]
    }

    /// The calibrated decision on a log with this ratio and group at `error_rate`: the
    /// p-value for each label, the list each was computed on, and the set of labels
    /// whose p-value is greater than the error rate.
    ///
    /// The same calibration, ratio, group and error rate always give the same decision
    /// (requirement 11.10). An error rate the calibration does not support is refused,
    /// see [`ConformalCalibrator::require_error_rate`]; a ratio that is not a number in
    /// `0.0..=1.0` is [`CalibrationError::InvalidRatio`]. A scorer's abstention has no
    /// ratio and is not passed here.
    pub fn predict(
        &self,
        ratio: f32,
        group: Option<&str>,
        error_rate: f32,
    ) -> Result<Calibrated, CalibrationError> {
        self.require_error_rate(error_rate)?;
        if !usable_ratio(ratio) {
            return Err(CalibrationError::InvalidRatio { ratio });
        }
        let (good, bad) = (self.list(false, group), self.list(true, group));
        let (p_good, p_bad) = (good.p_value(ratio), bad.p_value(ratio));
        Ok(Calibrated {
            set: LabelSet::of(p_good > error_rate, p_bad > error_rate),
            p_good,
            p_bad,
            good_stratum: good.stratum.clone(),
            bad_stratum: bad.stratum.clone(),
            error_rate,
        })
    }

    fn manifest_json(&self) -> Result<String, L2dError> {
        let manifest = Manifest {
            format: FORMAT,
            kind: KIND.into(),
            revision: self.revision.clone(),
            tool_version: env!("CARGO_PKG_VERSION").into(),
            scorer: ScorerFile {
                kind: self.scorer.as_str().into(),
                revision: self.scorer_revision.clone(),
            },
            settings: self.settings,
            good: self.good(),
            bad: self.bad(),
            lists: self
                .lists
                .iter()
                .map(|list| ListFile {
                    label: list.label().as_str().into(),
                    group: match &list.stratum {
                        Stratum::Pooled => None,
                        Stratum::Group(group) => Some(group.clone()),
                    },
                    logs: list.logs(),
                })
                .collect(),
        };
        serde_json::to_string_pretty(&manifest).map_err(|error| L2dError::Runtime {
            detail: format!("describing the calibration failed: {error}"),
        })
    }

    fn write_files(&self, dir: &Path) -> Result<(), L2dError> {
        let mut scores = Vec::new();
        for list in &self.lists {
            scores.extend_from_slice(&score_bytes(&list.scores));
        }
        write_file(&dir.join(SCORES_FILE), &scores)?;
        write_file(&dir.join(MANIFEST_FILE), self.manifest_json()?.as_bytes())
    }

    /// Saves the calibration as `models/calibrations/<name>-<rev12>/` and returns that
    /// directory. The directory appears complete or not at all.
    ///
    /// - A name that is not 1 to [`MAX_NAME_BYTES`] letters, digits, `.`, `_` or `-` is
    ///   [`L2dError::InvalidInput`].
    /// - A target that is not directly below the project's `models/calibrations/` is
    ///   [`L2dError::Refused`] (this includes a project root written with `..`).
    /// - When the directory already holds this very calibration, nothing is written.
    ///   When it holds anything else, the save is [`L2dError::Refused`] and the
    ///   directory is left unchanged.
    pub fn save(&self, paths: &ProjectPaths, name: &str) -> Result<PathBuf, L2dError> {
        validate_name(name)?;
        let parent = paths.calibrations_dir();
        let dir = paths.calibration_dir(name, &self.revision);
        if !paths.is_under_models(&dir) || dir.parent() != Some(parent.as_path()) {
            return Err(L2dError::Refused {
                reason: format!(
                    "calibrations are written only below the model directory {}; {} is not \
                     accepted as a place below it",
                    paths.models_dir().display(),
                    dir.display()
                ),
            });
        }
        if dir.exists() {
            return match Self::load(&dir) {
                Ok(stored) if stored == *self => Ok(dir),
                _ => Err(L2dError::Refused {
                    reason: format!(
                        "{} already exists and does not hold revision {} of this \
                         calibration; it was left unchanged",
                        dir.display(),
                        self.revision
                    ),
                }),
            };
        }

        fs::create_dir_all(&parent).map_err(|error| io_failure("creating", &parent, &error))?;
        let temporary = parent.join(format!(
            ".{name}-{}.saving-{}-{}",
            short_revision(&self.revision),
            std::process::id(),
            SAVES.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&temporary).map_err(|error| io_failure("creating", &temporary, &error))?;
        let written = self.write_files(&temporary).and_then(|()| {
            fs::rename(&temporary, &dir).map_err(|error| io_failure("completing", &dir, &error))
        });
        if written.is_err() {
            // Best effort: the failure itself is what the caller has to see.
            let _ = fs::remove_dir_all(&temporary);
        }
        written.map(|()| dir)
    }

    /// Loads a calibration directory after checking that `dir` lies below the project's
    /// model directory; anything else is [`L2dError::Refused`]. See
    /// [`ConformalCalibrator::load`].
    pub fn load_under(paths: &ProjectPaths, dir: &Path) -> Result<Self, L2dError> {
        if !paths.is_under_models(dir) {
            return Err(L2dError::Refused {
                reason: format!(
                    "calibrations are read only from below the model directory {}, not from {}",
                    paths.models_dir().display(),
                    dir.display()
                ),
            });
        }
        Self::load(dir)
    }

    /// Loads a saved calibration. The loaded calibration gives the same decisions.
    ///
    /// A directory or file that is not there is [`L2dError::Missing`], naming the item
    /// and where it was expected. Content that does not verify is [`L2dError::Refused`]:
    /// an unreadable `calibration.json`, another format or kind, a scores file of the
    /// wrong length, lists out of order, unsorted or with values outside `0.0..=1.0`, a
    /// revision that differs from the one computed from the files, or a directory name
    /// that does not end in the first twelve characters of the revision.
    pub fn load(dir: &Path) -> Result<Self, L2dError> {
        if !dir.is_dir() {
            return Err(L2dError::Missing {
                item: "calibration".into(),
                expected_at: dir.to_path_buf(),
            });
        }
        let manifest = read_calibration_file(dir, MANIFEST_FILE)?;
        let manifest: Manifest = serde_json::from_slice(&manifest).map_err(|error| {
            not_a_calibration(dir, format!("{MANIFEST_FILE} is not readable: {error}"))
        })?;
        if manifest.format != FORMAT || manifest.kind != KIND {
            return Err(not_a_calibration(
                dir,
                format!(
                    "{MANIFEST_FILE} has format {} and kind '{}', expected {FORMAT} and '{KIND}'",
                    manifest.format, manifest.kind
                ),
            ));
        }
        let scorer = ScorerKind::from_name(&manifest.scorer.kind).ok_or_else(|| {
            not_a_calibration(
                dir,
                format!(
                    "{MANIFEST_FILE} names the unknown scorer kind '{}'",
                    manifest.scorer.kind
                ),
            )
        })?;

        // The length is checked before the file is read, so a wrong file is never loaded.
        let expected_bytes = manifest
            .lists
            .iter()
            .try_fold(0_u64, |sum, list| sum.checked_add(list.logs))
            .and_then(|logs| logs.checked_mul(8))
            .ok_or_else(|| not_a_calibration(dir, "the list lengths do not add up"))?;
        let wrong_length = |found: u64| {
            not_a_calibration(
                dir,
                format!(
                    "{SCORES_FILE} has {found} bytes, but the lists of {MANIFEST_FILE} need \
                     {expected_bytes}"
                ),
            )
        };
        if let Ok(found) = fs::metadata(dir.join(SCORES_FILE))
            && found.len() != expected_bytes
        {
            return Err(wrong_length(found.len()));
        }
        let bytes = read_calibration_file(dir, SCORES_FILE)?;
        if bytes.len() as u64 != expected_bytes {
            return Err(wrong_length(bytes.len() as u64));
        }

        let mut values = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bits| f64::from_bits(u64::from_le_bytes(*bits)));
        let mut parts = Vec::with_capacity(manifest.lists.len());
        for list in &manifest.lists {
            let bad = match Label::from_name(&list.label) {
                Some(Label::Good) => false,
                Some(Label::Bad) => true,
                _ => {
                    return Err(not_a_calibration(
                        dir,
                        format!("{MANIFEST_FILE} has a list with the label '{}'", list.label),
                    ));
                }
            };
            let stratum = match &list.group {
                None => Stratum::Pooled,
                Some(group) => Stratum::Group(group.clone()),
            };
            let scores: Vec<f64> = values.by_ref().take(list.logs as usize).collect();
            parts.push((bad, stratum, scores));
        }
        drop(bytes);

        let calibrator = Self::assemble(scorer, manifest.scorer.revision, manifest.settings, parts)
            .map_err(|detail| not_a_calibration(dir, detail))?;
        if calibrator.good() != manifest.good || calibrator.bad() != manifest.bad {
            return Err(not_a_calibration(
                dir,
                format!(
                    "{MANIFEST_FILE} states {} good and {} bad calibration logs, the lists \
                     hold {} and {}",
                    manifest.good,
                    manifest.bad,
                    calibrator.good(),
                    calibrator.bad()
                ),
            ));
        }
        let revision = &calibrator.revision;
        if *revision != manifest.revision {
            return Err(L2dError::Refused {
                reason: format!(
                    "the calibration in {} does not match its revision: {MANIFEST_FILE} \
                     states {}, the files give {revision}",
                    dir.display(),
                    manifest.revision
                ),
            });
        }
        let short = short_revision(revision);
        let named = dir
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(&format!("-{short}")));
        if !named {
            return Err(L2dError::Refused {
                reason: format!(
                    "the calibration in {} has revision {revision}, but the directory name \
                     does not end in -{short}",
                    dir.display()
                ),
            });
        }
        Ok(calibrator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The p-value of the design, counted directly.
    fn direct(calibration: &[f64], candidate: f64) -> f64 {
        let at_least = calibration
            .iter()
            .filter(|score| **score >= candidate)
            .count();
        (at_least as f64 + 1.0) / (calibration.len() as f64 + 1.0)
    }

    fn sorted(mut scores: Vec<f64>) -> Vec<f64> {
        scores.sort_unstable_by(f64::total_cmp);
        scores
    }

    /// The p-value is `(count of calibration scores >= candidate + 1) / (n + 1)` with
    /// ties counted: the binary search on the sorted list against a direct count over
    /// the list as it came. (Until task 1.18 this test pinned the same of the crate
    /// `conformal-prediction` 2.0.1, on the same values.)
    #[test]
    fn the_p_value_is_the_documented_one_with_ties_counted() {
        // Many equal values: 200 scores on 21 distinct values.
        let calibration: Vec<f64> = (0..200)
            .map(|index| f64::from(index * 8 % 21) / 20.0)
            .collect();
        let list = sorted(calibration.clone());
        let candidates: Vec<f64> = (0..=80).map(|step| f64::from(step) / 80.0).collect();

        for asked in &candidates {
            assert_eq!(
                p_value_of(&list, *asked).to_bits(),
                direct(&calibration, *asked).to_bits(),
                "candidate {asked}"
            );
        }
        // A candidate equal to calibration scores counts them: 0.5 is one of the values.
        assert!(calibration.contains(&0.5));
        let ties = calibration.iter().filter(|score| **score == 0.5).count();
        let above = calibration.iter().filter(|score| **score > 0.5).count();
        assert!(ties > 0);
        assert_eq!(at_least(&list, 0.5), ties + above);
        assert_eq!(p_value_of(&list, 0.5), (ties + above + 1) as f64 / 201.0);

        // The set at a level: exactly the candidates with a greater p-value.
        for level in [0.05, 0.1, 0.25, 0.5] {
            let kept: Vec<f64> = candidates
                .iter()
                .copied()
                .filter(|candidate| p_value_of(&list, *candidate) > level)
                .collect();
            let expected: Vec<f64> = candidates
                .iter()
                .copied()
                .filter(|candidate| direct(&calibration, *candidate) > level)
                .collect();
            assert_eq!(kept, expected, "level {level}");
        }
    }

    /// The count at the edges: below, between, on and above the values, runs of equal
    /// values at both ends, one value, and both zeros.
    #[test]
    fn the_count_of_values_at_least_a_candidate_is_exact_at_the_edges() {
        let list = [0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0];
        for (candidate, expected) in [
            (0.0, 8),
            (-0.0, 8),
            (f64::MIN_POSITIVE, 6),
            (0.25, 6),
            (0.3, 5),
            (0.5, 5),
            (0.5 + f64::EPSILON, 2),
            (1.0 - f64::EPSILON / 2.0, 2),
            (1.0, 2),
        ] {
            assert_eq!(
                at_least(&list, candidate),
                expected,
                "candidate {candidate}"
            );
            let counted = list.iter().filter(|score| **score >= candidate).count();
            assert_eq!(counted, expected, "candidate {candidate}");
        }
        assert_eq!(at_least(&[0.5], 0.5), 1);
        assert_eq!(at_least(&[0.5], 0.6), 0);
        assert_eq!(at_least(&[0.5], 0.4), 1);
        // A stored negative zero (a hand-made file) is a zero.
        assert_eq!(at_least(&[0.0, -0.0, 0.0], 0.0), 3);
        assert_eq!(at_least(&[-0.0, 0.0], f64::MIN_POSITIVE), 0);
        // The smallest and the largest p-value of a list.
        assert_eq!(p_value_of(&list, 1.0 + f64::EPSILON), 1.0 / 9.0);
        assert_eq!(p_value_of(&list, 0.0), 1.0);
    }

    #[test]
    fn the_same_p_values_come_out_on_repeat() {
        let calibration: Vec<f64> = (0..500)
            .map(|index| f64::from(index * 37 % 101) / 100.0)
            .collect();
        let candidates: Vec<f64> = (0..=100).map(|step| f64::from(step) / 100.0).collect();
        let run = || {
            let list = sorted(calibration.clone());
            let p_values = |list: &[f64]| -> Vec<u64> {
                candidates
                    .iter()
                    .map(|candidate| p_value_of(list, *candidate).to_bits())
                    .collect()
            };
            let first = p_values(&list);
            assert_eq!(first, p_values(&list));
            for (candidate, p_value) in candidates.iter().zip(&first) {
                assert_eq!(*p_value, direct(&calibration, *candidate).to_bits());
            }
            first
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn nonconformity_is_the_ratio_for_good_and_one_less_the_ratio_for_bad() {
        assert_eq!(nonconformity(0.25, false), 0.25);
        assert_eq!(nonconformity(0.25, true), 0.75);
        assert_eq!(nonconformity(0.0, true), 1.0);
        assert_eq!(nonconformity(1.0, true), 0.0);
        // Negative zero is zero.
        assert_eq!(nonconformity(-0.0, false).to_bits(), 0.0_f64.to_bits());
        // A higher ratio never fits good better or bad worse.
        let ratios = [
            0.0_f32,
            f32::MIN_POSITIVE,
            1e-12,
            0.3,
            0.5,
            1.0 - f32::EPSILON,
            1.0,
        ];
        for pair in ratios.windows(2) {
            assert!(nonconformity(pair[0], false) < nonconformity(pair[1], false));
            assert!(nonconformity(pair[0], true) >= nonconformity(pair[1], true));
        }
    }

    #[test]
    fn the_smallest_rate_of_a_list_is_one_over_its_length_plus_one() {
        assert_eq!(smallest_rate(19), 0.05);
        assert_eq!(smallest_rate(99), 0.01);
        assert_eq!(smallest_rate(0), 1.0);
        assert!(smallest_rate(18) > 0.05);
    }
}
