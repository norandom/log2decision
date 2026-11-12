//! The contract every scorer meets (requirement 6), and the baseline scorer under it.
//!
//! A scorer gives each log a bad/good ratio or abstains. Triage ([`crate::triage`]) and
//! calibration ([`crate::conformal::calibration_points`]) work on this contract alone,
//! so they serve the baseline scorer now and the neighbour scorer and the linear head
//! when those exist.
//!
//! # The scores frame
//!
//! [`Scorer::score`] returns the Scores frame of the design's "Data frames": one row per
//! row of the logs frame it was given, in the same order.
//!
//! | Column | Type | Null | Meaning |
//! |--------|------|------|---------|
//! | `text` | String | no | the text of the row, equal to the logs frame's |
//! | `scorer` | String | no | [`ScorerKind::as_str`] of the scorer, on every row |
//! | `ratio` | Float32 | where abstained | the bad/good ratio, `0.0..=1.0` |
//! | `abstained` | Boolean | no | the scorer gives no ratio for this log |
//!
//! Further columns are the scorer's evidence (the baseline's `no_features`); they keep
//! their names, types and order from batch to batch. Triage keeps the evidence of its
//! primary scorer and takes String, Boolean and numeric columns only, as many as
//! [`ScoringCost::evidence_columns`] states: evidence of another shape (a list per log)
//! is written as such columns. [`check_scores`] holds a frame to the four columns above;
//! the rules for evidence columns are applied by triage when it keeps them, not by
//! `check_scores` (calibration drops evidence).
//!
//! # Memory
//!
//! A scorer says what scoring costs it ([`Scorer::cost`]), so that an operation that
//! scores in batches can plan them under the memory budget (requirement 12).

use l2d_baseline::BaselineScorer;
use l2d_core::polars::prelude::{Column, DataFrame, DataType};
use l2d_core::{L2dError, LogsFrame};

use crate::ScorerKind;

pub use l2d_baseline::scorer::{SCORES_ABSTAINED, SCORES_RATIO, SCORES_SCORER, SCORES_TEXT};

/// The columns every scores frame has, in the order of the table above.
pub const SCORES_COLUMNS: [&str; 4] = [SCORES_TEXT, SCORES_SCORER, SCORES_RATIO, SCORES_ABSTAINED];

/// What scoring costs a scorer in memory, in bytes.
///
/// The figures are the scorer's own statement. An operation plans its batches and its
/// share of the memory budget from them and cannot measure the scorer: a scorer that
/// understates its cost makes the operation exceed what it asked for (requirement
/// 12.9), without an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScoringCost {
    /// Taken while a batch is scored, whatever the number of its rows (for the
    /// baseline: hashing the longest text). A column of words in a batch's scores frame
    /// takes at least 8 KiB whatever the rows; that belongs here.
    pub fixed_bytes: u64,
    /// Taken per row of a batch while the batch is scored and its scores frame is
    /// alive.
    pub row_bytes: u64,
    /// Per row, what the scorer's evidence columns take: the width of a flag or number,
    /// and for a column of words 16 bytes and the mean bytes of its values over the logs
    /// to be scored (what is kept is the values end to end; a figure above the mean is
    /// safe, but stating the longest value for every log can make an operation refuse a
    /// run that would fit). An operation that keeps them for every log counts this for
    /// every log.
    pub evidence_row_bytes: u64,
    /// How many evidence columns its scores frame has.
    pub evidence_columns: u32,
}

/// A scorer: a ratio per log, or an abstention.
pub trait Scorer {
    /// Which scorer this is; named beside every ratio (requirement 6.6).
    fn kind(&self) -> ScorerKind;

    /// The revision of the fitted scorer; a calibration is tied to it.
    fn revision(&self) -> &str;

    /// What scoring batches of `logs` costs. `logs` is the whole set that is about to
    /// be scored batch by batch, so the scorer can look at its longest text.
    fn cost(&self, logs: &LogsFrame) -> ScoringCost;

    /// The scores frame of the rows of `logs`, in the same order (module
    /// documentation). Labels play no part.
    fn score(&mut self, logs: &LogsFrame) -> Result<DataFrame, L2dError>;
}

/// Memory per row of the baseline's scores frame while it is built: the 16 bytes of
/// the scorer's name twice (the list it is built from and the column), the ratio twice
/// and the two flags twice, rounded up.
const BASELINE_ROW_BYTES: u64 = 64;
/// The baseline's evidence is one flag per row (`no_features`).
const BASELINE_EVIDENCE_ROW_BYTES: u64 = 2;

/// The baseline scorer never abstains; a text without features gets the constant ratio
/// of the bias and is flagged in the evidence column `no_features`.
impl Scorer for BaselineScorer {
    fn kind(&self) -> ScorerKind {
        ScorerKind::Baseline
    }

    fn revision(&self) -> &str {
        BaselineScorer::revision(self)
    }

    fn cost(&self, logs: &LogsFrame) -> ScoringCost {
        let longest = logs
            .frame()
            .column(LogsFrame::TEXT)
            .and_then(Column::str)
            .map(|texts| {
                texts
                    .iter()
                    .flatten()
                    .map(|text| text.len() as u64)
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        ScoringCost {
            fixed_bytes: self.scoring_bytes(longest),
            row_bytes: BASELINE_ROW_BYTES,
            evidence_row_bytes: BASELINE_EVIDENCE_ROW_BYTES,
            evidence_columns: 1,
        }
    }

    fn score(&mut self, logs: &LogsFrame) -> Result<DataFrame, L2dError> {
        BaselineScorer::score(self, logs)
    }
}

/// Holds `scores` to the contract of the module documentation for the rows of `logs`
/// and the scorer `kind`; the error says what is wrong.
///
/// Checked: the four columns with their types, one row per log, the same text on every
/// row, the scorer's name on every row, no null flag, and on every row that is not
/// abstained a ratio that is a number in `0.0..=1.0`.
pub fn check_scores(scores: &DataFrame, logs: &LogsFrame, kind: ScorerKind) -> Result<(), String> {
    let typed = |name: &str, wanted: DataType| -> Result<&Column, String> {
        let column = scores
            .column(name)
            .map_err(|_| format!("the column '{name}' is missing"))?;
        if *column.dtype() != wanted {
            return Err(format!(
                "the column '{name}' has the type {}, not {wanted}",
                column.dtype()
            ));
        }
        Ok(column)
    };
    let text = typed(SCORES_TEXT, DataType::String)?;
    let scorer = typed(SCORES_SCORER, DataType::String)?;
    let ratio = typed(SCORES_RATIO, DataType::Float32)?;
    let abstained = typed(SCORES_ABSTAINED, DataType::Boolean)?;
    if scores.height() != logs.len() {
        return Err(format!(
            "it has {} rows for {} logs",
            scores.height(),
            logs.len()
        ));
    }
    let polars = |error| format!("reading it failed: {error}");
    let wanted = logs
        .frame()
        .column(LogsFrame::TEXT)
        .and_then(Column::str)
        .map_err(polars)?;
    let text = text.str().map_err(polars)?;
    if let Some(row) = text
        .iter()
        .zip(wanted.iter())
        .position(|(found, wanted)| found != wanted)
    {
        return Err(format!("row {row} does not hold the text of that log"));
    }
    let name = kind.as_str();
    if let Some(other) = scorer
        .str()
        .map_err(polars)?
        .iter()
        .find(|found| *found != Some(name))
    {
        return Err(format!(
            "the column '{SCORES_SCORER}' names '{}', not '{name}'",
            other.unwrap_or("null")
        ));
    }
    let abstained = abstained.bool().map_err(polars)?;
    if abstained.null_count() > 0 {
        return Err(format!(
            "the column '{SCORES_ABSTAINED}' holds {} null values",
            abstained.null_count()
        ));
    }
    for (row, (ratio, abstained)) in ratio
        .f32()
        .map_err(polars)?
        .iter()
        .zip(abstained.iter())
        .enumerate()
    {
        if abstained == Some(true) {
            continue;
        }
        match ratio {
            Some(ratio) if (0.0..=1.0).contains(&ratio) => {}
            Some(ratio) => {
                return Err(format!(
                    "row {row}: a ratio lies between 0 and 1; got {ratio}"
                ));
            }
            None => {
                return Err(format!(
                    "row {row} has no ratio and is not marked as abstained"
                ));
            }
        }
    }
    Ok(())
}
