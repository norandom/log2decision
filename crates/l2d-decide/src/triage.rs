//! Triage: every log of a set ranked for review, scored down, or abstained on
//! (requirements 6.4 to 6.6 and 6.9, 11.6, 11.7, 11.9, 11.10, 12.6, 12.9).
//!
//! # What triage does
//!
//! [`triage_with`] has every scorer score the logs ([`crate::scorer::Scorer`]) and
//! places each log in exactly one of three lists by the ratio of the **primary**
//! scorer:
//!
//! - **Without a calibration.** A log the primary scorer abstained on goes to the
//!   abstained list (`scorer_abstained`). Every other log is ranked by ratio, highest
//!   first. Without a review budget all of them are the review list; with a budget of
//!   `N` the first `N` are, and the rest are scored down (`budget_only`).
//! - **With a calibration** of the primary scorer, a log is placed by its label set at
//!   the chosen error rate (requirement 11.6): **bad** alone is ranked for review;
//!   **good** alone is scored down (`calibration`); **both** is abstained on
//!   (`uncertain`); **neither** is abstained on (`unlike_calibration`). A log the
//!   scorer abstained on gets no label set and is abstained on (`scorer_abstained`).
//!   The review budget then applies to the review ranking alone: logs ranked below it
//!   are scored down (`budget_only`), so every scored-down log says which applied
//!   (requirement 11.7).
//!
//! The review budget counts logs, which are rows of the logs frame (distinct texts),
//! not occurrences. A budget of 0 sends nothing to review.
//!
//! ## The tie rule
//!
//! The ranking is by ratio, descending; logs with equal ratios keep the order of the
//! logs frame, the earlier row first (the two zeros are one ratio). The same rule cuts
//! a tie at the review budget. The baseline's logs without features all have one
//! ratio, so they are ranked among themselves in the order of the logs frame; they are
//! placed like any other log and flagged in the evidence column `no_features`.
//!
//! ## What triage does not do
//!
//! Logs of an endpoint the calibration logs do not cover are placed by their label set
//! like any other log (owner decision of 2026-11-12); ratios are not normalised per
//! endpoint. The error rate holds only for logs that resemble the calibration logs,
//! so every result with a calibration carries that sentence
//! ([`CalibrationUse::note`], requirement 11.9) and says how many logs were judged on
//! their group's own list and how many on the pooled list, per label
//! ([`CalibrationUse::good`], [`CalibrationUse::bad`]).
//!
//! Triage reads the logs frame and writes nothing: it has no access to a store, takes
//! the logs by shared reference and returns new frames (requirement 6.9).
//!
//! # The triage frame
//!
//! The result is one Polars frame ([`TriageResult::frame`]) with one row per log, in
//! the order of the logs frame, and the list of each log in a column. One frame with a
//! list column, and not three frames, because then the postcondition "every log is in
//! exactly one list" is the shape of the data: a row has one value in `list`. The three
//! lists are views of it in reading order ([`TriageResult::review`],
//! [`TriageResult::scored_down`], [`TriageResult::abstained`]).
//!
//! | Column | Type | Null | Content |
//! |--------|------|------|---------|
//! | `text`, `label`, `group`, `first_position`, `occurrences` | as in the logs frame | | the logs frame's columns |
//! | `scorer` | String | no | the primary scorer's name (requirement 6.6) |
//! | `ratio` | Float32 | where it abstained | the primary scorer's ratio |
//! | `abstained` | Boolean | no | the primary scorer abstained |
//! | `list` | String | no | `review`, `scored_down` or `abstained` |
//! | `rank` | UInt32 | unless ranked | place in the review ranking from 1, before the budget cuts it |
//! | `scored_down_by` | String | unless scored down | `calibration` or `budget_only` |
//! | `abstain_reason` | String | unless abstained | `scorer_abstained`, `uncertain`, `unlike_calibration` |
//! | `label_set` | String | without a calibration, and where the scorer abstained | `good`, `bad`, `both`, `neither` |
//! | `p_good`, `p_bad` | Float32 | as `label_set` | the p-values the set was made from |
//! | `good_stratum`, `bad_stratum` | String | as `label_set` | `group` (the list of the row's own group) or `pooled` |
//! | `ratio_<kind>` | Float32 | where that scorer abstained | one column per scorer beside the primary |
//! | evidence | the scorer's | the scorer's | the primary scorer's evidence columns (`no_features`) |
//!
//! A ranked log is in review when its rank is within the budget and scored down by the
//! budget otherwise. The names of the first seven columns after the logs frame's are
//! those of the design's Scores and Predictions frames. A triage of no logs has no
//! evidence columns.
//!
//! Reading order of the lists: review by rank; scored down by rank for those the
//! budget scored down, then those calibration scored down by ratio, highest first,
//! with the tie rule; abstained in the order of the logs frame.
//!
//! # Batches and memory
//!
//! The logs are scored in batches planned under the caller's [`MemoryBudget`]
//! (requirement 12): the free memory is read before each batch, and the plan with its
//! changes is in the result ([`TriageResult::plan`]). The result does not depend on
//! the batch size (requirement 12.6): scoring and the label set are per log, and the
//! ranking is made once over all logs.
//!
//! Beside its batches triage holds, on top of what the caller's budget holds already
//! and for as long as it runs ([`TriageResult::held_bytes`]):
//!
//! - [`RESULT_ROW_BYTES`] per log for what it keeps per log and the triage frame it
//!   builds from that (the columns of the logs frame are shared with the caller's
//!   frame and cost nothing), [`OTHER_SCORER_ROW_BYTES`] per log and further scorer,
//!   and three times the primary scorer's [`ScoringCost::evidence_row_bytes`] per log;
//! - every scorer's [`ScoringCost::fixed_bytes`], and [`EVIDENCE_COLUMN_BYTES`] for each
//!   evidence column the primary scorer states;
//! - the calibration: [`ConformalCalibrator::held_bytes`] (requirement 12.9). The
//!   caller loads the calibration and does not count it itself.
//!
//! A batch is asked for at the largest [`ScoringCost::row_bytes`] of the scorers per
//! row. When the memory limit does not leave what triage holds, or the smallest batch,
//! the run is [`TriageError::Memory`] before a log is scored, stating the bytes needed
//! and left. The budget is given back holding exactly what it held, whatever the
//! outcome.
//!
//! ## Evidence
//!
//! Only the primary scorer's evidence columns are kept; a scorer beside it contributes
//! its ratio alone. Evidence columns are String, Boolean or numbers (another type is
//! [`TriageError::Scores`]), and no more of them than the scorer's cost states. They are
//! kept so that the memory does not depend on the batch size: flags and numbers in at
//! most 128 pieces that are merged as they accumulate; words copied out of each batch
//! into one text per column, from which the column is built once at the end. (A String
//! column of Polars is views into blocks of at least 8 KiB, and stacking or merging such
//! columns keeps the block of every batch alive: with batches of two rows that was
//! fourteen times what triage had asked for.) Words take up to three times their bytes
//! while the text grows and while the column is built, which is why the evidence is
//! counted three times.
//!
//! The String columns triage writes itself (`scorer`, `list`, the reasons, the label
//! set and the strata) are each built once, after the last batch, from one value per
//! log; none accumulates across batches. Only `scorer_abstained` and
//! `unlike_calibration` are longer than the 12 bytes a row holds in place; they are in
//! [`RESULT_ROW_BYTES`].
//! `tests/triage_memory.rs` measures the heap against these figures.

use std::collections::BTreeSet;

use l2d_core::polars::prelude::{
    Column, DataFrame, DataType, IdxCa, IdxSize, IntoColumn, PlSmallStr, PolarsError, StringChunked,
};
use l2d_core::{BatchPlan, L2dError, LogsFrame, MemoryBudget, MemorySettings};
use serde::Serialize;

use crate::ScorerKind;
use crate::conformal::{
    CalibrationError, ConformalCalibrator, EXCHANGEABILITY_NOTE, LabelSet, Stratum,
};
use crate::scorer::{
    SCORES_ABSTAINED, SCORES_COLUMNS, SCORES_RATIO, Scorer, ScoringCost, check_scores,
};

/// Column of the triage frame: the primary scorer's name, String, never null.
pub const TRIAGE_SCORER: &str = "scorer";
/// Column of the triage frame: the primary scorer's ratio, Float32, null where it
/// abstained.
pub const TRIAGE_RATIO: &str = "ratio";
/// Column of the triage frame: the primary scorer abstained, Boolean, never null.
pub const TRIAGE_ABSTAINED: &str = "abstained";
/// Column of the triage frame: the list the log is in ([`TriageList`]), String, never
/// null.
pub const TRIAGE_LIST: &str = "list";
/// Column of the triage frame: the place in the review ranking, UInt32 from 1, null for
/// a log that is not ranked.
pub const TRIAGE_RANK: &str = "rank";
/// Column of the triage frame: what scored the log down ([`ScoredDownBy`]), String, null
/// unless the list is `scored_down`.
pub const TRIAGE_SCORED_DOWN_BY: &str = "scored_down_by";
/// Column of the triage frame: why the log is abstained on ([`AbstainReason`]), String,
/// null unless the list is `abstained`.
pub const TRIAGE_ABSTAIN_REASON: &str = "abstain_reason";
/// Column of the triage frame: the label set, String, null without a calibration and
/// where the scorer abstained.
pub const TRIAGE_LABEL_SET: &str = "label_set";
/// Column of the triage frame: the p-value for good, Float32, null like `label_set`.
pub const TRIAGE_P_GOOD: &str = "p_good";
/// Column of the triage frame: the p-value for bad, Float32, null like `label_set`.
pub const TRIAGE_P_BAD: &str = "p_bad";
/// Column of the triage frame: the list `p_good` was computed on, `group` (the list of
/// the row's own group) or `pooled`, String, null like `label_set`.
pub const TRIAGE_GOOD_STRATUM: &str = "good_stratum";
/// Column of the triage frame: the list `p_bad` was computed on, as `good_stratum`.
pub const TRIAGE_BAD_STRATUM: &str = "bad_stratum";
/// Prefix of the columns that hold the ratios of the scorers beside the primary one:
/// `ratio_<kind>`, Float32, null where that scorer abstained.
pub const TRIAGE_RATIO_PREFIX: &str = "ratio_";
/// Value of the stratum columns: the list of the row's own group.
pub const STRATUM_GROUP: &str = "group";
/// Value of the stratum columns: the pooled list of the label.
pub const STRATUM_POOLED: &str = "pooled";

/// The columns triage adds after those of the logs frame, in order.
const TRIAGE_COLUMNS: [&str; 12] = [
    TRIAGE_SCORER,
    TRIAGE_RATIO,
    TRIAGE_ABSTAINED,
    TRIAGE_LIST,
    TRIAGE_RANK,
    TRIAGE_SCORED_DOWN_BY,
    TRIAGE_ABSTAIN_REASON,
    TRIAGE_LABEL_SET,
    TRIAGE_P_GOOD,
    TRIAGE_P_BAD,
    TRIAGE_GOOD_STRATUM,
    TRIAGE_BAD_STRATUM,
];

/// Memory per log that triage holds for its result, in bytes: about 20 for what it
/// keeps per log while the batches run and 12 for the reading order of the lists; the
/// triage frame's own columns take about 150 (seven String columns at 16 bytes a row,
/// up to 18 more where a reason is longer than 12 bytes, and the numbers and flags);
/// and one String column at a time is built from a list of 16 bytes a row. Rounded up.
/// `tests/triage_memory.rs` holds the heap to it.
pub const RESULT_ROW_BYTES: u64 = 256;
/// Memory per log and scorer beside the primary one: its ratio while the batches run
/// (8 bytes) and its column of the triage frame (about 5).
pub const OTHER_SCORER_ROW_BYTES: u64 = 16;
/// Memory per evidence column of the primary scorer, whatever the number of logs: the
/// pieces a fixed-width column is kept in while the batches run (at most 128, a few
/// hundred bytes each), or the smallest block of a column of words (8 KiB), with room
/// for what Polars rounds up.
pub const EVIDENCE_COLUMN_BYTES: u64 = 48 * 1024;

/// The three lists of a triage result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TriageList {
    /// Ranked for review, within the review budget.
    Review,
    /// Scored down, by calibration or by the review budget.
    ScoredDown,
    /// No decision: the scorer abstained, or the label set is both or neither.
    Abstained,
}

impl TriageList {
    /// The lower-case name used in the triage frame and in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            TriageList::Review => "review",
            TriageList::ScoredDown => "scored_down",
            TriageList::Abstained => "abstained",
        }
    }
}

/// Why a log is in the abstained list (requirement 11.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstainReason {
    /// The scorer gave no ratio.
    ScorerAbstained,
    /// The label set is both: neither label can be ruled out.
    Uncertain,
    /// The label set is neither: the log is unlike the calibration logs.
    UnlikeCalibration,
}

impl AbstainReason {
    /// The lower-case name used in the triage frame and in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            AbstainReason::ScorerAbstained => "scorer_abstained",
            AbstainReason::Uncertain => "uncertain",
            AbstainReason::UnlikeCalibration => "unlike_calibration",
        }
    }
}

/// What scored a log down (requirement 11.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoredDownBy {
    /// The label set is good alone, at the stated error rate.
    Calibration,
    /// Only the review budget: the log is ranked below it.
    BudgetOnly,
}

impl ScoredDownBy {
    /// The lower-case name used in the triage frame and in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            ScoredDownBy::Calibration => "calibration",
            ScoredDownBy::BudgetOnly => "budget_only",
        }
    }
}

/// A calibration and the error rate its label sets are made for.
#[derive(Debug, Clone, Copy)]
pub struct TriageCalibration<'c> {
    pub calibrator: &'c ConformalCalibrator,
    pub error_rate: f32,
}

/// What a triage run is asked to do.
#[derive(Debug, Clone, Copy)]
pub struct TriageSettings<'c> {
    /// The scorer whose ratio ranks the logs and is judged by the calibration.
    pub primary: ScorerKind,
    /// The review budget: how many logs (rows of the logs frame) go to review.
    pub review_budget: Option<u64>,
    /// The calibration of the primary scorer, if the logs are placed by label set.
    pub calibration: Option<TriageCalibration<'c>>,
}

/// How many logs are in each list, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct TriageCounts {
    /// The logs triaged: the rows of the logs frame.
    pub logs: u64,
    /// The review list.
    pub review: u64,
    /// Scored down because the label set is good alone.
    pub scored_down_by_calibration: u64,
    /// Scored down because ranked below the review budget.
    pub scored_down_by_budget: u64,
    /// Abstained on because the primary scorer gave no ratio.
    pub scorer_abstained: u64,
    /// Abstained on because the label set is both.
    pub uncertain: u64,
    /// Abstained on because the label set is neither.
    pub unlike_calibration: u64,
}

impl TriageCounts {
    /// The logs of the scored-down list.
    pub fn scored_down(&self) -> u64 {
        self.scored_down_by_calibration + self.scored_down_by_budget
    }

    /// The logs of the abstained list.
    pub fn abstained(&self) -> u64 {
        self.scorer_abstained + self.uncertain + self.unlike_calibration
    }
}

/// How many logs were judged on their group's own list and how many on the pooled
/// list, for one label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct ListUse {
    pub group: u64,
    pub pooled: u64,
}

/// What a triage result says about the calibration it used.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationUse {
    /// The revision of the calibration.
    pub revision: String,
    /// The error rate the label sets were made for.
    pub error_rate: f32,
    /// Requirement 11.9: [`EXCHANGEABILITY_NOTE`], the sentence every output of these
    /// decisions states.
    pub note: &'static str,
    /// The memory the calibration keeps, which triage held for as long as it ran.
    pub held_bytes: u64,
    /// The lists the p-values for good were computed on, over the logs with a label
    /// set.
    pub good: ListUse,
    /// The lists the p-values for bad were computed on.
    pub bad: ListUse,
    /// The logs whose label set is good alone.
    pub set_good: u64,
    /// The logs whose label set is bad alone.
    pub set_bad: u64,
    /// The logs whose label set is both.
    pub set_both: u64,
    /// The logs whose label set is neither.
    pub set_neither: u64,
}

/// Why a triage run was not made.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TriageError {
    /// The scorers given cannot be used as asked.
    #[error("unusable scorers: {detail}")]
    InvalidScorers { detail: String },
    /// The calibration does not fit the primary scorer or the error rate.
    #[error("{0}")]
    Calibration(CalibrationError),
    /// The memory limit does not leave room (requirement 12).
    #[error("{0}")]
    Memory(L2dError),
    /// A scorer failed.
    #[error("{0}")]
    Scorer(L2dError),
    /// A scorer returned something that is not its scores frame.
    #[error("the {scorer} scorer returned an unusable scores frame: {detail}")]
    Scores {
        scorer: &'static str,
        detail: String,
    },
    /// Polars failed.
    #[error("triage failed: {detail}")]
    Internal { detail: String },
}

impl From<TriageError> for L2dError {
    fn from(error: TriageError) -> Self {
        match error {
            TriageError::Memory(inner) | TriageError::Scorer(inner) => inner,
            TriageError::Calibration(inner) => inner.into(),
            TriageError::InvalidScorers { .. } => L2dError::InvalidInput {
                detail: error.to_string(),
            },
            TriageError::Scores { .. } | TriageError::Internal { .. } => L2dError::Runtime {
                detail: error.to_string(),
            },
        }
    }
}

fn internal(doing: &str, error: impl std::fmt::Display) -> TriageError {
    TriageError::Internal {
        detail: format!("{doing}: {error}"),
    }
}

/// The result of a triage run. See the module documentation.
#[derive(Debug, Clone)]
pub struct TriageResult {
    frame: DataFrame,
    /// The rows of each list in reading order: review, scored down, abstained.
    order: [Vec<IdxSize>; 3],
    primary: ScorerKind,
    scorers: Vec<ScorerKind>,
    review_budget: Option<u64>,
    counts: TriageCounts,
    calibration: Option<CalibrationUse>,
    plan: BatchPlan,
    held_bytes: u64,
}

impl TriageResult {
    /// The triage frame: one row per log, in the order of the logs frame.
    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    /// Gives up the result and returns the triage frame.
    pub fn into_frame(self) -> DataFrame {
        self.frame
    }

    /// The scorer that ranked the logs.
    pub fn primary(&self) -> ScorerKind {
        self.primary
    }

    /// Every scorer that scored the logs, in the order given (requirement 6.6).
    pub fn scorers(&self) -> &[ScorerKind] {
        &self.scorers
    }

    /// The review budget that was applied.
    pub fn review_budget(&self) -> Option<u64> {
        self.review_budget
    }

    /// How many logs are in each list.
    pub fn counts(&self) -> &TriageCounts {
        &self.counts
    }

    /// The calibration that placed the logs, if there was one.
    pub fn calibration(&self) -> Option<&CalibrationUse> {
        self.calibration.as_ref()
    }

    /// The batches the logs were scored in (requirement 12.3).
    pub fn plan(&self) -> &BatchPlan {
        &self.plan
    }

    /// The memory triage held beside the batches while it ran, the calibration's
    /// included.
    pub fn held_bytes(&self) -> u64 {
        self.held_bytes
    }

    /// The review list: the rows of the triage frame that are ranked within the
    /// budget, highest ratio first.
    pub fn review(&self) -> Result<DataFrame, L2dError> {
        self.list(TriageList::Review)
    }

    /// The scored-down list: the rows the budget scored down in rank order, then the
    /// rows calibration scored down, highest ratio first.
    pub fn scored_down(&self) -> Result<DataFrame, L2dError> {
        self.list(TriageList::ScoredDown)
    }

    /// The abstained list, in the order of the logs frame.
    pub fn abstained(&self) -> Result<DataFrame, L2dError> {
        self.list(TriageList::Abstained)
    }

    /// The rows of one list, in its reading order.
    pub fn list(&self, list: TriageList) -> Result<DataFrame, L2dError> {
        let rows = &self.order[list as usize];
        self.frame
            .take(&IdxCa::from_vec("row".into(), rows.clone()))
            .map_err(|error| L2dError::Runtime {
                detail: format!("reading the {} list failed: {error}", list.as_str()),
            })
    }
}

/// Where the primary scorer's ratio puts one log, before the ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Ranked: a ratio without a calibration, or the label set bad.
    Ranked,
    /// The label set good.
    Good,
    /// The label set both.
    Both,
    /// The label set neither.
    Neither,
    /// The scorer gave no ratio.
    ScorerAbstained,
}

/// The p-values of one log and whether each came from the group's own list.
#[derive(Debug, Clone, Copy)]
struct Judged {
    p_good: f32,
    p_bad: f32,
    good_own: bool,
    bad_own: bool,
}

/// What triage keeps per log while the batches run.
struct Kept {
    /// The primary scorer's ratio; 0 where it abstained.
    ratio: Vec<f32>,
    verdict: Vec<Verdict>,
    /// One entry per log, with a calibration only.
    judged: Vec<Option<Judged>>,
    /// The ratios of the scorers beside the primary one, in the order given.
    others: Vec<Vec<Option<f32>>>,
    /// The primary scorer's evidence columns.
    evidence: Evidence,
}

/// How many pieces of fixed-width evidence are stacked before they are made one.
const EVIDENCE_PIECES: usize = 64;

/// The fixed-width evidence columns (flags and numbers), batch below batch.
///
/// A frame stacked from many small batches keeps a piece per batch, and each piece
/// costs a few hundred bytes whatever it holds. So the batches are gathered in
/// `pending`, made one piece every [`EVIDENCE_PIECES`] batches and put below `settled`,
/// which is itself made one piece every [`EVIDENCE_PIECES`] times: at most twice that
/// many pieces exist at any time, whatever the batch size. Making fixed-width pieces
/// one copies their values into one new block and frees the old ones.
#[derive(Default)]
struct Pieces {
    settled: Option<DataFrame>,
    settled_pieces: usize,
    pending: Option<DataFrame>,
    pending_pieces: usize,
}

impl Pieces {
    fn stack(below: &mut Option<DataFrame>, piece: DataFrame) -> Result<(), PolarsError> {
        match below {
            None => *below = Some(piece),
            Some(all) => {
                all.vstack_mut(&piece)?;
            }
        }
        Ok(())
    }

    fn push(&mut self, batch: DataFrame) -> Result<(), PolarsError> {
        Self::stack(&mut self.pending, batch)?;
        self.pending_pieces += 1;
        if self.pending_pieces >= EVIDENCE_PIECES {
            self.settle()?;
        }
        Ok(())
    }

    fn settle(&mut self) -> Result<(), PolarsError> {
        let Some(mut pending) = self.pending.take() else {
            return Ok(());
        };
        self.pending_pieces = 0;
        pending.rechunk_mut();
        Self::stack(&mut self.settled, pending)?;
        self.settled_pieces += 1;
        if self.settled_pieces >= EVIDENCE_PIECES
            && let Some(settled) = &mut self.settled
        {
            settled.rechunk_mut();
            self.settled_pieces = 1;
        }
        Ok(())
    }

    /// All batches in one piece per column; `None` when no batch was pushed.
    fn finish(mut self) -> Result<Option<DataFrame>, PolarsError> {
        self.settle()?;
        Ok(self.settled.map(|mut settled| {
            settled.rechunk_mut();
            settled
        }))
    }
}

/// One evidence column of words, kept as its values and nothing else.
///
/// A String column of Polars is views into blocks of bytes, and stacking or merging
/// such columns keeps every block of every batch alive (a block is at least 8 KiB,
/// whatever the batch holds). So the words are copied out of each batch into one
/// growing text, with where each value ends and whether it is null, and the column is
/// built from that once, at the end.
struct Words {
    text: String,
    /// Where each value ends in `text`; a null value ends where it starts.
    ends: Vec<usize>,
    valid: Vec<bool>,
}

impl Words {
    fn new(rows: usize) -> Self {
        Self {
            text: String::new(),
            ends: Vec::with_capacity(rows),
            valid: Vec::with_capacity(rows),
        }
    }

    fn push(&mut self, column: &Column) -> Result<(), PolarsError> {
        for value in column.str()?.iter() {
            if let Some(value) = value {
                self.text.push_str(value);
            }
            self.ends.push(self.text.len());
            self.valid.push(value.is_some());
        }
        Ok(())
    }

    fn finish(mut self, name: PlSmallStr) -> Column {
        self.text.shrink_to_fit();
        let mut start = 0;
        let values: StringChunked = self
            .ends
            .iter()
            .zip(&self.valid)
            .map(|(end, valid)| {
                let value = valid.then(|| &self.text[start..*end]);
                start = *end;
                value
            })
            .collect();
        values.with_name(name).into_column()
    }
}

/// The primary scorer's evidence columns as they accumulate over the batches.
///
/// Evidence columns are String, Boolean or numbers; the first batch fixes their names,
/// types and order. Columns of words go to [`Words`], the others to [`Pieces`], so that
/// what is kept does not depend on the batch size: per log the values themselves (for
/// words up to three times their bytes while the text grows and while the column is
/// built), and per column at most [`EVIDENCE_COLUMN_BYTES`].
struct Evidence {
    rows: usize,
    /// The number of evidence columns the scorer's cost states.
    stated: u32,
    schema: Option<Vec<(PlSmallStr, DataType)>>,
    /// One entry per column of words, in column order.
    words: Vec<Words>,
    fixed: Pieces,
}

impl Evidence {
    fn new(rows: usize, stated: u32) -> Self {
        Self {
            rows,
            stated,
            schema: None,
            words: Vec::new(),
            fixed: Pieces::default(),
        }
    }

    /// Takes the evidence columns of one batch; the error says what is wrong with
    /// them.
    fn push(&mut self, batch: DataFrame, taken: &BTreeSet<String>) -> Result<(), String> {
        let schema: Vec<(PlSmallStr, DataType)> = batch
            .get_columns()
            .iter()
            .map(|column| (column.name().clone(), column.dtype().clone()))
            .collect();
        match &self.schema {
            Some(first) if *first == schema => {}
            Some(first) => {
                let describe = |schema: &[(PlSmallStr, DataType)]| {
                    schema
                        .iter()
                        .map(|(name, dtype)| format!("{name} {dtype}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                return Err(format!(
                    "its evidence columns changed from one batch to the next: [{}], then [{}]",
                    describe(first),
                    describe(&schema)
                ));
            }
            None => {
                if schema.len() as u64 > u64::from(self.stated) {
                    return Err(format!(
                        "it has {} evidence columns where its cost states {}",
                        schema.len(),
                        self.stated
                    ));
                }
                for (name, dtype) in &schema {
                    if taken.contains(name.as_str()) {
                        return Err(format!(
                            "its evidence column '{name}' has the name of a column of the \
                             triage frame"
                        ));
                    }
                    if !(*dtype == DataType::String
                        || dtype.is_bool()
                        || dtype.is_primitive_numeric())
                    {
                        return Err(format!(
                            "its evidence column '{name}' has the type {dtype}; evidence \
                             columns are String, Boolean or numbers"
                        ));
                    }
                }
                self.words = schema
                    .iter()
                    .filter(|(_, dtype)| *dtype == DataType::String)
                    .map(|_| Words::new(self.rows))
                    .collect();
                self.schema = Some(schema);
            }
        }
        let polars = |error: PolarsError| format!("reading its evidence columns failed: {error}");
        let mut words = self.words.iter_mut();
        let mut fixed = Vec::new();
        for column in batch.get_columns() {
            if *column.dtype() == DataType::String {
                if let Some(words) = words.next() {
                    words.push(column).map_err(polars)?;
                }
            } else {
                fixed.push(column.clone());
            }
        }
        if !fixed.is_empty() {
            let fixed = DataFrame::new(fixed).map_err(polars)?;
            self.fixed.push(fixed).map_err(polars)?;
        }
        Ok(())
    }

    /// The evidence columns of all batches, in the scorer's order, one piece each;
    /// none when no batch was scored.
    fn finish(self) -> Result<Vec<Column>, PolarsError> {
        let Some(schema) = self.schema else {
            return Ok(Vec::new());
        };
        let mut fixed = self
            .fixed
            .finish()?
            .map(DataFrame::take_columns)
            .unwrap_or_default()
            .into_iter();
        let mut words = self.words.into_iter();
        let mut columns = Vec::with_capacity(schema.len());
        for (name, dtype) in schema {
            let column = if dtype == DataType::String {
                words.next().map(|words| words.finish(name))
            } else {
                fixed.next()
            };
            columns.extend(column);
        }
        Ok(columns)
    }
}
/// Triage under a memory budget observed now. See [`triage_with`].
pub fn triage(
    logs: &LogsFrame,
    scorers: &mut [&mut dyn Scorer],
    settings: &TriageSettings<'_>,
) -> Result<TriageResult, TriageError> {
    let mut budget =
        MemoryBudget::observe(&MemorySettings::default()).map_err(TriageError::Memory)?;
    triage_with(logs, scorers, settings, &mut budget)
}

/// Triage of `logs` by `scorers` under `budget`: the module documentation says where
/// each log goes.
///
/// `logs` is not altered and nothing is written anywhere (requirement 6.9). The same
/// logs, scorers and settings give the same result, whatever the batch size
/// (requirements 11.10, 12.6).
///
/// - No scorer, two scorers of one kind, or a primary kind that is not among them is
///   [`TriageError::InvalidScorers`].
/// - A calibration made for another scorer than the primary one, or an error rate it
///   does not support, is [`TriageError::Calibration`].
/// - A memory limit that does not leave what triage holds or the smallest batch is
///   [`TriageError::Memory`].
/// - A scorer's failure is [`TriageError::Scorer`]; a scores frame that breaks the
///   contract ([`check_scores`]) is [`TriageError::Scores`].
///
/// The first three are found before any log is scored.
pub fn triage_with(
    logs: &LogsFrame,
    scorers: &mut [&mut dyn Scorer],
    settings: &TriageSettings<'_>,
    budget: &mut MemoryBudget,
) -> Result<TriageResult, TriageError> {
    let kinds: Vec<ScorerKind> = scorers.iter().map(|scorer| scorer.kind()).collect();
    let invalid = |detail: String| Err(TriageError::InvalidScorers { detail });
    if kinds.is_empty() {
        return invalid("triage needs at least one scorer".into());
    }
    if let Some(twice) = kinds
        .iter()
        .enumerate()
        .find(|(index, kind)| kinds[..*index].contains(kind))
        .map(|(_, kind)| kind)
    {
        return invalid(format!(
            "the {} scorer was given more than once",
            twice.as_str()
        ));
    }
    let Some(primary) = kinds.iter().position(|kind| *kind == settings.primary) else {
        return invalid(format!(
            "the primary scorer is the {} scorer, which is not among the scorers given ({})",
            settings.primary.as_str(),
            kinds
                .iter()
                .map(|kind| kind.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    };
    if let Some(calibration) = &settings.calibration {
        calibration
            .calibrator
            .require_scorer(settings.primary, scorers[primary].revision())
            .map_err(TriageError::Calibration)?;
        calibration
            .calibrator
            .require_error_rate(calibration.error_rate)
            .map_err(TriageError::Calibration)?;
    }

    // What triage holds beside its batches (module documentation).
    let rows = logs.len() as u64;
    let costs: Vec<ScoringCost> = scorers.iter().map(|scorer| scorer.cost(logs)).collect();
    let per_log = RESULT_ROW_BYTES
        .saturating_add(OTHER_SCORER_ROW_BYTES.saturating_mul(kinds.len() as u64 - 1))
        .saturating_add(costs[primary].evidence_row_bytes.saturating_mul(3));
    let result_bytes = rows.saturating_mul(per_log);
    let fixed_bytes = costs
        .iter()
        .fold(0_u64, |sum, cost| sum.saturating_add(cost.fixed_bytes))
        .saturating_add(
            EVIDENCE_COLUMN_BYTES.saturating_mul(u64::from(costs[primary].evidence_columns)),
        );
    let calibration_bytes = settings
        .calibration
        .map_or(0, |calibration| calibration.calibrator.held_bytes());
    let own = result_bytes
        .saturating_add(fixed_bytes)
        .saturating_add(calibration_bytes);
    let left = budget.left_bytes();
    if left < own {
        return Err(TriageError::Memory(L2dError::Refused {
            reason: format!(
                "triage holds {own} bytes beside its batches: {result_bytes} bytes for the \
                 result of {rows} logs ({per_log} bytes each), {fixed_bytes} bytes for its \
                 scorers and their evidence columns, and {calibration_bytes} bytes for the calibration; but {left} bytes \
                 are left of the memory limit of {} bytes; no log was scored",
                budget.limit_bytes()
            ),
        }));
    }
    let row_bytes = costs.iter().map(|cost| cost.row_bytes).max().unwrap_or(0);

    // What the caller holds already (its frames) stays counted while triage runs.
    let held = budget.held_bytes();
    budget.hold(held.saturating_add(own));
    let stated = costs[primary].evidence_columns;
    let scored = score_in_batches(logs, scorers, primary, settings, budget, row_bytes, stated);
    budget.hold(held);
    let (kept, plan) = scored?;

    assemble(logs, kept, plan, &kinds, primary, settings, own)
}

/// Scores `logs` batch by batch and keeps, per log, what the ranking needs.
fn score_in_batches(
    logs: &LogsFrame,
    scorers: &mut [&mut dyn Scorer],
    primary: usize,
    settings: &TriageSettings<'_>,
    budget: &mut MemoryBudget,
    row_bytes: u64,
    evidence_columns: u32,
) -> Result<(Kept, BatchPlan), TriageError> {
    let rows = logs.len();
    let mut plan = budget
        .plan(rows as u64, row_bytes)
        .map_err(TriageError::Memory)?;
    let mut kept = Kept {
        ratio: Vec::with_capacity(rows),
        verdict: Vec::with_capacity(rows),
        judged: Vec::with_capacity(if settings.calibration.is_some() {
            rows
        } else {
            0
        }),
        others: (1..scorers.len())
            .map(|_| Vec::with_capacity(rows))
            .collect(),
        evidence: Evidence::new(rows, evidence_columns),
    };
    // The names an evidence column must not take.
    let mut taken: BTreeSet<String> = LogsFrame::schema()
        .iter_names()
        .map(|name| name.to_string())
        .chain(TRIAGE_COLUMNS.iter().map(|name| (*name).to_owned()))
        .collect();
    for (index, scorer) in scorers.iter().enumerate() {
        if index != primary {
            taken.insert(format!("{TRIAGE_RATIO_PREFIX}{}", scorer.kind().as_str()));
        }
    }

    let mut done = 0_u64;
    while done < rows as u64 {
        let batch_rows = budget
            .next_batch_rows(&mut plan, done)
            .map_err(TriageError::Memory)?
            .min(rows as u64 - done);
        let batch = LogsFrame::new(logs.frame().slice(done as i64, batch_rows as usize))
            .map_err(|error| internal("taking a batch of the logs", error))?;
        let mut other = 0;
        for (index, scorer) in scorers.iter_mut().enumerate() {
            let kind = scorer.kind();
            let unusable = |detail: String| TriageError::Scores {
                scorer: kind.as_str(),
                detail,
            };
            let scores = scorer.score(&batch).map_err(TriageError::Scorer)?;
            check_scores(&scores, &batch, kind).map_err(unusable)?;
            let polars = |error: PolarsError| internal("reading a scores frame", error);
            let ratios = scores
                .column(SCORES_RATIO)
                .and_then(Column::f32)
                .map_err(polars)?;
            let abstained = scores
                .column(SCORES_ABSTAINED)
                .and_then(Column::bool)
                .map_err(polars)?;
            // Adding zero turns a negative zero into zero: the two are one ratio.
            let ratios = ratios
                .iter()
                .zip(abstained.iter())
                .map(|(ratio, abstained)| match (ratio, abstained) {
                    (Some(ratio), Some(false)) => Some(ratio + 0.0),
                    _ => None,
                });
            if index != primary {
                kept.others[other].extend(ratios);
                other += 1;
                continue;
            }
            let groups = batch
                .frame()
                .column(LogsFrame::GROUP)
                .and_then(Column::str)
                .map_err(polars)?;
            for (ratio, group) in ratios.zip(groups.iter()) {
                let (verdict, judged) = match (ratio, &settings.calibration) {
                    (None, _) => (Verdict::ScorerAbstained, None),
                    (Some(_), None) => (Verdict::Ranked, None),
                    (Some(ratio), Some(calibration)) => {
                        let decision = calibration
                            .calibrator
                            .predict(ratio, group, calibration.error_rate)
                            .map_err(TriageError::Calibration)?;
                        let verdict = match decision.set {
                            LabelSet::Bad => Verdict::Ranked,
                            LabelSet::Good => Verdict::Good,
                            LabelSet::Both => Verdict::Both,
                            LabelSet::Neither => Verdict::Neither,
                        };
                        let judged = Judged {
                            p_good: decision.p_good,
                            p_bad: decision.p_bad,
                            good_own: matches!(decision.good_stratum, Stratum::Group(_)),
                            bad_own: matches!(decision.bad_stratum, Stratum::Group(_)),
                        };
                        (verdict, Some(judged))
                    }
                };
                kept.ratio.push(ratio.unwrap_or(0.0));
                kept.verdict.push(verdict);
                if settings.calibration.is_some() {
                    kept.judged.push(judged);
                }
            }
            let evidence = scores.drop_many(SCORES_COLUMNS);
            kept.evidence.push(evidence, &taken).map_err(unusable)?;
        }
        done += batch_rows;
    }
    Ok((kept, plan))
}

/// Ranks the logs, applies the budget and builds the result.
fn assemble(
    logs: &LogsFrame,
    kept: Kept,
    plan: BatchPlan,
    kinds: &[ScorerKind],
    primary: usize,
    settings: &TriageSettings<'_>,
    held_bytes: u64,
) -> Result<TriageResult, TriageError> {
    let rows = logs.len();
    let Kept {
        ratio,
        verdict,
        judged,
        others,
        evidence,
    } = kept;
    let polars = |error: PolarsError| internal("building the triage frame", error);
    if ratio.len() != rows || verdict.len() != rows || IdxSize::try_from(rows).is_err() {
        return Err(internal(
            "building the triage frame",
            format!("{} of {rows} logs were scored", ratio.len()),
        ));
    }

    // By ratio, highest first; equal ratios in the order of the logs frame.
    let by_ratio = |a: &IdxSize, b: &IdxSize| {
        ratio[*b as usize]
            .total_cmp(&ratio[*a as usize])
            .then(a.cmp(b))
    };
    let of = |wanted: &[Verdict]| -> Vec<IdxSize> {
        (0..rows as IdxSize)
            .filter(|row| wanted.contains(&verdict[*row as usize]))
            .collect()
    };
    let mut ranked = of(&[Verdict::Ranked]);
    ranked.sort_unstable_by(by_ratio);
    let in_review = settings.review_budget.map_or(ranked.len(), |budget| {
        usize::try_from(budget).map_or(ranked.len(), |budget| budget.min(ranked.len()))
    });
    let mut rank: Vec<Option<u32>> = vec![None; rows];
    for (place, row) in ranked.iter().enumerate() {
        rank[*row as usize] = Some(place as u32 + 1);
    }
    let mut by_calibration = of(&[Verdict::Good]);
    by_calibration.sort_unstable_by(by_ratio);
    let abstained = of(&[Verdict::ScorerAbstained, Verdict::Both, Verdict::Neither]);

    let count = |wanted: Verdict| verdict.iter().filter(|found| **found == wanted).count() as u64;
    let counts = TriageCounts {
        logs: rows as u64,
        review: in_review as u64,
        scored_down_by_calibration: by_calibration.len() as u64,
        scored_down_by_budget: (ranked.len() - in_review) as u64,
        scorer_abstained: count(Verdict::ScorerAbstained),
        uncertain: count(Verdict::Both),
        unlike_calibration: count(Verdict::Neither),
    };
    let calibration = settings.calibration.map(|calibration| {
        let mut used = CalibrationUse {
            revision: calibration.calibrator.revision().to_owned(),
            error_rate: calibration.error_rate,
            note: EXCHANGEABILITY_NOTE,
            held_bytes: calibration.calibrator.held_bytes(),
            good: ListUse::default(),
            bad: ListUse::default(),
            set_good: counts.scored_down_by_calibration,
            set_bad: ranked.len() as u64,
            set_both: counts.uncertain,
            set_neither: counts.unlike_calibration,
        };
        for judged in judged.iter().flatten() {
            let (good, bad) = (&mut used.good, &mut used.bad);
            *(if judged.good_own {
                &mut good.group
            } else {
                &mut good.pooled
            }) += 1;
            *(if judged.bad_own {
                &mut bad.group
            } else {
                &mut bad.pooled
            }) += 1;
        }
        used
    });

    // The triage frame, column by column. The logs frame's columns share its buffers.
    let within = |row: usize| rank[row].is_some_and(|rank| rank as usize <= in_review);
    let list = |row: usize| match verdict[row] {
        Verdict::Ranked if within(row) => TriageList::Review,
        Verdict::Ranked | Verdict::Good => TriageList::ScoredDown,
        Verdict::Both | Verdict::Neither | Verdict::ScorerAbstained => TriageList::Abstained,
    };
    let scored_down_by = |row: usize| match verdict[row] {
        Verdict::Ranked if !within(row) => Some(ScoredDownBy::BudgetOnly.as_str()),
        Verdict::Good => Some(ScoredDownBy::Calibration.as_str()),
        _ => None,
    };
    let abstain_reason = |row: usize| match verdict[row] {
        Verdict::ScorerAbstained => Some(AbstainReason::ScorerAbstained.as_str()),
        Verdict::Both => Some(AbstainReason::Uncertain.as_str()),
        Verdict::Neither => Some(AbstainReason::UnlikeCalibration.as_str()),
        Verdict::Ranked | Verdict::Good => None,
    };
    let judged_of = |row: usize| judged.get(row).copied().flatten();
    let label_set = |row: usize| {
        judged_of(row).map(|_| {
            match verdict[row] {
                Verdict::Good => LabelSet::Good,
                Verdict::Both => LabelSet::Both,
                Verdict::Neither => LabelSet::Neither,
                // A judged log that is ranked has the set bad; an abstention is not
                // judged.
                Verdict::Ranked | Verdict::ScorerAbstained => LabelSet::Bad,
            }
            .as_str()
        })
    };
    let stratum = |own: bool| if own { STRATUM_GROUP } else { STRATUM_POOLED };
    let scored = |row: usize| verdict[row] != Verdict::ScorerAbstained;
    let all = 0..rows;
    let text = |name: &str, value: &dyn Fn(usize) -> Option<&'static str>| -> Column {
        Column::new(name.into(), all.clone().map(value).collect::<Vec<_>>())
    };
    let number = |name: &str, value: &dyn Fn(usize) -> Option<f32>| -> Column {
        Column::new(name.into(), all.clone().map(value).collect::<Vec<_>>())
    };

    let mut columns: Vec<Column> = logs.frame().get_columns().to_vec();
    columns.push(text(TRIAGE_SCORER, &|_| Some(kinds[primary].as_str())));
    columns.push(number(TRIAGE_RATIO, &|row| {
        scored(row).then_some(ratio[row])
    }));
    columns.push(Column::new(
        TRIAGE_ABSTAINED.into(),
        all.clone().map(|row| !scored(row)).collect::<Vec<_>>(),
    ));
    columns.push(text(TRIAGE_LIST, &|row| Some(list(row).as_str())));
    columns.push(Column::new(TRIAGE_RANK.into(), rank.clone()));
    columns.push(text(TRIAGE_SCORED_DOWN_BY, &scored_down_by));
    columns.push(text(TRIAGE_ABSTAIN_REASON, &abstain_reason));
    columns.push(text(TRIAGE_LABEL_SET, &label_set));
    columns.push(number(TRIAGE_P_GOOD, &|row| {
        judged_of(row).map(|judged| judged.p_good)
    }));
    columns.push(number(TRIAGE_P_BAD, &|row| {
        judged_of(row).map(|judged| judged.p_bad)
    }));
    columns.push(text(TRIAGE_GOOD_STRATUM, &|row| {
        judged_of(row).map(|judged| stratum(judged.good_own))
    }));
    columns.push(text(TRIAGE_BAD_STRATUM, &|row| {
        judged_of(row).map(|judged| stratum(judged.bad_own))
    }));
    let mut others = others.into_iter();
    for (index, kind) in kinds.iter().enumerate() {
        if index == primary {
            continue;
        }
        let ratios = others.next().unwrap_or_default();
        if ratios.len() != rows {
            return Err(internal(
                "building the triage frame",
                format!(
                    "the {} scorer scored {} of {rows} logs",
                    kind.as_str(),
                    ratios.len()
                ),
            ));
        }
        columns.push(Column::new(
            format!("{TRIAGE_RATIO_PREFIX}{}", kind.as_str()).into(),
            ratios,
        ));
    }
    // One piece per column, as the logs frame's columns are.
    columns.extend(evidence.finish().map_err(polars)?);
    let frame = DataFrame::new(columns).map_err(polars)?;

    let mut scored_down = ranked.split_off(in_review);
    scored_down.extend(by_calibration);
    Ok(TriageResult {
        frame,
        order: [ranked, scored_down, abstained],
        primary: kinds[primary],
        scorers: kinds.to_vec(),
        review_budget: settings.review_budget,
        counts,
        calibration,
        plan,
        held_bytes,
    })
}
