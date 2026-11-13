//! Grouped evaluation of the baseline scorer (requirements 9.1, 9.2, 9.8, 11.8).
//!
//! # What a fold does
//!
//! A fold is a mask over the logs: the logs of its held-out group keys, and the rest,
//! its fitting side. [`evaluate_fold`] does, in this order:
//!
//! 1. **The choice of the penalty**, on the fitting side alone. The fitting side is cut
//!    into [`EvalSettings::inner_folds`] inner folds by group key
//!    ([`Folds::balanced`]); for every penalty of the grid and every inner fold a scorer
//!    is fitted on the fitting side less that inner fold and scores it. The penalty
//!    with the smallest log loss over all inner held-out logs is chosen (ties: the
//!    smaller penalty). The penalty with the highest inner ROC AUC is recorded beside
//!    it, as a second rule that reads the fitting side only; which of the two a default
//!    should follow is not decided here.
//! 2. **The fits on the whole fitting side**, one per penalty of the grid, each scoring
//!    the held-out side. The default penalty is one of them, the chosen one another (or
//!    the same); the others are the sensitivity table.
//! 3. **The calibration**, for the default and for the chosen penalty. The calibration
//!    logs are the first inner fold, and the scorer that is calibrated is the one of
//!    step 1 that was fitted on the fitting side less that inner fold, so it has seen
//!    neither the calibration logs nor their group keys. That scorer, not the one of
//!    step 2, then judges the held-out logs.
//!
//! # How leakage is kept out
//!
//! - The logs are distinct texts (the input adapters deduplicate before anything is
//!   split), and a split is by group key: [`evaluate_fold`] refuses a mask that puts
//!   one group key on both sides.
//! - Steps 1 and 3 are functions of a `Side`, a copy of the fitting-side rows. They
//!   are not given the held-out rows, so they cannot read them.
//! - Before a scorer scores a log, the evaluation asks it whether it was fitted on that
//!   text (the per-log check of task 1.11, `BaselineScorer::was_fitted_on_text`). A yes
//!   ends the evaluation with an error.
//! - Nothing is fitted, scaled or chosen on all logs. The feature rows are hashed per
//!   text and normalized per row; there is no vocabulary and no statistic over rows.
//! - The labels of the held-out side are read by [`summarize`] and by the coverage
//!   count of the calibration, and by nothing else. Which group keys hold both labels,
//!   the subset the cross-endpoint figure is computed on, is read from all labels: it
//!   decides which logs a metric is computed on, never a fit.
//!
//! # Memory (requirement 12)
//!
//! Every fit asks the fold's [`MemoryBudget`] (`BaselineScorer::fit_with`, which calls
//! `require_whole`), and so does every calibration. Beside its fits a fold holds copies
//! of its rows and the scorers of the calibration, and says so to the budget before the
//! first fit. [`evaluate`] gives each worker thread a budget of its own, with an equal
//! share of what the caller's budget leaves, so the shares add up to the caller's limit.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use l2d_baseline::{BaselineScorer, BaselineSettings, RoutingMode, Termination};
use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::{L2dError, Label, LogRecord, LogsFrame, MemoryBudget, MemorySettings};
use l2d_decide::conformal::{CalibrationPoint, DEFAULT_ERROR_RATE, points_frame};
use l2d_decide::{
    CalibrationError, CalibrationSettings, ConformalCalibrator, LabelSet, ScorerKind,
};

use crate::metrics::{
    Spread, average_precision, budget_shares, log_loss_sum, roc_auc, share_sum, spread,
};
use crate::split::{Folds, units};

/// The grid of L2 penalties of the science gate's linear head and of this evaluation.
pub const DEFAULT_PENALTIES: [f64; 5] = [0.01, 0.1, 1.0, 10.0, 100.0];
/// Inner folds of the fitting side for the choice of the penalty.
pub const DEFAULT_INNER_FOLDS: usize = 4;
/// The review budget as a share of the held-out logs.
pub const DEFAULT_BUDGET_SHARE: f64 = 0.10;
/// The cross-endpoint ROC AUC the baseline must reach (task 1.12).
pub const GATE_ROC_AUC: f64 = 0.90;
/// An endpoint with fewer good or fewer bad logs than this is reported as small.
pub const DEFAULT_SMALL_ENDPOINT: u64 = 100;

/// The settings of an evaluation in one routing mode.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalSettings {
    /// The scorer's settings. Its `l2` is the default penalty.
    pub baseline: BaselineSettings,
    /// The grid of penalties, each once; holds the default penalty.
    pub penalties: Vec<f64>,
    /// Inner folds of the fitting side: at least 2, or 0 for no choice of the penalty
    /// and no calibration (the default penalty and the grid only).
    pub inner_folds: usize,
    /// The chosen error rate of the calibrated decisions.
    pub error_rate: f32,
    pub calibration: CalibrationSettings,
    /// The review budget as a share of the held-out logs of a fold: above 0, at most 1.
    pub budget_share: f64,
    /// An endpoint with fewer good or fewer bad logs is left out of the mean over the
    /// large endpoints.
    pub small_endpoint: u64,
}

/// The labelled logs of an evaluation, with a group key for every log.
#[derive(Debug, Clone)]
pub struct EvalLogs {
    records: Vec<LogRecord>,
    groups: Vec<String>,
    bad: Vec<bool>,
    unknown_left_out: u64,
    counts: BTreeMap<String, (u64, u64)>,
}

/// Start of the group key of a log that came without one.
const NO_GROUP: &str = "\u{0}no group key\u{0}";

impl EvalLogs {
    /// The logs of `logs` labelled good or bad, in frame order. A log labelled unknown
    /// (a text that occurred with both labels) is left out and counted. A log without
    /// a group key is its own group.
    pub fn new(logs: &LogsFrame) -> Self {
        let mut found = Self {
            records: Vec::new(),
            groups: Vec::new(),
            bad: Vec::new(),
            unknown_left_out: 0,
            counts: BTreeMap::new(),
        };
        for record in logs.records() {
            let bad = match record.label {
                Label::Good => false,
                Label::Bad => true,
                Label::Unknown => {
                    found.unknown_left_out += 1;
                    continue;
                }
            };
            let group = record
                .group
                .clone()
                .unwrap_or_else(|| format!("{NO_GROUP}{}", record.text));
            let count = found.counts.entry(group.clone()).or_default();
            if bad {
                count.1 += 1;
            } else {
                count.0 += 1;
            }
            found.groups.push(group);
            found.bad.push(bad);
            found.records.push(record);
        }
        found
    }

    /// The number of logs.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The logs, in order.
    pub fn records(&self) -> &[LogRecord] {
        &self.records
    }

    /// The group key of every log.
    pub fn groups(&self) -> &[String] {
        &self.groups
    }

    /// Whether each log is bad.
    pub fn bad(&self) -> &[bool] {
        &self.bad
    }

    /// Logs of the frame that were left out because they are labelled unknown.
    pub fn unknown_left_out(&self) -> u64 {
        self.unknown_left_out
    }

    /// Per group key: the number of good and of bad logs.
    pub fn group_counts(&self) -> &BTreeMap<String, (u64, u64)> {
        &self.counts
    }

    /// The group keys that hold both labels: the endpoints the cross-endpoint figures
    /// are computed on (requirement 9.8).
    pub fn mixed_groups(&self) -> BTreeSet<String> {
        self.counts
            .iter()
            .filter(|(_, (good, bad))| *good > 0 && *bad > 0)
            .map(|(group, _)| group.clone())
            .collect()
    }

    /// What these logs take in memory, in bytes (texts, group keys, 96 bytes per log).
    pub fn held_bytes(&self) -> u64 {
        rows_bytes(&self.records, &self.groups)
    }
}

pub(crate) fn rows_bytes(records: &[LogRecord], groups: &[String]) -> u64 {
    records
        .iter()
        .zip(groups)
        .map(|(record, group)| (record.text.len() + 2 * group.len() + 96) as u64)
        .sum()
}

/// Counts and ranking figures of a set of logs.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub logs: u64,
    pub good: u64,
    pub bad: u64,
    /// `None` unless the set holds both labels.
    pub roc_auc: Option<f64>,
    /// `None` unless the set holds both labels.
    pub average_precision: Option<f64>,
}

/// The logs of the group keys that hold a single label.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SingleLabel {
    pub groups: u64,
    pub logs: u64,
    pub good: u64,
    pub bad: u64,
}

/// What lies within one review budget of a held-out side, ranked by ratio as a whole.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetFigures {
    /// `share` (a share of the held-out logs) or `bad_count` (as many logs as the side
    /// holds bad logs, the budget of the science gate's records).
    pub name: String,
    /// The budget in logs.
    pub budget: u64,
    /// The share of the bad logs within the budget.
    pub recall: Option<f64>,
    /// The share of the bad logs of the endpoints with both labels within the budget.
    pub cross_endpoint_recall: Option<f64>,
    /// The share of the logs of single-label groups within the budget.
    pub single_label_share: Option<f64>,
    pub single_label_bad_share: Option<f64>,
    pub single_label_good_share: Option<f64>,
}

/// The figures of one held-out side under one scorer.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldMetrics {
    /// All held-out logs.
    pub overall: Block,
    /// The held-out logs of the group keys that hold both labels, pooled.
    pub cross_endpoint: Block,
    /// Each of those group keys on its own.
    pub per_endpoint: BTreeMap<String, Block>,
    pub single_label: SingleLabel,
    /// The budgets `share` and `bad_count`, in this order.
    pub budgets: Vec<BudgetFigures>,
}

fn block(bad: &[bool], ratios: &[f32], rows: &[usize]) -> Block {
    let chosen_bad: Vec<bool> = rows.iter().map(|&row| bad[row]).collect();
    let chosen_ratios: Vec<f32> = rows.iter().map(|&row| ratios[row]).collect();
    let bad_count = chosen_bad.iter().filter(|bad| **bad).count() as u64;
    let logs = rows.len() as u64;
    let both = bad_count > 0 && bad_count < logs;
    Block {
        logs,
        good: logs - bad_count,
        bad: bad_count,
        roc_auc: roc_auc(&chosen_ratios, &chosen_bad),
        average_precision: both
            .then(|| average_precision(&chosen_ratios, &chosen_bad))
            .flatten(),
    }
}

/// The figures of a held-out side: `groups`, `bad` and `ratios` give the group key, the
/// label and the ratio of each of its logs.
///
/// - `overall`: all logs.
/// - `cross_endpoint`: the logs whose group key is in `mixed` (the group keys that hold
///   both labels in the whole dataset), pooled; `per_endpoint`: each of them alone.
/// - `single_label`: the other logs, counted; their place in the ranking is in
///   `budgets`.
/// - `budgets`: the whole side ranked by ratio. `share`: `budget_share` of its logs,
///   rounded down. `bad_count`: as many logs as it holds bad logs. A tie at the budget
///   is shared ([`crate::metrics::budget_shares`]).
///
/// This is the only place the labels of a held-out side are read for a ranking figure.
pub fn summarize<S: AsRef<str>>(
    groups: &[S],
    bad: &[bool],
    ratios: &[f32],
    mixed: &BTreeSet<String>,
    budget_share: f64,
) -> HeldMetrics {
    assert!(
        groups.len() == bad.len() && bad.len() == ratios.len(),
        "a group key, a label and a ratio for every log"
    );
    let all: Vec<usize> = (0..groups.len()).collect();
    let in_mixed: Vec<bool> = groups
        .iter()
        .map(|group| mixed.contains(group.as_ref()))
        .collect();
    let cross: Vec<usize> = all.iter().copied().filter(|&row| in_mixed[row]).collect();
    let mut by_endpoint: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for &row in &cross {
        by_endpoint
            .entry(groups[row].as_ref())
            .or_default()
            .push(row);
    }
    let single_groups: BTreeSet<&str> = all
        .iter()
        .filter(|&&row| !in_mixed[row])
        .map(|&row| groups[row].as_ref())
        .collect();
    let single_logs = (all.len() - cross.len()) as u64;
    let single_bad = all
        .iter()
        .filter(|&&row| !in_mixed[row] && bad[row])
        .count() as u64;

    let bad_count = bad.iter().filter(|bad| **bad).count() as u64;
    let share_budget = (budget_share * groups.len() as f64 + 1e-9).floor() as u64;
    let budgets = [("share", share_budget), ("bad_count", bad_count)]
        .into_iter()
        .map(|(name, budget)| {
            let shares = budget_shares(ratios, budget);
            let of = |chosen: &dyn Fn(usize) -> bool| {
                let count = all.iter().filter(|&&row| chosen(row)).count();
                (count > 0)
                    .then(|| share_sum(&shares, all.iter().map(|&row| chosen(row))) / count as f64)
            };
            BudgetFigures {
                name: name.to_string(),
                budget,
                recall: of(&|row| bad[row]),
                cross_endpoint_recall: of(&|row| bad[row] && in_mixed[row]),
                single_label_share: of(&|row| !in_mixed[row]),
                single_label_bad_share: of(&|row| !in_mixed[row] && bad[row]),
                single_label_good_share: of(&|row| !in_mixed[row] && !bad[row]),
            }
        })
        .collect();

    HeldMetrics {
        overall: block(bad, ratios, &all),
        cross_endpoint: block(bad, ratios, &cross),
        per_endpoint: by_endpoint
            .into_iter()
            .map(|(group, rows)| (group.to_string(), block(bad, ratios, &rows)))
            .collect(),
        single_label: SingleLabel {
            groups: single_groups.len() as u64,
            logs: single_logs,
            good: single_logs - single_bad,
            bad: single_bad,
        },
        budgets,
    }
}

/// How a fit ended. Everything here repeats exactly on a rerun.
#[derive(Debug, Clone, PartialEq)]
pub struct FitNote {
    pub penalty: f64,
    pub rows: u64,
    pub good: u64,
    pub bad: u64,
    pub iterations: u64,
    /// `converged`, `iteration_cap` or `line_search_stopped`.
    pub termination: String,
    /// Whether the gradient criterion was met.
    pub gradient_converged: bool,
    pub no_feature_rows: u64,
    /// What the fit asked the memory budget for.
    pub estimated_bytes: u64,
}

/// One penalty of the grid on the inner folds of a fitting side.
#[derive(Debug, Clone, PartialEq)]
pub struct InnerScore {
    pub penalty: f64,
    /// Mean log loss over the inner held-out logs of all inner folds.
    pub log_loss: f64,
    /// Mean over the inner folds of the ROC AUC on the inner held-out logs of the group
    /// keys that hold both labels on the fitting side. Reported, not used for the
    /// choice.
    pub roc_auc: Option<f64>,
    /// The fit of each inner fold.
    pub fits: Vec<FitNote>,
}

/// The choice of the penalty on a fitting side.
#[derive(Debug, Clone, PartialEq)]
pub struct InnerChoice {
    /// The group keys each inner fold holds out.
    pub folds: Vec<BTreeSet<String>>,
    /// One entry per penalty of the grid, in its order.
    pub scores: Vec<InnerScore>,
    /// The penalty with the smallest `log_loss`; the smaller penalty on a tie.
    pub chosen: f64,
    /// The penalty with the highest `roc_auc`, for comparison; the smaller on a tie.
    pub chosen_by_roc_auc: Option<f64>,
}

/// Calibrated decisions on a set of logs, counted. Index 0 is good, 1 is bad.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Coverage {
    /// Logs per true label.
    pub logs: [u64; 2],
    /// Logs per true label whose label set does not hold the true label.
    pub missed: [u64; 2],
    /// Logs per label set: good, bad, both, neither.
    pub sets: [u64; 4],
}

impl Coverage {
    pub(crate) fn add(&mut self, bad: bool, set: LabelSet) {
        let label = if bad { Label::Bad } else { Label::Good };
        let index = usize::from(bad);
        self.logs[index] += 1;
        self.missed[index] += u64::from(!set.contains(label));
        self.sets[match set {
            LabelSet::Good => 0,
            LabelSet::Bad => 1,
            LabelSet::Both => 2,
            LabelSet::Neither => 3,
        }] += 1;
    }

    pub(crate) fn merge(&mut self, other: &Coverage) {
        for index in 0..2 {
            self.logs[index] += other.logs[index];
            self.missed[index] += other.missed[index];
        }
        for index in 0..4 {
            self.sets[index] += other.sets[index];
        }
    }

    /// The achieved error rate of a label (0 good, 1 bad): the share of its logs whose
    /// label set misses it. `None` without a log of that label.
    pub fn error_rate(&self, label: usize) -> Option<f64> {
        (self.logs[label] > 0).then(|| self.missed[label] as f64 / self.logs[label] as f64)
    }
}

/// The calibration of one fold at one penalty and its decisions on the held-out side.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationResult {
    pub penalty: f64,
    /// The fit of the scorer that was calibrated: the fitting side less the calibration
    /// logs.
    pub fit: FitNote,
    /// The group keys of the calibration logs: the first inner fold.
    pub calibration_groups: BTreeSet<String>,
    /// Calibration logs: good, bad.
    pub calibration_logs: [u64; 2],
    /// Lists of the calibration (pooled and per group).
    pub lists: u64,
    /// The smallest error rate the calibration supports.
    pub smallest_supported: Option<f32>,
    /// Why there are no decisions: the calibration could not be made, or it does not
    /// support the chosen error rate.
    pub refused: Option<String>,
    /// All held-out logs.
    pub overall: Coverage,
    /// The held-out logs of the group keys that hold both labels.
    pub cross_endpoint: Coverage,
    /// The held-out logs of the single-label group keys.
    pub single_label: Coverage,
    /// Each held-out group key that holds both labels.
    pub per_group: BTreeMap<String, Coverage>,
    /// The label set of every held-out log, in the order of `FoldResult::held_rows`;
    /// empty when refused.
    pub label_sets: Vec<LabelSet>,
}

/// One penalty of the grid, fitted on the whole fitting side.
#[derive(Debug, Clone, PartialEq)]
pub struct PenaltyResult {
    pub penalty: f64,
    pub fit: FitNote,
    /// The ratio of every held-out log, in the order of `FoldResult::held_rows`.
    pub ratios: Vec<f32>,
    pub metrics: HeldMetrics,
}

/// Time a fold took. Not part of a report: it does not repeat.
#[derive(Debug, Clone, Default)]
pub struct FoldTiming {
    pub fits: u64,
    pub fit_seconds: f64,
    pub score_seconds: f64,
    pub total_seconds: f64,
}

/// One fold of an evaluation in one routing mode.
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// The held-out logs: row numbers of the [`EvalLogs`], ascending.
    pub held_rows: Vec<u32>,
    pub fitting_logs: u64,
    /// `None` when the settings ask for no inner folds.
    pub inner: Option<InnerChoice>,
    /// One entry per penalty of the grid, in its order.
    pub by_penalty: Vec<PenaltyResult>,
    /// The calibration at the default penalty and, when it differs, at the chosen one.
    pub calibration: Vec<CalibrationResult>,
    /// The default penalty of the settings.
    pub default: f64,
    pub timing: FoldTiming,
}

impl FoldResult {
    /// The result at `penalty`.
    pub fn at(&self, penalty: f64) -> Option<&PenaltyResult> {
        self.by_penalty
            .iter()
            .find(|result| result.penalty == penalty)
    }

    /// The result at the default penalty.
    pub fn default_penalty(&self) -> Option<&PenaltyResult> {
        self.at(self.default)
    }

    /// The result at the penalty chosen inside the fold; `None` without inner folds.
    pub fn chosen(&self) -> Option<&PenaltyResult> {
        self.inner.as_ref().and_then(|inner| self.at(inner.chosen))
    }

    /// The result at the penalty with the highest inner ROC AUC: a second rule that
    /// reads the fitting side only, reported beside [`FoldResult::chosen`]. `None`
    /// without inner folds, or when no inner fold had both labels to rank.
    pub fn chosen_by_roc_auc(&self) -> Option<&PenaltyResult> {
        self.inner
            .as_ref()
            .and_then(|inner| inner.chosen_by_roc_auc)
            .and_then(|penalty| self.at(penalty))
    }
}

/// A way of cutting the logs into folds.
#[derive(Debug, Clone)]
pub struct Arrangement {
    pub name: String,
    /// The rule in words, for the report.
    pub rule: String,
    pub folds: Folds,
    /// Sets of group keys that stay on one side, in the inner folds too.
    pub together: Vec<Vec<String>>,
}

/// The folds of an arrangement in one routing mode.
#[derive(Debug, Clone)]
pub struct ModeResult {
    pub folds: Vec<FoldResult>,
}

/// An arrangement evaluated in every routing mode.
#[derive(Debug, Clone)]
pub struct ArrangementResult {
    pub modes: Vec<ModeResult>,
}

/// The figures of one penalty rule over the folds of an arrangement.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleSummary {
    /// The penalty used in each fold.
    pub penalties: Vec<Option<f64>>,
    /// Per fold: ROC AUC over its held-out logs of group keys with both labels.
    pub fold_roc_auc: Vec<Option<f64>>,
    pub fold_average_precision: Vec<Option<f64>>,
    /// Per fold: ROC AUC over all its held-out logs.
    pub fold_overall_roc_auc: Vec<Option<f64>>,
    /// Mean and spread of `fold_roc_auc` over the folds where it is defined.
    pub roc_auc: Option<Spread>,
    pub average_precision: Option<Spread>,
    pub overall_roc_auc: Option<Spread>,
    /// The held-out logs of group keys with both labels of all folds as one ranking.
    pub pooled: Block,
    /// All held-out logs of all folds as one ranking.
    pub pooled_overall: Block,
    /// Each group key with both labels, from the fold that held it out.
    pub per_endpoint: BTreeMap<String, Block>,
    /// Mean and spread of the ROC AUC over those group keys, each counting once.
    pub endpoint_roc_auc: Option<Spread>,
    /// The same over the group keys with at least `small_endpoint` good and bad logs.
    pub large_endpoint_roc_auc: Option<Spread>,
    /// The single-label group keys of all folds.
    pub single_label: SingleLabel,
    pub budgets: Vec<BudgetSummary>,
}

/// One review budget over the folds of an arrangement.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetSummary {
    pub name: String,
    /// Per fold: the budget in logs.
    pub fold_budget: Vec<u64>,
    /// Mean and spread of the recall over the folds.
    pub recall: Option<Spread>,
    /// The share of the single-label logs of all folds within their fold's budget.
    pub single_label_share: Option<f64>,
    pub single_label_bad_share: Option<f64>,
    pub single_label_good_share: Option<f64>,
}

/// The calibrated decisions of one penalty rule over the folds of an arrangement.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CalibrationSummary {
    /// Folds with decisions.
    pub folds: u64,
    /// Folds without, each with the reason.
    pub refused: Vec<(usize, String)>,
    /// All held-out logs of all folds.
    pub overall: Coverage,
    pub cross_endpoint: Coverage,
    pub single_label: Coverage,
    /// Each group key that holds both labels, from the fold that held it out.
    pub per_group: BTreeMap<String, Coverage>,
    /// Per fold: all its held-out logs; `None` when refused.
    pub fold_overall: Vec<Option<Coverage>>,
    /// Per fold: calibration logs (good, bad) and lists.
    pub fold_calibration_logs: Vec<[u64; 2]>,
    pub fold_lists: Vec<u64>,
}

fn invalid<T>(detail: String) -> Result<T, L2dError> {
    Err(L2dError::InvalidInput { detail })
}

impl EvalSettings {
    /// The settings of task 1.12 for one routing mode: the baseline's defaults (L2
    /// penalty 1.0), the grid [`DEFAULT_PENALTIES`], [`DEFAULT_INNER_FOLDS`] inner
    /// folds, the default error rate and calibration settings, and
    /// [`DEFAULT_BUDGET_SHARE`].
    pub fn new(routing: RoutingMode) -> Self {
        Self {
            baseline: BaselineSettings {
                routing,
                ..BaselineSettings::default()
            },
            penalties: DEFAULT_PENALTIES.to_vec(),
            inner_folds: DEFAULT_INNER_FOLDS,
            error_rate: DEFAULT_ERROR_RATE,
            calibration: CalibrationSettings::default(),
            budget_share: DEFAULT_BUDGET_SHARE,
            small_endpoint: DEFAULT_SMALL_ENDPOINT,
        }
    }

    fn validate(&self) -> Result<(), L2dError> {
        if self.penalties.is_empty()
            || self
                .penalties
                .iter()
                .any(|penalty| !(penalty.is_finite() && *penalty > 0.0))
        {
            return invalid(format!(
                "the grid of penalties must hold numbers above 0; got {:?}",
                self.penalties
            ));
        }
        for (index, penalty) in self.penalties.iter().enumerate() {
            if self.penalties[..index].contains(penalty) {
                return invalid(format!("the penalty {penalty} is in the grid twice"));
            }
        }
        if !self.penalties.contains(&self.baseline.l2) {
            return invalid(format!(
                "the default penalty {} is not in the grid {:?}",
                self.baseline.l2, self.penalties
            ));
        }
        if self.inner_folds == 1 {
            return invalid(
                "the choice of the penalty needs at least 2 inner folds (or 0 for none)".into(),
            );
        }
        if !(self.budget_share > 0.0 && self.budget_share <= 1.0) {
            return invalid(format!(
                "the review budget is a share of the held-out logs above 0 and at most 1; got {}",
                self.budget_share
            ));
        }
        if !(self.error_rate > 0.0 && self.error_rate < 1.0) {
            return invalid(format!(
                "the error rate must be above 0 and below 1; got {}",
                self.error_rate
            ));
        }
        Ok(())
    }
}

/// A copy of some logs: one side of a split, or a part of one. A function that is given
/// a side can read these logs and no others.
struct Side {
    records: Vec<LogRecord>,
    groups: Vec<String>,
    bad: Vec<bool>,
    frame: LogsFrame,
}

impl Side {
    fn new(records: Vec<LogRecord>, groups: Vec<String>, bad: Vec<bool>) -> Result<Self, L2dError> {
        let frame = LogsFrame::from_records(&records)?;
        Ok(Self {
            records,
            groups,
            bad,
            frame,
        })
    }

    /// The logs of `data` whose row number `keep` accepts, in order.
    fn of(data: &EvalLogs, keep: impl Fn(usize) -> bool) -> Result<Self, L2dError> {
        let rows: Vec<usize> = (0..data.len()).filter(|&row| keep(row)).collect();
        Self::new(
            rows.iter().map(|&row| data.records[row].clone()).collect(),
            rows.iter().map(|&row| data.groups[row].clone()).collect(),
            rows.iter().map(|&row| data.bad[row]).collect(),
        )
    }

    /// The logs of this side whose entry in `mask` equals `keep`, in order.
    fn part(&self, mask: &[bool], keep: bool) -> Result<Self, L2dError> {
        let rows: Vec<usize> = (0..self.len()).filter(|&row| mask[row] == keep).collect();
        Self::new(
            rows.iter().map(|&row| self.records[row].clone()).collect(),
            rows.iter().map(|&row| self.groups[row].clone()).collect(),
            rows.iter().map(|&row| self.bad[row]).collect(),
        )
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    /// Records and frame: the texts exist twice.
    fn bytes(&self) -> u64 {
        2 * rows_bytes(&self.records, &self.groups)
    }

    /// The group keys of this side that hold both labels on this side.
    fn mixed_groups(&self) -> BTreeSet<&str> {
        let mut seen: BTreeMap<&str, (bool, bool)> = BTreeMap::new();
        for (group, bad) in self.groups.iter().zip(&self.bad) {
            let entry = seen.entry(group.as_str()).or_default();
            if *bad {
                entry.1 = true;
            } else {
                entry.0 = true;
            }
        }
        seen.into_iter()
            .filter(|(_, (good, bad))| *good && *bad)
            .map(|(group, _)| group)
            .collect()
    }
}

/// Fits the baseline scorer on `side` with the L2 penalty `penalty`, under `budget`.
fn fit_scorer(
    side: &Side,
    settings: &EvalSettings,
    penalty: f64,
    budget: &MemoryBudget,
    timing: &mut FoldTiming,
) -> Result<(BaselineScorer, FitNote), L2dError> {
    let baseline = BaselineSettings {
        l2: penalty,
        ..settings.baseline
    };
    let started = Instant::now();
    // No input summary: the folds of one file all carry its hash, so the check of
    // whole inputs says nothing here. The scorer keeps the hash of every text it was
    // fitted on, and `score` below asks it per log.
    let scorer = BaselineScorer::fit_with(&side.frame, &[], &baseline, budget)?;
    timing.fits += 1;
    timing.fit_seconds += started.elapsed().as_secs_f64();
    let report = scorer.fit_report();
    let note = FitNote {
        penalty,
        rows: report.rows,
        good: report.good,
        bad: report.bad,
        iterations: report.iterations,
        termination: match report.termination {
            Termination::Converged => "converged",
            Termination::IterationCap => "iteration_cap",
            Termination::LineSearchStopped => "line_search_stopped",
        }
        .to_string(),
        gradient_converged: scorer.gradient_converged(),
        no_feature_rows: report.no_feature_rows,
        estimated_bytes: report.estimated_bytes,
    };
    Ok((scorer, note))
}

/// The ratio `scorer` gives each log of `side`, after the check that it was fitted on
/// none of them (requirement 11.1 and the leakage rule of the evaluation).
fn score(
    scorer: &BaselineScorer,
    side: &Side,
    timing: &mut FoldTiming,
) -> Result<Vec<f32>, L2dError> {
    let started = Instant::now();
    for record in &side.records {
        if scorer.was_fitted_on_text(&record.text) != Some(false) {
            return Err(L2dError::Runtime {
                detail: format!(
                    "the split leaks: the log first seen at position {} of the input is to be \
                     scored by a scorer that was fitted on its text (or that cannot say which \
                     texts it was fitted on)",
                    record.first_position
                ),
            });
        }
    }
    let failed = |error: &dyn std::fmt::Display| L2dError::Runtime {
        detail: format!("reading the scores frame failed: {error}"),
    };
    let scores = scorer.score(&side.frame)?;
    let column = scores
        .column("ratio")
        .and_then(Column::f32)
        .map_err(|error| failed(&error))?;
    let mut ratios = Vec::with_capacity(side.len());
    for ratio in column.iter() {
        match ratio {
            Some(ratio) if (0.0..=1.0).contains(&ratio) => ratios.push(ratio),
            other => {
                return Err(failed(&format!(
                    "a ratio is {other:?}, not a number from 0 to 1"
                )));
            }
        }
    }
    if ratios.len() != side.len() {
        return Err(failed(&format!(
            "{} ratios for {} logs",
            ratios.len(),
            side.len()
        )));
    }
    timing.score_seconds += started.elapsed().as_secs_f64();
    Ok(ratios)
}

/// A scorer of the first inner fold, kept for the calibration.
struct Seed {
    penalty: f64,
    scorer: BaselineScorer,
    fit: FitNote,
    /// Its ratio for every calibration log.
    ratios: Vec<f32>,
}

/// The calibration logs of a fold and the scorers that have not seen them.
struct Seeds {
    groups: BTreeSet<String>,
    logs: Side,
    by_penalty: Vec<Seed>,
}

/// Step 1 of a fold: the choice of the penalty on the fitting side. It is given the
/// fitting side and nothing else.
fn choose_penalty(
    fitting: &Side,
    together: &[Vec<String>],
    settings: &EvalSettings,
    budget: &MemoryBudget,
    timing: &mut FoldTiming,
) -> Result<(InnerChoice, Seeds), L2dError> {
    let folds = Folds::balanced(&units(&fitting.groups, together)?, settings.inner_folds)?;
    let mixed = fitting.mixed_groups();
    let grid = settings.penalties.len();
    let mut loss = vec![0.0_f64; grid];
    let mut areas: Vec<Vec<f64>> = vec![Vec::new(); grid];
    let mut fits: Vec<Vec<FitNote>> = vec![Vec::new(); grid];
    let mut seeds = None;
    for inner in 0..folds.len() {
        let mask = folds.mask(inner, &fitting.groups);
        let fitted = fitting.part(&mask, false)?;
        let scored = fitting.part(&mask, true)?;
        let in_mixed: Vec<usize> = (0..scored.len())
            .filter(|&row| mixed.contains(scored.groups[row].as_str()))
            .collect();
        let mut kept = Vec::new();
        for (index, &penalty) in settings.penalties.iter().enumerate() {
            let (scorer, fit) = fit_scorer(&fitted, settings, penalty, budget, timing).map_err(
                |error| match error {
                    L2dError::Refused { reason } => L2dError::Refused {
                        reason: format!(
                            "inner fold {inner} of the choice of the penalty: {reason}"
                        ),
                    },
                    other => other,
                },
            )?;
            let ratios = score(&scorer, &scored, timing)?;
            loss[index] += log_loss_sum(&ratios, &scored.bad);
            if let Some(area) = block(&scored.bad, &ratios, &in_mixed).roc_auc {
                areas[index].push(area);
            }
            fits[index].push(fit.clone());
            if inner == 0 {
                kept.push(Seed {
                    penalty,
                    scorer,
                    fit,
                    ratios,
                });
            }
        }
        if inner == 0 {
            seeds = Some(Seeds {
                groups: folds.held_out(0).clone(),
                logs: scored,
                by_penalty: kept,
            });
        }
    }
    let scores: Vec<InnerScore> = settings
        .penalties
        .iter()
        .zip(loss)
        .zip(areas)
        .zip(fits)
        .map(|(((&penalty, loss), areas), fits)| InnerScore {
            penalty,
            log_loss: loss / fitting.len() as f64,
            roc_auc: spread(&areas).map(|spread| spread.mean),
            fits,
        })
        .collect();
    let chosen = scores
        .iter()
        .min_by(|a, b| {
            a.log_loss
                .total_cmp(&b.log_loss)
                .then_with(|| a.penalty.total_cmp(&b.penalty))
        })
        .expect("the grid is not empty")
        .penalty;
    let chosen_by_roc_auc = scores
        .iter()
        .filter_map(|score| score.roc_auc.map(|area| (area, score.penalty)))
        .max_by(|a, b| a.0.total_cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)))
        .map(|(_, penalty)| penalty);
    Ok((
        InnerChoice {
            folds: (0..folds.len())
                .map(|inner| folds.held_out(inner).clone())
                .collect(),
            scores,
            chosen,
            chosen_by_roc_auc,
        },
        seeds.expect("there are at least 2 inner folds"),
    ))
}

/// Step 3 of a fold: the scorer of `penalty` that has not seen the calibration logs is
/// calibrated on them and judges the held-out logs (requirement 11.8).
fn calibrate(
    seeds: &Seeds,
    penalty: f64,
    held: &Side,
    mixed: &BTreeSet<String>,
    settings: &EvalSettings,
    budget: &MemoryBudget,
    timing: &mut FoldTiming,
) -> Result<CalibrationResult, L2dError> {
    let seed = seeds
        .by_penalty
        .iter()
        .find(|seed| seed.penalty == penalty)
        .expect("a scorer was kept for every penalty of the grid");
    let bad = seeds.logs.bad.iter().filter(|bad| **bad).count() as u64;
    let mut result = CalibrationResult {
        penalty,
        fit: seed.fit.clone(),
        calibration_groups: seeds.groups.clone(),
        calibration_logs: [seeds.logs.len() as u64 - bad, bad],
        lists: 0,
        smallest_supported: None,
        refused: None,
        overall: Coverage::default(),
        cross_endpoint: Coverage::default(),
        single_label: Coverage::default(),
        per_group: BTreeMap::new(),
        label_sets: Vec::new(),
    };
    let points: Vec<CalibrationPoint> = seeds
        .logs
        .records
        .iter()
        .zip(&seed.ratios)
        .map(|(record, &ratio)| CalibrationPoint {
            ratio,
            label: record.label,
            group: record.group.clone(),
        })
        .collect();
    let points = points_frame(&points)?;
    let calibrator = match ConformalCalibrator::fit_with(
        &points,
        ScorerKind::Baseline,
        seed.scorer.revision(),
        &settings.calibration,
        budget,
    ) {
        Ok(calibrator) => calibrator,
        Err(CalibrationError::Memory(refusal)) => return Err(refusal),
        Err(error) => {
            result.refused = Some(format!("no calibration: {error}"));
            return Ok(result);
        }
    };
    result.lists = calibrator.strata().len() as u64;
    result.smallest_supported = Some(calibrator.smallest_supported());
    if let Err(error) = calibrator.require_error_rate(settings.error_rate) {
        result.refused = Some(format!(
            "the error rate {} is not supported: {error}",
            settings.error_rate
        ));
        return Ok(result);
    }
    let ratios = score(&seed.scorer, held, timing)?;
    for ((record, &ratio), (group, &bad)) in held
        .records
        .iter()
        .zip(&ratios)
        .zip(held.groups.iter().zip(&held.bad))
    {
        let decision = calibrator.predict(ratio, record.group.as_deref(), settings.error_rate)?;
        result.overall.add(bad, decision.set);
        if mixed.contains(group) {
            result.cross_endpoint.add(bad, decision.set);
            result
                .per_group
                .entry(group.clone())
                .or_default()
                .add(bad, decision.set);
        } else {
            result.single_label.add(bad, decision.set);
        }
        result.label_sets.push(decision.set);
    }
    Ok(result)
}

/// Evaluates one fold: the logs whose entry in `held_out` is true are its held-out
/// side, the others its fitting side. See the module documentation for the steps.
///
/// `together` are the sets of group keys that stay on one side in the inner folds.
/// `budget` is asked before every fit and every calibration, and is given back holding
/// what it held.
///
/// [`L2dError::InvalidInput`] for settings out of range, a mask of another length, a
/// side without logs, or a group key on both sides. [`L2dError::Refused`] when a fit
/// does not fit the memory limit or a side has too few logs of a label for a fit.
/// [`L2dError::Runtime`] when a log would be scored by a scorer fitted on its text.
pub fn evaluate_fold(
    data: &EvalLogs,
    held_out: &[bool],
    together: &[Vec<String>],
    settings: &EvalSettings,
    budget: &mut MemoryBudget,
) -> Result<FoldResult, L2dError> {
    settings.validate()?;
    if held_out.len() != data.len() {
        return invalid(format!(
            "the mask of the held-out side has {} entries for {} logs",
            held_out.len(),
            data.len()
        ));
    }
    let held = held_out.iter().filter(|held| **held).count();
    if held == 0 || held == data.len() {
        return invalid(format!(
            "a fold needs logs on both sides; {held} of {} are held out",
            data.len()
        ));
    }
    let held_groups: BTreeSet<&str> = data
        .groups
        .iter()
        .zip(held_out)
        .filter(|(_, held)| **held)
        .map(|(group, _)| group.as_str())
        .collect();
    if let Some((group, _)) = data
        .groups
        .iter()
        .zip(held_out)
        .find(|(group, held)| !**held && held_groups.contains(group.as_str()))
    {
        return invalid(format!(
            "the group key '{group}' has logs on both sides of the split"
        ));
    }
    let before = budget.held_bytes();
    let result = run_fold(data, held_out, together, settings, budget, before);
    budget.hold(before);
    result
}

fn run_fold(
    data: &EvalLogs,
    held_out: &[bool],
    together: &[Vec<String>],
    settings: &EvalSettings,
    budget: &mut MemoryBudget,
    held_before: u64,
) -> Result<FoldResult, L2dError> {
    let started = Instant::now();
    let mut timing = FoldTiming::default();
    let fitting = Side::of(data, |row| !held_out[row])?;
    let held = Side::of(data, |row| held_out[row])?;
    // What the fold keeps beside a fit: its two sides, the two parts of an inner fold
    // (together one more fitting side), and one scorer per penalty for the calibration
    // (weights of at most twice the buckets, 16 bytes per fitted text).
    let scorers = settings.penalties.len() as u64
        * (16 * u64::from(settings.baseline.buckets) + 16 * fitting.len() as u64);
    budget.hold(
        held_before
            .saturating_add(2 * fitting.bytes())
            .saturating_add(held.bytes())
            .saturating_add(scorers),
    );
    let mixed = data.mixed_groups();

    // Step 1, on the fitting side alone, before the held-out side is scored at all.
    let choice = if settings.inner_folds >= 2 {
        Some(choose_penalty(
            &fitting,
            together,
            settings,
            budget,
            &mut timing,
        )?)
    } else {
        None
    };

    // Step 2: every penalty of the grid on the whole fitting side.
    let mut by_penalty = Vec::with_capacity(settings.penalties.len());
    for &penalty in &settings.penalties {
        let (scorer, fit) = fit_scorer(&fitting, settings, penalty, budget, &mut timing)?;
        let ratios = score(&scorer, &held, &mut timing)?;
        let metrics = summarize(
            &held.groups,
            &held.bad,
            &ratios,
            &mixed,
            settings.budget_share,
        );
        by_penalty.push(PenaltyResult {
            penalty,
            fit,
            ratios,
            metrics,
        });
    }

    // Step 3: the calibration at the default and at the chosen penalty.
    let default = settings.baseline.l2;
    let (inner, calibration) = match choice {
        Some((choice, seeds)) => {
            let mut calibration = vec![calibrate(
                &seeds,
                default,
                &held,
                &mixed,
                settings,
                budget,
                &mut timing,
            )?];
            if choice.chosen != default {
                calibration.push(calibrate(
                    &seeds,
                    choice.chosen,
                    &held,
                    &mixed,
                    settings,
                    budget,
                    &mut timing,
                )?);
            }
            (Some(choice), calibration)
        }
        None => (None, Vec::new()),
    };

    timing.total_seconds = started.elapsed().as_secs_f64();
    Ok(FoldResult {
        held_rows: (0..data.len() as u32)
            .filter(|&row| held_out[row as usize])
            .collect(),
        fitting_logs: fitting.len() as u64,
        inner,
        by_penalty,
        calibration,
        default,
        timing,
    })
}

/// Evaluates every fold of every arrangement in every mode, on `workers` threads.
///
/// The result has one entry per arrangement, in it one per mode, in it one per fold,
/// each in the order given. The number of workers changes no figure: a fold is
/// evaluated by one thread from start to end, and no fold reads another's result.
///
/// Memory: what `budget` leaves is divided equally between the workers, and each
/// worker's fits and calibrations are checked against its share. The caller's budget
/// should hold what the caller keeps (`EvalLogs::held_bytes` and the logs frame).
///
/// No worker is [`L2dError::InvalidInput`]; a share of nothing is
/// [`L2dError::Refused`]. The first fold that fails, in the order above, gives the
/// error.
pub fn evaluate(
    data: &EvalLogs,
    arrangements: &[Arrangement],
    modes: &[EvalSettings],
    workers: usize,
    budget: &MemoryBudget,
) -> Result<Vec<ArrangementResult>, L2dError> {
    evaluate_with_progress(data, arrangements, modes, workers, budget, &|_| {})
}

/// A fold of an evaluation that has just ended.
#[derive(Debug, Clone)]
pub struct Progress {
    /// Index of the arrangement, of the mode and of the fold.
    pub arrangement: usize,
    pub mode: usize,
    pub fold: usize,
    /// Folds ended so far, this one included, and folds in all.
    pub done: usize,
    pub total: usize,
    pub timing: FoldTiming,
}

/// [`evaluate`], calling `progress` from the worker threads after every fold that ends
/// well. `progress` changes no result.
pub fn evaluate_with_progress(
    data: &EvalLogs,
    arrangements: &[Arrangement],
    modes: &[EvalSettings],
    workers: usize,
    budget: &MemoryBudget,
    progress: &(dyn Fn(Progress) + Sync),
) -> Result<Vec<ArrangementResult>, L2dError> {
    if workers == 0 {
        return invalid("an evaluation needs at least one worker".into());
    }
    let mut tasks = Vec::new();
    for (arrangement, plan) in arrangements.iter().enumerate() {
        for mode in 0..modes.len() {
            for fold in 0..plan.folds.len() {
                tasks.push((arrangement, mode, fold));
            }
        }
    }
    let workers = workers.min(tasks.len().max(1));
    let share = budget.left_bytes() / workers as u64;
    if share == 0 {
        return Err(L2dError::Refused {
            reason: format!(
                "the memory limit of {} bytes leaves nothing for {workers} workers of the \
                 evaluation",
                budget.limit_bytes()
            ),
        });
    }
    let mut budgets = Vec::with_capacity(workers);
    for _ in 0..workers {
        budgets.push(MemoryBudget::observe(&MemorySettings {
            memory_limit_bytes: Some(share + PROCESS_BASELINE_BYTES),
            ..MemorySettings::default()
        })?);
    }

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let slots: Mutex<Vec<Option<Result<FoldResult, L2dError>>>> =
        Mutex::new((0..tasks.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for mut own in budgets {
            let (next, done, failed, slots, tasks) = (&next, &done, &failed, &slots, &tasks);
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::SeqCst);
                    if index >= tasks.len() || failed.load(Ordering::SeqCst) {
                        break;
                    }
                    let (arrangement, mode, fold) = tasks[index];
                    let plan = &arrangements[arrangement];
                    let mask = plan.folds.mask(fold, data.groups());
                    let result = evaluate_fold(data, &mask, &plan.together, &modes[mode], &mut own);
                    match &result {
                        Ok(result) => progress(Progress {
                            arrangement,
                            mode,
                            fold,
                            done: done.fetch_add(1, Ordering::SeqCst) + 1,
                            total: tasks.len(),
                            timing: result.timing.clone(),
                        }),
                        Err(_) => failed.store(true, Ordering::SeqCst),
                    }
                    slots
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())[index] = Some(result);
                }
            });
        }
    });

    let mut slots = slots
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(error) = slots.iter_mut().find_map(|slot| match slot {
        Some(Err(_)) => slot.take().and_then(Result::err),
        _ => None,
    }) {
        return Err(error);
    }
    let mut results = slots.into_iter();
    let mut found = Vec::with_capacity(arrangements.len());
    for plan in arrangements {
        let mut by_mode = Vec::with_capacity(modes.len());
        for _ in modes {
            let mut folds = Vec::with_capacity(plan.folds.len());
            for _ in 0..plan.folds.len() {
                match results.next() {
                    Some(Some(Ok(fold))) => folds.push(fold),
                    _ => {
                        return Err(L2dError::Runtime {
                            detail: "a fold of the evaluation has no result".into(),
                        });
                    }
                }
            }
            by_mode.push(ModeResult { folds });
        }
        found.push(ArrangementResult { modes: by_mode });
    }
    Ok(found)
}

fn defined(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().flatten().copied().collect()
}

/// The figures of one penalty rule over `folds`, the folds of an arrangement in one
/// mode. `pick` names the result of a fold the rule uses: `FoldResult::default_penalty`,
/// `FoldResult::chosen`, or a fixed penalty of the grid.
///
/// Three ways of pooling are stated side by side, because the choice matters:
///
/// - per fold (`fold_roc_auc`, with `roc_auc` as its mean and spread over the folds):
///   the ROC AUC over the fold's held-out logs of group keys with both labels. Its mean
///   is the gate figure of task 1.12 and the figure of the science gate.
/// - per endpoint (`per_endpoint`, `endpoint_roc_auc`): each group key with both labels
///   on its own; the mean counts every endpoint once, whatever its size.
/// - pooled over the folds (`pooled`): the held-out logs of all folds as one ranking.
///   Their ratios come from different fits, so a shift of the ratios between folds
///   counts against this figure.
pub fn rule_summary<'a>(
    data: &EvalLogs,
    folds: &'a [FoldResult],
    settings: &EvalSettings,
    pick: impl Fn(&'a FoldResult) -> Option<&'a PenaltyResult>,
) -> RuleSummary {
    let mixed = data.mixed_groups();
    let picked: Vec<Option<&PenaltyResult>> = folds.iter().map(pick).collect();
    let of = |figure: &dyn Fn(&PenaltyResult) -> Option<f64>| -> Vec<Option<f64>> {
        picked
            .iter()
            .map(|result| result.and_then(figure))
            .collect()
    };
    let fold_roc_auc = of(&|result| result.metrics.cross_endpoint.roc_auc);
    let fold_average_precision = of(&|result| result.metrics.cross_endpoint.average_precision);
    let fold_overall_roc_auc = of(&|result| result.metrics.overall.roc_auc);

    // All folds as one ranking.
    let (mut bad, mut ratios, mut cross) = (Vec::new(), Vec::new(), Vec::new());
    let mut per_endpoint = BTreeMap::new();
    let mut single_label = SingleLabel::default();
    for (fold, result) in folds.iter().zip(&picked) {
        let Some(result) = result else { continue };
        for (&row, &ratio) in fold.held_rows.iter().zip(&result.ratios) {
            if mixed.contains(&data.groups[row as usize]) {
                cross.push(bad.len());
            }
            bad.push(data.bad[row as usize]);
            ratios.push(ratio);
        }
        for (group, block) in &result.metrics.per_endpoint {
            per_endpoint
                .entry(group.clone())
                .or_insert_with(|| block.clone());
        }
        single_label.groups += result.metrics.single_label.groups;
        single_label.logs += result.metrics.single_label.logs;
        single_label.good += result.metrics.single_label.good;
        single_label.bad += result.metrics.single_label.bad;
    }
    let all: Vec<usize> = (0..bad.len()).collect();
    let endpoint_areas = |large: bool| -> Vec<f64> {
        per_endpoint
            .values()
            .filter(|block| {
                !large
                    || (block.good >= settings.small_endpoint
                        && block.bad >= settings.small_endpoint)
            })
            .filter_map(|block| block.roc_auc)
            .collect()
    };

    let budgets = (0..2)
        .map(|index| {
            let figures: Vec<&BudgetFigures> = picked
                .iter()
                .flatten()
                .map(|result| &result.metrics.budgets[index])
                .collect();
            // The share of the single-label logs of all folds: each fold's share
            // weighted by its number of such logs.
            let weighted = |share: &dyn Fn(&BudgetFigures) -> Option<f64>,
                            count: &dyn Fn(&SingleLabel) -> u64| {
                let (mut inside, mut logs) = (0.0_f64, 0_u64);
                for result in picked.iter().flatten() {
                    let count = count(&result.metrics.single_label);
                    if let Some(share) = share(&result.metrics.budgets[index]) {
                        inside += share * count as f64;
                        logs += count;
                    }
                }
                (logs > 0).then(|| inside / logs as f64)
            };
            let recalls: Vec<Option<f64>> = figures.iter().map(|figures| figures.recall).collect();
            BudgetSummary {
                name: figures
                    .first()
                    .map_or_else(String::new, |figures| figures.name.clone()),
                fold_budget: figures.iter().map(|figures| figures.budget).collect(),
                recall: spread(&defined(&recalls)),
                single_label_share: weighted(&|f| f.single_label_share, &|s| s.logs),
                single_label_bad_share: weighted(&|f| f.single_label_bad_share, &|s| s.bad),
                single_label_good_share: weighted(&|f| f.single_label_good_share, &|s| s.good),
            }
        })
        .collect();

    RuleSummary {
        penalties: picked
            .iter()
            .map(|result| result.map(|result| result.penalty))
            .collect(),
        roc_auc: spread(&defined(&fold_roc_auc)),
        average_precision: spread(&defined(&fold_average_precision)),
        overall_roc_auc: spread(&defined(&fold_overall_roc_auc)),
        fold_roc_auc,
        fold_average_precision,
        fold_overall_roc_auc,
        pooled: block(&bad, &ratios, &cross),
        pooled_overall: block(&bad, &ratios, &all),
        endpoint_roc_auc: spread(&endpoint_areas(false)),
        large_endpoint_roc_auc: spread(&endpoint_areas(true)),
        per_endpoint,
        single_label,
        budgets,
    }
}

/// The calibrated decisions over `folds`: of the calibration at the default penalty, or
/// with `chosen` of the one at the penalty chosen inside each fold (requirement 11.8).
pub fn calibration_summary(folds: &[FoldResult], chosen: bool) -> CalibrationSummary {
    let mut summary = CalibrationSummary::default();
    for (index, fold) in folds.iter().enumerate() {
        let penalty = match (&fold.inner, chosen) {
            (Some(inner), true) => inner.chosen,
            _ => fold.default,
        };
        let Some(calibration) = fold
            .calibration
            .iter()
            .find(|calibration| calibration.penalty == penalty)
        else {
            continue;
        };
        summary
            .fold_calibration_logs
            .push(calibration.calibration_logs);
        summary.fold_lists.push(calibration.lists);
        if let Some(reason) = &calibration.refused {
            summary.refused.push((index, reason.clone()));
            summary.fold_overall.push(None);
            continue;
        }
        summary.folds += 1;
        summary.overall.merge(&calibration.overall);
        summary.cross_endpoint.merge(&calibration.cross_endpoint);
        summary.single_label.merge(&calibration.single_label);
        for (group, coverage) in &calibration.per_group {
            summary
                .per_group
                .entry(group.clone())
                .or_default()
                .merge(coverage);
        }
        summary.fold_overall.push(Some(calibration.overall.clone()));
    }
    summary
}

/// The predictions frame of the design's "Data frames": one row per held-out log of
/// every fold, fold after fold, with the columns `text`, `label`, `group` (null for a
/// log without a group key), `fold` UInt32, `scorer`, `ratio` Float32 and `label_set`.
///
/// `pick` names the result of each fold, as for [`rule_summary`]; a fold without one is
/// left out. `label_set` is the set the fold's calibration at that penalty gave the log
/// and null where there is none; it was decided on the ratio of the scorer that was
/// calibrated, which is not the scorer of the `ratio` column (see the module
/// documentation).
pub fn predictions_frame<'a>(
    data: &EvalLogs,
    folds: &'a [FoldResult],
    pick: impl Fn(&'a FoldResult) -> Option<&'a PenaltyResult>,
) -> Result<DataFrame, L2dError> {
    let (mut text, mut label, mut group) = (Vec::new(), Vec::new(), Vec::new());
    let (mut number, mut ratio, mut label_set) = (Vec::new(), Vec::new(), Vec::new());
    for (index, fold) in folds.iter().enumerate() {
        let Some(result) = pick(fold) else { continue };
        let sets = fold
            .calibration
            .iter()
            .find(|calibration| calibration.penalty == result.penalty)
            .filter(|calibration| calibration.label_sets.len() == fold.held_rows.len());
        for (place, (&row, &value)) in fold.held_rows.iter().zip(&result.ratios).enumerate() {
            let record = &data.records[row as usize];
            text.push(record.text.as_str());
            label.push(record.label.as_str());
            group.push(record.group.as_deref());
            number.push(index as u32);
            ratio.push(value);
            label_set.push(sets.map(|calibration| calibration.label_sets[place].as_str()));
        }
    }
    let rows = text.len();
    DataFrame::new(vec![
        Column::new("text".into(), text),
        Column::new("label".into(), label),
        Column::new("group".into(), group),
        Column::new("fold".into(), number),
        Column::new("scorer".into(), vec![ScorerKind::Baseline.as_str(); rows]),
        Column::new("ratio".into(), ratio),
        Column::new("label_set".into(), label_set),
    ])
    .map_err(|error| L2dError::Runtime {
        detail: format!("building the predictions frame failed: {error}"),
    })
}
