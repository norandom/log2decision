//! The held-out-group report of a score fitted from a mark set (requirement 16.9; task
//! 1.23).
//!
//! A score must find the same kind of log on hosts and endpoints it was not fitted on
//! (Requirement 16). The report measures that on the fitted logs themselves: it fits the
//! score again, with the same settings and redaction rules, on part of them, and compares
//! the ratio quality on **groups held out of fitting** with the quality on **groups that
//! were fitted on**. When the held-out figure is lower by more than the margin (default
//! 0.05 ROC AUC, the owner's decision of 2026-11-15, to be checked by gate 2.18) it warns
//! that the score depends on the groups it has seen.
//!
//! # Folds
//!
//! The group key is the one the fitted logs carry (`l2d_baseline::markscore`: one
//! default per kind of log, CSIC the URL path, AIT the host, settable), named in the
//! report. With `k` folds:
//!
//! - a log's **group fold** is [`hash_fold`] of its group key ([`Folds::hashed`]), so a
//!   group is held out by one fold whatever the other groups and the order of the logs;
//! - a log's **text part** is [`hash_fold`] of its text.
//!
//! Fold `f` fits on the logs whose group fold and text part are both other than `f`.
//! Its **held-out** logs are those of group fold `f` (groups it never saw); its
//! **fitted-group** logs are those of the other group folds whose text part is `f`
//! (groups it saw, logs it did not). One model judges both, so the two figures differ
//! only in whether the group was seen. Over the folds every log is held out once, and
//! every log whose text part is not its group fold is a fitted-group log once. The
//! figures are pooled over the folds as one ranking each (as the gate figure of task
//! 1.12 is); each fold's figures are reported beside them. A log without a group key is
//! a group of its own, as in the evaluation of task 1.12.
//!
//! Before a model scores a log, it is asked whether it was fitted on that text; a yes
//! ends the report with an error. Nothing is random; the same fit and settings give the
//! same report.
//!
//! # Memory
//!
//! The report holds a copy of the fitted logs, and per fold the logs of its fitting side
//! twice (records and logs frame); each fit checks its estimate with that held
//! (`BaselineScorer::fit_redacted_with`). The budget holds afterwards what it held
//! before.

use std::fmt;

use l2d_baseline::{BaselineScorer, FitError, MarkFit};
use l2d_core::{L2dError, LogRecord, LogsFrame, MemoryBudget};

use crate::evaluate::{Block, EvalLogs, rows_bytes};
use crate::metrics::{average_precision, roc_auc};
use crate::split::{Folds, hash_fold, units};

/// The margin of 16.9: held-out ROC AUC lower than fitted-group ROC AUC by more than
/// this is warned about.
pub const DEFAULT_MARGIN: f64 = 0.05;
/// Folds of the report.
pub const DEFAULT_FOLDS: usize = 5;
/// The rule of the folds, as the report names it.
pub const FOLD_RULE: &str = "group fold: the first 8 bytes of the SHA-256 of the group key, \
     little-endian, modulo the folds; text part: the same of the text";

/// The settings of the report.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutSettings {
    /// At least 2.
    pub folds: usize,
    /// In ROC AUC; finite, 0 to 1.
    pub margin: f64,
}

impl Default for HeldOutSettings {
    fn default() -> Self {
        Self {
            folds: DEFAULT_FOLDS,
            margin: DEFAULT_MARGIN,
        }
    }
}

impl HeldOutSettings {
    fn validate(&self) -> Result<(), L2dError> {
        if self.folds < 2 {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "the held-out-group report needs at least 2 folds, not {}",
                    self.folds
                ),
            });
        }
        if !self.margin.is_finite() || !(0.0..=1.0).contains(&self.margin) {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "the margin of the held-out-group report is a ROC AUC difference from 0 \
                     to 1, not {}",
                    self.margin
                ),
            });
        }
        Ok(())
    }
}

/// One fold of the report.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutFold {
    pub fold: usize,
    /// The group keys the fold holds out, and the SHA-256 of their list (as
    /// `Folds::membership_sha256`).
    pub held_out_groups: u64,
    pub held_out_groups_sha256: String,
    /// The logs the fold's model was fitted on; 0 when it was not fitted.
    pub fitted_on: u64,
    /// The fold's logs of groups held out of its fit.
    pub held_out: Block,
    /// The fold's logs of groups its fit saw, which it was not fitted on.
    pub fitted: Block,
    /// Why the fold has no model (too few logs of a label on its fitting side, or no
    /// group hashes to it); its logs are then in neither pooled figure.
    pub not_fitted: Option<String>,
}

/// The held-out-group report (16.9).
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutReport {
    /// The group key, as named by the fit.
    pub group_key: String,
    /// Distinct group keys among the fitted logs.
    pub groups: u64,
    /// Fitted logs the report works on (logs without a label are left out).
    pub logs: u64,
    pub fold_rule: &'static str,
    pub folds: Vec<HeldOutFold>,
    /// All held-out logs of the folds, as one ranking.
    pub held_out: Block,
    /// All fitted-group logs of the folds, as one ranking.
    pub fitted: Block,
    /// Fitted-group ROC AUC less held-out ROC AUC.
    pub difference: Option<f64>,
    pub margin: f64,
    /// Set when the difference is above the margin.
    pub warning: Option<String>,
}

fn block(ratios: &[f32], bad: &[bool]) -> Block {
    let bad_count = bad.iter().filter(|bad| **bad).count() as u64;
    let logs = bad.len() as u64;
    let both = bad_count > 0 && bad_count < logs;
    Block {
        logs,
        good: logs - bad_count,
        bad: bad_count,
        roc_auc: roc_auc(ratios, bad),
        average_precision: both.then(|| average_precision(ratios, bad)).flatten(),
    }
}

/// The ratios and labels of one side of the report.
#[derive(Default)]
struct Side {
    ratios: Vec<f32>,
    bad: Vec<bool>,
}

impl Side {
    fn push(&mut self, ratio: f32, bad: bool) {
        self.ratios.push(ratio);
        self.bad.push(bad);
    }

    fn block(&self) -> Block {
        block(&self.ratios, &self.bad)
    }

    fn extend(&mut self, other: &Side) {
        self.ratios.extend_from_slice(&other.ratios);
        self.bad.extend_from_slice(&other.bad);
    }
}

/// The held-out-group report of the score `fit` (module documentation). `None` when no
/// fitted log has a group key (16.9 applies where group keys exist).
///
/// - [`L2dError::InvalidInput`]: settings out of range; fewer groups than folds.
/// - [`L2dError::Refused`]: a fold's fit does not fit the memory limit (with the
///   figures).
/// - [`L2dError::Runtime`]: a model was asked to score a text it was fitted on.
pub fn held_out_report(
    fit: &MarkFit,
    settings: &HeldOutSettings,
    budget: &mut MemoryBudget,
) -> Result<Option<HeldOutReport>, L2dError> {
    settings.validate()?;
    if fit.logs.records().all(|record| record.group.is_none()) {
        return Ok(None);
    }
    let before = budget.held_bytes();
    let report = report(fit, settings, budget, before);
    budget.hold(before);
    report.map(Some)
}

fn report(
    fit: &MarkFit,
    settings: &HeldOutSettings,
    budget: &mut MemoryBudget,
    before: u64,
) -> Result<HeldOutReport, L2dError> {
    let data = EvalLogs::new(&fit.logs);
    let data_bytes = data.held_bytes();
    budget.hold(before.saturating_add(data_bytes));
    let found = units(data.groups(), &[])?;
    let groups = found.len() as u64;
    let folds = Folds::hashed(&found, settings.folds)?;
    let group_fold: Vec<usize> = data
        .groups()
        .iter()
        .map(|group| folds.fold_of(group).expect("every group key is in a fold"))
        .collect();
    let text_part: Vec<usize> = data
        .records()
        .iter()
        .map(|record| hash_fold(&record.text, settings.folds))
        .collect();

    let mut held_all = Side::default();
    let mut fitted_all = Side::default();
    let mut fold_results = Vec::with_capacity(folds.len());
    for fold in 0..folds.len() {
        let held_groups = folds.held_out(fold).len() as u64;
        let fitting: Vec<usize> = (0..data.len())
            .filter(|&row| group_fold[row] != fold && text_part[row] != fold)
            .collect();
        let mut result = HeldOutFold {
            fold,
            held_out_groups: held_groups,
            held_out_groups_sha256: folds.membership_sha256(fold),
            fitted_on: 0,
            held_out: block(&[], &[]),
            fitted: block(&[], &[]),
            not_fitted: None,
        };
        if held_groups == 0 {
            result.not_fitted = Some("no group key hashes to this fold".into());
            fold_results.push(result);
            continue;
        }
        let records: Vec<LogRecord> = fitting
            .iter()
            .map(|&row| data.records()[row].clone())
            .collect();
        let groups_of: Vec<String> = fitting
            .iter()
            .map(|&row| data.groups()[row].clone())
            .collect();
        let side_bytes = 2 * rows_bytes(&records, &groups_of);
        drop(groups_of);
        budget.hold(before.saturating_add(data_bytes).saturating_add(side_bytes));
        let frame = LogsFrame::from_records(&records)?;
        drop(records);
        let scorer = match BaselineScorer::fit_redacted_with(
            &frame,
            &[],
            fit.scorer.settings(),
            fit.scorer.rules(),
            budget,
        ) {
            Ok(scorer) => scorer,
            Err(error @ FitError::TooFewLabels { .. }) => {
                result.not_fitted = Some(error.to_string());
                fold_results.push(result);
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        drop(frame);
        result.fitted_on = scorer.fit_report().rows;

        let mut held = Side::default();
        let mut seen = Side::default();
        for row in 0..data.len() {
            let unseen = group_fold[row] == fold;
            if !unseen && text_part[row] != fold {
                continue;
            }
            let text = &data.records()[row].text;
            if scorer.was_fitted_on_text(text) != Some(false) {
                return Err(L2dError::Runtime {
                    detail: format!(
                        "the held-out-group report leaks: the log first seen at position {} \
                         is to be scored by a model of fold {fold} that was fitted on its text",
                        data.records()[row].first_position
                    ),
                });
            }
            let ratio = scorer.ratio(text);
            if unseen {
                held.push(ratio, data.bad()[row]);
            } else {
                seen.push(ratio, data.bad()[row]);
            }
        }
        result.held_out = held.block();
        result.fitted = seen.block();
        held_all.extend(&held);
        fitted_all.extend(&seen);
        fold_results.push(result);
    }

    let held_out = held_all.block();
    let fitted = fitted_all.block();
    let difference = match (fitted.roc_auc, held_out.roc_auc) {
        (Some(fitted), Some(held)) => Some(fitted - held),
        _ => None,
    };
    let group_key = fit.group_key.name();
    let warning = difference
        .filter(|difference| *difference > settings.margin)
        .map(|difference| {
            format!(
                "the score depends on the groups it has seen: on groups held out of fitting \
                 its ROC AUC is {:.4}, on the groups it was fitted on {:.4}, lower by {:.4}, \
                 more than the margin of {} (group key: {group_key})",
                held_out.roc_auc.unwrap_or(f64::NAN),
                fitted.roc_auc.unwrap_or(f64::NAN),
                difference,
                settings.margin
            )
        });
    Ok(HeldOutReport {
        group_key,
        groups,
        logs: data.len() as u64,
        fold_rule: FOLD_RULE,
        folds: fold_results,
        held_out,
        fitted,
        difference,
        margin: settings.margin,
        warning,
    })
}

fn figure(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.4}"))
}

impl fmt::Display for HeldOutReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "held-out-group report (requirement 16.9)")?;
        writeln!(f, "group key: {}", self.group_key)?;
        writeln!(
            f,
            "{} logs in {} groups, {} folds ({})",
            self.logs,
            self.groups,
            self.folds.len(),
            self.fold_rule
        )?;
        writeln!(
            f,
            "fold | held-out groups | fitted on | held-out logs (bad) ROC AUC | fitted-group logs (bad) ROC AUC"
        )?;
        for fold in &self.folds {
            writeln!(
                f,
                "{} | {} | {} | {} ({}) {} | {} ({}) {}{}",
                fold.fold,
                fold.held_out_groups,
                fold.fitted_on,
                fold.held_out.logs,
                fold.held_out.bad,
                figure(fold.held_out.roc_auc),
                fold.fitted.logs,
                fold.fitted.bad,
                figure(fold.fitted.roc_auc),
                fold.not_fitted
                    .as_ref()
                    .map_or_else(String::new, |why| format!(" (not fitted: {why})"))
            )?;
        }
        writeln!(
            f,
            "pooled: held out {} logs ({} bad) ROC AUC {}; fitted groups {} logs ({} bad) ROC AUC {}",
            self.held_out.logs,
            self.held_out.bad,
            figure(self.held_out.roc_auc),
            self.fitted.logs,
            self.fitted.bad,
            figure(self.fitted.roc_auc)
        )?;
        writeln!(
            f,
            "difference (fitted less held out): {}; margin {}",
            figure(self.difference),
            self.margin
        )?;
        match &self.warning {
            Some(warning) => writeln!(f, "WARNING: {warning}"),
            None => writeln!(
                f,
                "no warning: held out is not lower by more than the margin"
            ),
        }
    }
}
