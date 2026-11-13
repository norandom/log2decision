//! The evaluation as a JSON report and a readable summary (requirement 9.6).
//!
//! A report names the dataset and its hash, the settings, the date and the fold
//! membership, and holds every figure of [`crate::evaluate`]: per fold, per endpoint and
//! over the folds, for the default penalty, for the penalty chosen inside each fold, and
//! for every penalty of the grid.
//!
//! Nothing in a report depends on the clock beyond the date its caller gives, on the
//! number of workers or on the host's memory: two runs on the same logs with the same
//! settings write the same bytes. Times and memory are measurements of a run and belong
//! in a file of their own.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use l2d_baseline::RoutingMode;
use l2d_core::{InputSummary, L2dError};
use serde_json::{Map, Value, json};

use crate::evaluate::{
    Arrangement, ArrangementResult, Block, BudgetFigures, CalibrationResult, CalibrationSummary,
    Coverage, EvalLogs, EvalSettings, FitNote, FoldResult, GATE_ROC_AUC, HeldMetrics, InnerChoice,
    RuleSummary, SingleLabel, calibration_summary, rule_summary,
};
use crate::metrics::Spread;

/// The dataset of a report.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// A short name, as in the name of the report files.
    pub name: String,
    /// What the input adapter read: path, content hash, record and text counts.
    pub summary: InputSummary,
}

/// Everything a report is written from.
pub struct Report<'a> {
    /// The task or command the evaluation belongs to.
    pub task: String,
    /// The calendar day of the evaluation, `YYYY-MM-DD` ([`date_of`]).
    pub date: String,
    pub dataset: Dataset,
    pub data: &'a EvalLogs,
    pub arrangements: &'a [Arrangement],
    /// The settings of each routing mode, in the order of `ArrangementResult::modes`.
    pub modes: &'a [EvalSettings],
    pub results: &'a [ArrangementResult],
    /// Anything else the caller wants on record; it must repeat exactly on a rerun.
    pub notes: Value,
}

/// The calendar day of `time` in UTC, `YYYY-MM-DD`.
pub fn date_of(time: SystemTime) -> String {
    let days = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() / 86_400) as i64;
    // Days since 1970-01-01 to a date of the proleptic Gregorian calendar.
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// The name of a routing mode in reports.
pub fn routing_name(routing: RoutingMode) -> &'static str {
    match routing {
        RoutingMode::Keep => "keep",
        RoutingMode::Strip => "strip",
        RoutingMode::Isolate => "isolate",
    }
}

fn number(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

/// A 32-bit float as the decimal number it prints as (0.05, not 0.05000000074505806).
fn short(value: f32) -> Value {
    value
        .to_string()
        .parse::<f64>()
        .map_or(Value::Null, |value| json!(value))
}

fn spread_json(spread: &Option<Spread>) -> Value {
    match spread {
        Some(spread) => json!({
            "count": spread.count,
            "mean": spread.mean,
            "min": spread.min,
            "max": spread.max,
            "sd": spread.sd,
        }),
        None => Value::Null,
    }
}

fn block_json(block: &Block) -> Value {
    json!({
        "logs": block.logs,
        "good": block.good,
        "bad": block.bad,
        "roc_auc": number(block.roc_auc),
        "average_precision": number(block.average_precision),
    })
}

fn blocks_json<'b>(blocks: impl Iterator<Item = (&'b String, &'b Block)>) -> Value {
    Value::Object(
        blocks
            .map(|(group, block)| (group.clone(), block_json(block)))
            .collect::<Map<String, Value>>(),
    )
}

fn single_json(single: &SingleLabel) -> Value {
    json!({
        "groups": single.groups,
        "logs": single.logs,
        "good": single.good,
        "bad": single.bad,
    })
}

fn budget_json(figures: &BudgetFigures) -> Value {
    json!({
        "name": figures.name,
        "budget_logs": figures.budget,
        "recall": number(figures.recall),
        "cross_endpoint_recall": number(figures.cross_endpoint_recall),
        "single_label_share": number(figures.single_label_share),
        "single_label_bad_share": number(figures.single_label_bad_share),
        "single_label_good_share": number(figures.single_label_good_share),
    })
}

fn metrics_json(metrics: &HeldMetrics) -> Value {
    json!({
        "overall": block_json(&metrics.overall),
        "cross_endpoint": block_json(&metrics.cross_endpoint),
        "per_endpoint": blocks_json(metrics.per_endpoint.iter()),
        "single_label": single_json(&metrics.single_label),
        "budgets": metrics.budgets.iter().map(budget_json).collect::<Vec<_>>(),
    })
}

fn fit_json(fit: &FitNote) -> Value {
    json!({
        "penalty": fit.penalty,
        "rows": fit.rows,
        "good": fit.good,
        "bad": fit.bad,
        "iterations": fit.iterations,
        "termination": fit.termination,
        "gradient_converged": fit.gradient_converged,
        "rows_without_features": fit.no_feature_rows,
        "estimated_bytes": fit.estimated_bytes,
    })
}

fn label_json(coverage: &Coverage, label: usize) -> Value {
    json!({
        "logs": coverage.logs[label],
        "missed": coverage.missed[label],
        "achieved_error_rate": number(coverage.error_rate(label)),
    })
}

fn coverage_json(coverage: &Coverage) -> Value {
    let logs = coverage.logs[0] + coverage.logs[1];
    let share = |count: u64| number((logs > 0).then(|| count as f64 / logs as f64));
    json!({
        "good": label_json(coverage, 0),
        "bad": label_json(coverage, 1),
        "label_set_share": {
            "good": share(coverage.sets[0]),
            "bad": share(coverage.sets[1]),
            "both": share(coverage.sets[2]),
            "neither": share(coverage.sets[3]),
        },
    })
}

/// The members of `groups` that hold both labels in the whole dataset.
fn both_labels(groups: &BTreeSet<String>, mixed: &BTreeSet<String>) -> Vec<String> {
    groups.intersection(mixed).cloned().collect()
}

fn inner_json(inner: &InnerChoice, mixed: &BTreeSet<String>) -> Value {
    json!({
        "folds": inner.folds.iter().map(|fold| json!({
            "groups": fold.len(),
            "groups_with_both_labels": both_labels(fold, mixed),
        })).collect::<Vec<_>>(),
        "scores": inner.scores.iter().map(|score| json!({
            "penalty": score.penalty,
            "log_loss": score.log_loss,
            "roc_auc": number(score.roc_auc),
            "fits": score.fits.iter().map(fit_json).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "chosen": inner.chosen,
        "chosen_by_roc_auc": number(inner.chosen_by_roc_auc),
    })
}

fn calibration_json(calibration: &CalibrationResult, mixed: &BTreeSet<String>) -> Value {
    let mut value = coverage_json(&calibration.overall);
    let object = value.as_object_mut().expect("a coverage is an object");
    object.insert("penalty".into(), json!(calibration.penalty));
    object.insert("fit".into(), fit_json(&calibration.fit));
    object.insert(
        "calibration_groups".into(),
        json!(calibration.calibration_groups.len()),
    );
    object.insert(
        "calibration_groups_with_both_labels".into(),
        json!(both_labels(&calibration.calibration_groups, mixed)),
    );
    object.insert(
        "calibration_logs".into(),
        json!({ "good": calibration.calibration_logs[0], "bad": calibration.calibration_logs[1] }),
    );
    object.insert("lists".into(), json!(calibration.lists));
    object.insert(
        "smallest_supported_error_rate".into(),
        calibration.smallest_supported.map_or(Value::Null, short),
    );
    object.insert("refused".into(), json!(calibration.refused));
    object.insert(
        "cross_endpoint".into(),
        coverage_json(&calibration.cross_endpoint),
    );
    object.insert(
        "single_label".into(),
        coverage_json(&calibration.single_label),
    );
    object.insert(
        "per_group".into(),
        Value::Object(
            calibration
                .per_group
                .iter()
                .map(|(group, coverage)| (group.clone(), coverage_json(coverage)))
                .collect(),
        ),
    );
    value
}

fn fold_json(index: usize, fold: &FoldResult, mixed: &BTreeSet<String>) -> Value {
    json!({
        "fold": index,
        "fitting_logs": fold.fitting_logs,
        "held_out_logs": fold.held_rows.len(),
        "inner": fold.inner.as_ref().map_or(Value::Null, |inner| inner_json(inner, mixed)),
        "by_penalty": fold.by_penalty.iter().map(|result| {
            let mut value = metrics_json(&result.metrics);
            let object = value.as_object_mut().expect("metrics are an object");
            object.insert("penalty".into(), json!(result.penalty));
            object.insert("fit".into(), fit_json(&result.fit));
            value
        }).collect::<Vec<_>>(),
        "calibration": fold.calibration.iter()
            .map(|calibration| calibration_json(calibration, mixed)).collect::<Vec<_>>(),
    })
}

fn calibration_summary_json(summary: &CalibrationSummary, error_rate: f32) -> Value {
    let mut value = coverage_json(&summary.overall);
    let object = value.as_object_mut().expect("a coverage is an object");
    object.insert("chosen_error_rate".into(), short(error_rate));
    object.insert("folds".into(), json!(summary.folds));
    object.insert(
        "refused".into(),
        json!(
            summary
                .refused
                .iter()
                .map(|(fold, reason)| json!({ "fold": fold, "reason": reason }))
                .collect::<Vec<_>>()
        ),
    );
    object.insert(
        "cross_endpoint".into(),
        coverage_json(&summary.cross_endpoint),
    );
    object.insert("single_label".into(), coverage_json(&summary.single_label));
    object.insert(
        "per_group".into(),
        Value::Object(
            summary
                .per_group
                .iter()
                .map(|(group, coverage)| (group.clone(), coverage_json(coverage)))
                .collect(),
        ),
    );
    object.insert(
        "per_fold".into(),
        json!(
            summary
                .fold_overall
                .iter()
                .map(|coverage| coverage.as_ref().map_or(Value::Null, coverage_json))
                .collect::<Vec<_>>()
        ),
    );
    object.insert(
        "per_fold_calibration_logs".into(),
        json!(
            summary
                .fold_calibration_logs
                .iter()
                .map(|logs| json!({ "good": logs[0], "bad": logs[1] }))
                .collect::<Vec<_>>()
        ),
    );
    object.insert("per_fold_lists".into(), json!(summary.fold_lists));
    value
}

/// Whether the mean over the folds of the cross-endpoint ROC AUC reaches the gate.
fn gate_json(summary: &RuleSummary) -> Value {
    match &summary.roc_auc {
        Some(spread) => json!({
            "figure": "mean over the folds of the cross-endpoint ROC AUC",
            "threshold": GATE_ROC_AUC,
            "value": spread.mean,
            "passed": spread.mean >= GATE_ROC_AUC,
            "lowest_fold": spread.min,
            "folds_below_threshold": summary.fold_roc_auc.iter().flatten()
                .filter(|area| **area < GATE_ROC_AUC).count(),
        }),
        None => Value::Null,
    }
}

fn rule_json(summary: &RuleSummary, calibration: Option<(&CalibrationSummary, f32)>) -> Value {
    json!({
        "penalties": summary.penalties.iter().map(|penalty| number(*penalty)).collect::<Vec<_>>(),
        "cross_endpoint": {
            "per_fold_roc_auc": summary.fold_roc_auc.iter().map(|v| number(*v)).collect::<Vec<_>>(),
            "per_fold_average_precision":
                summary.fold_average_precision.iter().map(|v| number(*v)).collect::<Vec<_>>(),
            "roc_auc": spread_json(&summary.roc_auc),
            "average_precision": spread_json(&summary.average_precision),
            "pooled_over_folds": block_json(&summary.pooled),
            "per_endpoint": blocks_json(summary.per_endpoint.iter()),
            "endpoint_roc_auc": spread_json(&summary.endpoint_roc_auc),
            "large_endpoint_roc_auc": spread_json(&summary.large_endpoint_roc_auc),
        },
        "overall": {
            "per_fold_roc_auc":
                summary.fold_overall_roc_auc.iter().map(|v| number(*v)).collect::<Vec<_>>(),
            "roc_auc": spread_json(&summary.overall_roc_auc),
            "pooled_over_folds": block_json(&summary.pooled_overall),
        },
        "single_label": single_json(&summary.single_label),
        "budgets": summary.budgets.iter().map(|budget| json!({
            "name": budget.name,
            "per_fold_budget_logs": budget.fold_budget,
            "recall": spread_json(&budget.recall),
            "single_label_share": number(budget.single_label_share),
            "single_label_bad_share": number(budget.single_label_bad_share),
            "single_label_good_share": number(budget.single_label_good_share),
        })).collect::<Vec<_>>(),
        "calibration": calibration.map_or(Value::Null, |(summary, error_rate)| {
            calibration_summary_json(summary, error_rate)
        }),
        "gate": gate_json(summary),
    })
}

/// The summaries of one mode of one arrangement: `(name, summary, calibration)`.
struct Rules {
    default: RuleSummary,
    default_calibration: CalibrationSummary,
    chosen: Option<(RuleSummary, CalibrationSummary)>,
    /// The second rule that reads the fitting side only: the highest inner ROC AUC.
    by_roc_auc: Option<RuleSummary>,
    sensitivity: Vec<(f64, RuleSummary)>,
}

fn rules(data: &EvalLogs, folds: &[FoldResult], settings: &EvalSettings) -> Rules {
    let has_choice = folds.iter().all(|fold| fold.inner.is_some());
    Rules {
        default: rule_summary(data, folds, settings, |fold| fold.default_penalty()),
        default_calibration: calibration_summary(folds, false),
        chosen: has_choice.then(|| {
            (
                rule_summary(data, folds, settings, |fold| fold.chosen()),
                calibration_summary(folds, true),
            )
        }),
        by_roc_auc: has_choice
            .then(|| rule_summary(data, folds, settings, |fold| fold.chosen_by_roc_auc())),
        sensitivity: settings
            .penalties
            .iter()
            .map(|&penalty| {
                (
                    penalty,
                    rule_summary(data, folds, settings, |fold| fold.at(penalty)),
                )
            })
            .collect(),
    }
}

fn settings_json(settings: &EvalSettings) -> Value {
    json!({
        "routing": routing_name(settings.baseline.routing),
        "buckets": settings.baseline.buckets,
        "ngram_max": settings.baseline.ngram_max,
        "max_iter": settings.baseline.max_iter,
        "tolerance": settings.baseline.tolerance,
        "default_penalty": settings.baseline.l2,
        "penalties": settings.penalties,
        "inner_folds": settings.inner_folds,
        "penalty_choice": if settings.inner_folds >= 2 {
            "smallest mean log loss over the inner held-out logs of the fitting side"
        } else {
            "none"
        },
        "error_rate": short(settings.error_rate),
        "calibration_min_per_group": settings.calibration.min_per_group,
        "budget_share": settings.budget_share,
        "small_endpoint_logs_per_label": settings.small_endpoint,
    })
}

fn dataset_json(report: &Report) -> Value {
    let data = report.data;
    let summary = &report.dataset.summary;
    let mixed = data.mixed_groups();
    let counts = data.group_counts();
    let bad = data.bad().iter().filter(|bad| **bad).count();
    let single = counts.iter().filter(|(group, _)| !mixed.contains(*group));
    let (mut good_only, mut bad_only, mut single_good, mut single_bad) =
        (0_u64, 0_u64, 0_u64, 0_u64);
    for (_, (good, bad)) in single {
        if *bad == 0 {
            good_only += 1;
        } else {
            bad_only += 1;
        }
        single_good += good;
        single_bad += bad;
    }
    json!({
        "name": report.dataset.name,
        "path": summary.source_path.display().to_string(),
        "sha256": summary.content_sha256,
        "records": summary.records,
        "distinct_texts": summary.distinct_texts,
        "skipped_records": summary.skipped.len(),
        "evaluated_logs": data.len(),
        "good": data.len() - bad,
        "bad": bad,
        "unlabelled_left_out": data.unknown_left_out(),
        "group_keys": counts.len(),
        "groups_with_both_labels": Value::Object(
            counts.iter().filter(|(group, _)| mixed.contains(*group))
                .map(|(group, (good, bad))| (group.clone(), json!({ "good": good, "bad": bad })))
                .collect(),
        ),
        "single_label_groups": {
            "groups": good_only + bad_only,
            "good_only_groups": good_only,
            "bad_only_groups": bad_only,
            "logs": single_good + single_bad,
            "good": single_good,
            "bad": single_bad,
        },
    })
}

fn definitions() -> Value {
    json!({
        "logs": "distinct texts; every figure counts a text once, whatever its occurrences",
        "ratio": "the baseline scorer's bad/good ratio, higher is more likely bad",
        "cross_endpoint": "the held-out logs of the group keys that hold both labels in the \
            whole dataset",
        "per_fold": "the ROC AUC and average precision over one fold's held-out cross-endpoint \
            logs, pooled; the mean over the folds is the gate figure",
        "per_endpoint": "one group key with both labels, from the fold that held it out; the \
            mean counts each endpoint once",
        "pooled_over_folds": "the held-out logs of all folds as one ranking; their ratios come \
            from different fits",
        "overall": "all held-out logs, single-label groups included",
        "single_label": "the logs of group keys that hold one label; never in a cross-endpoint \
            figure",
        "budgets": "the held-out side of a fold ranked by ratio as a whole; share: the \
            configured share of its logs, rounded down; bad_count: as many logs as it holds \
            bad logs; a tie at the budget is shared between the tied logs",
        "default": "the default L2 penalty of the baseline scorer",
        "chosen_inside_fold": "the penalty of the grid with the smallest mean log loss over \
            inner folds of the fitting side; the held-out side plays no part",
        "chosen_by_inner_roc_auc": "the penalty of the grid with the highest mean ROC AUC over \
            the inner folds of the fitting side; a second rule in which the held-out side plays \
            no part, reported beside the first; it has no calibration of its own",
        "sensitivity": "every penalty of the grid on the held-out side; not a setting, because \
            picking from it uses held-out labels",
        "calibration": "the calibration logs are the first inner fold of the fitting side; the \
            calibrated scorer was fitted on the fitting side less those logs and judges the \
            held-out logs; achieved_error_rate is the share of the logs of a label whose label \
            set does not hold it",
        "spread": "count, mean, min, max and sample standard deviation (divisor count - 1) \
            over the folds or endpoints where the figure is defined",
    })
}

/// The report as JSON: the same value, and the same text when printed, for the same
/// results.
pub fn report_json(report: &Report) -> Value {
    let data = report.data;
    let mixed = data.mixed_groups();
    let arrangements: Vec<Value> = report
        .arrangements
        .iter()
        .zip(report.results)
        .map(|(plan, result)| {
            let folds: Vec<Value> = (0..plan.folds.len())
                .map(|fold| {
                    let held = plan.folds.held_out(fold);
                    let mask = plan.folds.mask(fold, data.groups());
                    let logs = mask.iter().filter(|held| **held).count();
                    let bad = mask
                        .iter()
                        .zip(data.bad())
                        .filter(|(held, bad)| **held && **bad)
                        .count();
                    json!({
                        "fold": fold,
                        "held_out_groups": held,
                        "held_out_group_count": held.len(),
                        "held_out_groups_with_both_labels": both_labels(held, &mixed),
                        "held_out_sha256": plan.folds.membership_sha256(fold),
                        "held_out_logs": logs,
                        "held_out_good": logs - bad,
                        "held_out_bad": bad,
                    })
                })
                .collect();
            let modes: Vec<Value> = report
                .modes
                .iter()
                .zip(&result.modes)
                .map(|(settings, mode)| {
                    let rules = rules(data, &mode.folds, settings);
                    let with_choice = settings.inner_folds >= 2;
                    json!({
                        "settings": settings_json(settings),
                        "folds": mode.folds.iter().enumerate()
                            .map(|(index, fold)| fold_json(index, fold, &mixed))
                            .collect::<Vec<_>>(),
                        "summary": {
                            "default": rule_json(
                                &rules.default,
                                with_choice.then_some((&rules.default_calibration, settings.error_rate)),
                            ),
                            "chosen_inside_fold": rules.chosen.as_ref().map_or(
                                Value::Null,
                                |(summary, calibration)| {
                                    rule_json(summary, Some((calibration, settings.error_rate)))
                                },
                            ),
                            "chosen_by_inner_roc_auc": rules.by_roc_auc.as_ref()
                                .map_or(Value::Null, |summary| rule_json(summary, None)),
                            "sensitivity": rules.sensitivity.iter().map(|(penalty, summary)| {
                                let mut value = rule_json(summary, None);
                                value.as_object_mut().expect("a rule is an object")
                                    .insert("penalty".into(), json!(penalty));
                                value
                            }).collect::<Vec<_>>(),
                        },
                    })
                })
                .collect();
            json!({
                "name": plan.name,
                "rule": plan.rule,
                "kept_together": plan.together,
                "folds": folds,
                "modes": modes,
            })
        })
        .collect();
    json!({
        "task": report.task,
        "date": report.date,
        "dataset": dataset_json(report),
        "definitions": definitions(),
        "gate_threshold": GATE_ROC_AUC,
        "notes": report.notes,
        "arrangements": arrangements,
    })
}

fn figure(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |value| format!("{value:.4}"))
}

fn percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |value| format!("{:.1} %", 100.0 * value),
    )
}

fn spread_cells(spread: &Option<Spread>) -> String {
    match spread {
        Some(spread) => format!(
            "{:.4} | {:.4} | {:.4} | {:.4}",
            spread.mean, spread.min, spread.max, spread.sd
        ),
        None => "n/a | n/a | n/a | n/a".to_string(),
    }
}

fn rule_row(out: &mut String, name: &str, summary: &RuleSummary) {
    let folds: Vec<String> = summary.fold_roc_auc.iter().map(|v| figure(*v)).collect();
    let verdict = match &summary.roc_auc {
        Some(spread) if spread.mean >= GATE_ROC_AUC => "pass",
        Some(_) => "FAIL",
        None => "n/a",
    };
    let _ = writeln!(
        out,
        "| {name} | {} | {} | {} | {} | {} | {} | {verdict} |",
        folds.join(" / "),
        spread_cells(&summary.roc_auc),
        figure(summary.pooled.roc_auc),
        figure(summary.endpoint_roc_auc.map(|spread| spread.mean)),
        figure(summary.large_endpoint_roc_auc.map(|spread| spread.mean)),
        figure(summary.average_precision.map(|spread| spread.mean)),
    );
}

fn coverage_row(out: &mut String, name: &str, coverage: &Coverage) {
    let logs = coverage.logs[0] + coverage.logs[1];
    let share = |count: u64| percent((logs > 0).then(|| count as f64 / logs as f64));
    let _ = writeln!(
        out,
        "| {name} | {} | {} | {} | {} | {} | {} | {} | {} |",
        coverage.logs[0],
        percent(coverage.error_rate(0)),
        coverage.logs[1],
        percent(coverage.error_rate(1)),
        share(coverage.sets[0]),
        share(coverage.sets[1]),
        share(coverage.sets[2]),
        share(coverage.sets[3]),
    );
}

const COVERAGE_HEAD: &str = "| Logs | Good logs | Good wrongly ruled out | Bad logs | Bad wrongly \
    ruled out | Set good | Set bad | Set both | Set neither |\n| --- | ---: | ---: | ---: | ---: | \
    ---: | ---: | ---: | ---: |\n";

/// The report as a readable summary in Markdown: the same figures as the JSON, without
/// the per-fold detail of the fits.
pub fn report_markdown(report: &Report) -> String {
    let data = report.data;
    let summary = &report.dataset.summary;
    let mixed = data.mixed_groups();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Grouped evaluation of the baseline scorer: {}\n",
        report.dataset.name
    );
    let _ = writeln!(out, "Date: {} · {}\n", report.date, report.task);
    let _ = writeln!(
        out,
        "Dataset: `{}`, SHA-256 `{}`, {} records, {} distinct texts, {} evaluated ({} good, {} \
         bad, {} left out as unlabelled), {} group keys, {} of them with both labels.\n",
        summary.source_path.display(),
        summary.content_sha256,
        summary.records,
        summary.distinct_texts,
        data.len(),
        data.bad().iter().filter(|bad| !**bad).count(),
        data.bad().iter().filter(|bad| **bad).count(),
        data.unknown_left_out(),
        data.group_counts().len(),
        mixed.len(),
    );
    let _ = writeln!(
        out,
        "All figures count distinct texts. Cross-endpoint figures use the held-out logs of group \
         keys that hold both labels. The gate is a mean over the folds of the cross-endpoint ROC \
         AUC of at least {GATE_ROC_AUC}. The rows of a penalty of the grid are a sensitivity \
         table, not a setting.\n"
    );
    let _ = writeln!(
        out,
        "| Group key with both labels | Good | Bad |\n| --- | ---: | ---: |"
    );
    for group in &mixed {
        let (good, bad) = data.group_counts()[group];
        let _ = writeln!(out, "| `{group}` | {good} | {bad} |");
    }

    for (plan, result) in report.arrangements.iter().zip(report.results) {
        let _ = writeln!(out, "\n## Arrangement `{}`\n\n{}\n", plan.name, plan.rule);
        let _ = writeln!(
            out,
            "| Fold | Held-out group keys with both labels | Held-out group keys | Held-out logs | \
             SHA-256 of the held-out keys |\n| ---: | --- | ---: | ---: | --- |"
        );
        for fold in 0..plan.folds.len() {
            let held = plan.folds.held_out(fold);
            let logs = plan
                .folds
                .mask(fold, data.groups())
                .iter()
                .filter(|held| **held)
                .count();
            let _ = writeln!(
                out,
                "| {fold} | {} | {} | {logs} | `{}` |",
                both_labels(held, &mixed)
                    .iter()
                    .map(|group| format!("`{group}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
                held.len(),
                plan.folds.membership_sha256(fold),
            );
        }
        for (settings, mode) in report.modes.iter().zip(&result.modes) {
            let rules = rules(data, &mode.folds, settings);
            let _ = writeln!(
                out,
                "\n### `{}`, routing `{}`\n",
                plan.name,
                routing_name(settings.baseline.routing)
            );
            let _ = writeln!(
                out,
                "Cross-endpoint ROC AUC.\n\n| Penalty rule | Per fold | Mean | Min | Max | SD | \
                 Pooled over folds | Mean of endpoints | Mean of large endpoints | Average \
                 precision, mean | Gate |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | \
                 ---: | ---: | --- |"
            );
            rule_row(
                &mut out,
                &format!("default ({})", settings.baseline.l2),
                &rules.default,
            );
            let used = |summary: &RuleSummary| -> String {
                summary
                    .penalties
                    .iter()
                    .map(|penalty| penalty.map_or_else(|| "n/a".into(), |p| p.to_string()))
                    .collect::<Vec<String>>()
                    .join(" / ")
            };
            if let Some((chosen, _)) = &rules.chosen {
                rule_row(
                    &mut out,
                    &format!(
                        "chosen inside the fold by inner log loss ({})",
                        used(chosen)
                    ),
                    chosen,
                );
            }
            if let Some(chosen) = &rules.by_roc_auc {
                rule_row(
                    &mut out,
                    &format!("chosen inside the fold by inner ROC AUC ({})", used(chosen)),
                    chosen,
                );
            }
            for (penalty, summary) in &rules.sensitivity {
                rule_row(&mut out, &format!("sensitivity: {penalty}"), summary);
            }

            let _ = writeln!(
                out,
                "\nPer endpoint, ROC AUC (average precision).\n\n| Endpoint | Good | Bad | Default \
                 | Chosen by inner log loss | Chosen by inner ROC AUC |\n| --- | ---: | ---: | --- | \
                 --- | --- |"
            );
            for (group, block) in &rules.default.per_endpoint {
                let chosen = rules
                    .chosen
                    .as_ref()
                    .and_then(|(summary, _)| summary.per_endpoint.get(group));
                let by_roc_auc = rules
                    .by_roc_auc
                    .as_ref()
                    .and_then(|summary| summary.per_endpoint.get(group));
                let cell = |block: &Block| {
                    format!(
                        "{} ({})",
                        figure(block.roc_auc),
                        figure(block.average_precision)
                    )
                };
                let _ = writeln!(
                    out,
                    "| `{group}` | {} | {} | {} | {} | {} |",
                    block.good,
                    block.bad,
                    cell(block),
                    chosen.map_or_else(|| "n/a".to_string(), cell),
                    by_roc_auc.map_or_else(|| "n/a".to_string(), cell),
                );
            }

            if mode.folds.iter().any(|fold| fold.inner.is_some()) {
                let _ = writeln!(
                    out,
                    "\nInner mean log loss per penalty (inner ROC AUC), and the choice.\n\n| Fold \
                     | {} | Chosen by log loss | By inner ROC AUC |\n| ---: | {}--- | --- |",
                    settings
                        .penalties
                        .iter()
                        .map(|penalty| penalty.to_string())
                        .collect::<Vec<_>>()
                        .join(" | "),
                    "--- | ".repeat(settings.penalties.len()),
                );
                for (index, fold) in mode.folds.iter().enumerate() {
                    let Some(inner) = &fold.inner else { continue };
                    let cells: Vec<String> = inner
                        .scores
                        .iter()
                        .map(|score| format!("{:.4} ({})", score.log_loss, figure(score.roc_auc)))
                        .collect();
                    let _ = writeln!(
                        out,
                        "| {index} | {} | {} | {} |",
                        cells.join(" | "),
                        inner.chosen,
                        inner
                            .chosen_by_roc_auc
                            .map_or_else(|| "n/a".to_string(), |p| p.to_string()),
                    );
                }

                let _ = write!(
                    out,
                    "\nCalibration at error rate {}: achieved error rates and label sets on the \
                     held-out side, all folds together.\n\n{COVERAGE_HEAD}",
                    settings.error_rate
                );
                let mut rows = vec![("default", &rules.default_calibration)];
                if let Some((_, calibration)) = &rules.chosen {
                    rows.push(("chosen", calibration));
                }
                for (name, calibration) in rows {
                    coverage_row(&mut out, &format!("{name}: all"), &calibration.overall);
                    coverage_row(
                        &mut out,
                        &format!("{name}: both-label groups"),
                        &calibration.cross_endpoint,
                    );
                    coverage_row(
                        &mut out,
                        &format!("{name}: single-label groups"),
                        &calibration.single_label,
                    );
                    for (group, coverage) in &calibration.per_group {
                        coverage_row(&mut out, &format!("{name}: `{group}`"), coverage);
                    }
                    for (fold, reason) in &calibration.refused {
                        let _ = writeln!(
                            out,
                            "| {name}: fold {fold} refused: {reason} | | | | | | | | |"
                        );
                    }
                }
            }

            let single = &rules.default.single_label;
            let _ = writeln!(
                out,
                "\nSingle-label groups (default penalty): {} groups, {} logs ({} good, {} bad).\n\n\
                 | Budget | Per fold, logs | Recall, mean over folds | Single-label logs within \
                 the budget | Bad ones | Good ones |\n| --- | --- | ---: | ---: | ---: | ---: |",
                single.groups, single.logs, single.good, single.bad
            );
            for budget in &rules.default.budgets {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} |",
                    budget.name,
                    budget
                        .fold_budget
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(" / "),
                    figure(budget.recall.map(|spread| spread.mean)),
                    percent(budget.single_label_share),
                    percent(budget.single_label_bad_share),
                    percent(budget.single_label_good_share),
                );
            }
        }
    }
    out
}

/// Writes the report as `<stem>.json` (indented, one line feed at the end) and
/// `<stem>.md` into `dir`, which is created when missing, and returns the two paths.
/// Existing files of these names are replaced.
pub fn write_report(
    dir: &Path,
    stem: &str,
    report: &Report,
) -> Result<(PathBuf, PathBuf), L2dError> {
    let failed = |what: &Path, error: &dyn std::fmt::Display| L2dError::Runtime {
        detail: format!("writing {} failed: {error}", what.display()),
    };
    std::fs::create_dir_all(dir).map_err(|error| failed(dir, &error))?;
    let json_path = dir.join(format!("{stem}.json"));
    let text_path = dir.join(format!("{stem}.md"));
    let mut json = serde_json::to_string_pretty(&report_json(report))
        .map_err(|error| failed(&json_path, &error))?;
    json.push('\n');
    std::fs::write(&json_path, json).map_err(|error| failed(&json_path, &error))?;
    std::fs::write(&text_path, report_markdown(report))
        .map_err(|error| failed(&text_path, &error))?;
    Ok((json_path, text_path))
}
