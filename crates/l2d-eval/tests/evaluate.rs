//! The grouped evaluation of the baseline scorer on small synthetic logs and on the mini
//! fixture: what is pooled where, that nothing of a fold's held-out side reaches the
//! fit, the choice of the penalty or the calibration, and that a run repeats exactly.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use l2d_baseline::{BaselineSettings, RoutingMode};
use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LogRecord,
    LogsFrame, MemoryBudget, MemorySettings,
};
use l2d_decide::CalibrationSettings;
use l2d_eval::evaluate::{
    Arrangement, EvalLogs, EvalSettings, FoldResult, calibration_summary, evaluate, evaluate_fold,
    predictions_frame, rule_summary, summarize,
};
use l2d_eval::report::{Dataset, Report, date_of, report_json, report_markdown, write_report};
use l2d_eval::split::{Folds, units};

const GIB: u64 = 1024 * 1024 * 1024;

fn budget_with_limit(limit: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).expect("the budget is usable")
}

fn budget() -> MemoryBudget {
    budget_with_limit(None)
}

fn close(found: f64, expected: f64) {
    assert!(
        (found - expected).abs() < 1e-12,
        "found {found}, expected {expected}"
    );
}

const ATTACKS: [&str; 6] = [
    "'+OR+1%3D1--",
    "<script>alert(1)</script>",
    "../../../../etc/passwd",
    "%27%3B+DROP+TABLE+usuarios%3B--",
    "|cat+/etc/shadow",
    "%3Cimg+src%3Dx+onerror%3Dalert(2)%3E",
];
const WORDS: [&str; 7] = [
    "jamon", "queso", "vino", "aceite", "azafran", "pimenton", "chorizo",
];

fn record(group: &str, index: usize, bad: bool, position: u64) -> LogRecord {
    let value = if bad {
        ATTACKS[index % ATTACKS.len()]
    } else {
        WORDS[index % WORDS.len()]
    };
    LogRecord {
        text: format!(
            "GET http://localhost:8080{group}?id={index}&nombre={value}&cantidad={}",
            index % 9 + 1
        ),
        label: if bad { Label::Bad } else { Label::Good },
        group: Some(group.to_string()),
        first_position: position,
        occurrences: 1,
    }
}

/// Six endpoints with both labels (30 good and 20 bad logs each), ten endpoints with
/// two bad logs each and two with one good log each, interleaved.
fn synthetic_records() -> Vec<LogRecord> {
    let mut records = Vec::new();
    let mut position = 0;
    let mut push = |group: &str, index: usize, bad: bool| {
        position += 1;
        records.push(record(group, index, bad, position));
    };
    for index in 0..50 {
        for endpoint in 0..6 {
            push(
                &format!("/tienda/e{endpoint}.jsp"),
                index * 6 + endpoint,
                index % 5 >= 3,
            );
        }
        if index < 20 {
            push(&format!("/old/b{}.bak", index / 2), 1000 + index, true);
        }
        if index < 2 {
            push(&format!("/static/s{index}.css"), 2000 + index, false);
        }
    }
    records
}

fn synthetic() -> EvalLogs {
    EvalLogs::new(&LogsFrame::from_records(&synthetic_records()).unwrap())
}

fn settings() -> EvalSettings {
    EvalSettings {
        baseline: BaselineSettings {
            buckets: 1_024,
            routing: RoutingMode::Strip,
            ..BaselineSettings::default()
        },
        penalties: vec![0.1, 1.0, 10.0],
        inner_folds: 2,
        error_rate: 0.1,
        calibration: CalibrationSettings { min_per_group: 10 },
        budget_share: 0.1,
        small_endpoint: 10,
    }
}

fn mixed_first(data: &EvalLogs, folds: usize) -> Folds {
    Folds::dealt(
        &units(data.groups(), &[]).unwrap(),
        folds,
        &data.mixed_groups(),
    )
    .unwrap()
}

/// Everything of a fold's result that must repeat (all but the timing).
fn same(a: &FoldResult, b: &FoldResult) -> bool {
    a.held_rows == b.held_rows
        && a.fitting_logs == b.fitting_logs
        && a.inner == b.inner
        && a.by_penalty == b.by_penalty
        && a.calibration == b.calibration
}

// ---------------------------------------------------------------------------------------
// The logs of an evaluation
// ---------------------------------------------------------------------------------------

#[test]
fn unlabelled_logs_are_left_out_and_a_log_without_a_group_is_its_own_group() {
    let mut records = vec![
        record("/a", 1, false, 1),
        record("/a", 2, true, 2),
        record("/b", 3, true, 3),
        record("/b", 4, true, 4),
    ];
    records.push(LogRecord {
        label: Label::Unknown,
        ..record("/a", 5, false, 5)
    });
    records.push(LogRecord {
        group: None,
        ..record("/x", 6, false, 6)
    });
    records.push(LogRecord {
        group: None,
        ..record("/x", 7, false, 7)
    });
    let data = EvalLogs::new(&LogsFrame::from_records(&records).unwrap());
    assert_eq!(data.len(), 6);
    assert_eq!(data.unknown_left_out(), 1);
    assert_eq!(data.bad(), &[false, true, true, true, false, false]);
    assert_eq!(&data.groups()[..4], &["/a", "/a", "/b", "/b"]);
    assert_ne!(data.groups()[4], data.groups()[5], "each its own group");
    assert_eq!(data.group_counts().len(), 4);
    assert_eq!(data.group_counts()["/a"], (1, 1));
    assert_eq!(data.group_counts()["/b"], (0, 2));
    assert_eq!(
        data.mixed_groups(),
        ["/a".to_string()].into_iter().collect::<BTreeSet<_>>()
    );
}

// ---------------------------------------------------------------------------------------
// Pooling, calculated by hand
// ---------------------------------------------------------------------------------------

#[test]
fn the_figures_of_a_held_out_side_are_pooled_as_documented() {
    // /a holds both labels; /b only bad logs, /c only a good one.
    let groups = ["/a", "/a", "/a", "/a", "/b", "/b", "/c"];
    let bad = [true, false, true, false, true, true, false];
    let ratios = [0.9, 0.2, 0.6, 0.7, 0.8, 0.1, 0.3];
    let mixed: BTreeSet<String> = ["/a".to_string(), "/elsewhere".to_string()]
        .into_iter()
        .collect();
    let found = summarize(&groups, &bad, &ratios, &mixed, 0.5);

    // Overall: 4 bad, 3 good; the bad log wins 8 of the 12 pairs.
    assert_eq!(
        (found.overall.logs, found.overall.good, found.overall.bad),
        (7, 3, 4)
    );
    close(found.overall.roc_auc.unwrap(), 8.0 / 12.0);
    // Cross-endpoint: the logs of /a alone; 3 of 4 pairs.
    assert_eq!(
        (
            found.cross_endpoint.logs,
            found.cross_endpoint.good,
            found.cross_endpoint.bad
        ),
        (4, 2, 2)
    );
    close(found.cross_endpoint.roc_auc.unwrap(), 0.75);
    // From the top of /a: 0.9 bad (1/1), 0.7 good, 0.6 bad (2/3): 1/2 + 1/2 * 2/3.
    close(
        found.cross_endpoint.average_precision.unwrap(),
        0.5 + 1.0 / 3.0,
    );
    assert_eq!(
        found.per_endpoint.keys().collect::<Vec<_>>(),
        vec!["/a"],
        "only groups with both labels that are held out here"
    );
    assert_eq!(found.per_endpoint["/a"], found.cross_endpoint);

    // Single-label groups: reported apart, never in the cross-endpoint figure.
    let single = &found.single_label;
    assert_eq!(
        (single.groups, single.logs, single.good, single.bad),
        (2, 3, 1, 2)
    );

    // Budgets. Ranking: 0.9 0.8 0.7 | 0.6 | 0.3 0.2 0.1.
    let share = &found.budgets[0];
    assert_eq!((share.name.as_str(), share.budget), ("share", 3));
    close(share.recall.unwrap(), 2.0 / 4.0);
    close(share.cross_endpoint_recall.unwrap(), 1.0 / 2.0);
    close(share.single_label_share.unwrap(), 1.0 / 3.0);
    close(share.single_label_bad_share.unwrap(), 1.0 / 2.0);
    close(share.single_label_good_share.unwrap(), 0.0);
    let count = &found.budgets[1];
    assert_eq!((count.name.as_str(), count.budget), ("bad_count", 4));
    close(count.recall.unwrap(), 3.0 / 4.0);
    close(count.cross_endpoint_recall.unwrap(), 1.0);
    close(count.single_label_share.unwrap(), 1.0 / 3.0);
}

#[test]
fn a_held_out_side_with_one_label_has_counts_and_no_roc_auc() {
    let found = summarize(
        &["/b", "/b", "/d"],
        &[true, true, true],
        &[0.4, 0.4, 0.9],
        &BTreeSet::new(),
        0.34,
    );
    assert_eq!(found.overall.roc_auc, None);
    assert_eq!(found.overall.average_precision, None);
    assert_eq!(found.cross_endpoint.logs, 0);
    assert_eq!(found.cross_endpoint.roc_auc, None);
    assert!(found.per_endpoint.is_empty());
    assert_eq!(found.single_label.groups, 2);
    assert_eq!(found.single_label.bad, 3);
    // A budget of one log: the 0.9. No good log among the single-label ones.
    assert_eq!(found.budgets[0].budget, 1);
    close(found.budgets[0].single_label_share.unwrap(), 1.0 / 3.0);
    assert_eq!(found.budgets[0].single_label_good_share, None);
    assert_eq!(found.budgets[0].cross_endpoint_recall, None);
}

// ---------------------------------------------------------------------------------------
// One fold
// ---------------------------------------------------------------------------------------

#[test]
fn a_fold_fits_on_one_side_and_scores_the_other() {
    let data = synthetic();
    let folds = mixed_first(&data, 3);
    let settings = settings();
    let held_out = folds.mask(0, data.groups());
    let result = evaluate_fold(&data, &held_out, &[], &settings, &mut budget()).unwrap();

    let held: Vec<u32> = (0..data.len() as u32)
        .filter(|&row| held_out[row as usize])
        .collect();
    assert_eq!(result.held_rows, held);
    assert_eq!(result.fitting_logs as usize, data.len() - held.len());

    // Every penalty of the grid was fitted on the whole fitting side and scored the
    // held-out side.
    let penalties: Vec<f64> = result.by_penalty.iter().map(|p| p.penalty).collect();
    assert_eq!(penalties, settings.penalties);
    for scored in &result.by_penalty {
        assert_eq!(scored.fit.rows, result.fitting_logs);
        assert_eq!(scored.ratios.len(), held.len());
        assert_eq!(scored.metrics.overall.logs as usize, held.len());
        assert!(scored.ratios.iter().all(|r| (0.0..=1.0).contains(r)));
    }

    // The choice: inner folds of the fitting side only, the smallest inner log loss.
    let inner = result.inner.as_ref().expect("a penalty was chosen");
    assert_eq!(inner.folds.len(), 2);
    let fitting_groups: BTreeSet<&String> = data
        .groups()
        .iter()
        .zip(&held_out)
        .filter(|(_, held)| !**held)
        .map(|(group, _)| group)
        .collect();
    let inner_groups: BTreeSet<&String> = inner.folds.iter().flatten().collect();
    assert_eq!(inner_groups, fitting_groups);
    for group in folds.held_out(0) {
        assert!(!inner_groups.contains(group), "{group} is held out");
    }
    assert_eq!(inner.scores.len(), 3);
    let best = inner
        .scores
        .iter()
        .min_by(|a, b| a.log_loss.partial_cmp(&b.log_loss).unwrap())
        .unwrap();
    assert_eq!(inner.chosen, best.penalty);
    assert!(result.chosen().is_some());
    assert_eq!(result.chosen().unwrap().penalty, inner.chosen);
    assert_eq!(result.default_penalty().unwrap().penalty, 1.0);
    // The second rule: the highest inner ROC AUC, the smaller penalty on a tie.
    let highest = inner
        .scores
        .iter()
        .map(|score| score.roc_auc.unwrap())
        .fold(0.0, f64::max);
    let by_area = inner
        .scores
        .iter()
        .find(|score| score.roc_auc.unwrap() == highest)
        .unwrap();
    assert_eq!(inner.chosen_by_roc_auc, Some(by_area.penalty));
    assert_eq!(result.chosen_by_roc_auc().unwrap().penalty, by_area.penalty);
    // An inner fit sees the fitting side less one inner fold.
    for score in &inner.scores {
        assert_eq!(score.fits.len(), 2);
        for fit in &score.fits {
            assert!(fit.rows < result.fitting_logs);
            assert_eq!(fit.penalty, score.penalty);
        }
        assert_eq!(
            score.fits.iter().map(|fit| fit.rows).sum::<u64>(),
            result.fitting_logs,
            "two inner folds: each log is fitted on once"
        );
    }
}

#[test]
fn the_calibration_logs_are_a_part_of_the_fitting_side_the_scorer_did_not_see() {
    let data = synthetic();
    let folds = mixed_first(&data, 3);
    let settings = settings();
    let held_out = folds.mask(1, data.groups());
    let result = evaluate_fold(&data, &held_out, &[], &settings, &mut budget()).unwrap();
    let inner = result.inner.as_ref().unwrap();

    // One calibration for the default penalty and one for the chosen one, unless they
    // are the same.
    let penalties: Vec<f64> = result.calibration.iter().map(|c| c.penalty).collect();
    if inner.chosen == 1.0 {
        assert_eq!(penalties, vec![1.0]);
    } else {
        assert_eq!(penalties, vec![1.0, inner.chosen]);
    }
    for calibration in &result.calibration {
        // The calibration logs are the first inner fold: groups of the fitting side.
        assert_eq!(calibration.calibration_groups, inner.folds[0]);
        for group in &calibration.calibration_groups {
            assert!(!folds.held_out(1).contains(group));
        }
        let calibration_logs = data
            .groups()
            .iter()
            .filter(|group| calibration.calibration_groups.contains(*group))
            .count() as u64;
        assert_eq!(
            calibration.calibration_logs[0] + calibration.calibration_logs[1],
            calibration_logs
        );
        // Its scorer was fitted on the fitting side less those logs.
        assert_eq!(calibration.fit.rows, result.fitting_logs - calibration_logs);
        assert_eq!(calibration.refused, None);

        // Coverage on the held-out side: every log is judged once.
        let held = result.held_rows.len() as u64;
        assert_eq!(calibration.label_sets.len() as u64, held);
        let overall = &calibration.overall;
        assert_eq!(overall.logs[0] + overall.logs[1], held);
        assert_eq!(overall.sets.iter().sum::<u64>(), held);
        assert!(overall.missed[0] <= overall.logs[0] && overall.missed[1] <= overall.logs[1]);
        let parts = [&calibration.cross_endpoint, &calibration.single_label];
        for label in 0..2 {
            assert_eq!(
                parts.iter().map(|part| part.logs[label]).sum::<u64>(),
                overall.logs[label]
            );
            assert_eq!(
                parts.iter().map(|part| part.missed[label]).sum::<u64>(),
                overall.missed[label]
            );
            assert_eq!(
                calibration
                    .per_group
                    .values()
                    .map(|part| part.logs[label])
                    .sum::<u64>(),
                calibration.cross_endpoint.logs[label]
            );
        }
        // The count of missed logs is the count of sets without the true label.
        let missed = result
            .held_rows
            .iter()
            .zip(&calibration.label_sets)
            .filter(|(row, set)| {
                let label = if data.bad()[**row as usize] {
                    Label::Bad
                } else {
                    Label::Good
                };
                !set.contains(label)
            })
            .count() as u64;
        assert_eq!(overall.missed[0] + overall.missed[1], missed);
    }
}

#[test]
fn an_error_rate_the_calibration_logs_cannot_support_is_reported_not_guessed() {
    let data = synthetic();
    let folds = mixed_first(&data, 3);
    let settings = EvalSettings {
        error_rate: 0.001,
        ..settings()
    };
    let held_out = folds.mask(0, data.groups());
    let result = evaluate_fold(&data, &held_out, &[], &settings, &mut budget()).unwrap();
    for calibration in &result.calibration {
        let refusal = calibration
            .refused
            .as_ref()
            .expect("0.001 is not supported");
        assert!(
            refusal.contains("0.001") || refusal.contains("error rate"),
            "{refusal}"
        );
        assert!(calibration.label_sets.is_empty());
        assert_eq!(calibration.overall.logs, [0, 0]);
    }
}

/// `records` with the label of every log of `groups` flipped.
fn with_flipped_labels(records: &[LogRecord], groups: &BTreeSet<String>) -> Vec<LogRecord> {
    records
        .iter()
        .map(|record| {
            let mut record = record.clone();
            if groups.contains(record.group.as_ref().unwrap()) {
                record.label = match record.label {
                    Label::Good => Label::Bad,
                    Label::Bad => Label::Good,
                    Label::Unknown => Label::Unknown,
                };
            }
            record
        })
        .collect()
}

#[test]
fn the_labels_of_the_held_out_side_reach_nothing_but_the_metrics() {
    let records = synthetic_records();
    let data = EvalLogs::new(&LogsFrame::from_records(&records).unwrap());
    let folds = mixed_first(&data, 3);
    let settings = settings();
    let held_out = folds.mask(2, data.groups());
    let result = evaluate_fold(&data, &held_out, &[], &settings, &mut budget()).unwrap();

    let flipped = with_flipped_labels(&records, folds.held_out(2));
    let flipped = EvalLogs::new(&LogsFrame::from_records(&flipped).unwrap());
    assert_eq!(flipped.groups(), data.groups());
    let other = evaluate_fold(&flipped, &held_out, &[], &settings, &mut budget()).unwrap();

    // The choice of the penalty, every fit, every ratio and every label set are the
    // same, bit for bit.
    assert_eq!(result.inner, other.inner);
    assert_eq!(result.by_penalty.len(), other.by_penalty.len());
    for (a, b) in result.by_penalty.iter().zip(&other.by_penalty) {
        assert_eq!(a.fit, b.fit);
        assert_eq!(
            a.ratios.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
            b.ratios.iter().map(|r| r.to_bits()).collect::<Vec<_>>()
        );
        // Only the metrics read the labels: with every label flipped the ROC AUC is
        // mirrored.
        close(
            a.metrics.overall.roc_auc.unwrap() + b.metrics.overall.roc_auc.unwrap(),
            1.0,
        );
        assert_ne!(a.metrics, b.metrics);
    }
    for (a, b) in result.calibration.iter().zip(&other.calibration) {
        assert_eq!(a.fit, b.fit);
        assert_eq!(a.label_sets, b.label_sets);
        assert_eq!(a.calibration_logs, b.calibration_logs);
        assert_eq!(a.overall.sets, b.overall.sets);
    }
}

#[test]
fn the_texts_of_the_held_out_side_reach_no_fit_and_no_choice() {
    let records = synthetic_records();
    let data = EvalLogs::new(&LogsFrame::from_records(&records).unwrap());
    let folds = mixed_first(&data, 3);
    let settings = settings();
    let held_out = folds.mask(0, data.groups());
    let result = evaluate_fold(&data, &held_out, &[], &settings, &mut budget()).unwrap();

    // Other texts on the held-out side, same groups and labels.
    let changed: Vec<LogRecord> = records
        .iter()
        .map(|record| {
            let mut record = record.clone();
            if folds.held_out(0).contains(record.group.as_ref().unwrap()) {
                record.text = format!("{}&extra=zzzzzz{}", record.text, record.first_position);
            }
            record
        })
        .collect();
    let changed = EvalLogs::new(&LogsFrame::from_records(&changed).unwrap());
    let other = evaluate_fold(&changed, &held_out, &[], &settings, &mut budget()).unwrap();

    assert_eq!(result.inner, other.inner);
    for (a, b) in result.by_penalty.iter().zip(&other.by_penalty) {
        assert_eq!(a.fit, b.fit, "the fit saw the fitting side only");
    }
    for (a, b) in result.calibration.iter().zip(&other.calibration) {
        assert_eq!(a.fit, b.fit);
        assert_eq!(a.calibration_logs, b.calibration_logs);
    }
    assert_ne!(
        result.by_penalty[0].ratios, other.by_penalty[0].ratios,
        "the held-out texts did change"
    );
}

#[test]
fn a_held_out_log_whose_text_was_fitted_on_stops_the_evaluation() {
    // The same text under two group keys: one copy is fitted on, the other held out.
    let mut records = synthetic_records();
    let data = EvalLogs::new(&LogsFrame::from_records(&records).unwrap());
    let folds = mixed_first(&data, 3);
    let held_group = folds.held_out(0).iter().next().unwrap().clone();
    let fitted = records
        .iter()
        .find(|record| !folds.held_out(0).contains(record.group.as_ref().unwrap()))
        .unwrap()
        .clone();
    records.push(LogRecord {
        group: Some(held_group),
        first_position: 9_999,
        ..fitted
    });
    let data = EvalLogs::new(&LogsFrame::from_records(&records).unwrap());
    let held_out = folds.mask(0, data.groups());
    match evaluate_fold(&data, &held_out, &[], &settings(), &mut budget()) {
        Err(L2dError::Runtime { detail }) => {
            assert!(detail.contains("fitted on"), "{detail}")
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| "a result")),
    }
}

#[test]
fn a_fold_repeats_bit_for_bit() {
    let data = synthetic();
    let folds = mixed_first(&data, 3);
    let held_out = folds.mask(1, data.groups());
    let a = evaluate_fold(&data, &held_out, &[], &settings(), &mut budget()).unwrap();
    let b = evaluate_fold(&data, &held_out, &[], &settings(), &mut budget()).unwrap();
    assert!(same(&a, &b));
}

#[test]
fn without_inner_folds_nothing_is_chosen_and_nothing_is_calibrated() {
    let data = synthetic();
    let folds = mixed_first(&data, 3);
    let settings = EvalSettings {
        inner_folds: 0,
        penalties: vec![1.0],
        ..settings()
    };
    let held_out = folds.mask(1, data.groups());
    let result = evaluate_fold(&data, &held_out, &[], &settings, &mut budget()).unwrap();
    assert_eq!(result.inner, None);
    assert!(result.calibration.is_empty());
    assert!(result.chosen().is_none());
    assert_eq!(result.by_penalty.len(), 1);
}

#[test]
fn groups_that_stay_together_share_an_inner_fold() {
    let data = synthetic();
    let together = vec![vec![
        "/tienda/e1.jsp".to_string(),
        "/tienda/e4.jsp".to_string(),
    ]];
    let all = units(data.groups(), &together).unwrap();
    let folds = Folds::dealt(&all, 3, &data.mixed_groups()).unwrap();
    assert_eq!(
        folds.fold_of("/tienda/e1.jsp"),
        folds.fold_of("/tienda/e4.jsp")
    );
    let twin_fold = folds.fold_of("/tienda/e1.jsp").unwrap();
    let other = (twin_fold + 1) % 3;
    let settings = EvalSettings {
        inner_folds: 3,
        ..settings()
    };
    let held_out = folds.mask(other, data.groups());
    let result = evaluate_fold(&data, &held_out, &together, &settings, &mut budget()).unwrap();
    let inner = result.inner.unwrap();
    let holding = |group: &str| inner.folds.iter().position(|fold| fold.contains(group));
    assert!(holding("/tienda/e1.jsp").is_some());
    assert_eq!(holding("/tienda/e1.jsp"), holding("/tienda/e4.jsp"));
}

#[test]
fn settings_that_make_no_sense_are_refused() {
    let data = synthetic();
    let folds = mixed_first(&data, 3);
    let held_out = folds.mask(0, data.groups());
    let invalid = |settings: EvalSettings, mask: &[bool]| {
        matches!(
            evaluate_fold(&data, mask, &[], &settings, &mut budget()),
            Err(L2dError::InvalidInput { .. })
        )
    };
    // The default penalty is not in the grid.
    assert!(invalid(
        EvalSettings {
            penalties: vec![0.1, 10.0],
            ..settings()
        },
        &held_out
    ));
    assert!(invalid(
        EvalSettings {
            penalties: vec![1.0, 1.0],
            ..settings()
        },
        &held_out
    ));
    assert!(invalid(
        EvalSettings {
            inner_folds: 1,
            ..settings()
        },
        &held_out
    ));
    assert!(invalid(
        EvalSettings {
            budget_share: 1.5,
            ..settings()
        },
        &held_out
    ));
    // Nothing held out, everything held out, a mask of another length.
    assert!(invalid(settings(), &vec![false; data.len()]));
    assert!(invalid(settings(), &vec![true; data.len()]));
    assert!(invalid(settings(), &held_out[1..]));
}

#[test]
fn a_fit_that_does_not_fit_the_memory_limit_is_refused_before_it_runs() {
    let data = synthetic();
    let folds = mixed_first(&data, 3);
    let held_out = folds.mask(0, data.groups());
    // 33 MiB: 1 MiB beside what the tool keeps for itself. A fit with 65,536 buckets
    // needs about 25 MiB for the solver alone.
    let settings = EvalSettings {
        baseline: BaselineSettings {
            routing: RoutingMode::Strip,
            ..BaselineSettings::default()
        },
        ..settings()
    };
    let mut small = budget_with_limit(Some(33 * 1024 * 1024));
    match evaluate_fold(&data, &held_out, &[], &settings, &mut small) {
        Err(L2dError::Refused { reason }) => assert!(reason.contains("memory"), "{reason}"),
        other => panic!("expected a refusal, got {:?}", other.map(|_| "a result")),
    }
    assert_eq!(small.held_bytes(), 0, "the budget is given back as it was");
}

// ---------------------------------------------------------------------------------------
// All folds of an arrangement
// ---------------------------------------------------------------------------------------

fn arrangement(data: &EvalLogs) -> Arrangement {
    Arrangement {
        name: "mixed-first".into(),
        rule: "test".into(),
        folds: mixed_first(data, 3),
        together: Vec::new(),
    }
}

#[test]
fn every_log_is_held_out_in_exactly_one_fold_and_workers_change_nothing() {
    let data = synthetic();
    let arrangements = [arrangement(&data)];
    let modes = [
        settings(),
        EvalSettings {
            baseline: BaselineSettings {
                buckets: 1_024,
                routing: RoutingMode::Keep,
                ..BaselineSettings::default()
            },
            ..settings()
        },
    ];
    let one = evaluate(&data, &arrangements, &modes, 1, &budget()).unwrap();
    let three = evaluate(&data, &arrangements, &modes, 3, &budget()).unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one[0].modes.len(), 2);
    for (a, b) in one[0].modes.iter().zip(&three[0].modes) {
        assert_eq!(a.folds.len(), 3);
        for (x, y) in a.folds.iter().zip(&b.folds) {
            assert!(same(x, y), "the number of workers changed a result");
        }
        let mut times = vec![0_u32; data.len()];
        for fold in &a.folds {
            for row in &fold.held_rows {
                times[*row as usize] += 1;
            }
        }
        assert!(times.iter().all(|&count| count == 1));
    }
    // The two routing modes are different scorers.
    assert_ne!(
        one[0].modes[0].folds[0].by_penalty[0].ratios,
        one[0].modes[1].folds[0].by_penalty[0].ratios
    );
    assert!(matches!(
        evaluate(&data, &arrangements, &modes, 0, &budget()),
        Err(L2dError::InvalidInput { .. })
    ));
}

#[test]
fn the_summary_over_folds_states_each_way_of_pooling() {
    let data = synthetic();
    let arrangements = [arrangement(&data)];
    let modes = [settings()];
    let results = evaluate(&data, &arrangements, &modes, 2, &budget()).unwrap();
    let folds = &results[0].modes[0].folds;
    let summary = rule_summary(&data, folds, &modes[0], |fold| fold.default_penalty());

    // Per fold: the fold's own cross-endpoint figure.
    let per_fold: Vec<f64> = folds
        .iter()
        .map(|fold| {
            fold.default_penalty()
                .unwrap()
                .metrics
                .cross_endpoint
                .roc_auc
                .unwrap()
        })
        .collect();
    assert_eq!(
        summary.fold_roc_auc,
        per_fold.iter().map(|v| Some(*v)).collect::<Vec<_>>()
    );
    let spread = summary.roc_auc.unwrap();
    assert_eq!(spread.count, 3);
    close(spread.mean, per_fold.iter().sum::<f64>() / 3.0);
    close(spread.min, per_fold.iter().copied().fold(1.0, f64::min));
    assert_eq!(summary.penalties, vec![Some(1.0); 3]);

    // Pooled over all folds: the held-out logs of the six mixed endpoints, once each.
    assert_eq!(
        (summary.pooled.logs, summary.pooled.good, summary.pooled.bad),
        (300, 180, 120)
    );
    assert_eq!(summary.pooled_overall.logs as usize, data.len());
    // Per endpoint: each of the six, from the fold that held it out.
    assert_eq!(summary.per_endpoint.len(), 6);
    let endpoint_mean = summary
        .per_endpoint
        .values()
        .map(|block| block.roc_auc.unwrap())
        .sum::<f64>()
        / 6.0;
    close(summary.endpoint_roc_auc.unwrap().mean, endpoint_mean);
    assert_eq!(summary.endpoint_roc_auc.unwrap().count, 6);
    // Single-label groups: 10 bad-only with 2 logs each, 2 good-only with 1.
    assert_eq!(
        (
            summary.single_label.groups,
            summary.single_label.logs,
            summary.single_label.good,
            summary.single_label.bad
        ),
        (12, 22, 2, 20)
    );

    // The calibrated decisions of all folds: every held-out log is counted once, and the
    // parts add up to the whole.
    for chosen in [false, true] {
        let calibration = calibration_summary(folds, chosen);
        assert_eq!(calibration.folds, 3);
        assert!(calibration.refused.is_empty());
        let overall = &calibration.overall;
        assert_eq!(overall.logs, [180 + 2, 120 + 20], "good and bad logs");
        assert_eq!(overall.sets.iter().sum::<u64>(), data.len() as u64);
        assert_eq!(calibration.cross_endpoint.logs, [180, 120]);
        assert_eq!(calibration.single_label.logs, [2, 20]);
        assert_eq!(calibration.per_group.len(), 6);
        let per_fold: u64 = calibration
            .fold_overall
            .iter()
            .flatten()
            .map(|coverage| coverage.missed[0] + coverage.missed[1])
            .sum();
        assert_eq!(per_fold, overall.missed[0] + overall.missed[1]);
        let rate = overall.error_rate(0).unwrap();
        close(rate, overall.missed[0] as f64 / 182.0);
    }

    // The chosen penalty has a summary of its own.
    let chosen = rule_summary(&data, folds, &modes[0], |fold| fold.chosen());
    assert_eq!(chosen.fold_roc_auc.len(), 3);
    for (fold, penalty) in folds.iter().zip(&chosen.penalties) {
        assert_eq!(*penalty, Some(fold.inner.as_ref().unwrap().chosen));
    }
}

#[test]
fn the_predictions_frame_has_the_columns_of_the_design() {
    let data = synthetic();
    let arrangements = [arrangement(&data)];
    let modes = [settings()];
    let results = evaluate(&data, &arrangements, &modes, 2, &budget()).unwrap();
    let folds = &results[0].modes[0].folds;
    let frame = predictions_frame(&data, folds, |fold| fold.default_penalty()).unwrap();
    assert_eq!(frame.height(), data.len());
    let names: Vec<String> = frame
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();
    assert_eq!(
        names,
        [
            "text",
            "label",
            "group",
            "fold",
            "scorer",
            "ratio",
            "label_set"
        ]
    );
    let fold_column = frame.column("fold").unwrap().u32().unwrap();
    let per_fold: BTreeMap<u32, usize> =
        fold_column
            .into_no_null_iter()
            .fold(BTreeMap::new(), |mut counts, fold| {
                *counts.entry(fold).or_default() += 1;
                counts
            });
    assert_eq!(
        per_fold.values().copied().collect::<Vec<_>>(),
        folds
            .iter()
            .map(|fold| fold.held_rows.len())
            .collect::<Vec<_>>()
    );
    assert_eq!(frame.column("ratio").unwrap().null_count(), 0);
    assert_eq!(frame.column("label_set").unwrap().null_count(), 0);
}

// ---------------------------------------------------------------------------------------
// The mini fixture, end to end, with the report
// ---------------------------------------------------------------------------------------

fn fixture() -> (LogsFrame, InputSummary) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv");
    LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&path)
        .expect("the fixture is readable")
}

/// A directory of this test under the system's temporary directory, empty.
fn temporary(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("l2d-eval-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_fixture_is_evaluated_end_to_end_and_the_report_repeats_byte_for_byte() {
    let (logs, summary) = fixture();
    let data = EvalLogs::new(&logs);
    // 169 distinct texts, one of them with both labels: left out.
    assert_eq!(data.len(), 168);
    assert_eq!(data.unknown_left_out(), 1);
    assert_eq!(data.mixed_groups().len(), 8);

    let all = units(data.groups(), &[]).unwrap();
    let arrangements = [Arrangement {
        name: "balanced-4".into(),
        rule: "four balanced folds by group key".into(),
        folds: Folds::balanced(&all, 4).unwrap(),
        together: Vec::new(),
    }];
    // The fixture's POST rows put a space before the body, so only keep makes sense.
    let modes = [EvalSettings {
        baseline: BaselineSettings {
            buckets: 4_096,
            routing: RoutingMode::Keep,
            ..BaselineSettings::default()
        },
        penalties: vec![0.1, 1.0, 10.0],
        inner_folds: 3,
        error_rate: 0.2,
        calibration: CalibrationSettings::default(),
        budget_share: 0.25,
        small_endpoint: 5,
    }];
    let run = || evaluate(&data, &arrangements, &modes, 2, &budget()).unwrap();
    let results = run();

    let folds = &results[0].modes[0].folds;
    assert_eq!(folds.len(), 4);
    let mut times = vec![0_u32; data.len()];
    for (index, fold) in folds.iter().enumerate() {
        assert_eq!(fold.held_rows.len(), 42, "two groups of 21 texts");
        for row in &fold.held_rows {
            times[*row as usize] += 1;
            assert!(
                arrangements[0]
                    .folds
                    .held_out(index)
                    .contains(&data.groups()[*row as usize])
            );
        }
        assert_eq!(fold.fitting_logs, 126);
        assert_eq!(fold.calibration[0].refused, None);
    }
    assert!(times.iter().all(|&count| count == 1));
    let summary_default = rule_summary(&data, folds, &modes[0], |fold| fold.default_penalty());
    let auc = summary_default.roc_auc.unwrap();
    assert_eq!(auc.count, 4);
    assert!(
        auc.mean > 0.5,
        "the baseline ranks the fixture's attacks no better than chance: {auc:?}"
    );
    assert_eq!(summary_default.pooled.logs, 168);
    assert_eq!(summary_default.single_label.logs, 0);

    // The report: JSON and a readable summary, the same bytes on a second run.
    let report = |results| Report {
        task: "test".into(),
        date: "2026-11-13".into(),
        dataset: Dataset {
            name: "mini-labelled".into(),
            summary: summary.clone(),
        },
        data: &data,
        arrangements: &arrangements,
        modes: &modes,
        results,
        notes: serde_json::json!({ "note": "fixture" }),
    };
    let dir = temporary("fixture-report");
    let (json_path, text_path) = write_report(&dir, "first", &report(&results)).unwrap();
    let again = run();
    let (json_again, text_again) = write_report(&dir, "second", &report(&again)).unwrap();
    let json = std::fs::read(&json_path).unwrap();
    assert_eq!(json, std::fs::read(&json_again).unwrap());
    assert_eq!(
        std::fs::read(&text_path).unwrap(),
        std::fs::read(&text_again).unwrap()
    );
    assert_eq!(json_path.extension().unwrap(), "json");
    assert_eq!(text_path.extension().unwrap(), "md");

    // Requirement 9.6: the dataset and its hash, the settings, the date; and the fold
    // membership.
    let value: serde_json::Value = serde_json::from_slice(&json).unwrap();
    assert_eq!(
        String::from_utf8(json.clone()).unwrap(),
        serde_json::to_string_pretty(&report_json(&report(&results))).unwrap() + "\n"
    );
    assert_eq!(value["date"], "2026-11-13");
    assert_eq!(value["dataset"]["name"], "mini-labelled");
    assert_eq!(value["dataset"]["sha256"], summary.content_sha256.as_str());
    assert_eq!(value["dataset"]["distinct_texts"], 169);
    assert_eq!(value["dataset"]["evaluated_logs"], 168);
    assert_eq!(value["dataset"]["unlabelled_left_out"], 1);
    let arrangement = &value["arrangements"][0];
    assert_eq!(arrangement["name"], "balanced-4");
    let membership: usize = arrangement["folds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|fold| fold["held_out_groups"].as_array().unwrap().len())
        .sum();
    assert_eq!(membership, 8);
    assert_eq!(
        arrangement["folds"][0]["held_out_sha256"],
        arrangements[0].folds.membership_sha256(0).as_str()
    );
    let mode = &arrangement["modes"][0];
    assert_eq!(mode["settings"]["routing"], "keep");
    assert_eq!(
        mode["settings"]["penalties"],
        serde_json::json!([0.1, 1.0, 10.0])
    );
    assert_eq!(mode["settings"]["error_rate"].as_f64().unwrap() as f32, 0.2);
    for rule in ["default", "chosen_inside_fold"] {
        let summary = &mode["summary"][rule];
        assert_eq!(
            summary["cross_endpoint"]["per_fold_roc_auc"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert!(summary["cross_endpoint"]["roc_auc"]["mean"].is_number());
        assert!(summary["calibration"]["good"]["achieved_error_rate"].is_number());
    }
    let second = &mode["summary"]["chosen_by_inner_roc_auc"];
    assert_eq!(
        second["cross_endpoint"]["per_fold_roc_auc"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert!(second["calibration"].is_null());
    assert_eq!(
        mode["summary"]["sensitivity"].as_array().unwrap().len(),
        3,
        "one entry per penalty of the grid"
    );
    // No timing in the report: it would not repeat.
    let text = String::from_utf8(json).unwrap();
    assert!(!text.contains("seconds"));
    let markdown = report_markdown(&report(&results));
    assert!(markdown.contains(&summary.content_sha256));
    assert!(markdown.contains("balanced-4"));
    assert!(markdown.contains("2026-11-13"));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_date_of_a_moment_is_its_calendar_day() {
    use std::time::{Duration, UNIX_EPOCH};
    assert_eq!(date_of(UNIX_EPOCH), "1970-01-01");
    // 2026-11-13 00:00:00 UTC is 20,770 days after the epoch.
    let day = Duration::from_secs(20_770 * 86_400);
    assert_eq!(date_of(UNIX_EPOCH + day), "2026-11-13");
    assert_eq!(
        date_of(UNIX_EPOCH + day + Duration::from_secs(86_399)),
        "2026-11-13"
    );
    assert_eq!(
        date_of(UNIX_EPOCH + day + Duration::from_secs(86_400)),
        "2026-11-14"
    );
    // A leap day: 2024-02-29 is day 19,782.
    assert_eq!(
        date_of(UNIX_EPOCH + Duration::from_secs(19_782 * 86_400)),
        "2024-02-29"
    );
}
