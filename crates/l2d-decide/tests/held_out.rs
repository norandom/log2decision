//! The held-out check per log (task 1.11, requirement 11.1): calibration logs whose
//! text the baseline scorer was fitted on are left out and counted; a scorer saved
//! without the hashes of its fitted texts falls back to the whole-file check.

use std::path::{Path, PathBuf};

use l2d_baseline::{BaselineScorer, BaselineSettings};
use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LogRecord,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_decide::conformal::{HELD_OUT_ROW_BYTES, HeldOutCheck, WHOLE_FILE_NOTE, held_out_logs};
use l2d_decide::{CalibrationError, CalibrationSettings, ConformalCalibrator};

const GIB: u64 = 1024 * 1024 * 1024;

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB)).unwrap()
}

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
        .unwrap()
}

/// A summary as an input with this content hash would have.
fn summary(name: &str, hash_digit: char, logs: &LogsFrame) -> InputSummary {
    InputSummary {
        source_path: PathBuf::from(name),
        records: logs.len() as u64,
        distinct_texts: logs.len() as u64,
        content_sha256: hash_digit.to_string().repeat(64),
        skipped: Vec::new(),
        label_conflicts: 0,
    }
}

/// The fixture cut in two by row: even rows to fit on, odd rows held out.
fn parts() -> (Vec<LogRecord>, Vec<LogRecord>) {
    let (logs, _) = fixture();
    let mut fit = Vec::new();
    let mut held_out = Vec::new();
    for (index, record) in logs.records().enumerate() {
        if index % 2 == 0 {
            fit.push(record);
        } else {
            held_out.push(record);
        }
    }
    (fit, held_out)
}

fn small() -> BaselineSettings {
    BaselineSettings {
        buckets: 1_024,
        ..BaselineSettings::default()
    }
}

fn fitted(records: &[LogRecord]) -> BaselineScorer {
    let logs = LogsFrame::from_records(records).unwrap();
    BaselineScorer::fit_with(
        &logs,
        &[summary("fit.csv", 'a', &logs)],
        &small(),
        &budget(),
    )
    .unwrap()
}

#[test]
fn calibration_logs_the_scorer_was_fitted_on_are_left_out_and_counted() {
    let (fit, held_out) = parts();
    let scorer = fitted(&fit);

    // The calibration file: every held-out log, with ten fitting logs mixed in (a
    // reordered subset of the fitting file under other bytes passes the whole-file check).
    let mut mixed = Vec::new();
    let fitted_labelled: Vec<&LogRecord> = fit
        .iter()
        .filter(|record| record.label != Label::Unknown)
        .collect();
    for (index, record) in held_out.iter().enumerate() {
        if index % 8 == 0 && index / 8 < 10 {
            mixed.push(fitted_labelled[index / 8 * 3].clone());
        }
        mixed.push(record.clone());
    }
    let logs = LogsFrame::from_records(&mixed).unwrap();
    let inputs = [summary("calibration.csv", 'b', &logs)];

    let (kept, report) = held_out_logs(&scorer, &logs, &inputs, &budget()).unwrap();
    assert_eq!(report.check, HeldOutCheck::PerLog);
    assert_eq!(report.logs, mixed.len() as u64);
    assert_eq!(report.dropped, 10);
    assert_eq!(report.kept, held_out.len() as u64);
    assert_eq!(report.note(), None);
    assert_eq!(
        kept.to_records(),
        held_out,
        "the held-out logs, in their order"
    );

    // The calibration is the one of the held-out logs alone.
    let settings = CalibrationSettings { min_per_group: 5 };
    let calibrated =
        ConformalCalibrator::fit_baseline(&scorer, &logs, &inputs, &settings, &budget()).unwrap();
    let clean = LogsFrame::from_records(&held_out).unwrap();
    let expected = ConformalCalibrator::fit_baseline(
        &scorer,
        &clean,
        &[summary("clean.csv", 'c', &clean)],
        &settings,
        &budget(),
    )
    .unwrap();
    assert_eq!(calibrated.revision(), expected.revision());
    let labelled = held_out
        .iter()
        .filter(|record| record.label != Label::Unknown)
        .count() as u64;
    assert_eq!(calibrated.good() + calibrated.bad(), labelled);
}

#[test]
fn logs_that_are_all_held_out_come_back_unchanged() {
    let (fit, held_out) = parts();
    let scorer = fitted(&fit);
    let logs = LogsFrame::from_records(&held_out).unwrap();
    let (kept, report) =
        held_out_logs(&scorer, &logs, &[summary("c.csv", 'b', &logs)], &budget()).unwrap();
    assert_eq!(kept, logs);
    assert_eq!(
        (report.check, report.logs, report.dropped, report.kept),
        (
            HeldOutCheck::PerLog,
            held_out.len() as u64,
            0,
            held_out.len() as u64
        )
    );
}

#[test]
fn a_log_the_scorer_only_saw_unlabelled_is_not_a_fitted_log() {
    let (mut fit, held_out) = parts();
    // The fit ignores rows labelled unknown, so their texts stay usable for calibration.
    let mut unseen = held_out[0].clone();
    unseen.label = Label::Unknown;
    fit.push(unseen);
    let scorer = fitted(&fit);
    let logs = LogsFrame::from_records(&held_out).unwrap();
    let (_, report) =
        held_out_logs(&scorer, &logs, &[summary("c.csv", 'b', &logs)], &budget()).unwrap();
    assert_eq!(report.dropped, 0);
}

#[test]
fn the_fitting_file_itself_is_still_refused_as_a_whole() {
    let (fit, _) = parts();
    let scorer = fitted(&fit);
    let logs = LogsFrame::from_records(&fit).unwrap();
    let error =
        held_out_logs(&scorer, &logs, &[summary("fit.csv", 'a', &logs)], &budget()).unwrap_err();
    assert!(
        matches!(error, CalibrationError::SameInputAsFitting { .. }),
        "{error:?}"
    );
    // The same logs under other bytes: every log is left out, and the calibration is
    // refused for want of logs.
    let inputs = [summary("reordered.csv", 'b', &logs)];
    let (kept, report) = held_out_logs(&scorer, &logs, &inputs, &budget()).unwrap();
    let unknown = fit
        .iter()
        .filter(|record| record.label == Label::Unknown)
        .count() as u64;
    assert_eq!(report.dropped, fit.len() as u64 - unknown);
    assert_eq!(kept.len() as u64, unknown);
    let error = ConformalCalibrator::fit_baseline(
        &scorer,
        &logs,
        &inputs,
        &CalibrationSettings::default(),
        &budget(),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            CalibrationError::TooFewLogs {
                good: 0,
                bad: 0,
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn a_scorer_saved_without_text_hashes_falls_back_to_the_whole_file_check() {
    // The scorer of the fixture of task 1.11 in `l2d-baseline`: saved before the hashes
    // existed, fitted on the whole mini fixture.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("l2d-baseline")
        .join("tests")
        .join("fixtures")
        .join("scorer-1.7");
    let revision = std::fs::read_to_string(root.join("revision.txt")).unwrap();
    let dir = ProjectPaths::new(&root).baseline_dir("before-1.11", revision.trim());
    let scorer = BaselineScorer::load(&dir).unwrap();
    assert!(!scorer.has_fitted_text_hashes());

    let (logs, real) = fixture();
    // Byte for byte the fitting file: refused.
    let error = held_out_logs(&scorer, &logs, std::slice::from_ref(&real), &budget()).unwrap_err();
    assert!(
        matches!(error, CalibrationError::SameInputAsFitting { .. }),
        "{error:?}"
    );
    // The same logs under other bytes: not detected, and the report says so.
    let (kept, report) = held_out_logs(
        &scorer,
        &logs,
        &[summary("copy.csv", 'b', &logs)],
        &budget(),
    )
    .unwrap();
    assert_eq!(kept, logs);
    assert_eq!(report.check, HeldOutCheck::WholeFile);
    assert_eq!((report.dropped, report.kept), (0, logs.len() as u64));
    assert_eq!(report.note(), Some(WHOLE_FILE_NOTE));
    assert!(WHOLE_FILE_NOTE.contains("identical"), "{WHOLE_FILE_NOTE}");
    assert_eq!(
        serde_json::to_value(report.check).unwrap(),
        serde_json::json!("whole_file")
    );
    assert_eq!(
        serde_json::to_value(HeldOutCheck::PerLog).unwrap(),
        serde_json::json!("per_log")
    );
}

#[test]
fn the_check_asks_the_memory_budget_before_it_copies_rows() {
    let (fit, held_out) = parts();
    let scorer = fitted(&fit);
    let logs = LogsFrame::from_records(&held_out).unwrap();
    let inputs = [summary("c.csv", 'b', &logs)];
    let needed = logs.len() as u64 * HELD_OUT_ROW_BYTES;
    let with_limit = |limit: u64| {
        let settings = MemorySettings {
            memory_limit_bytes: Some(limit),
            ..MemorySettings::default()
        };
        MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).unwrap()
    };
    assert!(
        held_out_logs(
            &scorer,
            &logs,
            &inputs,
            &with_limit(PROCESS_BASELINE_BYTES + needed)
        )
        .is_ok()
    );
    let error = held_out_logs(
        &scorer,
        &logs,
        &inputs,
        &with_limit(PROCESS_BASELINE_BYTES + needed - 1),
    )
    .unwrap_err();
    match error {
        CalibrationError::Memory(L2dError::Refused { reason }) => {
            assert!(reason.contains(&format!("({needed} bytes)")), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn hashes_that_cover_only_part_of_the_fitting_inputs_are_reported_as_partial() {
    use l2d_decide::conformal::PARTIAL_NOTE;

    let (fit, held_out) = parts();
    let scorer = fitted(&fit);

    // A directory in the earlier form that names another fitting input, then the same
    // model saved over it: the hashes cover this fit only.
    let root = std::env::temp_dir().join(format!("l2d-held-out-partial-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let paths = ProjectPaths::new(&root);
    let dir = scorer.save(&paths, "partial").unwrap();
    std::fs::remove_file(dir.join("fitted-texts.bin")).unwrap();
    let manifest_path = dir.join("baseline.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["format"] = 1.into();
    manifest.as_object_mut().unwrap().remove("fitted_texts");
    manifest["fitting_inputs"][0]["content_sha256"] = "e".repeat(64).into();
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    scorer.save(&paths, "partial").unwrap();
    let merged = BaselineScorer::load(&dir).unwrap();
    assert_eq!(merged.fitted_text_hashes_complete(), Some(false));

    // The check per log still leaves out what it knows, and says what it cannot know.
    let mut mixed = held_out.clone();
    mixed.push(
        fit.iter()
            .find(|record| record.label != Label::Unknown)
            .unwrap()
            .clone(),
    );
    let logs = LogsFrame::from_records(&mixed).unwrap();
    let (kept, report) =
        held_out_logs(&merged, &logs, &[summary("c.csv", 'b', &logs)], &budget()).unwrap();
    assert_eq!(report.check, HeldOutCheck::PerLogPartial);
    assert_eq!((report.dropped, report.kept), (1, held_out.len() as u64));
    assert_eq!(kept.to_records(), held_out);
    assert_eq!(report.note(), Some(PARTIAL_NOTE));
    assert!(PARTIAL_NOTE.contains("part"), "{PARTIAL_NOTE}");
    assert_eq!(
        serde_json::to_value(report.check).unwrap(),
        serde_json::json!("per_log_partial")
    );
    // The earlier input itself is still refused as a whole.
    let error =
        held_out_logs(&merged, &logs, &[summary("e.csv", 'e', &logs)], &budget()).unwrap_err();
    assert!(
        matches!(error, CalibrationError::SameInputAsFitting { .. }),
        "{error:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
