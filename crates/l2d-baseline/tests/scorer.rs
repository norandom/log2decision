//! The baseline scorer (task 1.7): fit, ratio, scores frame, refusals, determinism,
//! memory check, save and load.
//!
//! Requirements 2.1 to 2.5, 10.5, 11.1 (fitting-input hashes), 12.5.
//!
//! Fixture fits use routing mode keep: the POST rows of the mini fixture have a space
//! before the body, so strip would leave them without features.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_baseline::scorer::{
    ESTIMATE_SAMPLE_ROWS, FITTED_TEXTS_FILE, MANIFEST_FILE, SCORER_KIND, SCORES_ABSTAINED,
    SCORES_NO_FEATURES, SCORES_RATIO, SCORES_SCORER, SCORES_TEXT, WEIGHTS_FILE,
};
use l2d_baseline::{
    BaselineScorer, BaselineSettings, CharHasher, FitError, RoutingMode, Termination,
};
use l2d_core::input::content_sha256;
use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::polars::prelude::{DataFrame, DataType};
use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LogRecord,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

const GIB: u64 = 1024 * 1024 * 1024;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv")
}

/// The mini fixture: 169 distinct texts, 96 good, 72 bad, 1 unknown.
fn fixture() -> &'static (LogsFrame, InputSummary) {
    static FIXTURE: OnceLock<(LogsFrame, InputSummary)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        LabelledAdapter::new(LabelledFormat::Csv)
            .with_group_column("group")
            .read(&fixture_path())
            .expect("the fixture is readable")
    })
}

/// A budget with a fixed free-memory reading, so the tests do not depend on the host.
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

/// Settings with a small feature space, for tests of the mechanics.
fn small() -> BaselineSettings {
    BaselineSettings {
        buckets: 4_096,
        ..BaselineSettings::default()
    }
}

fn fit(logs: &LogsFrame, settings: &BaselineSettings) -> BaselineScorer {
    BaselineScorer::fit_with(logs, &[], settings, &budget()).expect("the fit succeeds")
}

/// The fixture fitted with the default settings (65,536 buckets, keep).
fn fitted() -> &'static BaselineScorer {
    static FITTED: OnceLock<BaselineScorer> = OnceLock::new();
    FITTED.get_or_init(|| {
        let (logs, summary) = fixture();
        BaselineScorer::fit_with(
            logs,
            std::slice::from_ref(summary),
            &BaselineSettings::default(),
            &budget(),
        )
        .expect("the fixture fits")
    })
}

/// The fixture fitted with [`small`] settings.
fn fitted_small() -> &'static BaselineScorer {
    static FITTED: OnceLock<BaselineScorer> = OnceLock::new();
    FITTED.get_or_init(|| {
        let (logs, summary) = fixture();
        BaselineScorer::fit_with(logs, std::slice::from_ref(summary), &small(), &budget())
            .expect("the fixture fits")
    })
}

/// A project root in the system temporary directory, removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-baseline-scorer-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("the scratch root can be created");
        Self { root }
    }

    fn paths(&self) -> ProjectPaths {
        ProjectPaths::new(&self.root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn record(text: String, label: Label) -> LogRecord {
    LogRecord {
        text,
        label,
        group: None,
        first_position: 0,
        occurrences: 1,
    }
}

/// Distinct synthetic request texts: `good` good, `bad` bad and `unknown` unlabelled,
/// interleaved in a fixed order.
fn synthetic(good: usize, bad: usize, unknown: usize) -> LogsFrame {
    let mut records = Vec::new();
    for index in 0..good.max(bad).max(unknown) {
        if index < good {
            records.push(record(
                format!("GET http://localhost:8080/tienda1/index.jsp?id={index}&lang=es"),
                Label::Good,
            ));
        }
        if index < bad {
            records.push(record(
                format!("GET http://localhost:8080/tienda1/index.jsp?id={index}' OR '1'='1"),
                Label::Bad,
            ));
        }
        if index < unknown {
            records.push(record(
                format!("GET http://localhost:8080/tienda1/index.jsp?id={index}&x=<script>"),
                Label::Unknown,
            ));
        }
    }
    LogsFrame::from_records(&records).expect("the records form a logs frame")
}

fn texts_of(logs: &LogsFrame) -> Vec<String> {
    logs.records().map(|record| record.text).collect()
}

fn ratio_bits(scorer: &BaselineScorer, texts: &[String]) -> Vec<u32> {
    texts
        .iter()
        .map(|text| scorer.ratio(text).to_bits())
        .collect()
}

fn weight_bits(scorer: &BaselineScorer) -> Vec<u64> {
    scorer
        .weights()
        .iter()
        .map(|weight| weight.to_bits())
        .chain([scorer.bias().to_bits()])
        .collect()
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Area under the ROC curve: the share of (bad, good) pairs in which the bad log has the
/// higher ratio, ties counting one half.
fn roc_auc(scored: &[(f32, bool)]) -> f64 {
    let mut wins = 0.0;
    let mut pairs = 0.0;
    for (bad_ratio, _) in scored.iter().filter(|(_, bad)| *bad) {
        for (good_ratio, _) in scored.iter().filter(|(_, bad)| !*bad) {
            pairs += 1.0;
            if bad_ratio > good_ratio {
                wins += 1.0;
            } else if bad_ratio == good_ratio {
                wins += 0.5;
            }
        }
    }
    wins / pairs
}

fn ratios(frame: &DataFrame) -> Vec<f32> {
    frame
        .column(SCORES_RATIO)
        .unwrap()
        .f32()
        .unwrap()
        .iter()
        .map(|ratio| ratio.expect("the baseline never leaves a ratio empty"))
        .collect()
}

fn refusal(error: FitError) -> String {
    match L2dError::from(error) {
        L2dError::Refused { reason } => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// Fitting on the fixture (2.1, 2.2, 2.5)
// ---------------------------------------------------------------------------------------

#[test]
fn the_fixture_fits_and_scores_without_runtime_files() {
    // A project root that has no runtime directory at all: no AMD files, no encoder,
    // no store. The scorer needs none of them to fit, score, save and load (2.5).
    let scratch = Scratch::new("no-runtime");
    let paths = scratch.paths();
    assert!(!paths.runtime_dir().exists());
    assert!(!paths.ryzenai_dir().exists());

    let (logs, summary) = fixture();
    let scorer = BaselineScorer::fit_with(logs, std::slice::from_ref(summary), &small(), &budget())
        .expect("the fixture fits");
    let scores = scorer.score(logs).expect("the fixture is scored");
    assert_eq!(scores.height(), 169);

    let dir = scorer.save(&paths, "mini").expect("the scorer is saved");
    let loaded = BaselineScorer::load(&dir).expect("the scorer loads");
    assert_eq!(loaded.revision(), scorer.revision());
    assert!(
        !paths.runtime_dir().exists(),
        "nothing created a runtime directory"
    );
    assert!(!paths.encoders_dir().exists());
    assert!(!paths.stores_dir().exists());
}

#[test]
fn the_fit_report_states_the_counts_and_how_the_fit_ended() {
    let scorer = fitted();
    let report = scorer.fit_report();
    assert_eq!((report.rows, report.good, report.bad), (168, 96, 72));
    assert_eq!(report.unknown_ignored, 1);
    assert_eq!(report.no_feature_rows, 0);
    assert!(report.stored_entries > 168 * 20);
    assert!(report.objective.is_finite() && report.objective > 0.0);
    assert!(report.gradient_norm.is_finite());
    assert!(report.iterations >= 1);
    println!(
        "L2D_SCORER_FIXTURE_FIT default settings: iterations={} termination={:?} \
         gradient_norm={:.3e} objective={:.4} entries={} estimated_bytes={}",
        report.iterations,
        report.termination,
        report.gradient_norm,
        report.objective,
        report.stored_entries,
        report.estimated_bytes
    );
    // The default settings converge by the gradient at this size.
    assert_eq!(report.termination, Termination::Converged);
    assert!(scorer.gradient_converged());
    assert_eq!(scorer.weights().len(), 65_536);
}

#[test]
fn ratios_lie_between_zero_and_one_and_bad_texts_score_higher() {
    let (logs, _) = fixture();
    let scorer = fitted();
    let mut good = Vec::new();
    let mut bad = Vec::new();
    for record in logs.records() {
        let ratio = scorer.ratio(&record.text);
        assert!(
            (0.0..=1.0).contains(&ratio),
            "{ratio} for {:?}",
            record.text
        );
        match record.label {
            Label::Good => good.push(ratio),
            Label::Bad => bad.push(ratio),
            Label::Unknown => {}
        }
    }
    let mean =
        |ratios: &[f32]| ratios.iter().map(|r| f64::from(*r)).sum::<f64>() / ratios.len() as f64;
    println!(
        "L2D_SCORER_FIXTURE_TRAIN mean ratio: good {:.4} ({} texts), bad {:.4} ({} texts)",
        mean(&good),
        good.len(),
        mean(&bad),
        bad.len()
    );
    assert!(mean(&bad) > mean(&good) + 0.2);
    // Texts the model has never seen, of any shape, stay in range too.
    for text in ["", "?", "x", "\n", "' OR '1'='1", &"A".repeat(5_000)] {
        let ratio = scorer.ratio(text);
        assert!((0.0..=1.0).contains(&ratio), "{ratio} for {text:?}");
    }
}

#[test]
fn held_out_groups_of_the_fixture_are_ranked_clearly_better_than_chance() {
    let (logs, _) = fixture();
    let records = logs.to_records();
    let groups: BTreeSet<&str> = records.iter().filter_map(|r| r.group.as_deref()).collect();
    assert_eq!(groups.len(), 8);
    // Hold out every fourth group key (2 of the 8 endpoints); fit on the other six.
    let held: BTreeSet<&str> = groups.iter().copied().skip(3).step_by(4).collect();
    assert_eq!(held.len(), 2);
    let is_held = |record: &LogRecord| record.group.as_deref().is_some_and(|g| held.contains(g));

    let fitting: Vec<LogRecord> = records.iter().filter(|r| !is_held(r)).cloned().collect();
    let scorer = fit(
        &LogsFrame::from_records(&fitting).unwrap(),
        &BaselineSettings::default(),
    );
    let scored: Vec<(f32, bool)> = records
        .iter()
        .filter(|r| is_held(r) && r.label != Label::Unknown)
        .map(|r| (scorer.ratio(&r.text), r.label == Label::Bad))
        .collect();
    let bad = scored.iter().filter(|(_, bad)| *bad).count();
    let good = scored.len() - bad;
    let auc = roc_auc(&scored);
    let correct = scored
        .iter()
        .filter(|(ratio, bad)| (*ratio > 0.5) == *bad)
        .count();
    println!(
        "L2D_SCORER_FIXTURE_HELD_OUT groups={held:?} fitted_on={} held_out={} ({good} good, \
         {bad} bad) roc_auc={auc:.4} accuracy_at_0.5={:.4}",
        scorer.fit_report().rows,
        scored.len(),
        correct as f64 / scored.len() as f64
    );
    assert!(good >= 10 && bad >= 10);
    // Chance is 0.5. The bound is far below what is measured, so it is not brittle.
    assert!(auc > 0.8, "ROC AUC {auc} on held-out groups");
}

// ---------------------------------------------------------------------------------------
// Refusals (2.3)
// ---------------------------------------------------------------------------------------

#[test]
fn nine_labelled_logs_are_refused_with_the_counts() {
    // 5 good and 4 bad: nine labelled, and unknown rows do not count.
    let logs = synthetic(5, 4, 30);
    let error = BaselineScorer::fit_with(&logs, &[], &small(), &budget()).unwrap_err();
    assert_eq!(error, FitError::TooFewLabels { good: 5, bad: 4 });
    let reason = refusal(error);
    assert!(
        reason.contains("found 9 labelled (5 good, 4 bad)"),
        "{reason}"
    );
    assert!(reason.contains("at least 10 labelled logs"), "{reason}");

    let error =
        BaselineScorer::fit_with(&synthetic(9, 0, 0), &[], &small(), &budget()).unwrap_err();
    assert_eq!(error, FitError::TooFewLabels { good: 9, bad: 0 });
}

#[test]
fn ten_labelled_logs_with_four_of_one_label_are_refused_with_the_counts() {
    let error =
        BaselineScorer::fit_with(&synthetic(6, 4, 0), &[], &small(), &budget()).unwrap_err();
    assert_eq!(error, FitError::TooFewLabels { good: 6, bad: 4 });
    let reason = refusal(error);
    assert!(
        reason.contains("found 10 labelled (6 good, 4 bad)"),
        "{reason}"
    );
    assert!(reason.contains("at least 5 of each label"), "{reason}");

    let error =
        BaselineScorer::fit_with(&synthetic(4, 40, 0), &[], &small(), &budget()).unwrap_err();
    assert_eq!(error, FitError::TooFewLabels { good: 4, bad: 40 });
    assert_eq!(L2dError::from(error).exit_code(), 4);

    let error =
        BaselineScorer::fit_with(&LogsFrame::empty(), &[], &small(), &budget()).unwrap_err();
    assert_eq!(error, FitError::TooFewLabels { good: 0, bad: 0 });
}

#[test]
fn the_smallest_accepted_set_fits() {
    let scorer = fit(&synthetic(5, 5, 0), &small());
    let report = scorer.fit_report();
    assert_eq!((report.rows, report.good, report.bad), (10, 5, 5));
}

#[test]
fn rows_labelled_unknown_are_ignored() {
    let with_unknown = synthetic(12, 9, 25);
    let labelled: Vec<LogRecord> = with_unknown
        .records()
        .filter(|record| record.label != Label::Unknown)
        .collect();
    assert_eq!(labelled.len(), 21);
    let without_unknown = LogsFrame::from_records(&labelled).unwrap();

    let a = fit(&with_unknown, &small());
    let b = fit(&without_unknown, &small());
    assert_eq!(weight_bits(&a), weight_bits(&b));
    assert_eq!(a.revision(), b.revision());
    assert_eq!(a.fit_report().unknown_ignored, 25);
    assert_eq!(b.fit_report().unknown_ignored, 0);
    assert_eq!(a.fit_report().rows, 21);
}

#[test]
fn unusable_settings_are_invalid_input() {
    let logs = synthetic(6, 6, 0);
    let cases = [
        BaselineSettings {
            buckets: 0,
            ..small()
        },
        BaselineSettings {
            ngram_max: 0,
            ..small()
        },
        BaselineSettings {
            l2: -1.0,
            ..small()
        },
        BaselineSettings {
            l2: f64::NAN,
            ..small()
        },
        BaselineSettings {
            tolerance: f64::INFINITY,
            ..small()
        },
        BaselineSettings {
            max_iter: 0,
            ..small()
        },
    ];
    for settings in cases {
        let error = BaselineScorer::fit_with(&logs, &[], &settings, &budget()).unwrap_err();
        assert!(
            matches!(error, FitError::InvalidSettings { .. }),
            "{settings:?}: {error:?}"
        );
        assert_eq!(L2dError::from(error).exit_code(), 2);
    }
}

// ---------------------------------------------------------------------------------------
// Determinism (2.4)
// ---------------------------------------------------------------------------------------

#[test]
fn a_second_fit_gives_bit_equal_weights_ratios_and_revision() {
    let (logs, summary) = fixture();
    let first = fitted_small();
    let second =
        BaselineScorer::fit_with(logs, std::slice::from_ref(summary), &small(), &budget()).unwrap();
    assert_eq!(weight_bits(first), weight_bits(&second));
    assert_eq!(first.revision(), second.revision());
    let texts = texts_of(logs);
    assert_eq!(ratio_bits(first, &texts), ratio_bits(&second, &texts));
    assert_eq!(*first, second);
}

#[test]
fn row_order_is_part_of_the_input() {
    let (logs, _) = fixture();
    let mut reversed = logs.to_records();
    reversed.reverse();
    let other = fit(&LogsFrame::from_records(&reversed).unwrap(), &small());
    // The same optimum up to rounding, but not the same bits.
    assert_ne!(other.revision(), fitted_small().revision());
    for text in texts_of(logs) {
        let difference = (other.ratio(&text) - fitted_small().ratio(&text)).abs();
        assert!(difference < 1e-3, "{difference} for {text:?}");
    }
}

const PROBE_VARIABLE: &str = "L2D_SCORER_PROCESS_PROBE";
const PROBE_MARKER: &str = "L2D_SCORER_PROBE ";

/// Revision, weights and fixture ratios of two fits, as text.
fn probe_lines() -> Vec<String> {
    let (logs, _) = fixture();
    let texts = texts_of(logs);
    let mut lines = Vec::new();
    for settings in [
        small(),
        BaselineSettings {
            routing: RoutingMode::Isolate,
            ..small()
        },
    ] {
        let scorer = fit(logs, &settings);
        let weights: Vec<String> = weight_bits(&scorer)
            .iter()
            .filter(|bits| **bits != 0)
            .map(|bits| format!("{bits:016x}"))
            .collect();
        let ratios: Vec<String> = ratio_bits(&scorer, &texts)
            .iter()
            .map(|bits| format!("{bits:08x}"))
            .collect();
        lines.push(format!(
            "{:?} revision={} iterations={} weights=[{}] ratios=[{}]",
            settings.routing,
            scorer.revision(),
            scorer.fit_report().iterations,
            weights.join(","),
            ratios.join(",")
        ));
    }
    lines
}

/// Prints the probe fits when started as a child of
/// [`fits_are_bit_identical_across_processes`]; does nothing in a normal test run.
#[test]
fn process_probe() {
    if std::env::var_os(PROBE_VARIABLE).is_none() {
        return;
    }
    println!();
    for line in probe_lines() {
        println!("{PROBE_MARKER}{line}");
    }
}

fn fits_of_a_child_process() -> Vec<String> {
    let binary = std::env::current_exe().expect("the test binary has a path");
    let output = Command::new(binary)
        .args([
            "process_probe",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PROBE_VARIABLE, "1")
        .output()
        .expect("the test binary starts a second time");
    assert!(output.status.success(), "child failed: {output:?}");
    String::from_utf8(output.stdout)
        .expect("probe output is text")
        .lines()
        .filter_map(|line| {
            line.find(PROBE_MARKER)
                .map(|at| line[at + PROBE_MARKER.len()..].trim_end().to_string())
        })
        .collect()
}

#[test]
fn fits_are_bit_identical_across_processes() {
    let own = probe_lines();
    assert_eq!(own.len(), 2);
    assert!(own.iter().all(|line| line.matches(',').count() > 300));
    for run in ["first", "second"] {
        let child = fits_of_a_child_process();
        assert_eq!(child.len(), own.len(), "{run} child process: line count");
        for (position, (theirs, ours)) in child.iter().zip(&own).enumerate() {
            assert!(
                theirs == ours,
                "{run} child process differs from this process in fit {position}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------
// Settings and revision
// ---------------------------------------------------------------------------------------

#[test]
fn every_setting_changes_the_revision() {
    let logs = synthetic(30, 30, 0);
    let base = small();
    let variants = [
        (
            "buckets",
            BaselineSettings {
                buckets: 2_048,
                ..base
            },
        ),
        (
            "ngram_max",
            BaselineSettings {
                ngram_max: 2,
                ..base
            },
        ),
        (
            "routing strip",
            BaselineSettings {
                routing: RoutingMode::Strip,
                ..base
            },
        ),
        (
            "routing isolate",
            BaselineSettings {
                routing: RoutingMode::Isolate,
                ..base
            },
        ),
        ("l2", BaselineSettings { l2: 0.5, ..base }),
        (
            "max_iter",
            BaselineSettings {
                max_iter: base.max_iter + 1,
                ..base
            },
        ),
        (
            "tolerance",
            BaselineSettings {
                tolerance: base.tolerance * 2.0,
                ..base
            },
        ),
    ];
    let mut revisions = BTreeSet::new();
    let reference = fit(&logs, &base);
    assert_eq!(reference.revision().len(), 40);
    assert!(reference.revision().bytes().all(|b| b.is_ascii_hexdigit()));
    revisions.insert(reference.revision().to_string());
    for (name, settings) in variants {
        let scorer = fit(&logs, &settings);
        assert_eq!(scorer.settings(), &settings);
        assert!(
            revisions.insert(scorer.revision().to_string()),
            "changing {name} left the revision unchanged"
        );
    }
}

#[test]
fn the_revision_is_the_documented_hash_of_settings_weights_and_bias() {
    let scorer = fitted_small();
    let settings = scorer.settings();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"l2d-baseline-scorer/1\n");
    bytes.extend_from_slice(&settings.buckets.to_le_bytes());
    bytes.push(settings.ngram_max);
    bytes.push(0); // keep
    bytes.extend_from_slice(&settings.l2.to_bits().to_le_bytes());
    bytes.extend_from_slice(&settings.max_iter.to_le_bytes());
    bytes.extend_from_slice(&settings.tolerance.to_bits().to_le_bytes());
    bytes.extend_from_slice(&(scorer.weights().len() as u64).to_le_bytes());
    for weight in scorer.weights() {
        bytes.extend_from_slice(&weight.to_bits().to_le_bytes());
    }
    bytes.extend_from_slice(&scorer.bias().to_bits().to_le_bytes());
    assert_eq!(REVISION_DOMAIN_BYTES, 22);
    assert_eq!(scorer.revision(), &content_sha256(&bytes)[..40]);
}

const REVISION_DOMAIN_BYTES: usize = b"l2d-baseline-scorer/1\n".len();

#[test]
fn a_fit_stopped_at_the_iteration_cap_is_a_scorer_that_says_so() {
    let (logs, _) = fixture();
    let settings = BaselineSettings {
        max_iter: 2,
        ..small()
    };
    let scorer = fit(logs, &settings);
    let report = scorer.fit_report();
    assert_eq!(report.termination, Termination::IterationCap);
    assert_eq!(report.iterations, 2);
    assert!(report.gradient_norm > settings.tolerance);
    assert!(!scorer.gradient_converged());

    // The visibility survives saving and loading.
    let scratch = Scratch::new("capped");
    let dir = scorer.save(&scratch.paths(), "capped").unwrap();
    let loaded = BaselineScorer::load(&dir).unwrap();
    assert_eq!(loaded.fit_report(), report);
    assert!(!loaded.gradient_converged());
}

// ---------------------------------------------------------------------------------------
// Scores frame
// ---------------------------------------------------------------------------------------

#[test]
fn the_scores_frame_has_the_documented_columns_in_row_order() {
    let (logs, _) = fixture();
    let scorer = fitted_small();
    let scores = scorer.score(logs).unwrap();

    let schema: Vec<(String, DataType)> = scores
        .schema()
        .iter()
        .map(|(name, dtype)| (name.to_string(), dtype.clone()))
        .collect();
    assert_eq!(
        schema,
        [
            (SCORES_TEXT.to_string(), DataType::String),
            (SCORES_SCORER.to_string(), DataType::String),
            (SCORES_RATIO.to_string(), DataType::Float32),
            (SCORES_ABSTAINED.to_string(), DataType::Boolean),
            (SCORES_NO_FEATURES.to_string(), DataType::Boolean),
        ]
    );
    assert_eq!(
        [SCORES_TEXT, SCORES_SCORER, SCORES_RATIO, SCORES_ABSTAINED],
        ["text", "scorer", "ratio", "abstained"]
    );
    assert_eq!(scores.height(), logs.len());
    for column in scores.get_columns() {
        assert_eq!(column.null_count(), 0, "column {}", column.name());
    }

    // Same rows in the same order as the logs frame, unknown rows included.
    let texts = texts_of(logs);
    let scored: Vec<String> = scores
        .column(SCORES_TEXT)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|text| text.unwrap().to_string())
        .collect();
    assert_eq!(scored, texts);
    let expected: Vec<f32> = texts.iter().map(|text| scorer.ratio(text)).collect();
    assert_eq!(ratios(&scores), expected);

    let scorers = scores.column(SCORES_SCORER).unwrap().str().unwrap();
    assert!(scorers.iter().all(|name| name == Some(SCORER_KIND)));
    assert_eq!(SCORER_KIND, "baseline");
    let abstained = scores.column(SCORES_ABSTAINED).unwrap().bool().unwrap();
    assert!(
        abstained.iter().all(|flag| flag == Some(false)),
        "the baseline never abstains"
    );
    let no_features = scores.column(SCORES_NO_FEATURES).unwrap().bool().unwrap();
    assert!(no_features.iter().all(|flag| flag == Some(false)));
}

#[test]
fn texts_are_scored_in_the_given_order_duplicates_included() {
    let scorer = fitted_small();
    let texts = ["b' OR '1'='1", "a", "b' OR '1'='1", ""];
    let scores = scorer.score_texts(&texts).unwrap();
    assert_eq!(scores.height(), 4);
    let found = ratios(&scores);
    for (text, ratio) in texts.iter().zip(&found) {
        assert_eq!(scorer.ratio(text).to_bits(), ratio.to_bits());
    }
    assert_eq!(found[0].to_bits(), found[2].to_bits());
    assert_eq!(scorer.score_texts(&[]).unwrap().height(), 0);
}

#[test]
fn a_text_without_features_scores_the_logistic_function_of_the_bias_and_is_flagged() {
    // CSIC-layout texts, so that strip leaves a payload to learn from.
    let logs = synthetic(40, 40, 0);
    let scorer = fit(
        &logs,
        &BaselineSettings {
            routing: RoutingMode::Strip,
            ..small()
        },
    );
    assert_eq!(scorer.fit_report().no_feature_rows, 0);
    let constant = sigmoid(scorer.bias()) as f32;

    // No `?` and no line feed: routing only, so strip leaves nothing to hash.
    let texts = [
        "GET http://localhost:8080/tienda1/index.jsp",
        "",
        "2026-11-11 12:00:01 sshd[411]: Failed password for root",
        "GET http://localhost:8080/tienda1/index.jsp?id=1' OR '1'='1",
    ];
    let scores = scorer.score_texts(&texts).unwrap();
    let flags: Vec<bool> = scores
        .column(SCORES_NO_FEATURES)
        .unwrap()
        .bool()
        .unwrap()
        .iter()
        .map(Option::unwrap)
        .collect();
    assert_eq!(flags, [true, true, true, false]);
    let found = ratios(&scores);
    for ratio in &found[..3] {
        assert_eq!(ratio.to_bits(), constant.to_bits());
    }
    assert_ne!(found[3].to_bits(), constant.to_bits());
    assert!(
        found[3] > constant,
        "the attack text scores above the constant"
    );
    assert!((0.0..=1.0).contains(&constant));
}

#[test]
fn fitted_rows_without_features_are_counted() {
    // Strip on the mini fixture: its POST rows have a space before the body, so they are
    // routing only. The fit still runs; the report says how many rows had no features.
    let (logs, _) = fixture();
    let scorer = fit(
        logs,
        &BaselineSettings {
            routing: RoutingMode::Strip,
            ..small()
        },
    );
    let hasher = CharHasher::new(scorer.settings().hashing()).unwrap();
    let expected = logs
        .records()
        .filter(|r| r.label != Label::Unknown && hasher.row(&r.text).is_empty())
        .count() as u64;
    assert!(expected > 0);
    assert_eq!(scorer.fit_report().no_feature_rows, expected);
}

// ---------------------------------------------------------------------------------------
// Memory (12.5)
// ---------------------------------------------------------------------------------------

#[test]
fn a_fit_above_the_memory_limit_is_refused_with_estimate_and_limit() {
    let (logs, _) = fixture();
    let settings = BaselineSettings::default();
    let estimate = fitted().fit_report().estimated_bytes;
    // The feature matrix, and above all 48 vectors of 65,537 doubles.
    assert!(estimate > 48 * 65_537 * 8, "{estimate}");
    assert!(estimate < 64 * 1024 * 1024, "{estimate}");

    // One byte less than the estimate is left of the limit: refused.
    let limit = PROCESS_BASELINE_BYTES + estimate - 1;
    let error = BaselineScorer::fit_with(logs, &[], &settings, &budget_with_limit(Some(limit)))
        .unwrap_err();
    assert!(
        matches!(error, FitError::Memory(L2dError::Refused { .. })),
        "{error:?}"
    );
    let reason = refusal(error);
    assert!(reason.contains("fitting the baseline scorer"), "{reason}");
    assert!(
        reason.contains(&format!("({estimate} bytes)")),
        "estimate missing: {reason}"
    );
    assert!(
        reason.contains(&format!("({limit} bytes)")),
        "limit missing: {reason}"
    );
    assert!(reason.contains("nothing was loaded"), "{reason}");

    // Exactly the estimate is left: accepted, with the same model.
    let limit = PROCESS_BASELINE_BYTES + estimate;
    let scorer =
        BaselineScorer::fit_with(logs, &[], &settings, &budget_with_limit(Some(limit))).unwrap();
    assert_eq!(scorer.revision(), fitted().revision());
    assert_eq!(scorer.fit_report().estimated_bytes, estimate);
}

#[test]
fn the_fit_without_a_budget_observes_the_host_and_gives_the_same_model() {
    let (logs, summary) = fixture();
    let scorer = BaselineScorer::fit(logs, std::slice::from_ref(summary), &small())
        .expect("the host has room for a fixture fit");
    assert_eq!(&scorer, fitted_small());
}

#[test]
fn the_memory_estimate_is_exact_in_its_entries_for_a_small_set() {
    // Up to ESTIMATE_SAMPLE_ROWS labelled rows the sample is every row.
    let report = fitted().fit_report();
    assert!((report.rows as usize) <= ESTIMATE_SAMPLE_ROWS);
    let (logs, _) = fixture();
    let longest = logs
        .records()
        .filter(|r| r.label != Label::Unknown)
        .map(|r| r.text.len() as u64)
        .max()
        .unwrap();
    let expected = report.stored_entries * 12
        + (report.rows + 1) * 8
        + report.rows
        + 48 * 65_537 * 8
        + 6 * report.rows * 8
        + longest * 3 * 40
        + report.rows * 16;
    assert_eq!(report.estimated_bytes, expected);
}

/// A frame whose sampled rows say little about the others: with twice as many labelled
/// rows as the sample, the sample is every second row, and those are the sparse ones.
fn misleading_frame() -> LogsFrame {
    let rows = 2 * ESTIMATE_SAMPLE_ROWS;
    let mut state: u32 = 7;
    let mut records = Vec::with_capacity(rows);
    for index in 0..rows {
        let label = if index % 4 < 2 {
            Label::Good
        } else {
            Label::Bad
        };
        let text = if index % 2 == 0 {
            // One byte repeated: three entries, whatever the length.
            format!("{index:05}{}", "a".repeat(400))
        } else {
            let mut text = format!("{index:05}");
            for _ in 0..60 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let byte = 0x21 + ((state >> 16) % 90) as u8;
                text.push(char::from(byte));
            }
            if label == Label::Bad {
                text.push_str("' OR '1'='1");
            }
            text
        };
        records.push(record(text, label));
    }
    LogsFrame::from_records(&records).unwrap()
}

#[test]
fn a_sample_that_underestimates_is_corrected_before_the_matrix_grows() {
    let logs = misleading_frame();
    let settings = BaselineSettings {
        buckets: 1_024,
        max_iter: 20,
        ..BaselineSettings::default()
    };
    let scorer = fit(&logs, &settings);
    let report = scorer.fit_report();
    assert_eq!(report.rows as usize, 2 * ESTIMATE_SAMPLE_ROWS);

    // What the sample alone would have given: far fewer entries than the rows have.
    let hasher = CharHasher::new(settings.hashing()).unwrap();
    let texts = texts_of(&logs);
    let sampled: u64 = texts
        .iter()
        .step_by(2)
        .map(|t| hasher.row(t).len() as u64)
        .sum();
    let sampled_bytes: u64 = texts.iter().step_by(2).map(|t| t.len() as u64).sum();
    let all_bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();
    let first_guess =
        (sampled as f64 / sampled_bytes as f64 * all_bytes as f64 * 1.25).ceil() as u64;
    assert!(
        first_guess * 4 < report.stored_entries,
        "{first_guess} vs {}",
        report.stored_entries
    );

    // The reported estimate covers the real entries and the entries moved while growing.
    let fixed = (report.rows + 1) * 8 + report.rows + 48 * 1_025 * 8 + 6 * report.rows * 8;
    assert!(
        report.estimated_bytes >= report.stored_entries * 12 + first_guess * 12 + fixed,
        "{report:?}"
    );

    // The second check is a real check: a limit that admits the first estimate but not
    // the corrected one refuses the fit.
    let limit = PROCESS_BASELINE_BYTES + report.estimated_bytes - 1;
    let error = BaselineScorer::fit_with(&logs, &[], &settings, &budget_with_limit(Some(limit)))
        .unwrap_err();
    let reason = refusal(error);
    assert!(
        reason.contains(&format!("({} bytes)", report.estimated_bytes)),
        "{reason}"
    );
}

// ---------------------------------------------------------------------------------------
// Save and load
// ---------------------------------------------------------------------------------------

#[test]
fn a_saved_scorer_loads_back_with_bit_equal_ratios_and_the_same_revision() {
    let (logs, _) = fixture();
    let scorer = fitted();
    let scratch = Scratch::new("round-trip");
    let paths = scratch.paths();

    let dir = scorer.save(&paths, "mini").unwrap();
    assert_eq!(
        dir,
        paths
            .baselines_dir()
            .join(format!("mini-{}", &scorer.revision()[..12]))
    );
    assert_eq!(dir, paths.baseline_dir("mini", scorer.revision()));

    let loaded = BaselineScorer::load(&dir).unwrap();
    assert_eq!(loaded.revision(), scorer.revision());
    assert_eq!(weight_bits(&loaded), weight_bits(scorer));
    let mut texts = texts_of(logs);
    texts.extend(["".to_string(), "never seen ' OR 1=1 --".to_string()]);
    assert_eq!(ratio_bits(&loaded, &texts), ratio_bits(scorer, &texts));
    assert_eq!(loaded.settings(), scorer.settings());
    assert_eq!(loaded.fit_report(), scorer.fit_report());
    assert_eq!(loaded.fitting_inputs(), scorer.fitting_inputs());
    assert_eq!(&loaded, scorer);
    assert_eq!(BaselineScorer::load_under(&paths, &dir).unwrap(), loaded);
}

#[test]
fn the_saved_files_have_the_documented_layout() {
    let scorer = fitted_small();
    let scratch = Scratch::new("layout");
    let paths = scratch.paths();
    let dir = scorer.save(&paths, "layout").unwrap();

    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    // Task 1.11 added the hashes of the fitted texts (`tests/scorer_fitted_texts.rs`).
    assert_eq!(files, [MANIFEST_FILE, FITTED_TEXTS_FILE, WEIGHTS_FILE]);
    assert_eq!(
        (MANIFEST_FILE, WEIGHTS_FILE),
        ("baseline.json", "weights.bin")
    );

    // weights.bin: little-endian doubles in column order.
    let bytes = std::fs::read(dir.join(WEIGHTS_FILE)).unwrap();
    assert_eq!(bytes.len(), 4_096 * 8);
    for (chunk, weight) in bytes.as_chunks::<8>().0.iter().zip(scorer.weights()) {
        assert_eq!(*chunk, weight.to_bits().to_le_bytes());
    }

    // baseline.json: readable values beside exact bit patterns.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!(manifest["format"], 2);
    assert_eq!(manifest["kind"], "baseline");
    assert_eq!(manifest["revision"], scorer.revision());
    assert_eq!(manifest["tool_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(manifest["settings"]["buckets"], 4_096);
    assert_eq!(manifest["settings"]["ngram_max"], 3);
    assert_eq!(manifest["settings"]["routing"], "keep");
    assert_eq!(manifest["settings"]["l2"]["value"], 1.0);
    assert_eq!(manifest["settings"]["l2"]["bits"], "3ff0000000000000");
    assert_eq!(manifest["settings"]["max_iter"], scorer.settings().max_iter);
    assert_eq!(
        manifest["settings"]["tolerance"]["bits"],
        format!("{:016x}", scorer.settings().tolerance.to_bits())
    );
    assert_eq!(manifest["model"]["dimension"], 4_096);
    assert_eq!(
        manifest["model"]["bias"]["bits"],
        format!("{:016x}", scorer.bias().to_bits())
    );
    assert_eq!(manifest["fit"]["termination"], "converged");
    assert_eq!(
        manifest["fit"]["iterations"],
        scorer.fit_report().iterations
    );
    assert_eq!(manifest["fit"]["rows"], 168);
    assert_eq!(manifest["fit"]["good"], 96);
    assert_eq!(manifest["fit"]["bad"], 72);
    assert_eq!(manifest["fit"]["unknown_ignored"], 1);
    assert_eq!(
        manifest["fit"]["gradient_norm"]["bits"],
        format!("{:016x}", scorer.fit_report().gradient_norm.to_bits())
    );

    // Nothing else was written below the root, and no temporary directory is left.
    let saved: Vec<String> = std::fs::read_dir(paths.baselines_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(saved, [format!("layout-{}", &scorer.revision()[..12])]);
    let top: Vec<String> = std::fs::read_dir(&scratch.root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(top, ["models"]);
}

#[test]
fn isolate_mode_saves_one_weight_per_column_of_the_doubled_space() {
    let logs = synthetic(20, 20, 0);
    let scorer = fit(
        &logs,
        &BaselineSettings {
            routing: RoutingMode::Isolate,
            max_iter: 30,
            ..BaselineSettings::default()
        },
    );
    assert_eq!(scorer.weights().len(), 131_072);
    let scratch = Scratch::new("isolate");
    let dir = scorer.save(&scratch.paths(), "isolate").unwrap();
    assert_eq!(
        std::fs::metadata(dir.join(WEIGHTS_FILE)).unwrap().len(),
        131_072 * 8
    );
    let loaded = BaselineScorer::load(&dir).unwrap();
    assert_eq!(weight_bits(&loaded), weight_bits(&scorer));
    assert_eq!(loaded.settings().routing, RoutingMode::Isolate);
    let texts = texts_of(&logs);
    assert_eq!(ratio_bits(&loaded, &texts), ratio_bits(&scorer, &texts));
}

#[test]
fn the_fitting_inputs_are_stored_and_readable() {
    let (_, summary) = fixture();
    let scorer = fitted_small();
    let inputs = scorer.fitting_inputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].content_sha256, summary.content_sha256);
    assert_eq!(inputs[0].content_sha256.len(), 64);
    assert_eq!(inputs[0].source_path, summary.source_path.to_string_lossy());
    assert_eq!(inputs[0].records, 200);
    assert_eq!(inputs[0].distinct_texts, 169);

    let scratch = Scratch::new("inputs");
    let dir = scorer.save(&scratch.paths(), "inputs").unwrap();
    let loaded = BaselineScorer::load(&dir).unwrap();
    assert_eq!(loaded.fitting_inputs(), inputs);
    // What calibration asks (11.1): was this input used for fitting?
    assert!(loaded.was_fitted_on(&summary.content_sha256));
    assert!(loaded.was_fitted_on(&summary.content_sha256.to_uppercase()));
    assert!(!loaded.was_fitted_on(&"0".repeat(64)));

    // The hashes are in the file as plain text.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!(
        manifest["fitting_inputs"][0]["content_sha256"],
        summary.content_sha256.as_str()
    );

    // Several inputs are kept in order; none is fine too.
    let (logs, _) = fixture();
    let mut second = summary.clone();
    second.content_sha256 = "ab".repeat(32);
    second.source_path = PathBuf::from("elsewhere.csv");
    let both =
        BaselineScorer::fit_with(logs, &[summary.clone(), second], &small(), &budget()).unwrap();
    assert_eq!(both.fitting_inputs().len(), 2);
    assert!(both.was_fitted_on(&"ab".repeat(32)));
    assert_eq!(
        both.revision(),
        scorer.revision(),
        "inputs are not part of the revision"
    );
    assert!(fit(logs, &small()).fitting_inputs().is_empty());
}

#[test]
fn a_flipped_byte_in_the_weights_file_makes_load_refuse() {
    let scorer = fitted_small();
    let scratch = Scratch::new("flipped");
    let dir = scorer.save(&scratch.paths(), "flipped").unwrap();
    let weights = dir.join(WEIGHTS_FILE);
    let original = std::fs::read(&weights).unwrap();

    // A weight that is not zero, so the model really changes; and the very last byte.
    let used = scorer.weights().iter().position(|w| *w != 0.0).unwrap();
    for position in [used * 8, original.len() - 1] {
        let mut bytes = original.clone();
        bytes[position] ^= 0x01;
        std::fs::write(&weights, &bytes).unwrap();
        match BaselineScorer::load(&dir) {
            Err(L2dError::Refused { reason }) => {
                assert!(reason.contains("does not match its revision"), "{reason}");
                assert!(reason.contains(scorer.revision()), "{reason}");
            }
            other => panic!("byte {position} flipped: expected a refusal, got {other:?}"),
        }
    }
    std::fs::write(&weights, &original).unwrap();
    assert!(BaselineScorer::load(&dir).is_ok());
}

#[test]
fn changed_settings_bias_or_truncated_weights_make_load_refuse() {
    let scorer = fitted_small();
    let scratch = Scratch::new("tampered");
    let dir = scorer.save(&scratch.paths(), "tampered").unwrap();
    let manifest_path = dir.join(MANIFEST_FILE);
    let original = std::fs::read_to_string(&manifest_path).unwrap();
    let refused = |what: &str| match BaselineScorer::load(&dir) {
        Err(L2dError::Refused { reason }) => reason,
        other => panic!("{what}: expected a refusal, got {other:?}"),
    };

    // Another penalty in the settings: the files no longer give the stated revision.
    let l2 = format!("{:016x}", 1.0f64.to_bits());
    assert!(original.contains(&l2));
    std::fs::write(
        &manifest_path,
        original.replace(&l2, &format!("{:016x}", 2.0f64.to_bits())),
    )
    .unwrap();
    assert!(refused("l2").contains("does not match its revision"));

    // Another bias.
    let bias = format!("{:016x}", scorer.bias().to_bits());
    let other = format!("{:016x}", (scorer.bias() + 1.0).to_bits());
    std::fs::write(&manifest_path, original.replace(&bias, &other)).unwrap();
    assert!(refused("bias").contains("does not match its revision"));

    // Not JSON at all.
    std::fs::write(&manifest_path, "not json").unwrap();
    assert!(refused("manifest").contains("baseline.json is not readable"));
    std::fs::write(&manifest_path, &original).unwrap();

    // A weights file of the wrong length.
    let weights = dir.join(WEIGHTS_FILE);
    let bytes = std::fs::read(&weights).unwrap();
    std::fs::write(&weights, &bytes[..bytes.len() - 8]).unwrap();
    let reason = refused("truncated weights");
    assert!(reason.contains("weights.bin has 32760 bytes"), "{reason}");
    std::fs::write(&weights, &bytes).unwrap();
    assert!(BaselineScorer::load(&dir).is_ok());

    // A directory renamed to another revision prefix.
    let renamed = dir.with_file_name("tampered-000000000000");
    std::fs::rename(&dir, &renamed).unwrap();
    match BaselineScorer::load(&renamed) {
        Err(L2dError::Refused { reason }) => assert!(reason.contains("directory name"), "{reason}"),
        other => panic!("renamed: expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_missing_scorer_names_the_item_and_where_it_was_expected() {
    let scratch = Scratch::new("missing");
    let paths = scratch.paths();
    let absent = paths.baseline_dir("absent", "0123456789abcdef");
    match BaselineScorer::load(&absent) {
        Err(error @ L2dError::Missing { .. }) => {
            assert_eq!(error.exit_code(), 3);
            let message = error.to_string();
            assert!(message.contains("baseline scorer"), "{message}");
            assert!(message.contains(&absent.display().to_string()), "{message}");
        }
        other => panic!("expected a missing item, got {other:?}"),
    }

    // A directory that lacks one of its files names that file.
    let dir = fitted_small().save(&paths, "partial").unwrap();
    std::fs::remove_file(dir.join(WEIGHTS_FILE)).unwrap();
    match BaselineScorer::load(&dir) {
        Err(L2dError::Missing { item, expected_at }) => {
            assert!(item.contains("weights.bin"), "{item}");
            assert_eq!(expected_at, dir.join(WEIGHTS_FILE));
        }
        other => panic!("expected a missing file, got {other:?}"),
    }
}

#[test]
fn unusable_scorer_names_are_rejected_and_nothing_is_written() {
    let scorer = fitted_small();
    let scratch = Scratch::new("names");
    let paths = scratch.paths();
    let too_long = "n".repeat(65);
    for name in [
        "",
        ".",
        "..",
        ".hidden",
        "../escape",
        "..\\escape",
        "a/b",
        "a\\b",
        "C:evil",
        "with space",
        "tab\tname",
        "nul\0name",
        "ümlaut",
        "semi;colon",
        too_long.as_str(),
    ] {
        match scorer.save(&paths, name) {
            Err(error @ L2dError::InvalidInput { .. }) => assert_eq!(error.exit_code(), 2),
            other => panic!("name {name:?}: expected invalid input, got {other:?}"),
        }
    }
    assert!(
        !paths.models_dir().exists(),
        "a rejected name must not create anything"
    );
    assert!(std::fs::read_dir(&scratch.root).unwrap().next().is_none());

    // The longest accepted name, with every allowed kind of character.
    let name = format!("A-z_0.9{}", "n".repeat(57));
    assert_eq!(name.len(), 64);
    let dir = scorer.save(&paths, &name).unwrap();
    assert!(paths.is_under_models(&dir));
    assert!(BaselineScorer::load(&dir).is_ok());
}

#[test]
fn targets_outside_the_model_directory_are_refused() {
    let scorer = fitted_small();
    let scratch = Scratch::new("outside");
    // A root written with a parent-directory component is not accepted as a model
    // location (the check of `ProjectPaths::is_under_models` is lexical).
    let odd = ProjectPaths::new(scratch.root.join("inner").join(".."));
    match scorer.save(&odd, "odd") {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains("model directory"), "{reason}")
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(std::fs::read_dir(&scratch.root).unwrap().next().is_none());

    // Reading through the project paths refuses a directory outside `models/`.
    let paths = scratch.paths();
    let dir = scorer.save(&paths, "inside").unwrap();
    let copy = scratch
        .root
        .join("elsewhere")
        .join(dir.file_name().unwrap());
    std::fs::create_dir_all(&copy).unwrap();
    for file in [MANIFEST_FILE, WEIGHTS_FILE] {
        std::fs::copy(dir.join(file), copy.join(file)).unwrap();
    }
    match BaselineScorer::load_under(&paths, &copy) {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains("model directory"), "{reason}")
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(BaselineScorer::load_under(&paths, &dir).is_ok());
}

#[test]
fn saving_again_keeps_the_directory_and_a_foreign_directory_is_not_overwritten() {
    let scorer = fitted_small();
    let scratch = Scratch::new("again");
    let paths = scratch.paths();
    let dir = scorer.save(&paths, "again").unwrap();
    let before = std::fs::read(dir.join(MANIFEST_FILE)).unwrap();
    assert_eq!(scorer.save(&paths, "again").unwrap(), dir);
    assert_eq!(std::fs::read(dir.join(MANIFEST_FILE)).unwrap(), before);

    // The target exists but holds something else: refused, left as it was.
    std::fs::write(dir.join(MANIFEST_FILE), "something else").unwrap();
    match scorer.save(&paths, "again") {
        Err(L2dError::Refused { reason }) => assert!(reason.contains("left unchanged"), "{reason}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(
        std::fs::read(dir.join(MANIFEST_FILE)).unwrap(),
        b"something else"
    );
    let entries = std::fs::read_dir(paths.baselines_dir()).unwrap().count();
    assert_eq!(entries, 1, "no temporary directory is left behind");
}

// ---------------------------------------------------------------------------------------
// Saving a revision that is already there: the fitting inputs are merged (11.1)
// ---------------------------------------------------------------------------------------

/// The fixture with its first data row repeated at the end: another file (201 records,
/// another content hash) with the same distinct texts in the same order, hence the same
/// logs frame and the same model.
fn fixture_with_a_repeated_row(scratch: &Scratch) -> (LogsFrame, InputSummary) {
    let original = std::fs::read_to_string(fixture_path()).unwrap();
    let first_row = original.lines().nth(1).unwrap();
    let mut longer = original.clone();
    if !longer.ends_with('\n') {
        longer.push('\n');
    }
    longer.push_str(first_row);
    longer.push('\n');
    let path = scratch.root.join("fixture-with-a-repeated-row.csv");
    std::fs::write(&path, longer).unwrap();
    LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&path)
        .expect("the longer fixture is readable")
}

fn files_of(dir: &Path) -> Vec<String> {
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    files
}

#[test]
fn saving_the_same_model_fitted_on_another_input_adds_that_input() {
    let (_, summary) = fixture();
    let first = fitted_small();
    let scratch = Scratch::new("merge-second");
    let paths = scratch.paths();

    let (logs, other_summary) = fixture_with_a_repeated_row(&scratch);
    assert_eq!(other_summary.records, 201);
    assert_eq!(other_summary.distinct_texts, 169);
    assert_ne!(other_summary.content_sha256, summary.content_sha256);
    let second = BaselineScorer::fit_with(
        &logs,
        std::slice::from_ref(&other_summary),
        &small(),
        &budget(),
    )
    .unwrap();
    assert_eq!(second.revision(), first.revision(), "the same model");
    assert!(second.was_fitted_on(&other_summary.content_sha256));
    assert!(!second.was_fitted_on(&summary.content_sha256));

    let dir = first.save(&paths, "merge").unwrap();
    let weights_before = std::fs::read(dir.join(WEIGHTS_FILE)).unwrap();
    let weights_written = std::fs::metadata(dir.join(WEIGHTS_FILE))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(second.save(&paths, "merge").unwrap(), dir);

    // The stored scorer names every input the model was fitted on: what was stored
    // before, then what this save added.
    let loaded = BaselineScorer::load(&dir).unwrap();
    assert!(loaded.was_fitted_on(&summary.content_sha256));
    assert!(loaded.was_fitted_on(&other_summary.content_sha256));
    let hashes: Vec<&str> = loaded
        .fitting_inputs()
        .iter()
        .map(|input| input.content_sha256.as_str())
        .collect();
    assert_eq!(
        hashes,
        [
            summary.content_sha256.as_str(),
            other_summary.content_sha256.as_str()
        ]
    );
    assert_eq!(loaded.fitting_inputs()[1].records, 201);
    for input in first.fitting_inputs().iter().chain(second.fitting_inputs()) {
        assert!(loaded.was_fitted_on(&input.content_sha256));
    }
    // The model, the settings and the stored fit report are those saved first.
    assert_eq!(loaded.revision(), first.revision());
    assert_eq!(weight_bits(&loaded), weight_bits(first));
    assert_eq!(loaded.fit_report(), first.fit_report());
    assert_eq!(loaded.settings(), first.settings());
    // The weights file was not rewritten, and nothing temporary is left.
    assert_eq!(
        std::fs::read(dir.join(WEIGHTS_FILE)).unwrap(),
        weights_before
    );
    assert_eq!(
        std::fs::metadata(dir.join(WEIGHTS_FILE))
            .unwrap()
            .modified()
            .unwrap(),
        weights_written
    );
    assert_eq!(
        files_of(&dir),
        [MANIFEST_FILE, FITTED_TEXTS_FILE, WEIGHTS_FILE]
    );
    assert_eq!(files_of(&paths.baselines_dir()).len(), 1);

    // Saving either scorer again adds nothing: the file stays byte for byte.
    let merged = std::fs::read(dir.join(MANIFEST_FILE)).unwrap();
    assert_eq!(first.save(&paths, "merge").unwrap(), dir);
    assert_eq!(second.save(&paths, "merge").unwrap(), dir);
    assert_eq!(loaded.save(&paths, "merge").unwrap(), dir);
    assert_eq!(std::fs::read(dir.join(MANIFEST_FILE)).unwrap(), merged);
    assert_eq!(
        std::fs::read(dir.join(WEIGHTS_FILE)).unwrap(),
        weights_before
    );
}

#[test]
fn saving_over_a_scorer_stored_without_inputs_adds_the_inputs() {
    let (logs, summary) = fixture();
    let scratch = Scratch::new("merge-empty");
    let paths = scratch.paths();

    let without = fit(logs, &small());
    assert!(without.fitting_inputs().is_empty());
    assert!(!without.has_known_fitting_inputs());
    let dir = without.save(&paths, "merge").unwrap();
    assert!(
        !BaselineScorer::load(&dir)
            .unwrap()
            .has_known_fitting_inputs()
    );

    let with = fitted_small();
    assert!(with.has_known_fitting_inputs());
    assert_eq!(with.revision(), without.revision());
    assert_eq!(with.save(&paths, "merge").unwrap(), dir);
    let loaded = BaselineScorer::load(&dir).unwrap();
    assert!(loaded.was_fitted_on(&summary.content_sha256));
    assert_eq!(loaded.fitting_inputs(), with.fitting_inputs());
    assert!(loaded.has_known_fitting_inputs());

    // The other way round: a scorer without inputs saved over one with inputs takes
    // nothing away.
    let merged = std::fs::read(dir.join(MANIFEST_FILE)).unwrap();
    assert_eq!(without.save(&paths, "merge").unwrap(), dir);
    assert_eq!(std::fs::read(dir.join(MANIFEST_FILE)).unwrap(), merged);
    assert!(
        BaselineScorer::load(&dir)
            .unwrap()
            .was_fitted_on(&summary.content_sha256)
    );
}

#[test]
fn an_input_already_stored_is_not_added_twice() {
    let (logs, summary) = fixture();
    let scratch = Scratch::new("merge-duplicate");
    let paths = scratch.paths();
    let dir = fitted_small().save(&paths, "merge").unwrap();
    let before = std::fs::read(dir.join(MANIFEST_FILE)).unwrap();

    // The same content under another path and with the hash in upper case, twice.
    let mut moved = summary.clone();
    moved.source_path = PathBuf::from("copy-of-the-fixture.csv");
    moved.content_sha256 = summary.content_sha256.to_uppercase();
    let again =
        BaselineScorer::fit_with(logs, &[moved.clone(), moved], &small(), &budget()).unwrap();
    assert_eq!(again.save(&paths, "merge").unwrap(), dir);
    assert_eq!(std::fs::read(dir.join(MANIFEST_FILE)).unwrap(), before);
    assert_eq!(
        BaselineScorer::load(&dir).unwrap().fitting_inputs().len(),
        1
    );
}
