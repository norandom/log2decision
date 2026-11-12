//! Triage and the calibration it uses, measured against the memory they say they take
//! (task 1.10; requirements 12.6, 12.9), and triage of HTTP CSIC 2010 on request.
//!
//! This test binary counts every heap allocation of the process (a counting global
//! allocator around the system allocator), as `conformal_csic.rs` does.
//!
//! The CSIC measurement runs on request only, in release mode:
//!
//! ```text
//! cargo test --release -p l2d-decide --test triage_memory -- --ignored --nocapture --test-threads=1
//! ```
//!
//! It is skipped with a note when `data/csic-2010/csic_database.csv` is absent.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;

use l2d_baseline::{BaselineScorer, BaselineSettings};
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::{
    Csic2010Adapter, InputAdapter, L2dError, Label, LogRecord, LogsFrame, MemoryBudget,
    MemorySettings, ProjectPaths,
};
use l2d_decide::ScorerKind;
use l2d_decide::conformal::{
    CalibrationPoint, CalibrationSettings, ConformalCalibrator, points_frame,
};
use l2d_decide::scorer::{Scorer, ScoringCost};
use l2d_decide::triage::{TriageCalibration, TriageResult, TriageSettings, triage_with};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every call is forwarded unchanged to the system allocator; the counters are
// only bookkeeping beside it.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            if new_size >= layout.size() {
                grew(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The measurements share the counters: one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const MIB: f64 = 1024.0 * 1024.0;
const GIB: u64 = 1024 * 1024 * 1024;
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// What a run did to the heap: its peak above the level before, and what it left.
struct Heap {
    peak: u64,
    kept: u64,
}

/// Runs `run` and returns its result with the heap's peak above the level before and
/// the bytes still allocated afterwards (the result is alive then).
fn measure<T>(run: impl FnOnce() -> T) -> (T, Heap) {
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = run();
    let heap = Heap {
        peak: PEAK.load(Ordering::Relaxed).saturating_sub(before) as u64,
        kept: CURRENT.load(Ordering::Relaxed).saturating_sub(before) as u64,
    };
    (result, heap)
}

/// A budget with a fixed free-memory reading, so the tests do not depend on the host.
fn budget_of(batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).expect("the budget is usable")
}

/// FNV-1a over the bytes of `text`: a fixed pseudo-random number per text.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn unit(seed: &str) -> f64 {
    (fnv(seed) >> 11) as f64 / (1u64 << 53) as f64
}

/// A project root in the system temporary directory, removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-decide-triage-memory-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("the scratch root can be created");
        Self { root }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Calibration points: `groups` groups (none when 0) with `per_group` good and as many
/// bad logs each, the group names padded to `name_bytes` bytes.
fn points(groups: usize, per_group: usize, name_bytes: usize) -> Vec<CalibrationPoint> {
    let mut points = Vec::new();
    for group in 0..groups.max(1) {
        let name = (groups > 0).then(|| format!("{:/<name_bytes$}", format!("/g{group}")));
        for index in 0..per_group {
            for (label, centre) in [(Label::Good, 0.3), (Label::Bad, 0.7)] {
                let noise = unit(&format!("{group}-{index}-{centre}")) - 0.5;
                points.push(CalibrationPoint {
                    ratio: (centre + 0.4 * noise).clamp(0.0, 1.0) as f32,
                    label,
                    group: name.clone(),
                });
            }
        }
    }
    points
}

fn calibrate(points: &[CalibrationPoint], min_per_group: u32) -> ConformalCalibrator {
    ConformalCalibrator::fit_with(
        &points_frame(points).unwrap(),
        ScorerKind::Baseline,
        REVISION,
        &CalibrationSettings { min_per_group },
        &budget_of(None),
    )
    .expect("the points calibrate")
}

// ---------------------------------------------------------------------------------------
// What a calibration keeps (requirement 12.9)
// ---------------------------------------------------------------------------------------

/// The figure a calibration states (`held_bytes`) against the heap a loaded and a
/// fitted calibration really keep, over shapes from "two long lists" to "a list per
/// group and label, with long names". For a loaded calibration the figure is never
/// below the heap and at most an eighth above it (the most for many small groups, whose
/// index entries are counted at their largest). While it is being loaded a calibration
/// takes about twice what it keeps (the file's bytes beside the lists), which is
/// printed and not asserted: `load` takes no budget.
#[test]
fn a_calibration_keeps_what_it_says_it_keeps() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("kept");
    let paths = ProjectPaths::new(&scratch.root);
    // What Polars sets up once, on its first query, is not the calibration's.
    drop(calibrate(&points(3, 200, 0), 100));
    for (shape, groups, per_group, name_bytes, min_per_group) in [
        ("no groups, 40,000 logs", 0, 20_000, 0, 100),
        ("no groups, 400 logs", 0, 200, 0, 100),
        ("2,000 groups of 20 logs", 2_000, 10, 0, 10),
        (
            "300 groups of 200 logs, names of 200 bytes",
            300,
            100,
            200,
            100,
        ),
        ("5,000 groups too small for a list", 5_000, 4, 0, 100),
    ] {
        let points = points(groups, per_group, name_bytes);
        let frame = points_frame(&points).unwrap();
        let settings = CalibrationSettings { min_per_group };
        let budget = budget_of(None);
        let (fitted, fit) = measure(|| {
            ConformalCalibrator::fit_with(
                &frame,
                ScorerKind::Baseline,
                REVISION,
                &settings,
                &budget,
            )
            .expect("the points calibrate")
        });
        let values: u64 = fitted.strata().iter().map(|list| list.logs).sum();
        let lists = fitted.strata().len();
        let dir = fitted.save(&paths, "kept").expect("the calibration saves");
        let stated = fitted.held_bytes();
        drop(fitted);

        let (loaded, load) = measure(|| ConformalCalibrator::load(&dir).expect("it loads"));
        assert_eq!(loaded.held_bytes(), stated);
        println!(
            "L2D_CALIBRATION_KEPT {shape}: lists={lists} values={values} stated={stated} \
             loaded_keeps={} ({:.3} of stated) fitted_keeps={} load_peak={} ({:.2} bytes per \
             value)",
            load.kept,
            load.kept as f64 / stated as f64,
            fit.kept,
            load.peak,
            load.peak as f64 / values as f64,
        );
        // Loading allocates nothing but the calibration, so its figure is exact.
        assert!(
            load.kept <= stated,
            "{shape}: a loaded calibration keeps {} bytes of heap, above the {stated} it \
             states",
            load.kept
        );
        assert!(
            stated <= load.kept + load.kept / 8 + 512,
            "{shape}: a loaded calibration states {stated} bytes but keeps {}",
            load.kept
        );
        // Fitting counts in Polars, whose threads keep a little of their own (a few
        // hundred bytes were seen), so the fitted figure gets that much room.
        assert!(
            fit.kept <= stated + 4_096,
            "{shape}: a fitted calibration keeps {} bytes of heap, far above the {stated} it \
             states",
            fit.kept
        );
        assert!(stated >= 8 * values);
        drop(loaded);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

// ---------------------------------------------------------------------------------------
// The heap triage takes against what it holds and asks for (requirement 12.9)
// ---------------------------------------------------------------------------------------

/// What triage asked the budget for at its peak, beside the calibration (which was on
/// the heap before triage began): what it held and its largest batch.
fn asked(result: &TriageResult, logs: usize) -> u64 {
    let calibration = result.calibration().map_or(0, |used| used.held_bytes);
    let largest = result.plan().max_rows_used.min(logs as u64);
    result.held_bytes() - calibration + largest * result.plan().bytes_per_row
}

fn settings<'c>(
    calibrator: Option<&'c ConformalCalibrator>,
    error_rate: f32,
    review_budget: Option<u64>,
) -> TriageSettings<'c> {
    TriageSettings {
        primary: ScorerKind::Baseline,
        review_budget,
        calibration: calibrator.map(|calibrator| TriageCalibration {
            calibrator,
            error_rate,
        }),
    }
}

/// Synthetic request lines: half good around `/shop`, half bad with an injected piece,
/// in three groups.
fn synthetic_logs(count: usize, seed: &str) -> LogsFrame {
    let records: Vec<LogRecord> = (0..count)
        .map(|index| {
            let bad = index % 2 == 1;
            let group = format!("/shop/{}", index % 3);
            let noise = fnv(&format!("{seed}-{index}"));
            let text = if bad {
                format!(
                    "GET {group}?id={noise}&q=%27+OR+1%3D1--+{index}&cmd=..%2F..%2Fetc%2Fpasswd"
                )
            } else {
                format!(
                    "GET {group}?id={noise}&page={}&lang=es&item={index}",
                    noise % 40
                )
            };
            LogRecord {
                text,
                label: if bad { Label::Bad } else { Label::Good },
                group: Some(group),
                first_position: index as u64 + 1,
                occurrences: 1,
            }
        })
        .collect();
    LogsFrame::from_records(&records).unwrap()
}

#[test]
fn the_heap_triage_takes_stays_within_what_it_holds_and_its_batches() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let small = BaselineSettings {
        buckets: 4_096,
        ..BaselineSettings::default()
    };
    let mut scorer =
        BaselineScorer::fit_with(&synthetic_logs(600, "fit"), &[], &small, &budget_of(None))
            .expect("the synthetic logs fit");
    let calibrator = ConformalCalibrator::fit_scorer(
        &mut scorer,
        &synthetic_logs(900, "calibrate"),
        &CalibrationSettings { min_per_group: 100 },
        &budget_of(None),
    )
    .expect("the synthetic logs calibrate");
    let logs = synthetic_logs(20_000, "triage");

    let mut reference: Option<TriageResult> = None;
    for (with_calibration, batch_rows) in [
        (false, None),
        (false, Some(1_024)),
        (true, None),
        (true, Some(1_024)),
        (true, Some(1)),
    ] {
        let settings = settings(with_calibration.then_some(&calibrator), 0.05, Some(500));
        let mut budget = budget_of(batch_rows);
        let (result, heap) =
            measure(|| triage_with(&logs, &mut [&mut scorer], &settings, &mut budget));
        let result = result.expect("the logs are triaged");
        let asked = asked(&result, logs.len());
        println!(
            "L2D_TRIAGE_HEAP calibration={with_calibration} batch_rows={batch_rows:?}: \
             peak={} kept={} asked={asked} ({:.2} of asked) held={} bytes_per_row={}",
            heap.peak,
            heap.kept,
            heap.peak as f64 / asked as f64,
            result.held_bytes(),
            result.plan().bytes_per_row,
        );
        assert!(
            heap.peak <= asked,
            "triage took {} bytes of heap at its peak, above the {asked} it asked for",
            heap.peak
        );
        // The same result whatever the batch size (requirement 12.6).
        match &reference {
            Some(first) if first.calibration().is_some() == with_calibration => {
                assert!(result.frame().equals_missing(first.frame()));
            }
            _ => reference = Some(result),
        }
    }
}

// ---------------------------------------------------------------------------------------
// Evidence in words (review of task 1.10)
// ---------------------------------------------------------------------------------------

/// A scorer under the contract whose evidence is in words: a String column `why` of
/// `width` bytes a row, and with `sparse` a second one that is null on two rows of
/// three. With `abstains` it gives no ratio at all.
struct Wordy {
    width: usize,
    sparse: bool,
    abstains: bool,
}

impl Wordy {
    fn why(&self, text: &str) -> String {
        format!("{:0>width$}", fnv(text), width = self.width)
    }
}

impl Scorer for Wordy {
    fn kind(&self) -> ScorerKind {
        ScorerKind::Baseline
    }

    fn revision(&self) -> &str {
        REVISION
    }

    fn cost(&self, _logs: &LogsFrame) -> ScoringCost {
        // A scores frame of few rows still has a block of 8 KiB for each column of
        // words, and Polars rounds what it reserves; that is the fixed share.
        let columns = 1 + u64::from(self.sparse);
        ScoringCost {
            fixed_bytes: 64 * 1024,
            row_bytes: 64 + columns * (32 + 2 * self.width as u64),
            evidence_row_bytes: columns * (16 + self.width as u64),
            evidence_columns: columns as u32,
        }
    }

    fn score(&mut self, logs: &LogsFrame) -> Result<DataFrame, L2dError> {
        let text = logs.frame().column("text").unwrap().clone();
        let texts: Vec<&str> = text.str().unwrap().iter().flatten().collect();
        let rows = texts.len();
        let ratios: Vec<Option<f32>> = texts
            .iter()
            .map(|text| (!self.abstains).then(|| unit(text) as f32))
            .collect();
        let why: Vec<String> = texts.iter().map(|text| self.why(text)).collect();
        let mut columns = vec![
            text.clone(),
            Column::new("scorer".into(), vec!["baseline"; rows]),
            Column::new("ratio".into(), ratios),
            Column::new("abstained".into(), vec![self.abstains; rows]),
            Column::new("why".into(), why),
        ];
        if self.sparse {
            let seldom: Vec<Option<String>> = texts
                .iter()
                .map(|text| fnv(text).is_multiple_of(3).then(|| self.why(text)))
                .collect();
            columns.push(Column::new("seldom".into(), seldom));
        }
        Ok(DataFrame::new(columns).unwrap())
    }
}

/// The blocking finding of the review: with evidence in words, batches of 2 to 64 rows
/// made triage hold up to fourteen times what it asked, because every batch's column of
/// words brought a block of its own that stayed alive in the stacked frame. The heap
/// must stay within what was asked at every batch size, and the evidence must come
/// through as the scorer gave it.
#[test]
fn evidence_in_words_stays_within_what_triage_asked_at_every_batch_size() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let logs = synthetic_logs(20_000, "words");
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    for (width, sparse) in [(13, false), (200, false), (40, true)] {
        let mut scorer = Wordy {
            width,
            sparse,
            abstains: false,
        };
        let mut reference: Option<TriageResult> = None;
        for batch_rows in [
            None,
            Some(1),
            Some(2),
            Some(3),
            Some(7),
            Some(16),
            Some(64),
            Some(1_024),
        ] {
            let settings = settings(None, 0.05, Some(500));
            let mut budget = budget_of(batch_rows);
            let (result, heap) =
                measure(|| triage_with(&logs, &mut [&mut scorer], &settings, &mut budget));
            let result = result.expect("the logs are triaged");
            let asked = asked(&result, logs.len());
            println!(
                "L2D_TRIAGE_WORDS width={width} sparse={sparse} batch_rows={batch_rows:?}: \
                 peak={} kept={} asked={asked} ({:.2} of asked)",
                heap.peak,
                heap.kept,
                heap.peak as f64 / asked as f64,
            );
            assert!(
                heap.peak <= asked,
                "width {width}, batches of {batch_rows:?}: triage took {} bytes of heap at its \
                 peak, above the {asked} it asked for",
                heap.peak
            );
            match &reference {
                Some(first) => assert!(
                    result.frame().equals_missing(first.frame()),
                    "width {width}, batches of {batch_rows:?} give another frame"
                ),
                None => {
                    // The evidence is the scorer's, row for row.
                    let why = result.frame().column("why").unwrap().str().unwrap().clone();
                    assert_eq!(why.null_count(), 0);
                    for (found, text) in why.iter().zip(&texts) {
                        assert_eq!(found, Some(scorer.why(text).as_str()));
                    }
                    if sparse {
                        let seldom = result
                            .frame()
                            .column("seldom")
                            .unwrap()
                            .str()
                            .unwrap()
                            .clone();
                        for (found, text) in seldom.iter().zip(&texts) {
                            let wanted = fnv(text).is_multiple_of(3).then(|| scorer.why(text));
                            assert_eq!(found, wanted.as_deref());
                        }
                    }
                    reference = Some(result);
                }
            }
        }
    }
}

/// The reasons of the abstained list are the longest words triage writes itself
/// (`scorer_abstained`, 16 bytes, is kept outside the 16 bytes a row of its column):
/// a run in which every log is abstained on stays within what was asked.
#[test]
fn a_run_in_which_every_log_is_abstained_on_stays_within_what_triage_asked() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let logs = synthetic_logs(20_000, "abstained");
    let mut scorer = Wordy {
        width: 13,
        sparse: false,
        abstains: true,
    };
    for batch_rows in [None, Some(7)] {
        let settings = settings(None, 0.05, None);
        let mut budget = budget_of(batch_rows);
        let (result, heap) =
            measure(|| triage_with(&logs, &mut [&mut scorer], &settings, &mut budget));
        let result = result.expect("the logs are triaged");
        let asked = asked(&result, logs.len());
        println!(
            "L2D_TRIAGE_ABSTAINED batch_rows={batch_rows:?}: peak={} asked={asked} ({:.2} of \
             asked)",
            heap.peak,
            heap.peak as f64 / asked as f64
        );
        assert_eq!(result.counts().scorer_abstained, 20_000);
        assert!(heap.peak <= asked, "{} above {asked}", heap.peak);
    }
}

// ---------------------------------------------------------------------------------------
// HTTP CSIC 2010 (on request)// ---------------------------------------------------------------------------------------

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Triage of all 25,608 distinct CSIC logs with the baseline scorer, without and with
/// a calibration: time, heap against what was asked, the lists, and the lists of the
/// calibration that were used.
///
/// The scorer is fitted on half of the labelled logs and calibrated on a quarter
/// (split by a hash of the text, so every endpoint is in every part); all logs are
/// then triaged, the fitting and calibration logs among them. This measures triage; it
/// is not an evaluation (that is task 1.12).
#[test]
#[ignore = "CSIC measurement; run in release mode on request"]
fn csic_triage_measured() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let path = match Csic2010Adapter::locate(&ProjectPaths::new(repository()), None) {
        Ok(path) => path,
        Err(error @ L2dError::Missing { .. }) => {
            eprintln!("SKIPPED csic_triage_measured: {error}");
            return;
        }
        Err(other) => panic!("locating the dataset failed: {other:?}"),
    };
    let (logs, _) = Csic2010Adapter::new()
        .read(&path)
        .expect("the dataset is readable");
    let part = |wanted: &[u64]| -> LogsFrame {
        let records: Vec<LogRecord> = logs
            .records()
            .filter(|record| wanted.contains(&(fnv(&record.text) >> 20 & 3)))
            .collect();
        LogsFrame::from_records(&records).unwrap()
    };
    let (fitting, calibration) = (part(&[0, 1]), part(&[2]));
    let host = || MemoryBudget::observe(&MemorySettings::default()).expect("free memory");

    let started = Instant::now();
    let mut scorer = BaselineScorer::fit_with(&fitting, &[], &BaselineSettings::default(), &host())
        .expect("the fitting part fits");
    let fit_seconds = started.elapsed().as_secs_f64();
    let calibrator = ConformalCalibrator::fit_scorer(
        &mut scorer,
        &calibration,
        &CalibrationSettings::default(),
        &host(),
    )
    .expect("the calibration part calibrates");
    let own_lists = calibrator.strata().len() - 2;
    println!(
        "L2D_TRIAGE_CSIC logs={} fitting={} calibration={} (good {} bad {}, {own_lists} lists \
         of groups, keeps {} bytes) fit={fit_seconds:.1}s cost={:?}",
        logs.len(),
        fitting.len(),
        calibration.len(),
        calibrator.good(),
        calibrator.bad(),
        calibrator.held_bytes(),
        scorer.cost(&logs),
    );

    let mut references: [Option<TriageResult>; 2] = [None, None];
    for (name, with_calibration, review_budget, batch_rows) in [
        ("no calibration, no budget", false, None, None),
        ("no calibration, budget 500", false, Some(500), None),
        (
            "no calibration, budget 500, batches of 1,024",
            false,
            Some(500),
            Some(1_024),
        ),
        ("calibration 0.05, no budget", true, None, None),
        ("calibration 0.05, budget 500", true, Some(500), None),
        (
            "calibration 0.05, budget 500, batches of 1,024",
            true,
            Some(500),
            Some(1_024),
        ),
    ] {
        let settings = settings(with_calibration.then_some(&calibrator), 0.05, review_budget);
        let mut budget = MemoryBudget::observe(&MemorySettings {
            max_batch_rows: batch_rows,
            ..MemorySettings::default()
        })
        .expect("free memory");
        let started = Instant::now();
        let (result, heap) =
            measure(|| triage_with(&logs, &mut [&mut scorer], &settings, &mut budget));
        let seconds = started.elapsed().as_secs_f64();
        let result = result.expect("the logs are triaged");
        let asked = asked(&result, logs.len());
        let counts = *result.counts();
        println!(
            "L2D_TRIAGE_CSIC {name}: {seconds:.3}s heap_peak={:.2} MiB kept={:.2} MiB asked={:.2} \
             MiB ({:.2} of asked) held={:.2} MiB batches={}..{} rows",
            heap.peak as f64 / MIB,
            heap.kept as f64 / MIB,
            asked as f64 / MIB,
            heap.peak as f64 / asked as f64,
            result.held_bytes() as f64 / MIB,
            result.plan().min_rows_used,
            result.plan().max_rows_used,
        );
        println!(
            "L2D_TRIAGE_CSIC {name}: review={} scored_down_by_calibration={} \
             scored_down_by_budget={} uncertain={} unlike_calibration={} scorer_abstained={}",
            counts.review,
            counts.scored_down_by_calibration,
            counts.scored_down_by_budget,
            counts.uncertain,
            counts.unlike_calibration,
            counts.scorer_abstained,
        );
        if let Some(used) = result.calibration() {
            println!(
                "L2D_TRIAGE_CSIC {name}: good p-values on a group's list {} and on the pooled \
                 list {}; bad p-values on a group's list {} and on the pooled list {}; sets \
                 good {} bad {} both {} neither {}",
                used.good.group,
                used.good.pooled,
                used.bad.group,
                used.bad.pooled,
                used.set_good,
                used.set_bad,
                used.set_both,
                used.set_neither,
            );
            assert_eq!(used.good.group + used.good.pooled, counts.logs);
            assert_eq!(used.bad.group + used.bad.pooled, counts.logs);
        }
        assert_eq!(counts.logs, 25_608);
        assert_eq!(
            counts.review + counts.scored_down() + counts.abstained(),
            counts.logs,
            "every log is in exactly one list"
        );
        assert_eq!(
            result.review().unwrap().height()
                + result.scored_down().unwrap().height()
                + result.abstained().unwrap().height(),
            logs.len()
        );
        assert!(
            heap.peak <= asked,
            "{name}: triage took {} bytes of heap at its peak, above the {asked} it asked for",
            heap.peak
        );
        // Batches of 1,024 rows give the frame of the run before them.
        let reference = &mut references[usize::from(with_calibration)];
        if batch_rows.is_some() {
            let whole = reference.as_ref().expect("the unbatched run came first");
            assert!(result.frame().equals_missing(whole.frame()), "{name}");
        } else if review_budget.is_some() {
            *reference = Some(result);
        }
    }
}
