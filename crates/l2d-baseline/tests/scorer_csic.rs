//! The baseline scorer measured: the memory a fit really takes against its estimate
//! (requirements 12.5, 12.9), and a sanity run on HTTP CSIC 2010.
//!
//! This test binary counts every heap allocation of the process (a counting global
//! allocator around the system allocator), so the memory figures do not depend on how
//! the operating system trims working sets.
//!
//! The CSIC tests run on request only, in release mode, one at a time:
//!
//! ```text
//! cargo test --release -p l2d-baseline --test scorer_csic -- --ignored --nocapture --test-threads=1
//! ```
//!
//! They are skipped with a note when `data/csic-2010/csic_database.csv` is absent. They
//! print figures and assert only what must hold on any host (the memory estimate covers
//! the heap the fit takes; ratios are in range). The pass/fail gate on ranking quality
//! is the grouped cross-validation of task 1.12.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use l2d_baseline::{BaselineScorer, BaselineSettings, RoutingMode};
use l2d_core::{
    Csic2010Adapter, InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat,
    LogRecord, LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};

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

/// Runs `run` and returns its result and by how many bytes the heap grew at its peak.
fn measure<T>(run: impl FnOnce() -> T) -> (T, u64) {
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = run();
    let peak = PEAK.load(Ordering::Relaxed);
    (result, peak.saturating_sub(before) as u64)
}

fn budget() -> MemoryBudget {
    MemoryBudget::observe(&MemorySettings::default()).expect("the host has free memory")
}

fn repository() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Fits under the counting allocator and checks the estimate against the heap taken.
fn measured_fit(
    logs: &LogsFrame,
    inputs: &[InputSummary],
    settings: &BaselineSettings,
) -> (BaselineScorer, u64, f64) {
    let budget = budget();
    let started = Instant::now();
    let (scorer, heap) = measure(|| BaselineScorer::fit_with(logs, inputs, settings, &budget));
    let seconds = started.elapsed().as_secs_f64();
    let scorer = scorer.expect("the fit succeeds");
    let estimate = scorer.fit_report().estimated_bytes;
    assert!(
        heap <= estimate,
        "the fit took {heap} bytes of heap at its peak, above its estimate of {estimate}"
    );
    (scorer, heap, seconds)
}

/// Area under the ROC curve from ranks, ties counting one half.
fn roc_auc(scored: &mut [(f32, bool)]) -> f64 {
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    let bad = scored.iter().filter(|(_, bad)| *bad).count() as f64;
    let good = scored.len() as f64 - bad;
    let mut rank_sum = 0.0;
    let mut start = 0;
    while start < scored.len() {
        let mut end = start;
        while end < scored.len() && scored[end].0 == scored[start].0 {
            end += 1;
        }
        // Rows start..end tie; each gets the mean of the ranks start+1..=end.
        let rank = (start + 1 + end) as f64 / 2.0;
        rank_sum += rank * scored[start..end].iter().filter(|(_, bad)| *bad).count() as f64;
        start = end;
    }
    (rank_sum - bad * (bad + 1.0) / 2.0) / (bad * good)
}

// ---------------------------------------------------------------------------------------
// Always on: the fixture
// ---------------------------------------------------------------------------------------

#[test]
fn the_heap_a_fixture_fit_takes_stays_within_its_estimate() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let fixture = repository()
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv");
    let (logs, summary) = LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&fixture)
        .expect("the fixture is readable");
    for routing in [RoutingMode::Keep, RoutingMode::Isolate] {
        let settings = BaselineSettings {
            routing,
            ..BaselineSettings::default()
        };
        let (scorer, heap, seconds) =
            measured_fit(&logs, std::slice::from_ref(&summary), &settings);
        let report = scorer.fit_report();
        println!(
            "L2D_SCORER_MEMORY fixture {routing:?}: rows={} entries={} estimate={:.1} MiB \
             heap_peak={:.1} MiB ({:.2} of the estimate) iterations={} fit_seconds={seconds:.2}",
            report.rows,
            report.stored_entries,
            report.estimated_bytes as f64 / MIB,
            heap as f64 / MIB,
            heap as f64 / report.estimated_bytes as f64,
            report.iterations
        );
        // The estimate is not wildly above the need either.
        assert!(
            heap * 4 > report.estimated_bytes,
            "{heap} against {report:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------
// On request: HTTP CSIC 2010
// ---------------------------------------------------------------------------------------

/// The CSIC logs frame, or `None` with a note when the dataset is absent.
fn csic(test: &str) -> Option<(LogsFrame, InputSummary)> {
    let path = match Csic2010Adapter::locate(&ProjectPaths::new(repository()), None) {
        Ok(path) => path,
        Err(error @ L2dError::Missing { .. }) => {
            eprintln!("SKIPPED {test}: {error}");
            return None;
        }
        Err(other) => panic!("locating the dataset failed: {other:?}"),
    };
    Some(
        Csic2010Adapter::new()
            .read(&path)
            .expect("the dataset is readable"),
    )
}

fn mode_name(routing: RoutingMode) -> &'static str {
    match routing {
        RoutingMode::Keep => "keep",
        RoutingMode::Strip => "strip",
        RoutingMode::Isolate => "isolate",
    }
}

/// How many iterations the whole dataset needs at which tolerance, and what a looser
/// tolerance costs in the ratios. This is where the default cap and tolerance come from.
#[test]
#[ignore = "CSIC measurement; run in release mode on request"]
fn csic_iterations_and_tolerance() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some((logs, summary)) = csic("csic_iterations_and_tolerance") else {
        return;
    };
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    for routing in [RoutingMode::Keep, RoutingMode::Strip, RoutingMode::Isolate] {
        let mut reference: Option<Vec<f32>> = None;
        // Tightest first: it is the reference the looser fits are compared with.
        for tolerance in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] {
            let settings = BaselineSettings {
                routing,
                tolerance,
                max_iter: 5_000,
                ..BaselineSettings::default()
            };
            let (scorer, heap, seconds) =
                measured_fit(&logs, std::slice::from_ref(&summary), &settings);
            let report = scorer.fit_report();
            let ratios: Vec<f32> = texts.iter().map(|text| scorer.ratio(text)).collect();
            let reference = reference.get_or_insert_with(|| ratios.clone());
            let largest = ratios
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "L2D_SCORER_CSIC_TOLERANCE {} tolerance={tolerance:e} rows={} iterations={} \
                 termination={:?} gradient_norm={:.3e} objective={:.6} fit_seconds={seconds:.2} \
                 largest_ratio_difference_to_1e-6={largest:.2e} heap_peak={:.1} MiB \
                 estimate={:.1} MiB",
                mode_name(routing),
                report.rows,
                report.iterations,
                report.termination,
                report.gradient_norm,
                report.objective,
                heap as f64 / MIB,
                report.estimated_bytes as f64 / MIB,
            );
        }
    }
}

/// Fits on CSIC without some of the path groups that hold both labels and prints the ROC
/// AUC on the held-out groups, per routing mode, with the default settings.
#[test]
#[ignore = "CSIC sanity run; run in release mode on request"]
fn csic_held_out_groups_sanity_run() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some((logs, summary)) = csic("csic_held_out_groups_sanity_run") else {
        return;
    };
    let records = logs.to_records();

    // Per path group: (good texts, bad texts).
    let mut groups: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for record in &records {
        let counts = groups
            .entry(record.group.as_deref().expect("every CSIC log has a group"))
            .or_default();
        match record.label {
            Label::Good => counts.0 += 1,
            Label::Bad => counts.1 += 1,
            Label::Unknown => {}
        }
    }
    let mixed: Vec<&str> = groups
        .iter()
        .filter(|(_, (good, bad))| *good > 0 && *bad > 0)
        .map(|(group, _)| *group)
        .collect();
    assert_eq!(mixed.len(), 8, "path groups with both labels");
    // Every third of the eight mixed groups, in name order, is held out.
    let held: Vec<&str> = mixed.iter().copied().skip(1).step_by(3).collect();
    for group in &mixed {
        let (good, bad) = groups[group];
        println!(
            "L2D_SCORER_CSIC_GROUP {group} good={good} bad={bad} held_out={}",
            held.contains(group)
        );
    }
    let is_held = |record: &LogRecord| held.contains(&record.group.as_deref().unwrap_or(""));
    let fitting: Vec<LogRecord> = records.iter().filter(|r| !is_held(r)).cloned().collect();
    let fitting = LogsFrame::from_records(&fitting).expect("the fitting side is a logs frame");
    let held_out: Vec<&LogRecord> = records
        .iter()
        .filter(|r| is_held(r) && r.label != Label::Unknown)
        .collect();

    for routing in [RoutingMode::Keep, RoutingMode::Strip, RoutingMode::Isolate] {
        let settings = BaselineSettings {
            routing,
            ..BaselineSettings::default()
        };
        let (scorer, heap, seconds) =
            measured_fit(&fitting, std::slice::from_ref(&summary), &settings);
        let report = scorer.fit_report();
        let mut scored: Vec<(f32, bool)> = held_out
            .iter()
            .map(|record| (scorer.ratio(&record.text), record.label == Label::Bad))
            .collect();
        assert!(scored.iter().all(|(ratio, _)| (0.0..=1.0).contains(ratio)));
        let bad = scored.iter().filter(|(_, bad)| *bad).count();
        let auc = roc_auc(&mut scored);
        println!(
            "L2D_SCORER_CSIC_SANITY {} fitted_rows={} (good {}, bad {}, without features {}) \
             held_out_rows={} (good {}, bad {bad}) roc_auc={auc:.4} fit_seconds={seconds:.2} \
             iterations={} termination={:?} gradient_norm={:.3e} gradient_converged={} \
             entries={} estimate={:.1} MiB heap_peak={:.1} MiB",
            mode_name(routing),
            report.rows,
            report.good,
            report.bad,
            report.no_feature_rows,
            scored.len(),
            scored.len() - bad,
            report.iterations,
            report.termination,
            report.gradient_norm,
            scorer.gradient_converged(),
            report.stored_entries,
            report.estimated_bytes as f64 / MIB,
            heap as f64 / MIB,
        );
    }
}

/// One fit of the whole dataset with the default settings, for measuring the working
/// set of the process from outside. With `L2D_SCORER_PAUSE_MS` set, the test waits that
/// long before and after the fit, so that an observer sees the working set without the
/// fit, its peak during the fit, and what remains.
#[test]
#[ignore = "CSIC measurement; run in release mode on request"]
fn csic_whole_dataset_fit() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some((logs, summary)) = csic("csic_whole_dataset_fit") else {
        return;
    };
    let pause = std::env::var("L2D_SCORER_PAUSE_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(std::time::Duration::from_millis);
    let routing = match std::env::var("L2D_SCORER_ROUTING").as_deref() {
        Ok("strip") => RoutingMode::Strip,
        Ok("isolate") => RoutingMode::Isolate,
        _ => RoutingMode::Keep,
    };
    let settings = BaselineSettings {
        routing,
        ..BaselineSettings::default()
    };
    println!("L2D_SCORER_CSIC_WHOLE dataset read; fit starts after the pause");
    if let Some(pause) = pause {
        std::thread::sleep(pause);
    }
    let (scorer, heap, seconds) = measured_fit(&logs, std::slice::from_ref(&summary), &settings);
    let report = scorer.fit_report();
    println!(
        "L2D_SCORER_CSIC_WHOLE {} rows={} entries={} iterations={} termination={:?} \
         gradient_norm={:.3e} fit_seconds={seconds:.2} estimate={:.1} MiB ({} bytes) \
         heap_peak={:.1} MiB ({heap} bytes, {:.2} of the estimate)",
        mode_name(routing),
        report.rows,
        report.stored_entries,
        report.iterations,
        report.termination,
        report.gradient_norm,
        report.estimated_bytes as f64 / MIB,
        report.estimated_bytes,
        heap as f64 / MIB,
        heap as f64 / report.estimated_bytes as f64,
    );
    if let Some(pause) = pause {
        std::thread::sleep(pause);
    }
}
