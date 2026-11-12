//! The conformal calibration measured: the memory a calibration really takes against
//! its estimate (requirements 12.5, 12.9), and the achieved error rate on held-out
//! endpoints of HTTP CSIC 2010 (requirement 11).
//!
//! This test binary counts every heap allocation of the process (a counting global
//! allocator around the system allocator), as `l2d-baseline/tests/scorer_csic.rs` does.
//!
//! The CSIC evaluation runs on request only, in release mode:
//!
//! ```text
//! cargo test --release -p l2d-decide --test conformal_csic -- --ignored --nocapture --test-threads=1
//! ```
//!
//! It is skipped with a note when `data/csic-2010/csic_database.csv` is absent. It
//! prints what comes out, including where the achieved error rate misses the chosen
//! one, and asserts only what must hold: the counts add up, repeats are identical, the
//! memory estimate covers the heap taken, and logs that are exchangeable with the
//! calibration logs reach the chosen rate. The pass/fail gate on held-out endpoints is
//! the grouped cross-validation of task 1.12.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use l2d_baseline::{BaselineScorer, BaselineSettings, RoutingMode};
use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    Csic2010Adapter, InputAdapter, InputSummary, L2dError, Label, LogRecord, LogsFrame,
    MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_decide::ScorerKind;
use l2d_decide::conformal::{
    CalibrationError, CalibrationPoint, CalibrationSettings, ConformalCalibrator, LabelSet,
    Stratum, estimated_bytes, points_frame,
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

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// FNV-1a over the bytes of `text`: a fixed pseudo-random number per text.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A fixed pseudo-random ratio in `0.0..=1.0` for row `index`.
fn synthetic_ratio(index: u64, centre: f64) -> f32 {
    let unit = (fnv(&format!("ratio-{index}")) >> 11) as f64 / (1u64 << 53) as f64;
    (centre + (unit - 0.5) * 0.6).clamp(0.0, 1.0) as f32
}

/// Calibrates under the counting allocator and checks the estimate against the heap.
fn measured_calibration(points: &DataFrame, revision: &str) -> (ConformalCalibrator, u64) {
    let budget = budget();
    let settings = CalibrationSettings::default();
    let (calibrator, heap) = measure(|| {
        ConformalCalibrator::fit_with(points, ScorerKind::Baseline, revision, &settings, &budget)
    });
    let estimate = estimate_of(points);
    assert!(
        heap <= estimate,
        "the calibration of {} points took {heap} bytes of heap at its peak, above its \
         estimate of {estimate}",
        points.height()
    );
    (calibrator.expect("the calibration fits"), heap)
}

/// What `fit_with` asks the budget for: [`estimated_bytes`] of the number of points and
/// of the bytes of their group keys.
fn estimate_of(points: &DataFrame) -> u64 {
    let group_bytes: usize = points
        .column("group")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .flatten()
        .map(str::len)
        .sum();
    estimated_bytes(points.height() as u64, group_bytes as u64)
}

/// The estimate against the heap really taken, for shapes that stress its parts: few
/// large groups (two lists per point), many small groups and a group per point (the
/// count in Polars), long group keys, no groups, and a small set (the fixed part).
#[test]
fn the_heap_a_calibration_takes_stays_within_its_estimate() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // (points, distinct groups per label; 0 for no group, extra bytes per group key)
    for (rows, groups, padding) in [
        (200_000_u64, 20_u64, 0_usize),
        (200_000, 50_000, 0),
        (200_000, 100_000, 0),
        (200_000, 1_000, 300),
        (100_000, 50_000, 300),
        (200_000, 0, 0),
        (500, 3, 0),
    ] {
        let points: Vec<CalibrationPoint> = (0..rows)
            .map(|index| {
                let bad = index % 2 == 1;
                CalibrationPoint {
                    ratio: synthetic_ratio(index, if bad { 0.7 } else { 0.3 }),
                    label: if bad { Label::Bad } else { Label::Good },
                    group: (groups > 0).then(|| {
                        format!(
                            "/endpoint/{}/page.jsp{}",
                            index / 2 % groups,
                            "x".repeat(padding)
                        )
                    }),
                }
            })
            .collect();
        let frame = points_frame(&points).expect("the points make a frame");
        drop(points);
        let (calibrator, heap) = measured_calibration(&frame, "synthetic");
        let estimate = estimate_of(&frame);
        println!(
            "L2D_CONFORMAL_HEAP points={rows} groups={groups} key_padding={padding} lists={} \
             estimate={:.2} MiB heap_peak={:.2} MiB ({:.0} % of the estimate)",
            calibrator.strata().len(),
            estimate as f64 / MIB,
            heap as f64 / MIB,
            100.0 * heap as f64 / estimate as f64
        );
        assert_eq!(calibrator.good() + calibrator.bad(), rows);
    }
}

/// The same for the path through the baseline scorer, which adds the scores frame.
#[test]
fn the_heap_a_calibration_of_the_baseline_scorer_takes_stays_within_its_estimate() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let record = |index: u64, held_out: bool| {
        let bad = index % 2 == 1;
        let word = if bad {
            "<script>alert(1)</script>"
        } else {
            "especias"
        };
        LogRecord {
            text: format!(
                "GET http://localhost:8080/tienda1/buscar.jsp?q={word}&id={index}&part={held_out}"
            ),
            label: if bad { Label::Bad } else { Label::Good },
            group: Some(format!("/tienda1/{}.jsp", index % 40)),
            first_position: index + 1,
            occurrences: 1,
        }
    };
    let summary = |name: &str, rows: u64| InputSummary {
        source_path: PathBuf::from(name),
        records: rows,
        distinct_texts: rows,
        content_sha256: format!("{:064x}", fnv(name)),
        skipped: Vec::new(),
        label_conflicts: 0,
    };
    let fitting: Vec<LogRecord> = (0..400).map(|index| record(index, false)).collect();
    let fitting = LogsFrame::from_records(&fitting).unwrap();
    let scorer = BaselineScorer::fit_with(
        &fitting,
        &[summary("fitting", 400)],
        &BaselineSettings {
            buckets: 4_096,
            ..BaselineSettings::default()
        },
        &budget(),
    )
    .expect("the scorer fits");

    const ROWS: u64 = 40_000;
    let held_out: Vec<LogRecord> = (0..ROWS).map(|index| record(index, true)).collect();
    let held_out = LogsFrame::from_records(&held_out).unwrap();
    let inputs = [summary("held-out", ROWS)];
    let budget = budget();
    let (calibrator, heap) = measure(|| {
        ConformalCalibrator::fit_baseline(
            &scorer,
            &held_out,
            &inputs,
            &CalibrationSettings::default(),
            &budget,
        )
    });
    let calibrator = calibrator.expect("the held-out logs calibrate");
    // What `fit_baseline` asks the budget for: the calibration and 48 bytes per row.
    let group_bytes: usize = held_out
        .records()
        .map(|record| record.group.map_or(0, |group| group.len()))
        .sum();
    let estimate = estimated_bytes(ROWS, group_bytes as u64) + ROWS * 48;
    println!(
        "L2D_CONFORMAL_HEAP baseline rows={ROWS} lists={} estimate={:.2} MiB heap_peak={:.2} MiB",
        calibrator.strata().len(),
        estimate as f64 / MIB,
        heap as f64 / MIB
    );
    assert!(
        heap <= estimate,
        "{heap} bytes of heap, above the estimate of {estimate}"
    );
    assert_eq!(calibrator.good() + calibrator.bad(), ROWS);
}

// ---------------------------------------------------------------------------------------
// HTTP CSIC 2010 (on request)
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

/// What a set of test logs got: per true label the number of logs and of logs whose
/// true label is missing from the set, the number of each kind of set, and how many
/// decisions used a group's list for at least one label.
#[derive(Default, Clone, Copy)]
struct Outcome {
    logs: [u64; 2],
    missed: [u64; 2],
    /// good, bad, both, neither
    sets: [u64; 4],
    own_list: u64,
}

impl Outcome {
    fn add(&mut self, label: Label, set: LabelSet, own_list: bool) {
        let index = usize::from(label == Label::Bad);
        self.logs[index] += 1;
        self.missed[index] += u64::from(!set.contains(label));
        self.sets[match set {
            LabelSet::Good => 0,
            LabelSet::Bad => 1,
            LabelSet::Both => 2,
            LabelSet::Neither => 3,
        }] += 1;
        self.own_list += u64::from(own_list);
    }

    fn error(&self, label: Label) -> f64 {
        let index = usize::from(label == Label::Bad);
        self.missed[index] as f64 / self.logs[index].max(1) as f64
    }

    fn line(&self) -> String {
        let all = (self.logs[0] + self.logs[1]).max(1) as f64;
        let share = |count: u64| 100.0 * count as f64 / all;
        format!(
            "good n={} error={:.4} | bad n={} error={:.4} | sets: good {:.1} % bad {:.1} % both \
             {:.1} % neither {:.1} % | own group list {:.1} %",
            self.logs[0],
            self.error(Label::Good),
            self.logs[1],
            self.error(Label::Bad),
            share(self.sets[0]),
            share(self.sets[1]),
            share(self.sets[2]),
            share(self.sets[3]),
            share(self.own_list),
        )
    }
}

/// The decisions of `calibrator` on `logs` (with their ratios) at `error_rate`.
fn judge(
    calibrator: &ConformalCalibrator,
    logs: &[(&LogRecord, f32)],
    error_rate: f32,
) -> (Outcome, Vec<LabelSet>) {
    let mut outcome = Outcome::default();
    let mut sets = Vec::with_capacity(logs.len());
    for (record, ratio) in logs {
        let decision = calibrator
            .predict(*ratio, record.group.as_deref(), error_rate)
            .expect("the error rate is supported");
        let own =
            decision.good_stratum != Stratum::Pooled || decision.bad_stratum != Stratum::Pooled;
        outcome.add(record.label, decision.set, own);
        sets.push(decision.set);
    }
    (outcome, sets)
}

/// `records` with the ratio the scorer gives each.
fn with_ratio<'a>(scorer: &BaselineScorer, records: &[&'a LogRecord]) -> Vec<(&'a LogRecord, f32)> {
    records
        .iter()
        .map(|record| (*record, scorer.ratio(&record.text)))
        .collect()
}

fn mean_ratio(logs: &[(&LogRecord, f32)], label: Label) -> f64 {
    let ratios: Vec<f64> = logs
        .iter()
        .filter(|(record, _)| record.label == label)
        .map(|(_, ratio)| f64::from(*ratio))
        .collect();
    ratios.iter().sum::<f64>() / ratios.len().max(1) as f64
}

/// Where a log goes in one arrangement of the evaluation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Fitting,
    Calibration,
    /// Logs of endpoints the scorer and the calibration have seen other logs of.
    SeenEndpoints,
    /// Logs of the held-out endpoints.
    HeldOutEndpoints,
}

/// Achieved error rate on held-out CSIC endpoints, for two ways of taking the
/// calibration logs and two routing modes of the baseline scorer.
///
/// Three of the eight path groups that hold both labels are held out as test endpoints
/// (the same three as the sanity run of the baseline scorer). Of the other logs:
///
/// - **same endpoints**: every log goes, by a hash of its text, to the fitting part
///   (one half), the calibration part (one quarter) or a test part of seen endpoints
///   (one quarter). The calibration logs are exchangeable with the seen-endpoint test
///   logs, not with the held-out endpoints.
/// - **other endpoints**: two more of the mixed path groups, and every fourth of the
///   single-label groups, are kept from the scorer and used for calibration, so the
///   calibration ratios are ratios on endpoints the scorer has not seen.
#[test]
#[ignore = "CSIC evaluation; run in release mode on request"]
fn csic_achieved_error_rate_on_held_out_endpoints() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some((logs, summary)) = csic("csic_achieved_error_rate_on_held_out_endpoints") else {
        return;
    };
    let records: Vec<LogRecord> = logs
        .to_records()
        .into_iter()
        .filter(|record| record.label != Label::Unknown)
        .collect();
    let group_of = |record: &LogRecord| -> String {
        record.group.clone().expect("every CSIC log has a group")
    };

    // Per path group: (good texts, bad texts).
    let mut groups: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for record in &records {
        let counts = groups.entry(group_of(record)).or_default();
        match record.label {
            Label::Good => counts.0 += 1,
            _ => counts.1 += 1,
        }
    }
    let mixed: Vec<&String> = groups
        .iter()
        .filter(|(_, (good, bad))| *good > 0 && *bad > 0)
        .map(|(group, _)| group)
        .collect();
    assert_eq!(mixed.len(), 8, "path groups with both labels");
    let single: Vec<&String> = groups
        .iter()
        .filter(|(_, (good, bad))| *good == 0 || *bad == 0)
        .map(|(group, _)| group)
        .collect();
    // Every third of the eight mixed groups, in name order, is held out for testing.
    let held: Vec<&String> = mixed.iter().copied().skip(1).step_by(3).collect();
    // For "other endpoints": the first and the fourth mixed group calibrate.
    let calibration_groups: Vec<&String> = [mixed[0], mixed[3]]
        .into_iter()
        .chain(single.iter().copied().skip(2).step_by(4))
        .collect();
    for group in &mixed {
        let (good, bad) = groups[*group];
        println!(
            "L2D_CONFORMAL_CSIC_GROUP {group} good={good} bad={bad} held_out={} \
             calibrates_in_other_endpoints={}",
            held.contains(group),
            calibration_groups.contains(group)
        );
    }

    type Arrangement<'a> = (&'a str, Box<dyn Fn(&LogRecord) -> Part + 'a>);
    let arrangements: [Arrangement<'_>; 2] = [
        (
            "same-endpoints",
            Box::new(|record: &LogRecord| {
                if held.contains(&&group_of(record)) {
                    return Part::HeldOutEndpoints;
                }
                match fnv(&record.text) >> 20 & 3 {
                    0 | 1 => Part::Fitting,
                    2 => Part::Calibration,
                    _ => Part::SeenEndpoints,
                }
            }),
        ),
        (
            "other-endpoints",
            Box::new(|record: &LogRecord| {
                let group = group_of(record);
                if held.contains(&&group) {
                    Part::HeldOutEndpoints
                } else if calibration_groups.contains(&&group) {
                    Part::Calibration
                } else {
                    Part::Fitting
                }
            }),
        ),
    ];

    for (arrangement, part_of) in &arrangements {
        let part = |wanted: Part| -> Vec<&LogRecord> {
            records
                .iter()
                .filter(|record| part_of(record) == wanted)
                .collect()
        };
        let fitting_records = part(Part::Fitting);
        let calibration_records = part(Part::Calibration);
        let seen_records = part(Part::SeenEndpoints);
        let held_records = part(Part::HeldOutEndpoints);
        assert_eq!(
            fitting_records.len()
                + calibration_records.len()
                + seen_records.len()
                + held_records.len(),
            records.len(),
            "every labelled log is in exactly one part"
        );
        // The logs frame holds distinct texts, so the parts share no text.
        let fitting: Vec<LogRecord> = fitting_records.iter().map(|r| (*r).clone()).collect();
        let fitting = LogsFrame::from_records(&fitting).expect("the fitting part is a logs frame");

        for routing in [RoutingMode::Keep, RoutingMode::Strip] {
            let settings = BaselineSettings {
                routing,
                ..BaselineSettings::default()
            };
            let started = Instant::now();
            let scorer = BaselineScorer::fit_with(
                &fitting,
                std::slice::from_ref(&summary),
                &settings,
                &budget(),
            )
            .expect("the fit succeeds");
            let fit_seconds = started.elapsed().as_secs_f64();
            let tag = format!("{arrangement} {}", mode_name(routing));

            // The whole file is a fitting input of this scorer, so calibration through
            // the scorer refuses any part of it; the parts are kept disjoint here.
            let calibration_logs: Vec<LogRecord> =
                calibration_records.iter().map(|r| (*r).clone()).collect();
            let calibration_logs = LogsFrame::from_records(&calibration_logs).unwrap();
            assert_eq!(
                ConformalCalibrator::fit_baseline(
                    &scorer,
                    &calibration_logs,
                    std::slice::from_ref(&summary),
                    &CalibrationSettings::default(),
                    &budget(),
                )
                .unwrap_err(),
                CalibrationError::SameInputAsFitting {
                    content_sha256: summary.content_sha256.clone()
                }
            );

            let calibration = with_ratio(&scorer, &calibration_records);
            let seen = with_ratio(&scorer, &seen_records);
            let held_out = with_ratio(&scorer, &held_records);
            let points: Vec<CalibrationPoint> = calibration
                .iter()
                .map(|(record, ratio)| CalibrationPoint {
                    ratio: *ratio,
                    label: record.label,
                    group: record.group.clone(),
                })
                .collect();
            let points = points_frame(&points).unwrap();
            let started = Instant::now();
            let (calibrator, heap) = measured_calibration(&points, scorer.revision());
            let calibrate_seconds = started.elapsed().as_secs_f64();
            let again = measured_calibration(&points, scorer.revision()).0;
            assert_eq!(again, calibrator, "a second calibration is identical");

            let own_lists = calibrator
                .strata()
                .iter()
                .filter(|count| count.stratum != Stratum::Pooled)
                .count();
            println!(
                "L2D_CONFORMAL_CSIC {tag} fitted_rows={} fit_seconds={fit_seconds:.1} \
                 gradient_converged={} calibration: good={} bad={} group_lists={own_lists} \
                 smallest_supported={:.5} seconds={calibrate_seconds:.3} heap_peak={:.2} MiB \
                 estimate={:.2} MiB",
                scorer.fit_report().rows,
                scorer.gradient_converged(),
                calibrator.good(),
                calibrator.bad(),
                calibrator.smallest_supported(),
                heap as f64 / MIB,
                estimate_of(&points) as f64 / MIB,
            );
            for (name, logs) in [
                ("calibration", &calibration),
                ("seen-endpoints", &seen),
                ("held-out-endpoints", &held_out),
            ] {
                if logs.is_empty() {
                    continue;
                }
                println!(
                    "L2D_CONFORMAL_CSIC {tag} mean ratio on {name}: good {:.3} bad {:.3}",
                    mean_ratio(logs, Label::Good),
                    mean_ratio(logs, Label::Bad)
                );
            }

            for error_rate in [0.05_f32, 0.1, 0.2] {
                let started = Instant::now();
                let (outcome, sets) = judge(&calibrator, &held_out, error_rate);
                let seconds = started.elapsed().as_secs_f64();
                assert_eq!(outcome.logs[0] + outcome.logs[1], held_out.len() as u64);
                assert_eq!(outcome.sets.iter().sum::<u64>(), held_out.len() as u64);
                assert_eq!(
                    outcome.own_list, 0,
                    "held-out endpoints have no list of their own"
                );
                assert_eq!(
                    judge(&calibrator, &held_out, error_rate).1,
                    sets,
                    "repeatable"
                );
                println!(
                    "L2D_CONFORMAL_CSIC {tag} e={error_rate} HELD-OUT ENDPOINTS {} \
                     ({seconds:.2} s)",
                    outcome.line()
                );
                if error_rate == 0.05 {
                    for group in &held {
                        let of_group: Vec<(&LogRecord, f32)> = held_out
                            .iter()
                            .filter(|(record, _)| record.group.as_ref() == Some(*group))
                            .copied()
                            .collect();
                        println!(
                            "L2D_CONFORMAL_CSIC {tag} e={error_rate}   {group}: {}",
                            judge(&calibrator, &of_group, error_rate).0.line()
                        );
                    }
                }
                if !seen.is_empty() {
                    let (outcome, _) = judge(&calibrator, &seen, error_rate);
                    println!(
                        "L2D_CONFORMAL_CSIC {tag} e={error_rate} seen endpoints     {}",
                        outcome.line()
                    );
                    // These logs are exchangeable with the calibration logs, list by
                    // list, so here the chosen rate must be reached. With more than
                    // 2,000 logs per label one standard deviation is below 0.01.
                    for label in [Label::Good, Label::Bad] {
                        assert!(
                            outcome.error(label) <= f64::from(error_rate) + 0.03,
                            "{tag} e={error_rate}: {label:?} logs of seen endpoints miss at {:.4}",
                            outcome.error(label)
                        );
                    }
                }
            }

            // The calibration saves and loads back identically, below a scratch root.
            let scratch = std::env::temp_dir().join(format!(
                "l2d-decide-conformal-csic-{}-{arrangement}-{}",
                std::process::id(),
                mode_name(routing)
            ));
            let _ = std::fs::remove_dir_all(&scratch);
            let dir = calibrator
                .save(&ProjectPaths::new(&scratch), "csic")
                .expect("the calibration saves");
            let loaded = ConformalCalibrator::load(&dir).expect("the calibration loads");
            assert_eq!(loaded, calibrator);
            assert_eq!(
                judge(&loaded, &held_out, 0.05).1,
                judge(&calibrator, &held_out, 0.05).1
            );
            let _ = std::fs::remove_dir_all(&scratch);
        }
    }
}
