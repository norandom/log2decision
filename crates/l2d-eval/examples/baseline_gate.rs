//! Task 1.12: the grouped evaluation of the baseline scorer on HTTP CSIC 2010.
//!
//! ```text
//! cargo run --release -j 4 -p l2d-eval --example baseline_gate
//! cargo run --release -j 4 -p l2d-eval --example baseline_gate -- --workers 6 --out reports
//! ```
//!
//! Options: `--data <path>` (default `data/csic-2010/csic_database.csv` of the
//! repository), `--out <dir>` (default `reports/` of the repository), `--workers <n>`
//! (default 4), `--memory-limit <size>` (default: half of the free memory).
//!
//! It writes three files into the output directory and nothing anywhere else:
//!
//! - `<date>-csic-2010-baseline.json`: every figure, the settings, the dataset and its
//!   hash, the fold membership. Two runs on the same day write the same bytes.
//! - `<date>-csic-2010-baseline.md`: the same as tables.
//! - `<date>-csic-2010-baseline.run.json`: what was measured about the run itself and
//!   does not repeat: times, cost, memory.
//!
//! Four fold arrangements, all by URL path, fixed by rule before anything is fitted:
//!
//! - `a-grouped-5`: five folds of nearly equal numbers of logs (the task's own).
//! - `b-science-gate-5`: the five folds of the science gate (task 2.1); the membership
//!   is compared with the hashes in its run records.
//! - `c-twins-together-5`: the science gate's rule with `registro.jsp` and `editar.jsp`,
//!   which share one form, as one unit, so they are held out together.
//! - `d-leave-one-endpoint-out`: each path with at least 100 good and 100 bad logs held
//!   out alone, all other paths fitted.
//!
//! Routing modes: `strip` and `keep` with the choice of the penalty and the calibration;
//! `isolate` at the default penalty only.
//!
//! The dataset is data only; nothing in it is executed.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime};

use l2d_baseline::{BaselineScorer, BaselineSettings, RoutingMode};
use l2d_core::input::content_sha256;
use l2d_core::memory::parse_memory_size;
use l2d_core::{
    Csic2010Adapter, InputAdapter, L2dError, LogRecord, LogsFrame, MemoryBudget, MemorySettings,
    ProjectPaths,
};
use l2d_eval::evaluate::{
    Arrangement, DEFAULT_SMALL_ENDPOINT, EvalLogs, EvalSettings, GATE_ROC_AUC, Progress,
    evaluate_with_progress, rule_summary,
};
use l2d_eval::report::{Dataset, Report, date_of, routing_name, write_report};
use l2d_eval::split::{Folds, units};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------------------
// The heap of the process, counted (as in `l2d-baseline/tests/scorer_csic.rs`)
// ---------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------
// What is fixed before anything is fitted
// ---------------------------------------------------------------------------------------

/// The two CSIC endpoints with the same 13-parameter form (review of task 2.1).
const TWINS: [&str; 2] = [
    "/tienda1/miembros/editar.jsp",
    "/tienda1/publico/registro.jsp",
];

/// `held_out_paths_sha256` and `held_out_logs` of the science gate's run records
/// `cv5-f0-baseline.json` to `cv5-f4-baseline.json` (task 2.1,
/// `D:\BuildArtifacts\log2decision\science-2.1\runs\`), fold 0 to 4.
const SCIENCE_GATE_FOLDS: [(&str, usize); 5] = [
    (
        "3765ae6e21386bdb7cc0194d133bc5029668d7538fc3d7184d51d6032e706556",
        5371,
    ),
    (
        "2257591e06ed4f4bf4f0623876c4627cea61d6a6836f16f995c5e3f9aa2971e8",
        5271,
    ),
    (
        "0ed75e763ed959f8e4dcb59a0c5e1085ae06cfc6f12827503cf400d76ab3781b",
        5264,
    ),
    (
        "3b59128dc5a4776abd6acceec9de33563b9ac93c53b436cc61c827e3e5772d58",
        4925,
    ),
    (
        "4f33cee1376155e651435390c8ad0ba616b5ba5ccabdc453a09cbfcf6d309853",
        4777,
    ),
];

const FOLDS: usize = 5;

struct Options {
    data: Option<PathBuf>,
    out: Option<PathBuf>,
    workers: usize,
    memory: MemorySettings,
}

fn options(args: &[String]) -> Result<Options, L2dError> {
    let invalid = |detail: &str| L2dError::InvalidInput {
        detail: format!(
            "{detail}; usage: baseline_gate [--data PATH] [--out DIR] [--workers N] \
             [--memory-limit SIZE]"
        ),
    };
    let mut found = Options {
        data: None,
        out: None,
        workers: 4,
        memory: MemorySettings::default(),
    };
    for pair in args.chunks(2) {
        match pair {
            [name, value] if name == "--data" => found.data = Some(PathBuf::from(value)),
            [name, value] if name == "--out" => found.out = Some(PathBuf::from(value)),
            [name, value] if name == "--workers" => {
                found.workers = value
                    .parse()
                    .ok()
                    .filter(|workers| *workers > 0)
                    .ok_or_else(|| invalid("unreadable --workers"))?;
            }
            [name, value] if name == "--memory-limit" => {
                found.memory.memory_limit_bytes = Some(parse_memory_size(value)?);
            }
            _ => return Err(invalid("unknown option")),
        }
    }
    Ok(found)
}

fn arrangements(data: &EvalLogs) -> Result<(Vec<Arrangement>, Value), L2dError> {
    let mixed = data.mixed_groups();
    let counts = data.group_counts();
    for twin in TWINS {
        if !mixed.contains(twin) {
            return Err(L2dError::InvalidInput {
                detail: format!("the dataset has no path '{twin}' with both labels"),
            });
        }
    }
    let twins = vec![
        TWINS
            .iter()
            .map(|twin| twin.to_string())
            .collect::<Vec<_>>(),
    ];
    let free = units(data.groups(), &[])?;
    let tied = units(data.groups(), &twins)?;
    let large: Vec<String> = mixed
        .iter()
        .filter(|group| {
            let (good, bad) = counts[*group];
            good >= DEFAULT_SMALL_ENDPOINT && bad >= DEFAULT_SMALL_ENDPOINT
        })
        .cloned()
        .collect();
    let small: Vec<Value> = mixed
        .iter()
        .filter(|group| !large.contains(group))
        .map(|group| json!({ "path": group, "good": counts[group].0, "bad": counts[group].1 }))
        .collect();

    let science = Folds::dealt(&free, FOLDS, &mixed)?;
    let science_check: Vec<Value> = SCIENCE_GATE_FOLDS
        .iter()
        .enumerate()
        .map(|(fold, (sha256, logs))| {
            let held = science
                .mask(fold, data.groups())
                .iter()
                .filter(|held| **held)
                .count();
            json!({
                "fold": fold,
                "science_gate_held_out_paths_sha256": sha256,
                "held_out_paths_sha256": science.membership_sha256(fold),
                "science_gate_held_out_logs": logs,
                "held_out_logs": held,
                "identical": science.membership_sha256(fold) == *sha256 && held == *logs,
            })
        })
        .collect();
    let identical = science_check
        .iter()
        .all(|fold| fold["identical"] == json!(true));

    let found = vec![
        Arrangement {
            name: "a-grouped-5".into(),
            rule: "The task's own: five folds by URL path of nearly equal numbers of logs. The \
                   paths are taken from the largest to the smallest (ties in byte order) and \
                   each goes to the fold with the fewest logs so far (the rule of scikit-learn's \
                   GroupKFold). No label is read."
                .into(),
            folds: Folds::balanced(&free, FOLDS)?,
            together: Vec::new(),
        },
        Arrangement {
            name: "b-science-gate-5".into(),
            rule: "The folds of the science gate (task 2.1, cv5_fold_of_group): the paths that \
                   hold both labels are dealt in turn from the largest to the smallest, so each \
                   fold holds one large endpoint; the single-label paths are dealt in turn in \
                   byte order."
                .into(),
            folds: science,
            together: Vec::new(),
        },
        Arrangement {
            name: "c-twins-together-5".into(),
            rule: "The rule of b-science-gate-5 with registro.jsp and editar.jsp, which share \
                   one 13-parameter form, as one unit: they are held out together in one fold, \
                   and stay together in the inner folds of the other folds."
                .into(),
            folds: Folds::dealt(&tied, FOLDS, &mixed)?,
            together: twins,
        },
        Arrangement {
            name: "d-leave-one-endpoint-out".into(),
            rule: format!(
                "Each path with at least {DEFAULT_SMALL_ENDPOINT} good and {DEFAULT_SMALL_ENDPOINT} \
                 bad logs is held out alone, in byte order of the paths; every other path, the \
                 single-label ones included, is fitted. A twin is held out while the other is \
                 fitted."
            ),
            folds: Folds::leave_one_out(&free, &large)?,
            together: Vec::new(),
        },
    ];
    let notes = json!({
        "twin_endpoints": TWINS,
        "science_gate_folds": {
            "arrangement": "b-science-gate-5",
            "identical_to_the_run_records": identical,
            "folds": science_check,
        },
        "leave_one_endpoint_out": {
            "threshold_logs_per_label": DEFAULT_SMALL_ENDPOINT,
            "held_out_in_turn": large,
            "too_small": small,
        },
        "isolate": "evaluated at the default penalty only: no choice of the penalty, no \
                    sensitivity table, no calibration",
    });
    Ok((found, notes))
}

fn modes() -> Vec<EvalSettings> {
    vec![
        EvalSettings::new(RoutingMode::Strip),
        EvalSettings::new(RoutingMode::Keep),
        EvalSettings {
            penalties: vec![BaselineSettings::default().l2],
            inner_folds: 0,
            ..EvalSettings::new(RoutingMode::Isolate)
        },
    ]
}

// ---------------------------------------------------------------------------------------
// Cost (requirement 9.4), measured on one thread before the folds are evaluated
// ---------------------------------------------------------------------------------------

fn cost(
    data: &EvalLogs,
    plan: &Arrangement,
    modes: &[EvalSettings],
    budget: &MemoryBudget,
) -> Result<Value, L2dError> {
    let mask = plan.folds.mask(0, data.groups());
    let side = |held: bool| -> Result<LogsFrame, L2dError> {
        let records: Vec<LogRecord> = data
            .records()
            .iter()
            .zip(&mask)
            .filter(|(_, in_fold)| **in_fold == held)
            .map(|(record, _)| record.clone())
            .collect();
        LogsFrame::from_records(&records)
    };
    let (fitting, held) = (side(false)?, side(true)?);
    let first_text = data
        .records()
        .iter()
        .zip(&mask)
        .find(|(_, in_fold)| **in_fold)
        .map(|(record, _)| record.text.clone())
        .unwrap_or_default();
    let mut found = Vec::new();
    for settings in modes {
        let started = Instant::now();
        let scorer = BaselineScorer::fit_with(&fitting, &[], &settings.baseline, budget)?;
        let fit_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let _ = scorer.ratio(&first_text);
        let first_result_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let mut rates = Vec::new();
        for _ in 0..3 {
            let started = Instant::now();
            let scores = scorer.score(&held)?;
            let seconds = started.elapsed().as_secs_f64();
            rates.push(scores.height() as f64 / seconds);
        }
        rates.sort_by(f64::total_cmp);
        let report = scorer.fit_report();
        eprintln!(
            "cost {}: fit of {} logs {fit_seconds:.2} s ({} iterations), scoring {:.0} logs/s",
            routing_name(settings.baseline.routing),
            report.rows,
            report.iterations,
            rates[1]
        );
        found.push(json!({
            "routing": routing_name(settings.baseline.routing),
            "penalty": settings.baseline.l2,
            "fitting_logs": report.rows,
            "fit_wall_seconds": fit_seconds,
            "fit_iterations": report.iterations,
            "fit_estimated_bytes": report.estimated_bytes,
            "scored_logs": held.len(),
            "scoring_logs_per_second": { "lowest": rates[0], "median": rates[1], "highest": rates[2] },
            "first_result_ms_after_the_fit": first_result_ms,
        }));
    }
    Ok(json!({
        "what": "arrangement a-grouped-5, fold 0, default penalty; one thread, nothing else \
                 running in this process; scoring is the scores frame of the held-out side, \
                 three times",
        "energy_proxy": "process CPU time was not measured; this section runs on one thread, \
                         so its CPU time is at most its wall time",
        "modes": found,
    }))
}

// ---------------------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------------------

fn run(args: &[String]) -> Result<(), L2dError> {
    let options = options(args)?;
    // The repository: two levels above this crate.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap_or(Path::new("."));
    let paths = ProjectPaths::new(root);
    let out = options.out.clone().unwrap_or_else(|| paths.reports_dir());
    let source = Csic2010Adapter::locate(&paths, options.data.as_deref())?;

    let started = Instant::now();
    let mut budget = MemoryBudget::observe(&options.memory)?;
    let (logs, summary, plan) = Csic2010Adapter::new().read_with(&source, &mut budget)?;
    let data = EvalLogs::new(&logs);
    // The logs frame and the evaluation's copy of its rows stay for the whole run.
    budget.hold(2 * data.held_bytes());
    budget.require_held(data.len() as u64)?;
    eprintln!(
        "read {} records, {} distinct texts, {} evaluated, {} paths, {} with both labels; \
         memory limit {} MiB",
        summary.records,
        summary.distinct_texts,
        data.len(),
        data.group_counts().len(),
        data.mixed_groups().len(),
        budget.limit_bytes() / (1024 * 1024),
    );

    let (arrangements, notes) = arrangements(&data)?;
    let modes = modes();
    for plan in &arrangements {
        for fold in 0..plan.folds.len() {
            let mixed = data.mixed_groups();
            let both: BTreeSet<&String> = plan.folds.held_out(fold).intersection(&mixed).collect();
            eprintln!("{} fold {fold}: {both:?}", plan.name);
        }
    }

    let cost = cost(&data, &arrangements[0], &modes, &budget)?;

    let evaluation = Instant::now();
    let names: Vec<&str> = arrangements.iter().map(|plan| plan.name.as_str()).collect();
    let fold_times = std::sync::Mutex::new(Vec::new());
    let results = evaluate_with_progress(
        &data,
        &arrangements,
        &modes,
        options.workers,
        &budget,
        &|progress: Progress| {
            eprintln!(
                "{}/{} {} {} fold {}: {} fits, {:.0} s",
                progress.done,
                progress.total,
                names[progress.arrangement],
                routing_name(modes[progress.mode].baseline.routing),
                progress.fold,
                progress.timing.fits,
                progress.timing.total_seconds,
            );
            fold_times
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(json!({
                    "arrangement": names[progress.arrangement],
                    "routing": routing_name(modes[progress.mode].baseline.routing),
                    "fold": progress.fold,
                    "fits": progress.timing.fits,
                    "fit_wall_seconds": progress.timing.fit_seconds,
                    "scoring_wall_seconds": progress.timing.score_seconds,
                    "wall_seconds": progress.timing.total_seconds,
                }));
        },
    )?;
    let evaluation_seconds = evaluation.elapsed().as_secs_f64();
    let peak = PEAK.load(Ordering::Relaxed) as u64;

    let date = date_of(SystemTime::now());
    let stem = format!("{date}-csic-2010-baseline");
    let report = Report {
        task: "npu-log-decision 1.12: baseline validation, grouped cross-validation on HTTP \
               CSIC 2010"
            .into(),
        date,
        dataset: Dataset {
            name: "csic-2010".into(),
            summary: summary.clone(),
        },
        data: &data,
        arrangements: &arrangements,
        modes: &modes,
        results: &results,
        notes,
    };
    let (json_path, text_path) = write_report(&out, &stem, &report)?;
    let written = |path: &Path| -> Result<String, L2dError> {
        std::fs::read(path)
            .map(|bytes| content_sha256(&bytes))
            .map_err(|error| L2dError::Runtime {
                detail: format!("reading {} back failed: {error}", path.display()),
            })
    };
    let (json_sha256, text_sha256) = (written(&json_path)?, written(&text_path)?);

    let mut fold_times = fold_times
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    fold_times.sort_by_key(|fold| {
        (
            fold["arrangement"].as_str().map(str::to_string),
            fold["routing"].as_str().map(str::to_string),
            fold["fold"].as_u64(),
        )
    });
    let fits: u64 = fold_times
        .iter()
        .filter_map(|fold| fold["fits"].as_u64())
        .sum();
    let fit_seconds: f64 = fold_times
        .iter()
        .filter_map(|fold| fold["fit_wall_seconds"].as_f64())
        .sum();
    let run_path = out.join(format!("{stem}.run.json"));
    let run_record = json!({
        "report": { "json": json_path.display().to_string(), "json_sha256": json_sha256,
                    "markdown": text_path.display().to_string(), "markdown_sha256": text_sha256 },
        "workers": options.workers,
        "wall_seconds": { "all": started.elapsed().as_secs_f64(), "folds": evaluation_seconds },
        "fits": { "count": fits, "wall_seconds_summed_over_workers": fit_seconds,
                  "mean_wall_seconds": fit_seconds / fits.max(1) as f64 },
        "cost": cost,
        "memory": {
            "free_physical_bytes_at_start": budget.free_bytes(),
            "limit_bytes": budget.limit_bytes(),
            "held_by_the_logs_bytes": budget.held_bytes(),
            "share_of_each_worker_bytes": budget.left_bytes() / options.workers as u64,
            "heap_peak_bytes": peak,
            "heap_peak_share_of_limit": peak as f64 / budget.limit_bytes() as f64,
            "within_limit": peak <= budget.limit_bytes(),
            "read_plan": plan.to_string(),
        },
        "folds": fold_times,
    });
    let mut text =
        serde_json::to_string_pretty(&run_record).map_err(|error| L2dError::Runtime {
            detail: format!("writing the run record failed: {error}"),
        })?;
    text.push('\n');
    std::fs::write(&run_path, text).map_err(|error| L2dError::Runtime {
        detail: format!("writing {} failed: {error}", run_path.display()),
    })?;

    // The answer, on standard output.
    println!("report {} sha256 {json_sha256}", json_path.display());
    println!("summary {} sha256 {text_sha256}", text_path.display());
    println!("run record {}", run_path.display());
    println!(
        "{fits} fits on {} workers in {evaluation_seconds:.0} s; heap peak {:.0} MiB of a limit \
         of {:.0} MiB",
        options.workers,
        peak as f64 / (1024.0 * 1024.0),
        budget.limit_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "cross-endpoint ROC AUC, mean over folds (lowest to highest fold); gate {GATE_ROC_AUC}"
    );
    for (plan, result) in arrangements.iter().zip(&results) {
        for (settings, mode) in modes.iter().zip(&result.modes) {
            let line = |rule: &str, summary: l2d_eval::evaluate::RuleSummary| {
                if let Some(spread) = summary.roc_auc {
                    println!(
                        "{:<26} {:<8} {:<24} {:.4} ({:.4} to {:.4}) {}",
                        plan.name,
                        routing_name(settings.baseline.routing),
                        rule,
                        spread.mean,
                        spread.min,
                        spread.max,
                        if spread.mean >= GATE_ROC_AUC {
                            "pass"
                        } else {
                            "FAIL"
                        }
                    );
                }
            };
            line(
                "default penalty",
                rule_summary(&data, &mode.folds, settings, |fold| fold.default_penalty()),
            );
            if settings.inner_folds >= 2 {
                line(
                    "chosen by inner log loss",
                    rule_summary(&data, &mode.folds, settings, |fold| fold.chosen()),
                );
                line(
                    "chosen by inner ROC AUC",
                    rule_summary(&data, &mode.folds, settings, |fold| {
                        fold.chosen_by_roc_auc()
                    }),
                );
            }
        }
    }
    if peak > budget.limit_bytes() {
        return Err(L2dError::Runtime {
            detail: format!(
                "the heap peaked at {peak} bytes, above the memory limit of {} bytes; the \
                 reports were written",
                budget.limit_bytes()
            ),
        });
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(error.exit_code())
        }
    }
}
