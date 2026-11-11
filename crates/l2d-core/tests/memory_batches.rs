//! Batches sized to the free memory (requirements 12.1 to 12.9) and batched reading of
//! the input adapters.
//!
//! Free memory is injected wherever a test needs a fixed or changing figure; the real
//! reader is exercised on its own and by the reads that use the default budget.

use std::path::{Path, PathBuf};

use l2d_core::adapters::csic2010::Csic2010Adapter;
use l2d_core::input::{PARSE_BATCH_BYTES, PARSE_EXPANSION, content_sha256};
use l2d_core::memory::BatchCause;
use l2d_core::memory::{
    DEFAULT_MAX_BATCH_ROWS, MIN_BATCH_ROWS, RowEstimate, estimate_rows, free_physical_memory,
    parse_memory_size,
};
use l2d_core::{
    BatchChange, BatchPlan, EXIT_REFUSED, InputAdapter, InputSummary, L2dError, LabelledAdapter,
    LabelledFormat, LinesAdapter, LogsFrame, MemoryBudget, MemorySettings, ProjectPaths, RunKind,
    RunRecord, Source,
};

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;

/// The share of every limit the tool keeps for the process itself.
const BASELINE: u64 = l2d_core::memory::PROCESS_BASELINE_BYTES;

/// Free memory whose half, the default limit, leaves exactly 4 GiB beside the baseline.
const ROOMY: u64 = 8 * GIB + 2 * BASELINE;

/// A budget whose free memory is always `free`.
fn fixed(settings: &MemorySettings, free: u64) -> MemoryBudget {
    MemoryBudget::observe_with(settings, move || Ok(free)).unwrap()
}

/// A budget whose free memory follows `readings`: the first is the reading at the start,
/// then one per batch; the last one repeats.
fn scripted(settings: &MemorySettings, readings: &[u64]) -> MemoryBudget {
    let readings: Vec<u64> = readings.into();
    let mut readings = readings.into_iter();
    let mut last = 0;
    MemoryBudget::observe_with(settings, move || {
        if let Some(next) = readings.next() {
            last = next;
        }
        Ok(last)
    })
    .unwrap()
}

fn settings(memory_limit_bytes: Option<u64>, max_batch_rows: Option<u64>) -> MemorySettings {
    MemorySettings {
        memory_limit_bytes,
        max_batch_rows,
        ..MemorySettings::default()
    }
}

fn refusal(result: Result<impl std::fmt::Debug, L2dError>) -> String {
    match result {
        Err(error @ L2dError::Refused { .. }) => {
            assert_eq!(error.exit_code(), EXIT_REFUSED);
            error.to_string()
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// A file in a scratch directory of its own in the system temp folder; the directory is
/// removed when the value is dropped.
struct ScratchFile {
    path: PathBuf,
}

impl std::ops::Deref for ScratchFile {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn scratch_file(name: &str, content: &[u8]) -> ScratchFile {
    let dir = std::env::temp_dir().join(format!(
        "l2d-core-memory-test-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    ScratchFile { path }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv")
}

// --- Defaults and overrides (12.1, 12.2) ---------------------------------------------------

#[test]
fn default_settings_are_half_of_free_memory_with_a_quarter_in_reserve() {
    let defaults = MemorySettings::default();
    assert_eq!(defaults.memory_limit_bytes, None);
    assert_eq!(defaults.max_batch_rows, None);
    assert_eq!(defaults.free_share, 0.5);
    assert_eq!(defaults.reserve_share, 0.25);
    assert_eq!(DEFAULT_MAX_BATCH_ROWS, 65_536);
    assert_eq!(MIN_BATCH_ROWS, 1_024);

    let budget = fixed(&defaults, 8 * GIB);
    assert_eq!(budget.free_bytes(), 8 * GIB);
    assert_eq!(budget.limit_bytes(), 4 * GIB);
    assert_eq!(budget.reserve_bytes(), 2 * GIB);
}

#[test]
fn the_first_batch_is_the_maximum_or_what_the_limit_holds() {
    let budget = fixed(&MemorySettings::default(), 8 * GIB);

    // Small rows: the maximum batch fits.
    let plan = budget.plan(1_000_000, 2_000).unwrap();
    assert_eq!(
        plan,
        BatchPlan {
            free_bytes: 8 * GIB,
            limit_bytes: 4 * GIB,
            reserve_bytes: 2 * GIB,
            bytes_per_row: 2_000,
            batch_rows: 65_536,
            overridden: false,
            changes: vec![],
            min_rows_used: 0,
            max_rows_used: 0,
        }
    );

    // Large rows: what the limit leaves / bytes per row = 4 GiB / 1 MiB = 4,096 rows.
    let budget = fixed(&MemorySettings::default(), ROOMY);
    assert_eq!(budget.usable_bytes(), 4 * GIB);
    assert_eq!(budget.plan(1_000_000, MIB).unwrap().batch_rows, 4_096);
    // Not a power of two: 4 GiB / 3 MiB = 1,365 rows.
    assert_eq!(budget.plan(1_000_000, 3 * MIB).unwrap().batch_rows, 1_365);
    // Exactly the minimum still fits.
    assert_eq!(budget.plan(1_000_000, 4 * MIB).unwrap().batch_rows, 1_024);
    // A source without rows has no bytes per row; the plan is the maximum.
    assert_eq!(budget.plan(0, 0).unwrap().batch_rows, 65_536);
}

#[test]
fn the_analysts_limit_and_batch_size_replace_the_tools_choice() {
    // Memory limit: used as given, whatever the free memory.
    let budget = fixed(&settings(Some(BASELINE + 100 * MIB), None), 8 * GIB);
    assert_eq!(budget.limit_bytes(), BASELINE + 100 * MIB);
    assert_eq!(budget.usable_bytes(), 100 * MIB);
    assert_eq!(budget.reserve_bytes(), 2 * GIB);
    let plan = budget.plan(500_000, 10_000).unwrap();
    assert!(plan.overridden);
    assert_eq!(plan.limit_bytes, BASELINE + 100 * MIB);
    assert_eq!(plan.batch_rows, 100 * MIB / 10_000);

    // Batch size: the maximum is the analyst's.
    let budget = fixed(&settings(None, Some(10_000)), 8 * GIB);
    let plan = budget.plan(500_000, 2_000).unwrap();
    assert!(plan.overridden);
    assert_eq!(plan.limit_bytes, 4 * GIB);
    assert_eq!(plan.batch_rows, 10_000);

    // A batch size below the usual minimum is its own minimum.
    let budget = fixed(&settings(None, Some(100)), 8 * GIB);
    assert_eq!(budget.plan(500_000, 2_000).unwrap().batch_rows, 100);

    // Both.
    let budget = fixed(&settings(Some(GIB), Some(200_000)), 8 * GIB);
    let plan = budget.plan(500_000, 2_000).unwrap();
    assert!(plan.overridden);
    assert_eq!(plan.batch_rows, 200_000);
}

#[test]
fn unusable_settings_are_invalid_input() {
    let cases = [
        settings(Some(0), None),
        settings(None, Some(0)),
        MemorySettings {
            free_share: 0.0,
            ..MemorySettings::default()
        },
        MemorySettings {
            free_share: 1.5,
            ..MemorySettings::default()
        },
        MemorySettings {
            free_share: f32::NAN,
            ..MemorySettings::default()
        },
        MemorySettings {
            reserve_share: -0.1,
            ..MemorySettings::default()
        },
        MemorySettings {
            reserve_share: 1.0,
            ..MemorySettings::default()
        },
    ];
    for case in cases {
        let result = MemoryBudget::observe_with(&case, || Ok(8 * GIB));
        assert!(
            matches!(result, Err(L2dError::InvalidInput { .. })),
            "{case:?}"
        );
    }
    // A failing reader fails the observation with its own error.
    let failed = MemoryBudget::observe_with(&MemorySettings::default(), || {
        Err(L2dError::Runtime {
            detail: "no reading".into(),
        })
    });
    assert!(matches!(failed, Err(L2dError::Runtime { .. })));
}

// --- Refusals (12.4, 12.5) -----------------------------------------------------------------

#[test]
fn a_minimum_batch_that_does_not_fit_is_refused_with_the_figures() {
    let budget = fixed(&settings(Some(BASELINE + MIB), None), 8 * GIB);
    // 1,024 rows of 2,000 bytes are 2,048,000 bytes; the limit leaves 1,048,576.
    let message = refusal(budget.plan(50_000, 2_000));
    for part in ["1024 rows", "2048000 bytes", "1048576 bytes", "2000 bytes"] {
        assert!(message.contains(part), "`{part}` missing in: {message}");
    }
    // One byte per row less than the limit holds: accepted.
    assert_eq!(budget.plan(50_000, 1_024).unwrap().batch_rows, 1_024);

    // A source of fewer rows than the minimum batch is judged by its own rows: three
    // records of 12 MiB fit 4 GiB although 1,024 of them would not.
    let roomy = fixed(&MemorySettings::default(), ROOMY);
    assert_eq!(roomy.plan(3, 12 * MIB).unwrap().batch_rows, 341);
    let message = refusal(roomy.plan(3, 2 * GIB));
    assert!(message.contains("3 rows"), "{message}");
    assert!(message.contains("6442450944 bytes"), "{message}");

    // The same with the tool's own limit on a host with little free memory.
    let free = 2 * (BASELINE + MIB);
    let budget = fixed(&MemorySettings::default(), free);
    let message = refusal(budget.plan(50_000, 2_000));
    assert!(message.contains("1048576 bytes"), "{message}");
    assert!(
        message.contains(&format!("{free} bytes")),
        "free memory: {message}"
    );
}

#[test]
fn a_whole_dataset_over_the_limit_is_refused_before_loading() {
    let mut budget = fixed(&MemorySettings::default(), ROOMY);
    budget.require_whole(4 * GIB, "baseline fit").unwrap();
    budget.require_whole(0, "baseline fit").unwrap();
    let message = refusal(budget.require_whole(4 * GIB + 1, "baseline fit"));
    for part in ["baseline fit", "4294967297 bytes", "4294967296 bytes"] {
        assert!(message.contains(part), "`{part}` missing in: {message}");
    }

    // What an operation already holds counts.
    budget.hold(GIB);
    budget.require_whole(3 * GIB, "baseline fit").unwrap();
    let message = refusal(budget.require_whole(3 * GIB + 1, "baseline fit"));
    assert!(message.contains("3221225473 bytes"), "{message}");
    assert!(message.contains("3221225472 bytes"), "{message}");

    let budget = fixed(&settings(Some(BASELINE + 1_000), None), 8 * GIB);
    let message = refusal(budget.require_whole(1_001, "shift test"));
    for part in ["shift test", "1001 bytes", "1000 bytes"] {
        assert!(message.contains(part), "`{part}` missing in: {message}");
    }
}

// --- Dynamic scaling (12.7, 12.8, 12.9) ----------------------------------------------------

#[test]
fn the_batch_halves_below_the_reserve_and_doubles_when_memory_has_recovered() {
    // Reserve 2 GiB; recovered means above 4 GiB.
    let mut budget = scripted(
        &MemorySettings::default(),
        &[
            8 * GIB,     // at the start
            8 * GIB,     // batch 1: stays at the maximum
            GIB,         // batch 2: below the reserve, halved
            GIB,         // batch 3: halved again
            3 * GIB,     // batch 4: between reserve and recovery, unchanged
            4 * GIB,     // batch 5: exactly twice the reserve is not above it
            5 * GIB,     // batch 6: recovered, doubled
            8 * GIB,     // batch 7: doubled, back at the maximum
            8 * GIB,     // batch 8: stays at the maximum
            2 * GIB - 1, // batch 9: one byte below the reserve
            2 * GIB,     // batch 10: exactly the reserve is not below it
        ],
    );
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    let mut rows_done = 0;
    let mut sizes = Vec::new();
    for _ in 0..10 {
        let rows = budget.next_batch_rows(&mut plan, rows_done).unwrap();
        sizes.push(rows);
        rows_done += rows;
    }
    assert_eq!(
        sizes,
        [
            65_536, 32_768, 16_384, 16_384, 16_384, 32_768, 65_536, 65_536, 32_768, 32_768
        ]
    );
    // What a batch could take: the free memory, or the 4 GiB limit less the baseline.
    let change = |at_row, from_rows, to_rows, free_bytes: u64, cause| BatchChange {
        at_row,
        from_rows,
        to_rows,
        free_bytes,
        available_bytes: free_bytes.min(4 * GIB - BASELINE),
        cause,
    };
    let (fell, rose) = (BatchCause::BelowReserve, BatchCause::Recovered);
    assert_eq!(
        plan.changes,
        [
            change(65_536, 65_536, 32_768, GIB, fell),
            change(98_304, 32_768, 16_384, GIB, fell),
            change(147_456, 16_384, 32_768, 5 * GIB, rose),
            change(180_224, 32_768, 65_536, 8 * GIB, rose),
            change(311_296, 65_536, 32_768, 2 * GIB - 1, fell),
        ]
    );
    assert_eq!(plan.batch_rows, 65_536, "the first batch is not rewritten");
    assert_eq!(plan.min_rows_used, 16_384);
    assert_eq!(plan.max_rows_used, 65_536);
}

#[test]
fn the_batch_never_leaves_the_range_between_minimum_and_first_batch() {
    // Always below the reserve: down to the minimum and no further, as long as it fits.
    let mut budget = scripted(&MemorySettings::default(), &[8 * GIB, GIB]);
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    let sizes: Vec<u64> = (0..9)
        .map(|batch| budget.next_batch_rows(&mut plan, batch * 100).unwrap())
        .collect();
    assert_eq!(
        sizes,
        [
            32_768, 16_384, 8_192, 4_096, 2_048, 1_024, 1_024, 1_024, 1_024
        ]
    );
    assert_eq!(plan.changes.len(), 6);
    assert_eq!((plan.min_rows_used, plan.max_rows_used), (1_024, 32_768));

    // Plenty of memory: never above the first batch, here capped by the limit.
    let mut budget = fixed(&MemorySettings::default(), ROOMY);
    let mut plan = budget.plan(10_000_000, 3 * MIB).unwrap();
    for batch in 0..4 {
        assert_eq!(
            budget.next_batch_rows(&mut plan, batch * 1_365).unwrap(),
            1_365
        );
    }
    assert!(plan.changes.is_empty());
    assert_eq!((plan.min_rows_used, plan.max_rows_used), (1_365, 1_365));

    // An analyst's batch size below the usual minimum is neither halved nor doubled.
    let mut budget = scripted(&settings(None, Some(100)), &[8 * GIB, GIB, 8 * GIB]);
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    for batch in 0..3 {
        assert_eq!(budget.next_batch_rows(&mut plan, batch * 100).unwrap(), 100);
    }
    assert!(plan.changes.is_empty());
}

#[test]
fn a_batch_larger_than_the_free_memory_shrinks_until_it_fits() {
    // 65,536 rows of 2,000 bytes are 125 MiB. With 40 MiB free only 16,384 rows
    // (31.25 MiB) fit: the batch is cut to that in one step, not by one halving.
    let mut budget = scripted(&MemorySettings::default(), &[8 * GIB, 8 * GIB, 40 * MIB]);
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    assert_eq!(budget.next_batch_rows(&mut plan, 0).unwrap(), 65_536);
    assert_eq!(budget.next_batch_rows(&mut plan, 65_536).unwrap(), 16_384);
    assert_eq!(
        plan.changes,
        [BatchChange {
            at_row: 65_536,
            from_rows: 65_536,
            to_rows: 16_384,
            free_bytes: 40 * MIB,
            available_bytes: 40 * MIB,
            cause: BatchCause::FreeTooSmall,
        }]
    );
}

#[test]
fn memory_held_between_batches_counts_against_the_limit() {
    // Limit 4 GiB. With 3.9 GiB held, 0.1 GiB is left for a batch: 32,768 rows of
    // 2,000 bytes (62.5 MiB) fit, 65,536 do not.
    let mut budget = fixed(&MemorySettings::default(), ROOMY);
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    assert_eq!(budget.next_batch_rows(&mut plan, 0).unwrap(), 65_536);
    budget.hold(4 * GIB - 100 * MIB);
    assert_eq!(budget.left_bytes(), 100 * MIB);
    assert_eq!(budget.next_batch_rows(&mut plan, 65_536).unwrap(), 32_768);

    // With all of the limit held, nothing fits: refused, naming the held memory.
    budget.hold(4 * GIB - MIB);
    let message = refusal(budget.next_batch_rows(&mut plan, 98_304));
    assert!(message.contains("98304 rows"), "{message}");
    assert!(message.contains("1048576 bytes"), "available: {message}");
    assert!(
        message.contains(&format!("{} bytes", 4 * GIB - MIB)),
        "held: {message}"
    );
}

#[test]
fn a_minimum_batch_that_stops_fitting_mid_run_is_refused_with_the_rows_completed() {
    let mut budget = scripted(&MemorySettings::default(), &[8 * GIB, 8 * GIB, 1_000_000]);
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    assert_eq!(budget.next_batch_rows(&mut plan, 0).unwrap(), 65_536);
    // 1,024 rows of 2,000 bytes are 2,048,000 bytes; 1,000,000 are free.
    let message = refusal(budget.next_batch_rows(&mut plan, 65_536));
    for part in ["1024 rows", "2048000 bytes", "1000000 bytes", "65536 rows"] {
        assert!(message.contains(part), "`{part}` missing in: {message}");
    }
    assert!(plan.changes.is_empty(), "a refused batch is not a change");
}

// --- Estimate, sizes, report (12.1, 12.2, 12.3) --------------------------------------------

#[test]
fn bytes_per_row_come_from_the_sample_and_the_expansion_factor() {
    // 256 sampled rows in 51,200 bytes: 200 bytes per row in the source.
    assert_eq!(
        estimate_rows(2_000_000, 51_200, 256, 4.0),
        RowEstimate {
            total_rows: 10_000,
            bytes_per_row: 800,
        }
    );
    // Rounded up, never down: 10 bytes in 3 rows are 4 bytes per row.
    assert_eq!(
        estimate_rows(100, 10, 3, 1.5),
        RowEstimate {
            total_rows: 25,
            bytes_per_row: 6,
        }
    );
    // The whole source was sampled: the row count is exact.
    assert_eq!(estimate_rows(10, 10, 3, 1.0).total_rows, 3);
    // Nothing sampled.
    assert_eq!(
        estimate_rows(0, 0, 0, 4.0),
        RowEstimate {
            total_rows: 0,
            bytes_per_row: 0,
        }
    );
    // Parsing never needs less than the source.
    const { assert!(PARSE_EXPANSION >= 1.0) };
}

#[test]
fn memory_sizes_are_plain_bytes_or_carry_a_binary_suffix() {
    let accepted = [
        ("1", 1),
        ("1048576", MIB),
        ("512MB", 512 * MIB),
        ("2GB", 2 * GIB),
        ("2gb", 2 * GIB),
        ("2G", 2 * GIB),
        ("2GiB", 2 * GIB),
        ("64k", 64 * KIB),
        ("64KB", 64 * KIB),
        ("100b", 100),
        ("1TB", 1024 * GIB),
        (" 3 MB ", 3 * MIB),
    ];
    for (text, bytes) in accepted {
        assert_eq!(parse_memory_size(text).unwrap(), bytes, "{text:?}");
    }
    for text in [
        "",
        " ",
        "0",
        "0MB",
        "MB",
        "lots",
        "12XB",
        "-1",
        "+1",
        "1.5GB",
        "1e9",
        "0x10",
        "99999999999999999999",
        "99999999999GB",
        "2 G B",
    ] {
        let result = parse_memory_size(text);
        match result {
            Err(L2dError::InvalidInput { detail }) => {
                assert!(detail.contains(text.trim()), "{text:?}: {detail}");
            }
            other => panic!("{text:?} must be invalid input, got {other:?}"),
        }
    }
}

#[test]
fn the_plan_has_a_one_line_form_and_goes_into_the_run_record_settings() {
    let mut budget = scripted(&settings(None, Some(4_096)), &[8 * GIB, 8 * GIB, GIB]);
    let mut plan = budget.plan(1_000_000, 2_000).unwrap();
    budget.next_batch_rows(&mut plan, 0).unwrap();
    budget.next_batch_rows(&mut plan, 4_096).unwrap();

    let line = plan.to_string();
    assert!(!line.contains('\n'), "{line}");
    for part in [
        "8.0 GiB",    // free memory observed
        "4.0 GiB",    // limit
        "2.0 GiB",    // reserve
        "2000 bytes", // per row
        "4096 rows",  // first batch
        "set by the analyst",
        "1 change",
        "at row 4096",
        "4096 to 2048",
        "1.0 GiB", // the free memory that caused the change
        "smallest 2048",
        "largest 4096",
    ] {
        assert!(line.contains(part), "`{part}` missing in: {line}");
    }
    let unchanged = fixed(&MemorySettings::default(), 8 * GIB)
        .plan(10, 100)
        .unwrap()
        .to_string();
    assert!(
        unchanged.contains("chosen from the free memory"),
        "{unchanged}"
    );
    assert!(unchanged.contains("no change"), "{unchanged}");

    let mut record = RunRecord {
        kind: RunKind::Evaluate,
        started_utc: "2026-11-11T10:00:00Z".into(),
        duration_ms: 1,
        inputs: vec![],
        settings: serde_json::json!({ "folds": 5 }),
        seed: None,
        encoder: None,
        device: None,
        driver_version: None,
        logs_per_second: None,
        placement: None,
    };
    record.record_batch_plan(&plan).unwrap();
    assert_eq!(record.settings["folds"], 5, "other settings are kept");
    let stored = &record.settings["batch_plan"];
    assert_eq!(stored["free_bytes"], 8 * GIB);
    assert_eq!(stored["limit_bytes"], 4 * GIB);
    assert_eq!(stored["reserve_bytes"], 2 * GIB);
    assert_eq!(stored["bytes_per_row"], 2_000);
    assert_eq!(stored["batch_rows"], 4_096);
    assert_eq!(stored["overridden"], true);
    assert_eq!(stored["changes"][0]["at_row"], 4_096);
    assert_eq!(stored["changes"][0]["from_rows"], 4_096);
    assert_eq!(stored["changes"][0]["to_rows"], 2_048);
    assert_eq!(stored["changes"][0]["free_bytes"], GIB);
    assert_eq!(stored["min_rows_used"], 2_048);
    assert_eq!(stored["max_rows_used"], 4_096);

    // Through the JSON form of the record and back.
    let again = RunRecord::from_json(&record.to_json().unwrap()).unwrap();
    assert_eq!(again, record);
    let restored: BatchPlan = serde_json::from_value(again.settings["batch_plan"].clone()).unwrap();
    assert_eq!(restored, plan);

    // Empty settings become an object; settings of another shape are not overwritten.
    record.settings = serde_json::Value::Null;
    record.record_batch_plan(&plan).unwrap();
    assert_eq!(record.settings["batch_plan"]["batch_rows"], 4_096);
    record.settings = serde_json::json!([1, 2]);
    assert!(matches!(
        record.record_batch_plan(&plan),
        Err(L2dError::Runtime { .. })
    ));
}

// --- The real reader (12.1, 12.3) ----------------------------------------------------------

#[test]
fn the_host_reports_a_plausible_amount_of_free_memory() {
    let free = free_physical_memory().unwrap();
    assert!(free > 64 * MIB, "free physical memory: {free} bytes");
    assert!(free < 16 * 1024 * GIB, "free physical memory: {free} bytes");

    let budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let plan = budget.plan(61_065, 2_000).unwrap();
    assert!(plan.free_bytes > 64 * MIB);
    // Two readings a moment apart differ by less than a factor of two.
    assert!(plan.free_bytes / 2 < free && free / 2 < plan.free_bytes);
    let half = plan.free_bytes / 2;
    assert!(
        plan.limit_bytes.abs_diff(half) <= 1,
        "limit {} against half of {}",
        plan.limit_bytes,
        plan.free_bytes
    );
    assert!(plan.reserve_bytes.abs_diff(plan.free_bytes / 4) <= 1);
    assert!(!plan.overridden);
    eprintln!("host plan: {plan}");
}

// --- Batched reading: the same result for any batch size (12.6) ----------------------------

type Read = (LogsFrame, InputSummary);

fn split(read: (LogsFrame, InputSummary, BatchPlan)) -> (Read, BatchPlan) {
    ((read.0, read.1), read.2)
}

#[test]
fn the_fixture_reads_the_same_in_batches_of_any_size() {
    let path = fixture_path();
    let adapter = LabelledAdapter::new(LabelledFormat::Csv).with_group_column("group");
    let (expected, default_plan) = split(
        adapter
            .read_with(
                &path,
                &mut MemoryBudget::observe(&MemorySettings::default()).unwrap(),
            )
            .unwrap(),
    );
    assert_eq!(expected.1.records, 200);
    assert_eq!(expected.1.distinct_texts, 169);
    assert_eq!(expected.1.label_conflicts, 1);
    assert_eq!(expected.1.skipped.len(), 1);
    assert_eq!(expected.1.skipped[0].position, 121);
    assert_eq!(
        expected.1.content_sha256,
        content_sha256(&std::fs::read(&path).unwrap())
    );
    // `read` is the same read with the default budget.
    assert_eq!(adapter.read(&path).unwrap(), expected);

    // The default plan: free memory read from the host, the maximum batch.
    assert!(default_plan.free_bytes > 64 * MIB);
    assert_eq!(default_plan.batch_rows, DEFAULT_MAX_BATCH_ROWS);
    assert!(!default_plan.overridden);
    assert!(default_plan.bytes_per_row > 0);
    assert!(default_plan.changes.is_empty());
    assert_eq!(default_plan.min_rows_used, DEFAULT_MAX_BATCH_ROWS);
    assert_eq!(default_plan.max_rows_used, DEFAULT_MAX_BATCH_ROWS);

    for rows in [
        1, 2, 7, 64, 120, 121, 199, 200, 201, 1_024, 10_000, 1_000_000,
    ] {
        let mut budget = fixed(&settings(None, Some(rows)), 64 * GIB);
        let (read, plan) = split(adapter.read_with(&path, &mut budget).unwrap());
        assert_eq!(read, expected, "batches of {rows} rows");
        assert!(plan.overridden);
        assert_eq!(plan.batch_rows, rows);
        assert_eq!((plan.min_rows_used, plan.max_rows_used), (rows, rows));
        // The sample is never larger than a batch; from 200 rows on it is the whole
        // fixture and gives the same bytes per row as the default plan.
        assert!(plan.bytes_per_row > 0);
        if rows >= 200 {
            assert_eq!(plan.bytes_per_row, default_plan.bytes_per_row, "{rows}");
        }
    }
}

#[test]
fn the_real_csic_dataset_reads_the_same_in_batches_of_any_size() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let path = match Csic2010Adapter::locate(&ProjectPaths::new(root), None) {
        Ok(path) => path,
        Err(error @ L2dError::Missing { .. }) => {
            eprintln!(
                "SKIPPED the_real_csic_dataset_reads_the_same_in_batches_of_any_size: {error}"
            );
            return;
        }
        Err(other) => panic!("locating the dataset failed: {other:?}"),
    };
    let adapter = Csic2010Adapter::new();

    // One batch for the whole file.
    let mut whole = fixed(&settings(None, Some(1_000_000)), 1024 * GIB);
    let started = std::time::Instant::now();
    let (logs, summary, counts, plan) = adapter.read_counted_with(&path, &mut whole).unwrap();
    let whole_time = started.elapsed();
    assert_eq!(summary.records, 61_065);
    assert_eq!((counts.good, counts.bad), (36_000, 25_065));
    assert_eq!(summary.distinct_texts, 25_608);
    assert!(summary.skipped.is_empty());
    assert_eq!(
        summary.content_sha256,
        "c420f0bc0464376de75b6c419a0ac226fe69fe12c8ac4908843273721e44e637"
    );
    assert_eq!(plan.batch_rows, 1_000_000);
    assert!(
        plan.batch_rows > summary.records,
        "one batch holds the file"
    );
    eprintln!("CSIC 2010 in one batch: {whole_time:.2?}; {plan}");

    // 1,024 rows with the memory of this host, 10,000 rows, and the default plan.
    let mut budgets = [
        (
            "1,024 rows",
            MemoryBudget::observe(&settings(None, Some(1_024))).unwrap(),
        ),
        (
            "10,000 rows",
            fixed(&settings(None, Some(10_000)), 64 * GIB),
        ),
        (
            "default plan",
            MemoryBudget::observe(&MemorySettings::default()).unwrap(),
        ),
    ];
    for (name, budget) in &mut budgets {
        let started = std::time::Instant::now();
        let (logs_again, summary_again, counts_again, plan) =
            adapter.read_counted_with(&path, budget).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(logs_again, logs, "{name}");
        assert_eq!(summary_again, summary, "{name}");
        assert_eq!(counts_again, counts, "{name}");
        eprintln!("CSIC 2010 in batches ({name}): {elapsed:.2?}; {plan}");
    }
}

/// 20,000 lines of which 7,000 are distinct, some blank and some not UTF-8.
fn many_lines() -> Vec<u8> {
    let mut content = Vec::new();
    for index in 0..20_000u32 {
        if index % 4_001 == 4_000 {
            content.extend_from_slice(b"   \n");
        } else if index % 5_003 == 5_002 {
            content.extend_from_slice(b"GET /\xff\xfe broken\n");
        } else {
            let line = format!("GET /item/{}?page={} HTTP/1.1\n", index % 7_000, index % 13);
            content.extend_from_slice(line.as_bytes());
        }
    }
    content
}

#[test]
fn a_drop_and_recovery_of_free_memory_shrinks_and_regrows_the_batch_and_changes_nothing_else() {
    let path = scratch_file("drop-and-recovery.log", &many_lines());
    let expected = LinesAdapter.read(&path).unwrap();
    assert_eq!(expected.1.records, 20_000);
    assert_eq!(expected.1.skipped.len(), 4 + 3);

    let mut budget = scripted(
        &settings(None, Some(4_096)),
        &[
            8 * GIB, // at the start: reserve 2 GiB
            8 * GIB, // batch 1 at row 0: 4,096 rows
            GIB,     // batch 2 at row 4,096: halved to 2,048
            GIB,     // batch 3 at row 6,144: halved to 1,024
            GIB,     // batch 4 at row 7,168: the minimum stays
            8 * GIB, // batch 5 at row 8,192: doubled to 2,048
            8 * GIB, // batch 6 at row 10,240: doubled to 4,096
            8 * GIB, // batches 7 and 8: unchanged
        ],
    );
    let (read, plan) = split(LinesAdapter.read_with(&path, &mut budget).unwrap());
    assert_eq!(
        read, expected,
        "the result does not depend on the batch sizes"
    );

    let changes: Vec<(u64, u64, u64, u64, BatchCause)> = plan
        .changes
        .iter()
        .map(|change| {
            (
                change.at_row,
                change.from_rows,
                change.to_rows,
                change.free_bytes,
                change.cause,
            )
        })
        .collect();
    let (fell, rose) = (BatchCause::BelowReserve, BatchCause::Recovered);
    assert_eq!(
        changes,
        [
            (4_096, 4_096, 2_048, GIB, fell),
            (6_144, 2_048, 1_024, GIB, fell),
            (8_192, 1_024, 2_048, 8 * GIB, rose),
            (10_240, 2_048, 4_096, 8 * GIB, rose),
        ]
    );
    assert_eq!(plan.batch_rows, 4_096);
    assert_eq!((plan.min_rows_used, plan.max_rows_used), (1_024, 4_096));
    assert_eq!(plan.free_bytes, 8 * GIB);
    assert_eq!(plan.reserve_bytes, 2 * GIB);
    assert!(plan.overridden);
}

#[test]
fn free_memory_is_read_again_before_every_batch() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let path = scratch_file("readings.log", &many_lines());
    let readings = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&readings);
    let mut budget = MemoryBudget::observe_with(&settings(None, Some(3_000)), move || {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(8 * GIB)
    })
    .unwrap();
    assert_eq!(readings.load(Ordering::SeqCst), 1, "once before loading");
    LinesAdapter.read_with(&path, &mut budget).unwrap();
    // 20,000 rows in batches of 3,000 are seven batches.
    assert_eq!(readings.load(Ordering::SeqCst), 1 + 7);
}

#[test]
fn a_read_that_runs_out_of_memory_mid_run_is_refused_with_the_rows_completed() {
    let path = scratch_file("mid-run.log", &many_lines());
    let mut budget = scripted(
        &settings(None, Some(4_096)),
        &[8 * GIB, 8 * GIB, 8 * GIB, 10],
    );
    let message = refusal(LinesAdapter.read_with(&path, &mut budget));
    assert!(message.contains("8192 rows"), "{message}");
    assert!(message.contains("10 bytes"), "{message}");
    assert!(message.contains("1024 rows"), "{message}");
}

#[test]
fn an_artificially_low_memory_limit_refuses_the_read_with_the_figures() {
    let path = fixture_path();
    let adapter = LabelledAdapter::new(LabelledFormat::Csv).with_group_column("group");
    let bytes_per_row = adapter
        .read_with(&path, &mut fixed(&MemorySettings::default(), 8 * GIB))
        .unwrap()
        .2
        .bytes_per_row;

    // A limit that leaves 1,000 bytes beside the baseline and the fixed cost of a batch.
    let limit = BASELINE + PARSE_BATCH_BYTES + 1_000;
    let mut budget = fixed(&settings(Some(limit), None), 8 * GIB);
    let message = refusal(adapter.read_with(&path, &mut budget));
    // The fixture has 200 rows, fewer than the minimum batch: all of them must fit.
    for part in [
        "200 rows".to_string(),
        format!("{} bytes", 200 * bytes_per_row),
        "1000 bytes".to_string(),
    ] {
        assert!(message.contains(&part), "`{part}` missing in: {message}");
    }

    // The lines and CSIC adapters go through the same check.
    let lines = scratch_file("low-limit.log", &many_lines());
    let message = refusal(LinesAdapter.read_with(&lines, &mut budget));
    assert!(message.contains("1024 rows"), "{message}");
    assert!(message.contains("1000 bytes"), "{message}");
}

#[test]
fn a_single_record_larger_than_the_limit_is_refused_instead_of_loaded() {
    let mut content = Vec::new();
    for index in 0..300 {
        content.extend_from_slice(format!("short line {index}\n").as_bytes());
    }
    content.extend(std::iter::repeat_n(b'x', 3 * MIB as usize));
    content.extend_from_slice(b"\nlast line\n");
    let path = scratch_file("long-line.log", &content);

    // Without a limit of the analyst the line is read.
    let (logs, summary) = LinesAdapter.read(&path).unwrap();
    assert_eq!(summary.records, 302);
    assert_eq!(logs.len(), 302);

    // A limit that leaves 2 MiB for a batch: a third of a megabyte for one record.
    let limit = BASELINE + PARSE_BATCH_BYTES + 2 * MIB;
    let mut budget = fixed(&settings(Some(limit), None), 8 * GIB);
    let message = refusal(LinesAdapter.read_with(&path, &mut budget));
    assert!(message.contains("position 301"), "{message}");
    assert!(message.contains(&format!("{limit} bytes")), "{message}");

    // The same for a CSV field that never closes its quote and takes the rest of the file.
    let mut content = b"text,label\n".to_vec();
    for index in 0..300 {
        content.extend_from_slice(format!("row {index},good\n").as_bytes());
    }
    content.extend_from_slice(b"\"never closed,good\n");
    content.extend(std::iter::repeat_n(b'y', 3 * MIB as usize));
    let path = scratch_file("open-quote.csv", &content);
    let adapter = LabelledAdapter::new(LabelledFormat::Csv);
    let message = refusal(adapter.read_with(&path, &mut budget));
    assert!(message.contains("position 301"), "{message}");
}

// --- Record boundaries inside a read or a batch --------------------------------------------

/// Reads `content` with every combination of a small read size and a small batch size
/// and expects the result of the default read every time.
fn assert_same_for_any_read_and_batch_size(
    adapter: &impl InputAdapter,
    name: &str,
    content: &[u8],
) -> Read {
    let path = scratch_file(name, content);
    let expected = adapter.read(&path).unwrap();
    assert_eq!(expected.1.content_sha256, content_sha256(content));
    for read_size in [1, 2, 3, 4, 5, 7, 16, 4_096] {
        for rows in [1, 2, 3, 5, 1_024] {
            let mut budget = fixed(&settings(None, Some(rows)), 8 * GIB);
            let source = Source::open(&path).unwrap().with_read_size(read_size);
            let (read, _) = split(adapter.read_from(source, &mut budget).unwrap());
            assert_eq!(
                read, expected,
                "{name}: reads of {read_size} bytes, batches of {rows} rows"
            );
        }
    }
    expected
}

#[test]
fn csv_records_split_by_a_read_or_a_batch_are_read_as_one() {
    let content = concat!(
        "\u{feff}text,label,group\r\n",
        "\"GET /a?q=1,2\",good,g1\r\n",
        "\"say \"\"hi\"\", twice\",bad,\"g,\"\"2\"\"\"\r\n",
        "\"line one\nline two\",good,g3\n",
        "\"line one\r\nline two\r\n\r\nline four\",bad,g4\r\n",
        "GET /b \"quoted\" tail,good,g5\n",
        "\"\",good,g7\n",
        "\r\n",
        "\n",
        "GET /d,good\n",
        "\"GET /a?q=1,2\",\"good\",other\r\n",
        "ünïcödé ☃ 😀,bad,gü\n",
        "\"\"\"\",good,\"\t\"\r\n",
        "\"line one\nline two\",bad,again\r\n",
        "GET /e,good,g15",
    );
    let adapter = LabelledAdapter::new(LabelledFormat::Csv).with_group_column("group");
    let (logs, summary) =
        assert_same_for_any_read_and_batch_size(&adapter, "boundaries.csv", content.as_bytes());
    assert_eq!(summary.records, 12);
    let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(skipped, [6, 7]);
    assert_eq!(summary.label_conflicts, 1);
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    assert_eq!(
        texts,
        [
            "GET /a?q=1,2",
            "say \"hi\", twice",
            "line one\nline two",
            "line one\r\nline two\r\n\r\nline four",
            "GET /b \"quoted\" tail",
            "ünïcödé ☃ 😀",
            "\"",
            "GET /e",
        ]
    );
}

#[test]
fn lines_split_by_a_read_or_a_batch_are_read_as_one() {
    let content = "\u{feff}first line\r\nsecond line\n\r\nünïcödé ☃ 😀\r\nfirst line\n\n   \r\nlast line without end";
    let (logs, summary) = assert_same_for_any_read_and_batch_size(
        &LinesAdapter,
        "boundaries.log",
        content.as_bytes(),
    );
    assert_eq!(summary.records, 8);
    let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(skipped, [3, 6, 7]);
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    assert_eq!(
        texts,
        [
            "first line",
            "second line",
            "ünïcödé ☃ 😀",
            "last line without end"
        ]
    );

    // Only a byte-order mark, only line ends, nothing at all.
    for (name, content, records) in [
        ("only-bom.log", &b"\xef\xbb\xbf"[..], 0),
        ("bom-and-end.log", &b"\xef\xbb\xbf\r\n"[..], 1),
        ("only-ends.log", &b"\n\r\n\n"[..], 3),
        ("half-a-bom.log", &b"\xef\xbb"[..], 1),
        ("empty.log", &b""[..], 0),
    ] {
        let (logs, summary) = assert_same_for_any_read_and_batch_size(&LinesAdapter, name, content);
        assert_eq!(summary.records, records, "{name}");
        assert!(logs.is_empty(), "{name}");
    }
}

#[test]
fn json_lines_split_by_a_read_or_a_batch_are_read_as_one() {
    let content = concat!(
        "\u{feff}{\"text\":\"GET /a\",\"label\":\"good\",\"group\":\"g1\"}\r\n",
        "{\"text\":\"line one\\nline two\",\"label\":\"bad\"}\n",
        "not json\n",
        "\r\n",
        "{\"text\":\"GET /a\",\"label\":\"bad\",\"group\":\"g2\"}\r\n",
        "{\"text\":\"big\",\"label\":\"good\",\"n\":123456789012345678901234567890}\n",
        "{\"text\":\"ünïcödé ☃ 😀\",\"label\":null,\"extra\":[1,2]}\n",
        "{\"text\":\"GET /z\",\"label\":\"GOOD\"}",
    );
    let adapter = LabelledAdapter::new(LabelledFormat::Jsonl).with_group_column("group");
    let (logs, summary) =
        assert_same_for_any_read_and_batch_size(&adapter, "boundaries.jsonl", content.as_bytes());
    assert_eq!(summary.records, 8);
    let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(skipped, [3, 4, 6]);
    assert_eq!(summary.label_conflicts, 1);
    assert_eq!(logs.len(), 4);

    // A configured column that no record carries is still found after the last batch.
    let path = scratch_file(
        "no-group.jsonl",
        b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n{\"text\":\"c\"}\n",
    );
    let mut budget = fixed(&settings(None, Some(1)), 8 * GIB);
    let result = adapter.read_with(&path, &mut budget);
    assert!(
        matches!(&result, Err(L2dError::InvalidInput { detail }) if detail.contains("'label'")),
        "{result:?}"
    );
}

#[test]
fn a_missing_source_is_reported_as_missing_by_the_batched_read() {
    let path = std::env::temp_dir().join("l2d-core-memory-test-no-such-file.csv");
    let mut budget = fixed(&MemorySettings::default(), 8 * GIB);
    let result = LinesAdapter.read_with(&path, &mut budget);
    match result {
        Err(L2dError::Missing { item, expected_at }) => {
            assert_eq!(item, "input file");
            assert_eq!(expected_at, path);
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

// --- Remediation round 1: the limit is kept for real (12.8, 12.9) --------------------------

#[test]
fn a_limit_below_what_the_tool_itself_needs_is_rejected() {
    // The analyst's limit: an unusable option value.
    for limit in [1, 1_000, MIB, BASELINE] {
        let result = MemoryBudget::observe_with(&settings(Some(limit), None), || Ok(8 * GIB));
        match result {
            Err(L2dError::InvalidInput { detail }) => {
                assert!(detail.contains(&format!("{limit} bytes")), "{detail}");
                assert!(detail.contains(&format!("{BASELINE} bytes")), "{detail}");
            }
            other => panic!("limit {limit}: expected invalid input, got {other:?}"),
        }
    }
    // The tool's own limit on a host with too little free memory: a refusal.
    let result = MemoryBudget::observe_with(&MemorySettings::default(), || Ok(2 * BASELINE));
    let message = refusal(result);
    assert!(
        message.contains(&format!("{} bytes", 2 * BASELINE)),
        "{message}"
    );
    assert!(message.contains(&format!("{BASELINE} bytes")), "{message}");
    // One byte more is a budget.
    let budget = fixed(&settings(Some(BASELINE + 1), None), 8 * GIB);
    assert_eq!(budget.limit_bytes(), BASELINE + 1);
}

#[test]
fn skipped_records_count_as_held_memory() {
    let mut dedup = l2d_core::Dedup::new();
    assert_eq!(dedup.held_bytes(), 0);
    for position in 1..=10_000 {
        dedup.skip(
            position,
            "line is not valid JSON: expected value at line 1 column 1",
        );
    }
    let held = dedup.held_bytes();
    // At least the record and its reason text, each time.
    assert!(
        held >= 10_000 * (32 + 57),
        "{held} bytes held for 10,000 skipped records"
    );

    // A source of nothing but skipped records is refused under a limit it cannot fit.
    let mut content = Vec::new();
    for _ in 0..400_000 {
        content.extend_from_slice(b" \n");
    }
    let path = scratch_file("all-skipped.log", &content);
    let mut budget = fixed(&settings(Some(BASELINE + 8 * MIB), None), 8 * GIB);
    let message = refusal(LinesAdapter.read_with(&path, &mut budget));
    assert!(message.contains("rows were completed"), "{message}");
    assert!(message.contains("held"), "{message}");
}

#[test]
fn short_rows_are_planned_with_their_fixed_cost_per_row() {
    let mut content = Vec::new();
    for index in 0..2_000 {
        content.extend_from_slice(format!("{}\n", index % 10).as_bytes());
    }
    let path = scratch_file("one-byte-lines.log", &content);
    let mut budget = fixed(&MemorySettings::default(), 8 * GIB);
    let plan = LinesAdapter.read_with(&path, &mut budget).unwrap().2;
    // Two bytes of source per row; the memory per row is dominated by what every row
    // costs whatever its length.
    assert!(
        plan.bytes_per_row >= 100,
        "{} bytes per row",
        plan.bytes_per_row
    );
}

#[test]
fn a_change_states_what_caused_it() {
    let cause = |plan: &BatchPlan, index: usize| {
        serde_json::to_value(plan.changes[index]).unwrap()["cause"].clone()
    };

    // Free memory below the reserve, then recovered.
    let mut budget = scripted(
        &MemorySettings::default(),
        &[8 * GIB, 8 * GIB, GIB, 8 * GIB],
    );
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    for batch in 0..3 {
        budget.next_batch_rows(&mut plan, batch * 1_000).unwrap();
    }
    assert_eq!(cause(&plan, 0), "below_reserve");
    assert_eq!(cause(&plan, 1), "recovered");

    // The batch does not fit the free memory although that is above the reserve
    // (reserve 0): free memory too small.
    let no_reserve = MemorySettings {
        reserve_share: 0.0,
        ..MemorySettings::default()
    };
    let mut budget = scripted(&no_reserve, &[8 * GIB, 8 * GIB, 100 * MIB]);
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    budget.next_batch_rows(&mut plan, 0).unwrap();
    assert_eq!(budget.next_batch_rows(&mut plan, 65_536).unwrap(), 32_768);
    assert_eq!(cause(&plan, 0), "free_too_small");

    // Plenty of free memory, but the rows held leave too little of the limit.
    let mut budget = fixed(&MemorySettings::default(), 8 * GIB);
    let mut plan = budget.plan(10_000_000, 2_000).unwrap();
    budget.next_batch_rows(&mut plan, 0).unwrap();
    budget.hold(4 * GIB - BASELINE - 100 * MIB);
    assert_eq!(budget.next_batch_rows(&mut plan, 65_536).unwrap(), 32_768);
    assert_eq!(cause(&plan, 0), "limit_less_held");
    let stored = serde_json::to_value(plan.changes[0]).unwrap();
    assert_eq!(stored["available_bytes"], 100 * MIB);
    assert_eq!(stored["free_bytes"], 8 * GIB);
    let line = plan.to_string();
    assert!(!line.contains('\n'), "{line}");
    assert!(
        line.contains("100.0 MiB left of the memory limit"),
        "the line names the held memory as the cause: {line}"
    );
}
