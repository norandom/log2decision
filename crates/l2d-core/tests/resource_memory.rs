use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::{MemoryBudget, MemorySettings};

#[test]
fn variable_records_end_before_the_next_maximum_record_crosses_the_budget() {
    use l2d_core::memory::{BatchCost, Batcher};
    let usable = 512 * 1024;
    let mut budget = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(PROCESS_BASELINE_BYTES + usable),
            max_batch_rows: Some(10_000),
            ..MemorySettings::default()
        },
        || Ok(1024 * 1024 * 1024),
    )
    .unwrap();
    let mut batch = Batcher::new(
        &mut budget,
        10_000_000,
        BatchCost {
            expansion: 1.0,
            row_bytes: 0,
            batch_bytes: 0,
        },
    );
    let mut bytes = 0;
    for _ in 0..256 {
        bytes += 128;
        assert!(!batch.record_with_headroom(128, 65536).unwrap());
    }
    loop {
        bytes += 1000;
        if batch.record_with_headroom(1000, 65536).unwrap() {
            break;
        }
    }
    assert!(
        bytes <= usable - 65536 + 1000,
        "batch retained {bytes} before leaving room for its next bounded record"
    );
    assert!(batch.budget().usage().report().counted_peak_bytes <= PROCESS_BASELINE_BYTES + usable);
}

#[test]
fn final_accounting_survives_budget_drop_and_does_not_count_a_refused_allocation() {
    let limit = PROCESS_BASELINE_BYTES + 4096;
    let mut budget = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(limit),
            ..MemorySettings::default()
        },
        || Ok(1024 * 1024 * 1024),
    )
    .unwrap();
    let usage = budget.usage();
    budget.hold(1000);
    budget.require_whole(2000, "admitted").unwrap();
    assert!(budget.require_whole(5000, "refused").is_err());
    budget.note_spill();
    budget.hold(0);
    drop(budget);
    let report = usage.report();
    assert_eq!(report.memory_limit_bytes, limit);
    assert_eq!(report.counted_peak_bytes, PROCESS_BASELINE_BYTES + 3000);
    assert!(report.spilled);
}

#[test]
fn a_tracked_operation_writes_the_final_memory_and_admission_record() {
    use l2d_core::resource::{BusyChoice, ResourceGuard, ResourceReport};
    let mut budget =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(1024 * 1024 * 1024)).unwrap();
    let guard =
        ResourceGuard::tracked("accounting-test", BusyChoice::Wait, budget.usage()).unwrap();
    let path = guard.record_path().unwrap().to_owned();
    budget.hold(500);
    budget.require_whole(700, "allocation").unwrap();
    budget.note_spill();
    drop(budget);
    guard.finish().unwrap();
    let report: ResourceReport = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(report.operation, "accounting-test");
    assert!(report.gate_held);
    assert_eq!(
        report.memory.counted_peak_bytes,
        PROCESS_BASELINE_BYTES + 1200
    );
    assert!(report.memory.spilled);
    assert!(!report.ended_while_panicking);
    std::fs::remove_file(path).unwrap();
}
