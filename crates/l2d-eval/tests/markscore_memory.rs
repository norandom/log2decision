//! Heap bounds for the held-out-group report (16.9, 16.12, 12.9, 15.4).
use l2d_baseline::markscore::{MarkFitSettings, fit_from_marks};
use l2d_baseline::redact::RedactionInput;
use l2d_baseline::{BaselineSettings, MarkFit};
use l2d_core::marks::{MarkQuery, MarkSet};
use l2d_core::{InputSummary, L2dError, Label, LogRecord, LogsFrame, MemoryBudget, MemorySettings};
use l2d_eval::markscore::{HeldOutSettings, held_out_report};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
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

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const BASELINE: u64 = l2d_core::memory::PROCESS_BASELINE_BYTES;

/// Runs `run` and returns its result and by how many bytes the heap grew at its peak.
fn measure<T>(run: impl FnOnce() -> T) -> (T, u64) {
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = run();
    let peak = PEAK.load(Ordering::Relaxed);
    (result, peak.saturating_sub(before) as u64)
}
fn budget(room: u64, held: u64) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: Some(BASELINE + held + room),
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).unwrap();
    budget.hold(held);
    budget
}

fn fitted(text_bytes: usize, group_bytes: usize) -> MarkFit {
    let records: Vec<_> = (0..160)
        .map(|i| LogRecord {
            text: format!(
                "{i} {} {}",
                if i % 2 == 0 { "good" } else { "bad" },
                "a".repeat(text_bytes)
            ),
            label: Label::Unknown,
            group: Some(format!("group{}{}", i % 8, "g".repeat(group_bytes))),
            first_position: i + 1,
            occurrences: 1,
        })
        .collect();
    let logs = LogsFrame::from_records(&records).unwrap();
    let summary = InputSummary {
        source_path: "heap.log".into(),
        records: 160,
        distinct_texts: 160,
        content_sha256: "a".repeat(64),
        skipped: vec![],
        label_conflicts: 0,
    };
    let mut roomy = budget(GIB, 0);
    let marks = MarkSet::mark(
        "bad",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: " bad ".into(),
        },
        &logs,
        &summary,
        &roomy,
    )
    .unwrap();
    fit_from_marks(
        &marks,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &MarkFitSettings {
            baseline: BaselineSettings {
                buckets: 64,
                ngram_max: 1,
                ..BaselineSettings::default()
            },
            group_key: None,
        },
        &mut roomy,
    )
    .unwrap()
}

#[test]
fn report_refuses_before_copying_logs_when_the_held_fit_leaves_no_room() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let fit = fitted(8192, 4096);
    let held = fit.held_bytes() + fit.scorer.held_bytes();
    let mut budget = budget(64 * 1024, held);
    let room = budget.left_bytes();
    let (result, growth) =
        measure(|| held_out_report(&fit, &HeldOutSettings::default(), &mut budget));
    assert!(
        matches!(result, Err(L2dError::Refused { .. })),
        "{result:?}"
    );
    assert_eq!(budget.held_bytes(), held);
    eprintln!("report tight: peak={growth}, room={room}");
    assert!(
        growth <= room,
        "report copied {growth} bytes with only {room} left"
    );
}

#[test]
fn report_covers_groups_folds_metrics_and_restores_the_budget() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (text, group) in [(64, 8192), (8192, 32)] {
        let fit = fitted(text, group);
        let held = fit.held_bytes() + fit.scorer.held_bytes();
        let settings = HeldOutSettings {
            folds: 5,
            ..HeldOutSettings::default()
        };
        let expected = held_out_report(&fit, &settings, &mut budget(GIB, held)).unwrap();
        let mut completed = false;
        let mut refused = false;
        for room in [64 * 1024, MIB, 4 * MIB, 16 * MIB, 64 * MIB] {
            let mut budget = budget(room, held);
            let (result, growth) = measure(|| held_out_report(&fit, &settings, &mut budget));
            assert_eq!(budget.held_bytes(), held);
            eprintln!(
                "report text={text} group={group} room={room} peak={growth} completed={}",
                result.is_ok()
            );
            assert!(growth <= room, "report peak {growth} exceeds room {room}");
            match result {
                Ok(report) => {
                    assert_eq!(report, expected);
                    completed = true;
                }
                Err(L2dError::Refused { .. }) => refused = true,
                Err(error) => panic!("{error}"),
            }
        }
        assert!(completed && refused);
    }
}
