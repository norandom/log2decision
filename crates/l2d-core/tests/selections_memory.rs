use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_core::selections::{
    ReviewAction, ReviewDecision, ReviewState, SearchQuery, SearchSettings, SelectionSet,
};
use l2d_core::{
    InputAdapter, L2dError, Label, LinesAdapter, MemoryBudget, MemorySettings, ZipInputSettings,
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

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// The share of every limit the tool keeps for the process itself.
const BASELINE: u64 = l2d_core::memory::PROCESS_BASELINE_BYTES;

/// Runs `run` and returns its result and by how many bytes the heap grew at its peak.
fn measure<T>(run: impl FnOnce() -> T) -> (T, u64) {
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = run();
    let peak = PEAK.load(Ordering::Relaxed);
    (result, peak.saturating_sub(before) as u64)
}

fn budget(usable: u64) -> MemoryBudget {
    MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(BASELINE + usable),
            max_batch_rows: Some(64),
            ..Default::default()
        },
        || Ok(8 * GIB),
    )
    .unwrap()
}
#[test]
fn selection_search_preview_export_persistence_and_refusals_are_preallocated() {
    let _serial = ONE_AT_A_TIME.lock().unwrap();
    let root = std::env::temp_dir().join(format!("l2d-selection-memory-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("lines.log");
    {
        let mut file = std::fs::File::create(&path).unwrap();
        for i in 0..20_000 {
            writeln!(file, "record {i} matched").unwrap();
        }
    }
    let (logs, summary, _) = LinesAdapter
        .read_with(&path, &mut budget(256 * MIB))
        .unwrap();
    let mut b = budget(64 * MIB);
    let (result, peak) = measure(|| {
        SelectionSet::search_input(
            "memory",
            SearchQuery::literal("matched"),
            SearchSettings {
                max_members: 20_000,
            },
            &logs,
            &summary,
            &b,
        )
    });
    assert!(peak <= 64 * MIB, "search peak {peak}");
    let mut set = result.unwrap();
    let decisions: Vec<_> = set
        .members()
        .iter()
        .enumerate()
        .map(|(i, member)| ReviewDecision {
            member: member.clone(),
            state: if i % 2 == 0 {
                ReviewState::Bad
            } else {
                ReviewState::Unresolved
            },
        })
        .collect();
    let (preview, peak) = measure(|| set.preview_review(&decisions, &b));
    assert!(peak + set.held_bytes() <= 64 * MIB, "preview peak {peak}");
    let (reviewed, peak) = measure(|| {
        set.review_input(
            preview.unwrap(),
            ReviewAction {
                analyst: "test".into(),
                action: "explicit batch review".into(),
            },
            &summary,
            None,
            &b,
        )
    });
    reviewed.unwrap();
    assert!(peak + set.held_bytes() <= 64 * MIB);
    let before_export = CURRENT.load(Ordering::Relaxed);
    let (marks, peak) = measure(|| set.export_mark_set("reviewed", Label::Bad, &b));
    let marks = marks.unwrap();
    let retained_export = CURRENT
        .load(Ordering::Relaxed)
        .saturating_sub(before_export) as u64;
    let downstream_reservation = 8 * (marks.positions().len() + marks.deselected().len()) as u64
        + l2d_core::marks::MARKED_ID_BYTES
            * (marks.id_count() + marks.deselected_id_count()) as u64;
    assert!(
        retained_export <= downstream_reservation + 4096,
        "retained export {retained_export}, members reserved {downstream_reservation}"
    );
    assert_eq!(marks.selected_count(), 10_000);
    assert!(peak + set.held_bytes() <= 64 * MIB, "export peak {peak}");
    let saved = root.join("selection.json");
    let (result, peak) = measure(|| set.save(&saved, &b));
    result.unwrap();
    assert!(peak + set.held_bytes() <= 64 * MIB, "save peak {peak}");
    for usable in [MIB / 2, MIB, 8 * MIB, 32 * MIB, 256 * MIB] {
        let (loaded, peak) = measure(|| SelectionSet::load(&saved, &budget(usable)));
        assert!(peak <= usable, "load at {usable}: {peak}");
        if let Err(e) = loaded {
            assert!(matches!(e, L2dError::Refused { .. }));
        }
    }
    b.hold(b.usable_bytes() - 64 * 1024);
    let (refused, peak) = measure(|| set.preview_review(&decisions, &b));
    assert!(refused.is_err());
    assert!(peak < 64 * 1024, "preview allocated before refusal: {peak}");
    let (refused, peak) = measure(|| set.export_mark_set("refused", Label::Bad, &b));
    assert!(refused.is_err());
    assert!(peak < 64 * 1024, "export allocated before refusal: {peak}");
    let huge_query = SearchQuery::literal("x".repeat(8 * MIB as usize));
    let (refused, peak) = measure(|| {
        SelectionSet::search_input(
            "huge",
            huge_query,
            SearchSettings::default(),
            &logs,
            &summary,
            &b,
        )
    });
    assert!(refused.is_err());
    assert!(peak < 64 * 1024, "query copied before refusal: {peak}");
    let mut huge_summary = summary.clone();
    huge_summary.source_path = root.join("x".repeat(MIB as usize));
    let (refused, peak) = measure(|| {
        SelectionSet::search_input(
            "huge-metadata",
            SearchQuery::literal("x"),
            SearchSettings::default(),
            &logs,
            &huge_summary,
            &b,
        )
    });
    assert!(refused.is_err());
    assert!(peak < 64 * 1024, "metadata copied before refusal: {peak}");
    let (refused, peak) = measure(|| {
        SelectionSet::search_input(
            "huge-cap",
            SearchQuery::literal("x"),
            SearchSettings {
                max_members: u64::MAX,
            },
            &logs,
            &summary,
            &b,
        )
    });
    assert!(refused.is_err());
    assert!(peak < 64 * 1024);
    // A large selection with only one member of the requested resolved label must
    // not retain capacity for all the opposite-label members in either ID vector.
    let tiny_decisions: Vec<_> = set
        .members()
        .iter()
        .enumerate()
        .map(|(i, member)| ReviewDecision {
            member: member.clone(),
            state: if i == 0 {
                ReviewState::Bad
            } else {
                ReviewState::Good
            },
        })
        .collect();
    let ample = budget(64 * MIB);
    let preview = set.preview_review(&tiny_decisions, &ample).unwrap();
    set.review_input(
        preview,
        ReviewAction {
            analyst: "test".into(),
            action: "resolved remaining members".into(),
        },
        &summary,
        None,
        &ample,
    )
    .unwrap();
    let before = CURRENT.load(Ordering::Relaxed);
    let tiny = set.export_mark_set("tiny", Label::Bad, &ample).unwrap();
    let retained = CURRENT.load(Ordering::Relaxed).saturating_sub(before);
    assert_eq!(tiny.selected_count(), 1);
    assert!(retained <= 4104, "tiny export retained {retained} bytes");
    std::fs::remove_dir_all(&root).unwrap();
}
#[test]
fn archive_fixed_reservation_covers_batches_and_refusals() {
    let _serial = ONE_AT_A_TIME.lock().unwrap();
    let root = std::env::temp_dir().join(format!(
        "l2d-selection-archive-memory-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("input.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    zip.start_file("a.log", zip::write::SimpleFileOptions::default())
        .unwrap();
    for i in 0..10_000 {
        writeln!(zip, "text {i} {}", "x".repeat(500)).unwrap();
    }
    zip.finish().unwrap();
    for usable in [MIB, 8 * MIB, 16 * MIB, 32 * MIB, 64 * MIB] {
        let mut b = budget(usable);
        let (result, peak) = measure(|| {
            SelectionSet::search_archive(
                "archive",
                SearchQuery::glob("text*"),
                SearchSettings {
                    max_members: 10_000,
                },
                &path,
                &ZipInputSettings::default(),
                &mut b,
            )
        });
        assert!(peak <= usable, "archive at {usable}: peak {peak}");
        assert_eq!(b.held_bytes(), 0);
        match result {
            Ok((set, _)) => assert_eq!(set.members().len(), 10_000),
            Err(e) => assert!(matches!(e, L2dError::Refused { .. }), "{e}"),
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}
