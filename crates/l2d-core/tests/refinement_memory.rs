use std::alloc::{GlobalAlloc, Layout, System};
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

use l2d_core::patterns::{PatternCatalogue, PatternSettings, refinement::PatternRefinement};
use l2d_core::selections::{SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{InputAdapter, LinesAdapter, MemoryBudget, MemorySettings};
#[test]
fn refinement_peak_and_preallocation_refusal() {
    let root = std::env::temp_dir().join(format!("l2d-refine-memory-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let input = root.join("input.log");
    std::fs::write(
        &input,
        (0..1000)
            .map(|i| format!("login failed user=u{i}\n"))
            .collect::<String>(),
    )
    .unwrap();
    let mut b = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(128 * 1024 * 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&input, &mut b).unwrap();
    let s = SelectionSet::search_input(
        "all",
        SearchQuery::glob("*"),
        SearchSettings { max_members: 1000 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let c = PatternCatalogue::build_input(
        &s,
        &logs,
        &summary,
        &PatternSettings {
            max_patterns: 8,
            ..Default::default()
        },
        &root.join("catalogue"),
        &mut b,
    )
    .unwrap();
    drop(logs);
    b.hold(s.held_bytes() + c.held_bytes());
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let r = PatternRefinement::from_catalogue(&c, &root.join("catalogue"), &mut b).unwrap();
    let split = r.split(0, "user=u[02468]$", &b).unwrap();
    let p = split.preview_query(0, "login", &b).unwrap();
    split.save_query(&p, &root.join("query.json"), &b).unwrap();
    split.save(&root.join("refinement.json"), &b).unwrap();
    let loaded = PatternRefinement::load(&root.join("refinement.json"), &b).unwrap();
    assert_eq!(loaded, split);
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    println!(
        "refinement native heap growth={peak}; budget={:?}",
        b.usage().report()
    );
    assert!(peak < 32 * 1024 * 1024);
    let tiny = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(33 * 1024 * 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    assert!(PatternRefinement::load(&root.join("refinement.json"), &tiny).is_err());
    let growth = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    println!("refused load native heap growth={growth}");
    assert!(growth < 128 * 1024);
    assert!(r.split(0, "user", &tiny).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
