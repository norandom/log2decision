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

use l2d_core::patterns::{PatternCatalogue, PatternSettings};
use l2d_core::selections::{SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{InputAdapter, LinesAdapter, MemoryBudget, MemorySettings};
#[test]
fn catalogue_spooling_loading_and_refusal_are_bounded() {
    let root = std::env::temp_dir().join(format!("l2d-pattern-memory-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("input.log");
    let text = (0..2000)
        .map(|i| format!("login failed user=user{i} id={i}\n"))
        .collect::<String>();
    std::fs::write(&path, text).unwrap();
    let mut b = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(64 * 1024 * 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut b).unwrap();
    let s = SelectionSet::search_input(
        "all",
        SearchQuery::literal("login"),
        SearchSettings { max_members: 2000 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let settings = PatternSettings {
        max_patterns: 8,
        max_text_bytes: 1024,
        ..Default::default()
    };
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let c = PatternCatalogue::build_input(
        &s,
        &logs,
        &summary,
        &settings,
        &root.join("catalogue"),
        &mut b,
    )
    .unwrap();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    println!(
        "catalogue 2000 members: incremental Rust heap peak={peak}, usable={}, accounted={:?}",
        b.usable_bytes(),
        b.usage().report()
    );
    assert!(peak < 8 * 1024 * 1024);
    assert_eq!(c.patterns[0].counts.distinct_texts, 2000);
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let loaded = PatternCatalogue::load(&root.join("catalogue"), &mut b).unwrap();
    let mut seen = 0;
    loaded
        .visit_members(&root.join("catalogue"), &mut b, |_| {
            seen += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, 2000);
    assert!(PEAK.load(Ordering::Relaxed).saturating_sub(before) < 8 * 1024 * 1024);
    let mut tiny = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    assert!(
        PatternCatalogue::build_input(
            &s,
            &logs,
            &summary,
            &settings,
            &root.join("refused"),
            &mut tiny
        )
        .is_err()
    );
    assert!(!root.join("refused").exists());
    std::fs::create_dir(root.join("oversized")).unwrap();
    std::fs::File::create(root.join("oversized/manifest.json"))
        .unwrap()
        .set_len(4 * 1024 * 1024)
        .unwrap();
    assert!(PatternCatalogue::load(&root.join("oversized"), &mut tiny).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
