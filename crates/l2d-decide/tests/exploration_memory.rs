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

use l2d_core::patterns::{
    PatternCatalogue, PatternCounts, PatternMember, PatternSettings, refinement::PatternRefinement,
};
use l2d_core::selections::{Completeness, Member, ReviewState};
use l2d_core::{MarkedInput, MarkedSource, MemoryBudget, MemorySettings};
use l2d_decide::exploration::*;

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024)).unwrap()
}

#[test]
fn exploration_heap_and_refusal_include_repeated_provenance_and_report_buffers() {
    let mut seed = fixture("a", &[("login failed user=seed", 0, 1)]);
    for i in 1..500 {
        seed.rows.push(PatternMember {
            member: Member::Position(i),
            pattern: Some(0),
            text: format!("login failed user=u{i}"),
            occurrences: 1000,
            review: None,
        });
    }
    if let MarkedSource::Input(s) = &mut seed.catalogue.source {
        s.archive_entries = Some(vec!["x".repeat(1024); 4]);
    }
    let mut b = budget();
    b.hold(seed.held_bytes());
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let settings = ExploreSettings {
        exemplar_row: 0,
        scope: Scope::AllPatterns,
        top_k: 200,
        radius: 1.,
        expected_profile: None,
    };
    let report = explore(&seed, None, &settings, &mut CharacterTrigram, &mut b).unwrap();
    let json = serde_json::to_value(&report).unwrap();
    let bytes = serde_json::to_vec(&json).unwrap();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    println!(
        "exploration native heap growth={peak}, admitted={}, serialized={}",
        b.held_bytes(),
        bytes.len()
    );
    assert!((peak as u64) < b.held_bytes());
    assert_eq!(report.support.distinct_texts, 499);
    let mut small = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(34 * 1024 * 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    assert!(explore(&seed, None, &settings, &mut CharacterTrigram, &mut small).is_err());
    let refused = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    println!("refusal heap growth={refused}");
    assert!(refused < 256 * 1024);
}
fn fixture(name: &str, rows: &[(&str, usize, u64)]) -> PatternRefinement {
    let catalogue = PatternCatalogue {
        format: 1,
        algorithm_revision: "test".into(),
        selection_revision: format!("selection-{name}"),
        source: MarkedSource::Input(MarkedInput {
            path: format!("{name}.log"),
            content_sha256: name.repeat(64),
            archive_entries: None,
            interpretation: None,
        }),
        source_data_revision: None,
        completeness: Completeness {
            complete: true,
            records: Some(rows.len() as u64),
            distinct_scanned: rows.len() as u64,
            matched: rows.len() as u64,
            skipped_records: 0,
            omitted_entries: 0,
        },
        settings: PatternSettings::default(),
        occurrence_basis: "original".into(),
        patterns: vec![],
        ungrouped: PatternCounts::default(),
        selected_members: rows.len() as u64,
        memberships_sha256: "0".repeat(64),
        revision: name.into(),
    };
    PatternRefinement {
        format: 1,
        catalogue,
        catalogue_revision: name.into(),
        selection_revision: format!("selection-{name}"),
        parent_revision: None,
        action: "fixture".into(),
        groups: vec![],
        priorities: Default::default(),
        ungrouped: PatternCounts::default(),
        rows: rows
            .iter()
            .enumerate()
            .map(|(i, (t, p, n))| PatternMember {
                member: Member::Position(i as u64),
                pattern: Some(*p),
                text: (*t).into(),
                occurrences: *n,
                review: Some(ReviewState::Bad),
            })
            .collect(),
        revision: name.into(),
    }
}
