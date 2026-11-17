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

use l2d_core::patterns::{PatternCatalogue, PatternEngine, PatternSettings};
use l2d_core::selections::{SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{InputAdapter, LinesAdapter, MemoryBudget, MemorySettings};
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    text: String,
    group: String,
}

#[test]
#[ignore = "release engine qualification; writes measured evidence to stdout"]
fn compare_frozen_pattern_engines() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/pattern_pairs.json")).unwrap();
    let root =
        std::env::temp_dir().join(format!("l2d-pattern-qualification-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("fixture.log");
    std::fs::write(
        &path,
        fixture
            .cases
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let mut b = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(512 * 1024 * 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut b).unwrap();
    let selection = SelectionSet::search_input(
        "qualification",
        SearchQuery::regex(".*"),
        SearchSettings { max_members: 100 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    println!(
        "engine,repeat,tp,fp,fn,tn,precision,recall,coverage,representatives,collision_flags,microseconds"
    );
    for (engine, threshold) in [
        (PatternEngine::DrainStyle, 800),
        (PatternEngine::CharacterTrigram, 720),
    ] {
        let settings = PatternSettings {
            engine,
            similarity_per_mille: threshold,
            max_patterns: 64,
            ..Default::default()
        };
        let mut revision = None;
        for repeat in 0..7 {
            let heap_before = CURRENT.load(Ordering::Relaxed);
            PEAK.store(heap_before, Ordering::Relaxed);
            let destination = root.join(format!("{engine:?}-{repeat}"));
            let start = std::time::Instant::now();
            let catalogue = PatternCatalogue::build_input(
                &selection,
                &logs,
                &summary,
                &settings,
                &destination,
                &mut b,
            )
            .unwrap();
            let elapsed = start.elapsed().as_micros();
            let heap_peak = PEAK.load(Ordering::Relaxed).saturating_sub(heap_before);
            println!("heap_peak_bytes={heap_peak}");
            if let Some(ref previous) = revision {
                assert_eq!(previous, &catalogue.revision);
            }
            revision = Some(catalogue.revision.clone());
            let mut groups = Vec::new();
            catalogue
                .visit_members(&destination, &mut b, |m| {
                    groups.push(m.pattern);
                    Ok(())
                })
                .unwrap();
            let (mut tp, mut fp, mut fn_, mut tn) = (0, 0, 0, 0);
            for i in 0..groups.len() {
                for j in i + 1..groups.len() {
                    match (
                        fixture.cases[i].group == fixture.cases[j].group,
                        groups[i].is_some() && groups[i] == groups[j],
                    ) {
                        (true, true) => tp += 1,
                        (false, true) => fp += 1,
                        (true, false) => fn_ += 1,
                        (false, false) => tn += 1,
                    }
                }
            }
            assert_eq!(tp + fn_, 8, "frozen positive pair count");
            assert_eq!(fp + tn, 182, "frozen negative pair count");
            if engine == PatternEngine::DrainStyle {
                assert!(tp > 0, "undefined precision cannot qualify");
                assert!(
                    tp as f64 / (tp + fp) as f64 >= 0.95,
                    "selected engine must satisfy the frozen precision gate"
                );
            }
            let precision = if tp + fp == 0 {
                "undefined".into()
            } else {
                format!("{:.6}", tp as f64 / (tp + fp) as f64)
            };
            let recall = tp as f64 / (tp + fn_) as f64;
            println!(
                "{engine:?},{repeat},{tp},{fp},{fn_},{tn},{precision},{recall:.6},{},{},{},{elapsed}",
                groups.iter().filter(|g| g.is_some()).count(),
                catalogue.patterns.len(),
                catalogue
                    .patterns
                    .iter()
                    .filter(|p| p.possible_outcome_collision)
                    .count()
            );
            println!("resource_usage={:?}", b.usage().report());
            if repeat == 0 {
                for p in &catalogue.patterns {
                    println!(
                        "template={:?}; members={}; boundary={:?}; variations={:?}",
                        p.template, p.counts.members, p.boundary_example, p.differing_literals
                    );
                }
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
