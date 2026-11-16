//! Requirement 16.12 measured for a store (with 12.9 and 15.2): marking a store by a
//! phrase (its text index) or a regular expression (its logs read in batches) stays
//! within the memory limit (task 1.21, owner decision of 2026-11-16).
//!
//! Two heaps are read, as in `store_csic.rs`: the Rust side through a counting global
//! allocator, and SQLite's through its own counter (`sqlite3_memory_highwater`), since
//! SQLite allocates through the C runtime. For each shape a ladder of memory limits is
//! tried; whether the marking completes or is refused, the growth of the Rust heap at its
//! peak and SQLite's peak beyond what it held before, added up, may not exceed the limit
//! less the share the tool keeps for the process itself. The content hash, the search or
//! the batches, the ids and the mark set returned are inside the measurement; the store
//! is created, filled and opened before it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_core::marks::{MarkQuery, MarkSet};
use l2d_core::{
    EncoderIdentity, L2dError, Label, LogRecord, LogsFrame, MemoryBudget, MemorySettings,
    ProjectPaths,
};
use l2d_store::{LogFlags, Store, VectorOrigin, rows_frame};

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
const DIM: u16 = 4;
const BASELINE: u64 = l2d_core::memory::PROCESS_BASELINE_BYTES;

fn identity() -> EncoderIdentity {
    EncoderIdentity {
        name: "bytecnn".to_string(),
        revision: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        dimension: DIM,
    }
}

struct Scratch {
    root: PathBuf,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// SQLite's bytes in use now and its peak since the last reset, resetting the peak.
fn sqlite_now_and_peak() -> (u64, u64) {
    // SAFETY: both functions only read (and reset) SQLite's allocation counters; they
    // take no pointers and may be called at any time.
    unsafe {
        (
            rusqlite::ffi::sqlite3_memory_used() as u64,
            rusqlite::ffi::sqlite3_memory_highwater(1) as u64,
        )
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    Completed,
    Refused,
}

fn mark_within(
    shape: &str,
    store: &Store,
    query: &MarkQuery,
    usable: u64,
    batch_rows: Option<u64>,
    marked: u64,
) -> Outcome {
    let limit = BASELINE + usable;
    let settings = MemorySettings {
        memory_limit_bytes: Some(limit),
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap();
    let (sqlite_before, _) = sqlite_now_and_peak();
    let rust_before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(rust_before, Ordering::Relaxed);
    let started = std::time::Instant::now();
    let result = MarkSet::mark_store("marks", Label::Bad, query.clone(), store, &mut budget);
    let elapsed = started.elapsed();
    let rust = (PEAK.load(Ordering::Relaxed).saturating_sub(rust_before)) as u64;
    let (_, sqlite_peak) = sqlite_now_and_peak();
    let sqlite = sqlite_peak.saturating_sub(sqlite_before);
    let (outcome, detail) = match &result {
        Ok(set) => {
            assert_eq!(set.count(), marked, "{shape}");
            (Outcome::Completed, format!("{} marked", set.count()))
        }
        Err(L2dError::Refused { reason }) => (Outcome::Refused, reason.clone()),
        Err(other) => panic!("{shape}: {other:?}"),
    };
    let growth = rust + sqlite;
    eprintln!(
        "16.12 store {shape:<20} batch {:>6}  usable {:>4} MiB  {:<9}  rust {:>5.1} MiB + sqlite {:>5.1} MiB = {:>4.2} x usable  {elapsed:.1?}  {detail}",
        batch_rows.map_or("auto".to_string(), |rows| rows.to_string()),
        usable / MIB,
        format!("{outcome:?}"),
        rust as f64 / MIB as f64,
        sqlite as f64 / MIB as f64,
        growth as f64 / usable as f64,
    );
    assert!(
        growth <= usable,
        "{shape}: the heaps grew by {rust} + {sqlite} bytes under a limit that leaves {usable} \
         bytes beside the baseline ({outcome:?})"
    );
    assert_eq!(budget.held_bytes(), 0, "{shape}: nothing stays held");
    if outcome == Outcome::Refused {
        REFUSALS.lock().unwrap().push(detail);
    }
    outcome
}

/// The reasons of the refusals so far.
static REFUSALS: Mutex<Vec<String>> = Mutex::new(Vec::new());

#[test]
fn marking_a_store_stays_within_its_memory_limit() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch {
        root: std::env::temp_dir().join(format!("l2d-store-marks-memory-{}", std::process::id())),
    };
    let _ = std::fs::remove_dir_all(&scratch.root);
    let rows = 30_000u64;
    let records: Vec<LogRecord> = (0..rows)
        .map(|index| LogRecord {
            text: format!(
                "203.0.113.{} GET /shop/item/{index}/detail?session={} HTTP/1.1 200",
                index % 250,
                index * 7919
            ),
            label: Label::Unknown,
            group: None,
            first_position: index + 1,
            occurrences: 1,
        })
        .collect();
    let logs = LogsFrame::from_records(&records).unwrap();
    drop(records);
    let vectors: Vec<f32> = (0..rows).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let flags = vec![LogFlags::default(); rows as usize];
    let frame = rows_frame(&logs, "synthetic", &vectors, DIM, &flags).unwrap();
    let identity = identity();
    let origin = VectorOrigin::without_release(&identity);
    let paths = ProjectPaths::new(&scratch.root);
    {
        let mut store = Store::create(&paths, "s", origin).unwrap();
        let mut budget =
            MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
        store.add_with(&frame, origin, &mut budget).unwrap();
    }
    drop((frame, logs, vectors, flags));
    let store = Store::open(&paths, "s").unwrap();

    let every = MarkQuery::Regex {
        pattern: r"GET /shop/item/\d+/detail".into(),
    };
    let phrase = MarkQuery::Phrase {
        phrase: "item/1".into(),
    };
    let phrase_marked = (0..rows)
        .filter(|index| format!("/shop/item/{index}/").contains("item/1"))
        .count() as u64;
    // 25 and 26 MiB leave the regular expression room to start, not to hold every id.
    let ladder = [16, 24, 25, 26, 28, 32, 48].map(|mib| mib * MIB);
    for (shape, query, marked) in [
        ("regex, every log", &every, rows),
        ("phrase, some logs", &phrase, phrase_marked),
    ] {
        for batch_rows in [None, Some(1_024)] {
            let outcomes: Vec<Outcome> = ladder
                .iter()
                .map(|&usable| mark_within(shape, &store, query, usable, batch_rows, marked))
                .collect();
            assert_eq!(
                outcomes[0],
                Outcome::Refused,
                "{shape}: the smallest is refused"
            );
            assert_eq!(
                outcomes.last(),
                Some(&Outcome::Completed),
                "{shape}: the largest completes"
            );
        }
    }
    // Some read was refused while it held ids already: the ids are counted as they grow.
    let refusals = REFUSALS.lock().unwrap();
    assert!(
        refusals
            .iter()
            .any(|reason| reason.contains("logs marked so far")
                && !reason.contains("the ids of 0 logs")),
        "{refusals:#?}"
    );
}
