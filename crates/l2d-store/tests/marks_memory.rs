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
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_core::marks::{MarkQuery, MarkSet};
use l2d_core::{
    EncoderIdentity, L2dError, Label, LogRecord, LogsFrame, MemoryBudget, MemorySettings,
    ProjectPaths,
};
use l2d_store::{LogFlags, Store, VectorOrigin, rows_frame};
use sha2::{Digest, Sha256};

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
            if set.ids_spilled() {
                SPILLED.fetch_add(1, Ordering::Relaxed);
                assert!(budget.usage().report().spilled);
            }
            let mut hash = Sha256::new();
            set.for_each_id(|id| {
                hash.update(id.as_bytes());
                Ok(())
            })
            .unwrap();
            let digest = format!("{:x}", hash.finalize());
            let mut hashes = ID_HASHES.lock().unwrap();
            if let Some(previous) = hashes.insert(shape.to_string(), digest.clone()) {
                assert_eq!(
                    digest, previous,
                    "{shape}: the same sorted IDs at every budget and batch size"
                );
            }
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
    outcome
}

static SPILLED: AtomicUsize = AtomicUsize::new(0);
static ID_HASHES: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

#[test]
fn marking_a_store_stays_within_its_memory_limit() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch {
        root: std::env::temp_dir().join(format!("l2d-store-marks-memory-{}", std::process::id())),
    };
    let _ = std::fs::remove_dir_all(&scratch.root);
    let rows = 70_000u64;
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
    // The smaller budgets refuse before SQLite starts; successful tight budgets
    // spill the IDs, and larger ones retain them. Both must give the same IDs.
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
    assert!(
        SPILLED.load(Ordering::Relaxed) > 0,
        "tight successful marks must spill instead of refusing growing IDs"
    );
    assert_eq!(ID_HASHES.lock().unwrap().len(), 2);
}

#[test]
fn selection_fts_native_heap_is_reserved_before_match_execution() {
    use l2d_core::selections::{SearchQuery, SearchSettings, SelectionSet};
    use l2d_store::{INDEX_TEXT_FACTOR, PHRASE_BYTE_BYTES, SEARCH_FIXED_BYTES};
    let _serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch {
        root: std::env::temp_dir().join(format!("l2d-selection-fts-native-{}", std::process::id())),
    };
    let paths = ProjectPaths::new(&scratch.root);
    let mut records: Vec<LogRecord> = (0..200)
        .map(|i| LogRecord {
            text: format!("request {i} needle"),
            label: Label::Unknown,
            group: None,
            first_position: i + 1,
            occurrences: 1,
        })
        .collect();
    records.push(LogRecord {
        text: "a".repeat(64 * 1024),
        label: Label::Unknown,
        group: None,
        first_position: 201,
        occurrences: 1,
    });
    records.push(LogRecord {
        text: "a*b literal star".into(),
        label: Label::Unknown,
        group: None,
        first_position: 202,
        occurrences: 1,
    });
    let logs = LogsFrame::from_records(&records).unwrap();
    let vectors: Vec<f32> = (0..records.len())
        .flat_map(|_| [1.0, 0.0, 0.0, 0.0])
        .collect();
    let frame = rows_frame(
        &logs,
        "native-regression",
        &vectors,
        DIM,
        &vec![LogFlags::default(); records.len()],
    )
    .unwrap();
    let identity = identity();
    let mut roomy =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    {
        let mut store =
            Store::create(&paths, "fts", VectorOrigin::without_release(&identity)).unwrap();
        store
            .add_with(&frame, VectorOrigin::without_release(&identity), &mut roomy)
            .unwrap();
    }
    drop((records, logs, vectors, frame));
    let store = Store::open(&paths, "fts").unwrap();
    let phrase = format!("\"{}\"", "a".repeat(891));
    let expressions = [
        phrase.clone(),
        format!("{phrase} OR \"needle\""),
        format!("({phrase} AND \"aaaa\") NOT \"needle\""),
    ];
    let budget = |usable| {
        MemoryBudget::observe_with(
            &MemorySettings {
                memory_limit_bytes: Some(BASELINE + usable),
                ..Default::default()
            },
            || Ok(64 * GIB),
        )
        .unwrap()
    };
    for (expression, expected) in expressions.iter().zip([1, 201, 1]) {
        for (usable, should_refuse) in [(40 * MIB, true), (512 * MIB, false)] {
            let mut b = budget(usable);
            b.hold(MIB); // caller state, beside SelectionSet's own reservations
            let (sqlite_before, _) = sqlite_now_and_peak();
            let rust_before = CURRENT.load(Ordering::Relaxed);
            PEAK.store(rust_before, Ordering::Relaxed);
            let result = SelectionSet::search_store(
                "fts",
                SearchQuery::Fts5Trigram {
                    expression: expression.clone(),
                },
                SearchSettings { max_members: 512 },
                &store,
                &mut b,
            );
            let rust = PEAK.load(Ordering::Relaxed).saturating_sub(rust_before) as u64;
            let (_, sqlite_peak) = sqlite_now_and_peak();
            let sqlite = sqlite_peak.saturating_sub(sqlite_before);
            eprintln!(
                "FTS expression bytes={} usable={usable} rust={rust} sqlite={sqlite} refused={}",
                expression.len(),
                result.is_err()
            );
            assert_eq!(b.held_bytes(), MIB);
            if should_refuse {
                assert!(
                    matches!(result, Err(L2dError::Refused { .. })),
                    "long expression must refuse BEFORE MATCH at {usable} bytes; Rust={rust}, SQLite={sqlite}"
                );
                assert!(
                    sqlite <= SEARCH_FIXED_BYTES,
                    "native MATCH ran before refusal: {sqlite}"
                );
            } else {
                let set = result.unwrap();
                assert_eq!(set.members().len(), expected);
                let index = PHRASE_BYTE_BYTES * expression.len() as u64
                    + INDEX_TEXT_FACTOR * expression.chars().count() as u64 * 64 * 1024;
                assert!(sqlite <= SEARCH_FIXED_BYTES + index);
            }
            assert!(
                MIB + rust + sqlite <= usable,
                "combined heaps exceed usable: caller {} + Rust {rust} + SQLite {sqlite} > {usable}",
                MIB
            );
        }
    }
    // Isolate longest-text position lists: one byte below the exact native
    // reservation must refuse even though the phrase-reader share alone fits.
    use l2d_core::selections::SelectionStore;
    let short = "\"aaaa\"";
    let native = PHRASE_BYTE_BYTES * short.len() as u64
        + INDEX_TEXT_FACTOR * short.chars().count() as u64 * 64 * 1024;
    for (usable, refused) in [
        (MIB + SEARCH_FIXED_BYTES + native - 1, true),
        (MIB + SEARCH_FIXED_BYTES + native, false),
    ] {
        let mut b = budget(usable);
        b.hold(MIB);
        let (sqlite_before, _) = sqlite_now_and_peak();
        let mut callbacks = 0;
        let result = store.fts5_trigram_ids(short, &mut b, &mut |_| {
            callbacks += 1;
            Ok(())
        });
        let (_, peak) = sqlite_now_and_peak();
        assert!(MIB + peak.saturating_sub(sqlite_before) <= usable);
        assert_eq!(b.held_bytes(), MIB);
        if refused {
            assert!(matches!(result, Err(L2dError::Refused { .. })));
            assert_eq!(callbacks, 0);
        } else {
            assert_eq!(result.unwrap(), 202);
            assert_eq!(callbacks, 1);
        }
    }
    // A memory refusal precedes even malformed MATCH preparation.
    let error = SelectionSet::search_store(
        "syntax",
        SearchQuery::Fts5Trigram {
            expression: format!("{phrase} AND"),
        },
        SearchSettings { max_members: 512 },
        &store,
        &mut budget(40 * MIB),
    )
    .unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }));
    let prefix = SelectionSet::search_store(
        "prefix",
        SearchQuery::Fts5Trigram {
            expression: "aaa*".into(),
        },
        SearchSettings { max_members: 512 },
        &store,
        &mut budget(512 * MIB),
    )
    .unwrap_err();
    assert!(matches!(prefix, L2dError::Refused { .. }));
    assert!(prefix.to_string().contains("prefix expansion"));
    let literal_star = SelectionSet::search_store(
        "literal-star",
        SearchQuery::Fts5Trigram {
            expression: "\"a*b\"".into(),
        },
        SearchSettings { max_members: 512 },
        &store,
        &mut budget(40 * MIB),
    )
    .unwrap();
    assert_eq!(literal_star.members().len(), 1);
}
