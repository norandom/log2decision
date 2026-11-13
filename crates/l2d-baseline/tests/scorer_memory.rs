//! The heap a load and a refused fit take, counted (requirements 12.5, 12.9; review of
//! task 1.11).
//!
//! - `BaselineScorer::load` stays within `BaselineScorer::load_bytes`, and that figure
//!   is about the size of the files, not a multiple of the weights.
//! - A fit that is refused for memory has allocated next to nothing when it refuses,
//!   also when a text is very long: the refusal comes before a row is hashed.
//!
//! The tests share one heap counter and run one at a time.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_baseline::scorer::{FITTED_TEXTS_FILE, MANIFEST_FILE, WEIGHTS_FILE};
use l2d_baseline::{BaselineScorer, BaselineSettings, FitError, RoutingMode};
use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::{L2dError, Label, LogRecord, LogsFrame, MemoryBudget, MemorySettings, ProjectPaths};

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

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Runs `run` and returns its result and by how many bytes the heap grew at its peak.
fn measure<T>(run: impl FnOnce() -> T) -> (T, u64) {
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = run();
    let peak = PEAK.load(Ordering::Relaxed);
    (result, peak.saturating_sub(before) as u64)
}

fn budget_with_limit(limit: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).unwrap()
}

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("l2d-baseline-memory-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Self { root }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn record(index: u64, text: String) -> LogRecord {
    LogRecord {
        text,
        label: if index.is_multiple_of(2) {
            Label::Good
        } else {
            Label::Bad
        },
        group: None,
        first_position: index + 1,
        occurrences: 1,
    }
}

/// 400 short labelled rows.
fn short_rows() -> Vec<LogRecord> {
    (0..400)
        .map(|index: u64| {
            let payload = if index.is_multiple_of(2) {
                "q=wine"
            } else {
                "q=' OR 1=1 --"
            };
            record(
                index,
                format!("GET /shop/search.jsp?id={index:04}&{payload}"),
            )
        })
        .collect()
}

#[test]
fn a_load_stays_within_what_load_bytes_states_and_that_is_about_the_files() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // 524,288 columns: a weights file of 4 MiB, far above every buffer of the load.
    let settings = BaselineSettings {
        buckets: 262_144,
        routing: RoutingMode::Isolate,
        max_iter: 2,
        ..BaselineSettings::default()
    };
    let logs = LogsFrame::from_records(&short_rows()).unwrap();
    let scorer = BaselineScorer::fit_with(&logs, &[], &settings, &budget_with_limit(None)).unwrap();
    let scratch = Scratch::new("load");
    let dir = scorer
        .save(&ProjectPaths::new(&scratch.root), "wide")
        .unwrap();
    let size = |file: &str| std::fs::metadata(dir.join(file)).unwrap().len();
    let weights = size(WEIGHTS_FILE);
    assert_eq!(weights, 4 * MIB);
    let files = weights + size(FITTED_TEXTS_FILE) + size(MANIFEST_FILE);

    let stated = BaselineScorer::load_bytes(&dir);
    let (loaded, heap) = measure(|| BaselineScorer::load(&dir));
    let loaded = loaded.unwrap();
    println!(
        "load: weights {weights} bytes, files {files} bytes, load_bytes {stated}, heap peak \
         {heap} ({:.2} of what is stated, {:.2} of the weights)",
        heap as f64 / stated as f64,
        heap as f64 / weights as f64
    );
    assert_eq!(loaded, scorer);
    assert!(
        heap <= stated,
        "the load took {heap} bytes of heap at its peak, above the {stated} load_bytes states"
    );
    // The figure is the files and a buffer, not the weights several times over.
    assert!(stated >= files, "{stated} < {files}");
    assert!(
        stated <= files + 256 * 1024,
        "load_bytes states {stated} for files of {files} bytes"
    );
    // What the loaded scorer keeps is what it says it keeps.
    let kept = loaded.held_bytes();
    assert!(kept >= weights && kept <= weights + 64 * 1024, "{kept}");
}

#[test]
fn a_fit_with_a_very_long_text_is_refused_before_anything_is_hashed() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // One text of 2 MiB among short ones: hashing it is estimated at
    // 2 MiB * 3 n-grams * 40 bytes = 240 MiB.
    let mut records = short_rows();
    let long = format!("GET /shop/search.jsp?q={}", "a1b2c3d4".repeat(262_144));
    records.push(record(400, long));
    let logs = LogsFrame::from_records(&records).unwrap();
    let settings = BaselineSettings {
        buckets: 4_096,
        max_iter: 5,
        ..BaselineSettings::default()
    };

    for limit in [
        PROCESS_BASELINE_BYTES + 16 * MIB,
        PROCESS_BASELINE_BYTES + 200 * MIB,
    ] {
        let budget = budget_with_limit(Some(limit));
        let (result, heap) = measure(|| BaselineScorer::fit_with(&logs, &[], &settings, &budget));
        let reason = match result.unwrap_err() {
            FitError::Memory(L2dError::Refused { reason }) => reason,
            other => panic!("{other:?}"),
        };
        println!("refused fit at a limit of {limit} bytes: heap peak {heap} bytes");
        assert!(reason.contains("fitting the baseline scorer"), "{reason}");
        assert!(reason.contains("nothing was loaded"), "{reason}");
        assert!(
            heap <= 64 * 1024,
            "the refused fit took {heap} bytes of heap before it refused (limit {limit})"
        );
    }

    // With room the same fit succeeds, within its estimate.
    let budget = budget_with_limit(Some(PROCESS_BASELINE_BYTES + 400 * MIB));
    let (result, heap) = measure(|| BaselineScorer::fit_with(&logs, &[], &settings, &budget));
    let scorer = result.unwrap();
    let estimate = scorer.fit_report().estimated_bytes;
    println!(
        "fit with the long text: estimate {estimate} bytes, heap peak {heap} ({:.2})",
        heap as f64 / estimate as f64
    );
    assert!(estimate >= 240 * MIB, "{estimate}");
    assert!(heap <= estimate, "{heap} above the estimate {estimate}");
    assert_eq!(scorer.fit_report().rows, 401);
}
