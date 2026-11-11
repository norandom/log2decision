//! Requirement 12.9 measured: the memory a read really takes, against its memory limit.
//!
//! This test binary counts every heap allocation of the process (a counting global
//! allocator around the system allocator), so the figures do not depend on how the
//! operating system trims working sets. For each source shape a ladder of memory limits
//! is tried; whether a read completes or is refused, the heap may not have grown by more
//! than the limit less the fixed share the tool reserves for the process itself
//! (`PROCESS_BASELINE_BYTES`: code, stacks, thread pool; not on the counted heap).
//!
//! The bound asserted is factor 1: growth of the heap <= limit - baseline.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_core::{
    InputAdapter, L2dError, LabelledAdapter, LabelledFormat, LinesAdapter, MemoryBudget,
    MemorySettings,
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

struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "l2d-core-memory-limit-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    /// Writes `rows` lines made by `line` to a file called `name`.
    fn file(&self, name: &str, rows: u64, line: impl Fn(u64) -> String) -> PathBuf {
        let path = self.dir.join(name);
        let mut out = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        for index in 0..rows {
            out.write_all(line(index).as_bytes()).unwrap();
            out.write_all(b"\n").unwrap();
        }
        out.flush().unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Completed,
    Refused,
}

/// Reads `path` under a limit that leaves `usable` bytes beside the baseline, with
/// plenty of free memory, and checks the heap growth against it.
fn read_within(
    shape: &str,
    adapter: &dyn InputAdapter,
    path: &Path,
    usable: u64,
    batch_rows: Option<u64>,
    records: u64,
) -> Outcome {
    let limit = BASELINE + usable;
    let settings = MemorySettings {
        memory_limit_bytes: Some(limit),
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap();
    let started = std::time::Instant::now();
    let (result, growth) = measure(|| adapter.read_with(path, &mut budget));
    let elapsed = started.elapsed();
    let (outcome, detail) = match &result {
        Ok((logs, summary, plan)) => {
            assert_eq!(summary.records, records, "{shape}");
            (
                Outcome::Completed,
                format!(
                    "{} distinct, {} skipped, {} bytes per row, batches {}..{}",
                    logs.len(),
                    summary.skipped.len(),
                    plan.bytes_per_row,
                    plan.min_rows_used,
                    plan.max_rows_used
                ),
            )
        }
        Err(L2dError::Refused { reason }) => (Outcome::Refused, reason.clone()),
        Err(other) => panic!("{shape}: {other:?}"),
    };
    eprintln!(
        "12.9 {shape:<28} batch {:>6}  usable {:>4} MiB  {:<9}  heap growth {:>6.1} MiB = {:>4.2} x usable  {elapsed:.1?}  {detail}",
        batch_rows.map_or("auto".to_string(), |rows| rows.to_string()),
        usable / MIB,
        format!("{outcome:?}"),
        growth as f64 / MIB as f64,
        growth as f64 / usable as f64,
    );
    assert!(
        growth <= usable,
        "{shape}: the heap grew by {growth} bytes under a limit of {limit} bytes that leaves \
         {usable} bytes beside the baseline ({outcome:?})"
    );
    drop(result);
    outcome
}

/// The shapes at `scale` thousand rows each, every one under a ladder of limits.
fn shapes_stay_within_their_limits(scale: u64, ladder: &[u64]) {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new(&format!("scale-{scale}"));
    let rows = scale * 1_000;
    let jsonl = LabelledAdapter::new(LabelledFormat::Jsonl);

    let short = scratch.file("short-distinct.log", rows, |index| format!("u{index}"));
    let blank = scratch.file("all-blank.log", rows, |_| " ".to_string());
    let not_json = scratch.file("not-json.jsonl", rows / 2, |index| {
        format!("not json {index}")
    });
    let medium = scratch.file("medium-distinct.log", rows / 2, |index| {
        format!(
            "203.0.113.{} - - [11/Nov/2026:10:00:00 +0000] \"GET /shop/item/{index}/detail?session={} HTTP/1.1\" 200 5120",
            index % 250,
            index * 7919
        )
    });
    let repeats = scratch.file("thousand-distinct.log", rows, |index| {
        format!("GET /item/{}?page=1 HTTP/1.1", index % 1_000)
    });
    let labelled = scratch.file("labelled.jsonl", rows / 2, |index| {
        format!("{{\"text\":\"GET /a/{index}\",\"label\":\"good\"}}")
    });

    // Warm up: the first read starts the thread pool, which is not the read's memory.
    LinesAdapter.read(&repeats).unwrap();

    let cases: [(&str, &dyn InputAdapter, &Path, u64); 6] = [
        ("short distinct lines", &LinesAdapter, &short, rows),
        ("blank lines, all skipped", &LinesAdapter, &blank, rows),
        ("not JSON, all skipped", &jsonl, &not_json, rows / 2),
        ("medium distinct lines", &LinesAdapter, &medium, rows / 2),
        ("1,000 distinct lines", &LinesAdapter, &repeats, rows),
        ("labelled JSON lines", &jsonl, &labelled, rows / 2),
    ];
    for (shape, adapter, path, records) in cases {
        let outcomes: Vec<Outcome> = ladder
            .iter()
            .map(|usable| read_within(shape, adapter, path, usable * MIB, None, records))
            .collect();
        // Once a limit is enough, every larger one is.
        let first_completed = outcomes
            .iter()
            .position(|outcome| *outcome == Outcome::Completed);
        let completed_from = first_completed.unwrap_or(outcomes.len());
        assert!(
            outcomes[completed_from..]
                .iter()
                .all(|outcome| *outcome == Outcome::Completed),
            "{shape}: {outcomes:?}"
        );
        assert_eq!(
            outcomes.last(),
            Some(&Outcome::Completed),
            "{shape}: the largest limit of the ladder is enough"
        );
        if shape == "1,000 distinct lines" {
            assert_eq!(first_completed, Some(0), "few distinct texts need little");
        } else {
            assert_eq!(
                outcomes[0],
                Outcome::Refused,
                "{shape}: the smallest limit of the ladder is not enough"
            );
        }
        // Small batches, as memory pressure would choose them, at the limit that is
        // just enough.
        if let Some(index) = first_completed {
            read_within(
                shape,
                adapter,
                path,
                ladder[index] * MIB,
                Some(1_024),
                records,
            );
        }
    }
}

#[test]
fn a_read_stays_within_its_memory_limit_whether_it_completes_or_is_refused() {
    shapes_stay_within_their_limits(100, &[6, 16, 48, 160, 512]);
}

/// The same at twenty times the rows (2,000,000 per shape), for release builds:
/// `cargo test --release -p l2d-core --test memory_limit -- --ignored --nocapture`
#[test]
#[ignore = "large inputs: run in release mode"]
fn large_reads_stay_within_their_memory_limit() {
    shapes_stay_within_their_limits(2_000, &[6, 64, 256, 1_024, 4_096]);
}
