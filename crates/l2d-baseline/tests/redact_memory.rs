//! Requirement 16.12 measured (with 12.9, 13.8 and 15.2): proposing redactions over an
//! archive stays within its memory limit (task 1.22).
//!
//! Like `l2d-core`'s `marks_memory.rs`, this test binary counts every heap allocation
//! of the process (a counting global allocator around the system allocator). For a
//! ladder of memory limits the proposals are asked for; whether they complete or are
//! refused, the heap may not have grown by more than the limit less the share the tool
//! keeps for the process itself (`PROCESS_BASELINE_BYTES`). Both reads of the archive,
//! every batch, the counters and the proposals returned are inside the measurement; the
//! mark set is made before it, under a roomy budget.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_baseline::redact::{RedactionInput, RedactionSettings, propose_redactions};
use l2d_core::marks::{MarkQuery, MarkSet};
use l2d_core::{L2dError, Label, MemoryBudget, MemorySettings, ZipInputSettings};

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

struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "l2d-baseline-redact-memory-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    /// An archive with one deflated entry of `rows` lines made by `line`.
    fn lines(&self, name: &str, rows: u64, line: impl Fn(u64) -> String) -> PathBuf {
        let path = self.dir.join(name);
        let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut writer = zip::ZipWriter::new(file);
        writer
            .start_file("logs/all.log", zip::write::SimpleFileOptions::default())
            .unwrap();
        for index in 0..rows {
            writer.write_all(line(index).as_bytes()).unwrap();
            writer.write_all(b"\n").unwrap();
        }
        writer.finish().unwrap().flush().unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    Completed,
    Refused,
}

/// Proposes redactions for `set` over `path` under a limit that leaves `usable` bytes
/// beside the baseline, and checks the heap growth against it.
fn propose_within(
    set: &MarkSet,
    path: &Path,
    usable: u64,
    batch_rows: Option<u64>,
    settings: &RedactionSettings,
) -> Outcome {
    let limit = BASELINE + usable;
    let memory = MemorySettings {
        memory_limit_bytes: Some(limit),
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
    let zip = ZipInputSettings::default();
    let started = std::time::Instant::now();
    let (result, growth) = measure(|| {
        propose_redactions(
            set,
            RedactionInput::Archive {
                path,
                settings: &zip,
            },
            settings,
            &mut budget,
        )
    });
    let elapsed = started.elapsed();
    let (outcome, detail) = match &result {
        Ok(proposals) => {
            assert!(
                proposals.get("x.jsp").is_some(),
                "{:?}",
                proposals.proposals
            );
            (
                Outcome::Completed,
                format!(
                    "{} proposals, {} marked and {} unmarked logs, counters {:.1} MiB, \
                     dropped {}",
                    proposals.proposals.len(),
                    proposals.marked_logs,
                    proposals.unmarked_logs,
                    proposals.counters_bytes as f64 / MIB as f64,
                    proposals.dropped
                ),
            )
        }
        Err(L2dError::Refused { reason }) => (Outcome::Refused, reason.clone()),
        Err(other) => panic!("{other:?}"),
    };
    eprintln!(
        "16.12 redaction slots {:>6}  batch {:>6}  usable {:>4} MiB  {:<9}  heap growth {:>6.1} MiB = {:>4.2} x usable  {elapsed:.1?}  {detail}",
        settings.counter_slots,
        batch_rows.map_or("auto".to_string(), |rows| rows.to_string()),
        usable / MIB,
        format!("{outcome:?}"),
        growth as f64 / MIB as f64,
        growth as f64 / usable as f64,
    );
    assert!(
        growth <= usable,
        "the heap grew by {growth} bytes under a limit of {limit} bytes that leaves {usable} \
         bytes beside the baseline ({outcome:?})"
    );
    drop(result);
    outcome
}

#[test]
fn proposing_redactions_over_an_archive_stays_within_its_memory_limit() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("ladder");
    // 60,000 distinct lines: 50 hosts, 20 endpoints, a number and a token that differ in
    // almost every line, and a flag the analyst marks by (a third of the lines).
    let path = scratch.lines("synthetic.zip", 60_000, |i| {
        format!(
            "GET http://h{}.example/p{}/x.jsp?a={i}&b=z{}&m={}",
            i % 50,
            i % 20,
            (i * 31) % 99_991,
            (i / 7) % 3
        )
    });
    let mut roomy =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    let (set, _) = MarkSet::mark_archive(
        "flagged",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "&m=0".into(),
        },
        &path,
        &ZipInputSettings::default(),
        &mut roomy,
    )
    .unwrap();
    // The lines with `(i / 7) % 3 == 0`.
    assert_eq!(set.count(), 20_002);

    let defaults = RedactionSettings::default();
    let small = RedactionSettings {
        counter_slots: 2_048,
        max_fragment_bytes: 64,
        ..RedactionSettings::default()
    };
    let mut outcomes = Vec::new();
    for settings in [&small, &defaults] {
        for usable in [4 * MIB, 8 * MIB, 16 * MIB, 32 * MIB, 64 * MIB] {
            for batch_rows in [None, Some(1_024)] {
                outcomes.push((
                    settings.counter_slots,
                    usable,
                    propose_within(&set, &path, usable, batch_rows, settings),
                ));
            }
        }
    }
    // The smallest limit is refused, the largest completes, for both counter sizes.
    for slots in [small.counter_slots, defaults.counter_slots] {
        let of = |usable| {
            outcomes
                .iter()
                .filter(|(s, u, _)| *s == slots && *u == usable)
                .map(|(_, _, outcome)| *outcome)
                .collect::<Vec<_>>()
        };
        assert_eq!(of(4 * MIB), [Outcome::Refused; 2], "{slots} slots");
        assert_eq!(of(64 * MIB), [Outcome::Completed; 2], "{slots} slots");
    }
}
