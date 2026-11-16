//! Requirement 16.12 measured (with 12.9, 13.8 and 15.2): marking an archive stays
//! within its memory limit (task 1.21).
//!
//! Like `zip_memory.rs`, this test binary counts every heap allocation of the process
//! (a counting global allocator around the system allocator). For each shape a ladder
//! of memory limits is tried; whether the marking completes or is refused, the heap may
//! not have grown by more than the limit less the share the tool keeps for the process
//! itself (`PROCESS_BASELINE_BYTES`). Opening the archive, every batch, the marked
//! positions, the compiled pattern and the mark set returned are inside the
//! measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_core::marks::{MarkQuery, MarkSet};
use l2d_core::{
    ArchivePosition, L2dError, Label, MemoryBudget, MemorySettings, ZipInput, ZipInputSettings,
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
            "l2d-core-marks-memory-{}-{name}",
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

/// Marks `path` under a limit that leaves `usable` bytes beside the baseline, with
/// plenty of free memory, and checks the heap growth against it.
fn mark_within(
    shape: &str,
    path: &Path,
    query: &MarkQuery,
    usable: u64,
    batch_rows: Option<u64>,
    marked: u64,
) -> Outcome {
    let limit = BASELINE + usable;
    let memory = MemorySettings {
        memory_limit_bytes: Some(limit),
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
    let settings = ZipInputSettings::default();
    let started = std::time::Instant::now();
    let (result, growth) = measure(|| {
        MarkSet::mark_archive(
            "marks",
            Label::Bad,
            query.clone(),
            path,
            &settings,
            &mut budget,
        )
    });
    let elapsed = started.elapsed();
    let (outcome, detail) = match &result {
        Ok((set, summary)) => {
            assert_eq!(set.count(), marked, "{shape}");
            assert_eq!(set.positions().len() as u64, marked, "{shape}");
            (
                Outcome::Completed,
                format!(
                    "{} marked of {} distinct",
                    set.count(),
                    summary.input.distinct_texts
                ),
            )
        }
        Err(L2dError::Refused { reason }) => (Outcome::Refused, reason.clone()),
        Err(other) => panic!("{shape}: {other:?}"),
    };
    eprintln!(
        "16.12 {shape:<22} batch {:>6}  usable {:>4} MiB  {:<9}  heap growth {:>6.1} MiB = {:>4.2} x usable  {elapsed:.1?}  {detail}",
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

#[test]
fn marking_an_archive_stays_within_its_memory_limit() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("shapes");
    let rows = 100_000u64;
    let archive = scratch.lines("medium-distinct.zip", rows, |index| {
        format!(
            "203.0.113.{} - - [11/Nov/2026:10:00:00 +0000] \"GET /shop/item/{index}/detail?session={} HTTP/1.1\" 200 5120",
            index % 250,
            index * 7919
        )
    });
    // Every line is marked: the positions grow with every batch.
    let every = MarkQuery::Regex {
        pattern: r"GET /shop/item/\d+/detail".into(),
    };
    // Every other line, by position: the positions asked for are held as well.
    let every_other = MarkQuery::Positions {
        positions: (1..=rows)
            .step_by(2)
            .map(|line| ArchivePosition { entry: 0, line }.pack())
            .collect(),
    };
    let phrase = MarkQuery::Phrase {
        phrase: "item/1".into(),
    };
    let phrase_marked = (0..rows)
        .filter(|index| format!("/shop/item/{index}/").contains("item/1"))
        .count() as u64;

    let ladder = [8, 12, 16, 24, 48].map(|mib| mib * MIB);
    for (shape, query, marked) in [
        ("regex, every line", &every, rows),
        ("positions, every other", &every_other, rows.div_ceil(2)),
        ("phrase, some lines", &phrase, phrase_marked),
    ] {
        for batch_rows in [None, Some(1_024)] {
            let outcomes: Vec<Outcome> = ladder
                .iter()
                .map(|&usable| mark_within(shape, &archive, query, usable, batch_rows, marked))
                .collect();
            assert_eq!(
                outcomes.last(),
                Some(&Outcome::Completed),
                "{shape}: the largest limit completes"
            );
        }
    }
}

/// The compiled regular expression is counted as `REGEX_HELD_BYTES`: a pattern of
/// 2,000 alternatives, near the most the size limits let compile (3,000 of them did
/// not, on 2026-11-16), stays within it. One beyond the size limits is refused, with a
/// message of bounded length.
#[test]
fn a_large_regular_expression_stays_within_what_it_is_counted_as() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("large-pattern");
    let rows = 2_000u64;
    let archive = scratch.lines("small.zip", rows, |index| {
        format!(
            "request {index} from host-{} with token t{} done",
            index % 13,
            index * 7919
        )
    });
    // Alternatives of which every tenth is in the archive.
    let words = |count: u64| -> Vec<String> {
        (0..count)
            .map(|index| format!("t{}", index * 7919 * if index % 10 == 0 { 1 } else { 3 }))
            .collect()
    };
    let marked_by = |words: &[String]| {
        (0..rows)
            .filter(|index| words.contains(&format!("t{}", index * 7919)))
            .count() as u64
    };

    let held = l2d_core::marks::REGEX_HELD_BYTES;
    let ladder = [held / 2, held + 4 * MIB, held + 8 * MIB];
    for count in [500u64, 2_000] {
        let words = words(count);
        let query = MarkQuery::Regex {
            pattern: format!("token (?:{}) done", words.join("|")),
        };
        let shape = format!("regex, {count} alternatives");
        let outcomes: Vec<Outcome> = ladder
            .iter()
            .map(|&usable| mark_within(&shape, &archive, &query, usable, None, marked_by(&words)))
            .collect();
        assert_eq!(outcomes[0], Outcome::Refused, "{shape}: counted first");
        assert_eq!(outcomes.last(), Some(&Outcome::Completed), "{shape}");
    }

    // Beyond the size limit of the compiled program: refused before anything is read.
    let pattern = format!(r"token (?:{})\b", words(5_000).join("|"));
    assert!(pattern.len() <= l2d_core::marks::MAX_PATTERN_BYTES);
    let mut budget = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(BASELINE + 64 * MIB),
            ..MemorySettings::default()
        },
        || Ok(64 * GIB),
    )
    .unwrap();
    let error = MarkSet::mark_archive(
        "too-large",
        Label::Bad,
        MarkQuery::Regex { pattern },
        &archive,
        &ZipInputSettings::default(),
        &mut budget,
    )
    .unwrap_err();
    let L2dError::InvalidInput { detail } = error else {
        panic!("expected InvalidInput, got {error:?}");
    };
    assert!(detail.contains("size limit"), "{detail}");
    assert!(detail.len() < 1_000, "{} bytes: {detail}", detail.len());
}

/// The query's share is counted with what the archive holds once its directory is read,
/// before the pattern is compiled (review of task 1.21, round 1): an archive of many
/// entries whose directory fits the limit, but not with the share of a regular
/// expression of 2,000 alternatives, is refused before that expression is built.
#[test]
fn the_query_and_the_directory_are_counted_before_the_pattern_is_compiled() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("many-entries");
    let path = scratch.dir.join("many-entries.zip");
    {
        let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for index in 0..40_000u32 {
            writer
                .start_file(
                    format!(
                        "logs/host-{:05}/a/rather/long/directory/name/that/makes/the/central/\
                         directory/large/file-{index}.log",
                        index % 97
                    ),
                    options,
                )
                .unwrap();
            writer.write_all(b"one line\n").unwrap();
        }
        writer.finish().unwrap().flush().unwrap();
    }
    let settings = ZipInputSettings::default();
    // What the archive holds once its directory is read.
    let held = {
        let budget =
            MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
        ZipInput::open(&path, &settings, &budget)
            .unwrap()
            .held_bytes()
    };
    let regex = l2d_core::marks::REGEX_HELD_BYTES;
    assert!(
        held > regex,
        "the directory alone is larger than the regex's share"
    );
    // The directory fits with room to spare, the regex's share beside it does not.
    let usable = held + 2 * MIB;
    let words: Vec<String> = (0..2_000u64)
        .map(|index| format!("t{}", index * 7919))
        .collect();
    let query = MarkQuery::Regex {
        pattern: format!("line (?:{})", words.join("|")),
    };
    let mut budget = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(BASELINE + usable),
            ..MemorySettings::default()
        },
        || Ok(64 * GIB),
    )
    .unwrap();
    let (result, growth) =
        measure(|| MarkSet::mark_archive("many", Label::Bad, query, &path, &settings, &mut budget));
    eprintln!(
        "16.12 query before compiling: directory held {:.1} MiB, usable {:.1} MiB, heap growth \
         {:.1} MiB, {result:?}",
        held as f64 / MIB as f64,
        usable as f64 / MIB as f64,
        growth as f64 / MIB as f64
    );
    let Err(L2dError::Refused { reason }) = result else {
        panic!("expected Refused, got {result:?}");
    };
    assert!(reason.contains("before it builds its query"), "{reason}");
    assert!(
        growth <= usable,
        "the heap grew by {growth} bytes under a limit that leaves {usable} bytes"
    );
}
