//! Requirements 13.8, 13.12 and 12.9 measured: the memory reading a zip archive really
//! takes, against its memory limit.
//!
//! Like `memory_limit.rs`, this test binary counts every heap allocation of the process
//! (a counting global allocator around the system allocator). For each archive shape a
//! ladder of memory limits is tried; whether the read completes or is refused, the heap
//! may not have grown by more than the limit less the share the tool keeps for the
//! process itself (`PROCESS_BASELINE_BYTES`). Opening the archive (its hash and its
//! directory), every batch and the summary are inside the measurement.
//!
//! The last test reads an archive of the AIT Log Data Set and runs on request only:
//!
//! ```text
//! cargo test --release -j 4 -p l2d-core --test zip_memory -- --ignored --nocapture
//! ```
//!
//! `L2D_AIT_ARCHIVE` names another archive than `data/ait-lds-v2/russellmitchell.zip`,
//! `L2D_AIT_ENTRY` other entry patterns than `*.log` (separated by `;`), and
//! `L2D_AIT_LIMIT` another memory limit than `512MB`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_core::memory::parse_memory_size;
use l2d_core::{
    ArchiveSummary, BatchPlan, EntrySkip, L2dError, MemoryBudget, MemorySettings, ZipInput,
    ZipInputSettings,
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
        let dir =
            std::env::temp_dir().join(format!("l2d-core-zip-memory-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    /// Writes an archive called `name` whose entries are filled by `fill`.
    fn archive(
        &self,
        name: &str,
        fill: impl FnOnce(&mut zip::ZipWriter<std::io::BufWriter<std::fs::File>>),
    ) -> PathBuf {
        let path = self.dir.join(name);
        let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut writer = zip::ZipWriter::new(file);
        fill(&mut writer);
        writer.finish().unwrap().flush().unwrap();
        path
    }

    /// An archive with one deflated entry of `rows` lines made by `line`.
    fn lines(&self, name: &str, rows: u64, line: impl Fn(u64) -> String) -> PathBuf {
        self.archive(name, |writer| {
            writer
                .start_file("logs/all.log", zip::write::SimpleFileOptions::default())
                .unwrap();
            for index in 0..rows {
                writer.write_all(line(index).as_bytes()).unwrap();
                writer.write_all(b"\n").unwrap();
            }
        })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Reads the archive to its end, dropping every batch as a caller that only counts
/// would.
fn read_all(
    path: &Path,
    settings: &ZipInputSettings,
    budget: &mut MemoryBudget,
) -> Result<(ArchiveSummary, BatchPlan, u64), L2dError> {
    let mut input = ZipInput::open(path, settings, budget)?;
    let mut rows = 0;
    for batch in input.batches(budget) {
        rows += batch?.len() as u64;
    }
    let plan = input.batch_plan().cloned().expect("the plan of a read");
    Ok((input.finish(), plan, rows))
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    Completed,
    Refused,
}

/// Reads `path` under a limit that leaves `usable` bytes beside the baseline, with
/// plenty of free memory, and checks the heap growth against it.
fn read_within(
    shape: &str,
    path: &Path,
    settings: &ZipInputSettings,
    usable: u64,
    batch_rows: Option<u64>,
    records: u64,
) -> Outcome {
    let limit = BASELINE + usable;
    let memory = MemorySettings {
        memory_limit_bytes: Some(limit),
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
    let started = std::time::Instant::now();
    let (result, growth) = measure(|| read_all(path, settings, &mut budget));
    let elapsed = started.elapsed();
    let (outcome, detail) = match &result {
        Ok((summary, plan, rows)) => {
            assert_eq!(summary.input.records, records, "{shape}");
            assert_eq!(summary.input.distinct_texts, *rows, "{shape}");
            (
                Outcome::Completed,
                format!(
                    "{} distinct, {} skipped, {} entries, {} bytes per row, batches {}..{}",
                    summary.input.distinct_texts,
                    summary.input.skipped.len(),
                    summary.entries.len(),
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
        "13.8 {shape:<30} batch {:>6}  usable {:>4} MiB  {:<9}  heap growth {:>6.1} MiB = {:>4.2} x usable  {elapsed:.1?}  {detail}",
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

/// What a shape is expected to do at the smallest limit of the ladder.
#[derive(PartialEq, Clone, Copy)]
enum Smallest {
    Completes,
    IsRefused,
}

#[test]
fn archives_stay_within_their_memory_limits() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("shapes");
    let rows = 200_000u64;
    let defaults = ZipInputSettings::default();

    let short = scratch.lines("short-distinct.zip", rows, |index| format!("u{index}"));
    let blank = scratch.lines("all-blank.zip", rows, |_| " ".to_string());
    let medium = scratch.lines("medium-distinct.zip", rows / 2, |index| {
        format!(
            "203.0.113.{} - - [11/Nov/2026:10:00:00 +0000] \"GET /shop/item/{index}/detail?session={} HTTP/1.1\" 200 5120",
            index % 250,
            index * 7919
        )
    });
    let repeats = scratch.lines("thousand-distinct.zip", rows, |index| {
        format!("GET /item/{}?page=1 HTTP/1.1", index % 1_000)
    });
    // One line of 64 MiB between two short ones: it is skipped, never held.
    let long = scratch.archive("one-long-line.zip", |writer| {
        writer
            .start_file("long.log", zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"before\n").unwrap();
        let piece = vec![b'x'; MIB as usize];
        for _ in 0..64 {
            writer.write_all(&piece).unwrap();
        }
        writer.write_all(b"\nafter\n").unwrap();
    });
    // The same line in a gzip file that is a deflated entry: both inflaters at work,
    // and next to nothing else, so the growth of the heap is what the readers take.
    let long_gzip = scratch.archive("one-long-line-gzip.zip", |writer| {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(b"before\n").unwrap();
        let piece = vec![b'x'; MIB as usize];
        for _ in 0..64 {
            encoder.write_all(&piece).unwrap();
        }
        encoder.write_all(b"\nafter\n").unwrap();
        writer
            .start_file("long.log.gz", zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(&encoder.finish().unwrap()).unwrap();
    });
    // A gzip file of 1,000 distinct lines as a deflated entry.
    let gzip = scratch.archive("gzip-entry.zip", |writer| {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        for index in 0..rows {
            let line = format!("GET /item/{}?page=1 HTTP/1.1\n", index % 1_000);
            encoder.write_all(line.as_bytes()).unwrap();
        }
        writer
            .start_file("logs/all.log.gz", zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(&encoder.finish().unwrap()).unwrap();
    });
    // Lines of the largest record length the tool accepts.
    let widest = ZipInputSettings {
        max_record_bytes: l2d_core::zip_input::MAX_RECORD_BYTES,
        ..ZipInputSettings::default()
    };
    let wide = scratch.lines("wide-lines.zip", 60, |index| {
        format!("{index}:{}", "w".repeat(1_000_000))
    });
    // 30,000 entries of one line each, every tenth a gzip file.
    let many = scratch.archive("many-entries.zip", |writer| {
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for index in 0..30_000u32 {
            writer
                .start_file(format!("host-{}/logs/e{index}.log", index % 40), options)
                .unwrap();
            let line = format!("line of entry {}\n", index % 5_000);
            if index % 10 == 0 {
                let mut encoder =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                encoder.write_all(line.as_bytes()).unwrap();
                writer.write_all(&encoder.finish().unwrap()).unwrap();
            } else {
                writer.write_all(line.as_bytes()).unwrap();
            }
        }
    });

    // Warm up: the first read starts the thread pool, which is not the read's memory.
    let mut warm = MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    read_all(&repeats, &defaults, &mut warm).unwrap();

    // The usable memory in MiB. Sixty records of 1 MB are fewer than the smallest
    // batch, so the batcher wants them in one batch: six times their bytes.
    let ladder: &[u64] = &[6, 16, 48, 256];
    let wide_ladder: &[u64] = &[6, 64, 512];
    let many_ladder: &[u64] = &[3, 16, 256];
    type Case<'a> = (
        &'a str,
        &'a Path,
        &'a ZipInputSettings,
        u64,
        Smallest,
        &'a [u64],
    );
    let cases: [Case; 9] = [
        (
            "short distinct lines",
            &short,
            &defaults,
            rows,
            Smallest::IsRefused,
            ladder,
        ),
        (
            "blank lines, all skipped",
            &blank,
            &defaults,
            rows,
            Smallest::IsRefused,
            ladder,
        ),
        (
            "medium distinct lines",
            &medium,
            &defaults,
            rows / 2,
            Smallest::IsRefused,
            ladder,
        ),
        (
            "1,000 distinct lines",
            &repeats,
            &defaults,
            rows,
            Smallest::Completes,
            ladder,
        ),
        (
            "one line of 64 MiB, skipped",
            &long,
            &defaults,
            3,
            Smallest::Completes,
            ladder,
        ),
        (
            "64 MiB line in a gzip entry",
            &long_gzip,
            &defaults,
            3,
            Smallest::Completes,
            ladder,
        ),
        (
            "gzip entry, 1,000 distinct",
            &gzip,
            &defaults,
            rows,
            Smallest::Completes,
            ladder,
        ),
        (
            "60 lines of 1 MB",
            &wide,
            &widest,
            60,
            Smallest::IsRefused,
            wide_ladder,
        ),
        (
            "30,000 entries",
            &many,
            &defaults,
            30_000,
            Smallest::IsRefused,
            many_ladder,
        ),
    ];
    for (shape, path, settings, records, smallest, ladder) in cases {
        let largest = ladder[ladder.len() - 1] * MIB;
        let outcomes: Vec<Outcome> = ladder
            .iter()
            .map(|usable| read_within(shape, path, settings, usable * MIB, None, records))
            .collect();
        // Once a limit is enough, every larger one is.
        let completed_from = outcomes
            .iter()
            .position(|outcome| *outcome == Outcome::Completed)
            .unwrap_or(outcomes.len());
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
        let expected = match smallest {
            Smallest::Completes => Outcome::Completed,
            Smallest::IsRefused => Outcome::Refused,
        };
        assert_eq!(
            outcomes[0], expected,
            "{shape}: the smallest limit of the ladder"
        );
        // Small batches, as memory pressure would choose them, at the largest limit.
        assert_eq!(
            read_within(shape, path, settings, largest, Some(1_024), records),
            Outcome::Completed,
            "{shape}"
        );
    }
}

/// Requirements 13.11 and 13.12 (task 1.17): an archive whose set of distinct texts
/// does not fit the limit is read with deduplication on disk, every frame is written
/// as a Parquet part with a column of numbers beside the text (as a scoring run
/// does), and the heap stays within what the limit leaves, the encoding of the parts
/// included.
#[test]
fn deduplication_on_disk_and_writing_parts_stay_within_the_memory_limit() {
    use l2d_core::parts::{PartWriter, WorkDir};
    use l2d_core::polars::prelude::{Column, DataFrame};

    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("on-disk");
    let rows = 300_000u64;
    let short = scratch.lines("short.zip", 2 * rows, |index| {
        format!("line {}", index % rows)
    });
    let long = scratch.lines("long.zip", 2 * rows, |index| {
        let index = index % rows;
        format!(
            "2026-11-14T10:00:00Z host-{:03} sshd[{}]: Accepted publickey for user{} from \
             10.0.{}.{} port 22 ssh2",
            index % 7,
            index,
            index % 13,
            index % 250,
            index % 199
        )
    });
    let settings = ZipInputSettings::default();
    // The smallest limit of each shape is near the least it can be read with.
    let shapes = [
        (
            "300,000 short lines twice",
            &short,
            [(8, true), (16, true), (48, false)],
        ),
        (
            "300,000 long lines twice",
            &long,
            [(12, true), (18, true), (64, false)],
        ),
    ];
    for (shape, path, ladder) in shapes {
        for (usable, on_disk) in ladder {
            let usable = usable * MIB;
            let memory = MemorySettings {
                memory_limit_bytes: Some(BASELINE + usable),
                ..MemorySettings::default()
            };
            let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
            let work = WorkDir::create(&scratch.dir.join(format!("run-{usable}"))).unwrap();
            let started = std::time::Instant::now();
            let (result, growth) = measure(|| -> Result<_, L2dError> {
                let mut input = ZipInput::open(path, &settings, &budget)?;
                input.spill_into(&work.path().join("dedup"), None)?;
                let mut writer = PartWriter::new(&work, "rows")?;
                for batch in input.batches(&mut budget) {
                    let batch = batch?;
                    let size = batch.len() as u64;
                    let from = writer.rows();
                    let mut part = DataFrame::new(vec![
                        Column::new("row".into(), (from..from + size).collect::<Vec<u64>>()),
                        batch.frame().column("text").unwrap().clone(),
                        batch.frame().column("first_position").unwrap().clone(),
                    ])
                    .unwrap();
                    writer.write(&mut part)?;
                }
                let mut counted = 0u64;
                let mut total = 0u64;
                for chunk in input.occurrence_counts() {
                    let chunk = chunk?;
                    counted += chunk.len() as u64;
                    total += chunk.iter().map(|count| u64::from(*count)).sum::<u64>();
                }
                let dedup = input.disk_dedup();
                Ok((input.finish(), writer.finish(), dedup, counted, total))
            });
            let elapsed = started.elapsed();
            let (summary, table, dedup, counted, total) =
                result.unwrap_or_else(|error| panic!("{shape} at {usable}: {error}"));
            eprintln!(
                "13.12 {shape:<28} usable {:>3} MiB  heap growth {:>5.1} MiB = {:>4.2} x usable  \
                 {elapsed:.1?}  {} parts of {} bytes, on disk: {dedup:?}",
                usable / MIB,
                growth as f64 / MIB as f64,
                growth as f64 / usable as f64,
                table.parts.len(),
                table.bytes(),
            );
            assert_eq!(summary.input.records, 2 * rows, "{shape}");
            assert_eq!(summary.input.distinct_texts, rows, "{shape}");
            assert_eq!(
                (table.rows, counted, total),
                (rows, rows, 2 * rows),
                "{shape}"
            );
            assert_eq!(dedup.is_some(), on_disk, "{shape} at {usable}: {dedup:?}");
            assert!(
                growth <= usable,
                "{shape}: the heap grew by {growth} bytes under a limit that leaves {usable}"
            );
            std::fs::remove_dir_all(work.path()).unwrap();
        }
    }
}

/// The directory of an archive of 30,000 entries in memory: what opening takes here,
/// and what the `zip` crate's reader takes for the same directory. The tool does not
/// use the crate's reader (task notes, 1.16); this is the measured part of the reason.
#[test]
fn the_directory_of_many_entries_is_held_as_counted() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("directory");
    let many = scratch.archive("many-entries.zip", |writer| {
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for index in 0..30_000u32 {
            writer
                .start_file(format!("host-{}/logs/e{index}.log", index % 40), options)
                .unwrap();
            writer.write_all(b"one line\n").unwrap();
        }
    });
    let budget = MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    let before = CURRENT.load(Ordering::Relaxed);
    let (input, peak) =
        measure(|| ZipInput::open(&many, &ZipInputSettings::default(), &budget).unwrap());
    let kept = CURRENT.load(Ordering::Relaxed).saturating_sub(before) as u64;
    let counted = input.held_bytes();
    drop(input);

    let before = CURRENT.load(Ordering::Relaxed);
    let (reference, crate_peak) = measure(|| {
        let file = std::io::BufReader::new(std::fs::File::open(&many).unwrap());
        zip::ZipArchive::new(file).unwrap()
    });
    let crate_kept = CURRENT.load(Ordering::Relaxed).saturating_sub(before) as u64;
    assert_eq!(reference.len(), 30_000);
    drop(reference);

    eprintln!(
        "directory of 30,000 entries: this reader keeps {:.2} MiB (peak {:.2} MiB while it \
         opens, hash buffer included) and counts {:.2} MiB; the zip crate's reader keeps {:.2} \
         MiB (peak {:.2} MiB)",
        kept as f64 / MIB as f64,
        peak as f64 / MIB as f64,
        counted as f64 / MIB as f64,
        crate_kept as f64 / MIB as f64,
        crate_peak as f64 / MIB as f64,
    );
    assert!(
        kept <= counted && peak <= counted,
        "the opened archive keeps {kept} bytes (peak {peak}) and counts {counted}"
    );
}

/// A small generator of numbers that is the same on every run (xorshift64).
struct Seeded(u64);

impl Seeded {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound.max(1) as u64) as usize
    }
}

/// Archives with bytes changed at random, most of them in the directory and the end
/// records: whatever the directory then claims, opening and reading neither panics nor
/// takes more memory than the limit leaves. The changes are the same on every run.
#[test]
fn mutated_archives_are_read_or_refused_within_the_memory_limit() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("mutated");
    let good = scratch.archive("good.zip", |writer| {
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        let deflated = zip::write::SimpleFileOptions::default();
        writer.add_directory("logs/", stored).unwrap();
        writer.start_file("logs/a.log", stored).unwrap();
        writer.write_all(b"one\ntwo\n\nthree\n").unwrap();
        writer.start_file("logs/b.log", deflated).unwrap();
        writer
            .write_all("a line that repeats\n".repeat(300).as_bytes())
            .unwrap();
        writer
            .start_file("logs/large.log", deflated.large_file(true))
            .unwrap();
        writer.write_all(b"behind zip64 fields\n").unwrap();
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(b"inside gzip\n").unwrap();
        writer.start_file("logs/c.log.gz", stored).unwrap();
        writer.write_all(&encoder.finish().unwrap()).unwrap();
    });
    let good = std::fs::read(&good).unwrap();
    let directory_at = good
        .windows(4)
        .position(|window| window == b"PK\x01\x02")
        .unwrap();
    let defaults = ZipInputSettings::default();
    let usable = 16 * MIB;
    let memory = MemorySettings {
        memory_limit_bytes: Some(BASELINE + usable),
        ..MemorySettings::default()
    };

    // Warm up: the first read starts the thread pool, which is not the read's memory.
    let path = scratch.dir.join("mutated.zip");
    std::fs::write(&path, &good).unwrap();
    let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
    let (whole, _, _) = read_all(&path, &defaults, &mut budget).unwrap();
    assert_eq!(whole.input.records, 306);

    let mut random = Seeded(0x9e37_79b9_7f4a_7c15);
    let (mut completed, mut refused, mut invalid, mut damaged, mut largest) = (0, 0, 0, 0, 0u64);
    for round in 0..1_500 {
        let mut bytes = good.clone();
        for _ in 0..1 + random.below(4) {
            // Three of four changes fall into the directory and the end records.
            let tail = bytes.len().saturating_sub(directory_at);
            let at = if tail == 0 || random.below(4) == 0 {
                random.below(bytes.len())
            } else {
                directory_at + random.below(tail)
            };
            match random.below(5) {
                // A field of all ones is how zip64 announces itself.
                0 => {
                    let end = (at + 4).min(bytes.len());
                    bytes[at..end].fill(0xff);
                }
                1 => bytes[at] = 0,
                2 => bytes[at] ^= 1 << random.below(8),
                3 => bytes[at] = random.next() as u8,
                // Cut off, rarely: most archives should still have an end record.
                _ if random.below(8) == 0 => bytes.truncate(at.max(1)),
                _ => bytes[at] = bytes[at].wrapping_add(1),
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
        let (result, growth) = measure(|| read_all(&path, &defaults, &mut budget));
        largest = largest.max(growth);
        assert!(
            growth <= usable,
            "round {round}: the heap grew by {growth} bytes under a limit that leaves {usable}"
        );
        match result {
            Ok((summary, _, _)) => {
                assert_eq!(
                    summary
                        .entries
                        .iter()
                        .map(|entry| entry.records)
                        .sum::<u64>(),
                    summary.input.records,
                    "round {round}"
                );
                completed += 1;
                damaged += u64::from(summary.damaged().count() > 0);
            }
            Err(L2dError::Refused { .. }) => refused += 1,
            Err(L2dError::InvalidInput { .. }) => invalid += 1,
            Err(other) => panic!("round {round}: {other:?}"),
        }
    }
    eprintln!(
        "1,500 mutated archives: {completed} read ({damaged} of them with a damaged entry), \
         {invalid} not an archive or with a damaged directory, {refused} refused; largest heap \
         growth {:.2} MiB under a limit that leaves {} MiB",
        largest as f64 / MIB as f64,
        usable / MIB
    );
    assert!(completed > 100 && invalid > 100 && damaged > 20);
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

#[test]
#[ignore = "on request: reads an archive of the AIT Log Data Set (7.1 GB); run in release"]
fn the_ait_archive_is_read_within_the_memory_limit() {
    let _guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let archive = std::env::var_os("L2D_AIT_ARCHIVE").map_or_else(
        || {
            workspace_root()
                .join("data")
                .join("ait-lds-v2")
                .join("russellmitchell.zip")
        },
        PathBuf::from,
    );
    let entry_patterns = std::env::var("L2D_AIT_ENTRY").unwrap_or_else(|_| "*.log".to_string());
    let limit =
        parse_memory_size(&std::env::var("L2D_AIT_LIMIT").unwrap_or_else(|_| "512MB".to_string()))
            .unwrap();
    let batch_rows = std::env::var("L2D_AIT_BATCH_ROWS")
        .ok()
        .map(|rows| rows.parse().unwrap());
    let settings = ZipInputSettings {
        entries: entry_patterns
            .split(';')
            .filter(|pattern| !pattern.is_empty())
            .map(str::to_string)
            .collect(),
        ..ZipInputSettings::default()
    };
    let memory = MemorySettings {
        memory_limit_bytes: Some(limit),
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let usable = limit - BASELINE;
    let archive_bytes = std::fs::metadata(&archive).unwrap().len();
    eprintln!(
        "AIT archive {} ({archive_bytes} bytes), entry patterns {:?}, memory limit {limit} bytes \
         ({usable} bytes beside the tool's own share), batch rows {batch_rows:?}",
        archive.display(),
        settings.entries
    );

    // Opening: the content hash (one pass over the whole file) and the directory.
    let mut budget = MemoryBudget::observe(&memory).unwrap();
    eprintln!(
        "free physical memory at the start: {} bytes",
        budget.free_bytes()
    );
    let before = CURRENT.load(Ordering::Relaxed) as u64;
    let started = std::time::Instant::now();
    let ((mut input, opened), open_growth) = measure(|| {
        let input = ZipInput::open(&archive, &settings, &budget).unwrap();
        (input, started.elapsed())
    });
    eprintln!(
        "opened in {opened:.1?} ({:.0} MB/s over the file): heap growth {:.1} MiB, held {:.1} MiB",
        archive_bytes as f64 / 1e6 / opened.as_secs_f64(),
        open_growth as f64 / MIB as f64,
        input.held_bytes() as f64 / MIB as f64
    );

    // Reading: the peak stays counted from before the archive was opened.
    let reading = std::time::Instant::now();
    let mut rows = 0u64;
    let mut batches = 0u64;
    let mut largest = 0usize;
    let mut text_bytes = 0u64;
    for batch in input.batches(&mut budget) {
        let batch = batch.unwrap();
        rows += batch.len() as u64;
        batches += 1;
        largest = largest.max(batch.len());
        text_bytes += batch
            .frame()
            .column("text")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .map(|text| text.len() as u64)
            .sum::<u64>();
    }
    let read = reading.elapsed();
    let peak = PEAK.load(Ordering::Relaxed) as u64;
    let growth = peak.saturating_sub(before);
    let held = input.held_bytes();
    let plan = input.batch_plan().cloned().unwrap();
    let summary = input.finish();

    let mut reasons: BTreeMap<String, u64> = BTreeMap::new();
    let mut read_entries = 0u64;
    for entry in &summary.entries {
        match &entry.skipped {
            None => read_entries += 1,
            Some(EntrySkip::Damaged(reason)) => {
                eprintln!("damaged entry {}: {reason}", entry.path);
                *reasons.entry("damaged".into()).or_default() += 1;
            }
            Some(EntrySkip::UnsupportedMethod(method)) => {
                *reasons.entry(format!("method {method}")).or_default() += 1;
            }
            Some(other) => *reasons.entry(format!("{other:?}")).or_default() += 1,
        }
    }
    let mut skip_reasons: BTreeMap<&str, u64> = BTreeMap::new();
    for record in &summary.input.skipped {
        let kind = if record.reason.starts_with("record of") {
            "longer than the maximum record length"
        } else {
            record.reason.as_str()
        };
        *skip_reasons.entry(kind).or_default() += 1;
    }
    eprintln!(
        "entries: {} in the directory, {read_entries} read, not read {reasons:?}",
        summary.entries.len()
    );
    eprintln!(
        "records: {} read in {read:.1?} ({:.0} records/s), {} distinct texts ({} bytes of text) \
         in {batches} batches (largest {largest} rows), {} skipped {skip_reasons:?}",
        summary.input.records,
        summary.input.records as f64 / read.as_secs_f64(),
        summary.input.distinct_texts,
        text_bytes,
        summary.input.skipped.len(),
    );
    eprintln!("plan: {plan}");
    eprintln!(
        "memory: heap peak {:.1} MiB above the heap before opening = {:.2} x the {:.1} MiB the \
         limit leaves; held at the end {:.1} MiB; sha256 {}",
        growth as f64 / MIB as f64,
        growth as f64 / usable as f64,
        usable as f64 / MIB as f64,
        held as f64 / MIB as f64,
        summary.input.content_sha256
    );
    assert_eq!(summary.input.distinct_texts, rows);
    assert_eq!(
        summary
            .entries
            .iter()
            .map(|entry| entry.records)
            .sum::<u64>(),
        summary.input.records
    );
    assert!(read_entries > 0, "the patterns matched no readable entry");
    assert!(
        growth <= usable,
        "the heap grew by {growth} bytes under a limit that leaves {usable} bytes"
    );
}
