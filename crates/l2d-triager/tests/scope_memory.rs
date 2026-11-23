//! Triager task 1.1, requirements 1.2 and 6.2 measured: opening a scope over an archive
//! or a store and joining keys to it stays within the memory limit and within what the
//! budget admitted.
//!
//! As `l2d-core`'s `marks_memory.rs`, this test binary counts every heap allocation of
//! the process (a counting global allocator around the system allocator), reader
//! threads included; for a store SQLite's own counter is added, since SQLite allocates
//! through the C runtime. For each source a ladder of memory limits is tried. Whether a
//! run completes or is refused, the heaps may not have grown by more than the limit
//! leaves beside the share kept for the process, nor by more than the peak the budget
//! counted as admitted. A completed run gives the same scope and the same join at every
//! limit, batch size and index placement, and what the scope keeps in memory is at most
//! what it reports as held.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_core::{
    ArchivePosition, ArchiveSummary, BatchPlan, EncoderIdentity, InputAdapter, InputSummary,
    L2dError, Label, LinesAdapter, LogRecord, LogsFrame, MemoryBudget, MemorySettings,
    ProjectPaths, SkippedRecord, Source, ZipInputSettings,
};
use l2d_store::{LogFlags, Store, VectorOrigin, log_id, rows_frame};
use l2d_triager::scope::{
    IndexPlacement, NativeKey, Scope, ScopeSettings, archive_summary_bytes, input_summary_bytes,
};
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

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const BASELINE: u64 = l2d_core::memory::PROCESS_BASELINE_BYTES;

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

/// One measured run: what it gave, or that it was refused.
#[derive(Debug, PartialEq, Eq, Clone)]
enum Outcome {
    /// The handle, the originals, the occurrences, the digest of the join, and whether
    /// the index stayed on disk.
    Completed(String, u64, u64, String, bool),
    Refused,
}

impl Outcome {
    /// The originals, the occurrences and whether the index is on disk.
    fn sizes(&self) -> Option<(u64, u64, bool)> {
        match self {
            Outcome::Completed(_, originals, occurrences, _, on_disk) => {
                Some((*originals, *occurrences, *on_disk))
            }
            Outcome::Refused => None,
        }
    }

    /// The handle and the digest of the join.
    fn join(&self) -> Option<(&str, &str)> {
        match self {
            Outcome::Completed(handle, _, _, join, _) => Some((handle, join)),
            Outcome::Refused => None,
        }
    }
}

struct Run<'a> {
    shape: &'a str,
    usable: u64,
    batch_rows: Option<u64>,
    index: IndexPlacement,
    /// For an archive: deduplication goes on on disk after this many texts.
    dedup_set_texts: Option<u64>,
    work: PathBuf,
}

/// The reader's summary an open returns. It is the caller's to hold, so it stays alive
/// through the measurement.
enum Kept {
    Nothing,
    Input(InputSummary),
    Archive(ArchiveSummary),
}

impl Kept {
    fn bytes(&self) -> u64 {
        match self {
            Kept::Nothing => 0,
            Kept::Input(summary) => input_summary_bytes(summary),
            Kept::Archive(summary) => archive_summary_bytes(summary),
        }
    }
}

/// Opens a scope with `open`, joins `keys` to it, and checks the heaps against the
/// limit, the admitted peak and what the scope and the summary are said to hold.
fn measured<'k>(
    run: &Run<'_>,
    open: impl FnOnce(&ScopeSettings, &mut MemoryBudget) -> Result<(Scope, Kept), L2dError>,
    keys: impl Iterator<Item = NativeKey<'k>>,
) -> Outcome {
    let settings = MemorySettings {
        memory_limit_bytes: Some(BASELINE + run.usable),
        max_batch_rows: run.batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap();
    let scope_settings = ScopeSettings {
        index: run.index,
        dedup_set_texts: run.dedup_set_texts,
        ..ScopeSettings::new(&run.work)
    };
    let (sqlite_before, _) = sqlite_now_and_peak();
    let rust_before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(rust_before, Ordering::Relaxed);
    let started = std::time::Instant::now();
    let result = open(&scope_settings, &mut budget).and_then(|(scope, summary)| {
        // What stays in memory once the scope is open is what the scope reports as
        // held and what the summary takes.
        let kept = (CURRENT.load(Ordering::Relaxed).saturating_sub(rust_before)) as u64;
        let held = scope.held_bytes() + summary.bytes();
        assert!(
            kept <= held,
            "{}: the open scope and its summary keep {kept} bytes, and {} + {} are held",
            run.shape,
            scope.held_bytes(),
            summary.bytes()
        );
        assert_eq!(scope.index_on_disk(), scope.disk_bytes() > 0);
        budget.hold(held);
        let mut digest = Sha256::new();
        let report = scope
            .join_keys(scope.source(), keys, &mut budget, |index, original| {
                digest.update(index.to_le_bytes());
                digest.update(original.id.0.to_le_bytes());
                digest.update(original.occurrences.to_le_bytes());
                digest.update(original.native_id.to_hex().as_bytes());
                Ok(())
            })
            .map_err(L2dError::from)?;
        assert_eq!(report.keys, report.originals);
        assert_eq!(budget.held_bytes(), held);
        budget.hold(0);
        drop(summary);
        Ok(Outcome::Completed(
            scope.handle().to_string(),
            scope.originals(),
            scope.occurrences(),
            format!("{:x}", digest.finalize()),
            scope.index_on_disk(),
        ))
    });
    let elapsed = started.elapsed();
    let rust = (PEAK.load(Ordering::Relaxed).saturating_sub(rust_before)) as u64;
    let (_, sqlite_peak) = sqlite_now_and_peak();
    let sqlite = sqlite_peak.saturating_sub(sqlite_before);
    let admitted = budget
        .usage()
        .report()
        .counted_peak_bytes
        .saturating_sub(BASELINE);
    let (outcome, detail) = match result {
        Ok(outcome) => {
            let detail = match &outcome {
                Outcome::Completed(_, originals, occurrences, _, on_disk) => format!(
                    "{originals} originals, {occurrences} occurrences, index {}",
                    if *on_disk { "on disk" } else { "in memory" }
                ),
                Outcome::Refused => unreachable!(),
            };
            (outcome, detail)
        }
        Err(L2dError::Refused { reason }) => (Outcome::Refused, reason),
        Err(other) => panic!("{}: {other:?}", run.shape),
    };
    let growth = rust + sqlite;
    eprintln!(
        "scope {:<8} batch {:>6}  {:<18}  usable {:>3} MiB  rust {:>5.2} MiB + sqlite {:>5.2} MiB = {:>4.2} x usable, {:>4.2} x admitted ({:>5.2} MiB)  {elapsed:.1?}  {detail}",
        run.shape,
        run.batch_rows
            .map_or("auto".to_string(), |rows| rows.to_string()),
        format!("{:?}", run.index),
        run.usable / MIB,
        rust as f64 / MIB as f64,
        sqlite as f64 / MIB as f64,
        growth as f64 / run.usable as f64,
        growth as f64 / admitted.max(1) as f64,
        admitted as f64 / MIB as f64,
    );
    assert!(
        growth <= run.usable,
        "{}: the heaps grew by {rust} + {sqlite} bytes under a limit that leaves {} bytes",
        run.shape,
        run.usable
    );
    assert!(
        growth <= admitted,
        "{}: the heaps grew by {rust} + {sqlite} bytes, and the budget admitted a peak of \
         {admitted} bytes",
        run.shape
    );
    assert_eq!(budget.held_bytes(), 0, "{}: nothing stays held", run.shape);
    assert!(
        !run.work.exists(),
        "{}: the work directory is removed with the scope",
        run.shape
    );
    outcome
}

/// Runs the ladder and checks that every completed run gave the same scope and join.
fn ladder(
    shape: &str,
    dir: &Path,
    limits: &[u64],
    mut one: impl FnMut(&Run<'_>) -> Outcome,
) -> Vec<(IndexPlacement, u64, Outcome)> {
    let mut outcomes = Vec::new();
    let mut expected: Option<(String, u64, u64, String)> = None;
    for index in [IndexPlacement::MemoryWhenAdmitted, IndexPlacement::Disk] {
        for batch_rows in [None, Some(1_024)] {
            for &usable in limits {
                let run = Run {
                    shape,
                    usable,
                    batch_rows,
                    index,
                    dedup_set_texts: None,
                    work: dir.join(format!("work-{}", outcomes.len())),
                };
                let outcome = one(&run);
                if let Outcome::Completed(handle, originals, occurrences, join, _) = &outcome {
                    let found = (handle.clone(), *originals, *occurrences, join.clone());
                    match &expected {
                        None => expected = Some(found),
                        Some(expected) => assert_eq!(
                            &found, expected,
                            "{shape}: the same scope and join at {usable} bytes, batch \
                             {batch_rows:?}, {index:?}"
                        ),
                    }
                }
                outcomes.push((index, usable, outcome));
            }
        }
    }
    assert_eq!(
        outcomes.first().map(|run| &run.2),
        Some(&Outcome::Refused),
        "{shape}: the smallest limit is refused"
    );
    assert!(
        matches!(outcomes.last(), Some((_, _, Outcome::Completed(..)))),
        "{shape}: the largest limit completes"
    );
    outcomes
}

/// The measurements share the counters: one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// One open before the measurements: what the process sets up once and keeps (the data
/// layer's thread pool, the lease records) is no part of a scope.
fn warm_up(dir: &Path) {
    let path = dir.join("warm.log");
    std::fs::write(&path, "warm\n").unwrap();
    let mut budget =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    let settings = ScopeSettings::new(dir.join("warm"));
    Scope::open_file(&path, &LinesAdapter, &settings, &mut budget).unwrap();
}

/// Review round 1 of task 1.1: a summary with many skipped records is alive while the
/// index is placed, so it has to be charged then. Each limit is chosen so that the
/// index alone fits what the limit leaves, and the index with the summary does not.
#[test]
fn an_archive_summary_with_many_skipped_records_is_charged_while_the_index_is_placed() {
    let _serial = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().unwrap();
    warm_up(dir.path());
    let archived = 500_000u64;
    let run = |shape: &'static str, usable: u64, work: &str| Run {
        shape,
        usable: usable * MIB,
        batch_rows: None,
        index: IndexPlacement::MemoryWhenAdmitted,
        // The reader counts its set of distinct texts against the limit as well; on
        // disk it leaves the room in which the index is placed.
        dedup_set_texts: Some(1_000),
        work: dir.path().join(work),
    };

    // An archive: 500,000 distinct lines and 200,000 blank lines, each a skipped
    // record of the summary.
    let archive = dir.path().join("blank.zip");
    {
        let mut writer = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        writer
            .start_file("only.log", zip::write::SimpleFileOptions::default())
            .unwrap();
        for index in 0..archived {
            writeln!(writer, "request {index} served").unwrap();
        }
        writer.write_all("\n".repeat(200_000).as_bytes()).unwrap();
        writer.finish().unwrap();
    }
    let zip = ZipInputSettings::default();
    let open_archive = |usable: u64, work: &str| {
        measured(
            &run("blank zip", usable, work),
            |settings, budget| {
                let (scope, summary) = Scope::open_archive(&archive, &zip, settings, budget)?;
                assert_eq!(summary.input.skipped.len(), 200_000);
                Ok((scope, Kept::Archive(summary)))
            },
            (1..=archived)
                .step_by(7)
                .map(|line| NativeKey::Position(ArchivePosition { entry: 0, line }.pack())),
        )
    };
    let tight = open_archive(ARCHIVE_TIGHT_MIB, "zip-tight");
    let wide = open_archive(ARCHIVE_WIDE_MIB, "zip-wide");
    assert_eq!(tight.sizes(), Some((archived, archived, true)));
    assert_eq!(wide.sizes(), Some((archived, archived, false)));
    assert_eq!(tight.join(), wide.join());
}

/// As the archive above, for a plain file.
///
/// The line adapter counts a skipped record far above what it takes in memory: the
/// same shape through `LinesAdapter` (300,000 short lines and 1,000,000 blank ones) is
/// refused by the adapter itself at 64 and at 128 MiB, long before the summary and the
/// index could meet. So that the scope's own charge is what is measured, the rows and
/// the summary of such a read are handed out by an adapter that charges nothing; the
/// summary is copied inside the measurement, the rows share their buffers.
#[test]
fn a_file_summary_with_many_skipped_records_is_charged_while_the_index_is_placed() {
    struct Handed {
        logs: LogsFrame,
        summary: InputSummary,
    }
    impl InputAdapter for Handed {
        fn read_from(
            &self,
            _: Source,
            budget: &mut MemoryBudget,
        ) -> Result<(LogsFrame, InputSummary, BatchPlan), L2dError> {
            let plan = budget.plan(self.logs.len() as u64, 64)?;
            Ok((self.logs.clone(), self.summary.clone(), plan))
        }
    }

    let _serial = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().unwrap();
    warm_up(dir.path());
    let (originals, skipped) = (300_000u64, 500_000u64);
    let run = |shape: &'static str, usable: u64, work: &str| Run {
        shape,
        usable: usable * MIB,
        batch_rows: None,
        index: IndexPlacement::MemoryWhenAdmitted,
        dedup_set_texts: None,
        work: dir.path().join(work),
    };
    let plain = dir.path().join("blank.log");
    std::fs::write(&plain, "the rows are handed out\n").unwrap();
    let records: Vec<LogRecord> = (0..originals)
        .map(|index| LogRecord {
            text: format!("r{index}"),
            label: Label::Unknown,
            group: None,
            first_position: index + 1,
            occurrences: 1,
        })
        .collect();
    let handed = Handed {
        logs: LogsFrame::from_records(&records).unwrap(),
        summary: InputSummary {
            source_path: plain.clone(),
            records: originals + skipped,
            distinct_texts: originals,
            content_sha256: "a".repeat(64),
            skipped: (0..skipped)
                .map(|index| SkippedRecord {
                    position: originals + index + 1,
                    reason: "empty line".into(),
                })
                .collect(),
            label_conflicts: 0,
        },
    };
    drop(records);
    let open_file = |usable: u64, work: &str| {
        measured(
            &run("blank file", usable, work),
            |settings, budget| {
                let (scope, summary) = Scope::open_file(&plain, &handed, settings, budget)?;
                assert_eq!(summary.skipped.len() as u64, skipped);
                Ok((scope, Kept::Input(summary)))
            },
            (1..=originals).step_by(7).map(NativeKey::Position),
        )
    };
    let tight = open_file(FILE_TIGHT_MIB, "file-tight");
    let wide = open_file(FILE_WIDE_MIB, "file-wide");
    assert_eq!(tight.sizes(), Some((originals, originals, true)));
    assert_eq!(wide.sizes(), Some((originals, originals, false)));
    assert_eq!(tight.join(), wide.join());
}

/// The limits of the shapes with many skipped records, in MiB beside the baseline. At
/// the tight one the index (39 MiB for 500,000 originals, 30 MiB for 300,000) fits
/// beside the fixed share and does not fit beside the summary as well; at the wide
/// one both fit.
const ARCHIVE_TIGHT_MIB: u64 = 44;
const ARCHIVE_WIDE_MIB: u64 = 64;
const FILE_TIGHT_MIB: u64 = 40;
const FILE_WIDE_MIB: u64 = 96;

#[test]
fn scopes_and_joins_stay_within_the_memory_limit_and_the_admitted_estimate() {
    let _serial = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().unwrap();

    // An archive: 140,000 distinct lines, and 20,000 of them again in a second entry.
    // Its relation takes more memory than the archive read needs at the least, so a
    // limit exists that reads the archive and leaves the relation on disk.
    let line = |index: u64| {
        format!(
            "203.0.113.{} GET /shop/item/{index}/detail?session={} HTTP/1.1 200\n",
            index % 250,
            index * 7919
        )
    };
    let archive = dir.path().join("logs.zip");
    {
        let mut writer = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("first.log", options).unwrap();
        for index in 0..140_000 {
            writer.write_all(line(index).as_bytes()).unwrap();
        }
        writer.start_file("second.log", options).unwrap();
        for index in 0..20_000 {
            writer.write_all(line(index * 5).as_bytes()).unwrap();
        }
        writer.finish().unwrap();
    }
    let zip = ZipInputSettings::default();
    // What the process sets up once and keeps (the data layer's thread pool, the
    // lease records) is no part of a scope: one open before the measurements.
    {
        let mut budget =
            MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
        let warm = ScopeSettings::new(dir.path().join("warm"));
        Scope::open_archive(&archive, &zip, &warm, &mut budget).unwrap();
    }
    let outcomes = ladder(
        "archive",
        dir.path(),
        &[2, 10, 12, 16, 64].map(|mib| mib * MIB),
        |run| {
            measured(
                run,
                |settings, budget| {
                    Scope::open_archive(&archive, &zip, settings, budget)
                        .map(|(scope, summary)| (scope, Kept::Archive(summary)))
                },
                (1..=140_000u64)
                    .step_by(3)
                    .map(|line| NativeKey::Position(ArchivePosition { entry: 0, line }.pack())),
            )
        },
    );
    let completed = |on_disk: bool, index: IndexPlacement| {
        outcomes.iter().any(|(placed, _, outcome)| {
            *placed == index && outcome.sizes() == Some((140_000, 160_000, on_disk))
        })
    };
    assert!(
        completed(false, IndexPlacement::MemoryWhenAdmitted),
        "a wide limit keeps the index in memory"
    );
    assert!(
        completed(true, IndexPlacement::MemoryWhenAdmitted),
        "a tight limit that still reads the archive leaves the index on disk"
    );
    assert!(completed(true, IndexPlacement::Disk));

    // A store of 30,000 logs.
    let rows = 30_000u64;
    let text = |index: u64| line(index).trim_end().to_string();
    let records: Vec<LogRecord> = (0..rows)
        .map(|index| LogRecord {
            text: text(index),
            label: Label::Unknown,
            group: None,
            first_position: index + 1,
            occurrences: 1 + (index % 3) as u32,
        })
        .collect();
    let logs = LogsFrame::from_records(&records).unwrap();
    drop(records);
    let vectors: Vec<f32> = (0..rows).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let flags = vec![LogFlags::default(); rows as usize];
    let frame = rows_frame(&logs, "synthetic", &vectors, 4, &flags).unwrap();
    let identity = EncoderIdentity {
        name: "bytecnn".to_string(),
        revision: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        dimension: 4,
    };
    let origin = VectorOrigin::without_release(&identity);
    let paths = ProjectPaths::new(dir.path().join("project"));
    {
        let mut store = Store::create(&paths, "s", origin).unwrap();
        let mut budget =
            MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
        store.add_with(&frame, origin, &mut budget).unwrap();
    }
    drop((frame, logs, vectors, flags));
    let store = Store::open(&paths, "s").unwrap();
    let ids: Vec<String> = (0..rows)
        .step_by(3)
        .map(|index| log_id("synthetic", &text(index)))
        .collect();
    let outcomes = ladder(
        "store",
        dir.path(),
        &[2, 8, 16, 24, 32, 64].map(|mib| mib * MIB),
        |run| {
            measured(
                run,
                |settings, budget| {
                    Scope::open_store(&store, settings, budget).map(|scope| (scope, Kept::Nothing))
                },
                ids.iter().map(|id| NativeKey::StoreId(id)),
            )
        },
    );
    let expected_occurrences: u64 = (0..rows).map(|index| 1 + index % 3).sum();
    assert!(outcomes.iter().any(|(index, _, outcome)| {
        *index == IndexPlacement::Disk
            && outcome.sizes() == Some((30_000, expected_occurrences, true))
    }));
    assert!(outcomes.iter().any(|(index, _, outcome)| {
        *index == IndexPlacement::MemoryWhenAdmitted
            && outcome.sizes() == Some((30_000, expected_occurrences, false))
    }));

    // A plain file of 30,000 lines: its deduplicated rows are in memory as a whole
    // while the relation is written, as in every plain-file read.
    let plain = dir.path().join("plain.log");
    std::fs::write(&plain, (0..rows).map(line).collect::<String>()).unwrap();
    let outcomes = ladder(
        "file",
        dir.path(),
        &[2, 32, 64].map(|mib| mib * MIB),
        |run| {
            measured(
                run,
                |settings, budget| {
                    Scope::open_file(&plain, &LinesAdapter, settings, budget)
                        .map(|(scope, summary)| (scope, Kept::Input(summary)))
                },
                (1..=rows).step_by(3).map(NativeKey::Position),
            )
        },
    );
    for on_disk in [false, true] {
        assert!(
            outcomes
                .iter()
                .any(|(_, _, outcome)| outcome.sizes() == Some((rows, rows, on_disk)))
        );
    }
}
