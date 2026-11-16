//! Requirement 16.12 measured (with 12.9 and 15.2): fitting a score from a mark set over
//! an archive and over a store stays within its memory limit (task 1.23, review round 1).
//!
//! Like `redact_memory.rs`, this test binary counts every heap allocation of the process
//! (a counting global allocator around the system allocator). For a ladder of memory
//! limits the score is fitted; whether it completes or is refused, the heap may not have
//! grown by more than the limit less the share the tool keeps for the process itself
//! (`PROCESS_BASELINE_BYTES`). Every batch, the copies the fit takes of the logs, the logs
//! frame of the fit, the fit itself and the score returned are inside the measurement;
//! the mark set is made before it, under a roomy budget.
//!
//! The store is a test double that cuts its batches as `l2d-store` does
//! (`Store::text_batches_with`): a batch takes rows while its frame, charged at twice the
//! text and the id, a fixed share per row and the caller's `per_row`, fits beside what the
//! caller holds; it is refused with the figures when not even one row fits. Its charge is
//! close to what its frames really take, so what the caller copies of a batch and did not
//! ask room for shows in the heap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_baseline::BaselineSettings;
use l2d_baseline::markscore::{MarkFitSettings, fit_from_marks};
use l2d_baseline::redact::RedactionInput;
use l2d_core::marks::{MarkQuery, MarkSet, MarkStore};
use l2d_core::memory::exact;
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::{L2dError, Label, MemoryBudget, MemorySettings, ZipInputSettings};
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
const BASELINE: u64 = l2d_core::memory::PROCESS_BASELINE_BYTES;
const LOGS: u64 = 12_000;

/// Runs `run` and returns its result and by how many bytes the heap grew at its peak.
fn measure<T>(run: impl FnOnce() -> T) -> (T, u64) {
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = run();
    let peak = PEAK.load(Ordering::Relaxed);
    (result, peak.saturating_sub(before) as u64)
}

/// The log of line `i`: about 200 bytes, distinct, a third of them with the flag the
/// analyst marks by.
fn line(i: u64) -> String {
    format!(
        "GET http://h{}.example/p{}/x.jsp?a={i}&b=z{}&m={}&pad={}",
        i % 50,
        i % 20,
        (i * 31) % 99_991,
        (i / 7) % 3,
        "q".repeat(120 + (i % 40) as usize)
    )
}

struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "l2d-baseline-markscore-memory-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    /// An archive with the lines in two deflated entries of two hosts.
    fn archive(&self) -> PathBuf {
        let path = self.dir.join("testbed.zip");
        let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut writer = zip::ZipWriter::new(file);
        for (host, part) in [("web", 0), ("mail", 1)] {
            writer
                .start_file(
                    format!("gather/{host}/logs/all.log"),
                    zip::write::SimpleFileOptions::default(),
                )
                .unwrap();
            for index in (0..LOGS).filter(|index| index % 2 == part) {
                writer.write_all(line(index).as_bytes()).unwrap();
                writer.write_all(b"\n").unwrap();
            }
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

/// What the double charges for its frame: once per batch, and per row beside twice the
/// text and the id.
const DOUBLE_FIXED_BYTES: u64 = 64 * 1024;
const DOUBLE_ROW_BYTES: u64 = 64;

/// A store of the lines that cuts its batches to the budget, as `l2d-store` does.
struct BudgetStore {
    rows: Vec<(String, String)>,
    hash: String,
}

impl BudgetStore {
    fn new() -> Self {
        let rows: Vec<(String, String)> = (0..LOGS)
            .map(|index| {
                let text = line(index);
                (format!("{:x}", Sha256::digest(text.as_bytes())), text)
            })
            .collect();
        let mut ids: Vec<&str> = rows.iter().map(|(id, _)| id.as_str()).collect();
        ids.sort_unstable();
        let hash = format!("{:x}", Sha256::digest(ids.concat().as_bytes()));
        Self { rows, hash }
    }
}

impl MarkStore for BudgetStore {
    fn store_name(&self) -> &str {
        "budget"
    }

    fn content_sha256(&self, _budget: &mut MemoryBudget) -> Result<String, L2dError> {
        Ok(self.hash.clone())
    }

    fn phrase_ids(
        &self,
        phrase: &str,
        _id_bytes: u64,
        _budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&str),
    ) -> Result<(), L2dError> {
        for (id, text) in &self.rows {
            if text.contains(phrase) {
                each(id);
            }
        }
        Ok(())
    }

    fn text_batches(
        &self,
        beside: u64,
        per_row: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        let mut beside = beside;
        let mut next = 0;
        while next < self.rows.len() {
            let room = budget.left_bytes().saturating_sub(beside);
            let mut charged = DOUBLE_FIXED_BYTES;
            let mut end = next;
            while end < self.rows.len() && (end - next) < budget.max_batch_rows() as usize {
                let (id, text) = &self.rows[end];
                let row = DOUBLE_ROW_BYTES + 2 * (text.len() + id.len()) as u64 + per_row;
                if charged + row > room {
                    break;
                }
                charged += row;
                end += 1;
            }
            if end == next {
                return Err(L2dError::Refused {
                    reason: format!(
                        "reading the store in batches needs {} for one row beside the {} the \
                         caller holds, but {} is left",
                        exact(DOUBLE_FIXED_BYTES + DOUBLE_ROW_BYTES + per_row + 300),
                        exact(beside),
                        exact(budget.left_bytes())
                    ),
                });
            }
            let chunk = &self.rows[next..end];
            let frame = DataFrame::new(vec![
                Column::new(
                    "seq".into(),
                    (next..end).map(|row| row as i64).collect::<Vec<_>>(),
                ),
                Column::new(
                    "id".into(),
                    chunk.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
                ),
                Column::new(
                    "text".into(),
                    chunk
                        .iter()
                        .map(|(_, text)| text.as_str())
                        .collect::<Vec<_>>(),
                ),
            ])
            .unwrap();
            beside = each(&frame)?;
            next = end;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    Completed,
    Refused,
}

fn settings() -> MarkFitSettings {
    MarkFitSettings {
        baseline: BaselineSettings {
            buckets: 1024,
            ngram_max: 2,
            ..BaselineSettings::default()
        },
        group_key: None,
    }
}

/// Fits a score from `set` over `input` under a limit that leaves `usable` bytes beside
/// the baseline, and checks the heap growth against it.
fn fit_within(
    what: &str,
    set: &MarkSet,
    input: impl Fn() -> RedactionInput<'static>,
    usable: u64,
    batch_rows: Option<u64>,
) -> Outcome {
    let limit = BASELINE + usable;
    let memory = MemorySettings {
        memory_limit_bytes: Some(limit),
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
    let started = std::time::Instant::now();
    let (result, growth) = measure(|| fit_from_marks(set, None, input(), &settings(), &mut budget));
    let elapsed = started.elapsed();
    let (outcome, detail) = match &result {
        Ok(fit) => (
            Outcome::Completed,
            format!(
                "{} marked, {} other, {} rows fitted",
                fit.counts.marked,
                fit.counts.other,
                fit.scorer.fit_report().rows
            ),
        ),
        Err(L2dError::Refused { reason }) => (Outcome::Refused, reason.clone()),
        Err(other) => panic!("{other:?}"),
    };
    eprintln!(
        "16.12 fit {what:<7} batch {:>6}  usable {:>3} MiB  {:<9}  heap growth {:>6.2} MiB = {:>4.2} x usable  {elapsed:.1?}  {detail}",
        batch_rows.map_or("auto".to_string(), |rows| rows.to_string()),
        usable / MIB,
        format!("{outcome:?}"),
        growth as f64 / MIB as f64,
        growth as f64 / usable as f64,
    );
    assert!(
        growth <= usable,
        "fitting from {what}: the heap grew by {growth} bytes under a limit of {limit} bytes \
         that leaves {usable} bytes beside the baseline ({outcome:?}: {detail})"
    );
    assert_eq!(budget.held_bytes(), 0, "{what}: the budget is given back");
    drop(result);
    outcome
}

const LADDER: [u64; 6] = [2 * MIB, 4 * MIB, 6 * MIB, 8 * MIB, 16 * MIB, 64 * MIB];

#[test]
fn fitting_from_marks_over_an_archive_stays_within_its_memory_limit() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("archive");
    let path: &'static Path = Box::leak(scratch.archive().into_boxed_path());
    let zip: &'static ZipInputSettings = Box::leak(Box::new(ZipInputSettings::default()));
    let mut roomy =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    let (set, _) = MarkSet::mark_archive(
        "flagged",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "&m=0&".into(),
        },
        path,
        zip,
        &mut roomy,
    )
    .unwrap();
    assert_eq!(set.count(), 4_004);
    let mut outcomes = Vec::new();
    for usable in LADDER {
        for batch_rows in [None, Some(1_024)] {
            outcomes.push((
                usable,
                fit_within(
                    "archive",
                    &set,
                    || RedactionInput::Archive {
                        path,
                        settings: zip,
                    },
                    usable,
                    batch_rows,
                ),
            ));
        }
    }
    let of = |usable| {
        outcomes
            .iter()
            .filter(|(u, _)| *u == usable)
            .map(|(_, outcome)| *outcome)
            .collect::<Vec<_>>()
    };
    assert_eq!(of(64 * MIB), [Outcome::Completed; 2]);
    assert_eq!(of(2 * MIB), [Outcome::Refused; 2]);

    // The archive's batches are charged far above what they take (6 x the text and 500
    // bytes a row), so copies taken without room do not show in the heap on this ladder.
    // What shows is where the fit is refused: with the room for the copies of the first
    // batch kept beside the archive, a limit that leaves no room for a batch beside them
    // is refused before the first batch is read, not after reading it.
    let memory = MemorySettings {
        memory_limit_bytes: Some(BASELINE + 6 * MIB),
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
    let reason = match fit_from_marks(
        &set,
        None,
        RedactionInput::Archive {
            path,
            settings: zip,
        },
        &settings(),
        &mut budget,
    ) {
        Err(L2dError::Refused { reason }) => reason,
        other => panic!("{other:?}"),
    };
    assert!(reason.contains("before the first batch"), "{reason}");
    assert!(reason.contains("room it keeps"), "{reason}");
}

#[test]
fn fitting_from_marks_over_a_store_stays_within_its_memory_limit() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let store: &'static BudgetStore = Box::leak(Box::new(BudgetStore::new()));
    let mut roomy =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    let set = MarkSet::mark_store(
        "flagged",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "&m=0&".into(),
        },
        store,
        &mut roomy,
    )
    .unwrap();
    assert_eq!(set.count(), 4_004);
    let mut outcomes = Vec::new();
    for usable in LADDER {
        for batch_rows in [None, Some(1_024)] {
            outcomes.push((
                usable,
                fit_within(
                    "store",
                    &set,
                    || RedactionInput::Store(store),
                    usable,
                    batch_rows,
                ),
            ));
        }
    }
    let of = |usable| {
        outcomes
            .iter()
            .filter(|(u, _)| *u == usable)
            .map(|(_, outcome)| *outcome)
            .collect::<Vec<_>>()
    };
    assert_eq!(of(64 * MIB), [Outcome::Completed; 2]);
    assert_eq!(of(2 * MIB), [Outcome::Refused; 2]);
}

// ---------------------------------------------------------------------------------------
// Review round 2: the reader sizes batch j with what the fit told it after batch j - 2
// ---------------------------------------------------------------------------------------

/// The host of [`short_lines`]: a long directory name, so that each copy (its record,
/// a short text and the host as its group key) comes close to what a batch row is
/// charged by the reader (500 bytes and 6 times the text).
fn long_host() -> String {
    format!("host-{}", "h".repeat(190))
}

/// An archive of `rows` short distinct lines in one entry of [`long_host`].
fn short_lines(scratch: &Scratch, rows: u64) -> PathBuf {
    let path = scratch.dir.join("short.zip");
    let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    let mut writer = zip::ZipWriter::new(file);
    writer
        .start_file(
            format!("gather/{}/logs/all.log", long_host()),
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
    for index in 0..rows {
        writeln!(writer, "l{index:07}").unwrap();
    }
    writer.finish().unwrap().flush().unwrap();
    path
}

fn fit_archive(
    set: &MarkSet,
    path: &Path,
    usable: u64,
    batch_rows: Option<u64>,
) -> (Result<l2d_baseline::MarkFit, L2dError>, u64) {
    let memory = MemorySettings {
        memory_limit_bytes: Some(BASELINE + usable),
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&memory, || Ok(64 * GIB)).unwrap();
    let zip = ZipInputSettings::default();
    let (result, growth) = measure(|| {
        fit_from_marks(
            set,
            None,
            RedactionInput::Archive {
                path,
                settings: &zip,
            },
            &settings(),
            &mut budget,
        )
    });
    assert_eq!(budget.held_bytes(), 0);
    (result, growth)
}

#[test]
fn the_copies_of_every_archive_batch_fit_the_room_kept_when_it_was_sized() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Batches cut by their size: the copies of two batches in a row must fit the room
    // the fit kept when the first of them was sized. The check in the fit enforces it;
    // this test guards the combined accounting (host key and the reader's one-batch
    // lag). On this data room for one batch alone is not caught, because the bound on
    // a batch's copies has slack (task 1.23 notes, review round 2).
    let scratch = Scratch::new("lag");
    let path = short_lines(&scratch, 120_000);
    let mut roomy =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    let (set, _) = MarkSet::mark_archive(
        "early",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "l00".into(),
        },
        &path,
        &ZipInputSettings::default(),
        &mut roomy,
    )
    .unwrap();
    assert_eq!(set.count(), 100_000);
    let mut ladder_outcomes = Vec::new();
    for usable in [24 * MIB, 32 * MIB, 48 * MIB, 64 * MIB, 96 * MIB, 128 * MIB] {
        let started = std::time::Instant::now();
        let (result, growth) = fit_archive(&set, &path, usable, None);
        let detail = match &result {
            Ok(fit) => format!(
                "Completed, {} batches: {:?}",
                fit.archive_batches.len(),
                fit.archive_batches
            ),
            Err(L2dError::Refused { reason }) => format!("Refused: {reason}"),
            Err(other) => panic!("{other:?}"),
        };
        eprintln!(
            "16.12 lag usable {:>3} MiB  heap growth {:>6.2} MiB = {:.2} x usable  {:.1?}  {detail}",
            usable / MIB,
            growth as f64 / MIB as f64,
            growth as f64 / usable as f64,
            started.elapsed()
        );
        assert!(growth <= usable, "{growth} > {usable}: {detail}");
        assert!(!detail.contains("was too small"), "{detail}");
        if let Ok(fit) = &result {
            for batch in &fit.archive_batches {
                assert!(batch.held_after <= batch.reserved, "{batch:?}");
            }
            let batches = &fit.archive_batches;
            let cut_by_size = batches[..batches.len() - 1]
                .iter()
                .any(|batch| batch.rows < 65_536);
            ladder_outcomes.push((usable, batches.len(), cut_by_size));
        }
    }
    // The largest rung completes; a rung completes after reading at least three batches
    // cut by their size (fewer rows than the largest batch of 65,536), where the lag
    // between sizing a batch and the figure reaching the reader matters.
    assert!(
        ladder_outcomes
            .iter()
            .any(|(usable, batches, _)| *usable == 128 * MIB && *batches >= 2),
        "{ladder_outcomes:?}"
    );
    assert!(
        ladder_outcomes
            .iter()
            .any(|(_, batches, cut_by_size)| *batches >= 3 && *cut_by_size),
        "{ladder_outcomes:?}"
    );
}

#[test]
fn an_archive_read_in_many_batches_reports_each_batch_within_its_room() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("batches");
    let path = scratch.archive();
    let mut roomy =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 * GIB)).unwrap();
    let (set, _) = MarkSet::mark_archive(
        "flagged",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "&m=0&".into(),
        },
        &path,
        &ZipInputSettings::default(),
        &mut roomy,
    )
    .unwrap();
    let (result, growth) = fit_archive(&set, &path, 64 * MIB, Some(2_048));
    let fit = result.unwrap();
    eprintln!(
        "16.12 batches: heap growth {:.2} MiB; {:?}",
        growth as f64 / MIB as f64,
        fit.archive_batches
    );
    assert!(growth <= 64 * MIB);
    assert!(fit.archive_batches.len() >= 3, "{:?}", fit.archive_batches);
    assert_eq!(
        fit.archive_batches
            .iter()
            .map(|batch| batch.rows)
            .sum::<u64>(),
        LOGS
    );
    for (index, batch) in fit.archive_batches.iter().enumerate() {
        assert_eq!(batch.batch, index as u64 + 1);
        assert!(batch.held_after <= batch.reserved, "{batch:?}");
    }
    // Batches 1 and 2 were sized with the figure given before the first batch.
    assert_eq!(
        fit.archive_batches[0].reserved,
        fit.archive_batches[1].reserved
    );
}
