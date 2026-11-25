//! The operations that make an artifact: opening a snapshot, selecting a set by a
//! score, a filter or a registered membership, and composing a view. They run on the
//! service's worker, one at a time, against the one memory budget.
//!
//! # Memory
//!
//! Every operation asks the budget for all it holds before it allocates, as a constant
//! share that is held for the whole operation, so the archive's batches are sized with
//! it from the first one on and no share grows between batches:
//!
//! - a selection that reads the source: the loaded scorer and what it needs to score
//!   the longest record, or the compiled pattern; the scores of one batch; the kept
//!   candidates of a top-k (24 bytes each); [`SCAN_FIXED_BYTES`]; the run of rows of
//!   the relation its originals are found in; and its tables;
//! - a selection derived from a prepared set, and a view: the blocks it reads and
//!   writes at a time ([`DERIVE_BYTES`], [`COMPOSE_FIXED_BYTES`]) and its tables;
//! - a snapshot: what `Scope::open_archive` asks for.
//!
//! The tables of a set are the states of every original (two bits each) and, for a
//! score, the score of every original (four bytes each). They are kept in memory when
//! the budget admits them beside the rest and the host did not place them on disk
//! (`TablePlacement`), and in files of the work directory otherwise. A table on disk
//! is written and read in blocks through buffers that are part of the share: a
//! selection that reads the source writes its rows to a spool in the order the source
//! hands them out ([`SPOOL_BUFFER_BYTES`]), sorts the spool by row within the memory
//! it asked for (`crate::extsort`) and places it block by block ([`PLACE_BYTES`]).
//!
//! What a committed artifact keeps in memory is held from its commit on, for as long
//! as the service lives. A share that does not fit is refused with the figures.
//!
//! # Reuse
//!
//! A prepared set is not computed twice. The same producer over the same population
//! is the same set. Another rule of a scorer revision whose scores a set keeps for
//! the population is derived from those scores, and the same filter or membership
//! over a part of a snapshot is derived from the set over the whole snapshot: neither
//! reads the source nor scores a row (requirement 6.3).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use l2d_core::marks::REGEX_HELD_BYTES;
use l2d_core::polars::prelude::Column;
use l2d_core::selections::{SearchQuery, compile};
use l2d_core::zip_input::DiskDedup;
use l2d_core::{FreezeDiskCheck, L2dError, LogsFrame, MemoryBudget, ZipInput, ZipInputSettings};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::membership::{ABSTAINED, Expr, State, States, Tally, code_in, put_code};
use super::state::{
    Population, Producer, Progress, Refinement, Run, Set, SetKind, Shared, Snapshot,
};
use super::tables::{BLOCK_ROWS, Scores, SetDir, TablePlacement, require_disk, score_in};
use super::wire::{self, Fault, MAX_MESSAGE_BYTES, VERSION, cancelled, share};
use super::{FrozenScorer, ScorerSource};
use crate::extsort::{IO_BYTES, MIN_SORT_BYTES, sort_file};
use crate::scope::{RowBlock, Scope, ScopeSettings, archive_summary_bytes};

/// What reading a snapshot's source again holds beside the reader and the shares named
/// in the module description: the positions and texts of a batch on their way, the
/// summary the reader ends with, a result on its way out and an excerpt.
pub(crate) const SCAN_FIXED_BYTES: u64 = 1024 * 1024;
/// What composing a view holds beside its tables and the blocks of its operands: the
/// expression, the result and the rows its members are counted from at a time.
pub(crate) const COMPOSE_FIXED_BYTES: u64 = 64 * 1024 + TALLY_BYTES;
/// What an artifact keeps beside its tables and strings: its own structure, its place
/// in the registry and the room strings may have beyond their content.
pub(crate) const ARTIFACT_FIXED_BYTES: u64 = 2 * 1024;
/// The most reasons for what a source left out that an open result lists.
const MAX_REASONS: usize = 24;
/// What counting the members of a set reads at a time: a block of its codes, a block
/// of its parent's and a run of rows of the relation.
pub(crate) const TALLY_BYTES: u64 = 2 * States::BLOCK_BYTES as u64 + RowBlock::BYTES;
/// What deriving a set from a prepared one reads and writes at a time: a block of the
/// population, of the prepared codes, of the new codes and of the prepared scores.
const DERIVE_BYTES: u64 = 3 * States::BLOCK_BYTES as u64 + Scores::BLOCK_BYTES as u64;
/// One spooled row of a selection that goes to disk: its row, its score or the line
/// that named it, and its code.
const SPOOL_RECORD_BYTES: usize = 16;
/// The buffer the spool is written through.
const SPOOL_BUFFER_BYTES: u64 = 64 * 1024;
/// The spooled rows placed at a time, and what placing holds: those rows, a block of
/// codes and a block of scores.
const PLACE_RECORDS: usize = 4_096;
const PLACE_BYTES: u64 = (PLACE_RECORDS * SPOOL_RECORD_BYTES) as u64
    + States::BLOCK_BYTES as u64
    + Scores::BLOCK_BYTES as u64;
/// What reading a registered membership holds: the buffer it is read through, a line
/// and the hash of the file.
const MEMBERSHIP_READ_BYTES: u64 = 64 * 1024 + 1024;
/// The most bytes of a line of a membership file, its line end not counted: one
/// object `{"id","state"}` takes 46 at its longest without spaces.
pub(crate) const MAX_MEMBERSHIP_LINE: usize = 128;

/// What an operation works with on the worker.
pub(crate) struct Worker<'a> {
    pub shared: &'a Shared,
    pub budget: &'a mut MemoryBudget,
    pub scorers: &'a dyn ScorerSource,
    pub cancel: &'a AtomicBool,
    pub progress: &'a Progress,
    /// What the service holds between jobs: the adapter's share and every artifact.
    pub base: u64,
    /// The directories made so far, for the names of new ones.
    pub dirs: &'a mut u64,
    /// How long the job waited for the worker and the host before it ran.
    pub queue_ms: u64,
}

impl Worker<'_> {
    pub(crate) fn fresh_dir(&mut self, kind: &str) -> PathBuf {
        *self.dirs += 1;
        self.shared
            .config
            .work_root
            .join(format!("{kind}-{}", *self.dirs))
    }

    pub(crate) fn check_cancel(&self) -> Result<(), L2dError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(cancelled());
        }
        Ok(())
    }
}

/// The counters every job states (performance.md, "For each job report"): what was
/// read, inflated, scored and reused, what went to disk, how long it took and what it
/// was admitted. A counter that does not apply to a job is 0; one that this host
/// cannot observe is null and named in `unavailable`.
///
/// The two memory peaks are the server's, not the job's: `server_counted_peak_bytes`
/// is the highest figure the one budget of the server has admitted since it started,
/// and `server_process_peak_bytes` the peak working set of the process. A job that
/// stays below an earlier job's peak repeats that peak; the limit both are compared
/// with is `memory_limit_bytes`.
pub(crate) struct Cost {
    fields: Map<String, Value>,
    started: Instant,
}

impl Cost {
    pub fn new(started: Instant) -> Self {
        let mut fields = Map::new();
        for counter in [
            "archive_scans",
            "bytes_packed",
            "bytes_unpacked",
            "entries_read",
            "entries_skipped",
            "records_scanned",
            "records_skipped",
            "accepted_occurrences",
            "distinct_originals",
            "candidates",
            "output",
            "rows_scored",
            "rows_encoded",
            "genai_rows_submitted",
            "index_rows_read",
            "spilled_bytes",
            "spilled_records",
            "disk_bytes",
            "disk_peak_bytes",
        ] {
            fields.insert(counter.into(), json!(0));
        }
        fields.insert("reused".into(), Value::Null);
        fields.insert("dedup_on_disk".into(), json!(false));
        fields.insert("tables_on_disk".into(), json!(false));
        Self { fields, started }
    }

    pub fn set(&mut self, counter: &str, value: impl Into<Value>) -> &mut Self {
        self.fields.insert(counter.into(), value.into());
        self
    }

    /// What one read of a source cost.
    fn scanned(&mut self, scan: &ScanCost) -> &mut Self {
        self.set("archive_scans", 1)
            .set("bytes_packed", scan.packed)
            .set("bytes_unpacked", scan.unpacked)
            .set("entries_read", scan.entries_read)
            .set("entries_skipped", scan.entries_skipped)
            .set("entries_at_once", scan.entries_at_once)
            .set("records_scanned", scan.records)
            .set("records_skipped", scan.skipped)
            .set("accepted_occurrences", scan.records - scan.skipped)
            .set("distinct_originals", scan.distinct)
            .set("dedup_on_disk", scan.dedup.is_some())
            .set(
                "spilled_bytes",
                scan.dedup.map_or(0, |dedup| dedup.spilled_bytes),
            )
            .set(
                "spilled_records",
                scan.dedup.map_or(0, |dedup| dedup.spilled_records),
            )
    }

    /// The counters with the time, the wait and the memory of the job that ends here.
    pub fn finish(mut self, worker: &Worker<'_>) -> Value {
        let report = worker.shared.usage.report();
        let peak = super::process::peak_bytes();
        self.set("elapsed_ms", self.started.elapsed().as_millis() as u64)
            .set("queue_ms", worker.queue_ms)
            .set("memory_limit_bytes", report.memory_limit_bytes)
            .set("server_counted_peak_bytes", report.counted_peak_bytes)
            .set("server_process_peak_bytes", peak)
            .set(
                "unavailable",
                if peak.is_some() {
                    json!([])
                } else {
                    json!(["server_process_peak_bytes"])
                },
            )
            .set("response_bytes", 0);
        Value::Object(self.fields)
    }
}

/// What an operation ends with.
pub(crate) struct Done {
    /// The result as it is returned and replayed.
    pub result: String,
    pub commit: Option<Commit>,
}

pub(crate) enum Commit {
    Snapshot(Arc<Snapshot>),
    Set(Arc<Set>),
    Refinement(Arc<Refinement>),
    /// A run that issued a batch or took a submission: it is registered under
    /// `handles` (its own when it is new, and that of a new batch), it keeps `grown`
    /// bytes more, and `set` is what was collected from it.
    Run {
        run: Arc<Run>,
        handles: Vec<String>,
        grown: u64,
        set: Option<Arc<Set>>,
    },
    /// A snapshot reopened from a receipt with every set it retained, committed as one.
    Reopened {
        snapshot: Arc<Snapshot>,
        sets: Vec<Arc<Set>>,
    },
}

impl Commit {
    /// What the commit adds to what the service holds between jobs.
    pub fn held(&self) -> u64 {
        match self {
            Commit::Snapshot(snapshot) => snapshot.held,
            Commit::Set(set) => set.held,
            Commit::Refinement(refinement) => refinement.held,
            Commit::Run { grown, set, .. } => {
                grown.saturating_add(set.as_ref().map_or(0, |set| set.held))
            }
            Commit::Reopened { snapshot, sets } => sets
                .iter()
                .fold(snapshot.held, |held, set| held.saturating_add(set.held)),
        }
    }

    /// Registers what was made.
    pub fn register(self, registry: &mut super::state::Registry) {
        match self {
            Commit::Snapshot(snapshot) => registry.insert_snapshot(snapshot),
            Commit::Set(set) => registry.insert_set(set),
            Commit::Reopened { snapshot, sets } => {
                registry.insert_snapshot(snapshot);
                for set in sets {
                    registry.insert_set(set);
                }
            }
            Commit::Refinement(refinement) => registry.insert_refinement(refinement),
            Commit::Run {
                run, handles, set, ..
            } => {
                for handle in handles {
                    registry.insert_run(handle, run.clone());
                }
                if let Some(set) = set {
                    registry.insert_set(set);
                }
            }
        }
    }
}

/// A removed directory when the operation that made it ends, however it ends.
pub(crate) struct Scratch(pub PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A removed file when the operation that made it ends, however it ends.
struct Gone(PathBuf);

impl Drop for Gone {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn failed(error: impl std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("reading a batch for a triager set failed: {error}"),
    }
}

fn io_failed(doing: &str, path: &Path, error: &dyn std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

fn drift(detail: impl Into<String>) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "{}; the source no longer reads as the snapshot bound it, open it again",
            detail.into()
        ),
    }
}

// --- Open ----------------------------------------------------------------------------------

pub(crate) struct OpenTask {
    pub path: PathBuf,
    pub zip: ZipInputSettings,
    /// How results name the source: the registered name or the path as it was given.
    pub named: Value,
    pub public: bool,
}

pub(crate) fn open(worker: &mut Worker<'_>, task: &OpenTask) -> Result<Done, Fault> {
    let shared = worker.shared;
    let config = &shared.config;
    let settings = ScopeSettings {
        index: config.index,
        dedup_set_texts: config.dedup_set_texts,
        free_disk_bytes: config.free_disk_bytes,
        ..ScopeSettings::new(worker.fresh_dir("snapshot"))
    };
    let started = Instant::now();
    worker.budget.hold(worker.base);
    worker
        .shared
        .activity
        .archive_scans
        .fetch_add(1, Ordering::Relaxed);
    let (cancel, progress) = (worker.cancel, worker.progress);
    let (scope, summary) = Scope::open_archive_watched(
        &task.path,
        &task.zip,
        &settings,
        worker.budget,
        &mut |rows| {
            progress.batches.fetch_add(1, Ordering::Relaxed);
            progress.rows.store(rows, Ordering::Relaxed);
            if cancel.load(Ordering::Relaxed) {
                return Err(cancelled());
            }
            Ok(())
        },
    )?;
    let selected = scope.selected_bytes().unwrap_or_default();
    shared
        .activity
        .bytes_unpacked
        .fetch_add(selected.unpacked, Ordering::Relaxed);
    // The scope and the reader's summary are this operation's to hold from the return
    // on (tasks.md, note 1.1): they are held before anything else is built.
    let live = scope
        .held_bytes()
        .saturating_add(archive_summary_bytes(&summary));
    worker.budget.hold(worker.base.saturating_add(live));
    progress
        .spilled
        .store(scope.dedup_on_disk(), Ordering::Relaxed);
    let handle = wire::handle("snap", &config.project, &[scope.handle()]);
    if let Some(known) = registry(worker.shared).snapshot(&handle) {
        // The same source, revision and reading is the same snapshot: the scope just
        // opened is let go, and with it its work directory.
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }
    // What stays beside them: the snapshot's own strings and its result. It is asked
    // for before the result is built; when the relation and the summary alone are
    // more than the limit leaves, nothing is left for it and the open is refused.
    let beside = ARTIFACT_FIXED_BYTES + 4 * MAX_MESSAGE_BYTES as u64 * MAX_REASONS as u64;
    worker.budget.require_whole(
        beside,
        "keeping an open snapshot beside its relation and the summary of its read,",
    )?;
    let kept = live.saturating_add(beside);
    worker.budget.hold(worker.base.saturating_add(kept));

    let mut snapshot = Snapshot {
        handle,
        public: task.public,
        path: task.path.clone(),
        zip: task.zip.clone(),
        result: String::new(),
        held: kept,
        summary,
        scope,
        retained: None,
    };
    let mut source = task.named.clone();
    source["class"] = json!(if task.public {
        "public_fixture"
    } else {
        "private"
    });
    source["adapter"] = json!("zip");
    source["content_sha256"] = json!(snapshot.content_sha256());
    source["entries"] = json!(task.zip.entries);
    source["max_record_bytes"] = json!(task.zip.max_record_bytes);
    let scope = &snapshot.scope;
    let completeness = scope.completeness();
    let reasons: Vec<Value> = completeness
        .reasons
        .iter()
        .take(MAX_REASONS)
        .map(|reason| {
            json!({
                "unit": reason.unit,
                "count": reason.count,
                "reason": wire::display(&reason.reason, 200).0,
            })
        })
        .collect();
    let dedup = scope.disk_dedup();
    let spilled = dedup.map_or(0, |dedup| dedup.spilled_bytes);
    let mut cost = Cost::new(started);
    cost.set("archive_scans", 1)
        .set("bytes_packed", selected.packed)
        .set("bytes_unpacked", selected.unpacked)
        .set("entries_read", completeness.entries_read)
        .set(
            "entries_skipped",
            completeness.entries_incomplete
                + completeness.entries_unsupported
                + completeness.entries_excluded,
        )
        .set("entries_at_once", scope.archive_readers())
        .set("records_scanned", completeness.records_scanned)
        .set("records_skipped", completeness.skipped_records)
        .set("accepted_occurrences", completeness.occurrences)
        .set("distinct_originals", scope.originals())
        .set("candidates", completeness.occurrences)
        .set("output", scope.originals())
        .set("dedup_on_disk", dedup.is_some())
        .set("spilled_bytes", spilled)
        .set(
            "spilled_records",
            dedup.map_or(0, |dedup| dedup.spilled_records),
        )
        .set("index_on_disk", scope.index_on_disk())
        .set("index_ordered_on_disk", scope.index_sorted())
        .set("disk_bytes", scope.disk_bytes())
        .set(
            "disk_peak_bytes",
            scope.disk_peak_bytes().saturating_add(spilled),
        );
    let result = wire::with_response_bytes(json!({
        "v": VERSION,
        "status": "ready",
        "snapshot": snapshot.handle,
        "revision": scope.binding().relation,
        "source": source,
        "counts": {
            "distinct_originals": scope.originals(),
            "occurrences": scope.occurrences(),
        },
        "completeness": {
            "complete": completeness.complete,
            "records_scanned": completeness.records_scanned,
            "skipped_records": completeness.skipped_records,
            "entries_read": completeness.entries_read,
            "entries_incomplete": completeness.entries_incomplete,
            "entries_unsupported": completeness.entries_unsupported,
            "entries_excluded": completeness.entries_excluded,
            "reasons": reasons,
            "reasons_clipped": completeness.reasons.len() > MAX_REASONS,
        },
        "capabilities": {
            "producers": ["score", "filter", "membership", "recommend"],
            "score_rules": ["top_k", "threshold"],
            "ties": ["record_id_ascending"],
            "filter_syntax": ["literal", "glob", "regex", "pattern"],
            "recommend_methods": ["venn", "issued_batch"],
            "refine_actions": ["patterns", "list", "split", "merge", "preview", "neighbours"],
            "neighbour_methods": worker.shared.config.neighbour_methods(),
            "compose_modes": ["view", "allocate"],
            "operators": ["union", "intersection", "difference", "complement", "state"],
            "allocation_rule": super::allocate::ALLOCATION_RULE,
            "inspect_views": ["overview", "regions", "records", "evidence"],
            "sort": ["record_id_ascending", "allocation"],
            "job_actions": ["status", "cancel"],
            "receipts": worker.shared.config.receipts.is_some(),
            "occurrence_locators": "first_only",
            "not_supported": [
                "recommend method local_model",
                "refine region", "refine field", "refine time",
                "inspect histogram", "inspect fields", "inspect lineage",
            ],
        },
        "disclosure": worker.shared.disclosure(&snapshot),
        "cost": cost.finish(worker),
    }));
    snapshot.result = result;
    if wire::transport_bytes(&snapshot.result) > wire::PAYLOAD_TRANSPORT_BYTES {
        return Err(Fault::runtime(
            "the result of an open does not fit the response limit",
        ));
    }
    // What the snapshot keeps, now that it is known: the relation and the summary,
    // its result here and in the job table, its path and its patterns. It is at most
    // what was asked for above.
    let strings = 2 * snapshot.result.capacity()
        + snapshot.path.as_os_str().len()
        + snapshot.zip.entries.capacity() * std::mem::size_of::<String>()
        + snapshot
            .zip
            .entries
            .iter()
            .map(String::capacity)
            .sum::<usize>();
    snapshot.held = live
        .saturating_add(strings as u64)
        .saturating_add(ARTIFACT_FIXED_BYTES)
        .min(kept);
    worker
        .budget
        .hold(worker.base.saturating_add(snapshot.held));
    Ok(Done {
        result: snapshot.result.clone(),
        commit: Some(Commit::Snapshot(Arc::new(snapshot))),
    })
}

pub(crate) fn registry(shared: &Shared) -> std::sync::RwLockReadGuard<'_, super::state::Registry> {
    shared
        .registry
        .read()
        .unwrap_or_else(|held| held.into_inner())
}

// --- Reading a snapshot's source again -----------------------------------------------------

/// What one read of a snapshot's source cost.
pub(crate) struct ScanCost {
    records: u64,
    skipped: u64,
    entries_read: u64,
    entries_skipped: u64,
    distinct: u64,
    packed: u64,
    unpacked: u64,
    dedup: Option<DiskDedup>,
    entries_at_once: usize,
}

/// Streams the source of `snapshot` again, as it was opened, and gives every batch of
/// distinct originals to `each`, which says whether to go on.
///
/// `held` is what the caller holds in the budget for the whole read; the reader sizes
/// its batches beside it. The archive must still be the one the snapshot bound, and a
/// read to the end must give the snapshot's records and originals. `None` when `each`
/// ended the read early. Deduplication goes on on disk in a directory of its own,
/// which is removed when the read ends, also when it is cancelled or fails.
pub(crate) fn scan(
    worker: &mut Worker<'_>,
    snapshot: &Snapshot,
    held: u64,
    each: &mut dyn FnMut(&LogsFrame) -> Result<bool, L2dError>,
) -> Result<Option<ScanCost>, L2dError> {
    let shared = worker.shared;
    let config = &shared.config;
    worker.check_cancel()?;
    shared
        .activity
        .archive_scans
        .fetch_add(1, Ordering::Relaxed);
    let dir = worker.fresh_dir("job");
    std::fs::create_dir_all(&dir).map_err(|error| L2dError::Runtime {
        detail: format!("creating {} failed: {error}", dir.display()),
    })?;
    // Declared before the archive, so the archive removes its own files first.
    let scratch = Scratch(dir);
    worker.budget.hold(held);
    let mut input = ZipInput::open(&snapshot.path, &snapshot.zip, worker.budget)?;
    if input.content_sha256() != snapshot.content_sha256() {
        return Err(L2dError::Refused {
            reason: format!(
                "the archive {} changed since the snapshot was opened: it now has the \
                 SHA-256 {} and the snapshot binds {}; open it again",
                snapshot.path.display(),
                input.content_sha256(),
                snapshot.content_sha256()
            ),
        });
    }
    input.hold_beside(held);
    input.spill_into(&scratch.0.join("dedup"), config.dedup_set_texts)?;
    input.check_disk_at_freeze(FreezeDiskCheck {
        beside: Vec::new(),
        free_bytes: config.free_disk_bytes,
    });
    let selected = input.selected_bytes();
    shared
        .activity
        .bytes_unpacked
        .fetch_add(selected.unpacked, Ordering::Relaxed);
    let mut rows = 0u64;
    let mut ended_early = false;
    {
        let mut batches = input.batches(worker.budget);
        while let Some(batch) = batches.next() {
            let batch = batch?;
            if worker.cancel.load(Ordering::Relaxed) {
                return Err(cancelled());
            }
            let go_on = each(&batch)?;
            rows += batch.len() as u64;
            worker.progress.batches.fetch_add(1, Ordering::Relaxed);
            worker.progress.rows.store(rows, Ordering::Relaxed);
            worker
                .progress
                .spilled
                .store(batches.disk_dedup().is_some(), Ordering::Relaxed);
            if !go_on {
                ended_early = true;
                break;
            }
        }
    }
    if ended_early {
        drop(input);
        worker.budget.hold(held);
        return Ok(None);
    }
    let dedup = input.disk_dedup();
    let entries_at_once = input.entries_at_once();
    let summary = input.finish();
    worker.budget.hold(held);
    let bound = snapshot.scope.completeness();
    if summary.input.records != bound.records_scanned
        || summary.input.distinct_texts != bound.originals
        || rows != bound.originals
    {
        return Err(drift(format!(
            "the read gave {} records and {rows} originals, and the snapshot holds {} and {}",
            summary.input.records, bound.records_scanned, bound.originals
        )));
    }
    let entries_read = summary.read().count() as u64;
    Ok(Some(ScanCost {
        records: summary.input.records,
        skipped: summary.input.skipped.len() as u64,
        entries_read,
        entries_skipped: summary.entries.len() as u64 - entries_read,
        distinct: summary.input.distinct_texts,
        packed: selected.packed,
        unpacked: selected.unpacked,
        dedup,
        entries_at_once,
    }))
}

/// The texts and first positions of a batch.
pub(crate) fn rows_of(
    logs: &LogsFrame,
) -> Result<impl Iterator<Item = (&str, u64)> + '_, L2dError> {
    let frame = logs.frame();
    let texts = frame
        .column(LogsFrame::TEXT)
        .and_then(Column::str)
        .map_err(failed)?;
    let positions = frame
        .column(LogsFrame::FIRST_POSITION)
        .and_then(Column::u64)
        .map_err(failed)?;
    if texts.null_count() > 0 || positions.null_count() > 0 {
        return Err(failed("a batch holds a null"));
    }
    Ok(texts.into_no_null_iter().zip(positions.into_no_null_iter()))
}

// --- Tables of a new set -------------------------------------------------------------------

/// Whether the tables of a new set go to disk: when the host placed them there, or
/// when `in_memory` bytes are more than the budget has left beside what the operation
/// holds.
pub(crate) fn tables_on_disk(worker: &Worker<'_>, in_memory: u64) -> bool {
    place_on_disk(worker.shared.config.tables, worker.budget, in_memory)
}

fn place_on_disk(placement: TablePlacement, budget: &MemoryBudget, in_memory: u64) -> bool {
    placement == TablePlacement::Disk
        || budget
            .require_whole(in_memory, "keeping the tables of a set in memory")
            .is_err()
}

/// The directory of a new set's tables, after the disk was asked for `bytes`.
pub(crate) fn set_dir(
    worker: &mut Worker<'_>,
    bytes: u64,
    what: &str,
) -> Result<Arc<SetDir>, L2dError> {
    let config = &worker.shared.config;
    std::fs::create_dir_all(&config.work_root)
        .map_err(|error| io_failed("creating", &config.work_root, &error))?;
    require_disk(&config.work_root, bytes, config.free_disk_bytes, what)?;
    worker.budget.note_spill();
    SetDir::create(worker.fresh_dir("set"))
}

/// The rows of a selection on their way to its tables.
enum Sink {
    /// Tables in memory, written as the rows come.
    Direct {
        states: States,
        scores: Option<Scores>,
    },
    /// Tables on disk: the rows are spooled, sorted by row and placed. The spool is
    /// removed with the sink, and the directory after it.
    Spool {
        writer: BufWriter<File>,
        spool: Gone,
        records: u64,
        rows: u64,
        with_scores: bool,
        sort_bytes: u64,
        dir: Arc<SetDir>,
    },
}

impl Sink {
    /// What a sink for `rows` rows takes in memory, and whether it is on disk; `left`
    /// is what the budget has left beside the rest of the operation.
    fn plan(worker: &Worker<'_>, rows: u64, with_scores: bool) -> (u64, bool, u64) {
        let in_memory = States::bytes_for(rows)
            + if with_scores {
                Scores::bytes_for(rows)
            } else {
                0
            };
        if !tables_on_disk(worker, in_memory) {
            return (in_memory, false, 0);
        }
        let sort_bytes = Self::least_sort_bytes(rows);
        (Self::spool_share(sort_bytes), true, sort_bytes)
    }

    /// The memory the spool of `rows` rows is sorted in: the least that does it,
    /// [`MIN_SORT_BYTES`], or for very many rows twice what merges their runs in one
    /// pass (a run and its buffer in the merge both grow with the square root of the
    /// rows). A selection is on disk to leave the memory to the read of its source,
    /// whether the host placed it there or the budget did.
    fn least_sort_bytes(rows: u64) -> u64 {
        let merge = (rows as f64 * (SPOOL_RECORD_BYTES as f64 + 4.0) * 256.0).sqrt() as u64;
        (2 * merge + IO_BYTES + 4096).max(MIN_SORT_BYTES)
    }

    /// What a spooled selection holds: the buffer of its spool and, after the read,
    /// the memory its rows are sorted in or the blocks they are placed through.
    fn spool_share(sort_bytes: u64) -> u64 {
        SPOOL_BUFFER_BYTES + sort_bytes.max(PLACE_BYTES)
    }

    /// A spool in `dir` for `rows` rows, sorted within `sort_bytes`.
    fn spool(
        dir: Arc<SetDir>,
        rows: u64,
        with_scores: bool,
        sort_bytes: u64,
    ) -> Result<Self, L2dError> {
        let path = dir.path().join("spool.bin");
        let writer = File::create(&path)
            .map(|file| BufWriter::with_capacity(SPOOL_BUFFER_BYTES as usize, file))
            .map_err(|error| io_failed("creating", &path, &error))?;
        Ok(Sink::Spool {
            writer,
            spool: Gone(path),
            records: 0,
            rows,
            with_scores,
            sort_bytes,
            dir,
        })
    }

    /// The disk a spooled selection takes at most: the spool, its runs and the sorted
    /// rows, and the tables.
    fn disk_peak(rows: u64, with_scores: bool) -> u64 {
        3 * rows * SPOOL_RECORD_BYTES as u64
            + States::bytes_for(rows)
            + if with_scores {
                Scores::bytes_for(rows)
            } else {
                0
            }
    }

    fn create(
        worker: &mut Worker<'_>,
        rows: u64,
        with_scores: bool,
        on_disk: bool,
        sort_bytes: u64,
    ) -> Result<Self, L2dError> {
        if !on_disk {
            return Ok(Sink::Direct {
                states: States::in_memory(rows, State::Unknown),
                scores: with_scores.then(|| Scores::in_memory(rows)),
            });
        }
        let dir = set_dir(
            worker,
            Self::disk_peak(rows, with_scores),
            "placing the membership of a selection on disk",
        )?;
        Self::spool(dir, rows, with_scores, sort_bytes)
    }

    /// The code of `row`, with its score or the line that named it in `beside`.
    fn put(&mut self, row: u64, code: u8, beside: u32) -> Result<(), L2dError> {
        match self {
            Sink::Direct { states, scores } => {
                states.set(row, code)?;
                if let Some(scores) = scores {
                    scores.set(row, f32::from_bits(beside))?;
                }
                Ok(())
            }
            Sink::Spool {
                writer,
                spool,
                records,
                ..
            } => {
                let mut record = [0u8; SPOOL_RECORD_BYTES];
                record[..8].copy_from_slice(&row.to_le_bytes());
                record[8..12].copy_from_slice(&beside.to_le_bytes());
                record[12] = code;
                *records += 1;
                writer
                    .write_all(&record)
                    .map_err(|error| io_failed("writing", &spool.0, &error))
            }
        }
    }

    /// The code a row already has, where the tables are in memory.
    fn code(&self, row: u64) -> Result<Option<u8>, L2dError> {
        match self {
            Sink::Direct { states, .. } => states.code(row).map(Some),
            Sink::Spool { .. } => Ok(None),
        }
    }

    /// The tables. A spool is sorted and placed; `conflict` makes the error for a row
    /// that was put twice with different codes, from the two values put beside it.
    fn finish(
        self,
        conflict: &dyn Fn(u32, u32) -> L2dError,
    ) -> Result<(States, Option<Scores>), L2dError> {
        let (dir, writer, spool, records, rows, with_scores, sort_bytes) = match self {
            Sink::Direct { states, scores } => return Ok((states, scores)),
            Sink::Spool {
                writer,
                spool,
                records,
                rows,
                with_scores,
                sort_bytes,
                dir,
            } => (dir, writer, spool, records, rows, with_scores, sort_bytes),
        };
        // The spool and the sorted rows are removed however this ends, before the
        // directory is let go.
        let sorted = dir.path().join("spool-sorted.bin");
        let _sorted = Gone(sorted.clone());
        let path = spool.0.clone();
        writer
            .into_inner()
            .map(drop)
            .map_err(|error| io_failed("writing", &path, &error.into_error()))?;
        sort_file(
            &path,
            &sorted,
            dir.path(),
            SPOOL_RECORD_BYTES,
            records,
            sort_bytes,
        )?;
        drop(spool);
        let mut states = States::on_disk(&dir, rows)?;
        let mut scores = match with_scores {
            true => Some(Scores::on_disk(&dir, rows)?),
            false => None,
        };
        let mut from =
            File::open(&sorted).map_err(|error| io_failed("opening", &sorted, &error))?;
        let mut spooled = vec![0u8; PLACE_RECORDS * SPOOL_RECORD_BYTES];
        let mut codes = vec![0u8; States::BLOCK_BYTES];
        let mut values = vec![0u8; Scores::BLOCK_BYTES];
        // A row the spool does not name was not evaluated and has no score.
        let unknown = (State::Unknown as u8) * 0b0101_0101;
        let fresh = |codes: &mut [u8], values: &mut [u8]| {
            codes.fill(unknown);
            for value in values.as_chunks_mut::<4>().0 {
                *value = f32::NAN.to_le_bytes();
            }
        };
        fresh(&mut codes, &mut values);
        let mut block = 0u64;
        let flush = |block: u64,
                     codes: &[u8],
                     values: &[u8],
                     states: &mut States,
                     scores: &mut Option<Scores>|
         -> Result<(), L2dError> {
            let first = block * BLOCK_ROWS;
            let in_block = BLOCK_ROWS.min(rows.saturating_sub(first)) as usize;
            if in_block == 0 {
                return Ok(());
            }
            states.write_block(first, codes, in_block)?;
            if let Some(scores) = scores {
                scores.write_block(first, values, in_block)?;
            }
            Ok(())
        };
        let mut last: Option<(u64, u8, u32)> = None;
        let mut left = records;
        while left > 0 {
            let now = (left as usize).min(PLACE_RECORDS);
            let spooled = &mut spooled[..now * SPOOL_RECORD_BYTES];
            from.read_exact(spooled)
                .map_err(|error| io_failed("reading", &sorted, &error))?;
            left -= now as u64;
            for record in spooled.as_chunks::<SPOOL_RECORD_BYTES>().0 {
                let row = u64::from_le_bytes(record[..8].try_into().expect("8 bytes"));
                let beside = u32::from_le_bytes(record[8..12].try_into().expect("4 bytes"));
                let code = record[12];
                if row >= rows {
                    return Err(L2dError::Runtime {
                        detail: "a spooled row lies beyond the originals".into(),
                    });
                }
                if let Some((before, code_before, beside_before)) = last
                    && before == row
                {
                    if code_before != code {
                        return Err(conflict(beside_before, beside));
                    }
                    continue;
                }
                last = Some((row, code, beside));
                while block < row / BLOCK_ROWS {
                    flush(block, &codes, &values, &mut states, &mut scores)?;
                    fresh(&mut codes, &mut values);
                    block += 1;
                }
                let at = (row % BLOCK_ROWS) as usize;
                put_code(&mut codes, at, code);
                values[at * 4..at * 4 + 4].copy_from_slice(&beside.to_le_bytes());
            }
        }
        while block * BLOCK_ROWS < rows {
            flush(block, &codes, &values, &mut states, &mut scores)?;
            fresh(&mut codes, &mut values);
            block += 1;
        }
        Ok((states, scores))
    }
}

// --- Select --------------------------------------------------------------------------------

pub(crate) enum Rule {
    TopK(u64),
    Threshold(f64),
}

pub(crate) enum SelectProducer {
    Score {
        model: String,
        rule: Rule,
    },
    Filter {
        syntax: &'static str,
        query: SearchQuery,
    },
    /// A membership file the host registered under `name`.
    Membership {
        name: String,
        path: PathBuf,
    },
}

pub(crate) struct SelectTask {
    pub scope: Population,
    pub producer: SelectProducer,
}

/// A candidate of a top-k. The greater one is the better one: the higher score, and
/// among equal scores the lower record ID.
struct Kept {
    score: f32,
    position: u64,
    row: u64,
}

impl PartialEq for Kept {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Kept {}

impl Ord for Kept {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then(other.position.cmp(&self.position))
    }
}

impl PartialOrd for Kept {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

const KEPT_BYTES: u64 = std::mem::size_of::<Kept>() as u64;

/// The best `most` candidates seen so far.
struct Best {
    kept: BinaryHeap<Reverse<Kept>>,
    most: u64,
}

impl Best {
    fn new(most: u64) -> Self {
        Self {
            kept: BinaryHeap::with_capacity(most as usize),
            most,
        }
    }

    /// Forgets the candidates and keeps their room: a selection that reads its
    /// source again holds one list of candidates, the one it asked the budget for.
    fn clear(&mut self) {
        self.kept.clear();
    }

    fn offer(&mut self, candidate: Kept) {
        if (self.kept.len() as u64) < self.most {
            self.kept.push(Reverse(candidate));
        } else if self.kept.peek().is_some_and(|worst| candidate > worst.0) {
            self.kept.pop();
            self.kept.push(Reverse(candidate));
        }
    }
}

/// Counts the members of `parent` by their state in `states`, in both units, and
/// digests the membership. Also returns what lies outside the parent. The caller
/// holds [`TALLY_BYTES`] for the blocks this reads.
pub(crate) fn tally(
    snapshot: &Snapshot,
    parent: &Population,
    states: &States,
) -> Result<(Tally, String, (u64, u64)), L2dError> {
    let mut tally = Tally::default();
    let mut outside = (0u64, 0u64);
    let mut sum = 0u128;
    let mut codes = vec![0u8; States::BLOCK_BYTES];
    let mut within = vec![0u8; States::BLOCK_BYTES];
    let mut block = RowBlock::new();
    let mut first = 0;
    while first < states.rows() {
        let rows = states.read_block(first, &mut codes)?;
        parent.read_block(first, &mut within)?;
        let mut at = 0;
        while at < rows {
            let run = snapshot.scope.rows(first + at as u64, true, &mut block)? as usize;
            for at in at..(at + run).min(rows) {
                let row = first + at as u64;
                let code = code_in(&codes, at);
                let occurrences = block.occurrences(row);
                if code_in(&within, at) == State::In as u8 {
                    tally.add_code(code, occurrences);
                } else {
                    outside.0 += 1;
                    outside.1 += occurrences;
                }
                let state = State::of(code);
                if state != State::Out {
                    // Order does not matter to a sum, so the digest is the same
                    // however the source handed its originals out.
                    let mut hash = Sha256::new();
                    hash.update(block.position(row).to_le_bytes());
                    hash.update([state as u8]);
                    let mut part = [0u8; 16];
                    part.copy_from_slice(&hash.finalize()[..16]);
                    sum = sum.wrapping_add(u128::from_le_bytes(part));
                }
            }
            at += run;
        }
        first += BLOCK_ROWS;
    }
    Ok((tally, format!("{sum:032x}"), outside))
}

/// The coverage block of a reduction: what was evaluated and what was not, with the
/// reasons. What a producer abstained on is told apart from what it did not assess.
pub(crate) fn coverage(tally: &Tally, reason: &str) -> Value {
    let (unknown, unknown_occurrences) = tally.of(State::Unknown);
    let (parent, _) = tally.parent();
    let mut reasons = Vec::new();
    let (abstained, abstained_occurrences) = tally.abstained;
    if abstained > 0 {
        reasons.push(
            json!({"reason": "abstained", "distinct_originals": abstained,
                            "occurrences": abstained_occurrences}),
        );
    }
    if unknown > abstained {
        reasons.push(
            json!({"reason": reason, "distinct_originals": unknown - abstained,
                            "occurrences": unknown_occurrences - abstained_occurrences}),
        );
    }
    json!({
        "unit": "distinct_originals",
        "evaluated": parent - unknown,
        "not_evaluated": unknown,
        "reasons": reasons,
    })
}

/// Why the membership file registered as `name` could not be read. A client sees
/// the name it gave, never where the host keeps the file: the path is the host's.
fn membership_unread(name: &str, doing: &str, error: &std::io::Error) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "{doing} the file the host registered as the membership `{name}` failed ({:?}); \
             nothing was committed",
            error.kind()
        ),
    }
}

/// The SHA-256 and the length of the file at `path`, registered as `name`.
fn hash_membership(path: &Path, name: &str) -> Result<(String, u64), L2dError> {
    let mut file = File::open(path).map_err(|error| membership_unread(name, "opening", &error))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| membership_unread(name, "reading", &error))?;
        if read == 0 {
            return Ok((format!("{:x}", hash.finalize()), bytes));
        }
        bytes += read as u64;
        hash.update(&buffer[..read]);
    }
}

/// A prepared set a new selection can be derived from without reading the source.
enum Prepared {
    /// A set that keeps the scores of the same scorer revision for the population.
    Scores(Arc<Set>),
    /// The same filter or membership over the whole snapshot.
    Membership(Arc<Set>),
}

impl Prepared {
    fn set(&self) -> &Arc<Set> {
        match self {
            Prepared::Scores(set) | Prepared::Membership(set) => set,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Prepared::Scores(_) => "scores",
            Prepared::Membership(_) => "membership",
        }
    }
}

fn prepared(
    worker: &Worker<'_>,
    snapshot: &Snapshot,
    scope: &Population,
    producer: &Producer,
    described: &Value,
) -> Option<Prepared> {
    let registry = registry(worker.shared);
    match producer {
        Producer::Score { model, scorer, .. } => registry
            .sets()
            .filter(|set| {
                set.snapshot.handle == snapshot.handle
                    && set.scores.is_some()
                    && matches!(&set.producer, Producer::Score { model: other, scorer: revision, .. }
                        if other == model && revision == scorer)
                    && match (&set.parent, scope) {
                        (None, _) => true,
                        (Some(parent), Population::Of(scope)) => parent.handle == scope.handle,
                        (Some(_), Population::All(_)) => false,
                    }
            })
            // The set over the whole snapshot first, the one that made the scores
            // before one that shares them, then by handle: the same choice whatever
            // order the registry keeps its sets in.
            .min_by_key(|set| (set.parent.is_some(), !set.owns_scores, set.handle.clone()))
            .cloned()
            .map(Prepared::Scores),
        Producer::Filter { .. } | Producer::Membership { .. } => {
            scope.as_set()?;
            let whole = wire::handle(
                "set",
                &worker.shared.config.project,
                &[&snapshot.handle, &snapshot.handle, &described.to_string()],
            );
            registry.set(&whole).map(Prepared::Membership)
        }
        // A view is composed, and a producer that names its records is fed them
        // again: neither is derived from another set.
        _ => None,
    }
}

pub(crate) fn select(worker: &mut Worker<'_>, task: &SelectTask) -> Result<Done, Fault> {
    let started = Instant::now();
    let snapshot = task.scope.snapshot().clone();
    let rows = snapshot.scope.originals();
    let project = worker.shared.config.project.clone();
    worker.budget.hold(worker.base);

    // The producer is made ready first: its revision is part of the set's identity.
    enum Ready {
        Score(Box<dyn FrozenScorer>),
        Filter(regex::Regex),
        Membership,
    }
    let mut membership_hashed = 0u64;
    let (ready, producer, producer_bytes) = match &task.producer {
        SelectProducer::Score { model, rule } => {
            let scorer = worker.scorers.load(model, worker.budget)?;
            worker
                .shared
                .activity
                .scorer_loads
                .fetch_add(1, Ordering::Relaxed);
            let bytes = scorer
                .held_bytes()
                .saturating_add(scorer.scoring_bytes(snapshot.zip.max_record_bytes));
            let (top_k, threshold) = match rule {
                Rule::TopK(k) => (Some(*k), None),
                Rule::Threshold(threshold) => (None, Some(*threshold)),
            };
            let producer = Producer::Score {
                model: model.clone(),
                scorer: scorer.identity(),
                top_k,
                threshold,
            };
            (Ready::Score(scorer), producer, bytes)
        }
        SelectProducer::Filter { syntax, query } => {
            let matcher = compile(query, worker.budget)?;
            let (text, case_insensitive) = match query {
                SearchQuery::Literal { text, insensitive } => (text, *insensitive),
                SearchQuery::Glob {
                    pattern,
                    insensitive,
                }
                | SearchQuery::Regex {
                    pattern,
                    insensitive,
                } => (pattern, *insensitive),
                SearchQuery::Fts5Trigram { expression } => (expression, true),
            };
            let producer = Producer::Filter {
                syntax,
                query: text.clone(),
                case_insensitive,
            };
            let bytes = REGEX_HELD_BYTES.saturating_add(16 * text.len() as u64);
            (Ready::Filter(matcher), producer, bytes)
        }
        SelectProducer::Membership { name, path } => {
            worker.budget.require_whole(
                MEMBERSHIP_READ_BYTES,
                "reading a registered membership file",
            )?;
            worker
                .budget
                .hold(worker.base.saturating_add(MEMBERSHIP_READ_BYTES));
            let (content_sha256, bytes) = hash_membership(path, name)?;
            // The file is read whole to bind its content, also when the selection is
            // then derived from a prepared set.
            membership_hashed = bytes;
            let producer = Producer::Membership {
                name: name.clone(),
                content_sha256,
                bytes,
            };
            (Ready::Membership, producer, MEMBERSHIP_READ_BYTES)
        }
    };
    // The loaded scorer or the compiled pattern is alive from here on, so it is held
    // from here on: its loader asked the budget for it, and nothing kept it.
    worker
        .budget
        .hold(worker.base.saturating_add(producer_bytes));
    let described = producer.describe();
    let handle = wire::handle(
        "set",
        &project,
        &[
            &snapshot.handle,
            task.scope.handle(),
            &described.to_string(),
        ],
    );
    if let Some(known) = registry(worker.shared).set(&handle) {
        // The same producer revision over the same population is the same set.
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }

    let (parent_size, _) = task.scope.size();
    let kept_candidates = match &task.producer {
        SelectProducer::Score {
            rule: Rule::TopK(k),
            ..
        } => (*k).min(parent_size),
        _ => 0,
    };
    let is_score = matches!(ready, Ready::Score(_));
    let mut cost = Cost::new(started);
    let reused = prepared(worker, &snapshot, &task.scope, &producer, &described);
    let mut scored = 0u64;
    let mut best;
    let (mut states, scores, owns_scores, on_disk) = if let Some(reused) = &reused {
        // Derived from a prepared set: the source is not read and nothing is scored.
        drop(ready);
        worker.budget.hold(worker.base);
        let fixed = DERIVE_BYTES
            .saturating_add(TALLY_BYTES)
            .saturating_add(kept_candidates.saturating_mul(KEPT_BYTES))
            .saturating_add(ARTIFACT_FIXED_BYTES + 64 * 1024)
            .saturating_add(producer.held_bytes());
        worker.budget.require_whole(
            fixed,
            "deriving a selection from a prepared set: the blocks it reads and writes at a \
             time and the kept candidates of a top-k,",
        )?;
        worker.budget.hold(worker.base.saturating_add(fixed));
        let on_disk = tables_on_disk(worker, States::bytes_for(rows));
        let held = worker
            .base
            .saturating_add(fixed)
            .saturating_add(if on_disk { 0 } else { States::bytes_for(rows) });
        worker.budget.hold(held);
        let mut states = if on_disk {
            let dir = set_dir(
                worker,
                States::bytes_for(rows),
                "placing the membership of a selection on disk",
            )?;
            States::on_disk(&dir, rows)?
        } else {
            States::in_memory(rows, State::Out)
        };
        best = Best::new(kept_candidates);
        let source = reused.set();
        let mut within = vec![0u8; States::BLOCK_BYTES];
        let mut known = vec![0u8; States::BLOCK_BYTES];
        let mut codes = vec![0u8; States::BLOCK_BYTES];
        let mut values = vec![0u8; Scores::BLOCK_BYTES];
        let mut first = 0;
        while first < rows {
            worker.check_cancel()?;
            let in_block = task.scope.read_block(first, &mut within)?;
            match reused {
                Prepared::Membership(_) => {
                    source.states.read_block(first, &mut known)?;
                }
                Prepared::Scores(_) => {
                    let scores = source.scores.as_ref().expect("a set with scores");
                    scores.read_block(first, &mut values)?;
                }
            }
            for at in 0..in_block {
                let code = if code_in(&within, at) != State::In as u8 {
                    // Outside the population the producer does not evaluate.
                    State::Unknown as u8
                } else if let Prepared::Membership(_) = reused {
                    code_in(&known, at)
                } else {
                    let score = score_in(&values, at);
                    if !score.is_finite() {
                        State::Unknown as u8
                    } else {
                        match &task.producer {
                            SelectProducer::Score {
                                rule: Rule::Threshold(threshold),
                                ..
                            } => u8::from(f64::from(score) >= *threshold),
                            _ => {
                                // Rows are in the order of the record IDs.
                                let row = first + at as u64;
                                best.offer(Kept {
                                    score,
                                    position: row,
                                    row,
                                });
                                State::Out as u8
                            }
                        }
                    }
                };
                put_code(&mut codes, at, code);
            }
            states.write_block(first, &codes, in_block)?;
            first += BLOCK_ROWS;
        }
        cost.set(
            "reused",
            json!({"kind": reused.kind(), "set": source.handle}),
        )
        .set("index_rows_read", rows);
        let scores = match reused {
            Prepared::Scores(_) => source.scores.clone(),
            Prepared::Membership(_) => None,
        };
        (states, scores, false, on_disk)
    } else {
        // Everything the selection holds, asked for before any of it is allocated.
        let batch_scores_bytes = if is_score {
            worker.budget.max_batch_rows().saturating_mul(4)
        } else {
            0
        };
        let fixed = batch_scores_bytes
            .saturating_add(kept_candidates.saturating_mul(KEPT_BYTES))
            .saturating_add(SCAN_FIXED_BYTES + ARTIFACT_FIXED_BYTES + TALLY_BYTES)
            // The run of rows the originals of the source are found in.
            .saturating_add(RowBlock::BYTES)
            .saturating_add(producer.held_bytes());
        // Beside the producer, which is held already: when that alone is more than
        // the limit leaves, nothing is left for these and the refusal names what is
        // held.
        worker.budget.require_whole(
            fixed,
            "selecting a set: one read of the source, the scores of a batch and the kept \
             candidates of a top-k, beside the producer,",
        )?;
        let fixed_held = worker
            .base
            .saturating_add(producer_bytes)
            .saturating_add(fixed);
        worker.budget.hold(fixed_held);
        const ON_DISK: &str = "placing the states and scores of every original on disk: the \
                               spool's buffer and the memory its rows are sorted in,";
        let (tables, mut on_disk, sort_bytes) = Sink::plan(worker, rows, is_score);
        worker.budget.require_whole(
            tables,
            if on_disk {
                ON_DISK
            } else {
                "keeping the states and scores of every original,"
            },
        )?;
        let mut held = fixed_held.saturating_add(tables);
        worker.budget.hold(held);

        let mut sink = Sink::create(worker, rows, is_score, on_disk, sort_bytes)?;
        best = Best::new(kept_candidates);
        let parent = task.scope.clone();
        let scope = &snapshot.scope;
        let mut near = RowBlock::new();
        match &ready {
            Ready::Membership => {
                let SelectProducer::Membership { path, name } = &task.producer else {
                    return Err(Fault::runtime("a membership without its file"));
                };
                worker
                    .shared
                    .activity
                    .membership_reads
                    .fetch_add(1, Ordering::Relaxed);
                let Producer::Membership {
                    content_sha256,
                    bytes,
                    ..
                } = &producer
                else {
                    return Err(Fault::runtime("a membership without its identity"));
                };
                let read = read_membership(path, name, &mut |line, position, code| {
                    let refused = |what: String| L2dError::Refused {
                        reason: format!(
                            "line {line} of the membership `{name}` {what}; nothing was committed"
                        ),
                    };
                    let row = scope.row_near(position, &mut near)?.ok_or_else(|| {
                        refused(format!(
                            "names the record {}, which is no original of the snapshot",
                            wire::record_id(position)
                        ))
                    })?;
                    if parent.state(row)? != State::In {
                        // A judgment outside the population fills no coverage: there
                        // the set is not evaluated, as `outside_scope` states.
                        return Ok(());
                    }
                    match sink.code(row)? {
                        Some(before) if before != State::Unknown as u8 && before != code => {
                            return Err(refused(format!(
                                "gives the record {} another state than a line before it",
                                wire::record_id(position)
                            )));
                        }
                        Some(before) if before != State::Unknown as u8 => return Ok(()),
                        _ => {}
                    }
                    sink.put(row, code, line as u32)
                })?;
                if read != (content_sha256.clone(), *bytes) {
                    return Err(Fault::refused(format!(
                        "the file of the membership `{name}` changed while it was read; \
                         nothing was committed"
                    )));
                }
                // Read a second time, line by line.
                membership_hashed += *bytes;
            }
            Ready::Score(_) | Ready::Filter(_) => {
                let mut batch_scores: Vec<f32> = Vec::new();
                if is_score {
                    batch_scores.reserve_exact(worker.budget.max_batch_rows() as usize);
                }
                let mut placed_again = false;
                let mut scored_before = 0u64;
                let scan_cost = loop {
                    let read = scan(worker, &snapshot, held, &mut |logs| {
                        if let Ready::Score(scorer) = &ready {
                            batch_scores.clear();
                            if logs.len() > batch_scores.capacity() {
                                return Err(failed("a batch has more rows than the largest batch"));
                            }
                            scorer.score(logs, &mut batch_scores)?;
                            if batch_scores.len() != logs.len() {
                                return Err(failed(
                                    "the scorer gave another number of scores than rows",
                                ));
                            }
                            scored += logs.len() as u64;
                        }
                        for (at, (text, position)) in rows_of(logs)?.enumerate() {
                            let row = scope.row_near(position, &mut near)?.ok_or_else(|| {
                                drift(format!("the position {position} names no original"))
                            })?;
                            // Outside the population the producer does not evaluate: it
                            // stays unknown.
                            if parent.state(row)? != State::In {
                                continue;
                            }
                            match &ready {
                                Ready::Filter(matcher) => {
                                    sink.put(row, u8::from(matcher.is_match(text)), 0)?;
                                }
                                _ => {
                                    let score = batch_scores[at];
                                    // A score that is no number is no evaluation.
                                    let code = if !score.is_finite() {
                                        State::Unknown as u8
                                    } else {
                                        match &task.producer {
                                            SelectProducer::Score {
                                                rule: Rule::Threshold(threshold),
                                                ..
                                            } => u8::from(f64::from(score) >= *threshold),
                                            _ => {
                                                best.offer(Kept {
                                                    score,
                                                    position,
                                                    row,
                                                });
                                                State::Out as u8
                                            }
                                        }
                                    };
                                    sink.put(row, code, score.to_bits())?;
                                }
                            }
                        }
                        Ok(true)
                    });
                    // What was scored was scored, also by a read that did not end.
                    worker
                        .shared
                        .activity
                        .rows_scored
                        .fetch_add(scored, Ordering::Relaxed);
                    match read {
                        Err(L2dError::Refused { reason })
                            if !on_disk && !worker.cancel.load(Ordering::Relaxed) =>
                        {
                            // With the tables in memory the read of the source was
                            // refused. On disk they leave it more: the tables are let
                            // go, and the source is read again into a spool. When that
                            // does not fit either, the first refusal stands.
                            let nothing = Sink::Direct {
                                states: States::in_memory(0, State::Out),
                                scores: None,
                            };
                            drop(std::mem::replace(&mut sink, nothing));
                            worker.budget.hold(fixed_held);
                            let sort_bytes = Sink::least_sort_bytes(rows);
                            let share = Sink::spool_share(sort_bytes);
                            if worker.budget.require_whole(share, ON_DISK).is_err() {
                                return Err(Fault::refused(reason));
                            }
                            held = fixed_held.saturating_add(share);
                            worker.budget.hold(held);
                            sink = Sink::create(worker, rows, is_score, true, sort_bytes)?;
                            // The candidates and the run of rows are part of the
                            // fixed share, which is charged once: they are emptied
                            // and used again, not made a second time.
                            best.clear();
                            scored_before += scored;
                            scored = 0;
                            on_disk = true;
                            placed_again = true;
                        }
                        read => {
                            break read?.ok_or_else(|| {
                                Fault::runtime("a read for a selection ended early")
                            })?;
                        }
                    }
                };
                cost.set("read_again_for_disk", placed_again);
                cost.scanned(&scan_cost)
                    .set("rows_scored", scored_before + scored)
                    .set("index_rows_read", rows);
                if placed_again {
                    // The read that was refused started too: both reads are counted
                    // in every counter, as the server's work counters count them.
                    cost.set("archive_scans", 2)
                        .set("bytes_unpacked", 2 * scan_cost.unpacked);
                }
            }
        }
        drop(ready);
        worker.check_cancel()?;
        let named = match &task.producer {
            SelectProducer::Membership { name, .. } => name.as_str(),
            _ => "",
        };
        let (states, scores) = sink.finish(&|first, second| L2dError::Refused {
            reason: format!(
                "line {} of the membership `{named}` gives a record another state than line \
                 {}; nothing was committed",
                first.max(second),
                first.min(second)
            ),
        })?;
        if on_disk {
            let spilled = cost.fields["spilled_bytes"].as_u64().unwrap_or(0);
            cost.set(
                "disk_peak_bytes",
                Sink::disk_peak(rows, is_score).saturating_add(spilled),
            );
        }
        (states, scores.map(Arc::new), is_score, on_disk)
    };
    for Reverse(kept) in best.kept.drain() {
        states.set(kept.row, State::In as u8)?;
    }
    drop(best);
    if membership_hashed > 0 {
        // Every byte of the file that was read: once to bind its content, and once
        // more when its lines were read.
        cost.set("membership_bytes_read", membership_hashed);
    }
    worker.check_cancel()?;
    debug_assert_eq!(on_disk, states.is_on_disk());
    let reason = match &producer {
        Producer::Membership { .. } => "outside_producer_coverage",
        _ => "missing_model_output",
    };
    let (result, set) = seal(
        worker,
        Sealed {
            scope: &task.scope,
            handle,
            described,
            producer,
            states,
            scores,
            owns_scores,
            cost,
            reason,
            extra: Map::new(),
        },
    )?;
    worker.budget.hold(worker.base.saturating_add(set.held));
    Ok(Done {
        result,
        commit: Some(Commit::Set(set)),
    })
}

/// A new selection whose states are known: what is left is to count it, state it and
/// keep it.
pub(crate) struct Sealed<'a> {
    pub scope: &'a Population,
    pub handle: String,
    pub described: Value,
    pub producer: Producer,
    pub states: States,
    pub scores: Option<Arc<Scores>>,
    pub owns_scores: bool,
    pub cost: Cost,
    /// Why a member of the scope the producer gave no state is not evaluated.
    pub reason: &'static str,
    /// What the producer adds to the result of the call.
    pub extra: Map<String, Value>,
}

/// Counts a new selection within its scope, builds the result of the call that made
/// it and returns both; the caller commits the set. The caller holds [`TALLY_BYTES`]
/// and what the set keeps.
pub(crate) fn seal(
    worker: &mut Worker<'_>,
    sealed: Sealed<'_>,
) -> Result<(String, Arc<Set>), Fault> {
    let Sealed {
        scope,
        handle,
        described,
        producer,
        states,
        scores,
        owns_scores,
        mut cost,
        reason,
        extra,
    } = sealed;
    let snapshot = scope.snapshot().clone();
    let (tally, revision, outside) = tally(&snapshot, scope, &states)?;
    let (parent_originals, parent_occurrences) = tally.parent();
    if (parent_originals, parent_occurrences) != scope.size() {
        return Err(Fault::runtime(
            "the members of a selection's parent do not add up to the parent",
        ));
    }
    let disk_bytes = states.disk_bytes()
        + scores
            .as_ref()
            .filter(|_| owns_scores)
            .map_or(0, |scores| scores.disk_bytes());
    let peak = cost.fields["disk_peak_bytes"].as_u64().unwrap_or(0);
    cost.set("candidates", parent_originals)
        .set("output", tally.of(State::In).0)
        .set("tables_on_disk", states.is_on_disk())
        .set("disk_bytes", disk_bytes)
        .set("disk_peak_bytes", peak.max(disk_bytes));
    let coverage = coverage(&tally, reason);
    let mut result = json!({
        "v": VERSION,
        "status": "ready",
        "set": handle,
        "snapshot": snapshot.handle,
        "scope": scope.handle(),
        "producer": described,
        "revision": revision,
        "counts": tally.counts(),
        "occurrence_counts": tally.occurrence_counts(),
        "coverage": coverage,
        "outside_scope": {
            "distinct_originals": outside.0,
            "occurrences": outside.1,
            "state": State::Unknown.name(),
            "reason": "outside_producer_scope",
        },
        "complete": tally.of(State::Unknown).0 == 0 && snapshot.scope.completeness().complete,
        "evidence": {
            "tool": "triager_inspect",
            "handle": handle,
            "views": ["records", "evidence"],
        },
        "cost": cost.finish(worker),
    });
    if let Some(fields) = result.as_object_mut() {
        fields.extend(extra);
    }
    let result = wire::with_response_bytes(result);
    require_fit(&result)?;
    let kept = states
        .held_bytes()
        .saturating_add(
            scores
                .as_ref()
                .filter(|_| owns_scores)
                .map_or(0, |scores| scores.held_bytes()),
        )
        .saturating_add(producer.held_bytes())
        .saturating_add(2 * result.len() as u64 + handle.capacity() as u64)
        .saturating_add(ARTIFACT_FIXED_BYTES);
    let set = Set {
        handle,
        kind: SetKind::Selection,
        parent: scope.as_set().cloned(),
        snapshot,
        states,
        scores,
        owns_scores,
        producer,
        tally,
        revision,
        result: result.clone(),
        held: kept,
        inputs: Vec::new(),
        allocation: None,
    };
    Ok((result, Arc::new(set)))
}

/// A selection whose producer names its records itself: a pattern of a refinement,
/// the neighbours of an exemplar, the decisions of a saved policy report or the
/// judgments collected for a run.
pub(crate) struct Listed<'a> {
    pub scope: &'a Population,
    pub producer: Producer,
    /// Whether a value is kept beside every named record, as a score is.
    pub with_values: bool,
    /// Why a member of the scope the producer does not name is not evaluated.
    pub reason: &'static str,
    pub extra: Map<String, Value>,
    /// What the caller holds beside `worker.base` while the records are fed: the
    /// loaded artifact they come from.
    pub held: u64,
    pub cost: Cost,
}

/// The handle of the selection `producer` makes in `scope`, and the set when it was
/// made before: the same producer revision over the same population is the same set.
pub(crate) fn known_selection(
    worker: &Worker<'_>,
    scope: &Population,
    producer: &Producer,
) -> (String, Option<Arc<Set>>) {
    let handle = wire::handle(
        "set",
        &worker.shared.config.project,
        &[
            &scope.snapshot().handle,
            scope.handle(),
            &producer.describe().to_string(),
        ],
    );
    let known = registry(worker.shared).set(&handle);
    (handle, known)
}

/// Takes one named record of a listed selection: its position, the code of its state
/// and the value kept beside it.
pub(crate) type Put<'a> = &'a mut dyn FnMut(u64, u8, f32) -> Result<(), L2dError>;

/// What [`select_listed`] holds beside its tables and the caller's share: the run of
/// rows the named records are found in, the blocks its members are counted from, the
/// result and the artifact's own structure.
pub(crate) const LISTED_FIXED_BYTES: u64 =
    RowBlock::BYTES + TALLY_BYTES + ARTIFACT_FIXED_BYTES + 64 * 1024;

/// Makes the selection of a producer that names its records: `feed` gives every named
/// record's position, the code of its state and the value kept beside it. A record
/// outside the scope fills no coverage, and one the producer does not name is not
/// evaluated. Returns the result and, unless the set was made before, the set to
/// commit.
///
/// The tables are kept in memory when the budget admits them beside what the caller
/// holds and on disk otherwise, as those of every selection. A position that names no
/// original of the snapshot is refused.
pub(crate) fn select_listed(
    worker: &mut Worker<'_>,
    listed: Listed<'_>,
    feed: &mut dyn FnMut(Put<'_>) -> Result<(), L2dError>,
) -> Result<(String, Option<Arc<Set>>), Fault> {
    let Listed {
        scope,
        producer,
        with_values,
        reason,
        extra,
        held,
        mut cost,
    } = listed;
    let (handle, known) = known_selection(worker, scope, &producer);
    if let Some(known) = known {
        return Ok((known.result.clone(), None));
    }
    let snapshot = scope.snapshot().clone();
    let rows = snapshot.scope.originals();
    let fixed = LISTED_FIXED_BYTES.saturating_add(producer.held_bytes());
    let beside = worker.base.saturating_add(held);
    worker.budget.hold(beside);
    worker.budget.require_whole(
        fixed,
        "making a selection from the records its producer names: the rows they are found \
         in and the blocks its members are counted from,",
    )?;
    worker.budget.hold(beside.saturating_add(fixed));
    let (tables, on_disk, sort_bytes) = Sink::plan(worker, rows, with_values);
    worker.budget.require_whole(
        tables,
        if on_disk {
            "placing the states of every original on disk: the spool's buffer and the \
             memory its rows are sorted in,"
        } else {
            "keeping the states of every original,"
        },
    )?;
    worker
        .budget
        .hold(beside.saturating_add(fixed).saturating_add(tables));
    let mut sink = Sink::create(worker, rows, with_values, on_disk, sort_bytes)?;
    let mut near = RowBlock::new();
    let mut named = 0u64;
    feed(&mut |position, code, value| {
        let row = snapshot
            .scope
            .row_near(position, &mut near)?
            .ok_or_else(|| L2dError::Refused {
                reason: format!(
                    "the producer names the record {}, which is no original of the snapshot; \
                     nothing was committed",
                    wire::record_id(position)
                ),
            })?;
        named += 1;
        if scope.state(row)? != State::In {
            return Ok(());
        }
        sink.put(row, code, value.to_bits())
    })?;
    worker.check_cancel()?;
    let (states, scores) = sink.finish(&|_, _| L2dError::Runtime {
        detail: "a producer named one record with two states".into(),
    })?;
    cost.set("index_rows_read", named);
    if on_disk {
        cost.set("disk_peak_bytes", Sink::disk_peak(rows, with_values));
    }
    let described = producer.describe();
    let (result, set) = seal(
        worker,
        Sealed {
            scope,
            handle,
            described,
            producer,
            states,
            scores: scores.map(Arc::new),
            owns_scores: with_values,
            cost,
            reason,
            extra,
        },
    )?;
    worker.budget.hold(worker.base.saturating_add(set.held));
    Ok((result, Some(set)))
}

/// Reads the membership file at `path` line by line and gives `each` the line number,
/// the position its record handle names and the code of its state. Returns the
/// SHA-256 and the length of what was read. The caller holds
/// [`MEMBERSHIP_READ_BYTES`].
///
/// The file is JSON lines: a line is one object `{"id": ..., "state": ...}` with the
/// record handle as the records view returns it and one of `in`, `out` and `abstain`,
/// and nothing else, at most [`MAX_MEMBERSHIP_LINE`] bytes. A record the file does not
/// name was not assessed. The tab-separated form of task 1.3 is refused like any line
/// that is no such object.
fn read_membership(
    path: &Path,
    name: &str,
    each: &mut dyn FnMut(u64, u64, u8) -> Result<(), L2dError>,
) -> Result<(String, u64), L2dError> {
    let file = File::open(path).map_err(|error| membership_unread(name, "opening", &error))?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut line: Vec<u8> = Vec::with_capacity(MAX_MEMBERSHIP_LINE + 2);
    let mut number = 0u64;
    loop {
        line.clear();
        let read = (&mut reader)
            .take(MAX_MEMBERSHIP_LINE as u64 + 1)
            .read_until(b'\n', &mut line)
            .map_err(|error| membership_unread(name, "reading", &error))?;
        if read == 0 {
            return Ok((format!("{:x}", hash.finalize()), bytes));
        }
        number += 1;
        bytes += read as u64;
        hash.update(&line);
        let refused = |what: &str| L2dError::Refused {
            reason: format!(
                "line {number} of the membership `{name}` {what}; a membership file is JSON \
                 lines, one object {{\"id\": a record handle (`r-` and 16 hexadecimal \
                 digits), \"state\": \"in\", \"out\" or \"abstain\"}} per line (the \
                 tab-separated form is not read). Nothing was committed"
            ),
        };
        let Some(text) = line.strip_suffix(b"\n").or_else(|| {
            // The last line may end without a line feed; any other is too long.
            (line.len() <= MAX_MEMBERSHIP_LINE).then_some(&line[..])
        }) else {
            return Err(refused(&format!(
                "is longer than {MAX_MEMBERSHIP_LINE} bytes"
            )));
        };
        let text = text.strip_suffix(b"\r").unwrap_or(text);
        let (position, code) =
            membership_line(text).ok_or_else(|| refused("is no membership line"))?;
        each(number, position, code)?;
    }
}

/// One line of a membership file as it is read: both fields are borrowed from the
/// line, so parsing it allocates nothing, and a field that needs an escape is refused.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MembershipLine<'a> {
    id: &'a str,
    state: &'a str,
}

/// The position and the code of one line of a membership file; `None` when the line
/// is no such object.
fn membership_line(text: &[u8]) -> Option<(u64, u8)> {
    // A struct also reads from a list of its fields: only an object is a line.
    if text.trim_ascii_start().first() != Some(&b'{') {
        return None;
    }
    let line: MembershipLine<'_> = serde_json::from_slice(text).ok()?;
    let code = match line.state {
        "in" => State::In as u8,
        "out" => State::Out as u8,
        "abstain" => ABSTAINED,
        _ => return None,
    };
    Some((wire::record_position(line.id)?, code))
}

// --- Compose -------------------------------------------------------------------------------

pub(crate) struct ComposeTask {
    pub snapshot: Arc<Snapshot>,
    pub scope: Population,
    /// The named operands, in the order the expression indexes them.
    pub sets: Vec<(String, Arc<Set>)>,
    pub expr: Expr,
}

pub(crate) fn compose(worker: &mut Worker<'_>, task: &ComposeTask) -> Result<Done, Fault> {
    let started = Instant::now();
    let snapshot = &task.snapshot;
    let rows = snapshot.scope.originals();
    let canonical = task
        .expr
        .describe(&|index| json!(task.sets[index].1.handle));
    let handle = wire::handle(
        "view",
        &worker.shared.config.project,
        &[
            &snapshot.handle,
            task.scope.handle(),
            &canonical.to_string(),
        ],
    );
    if let Some(known) = registry(worker.shared).set(&handle) {
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }
    let producer = Producer::Expression {
        expr: canonical.clone(),
    };
    // A block of every operand, of the parent and of the view at a time, whether the
    // sets are in memory or on disk.
    let blocks = (task.sets.len() as u64 + 2) * States::BLOCK_BYTES as u64;
    let fixed = blocks
        .saturating_add(COMPOSE_FIXED_BYTES + ARTIFACT_FIXED_BYTES)
        .saturating_add(producer.held_bytes());
    worker.budget.hold(worker.base);
    worker.budget.require_whole(
        fixed,
        "composing a view: a block of every operand at a time,",
    )?;
    worker.budget.hold(worker.base.saturating_add(fixed));
    let on_disk = tables_on_disk(worker, States::bytes_for(rows));
    if !on_disk {
        worker
            .budget
            .hold(worker.base.saturating_add(fixed + States::bytes_for(rows)));
    }
    // The view is its parent and the expression: outside the parent nothing is in.
    let mut states = if on_disk {
        let dir = set_dir(
            worker,
            States::bytes_for(rows),
            "placing the membership of a view on disk",
        )?;
        States::on_disk(&dir, rows)?
    } else {
        States::in_memory(rows, State::Out)
    };
    let mut within = vec![0u8; States::BLOCK_BYTES];
    let mut codes = vec![0u8; States::BLOCK_BYTES];
    let mut operands: Vec<Vec<u8>> = task
        .sets
        .iter()
        .map(|_| vec![0u8; States::BLOCK_BYTES])
        .collect();
    let mut first = 0;
    while first < rows {
        worker.check_cancel()?;
        let in_block = task.scope.read_block(first, &mut within)?;
        for ((_, set), block) in task.sets.iter().zip(&mut operands) {
            set.states.read_block(first, block)?;
        }
        for at in 0..in_block {
            let parent = State::of(code_in(&within, at));
            let state = if parent == State::Out {
                State::Out
            } else {
                // Among the members of the parent the view's state is the
                // expression's.
                parent.and(
                    task.expr
                        .eval(&|index| State::of(code_in(&operands[index], at))),
                )
            };
            put_code(&mut codes, at, state as u8);
        }
        states.write_block(first, &codes, in_block)?;
        first += BLOCK_ROWS;
    }
    drop(operands);
    let (tally, revision, _) = tally(snapshot, &task.scope, &states)?;
    let (parent_originals, parent_occurrences) = tally.parent();
    if (parent_originals, parent_occurrences) != task.scope.size() {
        return Err(Fault::runtime(
            "the members of a view's parent do not add up to the parent",
        ));
    }
    let named: serde_json::Map<String, Value> = task
        .sets
        .iter()
        .map(|(name, set)| (name.clone(), json!(set.handle)))
        .collect();
    let (kept_originals, kept_occurrences) = tally.of(State::In);
    let mut cost = Cost::new(started);
    cost.set("index_rows_read", rows)
        .set("candidates", parent_originals)
        .set("output", kept_originals)
        .set("tables_on_disk", on_disk)
        .set("disk_bytes", states.disk_bytes())
        .set("disk_peak_bytes", states.disk_bytes());
    let result = wire::with_response_bytes(json!({
        "v": VERSION,
        "status": "ready",
        "view": handle,
        "snapshot": snapshot.handle,
        "parent": task.scope.handle(),
        "operation": {"mode": "view", "expr": canonical, "sets": named},
        "revision": revision,
        "counts": tally.counts(),
        "occurrence_counts": tally.occurrence_counts(),
        "coverage": coverage(&tally, "operand_not_evaluated"),
        "complete": tally.of(State::Unknown).0 == 0 && snapshot.scope.completeness().complete,
        "root": {
            "snapshot": snapshot.handle,
            "distinct_originals": snapshot.scope.originals(),
            "occurrences": snapshot.scope.occurrences(),
            "retained_share": share_of(kept_originals, snapshot.scope.originals()),
            "occurrence_retained_share": share_of(kept_occurrences, snapshot.scope.occurrences()),
        },
        "cost": cost.finish(worker),
    }));
    require_fit(&result)?;
    let kept = states
        .held_bytes()
        .saturating_add(producer.held_bytes())
        .saturating_add(2 * result.len() as u64 + handle.capacity() as u64)
        .saturating_add(ARTIFACT_FIXED_BYTES);
    let set = Set {
        handle,
        kind: SetKind::View,
        parent: task.scope.as_set().cloned(),
        snapshot: snapshot.clone(),
        states,
        scores: None,
        owns_scores: false,
        producer,
        tally,
        revision,
        result: result.clone(),
        held: kept,
        inputs: task.sets.iter().map(|(_, set)| set.clone()).collect(),
        allocation: None,
    };
    worker.budget.hold(worker.base.saturating_add(kept));
    Ok(Done {
        result,
        commit: Some(Commit::Set(Arc::new(set))),
    })
}

pub(crate) fn share_of(part: u64, parent: u64) -> Value {
    share(part, parent)
}

/// An artifact whose result would not fit the response limit is not committed: the
/// result is what every later call about it returns.
pub(crate) fn require_fit(result: &str) -> Result<(), Fault> {
    if wire::transport_bytes(result) > wire::PAYLOAD_TRANSPORT_BYTES {
        return Err(Fault::runtime(
            "the result of this call does not fit the response limit; nothing was committed",
        ));
    }
    Ok(())
}

/// The buffer a sort writes through is the least a spooled selection sorts with.
const _: () = assert!(MIN_SORT_BYTES > IO_BYTES);

#[cfg(test)]
mod tests {
    use super::*;

    /// The code and the score the test puts for a row; every fifth row is not put.
    fn put_for(row: u64) -> Option<(u8, f32)> {
        (row % 5 != 4).then(|| ([0, 1, ABSTAINED][(row % 3) as usize], row as f32 / 8.0))
    }

    /// A selection that goes to disk: rows put in another order than their rows,
    /// with rows that are never put, are sorted and placed within the share the
    /// selection asks the budget for, and read as they were put.
    #[test]
    fn a_spooled_selection_is_placed_within_its_share_and_reads_as_it_was_put() {
        let root = tempfile::tempdir().unwrap();
        let dir = SetDir::create(root.path().join("set-1")).unwrap();
        let rows = 3 * BLOCK_ROWS + 77;
        // A megabyte of spooled rows in the least memory a sort takes: six runs.
        let sort_bytes = MIN_SORT_BYTES;
        let ((states, scores), peak) = crate::test_heap::measure(|| {
            let mut sink = Sink::spool(dir.clone(), rows, true, sort_bytes).unwrap();
            for at in 0..rows {
                let row = (at * 7_919 + 13) % rows;
                if let Some((code, score)) = put_for(row) {
                    sink.put(row, code, score.to_bits()).unwrap();
                }
            }
            // A row put twice with one code is one row.
            let (code, score) = put_for(10).unwrap();
            sink.put(10, code, score.to_bits()).unwrap();
            assert_eq!(sink.code(10).unwrap(), None);
            sink.finish(&|_, _| panic!("no row was put with two codes"))
                .unwrap()
        });
        assert!(
            peak <= Sink::spool_share(sort_bytes),
            "the heap peaked at {peak} bytes, and the selection asks for {}",
            Sink::spool_share(sort_bytes)
        );
        let scores = scores.unwrap();
        assert!(states.is_on_disk());
        assert_eq!(states.held_bytes() + scores.held_bytes(), 0);
        for row in (0..rows)
            .step_by(97)
            .chain([0, 4, 10, BLOCK_ROWS - 1, BLOCK_ROWS, rows - 1])
        {
            match put_for(row) {
                Some((code, score)) => {
                    assert_eq!(states.code(row).unwrap(), code, "row {row}");
                    assert_eq!(scores.get(row).unwrap(), score, "row {row}");
                }
                None => {
                    assert_eq!(states.code(row).unwrap(), State::Unknown as u8, "row {row}");
                    assert!(scores.get(row).unwrap().is_nan(), "row {row}");
                }
            }
        }
        // Only the tables are left in the directory, and it goes with them.
        let mut left: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|file| file.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, ["scores.bin", "states.bin"]);
        let path = dir.path().to_path_buf();
        drop((dir, states, scores));
        assert!(!path.exists());
    }

    /// Tables that the budget does not admit beside what the operation holds go to
    /// disk; tables it admits stay in memory unless the host placed them on disk.
    #[test]
    fn tables_go_to_disk_when_the_budget_does_not_admit_them() {
        const MIB: u64 = 1024 * 1024;
        let settings = l2d_core::MemorySettings {
            memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + MIB),
            ..l2d_core::MemorySettings::default()
        };
        let mut budget = MemoryBudget::observe_with(&settings, || Ok(64 << 30)).unwrap();
        let auto = TablePlacement::MemoryWhenAdmitted;
        assert!(!place_on_disk(auto, &budget, MIB));
        assert!(place_on_disk(auto, &budget, MIB + 1));
        assert!(place_on_disk(TablePlacement::Disk, &budget, 1));
        // What the operation holds already counts.
        budget.hold(MIB / 2);
        assert!(!place_on_disk(auto, &budget, MIB / 2));
        assert!(place_on_disk(auto, &budget, MIB / 2 + 1));
    }

    /// A selection that reads its source again empties its candidates and fills the
    /// same list: the budget was asked for one list, and a second one beside it
    /// would be 24 bytes a candidate outside the budget (review round 1 of task 1.3).
    #[test]
    fn candidates_are_emptied_and_used_again_without_a_second_list() {
        const MOST: u64 = 50_000;
        let mut best = Best::new(MOST);
        let fill = |best: &mut Best| {
            for row in 0..2 * MOST {
                best.offer(Kept {
                    score: (row % 977) as f32,
                    position: row,
                    row,
                });
            }
        };
        fill(&mut best);
        assert_eq!(best.kept.len() as u64, MOST);
        let ((), peak) = crate::test_heap::measure(|| {
            best.clear();
            fill(&mut best);
        });
        assert_eq!(peak, 0, "emptying and filling again allocated {peak} bytes");
        assert_eq!(best.kept.len() as u64, MOST);
        // The best are kept: the highest score, the lowest position among equals.
        let worst = best.kept.peek().unwrap();
        assert!(worst.0.score >= 480.0, "{}", worst.0.score);
        // The least memory a spool is sorted in merges its runs, also for many rows.
        assert_eq!(Sink::least_sort_bytes(150_000), MIN_SORT_BYTES);
        for rows in [2_200_000u64, 100_000_000, 2_000_000_000] {
            let memory = Sink::least_sort_bytes(rows);
            let room = memory - IO_BYTES - 4096;
            let runs = (rows * 20).div_ceil(room);
            assert!(
                room / runs >= 2 * 256,
                "{rows} rows: {runs} runs in {room} bytes"
            );
        }
    }

    #[test]
    fn a_row_spooled_with_two_codes_is_refused_and_leaves_no_file() {
        let root = tempfile::tempdir().unwrap();
        let dir = SetDir::create(root.path().join("set-1")).unwrap();
        let path = dir.path().to_path_buf();
        let mut sink = Sink::spool(dir, 100, false, MIN_SORT_BYTES).unwrap();
        sink.put(7, State::In as u8, 3).unwrap();
        sink.put(2, State::Out as u8, 5).unwrap();
        sink.put(7, State::Out as u8, 9).unwrap();
        let refused = sink.finish(&|first, second| L2dError::Refused {
            reason: format!("lines {} and {}", first.min(second), first.max(second)),
        });
        match refused {
            Err(L2dError::Refused { reason }) => assert_eq!(reason, "lines 3 and 9"),
            Err(other) => panic!("{other:?}"),
            Ok(_) => panic!("two codes for one row were accepted"),
        }
        assert!(!path.exists(), "the spool and its directory are removed");
        // A sink that is let go before it is finished leaves nothing either.
        let dir = SetDir::create(root.path().join("set-2")).unwrap();
        let path = dir.path().to_path_buf();
        let mut sink = Sink::spool(dir, 100, true, MIN_SORT_BYTES).unwrap();
        sink.put(1, State::In as u8, 0).unwrap();
        drop(sink);
        assert!(!path.exists());
    }

    #[test]
    fn a_membership_file_is_read_line_by_line_and_refused_where_it_is_none() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("membership.jsonl");
        let read = |content: &[u8]| {
            std::fs::write(&path, content).unwrap();
            let mut lines = Vec::new();
            let hashed = read_membership(&path, "c", &mut |line, position, code| {
                lines.push((line, position, code));
                Ok(())
            });
            hashed.map(|hashed| (lines, hashed))
        };
        let content = b"{\"id\":\"r-0000000000000007\",\"state\":\"in\"}\r\n\
                        { \"state\": \"abstain\", \"id\": \"r-00000100000000ff\" }\n\
                        {\"id\":\"r-0000000000000009\",\"state\":\"out\"}";
        let (lines, hashed) = read(content).unwrap();
        assert_eq!(
            lines,
            [
                (1, 7, State::In as u8),
                (2, (1 << 40) | 0xff, ABSTAINED),
                (3, 9, State::Out as u8)
            ]
        );
        assert_eq!(hashed, hash_membership(&path, "c").unwrap());
        assert_eq!(hashed.1, content.len() as u64);
        assert_eq!(read(b"").unwrap().0, []);
        // The longest line that is taken, and one byte more.
        let padded = |pad: usize| {
            let mut line = format!(
                "{{\"id\":\"r-0000000000000007\",\"state\":\"in\"{}}}",
                " ".repeat(pad)
            );
            line.push('\n');
            line
        };
        let longest = padded(MAX_MEMBERSHIP_LINE - padded(0).len() + 1);
        assert_eq!(longest.len(), MAX_MEMBERSHIP_LINE + 1);
        assert_eq!(
            read(longest.as_bytes()).unwrap().0,
            [(1, 7, State::In as u8)]
        );
        let too_long = padded(MAX_MEMBERSHIP_LINE - padded(0).len() + 2);
        let good = "{\"id\":\"r-0000000000000007\",\"state\":\"in\"}\n";
        let refused: Vec<(Vec<u8>, &str)> = vec![
            (format!("{good}\n").into_bytes(), "line 2"),
            // The tab-separated form of task 1.3 is no longer read.
            (b"r-0000000000000007\tin\n".to_vec(), "JSON lines"),
            (
                b"{\"id\":\"r-0000000000000007\",\"state\":\"IN\"}\n".to_vec(),
                "line 1",
            ),
            (b"{\"id\":\"r-7\",\"state\":\"in\"}\n".to_vec(), "line 1"),
            (b"{\"id\":\"r-0000000000000007\"}\n".to_vec(), "line 1"),
            (
                b"{\"id\":\"r-0000000000000007\",\"state\":\"in\",\"why\":1}\n".to_vec(),
                "line 1",
            ),
            (
                b"{\"id\":\"r-0000000000000007\",\"state\":\"in\",\"state\":\"out\"}\n".to_vec(),
                "line 1",
            ),
            (b"[\"r-0000000000000007\",\"in\"]\n".to_vec(), "line 1"),
            (
                b"{\"id\":\"r-000000000000000\\u0037\",\"state\":\"in\"}\n".to_vec(),
                "line 1",
            ),
            (format!("{good}{too_long}").into_bytes(), "line 2"),
            (
                b"{\"id\":\"r-0000000000000007\",\"state\":\"\xff\"}\n".to_vec(),
                "line 1",
            ),
        ];
        for (content, needle) in refused {
            match read(&content) {
                Err(L2dError::Refused { reason }) => {
                    assert!(reason.contains(needle), "{reason}");
                    assert!(reason.contains("`c`"), "{reason}");
                }
                other => panic!("{}: {other:?}", String::from_utf8_lossy(&content)),
            }
        }
    }

    /// Reading a membership line allocates nothing: the line buffer is the reader's,
    /// and the fields are borrowed from it.
    #[test]
    fn a_membership_line_is_parsed_without_an_allocation() {
        let line = b"{\"id\":\"r-00000100000000ff\",\"state\":\"abstain\"}";
        let (parsed, peak) = crate::test_heap::measure(|| membership_line(line));
        assert_eq!(parsed, Some(((1 << 40) | 0xff, ABSTAINED)));
        assert_eq!(peak, 0, "parsing a line allocated {peak} bytes");
    }
}
