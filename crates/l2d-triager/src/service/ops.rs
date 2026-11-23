//! The operations that make an artifact: opening a snapshot, selecting a set by a
//! score or a filter, and composing a view. They run on the service's worker, one at
//! a time, against the one memory budget.
//!
//! # Memory
//!
//! Every operation asks the budget for all it holds before it allocates, as a constant
//! share that is held for the whole operation, so the archive's batches are sized with
//! it from the first one on and no share grows between batches:
//!
//! - a selection: the states of every original (two bits each); for a score the score
//!   of every original (four bytes each), the kept candidates of a top-k (24 bytes
//!   each), the scores of one batch, the loaded scorer and what it needs to score the
//!   longest record; for a filter the compiled pattern; and [`SCAN_FIXED_BYTES`];
//! - a view: its states and [`COMPOSE_FIXED_BYTES`];
//! - a snapshot: what `Scope::open_archive` asks for.
//!
//! What a committed artifact keeps is held from its commit on, for as long as the
//! service lives. A share that does not fit is refused with the figures.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use l2d_core::marks::REGEX_HELD_BYTES;
use l2d_core::polars::prelude::Column;
use l2d_core::selections::{SearchQuery, compile};
use l2d_core::zip_input::DiskDedup;
use l2d_core::{FreezeDiskCheck, L2dError, LogsFrame, MemoryBudget, ZipInput, ZipInputSettings};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::membership::{Expr, State, States, Tally};
use super::state::{Population, Producer, Progress, Set, SetKind, Shared, Snapshot};
use super::wire::{self, Fault, MAX_MESSAGE_BYTES, VERSION, cancelled, share};
use super::{FrozenScorer, ScorerSource};
use crate::scope::{Scope, ScopeSettings, archive_summary_bytes};

/// What reading a snapshot's source again holds beside the reader and the shares named
/// in the module description: the positions and texts of a batch on their way, the
/// summary the reader ends with, a result on its way out and an excerpt.
pub(crate) const SCAN_FIXED_BYTES: u64 = 1024 * 1024;
/// What composing a view holds beside its states: the expression and the result.
pub(crate) const COMPOSE_FIXED_BYTES: u64 = 64 * 1024;
/// What an artifact keeps beside its tables and strings: its own structure, its place
/// in the registry and the room strings may have beyond their content.
const ARTIFACT_FIXED_BYTES: u64 = 2 * 1024;
/// The most reasons for what a source left out that an open result lists.
const MAX_REASONS: usize = 24;

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
}

impl Worker<'_> {
    fn fresh_dir(&mut self, kind: &str) -> PathBuf {
        *self.dirs += 1;
        self.shared
            .config
            .work_root
            .join(format!("{kind}-{}", *self.dirs))
    }

    fn check_cancel(&self) -> Result<(), L2dError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(cancelled());
        }
        Ok(())
    }

    fn memory(&self) -> (u64, u64) {
        let report = self.shared.usage.report();
        (report.memory_limit_bytes, report.counted_peak_bytes)
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
}

impl Commit {
    pub fn held(&self) -> u64 {
        match self {
            Commit::Snapshot(snapshot) => snapshot.held,
            Commit::Set(set) => set.held,
        }
    }
}

/// A removed directory when the operation that made it ends, however it ends.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn failed(error: impl std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("reading a batch for a triager set failed: {error}"),
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

    let (limit, peak) = worker.memory();
    let mut snapshot = Snapshot {
        handle,
        public: task.public,
        path: task.path.clone(),
        zip: task.zip.clone(),
        result: String::new(),
        held: kept,
        summary,
        scope,
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
    let result = json!({
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
            "producers": ["score", "filter"],
            "score_rules": ["top_k", "threshold"],
            "ties": ["record_id_ascending"],
            "filter_syntax": ["literal", "glob", "regex"],
            "compose_modes": ["view"],
            "operators": ["union", "intersection", "difference", "complement"],
            "inspect_views": ["overview", "regions", "records", "evidence"],
            "sort": ["record_id_ascending"],
            "job_actions": ["status", "cancel"],
            "occurrence_locators": "first_only",
            "not_supported": [
                "triager_refine", "triager_save", "open receipt", "producer recommend",
                "filter syntax pattern", "compose mode allocate", "expression state",
                "inspect histogram", "inspect fields", "inspect lineage",
            ],
        },
        "disclosure": worker.shared.disclosure(&snapshot),
        "cost": {
            "archive_scans": 1,
            "records_scanned": completeness.records_scanned,
            "records_skipped": completeness.skipped_records,
            "entries_read": completeness.entries_read,
            "distinct_originals": scope.originals(),
            "rows_scored": 0,
            "bytes_packed": null,
            "bytes_unpacked": null,
            "dedup_on_disk": scope.dedup_on_disk(),
            "spilled_bytes": null,
            "index_on_disk": scope.index_on_disk(),
            "disk_bytes": scope.disk_bytes(),
            "entries_at_once": scope.archive_readers(),
            "elapsed_ms": started.elapsed().as_millis() as u64,
            "memory_limit_bytes": limit,
            "counted_peak_bytes": peak,
            "process_peak_bytes": null,
        },
    })
    .to_string();
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
    distinct: u64,
    packed: u64,
    unpacked: u64,
    dedup: Option<DiskDedup>,
    entries_at_once: usize,
    started: Instant,
}

impl ScanCost {
    pub fn describe(&self, worker: &Worker<'_>, rows_scored: u64, index_rows_read: u64) -> Value {
        let (limit, peak) = worker.memory();
        json!({
            "archive_scans": 1,
            "records_scanned": self.records,
            "records_skipped": self.skipped,
            "entries_read": self.entries_read,
            "distinct_originals": self.distinct,
            "rows_scored": rows_scored,
            "bytes_packed": self.packed,
            "bytes_unpacked": self.unpacked,
            "dedup_on_disk": self.dedup.is_some(),
            "spilled_bytes": self.dedup.map_or(0, |dedup| dedup.spilled_bytes),
            "spilled_records": self.dedup.map_or(0, |dedup| dedup.spilled_records),
            "index_rows_read": index_rows_read,
            "entries_at_once": self.entries_at_once,
            "elapsed_ms": self.started.elapsed().as_millis() as u64,
            "memory_limit_bytes": limit,
            "counted_peak_bytes": peak,
            "process_peak_bytes": null,
        })
    }
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
    let started = Instant::now();
    let shared = worker.shared;
    let config = &shared.config;
    worker.check_cancel()?;
    worker
        .shared
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
    Ok(Some(ScanCost {
        records: summary.input.records,
        skipped: summary.input.skipped.len() as u64,
        entries_read: summary.read().count() as u64,
        distinct: summary.input.distinct_texts,
        packed: selected.packed,
        unpacked: selected.unpacked,
        dedup,
        entries_at_once,
        started,
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

/// Counts the members of `parent` by their state in `states`, in both units, and
/// digests the membership. Also returns what lies outside the parent.
fn tally(
    snapshot: &Snapshot,
    parent: &Population,
    states: &States,
) -> Result<(Tally, String, (u64, u64)), L2dError> {
    let mut tally = Tally::default();
    let mut outside = (0u64, 0u64);
    let mut sum = 0u128;
    for row in 0..states.rows() {
        let original = snapshot.scope.row(row)?;
        let state = states.get(row);
        if parent.state(row) == State::In {
            tally.add(state, original.occurrences);
        } else {
            outside.0 += 1;
            outside.1 += original.occurrences;
        }
        if state != State::Out {
            // Order does not matter to a sum, so the digest is the same however the
            // source handed its originals out.
            let mut hash = Sha256::new();
            hash.update(original.id.0.to_le_bytes());
            hash.update([state as u8]);
            let mut part = [0u8; 16];
            part.copy_from_slice(&hash.finalize()[..16]);
            sum = sum.wrapping_add(u128::from_le_bytes(part));
        }
    }
    Ok((tally, format!("{sum:032x}"), outside))
}

/// The coverage block of a reduction: what was evaluated and what was not, with the
/// reason.
fn coverage(tally: &Tally, reason: &str) -> Value {
    let (unknown, unknown_occurrences) = tally.of(State::Unknown);
    let (parent, _) = tally.parent();
    let reasons = if unknown == 0 {
        json!([])
    } else {
        json!([{"reason": reason, "distinct_originals": unknown, "occurrences": unknown_occurrences}])
    };
    json!({
        "unit": "distinct_originals",
        "evaluated": parent - unknown,
        "not_evaluated": unknown,
        "reasons": reasons,
    })
}

pub(crate) fn select(worker: &mut Worker<'_>, task: &SelectTask) -> Result<Done, Fault> {
    let snapshot = task.scope.snapshot().clone();
    let rows = snapshot.scope.originals();
    let project = worker.shared.config.project.clone();
    worker.budget.hold(worker.base);

    // The producer is made ready first: its revision is part of the set's identity.
    enum Ready {
        Score(Box<dyn FrozenScorer>),
        Filter(regex::Regex),
    }
    let (ready, producer, producer_bytes) = match &task.producer {
        SelectProducer::Score { model, rule } => {
            let scorer = worker.scorers.load(model, worker.budget)?;
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

    // Everything the selection holds, asked for before any of it is allocated.
    let (parent_size, _) = task.scope.size();
    let kept_candidates = match &task.producer {
        SelectProducer::Score {
            rule: Rule::TopK(k),
            ..
        } => (*k).min(parent_size),
        _ => 0,
    };
    let is_score = matches!(ready, Ready::Score(_));
    let score_bytes = if is_score {
        rows.saturating_mul(4)
            .saturating_add(worker.budget.max_batch_rows().saturating_mul(4))
    } else {
        0
    };
    let tables = States::bytes_for(rows)
        .saturating_add(score_bytes)
        .saturating_add(kept_candidates.saturating_mul(KEPT_BYTES))
        .saturating_add(SCAN_FIXED_BYTES + ARTIFACT_FIXED_BYTES)
        .saturating_add(producer.held_bytes());
    // Beside the producer, which is held already: when that alone is more than the
    // limit leaves, nothing is left for these and the refusal names what is held.
    worker.budget.require_whole(
        tables,
        "selecting a set: the states and scores of every original and one read of the \
         source, beside the producer,",
    )?;
    let held = worker
        .base
        .saturating_add(producer_bytes)
        .saturating_add(tables);
    worker.budget.hold(held);

    let mut states = States::new(rows, State::Unknown);
    let mut scores: Option<Vec<f32>> = is_score.then(|| vec![f32::NAN; rows as usize]);
    let mut batch_scores: Vec<f32> = Vec::new();
    if is_score {
        batch_scores.reserve_exact(worker.budget.max_batch_rows() as usize);
    }
    let mut best: BinaryHeap<Reverse<Kept>> = BinaryHeap::with_capacity(kept_candidates as usize);
    let mut scored = 0u64;
    let parent = task.scope.clone();
    let scope = &snapshot.scope;
    let cost = scan(worker, &snapshot, held, &mut |logs| {
        if let Ready::Score(scorer) = &ready {
            batch_scores.clear();
            if logs.len() > batch_scores.capacity() {
                return Err(failed("a batch has more rows than the largest batch"));
            }
            scorer.score(logs, &mut batch_scores)?;
            if batch_scores.len() != logs.len() {
                return Err(failed("the scorer gave another number of scores than rows"));
            }
            scored += logs.len() as u64;
        }
        for (at, (text, position)) in rows_of(logs)?.enumerate() {
            let row = scope
                .row_of(position)?
                .ok_or_else(|| drift(format!("the position {position} names no original")))?;
            // Outside the population the producer does not evaluate: it stays unknown.
            if parent.state(row) != State::In {
                continue;
            }
            match &ready {
                Ready::Filter(matcher) => states.set(
                    row,
                    if matcher.is_match(text) {
                        State::In
                    } else {
                        State::Out
                    },
                ),
                Ready::Score(_) => {
                    let score = batch_scores[at];
                    if let Some(scores) = &mut scores {
                        scores[row as usize] = score;
                    }
                    // A score that is no number is no evaluation.
                    if !score.is_finite() {
                        continue;
                    }
                    states.set(row, State::Out);
                    match &task.producer {
                        SelectProducer::Score {
                            rule: Rule::Threshold(threshold),
                            ..
                        } => {
                            if f64::from(score) >= *threshold {
                                states.set(row, State::In);
                            }
                        }
                        _ => {
                            let candidate = Kept {
                                score,
                                position,
                                row,
                            };
                            if (best.len() as u64) < kept_candidates {
                                best.push(Reverse(candidate));
                            } else if best.peek().is_some_and(|worst| candidate > worst.0) {
                                best.pop();
                                best.push(Reverse(candidate));
                            }
                        }
                    }
                }
            }
        }
        Ok(true)
    })?
    .ok_or_else(|| Fault::runtime("a read for a selection ended early"))?;
    for Reverse(kept) in best.drain() {
        states.set(kept.row, State::In);
    }
    drop(best);
    drop(batch_scores);
    drop(ready);
    worker.check_cancel()?;

    let (tally, revision, outside) = tally(&snapshot, &task.scope, &states)?;
    let (parent_originals, parent_occurrences) = tally.parent();
    if (parent_originals, parent_occurrences) != task.scope.size() {
        return Err(Fault::runtime(
            "the members of a selection's parent do not add up to the parent",
        ));
    }
    let result = json!({
        "v": VERSION,
        "status": "ready",
        "set": handle,
        "snapshot": snapshot.handle,
        "scope": task.scope.handle(),
        "producer": described,
        "revision": revision,
        "counts": tally.counts(),
        "occurrence_counts": tally.occurrence_counts(),
        "coverage": coverage(&tally, "missing_model_output"),
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
        "cost": cost.describe(worker, scored, rows),
    })
    .to_string();
    require_fit(&result)?;
    let kept = states
        .held_bytes()
        .saturating_add(
            scores
                .as_ref()
                .map_or(0, |scores| 4 * scores.capacity() as u64),
        )
        .saturating_add(producer.held_bytes())
        .saturating_add(2 * result.len() as u64 + handle.capacity() as u64)
        .saturating_add(ARTIFACT_FIXED_BYTES);
    let set = Set {
        handle,
        kind: SetKind::Selection,
        parent: task.scope.as_set().cloned(),
        snapshot,
        states,
        scores,
        producer,
        tally,
        revision,
        result: result.clone(),
        held: kept,
    };
    worker.budget.hold(worker.base.saturating_add(kept));
    Ok(Done {
        result,
        commit: Some(Commit::Set(Arc::new(set))),
    })
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
    let share = States::bytes_for(rows)
        .saturating_add(COMPOSE_FIXED_BYTES + ARTIFACT_FIXED_BYTES)
        .saturating_add(producer.held_bytes());
    worker.budget.hold(worker.base);
    worker.budget.require_whole(
        share,
        "composing a view: the state of every original of the snapshot,",
    )?;
    worker.budget.hold(worker.base.saturating_add(share));

    // The view is its parent and the expression: outside the parent nothing is in.
    let mut states = States::new(rows, State::Out);
    for row in 0..rows {
        if row % 65_536 == 0 {
            worker.check_cancel()?;
        }
        let within = task.scope.state(row);
        if within == State::Out {
            continue;
        }
        let state = task.expr.eval(&|index| task.sets[index].1.states.get(row));
        states.set(row, within.and(state));
    }
    // Among the members of the parent the view's state is the expression's.
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
    let (limit, peak) = worker.memory();
    let result = json!({
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
        "cost": {
            "archive_scans": 0,
            "rows_scored": 0,
            "index_rows_read": rows,
            "elapsed_ms": started.elapsed().as_millis() as u64,
            "memory_limit_bytes": limit,
            "counted_peak_bytes": peak,
            "process_peak_bytes": null,
        },
    })
    .to_string();
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
        producer,
        tally,
        revision,
        result: result.clone(),
        held: kept,
    };
    worker.budget.hold(worker.base.saturating_add(kept));
    Ok(Done {
        result,
        commit: Some(Commit::Set(Arc::new(set))),
    })
}

fn share_of(part: u64, parent: u64) -> Value {
    share(part, parent)
}

/// An artifact whose result would not fit the response limit is not committed: the
/// result is what every later call about it returns.
fn require_fit(result: &str) -> Result<(), Fault> {
    if wire::transport_bytes(result) > wire::PAYLOAD_TRANSPORT_BYTES {
        return Err(Fault::runtime(
            "the result of this call does not fit the response limit; nothing was committed",
        ));
    }
    Ok(())
}
