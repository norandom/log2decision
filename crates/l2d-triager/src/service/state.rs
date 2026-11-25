//! What the service keeps between calls: committed snapshots, selections and views,
//! and the table of jobs with their idempotency keys.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use l2d_core::memory::MemoryUsage;
use l2d_core::{ArchiveSummary, L2dError, ZipInputSettings};
use serde_json::{Value, json};

use super::membership::{State, States, Tally};
use super::tables::{BLOCK_ROWS, Scores};
use super::wire::{Fault, VERSION};
use super::{HostConfig, ScorerIdentity};
use crate::scope::Scope;

/// An open source: the scope of task 1.1 with what reading it again needs.
pub(crate) struct Snapshot {
    pub handle: String,
    pub scope: Scope,
    /// The reader's summary: the entry paths a locator names, and what was skipped.
    pub summary: ArchiveSummary,
    pub path: PathBuf,
    pub zip: ZipInputSettings,
    /// Whether the host declared the source a public fixture.
    pub public: bool,
    /// The result of the open that made it, as it was returned.
    pub result: String,
    /// What the snapshot keeps in memory; held in the budget while it lives.
    pub held: u64,
    /// For a snapshot reopened from a receipt: the receipt and what it retained. Its
    /// relation holds the retained originals only and its source is not read.
    pub retained: Option<Retained>,
}

/// What a snapshot reopened from a receipt is: the receipt, the snapshot it was
/// saved from and that snapshot's denominators.
pub(crate) struct Retained {
    pub receipt: String,
    pub origin: String,
    pub origin_originals: u64,
    pub origin_occurrences: u64,
}

/// The refusal of an operation that reads the source or a host file of a snapshot
/// reopened from a receipt: the receipt retains no source text.
pub(crate) fn source_not_retained(snapshot: &Snapshot, what: &str) -> Result<(), Fault> {
    match &snapshot.retained {
        None => Ok(()),
        Some(retained) => Err(Fault::new(
            "source_not_retained",
            format!(
                "{what} reads the source or a file of the host, and `{}` was reopened from the \
                 receipt `{}`, which retains record IDs, memberships, scores and allocations \
                 but no source text; compose, allocate, inspect and save it, or open the source \
                 and replay the receipt's recipe",
                snapshot.handle, retained.receipt
            ),
        )),
    }
}

impl Snapshot {
    pub fn content_sha256(&self) -> &str {
        match self.scope.source() {
            l2d_core::MarkedSource::Input(input) => &input.content_sha256,
            l2d_core::MarkedSource::Store(store) => &store.content_sha256,
        }
    }
}

/// The most bytes of a filter's query that a result shows.
const MAX_QUERY_SHOWN_BYTES: usize = 512;

/// What made a set, with the revisions it is bound to.
pub(crate) enum Producer {
    Score {
        model: String,
        scorer: ScorerIdentity,
        top_k: Option<u64>,
        threshold: Option<f64>,
    },
    Filter {
        syntax: &'static str,
        query: String,
        case_insensitive: bool,
    },
    Expression {
        expr: Value,
    },
    /// A membership file the host registered, bound to its content.
    Membership {
        name: String,
        content_sha256: String,
        bytes: u64,
    },
    /// The exact members of one pattern of a refinement revision.
    Pattern {
        refinement: String,
        revision: String,
        pattern: u64,
    },
    /// The nearest neighbours of an exemplar among the members of a refinement,
    /// and of a wider one when the request named it as target.
    Neighbours {
        /// The request as results state it, with the revisions it is bound to.
        described: Value,
    },
    /// The routes of a saved Venn policy report the host registered.
    Venn {
        /// The report as results state it: name, content hash, policy, scorer,
        /// history and profile revisions, calibration and the membership rule.
        described: Value,
        /// The revision of the scorer the policy scored with.
        scorer_revision: String,
    },
    /// The judgments a model client submitted for the issued batches of a run.
    Issued {
        run: Arc<Run>,
        /// The submissions of the run the set covers: the first `upto`.
        upto: usize,
        /// The batches of the run the set was collected over: the first `batches`.
        /// What the run issues later is no part of the set.
        batches: usize,
        described: Value,
    },
    /// One review budget allocated once over named pools (`mode=allocate`): the
    /// canonical request with every name replaced by its handle.
    Allocation {
        spec: Value,
    },
    /// A set reopened from a receipt: the producer as the receipt states it, and what
    /// kind of producer that was.
    Retained {
        described: Value,
        origin: Origin,
        /// For an issued-batch set: the judgments the receipt retained.
        judged: Option<Arc<Judged>>,
    },
}

/// The judgments a receipt retained of an issued-batch set: the run as it states it
/// (name, caller-reported model and prompt revision, the batches with their revisions
/// and input digests, the submissions covered) and, per retained record by record ID
/// ascending, what the set's explanation said of it, as JSON text.
pub(crate) struct Judged {
    pub run: Value,
    pub records: Vec<(u64, Box<str>)>,
}

impl Judged {
    /// What the judgments keep in memory: the list, the text of every detail and the
    /// parsed run block (a parsed value takes several times its text).
    pub fn held_bytes(&self) -> u64 {
        (self.records.capacity() * std::mem::size_of::<(u64, Box<str>)>()
            + self
                .records
                .iter()
                .map(|(_, detail)| detail.len())
                .sum::<usize>()) as u64
            + 16 * self.run.to_string().len() as u64
    }

    /// The detail of the record with the record ID `position`.
    pub fn detail(&self, position: u64) -> Option<&str> {
        self.records
            .binary_search_by_key(&position, |(at, _)| *at)
            .ok()
            .map(|at| &*self.records[at].1)
    }
}

/// What kind of producer made a set a receipt retained: what its explanation reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Origin {
    Score { scorer_revision: String },
    Filter,
    Membership,
    Pattern { pattern: u64 },
    Neighbours,
    Venn { scorer_revision: String },
    Issued,
    Expression,
    Allocation,
}

impl Origin {
    /// The kind of the producer a receipt names, and what of it the explanation reads.
    pub fn of(producer: &Producer) -> Self {
        match producer {
            Producer::Score { scorer, .. } => Origin::Score {
                scorer_revision: scorer.revision.clone(),
            },
            Producer::Filter { .. } => Origin::Filter,
            Producer::Membership { .. } => Origin::Membership,
            Producer::Pattern { pattern, .. } => Origin::Pattern { pattern: *pattern },
            Producer::Neighbours { .. } => Origin::Neighbours,
            Producer::Venn {
                scorer_revision, ..
            } => Origin::Venn {
                scorer_revision: scorer_revision.clone(),
            },
            Producer::Issued { .. } => Origin::Issued,
            Producer::Expression { .. } => Origin::Expression,
            Producer::Allocation { .. } => Origin::Allocation,
            Producer::Retained { origin, .. } => origin.clone(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Origin::Score { .. } => "score",
            Origin::Filter => "filter",
            Origin::Membership => "membership",
            Origin::Pattern { .. } => "pattern",
            Origin::Neighbours => "neighbours",
            Origin::Venn { .. } => "venn",
            Origin::Issued => "issued_batch",
            Origin::Expression => "expression",
            Origin::Allocation => "allocation",
        }
    }

    /// The kind as a receipt writes it, with what the explanation reads.
    pub fn describe(&self) -> Value {
        match self {
            Origin::Score { scorer_revision } | Origin::Venn { scorer_revision } => {
                json!({"kind": self.name(), "scorer_revision": scorer_revision})
            }
            Origin::Pattern { pattern } => json!({"kind": self.name(), "pattern": pattern}),
            _ => json!({"kind": self.name()}),
        }
    }

    /// The kind a receipt names; `None` for one this revision does not know.
    pub fn parse(value: &Value) -> Option<Self> {
        let revision = || value.get("scorer_revision")?.as_str().map(str::to_string);
        Some(match value.get("kind")?.as_str()? {
            "score" => Origin::Score {
                scorer_revision: revision()?,
            },
            "venn" => Origin::Venn {
                scorer_revision: revision()?,
            },
            "filter" => Origin::Filter,
            "membership" => Origin::Membership,
            "pattern" => Origin::Pattern {
                pattern: value.get("pattern")?.as_u64()?,
            },
            "neighbours" => Origin::Neighbours,
            "issued_batch" => Origin::Issued,
            "expression" => Origin::Expression,
            "allocation" => Origin::Allocation,
            _ => return None,
        })
    }
}

impl Producer {
    /// The producer as results state it and as a set's handle is derived from it.
    pub fn describe(&self) -> Value {
        match self {
            Producer::Score {
                model,
                scorer,
                top_k,
                threshold,
            } => {
                let mut described = json!({
                    "kind": "score",
                    "version": 1,
                    "model": model,
                    "scorer": {"kind": scorer.kind, "name": scorer.name, "revision": scorer.revision},
                    "order": "score_desc",
                });
                if let Some(top_k) = top_k {
                    described["top_k"] = json!(top_k);
                    described["ties"] = json!("record_id_ascending");
                }
                if let Some(threshold) = threshold {
                    described["threshold"] = json!(threshold);
                    described["rule"] = json!("score >= threshold");
                }
                described
            }
            Producer::Filter {
                syntax,
                query,
                case_insensitive,
            } => {
                // The set keeps the exact query; a result shows at most this much of
                // it and names the whole by its length and hash.
                let (shown, clipped) = super::wire::display(query, MAX_QUERY_SHOWN_BYTES);
                json!({
                    "kind": "filter",
                    "version": 1,
                    "syntax": syntax,
                    "query": shown,
                    "query_bytes": query.len(),
                    "query_sha256": l2d_core::input::content_sha256(query.as_bytes()),
                    "query_clipped": clipped,
                    "case_insensitive": case_insensitive,
                })
            }
            Producer::Expression { expr } => json!({"kind": "expression", "expr": expr}),
            Producer::Membership {
                name,
                content_sha256,
                bytes,
            } => json!({
                "kind": "membership",
                "version": 1,
                "ref": name,
                "content_sha256": content_sha256,
                "bytes": bytes,
            }),
            Producer::Pattern {
                refinement,
                revision,
                pattern,
            } => json!({
                "kind": "filter",
                "version": 1,
                "syntax": "pattern",
                "refinement": refinement,
                "refinement_revision": revision,
                "pattern": pattern,
            }),
            Producer::Neighbours { described }
            | Producer::Venn { described, .. }
            | Producer::Issued { described, .. } => described.clone(),
            Producer::Allocation { spec } => {
                let mut described = spec.clone();
                described["kind"] = json!("allocation");
                described
            }
            Producer::Retained { described, .. } => described.clone(),
        }
    }

    pub fn held_bytes(&self) -> u64 {
        (match self {
            Producer::Score { model, scorer, .. } => {
                model.capacity()
                    + scorer.kind.capacity()
                    + scorer.name.capacity()
                    + scorer.revision.capacity()
            }
            Producer::Filter { query, .. } => query.capacity(),
            // A parsed tree takes several times its text.
            Producer::Expression { expr } => 16 * expr.to_string().len(),
            Producer::Membership {
                name,
                content_sha256,
                ..
            } => name.capacity() + content_sha256.capacity(),
            Producer::Pattern {
                refinement,
                revision,
                ..
            } => refinement.capacity() + revision.capacity(),
            // As an expression: a parsed tree takes several times its text.
            Producer::Neighbours { described } | Producer::Issued { described, .. } => {
                16 * described.to_string().len()
            }
            Producer::Venn {
                described,
                scorer_revision,
            } => 16 * described.to_string().len() + scorer_revision.capacity(),
            Producer::Allocation { spec } => 16 * spec.to_string().len(),
            Producer::Retained {
                described,
                origin,
                judged,
            } => {
                16 * described.to_string().len()
                    + 16 * origin.describe().to_string().len()
                    + judged
                        .as_ref()
                        .map_or(0, |judged| judged.held_bytes() as usize)
            }
        }) as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SetKind {
    Selection,
    View,
    /// A final work set: one review budget allocated once.
    Final,
}

impl SetKind {
    pub fn name(self) -> &'static str {
        match self {
            SetKind::Selection => "selection",
            SetKind::View => "view",
            SetKind::Final => "final",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "selection" => Some(SetKind::Selection),
            "view" => Some(SetKind::View),
            "final" => Some(SetKind::Final),
            _ => None,
        }
    }
}

/// An immutable selection or candidate view: the membership of every original of its
/// snapshot, within the population it was made in.
pub(crate) struct Set {
    pub handle: String,
    pub kind: SetKind,
    pub snapshot: Arc<Snapshot>,
    /// The population it was evaluated in; `None` is the whole snapshot.
    pub parent: Option<Arc<Set>>,
    pub states: States,
    /// For a score set: the score of every evaluated row of the relation; NaN where
    /// there is none. A set derived from the scores of another set of the same scorer
    /// revision shares its table, which then also holds scores outside this set's
    /// parent; they are read only for members of the parent.
    pub scores: Option<Arc<Scores>>,
    /// Whether this set made its scores, and so holds them.
    pub owns_scores: bool,
    pub producer: Producer,
    /// Originals and occurrences by state, among the members of the parent.
    pub tally: Tally,
    /// A digest of the membership, whatever order the rows were read in.
    pub revision: String,
    /// The result of the call that made it, as it was returned.
    pub result: String,
    pub held: u64,
    /// The sets it was composed or allocated from: the named sets of a view's
    /// expression, and those of an allocation with the score sets its pools order by.
    /// Empty for a selection, whose scope is `parent`.
    pub inputs: Vec<Arc<Set>>,
    /// For a final work set: the allocation.
    pub allocation: Option<Arc<super::allocate::Allocation>>,
}

impl Set {
    /// What the set's own tables take in the work directory; a table it shares with
    /// the set it was derived from is that set's.
    pub fn disk_bytes(&self) -> u64 {
        self.states.disk_bytes()
            + self
                .scores
                .as_ref()
                .filter(|_| self.owns_scores)
                .map_or(0, |scores| scores.disk_bytes())
    }
}

/// A population a producer or an expression is evaluated in.
#[derive(Clone)]
pub(crate) enum Population {
    All(Arc<Snapshot>),
    Of(Arc<Set>),
}

impl Population {
    pub fn snapshot(&self) -> &Arc<Snapshot> {
        match self {
            Population::All(snapshot) => snapshot,
            Population::Of(set) => &set.snapshot,
        }
    }

    pub fn handle(&self) -> &str {
        match self {
            Population::All(snapshot) => &snapshot.handle,
            Population::Of(set) => &set.handle,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Population::All(_) => "snapshot",
            Population::Of(set) => set.kind.name(),
        }
    }

    pub fn revision(&self) -> &str {
        match self {
            Population::All(snapshot) => &snapshot.scope.binding().relation,
            Population::Of(set) => &set.revision,
        }
    }

    pub fn state(&self, row: u64) -> Result<State, L2dError> {
        match self {
            Population::All(_) => Ok(State::In),
            Population::Of(set) => set.states.get(row),
        }
    }

    /// The codes of the block of rows that starts at `first`, as
    /// [`States::read_block`] gives them: every row in for a whole snapshot.
    pub fn read_block(&self, first: u64, out: &mut [u8]) -> Result<usize, L2dError> {
        match self {
            Population::All(snapshot) => {
                let rows = BLOCK_ROWS.min(snapshot.scope.originals().saturating_sub(first));
                out[..(rows as usize).div_ceil(4)].fill(0b0101_0101);
                Ok(rows as usize)
            }
            Population::Of(set) => set.states.read_block(first, out),
        }
    }

    /// Its members: distinct originals and occurrences.
    pub fn size(&self) -> (u64, u64) {
        match self {
            Population::All(snapshot) => (snapshot.scope.originals(), snapshot.scope.occurrences()),
            Population::Of(set) => set.tally.of(State::In),
        }
    }

    pub fn as_set(&self) -> Option<&Arc<Set>> {
        match self {
            Population::All(_) => None,
            Population::Of(set) => Some(set),
        }
    }
}

/// One immutable revision of the pattern catalogue of a filter selection: the
/// project's `PatternRefinement`, kept as the file its own `save` writes and loaded
/// under the budget by every action on it. A split or a merge makes another one.
pub(crate) struct Refinement {
    pub handle: String,
    /// The selection whose members it groups.
    pub set: Arc<Set>,
    /// The refinement it was split or merged from.
    pub parent: Option<String>,
    /// The revision the project's refinement states of itself.
    pub revision: String,
    /// `refinement.json` in a directory of its own, removed with the refinement.
    pub path: PathBuf,
    pub disk_bytes: u64,
    pub result: String,
    pub held: u64,
}

impl Drop for Refinement {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

/// The outcome of one judgment, as the code its record has in a collected set.
pub(crate) const OUTCOMES: [(&str, u8); 3] = [
    ("recommend", State::In as u8),
    ("pass", State::Out as u8),
    ("abstain", super::membership::ABSTAINED),
];

/// One judgment of a submission: an issued record, its outcome and the reason given.
pub(crate) struct Judgment {
    pub position: u64,
    pub code: u8,
    pub reason: String,
}

/// An accepted submission for one issued batch.
pub(crate) struct Submission {
    pub batch: usize,
    /// The digest of the payload: the same payload again is a retry, not another
    /// submission.
    pub digest: String,
    /// By position ascending.
    pub judgments: Vec<Judgment>,
    /// The batches the run had issued when the submission was taken: a retry
    /// collects over the same ones.
    pub batches_then: usize,
}

/// An issued batch: the records a model client may judge, and nothing else.
pub(crate) struct Batch {
    pub handle: String,
    /// The issued records, by position ascending.
    pub positions: Vec<u64>,
    /// The digest of the projected evidence the batch was issued with.
    pub input_sha256: String,
    /// One more than the submissions accepted for the batch.
    pub revision: u64,
}

/// What a run has issued and collected so far. It only grows.
#[derive(Default)]
pub(crate) struct RunState {
    pub batches: Vec<Batch>,
    pub submissions: Vec<Submission>,
    /// The model and the prompt revision the first submission reported; every later
    /// one must report the same.
    pub model: Option<(String, String)>,
    /// The row after the last record issued: the next batch goes on from there.
    pub next_row: u64,
    pub excerpt_bytes: u64,
}

/// The most batches of one run, and the most submissions accepted for one batch.
pub(crate) const MAX_BATCHES: usize = 64;
pub(crate) const MAX_BATCH_SUBMISSIONS: u64 = 8;

/// A recommendation run over one candidate scope: issued batches and the submissions
/// collected for them (interface.md, "For GenAI recommendations").
pub(crate) struct Run {
    pub handle: String,
    pub name: String,
    /// The candidate scope: no record outside it is issued, and no submission
    /// outside it fills coverage.
    pub scope: Arc<Set>,
    /// Whether batches carry excerpts.
    pub excerpts: bool,
    pub state: Mutex<RunState>,
}

impl Run {
    pub fn state(&self) -> std::sync::MutexGuard<'_, RunState> {
        self.state.lock().unwrap_or_else(|held| held.into_inner())
    }
}

#[derive(Default)]
pub(crate) struct Registry {
    snapshots: HashMap<String, Arc<Snapshot>>,
    sets: HashMap<String, Arc<Set>>,
    refinements: HashMap<String, Arc<Refinement>>,
    /// Runs by their own handle and by the handle of every batch they issued.
    runs: HashMap<String, Arc<Run>>,
}

impl Registry {
    pub fn refinement(&self, handle: &str) -> Option<Arc<Refinement>> {
        self.refinements.get(handle).cloned()
    }

    /// Every committed refinement, in no stated order.
    pub fn refinements(&self) -> impl Iterator<Item = &Arc<Refinement>> {
        self.refinements.values()
    }

    pub fn insert_refinement(&mut self, refinement: Arc<Refinement>) {
        self.refinements
            .insert(refinement.handle.clone(), refinement);
    }

    pub fn run(&self, handle: &str) -> Option<Arc<Run>> {
        self.runs.get(handle).cloned()
    }

    /// Registers `run` under `handle`: its own, or that of a batch it issued.
    pub fn insert_run(&mut self, handle: String, run: Arc<Run>) {
        self.runs.insert(handle, run);
    }

    /// The refinement `handle` names.
    pub fn require_refinement(&self, handle: &str) -> Result<Arc<Refinement>, Fault> {
        self.refinement(handle).ok_or_else(|| {
            Fault::new(
                "unknown_handle",
                format!(
                    "`{}` names no refinement of this project and server; refinements are \
                     issued by `triager_refine`",
                    super::wire::clip(handle, 80).0
                ),
            )
        })
    }

    pub fn snapshot(&self, handle: &str) -> Option<Arc<Snapshot>> {
        self.snapshots.get(handle).cloned()
    }

    pub fn set(&self, handle: &str) -> Option<Arc<Set>> {
        self.sets.get(handle).cloned()
    }

    pub fn insert_snapshot(&mut self, snapshot: Arc<Snapshot>) {
        self.snapshots.insert(snapshot.handle.clone(), snapshot);
    }

    pub fn insert_set(&mut self, set: Arc<Set>) {
        self.sets.insert(set.handle.clone(), set);
    }

    /// Every committed selection and view, in no stated order.
    pub fn sets(&self) -> impl Iterator<Item = &Arc<Set>> {
        self.sets.values()
    }

    /// Lets every artifact go: sets first, since they keep their snapshot.
    pub fn clear(&mut self) {
        self.runs.clear();
        self.refinements.clear();
        self.sets.clear();
        self.snapshots.clear();
    }

    fn unknown(handle: &str) -> Fault {
        Fault::new(
            "unknown_handle",
            format!(
                "`{}` names nothing of this project and server; handles are issued by \
                 `triager_open`, `triager_select` and `triager_compose`",
                super::wire::clip(handle, 80).0
            ),
        )
    }

    /// The snapshot `handle` names. A handle of another kind is told apart from one
    /// that names nothing.
    pub fn require_snapshot(&self, handle: &str) -> Result<Arc<Snapshot>, Fault> {
        if let Some(snapshot) = self.snapshot(handle) {
            return Ok(snapshot);
        }
        match self.set(handle) {
            Some(set) => Err(Fault::new(
                "wrong_handle_kind",
                format!(
                    "`{handle}` is a {} and a snapshot is expected here",
                    set.kind.name()
                ),
            )),
            None => Err(Self::unknown(handle)),
        }
    }

    /// The selection or view `handle` names.
    pub fn require_set(&self, handle: &str) -> Result<Arc<Set>, Fault> {
        if let Some(set) = self.set(handle) {
            return Ok(set);
        }
        match self.snapshot(handle) {
            Some(_) => Err(Fault::new(
                "wrong_handle_kind",
                format!("`{handle}` is a snapshot and a selection or view is expected here"),
            )),
            None => Err(Self::unknown(handle)),
        }
    }

    /// The snapshot, selection or view `handle` names, as a population.
    pub fn require_population(&self, handle: &str) -> Result<Population, Fault> {
        if let Some(snapshot) = self.snapshot(handle) {
            return Ok(Population::All(snapshot));
        }
        self.set(handle)
            .map(Population::Of)
            .ok_or_else(|| Self::unknown(handle))
    }
}

/// What a job does, with every handle resolved before it was scheduled.
pub(crate) enum Task {
    Open(super::ops::OpenTask),
    Select(super::ops::SelectTask),
    Compose(super::ops::ComposeTask),
    Evidence(super::inspect::EvidenceTask),
    Refine(super::refine::RefineTask),
    Recommend(super::recommend::RecommendTask),
    Allocate(super::allocate::AllocateTask),
    Save(super::receipt::SaveTask),
    Reopen(super::receipt::ReopenTask),
}

impl Task {
    /// Whether the task reads a source or runs one of the project's heavy operations
    /// and so takes the host lease. Composing a view, allocating, saving or reopening a
    /// receipt and taking or collecting the submissions of a run read committed state
    /// or a receipt only.
    pub fn reads_source(&self) -> bool {
        match self {
            Task::Compose(_) | Task::Allocate(_) | Task::Save(_) | Task::Reopen(_) => false,
            Task::Recommend(task) => task.reads_source(),
            _ => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobState {
    Waiting,
    Running,
    Cancelled,
    Completed,
    Interrupted,
    Failed,
}

impl JobState {
    pub fn ended(self) -> bool {
        !matches!(self, JobState::Waiting | JobState::Running)
    }
}

/// What a running job shows of itself.
#[derive(Default)]
pub(crate) struct Progress {
    pub batches: AtomicU64,
    pub rows: AtomicU64,
    pub spilled: AtomicBool,
}

pub(crate) struct Job {
    pub tool: &'static str,
    pub state: JobState,
    pub phase: &'static str,
    pub cancel: Arc<AtomicBool>,
    pub progress: Arc<Progress>,
    /// The committed result, or why there is none.
    pub outcome: Option<Result<String, Fault>>,
    pub task: Option<Task>,
    /// When the job was queued: what it waited is part of its cost.
    pub queued: std::time::Instant,
    key: Option<String>,
}

impl Job {
    /// The state a client sees: a cancellation that was asked for and not yet taken
    /// is its own state.
    pub fn status(&self) -> &'static str {
        match self.state {
            JobState::Waiting | JobState::Running if self.cancel.load(Ordering::Relaxed) => {
                "cancel_requested"
            }
            JobState::Waiting => "waiting",
            JobState::Running => "running",
            JobState::Cancelled => "cancelled",
            JobState::Completed => "completed",
            JobState::Interrupted => "interrupted",
            JobState::Failed => "failed",
        }
    }
}

/// The most jobs kept, ended ones included, and the most that wait for the worker.
pub(crate) const MAX_JOBS: usize = 64;
pub(crate) const MAX_QUEUED: usize = 8;
/// The most bytes of an idempotency key.
pub(crate) const MAX_KEY_BYTES: usize = 128;

#[derive(Default)]
pub(crate) struct Jobs {
    jobs: HashMap<String, Job>,
    /// Every kept job, oldest first.
    order: VecDeque<String>,
    /// The jobs the worker has not taken yet.
    pub queue: VecDeque<String>,
    /// Key to the digest of its payload and its job.
    keys: HashMap<String, (String, String)>,
    next: u64,
    pub shutdown: bool,
}

/// What a job-starting call finds under its idempotency key.
pub(crate) enum Keyed {
    /// No job, or one that ended without a committed result: the call starts one.
    Free,
    /// The job of the same payload.
    Job(String),
}

impl Jobs {
    pub fn get(&self, id: &str) -> Option<&Job> {
        self.jobs.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Job> {
        self.jobs.get_mut(id)
    }

    /// Looks `key` up for a payload with `digest`. The same key with another payload
    /// conflicts.
    pub fn keyed(&self, key: &str, digest: &str) -> Result<Keyed, Fault> {
        let Some((known, id)) = self.keys.get(key) else {
            return Ok(Keyed::Free);
        };
        if known != digest {
            return Err(Fault::new(
                "idempotency_conflict",
                "this idempotency key was used with another payload; a key names one \
                 request, so nothing was started",
            ));
        }
        match self.jobs.get(id) {
            Some(job)
                if matches!(
                    job.state,
                    JobState::Waiting | JobState::Running | JobState::Completed
                ) =>
            {
                Ok(Keyed::Job(id.clone()))
            }
            _ => Ok(Keyed::Free),
        }
    }

    /// Adds a job for the worker. The oldest ended job makes room when the table is
    /// full; with nothing ended, or too many waiting, the call is refused.
    pub fn add(
        &mut self,
        project: &str,
        tool: &'static str,
        task: Task,
        key: Option<(String, String)>,
    ) -> Result<String, Fault> {
        if self.shutdown {
            return Err(Fault::refused("the server is ending; nothing was started"));
        }
        if self.queue.len() >= MAX_QUEUED {
            return Err(Fault::new(
                "too_many_jobs",
                format!(
                    "{MAX_QUEUED} jobs wait for the one that is running; poll or cancel one \
                     with `triager_job` before starting another"
                ),
            ));
        }
        if self.order.len() >= MAX_JOBS {
            let ended = self
                .order
                .iter()
                .position(|id| self.jobs.get(id).is_none_or(|job| job.state.ended()));
            let Some(at) = ended else {
                return Err(Fault::new(
                    "too_many_jobs",
                    format!("{MAX_JOBS} jobs are kept and none of them has ended"),
                ));
            };
            if let Some(id) = self.order.remove(at)
                && let Some(job) = self.jobs.remove(&id)
                && let Some(key) = job.key
                && self.keys.get(&key).is_some_and(|(_, of)| *of == id)
            {
                self.keys.remove(&key);
            }
        }
        self.next += 1;
        let id = super::wire::handle("job", project, &[&self.next.to_string()]);
        if let Some((key, digest)) = &key {
            self.keys.insert(key.clone(), (digest.clone(), id.clone()));
        }
        self.jobs.insert(
            id.clone(),
            Job {
                tool,
                state: JobState::Waiting,
                phase: "queued",
                cancel: Arc::new(AtomicBool::new(false)),
                progress: Arc::new(Progress::default()),
                outcome: None,
                task: Some(task),
                queued: std::time::Instant::now(),
                key: key.map(|(key, _)| key),
            },
        );
        self.order.push_back(id.clone());
        self.queue.push_back(id.clone());
        Ok(id)
    }

    /// Asks every job that has not ended to stop, and lets the tasks not yet taken go.
    pub fn stop_all(&mut self) {
        self.shutdown = true;
        self.queue.clear();
        for job in self.jobs.values_mut() {
            job.cancel.store(true, Ordering::Relaxed);
            job.task = None;
        }
    }
}

/// Counters over the life of the server: what a replay must leave unchanged.
#[derive(Default)]
pub(crate) struct Activity {
    pub archive_scans: AtomicU64,
    /// The unpacked bytes of the entries whose inflation a read of a source started.
    pub bytes_unpacked: AtomicU64,
    pub rows_scored: AtomicU64,
    pub scorer_loads: AtomicU64,
    pub membership_reads: AtomicU64,
    /// Originals given to an encoder for a neighbour search.
    pub rows_encoded: AtomicU64,
    /// Judgments a model client submitted and the service accepted.
    pub genai_rows_submitted: AtomicU64,
    /// Records issued to a model client in batches.
    pub genai_rows_issued: AtomicU64,
    pub jobs_started: AtomicU64,
    pub evidence_bytes: AtomicU64,
    /// Receipts written, and receipts reopened.
    pub receipts_saved: AtomicU64,
    pub receipts_opened: AtomicU64,
    /// Tool calls answered, and the bytes of their MCP tool results.
    pub calls: AtomicU64,
    pub bytes_returned: AtomicU64,
    /// Calls refused as not served by this revision.
    pub unsupported: AtomicU64,
}

/// Whether `bytes` of text of a source may be shown under `profile`: always to the
/// local analyst or a local model; to a hosted development client only of a source
/// the host declared a public fixture, and within what is `left` of the allowance.
pub(crate) fn disclosable(
    profile: super::DisclosureProfile,
    public_fixture: bool,
    left: u64,
    bytes: u64,
) -> Result<(), Fault> {
    match profile {
        super::DisclosureProfile::PrivateLocal => Ok(()),
        super::DisclosureProfile::PublicDevelopment if !public_fixture => Err(Fault::new(
            "disclosure_refused",
            "the host runs the public-development profile and this source is not declared \
             a public fixture; nothing of its text is shown or issued",
        )),
        super::DisclosureProfile::PublicDevelopment if bytes > left => Err(Fault::new(
            "disclosure_refused",
            format!(
                "this would show {bytes} bytes of evidence and the allowance of this session \
                 has {left} left; counts, handles and memberships stay available"
            ),
        )),
        super::DisclosureProfile::PublicDevelopment => Ok(()),
    }
}

/// What the adapter threads and the worker share.
pub(crate) struct Shared {
    pub config: HostConfig,
    pub registry: RwLock<Registry>,
    pub jobs: Mutex<Jobs>,
    /// Signalled when a job is queued, changes state or the server ends.
    pub changed: Condvar,
    pub activity: Activity,
    /// Under the public-development profile: the evidence bytes still to disclose.
    pub allowance: Mutex<u64>,
    pub usage: MemoryUsage,
    /// What the service holds in the budget between jobs.
    pub held: AtomicU64,
}

impl Shared {
    pub fn activity(&self) -> Value {
        json!({
            "archive_scans": self.activity.archive_scans.load(Ordering::Relaxed),
            "bytes_unpacked": self.activity.bytes_unpacked.load(Ordering::Relaxed),
            "rows_scored": self.activity.rows_scored.load(Ordering::Relaxed),
            "scorer_loads": self.activity.scorer_loads.load(Ordering::Relaxed),
            "membership_reads": self.activity.membership_reads.load(Ordering::Relaxed),
            "rows_encoded": self.activity.rows_encoded.load(Ordering::Relaxed),
            "genai_rows_issued": self.activity.genai_rows_issued.load(Ordering::Relaxed),
            "genai_rows_submitted": self.activity.genai_rows_submitted.load(Ordering::Relaxed),
            "jobs_started": self.activity.jobs_started.load(Ordering::Relaxed),
            "evidence_bytes_disclosed": self.activity.evidence_bytes.load(Ordering::Relaxed),
            "receipts_saved": self.activity.receipts_saved.load(Ordering::Relaxed),
            "receipts_opened": self.activity.receipts_opened.load(Ordering::Relaxed),
        })
    }

    /// What the session was answered so far: tool calls, the bytes of their MCP tool
    /// results, and the calls refused as not served by this revision.
    pub fn session(&self) -> Value {
        json!({
            "calls": self.activity.calls.load(Ordering::Relaxed),
            "bytes_returned": self.activity.bytes_returned.load(Ordering::Relaxed),
            "unsupported_refusals": self.activity.unsupported.load(Ordering::Relaxed),
        })
    }

    /// The evidence bytes this session may still disclose; `None` without a limit.
    pub fn remaining_evidence(&self) -> Option<u64> {
        match self.config.profile {
            super::DisclosureProfile::PrivateLocal => None,
            super::DisclosureProfile::PublicDevelopment => Some(
                *self
                    .allowance
                    .lock()
                    .unwrap_or_else(|held| held.into_inner()),
            ),
        }
    }

    /// The disclosure block of a result about `snapshot`.
    pub fn disclosure(&self, snapshot: &Snapshot) -> Value {
        json!({
            "profile": self.config.profile.name(),
            "source_class": if snapshot.public { "public_fixture" } else { "private" },
            "excerpts": true,
            "remaining_evidence_bytes": self.remaining_evidence(),
        })
    }

    /// Admits showing `bytes` of text of `snapshot`'s source to this session, and
    /// counts them against its allowance: templates, proposed queries and the excerpts
    /// of an issued batch, as the excerpt of one record is counted.
    pub fn disclose(&self, snapshot: &Snapshot, bytes: u64) -> Result<(), Fault> {
        let mut left = self
            .allowance
            .lock()
            .unwrap_or_else(|held| held.into_inner());
        disclosable(self.config.profile, snapshot.public, *left, bytes)?;
        if self.config.profile == super::DisclosureProfile::PublicDevelopment {
            *left -= bytes;
        }
        self.activity
            .evidence_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    /// The disclosure block of a result that shows `bytes` of evidence, as it will
    /// read once they are counted: a result is built and found to fit before the
    /// allowance is spent on it.
    pub fn disclosure_after(&self, snapshot: &Snapshot, bytes: u64) -> Value {
        let mut disclosure = self.disclosure(snapshot);
        if let Some(left) = self.remaining_evidence() {
            disclosure["remaining_evidence_bytes"] = json!(left.saturating_sub(bytes));
        }
        disclosure
    }

    /// `text` with the paths of the host taken out: the work directory, and every
    /// source, policy and membership file the host registered, in the spelling the
    /// host gave and in their resolved one. An error of a drill-down or a
    /// recommendation job goes through this before a client sees it.
    pub fn scrub(&self, text: String) -> String {
        let config = &self.config;
        let mut hidden: Vec<(String, &str)> = Vec::new();
        let mut hide = |path: &std::path::Path, what: &'static str| {
            let given = path.to_string_lossy().into_owned();
            if let Ok(resolved) = std::fs::canonicalize(path) {
                let resolved = resolved.to_string_lossy().into_owned();
                if let Some(plain) = resolved.strip_prefix(r"\\?\") {
                    hidden.push((plain.to_string(), what));
                }
                hidden.push((resolved, what));
            }
            hidden.push((given, what));
        };
        hide(&config.work_root, "<work directory>");
        if let Some(receipts) = &config.receipts {
            hide(receipts, "<receipt directory>");
        }
        for source in &config.sources {
            hide(&source.path, "<source>");
        }
        for policy in &config.policies {
            hide(&policy.path, "<policy file>");
        }
        for membership in &config.memberships {
            hide(&membership.path, "<membership file>");
        }
        // The longest first, so that a path inside another is not left in halves.
        hidden.sort_by_key(|(path, _)| std::cmp::Reverse(path.len()));
        let mut text = text;
        for (path, what) in hidden {
            if path.len() >= 3 && text.contains(&path) {
                text = text.replace(&path, what);
            }
        }
        text
    }

    /// The status of a job as `triager_job` returns it.
    pub fn job_status(&self, id: &str, job: &Job) -> String {
        let report = self.usage.report();
        let mut status = json!({
            "v": VERSION,
            "job": id,
            "status": job.status(),
            "tool": job.tool,
            "phase": job.phase,
            "progress": {
                "batches": job.progress.batches.load(Ordering::Relaxed),
                "rows": job.progress.rows.load(Ordering::Relaxed),
                "spilled": job.progress.spilled.load(Ordering::Relaxed),
            },
            "admitted": {
                "memory_limit_bytes": report.memory_limit_bytes,
                "counted_peak_bytes": report.counted_peak_bytes,
            },
        })
        .to_string();
        // The committed result is spliced in as it was returned, not parsed again.
        match &job.outcome {
            Some(Ok(result)) if job.state == JobState::Completed => {
                status.pop();
                status.push_str(",\"result\":");
                status.push_str(result);
                status.push('}');
            }
            Some(Err(fault)) => {
                let payload: Value = serde_json::from_str(&fault.payload()).unwrap_or_default();
                status.pop();
                status.push_str(",\"error\":");
                status.push_str(&payload["error"].to_string());
                status.push('}');
            }
            _ => {}
        }
        status
    }
}
