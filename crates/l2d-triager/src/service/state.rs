//! What the service keeps between calls: committed snapshots, selections and views,
//! and the table of jobs with their idempotency keys.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use l2d_core::memory::MemoryUsage;
use l2d_core::{ArchiveSummary, ZipInputSettings};
use serde_json::{Value, json};

use super::membership::{State, States, Tally};
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
        }) as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SetKind {
    Selection,
    View,
}

impl SetKind {
    pub fn name(self) -> &'static str {
        match self {
            SetKind::Selection => "selection",
            SetKind::View => "view",
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
    /// there is none.
    pub scores: Option<Vec<f32>>,
    pub producer: Producer,
    /// Originals and occurrences by state, among the members of the parent.
    pub tally: Tally,
    /// A digest of the membership, whatever order the rows were read in.
    pub revision: String,
    /// The result of the call that made it, as it was returned.
    pub result: String,
    pub held: u64,
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

    pub fn state(&self, row: u64) -> State {
        match self {
            Population::All(_) => State::In,
            Population::Of(set) => set.states.get(row),
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

#[derive(Default)]
pub(crate) struct Registry {
    snapshots: HashMap<String, Arc<Snapshot>>,
    sets: HashMap<String, Arc<Set>>,
}

impl Registry {
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

    /// Lets every artifact go: sets first, since they keep their snapshot.
    pub fn clear(&mut self) {
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
}

impl Task {
    /// Whether the task reads a source and so takes the host lease.
    pub fn reads_source(&self) -> bool {
        !matches!(self, Task::Compose(_))
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
    pub jobs_started: AtomicU64,
    pub evidence_bytes: AtomicU64,
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
            "jobs_started": self.activity.jobs_started.load(Ordering::Relaxed),
            "evidence_bytes_disclosed": self.activity.evidence_bytes.load(Ordering::Relaxed),
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
