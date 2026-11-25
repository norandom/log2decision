//! Recommendation producers for C (triager task 2.1, requirements 2.2, 3.2, 4.4 and
//! 6.2): the routes of a saved Venn policy report, and the judgments a model client
//! submits for issued batches.
//!
//! # The Venn policy
//!
//! The policy is not run here. The host registers a report `l2d triage` saved; the
//! service reads it as the terminal does (`PolicyReport` with `validate_saved`), joins
//! its decisions to the snapshot with `Scope::join_policy` and keeps, beside the state
//! of every decided original, its route, its reason, its allocation and whether its
//! ratio was actionable. The scorer, history and profile revisions the policy
//! depended on are part of the producer.
//!
//! # Issued batches
//!
//! A run over one candidate scope issues bounded batches of its members in the order
//! of the record IDs, with an excerpt when the run projects one, counted against the
//! session's evidence allowance. A model client submits judgments for the records of
//! one batch under the batch revision it saw. A submission is taken whole or not at
//! all: an ID that was not issued in the batch, an ID named twice, an unknown outcome,
//! another model or prompt than the run's first submission reported, or a revision
//! that is no longer the batch's refuses it. The same payload again is a retry and
//! changes nothing. What is collected is one selection: `recommend` is in, `pass` is
//! out, an abstention, an omitted record, a record not yet issued and a record two
//! submissions judged differently are not evaluated. A suggestion is never a review:
//! the service keeps no analyst mark, and every result says so.
//!
//! Under the public-development profile only a source the host declared a public
//! fixture is issued; nothing of a private source reaches a batch. The service calls
//! no model and no network: the client that holds the batch does the inference.
//!
//! # Memory
//!
//! A run keeps its batches (the positions issued) and its submissions (a judgment per
//! record with a reason of at most [`MAX_REASON_BYTES`]). Every batch and submission
//! is asked of the budget before it is kept and held from then on; a run is bounded by
//! [`MAX_BATCHES`] batches of at most [`MAX_ROWS`] records and
//! [`MAX_BATCH_SUBMISSIONS`] submissions a batch.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use l2d_core::L2dError;
use l2d_decide::policy::{
    Allocation, CalibrationEvidence, PolicyDecision, PolicyReport, Queue, Reason,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::membership::{State, States, code_in};
use super::ops::{
    ARTIFACT_FIXED_BYTES, Commit, Cost, Done, Listed, SCAN_FIXED_BYTES, Worker, registry, rows_of,
    scan, select_listed,
};
use super::state::{
    Batch, Judgment, MAX_BATCH_SUBMISSIONS, MAX_BATCHES, OUTCOMES, Population, Producer, Run,
    RunState, Set, Submission, disclosable,
};
use super::tables::BLOCK_ROWS;
use super::wire::{self, Fault, MAX_EXCERPT_BYTES, MAX_ROWS, PAYLOAD_TRANSPORT_BYTES, VERSION};
use crate::scope::JoinError;

/// The most bytes of a saved policy report: the limit of the terminal's loader.
const MAX_POLICY_BYTES: u64 = 16 * 1024 * 1024;
/// The most bytes of the reason of one judgment.
pub(crate) const MAX_REASON_BYTES: usize = 200;
/// The most bytes of a model identity and of a prompt revision.
pub(crate) const MAX_MODEL_BYTES: usize = 128;
pub(crate) const MAX_PROMPT_BYTES: usize = 64;
/// What collecting a run holds: the state of every issued record on its way to the
/// tables, and the digests of the run.
const COLLECT_BYTES: u64 = (MAX_BATCHES * MAX_ROWS * 64) as u64 + 16 * 1024;
/// What a judgment and an issued record take beside their text.
const JUDGMENT_BYTES: u64 = std::mem::size_of::<Judgment>() as u64;
/// The values kept beside the records of a collected set.
const CONFLICT: f32 = 4.0;

/// Which routes of a policy are in C.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VennMembership {
    /// Investigate and Review are in, unsupported cases sent to Review included.
    InvestigateOrReview,
    Investigate,
}

impl VennMembership {
    pub fn name(self) -> &'static str {
        match self {
            VennMembership::InvestigateOrReview => "investigate_or_review",
            VennMembership::Investigate => "investigate",
        }
    }
}

/// A submission as the request gave it, validated in itself.
pub(crate) struct Submitted {
    pub batch: String,
    pub expected_revision: u64,
    pub model: String,
    pub prompt_revision: String,
    /// By position ascending, no position twice.
    pub judgments: Vec<Judgment>,
    pub digest: String,
}

pub(crate) enum RecommendTask {
    Venn {
        scope: Population,
        name: String,
        path: PathBuf,
        membership: VennMembership,
    },
    Issue {
        scope: Arc<Set>,
        /// The name of the run within the scope.
        name: String,
        limit: usize,
        excerpts: bool,
    },
    Submit {
        run: Arc<Run>,
        submitted: Submitted,
    },
    Collect {
        run: Arc<Run>,
        submissions_ref: Option<String>,
    },
}

impl RecommendTask {
    /// Whether the task reads a source: a batch with excerpts does.
    pub fn reads_source(&self) -> bool {
        matches!(self, RecommendTask::Issue { excerpts: true, .. })
    }
}

// --- The Venn policy -----------------------------------------------------------------------

const QUEUES: [&str; 3] = ["investigate", "review", "low_priority"];
const REASONS: [&str; 5] = [
    "unsupported",
    "disagreement",
    "high_corroborated",
    "low_corroborated",
    "uncertain",
];
const ALLOCATIONS: [&str; 5] = [
    "deferred",
    "unknown_conflict",
    "general",
    "low_priority_audit",
    "unsupported",
];

/// What a decision keeps beside its state, as one small whole number.
fn venn_value(
    queue: usize,
    reason: usize,
    allocation: usize,
    actionable: bool,
    assumed: bool,
) -> f32 {
    (1 + queue
        + 3 * reason
        + 15 * allocation
        + 75 * usize::from(actionable)
        + 150 * usize::from(assumed)) as f32
}

/// The route, reason and allocation a Venn set keeps for a decided record.
pub(crate) fn venn_detail(value: f32) -> Option<Value> {
    if !(1.0..=300.0).contains(&value) || value.fract() != 0.0 {
        return None;
    }
    let code = value as usize - 1;
    let allocation = (code / 15) % 5;
    Some(json!({
        "route": QUEUES[code % 3],
        "reason": REASONS[(code / 3) % 5],
        "allocation": if allocation == 0 { Value::Null } else { json!(ALLOCATIONS[allocation]) },
        "deferred_by_policy": allocation == 0,
        "actionable_ratio": if (code / 75) % 2 == 1 { "present" } else { "withheld" },
        "comparison_class": if code / 150 == 1 { "assumed" } else { "explicit" },
        "kind": "suggested route; not a reviewed outcome",
    }))
}

/// The route, the reason and the allocation of a decision, as indices of [`QUEUES`],
/// [`REASONS`] and [`ALLOCATIONS`].
fn decision_indices(decision: &PolicyDecision) -> (usize, usize, usize) {
    let queue = match decision.queue {
        Queue::Investigate => 0,
        Queue::Review => 1,
        Queue::LowPriority => 2,
    };
    let reason = match decision.reason {
        Reason::Unsupported => 0,
        Reason::Disagreement => 1,
        Reason::HighCorroborated => 2,
        Reason::LowCorroborated => 3,
        Reason::Uncertain => 4,
    };
    let allocation = match decision.allocation {
        None => 0,
        Some(Allocation::UnknownConflict) => 1,
        Some(Allocation::General) => 2,
        Some(Allocation::LowPriorityAudit) => 3,
        Some(Allocation::Unsupported) => 4,
    };
    (queue, reason, allocation)
}

/// What reading a saved policy report of `len` bytes is charged: the charge of the
/// project's saved policy loader, for the file, its parsed evidence, the working
/// memory of its validation and the states its decisions become.
fn policy_charge(len: u64) -> u64 {
    (len + 1) * 32 + 1024 * 1024
}

/// Reads the report in `file`, of `len` bytes, parses it and validates it as the
/// project's loader does. Returns it with the SHA-256 of the file. The caller holds
/// [`policy_charge`].
///
/// Why a file is no report is told by position and kind only: a policy report holds
/// text of the logs it decided, and the host's file is not the client's to read. The
/// detail goes to the host's diagnostics.
fn read_policy(name: &str, file: File, len: u64) -> Result<(PolicyReport, String), Fault> {
    let mut bytes = Vec::with_capacity(len as usize + 1);
    file.take(len + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| policy_unread(name, "reading", &error))?;
    if bytes.len() as u64 != len {
        return Err(Fault::refused(format!(
            "the policy report `{name}` changed size while it was read; nothing was committed"
        )));
    }
    let content_sha256 = l2d_core::input::content_sha256(&bytes);
    let report: PolicyReport = serde_json::from_slice(&bytes).map_err(|error| {
        eprintln!("triager: the policy `{name}` does not parse: {error}");
        Fault::refused(format!(
            "the file registered as the policy `{name}` is no saved `l2d triage` policy \
             report: it does not parse at line {} column {} ({:?}); nothing was committed",
            error.line(),
            error.column(),
            error.classify()
        ))
    })?;
    drop(bytes);
    let report = report.validate_saved().map_err(|error| {
        eprintln!("triager: the policy `{name}` does not validate: {error}");
        Fault::refused(format!(
            "the file registered as the policy `{name}` does not pass the validation of a \
             saved `l2d triage` policy report (its format, routes, identities, revisions and \
             allocations are recomputed and compared); the host's diagnostics state which \
             check. Nothing was committed"
        ))
    })?;
    Ok((report, content_sha256))
}

/// Why the decisions of a report were not joined to a snapshot, without the paths
/// of the host: the report's and the snapshot's source paths are not shown.
fn join_fault(error: JoinError) -> Fault {
    match error {
        JoinError::PolicyPath { .. } => Fault::refused(
            "the policy report derived its IDs from another source path than the snapshot \
             was opened at; this needs a new producer receipt or a replay with the original \
             path. Nothing was joined",
        ),
        JoinError::Path { .. } => Fault::refused(
            "the policy report was made on another source location than the snapshot; \
             nothing was joined",
        ),
        other => Fault::from(L2dError::from(other)),
    }
}

/// Why the file registered as the policy `name` could not be read. The path is the
/// host's and is not shown.
fn policy_unread(name: &str, doing: &str, error: &std::io::Error) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "{doing} the file the host registered as the policy `{name}` failed ({:?}); \
             nothing was committed",
            error.kind()
        ),
    }
}

fn venn(
    worker: &mut Worker<'_>,
    scope: &Population,
    name: &str,
    path: &std::path::Path,
    membership: VennMembership,
    started: Instant,
) -> Result<Done, Fault> {
    let base = worker.base;
    worker.budget.hold(base);
    let file = File::open(path).map_err(|error| policy_unread(name, "opening", &error))?;
    let len = file
        .metadata()
        .map_err(|error| policy_unread(name, "reading the size of", &error))?
        .len();
    if len > MAX_POLICY_BYTES {
        return Err(Fault::refused(format!(
            "the policy report `{name}` has {len} bytes, and a report of at most \
             {MAX_POLICY_BYTES} bytes is read; nothing was committed"
        )));
    }
    let needed = policy_charge(len);
    worker
        .budget
        .require_whole(needed, "reading a saved Venn policy report")?;
    worker.budget.hold(base.saturating_add(needed));
    let (report, content_sha256) = read_policy(name, file, len)?;
    let snapshot = scope.snapshot().clone();
    let mut listing: Vec<(u64, u8, f32)> = Vec::with_capacity(report.decisions.len());
    let (mut routes, mut reasons, mut allocations) = ([0u64; 3], [0u64; 5], [0u64; 5]);
    let mut scorer_kind = Value::Null;
    snapshot
        .scope
        .join_policy(&report, worker.budget, |decision, original| {
            let (queue, reason, allocation) = decision_indices(decision);
            routes[queue] += 1;
            reasons[reason] += 1;
            allocations[allocation] += 1;
            if scorer_kind.is_null() {
                scorer_kind = serde_json::to_value(decision.score.kind).unwrap_or_default();
            }
            let is_in = match membership {
                VennMembership::InvestigateOrReview => queue < 2,
                VennMembership::Investigate => queue == 0,
            };
            listing.push((
                original.id.0,
                u8::from(is_in),
                venn_value(
                    queue,
                    reason,
                    allocation,
                    decision.score.actionable_ratio.is_some(),
                    decision.score.comparison_class_assumed,
                ),
            ));
            Ok(())
        })
        .map_err(join_fault)?;
    let calibration = match &report.calibration {
        CalibrationEvidence::Missing => json!({"state": "missing"}),
        CalibrationEvidence::Invalid { revision, .. } => {
            json!({"state": "invalid", "revision": wire::display(revision, 80).0})
        }
        CalibrationEvidence::Compatible {
            revision,
            error_rate,
        } => json!({"state": "compatible", "revision": wire::display(revision, 80).0,
                    "error_rate": error_rate}),
    };
    let short = |text: &str| wire::display(text, 160).0;
    let scorer_revision = report.scorer_revision.clone();
    let depends_on = json!([
        {"kind": "scorer", "scorer_kind": scorer_kind, "revision": short(&scorer_revision)},
        {"kind": "history", "revision": short(&report.evidence_revision)},
        {"kind": "vector_profile", "revision": short(&report.actual_profile)},
    ]);
    let described = json!({
        "kind": "recommend",
        "version": 1,
        "method": "venn",
        "policy": name,
        "content_sha256": content_sha256,
        "bytes": len,
        "membership": membership.name(),
        "predicate": match membership {
            VennMembership::InvestigateOrReview => "route is investigate or review",
            VennMembership::Investigate => "route is investigate",
        },
        "policy_version": short(&report.policy.version),
        "selection_provenance": short(&report.policy.selection_provenance),
        "eligibility": short(&report.eligibility),
        "allocation_rule": short(&report.allocation_rule),
        "calibration": calibration,
        "qualification": short(&report.qualification),
        "depends_on": depends_on,
        "decisions": report.decisions.len(),
    });
    drop(report);
    let count = |names: &[&str], counts: &[u64]| -> Value {
        Value::Object(
            names
                .iter()
                .zip(counts)
                .map(|(name, count)| ((*name).to_string(), json!(count)))
                .collect(),
        )
    };
    let mut extra = Map::new();
    extra.insert(
        "policy".into(),
        json!({
            "decisions": listing.len(),
            "routes": count(&QUEUES, &routes),
            "reasons": count(&REASONS, &reasons),
            "allocations": count(&ALLOCATIONS, &allocations),
            "kind": "suggested routes; no analyst mark is created",
        }),
    );
    let mut cost = Cost::new(started);
    cost.set("policy_bytes_read", len);
    let listed = Listed {
        scope,
        producer: Producer::Venn {
            described,
            scorer_revision,
        },
        with_values: true,
        reason: "outside_producer_coverage",
        extra,
        held: needed,
        cost,
    };
    let (result, set) = select_listed(worker, listed, &mut |put| {
        listing
            .iter()
            .try_for_each(|(position, code, value)| put(*position, *code, *value))
    })?;
    Ok(Done {
        result,
        commit: set.map(Commit::Set),
    })
}

// --- Issued batches ------------------------------------------------------------------------

fn outcome_name(code: u8) -> &'static str {
    OUTCOMES
        .iter()
        .find(|(_, known)| *known == code)
        .map_or("abstain", |(name, _)| name)
}

/// The digest that names the first `upto` submissions of a run.
fn manifest(state: &RunState, upto: usize, pending: Option<&Submitted>) -> String {
    let mut hash = Sha256::new();
    for digest in state.submissions[..upto]
        .iter()
        .map(|submission| &submission.digest)
        .chain(pending.map(|pending| &pending.digest))
    {
        hash.update(digest.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// What a batch keeps in memory.
fn batch_bytes(batch: &Batch) -> u64 {
    (std::mem::size_of::<Batch>()
        + batch.handle.capacity()
        + batch.positions.capacity() * 8
        + batch.input_sha256.capacity()) as u64
        // Its place in the registry, under its own handle.
        + batch.handle.len() as u64
        + 64
}

/// What a submission keeps in memory.
fn submission_bytes(digest: &str, judgments: &[Judgment]) -> u64 {
    std::mem::size_of::<Submission>() as u64
        + digest.len() as u64
        + judgments
            .iter()
            .map(|judgment| JUDGMENT_BYTES + judgment.reason.capacity() as u64)
            .sum::<u64>()
}

/// What a new run keeps before its first batch.
fn run_bytes(name: &str, handle: &str) -> u64 {
    (std::mem::size_of::<Run>() + name.len() + 2 * handle.len()) as u64 + ARTIFACT_FIXED_BYTES
}

fn issue(
    worker: &mut Worker<'_>,
    scope: &Arc<Set>,
    name: &str,
    limit: usize,
    excerpts: bool,
    started: Instant,
) -> Result<Done, Fault> {
    let shared = worker.shared;
    let snapshot = scope.snapshot.clone();
    // Found here, on the worker: a run an earlier job of the queue started is known.
    let run_handle = wire::handle("run", &shared.config.project, &[&scope.handle, name]);
    let known = registry(shared).run(&run_handle);
    let known = known.as_ref();
    // Before anything is read: whether this session may be issued this source at all.
    let left = shared.remaining_evidence().unwrap_or(u64::MAX);
    disclosable(
        shared.config.profile,
        snapshot.public,
        left,
        u64::from(excerpts),
    )?;
    let project = &shared.config.project;
    let run = match known {
        Some(run) => run.clone(),
        None => Arc::new(Run {
            handle: run_handle,
            name: name.to_string(),
            scope: scope.clone(),
            excerpts,
            state: Default::default(),
        }),
    };
    if run.excerpts != excerpts {
        return Err(Fault::invalid(
            "the run was started with another `projection`; a run issues one projection, so \
             that its coverage has one input",
        ));
    }
    let (index, first_row) = {
        let state = run.state();
        (state.batches.len(), state.next_row)
    };
    if index >= MAX_BATCHES {
        return Err(Fault::refused(format!(
            "the run has issued {MAX_BATCHES} batches, the most of one run; collect it and \
             start another run"
        )));
    }
    // Everything the batch holds while it is made, and what it keeps.
    let making = making_share(limit, excerpts);
    let base = worker.base;
    worker.budget.hold(base);
    worker.budget.require_whole(
        making,
        "issuing a batch: its records, their excerpts and the result,",
    )?;
    let held = base.saturating_add(making);
    worker.budget.hold(held);

    // The next members of the scope after the last record issued, by record ID.
    let rows = snapshot.scope.originals();
    let mut wanted: Vec<(u64, u64)> = Vec::with_capacity(limit);
    let mut within = vec![0u8; States::BLOCK_BYTES];
    let mut first = first_row - first_row % BLOCK_ROWS;
    'blocks: while first < rows {
        let in_block = scope.states.read_block(first, &mut within)?;
        for at in (first_row.saturating_sub(first) as usize)..in_block {
            if code_in(&within, at) == State::In as u8 {
                let row = first + at as u64;
                wanted.push((snapshot.scope.row(row)?.id.0, row));
                if wanted.len() == limit {
                    break 'blocks;
                }
            }
        }
        first += BLOCK_ROWS;
    }
    if wanted.is_empty() {
        return Err(Fault::new(
            "run_exhausted",
            "every member of the scope was issued in an earlier batch of this run; collect \
             the run",
        ));
    }
    // Rows are in the order of the record IDs, so `wanted` is sorted by position.
    let mut texts: Vec<Option<(String, bool)>> = Vec::new();
    let mut cost = Cost::new(started);
    if excerpts {
        texts.resize_with(wanted.len(), || None);
        let mut missing = wanted.len();
        let ended = scan(worker, &snapshot, held, &mut |logs| {
            for (text, position) in rows_of(logs)? {
                if let Ok(at) = wanted.binary_search_by_key(&position, |(position, _)| *position)
                    && texts[at].is_none()
                {
                    texts[at] = Some(wire::display(text, MAX_EXCERPT_BYTES));
                    missing -= 1;
                    if missing == 0 {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        });
        worker.budget.hold(held);
        ended?;
        if missing > 0 {
            return Err(Fault::refused(
                "a record of the batch was not found when its source was read again; the \
                 source no longer reads as the snapshot bound it, open it again",
            ));
        }
        cost.set("archive_scans", 1);
    }

    // Records are added while the result fits the response limit and the allowance.
    let batch_handle = wire::handle("batch", project, &[&run.handle, &index.to_string()]);
    let mut frame = json!({
        "v": VERSION,
        "status": "ready",
        "batch": batch_handle,
        "revision": 1,
        "run": {"name": run.name, "handle": run.handle},
        "scope": scope.handle,
        "snapshot": snapshot.handle,
        "projection": if excerpts { json!(["id", "excerpt"]) } else { json!(["id"]) },
        "outcomes": ["recommend", "pass", "abstain"],
        "limits": {"reason_bytes": MAX_REASON_BYTES, "model_bytes": MAX_MODEL_BYTES,
                   "prompt_revision_bytes": MAX_PROMPT_BYTES,
                   "submissions_per_batch": MAX_BATCH_SUBMISSIONS, "batches_per_run": MAX_BATCHES},
        "suggestion_only": "a judgment is a suggestion; only an analyst review is a reviewed label",
    });
    // The whole result is built, and found to fit, before anything is counted, spent
    // or kept: a batch is issued whole or not at all.
    let (members, _) = scope.tally.of(State::In);
    let issued_before: u64 = run
        .state()
        .batches
        .iter()
        .map(|batch| batch.positions.len() as u64)
        .sum();
    cost.set("candidates", members);
    frame["cost"] = cost.finish(worker);
    frame["disclosure"] = shared.disclosure(&snapshot);
    frame["batch_index"] = json!(index);
    let allowance = shared.remaining_evidence();
    let assembled = assemble(frame, &wanted, &texts, allowance, issued_before, members)
        .ok_or_else(|| {
            Fault::new(
                "disclosure_refused",
                format!(
                    "the next record of the run does not fit: the evidence allowance of this \
                     session has {left} bytes left; collect what was issued"
                ),
            )
        })?;
    let batch = Batch {
        handle: batch_handle,
        positions: assembled.positions,
        input_sha256: assembled.input_sha256,
        revision: 1,
    };
    let grown = batch_bytes(&batch)
        + if known.is_none() {
            run_bytes(name, &run.handle)
        } else {
            0
        };
    worker.budget.require_whole(
        grown,
        "keeping an issued batch for the submissions it may take",
    )?;
    // The last step that can refuse; nothing after it fails.
    shared.disclose(&snapshot, assembled.disclosed)?;
    shared
        .activity
        .genai_rows_issued
        .fetch_add(batch.positions.len() as u64, Ordering::Relaxed);
    let mut handles = vec![batch.handle.clone()];
    if known.is_none() {
        handles.push(run.handle.clone());
    }
    {
        let mut state = run.state();
        state.batches.push(batch);
        state.next_row = assembled.last_row + 1;
        state.excerpt_bytes += assembled.disclosed;
    }
    worker.budget.hold(base.saturating_add(grown));
    Ok(Done {
        result: assembled.result,
        commit: Some(Commit::Run {
            run,
            handles,
            grown,
            set: None,
        }),
    })
}

/// What making a batch of at most `limit` records holds: a block of the scope's
/// states, per record its position, its excerpt as it was read and as the result
/// carries it, the result and its serialized forms, and for a batch with excerpts the
/// read of the source.
fn making_share(limit: usize, excerpts: bool) -> u64 {
    States::BLOCK_BYTES as u64
        + limit as u64 * (16 + 2 * (MAX_EXCERPT_BYTES as u64 + 96))
        + 6 * PAYLOAD_TRANSPORT_BYTES as u64
        + if excerpts { SCAN_FIXED_BYTES } else { 0 }
}

/// A batch as it will be issued: the records that fit, and the complete result.
struct Assembled {
    positions: Vec<u64>,
    /// The row of the last record issued.
    last_row: u64,
    input_sha256: String,
    /// The excerpt bytes the result shows.
    disclosed: u64,
    result: String,
}

/// Builds the complete result of a batch: the first of the `wanted` records (their
/// positions and rows, with `texts` their excerpts) whose excerpts the `allowance`
/// covers and with which the result, every field of it included, fits the response
/// limit. `frame` holds the fields that do not depend on the records, `cost` and
/// `disclosure` among them. `None` when not even the first record fits.
fn assemble(
    mut frame: Value,
    wanted: &[(u64, u64)],
    texts: &[Option<(String, bool)>],
    allowance: Option<u64>,
    issued_before: u64,
    members: u64,
) -> Option<Assembled> {
    let left = allowance.unwrap_or(u64::MAX);
    let mut records = Vec::with_capacity(wanted.len());
    let mut sizes = Vec::with_capacity(wanted.len());
    let (mut disclosed, mut by_allowance, mut by_bytes) = (0u64, false, false);
    // Records are taken while they can fit beside the fields known so far, so that
    // no result far beyond the limit is ever built; the loop below settles the rest.
    let mut used = wire::transport_bytes(&frame.to_string());
    for (at, (position, _)) in wanted.iter().enumerate() {
        let mut record = json!({"id": wire::record_id(*position)});
        let mut bytes = 0u64;
        if let Some((text, cut)) = texts.get(at).and_then(Option::as_ref) {
            bytes = text.len() as u64;
            if disclosed + bytes > left {
                by_allowance = true;
                break;
            }
            record["excerpt"] = json!(text);
            if *cut {
                record["excerpt_clipped"] = json!(true);
            }
        }
        let written = record.to_string();
        let transport = 2 * (written.len() + 1) + wire::escape_bytes(&written);
        if used + transport > PAYLOAD_TRANSPORT_BYTES {
            by_bytes = true;
            break;
        }
        used += transport;
        disclosed += bytes;
        sizes.push(bytes);
        records.push(record);
    }
    let taken = records.len();
    frame["records"] = Value::Array(records);
    let mut count = taken;
    // Records are dropped from the end until the result that states them fits.
    loop {
        if count == 0 {
            return None;
        }
        let mut hash = Sha256::new();
        let mut clipped = false;
        for (at, (position, _)) in wanted[..count].iter().enumerate() {
            hash.update(position.to_le_bytes());
            if let Some((text, cut)) = texts.get(at).and_then(Option::as_ref) {
                hash.update((text.len() as u64).to_le_bytes());
                hash.update(text.as_bytes());
                clipped |= *cut;
            }
        }
        let input_sha256 = format!("{:x}", hash.finalize());
        let disclosed: u64 = sizes[..count].iter().sum();
        let run_total = issued_before + count as u64;
        frame["input_sha256"] = json!(input_sha256);
        frame["issued"] = json!({
            "this_batch": count,
            "run_total": run_total,
            "scope_members": members,
            "not_issued": members.saturating_sub(run_total),
            "batch_index": frame["batch_index"].clone(),
        });
        frame["clipping"] = json!({
            "clipped": clipped || by_bytes || by_allowance,
            "rows_by_bytes": by_bytes,
            "rows_by_allowance": by_allowance,
            "fields": if clipped { json!(["records.excerpt"]) } else { json!([]) },
        });
        frame["disclosure"]["excerpt_bytes"] = json!(disclosed);
        if let Some(left) = allowance {
            frame["disclosure"]["remaining_evidence_bytes"] = json!(left - disclosed);
        }
        frame["cost"]["genai_rows_issued"] = json!(count);
        frame["cost"]["output"] = json!(count);
        let result = wire::settle_response_bytes(&mut frame, &|written| written);
        if wire::transport_bytes(&result) <= PAYLOAD_TRANSPORT_BYTES {
            return Some(Assembled {
                positions: wanted[..count]
                    .iter()
                    .map(|(position, _)| *position)
                    .collect(),
                last_row: wanted[count - 1].1,
                input_sha256,
                disclosed,
                result,
            });
        }
        drop(result);
        by_bytes = true;
        count -= 1;
        if let Some(records) = frame["records"].as_array_mut() {
            records.truncate(count);
        }
    }
}

/// What the judgments of the first `upto` submissions of a run, and of `pending`
/// beside them, say of every judged record: its code, and beside it the outcome or
/// [`CONFLICT`] when two submissions judged it differently.
fn judged(state: &RunState, upto: usize, pending: Option<&Submitted>) -> BTreeMap<u64, (u8, f32)> {
    let mut judged: BTreeMap<u64, (u8, f32)> = BTreeMap::new();
    let all = state.submissions[..upto]
        .iter()
        .flat_map(|submission| &submission.judgments)
        .chain(pending.iter().flat_map(|pending| &pending.judgments));
    for judgment in all {
        let value = match judgment.code {
            code if code == State::In as u8 => 1.0,
            code if code == State::Out as u8 => 2.0,
            _ => 3.0,
        };
        judged
            .entry(judgment.position)
            .and_modify(|known| {
                if known.0 != judgment.code || known.1 == CONFLICT {
                    *known = (State::Unknown as u8, CONFLICT);
                }
            })
            .or_insert((judgment.code, value));
    }
    judged
}

/// The selection collected from the first `upto` submissions of `run`, and `pending`
/// beside them.
fn collected(
    worker: &mut Worker<'_>,
    run: &Arc<Run>,
    (upto, over): (usize, Option<usize>),
    pending: Option<&Submitted>,
    mut extra: Map<String, Value>,
    held: u64,
    started: Instant,
) -> Result<(String, Option<Arc<Set>>), Fault> {
    worker.budget.hold(worker.base.saturating_add(held));
    worker
        .budget
        .require_whole(COLLECT_BYTES, "collecting the submissions of a run")?;
    let held = held.saturating_add(COLLECT_BYTES);
    worker.budget.hold(worker.base.saturating_add(held));
    let (judged, described, coverage, over) = {
        let state = run.state();
        let model = pending
            .map(|pending| (&pending.model, &pending.prompt_revision))
            .or(state.model.as_ref().map(|(model, prompt)| (model, prompt)));
        let Some((model, prompt)) = model else {
            return Err(Fault::refused(
                "the run has no submission yet, so there is nothing to collect; submit \
                 judgments for an issued batch first",
            ));
        };
        let judged = judged(&state, upto, pending);
        let mut inputs = Sha256::new();
        let mut issued = 0u64;
        let over = over.unwrap_or(state.batches.len());
        for batch in &state.batches[..over] {
            inputs.update(batch.input_sha256.as_bytes());
            issued += batch.positions.len() as u64;
        }
        let total = upto + usize::from(pending.is_some());
        let mut by_outcome = [0u64; 4];
        for (_, value) in judged.values() {
            by_outcome[*value as usize - 1] += 1;
        }
        let (members, _) = run.scope.tally.of(State::In);
        let coverage = json!({
            "scope_members": members,
            "issued": issued,
            "judged": judged.len(),
            "recommend": by_outcome[0],
            "pass": by_outcome[1],
            "abstain": by_outcome[2],
            "conflicting": by_outcome[3],
            "omitted": issued - judged.len() as u64,
            "not_issued": members - issued,
        });
        let described = json!({
            "kind": "recommend",
            "version": 1,
            "method": "issued_batch",
            "run": run.name,
            "run_handle": run.handle,
            "model": {"name": wire::display(model, MAX_MODEL_BYTES).0,
                      "identity": "caller_reported"},
            "prompt_revision": wire::display(prompt, MAX_PROMPT_BYTES).0,
            "projection": if run.excerpts { json!(["id", "excerpt"]) } else { json!(["id"]) },
            "input_sha256": format!("{:x}", inputs.finalize()),
            "batches": over,
            "submissions": total,
            "submissions_ref": manifest(&state, upto, pending),
            "membership": "recommend",
            "predicate": "recommend is in, pass is out; abstained, omitted, not issued and \
                          conflicting are not evaluated",
        });
        (judged, described, coverage, over)
    };
    extra.insert("run_coverage".into(), coverage);
    extra.insert(
        "suggestion_only".into(),
        json!("a judgment is a suggestion; no analyst mark is created"),
    );
    let total = upto + usize::from(pending.is_some());
    let listed = Listed {
        scope: &Population::Of(run.scope.clone()),
        producer: Producer::Issued {
            run: run.clone(),
            upto: total,
            batches: over,
            described,
        },
        with_values: true,
        reason: "outside_producer_coverage",
        extra,
        held,
        cost: Cost::new(started),
    };
    select_listed(worker, listed, &mut |put| {
        judged
            .iter()
            .try_for_each(|(position, (code, value))| put(*position, *code, *value))
    })
}

fn submit(
    worker: &mut Worker<'_>,
    run: &Arc<Run>,
    submitted: &Submitted,
    started: Instant,
) -> Result<Done, Fault> {
    let (upto, at, retry, over) = {
        let state = run.state();
        let at = state
            .batches
            .iter()
            .position(|batch| batch.handle == submitted.batch)
            .ok_or_else(|| Fault::runtime("a batch left its run"))?;
        let retry = state
            .submissions
            .iter()
            .position(|known| known.batch == at && known.digest == submitted.digest);
        if retry.is_none() {
            let batch = &state.batches[at];
            if let Some(foreign) = submitted
                .judgments
                .iter()
                .find(|judgment| batch.positions.binary_search(&judgment.position).is_err())
            {
                return Err(Fault::new(
                    "foreign_item",
                    format!(
                        "`{}` was not issued in this batch; a submission names only records \
                         of its batch, so nothing of it was taken",
                        wire::record_id(foreign.position)
                    ),
                ));
            }
            if batch.revision != submitted.expected_revision {
                return Err(Fault::new(
                    "stale_revision",
                    format!(
                        "the batch is at revision {} and the submission expects {}; nothing \
                         of it was taken",
                        batch.revision, submitted.expected_revision
                    ),
                ));
            }
            if batch.revision > MAX_BATCH_SUBMISSIONS {
                return Err(Fault::refused(format!(
                    "the batch took {MAX_BATCH_SUBMISSIONS} submissions, the most of one batch"
                )));
            }
            if let Some((model, prompt)) = &state.model
                && (model != &submitted.model || prompt != &submitted.prompt_revision)
            {
                return Err(Fault::new(
                    "run_mismatch",
                    "the submission reports another model or prompt revision than the first \
                     submission of the run; one coverage revision has one model and one \
                     prompt, so nothing of it was taken",
                ));
            }
        }
        let retry = retry.map(|retry| (retry, state.submissions[retry].batches_then));
        (state.submissions.len(), at, retry, state.batches.len())
    };
    let mut extra = Map::new();
    extra.insert("batch".into(), json!(submitted.batch));
    if let Some((retry, then)) = retry {
        // The same payload again: the selection collected when it was taken.
        extra.insert("retry".into(), json!(true));
        let (result, set) = collected(
            worker,
            run,
            (retry + 1, Some(then)),
            None,
            extra,
            0,
            started,
        )?;
        return Ok(Done {
            result,
            commit: set.map(Commit::Set),
        });
    }
    let grown = submission_bytes(&submitted.digest, &submitted.judgments);
    worker.budget.hold(worker.base);
    worker.budget.require_whole(
        grown,
        "keeping a submission: a judgment and its reason for every record,",
    )?;
    extra.insert(
        "accepted".into(),
        json!({"items": submitted.judgments.len(), "batch_revision": submitted.expected_revision + 1}),
    );
    let (result, set) = collected(
        worker,
        run,
        (upto, Some(over)),
        Some(submitted),
        extra,
        grown,
        started,
    )?;
    // Taken whole, now that what is collected from it is made.
    {
        let mut state = run.state();
        state.submissions.push(Submission {
            batch: at,
            digest: submitted.digest.clone(),
            batches_then: over,
            judgments: submitted
                .judgments
                .iter()
                .map(|judgment| Judgment {
                    position: judgment.position,
                    code: judgment.code,
                    reason: judgment.reason.clone(),
                })
                .collect(),
        });
        state.batches[at].revision += 1;
        if state.model.is_none() {
            state.model = Some((submitted.model.clone(), submitted.prompt_revision.clone()));
        }
    }
    worker
        .shared
        .activity
        .genai_rows_submitted
        .fetch_add(submitted.judgments.len() as u64, Ordering::Relaxed);
    let kept = grown.saturating_add(set.as_ref().map_or(0, |set| set.held));
    worker.budget.hold(worker.base.saturating_add(kept));
    Ok(Done {
        result,
        commit: Some(Commit::Run {
            run: run.clone(),
            handles: Vec::new(),
            grown,
            set,
        }),
    })
}

/// What a collected set says of one record: its outcome and every judgment the
/// submissions it covers gave it.
pub(crate) fn issued_detail(
    run: &Run,
    (upto, batches): (usize, usize),
    position: u64,
    value: f32,
) -> Value {
    let state = run.state();
    let mut judgments = Vec::new();
    for (index, submission) in state.submissions.iter().take(upto).enumerate() {
        if let Ok(at) = submission
            .judgments
            .binary_search_by_key(&position, |judgment| judgment.position)
        {
            let judgment = &submission.judgments[at];
            judgments.push(json!({
                "batch": state.batches[submission.batch].handle,
                "submission": index + 1,
                "outcome": outcome_name(judgment.code),
                "reason": wire::display(&judgment.reason, MAX_REASON_BYTES).0,
            }));
        }
    }
    let issued = state
        .batches
        .iter()
        .take(batches)
        .any(|batch| batch.positions.binary_search(&position).is_ok());
    json!({
        "outcome": if value == 1.0 {
            "recommend"
        } else if value == 2.0 {
            "pass"
        } else if value == 3.0 {
            "abstain"
        } else if value == CONFLICT {
            "conflicting_submissions"
        } else if issued {
            "omitted"
        } else {
            "not_issued"
        },
        "judgments": judgments,
        "kind": "model suggestion; not a reviewed outcome",
    })
}

pub(crate) fn run(worker: &mut Worker<'_>, task: &RecommendTask) -> Result<Done, Fault> {
    let started = Instant::now();
    match task {
        RecommendTask::Venn {
            scope,
            name,
            path,
            membership,
        } => venn(worker, scope, name, path, *membership, started),
        RecommendTask::Issue {
            scope,
            name,
            limit,
            excerpts,
        } => issue(worker, scope, name, *limit, *excerpts, started),
        RecommendTask::Submit { run, submitted } => submit(worker, run, submitted, started),
        RecommendTask::Collect {
            run,
            submissions_ref,
        } => {
            let (upto, current) = {
                let state = run.state();
                let upto = state.submissions.len();
                (upto, manifest(&state, upto, None))
            };
            if let Some(expected) = submissions_ref
                && *expected != current
            {
                return Err(Fault::new(
                    "stale_revision",
                    format!(
                        "`submissions_ref` names another set of submissions than the run \
                         holds now (`{current}`); nothing was collected"
                    ),
                ));
            }
            let (result, set) = collected(worker, run, (upto, None), None, Map::new(), 0, started)?;
            Ok(Done {
                result,
                commit: set.map(Commit::Set),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::DisclosureProfile;
    use crate::service::membership::ABSTAINED;

    /// Requirement 4.4: under the public-development profile nothing of a source the
    /// host did not declare a public fixture is issued or shown, whatever is left of
    /// the allowance; the local profile is not limited.
    #[test]
    fn a_private_source_is_never_disclosable_to_a_development_client() {
        let development = DisclosureProfile::PublicDevelopment;
        for bytes in [0, 1, 1024] {
            let refused = disclosable(development, false, u64::MAX, bytes).unwrap_err();
            assert_eq!(refused.code, "disclosure_refused");
            assert!(refused.message.contains("not declared a public fixture"));
        }
        assert!(disclosable(development, true, 100, 100).is_ok());
        let over = disclosable(development, true, 100, 101).unwrap_err();
        assert_eq!(over.code, "disclosure_refused");
        assert!(over.message.contains("101") && over.message.contains("100"));
        assert!(disclosable(DisclosureProfile::PrivateLocal, false, 0, 1 << 20).is_ok());
    }

    /// What a batch, a submission and the collecting of a full run are asked of the
    /// budget is at least what they allocate: the heap is measured while each is
    /// built as the service builds it.
    #[test]
    fn a_batch_a_submission_and_a_collection_allocate_no_more_than_they_ask_for() {
        let handle = crate::service::wire::handle("batch", "project", &["run", "0"]);
        let (batch, peak) = crate::test_heap::measure(|| {
            let mut positions = Vec::with_capacity(MAX_ROWS);
            positions.extend(0..MAX_ROWS as u64);
            Batch {
                handle: handle.clone(),
                positions,
                input_sha256: format!("{:x}", Sha256::new().finalize()),
                revision: 1,
            }
        });
        assert!(
            peak <= batch_bytes(&batch),
            "a batch allocates {peak} bytes and asks for {}",
            batch_bytes(&batch)
        );
        let judgments: Vec<Judgment> = (0..MAX_ROWS as u64)
            .map(|position| Judgment {
                position,
                code: 1,
                reason: "r".repeat(MAX_REASON_BYTES),
            })
            .collect();
        let digest = format!("{:x}", Sha256::new().finalize());
        let asked = submission_bytes(&digest, &judgments);
        assert!(asked >= (MAX_ROWS * MAX_REASON_BYTES) as u64);
        let (kept, peak) = crate::test_heap::measure(|| Submission {
            batch: 0,
            digest: digest.clone(),
            batches_then: 0,
            judgments: judgments
                .iter()
                .map(|judgment| Judgment {
                    position: judgment.position,
                    code: judgment.code,
                    reason: judgment.reason.clone(),
                })
                .collect(),
        });
        assert!(
            peak <= asked,
            "a submission allocates {peak} bytes and asks for {asked}"
        );
        // A run at its bounds: every batch full and every record judged twice.
        let mut state = RunState::default();
        for batch in 0..MAX_BATCHES {
            for _ in 0..2 {
                state.submissions.push(Submission {
                    batch,
                    digest: String::new(),
                    batches_then: 0,
                    judgments: (0..MAX_ROWS)
                        .map(|at| Judgment {
                            position: (batch * MAX_ROWS + at) as u64,
                            code: 1,
                            reason: String::new(),
                        })
                        .collect(),
                });
            }
        }
        let upto = state.submissions.len();
        let (all, peak) = crate::test_heap::measure(|| judged(&state, upto, None));
        assert_eq!(all.len(), MAX_BATCHES * MAX_ROWS);
        assert!(
            peak <= COLLECT_BYTES,
            "collecting a full run allocates {peak} bytes and asks for {COLLECT_BYTES}"
        );
        drop(kept);
    }

    /// The saved report of the 96-record panel of parent task 2.22, made by the
    /// project's own `l2d triage` with the NPU encoder: every route, reason and
    /// allocation it holds is what a Venn set keeps of it. Run with `--ignored`; the
    /// report is read from `L2D_TRIAGER_POLICY_REPORT` or the task's artefact.
    #[test]
    #[ignore = "reads a saved policy report of parent task 2.22 from the build artefacts"]
    fn the_routes_of_a_real_saved_report_survive_what_a_venn_set_keeps() {
        let path = std::env::var_os("L2D_TRIAGER_POLICY_REPORT").map_or_else(
            || {
                std::path::PathBuf::from(
                    r"D:\BuildArtifacts\log2decision\foreground-2.22\run01\policy-cli-fmy7e_ck\policy-report.json",
                )
            },
            std::path::PathBuf::from,
        );
        let bytes = std::fs::read(&path).unwrap();
        // Counted from the file as JSON, without the product's types.
        let raw: Value = serde_json::from_slice(&bytes).unwrap();
        let snake = |name: &str| {
            let mut out = String::new();
            for character in name.chars() {
                if character.is_ascii_uppercase() && !out.is_empty() {
                    out.push('_');
                }
                out.push(character.to_ascii_lowercase());
            }
            out
        };
        let mut expected: BTreeMap<(String, String, String), u64> = BTreeMap::new();
        for decision in raw["decisions"].as_array().unwrap() {
            let allocation = decision["allocation"]
                .as_str()
                .map_or("deferred".to_string(), snake);
            *expected
                .entry((
                    snake(decision["queue"].as_str().unwrap()),
                    snake(decision["reason"].as_str().unwrap()),
                    allocation,
                ))
                .or_default() += 1;
        }
        let report: PolicyReport = serde_json::from_slice(&bytes).unwrap();
        let report = report.validate_saved().unwrap();
        let mut kept: BTreeMap<(String, String, String), u64> = BTreeMap::new();
        for decision in &report.decisions {
            let (queue, reason, allocation) = decision_indices(decision);
            let detail = venn_detail(venn_value(
                queue,
                reason,
                allocation,
                decision.score.actionable_ratio.is_some(),
                decision.score.comparison_class_assumed,
            ))
            .unwrap();
            assert_eq!(
                detail["actionable_ratio"] == "present",
                decision.score.actionable_ratio.is_some()
            );
            let allocation = detail["allocation"].as_str().unwrap_or("deferred");
            *kept
                .entry((
                    detail["route"].as_str().unwrap().to_string(),
                    detail["reason"].as_str().unwrap().to_string(),
                    allocation.to_string(),
                ))
                .or_default() += 1;
        }
        assert_eq!(kept, expected);
        let routes: std::collections::BTreeSet<&str> =
            kept.keys().map(|(route, _, _)| route.as_str()).collect();
        assert_eq!(routes.len(), 3, "the report holds every route: {kept:?}");
        eprintln!("{} decisions: {kept:?}", report.decisions.len());
    }

    /// A saved report of `decisions` decisions in its densest form: no white space
    /// and the shortest texts, so that what it parses to is as large against its
    /// bytes as a valid report gets.
    fn dense_report(decisions: usize) -> Vec<u8> {
        let rows: Vec<Value> = (0..decisions)
            .map(|index| {
                json!({
                    "input": {"id": format!("{index:08x}"), "text": "a", "group": null},
                    "score": {"kind": "mark-set", "revision": "r",
                              "comparison_class_assumed": true,
                              "diagnostic_ratio": 0.5, "actionable_ratio": null},
                    "vector": null, "queue": "Review", "reason": "Unsupported",
                    "explanation": "incompatible exploration profile; actionable ratio withheld",
                    "allocation": match index { 0 => json!("UnknownConflict"),
                                                1 | 2 => json!("General"), _ => Value::Null },
                })
            })
            .collect();
        serde_json::to_vec(&json!({
            "format": 1,
            "policy": {
                "version": "v", "selection_provenance": "p", "low": 0.2, "high": 0.8,
                "corroboration": 1,
                "reach": {"radius": 0.2, "top_k": 2, "minimum_labelled_support": 1,
                          "expected_profile": "expected"},
                "require_calibration": false, "review_budget": 4, "unknown_slots": 1,
                "audit_slots": 1, "audit_seed": 5
            },
            "eligibility": l2d_decide::policy::ELIGIBILITY,
            "allocation_rule": l2d_decide::policy::ALLOCATION,
            "scorer_revision": "r", "evidence_revision": "h", "actual_profile": "other",
            "calibration": "Missing",
            "qualification": "Exploratory; qualification incomplete. Policy mechanics and local support do not establish deployment quality.",
            "decisions": rows,
            "unknown": {"configured": 1, "selected": 1, "unused": 0},
            "general": {"configured": 2, "selected": 2, "unused": 0},
            "audit": {"configured": 1, "selected": 0, "unused": 1},
            "deferred": decisions - 3
        }))
        .unwrap()
    }

    /// Reading a dense policy report, validating it and listing a state for every
    /// decision allocates no more than the read is charged (review round 1, finding
    /// 5). The heap is counted while the service's own function reads the file.
    #[test]
    fn the_read_of_a_dense_policy_report_stays_within_its_charge() {
        const DECISIONS: usize = 12_000;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dense.json");
        std::fs::write(&path, dense_report(DECISIONS)).unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len > 2 * 1024 * 1024, "{len}");
        let file = File::open(&path).unwrap();
        let ((report, listing), peak) = crate::test_heap::measure(|| {
            let (report, _) = read_policy("dense", file, len).unwrap();
            let mut listing: Vec<(u64, u8, f32)> = Vec::with_capacity(report.decisions.len());
            for decision in &report.decisions {
                let (queue, reason, allocation) = decision_indices(decision);
                listing.push((0, 1, venn_value(queue, reason, allocation, false, true)));
            }
            (report, listing)
        });
        assert_eq!(listing.len(), DECISIONS);
        assert_eq!(report.decisions.len(), DECISIONS);
        let charge = policy_charge(len);
        eprintln!(
            "a report of {len} bytes and {DECISIONS} decisions: heap peak {peak} bytes = {:.2} x \
             its bytes; charged {charge}",
            peak as f64 / len as f64
        );
        assert!(
            peak > len,
            "the measurement saw less than the file itself: {peak}"
        );
        assert!(
            peak <= charge,
            "reading a report of {len} bytes allocated {peak} bytes and is charged {charge}"
        );
    }

    /// Making a full batch of excerpts of the greatest length, the records the
    /// response limit then cuts included, allocates no more than the share `issue`
    /// asks for beside its read of the source (review round 1, finding 5). Every
    /// result it returns fits the limit, and what it says of the batch is what the
    /// batch holds.
    #[test]
    fn a_full_batch_of_excerpts_is_made_within_its_share_and_fits() {
        let long = "e".repeat(MAX_EXCERPT_BYTES + 50);
        let frame = || {
            json!({
                "v": VERSION, "status": "ready", "batch": "batch-0123456789abcdef0123456789abcdef",
                "revision": 1, "run": {"name": "n".repeat(64), "handle": "run-0123456789abcdef"},
                "scope": "set-0123456789abcdef0123456789abcdef",
                "snapshot": "snap-0123456789abcdef0123456789abcdef",
                "cost": {"archive_scans": 1, "elapsed_ms": 123_456, "candidates": 1_000_000},
                "disclosure": {"profile": "public_development", "source_class": "public_fixture",
                               "excerpts": true, "remaining_evidence_bytes": u64::MAX},
                "batch_index": 63,
            })
        };
        let (assembled, peak) = crate::test_heap::measure(|| {
            let mut wanted: Vec<(u64, u64)> = Vec::with_capacity(MAX_ROWS);
            wanted.extend((0..MAX_ROWS as u64).map(|at| ((7 << 40) | at, at * 3)));
            let texts: Vec<Option<(String, bool)>> = (0..MAX_ROWS)
                .map(|_| Some(wire::display(&long, MAX_EXCERPT_BYTES)))
                .collect();
            assemble(
                frame(),
                &wanted,
                &texts,
                Some(u64::MAX / 2),
                6_300,
                1_000_000,
            )
            .unwrap()
        });
        let share = making_share(MAX_ROWS, true) - SCAN_FIXED_BYTES - States::BLOCK_BYTES as u64;
        eprintln!("a full batch: heap peak {peak} bytes; share {share}");
        assert!(
            peak <= share,
            "making a full batch allocated {peak} bytes and asks for {share}"
        );
        let result: Value = serde_json::from_str(&assembled.result).unwrap();
        let records = result["records"].as_array().unwrap();
        assert!(wire::transport_bytes(&assembled.result) <= PAYLOAD_TRANSPORT_BYTES);
        assert!(!records.is_empty() && records.len() < MAX_ROWS);
        assert_eq!(records.len(), assembled.positions.len());
        assert_eq!(assembled.last_row, (records.len() as u64 - 1) * 3);
        assert_eq!(
            assembled.disclosed,
            (records.len() * MAX_EXCERPT_BYTES) as u64
        );
        assert_eq!(result["issued"]["this_batch"], records.len());
        assert_eq!(result["issued"]["run_total"], 6_300 + records.len() as u64);
        assert_eq!(result["clipping"]["rows_by_bytes"], true);
        assert_eq!(result["disclosure"]["excerpt_bytes"], assembled.disclosed);
        assert_eq!(
            result["disclosure"]["remaining_evidence_bytes"],
            u64::MAX / 2 - assembled.disclosed
        );
        // One more record would not have fitted: the batch is cut where it must be.
        let room = PAYLOAD_TRANSPORT_BYTES - wire::transport_bytes(&assembled.result);
        assert!(
            room < 2 * (MAX_EXCERPT_BYTES + 40),
            "{room} bytes were left unused"
        );
        // Every count of wanted records and every allowance gives a result that fits,
        // or none when not even one record does.
        let wanted: Vec<(u64, u64)> = (0..MAX_ROWS as u64).map(|at| (at, at)).collect();
        for step in 0..120usize {
            let texts: Vec<Option<(String, bool)>> = (0..MAX_ROWS)
                .map(|at| Some(("t".repeat(1 + (at * 31 + step * 17) % 400), false)))
                .collect();
            let take = 1 + step % MAX_ROWS;
            let allowance = (step % 7 != 0).then_some((step as u64 * 211) % 30_000);
            match assemble(
                frame(),
                &wanted[..take],
                &texts[..take],
                allowance,
                0,
                5_000,
            ) {
                Some(made) => {
                    assert!(wire::transport_bytes(&made.result) <= PAYLOAD_TRANSPORT_BYTES);
                    assert!(
                        allowance.is_none_or(|left| made.disclosed <= left),
                        "{step}"
                    );
                    let shown: u64 = texts[..made.positions.len()]
                        .iter()
                        .map(|text| text.as_ref().unwrap().0.len() as u64)
                        .sum();
                    assert_eq!(made.disclosed, shown, "{step}");
                }
                None => assert!(
                    allowance
                        .is_some_and(|left| (texts[0].as_ref().unwrap().0.len() as u64) > left),
                    "{step}: no record although the first one fits"
                ),
            }
        }
    }

    #[test]
    fn a_venn_value_keeps_route_reason_and_allocation() {
        let mut seen = std::collections::BTreeSet::new();
        for (queue, route) in QUEUES.iter().enumerate() {
            for (reason, named) in REASONS.iter().enumerate() {
                for allocation in 0..5 {
                    for flags in 0..4 {
                        let value =
                            venn_value(queue, reason, allocation, flags & 1 == 1, flags & 2 == 2);
                        assert!(seen.insert(value.to_bits()), "{value}");
                        let detail = venn_detail(value).unwrap();
                        assert_eq!(detail["route"], *route);
                        assert_eq!(detail["reason"], *named);
                        assert_eq!(detail["deferred_by_policy"], allocation == 0);
                        assert_eq!(
                            detail["actionable_ratio"],
                            if flags & 1 == 1 {
                                "present"
                            } else {
                                "withheld"
                            }
                        );
                        assert_eq!(
                            detail["comparison_class"],
                            if flags & 2 == 2 {
                                "assumed"
                            } else {
                                "explicit"
                            }
                        );
                    }
                }
            }
        }
        for none in [0.0, 0.5, 301.0, f32::NAN, -1.0] {
            assert!(venn_detail(none).is_none());
        }
    }

    #[test]
    fn two_submissions_that_judge_a_record_differently_leave_it_not_evaluated() {
        let judgment = |position, code| Judgment {
            position,
            code,
            reason: String::new(),
        };
        let submission = |judgments| Submission {
            batch: 0,
            digest: String::new(),
            batches_then: 0,
            judgments,
        };
        let state = RunState {
            submissions: vec![
                submission(vec![judgment(1, 1), judgment(2, 0), judgment(3, ABSTAINED)]),
                submission(vec![judgment(1, 1), judgment(2, 1), judgment(4, 0)]),
                submission(vec![judgment(2, 0)]),
            ],
            ..RunState::default()
        };
        let first = judged(&state, 1, None);
        assert_eq!(first[&1], (1, 1.0));
        assert_eq!(first[&2], (0, 2.0));
        assert_eq!(first[&3], (ABSTAINED, 3.0));
        assert!(!first.contains_key(&4));
        // The same judgment again is no conflict; another one is, and stays one.
        let all = judged(&state, 3, None);
        assert_eq!(all[&1], (1, 1.0));
        assert_eq!(all[&2], (State::Unknown as u8, CONFLICT));
        assert_eq!(all[&4], (0, 2.0));
    }
}
