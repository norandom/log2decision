//! The shared triager operations (triager requirements 1.2, 2.1, 2.3, 4.1 to 4.4, 6.1
//! and 6.2): one service that owns snapshots, named sets, views, jobs and disclosure,
//! with the MCP adapter and the CLI adapter of [`crate::mcp`] in front of it.
//!
//! A client opens a source as a snapshot (`triager_open`), derives named sets from a
//! frozen scorer or a filter (`triager_select`), composes them into candidate views
//! with three-valued set algebra (`triager_compose`), reads counts, regions, pages and
//! the evidence of one record (`triager_inspect`), and follows or cancels long calls
//! (`triager_job`). Every call is [`Service::call`] with the tool's name and its
//! arguments as JSON; the result is JSON within the transport limits of the interface
//! contract.
//!
//! # Threads and memory
//!
//! Operations that read a source or make an artifact run one at a time on a worker
//! thread that owns the [`MemoryBudget`]; a call waits for its job for the host's
//! inline wait and otherwise returns the job. Reads of committed artifacts run on the
//! calling thread within [`ADAPTER_FIXED_BYTES`], which the service holds in the
//! budget for its whole life: one request of at most [`crate::mcp::MAX_REQUEST_BYTES`]
//! and its parsed form, one response and its copies on the way out, at most
//! [`MAX_ROWS`] rows of a page, and the table of at most 64 jobs with their results.
//! What a committed artifact keeps is held in the budget from its commit on.
//!
//! # Tables, reuse and cost
//!
//! The membership and the scores of a set are kept in memory when the budget admits
//! them and in files of the work directory otherwise, or always there when the host
//! says so ([`TablePlacement`]); intersections, differences, region counts and pages
//! read either placement in blocks (requirement 6.2). A prepared set is not computed
//! twice: another rule of a scorer revision is derived from the scores a set keeps,
//! and a filter or membership over a part of a snapshot from the set over the whole
//! of it. Regions, pages and evidence read committed tables only, so they inflate
//! nothing and score nothing (requirement 6.3); the overview states the server's work
//! counters, which such calls leave unchanged, and what the session was returned.
//! Every job states its cost in the counters of the scale gate (requirement 6.4).
//!
//! # Drill-down and recommendations (task 2.1)
//!
//! `triager_refine` binds the project's pattern catalogue, its refinement and its
//! exemplar retrieval to these sets ([`refine`]): the patterns of a filter selection,
//! a split and a merge as new revisions with their parent, a preview that changes
//! nothing, and the neighbours of an exemplar as a selection. A pattern becomes a
//! set through the filter syntax `pattern`. The producer `recommend` makes C
//! ([`recommend`]): the routes of a saved Venn policy report the host registered
//! ([`PolicyRef`]), with the scorer revision the policy depended on, or the judgments
//! a model client submits for issued batches. A recommendation is a suggestion: the
//! service keeps no analyst mark.
//!
//! # Scope of this revision
//!
//! Sources are ZIP archives. A selection is a score, a filter, a membership file the
//! host registered ([`MembershipRef`]), a pattern, a neighbourhood or a
//! recommendation. Saved receipts, `triager_save`, allocation, occurrence projections
//! (field and time drill-down) and inference by a generative model on the host
//! (`recommend` with the method `local_model`) are refused by name
//! (`unsupported_operation`); they belong to later tasks of the plan or need a
//! provider this build does not have.

mod inspect;
mod membership;
mod ops;
mod process;
mod recommend;
mod refine;
mod schema;
mod state;
mod tables;
mod wire;

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use l2d_core::memory::MemoryUsage;
use l2d_core::resource::{BusyChoice, ResourceGuard, set_busy_choice};
use l2d_core::selections::SearchQuery;
use l2d_core::zip_input::MAX_RECORD_BYTES;
use l2d_core::{L2dError, LogsFrame, MemoryBudget, ZipInputSettings};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::scope::IndexPlacement;
use inspect::{EVIDENCE_DEFAULT, EVIDENCE_FIELDS, EvidenceTask, Named, RECORD_FIELDS};
use membership::{Expr, check_names};
use ops::{ComposeTask, OpenTask, Rule, SelectProducer, SelectTask, Worker};
use recommend::{RecommendTask, Submitted, VennMembership};
use refine::RefineTask;
use state::{Activity, JobState, Jobs, Keyed, MAX_KEY_BYTES, Registry, Shared, Task};
use wire::{CANCELLED, Fault, MAX_REGION_SETS, MAX_SETS, VERSION};

pub use tables::TablePlacement;
pub use wire::{
    DEFAULT_ROWS, MAX_EXCERPT_BYTES, MAX_MESSAGE_BYTES, MAX_RESULT_BYTES, MAX_ROWS, transport_bytes,
};

/// What the service holds in the budget for the threads that call it: see the module
/// description.
pub const ADAPTER_FIXED_BYTES: u64 = 4 * 1024 * 1024;

/// The blocks a call reads committed tables through are a small part of that share.
const _: () = assert!(inspect::READ_BLOCK_BYTES * 16 <= ADAPTER_FIXED_BYTES);

/// The most bytes the entry patterns of one open take together.
pub const MAX_ENTRY_PATTERN_BYTES: usize = 1024;
/// The most the entry patterns may take of a tool result, in both its forms. With
/// it, 24 reasons of 200 bytes and a source path of 320 bytes, the result of an open
/// stays below the response limit.
const ENTRY_ECHO_TRANSPORT_BYTES: usize = 4 * 1024;

/// The tools this revision serves, in the order they are listed.
pub const TOOLS: [&str; 6] = [
    "triager_open",
    "triager_select",
    "triager_refine",
    "triager_compose",
    "triager_inspect",
    "triager_job",
];
/// Tools of the interface contract that later tasks deliver.
const LATER_TOOLS: [(&str, &str); 1] = [("triager_save", "saved receipts (task 2.2)")];

/// Who may be shown raw evidence. It belongs to the host: no tool argument names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisclosureProfile {
    /// The client is the approved local analyst or a local model: evidence of every
    /// allowed source may be shown.
    PrivateLocal,
    /// The client is a hosted development agent: only sources the host declared
    /// public fixtures are opened, and excerpts count against a session allowance.
    PublicDevelopment,
}

impl DisclosureProfile {
    pub fn name(self) -> &'static str {
        match self {
            DisclosureProfile::PrivateLocal => "private_local",
            DisclosureProfile::PublicDevelopment => "public_development",
        }
    }
}

/// A source the host registered under a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRef {
    pub name: String,
    /// The archive, in the spelling the reader is given.
    pub path: PathBuf,
    /// Whether the host declares it a public fixture.
    pub public_fixture: bool,
}

/// A membership file the host registered under a name: one line per assessed record
/// with its record handle, a tab and `in`, `out` or `abstain`. A record the file does
/// not name was not assessed. A selection names it as
/// `{"kind": "membership", "ref": NAME}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembershipRef {
    pub name: String,
    pub path: PathBuf,
}

/// What the host configures; nothing of it can be changed through a tool argument.
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// The identity of the project: handles are bound to it.
    pub project: String,
    /// The directory the service writes into: absent or empty. Snapshots keep their
    /// relation there when it does not fit the memory, and reads spill there.
    pub work_root: PathBuf,
    pub sources: Vec<SourceRef>,
    /// Directories below which a client may name an archive by its path. Such a
    /// source is private.
    pub allowed_roots: Vec<PathBuf>,
    pub profile: DisclosureProfile,
    /// Under [`DisclosureProfile::PublicDevelopment`]: the excerpt bytes one session
    /// may be shown.
    pub evidence_allowance_bytes: u64,
    /// How long a job-starting call waits for its job before it returns the job;
    /// `None` waits for the end.
    pub inline_wait: Option<Duration>,
    /// Whether a job waits for a busy host or is refused with the holder.
    pub busy: BusyChoice,
    /// The most distinct texts deduplication keeps in memory before it goes on on
    /// disk; `None` leaves that to the memory limit.
    pub dedup_set_texts: Option<u64>,
    pub index: IndexPlacement,
    /// Where selections and views keep their membership and scores.
    pub tables: TablePlacement,
    /// The membership files a selection may name.
    pub memberships: Vec<MembershipRef>,
    /// The free disk space to compare with in place of the operating system's figure.
    pub free_disk_bytes: Option<u64>,
    /// The saved Venn policy reports a recommendation may name.
    pub policies: Vec<PolicyRef>,
    /// The limits of a pattern catalogue, as `l2d discover search` takes them.
    pub patterns: PatternLimits,
    /// The encoder of the neighbour method `vector`; without one that method is
    /// refused by name.
    pub encoder: Option<Encoders>,
}

impl HostConfig {
    /// The neighbour methods this host serves.
    pub(crate) fn neighbour_methods(&self) -> Vec<&'static str> {
        match self.encoder {
            Some(_) => vec!["character_trigram", "vector"],
            None => vec!["character_trigram"],
        }
    }
}

/// The limits of the pattern catalogue `triager_refine` builds: the most patterns,
/// and the most bytes of a text that is grouped. They are part of what a refinement
/// revision binds, and the catalogue reserves memory for all of them before it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternLimits {
    pub max_patterns: usize,
    pub max_text_bytes: usize,
}

impl Default for PatternLimits {
    /// The defaults of `l2d discover search`.
    fn default() -> Self {
        Self {
            max_patterns: 256,
            max_text_bytes: 4096,
        }
    }
}

/// A policy report `l2d triage` saved, registered by the host under a name. A
/// recommendation names it as `{"kind": "recommend", "method": "venn", "policy": NAME}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRef {
    pub name: String,
    pub path: PathBuf,
}

/// Where the service finds the encoder of a vector neighbour search: the project's
/// exploration encoder, opened on the worker for one search and let go after it.
pub trait EncoderSource: Send + Sync {
    /// What an open session is held at in the budget.
    fn session_bytes(&self) -> u64;
    /// The profile and the method the sessions of this source state of themselves:
    /// part of what a neighbour search is named by, before a session is opened.
    fn profile(&self) -> &str;
    fn method(&self) -> &str {
        "direct-exemplar-cosine"
    }
    /// Opens the encoder, after asking `budget` for what that takes. The service
    /// holds [`EncoderSource::session_bytes`] while it uses the encoder.
    fn open(
        &self,
        budget: &mut MemoryBudget,
    ) -> Result<Box<dyn l2d_decide::exploration::ExplorationEncoder>, L2dError>;
}

/// The encoder source of a host.
#[derive(Clone)]
pub struct Encoders(pub Arc<dyn EncoderSource>);

impl std::fmt::Debug for Encoders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Encoders")
    }
}

/// The identity of a frozen scorer: what a score set retains of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScorerIdentity {
    pub kind: String,
    pub name: String,
    pub revision: String,
}

/// A loaded scorer whose scores do not change.
pub trait FrozenScorer {
    fn identity(&self) -> ScorerIdentity;
    /// What the loaded scorer keeps in memory.
    fn held_bytes(&self) -> u64;
    /// What scoring a batch takes beside the batch, for texts of up to
    /// `longest_text_bytes`.
    fn scoring_bytes(&self, longest_text_bytes: u64) -> u64;
    /// Appends one score per row of `logs`, in row order, to `out`, whose capacity
    /// holds them. A higher score ranks first.
    fn score(&self, logs: &LogsFrame, out: &mut Vec<f32>) -> Result<(), L2dError>;
}

/// Where the service finds the scorer a request names.
pub trait ScorerSource: Send {
    /// Loads the scorer `model`, after asking `budget` for what the load takes. The
    /// service holds [`FrozenScorer::held_bytes`] while it uses the scorer.
    fn load(
        &self,
        model: &str,
        budget: &mut MemoryBudget,
    ) -> Result<Box<dyn FrozenScorer>, L2dError>;
}

/// The result of one call: JSON, and whether it is an error result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub json: String,
    pub is_error: bool,
}

impl Reply {
    fn ok(json: String) -> Self {
        Self {
            json,
            is_error: false,
        }
    }

    fn fault(fault: &Fault) -> Self {
        Self {
            json: fault.payload(),
            is_error: true,
        }
    }
}

/// The service: see the module description.
pub struct Service {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
    made_root: bool,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|held| held.into_inner())
}

impl Service {
    /// Starts the service with its worker. `budget` is the one budget of the process;
    /// the adapter's share is held in it from here on.
    ///
    /// [`L2dError::Refused`] when that share does not fit or the work directory holds
    /// something; [`L2dError::InvalidInput`] for a configuration that cannot be used.
    pub fn start(
        config: HostConfig,
        mut budget: MemoryBudget,
        scorers: Box<dyn ScorerSource>,
    ) -> Result<Self, L2dError> {
        let invalid = |detail: String| L2dError::InvalidInput { detail };
        if config.project.is_empty() {
            return Err(invalid("the project identity is empty".into()));
        }
        let names = config
            .sources
            .iter()
            .map(|source| ("source", &source.name))
            .chain(
                config
                    .memberships
                    .iter()
                    .map(|membership| ("membership", &membership.name)),
            )
            .chain(
                config
                    .policies
                    .iter()
                    .map(|policy| ("policy", &policy.name)),
            );
        let mut seen: Vec<(&str, &String)> = Vec::new();
        for (kind, name) in names {
            let usable = (1..=64).contains(&name.len())
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
            if !usable {
                return Err(invalid(format!(
                    "a {kind} name is 1 to 64 letters, digits, '.', '_' or '-'"
                )));
            }
            if seen.contains(&(kind, name)) {
                return Err(invalid(format!(
                    "the {kind} name '{name}' is registered twice"
                )));
            }
            seen.push((kind, name));
        }
        drop(seen);
        budget.require_whole(
            ADAPTER_FIXED_BYTES,
            "serving triager requests, their responses and the table of jobs",
        )?;
        budget.hold(budget.held_bytes().saturating_add(ADAPTER_FIXED_BYTES));
        let made_root = !config.work_root.exists();
        let failed = |error: std::io::Error| L2dError::Runtime {
            detail: format!(
                "preparing the work directory {} failed: {error}",
                config.work_root.display()
            ),
        };
        std::fs::create_dir_all(&config.work_root).map_err(failed)?;
        if std::fs::read_dir(&config.work_root)
            .map_err(failed)?
            .next()
            .is_some()
        {
            return Err(L2dError::Refused {
                reason: format!(
                    "the work directory {} is not empty; name an empty or absent one",
                    config.work_root.display()
                ),
            });
        }
        let shared = Arc::new(Shared {
            allowance: Mutex::new(config.evidence_allowance_bytes),
            usage: budget.usage(),
            held: std::sync::atomic::AtomicU64::new(budget.held_bytes()),
            config,
            registry: RwLock::new(Registry::default()),
            jobs: Mutex::new(Jobs::default()),
            changed: Condvar::new(),
            activity: Activity::default(),
        });
        let for_worker = shared.clone();
        let worker = std::thread::Builder::new()
            .name("triager-worker".into())
            .spawn(move || work(&for_worker, budget, scorers.as_ref()))
            .map_err(|error| L2dError::Runtime {
                detail: format!("starting the triager worker failed: {error}"),
            })?;
        Ok(Self {
            shared,
            worker: Some(worker),
            made_root,
        })
    }

    /// The accounting handle of the service's budget.
    pub fn usage(&self) -> MemoryUsage {
        self.shared.usage.clone()
    }

    /// What the service holds in the budget between jobs: the adapter's share and
    /// what every committed artifact keeps.
    pub fn held_bytes(&self) -> u64 {
        self.shared.held.load(Ordering::Relaxed)
    }

    /// The tools with their input and output schemas, as `tools/list` returns them.
    pub fn tools() -> Value {
        schema::tools()
    }

    /// Whether `tool` is a tool of the interface contract, served or refused by name.
    pub fn knows(tool: &str) -> bool {
        TOOLS.contains(&tool) || LATER_TOOLS.iter().any(|(name, _)| *name == tool)
    }

    /// Runs one tool call. The arguments are validated before anything is scheduled;
    /// a call that starts a job waits for it as long as the host configured.
    pub fn call(&self, tool: &str, args: &Value) -> Reply {
        let activity = &self.shared.activity;
        let reply = match self.dispatch(tool, args) {
            Ok(reply) => reply,
            Err(fault) => {
                if fault.code == "unsupported_operation" {
                    activity.unsupported.fetch_add(1, Ordering::Relaxed);
                }
                Reply::fault(&fault)
            }
        };
        // What the session was returned, counted the way the response limit is.
        activity.calls.fetch_add(1, Ordering::Relaxed);
        activity
            .bytes_returned
            .fetch_add(transport_bytes(&reply.json) as u64, Ordering::Relaxed);
        reply
    }

    fn dispatch(&self, tool: &str, args: &Value) -> Result<Reply, Fault> {
        if let Some((name, what)) = LATER_TOOLS.iter().find(|(name, _)| *name == tool) {
            return Err(Fault::unsupported(format!(
                "`{name}` is not served by this revision: {what}"
            )));
        }
        if !TOOLS.contains(&tool) {
            return Err(Fault::new(
                "unknown_tool",
                format!("`{}` is no triager tool", wire::clip(tool, 64).0),
            ));
        }
        let args = Args::new(args)?;
        match tool {
            "triager_open" => {
                let (task, key) = self.parse_open(&args)?;
                self.start_job("triager_open", Task::Open(task), key)
            }
            "triager_select" => {
                let (task, key) = self.parse_select(&args)?;
                self.start_job("triager_select", task, key)
            }
            "triager_refine" => {
                let (task, key) = self.parse_refine(&args)?;
                self.start_job("triager_refine", Task::Refine(task), key)
            }
            "triager_compose" => {
                let (task, key) = self.parse_compose(&args)?;
                self.start_job("triager_compose", Task::Compose(task), key)
            }
            "triager_inspect" => self.inspect(&args),
            _ => self.job(&args),
        }
    }

    fn registry(&self) -> std::sync::RwLockReadGuard<'_, Registry> {
        ops::registry(&self.shared)
    }

    // --- Requests --------------------------------------------------------------------

    fn parse_open(&self, args: &Args<'_>) -> Result<(OpenTask, Option<(String, String)>), Fault> {
        if args.has("receipt") {
            return Err(Fault::unsupported(
                "opening a saved receipt is not served by this revision (task 2.2); open a \
                 `source`",
            ));
        }
        args.only(&["v", "source", "idempotency_key"])?;
        let source = Args::of(args.required("source")?, "source")?;
        if let Some(adapter) = source.text("adapter")?
            && adapter != "zip"
        {
            return Err(Fault::unsupported(format!(
                "the adapter `{}` is not served by this revision: sources are ZIP archives \
                 (`zip`)",
                wire::clip(adapter, 64).0
            )));
        }
        source.only(&["ref", "path", "adapter", "entries", "max_record_bytes"])?;
        let config = &self.shared.config;
        let (path, named, public) = match (source.text("ref")?, source.text("path")?) {
            (Some(name), None) => {
                let found = config
                    .sources
                    .iter()
                    .find(|known| known.name == name)
                    .ok_or_else(|| {
                        Fault::new(
                            "unknown_source",
                            format!("the host registered no source `{}`", wire::clip(name, 64).0),
                        )
                    })?;
                (
                    found.path.clone(),
                    json!({"ref": name}),
                    found.public_fixture,
                )
            }
            (None, Some(path)) => {
                let path = PathBuf::from(path);
                self.require_allowed(&path)?;
                let shown = wire::display(&path.to_string_lossy(), wire::MAX_PATH_BYTES).0;
                (path, json!({"path": shown}), false)
            }
            _ => {
                return Err(Fault::invalid(
                    "`source` names exactly one of `ref` (a source the host registered) and \
                     `path` (an archive below a directory the host allowed)",
                ));
            }
        };
        if config.profile == DisclosureProfile::PublicDevelopment && !public {
            return Err(Fault::new(
                "disclosure_refused",
                "the host runs the public-development profile, which opens only sources it \
                 declared public fixtures; this source is private",
            ));
        }
        let mut zip = ZipInputSettings::default();
        if let Some(entries) = source.get("entries") {
            let entries = entries
                .as_array()
                .filter(|list| list.len() <= 64)
                .ok_or_else(|| {
                    Fault::invalid("`source.entries` is a list of at most 64 entry patterns")
                })?;
            // The patterns are part of what the snapshot binds, and every result about
            // the source states them whole: they are bounded here, before anything is
            // read, so that the result of the open always fits the response limit.
            let mut total = 0;
            for entry in entries {
                match entry.as_str() {
                    Some(pattern) if !pattern.is_empty() => {
                        total += pattern.len();
                        if total > MAX_ENTRY_PATTERN_BYTES {
                            break;
                        }
                        zip.entries.push(pattern.to_string());
                    }
                    _ => {
                        return Err(Fault::invalid(
                            "an entry pattern in `source.entries` is a string that is not \
                             empty",
                        ));
                    }
                }
            }
            // Written as JSON a pattern can take several times its bytes (a control
            // character is six); what counts is what the result will carry.
            let echoed = transport_bytes(&json!(zip.entries).to_string());
            if total > MAX_ENTRY_PATTERN_BYTES || echoed > ENTRY_ECHO_TRANSPORT_BYTES {
                return Err(Fault::invalid(format!(
                    "the patterns of `source.entries` take more than \
                     {MAX_ENTRY_PATTERN_BYTES} bytes together, or more than \
                     {ENTRY_ECHO_TRANSPORT_BYTES} bytes of a response once written as JSON; \
                     a snapshot states its entry patterns in every result about it, so they \
                     are kept short. Nothing was read"
                )));
            }
        }
        if let Some(bytes) = source.get("max_record_bytes") {
            zip.max_record_bytes = bytes
                .as_u64()
                .filter(|bytes| (1..=MAX_RECORD_BYTES).contains(bytes))
                .ok_or_else(|| {
                    Fault::invalid(format!(
                        "`source.max_record_bytes` is a whole number from 1 to \
                         {MAX_RECORD_BYTES}"
                    ))
                })?;
        }
        let task = OpenTask {
            path,
            zip,
            named,
            public,
        };
        Ok((task, args.key("triager_open")?))
    }

    /// A path a client names must lie below a directory the host allowed, by its
    /// resolved location.
    fn require_allowed(&self, path: &std::path::Path) -> Result<(), Fault> {
        if !path.is_absolute() {
            return Err(Fault::invalid("`source.path` is an absolute path"));
        }
        let refused = || {
            Fault::new(
                "path_not_allowed",
                "`source.path` does not name a file below a directory the host allowed; \
                 nothing was opened",
            )
        };
        let resolved = std::fs::canonicalize(path).map_err(|_| refused())?;
        let allowed =
            self.shared.config.allowed_roots.iter().any(|root| {
                std::fs::canonicalize(root).is_ok_and(|root| resolved.starts_with(root))
            });
        if allowed && resolved.is_file() {
            Ok(())
        } else {
            Err(refused())
        }
    }

    fn parse_select(&self, args: &Args<'_>) -> Result<(Task, Option<(String, String)>), Fault> {
        args.only(&["v", "scope", "producer", "idempotency_key"])?;
        let producer = Args::of(args.required("producer")?, "producer")?;
        let kind = producer.required_text("kind")?;
        if let Some(version) = producer.get("version")
            && version.as_u64() != Some(1)
        {
            return Err(Fault::invalid(
                "`producer.version` is 1, the only version of the producers",
            ));
        }
        let parsed = match kind {
            "score" => {
                producer.only(&["kind", "version", "model", "top_k", "threshold", "ties"])?;
                let model = producer.required_text("model")?;
                if model.is_empty() || model.len() > 128 {
                    return Err(Fault::invalid("`producer.model` is 1 to 128 bytes"));
                }
                let ties = producer.text("ties")?;
                if ties.is_some_and(|ties| ties != "record_id_ascending") {
                    return Err(Fault::invalid(
                        "`producer.ties` is `record_id_ascending`, the one stable tie rule",
                    ));
                }
                let rule = match (producer.get("top_k"), producer.get("threshold")) {
                    (Some(top_k), None) => {
                        let k = top_k.as_u64().filter(|k| *k > 0).ok_or_else(|| {
                            Fault::invalid("`producer.top_k` is a whole number above 0")
                        })?;
                        if ties.is_none() {
                            return Err(Fault::invalid(
                                "a `top_k` needs its tie rule: give `producer.ties` as \
                                 `record_id_ascending`",
                            ));
                        }
                        Rule::TopK(k)
                    }
                    (None, Some(threshold)) => Rule::Threshold(
                        threshold
                            .as_f64()
                            .filter(|threshold| threshold.is_finite())
                            .ok_or_else(|| {
                                Fault::invalid("`producer.threshold` is a finite number")
                            })?,
                    ),
                    _ => {
                        return Err(Fault::invalid(
                            "a score producer takes either `top_k` (with `ties`) or \
                             `threshold`",
                        ));
                    }
                };
                SelectProducer::Score {
                    model: model.to_string(),
                    rule,
                }
            }
            "filter" => {
                let syntax = producer.required_text("syntax")?;
                if syntax == "pattern" {
                    let task = self.parse_pattern(args, &producer)?;
                    return Ok((Task::Refine(task), args.key("triager_select")?));
                }
                producer.only(&["kind", "version", "syntax", "query", "case_insensitive"])?;
                let text = producer.required_text("query")?.to_string();
                // The length the existing search takes; the pattern itself is compiled
                // at the start of the job, where the budget is, before the source is
                // read.
                if text.is_empty() || text.len() > l2d_core::marks::MAX_PATTERN_BYTES {
                    return Err(Fault::invalid(format!(
                        "`producer.query` is 1 to {} bytes",
                        l2d_core::marks::MAX_PATTERN_BYTES
                    )));
                }
                let insensitive = match producer.get("case_insensitive") {
                    None => false,
                    Some(Value::Bool(insensitive)) => *insensitive,
                    Some(_) => {
                        return Err(Fault::invalid(
                            "`producer.case_insensitive` is true or false",
                        ));
                    }
                };
                let (syntax, query) = match syntax {
                    "literal" => ("literal", SearchQuery::Literal { text, insensitive }),
                    "glob" => (
                        "glob",
                        SearchQuery::Glob {
                            pattern: text,
                            insensitive,
                        },
                    ),
                    "regex" => (
                        "regex",
                        SearchQuery::Regex {
                            pattern: text,
                            insensitive,
                        },
                    ),
                    _ => {
                        return Err(Fault::invalid(
                            "`producer.syntax` is `literal`, `glob` or `regex`",
                        ));
                    }
                };
                SelectProducer::Filter { syntax, query }
            }
            "membership" => {
                producer.only(&["kind", "version", "ref"])?;
                let name = producer.required_text("ref")?;
                let found = self
                    .shared
                    .config
                    .memberships
                    .iter()
                    .find(|known| known.name == name)
                    .ok_or_else(|| {
                        Fault::new(
                            "unknown_membership",
                            format!(
                                "the host registered no membership `{}`",
                                wire::clip(name, 64).0
                            ),
                        )
                    })?;
                SelectProducer::Membership {
                    name: found.name.clone(),
                    path: found.path.clone(),
                }
            }
            "recommend" => {
                let task = self.parse_recommend(args, &producer)?;
                return Ok((Task::Recommend(task), args.key("triager_select")?));
            }
            _ => {
                return Err(Fault::invalid(
                    "`producer.kind` is `score`, `filter`, `membership` or `recommend`",
                ));
            }
        };
        let scope = self
            .registry()
            .require_population(args.required_text("scope")?)?;
        Ok((
            Task::Select(SelectTask {
                scope,
                producer: parsed,
            }),
            args.key("triager_select")?,
        ))
    }

    /// A filter with the syntax `pattern`: the members of one pattern of a refinement.
    fn parse_pattern(&self, args: &Args<'_>, producer: &Args<'_>) -> Result<RefineTask, Fault> {
        producer.only(&["kind", "version", "syntax", "refinement", "pattern"])?;
        let registry = self.registry();
        let refinement = registry.require_refinement(producer.required_text("refinement")?)?;
        let scope = registry.require_population(args.required_text("scope")?)?;
        if scope.snapshot().handle != refinement.set.snapshot.handle {
            return Err(Fault::new(
                "mixed_snapshots",
                "`producer.refinement` groups a selection of another snapshot than `scope`",
            ));
        }
        Ok(RefineTask::Select {
            scope,
            refinement,
            pattern: producer.whole("pattern", 0, u64::MAX)?,
        })
    }

    fn parse_recommend(
        &self,
        args: &Args<'_>,
        producer: &Args<'_>,
    ) -> Result<RecommendTask, Fault> {
        let scope = args.required_text("scope")?;
        match producer.required_text("method")? {
            "venn" => {
                producer.only(&["kind", "version", "method", "policy", "membership"])?;
                let name = producer.required_text("policy")?;
                let found = self
                    .shared
                    .config
                    .policies
                    .iter()
                    .find(|known| known.name == name)
                    .ok_or_else(|| {
                        Fault::new(
                            "unknown_policy",
                            format!(
                                "the host registered no policy report `{}`",
                                wire::clip(name, 64).0
                            ),
                        )
                    })?;
                let membership = match producer.required_text("membership")? {
                    "investigate_or_review" => VennMembership::InvestigateOrReview,
                    "investigate" => VennMembership::Investigate,
                    _ => {
                        return Err(Fault::invalid(
                            "`producer.membership` is `investigate_or_review` or `investigate`",
                        ));
                    }
                };
                Ok(RecommendTask::Venn {
                    scope: self.registry().require_population(scope)?,
                    name: found.name.clone(),
                    path: found.path.clone(),
                    membership,
                })
            }
            "issued_batch" => self.parse_issued(scope, producer),
            "local_model" => Err(Fault::unsupported(
                "the method `local_model` runs a generative model on the analysis host, and \
                 this build has no on-host generative provider; a local client takes \
                 `issued_batch` batches instead, and `venn` reads a saved policy report",
            )),
            _ => Err(Fault::invalid(
                "`producer.method` is `venn` or `issued_batch`",
            )),
        }
    }

    /// The three actions of the issued-batch protocol; `scope` is the candidate
    /// scope of the run.
    fn parse_issued(&self, scope: &str, producer: &Args<'_>) -> Result<RecommendTask, Fault> {
        let candidate = |registry: &Registry| {
            registry
                .require_set(scope)
                .map_err(|fault| match fault.code {
                    "wrong_handle_kind" => Fault::invalid(
                        "an issued-batch run takes a bounded candidate scope: `scope` is a \
                     selection or a view, not a whole snapshot",
                    ),
                    _ => fault,
                })
        };
        let run_name = |producer: &Args<'_>| -> Result<String, Fault> {
            let name = producer.text("run")?.unwrap_or("default");
            let usable = (1..=64).contains(&name.len())
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
            if !usable {
                return Err(Fault::invalid(
                    "`producer.run` is 1 to 64 letters, digits, '.', '_' or '-'",
                ));
            }
            Ok(name.to_string())
        };
        let project = &self.shared.config.project;
        match producer.required_text("action")? {
            "issue" => {
                producer.only(&[
                    "kind",
                    "version",
                    "method",
                    "action",
                    "run",
                    "limit",
                    "projection",
                ])?;
                let limit = match producer.get("limit") {
                    None => DEFAULT_ROWS,
                    Some(_) => producer.whole("limit", 1, MAX_ROWS as u64)? as usize,
                };
                let excerpts = match producer.get("projection") {
                    None => false,
                    Some(projection) => {
                        let fields: Option<Vec<&str>> = projection
                            .as_array()
                            .filter(|fields| (1..=2).contains(&fields.len()))
                            .and_then(|fields| fields.iter().map(Value::as_str).collect());
                        match fields.as_deref() {
                            Some(["id"]) => false,
                            Some(["excerpt"] | ["id", "excerpt"] | ["excerpt", "id"]) => true,
                            _ => {
                                return Err(Fault::invalid(
                                    "`producer.projection` is a list of `id` and `excerpt`",
                                ));
                            }
                        }
                    }
                };
                Ok(RecommendTask::Issue {
                    scope: candidate(&self.registry())?,
                    name: run_name(producer)?,
                    limit,
                    excerpts,
                })
            }
            "submit" => {
                producer.only(&[
                    "kind",
                    "version",
                    "method",
                    "action",
                    "batch",
                    "expected_revision",
                    "model",
                    "prompt_revision",
                    "items",
                ])?;
                let batch = producer.required_text("batch")?;
                let run = self.registry().run(batch).filter(|run| run.handle != batch);
                let run = run.ok_or_else(|| {
                    Fault::new(
                        "unknown_handle",
                        format!(
                            "`{}` names no issued batch of this project and server",
                            wire::clip(batch, 80).0
                        ),
                    )
                })?;
                if run.scope.handle != scope {
                    return Err(Fault::new(
                        "run_mismatch",
                        "the batch was issued for another scope than `scope`; a submission \
                         fills coverage only in the scope its batch was issued for",
                    ));
                }
                let bounded = |field: &str, most: usize| -> Result<String, Fault> {
                    let text = producer.required_text(field)?;
                    if text.is_empty() || text.len() > most {
                        return Err(Fault::invalid(format!(
                            "`producer.{field}` is 1 to {most} bytes"
                        )));
                    }
                    Ok(text.to_string())
                };
                let items = producer
                    .required("items")?
                    .as_array()
                    .filter(|items| (1..=MAX_ROWS).contains(&items.len()))
                    .ok_or_else(|| {
                        Fault::invalid(format!(
                            "`producer.items` is a list of 1 to {MAX_ROWS} judgments"
                        ))
                    })?;
                let mut judgments = Vec::with_capacity(items.len());
                for item in items {
                    let item = Args::of(item, "producer.items[]")?;
                    item.only(&["id", "outcome", "reason"])?;
                    let position =
                        wire::record_position(item.required_text("id")?).ok_or_else(|| {
                            Fault::invalid(
                                "an `id` in `producer.items` is a record handle as the batch \
                                 issued it",
                            )
                        })?;
                    let outcome = item.required_text("outcome")?;
                    let code = state::OUTCOMES
                        .iter()
                        .find(|(name, _)| *name == outcome)
                        .map(|(_, code)| *code)
                        .ok_or_else(|| {
                            Fault::invalid(
                                "an `outcome` in `producer.items` is `recommend`, `pass` or \
                                 `abstain`",
                            )
                        })?;
                    let reason = item.text("reason")?.unwrap_or_default();
                    if reason.len() > recommend::MAX_REASON_BYTES {
                        return Err(Fault::invalid(format!(
                            "a `reason` in `producer.items` is at most {} bytes",
                            recommend::MAX_REASON_BYTES
                        )));
                    }
                    judgments.push(state::Judgment {
                        position,
                        code,
                        reason: reason.to_string(),
                    });
                }
                judgments.sort_by_key(|judgment| judgment.position);
                if let Some(twice) = judgments
                    .windows(2)
                    .find(|pair| pair[0].position == pair[1].position)
                {
                    return Err(Fault::new(
                        "duplicate_item",
                        format!(
                            "`{}` is judged twice in one submission; nothing of it was taken",
                            wire::record_id(twice[0].position)
                        ),
                    ));
                }
                let mut hash = Sha256::new();
                canonical_fields(&mut hash, producer.fields, None);
                Ok(RecommendTask::Submit {
                    run,
                    submitted: Submitted {
                        batch: batch.to_string(),
                        expected_revision: producer.whole("expected_revision", 1, u64::MAX)?,
                        model: bounded("model", recommend::MAX_MODEL_BYTES)?,
                        prompt_revision: bounded("prompt_revision", recommend::MAX_PROMPT_BYTES)?,
                        judgments,
                        digest: format!("{:x}", hash.finalize()),
                    },
                })
            }
            "collect" => {
                producer.only(&[
                    "kind",
                    "version",
                    "method",
                    "action",
                    "run",
                    "membership",
                    "submissions_ref",
                ])?;
                if producer
                    .text("membership")?
                    .is_some_and(|membership| membership != "recommend")
                {
                    return Err(Fault::invalid(
                        "`producer.membership` is `recommend`: recommend is in and pass is out",
                    ));
                }
                let name = run_name(producer)?;
                let registry = self.registry();
                let scope = candidate(&registry)?;
                let handle = wire::handle("run", project, &[&scope.handle, &name]);
                let run = registry.run(&handle).ok_or_else(|| {
                    Fault::new(
                        "unknown_run",
                        format!("the scope has no run `{name}`; issue a batch for it first"),
                    )
                })?;
                Ok(RecommendTask::Collect {
                    run,
                    submissions_ref: producer.text("submissions_ref")?.map(str::to_string),
                })
            }
            _ => Err(Fault::invalid(
                "`producer.action` is `issue`, `submit` or `collect`",
            )),
        }
    }

    fn parse_refine(
        &self,
        args: &Args<'_>,
    ) -> Result<(RefineTask, Option<(String, String)>), Fault> {
        args.only(&["v", "set", "refinement", "action", "idempotency_key"])?;
        let action = Args::of(args.required("action")?, "action")?;
        let kind = action.required_text("kind")?;
        if let Some(later) = ["region", "field", "time"]
            .iter()
            .find(|later| **later == kind)
        {
            return Err(Fault::unsupported(format!(
                "the refine action `{later}` is not served by this revision: a region is \
                 composed with `triager_compose`, and field and time drill-down need \
                 occurrence projections (task 3.1); the actions are `patterns`, `list`, \
                 `split`, `merge`, `preview` and `neighbours`"
            )));
        }
        let key = args.key("triager_refine")?;
        if kind == "patterns" {
            action.only(&["kind"])?;
            if args.has("refinement") {
                return Err(Fault::invalid(
                    "`patterns` lists the patterns of a selection: give `set`, not `refinement`",
                ));
            }
            let set = self.registry().require_set(args.required_text("set")?)?;
            return Ok((RefineTask::Patterns { set }, key));
        }
        if args.has("set") {
            return Err(Fault::invalid(
                "this action works on a refinement revision: give `refinement`, the handle \
                 `patterns` returned, not `set`",
            ));
        }
        let registry = self.registry();
        let refinement = registry.require_refinement(args.required_text("refinement")?)?;
        let regex = |required: bool| -> Result<Option<String>, Fault> {
            let regex = match (action.text("regex")?, required) {
                (None, false) => return Ok(None),
                (None, true) => action.required_text("regex")?,
                (Some(regex), _) => regex,
            };
            if regex.is_empty() || regex.len() > l2d_core::marks::MAX_PATTERN_BYTES {
                return Err(Fault::invalid(format!(
                    "`action.regex` is 1 to {} bytes",
                    l2d_core::marks::MAX_PATTERN_BYTES
                )));
            }
            Ok(Some(regex.to_string()))
        };
        let task = match kind {
            "list" => {
                action.only(&["kind", "after", "limit", "projection"])?;
                let templates = match action.get("projection") {
                    None => false,
                    Some(projection) if *projection == json!(["template"]) => true,
                    Some(_) => {
                        return Err(Fault::invalid("`action.projection` is `[\"template\"]`"));
                    }
                };
                RefineTask::List {
                    refinement,
                    page: refine::Page {
                        after: match action.get("after") {
                            None => None,
                            Some(_) => Some(action.whole("after", 0, u64::MAX)?),
                        },
                        limit: match action.get("limit") {
                            None => DEFAULT_ROWS,
                            Some(_) => action.whole("limit", 1, MAX_ROWS as u64)? as usize,
                        },
                        templates,
                    },
                }
            }
            "split" => {
                action.only(&["kind", "pattern", "regex"])?;
                RefineTask::Split {
                    refinement,
                    pattern: action.whole("pattern", 0, u64::MAX)?,
                    regex: regex(true)?.unwrap_or_default(),
                }
            }
            "merge" => {
                action.only(&["kind", "patterns"])?;
                let patterns: Option<Vec<usize>> = action
                    .required("patterns")?
                    .as_array()
                    .filter(|patterns| (2..=256).contains(&patterns.len()))
                    .and_then(|patterns| {
                        patterns
                            .iter()
                            .map(|pattern| pattern.as_u64().map(|pattern| pattern as usize))
                            .collect()
                    });
                RefineTask::Merge {
                    refinement,
                    patterns: patterns.ok_or_else(|| {
                        Fault::invalid("`action.patterns` is a list of 2 to 256 pattern numbers")
                    })?,
                }
            }
            "preview" => {
                action.only(&["kind", "pattern", "regex"])?;
                RefineTask::Preview {
                    refinement,
                    pattern: action.whole("pattern", 0, u64::MAX)?,
                    regex: regex(false)?,
                }
            }
            "neighbours" => {
                action.only(&[
                    "kind", "exemplar", "method", "within", "top_k", "radius", "target",
                ])?;
                let exemplar = wire::record_position(action.required_text("exemplar")?)
                    .ok_or_else(|| {
                        Fault::invalid("`action.exemplar` is a record handle of a member")
                    })?;
                let method = match action.required_text("method")? {
                    "character_trigram" => refine::Method::CharacterTrigram,
                    "vector" if self.shared.config.encoder.is_some() => refine::Method::Vector,
                    "vector" => {
                        return Err(Fault::unsupported(
                            "the neighbour method `vector` needs an encoder profile, and the \
                             host configured none; use `character_trigram`",
                        ));
                    }
                    _ => {
                        return Err(Fault::invalid(
                            "`action.method` is `character_trigram` or `vector`",
                        ));
                    }
                };
                let within = match action.text("within")?.unwrap_or("all") {
                    "pattern" => refine::Within::Pattern,
                    "other_patterns" => refine::Within::OtherPatterns,
                    "all" => refine::Within::All,
                    _ => {
                        return Err(Fault::invalid(
                            "`action.within` is `pattern`, `other_patterns` or `all`",
                        ));
                    }
                };
                let radius = action
                    .required("radius")?
                    .as_f64()
                    .filter(|radius| (0.0..=2.0).contains(radius))
                    .ok_or_else(|| Fault::invalid("`action.radius` is a number from 0 to 2"))?;
                let target = match action.text("target")? {
                    None => None,
                    Some(_) if within == refine::Within::Pattern => {
                        return Err(Fault::invalid(
                            "a search within the exemplar's pattern has no wider `target`; \
                             use `within` `other_patterns` or `all` with it",
                        ));
                    }
                    Some(target) => {
                        let target = registry.require_refinement(target)?;
                        if target.set.snapshot.handle != refinement.set.snapshot.handle {
                            return Err(Fault::new(
                                "mixed_snapshots",
                                "`action.target` groups a selection of another snapshot",
                            ));
                        }
                        Some(target)
                    }
                };
                RefineTask::Neighbours {
                    refinement,
                    exemplar,
                    method,
                    within,
                    top_k: action.whole("top_k", 1, refine::MAX_NEIGHBOURS)?,
                    radius,
                    target,
                }
            }
            _ => {
                return Err(Fault::invalid(
                    "`action.kind` is `patterns`, `list`, `split`, `merge`, `preview` or \
                     `neighbours`",
                ));
            }
        };
        Ok((task, key))
    }

    /// The named sets of a request, each a selection or view of `snapshot`.
    fn named_sets(&self, args: &Args<'_>, snapshot: &str, most: usize) -> Result<Named, Fault> {
        let Some(sets) = args.get("sets") else {
            return Ok(Named::new());
        };
        let sets = sets
            .as_object()
            .ok_or_else(|| Fault::invalid("`sets` maps names to selection or view handles"))?;
        check_names(sets.keys(), most)?;
        let registry = self.registry();
        let mut named = Named::with_capacity(sets.len());
        for (name, handle) in sets {
            let handle = handle
                .as_str()
                .ok_or_else(|| Fault::invalid("a value in `sets` is a handle"))?;
            let set = registry.require_set(handle)?;
            if set.snapshot.handle != snapshot {
                return Err(Fault::new(
                    "mixed_snapshots",
                    format!(
                        "`{name}` is a set of the snapshot `{}` and this call is about `{}`; \
                         sets of different snapshots are not combined",
                        set.snapshot.handle, snapshot
                    ),
                ));
            }
            named.push((name.clone(), set));
        }
        Ok(named)
    }

    fn parse_compose(
        &self,
        args: &Args<'_>,
    ) -> Result<(ComposeTask, Option<(String, String)>), Fault> {
        match args.required_text("mode")? {
            "view" => {}
            "allocate" => {
                return Err(Fault::unsupported(
                    "`mode=allocate` (one review budget, reserves, audit and deferrals) is \
                     not served by this revision (task 2.2); compose with `mode=view`",
                ));
            }
            _ => return Err(Fault::invalid("`mode` is `view`")),
        }
        args.only(&[
            "v",
            "mode",
            "snapshot",
            "scope",
            "sets",
            "expr",
            "idempotency_key",
        ])?;
        let snapshot = self
            .registry()
            .require_snapshot(args.required_text("snapshot")?)?;
        let scope = match args.text("scope")? {
            None => state::Population::All(snapshot.clone()),
            Some(handle) => self.registry().require_population(handle)?,
        };
        if scope.snapshot().handle != snapshot.handle {
            return Err(Fault::new(
                "mixed_snapshots",
                format!(
                    "`scope` is of the snapshot `{}` and `snapshot` is `{}`",
                    scope.snapshot().handle,
                    snapshot.handle
                ),
            ));
        }
        let sets = self.named_sets(args, &snapshot.handle, MAX_SETS)?;
        if sets.is_empty() {
            return Err(Fault::invalid(
                "`sets` names the selections and views the expression uses",
            ));
        }
        let names: Vec<&str> = sets.iter().map(|(name, _)| name.as_str()).collect();
        let expr = Expr::parse(args.required("expr")?, &names)?;
        Ok((
            ComposeTask {
                snapshot,
                scope,
                sets,
                expr,
            },
            args.key("triager_compose")?,
        ))
    }

    /// The projection of a request: a list of distinct fields of `known`, returned
    /// in the order of `known`; `default` when the request names none.
    fn projection(
        args: &Args<'_>,
        known: &'static [&'static str],
        default: &'static [&'static str],
    ) -> Result<Vec<&'static str>, Fault> {
        let Some(projection) = args.get("projection") else {
            return Ok(default.to_vec());
        };
        let refused = || {
            Fault::invalid(format!(
                "`projection` is a list of distinct fields of this view: {}",
                known.join(", ")
            ))
        };
        let asked = projection
            .as_array()
            .filter(|list| !list.is_empty() && list.len() <= known.len())
            .ok_or_else(refused)?;
        let mut fields = Vec::with_capacity(asked.len());
        for field in asked {
            let field = field
                .as_str()
                .and_then(|field| known.iter().find(|known| **known == field))
                .ok_or_else(refused)?;
            if fields.contains(field) {
                return Err(refused());
            }
            fields.push(*field);
        }
        fields.sort_by_key(|field| known.iter().position(|known| known == field));
        Ok(fields)
    }

    fn inspect(&self, args: &Args<'_>) -> Result<Reply, Fault> {
        let view = args.required_text("view")?;
        if let Some(later) = ["histogram", "fields", "lineage"]
            .iter()
            .find(|later| **later == view)
        {
            return Err(Fault::unsupported(format!(
                "the view `{later}` is not served by this revision (occurrence projections \
                 are task 3.1, lineage across saved receipts task 2.2); the views are \
                 `overview`, `regions`, `records` and `evidence`"
            )));
        }
        let population = self
            .registry()
            .require_population(args.required_text("handle")?)?;
        let snapshot = population.snapshot().handle.clone();
        match view {
            "overview" => {
                args.only(&["v", "handle", "view"])?;
                inspect::overview(&self.shared, &population).map(Reply::ok)
            }
            "regions" => {
                args.only(&["v", "handle", "view", "sets"])?;
                let sets = self.named_sets(args, &snapshot, MAX_REGION_SETS)?;
                inspect::regions(&population, &sets).map(Reply::ok)
            }
            "records" => {
                args.only(&[
                    "v",
                    "handle",
                    "view",
                    "sets",
                    "projection",
                    "limit",
                    "cursor",
                    "sort",
                ])?;
                if let Some(sort) = args.get("sort")
                    && *sort != json!(["record_id_ascending"])
                {
                    return Err(Fault::unsupported(format!(
                        "the order {} is not served by this revision; records are paged by \
                         `[\"record_id_ascending\"]`",
                        wire::clip(&sort.to_string(), 120).0
                    )));
                }
                let limit = match args.get("limit") {
                    None => DEFAULT_ROWS,
                    Some(limit) => limit
                        .as_u64()
                        .filter(|limit| (1..=MAX_ROWS as u64).contains(limit))
                        .ok_or_else(|| {
                            Fault::invalid(format!(
                                "`limit` is a whole number from 1 to {MAX_ROWS}"
                            ))
                        })? as usize,
                };
                let sets = self.named_sets(args, &snapshot, MAX_SETS)?;
                let request = inspect::RecordsRequest {
                    population: &population,
                    sets: &sets,
                    projection: Self::projection(args, &RECORD_FIELDS, &RECORD_FIELDS)?,
                    limit,
                    cursor: args.text("cursor")?,
                };
                inspect::records(&self.shared, &request).map(Reply::ok)
            }
            "evidence" => {
                args.only(&[
                    "v",
                    "handle",
                    "view",
                    "sets",
                    "projection",
                    "record",
                    "idempotency_key",
                ])?;
                let row = inspect::find_record(&population, args.required_text("record")?)?;
                let task = EvidenceTask {
                    sets: self.named_sets(args, &snapshot, MAX_SETS)?,
                    projection: Self::projection(args, &EVIDENCE_FIELDS, &EVIDENCE_DEFAULT)?,
                    population,
                    row,
                };
                let key = args.key("triager_inspect")?;
                // What the service keeps of the record, built now: the whole answer
                // when no excerpt is asked for, and otherwise the proof, before the
                // source is read, that the answer will fit with an excerpt in it.
                let retained = inspect::evidence(&self.shared, &task, None)?;
                if !task.wants_excerpt() {
                    if key.is_some() {
                        return Err(Fault::invalid(
                            "`idempotency_key` is taken by calls that start a job; evidence \
                             without `excerpt` in its projection starts none",
                        ));
                    }
                    return Ok(Reply::ok(retained));
                }
                if transport_bytes(&retained) + inspect::EXCERPT_TRANSPORT_BYTES
                    > wire::PAYLOAD_TRANSPORT_BYTES
                {
                    return Err(Fault::invalid(
                        "the evidence of this record with these `sets` leaves no room for an \
                         excerpt within the response limit; name fewer sets or project fewer \
                         fields. The source was not read",
                    ));
                }
                drop(retained);
                // The excerpt is read from the source: a job like every read of a
                // source, refused before it is queued when it may not be shown. A
                // replay of a completed request returns its result and discloses
                // nothing new.
                let shared = self.shared.clone();
                let snapshot = task.population.snapshot().clone();
                self.start_job_when("triager_inspect", Task::Evidence(task), key, || {
                    inspect::excerpt_limit(&shared, &snapshot).map(|_| ())
                })
            }
            _ => Err(Fault::invalid(
                "`view` is `overview`, `regions`, `records` or `evidence`",
            )),
        }
    }

    fn job(&self, args: &Args<'_>) -> Result<Reply, Fault> {
        args.only(&["v", "job", "action"])?;
        let cancel = match args.required_text("action")? {
            "status" => false,
            "cancel" => true,
            _ => return Err(Fault::invalid("`action` is `status` or `cancel`")),
        };
        let id = args.required_text("job")?;
        let jobs = lock(&self.shared.jobs);
        let job = jobs.get(id).ok_or_else(|| {
            Fault::new(
                "unknown_handle",
                format!(
                    "`{}` names no job of this project and server",
                    wire::clip(id, 80).0
                ),
            )
        })?;
        if cancel && !job.state.ended() {
            job.cancel.store(true, Ordering::Relaxed);
            self.shared.changed.notify_all();
        }
        Ok(Reply::ok(self.shared.job_status(id, job)))
    }

    // --- Jobs ------------------------------------------------------------------------

    /// Queues `task`, or finds the job of the same idempotency key and payload, and
    /// waits for it as long as the host configured.
    fn start_job(
        &self,
        tool: &'static str,
        task: Task,
        key: Option<(String, String)>,
    ) -> Result<Reply, Fault> {
        self.start_job_when(tool, task, key, || Ok(()))
    }

    /// [`Service::start_job`] with `admit` asked once it is known that the call is no
    /// replay, before the job is queued.
    fn start_job_when(
        &self,
        tool: &'static str,
        task: Task,
        key: Option<(String, String)>,
        admit: impl FnOnce() -> Result<(), Fault>,
    ) -> Result<Reply, Fault> {
        let mut jobs = lock(&self.shared.jobs);
        if let Some((key, digest)) = &key
            && let Keyed::Job(id) = jobs.keyed(key, digest)?
        {
            return Ok(self.await_job(jobs, &id));
        }
        admit()?;
        let id = jobs.add(&self.shared.config.project, tool, task, key)?;
        self.shared.changed.notify_all();
        Ok(self.await_job(jobs, &id))
    }

    fn await_job(&self, mut jobs: MutexGuard<'_, Jobs>, id: &str) -> Reply {
        let deadline = self
            .shared
            .config
            .inline_wait
            .map(|wait| Instant::now() + wait);
        loop {
            let Some(job) = jobs.get(id) else {
                return Reply::fault(&Fault::runtime("a job left the table while it was awaited"));
            };
            if job.state.ended() {
                return match &job.outcome {
                    Some(Ok(result)) => Reply::ok(result.clone()),
                    Some(Err(fault)) => Reply::fault(fault),
                    None => Reply::fault(&Fault::runtime("a job ended without an outcome")),
                };
            }
            match deadline {
                None => {
                    jobs = self
                        .shared
                        .changed
                        .wait(jobs)
                        .unwrap_or_else(|held| held.into_inner());
                }
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Reply::ok(
                            json!({"v": VERSION, "status": "running", "job": id}).to_string(),
                        );
                    }
                    jobs = self
                        .shared
                        .changed
                        .wait_timeout(jobs, deadline - now)
                        .unwrap_or_else(|held| held.into_inner())
                        .0;
                }
            }
        }
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        lock(&self.shared.jobs).stop_all();
        self.shared.changed.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        // The artifacts go with the service, and with them their work directories.
        self.shared
            .registry
            .write()
            .unwrap_or_else(|held| held.into_inner())
            .clear();
        if self.made_root {
            let _ = std::fs::remove_dir(&self.shared.config.work_root);
        }
    }
}

/// The worker: takes one job after the other, admits it, runs it and commits what it
/// made.
fn work(shared: &Shared, mut budget: MemoryBudget, scorers: &dyn ScorerSource) {
    set_busy_choice(shared.config.busy);
    let mut dirs = 0u64;
    loop {
        // The next job that was not cancelled while it waited.
        let (id, task, cancel, progress, queued) = {
            let mut jobs = lock(&shared.jobs);
            loop {
                if jobs.shutdown {
                    return;
                }
                if let Some(id) = jobs.queue.pop_front() {
                    let Some(job) = jobs.get_mut(&id) else {
                        continue;
                    };
                    let task = job.task.take();
                    if job.cancel.load(Ordering::Relaxed) || task.is_none() {
                        job.state = JobState::Cancelled;
                        job.phase = "cancelled before it started";
                        job.outcome = Some(Err(Fault::new(
                            CANCELLED,
                            "the job was cancelled before it started",
                        )));
                        shared.changed.notify_all();
                        continue;
                    }
                    job.phase = "waiting for the host";
                    let taken = (
                        id.clone(),
                        task.expect("checked above"),
                        job.cancel.clone(),
                        job.progress.clone(),
                        job.queued,
                    );
                    break taken;
                }
                jobs = shared
                    .changed
                    .wait(jobs)
                    .unwrap_or_else(|held| held.into_inner());
            }
        };
        let base = shared.held.load(Ordering::Relaxed);
        budget.hold(base);
        // One heavy operation at a time across processes: the host lease, waited for
        // or refused with its holder, as the host configured.
        let guard = if task.reads_source() {
            match ResourceGuard::tracked("triager job", shared.config.busy, budget.usage()) {
                Ok(guard) => Some(guard),
                Err(error) => {
                    end(shared, &id, JobState::Failed, Err(Fault::from(error)));
                    continue;
                }
            }
        } else {
            None
        };
        if let Some(job) = lock(&shared.jobs).get_mut(&id) {
            job.state = JobState::Running;
            job.phase = match &task {
                Task::Open(_) => "reading the source",
                Task::Select(_) => "reading the source for a selection",
                Task::Compose(_) => "composing a view",
                Task::Evidence(_) => "reading one record",
                Task::Refine(_) => "refining patterns or neighbours",
                Task::Recommend(_) => "making a recommendation",
            };
        }
        shared.changed.notify_all();
        shared.activity.jobs_started.fetch_add(1, Ordering::Relaxed);
        let queue_ms = queued.elapsed().as_millis() as u64;
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut worker = Worker {
                shared,
                budget: &mut budget,
                scorers,
                cancel: &cancel,
                progress: &progress,
                base,
                dirs: &mut dirs,
                queue_ms,
            };
            match &task {
                Task::Open(task) => ops::open(&mut worker, task),
                Task::Select(task) => ops::select(&mut worker, task),
                Task::Compose(task) => ops::compose(&mut worker, task),
                Task::Refine(task) => refine::run(&mut worker, task),
                Task::Recommend(task) => recommend::run(&mut worker, task),
                Task::Evidence(task) => {
                    inspect::evidence_with_excerpt(&mut worker, task).map(|result| ops::Done {
                        result,
                        commit: None,
                    })
                }
            }
        }));
        // A drill-down or a recommendation works with files of the host: its errors
        // are shown without the host's paths.
        let scrubbed = matches!(task, Task::Refine(_) | Task::Recommend(_));
        let ran = match ran {
            Ok(Err(mut fault)) if scrubbed => {
                fault.message = shared.scrub(fault.message);
                Ok(Err(fault))
            }
            ran => ran,
        };
        // The task holds handles of the artifacts it worked on; it is let go before
        // the job is reported as ended.
        drop(task);
        let (state, outcome) = match ran {
            Ok(Ok(done)) => {
                if let Some(commit) = done.commit {
                    // Committed as a whole: the artifact is registered, and what it
                    // keeps is held from here on.
                    let held = base.saturating_add(commit.held());
                    let mut registry = shared
                        .registry
                        .write()
                        .unwrap_or_else(|held| held.into_inner());
                    commit.register(&mut registry);
                    shared.held.store(held, Ordering::Relaxed);
                }
                (JobState::Completed, Ok(done.result))
            }
            Ok(Err(fault)) if fault.code == CANCELLED => (JobState::Cancelled, Err(fault)),
            Ok(Err(fault)) => (JobState::Failed, Err(fault)),
            Err(_) => (
                JobState::Interrupted,
                Err(Fault::new(
                    "interrupted",
                    "the job ended abnormally; nothing of it was committed, and what was \
                     committed before stays usable",
                )),
            ),
        };
        budget.hold(shared.held.load(Ordering::Relaxed));
        if let Some(guard) = guard
            && let Err(error) = guard.finish()
        {
            eprintln!("triager: recording the resources of a job failed: {error}");
        }
        end(shared, &id, state, outcome);
    }
}

fn end(shared: &Shared, id: &str, state: JobState, outcome: Result<String, Fault>) {
    if let Some(job) = lock(&shared.jobs).get_mut(id) {
        job.state = state;
        job.phase = "ended";
        // A kept fault is bounded: the table's share does not depend on a message.
        job.outcome = Some(outcome.map_err(Fault::bounded));
    }
    shared.changed.notify_all();
}

/// The arguments of a call: a JSON object that carries `v`.
struct Args<'a> {
    fields: &'a Map<String, Value>,
    /// How the object is named in messages: empty for the arguments themselves.
    within: &'static str,
}

impl<'a> Args<'a> {
    fn new(args: &'a Value) -> Result<Self, Fault> {
        let fields = args
            .as_object()
            .ok_or_else(|| Fault::invalid("the arguments of a tool call are a JSON object"))?;
        match fields.get("v") {
            None => Err(Fault::invalid(
                "every request carries the version of the interface as `v`; this server \
                 speaks `\"v\": 1`",
            )),
            Some(version) if version.as_u64() == Some(VERSION) => Ok(Self { fields, within: "" }),
            Some(version) => Err(Fault::new(
                "unsupported_version",
                format!(
                    "this server speaks version 1 of the interface, and the request carries \
                     `v`: {}",
                    wire::clip(&version.to_string(), 40).0
                ),
            )),
        }
    }

    fn of(value: &'a Value, within: &'static str) -> Result<Self, Fault> {
        value
            .as_object()
            .map(|fields| Self { fields, within })
            .ok_or_else(|| Fault::invalid(format!("`{within}` is a JSON object")))
    }

    fn name(&self, field: &str) -> String {
        if self.within.is_empty() {
            format!("`{field}`")
        } else {
            format!("`{}.{field}`", self.within)
        }
    }

    fn has(&self, field: &str) -> bool {
        self.fields.contains_key(field)
    }

    fn get(&self, field: &str) -> Option<&'a Value> {
        self.fields.get(field)
    }

    fn required(&self, field: &str) -> Result<&'a Value, Fault> {
        self.get(field)
            .ok_or_else(|| Fault::invalid(format!("{} is required", self.name(field))))
    }

    fn text(&self, field: &str) -> Result<Option<&'a str>, Fault> {
        match self.get(field) {
            None => Ok(None),
            Some(Value::String(text)) => Ok(Some(text)),
            Some(_) => Err(Fault::invalid(format!("{} is a string", self.name(field)))),
        }
    }

    /// A required whole number from `least` to `most`.
    fn whole(&self, field: &str, least: u64, most: u64) -> Result<u64, Fault> {
        self.required(field)?
            .as_u64()
            .filter(|number| (least..=most).contains(number))
            .ok_or_else(|| {
                Fault::invalid(if most == u64::MAX {
                    format!("{} is a whole number of at least {least}", self.name(field))
                } else {
                    format!(
                        "{} is a whole number from {least} to {most}",
                        self.name(field)
                    )
                })
            })
    }

    fn required_text(&self, field: &str) -> Result<&'a str, Fault> {
        self.text(field)?
            .ok_or_else(|| Fault::invalid(format!("{} is required", self.name(field))))
    }

    /// Refuses an argument the tool does not take: nothing is ignored silently.
    fn only(&self, known: &[&str]) -> Result<(), Fault> {
        match self
            .fields
            .keys()
            .find(|field| !known.contains(&field.as_str()))
        {
            None => Ok(()),
            Some(field) => Err(Fault::invalid(format!(
                "{} is no argument here; the arguments are {}",
                self.name(wire::clip(field, 64).0),
                known.join(", ")
            ))),
        }
    }

    /// The idempotency key of a job-starting call with the digest of its payload.
    fn key(&self, tool: &str) -> Result<Option<(String, String)>, Fault> {
        let Some(key) = self.text("idempotency_key")? else {
            return Ok(None);
        };
        if key.is_empty() || key.len() > MAX_KEY_BYTES {
            return Err(Fault::invalid(format!(
                "`idempotency_key` is 1 to {MAX_KEY_BYTES} bytes"
            )));
        }
        let mut hash = Sha256::new();
        hash.update(tool.as_bytes());
        hash.update([0]);
        canonical_fields(&mut hash, self.fields, Some("idempotency_key"));
        Ok(Some((key.to_string(), format!("{:x}", hash.finalize()))))
    }
}

/// Feeds the fields of an object to `hash` by name, `skip` left out.
fn canonical_fields(hash: &mut Sha256, fields: &Map<String, Value>, skip: Option<&str>) {
    let mut names: Vec<&String> = fields
        .keys()
        .filter(|name| Some(name.as_str()) != skip)
        .collect();
    names.sort_unstable();
    hash.update(b"{");
    hash.update((names.len() as u64).to_le_bytes());
    for name in names {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        canonical(hash, &fields[name]);
    }
}

/// Feeds `value` to `hash` in a form that does not depend on the order of the fields
/// of an object.
fn canonical(hash: &mut Sha256, value: &Value) {
    match value {
        Value::Object(fields) => canonical_fields(hash, fields, None),
        Value::Array(items) => {
            hash.update(b"[");
            hash.update((items.len() as u64).to_le_bytes());
            for item in items {
                canonical(hash, item);
            }
        }
        scalar => {
            let text = scalar.to_string();
            hash.update(b"s");
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
    }
}
