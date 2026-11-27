//! `l2d mcp`: the triager operations as a local MCP server on standard input and
//! output, or, with `--requests`, as a recipe of tool calls read from a file (triager
//! task 1.2, requirement 4.1), or, with `--replay`, as the recipe a saved receipt
//! retains (task 2.2, requirement 3.3). All go through `l2d_triager::service::Service`
//! and save receipts in `--receipts`.
//!
//! Standard output carries JSON-RPC messages (or the result lines of a recipe) and
//! nothing else; diagnostics go to standard error. The sources, the allowed
//! directories and the disclosure profile are options of this command: the host's,
//! which no tool argument changes. `--memory-limit`, `--batch-rows` and `--busy` are
//! the global options of every command.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Args, ValueEnum};
use l2d_baseline::scorer::{BaselineScorer, SCORES_RATIO};
use l2d_core::polars::prelude::Column;
use l2d_core::{L2dError, LogsFrame, MemoryBudget, ProjectPaths};
use l2d_triager::mcp;
use l2d_triager::scope::IndexPlacement;
use l2d_triager::service::{
    DisclosureProfile, EncoderSource, Encoders, FrozenScorer, HostConfig, MembershipRef, PolicyRef,
    ScorerIdentity, ScorerSource, Service, SourceRef, TablePlacement,
};

use super::common::{Context, find_model, load_scorer, model_name};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Disclosure {
    /// The client is the local analyst or a local model
    PrivateLocal,
    /// The client is a hosted development agent: only public fixtures are opened
    PublicDevelopment,
}

#[derive(Debug, Args)]
pub struct McpArgs {
    #[command(flatten)]
    pub host: HostArgs,
    /// Run the tool calls of this file, one JSON object {"tool","args","bind"} per line,
    /// write one {"tool","result"} per line and end; every call waits for its job
    #[arg(long, value_name = "FILE", conflicts_with = "replay")]
    requests: Option<PathBuf>,
    /// Run the recipe a saved receipt retains, as --requests runs a file, against the
    /// sources registered now; the receipt it saves names this one as its parent
    #[arg(long, value_name = "RECEIPT")]
    replay: Option<String>,
}

/// What the host configures for the triager service: the options `l2d mcp` and the
/// workbench of `l2d tui` share (triager task 3.2), none of which a tool argument
/// changes.
#[derive(Debug, Args)]
pub struct HostArgs {
    /// Register a local ZIP archive under a name; private evidence. May be given several times
    #[arg(long = "source", value_name = "NAME=PATH")]
    sources: Vec<String>,
    /// Register a ZIP archive as an explicitly public fixture. May be given several times
    #[arg(long = "public-fixture", value_name = "NAME=PATH")]
    public_fixtures: Vec<String>,
    /// A directory below which a client may name an archive by its path. May be given several times
    #[arg(long = "allow-path", value_name = "DIR")]
    allow_paths: Vec<PathBuf>,
    /// Who the client is, for what may be shown of the evidence [default: private-local]
    #[arg(long, value_enum)]
    disclosure: Option<Disclosure>,
    /// public-development: the excerpt bytes one session may be shown [default: 262144]
    #[arg(long, value_name = "BYTES")]
    evidence_allowance_bytes: Option<u64>,
    /// The directory snapshots and reads spill into; empty or absent
    /// [default: work/triager-<process>-<time>/ below the project]
    #[arg(long, value_name = "DIR")]
    work_dir: Option<PathBuf>,
    /// How long a call waits for its job before it returns the job to poll (the
    /// workbench of `l2d tui` waits 50 ms and polls, so Esc can cancel) [default: 2000]
    #[arg(long, value_name = "MS")]
    inline_wait_ms: Option<u64>,
    /// The most distinct texts deduplication keeps in memory before it goes on on disk
    /// [default: what the memory limit allows]
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
    dedup_set_texts: Option<u64>,
    /// Keep the locator index of a snapshot and the membership and scores of every set
    /// in files of the work directory, also when the memory limit admits them
    #[arg(long)]
    tables_on_disk: bool,
    /// Register a membership file under a name: JSON lines, one
    /// {"id": "<record handle>", "state": "in"|"out"|"abstain"} per assessed record, at most
    /// 128 bytes a line. A selection names it as producer `membership`. May be given
    /// several times
    #[arg(long = "membership", value_name = "NAME=PATH")]
    memberships: Vec<String>,
    /// Register a policy report `l2d triage --out` saved under a name. A selection names
    /// it as producer `recommend` with the method `venn`. May be given several times
    #[arg(long = "policy", value_name = "NAME=PATH")]
    policies: Vec<String>,
    /// The verified NPU encoder profile of the neighbour method `vector`, as
    /// `l2d discover explore --method npu --profile-dir` takes it [default: that method
    /// is refused by name]
    #[arg(long, value_name = "DIR")]
    npu_profile_dir: Option<PathBuf>,
    /// The most patterns of a catalogue `triager_refine` builds, as
    /// `l2d discover search --max-patterns` [default: 256]
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..=4096))]
    max_patterns: Option<u64>,
    /// The most bytes of a text such a catalogue groups, as
    /// `l2d discover search --max-text-bytes` [default: 4096]
    #[arg(long, value_name = "BYTES", value_parser = clap::value_parser!(u64).range(1..=1_048_576))]
    max_text_bytes: Option<u64>,
    /// The directory receipts are saved in and reopened from; it outlives the work
    /// directory [default: receipts/ below the project]
    #[arg(long, value_name = "DIR")]
    receipts: Option<PathBuf>,
}

impl HostArgs {
    /// The host options given on the command line, by their names: what a command that
    /// starts no service refuses rather than ignores.
    pub fn given(&self) -> Vec<&'static str> {
        [
            ("--source", !self.sources.is_empty()),
            ("--public-fixture", !self.public_fixtures.is_empty()),
            ("--allow-path", !self.allow_paths.is_empty()),
            ("--disclosure", self.disclosure.is_some()),
            (
                "--evidence-allowance-bytes",
                self.evidence_allowance_bytes.is_some(),
            ),
            ("--work-dir", self.work_dir.is_some()),
            ("--inline-wait-ms", self.inline_wait_ms.is_some()),
            ("--dedup-set-texts", self.dedup_set_texts.is_some()),
            ("--tables-on-disk", self.tables_on_disk),
            ("--membership", !self.memberships.is_empty()),
            ("--policy", !self.policies.is_empty()),
            ("--npu-profile-dir", self.npu_profile_dir.is_some()),
            ("--max-patterns", self.max_patterns.is_some()),
            ("--max-text-bytes", self.max_text_bytes.is_some()),
            ("--receipts", self.receipts.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, given)| given.then_some(name))
        .collect()
    }
}

fn invalid(detail: String) -> L2dError {
    L2dError::InvalidInput { detail }
}

/// A fitted baseline scorer of the project, as a frozen scorer of the triager.
struct Baseline {
    scorer: BaselineScorer,
    name: String,
}

impl FrozenScorer for Baseline {
    fn identity(&self) -> ScorerIdentity {
        ScorerIdentity {
            kind: self.scorer.scorer_kind().to_string(),
            name: self.name.clone(),
            revision: self.scorer.revision().to_string(),
        }
    }

    fn held_bytes(&self) -> u64 {
        self.scorer.held_bytes()
    }

    fn scoring_bytes(&self, longest_text_bytes: u64) -> u64 {
        self.scorer.scoring_bytes(longest_text_bytes)
    }

    /// The bad/good ratio of every row: the score `l2d baseline score` writes.
    fn score(&self, logs: &LogsFrame, out: &mut Vec<f32>) -> Result<(), L2dError> {
        let failed = |error| L2dError::Runtime {
            detail: format!("reading the ratios of a batch failed: {error}"),
        };
        let scores = self.scorer.score(logs)?;
        let ratios = scores
            .column(SCORES_RATIO)
            .and_then(Column::f32)
            .map_err(failed)?;
        out.extend(ratios.iter().map(|ratio| ratio.unwrap_or(f32::NAN)));
        Ok(())
    }
}

/// The saved baseline scorers of the project: `producer.model` names one as
/// `--baseline` does.
struct ProjectScorers {
    paths: ProjectPaths,
}

impl ScorerSource for ProjectScorers {
    fn load(
        &self,
        model: &str,
        budget: &mut MemoryBudget,
    ) -> Result<Box<dyn FrozenScorer>, L2dError> {
        let directory = find_model(
            &self.paths.baselines_dir(),
            "producer.model",
            "baseline scorer",
            model,
        )?;
        let scorer = load_scorer(&self.paths, &directory, budget)?;
        Ok(Box::new(Baseline {
            scorer,
            name: model_name(&directory),
        }))
    }
}

/// The project's hierarchical NPU encoder, opened for one neighbour search.
struct NpuEncoder {
    profile: PathBuf,
    runtime: PathBuf,
    cache: PathBuf,
}

impl EncoderSource for NpuEncoder {
    fn session_bytes(&self) -> u64 {
        l2d_encoder::hierarchy::SESSION_BYTES
    }

    fn profile(&self) -> &str {
        l2d_encoder::hierarchy::PROFILE
    }

    fn open(
        &self,
        budget: &mut MemoryBudget,
    ) -> Result<Box<dyn l2d_decide::exploration::ExplorationEncoder>, L2dError> {
        let session = l2d_encoder::hierarchy::HierarchicalSession::open(
            &self.profile,
            &self.runtime,
            &self.cache,
            budget,
        )?;
        Ok(Box::new(session))
    }
}

/// `NAME=PATH` options as pairs.
fn named(given: &[String], what: &str) -> Result<Vec<(String, PathBuf)>, L2dError> {
    given
        .iter()
        .map(|text| match text.split_once('=') {
            Some((name, path)) if !name.is_empty() && !path.is_empty() => {
                Ok((name.to_string(), PathBuf::from(path)))
            }
            _ => Err(invalid(format!(
                "a {what} is given as NAME=PATH, and `{text}` is not"
            ))),
        })
        .collect()
}

fn sources(given: &[String], public_fixture: bool) -> Result<Vec<SourceRef>, L2dError> {
    given
        .iter()
        .map(|text| match text.split_once('=') {
            Some((name, path)) if !name.is_empty() && !path.is_empty() => Ok(SourceRef {
                name: name.to_string(),
                path: PathBuf::from(path),
                public_fixture,
            }),
            _ => Err(invalid(format!(
                "a source is given as NAME=PATH, and `{text}` is not"
            ))),
        })
        .collect()
}

pub fn run(context: &Context, args: &McpArgs) -> Result<(), L2dError> {
    // A recipe has nobody to poll: every call waits for its job.
    let inline_wait = match (&args.requests, &args.replay) {
        (None, None) => Some(Duration::from_millis(
            args.host.inline_wait_ms.unwrap_or(2_000),
        )),
        _ => None,
    };
    let service = start_service(context, &args.host, inline_wait)?;
    let output = io::stdout().lock();
    if let Some(receipt) = &args.replay {
        return mcp::run_recipe(&service, receipt, output);
    }
    match &args.requests {
        Some(path) => {
            let recipe = std::fs::File::open(path).map_err(|_| L2dError::Missing {
                item: "recipe of tool calls (`--requests`)".into(),
                expected_at: path.clone(),
            })?;
            mcp::run_requests(&service, recipe, output)
        }
        None => {
            mcp::serve(&service, io::stdin().lock(), output).map_err(|error| L2dError::Runtime {
                detail: format!("serving MCP on the standard streams failed: {error}"),
            })
        }
    }
}

/// Starts the triager service the host options describe, with the project's fitted
/// baseline scorers; a job-starting call waits `inline_wait` for its job (`None`:
/// to its end).
pub fn start_service(
    context: &Context,
    args: &HostArgs,
    inline_wait: Option<Duration>,
) -> Result<Service, L2dError> {
    let paths = context.paths()?;
    let budget = context.budget()?;
    let project = paths.work_dir();
    let project = project.parent().unwrap_or(Path::new("."));
    let mut registered = sources(&args.sources, false)?;
    registered.extend(sources(&args.public_fixtures, true)?);
    let work_root = args.work_dir.clone().unwrap_or_else(|| {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_millis());
        paths
            .work_dir()
            .join(format!("triager-{}-{stamp}", std::process::id()))
    });
    let config = HostConfig {
        project: project.to_string_lossy().into_owned(),
        work_root,
        sources: registered,
        allowed_roots: args.allow_paths.clone(),
        profile: match args.disclosure.unwrap_or(Disclosure::PrivateLocal) {
            Disclosure::PrivateLocal => DisclosureProfile::PrivateLocal,
            Disclosure::PublicDevelopment => DisclosureProfile::PublicDevelopment,
        },
        evidence_allowance_bytes: args.evidence_allowance_bytes.unwrap_or(262_144),
        inline_wait,
        busy: l2d_core::resource::busy_choice(),
        dedup_set_texts: args.dedup_set_texts,
        index: if args.tables_on_disk {
            IndexPlacement::Disk
        } else {
            IndexPlacement::MemoryWhenAdmitted
        },
        tables: if args.tables_on_disk {
            TablePlacement::Disk
        } else {
            TablePlacement::MemoryWhenAdmitted
        },
        memberships: args
            .memberships
            .iter()
            .map(|text| match text.split_once('=') {
                Some((name, path)) if !name.is_empty() && !path.is_empty() => Ok(MembershipRef {
                    name: name.to_string(),
                    path: PathBuf::from(path),
                }),
                _ => Err(invalid(format!(
                    "a membership is given as NAME=PATH, and `{text}` is not"
                ))),
            })
            .collect::<Result<_, _>>()?,
        free_disk_bytes: None,
        policies: named(&args.policies, "policy")?
            .into_iter()
            .map(|(name, path)| PolicyRef { name, path })
            .collect(),
        patterns: l2d_triager::service::PatternLimits {
            max_patterns: args.max_patterns.unwrap_or(256) as usize,
            max_text_bytes: args.max_text_bytes.unwrap_or(4096) as usize,
        },
        encoder: args.npu_profile_dir.as_ref().map(|profile| {
            Encoders(std::sync::Arc::new(NpuEncoder {
                profile: profile.clone(),
                runtime: paths
                    .runtime_files_dir(l2d_core::paths::runtime_dir_from_environment().as_deref()),
                cache: std::env::var_os("L2D_NPU_WORK_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| std::env::temp_dir().join("l2d-hierarchy"))
                    .join("cache"),
            }))
        }),
        receipts: Some(
            args.receipts
                .clone()
                .unwrap_or_else(|| project.join("receipts")),
        ),
    };
    Service::start(config, budget, Box::new(ProjectScorers { paths }))
}
