//! `l2d mcp`: the triager operations as a local MCP server on standard input and
//! output, or, with `--requests`, as a recipe of tool calls read from a file (triager
//! task 1.2, requirement 4.1). Both go through `l2d_triager::service::Service`.
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
    DisclosureProfile, FrozenScorer, HostConfig, MembershipRef, ScorerIdentity, ScorerSource,
    Service, SourceRef, TablePlacement,
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
    /// Register a local ZIP archive under a name; private evidence. May be given several times
    #[arg(long = "source", value_name = "NAME=PATH")]
    sources: Vec<String>,
    /// Register a ZIP archive as an explicitly public fixture. May be given several times
    #[arg(long = "public-fixture", value_name = "NAME=PATH")]
    public_fixtures: Vec<String>,
    /// A directory below which a client may name an archive by its path. May be given several times
    #[arg(long = "allow-path", value_name = "DIR")]
    allow_paths: Vec<PathBuf>,
    /// Who the client is, for what may be shown of the evidence
    #[arg(long, value_enum, default_value = "private-local")]
    disclosure: Disclosure,
    /// public-development: the excerpt bytes one session may be shown
    #[arg(long, value_name = "BYTES", default_value_t = 262_144)]
    evidence_allowance_bytes: u64,
    /// The directory snapshots and reads spill into; empty or absent
    /// [default: work/triager-<process>-<time>/ below the project]
    #[arg(long, value_name = "DIR")]
    work_dir: Option<PathBuf>,
    /// How long a call waits for its job before it returns the job to poll
    #[arg(long, value_name = "MS", default_value_t = 2_000)]
    inline_wait_ms: u64,
    /// The most distinct texts deduplication keeps in memory before it goes on on disk
    /// [default: what the memory limit allows]
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
    dedup_set_texts: Option<u64>,
    /// Keep the locator index of a snapshot and the membership and scores of every set
    /// in files of the work directory, also when the memory limit admits them
    #[arg(long)]
    tables_on_disk: bool,
    /// Register a membership file under a name: one `<record handle><TAB>in|out|abstain`
    /// per assessed record. A selection names it as producer `membership`. May be given
    /// several times
    #[arg(long = "membership", value_name = "NAME=PATH")]
    memberships: Vec<String>,
    /// Run the tool calls of this file, one JSON object {"tool","args","bind"} per line,
    /// write one {"tool","result"} per line and end; every call waits for its job
    #[arg(long, value_name = "FILE")]
    requests: Option<PathBuf>,
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
        profile: match args.disclosure {
            Disclosure::PrivateLocal => DisclosureProfile::PrivateLocal,
            Disclosure::PublicDevelopment => DisclosureProfile::PublicDevelopment,
        },
        evidence_allowance_bytes: args.evidence_allowance_bytes,
        // A recipe has nobody to poll: every call waits for its job.
        inline_wait: match args.requests {
            Some(_) => None,
            None => Some(Duration::from_millis(args.inline_wait_ms)),
        },
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
    };
    let service = Service::start(config, budget, Box::new(ProjectScorers { paths }))?;
    let output = io::stdout().lock();
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
