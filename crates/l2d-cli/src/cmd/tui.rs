use super::common::Context;
use super::mcp::{HostArgs, start_service};
use clap::{Args, ValueEnum};
use l2d_core::L2dError;
use l2d_triager::bench::{self, Start, Workbench};
use l2d_triager::flow::{CProducer, Counts, Search, SearchKind, Step, Workflow};
use l2d_triager::input::{self, PolicyMembership};
use l2d_triager::{mcp, session, ui};
use serde_json::{Value, json};
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

/// How long a call of the workbench waits for its job before it polls: short, so the
/// terminal stays responsive and Esc can cancel.
const WORKBENCH_INLINE_WAIT: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MatchMode {
    Literal,
    Glob,
    Regex,
    Spl,
}

#[derive(Debug, Args)]
pub struct TuiArgs {
    /// Use the synthetic authentication walkthrough
    #[arg(
        long,
        required_unless_present_any = ["report", "requests", "receipt"],
        conflicts_with_all = ["report", "requests", "receipt"]
    )]
    demo: bool,
    /// Open the workbench over the triager service at the handles a recipe of tool calls
    /// binds (as `l2d mcp --requests` runs it): the snapshot bound last, every bound
    /// selection as a named set (by its binding's name in lower case, for example a, b
    /// and c) and the view or final set bound last as the first view
    #[arg(long, value_name = "FILE", conflicts_with_all = ["report", "receipt"])]
    requests: Option<PathBuf>,
    /// Open the workbench over the triager service at a receipt that `l2d mcp` or the
    /// workbench saved in --receipts; its sets keep their names. An excerpt reads the
    /// source registered under the name the receipt states
    #[arg(long, value_name = "RECEIPT", conflicts_with = "report")]
    receipt: Option<String>,
    /// Workbench over the service: write every call, frame and resize as a JSON line
    /// to this file, with the time of the layout, of the drawing and of each call
    #[arg(long, value_name = "FILE")]
    trace: Option<PathBuf>,
    /// The host options of `l2d mcp`, for --requests and --receipt
    #[command(flatten)]
    host: HostArgs,
    /// Read a retained report written by l2d triage --policy ... --out FILE
    #[arg(long, value_name = "FILE")]
    report: Option<PathBuf>,
    /// Treat the saved Investigate and Review routes as C, including unsupported cases routed to Review
    #[arg(long, requires = "report")]
    policy_recommendations: bool,
    /// Save with s, or reopen this native receipt if it already exists
    #[arg(long, value_name = "FILE")]
    session: Option<PathBuf>,
    /// Size of score set A in a new investigation [default: 8]
    #[arg(long, value_name = "K")]
    top_k: Option<usize>,
    /// Query defining set B in a new investigation [default: *login*]
    #[arg(long, value_name = "QUERY")]
    filter: Option<String>,
    /// Syntax of the set B query [default: glob]
    #[arg(long, value_enum)]
    match_mode: Option<MatchMode>,
    /// Open a child of the current view with an SPL query: a search, optionally piped to where, regex, rex, head, tail, sort, dedup or fields
    #[arg(long, value_name = "EXPRESSION")]
    query: Option<String>,
}

fn invalid(detail: String) -> L2dError {
    L2dError::InvalidInput { detail }
}

/// The workbench over the triager service (triager task 3.2): the tool calls of
/// `l2d mcp` on a service of this process, from a recipe or a receipt.
fn run_workbench(context: &Context, args: &TuiArgs) -> Result<(), L2dError> {
    let retained = [
        ("--session", args.session.is_some()),
        ("--top-k", args.top_k.is_some()),
        ("--filter", args.filter.is_some()),
        ("--match-mode", args.match_mode.is_some()),
        ("--query", args.query.is_some()),
        ("--policy-recommendations", args.policy_recommendations),
    ];
    if let Some((option, _)) = retained.iter().find(|(_, given)| *given) {
        return Err(invalid(format!(
            "{option} belongs to --demo and --report; a recipe or a receipt defines the sets \
             of the workbench over the service"
        )));
    }
    if args.host.given().contains(&"--inline-wait-ms") {
        return Err(invalid(format!(
            "--inline-wait-ms belongs to `l2d mcp`; the workbench waits {} ms for a job and \
             then polls it, so that Esc can cancel it",
            WORKBENCH_INLINE_WAIT.as_millis()
        )));
    }
    if args.trace.is_some() && context.json {
        return Err(invalid(
            "--trace records an interactive session; --json prints the first view".into(),
        ));
    }
    let service = start_service(context, &args.host, Some(WORKBENCH_INLINE_WAIT))?;
    let start = match (&args.requests, &args.receipt) {
        (Some(path), _) => {
            let recipe = std::fs::File::open(path).map_err(|_| L2dError::Missing {
                item: "recipe of tool calls (`--requests`)".into(),
                expected_at: path.clone(),
            })?;
            if !context.json {
                eprintln!("Running the recipe {} …", path.display());
            }
            let bound = mcp::run_requests_bound(&service, recipe, std::io::sink())?;
            Start::from_bindings(&bound).map_err(invalid)?
        }
        (None, Some(receipt)) => {
            let reply = mcp::settle(
                &service,
                service.call("triager_open", &json!({"v": 1, "receipt": receipt})),
            );
            let value: Value = serde_json::from_str(&reply.json).unwrap_or_default();
            if reply.is_error {
                return Err(L2dError::Refused {
                    reason: format!(
                        "reopening the receipt failed: {} ({})",
                        value["error"]["message"].as_str().unwrap_or(""),
                        value["error"]["code"].as_str().unwrap_or("error")
                    ),
                });
            }
            Start::from_reopened(&value).map_err(invalid)?
        }
        (None, None) => unreachable!("run_workbench is called with --requests or --receipt"),
    };
    let mut workbench = Workbench::new(service, start, args.trace.as_deref()).map_err(invalid)?;
    if context.json {
        let summary = workbench.summary();
        let mut out = std::io::stdout().lock();
        writeln!(out, "{summary}").map_err(crate::output::output_failed)?;
        return Ok(());
    }
    bench::run(workbench).map_err(|detail| L2dError::Runtime { detail })
}

pub fn run(context: &Context, args: &TuiArgs) -> Result<(), L2dError> {
    if args.requests.is_some() || args.receipt.is_some() {
        return run_workbench(context, args);
    }
    // The retained-report workbench starts no service: an option of the workbench over
    // the service is refused, not ignored.
    let mut service_only = args.host.given();
    if args.trace.is_some() {
        service_only.insert(0, "--trace");
    }
    if !service_only.is_empty() {
        return Err(invalid(format!(
            "{} belongs to the workbench over the service: give --requests or --receipt, or \
             leave it out with --demo and --report",
            service_only.join(", ")
        )));
    }
    let budget = context.budget()?;
    let (receipt, revision) = match &args.session {
        Some(path) if path.try_exists().map_err(|e| invalid(e.to_string()))? => {
            let (receipt, revision) = session::load_versioned(path).map_err(invalid)?;
            (Some(receipt), Some(revision))
        }
        _ => (None, None),
    };
    let use_policy = args.policy_recommendations
        || receipt
            .as_ref()
            .is_some_and(|r| r.c_producer == CProducer::PolicyInvestigateOrReview);
    let data = match &args.report {
        Some(path) if use_policy => input::load_report_with_membership(
            path,
            &budget,
            PolicyMembership::InvestigateOrReview,
        )?,
        Some(path) => input::load_report(path, &budget)?,
        None => input::demo(),
    };
    let mut flow = if let Some(receipt) = receipt {
        if args.top_k.is_some() || args.filter.is_some() || args.match_mode.is_some() {
            return Err(invalid(
                "this receipt already defines A and B; choose a new session to change them".into(),
            ));
        }
        Workflow::reopen_with_budget(data, &receipt, budget)
    } else {
        let kind = match args.match_mode.unwrap_or(MatchMode::Glob) {
            MatchMode::Literal => SearchKind::Literal,
            MatchMode::Glob => SearchKind::Glob,
            MatchMode::Regex => SearchKind::Regex,
            MatchMode::Spl => SearchKind::Spl,
        };
        let query = args.filter.clone().unwrap_or_else(|| "*login*".into());
        Workflow::with_budget(
            data,
            args.top_k.unwrap_or(8),
            Search { kind, query },
            budget,
        )
    }
    .map_err(invalid)?;
    if let Some(query) = &args.query {
        flow.drill(Step::Search(Search {
            kind: SearchKind::Spl,
            query: query.clone(),
        }))
        .map_err(invalid)?;
    }
    if context.json {
        let view = flow.view();
        let counts = |v: Counts| v.for_receipt(flow.dataset().occurrences_known);
        let regions: std::collections::BTreeMap<_, _> = view
            .regions
            .iter()
            .map(|(region, value)| (region, counts(*value)))
            .collect();
        let preview: Vec<Value> = view
            .members
            .iter()
            .take(16)
            .map(|m| {
                json!({
                    "id": flow.dataset().records[m.record].id, "pool": m.allocation,
                })
            })
            .collect();
        return crate::output::report(
            context,
            flow.memory(),
            &json!({
                "source":flow.dataset().title, "source_revision":flow.dataset().source_revision,
                "c_producer":flow.dataset().c_producer,
                "view":view.id, "parent":view.parent, "title":view.title, "allocated":view.is_final(),
                "counts":counts(view.counts), "regions":regions, "preview":preview,
                "truncated":view.members.len() > preview.len(),
            }),
        );
    }
    ui::run(flow, args.session.as_deref(), revision).map_err(|detail| L2dError::Runtime { detail })
}
