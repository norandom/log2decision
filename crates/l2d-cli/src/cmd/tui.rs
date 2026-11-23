use super::common::Context;
use clap::{Args, ValueEnum};
use l2d_core::L2dError;
use l2d_triager::flow::{CProducer, Counts, Search, SearchKind, Step, Workflow};
use l2d_triager::input::{self, PolicyMembership};
use l2d_triager::{session, ui};
use serde_json::{Value, json};
use std::path::PathBuf;

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
    #[arg(long, required_unless_present = "report", conflicts_with = "report")]
    demo: bool,
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

pub fn run(context: &Context, args: &TuiArgs) -> Result<(), L2dError> {
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
