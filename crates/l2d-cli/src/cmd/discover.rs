//! First discovery workflow. Every write creates a new artifact; no inferred labels.
use super::common::{self, ArchiveArgs, AtOnceDefault, Context, InputArgs};
use clap::{Args, Subcommand, ValueEnum};
use l2d_core::patterns::refinement::{PatternRefinement, Priority, QueryPreview};
use l2d_core::patterns::{PatternCatalogue, PatternSettings};
use l2d_core::selections::{ReviewAction, ReviewState, SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{L2dError, Label, MarkedSource, MemoryBudget, ProjectPaths};
use serde::ser::{SerializeMap, Serializer};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Subcommand)]
pub enum DiscoverCommand {
    /// Broad search and initial patterns; writes selection.json, catalogue/ and refinement.json
    Search(SearchArgs),
    /// List patterns, real examples, counts, exceptions, and ungrouped remainder
    Patterns(ArtifactArgs),
    /// Inspect every exact member of a pattern (or the ungrouped remainder)
    Details(DetailsArgs),
    /// Split a pattern by an explicitly supplied regex; both sides must be nonempty
    Split(SplitArgs),
    /// Merge pattern IDs, retaining all members and showing differing outcome tokens
    Merge(MergeArgs),
    /// Promote, hide or restore priority; never changes labels
    Priority(PriorityArgs),
    /// Preview an edited or generated regex against members and all other selected members
    Preview(PreviewArgs),
    /// Save a query only after checking its saved preview against this exact revision
    SaveQuery(SaveQueryArgs),
    /// Preview explicit batch review; --apply requires the previewed revision
    Review(ReviewArgs),
    /// Export reviewed labels with a SelectionSet receipt in an immutable bundle
    Export(ExportArgs),
    /// Retrieve directly similar originals; explicit method and outside-query scope
    Explore(ExploreArgs),
}
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ExploreMethod {
    Npu,
    CharacterTrigram,
}
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ExploreScope {
    WithinPattern,
    AcrossPatterns,
    AllPatterns,
}
#[derive(Debug, Args)]
pub struct ExploreArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    /// Zero-based row in saved refinement membership (see details)
    #[arg(long)]
    pub exemplar_row: usize,
    #[arg(long, value_enum)]
    pub method: ExploreMethod,
    #[arg(long, value_enum, default_value = "all-patterns")]
    pub scope: ExploreScope,
    #[arg(long, default_value_t = 20)]
    pub top_k: usize,
    #[arg(long, default_value_t = 0.2)]
    /// NPU cosine distance, or lexical 1-Jaccard; these scales are not calibrated alike
    pub radius: f64,
    /// Explicit outside-query scope: another source-bound saved refinement
    #[arg(long)]
    pub expand_artifact: Option<PathBuf>,
    /// Immutable 2.17 f0-s0-mask directory (two graphs/table/layout/exp-table)
    #[arg(long)]
    pub profile_dir: Option<PathBuf>,
    /// Require the exact profile from a previous report when replaying
    #[arg(long)]
    pub expect_profile: Option<String>,
    /// New immutable output bundle with report.json
    #[arg(long)]
    pub out: PathBuf,
}
#[derive(Debug, Args)]
pub struct SearchArgs {
    #[command(flatten)]
    pub input: InputArgs,
    #[command(flatten)]
    pub archive: ArchiveArgs,
    /// Existing project store, instead of --input
    #[arg(long, conflicts_with = "input")]
    pub store: Option<String>,
    #[arg(long, group = "query")]
    pub phrase: Option<String>,
    /// Whole-record content glob, distinct from --entry archive path filtering
    #[arg(long, group = "query")]
    pub glob: Option<String>,
    #[arg(long, group = "query")]
    pub regex: Option<String>,
    /// Store-only SQLite FTS5 trigram expression (intrinsically insensitive)
    #[arg(long, group = "query")]
    pub fts: Option<String>,
    #[arg(long)]
    pub insensitive: bool,
    #[arg(long, default_value = "selection")]
    pub name: String,
    #[arg(long, default_value_t = 10000)]
    pub max_members: u64,
    #[arg(long, default_value_t = 256)]
    pub max_patterns: usize,
    #[arg(long, default_value_t = 4096)]
    pub max_text_bytes: usize,
    /// New output directory; never replaces existing artifacts
    #[arg(long)]
    pub out: PathBuf,
}
#[derive(Debug, Args)]
pub struct ArtifactArgs {
    #[arg(long)]
    pub artifact: PathBuf,
}
#[derive(Debug, Args)]
pub struct DetailsArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    #[arg(
        long,
        required_unless_present = "ungrouped",
        conflicts_with = "ungrouped"
    )]
    pub pattern: Option<usize>,
    #[arg(long)]
    pub ungrouped: bool,
}
#[derive(Debug, Args)]
pub struct SplitArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    #[arg(long)]
    pub pattern: usize,
    #[arg(long)]
    pub regex: String,
    #[arg(long)]
    pub out: PathBuf,
}
#[derive(Debug, Args)]
pub struct MergeArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    #[arg(long, value_delimiter = ',')]
    pub patterns: Vec<usize>,
    #[arg(long)]
    pub out: PathBuf,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum PriorityArg {
    Normal,
    Promoted,
    Hidden,
}
#[derive(Debug, Args)]
pub struct PriorityArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    #[arg(long)]
    pub pattern: usize,
    #[arg(long, value_enum)]
    pub priority: PriorityArg,
    #[arg(long)]
    pub out: PathBuf,
}
#[derive(Debug, Args)]
pub struct PreviewArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    #[arg(long)]
    pub pattern: usize,
    /// If omitted, propose escaped literals with only observed/masked variable slots
    #[arg(long)]
    pub regex: Option<String>,
    /// Save the preview receipt; inspect it before discover save-query
    #[arg(long)]
    pub out: Option<PathBuf>,
}
#[derive(Debug, Args)]
pub struct SaveQueryArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    #[arg(long)]
    pub preview: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum StateArg {
    Good,
    Bad,
    Unresolved,
    Conflict,
}
impl StateArg {
    fn state(self) -> ReviewState {
        match self {
            Self::Good => ReviewState::Good,
            Self::Bad => ReviewState::Bad,
            Self::Unresolved => ReviewState::Unresolved,
            Self::Conflict => ReviewState::Conflict,
        }
    }
}
#[derive(Debug, Args)]
pub struct ReviewArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    #[arg(long)]
    pub selection: PathBuf,
    #[arg(long)]
    pub pattern: usize,
    #[arg(long, value_enum)]
    pub state: StateArg,
    #[arg(long)]
    pub analyst: String,
    #[arg(long)]
    pub action: String,
    /// Explicitly apply; otherwise only preview affected members
    #[arg(long, requires_all=["expect_revision", "out"])]
    pub apply: bool,
    /// Review revision printed by the preceding preview (pins target, label and action too)
    #[arg(long, requires = "apply")]
    pub expect_revision: Option<String>,
    #[arg(long, requires = "apply")]
    pub out: Option<PathBuf>,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ExportLabel {
    Good,
    Bad,
}
#[derive(Debug, Args)]
pub struct ExportArgs {
    #[command(flatten)]
    pub artifact: ArtifactArgs,
    #[arg(long)]
    pub selection: PathBuf,
    #[arg(long, value_enum)]
    pub label: ExportLabel,
    #[arg(long)]
    pub name: String,
    /// New bundle containing marks/<name>.json, selection.json and refinement.json
    #[arg(long)]
    pub out: PathBuf,
}
fn invalid(s: impl Into<String>) -> L2dError {
    L2dError::InvalidInput { detail: s.into() }
}
fn failed(e: impl std::fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!("discovery: {e}"),
    }
}
fn output(context: &Context, b: &MemoryBudget, value: &serde_json::Value) -> Result<(), L2dError> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    let memory = context.memory_report(b, Vec::new());
    if context.json {
        let fields = value
            .as_object()
            .ok_or_else(|| failed("result must be an object"))?;
        let extra = usize::from(!fields.contains_key("memory"));
        let mut serializer = serde_json::Serializer::new(&mut out);
        let mut map = serializer
            .serialize_map(Some(fields.len() + extra))
            .map_err(failed)?;
        for (key, value) in fields {
            map.serialize_entry(key, value).map_err(failed)?;
        }
        if extra != 0 {
            map.serialize_entry("memory", &memory).map_err(failed)?;
        }
        map.end().map_err(failed)?;
    } else {
        human(&mut out, value, 0).map_err(failed)?;
        if value.get("memory").is_none() {
            writeln!(out, "memory:").map_err(failed)?;
            human(&mut out, &serde_json::to_value(&memory).map_err(failed)?, 2).map_err(failed)?;
        }
    }
    writeln!(out).map_err(failed)?;
    out.flush().map_err(failed)
}
fn human(out: &mut impl Write, value: &serde_json::Value, indent: usize) -> io::Result<()> {
    match value {
        serde_json::Value::Object(fields) => {
            for (name, value) in fields {
                write!(out, "{:indent$}{name}: ", "")?;
                if value.is_array() || value.is_object() {
                    writeln!(out)?;
                }
                human(out, value, indent + 2)?;
            }
        }
        serde_json::Value::Array(rows) => {
            if rows.is_empty() {
                writeln!(out, "{:indent$}(none)", "")?;
            }
            for row in rows {
                writeln!(out, "{:indent$}-", "")?;
                human(out, row, indent + 2)?;
            }
        }
        serde_json::Value::String(s) => writeln!(out, "{}", crate::output::shown_text(s, s.len()))?,
        other => writeln!(out, "{other}")?,
    }
    Ok(())
}
/// Directory publication makes review/export bundles indivisible.
fn bundle<T>(out: &Path, build: impl FnOnce(&Path) -> Result<T, L2dError>) -> Result<T, L2dError> {
    if out.exists() {
        return Err(failed(
            "output already exists; choose a new revision destination",
        ));
    }
    let parent = out
        .parent()
        .ok_or_else(|| invalid("output needs a parent"))?;
    fs::create_dir_all(parent).map_err(failed)?;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let stage = parent.join(format!(
        ".discover-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::create_dir(&stage).map_err(failed)?;
    let result = build(&stage).and_then(|v| {
        if out.exists() {
            return Err(failed("output appeared during publication"));
        }
        fs::rename(&stage, out).map_err(failed)?;
        Ok(v)
    });
    if result.is_err() {
        let _ = fs::remove_dir_all(&stage);
    }
    result
}
fn load(args: &ArtifactArgs, b: &mut MemoryBudget) -> Result<PatternRefinement, L2dError> {
    let r = PatternRefinement::load(&args.artifact, b)?;
    // Keep a fixed output reservation through the entire command, including a
    // retained review report beside old/new revisions. JSON Values have per-key
    // allocations beyond text bytes; no report is constructed before this hold.
    hold_output(b, r.held_bytes())?;
    Ok(r)
}
fn hold_output(b: &mut MemoryBudget, bytes: u64) -> Result<(), L2dError> {
    let held = bytes.saturating_mul(32).saturating_add(4 * 1024 * 1024);
    b.require_whole(
        held,
        "retained discovery artifacts, JSON report and output buffers",
    )?;
    b.hold(b.held_bytes().saturating_add(held));
    Ok(())
}
fn current(context: &Context, r: &PatternRefinement, b: &mut MemoryBudget) -> Result<(), L2dError> {
    match &r.catalogue.source {
        MarkedSource::Input(_) => r.require_current_input(),
        MarkedSource::Store(s) => {
            let store = l2d_store::Store::open(&context.paths()?, &s.name)?;
            r.require_current_store(&store, b)
        }
    }
}
fn search(context: &Context, a: &SearchArgs, b: &mut MemoryBudget) -> Result<(), L2dError> {
    const PLAN_BYTES: u64 = 1024 * 1024;
    b.require_whole(PLAN_BYTES, "bounded discovery batch reports")?;
    b.hold(b.held_bytes().saturating_add(PLAN_BYTES));
    let mut operations = Vec::with_capacity(2);
    let mut store_batches = Vec::with_capacity(2);
    if [
        a.phrase.is_some(),
        a.glob.is_some(),
        a.regex.is_some(),
        a.fts.is_some(),
    ]
    .into_iter()
    .filter(|x| *x)
    .count()
        != 1
    {
        return Err(invalid(
            "give exactly one of --phrase, --glob, --regex or --fts",
        ));
    }
    if a.store.is_none() && a.input.input.is_none() {
        return Err(invalid("discover search requires --input or --store"));
    }
    let query = if let Some(text) = &a.phrase {
        SearchQuery::Literal {
            text: text.clone(),
            insensitive: a.insensitive,
        }
    } else if let Some(pattern) = &a.glob {
        SearchQuery::Glob {
            pattern: pattern.clone(),
            insensitive: a.insensitive,
        }
    } else if let Some(pattern) = &a.regex {
        SearchQuery::Regex {
            pattern: pattern.clone(),
            insensitive: a.insensitive,
        }
    } else {
        SearchQuery::Fts5Trigram {
            expression: a.fts.clone().unwrap_or_default(),
        }
    };
    let paths = context.paths()?;
    let settings = PatternSettings {
        max_patterns: a.max_patterns,
        max_text_bytes: a.max_text_bytes,
        ..Default::default()
    };
    let search = SearchSettings {
        max_members: a.max_members,
    };
    let (selection, refinement) = bundle(&a.out, |stage| {
        let (selection, catalogue) = if let Some(name) = &a.store {
            a.archive.refuse_for_other_input(&[])?;
            let store = l2d_store::Store::open(&paths, name)?;
            let observed = super::discover_store::ObservedStore::new(&store, b.max_batch_rows());
            let s = SelectionSet::search_store(&a.name, query, search, &observed, b)?;
            let c = PatternCatalogue::build_store(
                &s,
                &observed,
                &settings,
                &stage.join("catalogue"),
                b,
            )?;
            store_batches.push(observed.selection.into_inner());
            store_batches.push(observed.catalogue.into_inner());
            (s, c)
        } else if let Some(path) = common::archive_input(&paths, &a.input)? {
            let options = a.archive.settings(AtOnceDefault::WhenTheyFit);
            let mut selected_plan = None;
            let (s, summary) = SelectionSet::search_archive_reported(
                &a.name,
                query,
                search,
                &path,
                &options,
                b,
                &mut |p| capture_plan(&mut selected_plan, p),
            )?;
            super::ingest::fail_when_damaged(&summary, &path)?;
            let records = summary.input.records;
            if let Some(plan) = selected_plan {
                operations.push(common::Operation::batches(
                    "archive selection scan",
                    records,
                    plan,
                ));
            }
            drop(summary);
            let mut catalogue_plan = None;
            let c = PatternCatalogue::build_archive_reported(
                &s,
                &path,
                &options,
                &settings,
                &stage.join("catalogue"),
                b,
                &mut |p| capture_plan(&mut catalogue_plan, p),
            )?;
            if let Some(plan) = catalogue_plan {
                operations.push(common::Operation::batches(
                    "archive catalogue scan",
                    records,
                    plan,
                ));
            }
            (s, c)
        } else {
            a.archive.refuse_for_other_input(&[])?;
            let input = common::read_input(&paths, &a.input, b)?;
            let s =
                SelectionSet::search_input(&a.name, query, search, &input.logs, &input.summary, b)?;
            let c = PatternCatalogue::build_input(
                &s,
                &input.logs,
                &input.summary,
                &settings,
                &stage.join("catalogue"),
                b,
            )?;
            let mut read_plan = None;
            capture_plan(&mut read_plan, &input.plan)?;
            if let Some(plan) = read_plan {
                operations.push(common::Operation::batches(
                    "read",
                    input.summary.records,
                    plan,
                ));
            }
            drop(input);
            b.hold(PLAN_BYTES);
            (s, c)
        };
        b.require_whole(
            selection
                .held_bytes()
                .saturating_add(catalogue.held_bytes())
                .saturating_add(PLAN_BYTES),
            "retaining selection and catalogue",
        )?;
        b.hold(
            selection
                .held_bytes()
                .saturating_add(catalogue.held_bytes())
                .saturating_add(PLAN_BYTES),
        );
        let r = PatternRefinement::from_catalogue(&catalogue, &stage.join("catalogue"), b)?;
        b.hold(
            selection
                .held_bytes()
                .saturating_add(catalogue.held_bytes())
                .saturating_add(r.held_bytes())
                .saturating_add(PLAN_BYTES),
        );
        current(context, &r, b)?;
        hold_output(b, r.held_bytes().saturating_add(selection.held_bytes()))?;
        selection.save(&stage.join("selection.json"), b)?;
        r.save(&stage.join("refinement.json"), b)?;
        Ok((selection, r))
    })?;
    output(
        context,
        b,
        &serde_json::json!({"saved":a.out,"selection_revision":selection.revision(b)?,"refinement_revision":refinement.revision,"source":selection.source(),"query":selection.query(),"completeness":selection.completeness(),"patterns":refinement.groups,"ungrouped":refinement.ungrouped,"labels":"unlabelled; source dataset labels are not reviews", "memory":context.memory_report(b,operations),"store_batches":store_batches}),
    )
}
fn capture_plan(
    slot: &mut Option<l2d_core::BatchPlan>,
    plan: &l2d_core::BatchPlan,
) -> Result<(), L2dError> {
    if plan.changes.len() > 1024 {
        return Err(failed(
            "more than 1024 batch changes exceed the bounded discovery report; narrow the search",
        ));
    }
    *slot = Some(plan.clone());
    Ok(())
}
pub fn run(context: &Context, command: &DiscoverCommand) -> Result<(), L2dError> {
    let mut b = context.budget()?;
    let guard = l2d_core::resource::ResourceGuard::tracked(
        "discovery",
        l2d_core::resource::busy_choice(),
        b.usage(),
    )?;
    let result = execute(context, command, &mut b);
    match (result, guard.finish()) {
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
        (Ok(v), Ok(_)) => Ok(v),
    }
}
fn execute(
    context: &Context,
    command: &DiscoverCommand,
    b: &mut MemoryBudget,
) -> Result<(), L2dError> {
    match command {
        DiscoverCommand::Explore(a) => explore(context, a, b),
        DiscoverCommand::Search(a) => search(context, a, b),
        DiscoverCommand::Patterns(a) => {
            let r = load(a, b)?;
            output(
                context,
                b,
                &serde_json::json!({"revision":r.revision,"parent_revision":r.parent_revision,"action":r.action,"source":r.catalogue.source,"selection_revision":r.selection_revision,"patterns":r.groups,"priorities":r.priorities,"ungrouped":r.ungrouped,"template_semantics":"discovery summaries; use preview to measure actual regex differences"}),
            )
        }
        DiscoverCommand::Details(a) => {
            let r = load(&a.artifact, b)?;
            if a.pattern
                .is_some_and(|id| !r.groups.iter().any(|p| p.id == id))
            {
                return Err(invalid("unknown pattern id"));
            }
            output(
                context,
                b,
                &serde_json::json!({"revision":r.revision,"pattern":a.pattern,"members":r.rows.iter().filter(|row|row.pattern==a.pattern).collect::<Vec<_>>(),"exploration_exemplars":r.rows.iter().enumerate().filter(|(_,row)|row.pattern==a.pattern).map(|(i,row)|serde_json::json!({"exemplar_row":i,"member":row.member,"text":row.text})).collect::<Vec<_>>(),"scope":"exact saved members, including exceptions and differing outcome tokens"}),
            )
        }
        DiscoverCommand::Split(a) => {
            let r = load(&a.artifact, b)?;
            current(context, &r, b)?;
            let next = r.split(a.pattern, &a.regex, b)?;
            next.save(&a.out, b)?;
            output(
                context,
                b,
                &serde_json::json!({"saved":a.out,"refinement":next}),
            )
        }
        DiscoverCommand::Merge(a) => {
            let r = load(&a.artifact, b)?;
            current(context, &r, b)?;
            let next = r.merge(&a.patterns, b)?;
            next.save(&a.out, b)?;
            output(
                context,
                b,
                &serde_json::json!({"saved":a.out,"refinement":next}),
            )
        }
        DiscoverCommand::Priority(a) => {
            let r = load(&a.artifact, b)?;
            current(context, &r, b)?;
            let p = match a.priority {
                PriorityArg::Normal => Priority::Normal,
                PriorityArg::Promoted => Priority::Promoted,
                PriorityArg::Hidden => Priority::Hidden,
            };
            let next = r.prioritize(a.pattern, p, b)?;
            next.save(&a.out, b)?;
            output(
                context,
                b,
                &serde_json::json!({"saved":a.out,"revision":next.revision,"parent_revision":next.parent_revision,"priorities":next.priorities,"labels":"unchanged"}),
            )
        }
        DiscoverCommand::Preview(a) => {
            let r = load(&a.artifact, b)?;
            current(context, &r, b)?;
            let regex = match &a.regex {
                Some(q) => q.clone(),
                None => r.proposed_regex(a.pattern, b)?,
            };
            let p = r.preview_query(a.pattern, &regex, b)?;
            if let Some(out) = &a.out {
                r.save_query(&p, out, b)?;
            }
            output(context, b, &serde_json::json!(p))
        }
        DiscoverCommand::SaveQuery(a) => {
            let r = load(&a.artifact, b)?;
            current(context, &r, b)?;
            let p = QueryPreview::load(&a.preview, b)?;
            hold_output(b, p.held_bytes())?;
            r.save_query(&p, &a.out, b)?;
            output(
                context,
                b,
                &serde_json::json!({"saved":a.out,"verified_preview":p}),
            )
        }
        DiscoverCommand::Review(a) => review(context, a, b),
        DiscoverCommand::Export(a) => export(context, a, b),
    }
}
fn review(context: &Context, a: &ReviewArgs, b: &mut MemoryBudget) -> Result<(), L2dError> {
    let r = load(&a.artifact, b)?;
    hold_output(
        b,
        (a.action.len() as u64).saturating_add(a.analyst.len() as u64),
    )?;
    let mut selection = SelectionSet::load(&a.selection, b)?;
    b.require_whole(selection.held_bytes(), "review selection receipt")?;
    b.hold(b.held_bytes().saturating_add(selection.held_bytes()));
    current(context, &r, b)?;
    let preview = r.preview_review(&selection, a.pattern, a.state.state(), b)?;
    let action = ReviewAction {
        analyst: a.analyst.clone(),
        action: a.action.clone(),
    };
    let review_revision = r.review_revision(&selection, a.pattern, a.state.state(), &action, b)?;
    let affected: Vec<_> = r
        .rows
        .iter()
        .filter(|row| row.pattern == Some(a.pattern))
        .collect();
    let report = serde_json::json!({"review_revision":review_revision,"refinement_revision":r.revision,"selection_revision":r.selection_revision,"pattern":a.pattern,"affected_members":affected,"new_state":a.state.state(),"analyst":a.analyst,"action":a.action,"other_members":"previous decisions preserved"});
    if !a.apply {
        return output(context, b, &report);
    }
    if a.expect_revision.as_deref() != Some(&review_revision) {
        return Err(failed("review preview revision is stale; preview again"));
    }
    match &r.catalogue.source {
        MarkedSource::Input(_) => r.review_saved_input(&mut selection, preview, action, b)?,
        MarkedSource::Store(s) => {
            let store = l2d_store::Store::open(&context.paths()?, &s.name)?;
            r.require_current_store(&store, b)?;
            selection.review_store(preview, action, &store, b)?;
            r.require_current_store(&store, b)?;
        }
    }
    let next = r.with_review(&selection, b)?;
    b.require_whole(next.held_bytes(), "retaining reviewed refinement")?;
    b.hold(b.held_bytes().saturating_add(next.held_bytes()));
    let out = a
        .out
        .as_ref()
        .ok_or_else(|| invalid("--apply needs --out"))?;
    bundle(out, |stage| {
        current(context, &next, b)?;
        selection.save(&stage.join("selection.json"), b)?;
        next.save(&stage.join("refinement.json"), b)?;
        Ok(())
    })?;
    output(
        context,
        b,
        &serde_json::json!({"preview":report,"saved":out,"revision":next.revision,"selection_revision":selection.revision(b)?}),
    )
}
fn export(context: &Context, a: &ExportArgs, b: &mut MemoryBudget) -> Result<(), L2dError> {
    let r = load(&a.artifact, b)?;
    let selection = SelectionSet::load(&a.selection, b)?;
    b.require_whole(selection.held_bytes(), "export selection receipt")?;
    b.hold(b.held_bytes().saturating_add(selection.held_bytes()));
    r.require_selection(&selection, b)?;
    current(context, &r, b)?;
    let label = match a.label {
        ExportLabel::Good => Label::Good,
        ExportLabel::Bad => Label::Bad,
    };
    let marks = selection.export_mark_set(&a.name, label, b)?;
    let selected = marks.selected_count();
    let total = marks.count();
    // Precharge both mark membership/exclusions and serialization, retaining the receipt.
    b.require_whole(
        total.saturating_mul(512).saturating_add(4 * 1024 * 1024),
        "export MarkSet and receipt",
    )?;
    b.hold(b.held_bytes().saturating_add(total.saturating_mul(512)));
    bundle(&a.out, |stage| {
        selection.save(&stage.join("selection.json"), b)?;
        r.save(&stage.join("refinement.json"), b)?;
        marks.save(&ProjectPaths::new(stage), b)?;
        current(context, &r, b)?;
        Ok(())
    })?;
    output(
        context,
        b,
        &serde_json::json!({"saved":a.out,"mark_set":format!("marks/{}.json",a.name),"review_receipt":"selection.json","format":3,"selected":selected,"excluded_unresolved_or_conflict":total-selected,"label":label}),
    )
}

fn explore(context: &Context, a: &ExploreArgs, b: &mut MemoryBudget) -> Result<(), L2dError> {
    use l2d_decide::exploration::{
        self, CharacterTrigram, ExplorationEncoder, ExploreSettings, Scope,
    };
    if a.out.exists() {
        return Err(invalid(
            "output already exists; choose a new exploration revision destination",
        ));
    }
    let seed = load(&a.artifact, b)?;
    current(context, &seed, b)?;
    let expansion = if let Some(path) = &a.expand_artifact {
        let r = load(
            &ArtifactArgs {
                artifact: path.clone(),
            },
            b,
        )?;
        current(context, &r, b)?;
        Some(r)
    } else {
        None
    };
    let settings = ExploreSettings {
        exemplar_row: a.exemplar_row,
        scope: match a.scope {
            ExploreScope::WithinPattern => Scope::WithinPattern,
            ExploreScope::AcrossPatterns => Scope::AcrossPatterns,
            ExploreScope::AllPatterns => Scope::AllPatterns,
        },
        top_k: a.top_k,
        radius: a.radius,
        expected_profile: a.expect_profile.clone(),
    };
    let prior = b.held_bytes();
    let mut encoder: Box<dyn ExplorationEncoder> = match a.method {
        ExploreMethod::CharacterTrigram => {
            if a.profile_dir.is_some() {
                return Err(invalid("--profile-dir applies only to --method npu"));
            }
            Box::new(CharacterTrigram)
        }
        ExploreMethod::Npu => {
            use l2d_encoder::hierarchy::{HierarchicalSession, PROFILE, SESSION_BYTES};
            if a.expect_profile.as_deref().is_some_and(|p| p != PROFILE) {
                return Err(invalid(format!(
                    "incompatible exploration profiles: requested {}; actual {PROFILE}",
                    a.expect_profile.as_deref().unwrap()
                )));
            }
            let directory=a.profile_dir.as_deref().ok_or_else(||invalid("--method npu requires --profile-dir; use character-trigram explicitly without an NPU"))?;
            let paths = context.paths()?;
            let runtime =
                paths.runtime_files_dir(l2d_core::paths::runtime_dir_from_environment().as_deref());
            let cache = std::env::var_os("L2D_NPU_WORK_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("l2d-hierarchy"))
                .join("cache");
            let session = HierarchicalSession::open(directory, &runtime, &cache, b)?;
            b.hold(prior.saturating_add(SESSION_BYTES));
            Box::new(session)
        }
    };
    let report = exploration::explore(&seed, expansion.as_ref(), &settings, encoder.as_mut(), b);
    drop(encoder);
    if matches!(a.method, ExploreMethod::Npu) {
        b.hold(
            b.held_bytes()
                .saturating_sub(l2d_encoder::hierarchy::SESSION_BYTES),
        );
    }
    let report = report?;
    // Recheck after retrieval, before atomically publishing source-bound results.
    current(context, &seed, b)?;
    if let Some(r) = &expansion {
        current(context, r, b)?;
    }
    bundle(&a.out, |stage| {
        let mut file = fs::File::create(stage.join("report.json")).map_err(failed)?;
        serde_json::to_writer_pretty(&mut file, &report).map_err(failed)?;
        file.sync_all().map_err(failed)?;
        Ok(())
    })?;
    output(
        context,
        b,
        &serde_json::json!({"saved":a.out,"exploration":report,"qualification":"retrieval only; no classification or deployment qualification","native_memory":"NPU provider admission estimate, not a measured allocation bound; external NPU utilization unknown"}),
    )
}
