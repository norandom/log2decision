//! Group, refine and fit the analyst's marks. Dataset labels never become reviewed marks.
use super::analyst_source::{HistoryArgs, Source, hold, invalid, load_marks};
use super::common::{Context, ScorerReport, check_name};
use super::encoder::ProfileArgs;
use clap::{Args, Subcommand, ValueEnum};
use l2d_baseline::markscore::{GroupKeyRule, MarkFitSettings, fit_from_marks};
use l2d_baseline::redact::{RedactionRules, RedactionSettings, propose_redactions};
use l2d_core::{L2dError, MemoryBudget, RedactionRule};
use l2d_decide::close_enough::{ReachSettings, group_marks};
use l2d_decide::exploration::{CharacterTrigram, ExplorationEncoder};
use l2d_eval::markscore::{HeldOutSettings, held_out_report_with_group_key};
use serde::de::DeserializeOwned;
use serde_json::json;
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Args)]
pub struct MarksArgs {
    /// Name of a saved MarkSet
    #[arg(long)]
    pub mark_set: String,
    #[command(flatten)]
    pub history: HistoryArgs,
}
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Method {
    Npu,
    CharacterTrigram,
}
impl Method {
    pub(super) fn open(
        self,
        ctx: &Context,
        profile: &ProfileArgs,
        b: &mut MemoryBudget,
    ) -> Result<Box<dyn ExplorationEncoder>, L2dError> {
        match self {
            Self::Npu => Ok(Box::new(profile.open(&ctx.paths()?, b)?)),
            Self::CharacterTrigram if profile.profile_dir.is_none() => {
                Ok(Box::new(CharacterTrigram))
            }
            Self::CharacterTrigram => Err(invalid("--profile-dir applies only to the NPU method")),
        }
    }
}
pub(super) fn add_encoding_run(
    output: &mut serde_json::Value,
    encoder: &dyn ExplorationEncoder,
    marks: &l2d_core::MarkSet,
    history: &l2d_decide::close_enough::MarkHistory,
    query: Option<&l2d_core::InputSummary>,
) {
    if let Some(mut run) = encoder.encoding_run() {
        run.settings["mark_source"] = json!(marks.source());
        run.settings["history_revision"] = json!(history.metadata_revision());
        if let Some(query) = query {
            run.inputs.push(query.clone());
        }
        output["run_record"] = json!(run);
    }
}

#[derive(Debug, Args)]
pub struct GroupArgs {
    #[command(flatten)]
    pub marks: MarksArgs,
    #[arg(long, value_enum)]
    pub method: Method,
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Maximum direct distance to the group's real representative
    #[arg(long)]
    pub radius: f64,
    /// Remove this zero-based group from fitting
    #[arg(long,requires="expect_grouping",conflicts_with_all=["deselect_position","deselect_id"])]
    pub deselect_group: Option<usize>,
    /// Grouping revision from the preview, binding the index to exact original members
    #[arg(long, requires = "deselect_group")]
    pub expect_grouping: Option<String>,
    #[arg(long, conflicts_with = "deselect_id")]
    pub deselect_position: Vec<u64>,
    #[arg(long)]
    pub deselect_id: Vec<String>,
}
pub fn group(ctx: &Context, args: &GroupArgs) -> Result<(), L2dError> {
    let paths = ctx.paths()?;
    let mut b = ctx.budget()?;
    let _guard = l2d_core::resource::ResourceGuard::enter(
        "CLI MarkSet grouping",
        l2d_core::resource::busy_choice(),
    )?;
    let mut marks = load_marks(&paths, &args.marks.mark_set, &mut b)?;
    let source = Source::open(ctx, &args.marks.history, &marks, &mut b)?;
    let history = source.history(&marks, None, &mut b)?;
    let mut encoder = args.method.open(ctx, &args.profile, &mut b)?;
    // Grouping uses only the radius. Neighbour top-k and corroboration are policy settings.
    let settings = ReachSettings {
        radius: args.radius,
        top_k: 1,
        minimum_labelled_support: 1,
        expected_profile: encoder.profile().into(),
    };
    let groups = group_marks(&history, &settings, encoder.as_mut(), &mut b)?;
    let revision = groups.revision(&history, &settings)?;
    if args
        .expect_grouping
        .as_ref()
        .is_some_and(|expected| expected != &revision)
    {
        return Err(L2dError::Refused {
            reason: "grouping revision differs from preview; marks unchanged".into(),
        });
    }
    let changed = args.deselect_group.is_some()
        || !args.deselect_position.is_empty()
        || !args.deselect_id.is_empty();
    if let Some(index) = args.deselect_group {
        groups.deselect(index, &mut marks, source.store(), &mut b)?;
    }
    if !args.deselect_position.is_empty() {
        marks.deselect(&args.deselect_position)?;
    }
    if !args.deselect_id.is_empty() {
        marks.deselect_ids(&args.deselect_id, &b)?;
    }
    source.current(&marks, &mut b)?;
    if let Some(store) = source.store() {
        history.require_store(store, &mut b)?;
    }
    let saved = if changed {
        Some(marks.save(&paths, &b)?)
    } else {
        None
    };
    let mut output = json!({"method":groups.method,"profile":groups.profile,"groups":groups.groups,
        "grouping_revision":revision,"support_basis":"distinct original texts; occurrences reported separately","connectivity":"direct reach from the representative",
        "unsupported":groups.unsupported_details,"source":marks.source(),"metadata_revision":history.metadata_revision(),
        "deselected":marks.deselected(),"deselected_ids":marks.deselected_ids(&b)?,"saved":saved,"selected":marks.selected_count()});
    add_encoding_run(&mut output, encoder.as_ref(), &marks, &history, None);
    crate::output::report(ctx, &b, &output)
}

#[derive(Debug, Args)]
pub struct RedactArgs {
    #[command(flatten)]
    pub marks: MarksArgs,
    /// JSON array containing exactly the ordered rules the analyst confirms; omitted rules are rejected
    #[arg(long)]
    pub rules: Option<PathBuf>,
    #[arg(long, default_value_t = 0.05)]
    pub min_marked_share: f64,
    #[arg(long, default_value_t = 0.01)]
    pub min_unmarked_share: f64,
    #[arg(long, default_value_t = 32768)]
    pub counter_slots: usize,
}
pub(super) fn read_json<T: DeserializeOwned>(
    path: &Path,
    b: &mut MemoryBudget,
) -> Result<T, L2dError> {
    let file = File::open(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            L2dError::Missing {
                item: "configuration artifact".into(),
                expected_at: path.into(),
            }
        } else {
            invalid(format!("{}: {error}", path.display()))
        }
    })?;
    let size = file.metadata().map_err(|e| invalid(e.to_string()))?.len();
    hold(
        b,
        size.saturating_add(1)
            .saturating_mul(32)
            .saturating_add(65536),
        "configuration JSON and parsed values",
    )?;
    let mut bytes = Vec::new();
    file.take(size.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| invalid(e.to_string()))?;
    if bytes.len() as u64 != size {
        return Err(invalid("configuration changed during read"));
    }
    serde_json::from_slice(&bytes).map_err(|e| invalid(format!("{}: {e}", path.display())))
}
pub fn redact(ctx: &Context, args: &RedactArgs) -> Result<(), L2dError> {
    let paths = ctx.paths()?;
    let mut b = ctx.budget()?;
    let _guard = l2d_core::resource::ResourceGuard::enter(
        "CLI redaction preparation",
        l2d_core::resource::busy_choice(),
    )?;
    let mut marks = load_marks(&paths, &args.marks.mark_set, &mut b)?;
    let source = Source::open(ctx, &args.marks.history, &marks, &mut b)?;
    let settings = RedactionSettings {
        min_marked_share: args.min_marked_share,
        min_unmarked_share: args.min_unmarked_share,
        counter_slots: args.counter_slots,
        ..Default::default()
    };
    let proposed = propose_redactions(&marks, source.redaction(), &settings, &mut b)?;
    if proposed
        .archive
        .as_ref()
        .is_some_and(|s| s.damaged().next().is_some())
    {
        return Err(invalid(
            "damaged archive cannot supply complete redaction evidence",
        ));
    }
    hold(
        &mut b,
        proposed.counters_bytes.saturating_mul(2),
        "proposal and JSON output copies",
    )?;
    let rows: Vec<_> = proposed
        .proposals
        .iter()
        .map(|p| {
            json!({"fragment":p.fragment,"kind":format!("{:?}",p.kind),
        "marked":p.marked,"unmarked":p.unmarked})
        })
        .collect();
    let rules = if let Some(path) = &args.rules {
        let selected: Vec<RedactionRule> = read_json(path, &mut b)?;
        let rules = RedactionRules::new(selected, &b)?;
        source.current(&marks, &mut b)?;
        rules.save_with(&mut marks, &paths, &b)?;
        rules
    } else {
        RedactionRules::of_mark_set(&marks, &b)?
    };
    crate::output::report(
        ctx,
        &b,
        &json!({"mark_set":marks.name(),"source":marks.source(),"proposals":rows,
        "marked_logs":proposed.marked_logs,"unmarked_logs":proposed.unmarked_logs,"deselected_logs":proposed.deselected_logs,
        "confirmed_rules":rules.rules(),"rules_sha256":rules.content_sha256(),"saved":args.rules.is_some()}),
    )
}
#[derive(Debug, Subcommand)]
pub enum ScoreCommand {
    /// Fit the character scorer from selected marks and confirmed rules
    Fit(FitArgs),
}
#[derive(Debug, Args)]
pub struct FitArgs {
    #[command(flatten)]
    pub marks: MarksArgs,
    #[arg(long)]
    pub other_mark_set: Option<String>,
    #[arg(long)]
    pub name: String,
    #[arg(long)]
    pub group_pattern: Option<String>,
    #[arg(long, default_value_t = 5)]
    pub folds: usize,
    #[arg(long, default_value_t = 0.05)]
    pub margin: f64,
    /// Publish an immutable qualification evidence bundle with the training report
    #[arg(long)]
    pub out: Option<PathBuf>,
}
pub fn score(ctx: &Context, command: &ScoreCommand) -> Result<(), L2dError> {
    let ScoreCommand::Fit(args) = command;
    check_name("--name", &args.name)?;
    let paths = ctx.paths()?;
    let mut b = ctx.budget()?;
    let _guard = l2d_core::resource::ResourceGuard::enter(
        "CLI MarkSet score fit",
        l2d_core::resource::busy_choice(),
    )?;
    let marks = load_marks(&paths, &args.marks.mark_set, &mut b)?;
    let other = args
        .other_mark_set
        .as_ref()
        .map(|n| load_marks(&paths, n, &mut b))
        .transpose()?;
    let source = Source::open(ctx, &args.marks.history, &marks, &mut b)?;
    let settings = MarkFitSettings {
        group_key: args
            .group_pattern
            .as_ref()
            .map(|p| GroupKeyRule::Pattern { pattern: p.clone() }),
        ..Default::default()
    };
    let store_generation = source
        .store()
        .map(l2d_store::Store::read_generation)
        .transpose()?;
    let mut fit = fit_from_marks(
        &marks,
        other.as_ref(),
        source.redaction(),
        &settings,
        &mut b,
    )?;
    if fit
        .archive
        .as_ref()
        .is_some_and(|s| s.damaged().next().is_some())
    {
        return Err(invalid(
            "damaged archive cannot supply complete training evidence",
        ));
    }
    hold(
        &mut b,
        fit.held_bytes().saturating_add(fit.scorer.held_bytes()),
        "retained fitted scorer, training frame and folds",
    )?;
    let use_stored_groups = source.store().is_some() && args.group_pattern.is_none();
    let group_key = if use_stored_groups {
        let prior = b.held_bytes();
        let history = source.history(&marks, other.as_ref(), &mut b)?;
        fit.logs = history.stored_groups(&fit.logs, &mut b)?;
        drop(history);
        b.hold(prior.saturating_add(fit.held_bytes()));
        "stored endpoint or host key (ambiguous duplicates have no group)".to_owned()
    } else {
        fit.group_key.name()
    };
    let report = held_out_report_with_group_key(
        &fit,
        &HeldOutSettings {
            folds: args.folds,
            margin: args.margin,
        },
        &group_key,
        &mut b,
    )?;
    if source
        .store()
        .map(l2d_store::Store::read_generation)
        .transpose()?
        != store_generation
    {
        return Err(L2dError::Refused {
            reason: "store changed during score fit or grouped assessment".into(),
        });
    }
    let group_report = super::analyst_source::report_value(&report, &mut b)?;
    source.current(&marks, &mut b)?;
    if args.out.as_ref().is_some_and(|out| out.exists()) {
        return Err(L2dError::Refused {
            reason: "qualification output already exists; choose a new destination".into(),
        });
    }
    b.require_whole(
        fit.scorer.held_bytes(),
        "saving fitted weights, mark copies and hashes",
    )?;
    let directory = fit.scorer.save(&paths, &args.name)?;
    let saved = super::common::load_scorer(&paths, &directory, &mut b)?;
    let origin = saved
        .mark_set_origin()
        .ok_or_else(|| invalid("saved score missing MarkSet origin"))?;
    let scorer = ScorerReport::of(&saved, &directory);
    use l2d_baseline::scorer::{
        FITTED_TEXTS_FILE, MANIFEST_FILE, MARK_SET_FILE, OTHER_MARK_SET_FILE, WEIGHTS_FILE,
    };
    let mut files = vec![
        MANIFEST_FILE,
        WEIGHTS_FILE,
        FITTED_TEXTS_FILE,
        MARK_SET_FILE,
    ];
    if origin.other().is_some() {
        files.push(OTHER_MARK_SET_FILE);
    }
    b.require_whole(1024 * 1024, "hashing frozen score artifacts")?;
    let mut hashes = std::collections::BTreeMap::new();
    for file in files {
        hashes.insert(
            file,
            l2d_encoder::runtime::sha256_of(&directory.join(file))?,
        );
    }
    let result = json!({
        "scorer":scorer,
        "source":marks.source(),
        "model_files_sha256":hashes,
        "marks_sha256":origin.marks_sha256(),
        "other_marks_sha256":origin.other_sha256(),
        "confirmed_rules":saved.rules().rules(),
        "rules_sha256":saved.rules().content_sha256(),
        "counts":fit.counts,
        "comparison_class":fit.other_class.to_string(),
        "group_key":group_key,
        "held_out_report":group_report,
        "qualification_status":"exploratory training evidence; independent final qualification incomplete"
    });
    if let Some(out) = &args.out {
        super::discover::bundle(out, |stage| {
            let model = stage.join("model");
            std::fs::create_dir(&model).map_err(|e| L2dError::Runtime {
                detail: e.to_string(),
            })?;
            for (file, expected) in &hashes {
                let frozen = model.join(file);
                std::fs::copy(directory.join(file), &frozen).map_err(|e| L2dError::Runtime {
                    detail: e.to_string(),
                })?;
                if l2d_encoder::runtime::sha256_of(&frozen)? != *expected {
                    return Err(L2dError::Refused {
                        reason: "saved score changed while exporting its artifacts".into(),
                    });
                }
            }
            let mut file = File::create_new(stage.join("score-redaction.json"))
                .map_err(|e| invalid(e.to_string()))?;
            serde_json::to_writer_pretty(&mut file, &result).map_err(|e| invalid(e.to_string()))?;
            file.sync_all().map_err(|e| invalid(e.to_string()))
        })?;
    }
    crate::output::report(ctx, &b, &result)
}
