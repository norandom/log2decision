//! MarkSet triage with an explicitly chosen method, reach settings or a versioned policy.
use super::analyst::{Method, read_json};
use super::analyst_source::{HistoryArgs, Source, hold, invalid, load_marks};
use super::common::{self, Context};
use super::encoder::ProfileArgs;
use super::triage::{PrimaryChoice, TriageArgs};
use clap::Args;
use l2d_core::L2dError;
use l2d_decide::close_enough::{Calibration, ReachSettings, close_enough};
use l2d_decide::policy::{PolicyInput, PolicyReport, PolicySource, TriagePolicy, triage_policy};
use serde_json::json;
use std::path::PathBuf;

#[derive(Debug, Clone, Args)]
pub struct MarkTriageArgs {
    /// Live source MarkSet bound to the saved scorer
    #[arg(long)]
    pub mark_set: Option<String>,
    /// Explicit named representation for MarkSet evidence
    #[arg(long, value_enum, requires = "mark_set")]
    pub method: Option<Method>,
    #[command(flatten)]
    pub history: HistoryArgs,
    #[command(flatten)]
    pub archive: common::ArchiveArgs,
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Versioned JSON policy with development-selected thresholds and reservations
    #[arg(long,requires="mark_set",conflicts_with_all=["radius","top_k","min_support"])]
    pub policy: Option<PathBuf>,
    /// Write the revision-bound policy report at this new path
    #[arg(long, requires = "policy")]
    pub out: Option<PathBuf>,
    /// Compare a persisted report with a fresh run
    #[arg(long, requires = "policy")]
    pub replay: Option<PathBuf>,
    #[arg(long, requires = "mark_set")]
    pub radius: Option<f64>,
    #[arg(long, requires = "mark_set")]
    pub top_k: Option<usize>,
    #[arg(long, requires = "mark_set")]
    pub min_support: Option<usize>,
}
pub fn run(ctx: &Context, args: &TriageArgs, name: &str) -> Result<(), L2dError> {
    let options = &args.mark;
    let method = options
        .method
        .ok_or_else(|| invalid("MarkSet triage requires --method npu or character-trigram"))?;
    if args.primary != PrimaryChoice::Baseline || args.limit.is_some() {
        return Err(invalid(
            "MarkSet triage preserves all decisions; --primary neighbour and --limit do not apply",
        ));
    }
    if args.error_rate.is_some() && args.calibration.is_none() {
        return Err(invalid("--error-rate requires --calibration"));
    }
    let paths = ctx.paths()?;
    let mut b = ctx.budget()?;
    let _guard = l2d_core::resource::ResourceGuard::enter(
        "CLI MarkSet triage",
        l2d_core::resource::busy_choice(),
    )?;
    let model = common::required("--baseline", &args.baseline, "the saved mark-set scorer")?;
    let dir = common::find_model(
        &paths.baselines_dir(),
        "--baseline",
        "mark-set scorer",
        model,
    )?;
    let scorer = common::load_scorer(&paths, &dir, &mut b)?;
    let origin = scorer
        .mark_set_origin()
        .ok_or_else(|| invalid("--baseline must identify a scorer fitted from marks"))?;
    if name != origin.marks().name() {
        return Err(invalid(
            "--mark-set differs from the fitted scorer's MarkSet",
        ));
    }
    let marks = load_marks(&paths, name, &mut b)?;
    let other = origin
        .other()
        .map(|m| load_marks(&paths, m.name(), &mut b))
        .transpose()?;
    if args
        .store
        .as_deref()
        .is_some_and(|s| marks.store().is_none_or(|m| m.name != s))
    {
        return Err(invalid("--store differs from the MarkSet source"));
    }
    let source = Source::open(ctx, &options.history, &marks, &mut b)?;
    let history = source.history(&marks, other.as_ref(), &mut b)?;
    let query =
        super::analyst_source::query_input(ctx, &paths, &args.input, &options.archive, &mut b)?;
    let room = (query.logs.frame().estimated_size() as u64)
        .saturating_mul(8)
        .saturating_add((query.logs.len() as u64).saturating_mul(1024))
        .saturating_add(262144);
    hold(
        &mut b,
        room,
        "policy query identities, originals and endpoint groups",
    )?;
    let calibration = args
        .calibration
        .as_ref()
        .map(|name| {
            let dir = common::find_model(
                &paths.calibrations_dir(),
                "--calibration",
                "calibration",
                name,
            )?;
            common::load_calibration(&paths, &dir, &b)
        })
        .transpose()?;
    if let Some(c) = &calibration {
        hold(&mut b, c.held_bytes(), "retained conformal calibration")?;
    }
    let cal = || {
        calibration.as_ref().map(|calibrator| Calibration {
            calibrator,
            error_rate: args.error_rate.unwrap_or(0.05),
        })
    };
    let inputs: Vec<_> = query
        .logs
        .records()
        .map(|r| PolicyInput {
            id: l2d_store::log_id(&query.summary.source_path.to_string_lossy(), &r.text),
            text: r.text,
            group: r.group,
        })
        .collect();
    let mut encoder = method.open(ctx, &options.profile, &mut b)?;
    let result = if let Some(policy_file) = &options.policy {
        let policy: TriagePolicy = read_json(policy_file, &mut b)?;
        if args
            .budget
            .is_some_and(|n| n != policy.review_budget as u64)
        {
            return Err(invalid(
                "--budget differs from the versioned policy's review budget",
            ));
        }
        let source_kind = source
            .store()
            .map_or(PolicySource::InputSnapshot, PolicySource::Store);
        let mut report = triage_policy(
            &inputs,
            source_kind,
            &history,
            &scorer,
            cal(),
            &policy,
            encoder.as_mut(),
            &mut b,
        )?;
        report.query_source = Some(l2d_core::MarkedInput {
            path: query.summary.source_path.to_string_lossy().into_owned(),
            content_sha256: query.summary.content_sha256.clone(),
            archive_entries: (query.adapter == "zip").then(|| options.archive.entries.clone()),
            interpretation: Some(if query.adapter == "zip" {
                l2d_core::marks::InputInterpretation::archive(
                    &options.archive.settings(common::AtOnceDefault::One),
                )
            } else {
                l2d_core::marks::InputInterpretation::file(&query.logs, &b)?
            }),
        });
        source.current(&marks, &mut b)?;
        if let Some(store) = source.store() {
            history.require_store(store, &mut b)?;
        }
        if let Some(replay) = &options.replay {
            let prior = b.held_bytes();
            let earlier = PolicyReport::load(
                replay,
                b.left_bytes(),
                &policy,
                scorer.revision(),
                encoder.profile(),
                history.metadata_revision(),
                &mut b,
            )?;
            report.require_same_decisions(&earlier, &b)?;
            drop(earlier);
            b.hold(prior);
        }
        let bytes = super::analyst_source::report_value(&report, &mut b)?;
        if let Some(out) = &options.out {
            report.save(out, &b)?;
        }
        let mut output = json!({"input":query.summary,"report_path":options.out,"replay_verified":options.replay.is_some()});
        output["policy_report"] = bytes;
        output
    } else {
        if args.budget.is_some() {
            return Err(invalid("review allocations require --policy"));
        }
        let reach = ReachSettings {
            radius: options
                .radius
                .ok_or_else(|| invalid("--radius required without --policy"))?,
            top_k: options
                .top_k
                .ok_or_else(|| invalid("--top-k required without --policy"))?,
            minimum_labelled_support: options
                .min_support
                .ok_or_else(|| invalid("--min-support required without --policy"))?,
            expected_profile: encoder.profile().into(),
        };
        let mut decisions = Vec::new();
        for input in &inputs {
            let result = close_enough(
                &history,
                &input.text,
                input.group.as_deref(),
                &scorer,
                cal(),
                &reach,
                encoder.as_mut(),
                &mut b,
            )?;
            let mut row = json!({"id":input.id});
            row["evidence"] = super::analyst_source::report_value(&result, &mut b)?;
            decisions.push(row);
        }
        source.current(&marks, &mut b)?;
        if let Some(store) = source.store() {
            history.require_store(store, &mut b)?;
        }
        let mut output = json!({"input":query.summary});
        output["decisions"] = serde_json::Value::Array(decisions);
        output
    };
    let mut result = result;
    super::analyst::add_encoding_run(
        &mut result,
        encoder.as_ref(),
        &marks,
        &history,
        Some(&query.summary),
    );
    if query.adapter == "zip" {
        result["archive_entries"] = json!(options.archive.entries);
    }
    crate::output::report(ctx, &b, &result)
}
