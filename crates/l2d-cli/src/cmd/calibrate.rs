//! `l2d calibrate`: calibrate a fitted scorer on held-out labelled logs (requirement
//! 11.1 to 11.5).
//!
//! The calibration logs must be logs the scorer was not fitted on. Two checks hold
//! them to that (`l2d_decide::conformal::held_out_logs`): an input whose content hash is
//! among the scorer's fitting inputs is refused as a whole, and every log whose text
//! the scorer was fitted on is left out and counted. A scorer saved before it kept the
//! hashes of its fitted texts gets the first check only, and the output says so.
//!
//! The command then calibrates on what remains, checks the error rate the analyst
//! intends to use (`--error-rate`), and saves the calibration below the project's model
//! directory. Nothing is saved when a check refuses.

use std::io::{self, Write};

use clap::Args;
use l2d_core::polars::prelude::Column;
use l2d_core::{L2dError, LogsFrame};
use l2d_decide::conformal::{
    DEFAULT_ERROR_RATE, DEFAULT_MIN_PER_GROUP, EXCHANGEABILITY_NOTE, HELD_OUT_ROW_BYTES, HeldOut,
    HeldOutCheck, estimated_bytes, held_out_logs,
};
use l2d_decide::{
    CalibrationError, CalibrationSettings, ConformalCalibrator, Scorer, ScorerKind, Stratum,
};
use serde::Serialize;

use super::common::{
    Context, DEFAULT_TEXT_LIMIT, InputArgs, InputReport, MemoryReport, Operation, ScorerReport,
    check_name, error_rate, find_model, load_scorer, model_name, read_input, required,
    store_not_available,
};
use crate::output::{JsonOut, output_failed, shown_text};

/// The options of `l2d calibrate`.
#[derive(Debug, Clone, Args)]
pub struct CalibrateArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// The saved baseline scorer to calibrate: its name, or <name>-<revision>
    #[arg(long, value_name = "NAME")]
    pub baseline: Option<String>,

    /// Not available yet: calibrating the neighbour scorer of a store needs an encoder
    #[arg(long, value_name = "NAME")]
    pub store: Option<String>,

    /// The name the calibration is saved under: models/calibrations/<NAME>-<revision>/
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,

    /// Calibration logs of a label a group needs for a list of its own
    #[arg(
        long,
        value_name = "N",
        default_value_t = DEFAULT_MIN_PER_GROUP,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub min_per_group: u32,

    /// The error rate the calibration must support: above 0, below 1
    #[arg(long, value_name = "RATE", default_value_t = DEFAULT_ERROR_RATE, value_parser = error_rate)]
    pub error_rate: f32,
}

/// One list of a calibration: an element of `calibration.strata` in the JSON output.
#[derive(Debug, Clone, Serialize)]
struct StratumJson {
    label: &'static str,
    /// `pooled`, or `group` for the list of one group.
    stratum: &'static str,
    /// The group of a group's list; `null` for the pooled list.
    group: Option<String>,
    logs: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CalibratedScorer {
    kind: &'static str,
    revision: String,
}

/// The `calibration` object of the JSON output.
#[derive(Debug, Clone, Serialize)]
struct CalibrationJson {
    name: String,
    revision: String,
    directory: String,
    scorer: CalibratedScorer,
    min_per_group: u32,
    good: u64,
    bad: u64,
    smallest_supported_error_rate: f32,
    /// The lists in their fixed order: good then bad, the pooled list first.
    strata: Vec<StratumJson>,
    /// What the calibration keeps in memory while it is in use, in bytes.
    held_bytes: u64,
}

/// The `held_out` object of the JSON output.
#[derive(Debug, Clone, Serialize)]
struct HeldOutJson {
    #[serde(flatten)]
    counts: HeldOut,
    /// What has to be said when only an identical file is detected; `null` otherwise.
    note: Option<&'static str>,
}

/// A refusal after logs were left out names how many, so that "too few" is understood.
fn with_dropped(error: CalibrationError, held_out: &HeldOut) -> L2dError {
    match error {
        CalibrationError::TooFewLogs { .. } if held_out.dropped > 0 => L2dError::Refused {
            reason: format!(
                "{} of {} calibration logs were left out because the scorer was fitted on \
                 their text; of the {} that remain, {error}",
                held_out.dropped, held_out.logs, held_out.kept
            ),
        },
        other => other.into(),
    }
}

/// Runs `l2d calibrate`.
pub fn run(context: &Context, args: &CalibrateArgs) -> Result<(), L2dError> {
    if args.store.is_some() {
        return Err(store_not_available("calibrate"));
    }
    let name = required(
        "--name",
        &args.name,
        "the name the calibration is saved under",
    )?;
    check_name("--name", name)?;
    let given = required(
        "--baseline",
        &args.baseline,
        "the saved scorer to calibrate",
    )?;
    let paths = context.paths()?;
    let mut budget = context.budget()?;
    let scorer_directory = find_model(
        &paths.baselines_dir(),
        "--baseline",
        "baseline scorer",
        given,
    )?;
    let input = read_input(&paths, &args.input, &mut budget)?;
    let input_report = InputReport::of(&input.logs, &input.summary, input.adapter, &budget)?;
    let mut scorer = load_scorer(&paths, &scorer_directory, &mut budget)?;
    let scorer_report = ScorerReport::of(&scorer, &scorer_directory);

    let (logs, held_out) = held_out_logs(
        &scorer,
        &input.logs,
        std::slice::from_ref(&input.summary),
        &budget,
    )?;
    let plan = input.plan;
    drop(input.logs);

    let settings = CalibrationSettings {
        min_per_group: args.min_per_group,
    };
    // The estimate `fit_scorer` asks the budget for, for the report.
    let cost = Scorer::cost(&scorer, &logs);
    let group_bytes: u64 = logs
        .frame()
        .column(LogsFrame::GROUP)
        .and_then(Column::str)
        .map(|groups| {
            groups
                .iter()
                .flatten()
                .map(|group| group.len() as u64)
                .sum()
        })
        .unwrap_or(0);
    let rows = logs.len() as u64;
    let calibrate_bytes = estimated_bytes(rows, group_bytes)
        .saturating_add(cost.fixed_bytes)
        .saturating_add(rows.saturating_mul(cost.row_bytes + cost.evidence_row_bytes));

    let calibrator = ConformalCalibrator::fit_scorer(&mut scorer, &logs, &settings, &budget)
        .map_err(|error| with_dropped(error, &held_out))?;
    calibrator
        .require_error_rate(args.error_rate)
        .map_err(|error| with_dropped(error, &held_out))?;
    let directory = calibrator.save(&paths, name)?;

    let calibration = CalibrationJson {
        name: model_name(&directory),
        revision: calibrator.revision().to_string(),
        directory: directory.to_string_lossy().into_owned(),
        scorer: CalibratedScorer {
            kind: ScorerKind::Baseline.as_str(),
            revision: calibrator.scorer_revision().to_string(),
        },
        min_per_group: calibrator.settings().min_per_group,
        good: calibrator.good(),
        bad: calibrator.bad(),
        smallest_supported_error_rate: calibrator.smallest_supported(),
        strata: calibrator
            .strata()
            .into_iter()
            .map(|stratum| StratumJson {
                label: stratum.label.as_str(),
                stratum: match stratum.stratum {
                    Stratum::Pooled => "pooled",
                    Stratum::Group(_) => "group",
                },
                group: match stratum.stratum {
                    Stratum::Pooled => None,
                    Stratum::Group(group) => Some(group),
                },
                logs: stratum.logs,
            })
            .collect(),
        held_bytes: calibrator.held_bytes(),
    };
    let mut operations = vec![Operation::batches("read", input.summary.records, plan)];
    if held_out.check != HeldOutCheck::WholeFile {
        operations.push(Operation::whole(
            "held_out_check",
            held_out.logs,
            held_out.logs.saturating_mul(HELD_OUT_ROW_BYTES),
        ));
    }
    operations.push(Operation::whole(
        "calibrate",
        held_out.kept,
        calibrate_bytes,
    ));
    let memory = context.memory_report(&budget, operations);
    let held_out = HeldOutJson {
        counts: held_out,
        note: held_out.note(),
    };

    let out = io::stdout().lock();
    let written = if context.json {
        json(
            out,
            &calibration,
            &held_out,
            args.error_rate,
            &scorer_report,
            &input_report,
            &memory,
        )
    } else {
        text(
            out,
            &calibration,
            &held_out,
            args.error_rate,
            &scorer_report,
            &input_report,
            &memory,
        )
    };
    written.map_err(output_failed)
}

fn json(
    out: impl Write,
    calibration: &CalibrationJson,
    held_out: &HeldOutJson,
    error_rate: f32,
    scorer: &ScorerReport,
    input: &InputReport<'_>,
    memory: &MemoryReport,
) -> io::Result<()> {
    let mut json = JsonOut::begin(io::BufWriter::new(out))?;
    json.field("command", "calibrate")?;
    json.field("calibration", calibration)?;
    json.field("held_out", held_out)?;
    json.field("error_rate", &error_rate)?;
    json.field("note", EXCHANGEABILITY_NOTE)?;
    json.field("scorer", scorer)?;
    json.field("input", input)?;
    json.field("memory", memory)?;
    json.end()
}

fn text(
    out: impl Write,
    calibration: &CalibrationJson,
    held_out: &HeldOutJson,
    error_rate: f32,
    scorer: &ScorerReport,
    input: &InputReport<'_>,
    memory: &MemoryReport,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(out);
    writeln!(out, "calibration: {}", calibration.name)?;
    writeln!(out, "  revision: {}", calibration.revision)?;
    writeln!(out, "  saved to: {}", calibration.directory)?;
    writeln!(
        out,
        "  for the {} scorer {}, revision {}",
        scorer.kind, scorer.name, scorer.revision
    )?;
    let counts = &held_out.counts;
    match (counts.check, held_out.note) {
        (HeldOutCheck::WholeFile, note) => writeln!(
            out,
            "  held-out check by whole file only: {}",
            note.unwrap_or_default()
        )?,
        (_, note) => {
            writeln!(
                out,
                "  held-out check per log: {} of {} calibration logs left out because the \
                 scorer was fitted on their text",
                counts.dropped, counts.logs
            )?;
            if let Some(note) = note {
                writeln!(out, "  the check per log is partial: {note}")?;
            }
        }
    }
    writeln!(
        out,
        "  calibration logs: {} good, {} bad ({} of the {} that remain are unlabelled and \
         play no part)",
        calibration.good,
        calibration.bad,
        counts
            .kept
            .saturating_sub(calibration.good + calibration.bad),
        counts.kept
    )?;
    writeln!(
        out,
        "  smallest error rate supported: {}",
        calibration.smallest_supported_error_rate
    )?;
    writeln!(out, "  error rate {error_rate}: supported")?;
    writeln!(
        out,
        "  lists (a group has its own from {} logs of a label):",
        calibration.min_per_group
    )?;
    for stratum in &calibration.strata {
        let name = match &stratum.group {
            None => "pooled".to_string(),
            Some(group) => format!("group {}", shown_text(group, 60)),
        };
        writeln!(
            out,
            "    {:<4}  {:>8} logs  {name}",
            stratum.label, stratum.logs
        )?;
    }
    writeln!(out, "  {EXCHANGEABILITY_NOTE}")?;
    input.write_text(&mut out, DEFAULT_TEXT_LIMIT)?;
    memory.write_text(&mut out)?;
    out.flush()
}
