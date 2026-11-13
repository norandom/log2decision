//! `l2d baseline`: fit or apply the baseline scorer (requirements 2.1 to 2.6).
//!
//! `fit` reads labelled logs, fits the scorer on all of them at once and saves it below
//! the project's model directory. `score` loads a saved scorer and prints a bad/good
//! ratio for every distinct text of an input, in input order; the texts are scored in
//! batches sized by the memory budget and printed as they are scored, so the ratios of
//! an input are never in memory as a whole.

use std::io::{self, Write};

use clap::{Args, Subcommand, ValueEnum};
use l2d_baseline::scorer::{SCORER_KIND, SCORES_NO_FEATURES, SCORES_RATIO};
use l2d_baseline::{BaselineFitReport, BaselineScorer, BaselineSettings, RoutingMode, Termination};
use l2d_core::polars::prelude::{Column, PolarsError};
use l2d_core::{L2dError, LogsFrame};
use l2d_decide::Scorer;
use serde::Serialize;

use super::common::{
    Context, DEFAULT_TEXT_LIMIT, InputArgs, InputReport, MemoryReport, Operation, ScorerReport,
    TEXT_CHARS, check_name, find_model, load_scorer, longest_text, output_bytes, read_input,
    required,
};
use crate::output::{JsonOut, output_failed, shown_text};

/// The operations below `l2d baseline`.
#[derive(Debug, Clone, Subcommand)]
pub enum BaselineCommand {
    /// Fit the baseline scorer on labelled logs and save it
    Fit(FitArgs),
    /// Score logs with a fitted baseline
    Score(ScoreArgs),
}

/// What is done with the routing part of a log (method and URL path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RoutingChoice {
    /// The whole text is hashed
    Keep,
    /// The routing part is left out; only the payload is hashed
    Strip,
    /// Routing part and payload are hashed into separate feature spaces
    Isolate,
}

/// The options of `l2d baseline fit`.
#[derive(Debug, Clone, Args)]
pub struct FitArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// The name the scorer is saved under: models/baselines/<NAME>-<revision>/
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,

    /// What is done with the method and URL path of a log [default: keep]
    #[arg(long, value_enum, value_name = "MODE")]
    pub routing: Option<RoutingChoice>,

    /// Hash buckets per feature space [default: 65536]
    #[arg(long, value_name = "N")]
    pub buckets: Option<u32>,

    /// Longest n-gram in bytes [default: 3]
    #[arg(long, value_name = "N")]
    pub ngram_max: Option<u8>,

    /// Strength of the L2 penalty [default: 1]
    #[arg(long, value_name = "X", allow_negative_numbers = true)]
    pub l2: Option<f64>,

    /// Most solver iterations [default: 1000]
    #[arg(long, value_name = "N")]
    pub max_iter: Option<u32>,

    /// The fit has converged when the gradient norm is below this [default: 0.001]
    #[arg(long, value_name = "X", allow_negative_numbers = true)]
    pub tolerance: Option<f64>,
}

/// The options of `l2d baseline score`.
#[derive(Debug, Clone, Args)]
pub struct ScoreArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// The saved scorer: its name, or its directory name <name>-<revision>
    #[arg(long, value_name = "NAME")]
    pub baseline: Option<String>,

    /// Most logs printed [default: 20 in the text output, all in JSON]
    #[arg(long, value_name = "N")]
    pub limit: Option<u64>,
}

/// Runs one `l2d baseline` operation.
pub fn run(context: &Context, command: &BaselineCommand) -> Result<(), L2dError> {
    match command {
        BaselineCommand::Fit(args) => fit(context, args),
        BaselineCommand::Score(args) => score(context, args),
    }
}

fn settings(args: &FitArgs) -> BaselineSettings {
    let defaults = BaselineSettings::default();
    BaselineSettings {
        buckets: args.buckets.unwrap_or(defaults.buckets),
        ngram_max: args.ngram_max.unwrap_or(defaults.ngram_max),
        routing: match args.routing {
            Some(RoutingChoice::Keep) => RoutingMode::Keep,
            Some(RoutingChoice::Strip) => RoutingMode::Strip,
            Some(RoutingChoice::Isolate) => RoutingMode::Isolate,
            None => defaults.routing,
        },
        l2: args.l2.unwrap_or(defaults.l2),
        max_iter: args.max_iter.unwrap_or(defaults.max_iter),
        tolerance: args.tolerance.unwrap_or(defaults.tolerance),
    }
}

/// How a fit went: the `fit` object of the JSON output.
#[derive(Debug, Clone, Serialize)]
struct FitJson {
    termination: &'static str,
    iterations: u64,
    objective: f64,
    gradient_norm: f64,
    /// Whether the gradient criterion of the settings was met.
    gradient_converged: bool,
    rows: u64,
    good: u64,
    bad: u64,
    unknown_ignored: u64,
    no_feature_rows: u64,
    stored_entries: u64,
    estimated_bytes: u64,
}

impl FitJson {
    fn of(report: &BaselineFitReport, gradient_converged: bool) -> Self {
        Self {
            termination: match report.termination {
                Termination::Converged => "converged",
                Termination::IterationCap => "iteration_cap",
                Termination::LineSearchStopped => "line_search_stopped",
            },
            iterations: report.iterations,
            objective: report.objective,
            gradient_norm: report.gradient_norm,
            gradient_converged,
            rows: report.rows,
            good: report.good,
            bad: report.bad,
            unknown_ignored: report.unknown_ignored,
            no_feature_rows: report.no_feature_rows,
            stored_entries: report.stored_entries,
            estimated_bytes: report.estimated_bytes,
        }
    }
}

fn fit(context: &Context, args: &FitArgs) -> Result<(), L2dError> {
    let name = required("--name", &args.name, "the name the scorer is saved under")?;
    check_name("--name", name)?;
    let paths = context.paths()?;
    let mut budget = context.budget()?;
    let input = read_input(&paths, &args.input, &mut budget)?;
    let input_report = InputReport::of(&input.logs, &input.summary, input.adapter, &budget)?;

    let scorer = BaselineScorer::fit_with(
        &input.logs,
        std::slice::from_ref(&input.summary),
        &settings(args),
        &budget,
    )?;
    let directory = scorer.save(&paths, name)?;
    // What is reported is the scorer as it is saved, not as it was fitted: a save
    // into a directory of the same model merges the fitting inputs and the hashes of
    // the fitted texts, and the merged set can be incomplete.
    let fit_report = *scorer.fit_report();
    let gradient_converged = scorer.gradient_converged();
    drop(scorer);
    let scorer = load_scorer(&paths, &directory, &mut budget)?;

    let fitted = FitJson::of(&fit_report, gradient_converged);
    let report = ScorerReport::of(&scorer, &directory);
    let memory = context.memory_report(
        &budget,
        vec![
            Operation::batches("read", input.summary.records, input.plan),
            Operation::whole("fit", fitted.rows, fitted.estimated_bytes),
        ],
    );
    let out = io::stdout().lock();
    let written = if context.json {
        fit_json(out, &report, &fitted, &input_report, &memory)
    } else {
        fit_text(out, &report, &fitted, &input_report, &memory)
    };
    written.map_err(output_failed)
}

fn fit_json(
    out: impl Write,
    scorer: &ScorerReport,
    fitted: &FitJson,
    input: &InputReport<'_>,
    memory: &MemoryReport,
) -> io::Result<()> {
    let mut json = JsonOut::begin(io::BufWriter::new(out))?;
    json.field("command", "baseline fit")?;
    json.field("scorer", scorer)?;
    json.field("fit", fitted)?;
    json.field("input", input)?;
    json.field("memory", memory)?;
    json.end()
}

/// The lines that name a scorer and its settings.
pub(crate) fn scorer_text(out: &mut impl Write, scorer: &ScorerReport) -> io::Result<()> {
    writeln!(out, "baseline scorer: {}", scorer.name)?;
    writeln!(out, "  revision: {}", scorer.revision)?;
    writeln!(out, "  saved to: {}", scorer.directory)?;
    let settings = &scorer.settings;
    writeln!(
        out,
        "  settings: {} buckets, n-grams up to {} bytes, routing {}, L2 penalty {}, at most \
         {} iterations, tolerance {}",
        settings.buckets,
        settings.ngram_max,
        settings.routing,
        settings.l2,
        settings.max_iter,
        settings.tolerance
    )
}

fn fit_text(
    out: impl Write,
    scorer: &ScorerReport,
    fitted: &FitJson,
    input: &InputReport<'_>,
    memory: &MemoryReport,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(out);
    scorer_text(&mut out, scorer)?;
    writeln!(
        out,
        "  fitted on: {} distinct labelled texts ({} good, {} bad; {} unlabelled ignored, {} \
         without features)",
        fitted.rows, fitted.good, fitted.bad, fitted.unknown_ignored, fitted.no_feature_rows
    )?;
    writeln!(
        out,
        "  fit: {} after {} iterations, gradient norm {:.3e} ({})",
        fitted.termination.replace('_', " "),
        fitted.iterations,
        fitted.gradient_norm,
        if fitted.gradient_converged {
            "gradient criterion met"
        } else {
            "gradient criterion not met"
        }
    )?;
    match (
        scorer.fitted_text_hashes,
        scorer.fitted_text_hashes_complete,
    ) {
        (Some(count), Some(false)) => writeln!(
            out,
            "  text hashes saved: {count} (incomplete: this directory also names fitting \
             inputs that were saved without hashes, so calibration can leave out the logs \
             with these texts only, and will say that its check per log is partial)"
        )?,
        (Some(count), _) => writeln!(
            out,
            "  text hashes saved: {count} (calibration leaves out logs with these texts)"
        )?,
        (None, _) => writeln!(out, "  text hashes saved: none")?,
    }
    input.write_text(&mut out, DEFAULT_TEXT_LIMIT)?;
    memory.write_text(&mut out)?;
    out.flush()
}

/// One scored log: an element of `scores` in the JSON output.
#[derive(Debug, Serialize)]
struct ScoreRow<'a> {
    /// The row of the log in the input, from 0, in order of first occurrence.
    row: u64,
    text: &'a str,
    label: &'a str,
    group: Option<&'a str>,
    first_position: u64,
    occurrences: u32,
    scorer: &'static str,
    ratio: f32,
    abstained: bool,
    /// The text has no features: its ratio is the scorer's constant for such texts.
    no_features: bool,
}

/// Where the rows of `l2d baseline score` go.
enum ScoreOut<W: Write> {
    Json(JsonOut<W>),
    Text(W),
}

fn polars(error: PolarsError) -> L2dError {
    L2dError::Runtime {
        detail: format!("reading the scores failed: {error}"),
    }
}

fn score(context: &Context, args: &ScoreArgs) -> Result<(), L2dError> {
    let given = required(
        "--baseline",
        &args.baseline,
        "the saved scorer to score with",
    )?;
    let paths = context.paths()?;
    let mut budget = context.budget()?;
    let directory = find_model(
        &paths.baselines_dir(),
        "--baseline",
        "baseline scorer",
        given,
    )?;
    let input = read_input(&paths, &args.input, &mut budget)?;
    let input_report = InputReport::of(&input.logs, &input.summary, input.adapter, &budget)?;
    let scorer = load_scorer(&paths, &directory, &mut budget)?;
    let report = ScorerReport::of(&scorer, &directory);

    let logs = &input.logs;
    let rows = logs.len() as u64;
    let cost = Scorer::cost(&scorer, logs);
    // Hashing the longest text and printing take memory for the whole run: asked for
    // first, then held, so that the batches are planned in what is left.
    let fixed = cost
        .fixed_bytes
        .saturating_add(output_bytes(longest_text(logs)));
    budget.require_whole(fixed, "scoring and printing the ratios")?;
    budget.hold(budget.held_bytes().saturating_add(fixed));
    let mut plan = budget.plan(rows, cost.row_bytes + cost.evidence_row_bytes)?;
    let limit = match (args.limit, context.json) {
        (Some(limit), _) => limit,
        (None, true) => u64::MAX,
        (None, false) => DEFAULT_TEXT_LIMIT,
    };

    let frame = logs.frame();
    let texts = frame
        .column(LogsFrame::TEXT)
        .and_then(Column::str)
        .map_err(polars)?;
    let labels = frame
        .column(LogsFrame::LABEL)
        .and_then(Column::str)
        .map_err(polars)?;
    let groups = frame
        .column(LogsFrame::GROUP)
        .and_then(Column::str)
        .map_err(polars)?;
    let positions = frame
        .column(LogsFrame::FIRST_POSITION)
        .and_then(Column::u64)
        .map_err(polars)?;
    let occurrences = frame
        .column(LogsFrame::OCCURRENCES)
        .and_then(Column::u32)
        .map_err(polars)?;

    let stdout = io::BufWriter::new(io::stdout().lock());
    let mut out = if context.json {
        let mut json = JsonOut::begin(stdout).map_err(output_failed)?;
        (|| {
            json.field("command", "baseline score")?;
            json.field("scorer", &report)?;
            json.field("input", &input_report)?;
            json.begin_list("scores")
        })()
        .map_err(output_failed)?;
        ScoreOut::Json(json)
    } else {
        let mut text = stdout;
        (|| {
            writeln!(
                text,
                "scorer: {} {} (revision {})",
                report.kind, report.name, report.revision
            )?;
            input_report.write_text(&mut text, DEFAULT_TEXT_LIMIT)?;
            writeln!(
                text,
                "ratios of {rows} distinct texts, in input order (0 looks good, 1 looks bad)"
            )?;
            writeln!(text, "   ratio  text")
        })()
        .map_err(output_failed)?;
        ScoreOut::Text(text)
    };

    let mut done: u64 = 0;
    let mut no_features: u64 = 0;
    while done < rows {
        let size = budget.next_batch_rows(&mut plan, done)?.min(rows - done);
        // A batch is a view of the logs frame: its texts are shared, not copied, and
        // the scores frame shares them too.
        let batch = LogsFrame::new(frame.slice(done as i64, size as usize))?;
        let scores = scorer.score(&batch)?;
        let ratios = scores
            .column(SCORES_RATIO)
            .and_then(Column::f32)
            .map_err(polars)?;
        let flags = scores
            .column(SCORES_NO_FEATURES)
            .and_then(Column::bool)
            .map_err(polars)?;
        for index in 0..size as usize {
            let row = done + index as u64;
            let text = texts.get(row as usize).unwrap_or_default();
            let flagged = flags.get(index).unwrap_or(false);
            no_features += u64::from(flagged);
            if row >= limit {
                continue;
            }
            let ratio = ratios.get(index).unwrap_or(f32::NAN);
            let at = row as usize;
            let written = match &mut out {
                ScoreOut::Json(json) => json.element(&ScoreRow {
                    row,
                    text,
                    label: labels.get(at).unwrap_or("unknown"),
                    group: groups.get(at),
                    first_position: positions.get(at).unwrap_or(0),
                    occurrences: occurrences.get(at).unwrap_or(0),
                    scorer: SCORER_KIND,
                    ratio,
                    abstained: false,
                    no_features: flagged,
                }),
                ScoreOut::Text(out) => writeln!(
                    out,
                    "  {ratio:.4}  {}{}",
                    if flagged { "[no features] " } else { "" },
                    shown_text(text, TEXT_CHARS)
                ),
            };
            written.map_err(output_failed)?;
        }
        done += size;
    }

    let omitted = rows.saturating_sub(limit);
    let memory = context.memory_report(
        &budget,
        vec![
            Operation::batches("read", input.summary.records, input.plan.clone()),
            Operation::batches("score", rows, plan),
        ],
    );
    let written = match out {
        ScoreOut::Json(mut json) => (|| {
            json.end_list()?;
            json.field("scored", &rows)?;
            json.field("rows_omitted", &omitted)?;
            json.field("no_features", &no_features)?;
            json.field("memory", &memory)?;
            json.end()
        })(),
        ScoreOut::Text(mut out) => (|| {
            if omitted > 0 {
                writeln!(
                    out,
                    "  ... {omitted} more (`--limit` sets how many are shown)"
                )?;
            }
            if no_features > 0 {
                writeln!(
                    out,
                    "logs with no features: {no_features} (they share one constant ratio)"
                )?;
            }
            memory.write_text(&mut out)?;
            out.flush()
        })(),
    };
    written.map_err(output_failed)
}
