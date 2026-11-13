//! `l2d triage`: rank logs for review, score down the rest, and list what cannot be
//! decided (requirements 6.4 to 6.6, 11.6, 11.7, 11.9).
//!
//! The logs come from a file (`--input`) and are scored by a saved baseline scorer.
//! With a calibration every log gets the set of labels that cannot be ruled out at the
//! error rate, and the set decides the list. `--store` and the neighbour scorer need an
//! encoder and are refused until there is one.
//!
//! # Output and memory
//!
//! The three lists are printed row by row from the triage frame, through the row
//! numbers triage keeps for each list (`TriageResult::rows`); no list is made into a
//! frame of its own, so the output takes no second copy of the result. While it prints,
//! the command holds in its memory budget what triage held while it ran: the result at
//! 256 bytes per log and the calibration, once. The text output shows at most `--limit`
//! rows per list (20 unless given); the JSON output has every row unless `--limit` is
//! given.
//!
//! # The JSON row
//!
//! Every row of `review`, `scored_down` and `abstained` has the same keys, whatever the
//! input: the logs columns, the triage columns (`null` where a column does not apply),
//! `ratios` (an object with the ratio of every scorer beside the primary one, by kind)
//! and `evidence` (an object with the primary scorer's evidence columns; for the
//! baseline `no_features`). `evidence_columns` at the top names the keys of `evidence`.
//! For an input without logs triage does not ask the scorer for anything, so the three
//! lists are empty and `evidence_columns` is empty: there is no row whose shape could
//! differ.

use std::io::{self, Write};

use clap::{Args, ValueEnum};
use l2d_core::polars::prelude::{
    AnyValue, BooleanChunked, Column, DataFrame, DataType, Float32Chunked, IdxSize, PolarsError,
    StringChunked, UInt32Chunked, UInt64Chunked,
};
use l2d_core::{L2dError, LogsFrame};
use l2d_decide::triage::{
    TRIAGE_ABSTAIN_REASON, TRIAGE_ABSTAINED, TRIAGE_BAD_STRATUM, TRIAGE_GOOD_STRATUM,
    TRIAGE_LABEL_SET, TRIAGE_LIST, TRIAGE_P_BAD, TRIAGE_P_GOOD, TRIAGE_RANK, TRIAGE_RATIO,
    TRIAGE_RATIO_PREFIX, TRIAGE_SCORED_DOWN_BY, TRIAGE_SCORER,
};
use l2d_decide::{
    ListUse, Scorer, ScorerKind, TriageCalibration, TriageList, TriageResult, TriageSettings,
    triage_with,
};
use serde::Serialize;
use serde_json::{Map, Value};

use super::common::{
    Context, DEFAULT_TEXT_LIMIT, InputArgs, InputReport, MemoryReport, Operation, ScorerReport,
    TEXT_CHARS, error_rate, find_model, load_calibration, load_scorer, longest_text, model_name,
    read_input, require_output, required, store_not_available,
};
use crate::output::{JsonOut, output_failed, shown_text};

/// The number of columns of the triage frame before the further scorers' ratios and
/// the evidence: the five of the logs frame and the twelve triage adds.
const FIXED_COLUMNS: usize = 17;
/// The baseline's evidence column: the text has no features.
const NO_FEATURES: &str = l2d_baseline::scorer::SCORES_NO_FEATURES;

/// The scorer that ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PrimaryChoice {
    /// The baseline scorer given with --baseline
    Baseline,
    /// Not available yet: the neighbour scorer of a store needs an encoder
    Neighbour,
}

/// The options of `l2d triage`.
#[derive(Debug, Clone, Args)]
pub struct TriageArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// The saved baseline scorer: its name, or its directory name <name>-<revision>
    #[arg(long, value_name = "NAME")]
    pub baseline: Option<String>,

    /// Not available yet: triage of a store needs an encoder
    #[arg(long, value_name = "NAME")]
    pub store: Option<String>,

    /// A saved calibration of the scorer: its name, or <name>-<revision>
    #[arg(long, value_name = "NAME")]
    pub calibration: Option<String>,

    /// The error rate of the label sets, with --calibration: above 0, below 1
    /// [default: 0.05]
    #[arg(long, value_name = "RATE", value_parser = error_rate)]
    pub error_rate: Option<f32>,

    /// The review budget: how many logs go to review; the rest of the ranked logs is
    /// scored down [default: no budget]
    #[arg(long, value_name = "N")]
    pub budget: Option<u64>,

    /// The scorer that ranks the logs
    #[arg(long, value_enum, value_name = "SCORER", default_value_t = PrimaryChoice::Baseline)]
    pub primary: PrimaryChoice,

    /// Most rows printed per list [default: 20 in the text output, all in JSON]
    #[arg(long, value_name = "N")]
    pub limit: Option<u64>,
}

fn polars(error: PolarsError) -> L2dError {
    L2dError::Runtime {
        detail: format!("reading the triage result failed: {error}"),
    }
}

/// The columns of the triage frame, for reading it row by row.
struct Columns<'a> {
    text: &'a StringChunked,
    label: &'a StringChunked,
    group: &'a StringChunked,
    first_position: &'a UInt64Chunked,
    occurrences: &'a UInt32Chunked,
    scorer: &'a StringChunked,
    ratio: &'a Float32Chunked,
    abstained: &'a BooleanChunked,
    list: &'a StringChunked,
    rank: &'a UInt32Chunked,
    scored_down_by: &'a StringChunked,
    abstain_reason: &'a StringChunked,
    label_set: &'a StringChunked,
    p_good: &'a Float32Chunked,
    p_bad: &'a Float32Chunked,
    good_stratum: &'a StringChunked,
    bad_stratum: &'a StringChunked,
    /// The ratios of the scorers beside the primary one, by kind.
    others: Vec<(&'a str, &'a Float32Chunked)>,
    /// The primary scorer's evidence columns.
    evidence: Vec<&'a Column>,
    /// The baseline's flag, when the frame has it.
    no_features: Option<&'a BooleanChunked>,
}

impl<'a> Columns<'a> {
    fn of(frame: &'a DataFrame) -> Result<Self, L2dError> {
        let text = |name: &str| frame.column(name).and_then(Column::str).map_err(polars);
        let number = |name: &str| frame.column(name).and_then(Column::f32).map_err(polars);
        let mut others = Vec::new();
        let mut evidence = Vec::new();
        for column in frame.get_columns().iter().skip(FIXED_COLUMNS) {
            match column.name().as_str().strip_prefix(TRIAGE_RATIO_PREFIX) {
                Some(kind) if column.dtype() == &DataType::Float32 => {
                    others.push((kind, column.f32().map_err(polars)?));
                }
                _ => evidence.push(column),
            }
        }
        let no_features = evidence
            .iter()
            .find(|column| column.name().as_str() == NO_FEATURES)
            .and_then(|column| column.bool().ok());
        Ok(Self {
            text: text(LogsFrame::TEXT)?,
            label: text(LogsFrame::LABEL)?,
            group: text(LogsFrame::GROUP)?,
            first_position: frame
                .column(LogsFrame::FIRST_POSITION)
                .and_then(Column::u64)
                .map_err(polars)?,
            occurrences: frame
                .column(LogsFrame::OCCURRENCES)
                .and_then(Column::u32)
                .map_err(polars)?,
            scorer: text(TRIAGE_SCORER)?,
            ratio: number(TRIAGE_RATIO)?,
            abstained: frame
                .column(TRIAGE_ABSTAINED)
                .and_then(Column::bool)
                .map_err(polars)?,
            list: text(TRIAGE_LIST)?,
            rank: frame
                .column(TRIAGE_RANK)
                .and_then(Column::u32)
                .map_err(polars)?,
            scored_down_by: text(TRIAGE_SCORED_DOWN_BY)?,
            abstain_reason: text(TRIAGE_ABSTAIN_REASON)?,
            label_set: text(TRIAGE_LABEL_SET)?,
            p_good: number(TRIAGE_P_GOOD)?,
            p_bad: number(TRIAGE_P_BAD)?,
            good_stratum: text(TRIAGE_GOOD_STRATUM)?,
            bad_stratum: text(TRIAGE_BAD_STRATUM)?,
            others,
            evidence,
            no_features,
        })
    }

    fn evidence_names(&self) -> Vec<&'a str> {
        self.evidence
            .iter()
            .map(|column| column.name().as_str())
            .collect()
    }

    /// How many logs the baseline flagged as without features.
    fn no_features_count(&self) -> u64 {
        self.no_features.map_or(0, |flags| {
            flags.iter().flatten().filter(|flag| *flag).count() as u64
        })
    }

    fn row(&self, row: IdxSize) -> Row<'a> {
        let at = row as usize;
        let mut ratios = Map::new();
        for (kind, column) in &self.others {
            ratios.insert((*kind).to_string(), serde_json::json!(column.get(at)));
        }
        let mut evidence = Map::new();
        for column in &self.evidence {
            evidence.insert(column.name().to_string(), json_value(column.get(at).ok()));
        }
        Row {
            row: u64::from(row),
            text: self.text.get(at).unwrap_or_default(),
            label: self.label.get(at).unwrap_or("unknown"),
            group: self.group.get(at),
            first_position: self.first_position.get(at).unwrap_or(0),
            occurrences: self.occurrences.get(at).unwrap_or(0),
            scorer: self.scorer.get(at).unwrap_or_default(),
            ratio: self.ratio.get(at),
            abstained: self.abstained.get(at).unwrap_or(false),
            list: self.list.get(at).unwrap_or_default(),
            rank: self.rank.get(at),
            scored_down_by: self.scored_down_by.get(at),
            abstain_reason: self.abstain_reason.get(at),
            label_set: self.label_set.get(at),
            p_good: self.p_good.get(at),
            p_bad: self.p_bad.get(at),
            good_stratum: self.good_stratum.get(at),
            bad_stratum: self.bad_stratum.get(at),
            ratios,
            evidence,
            no_features: self
                .no_features
                .is_some_and(|flags| flags.get(at) == Some(true)),
        }
    }
}

/// A value of an evidence column as JSON: a flag, a number, a word, or `null`.
fn json_value(value: Option<AnyValue<'_>>) -> Value {
    match value {
        None | Some(AnyValue::Null) => Value::Null,
        Some(AnyValue::Boolean(flag)) => Value::Bool(flag),
        Some(AnyValue::String(word)) => Value::String(word.to_string()),
        Some(AnyValue::StringOwned(word)) => Value::String(word.to_string()),
        Some(AnyValue::UInt8(number)) => number.into(),
        Some(AnyValue::UInt16(number)) => number.into(),
        Some(AnyValue::UInt32(number)) => number.into(),
        Some(AnyValue::UInt64(number)) => number.into(),
        Some(AnyValue::Int8(number)) => number.into(),
        Some(AnyValue::Int16(number)) => number.into(),
        Some(AnyValue::Int32(number)) => number.into(),
        Some(AnyValue::Int64(number)) => number.into(),
        Some(AnyValue::Float32(number)) => serde_json::json!(number),
        Some(AnyValue::Float64(number)) => serde_json::json!(number),
        Some(other) => Value::String(other.to_string()),
    }
}

/// One log of a list: an element of `review`, `scored_down` or `abstained` in the JSON
/// output (module documentation, "The JSON row").
#[derive(Debug, Serialize)]
struct Row<'a> {
    /// The row of the log in the input, from 0, in order of first occurrence.
    row: u64,
    text: &'a str,
    label: &'a str,
    group: Option<&'a str>,
    first_position: u64,
    occurrences: u32,
    /// The scorer that produced the ratio (requirement 6.6).
    scorer: &'a str,
    ratio: Option<f32>,
    abstained: bool,
    list: &'a str,
    rank: Option<u32>,
    scored_down_by: Option<&'a str>,
    abstain_reason: Option<&'a str>,
    label_set: Option<&'a str>,
    p_good: Option<f32>,
    p_bad: Option<f32>,
    good_stratum: Option<&'a str>,
    bad_stratum: Option<&'a str>,
    ratios: Map<String, Value>,
    evidence: Map<String, Value>,
    #[serde(skip)]
    no_features: bool,
}

fn optional(value: Option<f32>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.4}"))
}

impl Row<'_> {
    /// The row on one line of the text output; `why` is the list's own column.
    fn write_text(&self, out: &mut impl Write, why: Option<&str>) -> io::Result<()> {
        let rank = self
            .rank
            .map_or_else(|| "-".to_string(), |rank| rank.to_string());
        let lists = match (self.good_stratum, self.bad_stratum) {
            (Some(good), Some(bad)) => format!("{good}/{bad}"),
            _ => "-".to_string(),
        };
        write!(
            out,
            "  {rank:>7}  {:>6}  {:<7}  {:>7}  {:>6}  {lists:<13}  ",
            optional(self.ratio),
            self.label_set.unwrap_or("-"),
            optional(self.p_good),
            optional(self.p_bad),
        )?;
        if let Some(why) = why {
            write!(out, "{why:<27}  ")?;
        }
        writeln!(
            out,
            "{}{}",
            if self.no_features {
                "[no features] "
            } else {
                ""
            },
            shown_text(self.text, TEXT_CHARS)
        )
    }
}

#[derive(Debug, Clone, Serialize)]
struct ScorerJson {
    #[serde(flatten)]
    scorer: ScorerReport,
    /// Whether this scorer ranked the logs.
    primary: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ListsJson {
    good: ListUse,
    bad: ListUse,
}

#[derive(Debug, Clone, Serialize)]
struct SetsJson {
    good: u64,
    bad: u64,
    both: u64,
    neither: u64,
}

/// The `calibration` object of the JSON output.
#[derive(Debug, Clone, Serialize)]
struct CalibrationJson {
    name: String,
    revision: String,
    error_rate: f32,
    /// Requirement 11.9.
    note: &'static str,
    /// Per label, how many logs were judged on their group's own list and how many on
    /// the pooled list.
    lists: ListsJson,
    /// How many logs got each label set.
    sets: SetsJson,
}

/// The `counts` object of the JSON output.
#[derive(Debug, Clone, Serialize)]
struct CountsJson {
    logs: u64,
    review: u64,
    scored_down: u64,
    scored_down_by_calibration: u64,
    scored_down_by_budget: u64,
    abstained: u64,
    scorer_abstained: u64,
    uncertain: u64,
    unlike_calibration: u64,
    /// Logs the baseline has no features for: they share one constant ratio.
    no_features: u64,
}

/// Everything of the output but the rows.
struct Head<'a> {
    scorer: ScorerJson,
    review_budget: Option<u64>,
    calibration: Option<CalibrationJson>,
    counts: CountsJson,
    evidence_columns: Vec<&'a str>,
    input: InputReport<'a>,
    memory: MemoryReport,
    /// Most rows printed per list.
    limit: u64,
}

/// Runs `l2d triage`.
pub fn run(context: &Context, args: &TriageArgs) -> Result<(), L2dError> {
    if args.store.is_some() {
        return Err(store_not_available("triage"));
    }
    if args.primary == PrimaryChoice::Neighbour {
        return Err(L2dError::Refused {
            reason: "the neighbour scorer is not available yet: it scores against a store, \
                     and there is no encoder to fill one before tasks 2.8 and 2.9; \
                     `--primary baseline` is the scorer there is"
                .into(),
        });
    }
    let given = required(
        "--baseline",
        &args.baseline,
        "the saved scorer to rank with",
    )?;
    if args.error_rate.is_some() && args.calibration.is_none() {
        return Err(L2dError::InvalidInput {
            detail: "`--error-rate` needs `--calibration`: without a calibration the logs are \
                     ranked by ratio and no error rate applies"
                .into(),
        });
    }
    let paths = context.paths()?;
    let mut budget = context.budget()?;
    let scorer_directory = find_model(
        &paths.baselines_dir(),
        "--baseline",
        "baseline scorer",
        given,
    )?;
    let calibration_directory = args
        .calibration
        .as_deref()
        .map(|given| {
            find_model(
                &paths.calibrations_dir(),
                "--calibration",
                "calibration",
                given,
            )
        })
        .transpose()?;

    let input = read_input(&paths, &args.input, &mut budget)?;
    let input_report = InputReport::of(&input.logs, &input.summary, input.adapter, &budget)?;
    let mut scorer = load_scorer(&paths, &scorer_directory, &mut budget)?;
    let scorer_report = ScorerReport::of(&scorer, &scorer_directory);
    // The calibration is loaded but not held here: triage holds it while it runs.
    let calibrator = calibration_directory
        .as_deref()
        .map(|directory| load_calibration(&paths, directory, &budget))
        .transpose()?;

    let settings = TriageSettings {
        primary: ScorerKind::Baseline,
        review_budget: args.budget,
        calibration: calibrator.as_ref().map(|calibrator| TriageCalibration {
            calibrator,
            error_rate: args
                .error_rate
                .unwrap_or(l2d_decide::conformal::DEFAULT_ERROR_RATE),
        }),
    };
    let held_before = budget.held_bytes();
    let result = {
        let scorers: &mut [&mut dyn Scorer] = &mut [&mut scorer];
        triage_with(&input.logs, scorers, &settings, &mut budget)?
    };
    // From here on the command holds what triage held: the result and the calibration.
    budget.hold(held_before.saturating_add(result.held_bytes()));
    require_output(&budget, longest_text(&input.logs))?;

    let columns = Columns::of(result.frame())?;
    let counts = result.counts();
    let head = Head {
        scorer: ScorerJson {
            scorer: scorer_report,
            primary: true,
        },
        review_budget: result.review_budget(),
        calibration: result.calibration().map(|used| CalibrationJson {
            name: calibration_directory
                .as_deref()
                .map(model_name)
                .unwrap_or_default(),
            revision: used.revision.clone(),
            error_rate: used.error_rate,
            note: used.note,
            lists: ListsJson {
                good: used.good,
                bad: used.bad,
            },
            sets: SetsJson {
                good: used.set_good,
                bad: used.set_bad,
                both: used.set_both,
                neither: used.set_neither,
            },
        }),
        counts: CountsJson {
            logs: counts.logs,
            review: counts.review,
            scored_down: counts.scored_down(),
            scored_down_by_calibration: counts.scored_down_by_calibration,
            scored_down_by_budget: counts.scored_down_by_budget,
            abstained: counts.abstained(),
            scorer_abstained: counts.scorer_abstained,
            uncertain: counts.uncertain,
            unlike_calibration: counts.unlike_calibration,
            no_features: columns.no_features_count(),
        },
        evidence_columns: columns.evidence_names(),
        input: input_report,
        memory: context.memory_report(
            &budget,
            vec![
                Operation::batches("read", input.summary.records, input.plan.clone()),
                Operation::batches("triage", counts.logs, result.plan().clone()),
            ],
        ),
        limit: match (args.limit, context.json) {
            (Some(limit), _) => limit,
            (None, true) => u64::MAX,
            (None, false) => DEFAULT_TEXT_LIMIT,
        },
    };

    let out = io::BufWriter::new(io::stdout().lock());
    let written = if context.json {
        json(out, &head, &result, &columns)
    } else {
        text(out, &head, &result, &columns)
    };
    written.map_err(output_failed)
}

const LISTS: [(TriageList, &str); 3] = [
    (TriageList::Review, "review"),
    (TriageList::ScoredDown, "scored_down"),
    (TriageList::Abstained, "abstained"),
];

fn json(
    out: impl Write,
    head: &Head<'_>,
    result: &TriageResult,
    columns: &Columns<'_>,
) -> io::Result<()> {
    let mut json = JsonOut::begin(out)?;
    json.field("command", "triage")?;
    json.field("primary", result.primary().as_str())?;
    json.field("scorers", std::slice::from_ref(&head.scorer))?;
    json.field("review_budget", &head.review_budget)?;
    json.field("calibration", &head.calibration)?;
    json.field("counts", &head.counts)?;
    json.field("evidence_columns", &head.evidence_columns)?;
    json.field("input", &head.input)?;
    let mut omitted = Map::new();
    for (list, key) in LISTS {
        let rows = result.rows(list);
        json.begin_list(key)?;
        for row in rows.iter().take(head.limit.min(rows.len() as u64) as usize) {
            json.element(&columns.row(*row))?;
        }
        json.end_list()?;
        omitted.insert(
            key.to_string(),
            (rows.len() as u64).saturating_sub(head.limit).into(),
        );
    }
    json.field("rows_omitted", &omitted)?;
    json.field("memory", &head.memory)?;
    json.end()
}

/// The heading of a list: how many logs, in what order, and how many are shown.
fn heading(
    out: &mut impl Write,
    name: &str,
    logs: usize,
    order: &str,
    limit: u64,
) -> io::Result<()> {
    let shown = if logs as u64 > limit {
        format!("; the first {limit} shown")
    } else {
        String::new()
    };
    writeln!(out)?;
    writeln!(out, "{name} ({logs} logs, {order}{shown})")
}

fn text(
    mut out: impl Write,
    head: &Head<'_>,
    result: &TriageResult,
    columns: &Columns<'_>,
) -> io::Result<()> {
    let counts = &head.counts;
    let scorer = &head.scorer.scorer;
    writeln!(
        out,
        "triage of {} logs by the {} scorer {} (revision {})",
        counts.logs, scorer.kind, scorer.name, scorer.revision
    )?;
    match &head.calibration {
        None => writeln!(
            out,
            "calibration: none (every log with a ratio is ranked; no label sets)"
        )?,
        Some(calibration) => {
            writeln!(
                out,
                "calibration: {} (revision {}), error rate {}",
                calibration.name, calibration.revision, calibration.error_rate
            )?;
            writeln!(out, "  {}", calibration.note)?;
            let sets = &calibration.sets;
            writeln!(
                out,
                "  label sets: {} good, {} bad, {} both, {} neither",
                sets.good, sets.bad, sets.both, sets.neither
            )?;
            let lists = &calibration.lists;
            writeln!(
                out,
                "  judged on a group's own list / on the pooled list: good {} / {}, bad {} / {}",
                lists.good.group, lists.good.pooled, lists.bad.group, lists.bad.pooled
            )?;
        }
    }
    match head.review_budget {
        Some(budget) => writeln!(out, "review budget: {budget} logs")?,
        None => writeln!(out, "review budget: none")?,
    }
    writeln!(
        out,
        "counts: {} for review, {} scored down ({} by calibration, {} by the budget only), {} \
         abstained ({} the scorer abstained on, {} uncertain, {} unlike the calibration logs)",
        counts.review,
        counts.scored_down,
        counts.scored_down_by_calibration,
        counts.scored_down_by_budget,
        counts.abstained,
        counts.scorer_abstained,
        counts.uncertain,
        counts.unlike_calibration
    )?;
    if counts.no_features > 0 {
        writeln!(
            out,
            "logs with no features: {} (they share one constant ratio and carry no evidence)",
            counts.no_features
        )?;
    }
    head.input.write_text(&mut out, DEFAULT_TEXT_LIMIT)?;
    head.memory.write_text(&mut out)?;

    const COLUMNS: &str = "     rank   ratio  set      p(good)  p(bad)  lists good/bad";
    for (list, _) in LISTS {
        let rows = result.rows(list);
        let (name, order, why) = match list {
            TriageList::Review => ("review", "highest ratio first", None),
            TriageList::ScoredDown => (
                "scored down",
                "by the budget in rank order, then by calibration, highest ratio first",
                Some("scored down"),
            ),
            TriageList::Abstained => ("abstained", "in input order", Some("reason")),
        };
        heading(&mut out, name, rows.len(), order, head.limit)?;
        if rows.is_empty() {
            continue;
        }
        match why {
            Some(why) => writeln!(out, "{COLUMNS}  {why:<27}  text")?,
            None => writeln!(out, "{COLUMNS}  text")?,
        }
        for row in rows.iter().take(head.limit.min(rows.len() as u64) as usize) {
            let row = columns.row(*row);
            let why = match list {
                TriageList::Review => None,
                TriageList::ScoredDown => Some(match row.scored_down_by {
                    Some("calibration") => "by calibration",
                    Some("budget_only") => "budget only",
                    _ => "-",
                }),
                TriageList::Abstained => Some(match row.abstain_reason {
                    Some("scorer_abstained") => "the scorer abstained",
                    Some("uncertain") => "uncertain",
                    Some("unlike_calibration") => "unlike the calibration logs",
                    _ => "-",
                }),
            };
            row.write_text(&mut out, why)?;
        }
        if rows.len() as u64 > head.limit {
            writeln!(
                out,
                "  ... {} more (`--limit` sets how many are shown)",
                rows.len() as u64 - head.limit
            )?;
        }
    }
    out.flush()
}
