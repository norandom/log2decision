//! What the wired commands share: the project and its locations, the memory budget and
//! its report, reading an input, finding and loading saved models.

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use clap::{Args, ValueEnum};
use l2d_baseline::BaselineScorer;
use l2d_baseline::scorer::MAX_NAME_BYTES;
use l2d_core::input::PARSE_COST;
use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::polars::prelude::Column;
use l2d_core::types::SHORT_REVISION_LEN;
use l2d_core::{
    BatchPlan, Csic2010Adapter, InputAdapter, InputSummary, L2dError, LabelledAdapter,
    LabelledFormat, LinesAdapter, LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
    SkippedRecord,
};
use l2d_decide::ConformalCalibrator;
use serde::Serialize;

use crate::output::human_bytes;

/// Rows of a list shown in the text output when `--limit` is not given.
pub const DEFAULT_TEXT_LIMIT: u64 = 20;
/// Characters of a log text shown on one line of the text output.
pub const TEXT_CHARS: usize = 120;
/// Memory a command asks for while it prints, beside the longest text: the output
/// buffer and the row being written.
pub const OUTPUT_BYTES: u64 = 256 * 1024;
/// Memory per byte of the longest text while a row is printed: the text in the row
/// being written, its escaped form, and the output buffer growing to hold it.
pub const OUTPUT_TEXT_FACTOR: u64 = 8;
/// The least a memory limit must leave for the rows of a batch of a read, beside the
/// fixed memory of the batch: below it not even a small input can be read.
const SMALLEST_BATCH_ROOM_BYTES: u64 = 256 * 1024;
/// Memory per row while the labels and group keys of an input are counted.
const SUMMARY_ROW_BYTES: u64 = 64;

/// What the analyst set for every command: the project, the output form, the memory.
#[derive(Debug, Clone)]
pub struct Context {
    /// `--json`.
    pub json: bool,
    /// `--project`: the directory that holds `models/`, `data/` and the rest.
    pub project: PathBuf,
    /// `--batch-rows`, `--memory-limit`, `--memory-share`, `--reserve-share`.
    pub memory: MemorySettings,
    /// Whether `--memory-share` was typed.
    pub memory_share_given: bool,
    /// Whether `--reserve-share` was typed.
    pub reserve_share_given: bool,
}

fn invalid(detail: impl Into<String>) -> L2dError {
    L2dError::InvalidInput {
        detail: detail.into(),
    }
}

fn refused(reason: impl Into<String>) -> L2dError {
    L2dError::Refused {
        reason: reason.into(),
    }
}

/// The path without the `\\?\` prefix that making a path absolute puts before a drive
/// letter on Windows; other paths are returned as they are.
fn plain(path: PathBuf) -> PathBuf {
    match path.to_str() {
        Some(text) if text.starts_with(r"\\?\") && !text.starts_with(r"\\?\UNC\") => {
            PathBuf::from(&text[4..])
        }
        _ => path,
    }
}

impl Context {
    /// The project's locations, with the root made absolute and free of `..` (the
    /// scorer and the calibration refuse a model directory written with `..`).
    pub fn paths(&self) -> Result<ProjectPaths, L2dError> {
        std::fs::canonicalize(&self.project)
            .map(|root| ProjectPaths::new(plain(root)))
            .map_err(|_| L2dError::Missing {
                item: "project directory (`--project`)".into(),
                expected_at: self.project.clone(),
            })
    }

    /// Observes the free memory and derives the limit and the reserve (requirement
    /// 12.1). Every command calls this before it reads anything, so that an option
    /// value that cannot be honoured is reported first.
    pub fn budget(&self) -> Result<MemoryBudget, L2dError> {
        if self.memory_share_given && self.memory.memory_limit_bytes.is_some() {
            return Err(invalid(
                "`--memory-share` cannot be used with `--memory-limit`: the limit is either \
                 a share of the free memory or the size given",
            ));
        }
        let limit_given = self.memory.memory_limit_bytes.is_some();
        let shares = self.shares()?;
        let settings = MemorySettings {
            free_share: shares.limit,
            reserve_share: shares.reserve,
            ..self.memory.clone()
        };
        // The reserve is judged first, on a reading of its own, so that a reserve that
        // leaves nothing is named as such whatever it does to the limit.
        let free = l2d_core::memory::free_physical_memory()?;
        check_reserve(
            free,
            (free as f64 * f64::from(shares.reserve)) as u64,
            shares.reserve,
            self.reserve_share_given,
        )?;
        // A share the analyst typed (or one lowered because of what was typed) that
        // gives a limit the tool cannot work with is an option value that cannot be
        // honoured (exit 2), like a typed limit; with the default share it is the host
        // that is short (exit 4).
        let typed = self.memory_share_given || shares.limit_adjusted;
        let budget = MemoryBudget::observe(&settings).map_err(|error| match error {
            L2dError::Refused { reason } if typed => invalid(format!(
                "a limit share of {} ({}) cannot be honoured: {reason}",
                shares.limit,
                if shares.limit_adjusted {
                    "what `--reserve-share` leaves of 1"
                } else {
                    "`--memory-share`"
                }
            )),
            other => other,
        })?;
        check_reserve(
            budget.free_bytes(),
            budget.reserve_bytes(),
            shares.reserve,
            self.reserve_share_given,
        )?;
        check_room_for_a_batch(&budget, limit_given || typed)?;
        Ok(budget)
    }

    /// The shares in force (owner decision after the review of task 1.11). A limit
    /// share and a reserve share above 1 together would make the tool fall below its
    /// reserve, and halve its batches, before it has used its limit. So when one share
    /// was typed and the default of the other does not fit beside it, the other is
    /// lowered to what is left of 1, and the output says so; when both were typed and
    /// add up to more than 1 it is an analyst error. With `--memory-limit` the limit
    /// is not a share and nothing is adjusted.
    pub fn shares(&self) -> Result<Shares, L2dError> {
        let (limit, reserve) = (self.memory.free_share, self.memory.reserve_share);
        let mut shares = Shares {
            limit,
            reserve,
            limit_adjusted: false,
            reserve_adjusted: false,
        };
        let fits = f64::from(limit) + f64::from(reserve) <= 1.0 + 1e-6;
        if fits || self.memory.memory_limit_bytes.is_some() {
            return Ok(shares);
        }
        // What is left of 1 beside a typed share, to six decimals and never nothing.
        let rest = |typed: f32| (((1.0 - f64::from(typed)) * 1e6).round() / 1e6).max(1e-6) as f32;
        match (self.memory_share_given, self.reserve_share_given) {
            (true, true) => check_shares(limit, reserve)?,
            (true, false) => {
                shares.reserve = rest(limit);
                shares.reserve_adjusted = true;
            }
            (false, true) => {
                shares.limit = rest(reserve);
                shares.limit_adjusted = true;
            }
            (false, false) => {}
        }
        Ok(shares)
    }

    /// The memory report of a finished command (requirements 12.3, 12.8).
    pub fn memory_report(&self, budget: &MemoryBudget, operations: Vec<Operation>) -> MemoryReport {
        let limit_set = self.memory.memory_limit_bytes.is_some();
        // The budget was observed with these shares, so they cannot fail here.
        let shares = self.shares().unwrap_or(Shares {
            limit: self.memory.free_share,
            reserve: self.memory.reserve_share,
            limit_adjusted: false,
            reserve_adjusted: false,
        });
        MemoryReport {
            free_bytes: budget.free_bytes(),
            limit_bytes: budget.limit_bytes(),
            reserve_bytes: budget.reserve_bytes(),
            limit_share: (!limit_set).then_some(shares.limit),
            limit_share_adjusted: shares.limit_adjusted,
            reserve_share: shares.reserve,
            reserve_share_adjusted: shares.reserve_adjusted,
            limit_set_by_analyst: limit_set,
            operations,
        }
    }
}

/// Refuses a reserve that leaves no usable memory: the tool halves its batches while
/// the free memory is below the reserve, so a reserve that leaves no more free memory
/// than the tool itself needs before it reads anything leaves nothing to work with.
/// An analyst's `--reserve-share` is an option value that cannot be honoured (exit 2);
/// with the default share the host has too little free memory (exit 4).
pub fn check_reserve(
    free_bytes: u64,
    reserve_bytes: u64,
    reserve_share: f32,
    given: bool,
) -> Result<(), L2dError> {
    let left = free_bytes.saturating_sub(reserve_bytes);
    if left > PROCESS_BASELINE_BYTES {
        return Ok(());
    }
    let detail = format!(
        "a reserve of {} (`--reserve-share` {reserve_share} of the {} of free memory) leaves \
         {}, which is not above the {} the tool itself needs before it reads anything",
        human_bytes(reserve_bytes),
        human_bytes(free_bytes),
        human_bytes(left),
        human_bytes(PROCESS_BASELINE_BYTES),
    );
    Err(if given {
        invalid(detail)
    } else {
        refused(detail)
    })
}

/// The limit share and the reserve share in force, and whether one was lowered to fit
/// beside the other ([`Context::shares`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shares {
    pub limit: f32,
    pub reserve: f32,
    pub limit_adjusted: bool,
    pub reserve_adjusted: bool,
}

/// Refuses shares that add up to more than 1 (exit 2; for two shares the analyst
/// typed): the tool halves its batches as soon as the free memory is below the
/// reserve, so with a limit share and a reserve share above 1 together that happens
/// before the tool has used its limit, and the limit could never be reached at the
/// full batch size. The defaults (0.5 and 0.25) add up to 0.75.
pub fn check_shares(limit_share: f32, reserve_share: f32) -> Result<(), L2dError> {
    if f64::from(limit_share) + f64::from(reserve_share) <= 1.0 + 1e-6 {
        return Ok(());
    }
    Err(invalid(format!(
        "`--memory-share` {limit_share} and `--reserve-share` {reserve_share} add up to more \
         than 1: the tool would fall below its reserve, and halve its batches, before it has \
         used its memory limit, so the limit could never be reached; with `--memory-share` \
         {limit_share} type a `--reserve-share` of at most {}, or with `--reserve-share` \
         {reserve_share} a `--memory-share` of at most {}, or type one of the two alone and \
         the other is set to what is left of 1",
        ((1.0 - f64::from(limit_share)) * 1e6).round() / 1e6,
        ((1.0 - f64::from(reserve_share)) * 1e6).round() / 1e6,
    )))
}

/// Refuses a memory limit that leaves nothing for a batch of a read: without this the
/// read itself refuses with a message about a record "larger than 1 B" (task 1.15).
fn check_room_for_a_batch(budget: &MemoryBudget, limit_given: bool) -> Result<(), L2dError> {
    let needed = PARSE_COST.batch_bytes + SMALLEST_BATCH_ROOM_BYTES;
    if budget.usable_bytes() >= needed {
        return Ok(());
    }
    let detail = format!(
        "the memory limit of {} ({} bytes) leaves nothing for a batch: {} is kept for the \
         tool itself, reading takes {} per batch before its first row, and the rows of a \
         batch need at least {}; a limit of {} or more can read a small input",
        human_bytes(budget.limit_bytes()),
        budget.limit_bytes(),
        human_bytes(PROCESS_BASELINE_BYTES),
        human_bytes(PARSE_COST.batch_bytes),
        human_bytes(SMALLEST_BATCH_ROOM_BYTES),
        human_bytes(PROCESS_BASELINE_BYTES + needed),
    );
    Err(if limit_given {
        invalid(detail)
    } else {
        refused(detail)
    })
}

/// One step of a command and the memory it worked with.
#[derive(Debug, Clone, Serialize)]
pub struct Operation {
    /// `read`, `fit`, `score`, `held_out_check`, `calibrate` or `triage`.
    pub operation: &'static str,
    /// What the step went through: records for `read`, logs otherwise.
    pub rows: u64,
    /// The batches of a step that works in batches: what was observed and chosen, the
    /// changes of the batch size, and the smallest and largest batch size chosen. The
    /// sizes are upper bounds on a batch (`batch_rows`, `min_rows_used`,
    /// `max_rows_used`): an input of fewer rows fills one batch of that many rows at
    /// most, and `rows` says how many there were.
    pub batches: Option<BatchPlan>,
    /// The estimate of a step that needs its whole dataset at once, in bytes.
    pub whole_dataset_bytes: Option<u64>,
}

impl Operation {
    pub fn batches(operation: &'static str, rows: u64, plan: BatchPlan) -> Self {
        Self {
            operation,
            rows,
            batches: Some(plan),
            whole_dataset_bytes: None,
        }
    }

    pub fn whole(operation: &'static str, rows: u64, bytes: u64) -> Self {
        Self {
            operation,
            rows,
            batches: None,
            whole_dataset_bytes: Some(bytes),
        }
    }

    fn label(&self) -> &'static str {
        match self.operation {
            "read" => "reading the input",
            "fit" => "fitting",
            "score" => "scoring",
            "held_out_check" => "held-out check",
            "calibrate" => "calibrating",
            other => other,
        }
    }

    /// What the rows of the step are called.
    fn unit(&self) -> &'static str {
        match self.operation {
            "read" => "records",
            _ => "logs",
        }
    }
}

/// The batches of a step on one line. The batch size is an upper bound, and is worded
/// as one: an input of 64 rows in "batches of at most 65536 rows" is one batch of 64.
fn batches_text(rows: u64, unit: &str, plan: &BatchPlan) -> String {
    let chosen = if plan.overridden {
        "set by the analyst"
    } else {
        "chosen from the free memory"
    };
    let changes = match plan.changes.len() {
        0 => "the batch size did not change".to_string(),
        count => format!(
            "the batch size changed {count} time{} ({}), smallest {} rows, largest {} rows",
            if count == 1 { "" } else { "s" },
            plan.changes
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; "),
            plan.min_rows_used,
            plan.max_rows_used
        ),
    };
    format!(
        "{rows} {unit} in batches of at most {} rows ({chosen}), estimated {} bytes per row; \
         {changes}",
        plan.batch_rows, plan.bytes_per_row
    )
}

/// What a command reports about memory: the `memory` object of the JSON output.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryReport {
    /// The free physical memory observed before anything was read.
    pub free_bytes: u64,
    /// The memory limit applied.
    pub limit_bytes: u64,
    /// Free memory below this halves the next batch.
    pub reserve_bytes: u64,
    /// The share of the free memory used as the limit; `null` with `--memory-limit`.
    pub limit_share: Option<f32>,
    /// Whether the limit share was lowered from its default to what `--reserve-share`
    /// leaves of 1.
    pub limit_share_adjusted: bool,
    /// The share of the free memory kept as the reserve.
    pub reserve_share: f32,
    /// Whether the reserve share was lowered from its default to what `--memory-share`
    /// leaves of 1.
    pub reserve_share_adjusted: bool,
    pub limit_set_by_analyst: bool,
    /// The steps of the command, in order.
    pub operations: Vec<Operation>,
}

impl MemoryReport {
    /// The text form: one line for what was observed and applied, one per step.
    pub fn write_text(&self, out: &mut impl Write) -> io::Result<()> {
        let limit = match self.limit_share {
            Some(share) if self.limit_share_adjusted => format!(
                "share {share} of the free memory, lowered from the default {} to what \
                 --reserve-share leaves of 1",
                l2d_core::memory::DEFAULT_FREE_SHARE
            ),
            Some(share) => format!("share {share} of the free memory"),
            None => "set with --memory-limit".to_string(),
        };
        let reserve = if self.reserve_share_adjusted {
            format!(
                "share {}, lowered from the default {} to what --memory-share leaves of 1",
                self.reserve_share,
                l2d_core::memory::DEFAULT_RESERVE_SHARE
            )
        } else {
            format!("share {}", self.reserve_share)
        };
        writeln!(
            out,
            "memory: free memory {}, memory limit {} ({limit}), reserve {} ({reserve})",
            human_bytes(self.free_bytes),
            human_bytes(self.limit_bytes),
            human_bytes(self.reserve_bytes),
        )?;
        for operation in &self.operations {
            match (&operation.batches, operation.whole_dataset_bytes) {
                (Some(plan), _) => writeln!(
                    out,
                    "  {}: {}",
                    operation.label(),
                    batches_text(operation.rows, operation.unit(), plan)
                )?,
                (None, Some(bytes)) => writeln!(
                    out,
                    "  {}: the whole dataset at once ({} {}), estimated {} ({bytes} bytes)",
                    operation.label(),
                    operation.rows,
                    operation.unit(),
                    human_bytes(bytes),
                )?,
                (None, None) => {}
            }
        }
        Ok(())
    }
}

/// How an input file is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AdapterChoice {
    /// By the file extension: .csv, .jsonl or .ndjson as labelled data, anything else
    /// as one log per line
    Auto,
    /// One log per line, unlabelled
    Lines,
    /// Comma-separated values with a header row
    Csv,
    /// One JSON object per line
    Jsonl,
    /// The HTTP CSIC 2010 dataset (data/csic-2010/csic_database.csv unless --input)
    Csic2010,
}

/// The options that name an input; on every command that reads one.
#[derive(Debug, Clone, Args)]
pub struct InputArgs {
    /// The file to read: a path, or a path below the project's data directory
    #[arg(long, value_name = "PATH")]
    pub input: Option<PathBuf>,

    /// How the file is read
    #[arg(long, value_enum, value_name = "ADAPTER", default_value_t = AdapterChoice::Auto)]
    pub adapter: AdapterChoice,

    /// csv, jsonl: the column or key with the log text [default: text]
    #[arg(long, value_name = "NAME")]
    pub text_column: Option<String>,

    /// csv, jsonl: the column or key with the label good or bad [default: label]
    #[arg(long, value_name = "NAME", conflicts_with = "no_labels")]
    pub label_column: Option<String>,

    /// csv, jsonl: the data has no label column; every log is read as unlabelled
    #[arg(long)]
    pub no_labels: bool,

    /// csv, jsonl: the column or key with the group key [default: no groups]
    #[arg(long, value_name = "NAME")]
    pub group_column: Option<String>,
}

/// An input as it was read.
pub struct Input {
    pub logs: LogsFrame,
    pub summary: InputSummary,
    pub plan: BatchPlan,
    /// The adapter that read it: `lines`, `csv`, `jsonl` or `csic2010`.
    pub adapter: &'static str,
}

/// Where an input is: the path as given when something is there, otherwise the same
/// path below the project's data directory (requirement 10.4). What is at neither is
/// missing, named with the place it was expected at (requirement 10.5).
///
/// A directory is not an input and not a missing one: it is an analyst error that says
/// so (the `csic2010` adapter alone takes a directory, the one that holds its file).
fn locate_input(paths: &ProjectPaths, given: &Path) -> Result<PathBuf, L2dError> {
    let not_a_file = |path: &Path| {
        invalid(format!(
            "`--input` {} is a directory; give the file to read",
            path.display()
        ))
    };
    if given.is_dir() {
        return Err(not_a_file(given));
    }
    if given.exists() {
        return Ok(given.to_path_buf());
    }
    let below_data = given.is_relative()
        && given
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir));
    if !below_data {
        return Err(L2dError::Missing {
            item: format!("input {}", given.display()),
            expected_at: given.to_path_buf(),
        });
    }
    let in_data = paths.data_dir().join(given);
    if in_data.is_dir() {
        return Err(not_a_file(&in_data));
    }
    if in_data.exists() {
        return Ok(in_data);
    }
    Err(L2dError::Missing {
        item: format!(
            "input {} (not at that path from the current directory either)",
            given.display()
        ),
        expected_at: in_data,
    })
}

/// Reads the input the options name, in batches within `budget`.
pub fn read_input(
    paths: &ProjectPaths,
    args: &InputArgs,
    budget: &mut MemoryBudget,
) -> Result<Input, L2dError> {
    let required = || invalid("`--input` is required: the file to read");
    let by_extension = |path: &Path| match LabelledFormat::from_path(path) {
        Ok(LabelledFormat::Csv) => AdapterChoice::Csv,
        Ok(LabelledFormat::Jsonl) => AdapterChoice::Jsonl,
        Err(_) => AdapterChoice::Lines,
    };
    let adapter = match (args.adapter, &args.input) {
        (AdapterChoice::Auto, Some(path)) => by_extension(path),
        (AdapterChoice::Auto, None) => return Err(required()),
        (chosen, _) => chosen,
    };
    let labelled = matches!(adapter, AdapterChoice::Csv | AdapterChoice::Jsonl);
    if !labelled {
        let name = if adapter == AdapterChoice::Lines {
            "lines"
        } else {
            "csic2010"
        };
        for (option, given) in [
            ("--text-column", args.text_column.is_some()),
            ("--label-column", args.label_column.is_some()),
            ("--no-labels", args.no_labels),
            ("--group-column", args.group_column.is_some()),
        ] {
            if given {
                return Err(invalid(format!(
                    "`{option}` applies to the csv and jsonl adapters; this input is read by \
                     the {name} adapter"
                )));
            }
        }
    }
    let located = |given: &Option<PathBuf>| match given {
        Some(path) => locate_input(paths, path),
        None => Err(required()),
    };
    let (name, (logs, summary, plan)) = match adapter {
        AdapterChoice::Csic2010 => {
            let path = match &args.input {
                Some(path) if path.exists() => Csic2010Adapter::locate(paths, Some(path))?,
                Some(path) => locate_input(paths, path)?,
                None => Csic2010Adapter::locate(paths, None)?,
            };
            ("csic2010", Csic2010Adapter::new().read_with(&path, budget)?)
        }
        AdapterChoice::Lines => (
            "lines",
            LinesAdapter.read_with(&located(&args.input)?, budget)?,
        ),
        AdapterChoice::Csv | AdapterChoice::Jsonl | AdapterChoice::Auto => {
            let (name, format) = if adapter == AdapterChoice::Jsonl {
                ("jsonl", LabelledFormat::Jsonl)
            } else {
                ("csv", LabelledFormat::Csv)
            };
            let mut reader = LabelledAdapter::new(format);
            if let Some(column) = &args.text_column {
                reader.text_column = column.clone();
            }
            if let Some(column) = &args.label_column {
                reader.label_column = Some(column.clone());
            }
            if args.no_labels {
                reader.label_column = None;
            }
            reader.group_column = args.group_column.clone();
            (name, reader.read_with(&located(&args.input)?, budget)?)
        }
    };
    Ok(Input {
        logs,
        summary,
        plan,
        adapter: name,
    })
}

/// How many distinct texts carry each label.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct LabelCounts {
    pub good: u64,
    pub bad: u64,
    pub unknown: u64,
}

/// What a command reports about the input it read (requirements 1.4, 1.6): the `input`
/// object of the JSON output.
#[derive(Debug, Clone, Serialize)]
pub struct InputReport<'a> {
    pub source_path: String,
    pub adapter: &'static str,
    /// Records read, the skipped ones included. For CSV these are data rows, not lines
    /// of the file.
    pub records: u64,
    pub distinct_texts: u64,
    pub content_sha256: String,
    /// Distinct texts that occurred with both labels and are therefore `unknown`.
    pub label_conflicts: u64,
    pub skipped_count: u64,
    /// Every skipped record with its position and the reason (the summary's own list,
    /// not a copy).
    pub skipped: &'a [SkippedRecord],
    /// Distinct texts per label.
    pub labels: LabelCounts,
    /// Distinct group keys.
    pub group_keys: u64,
}

impl<'a> InputReport<'a> {
    /// Counts the labels and the group keys of the logs read by `adapter` and takes
    /// over their summary.
    pub fn of(
        logs: &LogsFrame,
        summary: &'a InputSummary,
        adapter: &'static str,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        budget.require_whole(
            (logs.len() as u64).saturating_mul(SUMMARY_ROW_BYTES),
            "summarizing the input",
        )?;
        let failed = |error: &dyn std::fmt::Display| L2dError::Runtime {
            detail: format!("summarizing the input failed: {error}"),
        };
        let frame = logs.frame();
        let labels = frame
            .column(LogsFrame::LABEL)
            .and_then(Column::str)
            .map_err(|error| failed(&error))?;
        let mut counts = LabelCounts::default();
        for label in labels.iter().flatten() {
            match label {
                "good" => counts.good += 1,
                "bad" => counts.bad += 1,
                _ => counts.unknown += 1,
            }
        }
        let groups = frame
            .column(LogsFrame::GROUP)
            .and_then(Column::str)
            .map_err(|error| failed(&error))?;
        let group_keys = groups.iter().flatten().collect::<HashSet<&str>>().len() as u64;
        Ok(Self {
            source_path: summary.source_path.to_string_lossy().into_owned(),
            adapter,
            records: summary.records,
            distinct_texts: summary.distinct_texts,
            content_sha256: summary.content_sha256.clone(),
            label_conflicts: summary.label_conflicts,
            skipped_count: summary.skipped.len() as u64,
            skipped: &summary.skipped,
            labels: counts,
            group_keys,
        })
    }

    /// The text form; at most `limit` skipped records are listed.
    pub fn write_text(&self, out: &mut impl Write, limit: u64) -> io::Result<()> {
        writeln!(out, "input: {}", self.source_path)?;
        writeln!(out, "  adapter: {}", self.adapter)?;
        writeln!(
            out,
            "  records read: {} ({} skipped)",
            self.records, self.skipped_count
        )?;
        writeln!(
            out,
            "  distinct texts: {} ({} good, {} bad, {} unknown)",
            self.distinct_texts, self.labels.good, self.labels.bad, self.labels.unknown
        )?;
        writeln!(
            out,
            "  texts with conflicting labels: {}",
            self.label_conflicts
        )?;
        writeln!(out, "  group keys: {}", self.group_keys)?;
        writeln!(out, "  content SHA-256: {}", self.content_sha256)?;
        if self.skipped_count > 0 {
            writeln!(out, "  skipped records: {}", self.skipped_count)?;
            for skipped in self.skipped.iter().take(limit as usize) {
                writeln!(
                    out,
                    "    position {}: {}",
                    skipped.position,
                    crate::output::shown_text(&skipped.reason, 2 * TEXT_CHARS)
                )?;
            }
            if self.skipped_count > limit {
                writeln!(out, "    ... {} more", self.skipped_count - limit)?;
            }
        }
        Ok(())
    }
}

/// Checks a model name the analyst typed for `option`: 1 to 64 letters, digits, `.`,
/// `_` or `-`, not starting with a dot. That is the rule of the scorer and of the
/// calibration; it is checked here so that a wrong name fails before anything is read.
pub fn check_name(option: &str, name: &str) -> Result<(), L2dError> {
    let usable = !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if usable {
        return Ok(());
    }
    Err(invalid(format!(
        "`{option}`: a name is 1 to {MAX_NAME_BYTES} letters, digits, '.', '_' or '-' and \
         does not start with a dot"
    )))
}

/// The option value that is required for this command.
pub fn required<'a>(
    option: &str,
    value: &'a Option<String>,
    what: &str,
) -> Result<&'a str, L2dError> {
    value
        .as_deref()
        .ok_or_else(|| invalid(format!("`{option}` is required: {what}")))
}

/// Whether `directory` is `<name>-<12 hexadecimal digits>`.
fn is_revision_of(directory: &str, name: &str) -> bool {
    directory
        .strip_prefix(name)
        .and_then(|rest| rest.strip_prefix('-'))
        .is_some_and(|revision| {
            revision.len() == SHORT_REVISION_LEN
                && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
}

/// Finds the directory of a saved model below `parent` (requirement 10.3: nowhere
/// else). `given` is the directory name `<name>-<rev12>`, or the name alone when one
/// revision is saved under it. Nothing there is [`L2dError::Missing`] with the expected
/// place; several revisions under one name are an analyst error that lists them.
pub fn find_model(
    parent: &Path,
    option: &str,
    item: &str,
    given: &str,
) -> Result<PathBuf, L2dError> {
    check_name(option, given)?;
    let exact = parent.join(given);
    if exact.is_dir() {
        return Ok(exact);
    }
    let mut found: Vec<String> = std::fs::read_dir(parent)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|directory| is_revision_of(directory, given))
        .collect();
    found.sort();
    match found.len() {
        0 => Err(L2dError::Missing {
            item: format!("{item} '{given}'"),
            expected_at: parent.join(format!("{given}-<revision>")),
        }),
        1 => Ok(parent.join(&found[0])),
        count => Err(invalid(format!(
            "{count} {item}s are saved under the name '{given}': {}; give `{option}` the \
             directory name of one of them",
            found.join(", ")
        ))),
    }
}

/// The name a model directory `<name>-<rev12>` was saved under.
pub fn model_name(directory: &Path) -> String {
    let full = directory
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    match full.len().checked_sub(SHORT_REVISION_LEN + 1) {
        Some(end) if full.is_char_boundary(end) && is_revision_of(&full, &full[..end]) => {
            full[..end].to_string()
        }
        _ => full,
    }
}

/// Loads a baseline scorer from below the model directory and holds what it keeps in
/// `budget` on top of what is held already (requirement 12.9). The budget is asked for
/// the memory of the load first.
pub fn load_scorer(
    paths: &ProjectPaths,
    directory: &Path,
    budget: &mut MemoryBudget,
) -> Result<BaselineScorer, L2dError> {
    budget.require_whole(
        BaselineScorer::load_bytes(directory),
        "loading the baseline scorer",
    )?;
    let scorer = BaselineScorer::load_under(paths, directory)?;
    budget.hold(budget.held_bytes().saturating_add(scorer.held_bytes()));
    Ok(scorer)
}

/// Loads a calibration from below the model directory after asking `budget` for the
/// memory of the load: four times the size of its lists and eight times that of its
/// description (task 1.10 measured 16 to 33 bytes per value of 8 bytes). What the
/// calibration keeps is not held here: triage holds it while it runs, and the command
/// after that.
pub fn load_calibration(
    paths: &ProjectPaths,
    directory: &Path,
    budget: &MemoryBudget,
) -> Result<ConformalCalibrator, L2dError> {
    let size = |file: &str| std::fs::metadata(directory.join(file)).map_or(0, |found| found.len());
    budget.require_whole(
        size(l2d_decide::conformal::SCORES_FILE)
            .saturating_mul(4)
            .saturating_add(size(l2d_decide::conformal::MANIFEST_FILE).saturating_mul(8)),
        "loading the calibration",
    )?;
    ConformalCalibrator::load_under(paths, directory)
}

/// What a command reports about the baseline scorer it used: the `scorer` object of the
/// JSON output.
#[derive(Debug, Clone, Serialize)]
pub struct ScorerReport {
    pub kind: &'static str,
    pub name: String,
    pub revision: String,
    pub directory: String,
    pub settings: SettingsReport,
    /// How many fitted texts the scorer keeps a hash of; `null` for a scorer saved
    /// without them (before task 1.11).
    pub fitted_text_hashes: Option<u64>,
    /// Whether those hashes cover every input the scorer was fitted on; `false` after
    /// hashes were added to a scorer saved without them; `null` without hashes.
    pub fitted_text_hashes_complete: Option<bool>,
}

/// The settings of a baseline scorer.
#[derive(Debug, Clone, Serialize)]
pub struct SettingsReport {
    pub buckets: u32,
    pub ngram_max: u8,
    pub routing: &'static str,
    pub l2: f64,
    pub max_iter: u32,
    pub tolerance: f64,
}

impl ScorerReport {
    pub fn of(scorer: &BaselineScorer, directory: &Path) -> Self {
        let settings = scorer.settings();
        Self {
            kind: l2d_baseline::scorer::SCORER_KIND,
            name: model_name(directory),
            revision: scorer.revision().to_string(),
            directory: directory.to_string_lossy().into_owned(),
            settings: SettingsReport {
                buckets: settings.buckets,
                ngram_max: settings.ngram_max,
                routing: match settings.routing {
                    l2d_baseline::RoutingMode::Keep => "keep",
                    l2d_baseline::RoutingMode::Strip => "strip",
                    l2d_baseline::RoutingMode::Isolate => "isolate",
                },
                l2: settings.l2,
                max_iter: settings.max_iter,
                tolerance: settings.tolerance,
            },
            fitted_text_hashes: scorer.fitted_text_hashes(),
            fitted_text_hashes_complete: scorer.fitted_text_hashes_complete(),
        }
    }
}

/// The longest text of a logs frame, in bytes.
pub fn longest_text(logs: &LogsFrame) -> u64 {
    logs.frame()
        .column(LogsFrame::TEXT)
        .and_then(Column::str)
        .map(|texts| {
            texts
                .iter()
                .flatten()
                .map(|text| text.len() as u64)
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

/// What printing rows with texts of up to `longest` bytes takes, in bytes.
pub fn output_bytes(longest: u64) -> u64 {
    OUTPUT_BYTES.saturating_add(longest.saturating_mul(OUTPUT_TEXT_FACTOR))
}

/// Asks `budget` for what printing rows with texts of up to `longest` bytes takes.
pub fn require_output(budget: &MemoryBudget, longest: u64) -> Result<(), L2dError> {
    budget.require_whole(output_bytes(longest), "printing the result")
}

/// Reads an error rate as the analyst types it: a number above 0 and below 1.
pub fn error_rate(text: &str) -> Result<f32, String> {
    match text.trim().parse::<f32>() {
        Ok(rate) if rate > 0.0 && rate < 1.0 => Ok(rate),
        _ => Err("the error rate must be a number above 0 and below 1, for example 0.05".into()),
    }
}

/// The refusal for `--store` on commands that will take a store once there is an
/// encoder to fill it.
pub fn store_not_available(command: &str) -> L2dError {
    refused(format!(
        "`l2d {command} --store` is not available yet: a store holds vectors, and there is \
         no encoder to make them before tasks 2.8 and 2.9; give the logs as a file with \
         `--input`"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn a_reserve_that_leaves_no_more_than_the_tool_needs_is_rejected() {
        // The default quarter of 8 GiB leaves 6 GiB.
        assert!(check_reserve(8 * GIB, 2 * GIB, 0.25, false).is_ok());
        // One byte above what the tool needs is enough; exactly that is not.
        let free = 4 * GIB;
        assert!(check_reserve(free, free - PROCESS_BASELINE_BYTES - 1, 0.99, true).is_ok());
        let error = check_reserve(free, free - PROCESS_BASELINE_BYTES, 0.99, true).unwrap_err();
        match &error {
            L2dError::InvalidInput { detail } => {
                assert!(detail.contains("--reserve-share"), "{detail}");
                assert!(detail.contains("leaves 32.0 MiB"), "{detail}");
            }
            other => panic!("{other:?}"),
        }
        // With the default share it is the host that is short, not the option.
        let error = check_reserve(40 << 20, 10 << 20, 0.25, false).unwrap_err();
        assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");
    }

    #[test]
    fn shares_that_add_up_to_more_than_one_are_rejected_with_the_reason() {
        for (limit, reserve) in [
            (0.5, 0.25),
            (0.75, 0.25),
            (0.5, 0.5),
            (0.9, 0.1),
            (0.001, 0.999),
        ] {
            assert!(check_shares(limit, reserve).is_ok(), "{limit} + {reserve}");
        }
        for (limit, reserve) in [(0.9, 0.9), (0.8, 0.25), (0.5, 0.6), (0.76, 0.25)] {
            let error = check_shares(limit, reserve).unwrap_err();
            assert_eq!(error.exit_code(), 2, "{limit} + {reserve}");
            let text = error.to_string();
            assert!(text.contains("--memory-share") && text.contains("--reserve-share"));
            assert!(
                text.contains("more than 1") && text.contains("before"),
                "{text}"
            );
        }
        let text = check_shares(0.8, 0.6).unwrap_err().to_string();
        assert!(text.contains("`--reserve-share` of at most 0.2"), "{text}");
        assert!(text.contains("`--memory-share` of at most 0.4"), "{text}");
    }

    #[test]
    fn one_typed_share_lowers_the_default_of_the_other_to_what_is_left_of_one() {
        let context = |limit: Option<f32>, reserve: Option<f32>, bytes: Option<u64>| Context {
            json: false,
            project: PathBuf::from("."),
            memory: MemorySettings {
                memory_limit_bytes: bytes,
                max_batch_rows: None,
                free_share: limit.unwrap_or(0.5),
                reserve_share: reserve.unwrap_or(0.25),
            },
            memory_share_given: limit.is_some(),
            reserve_share_given: reserve.is_some(),
        };
        let shares = |limit, reserve, bytes| context(limit, reserve, bytes).shares();
        let plain = |limit, reserve| Shares {
            limit,
            reserve,
            limit_adjusted: false,
            reserve_adjusted: false,
        };
        assert_eq!(shares(None, None, None).unwrap(), plain(0.5, 0.25));
        assert_eq!(shares(Some(0.75), None, None).unwrap(), plain(0.75, 0.25));
        assert_eq!(shares(Some(0.5), Some(0.5), None).unwrap(), plain(0.5, 0.5));
        assert_eq!(
            shares(Some(0.8), None, None).unwrap(),
            Shares {
                reserve: 0.2,
                reserve_adjusted: true,
                ..plain(0.8, 0.25)
            }
        );
        assert_eq!(
            shares(None, Some(0.6), None).unwrap(),
            Shares {
                limit: 0.4,
                limit_adjusted: true,
                ..plain(0.5, 0.6)
            }
        );
        // Never nothing, whatever is typed.
        assert!(shares(None, Some(0.999_999_9), None).unwrap().limit > 0.0);
        // Both typed: an error. A limit in bytes: no share to adjust.
        assert_eq!(
            shares(Some(0.8), Some(0.6), None).unwrap_err().exit_code(),
            2
        );
        assert_eq!(
            shares(None, Some(0.9), Some(1 << 30)).unwrap(),
            plain(0.5, 0.9)
        );
    }

    #[test]
    fn a_model_directory_gives_back_its_name() {
        assert_eq!(
            model_name(Path::new("models/baselines/base-0123456789ab")),
            "base"
        );
        assert_eq!(model_name(Path::new("a-b.c_d-0123456789AB")), "a-b.c_d");
        assert_eq!(model_name(Path::new("short")), "short");
        assert_eq!(
            model_name(Path::new("base-0123456789xy")),
            "base-0123456789xy"
        );
        assert!(is_revision_of("base-0123456789ab", "base"));
        assert!(!is_revision_of("base-two-0123456789ab", "base"));
        assert!(!is_revision_of("base-0123456789a", "base"));
    }

    #[test]
    fn names_that_could_leave_the_model_directory_are_not_names() {
        for name in ["a", "A-b_c.1", &"x".repeat(64)] {
            assert!(check_name("--name", name).is_ok(), "{name}");
        }
        for name in [
            "",
            ".a",
            "a/b",
            "a\\b",
            "..",
            "a b",
            "C:x",
            &"x".repeat(65),
            "\u{e9}",
        ] {
            let error = check_name("--name", name).unwrap_err();
            assert!(error.to_string().contains("--name"), "{error}");
        }
    }

    #[test]
    fn an_error_rate_lies_strictly_between_zero_and_one() {
        assert_eq!(error_rate("0.05"), Ok(0.05));
        for text in ["0", "1", "-0.1", "1.5", "NaN", "inf", "five", ""] {
            assert!(error_rate(text).is_err(), "{text}");
        }
    }
}
