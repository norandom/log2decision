//! Search an input and optionally persist its existing, labelled MarkSet contract.
//! Archives are read twice: collect positions, then print borrowed texts. No matching
//! text survives its batch, and the second pass holds a fixed reservation (no reader
//! lag or growing copies). Both passes use the same entry selection and source hash.
use std::io::{self, Write};
use std::path::Path;

use clap::{Args, ValueEnum};
use l2d_core::{
    ArchivePosition, L2dError, Label, LogsFrame, MarkQuery, MarkSet, MemoryBudget, ProjectPaths,
    ZipInput,
};
use serde::Serialize;

use super::common::{
    self, ArchiveArgs, AtOnceDefault, Context, InputArgs, MemoryReport, Operation,
};
use super::ingest::{ArchiveReport, archive_report_text, fail_when_damaged};
use crate::output::{JsonOut, output_failed, shown_text};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MarkLabel {
    Bad,
    Good,
}

#[derive(Debug, Args)]
pub struct MarkArgs {
    #[command(flatten)]
    pub input: InputArgs,
    #[command(flatten)]
    pub archive: ArchiveArgs,
    /// Match this literal, case-sensitive substring
    #[arg(long, required_unless_present = "regex", conflicts_with = "regex")]
    pub phrase: Option<String>,
    /// Match this regular expression (use (?i) for case-insensitive matching)
    #[arg(long, required_unless_present = "phrase")]
    pub regex: Option<String>,
    /// Analyst label saved with the marks; dataset labels do not choose matches
    #[arg(long, value_enum, default_value = "bad")]
    pub label: MarkLabel,
    /// Save as marks/<NAME>.json in the project
    #[arg(long)]
    pub name: Option<String>,
}

pub fn run(context: &Context, args: &MarkArgs) -> Result<(), L2dError> {
    if args.input.input.is_none() {
        return Err(L2dError::InvalidInput {
            detail: "`l2d mark` requires --input <file|archive>".into(),
        });
    }
    let paths = context.paths()?;
    let mut budget = context.budget()?;
    let query = match (&args.phrase, &args.regex) {
        (Some(phrase), None) => MarkQuery::Phrase {
            phrase: phrase.clone(),
        },
        (None, Some(pattern)) => MarkQuery::Regex {
            pattern: pattern.clone(),
        },
        _ => {
            return Err(L2dError::InvalidInput {
                detail: "give exactly one of --phrase or --regex".into(),
            });
        }
    };
    let name = args.name.as_deref().unwrap_or("search");
    let label = match args.label {
        MarkLabel::Bad => Label::Bad,
        MarkLabel::Good => Label::Good,
    };
    if let Some(path) = common::archive_input(&paths, &args.input)? {
        return archive(
            context,
            args,
            &paths,
            &path,
            name,
            label,
            query,
            &mut budget,
        );
    }
    args.archive.refuse_for_other_input(&[])?;
    let input = common::read_input(&paths, &args.input, &mut budget)?;
    let marks = MarkSet::mark(name, label, query, &input.logs, &input.summary, &budget)?;
    budget.require_whole(mark_bytes(&marks), "keeping the mark set for output")?;
    budget.hold(budget.held_bytes().saturating_add(mark_bytes(&marks)));
    common::require_output(&budget, common::longest_text(&input.logs))?;
    let saved = save(args, &marks, &paths, &budget)?;
    let memory = context.memory_report(
        &budget,
        vec![Operation::batches(
            "mark",
            input.summary.records,
            input.plan,
        )],
    );
    let mut out = MarkOut::begin(io::BufWriter::new(io::stdout().lock()), context.json)?;
    if let Err(error) = rows(&input.logs, &marks, &mut out, |_| None) {
        out.cut_off()?;
        return Err(error);
    }
    out.finish(&marks, saved.as_deref(), &input.summary, &memory)
}

// Only freshly created input mark sets reach here: there are no ids, de-selections
// or rules. Positions are copied into an exactly sized Vec by MarkSet; count twice
// their bytes for allocator slack. Include query and entry strings (including the
// command's copies), source identity, name, containers and serialization overhead.
fn mark_bytes(marks: &MarkSet) -> u64 {
    let query = match marks.query() {
        MarkQuery::Phrase { phrase } => phrase.len(),
        MarkQuery::Regex { pattern } => pattern.len(),
        MarkQuery::Positions { positions } => positions.len().saturating_mul(8),
    } as u64;
    let source = marks.input().map_or(0, |source| {
        source.path.len() as u64
            + source.content_sha256.len() as u64
            + source.archive_entries.as_ref().map_or(0, |entries| {
                entries.iter().map(|s| s.len() as u64 + 32).sum::<u64>()
            })
    });
    4096 + 16 * marks.count() + 4 * (query + source + marks.name().len() as u64)
}

fn save(
    args: &MarkArgs,
    marks: &MarkSet,
    paths: &ProjectPaths,
    budget: &MemoryBudget,
) -> Result<Option<std::path::PathBuf>, L2dError> {
    args.name
        .as_ref()
        .map(|_| marks.save(paths, budget))
        .transpose()
}

#[allow(clippy::too_many_arguments)]
fn archive(
    context: &Context,
    args: &MarkArgs,
    paths: &ProjectPaths,
    path: &Path,
    name: &str,
    label: Label,
    query: MarkQuery,
    budget: &mut MemoryBudget,
) -> Result<(), L2dError> {
    let settings = args.archive.settings(AtOnceDefault::WhenTheyFit);
    let (marks, summary) = MarkSet::mark_archive(name, label, query, path, &settings, budget)?;
    marks.require_input(&summary.input, Some(&settings.entries))?;
    // Release entry paths and skipped-record diagnostics before reopening. The next
    // reader supplies entry names by reference and its final summary for the report.
    drop(summary);
    let held = mark_bytes(&marks).saturating_add(common::output_bytes(settings.max_record_bytes));
    budget.hold(held);
    budget.require_held(0)?;
    let mut input = ZipInput::open(path, &settings, budget)?;
    if marks
        .input()
        .is_none_or(|source| source.content_sha256 != input.content_sha256())
    {
        return Err(L2dError::Refused {
            reason: "the archive changed between marking and printing; nothing was saved".into(),
        });
    }
    // ZipInput replaces budget.held while it reads, so it must know about this fixed
    // reservation too. No texts or entry names are copied while output is streamed.
    input.hold_beside(held);

    let mut out = MarkOut::begin(io::BufWriter::new(io::stdout().lock()), context.json)?;
    let result = (|| {
        let mut batches = input.batches(budget);
        while let Some(batch) = batches.next() {
            let batch = batch?;
            rows(&batch, &marks, &mut out, |position| {
                let p = ArchivePosition::unpack(position);
                batches.entry_path(p.entry).map(|entry| (entry, p.line))
            })?;
        }
        Ok::<_, L2dError>(())
    })();
    if let Err(error) = result {
        out.cut_off()?;
        return Err(error);
    }
    let plan = input
        .batch_plan()
        .cloned()
        .ok_or_else(|| L2dError::Runtime {
            detail: "the archive was read without a batch plan".into(),
        })?;
    let at_once = input.entries_at_once();
    let summary = input.finish();
    // Keep the reader's conservative holding in the budget: it includes the final
    // summary, positions and output buffers while persistence does its own checks.
    let completed = (|| {
        marks.require_input(&summary.input, Some(&settings.entries))?;
        if summary.damaged().next().is_some() {
            return Ok(None);
        }
        save(args, &marks, paths, budget)
    })();
    let saved = match completed {
        Ok(saved) => saved,
        Err(error) => {
            out.cut_off()?;
            return Err(error);
        }
    };
    let memory = context.memory_report(
        budget,
        vec![Operation::batches("mark", summary.input.records, plan).with_entries_at_once(at_once)],
    );
    let report = ArchiveReport::of(&summary);
    if let MarkOut::Text(text) = &mut out {
        archive_report_text(text, &report, u64::MAX).map_err(output_failed)?;
    }
    out.finish(&marks, saved.as_deref(), &report, &memory)?;
    fail_when_damaged(&summary, path)
}

#[derive(Serialize)]
struct Match<'a> {
    first_position: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    entry: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<u64>,
    text: &'a str,
}

fn rows<'a>(
    logs: &LogsFrame,
    marks: &MarkSet,
    out: &mut MarkOut<impl Write>,
    locate: impl Fn(u64) -> Option<(&'a str, u64)>,
) -> Result<(), L2dError> {
    let frame = logs.frame();
    let failed = |error| L2dError::Runtime {
        detail: format!("reading marked rows failed: {error}"),
    };
    let positions = frame
        .column(LogsFrame::FIRST_POSITION)
        .and_then(|column| column.u64())
        .map_err(failed)?;
    let texts = frame
        .column(LogsFrame::TEXT)
        .and_then(|column| column.str())
        .map_err(failed)?;
    for (position, text) in positions.into_no_null_iter().zip(texts.into_no_null_iter()) {
        if marks.is_selected(position) {
            let location = locate(position);
            out.row(&Match {
                first_position: position,
                entry: location.map(|p| p.0),
                line: location.map(|p| p.1),
                text,
            })?;
        }
    }
    Ok(())
}

enum MarkOut<W: Write> {
    Json(JsonOut<W>),
    Text(W),
}
impl<W: Write> MarkOut<W> {
    fn begin(out: W, json: bool) -> Result<Self, L2dError> {
        if json {
            let mut out = JsonOut::begin(out).map_err(output_failed)?;
            out.field("command", "mark").map_err(output_failed)?;
            out.begin_list("matches").map_err(output_failed)?;
            Ok(Self::Json(out))
        } else {
            Ok(Self::Text(out))
        }
    }
    fn row(&mut self, row: &Match<'_>) -> Result<(), L2dError> {
        match self {
            Self::Json(out) => out.element(row),
            Self::Text(out) => match (row.entry, row.line) {
                (Some(entry), Some(line)) => writeln!(
                    out,
                    "{}:{line}  {}",
                    shown_text(entry, 120),
                    shown_text(row.text, 120)
                ),
                _ => writeln!(out, "{}  {}", row.first_position, shown_text(row.text, 120)),
            },
        }
        .map_err(output_failed)
    }
    fn cut_off(self) -> Result<(), L2dError> {
        match self {
            Self::Json(out) => out.cut_off(),
            Self::Text(mut out) => out.flush(),
        }
        .map_err(output_failed)
    }
    fn finish(
        self,
        marks: &MarkSet,
        saved: Option<&Path>,
        input: &impl Serialize,
        memory: &MemoryReport,
    ) -> Result<(), L2dError> {
        (|| match self {
            Self::Json(mut out) => {
                out.end_list()?;
                out.field("count", &marks.count())?;
                out.field("label", &marks.label())?;
                out.field("query", marks.query())?;
                out.field("source", marks.source())?;
                out.field("saved", &saved)?;
                out.field("input", input)?;
                out.field("memory", memory)?;
                out.end()
            }
            Self::Text(mut out) => {
                writeln!(
                    out,
                    "{} distinct texts marked {}",
                    marks.count(),
                    marks.label().as_str()
                )?;
                if let Some(saved) = saved {
                    writeln!(out, "saved: {}", shown_text(&saved.to_string_lossy(), 240))?;
                }
                memory.write_text(&mut out)?;
                out.flush()
            }
        })()
        .map_err(output_failed)
    }
}
