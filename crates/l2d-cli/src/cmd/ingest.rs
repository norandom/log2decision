//! `l2d ingest`: read and summarize an input (requirements 1.1 to 1.4, 1.6), or a zip
//! archive of inputs (requirement 13).
//!
//! Nothing is kept: the command shows what a later command would read.
//!
//! # Archives
//!
//! A zip archive is read entry by entry, in batches that are counted and let go, so an
//! archive larger than the memory is summarized within the memory limit. The report
//! names every entry: read with its records, or not read with the reason. Entry names
//! are data from the archive; in the text output they are shown on one line with
//! control characters escaped.
//!
//! When the distinct texts of an archive outgrow the memory limit, they are
//! deduplicated on disk, as `l2d baseline score` does it and with the same code
//! ([`super::archive_run`]; task 1.20): the counts are those of a read in memory.
//! The disk is used only then (remediation of task 1.20): while the set fits, the
//! command makes no directory, looks at no disk, holds nothing more and prints what
//! it printed before task 1.20. When the set is frozen, the temporary files go to
//! `--work-dir DIR`, or to `<project>/work/ingest-<UTC time>-<process id>/` when it is
//! not given, which is made then; the free space of its volume is checked then against
//! an estimate, and a refusal there ends the command during the read (exit code 4, the
//! records read until then in the message, no report). A directory given with
//! `--work-dir` must be empty or absent, which is checked before the read. When the
//! command ends, however it ends, its files are gone, and so is every directory it
//! made; a directory that was there before stays, empty as it was. The output says
//! when the disk was used (`work` in JSON, two lines of the text output).
//!
//! The text output counts the entries that match no pattern and the directories and
//! lists the others; the JSON output lists every entry. In JSON a skipped record
//! carries `entry` and `line`, which are the fields to use, and the packed `position`
//! as a number, which exceeds 2^53 from entry index 8,192 on.
//!
//! A read of the archive file that fails (the disk, not the data) ends the command with
//! exit code 5 and no report.
//!
//! When an entry is damaged the command still prints the whole report, with the
//! records of the other entries, and then fails with exit code 2 (requirement 13.6).
//! With `--json` standard output then holds two lines: the report, and the error
//! envelope on the last line.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use clap::Args;
use l2d_core::{ArchiveSummary, EntryReport, EntrySkip, L2dError, ZipInput};
use serde::{Serialize, Serializer};

use super::archive_run::{DedupReport, spill_when_needed};
use super::common::{
    ArchiveArgs, Context, DEFAULT_TEXT_LIMIT, InputArgs, InputReport, LabelCounts, MemoryReport,
    Operation, TEXT_CHARS, archive_input, read_input,
};
use crate::output::{JsonOut, output_failed, shown_text};

/// The options of `l2d ingest`.
#[derive(Debug, Clone, Args)]
pub struct IngestArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// Most skipped records listed in the text output [default: 20]; JSON lists all.
    /// For a zip archive also the most entries listed
    #[arg(long, value_name = "N")]
    pub limit: Option<u64>,

    #[command(flatten)]
    pub archive: ArchiveArgs,

    /// zip: the directory for the temporary files of deduplication on disk; it must
    /// be empty or absent, and is removed again when the command made it
    /// [default: work/ingest-<UTC time>-<process id>/ below the project]
    #[arg(long, value_name = "DIR")]
    pub work_dir: Option<PathBuf>,
}

/// Runs `l2d ingest`.
pub fn run(context: &Context, args: &IngestArgs) -> Result<(), L2dError> {
    let paths = context.paths()?;
    let mut budget = context.budget()?;
    if let Some(archive) = archive_input(&paths, &args.input)? {
        return run_archive(context, args, &archive);
    }
    args.archive
        .refuse_for_other_input(&[("--work-dir", args.work_dir.is_some())])?;
    let input = read_input(&paths, &args.input, &mut budget)?;
    let report = InputReport::of(&input.logs, &input.summary, input.adapter, &budget)?;
    let memory = context.memory_report(
        &budget,
        vec![Operation::batches(
            "read",
            input.summary.records,
            input.plan,
        )],
    );

    let out = io::stdout().lock();
    let written = if context.json {
        json(out, &report, &memory)
    } else {
        text(
            out,
            &report,
            &memory,
            args.limit.unwrap_or(DEFAULT_TEXT_LIMIT),
        )
    };
    written.map_err(output_failed)
}

fn json(out: impl Write, input: &impl Serialize, memory: &MemoryReport) -> io::Result<()> {
    let mut json = JsonOut::begin(io::BufWriter::new(out))?;
    json.field("command", "ingest")?;
    json.field("input", input)?;
    json.field("memory", memory)?;
    json.end()
}

fn text(
    out: impl Write,
    input: &InputReport<'_>,
    memory: &MemoryReport,
    limit: u64,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(out);
    input.write_text(&mut out, limit)?;
    memory.write_text(&mut out)?;
    out.flush()
}

// --- Zip archives --------------------------------------------------------------------------

/// Where the temporary files of a read that went on on disk were: the `work` object of
/// the JSON output, which a read in memory does not have. The directory is gone again
/// when the command made it.
#[derive(Debug, Serialize)]
struct WorkReport {
    directory: String,
    deduplication: DedupReport,
    /// Free disk space of the directory's volume when the set was frozen, and what
    /// the read was estimated to need there.
    disk_free_bytes: u64,
    disk_estimated_bytes: u64,
}

/// Reads the archive at `archive` to its end and prints its report.
fn run_archive(context: &Context, args: &IngestArgs, archive: &Path) -> Result<(), L2dError> {
    let mut budget = context.budget()?;
    let mut input = ZipInput::open(archive, &args.archive.settings(), &budget)?;
    // Nothing is kept in the directory, so its name need not be told from the time
    // alone: with the process id two runs in one second do not meet.
    let name = format!("ingest-<UTC time>-{}", std::process::id());
    spill_when_needed(context, &mut input, &name, args.work_dir.as_deref())?;
    for batch in input.batches(&mut budget) {
        // Only the summary is wanted: the rows of a batch are let go at once.
        batch?;
    }
    let plan = input
        .batch_plan()
        .cloned()
        .ok_or_else(|| L2dError::Runtime {
            detail: "the archive was read without a batch plan".into(),
        })?;
    let work = match (input.disk_dedup(), input.freeze_disk()) {
        (Some(dedup), Some(disk)) => Some(WorkReport {
            directory: disk.directory.to_string_lossy().into_owned(),
            deduplication: DedupReport::of(Some(dedup), 0),
            disk_free_bytes: disk.free_bytes,
            disk_estimated_bytes: disk.estimated_bytes,
        }),
        _ => None,
    };
    // Nothing is kept: the files and what the run made of its directory go now.
    let summary = input.finish();
    let report = ArchiveReport::of(&summary);
    let memory = context.memory_report(
        &budget,
        vec![Operation::batches("read", summary.input.records, plan)],
    );

    let out = io::stdout().lock();
    let written = if context.json {
        archive_json(out, &report, work.as_ref(), &memory)
    } else {
        archive_text(
            out,
            &report,
            work.as_ref(),
            &memory,
            args.limit.unwrap_or(DEFAULT_TEXT_LIMIT),
        )
    };
    written.map_err(output_failed)?;
    fail_when_damaged(&summary, archive)
}

fn archive_json(
    out: impl Write,
    input: &ArchiveReport<'_>,
    work: Option<&WorkReport>,
    memory: &MemoryReport,
) -> io::Result<()> {
    let mut json = JsonOut::begin(io::BufWriter::new(out))?;
    json.field("command", "ingest")?;
    json.field("input", input)?;
    // A read in memory is reported as before task 1.20.
    if let Some(work) = work {
        json.field("work", work)?;
    }
    json.field("memory", memory)?;
    json.end()
}

/// Requirement 13.6: the records of the other entries are reported, and then the
/// command fails when an entry is damaged.
pub(crate) fn fail_when_damaged(summary: &ArchiveSummary, archive: &Path) -> Result<(), L2dError> {
    let mut damaged = summary.damaged();
    match damaged.next() {
        None => Ok(()),
        Some(first) => Err(L2dError::InvalidInput {
            detail: format!(
                "{} of the {} entries of {} are damaged; the first is `{}` ({}). The report \
                 above holds the records of the other entries",
                1 + damaged.count(),
                summary.entries.len(),
                archive.display(),
                shown_text(&first.path, TEXT_CHARS),
                first
                    .skipped
                    .as_ref()
                    .map_or_else(String::new, |skip| shown_text(
                        &skip.to_string(),
                        2 * TEXT_CHARS
                    )),
            ),
        }),
    }
}

/// The reasons an entry is not read, as they are counted in the report, in the order
/// they are shown.
const SKIP_KINDS: [&str; 9] = [
    "no_pattern_match",
    "not_text",
    "nested_archive",
    "encrypted",
    "unsupported_method",
    "directory",
    "link",
    "damaged",
    "not_reached",
];

fn skip_kind(skip: &EntrySkip) -> usize {
    match skip {
        EntrySkip::NoPatternMatch => 0,
        EntrySkip::NotText => 1,
        EntrySkip::NestedArchive => 2,
        EntrySkip::Encrypted => 3,
        EntrySkip::UnsupportedMethod(_) => 4,
        EntrySkip::Directory => 5,
        EntrySkip::Link => 6,
        EntrySkip::Damaged(_) => 7,
        EntrySkip::NotReached => 8,
    }
}

/// How many entries were not read, by reason: the `entries_not_read` object.
#[derive(Debug, Clone, Copy, Default)]
struct SkipCounts([u64; SKIP_KINDS.len()]);

impl Serialize for SkipCounts {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(SKIP_KINDS.iter().zip(&self.0))
    }
}

/// One skipped record of an archive in the JSON output.
#[derive(Serialize)]
struct SkippedInEntry<'a> {
    /// The packed position, as in the logs frame: a number, like the positions of the
    /// other commands. From entry index 8,192 on it is above 2^53 and a reader that
    /// keeps numbers as doubles does not hold it exactly: `entry` and `line` are the
    /// fields to use.
    position: u64,
    entry: &'a str,
    line: u64,
    reason: &'a str,
}

/// The skipped records of an archive: written one by one, never as a list in memory.
struct SkippedRecords<'a>(&'a ArchiveSummary);

impl SkippedRecords<'_> {
    fn iter(&self) -> impl Iterator<Item = SkippedInEntry<'_>> {
        self.0.input.skipped.iter().map(|record| {
            let (entry, line) = self.0.locate(record.position).unwrap_or(("", 0));
            SkippedInEntry {
                position: record.position,
                entry,
                line,
                reason: &record.reason,
            }
        })
    }
}

impl Serialize for SkippedRecords<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

/// What `l2d ingest` reports about an archive (requirements 13.3 and 13.13): the
/// `input` object of the JSON output. It has the fields of an ordinary input and, on
/// top, the entries.
#[derive(Serialize)]
pub(crate) struct ArchiveReport<'a> {
    source_path: String,
    adapter: &'static str,
    /// Records read from all entries, the skipped ones included.
    records: u64,
    distinct_texts: u64,
    /// The SHA-256 of the archive file.
    content_sha256: &'a str,
    label_conflicts: u64,
    skipped_count: u64,
    /// Every skipped record with its entry, its line and the reason.
    skipped: SkippedRecords<'a>,
    /// Distinct texts per label; the records of an archive carry no label.
    labels: LabelCounts,
    group_keys: u64,
    /// Entries in the archive's directory.
    entries_total: u64,
    /// Entries read to their end.
    entries_read: u64,
    /// Entries not read, by reason.
    entries_not_read: SkipCounts,
    /// Every entry in directory order: its path, the records read from it, and the
    /// reason it was not read (`null` when it was).
    entries: &'a [EntryReport],
}

impl<'a> ArchiveReport<'a> {
    pub(crate) fn of(summary: &'a ArchiveSummary) -> Self {
        let mut not_read = SkipCounts::default();
        for skip in summary
            .entries
            .iter()
            .filter_map(|entry| entry.skipped.as_ref())
        {
            not_read.0[skip_kind(skip)] += 1;
        }
        Self {
            source_path: summary.input.source_path.to_string_lossy().into_owned(),
            adapter: "zip",
            records: summary.input.records,
            distinct_texts: summary.input.distinct_texts,
            content_sha256: &summary.input.content_sha256,
            label_conflicts: summary.input.label_conflicts,
            skipped_count: summary.input.skipped.len() as u64,
            skipped: SkippedRecords(summary),
            labels: LabelCounts {
                unknown: summary.input.distinct_texts,
                ..LabelCounts::default()
            },
            group_keys: 0,
            entries_total: summary.entries.len() as u64,
            entries_read: summary.read().count() as u64,
            entries_not_read: not_read,
            entries: &summary.entries,
        }
    }
}

/// How a reason is called in the text output.
fn skip_words(kind: usize) -> &'static str {
    [
        "matching no entry pattern",
        "not text",
        "nested archives",
        "encrypted",
        "with an unsupported compression method",
        "directories",
        "symbolic links",
        "damaged",
        "not reached",
    ][kind]
}

/// The text form of an archive's report, and the memory report.
fn archive_text(
    out: impl Write,
    report: &ArchiveReport<'_>,
    work: Option<&WorkReport>,
    memory: &MemoryReport,
    limit: u64,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(out);
    archive_report_text(&mut out, report, limit)?;
    // A read that stayed in memory is reported as before task 1.20.
    if let Some(work) = work {
        work.deduplication
            .write_text(&mut out, work.disk_free_bytes, work.disk_estimated_bytes)?;
    }
    memory.write_text(&mut out)?;
    out.flush()
}

/// The text form of an archive's report; at most `limit` entries read, entries not
/// read and skipped records are listed. Entries that match no pattern and directories
/// are counted, not listed.
pub(crate) fn archive_report_text(
    out: &mut dyn Write,
    report: &ArchiveReport<'_>,
    limit: u64,
) -> io::Result<()> {
    let listed = limit as usize;
    let more = |out: &mut dyn Write, count: u64| match count.saturating_sub(limit) {
        0 => Ok(()),
        rest => writeln!(out, "    ... {rest} more"),
    };
    let name = |path: &str| shown_text(path, 2 * TEXT_CHARS);

    writeln!(out, "input: {}", report.source_path)?;
    writeln!(out, "  adapter: {}", report.adapter)?;
    writeln!(
        out,
        "  records read: {} ({} skipped)",
        report.records, report.skipped_count
    )?;
    writeln!(
        out,
        "  distinct texts: {} ({} good, {} bad, {} unknown)",
        report.distinct_texts, report.labels.good, report.labels.bad, report.labels.unknown
    )?;
    writeln!(out, "  content SHA-256: {}", report.content_sha256)?;
    let not_read = report.entries_total - report.entries_read;
    writeln!(
        out,
        "  entries: {} in the archive, {} read, {not_read} not read",
        report.entries_total, report.entries_read
    )?;
    if not_read > 0 {
        let counts: Vec<String> = (report.entries_not_read.0.iter().enumerate())
            .filter(|(_, count)| **count > 0)
            .map(|(kind, count)| format!("{count} {}", skip_words(kind)))
            .collect();
        writeln!(out, "    not read: {}", counts.join(", "))?;
    }
    if report.entries_read > 0 {
        writeln!(out, "  entries read: {}", report.entries_read)?;
        let read = report
            .entries
            .iter()
            .filter(|entry| entry.skipped.is_none());
        for entry in read.take(listed) {
            writeln!(out, "    {}: {} records", name(&entry.path), entry.records)?;
        }
        more(&mut *out, report.entries_read)?;
    }
    let reported = |entry: &&EntryReport| {
        !matches!(
            entry.skipped,
            None | Some(EntrySkip::NoPatternMatch | EntrySkip::Directory)
        )
    };
    let reported_count = report.entries.iter().filter(reported).count() as u64;
    if reported_count > 0 {
        writeln!(out, "  entries not read: {reported_count}")?;
        for entry in report.entries.iter().filter(reported).take(listed) {
            let reason = entry.skipped.as_ref().map(ToString::to_string);
            write!(
                out,
                "    {}: {}",
                name(&entry.path),
                shown_text(&reason.unwrap_or_default(), 2 * TEXT_CHARS)
            )?;
            match entry.records {
                0 => writeln!(out)?,
                records => writeln!(out, " ({records} records were read before that)")?,
            }
        }
        more(&mut *out, reported_count)?;
    }
    if report.skipped_count > 0 {
        writeln!(out, "  skipped records: {}", report.skipped_count)?;
        for record in report.skipped.iter().take(listed) {
            writeln!(
                out,
                "    {} line {}: {}",
                name(record.entry),
                record.line,
                shown_text(record.reason, 2 * TEXT_CHARS)
            )?;
        }
        more(&mut *out, report.skipped_count)?;
    }
    Ok(())
}
