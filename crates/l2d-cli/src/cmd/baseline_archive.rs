//! `l2d baseline score` on a zip archive (requirements 13.11 and 13.12, task 1.17).
//!
//! The archive is read entry by entry in batches ([`ZipInput`]); every batch's new
//! texts are scored and written at once as a Parquet part, so neither the logs nor the
//! ratios of an archive are in memory as a whole. Standard output shows the first rows
//! and says where the parts are.
//!
//! # The work directory
//!
//! `--work-dir DIR`, or `<project>/work/baseline-score-<UTC time>/` when it is not
//! given. The directory must be empty or absent; the run creates it and writes nothing
//! outside it. Before anything is read the free space of its volume is compared with
//! an estimate of what the run needs, and the run is refused when it is less:
//!
//! - for the parts, the bytes the selected entries take in the archive (text
//!   compressed by zstd in place of deflate), plus 16 bytes per 64 bytes of unpacked
//!   text for the numbers of a row;
//! - for deduplication on disk, the unpacked bytes of the selected entries: the texts
//!   that do not fit the set in memory are written there once per batch they occur in.
//!   A gzip entry declares its compressed size only, so this is an estimate too.
//!
//! # What is written
//!
//! - `scores-NNNNNN.parquet`, one part per batch that has new texts: `row` (from 0, in
//!   order of first occurrence), `entry` (the index of the entry in the archive's
//!   directory, which is its index in the manifest's `input.entries`), `line` (1-based,
//!   in that entry), `text`, `ratio`, `no_features`. One row per distinct log.
//! - `occurrences-NNNNNN.parquet`, written after the last entry: `row` and
//!   `occurrences`, how often the log of that row occurred in the archive. The count is
//!   only known at the end, which is why it is not a column of the scores.
//! - `manifest.json`, written last: the scorer, the archive with its hash, every entry
//!   with its record count or the reason it was not read (requirement 13.13), and per
//!   table its columns, rows, parts (file, rows, bytes, SHA-256) and `rows_sha256`.
//!   A directory without a manifest holds a run that did not end.
//!
//! The manifest and the parts hold no time and no host figure: the same archive,
//! scorer, options and batch size give the same bytes. Another batch size (or a memory
//! limit that moves the deduplication to disk) cuts the same rows into other parts:
//! the files differ, `rows_sha256` of each table does not.
//!
//! # Memory
//!
//! Held beside the archive for the whole run and counted by the archive's reader: the
//! loaded scorer, hashing a text of the maximum record length, and printing one. A
//! batch's frame, its scores and the encoding of its part live between two batches,
//! in the memory the batch itself was allowed (six times its text and 500 bytes a row),
//! which is free by then.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use l2d_baseline::scorer::{SCORER_KIND, SCORES_NO_FEATURES, SCORES_RATIO};
use l2d_core::parts::{PartTable, PartWriter, WorkDir, write_manifest};
use l2d_core::polars::prelude::{Column, DataFrame, PolarsError};
use l2d_core::{ArchivePosition, DiskDedup, EntryReport, L2dError, LogsFrame, ZipInput};
use serde::Serialize;

use super::baseline::ScoreArgs;
use super::common::{
    Context, DEFAULT_TEXT_LIMIT, MemoryReport, Operation, ScorerReport, TEXT_CHARS, find_model,
    load_scorer, output_bytes, required,
};
use super::ingest::{ArchiveReport, archive_report_text, fail_when_damaged};
use crate::output::{JsonOut, human_bytes, output_failed, shown_text};

/// The numbers of a row in a part, per this many bytes of unpacked text: an entry of
/// short lines has more rows than its packed size suggests.
const NUMBERS_PER_TEXT_BYTES: u64 = 4;
/// The least the parts are estimated at.
const SMALLEST_PARTS_BYTES: u64 = 1024 * 1024;
const SCORES_TABLE: &str = "scores";
const OCCURRENCES_TABLE: &str = "occurrences";

fn polars(error: PolarsError) -> L2dError {
    L2dError::Runtime {
        detail: format!("writing the scores failed: {error}"),
    }
}

/// `yyyymmddThhmmssZ` of `seconds` since 1970-01-01 UTC.
fn utc_stamp(seconds: u64) -> String {
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Days to a civil date (Howard Hinnant's algorithm), for the proleptic Gregorian
    // calendar.
    let shifted = days as i64 + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_shifted = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_shifted + 2) / 5 + 1;
    let month = if month_shifted < 10 {
        month_shifted + 3
    } else {
        month_shifted - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

/// The directory of this run: the analyst's, or one named after the command and the
/// time below the project's work directory.
fn run_directory(context: &Context, given: Option<&Path>) -> Result<PathBuf, L2dError> {
    if let Some(given) = given {
        return Ok(given.to_path_buf());
    }
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| L2dError::Runtime {
            detail: format!("the clock of the host is before 1970: {error}"),
        })?
        .as_secs();
    Ok(context
        .paths()?
        .work_dir()
        .join(format!("baseline-score-{}", utc_stamp(seconds))))
}

/// One scored log of an archive: an element of `scores` in the JSON output.
#[derive(Debug, Serialize)]
struct ArchiveScoreRow<'a> {
    /// The row of the log in the parts, from 0, in order of first occurrence.
    row: u64,
    /// The path of the entry the log was first seen in, and the line in it.
    entry: &'a str,
    line: u64,
    text: &'a str,
    scorer: &'static str,
    ratio: f32,
    abstained: bool,
    no_features: bool,
}

/// What a table of the run is on disk: an element of `parts.tables` in the output.
#[derive(Debug, Serialize)]
struct TableReport<'a> {
    name: &'a str,
    rows: u64,
    /// How many part files.
    parts: u64,
    bytes: u64,
    rows_sha256: &'a str,
}

/// Whether the deduplication went on on disk: `parts.deduplication` in the output.
#[derive(Debug, Serialize)]
struct DedupReport {
    on_disk: bool,
    /// Distinct texts the set in memory held.
    set_texts: u64,
    /// Rows written to disk for deduplication, and their bytes.
    spilled_records: u64,
    spilled_bytes: u64,
    /// Distinct texts among them.
    spilled_distinct: u64,
}

impl DedupReport {
    fn of(dedup: Option<DiskDedup>, distinct: u64) -> Self {
        match dedup {
            Some(dedup) => Self {
                on_disk: true,
                set_texts: dedup.set_texts,
                spilled_records: dedup.spilled_records,
                spilled_bytes: dedup.spilled_bytes,
                spilled_distinct: dedup.spilled_distinct,
            },
            None => Self {
                on_disk: false,
                set_texts: distinct,
                spilled_records: 0,
                spilled_bytes: 0,
                spilled_distinct: 0,
            },
        }
    }
}

/// Where the parts are (requirement 13.11): the `parts` object of the output.
#[derive(Debug, Serialize)]
struct PartsReport<'a> {
    directory: String,
    manifest: String,
    rows: u64,
    bytes: u64,
    tables: Vec<TableReport<'a>>,
    deduplication: DedupReport,
    /// Free disk space of the work directory's volume before the run, and what the
    /// run was estimated to need.
    disk_free_bytes: u64,
    disk_estimated_bytes: u64,
}

/// The archive as the manifest records it (requirement 13.13).
#[derive(Serialize)]
struct ManifestInput<'a> {
    source_path: &'a str,
    content_sha256: &'a str,
    records: u64,
    distinct_texts: u64,
    skipped_count: u64,
    entries_total: u64,
    entries_read: u64,
    /// Every entry in directory order; a row's `entry` is an index into this list.
    entries: &'a [EntryReport],
}

#[derive(Serialize)]
struct ManifestScorer<'a> {
    kind: &'a str,
    name: &'a str,
    revision: &'a str,
}

#[derive(Serialize)]
struct ManifestSettings<'a> {
    entries: &'a [String],
    max_record_bytes: u64,
}

/// `manifest.json` of a run.
#[derive(Serialize)]
struct Manifest<'a> {
    format: &'static str,
    version: u32,
    command: &'static str,
    scorer: ManifestScorer<'a>,
    settings: ManifestSettings<'a>,
    input: ManifestInput<'a>,
    /// The rows of the scores table: the distinct logs of the archive.
    rows: u64,
    tables: [&'a PartTable; 2],
}

/// Where the rows of the command go.
enum ScoreOut<W: Write> {
    Json(JsonOut<W>),
    Text(W),
}

/// Scores the archive at `archive` into result parts.
pub fn score(context: &Context, args: &ScoreArgs, archive: &Path) -> Result<(), L2dError> {
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
    let scorer = load_scorer(&paths, &directory, &mut budget)?;
    let report = ScorerReport::of(&scorer, &directory);

    let settings = args.archive.settings();
    let mut input = ZipInput::open(archive, &settings, &budget)?;

    // What the run holds beside the archive: the scorer, hashing the longest text an
    // entry can give, and printing it.
    let fixed = scorer
        .scoring_bytes(settings.max_record_bytes)
        .saturating_add(output_bytes(settings.max_record_bytes));
    budget.require_whole(fixed, "scoring and printing the ratios")?;
    input.hold_beside(budget.held_bytes().saturating_add(fixed));

    // The work directory, and the room on its disk (module documentation).
    let selected = input.selected_bytes();
    let work = WorkDir::create(&run_directory(context, args.work_dir.as_deref())?)?;
    let estimate = [
        (
            "the result parts",
            SMALLEST_PARTS_BYTES
                .saturating_add(selected.packed)
                .saturating_add(selected.unpacked / NUMBERS_PER_TEXT_BYTES),
        ),
        ("deduplication on disk", selected.unpacked),
    ];
    let disk_estimated: u64 = estimate.iter().map(|(_, bytes)| bytes).sum();
    let disk_free = match work.require_free(&estimate) {
        Ok(free) => free,
        Err(error) => {
            // Nothing was written: the directory this run named itself does not stay.
            if args.work_dir.is_none() {
                let _ = std::fs::remove_dir(work.path());
            }
            return Err(error);
        }
    };
    input.spill_into(&work.path().join("dedup"), None)?;

    let limit = args.limit.unwrap_or(DEFAULT_TEXT_LIMIT);
    let stdout = io::BufWriter::new(io::stdout().lock());
    let mut out = if context.json {
        let mut json = JsonOut::begin(stdout).map_err(output_failed)?;
        (|| {
            json.field("command", "baseline score")?;
            json.field("scorer", &report)?;
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
            writeln!(
                text,
                "ratios of the distinct texts of {}, in input order (0 looks good, 1 looks \
                 bad); all of them go to the result parts",
                archive.display()
            )?;
            writeln!(text, "   ratio  text")
        })()
        .map_err(output_failed)?;
        ScoreOut::Text(text)
    };

    let mut scores_out = PartWriter::new(&work, SCORES_TABLE)?;
    let mut rows: u64 = 0;
    let mut no_features: u64 = 0;
    let mut batches = input.batches(&mut budget);
    while let Some(batch) = batches.next() {
        let logs: LogsFrame = batch?;
        let scores = scorer.score(&logs)?;
        let frame = logs.frame();
        let size = frame.height();
        let positions = frame
            .column(LogsFrame::FIRST_POSITION)
            .and_then(Column::u64)
            .map_err(polars)?;
        let (mut entries, mut lines) = (Vec::with_capacity(size), Vec::with_capacity(size));
        for position in positions.iter() {
            let at = ArchivePosition::unpack(position.unwrap_or(0));
            entries.push(at.entry);
            lines.push(at.line);
        }
        let ratio = scores.column(SCORES_RATIO).map_err(polars)?.clone();
        let flags = scores.column(SCORES_NO_FEATURES).map_err(polars)?.clone();
        no_features += u64::from(flags.bool().map_err(polars)?.sum().unwrap_or(0));
        if rows < limit {
            let texts = frame
                .column(LogsFrame::TEXT)
                .and_then(Column::str)
                .map_err(polars)?;
            let ratios = ratio.f32().map_err(polars)?;
            let flagged = flags.bool().map_err(polars)?;
            for index in 0..size.min((limit - rows) as usize) {
                let text = texts.get(index).unwrap_or_default();
                let ratio = ratios.get(index).unwrap_or(f32::NAN);
                let flagged = flagged.get(index).unwrap_or(false);
                let written = match &mut out {
                    ScoreOut::Json(json) => json.element(&ArchiveScoreRow {
                        row: rows + index as u64,
                        entry: batches.entry_path(entries[index]).unwrap_or_default(),
                        line: lines[index],
                        text,
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
        }
        let mut part = DataFrame::new(vec![
            Column::new(
                "row".into(),
                (rows..rows + size as u64).collect::<Vec<u64>>(),
            ),
            Column::new("entry".into(), entries),
            Column::new("line".into(), lines),
            frame.column(LogsFrame::TEXT).map_err(polars)?.clone(),
            ratio,
            flags,
        ])
        .map_err(polars)?;
        scores_out.write(&mut part)?;
        rows += size as u64;
    }
    drop(batches);

    // The occurrence counts over the archive, known now.
    let mut occurrences_out = PartWriter::new(&work, OCCURRENCES_TABLE)?;
    let mut counted: u64 = 0;
    for chunk in input.occurrence_counts() {
        let counts = chunk?;
        let size = counts.len() as u64;
        let mut part = DataFrame::new(vec![
            Column::new(
                "row".into(),
                (counted..counted + size).collect::<Vec<u64>>(),
            ),
            Column::new("occurrences".into(), counts),
        ])
        .map_err(polars)?;
        occurrences_out.write(&mut part)?;
        counted += size;
    }
    if counted != rows {
        return Err(L2dError::Runtime {
            detail: format!("{rows} rows were scored, and {counted} occurrence counts were read"),
        });
    }

    let plan = input
        .batch_plan()
        .cloned()
        .ok_or_else(|| L2dError::Runtime {
            detail: "the archive was read without a batch plan".into(),
        })?;
    let dedup = input.disk_dedup();
    let summary = input.finish();
    let (scores_table, occurrences_table) = (scores_out.finish(), occurrences_out.finish());
    let archive_report = ArchiveReport::of(&summary);
    let source_path = summary.input.source_path.to_string_lossy();
    let manifest_path = write_manifest(
        &work,
        &Manifest {
            format: "l2d-result-parts",
            version: 1,
            command: "baseline score",
            scorer: ManifestScorer {
                kind: report.kind,
                name: &report.name,
                revision: &report.revision,
            },
            settings: ManifestSettings {
                entries: &settings.entries,
                max_record_bytes: settings.max_record_bytes,
            },
            input: ManifestInput {
                source_path: &source_path,
                content_sha256: &summary.input.content_sha256,
                records: summary.input.records,
                distinct_texts: summary.input.distinct_texts,
                skipped_count: summary.input.skipped.len() as u64,
                entries_total: summary.entries.len() as u64,
                entries_read: summary.read().count() as u64,
                entries: &summary.entries,
            },
            rows,
            tables: [&scores_table, &occurrences_table],
        },
    )?;

    let tables = vec![
        table_report(&scores_table),
        table_report(&occurrences_table),
    ];
    let parts = PartsReport {
        directory: work.path().to_string_lossy().into_owned(),
        manifest: manifest_path.to_string_lossy().into_owned(),
        rows,
        bytes: scores_table.bytes() + occurrences_table.bytes(),
        tables,
        deduplication: DedupReport::of(dedup, summary.input.distinct_texts),
        disk_free_bytes: disk_free,
        disk_estimated_bytes: disk_estimated,
    };
    let memory = context.memory_report(
        &budget,
        vec![Operation::batches("read", summary.input.records, plan)],
    );

    let omitted = rows.saturating_sub(limit);
    let written = match &mut out {
        ScoreOut::Json(json) => (|| {
            json.end_list()?;
            json.field("scored", &rows)?;
            json.field("rows_omitted", &omitted)?;
            json.field("no_features", &no_features)?;
            json.field("input", &archive_report)?;
            json.field("parts", &parts)?;
            json.field("memory", &memory)
        })(),
        ScoreOut::Text(out) => text_report(
            out,
            omitted,
            no_features,
            &archive_report,
            &parts,
            &memory,
            limit,
        ),
    };
    written.map_err(output_failed)?;
    match out {
        ScoreOut::Json(json) => json.end(),
        ScoreOut::Text(mut out) => out.flush(),
    }
    .map_err(output_failed)?;
    fail_when_damaged(&summary, archive)
}

fn table_report(table: &PartTable) -> TableReport<'_> {
    TableReport {
        name: &table.name,
        rows: table.rows,
        parts: table.parts.len() as u64,
        bytes: table.bytes(),
        rows_sha256: &table.rows_sha256,
    }
}

fn text_report(
    out: &mut impl Write,
    omitted: u64,
    no_features: u64,
    archive: &ArchiveReport<'_>,
    parts: &PartsReport<'_>,
    memory: &MemoryReport,
    limit: u64,
) -> io::Result<()> {
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
    archive_report_text(out, archive, limit)?;
    writeln!(out, "result parts: {}", parts.directory)?;
    writeln!(out, "  manifest: {}", parts.manifest)?;
    for table in &parts.tables {
        writeln!(
            out,
            "  {}: {} rows in {} {}, {}",
            table.name,
            table.rows,
            table.parts,
            if table.parts == 1 { "part" } else { "parts" },
            human_bytes(table.bytes)
        )?;
    }
    let dedup = &parts.deduplication;
    if dedup.on_disk {
        writeln!(
            out,
            "  deduplication: on disk after {} distinct texts in memory ({} rows, {}, written \
             and removed again; {} further distinct texts)",
            dedup.set_texts,
            dedup.spilled_records,
            human_bytes(dedup.spilled_bytes),
            dedup.spilled_distinct
        )?;
    } else {
        writeln!(out, "  deduplication: in memory")?;
    }
    writeln!(
        out,
        "  disk: {} was free, the run was estimated at {}",
        human_bytes(parts.disk_free_bytes),
        human_bytes(parts.disk_estimated_bytes)
    )?;
    memory.write_text(out)
}

#[cfg(test)]
mod tests {
    use super::utc_stamp;

    #[test]
    fn a_time_is_stamped_as_a_utc_date_and_time() {
        assert_eq!(utc_stamp(0), "19700101T000000Z");
        assert_eq!(utc_stamp(951_782_399), "20000228T235959Z");
        assert_eq!(utc_stamp(951_782_400), "20000229T000000Z");
        assert_eq!(utc_stamp(951_868_800), "20000301T000000Z");
        assert_eq!(utc_stamp(1_794_652_496), "20261114T103456Z");
        assert_eq!(utc_stamp(4_102_444_799), "20991231T235959Z");
    }
}
