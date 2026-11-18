//! Reopen the source recorded in a MarkSet, including its exact archive entry selection.
use super::common::{self, AdapterChoice, Context, Input, InputArgs};
use clap::Args;
use l2d_baseline::markscore::archive_host;
use l2d_baseline::redact::RedactionInput;
use l2d_core::{
    ArchivePosition, L2dError, LogsFrame, MarkSet, MarkedSource, MemoryBudget, ProjectPaths,
    ZipInput, ZipInputSettings,
};
use l2d_decide::close_enough::MarkHistory;
use l2d_store::Store;
use std::path::PathBuf;

pub(super) fn invalid(detail: impl Into<String>) -> L2dError {
    L2dError::InvalidInput {
        detail: detail.into(),
    }
}
pub(super) fn hold(b: &mut MemoryBudget, bytes: u64, what: &str) -> Result<(), L2dError> {
    b.require_whole(bytes, what)?;
    b.hold(b.held_bytes().saturating_add(bytes));
    Ok(())
}

/// JSON trees own strings and a node per value. Size the typed report by streaming
/// to a counting writer before constructing its presentation copy.
pub(super) fn report_value<T: serde::Serialize>(
    report: &T,
    b: &mut MemoryBudget,
) -> Result<serde_json::Value, L2dError> {
    struct Count(u64);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len() as u64);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut size = Count(0);
    serde_json::to_writer(&mut size, report).map_err(|e| invalid(e.to_string()))?;
    hold(
        b,
        size.0.saturating_mul(32).saturating_add(65536),
        "typed report and retained JSON output",
    )?;
    serde_json::to_value(report).map_err(|e| invalid(e.to_string()))
}
/// File adapters account from zero as they read. Give the reader only the remaining
/// headroom and carry its measured peak and retained bytes back to the shared budget.
pub(super) fn read_beside(
    ctx: &Context,
    paths: &ProjectPaths,
    args: &InputArgs,
    b: &mut MemoryBudget,
) -> Result<Input, L2dError> {
    use l2d_core::memory::PROCESS_BASELINE_BYTES;
    b.require_whole(1, "reading a file beside retained artifacts")?;
    let settings = l2d_core::MemorySettings {
        memory_limit_bytes: Some(b.left_bytes().saturating_add(PROCESS_BASELINE_BYTES)),
        reserve_share: ctx.shares()?.reserve,
        ..ctx.memory.clone()
    };
    let mut reader = MemoryBudget::observe(&settings)?;
    let input = common::read_input(paths, args, &mut reader);
    let report = reader.usage().report();
    b.count_peak(
        report
            .counted_peak_bytes
            .saturating_sub(PROCESS_BASELINE_BYTES),
    );
    if report.spilled {
        b.note_spill();
    }
    let input = input?;
    let retained = reader
        .held_bytes()
        .max(input.logs.frame().estimated_size() as u64);
    hold(b, retained, "retained input frame and summary")?;
    Ok(input)
}
#[derive(Debug, Clone, Args)]
pub struct HistoryArgs {
    /// Adapter used to mark the historical file
    #[arg(long,value_enum,default_value_t=AdapterChoice::Auto)]
    pub history_adapter: AdapterChoice,
    #[arg(long)]
    pub history_text_column: Option<String>,
    #[arg(long)]
    pub history_label_column: Option<String>,
    #[arg(long, conflicts_with = "history_label_column")]
    pub history_no_labels: bool,
    #[arg(long)]
    pub history_group_column: Option<String>,
    /// Use the original archive reader's maximum record length when marks used an override
    #[arg(long,value_parser=common::record_bytes)]
    pub history_max_record_bytes: Option<u64>,
}
impl HistoryArgs {
    pub fn supplied(&self) -> bool {
        self.history_adapter != AdapterChoice::Auto
            || self.history_text_column.is_some()
            || self.history_label_column.is_some()
            || self.history_no_labels
            || self.history_group_column.is_some()
            || self.history_max_record_bytes.is_some()
    }
}
pub(super) enum Source {
    Store(Store),
    File(Input),
    Archive(PathBuf, ZipInputSettings),
}
impl Source {
    pub fn open(
        ctx: &Context,
        args: &HistoryArgs,
        marks: &MarkSet,
        b: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        let paths = ctx.paths()?;
        if args.history_max_record_bytes.is_some()
            && !marks.input().is_some_and(|i| i.archive_entries.is_some())
        {
            return Err(invalid("--history-max-record-bytes requires archive marks"));
        }
        match marks.source() {
            MarkedSource::Store(s) => Ok(Self::Store(Store::open(&paths, &s.name)?)),
            MarkedSource::Input(i) => {
                let path = i.saved_location()?.to_path_buf();
                if let Some(entries) = &i.archive_entries {
                    let settings = ZipInputSettings {
                        entries: entries.clone(),
                        max_record_bytes: args
                            .history_max_record_bytes
                            .unwrap_or_else(|| ZipInputSettings::default().max_record_bytes),
                        ..Default::default()
                    };
                    i.require_interpretation(&l2d_core::marks::InputInterpretation::archive(
                        &settings,
                    ))?;
                    Ok(Self::Archive(path, settings))
                } else {
                    let options = InputArgs {
                        input: Some(path),
                        adapter: args.history_adapter,
                        text_column: args.history_text_column.clone(),
                        label_column: args.history_label_column.clone(),
                        no_labels: args.history_no_labels,
                        group_column: args.history_group_column.clone(),
                    };
                    let file = read_beside(ctx, &paths, &options, b)?;
                    marks.require_file(&file.summary, &file.logs, b)?;
                    Ok(Self::File(file))
                }
            }
        }
    }
    pub fn current(&self, marks: &MarkSet, b: &mut MemoryBudget) -> Result<(), L2dError> {
        if let Some(store) = self.store() {
            return marks.require_store(store, b);
        }
        let source = marks
            .input()
            .ok_or_else(|| invalid("expected an input MarkSet"))?;
        let path = match self {
            Self::File(f) => &f.summary.source_path,
            Self::Archive(path, _) => path,
            Self::Store(_) => unreachable!(),
        };
        b.require_whole(1024 * 1024, "verifying original history before publication")?;
        if l2d_encoder::runtime::sha256_of(path)? != source.content_sha256 {
            return Err(L2dError::Refused {
                reason: "original history changed before publication".into(),
            });
        }
        Ok(())
    }
    pub fn redaction(&self) -> RedactionInput<'_> {
        match self {
            Self::Store(s) => RedactionInput::Store(s),
            Self::File(f) => RedactionInput::File {
                logs: &f.logs,
                summary: &f.summary,
            },
            Self::Archive(path, settings) => RedactionInput::Archive { path, settings },
        }
    }
    pub fn store(&self) -> Option<&Store> {
        match self {
            Self::Store(s) => Some(s),
            _ => None,
        }
    }
    pub fn history(
        &self,
        marks: &MarkSet,
        other: Option<&MarkSet>,
        b: &mut MemoryBudget,
    ) -> Result<MarkHistory, L2dError> {
        match self {
            Self::Store(s) => MarkHistory::from_store(s, marks, other, b),
            Self::File(f) => MarkHistory::from_input(&f.logs, &f.summary, marks, other, b),
            Self::Archive(path, settings) => {
                let prior = b.held_bytes();
                let input = read_archive(path, settings, Some(marks), b)?;
                let history = MarkHistory::from_input_selection(
                    &input.logs,
                    &input.summary,
                    Some(settings),
                    marks,
                    other,
                    b,
                )?;
                b.hold(prior.saturating_add(history.retained_bytes()));
                Ok(history)
            }
        }
    }
}
/// Compare a sizing pass with a complete second read, accumulating the actual
/// occurrences of each original across all selected archive entries.
pub(super) fn read_archive(
    path: &std::path::Path,
    settings: &ZipInputSettings,
    marks: Option<&MarkSet>,
    b: &mut MemoryBudget,
) -> Result<Input, L2dError> {
    let prior = b.held_bytes();
    let mut zip = ZipInput::open(path, settings, b)?;
    zip.hold_beside(prior);
    // Measure actual delivered rows, including nested compression, before allocating
    // the whole-history frame. A second read must reproduce the same archive.
    fn row_bytes(text: &str, group: Option<&str>) -> u64 {
        1024u64.saturating_add(
            (text.len() as u64)
                .saturating_add(group.map_or(0, |g| g.len() as u64))
                .saturating_mul(8),
        )
    }
    let mut room = 1024 * 1024u64;
    let mut batches = zip.batches(b);
    while let Some(batch) = batches.next() {
        let batch = batch?;
        for row in batch.records() {
            let pos = ArchivePosition::unpack(row.first_position);
            let group = batches.entry_path(pos.entry).and_then(archive_host);
            room = room.saturating_add(row_bytes(&row.text, group));
        }
    }
    let summary = zip.finish();
    validate_archive(marks, &summary, &settings.entries)?;
    room = summary.input.skipped.iter().fold(room, |sum, row| {
        sum.saturating_add((row.reason.len() as u64 + 128).saturating_mul(8))
    });
    let expected = summary.input.content_sha256.clone();
    drop(summary);
    b.hold(prior);
    hold(
        b,
        room,
        "capturing original archive history and endpoint metadata",
    )?;
    let reserved = b.held_bytes();
    let mut zip = ZipInput::open(path, settings, b)?;
    if expected != zip.content_sha256() {
        return Err(invalid("archive changed before history capture"));
    }
    zip.hold_beside(reserved);
    let mut captured = 1024 * 1024u64;
    let mut rows = Vec::new();
    let mut batches = zip.batches(b);
    while let Some(batch) = batches.next() {
        for mut row in batch?.records() {
            let pos = ArchivePosition::unpack(row.first_position);
            row.group = batches
                .entry_path(pos.entry)
                .and_then(archive_host)
                .map(str::to_owned);
            captured = captured.saturating_add(row_bytes(&row.text, row.group.as_deref()));
            if captured > room {
                return Err(L2dError::Refused {
                    reason: "archive rows exceeded the admitted sizing pass".into(),
                });
            }
            rows.push(row);
        }
    }
    for row in &mut rows {
        row.occurrences = u32::try_from(zip.occurrences(&row.text))
            .map_err(|_| invalid("archive occurrence count exceeds u32"))?;
    }
    let plan = zip
        .batch_plan()
        .cloned()
        .ok_or_else(|| invalid("archive read completed without a batch plan"))?;
    let summary = zip.finish();
    validate_archive(marks, &summary, &settings.entries)?;
    b.hold(reserved);
    b.require_whole(1024 * 1024, "rechecking archive snapshot after capture")?;
    if l2d_encoder::runtime::sha256_of(path)? != expected {
        return Err(L2dError::Refused {
            reason: "archive changed during capture".into(),
        });
    }
    let logs = LogsFrame::from_records(&rows)?;
    Ok(Input {
        logs,
        summary: summary.input,
        plan,
        adapter: "zip",
    })
}
pub(super) fn query_input(
    ctx: &Context,
    paths: &ProjectPaths,
    args: &InputArgs,
    archive: &common::ArchiveArgs,
    b: &mut MemoryBudget,
) -> Result<Input, L2dError> {
    if let Some(path) = common::archive_input(paths, args)? {
        read_archive(
            &path,
            &archive.settings(common::AtOnceDefault::One),
            None,
            b,
        )
    } else {
        archive.refuse_for_other_input(&[])?;
        read_beside(ctx, paths, args, b)
    }
}
fn validate_archive(
    marks: Option<&MarkSet>,
    summary: &l2d_core::ArchiveSummary,
    entries: &[String],
) -> Result<(), L2dError> {
    if let Some(marks) = marks {
        marks.require_input(&summary.input, Some(entries))?;
    }
    if summary.damaged().next().is_some() {
        return Err(invalid(
            "damaged archive cannot provide complete MarkSet history",
        ));
    }
    Ok(())
}
pub(super) fn load_marks(
    paths: &ProjectPaths,
    name: &str,
    b: &mut MemoryBudget,
) -> Result<MarkSet, L2dError> {
    let marks = MarkSet::load(paths, name, b)?;
    let bytes = std::fs::metadata(paths.mark_set_path(name))
        .map_err(|e| invalid(e.to_string()))?
        .len();
    hold(
        b,
        bytes.saturating_mul(8).saturating_add(65536),
        "retained MarkSet, copy and output buffers",
    )?;
    Ok(marks)
}
