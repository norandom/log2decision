//! One log per line.

use polars::prelude::{Column, DataType};

use crate::error::L2dError;
use crate::input::{
    Accepted, Dedup, InputAdapter, PARSE_COST, Source, batch_end, batch_step, csv_field_name,
    parse_csv_records, row_digest,
};
use crate::memory::{BatchPlan, Batcher, MemoryBudget};
use crate::types::{InputSummary, Label, LogsFrame};

/// The field separator given to Polars for line-based input. The byte `0xFF` cannot
/// occur in valid UTF-8, and only valid UTF-8 lines are accepted, so every accepted
/// line is one field.
const NO_SEPARATOR: u8 = 0xFF;

/// Reads a plain text file with one log per line.
///
/// Every line is a record whose position is its line number. The text is the line
/// without its terminator, otherwise unchanged. Rows carry the label `unknown` and no
/// group. A line that is blank or not valid UTF-8 is skipped and reported.
///
/// The lines are read in batches; the accepted lines of a batch are parsed by Polars as
/// single-field records without quoting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinesAdapter;

impl InputAdapter for LinesAdapter {
    fn read_from(
        &self,
        source: Source,
        budget: &mut MemoryBudget,
    ) -> Result<(LogsFrame, InputSummary, BatchPlan), L2dError> {
        let path = source.path().to_path_buf();
        let mut batcher = Batcher::new(budget, source.size(), PARSE_COST);
        let mut lines = source.lines(batcher.record_limit());
        let mut dedup = Dedup::new();
        let mut accepted = Accepted::new();
        let mut process = push_lines;
        let mut line = Vec::new();
        let mut position = 0u64;
        while let Some(source_bytes) = lines.next_line(&mut line)? {
            position += 1;
            match std::str::from_utf8(&line) {
                Ok(text) if text.trim().is_empty() => dedup.skip(position, "empty line"),
                Ok(text) => {
                    accepted.record(
                        position,
                        row_digest(text, Label::Unknown, None),
                        Label::Unknown,
                    );
                    accepted.push_raw(&line);
                    // Polars removes one carriage return before the line feed. A text
                    // that itself ends in a carriage return gets a second one to lose.
                    accepted.push_raw(if line.ends_with(b"\r") {
                        b"\r\n"
                    } else {
                        b"\n"
                    });
                }
                Err(_) => dedup.skip(position, "line is not valid UTF-8"),
            }
            batch_step(
                &mut batcher,
                source_bytes,
                &mut dedup,
                &mut accepted,
                &mut process,
            )?;
        }
        batch_end(&mut batcher, &mut dedup, accepted, &mut process)?;
        let plan = batcher.into_plan()?;
        let (logs, summary) = dedup.finish_hashed(&path, lines.into_source().finish()?)?;
        Ok((logs, summary, plan))
    }
}

/// Has Polars parse the accepted lines of one batch and pushes them into the
/// deduplication.
fn push_lines(dedup: &mut Dedup, accepted: Accepted) -> Result<(), L2dError> {
    if accepted.is_empty() {
        return Ok(());
    }
    let parsed = parse_csv_records(accepted.raw(), 1, &[0], NO_SEPARATOR, None)?;
    let text = parsed
        .column(&csv_field_name(0))
        .map_err(|error| L2dError::Runtime {
            detail: format!("reading the parsed lines failed: {error}"),
        })?
        .clone();
    let rows = text.len();
    let label = Column::new("label".into(), vec![Label::Unknown.as_str(); rows]);
    let group = Column::full_null("group".into(), rows, &DataType::String);
    dedup.push(accepted, text, label, group)
}
