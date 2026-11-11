//! One log per line.

use std::path::Path;

use polars::prelude::{Column, DataType};

use crate::error::L2dError;
use crate::input::{
    Accepted, Dedup, InputAdapter, csv_field_name, parse_csv_records, read_source, row_digest,
    split_lines, strip_bom,
};
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
/// The accepted lines are parsed by Polars as single-field records without quoting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinesAdapter;

impl InputAdapter for LinesAdapter {
    fn read(&self, path: &Path) -> Result<(LogsFrame, InputSummary), L2dError> {
        let bytes = read_source(path)?;
        let mut dedup = Dedup::new();
        let mut accepted = Accepted::new();
        for (index, line) in split_lines(strip_bom(&bytes)).enumerate() {
            let position = index as u64 + 1;
            match std::str::from_utf8(line) {
                Ok(text) if text.trim().is_empty() => dedup.skip(position, "empty line"),
                Ok(text) => {
                    accepted.record(
                        position,
                        row_digest(text, Label::Unknown, None),
                        Label::Unknown,
                    );
                    accepted.push_raw(line);
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
        }
        if !accepted.is_empty() {
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
            dedup.push(accepted, text, label, group)?;
        }
        dedup.finish(path, &bytes)
    }
}
