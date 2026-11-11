//! One log per line.

use std::path::Path;

use crate::error::L2dError;
use crate::input::{Dedup, InputAdapter, read_source, split_lines, strip_bom};
use crate::types::{InputSummary, Label, LogRecord};

/// Reads a plain text file with one log per line.
///
/// Every line is a record whose position is its line number. The text is the line
/// without its terminator, otherwise unchanged. Records carry [`Label::Unknown`] and no
/// group. A line that is blank or not valid UTF-8 is skipped and reported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinesAdapter;

impl InputAdapter for LinesAdapter {
    fn read(&self, path: &Path) -> Result<(Vec<LogRecord>, InputSummary), L2dError> {
        let bytes = read_source(path)?;
        let mut dedup = Dedup::new();
        for (index, line) in split_lines(strip_bom(&bytes)).enumerate() {
            let position = index as u64 + 1;
            match std::str::from_utf8(line) {
                Ok(text) if text.trim().is_empty() => dedup.skip(position, "empty line"),
                Ok(text) => dedup.accept(position, text.to_string(), Label::Unknown, None),
                Err(_) => dedup.skip(position, "line is not valid UTF-8"),
            }
        }
        Ok(dedup.finish(path, &bytes))
    }
}
