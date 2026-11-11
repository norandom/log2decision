//! Labelled CSV and JSONL datasets with configurable text, label and group columns.
//!
//! Label values are `good` and `bad`, compared without case and surrounding white
//! space. An empty value, `unknown`, a JSON `null` or an absent JSON key means the
//! record is unlabelled ([`Label::Unknown`]). Any other value makes the record
//! unreadable: it is skipped and reported, never guessed.

use std::path::Path;

use serde_json::Value;

use crate::error::L2dError;
use crate::input::{Dedup, InputAdapter, read_source, split_lines, strip_bom};
use crate::types::{InputSummary, Label, LogRecord};

/// The file format of a labelled dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelledFormat {
    /// Comma-separated values with a header row naming the columns.
    Csv,
    /// One JSON object per line; the columns are object keys.
    Jsonl,
}

impl LabelledFormat {
    /// Chooses the format from the file extension (`.csv`, `.jsonl`, `.ndjson`).
    pub fn from_path(path: &Path) -> Result<Self, L2dError> {
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);
        match extension.as_deref() {
            Some("csv") => Ok(Self::Csv),
            Some("jsonl" | "ndjson") => Ok(Self::Jsonl),
            _ => Err(L2dError::InvalidInput {
                detail: format!(
                    "cannot tell the labelled format of {}: expected a .csv, .jsonl or .ndjson file",
                    path.display()
                ),
            }),
        }
    }
}

/// Reads a labelled CSV or JSONL dataset.
///
/// The text column is required. The label and group columns are optional; without a
/// label column every record is [`Label::Unknown`], without a group column no record has
/// a group. A configured column that the source does not have is
/// [`L2dError::InvalidInput`]: for CSV when the header lacks it, for JSONL when no
/// record carries the key.
///
/// A record is skipped and reported when it is malformed (CSV: field count differs from
/// the header; JSONL: not a JSON object), when its text is empty, absent or not a
/// string, or when its label is not one of the accepted values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelledAdapter {
    pub format: LabelledFormat,
    pub text_column: String,
    pub label_column: Option<String>,
    pub group_column: Option<String>,
}

impl LabelledAdapter {
    /// An adapter for `format` with text column `text`, label column `label` and no
    /// group column.
    pub fn new(format: LabelledFormat) -> Self {
        Self {
            format,
            text_column: "text".into(),
            label_column: Some("label".into()),
            group_column: None,
        }
    }

    pub fn with_text_column(mut self, name: impl Into<String>) -> Self {
        self.text_column = name.into();
        self
    }

    pub fn with_label_column(mut self, name: impl Into<String>) -> Self {
        self.label_column = Some(name.into());
        self
    }

    pub fn with_group_column(mut self, name: impl Into<String>) -> Self {
        self.group_column = Some(name.into());
        self
    }

    fn read_csv(&self, bytes: &[u8], dedup: &mut Dedup) -> Result<(), L2dError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            // Rows of any length are read so that a wrong field count is reported as a
            // skipped record with its position instead of ending the read.
            .flexible(true)
            .from_reader(bytes);
        let headers = reader
            .byte_headers()
            .map_err(|error| L2dError::InvalidInput {
                detail: format!("unreadable CSV header: {error}"),
            })?
            .clone();
        if headers.is_empty() {
            return Err(L2dError::InvalidInput {
                detail: "the CSV file has no header row".into(),
            });
        }
        let column = |name: &str, role: &str| {
            headers
                .iter()
                .position(|header| header == name.as_bytes())
                .ok_or_else(|| L2dError::InvalidInput {
                    detail: format!(
                        "{role} column '{name}' is not in the CSV header ({} columns)",
                        headers.len()
                    ),
                })
        };
        let text_at = column(&self.text_column, "text")?;
        let label_at = match &self.label_column {
            Some(name) => Some(column(name, "label")?),
            None => None,
        };
        let group_at = match &self.group_column {
            Some(name) => Some(column(name, "group")?),
            None => None,
        };

        let mut row = csv::ByteRecord::new();
        let mut position = 0u64;
        loop {
            let more = reader
                .read_byte_record(&mut row)
                .map_err(|error| L2dError::Runtime {
                    detail: format!("reading CSV data row {}: {error}", position + 1),
                })?;
            if !more {
                return Ok(());
            }
            position += 1;
            if row.len() != headers.len() {
                let line = row
                    .position()
                    .map(|at| format!(" (line {})", at.line()))
                    .unwrap_or_default();
                dedup.skip(
                    position,
                    format!(
                        "expected {} fields, found {}{line}",
                        headers.len(),
                        row.len()
                    ),
                );
                continue;
            }
            let field = |at: usize| std::str::from_utf8(&row[at]);
            let Ok(text) = field(text_at) else {
                dedup.skip(position, "text is not valid UTF-8");
                continue;
            };
            if text.is_empty() {
                dedup.skip(position, "empty text");
                continue;
            }
            let label = match label_at.map(field) {
                None => Label::Unknown,
                Some(Ok(raw)) => match parse_label(raw) {
                    Some(label) => label,
                    None => {
                        dedup.skip(position, UNKNOWN_LABEL_VALUE);
                        continue;
                    }
                },
                Some(Err(_)) => {
                    dedup.skip(position, "label is not valid UTF-8");
                    continue;
                }
            };
            let group = match group_at.map(field) {
                None => None,
                Some(Ok(raw)) => non_empty(raw),
                Some(Err(_)) => {
                    dedup.skip(position, "group is not valid UTF-8");
                    continue;
                }
            };
            dedup.accept(position, text.to_string(), label, group);
        }
    }

    fn read_jsonl(&self, bytes: &[u8], dedup: &mut Dedup) -> Result<(), L2dError> {
        let mut objects = 0u64;
        let mut text_seen = false;
        let mut label_seen = false;
        let mut group_seen = false;

        for (index, line) in split_lines(bytes).enumerate() {
            let position = index as u64 + 1;
            if line.iter().all(u8::is_ascii_whitespace) {
                dedup.skip(position, "empty line");
                continue;
            }
            let object = match serde_json::from_slice::<Value>(line) {
                Ok(Value::Object(object)) => object,
                Ok(_) => {
                    dedup.skip(position, "line is not a JSON object");
                    continue;
                }
                Err(error) => {
                    dedup.skip(position, format!("line is not valid JSON: {error}"));
                    continue;
                }
            };
            objects += 1;

            let text = object.get(&self.text_column);
            let label = self.label_column.as_ref().and_then(|key| object.get(key));
            let group = self.group_column.as_ref().and_then(|key| object.get(key));
            text_seen |= text.is_some();
            label_seen |= label.is_some();
            group_seen |= group.is_some();

            let text = match text {
                Some(Value::String(text)) if !text.is_empty() => text,
                Some(Value::String(_)) => {
                    dedup.skip(position, "empty text");
                    continue;
                }
                Some(_) => {
                    dedup.skip(position, "text is not a string");
                    continue;
                }
                None => {
                    dedup.skip(position, "text column is absent");
                    continue;
                }
            };
            let label = match label {
                None | Some(Value::Null) => Label::Unknown,
                Some(Value::String(raw)) => match parse_label(raw) {
                    Some(label) => label,
                    None => {
                        dedup.skip(position, UNKNOWN_LABEL_VALUE);
                        continue;
                    }
                },
                Some(_) => {
                    dedup.skip(position, "label is not a string");
                    continue;
                }
            };
            let group = match group {
                None | Some(Value::Null) => None,
                Some(Value::String(raw)) => non_empty(raw),
                Some(_) => {
                    dedup.skip(position, "group is not a string");
                    continue;
                }
            };
            dedup.accept(position, text.clone(), label, group);
        }

        if objects > 0 {
            let configured = [
                (Some(&self.text_column), text_seen, "text"),
                (self.label_column.as_ref(), label_seen, "label"),
                (self.group_column.as_ref(), group_seen, "group"),
            ];
            for (name, seen, role) in configured {
                if let (Some(name), false) = (name, seen) {
                    return Err(L2dError::InvalidInput {
                        detail: format!(
                            "{role} column '{name}' is not in any of the {objects} JSONL records"
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

impl InputAdapter for LabelledAdapter {
    fn read(&self, path: &Path) -> Result<(Vec<LogRecord>, InputSummary), L2dError> {
        let bytes = read_source(path)?;
        let content = strip_bom(&bytes);
        let mut dedup = Dedup::new();
        match self.format {
            LabelledFormat::Csv => self.read_csv(content, &mut dedup)?,
            LabelledFormat::Jsonl => self.read_jsonl(content, &mut dedup)?,
        }
        Ok(dedup.finish(path, &bytes))
    }
}

const UNKNOWN_LABEL_VALUE: &str = "label is not good, bad or unknown";

/// The label a raw value stands for, or `None` when the value is not accepted.
fn parse_label(raw: &str) -> Option<Label> {
    let value = raw.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("unknown") {
        Some(Label::Unknown)
    } else if value.eq_ignore_ascii_case("good") {
        Some(Label::Good)
    } else if value.eq_ignore_ascii_case("bad") {
        Some(Label::Bad)
    } else {
        None
    }
}

fn non_empty(raw: &str) -> Option<String> {
    (!raw.is_empty()).then(|| raw.to_string())
}
