//! Labelled CSV and JSONL datasets with configurable text, label and group columns.
//!
//! Label values are `good` and `bad`, compared without case and surrounding white
//! space. An empty value, `unknown`, a JSON `null` or an absent JSON key means the
//! record is unlabelled ([`Label::Unknown`]). Any other value makes the record
//! unreadable: it is skipped and reported, never guessed.
//!
//! A validation pass finds the unreadable records; Polars parses the accepted ones. The
//! same functions (`parse_label`, `non_empty`) judge a raw value in the validation
//! pass and turn the value Polars parsed into the frame value.

use std::path::Path;

use polars::prelude::{Column, DataFrame, DataType, IntoColumn};
use serde_json::Value;

use crate::error::L2dError;
use crate::input::{
    Accepted, CsvSource, Dedup, InputAdapter, csv_field_name, parse_csv_records, parse_json_lines,
    read_source, row_digest, split_lines, string_column, strip_bom,
};
use crate::types::{InputSummary, Label, LogsFrame};

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
/// label column every row is `unknown`, without a group column no row has a group. A
/// configured column that the source does not have is [`L2dError::InvalidInput`]: for
/// CSV when the header lacks it, for JSONL when no record carries the key.
///
/// A record is skipped and reported when it is malformed (CSV: field count differs from
/// the header; JSONL: not a JSON object), when its text is empty, absent or not a
/// string, or when its label is not one of the accepted values. A JSONL record is also
/// skipped when it repeats the text, label or group key, because JSON readers differ on
/// which value counts, and when Polars cannot parse a line that is valid JSON (for
/// example an integer beyond 64 bits).
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

    fn read_csv(&self, content: &[u8], dedup: &mut Dedup) -> Result<(), L2dError> {
        let source = CsvSource::open(content)?;
        let headers = source.headers().clone();
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

        let accepted = source.validate(true, dedup, |row| {
            let field = |at: usize| std::str::from_utf8(&row[at]);
            let text = field(text_at).map_err(|_| "text is not valid UTF-8")?;
            if text.is_empty() {
                return Err("empty text".into());
            }
            let label = match label_at.map(field) {
                None => Label::Unknown,
                Some(Ok(raw)) => parse_label(raw).ok_or(UNKNOWN_LABEL_VALUE)?,
                Some(Err(_)) => return Err("label is not valid UTF-8".into()),
            };
            let group = match group_at.map(field) {
                None => None,
                Some(Ok(raw)) => non_empty(raw),
                Some(Err(_)) => return Err("group is not valid UTF-8".into()),
            };
            Ok((row_digest(text, label, group), label))
        })?;
        if accepted.is_empty() {
            return Ok(());
        }

        let used: Vec<usize> = [Some(text_at), label_at, group_at]
            .into_iter()
            .flatten()
            .collect();
        let parsed = parse_csv_records(accepted.raw(), headers.len(), &used, b',', Some(b'"'))?;
        let name = |at: Option<usize>| at.map(csv_field_name);
        push_parsed(
            dedup,
            accepted,
            &parsed,
            &name(Some(text_at)),
            &name(label_at),
            &name(group_at),
        )
    }

    fn read_jsonl(&self, content: &[u8], dedup: &mut Dedup) -> Result<(), L2dError> {
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        let mut objects = 0u64;
        let mut text_seen = false;
        let mut label_seen = false;
        let mut group_seen = false;

        for (index, line) in split_lines(content).enumerate() {
            let position = index as u64 + 1;
            if line.iter().all(u8::is_ascii_whitespace) {
                dedup.skip(position, "empty line");
                continue;
            }
            let (object, repeated) = match serde_json::from_slice::<JsonLine>(line) {
                Ok(JsonLine::Object { fields, repeated }) => (fields, repeated),
                Ok(JsonLine::Other) => {
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

            // JSON readers differ on which of two values of one key they keep, so a
            // record that repeats a configured key is not guessed at.
            let configured = [
                Some(&self.text_column),
                self.label_column.as_ref(),
                self.group_column.as_ref(),
            ];
            if let Some(key) = repeated.iter().find(|key| configured.contains(&Some(*key))) {
                dedup.skip(position, format!("key '{key}' occurs more than once"));
                continue;
            }

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
            candidates.push(Candidate {
                position,
                digest: row_digest(text, label, group),
                label,
                line,
            });
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
        if candidates.is_empty() {
            return Ok(());
        }

        let keys: Vec<&str> = [
            Some(&self.text_column),
            self.label_column.as_ref(),
            self.group_column.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(String::as_str)
        .collect();
        let mut accepted = accepted_lines(&candidates);
        let parsed = match parse_json_lines(accepted.raw(), &keys) {
            Ok(parsed) => parsed,
            Err(_) => {
                // Polars refuses a line that is valid JSON for the validation pass, for
                // example an integer beyond 64 bits under a key that is not even read.
                // Such a line is a record that cannot be read: it is skipped and reported,
                // and the other lines are parsed again without it.
                candidates.retain(|candidate| {
                    let alone = accepted_lines(std::slice::from_ref(candidate));
                    match parse_json_lines(alone.raw(), &keys) {
                        Ok(_) => true,
                        Err(error) => {
                            let reason = match error {
                                L2dError::Runtime { detail } => detail,
                                other => other.to_string(),
                            };
                            dedup.skip(
                                candidate.position,
                                format!("line cannot be read by Polars: {reason}"),
                            );
                            false
                        }
                    }
                });
                if candidates.is_empty() {
                    return Ok(());
                }
                accepted = accepted_lines(&candidates);
                parse_json_lines(accepted.raw(), &keys)?
            }
        };
        push_parsed(
            dedup,
            accepted,
            &parsed,
            &Some(self.text_column.clone()),
            &self.label_column,
            &self.group_column,
        )
    }
}

impl InputAdapter for LabelledAdapter {
    fn read(&self, path: &Path) -> Result<(LogsFrame, InputSummary), L2dError> {
        let bytes = read_source(path)?;
        let content = strip_bom(&bytes);
        let mut dedup = Dedup::new();
        match self.format {
            LabelledFormat::Csv => self.read_csv(content, &mut dedup)?,
            LabelledFormat::Jsonl => self.read_jsonl(content, &mut dedup)?,
        }
        dedup.finish(path, &bytes)
    }
}

/// A JSON line the validation pass accepted.
struct Candidate<'a> {
    position: u64,
    digest: u64,
    label: Label,
    line: &'a [u8],
}

/// The batch of candidates for Polars to parse.
fn accepted_lines(candidates: &[Candidate<'_>]) -> Accepted {
    let mut accepted = Accepted::new();
    for candidate in candidates {
        accepted.record(candidate.position, candidate.digest, candidate.label);
        accepted.push_raw(candidate.line);
        accepted.push_raw(b"\n");
    }
    accepted
}

/// One JSON line as the validation pass reads it.
enum JsonLine {
    /// An object: its keys with the last value of each, and the keys that occur more
    /// than once.
    Object {
        fields: serde_json::Map<String, Value>,
        repeated: Vec<String>,
    },
    /// Any other JSON value.
    Other,
}

impl<'de> serde::Deserialize<'de> for JsonLine {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LineVisitor;

        impl<'de> serde::de::Visitor<'de> for LineVisitor {
            type Value = JsonLine;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON value")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<JsonLine, A::Error> {
                let mut fields = serde_json::Map::new();
                let mut repeated = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    let value: Value = map.next_value()?;
                    if fields.insert(key.clone(), value).is_some() {
                        repeated.push(key);
                    }
                }
                Ok(JsonLine::Object { fields, repeated })
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<JsonLine, A::Error> {
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(JsonLine::Other)
            }

            fn visit_bool<E>(self, _: bool) -> Result<JsonLine, E> {
                Ok(JsonLine::Other)
            }

            fn visit_i64<E>(self, _: i64) -> Result<JsonLine, E> {
                Ok(JsonLine::Other)
            }

            fn visit_u64<E>(self, _: u64) -> Result<JsonLine, E> {
                Ok(JsonLine::Other)
            }

            fn visit_f64<E>(self, _: f64) -> Result<JsonLine, E> {
                Ok(JsonLine::Other)
            }

            fn visit_str<E>(self, _: &str) -> Result<JsonLine, E> {
                Ok(JsonLine::Other)
            }

            fn visit_unit<E>(self) -> Result<JsonLine, E> {
                Ok(JsonLine::Other)
            }
        }

        deserializer.deserialize_any(LineVisitor)
    }
}

/// Takes the text, label and group columns out of what Polars parsed and hands them to
/// the deduplication, which checks them against the accepted records.
fn push_parsed(
    dedup: &mut Dedup,
    accepted: Accepted,
    parsed: &DataFrame,
    text: &Option<String>,
    label: &Option<String>,
    group: &Option<String>,
) -> Result<(), L2dError> {
    let rows = parsed.height();
    let text = match text {
        Some(name) => string_column(parsed, name)?.clone().into_column(),
        None => Column::full_null("text".into(), rows, &DataType::String),
    };
    let label = match label {
        // A value the validation pass would have refused is kept as it is, so that the
        // agreement check reports it.
        Some(name) => Column::new(
            "label".into(),
            string_column(parsed, name)?
                .iter()
                .map(|raw| {
                    let raw = raw.unwrap_or_default();
                    match parse_label(raw) {
                        Some(label) => label.as_str(),
                        None => raw,
                    }
                })
                .collect::<Vec<&str>>(),
        ),
        None => Column::new("label".into(), vec![Label::Unknown.as_str(); rows]),
    };
    let group = match group {
        Some(name) => Column::new(
            "group".into(),
            string_column(parsed, name)?
                .iter()
                .map(|raw| raw.and_then(non_empty))
                .collect::<Vec<Option<&str>>>(),
        ),
        None => Column::full_null("group".into(), rows, &DataType::String),
    };
    dedup.push(accepted, text, label, group)
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

/// `raw` as a group key; an empty value is no group.
fn non_empty(raw: &str) -> Option<&str> {
    (!raw.is_empty()).then_some(raw)
}
