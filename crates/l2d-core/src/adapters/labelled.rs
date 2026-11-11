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
    Accepted, CsvSource, Dedup, InputAdapter, PARSE_COST, Source, batch_end, batch_step,
    csv_field_name, parse_csv_records, parse_json_lines, row_digest, string_column,
};
use crate::memory::{BatchPlan, Batcher, MemoryBudget};
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

    fn read_csv(
        &self,
        source: Source,
        dedup: &mut Dedup,
        batcher: &mut Batcher<'_>,
    ) -> Result<Source, L2dError> {
        let source = CsvSource::open(source, batcher.record_limit())?;
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
        let used: Vec<usize> = [Some(text_at), label_at, group_at]
            .into_iter()
            .flatten()
            .collect();
        let name = |at: Option<usize>| at.map(csv_field_name);

        source.validate(
            true,
            dedup,
            batcher,
            |row| {
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
            },
            // Polars parses the accepted rows of one batch.
            |dedup, accepted| {
                if accepted.is_empty() {
                    return Ok(());
                }
                let parsed =
                    parse_csv_records(accepted.raw(), headers.len(), &used, b',', Some(b'"'))?;
                push_parsed(
                    dedup,
                    accepted,
                    &parsed,
                    &name(Some(text_at)),
                    &name(label_at),
                    &name(group_at),
                )
            },
        )
    }

    fn read_jsonl(
        &self,
        source: Source,
        dedup: &mut Dedup,
        batcher: &mut Batcher<'_>,
    ) -> Result<Source, L2dError> {
        let keys: Vec<&str> = [
            Some(&self.text_column),
            self.label_column.as_ref(),
            self.group_column.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(String::as_str)
        .collect();
        let mut process = |dedup: &mut Dedup, batch: JsonBatch| self.push_json(dedup, batch, &keys);

        let mut lines = source.lines(batcher.record_limit());
        let mut batch = JsonBatch::default();
        let mut line = Vec::new();
        let mut position = 0u64;
        let mut objects = 0u64;
        let mut text_seen = false;
        let mut label_seen = false;
        let mut group_seen = false;

        while let Some(source_bytes) = lines.next_line(&mut line)? {
            position += 1;
            // The validation pass of one line; `Err` is the reason it cannot be read.
            let judged = (|| -> Result<(u64, Label), String> {
                if line.iter().all(u8::is_ascii_whitespace) {
                    return Err("empty line".to_string());
                }
                let (object, repeated) = match serde_json::from_slice::<JsonLine>(&line) {
                    Ok(JsonLine::Object { fields, repeated }) => (fields, repeated),
                    Ok(JsonLine::Other) => return Err("line is not a JSON object".into()),
                    Err(error) => return Err(format!("line is not valid JSON: {error}")),
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
                    return Err(format!("key '{key}' occurs more than once"));
                }

                let text = match text {
                    Some(Value::String(text)) if !text.is_empty() => text,
                    Some(Value::String(_)) => return Err("empty text".into()),
                    Some(_) => return Err("text is not a string".into()),
                    None => return Err("text column is absent".into()),
                };
                let label = match label {
                    None | Some(Value::Null) => Label::Unknown,
                    Some(Value::String(raw)) => parse_label(raw).ok_or(UNKNOWN_LABEL_VALUE)?,
                    Some(_) => return Err("label is not a string".into()),
                };
                let group = match group {
                    None | Some(Value::Null) => None,
                    Some(Value::String(raw)) => non_empty(raw),
                    Some(_) => return Err("group is not a string".into()),
                };
                Ok((row_digest(text, label, group), label))
            })();
            match judged {
                Ok((digest, label)) => {
                    let start = batch.raw.len();
                    batch.raw.extend_from_slice(&line);
                    batch.candidates.push(Candidate {
                        position,
                        digest,
                        label,
                        start,
                        end: batch.raw.len(),
                    });
                }
                Err(reason) => dedup.skip(position, reason),
            }
            batch_step(batcher, source_bytes, dedup, &mut batch, &mut process)?;
        }
        batch_end(batcher, dedup, batch, &mut process)?;

        // Known only at the end of the source: a configured column no record carries.
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
        Ok(lines.into_source())
    }

    /// Has Polars parse the accepted lines of one batch and pushes them into the
    /// deduplication.
    fn push_json(
        &self,
        dedup: &mut Dedup,
        mut batch: JsonBatch,
        keys: &[&str],
    ) -> Result<(), L2dError> {
        if batch.candidates.is_empty() {
            return Ok(());
        }
        let mut accepted = batch.accepted();
        let parsed = match parse_json_lines(accepted.raw(), keys) {
            Ok(parsed) => parsed,
            Err(_) => {
                // Polars refuses a line that is valid JSON for the validation pass, for
                // example an integer beyond 64 bits under a key that is not even read.
                // Such a line is a record that cannot be read: it is skipped and reported,
                // and the other lines are parsed again without it.
                batch.candidates.retain(|candidate| {
                    let mut alone = batch.raw[candidate.start..candidate.end].to_vec();
                    alone.push(b'\n');
                    match parse_json_lines(&alone, keys) {
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
                if batch.candidates.is_empty() {
                    return Ok(());
                }
                accepted = batch.accepted();
                parse_json_lines(accepted.raw(), keys)?
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
    fn read_from(
        &self,
        source: Source,
        budget: &mut MemoryBudget,
    ) -> Result<(LogsFrame, InputSummary, BatchPlan), L2dError> {
        let path = source.path().to_path_buf();
        let mut dedup = Dedup::new();
        let mut batcher = Batcher::new(budget, source.size(), PARSE_COST);
        let source = match self.format {
            LabelledFormat::Csv => self.read_csv(source, &mut dedup, &mut batcher)?,
            LabelledFormat::Jsonl => self.read_jsonl(source, &mut dedup, &mut batcher)?,
        };
        let plan = batcher.into_plan()?;
        let (logs, summary) = dedup.finish_hashed(&path, source.finish()?)?;
        Ok((logs, summary, plan))
    }
}

/// A JSON line the validation pass accepted: where it is in [`JsonBatch::raw`], and
/// what the validation pass read in it.
struct Candidate {
    position: u64,
    digest: u64,
    label: Label,
    start: usize,
    end: usize,
}

/// The JSON lines of one batch that the validation pass accepted.
#[derive(Default)]
struct JsonBatch {
    /// The accepted lines one after the other, without terminators.
    raw: Vec<u8>,
    candidates: Vec<Candidate>,
}

impl JsonBatch {
    /// The candidates as the batch for Polars to parse.
    fn accepted(&self) -> Accepted {
        let mut accepted = Accepted::new();
        for candidate in &self.candidates {
            accepted.record(candidate.position, candidate.digest, candidate.label);
            accepted.push_raw(&self.raw[candidate.start..candidate.end]);
            accepted.push_raw(b"\n");
        }
        accepted
    }
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
