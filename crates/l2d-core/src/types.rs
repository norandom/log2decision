//! Shared record, label, logs-frame, input-summary and encoder-identity types.

use std::fmt;
use std::path::PathBuf;

use polars::prelude::{Column, DataFrame, DataType, Field, PolarsError, Schema};
use serde::{Deserialize, Serialize};

use crate::error::L2dError;

/// Number of leading revision characters used in directory names (`<name>-<rev12>`).
pub const SHORT_REVISION_LEN: usize = 12;

/// The outcome attached to a log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Good,
    Bad,
    Unknown,
}

impl Label {
    /// The lower-case name used in frames and in JSON: `good`, `bad` or `unknown`.
    pub fn as_str(self) -> &'static str {
        match self {
            Label::Good => "good",
            Label::Bad => "bad",
            Label::Unknown => "unknown",
        }
    }

    /// The label a frame value names; `None` for anything but the three exact names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "good" => Some(Label::Good),
            "bad" => Some(Label::Bad),
            "unknown" => Some(Label::Unknown),
            _ => None,
        }
    }
}

/// One distinct log text with its label, group key and occurrence bookkeeping.
///
/// This is the row view of a [`LogsFrame`], for code that handles one log at a time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogRecord {
    pub text: String,
    pub label: Label,
    pub group: Option<String>,
    pub first_position: u64,
    pub occurrences: u32,
}

/// The logs frame of the design's "Data frames" section: one row per distinct log text.
///
/// | Column | Type | Null | Content |
/// |--------|------|------|---------|
/// | `text` | String | no | the log text, unique within the frame |
/// | `label` | String | no | `good`, `bad` or `unknown` |
/// | `group` | String | yes | the group key, if the source has one |
/// | `first_position` | UInt64 | no | position of the first occurrence in the source |
/// | `occurrences` | UInt32 | no | how often the text occurred |
///
/// Row order is meaningful: rows are in order of first occurrence, and whatever is
/// derived from this frame row by row keeps that order.
///
/// The wrapper guarantees the column names, types and order, that only `group` holds
/// nulls, and that every label is one of the three names. It does not check that the
/// texts are unique; the input adapters guarantee that.
///
/// Other crates reach Polars through [`crate::polars`], so the whole workspace uses one
/// Polars version.
#[derive(Debug, Clone)]
pub struct LogsFrame {
    frame: DataFrame,
}

impl LogsFrame {
    pub const TEXT: &'static str = "text";
    pub const LABEL: &'static str = "label";
    pub const GROUP: &'static str = "group";
    pub const FIRST_POSITION: &'static str = "first_position";
    pub const OCCURRENCES: &'static str = "occurrences";

    /// The column names and types, in order.
    pub fn schema() -> Schema {
        Schema::from_iter([
            Field::new(Self::TEXT.into(), DataType::String),
            Field::new(Self::LABEL.into(), DataType::String),
            Field::new(Self::GROUP.into(), DataType::String),
            Field::new(Self::FIRST_POSITION.into(), DataType::UInt64),
            Field::new(Self::OCCURRENCES.into(), DataType::UInt32),
        ])
    }

    /// Wraps `frame` after checking it against the documented schema.
    ///
    /// A frame with other column names, types or order, with a null outside `group`, or
    /// with a label other than `good`, `bad` or `unknown` is [`L2dError::Runtime`].
    pub fn new(mut frame: DataFrame) -> Result<Self, L2dError> {
        let expected = Self::schema();
        let found = frame.schema();
        if **found != expected {
            let describe = |schema: &Schema| {
                schema
                    .iter()
                    .map(|(name, dtype)| format!("{name} {dtype}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(L2dError::Runtime {
                detail: format!(
                    "not a logs frame: expected the columns [{}], found [{}]",
                    describe(&expected),
                    describe(found)
                ),
            });
        }
        for column in frame.get_columns() {
            let nulls = column.null_count();
            if nulls > 0 && column.name().as_str() != Self::GROUP {
                return Err(L2dError::Runtime {
                    detail: format!(
                        "not a logs frame: column '{}' holds {nulls} null values",
                        column.name()
                    ),
                });
            }
        }
        let labels = frame
            .column(Self::LABEL)
            .and_then(Column::str)
            .map_err(|error| logs_frame_error(&error))?;
        if let Some(other) = labels
            .iter()
            .flatten()
            .find(|name| Label::from_name(name).is_none())
        {
            return Err(L2dError::Runtime {
                detail: format!("not a logs frame: label '{other}' is not good, bad or unknown"),
            });
        }
        // One chunk per column, so that row views and numeric code read plain slices.
        frame.rechunk_mut();
        Ok(Self { frame })
    }

    /// A logs frame without rows.
    pub fn empty() -> Self {
        Self {
            frame: DataFrame::empty_with_schema(&Self::schema()),
        }
    }

    /// The logs frame holding `records` in the given order.
    pub fn from_records(records: &[LogRecord]) -> Result<Self, L2dError> {
        let texts: Vec<&str> = records.iter().map(|r| r.text.as_str()).collect();
        let labels: Vec<&str> = records.iter().map(|r| r.label.as_str()).collect();
        let groups: Vec<Option<&str>> = records.iter().map(|r| r.group.as_deref()).collect();
        let positions: Vec<u64> = records.iter().map(|r| r.first_position).collect();
        let occurrences: Vec<u32> = records.iter().map(|r| r.occurrences).collect();
        let frame = DataFrame::new(vec![
            Column::new(Self::TEXT.into(), texts),
            Column::new(Self::LABEL.into(), labels),
            Column::new(Self::GROUP.into(), groups),
            Column::new(Self::FIRST_POSITION.into(), positions),
            Column::new(Self::OCCURRENCES.into(), occurrences),
        ])
        .map_err(|error| logs_frame_error(&error))?;
        Self::new(frame)
    }

    /// The Polars frame with the documented columns.
    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    /// Gives up the wrapper and returns the Polars frame.
    pub fn into_frame(self) -> DataFrame {
        self.frame
    }

    /// The number of rows, which is the number of distinct texts.
    pub fn len(&self) -> usize {
        self.frame.height()
    }

    pub fn is_empty(&self) -> bool {
        self.frame.height() == 0
    }

    /// The rows as [`LogRecord`]s, in frame order.
    pub fn records(&self) -> impl Iterator<Item = LogRecord> + '_ {
        // `new` checked names, types, nulls and label values, so none of these can fail.
        const CHECKED: &str = "the logs frame was checked on construction";
        let column = |name: &str| self.frame.column(name).expect(CHECKED);
        let texts = column(Self::TEXT).str().expect(CHECKED).iter();
        let labels = column(Self::LABEL).str().expect(CHECKED).iter();
        let groups = column(Self::GROUP).str().expect(CHECKED).iter();
        let positions = column(Self::FIRST_POSITION).u64().expect(CHECKED).iter();
        let occurrences = column(Self::OCCURRENCES).u32().expect(CHECKED).iter();
        texts
            .zip(labels)
            .zip(groups)
            .zip(positions)
            .zip(occurrences)
            .map(
                |((((text, label), group), first_position), occurrences)| LogRecord {
                    text: text.expect(CHECKED).to_string(),
                    label: label.and_then(Label::from_name).expect(CHECKED),
                    group: group.map(str::to_string),
                    first_position: first_position.expect(CHECKED),
                    occurrences: occurrences.expect(CHECKED),
                },
            )
    }

    /// All rows as [`LogRecord`]s, in frame order.
    pub fn to_records(&self) -> Vec<LogRecord> {
        self.records().collect()
    }
}

/// Two logs frames are equal when they hold the same rows in the same order.
impl PartialEq for LogsFrame {
    fn eq(&self, other: &Self) -> bool {
        self.frame.equals_missing(&other.frame)
    }
}

fn logs_frame_error(error: &PolarsError) -> L2dError {
    L2dError::Runtime {
        detail: format!("not a logs frame: {error}"),
    }
}

/// A source record that could not be read: where it was and why it was skipped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedRecord {
    pub position: u64,
    pub reason: String,
}

/// What was read from one input source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSummary {
    pub source_path: PathBuf,
    pub records: u64,
    pub distinct_texts: u64,
    pub content_sha256: String,
    pub skipped: Vec<SkippedRecord>,
    pub label_conflicts: u64,
}

/// Which encoder produced a vector: name, revision hash and output dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EncoderIdentity {
    pub name: String,
    pub revision: String,
    pub dimension: u16,
}

impl EncoderIdentity {
    /// The first twelve characters of the revision, as used in package directory names.
    pub fn short_revision(&self) -> &str {
        short_revision(&self.revision)
    }
}

impl fmt::Display for EncoderIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{} ({} dimensions)",
            self.name,
            self.short_revision(),
            self.dimension
        )
    }
}

/// The first [`SHORT_REVISION_LEN`] characters of `revision`, or all of it when shorter.
pub fn short_revision(revision: &str) -> &str {
    match revision.char_indices().nth(SHORT_REVISION_LEN) {
        Some((end, _)) => &revision[..end],
        None => revision,
    }
}
