//! Shared record, label, input-summary and encoder-identity types.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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

/// One distinct log text with its label, group key and occurrence bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogRecord {
    pub text: String,
    pub label: Label,
    pub group: Option<String>,
    pub first_position: u64,
    pub occurrences: u32,
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
