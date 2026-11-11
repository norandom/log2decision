//! The input contract shared by all adapters: reading a source, exact-text
//! deduplication, and the input summary with its content hash.
//!
//! # Positions
//!
//! A *position* is the 1-based index of a record in its source, counted over every
//! record the adapter saw, readable or not:
//!
//! - line-based and JSONL input: the line number in the file;
//! - CSV input: the data-row index, the header row not counted (data row 121 is file
//!   line 122 when no earlier field spans several lines). Blank lines between CSV rows
//!   are not rows and take no position.
//!
//! `LogRecord::first_position` and `SkippedRecord::position` both use this definition.
//!
//! # Untrusted data
//!
//! Log text, labels and group keys are data only. Nothing here executes them or uses
//! them to build paths or commands; the only path touched is the one the caller passed.

use std::collections::HashMap;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::L2dError;
use crate::types::{InputSummary, Label, LogRecord, SkippedRecord};

/// Turns one local file into deduplicated log records and a summary of what was read.
///
/// - Precondition: `path` exists and is readable; otherwise [`L2dError::Missing`].
/// - Postconditions: every returned text is unique; records are in order of first
///   occurrence; `InputSummary::records` equals the sum of occurrences plus the number
///   of skipped records. The source is never modified.
pub trait InputAdapter {
    fn read(&self, path: &Path) -> Result<(Vec<LogRecord>, InputSummary), L2dError>;
}

/// Reads the whole source file, opened read-only.
///
/// Any failure to read reports [`L2dError::Missing`] with the path as expected location.
pub fn read_source(path: &Path) -> Result<Vec<u8>, L2dError> {
    std::fs::read(path).map_err(|error| {
        let item = if error.kind() == std::io::ErrorKind::NotFound {
            "input file".to_string()
        } else {
            format!("readable input file ({error})")
        };
        L2dError::Missing {
            item,
            expected_at: path.to_path_buf(),
        }
    })
}

/// The lower-case hexadecimal SHA-256 of `bytes`.
pub fn content_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `bytes` without a leading UTF-8 byte-order mark.
pub fn strip_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes)
}

/// The lines of `bytes` without their `\n` or `\r\n` terminators.
///
/// A final terminator does not start another line; an empty input has no lines.
pub fn split_lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    let body = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    let has_lines = !bytes.is_empty();
    body.split(|byte| *byte == b'\n')
        .filter(move |_| has_lines)
        .map(|line| line.strip_suffix(b"\r").unwrap_or(line))
}

struct Entry {
    record: LogRecord,
    seen_good: bool,
    seen_bad: bool,
}

/// Collects the records of one source, deduplicated by exact text.
///
/// For a repeated text the first position and the first group key are kept and the
/// occurrence count grows. A text seen with both a good and a bad label becomes
/// [`Label::Unknown`] and is counted once in `InputSummary::label_conflicts`; an
/// unlabelled occurrence never conflicts with a labelled one.
#[derive(Default)]
pub struct Dedup {
    index: HashMap<String, usize>,
    entries: Vec<Entry>,
    skipped: Vec<SkippedRecord>,
    records: u64,
}

impl Dedup {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one readable record found at `position`.
    pub fn accept(&mut self, position: u64, text: String, label: Label, group: Option<String>) {
        self.records += 1;
        let entry = match self.index.get(&text) {
            Some(&at) => {
                let entry = &mut self.entries[at];
                entry.record.occurrences = entry.record.occurrences.saturating_add(1);
                entry
            }
            None => {
                self.index.insert(text.clone(), self.entries.len());
                self.entries.push(Entry {
                    record: LogRecord {
                        text,
                        label: Label::Unknown,
                        group,
                        first_position: position,
                        occurrences: 1,
                    },
                    seen_good: false,
                    seen_bad: false,
                });
                self.entries.last_mut().expect("entry was just pushed")
            }
        };
        match label {
            Label::Good => entry.seen_good = true,
            Label::Bad => entry.seen_bad = true,
            Label::Unknown => {}
        }
    }

    /// Records that the record at `position` could not be read, and why.
    pub fn skip(&mut self, position: u64, reason: impl Into<String>) {
        self.records += 1;
        self.skipped.push(SkippedRecord {
            position,
            reason: reason.into(),
        });
    }

    /// The deduplicated records in order of first occurrence, and the summary of the
    /// source at `source_path` whose raw content is `source_bytes`.
    pub fn finish(self, source_path: &Path, source_bytes: &[u8]) -> (Vec<LogRecord>, InputSummary) {
        let mut label_conflicts = 0;
        let records: Vec<LogRecord> = self
            .entries
            .into_iter()
            .map(|entry| {
                let mut record = entry.record;
                record.label = match (entry.seen_good, entry.seen_bad) {
                    (true, true) => {
                        label_conflicts += 1;
                        Label::Unknown
                    }
                    (true, false) => Label::Good,
                    (false, true) => Label::Bad,
                    (false, false) => Label::Unknown,
                };
                record
            })
            .collect();
        let summary = InputSummary {
            source_path: source_path.to_path_buf(),
            records: self.records,
            distinct_texts: records.len() as u64,
            content_sha256: content_sha256(source_bytes),
            skipped: self.skipped,
            label_conflicts,
        };
        (records, summary)
    }
}
