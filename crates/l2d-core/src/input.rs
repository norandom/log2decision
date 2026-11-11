//! The input contract shared by all adapters: reading a source, the validation pass,
//! parsing with Polars, exact-text deduplication as frame operations, and the input
//! summary with its content hash.
//!
//! # How a source becomes a logs frame
//!
//! 1. **Validation pass.** The adapter walks the raw records in source order. A record
//!    that cannot be read is reported to [`Dedup::skip`] with its position and the
//!    reason. A readable record is copied into an [`Accepted`] batch together with its
//!    position and a digest of the text, label and group key the adapter read in it.
//! 2. **Polars parses the accepted records** ([`parse_csv_records`],
//!    [`parse_json_lines`]) into string columns, from which the adapter takes the
//!    `text`, `label` and `group` columns.
//! 3. **Agreement check.** [`Dedup::push`] compares what Polars produced with what the
//!    validation pass accepted: the number of rows, and row by row the digest of text,
//!    label and group key. Any difference fails the read with [`L2dError::Runtime`]
//!    naming the first differing position; a frame that disagrees with the validation
//!    pass is never returned, and no row is dropped silently.
//! 4. **Deduplication** is a stable group-by on the text. [`Dedup`] keeps its state as a
//!    frame between pushes, so a source may be pushed in several batches; the result is
//!    the same for any batch size as long as the batches arrive in source order.
//!
//! Both Polars readers and the group-by may use Polars' thread pool. The result does not
//! depend on scheduling: the readers return rows in input order, the group-by keeps
//! groups in order of first occurrence, and the first position is a minimum.
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
//! The `first_position` column and `SkippedRecord::position` both use this definition.
//!
//! # Untrusted data
//!
//! Log text, labels and group keys are data only. Nothing here executes them or uses
//! them to build paths or commands; the only path touched is the one the caller passed.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use polars::prelude::{
    Column, CsvEncoding, CsvReadOptions, DataFrame, DataType, Field, IntoLazy, JsonLineReader,
    PolarsError, Schema, SerReader, col, lit, when,
};
use sha2::{Digest, Sha256};

use crate::error::L2dError;
use crate::types::{InputSummary, Label, LogsFrame, SkippedRecord};

/// Turns one local file into the logs frame and a summary of what was read.
///
/// - Precondition: `path` exists and is readable; otherwise [`L2dError::Missing`].
/// - Postconditions: every text in the frame is unique; rows are in order of first
///   occurrence; `InputSummary::records` equals the sum of occurrences plus the number
///   of skipped records. The source is never modified.
pub trait InputAdapter {
    fn read(&self, path: &Path) -> Result<(LogsFrame, InputSummary), L2dError>;
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

// --- The accepted batch ------------------------------------------------------------------

/// A digest of one row's text, label and group key (64-bit FNV-1a).
///
/// The validation pass notes the digest of every record it accepts; after parsing, the
/// same digest is taken of every row Polars produced. Equal digests in the same order
/// are the evidence that Polars read the accepted records as the validation pass did.
pub fn row_digest(text: &str, label: Label, group: Option<&str>) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    // 0xFF and 0xFE cannot occur in UTF-8, so the three parts cannot run into each other
    // and an absent group differs from an empty one.
    let parts: [&[u8]; 5] = [
        text.as_bytes(),
        &[0xFF],
        label.as_str().as_bytes(),
        &[0xFF],
        group.map_or(&[0xFE], str::as_bytes),
    ];
    parts.into_iter().flatten().fold(OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(PRIME)
    })
}

/// The records one validation pass accepted, in source order: their raw bytes for
/// Polars to parse, and what the validation pass expects Polars to find in them.
#[derive(Debug, Default)]
pub struct Accepted {
    raw: Vec<u8>,
    positions: Vec<u64>,
    digests: Vec<u64>,
    good: u64,
    bad: u64,
}

impl Accepted {
    pub fn new() -> Self {
        Self::default()
    }

    /// Notes one accepted record: its position, the [`row_digest`] of the text, label
    /// and group key the validation pass read in it, and its label.
    pub fn record(&mut self, position: u64, digest: u64, label: Label) {
        self.positions.push(position);
        self.digests.push(digest);
        match label {
            Label::Good => self.good += 1,
            Label::Bad => self.bad += 1,
            Label::Unknown => {}
        }
    }

    /// Appends raw source bytes for Polars to parse. The caller adds the record
    /// terminator.
    pub fn push_raw(&mut self, bytes: &[u8]) {
        self.raw.extend_from_slice(bytes);
    }

    /// The raw bytes of the accepted records.
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// The number of accepted records.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Joins the positions to the columns Polars produced, after checking that Polars
    /// and the validation pass agree. The result has the columns `text`,
    /// `first_position`, `occurrences`, `group`, `seen_good` and `seen_bad`, one row per
    /// accepted record.
    fn rows(self, text: Column, label: Column, group: Column) -> Result<DataFrame, L2dError> {
        let disagree = |what: String| L2dError::Runtime {
            detail: format!("Polars and the validation pass disagree: {what}"),
        };
        let accepted = self.positions.len();
        for (name, column) in [("text", &text), ("label", &label), ("group", &group)] {
            if column.len() != accepted {
                return Err(disagree(format!(
                    "Polars produced {} {name} rows, the validation pass accepted {accepted} \
                     records",
                    column.len()
                )));
            }
        }
        let column_error = |error: PolarsError| disagree(error.to_string());
        let texts = text.str().map_err(column_error)?;
        let labels = label.str().map_err(column_error)?;
        let groups = group.str().map_err(column_error)?;
        let parsed = texts.iter().zip(labels.iter()).zip(groups.iter());
        for (((text, label), group), (position, digest)) in
            parsed.zip(self.positions.iter().zip(&self.digests))
        {
            let same = match (text, label.and_then(Label::from_name)) {
                (Some(text), Some(label)) if !text.is_empty() => {
                    row_digest(text, label, group) == *digest
                }
                _ => false,
            };
            if !same {
                return Err(disagree(format!(
                    "the record at position {position} reads differently after parsing \
                     (text of {} bytes, label {label:?}, group {group:?})",
                    text.map_or(0, str::len)
                )));
            }
        }

        DataFrame::new(vec![
            text.with_name(LogsFrame::TEXT.into()),
            label.with_name(LogsFrame::LABEL.into()),
            group.with_name(LogsFrame::GROUP.into()),
            Column::new(LogsFrame::FIRST_POSITION.into(), self.positions),
        ])
        .and_then(|frame| {
            frame
                .lazy()
                .select([
                    col(LogsFrame::TEXT),
                    col(LogsFrame::FIRST_POSITION),
                    lit(1u32).alias(LogsFrame::OCCURRENCES),
                    col(LogsFrame::GROUP),
                    col(LogsFrame::LABEL)
                        .eq(lit(Label::Good.as_str()))
                        .alias(SEEN_GOOD),
                    col(LogsFrame::LABEL)
                        .eq(lit(Label::Bad.as_str()))
                        .alias(SEEN_BAD),
                ])
                .collect()
        })
        .map_err(|error| frame_error("joining positions to the parsed rows", &error))
    }
}

// --- Parsing with Polars -----------------------------------------------------------------

/// The name [`parse_csv_records`] gives the field at `index` (0-based).
pub fn csv_field_name(index: usize) -> String {
    format!("f{index}")
}

/// Parses CSV records without a header row into string columns with Polars.
///
/// Every record must hold `fields` fields. Only the fields listed in `used` are
/// materialized; their columns are named by [`csv_field_name`]. An empty field is a
/// null. Polars infers nothing: no header, no types, no dates.
pub fn parse_csv_records(
    raw: &[u8],
    fields: usize,
    used: &[usize],
    separator: u8,
    quote: Option<u8>,
) -> Result<DataFrame, L2dError> {
    let schema = Schema::from_iter(
        (0..fields).map(|index| Field::new(csv_field_name(index).into(), DataType::String)),
    );
    let mut projection = used.to_vec();
    projection.sort_unstable();
    projection.dedup();
    CsvReadOptions::default()
        .with_has_header(false)
        .with_schema(Some(Arc::new(schema)))
        .with_projection(Some(Arc::new(projection)))
        .with_infer_schema_length(Some(0))
        .with_ignore_errors(false)
        .with_raise_if_empty(false)
        .with_rechunk(true)
        .map_parse_options(|options| {
            options
                .with_separator(separator)
                .with_quote_char(quote)
                .with_eol_char(b'\n')
                // Only fields the validation pass found to be valid UTF-8 are used;
                // other fields may hold any bytes and must not fail the read.
                .with_encoding(CsvEncoding::LossyUtf8)
                .with_null_values(None)
                .with_missing_is_null(true)
                .with_truncate_ragged_lines(false)
                .with_try_parse_dates(false)
        })
        .into_reader_with_file_handle(Cursor::new(raw))
        .finish()
        .map_err(|error| frame_error("parsing the accepted CSV records", &error))
}

/// Parses JSON lines into string columns with Polars, one column per key in `keys`.
///
/// Other keys are ignored; an absent key or a JSON `null` is a null.
pub fn parse_json_lines(raw: &[u8], keys: &[&str]) -> Result<DataFrame, L2dError> {
    let mut unique: Vec<&str> = Vec::new();
    for key in keys {
        if !unique.contains(key) {
            unique.push(key);
        }
    }
    let schema = Schema::from_iter(
        unique
            .into_iter()
            .map(|key| Field::new(key.into(), DataType::String)),
    );
    JsonLineReader::new(Cursor::new(raw))
        .with_schema(Arc::new(schema))
        .with_ignore_errors(false)
        .with_rechunk(true)
        .finish()
        .map_err(|error| frame_error("parsing the accepted JSON lines", &error))
}

/// The string column `name` of a frame Polars parsed.
pub fn string_column<'a>(
    frame: &'a DataFrame,
    name: &str,
) -> Result<&'a polars::prelude::StringChunked, L2dError> {
    frame
        .column(name)
        .and_then(Column::str)
        .map_err(|error| frame_error("reading a parsed column", &error))
}

fn frame_error(doing: &str, error: &PolarsError) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} failed: {error}"),
    }
}

// --- The CSV validation pass -------------------------------------------------------------

/// A CSV source with a header row, opened for the validation pass.
pub struct CsvSource<'a> {
    content: &'a [u8],
    reader: csv::Reader<&'a [u8]>,
    headers: csv::ByteRecord,
}

impl<'a> CsvSource<'a> {
    /// Reads the header row of `content`, which must not start with a byte-order mark.
    pub fn open(content: &'a [u8]) -> Result<Self, L2dError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            // Rows of any length are read so that a wrong field count is reported as a
            // skipped record with its position instead of ending the read.
            .flexible(true)
            .from_reader(content);
        let headers = reader
            .byte_headers()
            .map_err(|error| L2dError::InvalidInput {
                detail: format!("unreadable CSV header: {error}"),
            })?
            .clone();
        Ok(Self {
            content,
            reader,
            headers,
        })
    }

    /// The header row.
    pub fn headers(&self) -> &csv::ByteRecord {
        &self.headers
    }

    /// Walks all data rows in source order.
    ///
    /// A row whose field count differs from the header is skipped; `line_in_reason`
    /// adds the file line to that reason. Every other row is handed to `check`, which
    /// returns the [`row_digest`] of what it read in the row and the row's label, or
    /// the reason the row cannot be read. Accepted rows are copied for Polars as they stand in the
    /// source; a row that is not in the standard form (see [`standard_record_end`]) is
    /// rewritten into it, so that Polars splits it into the same fields.
    pub fn validate(
        mut self,
        line_in_reason: bool,
        dedup: &mut Dedup,
        mut check: impl FnMut(&csv::ByteRecord) -> Result<(u64, Label), String>,
    ) -> Result<Accepted, L2dError> {
        let mut accepted = Accepted::new();
        let mut row = csv::ByteRecord::new();
        let mut position = 0u64;
        loop {
            let more =
                self.reader
                    .read_byte_record(&mut row)
                    .map_err(|error| L2dError::Runtime {
                        detail: format!("reading CSV data row {}: {error}", position + 1),
                    })?;
            if !more {
                return Ok(accepted);
            }
            position += 1;
            if row.len() != self.headers.len() {
                let line = row
                    .position()
                    .filter(|_| line_in_reason)
                    .map(|at| format!(" (line {})", at.line()))
                    .unwrap_or_default();
                dedup.skip(
                    position,
                    format!(
                        "expected {} fields, found {}{line}",
                        self.headers.len(),
                        row.len()
                    ),
                );
                continue;
            }
            match check(&row) {
                Ok((digest, label)) => {
                    accepted.record(position, digest, label);
                    let start = row.position().map_or(0, |at| at.byte() as usize);
                    let end = self.reader.position().byte() as usize;
                    let raw = self.content.get(start..end).unwrap_or_default();
                    match standard_record_end(raw, &row) {
                        Some(end) => accepted.push_raw(&raw[..end]),
                        None => push_standard_record(&mut accepted, &row),
                    }
                    accepted.push_raw(b"\n");
                }
                Err(reason) => dedup.skip(position, reason),
            }
        }
    }
}

/// Where the fields of `record` end in `raw`, if `raw` is the record in the standard
/// form of RFC 4180 followed only by its line terminator; `None` otherwise.
///
/// In the standard form a field is either enclosed in quotes, with every quote inside
/// doubled, or it holds no quote, comma or line break. CSV readers agree on how such a
/// record splits into fields; they differ on anything else, for example a quote in the
/// middle of an unquoted field or a lone carriage return as line end.
pub fn standard_record_end(raw: &[u8], record: &csv::ByteRecord) -> Option<usize> {
    let mut at = 0;
    for (index, field) in record.iter().enumerate() {
        if index > 0 {
            if raw.get(at) != Some(&b',') {
                return None;
            }
            at += 1;
        }
        if raw.get(at) == Some(&b'"') {
            at += 1;
            for byte in field {
                if raw.get(at) != Some(byte) {
                    return None;
                }
                at += 1;
                if *byte == b'"' {
                    if raw.get(at) != Some(&b'"') {
                        return None;
                    }
                    at += 1;
                }
            }
            if raw.get(at) != Some(&b'"') {
                return None;
            }
            at += 1;
        } else {
            if field
                .iter()
                .any(|byte| matches!(byte, b'"' | b',' | b'\r' | b'\n'))
                || !raw[at..].starts_with(field)
            {
                return None;
            }
            at += field.len();
        }
    }
    raw[at..]
        .iter()
        .all(|byte| matches!(byte, b'\r' | b'\n'))
        .then_some(at)
}

/// Appends `record` in the standard form, every field enclosed in quotes.
fn push_standard_record(accepted: &mut Accepted, record: &csv::ByteRecord) {
    for (index, field) in record.iter().enumerate() {
        if index > 0 {
            accepted.push_raw(b",");
        }
        accepted.push_raw(b"\"");
        for part in field.split_inclusive(|byte| *byte == b'"') {
            accepted.push_raw(part);
            if part.ends_with(b"\"") {
                accepted.push_raw(b"\"");
            }
        }
        accepted.push_raw(b"\"");
    }
}

// --- Deduplication -----------------------------------------------------------------------

const SEEN_GOOD: &str = "seen_good";
const SEEN_BAD: &str = "seen_bad";

/// Collects the records of one source, deduplicated by exact text, as frame operations.
///
/// For a repeated text the first position and the first group key are kept and the
/// occurrence count grows. A text seen with both a good and a bad label becomes
/// `unknown` and is counted once in `InputSummary::label_conflicts`; an unlabelled
/// occurrence never conflicts with a labelled one.
///
/// The state between pushes is a frame with one row per distinct text in order of first
/// occurrence, so batches pushed in source order give the same result as one batch.
#[derive(Default)]
pub struct Dedup {
    state: Option<DataFrame>,
    skipped: Vec<SkippedRecord>,
    records: u64,
    good: u64,
    bad: u64,
}

impl Dedup {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that the record at `position` could not be read, and why.
    pub fn skip(&mut self, position: u64, reason: impl Into<String>) {
        self.records += 1;
        self.skipped.push(SkippedRecord {
            position,
            reason: reason.into(),
        });
    }

    /// Adds the accepted records of one batch.
    ///
    /// `text`, `label` (`good`, `bad` or `unknown`) and `group` (nullable) are the
    /// string columns taken from what Polars parsed out of [`Accepted::raw`], one row
    /// per accepted record in the same order. The push fails with
    /// [`L2dError::Runtime`] when they disagree with what the validation pass
    /// accepted: another number of rows, or a row whose text, label or group key is
    /// not what the validation pass read at that position.
    pub fn push(
        &mut self,
        accepted: Accepted,
        text: Column,
        label: Column,
        group: Column,
    ) -> Result<(), L2dError> {
        let (records, good, bad) = (accepted.len() as u64, accepted.good, accepted.bad);
        let batch = combine(accepted.rows(text, label, group)?)?;
        self.state = Some(match self.state.take() {
            None => batch,
            Some(state) => combine(
                state
                    .vstack(&batch)
                    .map_err(|error| frame_error("appending a batch", &error))?,
            )?,
        });
        self.records += records;
        self.good += good;
        self.bad += bad;
        Ok(())
    }

    /// How many accepted records carried the good and the bad label, before
    /// deduplication.
    pub fn label_counts(&self) -> (u64, u64) {
        (self.good, self.bad)
    }

    /// The logs frame in order of first occurrence, and the summary of the source at
    /// `source_path` whose raw content is `source_bytes`.
    pub fn finish(
        self,
        source_path: &Path,
        source_bytes: &[u8],
    ) -> Result<(LogsFrame, InputSummary), L2dError> {
        let failed = |error: PolarsError| frame_error("building the logs frame", &error);
        let (logs, label_conflicts) = match self.state {
            None => (LogsFrame::empty(), 0),
            Some(state) => {
                let conflicts = col(SEEN_GOOD).and(col(SEEN_BAD));
                let label = when(conflicts.clone())
                    .then(lit(Label::Unknown.as_str()))
                    .when(col(SEEN_GOOD))
                    .then(lit(Label::Good.as_str()))
                    .when(col(SEEN_BAD))
                    .then(lit(Label::Bad.as_str()))
                    .otherwise(lit(Label::Unknown.as_str()));
                let label_conflicts = state
                    .clone()
                    .lazy()
                    .select([conflicts.cast(DataType::UInt64).sum().alias("conflicts")])
                    .collect()
                    .and_then(|frame| Ok(frame.column("conflicts")?.u64()?.get(0)))
                    .map_err(failed)?
                    .unwrap_or(0);
                let frame = state
                    .lazy()
                    .select([
                        col(LogsFrame::TEXT),
                        label.alias(LogsFrame::LABEL),
                        col(LogsFrame::GROUP),
                        col(LogsFrame::FIRST_POSITION),
                        col(LogsFrame::OCCURRENCES),
                    ])
                    .collect()
                    .map_err(failed)?;
                (LogsFrame::new(frame)?, label_conflicts)
            }
        };
        // An adapter may report a skipped record after later ones; the report is in
        // source order.
        let mut skipped = self.skipped;
        skipped.sort_by_key(|record| record.position);
        let summary = InputSummary {
            source_path: source_path.to_path_buf(),
            records: self.records,
            distinct_texts: logs.len() as u64,
            content_sha256: content_sha256(source_bytes),
            skipped,
            label_conflicts,
        };
        Ok((logs, summary))
    }
}

/// One row per distinct text in order of first occurrence: the smallest position, the
/// sum of the occurrences, the first group key, and whether any row was good or bad.
fn combine(rows: DataFrame) -> Result<DataFrame, L2dError> {
    rows.lazy()
        .group_by_stable([col(LogsFrame::TEXT)])
        .agg([
            col(LogsFrame::FIRST_POSITION).min(),
            col(LogsFrame::OCCURRENCES).sum().cast(DataType::UInt32),
            col(LogsFrame::GROUP).first(),
            col(SEEN_GOOD).any(true),
            col(SEEN_BAD).any(true),
        ])
        .collect()
        .map_err(|error| frame_error("deduplicating", &error))
}
