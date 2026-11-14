//! The input contract shared by all adapters: reading a source in batches, the
//! validation pass, parsing with Polars, exact-text deduplication as frame operations,
//! and the input summary with its content hash.
//!
//! # How a source becomes a logs frame
//!
//! The source is read front to back in small pieces ([`Source`]) and processed in
//! batches of records. A [`Batcher`] sizes the batches to the free memory
//! (requirement 12): it samples the first records for the bytes per row, plans the
//! batch size within the [`MemoryBudget`], and has the free memory read again before
//! every batch. Steps 1 to 3 below happen once per batch, in source order; the
//! deduplication of step 4 keeps its state across batches. Record boundaries never
//! depend on where a read or a batch ends, so frame and summary are the same for any
//! batch size.
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

use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use polars::prelude::{
    Column, CsvEncoding, CsvReadOptions, DataFrame, DataType, Field, IntoColumn, IntoLazy,
    JsonLineReader, PolarsError, Schema, SerReader, StringChunkedBuilder, col, lit, when,
};
use sha2::{Digest, Sha256};

use crate::error::L2dError;
use crate::memory::{BatchCost, BatchPlan, Batcher, MemoryBudget, MemorySettings, RecordLimit};
use crate::types::{InputSummary, Label, LogsFrame, SkippedRecord};

/// How many bytes of memory reading needs per byte of source while it processes one
/// batch: the expansion factor the adapters declare for
/// [`estimate_rows`](crate::memory::estimate_rows).
///
/// A batch is in memory several times before it is merged into the deduplicated rows:
/// the accepted records copied for Polars (in a buffer that doubles as it grows), the
/// columns Polars parsed, the text, label and group columns derived from them, the
/// rows grouped by text, and the copy that lets go of the parse buffers.
///
/// The three figures of [`PARSE_COST`] were set together from the heap a batch really
/// takes (counting allocator, lines of 1 to 5,000 bytes, all distinct, in batches of
/// 1,024 to 65,536 rows): with them the peak of every batch stayed below what was
/// held before it plus `rows * (PARSE_ROW_BYTES + PARSE_EXPANSION * source bytes per
/// row)` by more than the batch needed in all shapes but the smallest batches, which
/// exceeded it by at most 1.9 MB; [`PARSE_BATCH_BYTES`] covers that. The test
/// `tests/memory_limit.rs` holds the result to the limit.
pub const PARSE_EXPANSION: f64 = 6.0;

/// The fixed cost of a row while its batch is parsed and deduplicated, in bytes:
/// string views of text, label and group, position, digest, occurrences, the grouping,
/// and, for a row that cannot be read, its record in the list of skipped records with
/// a reason of up to about 350 bytes.
pub const PARSE_ROW_BYTES: u64 = 500;

/// The fixed cost of a batch while it is parsed and grouped, in bytes: buffers of the
/// data layer that do not follow the size of the batch.
pub const PARSE_BATCH_BYTES: u64 = 4 * 1024 * 1024;

/// What the adapters declare to the [`Batcher`] for reading.
pub const PARSE_COST: BatchCost = BatchCost {
    expansion: PARSE_EXPANSION,
    row_bytes: PARSE_ROW_BYTES,
    batch_bytes: PARSE_BATCH_BYTES,
};

/// From this many bytes on, a batch is parsed on all threads of the data layer.
const PARALLEL_PARSE_BYTES: usize = 4 * 1024 * 1024;

/// What a skipped record is counted as besides its reason, in bytes: the record three
/// times over, because its list doubles when it is full and holds the old and the new
/// room for a moment, and the bookkeeping of the allocation of its reason.
pub(crate) const SKIPPED_RECORD_BYTES: u64 = 3 * size_of::<SkippedRecord>() as u64 + 16;

/// What merging costs per row besides the frames, in bytes: see [`Dedup::held_bytes`].
///
/// Measured (counting allocator): the peak of a merge of all rows is 239 bytes per row
/// for texts of 7 bytes (26 bytes per row estimated by Polars), 278 for 30 bytes (49
/// estimated) and 516 for 120 bytes (139 estimated). `200 + 2.6 * estimated` is above
/// all three by 8 to 12 percent.
const HELD_ROW_BYTES: u64 = 200;

/// How many times the estimated size of the frames a merge takes: the rows before the
/// merge, the merged rows, and room the string buffers have beyond their content.
const HELD_FACTOR: f64 = 2.6;

/// How many bytes are asked of the file in one read.
pub const READ_BYTES: usize = 64 * 1024;

/// Turns one local file into the logs frame and a summary of what was read.
///
/// The source is read in batches sized by a [`MemoryBudget`]; it is never in memory as
/// a whole. An adapter implements [`InputAdapter::read_from`]; the other two methods
/// open the source and observe the memory of the host.
///
/// - Precondition: the path exists and is readable; otherwise [`L2dError::Missing`].
/// - Postconditions: every text in the frame is unique; rows are in order of first
///   occurrence; `InputSummary::records` equals the sum of occurrences plus the number
///   of skipped records. Frame and summary are the same for any batch size and any
///   read size. The source is never modified.
/// - [`L2dError::Refused`] when the smallest batch, or a single record, does not fit the
///   memory limit; nothing is returned then.
pub trait InputAdapter {
    /// Reads an opened source in batches within `budget` and also returns the batch
    /// plan: what was observed and chosen, and how the batch size moved.
    fn read_from(
        &self,
        source: Source,
        budget: &mut MemoryBudget,
    ) -> Result<(LogsFrame, InputSummary, BatchPlan), L2dError>;

    /// Reads the file at `path` in batches within `budget`.
    fn read_with(
        &self,
        path: &Path,
        budget: &mut MemoryBudget,
    ) -> Result<(LogsFrame, InputSummary, BatchPlan), L2dError> {
        self.read_from(Source::open(path)?, budget)
    }

    /// Reads the file at `path` with the default budget: the free memory of the host
    /// is read now, and the batch plan is not returned.
    fn read(&self, path: &Path) -> Result<(LogsFrame, InputSummary), L2dError> {
        let mut budget = MemoryBudget::observe(&MemorySettings::default())?;
        self.read_with(path, &mut budget)
            .map(|(logs, summary, _)| (logs, summary))
    }
}

/// The lower-case hexadecimal SHA-256 of `bytes`.
pub fn content_sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub(crate) fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

// --- The source --------------------------------------------------------------------------

const BYTE_ORDER_MARK: &[u8] = b"\xef\xbb\xbf";

/// A source file opened read-only and read front to back in small pieces.
///
/// As a [`Read`] it gives the content without a leading UTF-8 byte-order mark. Every
/// byte of the file, the mark included, goes into the SHA-256 that [`Source::finish`]
/// returns, so the hash is that of the raw file although the file is never in memory
/// as a whole.
pub struct Source {
    path: PathBuf,
    file: File,
    size: u64,
    hasher: Sha256,
    started: bool,
    /// Bytes read while looking for the byte-order mark that are content.
    head: Vec<u8>,
    read_size: usize,
    window: Option<Window>,
    failure: Option<SourceFailure>,
}

enum SourceFailure {
    /// The file could not be read.
    Read(L2dError),
    /// The record being read is larger than the [`RecordLimit`].
    RecordTooLarge,
}

impl Source {
    /// Opens the file at `path`.
    ///
    /// Any failure to open or read it reports [`L2dError::Missing`] with the path as
    /// expected location.
    pub fn open(path: &Path) -> Result<Self, L2dError> {
        let file = File::open(path).map_err(|error| missing(path, &error))?;
        let size = file
            .metadata()
            .map_err(|error| missing(path, &error))?
            .len();
        Ok(Self {
            path: path.to_path_buf(),
            file,
            size,
            hasher: Sha256::new(),
            started: false,
            head: Vec::new(),
            read_size: READ_BYTES,
            window: None,
            failure: None,
        })
    }

    /// Hands out at most `bytes` bytes per read (at least one). The result of a read
    /// does not depend on it; tests use small values to put the end of a read inside a
    /// record, a line end or the byte-order mark.
    pub fn with_read_size(mut self, bytes: usize) -> Self {
        self.read_size = bytes.max(1);
        self
    }

    /// The path the source was opened at.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The size of the file in bytes when it was opened.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The source as lines; a line longer than `limit` allows is refused.
    pub fn lines(self, limit: RecordLimit) -> Lines {
        Lines {
            reader: BufReader::with_capacity(READ_BYTES, self),
            limit,
            position: 0,
        }
    }

    /// Reads what is left of the file and returns the lower-case hexadecimal SHA-256 of
    /// all its bytes.
    pub fn finish(mut self) -> Result<String, L2dError> {
        self.window = None;
        let mut rest = [0u8; 4096];
        loop {
            match self.read(&mut rest) {
                Ok(0) => break,
                Ok(_) => {}
                Err(error) => return Err(self.failed(&error, None)),
            }
        }
        Ok(hex(&self.hasher.finalize()))
    }

    /// The error for a failed read: what the source noted, or `error` itself.
    /// `record` is the limit and the position of the record that was being read.
    fn failed(&mut self, error: &std::io::Error, record: Option<(RecordLimit, u64)>) -> L2dError {
        match (self.failure.take(), record) {
            (Some(SourceFailure::Read(failure)), _) => failure,
            (Some(SourceFailure::RecordTooLarge), Some((limit, position))) => {
                limit.refusal(position)
            }
            _ => missing(&self.path, error),
        }
    }

    fn read_file(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            match self.file.read(out) {
                Ok(count) => {
                    self.hasher.update(&out[..count]);
                    return Ok(count);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => {
                    self.failure = Some(SourceFailure::Read(missing(&self.path, &error)));
                    return Err(error);
                }
            }
        }
    }

    /// Reads the first three bytes and drops them when they are the byte-order mark.
    fn start(&mut self) -> std::io::Result<()> {
        let mut head = [0u8; BYTE_ORDER_MARK.len()];
        let mut filled = 0;
        while filled < head.len() {
            match self.read_file(&mut head[filled..])? {
                0 => break,
                count => filled += count,
            }
        }
        if &head[..filled] != BYTE_ORDER_MARK {
            self.head = head[..filled].to_vec();
        }
        self.started = true;
        Ok(())
    }
}

impl Read for Source {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if !self.started {
            self.start()?;
        }
        let wanted = out.len().min(self.read_size);
        let count = if self.head.is_empty() {
            self.read_file(&mut out[..wanted])?
        } else {
            let count = wanted.min(self.head.len());
            out[..count].copy_from_slice(&self.head[..count]);
            self.head.drain(..count);
            count
        };
        if let Some(window) = &mut self.window
            && !window.push(&out[..count])
        {
            self.failure = Some(SourceFailure::RecordTooLarge);
            return Err(std::io::Error::other("a record is larger than allowed"));
        }
        Ok(count)
    }
}

fn missing(path: &Path, error: &std::io::Error) -> L2dError {
    let item = if error.kind() == std::io::ErrorKind::NotFound {
        "input file".to_string()
    } else {
        format!("readable input file ({error})")
    };
    L2dError::Missing {
        item,
        expected_at: path.to_path_buf(),
    }
}

/// The bytes a reader has taken from a [`Source`] and may still ask for: from the start
/// of the record being read to the end of what the reader read ahead.
struct Window {
    bytes: Vec<u8>,
    /// The offset of `bytes[0]` in the content.
    base: u64,
    /// The offset before which nothing is asked for any more.
    keep_from: u64,
    /// How many bytes may be kept.
    limit: u64,
}

impl Window {
    /// Appends `bytes`; `false` when more than the limit would have to be kept.
    fn push(&mut self, bytes: &[u8]) -> bool {
        let done = (self.keep_from - self.base) as usize;
        if done >= READ_BYTES || done == self.bytes.len() {
            self.bytes.drain(..done);
            self.base = self.keep_from;
        }
        self.bytes.extend_from_slice(bytes);
        self.base + self.bytes.len() as u64 - self.keep_from <= self.limit
    }

    /// The content from offset `start` to offset `end`.
    fn get(&self, start: u64, end: u64) -> &[u8] {
        let at = |offset: u64| offset.saturating_sub(self.base) as usize;
        self.bytes.get(at(start)..at(end)).unwrap_or_default()
    }

    /// Lets go of everything before offset `end`.
    fn done_before(&mut self, end: u64) {
        let read = self.base + self.bytes.len() as u64;
        self.keep_from = end.clamp(self.keep_from, read);
    }
}

/// The lines of a [`Source`], without their `\n` or `\r\n` terminators.
///
/// A final terminator does not start another line; an empty source has no lines.
pub struct Lines {
    reader: BufReader<Source>,
    limit: RecordLimit,
    position: u64,
}

impl Lines {
    /// Reads the next line into `line` and returns how many source bytes it took, its
    /// terminator included; `None` at the end of the source.
    ///
    /// A line longer than the record limit is [`L2dError::Refused`], naming its
    /// position; it is not read to its end.
    pub fn next_line(&mut self, line: &mut Vec<u8>) -> Result<Option<u64>, L2dError> {
        line.clear();
        self.position += 1;
        let mut taken = 0u64;
        loop {
            let buffer = match self.reader.fill_buf() {
                Ok(buffer) => buffer,
                Err(error) => return Err(self.reader.get_mut().failed(&error, None)),
            };
            if buffer.is_empty() {
                if taken == 0 {
                    return Ok(None);
                }
                break;
            }
            let end = buffer.iter().position(|byte| *byte == b'\n');
            let used = end.map_or(buffer.len(), |end| end + 1);
            line.extend_from_slice(&buffer[..end.unwrap_or(used)]);
            self.reader.consume(used);
            taken += used as u64;
            if line.len() as u64 > self.limit.max_bytes {
                return Err(self.limit.refusal(self.position));
            }
            if end.is_some() {
                break;
            }
        }
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        Ok(Some(taken))
    }

    /// The source, once all lines are read.
    pub fn into_source(self) -> Source {
        self.reader.into_inner()
    }
}

// --- Batches -----------------------------------------------------------------------------

/// Notes one source record of `source_bytes` bytes, read or skipped, and processes the
/// batch with `process` when the record completes it.
///
/// `batch` is whatever the adapter collects for one batch; it is taken and left empty.
/// `process` must accept a batch without accepted records.
pub fn batch_step<B: Default>(
    batcher: &mut Batcher<'_>,
    source_bytes: u64,
    dedup: &mut Dedup,
    batch: &mut B,
    process: &mut impl FnMut(&mut Dedup, B) -> Result<(), L2dError>,
) -> Result<(), L2dError> {
    if batcher.record(source_bytes)? {
        process(dedup, std::mem::take(batch))?;
        batcher.processed(dedup.held_bytes())?;
    }
    Ok(())
}

/// Processes the last, partial batch at the end of the source.
pub fn batch_end<B>(
    batcher: &mut Batcher<'_>,
    dedup: &mut Dedup,
    batch: B,
    process: &mut impl FnMut(&mut Dedup, B) -> Result<(), L2dError>,
) -> Result<(), L2dError> {
    if batcher.end()? {
        process(dedup, batch)?;
        batcher.processed(dedup.held_bytes())?;
    }
    Ok(())
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
    /// Whether a byte-order mark was put in front of `raw` for Polars to drop.
    guarded: bool,
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
    ///
    /// Polars takes U+FEFF at the start of a buffer it parses for a byte-order mark and
    /// drops it, once. Here it is text: the byte-order mark of a source is dropped
    /// where the source is opened, so a record that starts with U+FEFF means it. When
    /// the first record of the batch starts with one, a second one is put in front of
    /// it for Polars to drop. This does not depend on how the bytes are cut into
    /// calls.
    pub fn push_raw(&mut self, bytes: &[u8]) {
        let at_start = !self.guarded && self.raw.len() < BYTE_ORDER_MARK.len();
        self.raw.extend_from_slice(bytes);
        if at_start && self.raw.starts_with(BYTE_ORDER_MARK) {
            self.raw
                .splice(..0, BYTE_ORDER_MARK.iter().copied())
                .for_each(drop);
            self.guarded = true;
        }
    }

    /// The bytes for Polars to parse: the raw bytes of the accepted records, behind a
    /// byte-order mark when the first record starts with U+FEFF (see
    /// [`Accepted::push_raw`]).
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
    pub(crate) fn rows(
        self,
        text: Column,
        label: Column,
        group: Column,
    ) -> Result<DataFrame, L2dError> {
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
///
/// Polars drops one U+FEFF at the start of `raw` as a byte-order mark. A caller whose
/// first record starts with one as text puts a second one in front;
/// [`Accepted::push_raw`] does.
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
    // Polars splits its input among its threads and gives every part buffers of a fixed
    // size, megabytes for a batch of a few kilobytes. A small batch is parsed on one
    // thread, where the memory follows the size of the batch.
    let threads = (raw.len() < PARALLEL_PARSE_BYTES).then_some(1);
    CsvReadOptions::default()
        .with_n_threads(threads)
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

pub(crate) fn frame_error(doing: &str, error: &PolarsError) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} failed: {error}"),
    }
}

// --- The CSV validation pass -------------------------------------------------------------

/// A CSV source with a header row, opened for the validation pass.
pub struct CsvSource {
    reader: csv::Reader<Source>,
    headers: csv::ByteRecord,
    limit: RecordLimit,
}

impl CsvSource {
    /// Reads the header row of `source`. A record larger than `limit` allows is
    /// refused instead of read.
    pub fn open(mut source: Source, limit: RecordLimit) -> Result<Self, L2dError> {
        // The raw bytes of a record are taken from what the CSV reader has read: at
        // most one record and one buffer of the reader ahead of it.
        source.window = Some(Window {
            bytes: Vec::new(),
            base: 0,
            keep_from: 0,
            limit: limit.max_bytes.saturating_add(READ_BYTES as u64),
        });
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            // Rows of any length are read so that a wrong field count is reported as a
            // skipped record with its position instead of ending the read.
            .flexible(true)
            .buffer_capacity(READ_BYTES)
            .from_reader(source);
        let headers = match reader.byte_headers() {
            Ok(headers) => headers.clone(),
            Err(error) => {
                return Err(match reader.get_mut().failure.take() {
                    Some(SourceFailure::Read(failure)) => failure,
                    Some(SourceFailure::RecordTooLarge) => limit.refusal(0),
                    None => L2dError::InvalidInput {
                        detail: format!("unreadable CSV header: {error}"),
                    },
                });
            }
        };
        Ok(Self {
            reader,
            headers,
            limit,
        })
    }

    /// The header row.
    pub fn headers(&self) -> &csv::ByteRecord {
        &self.headers
    }

    /// Walks all data rows in source order, in batches, and returns the source once it
    /// is read to its end.
    ///
    /// A row whose field count differs from the header is skipped; `line_in_reason`
    /// adds the file line to that reason. Every other row is handed to `check`, which
    /// returns the [`row_digest`] of what it read in the row and the row's label, or
    /// the reason the row cannot be read. Accepted rows are copied for Polars as they
    /// stand in the source; a row that is not in the standard form (see
    /// [`standard_record_end`]) is rewritten into it, so that Polars splits it into the
    /// same fields.
    ///
    /// Whenever `batcher` finds a batch complete, and at the end of the source, the
    /// accepted rows of the batch are handed to `process`, which parses them and pushes
    /// them into the deduplication. `process` is also called for a batch without
    /// accepted rows.
    pub fn validate(
        mut self,
        line_in_reason: bool,
        dedup: &mut Dedup,
        batcher: &mut Batcher<'_>,
        mut check: impl FnMut(&csv::ByteRecord) -> Result<(u64, Label), String>,
        mut process: impl FnMut(&mut Dedup, Accepted) -> Result<(), L2dError>,
    ) -> Result<Source, L2dError> {
        let mut accepted = Accepted::new();
        let mut row = csv::ByteRecord::new();
        let mut position = 0u64;
        loop {
            let more = match self.reader.read_byte_record(&mut row) {
                Ok(more) => more,
                Err(error) => {
                    return Err(match self.reader.get_mut().failure.take() {
                        Some(SourceFailure::Read(failure)) => failure,
                        Some(SourceFailure::RecordTooLarge) => self.limit.refusal(position + 1),
                        None => L2dError::Runtime {
                            detail: format!("reading CSV data row {}: {error}", position + 1),
                        },
                    });
                }
            };
            if !more {
                batch_end(batcher, dedup, accepted, &mut process)?;
                return Ok(self.reader.into_inner());
            }
            position += 1;
            let start = row.position().map_or(0, csv::Position::byte);
            let end = self.reader.position().byte();
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
            } else {
                match check(&row) {
                    Ok((digest, label)) => {
                        accepted.record(position, digest, label);
                        let raw = self
                            .reader
                            .get_ref()
                            .window
                            .as_ref()
                            .map_or(&[][..], |window| window.get(start, end));
                        match standard_record_end(raw, &row) {
                            Some(end) => accepted.push_raw(&raw[..end]),
                            None => push_standard_record(&mut accepted, &row),
                        }
                        accepted.push_raw(b"\n");
                    }
                    Err(reason) => dedup.skip(position, reason),
                }
            }
            if let Some(window) = &mut self.reader.get_mut().window {
                window.done_before(end);
            }
            batch_step(
                batcher,
                end.saturating_sub(start),
                dedup,
                &mut accepted,
                &mut process,
            )?;
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
/// The state between pushes is kept as frames with one row per distinct text in order of
/// first occurrence, so batches pushed in source order give the same result as one
/// batch. Every batch is deduplicated in itself when it is pushed. It is merged with the
/// earlier rows only when the rows waiting for a merge are as many as the rows already
/// merged, and at the end; so the work of all merges together grows with the number of
/// rows, not with the number of batches times the number of rows.
#[derive(Default)]
pub struct Dedup {
    /// The merged rows: one row per distinct text of the batches merged so far.
    state: Option<DataFrame>,
    /// The batches pushed since the last merge, each deduplicated in itself, in source
    /// order.
    waiting: Option<DataFrame>,
    skipped: Vec<SkippedRecord>,
    /// What the skipped records are counted as in [`Dedup::held_bytes`].
    skipped_bytes: u64,
    /// What the merged rows and the waiting rows are counted as in
    /// [`Dedup::held_bytes`]; kept up to date by `push`, so that asking is cheap.
    state_held: u64,
    waiting_held: u64,
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
        let mut reason: String = reason.into();
        reason.shrink_to_fit();
        self.records += 1;
        self.skipped_bytes += SKIPPED_RECORD_BYTES + reason.capacity() as u64;
        self.skipped.push(SkippedRecord { position, reason });
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
        let batch = own_strings(combine(accepted.rows(text, label, group)?)?)?;
        let batch_held = held_for(&batch);
        let waiting = append(self.waiting.take(), batch)?;
        match self.state.take() {
            // The first batch; rows wait only behind merged rows.
            None => {
                self.state = Some(waiting);
                self.state_held = batch_held;
            }
            Some(state) if waiting.height() >= state.height() => {
                let merged = combine(append(Some(state), waiting)?)?;
                self.state_held = held_for(&merged);
                self.waiting_held = 0;
                self.state = Some(merged);
            }
            Some(state) => {
                self.state = Some(state);
                self.waiting = Some(waiting);
                self.waiting_held += batch_held;
            }
        }
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

    /// An estimate of the memory a read needs between batches for what it has collected
    /// so far, in bytes ([`MemoryBudget::hold`]).
    ///
    /// For the rows it is the peak of merging all of them, which happens at the latest
    /// when the read ends: the rows before the merge, the grouping, and the merged rows
    /// (`HELD_ROW_BYTES` per row and `HELD_FACTOR` times the size Polars estimates for
    /// the frames). The skipped records are counted with their reasons and the room
    /// their list takes while it grows.
    pub fn held_bytes(&self) -> u64 {
        self.state_held + self.waiting_held + self.skipped_bytes
    }

    /// The logs frame in order of first occurrence, and the summary of the source at
    /// `source_path` whose raw content is `source_bytes`.
    pub fn finish(
        self,
        source_path: &Path,
        source_bytes: &[u8],
    ) -> Result<(LogsFrame, InputSummary), L2dError> {
        self.finish_hashed(source_path, content_sha256(source_bytes))
    }

    /// Like [`Dedup::finish`] for a source that was not kept in memory:
    /// `content_sha256` is the hash of its raw content ([`Source::finish`]).
    pub fn finish_hashed(
        self,
        source_path: &Path,
        content_sha256: String,
    ) -> Result<(LogsFrame, InputSummary), L2dError> {
        let failed = |error: PolarsError| frame_error("building the logs frame", &error);
        let state = match (self.state, self.waiting) {
            (state, Some(waiting)) => Some(combine(append(state, waiting)?)?),
            (state, None) => state,
        };
        let (logs, label_conflicts) = match state {
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
        if !skipped.is_sorted_by_key(|record| record.position) {
            skipped.sort_by_key(|record| record.position);
        }
        let summary = InputSummary {
            source_path: source_path.to_path_buf(),
            records: self.records,
            distinct_texts: logs.len() as u64,
            content_sha256,
            skipped,
            label_conflicts,
        };
        Ok((logs, summary))
    }
}

/// What keeping and merging the rows of `frame` is counted as: see
/// [`Dedup::held_bytes`].
fn held_for(frame: &DataFrame) -> u64 {
    frame.height() as u64 * HELD_ROW_BYTES + (frame.estimated_size() as f64 * HELD_FACTOR) as u64
}

/// `frame` with text and group columns that own their bytes.
///
/// The columns of a parsed batch point into the buffers of the parse, which are larger
/// than the strings they hold and stay allocated as long as one string is in use.
/// Copying the strings of the deduplicated batch lets go of those buffers.
pub(crate) fn own_strings(mut frame: DataFrame) -> Result<DataFrame, L2dError> {
    let failed = |error: PolarsError| frame_error("copying the strings of a batch", &error);
    for name in [LogsFrame::TEXT, LogsFrame::GROUP] {
        let strings = frame.column(name).and_then(Column::str).map_err(failed)?;
        let mut owned = StringChunkedBuilder::new(name.into(), strings.len());
        for value in strings.iter() {
            owned.append_option(value);
        }
        let owned = owned.finish().into_column();
        frame.with_column(owned).map_err(failed)?;
    }
    Ok(frame)
}

/// `batch` appended to `rows`, without copying either.
fn append(rows: Option<DataFrame>, batch: DataFrame) -> Result<DataFrame, L2dError> {
    match rows {
        None => Ok(batch),
        Some(rows) => rows
            .vstack(&batch)
            .map_err(|error| frame_error("appending a batch", &error)),
    }
}

/// One row per distinct text in order of first occurrence: the smallest position, the
/// sum of the occurrences, the first group key, and whether any row was good or bad.
pub(crate) fn combine(rows: DataFrame) -> Result<DataFrame, L2dError> {
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
