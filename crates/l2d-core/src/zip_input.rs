//! Zip archives as input (requirement 13): the log records in the entries of an
//! archive, read without unpacking it, in batches within the memory limit.
//!
//! # How an archive is read
//!
//! [`ZipInput::open`] hashes the archive file (SHA-256, one pass) and reads its
//! directory. [`ZipInput::batches`] then walks the entries in directory order, which
//! is the record order, so the result does not depend on timing. One entry is open at
//! a time and is read as a stream: the file, the inflater, for a gzip entry a second
//! inflater, then lines. Nothing is written anywhere unless a directory for
//! deduplication on disk is named, and then only there, under names the tool makes up;
//! an entry name is only ever a name in a report: it is never used as a path.
//!
//! An entry is read when it is a file, matches an entry pattern (when patterns are
//! given), is stored or deflated, is not encrypted, and its first 8 KiB are text: no
//! NUL byte and valid UTF-8. A gzip entry (by its first bytes, not by its name) is
//! read through and judged by its content. Every other entry is skipped with a reason
//! ([`EntrySkip`]); every entry is in the report ([`EntryReport`]).
//!
//! Lines are split and judged as the lines adapter does: `\n` or `\r\n` ends a line, a
//! byte-order mark at the start of an entry is dropped, a blank line and a line that is
//! not valid UTF-8 are skipped records. A line longer than the maximum record length is
//! a skipped record too, reported with its length; it is passed over without being
//! held. The accepted lines of a batch are parsed by Polars and checked against the
//! validation pass like those of every adapter ([`crate::input`]).
//!
//! # Batches and what they hold
//!
//! The batches are cut by the [`Batcher`]. A batch is handed out as a [`LogsFrame`] of
//! the texts that were **first seen in that batch**, in order of first occurrence; a
//! batch without a new text is not handed out. The rows of all batches, one after the
//! other, are the distinct texts of the archive in order of first occurrence, whatever
//! the batch size. `occurrences` in a batch counts the batch only; the count over the
//! whole archive is [`ZipInput::occurrences`] once the last batch is out.
//!
//! # Positions
//!
//! A position names the entry and the line: [`ArchivePosition`], packed into the `u64`
//! of `first_position` and of [`SkippedRecord::position`] as the entry's index in the
//! directory (upper 24 bits) and the 1-based line in the entry (lower 40 bits).
//! [`ArchiveSummary::locate`] gives the entry's path and the line. Positions grow in
//! record order. The first entry has index 0, so its positions are its line numbers.
//!
//! # U+FEFF
//!
//! At the very start of an entry (of the content of a gzip entry) U+FEFF is a
//! byte-order mark and is dropped, once. Anywhere else it is text: at the start of a
//! later line, inside a line, and as a second one at the start of the entry. The same
//! holds for a plain file read by the lines adapter.
//!
//! # Where this reader and other zip readers can differ
//!
//! The directory is authoritative. An entry's own header is used only to find where
//! its data starts; what it says about method, sizes and checksum is not compared with
//! the directory, and a difference shows as damage of the entry (wrong size or
//! checksum) when it is read. The end record is the last one whose comment ends at the
//! end of the file, so bytes in an archive comment that look like an end record and
//! reach the end of the file are taken for it. Entries in the directory beyond the
//! count of the end record are ignored, and so is data no directory entry points to.
//! Entries whose data overlaps are each read. A reader that trusts entry headers, or
//! picks another end record, can therefore see other content in a crafted archive.
//!
//! # Failures of the file
//!
//! A read of the archive file that fails (the disk, not the data) ends the read with
//! [`L2dError::Runtime`]; it is not reported as a damaged entry.
//!
//! # Memory
//!
//! Held for the whole read and counted against the [`MemoryBudget`]: the directory
//! (one small record and the name per entry), the readers of one entry, the
//! deduplication set, and the skipped records. The deduplication set holds, per
//! distinct text, 128 bits of the text's SHA-256 and the occurrence count: 21 bytes per
//! slot, 24 to 48 bytes per distinct text, and while it grows the old and the new
//! table. The room for the next growth is taken off what a batch may use before the
//! batch is read. When the set, the skipped records or the directory no longer fit, the
//! read is refused with the memory needed and available (requirement 13.12, refusal
//! branch).
//!
//! # Deduplication on disk
//!
//! With a directory for it ([`ZipInput::spill_into`]) a set that can no longer grow
//! does not end the read (requirement 13.12, the branch that continues): the set is
//! frozen, the rows whose text it does not hold are written to disk, partitioned by
//! their key, and deduplicated there after the last entry (`zip_spill.rs` has the
//! method). The set is frozen when its next growth, the fixed memory of a batch and
//! the smallest batch no longer fit together. The rows handed out are the same as
//! without it, in the same order: the texts of the set during the read, the texts on
//! disk after it, each group in order of first occurrence, and every text of the set
//! was first seen before any text on disk. A frame from disk carries the occurrence
//! counts over the archive; for every row, [`ZipInput::occurrence_counts`] gives them
//! afterwards. What the caller holds beside the archive ([`ZipInput::hold_beside`])
//! counts as held from the first batch on. Still refused, with the figures: skipped
//! records or a directory that do not fit, a frozen set that leaves no room for the
//! smallest batch, and a partition on disk whose distinct keys do not fit (while the
//! archive is read, checked between batches: task 1.20). What deduplication on disk
//! is estimated to write is [`SelectedBytes::dedup_disk_bytes`]; with
//! [`ZipInput::check_disk_at_freeze`] the free disk is compared with it at the freeze.
//!
//! A read can also name its work directory without having it made
//! ([`ZipInput::spill_when_needed`]; task 1.20, remediation): until the set is frozen
//! it is the read without a directory (nothing made, no disk looked at, nothing more
//! held); at the freeze, between batches, the directory is made, the disk checked and
//! the buffers held. While a batch is held the freeze waits for its end as long as the
//! set can still grow; when it cannot, the buffers must fit the room the batch kept for
//! the set's growth, or the growth is refused as without a directory. Such a read keeps
//! no occurrence counts.
//!
//! # What a hostile archive can and cannot do
//!
//! - Sizes the directory declares are checked against the file before they are used.
//!   An entry that inflates to more than it declares is cut off at the declared size
//!   and reported as damaged; one that holds less, or fails its CRC-32, is reported as
//!   damaged when its end is reached. The records read from it until then are kept.
//! - A high compression ratio costs time, not memory: an entry is never held, only the
//!   line being read, and that only up to the maximum record length. There is no limit
//!   on the time an archive may take. A gzip entry declares no size that could be
//!   checked while it is read.
//! - Many entries, long names: the directory is charged to the budget before and while
//!   it is read, and refused when it does not fit.
//! - Directories and symbolic links are not read. Nested archives other than gzip are
//!   not read.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use polars::prelude::{BooleanChunked, Column, DataFrame, DataType, NewChunkedArray};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::adapters::lines::NO_SEPARATOR;
use crate::error::L2dError;
use crate::input::{
    Accepted, PARSE_BATCH_BYTES, PARSE_COST, PARSE_EXPANSION, PARSE_ROW_BYTES, READ_BYTES,
    SKIPPED_RECORD_BYTES, combine, csv_field_name, frame_error, hex, own_strings,
    parse_csv_records, row_digest,
};
use crate::memory::{BatchPlan, Batcher, MemoryBudget, PROCESS_BASELINE_BYTES, exact};
use crate::parts::{WorkDir, free_disk_bytes, require_free_in};
use crate::types::{InputSummary, Label, LogsFrame, SkippedRecord};
use crate::zip_directory::{
    DataFault, DirectoryEntry, EntryKind, locate_data, read_directory, unreadable,
};
use crate::zip_spill::{
    CountFile, Merge, SPILL_HELD, SPILL_HELD_WITHOUT_COUNTS_LEAST, Spill, estimated_disk_bytes,
};

pub use crate::zip_spill::DiskDedup;

pub use crate::zip_directory::MAX_ENTRIES;

/// The maximum record length when the analyst sets none (owner decision of 2026-11-12).
pub const DEFAULT_MAX_RECORD_BYTES: u64 = 64 * 1024;
/// The largest maximum record length that can be set: what the store accepts as a text.
pub const MAX_RECORD_BYTES: u64 = 1024 * 1024;
/// The most lines of one entry: a position names the line in 40 bits.
pub const MAX_ENTRY_LINES: u64 = (1 << 40) - 1;

/// How many of the first bytes of an entry decide whether it is text.
const TEXT_CHECK_BYTES: usize = 8 * 1024;
/// How many bytes of the archive are hashed at a time.
const HASH_READ_BYTES: usize = 256 * 1024;
/// What the readers of one open entry are counted as, beside the record being read:
/// the buffer in front of the file and the line buffer (64 KiB each), two inflaters
/// with their 32 KiB windows, the buffer between them, and the two heads of 8 KiB.
///
/// Measured (counting allocator, `tests/zip_memory.rs`): reading an archive whose one
/// entry is a deflated gzip file with a line of 64 MiB between two short ones grows
/// the heap by 0.3 to 0.5 MiB at its peak, the 256 KiB hashing buffer of
/// [`ZipInput::open`] and the batch of two rows included; that read is counted as
/// 1.0 MiB with the default maximum record length.
const STREAM_BYTES: u64 = 512 * 1024;
/// How many times the maximum record length is held for the record being read: the
/// line buffer, which doubles as it grows, and one record in the batch beyond what the
/// batch was sized for (a batch ends with the record that fills it).
const RECORD_FACTOR: u64 = 2 + PARSE_EXPANSION as u64;
/// The bytes of one slot of the deduplication set: 16 of the key, 4 of the count, 1 of
/// the table's control byte.
const SLOT_BYTES: u64 = 21;
/// The longest reason kept for a damaged entry, in bytes.
const REASON_BYTES: usize = 300;

const BYTE_ORDER_MARK: &[u8] = b"\xef\xbb\xbf";
const GZIP_MAGIC: &[u8] = &[0x1f, 0x8b];

/// What the analyst may set for reading an archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipInputSettings {
    /// `--entry`: glob patterns over the entry path (`*` and `?` also match `/`,
    /// `[...]` is a character class; case-sensitive). An entry is read when it matches
    /// one of them; without patterns every entry is tried.
    pub entries: Vec<String>,
    /// `--max-record-bytes`: a line longer than this is skipped and reported. 1 to
    /// [`MAX_RECORD_BYTES`].
    pub max_record_bytes: u64,
}

impl Default for ZipInputSettings {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
        }
    }
}

/// Why an entry was not read, or not to its end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntrySkip {
    /// Its first 8 KiB hold a NUL byte or are not valid UTF-8.
    NotText,
    Encrypted,
    /// Compressed with a method other than deflate; the method's number.
    UnsupportedMethod(u16),
    /// An archive or compressed file other than gzip, or a gzip file that holds one.
    NestedArchive,
    /// Patterns were given and none matches its path.
    NoPatternMatch,
    /// Its header, its data or its checksum is wrong; reading stopped where that was
    /// found. [`EntryReport::records`] says how many records were read until then;
    /// they are kept.
    Damaged(String),
    Directory,
    /// A symbolic link.
    Link,
    /// The read ended before the end of this entry (a refusal, or the caller stopped).
    NotReached,
}

impl fmt::Display for EntrySkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntrySkip::NotText => f.write_str("not text"),
            EntrySkip::Encrypted => f.write_str("encrypted"),
            EntrySkip::UnsupportedMethod(method) => {
                write!(f, "compression method {method} is not supported")
            }
            EntrySkip::NestedArchive => f.write_str("nested archive"),
            EntrySkip::NoPatternMatch => f.write_str("matches no entry pattern"),
            EntrySkip::Damaged(reason) => write!(f, "damaged: {reason}"),
            EntrySkip::Directory => f.write_str("directory"),
            EntrySkip::Link => f.write_str("symbolic link"),
            EntrySkip::NotReached => f.write_str("not reached"),
        }
    }
}

/// One entry of the archive: its path, the records read from it (skipped records
/// included), and why it was not read, or `None` when it was read to its end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryReport {
    /// The entry's name in the archive, with replacement characters where it is not
    /// UTF-8. Data, never a path that is opened.
    pub path: String,
    pub records: u64,
    pub skipped: Option<EntrySkip>,
}

/// What was read from an archive (requirements 13.3 and 13.13).
///
/// `input` is the summary of the archive as one input: its path, the SHA-256 of the
/// archive file, the records of all entries, the distinct texts, and the skipped
/// records with packed positions ([`ArchivePosition`]). `entries` has one report per
/// directory entry, in directory order; the records of the entries add up to
/// `input.records`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveSummary {
    pub input: InputSummary,
    pub entries: Vec<EntryReport>,
}

impl ArchiveSummary {
    /// The entry path and the 1-based line a position names; `None` when the position
    /// names no entry of this archive.
    pub fn locate(&self, position: u64) -> Option<(&str, u64)> {
        let position = ArchivePosition::unpack(position);
        self.entries
            .get(position.entry as usize)
            .map(|entry| (entry.path.as_str(), position.line))
    }

    /// The entries that were read to their end.
    pub fn read(&self) -> impl Iterator<Item = &EntryReport> {
        self.entries.iter().filter(|entry| entry.skipped.is_none())
    }

    /// The damaged entries (requirement 13.6: a command reports them and exits with a
    /// non-zero status).
    pub fn damaged(&self) -> impl Iterator<Item = &EntryReport> {
        self.entries
            .iter()
            .filter(|entry| matches!(entry.skipped, Some(EntrySkip::Damaged(_))))
    }
}

/// The position of a record in an archive: the index of its entry in the directory and
/// the 1-based line in the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArchivePosition {
    /// Below [`MAX_ENTRIES`].
    pub entry: u32,
    /// At most [`MAX_ENTRY_LINES`].
    pub line: u64,
}

impl ArchivePosition {
    const LINE_BITS: u32 = 40;

    /// The position as the `u64` of `first_position` and [`SkippedRecord::position`].
    pub fn pack(self) -> u64 {
        u64::from(self.entry) << Self::LINE_BITS | (self.line & MAX_ENTRY_LINES)
    }

    pub fn unpack(position: u64) -> Self {
        Self {
            entry: (position >> Self::LINE_BITS) as u32,
            line: position & MAX_ENTRY_LINES,
        }
    }
}

// --- The stream of one entry ---------------------------------------------------------------

/// The content of an entry while it is read: counts the bytes and their CRC-32, and
/// holds them to what the directory declares.
struct Checked<R> {
    inner: R,
    crc: flate2::Crc,
    read: u64,
    declared_bytes: u64,
    declared_crc: u32,
    ended: bool,
}

fn damaged_data(what: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, what)
}

impl<R: Read> Read for Checked<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.ended || out.is_empty() {
            return Ok(0);
        }
        let count = self.inner.read(out)?;
        if count == 0 {
            self.ended = true;
            if self.read != self.declared_bytes {
                return Err(damaged_data(format!(
                    "it holds {} bytes, its directory entry declares {}",
                    self.read, self.declared_bytes
                )));
            }
            if self.crc.sum() != self.declared_crc {
                return Err(damaged_data("it fails its checksum".into()));
            }
            return Ok(0);
        }
        self.read += count as u64;
        if self.read > self.declared_bytes {
            self.ended = true;
            return Err(damaged_data(format!(
                "it holds more than the {} bytes its directory entry declares",
                self.declared_bytes
            )));
        }
        self.crc.update(&out[..count]);
        Ok(count)
    }
}

/// Reads into `out` until it is full or the reader ends.
fn read_full(reader: &mut impl Read, out: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < out.len() {
        match reader.read(&mut out[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

/// The first [`TEXT_CHECK_BYTES`] of a reader, or all of it when it is shorter.
fn read_head(reader: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut head = vec![0u8; TEXT_CHECK_BYTES];
    let filled = read_full(reader, &mut head)?;
    head.truncate(filled);
    Ok(head)
}

/// Whether content that starts with `head` is an archive or a compressed file (gzip is
/// judged by the caller).
fn is_nested_archive(head: &[u8]) -> bool {
    const STARTS: [&[u8]; 8] = [
        b"PK\x03\x04",         // zip
        b"PK\x05\x06",         // zip without entries
        b"PK\x07\x08",         // zip in parts
        b"\xfd7zXZ\x00",       // xz
        b"\x28\xb5\x2f\xfd",   // zstd
        b"7z\xbc\xaf\x27\x1c", // 7z
        b"\x04\x22\x4d\x18",   // lz4
        b"\x1f\x9d",           // compress
    ];
    let bzip2 = head.len() >= 10
        && head.starts_with(b"BZh")
        && head[3].is_ascii_digit()
        && head[4..10] == [0x31, 0x41, 0x59, 0x26, 0x53, 0x59];
    let lzip = head.len() >= 5 && head.starts_with(b"LZIP") && head[4] <= 1;
    let tar = head.get(257..262) == Some(b"ustar");
    STARTS.iter().any(|start| head.starts_with(start)) || bzip2 || lzip || tar
}

/// Whether `head` is text: no NUL byte and valid UTF-8. A character cut off at the end
/// of a full head is not held against it.
fn is_text(head: &[u8]) -> bool {
    if head.contains(&0) {
        return false;
    }
    match std::str::from_utf8(head) {
        Ok(_) => true,
        Err(error) => error.error_len().is_none() && head.len() == TEXT_CHECK_BYTES,
    }
}

/// One line as [`EntryLines::next_line`] read it.
struct LineRead {
    /// The bytes it took in the entry, its terminator included.
    source_bytes: u64,
    /// The bytes of the line without its terminator.
    bytes: u64,
    /// Whether the line is in the caller's buffer; `false` when it is longer than the
    /// maximum record length.
    kept: bool,
}

/// The lines of one entry, without their `\n` or `\r\n` terminators.
///
/// A final terminator does not start another line; an empty entry has no lines.
struct EntryLines {
    reader: Box<dyn Read>,
    buffer: Vec<u8>,
    start: usize,
    end: usize,
}

impl EntryLines {
    /// The lines of content that starts with `head` and goes on in `rest`, read
    /// through `buffer` (its content does not matter).
    fn new(head: &[u8], rest: Box<dyn Read>, mut buffer: Vec<u8>) -> Self {
        buffer.resize(READ_BYTES.max(head.len()), 0);
        buffer[..head.len()].copy_from_slice(head);
        Self {
            reader: rest,
            buffer,
            start: 0,
            end: head.len(),
        }
    }

    /// Reads the next line into `line`; `None` at the end of the entry.
    ///
    /// A line of more than `max_bytes` bytes is passed over: `line` is left empty and
    /// only its length is counted, so the memory taken does not follow the line.
    fn next_line(
        &mut self,
        line: &mut Vec<u8>,
        max_bytes: u64,
    ) -> std::io::Result<Option<LineRead>> {
        line.clear();
        // One byte more is kept for a carriage return before the line feed.
        let room = max_bytes as usize + 1;
        let mut read = LineRead {
            source_bytes: 0,
            bytes: 0,
            kept: true,
        };
        let mut last = None;
        loop {
            if self.start == self.end {
                let count = loop {
                    match self.reader.read(&mut self.buffer) {
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                        other => break other?,
                    }
                };
                self.start = 0;
                self.end = count;
                if count == 0 {
                    if read.source_bytes == 0 {
                        return Ok(None);
                    }
                    break;
                }
            }
            let piece = &self.buffer[self.start..self.end];
            let feed = piece.iter().position(|byte| *byte == b'\n');
            let used = feed.map_or(piece.len(), |feed| feed + 1);
            let content = &piece[..feed.unwrap_or(used)];
            if let Some(byte) = content.last() {
                last = Some(*byte);
            }
            read.bytes += content.len() as u64;
            if read.kept {
                if line.len() + content.len() > room {
                    read.kept = false;
                    line.clear();
                } else {
                    line.extend_from_slice(content);
                }
            }
            self.start += used;
            read.source_bytes += used as u64;
            if feed.is_some() {
                break;
            }
        }
        if last == Some(b'\r') {
            read.bytes -= 1;
            if read.kept {
                line.pop();
            }
        }
        if read.kept && read.bytes > max_bytes {
            read.kept = false;
            line.clear();
        }
        Ok(Some(read))
    }
}

/// The entry being read.
struct OpenEntry {
    index: usize,
    lines: EntryLines,
    /// The lines read so far.
    line: u64,
    /// Set when a read of the archive file itself failed under the entry's readers.
    failure: FileFailure,
}

/// The message of a failed read of the archive file, shared between the reader at the
/// bottom of an entry's readers and whoever reads the entry.
type FileFailure = Rc<RefCell<Option<String>>>;

/// The archive file under the readers of an entry. An error of the file itself is
/// noted, so that it can be told from an error of the data (a corrupt stream, a wrong
/// checksum), which the readers above it raise.
struct Watched<R> {
    inner: R,
    failure: FileFailure,
}

impl<R: Read> Read for Watched<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            match self.inner.read(out) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => {
                    *self.failure.borrow_mut() = Some(error.to_string());
                    return Err(error);
                }
                read => return read,
            }
        }
    }
}

/// The lines of an entry whose data starts where `data` stands, or why the entry is
/// not read; and where a failed read of `data` itself is noted.
fn entry_lines(
    data: impl Read + 'static,
    entry: &DirectoryEntry,
    spare: Vec<u8>,
) -> (Result<EntryLines, EntrySkip>, FileFailure) {
    let failure = FileFailure::default();
    let data = Watched {
        inner: data.take(entry.compressed_size),
        failure: Rc::clone(&failure),
    };
    (inflated_lines(data, entry, spare), failure)
}

fn inflated_lines(
    data: impl Read + 'static,
    entry: &DirectoryEntry,
    spare: Vec<u8>,
) -> Result<EntryLines, EntrySkip> {
    let unreadable = |error: std::io::Error| {
        EntrySkip::Damaged(short_reason(format!("it cannot be read: {error}")))
    };
    let buffer = READ_BYTES.min(entry.compressed_size.max(1) as usize);
    let data = BufReader::with_capacity(buffer, data);
    let inflated: Box<dyn Read> = match entry.method {
        0 => Box::new(data),
        _ => Box::new(flate2::bufread::DeflateDecoder::new(data)),
    };
    let mut content = Checked {
        inner: inflated,
        crc: flate2::Crc::new(),
        read: 0,
        declared_bytes: entry.size,
        declared_crc: entry.crc32,
        ended: false,
    };
    let head = read_head(&mut content).map_err(unreadable)?;
    if is_nested_archive(&head) {
        return Err(EntrySkip::NestedArchive);
    }
    let (head, rest): (Vec<u8>, Box<dyn Read>) = if head.starts_with(GZIP_MAGIC) {
        // Members that follow one another are read as one file, as gzip does.
        let outer = BufReader::with_capacity(READ_BYTES / 2, Cursor::new(head).chain(content));
        let mut inner = flate2::bufread::MultiGzDecoder::new(outer);
        let head = read_head(&mut inner).map_err(unreadable)?;
        if is_nested_archive(&head) || head.starts_with(GZIP_MAGIC) {
            return Err(EntrySkip::NestedArchive);
        }
        (head, Box::new(inner))
    } else {
        (head, Box::new(content))
    };
    if !is_text(&head) {
        return Err(EntrySkip::NotText);
    }
    // The byte-order mark of the entry, once. A second U+FEFF is text.
    let text = head.strip_prefix(BYTE_ORDER_MARK).unwrap_or(&head);
    Ok(EntryLines::new(text, rest, spare))
}

/// A reason of at most [`REASON_BYTES`] bytes.
fn short_reason(mut reason: String) -> String {
    if reason.len() > REASON_BYTES {
        let mut end = REASON_BYTES;
        while !reason.is_char_boundary(end) {
            end -= 1;
        }
        reason.truncate(end);
    }
    reason.shrink_to_fit();
    reason
}

// --- The deduplication set -----------------------------------------------------------------

/// The key of a text in the deduplication set: the first 128 bits of its SHA-256.
///
/// Two texts with the same key are taken for one text: the second is counted as an
/// occurrence of the first and is not handed out, without any notice. With a
/// cryptographic hash cut to 128 bits, finding two such texts on purpose takes about
/// 2^64 hash computations (a birthday search), and writing a text that collides with a
/// given one about 2^128; by accident the chance is negligible (about n^2 / 2^129 for
/// n texts). A fast, non-cryptographic hash would let a log line be written to collide
/// at once, which is why it is not used.
fn text_key(text: &str) -> [u8; 16] {
    let digest = Sha256::digest(text.as_bytes());
    let mut key = [0u8; 16];
    key.copy_from_slice(&digest[..16]);
    key
}

/// The most bytes the key file of deduplication on disk can take under a memory limit
/// of `limit_bytes` (a read that keeps occurrence counts, [`ZipInput::spill_into`]).
/// A key (16 bytes) is written for every text the set in memory takes, and none after
/// the set is frozen; every text takes a slot of [`SLOT_BYTES`], and the set never
/// holds more than the memory limit. A bound, not an estimate (task 1.20, review round 2).
pub fn most_key_file_bytes(limit_bytes: u64) -> u64 {
    (limit_bytes / SLOT_BYTES).saturating_mul(16)
}

/// The bytes of a hash table that holds `capacity` keys: its slots are a power of two
/// and at most seven eighths full.
pub(crate) fn table_bytes(capacity: usize) -> u64 {
    let slots: u64 = match capacity as u64 {
        0 => return 0,
        1..4 => 4,
        4..8 => 8,
        capacity => (capacity * 8 / 7).next_power_of_two(),
    };
    slots * SLOT_BYTES + 32
}

// --- The archive ---------------------------------------------------------------------------

/// A zip archive opened for reading its log records in batches.
///
/// - [`ZipInput::open`] checks the settings, hashes the file and reads the directory.
/// - [`ZipInput::batches`] reads the entries; iterate it to its end.
/// - [`ZipInput::finish`] gives the summary and the entry reports.
pub struct ZipInput {
    path: PathBuf,
    file: File,
    file_bytes: u64,
    content_sha256: String,
    patterns: Vec<glob::Pattern>,
    max_record_bytes: u64,
    directory: Vec<DirectoryEntry>,
    reports: Vec<EntryReport>,
    /// What directory and reports are counted as.
    directory_held: u64,
    /// The index of the next entry to open.
    next_entry: usize,
    /// Per distinct text: its key and its occurrences so far.
    counts: HashMap<[u8; 16], u32>,
    skipped: Vec<SkippedRecord>,
    /// What the skipped records and the reasons of damaged entries are counted as.
    skipped_held: u64,
    records: u64,
    plan: Option<BatchPlan>,
    /// The line buffer of the entry read last, for the next one.
    spare: Vec<u8>,
    /// Deduplication on disk, when a directory was named for it, or was made at the
    /// freeze.
    spill: Option<Spill>,
    /// The work directory to make at the freeze and the most texts the set may take,
    /// for a read that uses the disk only when it needs it
    /// ([`ZipInput::spill_when_needed`]); `None` once it is made.
    pending: Option<(PathBuf, Option<u64>)>,
    /// The free-disk check to take at the freeze, and the one that was taken.
    disk_check: Option<FreezeDiskCheck>,
    freeze_disk: Option<FreezeDisk>,
    /// What the batch being read was planned to leave held for the archive: the room
    /// for the set's growth by the batch included.
    reserved_held: u64,
    /// What the caller holds beside the archive for the whole read.
    beside: u64,
}

/// The free-disk check taken when the set of distinct texts is frozen and the read
/// goes on on disk ([`ZipInput::check_disk_at_freeze`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FreezeDiskCheck {
    /// What the run needs on the same disk beside deduplication on disk, each named
    /// (the result parts of `baseline score`).
    pub beside: Vec<(String, u64)>,
    /// The free space to compare with in place of the operating system's figure.
    pub free_bytes: Option<u64>,
}

/// The free-disk check that was taken at the freeze.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreezeDisk {
    /// The work directory whose volume was looked at.
    pub directory: PathBuf,
    /// The free space of that volume at the freeze.
    pub free_bytes: u64,
    /// What the run was estimated to need there: deduplication on disk, and what was
    /// named beside it.
    pub estimated_bytes: u64,
}

impl fmt::Debug for ZipInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZipInput")
            .field("path", &self.path)
            .field("entries", &self.directory.len())
            .field("next_entry", &self.next_entry)
            .field("records", &self.records)
            .field("distinct_texts", &self.counts.len())
            .finish_non_exhaustive()
    }
}

/// How entry patterns match: `*` and `?` match `/` too, so `*.log` selects the logs of
/// every directory.
const PATTERN_OPTIONS: glob::MatchOptions = glob::MatchOptions {
    case_sensitive: true,
    require_literal_separator: false,
    require_literal_leading_dot: false,
};

impl ZipInput {
    /// Opens the archive at `path`: hashes the file and reads its directory.
    ///
    /// The directory is held from here on; what it takes counts against `budget` from
    /// the first batch on ([`ZipInput::held_bytes`]).
    ///
    /// - [`L2dError::InvalidInput`]: a pattern that cannot be used, a maximum record
    ///   length outside 1 to [`MAX_RECORD_BYTES`], a file that is not a zip archive, or
    ///   a directory that is damaged.
    /// - [`L2dError::Missing`]: the file cannot be opened or read.
    /// - [`L2dError::Refused`]: the directory does not fit what `budget` leaves, or
    ///   holds more than [`MAX_ENTRIES`] entries.
    pub fn open(
        path: &Path,
        settings: &ZipInputSettings,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        if settings.max_record_bytes == 0 || settings.max_record_bytes > MAX_RECORD_BYTES {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "the maximum record length must be 1 to {MAX_RECORD_BYTES} bytes, not {}",
                    settings.max_record_bytes
                ),
            });
        }
        let patterns = settings
            .entries
            .iter()
            .map(|pattern| {
                glob::Pattern::new(pattern).map_err(|error| L2dError::InvalidInput {
                    detail: format!("the entry pattern `{pattern}` cannot be used: {error}"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut file = File::open(path).map_err(|error| unreadable(path, &error))?;
        let file_bytes = file
            .metadata()
            .map_err(|error| unreadable(path, &error))?
            .len();
        let content_sha256 = hash_file(&mut file).map_err(|error| unreadable(path, &error))?;
        let entry_bytes =
            (size_of::<DirectoryEntry>() + size_of::<String>() + size_of::<EntryReport>()) as u64;
        let (directory, names, directory_held) =
            read_directory(&mut file, file_bytes, path, entry_bytes, budget)?;
        let reports = names
            .into_iter()
            .map(|path| EntryReport {
                path,
                records: 0,
                skipped: Some(EntrySkip::NotReached),
            })
            .collect();
        Ok(Self {
            path: path.to_path_buf(),
            file,
            file_bytes,
            content_sha256,
            patterns,
            max_record_bytes: settings.max_record_bytes,
            directory,
            reports,
            directory_held,
            next_entry: 0,
            counts: HashMap::new(),
            skipped: Vec::new(),
            skipped_held: 0,
            records: 0,
            plan: None,
            spare: Vec::new(),
            spill: None,
            pending: None,
            disk_check: None,
            freeze_disk: None,
            reserved_held: 0,
            beside: 0,
        })
    }

    /// The batches of the archive, within `budget`: every item is the logs frame of the
    /// texts first seen in one batch (module documentation), or the error that ended
    /// the read. After an error nothing more is handed out.
    ///
    /// [`L2dError::Refused`] when what is held for the archive, or the smallest batch
    /// beside it, does not fit the memory limit; the message states the memory needed
    /// and available. A damaged entry is not an error of the read: it is in the entry
    /// reports.
    pub fn batches<'a>(&'a mut self, budget: &'a mut MemoryBudget) -> ZipBatches<'a> {
        // What the entries still to be read declare: only an estimate of the rows.
        let declared = self.directory[self.next_entry..]
            .iter()
            .zip(&self.reports[self.next_entry..])
            .filter(|(entry, report)| self.selected(entry, &report.path).is_ok())
            .fold(0u64, |sum, (entry, _)| sum.saturating_add(entry.size));
        ZipBatches {
            batcher: Some(Batcher::new(budget, declared, PARSE_COST)),
            input: self,
            accepted: Accepted::new(),
            current: None,
            line: Vec::new(),
            started: false,
            ended: false,
            budget: None,
            merge: None,
        }
    }

    /// How often `text` occurred in the records read so far; 0 when it did not. The
    /// count stops at `u32::MAX`, the largest count a logs frame holds.
    ///
    /// With deduplication on disk this answers for the texts of the set in memory
    /// only (0 for a text on disk): [`ZipInput::occurrence_counts`] gives the counts
    /// of all rows.
    pub fn occurrences(&self, text: &str) -> u64 {
        self.counts
            .get(&text_key(text))
            .map_or(0, |count| u64::from(*count))
    }

    /// What the archive holds between batches, in bytes: the directory and the entry
    /// reports, the readers of one entry with room for a record of the maximum record
    /// length, the deduplication set, and the skipped records.
    pub fn held_bytes(&self) -> u64 {
        self.directory_held
            + self.readers_held()
            + table_bytes(self.counts.capacity())
            + self.skipped_held
            + self.spill_held()
            + self.beside
    }

    /// Names the directory for deduplication on disk (requirement 13.12): when the set
    /// of distinct texts can no longer grow within the memory limit, the read goes on
    /// there instead of being refused. Call it before [`ZipInput::batches`].
    ///
    /// The directory must be absent or empty; it is created, and removed with the
    /// files in it when the archive is finished or dropped. The file names are made up
    /// by the tool.
    ///
    /// The rows handed out are the same with and without it, in the same order. With
    /// it, the rows on disk are handed out after the last entry was read, in frames of
    /// their own, with their occurrence counts over the archive; and
    /// [`ZipInput::occurrence_counts`] gives the counts of all rows.
    ///
    /// `max_set_texts` is the most distinct texts the set in memory may take before
    /// the read goes on on disk; `None` leaves that to the memory limit.
    ///
    /// [`L2dError::Refused`] when the directory holds something, [`L2dError::Runtime`]
    /// when it cannot be created.
    pub fn spill_into(&mut self, dir: &Path, max_set_texts: Option<u64>) -> Result<(), L2dError> {
        self.spill = Some(Spill::create(dir, max_set_texts)?);
        Ok(())
    }

    /// Names the work directory for deduplication on disk without making it (task
    /// 1.20, remediation): while the set of distinct texts fits the memory the read is
    /// the read without a directory, with nothing made, no disk looked at and nothing
    /// more held. When the set is frozen, the directory `work_dir` is made then (with
    /// the files in `dedup/` below it), the free-disk check is taken then
    /// ([`ZipInput::check_disk_at_freeze`]), the buffers of deduplication on disk are
    /// held from then on, and the read goes on on disk with the rows and counts of a
    /// read in memory. Call it before [`ZipInput::batches`].
    ///
    /// Such a read keeps no occurrence counts of all rows
    /// ([`ZipInput::occurrence_counts`] is an error): for a command that wants only the
    /// summary. What was made is removed when the archive is finished or dropped, the
    /// work directory and the absent directories above it as far as they are empty.
    ///
    /// The directory is not checked here: [`crate::parts::WorkDir::check`] does that.
    /// Failing to make it at the freeze ends the read with that error.
    pub fn spill_when_needed(&mut self, work_dir: &Path, max_set_texts: Option<u64>) {
        self.pending = Some((work_dir.to_path_buf(), max_set_texts));
    }

    /// Has the free space of the work directory's volume checked when the set is
    /// frozen, against what deduplication on disk is estimated to write
    /// ([`SelectedBytes::dedup_disk_bytes`]) and what `check` names beside it. A read
    /// whose set fits the memory takes no check.
    ///
    /// The read is then refused at the freeze, before a row is written to disk, with
    /// both figures and what was read until then ([`L2dError::Refused`]).
    pub fn check_disk_at_freeze(&mut self, check: FreezeDiskCheck) {
        self.disk_check = Some(check);
    }

    /// The free-disk check taken at the freeze; `None` when the set was not frozen or
    /// no check was asked for.
    pub fn freeze_disk(&self) -> Option<&FreezeDisk> {
        self.freeze_disk.as_ref()
    }

    /// Tells the archive what the caller holds in memory beside it for the whole read
    /// (a loaded model, an output buffer): it counts as held from the first batch on.
    pub fn hold_beside(&mut self, bytes: u64) {
        self.beside = bytes;
    }

    /// What the entries that will be opened declare, by the directory alone: their
    /// number, their bytes in the archive and their bytes unpacked. A gzip entry
    /// declares the size of its compressed content.
    pub fn selected_bytes(&self) -> SelectedBytes {
        let mut selected = SelectedBytes::default();
        for (entry, report) in self.directory.iter().zip(&self.reports) {
            if self.selected(entry, &report.path).is_ok() {
                selected.entries += 1;
                selected.packed = selected.packed.saturating_add(entry.compressed_size);
                selected.unpacked = selected.unpacked.saturating_add(entry.size);
            }
        }
        selected
    }

    /// What deduplication on disk did; `None` when no directory was named for it or
    /// the set in memory took every text.
    pub fn disk_dedup(&self) -> Option<DiskDedup> {
        self.spill.as_ref().and_then(Spill::report)
    }

    /// The occurrence counts over the archive of the rows handed out so far, in the
    /// order they were handed out, in chunks. Needs a directory for deduplication on
    /// disk ([`ZipInput::spill_into`]): without one the single item is an error.
    pub fn occurrence_counts(&mut self) -> OccurrenceCounts<'_> {
        let files = match &mut self.spill {
            Some(spill) => spill.count_readers(),
            None => Err(L2dError::Runtime {
                detail: "the occurrence counts of the rows are read from the directory for \
                         deduplication on disk, and none was named"
                    .into(),
            }),
        };
        let (keys, counts, failed) = match files {
            Ok((keys, counts)) => (Some(keys), counts, None),
            Err(error) => (None, None, Some(error)),
        };
        OccurrenceCounts {
            set: &self.counts,
            keys,
            counts,
            failed,
        }
    }

    /// Whether the set in memory takes no new text any more.
    fn frozen(&self) -> bool {
        self.spill.as_ref().is_some_and(Spill::frozen)
    }

    /// Whether the read can go on on disk when the set no longer fits: a directory was
    /// named, or will be made, and the set is not frozen yet.
    fn can_freeze(&self) -> bool {
        match &self.spill {
            Some(spill) => !spill.frozen(),
            None => self.pending.is_some(),
        }
    }

    /// The most texts the set may take before the read goes on on disk.
    fn max_set_texts(&self) -> Option<u64> {
        match (&self.spill, &self.pending) {
            (Some(spill), _) => spill.max_set_texts,
            (None, Some((_, most))) => *most,
            (None, None) => None,
        }
    }

    /// Freezes the set in memory, when a directory for deduplication on disk was
    /// named or is to be made: the directory is made now when it is to be made, with
    /// buffers that fit `room` where they can, and the free-disk check is taken now,
    /// before any row goes to disk.
    fn freeze(&mut self, room: u64) -> Result<(), L2dError> {
        if !self.can_freeze() {
            return Ok(());
        }
        if self.spill.is_none()
            && let Some((work_dir, most)) = self.pending.take()
        {
            let work = WorkDir::create(&work_dir)?;
            if let Err(error) = self.check_disk(work.path()) {
                work.remove_created();
                return Err(error);
            }
            self.spill = Some(Spill::create_in(work, most, room)?);
        } else if let Some(spill) = &self.spill {
            let shown = spill.dir().parent().unwrap_or(spill.dir()).to_path_buf();
            self.check_disk(&shown)?;
        }
        if let Some(spill) = &mut self.spill {
            spill.freeze(self.counts.len() as u64);
        }
        Ok(())
    }

    /// The free-disk check of the freeze, when one was asked for: the volume of `dir`.
    fn check_disk(&mut self, dir: &Path) -> Result<(), L2dError> {
        let Some(check) = &self.disk_check else {
            return Ok(());
        };
        let dedup = self.selected_bytes().dedup_disk_bytes();
        let mut needed: Vec<(&str, u64)> = check
            .beside
            .iter()
            .map(|(what, bytes)| (what.as_str(), *bytes))
            .collect();
        needed.push(("deduplication on disk", dedup));
        let free = match check.free_bytes {
            Some(free) => free,
            None => free_disk_bytes(dir)?,
        };
        let progress = format!(
            "the deduplication set of {} distinct texts no longer fits the memory limit, and the \
             read was to go on on disk; stopped after {} records of {} entries",
            self.counts.len(),
            self.records,
            self.next_entry,
        );
        require_free_in(dir, free, &needed, &progress)?;
        let estimated = needed
            .iter()
            .fold(0u64, |sum, (_, bytes)| sum.saturating_add(*bytes));
        self.freeze_disk = Some(FreezeDisk {
            directory: dir.to_path_buf(),
            free_bytes: free,
            estimated_bytes: estimated,
        });
        Ok(())
    }

    fn spill_held(&self) -> u64 {
        self.spill.as_ref().map_or(0, Spill::held)
    }

    /// The batch plan of the read, once [`ZipInput::batches`] has been iterated to its
    /// end: what was observed and chosen, and how the batch size moved.
    pub fn batch_plan(&self) -> Option<&BatchPlan> {
        self.plan.as_ref()
    }

    /// The summary of what was read and the report of every entry.
    ///
    /// Before the batches are at their end this is the summary of the records read so
    /// far, and the entries not read to their end are [`EntrySkip::NotReached`].
    pub fn finish(self) -> ArchiveSummary {
        ArchiveSummary {
            input: InputSummary {
                source_path: self.path,
                records: self.records,
                distinct_texts: self.counts.len() as u64
                    + self.spill.as_ref().map_or(0, Spill::spilled_distinct),
                content_sha256: self.content_sha256,
                skipped: self.skipped,
                label_conflicts: 0,
            },
            entries: self.reports,
        }
    }

    fn readers_held(&self) -> u64 {
        STREAM_BYTES + RECORD_FACTOR * self.max_record_bytes + PARSE_ROW_BYTES
    }

    /// The table the set must grow to before it can take `rows` more texts, in bytes;
    /// 0 when they fit as it is.
    fn growth_bytes(&self, rows: u64) -> u64 {
        if self.frozen() {
            return 0;
        }
        let wanted = self.counts.len() as u64 + rows;
        if wanted <= self.counts.capacity() as u64 {
            return 0;
        }
        table_bytes(wanted.max(self.counts.capacity() as u64 + 1) as usize)
    }

    /// The most rows the next batch can have within `budget`: the largest batch, or
    /// what the memory left can hold at the fixed cost of a row when that is less.
    fn next_rows(&self, budget: &MemoryBudget) -> u64 {
        let left = budget
            .usable_bytes()
            .saturating_sub(self.held_bytes() + PARSE_BATCH_BYTES);
        budget.max_batch_rows().min(left / PARSE_ROW_BYTES)
    }

    /// What is held with the room for the set's growth by a batch of `rows` rows.
    fn held_with_growth(&self, rows: u64) -> u64 {
        self.held_bytes() + self.growth_bytes(rows)
    }

    /// Checks that what is held, with the room for the set's growth by the next batch,
    /// leaves the fixed memory of a batch, and returns it.
    ///
    /// [`L2dError::Refused`] otherwise, with the memory needed and available
    /// (requirement 13.12).
    fn require_held(&mut self, budget: &MemoryBudget, smallest: u64) -> Result<u64, L2dError> {
        let rows = self.next_rows(budget);
        let mut held = self.held_with_growth(rows);
        if held + PARSE_BATCH_BYTES + smallest > budget.usable_bytes()
            && self.growth_bytes(rows) > 0
        {
            // The set cannot grow for the next batch and leave room for the smallest
            // batch (`smallest` bytes): with a directory for it, the deduplication
            // goes on on disk and the set stays as it is. A directory to be made is
            // made now, between batches, and its buffers are held from now on: what
            // is left beside the smallest batch is the room for them.
            let room = budget
                .usable_bytes()
                .saturating_sub(self.held_bytes() + PARSE_BATCH_BYTES + smallest);
            self.freeze(room)?;
            held = self.held_with_growth(rows);
        }
        let needed = held + PARSE_BATCH_BYTES;
        if needed <= budget.usable_bytes() {
            self.reserved_held = held;
            return Ok(held);
        }
        Err(L2dError::Refused {
            reason: format!(
                "reading the archive needs {} before its next record: {}, and the fixed \
                 memory of a batch {}; but the memory limit of {} leaves {} beside the {} \
                 kept for the tool itself; stopped after {} records of {} entries",
                exact(needed),
                self.held_parts(rows),
                exact(PARSE_BATCH_BYTES),
                exact(budget.limit_bytes()),
                exact(budget.usable_bytes()),
                exact(PROCESS_BASELINE_BYTES),
                self.records,
                self.next_entry,
            ),
        })
    }

    /// What is held, part by part, for a refusal.
    fn held_parts(&self, rows: u64) -> String {
        format!(
            "the deduplication set of {} distinct texts {} (and {} for its next growth), {} \
             skipped records and damaged entries {}, the directory of {} entries {}, the \
             readers of one entry with a record of the maximum record length of {} bytes {}",
            self.counts.len(),
            exact(table_bytes(self.counts.capacity())),
            exact(self.growth_bytes(rows)),
            self.skipped.len(),
            exact(self.skipped_held),
            self.directory.len(),
            exact(self.directory_held),
            self.max_record_bytes,
            exact(self.readers_held()),
        ) + &match &self.spill {
            Some(spill) if spill.held() == SPILL_HELD => format!(
                ", the buffers of deduplication on disk and of its occurrence counts {}",
                exact(SPILL_HELD)
            ),
            Some(spill) => format!(
                ", the buffers of deduplication on disk {}",
                exact(spill.held())
            ),
            None => String::new(),
        } + &match self.beside {
            0 => String::new(),
            beside => format!(", held beside the archive {}", exact(beside)),
        }
    }

    /// A refusal of the batcher, with what the archive holds added to it.
    fn explained(&self, error: L2dError, budget: &MemoryBudget) -> L2dError {
        let rows = self.next_rows(budget);
        match error {
            L2dError::Refused { reason } => L2dError::Refused {
                reason: format!(
                    "{reason}; held for the archive: {}; {} records of {} entries were read",
                    self.held_parts(rows),
                    self.records,
                    self.next_entry
                ),
            },
            other => other,
        }
    }

    /// Whether the entry is one to open, by its directory entry and its path alone.
    fn selected(&self, entry: &DirectoryEntry, path: &str) -> Result<(), EntrySkip> {
        if !self.patterns.is_empty()
            && !self
                .patterns
                .iter()
                .any(|pattern| pattern.matches_with(path, PATTERN_OPTIONS))
        {
            return Err(EntrySkip::NoPatternMatch);
        }
        match entry.kind {
            EntryKind::Directory => return Err(EntrySkip::Directory),
            EntryKind::Link => return Err(EntrySkip::Link),
            EntryKind::File => {}
        }
        if entry.encrypted() {
            return Err(EntrySkip::Encrypted);
        }
        if !matches!(entry.method, 0 | 8) {
            return Err(EntrySkip::UnsupportedMethod(entry.method));
        }
        Ok(())
    }

    /// The failure of reading the archive file itself while the entry at `index` was
    /// opened or read: not damage of the entry, an I/O failure of the run.
    fn read_failed(&self, index: usize, error: &dyn fmt::Display) -> L2dError {
        L2dError::Runtime {
            detail: format!(
                "reading {} failed in entry {} of {} after {} records: {error}",
                self.path.display(),
                index + 1,
                self.directory.len(),
                self.records
            ),
        }
    }

    /// Opens the entry at `index` as lines, or says why it is not read (`Ok(Err(..))`).
    ///
    /// [`L2dError::Runtime`] when the archive file itself cannot be read: that is a
    /// failure of the run, not a damaged entry.
    fn open_entry(&mut self, index: usize) -> Result<Result<OpenEntry, EntrySkip>, L2dError> {
        let entry = &self.directory[index];
        if let Err(skip) = self.selected(entry, &self.reports[index].path) {
            return Ok(Err(skip));
        }
        let data_at = match locate_data(&mut self.file, entry, self.file_bytes) {
            Ok(data_at) => data_at,
            Err(DataFault::Damaged(reason)) => {
                return Ok(Err(EntrySkip::Damaged(short_reason(reason))));
            }
            Err(DataFault::Io(error)) => return Err(self.read_failed(index, &error)),
        };
        // A handle of its own for the entry's readers; it shares the file with the
        // archive's, which is not used while the entry is open.
        let data = self
            .file
            .try_clone()
            .and_then(|mut data| data.seek(SeekFrom::Start(data_at)).map(|_| data))
            .map_err(|error| self.read_failed(index, &error))?;
        let (lines, failure) = entry_lines(data, entry, std::mem::take(&mut self.spare));
        if let Some(error) = failure.borrow().as_ref() {
            return Err(self.read_failed(index, error));
        }
        Ok(lines.map(|lines| OpenEntry {
            index,
            lines,
            line: 0,
            failure,
        }))
    }

    /// Notes why the entry at `index` was not read, or not to its end.
    fn skip_entry(&mut self, index: usize, skip: EntrySkip) {
        if let EntrySkip::Damaged(reason) = &skip {
            self.skipped_held += reason.capacity() as u64 + 16;
        }
        self.reports[index].skipped = Some(skip);
    }

    /// Notes that the record at `position` was skipped, and why.
    fn skip_record(&mut self, position: u64, reason: impl Into<String>) {
        let mut reason: String = reason.into();
        reason.shrink_to_fit();
        self.skipped_held += SKIPPED_RECORD_BYTES + reason.capacity() as u64;
        self.skipped.push(SkippedRecord { position, reason });
    }

    /// Has Polars parse the accepted lines of one batch, counts their texts in the
    /// set, and returns the texts that are new, as a logs frame; `None` when none is.
    fn merge(
        &mut self,
        accepted: Accepted,
        budget: &MemoryBudget,
        smallest: u64,
    ) -> Result<Option<LogsFrame>, L2dError> {
        let failed = |error| frame_error("reading a batch of the archive", &error);
        let text = parse_csv_records(accepted.raw(), 1, &[0], NO_SEPARATOR, None)?
            .column(&csv_field_name(0))
            .map_err(failed)?
            .clone();
        let rows = text.len();
        let label = Column::new("label".into(), vec![Label::Unknown.as_str(); rows]);
        let group = Column::full_null("group".into(), rows, &DataType::String);
        // One row per distinct text of the batch, in order of first occurrence.
        let batch = combine(accepted.rows(text, label, group)?)?;

        let texts = batch
            .column(LogsFrame::TEXT)
            .and_then(Column::str)
            .map_err(failed)?;
        let occurrences = batch
            .column(LogsFrame::OCCURRENCES)
            .and_then(Column::u32)
            .map_err(failed)?;
        // The keys once (16 bytes a row, within the fixed cost of a row), so that the
        // set grows for the texts that are new and not for every row of the batch.
        let keys: Vec<[u8; 16]> = texts
            .iter()
            .map(|text| text.map(text_key))
            .collect::<Option<_>>()
            .ok_or_else(|| L2dError::Runtime {
                detail: "a batch of the archive holds a row without text".into(),
            })?;
        let new = keys
            .iter()
            .filter(|key| !self.counts.contains_key(*key))
            .count();
        if self.can_freeze() {
            let over_most = self
                .max_set_texts()
                .is_some_and(|most| (self.counts.len() + new) as u64 > most);
            let growth = self.growth_bytes(new as u64);
            let held = self.held_bytes();
            let usable = budget.usable_bytes();
            // The room for buffers made now: what is left beside the smallest batch.
            let between = usable.saturating_sub(held + PARSE_BATCH_BYTES + smallest);
            let freeze = match self.spill {
                // While it grows the old and the new table exist together; after it
                // the smallest batch must still fit.
                Some(_) => (over_most
                    || (growth > 0 && held + growth + PARSE_BATCH_BYTES + smallest > usable))
                    .then_some(between),
                // No directory yet, and this batch was planned without its buffers.
                // While the set can grow it grows, as without a directory, and the
                // freeze waits for the end of the batch ([`ZipInput::require_held`]).
                // When it cannot, the buffers take the room that was kept for its
                // growth by this batch, when the smallest of them fit there; otherwise
                // the growth is refused as without a directory.
                None => {
                    let spare = self.reserved_held.saturating_sub(held);
                    if over_most {
                        Some(between)
                    } else if growth > 0
                        && held + growth > usable
                        && SPILL_HELD_WITHOUT_COUNTS_LEAST <= spare
                    {
                        Some(spare)
                    } else {
                        None
                    }
                }
            };
            if let Some(room) = freeze {
                self.freeze(room)?;
                // What is held now, the buffers of a directory made now included,
                // must leave the fixed memory of a batch.
                self.require_held(budget, 0)?;
            }
        }
        let (held, records, entries) = (self.held_bytes(), self.records, self.next_entry);
        if let Some(spill) = &mut self.spill
            && spill.frozen()
        {
            // The set takes no new text: the rows it does not hold go to disk. What a
            // partition's set may take when the partitions are deduplicated is what
            // the limit leaves now, or less.
            let room = budget
                .usable_bytes()
                .saturating_sub(held + PARSE_BATCH_BYTES);
            let positions = batch
                .column(LogsFrame::FIRST_POSITION)
                .and_then(Column::u64)
                .map_err(failed)?;
            let rows = keys
                .iter()
                .zip(texts.iter())
                .zip(positions.iter().zip(occurrences.iter()));
            for ((key, text), (position, count)) in rows {
                let (Some(text), Some(position), Some(count)) = (text, position, count) else {
                    return Err(L2dError::Runtime {
                        detail: "a batch of the archive holds a row without text, position or \
                                 count"
                            .into(),
                    });
                };
                match self.counts.get_mut(key) {
                    Some(seen) => *seen = seen.saturating_add(count),
                    None => spill.spill(key, position, count, text)?,
                }
            }
            // The batch is let go: its memory is the room, and a partition whose
            // distinct keys may no longer fit it is counted there.
            drop(keys);
            drop(batch);
            spill.check_partitions(room).map_err(|error| match error {
                L2dError::Refused { reason } => L2dError::Refused {
                    reason: format!(
                        "{reason}; stopped after {records} records of {entries} entries"
                    ),
                },
                other => other,
            })?;
            return Ok(None);
        }
        self.grow_for(new as u64, budget)?;
        let mut is_new = Vec::with_capacity(batch.height());
        for (key, count) in keys.into_iter().zip(occurrences.iter()) {
            let Some(count) = count else {
                return Err(L2dError::Runtime {
                    detail: "a batch of the archive holds a row without a count".into(),
                });
            };
            match self.counts.entry(key) {
                Entry::Occupied(mut seen) => {
                    *seen.get_mut() = seen.get().saturating_add(count);
                    is_new.push(false);
                }
                Entry::Vacant(new) => {
                    // For the counts over the archive, read out in row order later.
                    if let Some(spill) = &mut self.spill {
                        spill.note_key(new.key())?;
                    }
                    new.insert(count);
                    is_new.push(true);
                }
            }
        }
        if !is_new.contains(&true) {
            return Ok(None);
        }
        let new = batch
            .filter(&BooleanChunked::from_slice("new".into(), &is_new))
            .map_err(failed)?;
        let column = |name: &str| new.column(name).cloned().map_err(failed);
        let frame = DataFrame::new(vec![
            column(LogsFrame::TEXT)?,
            Column::new(
                LogsFrame::LABEL.into(),
                vec![Label::Unknown.as_str(); new.height()],
            ),
            Column::full_null(LogsFrame::GROUP.into(), new.height(), &DataType::String),
            column(LogsFrame::FIRST_POSITION)?,
            column(LogsFrame::OCCURRENCES)?,
        ])
        .map_err(failed)?;
        // The texts point into the buffers of the parse; own copies let go of them.
        LogsFrame::new(own_strings(frame)?).map(Some)
    }

    /// Deduplicates the rows on disk, partition by partition, and opens the result
    /// for merging; `None` without a directory for it.
    fn deduplicate_on_disk(
        &mut self,
        budget: &mut MemoryBudget,
    ) -> Result<Option<Merge>, L2dError> {
        let held = self.held_bytes();
        budget.hold(held.saturating_add(PARSE_BATCH_BYTES));
        let room = budget.left_bytes();
        match &mut self.spill {
            Some(spill) => {
                spill.deduplicate(room)?;
                spill.merge().map(Some)
            }
            None => Ok(None),
        }
    }

    /// The next frame of the rows on disk, in order of first occurrence, with their
    /// occurrence counts over the archive; `None` at their end.
    ///
    /// A frame is cut like a batch: at most the largest batch of rows, and no more
    /// text than a batch of that text would be allowed to hold.
    fn merged_frame(
        &mut self,
        merge: &mut Merge,
        budget: &mut MemoryBudget,
    ) -> Result<Option<LogsFrame>, L2dError> {
        let held = self.require_held(budget, 0)?;
        budget.hold(held.saturating_add(PARSE_BATCH_BYTES));
        let room = budget.left_bytes();
        let max_rows = budget.max_batch_rows().min(room / PARSE_ROW_BYTES);
        let max_bytes = (room as f64 / PARSE_EXPANSION) as u64;
        let Some((texts, positions, counts)) = merge.next_rows(max_rows, max_bytes)? else {
            return Ok(None);
        };
        if let Some(spill) = &mut self.spill {
            spill.note_counts(&counts)?;
        }
        let rows = texts.len();
        let text = Column::new(LogsFrame::TEXT.into(), &texts);
        drop(texts);
        let frame = DataFrame::new(vec![
            text,
            Column::new(LogsFrame::LABEL.into(), vec![Label::Unknown.as_str(); rows]),
            Column::full_null(LogsFrame::GROUP.into(), rows, &DataType::String),
            Column::new(LogsFrame::FIRST_POSITION.into(), positions),
            Column::new(LogsFrame::OCCURRENCES.into(), counts),
        ])
        .map_err(|error| frame_error("reading the rows deduplicated on disk", &error))?;
        LogsFrame::new(frame).map(Some)
    }

    /// Makes room in the set for `rows` more texts.
    ///
    /// The room was taken off what the batch could use before the batch was read
    /// ([`ZipInput::held_with_growth`]); this is the check that it is really there.
    fn grow_for(&mut self, rows: u64, budget: &MemoryBudget) -> Result<(), L2dError> {
        let growth = self.growth_bytes(rows);
        if growth == 0 {
            return Ok(());
        }
        let needed = self.held_bytes() + growth;
        let refused = |this: &Self| L2dError::Refused {
            reason: format!(
                "the deduplication set of {} distinct texts must grow to take {rows} more and \
                 needs {} for that, but the memory limit of {} leaves {} beside the {} kept for \
                 the tool itself; held for the archive: {}; stopped after {} records of {} \
                 entries",
                this.counts.len(),
                exact(needed),
                exact(budget.limit_bytes()),
                exact(budget.usable_bytes()),
                exact(PROCESS_BASELINE_BYTES),
                this.held_parts(rows),
                this.records,
                this.next_entry,
            ),
        };
        if needed > budget.usable_bytes() {
            return Err(refused(self));
        }
        self.counts
            .try_reserve(rows as usize)
            .map_err(|_| refused(self))
    }
}

/// The lower-case hexadecimal SHA-256 of the whole file.
fn hash_file(file: &mut File) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    let mut piece = vec![0u8; HASH_READ_BYTES];
    file.seek(SeekFrom::Start(0))?;
    loop {
        match file.read(&mut piece) {
            Ok(0) => break,
            Ok(count) => hasher.update(&piece[..count]),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(hex(&hasher.finalize()))
}

// --- Occurrence counts ---------------------------------------------------------------------

/// What the entries that will be opened declare: see [`ZipInput::selected_bytes`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelectedBytes {
    pub entries: u64,
    /// Their bytes in the archive.
    pub packed: u64,
    /// Their bytes unpacked, as the directory declares them.
    pub unpacked: u64,
}

impl SelectedBytes {
    /// What deduplication on disk is estimated to write for these entries, in bytes:
    /// their unpacked bytes, and the 32 bytes a row on disk takes beside its text for
    /// as many rows as records of 6 bytes would make of them.
    ///
    /// It is an estimate, not a bound: the rows of an archive are not known before it
    /// is read (records shorter than 6 bytes on average take more), a gzip entry
    /// declares its compressed size only, and a text is written once per batch it
    /// occurs in. For logs it is high: a text that the set in memory holds is not
    /// written at all.
    pub fn dedup_disk_bytes(&self) -> u64 {
        estimated_disk_bytes(self.unpacked)
    }
}

/// The occurrence counts of the rows handed out: see [`ZipInput::occurrence_counts`].
pub struct OccurrenceCounts<'a> {
    set: &'a HashMap<[u8; 16], u32>,
    /// The keys of the rows handed out during the read.
    keys: Option<CountFile>,
    /// The counts of the rows handed out from disk.
    counts: Option<CountFile>,
    failed: Option<L2dError>,
}

impl Iterator for OccurrenceCounts<'_> {
    type Item = Result<Vec<u32>, L2dError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(error) = self.failed.take() {
            return Some(Err(error));
        }
        let set = self.set;
        let from_keys = self.keys.as_mut().map(|keys| {
            keys.next_chunk().map(|chunk| {
                chunk
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(|key| set.get(key).copied().unwrap_or(0))
                    .collect::<Vec<u32>>()
            })
        });
        let from_counts = |file: &mut CountFile| {
            file.next_chunk().map(|chunk| {
                chunk
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|count| u32::from_le_bytes(*count))
                    .collect::<Vec<u32>>()
            })
        };
        let chunk = match from_keys {
            Some(Ok(counts)) if !counts.is_empty() => return Some(Ok(counts)),
            Some(Err(error)) => Err(error),
            _ => {
                self.keys = None;
                match self.counts.as_mut().map(from_counts) {
                    Some(Ok(counts)) if !counts.is_empty() => return Some(Ok(counts)),
                    Some(Err(error)) => Err(error),
                    _ => Ok(()),
                }
            }
        };
        self.keys = None;
        self.counts = None;
        chunk.err().map(Err)
    }
}

// --- The batches ---------------------------------------------------------------------------

/// The batches of an archive: see [`ZipInput::batches`].
pub struct ZipBatches<'a> {
    input: &'a mut ZipInput,
    /// `None` once the read has ended and the plan is with the archive.
    batcher: Option<Batcher<'a>>,
    accepted: Accepted,
    current: Option<OpenEntry>,
    /// The line being read; reused.
    line: Vec<u8>,
    started: bool,
    ended: bool,
    /// The budget, back from the batcher, while the rows on disk are handed out.
    budget: Option<&'a mut MemoryBudget>,
    /// The rows on disk, merged by position, once the last entry was read.
    merge: Option<Merge>,
}

impl Iterator for ZipBatches<'_> {
    type Item = Result<LogsFrame, L2dError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }
        match self.advance() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => {
                self.ended = true;
                None
            }
            Err(error) => {
                self.ended = true;
                self.current = None;
                Some(Err(error))
            }
        }
    }
}

impl ZipBatches<'_> {
    /// The path of the entry with the index `entry` in the directory, as a position
    /// names it ([`ArchivePosition::entry`]).
    pub fn entry_path(&self, entry: u32) -> Option<&str> {
        self.input
            .reports
            .get(entry as usize)
            .map(|report| report.path.as_str())
    }

    /// Reads on until a batch with new texts is complete; `None` at the end of the
    /// archive.
    fn advance(&mut self) -> Result<Option<LogsFrame>, L2dError> {
        let Some(batcher) = &mut self.batcher else {
            return self.merged();
        };
        if !self.started {
            self.started = true;
            // What the archive holds counts from the first batch on.
            let held = self.input.require_held(batcher.budget(), 0)?;
            batcher
                .processed(held)
                .map_err(|error| self.input.explained(error, batcher.budget()))?;
        }
        loop {
            let Some(entry) = &mut self.current else {
                let index = self.input.next_entry;
                if index == self.input.directory.len() {
                    return self.end();
                }
                self.input.next_entry += 1;
                match self.input.open_entry(index)? {
                    Ok(entry) => self.current = Some(entry),
                    Err(skip) => self.input.skip_entry(index, skip),
                }
                continue;
            };
            let index = entry.index;
            let read = if entry.line == MAX_ENTRY_LINES {
                Err(format!("it has more than {MAX_ENTRY_LINES} lines"))
            } else {
                entry
                    .lines
                    .next_line(&mut self.line, self.input.max_record_bytes)
                    .map_err(|error| format!("reading stopped after line {}: {error}", entry.line))
            };
            let read = match read {
                Ok(Some(read)) => read,
                Ok(None) => {
                    self.input.reports[index].skipped = None;
                    self.close();
                    continue;
                }
                Err(reason) => {
                    // The file could not be read: the run fails. Anything else is the
                    // entry's data.
                    if let Some(error) = entry.failure.borrow().as_ref() {
                        return Err(self.input.read_failed(index, error));
                    }
                    self.input
                        .skip_entry(index, EntrySkip::Damaged(short_reason(reason)));
                    self.close();
                    continue;
                }
            };
            entry.line += 1;
            let position = ArchivePosition {
                entry: index as u32,
                line: entry.line,
            }
            .pack();
            self.input.records += 1;
            self.input.reports[index].records += 1;
            // What follows is the validation pass of the lines adapter.
            let source_bytes = if !read.kept {
                self.input.skip_record(
                    position,
                    format!(
                        "record of {} bytes is longer than the maximum record length of {} bytes",
                        read.bytes, self.input.max_record_bytes
                    ),
                );
                0
            } else {
                match std::str::from_utf8(&self.line) {
                    Ok(text) if text.trim().is_empty() => {
                        self.input.skip_record(position, "empty line");
                    }
                    Ok(text) => {
                        self.accepted.record(
                            position,
                            row_digest(text, Label::Unknown, None),
                            Label::Unknown,
                        );
                        self.accepted.push_raw(&self.line);
                        // Polars removes one carriage return before the line feed. A
                        // text that itself ends in one gets a second one to lose.
                        self.accepted.push_raw(if self.line.ends_with(b"\r") {
                            b"\r\n"
                        } else {
                            b"\n"
                        });
                    }
                    Err(_) => self.input.skip_record(position, "line is not valid UTF-8"),
                }
                read.source_bytes
            };
            let Some(batcher) = &mut self.batcher else {
                return Ok(None);
            };
            let complete = batcher
                .record(source_bytes)
                .map_err(|error| self.input.explained(error, batcher.budget()))?;
            if complete && let Some(batch) = self.process(false)? {
                return Ok(Some(batch));
            }
        }
    }

    /// Closes the entry being read and keeps its line buffer for the next one.
    fn close(&mut self) {
        if let Some(entry) = self.current.take() {
            self.input.spare = entry.lines.buffer;
        }
    }

    /// Processes the batch that is complete; the frame of its new texts, if any.
    /// After the `last` batch no room is asked for a next one.
    fn process(&mut self, last: bool) -> Result<Option<LogsFrame>, L2dError> {
        let Some(batcher) = &mut self.batcher else {
            return Ok(None);
        };
        let accepted = std::mem::take(&mut self.accepted);
        let batch = if accepted.is_empty() {
            None
        } else {
            self.input
                .merge(accepted, batcher.budget(), batcher.smallest_batch_bytes())?
        };
        let held = if last {
            self.input.held_bytes()
        } else {
            self.input
                .require_held(batcher.budget(), batcher.smallest_batch_bytes())?
        };
        batcher
            .processed(held)
            .map_err(|error| self.input.explained(error, batcher.budget()))?;
        Ok(batch)
    }

    /// The last, partial batch, and the plan of the read. With rows on disk, their
    /// deduplication, and their first frame when the last batch has no new text.
    fn end(&mut self) -> Result<Option<LogsFrame>, L2dError> {
        let Some(batcher) = &mut self.batcher else {
            return Ok(None);
        };
        let last = batcher
            .end()
            .map_err(|error| self.input.explained(error, batcher.budget()))?;
        let batch = if last { self.process(true)? } else { None };
        if let Some(batcher) = self.batcher.take() {
            let (plan, budget) = batcher.finish()?;
            self.input.plan = Some(plan);
            if self.input.frozen() {
                self.merge = self.input.deduplicate_on_disk(budget)?;
                self.budget = Some(budget);
            }
        }
        match batch {
            Some(batch) => Ok(Some(batch)),
            None => self.merged(),
        }
    }

    /// The next frame of the rows on disk; `None` when there are none, or no more.
    fn merged(&mut self) -> Result<Option<LogsFrame>, L2dError> {
        match (&mut self.merge, &mut self.budget) {
            (Some(merge), Some(budget)) => self.input.merged_frame(merge, budget),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key file's bound covers a set that fills the whole limit: one 16-byte key
    /// per slot the limit can hold, and a full table never holds more texts than slots.
    #[test]
    fn the_key_file_bound_covers_a_set_that_fills_the_limit() {
        assert_eq!(most_key_file_bytes(0), 0);
        assert_eq!(most_key_file_bytes(21 * 1000), 16 * 1000);
        for limit in [1u64 << 20, 50_000_000, 2_000_000_000] {
            // The most texts a table within `limit` holds: seven eighths of its slots.
            let mut capacity = 1usize;
            while table_bytes(capacity * 2) <= limit {
                capacity *= 2;
            }
            let texts = (table_bytes(capacity) - 32) / SLOT_BYTES * 7 / 8;
            assert!(16 * texts <= most_key_file_bytes(limit), "{limit}");
        }
        assert_eq!(most_key_file_bytes(u64::MAX), u64::MAX / SLOT_BYTES * 16);
    }

    /// Gives out `good`, then fails as a disk would.
    struct Failing {
        good: Cursor<Vec<u8>>,
    }

    impl Read for Failing {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            match self.good.read(out)? {
                0 => Err(std::io::Error::other("the device is not ready")),
                count => Ok(count),
            }
        }
    }

    fn stored(bytes: u64, crc32: u32) -> DirectoryEntry {
        DirectoryEntry {
            method: 0,
            flags: 0,
            crc32,
            compressed_size: bytes,
            size: bytes,
            header_at: 0,
            kind: EntryKind::File,
        }
    }

    /// Reads the lines of an entry to their end or to the first error.
    fn read_out(lines: &mut EntryLines) -> (u64, Option<std::io::Error>) {
        let mut line = Vec::new();
        let mut count = 0;
        loop {
            match lines.next_line(&mut line, DEFAULT_MAX_RECORD_BYTES) {
                Ok(Some(_)) => count += 1,
                Ok(None) => return (count, None),
                Err(error) => return (count, Some(error)),
            }
        }
    }

    #[test]
    fn a_failed_read_of_the_file_is_told_from_damaged_data() {
        let content = "a line of the entry\n".repeat(1_000);

        // The file fails after 15,000 of the 20,000 bytes of a stored entry.
        let failing = Failing {
            good: Cursor::new(content.as_bytes()[..15_000].to_vec()),
        };
        let (lines, failure) = entry_lines(failing, &stored(20_000, 0), Vec::new());
        let (read, error) = read_out(&mut lines.unwrap());
        assert_eq!(read, 750);
        assert!(error.is_some());
        assert_eq!(
            failure.borrow().as_deref(),
            Some("the device is not ready"),
            "the failure of the file is noted"
        );

        // The same under an inflater, in the first bytes: opening fails, and it is
        // the file's failure, not the entry's.
        let mut deflated = stored(20_000, 0);
        deflated.method = 8;
        let failing = Failing {
            good: Cursor::new(Vec::new()),
        };
        let (lines, failure) = entry_lines(failing, &deflated, Vec::new());
        assert!(matches!(lines, Err(EntrySkip::Damaged(_))));
        assert!(failure.borrow().is_some());

        // Data that ends early or fails its checksum is the entry's damage: nothing
        // is noted for the file.
        let whole = Cursor::new(content.clone().into_bytes());
        let (lines, failure) = entry_lines(whole, &stored(20_000, 0x1234_5678), Vec::new());
        let (read, error) = read_out(&mut lines.unwrap());
        assert_eq!(read, 1_000);
        assert!(error.unwrap().to_string().contains("checksum"));
        assert!(failure.borrow().is_none());

        let short = Cursor::new(content.as_bytes()[..15_000].to_vec());
        let (lines, failure) = entry_lines(short, &stored(20_000, 0), Vec::new());
        let (_, error) = read_out(&mut lines.unwrap());
        assert!(error.unwrap().to_string().contains("15000"));
        assert!(failure.borrow().is_none());
    }

    #[test]
    fn the_table_model_follows_the_capacities_the_map_gives() {
        // The capacity the standard map gives for a request says how many slots it has;
        // the model must put a request and that capacity in the same table.
        for wanted in [
            1usize, 3, 4, 7, 8, 14, 15, 28, 29, 1_000, 65_536, 458_752, 458_753,
        ] {
            let mut map: HashMap<[u8; 16], u32> = HashMap::new();
            map.try_reserve(wanted).unwrap();
            let capacity = map.capacity();
            assert!(capacity >= wanted);
            assert_eq!(
                table_bytes(capacity),
                table_bytes(wanted),
                "{wanted} texts: the model gives the table the map allocates"
            );
        }
        assert_eq!(table_bytes(0), 0);
        assert_eq!(table_bytes(7_340_032), (1 << 23) * SLOT_BYTES + 32);
        assert_eq!(table_bytes(7_340_033), (1 << 24) * SLOT_BYTES + 32);
    }

    /// The figures `data/README.md` gives for the archive; on request.
    #[test]
    #[ignore = "on request: reads the directory of data/ait-lds-v2/russellmitchell.zip"]
    fn the_directory_of_the_ait_archive_is_counted() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
            .join("ait-lds-v2")
            .join("russellmitchell.zip");
        let mut file = File::open(&path).unwrap();
        let file_bytes = file.metadata().unwrap().len();
        let budget = MemoryBudget::observe(&crate::memory::MemorySettings::default()).unwrap();
        let (directory, names, _) =
            read_directory(&mut file, file_bytes, &path, 0, &budget).unwrap();
        let of = |kind: EntryKind| directory.iter().filter(move |entry| entry.kind == kind);
        let files = of(EntryKind::File).count();
        let non_empty = of(EntryKind::File).filter(|entry| entry.size > 0).count();
        let bytes: u64 = of(EntryKind::File).map(|entry| entry.size).sum();
        eprintln!(
            "{} directory entries: {} directories, {} links, {files} files of which {non_empty} \
             are not empty; {bytes} bytes unpacked; {} names",
            directory.len(),
            of(EntryKind::Directory).count(),
            of(EntryKind::Link).count(),
            names.len()
        );
    }
}
