//! The directory of a zip archive: the end record, the zip64 end record, the central
//! directory entries, and where the data of an entry starts.
//!
//! This is the tool's own reader, not the `zip` crate's (deviation from the task text,
//! see the task notes). The crate reads the whole directory into memory before the
//! caller sees an entry, sizes that allocation from the entry count the archive
//! declares, and falls back to earlier end records when the last one does not parse;
//! none of that can be held to a memory limit from outside. Here every figure an
//! archive declares is checked against the file before it is used, nothing is allocated
//! from a declared figure before it is charged to the [`MemoryBudget`], and the
//! directory is refused when it does not fit.
//!
//! What is read: single-part archives, with or without zip64 records, whose directory
//! is where the end record says. Archives with data in front that shifts all offsets
//! (self-extracting archives) are not read: their entries are reported as damaged.
//!
//! Everything in an archive is data. Entry names are never used as paths.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::L2dError;
use crate::input::READ_BYTES;
use crate::memory::{MemoryBudget, exact};

const END_SIGNATURE: &[u8] = b"PK\x05\x06";
const END64_LOCATOR_SIGNATURE: &[u8] = b"PK\x06\x07";
const END64_SIGNATURE: &[u8] = b"PK\x06\x06";
const CENTRAL_SIGNATURE: &[u8] = b"PK\x01\x02";
const LOCAL_SIGNATURE: &[u8] = b"PK\x03\x04";

const END_BYTES: usize = 22;
const END64_LOCATOR_BYTES: usize = 20;
const END64_BYTES: usize = 56;
const CENTRAL_BYTES: usize = 46;
const LOCAL_BYTES: usize = 30;
/// The longest comment an end record can carry.
const MAX_COMMENT_BYTES: usize = u16::MAX as usize;

/// The most entries of an archive the tool reads: a position names the entry in 24 bits
/// ([`crate::zip_input::ArchivePosition`]).
pub const MAX_ENTRIES: u64 = 1 << 24;

/// What kind of thing an entry is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryKind {
    File,
    Directory,
    /// A symbolic link: its content is the path it points to, not log text.
    Link,
}

/// One entry as the directory describes it. Its name is kept beside it, by index.
#[derive(Debug, Clone)]
pub(crate) struct DirectoryEntry {
    pub method: u16,
    pub flags: u16,
    pub crc32: u32,
    pub compressed_size: u64,
    pub size: u64,
    /// Where the entry's own header starts in the file.
    pub header_at: u64,
    pub kind: EntryKind,
}

impl DirectoryEntry {
    /// Whether the content is encrypted (traditional, strong, or AES).
    pub fn encrypted(&self) -> bool {
        self.flags & 0x0041 != 0 || self.method == 99
    }
}

fn le16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn le32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn le64(bytes: &[u8], at: usize) -> u64 {
    u64::from(le32(bytes, at)) | u64::from(le32(bytes, at + 4)) << 32
}

fn not_an_archive(path: &Path, what: impl std::fmt::Display) -> L2dError {
    L2dError::InvalidInput {
        detail: format!(
            "{} is not a zip archive the tool can read: {what}",
            path.display()
        ),
    }
}

/// The error for a failed read of the archive file itself.
pub(crate) fn unreadable(path: &Path, error: &std::io::Error) -> L2dError {
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

/// Reads exactly `out.len()` bytes at offset `at`.
fn read_at(file: &mut File, at: u64, out: &mut [u8]) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(at))?;
    file.read_exact(out)
}

/// Where the directory is and how many entries it holds, by the end record.
struct End {
    entries: u64,
    directory_bytes: u64,
    directory_at: u64,
}

/// Finds and checks the end record (and the zip64 end record when there is one).
fn find_end(file: &mut File, file_bytes: u64, path: &Path) -> Result<End, L2dError> {
    let failed = |error: std::io::Error| unreadable(path, &error);
    let tail_bytes =
        file_bytes.min((END_BYTES + MAX_COMMENT_BYTES + END64_LOCATOR_BYTES) as u64) as usize;
    if tail_bytes < END_BYTES {
        return Err(not_an_archive(
            path,
            format!("its {file_bytes} bytes are fewer than an end record takes"),
        ));
    }
    let tail_at = file_bytes - tail_bytes as u64;
    let mut tail = vec![0u8; tail_bytes];
    read_at(file, tail_at, &mut tail).map_err(failed)?;

    // The end record is the last thing in the file: its comment ends where the file
    // ends. The search goes backwards, so the record with the shortest comment wins.
    let start = (0..=tail_bytes - END_BYTES)
        .rev()
        .find(|start| {
            tail[*start..].starts_with(END_SIGNATURE)
                && start + END_BYTES + le16(&tail, start + 20) as usize == tail_bytes
        })
        .ok_or_else(|| {
            not_an_archive(
                path,
                "it has no end record (it is another kind of file, or it is cut off)",
            )
        })?;
    let end = &tail[start..start + END_BYTES];
    let end_at = tail_at + start as u64;
    let in_parts = || not_an_archive(path, "it is one part of an archive in several parts");

    let locator = start
        .checked_sub(END64_LOCATOR_BYTES)
        .map(|at| &tail[at..start])
        .filter(|locator| locator.starts_with(END64_LOCATOR_SIGNATURE));
    let (found, directory_ends_by) = match locator {
        Some(locator) => {
            if le32(locator, 4) != 0 || le32(locator, 16) > 1 {
                return Err(in_parts());
            }
            let end64_at = le64(locator, 8);
            let fits = end64_at
                .checked_add(END64_BYTES as u64)
                .is_some_and(|after| after <= end_at - END64_LOCATOR_BYTES as u64);
            if !fits {
                return Err(not_an_archive(
                    path,
                    format!(
                        "its zip64 end record is said to start at byte {end64_at}, which \
                         leaves no room for it"
                    ),
                ));
            }
            let mut end64 = [0u8; END64_BYTES];
            read_at(file, end64_at, &mut end64).map_err(failed)?;
            if !end64.starts_with(END64_SIGNATURE) {
                return Err(not_an_archive(
                    path,
                    format!("there is no zip64 end record at byte {end64_at}"),
                ));
            }
            if le32(&end64, 16) != 0
                || le32(&end64, 20) != 0
                || le64(&end64, 24) != le64(&end64, 32)
            {
                return Err(in_parts());
            }
            let found = End {
                entries: le64(&end64, 32),
                directory_bytes: le64(&end64, 40),
                directory_at: le64(&end64, 48),
            };
            (found, end64_at)
        }
        None => {
            let (entries, directory_bytes, directory_at) =
                (le16(end, 10), le32(end, 12), le32(end, 16));
            if le16(end, 4) != 0 || le16(end, 6) != 0 || le16(end, 8) != entries {
                return Err(in_parts());
            }
            if entries == u16::MAX || directory_bytes == u32::MAX || directory_at == u32::MAX {
                return Err(not_an_archive(
                    path,
                    "its end record refers to a zip64 end record that is not there",
                ));
            }
            let found = End {
                entries: u64::from(entries),
                directory_bytes: u64::from(directory_bytes),
                directory_at: u64::from(directory_at),
            };
            (found, end_at)
        }
    };

    let ends_in_time = found
        .directory_at
        .checked_add(found.directory_bytes)
        .is_some_and(|after| after <= directory_ends_by);
    if !ends_in_time {
        return Err(not_an_archive(
            path,
            format!(
                "its directory is said to take {} bytes from byte {}, but it must end by \
                 byte {directory_ends_by}",
                found.directory_bytes, found.directory_at
            ),
        ));
    }
    if found.entries > found.directory_bytes / CENTRAL_BYTES as u64 {
        return Err(not_an_archive(
            path,
            format!(
                "its end record counts {} entries, more than a directory of {} bytes can hold",
                found.entries, found.directory_bytes
            ),
        ));
    }
    Ok(found)
}

/// The data of the zip64 field in the extra fields of a directory entry.
fn zip64_field(extra: &[u8]) -> Option<&[u8]> {
    let mut at = 0;
    while at + 4 <= extra.len() {
        let bytes = le16(extra, at + 2) as usize;
        let data = extra.get(at + 4..at + 4 + bytes)?;
        if le16(extra, at) == 1 {
            return Some(data);
        }
        at += 4 + bytes;
    }
    None
}

/// Reads the directory of the archive: one [`DirectoryEntry`] and one name per entry,
/// in directory order, and the memory the two lists are counted as.
///
/// `entry_bytes` is what the caller keeps per entry beside its name. The lists are not
/// allocated before the entry count has been checked against the directory's size and
/// charged to `budget`; the names are charged as they are read.
///
/// - [`L2dError::InvalidInput`] when the file is not a zip archive or its directory is
///   damaged or contradicts itself.
/// - [`L2dError::Refused`] when the directory does not fit what `budget` leaves, or
///   holds more than [`MAX_ENTRIES`] entries.
/// - [`L2dError::Missing`] when the file cannot be read.
pub(crate) fn read_directory(
    file: &mut File,
    file_bytes: u64,
    path: &Path,
    entry_bytes: u64,
    budget: &MemoryBudget,
) -> Result<(Vec<DirectoryEntry>, Vec<String>, u64), L2dError> {
    let end = find_end(file, file_bytes, path)?;
    if end.entries > MAX_ENTRIES {
        return Err(L2dError::Refused {
            reason: format!(
                "{} holds {} entries; the tool reads archives of at most {MAX_ENTRIES} entries",
                path.display(),
                end.entries
            ),
        });
    }
    let left = budget.left_bytes();
    let refused = |needed: u64, read: u64| L2dError::Refused {
        reason: format!(
            "the directory of {} lists {} entries; holding it needs more than {}, but {} is \
             left of the memory limit of {} ({} is held); stopped after {read} entries of the \
             directory, nothing was read from the entries",
            path.display(),
            end.entries,
            exact(needed),
            exact(left),
            exact(budget.limit_bytes()),
            exact(budget.held_bytes()),
        ),
    };
    let fixed = end.entries * entry_bytes;
    if fixed > left {
        return Err(refused(fixed, 0));
    }

    let damaged = |index: u64, what: String| {
        not_an_archive(
            path,
            format!(
                "its directory is damaged at entry {} of {}: {what}",
                index + 1,
                end.entries
            ),
        )
    };
    let mut entries = Vec::with_capacity(end.entries as usize);
    let mut names = Vec::with_capacity(end.entries as usize);
    let mut name_bytes = 0u64;
    file.seek(SeekFrom::Start(end.directory_at))
        .map_err(|error| unreadable(path, &error))?;
    let mut reader = BufReader::with_capacity(READ_BYTES, file.take(end.directory_bytes));
    let mut header = [0u8; CENTRAL_BYTES];
    let mut variable = Vec::new();
    for index in 0..end.entries {
        let cut_off = |error: std::io::Error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => damaged(index, "the directory ends here".into()),
            _ => unreadable(path, &error),
        };
        reader.read_exact(&mut header).map_err(cut_off)?;
        if !header.starts_with(CENTRAL_SIGNATURE) {
            return Err(damaged(index, "no directory entry starts here".into()));
        }
        let (name_len, extra_len, comment_len) = (
            le16(&header, 28) as usize,
            le16(&header, 30) as usize,
            le16(&header, 32) as usize,
        );
        variable.resize(name_len + extra_len + comment_len, 0);
        reader.read_exact(&mut variable).map_err(cut_off)?;
        let (name, rest) = variable.split_at(name_len);
        let extra = &rest[..extra_len];

        // Sizes and offset are in the zip64 field when their 32 bits are all set, in
        // this order.
        let mut wide = zip64_field(extra)
            .unwrap_or_default()
            .as_chunks::<8>()
            .0
            .iter();
        let mut field = |narrow: u32| match narrow {
            u32::MAX => wide.next().map(|bytes| u64::from_le_bytes(*bytes)),
            narrow => Some(u64::from(narrow)),
        };
        let (size, compressed_size, header_at) = match (
            field(le32(&header, 24)),
            field(le32(&header, 20)),
            field(le32(&header, 42)),
        ) {
            (Some(size), Some(compressed_size), Some(header_at)) => {
                (size, compressed_size, header_at)
            }
            _ => {
                return Err(damaged(
                    index,
                    "it refers to zip64 fields that are not there".into(),
                ));
            }
        };

        let mut name = String::from_utf8_lossy(name).into_owned();
        name.shrink_to_fit();
        name_bytes += name.capacity() as u64;
        if fixed + name_bytes > left {
            return Err(refused(fixed + name_bytes, index));
        }
        // The file type of a Unix writer, in the upper half of the external attributes.
        let unix_type = match le16(&header, 4) >> 8 {
            3 => (le32(&header, 38) >> 16) & 0o170_000,
            _ => 0,
        };
        let kind = if name.ends_with('/') || unix_type == 0o040_000 {
            EntryKind::Directory
        } else if unix_type == 0o120_000 {
            EntryKind::Link
        } else {
            EntryKind::File
        };
        entries.push(DirectoryEntry {
            method: le16(&header, 10),
            flags: le16(&header, 8),
            crc32: le32(&header, 16),
            compressed_size,
            size,
            header_at,
            kind,
        });
        names.push(name);
    }
    Ok((entries, names, fixed + name_bytes))
}

/// Why the data of an entry could not be located.
#[derive(Debug)]
pub(crate) enum DataFault {
    /// The entry is damaged: its header or its data is not where the directory says,
    /// or lies beyond the end of the archive. The reason.
    Damaged(String),
    /// The archive file could not be read where the header is known to lie within it.
    Io(std::io::Error),
}

/// Where the data of `entry` starts, by its own header in the file.
///
/// Only the two lengths of the entry's header are used, to step over its name and its
/// extra fields. Method, flags, sizes and checksum are those of the directory; what the
/// entry's own header says about them is not read.
pub(crate) fn locate_data(
    file: &mut File,
    entry: &DirectoryEntry,
    file_bytes: u64,
) -> Result<u64, DataFault> {
    let damaged = |reason: String| Err(DataFault::Damaged(reason));
    let at = entry.header_at;
    if at
        .checked_add(LOCAL_BYTES as u64)
        .is_none_or(|after| after > file_bytes)
    {
        return damaged(format!(
            "its header is said to start at byte {at}, beyond the {file_bytes} bytes of the \
             archive"
        ));
    }
    let mut header = [0u8; LOCAL_BYTES];
    read_at(file, at, &mut header).map_err(DataFault::Io)?;
    if !header.starts_with(LOCAL_SIGNATURE) {
        return damaged(format!("no entry header starts at byte {at}"));
    }
    let data_at =
        at + LOCAL_BYTES as u64 + u64::from(le16(&header, 26)) + u64::from(le16(&header, 28));
    if data_at
        .checked_add(entry.compressed_size)
        .is_none_or(|after| after > file_bytes)
    {
        return damaged(format!(
            "its data is said to take {} bytes from byte {data_at}, beyond the {file_bytes} \
             bytes of the archive",
            entry.compressed_size
        ));
    }
    Ok(data_at)
}
