//! Deduplication on disk for a zip archive whose set of distinct texts outgrows the
//! memory limit (requirement 13.12, the branch that continues).
//!
//! # How it works
//!
//! While the set of text keys fits the memory, an archive is deduplicated in memory
//! ([`crate::zip_input`]). When the set can no longer grow it is **frozen**: it keeps
//! counting the texts it holds, and takes no new one. From then on every row of a
//! batch whose text is not in the set is written to one of [`PARTITIONS`] files, chosen
//! by the first bits of its key: key, position, occurrences in its batch, text. Such a
//! row may be new, or a repetition of a row written before; that is decided after the
//! read.
//!
//! After the last batch each partition is deduplicated on its own: one pass counts the
//! occurrences per key in a set that holds this partition's keys only, a second pass
//! writes the first record of every key with its count. The partitions were written in
//! record order, so each result is in order of first occurrence. The results are then
//! merged by position and handed out as frames.
//!
//! All texts the frozen set holds were first seen before any text on disk, so the rows
//! handed out during the read followed by the merged rows are the distinct texts of the
//! archive in order of first occurrence: the rows of the read in memory alone.
//!
//! # Occurrence counts
//!
//! A row handed out during the read carries the count of its batch; the count over the
//! archive is in the set at the end. For every such row its key is appended to a file,
//! so that the counts can be read out in row order afterwards. The rows from disk
//! carry their count over the archive; it is appended to a second file, for the same
//! reading.
//!
//! # Files
//!
//! All files are in one directory the caller names, which must be empty or absent, and
//! have names made up here (`keys.bin`, `partition-NN.bin`, `distinct-NN.bin`,
//! `counts.bin`). They are removed, and the directory with them, when the read is
//! finished or dropped.
//!
//! # Memory
//!
//! Counted from the moment a directory is named ([`WRITERS_HELD`]): one buffer per
//! partition and one for each of the two other files. A partition's set is asked for
//! before it is built, and refused when it does not fit; with [`PARTITIONS`] partitions
//! a read can go on to that many times the texts one set can hold.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::L2dError;
use crate::memory::exact;

/// The number of partition files.
pub(crate) const PARTITIONS: usize = 64;
/// The buffer of one file being written or read.
const BUFFER_BYTES: usize = 32 * 1024;
/// What deduplication on disk holds in memory: the buffers of the partitions (written
/// during the read, read during the merge), of the key file and of the count file.
pub(crate) const WRITERS_HELD: u64 = ((PARTITIONS + 2) * BUFFER_BYTES) as u64;
/// Keys or counts read at once when the occurrence counts are read out.
const COUNT_CHUNK_ROWS: usize = 65_536;

/// Key, position, occurrences, text length.
const RECORD_HEADER: usize = 16 + 8 + 4 + 4;
/// Position, occurrences, text length.
const DISTINCT_HEADER: usize = 8 + 4 + 4;

fn failed(doing: &str, path: &Path, error: &dyn std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!(
            "deduplication on disk: {doing} {} failed: {error}",
            path.display()
        ),
    }
}

/// What deduplication on disk did, once the set in memory was frozen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskDedup {
    /// Distinct texts the set in memory holds.
    pub set_texts: u64,
    /// Rows written to the partitions: texts not in the set, once per batch they
    /// occurred in.
    pub spilled_records: u64,
    /// The bytes of the partitions.
    pub spilled_bytes: u64,
    /// Distinct texts among the rows on disk; 0 until the partitions are deduplicated.
    pub spilled_distinct: u64,
    /// The number of partitions.
    pub partitions: u32,
}

struct Partition {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    records: u64,
}

/// The state of deduplication on disk for one archive.
pub(crate) struct Spill {
    dir: PathBuf,
    /// The most texts the set in memory may hold; `None`: what the memory allows.
    pub(crate) max_set_texts: Option<u64>,
    /// Every file created, for their removal.
    files: Vec<PathBuf>,
    keys_path: PathBuf,
    keys: Option<BufWriter<File>>,
    counts_path: PathBuf,
    counts: Option<BufWriter<File>>,
    /// `Some` once the set is frozen.
    partitions: Option<Vec<Partition>>,
    report: DiskDedup,
}

impl Spill {
    /// Takes the directory `dir`: creates it, or uses it when it is empty.
    pub(crate) fn create(dir: &Path, max_set_texts: Option<u64>) -> Result<Self, L2dError> {
        if dir.exists() {
            let holds_something = match std::fs::read_dir(dir) {
                Ok(mut entries) => entries.next().is_some(),
                Err(_) => true,
            };
            if holds_something {
                return Err(L2dError::Refused {
                    reason: format!(
                        "the directory for deduplication on disk, {}, exists and is not an \
                         empty directory; nothing in it was touched",
                        dir.display()
                    ),
                });
            }
        }
        std::fs::create_dir_all(dir).map_err(|error| failed("creating", dir, &error))?;
        let mut spill = Self {
            dir: dir.to_path_buf(),
            max_set_texts,
            files: Vec::new(),
            keys_path: dir.join("keys.bin"),
            keys: None,
            counts_path: dir.join("counts.bin"),
            counts: None,
            partitions: None,
            report: DiskDedup {
                set_texts: 0,
                spilled_records: 0,
                spilled_bytes: 0,
                spilled_distinct: 0,
                partitions: PARTITIONS as u32,
            },
        };
        let keys_path = spill.keys_path.clone();
        spill.keys = Some(spill.new_file(&keys_path)?);
        Ok(spill)
    }

    fn new_file(&mut self, path: &Path) -> Result<BufWriter<File>, L2dError> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| failed("creating", path, &error))?;
        self.files.push(path.to_path_buf());
        Ok(BufWriter::with_capacity(BUFFER_BYTES, file))
    }

    /// Whether the set in memory is frozen.
    pub(crate) fn frozen(&self) -> bool {
        self.partitions.is_some()
    }

    /// What was done on disk; `None` while the set in memory takes every text.
    pub(crate) fn report(&self) -> Option<DiskDedup> {
        self.frozen().then_some(self.report)
    }

    /// Distinct texts among the rows on disk, once they are deduplicated.
    pub(crate) fn spilled_distinct(&self) -> u64 {
        self.report.spilled_distinct
    }

    /// Freezes the set in memory at `set_texts` texts: rows not in it go to disk from
    /// here on.
    pub(crate) fn freeze(&mut self, set_texts: u64) {
        if self.partitions.is_some() {
            return;
        }
        self.report.set_texts = set_texts;
        self.partitions = Some(
            (0..PARTITIONS)
                .map(|index| Partition {
                    path: self.dir.join(format!("partition-{index:02}.bin")),
                    writer: None,
                    records: 0,
                })
                .collect(),
        );
    }

    /// Notes the key of a row that was handed out during the read.
    pub(crate) fn note_key(&mut self, key: &[u8; 16]) -> Result<(), L2dError> {
        match &mut self.keys {
            Some(keys) => keys
                .write_all(key)
                .map_err(|error| failed("writing", &self.keys_path, &error)),
            None => Ok(()),
        }
    }

    /// Writes a row whose text is not in the frozen set to its partition.
    pub(crate) fn spill(
        &mut self,
        key: &[u8; 16],
        position: u64,
        occurrences: u32,
        text: &str,
    ) -> Result<(), L2dError> {
        let index = usize::from(key[0]) % PARTITIONS;
        let not_frozen = || L2dError::Runtime {
            detail: "deduplication on disk: a row was written before the set was frozen".into(),
        };
        let (path, has_file) = match &self.partitions {
            Some(partitions) => (
                partitions[index].path.clone(),
                partitions[index].writer.is_some(),
            ),
            None => return Err(not_frozen()),
        };
        let new_file = match has_file {
            true => None,
            false => Some(self.new_file(&path)?),
        };
        let partition = match &mut self.partitions {
            Some(partitions) => &mut partitions[index],
            None => return Err(not_frozen()),
        };
        if new_file.is_some() {
            partition.writer = new_file;
        }
        let Some(writer) = &mut partition.writer else {
            return Err(not_frozen());
        };
        let mut header = [0u8; RECORD_HEADER];
        header[..16].copy_from_slice(key);
        header[16..24].copy_from_slice(&position.to_le_bytes());
        header[24..28].copy_from_slice(&occurrences.to_le_bytes());
        header[28..32].copy_from_slice(&(text.len() as u32).to_le_bytes());
        writer
            .write_all(&header)
            .and_then(|()| writer.write_all(text.as_bytes()))
            .map_err(|error| failed("writing", &path, &error))?;
        partition.records += 1;
        self.report.spilled_records += 1;
        self.report.spilled_bytes += (RECORD_HEADER + text.len()) as u64;
        Ok(())
    }

    /// Deduplicates every partition (module documentation): `distinct-NN.bin` takes
    /// the place of `partition-NN.bin`.
    ///
    /// `room` is the memory the set of one partition may take, `table_bytes` what a
    /// set of that many keys takes. [`L2dError::Refused`] when a partition's set does
    /// not fit; the message states the memory needed and available.
    pub(crate) fn deduplicate(
        &mut self,
        room: u64,
        table_bytes: impl Fn(usize) -> u64,
    ) -> Result<(), L2dError> {
        let Some(mut partitions) = self.partitions.take() else {
            return Ok(());
        };
        let result = self.deduplicate_partitions(&mut partitions, room, table_bytes);
        self.partitions = Some(partitions);
        result
    }

    fn deduplicate_partitions(
        &mut self,
        partitions: &mut [Partition],
        room: u64,
        table_bytes: impl Fn(usize) -> u64,
    ) -> Result<(), L2dError> {
        for (index, partition) in partitions.iter_mut().enumerate() {
            let Some(mut writer) = partition.writer.take() else {
                continue;
            };
            let path = partition.path.clone();
            writer
                .flush()
                .map_err(|error| failed("writing", &path, &error))?;
            drop(writer);

            let mut totals: HashMap<[u8; 16], u32> = HashMap::new();
            let refused = |needed: u64| L2dError::Refused {
                reason: format!(
                    "deduplication on disk: partition {index} of {PARTITIONS} holds {} rows, \
                     and the set of their keys needs {}, but {} is left of the memory limit \
                     beside what is held for the archive",
                    partition.records,
                    exact(needed),
                    exact(room),
                ),
            };
            let needed = table_bytes(partition.records as usize);
            if needed > room {
                return Err(refused(needed));
            }
            totals
                .try_reserve(partition.records as usize)
                .map_err(|_| refused(needed))?;

            // First pass: the occurrences of every key.
            let read_failed = |error: std::io::Error| failed("reading", &path, &error);
            let open =
                || File::open(&path).map(|file| BufReader::with_capacity(BUFFER_BYTES, file));
            let mut reader = open().map_err(read_failed)?;
            let mut header = [0u8; RECORD_HEADER];
            for _ in 0..partition.records {
                reader.read_exact(&mut header).map_err(read_failed)?;
                let (key, _, occurrences, length) = record_header(&header);
                reader
                    .seek_relative(i64::from(length))
                    .map_err(read_failed)?;
                let total = totals.entry(key).or_insert(0);
                *total = total.saturating_add(occurrences);
            }
            drop(reader);

            // Second pass: the first record of every key, with its count.
            let distinct_path = self.dir.join(format!("distinct-{index:02}.bin"));
            let mut out = self.new_file(&distinct_path)?;
            let write_failed = |error: std::io::Error| failed("writing", &distinct_path, &error);
            let mut reader = open().map_err(read_failed)?;
            let mut text = Vec::new();
            for _ in 0..partition.records {
                reader.read_exact(&mut header).map_err(read_failed)?;
                let (key, position, _, length) = record_header(&header);
                match totals.remove(&key) {
                    None => reader
                        .seek_relative(i64::from(length))
                        .map_err(read_failed)?,
                    Some(total) => {
                        text.resize(length as usize, 0);
                        reader.read_exact(&mut text).map_err(read_failed)?;
                        out.write_all(&position.to_le_bytes())
                            .and_then(|()| out.write_all(&total.to_le_bytes()))
                            .and_then(|()| out.write_all(&length.to_le_bytes()))
                            .and_then(|()| out.write_all(&text))
                            .map_err(write_failed)?;
                        self.report.spilled_distinct += 1;
                    }
                }
            }
            out.flush().map_err(write_failed)?;
            drop(reader);
            partition.path = distinct_path;
            // The partition is no longer needed; a failure to remove it now is left
            // to the removal at the end.
            let _ = std::fs::remove_file(&path);
        }
        Ok(())
    }

    /// Opens the deduplicated partitions for merging by position.
    pub(crate) fn merge(&self) -> Result<Merge, L2dError> {
        let mut merge = Merge {
            readers: Vec::new(),
            next: BinaryHeap::new(),
        };
        for partition in self.partitions.iter().flatten() {
            if partition.records == 0 {
                continue;
            }
            let file = File::open(&partition.path)
                .map_err(|error| failed("reading", &partition.path, &error))?;
            merge.readers.push((
                BufReader::with_capacity(BUFFER_BYTES, file),
                partition.path.clone(),
            ));
            merge.advance(merge.readers.len() - 1)?;
        }
        Ok(merge)
    }

    /// Notes the counts of rows handed out from disk, in the order handed out.
    pub(crate) fn note_counts(&mut self, counts: &[u32]) -> Result<(), L2dError> {
        if self.counts.is_none() {
            let path = self.counts_path.clone();
            self.counts = Some(self.new_file(&path)?);
        }
        let Some(out) = &mut self.counts else {
            return Ok(());
        };
        for count in counts {
            out.write_all(&count.to_le_bytes())
                .map_err(|error| failed("writing", &self.counts_path, &error))?;
        }
        Ok(())
    }

    /// Readers over the keys of the rows handed out during the read and over the
    /// counts of the rows handed out from disk, from their starts.
    pub(crate) fn count_readers(&mut self) -> Result<(CountFile, Option<CountFile>), L2dError> {
        if let Some(keys) = &mut self.keys {
            keys.flush()
                .map_err(|error| failed("writing", &self.keys_path, &error))?;
        }
        if let Some(counts) = &mut self.counts {
            counts
                .flush()
                .map_err(|error| failed("writing", &self.counts_path, &error))?;
        }
        let keys = CountFile::open(&self.keys_path, 16)?;
        let counts = match self.counts {
            Some(_) => Some(CountFile::open(&self.counts_path, 4)?),
            None => None,
        };
        Ok((keys, counts))
    }
}

impl Drop for Spill {
    fn drop(&mut self) {
        self.keys = None;
        self.counts = None;
        self.partitions = None;
        for file in &self.files {
            let _ = std::fs::remove_file(file);
        }
        // Only an empty directory goes: what is not from here stays.
        let _ = std::fs::remove_dir(&self.dir);
    }
}

fn record_header(header: &[u8; RECORD_HEADER]) -> ([u8; 16], u64, u32, u32) {
    let mut key = [0u8; 16];
    key.copy_from_slice(&header[..16]);
    let word = |at: usize| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    let mut position = [0u8; 8];
    position.copy_from_slice(&header[16..24]);
    (key, u64::from_le_bytes(position), word(24), word(28))
}

/// A file of fixed-size items, read in chunks.
pub(crate) struct CountFile {
    reader: BufReader<File>,
    path: PathBuf,
    item_bytes: usize,
    buffer: Vec<u8>,
}

impl CountFile {
    fn open(path: &Path, item_bytes: usize) -> Result<Self, L2dError> {
        let file = File::open(path).map_err(|error| failed("reading", path, &error))?;
        Ok(Self {
            reader: BufReader::with_capacity(BUFFER_BYTES, file),
            path: path.to_path_buf(),
            item_bytes,
            buffer: Vec::new(),
        })
    }

    /// The next items, at most [`COUNT_CHUNK_ROWS`], end to end; empty at the end.
    pub(crate) fn next_chunk(&mut self) -> Result<&[u8], L2dError> {
        self.buffer.resize(COUNT_CHUNK_ROWS * self.item_bytes, 0);
        let mut filled = 0;
        while filled < self.buffer.len() {
            match self.reader.read(&mut self.buffer[filled..]) {
                Ok(0) => break,
                Ok(count) => filled += count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(failed("reading", &self.path, &error)),
            }
        }
        if filled % self.item_bytes != 0 {
            return Err(failed(
                "reading",
                &self.path,
                &"the file ends inside an item",
            ));
        }
        self.buffer.truncate(filled);
        Ok(&self.buffer)
    }
}

/// The rows of one frame of merged rows: texts, positions, occurrence counts.
pub(crate) type MergedRows = (Vec<String>, Vec<u64>, Vec<u32>);

/// The deduplicated partitions, merged by position.
pub(crate) struct Merge {
    readers: Vec<(BufReader<File>, PathBuf)>,
    /// The next row of every partition that has one: position, reader, count, length.
    next: BinaryHeap<Reverse<(u64, usize, u32, u32)>>,
}

impl Merge {
    /// Reads the header of the next row of the reader at `index`, if it has one.
    fn advance(&mut self, index: usize) -> Result<(), L2dError> {
        let (reader, path) = &mut self.readers[index];
        let mut header = [0u8; DISTINCT_HEADER];
        let mut filled = 0;
        while filled < DISTINCT_HEADER {
            match reader.read(&mut header[filled..]) {
                Ok(0) => break,
                Ok(count) => filled += count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(failed("reading", path, &error)),
            }
        }
        match filled {
            0 => Ok(()),
            DISTINCT_HEADER => {
                let mut position = [0u8; 8];
                position.copy_from_slice(&header[..8]);
                let word = |at: usize| {
                    u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
                };
                self.next.push(Reverse((
                    u64::from_le_bytes(position),
                    index,
                    word(8),
                    word(12),
                )));
                Ok(())
            }
            _ => Err(failed("reading", path, &"the file ends inside a row")),
        }
    }

    /// The next rows in order of position: at most `max_rows`, and no more text than
    /// `max_bytes` unless the first row alone is longer. `None` at the end.
    pub(crate) fn next_rows(
        &mut self,
        max_rows: u64,
        max_bytes: u64,
    ) -> Result<Option<MergedRows>, L2dError> {
        let (mut texts, mut positions, mut counts) = (Vec::new(), Vec::new(), Vec::new());
        let mut bytes: u64 = 0;
        while let Some(Reverse((position, index, count, length))) = self.next.peek().copied() {
            let full =
                texts.len() as u64 >= max_rows.max(1) || bytes + u64::from(length) > max_bytes;
            if full && !texts.is_empty() {
                break;
            }
            self.next.pop();
            let (reader, path) = &mut self.readers[index];
            let mut text = vec![0u8; length as usize];
            reader
                .read_exact(&mut text)
                .map_err(|error| failed("reading", path, &error))?;
            let text = String::from_utf8(text)
                .map_err(|_| failed("reading", path, &"a text is not valid UTF-8"))?;
            bytes += u64::from(length);
            texts.push(text);
            positions.push(position);
            counts.push(count);
            self.advance(index)?;
        }
        Ok((!texts.is_empty()).then_some((texts, positions, counts)))
    }
}
