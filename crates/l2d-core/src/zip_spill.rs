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
//! A read whose directory is made only when the set is frozen ([`Spill::create_in`],
//! for `l2d ingest`) keeps no occurrence counts: it has no key file, and the rows
//! handed out before the freeze have no key on disk.
//!
//! # Files
//!
//! All files are in one directory the caller names, which must be empty or absent, and
//! have names made up here (`keys.bin`, `partition-NN.bin`, `distinct-NN.bin`,
//! `counts.bin`). They are removed, and the directory with them, when the read is
//! finished or dropped; so is the work directory made at the freeze, as far as it is
//! empty.
//!
//! # Memory
//!
//! Counted from the moment the directory is there ([`SPILL_HELD`]): one buffer per
//! partition and one for each of the two other files, and the chunks the occurrence
//! counts are read out in; for a read that keeps no occurrence counts, the buffers
//! alone, from the freeze on, sized at the freeze to what is left beside the smallest
//! batch: 32 KiB each down to 4 KiB ([`buffer_for`]). A partition's set must fit what the memory limit leaves
//! beside what is held for the archive (the room). What a partition's set needs is
//! decided by its distinct keys, not by its rows: a text that is not in the frozen set
//! is written once per batch it occurs in, so the rows of a partition grow as the
//! batches get smaller, its distinct keys do not (task 1.20, remediation).
//!
//! While the archive is read, the distinct keys of a partition are bounded from above:
//! the distinct keys counted at its last count plus the rows written since. A bound
//! above what the room holds sends the partition to be counted again
//! ([`Spill::check_partitions`], between batches, when a batch's memory is free): its
//! file is read with a set that takes at most what the room holds, and the partition
//! is refused when its keys do not fit that set. A partition is counted again only
//! when its rows have at least doubled since its last count, so that all counts of a
//! partition read at most twice its last file. A partition that does not fit is
//! therefore refused while the archive is read, at the latest when its rows have
//! doubled after its keys outgrew the room; the refusal depends on the distinct keys,
//! which do not depend on the batch size. The set of a count lives in the room, and its
//! file is read through a buffer nothing uses while the archive is read: one of the two
//! the occurrence counts are read with after the read, or, for a read that keeps no
//! occurrence counts, the one of the key file it does not have. With [`PARTITIONS`]
//! partitions a read can go on to that many times the texts one set can hold.
//!
//! # Disk
//!
//! A row on disk takes [`SPILLED_ROW_BYTES`] beside its text. How many rows an archive
//! has is not known before it is read, so [`estimated_disk_bytes`] takes a record to be
//! [`ESTIMATED_RECORD_BYTES`] long: an estimate, not a bound.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::L2dError;
use crate::memory::exact;
use crate::parts::WorkDir;
use crate::zip_input::table_bytes;

/// The number of partition files.
pub(crate) const PARTITIONS: usize = 64;
/// The buffer of one file being written or read.
const BUFFER_BYTES: usize = 32 * 1024;
/// What deduplication on disk holds in memory: the buffers of the partitions (written
/// during the read, read during the merge), of the key file and of the count file.
pub(crate) const WRITERS_HELD: u64 = ((PARTITIONS + 2) * BUFFER_BYTES) as u64;
/// Keys or counts read at once when the occurrence counts are read out.
const COUNT_CHUNK_ROWS: usize = 65_536;
/// What reading the occurrence counts out holds: a chunk of keys (16 bytes each) with
/// the chunk of counts made from it (4 bytes each), and the buffers of the two files
/// being read.
const COUNTS_HELD: u64 = (COUNT_CHUNK_ROWS * (16 + 4) + 2 * BUFFER_BYTES) as u64;
/// Everything deduplication on disk holds in memory, from the moment its directory is
/// named to the end of the read.
pub(crate) const SPILL_HELD: u64 = WRITERS_HELD + COUNTS_HELD;
/// The smallest buffer a read that keeps no occurrence counts takes when less is left
/// at its freeze.
const MIN_BUFFER_BYTES: usize = 4 * 1024;
/// The least a read that keeps no occurrence counts ([`Spill::create_in`]) holds from
/// the freeze to the end of the read: the buffers of the files only, at their
/// smallest. It has no key file and no count file, so two of the buffers are free for
/// reading a partition while the archive is read and for writing its deduplicated copy
/// after it. At most it holds [`WRITERS_HELD`] ([`buffer_for`]).
pub(crate) const SPILL_HELD_WITHOUT_COUNTS_LEAST: u64 =
    ((PARTITIONS + 2) * MIN_BUFFER_BYTES) as u64;

/// The buffer of every file of a read that keeps no occurrence counts, when `room`
/// bytes are left for the buffers at its freeze: the largest of 32, 16, 8 and 4 KiB
/// whose buffers fit, and 4 KiB when none does. The set grew into the memory until it
/// could no longer, so what is left at the freeze can be less than the buffers of a
/// read that holds them from the start; smaller buffers cost more writes, not memory.
pub(crate) fn buffer_for(room: u64) -> usize {
    let mut buffer = BUFFER_BYTES;
    while buffer > MIN_BUFFER_BYTES && ((PARTITIONS + 2) * buffer) as u64 > room {
        buffer /= 2;
    }
    buffer
}

/// Key, position, occurrences, text length.
const RECORD_HEADER: usize = 16 + 8 + 4 + 4;
/// What a row on disk takes beside its text.
pub(crate) const SPILLED_ROW_BYTES: u64 = RECORD_HEADER as u64;
/// The length a record is taken to have, line end included, where the rows of an
/// archive are estimated from its unpacked bytes: shorter than a line of any log, so
/// that the estimate is high. The archive of the review of task 1.17 that the estimate
/// was too low for has records of 6.9 bytes on average.
pub(crate) const ESTIMATED_RECORD_BYTES: u64 = 6;

/// What deduplication on disk is estimated to write for entries of `unpacked` bytes:
/// every text once (a text that is not in the set in memory is written once per batch
/// it occurs in, never more often than it occurs), and [`SPILLED_ROW_BYTES`] for every
/// row, with the rows estimated from [`ESTIMATED_RECORD_BYTES`].
pub(crate) fn estimated_disk_bytes(unpacked: u64) -> u64 {
    let rows = unpacked / ESTIMATED_RECORD_BYTES;
    unpacked.saturating_add(rows.saturating_mul(SPILLED_ROW_BYTES))
}

/// The most keys a set may hold within `room` bytes: the largest number whose
/// [`table_bytes`] is not above it.
pub(crate) fn keys_fitting(room: u64) -> u64 {
    // `table_bytes(n)` is above `n` for every `n` from 1 on, and grows with `n`.
    let (mut fits, mut above) = (0u64, room);
    while above - fits > 1 {
        let middle = fits + (above - fits) / 2;
        if table_bytes(middle as usize) <= room {
            fits = middle;
        } else {
            above = middle;
        }
    }
    fits
}

/// The refusal of a partition whose distinct keys do not fit the room: it holds more
/// than `most` of them, in `rows` rows on disk.
fn partition_refused(index: usize, most: u64, rows: u64, room: u64) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "deduplication on disk: partition {index} of {PARTITIONS} holds more than {most} \
             distinct keys (in {rows} rows on disk), and the set of their keys needs {}, but {} \
             is left of the memory limit beside what is held for the archive",
            exact(table_bytes(most as usize + 1)),
            exact(room),
        ),
    }
}

/// The distinct keys of the first `rows` rows of the partition file at `path`, when
/// they are at most `most`; `None` when there are more. The set never holds more than
/// `most` keys, so it takes at most [`table_bytes`] of `most`.
fn distinct_keys(
    path: &Path,
    rows: u64,
    most: u64,
    buffer: usize,
) -> Result<Option<u64>, L2dError> {
    let read_failed = |error: std::io::Error| failed("reading", path, &error);
    let mut keys: HashSet<[u8; 16]> = HashSet::new();
    keys.try_reserve(rows.min(most) as usize)
        .map_err(|error| failed("counting the keys of", path, &error))?;
    let file = File::open(path).map_err(read_failed)?;
    let mut reader = BufReader::with_capacity(buffer, file);
    let mut header = [0u8; RECORD_HEADER];
    for _ in 0..rows {
        reader.read_exact(&mut header).map_err(read_failed)?;
        let (key, _, _, length) = record_header(&header);
        reader
            .seek_relative(i64::from(length))
            .map_err(read_failed)?;
        if !keys.contains(&key) {
            if keys.len() as u64 == most {
                return Ok(None);
            }
            keys.insert(key);
        }
    }
    Ok(Some(keys.len() as u64))
}
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
    /// The rows when the distinct keys were last counted, and their number then.
    counted_records: u64,
    counted_distinct: u64,
}

/// The state of deduplication on disk for one archive.
pub(crate) struct Spill {
    dir: PathBuf,
    /// The most texts the set in memory may hold; `None`: what the memory allows.
    pub(crate) max_set_texts: Option<u64>,
    /// Whether the occurrence counts of all rows are kept (the key file and the count
    /// file).
    keeps_counts: bool,
    /// The work directory made at the freeze, removed again as far as it is empty.
    made: Option<WorkDir>,
    /// The buffer of every file written or read.
    buffer: usize,
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

/// Creates the directory `dir`, or takes it when it is empty.
fn take_directory(dir: &Path) -> Result<(), L2dError> {
    if dir.exists() {
        let holds_something = match std::fs::read_dir(dir) {
            Ok(mut entries) => entries.next().is_some(),
            Err(_) => true,
        };
        if holds_something {
            return Err(L2dError::Refused {
                reason: format!(
                    "the directory for deduplication on disk, {}, exists and is not an empty \
                     directory; nothing in it was touched",
                    dir.display()
                ),
            });
        }
    }
    std::fs::create_dir_all(dir).map_err(|error| failed("creating", dir, &error))
}

impl Spill {
    fn new(dir: &Path, max_set_texts: Option<u64>, made: Option<WorkDir>, buffer: usize) -> Self {
        Self {
            dir: dir.to_path_buf(),
            max_set_texts,
            keeps_counts: made.is_none(),
            made,
            buffer,
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
        }
    }

    /// Takes the directory `dir`: creates it, or uses it when it is empty. The key of
    /// every row handed out is written from here on, for the occurrence counts.
    pub(crate) fn create(dir: &Path, max_set_texts: Option<u64>) -> Result<Self, L2dError> {
        take_directory(dir)?;
        let mut spill = Self::new(dir, max_set_texts, None, BUFFER_BYTES);
        let keys_path = spill.keys_path.clone();
        spill.keys = Some(spill.new_file(&keys_path)?);
        Ok(spill)
    }

    /// Takes `dedup/` below the work directory `work`, which was just made because the
    /// set is about to be frozen; `work` is removed again as far as it is empty when
    /// the value is dropped. Keeps no occurrence counts. `room` is what is left for its
    /// buffers ([`buffer_for`]).
    pub(crate) fn create_in(
        work: WorkDir,
        max_set_texts: Option<u64>,
        room: u64,
    ) -> Result<Self, L2dError> {
        let dir = work.path().join("dedup");
        if let Err(error) = take_directory(&dir) {
            work.remove_created();
            return Err(error);
        }
        Ok(Self::new(&dir, max_set_texts, Some(work), buffer_for(room)))
    }

    /// The directory of the files.
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// What this value holds in memory until the end of the read.
    pub(crate) fn held(&self) -> u64 {
        match self.keeps_counts {
            true => SPILL_HELD,
            false => ((PARTITIONS + 2) * self.buffer) as u64,
        }
    }

    fn new_file(&mut self, path: &Path) -> Result<BufWriter<File>, L2dError> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| failed("creating", path, &error))?;
        self.files.push(path.to_path_buf());
        Ok(BufWriter::with_capacity(self.buffer, file))
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
                    counted_records: 0,
                    counted_distinct: 0,
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

    /// Writes a row whose text is not in the frozen set to its partition. Whether the
    /// partition still fits is checked between batches ([`Spill::check_partitions`]).
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

    /// Checks between batches that every partition's distinct keys can still fit
    /// `room`, the memory the set of one partition may take (module documentation):
    /// a partition whose upper bound is above what `room` holds is counted again when
    /// its rows have at least doubled since its last count.
    ///
    /// [`L2dError::Refused`] when a partition holds more distinct keys than `room`
    /// holds: it could not be deduplicated after the read, so the read stops here. The
    /// message states the memory needed and available. The count's set takes at most
    /// `room`, which is free between batches.
    pub(crate) fn check_partitions(&mut self, room: u64) -> Result<(), L2dError> {
        let most = keys_fitting(room);
        let buffer = self.buffer;
        let Some(partitions) = &mut self.partitions else {
            return Ok(());
        };
        for (index, partition) in partitions.iter_mut().enumerate() {
            let bound =
                partition.counted_distinct + (partition.records - partition.counted_records);
            if bound <= most || partition.records < 2 * partition.counted_records {
                continue;
            }
            let Some(writer) = &mut partition.writer else {
                continue;
            };
            writer
                .flush()
                .map_err(|error| failed("writing", &partition.path, &error))?;
            match distinct_keys(&partition.path, partition.records, most, buffer)? {
                Some(distinct) => {
                    partition.counted_records = partition.records;
                    partition.counted_distinct = distinct;
                }
                None => return Err(partition_refused(index, most, partition.records, room)),
            }
        }
        Ok(())
    }

    /// Deduplicates every partition (module documentation): `distinct-NN.bin` takes
    /// the place of `partition-NN.bin`.
    ///
    /// `room` is the memory the set of one partition may take. [`L2dError::Refused`]
    /// when a partition's distinct keys do not fit it (the room can have shrunk since
    /// the partition was last counted, and its rows since then were not counted); the
    /// message states the memory needed and available.
    pub(crate) fn deduplicate(&mut self, room: u64) -> Result<(), L2dError> {
        let Some(mut partitions) = self.partitions.take() else {
            return Ok(());
        };
        let result = self.deduplicate_partitions(&mut partitions, room);
        self.partitions = Some(partitions);
        result
    }

    fn deduplicate_partitions(
        &mut self,
        partitions: &mut [Partition],
        room: u64,
    ) -> Result<(), L2dError> {
        let buffer = self.buffer;
        for (index, partition) in partitions.iter_mut().enumerate() {
            let Some(mut writer) = partition.writer.take() else {
                continue;
            };
            let path = partition.path.clone();
            writer
                .flush()
                .map_err(|error| failed("writing", &path, &error))?;
            drop(writer);

            // The set holds at most what the room holds: it is made for that many keys
            // (or for the rows, when they are fewer) and never grows.
            let most = keys_fitting(room);
            let refused = || partition_refused(index, most, partition.records, room);
            let mut totals: HashMap<[u8; 16], u32> = HashMap::new();
            totals
                .try_reserve(partition.records.min(most) as usize)
                .map_err(|_| refused())?;

            // First pass: the occurrences of every key.
            let read_failed = |error: std::io::Error| failed("reading", &path, &error);
            let open = || File::open(&path).map(|file| BufReader::with_capacity(buffer, file));
            let mut reader = open().map_err(read_failed)?;
            let mut header = [0u8; RECORD_HEADER];
            for _ in 0..partition.records {
                reader.read_exact(&mut header).map_err(read_failed)?;
                let (key, _, occurrences, length) = record_header(&header);
                reader
                    .seek_relative(i64::from(length))
                    .map_err(read_failed)?;
                if let Some(total) = totals.get_mut(&key) {
                    *total = total.saturating_add(occurrences);
                } else if totals.len() as u64 == most {
                    return Err(refused());
                } else {
                    totals.insert(key, occurrences);
                }
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
                BufReader::with_capacity(self.buffer, file),
                partition.path.clone(),
            ));
            merge.advance(merge.readers.len() - 1)?;
        }
        Ok(merge)
    }

    /// Notes the counts of rows handed out from disk, in the order handed out.
    pub(crate) fn note_counts(&mut self, counts: &[u32]) -> Result<(), L2dError> {
        if !self.keeps_counts {
            return Ok(());
        }
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
        if !self.keeps_counts {
            return Err(L2dError::Runtime {
                detail: "the occurrence counts of the rows are not kept by a read whose \
                         directory for deduplication on disk was made when the set was frozen"
                    .into(),
            });
        }
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
        if let Some(work) = &self.made {
            work.remove_created();
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A key of partition 0 (its first byte is 0) for `index`.
    fn key(index: u64) -> [u8; 16] {
        let mut key = [0u8; 16];
        key[8..].copy_from_slice(&index.to_le_bytes());
        key
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("l2d-core-spill-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Writes the rows of one batch, the keys `keys`, to disk, and checks the
    /// partitions after it, as the read does.
    fn write_batch(
        spill: &mut Spill,
        keys: &[u64],
        position: &mut u64,
        room: u64,
    ) -> Result<(), L2dError> {
        for index in keys {
            *position += 1;
            spill.spill(&key(*index), *position, 1, "a text")?;
        }
        spill.check_partitions(room)
    }

    #[test]
    fn the_buffers_taken_at_the_freeze_fit_what_is_left() {
        let kib = 1024u64;
        assert_eq!(buffer_for(u64::MAX), 32 * 1024);
        assert_eq!(buffer_for(66 * 32 * kib), 32 * 1024);
        assert_eq!(buffer_for(66 * 32 * kib - 1), 16 * 1024);
        assert_eq!(buffer_for(66 * 8 * kib), 8 * 1024);
        assert_eq!(buffer_for(66 * 4 * kib), 4 * 1024);
        // Less than the smallest buffers: the smallest, and the read is refused
        // by the check of what is held.
        assert_eq!(buffer_for(0), 4 * 1024);
        assert_eq!(SPILL_HELD_WITHOUT_COUNTS_LEAST, 66 * 4 * kib);
    }

    #[test]
    fn the_keys_that_fit_a_room_are_those_of_the_largest_table_within_it() {
        assert_eq!(keys_fitting(0), 0);
        assert_eq!(keys_fitting(table_bytes(1) - 1), 0);
        assert_eq!(keys_fitting(table_bytes(1)), 3);
        assert_eq!(keys_fitting(table_bytes(1_000)), 1_792);
        assert_eq!(keys_fitting(table_bytes(1_793) - 1), 1_792);
        assert_eq!(keys_fitting(table_bytes(1_793)), 3_584);
    }

    /// Remediation of task 1.20, item 3: ten keys written once per batch, as often as
    /// small batches write them, do not outgrow a room that holds a thousand keys,
    /// however the rows are cut into batches.
    #[test]
    fn a_partition_is_given_room_by_its_distinct_keys_not_by_its_rows() {
        let room = table_bytes(1_000);
        for batch in [1usize, 10, 64] {
            let dir = scratch(&format!("ten-{batch}"));
            let mut spill = Spill::create(&dir, None).unwrap();
            spill.freeze(0);
            let mut position = 0;
            // 30,000 rows of ten keys.
            let keys: Vec<u64> = (0..30_000).map(|row| row % 10).collect();
            for rows in keys.chunks(batch) {
                write_batch(&mut spill, rows, &mut position, room)
                    .unwrap_or_else(|error| panic!("batches of {batch}: {error}"));
            }
            spill.deduplicate(room).unwrap();
            assert_eq!(spill.spilled_distinct(), 10, "batches of {batch}");
            drop(spill);
            assert!(!dir.exists());
        }
    }

    /// Item 3: a partition whose distinct keys outgrow the room is still refused while
    /// the rows are written, long before the last of them.
    #[test]
    fn a_partition_whose_distinct_keys_outgrow_the_room_is_refused_early() {
        let room = table_bytes(1_000);
        let dir = scratch("distinct");
        let mut spill = Spill::create(&dir, None).unwrap();
        spill.freeze(0);
        let mut position = 0;
        let keys: Vec<u64> = (0..100_000).collect();
        let mut written = 0;
        let error = keys
            .chunks(64)
            .find_map(|rows| {
                written += rows.len();
                write_batch(&mut spill, rows, &mut position, room).err()
            })
            .expect("the partition is refused");
        let text = error.to_string();
        assert!(matches!(error, L2dError::Refused { .. }), "{text}");
        assert!(text.contains("partition 0 of 64"), "{text}");
        assert!(written <= 2_000, "refused after {written} rows: {text}");
        drop(spill);
        assert!(!dir.exists());
    }
}
