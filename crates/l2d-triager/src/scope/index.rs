//! The local relation between the distinct originals of a scope and their native keys.
//!
//! One original is one fixed-width row: its first native position (the record of a
//! file, the packed entry and line of an archive, the `seq` of a stored log), its
//! 32-byte native identity (`l2d_store::log_id`) and its occurrence count. Two
//! open-addressing tables find a row by position and by identity.
//!
//! The rows are written to the work directory while the source is read, so a scan
//! keeps nothing per original in memory. Once the number of originals is known the
//! rows are put in the order of their positions, which is the order of the record IDs:
//! a source that hands its originals out in another order (several entries read at
//! once, deduplication on disk) has its rows sorted on disk within the memory the
//! budget admits (`crate::extsort`). A row number therefore does not depend on batch
//! sizes or spill, and a page of records is a run of rows. The relation is then loaded
//! into memory when the budget admits all of it, and stays on disk otherwise. Both
//! placements run the same code over [`Bytes`], so a lookup gives the same answer in
//! either.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use l2d_core::memory::exact;
use l2d_core::parts::{WorkDir, free_disk_bytes};
use l2d_core::{L2dError, MemoryBudget};
use sha2::{Digest, Sha256};

use super::IndexPlacement;
use crate::extsort::{sort_file, sort_memory};

pub(super) const ID_BYTES: usize = 32;
const POSITION_BYTES: u64 = 8;
pub(super) const KEY_BYTES: u64 = POSITION_BYTES + ID_BYTES as u64;
const COUNT_BYTES: u64 = 4;
const SLOT_BYTES: u64 = 8;
/// The buffer of each of the two files written during a scan.
const BUFFER_BYTES: usize = 64 * 1024;
/// How many rows are written between two looks at the free disk space, and how many
/// rows the tables are built from at a time.
const CHUNK_ROWS: u64 = 1_024;

const KEYS: &str = "keys.bin";
const COUNTS: &str = "counts.bin";
const BY_POSITION: &str = "by-position.bin";
const BY_ID: &str = "by-id.bin";
/// The rows with their counts while they are put in order, and the ordered rows.
const UNORDERED: &str = "rows-unordered.bin";
const ORDERED: &str = "rows-ordered.bin";
const ROW_BYTES: u64 = KEY_BYTES + COUNT_BYTES;

fn failed(doing: &str, path: &Path, error: &dyn std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

fn corrupt(detail: &str) -> L2dError {
    L2dError::Runtime {
        detail: format!("the scope's locator index is inconsistent: {detail}"),
    }
}

/// The work directory of one scope and the files it may hold. What this value made is
/// removed with it.
pub(super) struct Work {
    dir: WorkDir,
    /// The free space to compare with in place of the operating system's figure, and
    /// what was written against it.
    free: Option<u64>,
    written: u64,
}

impl Work {
    pub(super) fn create(path: &Path, free: Option<u64>) -> Result<Self, L2dError> {
        Ok(Self {
            dir: WorkDir::create(path)?,
            free,
            written: 0,
        })
    }

    pub(super) fn path(&self) -> &Path {
        self.dir.path()
    }

    fn file(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// [`L2dError::Refused`] with both figures unless `bytes` more fit the disk.
    fn require_disk(&self, bytes: u64, what: &str, progress: &str) -> Result<(), L2dError> {
        let free = match self.free {
            Some(given) => given.saturating_sub(self.written),
            None => free_disk_bytes(self.dir.path())?,
        };
        if bytes <= free {
            return Ok(());
        }
        Err(L2dError::Refused {
            reason: format!(
                "the scope's locator index needs {} of disk space in the work directory {} for \
                 {what}, but {} is free there; {progress}",
                exact(bytes),
                self.dir.path().display(),
                exact(free)
            ),
        })
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        for name in [KEYS, COUNTS, BY_POSITION, BY_ID, UNORDERED, ORDERED] {
            let _ = fs::remove_file(self.file(name));
        }
        self.dir.remove_created();
    }
}

#[cfg(windows)]
pub(crate) fn read_at(file: &File, mut at: u64, mut out: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !out.is_empty() {
        match file.seek_read(out, at) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => {
                out = &mut out[read..];
                at += read as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) fn write_at(file: &File, mut at: u64, mut data: &[u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !data.is_empty() {
        match file.seek_write(data, at) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(written) => {
                data = &data[written..];
                at += written as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(unix)]
pub(crate) fn read_at(file: &File, at: u64, out: &mut [u8]) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, out, at)
}

#[cfg(unix)]
pub(crate) fn write_at(file: &File, at: u64, data: &[u8]) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, data, at)
}

/// Fixed-width rows, in memory or in a file of the work directory.
enum Bytes {
    Memory(Vec<u8>),
    Disk(File),
}

impl Bytes {
    fn read(&self, at: u64, out: &mut [u8]) -> Result<(), L2dError> {
        match self {
            Bytes::Memory(bytes) => {
                let from = usize::try_from(at).ok();
                let part = from.and_then(|from| bytes.get(from..from.checked_add(out.len())?));
                out.copy_from_slice(part.ok_or_else(|| corrupt("a row lies beyond its table"))?);
                Ok(())
            }
            Bytes::Disk(file) => read_at(file, at, out)
                .map_err(|error| corrupt(&format!("reading a row from disk failed: {error}"))),
        }
    }

    fn write(&mut self, at: u64, data: &[u8]) -> Result<(), L2dError> {
        match self {
            Bytes::Memory(bytes) => {
                let from = usize::try_from(at).ok();
                let part = from.and_then(|from| bytes.get_mut(from..from.checked_add(data.len())?));
                part.ok_or_else(|| corrupt("a row lies beyond its table"))?
                    .copy_from_slice(data);
                Ok(())
            }
            Bytes::Disk(file) => write_at(file, at, data)
                .map_err(|error| corrupt(&format!("writing a row to disk failed: {error}"))),
        }
    }

    fn held(&self) -> u64 {
        match self {
            Bytes::Memory(bytes) => bytes.capacity() as u64,
            Bytes::Disk(_) => 0,
        }
    }
}

/// What putting the rows in order holds beside the memory of the sort: the rows it
/// joins and splits at a time and the buffer it writes them through.
const ORDER_BUFFER_BYTES: u64 = CHUNK_ROWS * ROW_BYTES * 2 + BUFFER_BYTES as u64 * 2;

/// Puts the rows of the work directory in the order of their positions: the keys and
/// the counts are joined, sorted by position within the memory `budget` admits, and
/// split again.
fn order_rows(work: &Work, rows: u64, budget: &MemoryBudget) -> Result<(), L2dError> {
    let memory = sort_memory(
        rows,
        ROW_BYTES as usize,
        budget.left_bytes().saturating_sub(ORDER_BUFFER_BYTES),
    );
    budget.require_whole(
        memory + ORDER_BUFFER_BYTES,
        "putting the originals of a scope in the order of their record IDs",
    )?;
    work.require_disk(
        3 * rows * ROW_BYTES,
        "its originals in the order of their record IDs",
        &format!("{rows} originals were read, and nothing is kept"),
    )?;
    let (keys, counts) = (work.file(KEYS), work.file(COUNTS));
    let (unordered, ordered) = (work.file(UNORDERED), work.file(ORDERED));
    let open = |path: &Path| File::open(path).map_err(|error| failed("opening", path, &error));
    let create = |path: &Path| {
        File::create(path)
            .map(|file| BufWriter::with_capacity(BUFFER_BYTES, file))
            .map_err(|error| failed("creating", path, &error))
    };
    let end = |writer: BufWriter<File>, path: &Path| {
        writer
            .into_inner()
            .map(drop)
            .map_err(|error| failed("writing", path, &error.into_error()))
    };
    let remove =
        |path: &Path| fs::remove_file(path).map_err(|error| failed("removing", path, &error));
    let mut key_rows = vec![0u8; (CHUNK_ROWS * KEY_BYTES) as usize];
    let mut count_rows = vec![0u8; (CHUNK_ROWS * COUNT_BYTES) as usize];
    {
        let (mut from_keys, mut from_counts) = (open(&keys)?, open(&counts)?);
        let mut joined = create(&unordered)?;
        let mut left = rows;
        while left > 0 {
            let now = CHUNK_ROWS.min(left) as usize;
            let key_rows = &mut key_rows[..now * KEY_BYTES as usize];
            let count_rows = &mut count_rows[..now * COUNT_BYTES as usize];
            from_keys
                .read_exact(key_rows)
                .map_err(|error| failed("reading", &keys, &error))?;
            from_counts
                .read_exact(count_rows)
                .map_err(|error| failed("reading", &counts, &error))?;
            let (key_rows, _) = key_rows.as_chunks::<{ KEY_BYTES as usize }>();
            let (count_rows, _) = count_rows.as_chunks::<{ COUNT_BYTES as usize }>();
            for (key, count) in key_rows.iter().zip(count_rows) {
                joined
                    .write_all(key)
                    .and_then(|()| joined.write_all(count))
                    .map_err(|error| failed("writing", &unordered, &error))?;
            }
            left -= now as u64;
        }
        end(joined, &unordered)?;
    }
    remove(&keys)?;
    remove(&counts)?;
    sort_file(
        &unordered,
        &ordered,
        work.path(),
        ROW_BYTES as usize,
        rows,
        memory,
    )?;
    remove(&unordered)?;
    {
        let mut from = open(&ordered)?;
        let (mut to_keys, mut to_counts) = (create(&keys)?, create(&counts)?);
        let mut joined = vec![0u8; (CHUNK_ROWS * ROW_BYTES) as usize];
        let mut left = rows;
        while left > 0 {
            let now = CHUNK_ROWS.min(left) as usize;
            let joined = &mut joined[..now * ROW_BYTES as usize];
            from.read_exact(joined)
                .map_err(|error| failed("reading", &ordered, &error))?;
            for row in joined.as_chunks::<{ ROW_BYTES as usize }>().0 {
                let (key, count) = row.split_at(KEY_BYTES as usize);
                to_keys
                    .write_all(key)
                    .map_err(|error| failed("writing", &keys, &error))?;
                to_counts
                    .write_all(count)
                    .map_err(|error| failed("writing", &counts, &error))?;
            }
            left -= now as u64;
        }
        end(to_keys, &keys)?;
        end(to_counts, &counts)?;
    }
    remove(&ordered)
}

/// The rows of a scan on their way to disk.
pub(super) struct Spool {
    keys: BufWriter<File>,
    counts: BufWriter<File>,
    rows: u64,
    counted: u64,
    /// Whether every position so far was above the one before it.
    ordered: bool,
    last: u64,
    work: Work,
}

impl Spool {
    pub(super) fn create(work: Work) -> Result<Self, L2dError> {
        let open = |name: &str| {
            let path = work.file(name);
            File::create(&path)
                .map(|file| BufWriter::with_capacity(BUFFER_BYTES, file))
                .map_err(|error| failed("creating", &path, &error))
        };
        Ok(Self {
            keys: open(KEYS)?,
            counts: open(COUNTS)?,
            rows: 0,
            counted: 0,
            ordered: true,
            last: 0,
            work,
        })
    }

    pub(super) fn work(&self) -> &Path {
        self.work.path()
    }

    /// The next original: its first native position and its native identity.
    pub(super) fn push(&mut self, position: u64, id: &[u8; ID_BYTES]) -> Result<(), L2dError> {
        if self.rows.is_multiple_of(CHUNK_ROWS) {
            self.work.require_disk(
                CHUNK_ROWS * (KEY_BYTES + COUNT_BYTES),
                "its next 1,024 originals",
                &format!("{} originals were indexed, and nothing is kept", self.rows),
            )?;
        }
        let path = || self.work.file(KEYS);
        self.keys
            .write_all(&position.to_le_bytes())
            .and_then(|()| self.keys.write_all(id))
            .map_err(|error| failed("writing", &path(), &error))?;
        self.ordered &= self.rows == 0 || position > self.last;
        self.last = position;
        self.rows += 1;
        self.work.written += KEY_BYTES;
        Ok(())
    }

    /// The occurrence count of the next original, in the order of [`Spool::push`].
    pub(super) fn count(&mut self, count: u32) -> Result<(), L2dError> {
        self.counts
            .write_all(&count.to_le_bytes())
            .map_err(|error| failed("writing", &self.work.file(COUNTS), &error))?;
        self.counted += 1;
        self.work.written += COUNT_BYTES;
        Ok(())
    }

    /// Builds the lookup tables: in memory when `placement` allows it and `budget`
    /// admits the whole relation, on disk otherwise.
    pub(super) fn finish(
        self,
        placement: IndexPlacement,
        budget: &MemoryBudget,
    ) -> Result<Index, L2dError> {
        let Spool {
            keys,
            counts,
            rows,
            counted,
            ordered,
            last: _,
            mut work,
        } = self;
        if counted != rows {
            return Err(corrupt(&format!(
                "{rows} originals were read, and {counted} occurrence counts"
            )));
        }
        for (name, writer) in [(KEYS, keys), (COUNTS, counts)] {
            writer
                .into_inner()
                .map_err(|error| failed("writing", &work.file(name), &error.into_error()))?;
        }
        if !ordered {
            order_rows(&work, rows, budget)?;
        }
        let slots = rows.saturating_mul(2).next_power_of_two().max(16);
        let table = slots * SLOT_BYTES;
        let memory = rows * (KEY_BYTES + COUNT_BYTES) + 2 * table;
        let in_memory = placement == IndexPlacement::MemoryWhenAdmitted
            && budget
                .require_whole(memory, "keeping a scope's locator index in memory")
                .is_ok();
        let (keys, counts, by_position, by_id, disk_bytes);
        if in_memory {
            let load = |name: &str, bytes: u64| -> Result<Bytes, L2dError> {
                let path = work.file(name);
                let mut file = File::open(&path).map_err(|e| failed("opening", &path, &e))?;
                let mut loaded = vec![0u8; bytes as usize];
                file.read_exact(&mut loaded)
                    .map_err(|error| failed("reading", &path, &error))?;
                if file
                    .read(&mut [0u8])
                    .map_err(|e| failed("reading", &path, &e))?
                    != 0
                {
                    return Err(corrupt("a table on disk is longer than its rows"));
                }
                drop(file);
                fs::remove_file(&path).map_err(|error| failed("removing", &path, &error))?;
                Ok(Bytes::Memory(loaded))
            };
            keys = load(KEYS, rows * KEY_BYTES)?;
            counts = load(COUNTS, rows * COUNT_BYTES)?;
            by_position = Bytes::Memory(vec![0u8; table as usize]);
            by_id = Bytes::Memory(vec![0u8; table as usize]);
            disk_bytes = 0;
        } else {
            work.require_disk(
                2 * table,
                "its two lookup tables",
                &format!("{rows} originals were read, and nothing is kept"),
            )?;
            let read = |name: &str| {
                let path = work.file(name);
                File::open(&path)
                    .map(Bytes::Disk)
                    .map_err(|error| failed("opening", &path, &error))
            };
            let zeroed = |name: &str| {
                let path = work.file(name);
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .and_then(|file| file.set_len(table).map(|()| Bytes::Disk(file)))
                    .map_err(|error| failed("creating", &path, &error))
            };
            keys = read(KEYS)?;
            counts = read(COUNTS)?;
            by_position = zeroed(BY_POSITION)?;
            by_id = zeroed(BY_ID)?;
            work.written += 2 * table;
            disk_bytes = rows * (KEY_BYTES + COUNT_BYTES) + 2 * table;
            budget.note_spill();
        }
        let mut index = Index {
            keys,
            counts,
            by_position,
            by_id,
            rows,
            mask: slots - 1,
            occurrences: 0,
            digest: String::new(),
            disk_bytes,
            // The rows as they were written, and while they were put in order the
            // joined rows, their runs and the ordered rows beside one another.
            disk_peak: (rows * ROW_BYTES * if ordered { 1 } else { 3 }).max(disk_bytes),
            sorted: !ordered,
            work,
        };
        index.build()?;
        Ok(index)
    }
}

/// One original as the relation keeps it.
pub(super) struct Entry {
    pub(super) position: u64,
    pub(super) id: [u8; ID_BYTES],
    pub(super) count: u32,
}

/// Where a key is, or the free slot it would take.
enum Probe {
    Found(u64),
    Free(u64),
}

/// Follows the slots of `table` from `hash` until the row whose key bytes at `offset`
/// equal `key`, or a free slot. The tables are at most half full, so this ends.
fn probe(
    table: &Bytes,
    keys: &Bytes,
    mask: u64,
    hash: u64,
    offset: u64,
    key: &[u8],
) -> Result<Probe, L2dError> {
    let mut slot = hash & mask;
    let mut other = [0u8; ID_BYTES];
    let other = &mut other[..key.len()];
    for _ in 0..=mask {
        let mut stored = [0u8; SLOT_BYTES as usize];
        table.read(slot * SLOT_BYTES, &mut stored)?;
        let Some(row) = u64::from_le_bytes(stored).checked_sub(1) else {
            return Ok(Probe::Free(slot));
        };
        keys.read(row * KEY_BYTES + offset, other)?;
        if other == key {
            return Ok(Probe::Found(row));
        }
        slot = (slot + 1) & mask;
    }
    Err(corrupt("a lookup table has no free slot"))
}

/// Spreads a position over the slots (the finalizer of SplitMix64).
fn mix(position: u64) -> u64 {
    let mut value = position.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// The identities are SHA-256 digests: their first bytes are spread already.
fn id_hash(id: &[u8; ID_BYTES]) -> u64 {
    let mut first = [0u8; 8];
    first.copy_from_slice(&id[..8]);
    u64::from_le_bytes(first)
}

/// The relation of a scope: see the module description.
pub(super) struct Index {
    keys: Bytes,
    counts: Bytes,
    by_position: Bytes,
    by_id: Bytes,
    rows: u64,
    mask: u64,
    occurrences: u64,
    digest: String,
    disk_bytes: u64,
    disk_peak: u64,
    /// Whether the source handed its originals out in another order than their
    /// positions, so that the rows were sorted.
    sorted: bool,
    // Last, so that the files above are closed before they are removed.
    work: Work,
}

impl Index {
    /// Fills both tables from the rows, and takes the totals and the digest.
    fn build(&mut self) -> Result<(), L2dError> {
        let mut keys = vec![0u8; (CHUNK_ROWS * KEY_BYTES) as usize];
        let mut counts = vec![0u8; (CHUNK_ROWS * COUNT_BYTES) as usize];
        let mut sum = 0u128;
        let mut row = 0u64;
        while row < self.rows {
            let rows = CHUNK_ROWS.min(self.rows - row) as usize;
            let keys = &mut keys[..rows * KEY_BYTES as usize];
            let counts = &mut counts[..rows * COUNT_BYTES as usize];
            self.keys.read(row * KEY_BYTES, keys)?;
            self.counts.read(row * COUNT_BYTES, counts)?;
            let (keys, _) = keys.as_chunks::<{ KEY_BYTES as usize }>();
            let (counts, _) = counts.as_chunks::<{ COUNT_BYTES as usize }>();
            for (key, count) in keys.iter().zip(counts) {
                let (position, id) = key.split_at(POSITION_BYTES as usize);
                let position_hash = mix(u64::from_le_bytes(position.try_into().expect("8 bytes")));
                let id_hash = id_hash(id.try_into().expect("32 bytes"));
                let count_value = u32::from_le_bytes(*count);
                if count_value == 0 {
                    return Err(corrupt("an original has no occurrence"));
                }
                if count_value == u32::MAX {
                    return Err(L2dError::Refused {
                        reason: format!(
                            "an original of the source occurred {} times or more; the readers \
                             stop counting there, so its exact count is unknown and no scope is \
                             opened",
                            u32::MAX
                        ),
                    });
                }
                let stored = (row + 1).to_le_bytes();
                match probe(
                    &self.by_position,
                    &self.keys,
                    self.mask,
                    position_hash,
                    0,
                    position,
                )? {
                    Probe::Free(slot) => self.by_position.write(slot * SLOT_BYTES, &stored)?,
                    Probe::Found(_) => {
                        return Err(corrupt("two originals share one first position"));
                    }
                }
                match probe(
                    &self.by_id,
                    &self.keys,
                    self.mask,
                    id_hash,
                    POSITION_BYTES,
                    id,
                )? {
                    Probe::Free(slot) => self.by_id.write(slot * SLOT_BYTES, &stored)?,
                    Probe::Found(_) => {
                        return Err(corrupt("two originals share one native identity"));
                    }
                }
                // Order does not matter to a sum, so the digest is the same however the
                // source handed its originals out.
                let mut hash = Sha256::new();
                hash.update(key);
                hash.update(count);
                let mut part = [0u8; 16];
                part.copy_from_slice(&hash.finalize()[..16]);
                sum = sum.wrapping_add(u128::from_le_bytes(part));
                self.occurrences += u64::from(count_value);
                row += 1;
            }
        }
        self.digest = format!("{sum:032x}");
        Ok(())
    }

    pub(super) fn rows(&self) -> u64 {
        self.rows
    }

    pub(super) fn occurrences(&self) -> u64 {
        self.occurrences
    }

    /// A digest over every original's position, identity and count, whatever their
    /// order.
    pub(super) fn digest(&self) -> &str {
        &self.digest
    }

    pub(super) fn on_disk(&self) -> bool {
        self.disk_bytes > 0
    }

    pub(super) fn disk_bytes(&self) -> u64 {
        self.disk_bytes
    }

    /// The most the relation took in the work directory while it was built.
    pub(super) fn disk_peak(&self) -> u64 {
        self.disk_peak
    }

    pub(super) fn sorted(&self) -> bool {
        self.sorted
    }

    pub(super) fn held_bytes(&self) -> u64 {
        self.keys.held() + self.counts.held() + self.by_position.held() + self.by_id.held()
    }

    pub(super) fn entry(&self, row: u64) -> Result<Entry, L2dError> {
        if row >= self.rows {
            return Err(corrupt("a row lies beyond the originals"));
        }
        let mut key = [0u8; KEY_BYTES as usize];
        let mut count = [0u8; COUNT_BYTES as usize];
        self.keys.read(row * KEY_BYTES, &mut key)?;
        self.counts.read(row * COUNT_BYTES, &mut count)?;
        let (position, id) = key.split_at(POSITION_BYTES as usize);
        Ok(Entry {
            position: u64::from_le_bytes(position.try_into().expect("8 bytes")),
            id: id.try_into().expect("32 bytes"),
            count: u32::from_le_bytes(count),
        })
    }

    /// The counts, and with `keys` the positions and identities, of the rows from
    /// `row` on: as many rows as `counts` holds counts of four bytes.
    pub(super) fn read_rows(
        &self,
        row: u64,
        keys: Option<&mut [u8]>,
        counts: &mut [u8],
    ) -> Result<(), L2dError> {
        let rows = (counts.len() / COUNT_BYTES as usize) as u64;
        if row.checked_add(rows).is_none_or(|end| end > self.rows) {
            return Err(corrupt("a run of rows lies beyond the originals"));
        }
        if let Some(keys) = keys {
            debug_assert_eq!(keys.len() as u64, rows * KEY_BYTES);
            self.keys.read(row * KEY_BYTES, keys)?;
        }
        self.counts.read(row * COUNT_BYTES, counts)
    }

    /// The row of the original first read at `position`.
    pub(super) fn find_position(&self, position: u64) -> Result<Option<u64>, L2dError> {
        let key = position.to_le_bytes();
        let found = probe(
            &self.by_position,
            &self.keys,
            self.mask,
            mix(position),
            0,
            &key,
        )?;
        Ok(match found {
            Probe::Found(row) => Some(row),
            Probe::Free(_) => None,
        })
    }

    /// The row of the original with the native identity `id`.
    pub(super) fn find_id(&self, id: &[u8; ID_BYTES]) -> Result<Option<u64>, L2dError> {
        let found = probe(
            &self.by_id,
            &self.keys,
            self.mask,
            id_hash(id),
            POSITION_BYTES,
            id,
        )?;
        Ok(match found {
            Probe::Found(row) => Some(row),
            Probe::Free(_) => None,
        })
    }

    /// One bit per original for a join: in memory when `budget` admits it, in an
    /// unnamed file of the work directory otherwise.
    pub(super) fn seen(&self, budget: &MemoryBudget) -> Result<Seen, L2dError> {
        let bytes = self.rows.div_ceil(8).max(1);
        if budget
            .require_whole(bytes, "noting the originals a join has resolved")
            .is_ok()
        {
            return Ok(Seen {
                bits: Bytes::Memory(vec![0u8; bytes as usize]),
            });
        }
        self.work.require_disk(
            bytes,
            "the originals a join has resolved",
            "nothing was joined",
        )?;
        let file = tempfile::tempfile_in(self.work.path())
            .and_then(|file| file.set_len(bytes).map(|()| file))
            .map_err(|error| failed("creating a file in", self.work.path(), &error))?;
        budget.note_spill();
        Ok(Seen {
            bits: Bytes::Disk(file),
        })
    }
}

/// Which originals a join has resolved so far.
pub(super) struct Seen {
    bits: Bytes,
}

impl Seen {
    pub(super) fn held_bytes(&self) -> u64 {
        self.bits.held()
    }

    /// Notes `row`; `true` when it was noted before.
    pub(super) fn mark(&mut self, row: u64) -> Result<bool, L2dError> {
        let mut byte = [0u8];
        self.bits.read(row / 8, &mut byte)?;
        let bit = 1u8 << (row % 8);
        if byte[0] & bit != 0 {
            return Ok(true);
        }
        byte[0] |= bit;
        self.bits.write(row / 8, &byte)?;
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use l2d_core::MemorySettings;
    use l2d_core::memory::PROCESS_BASELINE_BYTES;

    use super::*;

    fn budget(usable: u64) -> MemoryBudget {
        let settings = MemorySettings {
            memory_limit_bytes: Some(PROCESS_BASELINE_BYTES + usable),
            ..MemorySettings::default()
        };
        MemoryBudget::observe_with(&settings, || Ok(64 << 30)).unwrap()
    }

    fn identity(position: u64) -> [u8; ID_BYTES] {
        let mut id = [0u8; ID_BYTES];
        id[..8].copy_from_slice(&mix(position).to_le_bytes());
        id
    }

    /// Rows handed out in another order than their positions are ordered by position,
    /// in memory and on disk, also when they are far more than one run of the sort.
    #[test]
    fn rows_are_in_the_order_of_their_positions_however_they_were_handed_out() {
        const ROWS: u64 = 30_000;
        for (placement, usable) in [
            (IndexPlacement::MemoryWhenAdmitted, 64 << 20),
            (IndexPlacement::Disk, 64 << 20),
            // 600 KiB: the sort gets what is left beside its buffers, and merges runs.
            (IndexPlacement::MemoryWhenAdmitted, 600 << 10),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let work = Work::create(&dir.path().join("work"), None).unwrap();
            let mut spool = Spool::create(work).unwrap();
            for at in 0..ROWS {
                // A fixed permutation of the positions 10, 20, 30, ...
                let position = ((at * 7_919 + 13) % ROWS + 1) * 10;
                spool.push(position, &identity(position)).unwrap();
                spool.count((position % 7 + 1) as u32).unwrap();
            }
            let budget = budget(usable);
            // Ordering and placing the rows allocates no more than the budget has
            // left: 30,000 rows are 1.3 MB, and the tightest budget leaves 600 KiB.
            let (index, peak) =
                crate::test_heap::measure(|| spool.finish(placement, &budget).unwrap());
            assert!(
                peak <= usable,
                "{placement:?}: the heap peaked at {peak} bytes with {usable} left"
            );
            assert!(index.sorted());
            assert_eq!(index.rows(), ROWS);
            assert_eq!(
                index.on_disk(),
                usable < (1 << 20) || placement == IndexPlacement::Disk
            );
            for row in [0, 1, 4_095, 4_096, ROWS / 2, ROWS - 1] {
                let entry = index.entry(row).unwrap();
                assert_eq!(entry.position, (row + 1) * 10, "{placement:?} {usable}");
                assert_eq!(entry.id, identity(entry.position));
                assert_eq!(u64::from(entry.count), entry.position % 7 + 1);
                assert_eq!(index.find_position(entry.position).unwrap(), Some(row));
                assert_eq!(index.find_id(&entry.id).unwrap(), Some(row));
            }
            let mut counts = [0u8; 12];
            let mut keys = [0u8; 120];
            index
                .read_rows(ROWS - 3, Some(&mut keys), &mut counts)
                .unwrap();
            assert_eq!(keys[..8], ((ROWS - 2) * 10).to_le_bytes());
            assert!(index.read_rows(ROWS - 2, None, &mut counts).is_err());
            // Only the relation's own files are left beside it.
            let left: Vec<String> = fs::read_dir(dir.path().join("work"))
                .unwrap()
                .map(|file| file.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            assert!(
                left.iter()
                    .all(|name| [KEYS, COUNTS, BY_POSITION, BY_ID].contains(&name.as_str())),
                "{left:?}"
            );
            drop(index);
            assert!(!dir.path().join("work").exists());
        }
    }

    /// The digest of the relation does not depend on the order of the hand-out.
    #[test]
    fn the_digest_is_the_same_for_every_order_of_the_hand_out() {
        let digest = |reversed: bool| {
            let dir = tempfile::tempdir().unwrap();
            let work = Work::create(&dir.path().join("work"), None).unwrap();
            let mut spool = Spool::create(work).unwrap();
            for at in 0..500u64 {
                let position = if reversed { 500 - at } else { at + 1 };
                spool.push(position, &identity(position)).unwrap();
                spool.count(3).unwrap();
            }
            let index = spool
                .finish(IndexPlacement::MemoryWhenAdmitted, &budget(64 << 20))
                .unwrap();
            index.digest().to_string()
        };
        assert_eq!(digest(false), digest(true));
    }
}
