//! The tables of a selection or view: fixed-width cells for every original of a
//! snapshot, in memory or in a file of the service's work directory (triager
//! requirement 6.2). Both placements are read and written through the same calls, so
//! an expression over sets on disk gives what it gives over sets in memory.
//!
//! A table in memory is held in the budget by the set that owns it. A table on disk
//! holds nothing: whoever reads or writes it does so in blocks through a buffer of its
//! own, which it has asked the budget for, or one cell at a time without a buffer.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use l2d_core::L2dError;
use l2d_core::memory::exact;
use l2d_core::parts::free_disk_bytes;

use crate::scope::{read_at, write_at};

/// Where selections and views keep their membership and scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TablePlacement {
    /// In memory when the budget admits all of it, on disk otherwise.
    #[default]
    MemoryWhenAdmitted,
    /// In the work directory.
    Disk,
}

/// The rows every reader and writer of a table takes at a time.
pub(crate) const BLOCK_ROWS: u64 = 16_384;

fn failed(doing: &str, path: &Path, error: &dyn std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

/// [`L2dError::Refused`] with both figures unless `bytes` more fit the disk that holds
/// `dir`; `given` stands in for the operating system's figure.
pub(crate) fn require_disk(
    dir: &Path,
    bytes: u64,
    given: Option<u64>,
    what: &str,
) -> Result<(), L2dError> {
    let free = match given {
        Some(free) => free,
        None => free_disk_bytes(dir)?,
    };
    if bytes <= free {
        return Ok(());
    }
    Err(L2dError::Refused {
        reason: format!(
            "{what} needs {} of disk space in the work directory {}, but {} is free there; \
             nothing was committed",
            exact(bytes),
            dir.display(),
            exact(free)
        ),
    })
}

/// The directory of one set's tables; removed when the last table in it is let go.
pub(crate) struct SetDir(PathBuf);

impl SetDir {
    pub fn create(path: PathBuf) -> Result<Arc<Self>, L2dError> {
        std::fs::create_dir_all(&path).map_err(|error| failed("creating", &path, &error))?;
        Ok(Arc::new(Self(path)))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for SetDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}

enum Backing {
    Memory(Vec<u8>),
    Disk {
        file: File,
        path: PathBuf,
        // After the file, so the file is removed before its directory.
        _dir: Arc<SetDir>,
    },
}

/// `len` bytes, in memory or in a file.
pub(crate) struct Table {
    backing: Backing,
    len: u64,
}

impl Table {
    /// `len` bytes of `fill` in memory. The caller has asked the budget for them.
    pub fn in_memory(len: u64, fill: u8) -> Self {
        Self {
            backing: Backing::Memory(vec![fill; len as usize]),
            len,
        }
    }

    /// `len` zero bytes in the new file `name` of `dir`.
    pub fn on_disk(dir: &Arc<SetDir>, name: &str, len: u64) -> Result<Self, L2dError> {
        let path = dir.path().join(name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .and_then(|file| file.set_len(len).map(|()| file))
            .map_err(|error| failed("creating", &path, &error))?;
        Ok(Self {
            backing: Backing::Disk {
                file,
                path,
                _dir: dir.clone(),
            },
            len,
        })
    }

    /// The file `name` of `dir`, written before, as a table of its `len` bytes: a table
    /// that was put in order on disk is kept where it is, not copied. The file is
    /// removed with the table.
    pub fn adopt(dir: &Arc<SetDir>, name: &str, len: u64) -> Result<Self, L2dError> {
        let path = dir.path().join(name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| failed("opening", &path, &error))?;
        let found = file
            .metadata()
            .map_err(|error| failed("reading the length of", &path, &error))?
            .len();
        if found != len {
            return Err(L2dError::Runtime {
                detail: format!(
                    "{} holds {found} bytes where {len} were written",
                    path.display()
                ),
            });
        }
        Ok(Self {
            backing: Backing::Disk {
                file,
                path,
                _dir: dir.clone(),
            },
            len,
        })
    }

    /// `bytes` in memory as a table. The caller has asked the budget for them.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let len = bytes.len() as u64;
        Self {
            backing: Backing::Memory(bytes),
            len,
        }
    }

    pub fn byte_len(&self) -> u64 {
        self.len
    }

    pub fn is_on_disk(&self) -> bool {
        matches!(self.backing, Backing::Disk { .. })
    }

    /// What the table takes in memory.
    pub fn held_bytes(&self) -> u64 {
        match &self.backing {
            Backing::Memory(bytes) => bytes.capacity() as u64,
            Backing::Disk { .. } => 0,
        }
    }

    /// What the table takes in the work directory.
    pub fn disk_bytes(&self) -> u64 {
        match &self.backing {
            Backing::Memory(_) => 0,
            Backing::Disk { .. } => self.len,
        }
    }

    fn beyond() -> L2dError {
        L2dError::Runtime {
            detail: "a cell lies beyond its table".into(),
        }
    }

    pub fn read(&self, at: u64, out: &mut [u8]) -> Result<(), L2dError> {
        if at
            .checked_add(out.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(Self::beyond());
        }
        match &self.backing {
            Backing::Memory(bytes) => {
                out.copy_from_slice(&bytes[at as usize..at as usize + out.len()]);
                Ok(())
            }
            Backing::Disk { file, path, .. } => {
                read_at(file, at, out).map_err(|error| failed("reading", path, &error))
            }
        }
    }

    pub fn write(&mut self, at: u64, data: &[u8]) -> Result<(), L2dError> {
        if at
            .checked_add(data.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(Self::beyond());
        }
        match &mut self.backing {
            Backing::Memory(bytes) => {
                bytes[at as usize..at as usize + data.len()].copy_from_slice(data);
                Ok(())
            }
            Backing::Disk { file, path, .. } => {
                write_at(file, at, data).map_err(|error| failed("writing", path, &error))
            }
        }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        let backing = std::mem::replace(&mut self.backing, Backing::Memory(Vec::new()));
        if let Backing::Disk { file, path, _dir } = backing {
            drop(file);
            let _ = std::fs::remove_file(&path);
            // The directory goes after its file, when this was its last table.
            drop(_dir);
        }
    }
}

/// The score of every original of a snapshot under one frozen scorer: four bytes a
/// row, not a number where the scorer gave none or was not asked.
pub(crate) struct Scores {
    table: Table,
    rows: u64,
}

impl Scores {
    /// The bytes of one block of scores.
    pub const BLOCK_BYTES: usize = BLOCK_ROWS as usize * 4;

    pub fn bytes_for(rows: u64) -> u64 {
        rows.saturating_mul(4).max(4)
    }

    /// No score for any row, in memory. The caller has asked the budget for
    /// [`Scores::bytes_for`].
    pub fn in_memory(rows: u64) -> Self {
        let mut table = Table::in_memory(Self::bytes_for(rows), 0);
        if let Backing::Memory(bytes) = &mut table.backing {
            for cell in bytes.as_chunks_mut::<4>().0 {
                *cell = f32::NAN.to_le_bytes();
            }
        }
        Self { table, rows }
    }

    /// A file for the scores of `rows` rows, to be written block by block.
    pub fn on_disk(dir: &Arc<SetDir>, rows: u64) -> Result<Self, L2dError> {
        Ok(Self {
            table: Table::on_disk(dir, "scores.bin", Self::bytes_for(rows))?,
            rows,
        })
    }

    pub fn held_bytes(&self) -> u64 {
        self.table.held_bytes()
    }

    pub fn disk_bytes(&self) -> u64 {
        self.table.disk_bytes()
    }

    pub fn get(&self, row: u64) -> Result<f32, L2dError> {
        let mut cell = [0u8; 4];
        self.table.read(row * 4, &mut cell)?;
        Ok(f32::from_le_bytes(cell))
    }

    pub fn set(&mut self, row: u64, score: f32) -> Result<(), L2dError> {
        self.table.write(row * 4, &score.to_le_bytes())
    }

    /// Reads the scores of the block that starts at `first` into `out`, which holds
    /// [`Scores::BLOCK_BYTES`], and returns how many rows the block has.
    pub fn read_block(&self, first: u64, out: &mut [u8]) -> Result<usize, L2dError> {
        let rows = BLOCK_ROWS.min(self.rows.saturating_sub(first)) as usize;
        self.table.read(first * 4, &mut out[..rows * 4])?;
        Ok(rows)
    }

    /// Writes the first `rows` scores of `block` as the block that starts at `first`.
    pub fn write_block(&mut self, first: u64, block: &[u8], rows: usize) -> Result<(), L2dError> {
        self.table.write(first * 4, &block[..rows * 4])
    }
}

/// The score at `at` of a block read with [`Scores::read_block`].
pub(crate) fn score_in(block: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(block[at * 4..at * 4 + 4].try_into().expect("4 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_placements_read_what_was_written_and_a_file_goes_with_its_table() {
        let root = tempfile::tempdir().unwrap();
        let dir = SetDir::create(root.path().join("set-1")).unwrap();
        let mut tables = [
            Table::in_memory(10_000, 0),
            Table::on_disk(&dir, "cells.bin", 10_000).unwrap(),
        ];
        for table in &mut tables {
            table.write(9_990, &[7u8; 10]).unwrap();
            table.write(0, &[1, 2, 3]).unwrap();
            let mut out = [9u8; 12];
            table.read(9_988, &mut out).unwrap();
            assert_eq!(out, [0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]);
            table.read(1, &mut out[..3]).unwrap();
            assert_eq!(out[..3], [2, 3, 0]);
            assert!(table.read(9_995, &mut out).is_err());
            assert!(table.write(10_000, &[1]).is_err());
        }
        assert_eq!(tables[0].held_bytes(), 10_000);
        assert_eq!(tables[0].disk_bytes(), 0);
        assert!(tables[1].is_on_disk());
        assert_eq!(tables[1].held_bytes(), 0);
        assert_eq!(tables[1].disk_bytes(), 10_000);
        let path = dir.path().to_path_buf();
        drop(dir);
        assert!(path.join("cells.bin").exists());
        drop(tables);
        assert!(!path.exists(), "the directory goes with its last table");
    }

    #[test]
    fn scores_start_as_no_number_in_memory_and_are_written_in_blocks_on_disk() {
        let root = tempfile::tempdir().unwrap();
        let dir = SetDir::create(root.path().join("set-1")).unwrap();
        let mut memory = Scores::in_memory(5);
        assert!(memory.get(4).unwrap().is_nan());
        memory.set(2, 0.5).unwrap();
        assert_eq!(memory.get(2).unwrap(), 0.5);
        assert_eq!(memory.held_bytes(), 20);
        let mut disk = Scores::on_disk(&dir, BLOCK_ROWS + 3).unwrap();
        let mut block = vec![0u8; Scores::BLOCK_BYTES];
        block[..4].copy_from_slice(&1.5f32.to_le_bytes());
        block[8..12].copy_from_slice(&f32::NAN.to_le_bytes());
        disk.write_block(BLOCK_ROWS, &block, 3).unwrap();
        assert_eq!(disk.read_block(BLOCK_ROWS, &mut block).unwrap(), 3);
        assert_eq!(score_in(&block, 0), 1.5);
        assert!(score_in(&block, 2).is_nan());
        assert_eq!(disk.get(BLOCK_ROWS).unwrap(), 1.5);
        assert_eq!(disk.disk_bytes(), (BLOCK_ROWS + 3) * 4);
    }

    #[test]
    fn a_disk_without_room_is_refused_with_both_figures() {
        let root = tempfile::tempdir().unwrap();
        assert!(require_disk(root.path(), 4_096, Some(4_096), "a table").is_ok());
        match require_disk(root.path(), 4_097, Some(4_096), "the membership of a view") {
            Err(L2dError::Refused { reason }) => {
                assert!(
                    reason.starts_with("the membership of a view needs "),
                    "{reason}"
                );
                assert!(reason.contains("(4097 bytes) of disk space"), "{reason}");
                assert!(reason.contains("(4096 bytes) is free"), "{reason}");
            }
            other => panic!("{other:?}"),
        }
        assert!(require_disk(root.path(), 1, None, "a table").is_ok());
    }
}
