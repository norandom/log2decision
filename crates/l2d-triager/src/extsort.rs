//! Sorting fixed-width records of a file by their first eight bytes within a stated
//! amount of memory (triager requirement 6.2): the relation of a scope is ordered by
//! record ID with it, and a membership that does not fit the memory is placed with it.
//!
//! The records are read in runs that fit the memory, each run is sorted and written to
//! a file of its own, and the runs are merged into the output. A file that fits the
//! memory as a whole is one run and is written straight to the output. The memory a
//! sort takes is what its caller asked the budget for: [`sort_memory`] says how much to
//! ask for, and [`sort_file`] stays within what it is given.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use l2d_core::L2dError;
use l2d_core::memory::exact;

/// The write buffer of a sort.
pub(crate) const IO_BYTES: u64 = 64 * 1024;
/// The least memory a sort works with.
pub(crate) const MIN_SORT_BYTES: u64 = 256 * 1024;
/// The most memory a sort asks for, however many records it has.
pub(crate) const MAX_SORT_BYTES: u64 = 32 * 1024 * 1024;
/// What a record takes in memory beside its bytes: its place in the order.
const ORDER_BYTES: u64 = 4;
/// What a run takes in a merge beside its buffer: its file and its place in the heap.
const RUN_FIXED_BYTES: u64 = 128;
/// The fewest records of a run a merge buffers.
const MIN_MERGE_RECORDS: u64 = 8;
/// What a sort holds beside its records and buffers: the paths of its files and the
/// lists of its runs.
const SORT_FIXED_BYTES: u64 = 4 * 1024;

/// The memory to ask the budget for, to sort `records` records of `width` bytes when
/// `left` bytes are left: all of them at once when that fits, and otherwise as much as
/// is left, between [`MIN_SORT_BYTES`] and [`MAX_SORT_BYTES`].
pub(crate) fn sort_memory(records: u64, width: usize, left: u64) -> u64 {
    let whole = records
        .saturating_mul(width as u64 + ORDER_BYTES)
        .saturating_add(IO_BYTES + SORT_FIXED_BYTES);
    whole.min(left).clamp(MIN_SORT_BYTES, MAX_SORT_BYTES)
}

fn failed(doing: &str, path: &Path, error: &dyn std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

/// The key of a record: its first eight bytes, little-endian.
fn key(record: &[u8]) -> u64 {
    let mut first = [0u8; 8];
    first.copy_from_slice(&record[..8]);
    u64::from_le_bytes(first)
}

/// The run files of one sort; removed with it, however it ends.
struct Runs {
    dir: PathBuf,
    count: usize,
}

impl Runs {
    fn path(&self, run: usize) -> PathBuf {
        self.dir.join(format!("sort-run-{run}.bin"))
    }
}

impl Drop for Runs {
    fn drop(&mut self) {
        for run in 0..self.count {
            let _ = fs::remove_file(self.path(run));
        }
    }
}

/// One run of a merge: its file and the records of it that are buffered.
struct Run {
    file: File,
    buffer: Vec<u8>,
    at: usize,
    filled: usize,
    left: u64,
}

impl Run {
    /// The next record of the run, `None` at its end.
    fn next(&mut self, width: usize, path: &Path) -> Result<Option<&[u8]>, L2dError> {
        if self.at == self.filled {
            if self.left == 0 {
                return Ok(None);
            }
            let records = (self.buffer.len() / width).min(self.left as usize);
            self.file
                .read_exact(&mut self.buffer[..records * width])
                .map_err(|error| failed("reading", path, &error))?;
            self.left -= records as u64;
            self.at = 0;
            self.filled = records * width;
        }
        let record = &self.buffer[self.at..self.at + width];
        self.at += width;
        Ok(Some(record))
    }
}

/// What a sort did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sorted {
    /// The runs written beside the output; 0 when the file was sorted in memory.
    pub runs: u64,
    /// The bytes of those runs.
    pub run_bytes: u64,
}

/// Writes the `records` records of `width` bytes in `input` to `output`, ordered by
/// their first eight bytes as a little-endian number. Records of one key keep no
/// stated order. Run files are written into `dir` and removed again.
///
/// `memory` is what the caller asked the budget for; the sort allocates no more.
/// [`L2dError::Refused`] with the figures when the records need more runs than
/// `memory` can merge at once.
pub(crate) fn sort_file(
    input: &Path,
    output: &Path,
    dir: &Path,
    width: usize,
    records: u64,
    memory: u64,
) -> Result<Sorted, L2dError> {
    debug_assert!(width >= 8);
    let room = memory.saturating_sub(IO_BYTES + SORT_FIXED_BYTES);
    let run_records = (room / (width as u64 + ORDER_BYTES)).max(1);
    let run_count = records.div_ceil(run_records).max(1);
    let per_run = if run_count > 1 {
        let each = room / run_count;
        let buffered = each.saturating_sub(RUN_FIXED_BYTES) / width as u64;
        if buffered < MIN_MERGE_RECORDS {
            return Err(L2dError::Refused {
                reason: format!(
                    "ordering {records} records of {width} bytes within {} of memory takes \
                     {run_count} runs, more than that memory merges at once; it takes {} for \
                     this, raise the memory limit",
                    exact(memory),
                    exact(
                        run_count * (MIN_MERGE_RECORDS * width as u64 + RUN_FIXED_BYTES)
                            + IO_BYTES
                            + SORT_FIXED_BYTES
                    )
                ),
            });
        }
        buffered
    } else {
        0
    };
    let mut source = File::open(input).map_err(|error| failed("opening", input, &error))?;
    let create = |path: &Path| {
        File::create(path)
            .map(|file| BufWriter::with_capacity(IO_BYTES as usize, file))
            .map_err(|error| failed("creating", path, &error))
    };
    let flush = |writer: BufWriter<File>, path: &Path| {
        writer
            .into_inner()
            .map(drop)
            .map_err(|error| failed("writing", path, &error.into_error()))
    };
    let mut runs = Runs {
        dir: dir.to_path_buf(),
        count: 0,
    };
    let mut written = Sorted {
        runs: 0,
        run_bytes: 0,
    };
    {
        let most = run_records.min(records) as usize;
        let mut data = vec![0u8; most * width];
        let mut order: Vec<u32> = Vec::with_capacity(most);
        let mut left = records;
        loop {
            let now = (left.min(run_records)) as usize;
            let data = &mut data[..now * width];
            source
                .read_exact(data)
                .map_err(|error| failed("reading", input, &error))?;
            left -= now as u64;
            order.clear();
            order.extend(0..now as u32);
            order.sort_unstable_by_key(|at| key(&data[*at as usize * width..]));
            let path = if run_count == 1 {
                output.to_path_buf()
            } else {
                runs.count += 1;
                written.runs += 1;
                written.run_bytes += (now * width) as u64;
                runs.path(runs.count - 1)
            };
            let mut writer = create(&path)?;
            for at in &order {
                let from = *at as usize * width;
                writer
                    .write_all(&data[from..from + width])
                    .map_err(|error| failed("writing", &path, &error))?;
            }
            flush(writer, &path)?;
            if left == 0 {
                break;
            }
        }
    }
    drop(source);
    if run_count == 1 {
        return Ok(written);
    }
    // The merge: every run with its buffer, the lowest key of each in a heap.
    let mut open: Vec<Run> = Vec::with_capacity(runs.count);
    let mut heads: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::with_capacity(runs.count);
    for run in 0..runs.count {
        let path = runs.path(run);
        let file = File::open(&path).map_err(|error| failed("opening", &path, &error))?;
        let in_run = if run + 1 == runs.count {
            records - run_records * (runs.count as u64 - 1)
        } else {
            run_records
        };
        open.push(Run {
            file,
            buffer: vec![0u8; per_run as usize * width],
            at: 0,
            filled: 0,
            left: in_run,
        });
    }
    let mut writer = create(output)?;
    // The record at the head of each run is kept in its buffer; the heap names it.
    let mut pending: Vec<Option<(usize, usize)>> = vec![None; open.len()];
    for (run, state) in open.iter_mut().enumerate() {
        let path = runs.path(run);
        if let Some(record) = state.next(width, &path)? {
            heads.push(Reverse((key(record), run)));
            pending[run] = Some((state.at - width, state.at));
        }
    }
    while let Some(Reverse((_, run))) = heads.pop() {
        let path = runs.path(run);
        let state = &mut open[run];
        let (from, to) = pending[run].take().expect("a run in the heap has a record");
        writer
            .write_all(&state.buffer[from..to])
            .map_err(|error| failed("writing", output, &error))?;
        if let Some(record) = state.next(width, &path)? {
            heads.push(Reverse((key(record), run)));
            pending[run] = Some((state.at - width, state.at));
        }
    }
    flush(writer, output)?;
    drop(open);
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records(count: u64, width: usize) -> Vec<u8> {
        // A fixed permutation of the keys, each record filled with a byte of its key.
        let mut data = Vec::new();
        for at in 0..count {
            let key = (at * 7_919 + 13) % count;
            data.extend_from_slice(&key.to_le_bytes());
            data.extend(std::iter::repeat_n((key % 251) as u8, width - 8));
        }
        data
    }

    fn check(count: u64, width: usize, memory: u64) -> Sorted {
        let dir = tempfile::tempdir().unwrap();
        let (input, output) = (dir.path().join("in.bin"), dir.path().join("out.bin"));
        fs::write(&input, records(count, width)).unwrap();
        // The sort allocates no more than the memory it is given.
        let (sorted, peak) = crate::test_heap::measure(|| {
            sort_file(&input, &output, dir.path(), width, count, memory).unwrap()
        });
        assert!(
            peak <= memory,
            "{count} records of {width} bytes: the heap peaked at {peak} bytes of {memory}"
        );
        let data = fs::read(&output).unwrap();
        assert_eq!(data.len() as u64, count * width as u64);
        for (at, record) in data.chunks(width).enumerate() {
            assert_eq!(key(record), at as u64, "record {at}");
            assert!(record[8..].iter().all(|byte| *byte == (at % 251) as u8));
        }
        // Nothing but the input and the output is left.
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
        sorted
    }

    #[test]
    fn a_file_that_fits_is_sorted_in_one_run() {
        assert_eq!(check(5_000, 16, MIN_SORT_BYTES).runs, 0);
        assert_eq!(check(0, 44, MIN_SORT_BYTES).runs, 0);
        assert_eq!(check(1, 44, MIN_SORT_BYTES).runs, 0);
    }

    #[test]
    fn a_file_beyond_the_memory_is_merged_from_runs() {
        // 60,000 records of 44 bytes are 2.6 MB; of 256 KiB, 188 KiB are for records,
        // 4,010 of them at a time.
        let sorted = check(60_000, 44, MIN_SORT_BYTES);
        assert_eq!(sorted.runs, 15);
        assert_eq!(sorted.run_bytes, 60_000 * 44);
        let uneven = check(4_011, 44, MIN_SORT_BYTES);
        assert_eq!(uneven.runs, 2);
        assert_eq!(check(4_010, 44, MIN_SORT_BYTES).runs, 0);
        // Many runs with small buffers: 100 runs of 2,000 records, 17 buffered of each.
        let many = check(200_000, 16, IO_BYTES + SORT_FIXED_BYTES + 40_000);
        assert_eq!(many.runs, 100);
    }

    #[test]
    fn more_runs_than_the_memory_merges_are_refused_with_figures() {
        let dir = tempfile::tempdir().unwrap();
        let (input, output) = (dir.path().join("in.bin"), dir.path().join("out.bin"));
        fs::write(&input, records(40_000, 16)).unwrap();
        // 2 KiB beside the write buffer and the fixed share: runs of 102 records, 393
        // runs.
        let memory = IO_BYTES + SORT_FIXED_BYTES + 2_048;
        let refused = sort_file(&input, &output, dir.path(), 16, 40_000, memory);
        match refused {
            Err(L2dError::Refused { reason }) => {
                assert!(reason.contains("393 runs"), "{reason}");
                assert!(reason.contains("raise the memory limit"), "{reason}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn the_memory_asked_for_is_bounded_on_both_sides() {
        assert_eq!(sort_memory(10, 44, u64::MAX), MIN_SORT_BYTES);
        assert_eq!(sort_memory(100_000_000, 44, u64::MAX), MAX_SORT_BYTES);
        assert_eq!(
            sort_memory(100_000, 44, u64::MAX),
            100_000 * 48 + IO_BYTES + SORT_FIXED_BYTES
        );
        assert_eq!(sort_memory(100_000, 44, 1024 * 1024), 1024 * 1024);
        assert_eq!(sort_memory(100_000, 44, 0), MIN_SORT_BYTES);
    }
}
