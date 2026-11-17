//! Reading several entries of an archive at once (task 1.19).
//!
//! # What runs where
//!
//! A reader thread takes the next entry to be opened, in directory order, and reads it
//! as the calling thread would: it finds the entry's data, inflates it, splits it into
//! lines and judges every line as the validation pass does ([`super::judge`]: too long,
//! blank, not UTF-8, or a text with its row digest). It hands the lines on in pieces
//! ([`Piece`]): the texts one after the other, and per line what was judged. The last
//! piece of an entry says how the entry ended: read to its end, not read (with the
//! reason), or a failed read of the archive file.
//!
//! The calling thread takes the pieces entry by entry in directory order, and line by
//! line does what it does with a line it read itself: counts the record, notes the
//! skipped ones, hands the texts to the batch, cuts the batches, parses them,
//! deduplicates them, and goes to disk when the set is frozen. So the records arrive in
//! the order of a read of one entry after the other, every step after the reading is
//! the same code on the same thread, and the batches, the rows, the counts, the
//! summary and the point of every refusal are those of that read, whatever the timing
//! of the reader threads.
//!
//! # Order
//!
//! A reader takes an entry and notes which reader took it in one step, under one lock,
//! so the notes are in directory order. The calling thread takes the note of the entry
//! it is at, and then the pieces of that reader, which come in the order the reader
//! made them. A reader takes its next entry only after the last piece of the one
//! before, so the pieces of one reader are in directory order too.
//!
//! # Memory
//!
//! A reader makes at most [`PIECES_PER_READER`] pieces of a fixed size and reuses
//! them: the calling thread gives every piece back when it has taken its lines, and a
//! reader whose pieces are all on their way waits. What a reader holds is therefore
//! fixed ([`reader_bytes`]): the readers of one entry, the line being read, its pieces
//! and its stack. It is counted against the memory limit for every reader from the
//! first batch on, and a number of readers that does not fit is refused before a
//! reader is started. A reader that waits for pieces cannot hold up the entry the
//! calling thread is at: its pieces that are on their way are all of that entry, since
//! every entry before it has been taken.
//!
//! # The end
//!
//! When the batches are let go (at the end, on an error, or when the caller stops),
//! the readers are told to stop, the channels close, and the threads are joined before
//! anything else is let go.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use crate::error::L2dError;
use crate::zip_directory::{DataFault, DirectoryEntry, locate_data};

use super::{EntrySkip, Judged, MAX_ENTRY_LINES, STREAM_BYTES, entry_lines, judge, short_reason};

/// The text bytes a piece holds before it is handed on (at least one line of the
/// maximum record length).
pub(super) const PIECE_TEXT_BYTES: usize = 256 * 1024;
/// The most lines a piece holds.
pub(super) const PIECE_LINES: usize = 1024;
/// The most pieces a reader makes; it reuses them. 16 MiB of text a reader can read
/// ahead: while the calling thread parses and deduplicates a batch (up to 65,536 rows,
/// some 26 MB of text on the AIT archive) the readers go on. Measured on the AIT
/// archive (`ingest --entry *.log`, 512 MB, 2 entries at once, single runs, task 1.19):
/// 8 pieces 30.6 s, 64 pieces 27.5 s, 128 pieces 25.9 s, against 32.9 s for one entry
/// at a time; 64 was taken as the middle between time and memory.
pub(super) const PIECES_PER_READER: u64 = 64;
/// The stack of a reader thread: it is counted whole.
const READER_STACK_BYTES: usize = 256 * 1024;
/// What a reader holds beside its stack, its readers, its line and its pieces: the
/// thread's own records, the two channels, its note in the queue.
const READER_FIXED_BYTES: u64 = 16 * 1024;

/// One line of a piece: what was judged, and the bytes it took in the entry.
#[derive(Debug, Clone, Copy)]
pub(super) struct PieceLine {
    pub judged: Judged,
    /// The bytes of its text in [`Piece::text`]; 0 for a line that is not a text.
    pub text_bytes: u32,
    /// The bytes the line took in the entry, its terminator included.
    pub source_bytes: u64,
}

/// How an entry ended, in its last piece.
#[derive(Debug)]
pub(super) enum Ending {
    /// Read to its end.
    Read,
    /// Not read, or not to its end: why.
    Skipped(EntrySkip),
    /// A read of the archive file failed: the message.
    FileFailed(String),
}

/// Lines of one entry, in order, as a reader hands them on. The default holds nothing
/// and takes no memory.
#[derive(Debug, Default)]
pub(super) struct Piece {
    /// The index of the entry in the directory.
    pub entry: usize,
    /// The texts of the lines that are texts, one after the other.
    pub text: Vec<u8>,
    pub lines: Vec<PieceLine>,
    /// Set on the last piece of an entry.
    pub end: Option<Ending>,
}

impl Piece {
    fn new(text_capacity: usize) -> Self {
        Self {
            entry: 0,
            text: Vec::with_capacity(text_capacity),
            lines: Vec::with_capacity(PIECE_LINES),
            end: None,
        }
    }
}

/// The text a piece can hold without growing: a line of the maximum record length
/// always fits an empty piece.
fn piece_text_capacity(max_record_bytes: u64) -> usize {
    PIECE_TEXT_BYTES.max(max_record_bytes as usize)
}

/// What one reader thread holds while it reads, in bytes: the readers of one entry, the
/// line being read (counted twice, as for a read on the calling thread), its pieces,
/// its stack and its records.
pub(super) fn reader_bytes(max_record_bytes: u64) -> u64 {
    let piece = piece_text_capacity(max_record_bytes) as u64
        + (PIECE_LINES * size_of::<PieceLine>()) as u64
        + size_of::<Piece>() as u64;
    STREAM_BYTES
        + 2 * max_record_bytes
        + PIECES_PER_READER * piece
        + READER_STACK_BYTES as u64
        + READER_FIXED_BYTES
}

/// The archive file read at a position of its own: several readers share one handle
/// and none moves the position of another.
struct FileAt {
    file: Arc<File>,
    at: u64,
}

impl Read for FileAt {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        #[cfg(windows)]
        let count = std::os::windows::fs::FileExt::seek_read(&*self.file, out, self.at)?;
        #[cfg(unix)]
        let count = std::os::unix::fs::FileExt::read_at(&*self.file, out, self.at)?;
        self.at += count as u64;
        Ok(count)
    }
}

impl Seek for FileAt {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let beyond = || std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek out of range");
        self.at = match to {
            SeekFrom::Start(at) => at,
            SeekFrom::Current(by) => self.at.checked_add_signed(by).ok_or_else(beyond)?,
            SeekFrom::End(by) => self
                .file
                .metadata()?
                .len()
                .checked_add_signed(by)
                .ok_or_else(beyond)?,
        };
        Ok(self.at)
    }
}

/// Which entries are taken: by whom, in order.
struct Claims {
    /// The next of the entries to be opened to take.
    next: usize,
    /// Entry and reader, in the order they were taken (directory order).
    taken: VecDeque<(usize, usize)>,
    /// The readers still running.
    running: usize,
}

struct Shared {
    claims: Mutex<Claims>,
    /// Notified when an entry is taken and when a reader ends.
    changed: Condvar,
    stop: AtomicBool,
}

impl Shared {
    fn claims(&self) -> MutexGuard<'_, Claims> {
        self.claims.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The calling thread is gone or asked the readers to stop.
struct Gone;

/// One reader thread.
struct Reader {
    id: usize,
    file: Arc<File>,
    file_bytes: u64,
    directory: Arc<Vec<DirectoryEntry>>,
    /// The entries to be opened, by index, in directory order.
    opened: Arc<Vec<u32>>,
    max_record_bytes: u64,
    shared: Arc<Shared>,
    full: Sender<Piece>,
    empty: Receiver<Piece>,
    made: u64,
    /// The line buffer of the entry read last, for the next one.
    spare: Vec<u8>,
    line: Vec<u8>,
}

/// Notes the end of a reader, also when it panics.
struct Running(Arc<Shared>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.claims().running -= 1;
        self.0.changed.notify_all();
    }
}

impl Reader {
    fn run(mut self) {
        let _running = Running(Arc::clone(&self.shared));
        while let Some(index) = self.claim() {
            if self.read_entry(index).is_err() {
                return;
            }
        }
    }

    /// Takes the next entry and notes it; `None` when there is none or the readers
    /// are to stop.
    fn claim(&self) -> Option<usize> {
        let mut claims = self.shared.claims();
        if self.shared.stop.load(Ordering::Relaxed) || claims.next == self.opened.len() {
            return None;
        }
        let index = self.opened[claims.next] as usize;
        claims.next += 1;
        claims.taken.push_back((index, self.id));
        drop(claims);
        self.shared.changed.notify_all();
        Some(index)
    }

    /// An empty piece: one given back, a new one while fewer than
    /// [`PIECES_PER_READER`] are made, or one given back after a wait.
    fn piece(&mut self, entry: usize) -> Result<Piece, Gone> {
        let mut piece = match self.empty.try_recv() {
            Ok(piece) => piece,
            Err(TryRecvError::Empty) if self.made < PIECES_PER_READER => {
                self.made += 1;
                Piece::new(piece_text_capacity(self.max_record_bytes))
            }
            Err(TryRecvError::Empty) => self.empty.recv().map_err(|_| Gone)?,
            Err(TryRecvError::Disconnected) => return Err(Gone),
        };
        piece.entry = entry;
        piece.text.clear();
        piece.lines.clear();
        piece.end = None;
        Ok(piece)
    }

    fn hand_on(&self, piece: Piece) -> Result<(), Gone> {
        if self.shared.stop.load(Ordering::Relaxed) {
            return Err(Gone);
        }
        self.full.send(piece).map_err(|_| Gone)
    }

    /// Reads the entry at `index` as [`super::ZipInput::open_entry`] and the batches
    /// would, and hands its lines on.
    fn read_entry(&mut self, index: usize) -> Result<(), Gone> {
        let mut piece = self.piece(index)?;
        let ending = self.read_lines(index, &mut piece)?;
        piece.end = Some(ending);
        self.hand_on(piece)
    }

    /// The lines of the entry into pieces; `piece` is the one being filled. How the
    /// entry ended.
    fn read_lines(&mut self, index: usize, piece: &mut Piece) -> Result<Ending, Gone> {
        let directory = Arc::clone(&self.directory);
        let entry = &directory[index];
        let mut data = FileAt {
            file: Arc::clone(&self.file),
            at: 0,
        };
        match locate_data(&mut data, entry, self.file_bytes) {
            Ok(at) => data.at = at,
            Err(DataFault::Damaged(reason)) => {
                return Ok(Ending::Skipped(EntrySkip::Damaged(short_reason(reason))));
            }
            Err(DataFault::Io(error)) => return Ok(Ending::FileFailed(error.to_string())),
        }
        let (lines, failure) = entry_lines(data, entry, std::mem::take(&mut self.spare));
        if let Some(error) = failure.borrow().as_ref() {
            return Ok(Ending::FileFailed(error.clone()));
        }
        let mut lines = match lines {
            Ok(lines) => lines,
            Err(skip) => return Ok(Ending::Skipped(skip)),
        };
        let capacity = piece_text_capacity(self.max_record_bytes);
        let mut line_number = 0u64;
        let ending = loop {
            let read = if line_number == MAX_ENTRY_LINES {
                Err(format!("it has more than {MAX_ENTRY_LINES} lines"))
            } else {
                lines
                    .next_line_until(&mut self.line, self.max_record_bytes, || {
                        self.shared.stop.load(Ordering::Relaxed)
                    })
                    .map_err(|error| format!("reading stopped after line {line_number}: {error}"))
            };
            let read = match read {
                Ok(Some(read)) => read,
                Ok(None) => break Ending::Read,
                Err(reason) => {
                    if self.shared.stop.load(Ordering::Relaxed) {
                        return Err(Gone);
                    }
                    // The file could not be read: the run fails. Anything else is
                    // the entry's data.
                    if let Some(error) = failure.borrow().as_ref() {
                        break Ending::FileFailed(error.clone());
                    }
                    break Ending::Skipped(EntrySkip::Damaged(short_reason(reason)));
                }
            };
            line_number += 1;
            let judged = judge(&self.line, &read);
            let text_bytes = match judged {
                Judged::Text(_) => self.line.len(),
                _ => 0,
            };
            if piece.lines.len() == PIECE_LINES || piece.text.len() + text_bytes > capacity {
                // Handed on before an empty one is asked for: a reader never holds a
                // full piece while it waits.
                self.hand_on(std::mem::take(piece))?;
                *piece = self.piece(index)?;
            }
            piece.text.extend_from_slice(&self.line[..text_bytes]);
            piece.lines.push(PieceLine {
                judged,
                text_bytes: text_bytes as u32,
                source_bytes: read.source_bytes,
            });
        };
        self.spare = lines.buffer;
        Ok(ending)
    }
}

/// The reader threads of one read, and the calling thread's end of their channels.
pub(super) struct Readers {
    shared: Arc<Shared>,
    full: Vec<Receiver<Piece>>,
    empty: Vec<Sender<Piece>>,
    threads: Vec<JoinHandle<()>>,
}

impl Readers {
    /// Starts `count` readers over the entries `opened` (indices in directory order).
    pub(super) fn start(
        file: &File,
        file_bytes: u64,
        directory: &Arc<Vec<DirectoryEntry>>,
        opened: Vec<u32>,
        max_record_bytes: u64,
        count: usize,
    ) -> std::io::Result<Self> {
        let file = Arc::new(file.try_clone()?);
        let opened = Arc::new(opened);
        let shared = Arc::new(Shared {
            claims: Mutex::new(Claims {
                next: 0,
                taken: VecDeque::with_capacity(count * PIECES_PER_READER as usize + 1),
                running: count,
            }),
            changed: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let mut readers = Self {
            shared: Arc::clone(&shared),
            full: Vec::with_capacity(count),
            empty: Vec::with_capacity(count),
            threads: Vec::with_capacity(count),
        };
        for id in 0..count {
            let (full_in, full_out) = channel();
            let (empty_in, empty_out) = channel();
            let reader = Reader {
                id,
                file: Arc::clone(&file),
                file_bytes,
                directory: Arc::clone(directory),
                opened: Arc::clone(&opened),
                max_record_bytes,
                shared: Arc::clone(&shared),
                full: full_in,
                empty: empty_out,
                made: 0,
                spare: Vec::new(),
                line: Vec::with_capacity(max_record_bytes as usize + 1),
            };
            let spawned = std::thread::Builder::new()
                .name(format!("l2d-zip-reader-{id}"))
                .stack_size(READER_STACK_BYTES)
                .spawn(move || reader.run());
            match spawned {
                Ok(thread) => {
                    readers.full.push(full_out);
                    readers.empty.push(empty_in);
                    readers.threads.push(thread);
                }
                Err(error) => {
                    // The readers not started are not running; the others stop when
                    // `readers` is let go.
                    shared.claims().running -= count - id;
                    return Err(error);
                }
            }
        }
        Ok(readers)
    }

    /// The first piece of the entry at `index`, which must be the next entry to be
    /// opened, and the reader that reads it.
    pub(super) fn first(&mut self, index: usize) -> Result<(usize, Piece), L2dError> {
        let mut claims = self.shared.claims();
        let reader = loop {
            if let Some(&(entry, reader)) = claims.taken.front() {
                claims.taken.pop_front();
                if entry != index {
                    return Err(L2dError::Runtime {
                        detail: format!(
                            "the readers of the archive took entry {} where entry {} was next",
                            entry + 1,
                            index + 1
                        ),
                    });
                }
                break reader;
            }
            if claims.running == 0 {
                return Err(stopped(index));
            }
            claims = self
                .shared
                .changed
                .wait(claims)
                .unwrap_or_else(PoisonError::into_inner);
        };
        drop(claims);
        Ok((reader, self.next(reader, index)?))
    }

    /// The next piece of `reader`, which reads the entry at `index`.
    pub(super) fn next(&mut self, reader: usize, index: usize) -> Result<Piece, L2dError> {
        let piece = self.full[reader].recv().map_err(|_| stopped(index))?;
        if piece.entry != index {
            return Err(L2dError::Runtime {
                detail: format!(
                    "a reader of the archive handed on a line of entry {} where entry {} was \
                     read",
                    piece.entry + 1,
                    index + 1
                ),
            });
        }
        Ok(piece)
    }

    /// Gives a piece whose lines were taken back to its reader.
    pub(super) fn give_back(&mut self, reader: usize, piece: Piece) {
        // A reader that ended no longer needs it.
        let _ = self.empty[reader].send(piece);
    }
}

/// A reader ended before it handed on the end of the entry at `index`: it panicked,
/// or could not be started.
fn stopped(index: usize) -> L2dError {
    L2dError::Runtime {
        detail: format!(
            "a reader of the archive stopped before the end of entry {}",
            index + 1
        ),
    }
}

impl Drop for Readers {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.changed.notify_all();
        // Closing the channels wakes a reader that waits for a piece or hands one on.
        self.full.clear();
        self.empty.clear();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod hardening_tests {
    use super::*;
    use crate::zip_input::EntryLines;
    use std::sync::atomic::AtomicUsize;

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("l2d-reader-{name}-{}", std::process::id()))
    }

    #[test]
    fn panicking_reader_wakes_the_caller_and_joins() {
        let path = scratch("panic");
        let file = File::create(&path).unwrap();
        // The real reader claims an entry and then panics looking it up. Its Running
        // guard and channel teardown must wake a caller waiting for the first piece.
        let mut readers = Readers::start(&file, 0, &Arc::new(vec![]), vec![0], 1024, 1).unwrap();
        let error = readers.first(0).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("stopped before the end of entry 1")
        );
        drop(readers);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn failed_file_read_inside_a_reader_is_reported_and_joins() {
        let path = scratch("io");
        // A real write-only handle: seek_read/read_at fails inside the reader thread.
        let file = File::create(&path).unwrap();
        let entry = DirectoryEntry {
            method: 0,
            flags: 0,
            crc32: 0,
            compressed_size: 1,
            size: 1,
            header_at: 0,
            kind: crate::zip_directory::EntryKind::File,
        };
        let mut readers =
            Readers::start(&file, 64, &Arc::new(vec![entry]), vec![0], 1024, 1).unwrap();
        let (_, piece) = readers.first(0).unwrap();
        assert!(matches!(piece.end, Some(Ending::FileFailed(ref reason)) if !reason.is_empty()));
        drop(readers);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cancellation_is_checked_inside_an_overlong_line() {
        struct LongLine(Arc<AtomicUsize>);
        impl Read for LongLine {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let reads = self.0.fetch_add(1, Ordering::Relaxed);
                assert!(
                    reads < 4,
                    "reader kept draining the long line after cancellation"
                );
                out.fill(b'x');
                Ok(out.len())
            }
        }
        let reads = Arc::new(AtomicUsize::new(0));
        let mut lines = EntryLines::new(&[], Box::new(LongLine(Arc::clone(&reads))), Vec::new());
        let mut line = Vec::new();
        let error = lines.next_line_until(&mut line, 32, || reads.load(Ordering::Relaxed) > 0);
        assert!(error.is_err());
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        assert!(line.len() <= 33);
    }
}
