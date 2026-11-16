//! Reading several entries of a zip archive at once (task 1.19; requirements 13.8,
//! 13.10, 12.6, 12.9 and 15): the batches, the rows, the occurrence counts and the
//! summary are those of reading one entry after the other, at any batch size, with a
//! damaged entry, with gzip entries and with deduplication on disk; the readers count
//! against the memory limit, and a number of them that does not fit is refused.

use std::io::Write;
use std::path::{Path, PathBuf};

use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::zip_input::{DEFAULT_ENTRIES_AT_ONCE, MAX_ENTRIES_AT_ONCE};
use l2d_core::{
    ArchiveSummary, DiskDedup, EntrySkip, L2dError, MemoryBudget, MemorySettings, ZipInput,
    ZipInputSettings,
};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

// --- Helpers -------------------------------------------------------------------------------

/// A scratch directory in the system temp folder, removed when dropped.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("l2d-core-parallel-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A budget with plenty of free memory; `limit` and `batch_rows` as the analyst's.
fn budget(limit: Option<u64>, batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap()
}

fn settings(at_once: Option<usize>) -> ZipInputSettings {
    ZipInputSettings {
        max_record_bytes: 200,
        entries_at_once: at_once,
        ..ZipInputSettings::default()
    }
}

/// What one read gave, everything that must not depend on how many entries are read
/// at once.
#[derive(Debug, PartialEq)]
struct Read {
    /// Every batch as it was handed out: text, packed first position, count in the
    /// batch (for a frame from disk, over the archive).
    batches: Vec<Vec<(String, u64, u32)>>,
    /// The count over the archive of every row, in row order.
    counts: Vec<u32>,
    summary: ArchiveSummary,
    /// What deduplication on disk did, when it was used.
    dedup: Option<DiskDedup>,
}

impl Read {
    /// The rows of all batches, one after the other.
    fn rows(&self) -> Vec<&(String, u64, u32)> {
        self.batches.iter().flatten().collect()
    }
}

/// Reads `path` to its end; with `spill`, deduplication on disk goes there, with at
/// most `set_texts` texts in memory. Also returns how many entries were read at once.
fn read_with(
    path: &Path,
    settings: &ZipInputSettings,
    budget: &mut MemoryBudget,
    spill: Option<(&Path, Option<u64>)>,
) -> Result<(Read, usize), L2dError> {
    let mut input = ZipInput::open(path, settings, budget)?;
    if let Some((dir, set_texts)) = spill {
        input.spill_into(dir, set_texts)?;
    }
    let mut batches = Vec::new();
    for batch in input.batches(budget) {
        let batch = batch?;
        assert!(!batch.is_empty(), "a batch without rows is not handed out");
        batches.push(
            batch
                .records()
                .map(|record| (record.text, record.first_position, record.occurrences))
                .collect::<Vec<_>>(),
        );
    }
    let counts = if spill.is_some() {
        let mut counts = Vec::new();
        for chunk in input.occurrence_counts() {
            counts.extend(chunk?);
        }
        counts
    } else {
        batches
            .iter()
            .flatten()
            .map(|(text, _, _)| u32::try_from(input.occurrences(text)).unwrap())
            .collect()
    };
    assert!(input.batch_plan().is_some(), "the plan of a completed read");
    let at_once = input.entries_at_once();
    let dedup = input.disk_dedup();
    Ok((
        Read {
            batches,
            counts,
            summary: input.finish(),
            dedup,
        },
        at_once,
    ))
}

fn read(path: &Path, settings: &ZipInputSettings, batch_rows: Option<u64>) -> (Read, usize) {
    read_with(path, settings, &mut budget(None, batch_rows), None)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn gzip(content: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(content).unwrap();
    encoder.finish().unwrap()
}

#[derive(Clone, Copy, PartialEq)]
enum Pack {
    Stored,
    Deflated,
}

/// An archive of `(name, content, pack)`; a name ending in `/` is a directory.
fn archive(entries: &[(String, Vec<u8>, Pack)]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, content, pack) in entries {
        let options = zip::write::SimpleFileOptions::default().compression_method(match pack {
            Pack::Stored => zip::CompressionMethod::Stored,
            Pack::Deflated => zip::CompressionMethod::Deflated,
        });
        if name.ends_with('/') {
            writer.add_directory(name.as_str(), options).unwrap();
        } else {
            writer.start_file(name.as_str(), options).unwrap();
            writer.write_all(content).unwrap();
        }
    }
    writer.finish().unwrap().into_inner()
}

/// The marker inside the entry that is damaged after the archive is written.
const DAMAGE_MARK: &[u8] = b"damage-here";

/// An archive of many kinds of entries: logs deflated and stored, gzip entries, an
/// empty entry, a directory, a binary entry, a nested archive, many small entries,
/// texts that repeat within and across entries, blank lines, lines that are not UTF-8,
/// lines longer than the 200 bytes of [`settings`], CRLF, byte-order marks, and a
/// stored entry of more than 8 KiB whose checksum fails at its end. With `big`, two
/// entries of more than a megabyte each as well, read in many pieces.
fn varied(big: bool) -> Vec<u8> {
    let mut entries: Vec<(String, Vec<u8>, Pack)> = Vec::new();
    let mut access = String::new();
    for line in 0..300u32 {
        access.push_str(&format!("GET /page/{} 200\n", line % 37));
        if line % 50 == 7 {
            access.push('\n');
        }
    }
    entries.push(("a/".into(), Vec::new(), Pack::Stored));
    entries.push(("a/access.log".into(), access.into_bytes(), Pack::Deflated));
    entries.push(("a/empty.log".into(), Vec::new(), Pack::Deflated));
    // The lines a check would find are after the first 8 KiB, which must be text.
    let mut auth = b"\xef\xbb\xbfsshd: accepted for alice\r\n".to_vec();
    for line in 0..420u32 {
        auth.extend_from_slice(format!("sshd: failed for user{}\r\n", line % 13).as_bytes());
        if line >= 360 && line % 20 == 3 {
            auth.extend_from_slice(&[b'y'; 300]);
            auth.extend_from_slice(b"\r\n   \r\n\xff\xfe broken\r\n");
        }
    }
    auth.extend_from_slice("\u{feff}led by a mark\nlast line without end".as_bytes());
    entries.push(("b/auth.log".into(), auth, Pack::Stored));
    let mut rotated = String::new();
    for line in 0..250u32 {
        rotated.push_str(&format!(
            "GET /page/{} 200\nrotated {}\n",
            line % 41,
            line % 90
        ));
    }
    entries.push((
        "b/rotated.log.1.gz".into(),
        gzip(rotated.as_bytes()),
        Pack::Stored,
    ));
    let mut binary = b"\xd4\xc3\xb2\xa1".to_vec();
    binary.extend_from_slice(&[0; 64]);
    entries.push(("c/capture.pcap".into(), binary, Pack::Deflated));
    let nested = archive(&[("inner.log".into(), b"inside\n".to_vec(), Pack::Deflated)]);
    entries.push(("c/nested.zip".into(), nested, Pack::Stored));
    let mut damaged = String::new();
    for line in 0..600u32 {
        damaged.push_str(&format!(
            "damaged entry line {line} GET /page/{} 200\n",
            line % 7
        ));
    }
    damaged.push_str("damage-here and a little more\n");
    entries.push(("d/damaged.log".into(), damaged.into_bytes(), Pack::Stored));
    for small in 0..12u32 {
        let mut content = String::new();
        for line in 0..(small % 5 + 1) {
            content.push_str(&format!(
                "small {small} line {line}\nGET /page/{line} 200\n"
            ));
        }
        let pack = if small % 2 == 0 {
            Pack::Deflated
        } else {
            Pack::Stored
        };
        entries.push((format!("e/{small}.log"), content.into_bytes(), pack));
    }
    entries.push((
        "e/rotated.gz".into(),
        gzip(b"small 3 line 0\nzipped only here\n"),
        Pack::Deflated,
    ));
    if big {
        for (name, step) in [("f/big-one.log", 3u32), ("f/big-two.log", 7u32)] {
            let mut content = String::new();
            for line in 0..12_000u32 {
                content.push_str(&format!(
                    "{name} GET /search?q={}&page={} HTTP/1.1 200 {}\n",
                    (line * step) % 2_903,
                    line % 11,
                    "x".repeat((line % 60) as usize)
                ));
            }
            entries.push((name.into(), content.into_bytes(), Pack::Deflated));
        }
    }
    let mut after = String::new();
    for line in 0..80u32 {
        after.push_str(&format!("after {line}\nGET /page/{} 200\n", line % 37));
    }
    entries.push(("g/after.log".into(), after.into_bytes(), Pack::Deflated));

    let mut bytes = archive(&entries);
    // One byte of the stored entry's content is changed: it fails its checksum at its
    // end, after its lines were read.
    let at = bytes
        .windows(DAMAGE_MARK.len())
        .position(|window| window == DAMAGE_MARK)
        .unwrap();
    bytes[at] = b'D';
    bytes
}

/// Reads `path` one entry at a time and then at each of `at_once` entries at once,
/// all of which must be equal; returns the read one at a time.
fn same_as_one_at_a_time(path: &Path, batch_rows: Option<u64>, at_once: &[usize]) -> Read {
    let (one, counted) = read(path, &settings(Some(1)), batch_rows);
    assert_eq!(counted, 1);
    for &count in at_once {
        let (several, counted) = read(path, &settings(Some(count)), batch_rows);
        assert_eq!(
            counted, count,
            "{count} asked for, batches of {batch_rows:?}"
        );
        assert_eq!(
            several, one,
            "{count} entries at once, batches of {batch_rows:?}"
        );
    }
    one
}

// --- Equality with reading one entry at a time --------------------------------------------

#[test]
fn entries_read_at_once_give_the_batches_and_summary_of_one_at_a_time() {
    let scratch = Scratch::new("equal");
    let path = scratch.write("varied.zip", &varied(false));
    let reference = same_as_one_at_a_time(&path, None, &[2, 3, MAX_ENTRIES_AT_ONCE]);
    // 22 entries; 21 are opened (all but the directory), 18 are read to their end.
    assert_eq!(reference.summary.entries.len(), 22);
    assert_eq!(reference.summary.read().count(), 18);
    assert!(reference.summary.input.records > 1_000);
    // Six blank lines in one entry, three times a long, a blank and a broken line in
    // another.
    assert_eq!(reference.summary.input.skipped.len(), 15);
    // Batches of one row are slow to read: two numbers are enough there.
    let single = same_as_one_at_a_time(&path, Some(1), &[3, MAX_ENTRIES_AT_ONCE]);
    same_as_one_at_a_time(&path, Some(7), &[2, 5, MAX_ENTRIES_AT_ONCE]);
    // The rows of the default batch size are those of batches of one row.
    let rows = |read: &Read| {
        read.rows()
            .into_iter()
            .map(|(text, position, _)| (text.clone(), *position))
            .collect::<Vec<_>>()
    };
    assert_eq!(rows(&single), rows(&reference));
    assert_eq!(single.counts, reference.counts);
    assert_eq!(single.summary, reference.summary);
}

#[test]
fn entries_larger_than_a_piece_give_the_batches_of_one_at_a_time() {
    let scratch = Scratch::new("big");
    let path = scratch.write("big.zip", &varied(true));
    for (batch_rows, at_once) in [
        (Some(100), &[2, MAX_ENTRIES_AT_ONCE][..]),
        (None, &[2, 3, MAX_ENTRIES_AT_ONCE][..]),
    ] {
        let read = same_as_one_at_a_time(&path, batch_rows, at_once);
        let big = read
            .summary
            .entries
            .iter()
            .find(|entry| entry.path == "f/big-one.log")
            .unwrap();
        assert_eq!((big.records, &big.skipped), (12_000, &None));
    }
}

#[test]
fn a_damaged_entry_keeps_its_lines_when_entries_are_read_at_once() {
    let scratch = Scratch::new("damaged");
    let path = scratch.write("varied.zip", &varied(false));
    let read = same_as_one_at_a_time(&path, Some(7), &[4]);
    let damaged: Vec<_> = read.summary.damaged().collect();
    assert_eq!(damaged.len(), 1, "{:?}", read.summary.entries);
    assert_eq!(damaged[0].path, "d/damaged.log");
    assert_eq!(damaged[0].records, 601);
    assert_eq!(
        damaged[0].skipped,
        Some(EntrySkip::Damaged(
            "reading stopped after line 601: it fails its checksum".into()
        ))
    );
    // The lines before the damage and the entries after it are read.
    let texts: Vec<&str> = read
        .rows()
        .into_iter()
        .map(|(text, _, _)| text.as_str())
        .collect();
    assert!(texts.contains(&"damaged entry line 599 GET /page/4 200"));
    assert!(texts.contains(&"after 79"));
    let by_path = |path: &str| {
        read.summary
            .entries
            .iter()
            .find(|entry| entry.path == path)
            .unwrap()
            .skipped
            .clone()
    };
    assert_eq!(by_path("c/capture.pcap"), Some(EntrySkip::NotText));
    assert_eq!(by_path("c/nested.zip"), Some(EntrySkip::NestedArchive));
    assert_eq!(by_path("a/"), Some(EntrySkip::Directory));
    assert_eq!(by_path("b/rotated.log.1.gz"), None);
    assert_eq!(by_path("a/empty.log"), None);
}

#[test]
fn a_read_stopped_early_leaves_the_summary_of_one_at_a_time() {
    let scratch = Scratch::new("stopped");
    let path = scratch.write("big.zip", &varied(true));
    let stopped = |at_once: usize| {
        let mut budget = budget(None, Some(50));
        let mut input = ZipInput::open(&path, &settings(Some(at_once)), &budget).unwrap();
        let mut batches = input.batches(&mut budget);
        let first: Vec<Vec<(String, u64, u32)>> = (0..9)
            .map(|_| {
                let batch = batches.next().unwrap().unwrap();
                batch
                    .records()
                    .map(|record| (record.text, record.first_position, record.occurrences))
                    .collect()
            })
            .collect();
        // The batches are let go before the end of the archive: the readers stop.
        drop(batches);
        (first, input.finish())
    };
    let one = stopped(1);
    assert!(
        one.1
            .entries
            .iter()
            .any(|entry| entry.skipped == Some(EntrySkip::NotReached)),
        "the read stopped before the end"
    );
    for at_once in [2, MAX_ENTRIES_AT_ONCE] {
        assert_eq!(stopped(at_once), one, "{at_once} entries at once");
    }
}

// --- Deduplication on disk -----------------------------------------------------------------

#[test]
fn deduplication_on_disk_gives_the_rows_and_counts_of_one_at_a_time() {
    let scratch = Scratch::new("spill");
    let small = scratch.write("varied.zip", &varied(false));
    let big = scratch.write("big.zip", &varied(true));
    let spill = scratch.path("spill");
    for (path, set_texts, batch_rows) in [
        (&small, Some(0), Some(7)),
        (&small, Some(150), Some(7)),
        (&small, Some(150), None),
        (&big, Some(0), None),
        (&big, Some(2_000), None),
    ] {
        let (reference, _) = read(path, &settings(Some(1)), None);
        let what = format!("at most {set_texts:?} texts in memory, batches of {batch_rows:?}");
        let read_at = |at_once: usize| {
            read_with(
                path,
                &settings(Some(at_once)),
                &mut budget(None, batch_rows),
                Some((&spill, set_texts)),
            )
            .unwrap_or_else(|error| panic!("{what}, {at_once} at once: {error}"))
        };
        let (one, _) = read_at(1);
        assert!(one.dedup.is_some(), "{what}: on disk");
        assert!(!spill.exists(), "{what}: the files are removed");
        for at_once in [2, MAX_ENTRIES_AT_ONCE] {
            let (several, counted) = read_at(at_once);
            assert_eq!(counted, at_once, "{what}");
            assert_eq!(several, one, "{what}, {at_once} at once");
            assert!(!spill.exists(), "{what}: the files are removed");
        }
        // And the rows and counts of a read in memory alone.
        let rows = |read: &Read| {
            read.rows()
                .into_iter()
                .map(|(text, position, _)| (text.clone(), *position))
                .collect::<Vec<_>>()
        };
        assert_eq!(rows(&one), rows(&reference), "{what}");
        assert_eq!(one.counts, reference.counts, "{what}");
        assert_eq!(one.summary, reference.summary, "{what}");
    }
}

#[test]
fn a_set_that_outgrows_a_tight_limit_goes_to_disk_with_entries_read_at_once() {
    let scratch = Scratch::new("tight");
    // 250,000 distinct lines in each of four entries, each line in two of them: the
    // 500,000 keys need a table of 22 MiB, and 33 MiB to grow to it.
    let lines = 250_000u32;
    let entry = |from: u32, reverse: bool| {
        let mut content = String::new();
        for line in 0..lines {
            let line = if reverse { lines - 1 - line } else { line };
            content.push_str(&format!("line {}\n", from + line));
        }
        content.into_bytes()
    };
    let path = scratch.write(
        "many.zip",
        &archive(&[
            ("one.log".into(), entry(0, false), Pack::Deflated),
            ("two.log".into(), entry(lines, false), Pack::Deflated),
            ("three.log".into(), entry(0, true), Pack::Deflated),
            ("four.log".into(), entry(lines, true), Pack::Stored),
        ]),
    );
    let spill = scratch.path("spill");
    // What is held before the first record with two readers (some 38 MiB of readers,
    // the buffers of deduplication on disk, the fixed memory of a batch), as the
    // refusal under the smallest limit that reads at all states it.
    let mut tight = budget(Some(PROCESS_BASELINE_BYTES + 4 * MIB + 256 * 1024), None);
    let mut input = ZipInput::open(&path, &settings(Some(2)), &tight).unwrap();
    input.spill_into(&spill, None).unwrap();
    let message = input
        .batches(&mut tight)
        .next()
        .expect("an item")
        .expect_err("refused")
        .to_string();
    drop(input);
    let needs = message
        .split(" bytes) before its next record")
        .next()
        .filter(|before| before.len() < message.len())
        .and_then(|before| before.rsplit('(').next()?.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("{message}"));
    // 8 MiB beside that: the set goes to disk when entries are read at once (its growth
    // to 500,000 keys takes 33 MiB), and has room in memory one at a time, where the
    // memory of the second reader is free. The limit follows the readers' size.
    let limit = Some(PROCESS_BASELINE_BYTES + needs + 8 * MIB);
    let read_at = |at_once: usize| {
        let (read, counted) = read_with(
            &path,
            &settings(Some(at_once)),
            &mut budget(limit, None),
            Some((&spill, None)),
        )
        .unwrap_or_else(|error| panic!("{at_once} at once: {error}"));
        assert_eq!(counted, at_once);
        assert!(!spill.exists());
        read
    };
    let one = read_at(1);
    assert!(one.dedup.is_none(), "one at a time has room in memory");
    assert_eq!(one.summary.input.records, 4 * u64::from(lines));
    assert_eq!(one.summary.input.distinct_texts, 2 * u64::from(lines));
    for (index, ((text, position, _), count)) in one.rows().into_iter().zip(&one.counts).enumerate()
    {
        let index = index as u64;
        let (entry, line) = if index < u64::from(lines) {
            (0, index + 1)
        } else {
            (1, index - u64::from(lines) + 1)
        };
        assert_eq!(text, &format!("line {index}"));
        assert_eq!(*position, (entry << 40) | line);
        assert_eq!(*count, 2);
    }
    let several = read_at(2);
    let held = several.dedup.as_ref().expect("on disk").set_texts;
    assert!(held > 0 && held < 2 * u64::from(lines), "{held}");
    // The point of the freeze follows from what is held, so the batches may be cut
    // otherwise and a row's count in its batch may differ (a frame from disk counts
    // over the archive); the rows, the counts over the archive and the summary may not.
    let rows = |read: &Read| {
        read.rows()
            .into_iter()
            .map(|(text, position, _)| (text.clone(), *position))
            .collect::<Vec<_>>()
    };
    assert_eq!(rows(&several), rows(&one));
    assert_eq!(several.counts, one.counts);
    assert_eq!(several.summary, one.summary);
}

// --- What the readers take, and how many there are ----------------------------------------

#[test]
fn readers_that_do_not_fit_the_memory_limit_are_refused_before_a_record_is_read() {
    let scratch = Scratch::new("refused");
    let path = scratch.write("varied.zip", &varied(false));
    let limit = Some(PROCESS_BASELINE_BYTES + 10 * MIB);
    // One at a time the archive is read within the limit.
    let (one, _) = read_with(&path, &settings(Some(1)), &mut budget(limit, None), None)
        .expect("one entry at a time fits");
    // All of them at once do not fit; nothing is read.
    let mut short = budget(limit, None);
    let mut input = ZipInput::open(&path, &settings(Some(MAX_ENTRIES_AT_ONCE)), &short).unwrap();
    let first = input.batches(&mut short).next().expect("an item");
    let error = first.expect_err("refused");
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "the readers of {MAX_ENTRIES_AT_ONCE} entries read at once"
        )),
        "{message}"
    );
    assert!(message.contains("fewer entries at once"), "{message}");
    assert!(message.contains("stopped after 0 records"), "{message}");
    assert_eq!(input.entries_at_once(), MAX_ENTRIES_AT_ONCE);
    let summary = input.finish();
    assert_eq!(summary.input.records, 0);
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.skipped == Some(EntrySkip::NotReached) && entry.records == 0)
    );
    // The default takes one at a time under this limit and reads it.
    let (default, counted) = read_with(&path, &settings(None), &mut budget(limit, None), None)
        .expect("the default fits");
    assert_eq!(counted, 1);
    assert_eq!(default, one);
}

#[test]
fn the_entries_read_at_once_are_bounded_and_stated() {
    let scratch = Scratch::new("bounded");
    let path = scratch.write("varied.zip", &varied(false));
    for wrong in [0, MAX_ENTRIES_AT_ONCE + 1] {
        let error = ZipInput::open(&path, &settings(Some(wrong)), &budget(None, None))
            .expect_err("refused");
        assert!(matches!(error, L2dError::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains(&format!("1 to {MAX_ENTRIES_AT_ONCE}, not {wrong}")),
            "{error}"
        );
    }
    // Before the batches begin nothing is decided.
    let input = ZipInput::open(&path, &settings(Some(3)), &budget(None, None)).unwrap();
    assert_eq!(input.entries_at_once(), 1);
    drop(input);
    // The default with room for the readers, and what the analyst asks for.
    assert_eq!(
        read(&path, &settings(None), None).1,
        DEFAULT_ENTRIES_AT_ONCE
    );
    assert_eq!(read(&path, &settings(Some(3)), None).1, 3);
    // Never more than the entries that will be opened: one entry is read alone, and
    // so is an archive whose patterns select one entry.
    let single = scratch.write(
        "single.zip",
        &archive(&[("only.log".into(), b"one\ntwo\n".to_vec(), Pack::Deflated)]),
    );
    assert_eq!(read(&single, &settings(Some(4)), None).1, 1);
    let one_pattern = ZipInputSettings {
        entries: vec!["g/*".into()],
        ..settings(Some(4))
    };
    let (picked, counted) = read(&path, &one_pattern, None);
    assert_eq!(counted, 1);
    assert_eq!(picked.summary.read().count(), 1);
    let two_patterns = ZipInputSettings {
        entries: vec!["g/*".into(), "a/*.log".into()],
        ..settings(Some(MAX_ENTRIES_AT_ONCE))
    };
    assert_eq!(read(&path, &two_patterns, None).1, 3);
}
