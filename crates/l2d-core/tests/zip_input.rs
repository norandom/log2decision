//! Zip archives as input (requirement 13; task 1.16): entries read without unpacking,
//! in batches, with the same records, counts and summary as the unpacked files.
//!
//! The archives are built in a scratch directory: the ordinary ones by the `zip` crate
//! (a dev-dependency; the tool reads archives with its own code, so the crate is a
//! second opinion on the format), the hostile ones byte by byte.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use l2d_core::input::content_sha256;
use l2d_core::zip_input::{DEFAULT_MAX_RECORD_BYTES, MAX_RECORD_BYTES};
use l2d_core::{
    ArchivePosition, ArchiveSummary, EntrySkip, InputAdapter, L2dError, Label, LinesAdapter,
    LogRecord, MemoryBudget, MemorySettings, ZipInput, ZipInputSettings,
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
        let dir = std::env::temp_dir().join(format!("l2d-core-zip-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// The names of what is in the directory, sorted.
    fn listing(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
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

fn patterns(entries: &[&str]) -> ZipInputSettings {
    ZipInputSettings {
        entries: entries.iter().map(ToString::to_string).collect(),
        ..ZipInputSettings::default()
    }
}

/// What reading an archive to its end gave.
#[derive(Debug)]
struct ArchiveRead {
    /// The rows of all batches in the order they came, with the occurrence counts of
    /// the whole archive.
    rows: Vec<LogRecord>,
    summary: ArchiveSummary,
    /// The largest number of rows one batch held.
    largest_batch: usize,
}

impl ArchiveRead {
    fn texts(&self) -> Vec<&str> {
        self.rows.iter().map(|row| row.text.as_str()).collect()
    }

    fn entry(&self, path: &str) -> &l2d_core::EntryReport {
        self.summary
            .entries
            .iter()
            .find(|entry| entry.path == path)
            .unwrap_or_else(|| panic!("no entry {path} in {:?}", self.summary.entries))
    }

    /// Entry path, line and reason of every skipped record.
    fn skipped(&self) -> Vec<(String, u64, String)> {
        self.summary
            .input
            .skipped
            .iter()
            .map(|record| {
                let (path, line) = self.summary.locate(record.position).unwrap();
                (path.to_string(), line, record.reason.clone())
            })
            .collect()
    }
}

fn read_with(
    path: &Path,
    settings: &ZipInputSettings,
    budget: &mut MemoryBudget,
) -> Result<ArchiveRead, L2dError> {
    let mut input = ZipInput::open(path, settings, budget)?;
    let mut rows = Vec::new();
    let mut largest_batch = 0;
    for batch in input.batches(budget) {
        let batch = batch?;
        assert!(
            !batch.is_empty(),
            "a batch without new texts is not handed out"
        );
        largest_batch = largest_batch.max(batch.len());
        rows.extend(batch.records());
    }
    for row in &mut rows {
        row.occurrences = u32::try_from(input.occurrences(&row.text)).unwrap();
    }
    assert!(input.batch_plan().is_some(), "the plan of a completed read");
    Ok(ArchiveRead {
        rows,
        summary: input.finish(),
        largest_batch,
    })
}

fn read(path: &Path, settings: &ZipInputSettings) -> ArchiveRead {
    read_with(path, settings, &mut budget(None, None))
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn open_error(path: &Path, settings: &ZipInputSettings) -> L2dError {
    let budget = budget(None, None);
    match ZipInput::open(path, settings, &budget) {
        Ok(_) => panic!("{} was opened", path.display()),
        Err(error) => error,
    }
}

// --- Archives written by the `zip` crate ---------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Pack {
    Stored,
    Deflated,
    /// Deflated with the zip64 fields written for the entry.
    Large,
}

fn crate_zip(entries: &[(&str, &[u8], Pack)]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, content, pack) in entries {
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(match pack {
                Pack::Stored => zip::CompressionMethod::Stored,
                Pack::Deflated | Pack::Large => zip::CompressionMethod::Deflated,
            })
            .large_file(*pack == Pack::Large);
        writer.start_file(*name, options).unwrap();
        writer.write_all(content).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn gzip(content: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(content).unwrap();
    encoder.finish().unwrap()
}

// --- Archives written byte by byte ---------------------------------------------------------

/// One entry of a hand-made archive; every field is written as given, true or not.
#[derive(Clone)]
struct RawEntry {
    name: Vec<u8>,
    method: u16,
    flags: u16,
    /// The bytes in the archive.
    data: Vec<u8>,
    crc: u32,
    /// The size the directory declares for the content.
    size: u32,
    made_by: u16,
    external: u32,
    /// Sizes and offset are written to the directory as zip64 fields.
    zip64: bool,
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = flate2::Crc::new();
    crc.update(bytes);
    crc.sum()
}

impl RawEntry {
    fn stored(name: &str, content: &[u8]) -> Self {
        Self {
            name: name.as_bytes().to_vec(),
            method: 0,
            flags: 1 << 11,
            data: content.to_vec(),
            crc: crc32(content),
            size: content.len() as u32,
            made_by: 20,
            external: 0,
            zip64: false,
        }
    }

    fn deflated(name: &str, content: &[u8]) -> Self {
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(content).unwrap();
        Self {
            method: 8,
            data: encoder.finish().unwrap(),
            ..Self::stored(name, content)
        }
    }
}

/// A hand-made archive and where its parts are.
struct RawZip {
    bytes: Vec<u8>,
    /// Where the data of each entry starts.
    data_at: Vec<usize>,
    /// Where the central header of each entry starts.
    central_at: Vec<usize>,
    /// Where the end record starts.
    end_at: usize,
}

fn raw_zip(entries: &[RawEntry]) -> RawZip {
    let mut bytes = Vec::new();
    let mut local_at = Vec::new();
    let mut data_at = Vec::new();
    for entry in entries {
        local_at.push(bytes.len());
        bytes.extend_from_slice(b"PK\x03\x04");
        bytes.extend_from_slice(&20u16.to_le_bytes());
        bytes.extend_from_slice(&entry.flags.to_le_bytes());
        bytes.extend_from_slice(&entry.method.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&entry.crc.to_le_bytes());
        bytes.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&entry.size.to_le_bytes());
        bytes.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&entry.name);
        data_at.push(bytes.len());
        bytes.extend_from_slice(&entry.data);
    }
    let directory_at = bytes.len();
    let mut central_at = Vec::new();
    for (entry, local) in entries.iter().zip(&local_at) {
        central_at.push(bytes.len());
        bytes.extend_from_slice(b"PK\x01\x02");
        bytes.extend_from_slice(&entry.made_by.to_le_bytes());
        bytes.extend_from_slice(&20u16.to_le_bytes());
        bytes.extend_from_slice(&entry.flags.to_le_bytes());
        bytes.extend_from_slice(&entry.method.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&entry.crc.to_le_bytes());
        if entry.zip64 {
            bytes.extend_from_slice(&[0xff; 8]);
        } else {
            bytes.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&entry.size.to_le_bytes());
        }
        bytes.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&(if entry.zip64 { 28u16 } else { 0 }).to_le_bytes());
        bytes.extend_from_slice(&[0; 6]);
        bytes.extend_from_slice(&entry.external.to_le_bytes());
        if entry.zip64 {
            bytes.extend_from_slice(&[0xff; 4]);
        } else {
            bytes.extend_from_slice(&(*local as u32).to_le_bytes());
        }
        bytes.extend_from_slice(&entry.name);
        if entry.zip64 {
            bytes.extend_from_slice(&1u16.to_le_bytes());
            bytes.extend_from_slice(&24u16.to_le_bytes());
            bytes.extend_from_slice(&u64::from(entry.size).to_le_bytes());
            bytes.extend_from_slice(&(entry.data.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&(*local as u64).to_le_bytes());
        }
    }
    let end_at = bytes.len();
    bytes.extend_from_slice(b"PK\x05\x06");
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&((end_at - directory_at) as u32).to_le_bytes());
    bytes.extend_from_slice(&(directory_at as u32).to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    RawZip {
        bytes,
        data_at,
        central_at,
        end_at,
    }
}

fn put_u16(bytes: &mut [u8], at: usize, value: u16) {
    bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

/// The last place `needle` occurs at.
fn last_of(bytes: &[u8], needle: &[u8]) -> usize {
    bytes
        .windows(needle.len())
        .rposition(|window| window == needle)
        .expect("the bytes are there")
}

// --- The unpacked files as the reference ---------------------------------------------------

/// What the lines adapter reads from the files one after another, as one input: every
/// distinct text with the file and line it was first seen at and its occurrences over
/// all files; the records; the skipped records by file and line.
struct Unpacked {
    rows: Vec<(String, usize, u64, u32)>,
    records: u64,
    skipped: Vec<(usize, u64, String)>,
}

fn unpacked(scratch: &Scratch, files: &[(&str, &[u8])]) -> Unpacked {
    let mut rows: Vec<(String, usize, u64, u32)> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut records = 0;
    let mut skipped = Vec::new();
    for (index, (name, content)) in files.iter().enumerate() {
        let path = scratch.write(
            &format!("unpacked-{index}-{}", name.replace('/', "_")),
            content,
        );
        let (logs, summary) = LinesAdapter.read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        records += summary.records;
        for record in summary.skipped {
            skipped.push((index, record.position, record.reason));
        }
        for record in logs.records() {
            match seen.get(&record.text) {
                Some(row) => rows[*row].3 += record.occurrences,
                None => {
                    seen.insert(record.text.clone(), rows.len());
                    rows.push((
                        record.text,
                        index,
                        record.first_position,
                        record.occurrences,
                    ));
                }
            }
        }
    }
    Unpacked {
        rows,
        records,
        skipped,
    }
}

const ACCESS: &[u8] = b"GET /index.html 200\nGET /a 200\n\nGET /index.html 200\nPOST /login 302\n";
const AUTH: &[u8] =
    b"\xef\xbb\xbfsshd: accepted for alice\r\nsshd: failed for bob\r\n   \r\nGET /a 200\r\nlast line without end";
/// Text for more than the first 8 KiB (the part the text check looks at), then a line
/// that is not valid UTF-8, then lines the other files hold too.
fn mixed() -> Vec<u8> {
    let mut content = b"caf\xc3\xa9 opened\n".to_vec();
    content.extend_from_slice("GET /a 200\n".repeat(1_000).as_bytes());
    content.extend_from_slice(b"broken \xff\xfe line\nGET /index.html 200\nsshd: failed for bob\n");
    content
}

/// U+FEFF in every place: twice at the start of the file (the first is the byte-order
/// mark, the second is text), at the start of later lines, and inside a line.
const MARKS: &str = "\u{feff}\u{feff}first\nplain\n\u{feff}led\n\u{feff}led\nmid\u{feff}dle\n";

// --- Equality with the unpacked files ------------------------------------------------------

#[test]
fn an_archive_reads_like_its_unpacked_files_at_any_batch_size() {
    let scratch = Scratch::new("equal");
    let mixed = mixed();
    let files: [(&str, &[u8]); 4] = [
        ("logs/access.log", ACCESS),
        ("logs/auth.log", AUTH),
        ("mixed.txt", &mixed),
        ("logs/marks.log", MARKS.as_bytes()),
    ];
    let reference = unpacked(&scratch, &files);
    assert_eq!(reference.records, 1_019);
    assert_eq!(reference.rows.len(), 11);
    assert_eq!(reference.rows[7].0, "\u{feff}first");
    assert_eq!(reference.skipped.len(), 3);

    let archive = scratch.write(
        "equal.zip",
        &crate_zip(&[
            (files[0].0, files[0].1, Pack::Deflated),
            (files[1].0, files[1].1, Pack::Stored),
            (files[2].0, files[2].1, Pack::Deflated),
            (files[3].0, files[3].1, Pack::Stored),
        ]),
    );
    let mut first: Option<ArchiveRead> = None;
    for batch_rows in [Some(1), Some(2), Some(3), Some(5), Some(64), None] {
        let got = read_with(
            &archive,
            &ZipInputSettings::default(),
            &mut budget(None, batch_rows),
        )
        .unwrap();
        let rows: Vec<(String, usize, u64, u32)> = got
            .rows
            .iter()
            .map(|row| {
                let position = ArchivePosition::unpack(row.first_position);
                assert_eq!(row.label, Label::Unknown);
                assert_eq!(row.group, None);
                (
                    row.text.clone(),
                    position.entry as usize,
                    position.line,
                    row.occurrences,
                )
            })
            .collect();
        assert_eq!(rows, reference.rows, "batch size {batch_rows:?}");
        assert_eq!(got.summary.input.records, reference.records);
        assert_eq!(
            got.summary.input.distinct_texts,
            reference.rows.len() as u64
        );
        assert_eq!(got.summary.input.label_conflicts, 0);
        let skipped: Vec<(usize, u64, String)> = got
            .summary
            .input
            .skipped
            .iter()
            .map(|record| {
                let position = ArchivePosition::unpack(record.position);
                (
                    position.entry as usize,
                    position.line,
                    record.reason.clone(),
                )
            })
            .collect();
        assert_eq!(skipped, reference.skipped, "batch size {batch_rows:?}");
        if let Some(rows) = batch_rows {
            assert!(
                got.largest_batch as u64 <= rows,
                "batch size {batch_rows:?}"
            );
        }
        // Entry reports: every entry read, with its records.
        let entries: Vec<(&str, u64, &Option<EntrySkip>)> = got
            .summary
            .entries
            .iter()
            .map(|entry| (entry.path.as_str(), entry.records, &entry.skipped))
            .collect();
        assert_eq!(
            entries,
            [
                ("logs/access.log", 5, &None),
                ("logs/auth.log", 5, &None),
                ("mixed.txt", 1_004, &None),
                ("logs/marks.log", 5, &None),
            ]
        );
        // The summary is the same whatever the batch size.
        if let Some(first) = &first {
            assert_eq!(got.summary, first.summary, "batch size {batch_rows:?}");
        } else {
            first = Some(got);
        }
    }
}

#[test]
fn a_single_entry_archive_gives_the_frame_and_summary_of_the_unpacked_file() {
    let scratch = Scratch::new("single");
    let mut content = Vec::new();
    for index in 0..5_000u32 {
        content.extend_from_slice(format!("GET /item/{} HTTP/1.1\n", index % 700).as_bytes());
        if index % 911 == 0 {
            content.extend_from_slice(b"\n");
        }
    }
    let unpacked_path = scratch.write("single.log", &content);
    let (logs, summary) = LinesAdapter.read(&unpacked_path).unwrap();
    let archive = scratch.write(
        "single.zip",
        &crate_zip(&[("single.log", &content, Pack::Deflated)]),
    );
    for batch_rows in [Some(100), Some(1_024), None] {
        let got = read_with(
            &archive,
            &ZipInputSettings::default(),
            &mut budget(None, batch_rows),
        )
        .unwrap();
        // The first entry has index 0, so its positions are the line numbers.
        assert_eq!(got.rows, logs.to_records(), "batch size {batch_rows:?}");
        assert_eq!(got.summary.input.records, summary.records);
        assert_eq!(got.summary.input.distinct_texts, summary.distinct_texts);
        assert_eq!(got.summary.input.skipped, summary.skipped);
        assert_eq!(got.summary.input.label_conflicts, summary.label_conflicts);
        assert_eq!(got.summary.input.source_path, archive);
    }
}

#[test]
fn positions_name_the_entry_and_the_line() {
    let scratch = Scratch::new("positions");
    let archive = scratch.write(
        "positions.zip",
        &crate_zip(&[
            ("a.log", b"one\ntwo\n", Pack::Stored),
            ("deep/b.log", b"two\n\nthree\n", Pack::Deflated),
        ]),
    );
    let got = read(&archive, &ZipInputSettings::default());
    let located: Vec<(&str, &str, u64)> = got
        .rows
        .iter()
        .map(|row| {
            let (path, line) = got.summary.locate(row.first_position).unwrap();
            (row.text.as_str(), path, line)
        })
        .collect();
    assert_eq!(
        located,
        [
            ("one", "a.log", 1),
            ("two", "a.log", 2),
            ("three", "deep/b.log", 3)
        ]
    );
    assert_eq!(
        got.skipped(),
        [("deep/b.log".to_string(), 2, "empty line".to_string())]
    );
    assert_eq!(got.rows[1].occurrences, 2);

    let position = ArchivePosition {
        entry: 7,
        line: 123_456_789_012,
    };
    assert_eq!(ArchivePosition::unpack(position.pack()), position);
    assert_eq!(got.summary.locate(position.pack()), None, "no entry 7");
}

#[test]
fn the_content_hash_is_that_of_the_archive_file() {
    let scratch = Scratch::new("hash");
    let bytes = crate_zip(&[("a.log", b"one\n", Pack::Deflated)]);
    let archive = scratch.write("hash.zip", &bytes);
    let got = read(&archive, &patterns(&["nothing-matches"]));
    assert_eq!(got.summary.input.content_sha256, content_sha256(&bytes));
    assert_eq!(got.summary.input.records, 0);
}

#[test]
fn the_reader_agrees_with_the_zip_crate_on_archives_it_did_not_write() {
    // The hand-made archive is read by the crate, so the builder of these tests writes
    // the format as others read it.
    let raw = raw_zip(&[
        RawEntry::stored("a.log", b"one\ntwo\n"),
        RawEntry::deflated("b.log", b"three\nthree\nfour\n"),
    ]);
    let mut reference = zip::ZipArchive::new(std::io::Cursor::new(raw.bytes.clone())).unwrap();
    let mut lines = Vec::new();
    for index in 0..reference.len() {
        let mut content = String::new();
        std::io::Read::read_to_string(&mut reference.by_index(index).unwrap(), &mut content)
            .unwrap();
        lines.extend(content.lines().map(str::to_string));
    }
    let scratch = Scratch::new("agree");
    let got = read(
        &scratch.write("agree.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    assert_eq!(lines, ["one", "two", "three", "three", "four"]);
    assert_eq!(got.texts(), ["one", "two", "three", "four"]);
    assert_eq!(got.summary.input.records, lines.len() as u64);
}

// --- U+FEFF --------------------------------------------------------------------------------

#[test]
fn a_line_that_starts_with_a_byte_order_mark_is_text_at_any_batch_size() {
    let scratch = Scratch::new("marks");
    let archive = scratch.write(
        "marks.zip",
        &crate_zip(&[
            ("a.log", "a\n\u{feff}b\nc\n".as_bytes(), Pack::Stored),
            ("twice.log", MARKS.as_bytes(), Pack::Deflated),
            (
                "gz.log",
                &gzip("\u{feff}once\n\u{feff}b\n".as_bytes()),
                Pack::Stored,
            ),
        ]),
    );
    let mut first: Option<ArchiveRead> = None;
    for batch_rows in [Some(1), Some(2), Some(3), None] {
        let got = read_with(
            &archive,
            &ZipInputSettings::default(),
            &mut budget(None, batch_rows),
        )
        .unwrap_or_else(|error| panic!("batch size {batch_rows:?}: {error}"));
        assert_eq!(
            got.texts(),
            [
                "a",
                "\u{feff}b",
                "c",
                "\u{feff}first",
                "plain",
                "\u{feff}led",
                "mid\u{feff}dle",
                "once"
            ],
            "batch size {batch_rows:?}"
        );
        assert_eq!(got.rows[1].occurrences, 2);
        assert_eq!(got.rows[5].occurrences, 2);
        assert!(got.summary.input.skipped.is_empty());
        assert!(
            got.summary
                .entries
                .iter()
                .all(|entry| entry.skipped.is_none())
        );
        if let Some(first) = &first {
            assert_eq!(got.rows, first.rows, "batch size {batch_rows:?}");
            assert_eq!(got.summary, first.summary, "batch size {batch_rows:?}");
        } else {
            first = Some(got);
        }
    }
}

#[test]
fn a_line_with_a_byte_order_mark_after_a_full_default_batch_is_read() {
    let scratch = Scratch::new("marks-default");
    let mut content = String::new();
    for index in 0..65_536 {
        content.push_str(&format!("line {}\n", index % 100));
    }
    content.push_str("\u{feff}after the batch\nlast\n");
    let archive = scratch.write(
        "full.zip",
        &crate_zip(&[("full.log", content.as_bytes(), Pack::Deflated)]),
    );
    let got = read(&archive, &ZipInputSettings::default());
    assert_eq!(got.rows.len(), 102);
    assert_eq!(got.texts()[100..], ["\u{feff}after the batch", "last"]);
    assert_eq!(got.summary.input.records, 65_538);
}

// --- Entry patterns ------------------------------------------------------------------------

#[test]
fn entry_patterns_select_what_is_read() {
    let scratch = Scratch::new("patterns");
    let archive = scratch.write(
        "patterns.zip",
        &crate_zip(&[
            ("logs/apache/access.log", b"a1\n", Pack::Deflated),
            ("logs/auth.log", b"b1\n", Pack::Deflated),
            ("config/app.yml", b"key: value\n", Pack::Deflated),
            ("readme.txt", b"read me\n", Pack::Stored),
        ]),
    );
    let got = read(&archive, &patterns(&["*.log"]));
    assert_eq!(got.texts(), ["a1", "b1"]);
    assert_eq!(
        got.entry("config/app.yml").skipped,
        Some(EntrySkip::NoPatternMatch)
    );
    assert_eq!(
        got.entry("readme.txt").skipped,
        Some(EntrySkip::NoPatternMatch)
    );
    assert_eq!(got.entry("logs/auth.log").skipped, None);
    assert_eq!(got.summary.input.records, 2);

    let got = read(&archive, &patterns(&["logs/apache/*", "readme.???"]));
    assert_eq!(got.texts(), ["a1", "read me"]);

    let got = read(&archive, &patterns(&["LOGS/*"]));
    assert!(got.rows.is_empty(), "patterns are case-sensitive");

    let got = read(&archive, &ZipInputSettings::default());
    assert_eq!(got.texts(), ["a1", "b1", "key: value", "read me"]);

    match open_error(&archive, &patterns(&["logs/[a"])) {
        L2dError::InvalidInput { detail } => assert!(detail.contains("logs/[a"), "{detail}"),
        other => panic!("{other:?}"),
    }
}

// --- Entries that are not read -------------------------------------------------------------

#[test]
fn a_binary_entry_is_skipped_and_reported() {
    let scratch = Scratch::new("binary");
    let mut capture = b"\xd4\xc3\xb2\xa1\x02\x00\x04\x00".to_vec();
    capture.extend_from_slice(&[0u8; 200]);
    let latin1 = b"caf\xe9 without utf-8\n";
    // Text for more than the first 8 KiB, then bytes that are no text: the entry is
    // read, and the lines that are no text are skipped one by one.
    let mut late = "plain line\n".repeat(1_000).into_bytes();
    late.extend_from_slice(b"bad \xff line\nlast\n");
    let archive = scratch.write(
        "binary.zip",
        &crate_zip(&[
            ("before.log", b"before\n", Pack::Deflated),
            ("capture.pcap", &capture, Pack::Deflated),
            ("latin1.log", latin1, Pack::Stored),
            ("late.log", &late, Pack::Deflated),
            ("after.log", b"after\n", Pack::Deflated),
        ]),
    );
    let got = read(&archive, &ZipInputSettings::default());
    assert_eq!(got.texts(), ["before", "plain line", "last", "after"]);
    assert_eq!(got.entry("capture.pcap").skipped, Some(EntrySkip::NotText));
    assert_eq!(got.entry("capture.pcap").records, 0);
    assert_eq!(got.entry("latin1.log").skipped, Some(EntrySkip::NotText));
    assert_eq!(got.entry("late.log").skipped, None);
    assert_eq!(got.entry("late.log").records, 1_002);
    assert_eq!(
        got.skipped(),
        [(
            "late.log".to_string(),
            1_001,
            "line is not valid UTF-8".to_string()
        )]
    );
}

#[test]
fn an_encrypted_entry_is_skipped_and_reported() {
    let scratch = Scratch::new("encrypted");
    let mut secret = RawEntry::stored("secret.log", b"not really encrypted\n");
    secret.flags |= 1;
    let raw = raw_zip(&[
        RawEntry::stored("a.log", b"one\n"),
        secret,
        RawEntry::deflated("b.log", b"two\n"),
    ]);
    let got = read(
        &scratch.write("encrypted.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    assert_eq!(got.texts(), ["one", "two"]);
    assert_eq!(got.entry("secret.log").skipped, Some(EntrySkip::Encrypted));
    assert_eq!(got.summary.input.records, 2);
}

#[test]
fn an_entry_with_an_unsupported_method_is_skipped_and_reported() {
    let scratch = Scratch::new("method");
    let mut bzip2 = RawEntry::stored("b.log", b"whatever\n");
    bzip2.method = 12;
    let raw = raw_zip(&[RawEntry::stored("a.log", b"one\n"), bzip2]);
    let got = read(
        &scratch.write("method.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    assert_eq!(got.texts(), ["one"]);
    assert_eq!(
        got.entry("b.log").skipped,
        Some(EntrySkip::UnsupportedMethod(12))
    );
}

#[test]
fn directories_and_links_are_not_read() {
    let scratch = Scratch::new("kinds");
    let mut link = RawEntry::stored("current.log", b"logs/a.log");
    link.made_by = (3 << 8) | 20;
    link.external = 0o120777 << 16;
    let raw = raw_zip(&[
        RawEntry::stored("logs/", b""),
        RawEntry::stored("logs/a.log", b"one\n"),
        link,
    ]);
    let got = read(
        &scratch.write("kinds.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    assert_eq!(got.texts(), ["one"]);
    assert_eq!(got.entry("logs/").skipped, Some(EntrySkip::Directory));
    assert_eq!(got.entry("current.log").skipped, Some(EntrySkip::Link));
    assert_eq!(got.summary.entries.len(), 3);
}

// --- Damaged entries -----------------------------------------------------------------------

fn damaged_reason(read: &ArchiveRead, path: &str) -> String {
    match &read.entry(path).skipped {
        Some(EntrySkip::Damaged(reason)) => reason.clone(),
        other => panic!("{path}: {other:?}"),
    }
}

#[test]
fn a_damaged_entry_is_reported_and_the_other_entries_are_kept() {
    let scratch = Scratch::new("damaged");
    let content = "line that repeats\n".repeat(400);
    let entries = [
        RawEntry::stored("a.log", b"first\n"),
        RawEntry::stored("stored.log", b"stored one\nstored two\n"),
        RawEntry::deflated("deflated.log", content.as_bytes()),
        RawEntry::stored("big.log", "big line\n".repeat(10_000).as_bytes()),
        RawEntry::stored("z.log", b"last\n"),
    ];

    // A small stored entry with a changed byte: it is whole in the first 8 KiB, so its
    // checksum fails before a record of it is read.
    let mut raw = raw_zip(&entries);
    raw.bytes[raw.data_at[1] + 3] ^= 0x20;
    let got = read(
        &scratch.write("stored.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    assert!(damaged_reason(&got, "stored.log").contains("checksum"));
    assert_eq!(got.entry("stored.log").records, 0);
    assert_eq!(got.entry("a.log").skipped, None);
    assert_eq!(got.entry("z.log").skipped, None);
    assert_eq!(
        got.texts(),
        ["first", "line that repeats", "big line", "last"]
    );
    assert_eq!(got.summary.damaged().count(), 1);
    assert_eq!(got.summary.read().count(), 4);

    // A large stored entry with a changed byte fails its checksum at its end: the
    // records read from it until then are kept and counted.
    let mut raw = raw_zip(&entries);
    raw.bytes[raw.data_at[3] + 2] ^= 0x20;
    let got = read(
        &scratch.write("big.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    let reason = damaged_reason(&got, "big.log");
    assert!(
        reason.contains("checksum") && reason.contains("10000"),
        "{reason}"
    );
    assert_eq!(
        got.entry("big.log").records,
        10_000,
        "what was read is counted"
    );
    assert_eq!(
        got.texts(),
        [
            "first",
            "stored one",
            "stored two",
            "line that repeats",
            "biG line",
            "big line",
            "last"
        ]
    );
    assert_eq!(got.summary.input.records, 10_404);

    // A deflated entry whose stream is broken.
    let mut raw = raw_zip(&entries);
    for byte in &mut raw.bytes[raw.data_at[2]..raw.data_at[2] + 4] {
        *byte = 0xff;
    }
    let got = read(
        &scratch.write("deflated.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    damaged_reason(&got, "deflated.log");
    assert_eq!(
        got.texts(),
        ["first", "stored one", "stored two", "big line", "last"]
    );

    // An entry whose local header is not where the directory says.
    let mut raw = raw_zip(&entries);
    put_u32(&mut raw.bytes, raw.central_at[1] + 42, 3);
    let got = read(
        &scratch.write("offset.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    damaged_reason(&got, "stored.log");
    assert_eq!(
        got.texts(),
        ["first", "line that repeats", "big line", "last"]
    );

    // An entry whose local header is said to lie beyond the archive.
    let mut raw = raw_zip(&entries);
    put_u32(&mut raw.bytes, raw.central_at[1] + 42, 0x7fff_ffff);
    let got = read(
        &scratch.write("beyond.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    damaged_reason(&got, "stored.log");
    assert_eq!(
        got.texts(),
        ["first", "line that repeats", "big line", "last"]
    );
    assert_eq!(got.summary.input.records, 10_402);
}

#[test]
fn an_entry_that_holds_more_or_less_than_it_declares_is_damaged() {
    let scratch = Scratch::new("declared");
    let content = "0123456789\n".repeat(2_000);

    // The directory declares 100 bytes; the stream inflates to 22,000.
    let mut more = RawEntry::deflated("more.log", content.as_bytes());
    more.size = 100;
    let raw = raw_zip(&[more, RawEntry::stored("z.log", b"last\n")]);
    let got = read(
        &scratch.write("more.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    let reason = damaged_reason(&got, "more.log");
    assert!(reason.contains("100"), "{reason}");
    assert!(
        got.entry("more.log").records <= 10,
        "reading stops where the declared size ends: {}",
        got.entry("more.log").records
    );
    assert!(got.texts().contains(&"last"));

    // The directory declares more than there is.
    let mut less = RawEntry::deflated("less.log", content.as_bytes());
    less.size = 1_000_000;
    let raw = raw_zip(&[less, RawEntry::stored("z.log", b"last\n")]);
    let got = read(
        &scratch.write("less.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    let reason = damaged_reason(&got, "less.log");
    assert!(
        reason.contains("1000000") && reason.contains("22000"),
        "{reason}"
    );
    assert!(got.texts().contains(&"last"));

    // The data of an entry runs beyond the end of the archive.
    let mut raw = raw_zip(&[
        RawEntry::stored("a.log", b"first\n"),
        RawEntry::stored("long.log", b"abc\n"),
    ]);
    put_u32(&mut raw.bytes, raw.central_at[1] + 20, 5_000_000);
    put_u32(&mut raw.bytes, raw.central_at[1] + 24, 5_000_000);
    let got = read(
        &scratch.write("beyond.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    damaged_reason(&got, "long.log");
    assert_eq!(got.texts(), ["first"]);
}

// --- Gzip entries and other nested archives ------------------------------------------------

#[test]
fn a_gzip_entry_is_read_through() {
    let scratch = Scratch::new("gzip");
    let rotated = gzip(b"old one\nold two\n\nold one\n");
    // Two members one after the other, as appending to a .gz file leaves them.
    let mut members = gzip(b"\xef\xbb\xbfpart one\n");
    members.extend_from_slice(&gzip(b"part two\n"));
    let archive = scratch.write(
        "gzip.zip",
        &crate_zip(&[
            ("logs/access.log", b"new one\n", Pack::Deflated),
            // Named without `.gz`: the content decides.
            ("logs/access.log.1", &rotated, Pack::Stored),
            ("logs/members.gz", &members, Pack::Deflated),
        ]),
    );
    let got = read(&archive, &ZipInputSettings::default());
    assert_eq!(
        got.texts(),
        ["new one", "old one", "old two", "part one", "part two"]
    );
    assert_eq!(got.entry("logs/access.log.1").records, 4);
    assert_eq!(got.entry("logs/access.log.1").skipped, None);
    assert_eq!(got.rows[1].occurrences, 2);
    assert_eq!(
        got.skipped(),
        [("logs/access.log.1".to_string(), 3, "empty line".to_string())]
    );
    assert_eq!(
        got.summary.locate(got.rows[2].first_position),
        Some(("logs/access.log.1", 2))
    );
}

#[test]
fn a_gzip_entry_that_is_cut_off_or_holds_no_text_is_reported() {
    let scratch = Scratch::new("gzip-bad");
    let whole = gzip("a line of the rotated log\n".repeat(500).as_bytes());
    let cut = &whole[..whole.len() - 12];
    let binary = gzip(&[0u8; 300]);
    let archive = scratch.write(
        "gzip-bad.zip",
        &crate_zip(&[
            ("cut.gz", cut, Pack::Stored),
            ("binary.gz", &binary, Pack::Stored),
            ("z.log", b"last\n", Pack::Stored),
        ]),
    );
    let got = read(&archive, &ZipInputSettings::default());
    damaged_reason(&got, "cut.gz");
    assert_eq!(got.entry("binary.gz").skipped, Some(EntrySkip::NotText));
    assert!(got.texts().contains(&"last"));
}

#[test]
fn other_nested_archives_are_skipped_and_reported() {
    let scratch = Scratch::new("nested");
    let inner_zip = crate_zip(&[("inner.log", b"inside\n", Pack::Stored)]);
    let mut tar = vec![0u8; 1024];
    tar[..9].copy_from_slice(b"inner.log");
    tar[257..262].copy_from_slice(b"ustar");
    let mut bzip2 = b"BZh91AY&SY".to_vec();
    bzip2.extend_from_slice(b"some bytes that are not checked");
    let mut xz = b"\xfd7zXZ\x00".to_vec();
    xz.extend_from_slice(&[1, 2, 3, 4]);
    let archive = scratch.write(
        "nested.zip",
        &crate_zip(&[
            ("inner.zip", &inner_zip, Pack::Stored),
            ("logs.tar", &tar, Pack::Deflated),
            ("logs.tar.gz", &gzip(&tar), Pack::Stored),
            ("twice.gz", &gzip(&gzip(b"inside\n")), Pack::Stored),
            ("old.bz2", &bzip2, Pack::Stored),
            ("old.xz", &xz, Pack::Stored),
            ("plain.log", b"outside\n", Pack::Stored),
        ]),
    );
    let got = read(&archive, &ZipInputSettings::default());
    assert_eq!(got.texts(), ["outside"]);
    for path in [
        "inner.zip",
        "logs.tar",
        "logs.tar.gz",
        "twice.gz",
        "old.bz2",
        "old.xz",
    ] {
        assert_eq!(
            got.entry(path).skipped,
            Some(EntrySkip::NestedArchive),
            "{path}"
        );
        assert_eq!(got.entry(path).records, 0, "{path}");
    }
}

// --- The maximum record length -------------------------------------------------------------

#[test]
fn an_over_long_record_is_skipped_with_its_position_and_length() {
    let scratch = Scratch::new("long");
    let exact = "e".repeat(16);
    let long = "L".repeat(17);
    let huge = "H".repeat(300_000);
    let content = format!("short\n{exact}\n{long}\n{exact}\r\n{huge}\nafter\n{long}");
    let archive = scratch.write(
        "long.zip",
        &crate_zip(&[
            ("a.log", b"before\n", Pack::Stored),
            ("long.log", content.as_bytes(), Pack::Deflated),
        ]),
    );
    let settings = ZipInputSettings {
        max_record_bytes: 16,
        ..ZipInputSettings::default()
    };
    let got = read(&archive, &settings);
    assert_eq!(got.texts(), ["before", "short", exact.as_str(), "after"]);
    assert_eq!(
        got.rows[2].occurrences, 2,
        "a carriage return is not part of the record"
    );
    assert_eq!(got.entry("long.log").records, 7);
    let skipped = got.skipped();
    let lines: Vec<u64> = skipped.iter().map(|(_, line, _)| *line).collect();
    assert_eq!(lines, [3, 5, 7]);
    for ((path, _, reason), length) in skipped.iter().zip(["17", "300000", "17"]) {
        assert_eq!(path, "long.log");
        assert!(
            reason.contains(&format!("{length} bytes")) && reason.contains("16 bytes"),
            "{reason}"
        );
    }

    // With the default of 64 KiB only the huge line is too long.
    assert_eq!(DEFAULT_MAX_RECORD_BYTES, 64 * 1024);
    let got = read(&archive, &ZipInputSettings::default());
    assert_eq!(got.skipped().len(), 1);
    assert_eq!(got.summary.input.distinct_texts, 5);
}

#[test]
fn a_maximum_record_length_outside_its_range_is_an_analyst_error() {
    let scratch = Scratch::new("range");
    let archive = scratch.write(
        "range.zip",
        &crate_zip(&[("a.log", b"one\n", Pack::Stored)]),
    );
    for max_record_bytes in [0, MAX_RECORD_BYTES + 1] {
        let settings = ZipInputSettings {
            max_record_bytes,
            ..ZipInputSettings::default()
        };
        assert!(
            matches!(
                open_error(&archive, &settings),
                L2dError::InvalidInput { .. }
            ),
            "{max_record_bytes}"
        );
    }
    let settings = ZipInputSettings {
        max_record_bytes: MAX_RECORD_BYTES,
        ..ZipInputSettings::default()
    };
    assert_eq!(read(&archive, &settings).texts(), ["one"]);
}

// --- Zip64 ---------------------------------------------------------------------------------

#[test]
fn a_zip64_archive_is_read() {
    let scratch = Scratch::new("zip64");
    let content = "a zip64 line\nanother\n".repeat(50);
    let bytes = crate_zip(&[
        ("small.log", b"first\n", Pack::Stored),
        ("large.log", content.as_bytes(), Pack::Large),
        ("after.log", b"last\n", Pack::Large),
    ]);
    // The entry carries the zip64 extra field.
    assert!(
        bytes
            .windows(4)
            .any(|window| window == [0x01, 0x00, 0x10, 0x00]
                || window == [0x01, 0x00, 0x1c, 0x00]
                || window == [0x01, 0x00, 0x18, 0x00])
    );
    let got = read(
        &scratch.write("zip64.zip", &bytes),
        &ZipInputSettings::default(),
    );
    assert_eq!(got.texts(), ["first", "a zip64 line", "another", "last"]);
    assert_eq!(got.entry("large.log").records, 100);
    assert!(
        got.summary
            .entries
            .iter()
            .all(|entry| entry.skipped.is_none())
    );
}

#[test]
fn sizes_and_offsets_in_zip64_fields_of_the_directory_are_read() {
    let scratch = Scratch::new("zip64-fields");
    let content = "a line behind zip64 fields\nand another\n".repeat(30);
    let mut stored = RawEntry::stored("stored.log", b"first\n");
    stored.zip64 = true;
    let mut deflated = RawEntry::deflated("deflated.log", content.as_bytes());
    deflated.zip64 = true;
    let raw = raw_zip(&[stored, deflated, RawEntry::stored("plain.log", b"last\n")]);
    // The `zip` crate reads the same archive.
    let mut reference = zip::ZipArchive::new(std::io::Cursor::new(raw.bytes.clone())).unwrap();
    assert_eq!(reference.by_index(1).unwrap().size(), content.len() as u64);
    let got = read(
        &scratch.write("fields.zip", &raw.bytes),
        &ZipInputSettings::default(),
    );
    assert_eq!(
        got.texts(),
        ["first", "a line behind zip64 fields", "and another", "last"]
    );
    assert_eq!(got.entry("deflated.log").records, 60);
    assert!(
        got.summary
            .entries
            .iter()
            .all(|entry| entry.skipped.is_none())
    );

    // A directory entry that announces zip64 fields and has none.
    let mut broken = raw.bytes.clone();
    put_u16(
        &mut broken,
        raw.central_at[0] + 46 + "stored.log".len(),
        0x7777,
    );
    assert!(matches!(
        open_error(
            &scratch.write("no-fields.zip", &broken),
            &ZipInputSettings::default()
        ),
        L2dError::InvalidInput { .. }
    ));
}

/// An archive of `count` stored entries `e<index>.log`, each one line.
fn many_entries(count: u32) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for index in 0..count {
        writer.start_file(format!("e{index}.log"), options).unwrap();
        writer
            .write_all(format!("line of entry {}\n", index % 1_000).as_bytes())
            .unwrap();
    }
    writer.finish().unwrap().into_inner()
}

#[test]
fn an_archive_of_more_than_65535_entries_is_read() {
    let scratch = Scratch::new("many");
    let bytes = many_entries(66_000);
    // More entries than the classic end record can count: the zip64 end record is there.
    assert!(bytes.windows(4).any(|window| window == b"PK\x06\x06"));
    let archive = scratch.write("many.zip", &bytes);
    let got = read(&archive, &ZipInputSettings::default());
    assert_eq!(got.summary.entries.len(), 66_000);
    assert_eq!(got.summary.input.records, 66_000);
    assert_eq!(got.summary.input.distinct_texts, 1_000);
    assert_eq!(got.rows[999].occurrences, 66);
    assert_eq!(
        got.summary.locate(got.rows[999].first_position),
        Some(("e999.log", 1))
    );
    assert!(got.summary.entries.iter().all(|entry| entry.records == 1));

    // The directory of 66,000 entries does not fit a limit that leaves 2 MiB.
    let mut small = budget(Some(34 * MIB), None);
    match read_with(&archive, &ZipInputSettings::default(), &mut small) {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains("66000 entries"), "{reason}");
            assert!(reason.contains("memory limit"), "{reason}");
        }
        other => panic!("{other:?}"),
    }

    // A zip64 end record that claims more entries than its directory can hold.
    let mut lying = bytes.clone();
    let end64 = last_of(&lying, b"PK\x06\x06");
    put_u64(&mut lying, end64 + 24, 1 << 40);
    put_u64(&mut lying, end64 + 32, 1 << 40);
    assert!(matches!(
        open_error(
            &scratch.write("lying.zip", &lying),
            &ZipInputSettings::default()
        ),
        L2dError::InvalidInput { .. }
    ));
}

// --- Hostile archives ----------------------------------------------------------------------

#[test]
fn entry_names_are_only_names_and_nothing_is_written() {
    let scratch = Scratch::new("names");
    let names = [
        "../../escaped.log",
        "/absolute/escaped.log",
        "C:\\Windows\\escaped.log",
        "..\\..\\escaped.log",
        "nul",
        "ok/../../escaped.log",
    ];
    let entries: Vec<RawEntry> = names
        .iter()
        .enumerate()
        .map(|(index, name)| RawEntry::stored(name, format!("line {index}\n").as_bytes()))
        .collect();
    let archive = scratch.write("names.zip", &raw_zip(&entries).bytes);
    let before = scratch.listing();
    let got = read(&archive, &ZipInputSettings::default());
    // The entries are read as what they are: data under a name.
    assert_eq!(got.rows.len(), names.len());
    let paths: Vec<&str> = got
        .summary
        .entries
        .iter()
        .map(|entry| entry.path.as_str())
        .collect();
    assert_eq!(paths, names);
    assert_eq!(
        scratch.listing(),
        before,
        "nothing was written beside the archive"
    );
    assert_eq!(before, ["names.zip"]);
    assert!(!scratch.dir.parent().unwrap().join("escaped.log").exists());
    assert_eq!(got.summary.input.source_path, archive);

    // A name that is not UTF-8 is reported with replacement characters.
    let mut odd = RawEntry::stored("odd.log", b"odd\n");
    odd.name = b"odd-\xff\xfe.log".to_vec();
    let got = read(
        &scratch.write("odd.zip", &raw_zip(&[odd]).bytes),
        &ZipInputSettings::default(),
    );
    assert_eq!(got.summary.entries[0].path, "odd-\u{fffd}\u{fffd}.log");
    assert_eq!(got.texts(), ["odd"]);
}

#[test]
fn a_file_that_is_no_archive_or_whose_directory_is_damaged_is_an_analyst_error() {
    let scratch = Scratch::new("directory");
    let invalid = |name: &str, bytes: &[u8]| match open_error(
        &scratch.write(name, bytes),
        &ZipInputSettings::default(),
    ) {
        L2dError::InvalidInput { detail } => detail,
        other => panic!("{name}: {other:?}"),
    };
    let entries = [
        RawEntry::stored("a.log", b"one\n"),
        RawEntry::stored("b.log", b"two\n"),
    ];
    let good = raw_zip(&entries);

    invalid("empty.zip", b"");
    invalid("text.zip", b"just a text file, not an archive\n");
    // Cut off in the middle of the directory: no end record.
    invalid("cut.zip", &good.bytes[..good.central_at[1] + 10]);

    // The end record counts more entries than the directory holds.
    let mut lying = good.bytes.clone();
    put_u16(&mut lying, good.end_at + 8, 60_000);
    put_u16(&mut lying, good.end_at + 10, 60_000);
    invalid("count.zip", &lying);

    // The directory is said to start beyond the file.
    let mut lying = good.bytes.clone();
    put_u32(&mut lying, good.end_at + 16, 0x00ff_ffff);
    invalid("start.zip", &lying);

    // The directory is said to be larger than the file.
    let mut lying = good.bytes.clone();
    put_u32(&mut lying, good.end_at + 12, 0x00ff_ffff);
    invalid("size.zip", &lying);

    // A directory entry without its signature.
    let mut broken = good.bytes.clone();
    broken[good.central_at[1]] = b'X';
    let detail = invalid("signature.zip", &broken);
    assert!(detail.contains("directory"), "{detail}");

    // An archive in several parts.
    let mut parts = good.bytes.clone();
    put_u16(&mut parts, good.end_at + 4, 1);
    invalid("parts.zip", &parts);

    // An archive without entries is an archive.
    let got = read(
        &scratch.write("none.zip", &raw_zip(&[]).bytes),
        &ZipInputSettings::default(),
    );
    assert_eq!(got.summary.input.records, 0);
    assert!(got.summary.entries.is_empty());
}

#[test]
fn a_missing_archive_is_missing() {
    let scratch = Scratch::new("missing");
    let path = scratch.dir.join("absent.zip");
    match open_error(&path, &ZipInputSettings::default()) {
        L2dError::Missing { expected_at, .. } => assert_eq!(expected_at, path),
        other => panic!("{other:?}"),
    }
}

// --- Memory --------------------------------------------------------------------------------

#[test]
fn a_deduplication_set_that_does_not_fit_is_refused_with_needed_and_available() {
    let scratch = Scratch::new("set");
    let mut content = String::new();
    for index in 0..600_000u32 {
        content.push_str(&format!("u{index}\n"));
    }
    let archive = scratch.write(
        "set.zip",
        &crate_zip(&[("distinct.log", content.as_bytes(), Pack::Deflated)]),
    );
    // 600,000 distinct texts need a set of 2^20 slots; 12 MiB beside the tool's own
    // share do not hold it.
    let limit = 44 * MIB;
    let mut small = budget(Some(limit), None);
    match read_with(&archive, &ZipInputSettings::default(), &mut small) {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains("deduplication set"), "{reason}");
            assert!(reason.contains("needs"), "{reason}");
            assert!(reason.contains(&format!("({limit} bytes)")), "{reason}");
            assert!(reason.contains("distinct texts"), "{reason}");
        }
        other => panic!("{:?}", other.map(|read| read.summary.input.distinct_texts)),
    }
    // With room it is read, and the set is what is held at the end.
    let mut roomy = budget(Some(512 * MIB), None);
    let got = read_with(&archive, &ZipInputSettings::default(), &mut roomy).unwrap();
    assert_eq!(got.summary.input.distinct_texts, 600_000);
    assert_eq!(got.summary.input.records, 600_000);
}

#[test]
fn a_record_length_the_memory_limit_cannot_hold_is_refused_before_reading() {
    let scratch = Scratch::new("record-room");
    let archive = scratch.write("room.zip", &crate_zip(&[("a.log", b"one\n", Pack::Stored)]));
    let settings = ZipInputSettings {
        max_record_bytes: MAX_RECORD_BYTES,
        ..ZipInputSettings::default()
    };
    // 37 MiB leave 1 MiB beside the tool's share and the fixed memory of a batch: a
    // record of 1 MiB cannot be parsed in that.
    let mut small = budget(Some(37 * MIB), None);
    match read_with(&archive, &settings, &mut small) {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains("maximum record length"), "{reason}");
        }
        other => panic!("{:?}", other.map(|read| read.summary.input.records)),
    }
}
