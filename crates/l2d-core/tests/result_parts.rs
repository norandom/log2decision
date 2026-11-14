//! Result parts on disk and deduplication on disk (requirements 13.11 and 13.12; task
//! 1.17), on work directories and archives written into a temporary directory.

use std::io::Write;
use std::path::{Path, PathBuf};

use l2d_core::parts::{
    MANIFEST_FILE, PartTable, PartWriter, WorkDir, free_disk_bytes, read_part, write_manifest,
};
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::{
    ArchiveSummary, FreezeDiskCheck, L2dError, LogsFrame, MemoryBudget, MemorySettings, ZipInput,
    ZipInputSettings,
};
use sha2::{Digest, Sha256};

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
            std::env::temp_dir().join(format!("l2d-core-parts-{}-{name}", std::process::id()));
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

/// Every file and directory below `dir`, relative, sorted.
fn listing(dir: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            found.push(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
            if path.is_dir() {
                walk(root, &path, found);
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, dir, &mut found);
    found.sort();
    found
}

fn budget(limit: Option<u64>, batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap()
}

fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    for (name, content) in entries {
        writer.start_file(*name, options).unwrap();
        writer.write_all(content).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

/// Rows `from..to` of a table with every type a part takes.
fn rows(from: u64, to: u64) -> DataFrame {
    let range = || from..to;
    DataFrame::new(vec![
        Column::new("row".into(), range().collect::<Vec<u64>>()),
        Column::new(
            "text".into(),
            range()
                .map(|row| match row % 4 {
                    0 => None,
                    _ => Some(format!("GET /item/{row} 200 \u{e9}")),
                })
                .collect::<Vec<Option<String>>>(),
        ),
        Column::new(
            "count".into(),
            range().map(|row| row as u32 * 3).collect::<Vec<u32>>(),
        ),
        Column::new(
            "ratio".into(),
            range().map(|row| row as f32 / 7.0).collect::<Vec<f32>>(),
        ),
        Column::new(
            "flag".into(),
            range().map(|row| row % 3 == 0).collect::<Vec<bool>>(),
        ),
    ])
    .unwrap()
}

/// Writes rows `0..total` as the table `scores`, cut at `cuts`.
fn write_table(dir: &WorkDir, total: u64, cuts: &[u64]) -> PartTable {
    let mut writer = PartWriter::new(dir, "scores").unwrap();
    let mut from = 0;
    for to in cuts.iter().copied().chain([total]) {
        writer.write(&mut rows(from, to)).unwrap();
        from = to;
    }
    writer.finish()
}

fn file_sha256(path: &Path) -> String {
    Sha256::digest(std::fs::read(path).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// --- Parts ---------------------------------------------------------------------------------

#[test]
fn parts_hold_the_rows_and_the_table_names_them() {
    let scratch = Scratch::new("parts");
    let dir = WorkDir::create(&scratch.path("run")).unwrap();
    let table = write_table(&dir, 10, &[4, 4, 9]);

    // The empty frame between 4 and 4 writes no file.
    assert_eq!(table.name, "scores");
    assert_eq!(table.rows, 10);
    let files: Vec<&str> = table.parts.iter().map(|part| part.file.as_str()).collect();
    assert_eq!(
        files,
        [
            "scores-000000.parquet",
            "scores-000001.parquet",
            "scores-000002.parquet"
        ]
    );
    assert_eq!(listing(dir.path()), files);
    let names: Vec<(&str, &str)> = table
        .columns
        .iter()
        .map(|column| (column.name.as_str(), column.dtype.as_str()))
        .collect();
    assert_eq!(
        names,
        [
            ("row", "u64"),
            ("text", "str"),
            ("count", "u32"),
            ("ratio", "f32"),
            ("flag", "bool")
        ]
    );

    let mut from = 0;
    for part in &table.parts {
        let path = dir.path().join(&part.file);
        assert_eq!(part.bytes, std::fs::metadata(&path).unwrap().len());
        assert_eq!(part.sha256, file_sha256(&path));
        let read = read_part(&path).unwrap();
        assert_eq!(read.height() as u64, part.rows);
        assert!(read.equals_missing(&rows(from, from + part.rows)));
        from += part.rows;
    }
    assert_eq!(from, 10);
    assert_eq!(
        table.bytes(),
        table.parts.iter().map(|part| part.bytes).sum::<u64>()
    );
}

#[test]
fn the_same_rows_give_the_same_bytes_and_one_rows_hash_however_they_are_cut() {
    let scratch = Scratch::new("deterministic");
    let table_in = |name: &str, cuts: &[u64]| {
        let dir = WorkDir::create(&scratch.path(name)).unwrap();
        (write_table(&dir, 1_000, cuts), dir)
    };
    let (first, first_dir) = table_in("first", &[300, 650]);
    let (second, second_dir) = table_in("second", &[300, 650]);
    let (whole, _) = table_in("whole", &[]);
    let (single, _) = table_in("single", &(1..1_000).collect::<Vec<u64>>());

    // The same frames: the same files, byte for byte.
    assert_eq!(first, second);
    for part in &first.parts {
        assert_eq!(
            std::fs::read(first_dir.path().join(&part.file)).unwrap(),
            std::fs::read(second_dir.path().join(&part.file)).unwrap()
        );
    }
    // Other cuts: other files, the same rows hash.
    assert_eq!(whole.parts.len(), 1);
    assert_eq!(single.parts.len(), 1_000);
    assert_eq!(whole.rows_sha256, first.rows_sha256);
    assert_eq!(single.rows_sha256, first.rows_sha256);
    assert_ne!(whole.parts[0].sha256, first.parts[0].sha256);

    // Other rows, another order, another value: another rows hash.
    let other = |frames: Vec<DataFrame>| {
        let dir = WorkDir::create(&scratch.path(&format!("other-{}", frames.len()))).unwrap();
        let mut writer = PartWriter::new(&dir, "scores").unwrap();
        for mut frame in frames {
            writer.write(&mut frame).unwrap();
        }
        writer.finish().rows_sha256
    };
    assert_ne!(other(vec![rows(0, 999)]), first.rows_sha256);
    assert_ne!(
        other(vec![rows(500, 1_000), rows(0, 500)]),
        first.rows_sha256
    );
    let mut changed = rows(0, 1_000);
    changed
        .with_column(Column::new(
            "flag".into(),
            (0..1_000u64).map(|row| row % 3 == 1).collect::<Vec<bool>>(),
        ))
        .unwrap();
    assert_ne!(
        other(vec![changed, rows(0, 0), rows(0, 0)]),
        first.rows_sha256
    );
}

#[test]
fn a_part_with_other_columns_or_an_unknown_type_is_not_written() {
    let scratch = Scratch::new("columns");
    let dir = WorkDir::create(&scratch.path("run")).unwrap();
    let mut writer = PartWriter::new(&dir, "scores").unwrap();
    writer.write(&mut rows(0, 3)).unwrap();
    let mut fewer = rows(3, 6).drop("flag").unwrap();
    let error = writer.write(&mut fewer).unwrap_err();
    assert!(matches!(error, L2dError::Runtime { .. }), "{error}");

    let mut signed = DataFrame::new(vec![Column::new("value".into(), vec![1i64, 2, 3])]).unwrap();
    let mut other = PartWriter::new(&dir, "other").unwrap();
    let error = other.write(&mut signed).unwrap_err();
    assert!(error.to_string().contains("i64"), "{error}");
    assert_eq!(listing(dir.path()), ["scores-000000.parquet"]);

    for name in ["", "Scores", "a/b", "..", "a b", "a.b"] {
        assert!(
            matches!(
                PartWriter::new(&dir, name),
                Err(L2dError::InvalidInput { .. })
            ),
            "{name}"
        );
    }
}

#[test]
fn a_work_directory_that_holds_something_is_refused_and_left_alone() {
    let scratch = Scratch::new("workdir");
    // Absent, with absent parents: created. Empty: taken.
    let fresh = scratch.path("a/b/run");
    let dir = WorkDir::create(&fresh).unwrap();
    assert!(fresh.is_dir());
    assert_eq!(dir.path(), fresh);
    WorkDir::create(&fresh).unwrap();

    std::fs::write(fresh.join("notes.txt"), b"mine").unwrap();
    let error = WorkDir::create(&fresh).unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
    assert!(
        error.to_string().contains("not an empty directory"),
        "{error}"
    );
    assert_eq!(std::fs::read(fresh.join("notes.txt")).unwrap(), b"mine");

    let file = scratch.write("plain.txt", b"x");
    let error = WorkDir::create(&file).unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
    assert_eq!(std::fs::read(&file).unwrap(), b"x");
}

#[test]
fn too_little_free_disk_space_is_refused_with_both_figures() {
    let scratch = Scratch::new("disk");
    let dir = WorkDir::create(&scratch.path("run")).unwrap();
    let needed = [
        ("the result parts", 3 * MIB),
        ("deduplication on disk", 5 * MIB),
    ];
    dir.require_free_of(8 * MIB, &needed).unwrap();
    let error = dir.require_free_of(8 * MIB - 1, &needed).unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
    let text = error.to_string();
    for figure in [
        "8388608 bytes",
        "8388607 bytes",
        "3145728 bytes",
        "5242880 bytes",
        "the result parts",
        "deduplication on disk",
        "nothing was read",
    ] {
        assert!(text.contains(figure), "{figure}: {text}");
    }

    // The figure of the operating system: something is free where the tests run, and
    // not more than any disk holds.
    let free = free_disk_bytes(dir.path()).unwrap();
    assert!(free > 0 && free < 1 << 60, "{free}");
    assert!(dir.require_free(&[("a part", 1)]).unwrap() > 0);
    let error = dir.require_free(&[("a part", u64::MAX)]).unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
}

#[test]
fn the_manifest_is_written_once_and_whole() {
    let scratch = Scratch::new("manifest");
    let dir = WorkDir::create(&scratch.path("run")).unwrap();
    let table = write_table(&dir, 5, &[2]);
    let path = write_manifest(&dir, &serde_json::json!({ "tables": [table] })).unwrap();
    assert_eq!(path, dir.path().join(MANIFEST_FILE));
    let read: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let back: PartTable = serde_json::from_value(read["tables"][0].clone()).unwrap();
    assert_eq!(back, table);
    assert_eq!(
        listing(dir.path()),
        [
            "manifest.json",
            "scores-000000.parquet",
            "scores-000001.parquet"
        ]
    );
    let before = std::fs::read(&path).unwrap();
    let error = write_manifest(&dir, &serde_json::json!({})).unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

// --- Deduplication on disk -----------------------------------------------------------------

/// What reading an archive to its end gave: every row with its count over the archive.
#[derive(Debug, PartialEq)]
struct Read {
    /// Text, packed first position, occurrences over the archive.
    rows: Vec<(String, u64, u32)>,
    summary: ArchiveSummary,
    /// Whether the read went on on disk, and how many texts the set in memory held.
    on_disk: Option<u64>,
}

/// Reads the archive in memory alone: the reference.
fn read_in_memory(path: &Path, settings: &ZipInputSettings, batch_rows: Option<u64>) -> Read {
    let mut budget = budget(None, batch_rows);
    let mut input = ZipInput::open(path, settings, &budget).unwrap();
    let mut rows = Vec::new();
    for batch in input.batches(&mut budget) {
        for record in batch.unwrap().records() {
            rows.push((record.text, record.first_position, 0));
        }
    }
    for row in &mut rows {
        row.2 = u32::try_from(input.occurrences(&row.0)).unwrap();
    }
    assert!(input.disk_dedup().is_none());
    Read {
        rows,
        summary: input.finish(),
        on_disk: None,
    }
}

/// Reads the archive with a spill directory; the set in memory holds at most
/// `set_texts` texts when given.
fn read_with_spill(
    path: &Path,
    settings: &ZipInputSettings,
    spill: &Path,
    set_texts: Option<u64>,
    budget: &mut MemoryBudget,
) -> Result<Read, L2dError> {
    let mut input = ZipInput::open(path, settings, budget)?;
    input.spill_into(spill, set_texts)?;
    let mut rows = Vec::new();
    for batch in input.batches(budget) {
        let batch: LogsFrame = batch?;
        assert!(!batch.is_empty(), "a batch without rows is not handed out");
        for record in batch.records() {
            rows.push((record.text, record.first_position, 0));
        }
    }
    let mut counts = Vec::new();
    for chunk in input.occurrence_counts() {
        counts.extend(chunk?);
    }
    assert_eq!(counts.len(), rows.len(), "one count per row handed out");
    for (row, count) in rows.iter_mut().zip(counts) {
        row.2 = count;
    }
    assert!(input.batch_plan().is_some());
    let on_disk = input.disk_dedup().map(|report| {
        assert_eq!(
            report.set_texts + report.spilled_distinct,
            rows.len() as u64,
            "{report:?}"
        );
        assert!(
            report.spilled_records >= report.spilled_distinct,
            "{report:?}"
        );
        report.set_texts
    });
    Ok(Read {
        rows,
        summary: input.finish(),
        on_disk,
    })
}

/// Three entries whose texts repeat inside an entry, across entries and across any
/// batch boundary, with blank lines, an over-long line and a line that is not UTF-8.
fn repeating() -> Vec<u8> {
    let mut first = String::new();
    let mut second = String::new();
    let mut third = Vec::new();
    for line in 0..400u32 {
        first.push_str(&format!("GET /page/{} 200\n", line % 37));
        if line % 50 == 7 {
            first.push('\n');
        }
        second.push_str(&format!("GET /page/{} 200\n", line % 61));
        second.push_str(&format!("POST /form/{} 302\r\n", line % 5));
        third.extend_from_slice(format!("GET /page/{} 200\n", 60 - line % 61).as_bytes());
        third.extend_from_slice(format!("unique line {line} \u{feff}x\n").as_bytes());
    }
    third.extend_from_slice(b"\xff\xfe broken\n");
    third.extend_from_slice(&vec![b'x'; 300]);
    third.extend_from_slice(b"\nlast line\n");
    archive(&[
        ("logs/first.log", first.as_bytes()),
        ("logs/second.log", second.as_bytes()),
        ("third.log", &third),
    ])
}

#[test]
fn deduplication_on_disk_gives_the_rows_and_counts_of_the_memory_path() {
    let scratch = Scratch::new("equal");
    let path = scratch.write("repeating.zip", &repeating());
    let settings = ZipInputSettings {
        max_record_bytes: 200,
        ..ZipInputSettings::default()
    };
    let reference = read_in_memory(&path, &settings, None);
    assert_eq!(reference.summary.input.records, 2_011);
    assert_eq!(reference.rows.len(), 467);
    assert_eq!(reference.summary.input.skipped.len(), 10);
    let total: u64 = reference.rows.iter().map(|row| u64::from(row.2)).sum();
    assert_eq!(total, 2_011 - 10);
    assert_eq!(reference.rows[0], ("GET /page/0 200".to_string(), 1, 24));

    let spill = scratch.path("spill");
    for batch_rows in [1, 3, 100, 4_096] {
        assert_eq!(
            read_in_memory(&path, &settings, Some(batch_rows)),
            reference,
            "in memory, batches of {batch_rows}"
        );
        for set_texts in [None, Some(0), Some(37), Some(200), Some(466), Some(467)] {
            // Batches of one row are slow to read: two of the cases are enough there.
            if batch_rows == 1 && !matches!(set_texts, Some(0 | 200)) {
                continue;
            }
            let mut budget = budget(None, Some(batch_rows));
            let read = read_with_spill(&path, &settings, &spill, set_texts, &mut budget)
                .unwrap_or_else(|error| panic!("{batch_rows} {set_texts:?}: {error}"));
            let what = format!("batches of {batch_rows}, at most {set_texts:?} texts in memory");
            assert_eq!(read.rows, reference.rows, "{what}");
            assert_eq!(read.summary, reference.summary, "{what}");
            match set_texts {
                // With room for every text nothing goes to disk.
                None | Some(467) => assert_eq!(read.on_disk, None, "{what}"),
                Some(most) => {
                    let held = read
                        .on_disk
                        .unwrap_or_else(|| panic!("{what}: not on disk"));
                    assert!(held <= most, "{what}: {held} texts in memory");
                }
            }
            // Nothing is left of the deduplication on disk.
            assert!(!spill.exists(), "{what}: {:?}", listing(&spill));
        }
    }
    assert_eq!(listing(&scratch.dir), ["repeating.zip"]);
}

#[test]
fn the_files_of_deduplication_on_disk_stay_in_its_directory() {
    let scratch = Scratch::new("inside");
    // Entry names that would leave the directory if they were used as paths.
    let content = |tag: &str| {
        (0..300)
            .map(|line| format!("{tag} {}\n", line % 90))
            .collect::<String>()
    };
    let (up, absolute, plain) = (content("up"), content("absolute"), content("up"));
    let bytes = archive(&[
        ("../../escaped.log", up.as_bytes()),
        ("/absolute.log", absolute.as_bytes()),
        ("C:\\drive.log", plain.as_bytes()),
    ]);
    let path = scratch.write("names.zip", &bytes);
    let spill = scratch.path("work/dedup");
    let mut budget = budget(None, Some(16));
    let settings = ZipInputSettings::default();

    let mut input = ZipInput::open(&path, &settings, &budget).unwrap();
    input.spill_into(&spill, Some(10)).unwrap();
    let mut rows = 0;
    let mut seen_files = Vec::new();
    for batch in input.batches(&mut budget) {
        rows += batch.unwrap().len();
        seen_files = listing(&spill);
    }
    assert_eq!(rows, 180);
    // While it ran, the directory held files with made-up names only.
    assert!(!seen_files.is_empty());
    for file in &seen_files {
        assert!(
            file.ends_with(".bin") && !file.contains("log") && !file.contains('/'),
            "{seen_files:?}"
        );
    }
    let summary = input.finish();
    assert_eq!(summary.input.distinct_texts, 180);
    assert_eq!(listing(&scratch.dir), ["names.zip", "work"]);

    // A spill directory that holds something is not used.
    std::fs::create_dir_all(&spill).unwrap();
    std::fs::write(spill.join("keep.txt"), b"mine").unwrap();
    let mut input = ZipInput::open(&path, &settings, &budget).unwrap();
    let error = input.spill_into(&spill, None).unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
    drop(input);
    assert_eq!(std::fs::read(spill.join("keep.txt")).unwrap(), b"mine");
}

#[test]
fn a_read_that_is_dropped_or_fails_removes_its_files() {
    let scratch = Scratch::new("dropped");
    let path = scratch.write("repeating.zip", &repeating());
    let spill = scratch.path("spill");
    let mut budget = budget(None, Some(50));
    let mut input = ZipInput::open(&path, &ZipInputSettings::default(), &budget).unwrap();
    input.spill_into(&spill, Some(5)).unwrap();
    let mut batches = input.batches(&mut budget);
    batches.next().unwrap().unwrap();
    batches.next();
    drop(batches);
    assert!(!listing(&spill).is_empty());
    drop(input);
    assert!(!spill.exists());
}

#[test]
fn a_set_that_outgrows_the_memory_limit_goes_on_on_disk() {
    let scratch = Scratch::new("outgrown");
    // 300,000 distinct lines, each twice, the second time in another entry and in
    // reverse: the set of their keys needs a table of 11 MiB and 22 MiB to grow to it.
    let lines = 300_000u32;
    let mut forward = String::new();
    let mut backward = String::new();
    for line in 0..lines {
        forward.push_str(&format!("line {line}\n"));
        backward.push_str(&format!("line {}\n", lines - 1 - line));
    }
    let path = scratch.write(
        "many.zip",
        &archive(&[
            ("forward.log", forward.as_bytes()),
            ("backward.log", backward.as_bytes()),
        ]),
    );
    let settings = ZipInputSettings::default();
    let limit = Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 20 * MIB);

    // In memory alone the read is refused, with the memory needed and available.
    let mut short = budget(limit, None);
    let mut input = ZipInput::open(&path, &settings, &short).unwrap();
    let error = input
        .batches(&mut short)
        .find_map(Result::err)
        .expect("the read is refused in memory");
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
    assert!(error.to_string().contains("deduplication set"), "{error}");
    drop(input);

    // With a spill directory it completes, with the rows of a read with room.
    let mut short = budget(limit, None);
    let spill = scratch.path("spill");
    let read = read_with_spill(&path, &settings, &spill, None, &mut short).unwrap();
    let held = read.on_disk.expect("the read went on on disk");
    assert!(held > 0 && held < u64::from(lines), "{held}");
    assert_eq!(read.rows.len() as u64, u64::from(lines));
    assert_eq!(read.summary.input.distinct_texts, u64::from(lines));
    assert_eq!(read.summary.input.records, 2 * u64::from(lines));
    for (index, (text, position, count)) in read.rows.iter().enumerate() {
        assert_eq!(text, &format!("line {index}"));
        assert_eq!(*position, index as u64 + 1);
        assert_eq!(*count, 2);
    }
    assert!(!spill.exists());
}

#[test]
fn what_the_selected_entries_hold_is_known_before_the_read() {
    let scratch = Scratch::new("selected");
    let path = scratch.write("repeating.zip", &repeating());
    let budget = budget(None, None);
    let all = ZipInput::open(&path, &ZipInputSettings::default(), &budget).unwrap();
    let logs = ZipInput::open(
        &path,
        &ZipInputSettings {
            entries: vec!["logs/*".into()],
            ..ZipInputSettings::default()
        },
        &budget,
    )
    .unwrap();
    let (all, logs) = (all.selected_bytes(), logs.selected_bytes());
    assert_eq!((all.entries, logs.entries), (3, 2));
    assert!(
        logs.unpacked < all.unpacked && logs.packed < all.packed,
        "{all:?} {logs:?}"
    );
    assert!(all.packed < all.unpacked, "{all:?}");
}

// --- Task 1.20 -----------------------------------------------------------------------------

/// `count` distinct lines whose keys all fall into partition 0 of deduplication on
/// disk: the partition of a text is chosen by the first byte of its key, which is the
/// first byte of its SHA-256.
fn lines_of_one_partition(count: usize) -> String {
    let mut lines = String::new();
    let mut found = 0;
    let mut index = 0u64;
    while found < count {
        let text = format!("h{index}");
        if Sha256::digest(text.as_bytes())[0] % 64 == 0 {
            lines.push_str(&text);
            lines.push('\n');
            found += 1;
        }
        index += 1;
    }
    lines
}

/// Task 1.20: a partition whose keys will not fit is refused as soon as it outgrows
/// the room, while the archive is still being read, with what is needed and what is
/// left.
#[test]
fn a_partition_whose_keys_do_not_fit_is_refused_while_the_archive_is_read() {
    let scratch = Scratch::new("partition");
    let total = 70_000u64;
    let path = scratch.write(
        "one-partition.zip",
        &archive(&[(
            "hostile.log",
            lines_of_one_partition(total as usize).as_bytes(),
        )]),
    );
    let settings = ZipInputSettings {
        max_record_bytes: 1_024,
        ..ZipInputSettings::default()
    };
    // What the limit leaves beside the buffers and a batch holds the keys of fewer
    // rows than the archive puts into its one partition.
    let limit = Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 9 * MIB);
    let mut budget = budget(limit, Some(1_024));
    let spill = scratch.path("work/dedup");
    let mut input = ZipInput::open(&path, &settings, &budget).unwrap();
    // No text in memory: every row goes to disk.
    input.spill_into(&spill, Some(0)).unwrap();
    let error = input
        .batches(&mut budget)
        .find_map(Result::err)
        .expect("the partition is refused");
    assert!(matches!(error, L2dError::Refused { .. }), "{error}");
    let text = error.to_string();
    for part in ["partition 0 of 64", "needs", "is left", " bytes)"] {
        assert!(text.contains(part), "{part}: {text}");
    }
    let summary = input.finish();
    assert!(
        summary.input.records < total,
        "refused after {} of {total} records: {text}",
        summary.input.records
    );
    // Nothing is left of the deduplication on disk.
    assert!(!spill.exists(), "{:?}", listing(&scratch.dir));
}

/// Task 1.20: what deduplication on disk holds in memory is counted from the moment
/// its directory is named, the chunks the occurrence counts are read in included.
#[test]
fn the_buffers_of_deduplication_on_disk_and_of_the_counts_are_held() {
    let scratch = Scratch::new("held");
    let path = scratch.write("repeating.zip", &repeating());
    let budget = budget(None, None);
    let mut input = ZipInput::open(&path, &ZipInputSettings::default(), &budget).unwrap();
    let before = input.held_bytes();
    input.spill_into(&scratch.path("spill"), None).unwrap();
    let held = input.held_bytes() - before;
    // 66 buffers of 32 KiB; and for the counts a chunk of 65,536 keys of 16 bytes
    // with its 65,536 counts of 4 bytes.
    let (buffers, counts) = (66 * 32 * 1_024, 65_536 * (16 + 4));
    assert!(
        held >= buffers + counts,
        "{held} bytes are held, less than {buffers} and {counts}"
    );
    assert!(held < buffers + counts + MIB, "{held}");
}

/// Task 1.20: a work directory below a file is the analyst's error, said plainly.
#[test]
fn a_work_directory_below_a_file_is_an_analyst_error() {
    let scratch = Scratch::new("below-file");
    let file = scratch.write("plain.txt", b"x");
    for below in ["run", "a/b/run"] {
        let error = WorkDir::create(&file.join(below)).unwrap_err();
        assert!(matches!(error, L2dError::InvalidInput { .. }), "{error}");
        let text = error.to_string();
        assert!(text.contains("is a file"), "{text}");
        assert!(text.contains("plain.txt"), "{text}");
        assert!(!text.contains("os error"), "{text}");
    }
    assert_eq!(std::fs::read(&file).unwrap(), b"x");
    assert_eq!(listing(&scratch.dir), ["plain.txt"]);
}

/// Task 1.20: a run that leaves nothing in its directory removes what it made, and
/// nothing else.
#[test]
fn the_directories_a_run_made_are_removed_as_far_as_they_are_empty() {
    let scratch = Scratch::new("made");
    // Absent with absent parents: all of them go again.
    let dir = WorkDir::create(&scratch.path("a/b/run")).unwrap();
    dir.remove_created();
    assert_eq!(listing(&scratch.dir), Vec::<String>::new());

    // Something was written: the directory stays, and what is above it.
    let dir = WorkDir::create(&scratch.path("a/b/run")).unwrap();
    std::fs::write(dir.path().join("part.parquet"), b"x").unwrap();
    dir.remove_created();
    assert_eq!(
        listing(&scratch.dir),
        ["a", "a/b", "a/b/run", "a/b/run/part.parquet"]
    );
    std::fs::remove_file(dir.path().join("part.parquet")).unwrap();

    // A directory that was there before is not removed, nor what is above it.
    let dir = WorkDir::create(&scratch.path("a/b/run")).unwrap();
    dir.remove_created();
    assert_eq!(listing(&scratch.dir), ["a", "a/b", "a/b/run"]);
    // Below one that was there: only what was made.
    let dir = WorkDir::create(&scratch.path("a/b/run/deeper/still")).unwrap();
    dir.remove_created();
    assert_eq!(listing(&scratch.dir), ["a", "a/b", "a/b/run"]);
}

/// The bytes of the files below `dir`; 0 when it is absent.
fn bytes_below(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .map(|entry| entry.unwrap().metadata().map_or(0, |file| file.len()))
        .sum()
}

/// Task 1.20: the estimate of what deduplication on disk writes counts the 32 bytes of
/// a row on disk, so that it holds for an archive of very short lines.
#[test]
fn the_disk_estimate_of_deduplication_holds_for_short_lines() {
    let scratch = Scratch::new("short");
    // 400,000 distinct lines of 1 to 6 digits, 6.7 bytes each with the line end.
    let lines = 400_000u64;
    let content: String = (0..lines).map(|line| format!("{line}\n")).collect();
    let path = scratch.write("short.zip", &archive(&[("short.log", content.as_bytes())]));
    let spill = scratch.path("work/dedup");
    let mut budget = budget(None, None);
    let mut input = ZipInput::open(&path, &ZipInputSettings::default(), &budget).unwrap();
    let selected = input.selected_bytes();
    assert_eq!(selected.unpacked, content.len() as u64);
    let estimate = selected.dedup_disk_bytes();
    // The unpacked bytes, and 32 bytes for every 6 of them.
    assert_eq!(estimate, selected.unpacked + 32 * (selected.unpacked / 6));

    // All but 50,000 texts go to disk.
    input.spill_into(&spill, Some(50_000)).unwrap();
    let mut rows = 0;
    let mut most = 0;
    for batch in input.batches(&mut budget) {
        rows += batch.unwrap().len() as u64;
        most = most.max(bytes_below(&spill));
    }
    for chunk in input.occurrence_counts() {
        chunk.unwrap();
        most = most.max(bytes_below(&spill));
    }
    assert_eq!(rows, lines);
    let report = input.disk_dedup().expect("the read went on on disk");
    assert_eq!(report.spilled_records, lines - report.set_texts);
    // The partitions when they were complete, and the keys of the rows of the read.
    let written = report.spilled_bytes + 16 * report.set_texts;
    assert!(
        report.spilled_bytes > selected.unpacked,
        "the case: more is written than the archive unpacks to: {report:?}"
    );
    assert!(
        written <= estimate && most <= estimate,
        "{written} bytes were written, {most} seen at once, {estimate} estimated"
    );
    drop(input);
    assert!(!spill.exists());
}

// --- Task 1.20, remediation after the review of d03b1db ------------------------------------

/// The archive of the review of d03b1db: 300,000 distinct filler lines, then 1,000,000
/// rows of only 10 distinct texts, all of whose keys fall into partition 0.
fn filler_then_ten() -> Vec<u8> {
    let mut content = String::new();
    for line in 0..300_000 {
        content.push_str(&format!("filler line number {line} of the archive\n"));
    }
    let mut ten = Vec::new();
    let mut index = 0u64;
    while ten.len() < 10 {
        let text = format!("dup{index}");
        if Sha256::digest(text.as_bytes())[0] % 64 == 0 {
            ten.push(text);
        }
        index += 1;
    }
    for row in 0..1_000_000 {
        content.push_str(&ten[row % 10]);
        content.push('\n');
    }
    archive(&[("dup.log", content.as_bytes())])
}

/// Remediation, item 3: the room a partition needs is decided by the distinct keys it
/// holds, not by its rows on disk, whose number grows as the batches get smaller
/// (requirement 12.6: the outcome does not depend on the batch size). The archive of
/// the review, read four times; in batches of 10 that takes minutes in a debug build,
/// so the test runs on request. The default run has the same case on `Spill` itself
/// (`zip_spill::tests`).
///
/// ```text
/// cargo test -j 4 -p l2d-core --test result_parts -- --ignored --nocapture distinct_keys
/// ```
#[test]
#[ignore = "on request: reads 1,300,000 records in batches of 10; minutes in a debug build"]
fn a_partition_is_given_room_by_its_distinct_keys_at_any_batch_size() {
    let scratch = Scratch::new("ten");
    let path = scratch.write("dup.zip", &filler_then_ten());
    let settings = ZipInputSettings::default();
    let spill = scratch.path("spill");
    // The directory named before the read (`baseline score`) and made at the freeze
    // (`ingest`), each under the limit of the review.
    for when_needed in [false, true] {
        for batch_rows in [Some(10), Some(64), Some(1_024), None] {
            let what = format!("made at the freeze: {when_needed}, batches of {batch_rows:?}");
            let mut budget = budget(Some(42 * MIB), batch_rows);
            let mut input = ZipInput::open(&path, &settings, &budget).unwrap();
            match when_needed {
                false => input.spill_into(&spill, None).unwrap(),
                true => input.spill_when_needed(&spill, None),
            }
            let mut rows = 0u64;
            for batch in input.batches(&mut budget) {
                let batch = batch.unwrap_or_else(|error| panic!("{what}: {error}"));
                rows += batch.len() as u64;
            }
            let mut occurrences = 0u64;
            if !when_needed {
                for chunk in input.occurrence_counts() {
                    occurrences += chunk
                        .unwrap()
                        .iter()
                        .map(|count| u64::from(*count))
                        .sum::<u64>();
                }
            }
            let report = input
                .disk_dedup()
                .unwrap_or_else(|| panic!("{what}: the read stayed in memory"));
            let summary = input.finish();
            assert_eq!(
                (summary.input.records, summary.input.distinct_texts, rows),
                (1_300_000, 300_010, 300_010),
                "{what}: {report:?}"
            );
            if !when_needed {
                assert_eq!(occurrences, 1_300_000, "{what}");
            }
            assert!(!spill.exists(), "{what}: {:?}", listing(&scratch.dir));
        }
    }
}

/// Remediation, item 1: a read that names its work directory for when the set no
/// longer fits makes nothing, checks no disk and holds nothing more while the set fits.
#[test]
fn a_read_that_fits_in_memory_makes_no_directory_and_holds_nothing_more() {
    let scratch = Scratch::new("fits");
    let path = scratch.write("repeating.zip", &repeating());
    let settings = ZipInputSettings {
        max_record_bytes: 200,
        ..ZipInputSettings::default()
    };
    let reference = read_in_memory(&path, &settings, Some(100));
    let work = scratch.path("work/run");
    let mut budget = budget(None, Some(100));
    let mut input = ZipInput::open(&path, &settings, &budget).unwrap();
    let held = input.held_bytes();
    input.spill_when_needed(&work, None);
    // No free disk at all: a check would refuse the read.
    input.check_disk_at_freeze(FreezeDiskCheck {
        beside: Vec::new(),
        free_bytes: Some(0),
    });
    assert_eq!(input.held_bytes(), held, "nothing more is held");
    let mut rows = Vec::new();
    for batch in input.batches(&mut budget) {
        for record in batch.unwrap().records() {
            rows.push((record.text, record.first_position, 0));
        }
        assert!(!scratch.path("work").exists(), "a directory was made");
    }
    for row in &mut rows {
        row.2 = u32::try_from(input.occurrences(&row.0)).unwrap();
    }
    assert!(input.disk_dedup().is_none());
    assert!(input.freeze_disk().is_none());
    assert_eq!(rows, reference.rows);
    assert_eq!(input.finish(), reference.summary);
    assert_eq!(listing(&scratch.dir), ["repeating.zip"]);
}

/// Remediation, item 1: when the set no longer fits, the directory is made then, the
/// disk is checked then, the rows are those of a read in memory, and what was made
/// is gone when the read is finished.
#[test]
fn a_read_that_outgrows_the_memory_makes_its_directory_at_the_freeze() {
    let scratch = Scratch::new("freeze");
    let path = scratch.write("repeating.zip", &repeating());
    let settings = ZipInputSettings {
        max_record_bytes: 200,
        ..ZipInputSettings::default()
    };
    let reference = read_in_memory(&path, &settings, Some(100));
    let work = scratch.path("a/b/run");
    let mut budget = budget(None, Some(100));
    let mut input = ZipInput::open(&path, &settings, &budget).unwrap();
    input.spill_when_needed(&work, Some(200));
    input.check_disk_at_freeze(FreezeDiskCheck {
        beside: vec![("something else".to_string(), 7)],
        free_bytes: Some(u64::MAX),
    });
    let mut rows = Vec::new();
    let mut made = Vec::new();
    for batch in input.batches(&mut budget) {
        for record in batch.unwrap().records() {
            rows.push((record.text, record.first_position, 0));
        }
        made.push(work.exists());
    }
    assert_eq!(made.first(), Some(&false), "{made:?}");
    assert_eq!(made.last(), Some(&true), "{made:?}");
    let report = input.disk_dedup().expect("the read went on on disk");
    assert!(report.set_texts <= 200, "{report:?}");
    assert_eq!(report.set_texts + report.spilled_distinct, 467);
    let disk = input
        .freeze_disk()
        .expect("the disk was checked at the freeze");
    assert_eq!(disk.directory, work);
    assert_eq!(disk.free_bytes, u64::MAX);
    assert_eq!(
        disk.estimated_bytes,
        input.selected_bytes().dedup_disk_bytes() + 7
    );
    // The occurrence counts of every row are not kept by such a read.
    let counts = input.occurrence_counts().next().expect("one item");
    assert!(counts.is_err());
    let texts = |rows: &[(String, u64, u32)]| {
        rows.iter()
            .map(|(text, position, _)| (text.clone(), *position))
            .collect::<Vec<_>>()
    };
    assert_eq!(texts(&rows), texts(&reference.rows));
    assert_eq!(input.finish(), reference.summary);
    assert_eq!(listing(&scratch.dir), ["repeating.zip"]);
}

/// Remediation, item 1: too little free disk at the freeze is a refusal there, with
/// what was read, and nothing is left; a read with its directory named before the read
/// (`baseline score`) takes the same check at the freeze, and none while the set fits.
#[test]
fn too_little_disk_at_the_freeze_is_refused_there_and_leaves_nothing() {
    let scratch = Scratch::new("freeze-disk");
    let path = scratch.write("repeating.zip", &repeating());
    let settings = ZipInputSettings {
        max_record_bytes: 200,
        ..ZipInputSettings::default()
    };
    let check = || FreezeDiskCheck {
        beside: vec![("the result parts".to_string(), 5)],
        free_bytes: Some(1),
    };
    let refused = |input: &mut ZipInput, budget: &mut MemoryBudget| {
        let error = input
            .batches(budget)
            .find_map(Result::err)
            .expect("the read is refused at the freeze");
        assert!(matches!(error, L2dError::Refused { .. }), "{error}");
        let text = error.to_string();
        for part in [
            "for deduplication on disk",
            "5 B (5 bytes) for the result parts",
            "1 B (1 bytes) is free there",
            "stopped after ",
        ] {
            assert!(text.contains(part), "{part}: {text}");
        }
        assert!(!text.contains("nothing was read"), "{text}");
    };

    // The directory is made at the freeze.
    let work = scratch.path("a/run");
    let mut memory = budget(None, Some(100));
    let mut input = ZipInput::open(&path, &settings, &memory).unwrap();
    input.spill_when_needed(&work, Some(200));
    input.check_disk_at_freeze(check());
    refused(&mut input, &mut memory);
    let summary = input.finish();
    assert!(summary.input.records > 0 && summary.input.records < 2_011);
    assert_eq!(listing(&scratch.dir), ["repeating.zip"]);

    // The directory is named before the read: no check while the set fits ...
    let spill = scratch.path("work/dedup");
    let mut memory = budget(None, Some(100));
    let mut input = ZipInput::open(&path, &settings, &memory).unwrap();
    input.spill_into(&spill, None).unwrap();
    input.check_disk_at_freeze(check());
    for batch in input.batches(&mut memory) {
        batch.unwrap();
    }
    assert!(input.freeze_disk().is_none());
    drop(input);
    // ... and the check at the freeze when it does not.
    let mut memory = budget(None, Some(100));
    let mut input = ZipInput::open(&path, &settings, &memory).unwrap();
    input.spill_into(&spill, Some(200)).unwrap();
    input.check_disk_at_freeze(check());
    refused(&mut input, &mut memory);
    drop(input);
    assert!(!spill.exists(), "{:?}", listing(&scratch.dir));
}
