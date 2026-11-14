//! Result parts on disk and deduplication on disk (requirements 13.11 and 13.12; task
//! 1.17), on work directories and archives written into a temporary directory.

use std::io::Write;
use std::path::{Path, PathBuf};

use l2d_core::parts::{
    MANIFEST_FILE, PartTable, PartWriter, WorkDir, free_disk_bytes, read_part, write_manifest,
};
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::{
    ArchiveSummary, L2dError, LogsFrame, MemoryBudget, MemorySettings, ZipInput, ZipInputSettings,
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
