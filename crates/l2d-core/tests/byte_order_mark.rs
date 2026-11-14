//! What U+FEFF means in an input (review of task 1.16, finding B1).
//!
//! At the very start of a file it is a byte-order mark and is dropped, once. Anywhere
//! else it is text: at the start of a later line, in the middle of a line, and as a
//! second one at the start of the file. Polars drops one at the start of every buffer it
//! parses, so a record that starts with one and is the first of a batch used to read
//! differently from the validation pass, and the read failed for some batch sizes only.

use std::path::{Path, PathBuf};

use l2d_core::{
    InputAdapter, Label, LabelledAdapter, LabelledFormat, LinesAdapter, LogRecord, MemoryBudget,
    MemorySettings,
};

const GIB: u64 = 1024 * 1024 * 1024;

struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("l2d-core-bom-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn write(&self, name: &str, content: &str) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn read(adapter: &dyn InputAdapter, path: &Path, batch_rows: Option<u64>) -> Vec<LogRecord> {
    let settings = MemorySettings {
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap();
    let (logs, summary, _) = adapter
        .read_with(path, &mut budget)
        .unwrap_or_else(|error| panic!("batch size {batch_rows:?}: {error}"));
    assert!(summary.skipped.is_empty(), "{:?}", summary.skipped);
    logs.to_records()
}

fn texts(records: &[LogRecord]) -> Vec<&str> {
    records.iter().map(|record| record.text.as_str()).collect()
}

const BATCH_SIZES: [Option<u64>; 4] = [Some(1), Some(2), Some(3), None];

#[test]
fn a_line_that_starts_with_the_mark_is_text_at_any_batch_size() {
    let scratch = Scratch::new("lines");
    let path = scratch.write("marks.log", "a\n\u{feff}b\nc\n\u{feff}b\nmid\u{feff}dle\n");
    let expected = ["a", "\u{feff}b", "c", "mid\u{feff}dle"];
    let mut first = None;
    for batch_rows in BATCH_SIZES {
        let records = read(&LinesAdapter, &path, batch_rows);
        assert_eq!(texts(&records), expected, "batch size {batch_rows:?}");
        assert_eq!(records[1].occurrences, 2);
        assert_eq!(records[1].first_position, 2);
        assert_eq!(first.get_or_insert(records.clone()), &records);
    }
}

#[test]
fn only_the_first_mark_at_the_start_of_a_file_is_dropped() {
    let scratch = Scratch::new("start");
    let once = scratch.write("once.log", "\u{feff}first\nsecond\n");
    let twice = scratch.write(
        "twice.log",
        "\u{feff}\u{feff}first\nsecond\n\u{feff}first\n",
    );
    for batch_rows in BATCH_SIZES {
        assert_eq!(
            texts(&read(&LinesAdapter, &once, batch_rows)),
            ["first", "second"]
        );
        let records = read(&LinesAdapter, &twice, batch_rows);
        assert_eq!(texts(&records), ["\u{feff}first", "second"]);
        assert_eq!(records[0].occurrences, 2, "batch size {batch_rows:?}");
    }
}

#[test]
fn a_line_with_the_mark_after_a_full_default_batch_is_read() {
    let scratch = Scratch::new("default");
    let mut content = String::new();
    for index in 0..65_536 {
        content.push_str(&format!("line {}\n", index % 100));
    }
    content.push_str("\u{feff}after the batch\nlast\n");
    let path = scratch.write("full.log", &content);
    let records = read(&LinesAdapter, &path, None);
    assert_eq!(records.len(), 102);
    assert_eq!(texts(&records[100..]), ["\u{feff}after the batch", "last"]);
}

#[test]
fn a_csv_field_that_starts_with_the_mark_is_text_at_any_batch_size() {
    let scratch = Scratch::new("csv");
    // Unquoted and quoted, as the first field of a data row.
    let path = scratch.write(
        "marks.csv",
        "text,label\na,good\n\u{feff}b,bad\n\"\u{feff}c\",good\nd,\u{feff}good\n",
    );
    let adapter = LabelledAdapter::new(LabelledFormat::Csv);
    for batch_rows in BATCH_SIZES {
        let settings = MemorySettings {
            max_batch_rows: batch_rows,
            ..MemorySettings::default()
        };
        let mut budget = MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap();
        let (logs, summary, _) = adapter
            .read_with(&path, &mut budget)
            .unwrap_or_else(|error| panic!("batch size {batch_rows:?}: {error}"));
        let records = logs.to_records();
        assert_eq!(
            texts(&records),
            ["a", "\u{feff}b", "\u{feff}c"],
            "batch size {batch_rows:?}"
        );
        assert_eq!(records[1].label, Label::Bad);
        // A label with the mark in it is no label: the row is skipped, not misread.
        assert_eq!(summary.skipped.len(), 1, "{:?}", summary.skipped);
        assert_eq!(summary.skipped[0].position, 4);
    }
}

#[test]
fn a_json_line_is_read_the_same_at_any_batch_size_with_the_mark_in_it() {
    let scratch = Scratch::new("jsonl");
    // In a text the mark is text. Before the object it is not JSON: the line is skipped.
    let path = scratch.write(
        "marks.jsonl",
        "{\"text\":\"a\"}\n{\"text\":\"\u{feff}b\"}\n\u{feff}{\"text\":\"c\"}\n{\"text\":\"d\"}\n",
    );
    let mut adapter = LabelledAdapter::new(LabelledFormat::Jsonl);
    adapter.label_column = None;
    let mut first = None;
    for batch_rows in BATCH_SIZES {
        let settings = MemorySettings {
            max_batch_rows: batch_rows,
            ..MemorySettings::default()
        };
        let mut budget = MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap();
        let (logs, summary, _) = adapter
            .read_with(&path, &mut budget)
            .unwrap_or_else(|error| panic!("batch size {batch_rows:?}: {error}"));
        let records = logs.to_records();
        assert_eq!(texts(&records), ["a", "\u{feff}b", "d"], "{batch_rows:?}");
        let skipped: Vec<u64> = summary.skipped.iter().map(|skip| skip.position).collect();
        assert_eq!(skipped, [3], "batch size {batch_rows:?}");
        assert_eq!(first.get_or_insert(records.clone()), &records);
    }
}
