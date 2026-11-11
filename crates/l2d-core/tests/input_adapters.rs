//! Input adapters, exact-text deduplication and the input summary
//! (requirements 1.1, 1.2, 1.3, 1.4, 1.6).
//!
//! "Position" is the 1-based index of a record in its source: the line number for
//! line-based and JSONL input, the data-row index (header not counted) for CSV input.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use l2d_core::input::{content_sha256, row_digest, standard_record_end};
use l2d_core::polars::prelude::{Column, DataFrame, DataType};
use l2d_core::{
    Accepted, Dedup, InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat,
    LinesAdapter, LogRecord, LogsFrame,
};

/// A file in a scratch directory of its own in the system temp folder; the directory
/// is removed when the value is dropped.
struct ScratchFile {
    path: PathBuf,
}

impl std::ops::Deref for ScratchFile {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// Writes `content` to a file called `name`. Every test uses its own file names, so the
/// directory, named after the process and the file, is not shared between tests.
fn scratch_file(name: &str, content: &[u8]) -> ScratchFile {
    let dir =
        std::env::temp_dir().join(format!("l2d-core-input-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    ScratchFile { path }
}

/// The documented logs frame: column names, types and order; nulls only in `group`;
/// rows in order of first occurrence.
fn assert_logs_frame(logs: &LogsFrame) {
    let frame = logs.frame();
    let columns: Vec<(String, DataType)> = frame
        .get_columns()
        .iter()
        .map(|column| (column.name().to_string(), column.dtype().clone()))
        .collect();
    assert_eq!(
        columns,
        [
            ("text".to_string(), DataType::String),
            ("label".to_string(), DataType::String),
            ("group".to_string(), DataType::String),
            ("first_position".to_string(), DataType::UInt64),
            ("occurrences".to_string(), DataType::UInt32),
        ]
    );
    assert_eq!(*frame.schema().as_ref(), LogsFrame::schema());
    for name in ["text", "label", "first_position", "occurrences"] {
        assert_eq!(frame.column(name).unwrap().null_count(), 0, "{name}");
    }
    let positions: Vec<u64> = frame
        .column("first_position")
        .unwrap()
        .u64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "rows are in order of first occurrence"
    );
    let labels = frame.column("label").unwrap().str().unwrap();
    assert!(
        labels
            .into_no_null_iter()
            .all(|label| matches!(label, "good" | "bad" | "unknown"))
    );
    assert_eq!(logs.len(), frame.height());
}

/// Reads through the frame interface and returns the row views, after checking that
/// the frame is a documented logs frame and that the row views carry the same data.
trait ReadRecords {
    fn read_records(&self, path: &Path) -> (Vec<LogRecord>, InputSummary);
}

impl<A: InputAdapter> ReadRecords for A {
    fn read_records(&self, path: &Path) -> (Vec<LogRecord>, InputSummary) {
        let (logs, summary) = self.read(path).unwrap();
        assert_logs_frame(&logs);
        let records = logs.to_records();
        assert_eq!(records.len(), logs.len());
        assert_eq!(LogsFrame::from_records(&records).unwrap(), logs);
        (records, summary)
    }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv")
}

fn csv() -> LabelledAdapter {
    LabelledAdapter::new(LabelledFormat::Csv)
}

fn jsonl() -> LabelledAdapter {
    LabelledAdapter::new(LabelledFormat::Jsonl)
}

fn record<'a>(records: &'a [LogRecord], text: &str) -> &'a LogRecord {
    records
        .iter()
        .find(|r| r.text == text)
        .unwrap_or_else(|| panic!("no record with text {text:?}"))
}

/// Design postconditions: every text is unique; `records` = occurrences + skipped.
fn assert_postconditions(records: &[LogRecord], summary: &InputSummary) {
    let texts: HashSet<&str> = records.iter().map(|r| r.text.as_str()).collect();
    assert_eq!(texts.len(), records.len(), "every returned text is unique");
    assert_eq!(summary.distinct_texts, records.len() as u64);
    let occurrences: u64 = records.iter().map(|r| u64::from(r.occurrences)).sum();
    assert_eq!(
        summary.records,
        occurrences + summary.skipped.len() as u64,
        "records = sum of occurrences + skipped"
    );
}

// --- Fixture: dedup, malformed row, summary counts (1.2, 1.3, 1.4, 1.6) -------------------

#[test]
fn fixture_is_deduplicated_and_summarized() {
    let path = fixture_path();
    let before = std::fs::read(&path).unwrap();
    let (records, summary) = csv().with_group_column("group").read_records(&path);

    assert_eq!(summary.source_path, path);
    assert_eq!(summary.records, 200);
    assert_eq!(summary.distinct_texts, 169);
    assert_eq!(records.len(), 169);
    assert_eq!(summary.label_conflicts, 1);
    assert_postconditions(&records, &summary);

    let occurrences: u64 = records.iter().map(|r| u64::from(r.occurrences)).sum();
    assert_eq!(occurrences, 199);

    let count = |label: Label| records.iter().filter(|r| r.label == label).count();
    assert_eq!(count(Label::Good), 96);
    assert_eq!(count(Label::Bad), 72);
    assert_eq!(count(Label::Unknown), 1);
    assert_eq!(records.iter().filter(|r| r.occurrences > 1).count(), 27);

    let groups: HashSet<&str> = records.iter().filter_map(|r| r.group.as_deref()).collect();
    assert_eq!(groups.len(), 8);
    assert!(records.iter().all(|r| r.group.is_some()));

    // Source order is kept: first positions strictly increase.
    assert!(
        records
            .windows(2)
            .all(|w| w[0].first_position < w[1].first_position)
    );
    assert_eq!(records[0].first_position, 1);

    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "the source is never modified"
    );
}

#[test]
fn fixture_row_with_wrong_field_count_is_skipped_at_position_121() {
    let (records, summary) = csv()
        .with_group_column("group")
        .read_records(&fixture_path());
    assert_eq!(summary.skipped.len(), 1);
    let skipped = &summary.skipped[0];
    assert_eq!(skipped.position, 121);
    assert!(skipped.reason.contains("3 fields"), "{}", skipped.reason);
    assert!(skipped.reason.contains("found 2"), "{}", skipped.reason);
    assert!(skipped.reason.contains("line 122"), "{}", skipped.reason);
    assert!(
        records.iter().all(
            |r| r.text != "GET http://localhost:8080/tienda1/publico/caracteristicas.jsp?id=99"
        ),
        "the malformed row must not become a record"
    );
}

#[test]
fn fixture_read_is_repeatable() {
    let adapter = csv().with_group_column("group");
    let first = adapter.read(&fixture_path()).unwrap();
    let second = adapter.read(&fixture_path()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.1.content_sha256.len(), 64);
}

// --- Content hash (1.6) --------------------------------------------------------------------

#[test]
fn content_hash_is_the_sha256_of_the_file_bytes() {
    let path = scratch_file("abc.log", b"abc");
    let (records, summary) = LinesAdapter.read_records(&path);
    assert_eq!(
        summary.content_sha256,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(summary.records, 1);
    assert_eq!(records[0].text, "abc");
}

// --- Plain line-based logs (1.1, 1.3, 1.4) -------------------------------------------------

#[test]
fn lines_are_read_with_positions_and_deduplicated() {
    let hostile = "GET /a?x=..\\..\\windows\\system32 & del /q * ; $(rm -rf /)";
    let content = format!("alpha\nbeta\r\nalpha\n\n{hostile}\nalpha\n");
    let path = scratch_file("plain.log", content.as_bytes());
    let (records, summary) = LinesAdapter.read_records(&path);

    assert_eq!(summary.source_path, *path);
    assert_eq!(summary.records, 6);
    assert_eq!(summary.distinct_texts, 3);
    assert_eq!(summary.label_conflicts, 0);
    assert_postconditions(&records, &summary);

    let texts: Vec<&str> = records.iter().map(|r| r.text.as_str()).collect();
    assert_eq!(texts, ["alpha", "beta", hostile]);
    assert_eq!(records[0].occurrences, 3);
    assert_eq!(records[0].first_position, 1);
    assert_eq!(records[1].occurrences, 1);
    assert_eq!(records[1].first_position, 2);
    assert_eq!(records[2].first_position, 5);
    assert!(
        records
            .iter()
            .all(|r| r.label == Label::Unknown && r.group.is_none())
    );

    assert_eq!(summary.skipped.len(), 1);
    assert_eq!(summary.skipped[0].position, 4);
    assert!(summary.skipped[0].reason.contains("empty"));
}

#[test]
fn a_line_that_is_not_utf8_is_skipped_and_reading_continues() {
    let path = scratch_file("bad-utf8.log", b"first\n\xff\xfe broken\nlast");
    let (records, summary) = LinesAdapter.read_records(&path);
    assert_eq!(summary.records, 3);
    assert_eq!(summary.skipped.len(), 1);
    assert_eq!(summary.skipped[0].position, 2);
    assert!(summary.skipped[0].reason.contains("UTF-8"));
    let texts: Vec<&str> = records.iter().map(|r| r.text.as_str()).collect();
    assert_eq!(texts, ["first", "last"]);
    assert_eq!(records[1].first_position, 3);
    assert_postconditions(&records, &summary);
}

#[test]
fn an_empty_file_gives_no_records() {
    let path = scratch_file("empty.log", b"");
    let (records, summary) = LinesAdapter.read_records(&path);
    assert!(records.is_empty());
    assert_eq!(summary.records, 0);
    assert_eq!(summary.distinct_texts, 0);
    assert!(summary.skipped.is_empty());
    assert_eq!(
        summary.content_sha256,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

// --- Labelled CSV with configured columns (1.2, 1.3, 1.4) ----------------------------------

#[test]
fn csv_columns_are_configurable_and_bad_rows_are_skipped() {
    let content = concat!(
        "id,message,verdict,route\n",
        "1,GET /a,good,GET /a\n",
        "2,\"GET /b?q=1,2\",BAD,GET /b\n",
        "3,\"two\nlines\", good ,GET /c\n",
        "4,GET /d,maybe,GET /d\n",
        "5,GET /e,,\n",
        "6,,good,GET /f\n",
        "7,GET /a,good,GET /other\n",
        "8,GET /g,good,GET /g,extra\n",
    );
    let path = scratch_file("custom.csv", content.as_bytes());
    let adapter = csv()
        .with_text_column("message")
        .with_label_column("verdict")
        .with_group_column("route");
    let (records, summary) = adapter.read_records(&path);

    assert_eq!(summary.records, 8);
    assert_postconditions(&records, &summary);

    let a = record(&records, "GET /a");
    assert_eq!(
        (a.label, a.occurrences, a.first_position),
        (Label::Good, 2, 1)
    );
    assert_eq!(a.group.as_deref(), Some("GET /a"), "first group is kept");

    let b = record(&records, "GET /b?q=1,2");
    assert_eq!(b.label, Label::Bad, "labels are matched without case");
    assert_eq!(b.first_position, 2);

    let c = record(&records, "two\nlines");
    assert_eq!(c.label, Label::Good, "labels are trimmed");
    assert_eq!(c.first_position, 3);

    let e = record(&records, "GET /e");
    assert_eq!(e.label, Label::Unknown, "an empty label is unlabelled");
    assert_eq!(e.group, None, "an empty group is no group");
    assert_eq!(e.first_position, 5);
    assert_eq!(summary.label_conflicts, 0);

    let positions: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(positions, [4, 6, 8]);
    assert!(summary.skipped[0].reason.contains("label"));
    assert!(summary.skipped[1].reason.contains("empty"));
    assert!(summary.skipped[2].reason.contains("found 5"));
    assert_eq!(records.len(), 4);
}

#[test]
fn csv_without_label_column_reads_unlabelled_text() {
    let path = scratch_file("unlabelled.csv", b"text\nGET /a\nGET /a\nGET /b\n");
    let mut adapter = csv();
    adapter.label_column = None;
    let (records, summary) = adapter.read_records(&path);
    assert_eq!(summary.records, 3);
    assert_eq!(records.len(), 2);
    assert!(records.iter().all(|r| r.label == Label::Unknown));
}

#[test]
fn csv_with_a_byte_order_mark_still_finds_its_columns() {
    let path = scratch_file("bom.csv", b"\xef\xbb\xbftext,label\nGET /a,bad\n");
    let (records, summary) = csv().read_records(&path);
    assert_eq!(summary.records, 1);
    assert_eq!(records[0].label, Label::Bad);
}

// --- Conflicting labels (1.3, design "Input adapters and dedup") ---------------------------

#[test]
fn conflicting_labels_become_unknown_and_are_counted() {
    let content = concat!(
        "text,label,group\n",
        "GET /x,good,g1\n",
        "GET /y,bad,g2\n",
        "GET /x,bad,g9\n",
        "GET /x,good,g1\n",
        "GET /z,good,g3\n",
        "GET /z,,g3\n",
        "GET /z,unknown,g3\n",
    );
    let path = scratch_file("conflict.csv", content.as_bytes());
    let (records, summary) = csv().with_group_column("group").read_records(&path);

    assert_eq!(summary.label_conflicts, 1);
    let x = record(&records, "GET /x");
    assert_eq!(x.label, Label::Unknown);
    assert_eq!(x.occurrences, 3);
    assert_eq!(x.first_position, 1);
    assert_eq!(x.group.as_deref(), Some("g1"));

    assert_eq!(record(&records, "GET /y").label, Label::Bad);
    let z = record(&records, "GET /z");
    assert_eq!(
        z.label,
        Label::Good,
        "an unlabelled occurrence does not conflict"
    );
    assert_eq!(z.occurrences, 3);
    assert_postconditions(&records, &summary);
}

// --- Labelled JSONL (1.2, 1.4) -------------------------------------------------------------

#[test]
fn jsonl_records_are_read_and_unreadable_lines_are_skipped() {
    let content = concat!(
        "{\"text\":\"GET /a\",\"label\":\"good\",\"group\":\"GET /a\"}\n",
        "{\"text\":\"GET /b\",\"label\":\"bad\",\"group\":\"GET /b\"}\r\n",
        "{not json}\n",
        "{\"label\":\"good\"}\n",
        "[\"GET /c\",\"good\"]\n",
        "{\"text\":\"GET /a\",\"label\":\"good\",\"group\":\"GET /a\"}\n",
        "{\"text\":\"GET /d\",\"label\":null}\n",
        "{\"text\":42,\"label\":\"good\"}\n",
        "{\"text\":\"GET /e\",\"label\":\"evil\"}\n",
    );
    let path = scratch_file("data.jsonl", content.as_bytes());
    let (records, summary) = jsonl().with_group_column("group").read_records(&path);

    assert_eq!(summary.records, 9);
    assert_postconditions(&records, &summary);
    let texts: Vec<&str> = records.iter().map(|r| r.text.as_str()).collect();
    assert_eq!(texts, ["GET /a", "GET /b", "GET /d"]);

    let a = record(&records, "GET /a");
    assert_eq!(
        (a.label, a.occurrences, a.first_position),
        (Label::Good, 2, 1)
    );
    assert_eq!(a.group.as_deref(), Some("GET /a"));
    assert_eq!(record(&records, "GET /b").label, Label::Bad);
    let d = record(&records, "GET /d");
    assert_eq!(
        (d.label, d.group.clone(), d.first_position),
        (Label::Unknown, None, 7)
    );

    let positions: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(positions, [3, 4, 5, 8, 9]);
}

#[test]
fn jsonl_columns_are_configurable() {
    let content = "{\"msg\":\"GET /a\",\"verdict\":\"bad\",\"route\":\"r1\"}\n";
    let path = scratch_file("custom.jsonl", content.as_bytes());
    let adapter = jsonl()
        .with_text_column("msg")
        .with_label_column("verdict")
        .with_group_column("route");
    let (records, _) = adapter.read_records(&path);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].text, "GET /a");
    assert_eq!(records[0].label, Label::Bad);
    assert_eq!(records[0].group.as_deref(), Some("r1"));
}

// --- Edge validation: missing path, unknown columns ----------------------------------------

#[test]
fn a_missing_input_path_names_the_item_and_expected_location() {
    let path = std::env::temp_dir()
        .join(format!("l2d-core-input-test-{}", std::process::id()))
        .join("does-not-exist.csv");
    let adapters: Vec<Box<dyn InputAdapter>> =
        vec![Box::new(LinesAdapter), Box::new(csv()), Box::new(jsonl())];
    for adapter in &adapters {
        match adapter.read(&path) {
            Err(L2dError::Missing { item, expected_at }) => {
                assert!(item.contains("input file"), "{item}");
                assert_eq!(expected_at, path);
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }
    assert!(!path.exists(), "reading must not create the file");
}

#[test]
fn an_unknown_csv_column_is_invalid_input() {
    let path = scratch_file("columns.csv", b"text,label,group\nGET /a,good,g\n");
    let cases = [
        (csv().with_text_column("message"), "message"),
        (csv().with_label_column("verdict"), "verdict"),
        (csv().with_group_column("route"), "route"),
    ];
    for (adapter, column) in cases {
        match adapter.read(&path) {
            Err(L2dError::InvalidInput { detail }) => {
                assert!(detail.contains(column), "{detail}");
                assert!(detail.contains("column"), "{detail}");
            }
            other => panic!("expected InvalidInput for {column}, got {other:?}"),
        }
    }
}

#[test]
fn a_csv_without_a_header_is_invalid_input() {
    let path = scratch_file("no-header.csv", b"");
    assert!(matches!(
        csv().read(&path),
        Err(L2dError::InvalidInput { .. })
    ));
}

#[test]
fn a_jsonl_column_absent_from_every_record_is_invalid_input() {
    let content =
        "{\"text\":\"GET /a\",\"label\":\"good\"}\n{\"text\":\"GET /b\",\"label\":\"bad\"}\n";
    let path = scratch_file("columns.jsonl", content.as_bytes());
    let cases = [
        (jsonl().with_text_column("message"), "message"),
        (jsonl().with_label_column("verdict"), "verdict"),
        (jsonl().with_group_column("route"), "route"),
    ];
    for (adapter, column) in cases {
        match adapter.read(&path) {
            Err(L2dError::InvalidInput { detail }) => {
                assert!(detail.contains(column), "{detail}");
            }
            other => panic!("expected InvalidInput for {column}, got {other:?}"),
        }
    }
}

#[test]
fn the_labelled_format_follows_the_file_extension() {
    assert_eq!(
        LabelledFormat::from_path(Path::new("data/x.csv")).unwrap(),
        LabelledFormat::Csv
    );
    assert_eq!(
        LabelledFormat::from_path(Path::new("data/x.JSONL")).unwrap(),
        LabelledFormat::Jsonl
    );
    assert_eq!(
        LabelledFormat::from_path(Path::new("data/x.ndjson")).unwrap(),
        LabelledFormat::Jsonl
    );
    assert!(matches!(
        LabelledFormat::from_path(Path::new("data/x.txt")),
        Err(L2dError::InvalidInput { .. })
    ));
}

// --- The logs frame: schema, row order, row views (design "Data frames") -----------------

/// What a plain row-by-row pass over a labelled CSV gives: the `csv` crate for the
/// fields and a hash map for the deduplication, with the rules of the adapter
/// (labels without case, empty label or group = none, wrong field count or empty text
/// skipped). Independent of Polars and of the adapter code. Returns the records in
/// order of first occurrence and the skipped positions.
fn reference_csv(bytes: &[u8], text: &str, label: &str, group: &str) -> (Vec<LogRecord>, Vec<u64>) {
    let content = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(content);
    let headers = reader.headers().unwrap().clone();
    let at = |name: &str| headers.iter().position(|header| header == name).unwrap();
    let (text_at, label_at, group_at) = (at(text), at(label), at(group));

    let mut index: HashMap<String, usize> = HashMap::new();
    let mut seen: Vec<(bool, bool)> = Vec::new();
    let mut records: Vec<LogRecord> = Vec::new();
    let mut skipped = Vec::new();
    for (row_index, row) in reader.records().enumerate() {
        let position = row_index as u64 + 1;
        let row = row.unwrap();
        let label = row.get(label_at).map(|raw| raw.trim().to_ascii_lowercase());
        let label = match label.as_deref() {
            Some("good") => Some(Label::Good),
            Some("bad") => Some(Label::Bad),
            Some("" | "unknown") => Some(Label::Unknown),
            _ => None,
        };
        let (true, Some(label)) = (
            row.len() == headers.len() && !row[text_at].is_empty(),
            label,
        ) else {
            skipped.push(position);
            continue;
        };
        let at = *index.entry(row[text_at].to_string()).or_insert_with(|| {
            records.push(LogRecord {
                text: row[text_at].to_string(),
                label: Label::Unknown,
                group: Some(row[group_at].to_string()).filter(|group| !group.is_empty()),
                first_position: position,
                occurrences: 0,
            });
            seen.push((false, false));
            records.len() - 1
        });
        records[at].occurrences += 1;
        seen[at].0 |= label == Label::Good;
        seen[at].1 |= label == Label::Bad;
    }
    for (record, seen) in records.iter_mut().zip(seen) {
        record.label = match seen {
            (true, false) => Label::Good,
            (false, true) => Label::Bad,
            _ => Label::Unknown,
        };
    }
    (records, skipped)
}

#[test]
fn fixture_frame_has_the_documented_schema_and_first_occurrence_order() {
    let path = fixture_path();
    let (logs, summary) = csv().with_group_column("group").read(&path).unwrap();
    assert_logs_frame(&logs);

    let frame = logs.frame();
    assert_eq!(frame.shape(), (169, 5));
    assert_eq!(frame.column("group").unwrap().null_count(), 0);
    assert_eq!(summary.distinct_texts, 169);

    // Counts taken from the frame itself, not from the row views.
    let occurrences = frame.column("occurrences").unwrap().u32().unwrap();
    assert_eq!(
        occurrences.into_no_null_iter().map(u64::from).sum::<u64>(),
        199
    );
    let labels = frame.column("label").unwrap().str().unwrap();
    let count = |name: &str| labels.into_no_null_iter().filter(|l| *l == name).count();
    assert_eq!((count("good"), count("bad"), count("unknown")), (96, 72, 1));
    assert_eq!(
        frame
            .column("first_position")
            .unwrap()
            .u64()
            .unwrap()
            .get(0),
        Some(1)
    );

    // Row for row what a plain pass without Polars gives, in the same order.
    let (reference, skipped) =
        reference_csv(&std::fs::read(&path).unwrap(), "text", "label", "group");
    assert_eq!(logs.to_records(), reference);
    assert_eq!(skipped, [121]);
}

#[test]
fn row_views_and_frame_convert_into_each_other() {
    let records = vec![
        LogRecord {
            text: "GET /a".into(),
            label: Label::Good,
            group: Some("/a".into()),
            first_position: 1,
            occurrences: 2,
        },
        LogRecord {
            text: "two\nlines".into(),
            label: Label::Unknown,
            group: None,
            first_position: 3,
            occurrences: 1,
        },
        LogRecord {
            text: "GET /b".into(),
            label: Label::Bad,
            group: Some("/b".into()),
            first_position: 7,
            occurrences: u32::MAX,
        },
    ];
    let logs = LogsFrame::from_records(&records).unwrap();
    assert_logs_frame(&logs);
    assert_eq!(logs.len(), 3);
    assert!(!logs.is_empty());
    assert_eq!(logs.to_records(), records);
    assert_eq!(logs.records().nth(1).unwrap(), records[1]);
    assert_eq!(logs.frame().column("group").unwrap().null_count(), 1);

    // The same rows in another order are another frame.
    let mut reordered = records.clone();
    reordered.swap(0, 2);
    assert_ne!(LogsFrame::from_records(&reordered).unwrap(), logs);

    let empty = LogsFrame::empty();
    assert_logs_frame(&empty);
    assert!(empty.is_empty());
    assert!(empty.to_records().is_empty());
    assert_eq!(LogsFrame::from_records(&[]).unwrap(), empty);
    assert_eq!(LogsFrame::new(logs.clone().into_frame()).unwrap(), logs);
}

#[test]
fn a_frame_with_another_schema_or_content_is_not_a_logs_frame() {
    let good = LogsFrame::from_records(&[LogRecord {
        text: "GET /a".into(),
        label: Label::Good,
        group: Some("/a".into()),
        first_position: 1,
        occurrences: 1,
    }])
    .unwrap()
    .into_frame();
    let replaced = |column: Column| {
        let mut frame = good.clone();
        frame.with_column(column).unwrap();
        frame
    };
    let cases: Vec<(&str, DataFrame)> = vec![
        ("a missing column", good.drop("occurrences").unwrap()),
        (
            "another column order",
            good.select(["label", "text", "group", "first_position", "occurrences"])
                .unwrap(),
        ),
        ("another column name", {
            let mut frame = good.clone();
            frame.rename("group", "endpoint".into()).unwrap();
            frame
        }),
        (
            "another type",
            replaced(Column::new("occurrences".into(), vec![1u64])),
        ),
        (
            "an extra column",
            replaced(Column::new("score".into(), vec![0.5f64])),
        ),
        (
            "a label outside good, bad and unknown",
            replaced(Column::new("label".into(), vec!["Good"])),
        ),
        (
            "a null text",
            replaced(Column::new("text".into(), vec![None::<&str>])),
        ),
        (
            "a null label",
            replaced(Column::new("label".into(), vec![None::<&str>])),
        ),
    ];
    for (what, frame) in cases {
        match LogsFrame::new(frame) {
            Err(L2dError::Runtime { detail }) => {
                assert!(detail.contains("not a logs frame"), "{what}: {detail}")
            }
            other => panic!("{what}: expected a runtime failure, got {other:?}"),
        }
    }
    // A null group is allowed.
    let no_group = replaced(Column::new("group".into(), vec![None::<&str>]));
    assert_eq!(
        LogsFrame::new(no_group).unwrap().to_records()[0].group,
        None
    );
}

// --- Agreement between the validation pass and Polars (1.4: no silent drop) --------------

type Row = (u64, &'static str, Label, Option<&'static str>);

/// A batch as an adapter builds it: what the validation pass accepted, and the columns
/// taken from the parse.
fn batch(rows: &[Row]) -> (Accepted, Column, Column, Column) {
    let mut accepted = Accepted::new();
    for (position, text, label, group) in rows {
        accepted.record(*position, row_digest(text, *label, *group), *label);
    }
    let texts: Vec<&str> = rows.iter().map(|row| row.1).collect();
    let labels: Vec<&str> = rows.iter().map(|row| row.2.as_str()).collect();
    let groups: Vec<Option<&str>> = rows.iter().map(|row| row.3).collect();
    (
        accepted,
        Column::new("text".into(), texts),
        Column::new("label".into(), labels),
        Column::new("group".into(), groups),
    )
}

fn strings(values: &[Option<&str>]) -> Column {
    Column::new("parsed".into(), values.to_vec())
}

#[test]
fn a_parse_that_disagrees_with_the_validation_pass_fails_instead_of_returning_a_frame() {
    let rows: [Row; 3] = [
        (1, "GET /a", Label::Good, Some("g1")),
        (2, "GET /b", Label::Bad, None),
        (4, "GET /c", Label::Unknown, Some("g3")),
    ];
    let text = |values: [&'static str; 3]| strings(&values.map(Some));
    let (_, texts, labels, groups) = batch(&rows);

    // (what Polars is made to return, what the failure must mention)
    let cases: Vec<(&str, Column, Column, Column, &str)> = vec![
        (
            "a row was dropped",
            strings(&[Some("GET /a"), Some("GET /c")]),
            strings(&[Some("good"), Some("unknown")]),
            strings(&[Some("g1"), Some("g3")]),
            "2 text rows, the validation pass accepted 3",
        ),
        (
            "a row was added",
            strings(&[
                Some("GET /a"),
                Some("GET /b"),
                Some("GET /b2"),
                Some("GET /c"),
            ]),
            strings(&[Some("good"), Some("bad"), Some("bad"), Some("unknown")]),
            strings(&[Some("g1"), None, None, Some("g3")]),
            "4 text rows, the validation pass accepted 3",
        ),
        (
            "a field boundary moved",
            text(["GET /a", "GET /b,bad", "GET /c"]),
            labels.clone(),
            groups.clone(),
            "position 2",
        ),
        (
            "a text of the same length differs",
            text(["GET /a", "GET /b", "GET /d"]),
            labels.clone(),
            groups.clone(),
            "position 4",
        ),
        (
            "a text is empty",
            strings(&[Some("GET /a"), None, Some("GET /c")]),
            labels.clone(),
            groups.clone(),
            "position 2",
        ),
        (
            "a label differs",
            texts.clone(),
            text(["bad", "bad", "unknown"]),
            groups.clone(),
            "position 1",
        ),
        (
            "a label is not one of the three names",
            texts.clone(),
            text(["good", "bad", "maybe"]),
            groups.clone(),
            "position 4",
        ),
        (
            "a group differs",
            texts.clone(),
            labels.clone(),
            strings(&[Some("g1"), Some(""), Some("g3")]),
            "position 2",
        ),
        (
            "two rows changed places",
            text(["GET /b", "GET /a", "GET /c"]),
            text(["bad", "good", "unknown"]),
            strings(&[None, Some("g1"), Some("g3")]),
            "position 1",
        ),
    ];
    for (what, texts, labels, groups, mention) in cases {
        let (accepted, _, _, _) = batch(&rows);
        let mut dedup = Dedup::new();
        match dedup.push(accepted, texts, labels, groups) {
            Err(L2dError::Runtime { detail }) => {
                assert!(detail.contains("disagree"), "{what}: {detail}");
                assert!(detail.contains(mention), "{what}: {detail}");
            }
            other => panic!("{what}: expected a runtime failure, got {other:?}"),
        }
        // Nothing of the refused batch was kept.
        let (logs, summary) = dedup.finish(Path::new("refused"), b"").unwrap();
        assert!(logs.is_empty(), "{what}");
        assert_eq!(summary.records, 0, "{what}");
    }

    // The same batch with the columns the validation pass expects is accepted.
    let (accepted, texts, labels, groups) = batch(&rows);
    let mut dedup = Dedup::new();
    dedup.push(accepted, texts, labels, groups).unwrap();
    assert_eq!(dedup.label_counts(), (1, 1));
    let (logs, summary) = dedup.finish(Path::new("agreed"), b"").unwrap();
    assert_eq!(logs.len(), 3);
    assert_eq!(summary.records, 3);
}

#[test]
fn batches_pushed_in_source_order_give_the_same_frame_as_one_batch() {
    let rows: [Row; 8] = [
        (1, "GET /x", Label::Good, Some("g1")),
        (2, "GET /y", Label::Bad, None),
        (4, "GET /x", Label::Bad, Some("g9")),
        (5, "GET /z", Label::Unknown, Some("g3")),
        (6, "GET /y", Label::Bad, Some("late")),
        (8, "GET /z", Label::Good, None),
        (9, "GET /w", Label::Unknown, None),
        (10, "GET /x", Label::Unknown, Some("g1")),
    ];
    let read = |sizes: &[usize]| {
        let mut dedup = Dedup::new();
        dedup.skip(3, "unreadable");
        dedup.skip(7, "unreadable");
        let mut rest = &rows[..];
        for size in sizes {
            let (now, later) = rest.split_at(*size);
            let (accepted, texts, labels, groups) = batch(now);
            dedup.push(accepted, texts, labels, groups).unwrap();
            rest = later;
        }
        assert!(rest.is_empty());
        assert_eq!(dedup.label_counts(), (2, 3));
        dedup.finish(Path::new("batches"), b"content").unwrap()
    };

    let (logs, summary) = read(&[8]);
    assert_logs_frame(&logs);
    let expected =
        |text: &str, label, group: Option<&str>, first_position, occurrences| LogRecord {
            text: text.into(),
            label,
            group: group.map(str::to_string),
            first_position,
            occurrences,
        };
    assert_eq!(
        logs.to_records(),
        [
            // Good and bad: a conflict. The first group key is kept.
            expected("GET /x", Label::Unknown, Some("g1"), 1, 3),
            // The first occurrence had no group; a later one does not add it.
            expected("GET /y", Label::Bad, None, 2, 2),
            // An unlabelled occurrence does not conflict with a labelled one.
            expected("GET /z", Label::Good, Some("g3"), 5, 2),
            expected("GET /w", Label::Unknown, None, 9, 1),
        ]
    );
    assert_eq!(summary.records, 10);
    assert_eq!(summary.distinct_texts, 4);
    assert_eq!(summary.label_conflicts, 1);
    assert_eq!(summary.skipped.len(), 2);
    assert_eq!(summary.content_sha256, content_sha256(b"content"));

    let splits: [&[usize]; 7] = [
        &[1, 7],
        &[2, 6],
        &[3, 5],
        &[4, 4],
        &[7, 1],
        &[3, 3, 2],
        &[1, 1, 1, 1, 1, 1, 1, 1],
    ];
    for sizes in splits {
        assert_eq!(
            read(sizes),
            (logs.clone(), summary.clone()),
            "batches of {sizes:?}"
        );
    }
}

// --- Record boundaries: the validation pass and Polars must split alike -------------------

fn byte_record(fields: &[&str]) -> csv::ByteRecord {
    csv::ByteRecord::from(fields.to_vec())
}

#[test]
fn only_records_in_the_standard_form_are_passed_to_polars_as_they_are() {
    // Standard: the end of the fields is returned, the line end is left out.
    let standard: [(&[u8], &[&str]); 6] = [
        (b"a,b,c\n", &["a", "b", "c"]),
        (b"a,b,c\r\n", &["a", "b", "c"]),
        (b"a,b,c", &["a", "b", "c"]),
        (b"a,,\r", &["a", "", ""]),
        (
            b"\"a,1\",\"say \"\"hi\"\"\",\"two\r\nlines\"\n",
            &["a,1", "say \"hi\"", "two\r\nlines"],
        ),
        (b"\"\", padded ,'x'\n", &["", " padded ", "'x'"]),
    ];
    for (raw, fields) in standard {
        let end = raw
            .iter()
            .rposition(|byte| !matches!(byte, b'\r' | b'\n'))
            .map_or(0, |last| last + 1);
        assert_eq!(
            standard_record_end(raw, &byte_record(fields)),
            Some(end),
            "{:?}",
            String::from_utf8_lossy(raw)
        );
    }

    // Not standard: readers differ on these, so the record is rewritten.
    let other: [(&[u8], &[&str]); 7] = [
        // A quote inside an unquoted field.
        (b"a\"b,c\n", &["a\"b", "c"]),
        (b"say \"hi\",c\n", &["say \"hi\"", "c"]),
        // Text after the closing quote.
        (b"\"a\"x,c\n", &["ax", "c"]),
        // A quote that is never closed.
        (b"a,\"b\nc\n", &["a", "b\nc\n"]),
        // A carriage return inside an unquoted field.
        (b"a\rb,c\n", &["a\rb", "c"]),
        // The fields are not the ones in the raw bytes.
        (b"a,b\n", &["a", "c"]),
        (b"a,b\n", &["a", "b", "c"]),
    ];
    for (raw, fields) in other {
        assert_eq!(
            standard_record_end(raw, &byte_record(fields)),
            None,
            "{:?}",
            String::from_utf8_lossy(raw)
        );
    }
}

#[test]
fn csv_quotes_commas_line_breaks_and_line_ends_are_read_exactly() {
    let content = concat!(
        "\u{feff}text,label,group\r\n",
        // 1: a quoted comma, CRLF line end.
        "\"GET /a?q=1,2\",good,g1\r\n",
        // 2: doubled quotes and a comma, in the text and in the group.
        "\"say \"\"hi\"\", twice\",bad,\"g,\"\"2\"\"\"\r\n",
        // 3: a line feed inside a quoted field.
        "\"line one\nline two\",good,g3\n",
        // 4: a carriage return and line feed inside a quoted field.
        "\"line one\r\nline two\",bad,g4\r\n",
        // 5: quotes inside an unquoted field (not the standard form).
        "GET /b \"quoted\" tail,good,g5\n",
        // 6: surrounding spaces are part of the text and of the group.
        "  padded  ,GOOD, g6 \n",
        // 7: an empty quoted text.
        "\"\",good,g7\n",
        // 8: text after the closing quote (not the standard form).
        "\"GET /c\"x,bad,g8\n",
        // 9: single quotes are ordinary characters; an empty quoted group is no group.
        "'single',bad,\"\"\n",
        // 10: too few fields.
        "GET /d,good\n",
        // 11: the text of row 1 again, written without quotes around the label.
        "\"GET /a?q=1,2\",\"good\",other\n",
        // 12: not ASCII.
        "ünïcödé ☃ 😀,bad,gü\n",
        // 13: a quoted field holding only a quote, and a tab.
        "\"\"\"\",good,\"\t\"\n",
        // 14: an odd number of quotes inside an unquoted field (not the standard form).
        "5\" disk,bad,g14\n",
        // 15: the row after it is still its own record.
        "GET /e,good,g15\r\n",
    );
    let path = scratch_file("adversarial.csv", content.as_bytes());
    let (records, summary) = csv().with_group_column("group").read_records(&path);

    let expected: Vec<(&str, Label, Option<&str>, u64, u32)> = vec![
        ("GET /a?q=1,2", Label::Good, Some("g1"), 1, 2),
        ("say \"hi\", twice", Label::Bad, Some("g,\"2\""), 2, 1),
        ("line one\nline two", Label::Good, Some("g3"), 3, 1),
        ("line one\r\nline two", Label::Bad, Some("g4"), 4, 1),
        ("GET /b \"quoted\" tail", Label::Good, Some("g5"), 5, 1),
        ("  padded  ", Label::Good, Some(" g6 "), 6, 1),
        ("GET /cx", Label::Bad, Some("g8"), 8, 1),
        ("'single'", Label::Bad, None, 9, 1),
        ("ünïcödé ☃ 😀", Label::Bad, Some("gü"), 12, 1),
        ("\"", Label::Good, Some("\t"), 13, 1),
        ("5\" disk", Label::Bad, Some("g14"), 14, 1),
        ("GET /e", Label::Good, Some("g15"), 15, 1),
    ];
    let found: Vec<(&str, Label, Option<&str>, u64, u32)> = records
        .iter()
        .map(|r| {
            (
                r.text.as_str(),
                r.label,
                r.group.as_deref(),
                r.first_position,
                r.occurrences,
            )
        })
        .collect();
    assert_eq!(found, expected);

    assert_eq!(summary.records, 15);
    let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(skipped, [7, 10]);
    assert!(summary.skipped[0].reason.contains("empty text"));
    assert!(summary.skipped[1].reason.contains("found 2"));
    assert_postconditions(&records, &summary);

    // The same as a plain pass without Polars.
    let (reference, reference_skipped) =
        reference_csv(content.as_bytes(), "text", "label", "group");
    assert_eq!(records, reference);
    assert_eq!(skipped, reference_skipped);

    // The hash covers the raw bytes, byte-order mark included.
    assert_eq!(summary.content_sha256, content_sha256(content.as_bytes()));
    assert_ne!(
        summary.content_sha256,
        content_sha256(&content.as_bytes()[3..])
    );
}

#[test]
fn an_unbalanced_quote_absorbs_the_following_rows_into_one_record() {
    // The quote opens the first field: everything up to the end of the file is one
    // field, so the record has the wrong field count and is skipped as a whole.
    let content = "text,label\nGET /a,good\n\"GET /b,bad\nGET /c,good\nGET /d,bad\n";
    let path = scratch_file("unbalanced-first.csv", content.as_bytes());
    let (records, summary) = csv().read_records(&path);
    assert_eq!(summary.records, 2);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].text, "GET /a");
    assert_eq!(summary.skipped.len(), 1);
    assert_eq!(summary.skipped[0].position, 2);
    assert!(summary.skipped[0].reason.contains("found 1"));
    assert_postconditions(&records, &summary);

    // The quote opens the label: the absorbed rows become an unknown label value.
    let content = "text,label\nGET /a,good\nGET /b,\"bad\nGET /c,good\n";
    let path = scratch_file("unbalanced-label.csv", content.as_bytes());
    let (records, summary) = csv().read_records(&path);
    assert_eq!(summary.records, 2);
    assert_eq!(records.len(), 1);
    assert_eq!(summary.skipped[0].position, 2);
    assert!(summary.skipped[0].reason.contains("label"));

    // The quote opens the text in the last column: the record has the right field
    // count and a readable label, so it is a record whose text is the rest of the file.
    // Polars reads the same text, because the record is passed on in the standard form.
    let content = "label,text\ngood,GET /a\nbad,\"GET /b\ngood,GET /c\n";
    let path = scratch_file("unbalanced-text.csv", content.as_bytes());
    let (records, summary) = csv().read_records(&path);
    assert_eq!(summary.records, 2);
    assert!(summary.skipped.is_empty());
    let found: Vec<(&str, Label, u64)> = records
        .iter()
        .map(|r| (r.text.as_str(), r.label, r.first_position))
        .collect();
    assert_eq!(
        found,
        [
            ("GET /a", Label::Good, 1),
            ("GET /b\ngood,GET /c\n", Label::Bad, 2)
        ]
    );
    assert_postconditions(&records, &summary);
}

#[test]
fn csv_line_ends_of_every_kind_and_blank_lines_give_the_same_rows() {
    let rows = [
        "text,label",
        "GET /a,good",
        "\"GET /b,1\",bad",
        "GET /a,good",
    ];
    let mut frames = Vec::new();
    for (name, line_end, blank) in [
        ("lf.csv", "\n", ""),
        ("crlf.csv", "\r\n", ""),
        ("cr.csv", "\r", ""),
        ("lf-blank.csv", "\n", "\n\n"),
        ("crlf-blank.csv", "\r\n", "\r\n"),
    ] {
        // Blank lines between rows take no position; the last row has no line end.
        let content = rows.join(&format!("{line_end}{blank}"));
        let path = scratch_file(name, content.as_bytes());
        let (logs, summary) = csv().read(&path).unwrap();
        assert_logs_frame(&logs);
        assert_eq!(summary.records, 3, "{name}");
        assert!(summary.skipped.is_empty(), "{name}");
        let found: Vec<(String, Label, u64, u32)> = logs
            .records()
            .map(|r| (r.text, r.label, r.first_position, r.occurrences))
            .collect();
        assert_eq!(
            found,
            [
                ("GET /a".to_string(), Label::Good, 1, 2),
                ("GET /b,1".to_string(), Label::Bad, 2, 1)
            ],
            "{name}"
        );
        frames.push(logs);
    }
    assert!(frames.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn jsonl_escapes_extra_keys_and_line_ends_are_read_exactly() {
    let content = concat!(
        "\u{feff}",
        // 1: escapes, a surrogate pair, nested values under keys that are not read.
        r#"{"text":"line one\nline two \"quoted\" \\ café 😀\t.","label":"GOOD","group":"gü","extra":{"nested":[1,2,{"a":null}]},"n":1.5e3,"flag":true}"#,
        "\n",
        // 2: another key order, a null group, a padded label, CRLF line end.
        r#"{"group":null,"label":" bad ","text":"GET /b"}"#,
        "\r\n",
        // 3: the text of line 2 again.
        r#"{ "text" : "GET /b" , "label" : "bad" , "other" : "x" }"#,
        "\n",
        // 4: a null label.
        r#"{"text":"tab\there, comma","label":null}"#,
        "\n",
        // 5: an empty group is no group.
        r#"{"text":"GET /c","label":"unknown","group":""}"#,
        "\n",
        // 6: a blank line.
        "   \n",
        // 7: only the text key.
        r#"{"text":"GET /d"}"#,
        "\n",
        // 8: a text that looks like JSON and like a number.
        r#"{"text":"{\"text\":\"inner\"}","label":"bad","group":"42"}"#,
        "\n",
        // 9: a number as group.
        r#"{"text":"GET /e","label":"good","group":42}"#,
        "\n",
        // 10: not ASCII, written out; the last line has no line end.
        r#"{"text":"ünïcödé ☃","label":"good","group":"gü"}"#,
    );
    let path = scratch_file("adversarial.jsonl", content.as_bytes());
    let (records, summary) = jsonl().with_group_column("group").read_records(&path);

    let found: Vec<(&str, Label, Option<&str>, u64, u32)> = records
        .iter()
        .map(|r| {
            (
                r.text.as_str(),
                r.label,
                r.group.as_deref(),
                r.first_position,
                r.occurrences,
            )
        })
        .collect();
    assert_eq!(
        found,
        [
            (
                "line one\nline two \"quoted\" \\ café 😀\t.",
                Label::Good,
                Some("gü"),
                1,
                1
            ),
            ("GET /b", Label::Bad, None, 2, 2),
            ("tab\there, comma", Label::Unknown, None, 4, 1),
            ("GET /c", Label::Unknown, None, 5, 1),
            ("GET /d", Label::Unknown, None, 7, 1),
            ("{\"text\":\"inner\"}", Label::Bad, Some("42"), 8, 1),
            ("ünïcödé ☃", Label::Good, Some("gü"), 10, 1),
        ]
    );
    assert_eq!(summary.records, 10);
    let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(skipped, [6, 9]);
    assert!(summary.skipped[1].reason.contains("group is not a string"));
    assert_postconditions(&records, &summary);
    assert_eq!(summary.content_sha256, content_sha256(content.as_bytes()));
}

#[test]
fn lines_keep_their_bytes_whatever_they_contain() {
    let lines = [
        "plain",
        "with,comma and \"quotes\" and 'single'",
        "  leading and trailing  ",
        "tab\tseparated\tfields",
        "ünïcödé ☃ 😀",
        "\"",
        "\"unbalanced",
        "#not a comment",
        "null",
        "a\rb",
        "",
        "ends with a carriage return\r",
        "{\"json\":\"like\"}",
        "plain",
    ];
    for (name, line_end, last_line_end) in [
        ("bytes-lf.log", "\n", "\n"),
        ("bytes-crlf.log", "\r\n", "\r\n"),
        ("bytes-lf-open.log", "\n", ""),
    ] {
        let content = format!("\u{feff}{}{last_line_end}", lines.join(line_end));
        let path = scratch_file(name, content.as_bytes());
        let (records, summary) = LinesAdapter.read_records(&path);

        assert_eq!(summary.records, 14, "{name}");
        let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
        assert_eq!(skipped, [11], "{name}");
        let found: Vec<(&str, u64, u32)> = records
            .iter()
            .map(|r| (r.text.as_str(), r.first_position, r.occurrences))
            .collect();
        // With line feeds as line ends, a carriage return before the line feed is part
        // of the line end; with CRLF line ends it is the last character of the text.
        let text_of = |line: &'static str| match line_end {
            "\n" => line.strip_suffix('\r').unwrap_or(line),
            _ => line,
        };
        let mut expected: Vec<(&str, u64, u32)> = lines
            .iter()
            .enumerate()
            .filter(|(index, _)| ![10, 13].contains(index))
            .map(|(index, line)| (text_of(line), index as u64 + 1, 1))
            .collect();
        expected[0].2 = 2;
        assert_eq!(found, expected, "{name}");
        assert_postconditions(&records, &summary);
        assert_eq!(summary.content_sha256, content_sha256(content.as_bytes()));
    }
}

// --- Determinism: repeated reads give equal frames ----------------------------------------

#[test]
fn repeated_reads_give_equal_frames_and_summaries() {
    // Enough rows and repeats for Polars to split the work over its threads.
    let mut content = String::from("text,label,group\n");
    for index in 0..20_000u32 {
        let text = index % 7_919;
        let label = if text % 3 == 0 { "bad" } else { "good" };
        content.push_str(&format!(
            "\"GET /item?id={text},{}\",{label},/group/{}\n",
            index % 5,
            text % 97
        ));
    }
    let path = scratch_file("repeat.csv", content.as_bytes());
    let adapter = csv().with_group_column("group");
    let first = adapter.read(&path).unwrap();
    assert_logs_frame(&first.0);
    assert_eq!(first.1.records, 20_000);
    let (reference, _) = reference_csv(content.as_bytes(), "text", "label", "group");
    assert_eq!(first.0.to_records(), reference);
    for _ in 0..5 {
        assert_eq!(adapter.read(&path).unwrap(), first);
    }

    let lines = scratch_file("repeat.log", content.as_bytes());
    let first = LinesAdapter.read(&lines).unwrap();
    assert_eq!(first.1.records, 20_001);
    for _ in 0..3 {
        assert_eq!(LinesAdapter.read(&lines).unwrap(), first);
    }
}

// --- Records the two readers could read differently are skipped or passed on safely -------

#[test]
fn jsonl_lines_that_json_readers_disagree_on_are_skipped_not_guessed() {
    let content = concat!(
        r#"{"text":"GET /a","label":"good","group":"g"}"#,
        "\n",
        // 2 to 4: a key that is read occurs twice. Readers differ on which value counts.
        r#"{"text":"GET /e","label":"good","text":"GET /f"}"#,
        "\n",
        r#"{"text":"GET /e","label":"good","label":"bad"}"#,
        "\n",
        r#"{"text":"GET /e","group":"a","group":"b"}"#,
        "\n",
        // 5: a key that is not read occurs twice; that changes nothing.
        r#"{"text":"GET /b","label":"bad","other":1,"other":2}"#,
        "\n",
        // 6: an integer beyond 64 bits under a key that is not read. Polars refuses the
        // line; it is reported, and the lines around it are still read.
        r#"{"text":"GET /c","label":"bad","n":123456789012345678901234567890}"#,
        "\n",
        // 7: deeply nested values under a key that is not read.
        r#"{"text":"GET /d","label":"good","n":[[[{"a":{"b":[1,"x",null,true,1e308]}}]]]}"#,
        "\n",
        // 8: a NUL character in the text.
        r#"{"text":"a\u0000b","label":"good"}"#,
        "\n",
    );
    let path = scratch_file("disagree.jsonl", content.as_bytes());
    let (records, summary) = jsonl().with_group_column("group").read_records(&path);

    let found: Vec<(&str, Label, Option<&str>, u64)> = records
        .iter()
        .map(|r| {
            (
                r.text.as_str(),
                r.label,
                r.group.as_deref(),
                r.first_position,
            )
        })
        .collect();
    assert_eq!(
        found,
        [
            ("GET /a", Label::Good, Some("g"), 1),
            ("GET /b", Label::Bad, None, 5),
            ("GET /d", Label::Good, None, 7),
            ("a\0b", Label::Good, None, 8),
        ]
    );
    assert_eq!(summary.records, 8);
    let skipped: Vec<(u64, &str)> = summary
        .skipped
        .iter()
        .map(|s| (s.position, s.reason.as_str()))
        .collect();
    assert_eq!(skipped.len(), 4, "{skipped:?}");
    assert_eq!(skipped[0], (2, "key 'text' occurs more than once"));
    assert_eq!(skipped[1], (3, "key 'label' occurs more than once"));
    assert_eq!(skipped[2], (4, "key 'group' occurs more than once"));
    assert_eq!(
        skipped[3].0, 6,
        "skipped records are reported in source order"
    );
    assert!(
        skipped[3].1.contains("cannot be read by Polars"),
        "{}",
        skipped[3].1
    );
    assert_postconditions(&records, &summary);
}

#[test]
fn csv_bytes_that_are_not_text_fail_only_the_rows_that_use_them() {
    let content: &[u8] = b"text,label,other\n\
        GET /a,good,caf\xe9\n\
        GET /b,bad,ok\n\
        bad\xfftext,good,x\n\
        GET /c,go\xffod,x\n\
        GET /d,good,\"a\x00b,\xfe\"\n\
        a\x00b,bad,\n";
    let path = scratch_file("bytes.csv", content);
    let (records, summary) = csv().read_records(&path);

    let found: Vec<(&str, Label, u64)> = records
        .iter()
        .map(|r| (r.text.as_str(), r.label, r.first_position))
        .collect();
    assert_eq!(
        found,
        [
            // Row 1 holds a byte that is not UTF-8 in a column that is not read.
            ("GET /a", Label::Good, 1),
            ("GET /b", Label::Bad, 2),
            ("GET /d", Label::Good, 5),
            ("a\0b", Label::Bad, 6),
        ]
    );
    let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(skipped, [3, 4]);
    assert!(
        summary.skipped[0]
            .reason
            .contains("text is not valid UTF-8")
    );
    assert!(
        summary.skipped[1]
            .reason
            .contains("label is not valid UTF-8")
    );
    assert_postconditions(&records, &summary);
}

#[test]
fn one_csv_column_may_serve_as_text_and_group() {
    let path = scratch_file("same-column.csv", b"text,label\nGET /a,good\nGET /a,bad\n");
    let (records, summary) = csv().with_group_column("text").read_records(&path);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].group.as_deref(), Some("GET /a"));
    assert_eq!(records[0].label, Label::Unknown);
    assert_eq!(summary.label_conflicts, 1);
}

#[test]
fn lines_may_be_long_and_hold_control_characters() {
    let long = "x".repeat(2_000_000);
    let content = format!("{long}\na\0b\n\u{1}\u{2}\u{7f}\n{long}\nshort\n");
    let path = scratch_file("long.log", content.as_bytes());
    let (records, summary) = LinesAdapter.read_records(&path);
    assert_eq!(summary.records, 5);
    assert!(summary.skipped.is_empty());
    let found: Vec<(&str, u64, u32)> = records
        .iter()
        .map(|r| (r.text.as_str(), r.first_position, r.occurrences))
        .collect();
    assert_eq!(
        found,
        [
            (long.as_str(), 1, 2),
            ("a\0b", 2, 1),
            ("\u{1}\u{2}\u{7f}", 3, 1),
            ("short", 5, 1)
        ]
    );
}
