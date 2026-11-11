//! Input adapters, exact-text deduplication and the input summary
//! (requirements 1.1, 1.2, 1.3, 1.4, 1.6).
//!
//! "Position" is the 1-based index of a record in its source: the line number for
//! line-based and JSONL input, the data-row index (header not counted) for CSV input.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LinesAdapter,
    LogRecord,
};

fn scratch_file(name: &str, content: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("l2d-core-input-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    path
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
    let (records, summary) = csv().with_group_column("group").read(&path).unwrap();

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
        .read(&fixture_path())
        .unwrap();
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
    let (records, summary) = LinesAdapter.read(&path).unwrap();
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
    let (records, summary) = LinesAdapter.read(&path).unwrap();

    assert_eq!(summary.source_path, path);
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
    let (records, summary) = LinesAdapter.read(&path).unwrap();
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
    let (records, summary) = LinesAdapter.read(&path).unwrap();
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
    let (records, summary) = adapter.read(&path).unwrap();

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
    let (records, summary) = adapter.read(&path).unwrap();
    assert_eq!(summary.records, 3);
    assert_eq!(records.len(), 2);
    assert!(records.iter().all(|r| r.label == Label::Unknown));
}

#[test]
fn csv_with_a_byte_order_mark_still_finds_its_columns() {
    let path = scratch_file("bom.csv", b"\xef\xbb\xbftext,label\nGET /a,bad\n");
    let (records, summary) = csv().read(&path).unwrap();
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
    let (records, summary) = csv().with_group_column("group").read(&path).unwrap();

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
    let (records, summary) = jsonl().with_group_column("group").read(&path).unwrap();

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
    let (records, _) = adapter.read(&path).unwrap();
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
