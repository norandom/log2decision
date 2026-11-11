//! Properties of the synthetic labelled fixture `tests/fixtures/mini-labelled.csv`.
//!
//! Later tasks rely on these properties (dedup and malformed-row handling, baseline fitting,
//! triage, the 30-per-group shift test, grouped evaluation). The file is read here with a
//! minimal CSV reader on purpose: the input adapters are a separate task.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const HEADER: &str = "text,label,group";
const HOST: &str = "http://localhost:8080";

#[derive(Debug, Clone, PartialEq, Eq)]
struct Row {
    text: String,
    label: String,
    group: String,
}

/// The data rows of the fixture: `Ok` for a well-formed row, `Err(raw line)` otherwise.
struct Fixture {
    rows: Vec<Result<Row, String>>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mini-labelled.csv")
}

/// Splits one CSV line into fields (quoted fields, doubled quotes). `None` when a quote is
/// left open.
fn split_csv_line(line: &str) -> Option<Vec<String>> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') if chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            (true, '"') => quoted = false,
            (true, other) => field.push(other),
            (false, '"') if field.is_empty() => quoted = true,
            (false, ',') => fields.push(std::mem::take(&mut field)),
            (false, other) => field.push(other),
        }
    }
    if quoted {
        return None;
    }
    fields.push(field);
    Some(fields)
}

fn parse_row(line: &str) -> Result<Row, String> {
    match split_csv_line(line).as_deref() {
        Some([text, label, group])
            if !text.is_empty()
                && !group.is_empty()
                && matches!(label.as_str(), "good" | "bad") =>
        {
            Ok(Row {
                text: text.clone(),
                label: label.clone(),
                group: group.clone(),
            })
        }
        _ => Err(line.to_string()),
    }
}

fn load() -> Fixture {
    let path = fixture_path();
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture {} is readable: {e}", path.display()));
    let mut lines = content.lines();
    assert_eq!(lines.next(), Some(HEADER), "header row");
    Fixture {
        rows: lines.map(parse_row).collect(),
    }
}

impl Fixture {
    fn valid(&self) -> Vec<&Row> {
        self.rows.iter().filter_map(|r| r.as_ref().ok()).collect()
    }

    /// For every distinct text: the labels it carries and how often it occurs.
    fn by_text(&self) -> BTreeMap<&str, (BTreeSet<&str>, usize)> {
        let mut map: BTreeMap<&str, (BTreeSet<&str>, usize)> = BTreeMap::new();
        for row in self.valid() {
            let entry = map.entry(row.text.as_str()).or_default();
            entry.0.insert(row.label.as_str());
            entry.1 += 1;
        }
        map
    }

    /// Distinct texts that carry only `label`.
    fn distinct_with_only(&self, label: &str) -> usize {
        self.by_text()
            .values()
            .filter(|(labels, _)| labels.len() == 1 && labels.contains(label))
            .count()
    }
}

#[test]
fn fixture_is_plain_ascii_with_a_header_and_200_data_rows() {
    let content = std::fs::read_to_string(fixture_path()).expect("fixture is readable");
    assert!(content.is_ascii(), "the fixture is plain ASCII");
    assert!(content.ends_with('\n'), "the last row is terminated");
    assert!(content.lines().all(|line| !line.trim().is_empty()));
    assert_eq!(load().rows.len(), 200);
}

#[test]
fn fixture_has_exactly_one_malformed_row() {
    let fixture = load();
    let malformed: Vec<(usize, &String)> = fixture
        .rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.as_ref().err().map(|raw| (i + 1, raw)))
        .collect();
    assert_eq!(malformed.len(), 1, "malformed rows: {malformed:?}");
    // Data row 121 (line 122 of the file): the group column is missing.
    let (position, raw) = malformed[0];
    assert_eq!(position, 121);
    assert_eq!(split_csv_line(raw).map(|fields| fields.len()), Some(2));
    assert_eq!(fixture.valid().len(), 199);
}

#[test]
fn every_row_is_a_request_line_whose_group_key_is_method_and_path() {
    for row in load().valid() {
        let (method, rest) = row.text.split_once(' ').expect("METHOD URL");
        assert!(matches!(method, "GET" | "POST"), "method of {:?}", row.text);
        let (url, body) = match rest.split_once(' ') {
            Some((url, body)) => (url, Some(body)),
            None => (rest, None),
        };
        let path_and_query = url.strip_prefix(HOST).expect("URL on the fixture host");
        let path = path_and_query.split('?').next().unwrap();
        assert!(path.starts_with("/tienda1/"), "path of {:?}", row.text);
        assert_eq!(row.group, format!("{method} {path}"), "group key");
        match method {
            "POST" => assert!(body.is_some_and(|b| !b.is_empty()), "POST carries a body"),
            _ => assert!(body.is_none(), "GET carries no body: {:?}", row.text),
        }
    }
}

#[test]
fn at_least_six_group_keys_each_hold_both_labels() {
    let fixture = load();
    let mut labels_by_group: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for row in fixture.valid() {
        labels_by_group
            .entry(row.group.as_str())
            .or_default()
            .insert(row.label.as_str());
    }
    assert!(labels_by_group.len() >= 6, "groups: {labels_by_group:?}");
    for (group, labels) in &labels_by_group {
        assert_eq!(labels.len(), 2, "group {group:?} holds both labels");
    }
}

#[test]
fn at_least_thirty_distinct_texts_per_label() {
    let fixture = load();
    assert!(fixture.distinct_with_only("good") >= 30);
    assert!(fixture.distinct_with_only("bad") >= 30);
}

#[test]
fn some_texts_are_exact_duplicates_with_one_label() {
    let fixture = load();
    let by_text = fixture.by_text();
    let repeated = by_text
        .values()
        .filter(|(labels, count)| labels.len() == 1 && *count > 1)
        .count();
    assert!(repeated >= 10, "only {repeated} repeated texts");
    assert!(
        by_text
            .values()
            .any(|(labels, n)| labels.len() == 1 && *n >= 3),
        "at least one text occurs three times"
    );
    // A duplicate repeats the whole row: same text, same label, same group.
    let mut groups_by_text: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for row in fixture.valid() {
        groups_by_text
            .entry(row.text.as_str())
            .or_default()
            .insert(row.group.as_str());
    }
    assert!(groups_by_text.values().all(|groups| groups.len() == 1));
}

#[test]
fn exactly_one_text_carries_conflicting_labels() {
    let fixture = load();
    let by_text = fixture.by_text();
    let conflicting: Vec<_> = by_text
        .iter()
        .filter(|(_, (labels, _))| labels.len() > 1)
        .collect();
    assert_eq!(conflicting.len(), 1, "conflicting texts: {conflicting:?}");
    let (_, (_, occurrences)) = conflicting[0];
    assert_eq!(*occurrences, 2, "once good, once bad");
}

#[test]
fn bad_rows_carry_attack_like_payloads_and_good_rows_do_not() {
    let fixture = load();
    let by_text = fixture.by_text();
    let texts_with_only = |label: &str| -> Vec<String> {
        by_text
            .iter()
            .filter(|(_, (labels, _))| labels.len() == 1 && labels.contains(label))
            .map(|(text, _)| text.to_ascii_lowercase())
            .collect()
    };
    let bad = texts_with_only("bad");
    let good = texts_with_only("good");
    // SQL injection, cross-site scripting, path traversal, command injection.
    for marker in ["union", "script", "etc/passwd", "ping"] {
        assert!(
            bad.iter().any(|text| text.contains(marker)),
            "no bad text contains {marker:?}"
        );
        assert!(
            good.iter().all(|text| !text.contains(marker)),
            "a good text contains {marker:?}"
        );
    }
    // Some texts need CSV quoting, so readers are exercised on commas and doubled quotes.
    assert!(bad.iter().any(|text| text.contains(',')));
    assert!(bad.iter().any(|text| text.contains('"')));
}

/// The exact counts later tasks may assert against (dedup and summary tests).
#[test]
fn documented_counts_hold() {
    let fixture = load();
    let valid = fixture.valid();
    let count = |label: &str| valid.iter().filter(|r| r.label == label).count();
    let by_text = fixture.by_text();
    let groups: BTreeSet<&str> = valid.iter().map(|r| r.group.as_str()).collect();
    assert_eq!(groups.len(), 8, "group keys");
    assert_eq!(by_text.len(), 169, "distinct texts");
    assert_eq!(fixture.distinct_with_only("good"), 96);
    assert_eq!(fixture.distinct_with_only("bad"), 72);
    assert_eq!(
        by_text.values().filter(|(_, n)| *n > 1).count(),
        27,
        "texts occurring more than once, including the conflicting one"
    );
    assert_eq!((count("good"), count("bad")), (115, 84), "rows per label");
}
