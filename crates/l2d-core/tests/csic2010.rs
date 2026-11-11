//! The HTTP CSIC 2010 adapter (requirements 1.2, 9.7, 10.4): text layout, labels, group
//! keys, routing/payload separation and dataset location.
//!
//! The excerpt below is committed with the tests. The real dataset is untracked; the
//! real-file test is skipped with a printed note when it is absent.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use l2d_core::adapters::csic2010::{
    CSIC2010_FILE_NAME, Csic2010Adapter, Csic2010Counts, group_key, split_routing,
};
use l2d_core::{InputAdapter, InputSummary, L2dError, Label, LogRecord, ProjectPaths};

const HEADER: &str = ",Method,User-Agent,Pragma,Cache-Control,Accept,Accept-encoding,Accept-charset,language,host,cookie,content-type,connection,lenght,content,classification,URL";

/// The request-header columns between `Method` and `cookie`, as in the dataset. The
/// quoted fields with commas are what makes every data row need a real CSV reader.
const AGENT: &str = r#"Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko),no-cache,no-cache,"text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5","x-gzip, x-deflate, gzip, deflate","utf-8, utf-8;q=0.5, *;q=0.5",en,localhost:8080"#;

const SQL_BODY: &str = "id=2&nombre=Jam%F3n+Ib%E9rico&precio=85&cantidad=%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre+LIKE+%27%25&B1=A%F1adir+al+carrito";

fn get(status: &str, class: &str, cookie: &str, url_field: &str) -> String {
    format!("{status},GET,{AGENT},JSESSIONID={cookie},,close,,,{class},{url_field}")
}

fn with_body(status: &str, method: &str, class: &str, body: &str, url_field: &str) -> String {
    format!(
        "{status},{method},{AGENT},JSESSIONID=B0,application/x-www-form-urlencoded,Connection: close,Content-Length: {},{body},{class},{url_field}",
        body.len()
    )
}

/// Ten data rows in the layout of `csic_database.csv`.
fn excerpt() -> String {
    let rows = [
        // 1: plain GET.
        get(
            "Normal",
            "0",
            "A1",
            "http://localhost:8080/tienda1/index.jsp HTTP/1.1",
        ),
        // 2: GET with a query string.
        get(
            "Normal",
            "0",
            "A2",
            "http://localhost:8080/tienda1/publico/anadir.jsp?id=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito HTTP/1.1",
        ),
        // 3: POST with a body.
        with_body(
            "Normal",
            "POST",
            "0",
            "id=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito",
            "http://localhost:8080/tienda1/publico/anadir.jsp HTTP/1.1",
        ),
        // 4: POST with an attack in the body.
        with_body(
            "Anomalous",
            "POST",
            "1",
            SQL_BODY,
            "http://localhost:8080/tienda1/publico/anadir.jsp HTTP/1.1",
        ),
        // 5: attack with raw quotes, commas and spaces in the query string.
        get(
            "Anomalous",
            "1",
            "A5",
            r#""http://localhost:8080/tienda1/publico/pagar.jsp?modo=insertar&precio=183','0','0');waitfor delay ""0:0:15"";--&B1=Pasar+por+caja HTTP/1.1""#,
        ),
        // 6: the request of row 1 again with another session cookie.
        get(
            "Normal",
            "0",
            "A6",
            "http://localhost:8080/tienda1/index.jsp HTTP/1.1",
        ),
        // 7: PUT with a body on a path that only attacks use.
        with_body(
            "Anomalous",
            "PUT",
            "1",
            "modo=entrar&login=bob",
            "http://localhost:8080/tienda1/publico/entrar.jsp.BAK HTTP/1.1",
        ),
        // 8: the two label columns disagree.
        get(
            "Normal",
            "1",
            "A8",
            "http://localhost:8080/tienda1/miembros/salir.jsp HTTP/1.1",
        ),
        // 9: too few fields.
        "Anomalous,GET,only-three-fields".to_string(),
        // 10: the endpoint of row 2 with an attack in the query string.
        get(
            "Anomalous",
            "1",
            "A10",
            "http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=%3CSCRIPT%3Ealert%28%22Paros%22%29%3B%3C%2FSCRIPT%3E&precio=85 HTTP/1.1",
        ),
    ];
    format!("{HEADER}\n{}\n", rows.join("\n"))
}

/// A scratch directory in the system temp folder, removed when dropped.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("l2d-core-csic2010-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn file(&self, name: &str, content: &str) -> PathBuf {
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

fn read_counted(name: &str, content: &str) -> (Vec<LogRecord>, InputSummary, Csic2010Counts) {
    let scratch = Scratch::new(name);
    let path = scratch.file(CSIC2010_FILE_NAME, content);
    Csic2010Adapter::new().read_counted(&path).unwrap()
}

/// Design postconditions: every text is unique; `records` = occurrences + skipped.
fn assert_postconditions(records: &[LogRecord], summary: &InputSummary) {
    let texts: HashSet<&str> = records.iter().map(|r| r.text.as_str()).collect();
    assert_eq!(texts.len(), records.len(), "every returned text is unique");
    assert_eq!(summary.distinct_texts, records.len() as u64);
    let occurrences: u64 = records.iter().map(|r| u64::from(r.occurrences)).sum();
    assert_eq!(summary.records, occurrences + summary.skipped.len() as u64);
}

// --- Excerpt: text, label and group key per row ------------------------------------------

#[test]
fn excerpt_rows_become_text_label_and_group_key() {
    let (records, summary, counts) = read_counted("excerpt", &excerpt());

    let expected: Vec<(String, Label, &str, u64, u32)> = vec![
        (
            "GET http://localhost:8080/tienda1/index.jsp".into(),
            Label::Good,
            "GET /tienda1/index.jsp",
            1,
            2,
        ),
        (
            "GET http://localhost:8080/tienda1/publico/anadir.jsp?id=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito".into(),
            Label::Good,
            "GET /tienda1/publico/anadir.jsp",
            2,
            1,
        ),
        (
            "POST http://localhost:8080/tienda1/publico/anadir.jsp\nid=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito".into(),
            Label::Good,
            "POST /tienda1/publico/anadir.jsp",
            3,
            1,
        ),
        (
            format!("POST http://localhost:8080/tienda1/publico/anadir.jsp\n{SQL_BODY}"),
            Label::Bad,
            "POST /tienda1/publico/anadir.jsp",
            4,
            1,
        ),
        (
            r#"GET http://localhost:8080/tienda1/publico/pagar.jsp?modo=insertar&precio=183','0','0');waitfor delay "0:0:15";--&B1=Pasar+por+caja"#.into(),
            Label::Bad,
            "GET /tienda1/publico/pagar.jsp",
            5,
            1,
        ),
        (
            "PUT http://localhost:8080/tienda1/publico/entrar.jsp.BAK\nmodo=entrar&login=bob".into(),
            Label::Bad,
            "PUT /tienda1/publico/entrar.jsp.BAK",
            7,
            1,
        ),
        (
            "GET http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=%3CSCRIPT%3Ealert%28%22Paros%22%29%3B%3C%2FSCRIPT%3E&precio=85".into(),
            Label::Bad,
            "GET /tienda1/publico/anadir.jsp",
            10,
            1,
        ),
    ];
    assert_eq!(records.len(), expected.len());
    for (record, (text, label, group, position, occurrences)) in records.iter().zip(&expected) {
        assert_eq!(&record.text, text);
        assert_eq!(record.label, *label, "label of {text:?}");
        assert_eq!(record.group.as_deref(), Some(*group), "group of {text:?}");
        assert_eq!(record.first_position, *position, "position of {text:?}");
        assert_eq!(record.occurrences, *occurrences, "occurrences of {text:?}");
    }

    assert_eq!(summary.records, 10);
    assert_eq!(summary.distinct_texts, 7);
    assert_eq!(summary.label_conflicts, 0);
    assert_eq!(counts, Csic2010Counts { good: 4, bad: 4 });
    assert_postconditions(&records, &summary);

    let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(skipped, vec![8, 9]);
    assert!(
        summary.skipped[0].reason.contains("disagree"),
        "{}",
        summary.skipped[0].reason
    );
    assert!(
        summary.skipped[1].reason.contains("fields"),
        "{}",
        summary.skipped[1].reason
    );
}

#[test]
fn group_key_is_isolated_from_host_query_body_and_protocol_version() {
    let (records, _, _) = read_counted("isolation", &excerpt());

    for record in &records {
        let group = record.group.as_deref().expect("every record has a group");
        assert!(!group.contains("localhost"), "{group}");
        assert!(!group.contains("://"), "{group}");
        assert!(!group.contains('?'), "{group}");
        assert!(!group.contains('='), "{group}");
        assert!(!group.contains('\n'), "{group}");
        assert!(!group.contains("HTTP/"), "{group}");
        assert!(!record.text.contains("HTTP/1.1"), "{}", record.text);
        assert_eq!(group, group_key(&record.text));
    }

    // One endpoint, different payloads and labels: same key.
    let anadir_get: Vec<&LogRecord> = records
        .iter()
        .filter(|r| r.group.as_deref() == Some("GET /tienda1/publico/anadir.jsp"))
        .collect();
    assert_eq!(anadir_get.len(), 2);
    assert_ne!(anadir_get[0].text, anadir_get[1].text);
    assert_ne!(anadir_get[0].label, anadir_get[1].label);

    // Same path, different method: different key.
    let keys: HashSet<&str> = records.iter().filter_map(|r| r.group.as_deref()).collect();
    assert!(keys.contains("POST /tienda1/publico/anadir.jsp"));
    assert_eq!(keys.len(), 5);
}

// --- Routing is separable from payload ----------------------------------------------------

#[test]
fn text_splits_into_routing_and_payload() {
    let split = split_routing("GET http://localhost:8080/tienda1/index.jsp");
    assert_eq!(split.routing, "GET http://localhost:8080/tienda1/index.jsp");
    assert_eq!(split.payload, "");

    let split = split_routing("GET http://localhost:8080/t/a.jsp?id=3&x=a?b");
    assert_eq!(split.routing, "GET http://localhost:8080/t/a.jsp");
    assert_eq!(split.payload, "id=3&x=a?b");

    let split = split_routing("POST http://localhost:8080/t/a.jsp\nid=3&q=what?\nline two");
    assert_eq!(split.routing, "POST http://localhost:8080/t/a.jsp");
    assert_eq!(split.payload, "id=3&q=what?\nline two");

    let split = split_routing("POST http://localhost:8080/t/a.jsp?m=1\nid=3");
    assert_eq!(split.routing, "POST http://localhost:8080/t/a.jsp");
    assert_eq!(split.payload, "m=1\nid=3");
}

#[test]
fn excerpt_payload_never_carries_the_endpoint_and_routing_never_the_payload() {
    let (records, _, _) = read_counted("split", &excerpt());

    for record in &records {
        let split = split_routing(&record.text);
        assert!(record.text.starts_with(split.routing));
        assert!(record.text.ends_with(split.payload));
        assert!(!split.routing.contains('?'));
        assert!(!split.routing.contains('\n'));
        assert!(!split.payload.contains("tienda1"), "{}", split.payload);
        assert_eq!(
            group_key(split.routing),
            record.group.clone().unwrap(),
            "the routing part alone determines the group key"
        );
    }

    let attack = records.iter().find(|r| r.first_position == 4).unwrap();
    assert_eq!(split_routing(&attack.text).payload, SQL_BODY);
}

#[test]
fn group_key_drops_scheme_host_query_and_body() {
    assert_eq!(
        group_key("GET http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=x"),
        "GET /tienda1/publico/anadir.jsp"
    );
    assert_eq!(group_key("POST https://shop.example/a/b\nx=1"), "POST /a/b");
    assert_eq!(
        group_key("GET /already/a/path?x=http://evil/"),
        "GET /already/a/path"
    );
    assert_eq!(group_key("GET http://localhost:8080"), "GET /");
    assert_eq!(group_key("GET http://localhost:8080?x=/a/b"), "GET /");
}

// --- Labels and unreadable rows -----------------------------------------------------------

#[test]
fn one_text_with_both_labels_is_a_conflict_and_counts_keep_both() {
    let content = format!(
        "{HEADER}\n{}\n{}\n",
        get(
            "Normal",
            "0",
            "C1",
            "http://localhost:8080/tienda1/index.jsp HTTP/1.1"
        ),
        get(
            "Anomalous",
            "1",
            "C2",
            "http://localhost:8080/tienda1/index.jsp HTTP/1.1"
        ),
    );
    let (records, summary, counts) = read_counted("conflict", &content);

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].label, Label::Unknown);
    assert_eq!(records[0].occurrences, 2);
    assert_eq!(summary.label_conflicts, 1);
    assert_eq!(counts, Csic2010Counts { good: 1, bad: 1 });
}

#[test]
fn unreadable_rows_are_skipped_with_position_and_reason() {
    let content = format!(
        "{HEADER}\n{}\n{}\nNormal,,{AGENT},JSESSIONID=S3,,close,,,0,http://localhost:8080/a HTTP/1.1\n{}\n",
        get("Normal", "2", "S1", "http://localhost:8080/a HTTP/1.1"),
        get("Normal", "0", "S2", ""),
        get("Normal", "0", "S4", "http://localhost:8080/ok HTTP/1.1"),
    );
    let (records, summary, counts) = read_counted("skips", &content);

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].text, "GET http://localhost:8080/ok");
    assert_eq!(records[0].first_position, 4);
    assert_eq!(counts, Csic2010Counts { good: 1, bad: 0 });
    let skipped: Vec<u64> = summary.skipped.iter().map(|s| s.position).collect();
    assert_eq!(skipped, vec![1, 2, 3]);
    assert!(summary.skipped[0].reason.contains("classification"));
    assert!(summary.skipped[1].reason.contains("URL"));
    assert!(summary.skipped[2].reason.contains("method"));
    assert_postconditions(&records, &summary);
}

#[test]
fn a_file_without_the_csic_columns_is_invalid_input() {
    let scratch = Scratch::new("columns");
    let path = scratch.file("other.csv", "text,label\nGET /a,good\n");
    let error = Csic2010Adapter::new().read(&path).unwrap_err();
    match error {
        L2dError::InvalidInput { detail } => assert!(detail.contains("Method"), "{detail}"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

// --- Location (10.4, 10.5) ----------------------------------------------------------------

#[test]
fn dataset_is_located_under_the_data_directory_or_at_the_supplied_path() {
    let scratch = Scratch::new("locate");
    let paths = ProjectPaths::new(&scratch.dir);
    let default_location = scratch
        .dir
        .join("data")
        .join("csic-2010")
        .join(CSIC2010_FILE_NAME);

    match Csic2010Adapter::locate(&paths, None).unwrap_err() {
        L2dError::Missing { item, expected_at } => {
            assert!(item.contains("CSIC 2010"), "{item}");
            assert_eq!(expected_at, default_location);
        }
        other => panic!("expected Missing, got {other:?}"),
    }

    std::fs::create_dir_all(default_location.parent().unwrap()).unwrap();
    std::fs::write(&default_location, excerpt()).unwrap();
    assert_eq!(
        Csic2010Adapter::locate(&paths, None).unwrap(),
        default_location
    );

    // A supplied file is used as given; a supplied directory is searched for the file.
    let elsewhere = scratch.file("copy.csv", &excerpt());
    assert_eq!(
        Csic2010Adapter::locate(&paths, Some(&elsewhere)).unwrap(),
        elsewhere
    );
    assert_eq!(
        Csic2010Adapter::locate(&paths, Some(default_location.parent().unwrap())).unwrap(),
        default_location
    );

    let absent = scratch.dir.join("absent.csv");
    match Csic2010Adapter::locate(&paths, Some(&absent)).unwrap_err() {
        L2dError::Missing { expected_at, .. } => assert_eq!(expected_at, absent),
        other => panic!("expected Missing, got {other:?}"),
    }
    match Csic2010Adapter::new().read(&absent).unwrap_err() {
        L2dError::Missing { expected_at, .. } => assert_eq!(expected_at, absent),
        other => panic!("expected Missing, got {other:?}"),
    }
}

// --- The real dataset (untracked): known totals -------------------------------------------

#[test]
fn real_dataset_has_the_known_totals() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let path = match Csic2010Adapter::locate(&ProjectPaths::new(root), None) {
        Ok(path) => path,
        Err(error @ L2dError::Missing { .. }) => {
            eprintln!("SKIPPED real_dataset_has_the_known_totals: {error}");
            return;
        }
        Err(other) => panic!("locating the dataset failed: {other:?}"),
    };

    let started = std::time::Instant::now();
    let (records, summary, counts) = Csic2010Adapter::new().read_counted(&path).unwrap();
    let elapsed = started.elapsed();

    // Before deduplication. A swallowed row (unbalanced quote) would lower the total.
    assert_eq!(summary.records, 61_065);
    assert_eq!(
        counts,
        Csic2010Counts {
            good: 36_000,
            bad: 25_065
        }
    );
    // No row skipped: among other things, the classification column agrees with the
    // first column (`Normal` / `Anomalous`) on every row.
    assert!(summary.skipped.is_empty(), "{:?}", summary.skipped.first());
    assert_postconditions(&records, &summary);

    let mut groups: HashSet<&str> = HashSet::new();
    for record in &records {
        let group = record.group.as_deref().expect("every record has a group");
        let split = split_routing(&record.text);
        assert_eq!(group, group_key(split.routing), "{}", record.text);
        assert!(!group.contains('?') && !group.contains("://") && !group.contains('\n'));
        assert!(!record.text.ends_with(" HTTP/1.1"), "{}", record.text);
        groups.insert(group);
    }
    let label = |label: Label| records.iter().filter(|r| r.label == label).count();

    eprintln!(
        "CSIC 2010: {} records ({} good, {} bad) before dedup; {} distinct texts \
         ({} good, {} bad, {} conflicting); {} distinct group keys; sha256 {}; read in {:.2?}",
        summary.records,
        counts.good,
        counts.bad,
        summary.distinct_texts,
        label(Label::Good),
        label(Label::Bad),
        summary.label_conflicts,
        groups.len(),
        summary.content_sha256,
        elapsed,
    );
    assert_eq!(label(Label::Unknown) as u64, summary.label_conflicts);
}
