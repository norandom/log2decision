//! Marking against a store (requirement 16.1, 16.3; task 1.21, owner decision of
//! 2026-11-16): a stored log is named in a mark set by its `log_id`; the store's content
//! hash is the SHA-256 over its ids in sorted order; a phrase goes through the store's
//! text index, a regular expression over the store read in batches of `(seq, id, text)`.
//!
//! The fixture `tests/fixtures/mini-labelled.csv` (169 logs) is loaded into a store
//! with synthetic vectors (no encoder is needed to mark). The expected logs are those
//! the same query marks on the fixture read as an input, named by their ids.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_core::marks::{MarkQuery, MarkSet, MarkStore};
use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    EncoderIdentity, InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_store::{
    CONTENT_HASH_SQL, LogFlags, SEARCH_FIXED_BYTES, Store, VectorOrigin, log_id, rows_frame,
};
use sha2::{Digest, Sha256};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const DIM: u16 = 4;
const SOURCE: &str = "mini";
const DROP_TABLE_PATTERN: &str = r"(?i)drop(\s|\+)+table";

fn identity() -> EncoderIdentity {
    EncoderIdentity {
        name: "bytecnn".to_string(),
        revision: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        dimension: DIM,
    }
}

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-store-marks-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    fn paths(&self) -> ProjectPaths {
        ProjectPaths::new(&self.root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn budget_with(limit: Option<u64>, batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).unwrap()
}

fn budget() -> MemoryBudget {
    budget_with(None, None)
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv")
}

fn fixture() -> (LogsFrame, InputSummary) {
    let (logs, summary, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .read_with(&fixture_path(), &mut budget())
        .unwrap();
    assert_eq!(logs.len(), 169);
    (logs, summary)
}

/// The store rows of `logs` from [`SOURCE`], with a fixed made-up vector each.
fn rows(logs: &LogsFrame) -> DataFrame {
    let vectors: Vec<f32> = (0..logs.len())
        .flat_map(|index| [index as f32, 1.0, 0.0, 0.0])
        .collect();
    let flags = vec![LogFlags::default(); logs.len()];
    rows_frame(logs, SOURCE, &vectors, DIM, &flags).unwrap()
}

fn store_named(scratch: &Scratch, name: &str, logs: &LogsFrame) -> Store {
    let identity = identity();
    let origin = VectorOrigin::without_release(&identity);
    let mut store = Store::create(&scratch.paths(), name, origin).unwrap();
    store.add_with(&rows(logs), origin, &mut budget()).unwrap();
    store
}

/// The ids, sorted, of the logs the query marks on the fixture read as an input.
fn expected_ids(query: MarkQuery) -> Vec<String> {
    let (logs, summary) = fixture();
    let set = MarkSet::mark("input", Label::Bad, query, &logs, &summary, &budget()).unwrap();
    let mut ids: Vec<String> = set
        .marked_in(&logs)
        .unwrap()
        .records()
        .map(|record| log_id(SOURCE, &record.text))
        .collect();
    ids.sort();
    ids
}

/// The content hash as the owner's decision defines it, computed here independently.
fn hash_of_ids(mut ids: Vec<String>) -> String {
    ids.sort();
    let mut hasher = Sha256::new();
    for id in &ids {
        hasher.update(id.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn all_ids(logs: &LogsFrame) -> Vec<String> {
    logs.records()
        .map(|record| log_id(SOURCE, &record.text))
        .collect()
}

#[test]
fn concurrent_mark_writer_child() {
    let Ok(root) = std::env::var("L2D_MARK_WRITER_PROJECT") else {
        return;
    };
    let mut store = Store::open(&ProjectPaths::new(root), "s").unwrap();
    let (logs, _) = fixture();
    let vectors: Vec<f32> = (0..logs.len()).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let data = rows_frame(
        &logs,
        "second-process",
        &vectors,
        DIM,
        &vec![LogFlags::default(); logs.len()],
    )
    .unwrap();
    store
        .add_with(
            &data,
            VectorOrigin::without_release(&identity()),
            &mut budget(),
        )
        .unwrap();
}

#[test]
fn direct_fill_accounting_child() {
    let Some(root) = std::env::var_os("L2D_DIRECT_FILL_ACCOUNTING") else {
        return;
    };
    let root = PathBuf::from(root);
    let identity = identity();
    let origin = VectorOrigin::without_release(&identity);
    let mut store = Store::create(&ProjectPaths::new(root.join("project")), "s", origin).unwrap();
    let (logs, _) = fixture();
    let rows = rows(&logs);
    std::fs::write(root.join("ready"), "").unwrap();
    let mut fill = store.begin_fill(origin).unwrap();
    fill.add_with(&rows, &mut budget()).unwrap();
    let mut tiny = budget_with(Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 1), None);
    assert!(fill.add_with(&rows, &mut tiny).is_err());
    drop(fill);
    let records =
        std::fs::read_dir(std::env::temp_dir().join("log2decision-resource-records")).unwrap();
    let mut seen = 0;
    for record in records {
        let report: l2d_core::resource::ResourceReport =
            serde_json::from_slice(&std::fs::read(record.unwrap().path()).unwrap()).unwrap();
        assert_eq!(report.operation, "store fill add");
        assert!(
            report.waited_for_host,
            "a direct fill waited before its adds: {report:?}"
        );
        assert_eq!(
            report.observation.host_holder.as_ref().unwrap().operation,
            "direct store fill holder"
        );
        assert!(report.memory.counted_peak_bytes <= report.memory.memory_limit_bytes);
        assert!(!report.memory.spilled);
        seen += 1;
    }
    assert_eq!(
        seen, 2,
        "both successful and refused adds retain admission accounting"
    );
}

#[test]
fn a_direct_store_fill_records_the_admission_it_waited_for() {
    use l2d_core::resource::{BusyChoice, ResourceGuard};
    use std::time::{Duration, Instant};
    let scratch = Scratch::new("direct-fill-accounting");
    let temporary = scratch.root.join("temporary");
    std::fs::create_dir_all(&temporary).unwrap();
    let holder = ResourceGuard::enter("direct store fill holder", BusyChoice::Wait).unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "direct_fill_accounting_child", "--nocapture"])
        .env("L2D_DIRECT_FILL_ACCOUNTING", &scratch.root)
        .env("TEMP", &temporary)
        .env("TMP", &temporary)
        .env("TMPDIR", &temporary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while !scratch.root.join("ready").exists() {
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(child.try_wait().unwrap().is_none());
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(200));
    assert!(child.try_wait().unwrap().is_none());
    drop(holder);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn an_actual_second_process_writer_between_hash_reads_is_refused() {
    struct BetweenHashes<'a> {
        store: &'a Store,
        root: &'a Path,
        hashes: std::cell::Cell<usize>,
    }
    impl MarkStore for BetweenHashes<'_> {
        fn store_name(&self) -> &str {
            self.store.store_name()
        }
        fn content_sha256(&self, budget: &mut MemoryBudget) -> Result<String, L2dError> {
            let hash = self.store.content_sha256(budget)?;
            let calls = self.hashes.get();
            self.hashes.set(calls + 1);
            if calls == 0 {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "concurrent_mark_writer_child", "--nocapture"])
                    .env("L2D_MARK_WRITER_PROJECT", self.root)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Ok(hash)
        }
        fn phrase_ids(
            &self,
            phrase: &str,
            id_bytes: u64,
            budget: &mut MemoryBudget,
            each: &mut dyn FnMut(&str),
        ) -> Result<(), L2dError> {
            self.store.phrase_ids(phrase, id_bytes, budget, each)
        }
        fn text_batches(
            &self,
            beside: u64,
            per_row: u64,
            budget: &mut MemoryBudget,
            each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
        ) -> Result<(), L2dError> {
            self.store.text_batches(beside, per_row, budget, each)
        }
    }
    let scratch = Scratch::new("actual-concurrent-writer");
    let (logs, _) = fixture();
    let store = store_named(&scratch, "s", &logs);
    let before = store.content_sha256(&mut budget()).unwrap();
    let wrapped = BetweenHashes {
        store: &store,
        root: &scratch.root,
        hashes: std::cell::Cell::new(0),
    };
    let error = MarkSet::mark_store(
        "concurrent",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "UNION SELECT".into(),
        },
        &wrapped,
        &mut budget(),
    )
    .unwrap_err();
    assert_eq!(wrapped.hashes.get(), 2);
    let after = store.content_sha256(&mut budget()).unwrap();
    assert_ne!(before, after, "the other process really committed new logs");
    assert!(
        error
            .to_string()
            .contains("changed while `concurrent` was marked"),
        "{error}"
    );
    assert!(error.to_string().contains(&before));
    assert!(error.to_string().contains(&after));
    assert!(!scratch.paths().mark_set_path("concurrent").exists());
}

#[test]
fn phrase_and_regex_mark_the_expected_logs_of_a_store() {
    let scratch = Scratch::new("phrase-regex");
    let (logs, _) = fixture();
    let store = store_named(&scratch, "s", &logs);

    let union = MarkQuery::Phrase {
        phrase: "UNION SELECT".into(),
    };
    let set =
        MarkSet::mark_store("union", Label::Bad, union.clone(), &store, &mut budget()).unwrap();
    let expected = expected_ids(union);
    assert_eq!(expected.len(), 4);
    assert_eq!(set.ids(&budget()).unwrap(), expected);
    assert_eq!(set.count(), 4);
    assert!(set.positions().is_empty());
    let source = set.store().expect("a mark set of a store");
    assert_eq!(source.name, "s");
    assert_eq!(source.content_sha256, hash_of_ids(all_ids(&logs)));
    assert!(set.input().is_none());

    // Byte for byte, as on an input.
    let lower = MarkSet::mark_store(
        "lower",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "union select".into(),
        },
        &store,
        &mut budget(),
    )
    .unwrap();
    assert_eq!(lower.count(), 0);

    let drop = MarkQuery::Regex {
        pattern: DROP_TABLE_PATTERN.into(),
    };
    let expected = expected_ids(drop.clone());
    assert_eq!(expected.len(), 8);
    let set = MarkSet::mark_store("drop", Label::Bad, drop.clone(), &store, &mut budget()).unwrap();
    assert_eq!(set.ids(&budget()).unwrap(), expected);
    // In batches of seven rows the same.
    let small = MarkSet::mark_store(
        "drop",
        Label::Bad,
        drop,
        &store,
        &mut budget_with(None, Some(7)),
    )
    .unwrap();
    assert!(small.try_eq(&set).unwrap());
}

#[test]
fn the_store_is_read_in_batches_of_seq_id_and_text() {
    let scratch = Scratch::new("batches");
    let (logs, _) = fixture();
    let store = store_named(&scratch, "s", &logs);
    let mut budget = budget_with(None, Some(10));
    let held = budget.held_bytes();
    let mut seen: Vec<(i64, String, String)> = Vec::new();
    let mut sizes = Vec::new();
    store
        .text_batches(0, 0, &mut budget, &mut |batch: &DataFrame| {
            let names: Vec<String> = batch
                .get_column_names()
                .iter()
                .map(|name| name.to_string())
                .collect();
            assert_eq!(names, ["seq", "id", "text"]);
            sizes.push(batch.height());
            let seq = batch.column("seq").unwrap().i64().unwrap();
            let id = batch.column("id").unwrap().str().unwrap();
            let text = batch.column("text").unwrap().str().unwrap();
            for ((seq, id), text) in seq
                .into_no_null_iter()
                .zip(id.into_no_null_iter())
                .zip(text.into_no_null_iter())
            {
                seen.push((seq, id.to_string(), text.to_string()));
            }
            Ok(0)
        })
        .unwrap();
    assert_eq!(
        budget.held_bytes(),
        held,
        "the budget holds what it held before"
    );
    assert!(sizes.iter().all(|&size| size <= 10), "{sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 169);
    assert!(
        seen.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "in seq order"
    );
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    assert_eq!(
        seen.iter().map(|row| row.2.clone()).collect::<Vec<_>>(),
        texts,
        "in the order stored"
    );
    assert!(seen.iter().all(|(_, id, text)| *id == log_id(SOURCE, text)));
}

#[test]
fn the_content_hash_follows_the_set_of_logs_not_their_order() {
    let scratch = Scratch::new("hash");
    let (logs, _) = fixture();
    let forward = store_named(&scratch, "forward", &logs);
    let mut reversed: Vec<_> = logs.to_records();
    reversed.reverse();
    let backward = store_named(
        &scratch,
        "backward",
        &LogsFrame::from_records(&reversed).unwrap(),
    );
    let hash = forward.content_sha256(&mut budget()).unwrap();
    assert_eq!(hash, hash_of_ids(all_ids(&logs)));
    assert_eq!(backward.content_sha256(&mut budget()).unwrap(), hash);

    // The ids are read in the order of the unique index: no sort, nothing held.
    let connection = rusqlite::Connection::open(scratch.paths().store_path("forward")).unwrap();
    let mut plan = connection
        .prepare(&format!("EXPLAIN QUERY PLAN {CONTENT_HASH_SQL}"))
        .unwrap();
    let steps: Vec<String> = plan
        .query_map([], |row| row.get(3))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert!(
        steps.iter().all(|step| !step.contains("TEMP B-TREE")),
        "{steps:?}"
    );
    assert!(steps.iter().any(|step| step.contains("INDEX")), "{steps:?}");
}

#[test]
fn a_store_whose_logs_changed_is_refused() {
    let scratch = Scratch::new("changed");
    let (logs, _) = fixture();
    let mut store = store_named(&scratch, "s", &logs);
    let set = MarkSet::mark_store(
        "union",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "UNION SELECT".into(),
        },
        &store,
        &mut budget(),
    )
    .unwrap();
    set.require_store(&store, &mut budget()).unwrap();
    let before = set.store().unwrap().content_sha256.clone();

    // One more log: the set of logs changed.
    let more = LogsFrame::from_records(&[l2d_core::LogRecord {
        text: "GET http://localhost:8080/tienda1/new.jsp".into(),
        label: Label::Good,
        group: None,
        first_position: 1,
        occurrences: 1,
    }])
    .unwrap();
    store
        .add_with(
            &rows(&more),
            VectorOrigin::without_release(&identity()),
            &mut budget(),
        )
        .unwrap();
    let error = set.require_store(&store, &mut budget()).unwrap_err();
    let L2dError::Refused { reason } = error else {
        panic!("expected Refused, got {error:?}");
    };
    let now = store.content_sha256(&mut budget()).unwrap();
    assert_ne!(now, before);
    assert!(
        reason.contains(&before) && reason.contains(&now),
        "{reason}"
    );
    assert!(reason.contains("union"), "{reason}");
}

#[test]
fn a_mark_set_of_one_kind_is_refused_for_the_other() {
    let scratch = Scratch::new("kinds");
    let (logs, summary) = fixture();
    let store = store_named(&scratch, "s", &logs);
    let query = MarkQuery::Phrase {
        phrase: "UNION SELECT".into(),
    };
    let of_store =
        MarkSet::mark_store("of-store", Label::Bad, query.clone(), &store, &mut budget()).unwrap();
    let of_input =
        MarkSet::mark("of-input", Label::Bad, query, &logs, &summary, &budget()).unwrap();

    let error = of_store.require_input(&summary, None).unwrap_err();
    assert!(
        matches!(&error, L2dError::Refused { reason } if reason.contains("store")),
        "{error:?}"
    );
    let error = of_input.require_store(&store, &mut budget()).unwrap_err();
    assert!(
        matches!(&error, L2dError::Refused { reason } if reason.contains("input")),
        "{error:?}"
    );
    // Positions name logs of an input, not of a store.
    let error = MarkSet::mark_store(
        "positions",
        Label::Bad,
        MarkQuery::Positions { positions: vec![1] },
        &store,
        &mut budget(),
    )
    .unwrap_err();
    assert!(matches!(error, L2dError::InvalidInput { .. }), "{error:?}");
}

#[test]
fn a_mark_set_of_a_store_survives_save_and_load_with_its_deselection() {
    let scratch = Scratch::new("save-load");
    let (logs, _) = fixture();
    let store = store_named(&scratch, "s", &logs);
    let paths = scratch.paths();
    let mut set = MarkSet::mark_store(
        "drop",
        Label::Bad,
        MarkQuery::Regex {
            pattern: DROP_TABLE_PATTERN.into(),
        },
        &store,
        &mut budget(),
    )
    .unwrap();
    let first = set.ids(&budget()).unwrap()[0].clone();
    set.deselect_ids(std::slice::from_ref(&first), &budget())
        .unwrap();
    assert_eq!(
        set.deselected_ids(&budget()).unwrap(),
        std::slice::from_ref(&first)
    );
    assert_eq!(set.selected_count(), 7);
    assert_eq!(set.selected_ids().count(), 7);
    // An id that is not marked is refused; positions are not this set's.
    assert!(matches!(
        set.deselect_ids(&["0".repeat(64)], &budget()),
        Err(L2dError::InvalidInput { .. })
    ));
    assert!(matches!(
        set.deselect(&[1]),
        Err(L2dError::InvalidInput { .. })
    ));

    let saved = set.save(&paths, &budget()).unwrap();
    let loaded = MarkSet::load(&paths, "drop", &budget()).unwrap();
    assert!(loaded.try_eq(&set).unwrap());
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&saved).unwrap()).unwrap();
    assert_eq!(json["source"]["kind"], "store");
    assert_eq!(json["source"]["name"], "s");
    assert_eq!(json["count"], 8);
    assert_eq!(json["ids"].as_array().unwrap().len(), 8);
    assert_eq!(json["deselected_ids"][0], first.as_str());
    assert_eq!(json["positions"].as_array().unwrap().len(), 0);
    loaded.require_store(&store, &mut budget()).unwrap();
}

#[test]
fn marking_a_store_refuses_what_does_not_fit() {
    let scratch = Scratch::new("tight");
    let (logs, _) = fixture();
    let store = store_named(&scratch, "s", &logs);
    // The store's own search share and a little: the regex's share does not fit.
    let mut tight = budget_with(
        Some(l2d_core::memory::PROCESS_BASELINE_BYTES + SEARCH_FIXED_BYTES + MIB),
        None,
    );
    let error = MarkSet::mark_store(
        "drop",
        Label::Bad,
        MarkQuery::Regex {
            pattern: DROP_TABLE_PATTERN.into(),
        },
        &store,
        &mut tight,
    )
    .unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");
    assert_eq!(tight.held_bytes(), 0, "nothing stays held after a refusal");
}
