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

#[test]
fn selections_keep_literal_glob_regex_and_trigram_fts_distinct() {
    use l2d_core::selections::{
        ReviewAction, ReviewDecision, ReviewState, SearchQuery, SearchSettings, SelectionSet,
    };
    let scratch = Scratch::new("selections");
    let (logs, _) = fixture();
    let mut store = store_named(&scratch, "selections", &logs);
    let mut b = budget();
    let search = |query, b: &mut MemoryBudget| {
        SelectionSet::search_store("s", query, SearchSettings { max_members: 1000 }, &store, b)
            .unwrap()
    };
    let literal = search(SearchQuery::literal("UNION SELECT"), &mut b);
    assert_eq!(literal.members().len(), 4);
    let glob = search(SearchQuery::glob("UNION SELECT"), &mut b);
    assert!(glob.members().is_empty());
    let regex = search(SearchQuery::regex("UNION.SELECT"), &mut b);
    assert_eq!(regex.members().len(), 8);
    assert!(
        literal
            .members()
            .iter()
            .all(|m| regex.members().contains(m))
    );
    let fts = search(
        SearchQuery::Fts5Trigram {
            expression: "UNION OR DROP".into(),
        },
        &mut b,
    );
    let words = search(SearchQuery::literal("UNION OR DROP"), &mut b);
    assert!(words.members().is_empty());
    assert!(fts.members().len() > 4);
    let short = search(
        SearchQuery::Fts5Trigram {
            expression: "an".into(),
        },
        &mut b,
    );
    assert!(short.members().is_empty());
    assert!(
        !search(SearchQuery::literal("an"), &mut b)
            .members()
            .is_empty()
    );
    assert!(
        SelectionSet::search_store(
            "limit",
            SearchQuery::regex("."),
            SearchSettings { max_members: 1 },
            &store,
            &mut b
        )
        .is_err()
    );
    assert!(
        SelectionSet::search_store(
            "invalid",
            SearchQuery::Fts5Trigram {
                expression: "\"unterminated".into()
            },
            SearchSettings::default(),
            &store,
            &mut b
        )
        .is_err()
    );
    let mut selected = fts;
    let decisions = [
        ReviewDecision {
            member: selected.members()[0].clone(),
            state: ReviewState::Bad,
        },
        ReviewDecision {
            member: selected.members()[1].clone(),
            state: ReviewState::Conflict,
        },
    ];
    let preview = selected.preview_review(&decisions, &b).unwrap();
    selected
        .review_store(
            preview,
            ReviewAction {
                analyst: "reviewer".into(),
                action: "inspected selected records".into(),
            },
            &store,
            &mut b,
        )
        .unwrap();
    let marks = selected
        .export_mark_set("fts-reviewed", Label::Bad, &b)
        .unwrap();
    assert_eq!(marks.selected_count(), 1);
    assert_eq!(marks.deselected_id_count(), selected.members().len() - 1);
    marks.save(&scratch.paths(), &b).unwrap();
    assert!(
        MarkSet::load(&scratch.paths(), "fts-reviewed", &b)
            .unwrap()
            .try_eq(&marks)
            .unwrap()
    );
    let saved = scratch.root.join("selection.json");
    selected.save(&saved, &b).unwrap();
    let loaded = SelectionSet::load(&saved, &b).unwrap();
    assert_eq!(loaded.revision(&b).unwrap(), selected.revision(&b).unwrap());
    loaded.require_store(&store, &mut b).unwrap();
    let extra = l2d_core::LogsFrame::from_records(&[l2d_core::LogRecord {
        text: "new content".into(),
        label: Label::Unknown,
        group: None,
        first_position: 1,
        occurrences: 1,
    }])
    .unwrap();
    store
        .add_with(
            &rows(&extra),
            VectorOrigin::without_release(&identity()),
            &mut b,
        )
        .unwrap();
    assert!(loaded.require_store(&store, &mut b).is_err());
}

#[test]
fn pattern_store_counts_and_revisions_track_real_occurrences() {
    use l2d_core::patterns::{PatternCatalogue, PatternSettings, PatternStore};
    use l2d_core::selections::{
        ReviewAction, ReviewDecision, ReviewState, SearchQuery, SearchSettings, SelectionSet,
    };
    let scratch = Scratch::new("patterns");
    let logs = LogsFrame::from_records(&[l2d_core::LogRecord {
        text: "login failed user=alice".into(),
        label: Label::Unknown,
        group: None,
        first_position: 1,
        occurrences: 3,
    }])
    .unwrap();
    let mut store = store_named(&scratch, "patterns", &logs);
    let id = identity();
    let origin = VectorOrigin::without_release(&id);
    let other = rows_frame(
        &logs,
        "other-source",
        &[1., 0., 0., 0.],
        DIM,
        &[LogFlags::default()],
    )
    .unwrap();
    store.add_with(&other, origin, &mut budget()).unwrap();
    let mut b = budget();
    let mut selection = SelectionSet::search_store(
        "all",
        SearchQuery::literal("login"),
        SearchSettings { max_members: 10 },
        &store,
        &mut b,
    )
    .unwrap();
    assert_eq!(selection.members().len(), 2);
    let decisions: Vec<_> = selection
        .members()
        .iter()
        .enumerate()
        .map(|(i, m)| ReviewDecision {
            member: m.clone(),
            state: if i == 0 {
                ReviewState::Good
            } else {
                ReviewState::Bad
            },
        })
        .collect();
    let preview = selection.preview_review(&decisions, &b).unwrap();
    selection
        .review_store(
            preview,
            ReviewAction {
                analyst: "test".into(),
                action: "explicit differing reviews".into(),
            },
            &store,
            &mut b,
        )
        .unwrap();
    let settings = PatternSettings {
        max_patterns: 8,
        ..Default::default()
    };
    let first = PatternCatalogue::build_store(
        &selection,
        &store,
        &settings,
        &scratch.root.join("first"),
        &mut b,
    )
    .unwrap();
    let counts = &first.patterns[0].counts;
    assert_eq!(
        (
            counts.members,
            counts.occurrences,
            counts.distinct_texts,
            counts.reviewed_texts,
            counts.conflicting_texts
        ),
        (2, 6, 1, 1, 1)
    );
    let snapshot = store.pattern_snapshot(&mut b).unwrap();
    store.add_with(&other, origin, &mut b).unwrap();
    assert_ne!(snapshot, store.pattern_snapshot(&mut b).unwrap());
    let second = PatternCatalogue::build_store(
        &selection,
        &store,
        &settings,
        &scratch.root.join("second"),
        &mut b,
    )
    .unwrap();
    assert_eq!(second.patterns[0].counts.occurrences, 9);
    assert_eq!(second.patterns[0].counts.distinct_texts, 1);
    assert_ne!(first.revision, second.revision);
}

#[test]
fn pattern_store_refuses_count_snapshot_change_without_id_change() {
    use l2d_core::patterns::{PatternCatalogue, PatternSettings, PatternStore};
    use l2d_core::selections::{SearchQuery, SearchSettings, SelectionSet};
    struct Changes<'a> {
        store: &'a Store,
        read: std::cell::Cell<bool>,
    }
    impl MarkStore for Changes<'_> {
        fn store_name(&self) -> &str {
            self.store.store_name()
        }
        fn content_sha256(&self, b: &mut MemoryBudget) -> Result<String, L2dError> {
            self.store.content_sha256(b)
        }
        fn phrase_ids(
            &self,
            p: &str,
            n: u64,
            b: &mut MemoryBudget,
            f: &mut dyn FnMut(&str),
        ) -> Result<(), L2dError> {
            self.store.phrase_ids(p, n, b, f)
        }
        fn text_batches(
            &self,
            x: u64,
            y: u64,
            b: &mut MemoryBudget,
            f: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
        ) -> Result<(), L2dError> {
            self.store.text_batches(x, y, b, f)
        }
    }
    impl PatternStore for Changes<'_> {
        fn pattern_snapshot(&self, b: &mut MemoryBudget) -> Result<String, L2dError> {
            let mut digest = self.store.pattern_snapshot(b)?;
            if self.read.get() {
                digest.replace_range(..1, if digest.starts_with('0') { "1" } else { "0" });
            }
            Ok(digest)
        }
        fn pattern_batches(
            &self,
            x: u64,
            y: u64,
            b: &mut MemoryBudget,
            f: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
        ) -> Result<(), L2dError> {
            self.store.pattern_batches(x, y, b, f)?;
            self.read.set(true);
            Ok(())
        }
    }
    let scratch = Scratch::new("pattern-mutation");
    let (logs, _) = fixture();
    let store = store_named(&scratch, "patterns", &logs);
    let mut b = budget();
    let s = SelectionSet::search_store(
        "all",
        SearchQuery::regex(".*"),
        SearchSettings { max_members: 1000 },
        &store,
        &mut b,
    )
    .unwrap();
    let changed = Changes {
        store: &store,
        read: std::cell::Cell::new(false),
    };
    let dest = scratch.root.join("failed");
    let before = b.held_bytes();
    let error =
        PatternCatalogue::build_store(&s, &changed, &PatternSettings::default(), &dest, &mut b)
            .unwrap_err();
    assert!(error.to_string().contains("snapshot changed"));
    assert!(!dest.exists());
    assert_eq!(before, b.held_bytes());
}

#[test]
fn duplicate_store_texts_reuse_assignment_after_catalogue_overflow() {
    use l2d_core::patterns::{PatternCatalogue, PatternSettings};
    use l2d_core::selections::{SearchQuery, SearchSettings, SelectionSet};
    let scratch = Scratch::new("patterns-overflow");
    let logs = LogsFrame::from_records(
        &[
            "cache shard blue lookup complete",
            "cache shard green lookup complete",
            "cache shard red lookup complete",
            "unrelated rare event",
        ]
        .iter()
        .enumerate()
        .map(|(i, t)| l2d_core::LogRecord {
            text: t.to_string(),
            label: Label::Unknown,
            group: None,
            first_position: i as u64 + 1,
            occurrences: 1,
        })
        .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut store = store_named(&scratch, "patterns", &logs);
    let id = identity();
    let origin = VectorOrigin::without_release(&id);
    for source in ["b", "c", "d"] {
        let rows = rows_frame(&logs, source, &[1.; 16], DIM, &[LogFlags::default(); 4]).unwrap();
        store.add_with(&rows, origin, &mut budget()).unwrap();
    }
    let mut b = budget();
    let selection = SelectionSet::search_store(
        "all",
        SearchQuery::regex(".*"),
        SearchSettings { max_members: 100 },
        &store,
        &mut b,
    )
    .unwrap();
    let c = PatternCatalogue::build_store(
        &selection,
        &store,
        &PatternSettings {
            max_patterns: 1,
            ..Default::default()
        },
        &scratch.root.join("cat"),
        &mut b,
    )
    .unwrap();
    let mut assignments = std::collections::BTreeMap::new();
    c.visit_members(&scratch.root.join("cat"), &mut b, |m| {
        if let Some(prior) = assignments.insert(m.text, m.pattern) {
            assert_eq!(prior, m.pattern);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(assignments.len(), 4);
    assert_eq!(c.selected_members, 16);
    assert_eq!(c.patterns[0].counts.members + c.ungrouped.members, 16);
    assert_eq!(
        c.patterns[0].counts.distinct_texts + c.ungrouped.distinct_texts,
        4
    );
}

#[test]
fn refined_store_snapshot_binds_occurrences_and_conflicting_text_reviews() {
    use l2d_core::patterns::{PatternCatalogue, PatternSettings, refinement::PatternRefinement};
    use l2d_core::selections::{
        ReviewAction, ReviewDecision, ReviewState, SearchQuery, SearchSettings, SelectionSet,
    };
    let scratch = Scratch::new("refinement-counts");
    let (logs, _) = fixture();
    let mut store = store_named(&scratch, "refine", &logs);
    let mut b = budget();
    let mut selection = SelectionSet::search_store(
        "all",
        SearchQuery::glob("*"),
        SearchSettings { max_members: 1000 },
        &store,
        &mut b,
    )
    .unwrap();
    let decisions: Vec<_> = selection
        .members()
        .iter()
        .enumerate()
        .map(|(i, m)| ReviewDecision {
            member: m.clone(),
            state: if i % 3 == 0 {
                ReviewState::Conflict
            } else if i % 3 == 1 {
                ReviewState::Good
            } else {
                ReviewState::Unresolved
            },
        })
        .collect();
    let preview = selection.preview_review(&decisions, &b).unwrap();
    selection
        .review_store(
            preview,
            ReviewAction {
                analyst: "fixture".into(),
                action: "explicit mixed review".into(),
            },
            &store,
            &mut b,
        )
        .unwrap();
    let c = PatternCatalogue::build_store(
        &selection,
        &store,
        &PatternSettings::default(),
        &scratch.root.join("catalogue"),
        &mut b,
    )
    .unwrap();
    let r = PatternRefinement::from_catalogue(&c, &scratch.root.join("catalogue"), &mut b).unwrap();
    r.require_current_store(&store, &mut b).unwrap();
    r.require_selection(&selection, &b).unwrap();
    let ids: Vec<_> = r.groups.iter().map(|p| p.id).collect();
    let merged = r.merge(&ids, &b).unwrap();
    assert_eq!(
        merged
            .rows
            .iter()
            .filter(|r| r.review == Some(ReviewState::Conflict))
            .count(),
        decisions
            .iter()
            .filter(|r| r.state == ReviewState::Conflict)
            .count()
    );
    assert_eq!(
        merged.groups[0].counts.conflicting_texts,
        r.groups
            .iter()
            .map(|p| p.counts.conflicting_texts)
            .sum::<u64>()
    );
    let before = store.content_sha256(&mut b).unwrap();
    store
        .add_with(
            &rows(&logs),
            VectorOrigin::without_release(&identity()),
            &mut b,
        )
        .unwrap();
    assert_eq!(
        before,
        store.content_sha256(&mut b).unwrap(),
        "duplicate input preserves store ID hash"
    );
    assert!(
        r.require_current_store(&store, &mut b).is_err(),
        "occurrence changes must invalidate refinement snapshot"
    );
}
