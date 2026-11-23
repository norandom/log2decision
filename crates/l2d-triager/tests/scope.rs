//! Triager task 1.1: one scope over a file, selected archive entries or a store, and
//! exact joins of existing selection, score and policy keys to its originals
//! (requirements 1.1, 1.2, 1.3 and 6.2).

use std::io::Write;
use std::path::{Path, PathBuf};

use l2d_core::selections::{Member, SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{
    ArchivePosition, EncoderIdentity, InputAdapter, L2dError, Label, LabelledAdapter,
    LabelledFormat, LinesAdapter, LogRecord, LogsFrame, MarkedSource, MemoryBudget, MemorySettings,
    ProjectPaths, ZipInputSettings,
};
use l2d_decide::policy::PolicyReport;
use l2d_store::{LogFlags, Store, VectorOrigin, log_id, rows_frame};
use l2d_triager::scope::{
    IndexPlacement, JoinError, Locator, NativeKey, Original, OriginalId, Scope, ScopeSettings, Unit,
};
use serde_json::{Value, json};

const GIB: u64 = 1024 * 1024 * 1024;

fn budget() -> MemoryBudget {
    budget_of(None)
}

fn budget_of(max_batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        max_batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).unwrap()
}

fn archive(dir: &Path, name: &str, entries: &[(&str, &[u8])]) -> PathBuf {
    let path = dir.join(name);
    let mut writer = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    for (entry, content) in entries {
        writer
            .start_file(*entry, zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(content).unwrap();
    }
    writer.finish().unwrap();
    path
}

fn file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    path
}

fn settings(dir: &Path, name: &str) -> ScopeSettings {
    ScopeSettings::new(dir.join(name))
}

/// What a join handed out, as plain values to compare between runs.
type Joined = Vec<(u64, u64, String)>;

fn joined_selection(scope: &Scope, selection: &SelectionSet) -> Result<Joined, JoinError> {
    let mut rows = Joined::new();
    let report = scope.join_selection(selection, &mut budget(), |_, original| {
        rows.push((
            original.id.0,
            original.occurrences,
            original.native_id.to_hex(),
        ));
        Ok(())
    })?;
    assert_eq!(report.keys, rows.len() as u64);
    assert_eq!(report.originals, rows.len() as u64);
    assert_eq!(
        report.occurrences,
        rows.iter().map(|row| row.1).sum::<u64>()
    );
    Ok(rows)
}

fn joined_policy(scope: &Scope, report: &PolicyReport) -> Result<Joined, JoinError> {
    let mut rows = Joined::new();
    scope.join_policy(report, &mut budget(), |decision, original| {
        assert_eq!(decision.input.id, original.native_id.to_hex());
        rows.push((
            original.id.0,
            original.occurrences,
            original.native_id.to_hex(),
        ));
        Ok(())
    })?;
    Ok(rows)
}

/// A saved policy report over `source` with one decision per text, as the CLI writes
/// it: every ID is `log_id` of the source path and the original text.
fn policy_report(source: Option<Value>, decisions: &[(String, &str)]) -> PolicyReport {
    let decisions: Vec<Value> = decisions
        .iter()
        .map(|(id, text)| {
            json!({
                "input": {"id": id, "text": text, "group": null},
                "score": {"kind": "mark-set", "revision": "scorer-v1", "comparison_class_assumed": true,
                          "diagnostic_ratio": 0.75, "actionable_ratio": null},
                "vector": null, "queue": "Review", "reason": "Unsupported",
                "explanation": "incompatible exploration profile; actionable ratio withheld",
                "allocation": null
            })
        })
        .collect();
    let mut report = json!({
        "format": 1,
        "policy": {
            "version": "test-v1", "selection_provenance": "synthetic test", "low": 0.2, "high": 0.8,
            "corroboration": 1,
            "reach": {"radius": 0.2, "top_k": 2, "minimum_labelled_support": 1, "expected_profile": "expected"},
            "require_calibration": false, "review_budget": 4, "unknown_slots": 1, "audit_slots": 1, "audit_seed": 5
        },
        "eligibility": l2d_decide::policy::ELIGIBILITY,
        "allocation_rule": l2d_decide::policy::ALLOCATION,
        "scorer_revision": "scorer-v1", "evidence_revision": "history-v1", "actual_profile": "incompatible",
        "calibration": "Missing",
        "qualification": "Exploratory; qualification incomplete. Policy mechanics and local support do not establish deployment quality.",
        "decisions": decisions,
        "unknown": {"configured": 1, "selected": 0, "unused": 1},
        "general": {"configured": 2, "selected": 0, "unused": 2},
        "audit": {"configured": 1, "selected": 0, "unused": 1},
        "deferred": 0
    });
    if let Some(source) = source {
        report["query_source"] = source;
    }
    serde_json::from_value(report).unwrap()
}

/// The query source the CLI records for a policy over the scope's own source: the
/// path as the reader was given it, and the scope's bytes, filter and interpretation.
fn policy_source(scope: &Scope, path: &Path) -> Value {
    let MarkedSource::Input(input) = scope.source() else {
        panic!("a policy names a file or an archive");
    };
    json!({
        "path": path.to_string_lossy(),
        "content_sha256": input.content_sha256,
        "archive_entries": input.archive_entries,
        "interpretation": input.interpretation,
    })
}

fn open_archive(path: &Path, zip: &ZipInputSettings, settings: &ScopeSettings) -> Scope {
    Scope::open_archive(path, zip, settings, &mut budget())
        .unwrap()
        .0
}

fn search_archive(path: &Path, zip: &ZipInputSettings, query: SearchQuery) -> SelectionSet {
    SelectionSet::search_archive(
        "found",
        query,
        SearchSettings::default(),
        path,
        zip,
        &mut budget(),
    )
    .unwrap()
    .0
}

#[test]
fn duplicate_texts_across_archive_entries_are_one_original_with_its_first_locator_and_exact_count()
{
    let dir = tempfile::tempdir().unwrap();
    let path = archive(
        dir.path(),
        "logs.zip",
        &[
            (
                "a.log",
                b"login failed user=a\nlogin failed user=b\nlogin failed user=a\n",
            ),
            ("b.log", b"ping\nlogin failed user=a\nlogin failed user=c\n"),
        ],
    );
    let zip = ZipInputSettings::default();
    let (scope, summary) =
        Scope::open_archive(&path, &zip, &settings(dir.path(), "work"), &mut budget()).unwrap();
    assert_eq!(scope.originals(), 4);
    assert_eq!(scope.occurrences(), 6);
    assert_eq!(summary.input.records, 6);
    let completeness = scope.completeness();
    assert!(completeness.complete);
    assert_eq!(completeness.records_scanned, 6);
    assert_eq!(completeness.occurrences, 6);
    assert_eq!(completeness.originals, 4);
    assert_eq!(completeness.entries_read, 2);

    let selection = search_archive(&path, &zip, SearchQuery::literal("login failed"));
    let mut found: Vec<Original> = Vec::new();
    let report = scope
        .join_selection(&selection, &mut budget(), |member, original| {
            assert_eq!(*member, Member::Position(original.id.0));
            found.push(original.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(
        (report.keys, report.originals, report.occurrences),
        (3, 3, 5)
    );
    // The text of both entries is one original, located where it was first read.
    let repeated = &found[0];
    assert_eq!(repeated.first, Locator::ArchiveLine { entry: 0, line: 1 });
    assert_eq!(repeated.occurrences, 3);
    assert_eq!(
        repeated.native_id.to_hex(),
        log_id(&path.to_string_lossy(), "login failed user=a")
    );
    assert_eq!(found[1].first, Locator::ArchiveLine { entry: 0, line: 2 });
    assert_eq!(found[1].occurrences, 1);
    assert_eq!(found[2].first, Locator::ArchiveLine { entry: 1, line: 3 });
    // A handle of this scope names the same original again.
    assert_eq!(
        scope.original(repeated.id).unwrap().as_ref(),
        Some(repeated)
    );
    // A later occurrence is no original of its own: its position resolves to nothing.
    let later = ArchivePosition { entry: 1, line: 2 }.pack();
    assert_eq!(scope.original(OriginalId(later)).unwrap(), None);
    let error = scope
        .join_keys(
            scope.source(),
            [NativeKey::Position(later)],
            &mut budget(),
            |_, _| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(error, JoinError::Unknown { .. }), "{error}");
}

/// Three entries with texts repeated inside and across them.
fn mixed_archive(dir: &Path) -> PathBuf {
    let entry = |from: u64, rows: u64| -> Vec<u8> {
        (from..from + rows)
            .map(|index| {
                let n = index % 2_500;
                format!(
                    "{} request {n} from 203.0.113.{}\n",
                    if n % 7 == 0 { "denied" } else { "served" },
                    n % 250
                )
            })
            .collect::<String>()
            .into_bytes()
    };
    archive(
        dir,
        "mixed.zip",
        &[
            ("first.log", &entry(0, 3_000)),
            ("second.log", &entry(1_500, 3_000)),
            ("third.log", &entry(0, 4_000)),
        ],
    )
}

#[test]
fn joins_are_exact_across_batch_sizes_readers_dedup_spill_and_index_placement() {
    let dir = tempfile::tempdir().unwrap();
    let path = mixed_archive(dir.path());
    let zip = ZipInputSettings::default();
    let selection = search_archive(&path, &zip, SearchQuery::regex(r"^denied request \d+5 "));
    assert!(selection.members().len() > 20);
    // Score part rows name an entry and a line; the first entry introduces 2,500 texts.
    let score_rows: Vec<u64> = (1..=2_500u64)
        .step_by(11)
        .map(|line| ArchivePosition { entry: 0, line }.pack())
        .collect();

    let mut expected: Option<(String, u64, u64, Joined, Joined, Joined)> = None;
    let mut run = 0;
    let mut on_disk = 0;
    for max_batch_rows in [None, Some(64), Some(1_000)] {
        for entries_at_once in [Some(1), Some(2)] {
            for dedup_set_texts in [None, Some(100)] {
                for index in [IndexPlacement::MemoryWhenAdmitted, IndexPlacement::Disk] {
                    run += 1;
                    let zip = ZipInputSettings {
                        entries_at_once,
                        ..ZipInputSettings::default()
                    };
                    let settings = ScopeSettings {
                        index,
                        dedup_set_texts,
                        ..settings(dir.path(), &format!("work-{run}"))
                    };
                    let mut budget = budget_of(max_batch_rows);
                    let (scope, _) =
                        Scope::open_archive(&path, &zip, &settings, &mut budget).unwrap();
                    assert_eq!(budget.held_bytes(), 0, "nothing stays held in the budget");
                    assert_eq!(scope.index_on_disk(), index == IndexPlacement::Disk);
                    assert_eq!(scope.dedup_on_disk(), dedup_set_texts.is_some());
                    assert_eq!(scope.archive_readers(), entries_at_once);
                    on_disk += u32::from(scope.index_on_disk());

                    let policy = policy_report(
                        Some(policy_source(&scope, &path)),
                        &[7u64, 0, 2_499, 1_234]
                            .map(|n| {
                                let text = format!(
                                    "{} request {n} from 203.0.113.{}",
                                    if n % 7 == 0 { "denied" } else { "served" },
                                    n % 250
                                );
                                (log_id(&path.to_string_lossy(), &text), text)
                            })
                            .iter()
                            .map(|(id, text)| (id.clone(), text.as_str()))
                            .collect::<Vec<_>>(),
                    );
                    let mut scores = Joined::new();
                    scope
                        .join_keys(
                            scope.source(),
                            score_rows.iter().map(|row| NativeKey::Position(*row)),
                            &mut budget,
                            |index, original| {
                                assert_eq!(original.id.0, score_rows[index as usize]);
                                scores.push((
                                    original.id.0,
                                    original.occurrences,
                                    original.native_id.to_hex(),
                                ));
                                Ok(())
                            },
                        )
                        .unwrap();
                    let found = (
                        scope.handle().to_string(),
                        scope.originals(),
                        scope.occurrences(),
                        joined_selection(&scope, &selection).unwrap(),
                        joined_policy(&scope, &policy).unwrap(),
                        scores,
                    );
                    match &expected {
                        None => {
                            assert_eq!((found.1, found.2), (2_500, 10_000));
                            assert_eq!(found.3.len(), selection.members().len());
                            assert_eq!(found.4.len(), 4);
                            expected = Some(found);
                        }
                        Some(expected) => assert_eq!(
                            &found, expected,
                            "batch {max_batch_rows:?}, readers {entries_at_once:?}, dedup \
                             {dedup_set_texts:?}, index {index:?}"
                        ),
                    }
                }
            }
        }
    }
    assert_eq!(on_disk, 12);
}

#[test]
fn identical_text_in_unrelated_snapshots_is_not_joined() {
    let dir = tempfile::tempdir().unwrap();
    let content = "shared text\nonly here\n";
    let first = file(dir.path(), "first.log", content);
    let second = file(dir.path(), "second.log", content);
    let open = |path: &Path, work: &str| {
        Scope::open_file(
            path,
            &LinesAdapter,
            &settings(dir.path(), work),
            &mut budget(),
        )
        .unwrap()
        .0
    };
    let (one, two) = (open(&first, "one"), open(&second, "two"));
    assert_ne!(one.handle(), two.handle());
    // The same bytes read the same way: only the source identity differs.
    let key = |scope: &Scope| scope.original(OriginalId(1)).unwrap().unwrap();
    assert_eq!(key(&one).occurrences, key(&two).occurrences);
    assert_ne!(key(&one).native_id, key(&two).native_id);

    let (logs, summary, _) = LinesAdapter.read_with(&second, &mut budget()).unwrap();
    let selection = SelectionSet::search_input(
        "shared",
        SearchQuery::literal("shared"),
        SearchSettings::default(),
        &logs,
        &summary,
        &budget(),
    )
    .unwrap();
    assert_eq!(joined_selection(&two, &selection).unwrap().len(), 1);
    let error = joined_selection(&one, &selection).unwrap_err();
    assert!(matches!(error, JoinError::Path { .. }), "{error}");

    let report = policy_report(
        Some(policy_source(&two, &second)),
        &[(
            log_id(&second.to_string_lossy(), "shared text"),
            "shared text",
        )],
    );
    assert_eq!(joined_policy(&two, &report).unwrap().len(), 1);
    let error = joined_policy(&one, &report).unwrap_err();
    assert!(matches!(error, JoinError::PolicyPath { .. }), "{error}");
    assert!(error.to_string().contains("replay"), "{error}");
}

#[test]
fn quoted_multiline_csv_joins_by_logical_record_not_by_physical_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(
        dir.path(),
        "cases.csv",
        "text,label\n\"first line\nsecond line\",bad\nplain,good\n\"another\nmulti\nline\",bad\nplain,good\n",
    );
    let adapter = LabelledAdapter::new(LabelledFormat::Csv);
    let (scope, summary) = Scope::open_file(
        &path,
        &adapter,
        &settings(dir.path(), "work"),
        &mut budget(),
    )
    .unwrap();
    assert_eq!(summary.records, 4);
    assert_eq!((scope.originals(), scope.occurrences()), (3, 4));

    let (logs, summary, _) = adapter.read_with(&path, &mut budget()).unwrap();
    let selection = SelectionSet::search_input(
        "another",
        SearchQuery::literal("another"),
        SearchSettings::default(),
        &logs,
        &summary,
        &budget(),
    )
    .unwrap();
    // The third logical record starts on the fifth physical line.
    assert_eq!(selection.members(), [Member::Position(3)]);
    let mut found = Vec::new();
    scope
        .join_selection(&selection, &mut budget(), |_, original| {
            found.push(original.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].first, Locator::FileRecord { position: 3 });
    assert_eq!(
        found[0].native_id.to_hex(),
        log_id(&path.to_string_lossy(), "another\nmulti\nline")
    );
    let plain = scope.original(OriginalId(2)).unwrap().unwrap();
    assert_eq!(plain.occurrences, 2);
    // Physical line numbers are not record positions.
    for line in [5, 8] {
        let error = scope
            .join_keys(
                scope.source(),
                [NativeKey::Position(line)],
                &mut budget(),
                |_, _| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(error, JoinError::Unknown { .. }), "{error}");
    }
}

#[test]
fn changed_options_are_another_scope_and_refuse_keys_made_under_the_old_ones() {
    let dir = tempfile::tempdir().unwrap();
    let path = archive(
        dir.path(),
        "logs.zip",
        &[
            ("a.log", b"alpha one\nalpha two\n"),
            ("b.log", b"beta one\nalpha one\n"),
        ],
    );
    let all = ZipInputSettings::default();
    let only_a = ZipInputSettings {
        entries: vec!["a.log".into()],
        ..ZipInputSettings::default()
    };
    let shorter = ZipInputSettings {
        max_record_bytes: 4_096,
        ..ZipInputSettings::default()
    };
    let whole = open_archive(&path, &all, &settings(dir.path(), "all"));
    let filtered = open_archive(&path, &only_a, &settings(dir.path(), "only-a"));
    let limited = open_archive(&path, &shorter, &settings(dir.path(), "shorter"));
    assert_ne!(whole.handle(), filtered.handle());
    assert_ne!(whole.handle(), limited.handle());
    assert_eq!((whole.originals(), whole.occurrences()), (3, 4));
    assert_eq!((filtered.originals(), filtered.occurrences()), (2, 2));
    assert_eq!(filtered.completeness().entries_excluded, 1);
    // The same options give the same handle again.
    let again = open_archive(&path, &all, &settings(dir.path(), "again"));
    assert_eq!(whole.handle(), again.handle());

    let selection = search_archive(&path, &all, SearchQuery::literal("alpha"));
    assert_eq!(joined_selection(&whole, &selection).unwrap().len(), 2);
    let error = joined_selection(&filtered, &selection).unwrap_err();
    assert!(matches!(error, JoinError::Filters { .. }), "{error}");
    let error = joined_selection(&limited, &selection).unwrap_err();
    assert!(matches!(error, JoinError::Interpretation { .. }), "{error}");

    // A file read with other columns is another interpretation of the same bytes.
    let csv = file(
        dir.path(),
        "cases.csv",
        "text,note,label\nfirst,alpha,bad\nsecond,alpha,good\n",
    );
    let by_text = LabelledAdapter::new(LabelledFormat::Csv);
    let by_note = LabelledAdapter::new(LabelledFormat::Csv).with_text_column("note");
    let open = |adapter: &LabelledAdapter, work: &str| {
        Scope::open_file(&csv, adapter, &settings(dir.path(), work), &mut budget())
            .unwrap()
            .0
    };
    let (texts, notes) = (open(&by_text, "texts"), open(&by_note, "notes"));
    assert_ne!(texts.handle(), notes.handle());
    assert_eq!((texts.originals(), notes.originals()), (2, 1));
    let (logs, summary, _) = by_text.read_with(&csv, &mut budget()).unwrap();
    let selection = SelectionSet::search_input(
        "second",
        SearchQuery::literal("second"),
        SearchSettings::default(),
        &logs,
        &summary,
        &budget(),
    )
    .unwrap();
    assert_eq!(joined_selection(&texts, &selection).unwrap().len(), 1);
    let error = joined_selection(&notes, &selection).unwrap_err();
    assert!(matches!(error, JoinError::Interpretation { .. }), "{error}");
}

#[test]
fn changed_and_moved_files_are_surfaced_as_incompatible() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "auth.log", "login failed\nlogin ok\n");
    let open = |path: &Path, work: &str| {
        Scope::open_file(
            path,
            &LinesAdapter,
            &settings(dir.path(), work),
            &mut budget(),
        )
        .unwrap()
        .0
    };
    let before = open(&path, "before");
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut budget()).unwrap();
    let selection = SelectionSet::search_input(
        "failed",
        SearchQuery::literal("failed"),
        SearchSettings::default(),
        &logs,
        &summary,
        &budget(),
    )
    .unwrap();
    let report = policy_report(
        Some(policy_source(&before, &path)),
        &[(
            log_id(&path.to_string_lossy(), "login failed"),
            "login failed",
        )],
    );
    assert_eq!(joined_selection(&before, &selection).unwrap().len(), 1);
    assert_eq!(joined_policy(&before, &report).unwrap().len(), 1);

    // The moved file has the same bytes, but policy IDs name the old path.
    let moved = dir.path().join("moved.log");
    std::fs::copy(&path, &moved).unwrap();
    let elsewhere = open(&moved, "moved");
    let error = joined_policy(&elsewhere, &report).unwrap_err();
    assert!(matches!(error, JoinError::PolicyPath { .. }), "{error}");

    // The changed file is another revision under the same path.
    std::fs::write(&path, "login failed\nlogin ok\nlogin failed again\n").unwrap();
    let after = open(&path, "after");
    assert_ne!(before.handle(), after.handle());
    for error in [
        joined_selection(&after, &selection).unwrap_err(),
        joined_policy(&after, &report).unwrap_err(),
    ] {
        assert!(matches!(error, JoinError::SourceChanged { .. }), "{error}");
        let L2dError::Refused { reason } = L2dError::from(error) else {
            panic!("an incompatible mapping is a refusal");
        };
        assert!(reason.contains(&summary.content_sha256), "{reason}");
    }
}

#[test]
fn policy_keys_are_verified_against_path_text_and_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(
        dir.path(),
        "auth.log",
        "login failed\nlogin ok\nlogin failed\n",
    );
    let scope = Scope::open_file(
        &path,
        &LinesAdapter,
        &settings(dir.path(), "work"),
        &mut budget(),
    )
    .unwrap()
    .0;
    let source = || Some(policy_source(&scope, &path));
    let id = |text: &str| log_id(&path.to_string_lossy(), text);

    let good = policy_report(
        source(),
        &[
            (id("login ok"), "login ok"),
            (id("login failed"), "login failed"),
        ],
    );
    let rows = joined_policy(&scope, &good).unwrap();
    // Decisions keep the report's order; each names one original with its count.
    assert_eq!(rows, [(2, 1, id("login ok")), (1, 2, id("login failed"))]);

    let unbound = policy_report(None, &[(id("login ok"), "login ok")]);
    let error = joined_policy(&scope, &unbound).unwrap_err();
    assert!(matches!(error, JoinError::Unverified { .. }), "{error}");

    let mut legacy = policy_source(&scope, &path);
    legacy.as_object_mut().unwrap().remove("interpretation");
    let error = joined_policy(
        &scope,
        &policy_report(Some(legacy), &[(id("login ok"), "login ok")]),
    )
    .unwrap_err();
    assert!(matches!(error, JoinError::Unverified { .. }), "{error}");

    let drifted = policy_report(source(), &[(id("login ok"), "login failed")]);
    let error = joined_policy(&scope, &drifted).unwrap_err();
    assert!(matches!(error, JoinError::Drift { .. }), "{error}");

    let outside = policy_report(source(), &[(id("never read"), "never read")]);
    let error = joined_policy(&scope, &outside).unwrap_err();
    assert!(matches!(error, JoinError::Unknown { .. }), "{error}");

    let twice = policy_report(
        source(),
        &[(id("login ok"), "login ok"), (id("login ok"), "login ok")],
    );
    let error = joined_policy(&scope, &twice).unwrap_err();
    assert!(
        matches!(
            error,
            JoinError::Ambiguous {
                original: OriginalId(2),
                ..
            }
        ),
        "{error}"
    );
}

fn identity() -> EncoderIdentity {
    EncoderIdentity {
        name: "bytecnn".to_string(),
        revision: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        dimension: 4,
    }
}

fn store(root: &Path, name: &str, rows: &[(&str, u32)]) -> Store {
    let records: Vec<LogRecord> = rows
        .iter()
        .enumerate()
        .map(|(index, (text, occurrences))| LogRecord {
            text: (*text).to_string(),
            label: Label::Unknown,
            group: None,
            first_position: index as u64 + 1,
            occurrences: *occurrences,
        })
        .collect();
    let logs = LogsFrame::from_records(&records).unwrap();
    let vectors: Vec<f32> = rows.iter().flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let flags = vec![LogFlags::default(); rows.len()];
    let frame = rows_frame(&logs, "synthetic", &vectors, 4, &flags).unwrap();
    let identity = identity();
    let origin = VectorOrigin::without_release(&identity);
    let paths = ProjectPaths::new(root);
    Store::create(&paths, name, origin)
        .unwrap()
        .add_with(&frame, origin, &mut budget())
        .unwrap();
    Store::open(&paths, name).unwrap()
}

#[test]
fn a_store_scope_resolves_native_store_ids_within_its_pinned_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let rows = [("login failed", 3), ("login ok", 1), ("ping", 7)];
    let first = store(dir.path(), "first", &rows);
    let mut budget = budget();
    let scope = Scope::open_store(&first, &settings(dir.path(), "work"), &mut budget).unwrap();
    assert_eq!(budget.held_bytes(), 0);
    assert_eq!((scope.originals(), scope.occurrences()), (3, 11));
    assert!(matches!(scope.source(), MarkedSource::Store(_)));

    let selection = SelectionSet::search_store(
        "login",
        SearchQuery::literal("login"),
        SearchSettings::default(),
        &first,
        &mut budget,
    )
    .unwrap();
    let mut found = Vec::new();
    scope
        .join_selection(&selection, &mut budget, |member, original| {
            assert_eq!(*member, Member::StoreId(original.native_id.to_hex()));
            found.push(original.clone());
            Ok(())
        })
        .unwrap();
    found.sort_by_key(|original| original.id);
    assert_eq!(found.len(), 2);
    assert_eq!(
        found[0].native_id.to_hex(),
        log_id("synthetic", "login failed")
    );
    assert_eq!(found[0].occurrences, 3);
    assert!(matches!(found[0].first, Locator::StoreRow { .. }));
    assert_eq!(found[1].occurrences, 1);

    // A position names a record of an input, never a stored log.
    let error = scope
        .join_keys(
            scope.source(),
            [NativeKey::Position(1)],
            &mut budget,
            |_, _| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(error, JoinError::KeyKind { .. }), "{error}");
    // The existing policy accepts file and archive queries only.
    let report = policy_report(None, &[(log_id("synthetic", "ping"), "ping")]);
    let error = joined_policy(&scope, &report).unwrap_err();
    assert!(
        matches!(error, JoinError::UnsupportedProducer { .. }),
        "{error}"
    );

    // Another snapshot of the same logs plus one: the old selection is refused.
    let mut more = rows.to_vec();
    more.push(("login denied", 1));
    let second = store(dir.path(), "second", &more);
    let wider = Scope::open_store(&second, &settings(dir.path(), "wider"), &mut budget).unwrap();
    assert_ne!(scope.handle(), wider.handle());
    let error = joined_selection(&wider, &selection).unwrap_err();
    assert!(matches!(error, JoinError::SourceChanged { .. }), "{error}");
    // A store key against a file scope is another kind of source.
    let path = file(dir.path(), "auth.log", "login failed\n");
    let lines = Scope::open_file(
        &path,
        &LinesAdapter,
        &settings(dir.path(), "lines"),
        &mut budget,
    )
    .unwrap()
    .0;
    let error = joined_selection(&lines, &selection).unwrap_err();
    assert!(matches!(error, JoinError::SourceKind { .. }), "{error}");
}

#[test]
fn skipped_incomplete_and_unsupported_input_is_counted_with_units_and_reasons() {
    let dir = tempfile::tempdir().unwrap();
    let long = "x".repeat(200);
    let path = archive(
        dir.path(),
        "logs.zip",
        &[
            ("a.log", format!("kept\n{long}\nkept\n").as_bytes()),
            ("b.bin", b"\x00\x01\x02 not text\n"),
            ("c.txt", b"outside the filter\n"),
        ],
    );
    let zip = ZipInputSettings {
        entries: vec!["a.log".into(), "b.bin".into()],
        max_record_bytes: 64,
        ..ZipInputSettings::default()
    };
    let (scope, summary) =
        Scope::open_archive(&path, &zip, &settings(dir.path(), "work"), &mut budget()).unwrap();
    let completeness = scope.completeness();
    assert!(!completeness.complete);
    assert_eq!(completeness.records_scanned, summary.input.records);
    assert_eq!(completeness.skipped_records, 1);
    assert_eq!((completeness.originals, completeness.occurrences), (1, 2));
    assert_eq!(completeness.entries_read, 1);
    assert_eq!(completeness.entries_unsupported, 1);
    assert_eq!(completeness.entries_incomplete, 0);
    assert_eq!(completeness.entries_excluded, 1);
    let reason = |unit: Unit| {
        completeness
            .reasons
            .iter()
            .filter(|reason| reason.unit == unit)
            .map(|reason| (reason.count, reason.reason.as_str()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        reason(Unit::Records),
        [(1, summary.input.skipped[0].reason.as_str())]
    );
    assert_eq!(
        reason(Unit::Entries),
        [(1, "matches no entry pattern"), (1, "not text")]
    );
}

#[test]
fn a_scope_without_room_on_disk_is_refused_with_both_figures() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "auth.log", "login failed\nlogin ok\n");
    let settings = ScopeSettings {
        free_disk_bytes: Some(1_000),
        ..settings(dir.path(), "work")
    };
    let error = Scope::open_file(&path, &LinesAdapter, &settings, &mut budget()).unwrap_err();
    let L2dError::Refused { reason } = error else {
        panic!("{error:?}");
    };
    assert!(reason.contains("(1000 bytes)"), "{reason}");
    assert!(reason.contains("locator index"), "{reason}");
    assert!(!dir.path().join("work").exists(), "nothing is left behind");

    // A work directory that holds something is never written into.
    std::fs::create_dir_all(dir.path().join("used")).unwrap();
    std::fs::write(dir.path().join("used").join("other"), "kept").unwrap();
    let error = Scope::open_file(
        &path,
        &LinesAdapter,
        &ScopeSettings::new(dir.path().join("used")),
        &mut budget(),
    )
    .unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("used").join("other")).unwrap(),
        "kept"
    );
}

#[test]
fn a_join_without_room_for_its_bits_notes_them_on_disk_and_resolves_the_same_originals() {
    use l2d_core::memory::PROCESS_BASELINE_BYTES;
    use l2d_triager::scope::JOIN_FIXED_BYTES;

    let dir = tempfile::tempdir().unwrap();
    let path = mixed_archive(dir.path());
    let zip = ZipInputSettings::default();
    let scope = open_archive(&path, &zip, &settings(dir.path(), "work"));
    let keys = || {
        (1..=2_500u64).map(|line| NativeKey::Position(ArchivePosition { entry: 0, line }.pack()))
    };
    let join = |budget: &mut MemoryBudget| {
        let mut rows = Joined::new();
        scope
            .join_keys(scope.source(), keys(), budget, |_, original| {
                rows.push((
                    original.id.0,
                    original.occurrences,
                    original.native_id.to_hex(),
                ));
                Ok(())
            })
            .map(|report| (report, rows))
    };
    let (wide_report, wide) = join(&mut budget()).unwrap();
    assert_eq!((wide_report.keys, wide_report.occurrences), (2_500, 10_000));

    // 2,500 originals take 313 bytes of bits: 100 bytes beside the fixed share do not.
    let usable = 64 * 1024 * 1024;
    let tight = |left: u64| {
        let settings = MemorySettings {
            memory_limit_bytes: Some(PROCESS_BASELINE_BYTES + usable),
            ..MemorySettings::default()
        };
        let mut budget = MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).unwrap();
        budget.hold(usable - left);
        budget
    };
    let mut budget = tight(JOIN_FIXED_BYTES + 100);
    assert!(!budget.usage().report().spilled);
    let (report, rows) = join(&mut budget).unwrap();
    assert!(budget.usage().report().spilled, "the bits went to disk");
    assert_eq!(budget.held_bytes(), usable - JOIN_FIXED_BYTES - 100);
    assert_eq!((report, rows), (wide_report, wide));
    // A repeated key is found on disk as it is in memory.
    let error = scope
        .join_keys(
            scope.source(),
            keys().chain(keys().take(1)),
            &mut tight(JOIN_FIXED_BYTES + 100),
            |_, _| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(error, JoinError::Ambiguous { .. }), "{error}");

    // Without the fixed share the join is refused with the figures, and nothing is held.
    let mut budget = tight(JOIN_FIXED_BYTES - 1);
    let error = join(&mut budget).unwrap_err();
    let JoinError::Failed(L2dError::Refused { reason }) = error else {
        panic!("{error}");
    };
    assert!(
        reason.contains("(4096 bytes)") && reason.contains("(4095 bytes)"),
        "{reason}"
    );
    assert_eq!(budget.held_bytes(), usable - JOIN_FIXED_BYTES + 1);
}

#[test]
fn an_original_whose_count_reached_the_readers_ceiling_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let flooded = store(
        dir.path(),
        "flooded",
        &[("login ok", 1), ("flood", u32::MAX)],
    );
    let error =
        Scope::open_store(&flooded, &settings(dir.path(), "work"), &mut budget()).unwrap_err();
    let L2dError::Refused { reason } = error else {
        panic!("{error:?}");
    };
    assert!(reason.contains("4294967295"), "{reason}");
    assert!(reason.contains("exact count is unknown"), "{reason}");
    assert!(!dir.path().join("work").exists(), "nothing is left behind");
}

/// The source of score part rows as a score manifest records it: the archive's path
/// and SHA-256, the entry patterns and the maximum record length (`manifest.json` of
/// `baseline score` over an archive). Nothing of it is taken from the scope.
fn score_manifest_source(
    source_path: &Path,
    content_sha256: &str,
    entries: &[&str],
    max_record_bytes: u64,
) -> MarkedSource {
    MarkedSource::Input(l2d_core::MarkedInput {
        path: source_path.to_string_lossy().into_owned(),
        content_sha256: content_sha256.to_string(),
        archive_entries: Some(entries.iter().map(|entry| entry.to_string()).collect()),
        interpretation: Some(l2d_core::marks::InputInterpretation::ArchiveLinesV1 {
            max_record_bytes,
        }),
    })
}

#[test]
fn score_rows_join_under_the_source_their_manifest_records_and_no_other() {
    use sha2::{Digest, Sha256};

    let dir = tempfile::tempdir().unwrap();
    let path = mixed_archive(dir.path());
    let zip = ZipInputSettings::default();
    let scope = open_archive(&path, &zip, &settings(dir.path(), "work"));
    let digest = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
    let rows = [(0u32, 1u64), (0, 2_500), (0, 36)];
    let join = |produced_under: &MarkedSource| {
        let mut found = Vec::new();
        scope
            .join_keys(
                produced_under,
                rows.map(|(entry, line)| {
                    NativeKey::Position(ArchivePosition { entry, line }.pack())
                }),
                &mut budget(),
                |_, original| {
                    found.push((original.first, original.occurrences));
                    Ok(())
                },
            )
            .map(|_| found)
    };

    let max = zip.max_record_bytes;
    let found = join(&score_manifest_source(&path, &digest, &[], max)).unwrap();
    // Request 0 and request 35 occur five times over the three entries, request 2499
    // three times.
    assert_eq!(
        found,
        [
            (Locator::ArchiveLine { entry: 0, line: 1 }, 5),
            (
                Locator::ArchiveLine {
                    entry: 0,
                    line: 2_500
                },
                3
            ),
            (Locator::ArchiveLine { entry: 0, line: 36 }, 5),
        ]
    );

    let error = join(&score_manifest_source(&path, &"0".repeat(64), &[], max)).unwrap_err();
    assert!(matches!(error, JoinError::SourceChanged { .. }), "{error}");
    let error = join(&score_manifest_source(&path, &digest, &["first.log"], max)).unwrap_err();
    assert!(matches!(error, JoinError::Filters { .. }), "{error}");
    let error = join(&score_manifest_source(&path, &digest, &[], 4_096)).unwrap_err();
    assert!(matches!(error, JoinError::Interpretation { .. }), "{error}");
    let copy = dir.path().join("copy.zip");
    std::fs::copy(&path, &copy).unwrap();
    let error = join(&score_manifest_source(&copy, &digest, &[], max)).unwrap_err();
    assert!(matches!(error, JoinError::Path { .. }), "{error}");
}
