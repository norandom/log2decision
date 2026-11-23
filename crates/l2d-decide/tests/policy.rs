//! Task 2.20 exercises policy through a real SQLite store and fitted MarkSet scorer.
use l2d_baseline::markscore::{MarkFitSettings, fit_from_marks};
use l2d_baseline::redact::RedactionInput;
use l2d_baseline::{BaselineScorer, BaselineSettings};
use l2d_core::{
    EncoderIdentity, Label, LogRecord, LogsFrame, MarkQuery, MarkSet, MemoryBudget, MemorySettings,
    ProjectPaths,
};
use l2d_decide::close_enough::*;

use l2d_decide::exploration::{CharacterTrigram, ExplorationEncoder, Representation};
use l2d_store::{LogFlags, Store, VectorOrigin, rows_frame};

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024)).unwrap()
}
struct Fixture {
    paths: ProjectPaths,
    store: Store,
    bad: MarkSet,
    good: MarkSet,
    scorer: BaselineScorer,
    // Declared last so SQLite closes before the temporary directory is removed.
    _directory: tempfile::TempDir,
}
fn fixture() -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let paths = ProjectPaths::new(directory.path());
    let identity = EncoderIdentity {
        name: "fixture".into(),
        revision: "a".repeat(64),
        dimension: 2,
    };
    let mut store =
        Store::create(&paths, "history", VectorOrigin::without_release(&identity)).unwrap();
    let rows: Vec<_> = (0..16)
        .map(|i| LogRecord {
            text: format!(
                "GET /endpoint value={} item={i}",
                if i < 6 {
                    "BAD attack"
                } else if i < 12 {
                    "GOOD benign"
                } else {
                    "unreviewed"
                }
            ),
            label: Label::Unknown,
            group: if i == 4 {
                None
            } else if (6..12).contains(&i) {
                Some("endpoint0".into())
            } else {
                Some(format!("endpoint{}", i % 2))
            },
            first_position: i,
            occurrences: if i == 0 { 900 } else { 1 },
        })
        .collect();
    add(&mut store, "first", &rows);
    let mut b = budget();
    let bad = MarkSet::mark_store(
        "bad",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "BAD".into(),
        },
        &store,
        &mut b,
    )
    .unwrap();
    let good = MarkSet::mark_store(
        "good",
        Label::Good,
        MarkQuery::Phrase {
            phrase: "GOOD".into(),
        },
        &store,
        &mut b,
    )
    .unwrap();
    let scorer = fit(&store, &bad, Some(&good));
    Fixture {
        paths,
        store,
        bad,
        good,
        scorer,
        _directory: directory,
    }
}
fn add(store: &mut Store, source: &str, rows: &[LogRecord]) {
    let identity = store.identity().clone();
    let logs = LogsFrame::from_records(rows).unwrap();
    let vectors = vec![1.; rows.len() * 2];
    let frame = rows_frame(
        &logs,
        source,
        &vectors,
        2,
        &vec![LogFlags::default(); rows.len()],
    )
    .unwrap();
    store
        .add_with(
            &frame,
            VectorOrigin::without_release(&identity),
            &mut budget(),
        )
        .unwrap();
}
fn fit(store: &Store, marks: &MarkSet, other: Option<&MarkSet>) -> BaselineScorer {
    fit_from_marks(
        marks,
        other,
        RedactionInput::Store(store),
        &MarkFitSettings {
            baseline: BaselineSettings {
                buckets: 1024,
                ..Default::default()
            },
            group_key: None,
        },
        &mut budget(),
    )
    .unwrap()
    .scorer
}
fn settings() -> ReachSettings {
    ReachSettings {
        radius: 0.45,
        top_k: 30,
        minimum_labelled_support: 2,
        expected_profile: CharacterTrigram.profile().into(),
    }
}

use l2d_decide::policy::{Queue, Reason, TriagePolicy};
fn policy() -> TriagePolicy {
    TriagePolicy {
        version: "synthetic-v1".into(),
        selection_provenance: "synthetic fixtures; exploratory only".into(),
        low: 0.0,
        high: 0.01,
        corroboration: 2,
        reach: settings(),
        require_calibration: false,
        review_budget: 4,
        unknown_slots: 1,
        audit_slots: 1,
        audit_seed: 42,
    }
}

#[test]
fn fitted_mark_score_with_bad_vector_evidence_routes_to_investigate() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let near = close_enough(
        &history,
        "GET /endpoint value=BAD attack item=new",
        Some("endpoint0"),
        &f.scorer,
        None,
        &settings(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(near.close_enough);
    assert!(near.bad.count >= 2);
    let rows = [PolicyInput {
        id: "new".into(),
        text: "GET /endpoint value=BAD attack item=new".into(),
        group: Some("endpoint0".into()),
    }];
    let out = triage_policy(
        &rows,
        &f.store,
        &history,
        &f.scorer,
        None,
        &policy(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(
        (out.decisions[0].queue, out.decisions[0].reason),
        (Queue::Investigate, Reason::HighCorroborated)
    );
}

use l2d_decide::conformal::points_frame;
use l2d_decide::policy::{
    Allocation, CalibrationEvidence, PolicyInput, PolicyReport, triage_policy,
};
use l2d_decide::{CalibrationPoint, CalibrationSettings, ConformalCalibrator, ScorerKind};
fn inputs() -> Vec<PolicyInput> {
    vec![
        PolicyInput {
            id: "a".into(),
            text: "GET /endpoint value=BAD attack item=0".into(),
            group: Some("endpoint0".into()),
        },
        PolicyInput {
            id: "b".into(),
            text: "GET /endpoint value=GOOD benign item=6".into(),
            group: Some("endpoint0".into()),
        },
        PolicyInput {
            id: "c".into(),
            text: "unrelated totally remote".into(),
            group: None,
        },
    ]
}

#[test]
fn actual_workflow_preserves_both_signals_reloads_and_replays_all_members() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let mut p = policy();
    p.reach.radius = 0.15;
    p.low = 0.49;
    p.high = 0.51;
    let rows = inputs();
    let report = triage_policy(
        &rows,
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(matches!(report.calibration, CalibrationEvidence::Missing));
    assert_eq!(report.decisions.len(), 3);
    let unsupported = &report.decisions[2];
    assert_eq!(unsupported.reason, Reason::Unsupported);
    assert!(unsupported.score.actionable_ratio.is_none());
    assert_eq!(
        unsupported.score.diagnostic_ratio,
        f.scorer.ratio(&rows[2].text)
    );
    for row in &report.decisions {
        let vector = row.vector.as_ref().unwrap();
        assert_eq!(row.score.revision, vector.scorer_revision);
        assert!(!row.explanation.is_empty());
        assert!(vector.calibrated.is_none());
    }
    let path = f.paths.root.join("policy.json");
    report.save(&path, &b).unwrap();
    println!("persisted policy fixture: {}", path.display());
    let loaded = PolicyReport::load(
        &path,
        1_000_000,
        &p,
        f.scorer.revision(),
        CharacterTrigram.profile(),
        history.metadata_revision(),
        &mut b,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(&report).unwrap(),
        serde_json::to_value(&loaded).unwrap()
    );
    let mut reversed = rows;
    reversed.reverse();
    let replay = triage_policy(
        &reversed,
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    report.require_same_decisions(&replay, &b).unwrap();
    reversed[0].text.push_str(" changed input");
    let changed_input = triage_policy(
        &reversed,
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(report.require_same_decisions(&changed_input, &b).is_err());
    assert!(report.save(&path, &b).is_err());
    assert!(
        PolicyReport::load(
            &path,
            1,
            &p,
            f.scorer.revision(),
            CharacterTrigram.profile(),
            history.metadata_revision(),
            &mut b
        )
        .is_err()
    );
    assert!(
        PolicyReport::load(
            &path,
            1_000_000,
            &p,
            "changed",
            CharacterTrigram.profile(),
            history.metadata_revision(),
            &mut b
        )
        .is_err()
    );
    let mut changed = p.clone();
    changed.audit_seed += 1;
    assert!(
        PolicyReport::load(
            &path,
            1_000_000,
            &changed,
            f.scorer.revision(),
            CharacterTrigram.profile(),
            history.metadata_revision(),
            &mut b
        )
        .is_err()
    );
}

#[test]
fn missing_and_wrong_scorer_calibration_are_explicit_and_profile_mismatch_is_unsupported() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let points = points_frame(
        &(0..40)
            .map(|i| CalibrationPoint {
                ratio: if i < 20 { 0.1 } else { 0.9 },
                label: if i < 20 { Label::Good } else { Label::Bad },
                group: None,
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let wrong = ConformalCalibrator::fit(
        &points,
        ScorerKind::Baseline,
        f.scorer.revision(),
        &CalibrationSettings::default(),
    )
    .unwrap();
    let p = policy();
    let out = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        Some(Calibration {
            calibrator: &wrong,
            error_rate: 0.1,
        }),
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(matches!(
        out.calibration,
        CalibrationEvidence::Invalid { .. }
    ));
    assert!(
        out.decisions
            .iter()
            .all(|r| r.vector.as_ref().unwrap().calibrated.is_none())
    );
    let right = ConformalCalibrator::fit(
        &points,
        ScorerKind::MarkSet,
        f.scorer.revision(),
        &CalibrationSettings::default(),
    )
    .unwrap();
    let out = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        Some(Calibration {
            calibrator: &right,
            error_rate: 0.1,
        }),
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(matches!(
        out.calibration,
        CalibrationEvidence::Compatible { .. }
    ));
    assert!(
        out.decisions[0]
            .vector
            .as_ref()
            .unwrap()
            .calibrated
            .is_some()
    );
    let mut incompatible = p;
    let path = f.paths.root.join("calibrated-policy.json");
    out.save(&path, &b).unwrap();
    let loaded = PolicyReport::load(
        &path,
        1_000_000,
        &incompatible,
        f.scorer.revision(),
        CharacterTrigram.profile(),
        history.metadata_revision(),
        &mut b,
    )
    .unwrap();
    out.require_same_decisions(&loaded, &b).unwrap();
    incompatible.reach.expected_profile = "different-profile".into();
    let out = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        None,
        &incompatible,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(out.decisions.iter().all(|r| r.reason == Reason::Unsupported
        && r.score.actionable_ratio.is_none()
        && r.explanation.contains("profile")));
}

#[test]
fn reservations_keep_deferred_records_and_empty_pools_do_not_steal_capacity() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let mut p = policy();
    p.low = 0.99;
    p.high = 1.0;
    p.reach.radius = 0.15;
    let mut rows = inputs();
    for i in 3..12 {
        let mut r = rows[1].clone();
        r.id = format!("{i:02}");
        rows.push(r);
    }
    let out = triage_policy(
        &rows,
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(out.decisions.len(), 12);
    assert_eq!(out.audit.selected, 1);
    assert!(
        out.decisions
            .iter()
            .any(|r| r.allocation == Some(Allocation::LowPriorityAudit))
    );
    assert_eq!(
        out.deferred,
        12 - out.unknown.selected - out.general.selected - out.audit.selected
    );
    assert_eq!(
        out.unknown.configured + out.general.configured + out.audit.configured,
        p.review_budget
    );
    let only_good = vec![rows[1].clone()];
    let out = triage_policy(
        &only_good,
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(out.unknown.unused, 1);
    assert_eq!(out.general.unused, 2);
    assert_eq!(out.audit.selected, 1);
    let duplicate = vec![rows[0].clone(), rows[0].clone()];
    let held = b.held_bytes();
    assert!(
        triage_policy(
            &duplicate,
            &f.store,
            &history,
            &f.scorer,
            None,
            &p,
            &mut CharacterTrigram,
            &mut b
        )
        .is_err()
    );
    assert_eq!(b.held_bytes(), held);
    p.review_budget = 1;
    assert!(p.validate().is_err());
}

#[test]
fn routing_branches_are_explained_by_real_retrieved_evidence() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let cases = [
        (0.0, 0.01, false, 0.15, Reason::HighCorroborated),
        (0.99, 1.0, false, 0.15, Reason::Disagreement),
        (0.0, 1.0, false, 0.15, Reason::Uncertain),
        (0.0, 0.01, true, 0.15, Reason::Uncertain),
        (0.0, 0.01, false, 2.0, Reason::Disagreement),
    ];
    for (low, high, require_calibration, radius, want) in cases {
        let mut p = policy();
        p.low = low;
        p.high = high;
        p.require_calibration = require_calibration;
        p.reach.radius = radius;
        let out = triage_policy(
            &inputs()[..1],
            &f.store,
            &history,
            &f.scorer,
            None,
            &p,
            &mut CharacterTrigram,
            &mut b,
        )
        .unwrap();
        assert_eq!(out.decisions[0].reason, want);
        assert!(!out.decisions[0].explanation.is_empty());
    }
}

struct BrokenEncoder;
impl ExplorationEncoder for BrokenEncoder {
    fn unsupported_reason(&self) -> &str {
        "fixture unsupported"
    }
    fn profile(&self) -> &str {
        CharacterTrigram.profile()
    }
    fn scratch_bytes(&self) -> u64 {
        1024
    }
    fn encode(&mut self, _: &str) -> Result<Option<Representation>, l2d_core::L2dError> {
        Err(l2d_core::L2dError::Refused {
            reason: "real runtime failure".into(),
        })
    }
}
struct UnsupportedEncoder;
impl ExplorationEncoder for UnsupportedEncoder {
    fn profile(&self) -> &str {
        CharacterTrigram.profile()
    }
    fn unsupported_reason(&self) -> &str {
        "fixture original cannot be represented"
    }
    fn scratch_bytes(&self) -> u64 {
        1024
    }
    fn encode(&mut self, _: &str) -> Result<Option<Representation>, l2d_core::L2dError> {
        Ok(None)
    }
}
struct MutatingEncoder {
    connection: rusqlite::Connection,
    mutated: bool,
}
impl ExplorationEncoder for MutatingEncoder {
    fn unsupported_reason(&self) -> &str {
        "fixture unsupported"
    }
    fn profile(&self) -> &str {
        CharacterTrigram.profile()
    }
    fn scratch_bytes(&self) -> u64 {
        CharacterTrigram.scratch_bytes()
    }
    fn encode(&mut self, text: &str) -> Result<Option<Representation>, l2d_core::L2dError> {
        if !self.mutated {
            self.connection
                .execute("UPDATE logs SET group_key='changed'", [])
                .unwrap();
            self.mutated = true;
        }
        CharacterTrigram.encode(text)
    }
}

#[test]
fn stale_metadata_before_or_during_run_and_runtime_failures_refuse_without_result() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let held = b.held_bytes();
    let err = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        None,
        &policy(),
        &mut BrokenEncoder,
        &mut b,
    )
    .unwrap_err();
    assert!(err.to_string().contains("real runtime failure"));
    assert_eq!(b.held_bytes(), held);
    let connection = rusqlite::Connection::open(f.store.path()).unwrap();
    let mut encoder = MutatingEncoder {
        connection,
        mutated: false,
    };
    let err = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        None,
        &policy(),
        &mut encoder,
        &mut b,
    )
    .unwrap_err();
    assert!(err.to_string().contains("store changed during"));
    assert_eq!(b.held_bytes(), held);
    let err = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        None,
        &policy(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap_err();
    assert!(err.to_string().contains("metadata changed"));
    assert_eq!(b.held_bytes(), held);
}

#[test]
fn memory_limits_cover_queues_persistence_reload_and_replay() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let p = policy();
    let out = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    let path = f.paths.root.join("memory-policy.json");
    out.save(&path, &b).unwrap();
    let mut small = budget();
    small.hold(small.usable_bytes() - 1);
    let held = small.held_bytes();
    assert!(
        triage_policy(
            &inputs(),
            &f.store,
            &history,
            &f.scorer,
            None,
            &p,
            &mut CharacterTrigram,
            &mut small
        )
        .is_err()
    );
    assert!(
        out.save(&f.paths.root.join("refused-policy.json"), &small)
            .is_err()
    );
    assert!(!f.paths.root.join("refused-policy.json").exists());
    assert!(
        PolicyReport::load(
            &path,
            1_000_000,
            &p,
            f.scorer.revision(),
            CharacterTrigram.profile(),
            history.metadata_revision(),
            &mut small
        )
        .is_err()
    );
    assert!(out.require_same_decisions(&out, &small).is_err());
    assert_eq!(small.held_bytes(), held);
}

#[test]
fn immutable_input_history_routes_without_a_store_and_rejects_wrong_source_adapter() {
    use l2d_core::{InputAdapter, LinesAdapter};
    use l2d_decide::policy::PolicySource;
    let f = fixture();
    let mut b = budget();
    let file = f.paths.root.join("input.txt");
    let text = (0..16)
        .map(|i| {
            format!(
                "GET /endpoint value={} item={i}\n",
                if i < 8 { "BAD attack" } else { "GOOD benign" }
            )
        })
        .collect::<String>();
    std::fs::write(&file, text).unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&file, &mut b).unwrap();
    let bad = MarkSet::mark(
        "bad",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "BAD".into(),
        },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let good = MarkSet::mark(
        "good",
        Label::Good,
        MarkQuery::Phrase {
            phrase: "GOOD".into(),
        },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let scorer = fit_from_marks(
        &bad,
        Some(&good),
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &MarkFitSettings {
            baseline: BaselineSettings {
                buckets: 1024,
                ..Default::default()
            },
            group_key: None,
        },
        &mut b,
    )
    .unwrap()
    .scorer;
    let history = MarkHistory::from_input(&logs, &summary, &bad, Some(&good), &mut b).unwrap();
    let out = triage_policy(
        &inputs(),
        PolicySource::InputSnapshot,
        &history,
        &scorer,
        None,
        &policy(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(out.decisions[0].queue, Queue::Investigate);
    assert_eq!(out.evidence_revision, history.metadata_revision());
    assert!(
        triage_policy(
            &inputs(),
            &f.store,
            &history,
            &scorer,
            None,
            &policy(),
            &mut CharacterTrigram,
            &mut b
        )
        .is_err()
    );
    let store_history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    assert!(
        triage_policy(
            &inputs(),
            PolicySource::InputSnapshot,
            &store_history,
            &f.scorer,
            None,
            &policy(),
            &mut CharacterTrigram,
            &mut b
        )
        .is_err()
    );
}

#[test]
fn unsupported_representations_empty_pools_and_invalid_calibration_rates_are_explicit() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let p = policy();
    let out = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut UnsupportedEncoder,
        &mut b,
    )
    .unwrap();
    assert!(out.decisions.iter().all(|r| r.reason == Reason::Unsupported
        && r.score.actionable_ratio.is_none()
        && r.explanation.contains("cannot be represented")));
    assert_eq!(out.unknown.selected, 1);
    assert_eq!(out.general.selected, 2);
    assert_eq!(out.audit.unused, 1);
    let out = triage_policy(
        &[],
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(out.deferred, 0);
    assert_eq!(
        out.unknown.unused + out.general.unused + out.audit.unused,
        4
    );
    let points = points_frame(
        &(0..40)
            .map(|i| CalibrationPoint {
                ratio: if i < 20 { 0.1 } else { 0.9 },
                label: if i < 20 { Label::Good } else { Label::Bad },
                group: None,
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for (revision, rate) in [(f.scorer.revision(), -0.1), ("changed-revision", 0.1)] {
        let c = ConformalCalibrator::fit(
            &points,
            ScorerKind::MarkSet,
            revision,
            &CalibrationSettings::default(),
        )
        .unwrap();
        let out = triage_policy(
            &inputs(),
            &f.store,
            &history,
            &f.scorer,
            Some(Calibration {
                calibrator: &c,
                error_rate: rate,
            }),
            &p,
            &mut CharacterTrigram,
            &mut b,
        )
        .unwrap();
        assert!(matches!(
            out.calibration,
            CalibrationEvidence::Invalid { .. }
        ));
        assert!(
            out.decisions
                .iter()
                .all(|r| r.vector.as_ref().unwrap().calibrated.is_none())
        );
    }
}

#[test]
fn reload_rejects_inconsistent_decisions_evidence_and_allocation() {
    use serde_json::json;
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let p = policy();
    let report = triage_policy(
        &inputs(),
        &f.store,
        &history,
        &f.scorer,
        None,
        &p,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    let path = f.paths.root.join("corrupt.json");
    let original = serde_json::to_value(&report).unwrap();
    let load = |revision: &str, budget: &mut MemoryBudget| {
        PolicyReport::load(
            &path,
            1_000_000,
            &p,
            f.scorer.revision(),
            CharacterTrigram.profile(),
            revision,
            budget,
        )
    };
    report.save(&path, &b).unwrap();
    assert!(load("stale-history", &mut b).is_err());
    let held = b.held_bytes();
    for (pointer, replacement) in [
        ("/deferred", json!(100)),
        ("/decisions/0/allocation", json!("LowPriorityAudit")),
        ("/decisions/0/score/revision", json!("other")),
        ("/decisions/0/score/actionable_ratio", json!(0.123)),
        ("/decisions/0/vector/evidence_revision", json!("stale")),
        ("/decisions/0/vector/bad/count", json!(333)),
        ("/decisions/0/reason", json!("LowCorroborated")),
        (
            "/decisions/0/input/id",
            original["decisions"][1]["input"]["id"].clone(),
        ),
    ] {
        let mut doc = original.clone();
        *doc.pointer_mut(pointer).unwrap() = replacement;
        std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();
        assert!(
            load(history.metadata_revision(), &mut b).is_err(),
            "accepted {pointer}"
        );
        assert_eq!(b.held_bytes(), held);
    }
}

#[test]
fn a_batch_keeps_the_returned_evidence_and_reuses_retrieval_scratch() {
    let f = fixture();
    let mut b = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(128 * 1024 * 1024),
            ..MemorySettings::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let rows: Vec<_> = (0..12)
        .map(|i| PolicyInput {
            id: format!("original-{i}"),
            ..inputs()[i % 3].clone()
        })
        .collect();
    let held = b.held_bytes();
    let report = triage_policy(
        &rows,
        &f.store,
        &history,
        &f.scorer,
        None,
        &policy(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    let retained = b.held_bytes() - held;
    println!("12 query outputs retained {retained} bytes with a 128 MiB limit");
    assert_eq!(report.decisions.len(), rows.len());
    assert!(retained > 0 && retained < 8 * 1024 * 1024);
    drop(report);
    b.hold(held);
    let evidence = close_enough(
        &history,
        &rows[0].text,
        rows[0].group.as_deref(),
        &f.scorer,
        None,
        &settings(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(!evidence.neighbours.is_empty());
    assert!(b.held_bytes() > held && b.held_bytes() - held < 1024 * 1024);
    drop(evidence);
    b.hold(b.usable_bytes() - 1);
    let constrained = b.held_bytes();
    assert!(
        close_enough(
            &history,
            &rows[0].text,
            None,
            &f.scorer,
            None,
            &settings(),
            &mut CharacterTrigram,
            &mut b
        )
        .is_err()
    );
    assert_eq!(b.held_bytes(), constrained);
}
