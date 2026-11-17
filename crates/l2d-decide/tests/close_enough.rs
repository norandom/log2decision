//! Task 2.8: a real SQLite store, saved MarkSets and fitted character scorer.
use l2d_baseline::markscore::{MarkFitSettings, fit_from_marks};
use l2d_baseline::redact::RedactionInput;
use l2d_baseline::{BaselineScorer, BaselineSettings};
use l2d_core::{
    EncoderIdentity, Label, LogRecord, LogsFrame, MarkQuery, MarkSet, MemoryBudget, MemorySettings,
    ProjectPaths,
};
use l2d_decide::close_enough::*;
use l2d_decide::conformal::points_frame;
use l2d_decide::exploration::{CharacterTrigram, ExplorationEncoder, Representation};
use l2d_decide::{CalibrationPoint, CalibrationSettings, ConformalCalibrator, Scorer, ScorerKind};
use l2d_store::{LogFlags, Store, VectorOrigin, rows_frame};
use std::sync::atomic::{AtomicU64, Ordering};

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024)).unwrap()
}
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    paths: ProjectPaths,
    store: Store,
    bad: MarkSet,
    good: MarkSet,
    scorer: BaselineScorer,
}
fn fixture() -> Fixture {
    let paths = ProjectPaths::new(std::env::temp_dir().join(format!(
        "l2d-close-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
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
#[test]
fn store_groups_deselects_and_supported_ratio_is_saved_scorer_not_vote() {
    let mut f = fixture();
    let mut b = budget();
    let path = f.scorer.save(&f.paths, "score").unwrap();
    let saved = BaselineScorer::load(&path).unwrap();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let groups = group_marks(&history, &settings(), &mut CharacterTrigram, &mut b).unwrap();
    assert_eq!(groups.groups.len(), 1);
    assert_eq!(groups.groups[0].size, 6);
    assert_eq!(groups.groups[0].distinct_texts, 6);
    assert_eq!(groups.groups[0].occurrences, 905);
    assert!(groups.groups[0].representative.text.contains("BAD"));
    assert!(groups.groups[0].distance_range[1] > 0.);
    assert!(groups.method.contains("character-trigram"));
    let query = "GET /endpoint value=BAD attack item=new";
    let near = close_enough(
        &history,
        query,
        Some("endpoint0"),
        &saved,
        None,
        &settings(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(near.close_enough);
    assert_eq!(near.ratio, Some(saved.ratio(query)));
    assert_eq!(near.scorer, "mark-set");
    assert_eq!(near.bad.count, 6);
    assert!(near.bad.mean_distance.unwrap() > 0.);
    assert_eq!(near.unknown_group, 1);
    assert!(near.same_group > 0 && near.cross_group > 0);
    assert_ne!(
        near.ratio.unwrap(),
        near.bad.count as f32 / (near.bad.count + near.good.count) as f32
    );
    let far = close_enough(
        &history,
        "zzzzzzzzzzzzzz",
        None,
        &saved,
        None,
        &settings(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(!far.close_enough);
    assert!(far.ratio.is_none());
    assert!(far.reason.unwrap().contains("not close enough"));
    groups
        .deselect(0, &mut f.bad, Some(&f.store), &mut b)
        .unwrap();
    assert_eq!(f.bad.selected_count(), 0);
    assert_eq!(f.bad.deselected_id_count(), 6);
    let serialized = f.bad.to_file_bytes().unwrap();
    let restored =
        MarkSet::from_file_bytes(&serialized, std::path::Path::new("marks.json"), "bad").unwrap();
    assert_eq!(restored.deselected_id_count(), 6);
    let changed = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    assert!(
        close_enough(
            &changed,
            query,
            None,
            &saved,
            None,
            &settings(),
            &mut CharacterTrigram,
            &mut b
        )
        .unwrap_err()
        .to_string()
        .contains("scorer marks differ")
    );
}
#[test]
fn assumed_background_never_supplies_labelled_support() {
    let f = fixture();
    let mut b = budget();
    let scorer = fit(&f.store, &f.bad, None);
    let history = MarkHistory::from_store(&f.store, &f.bad, None, &mut b).unwrap();
    let mut s = settings();
    s.radius = 1.;
    s.minimum_labelled_support = 7;
    let out = close_enough(
        &history,
        "GET /endpoint value=BAD attack item=new",
        None,
        &scorer,
        None,
        &s,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(out.bad.count, 6);
    assert_eq!(out.good.count, 0);
    assert_eq!(out.assumed_background, 10);
    assert!(out.ratio.is_none());
    assert!(
        out.neighbours
            .iter()
            .filter(|h| h.provenance == Provenance::AssumedBackground)
            .all(|h| h.label.is_none())
    );
}
#[test]
fn duplicate_texts_and_occurrences_do_not_inflate_support_and_conflicts_are_excluded() {
    let mut f = fixture();
    let mut b = budget();
    let duplicate = LogRecord {
        text: "GET /endpoint value=BAD attack item=0".into(),
        label: Label::Unknown,
        group: Some("elsewhere".into()),
        first_position: 0,
        occurrences: 1000,
    };
    add(&mut f.store, "duplicate", &[duplicate]);
    f.bad = MarkSet::mark_store(
        "bad",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "BAD".into(),
        },
        &f.store,
        &mut b,
    )
    .unwrap();
    let history = MarkHistory::from_store(&f.store, &f.bad, None, &mut b).unwrap();
    let scorer = fit(&f.store, &f.bad, None);
    let mut s = settings();
    s.radius = 1.;
    s.minimum_labelled_support = 7;
    let out = close_enough(
        &history,
        "new",
        Some("endpoint0"),
        &scorer,
        None,
        &s,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(out.bad.count, 6);
    assert!(out.ratio.is_none());
    assert!(out.unknown_group >= 2);
    let opposite = MarkSet::mark_store(
        "conflicting",
        Label::Good,
        MarkQuery::Phrase {
            phrase: "item=0".into(),
        },
        &f.store,
        &mut b,
    )
    .unwrap();
    let conflicting = MarkHistory::from_store(&f.store, &f.bad, Some(&opposite), &mut b).unwrap();
    assert_eq!(conflicting.excluded_members(), 2);
    let groups = group_marks(&conflicting, &settings(), &mut CharacterTrigram, &mut b).unwrap();
    assert_eq!(groups.groups.iter().map(|g| g.size).sum::<usize>(), 5);
}
#[test]
fn template_only_benign_support_abstains_but_exact_original_has_payload_evidence() {
    let f = fixture();
    let mut b = budget();
    let mut good = f.good.clone();
    let ids: Vec<_> = good.selected_ids().map(Result::unwrap).collect();
    // Restrict this mark set to one real endpoint, leaving multiple distinct originals.
    for id in ids {
        let frame = f
            .store
            .text_search(&l2d_store::Filter::default(), usize::MAX)
            .unwrap();
        let index = frame
            .column("id")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .position(|v| v == Some(&id))
            .unwrap();
        if frame.column("group").unwrap().str().unwrap().get(index) != Some("endpoint0") {
            good.deselect_ids(&[id], &b).unwrap();
        }
    }
    let scorer = fit(&f.store, &good, None);
    let history = MarkHistory::from_store(&f.store, &good, None, &mut b).unwrap();
    let mut s = settings();
    s.radius = 0.3;
    let out = close_enough(
        &history,
        "GET /endpoint value=GOOD benign item=newpayload",
        Some("endpoint0"),
        &scorer,
        None,
        &s,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert!(out.ratio.is_none());
    assert!(out.reason.unwrap().contains("template-only"));
    let query = "GET /endpoint value=GOOD benign item=6";
    let exact = close_enough(
        &history,
        query,
        Some("endpoint0"),
        &scorer,
        None,
        &s,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(exact.ratio, Some(scorer.ratio(query)));
    assert!(exact.neighbours.iter().any(|h| h.exact_original_match));
}
#[test]
fn calibration_is_bound_to_real_mark_set_kind_and_revision() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    assert_eq!(f.scorer.kind(), ScorerKind::MarkSet);
    let points = points_frame(
        &(0..20)
            .map(|i| CalibrationPoint {
                ratio: if i < 10 { 0.1 } else { 0.9 },
                label: if i < 10 { Label::Good } else { Label::Bad },
                group: None,
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let calibrate = |kind, revision: &str| {
        ConformalCalibrator::fit_with(
            &points,
            kind,
            revision,
            &CalibrationSettings::default(),
            &budget(),
        )
        .unwrap()
    };
    let proper = calibrate(ScorerKind::MarkSet, f.scorer.revision());
    let old = calibrate(ScorerKind::Baseline, f.scorer.revision());
    let wrong = calibrate(
        ScorerKind::MarkSet,
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    );
    let query = "GET /endpoint value=BAD attack item=new";
    let out = close_enough(
        &history,
        query,
        Some("endpoint0"),
        &f.scorer,
        Some(Calibration {
            calibrator: &proper,
            error_rate: 0.2,
        }),
        &settings(),
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(
        out.calibrated,
        Some(
            proper
                .predict(f.scorer.ratio(query), Some("endpoint0"), 0.2)
                .unwrap()
        )
    );
    for calibrator in [&old, &wrong] {
        assert!(
            close_enough(
                &history,
                query,
                None,
                &f.scorer,
                Some(Calibration {
                    calibrator,
                    error_rate: 0.2
                }),
                &settings(),
                &mut CharacterTrigram,
                &mut b
            )
            .is_err()
        );
    }
    let path = proper.save(&f.paths, "calibration").unwrap();
    let loaded = ConformalCalibrator::load(&path).unwrap();
    assert_eq!(loaded.scorer_kind(), ScorerKind::MarkSet);
}
#[test]
fn stale_store_profile_and_budget_refuse_without_losing_prior_hold() {
    let mut f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let held = b.held_bytes();
    let mut s = settings();
    s.expected_profile = "unrelated".into();
    assert!(group_marks(&history, &s, &mut CharacterTrigram, &mut b).is_err());
    assert_eq!(b.held_bytes(), held);
    b.hold(b.usable_bytes() - 1);
    let held = b.held_bytes();
    assert!(group_marks(&history, &settings(), &mut CharacterTrigram, &mut b).is_err());
    assert_eq!(b.held_bytes(), held);
    b.hold(0);
    let duplicate = LogRecord {
        text: "GET /endpoint value=BAD attack item=0".into(),
        label: Label::Unknown,
        group: Some("endpoint0".into()),
        first_position: 0,
        occurrences: 1,
    };
    add(&mut f.store, "first", &[duplicate]);
    assert!(history.require_store(&f.store, &mut b).is_err());
}
struct Unsupported;

#[derive(Clone, Copy)]
enum RepresentationFault {
    Zero,
    Nan,
    Infinity,
    Runtime,
}
struct FaultEncoder {
    text: String,
    fault: RepresentationFault,
    calls: Vec<String>,
}
impl ExplorationEncoder for FaultEncoder {
    fn profile(&self) -> &str {
        "fault-fixture"
    }
    fn scratch_bytes(&self) -> u64 {
        1024
    }
    fn unsupported_reason(&self) -> &str {
        "original is too long"
    }
    fn encode(&mut self, text: &str) -> Result<Option<Representation>, l2d_core::L2dError> {
        self.calls.push(text.into());
        let values = if text == self.text {
            match self.fault {
                RepresentationFault::Zero => vec![0., 0.],
                RepresentationFault::Nan => vec![f32::NAN, 0.],
                RepresentationFault::Infinity => vec![f32::INFINITY, 0.],
                RepresentationFault::Runtime => {
                    return Err(l2d_core::L2dError::Refused {
                        // Same variant and validation-like wording: classification
                        // must follow the typed boundary, not inspect error messages.
                        reason: "encoder runtime exploded: zero vector is unsupported".into(),
                    });
                }
            }
        } else {
            vec![1., 0.]
        };
        Ok(Some(Representation::Vector(values)))
    }
}
fn fault_settings() -> ReachSettings {
    ReachSettings {
        expected_profile: "fault-fixture".into(),
        ..settings()
    }
}
fn marked_in_stable_order(f: &Fixture) -> Vec<(String, String)> {
    let frame = f
        .store
        .text_search(&l2d_store::Filter::default(), usize::MAX)
        .unwrap();
    let ids = frame.column("id").unwrap().str().unwrap();
    let texts = frame.column("text").unwrap().str().unwrap();
    let mut rows: Vec<_> = ids
        .iter()
        .zip(texts.iter())
        .filter_map(|(id, text)| {
            let id = id.unwrap();
            f.bad
                .is_selected_id(id)
                .unwrap()
                .then(|| (id.to_owned(), text.unwrap().to_owned()))
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn invalid_grouping_exemplars_and_candidates_are_reported_without_losing_valid_members() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let rows = marked_in_stable_order(&f);
    for fault in [
        RepresentationFault::Zero,
        RepresentationFault::Nan,
        RepresentationFault::Infinity,
    ] {
        for index in [0, 1] {
            let mut encoder = FaultEncoder {
                text: rows[index].1.clone(),
                fault,
                calls: vec![],
            };
            let prior = b.held_bytes();
            let groups = group_marks(&history, &fault_settings(), &mut encoder, &mut b).unwrap();
            assert_eq!(groups.groups.len(), 1);
            assert_eq!(groups.groups[0].size, 5);
            assert_eq!(
                groups.unsupported,
                vec![l2d_core::selections::Member::StoreId(rows[index].0.clone())]
            );
            assert_eq!(groups.unsupported_details.len(), 1);
            let reason = &groups.unsupported_details[0].reason;
            assert!(
                reason.contains(if matches!(fault, RepresentationFault::Zero) {
                    "zero vector"
                } else {
                    "non-finite"
                })
            );
            assert!(!reason.contains("too long"));
            assert_ne!(groups.groups[0].representative.text, rows[index].1);
            assert_eq!(
                encoder
                    .calls
                    .iter()
                    .filter(|text| *text == &rows[index].1)
                    .count(),
                1,
                "unsupported members are not retried as exemplars"
            );
            drop(groups);
            b.hold(prior);
        }
    }
}

#[test]
fn invalid_proximity_candidates_do_not_abort_or_inflate_labelled_support() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let rows = marked_in_stable_order(&f);
    for fault in [
        RepresentationFault::Zero,
        RepresentationFault::Nan,
        RepresentationFault::Infinity,
    ] {
        let mut encoder = FaultEncoder {
            text: rows[0].1.clone(),
            fault,
            calls: vec![],
        };
        let prior = b.held_bytes();
        let query = "query";
        let out = close_enough(
            &history,
            query,
            None,
            &f.scorer,
            None,
            &fault_settings(),
            &mut encoder,
            &mut b,
        )
        .unwrap();
        assert_eq!(out.unsupported, 1);
        assert_eq!(out.unsupported_details.len(), 1);
        let reason = &out.unsupported_details[0].reason;
        assert!(
            reason.contains(if matches!(fault, RepresentationFault::Zero) {
                "zero vector"
            } else {
                "non-finite"
            })
        );
        assert!(!reason.contains("too long"));
        assert_eq!(out.bad.count, 5);
        assert_eq!(out.good.count, 6);
        assert_eq!(out.ratio, Some(f.scorer.ratio(query)));
        assert!(out.neighbours.iter().all(|hit| hit.text != rows[0].1));
        drop(out);
        b.hold(prior);
    }
}

#[test]
fn invalid_query_returns_specific_unsupported_reason_and_no_ratio() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    for (fault, reason) in [
        (RepresentationFault::Zero, "zero vector"),
        (RepresentationFault::Nan, "non-finite"),
        (RepresentationFault::Infinity, "non-finite"),
    ] {
        let mut encoder = FaultEncoder {
            text: "query".into(),
            fault,
            calls: vec![],
        };
        let prior = b.held_bytes();
        let out = close_enough(
            &history,
            "query",
            None,
            &f.scorer,
            None,
            &fault_settings(),
            &mut encoder,
            &mut b,
        )
        .unwrap();
        assert!(!out.close_enough);
        assert!(out.ratio.is_none());
        assert!(out.calibrated.is_none());
        assert!(out.neighbours.is_empty());
        let explanation = out.reason.as_deref().unwrap();
        assert!(explanation.contains("unsupported query") && explanation.contains(reason));
        assert!(!explanation.contains("too long"));
        assert_eq!(encoder.calls, ["query"]);
        drop(out);
        b.hold(prior);
    }
}

#[test]
fn encoder_runtime_errors_still_propagate_and_restore_both_apis_incoming_hold() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let rows = marked_in_stable_order(&f);
    let prior = b.held_bytes();
    for index in [0, 1] {
        let mut encoder = FaultEncoder {
            text: rows[index].1.clone(),
            fault: RepresentationFault::Runtime,
            calls: vec![],
        };
        let error = group_marks(&history, &fault_settings(), &mut encoder, &mut b).unwrap_err();
        assert!(error.to_string().contains("encoder runtime exploded"));
        assert_eq!(b.held_bytes(), prior);
    }
    for text in ["query", rows[0].1.as_str()] {
        let mut encoder = FaultEncoder {
            text: text.into(),
            fault: RepresentationFault::Runtime,
            calls: vec![],
        };
        let error = close_enough(
            &history,
            "query",
            None,
            &f.scorer,
            None,
            &fault_settings(),
            &mut encoder,
            &mut b,
        )
        .unwrap_err();
        assert!(error.to_string().contains("encoder runtime exploded"));
        assert_eq!(b.held_bytes(), prior);
    }
}

#[test]
fn opposite_reviewed_marks_support_symmetrically_and_background_cannot_crowd_them_out() {
    let f = fixture();
    let mut b = budget();
    let mut s = settings();
    s.top_k = 2;
    s.minimum_labelled_support = 2;
    s.radius = 1.;
    for (marks, other) in [(&f.bad, &f.good), (&f.good, &f.bad)] {
        let scorer = fit(&f.store, marks, Some(other));
        let history = MarkHistory::from_store(&f.store, marks, Some(other), &mut b).unwrap();
        let query = "GET /endpoint value=BAD attack item=new";
        let out = close_enough(
            &history,
            query,
            Some("endpoint0"),
            &scorer,
            None,
            &s,
            &mut CharacterTrigram,
            &mut b,
        )
        .unwrap();
        assert_eq!(out.ratio, Some(scorer.ratio(query)));
        assert_eq!(out.bad.count, 2);
        if marks.label() == Label::Good {
            assert!(
                out.neighbours
                    .iter()
                    .filter(|h| h.label.is_some())
                    .all(|h| h.provenance == Provenance::ReviewedOpposite)
            );
        }
    }
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let query = "GET /endpoint value=unreviewed item=12";
    let out = close_enough(
        &history,
        query,
        Some("newendpoint"),
        &f.scorer,
        None,
        &s,
        &mut CharacterTrigram,
        &mut b,
    )
    .unwrap();
    assert_eq!(out.assumed_background, 2);
    assert_eq!(out.bad.count + out.good.count, 2);
    assert_eq!(out.neighbours.len(), 4);
    assert!(out.close_enough);
}
impl ExplorationEncoder for Unsupported {
    fn profile(&self) -> &str {
        "unsupported"
    }
    fn scratch_bytes(&self) -> u64 {
        1
    }
    fn unsupported_reason(&self) -> &str {
        "original beyond pinned profile"
    }
    fn encode(&mut self, _: &str) -> Result<Option<Representation>, l2d_core::L2dError> {
        Ok(None)
    }
}
#[test]
fn unsupported_original_is_not_truncated_or_scored() {
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let mut s = settings();
    s.expected_profile = "unsupported".into();
    let groups = group_marks(&history, &s, &mut Unsupported, &mut b).unwrap();
    assert!(groups.groups.is_empty());
    assert_eq!(groups.unsupported.len(), 6);
    let out = close_enough(
        &history,
        "long-original",
        None,
        &f.scorer,
        None,
        &s,
        &mut Unsupported,
        &mut b,
    )
    .unwrap();
    assert!(out.ratio.is_none());
    assert!(out.reason.unwrap().contains("unsupported query"));
}

#[test]
#[ignore = "requires pinned 2.17 artifacts and reference NPU/runtime"]
fn actual_npu_grouping_and_mark_score_use_the_same_pinned_profile() {
    use l2d_encoder::hierarchy::{HierarchicalSession, PROFILE, SESSION_BYTES};
    use std::path::PathBuf;
    let f = fixture();
    let mut b = budget();
    let history = MarkHistory::from_store(&f.store, &f.bad, Some(&f.good), &mut b).unwrap();
    let package = PathBuf::from(std::env::var_os("L2D_HG_FIXTURE").expect("L2D_HG_FIXTURE"));
    let runtime = PathBuf::from(std::env::var_os("L2D_RUNTIME_DIR").expect("L2D_RUNTIME_DIR"));
    let work = PathBuf::from(std::env::var_os("L2D_NPU_WORK_DIR").expect("L2D_NPU_WORK_DIR"));
    let mut session =
        HierarchicalSession::open(&package, &runtime, &work.join("cache"), &b).unwrap();
    b.hold(b.held_bytes() + SESSION_BYTES);
    // Radius 2 is a wiring test setting, not an operational qualification threshold.
    let s = ReachSettings {
        radius: 2.,
        top_k: 30,
        minimum_labelled_support: 2,
        expected_profile: PROFILE.into(),
    };
    let groups = group_marks(&history, &s, &mut session, &mut b).unwrap();
    assert_eq!(groups.groups.len(), 1);
    assert_eq!(groups.groups[0].size, 6);
    let query = "GET /endpoint value=BAD attack item=new";
    let out = close_enough(
        &history,
        query,
        Some("endpoint0"),
        &f.scorer,
        None,
        &s,
        &mut session,
        &mut b,
    )
    .unwrap();
    assert_eq!(out.ratio, Some(f.scorer.ratio(query)));
    assert_eq!(out.bad.count, 6);
    assert_eq!(out.good.count, 6);
    let q = session.encode(query).unwrap().unwrap();
    for hit in &out.neighbours {
        let v = session.encode(&hit.text).unwrap().unwrap();
        assert_eq!(
            hit.distance,
            l2d_decide::exploration::cosine_distance(&q, &v).unwrap()
        );
    }
    let long = close_enough(
        &history,
        &"é".repeat(513),
        None,
        &f.scorer,
        None,
        &s,
        &mut session,
        &mut b,
    )
    .unwrap();
    assert!(long.ratio.is_none());
    println!(
        "actual NPU profile={PROFILE}; groups={}, labelled={}, counted_peak={}",
        groups.groups.len(),
        out.bad.count + out.good.count,
        b.usage().report().counted_peak_bytes
    );
}
