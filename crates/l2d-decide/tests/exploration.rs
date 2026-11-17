use l2d_core::patterns::{
    PatternCatalogue, PatternCounts, PatternMember, PatternSettings, refinement::PatternRefinement,
};
use l2d_core::selections::{Completeness, Member, ReviewState};
use l2d_core::{L2dError, MarkedInput, MarkedSource, MemoryBudget, MemorySettings};
use l2d_decide::exploration::*;

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024)).unwrap()
}
fn fixture(name: &str, rows: &[(&str, usize, u64)]) -> PatternRefinement {
    let catalogue = PatternCatalogue {
        format: 1,
        algorithm_revision: "test".into(),
        selection_revision: format!("selection-{name}"),
        source: MarkedSource::Input(MarkedInput {
            path: format!("{name}.log"),
            content_sha256: name.repeat(64),
            archive_entries: None,
        }),
        source_data_revision: None,
        completeness: Completeness {
            complete: true,
            records: Some(rows.len() as u64),
            distinct_scanned: rows.len() as u64,
            matched: rows.len() as u64,
            skipped_records: 0,
            omitted_entries: 0,
        },
        settings: PatternSettings::default(),
        occurrence_basis: "original".into(),
        patterns: vec![],
        ungrouped: PatternCounts::default(),
        selected_members: rows.len() as u64,
        memberships_sha256: "0".repeat(64),
        revision: name.into(),
    };
    PatternRefinement {
        format: 1,
        catalogue,
        catalogue_revision: name.into(),
        selection_revision: format!("selection-{name}"),
        parent_revision: None,
        action: "fixture".into(),
        groups: vec![],
        priorities: Default::default(),
        ungrouped: PatternCounts::default(),
        rows: rows
            .iter()
            .enumerate()
            .map(|(i, (t, p, n))| PatternMember {
                member: Member::Position(i as u64),
                pattern: Some(*p),
                text: (*t).into(),
                occurrences: *n,
                review: Some(ReviewState::Bad),
            })
            .collect(),
        revision: name.into(),
    }
}
struct Vectors(usize);
impl ExplorationEncoder for Vectors {
    fn profile(&self) -> &str {
        "test-profile"
    }
    fn scratch_bytes(&self) -> u64 {
        1024
    }
    fn unsupported_reason(&self) -> &str {
        "not supported"
    }
    fn encode(&mut self, text: &str) -> Result<Option<Representation>, L2dError> {
        self.0 += 1;
        if text == "failed" {
            return Err(L2dError::Refused {
                reason: "encoder runtime failed".into(),
            });
        }
        Ok(Some(Representation::Vector(match text {
            "zero" => vec![0., 0.],
            "nan" => vec![f32::NAN, 1.],
            "distant" => vec![0., 1.],
            _ => vec![1., 0.],
        })))
    }
}
fn settings() -> ExploreSettings {
    ExploreSettings {
        exemplar_row: 0,
        scope: Scope::AllPatterns,
        top_k: 10,
        radius: 0.25,
        expected_profile: Some("test-profile".into()),
    }
}
#[test]
fn expansion_keeps_origins_unlabelled_and_counts_global_independent_texts() {
    let seed = fixture(
        "a",
        &[
            ("seed", 0, 1),
            ("related", 0, 400),
            ("zero", 1, 1),
            ("nan", 1, 1),
            ("distant", 1, 1),
        ],
    );
    let other = fixture("b", &[("related", 0, 900), ("new", 0, 1), ("seed", 0, 800)]);
    let before = other.clone();
    let mut b = budget();
    let held = b.held_bytes();
    let report = explore(&seed, Some(&other), &settings(), &mut Vectors(0), &mut b).unwrap();
    assert!(b.held_bytes() > held, "returned report remains admitted");
    assert_eq!(report.unsupported.len(), 2);
    assert_eq!(report.support.distinct_texts, 2);
    assert_eq!(report.support.same_group_distinct_texts, 1);
    assert_eq!(report.support.cross_group_distinct_texts, 2);
    assert_eq!(report.support.occurrences, 2101);
    assert_eq!(report.representative.row.text, "seed");
    assert!(
        report
            .neighbours
            .iter()
            .filter(|h| h.outside_saved_search)
            .all(|h| h.row.review.is_none()
                && h.source == other.catalogue.source
                && h.selection_revision == other.selection_revision)
    );
    assert_eq!(other, before);
    assert_eq!(report.distance_range, Some([0., 0.]));
    assert!(report.connectivity.contains("not computed"));
    let mut s = settings();
    s.scope = Scope::WithinPattern;
    assert!(explore(&seed, Some(&other), &s, &mut Vectors(0), &mut b).is_err());
    let r = explore(&seed, None, &s, &mut Vectors(0), &mut b).unwrap();
    assert_eq!(r.neighbours.len(), 1);
    s.scope = Scope::AcrossPatterns;
    let r = explore(&seed, Some(&other), &s, &mut Vectors(0), &mut b).unwrap();
    assert_eq!(r.support.same_group_distinct_texts, 0);
    assert_eq!(r.support.cross_group_distinct_texts, 2);
}
#[test]
fn profile_and_memory_refusals_happen_before_encoding_and_same_source_expansion_is_not_recounted() {
    let seed = fixture("a", &[("seed", 0, 1), ("related", 0, 1)]);
    let mut encoder = Vectors(0);
    let mut s = settings();
    s.expected_profile = Some("wrong".into());
    let e = explore(&seed, None, &s, &mut encoder, &mut budget())
        .unwrap_err()
        .to_string();
    assert!(e.contains("wrong") && e.contains("test-profile"));
    assert_eq!(encoder.0, 0);
    s.expected_profile = None;
    let mut small = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    assert!(explore(&seed, None, &s, &mut encoder, &mut small).is_err());
    assert_eq!(encoder.0, 0);
    let r = explore(&seed, Some(&seed), &s, &mut encoder, &mut budget()).unwrap();
    assert_eq!(r.support.members, 1);
    s.exemplar_row = 2;
    assert!(explore(&seed, None, &s, &mut encoder, &mut budget()).is_err());
}
#[test]
fn topk_ties_are_deterministic_filters_precede_ranking_and_batches_do_not_change_results() {
    let seed = fixture(
        "a",
        &[
            ("seed", 0, 1),
            ("related", 0, 1),
            ("new", 1, 1),
            ("distant", 1, 1),
        ],
    );
    let mut s = settings();
    s.top_k = 1;
    s.scope = Scope::AcrossPatterns;
    s.radius = 2.;
    let mut b = budget();
    let r = explore(&seed, None, &s, &mut Vectors(0), &mut b).unwrap();
    assert_eq!(r.neighbours[0].row.text, "new");
    assert_eq!(r.support.distinct_texts, 2);
    let mut b = MemoryBudget::observe_with(
        &MemorySettings {
            max_batch_rows: Some(1),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    let r2 = explore(&seed, None, &s, &mut Vectors(0), &mut b).unwrap();
    assert_eq!(
        serde_json::to_value(r.neighbours).unwrap(),
        serde_json::to_value(r2.neighbours).unwrap()
    );
}

#[test]
fn encoder_failure_propagates_but_invalid_candidate_vector_is_reported_unsupported() {
    let mut b = budget();
    b.hold(1234);
    let prior = b.held_bytes();
    let failed = fixture("a", &[("seed", 0, 1), ("failed", 0, 1)]);
    let error = explore(&failed, None, &settings(), &mut Vectors(0), &mut b).unwrap_err();
    assert!(error.to_string().contains("encoder runtime failed"));
    assert_eq!(b.held_bytes(), prior);
    let invalid = fixture("a", &[("seed", 0, 1), ("zero", 0, 1), ("related", 0, 1)]);
    let report = explore(&invalid, None, &settings(), &mut Vectors(0), &mut b).unwrap();
    assert_eq!(report.unsupported.len(), 1);
    assert!(report.unsupported[0].reason.contains("zero vector"));
    assert_eq!(report.neighbours.len(), 1);
    assert_eq!(report.neighbours[0].row.text, "related");
}
