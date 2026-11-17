use l2d_core::patterns::{PatternCatalogue, PatternSettings, refinement::PatternRefinement};
use l2d_core::selections::{SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{InputAdapter, LinesAdapter, MemoryBudget, MemorySettings};

#[test]
fn discovery_refinement_keeps_exact_members_and_occurrences() {
    let root = std::env::temp_dir().join(format!("l2d-refinement-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("input.log");
    std::fs::write(&file, "login failed user=alice\nlogin failed user=bob\nlogin failed user=alice\nlogin succeeded user=carol\nrare [x].* exception\n").unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&file, &mut b).unwrap();
    let selection = SelectionSet::search_input(
        "all",
        SearchQuery::glob("*"),
        SearchSettings { max_members: 100 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let catalogue = PatternCatalogue::build_input(
        &selection,
        &logs,
        &summary,
        &PatternSettings::default(),
        &root.join("catalogue"),
        &mut b,
    )
    .unwrap();
    let refined = PatternRefinement::from_catalogue(&catalogue, &root.join("catalogue"), &mut b);
    assert!(
        refined.is_ok(),
        "discovery must support source-bound refinement: {refined:?}"
    );
    let refined = refined.unwrap();
    assert_eq!(
        refined.rows.iter().map(|r| &r.member).collect::<Vec<_>>(),
        selection.members().iter().collect::<Vec<_>>()
    );
    assert_eq!(refined.rows.iter().map(|r| r.occurrences).sum::<u64>(), 5);
    let split = refined.split(0, "alice", &b).unwrap();
    assert!(refined.split(0, ".*", &b).is_err());
    assert_eq!(
        split.parent_revision.as_deref(),
        Some(refined.revision.as_str())
    );
    assert_eq!(split.rows.iter().map(|r| r.occurrences).sum::<u64>(), 5);
    let id = split
        .rows
        .iter()
        .find(|r| r.text.contains("alice"))
        .unwrap()
        .pattern
        .unwrap();
    let merged = split.merge(&[0, id], &b).unwrap();
    assert_eq!(merged.rows, refined.rows);
    assert!(merged.merge(&[0, 0], &b).is_err());
    let exact = merged.proposed_regex(0, &b).unwrap();
    let preview = merged.preview_query(0, &exact, &b).unwrap();
    assert!(preview.member_misses.is_empty());
    assert!(preview.background_matches.is_empty());
    let broad = merged.preview_query(0, "login", &b).unwrap();
    assert_eq!(broad.background_matches.len(), 1);
    let misses = merged.preview_query(0, "alice", &b).unwrap();
    assert_eq!(misses.member_misses.len(), 1);
    let priority = merged
        .prioritize(0, l2d_core::patterns::refinement::Priority::Hidden, &b)
        .unwrap();
    assert_eq!(priority.rows, merged.rows);
    assert!(
        priority
            .save_query(&preview, &root.join("stale.json"), &b)
            .is_err()
    );
    assert!(!root.join("stale.json").exists());
    merged
        .save_query(&preview, &root.join("query.json"), &b)
        .unwrap();
    merged.save(&root.join("refinement.json"), &b).unwrap();
    assert_eq!(
        PatternRefinement::load(&root.join("refinement.json"), &b).unwrap(),
        merged
    );
    assert!(merged.save(&root.join("refinement.json"), &b).is_err());
    use l2d_core::selections::{ReviewAction, ReviewState};
    let mut reviewed = selection;
    let p = merged
        .preview_review(&reviewed, 0, ReviewState::Bad, &b)
        .unwrap();
    reviewed
        .review_input(
            p,
            ReviewAction {
                analyst: "test".into(),
                action: "first explicit review".into(),
            },
            &summary,
            None,
            &b,
        )
        .unwrap();
    let reviewed_r = merged.with_review(&reviewed, &b).unwrap();
    assert!(
        merged
            .preview_review(&reviewed, 1, ReviewState::Good, &b)
            .is_err()
    );
    let p = reviewed_r
        .preview_review(&reviewed, 1, ReviewState::Good, &b)
        .unwrap();
    reviewed
        .review_input(
            p,
            ReviewAction {
                analyst: "test".into(),
                action: "second explicit review".into(),
            },
            &summary,
            None,
            &b,
        )
        .unwrap();
    let twice = reviewed_r.with_review(&reviewed, &b).unwrap();
    assert_eq!(
        twice
            .rows
            .iter()
            .filter(|r| r.review == Some(ReviewState::Bad))
            .count(),
        2
    );
    assert_eq!(
        twice
            .rows
            .iter()
            .filter(|r| r.review == Some(ReviewState::Good))
            .count(),
        1
    );
    assert_eq!(
        twice
            .rows
            .iter()
            .filter(|r| r.review == Some(ReviewState::Unresolved))
            .count(),
        1
    );
    let marks = reviewed
        .export_mark_set("reviewed", l2d_core::Label::Bad, &b)
        .unwrap();
    assert_eq!(marks.selected_count(), 2);
    assert_eq!(marks.count(), 3);
    let tiny = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(33 * 1024 * 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    assert!(PatternRefinement::load(&root.join("refinement.json"), &tiny).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn generated_query_escapes_literals_and_exposes_outcome_collisions() {
    use l2d_core::{LabelledAdapter, LabelledFormat};
    let root = std::env::temp_dir().join(format!("l2d-refinement-literals-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("input.jsonl");
    let texts = [
        "failed [x].* literal<*> user=Åsa\nline",
        "failed [x].* literal<*> user=Éva\nline",
        "succeeded [x].* literal<*> user=Åsa\nline",
        "failed ZZZ literal<*> user=Åsa\nline",
    ];
    let data = texts
        .iter()
        .map(|text| serde_json::json!({"text":text,"label":"good"}).to_string() + "\n")
        .collect::<String>();
    std::fs::write(&file, data).unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, summary, _) = LabelledAdapter::new(LabelledFormat::Jsonl)
        .read_with(&file, &mut b)
        .unwrap();
    let s = SelectionSet::search_input(
        "unicode",
        SearchQuery::glob("*"),
        SearchSettings { max_members: 10 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let c = PatternCatalogue::build_input(
        &s,
        &logs,
        &summary,
        &PatternSettings::default(),
        &root.join("catalogue"),
        &mut b,
    )
    .unwrap();
    let r = PatternRefinement::from_catalogue(&c, &root.join("catalogue"), &mut b).unwrap();
    let first = r
        .rows
        .iter()
        .find(|r| r.text == texts[0])
        .unwrap()
        .pattern
        .unwrap();
    let query = r.proposed_regex(first, &b).unwrap();
    assert!(query.contains(r"\[x\]\.\*"));
    assert!(query.contains(r"<\*>"));
    let p = r.preview_query(first, &query, &b).unwrap();
    assert!(p.member_misses.is_empty());
    assert!(p.background_matches.is_empty());
    assert!(
        r.rows.iter().all(|r| r.review.is_none()),
        "dataset good labels are not analyst review"
    );
    let success = r
        .rows
        .iter()
        .find(|r| r.text == texts[2])
        .unwrap()
        .pattern
        .unwrap();
    let merged = r.merge(&[first, success], &b).unwrap();
    let group = merged
        .groups
        .iter()
        .find(|p| p.id == first.min(success))
        .unwrap();
    assert!(group.possible_outcome_collision);
    assert!(!group.differing_literals.is_empty());
    assert_eq!(merged.rows.len(), r.rows.len());
    let changed = root.join("changed.json");
    let mut v = serde_json::to_value(&r).unwrap();
    v["rows"][0]["occurrences"] = serde_json::json!(999);
    std::fs::write(&changed, serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(PatternRefinement::load(&changed, &b).is_err());
    std::fs::write(&file, "changed").unwrap();
    assert!(r.require_current_input().is_err());
    std::fs::remove_dir_all(root).unwrap();
}
