use l2d_core::patterns::{PatternCatalogue, PatternSettings};
use l2d_core::selections::{SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{InputAdapter, LinesAdapter, MemoryBudget, MemorySettings};
#[test]
fn catalogue_accounts_for_selected_records() {
    let root = std::env::temp_dir().join(format!("l2d-pattern-red-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("input.log");
    std::fs::write(
        &file,
        "login failed user=alice\nlogin failed user=bob\nlogin failed user=alice\n",
    )
    .unwrap();
    let mut budget =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&file, &mut budget).unwrap();
    let selection = SelectionSet::search_input(
        "login",
        SearchQuery::literal("login"),
        SearchSettings { max_members: 100 },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    let result = PatternCatalogue::build_input(
        &selection,
        &logs,
        &summary,
        &PatternSettings::default(),
        &root.join("catalogue"),
        &mut budget,
    );
    assert!(
        result.is_ok(),
        "catalogue must account for every selected member: {result:?}"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn repeat_overflow_reviews_and_disk_corruption() {
    use l2d_core::selections::{Member, ReviewAction, ReviewDecision, ReviewState};
    let root = std::env::temp_dir().join(format!("l2d-pattern-contract-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("input.log");
    std::fs::write(&file,"login failed user=alice\nlogin failed user=bob\nlogin failed user=alice\nlogin succeeded user=c\nrare emergency\n").unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&file, &mut b).unwrap();
    let mut s = SelectionSet::search_input(
        "all",
        SearchQuery::regex(".*"),
        SearchSettings { max_members: 100 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let preview = s
        .preview_review(
            &[ReviewDecision {
                member: Member::Position(1),
                state: ReviewState::Bad,
            }],
            &b,
        )
        .unwrap();
    s.review_input(
        preview,
        ReviewAction {
            analyst: "fixture".into(),
            action: "explicit review".into(),
        },
        &summary,
        None,
        &b,
    )
    .unwrap();
    let settings = PatternSettings {
        max_patterns: 1,
        ..Default::default()
    };
    let a = PatternCatalogue::build_input(&s, &logs, &summary, &settings, &root.join("a"), &mut b)
        .unwrap();
    let c = PatternCatalogue::build_input(&s, &logs, &summary, &settings, &root.join("b"), &mut b)
        .unwrap();
    assert_eq!(a.revision, c.revision);
    assert_eq!(a.patterns.len(), 1);
    assert_eq!(a.patterns[0].counts.occurrences, 3);
    assert_eq!(a.patterns[0].counts.distinct_texts, 2);
    assert_eq!(a.patterns[0].counts.reviewed_texts, 1);
    assert_eq!(a.ungrouped.members, 2);
    assert_eq!(a.selected_members, 4);
    assert_eq!(a.patterns[0].representative, "login failed user=alice");
    assert!(!a.patterns[0].variable_examples.is_empty());
    let loaded = PatternCatalogue::load(&root.join("a"), &mut b).unwrap();
    assert_eq!(loaded.revision, a.revision);
    let mut members = Vec::new();
    loaded
        .visit_members(&root.join("a"), &mut b, |m| {
            members.push(m.member);
            Ok(())
        })
        .unwrap();
    assert_eq!(members, s.members());
    std::fs::write(root.join("a/members.jsonl"), "corrupted").unwrap();
    assert!(PatternCatalogue::load(&root.join("a"), &mut b).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn refuses_source_and_memory_mismatch_before_publication() {
    let root = std::env::temp_dir().join(format!("l2d-pattern-refuse-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("input.log");
    std::fs::write(&file, "hello id=1\n").unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, mut summary, _) = LinesAdapter.read_with(&file, &mut b).unwrap();
    let s = SelectionSet::search_input(
        "all",
        SearchQuery::literal("hello"),
        SearchSettings { max_members: 10 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    summary.content_sha256 = "0".repeat(64);
    assert!(
        PatternCatalogue::build_input(
            &s,
            &logs,
            &summary,
            &PatternSettings::default(),
            &root.join("bad"),
            &mut b
        )
        .is_err()
    );
    assert!(!root.join("bad/manifest.json").exists());
    let mut tiny = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    assert!(
        PatternCatalogue::build_input(
            &s,
            &logs,
            &summary,
            &PatternSettings::default(),
            &root.join("tiny"),
            &mut tiny
        )
        .is_err()
    );
    assert!(!root.join("tiny").exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn archive_counts_include_late_duplicates_and_scope_is_checked() {
    use l2d_core::ZipInputSettings;
    use std::io::Write;
    let root = std::env::temp_dir().join(format!("l2d-pattern-archive-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("input.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&file).unwrap());
    zip.start_file("a.log", zip::write::SimpleFileOptions::default())
        .unwrap();
    zip.write_all(b"login failed user=a\nlogin failed user=b\n")
        .unwrap();
    zip.start_file("b.log", zip::write::SimpleFileOptions::default())
        .unwrap();
    zip.write_all(b"login failed user=a\n").unwrap();
    zip.finish().unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let options = ZipInputSettings::default();
    let (s, _) = SelectionSet::search_archive(
        "login",
        SearchQuery::literal("login"),
        SearchSettings { max_members: 10 },
        &file,
        &options,
        &mut b,
    )
    .unwrap();
    let c = PatternCatalogue::build_archive(
        &s,
        &file,
        &options,
        &PatternSettings::default(),
        &root.join("good"),
        &mut b,
    )
    .unwrap();
    assert_eq!(c.patterns[0].counts.occurrences, 3);
    assert_eq!(c.patterns[0].counts.distinct_texts, 2);
    let different = ZipInputSettings {
        entries: vec!["a.log".into()],
        ..Default::default()
    };
    assert!(
        PatternCatalogue::build_archive(
            &s,
            &file,
            &different,
            &PatternSettings::default(),
            &root.join("bad"),
            &mut b
        )
        .is_err()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn heldout_evolution_outcomes_and_membership_failure() {
    let root = std::env::temp_dir().join(format!("l2d-pattern-heldout-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("input.log");
    std::fs::write(&file,"cache shard blue lookup complete\ncache shard green lookup complete\ncache shard red lookup complete\ncache shard green lookup aborted\nrequest done rc=0 id=1\nrequest done rc=1 id=2\nunique singular alarm!\n").unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&file, &mut b).unwrap();
    let s = SelectionSet::search_input(
        "all",
        SearchQuery::regex(".*"),
        SearchSettings { max_members: 100 },
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
        &root.join("a"),
        &mut b,
    )
    .unwrap();
    assert_eq!(c.selected_members, 7);
    assert!(c.patterns.iter().any(|p| {
        p.possible_outcome_collision
            && p.differing_literals
                .iter()
                .any(|v| v == "complete -> aborted")
    }));
    assert!(
        c.patterns
            .iter()
            .any(|p| p.representative == "unique singular alarm!")
    );
    std::fs::remove_file(root.join("a/members.jsonl")).unwrap();
    assert!(
        c.visit_members(&root.join("a"), &mut b, |_| Ok(()))
            .is_err()
    );
    assert!(PatternCatalogue::load(&root.join("a"), &mut b).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn canonical_order_mask_revision_and_record_limit() {
    let root = std::env::temp_dir().join(format!("l2d-pattern-order-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("input.log");
    std::fs::write(&path,"worker alpha completed batch\nworker beta completed batch\nworker gamma completed batch\nrequest status=200 id=1\nrequest status=500 id=2\n").unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut b).unwrap();
    let s = SelectionSet::search_input(
        "all",
        SearchQuery::regex(".*"),
        SearchSettings { max_members: 10 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let settings = PatternSettings {
        max_patterns: 8,
        ..Default::default()
    };
    let a = PatternCatalogue::build_input(&s, &logs, &summary, &settings, &root.join("a"), &mut b)
        .unwrap();
    let mut records = logs.to_records();
    records.reverse();
    let reversed = l2d_core::LogsFrame::from_records(&records).unwrap();
    let z =
        PatternCatalogue::build_input(&s, &reversed, &summary, &settings, &root.join("z"), &mut b)
            .unwrap();
    assert_eq!(a.revision, z.revision);
    let mut alternate = settings.clone();
    alternate.masks.revision = "separate-discovery-v2".into();
    let c = PatternCatalogue::build_input(&s, &logs, &summary, &alternate, &root.join("c"), &mut b)
        .unwrap();
    assert_ne!(a.revision, c.revision);
    alternate.max_text_bytes = 8;
    assert!(
        PatternCatalogue::build_input(&s, &logs, &summary, &alternate, &root.join("short"), &mut b)
            .is_err()
    );
    assert!(!root.join("short").exists());
    assert!(
        PatternCatalogue::build_input(&s, &logs, &summary, &settings, &root.join("a"), &mut b)
            .is_err()
    );
    assert_eq!(
        PatternCatalogue::load(&root.join("a"), &mut b)
            .unwrap()
            .revision,
        a.revision
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn member_visit_charges_resident_catalogue_and_binds_manifest() {
    let root = std::env::temp_dir().join(format!("l2d-pattern-resident-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("input.log");
    std::fs::write(&path, "hello id=1\n").unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut b).unwrap();
    let s = SelectionSet::search_input(
        "all",
        SearchQuery::literal("hello"),
        SearchSettings { max_members: 10 },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let mut c = PatternCatalogue::build_input(
        &s,
        &logs,
        &summary,
        &PatternSettings {
            max_patterns: 1,
            ..Default::default()
        },
        &root.join("cat"),
        &mut b,
    )
    .unwrap();
    let held = c.held_bytes();
    c.patterns[0].representative.reserve(16 * 1024 * 1024);
    assert!(c.held_bytes() > held + 15 * 1024 * 1024);
    std::fs::write(root.join("cat/manifest.json"), "{}").unwrap();
    let mut small = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 8 * 1024 * 1024),
            ..Default::default()
        },
        || Ok(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    let err = c
        .visit_members(&root.join("cat"), &mut small, |_| {
            panic!("must refuse before handing out rows")
        })
        .unwrap_err();
    assert!(err.to_string().contains("resident catalogue"));
    assert!(
        c.visit_members(&root.join("cat"), &mut b, |_| panic!(
            "wrong manifest must refuse"
        ))
        .unwrap_err()
        .to_string()
        .contains("manifest differs")
    );
    std::fs::remove_dir_all(root).unwrap();
}
