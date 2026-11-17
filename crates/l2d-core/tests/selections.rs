use l2d_core::selections::{
    Member, ReviewAction, ReviewDecision, ReviewState, SearchQuery, SearchSettings, SelectionSet,
};
use l2d_core::{
    InputAdapter, Label, LinesAdapter, MarkSet, MemoryBudget, MemorySettings, ProjectPaths,
    ZipInputSettings,
};
use std::{io::Write, path::PathBuf};

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("l2d-selections-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024)).unwrap()
}
fn query(text: &str) -> SearchQuery {
    SearchQuery::Literal {
        text: text.into(),
        insensitive: false,
    }
}
fn settings() -> SearchSettings {
    SearchSettings { max_members: 100 }
}

#[test]
fn file_modes_roundtrip_identity_and_explicit_review() {
    let temp = Scratch::new("file");
    let path = temp.0.join("lines.log");
    std::fs::write(
        &path,
        "login failed\nprefix login failed\nLOGIN failed\nlogin failed\na*b\naéb\n",
    )
    .unwrap();
    let mut b = budget();
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut b).unwrap();
    let mut set =
        SelectionSet::search_input("failures", query("login"), settings(), &logs, &summary, &b)
            .unwrap();
    assert_eq!(set.members(), &[Member::Position(1), Member::Position(2)]);
    assert_eq!(set.completeness().distinct_scanned, 5);
    assert_eq!(set.completeness().records, Some(6));
    assert!(set.export_mark_set("bad", Label::Bad, &b).is_err());
    let glob = SelectionSet::search_input(
        "glob",
        SearchQuery::Glob {
            pattern: "login*".into(),
            insensitive: false,
        },
        settings(),
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    assert_eq!(glob.members(), &[Member::Position(1)]);
    let insensitive = SelectionSet::search_input(
        "case",
        SearchQuery::Glob {
            pattern: "login*".into(),
            insensitive: true,
        },
        settings(),
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    assert_eq!(
        insensitive.members(),
        &[Member::Position(1), Member::Position(3)]
    );
    assert!(
        SelectionSet::search_input(
            "fts",
            SearchQuery::Fts5Trigram {
                expression: "login".into()
            },
            settings(),
            &logs,
            &summary,
            &b
        )
        .is_err()
    );
    let revision = set.revision(&b).unwrap();
    let saved = temp.0.join("selection.json");
    set.save(&saved, &b).unwrap();
    let loaded = SelectionSet::load(&saved, &b).unwrap();
    assert_eq!(loaded.revision(&b).unwrap(), revision);
    assert_eq!(loaded.members(), set.members());
    let decisions = [
        ReviewDecision {
            member: Member::Position(1),
            state: ReviewState::Bad,
        },
        ReviewDecision {
            member: Member::Position(2),
            state: ReviewState::Conflict,
        },
    ];
    let preview = set.preview_review(&decisions, &b).unwrap();
    let stale = set.preview_review(&decisions, &b).unwrap();
    assert_eq!(preview.affected(), 2);
    let action = ReviewAction {
        analyst: "analyst".into(),
        action: "Reviewed both records; conflicting second remains open".into(),
    };
    set.review_input(preview, action.clone(), &summary, None, &b)
        .unwrap();
    assert!(set.review_input(stale, action, &summary, None, &b).is_err());
    let marks = set.export_mark_set("reviewed", Label::Bad, &b).unwrap();
    assert_eq!(marks.selected().collect::<Vec<_>>(), vec![1]);
    assert_eq!(marks.deselected(), &[2]);
    assert_eq!(
        set.review().unwrap().decisions[1].state,
        ReviewState::Conflict
    );
    let project = ProjectPaths::new(temp.0.join("project"));
    marks.save(&project, &b).unwrap();
    assert!(
        MarkSet::load(&project, "reviewed", &b)
            .unwrap()
            .try_eq(&marks)
            .unwrap()
    );
    let reviewed_path = temp.0.join("reviewed.json");
    set.save(&reviewed_path, &b).unwrap();
    assert_eq!(
        SelectionSet::load(&reviewed_path, &b)
            .unwrap()
            .revision(&b)
            .unwrap(),
        set.revision(&b).unwrap()
    );
    let mut changed = summary.clone();
    changed.content_sha256 = "0".repeat(64);
    assert!(set.require_input(&changed, None).is_err());
    assert!(set.require_input(&summary, Some(&[])).is_err());
    let preview = set.preview_review(&decisions, &b).unwrap();
    assert!(
        set.review_input(
            preview,
            ReviewAction {
                analyst: "x".into(),
                action: "x".into()
            },
            &changed,
            None,
            &b
        )
        .is_err()
    );
}

#[test]
fn archive_scope_and_limits_are_explicit() {
    let temp = Scratch::new("archive");
    let path = temp.0.join("input.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    zip.start_file("a.log", zip::write::SimpleFileOptions::default())
        .unwrap();
    zip.write_all(b"login failed\nprefix login failed\n")
        .unwrap();
    zip.start_file("b.log", zip::write::SimpleFileOptions::default())
        .unwrap();
    zip.write_all(b"login failed again\n").unwrap();
    zip.finish().unwrap();
    let options = ZipInputSettings {
        entries: vec!["a.log".into()],
        ..Default::default()
    };
    let (set, report) = SelectionSet::search_archive(
        "archive",
        SearchQuery::Glob {
            pattern: "login*".into(),
            insensitive: false,
        },
        settings(),
        &path,
        &options,
        &mut budget(),
    )
    .unwrap();
    assert_eq!(set.members().len(), 1);
    assert_eq!(set.completeness().distinct_scanned, 2);
    for query in [
        SearchQuery::literal("login"),
        SearchQuery::regex("login.*failed"),
    ] {
        let (other, _) = SelectionSet::search_archive(
            "other",
            query,
            settings(),
            &path,
            &options,
            &mut budget(),
        )
        .unwrap();
        assert_eq!(other.members().len(), 2);
    }
    assert!(
        SelectionSet::search_archive(
            "fts",
            SearchQuery::Fts5Trigram {
                expression: "login".into()
            },
            settings(),
            &path,
            &options,
            &mut budget()
        )
        .is_err()
    );
    let saved = temp.0.join("archive-selection.json");
    set.save(&saved, &budget()).unwrap();
    let loaded = SelectionSet::load(&saved, &budget()).unwrap();
    assert_eq!(
        loaded.revision(&budget()).unwrap(),
        set.revision(&budget()).unwrap()
    );

    assert!(set.completeness().complete);
    set.require_input(&report.input, Some(&options.entries))
        .unwrap();
    assert!(set.require_input(&report.input, Some(&[])).is_err());
    assert!(
        SelectionSet::search_archive(
            "limited",
            query("login"),
            SearchSettings { max_members: 1 },
            &path,
            &options,
            &mut budget()
        )
        .is_err()
    );
}

#[test]
fn allocation_refusal_precedes_member_or_query_allocation_and_load() {
    let temp = Scratch::new("memory");
    let path = temp.0.join("input.log");
    std::fs::write(&path, "a\nb\n").unwrap();
    let mut b = budget();
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut b).unwrap();
    assert!(
        SelectionSet::search_input(
            "overflow",
            query("a"),
            SearchSettings {
                max_members: u64::MAX
            },
            &logs,
            &summary,
            &b
        )
        .is_err()
    );
    let set =
        SelectionSet::search_input("ok", query("a"), settings(), &logs, &summary, &b).unwrap();
    let saved = temp.0.join("saved.json");
    set.save(&saved, &b).unwrap();
    b.hold(b.usable_bytes());
    assert!(
        SelectionSet::search_input("small", query("a"), settings(), &logs, &summary, &b).is_err()
    );
    assert!(SelectionSet::load(&saved, &b).is_err());
    assert!(set.save(&temp.0.join("refused.json"), &b).is_err());
    assert!(!temp.0.join("refused.json").exists());
    assert!(set.preview_review(&[], &b).is_err());
}

#[test]
fn saved_selections_reject_inconsistent_members_modes_and_versions() {
    let temp = Scratch::new("invalid-artifacts");
    let path = temp.0.join("input.log");
    std::fs::write(&path, "one\ntwo\n").unwrap();
    let mut b = budget();
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut b).unwrap();
    let set = SelectionSet::search_input(
        "valid",
        SearchQuery::regex("."),
        settings(),
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let saved = temp.0.join("valid.json");
    set.save(&saved, &b).unwrap();
    let original = std::fs::read(&saved).unwrap();
    assert!(set.save(&saved, &b).is_err());
    assert_eq!(std::fs::read(&saved).unwrap(), original);
    let value: serde_json::Value = serde_json::from_slice(&original).unwrap();
    for mode in 0..5 {
        let mut broken = value.clone();
        match mode {
            0 => broken["format"] = 99.into(),
            1 => broken["members"][1] = broken["members"][0].clone(),
            2 => broken["completeness"]["matched"] = 999.into(),
            3 => {
                broken["query"] = serde_json::json!({"mode":"fts5_trigram","expression":"one", "insensitive": false})
            }
            _ => broken["source"]["content_sha256"] = "bad".into(),
        }
        let path = temp.0.join(format!("broken-{mode}.json"));
        std::fs::write(&path, serde_json::to_vec(&broken).unwrap()).unwrap();
        assert!(SelectionSet::load(&path, &b).is_err());
    }
}

#[test]
fn store_change_during_scan_refuses_and_restores_budget() {
    use l2d_core::polars::prelude::DataFrame;
    use l2d_core::selections::SelectionStore;
    use l2d_core::{L2dError, MarkStore};
    use std::cell::Cell;
    struct Changing(Cell<u64>);
    impl MarkStore for Changing {
        fn store_name(&self) -> &str {
            "changing"
        }
        fn content_sha256(&self, _: &mut MemoryBudget) -> Result<String, L2dError> {
            let n = self.0.get();
            self.0.set(n + 1);
            Ok(format!("{n:064x}"))
        }
        fn phrase_ids(
            &self,
            _: &str,
            _: u64,
            _: &mut MemoryBudget,
            _: &mut dyn FnMut(&str),
        ) -> Result<(), L2dError> {
            unreachable!()
        }
        fn text_batches(
            &self,
            _: u64,
            _: u64,
            _: &mut MemoryBudget,
            _: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
        ) -> Result<(), L2dError> {
            Ok(())
        }
    }
    impl SelectionStore for Changing {
        fn fts5_trigram_ids(
            &self,
            _: &str,
            _: &mut MemoryBudget,
            _: &mut dyn FnMut(&str) -> Result<(), L2dError>,
        ) -> Result<u64, L2dError> {
            Ok(0)
        }
    }
    let mut b = budget();
    b.hold(4096);
    let error = SelectionSet::search_store(
        "change",
        query("x"),
        settings(),
        &Changing(Cell::new(0)),
        &mut b,
    )
    .unwrap_err();
    assert!(error.to_string().contains("changed"));
    assert_eq!(b.held_bytes(), 4096);
}
