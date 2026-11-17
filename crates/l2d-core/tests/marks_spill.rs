use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::polars::prelude::*;
use l2d_core::{L2dError, Label, MarkQuery, MarkSet, MarkStore, MemoryBudget, MemorySettings};

struct Many;
impl MarkStore for Many {
    fn store_name(&self) -> &str {
        "many"
    }
    fn content_sha256(&self, _: &mut MemoryBudget) -> Result<String, L2dError> {
        Ok("a".repeat(64))
    }
    fn phrase_ids(
        &self,
        _: &str,
        per_id: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&str),
    ) -> Result<(), L2dError> {
        budget.require_whole(70_000 * per_id, "phrase IDs")?;
        // Descending order and duplicates exercise external ordering/deduplication.
        for index in (0..70_000).rev() {
            let id = format!("{index:064x}");
            each(&id);
            if index % 100 == 0 {
                each(&id);
            }
        }
        Ok(())
    }
    fn text_batches(
        &self,
        _: u64,
        _: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        budget.require_whole(128 * 1024, "synthetic store batch")?;
        for start in (0..70_000).step_by(100) {
            let ids: Vec<_> = (start..start + 100)
                .map(|index| format!("{index:064x}"))
                .collect();
            let frame = DataFrame::new(vec![
                Column::new("id".into(), ids),
                Column::new("text".into(), vec!["match"; 100]),
            ])
            .unwrap();
            each(&frame)?;
        }
        Ok(())
    }
}

#[test]
fn regex_store_marks_also_spill_without_changing_the_selection() {
    let mut small = budget(12 * 1024 * 1024);
    let disk = MarkSet::mark_store(
        "many",
        Label::Bad,
        MarkQuery::Regex {
            pattern: "^match$".into(),
        },
        &Many,
        &mut small,
    )
    .unwrap();
    assert!(disk.ids_spilled());
    assert_eq!(disk.count(), 70_000);
    let mut seen = 0;
    disk.for_each_id(|id| {
        assert_eq!(id, format!("{seen:064x}"));
        seen += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, 70_000);
}

fn budget(usable: u64) -> MemoryBudget {
    MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(PROCESS_BASELINE_BYTES + usable),
            ..MemorySettings::default()
        },
        || Ok(4 * 1024 * 1024 * 1024),
    )
    .unwrap()
}

#[test]
fn store_ids_spill_with_identical_order_membership_and_saved_bytes() {
    let query = MarkQuery::Phrase {
        phrase: "match".into(),
    };
    let mut small = budget(2 * 1024 * 1024);
    let mut disk =
        MarkSet::mark_store("many", Label::Bad, query.clone(), &Many, &mut small).unwrap();
    assert!(disk.ids_spilled());
    assert!(small.usage().report().spilled);
    let mut large = budget(128 * 1024 * 1024);
    let mut memory = MarkSet::mark_store("many", Label::Bad, query, &Many, &mut large).unwrap();
    assert!(!memory.ids_spilled());
    assert_eq!(disk.count(), 70_000);
    assert!(disk.is_selected_id(&format!("{:064x}", 69999)).unwrap());
    assert!(!disk.is_selected_id(&format!("{:064x}", 70000)).unwrap());
    assert_eq!(
        disk.to_file_bytes().unwrap(),
        memory.to_file_bytes().unwrap()
    );
    assert!(disk.try_eq(&memory).unwrap());
    let mut index = 0;
    disk.for_each_id(|id| {
        assert_eq!(id, format!("{index:064x}"));
        index += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(index, 70_000);
    assert!(
        disk.ids(&small).is_err(),
        "materializing is explicitly budget checked"
    );
    let deselected = (0..1000)
        .map(|index| format!("{index:064x}"))
        .collect::<Vec<_>>();
    disk.deselect_ids(&deselected, &small).unwrap();
    memory.deselect_ids(&deselected, &large).unwrap();
    assert_eq!(
        disk.to_file_bytes().unwrap(),
        memory.to_file_bytes().unwrap()
    );
    let root = std::env::temp_dir().join(format!("l2d-marks-spill-save-{}", std::process::id()));
    let paths = l2d_core::ProjectPaths::new(&root);
    disk.save(&paths, &small).unwrap();
    let loaded = MarkSet::load(&paths, "many", &small).unwrap();
    assert!(loaded.ids_spilled());
    assert!(loaded.try_eq(&disk).unwrap());
    assert_eq!(loaded.selected_count(), 69000);
    assert!(!loaded.is_selected_id(&format!("{:064x}", 5)).unwrap());
    loaded.save(&paths, &small).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn large_non_id_json_metadata_is_refused_before_unbounded_deserialization() {
    use std::io::Write;
    let root =
        std::env::temp_dir().join(format!("l2d-marks-spill-metadata-{}", std::process::id()));
    let paths = l2d_core::ProjectPaths::new(&root);
    std::fs::create_dir_all(paths.marks_dir()).unwrap();
    let mut file = std::fs::File::create(paths.mark_set_path("many")).unwrap();
    file.write_all(b"{\"format\":1,\"kind\":\"mark-set\",\"name\":\"")
        .unwrap();
    for _ in 0..1024 {
        file.write_all(&[b'x'; 1024]).unwrap();
    }
    file.write_all(b"\"}").unwrap();
    drop(file);
    let error = MarkSet::load(&paths, "many", &budget(2 * 1024 * 1024)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("metadata exceeds the memory allowance"),
        "{error}"
    );
    std::fs::remove_dir_all(root).unwrap();
}
