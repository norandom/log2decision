//! A baseline scorer saved before task 1.11 changed the saved form still loads
//! (task 1.11: "keep scorers saved before this change loadable").
//!
//! `tests/fixtures/scorer-1.7/` is a project root written by the code of task 1.7 as it
//! stood at commit `de2666a`, before the scorer saved a hash of each text it was fitted
//! on: `models/baselines/before-1.11-<rev12>/` with `baseline.json` (format 1) and
//! `weights.bin`, `revision.txt` with the revision, and `ratios.txt` with the ratio of
//! every probe text as the bit pattern of the `f32` in hexadecimal, one per line, in the
//! order of [`probe_texts`].
//!
//! To regenerate: the reference is the code before task 1.11, so check out the last
//! commit before it, copy this file into it (the `regenerate` test uses nothing newer),
//! set `L2D_SCORER_FIXTURE_OUT` to an empty directory, run
//! `cargo test -p l2d-baseline --test scorer_fixture -- --ignored regenerate`, and
//! replace the fixture directory with what was written. On the present code the same
//! command writes a scorer in the present form, which is not what this fixture is for.

use std::path::{Path, PathBuf};

use l2d_baseline::{BaselineScorer, BaselineSettings, RoutingMode};
use l2d_core::{
    InputAdapter, LabelledAdapter, LabelledFormat, LogsFrame, MemoryBudget, MemorySettings,
    ProjectPaths,
};

const NAME: &str = "before-1.11";
const REVISION_FILE: &str = "revision.txt";
const RATIOS_FILE: &str = "ratios.txt";
const GIB: u64 = 1024 * 1024 * 1024;

fn mini_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv")
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("scorer-1.7")
}

fn settings() -> BaselineSettings {
    BaselineSettings {
        buckets: 512,
        routing: RoutingMode::Keep,
        ..BaselineSettings::default()
    }
}

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB)).unwrap()
}

/// The texts whose ratios the fixture records: every text of the mini fixture in frame
/// order, then an empty text and one the scorer has not seen.
fn probe_texts(logs: &LogsFrame) -> Vec<String> {
    let mut texts: Vec<String> = logs.records().map(|record| record.text).collect();
    texts.push(String::new());
    texts.push("never seen ' OR 1=1 --".to_string());
    texts
}

fn ratio_lines(scorer: &BaselineScorer, texts: &[String]) -> String {
    texts
        .iter()
        .map(|text| format!("{:08x}\n", scorer.ratio(text).to_bits()))
        .collect()
}

#[test]
#[ignore = "writes the fixture; see the module documentation"]
fn regenerate() {
    let out = PathBuf::from(
        std::env::var_os("L2D_SCORER_FIXTURE_OUT").expect("L2D_SCORER_FIXTURE_OUT is set"),
    );
    std::fs::create_dir_all(&out).unwrap();
    let (logs, summary) = LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&mini_fixture())
        .unwrap();
    // The stored source path is information only; a path of this checkout would differ
    // from host to host.
    let mut summary = summary;
    summary.source_path = PathBuf::from("tests/fixtures/mini-labelled.csv");
    let scorer = BaselineScorer::fit_with(&logs, &[summary], &settings(), &budget()).unwrap();
    let dir = scorer.save(&ProjectPaths::new(&out), NAME).unwrap();
    std::fs::write(out.join(REVISION_FILE), scorer.revision()).unwrap();
    std::fs::write(
        out.join(RATIOS_FILE),
        ratio_lines(&scorer, &probe_texts(&logs)),
    )
    .unwrap();
    println!("wrote {}", dir.display());
}

fn fixture_dir() -> PathBuf {
    let revision = std::fs::read_to_string(fixture_root().join(REVISION_FILE)).unwrap();
    ProjectPaths::new(fixture_root()).baseline_dir(NAME, revision.trim())
}

fn files_of(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect();
    files.sort();
    files
}

#[test]
fn a_scorer_saved_before_task_1_11_loads_and_gives_the_ratios_it_gave() {
    let dir = fixture_dir();
    let before = files_of(&dir);
    assert_eq!(
        before
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["baseline.json", "weights.bin"],
        "the fixture is in the form of task 1.7"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&before[0].1).unwrap();
    assert_eq!(manifest["format"], 1);
    assert!(manifest.get("fitted_texts").is_none());

    let scorer = BaselineScorer::load_under(&ProjectPaths::new(fixture_root()), &dir)
        .expect("a scorer of the earlier form loads");
    let revision = std::fs::read_to_string(fixture_root().join(REVISION_FILE)).unwrap();
    assert_eq!(scorer.revision(), revision.trim());
    assert_eq!(scorer.settings(), &settings());
    assert_eq!(scorer.fit_report().rows, 168);

    let (logs, summary) = LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&mini_fixture())
        .unwrap();
    let recorded = std::fs::read_to_string(fixture_root().join(RATIOS_FILE)).unwrap();
    assert_eq!(
        ratio_lines(&scorer, &probe_texts(&logs)),
        recorded,
        "bit-equal ratios"
    );

    // The whole-file check still works; the per-log check has nothing to go by.
    assert!(scorer.has_known_fitting_inputs());
    assert!(scorer.was_fitted_on(&summary.content_sha256));
    assert!(!scorer.has_fitted_text_hashes());
    assert_eq!(scorer.fitted_text_hashes(), None);
    let first = logs.records().next().unwrap().text;
    assert_eq!(scorer.was_fitted_on_text(&first), None);

    // Loading reads only.
    assert_eq!(files_of(&dir), before);
}

#[test]
fn a_scorer_of_the_earlier_form_saved_again_still_has_no_text_hashes() {
    let scorer = BaselineScorer::load(&fixture_dir()).unwrap();
    let root = std::env::temp_dir().join(format!("l2d-scorer-fixture-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let dir = scorer.save(&ProjectPaths::new(&root), "again").unwrap();
    let names: Vec<String> = files_of(&dir).into_iter().map(|(name, _)| name).collect();
    assert_eq!(names, ["baseline.json", "weights.bin"]);
    let loaded = BaselineScorer::load(&dir).unwrap();
    assert_eq!(loaded, scorer);
    assert!(!loaded.has_fitted_text_hashes());
    let _ = std::fs::remove_dir_all(&root);
}
