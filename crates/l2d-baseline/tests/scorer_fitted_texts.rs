//! The hashes of the texts a baseline scorer was fitted on (task 1.11, requirement 11.1):
//! computed by the fit, saved in `fitted-texts.bin`, verified on load, merged on a save
//! of the same model, and counted in the memory a fit asks for.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_baseline::scorer::{
    FITTED_TEXTS_FILE, MANIFEST_FILE, TEXT_HASH_BYTES, TEXT_HASH_NAME, WEIGHTS_FILE, text_hash,
};
use l2d_baseline::{BaselineScorer, BaselineSettings};
use l2d_core::input::content_sha256;
use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LogRecord,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};

const GIB: u64 = 1024 * 1024 * 1024;

fn fixture() -> &'static (LogsFrame, InputSummary) {
    static FIXTURE: OnceLock<(LogsFrame, InputSummary)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("mini-labelled.csv");
        LabelledAdapter::new(LabelledFormat::Csv)
            .with_group_column("group")
            .read(&path)
            .expect("the fixture is readable")
    })
}

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB)).unwrap()
}

fn small() -> BaselineSettings {
    BaselineSettings {
        buckets: 1_024,
        ..BaselineSettings::default()
    }
}

fn fitted() -> &'static BaselineScorer {
    static FITTED: OnceLock<BaselineScorer> = OnceLock::new();
    FITTED.get_or_init(|| {
        let (logs, summary) = fixture();
        BaselineScorer::fit_with(logs, std::slice::from_ref(summary), &small(), &budget())
            .expect("the fixture fits")
    })
}

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-baseline-texts-{}-{}-{name}",
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

fn files_of(dir: &Path) -> Vec<String> {
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    files
}

fn refusal(error: L2dError) -> String {
    match error {
        L2dError::Refused { reason } => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// The hash the documentation states: the first 16 bytes of the SHA-256 of the text.
fn documented_hash(text: &str) -> [u8; 16] {
    let hex = content_sha256(text.as_bytes());
    let mut hash = [0_u8; 16];
    for (index, byte) in hash.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
    }
    hash
}

/// Rewrites a saved scorer into the form before task 1.11: no hash file, format 1, no
/// `fitted_texts` entry.
fn make_earlier_form(dir: &Path) {
    std::fs::remove_file(dir.join(FITTED_TEXTS_FILE)).unwrap();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    let object = manifest.as_object_mut().unwrap();
    object.insert("format".into(), 1.into());
    assert!(object.remove("fitted_texts").is_some());
    std::fs::write(
        dir.join(MANIFEST_FILE),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn a_fit_knows_every_labelled_text_it_was_fitted_on_and_no_other() {
    let (logs, _) = fixture();
    let scorer = fitted();
    assert!(scorer.has_fitted_text_hashes());
    assert_eq!(scorer.fitted_text_hashes(), Some(168));
    for record in logs.records() {
        let expected = record.label != Label::Unknown;
        assert_eq!(
            scorer.was_fitted_on_text(&record.text),
            Some(expected),
            "{} ({:?})",
            record.text,
            record.label
        );
    }
    assert_eq!(scorer.was_fitted_on_text(""), Some(false));
    assert_eq!(scorer.was_fitted_on_text("never seen"), Some(false));
    // A text is compared as its bytes: nothing is trimmed or folded.
    let first = logs.records().next().unwrap().text;
    assert_eq!(scorer.was_fitted_on_text(&format!("{first} ")), Some(false));
    assert_eq!(
        scorer.was_fitted_on_text(&first.to_uppercase()),
        Some(false)
    );
}

#[test]
fn the_text_hash_is_the_documented_one() {
    assert_eq!((TEXT_HASH_BYTES, TEXT_HASH_NAME), (16, "sha256-128"));
    for text in ["", "a", "GET /index.jsp?id=1", "\u{e9}t\u{e9}\n"] {
        assert_eq!(text_hash(text), documented_hash(text), "{text:?}");
    }
}

#[test]
fn the_hashes_are_saved_sorted_in_their_own_file_and_described_in_the_manifest() {
    let (logs, _) = fixture();
    let scorer = fitted();
    let scratch = Scratch::new("layout");
    let dir = scorer.save(&scratch.paths(), "texts").unwrap();
    assert_eq!(
        files_of(&dir),
        [MANIFEST_FILE, FITTED_TEXTS_FILE, WEIGHTS_FILE]
    );
    assert_eq!(FITTED_TEXTS_FILE, "fitted-texts.bin");

    let bytes = std::fs::read(dir.join(FITTED_TEXTS_FILE)).unwrap();
    assert_eq!(bytes.len(), 168 * 16);
    let stored: Vec<[u8; 16]> = bytes.as_chunks::<16>().0.to_vec();
    assert!(stored.windows(2).all(|pair| pair[0] < pair[1]), "sorted");
    let mut expected: Vec<[u8; 16]> = logs
        .records()
        .filter(|record| record.label != Label::Unknown)
        .map(|record| documented_hash(&record.text))
        .collect();
    expected.sort_unstable();
    assert_eq!(stored, expected);

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!(manifest["format"], 2);
    assert_eq!(manifest["fitted_texts"]["file"], "fitted-texts.bin");
    assert_eq!(manifest["fitted_texts"]["hash"], "sha256-128");
    assert_eq!(manifest["fitted_texts"]["count"], 168);
    assert_eq!(manifest["fitted_texts"]["sha256"], content_sha256(&bytes));

    let loaded = BaselineScorer::load(&dir).unwrap();
    assert_eq!(&loaded, scorer);
    assert_eq!(loaded.fitted_text_hashes(), Some(168));
}

#[test]
fn the_revision_does_not_depend_on_the_hashes() {
    let scorer = fitted();
    let scratch = Scratch::new("revision");
    let dir = scorer.save(&scratch.paths(), "texts").unwrap();
    make_earlier_form(&dir);
    let earlier = BaselineScorer::load(&dir).expect("the earlier form loads");
    assert_eq!(earlier.revision(), scorer.revision());
    assert!(!earlier.has_fitted_text_hashes());
    assert_eq!(earlier.was_fitted_on_text("anything"), None);
    assert_eq!(earlier.weights(), scorer.weights());
}

#[test]
fn saving_the_same_model_over_the_earlier_form_adds_the_hashes() {
    let scorer = fitted();
    let scratch = Scratch::new("upgrade");
    let dir = scorer.save(&scratch.paths(), "texts").unwrap();
    let weights = std::fs::read(dir.join(WEIGHTS_FILE)).unwrap();
    make_earlier_form(&dir);
    assert_eq!(files_of(&dir), [MANIFEST_FILE, WEIGHTS_FILE]);

    assert_eq!(scorer.save(&scratch.paths(), "texts").unwrap(), dir);
    assert_eq!(
        files_of(&dir),
        [MANIFEST_FILE, FITTED_TEXTS_FILE, WEIGHTS_FILE],
        "nothing temporary is left"
    );
    let loaded = BaselineScorer::load(&dir).unwrap();
    assert_eq!(&loaded, scorer);
    assert_eq!(std::fs::read(dir.join(WEIGHTS_FILE)).unwrap(), weights);

    // Saving again changes no file.
    let manifest = std::fs::read(dir.join(MANIFEST_FILE)).unwrap();
    assert_eq!(scorer.save(&scratch.paths(), "texts").unwrap(), dir);
    assert_eq!(std::fs::read(dir.join(MANIFEST_FILE)).unwrap(), manifest);

    // A scorer of the earlier form saved over the present form takes nothing away.
    let other = Scratch::new("upgrade-earlier");
    let earlier_dir = scorer.save(&other.paths(), "texts").unwrap();
    make_earlier_form(&earlier_dir);
    let earlier = BaselineScorer::load(&earlier_dir).unwrap();
    assert_eq!(earlier.save(&scratch.paths(), "texts").unwrap(), dir);
    assert_eq!(BaselineScorer::load(&dir).unwrap(), loaded);
}

#[test]
fn a_hash_file_that_does_not_verify_is_refused() {
    let scorer = fitted();
    let scratch = Scratch::new("corrupt");
    let dir = scorer.save(&scratch.paths(), "texts").unwrap();
    let path = dir.join(FITTED_TEXTS_FILE);
    let good = std::fs::read(&path).unwrap();
    let manifest_path = dir.join(MANIFEST_FILE);
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();

    // One bit changed.
    let mut changed = good.clone();
    changed[5] ^= 1;
    std::fs::write(&path, &changed).unwrap();
    let reason = refusal(BaselineScorer::load(&dir).unwrap_err());
    assert!(reason.contains(FITTED_TEXTS_FILE), "{reason}");

    // An entry removed.
    std::fs::write(&path, &good[16..]).unwrap();
    let reason = refusal(BaselineScorer::load(&dir).unwrap_err());
    assert!(reason.contains(FITTED_TEXTS_FILE), "{reason}");

    // The file gone: a missing item with its place.
    std::fs::remove_file(&path).unwrap();
    match BaselineScorer::load(&dir).unwrap_err() {
        L2dError::Missing { item, expected_at } => {
            assert!(item.contains(FITTED_TEXTS_FILE), "{item}");
            assert_eq!(expected_at, path);
        }
        other => panic!("{other:?}"),
    }
    std::fs::write(&path, &good).unwrap();
    assert_eq!(&BaselineScorer::load(&dir).unwrap(), scorer);

    // The manifest names another file: never opened.
    let outside = scratch.root.join("outside.bin");
    std::fs::write(&outside, &good).unwrap();
    for name in ["../../../outside.bin", "weights.bin", ""] {
        let edited = manifest.replacen(
            "\"file\": \"fitted-texts.bin\"",
            &format!("\"file\": \"{name}\""),
            1,
        );
        assert_ne!(edited, manifest);
        std::fs::write(&manifest_path, edited).unwrap();
        let reason = refusal(BaselineScorer::load(&dir).unwrap_err());
        assert!(reason.contains("fitted_texts"), "{name}: {reason}");
    }
    // Another hash rule, or a count that the file does not have.
    for (from, to) in [
        ("\"hash\": \"sha256-128\"", "\"hash\": \"md5\""),
        ("\"count\": 168", "\"count\": 167"),
    ] {
        let edited = manifest.replacen(from, to, 1);
        assert_ne!(edited, manifest);
        std::fs::write(&manifest_path, edited).unwrap();
        let reason = refusal(BaselineScorer::load(&dir).unwrap_err());
        assert!(reason.contains("fitted"), "{from}: {reason}");
    }
    // Format 1 never had the entry.
    let edited = manifest.replacen("\"format\": 2", "\"format\": 1", 1);
    std::fs::write(&manifest_path, edited).unwrap();
    assert!(matches!(
        BaselineScorer::load(&dir).unwrap_err(),
        L2dError::Refused { .. }
    ));
    // A format of the future is not guessed at.
    let edited = manifest.replacen("\"format\": 2", "\"format\": 3", 1);
    std::fs::write(&manifest_path, edited).unwrap();
    let reason = refusal(BaselineScorer::load(&dir).unwrap_err());
    assert!(reason.contains("format 3"), "{reason}");
}

#[test]
fn the_fit_counts_the_hashes_in_its_memory_estimate_and_the_scorer_says_what_it_keeps() {
    // 2,000 labelled rows of one short text each: the hashes are 32,000 bytes.
    let records: Vec<LogRecord> = (0..2_000)
        .map(|index| LogRecord {
            text: format!(
                "row {index:04} {}",
                if index % 2 == 0 { "ok" } else { "' OR 1=1" }
            ),
            label: if index % 2 == 0 {
                Label::Good
            } else {
                Label::Bad
            },
            group: None,
            first_position: index + 1,
            occurrences: 1,
        })
        .collect();
    let logs = LogsFrame::from_records(&records).unwrap();
    let settings = BaselineSettings {
        buckets: 256,
        max_iter: 5,
        ..BaselineSettings::default()
    };
    let scorer = BaselineScorer::fit_with(&logs, &[], &settings, &budget()).unwrap();
    let report = scorer.fit_report();
    let longest = records.iter().map(|r| r.text.len() as u64).max().unwrap();
    // Above the sample size the entries are an estimate; everything else is exact.
    let without_entries = (report.rows + 1) * 8
        + report.rows
        + 48 * 257 * 8
        + 6 * report.rows * 8
        + longest * 3 * 40
        + report.rows * 16;
    assert!(report.estimated_bytes >= without_entries + report.stored_entries * 12);
    assert!(
        report.estimated_bytes < without_entries + report.stored_entries * 12 * 2,
        "{report:?}"
    );

    assert_eq!(scorer.fitted_text_hashes(), Some(2_000));
    let kept = scorer.held_bytes();
    assert!(kept >= 256 * 8 + 2_000 * 16, "{kept}");
    assert!(kept < 256 * 8 + 2_000 * 16 + 4_096, "{kept}");

    // What a load needs is stated from the files, before anything is read.
    let scratch = Scratch::new("load-bytes");
    let dir = scorer.save(&scratch.paths(), "big").unwrap();
    let needed = BaselineScorer::load_bytes(&dir);
    assert!(needed >= 2 * 256 * 8 + 2_000 * 16, "{needed}");
    assert!(needed < 256 * 1024, "{needed}");
    assert_eq!(BaselineScorer::load_bytes(&scratch.root.join("absent")), 0);
}

/// Rewrites the first fitting input of a saved scorer as another input: another path
/// and another content hash.
fn rename_fitting_input(dir: &Path) {
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    let input = &mut manifest["fitting_inputs"][0];
    input["source_path"] = "earlier.csv".into();
    input["content_sha256"] = "e".repeat(64).into();
    std::fs::write(
        dir.join(MANIFEST_FILE),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn hashes_merged_into_an_earlier_form_fitted_on_another_input_are_marked_incomplete() {
    let scorer = fitted();
    assert_eq!(scorer.fitted_text_hashes_complete(), Some(true));

    // The earlier form names an input this fit did not read: its texts are unknown.
    let scratch = Scratch::new("incomplete");
    let dir = scorer.save(&scratch.paths(), "texts").unwrap();
    make_earlier_form(&dir);
    rename_fitting_input(&dir);
    let earlier = BaselineScorer::load(&dir).unwrap();
    assert_eq!(earlier.fitted_text_hashes_complete(), None);

    assert_eq!(scorer.save(&scratch.paths(), "texts").unwrap(), dir);
    let merged = BaselineScorer::load(&dir).unwrap();
    assert_eq!(merged.fitted_text_hashes(), Some(168));
    assert_eq!(
        merged.fitted_text_hashes_complete(),
        Some(false),
        "the hashes cover the new fit only, not the earlier input"
    );
    assert_eq!(merged.fitting_inputs().len(), 2);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!(manifest["fitted_texts"]["complete"], false);

    // It stays incomplete, whoever saves over it again, and a complete set merged
    // into another directory of the incomplete one becomes incomplete too.
    assert_eq!(scorer.save(&scratch.paths(), "texts").unwrap(), dir);
    assert_eq!(merged.save(&scratch.paths(), "texts").unwrap(), dir);
    let again = BaselineScorer::load(&dir).unwrap();
    assert_eq!(again, merged);
    let other = Scratch::new("incomplete-spread");
    let other_dir = scorer.save(&other.paths(), "texts").unwrap();
    assert_eq!(merged.save(&other.paths(), "texts").unwrap(), other_dir);
    assert_eq!(
        BaselineScorer::load(&other_dir)
            .unwrap()
            .fitted_text_hashes_complete(),
        Some(false)
    );

    // The earlier form fitted on the very input of this fit: the hashes cover it.
    let same = Scratch::new("complete");
    let same_dir = scorer.save(&same.paths(), "texts").unwrap();
    make_earlier_form(&same_dir);
    assert_eq!(scorer.save(&same.paths(), "texts").unwrap(), same_dir);
    let upgraded = BaselineScorer::load(&same_dir).unwrap();
    assert_eq!(upgraded.fitted_text_hashes_complete(), Some(true));
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(same_dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!(manifest["fitted_texts"]["complete"], true);

    // An earlier form that names no input at all has unknown provenance: incomplete.
    let unknown = Scratch::new("unknown");
    let unknown_dir = scorer.save(&unknown.paths(), "texts").unwrap();
    make_earlier_form(&unknown_dir);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(unknown_dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    manifest["fitting_inputs"] = serde_json::json!([]);
    std::fs::write(
        unknown_dir.join(MANIFEST_FILE),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert_eq!(scorer.save(&unknown.paths(), "texts").unwrap(), unknown_dir);
    assert_eq!(
        BaselineScorer::load(&unknown_dir)
            .unwrap()
            .fitted_text_hashes_complete(),
        Some(false)
    );
}
