//! Contract tests for the shared core types (requirements 10.3, 10.4, 10.5, 10.6).

use std::path::{Path, PathBuf};

use l2d_core::{
    Device, EXIT_SUCCESS, EncoderIdentity, InputSummary, L2dError, Label, LogRecord, Placement,
    PlacementSource, ProjectPaths, RunKind, RunRecord, SkippedRecord,
};

fn scratch_root(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("l2d-core-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// --- 10.6: exit-code mapping -------------------------------------------------------------

#[test]
fn exit_codes_follow_the_design_mapping() {
    assert_eq!(EXIT_SUCCESS, 0);
    let invalid = L2dError::InvalidInput {
        detail: "bad option".into(),
    };
    let missing = L2dError::Missing {
        item: "store".into(),
        expected_at: PathBuf::from("stores/x"),
    };
    let refused = L2dError::Refused {
        reason: "too few labels".into(),
    };
    let runtime = L2dError::Runtime {
        detail: "database error".into(),
    };
    assert_eq!(invalid.exit_code(), 2);
    assert_eq!(missing.exit_code(), 3);
    assert_eq!(refused.exit_code(), 4);
    assert_eq!(runtime.exit_code(), 5);
}

#[test]
fn every_error_is_non_zero_and_states_what_failed() {
    let errors = [
        L2dError::InvalidInput {
            detail: "unknown adapter 'xml'".into(),
        },
        L2dError::Missing {
            item: "dataset csic-2010".into(),
            expected_at: PathBuf::from("data/csic-2010"),
        },
        L2dError::Refused {
            reason: "9 labelled logs, 10 required".into(),
        },
        L2dError::Runtime {
            detail: "provider failed to load".into(),
        },
    ];
    let details = [
        "unknown adapter 'xml'",
        "dataset csic-2010",
        "9 labelled logs, 10 required",
        "provider failed to load",
    ];
    for (error, detail) in errors.iter().zip(details) {
        assert_ne!(error.exit_code(), EXIT_SUCCESS);
        assert!(
            error.to_string().contains(detail),
            "message {:?} lacks {detail:?}",
            error.to_string()
        );
    }
}

// --- 10.5: missing items name the item and its expected location -------------------------

#[test]
fn missing_error_message_names_item_and_expected_location() {
    let expected_at = Path::new("models")
        .join("encoders")
        .join("bytecnn-0123456789ab");
    let error = L2dError::Missing {
        item: "encoder bytecnn".into(),
        expected_at: expected_at.clone(),
    };
    let message = error.to_string();
    assert!(message.contains("encoder bytecnn"), "{message}");
    assert!(
        message.contains(&expected_at.display().to_string()),
        "{message}"
    );
}

#[test]
fn require_reports_missing_item_with_expected_location() {
    let root = scratch_root("require");
    let paths = ProjectPaths::new(&root);

    let error = paths
        .require("runtime files", paths.ryzenai_dir())
        .unwrap_err();
    assert_eq!(
        error,
        L2dError::Missing {
            item: "runtime files".into(),
            expected_at: root.join("runtime").join("ryzenai")
        }
    );
    assert_eq!(error.exit_code(), 3);

    std::fs::create_dir_all(paths.ryzenai_dir()).unwrap();
    assert_eq!(
        paths.require("runtime files", paths.ryzenai_dir()).unwrap(),
        paths.ryzenai_dir()
    );
    std::fs::remove_dir_all(&root).unwrap();
}

// --- 10.3 / 10.4: project locations -------------------------------------------------------

#[test]
fn project_directories_are_fixed_below_the_root() {
    let root = PathBuf::from("project-root");
    let paths = ProjectPaths::new(&root);
    assert_eq!(paths.root, root);
    assert_eq!(paths.models_dir(), root.join("models"));
    assert_eq!(paths.encoders_dir(), root.join("models").join("encoders"));
    assert_eq!(paths.baselines_dir(), root.join("models").join("baselines"));
    assert_eq!(paths.data_dir(), root.join("data"));
    assert_eq!(paths.stores_dir(), root.join("stores"));
    assert_eq!(paths.runtime_dir(), root.join("runtime"));
    assert_eq!(paths.ryzenai_dir(), root.join("runtime").join("ryzenai"));
    assert_eq!(
        paths.runtime_cache_dir(),
        root.join("runtime").join("cache")
    );
    assert_eq!(paths.reports_dir(), root.join("reports"));
}

#[test]
fn encoder_and_baseline_directories_use_name_and_twelve_character_revision() {
    let paths = ProjectPaths::new("r");
    let revision = "0123456789abcdef0123456789abcdef01234567";
    assert_eq!(
        paths.encoder_dir("bytecnn", revision),
        Path::new("r")
            .join("models")
            .join("encoders")
            .join("bytecnn-0123456789ab")
    );
    assert_eq!(
        paths.baseline_dir("ngram", revision),
        Path::new("r")
            .join("models")
            .join("baselines")
            .join("ngram-0123456789ab")
    );
    assert_eq!(
        paths.calibrations_dir(),
        Path::new("r").join("models").join("calibrations")
    );
    assert_eq!(
        paths.calibration_dir("csic", revision),
        Path::new("r")
            .join("models")
            .join("calibrations")
            .join("csic-0123456789ab")
    );
    assert!(paths.is_under_models(&paths.calibration_dir("csic", revision)));
    // A revision shorter than twelve characters is used as is.
    assert_eq!(
        paths.encoder_dir("e", "abc"),
        paths.encoders_dir().join("e-abc")
    );
}

#[test]
fn only_paths_below_the_model_directory_are_accepted_for_encoders() {
    let paths = ProjectPaths::new("r");
    assert!(paths.is_under_models(&paths.encoder_dir("e", "0123456789abcdef")));
    assert!(!paths.is_under_models(&paths.reports_dir().join("e")));
    assert!(!paths.is_under_models(Path::new("elsewhere").join("models").join("e").as_path()));
    assert!(!paths.is_under_models(&paths.models_dir().join("..").join("reports")));
}

#[test]
fn dataset_resolves_to_data_directory_or_supplied_path() {
    let root = scratch_root("dataset");
    let paths = ProjectPaths::new(&root);

    // Neither exists: the missing item and its expected location are named.
    let error = paths.dataset("csic-2010", None).unwrap_err();
    assert_eq!(
        error,
        L2dError::Missing {
            item: "dataset csic-2010".into(),
            expected_at: root.join("data").join("csic-2010")
        }
    );
    let supplied = root.join("elsewhere").join("logs.csv");
    let error = paths.dataset("csic-2010", Some(&supplied)).unwrap_err();
    assert_eq!(
        error,
        L2dError::Missing {
            item: "dataset csic-2010".into(),
            expected_at: supplied.clone()
        }
    );

    // Project data directory.
    std::fs::create_dir_all(root.join("data").join("csic-2010")).unwrap();
    assert_eq!(
        paths.dataset("csic-2010", None).unwrap(),
        root.join("data").join("csic-2010")
    );

    // A path the analyst supplies wins over the data directory.
    std::fs::create_dir_all(supplied.parent().unwrap()).unwrap();
    std::fs::write(&supplied, "text\n").unwrap();
    assert_eq!(
        paths.dataset("csic-2010", Some(&supplied)).unwrap(),
        supplied
    );
    std::fs::remove_dir_all(&root).unwrap();
}

// --- Shared types and the run record's JSON form ------------------------------------------

#[test]
fn label_has_three_values_with_stable_json_names() {
    assert_eq!(serde_json::to_string(&Label::Good).unwrap(), "\"good\"");
    assert_eq!(serde_json::to_string(&Label::Bad).unwrap(), "\"bad\"");
    assert_eq!(
        serde_json::to_string(&Label::Unknown).unwrap(),
        "\"unknown\""
    );
    assert_eq!(
        serde_json::from_str::<Label>("\"bad\"").unwrap(),
        Label::Bad
    );
}

#[test]
fn encoder_identity_short_revision_is_first_twelve_characters() {
    let identity = EncoderIdentity {
        name: "bytecnn".into(),
        revision: "0123456789abcdef0123456789abcdef01234567".into(),
        dimension: 128,
    };
    assert_eq!(identity.short_revision(), "0123456789ab");
    assert_eq!(
        identity.to_string(),
        "bytecnn-0123456789ab (128 dimensions)"
    );
}

fn sample_run_record() -> RunRecord {
    RunRecord {
        kind: RunKind::Encode,
        started_utc: "2026-11-11T10:00:00Z".into(),
        duration_ms: 1234,
        inputs: vec![InputSummary {
            source_path: PathBuf::from("data/csic-2010/csic_database.csv"),
            records: 5,
            distinct_texts: 3,
            content_sha256: "ab".repeat(32),
            skipped: vec![SkippedRecord {
                position: 4,
                reason: "unterminated quote".into(),
            }],
            label_conflicts: 1,
        }],
        settings: serde_json::json!({ "batch": 32 }),
        seed: Some(7),
        encoder: Some(EncoderIdentity {
            name: "bytecnn".into(),
            revision: "0f".repeat(20),
            dimension: 128,
        }),
        device: Some(Device::Npu),
        driver_version: Some("32.0.203.280".into()),
        logs_per_second: Some(1500.5),
        placement: Some(Placement {
            npu_nodes: 12,
            cpu_nodes: 1,
            source: PlacementSource::ProviderReport,
        }),
    }
}

#[test]
fn run_record_round_trips_through_its_json_form() {
    let record = sample_run_record();
    let json = record.to_json().unwrap();
    assert_eq!(RunRecord::from_json(&json).unwrap(), record);

    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["kind"], "encode");
    assert_eq!(value["device"], "npu");
    assert_eq!(value["duration_ms"], 1234);
    assert_eq!(value["encoder"]["dimension"], 128);
    assert_eq!(value["inputs"][0]["skipped"][0]["position"], 4);
    assert_eq!(value["placement"]["source"], "provider_report");
}

#[test]
fn run_record_absent_optionals_are_null_and_bad_json_is_invalid_input() {
    let record = RunRecord {
        kind: RunKind::Train,
        started_utc: "2026-11-11T10:00:00Z".into(),
        duration_ms: 0,
        inputs: Vec::new(),
        settings: serde_json::Value::Null,
        seed: None,
        encoder: None,
        device: None,
        driver_version: None,
        logs_per_second: None,
        placement: None,
    };
    let json = record.to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value["encoder"].is_null());
    assert!(value["device"].is_null());
    assert_eq!(RunRecord::from_json(&json).unwrap(), record);

    let error = RunRecord::from_json("{ not json").unwrap_err();
    assert_eq!(error.exit_code(), 2);
    assert!(error.to_string().contains("run record"), "{error}");
}

#[test]
fn log_record_carries_text_label_group_position_and_occurrences() {
    let record = LogRecord {
        text: "GET /index.jsp".into(),
        label: Label::Good,
        group: Some("GET /index.jsp".into()),
        first_position: 0,
        occurrences: 2,
    };
    let json = serde_json::to_string(&record).unwrap();
    assert_eq!(serde_json::from_str::<LogRecord>(&json).unwrap(), record);
}
