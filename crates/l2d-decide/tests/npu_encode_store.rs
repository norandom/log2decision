//! Hardware test of task 2.7: the fixture logs are encoded by `EncoderSession` on the NPU
//! and stored, with the session's runtime release beside the encoder identity.
//!
//! - `the_fixture_is_encoded_on_the_npu_and_stored`: the task's done-when on the device.
//!   The 169 distinct logs of `tests/fixtures/mini-labelled.csv` are encoded and added to
//!   a new store; the stored vectors are unit vectors of the identity's dimension and are
//!   bit for bit what the session gives the same texts in one call; adding the same logs
//!   again only adds their occurrences; a store of another runtime release refuses the
//!   session's vectors before a text is encoded; the run record is printed.
//!
//! What is asserted about the encoder is read from the session (identity, dimension,
//! release): the test packages the published model of task 2.1 because that is the model
//! there is, and holds no figure of it.
//!
//! No CPU provider is registered here or anywhere in the workspace. Everything that needs
//! no device is in `encode_store.rs`, in the default run.
//!
//! Run on request, in release, with no Python interpreter on `PATH` (as the hardware
//! tests of `l2d-encoder`; this file does not check `PATH` itself):
//!
//! ```text
//! $env:L2D_RUNTIME_DIR = "D:\Source\log2decision\runtime\ryzenai"
//! $env:L2D_SCIENCE_DIR = "D:\Source\log2decision\models\science-2.1"
//! $env:L2D_NPU_WORK_DIR = "D:\BuildArtifacts\log2decision\task-2.7"
//! cargo test --release -p l2d-decide --test npu_encode_store -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Without the runtime files, the NPU or `L2D_SCIENCE_DIR` the test fails; with
//! `L2D_GATE_ALLOW_SKIP=1` it prints `SKIPPED` with the reason and passes.

#![cfg(windows)]

use std::path::{Path, PathBuf};

use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    InputAdapter, LabelledAdapter, LabelledFormat, LogsFrame, MemoryBudget, MemorySettings,
    ProjectPaths,
};
use l2d_decide::{FillCause, encode_into_store};
use l2d_encoder::{
    EncoderArch, EncoderError, EncoderPackage, EncoderSession, InputMode, PackageParts,
};
use l2d_store::{Filter, Store, StoreError, VectorOrigin};

const TAG: &str = "npu_encode_store";
const ALLOW_SKIP_VARIABLE: &str = "L2D_GATE_ALLOW_SKIP";
/// The published model of task 2.1, as the hardware tests of `l2d-encoder` package it.
const PUBLISHED_STEM: &str = "split17-f0-s0-strip-bce-norm-unit";
const SOURCE: &str = "tests/fixtures/mini-labelled.csv";

/// The test cannot run on this host: a failure, unless a skip was asked for.
fn skip(reason: &str) {
    let allowed = std::env::var_os(ALLOW_SKIP_VARIABLE)
        .is_some_and(|value| value != "0" && !value.is_empty());
    assert!(
        allowed,
        "the hardware test could not run: {reason} (set {ALLOW_SKIP_VARIABLE}=1 to skip \
         instead of failing)"
    );
    println!("{TAG}: SKIPPED ({ALLOW_SKIP_VARIABLE} is set): {reason}");
}

/// A path in the plain drive form the provider was proven with (no `\\?\` prefix).
fn plain(path: &Path) -> PathBuf {
    let resolved = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("resolving {}: {error}", path.display()));
    PathBuf::from(resolved.to_string_lossy().trim_start_matches(r"\\?\"))
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

fn fixture_logs() -> LogsFrame {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv");
    let (logs, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&path)
        .expect("the fixture is readable");
    logs
}

/// Text, occurrences, truncation flag and vector bits of every stored log, in order.
fn stored(store: &Store) -> Vec<(String, u32, bool, Vec<u32>)> {
    let frame: DataFrame = store.vectors(&Filter::default()).unwrap();
    let text = frame.column(l2d_store::TEXT).unwrap().str().unwrap();
    let occurrences = frame.column(l2d_store::OCCURRENCES).unwrap().u32().unwrap();
    let truncated = frame.column(l2d_store::TRUNCATED).unwrap().bool().unwrap();
    let vector = frame.column(l2d_store::VECTOR).unwrap().list().unwrap();
    (0..frame.height())
        .map(|row| {
            (
                text.get(row).unwrap().to_string(),
                occurrences.get(row).unwrap(),
                truncated.get(row).unwrap(),
                vector
                    .get_as_series(row)
                    .unwrap()
                    .f32()
                    .unwrap()
                    .into_no_null_iter()
                    .map(f32::to_bits)
                    .collect(),
            )
        })
        .collect()
}

#[test]
#[ignore = "hardware test: needs the AMD runtime files and the NPU; run on request"]
fn the_fixture_is_encoded_on_the_npu_and_stored() {
    let Some(science) = std::env::var_os("L2D_SCIENCE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .filter(|dir| dir.is_dir())
    else {
        skip("L2D_SCIENCE_DIR does not name a directory (the published model of task 2.1)");
        return;
    };
    let work_dir = std::env::var_os("L2D_NPU_WORK_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("l2d-npu-work"));
    let project = work_dir.join("encode-store-project");
    std::fs::create_dir_all(&project)
        .unwrap_or_else(|error| panic!("creating {}: {error}", project.display()));
    let paths = ProjectPaths::new(plain(&project));
    println!(
        "{TAG}: scratch project {}; {} build",
        paths.root.display(),
        if cfg!(debug_assertions) {
            "DEBUG (timings mean nothing)"
        } else {
            "release"
        }
    );
    // The stores of an earlier run of this test.
    let _ = std::fs::remove_dir_all(paths.stores_dir());

    let file = |suffix: &str| read(&science.join(format!("{PUBLISHED_STEM}{suffix}")));
    let (model, float, table) = (file(".int8.onnx"), file(".f32.onnx"), file(".table.bin"));
    let record = read(&science.join("record.json"));
    let written = EncoderPackage::write(
        &paths,
        &PackageParts {
            name: "csic-science-2.1",
            arch: EncoderArch::V1,
            input_mode: InputMode::Payload,
            model_int8: &model,
            table: &table,
            training_record: &record,
            model_f32: Some(&float),
        },
    )
    .unwrap_or_else(|error| panic!("packaging the published model: {error}"));
    let package = EncoderPackage::open(&paths, written.package.dir())
        .unwrap_or_else(|error| panic!("{error}"));

    // NPU only: without the device or the runtime files there is no session, the reason
    // is the error's, and nothing is stored.
    let mut session = match EncoderSession::open(&package, &paths) {
        Ok(session) => session,
        Err(
            error @ (EncoderError::RuntimeMissing { .. } | EncoderError::NpuUnavailable { .. }),
        ) => {
            assert!(!paths.stores_dir().exists(), "nothing was stored");
            skip(&error.to_string());
            return;
        }
        Err(error) => panic!("{error}"),
    };
    let identity = session.info().identity.clone();
    let release = session.info().runtime_release.clone();
    let dimension = usize::from(identity.dimension);
    println!(
        "{TAG}: encoder {identity} open on the NPU; AMD Ryzen AI {release}; driver {}; {} of \
         {} nodes on the NPU",
        session
            .info()
            .driver_version
            .as_deref()
            .unwrap_or("unknown"),
        session.info().placement.npu_nodes,
        session.info().placement.all_nodes
    );

    let logs = fixture_logs();
    let rows = logs.len() as u64;
    let origin = VectorOrigin::made_under(&identity, &release);
    let mut store = Store::create(&paths, "fixture", origin).unwrap();
    assert_eq!(store.runtime_release(), Some(release.as_str()));
    assert_eq!(store.identity(), &identity);
    let mut budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();

    // --- Encoded and stored. ---
    let report = encode_into_store(&mut session, &mut store, &logs, SOURCE, &mut budget)
        .unwrap_or_else(|error| panic!("{error}"));
    println!(
        "{TAG}: {} logs: {} encoded, {} added, {} repeated, {} skipped; {} truncated, {} \
         longer than the input, {} empty inputs; {} adds, {} bytes per row planned, {} bytes \
         held by the store per add",
        report.logs,
        report.encoded,
        report.added,
        report.repeated,
        report.skipped.len(),
        report.truncated,
        report.longer_than_input,
        report.empty_inputs,
        report.adds,
        report.plan.bytes_per_row,
        report.add_held_bytes
    );
    assert_eq!(
        (report.encoded, report.added, report.repeated),
        (rows, rows, 0)
    );
    assert!(report.skipped.is_empty());
    assert_eq!(store.len().unwrap(), rows);
    assert_eq!(session.stats().texts, rows);

    // What is stored is what the session gives the same texts, bit for bit.
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    let borrowed: Vec<&str> = texts.iter().map(String::as_str).collect();
    let direct = session
        .encode(&borrowed)
        .unwrap_or_else(|error| panic!("{error}"));
    let first = stored(&store);
    assert_eq!(first.len(), texts.len());
    let mut longest_off_unit = 0f64;
    for (((text, _, truncated, vector), expected_text), encoded) in
        first.iter().zip(&texts).zip(&direct)
    {
        assert_eq!(text, expected_text);
        assert_eq!(vector.len(), dimension);
        let expected: Vec<u32> = encoded.vector.iter().map(|value| value.to_bits()).collect();
        assert_eq!(vector, &expected, "{text}");
        assert_eq!(*truncated, encoded.input.truncated, "{text}");
        let length = vector
            .iter()
            .map(|bits| f64::from(f32::from_bits(*bits)).powi(2))
            .sum::<f64>()
            .sqrt();
        longest_off_unit = longest_off_unit.max((length - 1.0).abs());
    }
    assert!(longest_off_unit < 1e-5, "unit vectors: {longest_off_unit}");
    println!(
        "{TAG}: {} stored vectors of dimension {dimension}, bit-identical to the session's \
         own; furthest from length 1: {longest_off_unit:.2e}",
        first.len()
    );

    // --- The same logs again: only the occurrences. ---
    let again = encode_into_store(&mut session, &mut store, &logs, SOURCE, &mut budget)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!((again.added, again.repeated), (0, rows));
    assert_eq!(store.len().unwrap(), rows);
    let second = stored(&store);
    for (before, after) in first.iter().zip(&second) {
        assert_eq!(after.1, 2 * before.1, "{}", before.0);
        assert_eq!(
            (&after.0, after.2, &after.3),
            (&before.0, before.2, &before.3)
        );
    }
    println!("{TAG}: added again: 0 new logs, {rows} repeated, every count doubled");

    // --- A store of another runtime release refuses the session's vectors. ---
    let other_release = format!("{release}-other");
    let mut other = Store::create(
        &paths,
        "other-release",
        VectorOrigin::made_under(&identity, &other_release),
    )
    .unwrap();
    let encoded_before = session.stats().texts;
    let error = encode_into_store(&mut session, &mut other, &logs, SOURCE, &mut budget)
        .expect_err("a store of another runtime release");
    println!("{TAG}: refused: {error}");
    assert!(
        matches!(
            &*error.cause,
            FillCause::Store(StoreError::ReleaseMismatch { store, given })
                if store.as_deref() == Some(other_release.as_str())
                    && given.as_deref() == Some(release.as_str())
        ),
        "{error}"
    );
    assert_eq!(session.stats().texts, encoded_before, "nothing was encoded");
    assert!(other.is_empty().unwrap());

    // --- The run record. ---
    let mut record = session.finish();
    report.record(&store, &mut record).unwrap();
    assert_eq!(record.settings["runtime"]["release"], release.as_str());
    assert_eq!(
        record.settings["store"]["runtime_release"],
        release.as_str()
    );
    assert_eq!(record.settings["store"]["added"], rows);
    println!("{TAG}: run record:\n{}", record.to_json().unwrap());
    println!(
        "{TAG}: PASSED: {rows} fixture logs encoded on the NPU and stored under {identity}, \
         AMD Ryzen AI {release}"
    );
}
