//! Task 2.4: the encoder package, its revision and its verification.
//!
//! Every test works in a project of its own in the temporary directory. No model is run:
//! a model file is only hashed and read for its interface.
//!
//! On request (reads the science gate's published files, which are not tracked):
//! `cargo test -p l2d-encoder --test encoder_package -- --ignored --nocapture`.

mod common;

use std::path::{Path, PathBuf};

use common::{Model, Parts, Project, fixture, link_dir, snapshot, table_bytes, tiny_arch};
use l2d_core::{L2dError, MemoryBudget, MemorySettings, ProjectPaths};
use l2d_encoder::package::{
    ENCODER_JSON, MODEL_F32_FILE, MODEL_INT8_FILE, REVISION_DOMAIN, TABLE_FILE,
    TRAINING_RECORD_FILE,
};
use l2d_encoder::{
    ByteCodec, ByteTable, EncoderArch, EncoderPackage, InputArch, InputMode, PackageParts,
    WriteOutcome,
};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Writes a package and returns it, whatever the outcome was.
fn write(paths: &ProjectPaths, parts: &PackageParts<'_>) -> Result<EncoderPackage, L2dError> {
    EncoderPackage::write(paths, parts).map(|written| written.package)
}

fn refusal(result: Result<EncoderPackage, L2dError>) -> String {
    match result {
        Err(L2dError::Refused { reason }) => reason,
        Err(other) => panic!("expected a refusal, got {other:?}"),
        Ok(package) => panic!("expected a refusal, got {}", package.identity()),
    }
}

fn missing(result: Result<EncoderPackage, L2dError>) -> (String, PathBuf) {
    match result {
        Err(L2dError::Missing { item, expected_at }) => (item, expected_at),
        Err(other) => panic!("expected a missing item, got {other:?}"),
        Ok(package) => panic!("expected a missing item, got {}", package.identity()),
    }
}

/// A project with the tiny package written into it.
fn written(tag: &str) -> (Project, ProjectPaths, Parts, EncoderPackage) {
    let project = Project::new(tag);
    let paths = project.paths();
    let parts = Parts::tiny();
    let package = write(&paths, &parts.parts()).unwrap();
    (project, paths, parts, package)
}

fn flip_bit(path: &Path, at: usize) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[at] ^= 0x01;
    std::fs::write(path, bytes).unwrap();
}

/// Rewrites `encoder.json` with one piece of text replaced by another.
fn edit_json(dir: &Path, from: &str, to: &str) {
    let path = dir.join(ENCODER_JSON);
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        text.matches(from).count(),
        1,
        "`{from}` must occur exactly once in:\n{text}"
    );
    std::fs::write(&path, text.replacen(from, to, 1)).unwrap();
}

// ---------------------------------------------------------------------------------------
// The revision.

/// The documented byte order, computed here without the crate's code.
fn revision_by_hand(arch: &EncoderArch, mode: &str, table: &[u8], model: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"l2d-encoder-package/1\n");
    hash.update(arch.input.input_bytes.to_le_bytes());
    hash.update(arch.input.channels.to_le_bytes());
    hash.update(arch.input.batch.to_le_bytes());
    hash.update(u32::from(arch.dimension).to_le_bytes());
    hash.update((mode.len() as u64).to_le_bytes());
    hash.update(mode.as_bytes());
    hash.update((table.len() as u64).to_le_bytes());
    hash.update(table);
    hash.update((model.len() as u64).to_le_bytes());
    hash.update(model);
    hex(&hash.finalize())[..40].to_string()
}

#[test]
fn the_revision_is_the_documented_hash_over_architecture_mode_table_and_model() {
    assert_eq!(REVISION_DOMAIN, b"l2d-encoder-package/1\n");
    let parts = Parts::tiny();
    let revision = EncoderPackage::revision_of(
        &parts.arch,
        parts.input_mode,
        &parts.table,
        &parts.model_int8,
    );
    assert_eq!(revision.len(), 40);
    assert!(
        revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
    assert_eq!(
        revision,
        revision_by_hand(&parts.arch, "payload", &parts.table, &parts.model_int8)
    );
    let whole = EncoderPackage::revision_of(
        &parts.arch,
        InputMode::Whole,
        &parts.table,
        &parts.model_int8,
    );
    assert_eq!(
        whole,
        revision_by_hand(&parts.arch, "whole", &parts.table, &parts.model_int8)
    );
}

#[test]
fn every_ingredient_changes_the_revision() {
    let parts = Parts::tiny();
    let of = |arch: &EncoderArch, mode: InputMode, table: &[u8], model: &[u8]| {
        EncoderPackage::revision_of(arch, mode, table, model)
    };
    let base = of(
        &parts.arch,
        parts.input_mode,
        &parts.table,
        &parts.model_int8,
    );
    let mut revisions = vec![("base", base.clone())];

    // One bit of the model, at the start, in the middle and at the end.
    for at in [0, parts.model_int8.len() / 2, parts.model_int8.len() - 1] {
        let mut model = parts.model_int8.clone();
        model[at] ^= 0x01;
        revisions.push((
            "model bit",
            of(&parts.arch, parts.input_mode, &parts.table, &model),
        ));
    }
    // One bit of the table.
    for at in [12, parts.table.len() - 1] {
        let mut table = parts.table.clone();
        table[at] ^= 0x80;
        revisions.push((
            "table bit",
            of(&parts.arch, parts.input_mode, &table, &parts.model_int8),
        ));
    }
    // Every field of the architecture. The revision function hashes what it is given;
    // that a package's description fits its model is checked when it is written.
    let mut arch = parts.arch;
    arch.input.input_bytes += 1;
    revisions.push((
        "input_bytes",
        of(&arch, parts.input_mode, &parts.table, &parts.model_int8),
    ));
    let mut arch = parts.arch;
    arch.input.channels += 1;
    revisions.push((
        "channels",
        of(&arch, parts.input_mode, &parts.table, &parts.model_int8),
    ));
    let mut arch = parts.arch;
    arch.input.batch += 1;
    revisions.push((
        "batch",
        of(&arch, parts.input_mode, &parts.table, &parts.model_int8),
    ));
    let mut arch = parts.arch;
    arch.dimension += 1;
    revisions.push((
        "dimension",
        of(&arch, parts.input_mode, &parts.table, &parts.model_int8),
    ));
    // Two fields exchanged: the fields are hashed in their places, not as a sum.
    let mut arch = parts.arch;
    (arch.input.input_bytes, arch.input.channels) = (arch.input.channels, arch.input.input_bytes);
    revisions.push((
        "fields exchanged",
        of(&arch, parts.input_mode, &parts.table, &parts.model_int8),
    ));
    // The input mode.
    revisions.push((
        "input mode",
        of(
            &parts.arch,
            InputMode::Whole,
            &parts.table,
            &parts.model_int8,
        ),
    ));
    // A byte moved from the end of the table to the start of the model: the lengths are
    // hashed, so the boundary between the two files counts.
    let (mut table, mut model) = (parts.table.clone(), parts.model_int8.clone());
    model.insert(0, table.pop().unwrap());
    revisions.push((
        "boundary",
        of(&parts.arch, parts.input_mode, &table, &model),
    ));

    for (at, (what, revision)) in revisions.iter().enumerate() {
        for (other_what, other) in &revisions[..at] {
            assert_ne!(revision, other, "{what} and {other_what} share a revision");
        }
    }
}

#[test]
fn name_training_record_float_model_and_file_times_do_not_change_the_revision() {
    let project = Project::new("revision-stable");
    let paths = project.paths();
    let parts = Parts::tiny();
    let first = write(&paths, &parts.parts()).unwrap();
    let revision = first.identity().revision.clone();
    assert_eq!(
        revision,
        EncoderPackage::revision_of(
            &parts.arch,
            parts.input_mode,
            &parts.table,
            &parts.model_int8
        )
    );

    // Another name: another directory, the same revision.
    let renamed = Parts {
        name: "other-name".into(),
        ..parts.clone()
    };
    let second = write(&paths, &renamed.parts()).unwrap();
    assert_eq!(second.identity().revision, revision);
    assert_ne!(second.dir(), first.dir());

    // Another training record and no float model, in another project (in the same one
    // the directory would be taken).
    let other_project = Project::new("revision-stable-2");
    let other = Parts {
        training_record: br#"{"kind":"train","seed":8,"note":"another run"}"#.to_vec(),
        model_f32: None,
        ..parts.clone()
    };
    std::thread::sleep(std::time::Duration::from_millis(30));
    let third = write(&other_project.paths(), &other.parts()).unwrap();
    assert_eq!(third.identity().revision, revision);

    // Files rewritten with the same bytes get new times; the package still opens under
    // the same revision.
    for file in [MODEL_INT8_FILE, TABLE_FILE, TRAINING_RECORD_FILE] {
        let path = first.dir().join(file);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, bytes).unwrap();
    }
    let reopened = EncoderPackage::open(&paths, first.dir()).unwrap();
    assert_eq!(reopened.identity().revision, revision);
}

// ---------------------------------------------------------------------------------------
// Round trip.

#[test]
fn a_written_package_opens_with_the_same_identity() {
    let (_project, paths, parts, written) = written("round-trip");
    let identity = written.identity().clone();
    assert_eq!(identity.name, "tiny");
    assert_eq!(identity.dimension, 4);
    assert_eq!(identity.revision.len(), 40);

    let dir = paths.encoder_dir("tiny", &identity.revision);
    assert_eq!(written.dir(), dir);
    assert_eq!(
        dir.file_name().unwrap().to_str().unwrap(),
        format!("tiny-{}", &identity.revision[..12])
    );
    assert_eq!(dir.parent().unwrap(), paths.encoders_dir());

    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            ENCODER_JSON,
            MODEL_F32_FILE,
            MODEL_INT8_FILE,
            TABLE_FILE,
            TRAINING_RECORD_FILE
        ]
    );
    assert_eq!(
        std::fs::read(dir.join(MODEL_INT8_FILE)).unwrap(),
        parts.model_int8
    );
    assert_eq!(std::fs::read(dir.join(TABLE_FILE)).unwrap(), parts.table);
    assert_eq!(
        std::fs::read(dir.join(TRAINING_RECORD_FILE)).unwrap(),
        parts.training_record
    );
    assert_eq!(
        std::fs::read(dir.join(MODEL_F32_FILE)).unwrap(),
        parts.model_f32.clone().unwrap()
    );
    // Nothing else is left in `models/encoders/`.
    assert_eq!(std::fs::read_dir(paths.encoders_dir()).unwrap().count(), 1);

    let opened = EncoderPackage::open(&paths, &dir).unwrap();
    assert_eq!(opened.identity(), &identity);
    assert_eq!(opened.arch(), &tiny_arch());
    assert_eq!(opened.input_mode(), InputMode::Payload);
    assert_eq!(opened.dir(), dir);
    assert_eq!(opened.model_path(), dir.join(MODEL_INT8_FILE));
    assert_eq!(opened.cache_key(), identity.revision);

    let model = opened.model();
    assert_eq!(model.input.name, "x");
    assert_eq!(model.input.element_type, "float32");
    assert_eq!(model.input.shape, [2, 3, 1, 8]);
    assert_eq!(model.output.name, "e");
    assert_eq!(model.output.element_type, "float32");
    assert_eq!(model.output.shape, [2, 4]);
    assert_eq!(model.input_quantization.tensor, "x_q");
    assert_eq!(model.input_quantization.element_type, "uint8");
    assert_eq!(model.input_quantization.scale, 0.015_625);
    assert_eq!(model.input_quantization.zero_point, 128);
    assert_eq!(model.output_quantization.tensor, "e_q");
    assert_eq!(model.output_quantization.scale, 0.031_372_55);
    assert_eq!(model.output_quantization.zero_point, 121);
    assert_eq!(written.model(), model);

    let files = opened.files();
    assert_eq!(
        files.iter().map(|file| file.name).collect::<Vec<_>>(),
        [
            MODEL_INT8_FILE,
            TABLE_FILE,
            TRAINING_RECORD_FILE,
            MODEL_F32_FILE
        ]
    );
    assert_eq!(files[0].bytes, parts.model_int8.len() as u64);
    assert_eq!(files[0].sha256, hex(&Sha256::digest(&parts.model_int8)));
    assert_eq!(files[1].sha256, hex(&Sha256::digest(&parts.table)));
    assert_eq!(
        files[2].sha256,
        hex(&Sha256::digest(&parts.training_record))
    );
}

#[test]
fn encoder_json_has_the_documented_shape() {
    let (_project, _paths, parts, written) = written("json-shape");
    let text = std::fs::read_to_string(written.dir().join(ENCODER_JSON)).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let revision = written.identity().revision.clone();
    let file = |bytes: &[u8]| serde_json::json!({ "bytes": bytes.len(), "sha256": hex(&Sha256::digest(bytes)) });
    let expected = serde_json::json!({
        "format": 1,
        "kind": "l2d-encoder",
        "identity": { "name": "tiny", "revision": revision, "dimension": 4 },
        "arch": {
            "input": { "input_bytes": 8, "channels": 3, "batch": 2 },
            "dimension": 4
        },
        "input_mode": "payload",
        "model": {
            "input": { "name": "x", "element_type": "float32", "shape": [2, 3, 1, 8] },
            "output": { "name": "e", "element_type": "float32", "shape": [2, 4] }
        },
        "quantization": {
            "input": {
                "tensor": "x_q", "element_type": "uint8",
                "scale": 0.015625, "zero_point": 128
            },
            "output": {
                "tensor": "e_q", "element_type": "uint8",
                "scale": 0.03137255, "zero_point": 121
            }
        },
        "files": {
            "model.int8.onnx": file(&parts.model_int8),
            "table.bin": file(&parts.table),
            "training-record.json": file(&parts.training_record),
            "model.f32.onnx": file(parts.model_f32.as_ref().unwrap())
        }
    });
    // The output scale is a 32-bit float; compare it as one.
    let scale =
        |json: &serde_json::Value| json["quantization"]["output"]["scale"].as_f64().unwrap() as f32;
    assert_eq!(scale(&json), scale(&expected));
    let mut json = json;
    json["quantization"]["output"]["scale"] = expected["quantization"]["output"]["scale"].clone();
    assert_eq!(json, expected);
    // The keys come in the order of the documentation, and the file ends with a line feed.
    let order: Vec<usize> = [
        "\"format\"",
        "\"kind\"",
        "\"identity\"",
        "\"arch\"",
        "\"input_mode\"",
        "\"model\"",
        "\"quantization\"",
        "\"files\"",
    ]
    .iter()
    .map(|key| text.find(key).unwrap())
    .collect();
    assert!(order.is_sorted(), "{text}");
    assert!(text.ends_with("}\n"));
}

#[test]
fn a_package_without_a_float_model_round_trips() {
    let project = Project::new("no-float");
    let paths = project.paths();
    let parts = Parts {
        model_f32: None,
        ..Parts::tiny()
    };
    let written = write(&paths, &parts.parts()).unwrap();
    assert!(!written.dir().join(MODEL_F32_FILE).exists());
    let text = std::fs::read_to_string(written.dir().join(ENCODER_JSON)).unwrap();
    assert!(!text.contains(MODEL_F32_FILE), "{text}");
    let opened = EncoderPackage::open(&paths, written.dir()).unwrap();
    assert_eq!(opened.identity(), written.identity());
    assert_eq!(opened.files().len(), 3);

    // A float model that appears later is a stray file.
    std::fs::write(written.dir().join(MODEL_F32_FILE), b"late").unwrap();
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains(MODEL_F32_FILE), "{reason}");
}

#[test]
fn the_package_hands_out_a_ready_codec() {
    let (_project, paths, parts, written) = written("codec");
    let opened = EncoderPackage::open(&paths, written.dir()).unwrap();
    let codec = opened.codec();
    assert_eq!(codec.arch(), &parts.arch.input);
    assert_eq!(codec.mode(), InputMode::Payload);

    let by_hand = ByteCodec::new(
        parts.arch.input,
        ByteTable::from_le_bytes(&parts.table, &parts.arch.input).unwrap(),
        InputMode::Payload,
    )
    .unwrap();
    for text in [
        "GET /a?id=1",
        "GET /a",
        "",
        "POST /b\nname=a+very+long+value",
    ] {
        let mut row = vec![0.0f32; parts.arch.input.row_values()];
        let mut expected = row.clone();
        let flags = codec.encode_text(text, &mut row);
        let expected_flags = by_hand.encode_text(text, &mut expected);
        assert_eq!(flags, expected_flags, "{text:?}");
        assert_eq!(row, expected, "{text:?}");
    }
    // `GET /a` has no payload part: the mode of the package is in force.
    assert!(codec.describe("GET /a").empty);
    assert!(!codec.describe("GET /a?id=1").empty);

    let batch = codec.new_batch();
    assert_eq!(batch.dims(), opened.model().input.shape.as_slice());
    let owned: ByteCodec = opened.into_codec();
    assert_eq!(owned.mode(), InputMode::Payload);
}

#[test]
fn the_same_package_written_twice_is_not_an_error_and_writes_nothing() {
    let project = Project::new("twice");
    let paths = project.paths();
    let parts = Parts::tiny();
    let first = EncoderPackage::write(&paths, &parts.parts()).unwrap();
    assert_eq!(first.outcome, WriteOutcome::Written);
    let before = snapshot(&paths.models_dir());
    std::thread::sleep(std::time::Duration::from_millis(30));
    let second = EncoderPackage::write(&paths, &parts.parts()).unwrap();
    assert_eq!(second.outcome, WriteOutcome::Unchanged);
    assert_eq!(second.package.identity(), first.package.identity());
    assert_eq!(second.package.dir(), first.package.dir());
    assert_eq!(snapshot(&paths.models_dir()), before);
}

#[test]
fn the_same_encoder_with_another_record_is_found_reported_and_not_overwritten() {
    let (_project, paths, parts, first) = written("other-record");
    let before = snapshot(&paths.models_dir());
    let record_before = std::fs::read(first.dir().join(TRAINING_RECORD_FILE)).unwrap();

    // Same model, table, architecture and mode, so the same revision and directory, but
    // another training record: a second training run that ended in the same encoder.
    let other = Parts {
        training_record: br#"{"kind":"train","seed":99}"#.to_vec(),
        ..parts.clone()
    };
    let second = EncoderPackage::write(&paths, &other.parts()).unwrap();
    assert_eq!(
        second.outcome,
        WriteOutcome::ExistsWithOtherRecord {
            differing: vec![TRAINING_RECORD_FILE]
        }
    );
    // The package returned is the one that was there, with its own record.
    assert_eq!(second.package.identity(), first.identity());
    assert_eq!(second.package.files(), first.files());
    assert_eq!(snapshot(&paths.models_dir()), before);
    assert_eq!(
        std::fs::read(first.dir().join(TRAINING_RECORD_FILE)).unwrap(),
        record_before
    );

    // Another float model, no float model, and both files at once.
    let other_float = Parts {
        model_f32: Some(
            Model {
                quantized: false,
                seed: 9,
                ..Model::tiny()
            }
            .bytes(),
        ),
        ..parts.clone()
    };
    let no_float = Parts {
        model_f32: None,
        ..parts.clone()
    };
    for changed in [&other_float, &no_float] {
        assert_eq!(
            EncoderPackage::write(&paths, &changed.parts())
                .unwrap()
                .outcome,
            WriteOutcome::ExistsWithOtherRecord {
                differing: vec![MODEL_F32_FILE]
            }
        );
    }
    let both = Parts {
        training_record: other.training_record.clone(),
        ..no_float.clone()
    };
    assert_eq!(
        EncoderPackage::write(&paths, &both.parts())
            .unwrap()
            .outcome,
        WriteOutcome::ExistsWithOtherRecord {
            differing: vec![TRAINING_RECORD_FILE, MODEL_F32_FILE]
        }
    );
    assert_eq!(snapshot(&paths.models_dir()), before);
}

/// Eight writers at once, of the same package and of the same encoder with eight
/// training records: one writes, the others find what it wrote, nobody fails, and the
/// package verifies.
#[test]
fn eight_writers_at_once_end_in_one_package_and_distinguishable_outcomes() {
    for same_record in [true, false] {
        let project = Project::new("eight");
        let paths = project.paths();
        let parts: Vec<Parts> = (0..8)
            .map(|writer| Parts {
                training_record: if same_record {
                    Parts::tiny().training_record
                } else {
                    format!(r#"{{"kind":"train","writer":{writer}}}"#).into_bytes()
                },
                ..Parts::tiny()
            })
            .collect();
        let barrier = std::sync::Barrier::new(parts.len());
        let (barrier, paths) = (&barrier, &paths);
        let results: Vec<_> = std::thread::scope(|scope| {
            let writers: Vec<_> = parts
                .iter()
                .map(|parts| {
                    scope.spawn(move || {
                        barrier.wait();
                        EncoderPackage::write(paths, &parts.parts())
                    })
                })
                .collect();
            writers
                .into_iter()
                .map(|writer| writer.join().unwrap())
                .collect()
        });
        let written: Vec<_> = results
            .into_iter()
            .map(|result: Result<_, L2dError>| result.unwrap())
            .collect();

        let count = |wanted: fn(&WriteOutcome) -> bool| {
            written
                .iter()
                .filter(|written| wanted(&written.outcome))
                .count()
        };
        assert_eq!(count(|outcome| *outcome == WriteOutcome::Written), 1);
        if same_record {
            assert_eq!(count(|outcome| *outcome == WriteOutcome::Unchanged), 7);
        } else {
            // A loser's record is not the stored one.
            assert_eq!(
                count(|outcome| matches!(
                    outcome,
                    WriteOutcome::ExistsWithOtherRecord { differing }
                        if differing == &[TRAINING_RECORD_FILE]
                )),
                7
            );
        }
        // Everyone got the one package, and it is the winner's.
        let winner = written
            .iter()
            .position(|written| written.outcome == WriteOutcome::Written)
            .unwrap();
        for one in &written {
            assert_eq!(one.package.identity(), written[winner].package.identity());
            assert_eq!(one.package.files(), written[winner].package.files());
        }
        let dir = written[winner].package.dir();
        assert_eq!(
            std::fs::read(dir.join(TRAINING_RECORD_FILE)).unwrap(),
            parts[winner].training_record
        );
        EncoderPackage::open(paths, dir).unwrap();
        // No temporary directory is left.
        assert_eq!(std::fs::read_dir(paths.encoders_dir()).unwrap().count(), 1);
    }
}

#[test]
fn what_is_in_the_place_and_is_no_such_package_is_refused_and_left_unchanged() {
    let (_project, paths, parts, first) = written("no-overwrite");

    // A damaged package in the place is not repaired by writing over it.
    flip_bit(&first.dir().join(TABLE_FILE), 40);
    let damaged = snapshot(&paths.models_dir());
    let reason = refusal(write(&paths, &parts.parts()));
    assert!(reason.contains("left unchanged"), "{reason}");
    assert!(reason.contains(TABLE_FILE), "{reason}");
    assert_eq!(snapshot(&paths.models_dir()), damaged);
}

#[test]
fn a_directory_left_by_an_interrupted_write_is_no_package_and_is_not_in_the_way() {
    let (_project, paths, parts, first) = written("leftover");
    // What a killed process leaves: a temporary sibling with some of the files.
    let leftover = paths.encoders_dir().join(format!(
        ".tiny-{}.writing-4242-0",
        &first.identity().revision[..12]
    ));
    std::fs::create_dir(&leftover).unwrap();
    std::fs::write(leftover.join(MODEL_INT8_FILE), &parts.model_int8).unwrap();
    std::fs::write(leftover.join(TABLE_FILE), &parts.table[..100]).unwrap();

    // It does not open, half or whole.
    let (item, at) = missing(EncoderPackage::open(&paths, &leftover));
    assert!(item.contains(ENCODER_JSON), "{item}");
    assert_eq!(at, leftover.join(ENCODER_JSON));
    for file in [
        ENCODER_JSON,
        MODEL_INT8_FILE,
        TABLE_FILE,
        TRAINING_RECORD_FILE,
        MODEL_F32_FILE,
    ] {
        std::fs::copy(first.dir().join(file), leftover.join(file)).unwrap();
    }
    let reason = refusal(EncoderPackage::open(&paths, &leftover));
    assert!(reason.contains("directory name"), "{reason}");

    // The completed package is as it was, and writing goes on beside the leftover.
    let opened = EncoderPackage::open(&paths, first.dir()).unwrap();
    assert_eq!(opened.identity(), first.identity());
    let again = write(&paths, &parts.parts()).unwrap();
    assert_eq!(again.identity(), first.identity());
    let other = Parts {
        name: "second".into(),
        ..parts.clone()
    };
    let second = write(&paths, &other.parts()).unwrap();
    assert_eq!(second.identity().name, "second");
    assert!(leftover.is_dir());
}

// ---------------------------------------------------------------------------------------
// Tampering.

#[test]
fn one_altered_byte_in_any_file_is_refused_naming_the_file() {
    for file in [
        MODEL_INT8_FILE,
        TABLE_FILE,
        TRAINING_RECORD_FILE,
        MODEL_F32_FILE,
    ] {
        let (_project, paths, _parts, written) = written("tamper-bit");
        let path = written.dir().join(file);
        let length = std::fs::metadata(&path).unwrap().len() as usize;
        for at in [0, length / 2, length - 1] {
            flip_bit(&path, at);
            let reason = refusal(EncoderPackage::open(&paths, written.dir()));
            assert!(reason.contains(file), "{file} at {at}: {reason}");
            assert!(reason.contains("SHA-256"), "{file} at {at}: {reason}");
            flip_bit(&path, at);
        }
        // Restored, it opens again.
        EncoderPackage::open(&paths, written.dir()).unwrap();
    }
}

#[test]
fn a_file_of_another_size_is_refused_naming_the_file_and_both_sizes() {
    for file in [
        MODEL_INT8_FILE,
        TABLE_FILE,
        TRAINING_RECORD_FILE,
        MODEL_F32_FILE,
    ] {
        let (_project, paths, _parts, written) = written("tamper-size");
        let path = written.dir().join(file);
        let mut bytes = std::fs::read(&path).unwrap();
        let before = bytes.len();
        bytes.push(0);
        std::fs::write(&path, &bytes).unwrap();
        let reason = refusal(EncoderPackage::open(&paths, written.dir()));
        assert!(reason.contains(file), "{reason}");
        assert!(reason.contains(&before.to_string()), "{reason}");
        assert!(reason.contains(&(before + 1).to_string()), "{reason}");

        bytes.truncate(before - 1);
        std::fs::write(&path, &bytes).unwrap();
        let reason = refusal(EncoderPackage::open(&paths, written.dir()));
        assert!(reason.contains(file), "{reason}");
    }
}

#[test]
fn a_missing_file_is_named_with_its_place() {
    for file in [
        ENCODER_JSON,
        MODEL_INT8_FILE,
        TABLE_FILE,
        TRAINING_RECORD_FILE,
        MODEL_F32_FILE,
    ] {
        let (_project, paths, _parts, written) = written("missing-file");
        std::fs::remove_file(written.dir().join(file)).unwrap();
        let (item, at) = missing(EncoderPackage::open(&paths, written.dir()));
        assert!(item.contains(file), "{item}");
        assert_eq!(at, written.dir().join(file));
    }
}

#[test]
fn a_missing_package_is_named_with_its_expected_location() {
    let project = Project::new("missing-package");
    let paths = project.paths();
    let dir = paths.encoder_dir("absent", "0123456789abcdef0123456789abcdef01234567");
    let (item, at) = missing(EncoderPackage::open(&paths, &dir));
    assert!(item.contains("encoder package"), "{item}");
    assert_eq!(at, dir);
    // Nothing was created on the way.
    assert!(!paths.models_dir().exists());

    let error = EncoderPackage::open(&paths, &dir).unwrap_err();
    assert_eq!(error.exit_code(), l2d_core::EXIT_MISSING);
    assert!(error.to_string().contains(&dir.display().to_string()));
}

#[test]
fn a_stray_entry_in_the_package_is_refused_naming_it() {
    let (_project, paths, _parts, written) = written("stray");
    std::fs::write(written.dir().join("notes.txt"), b"hello").unwrap();
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains("notes.txt"), "{reason}");
    std::fs::remove_file(written.dir().join("notes.txt")).unwrap();

    std::fs::create_dir(written.dir().join("cache")).unwrap();
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains("cache"), "{reason}");
    std::fs::remove_dir(written.dir().join("cache")).unwrap();

    // A file of the package that is a directory.
    std::fs::remove_file(written.dir().join(TABLE_FILE)).unwrap();
    std::fs::create_dir(written.dir().join(TABLE_FILE)).unwrap();
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains(TABLE_FILE), "{reason}");
}

#[test]
fn an_altered_encoder_json_is_refused() {
    let (_project, paths, parts, written) = written("tamper-json");
    let dir = written.dir().to_path_buf();
    let original = std::fs::read(dir.join(ENCODER_JSON)).unwrap();
    let revision = written.identity().revision.clone();
    let restore = || std::fs::write(dir.join(ENCODER_JSON), &original).unwrap();
    let refused = || refusal(EncoderPackage::open(&paths, &dir));

    // The stored revision: both revisions are named.
    let mut other_revision = revision.clone().into_bytes();
    other_revision[39] = if other_revision[39] == b'0' {
        b'1'
    } else {
        b'0'
    };
    let other_revision = String::from_utf8(other_revision).unwrap();
    edit_json(&dir, &revision, &other_revision);
    let reason = refused();
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    assert!(reason.contains(&revision), "{reason}");
    assert!(reason.contains(&other_revision), "{reason}");
    restore();

    // The input mode: the same files read in the other mode are another encoder.
    edit_json(&dir, "\"payload\"", "\"whole\"");
    let reason = refused();
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    assert!(reason.contains(&revision), "{reason}");
    let as_whole = EncoderPackage::revision_of(
        &parts.arch,
        InputMode::Whole,
        &parts.table,
        &parts.model_int8,
    );
    assert!(reason.contains(&as_whole), "{reason}");
    restore();

    // The name: the directory is named after it.
    edit_json(&dir, "\"name\": \"tiny\"", "\"name\": \"tinz\"");
    let reason = refused();
    assert!(reason.contains("directory name"), "{reason}");
    assert!(reason.contains("tinz"), "{reason}");
    restore();

    // The output dimension of the identity and of the architecture.
    edit_json(
        &dir,
        "\"dimension\": 4\n  },\n  \"arch\"",
        "\"dimension\": 5\n  },\n  \"arch\"",
    );
    let reason = refused();
    assert!(reason.contains("dimension"), "{reason}");
    restore();

    // What the file says about the model: the model file decides.
    edit_json(&dir, "\"zero_point\": 128", "\"zero_point\": 127");
    let reason = refused();
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    assert!(reason.contains(MODEL_INT8_FILE), "{reason}");
    assert!(
        reason.contains(
            "states the quantized input as `x_q` uint8 with scale 0.015625 and zero point \
             127, but in model.int8.onnx it is `x_q` uint8 with scale 0.015625 and zero point \
             128"
        ),
        "{reason}"
    );
    assert!(!reason.contains('{'), "{reason}");
    restore();
    edit_json(&dir, "0.015625", "0.015626");
    let reason = refused();
    assert!(reason.contains(MODEL_INT8_FILE), "{reason}");
    restore();
    edit_json(&dir, "\"name\": \"x\"", "\"name\": \"y\"");
    let reason = refused();
    assert!(reason.contains(MODEL_INT8_FILE), "{reason}");
    restore();

    // A recorded hash and a recorded size.
    let record_hash = hex(&Sha256::digest(&parts.training_record));
    let mut other_hash = record_hash.clone().into_bytes();
    other_hash[0] = if other_hash[0] == b'0' { b'1' } else { b'0' };
    edit_json(
        &dir,
        &record_hash,
        std::str::from_utf8(&other_hash).unwrap(),
    );
    let reason = refused();
    assert!(reason.contains(TRAINING_RECORD_FILE), "{reason}");
    restore();

    // Bytes that change no value: the file must be exactly what the tool writes.
    let mut spaced = original.clone();
    spaced.push(b'\n');
    std::fs::write(dir.join(ENCODER_JSON), &spaced).unwrap();
    let reason = refused();
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    restore();
    edit_json(&dir, "\"format\": 1", "\"format\":  1");
    let reason = refused();
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    restore();

    // Another format or kind.
    edit_json(&dir, "\"format\": 1", "\"format\": 2");
    let reason = refused();
    assert!(reason.contains("format 2"), "{reason}");
    restore();
    edit_json(&dir, "\"l2d-encoder\"", "\"l2d-baseline\"");
    let reason = refused();
    assert!(reason.contains("l2d-baseline"), "{reason}");
    restore();

    // Not JSON at all, and empty.
    std::fs::write(dir.join(ENCODER_JSON), b"{ not json").unwrap();
    let reason = refused();
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    std::fs::write(dir.join(ENCODER_JSON), b"").unwrap();
    let reason = refused();
    assert!(reason.contains(ENCODER_JSON), "{reason}");

    restore();
    EncoderPackage::open(&paths, &dir).unwrap();
}

#[test]
fn encoder_json_without_the_input_mode_is_refused() {
    let (_project, paths, _parts, written) = written("no-mode");
    edit_json(written.dir(), "  \"input_mode\": \"payload\",\n", "");
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    assert!(reason.contains("input_mode"), "{reason}");
}

#[test]
fn encoder_json_with_an_unknown_input_mode_is_refused() {
    let (_project, paths, _parts, written) = written("odd-mode");
    edit_json(written.dir(), "\"payload\"", "\"strip\"");
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    assert!(reason.contains("strip"), "{reason}");
}

#[test]
fn encoder_json_with_an_unknown_field_is_refused() {
    // At the top, in the identity, in the architecture, in its input, in a tensor, in the
    // quantization and in a file entry.
    for (anchor, addition) in [
        ("  \"kind\"", "  \"note\": 1,\n  \"kind\""),
        (
            "    \"name\": \"tiny\"",
            "    \"alias\": \"t\",\n    \"name\": \"tiny\"",
        ),
        (
            "    \"input\": {\n      \"input_bytes\"",
            "    \"layers\": 5,\n    \"input\": {\n      \"input_bytes\"",
        ),
        (
            "      \"input_bytes\"",
            "      \"pad_byte\": 0,\n      \"input_bytes\"",
        ),
        (
            "      \"name\": \"x\"",
            "      \"layout\": \"NCHW\",\n      \"name\": \"x\"",
        ),
        (
            "      \"tensor\": \"x_q\"",
            "      \"axis\": 1,\n      \"tensor\": \"x_q\"",
        ),
        (
            "    \"table.bin\": {\n",
            "    \"table.bin\": {\n      \"modified\": 1,\n",
        ),
    ] {
        let (_project, paths, _parts, written) = written("unknown-field");
        edit_json(written.dir(), anchor, addition);
        let reason = refusal(EncoderPackage::open(&paths, written.dir()));
        assert!(reason.contains(ENCODER_JSON), "{addition}: {reason}");
        assert!(reason.contains("unknown field"), "{addition}: {reason}");
    }
}

#[test]
fn encoder_json_with_an_architecture_that_does_not_validate_is_refused() {
    for (from, to, names) in [
        ("\"channels\": 3", "\"channels\": 0", "channels"),
        ("\"batch\": 2", "\"batch\": 5000", "batch"),
        (
            "\"input_bytes\": 8",
            "\"input_bytes\": 70000",
            "input_bytes",
        ),
        (
            "\"dimension\": 4\n  },\n  \"input_mode\"",
            "\"dimension\": 0\n  },\n  \"input_mode\"",
            "dimension",
        ),
    ] {
        let (_project, paths, _parts, written) = written("bad-arch");
        edit_json(written.dir(), from, to);
        let reason = refusal(EncoderPackage::open(&paths, written.dir()));
        assert!(reason.contains(ENCODER_JSON), "{to}: {reason}");
        assert!(reason.contains(names), "{to}: {reason}");
    }
}

#[test]
fn a_renamed_package_directory_is_refused() {
    let (_project, paths, _parts, written) = written("renamed");
    let revision = &written.identity().revision;
    for other in [
        format!("tiny-{}", "0".repeat(12)),
        format!("tinz-{}", &revision[..12]),
        format!("tiny-{}", &revision[..11]),
        format!("tiny-{revision}"),
    ] {
        let moved = paths.encoders_dir().join(&other);
        std::fs::rename(written.dir(), &moved).unwrap();
        let reason = refusal(EncoderPackage::open(&paths, &moved));
        assert!(reason.contains("directory name"), "{reason}");
        assert!(reason.contains(&other), "{reason}");
        assert!(
            reason.contains(&format!("tiny-{}", &revision[..12])),
            "{reason}"
        );
        std::fs::rename(&moved, written.dir()).unwrap();
    }
    EncoderPackage::open(&paths, written.dir()).unwrap();
}

#[test]
fn opening_never_writes() {
    let (project, paths, _parts, written) = written("read-only");
    let before = snapshot(&project.root);
    for _ in 0..2 {
        EncoderPackage::open(&paths, written.dir()).unwrap();
    }
    // Refusals do not write either.
    flip_bit(&written.dir().join(MODEL_INT8_FILE), 3);
    let damaged = snapshot(&project.root);
    refusal(EncoderPackage::open(&paths, written.dir()));
    assert_eq!(snapshot(&project.root), damaged);
    flip_bit(&written.dir().join(MODEL_INT8_FILE), 3);
    assert_eq!(
        snapshot(&project.root)
            .iter()
            .map(|(path, dir, bytes, _)| (path.clone(), *dir, *bytes))
            .collect::<Vec<_>>(),
        before
            .iter()
            .map(|(path, dir, bytes, _)| (path.clone(), *dir, *bytes))
            .collect::<Vec<_>>()
    );
}

#[test]
fn the_model_is_read_back_only_as_verified() {
    let (_project, paths, parts, written) = written("read-model");
    let opened = EncoderPackage::open(&paths, written.dir()).unwrap();
    assert_eq!(opened.read_model().unwrap(), parts.model_int8);

    // Altered after the package was opened: the bytes are not handed out.
    flip_bit(&opened.model_path(), 9);
    match opened.read_model() {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains(MODEL_INT8_FILE), "{reason}")
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    std::fs::remove_file(opened.model_path()).unwrap();
    match opened.read_model() {
        Err(L2dError::Missing { expected_at, .. }) => {
            assert_eq!(expected_at, opened.model_path())
        }
        other => panic!("expected a missing file, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// Only under the project's model directory (requirement 10.3).

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

#[test]
fn a_package_outside_the_model_directory_is_refused() {
    let (project, paths, _parts, written) = written("outside");
    let name = written.dir().file_name().unwrap().to_owned();
    let outside_message = |reason: &str| {
        assert!(reason.contains("model directory"), "{reason}");
        assert!(
            reason.contains(&paths.encoders_dir().display().to_string()),
            "{reason}"
        );
    };

    // A complete copy elsewhere in the project, and outside it.
    let elsewhere = project.root.join("elsewhere").join("encoders").join(&name);
    copy_dir(written.dir(), &elsewhere);
    outside_message(&refusal(EncoderPackage::open(&paths, &elsewhere)));
    let other_project = Project::new("outside-other");
    let foreign = other_project.paths().encoders_dir().join(&name);
    copy_dir(written.dir(), &foreign);
    outside_message(&refusal(EncoderPackage::open(&paths, &foreign)));
    // The same directory is fine for its own project.
    EncoderPackage::open(&other_project.paths(), &foreign).unwrap();

    // Below `models/`, but not a package directory of `models/encoders/`.
    let beside = paths.models_dir().join(&name);
    copy_dir(written.dir(), &beside);
    outside_message(&refusal(EncoderPackage::open(&paths, &beside)));
    let nested = paths.encoders_dir().join("group").join(&name);
    copy_dir(written.dir(), &nested);
    outside_message(&refusal(EncoderPackage::open(&paths, &nested)));

    // A path with a parent-directory step, although it ends at the real package.
    let dotted = paths.encoders_dir().join("..").join("encoders").join(&name);
    assert!(dotted.is_dir());
    outside_message(&refusal(EncoderPackage::open(&paths, &dotted)));
    let dotted_root = ProjectPaths::new(project.root.join("elsewhere").join(".."));
    let reason = refusal(EncoderPackage::open(
        &dotted_root,
        &dotted_root.encoders_dir().join(&name),
    ));
    assert!(reason.contains("model directory"), "{reason}");

    // A place that is outside is refused as outside whether or not anything is there.
    let absent = project.root.join("reports").join(&name);
    outside_message(&refusal(EncoderPackage::open(&paths, &absent)));

    EncoderPackage::open(&paths, written.dir()).unwrap();
}

#[test]
fn a_package_reached_through_a_link_is_refused() {
    let (project, paths, _parts, written) = written("linked");
    let name = written.dir().file_name().unwrap().to_owned();

    // The package directory is a link to a complete package outside the project.
    let outside = Project::new("linked-target");
    let target = outside.root.join("store").join(&name);
    copy_dir(written.dir(), &target);
    std::fs::remove_dir_all(written.dir()).unwrap();
    link_dir(written.dir(), &target);
    assert!(written.dir().join(ENCODER_JSON).is_file());
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains("link"), "{reason}");
    assert!(
        reason.contains(&paths.encoders_dir().display().to_string()),
        "{reason}"
    );

    // `models/` itself is a link to a directory outside the project.
    let linked_project = Project::new("linked-models");
    let linked_paths = linked_project.paths();
    let models_target = outside.root.join("models");
    copy_dir(&target, &models_target.join("encoders").join(&name));
    link_dir(&linked_paths.models_dir(), &models_target);
    let through = linked_paths.encoders_dir().join(&name);
    assert!(through.join(ENCODER_JSON).is_file());
    let reason = refusal(EncoderPackage::open(&linked_paths, &through));
    assert!(reason.contains("link"), "{reason}");

    // Writing through that link is refused too, and nothing arrives at its target.
    let before = snapshot(&outside.root);
    let other = Parts {
        name: "through-link".into(),
        ..Parts::tiny()
    };
    let reason = refusal(write(&linked_paths, &other.parts()));
    assert!(reason.contains("link"), "{reason}");
    assert_eq!(snapshot(&outside.root), before);
    drop(project);
}

#[test]
fn a_package_is_written_only_below_the_model_directory() {
    let project = Project::new("write-outside");
    let parts = Parts::tiny();
    let dotted = ProjectPaths::new(project.root.join("a").join(".."));
    let reason = refusal(write(&dotted, &parts.parts()));
    assert!(reason.contains("model directory"), "{reason}");
    assert!(snapshot(&project.root).is_empty());
}

// ---------------------------------------------------------------------------------------
// What is refused when a package is written.

fn write_error(parts: &Parts) -> L2dError {
    let project = Project::new("write-refused");
    let paths = project.paths();
    let error = write(&paths, &parts.parts()).unwrap_err();
    assert!(
        snapshot(&project.root)
            .iter()
            .all(|(_, is_dir, _, _)| *is_dir),
        "a refused write left a file behind"
    );
    assert!(
        !paths.encoders_dir().exists()
            || std::fs::read_dir(paths.encoders_dir()).unwrap().count() == 0,
        "a refused write left a directory behind"
    );
    error
}

#[test]
fn an_unusable_name_is_invalid_input() {
    for name in [
        "",
        ".hidden",
        "a/b",
        "a\\b",
        "a b",
        "..",
        &"n".repeat(65),
        "ümlaut",
    ] {
        let parts = Parts {
            name: name.into(),
            ..Parts::tiny()
        };
        match write_error(&parts) {
            L2dError::InvalidInput { detail } => assert!(detail.contains("name"), "{detail}"),
            other => panic!("{name:?}: expected invalid input, got {other:?}"),
        }
    }
    let project = Project::new("name-ok");
    let parts = Parts {
        name: "csic-2010_v1.a".into(),
        ..Parts::tiny()
    };
    write(&project.paths(), &parts.parts()).unwrap();
}

#[test]
fn a_model_that_does_not_fit_the_architecture_is_refused() {
    let refused = |parts: &Parts| match write_error(parts) {
        L2dError::Refused { reason } => reason,
        other => panic!("expected a refusal, got {other:?}"),
    };
    let with_model = |model: Model| Parts {
        model_int8: model.bytes(),
        ..Parts::tiny()
    };

    // The input shape against the architecture, field by field.
    for shape in [
        vec![3, 3, 1, 8],
        vec![2, 4, 1, 8],
        vec![2, 3, 2, 8],
        vec![2, 3, 1, 9],
        vec![2, 3, 8],
    ] {
        let reason = refused(&with_model(Model {
            input_shape: shape.clone(),
            ..Model::tiny()
        }));
        assert!(reason.contains(MODEL_INT8_FILE), "{shape:?}: {reason}");
        assert!(reason.contains("[2, 3, 1, 8]"), "{shape:?}: {reason}");
    }
    // The output shape against batch and dimension.
    for shape in [vec![2, 5], vec![3, 4], vec![2, 4, 1]] {
        let reason = refused(&with_model(Model {
            output_shape: shape.clone(),
            ..Model::tiny()
        }));
        assert!(reason.contains("[2, 4]"), "{shape:?}: {reason}");
    }
    // Element types other than 32-bit float.
    let reason = refused(&with_model(Model {
        input_type: common::DOUBLE,
        ..Model::tiny()
    }));
    assert!(reason.contains("float64"), "{reason}");
    let reason = refused(&with_model(Model {
        output_type: common::UINT8,
        ..Model::tiny()
    }));
    assert!(reason.contains("uint8"), "{reason}");

    // The architecture against the model, the other way round.
    let mut parts = Parts::tiny();
    parts.arch.dimension = 5;
    let reason = refused(&parts);
    assert!(reason.contains("[2, 5]"), "{reason}");

    // A float model in the place of the 8-bit one.
    let reason = refused(&with_model(Model {
        quantized: false,
        ..Model::tiny()
    }));
    assert!(reason.contains("QuantizeLinear"), "{reason}");

    // Not a model at all.
    let mut parts = Parts::tiny();
    parts.model_int8 = b"this is not a model".to_vec();
    let reason = refused(&parts);
    assert!(reason.contains(MODEL_INT8_FILE), "{reason}");
    parts.model_int8 = Vec::new();
    refused(&parts);
    // A model cut short.
    let whole = Model::tiny().bytes();
    parts.model_int8 = whole[..whole.len() - 7].to_vec();
    let reason = refused(&parts);
    assert!(reason.contains(MODEL_INT8_FILE), "{reason}");
}

#[test]
fn a_table_or_architecture_or_record_that_is_unusable_is_refused_when_written() {
    // The table of another width.
    let parts = Parts {
        table: table_bytes(4),
        ..Parts::tiny()
    };
    match write_error(&parts) {
        L2dError::Refused { reason } => assert!(reason.contains(TABLE_FILE), "{reason}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    // Row 0 of the table is not zero.
    let mut parts = Parts::tiny();
    parts.table[4..8].copy_from_slice(&1.0f32.to_le_bytes());
    match write_error(&parts) {
        L2dError::Refused { reason } => assert!(reason.contains("row 0"), "{reason}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    // An architecture outside the accepted range.
    let mut parts = Parts::tiny();
    parts.arch.input.batch = 0;
    match write_error(&parts) {
        L2dError::InvalidInput { detail } => assert!(detail.contains("batch"), "{detail}"),
        other => panic!("expected invalid input, got {other:?}"),
    }
    let mut parts = Parts::tiny();
    parts.arch.dimension = 0;
    match write_error(&parts) {
        L2dError::InvalidInput { detail } => assert!(detail.contains("dimension"), "{detail}"),
        other => panic!("expected invalid input, got {other:?}"),
    }
    // A training record that is not JSON.
    let parts = Parts {
        training_record: b"seed = 7".to_vec(),
        ..Parts::tiny()
    };
    match write_error(&parts) {
        L2dError::InvalidInput { detail } => {
            assert!(detail.contains(TRAINING_RECORD_FILE), "{detail}")
        }
        other => panic!("expected invalid input, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// Memory.

fn budget(limit: u64) -> MemoryBudget {
    MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(limit),
            ..MemorySettings::default()
        },
        || Ok(64 << 30),
    )
    .unwrap()
}

#[test]
fn an_open_package_holds_the_table_and_small_things_but_not_the_model() {
    let project = Project::new("memory");
    let paths = project.paths();
    // A model of 4 MiB of weights.
    let model = Model {
        weights: 1 << 20,
        ..Model::tiny()
    };
    let parts = Parts {
        model_int8: model.bytes(),
        model_f32: None,
        ..Parts::tiny()
    };
    assert!(parts.model_int8.len() > 4 << 20);
    let written = write(&paths, &parts.parts()).unwrap();
    let opened = EncoderPackage::open(&paths, written.dir()).unwrap();

    // Held: the codec's table (256 x 3 x 4 bytes here) and under 4 KiB of names, hashes
    // and paths.
    let table = opened.codec().held_bytes();
    assert_eq!(table, 3_072);
    assert!(opened.held_bytes() >= table);
    assert!(
        opened.held_bytes() < table + 4_096,
        "{}",
        opened.held_bytes()
    );

    // While it is opened: the model once, the table twice, `encoder.json` and a buffer.
    let estimate = EncoderPackage::open_bytes(written.dir()).unwrap();
    assert!(estimate >= parts.model_int8.len() as u64 + 2 * parts.table.len() as u64);
    assert!(
        estimate < parts.model_int8.len() as u64 + (3 << 20),
        "{estimate}"
    );

    // Under a budget that leaves less, the package is not opened; with enough, it is.
    let small = budget((32 << 20) + (1 << 20));
    match EncoderPackage::open_with(&paths, written.dir(), &small) {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains("memory limit"), "{reason}");
            assert!(reason.contains("encoder package"), "{reason}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    let enough = budget((32 << 20) + estimate);
    let opened = EncoderPackage::open_with(&paths, written.dir(), &enough).unwrap();
    assert_eq!(opened.identity(), written.identity());
}

// ---------------------------------------------------------------------------------------
// The model reader.

#[test]
fn the_quantization_facts_are_read_from_every_storage_form() {
    use common::Storage;
    let reference = l2d_encoder::ModelInterface::from_onnx(&Model::tiny().bytes()).unwrap();
    for storage in [Storage::Typed, Storage::TypedUnpacked] {
        let model = Model {
            storage,
            ..Model::tiny()
        };
        let read = l2d_encoder::ModelInterface::from_onnx(&model.bytes()).unwrap();
        assert_eq!(read, reference, "{storage:?}");
    }
    // Signed 8-bit zero points, negative.
    for storage in [Storage::Raw, Storage::Typed, Storage::TypedUnpacked] {
        let model = Model {
            zero_type: common::INT8,
            input_zero: -3,
            output_zero: -128,
            storage,
            ..Model::tiny()
        };
        let read = l2d_encoder::ModelInterface::from_onnx(&model.bytes()).unwrap();
        assert_eq!(read.input_quantization.element_type, "int8", "{storage:?}");
        assert_eq!(read.input_quantization.zero_point, -3, "{storage:?}");
        assert_eq!(read.output_quantization.zero_point, -128, "{storage:?}");
    }
    // A scale that is not a positive finite number is not a scale.
    for scale in [0.0f32, -0.5, f32::NAN, f32::INFINITY] {
        let model = Model {
            input_scale: scale,
            ..Model::tiny()
        };
        let error = l2d_encoder::ModelInterface::from_onnx(&model.bytes()).unwrap_err();
        assert!(error.to_string().contains("scale"), "{error}");
    }
}

#[test]
fn the_tracked_model_fixture_packages_with_the_science_table() {
    // The hardware gate's fixture has the interface of the trained model (2.2 notes):
    // input `x` float32 [32, 64, 1, 256], output `e` float32 [32, 128].
    let project = Project::new("fixture");
    let paths = project.paths();
    let model = std::fs::read(fixture("spike-enc-b32-int8.onnx")).unwrap();
    let table = std::fs::read(fixture("science-2.1.table.bin")).unwrap();
    let parts = PackageParts {
        name: "spike",
        arch: EncoderArch::V1,
        input_mode: InputMode::Whole,
        model_int8: &model,
        table: &table,
        training_record: b"{}",
        model_f32: None,
    };
    let written = write(&paths, &parts).unwrap();
    let opened = EncoderPackage::open(&paths, written.dir()).unwrap();
    assert_eq!(opened.identity().dimension, 128);
    assert_eq!(opened.arch().input, InputArch::V1);
    assert_eq!(opened.model().input.name, "x");
    assert_eq!(opened.model().input.shape, [32, 64, 1, 256]);
    assert_eq!(opened.model().output.name, "e");
    assert_eq!(opened.model().output.shape, [32, 128]);
    assert_eq!(opened.model().input_quantization.element_type, "uint8");
    assert!(opened.model().input_quantization.scale > 0.0);
    assert_eq!(opened.codec().held_bytes(), 65_536);
    assert!(opened.held_bytes() < 65_536 + 4_096);
    assert_eq!(opened.codec().mode(), InputMode::Whole);
}

// ---------------------------------------------------------------------------------------
// Review round 1.

/// The bytes of `encoder.json` for the tiny package, as a literal. A change in how
/// `serde_json` indents or prints a 32-bit float, or in the order of the keys, would make
/// every existing package unopenable (the file must be byte for byte what the tool
/// writes); this test fails first.
#[test]
fn encoder_json_is_this_file_byte_for_byte() {
    let (_project, _paths, _parts, written) = written("literal");
    let text = std::fs::read_to_string(written.dir().join(ENCODER_JSON)).unwrap();
    assert_eq!(text, TINY_ENCODER_JSON);
}

/// Written line by line with explicit line feeds, so that a checkout that converts line
/// endings does not change it.
const TINY_ENCODER_JSON: &str = concat!(
    "{\n",
    "  \"format\": 1,\n",
    "  \"kind\": \"l2d-encoder\",\n",
    "  \"identity\": {\n",
    "    \"name\": \"tiny\",\n",
    "    \"revision\": \"6e0cad4ec57168576a2a1000a45513035a5f2764\",\n",
    "    \"dimension\": 4\n",
    "  },\n",
    "  \"arch\": {\n",
    "    \"input\": {\n",
    "      \"input_bytes\": 8,\n",
    "      \"channels\": 3,\n",
    "      \"batch\": 2\n",
    "    },\n",
    "    \"dimension\": 4\n",
    "  },\n",
    "  \"input_mode\": \"payload\",\n",
    "  \"model\": {\n",
    "    \"input\": {\n",
    "      \"name\": \"x\",\n",
    "      \"element_type\": \"float32\",\n",
    "      \"shape\": [\n",
    "        2,\n",
    "        3,\n",
    "        1,\n",
    "        8\n",
    "      ]\n",
    "    },\n",
    "    \"output\": {\n",
    "      \"name\": \"e\",\n",
    "      \"element_type\": \"float32\",\n",
    "      \"shape\": [\n",
    "        2,\n",
    "        4\n",
    "      ]\n",
    "    }\n",
    "  },\n",
    "  \"quantization\": {\n",
    "    \"input\": {\n",
    "      \"tensor\": \"x_q\",\n",
    "      \"element_type\": \"uint8\",\n",
    "      \"scale\": 0.015625,\n",
    "      \"zero_point\": 128\n",
    "    },\n",
    "    \"output\": {\n",
    "      \"tensor\": \"e_q\",\n",
    "      \"element_type\": \"uint8\",\n",
    "      \"scale\": 0.03137255,\n",
    "      \"zero_point\": 121\n",
    "    }\n",
    "  },\n",
    "  \"files\": {\n",
    "    \"model.int8.onnx\": {\n",
    "      \"bytes\": 647,\n",
    "      \"sha256\": \"dc81af85a73f0a8d4950f34d8eb80ec5f78110a699a631bbeb7e4d6979555774\"\n",
    "    },\n",
    "    \"table.bin\": {\n",
    "      \"bytes\": 3072,\n",
    "      \"sha256\": \"ef9b6a451d22ee077307318f5f951e0574d0685cbcea22ba1275919e1bc51f2a\"\n",
    "    },\n",
    "    \"training-record.json\": {\n",
    "      \"bytes\": 25,\n",
    "      \"sha256\": \"c2dbcbd0e39fc4ef96b1b3b1d9cfe1a21fd78326e47510babf7892eb2a9f484d\"\n",
    "    },\n",
    "    \"model.f32.onnx\": {\n",
    "      \"bytes\": 441,\n",
    "      \"sha256\": \"3cb8f97fb0e2cd41dae78a72ec23e3113b44cb488c7646ae69a182da662fb179\"\n",
    "    }\n",
    "  }\n",
    "}\n",
);

#[test]
fn a_directory_in_the_place_of_encoder_json_is_refused() {
    let (_project, paths, _parts, written) = written("json-directory");
    std::fs::remove_file(written.dir().join(ENCODER_JSON)).unwrap();
    std::fs::create_dir(written.dir().join(ENCODER_JSON)).unwrap();
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(
        reason.contains(&format!("{ENCODER_JSON} is not a plain file")),
        "{reason}"
    );
}

#[test]
fn an_encoder_json_above_the_size_that_is_read_is_refused() {
    let (_project, paths, _parts, written) = written("json-size");
    let path = written.dir().join(ENCODER_JSON);
    let mut bytes = std::fs::read(&path).unwrap();
    // White space after the document: still JSON, and one byte above 1 MiB.
    bytes.resize((1 << 20) + 1, b' ');
    std::fs::write(&path, &bytes).unwrap();
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains(ENCODER_JSON), "{reason}");
    assert!(reason.contains("1048577 bytes"), "{reason}");
    assert!(reason.contains("1048576"), "{reason}");

    // At exactly 1 MiB it is read, and refused for what it is.
    bytes.truncate(1 << 20);
    std::fs::write(&path, &bytes).unwrap();
    let reason = refusal(EncoderPackage::open(&paths, written.dir()));
    assert!(reason.contains("byte for byte"), "{reason}");
}

#[test]
fn a_model_file_with_two_graphs_is_refused() {
    let model = Model {
        graphs: 2,
        ..Model::tiny()
    };
    let error = l2d_encoder::ModelInterface::from_onnx(&model.bytes()).unwrap_err();
    assert!(error.to_string().contains("more than one graph"), "{error}");
    let parts = Parts {
        model_int8: model.bytes(),
        ..Parts::tiny()
    };
    match write_error(&parts) {
        L2dError::Refused { reason } => {
            assert!(reason.contains(MODEL_INT8_FILE), "{reason}");
            assert!(reason.contains("more than one graph"), "{reason}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_float_model_that_is_not_the_8_bit_models_is_refused_when_written() {
    let float = |change: fn(&mut Model)| {
        let mut model = Model {
            quantized: false,
            ..Model::tiny()
        };
        change(&mut model);
        model.bytes()
    };
    let whole = float(|_| {});
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("empty", Vec::new(), "no graph"),
        (
            "not a model",
            b"float model to follow".to_vec(),
            MODEL_F32_FILE,
        ),
        ("cut short", whole[..whole.len() - 9].to_vec(), "cut short"),
        (
            "another input shape",
            float(|model| model.input_shape = vec![2, 3, 1, 16]),
            "[2, 3, 1, 16]",
        ),
        (
            "another output shape",
            float(|model| model.output_shape = vec![2, 8]),
            "[2, 8]",
        ),
        (
            "another input name",
            float(|model| model.input_name = "input".into()),
            "`input`",
        ),
        (
            "another output name",
            float(|model| model.output_name = "out".into()),
            "`out`",
        ),
        (
            "another element type",
            float(|model| model.input_type = common::DOUBLE),
            "float64",
        ),
        (
            "two graphs",
            float(|model| model.graphs = 2),
            "more than one graph",
        ),
    ];
    for (what, bytes, names) in cases {
        let parts = Parts {
            model_f32: Some(bytes),
            ..Parts::tiny()
        };
        match write_error(&parts) {
            L2dError::Refused { reason } => {
                assert!(reason.contains(MODEL_F32_FILE), "{what}: {reason}");
                assert!(reason.contains(names), "{what}: {reason}");
            }
            other => panic!("{what}: expected a refusal, got {other:?}"),
        }
    }
}

#[test]
fn a_refusal_names_a_few_of_many_nodes_and_a_graph_has_a_ceiling() {
    // 100,000 nodes that read the graph input beside the QuantizeLinear.
    let model = Model {
        extra_nodes: 100_000,
        extra_read_input: true,
        ..Model::tiny()
    };
    let error = l2d_encoder::ModelInterface::from_onnx(&model.bytes()).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("it is read by 100001 node(s)"),
        "{message}"
    );
    assert!(
        message.contains("(`QuantizeLinear`, `Relu`, `Relu` and 99998 more)"),
        "{message}"
    );
    assert!(message.len() < 400, "{} bytes", message.len());

    // The same number of nodes that do not concern the input or the output: the model
    // reads as the tiny one.
    let model = Model {
        extra_nodes: 100_000,
        ..Model::tiny()
    };
    assert_eq!(
        l2d_encoder::ModelInterface::from_onnx(&model.bytes()).unwrap(),
        l2d_encoder::ModelInterface::from_onnx(&Model::tiny().bytes()).unwrap()
    );

    // More graph inputs than the reader keeps, and more dimensions.
    let mut graph = Vec::new();
    for at in 0..=l2d_encoder::onnx::MAX_GRAPH_TENSORS {
        let info = common::value_info(&format!("input-{at}"), common::FLOAT, &[1]);
        common::bytes_field(&mut graph, 11, &info);
    }
    let mut many = Vec::new();
    common::bytes_field(&mut many, 7, &graph);
    let error = l2d_encoder::ModelInterface::from_onnx(&many).unwrap_err();
    assert!(
        error.to_string().contains("more than 256 inputs"),
        "{error}"
    );
    let model = Model {
        input_shape: vec![1; l2d_encoder::onnx::MAX_TENSOR_DIMS + 1],
        ..Model::tiny()
    };
    let error = l2d_encoder::ModelInterface::from_onnx(&model.bytes()).unwrap_err();
    assert!(
        error.to_string().contains("more than 8 dimensions"),
        "{error}"
    );
    // A name too long to be shown is refused, and not shown whole.
    let model = Model {
        input_name: "n".repeat(100_000),
        ..Model::tiny()
    };
    let error = l2d_encoder::ModelInterface::from_onnx(&model.bytes()).unwrap_err();
    assert!(error.to_string().contains("100000 bytes"), "{error}");
    assert!(error.to_string().len() < 400);
}

/// On a file system that does not tell upper case from lower case, `TINY` beside `tiny`
/// with the same revision is the same directory: the write says so and what to do.
#[test]
fn a_name_that_differs_only_by_case_is_explained_where_the_file_system_merges_them() {
    let (_project, paths, parts, first) = written("case");
    let before = snapshot(&paths.models_dir());
    let shouted = Parts {
        name: "TINY".into(),
        ..parts.clone()
    };
    let shouted_dir = paths.encoder_dir("TINY", &first.identity().revision);
    if shouted_dir.is_dir() {
        let reason = refusal(write(&paths, &shouted.parts()));
        assert!(reason.contains("upper case"), "{reason}");
        assert!(
            reason.contains(&format!("`tiny-{}`", &first.identity().revision[..12])),
            "{reason}"
        );
        assert!(reason.contains("a name that differs"), "{reason}");
        assert!(reason.contains("left unchanged"), "{reason}");
        assert_eq!(snapshot(&paths.models_dir()), before);
        // Opening under the other spelling is refused the same way.
        let reason = refusal(EncoderPackage::open(&paths, &shouted_dir));
        assert!(reason.contains("upper case"), "{reason}");
    } else {
        let second = EncoderPackage::write(&paths, &shouted.parts()).unwrap();
        assert_eq!(second.outcome, WriteOutcome::Written);
    }
    EncoderPackage::open(&paths, first.dir()).unwrap();
}

/// On request: packages the model the science gate published (task 2.1) into a temporary
/// project, opens it, builds the codec and encodes a few texts. No model is run. The
/// files are read from the directory the environment variable `L2D_SCIENCE_DIR` names
/// (on the reference host `<main checkout>\models\science-2.1`); without it the test
/// says so and does nothing.
#[test]
#[ignore = "reads the science gate's published files, which are not tracked"]
fn the_science_gate_model_packages_and_opens() {
    let Some(source) = std::env::var_os("L2D_SCIENCE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    else {
        println!(
            "SKIPPED: set L2D_SCIENCE_DIR to the directory that holds the science gate's \
             files (split17-f0-s0-strip-bce-norm-unit.int8.onnx, .f32.onnx, .table.bin and \
             record.json); nothing was packaged"
        );
        return;
    };
    let stem = "split17-f0-s0-strip-bce-norm-unit";
    let read = |suffix: &str| {
        let path = source.join(format!("{stem}{suffix}"));
        std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    };
    let model = read(".int8.onnx");
    let float = read(".f32.onnx");
    let table = read(".table.bin");
    let record = std::fs::read(source.join("record.json")).unwrap();
    println!(
        "source: {} (int8 {} bytes {}, table {} bytes {})",
        source.display(),
        model.len(),
        hex(&Sha256::digest(&model)),
        table.len(),
        hex(&Sha256::digest(&table)),
    );

    let project = Project::new("science");
    let paths = project.paths();
    let parts = PackageParts {
        name: "csic-science-2.1",
        arch: EncoderArch::V1,
        // The gate model was fitted on the payload part (2.1 and 2.5 notes).
        input_mode: InputMode::Payload,
        model_int8: &model,
        table: &table,
        training_record: &record,
        model_f32: Some(&float),
    };
    let written = write(&paths, &parts).unwrap();
    let opened = EncoderPackage::open(&paths, written.dir()).unwrap();
    assert_eq!(opened.identity(), written.identity());
    println!("identity: {}", opened.identity());
    println!("revision: {}", opened.identity().revision);
    println!(
        "directory: {}",
        opened.dir().strip_prefix(&project.root).unwrap().display()
    );
    println!("model: {:?}", opened.model());
    for file in opened.files() {
        println!("file: {} {} bytes {}", file.name, file.bytes, file.sha256);
    }
    println!(
        "held: {} bytes; opening needs about {} bytes",
        opened.held_bytes(),
        EncoderPackage::open_bytes(opened.dir()).unwrap()
    );
    println!(
        "{}",
        std::fs::read_to_string(opened.dir().join(ENCODER_JSON)).unwrap()
    );

    assert_eq!(opened.identity().dimension, 128);
    assert_eq!(opened.model().input.shape, [32, 64, 1, 256]);
    assert_eq!(opened.model().output.shape, [32, 128]);

    let codec = opened.codec();
    let mut batch = codec.new_batch();
    let texts = [
        "GET http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=Jam%F3n+Ib%E9rico&precio=85&cantidad=%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre+LIKE+%27%25&B1=A%F1adir+al+carrito",
        "POST http://localhost:8080/tienda1/publico/entrar.jsp\nmodo=entrar&login=bob&pwd=secret&remember=off&B1=Entrar",
        "GET http://localhost:8080/tienda1/index.jsp",
        "",
    ];
    for text in texts {
        let flags = codec.push(&mut batch, text).unwrap();
        println!("text of {} bytes: {flags:?}", text.len());
    }
    assert_eq!(batch.rows(), texts.len());
    assert_eq!(batch.dims(), [32, 64, 1, 256]);
    assert!(batch.texts()[2].empty, "no payload part in `payload` mode");
    assert!(batch.values().iter().any(|value| *value != 0.0));
}
