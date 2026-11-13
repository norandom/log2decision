//! `ByteCodec` (task 2.5): a text becomes the model's input, with its flags.
//!
//! The decisive test holds the Rust mapping to the Python mapping the model was trained
//! with (`tools/science/l2dsci.py`: `byte_ids`, `table_input`, and the padding of the
//! last batch in `onnx_vectors`), bit for bit. The expected values are in
//! `tests/fixtures/byte-codec-expected.json`, written by `tools/science/codec_fixture.py`
//! with that code and the published byte table of the science gate; a copy of the table
//! (65,536 bytes, the same SHA-256) is `tests/fixtures/science-2.1.table.bin`. To
//! regenerate, from `tools\science`:
//!
//! ```text
//! uv run python codec_fixture.py D:\Source\log2decision\models\science-2.1 ..\..\crates\l2d-encoder\tests\fixtures
//! ```
//!
//! The fixture holds, per text and input mode, the byte ids, the truncation flag and the
//! SHA-256 of the float input; the tests compare the ids, the flag and the hash of the
//! codec's output, and also compute the input from the table file by the formula of the
//! 2.1 note, independently of the codec.

use std::path::{Path, PathBuf};

use l2d_core::L2dError;
use l2d_encoder::codec::{BYTE_VALUES, PAD_BYTE};
use l2d_encoder::{ByteCodec, ByteTable, InputArch, InputBatch, InputMode, TextInput};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// SHA-256 of the published table `split17-f0-s0-strip-bce-norm-unit.table.bin` (2.1 notes).
const TABLE_SHA256: &str = "f90c5f6812989e6bd087a67ea72bf6d5f3600ed37a7899e623fce30ac65af58e";

#[derive(Deserialize)]
struct Expected {
    table_sha256: String,
    input_bytes: u32,
    channels: u32,
    batch: u32,
    pad_byte: u8,
    texts: Vec<NamedText>,
    modes: std::collections::BTreeMap<String, ExpectedMode>,
}

#[derive(Deserialize)]
struct NamedText {
    name: String,
    text: String,
}

#[derive(Deserialize)]
struct ExpectedMode {
    l2dsci_routing: String,
    rows: Vec<ExpectedRow>,
    batches: Vec<ExpectedBatch>,
}

#[derive(Deserialize)]
struct ExpectedRow {
    ids: String,
    truncated: bool,
    input_sha256: String,
}

#[derive(Deserialize)]
struct ExpectedBatch {
    rows: usize,
    sha256: String,
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn expected() -> Expected {
    let text = std::fs::read_to_string(fixture("byte-codec-expected.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn table_bytes() -> Vec<u8> {
    std::fs::read(fixture("science-2.1.table.bin")).unwrap()
}

fn science_table() -> ByteTable {
    ByteTable::from_le_bytes(&table_bytes(), &InputArch::V1).unwrap()
}

fn codec(mode: InputMode) -> ByteCodec {
    ByteCodec::new(InputArch::V1, science_table(), mode).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|at| u8::from_str_radix(&text[2 * at..2 * at + 2], 16).unwrap())
        .collect()
}

/// SHA-256 of the values as little-endian 32-bit floats, in order.
fn sha256_of_values(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    hex(&digest.finalize())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The input of one text by the formula of the 2.1 note, straight from the table file:
/// `x[c, 0, t] = table[ids[t]][c]`, the table being 256 rows of `channels` little-endian
/// float32 in byte order. Nothing of the codec is used.
fn input_by_formula(table_file: &[u8], ids: &[u8], channels: usize) -> Vec<f32> {
    let length = ids.len();
    let mut out = vec![0.0f32; channels * length];
    for (t, &id) in ids.iter().enumerate() {
        for c in 0..channels {
            let at = (id as usize * channels + c) * 4;
            out[c * length + t] = f32::from_le_bytes(table_file[at..at + 4].try_into().unwrap());
        }
    }
    out
}

fn refusal(result: Result<impl std::fmt::Debug, L2dError>) -> String {
    match result {
        Err(L2dError::Refused { reason }) => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// Bit identity with the Python mapping
// ---------------------------------------------------------------------------------------

#[test]
fn the_fixture_is_the_one_made_with_the_published_table() {
    let expected = expected();
    assert_eq!(expected.table_sha256, TABLE_SHA256);
    assert_eq!(hex(&Sha256::digest(table_bytes())), TABLE_SHA256);
    let arch = InputArch::V1;
    assert_eq!(
        (arch.input_bytes, arch.channels, arch.batch),
        (expected.input_bytes, expected.channels, expected.batch)
    );
    assert_eq!(PAD_BYTE, expected.pad_byte);
    assert_eq!(BYTE_VALUES, 256);
    // Both modes, and the Python routing mode each one stands for.
    assert_eq!(expected.modes["whole"].l2dsci_routing, "keep");
    assert_eq!(expected.modes["payload"].l2dsci_routing, "strip");
    assert_eq!(expected.modes.len(), 2);
    // The cases the task names are in the set.
    let names: Vec<&str> = expected.texts.iter().map(|t| t.name.as_str()).collect();
    for name in [
        "empty",
        "whole-255",
        "whole-256",
        "whole-257",
        "payload-255",
        "payload-256",
        "payload-257",
        "whole-2byte-cut-at-256",
        "whole-3byte-cut-after-2",
        "whole-4byte-cut-after-3",
        "payload-3byte-cut-after-2",
        "nul-in-payload",
        "ascii-no-delimiter",
    ] {
        assert!(names.contains(&name), "{name} is missing from the fixture");
    }
    assert_eq!(expected.texts.len(), 41);
}

#[test]
fn ids_flags_and_inputs_are_bit_identical_to_the_python_mapping() {
    let expected = expected();
    let table_file = table_bytes();
    let mut compared = 0;
    for mode in InputMode::ALL {
        let codec = codec(mode);
        let rows = &expected.modes[mode.as_str()].rows;
        assert_eq!(rows.len(), expected.texts.len());
        let mut ids = vec![0xAAu8; 256];
        // The buffers are reused and start dirty: every position must be written.
        let mut input = vec![f32::NAN; InputArch::V1.row_values()];
        for (named, row) in expected.texts.iter().zip(rows) {
            let case = format!("{} ({})", named.name, mode.as_str());
            let python_ids = unhex(&row.ids);
            assert_eq!(python_ids.len(), 256, "{case}");

            let from_ids = codec.byte_ids(&named.text, &mut ids);
            assert_eq!(hex(&ids), row.ids, "{case}: byte ids");
            assert_eq!(from_ids.truncated, row.truncated, "{case}: truncated");

            let from_text = codec.encode_text(&named.text, &mut input);
            assert_eq!(from_text, from_ids, "{case}");
            assert_eq!(from_text, codec.describe(&named.text), "{case}");
            assert_eq!(
                sha256_of_values(&input),
                row.input_sha256,
                "{case}: the float input"
            );
            // The same from the table file by the formula, without the codec.
            let formula = input_by_formula(&table_file, &python_ids, 64);
            assert_eq!(sha256_of_values(&formula), row.input_sha256, "{case}");
            assert_eq!(bits(&input), bits(&formula), "{case}");
            compared += 1;
        }
    }
    assert_eq!(compared, 82);
}

#[test]
fn batches_are_bit_identical_to_the_ones_the_python_pipeline_feeds_the_model() {
    let expected = expected();
    for mode in InputMode::ALL {
        let codec = codec(mode);
        let python = &expected.modes[mode.as_str()].batches;
        assert_eq!(python.len(), 2, "41 texts are one full batch and one of 9");
        let mut texts = expected.texts.iter().map(|named| named.text.as_str());
        // One buffer for all batches: the second batch is shorter than the first, so its
        // tail must be all-zero inputs again.
        let mut batch = codec.new_batch();
        let mut flags: Vec<TextInput> = Vec::new();
        for (index, want) in python.iter().enumerate() {
            let rows = codec.fill(&mut batch, &mut texts);
            assert_eq!(rows, want.rows, "batch {index} ({})", mode.as_str());
            assert_eq!(batch.rows(), want.rows);
            assert_eq!(batch.values().len(), 32 * 64 * 256);
            assert_eq!(
                sha256_of_values(batch.values()),
                want.sha256,
                "batch {index} ({})",
                mode.as_str()
            );
            flags.extend_from_slice(batch.texts());
        }
        assert_eq!(codec.fill(&mut batch, &mut texts), 0, "no text is left");
        assert_eq!(batch.rows(), 0);
        assert!(batch.values().iter().all(|value| value.to_bits() == 0));
        // One flag set per text, in input order.
        let rows = &expected.modes[mode.as_str()].rows;
        assert_eq!(flags.len(), rows.len());
        for ((named, row), flag) in expected.texts.iter().zip(rows).zip(&flags) {
            assert_eq!(flag.truncated, row.truncated, "{}", named.name);
            assert_eq!(*flag, codec.describe(&named.text), "{}", named.name);
        }
    }
}

// ---------------------------------------------------------------------------------------
// The done criterion and the truncation rule
// ---------------------------------------------------------------------------------------

#[test]
fn a_300_byte_text_is_truncated_a_256_byte_text_is_not_and_the_input_has_the_models_dimensions() {
    let codec = codec(InputMode::Whole);
    let arch = *codec.arch();
    assert_eq!(arch, InputArch::V1);
    assert_eq!(
        (arch.input_bytes, arch.channels, arch.batch),
        (256, 64, 32),
        "architecture v1"
    );
    assert_eq!(arch.dims(), [32, 64, 1, 256]);
    assert_eq!(arch.row_values(), 64 * 256);
    assert_eq!(arch.batch_values(), 32 * 64 * 256);

    let long = "x".repeat(300);
    let exact = "x".repeat(256);
    let mut batch = codec.new_batch();
    let long_flags = codec.push(&mut batch, &long).unwrap();
    let exact_flags = codec.push(&mut batch, &exact).unwrap();
    assert!(long_flags.truncated);
    assert_eq!((long_flags.part_bytes, long_flags.read_bytes), (300, 256));
    assert!(!exact_flags.truncated);
    assert_eq!((exact_flags.part_bytes, exact_flags.read_bytes), (256, 256));
    assert!(!codec.describe(&"x".repeat(255)).truncated);
    assert!(codec.describe(&"x".repeat(257)).truncated);

    assert_eq!(batch.dims(), [32, 64, 1, 256]);
    assert_eq!(batch.values().len(), 32 * 64 * 256);
    assert_eq!(batch.rows(), 2);
    assert_eq!(batch.texts(), [long_flags, exact_flags]);
    // The first 256 bytes of both are the same, so both rows are the same input.
    let row = arch.row_values();
    assert_eq!(
        bits(&batch.values()[..row]),
        bits(&batch.values()[row..2 * row])
    );
    assert!(batch.values()[..row].iter().any(|value| *value != 0.0));
}

#[test]
fn truncation_counts_the_bytes_of_the_part_that_is_encoded() {
    // 7 bytes of routing and delimiter, then 256 bytes of payload.
    let text = format!("GET /p?{}", "v".repeat(256));
    let whole = codec(InputMode::Whole).describe(&text);
    assert_eq!((whole.part_bytes, whole.read_bytes), (263, 256));
    assert!(whole.truncated, "263 bytes are read whole");
    let payload = codec(InputMode::Payload).describe(&text);
    assert_eq!((payload.part_bytes, payload.read_bytes), (256, 256));
    assert!(!payload.truncated, "the payload fits exactly");

    let text = format!("GET /p?{}", "v".repeat(257));
    assert!(codec(InputMode::Payload).describe(&text).truncated);
    // A long routing part does not truncate a short payload.
    let text = format!("GET /{}?a=1", "p".repeat(400));
    let payload = codec(InputMode::Payload).describe(&text);
    assert_eq!((payload.part_bytes, payload.truncated), (3, false));
    assert!(codec(InputMode::Whole).describe(&text).truncated);
    // Bytes are counted, not characters: 200 two-byte characters are 400 bytes.
    let text = "\u{e9}".repeat(200);
    let flags = codec(InputMode::Whole).describe(&text);
    assert_eq!((flags.part_bytes, flags.truncated), (400, true));
    let text = "\u{e9}".repeat(128);
    let flags = codec(InputMode::Whole).describe(&text);
    assert_eq!((flags.part_bytes, flags.truncated), (256, false));
}

#[test]
fn the_cut_is_at_the_byte_also_inside_a_character() {
    let codec = codec(InputMode::Whole);
    let mut ids = vec![0u8; 256];
    // 255 ASCII bytes, then a two-byte character: its first byte is the last id.
    let text = format!("{}\u{e9}", "a".repeat(255));
    let flags = codec.byte_ids(&text, &mut ids);
    assert!(flags.truncated);
    assert_eq!(flags.read_bytes, 256);
    assert_eq!(ids[255], 0xC3, "the lead byte of the cut character stays");
    assert_eq!(ids[..], text.as_bytes()[..256]);
    // A four-byte character with three bytes inside the window.
    let text = format!("{}\u{1f600}", "a".repeat(253));
    codec.byte_ids(&text, &mut ids);
    assert_eq!(ids[253..], [0xF0, 0x9F, 0x98]);
    // The float input is that of the raw bytes.
    let mut input = vec![0.0f32; InputArch::V1.row_values()];
    codec.encode_text(&text, &mut input);
    assert_eq!(
        bits(&input),
        bits(&input_by_formula(&table_bytes(), &ids, 64))
    );
}

// ---------------------------------------------------------------------------------------
// Input modes, the empty input, NUL bytes
// ---------------------------------------------------------------------------------------

#[test]
fn the_two_input_modes_select_the_whole_text_or_the_payload_part() {
    assert_eq!(InputMode::ALL, [InputMode::Whole, InputMode::Payload]);
    let text = "POST /a/b.jsp?x=1\ny=2";
    assert_eq!(InputMode::Whole.select(text), text);
    assert_eq!(InputMode::Payload.select(text), "x=1\ny=2");
    assert_eq!(
        InputMode::Payload.select("POST /a\ny=2?z=3"),
        "y=2?z=3",
        "the first `?` or line feed, whichever comes first"
    );
    assert_eq!(InputMode::Payload.select("GET /a?"), "");
    assert_eq!(InputMode::Payload.select("?a=1"), "a=1");
    // No `?` and no line feed: routing only, the payload is empty.
    assert_eq!(InputMode::Payload.select("GET /tienda1/index.jsp"), "");
    assert_eq!(InputMode::Payload.select("sshd[411]: Failed password"), "");
    assert_eq!(InputMode::Whole.select(""), "");
    assert_eq!(InputMode::Payload.select(""), "");

    // The names that go into the encoder identity.
    assert_eq!(InputMode::Whole.as_str(), "whole");
    assert_eq!(InputMode::Payload.as_str(), "payload");
    for mode in InputMode::ALL {
        assert_eq!(InputMode::parse(mode.as_str()).unwrap(), mode);
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, format!("\"{}\"", mode.as_str()));
        assert_eq!(serde_json::from_str::<InputMode>(&json).unwrap(), mode);
        assert_eq!(codec(mode).mode(), mode);
    }
    // No other name is understood, the baseline's among them: nothing strips silently.
    for name in ["strip", "keep", "isolate", "Whole", "PAYLOAD", ""] {
        let error = InputMode::parse(name).unwrap_err();
        assert!(
            matches!(&error, L2dError::InvalidInput { detail }
                if detail.contains("whole") && detail.contains("payload")),
            "{name}: {error}"
        );
        assert!(serde_json::from_str::<InputMode>(&format!("\"{name}\"")).is_err());
    }
}

#[test]
fn a_text_without_delimiter_is_an_empty_input_in_payload_mode_and_is_flagged() {
    let line = "Nov 13 10:02:11 host sshd[411]: Failed password for root";
    let whole = codec(InputMode::Whole);
    let payload = codec(InputMode::Payload);
    let mut input = vec![1.0f32; InputArch::V1.row_values()];

    let flags = payload.encode_text(line, &mut input);
    assert!(flags.empty);
    assert!(!flags.truncated);
    assert_eq!((flags.part_bytes, flags.read_bytes), (0, 0));
    assert!(input.iter().all(|value| value.to_bits() == 0));

    let flags = whole.encode_text(line, &mut input);
    assert!(!flags.empty);
    assert_eq!(flags.read_bytes, line.len());
    assert!(input.iter().any(|value| *value != 0.0));

    // The empty text is empty in both modes; a routing part alone is not, read whole.
    for codec in [&whole, &payload] {
        let flags = codec.describe("");
        assert_eq!(
            flags,
            TextInput {
                part_bytes: 0,
                read_bytes: 0,
                nul_bytes: 0,
                truncated: false,
                empty: true,
            }
        );
    }
    assert!(payload.describe("GET /a?").empty);
    assert!(!payload.describe("GET /a?x").empty);
    assert!(!whole.describe("GET /a?").empty);
}

#[test]
fn a_nul_byte_reads_as_padding_at_its_position_and_is_counted() {
    let codec = codec(InputMode::Whole);
    let arch = InputArch::V1;
    let length = arch.input_bytes as usize;
    let mut with_nul = vec![f32::NAN; arch.row_values()];
    let mut without = vec![f32::NAN; arch.row_values()];

    let flags = codec.encode_text("ab\0cd", &mut with_nul);
    assert_eq!(
        (flags.read_bytes, flags.nul_bytes, flags.empty),
        (5, 1, false)
    );
    codec.encode_text("abXcd", &mut without);
    for channel in 0..arch.channels as usize {
        let row = &with_nul[channel * length..(channel + 1) * length];
        assert_eq!(row[2].to_bits(), 0, "zero input at the NUL byte");
        // Every other position is what it is without the NUL byte.
        for at in [0, 1, 3, 4, 5, 255] {
            assert_eq!(
                row[at].to_bits(),
                without[channel * length + at].to_bits(),
                "channel {channel}, position {at}"
            );
        }
    }
    let mut ids = [0xFFu8; 256];
    codec.byte_ids("ab\0cd", &mut ids);
    assert_eq!(ids[..6], [b'a', b'b', 0, b'c', b'd', 0]);

    // A trailing NUL byte is indistinguishable from padding: the same input as without.
    let mut trailing = vec![f32::NAN; arch.row_values()];
    let flags = codec.encode_text("abXcd\0", &mut trailing);
    assert_eq!((flags.read_bytes, flags.nul_bytes), (6, 1));
    assert_eq!(bits(&trailing), bits(&without));

    // Only NUL bytes: no byte reaches the model, so the input is flagged empty.
    let flags = codec.encode_text("\0\0\0", &mut with_nul);
    assert_eq!(
        (flags.part_bytes, flags.read_bytes, flags.nul_bytes),
        (3, 3, 3)
    );
    assert!(flags.empty);
    assert!(with_nul.iter().all(|value| value.to_bits() == 0));
    // NUL bytes beyond the input length are not read and not counted.
    let text = format!("{}\0\0", "a".repeat(256));
    let flags = codec.describe(&text);
    assert_eq!((flags.nul_bytes, flags.truncated), (0, true));
}

// ---------------------------------------------------------------------------------------
// The table file and the architecture description
// ---------------------------------------------------------------------------------------

#[test]
fn a_table_of_the_wrong_size_is_refused_naming_both_sizes() {
    let arch = InputArch::V1;
    assert_eq!(arch.table_file_bytes(), 65_536);
    let good = table_bytes();
    for wrong in [0usize, 4, 65_535, 65_532, 65_540, 131_072] {
        let mut bytes = good.clone();
        bytes.resize(wrong, 0);
        let reason = refusal(ByteTable::from_le_bytes(&bytes, &arch));
        assert!(
            reason.contains("65,536") || reason.contains("65536"),
            "{reason}"
        );
        assert!(reason.contains(&wrong.to_string()), "{reason}");
        assert!(reason.contains("256 rows of 64"), "{reason}");
    }
    // The size follows the architecture: the same bytes are wrong for 32 channels.
    let narrow = InputArch {
        channels: 32,
        ..arch
    };
    assert_eq!(narrow.table_file_bytes(), 32_768);
    let reason = refusal(ByteTable::from_le_bytes(&good, &narrow));
    assert!(reason.contains("256 rows of 32"), "{reason}");
    assert!(ByteTable::from_le_bytes(&good[..32_768], &narrow).is_ok());
}

#[test]
fn a_table_whose_row_0_is_not_all_zero_is_refused_naming_the_value() {
    let arch = InputArch::V1;
    let good = table_bytes();
    assert!(
        good[..256].iter().all(|byte| *byte == 0),
        "the published table"
    );
    let mut bytes = good.clone();
    bytes[5 * 4..6 * 4].copy_from_slice(&0.25f32.to_le_bytes());
    let reason = refusal(ByteTable::from_le_bytes(&bytes, &arch));
    assert!(reason.contains("row 0"), "{reason}");
    assert!(reason.contains("padding"), "{reason}");
    assert!(
        reason.contains("channel 5") && reason.contains("0.25"),
        "{reason}"
    );
    // Negative zero is not the all-zero input a padded row of a batch has.
    let mut bytes = good.clone();
    bytes[63 * 4..64 * 4].copy_from_slice(&(-0.0f32).to_le_bytes());
    let reason = refusal(ByteTable::from_le_bytes(&bytes, &arch));
    assert!(
        reason.contains("row 0") && reason.contains("channel 63"),
        "{reason}"
    );
    // A value that is not a finite number is refused wherever it is.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bytes = good.clone();
        let at = (200 * 64 + 7) * 4;
        bytes[at..at + 4].copy_from_slice(&bad.to_le_bytes());
        let reason = refusal(ByteTable::from_le_bytes(&bytes, &arch));
        assert!(
            reason.contains("row 200") && reason.contains("channel 7"),
            "{reason}"
        );
        assert!(reason.contains("finite"), "{reason}");
    }
}

#[test]
fn the_table_is_read_from_a_file_and_a_wrong_or_missing_file_is_named() {
    let arch = InputArch::V1;
    let path = fixture("science-2.1.table.bin");
    let table = ByteTable::read(&path, &arch).unwrap();
    assert_eq!(table, science_table());
    assert_eq!(table.channels(), 64);
    // Values are the file's, row by row.
    let file = table_bytes();
    for (byte, channel) in [(0u8, 0usize), (1, 0), (65, 13), (255, 63), (0xC3, 31)] {
        let at = (byte as usize * 64 + channel) * 4;
        let want = f32::from_le_bytes(file[at..at + 4].try_into().unwrap());
        assert_eq!(table.value(byte, channel).to_bits(), want.to_bits());
    }

    // The 989,260-byte model of the spike is not a table; it is refused by its size,
    // with the path.
    let other = fixture("spike-enc-b32-int8.onnx");
    let reason = refusal(ByteTable::read(&other, &arch));
    assert!(reason.contains("spike-enc-b32-int8.onnx"), "{reason}");
    assert!(
        reason.contains("989,260") || reason.contains("989260"),
        "{reason}"
    );

    let missing = fixture("no-such-table.bin");
    match ByteTable::read(&missing, &arch) {
        Err(L2dError::Missing { item, expected_at }) => {
            assert!(item.contains("byte table"), "{item}");
            assert_eq!(expected_at, missing);
        }
        other => panic!("expected a missing file, got {other:?}"),
    }
}

#[test]
fn the_architecture_description_is_validated_and_round_trips_as_json() {
    let arch = InputArch::V1;
    arch.validate().unwrap();
    let json = serde_json::to_string(&arch).unwrap();
    assert_eq!(json, r#"{"input_bytes":256,"channels":64,"batch":32}"#);
    assert_eq!(serde_json::from_str::<InputArch>(&json).unwrap(), arch);
    assert!(
        serde_json::from_str::<InputArch>(
            r#"{"input_bytes":256,"channels":64,"batch":32,"pad":1}"#
        )
        .is_err(),
        "an unknown field is not ignored"
    );

    for (bad, name) in [
        (
            InputArch {
                input_bytes: 0,
                ..arch
            },
            "input_bytes",
        ),
        (
            InputArch {
                channels: 0,
                ..arch
            },
            "channels",
        ),
        (InputArch { batch: 0, ..arch }, "batch"),
        (
            InputArch {
                input_bytes: u32::MAX,
                ..arch
            },
            "input_bytes",
        ),
        (
            InputArch {
                channels: u32::MAX,
                ..arch
            },
            "channels",
        ),
        (
            InputArch {
                batch: u32::MAX,
                ..arch
            },
            "batch",
        ),
    ] {
        let error = bad.validate().unwrap_err();
        assert!(
            matches!(&error, L2dError::InvalidInput { detail } if detail.contains(name)),
            "{bad:?}: {error}"
        );
        // Neither a table nor a codec is made for it.
        assert!(ByteTable::from_le_bytes(&table_bytes(), &bad).is_err());
    }
    // A table made for another width does not fit the codec's architecture.
    let narrow = InputArch {
        channels: 32,
        ..arch
    };
    let table = ByteTable::from_le_bytes(&table_bytes()[..32_768], &narrow).unwrap();
    let reason = refusal(ByteCodec::new(arch, table, InputMode::Whole));
    assert!(reason.contains("32") && reason.contains("64"), "{reason}");
}

#[test]
fn lengths_and_batch_size_come_from_the_architecture_description() {
    // A small architecture with a table that is easy to read: table[b][c] = b + c / 10.
    let arch = InputArch {
        input_bytes: 4,
        channels: 2,
        batch: 3,
    };
    let mut file = Vec::new();
    for byte in 0..256u32 {
        for channel in 0..2u32 {
            let value = if byte == 0 {
                0.0
            } else {
                byte as f32 + channel as f32 / 10.0
            };
            file.extend_from_slice(&value.to_le_bytes());
        }
    }
    assert_eq!(arch.table_file_bytes(), 2_048);
    let table = ByteTable::from_le_bytes(&file, &arch).unwrap();
    let codec = ByteCodec::new(arch, table, InputMode::Payload).unwrap();
    assert_eq!(arch.dims(), [3, 2, 1, 4]);

    let mut ids = [9u8; 4];
    let flags = codec.byte_ids("r?ABCDE", &mut ids);
    assert_eq!(ids, *b"ABCD");
    assert_eq!(
        (flags.part_bytes, flags.read_bytes, flags.truncated),
        (5, 4, true)
    );
    assert!(!codec.describe("r?ABCD").truncated);

    let mut batch = codec.new_batch();
    let mut texts = ["r?AB", "none", "r?ABCDE", "r\nC", "r?D"].into_iter();
    assert_eq!(codec.fill(&mut batch, &mut texts), 3);
    assert!(batch.is_full());
    assert_eq!(batch.capacity(), 3);
    #[rustfmt::skip]
    assert_eq!(
        batch.values(),
        [
            // "AB": channel 0, then channel 1
            65.0, 66.0, 0.0, 0.0,   65.1, 66.1, 0.0, 0.0,
            // "none" has no payload
            0.0, 0.0, 0.0, 0.0,     0.0, 0.0, 0.0, 0.0,
            // "ABCDE" cut after four bytes
            65.0, 66.0, 67.0, 68.0, 65.1, 66.1, 67.1, 68.1,
        ]
    );
    assert_eq!(
        batch
            .texts()
            .iter()
            .map(|t| (t.empty, t.truncated))
            .collect::<Vec<_>>(),
        [(false, false), (true, false), (false, true)]
    );
    // The last batch: two texts, the third row is an all-zero input.
    assert_eq!(codec.fill(&mut batch, &mut texts), 2);
    assert!(!batch.is_full());
    #[rustfmt::skip]
    assert_eq!(
        batch.values(),
        [
            67.0, 0.0, 0.0, 0.0,    67.1, 0.0, 0.0, 0.0,
            68.0, 0.0, 0.0, 0.0,    68.1, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,     0.0, 0.0, 0.0, 0.0,
        ]
    );
    assert!(batch.values()[16..].iter().all(|v| v.to_bits() == 0));
}

// ---------------------------------------------------------------------------------------
// Batches: padding, reuse, determinism, cost
// ---------------------------------------------------------------------------------------

/// Fills one batch from the texts and returns its rows.
fn fill(codec: &ByteCodec, batch: &mut InputBatch, texts: &[&str]) -> usize {
    codec.fill(batch, &mut texts.iter().copied())
}

#[test]
fn the_last_batch_is_padded_with_all_zero_inputs_and_tells_its_row_count() {
    let codec = codec(InputMode::Whole);
    let row = InputArch::V1.row_values();
    let mut batch = codec.new_batch();
    assert_eq!((batch.rows(), batch.capacity()), (0, 32));
    assert!(batch.is_empty() && !batch.is_full());
    assert!(batch.values().iter().all(|value| value.to_bits() == 0));

    // A full batch of long texts first, so every value of the buffer has been written.
    let long = "z".repeat(300);
    let full: Vec<&str> = (0..40).map(|_| long.as_str()).collect();
    assert_eq!(
        fill(&codec, &mut batch, &full),
        32,
        "a batch takes 32 texts"
    );
    assert!(batch.is_full());
    assert!(
        batch.values()[31 * row..].iter().any(|value| *value != 0.0),
        "the last row holds a text"
    );
    assert_eq!(codec.push(&mut batch, "one more"), None, "a full batch");
    assert_eq!(batch.rows(), 32);

    // Then three short texts in the same buffer.
    assert_eq!(fill(&codec, &mut batch, &["abc", "", "de"]), 3);
    assert_eq!(batch.rows(), 3);
    assert_eq!(batch.texts().len(), 3);
    assert!(!batch.is_full() && !batch.is_empty());
    assert_eq!(batch.values().len(), 32 * row, "always the whole batch");
    let mut single = vec![f32::NAN; row];
    for (at, text) in ["abc", "", "de"].into_iter().enumerate() {
        let flags = codec.encode_text(text, &mut single);
        assert_eq!(batch.texts()[at], flags);
        assert_eq!(
            bits(&batch.values()[at * row..(at + 1) * row]),
            bits(&single)
        );
    }
    assert!(
        batch.values()[3 * row..]
            .iter()
            .all(|value| value.to_bits() == 0),
        "rows 3 to 31 are all-zero inputs"
    );

    // Pushing one by one gives the same batch as filling.
    let mut pushed = codec.new_batch();
    for text in ["abc", "", "de"] {
        assert!(codec.push(&mut pushed, text).is_some());
    }
    assert_eq!(bits(pushed.values()), bits(batch.values()));
    assert_eq!(pushed.texts(), batch.texts());

    batch.clear();
    assert_eq!(batch.rows(), 0);
    assert!(batch.values().iter().all(|value| value.to_bits() == 0));
}

#[test]
fn encoding_is_deterministic_and_independent_of_what_the_buffers_held() {
    let expected = expected();
    let texts: Vec<&str> = expected.texts.iter().map(|t| t.text.as_str()).collect();
    for mode in InputMode::ALL {
        let first = codec(mode);
        let second = codec(mode);
        let mut a = first.new_batch();
        let mut b = second.new_batch();
        // `b` has held other texts before.
        fill(&second, &mut b, &texts[20..]);
        for chunk in texts.chunks(32) {
            assert_eq!(fill(&first, &mut a, chunk), chunk.len());
            assert_eq!(fill(&second, &mut b, chunk), chunk.len());
            assert_eq!(bits(a.values()), bits(b.values()));
            assert_eq!(a.texts(), b.texts());
            // A row does not depend on its place in the batch or on its neighbours.
            let row = InputArch::V1.row_values();
            let mut single = vec![f32::NAN; row];
            for (at, text) in chunk.iter().enumerate() {
                first.encode_text(text, &mut single);
                assert_eq!(bits(&a.values()[at * row..(at + 1) * row]), bits(&single));
            }
        }
    }
}

#[test]
fn a_batch_costs_two_mebibytes_and_the_cost_is_exposed() {
    let arch = InputArch::V1;
    assert_eq!(
        arch.batch_bytes(),
        2_097_152,
        "32 x 64 x 256 values of 4 bytes"
    );
    let codec = codec(InputMode::Payload);
    assert_eq!(codec.held_bytes(), 65_536, "the table");
    let batch = codec.new_batch();
    let flags = (32 * std::mem::size_of::<TextInput>()) as u64;
    assert_eq!(batch.held_bytes(), 2_097_152 + flags);
    assert!(flags <= 2_048, "the flags of a batch are {flags} bytes");
    assert_eq!(batch.arch(), &arch);
    // The cost does not grow with use.
    let mut batch = batch;
    let long = "q".repeat(100_000);
    for _ in 0..3 {
        fill(&codec, &mut batch, &[long.as_str(); 32]);
    }
    assert_eq!(batch.held_bytes(), 2_097_152 + flags);
}

#[test]
#[should_panic(expected = "256")]
fn a_row_buffer_of_the_wrong_length_is_a_programming_error() {
    let codec = codec(InputMode::Whole);
    let mut ids = [0u8; 255];
    codec.byte_ids("abc", &mut ids);
}

#[test]
#[should_panic(expected = "another architecture")]
fn a_batch_of_another_architecture_is_a_programming_error() {
    let small = InputArch {
        input_bytes: 4,
        channels: 64,
        batch: 3,
    };
    let other = ByteCodec::new(small, science_table(), InputMode::Whole).unwrap();
    let mut batch = other.new_batch();
    codec(InputMode::Whole).push(&mut batch, "abc");
}
