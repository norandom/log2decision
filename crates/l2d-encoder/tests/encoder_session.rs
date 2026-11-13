//! `EncoderSession` (task 2.6), everything that needs no NPU: the encoding around the
//! model through its seam ([`ImageModel`]), the placement rule, the model that is handed
//! to the provider and its cache key, the run record, and the refusals of `open` that
//! come before the runtime is loaded.
//!
//! No model is run here, on any provider. The model behind the seam is a formula of this
//! file that computes each row from that row's values alone, so that what the tests see
//! is the work around the model: which text is written where, what the rows without a
//! text hold, which output row becomes whose vector. The run on the device is in
//! `npu_session.rs`.

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::{Parts, Project, table_bytes, tiny_arch};
use l2d_core::{Device, L2dError, PlacementSource, RunKind};
use l2d_encoder::encode::{RowImage, l2_normalize, provider_cache_key};
use l2d_encoder::session::{ExecutedNodes, PROVIDER_OPTIONS, ROWS_PER_IMAGE, provider_model};
use l2d_encoder::{
    ByteCodec, ByteTable, ComputeAccelerator, EncodeStats, EncoderArch, EncoderError,
    EncoderPackage, EncoderSession, ImageEncoder, ImageModel, InputMode, NpuMatch, PlacementReport,
    RuntimeManifest, SessionInfo, TextInput,
};

const DIMENSION: usize = 4;
const ROWS: usize = 4;

fn weight(d: usize, c: usize, t: usize) -> f32 {
    ((d * 31 + c * 7 + t * 3) % 11) as f32 - 5.0
}

fn bias(d: usize) -> f32 {
    (d + 1) as f32 * 0.25
}

/// What the stand-in computes for one text's input `[channels][input_bytes]`.
fn formula(input: &[f32], channels: usize, width: usize) -> Vec<f32> {
    (0..DIMENSION)
        .map(|d| {
            let mut value = bias(d);
            for c in 0..channels {
                for t in 0..width {
                    value += input[c * width + t] * weight(d, c, t);
                }
            }
            value
        })
        .collect()
}

/// The images a stand-in was given, one after the other.
type Seen = Rc<RefCell<Vec<Vec<f32>>>>;

/// The stand-in for the model: row `n` of the output is the formula of row `n` of the
/// image. It keeps every image it was given.
#[derive(Default)]
struct StandIn {
    seen: Seen,
    /// Fail the run with this number (from 0).
    fail_at: Option<usize>,
}

impl ImageModel for StandIn {
    fn run(&mut self, image: &RowImage, output: &mut [f32]) -> Result<(), EncoderError> {
        let run = self.seen.borrow().len();
        self.seen.borrow_mut().push(image.values().to_vec());
        if self.fail_at == Some(run) {
            return Err(EncoderError::Failed {
                detail: format!("run {run} failed"),
            });
        }
        let [one, channels, rows, width] = image.dims().map(|dim| dim as usize);
        assert_eq!(one, 1);
        assert_eq!(output.len(), DIMENSION * rows);
        let values = image.values();
        let mut input = vec![0f32; channels * width];
        for n in 0..rows {
            for c in 0..channels {
                let at = c * rows * width + n * width;
                input[c * width..(c + 1) * width].copy_from_slice(&values[at..at + width]);
            }
            for (d, value) in formula(&input, channels, width).into_iter().enumerate() {
                output[d * rows + n] = value;
            }
        }
        Ok(())
    }
}

fn tiny_codec(mode: InputMode) -> ByteCodec {
    let arch = tiny_arch().input;
    let table = ByteTable::from_le_bytes(&table_bytes(arch.channels), &arch).unwrap();
    ByteCodec::new(arch, table, mode).unwrap()
}

fn encoder(mode: InputMode) -> (ImageEncoder<StandIn>, Seen) {
    let model = StandIn::default();
    let seen = model.seen.clone();
    (
        ImageEncoder::new(tiny_codec(mode), DIMENSION, ROWS, model),
        seen,
    )
}

/// The vector of a text by the codec's own row layout and the formula: no image.
fn alone(codec: &ByteCodec, text: &str) -> (Vec<f32>, TextInput) {
    let arch = codec.arch();
    let mut input = vec![0f32; arch.row_values()];
    let flags = codec.encode_text(text, &mut input);
    let mut vector = formula(&input, arch.channels as usize, arch.input_bytes as usize);
    l2_normalize(&mut vector);
    (vector, flags)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Eleven texts for images of four rows: two full images and one of three. Lengths
/// around the input length of 8 bytes, a NUL byte, an empty text, a text without payload.
const TEXTS: [&str; 11] = [
    "GET /a?x=1",
    "POST /b\nuser=admin",
    "",
    "GET /c?0123456789abcdef",
    "no payload at all here",
    "GET /d?12345678",
    "a?b\0c",
    "GET /e?",
    "x?y",
    "GET /a?x=1&y=2",
    "?zzzzzzzzz",
];

#[test]
fn one_unit_vector_per_text_in_input_order_with_a_last_image_that_is_not_full() {
    for mode in InputMode::ALL {
        let (mut encoder, seen) = encoder(mode);
        let codec = tiny_codec(mode);
        let encoded = encoder.encode(&TEXTS).unwrap();
        assert_eq!(encoded.len(), TEXTS.len());
        assert_eq!(seen.borrow().len(), 3, "11 texts are 3 images of 4 rows");
        for (text, entry) in TEXTS.iter().zip(&encoded) {
            let (vector, flags) = alone(&codec, text);
            assert_eq!(entry.vector.len(), DIMENSION);
            assert_eq!(bits(&entry.vector), bits(&vector), "{mode:?} {text:?}");
            assert_eq!(entry.input, flags, "{mode:?} {text:?}");
            let length: f64 = entry
                .vector
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((length - 1.0).abs() < 1e-6, "{text:?} has length {length}");
        }
        let stats = encoder.stats();
        assert_eq!((stats.texts, stats.images), (11, 3));
        assert!(stats.seconds > 0.0 && stats.logs_per_second().unwrap() > 0.0);
        assert!(stats.write_seconds + stats.run_seconds + stats.read_seconds <= stats.seconds);
    }
}

#[test]
fn both_truncation_flags_come_back_with_each_vector() {
    let (mut encoder, _) = encoder(InputMode::Payload);
    let encoded = encoder.encode(&TEXTS).unwrap();
    let flags: Vec<(bool, bool, bool)> = encoded
        .iter()
        .map(|entry| {
            (
                entry.input.truncated,
                entry.input.text_longer_than_input,
                entry.input.empty,
            )
        })
        .collect();
    assert_eq!(
        flags,
        [
            (false, true, false),  // "GET /a?x=1": payload `x=1`, 10 bytes in all
            (true, true, false),   // payload `user=admin`
            (false, false, true),  // the empty text
            (true, true, false),   // a payload of 16 bytes
            (false, true, true),   // no payload: nothing read, yet longer than the input
            (false, true, false),  // a payload of exactly 8 bytes
            (false, false, false), // `b\0c`
            (false, false, true),  // an empty payload
            (false, false, false), // `y`
            (false, true, false),  // payload `x=1&y=2`, 7 bytes
            (true, true, false),   // a payload of 9 bytes
        ]
    );
}

#[test]
fn a_text_gets_the_same_vector_whatever_its_row_and_its_neighbours() {
    let (mut encoder, _) = encoder(InputMode::Whole);
    let probe = "GET /a?x=1";
    let expected = bits(&alone(&tiny_codec(InputMode::Whole), probe).0);
    // Alone, in every row of a full image among changing neighbours, in an image that
    // is not full, and repeated.
    assert_eq!(bits(&encoder.encode(&[probe]).unwrap()[0].vector), expected);
    for row in 0..ROWS {
        for others in [["zzzzzzzzzzzz"; 3], ["", "", ""], ["a", "bb", "ccc"]] {
            let mut texts = others.to_vec();
            texts.insert(row, probe);
            let encoded = encoder.encode(&texts).unwrap();
            assert_eq!(
                bits(&encoded[row].vector),
                expected,
                "row {row}, {others:?}"
            );
        }
    }
    for count in 1..=ROWS * 2 + 1 {
        let encoded = encoder.encode(&vec![probe; count]).unwrap();
        assert!(
            encoded.iter().all(|entry| bits(&entry.vector) == expected),
            "{count} times"
        );
    }
}

#[test]
fn the_rows_after_the_last_text_are_empty_inputs() {
    let (mut encoder, seen) = encoder(InputMode::Whole);
    let arch = tiny_arch().input;
    let (channels, width) = (arch.channels as usize, arch.input_bytes as usize);
    encoder.encode(&["aaaaaaaaaaaa"; ROWS]).unwrap();
    encoder.encode(&["b"]).unwrap();
    encoder.encode(&["cc", "dd", "ee"]).unwrap();
    encoder.encode(&["f", "g"]).unwrap();
    let seen = seen.borrow();
    let used = [4usize, 1, 3, 2];
    assert_eq!(seen.len(), used.len());
    for (image, used) in seen.iter().zip(used) {
        for c in 0..channels {
            for row in 0..ROWS {
                let line = &image[c * ROWS * width + row * width..][..width];
                assert_eq!(
                    line.iter().all(|value| value.to_bits() == 0),
                    row >= used,
                    "channel {c}, row {row} of an image of {used} texts"
                );
            }
        }
    }
}

#[test]
fn vectors_and_flags_are_appended_to_the_callers_buffers() {
    let (mut encoder, _) = encoder(InputMode::Whole);
    let codec = tiny_codec(InputMode::Whole);
    let mut vectors = vec![7.0f32; 3];
    let mut flags = vec![TextInput::default()];
    encoder
        .encode_into(&TEXTS[..6], &mut vectors, &mut flags)
        .unwrap();
    encoder
        .encode_into(&TEXTS[6..], &mut vectors, &mut flags)
        .unwrap();
    assert_eq!(vectors.len(), 3 + TEXTS.len() * DIMENSION);
    assert_eq!(flags.len(), 1 + TEXTS.len());
    assert_eq!(vectors[..3], [7.0; 3]);
    for (index, text) in TEXTS.iter().enumerate() {
        let (vector, input) = alone(&codec, text);
        let at = 3 + index * DIMENSION;
        assert_eq!(
            bits(&vectors[at..at + DIMENSION]),
            bits(&vector),
            "{text:?}"
        );
        assert_eq!(flags[1 + index], input);
    }
    // The same vectors as `encode`.
    let encoded = encoder.encode(&TEXTS).unwrap();
    let flat: Vec<f32> = encoded
        .iter()
        .flat_map(|entry| entry.vector.clone())
        .collect();
    assert_eq!(bits(&flat), bits(&vectors[3..]));
}

#[test]
fn no_text_is_no_run() {
    let (mut encoder, seen) = encoder(InputMode::Whole);
    assert_eq!(encoder.encode(&[]).unwrap(), Vec::new());
    assert!(seen.borrow().is_empty());
    assert_eq!(encoder.stats().logs_per_second(), None);
    assert_eq!((encoder.stats().texts, encoder.stats().images), (0, 0));
}

#[test]
fn a_failed_run_is_an_error_and_leaves_the_callers_buffers_as_they_were() {
    let model = StandIn {
        fail_at: Some(1),
        ..StandIn::default()
    };
    let mut encoder = ImageEncoder::new(tiny_codec(InputMode::Whole), DIMENSION, ROWS, model);
    let error = encoder.encode(&TEXTS).unwrap_err();
    assert_eq!(
        error,
        EncoderError::Failed {
            detail: "run 1 failed".into()
        }
    );
    // Only the texts of the image that ran are counted.
    assert_eq!((encoder.stats().texts, encoder.stats().images), (4, 1));

    let model = StandIn {
        fail_at: Some(2),
        ..StandIn::default()
    };
    let mut encoder = ImageEncoder::new(tiny_codec(InputMode::Whole), DIMENSION, ROWS, model);
    let mut vectors = vec![1.0f32, 2.0];
    let mut flags = Vec::new();
    assert!(
        encoder
            .encode_into(&TEXTS, &mut vectors, &mut flags)
            .is_err()
    );
    assert_eq!(vectors, [1.0, 2.0]);
    assert!(flags.is_empty());
}

#[test]
fn the_buffers_of_a_session_are_fixed_and_stated() {
    // Architecture v1 at the session's 64 rows, dimension 128.
    let parts = Parts::spike();
    let arch = EncoderArch::V1;
    let table = ByteTable::from_le_bytes(&parts.table, &arch.input).unwrap();
    let codec = ByteCodec::new(arch.input, table, InputMode::Payload).unwrap();
    let rows = ROWS_PER_IMAGE as usize;
    assert_eq!(rows, 64);
    let encoder = ImageEncoder::new(codec, 128, rows, StandIn::default());
    let image = 64 * 64 * 256 * 4; // the image: channels x rows x input bytes floats
    let tables = 2 * 256 * 64 * 4; // the codec's table and its copy in channel order
    let texts = 64 * 256; // the bytes of one image's texts
    let per_row = 64 * (std::mem::size_of::<usize>() + std::mem::size_of::<TextInput>());
    let output = 128 * 64 * 4; // the model's output of one image
    let vectors = 64 * 128 * 4; // one image's vectors
    assert_eq!(
        encoder.held_bytes(),
        (image + tables + texts + per_row + output + vectors) as u64
    );
    assert_eq!(encoder.held_bytes(), 4_409_856);
    assert_eq!((encoder.rows(), encoder.dimension()), (64, 128));
}

// --- The placement rule -----------------------------------------------------------------

fn node(op: &str, inputs: &[&str], outputs: &[&str], device: &str) -> serde_json::Value {
    serde_json::json!({
        "output": outputs,
        "input": inputs,
        "opType": op,
        "comment": "@1 ...",
        "device": device,
    })
}

/// A report in the provider's shape for a small graph: quantize, three nodes, dequantize.
fn report(devices: [&str; 5]) -> String {
    serde_json::json!({
        "deviceStat": [{"name": "all", "nodeNum": 5}],
        "nodeStat": [
            node("QuantizeLinear", &["x", "x_scale", "x_zero_point"], &["x_q"], devices[0]),
            node("DequantizeLinear", &["x_q", "x_scale", "x_zero_point"], &["x_dq"], devices[1]),
            node("Conv", &["x_dq", "w", "b"], &["h"], devices[2]),
            node("QuantizeLinear", &["h", "h_scale", "h_zero_point"], &["h_q"], devices[3]),
            node("DequantizeLinear", &["h_q", "h_scale", "h_zero_point"], &["e"], devices[4]),
        ],
        "subgraphStat": [{"device": "NPU"}],
    })
    .to_string()
}

#[test]
fn the_placement_is_read_from_the_providers_report() {
    let placement = PlacementReport::parse(&report(["CPU", "NPU", "NPU", "NPU", "CPU"])).unwrap();
    assert_eq!(
        (
            placement.npu_nodes,
            placement.cpu_nodes,
            placement.all_nodes
        ),
        (3, 2, 5)
    );
    assert_eq!(
        placement.cpu_operators(),
        ["QuantizeLinear", "DequantizeLinear"]
    );
    assert_eq!(placement.cpu[0].inputs[0], "x");
    assert_eq!(placement.cpu[1].outputs, ["e"]);
    assert_eq!(placement.subgraphs, Some(1));
    let counts = placement.placement();
    assert_eq!((counts.npu_nodes, counts.cpu_nodes), (3, 2));
    assert_eq!(counts.source, PlacementSource::ProviderReport);

    for (bad, expected) in [
        ("not json", "not JSON"),
        ("{}", "nodeStat"),
        (
            r#"{"nodeStat": [{"opType": "Conv", "device": "GPU"}]}"#,
            "node 0",
        ),
        (r#"{"nodeStat": [{"opType": "Conv"}]}"#, "node 0"),
    ] {
        let error = PlacementReport::parse(bad).unwrap_err();
        assert!(error.contains(expected), "{bad}: {error}");
    }
}

#[test]
fn only_the_quantize_nodes_at_the_two_ends_may_be_on_the_cpu() {
    let check = |devices: [&str; 5]| {
        PlacementReport::parse(&report(devices))
            .unwrap()
            .npu_only(5, "x", "e")
    };
    // What the provider does with this architecture: the two ends on the CPU.
    assert_eq!(check(["CPU", "NPU", "NPU", "NPU", "CPU"]), Ok(()));
    // Fewer nodes on the CPU are still NPU only.
    assert_eq!(check(["NPU", "NPU", "NPU", "NPU", "CPU"]), Ok(()));
    assert_eq!(check(["NPU"; 5]), Ok(()));

    // The provider's compiler failed and the session runs on the CPU (task 2.12).
    let error = check(["CPU"; 5]).unwrap_err();
    assert!(
        error.contains("no node of the model is on the NPU"),
        "{error}"
    );
    // A convolution on the CPU.
    let error = check(["CPU", "NPU", "CPU", "NPU", "CPU"]).unwrap_err();
    assert!(
        error.contains("3 of 5 nodes are on the CPU") && error.contains("Conv (output `h`)"),
        "{error}"
    );
    // Quantize nodes on the CPU that are not at the ends: the same operators as the
    // ends, but inside the graph.
    let error = check(["CPU", "CPU", "NPU", "NPU", "CPU"]).unwrap_err();
    assert!(
        error.contains("DequantizeLinear (output `x_dq`)"),
        "{error}"
    );
    let error = check(["CPU", "NPU", "NPU", "CPU", "CPU"]).unwrap_err();
    assert!(error.contains("QuantizeLinear (output `h_q`)"), "{error}");

    // A report of another model: other names at the ends, or another number of nodes.
    let placement = PlacementReport::parse(&report(["CPU", "NPU", "NPU", "NPU", "CPU"])).unwrap();
    assert!(placement.npu_only(5, "input", "e").is_err());
    assert!(placement.npu_only(5, "x", "output").is_err());
    let error = placement.npu_only(35, "x", "e").unwrap_err();
    assert!(
        error.contains("lists 5 nodes") && error.contains("has 35"),
        "{error}"
    );
}

/// A profile in ONNX Runtime's shape: one event per executed node, among other events.
fn profile(nodes: &[(&str, &str)]) -> String {
    let mut events = vec![
        serde_json::json!({"cat": "Session", "name": "model_loading_array", "args": {}}),
        serde_json::json!({"cat": "Session", "name": "session_initialization", "args": {}}),
    ];
    for (index, (operator, provider)) in nodes.iter().enumerate() {
        // The time before a node's kernel runs is an event of its own and not a node.
        events.push(serde_json::json!({
            "cat": "Node", "name": format!("node_{index}_fence_before"),
            "args": {"op_name": operator},
        }));
        events.push(serde_json::json!({
            "cat": "Node", "name": format!("node_{index}_kernel_time"), "dur": 12,
            "args": {"op_name": operator, "provider": provider, "thread_scheduling_stats": ""},
        }));
    }
    events.push(serde_json::json!({"cat": "Session", "name": "SequentialExecutor::Execute"}));
    serde_json::Value::Array(events).to_string()
}

const CPU: &str = "CPUExecutionProvider";
const VITIS: &str = "VitisAIExecutionProvider";

#[test]
fn the_first_run_must_have_been_executed_by_the_npu_provider() {
    let read = |nodes: &[(&str, &str)]| ExecutedNodes::parse(&profile(nodes)).unwrap();
    // What a session on the NPU shows on the reference host (`final2-rows.log`): the
    // compiled part as one node of the provider, and the two ends through it as well.
    let on_npu = read(&[
        ("QuantizeLinear", VITIS),
        ("vitis_ai_ep_1", VITIS),
        ("DequantizeLinear", VITIS),
    ]);
    assert_eq!(
        on_npu.on_provider,
        ["QuantizeLinear", "vitis_ai_ep_1", "DequantizeLinear"]
    );
    assert!(on_npu.elsewhere.is_empty());
    assert_eq!(on_npu.npu_only(), Ok(()));
    // The two ends through another provider are a run on the NPU too.
    let ends_on_cpu = read(&[
        ("QuantizeLinear", CPU),
        ("vitis_ai_ep_0", VITIS),
        ("DequantizeLinear", CPU),
    ]);
    assert_eq!(ends_on_cpu.on_provider, ["vitis_ai_ep_0"]);
    assert_eq!(
        ends_on_cpu.elsewhere,
        [
            ("QuantizeLinear".to_string(), CPU.to_string()),
            ("DequantizeLinear".to_string(), CPU.to_string())
        ]
    );
    assert_eq!(ends_on_cpu.npu_only(), Ok(()));
    assert_eq!(read(&[("vitis_ai_ep_0", VITIS)]).npu_only(), Ok(()));
    assert_eq!(
        read(&[("vitis_ai_ep_0", VITIS), ("DequantizeLinear", CPU)]).npu_only(),
        Ok(())
    );

    // The session the reference host opened from a cache folder with a damaged
    // `context.json` and an intact report (`damage-1.log`, `final2-damage.log`): the
    // two ends through the provider, the model itself on the CPU provider.
    let fell_back = read(&[
        ("QuantizeLinear", VITIS),
        ("Transpose", CPU),
        ("QLinearConv", CPU),
        ("QLinearConv", CPU),
        ("QLinearConv", CPU),
        ("QLinearConv", CPU),
        ("QLinearConv", CPU),
        ("QLinearAveragePool", CPU),
        ("QLinearConv", CPU),
        ("Transpose", CPU),
        ("DequantizeLinear", VITIS),
    ]);
    let error = fell_back.npu_only().unwrap_err();
    assert!(
        error.contains("2 node(s) through VitisAIExecutionProvider")
            && error.contains("9 through other providers")
            && error.contains("QLinearConv (CPUExecutionProvider)"),
        "{error}"
    );
    // The provider ran nothing but the two ends, and nothing ran elsewhere: no
    // compiled part was executed.
    let only_ends = read(&[("QuantizeLinear", VITIS), ("DequantizeLinear", VITIS)]);
    assert!(only_ends.npu_only().is_err());
    // Every node on the CPU provider.
    let on_cpu = read(&[
        ("QuantizeLinear", CPU),
        ("Conv", CPU),
        ("AveragePool", CPU),
        ("DequantizeLinear", CPU),
    ]);
    let error = on_cpu.npu_only().unwrap_err();
    assert!(
        error.contains("0 node(s) through VitisAIExecutionProvider")
            && error.contains("4 through other providers")
            && error.contains("Conv (CPUExecutionProvider)"),
        "{error}"
    );
    // The provider ran a part, another provider ran more than the two ends.
    for nodes in [
        &[
            ("QuantizeLinear", CPU),
            ("vitis_ai_ep_0", VITIS),
            ("Conv", CPU),
            ("DequantizeLinear", CPU),
        ][..],
        &[
            ("QuantizeLinear", CPU),
            ("vitis_ai_ep_0", VITIS),
            ("QuantizeLinear", CPU),
        ],
        &[
            ("DequantizeLinear", CPU),
            ("vitis_ai_ep_0", VITIS),
            ("DequantizeLinear", CPU),
        ],
    ] {
        assert!(read(nodes).npu_only().is_err(), "{nodes:?}");
    }
    // Nothing was run at all.
    assert!(read(&[]).npu_only().is_err());

    for (bad, expected) in [
        ("not json", "not JSON"),
        ("{}", "list of events"),
        (
            r#"[{"cat":"Node","name":"a_kernel_time","args":{}}]"#,
            "names no provider",
        ),
    ] {
        let error = ExecutedNodes::parse(bad).unwrap_err();
        assert!(error.contains(expected), "{bad}: {error}");
    }
}

// --- The model handed to the provider ---------------------------------------------------

#[test]
fn the_model_is_rewritten_for_64_rows_and_its_compilation_is_named() {
    let project = Project::new("session-model");
    let paths = project.paths();
    let written = EncoderPackage::write(&paths, &Parts::spike().parts()).unwrap();
    let package = written.package;
    let before = common::snapshot(&project.root);

    let model = provider_model(&package, "1.9.0").unwrap();
    assert_eq!(model.rows, 64);
    assert_eq!((model.input.as_str(), model.output.as_str()), ("x", "e"));
    // 38 nodes in the file; `Flatten` and its quantize pair are gone.
    assert_eq!(model.nodes, 35);
    let graph = l2d_encoder::onnx_edit::graph_info(&model.bytes).unwrap();
    assert_eq!(graph.inputs[0].shape, [1, 64, 64, 256]);
    assert_eq!(graph.outputs[0].shape, [1, 128, 64, 1]);
    assert_eq!(
        model.bytes,
        l2d_encoder::onnx_edit::with_rows_as_height(&package.read_model().unwrap(), 64).unwrap()
    );

    // The key: revision, release, the bytes handed over, the options.
    let revision = &package.identity().revision;
    assert_eq!(
        model.cache_key,
        provider_cache_key(revision, "1.9.0", &model.bytes, PROVIDER_OPTIONS)
    );
    assert!(
        model
            .cache_key
            .starts_with(&format!("{}-1.9.0-", &revision[..12]))
    );
    assert_eq!(PROVIDER_OPTIONS, [("enable_cache_file_io_in_mem", "0")]);
    let other = provider_model(&package, "1.10.0").unwrap();
    assert_ne!(other.cache_key, model.cache_key);
    assert_eq!(other.bytes, model.bytes);
    // Not the key of the file in the package, and not of another row count.
    let file = package.read_model().unwrap();
    assert_ne!(
        model.cache_key,
        provider_cache_key(revision, "1.9.0", &file, PROVIDER_OPTIONS)
    );
    let rows_32 = l2d_encoder::onnx_edit::with_rows_as_height(&file, 32).unwrap();
    assert_ne!(
        model.cache_key,
        provider_cache_key(revision, "1.9.0", &rows_32, PROVIDER_OPTIONS)
    );

    // Nothing was written: the package is as it was and there is no cache yet.
    assert_eq!(common::snapshot(&project.root), before);
}

#[test]
fn a_model_that_cannot_be_rewritten_is_refused_with_the_reason() {
    let project = Project::new("session-tiny");
    let paths = project.paths();
    // The tiny model of the package tests: a quantize pair around one node, nothing
    // that pools over the length of a text.
    let package = EncoderPackage::write(&paths, &Parts::tiny().parts())
        .unwrap()
        .package;
    let error = provider_model(&package, "1.9.0").unwrap_err();
    let EncoderError::ModelRefused { reason } = &error else {
        panic!("{error:?}");
    };
    assert!(
        reason.contains("cannot be rewritten") && reason.contains("tiny"),
        "{reason}"
    );
    assert!(matches!(L2dError::from(error), L2dError::Refused { .. }));

    // The model file changed after the package was opened.
    let package = EncoderPackage::write(&paths, &Parts::spike().parts())
        .unwrap()
        .package;
    let path = package.model_path();
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    let error = provider_model(&package, "1.9.0").unwrap_err();
    assert!(
        matches!(&error, EncoderError::ModelRefused { reason } if reason.contains("changed after")),
        "{error:?}"
    );
}

// --- Opening without the runtime --------------------------------------------------------

#[test]
fn opening_fails_with_the_reason_when_the_runtime_files_are_missing() {
    let project = Project::new("session-missing");
    let paths = project.paths();
    let package = EncoderPackage::write(&paths, &Parts::spike().parts())
        .unwrap()
        .package;

    // No runtime directory.
    let dir = paths.ryzenai_dir();
    let error = EncoderSession::open_at(&package, &paths, &dir).unwrap_err();
    let EncoderError::RuntimeMissing { file, expected_at } = &error else {
        panic!("{error:?}");
    };
    assert!(file.contains("runtime"), "{file}");
    assert_eq!(expected_at, &dir);
    let shown = error.to_string();
    assert!(
        shown.contains("cannot run") && shown.contains(&dir.display().to_string()),
        "{shown}"
    );
    assert!(matches!(L2dError::from(error), L2dError::Missing { .. }));

    // A directory without the files: the first missing file is named with its place.
    std::fs::create_dir_all(&dir).unwrap();
    let error = EncoderSession::open_at(&package, &paths, &dir).unwrap_err();
    let first = &RuntimeManifest::expected().files[0].name;
    assert!(
        matches!(&error, EncoderError::RuntimeMissing { file, expected_at }
            if file.contains(first.as_str()) && expected_at.starts_with(&dir)),
        "{error:?}"
    );

    // Files of the right names that are not the release's.
    for file in &RuntimeManifest::expected().files {
        std::fs::write(dir.join(&file.name), b"x").unwrap();
    }
    let error = EncoderSession::open_at(&package, &paths, &dir).unwrap_err();
    assert!(
        matches!(&error, EncoderError::RuntimeRefused { reason } if reason.contains(first.as_str())),
        "{error:?}"
    );

    // Nothing was loaded, compiled or cached on the way.
    assert!(!paths.runtime_cache_dir().exists());
}

// --- The run record ---------------------------------------------------------------------

fn info() -> SessionInfo {
    let placement = PlacementReport::parse(&report(["CPU", "NPU", "NPU", "NPU", "CPU"])).unwrap();
    SessionInfo {
        identity: l2d_core::EncoderIdentity {
            name: "csic".into(),
            revision: "099570ec739ce4d0548c2d15629e857cf516664f".into(),
            dimension: 128,
        },
        input_mode: InputMode::Payload,
        rows: 64,
        runtime_release: "1.9.0".into(),
        onnxruntime: "1.30.0".into(),
        runtime_source: "nuget".into(),
        runtime_dir: "runtime/ryzenai".into(),
        npu: ComputeAccelerator {
            instance_id: r"PCI\VEN_1022&DEV_17F0".into(),
            description: "NPU Compute Accelerator Device".into(),
            driver_version: "32.0.203.329".into(),
        },
        npu_match: NpuMatch::KnownDevice,
        driver_version: Some("32.0.203.329".into()),
        placement,
        executed: ExecutedNodes {
            on_provider: vec!["vitis_ai_ep_0".into()],
            elsewhere: vec![
                ("QuantizeLinear".into(), "CPUExecutionProvider".into()),
                ("DequantizeLinear".into(), "CPUExecutionProvider".into()),
            ],
        },
        cache_key: "099570ec739c-1.9.0-0123456789abcdef".into(),
        cache_folder: "runtime/cache/099570ec739c-1.9.0-0123456789abcdef".into(),
        compiled: false,
        open_seconds: 1.0,
        load_seconds: 0.25,
        session_seconds: 0.5,
        started_utc: "2026-11-13T10:11:11Z".into(),
        held_bytes: 4_409_856,
    }
}

#[test]
fn the_run_record_names_encoder_device_driver_runtime_rate_and_placement() {
    let stats = EncodeStats {
        texts: 25_608,
        images: 401,
        seconds: 1.6,
        write_seconds: 0.2,
        run_seconds: 1.3,
        read_seconds: 0.05,
    };
    let info = info();
    let record = info.run_record(&stats, 2_750);
    assert_eq!(record.kind, RunKind::Encode);
    assert_eq!(record.started_utc, "2026-11-13T10:11:11Z");
    assert_eq!(record.duration_ms, 2_750);
    assert_eq!(record.encoder.as_ref(), Some(&info.identity));
    assert_eq!(record.device, Some(Device::Npu));
    assert_eq!(record.driver_version.as_deref(), Some("32.0.203.329"));
    assert_eq!(record.logs_per_second, Some(25_608.0 / 1.6));
    let placement = record.placement.unwrap();
    assert_eq!((placement.npu_nodes, placement.cpu_nodes), (3, 2));
    assert_eq!(placement.source, PlacementSource::ProviderReport);
    assert!(record.inputs.is_empty() && record.seed.is_none());

    let settings = &record.settings;
    assert_eq!(settings["npu"]["matched_by"], "known_device_id");
    assert_eq!(settings["npu"]["instance_id"], r"PCI\VEN_1022&DEV_17F0");
    assert_eq!(settings["runtime"]["release"], "1.9.0");
    assert_eq!(settings["runtime"]["onnxruntime"], "1.30.0");
    assert_eq!(settings["runtime"]["source"], "nuget");
    assert_eq!(settings["input_mode"], "payload");
    assert_eq!(settings["session"]["rows_per_image"], 64);
    assert_eq!(settings["session"]["compiled_now"], false);
    assert_eq!(settings["session"]["held_bytes"], 4_409_856);
    assert_eq!(settings["placement"]["all_nodes"], 5);
    assert_eq!(
        settings["placement"]["cpu_operators"],
        serde_json::json!(["QuantizeLinear", "DequantizeLinear"])
    );
    assert_eq!(
        settings["placement"]["first_run_nodes_on_provider"],
        serde_json::json!(["vitis_ai_ep_0"])
    );
    assert_eq!(
        settings["placement"]["first_run_nodes_elsewhere"],
        serde_json::json!([
            "QuantizeLinear (CPUExecutionProvider)",
            "DequantizeLinear (CPUExecutionProvider)"
        ])
    );
    assert_eq!(settings["encoding"]["texts"], 25_608);
    assert_eq!(settings["encoding"]["images"], 401);

    // The record survives its JSON form, and the settings stay an object, so that the
    // caller can add the batch plan (requirement 12.3).
    let json = record.to_json().unwrap();
    assert_eq!(l2d_core::RunRecord::from_json(&json).unwrap(), record);
    assert!(record.settings.is_object());

    // A device matched by vendor only says so; no text encoded gives no rate.
    let vendor_only = SessionInfo {
        npu_match: NpuMatch::AmdVendorOnly,
        driver_version: None,
        ..info
    };
    let record = vendor_only.run_record(&EncodeStats::default(), 10);
    assert_eq!(record.settings["npu"]["matched_by"], "amd_vendor_only");
    assert_eq!(record.logs_per_second, None);
    assert_eq!(record.driver_version, None);
}
