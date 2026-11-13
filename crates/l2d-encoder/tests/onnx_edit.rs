//! The mechanical model edits of task 2.12 (`l2d_encoder::onnx_edit`), without hardware:
//! every edited file is read again, by the package's own model reader (`onnx.rs`) where
//! that reader accepts the result and by `graph_info` everywhere. Whether the NPU runs
//! the edited files is shown by the hardware tests (`npu_rows.rs`, `npu_throughput.rs`).

mod common;

use l2d_core::L2dError;
use l2d_encoder::ModelInterface;
use l2d_encoder::onnx_edit::{
    ELEMENT_FLOAT32, ELEMENT_UINT8, GraphInfo, QuantizedEnds, TensorInfo, graph_info,
    node_ints_attribute, with_batch, with_output_at, with_quantized_ends, with_rows_as_height,
};

/// The spike's 8-bit model, node for node the trained model's graph (2.1 notes).
fn spike() -> Vec<u8> {
    std::fs::read(common::fixture("spike-enc-b32-int8.onnx")).unwrap()
}

fn reason(error: L2dError) -> String {
    match error {
        L2dError::Refused { reason } => reason,
        other => panic!("expected a refusal, got {other}"),
    }
}

/// The bytes of every stored tensor of a model, by name: an edit must not touch them.
fn names_of_initializers(info: &GraphInfo) -> Vec<&str> {
    info.initializers
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect()
}

#[test]
#[ignore = "prints the graph of the fixture; run on request"]
fn print_the_graph_of_the_fixture() {
    let info = graph_info(&spike()).unwrap();
    println!("inputs {:?}", info.inputs);
    println!("outputs {:?}", info.outputs);
    for node in &info.nodes {
        println!(
            "node {} `{}` {:?} -> {:?}",
            node.op, node.name, node.inputs, node.outputs
        );
    }
    for tensor in &info.value_info {
        println!("value_info {tensor:?}");
    }
    for tensor in &info.initializers {
        println!("initializer {tensor:?}");
    }
}

#[test]
fn the_graph_of_the_fixture_is_read() {
    let info = graph_info(&spike()).unwrap();
    assert_eq!(
        info.inputs,
        [TensorInfo {
            name: "x".into(),
            element_type: ELEMENT_FLOAT32,
            shape: vec![32, 64, 1, 256]
        }]
    );
    assert_eq!(
        info.outputs,
        [TensorInfo {
            name: "e".into(),
            element_type: ELEMENT_FLOAT32,
            shape: vec![32, 128]
        }]
    );
    assert_eq!(info.nodes.len(), 38);
    let counts = info.operator_counts();
    let count = |op: &str| counts.iter().find(|(name, _)| name == op).map(|(_, n)| *n);
    assert_eq!(count("Conv"), Some(5));
    assert_eq!(count("GlobalAveragePool"), Some(1));
    assert_eq!(count("Flatten"), Some(1));
    assert_eq!(count("Gemm"), Some(1));
    assert_eq!(count("QuantizeLinear"), Some(9));
    assert_eq!(count("DequantizeLinear"), Some(21));
    assert_eq!(info.producer("e").unwrap().op, "DequantizeLinear");
    assert_eq!(info.readers("x").len(), 1);
    assert_eq!(info.initializers.len(), 52);
}

#[test]
fn another_batch_size_changes_the_first_dimensions_and_nothing_else() {
    let model = spike();
    let before = graph_info(&model).unwrap();
    for batch in [1u32, 8, 64, 128, 256] {
        let edited = with_batch(&model, batch).unwrap();
        // The package's reader accepts it and reads the new shapes; the quantization of
        // the input and the output is what it was.
        let interface = ModelInterface::from_onnx(&edited).unwrap();
        let original = ModelInterface::from_onnx(&model).unwrap();
        assert_eq!(interface.input.shape, [i64::from(batch), 64, 1, 256]);
        assert_eq!(interface.output.shape, [i64::from(batch), 128]);
        assert_eq!(interface.input_quantization, original.input_quantization);
        assert_eq!(interface.output_quantization, original.output_quantization);

        let after = graph_info(&edited).unwrap();
        assert_eq!(after.nodes, before.nodes, "nodes are untouched");
        assert_eq!(after.initializers, before.initializers);
        // Every recorded inner shape now starts with the new batch; the records of
        // tensors no node produces (left by the quantizer) are gone.
        assert_eq!(after.value_info.len(), 7);
        for tensor in &after.value_info {
            assert_eq!(tensor.shape[0], i64::from(batch), "{tensor:?}");
            assert!(after.producer(&tensor.name).is_some(), "{tensor:?}");
            let old = before
                .value_info
                .iter()
                .find(|old| old.name == tensor.name)
                .unwrap();
            assert_eq!(tensor.shape[1..], old.shape[1..]);
            assert_eq!(tensor.element_type, old.element_type);
        }
        // The weights are the same bytes: the file differs only where shapes are stated.
        assert!(edited.len().abs_diff(model.len()) < 400);
    }
    // The model's own batch gives the same interface back, and the edit is repeatable.
    let same = with_batch(&model, 32).unwrap();
    assert_eq!(
        ModelInterface::from_onnx(&same).unwrap(),
        ModelInterface::from_onnx(&model).unwrap()
    );
    assert_eq!(
        with_batch(&with_batch(&model, 8).unwrap(), 32).unwrap(),
        same
    );
}

#[test]
fn a_batch_edit_that_is_not_mechanical_is_refused() {
    let model = spike();
    assert!(
        reason(with_batch(&model, 0).unwrap_err()).contains("at least one text"),
        "a batch of zero"
    );
    assert!(reason(with_batch(b"not a model", 8).unwrap_err()).contains("not a model file"));
    // An operator that may depend on the batch size.
    let mut graph = Vec::new();
    common::bytes_field(
        &mut graph,
        1,
        &common::node("Reshape", &["x", "shape"], &["e"]),
    );
    common::bytes_field(
        &mut graph,
        11,
        &common::value_info("x", common::FLOAT, &[2, 3, 1, 8]),
    );
    common::bytes_field(
        &mut graph,
        12,
        &common::value_info("e", common::FLOAT, &[2, 24]),
    );
    let mut reshaped = Vec::new();
    common::bytes_field(&mut reshaped, 7, &graph);
    let error = reason(with_batch(&reshaped, 4).unwrap_err());
    assert!(error.contains("`Reshape`"), "{error}");

    // The hand-written tiny encoder (Conv between the two quantize pairs) is accepted.
    let tiny = common::Model::tiny().bytes();
    let edited = with_batch(&tiny, 5).unwrap();
    let interface = ModelInterface::from_onnx(&edited).unwrap();
    assert_eq!(interface.input.shape, [5, 3, 1, 8]);
    assert_eq!(interface.output.shape, [5, 4]);
    // A first dimension that is not shared by input and output is not a batch.
    let uneven = common::Model {
        output_shape: vec![3, 4],
        ..common::Model::tiny()
    };
    let error = reason(with_batch(&uneven.bytes(), 4).unwrap_err());
    assert!(error.contains("it is not a batch"), "{error}");
}

#[test]
fn the_quantized_ends_are_opened() {
    let model = spike();
    let before = graph_info(&model).unwrap();

    let both = with_quantized_ends(
        &model,
        QuantizedEnds {
            input: true,
            output: true,
        },
    )
    .unwrap();
    let info = graph_info(&both).unwrap();
    assert_eq!(
        info.inputs,
        [TensorInfo {
            name: "x_QuantizeLinear_Output".into(),
            element_type: ELEMENT_UINT8,
            shape: vec![32, 64, 1, 256]
        }]
    );
    assert_eq!(
        info.outputs,
        [TensorInfo {
            name: "e_QuantizeLinear_Output".into(),
            element_type: ELEMENT_UINT8,
            shape: vec![32, 128]
        }]
    );
    assert_eq!(info.nodes.len(), 36);
    assert!(
        info.nodes
            .iter()
            .all(|node| node.name != "x_QuantizeLinear")
    );
    assert!(
        info.nodes
            .iter()
            .all(|node| node.name != "e_DequantizeLinear")
    );
    // The neighbours still read the scales and zero points, so they all stay.
    assert_eq!(names_of_initializers(&info), names_of_initializers(&before));
    // The package's reader refuses such a file as it stands (2.4 notes): the input is
    // not read by a QuantizeLinear any more.
    assert!(ModelInterface::from_onnx(&both).is_err());

    let input_only = with_quantized_ends(
        &model,
        QuantizedEnds {
            input: true,
            output: false,
        },
    )
    .unwrap();
    let info = graph_info(&input_only).unwrap();
    assert_eq!(info.inputs[0].element_type, ELEMENT_UINT8);
    assert_eq!(info.outputs, before.outputs);
    assert_eq!(info.nodes.len(), 37);

    let output_only = with_quantized_ends(
        &model,
        QuantizedEnds {
            input: false,
            output: true,
        },
    )
    .unwrap();
    let info = graph_info(&output_only).unwrap();
    assert_eq!(info.inputs, before.inputs);
    assert_eq!(info.outputs[0].element_type, ELEMENT_UINT8);
    assert_eq!(info.nodes.len(), 37);

    let neither = with_quantized_ends(
        &model,
        QuantizedEnds {
            input: false,
            output: false,
        },
    )
    .unwrap();
    assert_eq!(graph_info(&neither).unwrap(), before);

    // The edits compose with the batch edit in either order.
    let small = with_batch(&both, 8).unwrap();
    let info = graph_info(&small).unwrap();
    assert_eq!(info.inputs[0].shape, [8, 64, 1, 256]);
    assert_eq!(info.outputs[0].shape, [8, 128]);
    assert_eq!(
        small,
        with_quantized_ends(
            &with_batch(&model, 8).unwrap(),
            QuantizedEnds {
                input: true,
                output: true
            }
        )
        .unwrap()
    );

    // A float model has no ends to open.
    let float = common::Model {
        quantized: false,
        ..common::Model::tiny()
    };
    let error = reason(
        with_quantized_ends(
            &float.bytes(),
            QuantizedEnds {
                input: true,
                output: false,
            },
        )
        .unwrap_err(),
    );
    assert!(error.contains("QuantizeLinear"), "{error}");
}

#[test]
fn the_texts_become_the_rows_of_one_image() {
    let model = spike();
    let before = graph_info(&model).unwrap();
    for rows in [1u32, 32, 64, 256] {
        let edited = with_rows_as_height(&model, rows).unwrap();
        // The package's reader reads it: float in, float out, the same quantization.
        let interface = ModelInterface::from_onnx(&edited).unwrap();
        let original = ModelInterface::from_onnx(&model).unwrap();
        assert_eq!(interface.input.shape, [1, 64, i64::from(rows), 256]);
        assert_eq!(interface.output.shape, [1, 128, i64::from(rows), 1]);
        assert_eq!(interface.input.name, original.input.name);
        assert_eq!(interface.output.name, original.output.name);
        assert_eq!(interface.input_quantization, original.input_quantization);
        assert_eq!(interface.output_quantization, original.output_quantization);

        let info = graph_info(&edited).unwrap();
        // Flatten and its quantize pair are gone; the pool and the projection changed
        // their operator and nothing else moved.
        assert_eq!(info.nodes.len(), 35);
        let counts = info.operator_counts();
        let count = |op: &str| counts.iter().find(|(name, _)| name == op).map(|(_, n)| *n);
        assert_eq!(count("Conv"), Some(6));
        assert_eq!(count("AveragePool"), Some(1));
        assert_eq!(count("QuantizeLinear"), Some(8));
        assert_eq!(count("DequantizeLinear"), Some(20));
        assert_eq!(count("GlobalAveragePool"), None);
        assert_eq!(count("Flatten"), None);
        assert_eq!(count("Gemm"), None);
        let pool = info
            .nodes
            .iter()
            .find(|node| node.op == "AveragePool")
            .unwrap();
        let old_pool = before
            .nodes
            .iter()
            .find(|node| node.op == "GlobalAveragePool")
            .unwrap();
        assert_eq!(
            (&pool.inputs, &pool.outputs),
            (&old_pool.inputs, &old_pool.outputs)
        );
        let head = info
            .nodes
            .iter()
            .find(|node| node.name == "/head/Gemm")
            .unwrap();
        assert_eq!(head.op, "Conv");
        assert_eq!(
            head.inputs,
            [
                "/pool/GlobalAveragePool_output_0_DequantizeLinear_Output",
                "head.weight_DequantizeLinear_Output",
                "head.bias"
            ]
        );
        assert_eq!(head.outputs, ["e_QuantizeLinear_Input"]);
        // Every input of every node has a source.
        for node in &info.nodes {
            for input in &node.inputs {
                assert!(
                    info.producer(input).is_some()
                        || info.inputs.iter().any(|t| t.name == *input)
                        || info.initializers.iter().any(|t| t.name == *input),
                    "{input} of {} has no source",
                    node.name
                );
            }
        }
        // The stored tensors are the same but for the dimensions of the projection.
        assert_eq!(info.initializers.len(), before.initializers.len());
        for (after, was) in info.initializers.iter().zip(&before.initializers) {
            assert_eq!(after.name, was.name);
            assert_eq!(after.element_type, was.element_type);
            if after.name == "head.weight_quantized" {
                assert_eq!(after.shape, [128, 256, 1, 1]);
            } else {
                assert_eq!(after.shape, was.shape, "{}", after.name);
            }
        }
        assert!(info.value_info.is_empty());
        assert!(edited.len().abs_diff(model.len()) < 2_500);
    }
    // The rows do not depend on the batch the file was made for.
    assert_eq!(
        with_rows_as_height(&with_batch(&model, 1).unwrap(), 64).unwrap(),
        with_rows_as_height(&model, 64).unwrap()
    );
    // The quantized ends open on an image as on a batch.
    let bytes = with_quantized_ends(
        &with_rows_as_height(&model, 64).unwrap(),
        QuantizedEnds {
            input: true,
            output: true,
        },
    )
    .unwrap();
    let info = graph_info(&bytes).unwrap();
    assert_eq!(info.inputs[0].shape, [1, 64, 64, 256]);
    assert_eq!(info.inputs[0].element_type, ELEMENT_UINT8);
    assert_eq!(info.outputs[0].shape, [1, 128, 64, 1]);
    assert_eq!(info.outputs[0].element_type, ELEMENT_UINT8);

    // Refusals: no rows, an image made twice, a graph without the encoder's pool.
    assert!(reason(with_rows_as_height(&model, 0).unwrap_err()).contains("at least one text"));
    let twice = with_rows_as_height(&with_rows_as_height(&model, 4).unwrap(), 4).unwrap_err();
    assert!(reason(twice).contains("[batch, channels, 1, length]"));
    let tiny = with_rows_as_height(&common::Model::tiny().bytes(), 4).unwrap_err();
    assert!(reason(tiny).contains("GlobalAveragePool"));
}

#[test]
fn the_graph_is_cut_at_an_inner_tensor() {
    let model = spike();
    // Before the pooling step: the float tensor the GlobalAveragePool reads. The file
    // records no shape for it, so the shape is given.
    let before_pool = TensorInfo {
        name: "/f/f.4/f.4.1/Relu_output_0_DequantizeLinear_Output".into(),
        element_type: ELEMENT_FLOAT32,
        shape: vec![32, 256, 1, 32],
    };
    let cut = with_output_at(&model, &before_pool.name, Some(&before_pool)).unwrap();
    let info = graph_info(&cut).unwrap();
    assert_eq!(info.outputs, std::slice::from_ref(&before_pool));
    assert_eq!(info.inputs[0].shape, [32, 64, 1, 256]);
    assert!(info.nodes.iter().all(|node| node.op != "GlobalAveragePool"));
    assert!(info.nodes.iter().all(|node| node.op != "Gemm"));
    assert_eq!(
        info.operator_counts(),
        [
            ("DequantizeLinear".to_string(), 16),
            ("QuantizeLinear".to_string(), 6),
            ("Conv".to_string(), 5)
        ]
    );
    // Only what the kept nodes read is still stored.
    assert!(
        info.initializers
            .iter()
            .all(|t| !t.name.starts_with("head."))
    );
    assert!(info.initializers.iter().all(|t| !t.name.starts_with("e_")));
    for node in &info.nodes {
        for input in &node.inputs {
            assert!(
                info.producer(input).is_some()
                    || info.inputs.iter().any(|t| t.name == *input)
                    || info.initializers.iter().any(|t| t.name == *input),
                "{input} of {} has no source",
                node.name
            );
        }
    }
    // It ends in a DequantizeLinear, so the package's reader reads it.
    let interface = ModelInterface::from_onnx(&cut).unwrap();
    assert_eq!(interface.output.shape, [32, 256, 1, 32]);
    assert_eq!(
        interface.output_quantization.tensor,
        "/f/f.4/f.4.1/Relu_output_0_QuantizeLinear_Output"
    );

    // After the pooling step.
    let after_pool = TensorInfo {
        name: "/pool/GlobalAveragePool_output_0_DequantizeLinear_Output".into(),
        element_type: ELEMENT_FLOAT32,
        shape: vec![32, 256, 1, 1],
    };
    let cut = with_output_at(&model, &after_pool.name, Some(&after_pool)).unwrap();
    let info = graph_info(&cut).unwrap();
    assert_eq!(info.outputs, std::slice::from_ref(&after_pool));
    assert_eq!(
        info.nodes.last().unwrap().name,
        "/pool/GlobalAveragePool_output_0_DequantizeLinear"
    );
    assert!(info.nodes.iter().any(|node| node.op == "GlobalAveragePool"));
    assert!(info.nodes.iter().all(|node| node.op != "Flatten"));
    assert_eq!(
        ModelInterface::from_onnx(&cut).unwrap().output.shape,
        [32, 256, 1, 1]
    );
    // A cut model takes the batch edit like the whole one.
    let small = with_batch(&cut, 1).unwrap();
    assert_eq!(graph_info(&small).unwrap().outputs[0].shape, [1, 256, 1, 1]);

    // A tensor whose shape the file records needs none given.
    let recorded = with_output_at(&model, "/Flatten_output_0", None).unwrap();
    assert_eq!(
        graph_info(&recorded).unwrap().outputs,
        [TensorInfo {
            name: "/Flatten_output_0".into(),
            element_type: ELEMENT_FLOAT32,
            shape: vec![32, 256]
        }]
    );

    // Refusals.
    let error = reason(with_output_at(&model, &before_pool.name, None).unwrap_err());
    assert!(error.contains("records no type and shape"), "{error}");
    let nowhere = TensorInfo {
        name: "nowhere".into(),
        element_type: ELEMENT_FLOAT32,
        shape: vec![32, 4],
    };
    let error = reason(with_output_at(&model, "nowhere", Some(&nowhere)).unwrap_err());
    assert!(error.contains("no node produces"), "{error}");
}

// ---------------------------------------------------------------------------------------
// A small encoder written by hand, to hold `with_rows_as_height` to what it promises.

fn int_attribute(name: &str, value: i64) -> Vec<u8> {
    let mut attribute = Vec::new();
    common::bytes_field(&mut attribute, 1, name.as_bytes());
    common::varint_field(&mut attribute, 3, value as u64);
    common::varint_field(&mut attribute, 20, 2);
    attribute
}

fn float_attribute(name: &str, value: f32) -> Vec<u8> {
    let mut attribute = Vec::new();
    common::bytes_field(&mut attribute, 1, name.as_bytes());
    common::varint(&mut attribute, 2 << 3 | 5);
    attribute.extend_from_slice(&value.to_le_bytes());
    common::varint_field(&mut attribute, 20, 1);
    attribute
}

fn ints_attribute(name: &str, values: &[i64]) -> Vec<u8> {
    let mut attribute = Vec::new();
    common::bytes_field(&mut attribute, 1, name.as_bytes());
    for value in values {
        common::varint_field(&mut attribute, 8, *value as u64);
    }
    common::varint_field(&mut attribute, 20, 7);
    attribute
}

fn string_attribute(name: &str, value: &str) -> Vec<u8> {
    let mut attribute = Vec::new();
    common::bytes_field(&mut attribute, 1, name.as_bytes());
    common::bytes_field(&mut attribute, 4, value.as_bytes());
    common::varint_field(&mut attribute, 20, 3);
    attribute
}

fn named_node(
    op: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attributes: &[Vec<u8>],
) -> Vec<u8> {
    let mut node = common::node(op, inputs, outputs);
    common::bytes_field(&mut node, 3, name.as_bytes());
    for attribute in attributes {
        common::bytes_field(&mut node, 5, attribute);
    }
    node
}

/// A stored float tensor of the given dimensions, all zeros.
fn stored(name: &str, dims: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    for dim in dims {
        common::varint_field(&mut tensor, 1, *dim as u64);
    }
    common::varint_field(&mut tensor, 2, common::FLOAT);
    common::bytes_field(&mut tensor, 8, name.as_bytes());
    let values: i64 = dims.iter().product();
    common::bytes_field(&mut tensor, 9, &vec![0u8; values as usize * 4]);
    tensor
}

/// Input `[2, 3, 1, length]`, one convolution (4 channels), pool, Flatten, a projection
/// to 5 values, with a quantize pair at each end. Every field is something a test turns.
#[derive(Clone)]
struct Encoder {
    length: i64,
    conv_weight: Vec<i64>,
    conv_attributes: Vec<Vec<u8>>,
    flatten_attributes: Vec<Vec<u8>>,
    gemm_attributes: Vec<Vec<u8>>,
    gemm_bias: Vec<i64>,
    /// A quantize pair between the pool and Flatten (scale 0.5, zero point 3).
    pair_before: bool,
    /// A quantize pair between Flatten and the projection: its scale and zero point,
    /// stored under names of their own.
    pair_behind: Option<(f32, i32)>,
}

impl Encoder {
    fn new(length: i64) -> Self {
        Self {
            length,
            conv_weight: vec![4, 3, 1, 3],
            conv_attributes: vec![ints_attribute("pads", &[0, 1, 0, 1])],
            flatten_attributes: Vec::new(),
            gemm_attributes: vec![int_attribute("transB", 1)],
            gemm_bias: vec![5],
            pair_before: false,
            pair_behind: None,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        use common::Storage::Raw;
        let mut nodes = vec![
            named_node("QuantizeLinear", "x_q", &["x", "x_s", "x_z"], &["xq"], &[]),
            named_node(
                "DequantizeLinear",
                "x_dq",
                &["xq", "x_s", "x_z"],
                &["xdq"],
                &[],
            ),
            named_node("Conv", "conv", &["xdq", "w"], &["h"], &self.conv_attributes),
            named_node("GlobalAveragePool", "pool", &["h"], &["p"], &[]),
        ];
        let mut at = "p";
        if self.pair_before {
            nodes.push(named_node(
                "QuantizeLinear",
                "p_q",
                &["p", "p_s", "p_z"],
                &["pq"],
                &[],
            ));
            nodes.push(named_node(
                "DequantizeLinear",
                "p_dq",
                &["pq", "p_s", "p_z"],
                &["pdq"],
                &[],
            ));
            at = "pdq";
        }
        nodes.push(named_node(
            "Flatten",
            "flatten",
            &[at],
            &["f"],
            &self.flatten_attributes,
        ));
        at = "f";
        if self.pair_behind.is_some() {
            nodes.push(named_node(
                "QuantizeLinear",
                "f_q",
                &["f", "f_s", "f_z"],
                &["fq"],
                &[],
            ));
            nodes.push(named_node(
                "DequantizeLinear",
                "f_dq",
                &["fq", "f_s", "f_z"],
                &["fdq"],
                &[],
            ));
            at = "fdq";
        }
        nodes.push(named_node(
            "Gemm",
            "head",
            &[at, "gw", "gb"],
            &["g"],
            &self.gemm_attributes,
        ));
        nodes.push(named_node(
            "QuantizeLinear",
            "e_q",
            &["g", "e_s", "e_z"],
            &["eq"],
            &[],
        ));
        nodes.push(named_node(
            "DequantizeLinear",
            "e_dq",
            &["eq", "e_s", "e_z"],
            &["e"],
            &[],
        ));

        let mut tensors = vec![
            common::float_scalar("x_s", 0.25, Raw),
            common::byte_scalar("x_z", common::UINT8, 100, Raw),
            common::float_scalar("e_s", 0.125, Raw),
            common::byte_scalar("e_z", common::UINT8, 90, Raw),
            common::float_scalar("p_s", 0.5, Raw),
            common::byte_scalar("p_z", common::UINT8, 3, Raw),
            stored("w", &self.conv_weight),
            stored("gw", &[5, 4]),
            stored("gb", &self.gemm_bias),
        ];
        if let Some((scale, zero)) = self.pair_behind {
            tensors.push(common::float_scalar("f_s", scale, Raw));
            tensors.push(common::byte_scalar("f_z", common::UINT8, zero, Raw));
        }
        let mut graph = Vec::new();
        for node in &nodes {
            common::bytes_field(&mut graph, 1, node);
        }
        for tensor in &tensors {
            common::bytes_field(&mut graph, 5, tensor);
        }
        let input = common::value_info("x", common::FLOAT, &[2, 3, 1, self.length]);
        let output = common::value_info("e", common::FLOAT, &[2, 5]);
        let inner = common::value_info("h", common::FLOAT, &[2, 4, 1, self.length]);
        common::bytes_field(&mut graph, 11, &input);
        common::bytes_field(&mut graph, 12, &output);
        common::bytes_field(&mut graph, 13, &inner);
        let mut model = Vec::new();
        common::bytes_field(&mut model, 7, &graph);
        model
    }
}

#[test]
fn the_pool_of_an_image_averages_over_the_models_own_length() {
    // The fixture pools over 32 positions.
    let edited = with_rows_as_height(&spike(), 64).unwrap();
    let pool = "/pool/GlobalAveragePool";
    assert_eq!(
        node_ints_attribute(&edited, pool, "kernel_shape").unwrap(),
        Some(vec![1, 32])
    );
    assert_eq!(
        node_ints_attribute(&edited, pool, "strides").unwrap(),
        Some(vec![1, 1])
    );
    assert_eq!(node_ints_attribute(&edited, pool, "pads").unwrap(), None);
    assert!(node_ints_attribute(&edited, "no such node", "strides").is_err());
    // The convolutions keep their attributes, and the projection needs none.
    let conv = "/f/f.1/f.1.0/Conv";
    assert_eq!(
        node_ints_attribute(&edited, conv, "pads").unwrap(),
        Some(vec![0, 2, 0, 2])
    );
    assert_eq!(
        node_ints_attribute(&edited, conv, "strides").unwrap(),
        Some(vec![1, 2])
    );
    assert_eq!(
        node_ints_attribute(&edited, "/head/Gemm", "kernel_shape").unwrap(),
        None
    );

    // Hand-written encoders that pool over other lengths: the kernel follows the file.
    for length in [8i64, 20] {
        let model = Encoder::new(length).bytes();
        let edited = with_rows_as_height(&model, 7).unwrap();
        assert_eq!(
            node_ints_attribute(&edited, "pool", "kernel_shape").unwrap(),
            Some(vec![1, length])
        );
        assert_eq!(
            node_ints_attribute(&edited, "pool", "strides").unwrap(),
            Some(vec![1, 1])
        );
        let interface = ModelInterface::from_onnx(&edited).unwrap();
        assert_eq!(interface.input.shape, [1, 3, 7, length]);
        assert_eq!(interface.output.shape, [1, 5, 7, 1]);
        let info = graph_info(&edited).unwrap();
        assert_eq!(info.nodes.len(), 7);
        assert_eq!(
            info.nodes
                .iter()
                .find(|node| node.name == "head")
                .unwrap()
                .op,
            "Conv"
        );
        let shape = |name: &str| {
            &info
                .initializers
                .iter()
                .find(|t| t.name == name)
                .unwrap()
                .shape
        };
        assert_eq!(shape("gw"), &[5, 4, 1, 1]);
        assert_eq!(shape("gb"), &[5]);
        assert_eq!(shape("w"), &[4, 3, 1, 3]);
    }
}

#[test]
fn a_quantize_pair_behind_flatten_is_removed_only_when_it_changes_nothing() {
    // The same scale and zero point as the pair before Flatten, under other names.
    let same = Encoder {
        pair_before: true,
        pair_behind: Some((0.5, 3)),
        ..Encoder::new(8)
    };
    let edited = with_rows_as_height(&same.bytes(), 4).unwrap();
    let info = graph_info(&edited).unwrap();
    assert_eq!(
        info.nodes.len(),
        9,
        "Flatten and the pair behind it are gone"
    );
    assert!(
        info.nodes
            .iter()
            .all(|node| !["flatten", "f_q", "f_dq"].contains(&node.name.as_str()))
    );
    let head = info.nodes.iter().find(|node| node.name == "head").unwrap();
    assert_eq!(head.inputs, ["pdq", "gw", "gb"]);

    // Another scale, another zero point, or no pair before Flatten: refused.
    for (pair_before, behind, expected) in [
        (true, (0.25, 3), "another scale"),
        (true, (0.5, 4), "another zero point"),
        (false, (0.5, 3), "none precedes it"),
    ] {
        let model = Encoder {
            pair_before,
            pair_behind: Some(behind),
            ..Encoder::new(8)
        };
        let error = reason(with_rows_as_height(&model.bytes(), 4).unwrap_err());
        assert!(error.contains(expected), "{error}");
        assert!(error.contains("would change the arithmetic"), "{error}");
    }
    // A pair before Flatten alone stays where it is.
    let before_only = Encoder {
        pair_before: true,
        ..Encoder::new(8)
    };
    let info = graph_info(&with_rows_as_height(&before_only.bytes(), 4).unwrap()).unwrap();
    assert_eq!(info.nodes.len(), 9);
}

#[test]
fn an_image_is_refused_when_its_rows_would_not_stay_apart() {
    let refusal = |model: Encoder| reason(with_rows_as_height(&model.bytes(), 4).unwrap_err());
    let base = Encoder::new(8);
    let with_conv = |attributes: Vec<Vec<u8>>| Encoder {
        conv_attributes: attributes,
        ..base.clone()
    };
    // A kernel higher than one row reads the neighbouring texts.
    let error = refusal(Encoder {
        conv_weight: vec![4, 3, 3, 3],
        conv_attributes: vec![ints_attribute("pads", &[1, 1, 1, 1])],
        ..base.clone()
    });
    assert!(
        error.contains("`conv`") && error.contains("[4, 3, 3, 3]"),
        "{error}"
    );
    let error = refusal(Encoder {
        conv_weight: vec![4, 3, 3],
        ..base.clone()
    });
    assert!(error.contains("[4, 3, 3]"), "{error}");
    for (attributes, expected) in [
        (
            vec![ints_attribute("kernel_shape", &[3, 3])],
            "kernel_shape is [3, 3]",
        ),
        (
            vec![ints_attribute("pads", &[1, 1, 1, 1])],
            "pads are [1, 1, 1, 1]",
        ),
        (
            vec![ints_attribute("pads", &[0, 1, 1, 1])],
            "pads are [0, 1, 1, 1]",
        ),
        (
            vec![ints_attribute("strides", &[2, 1])],
            "strides are [2, 1]",
        ),
        (
            vec![ints_attribute("dilations", &[2, 1])],
            "dilations are [2, 1]",
        ),
        (
            vec![string_attribute("auto_pad", "SAME_UPPER")],
            "auto_pad is `SAME_UPPER`",
        ),
        (
            vec![string_attribute("auto_pad", "VALID")],
            "auto_pad is `VALID`",
        ),
    ] {
        let error = refusal(with_conv(attributes));
        assert!(
            error.contains("would not keep the rows of an image apart"),
            "{error}"
        );
        assert!(error.contains(expected), "{error}");
    }
    // What moves along the width only is the encoder's own and is accepted.
    for attributes in [
        vec![
            ints_attribute("kernel_shape", &[1, 3]),
            ints_attribute("strides", &[1, 2]),
        ],
        vec![
            ints_attribute("dilations", &[1, 2]),
            ints_attribute("pads", &[0, 2, 0, 2]),
        ],
        vec![
            string_attribute("auto_pad", "NOTSET"),
            int_attribute("group", 1),
        ],
        Vec::new(),
    ] {
        with_rows_as_height(&with_conv(attributes).bytes(), 4).unwrap();
    }

    // The projection must be y = x W^T + b with one bias value per output.
    let with_gemm = |attributes: Vec<Vec<u8>>| Encoder {
        gemm_attributes: attributes,
        ..base.clone()
    };
    for attributes in [
        Vec::new(),
        vec![int_attribute("transB", 0)],
        vec![int_attribute("transB", 1), int_attribute("transA", 1)],
        vec![int_attribute("transB", 1), float_attribute("alpha", 0.5)],
        vec![int_attribute("transB", 1), float_attribute("beta", 2.0)],
    ] {
        let error = refusal(with_gemm(attributes));
        assert!(error.contains("not y = x W^T + b"), "{error}");
    }
    with_rows_as_height(
        &with_gemm(vec![
            int_attribute("transB", 1),
            int_attribute("transA", 0),
            float_attribute("alpha", 1.0),
            float_attribute("beta", 1.0),
        ])
        .bytes(),
        4,
    )
    .unwrap();
    for dims in [vec![1, 5], vec![5, 1], vec![4], vec![]] {
        let error = refusal(Encoder {
            gemm_bias: dims.clone(),
            ..base.clone()
        });
        assert!(
            error.contains("bias") && error.contains(&format!("{dims:?}")),
            "{error}"
        );
    }

    // Flatten at another axis does not give one row per text.
    for axis in [0, 2] {
        let error = refusal(Encoder {
            flatten_attributes: vec![int_attribute("axis", axis)],
            ..base.clone()
        });
        assert!(error.contains("Flatten node is not at axis 1"), "{error}");
    }
    with_rows_as_height(
        &Encoder {
            flatten_attributes: vec![int_attribute("axis", 1)],
            ..base.clone()
        }
        .bytes(),
        4,
    )
    .unwrap();
}
