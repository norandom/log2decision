//! The mechanical model edits of task 2.12 (`l2d_encoder::onnx_edit`), without hardware:
//! every edited file is read again, by the package's own model reader (`onnx.rs`) where
//! that reader accepts the result and by `graph_info` everywhere. Whether the NPU runs
//! the edited files is shown by the hardware tests (`npu_rows.rs`, `npu_throughput.rs`).

mod common;

use l2d_core::L2dError;
use l2d_encoder::ModelInterface;
use l2d_encoder::onnx_edit::{
    ELEMENT_FLOAT32, ELEMENT_UINT8, GraphInfo, QuantizedEnds, TensorInfo, graph_info, with_batch,
    with_output_at, with_quantized_ends, with_rows_as_height,
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
