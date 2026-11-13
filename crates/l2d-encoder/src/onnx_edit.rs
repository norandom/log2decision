//! Mechanical edits of an encoder model file (task 2.12).
//!
//! The trained 8-bit model is a file with fixed shapes: a batch of 32, a float input that
//! a `QuantizeLinear` node turns into bytes, and a float output a `DequantizeLinear` node
//! produces. To measure what the NPU does with other batch sizes, with the quantized
//! bytes fed and read directly, and with the graph cut before its pooling step, the file
//! is rewritten here, without retraining and without an ONNX library:
//!
//! - [`with_rows_as_height`]: the same encoder with its texts as the rows of one image,
//!   the layout in which the NPU computes every text right (task 2.12);
//! - [`with_batch`]: the same graph for another batch size;
//! - [`with_quantized_ends`]: the same graph without the input's `QuantizeLinear` and/or
//!   the output's `DequantizeLinear`, so that its input and/or output are the bytes;
//! - [`with_output_at`]: the graph up to an inner tensor, which becomes the output;
//! - [`graph_info`]: what a file holds (nodes, tensors with their shapes), which is how
//!   the edits are checked.
//!
//! Every edit copies the protobuf fields it does not understand byte for byte and
//! rewrites only what the edit is about; weights are never touched. An edit that cannot
//! be shown to be mechanical is refused: [`with_batch`] accepts only graphs whose every
//! operator treats the first dimension as the batch (the operators of the encoder).
//!
//! These are development tools for model files the project made itself. Unlike
//! [`crate::onnx`], which reads files it is handed and bounds what it keeps, the edits
//! hold the list of a graph's fields in memory.

use std::collections::{HashMap, HashSet};

use l2d_core::L2dError;

/// ONNX's element type numbers used here.
pub const ELEMENT_FLOAT32: i64 = 1;
/// ONNX's element type number of unsigned bytes.
pub const ELEMENT_UINT8: i64 = 2;
/// ONNX's element type number of signed bytes.
pub const ELEMENT_INT8: i64 = 3;

/// The operators whose first input carries the batch in its first dimension and whose
/// output keeps it there, whatever the batch size: those of the encoder graph.
pub const BATCH_FREE_OPERATORS: [&str; 7] = [
    "Conv",
    "Relu",
    "GlobalAveragePool",
    "Flatten",
    "Gemm",
    "QuantizeLinear",
    "DequantizeLinear",
];

/// One node of a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeInfo {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// One tensor a graph declares: a graph input or output, or an inner tensor whose shape
/// the file records. A dimension that is not a fixed number is `-1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// ONNX's `TensorProto.DataType` number; `0` when the file gives none.
    pub element_type: i64,
    pub shape: Vec<i64>,
}

/// What a model file holds, as far as the edits and their checks need it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphInfo {
    /// The graph inputs that are not initializers.
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
    /// Inner tensors with a recorded type and shape (`value_info`).
    pub value_info: Vec<TensorInfo>,
    pub nodes: Vec<NodeInfo>,
    /// The stored tensors: name, element type and dimensions.
    pub initializers: Vec<TensorInfo>,
}

impl GraphInfo {
    /// The node that produces `tensor`.
    pub fn producer(&self, tensor: &str) -> Option<&NodeInfo> {
        self.nodes
            .iter()
            .find(|node| node.outputs.iter().any(|output| output == tensor))
    }

    /// The nodes that read `tensor`.
    pub fn readers(&self, tensor: &str) -> Vec<&NodeInfo> {
        self.nodes
            .iter()
            .filter(|node| node.inputs.iter().any(|input| input == tensor))
            .collect()
    }

    /// How many nodes there are of each operator, in order of first appearance.
    pub fn operator_counts(&self) -> Vec<(String, usize)> {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for node in &self.nodes {
            match counts.iter_mut().find(|(op, _)| *op == node.op) {
                Some((_, count)) => *count += 1,
                None => counts.push((node.op.clone(), 1)),
            }
        }
        counts
    }
}

/// Which ends of the graph [`with_quantized_ends`] opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedEnds {
    /// Remove the `QuantizeLinear` that reads the graph input: the input becomes bytes.
    pub input: bool,
    /// Remove the `DequantizeLinear` that produces the graph output: the output becomes
    /// bytes.
    pub output: bool,
}

fn refused(reason: String) -> L2dError {
    L2dError::Refused {
        reason: format!("the model file cannot be edited: {reason}"),
    }
}

// ---------------------------------------------------------------------------------------
// The protobuf wire format, with the bytes of every field kept for copying.

#[derive(Clone, Copy)]
struct Field<'a> {
    number: u64,
    /// The whole field: key, length and payload.
    raw: &'a [u8],
    /// The payload of a length-delimited field; empty for the other wire types.
    payload: &'a [u8],
    /// The value of a varint field; `0` for the other wire types.
    value: u64,
    /// Wire type: 0 varint, 1 64-bit, 2 length-delimited, 5 32-bit.
    wire: u8,
}

fn read_varint(bytes: &[u8], at: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    for shift in 0..10 {
        let byte = *bytes.get(*at)?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << (7 * shift);
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

fn split<'a>(message: &'a [u8], what: &str) -> Result<Vec<Field<'a>>, String> {
    let broken = || format!("it is not a model file or is cut short (in {what})");
    let mut fields = Vec::new();
    let mut at = 0usize;
    while at < message.len() {
        let start = at;
        let key = read_varint(message, &mut at).ok_or_else(broken)?;
        let wire = (key & 7) as u8;
        let mut payload: &[u8] = &[];
        let mut value = 0u64;
        match wire {
            0 => value = read_varint(message, &mut at).ok_or_else(broken)?,
            1 => at = at.checked_add(8).ok_or_else(broken)?,
            5 => at = at.checked_add(4).ok_or_else(broken)?,
            2 => {
                let length = read_varint(message, &mut at).ok_or_else(broken)?;
                let length = usize::try_from(length).map_err(|_| broken())?;
                let end = at.checked_add(length).ok_or_else(broken)?;
                payload = message.get(at..end).ok_or_else(broken)?;
                at = end;
            }
            _ => return Err(broken()),
        }
        let raw = message.get(start..at).ok_or_else(broken)?;
        fields.push(Field {
            number: key >> 3,
            raw,
            payload,
            value,
            wire,
        });
    }
    Ok(fields)
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn put_varint_field(out: &mut Vec<u8>, number: u64, value: u64) {
    put_varint(out, number << 3);
    put_varint(out, value);
}

fn put_bytes_field(out: &mut Vec<u8>, number: u64, payload: &[u8]) {
    put_varint(out, (number << 3) | 2);
    put_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn text(bytes: &[u8], what: &str) -> Result<String, String> {
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| format!("{what} is not UTF-8 text"))
}

// GraphProto fields.
const GRAPH_NODE: u64 = 1;
const GRAPH_INITIALIZER: u64 = 5;
const GRAPH_INPUT: u64 = 11;
const GRAPH_OUTPUT: u64 = 12;
const GRAPH_VALUE_INFO: u64 = 13;
/// `ModelProto.graph`.
const MODEL_GRAPH: u64 = 7;

fn graph_of(model: &[u8]) -> Result<&[u8], String> {
    let mut graph = None;
    for field in split(model, "the model")? {
        if field.number == MODEL_GRAPH && field.wire == 2 {
            if graph.is_some() {
                return Err("it holds more than one graph".into());
            }
            graph = Some(field.payload);
        }
    }
    graph.ok_or_else(|| "it holds no graph: not an ONNX model file".to_string())
}

/// The model with its graph replaced; every other field of the model is copied.
fn with_graph(model: &[u8], graph: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(model.len() + 64);
    for field in split(model, "the model")? {
        if field.number == MODEL_GRAPH && field.wire == 2 {
            put_bytes_field(&mut out, MODEL_GRAPH, graph);
        } else {
            out.extend_from_slice(field.raw);
        }
    }
    Ok(out)
}

fn parse_node(node: &[u8]) -> Result<NodeInfo, String> {
    let mut info = NodeInfo {
        name: String::new(),
        op: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    for field in split(node, "a node")? {
        if field.wire != 2 {
            continue;
        }
        match field.number {
            1 => info.inputs.push(text(field.payload, "a node input")?),
            2 => info.outputs.push(text(field.payload, "a node output")?),
            3 => info.name = text(field.payload, "a node name")?,
            4 => info.op = text(field.payload, "an operator name")?,
            _ => {}
        }
    }
    Ok(info)
}

/// The integer attribute `name` of a node, when it has one.
fn int_attribute(node: &[u8], name: &str) -> Result<Option<i64>, String> {
    for field in split(node, "a node")? {
        if field.number != 5 || field.wire != 2 {
            continue;
        }
        let mut found_name = false;
        let mut value = None;
        for part in split(field.payload, "an attribute")? {
            match (part.number, part.wire) {
                (1, 2) => found_name = part.payload == name.as_bytes(),
                (3, 0) => value = Some(part.value as i64),
                _ => {}
            }
        }
        if found_name {
            return Ok(Some(value.unwrap_or(0)));
        }
    }
    Ok(None)
}

fn parse_value_info(info: &[u8]) -> Result<TensorInfo, String> {
    let mut tensor = TensorInfo {
        name: String::new(),
        element_type: 0,
        shape: Vec::new(),
    };
    for field in split(info, "a tensor description")? {
        match (field.number, field.wire) {
            (1, 2) => tensor.name = text(field.payload, "a tensor name")?,
            (2, 2) => {
                for kind in split(field.payload, "a tensor type")? {
                    if kind.number != 1 || kind.wire != 2 {
                        continue;
                    }
                    for part in split(kind.payload, "a tensor type")? {
                        match (part.number, part.wire) {
                            (1, 0) => tensor.element_type = part.value as i64,
                            (2, 2) => {
                                for dim in split(part.payload, "a tensor shape")? {
                                    if dim.number != 1 || dim.wire != 2 {
                                        continue;
                                    }
                                    let mut size = -1i64;
                                    for value in split(dim.payload, "a dimension")? {
                                        if value.number == 1 && value.wire == 0 {
                                            size = value.value as i64;
                                        }
                                    }
                                    tensor.shape.push(size);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(tensor)
}

fn parse_initializer(tensor: &[u8]) -> Result<TensorInfo, String> {
    let mut info = TensorInfo {
        name: String::new(),
        element_type: 0,
        shape: Vec::new(),
    };
    for field in split(tensor, "a stored tensor")? {
        match (field.number, field.wire) {
            (1, 0) => info.shape.push(field.value as i64),
            // Packed dimensions.
            (1, 2) => {
                let mut at = 0usize;
                while at < field.payload.len() {
                    let value = read_varint(field.payload, &mut at)
                        .ok_or("a stored tensor's dimensions are cut short")?;
                    info.shape.push(value as i64);
                }
            }
            (2, 0) => info.element_type = field.value as i64,
            (8, 2) => info.name = text(field.payload, "a stored tensor's name")?,
            _ => {}
        }
    }
    Ok(info)
}

/// A `ValueInfoProto` of a tensor with a fixed shape.
fn value_info_bytes(tensor: &TensorInfo) -> Vec<u8> {
    let mut shape = Vec::new();
    for dim in &tensor.shape {
        let mut dimension = Vec::new();
        put_varint_field(&mut dimension, 1, *dim as u64);
        put_bytes_field(&mut shape, 1, &dimension);
    }
    let mut tensor_type = Vec::new();
    put_varint_field(&mut tensor_type, 1, tensor.element_type as u64);
    put_bytes_field(&mut tensor_type, 2, &shape);
    let mut kind = Vec::new();
    put_bytes_field(&mut kind, 1, &tensor_type);
    let mut out = Vec::new();
    put_bytes_field(&mut out, 1, tensor.name.as_bytes());
    put_bytes_field(&mut out, 2, &kind);
    out
}

fn info_of_graph(graph: &[u8]) -> Result<GraphInfo, String> {
    let mut info = GraphInfo {
        inputs: Vec::new(),
        outputs: Vec::new(),
        value_info: Vec::new(),
        nodes: Vec::new(),
        initializers: Vec::new(),
    };
    for field in split(graph, "the graph")? {
        if field.wire != 2 {
            continue;
        }
        match field.number {
            GRAPH_NODE => info.nodes.push(parse_node(field.payload)?),
            GRAPH_INITIALIZER => info.initializers.push(parse_initializer(field.payload)?),
            GRAPH_INPUT => info.inputs.push(parse_value_info(field.payload)?),
            GRAPH_OUTPUT => info.outputs.push(parse_value_info(field.payload)?),
            GRAPH_VALUE_INFO => info.value_info.push(parse_value_info(field.payload)?),
            _ => {}
        }
    }
    // Old exporters list initializers among the inputs: those are not inputs.
    let stored: HashSet<&str> = info
        .initializers
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    info.inputs
        .retain(|input| !stored.contains(input.name.as_str()));
    Ok(info)
}

/// What the model file holds: its nodes in order, its inputs and outputs, the inner
/// tensors whose shapes it records, and its stored tensors.
pub fn graph_info(model: &[u8]) -> Result<GraphInfo, L2dError> {
    graph_of(model).and_then(info_of_graph).map_err(refused)
}

/// The one input and the one output of an encoder graph, with fixed shapes.
fn ends(info: &GraphInfo) -> Result<(&TensorInfo, &TensorInfo), String> {
    let [input] = info.inputs.as_slice() else {
        return Err(format!(
            "an encoder has one input, this graph has {}",
            info.inputs.len()
        ));
    };
    let [output] = info.outputs.as_slice() else {
        return Err(format!(
            "an encoder has one output, this graph has {}",
            info.outputs.len()
        ));
    };
    for tensor in [input, output] {
        if tensor.shape.is_empty() || tensor.shape.iter().any(|dim| *dim <= 0) {
            return Err(format!(
                "the tensor `{}` has no fixed shape ({:?})",
                tensor.name, tensor.shape
            ));
        }
    }
    Ok((input, output))
}

/// The tensors that carry the batch in their first dimension: the graph input and every
/// output of a node whose first input carries it.
fn batch_tensors(info: &GraphInfo, input: &str) -> HashSet<String> {
    let mut carrying: HashSet<String> = HashSet::from([input.to_string()]);
    // The nodes of a model file are in topological order; a second pass costs nothing
    // and covers a file that is not.
    for _ in 0..2 {
        for node in &info.nodes {
            if node
                .inputs
                .first()
                .is_some_and(|first| carrying.contains(first))
            {
                carrying.extend(node.outputs.iter().cloned());
            }
        }
    }
    carrying
}

/// The same graph for a batch of `batch` texts: the first dimension of the input, of the
/// output and of every inner tensor that carries the batch is rewritten, and a recorded
/// shape of a tensor that no node produces is dropped. Weights, nodes and everything
/// else are copied byte for byte.
///
/// Refused unless the edit is mechanical: one input and one output with fixed shapes and
/// the same first dimension, only operators of [`BATCH_FREE_OPERATORS`], and every
/// `Flatten` at axis 1 (which keeps the batch apart whatever its size).
pub fn with_batch(model: &[u8], batch: u32) -> Result<Vec<u8>, L2dError> {
    batch_edit(model, batch).map_err(refused)
}

fn batch_edit(model: &[u8], batch: u32) -> Result<Vec<u8>, String> {
    if batch == 0 {
        return Err("a batch holds at least one text".into());
    }
    let graph = graph_of(model)?;
    let info = info_of_graph(graph)?;
    let (input, output) = ends(&info)?;
    let old = input.shape[0];
    if output.shape[0] != old {
        return Err(format!(
            "the first dimension of the input ({old}) and of the output ({}) differ: it is \
             not a batch",
            output.shape[0]
        ));
    }
    let fields = split(graph, "the graph")?;
    for field in &fields {
        if field.number != GRAPH_NODE || field.wire != 2 {
            continue;
        }
        let node = parse_node(field.payload)?;
        if !BATCH_FREE_OPERATORS.contains(&node.op.as_str()) {
            return Err(format!(
                "the operator `{}` (node `{}`) is not known to keep the batch apart; the \
                 known ones are {}",
                node.op,
                node.name,
                BATCH_FREE_OPERATORS.join(", ")
            ));
        }
        if node.op == "Flatten" {
            let axis = int_attribute(field.payload, "axis")?.unwrap_or(1);
            if axis != 1 {
                return Err(format!(
                    "the Flatten node `{}` has axis {axis}: only axis 1 keeps the batch apart",
                    node.name
                ));
            }
        }
    }
    let carrying = batch_tensors(&info, &input.name);
    if !carrying.contains(&output.name) {
        return Err(format!(
            "the output `{}` is not reached from the input through first inputs",
            output.name
        ));
    }
    let mut out = Vec::with_capacity(graph.len());
    for field in &fields {
        let declares = matches!(field.number, GRAPH_INPUT | GRAPH_OUTPUT | GRAPH_VALUE_INFO)
            && field.wire == 2;
        if !declares {
            out.extend_from_slice(field.raw);
            continue;
        }
        let mut tensor = parse_value_info(field.payload)?;
        // A record of a tensor no node produces is left over from an earlier rewrite of
        // the graph (the quantizer folds each Relu into its Conv and keeps the record of
        // the Conv's old output). It would state the old batch; it is dropped.
        if field.number == GRAPH_VALUE_INFO && info.producer(&tensor.name).is_none() {
            continue;
        }
        if !carrying.contains(&tensor.name) {
            out.extend_from_slice(field.raw);
            continue;
        }
        if tensor.shape.first() != Some(&old) || tensor.shape.iter().any(|dim| *dim <= 0) {
            return Err(format!(
                "the tensor `{}` carries the batch but its recorded shape is {:?}, not \
                 [{old}, ...]",
                tensor.name, tensor.shape
            ));
        }
        tensor.shape[0] = i64::from(batch);
        put_bytes_field(&mut out, field.number, &value_info_bytes(&tensor));
    }
    with_graph(model, &out)
}

/// The same graph without the `QuantizeLinear` node that reads its input and/or the
/// `DequantizeLinear` node that produces its output. The graph input then is the tensor
/// of bytes that node produced (same shape, the element type of its zero point), and the
/// output the tensor of bytes that node read. Whoever feeds such a model must quantize
/// as the removed node did, `q = saturate(round_half_even(x / scale) + zero_point)`, and
/// whoever reads it must compute `e = (q - zero_point) * scale`; the scales and zero
/// points stay in the file (the neighbouring nodes use them).
pub fn with_quantized_ends(model: &[u8], open: QuantizedEnds) -> Result<Vec<u8>, L2dError> {
    ends_edit(model, open).map_err(refused)
}

fn ends_edit(model: &[u8], open: QuantizedEnds) -> Result<Vec<u8>, String> {
    let graph = graph_of(model)?;
    let info = info_of_graph(graph)?;
    let (input, output) = ends(&info)?;
    let element_type_of = |zero_point: Option<&String>| -> Result<i64, String> {
        let name = zero_point.ok_or("a quantize node without a zero point")?;
        let stored = info
            .initializers
            .iter()
            .find(|tensor| tensor.name == *name)
            .ok_or(format!("the zero point `{name}` is not stored in the file"))?;
        match stored.element_type {
            ELEMENT_UINT8 | ELEMENT_INT8 => Ok(stored.element_type),
            other => Err(format!(
                "the zero point `{name}` has element type {other}, not 8-bit"
            )),
        }
    };

    // The node to remove at each end, and the tensor that takes the end's place.
    let mut removed: Vec<&NodeInfo> = Vec::new();
    let mut new_input = None;
    let mut new_output = None;
    if open.input {
        let readers = info.readers(&input.name);
        let [quantize] = readers.as_slice() else {
            return Err(format!(
                "the input `{}` is read by {} nodes, not by one QuantizeLinear",
                input.name,
                readers.len()
            ));
        };
        if quantize.op != "QuantizeLinear" || quantize.outputs.len() != 1 {
            return Err(format!(
                "the input `{}` is read by a {} node, not by a QuantizeLinear",
                input.name, quantize.op
            ));
        }
        new_input = Some(TensorInfo {
            name: quantize.outputs[0].clone(),
            element_type: element_type_of(quantize.inputs.get(2))?,
            shape: input.shape.clone(),
        });
        removed.push(quantize);
    }
    if open.output {
        let dequantize = info
            .producer(&output.name)
            .filter(|node| node.op == "DequantizeLinear")
            .ok_or(format!(
                "the output `{}` is not produced by a DequantizeLinear",
                output.name
            ))?;
        let source = dequantize
            .inputs
            .first()
            .ok_or("a DequantizeLinear without an input")?;
        if info.readers(source).len() != 1 {
            return Err(format!(
                "the tensor `{source}` is read by more than the output's DequantizeLinear"
            ));
        }
        new_output = Some(TensorInfo {
            name: source.clone(),
            element_type: element_type_of(dequantize.inputs.get(2))?,
            shape: output.shape.clone(),
        });
        removed.push(dequantize);
    }
    let now_ends: HashSet<&str> = [&new_input, &new_output]
        .into_iter()
        .flatten()
        .map(|tensor| tensor.name.as_str())
        .collect();

    let mut out = Vec::with_capacity(graph.len());
    for field in split(graph, "the graph")? {
        if field.wire != 2 {
            out.extend_from_slice(field.raw);
            continue;
        }
        match field.number {
            GRAPH_NODE => {
                let node = parse_node(field.payload)?;
                if !removed.iter().any(|gone| **gone == node) {
                    out.extend_from_slice(field.raw);
                }
            }
            GRAPH_INPUT => match &new_input {
                Some(tensor) if parse_value_info(field.payload)?.name == input.name => {
                    put_bytes_field(&mut out, GRAPH_INPUT, &value_info_bytes(tensor));
                }
                _ => out.extend_from_slice(field.raw),
            },
            GRAPH_OUTPUT => match &new_output {
                Some(tensor) => {
                    put_bytes_field(&mut out, GRAPH_OUTPUT, &value_info_bytes(tensor));
                }
                None => out.extend_from_slice(field.raw),
            },
            // A tensor that is now an input or the output is declared there, once.
            GRAPH_VALUE_INFO
                if now_ends.contains(parse_value_info(field.payload)?.name.as_str()) => {}
            _ => out.extend_from_slice(field.raw),
        }
    }
    with_graph(model, &out)
}

/// The same encoder with its texts laid along the height of one image instead of along
/// the batch: input `[1, channels, rows, length]`, output `[1, dimension, rows, 1]`.
/// Every convolution of the encoder has a kernel of height 1, so the texts stay apart;
/// what pooled and projected per batch row is rewritten to do so per image row:
///
/// - `GlobalAveragePool` becomes `AveragePool` with the kernel `[1, pooled length]`;
/// - `Flatten`, and the quantize pair behind it, is removed (nothing is left to flatten);
/// - `Gemm` (`y = x W^T + b`) becomes a `Conv` with a 1 x 1 kernel: its stored weight
///   keeps its bytes and gets the dimensions `[out, in, 1, 1]`.
///
/// The arithmetic of each text is that of the original graph. Whoever feeds the model
/// writes `x[0, c, n, t]` for text `n`, and reads `e[0, d, n, 0]`.
///
/// Refused unless the graph is the encoder's: operators of [`BATCH_FREE_OPERATORS`], one
/// `GlobalAveragePool` whose input length the file records, `Flatten` at axis 1, and a
/// `Gemm` with `transB = 1` and no scaling.
pub fn with_rows_as_height(model: &[u8], rows: u32) -> Result<Vec<u8>, L2dError> {
    height_edit(model, rows).map_err(refused)
}

/// An attribute holding a list of integers.
fn ints_attribute(name: &str, values: &[i64]) -> Vec<u8> {
    let mut attribute = Vec::new();
    put_bytes_field(&mut attribute, 1, name.as_bytes());
    for value in values {
        put_varint_field(&mut attribute, 8, *value as u64);
    }
    // AttributeProto.type = INTS.
    put_varint_field(&mut attribute, 20, 7);
    attribute
}

fn node_bytes(node: &NodeInfo, attributes: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for input in &node.inputs {
        put_bytes_field(&mut out, 1, input.as_bytes());
    }
    for output in &node.outputs {
        put_bytes_field(&mut out, 2, output.as_bytes());
    }
    put_bytes_field(&mut out, 3, node.name.as_bytes());
    put_bytes_field(&mut out, 4, node.op.as_bytes());
    for attribute in attributes {
        put_bytes_field(&mut out, 5, attribute);
    }
    out
}

/// A float attribute of a node, when it has one.
fn float_attribute(node: &[u8], name: &str) -> Result<Option<f32>, String> {
    for field in split(node, "a node")? {
        if field.number != 5 || field.wire != 2 {
            continue;
        }
        let mut found_name = false;
        let mut value = None;
        for part in split(field.payload, "an attribute")? {
            match (part.number, part.wire) {
                (1, 2) => found_name = part.payload == name.as_bytes(),
                (2, 5) => {
                    let bytes: [u8; 4] = part.raw[part.raw.len() - 4..].try_into().unwrap();
                    value = Some(f32::from_le_bytes(bytes));
                }
                _ => {}
            }
        }
        if found_name {
            return Ok(Some(value.unwrap_or(0.0)));
        }
    }
    Ok(None)
}

/// A stored tensor with other dimensions and the same bytes.
fn with_dims(tensor: &[u8], dims: &[i64]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(tensor.len() + 8);
    for dim in dims {
        put_varint_field(&mut out, 1, *dim as u64);
    }
    for field in split(tensor, "a stored tensor")? {
        if field.number != 1 {
            out.extend_from_slice(field.raw);
        }
    }
    Ok(out)
}

fn height_edit(model: &[u8], rows: u32) -> Result<Vec<u8>, String> {
    if rows == 0 {
        return Err("an image holds at least one text".into());
    }
    let graph = graph_of(model)?;
    let info = info_of_graph(graph)?;
    let (input, output) = ends(&info)?;
    if input.shape.len() != 4 || input.shape[2] != 1 || output.shape.len() != 2 {
        return Err(format!(
            "the input must be [batch, channels, 1, length] and the output [batch, \
             dimension]; they are {:?} and {:?}",
            input.shape, output.shape
        ));
    }
    let only = |op: &str| -> Result<&NodeInfo, String> {
        let found: Vec<&NodeInfo> = info.nodes.iter().filter(|node| node.op == op).collect();
        match found.as_slice() {
            [node] => Ok(*node),
            _ => Err(format!(
                "the graph must hold one {op} node, it holds {}",
                found.len()
            )),
        }
    };
    let pool = only("GlobalAveragePool")?;
    let flatten = only("Flatten")?;
    let gemm = only("Gemm")?;

    // The length the pool averages over: the recorded shape of the nearest tensor
    // before it (the quantize pair in between keeps the shape).
    let mut at = pool.inputs.first().cloned().unwrap_or_default();
    let pooled_length = loop {
        if let Some(known) = info.value_info.iter().find(|tensor| tensor.name == at) {
            match known.shape.as_slice() {
                [_, _, 1, length] if *length > 0 => break *length,
                other => return Err(format!("the pool's input has the shape {other:?}")),
            }
        }
        at = info
            .producer(&at)
            .filter(|node| matches!(node.op.as_str(), "QuantizeLinear" | "DequantizeLinear"))
            .and_then(|node| node.inputs.first().cloned())
            .ok_or("the file records no shape for the pool's input")?;
    };

    // Flatten goes, with the quantize pair behind it when there is one.
    let mut removed: Vec<&NodeInfo> = vec![flatten];
    let mut renamed = (flatten.outputs[0].clone(), flatten.inputs[0].clone());
    if let [quantize] = info.readers(&flatten.outputs[0]).as_slice()
        && quantize.op == "QuantizeLinear"
        && let [dequantize] = info.readers(&quantize.outputs[0]).as_slice()
        && dequantize.op == "DequantizeLinear"
    {
        removed.extend([*quantize, *dequantize]);
        renamed.0 = dequantize.outputs[0].clone();
    }
    if gemm.inputs.first() != Some(&renamed.0) || gemm.inputs.len() != 3 {
        return Err(
            "the Gemm node does not read the flattened tensor with a weight and a bias".into(),
        );
    }
    // The stored tensor behind the Gemm's weight.
    let weight = match info.producer(&gemm.inputs[1]) {
        Some(node) if node.op == "DequantizeLinear" => node.inputs[0].clone(),
        Some(node) => {
            return Err(format!(
                "the Gemm's weight is produced by a {} node",
                node.op
            ));
        }
        None => gemm.inputs[1].clone(),
    };
    let weight_dims = info
        .initializers
        .iter()
        .find(|tensor| tensor.name == weight)
        .map(|tensor| tensor.shape.clone())
        .ok_or(format!(
            "the Gemm's weight `{weight}` is not stored in the file"
        ))?;
    let [out_features, in_features] = weight_dims.as_slice() else {
        return Err(format!(
            "the Gemm's weight has the dimensions {weight_dims:?}"
        ));
    };

    let mut out = Vec::with_capacity(graph.len());
    for field in split(graph, "the graph")? {
        if field.wire != 2 {
            out.extend_from_slice(field.raw);
            continue;
        }
        match field.number {
            GRAPH_NODE => {
                let node = parse_node(field.payload)?;
                if !BATCH_FREE_OPERATORS.contains(&node.op.as_str()) {
                    return Err(format!("the operator `{}` is not the encoder's", node.op));
                }
                if removed.iter().any(|gone| **gone == node) {
                    if node.op == "Flatten"
                        && int_attribute(field.payload, "axis")?.unwrap_or(1) != 1
                    {
                        return Err("the Flatten node is not at axis 1".into());
                    }
                    continue;
                }
                if node == *pool {
                    let pooled = NodeInfo {
                        op: "AveragePool".into(),
                        ..node
                    };
                    // The strides are ONNX's default, written out: the provider's
                    // compiler refuses an average pool without them.
                    let attributes = [
                        ints_attribute("kernel_shape", &[1, pooled_length]),
                        ints_attribute("strides", &[1, 1]),
                    ];
                    put_bytes_field(&mut out, GRAPH_NODE, &node_bytes(&pooled, &attributes));
                } else if node == *gemm {
                    let scaled = |name: &str| float_attribute(field.payload, name);
                    if int_attribute(field.payload, "transB")? != Some(1)
                        || int_attribute(field.payload, "transA")?.unwrap_or(0) != 0
                        || scaled("alpha")?.unwrap_or(1.0) != 1.0
                        || scaled("beta")?.unwrap_or(1.0) != 1.0
                    {
                        return Err("the Gemm node is not y = x W^T + b".into());
                    }
                    let mut conv = NodeInfo {
                        op: "Conv".into(),
                        ..node
                    };
                    conv.inputs[0] = renamed.1.clone();
                    put_bytes_field(&mut out, GRAPH_NODE, &node_bytes(&conv, &[]));
                } else {
                    out.extend_from_slice(field.raw);
                }
            }
            GRAPH_INITIALIZER if parse_initializer(field.payload)?.name == weight => {
                let dims = [*out_features, *in_features, 1, 1];
                put_bytes_field(
                    &mut out,
                    GRAPH_INITIALIZER,
                    &with_dims(field.payload, &dims)?,
                );
            }
            GRAPH_INPUT if parse_value_info(field.payload)?.name == input.name => {
                let tensor = TensorInfo {
                    shape: vec![1, input.shape[1], i64::from(rows), input.shape[3]],
                    ..input.clone()
                };
                put_bytes_field(&mut out, GRAPH_INPUT, &value_info_bytes(&tensor));
            }
            GRAPH_OUTPUT => {
                let tensor = TensorInfo {
                    shape: vec![1, output.shape[1], i64::from(rows), 1],
                    ..output.clone()
                };
                put_bytes_field(&mut out, GRAPH_OUTPUT, &value_info_bytes(&tensor));
            }
            // Every recorded inner shape is the batch layout's; the runtime infers anew.
            GRAPH_VALUE_INFO => {}
            _ => out.extend_from_slice(field.raw),
        }
    }
    with_graph(model, &out)
}

/// The graph up to the inner tensor `tensor`, which becomes the one output: the nodes it
/// does not depend on and the stored tensors only they read are left out. The tensor's
/// type and shape are taken from the file's own record of it (`value_info`), or from
/// `declared` when the file has none.
pub fn with_output_at(
    model: &[u8],
    tensor: &str,
    declared: Option<&TensorInfo>,
) -> Result<Vec<u8>, L2dError> {
    cut_edit(model, tensor, declared).map_err(refused)
}

fn cut_edit(model: &[u8], tensor: &str, declared: Option<&TensorInfo>) -> Result<Vec<u8>, String> {
    let graph = graph_of(model)?;
    let info = info_of_graph(graph)?;
    let new_output = match info.value_info.iter().find(|known| known.name == tensor) {
        Some(known) => known.clone(),
        None => declared
            .filter(|given| given.name == tensor)
            .cloned()
            .ok_or(format!(
                "the file records no type and shape for the tensor `{tensor}`, and none was \
                 given"
            ))?,
    };
    if new_output.shape.is_empty() || new_output.shape.iter().any(|dim| *dim <= 0) {
        return Err(format!(
            "the tensor `{tensor}` has no fixed shape ({:?})",
            new_output.shape
        ));
    }
    // The nodes the tensor depends on, found backwards from it.
    let producer_of: HashMap<&str, usize> = info
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(index, node)| node.outputs.iter().map(move |name| (name.as_str(), index)))
        .collect();
    if !producer_of.contains_key(tensor) {
        return Err(format!("no node produces the tensor `{tensor}`"));
    }
    let mut kept = vec![false; info.nodes.len()];
    let mut wanted = vec![tensor];
    let mut read: HashSet<&str> = HashSet::new();
    while let Some(name) = wanted.pop() {
        if !read.insert(name) {
            continue;
        }
        if let Some(index) = producer_of.get(name) {
            kept[*index] = true;
            wanted.extend(info.nodes[*index].inputs.iter().map(String::as_str));
        }
    }
    let produced: HashSet<&str> = info
        .nodes
        .iter()
        .zip(&kept)
        .filter(|(_, keep)| **keep)
        .flat_map(|(node, _)| node.outputs.iter().map(String::as_str))
        .collect();

    let mut out = Vec::with_capacity(graph.len());
    let mut node_index = 0usize;
    let mut output_written = false;
    for field in split(graph, "the graph")? {
        if field.wire != 2 {
            out.extend_from_slice(field.raw);
            continue;
        }
        match field.number {
            GRAPH_NODE => {
                if kept[node_index] {
                    out.extend_from_slice(field.raw);
                }
                node_index += 1;
            }
            GRAPH_INITIALIZER => {
                if read.contains(parse_initializer(field.payload)?.name.as_str()) {
                    out.extend_from_slice(field.raw);
                }
            }
            GRAPH_OUTPUT => {
                if !output_written {
                    put_bytes_field(&mut out, GRAPH_OUTPUT, &value_info_bytes(&new_output));
                    output_written = true;
                }
            }
            GRAPH_VALUE_INFO => {
                let name = parse_value_info(field.payload)?.name;
                if name != tensor && produced.contains(name.as_str()) {
                    out.extend_from_slice(field.raw);
                }
            }
            _ => out.extend_from_slice(field.raw),
        }
    }
    if !output_written {
        put_bytes_field(&mut out, GRAPH_OUTPUT, &value_info_bytes(&new_output));
    }
    with_graph(model, &out)
}
