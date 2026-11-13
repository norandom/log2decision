//! What a model file says about its own input and output (task 2.4).
//!
//! The encoder package records, for the session of task 2.6 and the tuning of task 2.12,
//! how the model is fed and read: the names, element types and shapes of its one input
//! and its one output, and the scale and zero point with which the 8-bit graph quantizes
//! the input and dequantizes the output. These facts are not typed in by whoever makes a
//! package: they are read from `model.int8.onnx` when the package is written and again
//! when it is opened, so `encoder.json` cannot say anything else than the model does.
//!
//! This is not an ONNX reader. It walks the protobuf wire format of the file as far as
//! these facts need it (`ModelProto.graph`, the graph's nodes, initializers, inputs and
//! outputs) and skips everything else without copying it: weights stay where they are.
//! Nothing is executed and nothing is written.
//!
//! # What a model must look like
//!
//! The 8-bit model in the QDQ form that ONNX Runtime's static quantization writes and
//! the design's quantizer is to write:
//! - exactly one graph input that is not an initializer and exactly one graph output,
//!   both tensors with a fixed shape (every dimension a positive number);
//! - the input is read by exactly one node, a `QuantizeLinear` whose scale and zero point
//!   are scalar initializers stored in the file;
//! - the output is produced by a `DequantizeLinear` with the same kind of scale and zero
//!   point.
//!
//! Anything else is refused with the reason. With scale `s` and zero point `z` of the
//! element type `T` (`uint8` or `int8`), the graph computes
//! `q = saturate_T(round_half_even(x / s) + z)` for the input and `e = (q - z) * s` for
//! the output.

use std::collections::HashMap;

use l2d_core::L2dError;
use serde::{Deserialize, Serialize};

/// The element type every encoder input and output has.
pub const FLOAT32: &str = "float32";

/// A graph input or output: its name, element type and fixed shape.
///
/// The element type is ONNX's, by its usual name (`float32`, `uint8`, `int8`, ...; a
/// type without a name here reads `onnx-type-<number>`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub name: String,
    pub element_type: String,
    pub shape: Vec<i64>,
}

/// How the graph quantizes its input or dequantizes its output (see the module
/// documentation for the formula).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantSpec {
    /// The name of the quantized tensor: what the `QuantizeLinear` node of the input
    /// produces, or what the `DequantizeLinear` node of the output reads.
    pub tensor: String,
    /// `uint8` or `int8`.
    pub element_type: String,
    /// A positive finite number.
    pub scale: f32,
    pub zero_point: i32,
}

/// The interface of an 8-bit encoder model, read from the model file.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInterface {
    pub input: TensorSpec,
    pub output: TensorSpec,
    pub input_quantization: QuantSpec,
    pub output_quantization: QuantSpec,
}

impl ModelInterface {
    /// Reads the interface from the bytes of a model file. Bytes that are not such a
    /// model (see the module documentation) are [`L2dError::Refused`] with the reason.
    pub fn from_onnx(bytes: &[u8]) -> Result<Self, L2dError> {
        Self::read(bytes).map_err(|reason| L2dError::Refused {
            reason: format!("the model is not usable as an 8-bit encoder model: {reason}"),
        })
    }

    /// As [`ModelInterface::from_onnx`], with the bare reason for the caller to place.
    pub(crate) fn read(bytes: &[u8]) -> Result<Self, String> {
        let graph = graph_of(bytes)?;
        let graph = Graph::parse(graph)?;

        let inputs: Vec<&ValueInfo<'_>> = graph
            .inputs
            .iter()
            .filter(|input| !graph.initializers.contains_key(input.name))
            .collect();
        let [input] = inputs.as_slice() else {
            return Err(format!(
                "the graph has {} inputs that are not initializers ({}), an encoder has one",
                inputs.len(),
                names(inputs.iter().map(|input| input.name))
            ));
        };
        let [output] = graph.outputs.as_slice() else {
            return Err(format!(
                "the graph has {} outputs ({}), an encoder has one",
                graph.outputs.len(),
                names(graph.outputs.iter().map(|output| output.name))
            ));
        };

        // The input: read by one node, a QuantizeLinear.
        let readers: Vec<&Node<'_>> = graph
            .nodes
            .iter()
            .filter(|node| node.inputs.contains(&input.name))
            .collect();
        let quantize = match readers.as_slice() {
            [node] if node.op == "QuantizeLinear" && node.inputs.first() == Some(&input.name) => {
                *node
            }
            _ => {
                return Err(format!(
                    "the input `{}` must be read by exactly one node, a QuantizeLinear, as in \
                     the 8-bit QDQ form; it is read by {} node(s) ({})",
                    input.name,
                    readers.len(),
                    names(readers.iter().map(|node| node.op))
                ));
            }
        };
        // The output: produced by a DequantizeLinear.
        let writers: Vec<&Node<'_>> = graph
            .nodes
            .iter()
            .filter(|node| node.outputs.contains(&output.name))
            .collect();
        let dequantize = match writers.as_slice() {
            [node] if node.op == "DequantizeLinear" => *node,
            _ => {
                return Err(format!(
                    "the output `{}` must be produced by exactly one node, a DequantizeLinear, \
                     as in the 8-bit QDQ form; it is produced by {} node(s) ({})",
                    output.name,
                    writers.len(),
                    names(writers.iter().map(|node| node.op))
                ));
            }
        };

        let input_quantization = QuantSpec::of(
            &graph,
            quantize,
            quantize.outputs.first().copied().unwrap_or(""),
        )?;
        let output_quantization = QuantSpec::of(
            &graph,
            dequantize,
            dequantize.inputs.first().copied().unwrap_or(""),
        )?;
        Ok(Self {
            input: input.spec()?,
            output: output.spec()?,
            input_quantization,
            output_quantization,
        })
    }
}

impl QuantSpec {
    /// The scale and zero point of a `QuantizeLinear` or `DequantizeLinear` node: its
    /// second and third input, both scalar initializers.
    fn of(graph: &Graph<'_>, node: &Node<'_>, tensor: &str) -> Result<Self, String> {
        let what = format!("the {} node at `{tensor}`", node.op);
        if tensor.is_empty() {
            return Err(format!("a {} node names no quantized tensor", node.op));
        }
        let initializer = |at: usize, role: &str| -> Result<&Initializer<'_>, String> {
            let name = node
                .inputs
                .get(at)
                .copied()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| format!("{what} has no {role}"))?;
            let found = graph.initializers.get(name).ok_or_else(|| {
                format!("{what}: its {role} `{name}` is not an initializer of the graph")
            })?;
            if found.external {
                return Err(format!(
                    "{what}: its {role} `{name}` is stored outside the model file"
                ));
            }
            if found.elements != Some(1) {
                return Err(format!(
                    "{what}: its {role} `{name}` is not a scalar (the quantization must be per \
                     tensor)"
                ));
            }
            Ok(found)
        };

        let scale = initializer(1, "scale")?;
        if scale.data_type != ONNX_FLOAT {
            return Err(format!(
                "{what}: its scale has the element type {}, not {FLOAT32}",
                element_type_name(scale.data_type)
            ));
        }
        let scale = match (scale.raw, scale.first_float) {
            (Some(raw), _) => <[u8; 4]>::try_from(raw).ok().map(f32::from_le_bytes),
            (None, first) => first,
        }
        .ok_or_else(|| format!("{what}: its scale holds no 32-bit value"))?;
        if !(scale.is_finite() && scale > 0.0) {
            return Err(format!(
                "{what}: its scale is {scale}, not a positive finite number"
            ));
        }

        let zero = initializer(2, "zero point")?;
        let signed = match zero.data_type {
            ONNX_UINT8 => false,
            ONNX_INT8 => true,
            other => {
                return Err(format!(
                    "{what}: its zero point has the element type {}, not uint8 or int8",
                    element_type_name(other)
                ));
            }
        };
        let zero_point = match (zero.raw, zero.first_int) {
            (Some([byte]), _) if signed => Some(i64::from(*byte as i8)),
            (Some([byte]), _) => Some(i64::from(*byte)),
            (Some(_), _) => None,
            (None, first) => first,
        }
        .ok_or_else(|| format!("{what}: its zero point holds no 8-bit value"))?;
        let range = if signed { -128..=127 } else { 0..=255 };
        if !range.contains(&zero_point) {
            return Err(format!(
                "{what}: its zero point {zero_point} is outside the range of {}",
                element_type_name(zero.data_type)
            ));
        }
        Ok(Self {
            tensor: tensor.to_string(),
            element_type: element_type_name(zero.data_type),
            scale,
            zero_point: zero_point as i32,
        })
    }
}

fn names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let listed: Vec<String> = names.map(|name| format!("`{name}`")).collect();
    if listed.is_empty() {
        "none".into()
    } else {
        listed.join(", ")
    }
}

const ONNX_FLOAT: i64 = 1;
const ONNX_UINT8: i64 = 2;
const ONNX_INT8: i64 = 3;

/// ONNX's `TensorProto.DataType` by its usual name.
fn element_type_name(data_type: i64) -> String {
    match data_type {
        1 => FLOAT32.into(),
        2 => "uint8".into(),
        3 => "int8".into(),
        4 => "uint16".into(),
        5 => "int16".into(),
        6 => "int32".into(),
        7 => "int64".into(),
        8 => "string".into(),
        9 => "bool".into(),
        10 => "float16".into(),
        11 => "float64".into(),
        12 => "uint32".into(),
        13 => "uint64".into(),
        other => format!("onnx-type-{other}"),
    }
}

// ---------------------------------------------------------------------------------------
// The protobuf wire format.

/// One field of a message: its payload by wire type.
enum Wire<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32([u8; 4]),
}

/// The fields of a message, in order. Every item borrows from the message.
struct Fields<'a> {
    rest: &'a [u8],
    what: &'static str,
}

fn fields<'a>(message: &'a [u8], what: &'static str) -> Fields<'a> {
    Fields {
        rest: message,
        what,
    }
}

fn varint(rest: &mut &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for (at, &byte) in rest.iter().enumerate().take(10) {
        value |= u64::from(byte & 0x7f) << (7 * at);
        if byte & 0x80 == 0 {
            *rest = &rest[at + 1..];
            return Some(value);
        }
    }
    None
}

fn take<'a>(rest: &mut &'a [u8], bytes: usize) -> Option<&'a [u8]> {
    let (taken, left) = rest.split_at_checked(bytes)?;
    *rest = left;
    Some(taken)
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<(u64, Wire<'a>), String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }
        let broken = || format!("it is not a model file or is cut short (in {})", self.what);
        let mut read = || -> Option<(u64, Wire<'a>)> {
            let key = varint(&mut self.rest)?;
            let payload = match key & 7 {
                0 => Wire::Varint(varint(&mut self.rest)?),
                1 => take(&mut self.rest, 8).map(|_| Wire::Fixed64)?,
                2 => {
                    let length = usize::try_from(varint(&mut self.rest)?).ok()?;
                    Wire::Bytes(take(&mut self.rest, length)?)
                }
                5 => Wire::Fixed32(take(&mut self.rest, 4)?.try_into().ok()?),
                _ => return None,
            };
            Some((key >> 3, payload))
        };
        match read() {
            Some(field) => Some(Ok(field)),
            None => {
                // Nothing after a broken field can be trusted.
                self.rest = &[];
                Some(Err(broken()))
            }
        }
    }
}

fn text<'a>(bytes: &'a [u8], what: &str) -> Result<&'a str, String> {
    std::str::from_utf8(bytes).map_err(|_| format!("{what} is not UTF-8 text"))
}

/// `ModelProto.graph` (field 7).
fn graph_of(model: &[u8]) -> Result<&[u8], String> {
    let mut graph = None;
    for field in fields(model, "the model") {
        if let (7, Wire::Bytes(bytes)) = field? {
            graph = Some(bytes);
        }
    }
    graph.ok_or_else(|| "it holds no graph: not an ONNX model file".to_string())
}

struct Node<'a> {
    op: &'a str,
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
}

/// An initializer, without its values: only what a scalar needs is kept.
struct Initializer<'a> {
    data_type: i64,
    /// The product of the dimensions; none when it does not fit.
    elements: Option<u64>,
    raw: Option<&'a [u8]>,
    first_float: Option<f32>,
    first_int: Option<i64>,
    external: bool,
}

enum Dim {
    Fixed(i64),
    Named(String),
    Unknown,
}

struct ValueInfo<'a> {
    name: &'a str,
    /// None when the value is not a tensor.
    element_type: Option<i64>,
    /// None when the tensor has no shape.
    shape: Option<Vec<Dim>>,
}

struct Graph<'a> {
    nodes: Vec<Node<'a>>,
    initializers: HashMap<&'a str, Initializer<'a>>,
    inputs: Vec<ValueInfo<'a>>,
    outputs: Vec<ValueInfo<'a>>,
}

impl<'a> Graph<'a> {
    /// `GraphProto`: node 1, initializer 5, input 11, output 12.
    fn parse(graph: &'a [u8]) -> Result<Self, String> {
        let mut parsed = Graph {
            nodes: Vec::new(),
            initializers: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        for field in fields(graph, "the graph") {
            match field? {
                (1, Wire::Bytes(bytes)) => parsed.nodes.push(Node::parse(bytes)?),
                (5, Wire::Bytes(bytes)) => {
                    let (name, initializer) = Initializer::parse(bytes)?;
                    parsed.initializers.insert(name, initializer);
                }
                (11, Wire::Bytes(bytes)) => parsed.inputs.push(ValueInfo::parse(bytes)?),
                (12, Wire::Bytes(bytes)) => parsed.outputs.push(ValueInfo::parse(bytes)?),
                _ => {}
            }
        }
        Ok(parsed)
    }
}

impl<'a> Node<'a> {
    /// `NodeProto`: input 1, output 2, op_type 4.
    fn parse(node: &'a [u8]) -> Result<Self, String> {
        let mut parsed = Node {
            op: "",
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        for field in fields(node, "a node") {
            match field? {
                (1, Wire::Bytes(bytes)) => parsed.inputs.push(text(bytes, "a node input")?),
                (2, Wire::Bytes(bytes)) => parsed.outputs.push(text(bytes, "a node output")?),
                (4, Wire::Bytes(bytes)) => parsed.op = text(bytes, "a node's operator")?,
                _ => {}
            }
        }
        Ok(parsed)
    }
}

impl<'a> Initializer<'a> {
    /// `TensorProto`: dims 1, data_type 2, float_data 4, int32_data 5, name 8,
    /// raw_data 9, external_data 13, data_location 14. Repeated numbers come packed or
    /// one per field.
    fn parse(tensor: &'a [u8]) -> Result<(&'a str, Self), String> {
        let mut name = "";
        let mut parsed = Initializer {
            data_type: 0,
            elements: Some(1),
            raw: None,
            first_float: None,
            first_int: None,
            external: false,
        };
        let dim = |parsed: &mut Initializer<'a>, dim: u64| {
            parsed.elements = parsed
                .elements
                .and_then(|elements| elements.checked_mul(dim));
        };
        for field in fields(tensor, "an initializer") {
            match field? {
                (1, Wire::Varint(value)) => dim(&mut parsed, value),
                (1, Wire::Bytes(mut packed)) => {
                    while !packed.is_empty() {
                        let value = varint(&mut packed)
                            .ok_or("the dimensions of an initializer are cut short")?;
                        dim(&mut parsed, value);
                    }
                }
                (2, Wire::Varint(value)) => parsed.data_type = value as i64,
                (4, Wire::Fixed32(value)) if parsed.first_float.is_none() => {
                    parsed.first_float = Some(f32::from_le_bytes(value));
                }
                (4, Wire::Bytes(packed)) if parsed.first_float.is_none() => {
                    parsed.first_float = packed
                        .first_chunk::<4>()
                        .map(|value| f32::from_le_bytes(*value));
                }
                (5, Wire::Varint(value)) if parsed.first_int.is_none() => {
                    parsed.first_int = Some(i64::from(value as i32));
                }
                (5, Wire::Bytes(mut packed)) if parsed.first_int.is_none() => {
                    parsed.first_int = varint(&mut packed).map(|value| i64::from(value as i32));
                }
                (8, Wire::Bytes(bytes)) => name = text(bytes, "an initializer's name")?,
                (9, Wire::Bytes(bytes)) => parsed.raw = Some(bytes),
                (13, _) => parsed.external = true,
                (14, Wire::Varint(location)) if location != 0 => parsed.external = true,
                _ => {}
            }
        }
        Ok((name, parsed))
    }
}

impl<'a> ValueInfo<'a> {
    /// `ValueInfoProto`: name 1, type 2; `TypeProto.tensor_type` 1 with elem_type 1 and
    /// shape 2; `TensorShapeProto.dim` 1 with dim_value 1 or dim_param 2.
    fn parse(info: &'a [u8]) -> Result<Self, String> {
        let mut parsed = ValueInfo {
            name: "",
            element_type: None,
            shape: None,
        };
        for field in fields(info, "a graph input or output") {
            match field? {
                (1, Wire::Bytes(bytes)) => parsed.name = text(bytes, "a tensor name")?,
                (2, Wire::Bytes(kind)) => {
                    for field in fields(kind, "a type") {
                        let (1, Wire::Bytes(tensor)) = field? else {
                            continue;
                        };
                        // A tensor type: the element type is there even when it is not
                        // written (protobuf leaves out a zero).
                        parsed.element_type = Some(0);
                        for field in fields(tensor, "a tensor type") {
                            match field? {
                                (1, Wire::Varint(value)) => {
                                    parsed.element_type = Some(value as i64);
                                }
                                (2, Wire::Bytes(shape)) => {
                                    parsed.shape = Some(Self::shape(shape)?);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(parsed)
    }

    fn shape(shape: &[u8]) -> Result<Vec<Dim>, String> {
        let mut dims = Vec::new();
        for field in fields(shape, "a shape") {
            let (1, Wire::Bytes(one)) = field? else {
                continue;
            };
            let mut dim = Dim::Unknown;
            for field in fields(one, "a dimension") {
                match field? {
                    (1, Wire::Varint(value)) => dim = Dim::Fixed(value as i64),
                    (2, Wire::Bytes(name)) => {
                        dim = Dim::Named(text(name, "a dimension's name")?.to_string());
                    }
                    _ => {}
                }
            }
            dims.push(dim);
        }
        Ok(dims)
    }

    /// The tensor as the package records it; a tensor without a fixed shape is refused.
    fn spec(&self) -> Result<TensorSpec, String> {
        let element_type = self
            .element_type
            .ok_or_else(|| format!("`{}` is not a tensor", self.name))?;
        let dims = self
            .shape
            .as_ref()
            .ok_or_else(|| format!("the tensor `{}` has no shape", self.name))?;
        let mut shape = Vec::with_capacity(dims.len());
        for (at, dim) in dims.iter().enumerate() {
            match dim {
                Dim::Fixed(value) if *value > 0 => shape.push(*value),
                Dim::Fixed(value) => {
                    return Err(format!(
                        "dimension {at} of the tensor `{}` is {value}, not a positive number",
                        self.name
                    ));
                }
                Dim::Named(name) => {
                    return Err(format!(
                        "dimension {at} of the tensor `{}` is the variable `{name}`; an encoder \
                         model has a fixed shape",
                        self.name
                    ));
                }
                Dim::Unknown => {
                    return Err(format!(
                        "dimension {at} of the tensor `{}` is not stated; an encoder model has \
                         a fixed shape",
                        self.name
                    ));
                }
            }
        }
        Ok(TensorSpec {
            name: self.name.to_string(),
            element_type: element_type_name(element_type),
            shape,
        })
    }
}
