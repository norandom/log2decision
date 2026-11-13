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
//! - one `graph` in the file;
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
//!
//! # Memory
//!
//! The reader walks the graph several times and keeps, besides the bytes it is given,
//! only the graph's inputs and outputs: at most [`MAX_GRAPH_TENSORS`] of each with at
//! most [`MAX_TENSOR_DIMS`] dimensions (a graph with more is refused), under 200 KiB at
//! that ceiling and a few hundred bytes for an encoder model. Nodes and initializers are
//! looked at one at a time and not kept, so what it allocates does not grow with the
//! size of the file or the number of its nodes, and a refusal names at most
//! [`LISTED_NAMES`] of whatever it counts.

use std::collections::HashSet;

use l2d_core::L2dError;
use serde::{Deserialize, Serialize};

/// The element type every encoder input and output has.
pub const FLOAT32: &str = "float32";
/// The most graph inputs, and the most graph outputs, the reader accepts (old exporters
/// list every initializer as an input).
pub const MAX_GRAPH_TENSORS: usize = 256;
/// The most dimensions a graph input or output may have.
pub const MAX_TENSOR_DIMS: usize = 8;
/// The longest name a graph input, a graph output or a quantized tensor may have.
pub const MAX_TENSOR_NAME_BYTES: usize = 256;
/// How many names a refusal lists before it says how many more there are.
pub const LISTED_NAMES: usize = 3;

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

impl std::fmt::Display for TensorSpec {
    /// `` `x` float32 [32, 64, 1, 256] ``
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}` {} {:?}", self.name, self.element_type, self.shape)
    }
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

impl std::fmt::Display for QuantSpec {
    /// `` `x_q` uint8 with scale 0.015625 and zero point 128 ``
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "`{}` {} with scale {} and zero point {}",
            self.tensor, self.element_type, self.scale, self.zero_point
        )
    }
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
        let (input, output) = graph_tensors(graph)?;

        // The nodes, one at a time: who reads the input, who produces the output.
        let mut readers = Tally::default();
        let mut writers = Tally::default();
        for field in fields(graph, "the graph") {
            if let (1, Wire::Bytes(node)) = field? {
                let node = Node::parse(node, input.name, output.name)?;
                if node.reads {
                    readers.add(node);
                }
                if node.writes {
                    writers.add(node);
                }
            }
        }
        let quantize = match readers.only {
            Some(node) if node.op == "QuantizeLinear" && node.inputs[0] == input.name => node,
            _ => {
                return Err(format!(
                    "the input `{}` must be read by exactly one node, a QuantizeLinear, as in \
                     the 8-bit QDQ form; it is read by {} node(s) ({})",
                    input.name,
                    readers.count,
                    readers.listed()
                ));
            }
        };
        let dequantize = match writers.only {
            Some(node) if node.op == "DequantizeLinear" => node,
            _ => {
                return Err(format!(
                    "the output `{}` must be produced by exactly one node, a DequantizeLinear, \
                     as in the 8-bit QDQ form; it is produced by {} node(s) ({})",
                    output.name,
                    writers.count,
                    writers.listed()
                ));
            }
        };

        // The four scalars of the two nodes, found by name among the initializers.
        let wanted = [
            quantize.inputs[1],
            quantize.inputs[2],
            dequantize.inputs[1],
            dequantize.inputs[2],
        ];
        let mut found: [Option<Initializer<'_>>; 4] = [None, None, None, None];
        for field in fields(graph, "the graph") {
            if let (5, Wire::Bytes(tensor)) = field? {
                let (name, initializer) = Initializer::parse(tensor)?;
                // An initializer without a name cannot be meant by a node (an absent
                // node input is the empty name); it is passed over here as in the pass
                // over the graph inputs.
                if name.is_empty() {
                    continue;
                }
                for (slot, wanted) in found.iter_mut().zip(wanted) {
                    if wanted == name {
                        // Protobuf would let the last of two win; which one a runtime
                        // takes is not ours to guess.
                        if slot.is_some() {
                            return Err(format!(
                                "the initializer `{}`, a scale or zero point of the input or \
                                 the output, is defined more than once",
                                clipped(name)
                            ));
                        }
                        *slot = Some(initializer.clone());
                    }
                }
            }
        }
        let [in_scale, in_zero, out_scale, out_zero] = found;
        let input_quantization = QuantSpec::of(
            &quantize,
            quantize.output,
            in_scale.as_ref(),
            in_zero.as_ref(),
        )?;
        let output_quantization = QuantSpec::of(
            &dequantize,
            dequantize.inputs[0],
            out_scale.as_ref(),
            out_zero.as_ref(),
        )?;
        Ok(Self {
            input: input.spec()?,
            output: output.spec()?,
            input_quantization,
            output_quantization,
        })
    }
}

/// The one input and the one output of a float model file, to hold it to its 8-bit
/// model. What tells a float model from an 8-bit one here: no `QuantizeLinear` node
/// reads its input and no `DequantizeLinear` node produces its output. A file where one
/// does (the 8-bit model given a second time, for example) is refused.
pub(crate) fn float_tensors(bytes: &[u8]) -> Result<(TensorSpec, TensorSpec), String> {
    let graph = graph_of(bytes)?;
    let (input, output) = graph_tensors(graph)?;
    for field in fields(graph, "the graph") {
        if let (1, Wire::Bytes(node)) = field? {
            let node = Node::parse(node, input.name, output.name)?;
            if node.reads && node.op == "QuantizeLinear" {
                return Err(format!(
                    "its input `{}` is read by a QuantizeLinear node: this is a quantized \
                     model, not a float model",
                    clipped(input.name)
                ));
            }
            if node.writes && node.op == "DequantizeLinear" {
                return Err(format!(
                    "its output `{}` is produced by a DequantizeLinear node: this is a \
                     quantized model, not a float model",
                    clipped(output.name)
                ));
            }
        }
    }
    Ok((input.spec()?, output.spec()?))
}

impl QuantSpec {
    /// The scale and zero point of a `QuantizeLinear` or `DequantizeLinear` node: its
    /// second and third input, both scalar initializers.
    fn of(
        node: &Node<'_>,
        tensor: &str,
        scale: Option<&Initializer<'_>>,
        zero: Option<&Initializer<'_>>,
    ) -> Result<Self, String> {
        if tensor.is_empty() {
            return Err(format!("a {} node names no quantized tensor", node.op));
        }
        if tensor.len() > MAX_TENSOR_NAME_BYTES {
            return Err(format!(
                "the quantized tensor of a {} node has a name of {} bytes, more than \
                 {MAX_TENSOR_NAME_BYTES}",
                node.op,
                tensor.len()
            ));
        }
        let what = format!("the {} node at `{tensor}`", node.op);
        fn scalar<'i, 'a>(
            what: &str,
            name: &str,
            role: &str,
            found: Option<&'i Initializer<'a>>,
        ) -> Result<&'i Initializer<'a>, String> {
            if name.is_empty() {
                return Err(format!("{what} has no {role}"));
            }
            let name = clipped(name);
            let found = found.ok_or_else(|| {
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
        }

        let scale = scalar(&what, node.inputs[1], "scale", scale)?;
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

        let zero = scalar(&what, node.inputs[2], "zero point", zero)?;
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

/// The first [`LISTED_NAMES`] of `count` names, and how many more there are.
fn listed(names: &[&str], count: usize) -> String {
    if count == 0 {
        return "none".into();
    }
    let shown: Vec<String> = names
        .iter()
        .take(LISTED_NAMES.min(count))
        .map(|name| format!("`{}`", clipped(name)))
        .collect();
    let more = count - shown.len();
    if more == 0 {
        shown.join(", ")
    } else {
        format!("{} and {more} more", shown.join(", "))
    }
}

/// A name as a message shows it: its first 64 characters.
fn clipped(name: &str) -> &str {
    match name.char_indices().nth(64) {
        Some((end, _)) => &name[..end],
        None => name,
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

/// `ModelProto.graph` (field 7). Protobuf would merge a second `graph` field into the
/// first; a model file has one, and a file with two is refused rather than read as
/// either.
fn graph_of(model: &[u8]) -> Result<&[u8], String> {
    let mut graph = None;
    for field in fields(model, "the model") {
        if let (7, Wire::Bytes(bytes)) = field? {
            if graph.is_some() {
                return Err("it holds more than one graph".into());
            }
            graph = Some(bytes);
        }
    }
    graph.ok_or_else(|| "it holds no graph: not an ONNX model file".to_string())
}

/// The one graph input that is not an initializer and the one graph output
/// (`GraphProto`: initializer 5, input 11, output 12).
fn graph_tensors(graph: &[u8]) -> Result<(ValueInfo<'_>, ValueInfo<'_>), String> {
    let mut inputs: Vec<ValueInfo<'_>> = Vec::new();
    let mut outputs: Vec<ValueInfo<'_>> = Vec::new();
    for field in fields(graph, "the graph") {
        let (list, role, info) = match field? {
            (11, Wire::Bytes(info)) => (&mut inputs, "inputs", info),
            (12, Wire::Bytes(info)) => (&mut outputs, "outputs", info),
            _ => continue,
        };
        if list.len() == MAX_GRAPH_TENSORS {
            return Err(format!(
                "the graph lists more than {MAX_GRAPH_TENSORS} {role}; an encoder has one"
            ));
        }
        let info = ValueInfo::parse(info)?;
        // A tensor without a name cannot be fed, read or told from an absent node
        // input; it is refused, so that no pass has to give the empty name a meaning.
        if info.name.is_empty() {
            return Err(format!("one of the graph's {role} has no name"));
        }
        list.push(info);
    }
    // Old exporters list initializers among the inputs: those are not inputs. One
    // look-up per initializer, so the pass is linear in the file whatever the number of
    // initializers and of listed inputs.
    let listed_inputs: HashSet<&str> = inputs.iter().map(|input| input.name).collect();
    let mut initialized: HashSet<&str> = HashSet::new();
    for field in fields(graph, "the graph") {
        if let (5, Wire::Bytes(tensor)) = field? {
            let name = Initializer::name(tensor)?;
            if listed_inputs.contains(name) {
                initialized.insert(name);
            }
        }
    }
    let mut real = inputs
        .into_iter()
        .filter(|input| !initialized.contains(input.name));
    let (Some(input), None) = (real.next(), real.next()) else {
        let (names, count) = graph_tensor_names(graph, 11)?;
        return Err(format!(
            "the graph does not have exactly one input that is not an initializer (it lists \
             {count} inputs, initializers among them: {}); an encoder has one",
            listed(&names, count)
        ));
    };
    let mut outputs = outputs.into_iter();
    let (Some(output), None) = (outputs.next(), outputs.next()) else {
        let (names, count) = graph_tensor_names(graph, 12)?;
        return Err(format!(
            "the graph has {count} outputs ({}), an encoder has one",
            listed(&names, count)
        ));
    };
    Ok((input, output))
}

/// The first names of the graph's inputs (field 11) or outputs (field 12) and their
/// number, for a refusal.
fn graph_tensor_names(graph: &[u8], which: u64) -> Result<(Vec<&str>, usize), String> {
    let mut names = Vec::new();
    let mut count = 0;
    for field in fields(graph, "the graph") {
        if let (found, Wire::Bytes(info)) = field?
            && found == which
        {
            count += 1;
            if names.len() < LISTED_NAMES {
                names.push(ValueInfo::parse(info)?.name);
            }
        }
    }
    Ok((names, count))
}

/// What the reader needs of a node (`NodeProto`: input 1, output 2, op_type 4): its
/// operator, its first three inputs and its first output (an absent one is empty), and
/// whether it reads or produces the tensors asked about. Nothing is allocated.
#[derive(Clone, Copy)]
struct Node<'a> {
    op: &'a str,
    inputs: [&'a str; 3],
    output: &'a str,
    reads: bool,
    writes: bool,
}

impl<'a> Node<'a> {
    fn parse(node: &'a [u8], read: &str, written: &str) -> Result<Self, String> {
        let mut parsed = Node {
            op: "",
            inputs: [""; 3],
            output: "",
            reads: false,
            writes: false,
        };
        let (mut inputs, mut outputs) = (0usize, 0usize);
        for field in fields(node, "a node") {
            match field? {
                (1, Wire::Bytes(bytes)) => {
                    let name = text(bytes, "a node input")?;
                    if let Some(slot) = parsed.inputs.get_mut(inputs) {
                        *slot = name;
                    }
                    inputs += 1;
                    parsed.reads |= name == read;
                }
                (2, Wire::Bytes(bytes)) => {
                    let name = text(bytes, "a node output")?;
                    if outputs == 0 {
                        parsed.output = name;
                    }
                    outputs += 1;
                    parsed.writes |= name == written;
                }
                (4, Wire::Bytes(bytes)) => parsed.op = text(bytes, "a node's operator")?,
                _ => {}
            }
        }
        Ok(parsed)
    }
}

/// The nodes that read the input, or that produce the output: how many, the only one
/// when there is exactly one, and the operators of the first few for a refusal.
#[derive(Default)]
struct Tally<'a> {
    count: usize,
    only: Option<Node<'a>>,
    first_ops: [&'a str; LISTED_NAMES],
}

impl<'a> Tally<'a> {
    fn add(&mut self, node: Node<'a>) {
        if let Some(slot) = self.first_ops.get_mut(self.count) {
            *slot = node.op;
        }
        self.count += 1;
        self.only = (self.count == 1).then_some(node);
    }

    fn listed(&self) -> String {
        listed(&self.first_ops, self.count)
    }
}

/// An initializer, without its values: only what a scalar needs is kept.
#[derive(Clone)]
struct Initializer<'a> {
    data_type: i64,
    /// The product of the dimensions; none when it does not fit.
    elements: Option<u64>,
    raw: Option<&'a [u8]>,
    first_float: Option<f32>,
    first_int: Option<i64>,
    external: bool,
}

impl<'a> Initializer<'a> {
    /// Only the name (`TensorProto.name`, field 8).
    fn name(tensor: &'a [u8]) -> Result<&'a str, String> {
        let mut name = "";
        for field in fields(tensor, "an initializer") {
            if let (8, Wire::Bytes(bytes)) = field? {
                name = text(bytes, "an initializer's name")?;
            }
        }
        Ok(name)
    }

    /// `TensorProto`: dims 1, data_type 2, float_data 4, int32_data 5, name 8,
    /// raw_data 9, external_data 13, data_location 14. Repeated numbers come packed or
    /// one per field. Nothing is allocated.
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

enum Dim<'a> {
    Fixed(i64),
    Named(&'a str),
    Unknown,
}

struct ValueInfo<'a> {
    name: &'a str,
    /// None when the value is not a tensor.
    element_type: Option<i64>,
    /// None when the tensor has no shape.
    shape: Option<Vec<Dim<'a>>>,
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

    fn shape(shape: &'a [u8]) -> Result<Vec<Dim<'a>>, String> {
        let mut dims = Vec::new();
        for field in fields(shape, "a shape") {
            let (1, Wire::Bytes(one)) = field? else {
                continue;
            };
            if dims.len() == MAX_TENSOR_DIMS {
                return Err(format!(
                    "a graph input or output has more than {MAX_TENSOR_DIMS} dimensions"
                ));
            }
            let mut dim = Dim::Unknown;
            for field in fields(one, "a dimension") {
                match field? {
                    (1, Wire::Varint(value)) => dim = Dim::Fixed(value as i64),
                    (2, Wire::Bytes(name)) => {
                        dim = Dim::Named(text(name, "a dimension's name")?);
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
        let name = clipped(self.name);
        if self.name.len() > MAX_TENSOR_NAME_BYTES {
            return Err(format!(
                "the tensor `{name}`... has a name of {} bytes, more than \
                 {MAX_TENSOR_NAME_BYTES}",
                self.name.len()
            ));
        }
        let element_type = self
            .element_type
            .ok_or_else(|| format!("`{name}` is not a tensor"))?;
        let dims = self
            .shape
            .as_ref()
            .ok_or_else(|| format!("the tensor `{name}` has no shape"))?;
        let mut shape = Vec::with_capacity(dims.len());
        for (at, dim) in dims.iter().enumerate() {
            match dim {
                Dim::Fixed(value) if *value > 0 => shape.push(*value),
                Dim::Fixed(value) => {
                    return Err(format!(
                        "dimension {at} of the tensor `{name}` is {value}, not a positive number"
                    ));
                }
                Dim::Named(variable) => {
                    return Err(format!(
                        "dimension {at} of the tensor `{name}` is the variable `{}`; an encoder \
                         model has a fixed shape",
                        clipped(variable)
                    ));
                }
                Dim::Unknown => {
                    return Err(format!(
                        "dimension {at} of the tensor `{name}` is not stated; an encoder model \
                         has a fixed shape"
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
