//! Shared by the tests of the encoder package (task 2.4): temporary projects and model
//! files small enough to write by hand.
//!
//! The model files are ONNX files in the shape the package asks for (one input that is
//! quantized first, one output that is dequantized last), written field by field in the
//! protobuf wire format. Nothing here runs a model.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::SystemTime;

use l2d_core::ProjectPaths;
use l2d_encoder::{EncoderArch, InputArch, InputMode, PackageParts};

static PROJECTS: AtomicU32 = AtomicU32::new(0);

/// A project root in the temporary directory, removed when dropped.
pub struct Project {
    pub root: PathBuf,
}

impl Project {
    pub fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "l2d-encoder-package-{tag}-{}-{}",
            std::process::id(),
            PROJECTS.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    pub fn paths(&self) -> ProjectPaths {
        ProjectPaths::new(&self.root)
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Every entry below `dir` with its kind, size and modification time, sorted.
pub fn snapshot(dir: &Path) -> Vec<(PathBuf, bool, u64, SystemTime)> {
    fn walk(dir: &Path, base: &Path, found: &mut Vec<(PathBuf, bool, u64, SystemTime)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let metadata = std::fs::symlink_metadata(entry.path()).unwrap();
            found.push((
                entry.path().strip_prefix(base).unwrap().to_path_buf(),
                metadata.is_dir(),
                if metadata.is_dir() { 0 } else { metadata.len() },
                metadata.modified().unwrap(),
            ));
            if metadata.is_dir() {
                walk(&entry.path(), base, found);
            }
        }
    }
    let mut found = Vec::new();
    if dir.is_dir() {
        walk(dir, dir, &mut found);
    }
    found.sort();
    found
}

/// Makes `link` a directory link to `target`: a symbolic link, or on Windows, where that
/// needs a privilege, a junction.
pub fn link_dir(link: &Path, target: &Path) {
    #[cfg(windows)]
    {
        if std::os::windows::fs::symlink_dir(target, link).is_ok() {
            return;
        }
        let made = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .unwrap();
        assert!(
            made.status.success(),
            "no directory link could be made: {}",
            String::from_utf8_lossy(&made.stderr)
        );
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, link).unwrap();
}

// ---------------------------------------------------------------------------------------
// Protobuf wire format, as far as a model file needs it.

pub fn varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let low = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(low);
            return;
        }
        out.push(low | 0x80);
    }
}

pub fn varint_field(out: &mut Vec<u8>, field: u64, value: u64) {
    varint(out, field << 3);
    varint(out, value);
}

pub fn bytes_field(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    varint(out, field << 3 | 2);
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// ONNX element types.
pub const FLOAT: u64 = 1;
pub const UINT8: u64 = 2;
pub const INT8: u64 = 3;
pub const DOUBLE: u64 = 11;

/// `ValueInfoProto`: a named tensor with a fixed shape.
pub fn value_info(name: &str, element_type: u64, shape: &[i64]) -> Vec<u8> {
    let mut dims = Vec::new();
    for &dim in shape {
        let mut one = Vec::new();
        varint_field(&mut one, 1, dim as u64);
        bytes_field(&mut dims, 1, &one);
    }
    let mut tensor = Vec::new();
    varint_field(&mut tensor, 1, element_type);
    bytes_field(&mut tensor, 2, &dims);
    let mut kind = Vec::new();
    bytes_field(&mut kind, 1, &tensor);
    let mut info = Vec::new();
    bytes_field(&mut info, 1, name.as_bytes());
    bytes_field(&mut info, 2, &kind);
    info
}

/// `NodeProto`.
pub fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
    let mut node = Vec::new();
    for input in inputs {
        bytes_field(&mut node, 1, input.as_bytes());
    }
    for output in outputs {
        bytes_field(&mut node, 2, output.as_bytes());
    }
    bytes_field(&mut node, 4, op.as_bytes());
    node
}

/// How a scalar initializer holds its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// In `raw_data`, as ONNX Runtime's quantizer writes it.
    Raw,
    /// In the typed fields `float_data` and `int32_data`, packed.
    Typed,
    /// In the typed fields, one element per field (not packed).
    TypedUnpacked,
}

/// `TensorProto` of a 32-bit float scalar.
pub fn float_scalar(name: &str, value: f32, storage: Storage) -> Vec<u8> {
    let mut tensor = Vec::new();
    varint_field(&mut tensor, 2, FLOAT);
    bytes_field(&mut tensor, 8, name.as_bytes());
    match storage {
        Storage::Raw => bytes_field(&mut tensor, 9, &value.to_le_bytes()),
        Storage::Typed => bytes_field(&mut tensor, 4, &value.to_le_bytes()),
        Storage::TypedUnpacked => {
            varint(&mut tensor, 4 << 3 | 5);
            tensor.extend_from_slice(&value.to_le_bytes());
        }
    }
    tensor
}

/// `TensorProto` of an 8-bit scalar (`element_type` [`UINT8`] or [`INT8`]).
pub fn byte_scalar(name: &str, element_type: u64, value: i32, storage: Storage) -> Vec<u8> {
    let mut tensor = Vec::new();
    varint_field(&mut tensor, 2, element_type);
    bytes_field(&mut tensor, 8, name.as_bytes());
    match storage {
        Storage::Raw => bytes_field(&mut tensor, 9, &[value as u8]),
        Storage::Typed => {
            let mut packed = Vec::new();
            varint(&mut packed, value as i64 as u64);
            bytes_field(&mut tensor, 5, &packed);
        }
        Storage::TypedUnpacked => varint_field(&mut tensor, 5, value as i64 as u64),
    }
    tensor
}

/// `TensorProto` of a float vector in `raw_data`: the weights that make a model large.
pub fn float_weights(name: &str, values: usize, seed: u8) -> Vec<u8> {
    let mut tensor = Vec::new();
    varint_field(&mut tensor, 1, values as u64);
    varint_field(&mut tensor, 2, FLOAT);
    bytes_field(&mut tensor, 8, name.as_bytes());
    let raw: Vec<u8> = (0..values * 4)
        .map(|at| (at as u8).wrapping_mul(31).wrapping_add(seed))
        .collect();
    bytes_field(&mut tensor, 9, &raw);
    tensor
}

/// A model file to be written by hand. The default is the 8-bit form of a tiny encoder:
/// input `x` float32 `[2, 3, 1, 8]`, output `e` float32 `[2, 4]`.
#[derive(Debug, Clone)]
pub struct Model {
    pub input_name: String,
    pub input_type: u64,
    pub input_shape: Vec<i64>,
    pub output_name: String,
    pub output_type: u64,
    pub output_shape: Vec<i64>,
    pub input_scale: f32,
    pub input_zero: i32,
    pub output_scale: f32,
    pub output_zero: i32,
    pub zero_type: u64,
    pub storage: Storage,
    /// Whether the input is quantized first and the output dequantized last.
    pub quantized: bool,
    /// Number of 32-bit weights.
    pub weights: usize,
    pub seed: u8,
}

impl Model {
    pub fn tiny() -> Self {
        Self {
            input_name: "x".into(),
            input_type: FLOAT,
            input_shape: vec![2, 3, 1, 8],
            output_name: "e".into(),
            output_type: FLOAT,
            output_shape: vec![2, 4],
            input_scale: 0.015_625,
            input_zero: 128,
            output_scale: 0.031_372_55,
            output_zero: 121,
            zero_type: UINT8,
            storage: Storage::Raw,
            quantized: true,
            weights: 64,
            seed: 0,
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        let (x, e) = (self.input_name.as_str(), self.output_name.as_str());
        let mut graph = Vec::new();
        let nodes = if self.quantized {
            vec![
                node("QuantizeLinear", &[x, "x_scale", "x_zero_point"], &["x_q"]),
                node(
                    "DequantizeLinear",
                    &["x_q", "x_scale", "x_zero_point"],
                    &["x_dq"],
                ),
                node("Conv", &["x_dq", "w"], &["h"]),
                node(
                    "QuantizeLinear",
                    &["h", "e_scale", "e_zero_point"],
                    &["e_q"],
                ),
                node(
                    "DequantizeLinear",
                    &["e_q", "e_scale", "e_zero_point"],
                    &[e],
                ),
            ]
        } else {
            vec![node("Conv", &[x, "w"], &[e])]
        };
        for one in &nodes {
            bytes_field(&mut graph, 1, one);
        }
        bytes_field(&mut graph, 2, b"tiny");
        let initializers = [
            float_scalar("x_scale", self.input_scale, self.storage),
            byte_scalar(
                "x_zero_point",
                self.zero_type,
                self.input_zero,
                self.storage,
            ),
            float_scalar("e_scale", self.output_scale, self.storage),
            byte_scalar(
                "e_zero_point",
                self.zero_type,
                self.output_zero,
                self.storage,
            ),
            float_weights("w", self.weights, self.seed),
        ];
        for one in &initializers {
            bytes_field(&mut graph, 5, one);
        }
        bytes_field(
            &mut graph,
            11,
            &value_info(x, self.input_type, &self.input_shape),
        );
        bytes_field(
            &mut graph,
            12,
            &value_info(e, self.output_type, &self.output_shape),
        );

        let mut opset = Vec::new();
        varint_field(&mut opset, 2, 17);
        let mut model = Vec::new();
        varint_field(&mut model, 1, 8);
        bytes_field(&mut model, 2, b"l2d-test");
        bytes_field(&mut model, 7, &graph);
        bytes_field(&mut model, 8, &opset);
        model
    }
}

/// The architecture of [`Model::tiny`].
pub fn tiny_arch() -> EncoderArch {
    EncoderArch {
        input: InputArch {
            input_bytes: 8,
            channels: 3,
            batch: 2,
        },
        dimension: 4,
    }
}

/// A byte table for `channels` values per byte: row 0 zeros, the rest distinct.
pub fn table_bytes(channels: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for byte in 0..256u32 {
        for channel in 0..channels {
            let value = if byte == 0 {
                0.0f32
            } else {
                (byte * channels + channel) as f32 / 1024.0 - 0.25
            };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// Everything a package is written from, owned.
#[derive(Debug, Clone)]
pub struct Parts {
    pub name: String,
    pub arch: EncoderArch,
    pub input_mode: InputMode,
    pub model_int8: Vec<u8>,
    pub table: Vec<u8>,
    pub training_record: Vec<u8>,
    pub model_f32: Option<Vec<u8>>,
}

impl Parts {
    /// The tiny encoder in `payload` mode, with a float model next to it.
    pub fn tiny() -> Self {
        Self {
            name: "tiny".into(),
            arch: tiny_arch(),
            input_mode: InputMode::Payload,
            model_int8: Model::tiny().bytes(),
            table: table_bytes(3),
            training_record: br#"{"kind":"train","seed":7}"#.to_vec(),
            model_f32: Some(
                Model {
                    quantized: false,
                    ..Model::tiny()
                }
                .bytes(),
            ),
        }
    }

    pub fn parts(&self) -> PackageParts<'_> {
        PackageParts {
            name: &self.name,
            arch: self.arch,
            input_mode: self.input_mode,
            model_int8: &self.model_int8,
            table: &self.table,
            training_record: &self.training_record,
            model_f32: self.model_f32.as_deref(),
        }
    }
}

/// A file of the crate's test fixtures.
pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
