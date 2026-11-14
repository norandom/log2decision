//! `EncoderSession`: log texts become vectors on the NPU (task 2.6; requirements 1.5,
//! 4.1, 4.3, 4.4, 4.6, 4.7).
//!
//! # NPU only
//!
//! The encoder runs on the NPU or not at all (owner, 2026-11-13). There is no CPU
//! provider run in this crate, no fallback and no flag that allows one. Opening fails
//! with a typed error that carries the reason ([`EncoderError`]) when
//!
//! - the runtime files are missing or are not the built release's
//!   ([`RuntimeFiles::check`], before anything is loaded);
//! - the host has no AMD NPU device, which is also what a missing driver looks like
//!   ([`NpuStatus::query`]);
//! - the package's model cannot be rewritten into the layout the NPU computes right
//!   ([`provider_model`]);
//! - the provider does not open the model, its placement report does not show that the
//!   model runs on the NPU ([`PlacementReport::npu_only`]), or the session's first run
//!   was not executed by the provider ([`ExecutedNodes::npu_only`]).
//!
//! The last two checks are what requirement 4.4 rests on. The provider option
//! `error_on_failure` does not make a session NPU only: when the provider's compiler
//! fails, the session opens all the same and runs every node on the CPU (task 2.12). So
//! the report the provider writes is read, and the session is refused unless the only
//! nodes on the CPU are the `QuantizeLinear` that reads the model input and the
//! `DequantizeLinear` that produces the model output.
//!
//! The report alone is not enough either. It is written when the provider compiles; a
//! start from the cache finds the report of that earlier compilation and the provider
//! does not write it again. With a `context.json` in the cache folder that the provider
//! could not read, it opened a session without an error that ran the model on the CPU,
//! seven times slower and with other vectors, beside a report that said 33 of 35 nodes
//! on the NPU (found in task 2.6). So every open also runs the model once, on an image
//! without a text, with ONNX Runtime's profiling on, and reads from the profile which
//! provider executed each node. Profiling ends with that run. Accepted are the
//! provider's own nodes of the parts it compiled (`vitis_ai_ep_<n>`, at least one) and
//! the quantize nodes at the two ends, through the provider or one each on ONNX
//! Runtime's CPU provider; every other node is refused.
//!
//! The session is also created with ONNX Runtime's `session.disable_cpu_ep_fallback=1`.
//! With AMD Ryzen AI 1.9.0 on the reference host the session on the NPU opens with it,
//! and the session of the unreadable `context.json` is then refused by ONNX Runtime
//! itself when it is created; the report and the profile are checked all the same.
//!
//! A model output that is all zeros or not finite for a text is an error of the run
//! ([`EncoderError::Failed`]), not a vector.
//!
//! # What runs
//!
//! The package's 8-bit model, verified again when it is read, rewritten in memory by
//! [`onnx_edit::with_rows_as_height`] so that the texts are the rows of one image
//! `[1, channels, 64, input_bytes]`, and handed to the provider from memory. In the
//! package's own layout (a batch along the first dimension) the NPU computes only the
//! first text of a batch right (task 2.12); in the image layout every row is
//! bit-identical to the text encoded alone, which is requirement 4.6 with tolerance
//! zero. The package is not changed and neither is its revision: the rows per image are
//! the session's choice. One session, [`ROWS_PER_IMAGE`] rows, float input.
//!
//! The provider's compiled model is cached below the project's `runtime/cache/` in a
//! folder named by [`encode::provider_cache_key`]: the revision, the runtime release,
//! the rewritten model bytes and the provider options. The provider looks at nothing but
//! that name, so everything that changes what it compiles is in it.
//!
//! # Memory
//!
//! A session holds fixed buffers, whatever the number and the length of the texts
//! ([`ImageEncoder::held_bytes`]; for architecture v1 at 64 rows 4,409,856 bytes): the
//! image (4,194,304 bytes), the byte table twice (65,536 bytes each: the codec's and a
//! copy in channel order), the bytes of one image's texts (16,384), a length and the
//! flags per row (512 and 2,048), the model's output (32,768) and one image's vectors
//! (32,768). What the runtime and the provider hold on top is theirs and is not
//! derived here: it is measured as the process figure in `tests/npu_session.rs`.
//! [`EncoderSession::encode`] returns vectors the caller owns (512 bytes of values per
//! text for dimension 128); [`EncoderSession::encode_into`] appends them to one buffer.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use l2d_core::{
    Device, EncoderIdentity, L2dError, Placement, PlacementSource, ProjectPaths, RunKind, RunRecord,
};
use ort::environment::Environment;
use ort::ep::{ArbitrarilyConfigurableExecutionProvider, Vitis};
use ort::session::Session;
use ort::value::TensorRef;

use crate::codec::{ByteCodec, InputMode, TextInput};
use crate::encode::{self, RowImage};
use crate::npu::{ComputeAccelerator, NpuMatch, NpuStatus};
use crate::onnx_edit;
use crate::package::EncoderPackage;
use crate::runtime::{CheckDepth, RuntimeFiles};

/// The rows of the image a session runs: 64 texts per run (task 2.12).
pub const ROWS_PER_IMAGE: u32 = 64;

/// The environment variable that makes the provider write its placement report, and the
/// file name it is set to. The report lands in the model's cache folder.
pub const REPORT_VARIABLE: &str = "XLNX_ONNX_EP_REPORT_FILE";
pub const REPORT_FILE: &str = "vitisai_ep_report.json";

/// The provider options a session sets besides the place of the cache, each with its
/// value. They are part of the cache key. `enable_cache_file_io_in_mem` = 0 keeps the
/// compiled model on disk, so that a later start does not compile.
pub const PROVIDER_OPTIONS: &[(&str, &str)] = &[("enable_cache_file_io_in_mem", "0")];

/// Why a session could not be opened or a run failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderError {
    /// A runtime file or the runtime directory is not there.
    RuntimeMissing { file: String, expected_at: PathBuf },
    /// The runtime directory holds something else than the built release's files.
    RuntimeRefused { reason: String },
    /// The NPU cannot be used: no device (or no driver), the provider did not open the
    /// model, or the model would not run on the NPU. Nothing is encoded.
    NpuUnavailable { reason: String },
    /// The package's model cannot be made into what the NPU is fed.
    ModelRefused { reason: String },
    /// The runtime failed.
    Failed { detail: String },
}

impl std::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderError::RuntimeMissing { file, expected_at } => write!(
                f,
                "the encoder cannot run: missing {file}: expected at {}",
                expected_at.display()
            ),
            EncoderError::RuntimeRefused { reason } => {
                write!(f, "the encoder cannot run: the runtime files: {reason}")
            }
            EncoderError::NpuUnavailable { reason } => write!(
                f,
                "the NPU cannot be used, nothing was encoded (the encoder runs on the NPU \
                 only): {reason}"
            ),
            EncoderError::ModelRefused { reason } => {
                write!(f, "the encoder cannot run: its model: {reason}")
            }
            EncoderError::Failed { detail } => write!(f, "the encoder failed: {detail}"),
        }
    }
}

impl std::error::Error for EncoderError {}

impl From<EncoderError> for L2dError {
    fn from(error: EncoderError) -> Self {
        match error {
            EncoderError::RuntimeMissing { file, expected_at } => L2dError::Missing {
                item: file,
                expected_at,
            },
            EncoderError::Failed { .. } => L2dError::Runtime {
                detail: error.to_string(),
            },
            refused => L2dError::Refused {
                reason: refused.to_string(),
            },
        }
    }
}

fn failed(what: &str, error: impl std::fmt::Display) -> EncoderError {
    EncoderError::Failed {
        detail: format!("{what}: {error}"),
    }
}

// ---------------------------------------------------------------------------------------
// The placement report
// ---------------------------------------------------------------------------------------

/// A node the provider left to the CPU, as its report names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportNode {
    pub operator: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// What the provider's report (`vitisai_ep_report.json`) says about where the nodes of a
/// model run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementReport {
    pub npu_nodes: u32,
    pub cpu_nodes: u32,
    pub all_nodes: u32,
    /// The nodes left to the CPU, in the report's order.
    pub cpu: Vec<ReportNode>,
    /// How many subgraphs the provider compiled for the NPU, when the report says.
    pub subgraphs: Option<usize>,
}

impl PlacementReport {
    /// Reads a report. Every entry of its `nodeStat` list must be on `NPU` or on `CPU`;
    /// anything else is an error that says what was found.
    pub fn parse(report: &str) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_str(report).map_err(|error| format!("report is not JSON: {error}"))?;
        let nodes = json
            .get("nodeStat")
            .and_then(|value| value.as_array())
            .ok_or("report has no `nodeStat` list")?;
        let mut placement = PlacementReport {
            npu_nodes: 0,
            cpu_nodes: 0,
            all_nodes: u32::try_from(nodes.len()).map_err(|_| "report lists too many nodes")?,
            cpu: Vec::new(),
            subgraphs: json
                .get("subgraphStat")
                .and_then(|value| value.as_array())
                .map(Vec::len),
        };
        let names = |node: &serde_json::Value, key: &str| -> Vec<String> {
            node.get(key)
                .and_then(|value| value.as_array())
                .map(|names| {
                    names
                        .iter()
                        .filter_map(|name| name.as_str())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default()
        };
        for (index, node) in nodes.iter().enumerate() {
            match node.get("device").and_then(|device| device.as_str()) {
                Some("NPU") => placement.npu_nodes += 1,
                Some("CPU") => {
                    placement.cpu_nodes += 1;
                    placement.cpu.push(ReportNode {
                        operator: node
                            .get("opType")
                            .and_then(|op| op.as_str())
                            .unwrap_or("?")
                            .to_string(),
                        inputs: names(node, "input"),
                        outputs: names(node, "output"),
                    });
                }
                other => return Err(format!("node {index} of `nodeStat` is on {other:?}")),
            }
        }
        Ok(placement)
    }

    /// The operators of the nodes left to the CPU, in the report's order.
    pub fn cpu_operators(&self) -> Vec<&str> {
        self.cpu.iter().map(|node| node.operator.as_str()).collect()
    }

    /// Whether the model runs on the NPU: the report covers the `nodes` nodes of the
    /// model that was handed to the provider, at least one node is on the NPU, and a
    /// node on the CPU is either the `QuantizeLinear` that reads the model input `input`
    /// or the `DequantizeLinear` that produces the model output `output`. Those two turn
    /// the float input into bytes and the output bytes into floats at the two ends of
    /// the graph; the provider leaves them to the CPU (33 of 35 nodes on the NPU for
    /// architecture v1). Anything else on the CPU is refused, with the reason.
    pub fn npu_only(&self, nodes: usize, input: &str, output: &str) -> Result<(), String> {
        if self.all_nodes as usize != nodes {
            return Err(format!(
                "the placement report lists {} nodes, the model handed to the provider has \
                 {nodes}: the report is not this model's",
                self.all_nodes
            ));
        }
        if self.npu_nodes == 0 {
            return Err(format!(
                "no node of the model is on the NPU: all {} are on the CPU (the provider did \
                 not compile the model)",
                self.cpu_nodes
            ));
        }
        let at_the_input = |node: &ReportNode| {
            node.operator == "QuantizeLinear" && node.inputs.first().is_some_and(|x| x == input)
        };
        let at_the_output = |node: &ReportNode| {
            node.operator == "DequantizeLinear" && node.outputs.iter().any(|y| y == output)
        };
        let elsewhere: Vec<String> = self
            .cpu
            .iter()
            .filter(|node| !at_the_input(node) && !at_the_output(node))
            .map(|node| {
                format!(
                    "{} (output `{}`)",
                    node.operator,
                    node.outputs.first().map_or("?", String::as_str)
                )
            })
            .collect();
        if !elsewhere.is_empty() {
            let listed: Vec<&str> = elsewhere.iter().take(5).map(String::as_str).collect();
            return Err(format!(
                "{} of {} nodes are on the CPU, and {} of them are not the quantize nodes at \
                 the model's input `{input}` and output `{output}`: {}{}",
                self.cpu_nodes,
                self.all_nodes,
                elsewhere.len(),
                listed.join(", "),
                if elsewhere.len() > listed.len() {
                    ", ..."
                } else {
                    ""
                }
            ));
        }
        Ok(())
    }

    /// The node counts in the run record's form.
    pub fn placement(&self) -> Placement {
        Placement {
            npu_nodes: self.npu_nodes,
            cpu_nodes: self.cpu_nodes,
            source: PlacementSource::ProviderReport,
        }
    }
}

/// The name ONNX Runtime gives the Vitis AI provider.
pub const VITIS_PROVIDER: &str = "VitisAIExecutionProvider";

/// The name ONNX Runtime gives its CPU provider. The session never registers it; ONNX
/// Runtime has it in every session, and a profile may attribute the quantize nodes at
/// the two ends of the graph to it.
pub const CPU_PROVIDER: &str = "CPUExecutionProvider";

/// How the Vitis AI provider names, in a profile, the node it made of a part of the
/// graph it compiled: this prefix and a number.
pub const COMPILED_PART_PREFIX: &str = "vitis_ai_ep_";

/// Which provider executed the nodes of one run, from ONNX Runtime's own profile of that
/// run. This is evidence of the session that is open, where the placement report is the
/// plan of the compilation that filled the cache folder, possibly an earlier one: with a
/// damaged `context.json` beside an intact report the provider opened a session that
/// ran every node on the CPU, without an error (found on the reference host, task 2.6).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExecutedNodes {
    /// The operators of the nodes executed through the Vitis AI provider, in the order
    /// of the run. The provider runs the part of the graph it compiled as one node.
    pub on_provider: Vec<String>,
    /// Nodes executed through another provider: operator and provider, in the order of
    /// the run.
    pub elsewhere: Vec<(String, String)>,
}

impl ExecutedNodes {
    /// Reads a profile: a JSON list of events, of which an event of the category `Node`
    /// whose name ends in `_kernel_time` is one execution of one node, with the provider
    /// and the operator in its `args`.
    pub fn parse(profile: &str) -> Result<Self, String> {
        let json: serde_json::Value = serde_json::from_str(profile)
            .map_err(|error| format!("profile is not JSON: {error}"))?;
        let events = json.as_array().ok_or("profile is not a list of events")?;
        let mut executed = ExecutedNodes::default();
        for event in events {
            let text = |key: &str| event.get(key).and_then(|value| value.as_str());
            let name = text("name").unwrap_or("");
            if text("cat") != Some("Node") || !name.ends_with("_kernel_time") {
                continue;
            }
            let arg = |key: &str| {
                event
                    .get("args")
                    .and_then(|args| args.get(key))
                    .and_then(|value| value.as_str())
            };
            let provider =
                arg("provider").ok_or(format!("node event `{name}` names no provider"))?;
            let operator = arg("op_name").unwrap_or("?").to_string();
            if provider == VITIS_PROVIDER {
                executed.on_provider.push(operator);
            } else {
                executed.elsewhere.push((operator, provider.to_string()));
            }
        }
        Ok(executed)
    }

    /// Whether the run was a run on the NPU. Only this is accepted:
    ///
    /// - through the Vitis AI provider: nodes the provider made of the parts of the
    ///   graph it compiled (named [`COMPILED_PART_PREFIX`]`<n>`), at least one of them,
    ///   and the quantize nodes at the two ends of the graph (`QuantizeLinear`,
    ///   `DequantizeLinear`);
    /// - through [`CPU_PROVIDER`]: at most one `QuantizeLinear` and at most one
    ///   `DequantizeLinear`.
    ///
    /// Every other node is refused and named in the reason: any other operator through
    /// the Vitis AI provider, any other or further node on the CPU provider, and any
    /// node on a provider that is neither of the two.
    ///
    /// On the reference host a session on the NPU shows three nodes through the
    /// provider (`QuantizeLinear`, `vitis_ai_ep_<n>`, `DequantizeLinear`) and none
    /// elsewhere; the session that fell back shows the two quantize nodes through the
    /// provider and nine nodes on the CPU provider (`Transpose`, `QLinearConv`, ...).
    pub fn npu_only(&self) -> Result<(), String> {
        let mut refused: Vec<String> = Vec::new();
        // The quantize nodes at the ends run through the provider also in a session
        // that fell back to the CPU, so they are accepted but are no evidence.
        let mut compiled = 0usize;
        for operator in &self.on_provider {
            if operator.starts_with(COMPILED_PART_PREFIX) {
                compiled += 1;
            } else if operator != "QuantizeLinear" && operator != "DequantizeLinear" {
                refused.push(format!("{operator} ({VITIS_PROVIDER})"));
            }
        }
        let (mut quantize, mut dequantize) = (0usize, 0usize);
        for (operator, provider) in &self.elsewhere {
            let seen = match operator.as_str() {
                "QuantizeLinear" if provider == CPU_PROVIDER => &mut quantize,
                "DequantizeLinear" if provider == CPU_PROVIDER => &mut dequantize,
                _ => {
                    refused.push(format!("{operator} ({provider})"));
                    continue;
                }
            };
            *seen += 1;
            if *seen > 1 {
                refused.push(format!("{operator} ({provider})"));
            }
        }
        if compiled > 0 && refused.is_empty() {
            return Ok(());
        }
        let listed = if refused.is_empty() {
            "no node of a compiled part was executed".to_string()
        } else {
            format!(
                "not accepted: {}{}",
                refused[..refused.len().min(6)].join(", "),
                if refused.len() > 6 { ", ..." } else { "" }
            )
        };
        Err(format!(
            "in a run of the open session ONNX Runtime executed {} node(s) through \
             {VITIS_PROVIDER}, {compiled} of them compiled parts \
             (`{COMPILED_PART_PREFIX}<n>`), and {} through other providers; {listed}",
            self.on_provider.len(),
            self.elsewhere.len(),
        ))
    }
}

// ---------------------------------------------------------------------------------------
// The model that is handed to the provider
// ---------------------------------------------------------------------------------------

/// The model file a session hands to the provider, with what names its compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderModel {
    /// The package's 8-bit model with the texts as the rows of one image.
    pub bytes: Vec<u8>,
    /// The name of the provider's cache folder ([`encode::provider_cache_key`]).
    pub cache_key: String,
    /// The name of the model input and of the model output.
    pub input: String,
    pub output: String,
    /// The nodes of the rewritten graph.
    pub nodes: usize,
    pub rows: u32,
}

/// Reads the package's model (its size and hash are checked again), rewrites it with
/// [`onnx_edit::with_rows_as_height`] for [`ROWS_PER_IMAGE`] rows and names its
/// compilation under the runtime release `release`. Nothing is loaded and nothing is
/// written. A model that cannot be rewritten, or whose rewritten form does not take the
/// image and return one vector per row, is [`EncoderError::ModelRefused`].
pub fn provider_model(pkg: &EncoderPackage, release: &str) -> Result<ProviderModel, EncoderError> {
    let refused = |reason: String| EncoderError::ModelRefused { reason };
    let model = pkg.read_model().map_err(|error| match error {
        L2dError::Refused { reason } => refused(reason),
        other => failed("reading the model of the package", other),
    })?;
    let rows = ROWS_PER_IMAGE;
    let bytes = onnx_edit::with_rows_as_height(&model, rows).map_err(|error| {
        refused(format!(
            "the model of {} cannot be rewritten so that the texts are the rows of one image, \
             the layout the NPU computes right: {error}",
            pkg.identity()
        ))
    })?;
    let graph = onnx_edit::graph_info(&bytes)
        .map_err(|error| refused(format!("the rewritten model cannot be read back: {error}")))?;
    let arch = pkg.arch();
    let image = [
        1,
        i64::from(arch.input.channels),
        i64::from(rows),
        i64::from(arch.input.input_bytes),
    ];
    let vectors = [1, i64::from(arch.dimension), i64::from(rows), 1];
    let (input, output) = match (&graph.inputs[..], &graph.outputs[..]) {
        ([input], [output]) if input.shape == image && output.shape == vectors => {
            (input.name.clone(), output.name.clone())
        }
        (inputs, outputs) => {
            let shapes = |tensors: &[onnx_edit::TensorInfo]| {
                tensors
                    .iter()
                    .map(|tensor| format!("`{}` {:?}", tensor.name, tensor.shape))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(refused(format!(
                "the rewritten model of {} must take one image {image:?} and return \
                 {vectors:?}; it takes {} and returns {}",
                pkg.identity(),
                shapes(inputs),
                shapes(outputs)
            )));
        }
    };
    Ok(ProviderModel {
        cache_key: encode::provider_cache_key(
            &pkg.identity().revision,
            release,
            &bytes,
            PROVIDER_OPTIONS,
        ),
        nodes: graph.nodes.len(),
        bytes,
        input,
        output,
        rows,
    })
}

// ---------------------------------------------------------------------------------------
// Encoding, without the device
// ---------------------------------------------------------------------------------------

/// Whatever computes the model on one image. The tool has one implementation, the
/// session on the NPU; the trait is the seam that lets everything around the model (the
/// images, the order, the flags, the last image that is not full, the normalisation) be
/// tested on a host without the device.
pub trait ImageModel {
    /// Runs the model on `image` (`[1, channels, rows, input_bytes]`) and writes its
    /// output `[1, dimension, rows, 1]` to `output`: `output[d * rows + n]` is element
    /// `d` of row `n`.
    fn run(&mut self, image: &RowImage, output: &mut [f32]) -> Result<(), EncoderError>;
}

/// One encoded text.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    /// The L2-normalized vector, of the encoder identity's dimension.
    pub vector: Vec<f32>,
    /// What the codec did with the text: both truncation flags of requirement 1.5
    /// ([`TextInput::truncated`], [`TextInput::text_longer_than_input`]) and whether the
    /// input was empty.
    pub input: TextInput,
}

/// What an encoder did so far.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeStats {
    /// Texts encoded.
    pub texts: u64,
    /// Images run; the last image of a call may hold fewer texts than rows.
    pub images: u64,
    /// Seconds inside the encoding calls, from the texts to the unit vectors in the
    /// caller's hands: the end-to-end time of the logs per second.
    pub seconds: f64,
    /// Of those: writing the images, the runs, and reading and normalising the output.
    pub write_seconds: f64,
    pub run_seconds: f64,
    pub read_seconds: f64,
}

impl EncodeStats {
    /// Texts per second of encoding, end to end; none before the first text.
    pub fn logs_per_second(&self) -> Option<f64> {
        (self.texts > 0 && self.seconds > 0.0).then(|| self.texts as f64 / self.seconds)
    }
}

/// Texts in, unit vectors out, around an [`ImageModel`]: the texts are written into the
/// one image buffer, [`RowImage::rows`] at a time, the model runs once per image, and
/// each row of its output becomes the vector of the row's text.
pub struct ImageEncoder<M> {
    model: M,
    image: RowImage,
    dimension: usize,
    /// The model's output of one image, `dimension * rows`.
    output: Box<[f32]>,
    /// The vectors of one image, `rows * dimension`.
    vectors: Box<[f32]>,
    stats: EncodeStats,
}

impl<M: ImageModel> ImageEncoder<M> {
    /// # Panics
    /// When `dimension` or `rows` is zero.
    pub fn new(codec: ByteCodec, dimension: usize, rows: usize, model: M) -> Self {
        assert!(dimension > 0, "a vector has at least one element");
        Self {
            model,
            image: RowImage::new(codec, rows),
            dimension,
            output: vec![0.0; dimension * rows].into_boxed_slice(),
            vectors: vec![0.0; rows * dimension].into_boxed_slice(),
            stats: EncodeStats::default(),
        }
    }

    pub fn rows(&self) -> usize {
        self.image.rows()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn codec(&self) -> &ByteCodec {
        self.image.codec()
    }

    pub fn stats(&self) -> &EncodeStats {
        &self.stats
    }

    /// The heap bytes the encoder's own buffers hold for as long as it lives, whatever
    /// the texts: the image buffer ([`RowImage::held_bytes`]), one image's output and one
    /// image's vectors. What the model holds is not counted.
    pub fn held_bytes(&self) -> u64 {
        self.image.held_bytes()
            + (std::mem::size_of_val(&*self.output) + std::mem::size_of_val(&*self.vectors)) as u64
    }

    /// Runs the model once on an image without a text and drops the output. Not counted
    /// in the statistics.
    fn probe(&mut self) -> Result<(), EncoderError> {
        self.image.fill(&[]);
        self.model.run(&self.image, &mut self.output)
    }

    /// Encodes `texts` and hands each text's unit vector and flags to `each`, in the
    /// order of the texts. When a run fails, or the model's output for a text is all
    /// zeros or not finite, the texts of the images before it were handed over already.
    pub fn encode_each(
        &mut self,
        texts: &[&str],
        mut each: impl FnMut(&[f32], TextInput),
    ) -> Result<(), EncoderError> {
        let started = Instant::now();
        let rows = self.image.rows();
        let mut outcome = Ok(());
        for (index, image) in texts.chunks(rows).enumerate() {
            let begun = Instant::now();
            self.image.fill(image);
            let written = Instant::now();
            self.stats.write_seconds += (written - begun).as_secs_f64();
            if let Err(error) = self.model.run(&self.image, &mut self.output) {
                outcome = Err(error);
                break;
            }
            let ran = Instant::now();
            self.stats.run_seconds += (ran - written).as_secs_f64();
            let vectors = &mut self.vectors[..image.len() * self.dimension];
            let no_unit_vector = encode::read_rows(&self.output, rows, image.len(), vectors);
            self.stats.read_seconds += ran.elapsed().as_secs_f64();
            // An output that is all zeros or not finite has no direction: it is an
            // error of the run, not a vector, and no text of its image is handed over.
            if let Some((row, length)) = no_unit_vector {
                outcome = Err(EncoderError::Failed {
                    detail: format!(
                        "the model's output for text {} of this call (from 0; row {row} of \
                         its image) has the length {length} and cannot be a unit vector",
                        index * rows + row
                    ),
                });
                break;
            }
            for (vector, flags) in vectors.chunks_exact(self.dimension).zip(self.image.flags()) {
                each(vector, *flags);
            }
            self.stats.images += 1;
            self.stats.texts += image.len() as u64;
        }
        self.stats.seconds += started.elapsed().as_secs_f64();
        outcome
    }

    /// One entry per text, in the order of the texts.
    pub fn encode(&mut self, texts: &[&str]) -> Result<Vec<Encoded>, EncoderError> {
        let mut encoded = Vec::with_capacity(texts.len());
        self.encode_each(texts, |vector, input| {
            encoded.push(Encoded {
                vector: vector.to_vec(),
                input,
            })
        })?;
        Ok(encoded)
    }

    /// Appends the vectors of `texts` to `vectors`, one after the other
    /// ([`ImageEncoder::dimension`] values per text), and their flags to `flags`. When a
    /// run fails, both are left as long as they were.
    pub fn encode_into(
        &mut self,
        texts: &[&str],
        vectors: &mut Vec<f32>,
        flags: &mut Vec<TextInput>,
    ) -> Result<(), EncoderError> {
        let (vectors_before, flags_before) = (vectors.len(), flags.len());
        vectors.reserve(texts.len() * self.dimension);
        flags.reserve(texts.len());
        let outcome = self.encode_each(texts, |vector, input| {
            vectors.extend_from_slice(vector);
            flags.push(input);
        });
        if outcome.is_err() {
            vectors.truncate(vectors_before);
            flags.truncate(flags_before);
        }
        outcome
    }
}

// ---------------------------------------------------------------------------------------
// The session on the NPU
// ---------------------------------------------------------------------------------------

/// The model on the Vitis AI provider.
struct NpuModel {
    session: Session,
    input: String,
}

impl ImageModel for NpuModel {
    fn run(&mut self, image: &RowImage, output: &mut [f32]) -> Result<(), EncoderError> {
        // The image is lent to the runtime for the run, not copied (task 2.12).
        let tensor = TensorRef::from_array_view((image.dims(), image.values()))
            .map_err(|error| failed("handing the image to the runtime", error))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input.as_str() => tensor])
            .map_err(|error| failed("the run on the NPU", error))?;
        let (_, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| failed("reading the model's output", error))?;
        if values.len() != output.len() {
            return Err(failed(
                "reading the model's output",
                format!("{} values, expected {}", values.len(), output.len()),
            ));
        }
        output.copy_from_slice(values);
        Ok(())
    }
}

/// What a session is: the encoder, the runtime, the device, the placement and how it
/// was opened. The run record is made from this and the [`EncodeStats`].
#[derive(Debug, Clone, PartialEq)]
pub struct SessionInfo {
    pub identity: EncoderIdentity,
    pub input_mode: InputMode,
    /// Texts per image.
    pub rows: u32,
    /// The AMD Ryzen AI release of the runtime files, its ONNX Runtime version, and the
    /// source whose manifest the directory matched (`wheel` or `nuget`). The source is
    /// what the directory's record says: the session checks the files at
    /// [`CheckDepth::Fast`].
    pub runtime_release: String,
    pub onnxruntime: String,
    pub runtime_source: String,
    pub runtime_dir: PathBuf,
    /// The device taken as the NPU and how it was matched.
    pub npu: ComputeAccelerator,
    pub npu_match: NpuMatch,
    /// The driver version of that device, when it could be read.
    pub driver_version: Option<String>,
    pub placement: PlacementReport,
    /// Which provider executed the nodes of the session's first run, by ONNX Runtime's
    /// profile of it.
    pub executed: ExecutedNodes,
    /// The provider's cache folder of the compiled model and its name.
    pub cache_key: String,
    pub cache_folder: PathBuf,
    /// Whether the provider compiled the model during this open (it wrote its report).
    pub compiled: bool,
    /// Seconds from the call of `open` to a session that can encode, and of those the
    /// load of the runtime library (zero when this process had loaded it) and the
    /// creation of the session on the provider.
    pub open_seconds: f64,
    pub load_seconds: f64,
    pub session_seconds: f64,
    /// When `open` was called, `YYYY-MM-DDTHH:MM:SSZ`.
    pub started_utc: String,
    /// The bytes the session's own buffers hold ([`ImageEncoder::held_bytes`]).
    pub held_bytes: u64,
}

impl SessionInfo {
    /// The run record of an encoding run (requirement 4.3): the encoder identity, the
    /// device (always the NPU) and how it was matched, the driver version, the runtime
    /// release and its source, the logs per second end to end and the placement node
    /// counts. `inputs` is empty: the caller knows the sources and adds them.
    ///
    /// The fields the record's shape has no place for are in `settings`: `npu` (the
    /// device and `matched_by`), `runtime` (`release`, `onnxruntime`, `source`),
    /// `session` (rows, cache key, whether the provider compiled, the open times, the
    /// buffers' bytes), `placement` (all nodes and the operators on the CPU) and
    /// `encoding` (texts, images and where the seconds went).
    pub fn run_record(&self, stats: &EncodeStats, duration_ms: u64) -> RunRecord {
        RunRecord {
            kind: RunKind::Encode,
            started_utc: self.started_utc.clone(),
            duration_ms,
            inputs: Vec::new(),
            settings: serde_json::json!({
                "input_mode": self.input_mode.as_str(),
                "npu": {
                    "instance_id": self.npu.instance_id,
                    "description": self.npu.description,
                    "matched_by": match self.npu_match {
                        NpuMatch::KnownDevice => "known_device_id",
                        NpuMatch::AmdVendorOnly => "amd_vendor_only",
                    },
                },
                "runtime": {
                    "release": self.runtime_release,
                    "onnxruntime": self.onnxruntime,
                    "source": self.runtime_source,
                },
                "session": {
                    "rows_per_image": self.rows,
                    "cache_key": self.cache_key,
                    "compiled_now": self.compiled,
                    "open_seconds": self.open_seconds,
                    "runtime_load_seconds": self.load_seconds,
                    "session_seconds": self.session_seconds,
                    "held_bytes": self.held_bytes,
                },
                "placement": {
                    "all_nodes": self.placement.all_nodes,
                    "cpu_operators": self.placement.cpu_operators(),
                    "first_run_nodes_on_provider": self.executed.on_provider,
                    "first_run_nodes_elsewhere": self
                        .executed
                        .elsewhere
                        .iter()
                        .map(|(operator, provider)| format!("{operator} ({provider})"))
                        .collect::<Vec<_>>(),
                },
                "encoding": {
                    "texts": stats.texts,
                    "images": stats.images,
                    "seconds": stats.seconds,
                    "write_seconds": stats.write_seconds,
                    "run_seconds": stats.run_seconds,
                    "read_seconds": stats.read_seconds,
                },
            }),
            seed: None,
            encoder: Some(self.identity.clone()),
            device: Some(Device::Npu),
            driver_version: self.driver_version.clone(),
            logs_per_second: stats.logs_per_second(),
            placement: Some(self.placement.placement()),
        }
    }
}

/// An encoder open on the NPU (see the module documentation).
pub struct EncoderSession {
    encoder: ImageEncoder<NpuModel>,
    info: SessionInfo,
    opened: Instant,
}

impl std::fmt::Debug for EncoderSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderSession")
            .field("info", &self.info)
            .field("stats", self.encoder.stats())
            .finish_non_exhaustive()
    }
}

impl EncoderSession {
    /// Opens `pkg` on the NPU with the runtime files of the project: the directory the
    /// environment variable `L2D_RUNTIME_DIR` names, or `runtime/ryzenai/`. A command
    /// with a directory option of its own calls [`EncoderSession::open_at`].
    pub fn open(pkg: &EncoderPackage, paths: &ProjectPaths) -> Result<Self, EncoderError> {
        let runtime_dir =
            paths.runtime_files_dir(l2d_core::paths::runtime_dir_from_environment().as_deref());
        Self::open_at(pkg, paths, &runtime_dir)
    }

    /// Opens `pkg` on the NPU with the runtime files in `runtime_dir`. In this order:
    /// the runtime files are checked (nothing is loaded before that: a missing DLL would
    /// end the process without an error), the NPU device is looked for, the model is
    /// read and rewritten, the runtime is loaded, the session is created on the Vitis AI
    /// provider with its cache below `runtime/cache/` of the project, and the placement
    /// report is read. Each step that fails is an error with the reason; a session is
    /// returned only when the model runs on the NPU.
    ///
    /// One process loads one runtime: a second directory in the same process is an
    /// error. The first open sets the environment variable [`REPORT_VARIABLE`] for the
    /// process.
    pub fn open_at(
        pkg: &EncoderPackage,
        paths: &ProjectPaths,
        runtime_dir: &Path,
    ) -> Result<Self, EncoderError> {
        let opened = Instant::now();
        let started_utc = utc_timestamp(SystemTime::now());
        let (files, npu, npu_match) = preconditions(
            RuntimeFiles::check(runtime_dir, CheckDepth::Fast),
            NpuStatus::query,
        )?;
        let model = provider_model(pkg, &files.release)?;

        let dll = plain(&files.onnxruntime_dll())?;
        let cache_dir = paths.runtime_cache_dir();
        std::fs::create_dir_all(&cache_dir)
            .map_err(|error| failed(&format!("creating {}", cache_dir.display()), error))?;
        let cache_dir = plain(&cache_dir)?;
        let cache_folder = cache_dir.join(&model.cache_key);
        let report_path = cache_folder.join(REPORT_FILE);
        let written = |path: &Path| {
            std::fs::metadata(path)
                .and_then(|meta| meta.modified())
                .ok()
        };
        let report_before = written(&report_path);

        let (environment, load_seconds) = runtime(&dll)?;
        let creating = Instant::now();
        let unavailable =
            |what: &str, error: &dyn std::fmt::Display| EncoderError::NpuUnavailable {
                reason: format!("{what}: {error}"),
            };
        let mut provider = Vitis::default()
            .with_cache_dir(cache_dir.display())
            .with_cache_key(&model.cache_key);
        for (name, value) in PROVIDER_OPTIONS {
            provider = provider.with_arbitrary_config(*name, *value);
        }
        // The first run of the session is profiled by ONNX Runtime (see below). The
        // runtime writes the profile to a file of this name plus a time stamp, beside
        // the cache folders; it is removed again, also when no session comes of it.
        static OPENS: AtomicU64 = AtomicU64::new(0);
        let profile_name = format!(
            "{}-first-run-{}-{}",
            model.cache_key,
            std::process::id(),
            OPENS.fetch_add(1, Ordering::Relaxed)
        );
        let profile_files = format!("{profile_name}_");
        let remove_profiles = || {
            for entry in std::fs::read_dir(&cache_dir)
                .into_iter()
                .flatten()
                .flatten()
            {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(&profile_files)
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        };
        let create = || {
            Session::builder(&environment)
                .map_err(|error| failed("creating the session builder", error))?
                .with_profiling(cache_dir.join(&profile_name))
                .map_err(|error| failed("enabling the profile of the first run", error))?
                // `session.disable_cpu_ep_fallback=1`: ONNX Runtime itself refuses to
                // create a session in which a node is left to its CPU provider.
                .with_disable_cpu_fallback()
                .map_err(|error| failed("disabling ONNX Runtime's CPU fallback", error))?
                .with_execution_providers([provider.build().error_on_failure()])
                .map_err(|error| {
                    unavailable("the Vitis AI provider could not be registered", &error)
                })?
                .commit_from_memory(&model.bytes)
                .map_err(|error| EncoderError::NpuUnavailable {
                    reason: format!(
                        "the Vitis AI provider did not open the model: {error} (its cache \
                         folder is {}; if that folder exists, it may be damaged: remove it so \
                         that the provider compiles the model again)",
                        cache_folder.display()
                    ),
                })
        };
        let session = create().inspect_err(|_| remove_profiles())?;
        let session_seconds = creating.elapsed().as_secs_f64();
        let report_after = written(&report_path);

        let mut encoder = ImageEncoder::new(
            pkg.codec().clone(),
            usize::from(pkg.arch().dimension),
            model.rows as usize,
            NpuModel {
                session,
                input: model.input.clone(),
            },
        );
        // Two pieces of evidence that the model runs on the NPU, and both must hold.
        //
        // The provider's report is the plan of a compilation: of this open when the
        // provider compiled now, of an earlier one when the compiled model came from the
        // cache (the provider does not write it again). So what this session does is
        // also asked of ONNX Runtime itself: one run of an image without a text is
        // profiled, the profile says which provider executed each node, and then
        // profiling is over for the life of the session. The profile is taken and its
        // file removed before anything is decided, so that no refusal leaves it behind.
        let probed = encoder.probe();
        let profile_path = encoder
            .model
            .session
            .end_profiling()
            .map_err(|error| failed("ending the profile of the first run", error))
            .inspect_err(|_| remove_profiles())?;
        let profile = std::fs::read_to_string(&profile_path);
        remove_profiles();

        let report = std::fs::read_to_string(&report_path).map_err(|error| {
            EncoderError::NpuUnavailable {
                reason: format!(
                    "the provider left no placement report at {} ({error}), so it is not known \
                     whether the model runs on the NPU; remove the folder {} so that the \
                     provider compiles the model again",
                    report_path.display(),
                    cache_folder.display()
                ),
            }
        })?;
        let placement = PlacementReport::parse(&report)
            .map_err(|error| failed(&format!("reading {}", report_path.display()), error))?;
        placement
            .npu_only(model.nodes, &model.input, &model.output)
            .map_err(|reason| EncoderError::NpuUnavailable {
                reason: format!(
                    "{reason} (placement report {}, encoder {})",
                    report_path.display(),
                    pkg.identity()
                ),
            })?;

        let not_on_the_npu = |reason: String| EncoderError::NpuUnavailable {
            reason: format!(
                "{reason}; the model of {} does not run on the NPU with the provider's cache \
                 folder {}: remove the folder so that the provider compiles the model again",
                pkg.identity(),
                cache_folder.display()
            ),
        };
        probed.map_err(|error| not_on_the_npu(format!("the first run failed ({error})")))?;
        let profile = profile
            .map_err(|error| failed(&format!("reading the profile {profile_path}"), error))?;
        let executed = ExecutedNodes::parse(&profile)
            .map_err(|error| failed(&format!("reading the profile {profile_path}"), error))?;
        executed.npu_only().map_err(not_on_the_npu)?;
        let info = SessionInfo {
            identity: pkg.identity().clone(),
            input_mode: pkg.input_mode(),
            rows: model.rows,
            runtime_release: files.release,
            onnxruntime: files.onnxruntime,
            runtime_source: files.source,
            runtime_dir: files.dir,
            driver_version: (npu.driver_version != crate::npu::UNKNOWN)
                .then(|| npu.driver_version.clone()),
            npu,
            npu_match,
            placement,
            executed,
            cache_key: model.cache_key,
            cache_folder,
            compiled: report_after.is_some() && report_after != report_before,
            open_seconds: opened.elapsed().as_secs_f64(),
            load_seconds,
            session_seconds,
            started_utc,
            held_bytes: encoder.held_bytes(),
        };
        Ok(Self {
            encoder,
            info,
            opened,
        })
    }

    /// Where the encoder runs: the NPU. A session that would run elsewhere is not opened.
    pub fn device(&self) -> Device {
        Device::Npu
    }

    /// The encoder, the runtime, the device and the placement of this session.
    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    /// Texts, images and seconds so far.
    pub fn stats(&self) -> &EncodeStats {
        self.encoder.stats()
    }

    /// One entry per text, in the order of the texts: the L2-normalized vector of the
    /// identity's dimension, and the text's flags. A text gets the same vector whatever
    /// its place among `texts` and whatever the other texts are (requirement 4.6).
    /// Callers pass distinct texts; the session does not deduplicate.
    pub fn encode(&mut self, texts: &[&str]) -> Result<Vec<Encoded>, EncoderError> {
        self.encoder.encode(texts)
    }

    /// As [`EncoderSession::encode`], with the vectors appended to one buffer
    /// (`dimension` values per text) and the flags to another, so that a caller that
    /// encodes many texts makes no allocation per text.
    pub fn encode_into(
        &mut self,
        texts: &[&str],
        vectors: &mut Vec<f32>,
        flags: &mut Vec<TextInput>,
    ) -> Result<(), EncoderError> {
        self.encoder.encode_into(texts, vectors, flags)
    }

    /// Ends the session and gives its run record ([`SessionInfo::run_record`]); the
    /// duration is the time since `open` was called.
    pub fn finish(self) -> RunRecord {
        let duration_ms = u64::try_from(self.opened.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.info.run_record(self.encoder.stats(), duration_ms)
    }
}

/// What must hold before the runtime is loaded: the runtime files passed their check
/// and the host has the NPU. The device is looked for only when the files are right.
fn preconditions(
    files: Result<RuntimeFiles, L2dError>,
    npu: impl FnOnce() -> Result<NpuStatus, L2dError>,
) -> Result<(RuntimeFiles, ComputeAccelerator, NpuMatch), EncoderError> {
    let files = files.map_err(|error| match error {
        L2dError::Missing { item, expected_at } => EncoderError::RuntimeMissing {
            file: item,
            expected_at,
        },
        L2dError::Refused { reason } => EncoderError::RuntimeRefused { reason },
        other => failed("checking the runtime files", other),
    })?;
    match npu().map_err(|error| failed("looking for the NPU", error))? {
        NpuStatus::Present { taken, matched, .. } => Ok((files, taken, matched)),
        absent @ NpuStatus::Absent { .. } => Err(EncoderError::NpuUnavailable {
            reason: absent.log_lines().join("; "),
        }),
    }
}

/// A path in the plain drive form the provider was proven with (no `\\?\` prefix).
fn plain(path: &Path) -> Result<PathBuf, EncoderError> {
    let resolved = path
        .canonicalize()
        .map_err(|error| failed(&format!("resolving {}", path.display()), error))?;
    Ok(PathBuf::from(
        resolved
            .to_string_lossy()
            .trim_start_matches(r"\\?\")
            .to_string(),
    ))
}

/// The runtime this process loaded, and from which library.
static RUNTIME: Mutex<Option<(PathBuf, Environment)>> = Mutex::new(None);

/// Loads ONNX Runtime from `dll`, once per process, and returns its environment with
/// the seconds the load took now.
fn runtime(dll: &Path) -> Result<(Environment, f64), EncoderError> {
    let mut loaded = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((from, environment)) = &*loaded {
        return if from == dll {
            Ok((environment.clone(), 0.0))
        } else {
            Err(EncoderError::Failed {
                detail: format!(
                    "this process loaded the runtime from {} already and cannot load another \
                     from {}",
                    from.display(),
                    dll.display()
                ),
            })
        };
    }
    let started = Instant::now();
    if std::env::var_os(REPORT_VARIABLE).is_none_or(|value| value != REPORT_FILE) {
        // SAFETY: the variable must be set before the provider compiles a model. The
        // provider runs on Windows only, where changing the environment is safe in a
        // process with several threads (the system call is synchronized and nothing
        // holds a pointer into the environment block); on other systems no session gets
        // this far, because no NPU device is found.
        unsafe { std::env::set_var(REPORT_VARIABLE, REPORT_FILE) };
    }
    let environment = ort::init_from(dll)
        .map_err(|error| failed(&format!("loading {}", dll.display()), error))?
        .with_name("l2d")
        .build()
        .map_err(|error| failed("creating the runtime environment", error))?;
    *loaded = Some((dll.to_path_buf(), environment.clone()));
    Ok((environment, started.elapsed().as_secs_f64()))
}

/// `time` as `YYYY-MM-DDTHH:MM:SSZ` (UTC, proleptic Gregorian calendar).
fn utc_timestamp(time: SystemTime) -> String {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0);
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Days since 1970-01-01 to a civil date (Howard Hinnant's `civil_from_days`).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> RuntimeFiles {
        RuntimeFiles {
            dir: PathBuf::from("runtime/ryzenai"),
            release: "1.9.0".into(),
            onnxruntime: "1.30.0".into(),
            library: "onnxruntime.dll".into(),
            source: "nuget".into(),
            record: crate::runtime::RecordState::Provisioning,
            refresh: crate::runtime::RecordRefresh::NotNeeded,
            hashed: Vec::new(),
        }
    }

    fn device(instance_id: &str) -> ComputeAccelerator {
        ComputeAccelerator {
            instance_id: instance_id.into(),
            description: "NPU Compute Accelerator Device".into(),
            driver_version: "32.0.203.329".into(),
        }
    }

    #[test]
    fn the_runtime_files_come_first_and_the_device_is_not_asked_without_them() {
        let asked = std::cell::Cell::new(false);
        let npu = || {
            asked.set(true);
            Ok(NpuStatus::from_devices(Vec::new()))
        };
        let error = preconditions(
            Err(L2dError::Missing {
                item: "runtime file dyn_dispatch_core.dll".into(),
                expected_at: PathBuf::from("r/dyn_dispatch_core.dll"),
            }),
            npu,
        )
        .unwrap_err();
        assert_eq!(
            error,
            EncoderError::RuntimeMissing {
                file: "runtime file dyn_dispatch_core.dll".into(),
                expected_at: PathBuf::from("r/dyn_dispatch_core.dll"),
            }
        );
        assert!(!asked.get(), "the device query ran without runtime files");

        let error = preconditions(
            Err(L2dError::Refused {
                reason: "onnxruntime.dll has 5 bytes".into(),
            }),
            || panic!("the device query ran"),
        )
        .unwrap_err();
        assert!(
            matches!(&error, EncoderError::RuntimeRefused { reason } if reason.contains("5 bytes"))
        );
    }

    #[test]
    fn a_host_without_the_npu_is_refused_with_the_reason() {
        // No device of the class at all: no NPU, or no driver.
        let error =
            preconditions(Ok(files()), || Ok(NpuStatus::from_devices(Vec::new()))).unwrap_err();
        let EncoderError::NpuUnavailable { reason } = &error else {
            panic!("{error:?}");
        };
        assert!(reason.contains("driver is not installed"), "{reason}");
        let shown = error.to_string();
        assert!(
            shown.contains("NPU only") && shown.contains("nothing was encoded"),
            "{shown}"
        );
        assert!(matches!(L2dError::from(error), L2dError::Refused { .. }));

        // A device of the class that is not AMD's.
        let other = device(r"PCI\VEN_8086&DEV_7D1D&SUBSYS_00000000&REV_04\3&11583659&0&58");
        let error =
            preconditions(Ok(files()), || Ok(NpuStatus::from_devices(vec![other]))).unwrap_err();
        assert!(
            matches!(&error, EncoderError::NpuUnavailable { reason } if reason.contains("AMD")),
            "{error:?}"
        );

        // The device query itself failed.
        let error = preconditions(Ok(files()), || {
            Err(L2dError::Runtime {
                detail: "CM_Get_Device_ID_List failed".into(),
            })
        })
        .unwrap_err();
        assert!(matches!(error, EncoderError::Failed { .. }));

        // The NPU is there: its device and how it was matched are handed on.
        let amd = device(r"PCI\VEN_1022&DEV_17F0&SUBSYS_17F01022&REV_11\4&2F53B1E2&0&0138");
        let (_, taken, matched) = preconditions(Ok(files()), || {
            Ok(NpuStatus::from_devices(vec![amd.clone()]))
        })
        .unwrap();
        assert_eq!((taken, matched), (amd, NpuMatch::KnownDevice));
    }

    #[test]
    fn a_time_is_written_as_a_utc_timestamp() {
        let at = |seconds: u64| utc_timestamp(UNIX_EPOCH + std::time::Duration::from_secs(seconds));
        assert_eq!(at(0), "1970-01-01T00:00:00Z");
        assert_eq!(at(951_782_399), "2000-02-28T23:59:59Z");
        assert_eq!(at(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(at(1_794_564_671), "2026-11-13T10:11:11Z");
        assert_eq!(at(4_107_542_399), "2100-02-28T23:59:59Z");
        assert_eq!(at(4_107_542_400), "2100-03-01T00:00:00Z");
    }

    #[test]
    fn every_error_keeps_its_kind_in_the_tool_wide_envelope() {
        let missing = EncoderError::RuntimeMissing {
            file: "runtime directory".into(),
            expected_at: PathBuf::from("x"),
        };
        assert_eq!(
            L2dError::from(missing).exit_code(),
            l2d_core::error::EXIT_MISSING
        );
        for refused in [
            EncoderError::RuntimeRefused { reason: "r".into() },
            EncoderError::NpuUnavailable { reason: "r".into() },
            EncoderError::ModelRefused { reason: "r".into() },
        ] {
            assert_eq!(
                L2dError::from(refused).exit_code(),
                l2d_core::error::EXIT_REFUSED
            );
        }
        let failure = EncoderError::Failed { detail: "d".into() };
        assert_eq!(
            L2dError::from(failure).exit_code(),
            l2d_core::error::EXIT_RUNTIME
        );
    }
}
