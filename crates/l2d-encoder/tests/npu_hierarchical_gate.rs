//! Hardware part of the gate of task 2.16: does the hierarchical layout (bytes, then
//! patches, then one vector per log line) run on the NPU, with every row right, and how
//! fast?
//!
//! The layout is two fixed-shape 8-bit graphs with plain arithmetic between them, the
//! texts being the rows of one image in both (task 2.12):
//!
//! 1. byte graph: `[1, 64, rows, 1024]` (the byte table's rows of each line) to the byte
//!    features `h` `[1, 96, rows, 1024]` and, for the entropy patcher, the next-byte
//!    logits `y` `[1, 256, rows, 1024]`;
//! 2. here, on the CPU, not as a model run: the entropy of each next-byte prediction,
//!    the patch boundaries, and the mean of the byte features of each patch;
//! 3. line graph: `[1, 96, rows, 128]` (the patch features of each line) to
//!    `[1, 128, rows, 1]`; the vector is L2-normalised here.
//!
//! The graphs, the byte table, the texts and the vectors of the Python 8-bit reference
//! are written by `tools/science/hg_run.py npu-fixture`; this file is a spike's test and
//! not a part of the tool. It follows the evidence rule of the session (task 2.6): a
//! graph counts as running on the NPU only when the provider's placement report and
//! ONNX Runtime's profile of a first run both say so; there is no CPU provider.
//!
//! Run on request, in release, one test at a time:
//!
//! ```text
//! $env:L2D_RUNTIME_DIR = "D:\Source\log2decision\runtime\ryzenai"
//! $env:L2D_NPU_WORK_DIR = "D:\BuildArtifacts\log2decision\task-2.16\npu-work"
//! $env:L2D_HG_FIXTURE = "D:\BuildArtifacts\log2decision\task-2.16\npu\<variant>"
//! cargo test --release -p l2d-encoder --test npu_hierarchical_gate -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(windows)]

mod npu_support;

use std::path::{Path, PathBuf};
use std::time::Instant;

use l2d_encoder::encode::{l2_normalize, provider_cache_key};
use l2d_encoder::onnx_edit;
use l2d_encoder::session::{ExecutedNodes, PROVIDER_OPTIONS, PlacementReport};
use npu_support::*;
use ort::ep::{ArbitrarilyConfigurableExecutionProvider, Vitis};
use ort::session::Session;
use ort::value::TensorRef;

const TAG: &str = "npu_hierarchical_gate";

// ---------------------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Layout {
    variant: String,
    patcher: String,
    setting: f64,
    input_bytes: usize,
    patch_limit: usize,
    max_patch: usize,
    stride: usize,
    delimiter: [bool; 256],
    table_width: usize,
    features: usize,
    dimension: usize,
    byte_rows: usize,
    patch_rows: usize,
    csic_texts: usize,
    /// Step of the 8-bit logits, when the byte graph gives them.
    logit_scale: Option<f32>,
}

struct Fixture {
    rig: Rig,
    dir: PathBuf,
    layout: Layout,
    table: Vec<f32>,
    byte_model: Vec<u8>,
    line_model: Vec<u8>,
    texts: Vec<Vec<u8>>,
}

/// Little-endian 32-bit floats of a file.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect()
}

fn fixture() -> Option<Fixture> {
    let rig = Rig::prepare(TAG)?;
    let dir = directory_from(
        TAG,
        "L2D_HG_FIXTURE",
        "the fixture of `hg_run.py npu-fixture`",
    )?;
    let json: serde_json::Value =
        serde_json::from_slice(&read(&dir.join("layout.json"))).expect("layout.json");
    let number = |key: &str| json[key].as_u64().unwrap_or_else(|| panic!("layout.{key}")) as usize;
    let mut delimiter = [false; 256];
    for byte in json["delimiters"]
        .as_str()
        .expect("layout.delimiters")
        .bytes()
    {
        delimiter[usize::from(byte)] = true;
    }
    let layout = Layout {
        variant: json["variant"]
            .as_str()
            .expect("layout.variant")
            .to_string(),
        patcher: json["patcher"]
            .as_str()
            .expect("layout.patcher")
            .to_string(),
        setting: json["patcher_setting"].as_f64().unwrap_or(0.0),
        input_bytes: number("input_bytes"),
        patch_limit: number("patch_limit"),
        max_patch: number("max_patch_bytes"),
        stride: number("stride"),
        delimiter,
        table_width: number("table_width"),
        features: number("features"),
        dimension: number("dimension"),
        byte_rows: number("byte_rows"),
        patch_rows: number("patch_rows"),
        csic_texts: number("csic_texts"),
        logit_scale: json["logit_scale"].as_f64().map(|scale| scale as f32),
    };
    let table = floats(&read(&dir.join("table.bin")));
    assert_eq!(table.len(), 256 * layout.table_width);
    let texts: Vec<Vec<u8>> = String::from_utf8(read(&dir.join("texts.jsonl")))
        .expect("texts.jsonl is UTF-8")
        .lines()
        .map(|line| {
            let text: String = serde_json::from_str(line).expect("a JSON string per line");
            let mut bytes = text.into_bytes();
            // Requirement 1.5: a line longer than the layout reads is cut at the byte.
            bytes.truncate(layout.input_bytes);
            bytes
        })
        .collect();
    assert_eq!(texts.len(), number("texts"));
    println!(
        "{TAG}: variant {}, patcher {} (setting {}), {} texts ({} of CSIC), byte graph {} rows \
         of {} bytes, line graph {} rows of {} patches",
        layout.variant,
        layout.patcher,
        layout.setting,
        texts.len(),
        layout.csic_texts,
        layout.byte_rows,
        layout.input_bytes,
        layout.patch_rows,
        layout.patch_limit
    );
    Some(Fixture {
        byte_model: read(&dir.join("byte.int8.onnx")),
        line_model: read(&dir.join("line.int8.onnx")),
        rig,
        dir,
        layout,
        table,
        texts,
    })
}

// ---------------------------------------------------------------------------------------
// A graph on the NPU, by evidence
// ---------------------------------------------------------------------------------------

struct Graph {
    session: Session,
    input: String,
    dims: Vec<i64>,
    outputs: Vec<String>,
    placement: PlacementReport,
    executed: ExecutedNodes,
    compiled: bool,
    seconds: f64,
}

/// Opens `model` on the Vitis AI provider, without a CPU fallback, and holds it to the
/// rule of the session of task 2.6: the provider's report must place every node on the
/// NPU except the quantize node at the input and the dequantize nodes at the outputs,
/// and ONNX Runtime's profile of a first run must show the compiled part executed
/// through the provider and nothing but the end nodes elsewhere.
fn open_graph(rig: &Rig, name: &str, model: &[u8], fresh: bool) -> Result<Graph, String> {
    let info = onnx_edit::graph_info(model).map_err(|error| error.to_string())?;
    let input = info.inputs.first().ok_or("the model has no input")?.clone();
    let outputs: Vec<String> = info
        .outputs
        .iter()
        .map(|output| output.name.clone())
        .collect();
    let key = provider_cache_key(
        &format!("hg216{name}"),
        &rig.release,
        model,
        PROVIDER_OPTIONS,
    );
    let folder = rig.cache_dir.join(&key);
    if fresh && folder.is_dir() {
        std::fs::remove_dir_all(&folder)
            .map_err(|error| format!("{}: {error}", folder.display()))?;
    }
    let report_path = folder.join(REPORT_FILE);
    let written = |path: &Path| {
        std::fs::metadata(path)
            .and_then(|meta| meta.modified())
            .ok()
    };
    let before = written(&report_path);

    let started = Instant::now();
    let mut provider = Vitis::default()
        .with_cache_dir(rig.cache_dir.display())
        .with_cache_key(&key);
    for (option, value) in PROVIDER_OPTIONS {
        provider = provider.with_arbitrary_config(*option, *value);
    }
    let profile_name = format!("{key}-first-run-{}", std::process::id());
    let mut session = Session::builder(rig.environment)
        .map_err(|error| format!("creating the session builder: {error}"))?
        .with_profiling(rig.cache_dir.join(&profile_name))
        .map_err(|error| format!("enabling the profile: {error}"))?
        .with_disable_cpu_fallback()
        .map_err(|error| format!("disabling the CPU fallback: {error}"))?
        .with_execution_providers([provider.build().error_on_failure()])
        .map_err(|error| format!("registering the Vitis AI provider: {error}"))?
        .commit_from_memory(model)
        .map_err(|error| format!("the Vitis AI provider did not open the {name} graph: {error}"))?;
    let seconds = started.elapsed().as_secs_f64();

    // The first run, on an image without a text, profiled.
    let values = vec![0f32; input.shape.iter().product::<i64>() as usize];
    {
        let tensor = TensorRef::from_array_view((input.shape.clone(), &values[..])).unwrap();
        session
            .run(ort::inputs![input.name.as_str() => tensor])
            .map_err(|error| format!("the first run of the {name} graph failed: {error}"))?;
    }
    let profile_path = session
        .end_profiling()
        .map_err(|error| format!("ending the profile: {error}"))?;
    let profile = std::fs::read_to_string(&profile_path).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_file(&profile_path);
    let executed = ExecutedNodes::parse(&profile)?;

    let report = std::fs::read_to_string(&report_path)
        .map_err(|error| format!("no placement report at {}: {error}", report_path.display()))?;
    let placement = PlacementReport::parse(&report)?;
    println!(
        "{TAG}: {name} graph: input `{}` {:?}, outputs {:?}; {} nodes in the file ({}); report: \
         {} on the NPU, {} on the CPU {:?}, {:?} subgraph(s); first run through the provider \
         {:?}, elsewhere {:?}; ready in {seconds:.2} s",
        input.name,
        input.shape,
        info.outputs
            .iter()
            .map(|o| (&o.name, &o.shape))
            .collect::<Vec<_>>(),
        info.nodes.len(),
        info.operator_counts()
            .iter()
            .map(|(operator, count)| format!("{count} {operator}"))
            .collect::<Vec<_>>()
            .join(", "),
        placement.npu_nodes,
        placement.cpu_nodes,
        placement.cpu_operators(),
        placement.subgraphs,
        executed.on_provider,
        executed.elsewhere,
    );
    // The report's rule, for a graph with one input and one or more outputs. The
    // provider splits a dequantize node whose tensor is both a model output and the input
    // of a further node into two (`h` and `h/duplicated`, seen on the reference host), so
    // the report may list one node more per output beyond the first.
    let listed = placement.all_nodes as usize;
    if listed < info.nodes.len() || listed > info.nodes.len() + outputs.len() - 1 {
        return Err(format!(
            "the report lists {listed} nodes, the {name} graph has {} and {} output(s)",
            info.nodes.len(),
            outputs.len()
        ));
    }
    if placement.npu_nodes == 0 {
        return Err(format!("no node of the {name} graph is on the NPU"));
    }
    for node in &placement.cpu {
        let at_the_input = node.operator == "QuantizeLinear"
            && node.inputs.first().is_some_and(|x| *x == input.name);
        let at_an_output =
            node.operator == "DequantizeLinear" && node.outputs.iter().any(|y| outputs.contains(y));
        if !at_the_input && !at_an_output {
            return Err(format!(
                "the {name} graph leaves `{}` (outputs {:?}) to the CPU",
                node.operator, node.outputs
            ));
        }
    }
    // The profile's rule: `ExecutedNodes::npu_only` allows one dequantize node on the
    // CPU provider; a graph with two outputs has two at its ends, so the end nodes are
    // counted here against the number of outputs.
    let compiled_parts = executed
        .on_provider
        .iter()
        .filter(|operator| operator.starts_with(l2d_encoder::session::COMPILED_PART_PREFIX))
        .count();
    let strange: Vec<String> = executed
        .on_provider
        .iter()
        .filter(|operator| {
            !operator.starts_with(l2d_encoder::session::COMPILED_PART_PREFIX)
                && *operator != "QuantizeLinear"
                && *operator != "DequantizeLinear"
        })
        .cloned()
        .chain(
            executed
                .elsewhere
                .iter()
                .filter(|(operator, _)| {
                    operator != "QuantizeLinear" && operator != "DequantizeLinear"
                })
                .map(|(operator, provider)| format!("{operator} ({provider})")),
        )
        .collect();
    let ends_elsewhere =
        executed.elsewhere.len() - strange.iter().filter(|s| s.contains('(')).count();
    if compiled_parts == 0 || !strange.is_empty() || ends_elsewhere > 1 + outputs.len() {
        return Err(format!(
            "the first run of the {name} graph was not a run on the NPU: {compiled_parts} \
             compiled part(s) executed, not accepted: {strange:?}, end nodes off the provider: \
             {ends_elsewhere}"
        ));
    }
    Ok(Graph {
        session,
        dims: input.shape.clone(),
        input: input.name,
        outputs,
        placement,
        executed,
        compiled: written(&report_path).is_some() && written(&report_path) != before,
        seconds,
    })
}

// ---------------------------------------------------------------------------------------
// The patchers and the pooling: plain arithmetic between the two graphs
// ---------------------------------------------------------------------------------------

/// Splits patches longer than `max_patch` into chunks of that length and keeps the first
/// `limit` starts (the bytes of later patches fall into the last one). `starts` holds
/// the start of every patch of a line of `n` bytes, the first being 0.
fn finish_starts(
    starts: &mut Vec<usize>,
    n: usize,
    max_patch: usize,
    limit: usize,
    scratch: &mut Vec<usize>,
) {
    scratch.clear();
    for (index, &start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(n);
        let mut at = start;
        while at < end {
            scratch.push(at);
            at += max_patch;
        }
    }
    scratch.truncate(limit);
    std::mem::swap(starts, scratch);
}

fn starts_of(
    layout: &Layout,
    text: &[u8],
    entropy: &[f64],
    starts: &mut Vec<usize>,
    scratch: &mut Vec<usize>,
) {
    let n = text.len();
    starts.clear();
    if n == 0 {
        return;
    }
    starts.push(0);
    match layout.patcher.as_str() {
        "fixed" => starts.extend((layout.stride..n).step_by(layout.stride)),
        "delimiter" => {
            let min_len = layout.setting as usize;
            let mut last = 0;
            for (t, byte) in text.iter().enumerate().take(n - 1) {
                if layout.delimiter[usize::from(*byte)] && t + 1 - last >= min_len {
                    starts.push(t + 1);
                    last = t + 1;
                }
            }
        }
        // `entropy[t]` is the entropy of the prediction of byte t + 1.
        "entropy-jump" => {
            for t in 1..n.saturating_sub(1) {
                if entropy[t] - entropy[t - 1] > layout.setting {
                    starts.push(t + 1);
                }
            }
        }
        "entropy-global" => {
            for (t, value) in entropy.iter().enumerate().take(n - 1) {
                if *value > layout.setting {
                    starts.push(t + 1);
                }
            }
        }
        other => panic!("unknown patcher {other}"),
    }
    finish_starts(starts, n, layout.max_patch, layout.patch_limit, scratch);
}

/// The byte graph's input image and what the previous run wrote into it.
struct ByteImage {
    values: Vec<f32>,
    /// Bytes written into each row by the previous run.
    written: Vec<usize>,
}

/// What sits between the graphs, with its buffers: the entropies, the patch boundaries
/// and the pooling of the byte features into the line graph's input image.
struct Between {
    layout: Layout,
    patch_image: Vec<f32>,
    patch_written: Vec<usize>,
    /// exp(q * step) and q * exp(q * step) for the 256 values q of an 8-bit logit.
    exp_table: Vec<[f64; 2]>,
    total: Vec<f64>,
    weighted: Vec<f64>,
    entropy: Vec<f64>,
    starts: Vec<usize>,
    scratch: Vec<usize>,
}

struct Hierarchy {
    table: Vec<f32>,
    byte: Graph,
    line: Graph,
    byte_image: ByteImage,
    between: Between,
}

#[derive(Debug, Clone, Copy, Default)]
struct Seconds {
    write: f64,
    byte_run: f64,
    between: f64,
    line_run: f64,
    read: f64,
}

impl Seconds {
    fn total(&self) -> f64 {
        self.write + self.byte_run + self.between + self.line_run + self.read
    }
}

impl ByteImage {
    fn write(&mut self, layout: &Layout, table: &[f32], texts: &[&[u8]]) {
        let (rows, width, channels) = (layout.byte_rows, layout.input_bytes, layout.table_width);
        for row in 0..rows {
            let text: &[u8] = texts.get(row).copied().unwrap_or(&[]);
            let stale = self.written[row];
            for channel in 0..channels {
                let at = channel * rows * width + row * width;
                let line = &mut self.values[at..at + width];
                for (value, byte) in line.iter_mut().zip(text) {
                    *value = table[usize::from(*byte) * channels + channel];
                }
                if stale > text.len() {
                    line[text.len()..stale].fill(0.0);
                }
            }
            self.written[row] = text.len();
        }
    }
}

impl Between {
    /// The entropy (natural log) of the softmax of the 256 logits at each of the first
    /// `n` positions of row `row`. The graph gives the logits as relu(logit + 12) in 8
    /// bits with zero point 0, so a value is a whole number of steps from 0 to 255 and
    /// the exponentials come from a table of 256 entries: with q the steps of a class,
    /// s the step and Z the sum of exp(q s), the entropy is ln Z - s sum(q exp(q s)) / Z.
    fn entropies(&mut self, logits: &[f32], row: usize, n: usize) {
        let (rows, width) = (self.layout.byte_rows, self.layout.input_bytes);
        let scale = self
            .layout
            .logit_scale
            .expect("the layout names the step of the logits");
        let inverse = 1.0 / scale;
        let total = &mut self.total[..n];
        let weighted = &mut self.weighted[..n];
        total.fill(0.0);
        weighted.fill(0.0);
        for class in 0..256 {
            let at = class * rows * width + row * width;
            let values = &logits[at..at + n];
            for ((value, total), weighted) in
                values.iter().zip(total.iter_mut()).zip(weighted.iter_mut())
            {
                let [exp, exp_times_steps] =
                    self.exp_table[((value * inverse + 0.5) as usize).min(255)];
                *total += exp;
                *weighted += exp_times_steps;
            }
        }
        for t in 0..n {
            self.entropy[t] = total[t].ln() - f64::from(scale) * weighted[t] / total[t];
        }
    }

    /// The patch boundaries of `text` (its entropies are in `self.entropy`).
    fn cut(&mut self, text: &[u8]) {
        let mut starts = std::mem::take(&mut self.starts);
        let mut scratch = std::mem::take(&mut self.scratch);
        starts_of(
            &self.layout,
            text,
            &self.entropy[..text.len()],
            &mut starts,
            &mut scratch,
        );
        self.starts = starts;
        self.scratch = scratch;
    }

    /// The mean of the byte features of each patch of the line in row `byte_row` of the
    /// byte graph's output, written as row `patch_row` of the patch image. Returns the
    /// number of patches.
    fn pool(&mut self, features: &[f32], byte_row: usize, patch_row: usize, n: usize) -> usize {
        let (rows, width) = (self.layout.byte_rows, self.layout.input_bytes);
        let (patch_rows, limit) = (self.layout.patch_rows, self.layout.patch_limit);
        let patches = self.starts.len();
        let stale = self.patch_written[patch_row];
        for channel in 0..self.layout.features {
            let from = channel * rows * width + byte_row * width;
            let to = channel * patch_rows * limit + patch_row * limit;
            for patch in 0..patches {
                let start = self.starts[patch];
                let end = self.starts.get(patch + 1).copied().unwrap_or(n);
                let sum: f64 = features[from + start..from + end]
                    .iter()
                    .map(|value| f64::from(*value))
                    .sum();
                self.patch_image[to + patch] = (sum / (end - start) as f64) as f32;
            }
            if stale > patches {
                self.patch_image[to + patches..to + stale].fill(0.0);
            }
        }
        self.patch_written[patch_row] = patches;
        patches
    }
}

impl Hierarchy {
    fn open(fixture: &Fixture, fresh: bool) -> Self {
        let layout = fixture.layout.clone();
        let byte = open_graph(&fixture.rig, "byte", &fixture.byte_model, fresh)
            .unwrap_or_else(|error| panic!("{error}"));
        let line = open_graph(&fixture.rig, "line", &fixture.line_model, fresh)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            byte.dims,
            [
                1,
                layout.table_width as i64,
                layout.byte_rows as i64,
                layout.input_bytes as i64
            ]
        );
        assert_eq!(
            line.dims,
            [
                1,
                layout.features as i64,
                layout.patch_rows as i64,
                layout.patch_limit as i64
            ]
        );
        let with_entropy = layout.patcher.starts_with("entropy");
        assert_eq!(byte.outputs.len(), if with_entropy { 2 } else { 1 });
        let scale = f64::from(layout.logit_scale.unwrap_or(1.0));
        Self {
            table: fixture.table.clone(),
            byte,
            line,
            byte_image: ByteImage {
                values: vec![0.0; layout.table_width * layout.byte_rows * layout.input_bytes],
                written: vec![0; layout.byte_rows],
            },
            between: Between {
                patch_image: vec![0.0; layout.features * layout.patch_rows * layout.patch_limit],
                patch_written: vec![0; layout.patch_rows],
                exp_table: (0..256)
                    .map(|q| {
                        let exp = (f64::from(q) * scale).exp();
                        [exp, f64::from(q) * exp]
                    })
                    .collect(),
                total: vec![0.0; layout.input_bytes],
                weighted: vec![0.0; layout.input_bytes],
                entropy: vec![0.0; layout.input_bytes],
                starts: Vec::new(),
                scratch: Vec::new(),
                layout,
            },
        }
    }

    /// Unit vectors and patch counts of `texts`, appended, from text to unit vector.
    fn encode(
        &mut self,
        texts: &[&[u8]],
        vectors: &mut Vec<f32>,
        patches: &mut Vec<u32>,
        seconds: &mut Seconds,
    ) {
        let Self {
            table,
            byte,
            line,
            byte_image,
            between,
        } = self;
        let layout = between.layout.clone();
        let (byte_rows, patch_rows, dimension) =
            (layout.byte_rows, layout.patch_rows, layout.dimension);
        let with_entropy = layout.patcher.starts_with("entropy");
        for image in texts.chunks(patch_rows) {
            for (part_index, part) in image.chunks(byte_rows).enumerate() {
                let begun = Instant::now();
                byte_image.write(&layout, table, part);
                let written = Instant::now();
                let tensor =
                    TensorRef::from_array_view((byte.dims.clone(), &byte_image.values[..]))
                        .unwrap();
                let outputs = byte
                    .session
                    .run(ort::inputs![byte.input.as_str() => tensor])
                    .unwrap_or_else(|error| panic!("the byte graph failed: {error}"));
                let ran = Instant::now();
                // The outputs are read in place, not copied.
                let (_, features) = outputs[0].try_extract_tensor::<f32>().unwrap();
                let logits =
                    with_entropy.then(|| outputs[1].try_extract_tensor::<f32>().unwrap().1);
                for (row, text) in part.iter().enumerate() {
                    if let Some(logits) = logits {
                        between.entropies(logits, row, text.len());
                    }
                    between.cut(text);
                    let count =
                        between.pool(features, row, part_index * byte_rows + row, text.len());
                    patches.push(count as u32);
                }
                let pooled = Instant::now();
                seconds.write += (written - begun).as_secs_f64();
                seconds.byte_run += (ran - written).as_secs_f64();
                seconds.between += (pooled - ran).as_secs_f64();
            }
            let begun = Instant::now();
            // Rows of the patch image after this image's last line hold no line.
            for row in image.len()..patch_rows {
                between.starts.clear();
                between.pool(&[], 0, row, 0);
            }
            let cleared = Instant::now();
            let at = vectors.len();
            vectors.resize(at + image.len() * dimension, 0.0);
            let tensor =
                TensorRef::from_array_view((line.dims.clone(), &between.patch_image[..])).unwrap();
            let outputs = line
                .session
                .run(ort::inputs![line.input.as_str() => tensor])
                .unwrap_or_else(|error| panic!("the line graph failed: {error}"));
            let ran = Instant::now();
            let (_, out) = outputs[0].try_extract_tensor::<f32>().unwrap();
            for (n, vector) in vectors[at..].chunks_exact_mut(dimension).enumerate() {
                for (d, value) in vector.iter_mut().enumerate() {
                    *value = out[d * patch_rows + n];
                }
                l2_normalize(vector);
            }
            seconds.between += (cleared - begun).as_secs_f64();
            seconds.line_run += (ran - cleared).as_secs_f64();
            seconds.read += ran.elapsed().as_secs_f64();
        }
    }
}

fn refs(texts: &[Vec<u8>]) -> Vec<&[u8]> {
    texts.iter().map(Vec::as_slice).collect()
}

// ---------------------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------------------

/// Both graphs compile for the NPU and run there: the provider's report and the profile
/// of a first run, as the session of task 2.6 demands them. A fresh compilation.
#[test]
#[ignore = "needs the NPU and the fixture of task 2.16; run on request"]
fn both_graphs_compile_and_run_on_the_npu() {
    let Some(fixture) = fixture() else { return };
    let hierarchy = Hierarchy::open(&fixture, true);
    for (name, graph) in [("byte", &hierarchy.byte), ("line", &hierarchy.line)] {
        assert!(
            graph.compiled,
            "the {name} graph was not compiled in this open"
        );
        println!(
            "{TAG}: {name} graph on the NPU: {} of {} nodes, {} at the ends on the CPU; compiled \
             and ready in {:.2} s; first run through the provider {:?}",
            graph.placement.npu_nodes,
            graph.placement.all_nodes,
            graph.placement.cpu_nodes,
            graph.seconds,
            graph.executed.on_provider
        );
    }
}

/// Requirement 4.6 for the layout: a text's vector does not depend on its row in either
/// image nor on the other texts. Tolerance zero.
#[test]
#[ignore = "needs the NPU and the fixture of task 2.16; run on request"]
fn every_row_is_right_in_both_graphs() {
    let Some(fixture) = fixture() else { return };
    let mut hierarchy = Hierarchy::open(&fixture, false);
    let dimension = fixture.layout.dimension;
    let count = 4 * fixture.layout.patch_rows + 7;
    let step = fixture.layout.csic_texts / count;
    let chosen: Vec<Vec<u8>> = (0..count)
        .map(|i| fixture.texts[i * step].clone())
        .collect();
    let mut seconds = Seconds::default();

    let (mut together, mut patches) = (Vec::new(), Vec::new());
    hierarchy.encode(&refs(&chosen), &mut together, &mut patches, &mut seconds);

    // Other orders: every text sits in another row of both images, beside other texts.
    let mut differing = 0usize;
    let mut compared = 0usize;
    for (name, order) in [
        ("reversed", (0..count).rev().collect::<Vec<_>>()),
        (
            "rotated by 5",
            (0..count).map(|i| (i + 5) % count).collect(),
        ),
        (
            "rotated by 37",
            (0..count).map(|i| (i + 37) % count).collect(),
        ),
        (
            "strided by 3",
            (0..count).map(|i| (i * 3) % count).collect(),
        ),
    ] {
        let texts: Vec<&[u8]> = order.iter().map(|&i| chosen[i].as_slice()).collect();
        let (mut vectors, mut counts) = (Vec::new(), Vec::new());
        hierarchy.encode(&texts, &mut vectors, &mut counts, &mut seconds);
        let mut here = 0;
        for (at, &i) in order.iter().enumerate() {
            compared += 1;
            if vectors[at * dimension..(at + 1) * dimension]
                != together[i * dimension..(i + 1) * dimension]
                || counts[at] != patches[i]
            {
                here += 1;
            }
        }
        println!(
            "{TAG}: order `{name}`: {here} of {count} texts with another vector or patch count"
        );
        differing += here;
    }
    // Alone: a text in an image that holds nothing else.
    let mut alone_differing = 0;
    for i in (0..count).step_by(4) {
        let (mut vector, mut counts) = (Vec::new(), Vec::new());
        hierarchy.encode(
            &[chosen[i].as_slice()],
            &mut vector,
            &mut counts,
            &mut seconds,
        );
        compared += 1;
        if vector[..] != together[i * dimension..(i + 1) * dimension] || counts[0] != patches[i] {
            alone_differing += 1;
        }
    }
    println!(
        "{TAG}: alone: {alone_differing} of {} texts with another vector than among {count}",
        count.div_ceil(4)
    );
    // The same text in every row of both images.
    let same: Vec<&[u8]> = vec![chosen[3].as_slice(); fixture.layout.patch_rows];
    let (mut vectors, mut counts) = (Vec::new(), Vec::new());
    hierarchy.encode(&same, &mut vectors, &mut counts, &mut seconds);
    let rows_differing = vectors
        .chunks_exact(dimension)
        .filter(|vector| *vector != &together[3 * dimension..4 * dimension])
        .count();
    println!(
        "{TAG}: one text in each of the {} rows: {rows_differing} rows with another vector",
        fixture.layout.patch_rows
    );
    println!(
        "{TAG}: RESULT rows: {} of {compared} comparisons differ (tolerance zero bits)",
        differing + alone_differing + rows_differing
    );
    assert_eq!(differing + alone_differing + rows_differing, 0);
}

/// Encodes every text of the fixture, writes the vectors and patch counts for the
/// comparison with the Python 8-bit reference (`hg_run.py npu-compare`), compares them
/// here as well, and measures logs per second from text to unit vector.
#[test]
#[ignore = "needs the NPU and the fixture of task 2.16; run on request"]
fn vectors_against_the_reference_and_throughput() {
    let Some(fixture) = fixture() else { return };
    let mut hierarchy = Hierarchy::open(&fixture, false);
    let dimension = fixture.layout.dimension;
    let texts = refs(&fixture.texts);

    let (mut vectors, mut patches) = (Vec::new(), Vec::new());
    let mut seconds = Seconds::default();
    hierarchy.encode(&texts, &mut vectors, &mut patches, &mut seconds);
    println!(
        "{TAG}: first pass (untimed for the range): {:.0} logs/s",
        texts.len() as f64 / seconds.total()
    );
    let write = |name: &str, bytes: Vec<u8>| {
        let path = fixture.dir.join(name);
        std::fs::write(&path, bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    };
    write(
        "npu.f32",
        vectors
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    );
    write(
        "npu-patches.u32",
        patches
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    );

    let reference = floats(&read(&fixture.dir.join("reference.f32")));
    assert_eq!(reference.len(), vectors.len());
    let mut cosines: Vec<f64> = vectors
        .chunks_exact(dimension)
        .zip(reference.chunks_exact(dimension))
        .map(|(a, b)| cosine(a, b))
        .collect();
    let identical = vectors
        .chunks_exact(dimension)
        .zip(reference.chunks_exact(dimension))
        .filter(|(a, b)| a == b)
        .count();
    cosines.sort_by(f64::total_cmp);
    println!(
        "{TAG}: RESULT reference: {} texts; bit-identical to the Python 8-bit reference {identical}; \
         cosine min {:.6}, p01 {:.6}, median {:.6}; below 0.999: {}, below 0.99: {}",
        texts.len(),
        cosines[0],
        quantile(&cosines, 0.01),
        quantile(&cosines, 0.5),
        cosines.iter().filter(|c| **c < 0.999).count(),
        cosines.iter().filter(|c| **c < 0.99).count()
    );

    // A second pass must give the same bits.
    let (mut again, mut again_patches) = (Vec::new(), Vec::new());
    let mut passes = Vec::new();
    for pass in 0..5 {
        again.clear();
        again_patches.clear();
        let mut seconds = Seconds::default();
        let cpu = process_cpu_seconds();
        hierarchy.encode(&texts, &mut again, &mut again_patches, &mut seconds);
        let cpu = process_cpu_seconds() - cpu;
        let rate = texts.len() as f64 / seconds.total();
        let share = |part: f64| 100.0 * part / seconds.total();
        println!(
            "{TAG}: pass {pass}: {rate:.0} logs/s ({:.1} us per log: writing the byte image {:.0} %, \
             byte graph {:.0} %, entropy, patches and pooling {:.0} %, line graph {:.0} %, reading \
             {:.0} %); process CPU {:.1} s per million logs",
            1e6 / rate,
            share(seconds.write),
            share(seconds.byte_run),
            share(seconds.between),
            share(seconds.line_run),
            share(seconds.read),
            cpu / texts.len() as f64 * 1e6
        );
        passes.push(rate);
        assert!(
            again == vectors && again_patches == patches,
            "pass {pass} gave other bits"
        );
    }
    passes.sort_by(f64::total_cmp);
    let (working_set, peak, private) = process_memory();
    println!(
        "{TAG}: RESULT throughput: {:.0} to {:.0} logs/s over 5 passes of {} texts (median {:.0}); \
         requirement 4.2 asks for 1,000; working set {:.0} MiB, peak {:.0} MiB, private {:.0} MiB",
        passes[0],
        passes[4],
        texts.len(),
        passes[2],
        mib(working_set),
        mib(peak),
        mib(private)
    );
    assert!(
        passes[0] >= 1000.0 || cfg!(debug_assertions),
        "below 1,000 logs per second"
    );
}

/// Writes what each stage gives on the NPU for the first lines of the fixture, so that
/// `hg_run.py npu-stages` can say at which stage the NPU and the Python 8-bit reference
/// part: the byte graph's outputs for the first image, the patch image of the first
/// lines, and the line graph's raw output for it.
#[test]
#[ignore = "needs the NPU and the fixture of task 2.16; run on request"]
fn stage_outputs_for_the_comparison() {
    let Some(fixture) = fixture() else { return };
    let mut hierarchy = Hierarchy::open(&fixture, false);
    let layout = fixture.layout.clone();
    let texts = refs(&fixture.texts);
    let write = |name: &str, values: &[f32]| {
        let path = fixture.dir.join(name);
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    };
    {
        let Hierarchy {
            table,
            byte,
            byte_image,
            ..
        } = &mut hierarchy;
        byte_image.write(&layout, table, &texts[..layout.byte_rows]);
        let tensor =
            TensorRef::from_array_view((byte.dims.clone(), &byte_image.values[..])).unwrap();
        let outputs = byte
            .session
            .run(ort::inputs![byte.input.as_str() => tensor])
            .unwrap_or_else(|error| panic!("the byte graph failed: {error}"));
        write(
            "npu-stage-h.f32",
            outputs[0].try_extract_tensor::<f32>().unwrap().1,
        );
        if byte.outputs.len() > 1 {
            write(
                "npu-stage-y.f32",
                outputs[1].try_extract_tensor::<f32>().unwrap().1,
            );
        }
    }
    let (mut vectors, mut patches) = (Vec::new(), Vec::new());
    hierarchy.encode(
        &texts[..layout.patch_rows],
        &mut vectors,
        &mut patches,
        &mut Seconds::default(),
    );
    write("npu-stage-p.f32", &hierarchy.between.patch_image);
    let Hierarchy { line, between, .. } = &mut hierarchy;
    let tensor = TensorRef::from_array_view((line.dims.clone(), &between.patch_image[..])).unwrap();
    let outputs = line
        .session
        .run(ort::inputs![line.input.as_str() => tensor])
        .unwrap_or_else(|error| panic!("the line graph failed: {error}"));
    write(
        "npu-stage-e.f32",
        outputs[0].try_extract_tensor::<f32>().unwrap().1,
    );
    println!(
        "{TAG}: wrote the stage outputs of the first {} and {} lines to {}",
        layout.byte_rows,
        layout.patch_rows,
        fixture.dir.display()
    );
}
