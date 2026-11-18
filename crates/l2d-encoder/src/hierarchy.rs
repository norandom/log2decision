//! Product adapter for the immutable, NPU-verified 2.17 f0-s0-mask profile.
//! Both graphs run on the NPU; entropy cuts and pooling are ordinary host arithmetic.
//! The stable gate covers at most 1024 original UTF-8 bytes. Longer originals are
//! unsupported, never cut. This is a retrieval profile, not a classifier.
use crate::encode::provider_cache_key;
use crate::session::{
    self, COMPILED_PART_PREFIX, ExecutedNodes, PROVIDER_OPTIONS, PlacementReport, REPORT_FILE,
};
use crate::{CheckDepth, NpuStatus, RuntimeFiles, onnx_edit};
use l2d_core::resource::{ResourceGuard, busy_choice};
use l2d_core::{L2dError, MemoryBudget};
use ort::{
    environment::Environment,
    ep::{ArbitrarilyConfigurableExecutionProvider, Vitis},
    session::Session,
    value::TensorRef,
};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub const PROFILE: &str = "entropy-2.17-f0-s0-mask-dcc6e831-d5006041-ryzenai1.9.0-driver32.0.20102.4480-original-utf8-no-cut-1024-cosine-v1";
/// Conservative admission estimate from the gate's measured 524 MiB native peak,
/// including graph compilation and buffers. Native provider accounting is estimated.
pub const SESSION_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_BYTES: usize = 1024;
const FILES: [(&str, u64, &str); 5] = [
    (
        "byte.int8.onnx",
        337586,
        "dcc6e831721e1234596b539e3ad6f9f4adad3a897e19f3999899a1ce624c1760",
    ),
    (
        "line.int8.onnx",
        679792,
        "d5006041c482017c17c66fe6aee63f658047764642fac7a4e9338d2266f552e1",
    ),
    (
        "table.bin",
        65536,
        "0a850663e7020bff6da5a295ca303e7cc052e909b031ef47f47cfd364e94879e",
    ),
    (
        "exp-table.f64",
        4096,
        "2ba97d3aa87534936650a57c27688efd2f38762d8e3d1b5f1c68d6071a8f041c",
    ),
    (
        "layout.json",
        502,
        "0db323de50bda05e6ee7feb07be2425d371d414ae9ec736d34488a3e62c034a1",
    ),
];
pub fn identity() -> l2d_core::EncoderIdentity {
    let mut hash = Sha256::new();
    hash.update(b"l2d-pinned-hierarchical-original-v1\0");
    hash.update(PROFILE);
    for (name, length, digest) in FILES {
        hash.update(name);
        hash.update(length.to_le_bytes());
        hash.update(digest);
    }
    l2d_core::EncoderIdentity {
        name: "f0-s0-mask".into(),
        revision: format!("{:x}", hash.finalize()),
        dimension: 128,
    }
}

/// Observations about the original as accepted or refused by this fixed no-cut codec.
#[derive(serde::Serialize, Clone, Copy)]
pub struct OriginalFlags {
    pub truncated: bool,
    pub text_longer_than_input: bool,
    pub empty: bool,
}
impl OriginalFlags {
    pub fn of(text: &str) -> Self {
        Self {
            truncated: false,
            text_longer_than_input: text.len() > MAX_BYTES,
            empty: text.is_empty(),
        }
    }
    pub fn supported(self) -> bool {
        !self.text_longer_than_input && !self.empty
    }
}
fn fail(e: impl std::fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "stable NPU retrieval profile: {e}; no CPU encoder fallback (choose character-trigram explicitly)"
        ),
    }
}
/// Exact artifacts from the accepted gate, checked before any runtime is loaded.
pub fn verify_profile(directory: &Path, budget: &MemoryBudget) -> Result<(), L2dError> {
    budget.require_whole(4 * 1024 * 1024, "verify pinned hierarchical package")?;
    for (name, length, hash) in FILES {
        let path = directory.join(name);
        let metadata = fs::metadata(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                L2dError::Missing {
                    item: format!("pinned encoder artifact {name}"),
                    expected_at: path.clone(),
                }
            } else {
                fail(format!("{}: {error}", path.display()))
            }
        })?;
        if metadata.len() != length {
            return Err(fail(format!(
                "incompatible {name}: expected {length} bytes for {PROFILE}"
            )));
        }
        let bytes = fs::read(&path).map_err(fail)?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != hash {
            return Err(fail(format!(
                "incompatible {name}: expected SHA256 {hash}, actual {actual}"
            )));
        }
    }
    Ok(())
}
/// Placement counts and first-execution operators read from the actual provider reports.
#[derive(serde::Serialize)]
pub struct GraphEvidence {
    pub graph: String,
    pub npu_nodes: u32,
    pub cpu_boundary_nodes: u32,
    pub all_nodes: u32,
    pub executed_on_provider: Vec<String>,
    pub executed_elsewhere: Vec<(String, String)>,
}
struct Graph {
    evidence: GraphEvidence,
    session: Session,
    input: String,
    dims: Vec<i64>,
}
fn graph(
    environment: &Environment,
    cache: &Path,
    name: &str,
    model: &[u8],
) -> Result<Graph, L2dError> {
    let info = onnx_edit::graph_info(model).map_err(fail)?;
    let input = info.inputs.first().ok_or_else(|| fail("missing input"))?;
    let outputs: Vec<_> = info.outputs.iter().map(|o| o.name.clone()).collect();
    let key = provider_cache_key(
        &format!("hierarchy219-{name}"),
        "1.9.0",
        model,
        PROVIDER_OPTIONS,
    );
    let mut provider = Vitis::default()
        .with_cache_dir(cache.display())
        .with_cache_key(&key);
    for (k, v) in PROVIDER_OPTIONS {
        provider = provider.with_arbitrary_config(*k, *v);
    }
    let mut session = Session::builder(environment)
        .map_err(fail)?
        .with_profiling(cache.join(format!("{key}-first-{}", std::process::id())))
        .map_err(fail)?
        .with_disable_cpu_fallback()
        .map_err(fail)?
        .with_execution_providers([provider.build().error_on_failure()])
        .map_err(fail)?
        .commit_from_memory(model)
        .map_err(fail)?;
    let zeros = vec![0f32; input.shape.iter().product::<i64>() as usize];
    let tensor = TensorRef::from_array_view((input.shape.clone(), &zeros[..])).map_err(fail)?;
    session
        .run(ort::inputs![input.name.as_str()=>tensor])
        .map_err(fail)?;
    let path = session.end_profiling().map_err(fail)?;
    let profile = fs::read_to_string(&path).map_err(fail)?;
    let _ = fs::remove_file(&path);
    let executed = ExecutedNodes::parse(&profile).map_err(fail)?;
    let placement = PlacementReport::parse(
        &fs::read_to_string(cache.join(&key).join(REPORT_FILE)).map_err(fail)?,
    )
    .map_err(fail)?;
    let listed = placement.all_nodes as usize;
    if listed < info.nodes.len()
        || listed > info.nodes.len() + outputs.len() - 1
        || placement.npu_nodes == 0
    {
        return Err(fail("invalid NPU placement count"));
    }
    for node in &placement.cpu {
        let at_input =
            node.operator == "QuantizeLinear" && node.inputs.first() == Some(&input.name);
        let at_output =
            node.operator == "DequantizeLinear" && node.outputs.iter().any(|o| outputs.contains(o));
        if !at_input && !at_output {
            return Err(fail("internal model node placed on CPU"));
        }
    }
    let compiled = executed
        .on_provider
        .iter()
        .any(|op| op.starts_with(COMPILED_PART_PREFIX));
    let invalid = executed.on_provider.iter().any(|op| {
        !op.starts_with(COMPILED_PART_PREFIX) && op != "QuantizeLinear" && op != "DequantizeLinear"
    }) || executed
        .elsewhere
        .iter()
        .any(|(op, _)| op != "QuantizeLinear" && op != "DequantizeLinear");
    if !compiled || invalid || executed.elsewhere.len() > 1 + outputs.len() {
        return Err(fail(
            "first-run profile did not prove NPU-only graph execution",
        ));
    }
    Ok(Graph {
        evidence: GraphEvidence {
            graph: name.to_owned(),
            npu_nodes: placement.npu_nodes,
            cpu_boundary_nodes: placement.cpu_nodes,
            all_nodes: placement.all_nodes,
            executed_on_provider: executed.on_provider,
            executed_elsewhere: executed.elsewhere,
        },
        session,
        input: input.name.clone(),
        dims: input.shape.clone(),
    })
}

pub struct HierarchicalSession {
    byte: Graph,
    line: Graph,
    table: Vec<f32>,
    exp: Vec<[f64; 2]>,
    image: Vec<f32>,
    patches: Vec<f32>,
    started_utc: String,
    started: std::time::Instant,
    encoded: u64,
    encoding_time: std::time::Duration,
    driver_version: String,
    runtime_release: String,
    _guard: ResourceGuard,
}
impl HierarchicalSession {
    pub fn open(
        directory: &Path,
        runtime_dir: &Path,
        cache: &Path,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        budget.require_whole(
            SESSION_BYTES,
            "hierarchical NPU session including estimated native provider peak",
        )?;
        verify_profile(directory, budget)?;
        let started = std::time::Instant::now();
        let started_utc = session::utc_timestamp(std::time::SystemTime::now());
        let guard = ResourceGuard::npu("stable hierarchical retrieval encoder", busy_choice())?;
        let files = RuntimeFiles::check(runtime_dir, CheckDepth::Fast)?;
        let npu = NpuStatus::query()?;
        if files.release != "1.9.0" || npu.driver_version() != Some("32.0.20102.4480") {
            return Err(fail(format!(
                "incompatible runtime/device for {PROFILE}: Ryzen AI {}, NPU driver {:?}; recheck byte-graph bit identity before using a new profile",
                files.release,
                npu.driver_version()
            )));
        }
        let dll = session::plain(&files.onnxruntime_dll())?;
        let (environment, _) = session::runtime(&dll)?;
        fs::create_dir_all(cache).map_err(fail)?;
        let cache = session::plain(cache)?;
        // Read after verification and hash the actual bytes again before executing;
        // replacement between inspection and loading cannot execute another profile.
        let read = |index: usize| -> Result<Vec<u8>, L2dError> {
            let (name, length, hash) = FILES[index];
            budget.require_whole(length.saturating_mul(4), "pinned graph load")?;
            let bytes = fs::read(directory.join(name)).map_err(fail)?;
            if bytes.len() as u64 != length || format!("{:x}", Sha256::digest(&bytes)) != hash {
                return Err(fail("package changed during open"));
            }
            Ok(bytes)
        };
        let byte = graph(&environment, &cache, "byte", &read(0)?)?;
        let line = graph(&environment, &cache, "line", &read(1)?)?;
        let table = read(2)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let exp = read(3)?
            .as_chunks::<16>()
            .0
            .iter()
            .map(|v| {
                [
                    f64::from_le_bytes(v[..8].try_into().unwrap()),
                    f64::from_le_bytes(v[8..].try_into().unwrap()),
                ]
            })
            .collect();
        Ok(Self {
            byte,
            line,
            table,
            exp,
            image: vec![0.; 64 * 16 * 1024],
            patches: vec![0.; 96 * 64 * 128],
            started_utc,
            started,
            encoded: 0,
            encoding_time: std::time::Duration::ZERO,
            runtime_release: files.release,
            driver_version: npu.driver_version().unwrap_or_default().into(),
            _guard: guard,
        })
    }
    pub fn placement(&self) -> [&GraphEvidence; 2] {
        [&self.byte.evidence, &self.line.evidence]
    }
    pub fn run_record(
        &self,
        input: l2d_core::InputSummary,
        count: u64,
        encoding: std::time::Duration,
        duration: std::time::Duration,
    ) -> l2d_core::RunRecord {
        self.record(vec![input], count, encoding, duration)
    }
    /// Include every completed original encoding across the method adapter's calls.
    pub fn observed_run(&self) -> l2d_core::RunRecord {
        self.record(
            Vec::new(),
            self.encoded,
            self.encoding_time,
            self.started.elapsed(),
        )
    }
    fn record(
        &self,
        inputs: Vec<l2d_core::InputSummary>,
        count: u64,
        encoding: std::time::Duration,
        duration: std::time::Duration,
    ) -> l2d_core::RunRecord {
        use l2d_core::{Device, Placement, PlacementSource, RunKind, RunRecord};
        let evidence = self.placement();
        RunRecord {
            kind: RunKind::Encode,
            started_utc: self.started_utc.clone(),
            duration_ms: u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
            inputs,
            settings: serde_json::json!({"profile":PROFILE,"runtime_release":self.runtime_release,
                "graph_placement":evidence,"input_bytes":MAX_BYTES,"normalization":"raw pooled f32 outputs; cosine normalizes during comparison",
                "encoded":count,"encoding_seconds":encoding.as_secs_f64()}),
            seed: None,
            encoder: Some(identity()),
            device: Some(Device::Npu),
            driver_version: Some(self.driver_version.clone()),
            logs_per_second: (!encoding.is_zero()).then(|| count as f64 / encoding.as_secs_f64()),
            placement: Some(Placement {
                npu_nodes: evidence.iter().map(|g| g.npu_nodes).sum(),
                cpu_nodes: evidence.iter().map(|g| g.cpu_boundary_nodes).sum(),
                source: PlacementSource::ProviderReport,
            }),
        }
    }
    /// One original in row zero; other rows are zero. No input cutting or text masks.
    pub fn encode(&mut self, text: &str) -> Result<Option<Vec<f32>>, L2dError> {
        if !OriginalFlags::of(text).supported() {
            return Ok(None);
        }
        let start = std::time::Instant::now();
        self.image.fill(0.);
        self.patches.fill(0.);
        for (t, byte) in text.bytes().enumerate() {
            for c in 0..64 {
                self.image[c * 16 * 1024 + t] = self.table[byte as usize * 64 + c];
            }
        }
        let tensor =
            TensorRef::from_array_view((self.byte.dims.clone(), &self.image[..])).map_err(fail)?;
        let outputs = self
            .byte
            .session
            .run(ort::inputs![self.byte.input.as_str()=>tensor])
            .map_err(fail)?;
        let (_, features) = outputs[0].try_extract_tensor::<f32>().map_err(fail)?;
        let (_, logits) = outputs[1].try_extract_tensor::<f32>().map_err(fail)?;
        let n = text.len();
        let scale = 0.057_861_84_f32;
        let inverse = 1. / scale;
        let mut entropy = vec![0f64; n];
        for t in 0..n {
            let (mut total, mut weighted) = (0., 0.);
            for c in 0..256 {
                let [exp, ew] =
                    self.exp[((logits[c * 16 * 1024 + t] * inverse + 0.5) as usize).min(255)];
                total += exp;
                weighted += ew;
            }
            entropy[t] = total.ln() - f64::from(scale) * weighted / total;
        }
        let mut starts = vec![0usize];
        for t in 1..n.saturating_sub(1) {
            if entropy[t] - entropy[t - 1] > 0.48710060030316527 {
                starts.push(t + 1);
            }
        }
        let mut bounded = Vec::new();
        for (i, start) in starts.iter().enumerate() {
            let end = starts.get(i + 1).copied().unwrap_or(n);
            bounded.extend((*start..end).step_by(32));
        }
        bounded.truncate(128); // Remaining bytes belong to the last patch; none is cut.
        let count = bounded.len();
        for c in 0..96 {
            for (p, start) in bounded.iter().enumerate() {
                let end = bounded.get(p + 1).copied().unwrap_or(n);
                let sum: f64 = features[c * 16 * 1024 + start..c * 16 * 1024 + end]
                    .iter()
                    .map(|v| f64::from(*v))
                    .sum();
                self.patches[c * 64 * 128 + p] = (sum / (end - start) as f64) as f32;
            }
        }
        drop(outputs);
        let tensor = TensorRef::from_array_view((self.line.dims.clone(), &self.patches[..]))
            .map_err(fail)?;
        let outputs = self
            .line
            .session
            .run(ort::inputs![self.line.input.as_str()=>tensor])
            .map_err(fail)?;
        let (_, out) = outputs[0].try_extract_tensor::<f32>().map_err(fail)?;
        let v = (0..128)
            .map(|d| {
                (out[d * 64 * 128..d * 64 * 128 + count]
                    .iter()
                    .map(|v| f64::from(*v))
                    .sum::<f64>()
                    / count as f64) as f32
            })
            .collect();
        self.encoded += 1;
        self.encoding_time += start.elapsed();
        Ok(Some(v))
    }
}
