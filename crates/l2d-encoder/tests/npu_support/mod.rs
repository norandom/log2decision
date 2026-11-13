//! Shared by the hardware tests of task 2.12 (`npu_rows.rs`, `npu_throughput.rs`,
//! `npu_quality.rs`): the preconditions of a run on the NPU, the opening of a session on
//! the Vitis AI provider with its cache and placement report, the trained models and the
//! CSIC texts, and process measurements.
//!
//! Like the gate of task 2.2 these tests are asked for explicitly, so they fail when they
//! cannot prove what they are for (no runtime directory, no NPU, a Python interpreter on
//! `PATH`, no trained model) and skip only with `L2D_GATE_ALLOW_SKIP=1`.
//!
//! Where things are (environment variables, all read here):
//! - `L2D_RUNTIME_DIR`: the runtime files (default `runtime/ryzenai/` of the workspace);
//! - `L2D_NPU_WORK_DIR`: everything these tests write (provider caches, edited models,
//!   the scratch project, vectors); default: `l2d-npu-work` in the temporary directory;
//! - `L2D_SCIENCE_DIR`: the directory with the published model of task 2.1
//!   (`split17-f0-s0-strip-bce-norm-unit.{int8.onnx,f32.onnx,table.bin}`, `record.json`);
//! - `L2D_SCIENCE_RUNS`: the directory with the fold runs of task 2.1 (`npu_quality.rs`);
//! - `L2D_CSIC_CSV`: `csic_database.csv` (default `data/csic-2010/` of the workspace).
#![allow(dead_code)]

pub mod quality;

use std::path::{Path, PathBuf};
use std::time::Instant;

use l2d_core::{Csic2010Adapter, InputAdapter, L2dError, Label, ProjectPaths};
use l2d_encoder::{
    CheckDepth, EncoderArch, EncoderPackage, InputMode, NpuStatus, PackageParts, RuntimeFiles,
};
use ort::environment::Environment;
use ort::ep::{ArbitrarilyConfigurableExecutionProvider, Vitis};
use ort::session::Session;

pub const ALLOW_SKIP_VARIABLE: &str = "L2D_GATE_ALLOW_SKIP";
const REPORT_VARIABLE: &str = "XLNX_ONNX_EP_REPORT_FILE";
pub const REPORT_FILE: &str = "vitisai_ep_report.json";
/// The published model of task 2.1 (fitted without the three endpoints of the 1.7 split).
pub const PUBLISHED_STEM: &str = "split17-f0-s0-strip-bce-norm-unit";

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

pub fn skip_allowed() -> bool {
    std::env::var_os(ALLOW_SKIP_VARIABLE).is_some_and(|value| value != "0" && !value.is_empty())
}

/// The test cannot run on this host: a failure, unless a skip was asked for.
pub fn skip(tag: &str, reason: &str) {
    assert!(
        skip_allowed(),
        "the hardware test could not run: {reason} (set {ALLOW_SKIP_VARIABLE}=1 to skip instead \
         of failing)"
    );
    println!("{tag}: SKIPPED ({ALLOW_SKIP_VARIABLE} is set): {reason}");
}

/// A path in the plain drive form the provider was proven with (no `\\?\` prefix).
pub fn plain(path: &Path) -> PathBuf {
    let resolved = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("resolving {}: {error}", path.display()));
    PathBuf::from(
        resolved
            .to_string_lossy()
            .trim_start_matches(r"\\?\")
            .to_string(),
    )
}

fn pythons_on_path() -> Vec<PathBuf> {
    let mut real = Vec::new();
    let Some(path) = std::env::var_os("PATH") else {
        return real;
    };
    for dir in std::env::split_paths(&path) {
        for name in ["python.exe", "python3.exe", "py.exe", "pythonw.exe"] {
            let candidate = dir.join(name);
            // A zero-length entry is an app execution alias, not an interpreter.
            if std::fs::symlink_metadata(&candidate).is_ok_and(|meta| meta.len() > 0) {
                real.push(candidate);
            }
        }
    }
    real
}

/// Node counts of the provider's placement report (see `npu_gate.rs`).
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub npu_nodes: u64,
    pub cpu_nodes: u64,
    pub all_nodes: u64,
    /// The operators of the nodes left to the CPU, in graph order.
    pub cpu_operators: Vec<String>,
    /// How many subgraphs the provider compiled for the NPU, when the report says.
    pub subgraphs: Option<usize>,
}

pub fn placement_from_report(report: &str) -> Result<Placement, String> {
    let json: serde_json::Value =
        serde_json::from_str(report).map_err(|error| format!("report is not JSON: {error}"))?;
    let nodes = json
        .get("nodeStat")
        .and_then(|value| value.as_array())
        .ok_or("report has no `nodeStat` list")?;
    let mut placement = Placement {
        npu_nodes: 0,
        cpu_nodes: 0,
        all_nodes: nodes.len() as u64,
        cpu_operators: Vec::new(),
        subgraphs: json
            .get("subgraphStat")
            .and_then(|value| value.as_array())
            .map(Vec::len),
    };
    for (index, node) in nodes.iter().enumerate() {
        match node.get("device").and_then(|device| device.as_str()) {
            Some("NPU") => placement.npu_nodes += 1,
            Some("CPU") => {
                placement.cpu_nodes += 1;
                placement.cpu_operators.push(
                    node.get("opType")
                        .and_then(|op| op.as_str())
                        .unwrap_or("?")
                        .to_string(),
                );
            }
            other => return Err(format!("node {index} of `nodeStat` is on {other:?}")),
        }
    }
    Ok(placement)
}

/// How a session is opened.
#[derive(Debug, Clone, Default)]
pub struct Open<'a> {
    /// The name of the model's folder in the provider cache.
    pub cache_key: &'a str,
    /// Remove the cache folder first, so that the provider compiles.
    pub fresh: bool,
    pub intra_threads: Option<usize>,
    /// Further provider options, by the provider's own names.
    pub provider: &'a [(&'a str, &'a str)],
    /// Keep the compiled model in memory only (`enable_cache_file_io_in_mem` = 1).
    pub cache_in_memory: bool,
    /// Another cache directory than the rig's.
    pub cache_dir: Option<&'a Path>,
}

pub enum Model<'a> {
    File(&'a Path),
    Bytes(&'a [u8]),
}

pub struct Opened {
    pub session: Session,
    /// Seconds from the call to a session that can run.
    pub seconds: f64,
    pub cache_folder: PathBuf,
    /// Whether the provider wrote its report during this open: it compiled.
    pub compiled: bool,
    /// The placement of the compilation that filled the cache folder.
    pub placement: Option<Placement>,
}

/// Everything a hardware test needs before it opens a model.
pub struct Rig {
    pub tag: &'static str,
    pub environment: &'static Environment,
    pub release: String,
    pub runtime_dir: PathBuf,
    pub work_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub driver: String,
    /// Seconds the load of the runtime library took.
    pub load_seconds: f64,
}

impl Rig {
    /// Checks the preconditions and loads ONNX Runtime from the runtime directory.
    /// `None` when the test was skipped on request.
    pub fn prepare(tag: &'static str) -> Option<Self> {
        let runtime_dir = ProjectPaths::new(workspace_root())
            .runtime_files_dir(l2d_core::paths::runtime_dir_from_environment().as_deref());
        // The file check comes before anything is loaded (2.3): a missing DLL would end
        // the process without an error.
        let files = match RuntimeFiles::check(&runtime_dir, CheckDepth::Fast) {
            Ok(files) => files,
            Err(error @ L2dError::Missing { .. }) => {
                skip(tag, &error.to_string());
                return None;
            }
            Err(error) => panic!("the runtime file check failed: {error}"),
        };
        let dll = plain(&files.onnxruntime_dll());
        let npu = NpuStatus::query().unwrap_or_else(|error| panic!("{error}"));
        if !matches!(npu, NpuStatus::Present { .. }) {
            for line in npu.log_lines() {
                println!("{tag}: {line}");
            }
            skip(tag, "no AMD NPU device on this host");
            return None;
        }
        let driver = npu.driver_version().unwrap_or("unknown").to_string();
        let pythons = pythons_on_path();
        assert!(
            pythons.is_empty() || skip_allowed(),
            "PATH has a Python interpreter, so this run does not show that none is needed: \
             {pythons:?}"
        );

        let work_dir = std::env::var_os("L2D_NPU_WORK_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("l2d-npu-work"));
        let cache_dir = work_dir.join("cache");
        std::fs::create_dir_all(&cache_dir)
            .unwrap_or_else(|error| panic!("creating {}: {error}", cache_dir.display()));
        let work_dir = plain(&work_dir);
        let cache_dir = plain(&cache_dir);
        // SAFETY: set before the runtime is loaded and before this test starts a thread;
        // each hardware test binary runs its tests one at a time (`--test-threads=1`),
        // and nothing else in these binaries touches the environment concurrently.
        unsafe { std::env::set_var(REPORT_VARIABLE, REPORT_FILE) };

        let started = Instant::now();
        // The runtime is loaded once per process; the tests of a binary share it.
        static ENVIRONMENT: std::sync::OnceLock<Environment> = std::sync::OnceLock::new();
        let environment = ENVIRONMENT.get_or_init(|| {
            ort::init_from(&dll)
                .unwrap_or_else(|error| panic!("loading {} failed: {error}", dll.display()))
                .with_name("l2d-task-2.12")
                .build()
                .unwrap_or_else(|error| panic!("creating the runtime environment failed: {error}"))
        });
        let load_seconds = started.elapsed().as_secs_f64();
        println!(
            "{tag}: AMD Ryzen AI {} (ONNX Runtime {}, source `{}`) from {}; NPU driver {driver}; \
             work directory {}; {} build",
            files.release,
            files.onnxruntime,
            files.source,
            files.dir.display(),
            work_dir.display(),
            if cfg!(debug_assertions) {
                "DEBUG (timings mean nothing)"
            } else {
                "release"
            }
        );
        Some(Self {
            tag,
            environment,
            release: files.release.clone(),
            runtime_dir: dll.parent().unwrap().to_path_buf(),
            work_dir,
            cache_dir,
            driver,
            load_seconds,
        })
    }

    /// Opens a model on the Vitis AI provider (`error_on_failure`). An error is returned,
    /// not raised, so that a trial of an option can report it.
    pub fn open(&self, model: Model<'_>, how: &Open<'_>) -> Result<Opened, String> {
        let cache_dir = how.cache_dir.unwrap_or(&self.cache_dir);
        let cache_folder = cache_dir.join(how.cache_key);
        if how.fresh && cache_folder.is_dir() {
            std::fs::remove_dir_all(&cache_folder)
                .map_err(|error| format!("removing {}: {error}", cache_folder.display()))?;
        }
        let report_path = cache_folder.join(REPORT_FILE);
        let written = |path: &Path| {
            std::fs::metadata(path)
                .and_then(|meta| meta.modified())
                .ok()
        };
        let report_before = written(&report_path);

        let started = Instant::now();
        let mut provider = Vitis::default()
            .with_cache_dir(cache_dir.display())
            .with_cache_key(how.cache_key)
            .with_cache_in_memory(how.cache_in_memory);
        for (name, value) in how.provider {
            provider = provider.with_arbitrary_config(*name, *value);
        }
        let mut builder = Session::builder(self.environment)
            .map_err(|error| format!("creating the session builder: {error}"))?;
        if let Some(threads) = how.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| format!("setting the intra-op threads: {error}"))?;
        }
        let mut builder = builder
            .with_execution_providers([provider.build().error_on_failure()])
            .map_err(|error| format!("registering the Vitis AI provider: {error}"))?;
        let session = match model {
            Model::File(path) => builder.commit_from_file(path),
            Model::Bytes(bytes) => builder.commit_from_memory(bytes),
        }
        .map_err(|error| format!("opening the model on the Vitis AI provider: {error}"))?;
        let seconds = started.elapsed().as_secs_f64();

        let report_after = written(&report_path);
        let placement = std::fs::read_to_string(&report_path)
            .ok()
            .map(|report| placement_from_report(&report))
            .transpose()
            .map_err(|error| format!("{}: {error}", report_path.display()))?;
        Ok(Opened {
            session,
            seconds,
            cache_folder,
            compiled: report_after.is_some() && report_after != report_before,
            placement,
        })
    }

    /// A directory below the work directory, created.
    pub fn dir(&self, name: &str) -> PathBuf {
        let dir = self.work_dir.join(name);
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));
        dir
    }

    /// Writes an edited model into the work directory and returns its path.
    pub fn model_file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.dir("models").join(name);
        std::fs::write(&path, bytes)
            .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
        path
    }
}

/// Seconds spent in each stage of encoding, summed over the images of a pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stages {
    /// Text to bytes and table lookup, into the image buffer.
    pub codec: f64,
    /// Handing the buffer to the runtime (building the tensor that borrows it).
    pub hand: f64,
    /// The run: the provider's two CPU nodes and the NPU.
    pub run: f64,
    /// Reading the output: de-interleaving the rows and normalising.
    pub read: f64,
}

impl Stages {
    pub fn total(&self) -> f64 {
        self.codec + self.hand + self.run + self.read
    }
}

/// The encoder as task 2.12 found it right and fast: the model rewritten so that the
/// texts are the rows of one image (`onnx_edit::with_rows_as_height`), opened on the NPU
/// from memory, the float image lent to the runtime without a copy, one buffer reused.
/// This is what task 2.6 turns into `EncoderSession`.
pub struct HeightEncoder {
    pub session: Session,
    pub codec: l2d_encoder::ByteCodec,
    pub rows: usize,
    pub width: usize,
    pub ready_seconds: f64,
    pub compiled: bool,
    pub placement: Option<Placement>,
    image: l2d_encoder::encode::ImageInput<f32>,
    values: Vec<f32>,
    /// Rows of the buffer that hold a text from the previous image.
    written: usize,
    input: String,
}

impl HeightEncoder {
    /// `model` is the trained 8-bit model in its batch layout; `how.cache_key` names the
    /// compiled model.
    pub fn open(
        rig: &Rig,
        model: &[u8],
        codec: &l2d_encoder::ByteCodec,
        rows: usize,
        how: &Open<'_>,
    ) -> Result<Self, String> {
        let interface =
            l2d_encoder::ModelInterface::from_onnx(model).map_err(|error| error.to_string())?;
        let edited = l2d_encoder::onnx_edit::with_rows_as_height(model, rows as u32)
            .map_err(|error| error.to_string())?;
        let opened = rig.open(Model::Bytes(&edited), how)?;
        let image = l2d_encoder::encode::ImageInput::float(codec);
        Ok(Self {
            session: opened.session,
            codec: codec.clone(),
            rows,
            width: interface.output.shape[1] as usize,
            ready_seconds: opened.seconds,
            compiled: opened.compiled,
            placement: opened.placement,
            values: vec![0.0; image.image_values(rows)],
            image,
            written: 0,
            input: interface.input.name,
        })
    }

    fn dims(&self) -> Vec<i64> {
        let arch = self.codec.arch();
        vec![
            1,
            i64::from(arch.channels),
            self.rows as i64,
            i64::from(arch.input_bytes),
        ]
    }

    fn fill(&mut self, texts: &[&str]) {
        assert!(texts.len() <= self.rows);
        for (row, text) in texts.iter().enumerate() {
            self.image
                .write(&self.codec, text, self.rows, row, &mut self.values);
        }
        for row in texts.len()..self.written {
            self.image.clear_row(self.rows, row, &mut self.values);
        }
        self.written = texts.len();
    }

    /// The model's own output for up to `rows` texts, one row after the other, not
    /// normalised: for comparing bit for bit.
    pub fn raw(&mut self, texts: &[&str]) -> Vec<f32> {
        self.fill(texts);
        let dims = self.dims();
        let tensor = ort::value::TensorRef::from_array_view((dims, &self.values[..])).unwrap();
        let outputs = self
            .session
            .run(ort::inputs![self.input.as_str() => tensor])
            .unwrap_or_else(|error| panic!("inference failed: {error}"));
        let (_, out) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let mut rows = vec![0f32; texts.len() * self.width];
        for (n, row) in rows.chunks_exact_mut(self.width).enumerate() {
            for (d, value) in row.iter_mut().enumerate() {
                *value = out[d * self.rows + n];
            }
        }
        rows
    }

    /// Runs the image that is in the buffer once more and returns the first values of
    /// its output: the cost of a run without any text being encoded.
    pub fn raw_again(&mut self) -> [f32; 4] {
        let dims = self.dims();
        let tensor = ort::value::TensorRef::from_array_view((dims, &self.values[..])).unwrap();
        let outputs = self
            .session
            .run(ort::inputs![self.input.as_str() => tensor])
            .unwrap_or_else(|error| panic!("inference failed: {error}"));
        let (_, out) = outputs[0].try_extract_tensor::<f32>().unwrap();
        [out[0], out[1], out[2], out[3]]
    }

    /// The L2-normalised vectors of `texts`, in order, appended to `vectors`.
    pub fn encode(&mut self, texts: &[&str], vectors: &mut Vec<f32>, stages: &mut Stages) {
        let dims = self.dims();
        for image in texts.chunks(self.rows) {
            let started = Instant::now();
            self.fill(image);
            let filled = Instant::now();
            let tensor =
                ort::value::TensorRef::from_array_view((dims.clone(), &self.values[..])).unwrap();
            let handed = Instant::now();
            let outputs = self
                .session
                .run(ort::inputs![self.input.as_str() => tensor])
                .unwrap_or_else(|error| panic!("inference failed: {error}"));
            let ran = Instant::now();
            let (_, out) = outputs[0].try_extract_tensor::<f32>().unwrap();
            let at = vectors.len();
            vectors.resize(at + image.len() * self.width, 0.0);
            l2d_encoder::encode::read_rows(out, self.rows, image.len(), &mut vectors[at..]);
            let read = Instant::now();
            stages.codec += (filled - started).as_secs_f64();
            stages.hand += (handed - filled).as_secs_f64();
            stages.run += (ran - handed).as_secs_f64();
            stages.read += (read - ran).as_secs_f64();
        }
    }
}

/// The directory an environment variable names, or a failure (a skip on request).
pub fn directory_from(tag: &str, variable: &str, what: &str) -> Option<PathBuf> {
    match std::env::var_os(variable).filter(|value| !value.is_empty()) {
        Some(dir) if Path::new(&dir).is_dir() => Some(PathBuf::from(dir)),
        Some(dir) => {
            skip(
                tag,
                &format!("{variable} names {dir:?}, which is not a directory ({what})"),
            );
            None
        }
        None => {
            skip(tag, &format!("{variable} is not set ({what})"));
            None
        }
    }
}

pub fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

/// The published model of task 2.1 packaged into a scratch project below the work
/// directory, as `EncoderPackage` (payload mode: the model was fitted on the payload).
pub fn published_package(rig: &Rig, science_dir: &Path) -> EncoderPackage {
    let file = |suffix: &str| read(&science_dir.join(format!("{PUBLISHED_STEM}{suffix}")));
    let model = file(".int8.onnx");
    let float = file(".f32.onnx");
    let table = file(".table.bin");
    let record = read(&science_dir.join("record.json"));
    let paths = ProjectPaths::new(rig.dir("project"));
    let written = EncoderPackage::write(
        &paths,
        &PackageParts {
            name: "csic-science-2.1",
            arch: EncoderArch::V1,
            input_mode: InputMode::Payload,
            model_int8: &model,
            table: &table,
            training_record: &record,
            model_f32: Some(&float),
        },
    )
    .unwrap_or_else(|error| panic!("packaging the published model: {error}"));
    let dir = written.package.dir().to_path_buf();
    EncoderPackage::open(&paths, &dir).unwrap_or_else(|error| panic!("{error}"))
}

/// One distinct CSIC text with its label and its group (the URL path).
pub struct CsicLog {
    pub text: String,
    pub bad: bool,
    pub group: String,
}

/// CSIC's distinct texts in order of first occurrence.
pub fn csic_logs() -> Vec<CsicLog> {
    let supplied = std::env::var_os("L2D_CSIC_CSV").map(PathBuf::from);
    let path = Csic2010Adapter::locate(&ProjectPaths::new(workspace_root()), supplied.as_deref())
        .unwrap_or_else(|error| panic!("{error} (set L2D_CSIC_CSV to csic_database.csv)"));
    let (logs, _) = Csic2010Adapter::new()
        .read(&path)
        .unwrap_or_else(|error| panic!("{error}"));
    logs.records()
        .map(|record| CsicLog {
            bad: match record.label {
                Label::Bad => true,
                Label::Good => false,
                Label::Unknown => panic!("a CSIC record without a label"),
            },
            group: record.group.expect("CSIC records carry their URL path"),
            text: record.text,
        })
        .collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

/// Squared L2 distance.
pub fn distance2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum()
}

/// The value at the quantile `q` (0 to 1) of sorted values.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty());
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// CPU seconds (kernel and user) this process has used so far.
pub fn process_cpu_seconds() -> f64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: four valid out-pointers and the pseudo-handle of this process.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut created,
            &mut exited,
            &mut kernel,
            &mut user,
        )
    };
    assert_ne!(ok, 0, "GetProcessTimes failed");
    let ticks = |time: FILETIME| {
        (u64::from(time.dwHighDateTime) << 32 | u64::from(time.dwLowDateTime)) as f64
    };
    (ticks(kernel) + ticks(user)) / 1e7
}

/// (working set now, peak working set, private commit now), in bytes.
pub fn process_memory() -> (u64, u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: a zeroed counters structure of the size passed, and this process's handle.
    let counters = unsafe {
        let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        counters.cb = size;
        assert_ne!(
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size),
            0,
            "GetProcessMemoryInfo failed"
        );
        counters
    };
    (
        counters.WorkingSetSize as u64,
        counters.PeakWorkingSetSize as u64,
        counters.PagefileUsage as u64,
    )
}

pub fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
