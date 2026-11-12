//! Hardware gate of task 2.2: prove from Rust that the spike's 8-bit model runs on the AMD
//! NPU through ONNX Runtime's Vitis AI provider, loaded at run time through `ort`.
//!
//! Run on request (the runtime files are not in the repository, see `runtime/README.md`):
//!
//! ```text
//! $env:L2D_RUNTIME_DIR = "D:\Source\log2decision\runtime\ryzenai"   # default: <workspace>/runtime/ryzenai
//! cargo test -p l2d-encoder --test npu_gate -- --ignored --nocapture
//! ```
//!
//! The test is asked for explicitly, so it fails when it cannot prove what it is for:
//! without the runtime directory or one of its files, without an AMD NPU device, and with
//! a Python interpreter on `PATH` (the proof is "no Python at run time"). Only with
//! `L2D_GATE_ALLOW_SKIP=1` does it instead print `SKIPPED` with the reason and pass
//! (missing runtime files or device), or report the interpreter and go on (Python). With
//! the runtime and the device present, every failure is a failure: the provider is
//! registered with `error_on_failure`, zero nodes on the NPU fail the test, and so does an
//! output that is the CPU provider's.
//!
//! This file holds everything the proof needs and nothing of the encoder itself:
//! `RuntimeFiles::check` is task 2.3, the session is task 2.6.

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use ort::ep::Vitis;
use ort::session::Session;
use ort::value::TensorRef;
use sha2::{Digest, Sha256};

/// The spike's 8-bit model with a fixed batch of 32 (`enc_b32_int8.onnx` of 2026-11-11).
const FIXTURE: &str = "tests/fixtures/spike-enc-b32-int8.onnx";
const FIXTURE_BYTES: u64 = 989_260;
const FIXTURE_SHA256: &str = "6901c42a8d1187b71b986eaaae6b7306914e9da675a17a0f4e2776719f49a638";

const BATCH: usize = 32;
const WARM_UP_BATCHES: usize = 5;
const TIMED_BATCHES: usize = 300;

/// The files the proof needs in the runtime directory (`runtime/README.md` has the table).
/// The presence check is not a courtesy: with `dyn_dispatch_core.dll` missing the process
/// dies with an access violation while the provider is registered, and no error is
/// returned.
const RUNTIME_FILES: &[&str] = &[
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_vitisai.dll",
    "onnxruntime_vitisai_ep.dll",
    "dyn_dispatch_core.dll",
];

/// Device setup class `ComputeAccelerator`, the class the AMD NPU driver installs under.
const COMPUTE_ACCELERATOR_CLASS: &str = "{F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}";
/// PCI vendor id of AMD.
const AMD_PCI_PREFIX: &str = "PCI\\VEN_1022&";

/// The file the provider writes its placement report to, inside the cache directory of
/// the model. The provider reads the name from this environment variable.
const REPORT_VARIABLE: &str = "XLNX_ONNX_EP_REPORT_FILE";
const REPORT_FILE: &str = "vitisai_ep_report.json";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// The opt-in that turns "the gate cannot run here" from a failure into a skip.
const ALLOW_SKIP_VARIABLE: &str = "L2D_GATE_ALLOW_SKIP";

fn skip_allowed() -> bool {
    std::env::var_os(ALLOW_SKIP_VARIABLE).is_some_and(|value| value != "0" && !value.is_empty())
}

/// The gate cannot run on this host: a failure, unless a skip was asked for.
fn skip(reason: &str) {
    assert!(
        skip_allowed(),
        "the gate could not run: {reason} (set {ALLOW_SKIP_VARIABLE}=1 to skip instead of \
         failing)"
    );
    println!("npu_gate: SKIPPED ({ALLOW_SKIP_VARIABLE} is set): {reason}");
}

/// Python interpreters reachable through `PATH`. A zero-length file is a Windows app
/// execution alias (the Store stub), not an interpreter, and is reported apart.
fn python_on_path() -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut real = Vec::new();
    let mut aliases = Vec::new();
    let Some(path) = std::env::var_os("PATH") else {
        return (real, aliases);
    };
    for dir in std::env::split_paths(&path) {
        for name in ["python.exe", "python3.exe", "py.exe", "pythonw.exe"] {
            let candidate = dir.join(name);
            // An app execution alias is a reparse point: `metadata` cannot follow it, the
            // link itself has length zero.
            match std::fs::symlink_metadata(&candidate) {
                Ok(meta) if meta.len() == 0 => aliases.push(candidate),
                Ok(_) => real.push(candidate),
                Err(_) => {}
            }
        }
    }
    (real, aliases)
}

#[derive(Debug)]
struct NpuDevice {
    instance_id: String,
    description: String,
    driver_version: String,
}

impl NpuDevice {
    /// A PCI device of AMD. The device id is not compared: the NPU of the reference host
    /// is `DEV_17F0`, other processor generations have other ids.
    fn is_amd(&self) -> bool {
        self.instance_id
            .to_ascii_uppercase()
            .starts_with(AMD_PCI_PREFIX)
    }
}

/// Every present device of the class `ComputeAccelerator`, as the Windows configuration
/// manager reports it, in the manager's order.
fn compute_accelerators() -> Result<Vec<NpuDevice>, String> {
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        CM_GETIDLIST_FILTER_CLASS, CM_GETIDLIST_FILTER_PRESENT, CM_Get_Device_ID_List_SizeW,
        CM_Get_Device_ID_ListW, CM_LOCATE_DEVNODE_NORMAL, CM_Locate_DevNodeW, CR_SUCCESS,
    };
    use windows_sys::Win32::Devices::Properties::{
        DEVPKEY_Device_DeviceDesc, DEVPKEY_Device_DriverVersion,
    };

    let class: Vec<u16> = COMPUTE_ACCELERATOR_CLASS
        .encode_utf16()
        .chain(Some(0))
        .collect();
    let flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
    let mut length = 0u32;
    // SAFETY: `class` is a NUL-terminated UTF-16 string and `length` a valid output.
    let status = unsafe { CM_Get_Device_ID_List_SizeW(&mut length, class.as_ptr(), flags) };
    if status != CR_SUCCESS {
        return Err(format!("CM_Get_Device_ID_List_SizeW failed with {status}"));
    }
    let mut list = vec![0u16; length as usize];
    // SAFETY: `list` has the `length` elements the size call asked for.
    let status =
        unsafe { CM_Get_Device_ID_ListW(class.as_ptr(), list.as_mut_ptr(), length, flags) };
    if status != CR_SUCCESS {
        return Err(format!("CM_Get_Device_ID_ListW failed with {status}"));
    }
    let mut devices = Vec::new();
    for id in list.split(|unit| *unit == 0).filter(|id| !id.is_empty()) {
        let instance_id = String::from_utf16_lossy(id);
        let id_z: Vec<u16> = id.iter().copied().chain(Some(0)).collect();
        let mut node = 0u32;
        // SAFETY: `id_z` is a NUL-terminated device instance id.
        let status =
            unsafe { CM_Locate_DevNodeW(&mut node, id_z.as_ptr(), CM_LOCATE_DEVNODE_NORMAL) };
        if status != CR_SUCCESS {
            return Err(format!(
                "CM_Locate_DevNodeW({instance_id}) failed with {status}"
            ));
        }
        let description = device_string(node, &DEVPKEY_Device_DeviceDesc)
            .map_err(|error| format!("description of {instance_id}: {error}"))?;
        let driver_version = device_string(node, &DEVPKEY_Device_DriverVersion)
            .map_err(|error| format!("driver version of {instance_id}: {error}"))?;
        devices.push(NpuDevice {
            instance_id,
            description,
            driver_version,
        });
    }
    Ok(devices)
}

/// A string property of a device node.
fn device_string(
    node: u32,
    key: &windows_sys::Win32::Foundation::DEVPROPKEY,
) -> Result<String, String> {
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        CM_Get_DevNode_PropertyW, CR_SUCCESS,
    };
    use windows_sys::Win32::Devices::Properties::DEVPROP_TYPE_STRING;

    let mut kind = 0u32;
    let mut buffer = vec![0u16; 512];
    let mut bytes = (buffer.len() * 2) as u32;
    // SAFETY: `buffer` is `bytes` bytes long and aligned for UTF-16; the other pointers
    // are valid for the call.
    let status = unsafe {
        CM_Get_DevNode_PropertyW(
            node,
            key,
            &mut kind,
            buffer.as_mut_ptr().cast(),
            &mut bytes,
            0,
        )
    };
    if status != CR_SUCCESS {
        return Err(format!("CM_Get_DevNode_PropertyW failed with {status}"));
    }
    if kind != DEVPROP_TYPE_STRING {
        return Err(format!("property has type {kind}, not a string"));
    }
    let units = &buffer[..(bytes as usize / 2).min(buffer.len())];
    let end = units
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(units.len());
    Ok(String::from_utf16_lossy(&units[..end]))
}

/// Node counts of the provider's placement report.
#[derive(Debug, PartialEq)]
struct Placement {
    npu_nodes: u64,
    cpu_nodes: u64,
    all_nodes: u64,
}

/// Counts the nodes of the provider's report by device. `nodeStat` has one entry per node
/// of the model with its `device` (`NPU` or `CPU`); `deviceStat` has the totals per device
/// (`all`, `NPU`, and `VITIS_EP_CPU` for the nodes the provider leaves to the CPU). The
/// nodes are counted from `nodeStat` and must agree with each total; a device of another
/// name is an error, not a CPU node, and the two devices must add up to the total.
fn placement_from_report(report: &str) -> Result<Placement, String> {
    let json: serde_json::Value =
        serde_json::from_str(report).map_err(|error| format!("report is not JSON: {error}"))?;
    let list = |name: &str| {
        json.get(name)
            .and_then(|value| value.as_array())
            .ok_or(format!("report has no `{name}` list"))
    };
    let nodes = list("nodeStat")?;
    let mut npu_nodes = 0u64;
    let mut cpu_nodes = 0u64;
    for (index, node) in nodes.iter().enumerate() {
        match node.get("device").and_then(|device| device.as_str()) {
            Some("NPU") => npu_nodes += 1,
            Some("CPU") => cpu_nodes += 1,
            Some(other) => {
                return Err(format!(
                    "node {index} of `nodeStat` is on the unknown device `{other}`"
                ));
            }
            None => return Err(format!("node {index} of `nodeStat` has no device")),
        }
    }
    let all_nodes = nodes.len() as u64;
    let totals = list("deviceStat")?;
    let total = |device: &str| -> Option<u64> {
        totals
            .iter()
            .find(|entry| entry.get("name").and_then(|name| name.as_str()) == Some(device))
            .and_then(|entry| entry.get("nodeNum").and_then(|nodes| nodes.as_u64()))
    };
    // A device without nodes may be left out of `deviceStat`.
    let (npu_total, cpu_total) = (
        total("NPU").unwrap_or(0),
        total("VITIS_EP_CPU").unwrap_or(0),
    );
    if total("all") != Some(all_nodes)
        || npu_total != npu_nodes
        || cpu_total != cpu_nodes
        || npu_total + cpu_total != all_nodes
    {
        return Err(format!(
            "`nodeStat` ({npu_nodes} nodes on the NPU, {cpu_nodes} on the CPU, {all_nodes} in \
             all) disagrees with `deviceStat` (all {:?}, NPU {:?}, VITIS_EP_CPU {:?})",
            total("all"),
            total("NPU"),
            total("VITIS_EP_CPU")
        ));
    }
    Ok(Placement {
        npu_nodes,
        cpu_nodes,
        all_nodes,
    })
}

/// The provider name ONNX Runtime uses for the Vitis AI provider.
const VITIS_PROVIDER: &str = "VitisAIExecutionProvider";

/// Counts the executed nodes of an ONNX Runtime profile by provider. The profile is a
/// JSON list of events; an event of the category `Node` whose name ends in `_kernel_time`
/// is one execution of one node, and its `args.provider` names the provider.
fn executed_by_provider(profile: &str) -> Result<std::collections::BTreeMap<String, u64>, String> {
    let json: serde_json::Value =
        serde_json::from_str(profile).map_err(|error| format!("profile is not JSON: {error}"))?;
    let events = json.as_array().ok_or("profile is not a list of events")?;
    let mut counts = std::collections::BTreeMap::new();
    for event in events {
        let text = |key: &str| event.get(key).and_then(|value| value.as_str());
        if text("cat") != Some("Node") || !text("name").is_some_and(|n| n.ends_with("_kernel_time"))
        {
            continue;
        }
        let provider = event
            .get("args")
            .and_then(|args| args.get("provider"))
            .and_then(|provider| provider.as_str())
            .ok_or(format!("node event {:?} names no provider", text("name")))?;
        *counts.entry(provider.to_string()).or_insert(0) += 1;
    }
    Ok(counts)
}

#[test]
fn the_placement_is_read_from_a_provider_report() {
    // The shape of the report of Ryzen AI 1.9 (`voe` 1.9.0), cut down to three nodes.
    let report = r#"{"deviceStat":[
        {"name":"all","nodeNum":3,"supportedOpType":["::Conv","::QuantizeLinear"]},
        {"name":"NPU","nodeNum":2,"supportedOpType":["::Conv"]},
        {"name":"VITIS_EP_CPU","nodeNum":1,"supportedOpType":["::QuantizeLinear"]}],
        "nodeStat":[
        {"opType":"QuantizeLinear","device":"CPU"},
        {"opType":"Conv","device":"NPU"},
        {"opType":"Conv","device":"NPU"}],
        "shapeInfo":[],"subgraphStat":[]}"#;
    assert_eq!(
        placement_from_report(report),
        Ok(Placement {
            npu_nodes: 2,
            cpu_nodes: 1,
            all_nodes: 3
        })
    );
    let cpu_only = r#"{"deviceStat":[{"name":"all","nodeNum":2},{"name":"VITIS_EP_CPU","nodeNum":2}],
        "nodeStat":[{"device":"CPU"},{"device":"CPU"}]}"#;
    assert_eq!(
        placement_from_report(cpu_only),
        Ok(Placement {
            npu_nodes: 0,
            cpu_nodes: 2,
            all_nodes: 2
        })
    );
    let disagreeing = r#"{"deviceStat":[{"name":"all","nodeNum":2},{"name":"NPU","nodeNum":2}],
        "nodeStat":[{"device":"NPU"},{"device":"CPU"}]}"#;
    assert!(placement_from_report(disagreeing).is_err());
    // The CPU total is checked as well: here `all` and `NPU` agree with the nodes, but
    // the two devices do not add up to the total.
    let cpu_total_wrong = r#"{"deviceStat":[{"name":"all","nodeNum":2},{"name":"NPU","nodeNum":1},
        {"name":"VITIS_EP_CPU","nodeNum":0}],
        "nodeStat":[{"device":"NPU"},{"device":"CPU"}]}"#;
    let error = placement_from_report(cpu_total_wrong).unwrap_err();
    assert!(error.contains("VITIS_EP_CPU Some(0)"), "{error}");
    let cpu_total_absent = r#"{"deviceStat":[{"name":"all","nodeNum":2},{"name":"NPU","nodeNum":1}],
        "nodeStat":[{"device":"NPU"},{"device":"CPU"}]}"#;
    assert!(placement_from_report(cpu_total_absent).is_err());
    // A device the parser does not know is named, not counted as CPU.
    let unknown_device = r#"{"deviceStat":[{"name":"all","nodeNum":2},{"name":"NPU","nodeNum":1},
        {"name":"VITIS_EP_CPU","nodeNum":1}],
        "nodeStat":[{"device":"NPU"},{"device":"GPU"}]}"#;
    let error = placement_from_report(unknown_device).unwrap_err();
    assert!(error.contains("unknown device `GPU`"), "{error}");

    // The profile: only executed nodes count, each under its provider.
    let profile = r#"[
        {"cat":"Session","name":"model_loading_uri","args":{}},
        {"cat":"Node","name":"a_fence_before","args":{"provider":"VitisAIExecutionProvider"}},
        {"cat":"Node","name":"a_kernel_time","args":{"provider":"VitisAIExecutionProvider"}},
        {"cat":"Node","name":"b_kernel_time","args":{"provider":"VitisAIExecutionProvider"}},
        {"cat":"Node","name":"c_kernel_time","args":{"provider":"CPUExecutionProvider"}},
        {"cat":"Session","name":"model_run","args":{}}]"#;
    let executed = executed_by_provider(profile).unwrap();
    assert_eq!(executed.get(VITIS_PROVIDER), Some(&2));
    assert_eq!(executed.get("CPUExecutionProvider"), Some(&1));
    assert_eq!(executed.len(), 2);
    assert!(executed_by_provider(r#"[{"cat":"Node","name":"a_kernel_time","args":{}}]"#).is_err());
    assert!(executed_by_provider("{}").is_err());
    assert!(placement_from_report(r#"{"deviceStat":[]}"#).is_err());
    assert!(placement_from_report(r#"{"nodeStat":[{"opType":"Conv"}],"deviceStat":[]}"#).is_err());
    assert!(placement_from_report("{}").is_err());
    assert!(placement_from_report("not json").is_err());
}

/// Deterministic input values in [-1, 1): the proof needs no random number crate.
fn fill(values: &mut [f32], mut state: u64) {
    for value in values {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *value = ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

/// One inference of the batch; returns the output values, row after row.
fn infer(session: &mut Session, input_name: &str, shape: &[i64], data: &[f32]) -> Vec<f32> {
    let tensor = TensorRef::from_array_view((shape.to_vec(), data))
        .unwrap_or_else(|error| panic!("building the input tensor: {error}"));
    let outputs = session
        .run(ort::inputs![input_name => tensor])
        .unwrap_or_else(|error| panic!("inference failed: {error}"));
    let (_, values) = outputs[0]
        .try_extract_tensor::<f32>()
        .unwrap_or_else(|error| panic!("the output is not a 32-bit float tensor: {error}"));
    values.to_vec()
}

#[test]
#[ignore = "hardware gate: needs the AMD runtime files and the NPU; run on request"]
fn the_spike_model_runs_on_the_npu_through_ort() {
    // --- Preconditions: where the hardware proof cannot run, fail (or skip on request). ---
    let runtime_dir = std::env::var_os("L2D_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("runtime").join("ryzenai"));
    if !runtime_dir.is_dir() {
        return skip(&format!(
            "the runtime directory {} does not exist (set L2D_RUNTIME_DIR; see runtime/README.md)",
            runtime_dir.display()
        ));
    }
    let runtime_dir = runtime_dir
        .canonicalize()
        .unwrap_or_else(|error| panic!("resolving {}: {error}", runtime_dir.display()));
    // `canonicalize` gives a `\\?\` path. The plain drive path is used from here on: it is
    // what the spike passed, and the `\\?\` form was not tried with the provider.
    let runtime_dir = PathBuf::from(
        runtime_dir
            .to_string_lossy()
            .trim_start_matches(r"\\?\")
            .to_string(),
    );
    let missing: Vec<&str> = RUNTIME_FILES
        .iter()
        .copied()
        .filter(|name| !runtime_dir.join(name).is_file())
        .collect();
    if !missing.is_empty() {
        return skip(&format!(
            "runtime files missing in {}: {}",
            runtime_dir.display(),
            missing.join(", ")
        ));
    }
    let accelerators = compute_accelerators()
        .unwrap_or_else(|error| panic!("the Windows device query failed: {error}"));
    println!(
        "npu_gate: present devices of the class ComputeAccelerator: {}",
        accelerators.len()
    );
    for (index, found) in accelerators.iter().enumerate() {
        println!(
            "npu_gate:   {}. {} ({}), driver {}{}",
            index + 1,
            found.description,
            found.instance_id,
            found.driver_version,
            if found.is_amd() {
                ""
            } else {
                " - not an AMD device, passed over"
            }
        );
    }
    let Some(device) = accelerators.iter().find(|found| found.is_amd()) else {
        return skip(
            "no present AMD device of the class ComputeAccelerator: this host has no NPU or \
             its driver is not installed",
        );
    };
    println!(
        "npu_gate: NPU device taken (the first AMD one of {}): {} ({})",
        accelerators.iter().filter(|found| found.is_amd()).count(),
        device.description,
        device.instance_id
    );
    println!("npu_gate: NPU driver version: {}", device.driver_version);
    assert!(
        !device.driver_version.is_empty(),
        "the driver version is empty"
    );

    let (pythons, aliases) = python_on_path();
    println!(
        "npu_gate: Python interpreters on PATH: {} {:?}; app execution aliases (stubs): {:?}",
        pythons.len(),
        pythons,
        aliases
    );
    assert!(
        pythons.is_empty() || skip_allowed(),
        "PATH has a Python interpreter, so this run does not show that none is needed: \
         {pythons:?} (run with a PATH without it, or set {ALLOW_SKIP_VARIABLE}=1 to go on)"
    );

    // --- The fixture is the spike's model, byte for byte. ---
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let bytes = std::fs::read(&model)
        .unwrap_or_else(|error| panic!("reading {}: {error}", model.display()));
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    println!(
        "npu_gate: model {} ({} bytes, sha256 {digest})",
        FIXTURE,
        bytes.len()
    );
    assert_eq!(bytes.len() as u64, FIXTURE_BYTES, "fixture size");
    assert_eq!(digest, FIXTURE_SHA256, "fixture hash");

    // --- Provider cache and placement report. ---
    let cache_dir = std::env::var_os("L2D_GATE_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_dir.join("..").join("cache"));
    std::fs::create_dir_all(&cache_dir)
        .unwrap_or_else(|error| panic!("creating {}: {error}", cache_dir.display()));
    let cache_dir = PathBuf::from(
        cache_dir
            .canonicalize()
            .unwrap_or_else(|error| panic!("resolving {}: {error}", cache_dir.display()))
            .to_string_lossy()
            .trim_start_matches(r"\\?\")
            .to_string(),
    );
    let cache_key = format!("gate-2.2-{}", &digest[..12]);
    let model_cache = cache_dir.join(&cache_key);
    let report_path = model_cache.join(REPORT_FILE);
    // The provider writes its report only when it compiles, not when it loads a compiled
    // model from the cache. The gate's own cache entry is therefore removed first, so that
    // the model is compiled, and the report written, by this run.
    if model_cache.is_dir() {
        std::fs::remove_dir_all(&model_cache)
            .unwrap_or_else(|error| panic!("removing {}: {error}", model_cache.display()));
    }
    println!(
        "npu_gate: provider cache {} (emptied: the model is compiled now)",
        model_cache.display()
    );
    // SAFETY: changing the environment is unsound only while another thread reads or
    // writes it. The variable is set before the runtime is loaded and before this test
    // starts any thread of its own. Under `--include-ignored` the test harness may run the
    // other test of this binary, the parser test, on another thread at the same moment;
    // that is harmless because that test and everything it calls (`serde_json` on string
    // literals, assertions) never touch the environment, and the harness itself reads its
    // variables before it starts the tests. A test added to this file that reads or sets
    // an environment variable would break this.
    unsafe { std::env::set_var(REPORT_VARIABLE, REPORT_FILE) };

    // --- Load ONNX Runtime from the runtime directory and open the model on the NPU. ---
    let started = Instant::now();
    let dll = runtime_dir.join("onnxruntime.dll");
    let environment = ort::init_from(&dll)
        .unwrap_or_else(|error| panic!("loading {} failed: {error}", dll.display()))
        .with_name("l2d-gate-2.2")
        .build()
        .unwrap_or_else(|error| panic!("creating the runtime environment failed: {error}"));
    println!(
        "npu_gate: loaded {} through ort at API level {}",
        dll.display(),
        ort::MINOR_VERSION
    );
    println!("npu_gate: runtime build: {}", ort::info());

    let open_on_npu = |profile: Option<&Path>| {
        let provider = Vitis::default()
            .with_cache_dir(cache_dir.display())
            .with_cache_key(&cache_key)
            .with_cache_in_memory(false)
            .build()
            .error_on_failure();
        let mut builder = Session::builder(&environment)
            .unwrap_or_else(|error| panic!("creating the session builder failed: {error}"));
        if let Some(prefix) = profile {
            builder = builder
                .with_profiling(prefix)
                .unwrap_or_else(|error| panic!("enabling profiling failed: {error}"));
        }
        builder
            .with_execution_providers([provider])
            .unwrap_or_else(|error| {
                panic!("registering the Vitis AI execution provider failed: {error}")
            })
            .commit_from_file(&model)
            .unwrap_or_else(|error| {
                panic!("opening the model on the Vitis AI provider failed: {error}")
            })
    };
    let mut session = open_on_npu(None);
    println!(
        "npu_gate: session ready after {:.2} s (runtime load, provider registration, \
         compilation)",
        started.elapsed().as_secs_f64()
    );

    // --- The model's interface, as the runtime reports it. ---
    assert_eq!(session.inputs().len(), 1, "the spike model has one input");
    let input = &session.inputs()[0];
    let input_name = input.name().to_string();
    let shape: Vec<i64> = input
        .dtype()
        .tensor_shape()
        .unwrap_or_else(|| panic!("input {input_name} is not a tensor"))
        .iter()
        .copied()
        .collect();
    println!(
        "npu_gate: input {input_name} {:?} shape {shape:?}; output {} {:?}",
        input.dtype().tensor_type(),
        session.outputs()[0].name(),
        session.outputs()[0].dtype()
    );
    assert_eq!(
        shape[0], BATCH as i64,
        "the fixture has a fixed batch of 32"
    );
    assert!(shape.iter().all(|dim| *dim > 0), "fixed shape expected");
    let elements: usize = shape.iter().map(|dim| *dim as usize).product();

    // --- Inference on a batch of 32. ---
    let mut data = vec![0f32; elements];
    fill(&mut data, 1);
    let first_started = Instant::now();
    let first = infer(&mut session, &input_name, &shape, &data);
    let first_run = first_started.elapsed();
    println!(
        "npu_gate: first inference of a batch of {BATCH}: {:.1} ms; time from start to first \
         result: {:.2} s",
        first_run.as_secs_f64() * 1e3,
        started.elapsed().as_secs_f64()
    );
    assert_eq!(first.len() % BATCH, 0, "output is not a batch of 32");
    let width = first.len() / BATCH;
    assert!(width > 0 && first.iter().all(|value| value.is_finite()));
    assert!(
        first.iter().any(|value| *value != 0.0),
        "the output is all zeros"
    );
    assert!(
        (1..BATCH).any(|row| first[row * width..(row + 1) * width] != first[..width]),
        "all 32 rows are equal although the 32 inputs differ"
    );
    let again = infer(&mut session, &input_name, &shape, &data);
    let repeat_cosine = (0..BATCH)
        .map(|row| {
            cosine(
                &first[row * width..(row + 1) * width],
                &again[row * width..(row + 1) * width],
            )
        })
        .fold(f64::INFINITY, f64::min);
    println!(
        "npu_gate: output {BATCH} x {width}; same input again: bit-identical {}, lowest cosine \
         {repeat_cosine:.6}",
        first == again
    );

    for _ in 0..WARM_UP_BATCHES {
        infer(&mut session, &input_name, &shape, &data);
    }
    let timed = Instant::now();
    for _ in 0..TIMED_BATCHES {
        infer(&mut session, &input_name, &shape, &data);
    }
    let elapsed = timed.elapsed().as_secs_f64();
    println!(
        "npu_gate: {TIMED_BATCHES} batches of {BATCH} in {elapsed:.3} s: {:.2} ms per batch, \
         {:.0} logs/s",
        elapsed * 1e3 / TIMED_BATCHES as f64,
        (TIMED_BATCHES * BATCH) as f64 / elapsed
    );

    // --- Placement, from the report the provider wrote for this session. ---
    let mut files: Vec<String> = std::fs::read_dir(&model_cache)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| {
                    format!(
                        "{} ({} bytes)",
                        entry.file_name().to_string_lossy(),
                        entry.metadata().map(|meta| meta.len()).unwrap_or(0)
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    println!("npu_gate: provider cache holds: {}", files.join(", "));
    let report = std::fs::read_to_string(&report_path).unwrap_or_else(|error| {
        panic!(
            "the provider wrote no placement report at {}: {error}",
            report_path.display()
        )
    });
    let placement = placement_from_report(&report)
        .unwrap_or_else(|error| panic!("{}: {error}", report_path.display()));
    println!(
        "npu_gate: placement from {}: NPU nodes {}, CPU nodes {}, all nodes {}",
        report_path.display(),
        placement.npu_nodes,
        placement.cpu_nodes,
        placement.all_nodes
    );

    // --- A second session finds the compiled model in the cache. ---
    let written = |path: &Path| {
        std::fs::metadata(path)
            .and_then(|meta| meta.modified())
            .ok()
    };
    let report_written = written(&report_path);
    drop(session);
    let reopened = Instant::now();
    // This session is profiled by ONNX Runtime: the profile names, for every node that
    // was executed, the provider that executed it. That is run-time evidence, where the
    // report above is the compiler's plan.
    let profile_prefix = cache_dir.join(format!("{cache_key}-profile"));
    let mut session = open_on_npu(Some(&profile_prefix));
    let reopen = reopened.elapsed();
    let from_cache = infer(&mut session, &input_name, &shape, &data);
    println!(
        "npu_gate: second session, compiled model from the cache: ready after {:.2} s; output \
         bit-identical to the first session: {}; report rewritten: {}",
        reopen.as_secs_f64(),
        from_cache == first,
        written(&report_path) != report_written
    );
    let profile_path = session
        .end_profiling()
        .unwrap_or_else(|error| panic!("ending the profile failed: {error}"));
    drop(session);
    let profile = std::fs::read_to_string(&profile_path)
        .unwrap_or_else(|error| panic!("reading the profile {profile_path}: {error}"));
    let executed = executed_by_provider(&profile)
        .unwrap_or_else(|error| panic!("the profile {profile_path}: {error}"));
    println!("npu_gate: executed nodes by provider, from ONNX Runtime's profile: {executed:?}");
    std::fs::remove_file(&profile_path)
        .unwrap_or_else(|error| panic!("removing {profile_path}: {error}"));

    // --- The same model on the CPU provider of the same runtime, for comparison. ---
    let mut cpu = Session::builder(&environment)
        .unwrap_or_else(|error| panic!("creating the CPU session builder failed: {error}"))
        .commit_from_file(&model)
        .unwrap_or_else(|error| panic!("opening the model on the CPU provider failed: {error}"));
    let on_cpu = infer(&mut cpu, &input_name, &shape, &data);
    let lowest = (0..BATCH)
        .map(|row| {
            cosine(
                &first[row * width..(row + 1) * width],
                &on_cpu[row * width..(row + 1) * width],
            )
        })
        .fold(f64::INFINITY, f64::min);
    for _ in 0..WARM_UP_BATCHES {
        infer(&mut cpu, &input_name, &shape, &data);
    }
    let timed = Instant::now();
    for _ in 0..TIMED_BATCHES {
        infer(&mut cpu, &input_name, &shape, &data);
    }
    let cpu_elapsed = timed.elapsed().as_secs_f64();
    println!(
        "npu_gate: CPU provider, same model: lowest cosine against the NPU output over the \
         {BATCH} rows {lowest:.4}; {:.2} ms per batch, {:.0} logs/s",
        cpu_elapsed * 1e3 / TIMED_BATCHES as f64,
        (TIMED_BATCHES * BATCH) as f64 / cpu_elapsed
    );

    // --- The gate. ---
    assert!(
        placement.npu_nodes > 0,
        "GATE FAILED: the provider placed 0 nodes on the NPU ({placement:?})"
    );
    // Requirement 4.6: the NPU and the CPU agree to a cosine of at least 0.99 per vector.
    assert!(
        lowest >= 0.99,
        "the NPU output is too far from the CPU output of the same model (lowest cosine \
         {lowest}, required 0.99)"
    );
    // And they are not the same numbers: the NPU computes in its own 8-bit arithmetic, so
    // a bit-identical output would mean the "NPU" session was served by the CPU provider.
    assert!(
        first != on_cpu,
        "the NPU session's output is bit-identical to the CPU provider's: the model did \
         not run on the NPU"
    );
    let on_vitis = executed.get(VITIS_PROVIDER).copied().unwrap_or(0);
    assert!(
        on_vitis > 0,
        "ONNX Runtime's profile attributes no executed node to {VITIS_PROVIDER}: {executed:?}"
    );
    println!(
        "npu_gate: PASSED: {} nodes on the NPU, {} on the CPU, {on_vitis} executed nodes \
         attributed to {VITIS_PROVIDER}, driver {}",
        placement.npu_nodes, placement.cpu_nodes, device.driver_version
    );
}
