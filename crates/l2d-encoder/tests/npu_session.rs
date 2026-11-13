//! Hardware tests of `EncoderSession` (task 2.6), through its public interface only: the
//! session opens the package on the NPU itself (runtime check, device, rewrite, provider
//! cache, placement), so nothing here touches `ort`.
//!
//! - `the_fixture_is_encoded_on_the_npu`: the task's done-when. The spike model with the
//!   science gate's byte table, 128-dimensional unit vectors, the placement logged, the
//!   run record.
//! - `a_text_gets_the_same_vector_whatever_its_row_and_its_neighbours`: requirement 4.6
//!   on the trained model, tolerance zero, with images that are not full.
//! - `start_up_cold_and_warm`: requirement 4.7.
//! - `throughput_from_text_to_vector_on_csic`: logs per second end to end over CSIC's
//!   distinct texts, where the time goes, and the memory of the process.
//!
//! No CPU provider runs here or anywhere else, not even for comparison.
//!
//! Run on request, in release, one test at a time (see `npu_support` for the variables):
//!
//! ```text
//! $env:L2D_RUNTIME_DIR = "D:\Source\log2decision\runtime\ryzenai"
//! $env:L2D_SCIENCE_DIR = "D:\Source\log2decision\models\science-2.1"
//! $env:L2D_CSIC_CSV = "D:\Source\log2decision\data\csic-2010\csic_database.csv"
//! $env:L2D_NPU_WORK_DIR = "D:\BuildArtifacts\log2decision\task-2.6"
//! cargo test --release -p l2d-encoder --test npu_session -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(windows)]

mod common;
mod npu_support;

use std::time::Instant;

use l2d_core::{Device, PlacementSource, ProjectPaths, RunKind};
use l2d_encoder::session::provider_model;
use l2d_encoder::{
    EncoderPackage, EncoderSession, InputMode, PackageParts, RuntimeManifest, TextInput,
};
use npu_support::*;

const TAG: &str = "npu_session";
/// The revision of the published model of task 2.1 as a package in `payload` mode, and
/// the checksum task 2.12 printed for the vectors of CSIC's 25,608 distinct texts at 64
/// rows with float input (`final-throughput.log` of that task, every setting).
const PUBLISHED_CHECKSUM: &str = "1570811.2676";

fn open(package: &EncoderPackage, paths: &ProjectPaths) -> EncoderSession {
    let session = EncoderSession::open(package, paths).unwrap_or_else(|error| panic!("{error}"));
    let info = session.info();
    println!(
        "{TAG}: encoder {} ({} mode) open on the NPU after {:.2} s (runtime load {:.2} s, session \
         {:.2} s; provider compiled now: {}); AMD Ryzen AI {} (ONNX Runtime {}, source `{}`); \
         device `{}` matched by {:?}, driver {}; cache key {}",
        info.identity,
        info.input_mode.as_str(),
        info.open_seconds,
        info.load_seconds,
        info.session_seconds,
        info.compiled,
        info.runtime_release,
        info.onnxruntime,
        info.runtime_source,
        info.npu.description,
        info.npu_match,
        info.driver_version.as_deref().unwrap_or("unknown"),
        info.cache_key
    );
    println!(
        "{TAG}: placement: {} of {} nodes on the NPU, {} on the CPU: {:?}; subgraphs: {:?}; \
         first run by ONNX Runtime's profile: through the Vitis AI provider {:?}, \
         elsewhere {:?}; session buffers {} bytes",
        info.placement.npu_nodes,
        info.placement.all_nodes,
        info.placement.cpu_nodes,
        info.placement.cpu_operators(),
        info.placement.subgraphs,
        info.executed.on_provider,
        info.executed.elsewhere,
        info.held_bytes
    );
    session
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn length(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn checksum(vectors: &[f32]) -> f64 {
    vectors
        .iter()
        .enumerate()
        .map(|(at, value)| f64::from(*value) * (1.0 + (at % 97) as f64))
        .sum()
}

fn published(paths: &ProjectPaths) -> Option<EncoderPackage> {
    let science = directory_from(TAG, "L2D_SCIENCE_DIR", "the published model of task 2.1")?;
    Some(published_package_in(paths, &science))
}

#[test]
#[ignore = "hardware test: needs the AMD runtime files and the NPU; run on request"]
fn the_fixture_is_encoded_on_the_npu() {
    let Some(paths) = session_project(TAG) else {
        return;
    };
    let parts = common::Parts::spike();
    let written = EncoderPackage::write(
        &paths,
        &PackageParts {
            name: "spike-fixture",
            ..parts.parts()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let package = written.package;
    let mut session = open(&package, &paths);
    assert_eq!(session.device(), Device::Npu);
    let info = session.info().clone();
    assert_eq!(
        (
            info.placement.npu_nodes,
            info.placement.cpu_nodes,
            info.placement.all_nodes
        ),
        (33, 2, 35),
        "all nodes but the input's QuantizeLinear and the output's DequantizeLinear on the NPU"
    );
    assert_eq!(
        info.placement.cpu_operators(),
        ["QuantizeLinear", "DequantizeLinear"]
    );
    // And the run itself, by ONNX Runtime's profile of the session's first run.
    assert!(!info.executed.on_provider.is_empty(), "{:?}", info.executed);
    assert_eq!(info.executed.npu_only(), Ok(()));
    // No profile is left behind in the cache directory.
    let left: Vec<String> = std::fs::read_dir(paths.runtime_cache_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains("first-run"))
        .collect();
    assert!(left.is_empty(), "profiles left behind: {left:?}");
    assert_eq!(info.rows, 64);
    assert_eq!(info.runtime_release, RuntimeManifest::expected().release);
    assert_eq!(info.held_bytes, 4_409_856);
    // The compiled model and its report are below the project's `runtime/cache/`.
    assert_eq!(
        info.cache_folder,
        paths.runtime_cache_dir().join(&info.cache_key)
    );
    assert!(info.cache_folder.join(REPORT_FILE).is_file());
    assert!(info.open_seconds < 30.0, "requirement 4.7");

    // 70 texts: one full image and one of six rows. Short and long payloads, a text
    // without payload, the empty text, a routing part longer than the input.
    let long_path = format!(
        "GET http://localhost:8080/{}/index.jsp?id=2",
        "tienda1/".repeat(40)
    );
    let mut texts: Vec<String> = (0..66)
        .map(|n| {
            format!(
                "GET http://localhost:8080/tienda1/publico/anadir.jsp?id={n}&nombre=Vino+Rioja&\
                 precio={}&cantidad={}&B1=A%F1adir+al+carrito",
                n * 7,
                "9".repeat(n * 5)
            )
        })
        .collect();
    texts.push(String::new());
    texts.push("Nov 13 10:02:11 host sshd[411]: Failed password for root".into());
    texts.push(long_path.clone());
    texts.push("POST http://localhost:8080/tienda1/login.jsp\nlogin=a' OR '1'='1&pwd=x".into());
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encoded = session
        .encode(&refs)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(encoded.len(), 70);
    for (text, entry) in refs.iter().zip(&encoded) {
        assert_eq!(entry.vector.len(), 128);
        assert_eq!(entry.vector.len(), usize::from(info.identity.dimension));
        let length = length(&entry.vector);
        assert!(
            (length - 1.0).abs() < 1e-6,
            "the vector of {text:?} has length {length}"
        );
        assert_eq!(entry.input, package.codec().describe(text));
    }
    // The two flags of requirement 1.5 (payload mode, 256 bytes).
    let flags = |index: usize| {
        let input: TextInput = encoded[index].input;
        (input.truncated, input.text_longer_than_input, input.empty)
    };
    assert_eq!(flags(0), (false, false, false));
    assert_eq!(
        flags(65),
        (true, true, false),
        "a payload of over 256 bytes"
    );
    assert_eq!(flags(66), (false, false, true), "the empty text");
    assert_eq!(flags(67), (false, false, true), "no payload");
    assert_eq!(flags(68), (false, true, false), "a long routing part");
    assert_eq!(flags(69), (false, false, false));
    // Texts whose input is empty share one vector; other texts do not have it.
    assert_eq!(bits(&encoded[66].vector), bits(&encoded[67].vector));
    // A text has its vector in whatever call, row and company it comes.
    let again = session.encode(&[refs[69], refs[3]]).unwrap();
    assert_eq!(bits(&again[0].vector), bits(&encoded[69].vector));
    assert_eq!(bits(&again[1].vector), bits(&encoded[3].vector));
    let distinct: std::collections::HashSet<Vec<u32>> =
        encoded.iter().map(|entry| bits(&entry.vector)).collect();
    println!(
        "{TAG}: 70 fixture texts gave {} distinct vectors of 128 values, each of length 1 \
         (the spike model has random weights: its vectors lie close together)",
        distinct.len()
    );

    let stats = *session.stats();
    assert_eq!((stats.texts, stats.images), (72, 3));
    let record = session.finish();
    assert_eq!(record.kind, RunKind::Encode);
    assert_eq!(record.encoder.as_ref(), Some(&info.identity));
    assert_eq!(record.device, Some(Device::Npu));
    assert_eq!(record.driver_version, info.driver_version);
    assert!(
        record.driver_version.is_some(),
        "the driver version was not read"
    );
    let placement = record.placement.expect("the placement");
    assert_eq!((placement.npu_nodes, placement.cpu_nodes), (33, 2));
    assert_eq!(placement.source, PlacementSource::ProviderReport);
    assert!(record.logs_per_second.is_some_and(|rate| rate > 0.0));
    assert_eq!(record.settings["runtime"]["release"], "1.9.0");
    assert_eq!(record.settings["npu"]["matched_by"], "known_device_id");
    println!(
        "{TAG}: run record:\n{}",
        record.to_json().unwrap_or_else(|error| panic!("{error}"))
    );
}

#[test]
#[ignore = "hardware test: needs the AMD runtime files, the NPU and the trained model; run on request"]
fn a_text_gets_the_same_vector_whatever_its_row_and_its_neighbours() {
    let Some(paths) = session_project(TAG) else {
        return;
    };
    let Some(package) = published(&paths) else {
        return;
    };
    assert_eq!(package.input_mode(), InputMode::Payload);
    let codec = package.codec().clone();
    let texts: Vec<String> = csic_logs()
        .into_iter()
        .filter(|log| !codec.describe(&log.text).empty)
        .map(|log| log.text)
        .collect();
    let probes: Vec<&str> = (0..677)
        .map(|i| texts[(i * (texts.len() / 677) + 5) % texts.len()].as_str())
        .collect();
    let distinct: std::collections::HashSet<&str> = probes.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        probes.len(),
        "the probes are distinct texts"
    );
    let mut session = open(&package, &paths);
    assert_eq!(
        (
            session.info().placement.npu_nodes,
            session.info().placement.cpu_nodes
        ),
        (33, 2)
    );
    let dimension = usize::from(package.identity().dimension);
    let mut flags = Vec::new();

    // The reference: 677 texts in one call, ten full images and one of 37 rows.
    let mut reference = Vec::new();
    session
        .encode_into(&probes, &mut reference, &mut flags)
        .unwrap_or_else(|error| panic!("{error}"));
    let of = |index: usize| bits(&reference[index * dimension..(index + 1) * dimension]);
    let different: std::collections::HashSet<Vec<u32>> = (0..probes.len()).map(of).collect();
    println!(
        "{TAG}: {} probes gave {} different vectors",
        probes.len(),
        different.len()
    );
    assert!(
        different.len() > probes.len() / 2,
        "the probes must tell rows apart: only {} different vectors",
        different.len()
    );

    // 1. The same texts in another order: every text in another row, among other texts,
    //    and the image that is not full holds other texts.
    let mut wrong = 0usize;
    for shift in [1usize, 13, 64, 331] {
        let rotated: Vec<&str> = (0..probes.len())
            .map(|i| probes[(i + shift) % probes.len()])
            .collect();
        let mut vectors = Vec::new();
        session
            .encode_into(&rotated, &mut vectors, &mut flags)
            .unwrap();
        for i in 0..probes.len() {
            let from = (i + shift) % probes.len();
            wrong += usize::from(bits(&vectors[i * dimension..(i + 1) * dimension]) != of(from));
        }
    }
    println!(
        "{TAG}: 4 other orders of the 677 texts: {wrong} of {} vectors differ from the reference",
        4 * probes.len()
    );
    assert_eq!(
        wrong, 0,
        "a vector depends on the row or on the other texts"
    );

    // 2. Each of 128 texts alone: an image of one text and 63 empty rows.
    let mut alone_wrong = 0usize;
    for index in (0..probes.len()).step_by(probes.len() / 128).take(128) {
        let encoded = session.encode(&[probes[index]]).unwrap();
        alone_wrong += usize::from(bits(&encoded[0].vector) != of(index));
    }
    println!("{TAG}: 128 texts alone: {alone_wrong} differ from the reference");
    assert_eq!(
        alone_wrong, 0,
        "a text alone has another vector than among others"
    );

    // 3. One text in every row of a full image, and images of every size from 1 to 64
    //    filled with the same text.
    let probe = probes[200];
    let expected = of(200);
    let mut row_wrong = 0usize;
    for row in 0..64 {
        let mut image: Vec<&str> = probes[300..363].to_vec();
        image.insert(row, probe);
        let encoded = session.encode(&image).unwrap();
        row_wrong += usize::from(bits(&encoded[row].vector) != expected);
    }
    for size in 1..=64 {
        let encoded = session.encode(&vec![probe; size]).unwrap();
        row_wrong += encoded
            .iter()
            .filter(|entry| bits(&entry.vector) != expected)
            .count();
    }
    println!(
        "{TAG}: one text in each of 64 rows and in images of 1 to 64 identical rows: \
         {row_wrong} vectors differ"
    );
    assert_eq!(row_wrong, 0);

    // 4. A long image after a short one and back: rows that held a text are empty again.
    let empty_before = bits(&session.encode(&[""]).unwrap()[0].vector);
    session.encode(&probes[..64]).unwrap();
    let after = session.encode(&["", probes[7]]).unwrap();
    assert_eq!(bits(&after[0].vector), empty_before);
    assert_eq!(bits(&after[1].vector), of(7));
    println!(
        "{TAG}: requirement 4.6 holds with tolerance zero over {} encoded texts",
        session.stats().texts
    );
}

/// The placement is read from the report in the cache folder. A folder that holds a
/// compiled model and no report must not give a session: nothing would show that the
/// model runs on the NPU.
#[test]
#[ignore = "hardware test: needs the AMD runtime files and the NPU; run on request"]
fn a_compiled_model_without_its_report_gives_no_session() {
    let Some(paths) = session_project(TAG) else {
        return;
    };
    let parts = common::Parts::spike();
    let package = EncoderPackage::write(
        &paths,
        &PackageParts {
            name: "spike-fixture",
            ..parts.parts()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"))
    .package;
    let first = open(&package, &paths);
    let folder = first.info().cache_folder.clone();
    let report = folder.join(REPORT_FILE);
    drop(first);
    std::fs::remove_file(&report).unwrap_or_else(|error| panic!("{error}"));

    let outcome = EncoderSession::open(&package, &paths);
    match &outcome {
        Ok(session) => println!(
            "{TAG}: without the report the provider wrote a new one: compiled now: {}, {} of {} \
             nodes on the NPU",
            session.info().compiled,
            session.info().placement.npu_nodes,
            session.info().placement.all_nodes
        ),
        Err(error) => println!("{TAG}: without the report: {error}"),
    }
    match outcome {
        Ok(session) => {
            assert!(
                report.is_file(),
                "a session was opened without a placement report"
            );
            assert_eq!(session.info().placement.npu_nodes, 33);
        }
        Err(error) => {
            let shown = error.to_string();
            assert!(
                matches!(error, l2d_encoder::EncoderError::NpuUnavailable { .. })
                    && shown.contains(&folder.display().to_string())
                    && shown.contains("remove the folder"),
                "{shown}"
            );
            // What the message says to do works: without the folder the provider
            // compiles again and the session opens.
            std::fs::remove_dir_all(&folder).unwrap_or_else(|error| panic!("{error}"));
            let again = open(&package, &paths);
            assert!(again.info().compiled && report.is_file());
        }
    }
}

/// The report in a cache folder is the report of the compilation that filled it. What
/// if the compiled model beside it is damaged or gone: does the provider compile again
/// (and write a new report), fail, or run the model somewhere else under the old report?
#[test]
#[ignore = "hardware test: needs the AMD runtime files and the NPU; run on request"]
fn a_damaged_cache_folder_does_not_give_a_session_off_the_npu() {
    let Some(paths) = session_project(TAG) else {
        return;
    };
    let parts = common::Parts::spike();
    let package = EncoderPackage::write(
        &paths,
        &PackageParts {
            name: "spike-fixture",
            ..parts.parts()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"))
    .package;
    let texts: Vec<String> = (0..64)
        .map(|n| format!("GET /tienda1/publico/anadir.jsp?id={n}&precio={}", n * 31))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    // Vectors and the seconds of 50 images.
    let measure = |session: &mut EncoderSession| {
        let vectors: Vec<Vec<u32>> = session
            .encode(&refs)
            .unwrap_or_else(|error| panic!("{error}"))
            .iter()
            .map(|entry| bits(&entry.vector))
            .collect();
        let started = Instant::now();
        for _ in 0..50 {
            session.encode(&refs).unwrap();
        }
        (vectors, started.elapsed().as_secs_f64())
    };
    // Start from a folder the provider fills now, whatever an earlier run left there.
    let key = provider_model(&package, &RuntimeManifest::expected().release)
        .unwrap_or_else(|error| panic!("{error}"))
        .cache_key;
    let _ = std::fs::remove_dir_all(paths.runtime_cache_dir().join(key));
    let mut first = open(&package, &paths);
    assert!(first.info().compiled);
    let folder = first.info().cache_folder.clone();
    let (right, seconds) = measure(&mut first);
    drop(first);
    let files: Vec<std::path::PathBuf> = std::fs::read_dir(&folder)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    println!(
        "{TAG}: the cache folder holds {:?}; 50 images take {seconds:.3} s",
        files
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
    );
    let compiled: Vec<&std::path::PathBuf> = files
        .iter()
        .filter(|path| path.file_name().unwrap() != REPORT_FILE)
        .collect();
    assert!(!compiled.is_empty(), "nothing but the report in the folder");

    for file in compiled {
        let name = file.file_name().unwrap().to_string_lossy().into_owned();
        let original = std::fs::read(file).unwrap();
        for (damage, content) in [
            ("removed", None),
            ("filled with zeros", Some(vec![0u8; original.len()])),
            (
                "cut to its first half",
                Some(original[..original.len() / 2].to_vec()),
            ),
        ] {
            match &content {
                None => std::fs::remove_file(file).unwrap(),
                Some(content) => std::fs::write(file, content).unwrap(),
            }
            match EncoderSession::open(&package, &paths) {
                Err(error) => {
                    let shown = error.to_string();
                    println!(
                        "{TAG}: {name} {damage}: no session: {}",
                        &shown[..shown.len().min(300)]
                    );
                }
                Ok(mut session) => {
                    let (vectors, took) = measure(&mut session);
                    println!(
                        "{TAG}: {name} {damage}: a session opened; provider compiled now: {}; \
                         vectors bit-identical to the undamaged folder's: {}; 50 images take \
                         {took:.3} s (undamaged {seconds:.3} s)",
                        session.info().compiled,
                        vectors == right
                    );
                    // A session that opened must be the model on the NPU: the provider
                    // compiled again and reported, or the damage did not matter. Either
                    // way the vectors are those of the NPU.
                    assert!(
                        vectors == right,
                        "{name} {damage}: a session opened and gives other vectors than the \
                         NPU's: it does not run the compiled model"
                    );
                }
            }
            // Back to an undamaged folder: the provider compiles anew.
            std::fs::remove_dir_all(&folder).unwrap();
            let fresh = open(&package, &paths);
            assert!(fresh.info().compiled);
        }
    }
}

#[test]
#[ignore = "hardware measurement: needs the AMD runtime files, the NPU and the trained model; run on request"]
fn start_up_cold_and_warm() {
    let process_started = Instant::now();
    let Some(paths) = session_project(TAG) else {
        return;
    };
    let Some(package) = published(&paths) else {
        return;
    };
    let texts: Vec<String> = csic_logs()
        .into_iter()
        .take(64)
        .map(|log| log.text)
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let key = provider_model(&package, &RuntimeManifest::expected().release)
        .unwrap_or_else(|error| panic!("{error}"))
        .cache_key;
    let folder = paths.runtime_cache_dir().join(&key);
    // `L2D_NPU_WARM_ONLY`: keep the cache, to time a warm start in a fresh process.
    let warm_only = std::env::var_os("L2D_NPU_WARM_ONLY").is_some();
    if !warm_only && folder.is_dir() {
        std::fs::remove_dir_all(&folder)
            .unwrap_or_else(|error| panic!("removing {}: {error}", folder.display()));
    }
    println!(
        "{TAG}: before the first open (CSIC read, package verified): {:.2} s since the start of \
         the test; cache folder {} {}",
        process_started.elapsed().as_secs_f64(),
        folder.display(),
        if folder.is_dir() {
            "exists"
        } else {
            "is absent"
        }
    );

    let started = Instant::now();
    let mut first = open(&package, &paths);
    let ready = started.elapsed().as_secs_f64();
    let right: Vec<Vec<u32>> = first
        .encode(&refs)
        .unwrap()
        .iter()
        .map(|entry| bits(&entry.vector))
        .collect();
    println!(
        "{TAG}: {} start in a fresh process: ready after {ready:.2} s (open {:.2} s: runtime \
         load {:.2} s, session {:.2} s), first 64 vectors after {:.2} s; compiled now: {}",
        if warm_only { "warm" } else { "cold" },
        first.info().open_seconds,
        first.info().load_seconds,
        first.info().session_seconds,
        started.elapsed().as_secs_f64(),
        first.info().compiled
    );
    assert_eq!(
        first.info().compiled,
        !warm_only,
        "cold compiles, warm does not"
    );
    assert!(
        ready < 30.0,
        "requirement 4.7: ready within 30 s, took {ready:.2} s"
    );
    drop(first);

    for attempt in 1..=3 {
        let started = Instant::now();
        let mut warm = open(&package, &paths);
        let ready = started.elapsed().as_secs_f64();
        let vectors: Vec<Vec<u32>> = warm
            .encode(&refs)
            .unwrap()
            .iter()
            .map(|entry| bits(&entry.vector))
            .collect();
        println!(
            "{TAG}: warm start {attempt} in the same process: ready after {ready:.2} s, first 64 \
             vectors after {:.2} s; compiled now: {}; vectors bit-identical to the first start: {}",
            started.elapsed().as_secs_f64(),
            warm.info().compiled,
            vectors == right
        );
        assert!(!warm.info().compiled, "a warm start must not compile");
        assert_eq!(warm.info().load_seconds, 0.0, "the runtime is loaded once");
        assert!(
            vectors == right,
            "a warm start must give the first start's vectors"
        );
    }
}

#[test]
#[ignore = "hardware measurement: needs the AMD runtime files, the NPU and the trained model; run on request"]
fn throughput_from_text_to_vector_on_csic() {
    let Some(paths) = session_project(TAG) else {
        return;
    };
    let Some(package) = published(&paths) else {
        return;
    };
    let texts: Vec<String> = csic_logs().into_iter().map(|log| log.text).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let logs = refs.len();
    let dimension = usize::from(package.identity().dimension);
    println!(
        "{TAG}: {logs} distinct CSIC texts ({} bytes)",
        texts.iter().map(String::len).sum::<usize>()
    );
    let memory = |what: &str| {
        let (working, peak, commit) = process_memory();
        println!(
            "{TAG}: process memory {what}: working set {:.1} MiB, peak {:.1} MiB, private commit \
             {:.1} MiB",
            mib(working),
            mib(peak),
            mib(commit)
        );
        (working, commit)
    };
    let (working_before, commit_before) = memory("before the session is opened");
    let mut session = open(&package, &paths);
    let (working_open, commit_open) = memory("with the session open");
    println!(
        "{TAG}: opening the session added {:.1} MiB working set and {:.1} MiB private commit; \
         the session's own buffers are {} bytes ({:.1} MiB) of that",
        mib(working_open.saturating_sub(working_before)),
        mib(commit_open.saturating_sub(commit_before)),
        session.info().held_bytes,
        mib(session.info().held_bytes)
    );

    // One pass that is not timed, then five that are, into buffers that are reused.
    let mut vectors: Vec<f32> = Vec::with_capacity(logs * dimension);
    let mut flags: Vec<TextInput> = Vec::with_capacity(logs);
    session
        .encode_into(&refs, &mut vectors, &mut flags)
        .unwrap_or_else(|error| panic!("{error}"));
    let sum = checksum(&vectors);
    let first = bits(&vectors);
    let truncated = flags.iter().filter(|input| input.truncated).count();
    let longer = flags
        .iter()
        .filter(|input| input.text_longer_than_input)
        .count();
    let empty = flags.iter().filter(|input| input.empty).count();
    println!(
        "{TAG}: flags over the {logs} texts ({} mode): truncated {truncated}, whole text longer \
         than the input {longer}, empty input {empty}",
        package.input_mode().as_str()
    );
    println!(
        "{TAG}: checksum of the vectors {sum:.4} (task 2.12, same texts and model, row writer \
         of that task: {PUBLISHED_CHECKSUM})"
    );
    assert_eq!(
        format!("{sum:.4}"),
        PUBLISHED_CHECKSUM,
        "the session's vectors are not those task 2.12 measured for the published model"
    );

    let mut rates = Vec::new();
    let before = *session.stats();
    let cpu_before = process_cpu_seconds();
    for _ in 0..5 {
        vectors.clear();
        flags.clear();
        let started = Instant::now();
        session
            .encode_into(&refs, &mut vectors, &mut flags)
            .unwrap();
        rates.push(logs as f64 / started.elapsed().as_secs_f64());
        assert!(bits(&vectors) == first, "a pass gave other vectors");
    }
    let cpu = process_cpu_seconds() - cpu_before;
    let after = *session.stats();
    rates.sort_by(f64::total_cmp);
    let timed = (after.texts - before.texts) as f64;
    let per_log = |seconds: f64| seconds * 1e6 / timed;
    let (write, run, read) = (
        after.write_seconds - before.write_seconds,
        after.run_seconds - before.run_seconds,
        after.read_seconds - before.read_seconds,
    );
    let inside = after.seconds - before.seconds;
    println!(
        "{TAG}: encode_into, 5 passes over {logs} texts: median {:.0} logs/s (range {:.0} to \
         {:.0}); by the session's own clock {:.0} logs/s",
        rates[2],
        rates[0],
        rates[4],
        timed / inside
    );
    println!(
        "{TAG}: per log: {:.1} us in all; writing the image {:.2} us ({:.1} %), the run {:.2} us \
         ({:.1} %), reading and normalising {:.2} us ({:.1} %), handing the vectors over {:.2} \
         us; process CPU {:.1} s per million logs",
        per_log(inside),
        per_log(write),
        100.0 * write / inside,
        per_log(run),
        100.0 * run / inside,
        per_log(read),
        100.0 * read / inside,
        per_log(inside - write - run - read),
        cpu * 1e6 / timed
    );
    assert!(
        rates[0] >= 1_000.0,
        "requirement 4.2: at least 1,000 logs per second, measured {:.0}",
        rates[0]
    );

    // Sustained: four passes in one call, 102,432 texts.
    let four: Vec<&str> = refs.iter().cycle().take(4 * logs).copied().collect();
    let mut long_vectors: Vec<f32> = Vec::with_capacity(4 * logs * dimension);
    flags.clear();
    let started = Instant::now();
    session
        .encode_into(&four, &mut long_vectors, &mut flags)
        .unwrap();
    println!(
        "{TAG}: sustained over {} texts in one call: {:.0} logs/s",
        four.len(),
        four.len() as f64 / started.elapsed().as_secs_f64()
    );
    drop(long_vectors);

    // The interface that returns one entry per text (a vector allocated per text).
    let started = Instant::now();
    let encoded = session.encode(&refs).unwrap();
    println!(
        "{TAG}: encode (one `Encoded` per text): {:.0} logs/s",
        logs as f64 / started.elapsed().as_secs_f64()
    );
    assert!(
        encoded
            .iter()
            .zip(vectors.chunks_exact(dimension))
            .all(|(entry, vector)| bits(&entry.vector) == bits(vector)),
        "encode and encode_into give other vectors"
    );
    drop(encoded);

    let (working_end, commit_end) = memory("after encoding");
    println!(
        "{TAG}: encoding added {:.1} MiB working set and {:.1} MiB private commit over the open \
         session (the caller's vectors of one pass are {:.1} MiB of it)",
        mib(working_end.saturating_sub(working_open)),
        mib(commit_end.saturating_sub(commit_open)),
        mib((logs * dimension * 4) as u64)
    );
    let record = session.finish();
    println!(
        "{TAG}: run record: {:.0} logs/s over {} texts in {} images; placement {:?}; driver {:?}",
        record.logs_per_second.unwrap_or(0.0),
        record.settings["encoding"]["texts"],
        record.settings["encoding"]["images"],
        record.placement,
        record.driver_version
    );
}
