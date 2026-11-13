//! Task 2.12, part 2: logs per second on the NPU from log text to L2-normalised vector,
//! with the trained model of task 2.1 on CSIC's distinct texts, in the layout in which
//! every row is right (the texts as rows of one image, `npu_rows.rs`).
//!
//! `throughput_from_text_to_vector` measures one setting after the other and prints a
//! table row for each: logs per second (median and range over the passes), where the
//! time goes, CPU seconds beside the NPU, memory. `L2D_NPU_SETTINGS` (comma-separated
//! parts of setting names) selects settings, `L2D_NPU_PASSES` the timed passes (default
//! 5). `start_up_and_the_provider_cache` measures cold and warm starts and what the
//! cache key must hold. Run on request, in release:
//!
//! ```text
//! cargo test --release -p l2d-encoder --test npu_throughput -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(windows)]

mod npu_support;

use std::time::Instant;

use l2d_encoder::encode::{ImageInput, dequantize_uint8, read_rows};
use l2d_encoder::onnx_edit::{self, QuantizedEnds};
use l2d_encoder::{ByteCodec, ModelInterface};
use npu_support::*;
use ort::memory::Allocator;
use ort::value::{Tensor, TensorRef};

const TAG: &str = "npu_throughput";
/// The rows per image the other settings are compared at.
const ROWS: usize = 64;

struct Fixture {
    rig: Rig,
    model: Vec<u8>,
    interface: ModelInterface,
    codec: ByteCodec,
    texts: Vec<String>,
}

fn fixture() -> Option<Fixture> {
    let rig = Rig::prepare(TAG)?;
    let science = directory_from(TAG, "L2D_SCIENCE_DIR", "the published model of task 2.1")?;
    let package = published_package(&rig, &science);
    let model = package
        .read_model()
        .unwrap_or_else(|error| panic!("{error}"));
    let texts: Vec<String> = csic_logs().into_iter().map(|log| log.text).collect();
    println!(
        "{TAG}: encoder {}, {} distinct CSIC texts ({} bytes)",
        package.identity(),
        texts.len(),
        texts.iter().map(String::len).sum::<usize>()
    );
    Some(Fixture {
        rig,
        interface: package.model().clone(),
        codec: package.codec().clone(),
        model,
        texts,
    })
}

/// One measured setting: what a pass over the texts cost.
struct Measured {
    name: String,
    ready_seconds: f64,
    placement: String,
    /// Logs per second of each timed pass.
    rates: Vec<f64>,
    stages: Stages,
    cpu_seconds_per_million: f64,
    checksum: f64,
}

fn report(measured: &Measured, logs_per_pass: usize) {
    let mut rates = measured.rates.clone();
    rates.sort_by(f64::total_cmp);
    let (working, peak, commit) = process_memory();
    let share = |part: f64| 100.0 * part / measured.stages.total().max(1e-12);
    println!(
        "{TAG}: | {} | {:.0} | {:.0} to {:.0} | {:.0} | {:.1} / {:.1} / {:.1} / {:.1} | {:.1} | \
         {} | {:.2} | {:.0} / {:.0} / {:.0} | {:.4} |",
        measured.name,
        rates[rates.len() / 2],
        rates[0],
        rates[rates.len() - 1],
        measured.stages.total() * 1e9 / (logs_per_pass * measured.rates.len()) as f64,
        share(measured.stages.codec),
        share(measured.stages.hand),
        share(measured.stages.run),
        share(measured.stages.read),
        measured.cpu_seconds_per_million,
        measured.placement,
        measured.ready_seconds,
        mib(working),
        mib(peak),
        mib(commit),
        measured.checksum
    );
}

fn passes() -> usize {
    std::env::var("L2D_NPU_PASSES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5)
}

fn selected(name: &str) -> bool {
    match std::env::var("L2D_NPU_SETTINGS") {
        Ok(list) if !list.is_empty() => list.split(',').any(|part| name.contains(part.trim())),
        _ => true,
    }
}

fn placement_text(placement: &Option<Placement>) -> String {
    placement
        .as_ref()
        .map(|p| format!("{} NPU, {} CPU", p.npu_nodes, p.cpu_nodes))
        .unwrap_or_else(|| "no report".into())
}

/// Times `pass` (one pass over all texts; it returns the vectors' checksum) after one
/// untimed pass.
fn timed(
    name: &str,
    ready_seconds: f64,
    placement: String,
    logs: usize,
    mut pass: impl FnMut(&mut Stages) -> f64,
) -> Measured {
    let checksum = pass(&mut Stages::default());
    let mut stages = Stages::default();
    let mut rates = Vec::new();
    let cpu_before = process_cpu_seconds();
    for _ in 0..passes() {
        let started = Instant::now();
        let again = pass(&mut stages);
        rates.push(logs as f64 / started.elapsed().as_secs_f64());
        assert_eq!(again, checksum, "{name}: a pass gave other vectors");
    }
    let cpu = process_cpu_seconds() - cpu_before;
    Measured {
        name: name.to_string(),
        ready_seconds,
        placement,
        cpu_seconds_per_million: cpu * 1e6 / (logs * rates.len()) as f64,
        rates,
        stages,
        checksum,
    }
}

fn checksum(vectors: &[f32]) -> f64 {
    vectors
        .iter()
        .enumerate()
        .map(|(at, value)| f64::from(*value) * (1.0 + (at % 97) as f64))
        .sum()
}

#[test]
#[ignore = "hardware measurement: needs the AMD runtime files, the NPU and the trained model; run on request"]
fn throughput_from_text_to_vector() {
    let Some(fixture) = fixture() else { return };
    let texts: Vec<&str> = fixture.texts.iter().map(String::as_str).collect();
    let logs = texts.len();
    println!(
        "{TAG}: {} timed passes over {logs} texts per setting; columns: setting | logs/s \
         median | range | ns per log in the stages | share codec / hand / run / read (%) | \
         process CPU s per million logs | placement | ready (s) | working set / peak / \
         commit (MiB) | checksum",
        passes()
    );
    let width = fixture.interface.output.shape[1] as usize;

    // --- The rows per image: float input, a tensor that borrows the buffer per run. ---
    let mut reference: Option<Vec<f32>> = None;
    for rows in [1usize, 8, 16, 32, 48, 64, 96, 128, 256] {
        let name = format!("rows {rows}, float input");
        if !selected(&name) {
            continue;
        }
        let key = format!("throughput-h{rows}");
        let mut encoder = HeightEncoder::open(
            &fixture.rig,
            &fixture.model,
            &fixture.codec,
            rows,
            &Open {
                cache_key: &key,
                fresh: true,
                ..Open::default()
            },
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut vectors: Vec<f32> = Vec::with_capacity(logs * width);
        let placement = placement_text(&encoder.placement);
        let ready = encoder.ready_seconds;
        let measured = timed(&name, ready, placement, logs, |stages| {
            vectors.clear();
            encoder.encode(&texts, &mut vectors, stages);
            checksum(&vectors)
        });
        report(&measured, logs);
        // Every setting must give the same vectors, bit for bit.
        match &reference {
            None => reference = Some(vectors.clone()),
            Some(first) => assert!(
                first == &vectors,
                "{name}: other vectors than the first setting"
            ),
        }
    }

    // --- The quantized input fed directly; then the quantized output read directly. ---
    for both in [false, true] {
        let name = if both {
            format!("rows {ROWS}, uint8 input and output")
        } else {
            format!("rows {ROWS}, uint8 input")
        };
        if !selected(&name) {
            continue;
        }
        let image_model = onnx_edit::with_rows_as_height(&fixture.model, ROWS as u32).unwrap();
        let edited = onnx_edit::with_quantized_ends(
            &image_model,
            QuantizedEnds {
                input: true,
                output: both,
            },
        )
        .unwrap();
        let key = format!("throughput-h{ROWS}-u8{}", if both { "-both" } else { "" });
        let opened = fixture
            .rig
            .open(
                Model::Bytes(&edited),
                &Open {
                    cache_key: &key,
                    fresh: true,
                    ..Open::default()
                },
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let placement = placement_text(&opened.placement);
        let ready = opened.seconds;
        let mut session = opened.session;
        let (input_q, output_q) = (
            &fixture.interface.input_quantization,
            &fixture.interface.output_quantization,
        );
        let mut image =
            ImageInput::quantized(&fixture.codec, input_q.scale, input_q.zero_point as u8);
        let mut values = vec![image.zero(); image.image_values(ROWS)];
        let arch = fixture.codec.arch();
        let dims = vec![
            1,
            i64::from(arch.channels),
            ROWS as i64,
            i64::from(arch.input_bytes),
        ];
        let mut floats = vec![0f32; ROWS * width];
        let mut vectors: Vec<f32> = Vec::with_capacity(logs * width);
        let mut written = 0usize;
        let measured = timed(&name, ready, placement, logs, |stages| {
            vectors.clear();
            for chunk in texts.chunks(ROWS) {
                let started = Instant::now();
                for (row, text) in chunk.iter().enumerate() {
                    image.write(&fixture.codec, text, ROWS, row, &mut values);
                }
                for row in chunk.len()..written {
                    image.clear_row(ROWS, row, &mut values);
                }
                written = chunk.len();
                let filled = Instant::now();
                let tensor = TensorRef::from_array_view((dims.clone(), &values[..])).unwrap();
                let handed = Instant::now();
                let outputs = session
                    .run(ort::inputs![input_q.tensor.as_str() => tensor])
                    .unwrap();
                let ran = Instant::now();
                let at = vectors.len();
                vectors.resize(at + chunk.len() * width, 0.0);
                if both {
                    let (_, bytes) = outputs[0].try_extract_tensor::<u8>().unwrap();
                    for (value, byte) in floats.iter_mut().zip(bytes) {
                        *value = dequantize_uint8(*byte, output_q.scale, output_q.zero_point as u8);
                    }
                    read_rows(&floats, ROWS, chunk.len(), &mut vectors[at..]);
                } else {
                    let (_, out) = outputs[0].try_extract_tensor::<f32>().unwrap();
                    read_rows(out, ROWS, chunk.len(), &mut vectors[at..]);
                }
                let read = Instant::now();
                stages.codec += (filled - started).as_secs_f64();
                stages.hand += (handed - filled).as_secs_f64();
                stages.run += (ran - handed).as_secs_f64();
                stages.read += (read - ran).as_secs_f64();
            }
            checksum(&vectors)
        });
        report(&measured, logs);
        if let Some(first) = &reference {
            let equal = first == &vectors;
            println!("{TAG}: {name}: vectors bit-identical to the float-input path: {equal}");
            if !equal {
                let differing = first.iter().zip(&vectors).filter(|(a, b)| a != b).count();
                println!(
                    "{TAG}: {name}: {differing} of {} values differ",
                    first.len()
                );
            }
        }
    }

    // --- Input and output bound once (IO binding). ---
    let name = format!("rows {ROWS}, float input, bound once");
    if selected(&name) {
        let image_model = onnx_edit::with_rows_as_height(&fixture.model, ROWS as u32).unwrap();
        let key = format!("throughput-h{ROWS}");
        let opened = fixture
            .rig
            .open(
                Model::Bytes(&image_model),
                &Open {
                    cache_key: &key,
                    ..Open::default()
                },
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let placement = placement_text(&opened.placement);
        let ready = opened.seconds;
        let mut session = opened.session;
        let arch = fixture.codec.arch();
        let shape = [
            1usize,
            arch.channels as usize,
            ROWS,
            arch.input_bytes as usize,
        ];
        let attempt = (|| -> Result<Measured, ort::Error> {
            let allocator = Allocator::default();
            let mut input = Tensor::<f32>::new(&allocator, shape)?;
            let output = Tensor::<f32>::new(&allocator, [1usize, width, ROWS, 1])?;
            let mut binding = session.create_binding()?;
            binding.bind_output(fixture.interface.output.name.as_str(), output)?;
            let mut image = ImageInput::float(&fixture.codec);
            let mut vectors: Vec<f32> = Vec::with_capacity(logs * width);
            let mut written = 0usize;
            let measured = timed(&name, ready, placement.clone(), logs, |stages| {
                vectors.clear();
                for chunk in texts.chunks(ROWS) {
                    let started = Instant::now();
                    {
                        let (_, values) = input.extract_tensor_mut();
                        for (row, text) in chunk.iter().enumerate() {
                            image.write(&fixture.codec, text, ROWS, row, values);
                        }
                        for row in chunk.len()..written {
                            image.clear_row(ROWS, row, values);
                        }
                        written = chunk.len();
                    }
                    let filled = Instant::now();
                    binding
                        .bind_input(fixture.interface.input.name.as_str(), &input)
                        .unwrap();
                    let handed = Instant::now();
                    let outputs = session.run_binding(&binding).unwrap();
                    let ran = Instant::now();
                    let (_, out) = outputs[0].try_extract_tensor::<f32>().unwrap();
                    let at = vectors.len();
                    vectors.resize(at + chunk.len() * width, 0.0);
                    read_rows(out, ROWS, chunk.len(), &mut vectors[at..]);
                    let read = Instant::now();
                    stages.codec += (filled - started).as_secs_f64();
                    stages.hand += (handed - filled).as_secs_f64();
                    stages.run += (ran - handed).as_secs_f64();
                    stages.read += (read - ran).as_secs_f64();
                }
                checksum(&vectors)
            });
            if let Some(first) = &reference {
                println!(
                    "{TAG}: {name}: vectors bit-identical to the tensor-per-run path: {}",
                    first == &vectors
                );
            }
            Ok(measured)
        })();
        match attempt {
            Ok(measured) => report(&measured, logs),
            Err(error) => println!("{TAG}: {name}: not available: {error}"),
        }
    }

    // --- Intra-op threads of the session. ---
    for threads in [1usize, 4] {
        let name = format!("rows {ROWS}, float input, {threads} intra-op thread(s)");
        if !selected(&name) {
            continue;
        }
        let key = format!("throughput-h{ROWS}");
        let mut encoder = HeightEncoder::open(
            &fixture.rig,
            &fixture.model,
            &fixture.codec,
            ROWS,
            &Open {
                cache_key: &key,
                intra_threads: Some(threads),
                ..Open::default()
            },
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut vectors: Vec<f32> = Vec::with_capacity(logs * width);
        let placement = placement_text(&encoder.placement);
        let ready = encoder.ready_seconds;
        let measured = timed(&name, ready, placement, logs, |stages| {
            vectors.clear();
            encoder.encode(&texts, &mut vectors, stages);
            checksum(&vectors)
        });
        report(&measured, logs);
    }

    // --- The run alone: the same image again and again, no text encoded in between. ---
    let name = format!("rows {ROWS}, float input, the run alone (one image repeated)");
    if selected(&name) {
        let key = format!("throughput-h{ROWS}");
        let mut encoder = HeightEncoder::open(
            &fixture.rig,
            &fixture.model,
            &fixture.codec,
            ROWS,
            &Open {
                cache_key: &key,
                ..Open::default()
            },
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        // `raw_again` runs the image that `raw` wrote last.
        encoder.raw(&texts[..ROWS]);
        let images = logs / ROWS;
        let placement = placement_text(&encoder.placement);
        let ready = encoder.ready_seconds;
        let measured = timed(&name, ready, placement, images * ROWS, |stages| {
            let mut sum = 0.0;
            let started = Instant::now();
            for _ in 0..images {
                sum += f64::from(encoder.raw_again()[0]);
            }
            stages.run += started.elapsed().as_secs_f64();
            sum
        });
        report(&measured, images * ROWS);
    }

    // --- Several sessions on the NPU at once, each on its own thread. ---
    for sessions in [2usize, 3, 4] {
        let name = format!("rows {ROWS}, float input, {sessions} sessions on {sessions} threads");
        if !selected(&name) {
            continue;
        }
        let key = format!("throughput-h{ROWS}");
        let started = Instant::now();
        let mut encoders: Vec<HeightEncoder> = (0..sessions)
            .map(|_| {
                HeightEncoder::open(
                    &fixture.rig,
                    &fixture.model,
                    &fixture.codec,
                    ROWS,
                    &Open {
                        cache_key: &key,
                        ..Open::default()
                    },
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"))
            })
            .collect();
        let ready = started.elapsed().as_secs_f64();
        let placement = placement_text(&encoders[0].placement);
        // Whole images per session, so that the vectors are those of one session.
        let share = logs.div_ceil(sessions).div_ceil(ROWS) * ROWS;
        let measured = timed(&name, ready, placement, logs, |stages| {
            let parts: Vec<(Vec<f32>, Stages)> = std::thread::scope(|scope| {
                let workers: Vec<_> = encoders
                    .iter_mut()
                    .zip(texts.chunks(share))
                    .map(|(encoder, part)| {
                        scope.spawn(move || {
                            let mut vectors = Vec::with_capacity(part.len() * encoder.width);
                            let mut stages = Stages::default();
                            encoder.encode(part, &mut vectors, &mut stages);
                            (vectors, stages)
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|worker| worker.join().unwrap())
                    .collect()
            });
            let mut all = Vec::with_capacity(logs * width);
            for (vectors, part) in parts {
                all.extend_from_slice(&vectors);
                stages.codec += part.codec;
                stages.hand += part.hand;
                stages.run += part.run;
                stages.read += part.read;
            }
            if let Some(first) = &reference {
                assert!(first == &all, "{name}: other vectors than one session");
            }
            checksum(&all)
        });
        report(&measured, logs);
    }

    // --- Sustained: the texts four times over, in one run, at the rows of the gate. ---
    let name = format!(
        "rows {ROWS}, float input, sustained over {} texts",
        4 * logs
    );
    if selected(&name) {
        let key = format!("throughput-h{ROWS}");
        let mut encoder = HeightEncoder::open(
            &fixture.rig,
            &fixture.model,
            &fixture.codec,
            ROWS,
            &Open {
                cache_key: &key,
                ..Open::default()
            },
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let repeated: Vec<&str> = texts.iter().cycle().take(4 * logs).copied().collect();
        let mut vectors: Vec<f32> = Vec::with_capacity(repeated.len() * width);
        let placement = placement_text(&encoder.placement);
        let ready = encoder.ready_seconds;
        let measured = timed(&name, ready, placement, repeated.len(), |stages| {
            vectors.clear();
            encoder.encode(&repeated, &mut vectors, stages);
            checksum(&vectors)
        });
        report(&measured, repeated.len());
        let mut slowest = measured.rates.clone();
        slowest.sort_by(f64::total_cmp);
        assert!(
            cfg!(debug_assertions) || slowest[0] >= 1_000.0,
            "requirement 4.2 asks for at least 1,000 logs per second sustained; the slowest \
             run gave {:.0}",
            slowest[0]
        );
    }
}

#[test]
#[ignore = "hardware measurement: needs the AMD runtime files, the NPU and the trained model; run on request"]
fn start_up_and_the_provider_cache() {
    let process_started = Instant::now();
    let Some(fixture) = fixture() else { return };
    let texts: Vec<&str> = fixture
        .texts
        .iter()
        .take(ROWS)
        .map(String::as_str)
        .collect();
    println!(
        "{TAG}: runtime library loaded in {:.2} s",
        fixture.rig.load_seconds
    );
    let first_vectors = |encoder: &mut HeightEncoder| {
        let mut vectors = Vec::new();
        encoder.encode(&texts, &mut vectors, &mut Stages::default());
        vectors
    };
    let open = |key: &str, fresh: bool, model: &[u8], rows: usize, provider: &[(&str, &str)]| {
        let started = Instant::now();
        let mut encoder = HeightEncoder::open(
            &fixture.rig,
            model,
            &fixture.codec,
            rows,
            &Open {
                cache_key: key,
                fresh,
                provider,
                ..Open::default()
            },
        )
        .unwrap_or_else(|error| panic!("{key}: {error}"));
        let ready = started.elapsed().as_secs_f64();
        let vectors = first_vectors(&mut encoder);
        (encoder, ready, started.elapsed().as_secs_f64(), vectors)
    };

    // Cold (the provider compiles) and warm (the compiled model comes from the cache).
    let warm_only = std::env::var_os("L2D_NPU_WARM_ONLY").is_some();
    let (cold, ready, first, right) = open("startup", !warm_only, &fixture.model, ROWS, &[]);
    println!(
        "{TAG}: {} start: session ready after {ready:.2} s, first {ROWS} vectors after \
         {first:.2} s; since the start of the test (CSIC read, package verified, runtime \
         loaded): {:.2} s; compiled now: {}",
        if warm_only {
            "warm, fresh process"
        } else {
            "cold"
        },
        process_started.elapsed().as_secs_f64(),
        cold.compiled
    );
    drop(cold);
    for attempt in 1..=3 {
        let (warm, ready, first, vectors) = open("startup", false, &fixture.model, ROWS, &[]);
        println!(
            "{TAG}: warm start {attempt} in the same process: ready after {ready:.2} s, first \
             vectors after {first:.2} s; compiled now: {}; vectors bit-identical to the cold \
             start: {}",
            warm.compiled,
            vectors == right
        );
        assert!(!warm.compiled, "a warm start must not compile");
        assert!(
            vectors == right,
            "a warm start must give the cold start's vectors"
        );
    }

    // What the key must hold: the same key with something else behind it.
    // (a) Another number of rows (another model file of the same encoder).
    let (other, _, _, _) = open("startup-key", true, &fixture.model, ROWS, &[]);
    drop(other);
    let rows = 32usize;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut encoder = HeightEncoder::open(
            &fixture.rig,
            &fixture.model,
            &fixture.codec,
            rows,
            &Open {
                cache_key: "startup-key",
                ..Open::default()
            },
        )?;
        let texts = &texts[..rows];
        let mut vectors = Vec::new();
        encoder.encode(texts, &mut vectors, &mut Stages::default());
        Ok::<_, String>((encoder.compiled, vectors == right[..vectors.len()]))
    }));
    println!(
        "{TAG}: key reused for {rows} rows after {ROWS}: {}",
        match outcome {
            Ok(Ok((compiled, equal))) =>
                format!("opened; compiled anew: {compiled}; vectors right: {equal}"),
            Ok(Err(error)) => format!("refused: {}", &error[..error.len().min(200)]),
            Err(_) => "the run failed (panic)".to_string(),
        }
    );
    // (b) Another model of the same shape: the published model's weights changed in one
    // byte of the last stored tensor would be another encoder; here, a fold model when
    // the runs directory is given.
    if let Some(runs) = std::env::var_os("L2D_SCIENCE_RUNS").map(std::path::PathBuf::from) {
        let other_model = read(&runs.join("cv5-f0-s0-strip-bce-norm-unit.int8.onnx"));
        let (other, _, _, own) = open("startup-own", true, &other_model, ROWS, &[]);
        drop(other);
        let (stale, _, _, vectors) = open("startup", false, &other_model, ROWS, &[]);
        println!(
            "{TAG}: key of the published model reused for another model of the same shape: \
             compiled anew: {}; vectors are the other model's own: {}; vectors are the \
             published model's: {}",
            stale.compiled,
            vectors == own,
            vectors == right
        );
    }
    // (c) Another provider option under the same key.
    let (optioned, ready, _, vectors) = open(
        "startup",
        false,
        &fixture.model,
        ROWS,
        &[("opt_level", "3")],
    );
    println!(
        "{TAG}: key reused with the provider option opt_level=3: ready after {ready:.2} s, \
         compiled anew: {}; vectors bit-identical: {}",
        optioned.compiled,
        vectors == right
    );
}
