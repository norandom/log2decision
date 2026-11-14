//! Hardware gate of task 2.12, part 1: is every row right on the NPU (requirement 4.6)?
//!
//! Found with the trained model of task 2.1 on AMD Ryzen AI 1.9.0:
//!
//! - In the model's own layout, a batch along the first dimension, only the first text
//!   of a batch is computed right. The other rows are wrong from the first convolution
//!   on, by as much as different texts differ. A row's vector changed when all other
//!   rows changed (seen once, in an exploration that is not committed), while row 1
//!   keeps its vector when only rows 2 to 31 are exchanged (printed here): which rows
//!   a row depends on was not worked out.
//!   `in_the_batch_layout_only_the_first_row_is_right` records this and fails when it
//!   stops being true, so that a runtime release that repairs it is noticed.
//! - With the texts laid along the height of one image
//!   (`onnx_edit::with_rows_as_height`, no retraining), every row is bit-identical to
//!   the text encoded alone. `every_row_is_right_when_the_texts_are_rows_of_an_image` is
//!   the gate: identical rows in give identical rows out, and a text's vector depends
//!   neither on its row nor on the other texts. The tolerance is zero.
//!
//! Run on request, in release, one test at a time (see `npu_support` for the variables):
//!
//! ```text
//! $env:L2D_RUNTIME_DIR = "D:\Source\log2decision\runtime\ryzenai"
//! $env:L2D_SCIENCE_DIR = "D:\Source\log2decision\models\science-2.1"
//! $env:L2D_NPU_WORK_DIR = "D:\BuildArtifacts\log2decision\task-2.12"
//! cargo test --release -p l2d-encoder --test npu_rows -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(windows)]

mod npu_support;

use l2d_encoder::onnx_edit::{self, ELEMENT_FLOAT32, TensorInfo};
use l2d_encoder::{ByteCodec, EncoderPackage};
use npu_support::*;
use ort::session::Session;
use ort::value::TensorRef;

const TAG: &str = "npu_rows";
/// The rows of an image in the gate: the size task 2.12 recommends for the session.
const ROWS: usize = 64;

fn run(session: &mut Session, name: &str, dims: &[i64], values: &[f32]) -> Vec<f32> {
    let tensor = TensorRef::from_array_view((dims.to_vec(), values)).unwrap();
    let outputs = session.run(ort::inputs![name => tensor]).unwrap();
    let (_, out) = outputs[0].try_extract_tensor::<f32>().unwrap();
    out.to_vec()
}

struct Fixture {
    rig: Rig,
    package: EncoderPackage,
    model: Vec<u8>,
    codec: ByteCodec,
    /// CSIC texts with a non-empty input.
    texts: Vec<String>,
    /// Output step of the model.
    step: f32,
}

fn fixture() -> Option<Fixture> {
    let rig = Rig::prepare(TAG)?;
    let science = directory_from(TAG, "L2D_SCIENCE_DIR", "the published model of task 2.1")?;
    let package = published_package(&rig, &science);
    // The bytes the package verified: the session is created from memory.
    let model = package
        .read_model()
        .unwrap_or_else(|error| panic!("{error}"));
    let codec = package.codec().clone();
    let texts: Vec<String> = csic_logs()
        .into_iter()
        .filter(|log| !codec.describe(&log.text).empty)
        .map(|log| log.text)
        .collect();
    println!(
        "{TAG}: encoder {}, {} CSIC texts with a non-empty input",
        package.identity(),
        texts.len()
    );
    let step = package.model().output_quantization.scale;
    Some(Fixture {
        rig,
        package,
        model,
        codec,
        texts,
        step,
    })
}

impl Fixture {
    fn probes(&self, count: usize, offset: usize) -> Vec<&str> {
        (0..count)
            .map(|i| {
                self.texts[(offset + i * (self.texts.len() / count)) % self.texts.len()].as_str()
            })
            .collect()
    }

    /// Each text alone through the batch-of-one model: the reference of both tests.
    fn alone(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        let model = onnx_edit::with_batch(&self.model, 1).unwrap();
        let opened = self
            .rig
            .open(
                Model::Bytes(&model),
                &Open {
                    cache_key: "rows-batch-of-one",
                    fresh: true,
                    ..Open::default()
                },
            )
            .unwrap_or_else(|error| panic!("{error}"));
        let placement = opened.placement.expect("a placement report");
        assert_eq!((placement.npu_nodes, placement.cpu_nodes), (36, 2));
        let mut session = opened.session;
        let arch = self.codec.arch();
        let dims = [1, i64::from(arch.channels), 1, i64::from(arch.input_bytes)];
        let mut row = vec![0f32; arch.row_values()];
        texts
            .iter()
            .map(|text| {
                self.codec.encode_text(text, &mut row);
                run(&mut session, &self.package.model().input.name, &dims, &row)
            })
            .collect()
    }
}

/// (rows that are not bit-identical, largest element difference in output steps, lowest
/// cosine) of `rows` against their references.
fn against(rows: &[f32], reference: &[&Vec<f32>], step: f32) -> (usize, f32, f64) {
    let width = reference[0].len();
    let mut differing = 0;
    let mut largest = 0f32;
    let mut lowest = 1f64;
    for (row, right) in rows.chunks_exact(width).zip(reference) {
        if row != right.as_slice() {
            differing += 1;
        }
        for (a, b) in row.iter().zip(right.iter()) {
            largest = largest.max((a - b).abs() / step);
        }
        lowest = lowest.min(cosine(row, right));
    }
    (differing, largest, lowest)
}

fn quantiles(values: &mut [f64]) -> String {
    values.sort_by(f64::total_cmp);
    format!(
        "min {:.5}, 1 % {:.5}, 10 % {:.5}, median {:.5}, 90 % {:.5}, max {:.5}",
        values[0],
        quantile(values, 0.01),
        quantile(values, 0.10),
        quantile(values, 0.50),
        quantile(values, 0.90),
        values[values.len() - 1]
    )
}

fn unit(row: &[f32]) -> Vec<f32> {
    let mut vector = row.to_vec();
    l2d_encoder::encode::l2_normalize(&mut vector);
    vector
}

#[test]
#[ignore = "hardware gate: needs the AMD runtime files, the NPU and the trained model; run on request"]
fn every_row_is_right_when_the_texts_are_rows_of_an_image() {
    let Some(fixture) = fixture() else { return };
    let probes = fixture.probes(4 * ROWS, 5);
    let alone = fixture.alone(&probes);
    let mut encoder = HeightEncoder::open(
        &fixture.rig,
        &fixture.model,
        &fixture.codec,
        ROWS,
        &Open {
            cache_key: "rows-image",
            fresh: true,
            ..Open::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let placement = encoder.placement.clone().expect("a placement report");
    println!(
        "{TAG}: image of {ROWS} rows: ready after {:.2} s, {} of {} nodes on the NPU, on the \
         CPU: {:?}",
        encoder.ready_seconds,
        placement.npu_nodes,
        placement.all_nodes,
        placement.cpu_operators()
    );
    // `error_on_failure` does not stop a session whose graph the provider could not
    // compile: it then runs every node on the CPU. The placement is the check.
    assert_eq!(
        (placement.npu_nodes, placement.cpu_nodes),
        (33, 2),
        "all nodes but the input's QuantizeLinear and the output's DequantizeLinear must be \
         on the NPU"
    );
    assert_eq!(
        placement.cpu_operators(),
        ["QuantizeLinear", "DequantizeLinear"]
    );

    // 1. Identical rows in give identical rows out, equal to the text alone.
    let mut worst = (0usize, 0f32, 1f64);
    for (text, right) in probes.iter().zip(&alone).take(8) {
        let rows = encoder.raw(&vec![*text; ROWS]);
        let found = against(&rows, &vec![right; ROWS], fixture.step);
        worst = (
            worst.0 + found.0,
            worst.1.max(found.1),
            worst.2.min(found.2),
        );
    }
    println!(
        "{TAG}: identical rows in (8 texts x {ROWS} rows): {} rows differ from the text \
         alone, largest difference {} output steps, lowest cosine {:.6}",
        worst.0, worst.1, worst.2
    );
    assert_eq!(worst.0, 0, "identical rows in must give identical rows out");

    // 2. A text in every row, among other texts that change with the row.
    let mut worst = (0usize, 0f32, 1f64);
    for probe in 0..4 {
        for row in 0..ROWS {
            let mut image: Vec<&str> = (0..ROWS)
                .map(|i| probes[(row * 7 + i * 3 + 11) % probes.len()])
                .collect();
            image[row] = probes[probe];
            let rows = encoder.raw(&image);
            let own = &rows[row * encoder.width..(row + 1) * encoder.width];
            let found = against(own, &[&alone[probe]], fixture.step);
            worst = (
                worst.0 + found.0,
                worst.1.max(found.1),
                worst.2.min(found.2),
            );
        }
    }
    println!(
        "{TAG}: a text in each of the {ROWS} rows among changing other texts (4 texts): {} \
         vectors differ from the text alone, largest difference {} steps, lowest cosine {:.6}",
        worst.0, worst.1, worst.2
    );
    assert_eq!(
        worst.0, 0,
        "a text's vector must not depend on its row or on the other texts"
    );

    // 3. Whole images of different texts, and a last image that is not full.
    let mut differing = 0;
    for (image, right) in probes.chunks(ROWS).zip(alone.chunks(ROWS)) {
        let rows = encoder.raw(image);
        differing += against(&rows, &right.iter().collect::<Vec<_>>(), fixture.step).0;
    }
    let partial = encoder.raw(&probes[..3]);
    differing += against(
        &partial,
        &alone[..3].iter().collect::<Vec<_>>(),
        fixture.step,
    )
    .0;
    println!(
        "{TAG}: {} different texts in images of {ROWS} and one image of 3: {differing} vectors \
         differ from the text alone",
        probes.len()
    );
    assert_eq!(differing, 0, "every row must be the text's own vector");

    // 4. Repeatable, and for scale: how far apart different texts are.
    let first = encoder.raw(&probes[..ROWS]);
    assert_eq!(first, encoder.raw(&probes[..ROWS]), "the same image twice");
    let sample = fixture.probes(2_000, 1);
    let mut vectors = Vec::new();
    encoder.encode(&sample, &mut vectors, &mut Stages::default());
    let width = encoder.width;
    let mut nearest: Vec<f64> = (0..sample.len())
        .map(|i| {
            (0..sample.len())
                .filter(|j| *j != i && sample[*j] != sample[i])
                .map(|j| {
                    distance2(
                        &vectors[i * width..(i + 1) * width],
                        &vectors[j * width..(j + 1) * width],
                    )
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    println!(
        "{TAG}: squared L2 distance from a text's unit vector to the nearest other text's \
         (2,000 texts): {}",
        quantiles(&mut nearest)
    );
    println!("{TAG}: requirement 4.6 holds exactly in this layout (tolerance 0 output steps)");
}

#[test]
#[ignore = "hardware gate: needs the AMD runtime files, the NPU and the trained model; run on request"]
fn in_the_batch_layout_only_the_first_row_is_right() {
    let Some(fixture) = fixture() else { return };
    let probes = fixture.probes(64, 9);
    let alone = fixture.alone(&probes);
    let arch = *fixture.codec.arch();
    let batch = arch.batch as usize;
    let dims = [
        batch as i64,
        i64::from(arch.channels),
        1,
        i64::from(arch.input_bytes),
    ];
    let name = fixture.package.model().input.name.clone();
    let batch_of = |texts: &[&str]| {
        let mut values = vec![0f32; arch.batch_values()];
        for (row, text) in texts.iter().enumerate() {
            fixture.codec.encode_text(
                text,
                &mut values[row * arch.row_values()..(row + 1) * arch.row_values()],
            );
        }
        values
    };
    let opened = fixture
        .rig
        .open(
            Model::Bytes(&fixture.model),
            &Open {
                cache_key: "rows-batch",
                fresh: true,
                ..Open::default()
            },
        )
        .unwrap_or_else(|error| panic!("{error}"));
    let placement = opened.placement.expect("a placement report");
    assert_eq!((placement.npu_nodes, placement.cpu_nodes), (36, 2));
    let mut session = opened.session;

    // Identical rows in.
    let mut wrong_rows = 0;
    let mut first_rows_right = 0;
    let (mut largest, mut lowest) = (0f32, 1f64);
    for (text, right) in probes.iter().zip(&alone).take(16) {
        let out = run(&mut session, &name, &dims, &batch_of(&vec![*text; batch]));
        if out[..right.len()] == right[..] {
            first_rows_right += 1;
        }
        let found = against(&out, &vec![right; batch], fixture.step);
        wrong_rows += found.0;
        largest = largest.max(found.1);
        lowest = lowest.min(found.2);
    }
    println!(
        "{TAG}: batch layout, identical rows in (16 texts x {batch} rows): first row equal \
         to the text alone in {first_rows_right} of 16 batches; {wrong_rows} of {} other rows \
         differ, by up to {largest} output steps, lowest cosine {lowest:.4}",
        16 * (batch - 1)
    );
    assert_eq!(
        first_rows_right, 16,
        "the first row of a batch is the text's own vector"
    );

    // Different texts: how wrong the other rows are, on the scale of the vectors.
    let mut errors = Vec::new();
    let mut moved_by_others = 0;
    for (texts, right) in probes.chunks(batch).zip(alone.chunks(batch)) {
        let out = run(&mut session, &name, &dims, &batch_of(texts));
        // The same batch with the other rows exchanged: does row 1 keep its vector?
        let mut swapped = texts.to_vec();
        swapped[2..].reverse();
        let again = run(&mut session, &name, &dims, &batch_of(&swapped));
        let width = right[0].len();
        if out[width..2 * width] != again[width..2 * width] {
            moved_by_others += 1;
        }
        for (row, right) in out.chunks_exact(width).zip(right).skip(1) {
            errors.push(distance2(&unit(row), &unit(right)));
        }
    }
    println!(
        "{TAG}: batch layout, different texts: squared L2 distance between a row's unit \
         vector and the text's own ({} rows other than the first): {}",
        errors.len(),
        quantiles(&mut errors)
    );
    println!(
        "{TAG}: batch layout: the vector of row 1 changed when the texts of rows 2 to {} \
         were exchanged in {moved_by_others} of 2 batches",
        batch - 1
    );

    // Where it starts: the output of the first convolution.
    let tensor = "/f/f.0/f.0.1/Relu_output_0_DequantizeLinear_Output";
    let declared = TensorInfo {
        name: tensor.into(),
        element_type: ELEMENT_FLOAT32,
        shape: vec![batch as i64, 128, 1, i64::from(arch.input_bytes)],
    };
    let cut = onnx_edit::with_output_at(&fixture.model, tensor, Some(&declared)).unwrap();
    let mut first_conv = fixture
        .rig
        .open(
            Model::Bytes(&cut),
            &Open {
                cache_key: "rows-batch-first-conv",
                fresh: true,
                ..Open::default()
            },
        )
        .unwrap_or_else(|error| panic!("{error}"))
        .session;
    let zero = vec![0f32; arch.batch_values()];
    let base = run(&mut first_conv, &name, &dims, &zero);
    let width = base.len() / batch;
    let equal_to_first = base
        .chunks_exact(width)
        .filter(|row| *row == &base[..width])
        .count();
    let mut moved = |row: usize| {
        let mut input = zero.clone();
        input[row * arch.row_values() + 5 * arch.input_bytes as usize + 100] = 3.0;
        let out = run(&mut first_conv, &name, &dims, &input);
        out.iter().zip(&base).filter(|(a, b)| a != b).count()
    };
    let (in_first, in_second) = (moved(0), moved(1));
    println!(
        "{TAG}: batch layout, output of the first convolution: with an all-zero input {} of \
         {batch} rows equal the first; one input value set in row 0 moves {in_first} output \
         values, the same value set in row 1 moves {in_second}",
        equal_to_first
    );

    // The tripwire: when this fails, the runtime computes the batch layout right and the
    // choice of layout (tasks.md, 2.12 notes) should be measured again.
    assert!(
        wrong_rows > 0,
        "every row of the batch layout is now right: the defect this test records is gone; \
         measure the batch layout again (task 2.12 notes)"
    );
}
