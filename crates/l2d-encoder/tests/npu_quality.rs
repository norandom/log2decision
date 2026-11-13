//! Task 2.12, part 3: the quality of the vectors that come off the NPU, for the science
//! gate's decision.
//!
//! For each fold model of task 2.1 (stripped input, objective `bce-norm-unit`, folds 0 to
//! 4, seeds 0 to 2) the vectors of CSIC's distinct texts are computed on the NPU, in the
//! layout in which every row is right, and scored in Rust with the science gate's own
//! fold membership and metric definitions (`npu_support/quality.rs`): cross-endpoint ROC
//! AUC and average precision of the linear head and of the nearest-neighbour ratio at
//! k = 11 (and k = 51), next to the figures the science gate recorded for the same runs.
//! No model is run on the CPU provider here.
//!
//! The definitions are tested without hardware by the default tests of this file. The
//! measurement runs on request, in release (about ten minutes for 15 models):
//!
//! ```text
//! $env:L2D_SCIENCE_RUNS = "D:\BuildArtifacts\log2decision\science-2.1\runs"
//! cargo test --release -p l2d-encoder --test npu_quality -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `L2D_NPU_QUALITY_SEEDS` (default `0,1,2`) and `L2D_NPU_QUALITY_FOLDS` (default
//! `0,1,2,3,4`) select runs.

#![cfg(windows)]

mod npu_support;

use std::collections::BTreeSet;
use std::time::Instant;

use l2d_encoder::{ByteCodec, ByteTable, InputArch, InputMode};
use npu_support::quality::*;
use npu_support::*;
use sha2::{Digest, Sha256};

const TAG: &str = "npu_quality";
const ROWS: usize = 64;

#[test]
fn roc_auc_and_average_precision_follow_the_usual_definitions() {
    let bad = [false, false, true, true];
    // scikit-learn's documentation example: 0.75 and 0.8333.
    let score = [0.1, 0.4, 0.35, 0.8];
    assert_eq!(roc_auc(&score, &bad), Some(0.75));
    assert!((average_precision(&score, &bad).unwrap() - 0.833_333_333).abs() < 1e-8);
    // Ties count half; a constant score is chance.
    assert_eq!(roc_auc(&[0.5; 4], &bad), Some(0.5));
    assert_eq!(average_precision(&[0.5; 4], &bad), Some(0.5));
    assert_eq!(roc_auc(&[0.0, 0.0, 1.0, 1.0], &bad), Some(1.0));
    assert_eq!(average_precision(&[0.0, 0.0, 1.0, 1.0], &bad), Some(1.0));
    assert_eq!(roc_auc(&[1.0, 1.0, 0.0, 0.0], &bad), Some(0.0));
    // One tie across the labels: pairs (g2, b1) tie, the other three are right.
    assert_eq!(roc_auc(&[0.1, 0.5, 0.5, 0.9], &bad), Some(0.875));
    assert_eq!(roc_auc(&score, &[true; 4]), None);
    assert_eq!(average_precision(&score, &[false; 4]), None);
}

#[test]
fn the_neighbour_ratio_is_the_bad_share_among_the_nearest() {
    // Stored unit vectors on a circle; the first three are bad.
    let angles = [0.0f32, 0.1, 0.2, 1.0, 1.1, 1.2, 1.3];
    let store: Vec<f32> = angles.iter().flat_map(|a| [a.cos(), a.sin()]).collect();
    let store_bad = [true, true, true, false, false, false, false];
    let query: Vec<f32> = [0.05f32, 1.15, 0.6]
        .iter()
        .flat_map(|a| [a.cos(), a.sin()])
        .collect();
    for threads in [1, 2, 5] {
        let ratios = knn_ratios(&store, &store_bad, &query, 2, &[1, 3, 5], threads);
        assert_eq!(ratios[0], [1.0, 0.0, 1.0]);
        assert_eq!(ratios[1], [1.0, 0.0, 2.0 / 3.0]);
        assert_eq!(ratios[2], [0.6, 0.2, 0.6]);
    }
}

#[test]
fn the_folds_deal_the_mixed_paths_by_size_and_the_others_in_byte_order() {
    // Mixed paths: /m3 (3 texts), /m2 (2), /a (2): by size, ties in byte order.
    let groups = [
        "/m3", "/m3", "/m3", "/m2", "/m2", "/a", "/a", "/s1", "/s2", "/s3", "/s4", "/s5", "/s6",
        "/B",
    ];
    let bad = [
        true, false, true, true, false, false, true, true, true, false, false, true, true, false,
    ];
    assert_eq!(mixed_groups(&groups, &bad), ["/a", "/m2", "/m3"]);
    let fold = cv5_fold_of_group(&groups, &bad);
    assert_eq!((fold["/m3"], fold["/a"], fold["/m2"]), (0, 1, 2));
    // Single-label paths in byte order: /B, /s1, ..., /s6.
    assert_eq!(
        (
            fold["/B"],
            fold["/s1"],
            fold["/s4"],
            fold["/s5"],
            fold["/s6"]
        ),
        (0, 1, 4, 0, 1)
    );
    // The fold of a group in the head's cross-validation: the first eight bytes of the
    // SHA-256 of the path, little-endian, modulo five.
    let digest = Sha256::digest(b"/tienda1/index.jsp");
    let expected = u64::from_le_bytes(digest[..8].try_into().unwrap()) % 5;
    assert_eq!(group_hash_fold("/tienda1/index.jsp") as u64, expected);
    let held: BTreeSet<&str> = ["/b", "/a"].into();
    let joined: String = Sha256::digest(b"/a\n/b")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(paths_sha256(&held), joined);
}

#[test]
fn the_linear_head_separates_separable_vectors() {
    // Two clusters of unit vectors in the plane, three groups each.
    let mut store = Vec::new();
    let mut bad = Vec::new();
    let mut groups = Vec::new();
    for index in 0..60 {
        let is_bad = index % 2 == 0;
        let angle = if is_bad { 0.2 } else { 1.3 } + (index as f32) * 0.004;
        store.extend([angle.cos(), angle.sin()]);
        bad.push(is_bad);
        groups.push(["/a", "/b", "/c", "/d", "/e", "/f"][index % 6]);
    }
    let query = [0.3f32.cos(), 0.3f32.sin(), 1.2f32.cos(), 1.2f32.sin()];
    let (scores, head) = linear_head(&store, &bad, &groups, &query, 2, 3);
    assert!(scores[0] > 0.9 && scores[1] < 0.1, "{scores:?}");
    assert!(PENALTIES.contains(&head.penalty));
    assert_eq!(head.cv_log_loss.len(), PENALTIES.len());
    // The same call again gives the same bits: the fits are deterministic.
    let (again, _) = linear_head(&store, &bad, &groups, &query, 2, 1);
    assert_eq!(scores, again);
}

fn selection(variable: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(variable) {
        Ok(list) if !list.is_empty() => list
            .split(',')
            .map(|part| part.trim().parse().expect("a number"))
            .collect(),
        _ => default.to_vec(),
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// (ROC AUC, average precision) of the scores on the cross-endpoint logs.
fn cross_endpoint(score: &[f64], bad: &[bool], in_mixed: &[bool]) -> (f64, f64) {
    let pick = |values: &[f64]| -> Vec<f64> {
        values
            .iter()
            .zip(in_mixed)
            .filter(|(_, mixed)| **mixed)
            .map(|(value, _)| *value)
            .collect()
    };
    let labels: Vec<bool> = bad
        .iter()
        .zip(in_mixed)
        .filter(|(_, mixed)| **mixed)
        .map(|(bad, _)| *bad)
        .collect();
    let score = pick(score);
    (
        roc_auc(&score, &labels).expect("both labels on the mixed paths"),
        average_precision(&score, &labels).expect("both labels on the mixed paths"),
    )
}

#[test]
#[ignore = "hardware measurement: needs the AMD runtime files, the NPU and the fold models of task 2.1; run on request"]
fn the_vectors_from_the_npu_are_scored_as_the_science_gate_scored() {
    let Some(rig) = Rig::prepare(TAG) else { return };
    let Some(runs) = directory_from(TAG, "L2D_SCIENCE_RUNS", "the fold runs of task 2.1") else {
        return;
    };
    let logs = csic_logs();
    let texts: Vec<&str> = logs.iter().map(|log| log.text.as_str()).collect();
    let bad: Vec<bool> = logs.iter().map(|log| log.bad).collect();
    let groups: Vec<&str> = logs.iter().map(|log| log.group.as_str()).collect();
    let mixed: BTreeSet<String> = mixed_groups(&groups, &bad).into_iter().collect();
    let fold_of = cv5_fold_of_group(&groups, &bad);
    println!(
        "{TAG}: {} distinct texts, {} paths, {} of them with both labels",
        texts.len(),
        fold_of.len(),
        mixed.len()
    );
    let threads = 6;
    let seeds = selection("L2D_NPU_QUALITY_SEEDS", &[0, 1, 2]);
    let folds = selection("L2D_NPU_QUALITY_FOLDS", &[0, 1, 2, 3, 4]);
    println!(
        "{TAG}: columns: fold | seed | linear head ROC AUC NPU / recorded | linear head AP NPU \
         / recorded | neighbours k=11 ROC AUC NPU / recorded | k=11 AP NPU / recorded | k=51 \
         ROC AUC NPU / recorded | head penalty NPU / recorded | head fits at the iteration cap \
         | encode logs/s"
    );
    // Per metric: (NPU, recorded) of every run.
    let mut all: Vec<[(f64, f64); 5]> = Vec::new();
    let mut by_seed: std::collections::BTreeMap<usize, Vec<[(f64, f64); 5]>> = Default::default();
    for seed in &seeds {
        for fold in &folds {
            let stem = format!("cv5-f{fold}-s{seed}-strip-bce-norm-unit");
            let model = read(&runs.join(format!("{stem}.int8.onnx")));
            let table = read(&runs.join(format!("{stem}.table.bin")));
            let record: serde_json::Value =
                serde_json::from_slice(&read(&runs.join(format!("{stem}.json")))).unwrap();
            // The files are the run's, and the split is the run's.
            assert_eq!(
                record["export"]["sha256"]["int8_onnx"],
                hex_sha256(&model),
                "{stem}"
            );
            assert_eq!(
                record["export"]["sha256"]["table"],
                hex_sha256(&table),
                "{stem}"
            );
            let held_paths: BTreeSet<&str> = fold_of
                .iter()
                .filter(|(_, f)| *f == fold)
                .map(|(group, _)| group.as_str())
                .collect();
            let held: Vec<bool> = groups
                .iter()
                .map(|group| held_paths.contains(group))
                .collect();
            let split = &record["split"];
            assert_eq!(split["held_out_paths"], held_paths.len(), "{stem}");
            assert_eq!(
                split["held_out_paths_sha256"],
                paths_sha256(&held_paths),
                "{stem}"
            );
            assert_eq!(
                split["held_out_logs"],
                held.iter().filter(|held| **held).count(),
                "{stem}"
            );
            assert_eq!(record["data"]["distinct_texts"], texts.len(), "{stem}");

            let arch = InputArch::V1;
            let codec = ByteCodec::new(
                arch,
                ByteTable::from_le_bytes(&table, &arch).unwrap(),
                InputMode::Payload,
            )
            .unwrap();
            let key = format!("quality-{stem}");
            let mut encoder = HeightEncoder::open(
                &rig,
                &model,
                &codec,
                ROWS,
                &Open {
                    cache_key: &key,
                    fresh: true,
                    ..Open::default()
                },
            )
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
            let placement = encoder.placement.clone().expect("a placement report");
            assert_eq!(
                (placement.npu_nodes, placement.cpu_nodes),
                (33, 2),
                "{stem}"
            );
            let width = encoder.width;
            let mut vectors = Vec::with_capacity(texts.len() * width);
            let started = Instant::now();
            encoder.encode(&texts, &mut vectors, &mut Stages::default());
            let rate = texts.len() as f64 / started.elapsed().as_secs_f64();
            drop(encoder);

            let (mut store, mut store_bad, mut store_groups) = (Vec::new(), Vec::new(), Vec::new());
            let (mut query, mut query_bad, mut query_mixed) = (Vec::new(), Vec::new(), Vec::new());
            for (index, vector) in vectors.chunks_exact(width).enumerate() {
                if held[index] {
                    query.extend_from_slice(vector);
                    query_bad.push(bad[index]);
                    query_mixed.push(mixed.contains(groups[index]));
                } else {
                    store.extend_from_slice(vector);
                    store_bad.push(bad[index]);
                    store_groups.push(groups[index]);
                }
            }
            let ratios = knn_ratios(&store, &store_bad, &query, width, &[11, 51], threads);
            let (probability, head) =
                linear_head(&store, &store_bad, &store_groups, &query, width, threads);
            let head_npu = cross_endpoint(&probability, &query_bad, &query_mixed);
            let k11_npu = cross_endpoint(&ratios[0], &query_bad, &query_mixed);
            let k51_npu = cross_endpoint(&ratios[1], &query_bad, &query_mixed);
            let recorded = |scorer: &str, metric: &str| {
                record["int8"][scorer]["cross_endpoint"][metric]
                    .as_f64()
                    .unwrap_or_else(|| panic!("{stem}: no recorded {scorer} {metric}"))
            };
            let row = [
                (head_npu.0, recorded("linear_head", "roc_auc")),
                (head_npu.1, recorded("linear_head", "average_precision")),
                (k11_npu.0, recorded("knn_k11", "roc_auc")),
                (k11_npu.1, recorded("knn_k11", "average_precision")),
                (k51_npu.0, recorded("knn_k51", "roc_auc")),
            ];
            println!(
                "{TAG}: | {fold} | {seed} | {:.4} / {:.4} | {:.4} / {:.4} | {:.4} / {:.4} | {:.4} \
                 / {:.4} | {:.4} / {:.4} | {} / {} | {} of {} | {rate:.0} |",
                row[0].0,
                row[0].1,
                row[1].0,
                row[1].1,
                row[2].0,
                row[2].1,
                row[3].0,
                row[3].1,
                row[4].0,
                row[4].1,
                head.penalty,
                record["int8"]["linear_head"]["head"]["penalty"],
                head.capped,
                head.fits
            );
            for (value, _) in row {
                assert!(
                    (0.0..=1.0).contains(&value),
                    "{stem}: a metric outside 0 to 1"
                );
            }
            all.push(row);
            by_seed.entry(*seed).or_default().push(row);
        }
    }
    let mean = |rows: &[[(f64, f64); 5]], metric: usize| {
        let n = rows.len() as f64;
        (
            rows.iter().map(|row| row[metric].0).sum::<f64>() / n,
            rows.iter().map(|row| row[metric].1).sum::<f64>() / n,
        )
    };
    let line = |label: String, rows: &[[(f64, f64); 5]]| {
        let cells: Vec<String> = (0..5)
            .map(|metric| {
                let (npu, recorded) = mean(rows, metric);
                format!("{npu:.4} / {recorded:.4}")
            })
            .collect();
        println!("{TAG}: | mean | {label} | {} |", cells.join(" | "));
    };
    for (seed, rows) in &by_seed {
        line(format!("seed {seed} ({} folds)", rows.len()), rows);
    }
    line(format!("all {} runs", all.len()), &all);
    let largest = (0..5)
        .map(|metric| {
            all.iter()
                .map(|row| (row[metric].0 - row[metric].1).abs())
                .fold(0.0, f64::max)
        })
        .collect::<Vec<_>>();
    println!(
        "{TAG}: largest difference between the NPU figure and the recorded one, per column: \
         {largest:.4?}"
    );
}
