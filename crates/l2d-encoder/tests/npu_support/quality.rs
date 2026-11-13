//! The science gate's definitions (task 2.1, `tools/science/l2dsci.py`) in Rust, for the
//! quality of the vectors that come off the NPU: the five folds of path groups, the
//! linear head with its penalty chosen on the fitting side, the nearest-neighbour ratio,
//! and the two metrics. Nothing here needs hardware; `npu_quality.rs` tests these
//! functions by default and uses them on request.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use l2d_baseline::{LogReg, LogRegSettings};
use sha2::{Digest, Sha256};

/// The penalties the science gate's linear head chooses from.
pub const PENALTIES: [f64; 5] = [0.01, 0.1, 1.0, 10.0, 100.0];

/// The path groups that hold both labels, in byte order.
pub fn mixed_groups(groups: &[&str], bad: &[bool]) -> Vec<String> {
    let mut labels: BTreeMap<&str, (bool, bool)> = BTreeMap::new();
    for (group, bad) in groups.iter().zip(bad) {
        let seen = labels.entry(group).or_insert((false, false));
        if *bad {
            seen.1 = true;
        } else {
            seen.0 = true;
        }
    }
    labels
        .into_iter()
        .filter(|(_, (good, bad))| *good && *bad)
        .map(|(group, _)| group.to_string())
        .collect()
}

/// `cv5_fold_of_group`: the paths with both labels are dealt round-robin from the
/// largest to the smallest (distinct texts; ties in byte order), the single-label paths
/// round-robin in byte order.
pub fn cv5_fold_of_group(groups: &[&str], bad: &[bool]) -> BTreeMap<String, usize> {
    let mixed = mixed_groups(groups, bad);
    let mut size: BTreeMap<&str, usize> = mixed.iter().map(|group| (group.as_str(), 0)).collect();
    for group in groups {
        if let Some(count) = size.get_mut(group) {
            *count += 1;
        }
    }
    let mut by_size: Vec<&str> = mixed.iter().map(String::as_str).collect();
    by_size.sort_by(|a, b| size[b].cmp(&size[a]).then(a.as_bytes().cmp(b.as_bytes())));
    let all: BTreeSet<&str> = groups.iter().copied().collect();
    let single: Vec<&str> = all
        .into_iter()
        .filter(|group| !size.contains_key(group))
        .collect();
    let mut fold = BTreeMap::new();
    for (index, group) in by_size.iter().enumerate() {
        fold.insert(group.to_string(), index % 5);
    }
    for (index, group) in single.iter().enumerate() {
        fold.insert(group.to_string(), index % 5);
    }
    fold
}

/// The record's `held_out_paths_sha256`: SHA-256 of the held-out paths, sorted, joined
/// by line feeds.
pub fn paths_sha256(paths: &BTreeSet<&str>) -> String {
    let joined = paths.iter().copied().collect::<Vec<_>>().join("\n");
    Sha256::digest(joined.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `group_hash_fold`: the fold of a group inside the linear head's own cross-validation.
pub fn group_hash_fold(group: &str) -> usize {
    let digest = Sha256::digest(group.as_bytes());
    (u64::from_le_bytes(digest[..8].try_into().unwrap()) % 5) as usize
}

/// ROC AUC with tied scores counted half, as scikit-learn computes it. `None` when one
/// of the labels is absent.
pub fn roc_auc(score: &[f64], bad: &[bool]) -> Option<f64> {
    let positives = bad.iter().filter(|bad| **bad).count();
    let negatives = bad.len() - positives;
    if positives == 0 || negatives == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|a, b| score[*a].total_cmp(&score[*b]));
    let mut rank_sum = 0.0f64;
    let mut at = 0usize;
    while at < order.len() {
        let mut end = at + 1;
        while end < order.len() && score[order[end]] == score[order[at]] {
            end += 1;
        }
        // Ranks at + 1 ..= end share their mean.
        let mean_rank = (at + 1 + end) as f64 / 2.0;
        let tied_positives = order[at..end].iter().filter(|index| bad[**index]).count();
        rank_sum += mean_rank * tied_positives as f64;
        at = end;
    }
    let positives = positives as f64;
    Some((rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives as f64))
}

/// Average precision as scikit-learn computes it: the sum over the distinct scores, from
/// the highest down, of the gain in recall times the precision at that score.
pub fn average_precision(score: &[f64], bad: &[bool]) -> Option<f64> {
    let positives = bad.iter().filter(|bad| **bad).count();
    if positives == 0 || positives == bad.len() {
        return None;
    }
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|a, b| score[*b].total_cmp(&score[*a]));
    let mut sum = 0.0f64;
    let mut hits = 0usize;
    let mut at = 0usize;
    while at < order.len() {
        let mut end = at + 1;
        while end < order.len() && score[order[end]] == score[order[at]] {
            end += 1;
        }
        let gained = order[at..end].iter().filter(|index| bad[**index]).count();
        hits += gained;
        sum += gained as f64 / positives as f64 * (hits as f64 / end as f64);
        at = end;
    }
    Some(sum)
}

/// The bad share among the `k` nearest stored vectors of each query, for each `k` of
/// `ks` (ascending). Unit vectors: nearest by squared L2 is largest by dot product. Ties
/// go to the earlier stored vector. `width` is the vector dimension.
pub fn knn_ratios(
    store: &[f32],
    store_bad: &[bool],
    query: &[f32],
    width: usize,
    ks: &[usize],
    threads: usize,
) -> Vec<Vec<f64>> {
    let queries = query.len() / width;
    let largest = *ks.last().expect("at least one k");
    assert!(store_bad.len() >= largest, "fewer stored vectors than k");
    let chunk = queries.div_ceil(threads.max(1)).max(1);
    let mut ratios: Vec<Vec<f64>> = ks.iter().map(|_| vec![0.0; queries]).collect();
    let parts: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
        let workers: Vec<_> = query
            .chunks(chunk * width)
            .map(|rows| {
                scope.spawn(move || {
                    let mut out: Vec<Vec<f64>> = ks.iter().map(|_| Vec::new()).collect();
                    let mut sims: Vec<(f32, u32)> = Vec::with_capacity(store_bad.len());
                    for row in rows.chunks_exact(width) {
                        sims.clear();
                        for (index, stored) in store.chunks_exact(width).enumerate() {
                            let dot: f32 = row.iter().zip(stored).map(|(a, b)| a * b).sum();
                            sims.push((dot, index as u32));
                        }
                        let nearer = |a: &(f32, u32), b: &(f32, u32)| {
                            b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
                        };
                        sims.select_nth_unstable_by(largest - 1, nearer);
                        sims[..largest].sort_by(nearer);
                        for (slot, k) in ks.iter().enumerate() {
                            let bad = sims[..*k]
                                .iter()
                                .filter(|(_, index)| store_bad[*index as usize])
                                .count();
                            out[slot].push(bad as f64 / *k as f64);
                        }
                    }
                    out
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("a neighbour worker panicked"))
            .collect()
    });
    let mut at = 0usize;
    for part in parts {
        let count = part[0].len();
        for (slot, values) in part.into_iter().enumerate() {
            ratios[slot][at..at + count].copy_from_slice(&values);
        }
        at += count;
    }
    ratios
}

/// What the linear head reports beside its scores.
#[derive(Debug, Clone)]
pub struct Head {
    pub penalty: f64,
    /// Cross-validated log loss per penalty of [`PENALTIES`].
    pub cv_log_loss: Vec<f64>,
    /// Fits that stopped at the iteration cap, of all fits.
    pub capped: usize,
    pub fits: usize,
}

fn fit(rows: &[&[f32]], labels: &[bool], l2: f64) -> (LogReg, bool) {
    let settings = LogRegSettings {
        l2,
        max_iter: 2_000,
        tolerance: 1e-3,
    };
    let (model, report) = LogReg::fit_dense(rows, labels, &settings)
        .unwrap_or_else(|error| panic!("fitting the linear head: {error}"));
    (model, report.converged())
}

/// The science gate's linear head: logistic regression on the stored unit vectors (the
/// baseline's deterministic regression, dense path), the penalty chosen from
/// [`PENALTIES`] by five-fold cross-validation on log loss over the stored vectors only,
/// folds by a hash of the group; then the probability of each query.
pub fn linear_head(
    store: &[f32],
    store_bad: &[bool],
    store_groups: &[&str],
    query: &[f32],
    width: usize,
    threads: usize,
) -> (Vec<f64>, Head) {
    let rows: Vec<&[f32]> = store.chunks_exact(width).collect();
    let folds: Vec<usize> = store_groups
        .iter()
        .map(|group| group_hash_fold(group))
        .collect();
    // One job per penalty and fold; each fit is deterministic on its own.
    let jobs: Vec<(usize, usize)> = (0..PENALTIES.len())
        .flat_map(|penalty| (0..5).map(move |fold| (penalty, fold)))
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<(usize, f64, bool, bool)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.max(1))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((penalty, fold)) = jobs.get(at).copied() else {
                            break;
                        };
                        let fit_rows: Vec<&[f32]> = rows
                            .iter()
                            .zip(&folds)
                            .filter(|(_, f)| **f != fold)
                            .map(|(row, _)| *row)
                            .collect();
                        let fit_labels: Vec<bool> = store_bad
                            .iter()
                            .zip(&folds)
                            .filter(|(_, f)| **f != fold)
                            .map(|(bad, _)| *bad)
                            .collect();
                        let test: Vec<usize> =
                            (0..rows.len()).filter(|i| folds[*i] == fold).collect();
                        let both = fit_labels.iter().any(|b| *b) && fit_labels.iter().any(|b| !*b);
                        if test.is_empty() || !both {
                            done.push((penalty, 0.0, false, false));
                            continue;
                        }
                        let (model, converged) = fit(&fit_rows, &fit_labels, PENALTIES[penalty]);
                        // The summed log loss of the fold, probabilities clipped as
                        // scikit-learn clips them.
                        let eps = f64::EPSILON;
                        let loss: f64 = test
                            .iter()
                            .map(|index| {
                                let p = model
                                    .probability_dense(rows[*index])
                                    .unwrap()
                                    .clamp(eps, 1.0 - eps);
                                if store_bad[*index] {
                                    -p.ln()
                                } else {
                                    -(1.0 - p).ln()
                                }
                            })
                            .sum();
                        done.push((penalty, loss, true, !converged));
                    }
                    done
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("a head worker panicked"))
            .collect()
    });
    let mut cv_log_loss = vec![0.0f64; PENALTIES.len()];
    let (mut fits, mut capped) = (0usize, 0usize);
    for (penalty, loss, fitted, was_capped) in results {
        cv_log_loss[penalty] += loss / rows.len() as f64;
        fits += usize::from(fitted);
        capped += usize::from(was_capped);
    }
    // The lowest loss; ties go to the lower penalty.
    let chosen = (0..PENALTIES.len())
        .min_by(|a, b| cv_log_loss[*a].total_cmp(&cv_log_loss[*b]).then(a.cmp(b)))
        .unwrap();
    let (model, converged) = fit(&rows, store_bad, PENALTIES[chosen]);
    fits += 1;
    capped += usize::from(!converged);
    let scores = query
        .chunks_exact(width)
        .map(|row| model.probability_dense(row).unwrap())
        .collect();
    (
        scores,
        Head {
            penalty: PENALTIES[chosen],
            cv_log_loss,
            capped,
            fits,
        },
    )
}
