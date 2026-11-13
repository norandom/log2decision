//! Quality metrics of a ranking by ratio (requirement 9.1), computed exactly.
//!
//! Every function takes the ratio of each log (higher is more likely bad) and whether
//! the log is bad, in the same order. The ranking figures ([`roc_auc`],
//! [`average_precision`], [`budget_shares`], [`recall_at_budget`]) do not depend on the
//! order of the logs: equal ratios are treated as one tie, never by their position, and
//! the sums run over the distinct ratios from the highest to the lowest, so a figure
//! repeats bit for bit. [`log_loss_sum`] and [`spread`] add their values in the order
//! given: they repeat exactly for the same order, and another order can change the
//! last bits.
//!
//! The two slices must have the same length and no ratio may be NaN; both are
//! programming errors and panic.

use std::cmp::Ordering;

/// Mean, range and spread of a list of figures (the folds of an evaluation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spread {
    pub count: usize,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    /// The sample standard deviation (divisor `count - 1`); 0 for a single figure.
    pub sd: f64,
}

/// The logs as (ratio, bad) from the highest ratio to the lowest.
fn ranked(ratios: &[f32], bad: &[bool]) -> Vec<(f32, bool)> {
    assert_eq!(ratios.len(), bad.len(), "a ratio and a label for every log");
    assert!(
        ratios.iter().all(|ratio| !ratio.is_nan()),
        "a ratio is not a number and cannot be ranked"
    );
    let mut logs: Vec<(f32, bool)> = ratios.iter().copied().zip(bad.iter().copied()).collect();
    // Bad before good inside a tie only so that the order is total; no figure below
    // depends on the order inside a tie.
    logs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.1.cmp(&a.1))
    });
    logs
}

/// The ties of a ranking from the highest ratio down: (bad logs, good logs) per ratio.
fn ties(ranked: &[(f32, bool)]) -> Vec<(u64, u64)> {
    let mut found: Vec<(u64, u64)> = Vec::new();
    let mut last: Option<f32> = None;
    for &(ratio, bad) in ranked {
        if last != Some(ratio) {
            found.push((0, 0));
            last = Some(ratio);
        }
        let tie = found.last_mut().expect("a tie was just opened");
        if bad {
            tie.0 += 1;
        } else {
            tie.1 += 1;
        }
    }
    found
}

/// The area under the ROC curve: the share of pairs of one bad and one good log in
/// which the bad log has the higher ratio, a pair with equal ratios counting one half.
///
/// `None` when there is no bad or no good log (the figure is undefined).
pub fn roc_auc(ratios: &[f32], bad: &[bool]) -> Option<f64> {
    let ties = ties(&ranked(ratios, bad));
    let all_bad: u64 = ties.iter().map(|tie| tie.0).sum();
    let all_good: u64 = ties.iter().map(|tie| tie.1).sum();
    if all_bad == 0 || all_good == 0 {
        return None;
    }
    // Twice the number of won pairs, as an integer: a bad log wins against every good
    // log below its tie (2) and half against every good log inside it (1).
    let mut good_below = all_good;
    let mut twice_won: u128 = 0;
    for (bad_here, good_here) in ties {
        good_below -= good_here;
        twice_won += u128::from(bad_here) * (2 * u128::from(good_below) + u128::from(good_here));
    }
    Some(twice_won as f64 / (2.0 * all_bad as f64 * all_good as f64))
}

/// Average precision: the sum, over the distinct ratios from the highest down, of the
/// gain in recall at that ratio times the precision there. Logs with equal ratios enter
/// together, as one threshold (the definition of scikit-learn's
/// `average_precision_score`).
///
/// `None` when there is no bad log.
pub fn average_precision(ratios: &[f32], bad: &[bool]) -> Option<f64> {
    let ties = ties(&ranked(ratios, bad));
    let all_bad: u64 = ties.iter().map(|tie| tie.0).sum();
    if all_bad == 0 {
        return None;
    }
    let (mut bad_so_far, mut logs_so_far) = (0_u64, 0_u64);
    let mut sum = 0.0_f64;
    for (bad_here, good_here) in ties {
        bad_so_far += bad_here;
        logs_so_far += bad_here + good_here;
        if bad_here > 0 {
            sum += bad_here as f64 * (bad_so_far as f64 / logs_so_far as f64);
        }
    }
    // Divided once at the end: a ranking with every bad log on top gives exactly 1.
    Some(sum / all_bad as f64)
}

/// For every log, in the order given: the share of it that lies within a review budget
/// of `budget` logs when the logs are ranked from the highest ratio down.
///
/// 1 for a log ranked inside the budget, 0 for one outside. When the budget ends inside
/// a tie, the places left are shared equally between the tied logs (the expected value
/// when a tie is broken at random), so the figure does not depend on the order of the
/// logs. The shares add up to the budget, or to the number of logs when that is less.
///
/// Triage itself breaks a tie by the order of the logs frame (task 1.10); this function
/// states what a ranking by ratio alone supports.
pub fn budget_shares(ratios: &[f32], budget: u64) -> Vec<f64> {
    assert!(
        ratios.iter().all(|ratio| !ratio.is_nan()),
        "a ratio is not a number and cannot be ranked"
    );
    let mut order: Vec<usize> = (0..ratios.len()).collect();
    order.sort_by(|&a, &b| {
        ratios[b]
            .partial_cmp(&ratios[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut shares = vec![0.0_f64; ratios.len()];
    let mut start = 0;
    while start < order.len() {
        let ratio = ratios[order[start]];
        let mut end = start;
        while end < order.len() && ratios[order[end]] == ratio {
            end += 1;
        }
        let (above, tied) = (start as u64, (end - start) as u64);
        let share = if budget >= above + tied {
            1.0
        } else if budget <= above {
            0.0
        } else {
            (budget - above) as f64 / tied as f64
        };
        for &log in &order[start..end] {
            shares[log] = share;
        }
        start = end;
    }
    shares
}

/// Recall at a review budget: the share of the bad logs that lies within the first
/// `budget` places of the ranking, with ties at the budget shared ([`budget_shares`]).
///
/// `None` when there is no bad log.
pub fn recall_at_budget(ratios: &[f32], bad: &[bool], budget: u64) -> Option<f64> {
    assert_eq!(ratios.len(), bad.len(), "a ratio and a label for every log");
    let all_bad = bad.iter().filter(|bad| **bad).count();
    if all_bad == 0 {
        return None;
    }
    Some(share_sum(&budget_shares(ratios, budget), bad.iter().copied()) / all_bad as f64)
}

/// The sum of the shares of the chosen logs. Shares of whole logs are added as a count
/// and the shares of a tie afterwards, so the sum does not depend on the order.
pub(crate) fn share_sum(shares: &[f64], chosen: impl Iterator<Item = bool>) -> f64 {
    let (mut whole, mut partial, mut parts) = (0_u64, 0.0_f64, 0_u64);
    for (share, chosen) in shares.iter().zip(chosen) {
        if !chosen {
            continue;
        }
        if *share == 1.0 {
            whole += 1;
        } else if *share > 0.0 {
            // All partial shares of one ranking are equal: they are one tie.
            partial = *share;
            parts += 1;
        }
    }
    whole as f64 + partial * parts as f64
}

/// The summed log loss of the ratios read as the probability of bad:
/// `-ln(ratio)` for a bad log, `-ln(1 - ratio)` for a good one.
///
/// A ratio is a 32-bit float and can be exactly 0 or 1; it is clipped to
/// `[f32::EPSILON, 1 - f32::EPSILON]` first, so one certain and wrong log costs about
/// 15.9 and not infinity (scikit-learn's `log_loss` clips in the same way). The logs
/// are added in the order given.
pub fn log_loss_sum(ratios: &[f32], bad: &[bool]) -> f64 {
    assert_eq!(ratios.len(), bad.len(), "a ratio and a label for every log");
    let floor = f64::from(f32::EPSILON);
    ratios
        .iter()
        .zip(bad)
        .map(|(ratio, bad)| {
            assert!(!ratio.is_nan(), "a ratio is not a number");
            let p = f64::from(*ratio).clamp(floor, 1.0 - floor);
            if *bad { -p.ln() } else { -(1.0 - p).ln() }
        })
        .sum()
}

/// Mean, lowest, highest and sample standard deviation of `values`, added in the order
/// given; `None` for an empty list.
pub fn spread(values: &[f64]) -> Option<Spread> {
    if values.is_empty() {
        return None;
    }
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let sd = if count > 1 {
        (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (count - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    Some(Spread {
        count,
        mean,
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        sd,
    })
}
