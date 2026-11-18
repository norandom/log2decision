//! Unbiased squared maximum mean discrepancy (MMD) with a Gaussian kernel.
//!
//! Each group contributes at most 2,000 rows, selected without replacement.
//! The observed assignment and every permutation use the same pooled sample and
//! bandwidth. All distances and sums are computed in double precision.

use l2d_core::{L2dError, MemoryBudget, MemorySettings};
use thiserror::Error;

/// Number of random label assignments in the default test.
pub const DEFAULT_PERMUTATIONS: u32 = 199;
const MIN_GROUP: usize = 30;
const MAX_GROUP: usize = 2_000;

/// The unbiased statistic (which can be negative), its permutation p-value,
/// and the sample sizes, number of permutations and seed actually used.
#[derive(Debug, PartialEq)]
pub struct ShiftResult {
    pub statistic: f64,
    pub p_value: f64,
    pub n_a: usize,
    pub n_b: usize,
    pub permutations: u32,
    pub seed: u64,
}

#[derive(Debug, Error)]
pub enum ShiftError {
    #[error("shift test needs {minimum} rows per group, received {n_a} and {n_b}")]
    GroupTooSmall {
        n_a: usize,
        n_b: usize,
        minimum: usize,
    },
    #[error("group {group}, row {row}: vectors must be finite and share one nonzero dimension")]
    InvalidVector { group: &'static str, row: usize },
    #[error(transparent)]
    Memory(#[from] L2dError),
}

fn check_sizes(n_a: usize, n_b: usize) -> Result<(), ShiftError> {
    if n_a < MIN_GROUP || n_b < MIN_GROUP {
        return Err(ShiftError::GroupTooSmall {
            n_a,
            n_b,
            minimum: MIN_GROUP,
        });
    }
    Ok(())
}

/// Test two groups under the default host memory budget. With zero random
/// permutations the observed statistic is still computed and the p-value is 1.
pub fn shift_test(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
    permutations: u32,
    seed: u64,
) -> Result<ShiftResult, ShiftError> {
    check_sizes(a.len(), b.len())?;
    let budget = MemoryBudget::observe(&MemorySettings::default())?;
    shift_test_with(a, b, permutations, seed, &budget)
}

/// Test under an existing process budget. The caller owns the input vectors.
/// Before sampling, admit both packed pairwise arrays (kernels and the
/// bandwidth selection scratch), sampled references, labels and fixed overhead.
pub fn shift_test_with(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
    permutations: u32,
    seed: u64,
    budget: &MemoryBudget,
) -> Result<ShiftResult, ShiftError> {
    check_sizes(a.len(), b.len())?;
    let n_a = a.len().min(MAX_GROUP);
    let n_b = b.len().min(MAX_GROUP);
    let n = n_a + n_b;
    let pairs = n * (n - 1) / 2;
    let bytes = 2 * pairs * size_of::<f64>() + n * (size_of::<&[f32]>() + size_of::<bool>()) + 4096;
    budget.require_whole(bytes as u64, "shift test pairwise workspace")?;
    let dimension = a[0].len();
    for (group, rows) in [("a", a), ("b", b)] {
        for (row, vector) in rows.iter().enumerate() {
            if dimension == 0 || vector.len() != dimension || !vector.iter().all(|x| x.is_finite())
            {
                return Err(ShiftError::InvalidVector { group, row });
            }
        }
    }

    let mut rng = Draws(seed);
    let mut sample = Vec::with_capacity(n);
    sample_group(&mut sample, a, &mut rng);
    sample_group(&mut sample, b, &mut rng);
    let mut kernels = Vec::with_capacity(pairs);
    for (i, x) in sample.iter().enumerate() {
        for y in &sample[..i] {
            let squared_distance = x
                .iter()
                .zip(*y)
                .map(|(&u, &v)| (f64::from(u) - f64::from(v)).powi(2))
                .sum::<f64>();
            kernels.push(squared_distance);
        }
    }

    let bandwidth = median_distance(kernels.clone());
    let denominator = 2. * bandwidth * bandwidth;
    for d2 in &mut kernels {
        // Pointwise Gaussian limit as the bandwidth approaches zero: repeated
        // vectors still have similarity one and distinct vectors have zero.
        *d2 = if bandwidth == 0. {
            f64::from(*d2 == 0.)
        } else {
            (-*d2 / denominator).exp()
        };
    }

    let mut in_a: Vec<_> = (0..n).map(|i| i < n_a).collect();
    let statistic = mmd(&kernels, &in_a, n_a, n_b);
    let mut extreme = 0u32;
    for _ in 0..permutations {
        for i in (1..n).rev() {
            in_a.swap(i, rng.below(i + 1));
        }
        extreme += u32::from(mmd(&kernels, &in_a, n_a, n_b) >= statistic);
    }
    Ok(ShiftResult {
        statistic,
        p_value: (1. + f64::from(extreme)) / (1. + f64::from(permutations)),
        n_a,
        n_b,
        permutations,
        seed,
    })
}

// Reservoir sampling keeps only the selected references, even for very large
// input slices. A group at or below the cap consumes no random draws.
fn sample_group<'a>(pool: &mut Vec<&'a [f32]>, group: &'a [Vec<f32>], rng: &mut Draws) {
    let start = pool.len();
    for (i, row) in group.iter().enumerate() {
        if i < MAX_GROUP {
            pool.push(row);
        } else {
            let slot = rng.below(i + 1);
            if slot < MAX_GROUP {
                pool[start + slot] = row;
            }
        }
    }
}

fn median_distance(mut squared_distances: Vec<f64>) -> f64 {
    let middle = squared_distances.len() / 2;
    let even = squared_distances.len().is_multiple_of(2);
    let (lower, upper_middle, _) = squared_distances.select_nth_unstable_by(middle, f64::total_cmp);
    if even {
        let lower_middle = lower.iter().copied().fold(0., f64::max);
        (lower_middle.sqrt() + upper_middle.sqrt()) / 2.
    } else {
        upper_middle.sqrt()
    }
}

fn mmd(kernels: &[f64], in_a: &[bool], n_a: usize, n_b: usize) -> f64 {
    // Each off-diagonal pair occurs once. Accumulate the three pair types
    // separately before applying their different U-statistic denominators.
    let mut sums = [0.; 3];
    let mut corrections = [0.; 3];
    let mut offset = 0;
    for (i, &left) in in_a.iter().enumerate() {
        for (&kernel, &right) in kernels[offset..offset + i].iter().zip(&in_a[..i]) {
            let bin = usize::from(left) + usize::from(right);
            // Compensated addition keeps repeated kernel values and ties from
            // accumulating appreciable order-dependent rounding error.
            let adjusted = kernel - corrections[bin];
            let next = sums[bin] + adjusted;
            corrections[bin] = (next - sums[bin]) - adjusted;
            sums[bin] = next;
        }
        offset += i;
    }
    2. * sums[2] / (n_a * (n_a - 1)) as f64 + 2. * sums[0] / (n_b * (n_b - 1)) as f64
        - 2. * sums[1] / (n_a * n_b) as f64
}

// SplitMix64 gives the same draws on every host. Reject the incomplete residue
// cycle so that every position in a reservoir or shuffle is equally likely.
struct Draws(u64);

impl Draws {
    fn below(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            if z >= threshold {
                return (z % bound) as usize;
            }
        }
    }
}
