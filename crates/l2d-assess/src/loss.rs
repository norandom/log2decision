use crate::overlap::{SAMPLE_CAP, sample_indices};
use crate::{Separability, VectorSet, invalid, neighbour_overlap_with, separability_with};
use faer::{
    MatRef, Side,
    linalg::evd::{ComputeEigenvectors, self_adjoint_evd_scratch},
};
use l2d_core::{L2dError, Label, MemoryBudget, MemorySettings};

#[derive(Debug)]
pub struct LossReport {
    /// Per-component shares of total centered variance in descending order.
    /// Empty if all rows coincide. The profile includes min(rows, width) components.
    pub explained_variance: Vec<f32>,
    /// TwoNN likelihood estimate, absent with fewer than three sampled rows,
    /// coincident sampled rows, or equal first and second distances.
    pub intrinsic_dimension: Option<f64>,
    /// Number of original rows used for each pairwise measure.
    pub sample_size: usize,
    pub seed: u64,
    /// Full-width held-out reference using the same folds as every prefix.
    pub separability: Option<Separability>,
    pub shortened: Vec<ShortenedResult>,
}

#[derive(Debug)]
pub struct ShortenedResult {
    pub dimension: u16,
    pub neighbour_overlap: f32,
    /// Up to ten nearest neighbours, excluding the query itself.
    pub k: usize,
    pub separability_auc: Option<f64>,
}

/// Measure centered PCA on all originals and TwoNN on a seeded reservoir.
/// Each requested length retains the leading coordinates in their original
/// units and order. Logistic regression evaluates held-out labelled rows.
pub fn assess_loss(set: &VectorSet, shorten_to: &[u16], seed: u64) -> Result<LossReport, L2dError> {
    assess_loss_with(
        set,
        shorten_to,
        seed,
        &mut MemoryBudget::observe(&MemorySettings::default())?,
    )
}

/// Run the assessment within a caller-owned memory budget and restore its held bytes.
pub fn assess_loss_with(
    set: &VectorSet,
    shorten_to: &[u16],
    seed: u64,
    budget: &mut MemoryBudget,
) -> Result<LossReport, L2dError> {
    set.validate()?;
    let n = set.vectors.len();
    if n < 2 {
        return Err(invalid("loss assessment requires at least two originals"));
    }
    let width = set.vectors[0].len();
    if shorten_to
        .iter()
        .any(|&d| d == 0 || usize::from(d) >= width)
    {
        return Err(invalid(
            "each requested prefix must be positive and shorter than the input width",
        ));
    }
    let full_bytes = set.resident_bytes();
    let explained_variance = pca(set, budget)?;
    let sample_size = n.min(SAMPLE_CAP);
    let retained = (explained_variance.capacity() * size_of::<f32>()) as u64
        + (shorten_to.len() * size_of::<ShortenedResult>()) as u64
        + 4096;
    budget.require_whole(
        full_bytes + retained + (sample_size * size_of::<usize>()) as u64,
        "loss report, original vectors and TwoNN sample",
    )?;
    let intrinsic_dimension = twonn(set, &sample_indices(n, seed));
    let held = budget.held_bytes();
    budget.hold(held + retained);
    let results: Result<_, L2dError> = (|| {
        let separability = separability_with(set, seed, budget)?;
        let mut shortened = Vec::with_capacity(shorten_to.len());
        let k = 10.min(sample_size - 1);
        for &dimension in shorten_to {
            let d = usize::from(dimension);
            budget.hold(held + retained);
            let prefix_bytes =
                (n * (size_of::<Vec<f32>>() + d * size_of::<f32>() + size_of::<Label>())) as u64
                    + 4096;
            budget.require_whole(
                full_bytes + prefix_bytes,
                "aligned prefix and original vectors",
            )?;
            // These two measures use row identity and labels only.
            let prefix = VectorSet {
                vectors: set.vectors.iter().map(|v| v[..d].to_vec()).collect(),
                labels: set.labels.clone(),
                groups: None,
                name: String::new(),
            };
            let overlap = neighbour_overlap_with(&prefix, set, k, seed, budget)?;
            budget.hold(held + retained + full_bytes);
            let auc = separability_with(&prefix, seed, budget)?.map(|s| s.auc);
            shortened.push(ShortenedResult {
                dimension,
                k,
                neighbour_overlap: overlap.overlap,
                separability_auc: auc,
            });
        }
        Ok((separability, shortened))
    })();
    budget.hold(held);
    let (separability, shortened) = results?;
    Ok(LossReport {
        explained_variance,
        intrinsic_dimension,
        sample_size,
        seed,
        separability,
        shortened,
    })
}

// Work in the smaller of feature and observation space. X'X and XX' have
// the same nonzero eigenvalues. Accumulate centered outer products in f64;
// normalization cancels the sample covariance denominator.
fn pca(set: &VectorSet, budget: &MemoryBudget) -> Result<Vec<f32>, L2dError> {
    let n = set.vectors.len();
    let width = set.vectors[0].len();
    let m = n.min(width);
    let gram_bytes = (m as u64).saturating_mul(m as u64).saturating_mul(8);
    let resident = set.resident_bytes();
    budget.require_whole(
        resident.saturating_add(gram_bytes),
        "PCA Gram matrix and input",
    )?;
    let scratch = self_adjoint_evd_scratch::<f64>(
        m,
        ComputeEigenvectors::No,
        faer::get_global_parallelism(),
        Default::default(),
    )
    .unaligned_bytes_required();
    budget.require_whole(
        resident + gram_bytes + scratch as u64 + ((width + 5 * m) * 8) as u64 + 4096,
        "centered PCA and eigensolver workspace",
    )?;
    let mut means = vec![0.; width];
    for row in &set.vectors {
        for (mean, &x) in means.iter_mut().zip(row) {
            *mean += f64::from(x);
        }
    }
    for mean in &mut means {
        *mean /= n as f64;
    }
    let mut gram = vec![0.; m * m];
    let mut centered = vec![0.; m];
    for i in 0..n.max(width) {
        if width <= n {
            for ((v, &x), mean) in centered.iter_mut().zip(&set.vectors[i]).zip(&means) {
                *v = f64::from(x) - mean;
            }
        } else {
            for (v, row) in centered.iter_mut().zip(&set.vectors) {
                *v = f64::from(row[i]) - means[i];
            }
        }
        for (col, &a) in centered.iter().enumerate() {
            for (entry, &b) in gram[col * m + col..(col + 1) * m]
                .iter_mut()
                .zip(&centered[col..])
            {
                *entry += a * b;
            }
        }
    }
    let trace = (0..m).map(|i| gram[i * m + i]).sum::<f64>();
    if trace == 0. {
        return Ok(Vec::new());
    }
    let values = MatRef::from_column_major_slice(&gram, m, m)
        .self_adjoint_eigenvalues(Side::Lower)
        .map_err(|e| L2dError::Runtime {
            detail: format!("PCA eigendecomposition: {e:?}"),
        })?;
    Ok(values
        .iter()
        .rev()
        .map(|v| (v.max(0.) / trace) as f32)
        .collect())
}

// Under the locally homogeneous TwoNN model log(r2/r1) has an exponential
// distribution. Denti et al. (2022), doi:10.1038/s41598-022-20991-1,
// give the finite-sample corrected likelihood estimator (n-1)/sum(log(mu)).
fn twonn(set: &VectorSet, sample: &[usize]) -> Option<f64> {
    if sample.len() < 3 {
        return None;
    }
    let mut sum = 0.;
    for &i in sample {
        let (mut first, mut second) = (f64::INFINITY, f64::INFINITY);
        for &j in sample {
            if i == j {
                continue;
            }
            let d = set.vectors[i]
                .iter()
                .zip(&set.vectors[j])
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>();
            if d < first {
                second = first;
                first = d;
            } else if d < second {
                second = d;
            }
        }
        if first == 0. || first == second {
            return None;
        }
        sum += 0.5 * ((second - first) / first).ln_1p();
    }
    Some((sample.len() - 1) as f64 / sum)
}
