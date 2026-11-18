use crate::overlap::{SAMPLE_CAP, sample_indices};
use crate::separability::separability_rows;
use crate::{VectorSet, invalid};
use l2d_core::{L2dError, Label, MemoryBudget, MemorySettings};
use rand::{Rng, SeedableRng, rngs::SmallRng};

#[derive(Debug, PartialEq)]
pub struct ClusterReport {
    pub k: u32,
    /// Mean Euclidean silhouette on the sample. Singleton rows contribute zero.
    pub silhouette: f32,
    pub sample_size: usize,
    pub seed: u64,
    /// Held-out row-stratified dense LogReg AUC over all labelled originals.
    pub separability_auc: Option<f32>,
    /// Weighted by the number of labelled originals in each evaluated group.
    pub within_group_auc: Option<f32>,
    pub evaluated_groups: usize,
}

/// Fit KMeans and evaluate silhouette on one seeded sample of up to 5,000
/// originals. Evaluate label separability on the full population independently.
pub fn assess_clusters(set: &VectorSet, k: u32, seed: u64) -> Result<ClusterReport, L2dError> {
    assess_clusters_with(
        set,
        k,
        seed,
        &MemoryBudget::observe(&MemorySettings::default())?,
    )
}

/// Refuse before allocating if the whole input and current phase cannot fit.
pub fn assess_clusters_with(
    set: &VectorSet,
    k: u32,
    seed: u64,
    budget: &MemoryBudget,
) -> Result<ClusterReport, L2dError> {
    set.validate()?;
    let n = set.vectors.len();
    let sample_size = n.min(SAMPLE_CAP);
    if k < 2 || k as usize >= sample_size {
        return Err(invalid(format!(
            "silhouette requires 2 <= k < sample size {sample_size}, received {k}"
        )));
    }
    let resident = set.resident_bytes();
    let width = set.vectors[0].len() as u64;
    // Two contiguous centroid matrices, plus row indices, assignments, seed
    // weights, cluster counts and a distance accumulator. No pairwise matrix.
    let workspace = u64::from(k)
        .saturating_mul(width)
        .saturating_mul(16)
        .saturating_add((sample_size as u64) * 48 + 4096);
    budget.require_whole(
        resident.saturating_add(workspace),
        "cluster assessment inputs and KMeans workspace",
    )?;
    let silhouette = {
        let sample = sample_indices(n, seed);
        let (assignments, counts) = kmeans(set, &sample, k as usize, seed)?;
        silhouette(set, &sample, &assignments, &counts)
    };
    let separability_auc = separability_rows(set, 0..n, seed, budget, resident)?.map(|s| s.auc);
    let (within_group_auc, evaluated_groups) = keyed_auc(set, seed, budget, resident)?;
    Ok(ClusterReport {
        k,
        silhouette: silhouette as f32,
        sample_size,
        seed,
        separability_auc: separability_auc.map(|auc| auc as f32),
        within_group_auc: within_group_auc.map(|auc| auc as f32),
        evaluated_groups,
    })
}

fn keyed_auc(
    set: &VectorSet,
    seed: u64,
    budget: &MemoryBudget,
    resident: u64,
) -> Result<(Option<f64>, usize), L2dError> {
    let (Some(keys), Some(labels)) = (&set.groups, &set.labels) else {
        return Ok((None, 0));
    };
    let n = labels
        .iter()
        .filter(|&&label| label != Label::Unknown)
        .count();
    if n < 20 {
        return Ok((None, 0));
    }
    let bytes = resident + (n * size_of::<usize>()) as u64;
    budget.require_whole(bytes + 4096, "within-group separability index workspace")?;
    let mut rows = Vec::with_capacity(n);
    rows.extend((0..labels.len()).filter(|&i| labels[i] != Label::Unknown));
    // Preserve input order within each key for reproducible stratification.
    rows.sort_unstable_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));
    let mut sum = 0.;
    let mut total = 0;
    let mut evaluated = 0;
    for group in rows
        .chunk_by(|&a, &b| keys[a] == keys[b])
        .filter(|g| g.len() >= 20)
    {
        if let Some(score) = separability_rows(set, group.iter().copied(), seed, budget, bytes)? {
            sum += score.auc * score.labelled as f64;
            total += score.labelled;
            evaluated += 1;
        }
    }
    Ok(((total > 0).then(|| sum / total as f64), evaluated))
}

fn squared_distance(row: &[f32], center: &[f64]) -> f64 {
    row.iter()
        .zip(center)
        .map(|(&x, &c)| (f64::from(x) - c).powi(2))
        .sum()
}

fn kmeans(
    set: &VectorSet,
    sample: &[usize],
    k: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>), L2dError> {
    let width = set.vectors[0].len();
    let mut centers = Vec::with_capacity(k * width);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut next = rng.random_range(0..sample.len());
    let mut nearest = vec![f64::INFINITY; sample.len()];
    // KMeans++ selects distinct starting locations, with probability
    // proportional to squared distance from the closest existing center.
    for c in 0..k {
        centers.extend(set.vectors[sample[next]].iter().copied().map(f64::from));
        if c + 1 == k {
            break;
        }
        let center = &centers[c * width..];
        for (best, &i) in nearest.iter_mut().zip(sample) {
            *best = best.min(squared_distance(&set.vectors[i], center));
        }
        let total: f64 = nearest.iter().sum();
        if total == 0. {
            return Err(invalid("requested k exceeds distinct sampled vectors"));
        }
        let mut threshold = rng.random::<f64>() * total;
        for (i, &weight) in nearest.iter().enumerate() {
            if weight > 0. {
                next = i;
                threshold -= weight;
                if threshold < 0. {
                    break;
                }
            }
        }
    }
    let mut sums = vec![0.; k * width];
    let mut counts = vec![0; k];
    let mut assigned = vec![usize::MAX; sample.len()];
    for _ in 0..300 {
        sums.fill(0.);
        counts.fill(0);
        let mut changed = false;
        for (pos, &i) in sample.iter().enumerate() {
            let row = &set.vectors[i];
            let (c, distance) = centers
                .chunks_exact(width)
                .enumerate()
                .map(|(c, center)| (c, squared_distance(row, center)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            changed |= assigned[pos] != c;
            assigned[pos] = c;
            nearest[pos] = distance;
            counts[c] += 1;
        }
        // Restore an empty cluster with the worst represented row whose
        // donor cluster has another member, then recompute both centroids.
        for empty in 0..k {
            if counts[empty] == 0 {
                let donor = (0..sample.len())
                    .filter(|&i| counts[assigned[i]] > 1)
                    .max_by(|&a, &b| nearest[a].total_cmp(&nearest[b]))
                    .unwrap();
                counts[assigned[donor]] -= 1;
                counts[empty] = 1;
                assigned[donor] = empty;
                changed = true;
            }
        }
        if !changed {
            return Ok((assigned, counts));
        }
        for (&i, &c) in sample.iter().zip(&assigned) {
            for (sum, &x) in sums[c * width..(c + 1) * width]
                .iter_mut()
                .zip(&set.vectors[i])
            {
                *sum += f64::from(x);
            }
        }
        for ((center, sum), &count) in centers
            .chunks_exact_mut(width)
            .zip(sums.chunks_exact(width))
            .zip(&counts)
        {
            for (value, &x) in center.iter_mut().zip(sum) {
                *value = x / count as f64;
            }
        }
    }
    Err(L2dError::Runtime {
        detail: "KMeans did not stabilize after 300 iterations".into(),
    })
}

fn silhouette(set: &VectorSet, sample: &[usize], assigned: &[usize], counts: &[usize]) -> f64 {
    let mut sum = 0.;
    let mut distances = vec![0.; counts.len()];
    for (&i, &own) in sample.iter().zip(assigned) {
        if counts[own] == 1 {
            continue;
        }
        distances.fill(0.);
        for (&j, &c) in sample.iter().zip(assigned) {
            let distance = set.vectors[i]
                .iter()
                .zip(&set.vectors[j])
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum::<f64>()
                .sqrt();
            distances[c] += distance;
        }
        let a = distances[own] / (counts[own] - 1) as f64;
        let b = (0..counts.len())
            .filter(|&c| c != own)
            .map(|c| distances[c] / counts[c] as f64)
            .fold(f64::INFINITY, f64::min);
        if a.max(b) > 0. {
            sum += (b - a) / a.max(b);
        }
    }
    sum / sample.len() as f64
}
