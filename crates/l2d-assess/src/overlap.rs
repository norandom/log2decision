use crate::{VectorSet, invalid};
use l2d_core::{L2dError, MemoryBudget, MemorySettings};
use rand::{Rng, SeedableRng, rngs::SmallRng};

pub(crate) const SAMPLE_CAP: usize = 5000;

#[derive(Debug, PartialEq)]
pub struct NeighbourOverlap {
    /// Mean intersection size divided by k, excluding the query itself.
    pub overlap: f32,
    pub sample_size: usize,
    pub seed: u64,
    pub k: usize,
}

/// Compare the same originals, using seed zero and the host memory budget.
pub fn neighbour_overlap(
    set: &VectorSet,
    reference: &VectorSet,
    k: usize,
) -> Result<NeighbourOverlap, L2dError> {
    let budget = MemoryBudget::observe(&MemorySettings::default())?;
    neighbour_overlap_with(set, reference, k, 0, &budget)
}

pub(crate) fn sample_indices(total: usize, seed: u64) -> Vec<usize> {
    let mut sample: Vec<_> = (0..total.min(SAMPLE_CAP)).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    for i in SAMPLE_CAP..total {
        let slot = rng.random_range(0..=i);
        if slot < SAMPLE_CAP {
            sample[slot] = i;
        }
    }
    sample
}

type Candidate = (f64, usize);

// Reuse one linear work buffer per representation. Partial selection avoids
// sorting every candidate. Original row index resolves all distance ties.
fn neighbours<'a>(
    set: &VectorSet,
    sample: &[usize],
    query: usize,
    k: usize,
    scratch: &'a mut Vec<Candidate>,
) -> &'a [Candidate] {
    scratch.clear();
    for &i in sample {
        if i == query {
            continue;
        }
        let distance = set.vectors[query]
            .iter()
            .zip(&set.vectors[i])
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        scratch.push((distance, i));
    }
    scratch.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let nearest = &mut scratch[..k];
    nearest.sort_unstable_by_key(|entry| entry.1);
    nearest
}

/// Compare exact squared-L2 neighbours among one common seeded sample. Both
/// query and candidate populations are that sample; a row never finds itself.
/// The caller supplies aligned sets and a budget counting other live memory.
pub fn neighbour_overlap_with(
    set: &VectorSet,
    reference: &VectorSet,
    k: usize,
    seed: u64,
    budget: &MemoryBudget,
) -> Result<NeighbourOverlap, L2dError> {
    set.validate()?;
    if !std::ptr::eq(set, reference) {
        reference.validate()?;
    }
    if set.vectors.len() != reference.vectors.len() {
        return Err(invalid(
            "reference and query set must contain the same original logs in the same order",
        ));
    }
    let n = set.vectors.len().min(SAMPLE_CAP);
    if k == 0 || k >= n {
        return Err(invalid(format!(
            "neighbour count {k} must be positive and smaller than the sample of {n} originals"
        )));
    }
    let mut bytes = set.resident_bytes();
    if !std::ptr::eq(set, reference) {
        bytes += reference.resident_bytes();
    }
    budget.require_whole(
        bytes + (n * (2 * size_of::<Candidate>() + size_of::<usize>())) as u64 + 4096,
        "neighbour overlap inputs and sampled workspace",
    )?;
    let sample = sample_indices(set.vectors.len(), seed);
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    let mut matches = 0;
    for &query in &sample {
        let a = neighbours(set, &sample, query, k, &mut left);
        let b = neighbours(reference, &sample, query, k, &mut right);
        let mut j = 0;
        for &(_, i) in a {
            while j < k && b[j].1 < i {
                j += 1;
            }
            matches += usize::from(j < k && b[j].1 == i);
        }
    }
    Ok(NeighbourOverlap {
        overlap: (matches as f64 / (n * k) as f64) as f32,
        sample_size: n,
        seed,
        k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservoir_is_capped_shared_and_reproducible() {
        let a = sample_indices(25_000, 17);
        assert_eq!(a.len(), 5000);
        assert_eq!(a, sample_indices(25_000, 17));
        assert_ne!(a, sample_indices(25_000, 18));
        let mut unique = a.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), a.len());
        assert!(unique[4999] >= 5000 && unique[4999] < 25000);
    }
}
