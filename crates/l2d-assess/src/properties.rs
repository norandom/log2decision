use crate::overlap::{SAMPLE_CAP, sample_indices};
use crate::{VectorSet, invalid};
use l2d_core::{L2dError, MemoryBudget, MemorySettings};

#[derive(Debug, PartialEq)]
pub struct PropertiesReport {
    pub dimension: u16,
    /// Min, lower quartile, median, upper quartile, max (linear interpolation).
    pub cosine_quantiles: [f32; 5],
    /// Same quantiles of Euclidean norms over all originals, including zeros.
    pub norm_quantiles: [f32; 5],
    /// Mean cosine among distinct unordered sample rows with defined cosine.
    pub anisotropy: f32,
    /// Fraction of originals beyond the first occurrence of each exact vector.
    pub duplicate_rate: f32,
    pub sample_size: usize,
    pub cosine_pairs: usize,
    pub seed: u64,
}

pub fn assess_properties(set: &VectorSet, seed: u64) -> Result<PropertiesReport, L2dError> {
    assess_properties_with(
        set,
        seed,
        &MemoryBudget::observe(&MemorySettings::default())?,
    )
}

pub fn assess_properties_with(
    set: &VectorSet,
    seed: u64,
    budget: &MemoryBudget,
) -> Result<PropertiesReport, L2dError> {
    set.validate()?;
    let n = set.vectors.len();
    if n < 2 {
        return Err(invalid("cosine distribution needs at least two vectors"));
    }
    let dimension =
        u16::try_from(set.vectors[0].len()).map_err(|_| invalid("dimension exceeds u16"))?;
    let sample_size = n.min(SAMPLE_CAP);
    let pairs = sample_size * (sample_size - 1) / 2;
    // Norms and ordered row indices (no copies of the vectors), sample
    // indices, and one f32 per unordered pair. All sorts are in place.
    budget.require_whole(
        set.resident_bytes()
            .saturating_add(n as u64 * 16)
            .saturating_add((sample_size * 8 + pairs * 4 + 4096) as u64),
        "vector properties inputs and workspace",
    )?;
    let mut norms = set
        .vectors
        .iter()
        .map(|v| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt())
        .collect::<Vec<_>>();
    if norms.iter().any(|&norm| norm > f64::from(f32::MAX)) {
        return Err(invalid("vector norm exceeds the f32 report range"));
    }
    let sample = sample_indices(n, seed);
    let mut cosines = Vec::with_capacity(pairs);
    let mut sum = 0.;
    for (slot, &i) in sample.iter().enumerate() {
        for &j in &sample[..slot] {
            let denominator = norms[i] * norms[j];
            if denominator != 0. {
                let dot: f64 = set.vectors[i]
                    .iter()
                    .zip(&set.vectors[j])
                    .map(|(&x, &y)| f64::from(x) * f64::from(y))
                    .sum();
                let cosine = (dot / denominator).clamp(-1., 1.);
                cosines.push(cosine as f32);
                sum += cosine;
            }
        }
    }
    let cosine_pairs = cosines.len();
    if cosine_pairs == 0 {
        return Err(invalid(
            "cosine distribution needs two nonzero sampled vectors",
        ));
    }
    let mut rows: Vec<_> = (0..n).collect();
    // Validation rules out NaN. Numeric ordering treats +0 and -0 as equal.
    rows.sort_unstable_by(|&a, &b| set.vectors[a].partial_cmp(&set.vectors[b]).unwrap());
    let duplicates = rows
        .windows(2)
        .filter(|pair| set.vectors[pair[0]] == set.vectors[pair[1]])
        .count();
    norms.sort_unstable_by(f64::total_cmp);
    cosines.sort_unstable_by(f32::total_cmp);
    Ok(PropertiesReport {
        dimension,
        cosine_quantiles: quartiles(&cosines),
        norm_quantiles: quartiles(&norms),
        anisotropy: (sum / cosine_pairs as f64) as f32,
        duplicate_rate: (duplicates as f64 / n as f64) as f32,
        sample_size,
        cosine_pairs,
        seed,
    })
}

fn quartiles<T: Copy + Into<f64>>(sorted: &[T]) -> [f32; 5] {
    std::array::from_fn(|i| {
        let position = (sorted.len() - 1) as f64 * i as f64 / 4.;
        let low = sorted[position.floor() as usize].into();
        let high = sorted[position.ceil() as usize].into();
        (low + (high - low) * position.fract()) as f32
    })
}
