use crate::VectorSet;
use l2d_baseline::{LogReg, LogRegError, LogRegSettings};
use l2d_core::{L2dError, Label, MemoryBudget, MemorySettings};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

/// Exploratory, row-stratified separability. Group keys are retained for
/// group-specific assessments; these folds do not measure held-out-group transfer.
#[derive(Debug, PartialEq)]
pub struct Separability {
    /// Arithmetic mean of the held-out fold ROC AUCs, with half credit for ties.
    pub auc: f64,
    pub labelled: usize,
    pub folds: usize,
    pub seed: u64,
    pub converged_fits: usize,
}

pub fn separability(set: &VectorSet, seed: u64) -> Result<Option<Separability>, L2dError> {
    let budget = MemoryBudget::observe(&MemorySettings::default())?;
    separability_with(set, seed, &budget)
}

fn fit_error(e: LogRegError) -> L2dError {
    L2dError::Runtime {
        detail: format!("vector separability logistic regression: {e}"),
    }
}

/// Fit dense logistic regression separately on the training rows of each fold.
/// Use up to five folds, bounded by the minority-class count. Unknown rows
/// never enter fitting or scoring. With fewer than two known originals of
/// either class a held-out AUC cannot be measured and the result is absent.
/// The supplied budget should include other live allocations in its held bytes.
pub fn separability_with(
    set: &VectorSet,
    seed: u64,
    budget: &MemoryBudget,
) -> Result<Option<Separability>, L2dError> {
    set.validate()?;
    separability_rows(
        set,
        0..set.vectors.len(),
        seed,
        budget,
        set.resident_bytes(),
    )
}

/// Group assessments borrow original rows in input order. The resident estimate
/// includes the full input and any index storage retained by the caller.
pub(crate) fn separability_rows(
    set: &VectorSet,
    rows: impl Iterator<Item = usize> + Clone,
    seed: u64,
    budget: &MemoryBudget,
    resident: u64,
) -> Result<Option<Separability>, L2dError> {
    let Some(labels) = &set.labels else {
        return Ok(None);
    };
    let good = rows.clone().filter(|&i| labels[i] == Label::Good).count();
    let bad = rows.clone().filter(|&i| labels[i] == Label::Bad).count();
    let folds = 5.min(good).min(bad);
    if folds < 2 {
        return Ok(None);
    }
    let n = good + bad;
    let width = set.vectors[0].len() as u64;
    // fit_dense copies training coordinates into f64. The baseline scorer's
    // conservative L-BFGS estimate is 48 parameter and six row-sized vectors.
    // Include class indices, training references/labels and held-out scores.
    let bytes = resident
        .saturating_add((n as u64).saturating_mul(width.saturating_mul(8).saturating_add(128)))
        .saturating_add((width + 1).saturating_mul(48 * 8))
        .saturating_add(4096);
    budget.require_whole(
        bytes,
        "vector separability inputs and dense training workspace",
    )?;
    let mut classes = [Vec::with_capacity(good), Vec::with_capacity(bad)];
    for i in rows {
        match labels[i] {
            Label::Good => classes[0].push(i),
            Label::Bad => classes[1].push(i),
            Label::Unknown => {}
        }
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    for class in &mut classes {
        class.shuffle(&mut rng);
    }
    let mut sum = 0.;
    let mut converged_fits = 0;
    let mut train = Vec::with_capacity(n);
    let mut outcomes = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    for held_out in 0..folds {
        train.clear();
        outcomes.clear();
        for (class, indices) in classes.iter().enumerate() {
            for (pos, &i) in indices.iter().enumerate() {
                if pos % folds != held_out {
                    train.push(set.vectors[i].as_slice());
                    outcomes.push(class == 1);
                }
            }
        }
        let (model, fit) =
            LogReg::fit_dense(&train, &outcomes, &LogRegSettings::default()).map_err(fit_error)?;
        converged_fits += usize::from(fit.converged());
        scores.clear();
        for (class, indices) in classes.iter().enumerate() {
            for &i in indices.iter().skip(held_out).step_by(folds) {
                scores.push((
                    model
                        .probability_dense(&set.vectors[i])
                        .map_err(fit_error)?,
                    class == 1,
                ));
            }
        }
        sum += fold_auc(&mut scores);
    }
    Ok(Some(Separability {
        auc: sum / folds as f64,
        folds,
        labelled: n,
        seed,
        converged_fits,
    }))
}

// Count the fraction of positive-negative pairs for which the positive ranks
// higher. Fold stratification guarantees at least one member of each class.
fn fold_auc(scores: &mut [(f64, bool)]) -> f64 {
    scores.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut negatives = 0;
    let mut wins = 0.;
    for tie in scores.chunk_by(|a, b| a.0 == b.0) {
        let pos = tie.iter().filter(|s| s.1).count();
        let neg = tie.len() - pos;
        wins += pos as f64 * (negatives as f64 + 0.5 * neg as f64);
        negatives += neg;
    }
    wins / (negatives as f64 * (scores.len() - negatives) as f64)
}

#[cfg(test)]
mod tests {
    use super::fold_auc;

    #[test]
    fn mixed_ties_match_exhaustive_pair_comparisons() {
        let mut scores = [
            (2., true),
            (0., false),
            (1., true),
            (2., false),
            (0., true),
            (3., false),
        ];
        let mut wins = 0.;
        let mut pairs = 0.;
        for &(positive, _) in scores.iter().filter(|s| s.1) {
            for &(negative, _) in scores.iter().filter(|s| !s.1) {
                wins += if positive > negative {
                    1.
                } else if positive == negative {
                    0.5
                } else {
                    0.
                };
                pairs += 1.;
            }
        }
        assert_eq!(fold_auc(&mut scores), wins / pairs);
    }
}
