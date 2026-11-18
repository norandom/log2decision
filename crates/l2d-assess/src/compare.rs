use crate::{
    ClusterReport, LossReport, NeighbourOverlap, PropertiesReport, ShortenedResult, VectorSet,
    assess_clusters_with, assess_loss_with, assess_properties_with, invalid,
    neighbour_overlap_with,
};
use l2d_core::{L2dError, MemoryBudget, MemorySettings};
use std::fmt;

#[derive(Debug)]
pub struct ComparisonColumn {
    pub name: String,
    pub originals: usize,
    pub loss: LossReport,
    pub clusters: ClusterReport,
    pub properties: PropertiesReport,
}

#[derive(Debug)]
pub struct ComparisonReport {
    pub left: ComparisonColumn,
    pub right: ComparisonColumn,
    /// Direct comparison between encoders on the same sampled originals.
    pub neighbour_overlap: NeighbourOverlap,
}

/// Compare aligned encoders with two clusters and up to ten nearest neighbours.
/// Leave at least one candidate outside each neighbourhood so the overlap can vary.
/// Both loss assessments also evaluate the same prefix, half the smaller width
/// rounded down, when that leaves at least one coordinate. Callers align rows
/// by original log; equal labels and group keys alone cannot establish identity.
pub fn compare(a: &VectorSet, b: &VectorSet, seed: u64) -> Result<ComparisonReport, L2dError> {
    compare_with(
        a,
        b,
        seed,
        &mut MemoryBudget::observe(&MemorySettings::default())?,
    )
}

/// Use a shared budget for both resident inputs, all retained report data and
/// the current measurement workspace. Restore the caller's hold on every exit.
pub fn compare_with(
    a: &VectorSet,
    b: &VectorSet,
    seed: u64,
    budget: &mut MemoryBudget,
) -> Result<ComparisonReport, L2dError> {
    a.validate()?;
    let same = std::ptr::eq(a, b);
    if !same {
        b.validate()?;
    }
    if a.vectors.len() != b.vectors.len() || a.labels != b.labels || a.groups != b.groups {
        return Err(invalid(
            "comparison requires the same originals in the same order, with matching labels and group keys",
        ));
    }
    let n = a.vectors.len();
    if n < 3 {
        return Err(invalid(
            "comparison with two clusters requires at least three originals",
        ));
    }
    let a_width =
        u16::try_from(a.vectors[0].len()).map_err(|_| invalid("dimension exceeds u16"))?;
    let b_width =
        u16::try_from(b.vectors[0].len()).map_err(|_| invalid("dimension exceeds u16"))?;
    let half = a_width.min(b_width) / 2;
    let prefix = (half > 0).then_some(half);
    let a_bytes = a.resident_bytes();
    let b_bytes = if same { 0 } else { b.resident_bytes() };
    let report_bytes =
        size_of::<ComparisonReport>() as u64 + a.name.len() as u64 + b.name.len() as u64;
    budget.require_whole(
        a_bytes + b_bytes + report_bytes,
        "comparison inputs, column names and report",
    )?;
    let held = budget.held_bytes();
    let base = held + report_bytes;
    let result = (|| {
        budget.hold(base);
        let neighbour_overlap = neighbour_overlap_with(a, b, 10.min(n - 2), seed, budget)?;
        // Each component charges its own input, so hold just the other input
        // alongside previously completed measurements.
        budget.hold(base + b_bytes);
        let left = column(a, prefix.as_slice(), seed, budget)?;
        budget.hold(base + retained_bytes(&left.loss) + if same { 0 } else { a_bytes });
        let right = column(b, prefix.as_slice(), seed, budget)?;
        Ok(ComparisonReport {
            left,
            right,
            neighbour_overlap,
        })
    })();
    budget.hold(held);
    result
}

// All other report fields are inline or covered by the name reservation.
fn retained_bytes(loss: &LossReport) -> u64 {
    (loss.explained_variance.capacity() * size_of::<f32>()
        + loss.shortened.capacity() * size_of::<ShortenedResult>()) as u64
}

fn column(
    set: &VectorSet,
    prefix: &[u16],
    seed: u64,
    budget: &mut MemoryBudget,
) -> Result<ComparisonColumn, L2dError> {
    let loss = assess_loss_with(set, prefix, seed, budget)?;
    budget.hold(budget.held_bytes() + retained_bytes(&loss));
    let clusters = assess_clusters_with(set, 2, seed, budget)?;
    let properties = assess_properties_with(set, seed, budget)?;
    Ok(ComparisonColumn {
        name: set.name.clone(),
        originals: set.vectors.len(),
        loss,
        clusters,
        properties,
    })
}

fn optional<T: fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "N/A".into(), |x| x.to_string())
}

/// A tab-separated table. Names are escaped so control characters cannot
/// introduce extra rows or columns. The cross-encoder overlap is relative to the first set.
impl fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (a, b) = (&self.left, &self.right);
        macro_rules! row {
            ($label:expr, $a:expr, $b:expr) => {
                writeln!(f, "{}\t{}\t{}", $label, $a, $b)?
            };
        }
        row!("Measurement", a.name.escape_debug(), b.name.escape_debug());
        row!("Originals", a.originals, b.originals);
        row!("Sample size", a.loss.sample_size, b.loss.sample_size);
        row!("Seed", a.loss.seed, b.loss.seed);
        row!("Dimension", a.properties.dimension, b.properties.dimension);
        row!(
            "Explained variance",
            format_args!("{:?}", a.loss.explained_variance),
            format_args!("{:?}", b.loss.explained_variance)
        );
        row!(
            "Intrinsic dimension",
            optional(a.loss.intrinsic_dimension),
            optional(b.loss.intrinsic_dimension)
        );
        row!(
            "Full-width AUC",
            optional(a.loss.separability.as_ref().map(|s| s.auc)),
            optional(b.loss.separability.as_ref().map(|s| s.auc))
        );
        for (l, r) in a.loss.shortened.iter().zip(&b.loss.shortened) {
            row!("Prefix dimension", l.dimension, r.dimension);
            row!("Prefix neighbours", l.k, r.k);
            row!(
                "Prefix overlap vs own full width",
                l.neighbour_overlap,
                r.neighbour_overlap
            );
            row!(
                "Prefix AUC",
                optional(l.separability_auc),
                optional(r.separability_auc)
            );
        }
        row!("Clusters", a.clusters.k, b.clusters.k);
        row!("Silhouette", a.clusters.silhouette, b.clusters.silhouette);
        row!(
            "Separability AUC",
            optional(a.clusters.separability_auc),
            optional(b.clusters.separability_auc)
        );
        row!(
            "Within-group AUC",
            optional(a.clusters.within_group_auc),
            optional(b.clusters.within_group_auc)
        );
        row!(
            "Evaluated groups",
            a.clusters.evaluated_groups,
            b.clusters.evaluated_groups
        );
        row!(
            "Cosine quantiles",
            format_args!("{:?}", a.properties.cosine_quantiles),
            format_args!("{:?}", b.properties.cosine_quantiles)
        );
        row!(
            "Norm quantiles",
            format_args!("{:?}", a.properties.norm_quantiles),
            format_args!("{:?}", b.properties.norm_quantiles)
        );
        row!(
            "Cosine pairs",
            a.properties.cosine_pairs,
            b.properties.cosine_pairs
        );
        row!(
            "Anisotropy",
            a.properties.anisotropy,
            b.properties.anisotropy
        );
        row!(
            "Duplicate rate",
            a.properties.duplicate_rate,
            b.properties.duplicate_rate
        );
        row!(
            "Neighbours",
            self.neighbour_overlap.k,
            self.neighbour_overlap.k
        );
        row!(
            "Neighbour overlap vs first",
            1,
            self.neighbour_overlap.overlap
        );
        Ok(())
    }
}
