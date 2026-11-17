//! Descriptive figures for actual neighbours; no neighbour classifier.
use serde::Serialize;
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct LabelEvidence {
    pub count: usize,
    /// Absent for an empty label. The enclosing evidence names the metric.
    pub mean_distance: Option<f64>,
}
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct NeighbourEvidence {
    pub bad: LabelEvidence,
    pub good: LabelEvidence,
    /// Unlabelled records are not verified good.
    pub unknown: LabelEvidence,
}
impl NeighbourEvidence {
    pub(crate) fn add(&mut self, label: &str, distance: f64) {
        let value = match label {
            "bad" => &mut self.bad,
            "good" => &mut self.good,
            _ => &mut self.unknown,
        };
        let sum = value.mean_distance.unwrap_or(0.) * value.count as f64;
        value.count += 1;
        value.mean_distance = Some((sum + distance) / value.count as f64);
    }
}
