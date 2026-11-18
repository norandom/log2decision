//! Shared measurements for vector representations of logs.
mod clusters;
mod loss;
mod overlap;
mod projection;
mod properties;
mod separability;

pub use clusters::{ClusterReport, assess_clusters, assess_clusters_with};
use l2d_core::{L2dError, Label};
pub use loss::{LossReport, ShortenedResult, assess_loss, assess_loss_with};
pub use overlap::{NeighbourOverlap, neighbour_overlap, neighbour_overlap_with};
pub use projection::{project_to_csv, project_to_csv_with};
pub use properties::{PropertiesReport, assess_properties, assess_properties_with};
pub use separability::{Separability, separability, separability_with};

/// One vector per original log. When comparing representations, callers must
/// align both sets by original log before constructing them. Groups can repeat
/// and cannot establish row identity. Dimensions may differ between sets.
#[derive(Debug)]
pub struct VectorSet {
    pub vectors: Vec<Vec<f32>>,
    pub labels: Option<Vec<Label>>,
    pub groups: Option<Vec<String>>,
    pub name: String,
}

fn invalid(message: impl Into<String>) -> L2dError {
    L2dError::InvalidInput {
        detail: message.into(),
    }
}

impl VectorSet {
    /// Check row-aligned metadata and finite, nonempty, equal-width vectors.
    pub fn validate(&self) -> Result<(), L2dError> {
        let n = self.vectors.len();
        if self.labels.as_ref().is_some_and(|v| v.len() != n)
            || self.groups.as_ref().is_some_and(|v| v.len() != n)
        {
            return Err(invalid(
                "vector set labels and group keys must match the number of rows",
            ));
        }
        let width = self.vectors.first().map_or(0, Vec::len);
        for (i, row) in self.vectors.iter().enumerate() {
            if width == 0 || row.len() != width || !row.iter().all(|x| x.is_finite()) {
                return Err(invalid(format!(
                    "row {i} must have finite coordinates and the same nonzero width as the other vectors"
                )));
            }
        }
        Ok(())
    }

    // Include spare capacities in these caller-owned inputs. The budget passed
    // to a measure accounts for any other live allocations in its held bytes.
    fn resident_bytes(&self) -> u64 {
        let mut bytes = size_of::<Self>()
            + self.name.capacity()
            + self.vectors.capacity() * size_of::<Vec<f32>>();
        bytes += self
            .vectors
            .iter()
            .map(|v| v.capacity() * size_of::<f32>())
            .sum::<usize>();
        if let Some(labels) = &self.labels {
            bytes += labels.capacity() * size_of::<Label>();
        }
        if let Some(groups) = &self.groups {
            bytes += groups.capacity() * size_of::<String>()
                + groups.iter().map(String::capacity).sum::<usize>();
        }
        bytes as u64
    }
}
