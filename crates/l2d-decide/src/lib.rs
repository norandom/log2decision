//! Decision operations: ratio with evidence and abstention, triage, population shift,
//! and the conformal calibration of a scorer's ratio.

pub mod conformal;

pub use conformal::{
    Calibrated, CalibrationError, CalibrationPoint, CalibrationSettings, ConformalCalibrator,
    LabelSet, Stratum, StratumCount,
};

use serde::{Deserialize, Serialize};

/// Which scorer produced a ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScorerKind {
    /// Character n-grams and logistic regression (`l2d-baseline`).
    Baseline,
    /// Nearest neighbours in an encoder's vector space.
    Neighbour,
}

impl ScorerKind {
    /// The lower-case name used in frames, in JSON and in saved files.
    pub fn as_str(self) -> &'static str {
        match self {
            ScorerKind::Baseline => "baseline",
            ScorerKind::Neighbour => "neighbour",
        }
    }

    /// The kind a stored name stands for; `None` for anything but the exact names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "baseline" => Some(ScorerKind::Baseline),
            "neighbour" => Some(ScorerKind::Neighbour),
            _ => None,
        }
    }
}
