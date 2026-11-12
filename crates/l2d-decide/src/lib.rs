//! Decision operations: ratio with evidence and abstention, triage, population shift,
//! and the conformal calibration of a scorer's ratio.
//!
//! - [`scorer`]: the contract every scorer meets and the scores frame it returns.
//! - [`conformal`]: from a scorer's ratio to a label set with a stated error rate.
//! - [`triage`]: every log of a set ranked for review, scored down, or abstained on.

pub mod conformal;
pub mod scorer;
pub mod triage;

pub use conformal::{
    Calibrated, CalibrationError, CalibrationPoint, CalibrationSettings, ConformalCalibrator,
    LabelSet, Stratum, StratumCount,
};
pub use scorer::{Scorer, ScoringCost};
pub use triage::{
    AbstainReason, CalibrationUse, ListUse, ScoredDownBy, TriageCalibration, TriageCounts,
    TriageError, TriageList, TriageResult, TriageSettings, triage, triage_with,
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

#[cfg(test)]
mod tests {
    use super::ScorerKind;

    /// The kind after `kind` in the order of the declaration; the first for `None`.
    ///
    /// This match has no arm for "anything else" on purpose: a kind added to the enum
    /// without an arm here does not compile, and with an arm it is in the round trip
    /// below, which fails until `from_name` knows its name (`from_name` matches on
    /// strings, so the compiler cannot see a missing name there).
    fn next(kind: Option<ScorerKind>) -> Option<ScorerKind> {
        match kind {
            None => Some(ScorerKind::Baseline),
            Some(ScorerKind::Baseline) => Some(ScorerKind::Neighbour),
            Some(ScorerKind::Neighbour) => None,
        }
    }

    #[test]
    fn every_scorer_kind_is_found_again_by_its_name() {
        let mut names = Vec::new();
        let mut kind = next(None);
        while let Some(found) = kind {
            assert_eq!(ScorerKind::from_name(found.as_str()), Some(found));
            assert_eq!(
                serde_json::to_string(&found).unwrap(),
                format!("\"{}\"", found.as_str()),
                "the name in JSON is the name in frames"
            );
            assert!(!names.contains(&found.as_str()), "two kinds share a name");
            names.push(found.as_str());
            kind = next(kind);
        }
        assert_eq!(names, ["baseline", "neighbour"]);
        for other in ["", "Baseline", "baseline ", "neighbor"] {
            assert_eq!(ScorerKind::from_name(other), None, "'{other}'");
        }
    }
}
