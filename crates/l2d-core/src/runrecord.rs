//! The one run-record shape for training, encoding and evaluation, and its JSON form.

use serde::{Deserialize, Serialize};

use crate::error::L2dError;
use crate::memory::BatchPlan;
use crate::types::{EncoderIdentity, InputSummary};

/// The key of the batch plan in [`RunRecord::settings`].
pub const BATCH_PLAN_KEY: &str = "batch_plan";

/// What kind of run a record describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    Train,
    Encode,
    Evaluate,
}

/// Where the encoder ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    Npu,
    Cpu,
}

/// Where the placement figures came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementSource {
    ProviderReport,
    Unavailable,
}

/// How many model nodes ran on the NPU and on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    pub npu_nodes: u32,
    pub cpu_nodes: u32,
    pub source: PlacementSource,
}

/// The record written for every training, encoding and evaluation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub kind: RunKind,
    pub started_utc: String,
    pub duration_ms: u64,
    pub inputs: Vec<InputSummary>,
    pub settings: serde_json::Value,
    pub seed: Option<u64>,
    pub encoder: Option<EncoderIdentity>,
    pub device: Option<Device>,
    pub driver_version: Option<String>,
    pub logs_per_second: Option<f64>,
    pub placement: Option<Placement>,
}

impl RunRecord {
    /// Stores the batch plan of the run (requirement 12.3) in the settings, under
    /// [`BATCH_PLAN_KEY`]: free memory observed, limit, reserve, bytes per row, first
    /// batch, whether the analyst overrode it, every change, smallest and largest batch.
    ///
    /// The settings must be a JSON object or empty (`null`); other settings are left
    /// as they are and the call fails with [`L2dError::Runtime`].
    pub fn record_batch_plan(&mut self, plan: &BatchPlan) -> Result<(), L2dError> {
        let failed = |what: String| L2dError::Runtime {
            detail: format!("the batch plan could not be stored in the run record: {what}"),
        };
        let value = serde_json::to_value(plan).map_err(|error| failed(error.to_string()))?;
        if self.settings.is_null() {
            self.settings = serde_json::Value::Object(serde_json::Map::new());
        }
        match &mut self.settings {
            serde_json::Value::Object(settings) => {
                settings.insert(BATCH_PLAN_KEY.to_string(), value);
                Ok(())
            }
            _ => Err(failed("the settings are not a JSON object".into())),
        }
    }

    /// The record as indented JSON.
    pub fn to_json(&self) -> Result<String, L2dError> {
        serde_json::to_string_pretty(self).map_err(|error| L2dError::Runtime {
            detail: format!("run record could not be written as JSON: {error}"),
        })
    }

    /// Reads a record from its JSON form.
    pub fn from_json(json: &str) -> Result<Self, L2dError> {
        serde_json::from_str(json).map_err(|error| L2dError::InvalidInput {
            detail: format!("run record is not valid JSON: {error}"),
        })
    }
}
