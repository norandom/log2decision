//! `l2d runtime`: check the runtime files, the NPU and its driver.

use super::common::Context;
use clap::Subcommand;
use l2d_core::L2dError;
use l2d_encoder::{CheckDepth, NpuStatus, RuntimeFiles};
use serde_json::json;

/// The operations below `l2d runtime`.
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum RuntimeCommand {
    /// Check the runtime files, the NPU device and the driver
    Check,
}

/// Runs one `l2d runtime` operation.
pub fn run(context: &Context, command: &RuntimeCommand) -> Result<(), L2dError> {
    match command {
        RuntimeCommand::Check => {
            let paths = context.paths()?;
            let budget = context.budget()?;
            let dir =
                paths.runtime_files_dir(l2d_core::paths::runtime_dir_from_environment().as_deref());
            let files = RuntimeFiles::check(&dir, CheckDepth::Full)?;
            let npu = NpuStatus::query()?;
            crate::output::report(
                context,
                &budget,
                &json!({
                    "runtime_directory": dir, "runtime_release": files.release,
                    "onnxruntime": files.onnxruntime, "manifest_source": files.source,
                    "hashed_files": files.hashed, "record": files.record.describe(),
                    "npu_present": npu.is_present(), "driver_version": npu.driver_version(),
                    "device_evidence": npu.log_lines()
                }),
            )
        }
    }
}
