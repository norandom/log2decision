//! `l2d runtime`: check the runtime files, the NPU and its driver.

use clap::Subcommand;
use l2d_core::L2dError;

/// The operations below `l2d runtime`.
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum RuntimeCommand {
    /// Check the runtime files, the NPU device and the driver
    Check,
}

/// Runs one `l2d runtime` operation.
pub fn run(command: &RuntimeCommand) -> Result<(), L2dError> {
    match command {
        RuntimeCommand::Check => Err(super::not_implemented("runtime check")),
    }
}
