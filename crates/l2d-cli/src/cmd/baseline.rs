//! `l2d baseline`: fit or apply the baseline scorer.

use clap::Subcommand;
use l2d_core::L2dError;

/// The operations below `l2d baseline`.
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum BaselineCommand {
    /// Fit the baseline scorer on labelled logs
    Fit,
    /// Score logs with a fitted baseline
    Score,
}

/// Runs one `l2d baseline` operation.
pub fn run(command: &BaselineCommand) -> Result<(), L2dError> {
    match command {
        BaselineCommand::Fit => Err(super::not_implemented("baseline fit")),
        BaselineCommand::Score => Err(super::not_implemented("baseline score")),
    }
}
