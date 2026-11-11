//! `l2d store`: fill and query a store.

use clap::Subcommand;
use l2d_core::L2dError;

/// The operations below `l2d store`.
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum StoreCommand {
    /// Add logs and their vectors to a store
    Add,
    /// Search a store by vector, text or metadata
    Search,
}

/// Runs one `l2d store` operation.
pub fn run(command: &StoreCommand) -> Result<(), L2dError> {
    match command {
        StoreCommand::Add => Err(super::not_implemented("store add")),
        StoreCommand::Search => Err(super::not_implemented("store search")),
    }
}
