//! Store fills and external distance queries use the identical pinned encoder adapter.
use super::common::Context;
use super::encoder::{EncodeArgs, Operation, encoded};
use clap::{Args, Subcommand};
use l2d_core::L2dError;

#[derive(Debug, Clone, Args)]
pub struct StoreInput {
    #[arg(long)]
    pub store: String,
    #[command(flatten)]
    pub encode: EncodeArgs,
}
#[derive(Debug, Clone, Args)]
pub struct Outliers {
    #[command(flatten)]
    pub input: StoreInput,
    /// Rank by distance to the k-th nearest eligible stored record
    #[arg(long)]
    pub k: Option<usize>,
}
#[derive(Debug, Clone, Subcommand)]
pub enum StoreCommand {
    /// Add the original records and their NPU vectors in one transaction
    Add(Box<StoreInput>),
    /// Search a store by vector, text or metadata
    Search,
}
pub fn run(context: &Context, command: &StoreCommand) -> Result<(), L2dError> {
    match command {
        StoreCommand::Add(args) => encoded(context, &args.encode, Operation::StoreAdd(&args.store)),
        StoreCommand::Search => Err(super::not_implemented("store search")),
    }
}
pub fn outliers(context: &Context, args: &Outliers) -> Result<(), L2dError> {
    encoded(
        context,
        &args.input.encode,
        Operation::Outliers(&args.input.store, args.k),
    )
}
