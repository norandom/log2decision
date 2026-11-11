//! `l2d encoder`: train an encoder, encode logs, show an encoder identity.

use clap::Subcommand;
use l2d_core::L2dError;

/// The operations below `l2d encoder`.
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum EncoderCommand {
    /// Train an encoder and write its package
    Train,
    /// Encode logs into vectors
    Encode,
    /// Show the identity of an encoder package
    Info,
}

/// Runs one `l2d encoder` operation.
pub fn run(command: &EncoderCommand) -> Result<(), L2dError> {
    match command {
        EncoderCommand::Train => Err(super::not_implemented("encoder train")),
        EncoderCommand::Encode => Err(super::not_implemented("encoder encode")),
        EncoderCommand::Info => Err(super::not_implemented("encoder info")),
    }
}
