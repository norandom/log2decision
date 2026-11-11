//! The tool-wide error envelope and its exit-code mapping.

use std::path::PathBuf;

/// Exit code of a successful run.
pub const EXIT_SUCCESS: u8 = 0;
/// Exit code for analyst errors: unknown adapter, bad option values, unreadable file format.
pub const EXIT_INVALID_INPUT: u8 = 2;
/// Exit code for a missing encoder, baseline, store, dataset or runtime file.
pub const EXIT_MISSING: u8 = 3;
/// Exit code for an operation the tool refuses to perform.
pub const EXIT_REFUSED: u8 = 4;
/// Exit code for provider, database and other runtime failures.
pub const EXIT_RUNTIME: u8 = 5;

/// The one error shape every operation ends in. Each message states what failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum L2dError {
    /// A required item was not found; names the item and where it was expected.
    #[error("missing {item}: expected at {}", .expected_at.display())]
    Missing { item: String, expected_at: PathBuf },
    /// The analyst supplied something unusable.
    #[error("invalid input: {detail}")]
    InvalidInput { detail: String },
    /// A rule forbids the operation; the reason states the rule and the observed values.
    #[error("refused: {reason}")]
    Refused { reason: String },
    /// An underlying provider, database or I/O failure.
    #[error("runtime failure: {detail}")]
    Runtime { detail: String },
}

impl L2dError {
    /// The process exit code for this error; never [`EXIT_SUCCESS`].
    pub fn exit_code(&self) -> u8 {
        match self {
            L2dError::InvalidInput { .. } => EXIT_INVALID_INPUT,
            L2dError::Missing { .. } => EXIT_MISSING,
            L2dError::Refused { .. } => EXIT_REFUSED,
            L2dError::Runtime { .. } => EXIT_RUNTIME,
        }
    }
}
