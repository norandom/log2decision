//! One module per subcommand of the design's command table.

use l2d_core::L2dError;

pub mod assess;
pub mod baseline;
pub mod calibrate;
pub mod common;
pub mod encoder;
pub mod eval;
pub mod ingest;
pub mod runtime;
pub mod shift;
pub mod store;
pub mod triage;

/// The failure of an operation that is offered on the command line but not wired yet.
///
/// `operation` is what the analyst typed after `l2d`, for example `baseline fit`.
pub(crate) fn not_implemented(operation: &str) -> L2dError {
    L2dError::Refused {
        reason: format!("`l2d {operation}` is not implemented yet"),
    }
}
