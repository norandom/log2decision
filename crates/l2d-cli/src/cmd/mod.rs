//! One module per subcommand of the design's command table.

use l2d_core::L2dError;

pub mod analyst;
mod analyst_source;
mod analyst_triage;
pub mod archive_run;
pub mod assess;
pub mod baseline;
pub mod baseline_archive;
pub mod calibrate;
pub mod common;
pub mod discover;
mod discover_store;
pub mod encoder;
pub mod eval;
pub mod ingest;
pub mod mark;
pub mod runtime;
pub mod shift;
pub mod store;
pub mod triage;
pub mod tui;

/// The failure of an operation that is offered on the command line but not wired yet.
///
/// `operation` is what the analyst typed after `l2d`, for example `baseline fit`.
pub(crate) fn not_implemented(operation: &str) -> L2dError {
    L2dError::Refused {
        reason: format!("`l2d {operation}` is not implemented yet"),
    }
}
