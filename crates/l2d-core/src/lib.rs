//! Shared record, label, identity, run-record, path, and error types.

pub mod adapters;
pub mod error;
pub mod input;
pub mod paths;
pub mod runrecord;
pub mod types;

pub use adapters::labelled::{LabelledAdapter, LabelledFormat};
pub use adapters::lines::LinesAdapter;
pub use input::{Dedup, InputAdapter};

pub use error::{
    EXIT_INVALID_INPUT, EXIT_MISSING, EXIT_REFUSED, EXIT_RUNTIME, EXIT_SUCCESS, L2dError,
};
pub use paths::ProjectPaths;
pub use runrecord::{Device, Placement, PlacementSource, RunKind, RunRecord};
pub use types::{EncoderIdentity, InputSummary, Label, LogRecord, SkippedRecord};
