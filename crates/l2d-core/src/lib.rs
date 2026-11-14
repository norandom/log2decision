//! Shared record, label, identity, run-record, path, and error types, the input
//! adapters, and the data layer.
//!
//! # Data layer
//!
//! Polars is the one tabular data layer of the workspace. Tabular data crosses crate
//! boundaries as Polars frames with the fixed schemas of the design's "Data frames"
//! section; [`LogsFrame`] is the first of them. Other crates do not depend on Polars
//! themselves: they import it from here (`use l2d_core::polars::prelude::*;`), so the
//! whole workspace uses the one version and feature set chosen in the workspace
//! manifest.

pub use polars;

pub mod adapters;
pub mod error;
pub mod input;
pub mod memory;
pub mod parts;
pub mod paths;
pub mod runrecord;
pub mod types;
mod zip_directory;
pub mod zip_input;
mod zip_spill;

pub use adapters::csic2010::{Csic2010Adapter, Csic2010Counts};
pub use adapters::labelled::{LabelledAdapter, LabelledFormat};
pub use adapters::lines::LinesAdapter;
pub use input::{Accepted, Dedup, InputAdapter, Source};
pub use memory::{BatchChange, BatchPlan, MemoryBudget, MemorySettings};

pub use error::{
    EXIT_INVALID_INPUT, EXIT_MISSING, EXIT_REFUSED, EXIT_RUNTIME, EXIT_SUCCESS, L2dError,
};
pub use paths::ProjectPaths;
pub use runrecord::{BATCH_PLAN_KEY, Device, Placement, PlacementSource, RunKind, RunRecord};
pub use types::{EncoderIdentity, InputSummary, Label, LogRecord, LogsFrame, SkippedRecord};
pub use zip_input::{
    ArchivePosition, ArchiveSummary, DiskDedup, EntryReport, EntrySkip, FreezeDisk,
    FreezeDiskCheck, SelectedBytes, ZipInput, ZipInputSettings, most_key_file_bytes,
};

pub use parts::{PartInfo, PartTable, PartWriter, WorkDir};
