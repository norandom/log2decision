//! Encoder package format, identity, and execution on NPU or CPU.
//!
//! So far (task 2.3) the two things that are settled before the runtime is loaded:
//! - [`runtime`]: the AMD runtime files, checked against a manifest ([`RuntimeFiles`]);
//! - [`npu`]: presence of the AMD NPU and its driver version ([`NpuStatus`]).

pub mod npu;
pub mod runtime;

pub use npu::{ComputeAccelerator, NpuMatch, NpuStatus};
pub use runtime::{CheckDepth, RuntimeFiles, RuntimeManifest};
