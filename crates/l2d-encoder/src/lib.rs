//! Encoder package format, identity, and execution on NPU or CPU.
//!
//! So far (task 2.3) the two things that are settled before the runtime is loaded:
//! - [`runtime`]: the AMD runtime files, checked against a manifest ([`RuntimeFiles`]);
//! - [`npu`]: presence of the AMD NPU and its driver version ([`NpuStatus`]).
//!
//! And (task 2.5) the step in front of the model:
//! - [`codec`]: a log text becomes the model's input, with its truncation flag
//!   ([`ByteCodec`], [`InputArch`], [`InputMode`]).

pub mod codec;
pub mod npu;
pub mod runtime;

pub use codec::{ByteCodec, ByteTable, InputArch, InputBatch, InputMode, TextInput};
pub use npu::{ComputeAccelerator, NpuMatch, NpuStatus};
pub use runtime::{CheckDepth, RecordRefresh, RecordState, RuntimeFiles, RuntimeManifest};
