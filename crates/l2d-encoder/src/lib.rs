//! Encoder package format, identity, and execution on NPU or CPU.
//!
//! So far (task 2.3) the two things that are settled before the runtime is loaded:
//! - [`runtime`]: the AMD runtime files, checked against a manifest ([`RuntimeFiles`]);
//! - [`npu`]: presence of the AMD NPU and its driver version ([`NpuStatus`]).
//!
//! And (task 2.5) the step in front of the model:
//! - [`codec`]: a log text becomes the model's input, with its truncation flag
//!   ([`ByteCodec`], [`InputArch`], [`InputMode`]).

//!
//! And (task 2.4) the encoder as it lies on disk:
//! - [`package`]: the package directory below `models/encoders/`, its revision and its
//!   verification ([`EncoderPackage`], [`EncoderArch`]);
//! - [`onnx`]: what a model file says about its own input and output
//!   ([`ModelInterface`]).
//!
//! And (task 2.12) what the measurements on the NPU needed:
//! - [`encode`]: the quantization rule of the model's ends, the normalisation after the
//!   graph and the provider's cache key, for the session of task 2.6;
//! - [`onnx_edit`]: the same model file for another batch size, with its quantized bytes
//!   as input or output, or cut at an inner tensor.

pub mod codec;
pub mod encode;
pub mod npu;
pub mod onnx;
pub mod onnx_edit;
pub mod package;
pub mod runtime;

pub use codec::{ByteCodec, ByteTable, InputArch, InputBatch, InputMode, TextInput};
pub use npu::{ComputeAccelerator, NpuMatch, NpuStatus};
pub use onnx::{ModelInterface, QuantSpec, TensorSpec};
pub use package::{
    EncoderArch, EncoderPackage, PackageFile, PackageParts, WriteOutcome, WrittenPackage,
};
pub use runtime::{CheckDepth, RecordRefresh, RecordState, RuntimeFiles, RuntimeManifest};
