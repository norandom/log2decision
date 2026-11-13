//! The encoder as it lies on disk: the package directory, its revision and its
//! verification (task 2.4; requirements 3.3, 3.7, 10.3, 10.5).
//!
//! # Layout
//!
//! A package is one directory directly below `models/encoders/` of the project, named
//! `<name>-<first 12 characters of the revision>`, and it holds exactly these files:
//!
//! | File | Content | In the revision |
//! |------|---------|-----------------|
//! | `encoder.json` | identity, architecture, input mode, the model's interface, file sizes and hashes | describes it |
//! | `model.int8.onnx` | the 8-bit model that runs on the NPU | yes |
//! | `table.bin` | the byte table ([`ByteTable`]) | yes |
//! | `training-record.json` | how the encoder was made; any JSON | no |
//! | `model.f32.onnx` | the float model the 8-bit one was quantized from; optional | no |
//!
//! The float model is kept when the maker has one, as the source of the 8-bit model. The
//! tool never runs it (the encoder runs on the NPU only), so it is optional, it is not
//! part of the revision, and nothing reads it but the hash check.
//!
//! # The revision
//!
//! The first 40 characters of the lower-case hexadecimal SHA-256 over these bytes, in
//! this order ([`EncoderPackage::revision_of`]):
//!
//! 1. the 22 bytes of [`REVISION_DOMAIN`], `l2d-encoder-package/1` and a line feed;
//! 2. `input_bytes`, `channels` and `batch` of the input architecture and the output
//!    dimension, each as a 32-bit unsigned integer, little-endian (16 bytes);
//! 3. the name of the input mode (`whole` or `payload`): its length in bytes as a 64-bit
//!    unsigned integer, little-endian, then its ASCII bytes;
//! 4. `table.bin`: its length in bytes as a 64-bit unsigned integer, little-endian, then
//!    its bytes;
//! 5. `model.int8.onnx`: its length the same way, then its bytes.
//!
//! So two packages share a revision only when the same model reads the same table over
//! the same part of a text in the same shape: they give the same vector for every text.
//! The name, the training record, the float model and the times of the files are not
//! hashed. What `encoder.json` says about the model (tensor names, shapes, element
//! types, quantization) is not hashed either: it is read from the model file, which is.
//!
//! The `/1` in the first line stands for everything in the tool's code that shapes a
//! vector and is not among the hashed bytes: the padding byte (0), the truncation rule
//! (the first `input_bytes` bytes of the part the mode selects), how a NUL byte in a text
//! is read (as padding), the layout of the input tensor
//! (`x[n, c, 0, t] = table[byte[n, t]][c]`), and the L2 normalisation of the output after
//! the graph (task 2.6). A change to any of them gives other vectors from the same
//! files, so it must raise that number, which changes every revision. If the tool ever
//! feeds quantized 8-bit input to the model itself (task 2.12), its rounding rule joins
//! this list.
//!
//! # Verification
//!
//! [`EncoderPackage::open`] checks, before it hands anything out and without writing:
//! the place (directly below the project's `models/encoders/`, no `..` step, not through
//! a link), that every file is there and nothing else is, each file's size and SHA-256
//! against `encoder.json`, the revision recomputed from the files against the stored one
//! and against the directory name, what `encoder.json` says about the model against the
//! model file, and that `encoder.json` is byte for byte what this tool writes for its
//! content. One altered byte anywhere is a refusal that names the file.
//!
//! # Writing
//!
//! [`EncoderPackage::write`] writes the files into a temporary sibling directory
//! (`.<name>-<rev12>.writing-<process>-<count>`), each file synced, `encoder.json` last,
//! and renames the directory into place. A package directory therefore appears complete
//! or not at all, and an existing package is never written into: the same package again
//! is found and returned, the same encoder with another training record is found,
//! returned and reported as such ([`WriteOutcome`]), and anything else in the place is a
//! refusal. A process that dies on the way leaves its temporary directory behind; it
//! does not open as a package, it is not removed by a later write (it could belong to a
//! process that is still writing), and a later write whose temporary name is taken by
//! such a leftover takes the next count.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use l2d_core::types::short_revision;
use l2d_core::{EncoderIdentity, L2dError, MemoryBudget, MemorySettings, ProjectPaths};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::codec::{ByteCodec, ByteTable, InputArch, InputMode};
use crate::onnx::{FLOAT32, ModelInterface, QuantSpec, TensorSpec};
use crate::runtime::sha256_of;

/// The description of a package.
pub const ENCODER_JSON: &str = "encoder.json";
/// The 8-bit model.
pub const MODEL_INT8_FILE: &str = "model.int8.onnx";
/// The byte table.
pub const TABLE_FILE: &str = "table.bin";
/// The record of how the encoder was made.
pub const TRAINING_RECORD_FILE: &str = "training-record.json";
/// The float model, when the package has one.
pub const MODEL_F32_FILE: &str = "model.f32.onnx";

/// The format number of `encoder.json` this tool writes and reads.
pub const FORMAT: u32 = 1;
/// The kind `encoder.json` states.
pub const KIND: &str = "l2d-encoder";
/// The first bytes the revision is hashed over (see the module documentation).
pub const REVISION_DOMAIN: &[u8] = b"l2d-encoder-package/1\n";
/// The length of a revision in characters.
pub const REVISION_CHARS: usize = 40;
/// The longest encoder name in bytes.
pub const MAX_NAME_BYTES: usize = 64;
/// The largest output dimension an architecture may state.
pub const MAX_DIMENSION: u16 = 4_096;
/// The largest `encoder.json` that is read (the tool writes about 1.5 KiB).
pub const MAX_ENCODER_JSON_BYTES: u64 = 1024 * 1024;

/// The buffer a file is hashed through when it is not kept ([`sha256_of`]).
const HASH_BUFFER_BYTES: u64 = 1024 * 1024;
/// Memory of an open besides the files: paths, the parsed description, and what the
/// model reader keeps, which has a ceiling of its own (under 200 KiB: at most 256 graph
/// inputs and outputs each with at most 8 dimensions; nodes and initializers are not
/// kept, see [`crate::onnx`]).
const OPEN_FIXED_BYTES: u64 = 256 * 1024;
/// How many temporary names a write tries before it gives up.
const TEMPORARY_TRIES: u32 = 64;
const OPERATION: &str = "opening the encoder package";

/// The architecture description of an encoder: the shape of the model input
/// ([`InputArch`], nested, because it refuses unknown fields) and the output dimension.
///
/// JSON: `{"input":{"input_bytes":256,"channels":64,"batch":32},"dimension":128}`.
/// A value read through serde is validated and unknown fields are refused. Layer
/// definitions are not part of it: the model file is the definition of the layers, and
/// the trainer of task 4.4 has its own description of what it builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "EncoderArchFields")]
pub struct EncoderArch {
    /// The shape of the model input, as the codec needs it.
    pub input: InputArch,
    /// Values per vector: the output dimension of the encoder identity.
    pub dimension: u16,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename = "EncoderArch")]
struct EncoderArchFields {
    input: InputArch,
    dimension: u16,
}

impl TryFrom<EncoderArchFields> for EncoderArch {
    type Error = L2dError;

    fn try_from(fields: EncoderArchFields) -> Result<Self, L2dError> {
        let arch = EncoderArch {
            input: fields.input,
            dimension: fields.dimension,
        };
        arch.validate()?;
        Ok(arch)
    }
}

impl EncoderArch {
    /// Architecture v1: 256 bytes, 64 values per byte, batches of 32, 128 dimensions.
    pub const V1: EncoderArch = EncoderArch {
        input: InputArch::V1,
        dimension: 128,
    };

    /// Refuses an input architecture that does not validate ([`InputArch::validate`])
    /// and a dimension outside 1 to [`MAX_DIMENSION`].
    pub fn validate(&self) -> Result<(), L2dError> {
        self.input.validate()?;
        if self.dimension == 0 || self.dimension > MAX_DIMENSION {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "encoder architecture: dimension is {}, outside the range 1 to \
                     {MAX_DIMENSION}",
                    self.dimension
                ),
            });
        }
        Ok(())
    }

    /// The shape the model input must have: `[batch, channels, 1, input_bytes]`.
    pub fn model_input_shape(&self) -> [i64; 4] {
        self.input.dims()
    }

    /// The shape the model output must have: `[batch, dimension]`.
    pub fn model_output_shape(&self) -> [i64; 2] {
        [i64::from(self.input.batch), i64::from(self.dimension)]
    }

    /// The reason a model does not fit this architecture, if it does not: both tensors
    /// must be 32-bit float with the shapes above.
    fn misfit(&self, model: &ModelInterface) -> Option<String> {
        let check = |role: &str, tensor: &TensorSpec, shape: &[i64]| {
            (tensor.element_type != FLOAT32 || tensor.shape != shape).then(|| {
                format!(
                    "its {role} `{}` is {} {:?}, the architecture needs {FLOAT32} {shape:?}",
                    tensor.name, tensor.element_type, tensor.shape
                )
            })
        };
        check("input", &model.input, &self.model_input_shape())
            .or_else(|| check("output", &model.output, &self.model_output_shape()))
    }
}

/// What a package is written from. The bytes are the caller's; nothing is read from disk.
#[derive(Debug, Clone, Copy)]
pub struct PackageParts<'a> {
    /// 1 to [`MAX_NAME_BYTES`] letters, digits, `.`, `_` or `-`, not starting with a dot.
    pub name: &'a str,
    pub arch: EncoderArch,
    /// The part of a text the model was fitted on. There is no default.
    pub input_mode: InputMode,
    /// The 8-bit model (see [`crate::onnx`] for the form it must have).
    pub model_int8: &'a [u8],
    /// The byte table file ([`ByteTable`]).
    pub table: &'a [u8],
    /// The training record; any JSON.
    pub training_record: &'a [u8],
    /// The float model, when there is one.
    pub model_f32: Option<&'a [u8]>,
}

/// A file of a package as `encoder.json` records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageFile {
    pub name: &'static str,
    pub bytes: u64,
    /// Lower-case hexadecimal SHA-256.
    pub sha256: String,
}

/// An encoder package that was verified when it was opened.
///
/// It holds the small things (identity, architecture, the model's interface, the list of
/// files with their hashes, the directory path) and the codec with the byte table. It
/// does not hold the model: [`EncoderPackage::model_path`] names the file and
/// [`EncoderPackage::read_model`] reads it again, verified.
#[derive(Debug, Clone)]
pub struct EncoderPackage {
    dir: PathBuf,
    identity: EncoderIdentity,
    arch: EncoderArch,
    model: ModelInterface,
    codec: ByteCodec,
    files: Vec<PackageFile>,
}

// ---------------------------------------------------------------------------------------
// encoder.json

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: u32,
    kind: String,
    identity: Identity,
    arch: EncoderArch,
    input_mode: InputMode,
    model: ModelTensors,
    quantization: Quantization,
    files: Files,
}

/// [`EncoderIdentity`] with unknown fields refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    name: String,
    revision: String,
    dimension: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelTensors {
    input: TensorSpec,
    output: TensorSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Quantization {
    input: QuantSpec,
    output: QuantSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Files {
    #[serde(rename = "model.int8.onnx")]
    model_int8: FileEntry,
    #[serde(rename = "table.bin")]
    table: FileEntry,
    #[serde(rename = "training-record.json")]
    training_record: FileEntry,
    #[serde(
        rename = "model.f32.onnx",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    model_f32: Option<FileEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileEntry {
    bytes: u64,
    sha256: String,
}

/// The first look at `encoder.json`: only what tells whether the rest can be read.
#[derive(Deserialize)]
struct ManifestHead {
    format: Option<serde_json::Value>,
    kind: Option<serde_json::Value>,
}

impl FileEntry {
    fn of(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.len() as u64,
            sha256: hex(&Sha256::digest(bytes)),
        }
    }
}

impl Files {
    /// The files in the order they are listed and written.
    fn list(&self) -> Vec<PackageFile> {
        let entry = |name: &'static str, entry: &FileEntry| PackageFile {
            name,
            bytes: entry.bytes,
            sha256: entry.sha256.clone(),
        };
        let mut list = vec![
            entry(MODEL_INT8_FILE, &self.model_int8),
            entry(TABLE_FILE, &self.table),
            entry(TRAINING_RECORD_FILE, &self.training_record),
        ];
        if let Some(float) = &self.model_f32 {
            list.push(entry(MODEL_F32_FILE, float));
        }
        list
    }
}

impl Manifest {
    /// The bytes of `encoder.json`: indented JSON in the order of the fields, one line
    /// feed at the end.
    fn to_bytes(&self) -> Result<Vec<u8>, L2dError> {
        let mut bytes = serde_json::to_vec_pretty(self).map_err(|error| L2dError::Runtime {
            detail: format!("writing {ENCODER_JSON}: {error}"),
        })?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    fn with_model(mut self, model: &ModelInterface) -> Self {
        self.model = ModelTensors {
            input: model.input.clone(),
            output: model.output.clone(),
        };
        self.quantization = Quantization {
            input: model.input_quantization.clone(),
            output: model.output_quantization.clone(),
        };
        self
    }
}

// ---------------------------------------------------------------------------------------
// Helpers

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn io_failure(doing: &str, path: &Path, error: &std::io::Error) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

fn not_a_package(dir: &Path, why: impl std::fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "{} does not hold a usable encoder package: {why}",
            dir.display()
        ),
    }
}

/// The rule of the baseline scorer and the calibration for names.
fn usable_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn is_revision(revision: &str) -> bool {
    revision.len() == REVISION_CHARS
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Requirement 10.3 for a package directory, as far as the path and the links on it
/// tell: `dir` must be a directory directly below `models/encoders/` of the project,
/// written without a `..` step, and neither it nor `models/` nor `models/encoders/` may
/// be a link (a symbolic link or a junction). Nothing has to exist.
fn require_location(paths: &ProjectPaths, dir: &Path) -> Result<(), L2dError> {
    let encoders = paths.encoders_dir();
    let outside = |why: String| L2dError::Refused {
        reason: format!(
            "encoder packages are read and written only directly below the project's model \
             directory, in {}; {} is not accepted: {why}",
            encoders.display(),
            dir.display()
        ),
    };
    if !paths.is_under_models(dir) || dir.parent() != Some(encoders.as_path()) {
        return Err(outside(
            "it is not a directory directly below that one, or its path holds a `..` step".into(),
        ));
    }
    for place in [paths.models_dir(), encoders.clone(), dir.to_path_buf()] {
        if fs::symlink_metadata(&place).is_ok_and(|found| found.file_type().is_symlink()) {
            return Err(outside(format!(
                "{} is a link, and a package is not reached through a link",
                place.display()
            )));
        }
    }
    Ok(())
}

/// The rest of requirement 10.3 for a directory that exists: resolved by the file
/// system, it is still that directory below the project's `models/encoders/`.
fn require_resolved(paths: &ProjectPaths, dir: &Path) -> Result<(), L2dError> {
    let encoders = paths.encoders_dir();
    let resolve =
        |path: &Path| fs::canonicalize(path).map_err(|error| io_failure("resolving", path, &error));
    let (real, real_encoders) = (resolve(dir)?, resolve(&encoders)?);
    let same_place = real.parent() == Some(real_encoders.as_path());
    if let (true, Some(on_disk), Some(asked)) = (same_place, real.file_name(), dir.file_name())
        && on_disk != asked
        && on_disk
            .to_string_lossy()
            .eq_ignore_ascii_case(&asked.to_string_lossy())
    {
        // A file system that does not tell upper case from lower case: the directory
        // that answers to this name is another encoder's.
        return Err(L2dError::Refused {
            reason: format!(
                "{} is the directory `{}` on this file system, which does not tell upper \
                 case from lower case in names: it belongs to the encoder of that spelling. \
                 Open it under its own name, or give the new encoder a name that differs by \
                 more than the case of its letters",
                dir.display(),
                on_disk.to_string_lossy()
            ),
        });
    }
    if !same_place || real.file_name() != dir.file_name() {
        return Err(L2dError::Refused {
            reason: format!(
                "encoder packages are read and written only directly below the project's model \
                 directory, in {}; {} is not accepted: it resolves to {}, through a link or \
                 under another spelling",
                encoders.display(),
                dir.display(),
                real.display()
            ),
        });
    }
    Ok(())
}

/// The size of a file of a package; a file that is not there is [`L2dError::Missing`].
fn file_metadata(dir: &Path, file: &str) -> Result<fs::Metadata, L2dError> {
    let path = dir.join(file);
    fs::symlink_metadata(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            L2dError::Missing {
                item: format!("{file} of the encoder package"),
                expected_at: path.clone(),
            }
        } else {
            io_failure("reading", &path, &error)
        }
    })
}

fn read_file(dir: &Path, file: &str) -> Result<Vec<u8>, L2dError> {
    let path = dir.join(file);
    fs::read(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            L2dError::Missing {
                item: format!("{file} of the encoder package"),
                expected_at: path.clone(),
            }
        } else {
            io_failure("reading", &path, &error)
        }
    })
}

/// The byte table of a package, with the file named in a refusal.
fn table_of(bytes: &[u8], arch: &EncoderArch) -> Result<ByteTable, L2dError> {
    ByteTable::from_le_bytes(bytes, &arch.input).map_err(|error| match error {
        L2dError::Refused { reason } => L2dError::Refused {
            reason: format!("{TABLE_FILE}: {reason}"),
        },
        other => other,
    })
}

/// The interface of the 8-bit model, which must fit the architecture.
fn interface_of(bytes: &[u8], arch: &EncoderArch) -> Result<ModelInterface, String> {
    let model = ModelInterface::read(bytes).map_err(|reason| {
        format!("{MODEL_INT8_FILE} is not usable as an 8-bit encoder model: {reason}")
    })?;
    match arch.misfit(&model) {
        Some(misfit) => Err(format!(
            "{MODEL_INT8_FILE} does not fit the architecture: {misfit}"
        )),
        None => Ok(model),
    }
}

/// Makes temporary directory names of one process distinct.
static WRITES: AtomicU64 = AtomicU64::new(0);

/// Makes the temporary sibling directory of a write,
/// `.<stem>.writing-<process id>-<count>`. A name that is taken (the leftover of an
/// interrupted write of a process that had this process id) is passed over for the next
/// count, [`TEMPORARY_TRIES`] times at most. The baseline scorer and the calibration
/// name their temporary directories by the same scheme, without the retry.
fn create_temporary(encoders: &Path, stem: &str, counter: &AtomicU64) -> Result<PathBuf, L2dError> {
    let mut taken = Vec::new();
    for _ in 0..TEMPORARY_TRIES {
        let temporary = encoders.join(format!(
            ".{stem}.writing-{}-{}",
            std::process::id(),
            counter.fetch_add(1, Ordering::Relaxed)
        ));
        match fs::create_dir(&temporary) {
            Ok(()) => return Ok(temporary),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                taken.push(temporary);
            }
            Err(error) => return Err(io_failure("creating", &temporary, &error)),
        }
    }
    Err(L2dError::Runtime {
        detail: format!(
            "no temporary directory could be made in {}: the {TEMPORARY_TRIES} names from {} \
             to {} are taken, by what interrupted writes left behind; remove the \
             `.{stem}.writing-*` directories there when no other write is running",
            encoders.display(),
            taken
                .first()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            taken
                .last()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
        ),
    })
}

/// What a write found and did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteOutcome {
    /// Nothing was in the place; the package was written.
    Written,
    /// The very same package was there already; nothing was written.
    Unchanged,
    /// The same encoder (same name and revision, so the same model, table, architecture
    /// and input mode) was there already with other content in the files the revision
    /// does not cover. Nothing was written or overwritten: the package that was there is
    /// the one returned, and the new training record (or float model) was not stored.
    ExistsWithOtherRecord {
        /// The files that differ: `training-record.json`, `model.f32.onnx` or both.
        differing: Vec<&'static str>,
    },
}

/// The result of [`EncoderPackage::write`]: the package that is in the place now, and
/// how it got there.
#[derive(Debug, Clone)]
pub struct WrittenPackage {
    pub package: EncoderPackage,
    pub outcome: WriteOutcome,
}

/// Where a write stands; told to the observer of [`EncoderPackage::write_observed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Progress {
    /// The temporary directory exists and is empty.
    Started,
    /// This file is written and synced.
    File(&'static str),
    /// Every file is written; the rename comes next.
    Complete,
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), L2dError> {
    let failed = |error: std::io::Error| io_failure("writing", path, &error);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(failed)?;
    file.write_all(bytes).map_err(failed)?;
    file.sync_all().map_err(failed)
}

/// Syncs a directory, so that the names in it are on disk. Windows has no such call for
/// a directory opened this way; there the files are synced and the rename is left to
/// the file system's journal.
fn sync_dir(dir: &Path) -> Result<(), L2dError> {
    #[cfg(unix)]
    fs::File::open(dir)
        .and_then(|opened| opened.sync_all())
        .map_err(|error| io_failure("syncing", dir, &error))?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Which memory check an open runs.
enum Budget<'b> {
    /// The budget is observed now, with the default settings.
    Observe,
    Given(&'b MemoryBudget),
    /// No check: the caller holds the same bytes already (a write).
    Unchecked,
}

impl EncoderPackage {
    /// The revision of an encoder: see the module documentation for the bytes and their
    /// order. The function hashes what it is given; that the architecture fits the model
    /// and the table is checked when a package is written or opened.
    pub fn revision_of(
        arch: &EncoderArch,
        input_mode: InputMode,
        table: &[u8],
        model_int8: &[u8],
    ) -> String {
        let mode = input_mode.as_str().as_bytes();
        let mut hash = Sha256::new();
        hash.update(REVISION_DOMAIN);
        hash.update(arch.input.input_bytes.to_le_bytes());
        hash.update(arch.input.channels.to_le_bytes());
        hash.update(arch.input.batch.to_le_bytes());
        hash.update(u32::from(arch.dimension).to_le_bytes());
        hash.update((mode.len() as u64).to_le_bytes());
        hash.update(mode);
        hash.update((table.len() as u64).to_le_bytes());
        hash.update(table);
        hash.update((model_int8.len() as u64).to_le_bytes());
        hash.update(model_int8);
        let mut revision = hex(&hash.finalize());
        revision.truncate(REVISION_CHARS);
        revision
    }

    /// Writes a package as `models/encoders/<name>-<rev12>/` and returns it, opened and
    /// verified from the files as they lie on disk.
    ///
    /// Before anything is written: a name outside the rule, an architecture that does
    /// not validate or a training record that is not JSON is
    /// [`L2dError::InvalidInput`]; a table that is not the architecture's, a model that
    /// is not an 8-bit encoder model of the architecture's shapes, or a place that is
    /// not the project's own `models/encoders/` is [`L2dError::Refused`].
    ///
    /// A float model, when given, must have the input and output of the 8-bit model
    /// (names, 32-bit float, shapes), or the write is refused.
    ///
    /// The directory appears complete or not at all (see the module documentation). What
    /// happened is in the result ([`WriteOutcome`]), with the package that is now in the
    /// place, opened and verified:
    /// - nothing was there: the package was written ([`WriteOutcome::Written`]);
    /// - this very package was there: nothing was written ([`WriteOutcome::Unchanged`]);
    /// - the same encoder (same revision and name) was there with another training
    ///   record or float model: nothing was written or overwritten, the existing package
    ///   is returned, and the caller should say that the new record was not stored
    ///   ([`WriteOutcome::ExistsWithOtherRecord`]). A training run does not fail on it:
    ///   the encoder it produced is the one that is there.
    ///
    /// Anything else in the place (a damaged package, something that is no package) is
    /// [`L2dError::Refused`] and left unchanged.
    ///
    /// Memory: the caller holds the parts; the write adds the parsed table and, for the
    /// final verification, the model and the table once more.
    pub fn write(
        paths: &ProjectPaths,
        parts: &PackageParts<'_>,
    ) -> Result<WrittenPackage, L2dError> {
        Self::write_observed(paths, parts, &mut |_| Ok(()))
    }

    /// [`EncoderPackage::write`], telling `observer` where it stands. An error from the
    /// observer ends the write as a failed file operation would.
    fn write_observed(
        paths: &ProjectPaths,
        parts: &PackageParts<'_>,
        observer: &mut dyn FnMut(Progress) -> Result<(), L2dError>,
    ) -> Result<WrittenPackage, L2dError> {
        let name = parts.name;
        if !usable_name(name) {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "the encoder name must be 1 to {MAX_NAME_BYTES} letters, digits, '.', '_' \
                     or '-' and must not start with a dot; got a name of {} bytes that is not",
                    name.len()
                ),
            });
        }
        parts.arch.validate()?;
        serde_json::from_slice::<serde::de::IgnoredAny>(parts.training_record).map_err(
            |error| L2dError::InvalidInput {
                detail: format!("{TRAINING_RECORD_FILE} must be JSON: {error}"),
            },
        )?;
        table_of(parts.table, &parts.arch)?;
        let model = interface_of(parts.model_int8, &parts.arch)
            .map_err(|reason| L2dError::Refused { reason })?;
        if let Some(float) = parts.model_f32 {
            // The float model is the source of the 8-bit one: the same input and output.
            let (input, output) =
                crate::onnx::float_tensors(float).map_err(|reason| L2dError::Refused {
                    reason: format!(
                        "{MODEL_F32_FILE} is not usable as the float model of the encoder: \
                         {reason}"
                    ),
                })?;
            if input != model.input || output != model.output {
                return Err(L2dError::Refused {
                    reason: format!(
                        "{MODEL_F32_FILE} is not the float model of {MODEL_INT8_FILE}: its \
                         input is {input} and its output {output}, the 8-bit model has {} and \
                         {}",
                        model.input, model.output
                    ),
                });
            }
        }

        let revision =
            Self::revision_of(&parts.arch, parts.input_mode, parts.table, parts.model_int8);
        let dir = paths.encoder_dir(name, &revision);
        require_location(paths, &dir)?;

        let manifest = Manifest {
            format: FORMAT,
            kind: KIND.into(),
            identity: Identity {
                name: name.into(),
                revision: revision.clone(),
                dimension: parts.arch.dimension,
            },
            arch: parts.arch,
            input_mode: parts.input_mode,
            model: ModelTensors {
                input: model.input.clone(),
                output: model.output.clone(),
            },
            quantization: Quantization {
                input: model.input_quantization.clone(),
                output: model.output_quantization.clone(),
            },
            files: Files {
                model_int8: FileEntry::of(parts.model_int8),
                table: FileEntry::of(parts.table),
                training_record: FileEntry::of(parts.training_record),
                model_f32: parts.model_f32.map(FileEntry::of),
            },
        };
        if fs::symlink_metadata(&dir).is_ok() {
            return Self::existing(paths, &dir, &manifest);
        }

        let encoders = paths.encoders_dir();
        fs::create_dir_all(&encoders).map_err(|error| io_failure("creating", &encoders, &error))?;
        // Again, now that the directories exist: none of them may have become a link.
        require_location(paths, &dir)?;
        let temporary = create_temporary(
            &encoders,
            &format!("{name}-{}", short_revision(&revision)),
            &WRITES,
        )?;

        let mut files: Vec<(&'static str, &[u8])> = vec![
            (MODEL_INT8_FILE, parts.model_int8),
            (TABLE_FILE, parts.table),
            (TRAINING_RECORD_FILE, parts.training_record),
        ];
        if let Some(float) = parts.model_f32 {
            files.push((MODEL_F32_FILE, float));
        }
        let manifest_bytes = manifest.to_bytes()?;
        // The description goes last: a directory without it is plainly unfinished.
        files.push((ENCODER_JSON, &manifest_bytes));

        let mut fill = || -> Result<(), L2dError> {
            observer(Progress::Started)?;
            for &(file, bytes) in &files {
                write_new_file(&temporary.join(file), bytes)?;
                observer(Progress::File(file))?;
            }
            sync_dir(&temporary)?;
            observer(Progress::Complete)
        };
        if let Err(error) = fill() {
            // Best effort: the failure itself is what the caller has to see.
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
        if let Err(error) = fs::rename(&temporary, &dir) {
            let _ = fs::remove_dir_all(&temporary);
            // Another writer may have completed the same package in the meantime.
            if fs::symlink_metadata(&dir).is_ok() {
                return Self::existing(paths, &dir, &manifest);
            }
            return Err(io_failure("completing", &dir, &error));
        }
        sync_dir(&encoders)?;
        Ok(WrittenPackage {
            package: Self::open_checked(paths, &dir, Budget::Unchecked)?,
            outcome: WriteOutcome::Written,
        })
    }

    /// What is already in the place of the package to be written: the same package
    /// ([`WriteOutcome::Unchanged`]), the same encoder with another training record or
    /// float model ([`WriteOutcome::ExistsWithOtherRecord`]), or something that is a
    /// refusal. Nothing is written.
    fn existing(
        paths: &ProjectPaths,
        dir: &Path,
        wanted: &Manifest,
    ) -> Result<WrittenPackage, L2dError> {
        let found = match Self::open_checked(paths, dir, Budget::Unchecked) {
            Ok(found) => found,
            Err(error) => {
                return Err(L2dError::Refused {
                    reason: format!(
                        "{} already exists and does not hold revision {} of the encoder `{}` \
                         ({error}); it was left unchanged",
                        dir.display(),
                        wanted.identity.revision,
                        wanted.identity.name
                    ),
                });
            }
        };
        // The directory name holds the first twelve characters of the revision only: two
        // encoders can meet in one name (the same files read in another mode or shape
        // would even have the same file list). The whole revision decides.
        if found.identity.revision != wanted.identity.revision
            || found.identity.name != wanted.identity.name
        {
            return Err(L2dError::Refused {
                reason: format!(
                    "{} already holds the encoder `{}` of revision {}, not revision {} of the \
                     encoder `{}` that was to be written; the two share the first twelve \
                     characters. It was left unchanged; give the new encoder another name",
                    dir.display(),
                    found.identity.name,
                    found.identity.revision,
                    wanted.identity.revision,
                    wanted.identity.name
                ),
            });
        }
        let wanted_files = wanted.files.list();
        if found.files == wanted_files {
            return Ok(WrittenPackage {
                package: found,
                outcome: WriteOutcome::Unchanged,
            });
        }
        let mut differing: Vec<&'static str> = Vec::new();
        for name in [
            MODEL_INT8_FILE,
            TABLE_FILE,
            TRAINING_RECORD_FILE,
            MODEL_F32_FILE,
        ] {
            let of = |files: &[PackageFile]| files.iter().find(|file| file.name == name).cloned();
            if of(&found.files) != of(&wanted_files) {
                differing.push(name);
            }
        }
        // The same encoder: the revision covers the model, the table, the architecture and
        // the input mode. Only what the revision does not cover can differ.
        if differing
            .iter()
            .all(|name| [TRAINING_RECORD_FILE, MODEL_F32_FILE].contains(name))
        {
            return Ok(WrittenPackage {
                package: found,
                outcome: WriteOutcome::ExistsWithOtherRecord { differing },
            });
        }
        Err(L2dError::Refused {
            reason: format!(
                "{} already holds revision {} of the encoder `{}` with other content in {}, \
                 which the revision covers; it was left unchanged",
                dir.display(),
                wanted.identity.revision,
                wanted.identity.name,
                differing.join(", ")
            ),
        })
    }

    /// Opens and verifies a package directory (see the module documentation for what is
    /// checked). Nothing is written.
    ///
    /// - A place that is not a directory directly below the project's `models/encoders/`
    ///   is [`L2dError::Refused`], whether or not anything is there.
    /// - A package or a file of it that is not there is [`L2dError::Missing`], naming it
    ///   and where it was expected.
    /// - Anything that does not verify is [`L2dError::Refused`], naming the file.
    ///
    /// The memory an open needs ([`EncoderPackage::open_bytes`]) is checked against a
    /// budget observed with the default settings; [`EncoderPackage::open_with`] takes
    /// the caller's.
    pub fn open(paths: &ProjectPaths, dir: &Path) -> Result<Self, L2dError> {
        Self::open_checked(paths, dir, Budget::Observe)
    }

    /// [`EncoderPackage::open`] under the caller's memory budget: when the estimate of
    /// [`EncoderPackage::open_bytes`] is more than the budget leaves, the package is
    /// [`L2dError::Refused`] before a file is read.
    pub fn open_with(
        paths: &ProjectPaths,
        dir: &Path,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        Self::open_checked(paths, dir, Budget::Given(budget))
    }

    /// The memory an open of this directory needs at its peak, from the sizes of the
    /// files: the 8-bit model once (it is hashed and read for its interface in memory,
    /// then dropped), the table twice (its bytes and the parsed table), `encoder.json`
    /// three times, the 1 MiB buffer the other files are hashed through, and 256 KiB for
    /// what does not depend on the file sizes: the paths, the parsed description and what
    /// the model reader keeps, which is bounded whatever the model holds (at most 256
    /// graph inputs and outputs each, see [`crate::onnx`]; a model file of very many
    /// nodes costs no more). For architecture v1 and the science gate's model that is
    /// about 2.3 MiB. What stays after the open is [`EncoderPackage::held_bytes`].
    pub fn open_bytes(dir: &Path) -> Result<u64, L2dError> {
        let size = |file: &str| file_metadata(dir, file).map(|found| found.len());
        let (json, model, table) = (
            size(ENCODER_JSON)?,
            size(MODEL_INT8_FILE)?,
            size(TABLE_FILE)?,
        );
        Ok(model
            .saturating_add(table.saturating_mul(2))
            .saturating_add(json.saturating_mul(3))
            .saturating_add(HASH_BUFFER_BYTES + OPEN_FIXED_BYTES))
    }

    fn open_checked(
        paths: &ProjectPaths,
        dir: &Path,
        budget: Budget<'_>,
    ) -> Result<Self, L2dError> {
        require_location(paths, dir)?;
        if !dir.is_dir() {
            return Err(L2dError::Missing {
                item: "encoder package".into(),
                expected_at: dir.to_path_buf(),
            });
        }
        require_resolved(paths, dir)?;
        match budget {
            Budget::Observe => MemoryBudget::observe(&MemorySettings::default())?
                .require_whole(Self::open_bytes(dir)?, OPERATION)?,
            Budget::Given(budget) => budget.require_whole(Self::open_bytes(dir)?, OPERATION)?,
            Budget::Unchecked => {}
        }
        let refused = |why: String| not_a_package(dir, why);

        // The description: a plain file of a size worth reading.
        let json_file = file_metadata(dir, ENCODER_JSON)?;
        if !json_file.file_type().is_file() {
            return Err(refused(format!("{ENCODER_JSON} is not a plain file")));
        }
        let json_size = json_file.len();
        if json_size > MAX_ENCODER_JSON_BYTES {
            return Err(refused(format!(
                "{ENCODER_JSON} has {json_size} bytes, more than the {MAX_ENCODER_JSON_BYTES} \
                 that are read"
            )));
        }
        let manifest_bytes = read_file(dir, ENCODER_JSON)?;
        let head: ManifestHead = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| refused(format!("{ENCODER_JSON} is not readable: {error}")))?;
        let stated = |value: &Option<serde_json::Value>| match value {
            Some(serde_json::Value::String(text)) => format!("'{text}'"),
            Some(other) => other.to_string(),
            None => "none".into(),
        };
        if head.format != Some(FORMAT.into()) || head.kind != Some(KIND.into()) {
            return Err(refused(format!(
                "{ENCODER_JSON} has format {} and kind {}, this tool reads format {FORMAT} and \
                 kind '{KIND}'",
                stated(&head.format),
                stated(&head.kind)
            )));
        }
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| refused(format!("{ENCODER_JSON} is not readable: {error}")))?;

        // The identity and the directory name.
        let Identity {
            name,
            revision,
            dimension,
        } = &manifest.identity;
        if !usable_name(name) {
            return Err(refused(format!(
                "{ENCODER_JSON} names the encoder {name:?}, which is not a usable name"
            )));
        }
        if !is_revision(revision) {
            return Err(refused(format!(
                "{ENCODER_JSON} records the revision {revision:?}, which is not \
                 {REVISION_CHARS} lower-case hexadecimal characters"
            )));
        }
        let expected_name = format!("{name}-{}", short_revision(revision));
        let found_name = dir
            .file_name()
            .map(|found| found.to_string_lossy().into_owned())
            .unwrap_or_default();
        if found_name != expected_name {
            return Err(refused(format!(
                "the directory name `{found_name}` is not `{expected_name}`, the name and the \
                 first twelve characters of the revision that {ENCODER_JSON} records"
            )));
        }
        if *dimension != manifest.arch.dimension {
            return Err(refused(format!(
                "{ENCODER_JSON}: the identity has the dimension {dimension}, the architecture \
                 has the dimension {}",
                manifest.arch.dimension
            )));
        }

        // Exactly the files of the description, each a plain file of the recorded size.
        let files = manifest.files.list();
        let entries = fs::read_dir(dir).map_err(|error| io_failure("listing", dir, &error))?;
        for entry in entries {
            let entry = entry.map_err(|error| io_failure("listing", dir, &error))?;
            let entry_name = entry.file_name();
            let known =
                entry_name == ENCODER_JSON || files.iter().any(|file| entry_name == file.name);
            if !known {
                return Err(refused(format!(
                    "it holds `{}`, which {ENCODER_JSON} does not list; a package holds its \
                     own files only",
                    entry_name.to_string_lossy()
                )));
            }
        }
        for file in &files {
            let found = file_metadata(dir, file.name)?;
            if !found.file_type().is_file() {
                return Err(refused(format!("{} is not a plain file", file.name)));
            }
            if found.len() != file.bytes {
                return Err(refused(format!(
                    "{} has {} bytes, {ENCODER_JSON} records {}",
                    file.name,
                    found.len(),
                    file.bytes
                )));
            }
        }
        if !file_metadata(dir, ENCODER_JSON)?.file_type().is_file() {
            return Err(refused(format!("{ENCODER_JSON} is not a plain file")));
        }

        // The content of every file against its recorded hash. The table and the model
        // are read once and used from the bytes that were hashed.
        let altered = |file: &PackageFile, found: &str| {
            refused(format!(
                "{}: its SHA-256 is {found}, {ENCODER_JSON} records {}; the file was altered",
                file.name, file.sha256
            ))
        };
        let mut table_bytes = Vec::new();
        let mut model_bytes = Vec::new();
        for file in &files {
            let found = match file.name {
                TABLE_FILE => {
                    table_bytes = read_file(dir, file.name)?;
                    hex(&Sha256::digest(&table_bytes))
                }
                MODEL_INT8_FILE => {
                    model_bytes = read_file(dir, file.name)?;
                    hex(&Sha256::digest(&model_bytes))
                }
                _ => sha256_of(&dir.join(file.name))?,
            };
            if found != file.sha256 {
                return Err(altered(file, &found));
            }
        }

        // The revision, from the files and the description.
        let recomputed = Self::revision_of(
            &manifest.arch,
            manifest.input_mode,
            &table_bytes,
            &model_bytes,
        );
        if recomputed != *revision {
            return Err(refused(format!(
                "{ENCODER_JSON} records the revision {revision}, but the model, the table, the \
                 architecture and the input mode `{}` give {recomputed}; the description was \
                 altered or does not belong to these files",
                manifest.input_mode.as_str()
            )));
        }

        // What the description says about the model, against the model.
        let model = interface_of(&model_bytes, &manifest.arch).map_err(refused)?;
        drop(model_bytes);
        let expected = manifest.clone().with_model(&model);
        if expected.to_bytes()? != manifest_bytes {
            // The first statement about the model that is not the model's own.
            let stated_and_read = [
                (
                    "the input",
                    manifest.model.input.to_string(),
                    expected.model.input.to_string(),
                ),
                (
                    "the output",
                    manifest.model.output.to_string(),
                    expected.model.output.to_string(),
                ),
                (
                    "the quantized input",
                    manifest.quantization.input.to_string(),
                    expected.quantization.input.to_string(),
                ),
                (
                    "the quantized output",
                    manifest.quantization.output.to_string(),
                    expected.quantization.output.to_string(),
                ),
            ];
            if expected != manifest {
                let (what, stated, read) = stated_and_read
                    .iter()
                    .find(|(_, stated, read)| stated != read)
                    .unwrap_or(&stated_and_read[0]);
                return Err(refused(format!(
                    "{ENCODER_JSON} states {what} as {stated}, but in {MODEL_INT8_FILE} it is \
                     {read}; the description was altered or does not belong to this model"
                )));
            }
            return Err(refused(format!(
                "{ENCODER_JSON} is not byte for byte the file this tool writes for its \
                 content; it was edited or written by something else"
            )));
        }

        let table = table_of(&table_bytes, &manifest.arch).map_err(|error| match error {
            L2dError::Refused { reason } => refused(reason),
            other => other,
        })?;
        drop(table_bytes);
        let codec = ByteCodec::new(manifest.arch.input, table, manifest.input_mode)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            identity: EncoderIdentity {
                name: manifest.identity.name,
                revision: manifest.identity.revision,
                dimension: manifest.identity.dimension,
            },
            arch: manifest.arch,
            model,
            codec,
            files,
        })
    }

    /// Name, revision and output dimension.
    pub fn identity(&self) -> &EncoderIdentity {
        &self.identity
    }

    /// The package directory, as it was given to [`EncoderPackage::open`].
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn arch(&self) -> &EncoderArch {
        &self.arch
    }

    /// The part of a text the encoder reads.
    pub fn input_mode(&self) -> InputMode {
        self.codec.mode()
    }

    /// The model's input and output and their quantization, as read from the model file.
    pub fn model(&self) -> &ModelInterface {
        &self.model
    }

    /// The codec for this encoder: its architecture, its table and its input mode.
    pub fn codec(&self) -> &ByteCodec {
        &self.codec
    }

    /// The codec, when the package itself is no longer needed.
    pub fn into_codec(self) -> ByteCodec {
        self.codec
    }

    /// The files of the package besides `encoder.json`, with sizes and hashes, in the
    /// order model, table, training record, float model.
    pub fn files(&self) -> &[PackageFile] {
        &self.files
    }

    /// The path of the 8-bit model. The file was verified when the package was opened;
    /// [`EncoderPackage::read_model`] verifies it again at the time of reading.
    pub fn model_path(&self) -> PathBuf {
        self.dir.join(MODEL_INT8_FILE)
    }

    /// The bytes of the 8-bit model, read now and handed out only when their size and
    /// SHA-256 are those the package was opened with. The caller holds them (989,261
    /// bytes for the science gate's model); a session can be created from them in
    /// memory, so that what runs is what was verified.
    pub fn read_model(&self) -> Result<Vec<u8>, L2dError> {
        let recorded = &self.files[0];
        let bytes = read_file(&self.dir, MODEL_INT8_FILE)?;
        let found = hex(&Sha256::digest(&bytes));
        if bytes.len() as u64 != recorded.bytes || found != recorded.sha256 {
            return Err(not_a_package(
                &self.dir,
                format!(
                    "{MODEL_INT8_FILE} changed after the package was opened: it has {} bytes \
                     and the SHA-256 {found}, the package was opened with {} bytes and {}",
                    bytes.len(),
                    recorded.bytes,
                    recorded.sha256
                ),
            ));
        }
        Ok(bytes)
    }

    /// The package's part of the key for the provider's compilation cache: the revision
    /// (40 characters). It is not the whole key. The provider compiles a model for one
    /// release of the runtime and under its options, so the session (task 2.6) adds the
    /// runtime release and a digest of the provider options that change compilation.
    pub fn cache_key(&self) -> &str {
        &self.identity.revision
    }

    /// The heap bytes an open package holds for as long as it lives: the codec's table
    /// ([`ByteCodec::held_bytes`], 65,536 bytes for architecture v1) and the names,
    /// shapes, hashes and the path (under 1 KiB). It does not grow with the model.
    pub fn held_bytes(&self) -> u64 {
        let tensor = |tensor: &TensorSpec| {
            tensor.name.len() + tensor.element_type.len() + tensor.shape.len() * 8
        };
        let quant = |quant: &QuantSpec| quant.tensor.len() + quant.element_type.len();
        let small = self.dir.as_os_str().len()
            + self.identity.name.len()
            + self.identity.revision.len()
            + tensor(&self.model.input)
            + tensor(&self.model.output)
            + quant(&self.model.input_quantization)
            + quant(&self.model.output_quantization)
            + self
                .files
                .iter()
                .map(|file| std::mem::size_of::<PackageFile>() + file.sha256.len())
                .sum::<usize>();
        self.codec.held_bytes() + small as u64
    }
}

#[cfg(test)]
mod tests {
    //! Interrupted writes (requirement 3.7). These tests need the observer of a write,
    //! which is not public; everything else is tested in `tests/encoder_package.rs`.

    use super::*;

    struct Project {
        root: PathBuf,
    }

    impl Project {
        fn new(tag: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "l2d-encoder-package-unit-{tag}-{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).unwrap();
            Self { root }
        }
    }

    impl Drop for Project {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn fixture(name: &str) -> Vec<u8> {
        fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(name),
        )
        .unwrap()
    }

    /// Everything below `dir`: the names, and the sizes and modification times of the
    /// files (a directory's time changes when a temporary sibling comes and goes).
    fn snapshot(dir: &Path) -> Vec<(PathBuf, u64, std::time::SystemTime)> {
        let mut found = Vec::new();
        let mut pending = vec![dir.to_path_buf()];
        while let Some(next) = pending.pop() {
            for entry in fs::read_dir(&next).unwrap() {
                let entry = entry.unwrap();
                let metadata = entry.metadata().unwrap();
                if metadata.is_dir() {
                    pending.push(entry.path());
                    found.push((entry.path(), 0, std::time::UNIX_EPOCH));
                } else {
                    found.push((entry.path(), metadata.len(), metadata.modified().unwrap()));
                }
            }
        }
        found.sort();
        found
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    const STEPS: [Progress; 6] = [
        Progress::Started,
        Progress::File(MODEL_INT8_FILE),
        Progress::File(TABLE_FILE),
        Progress::File(TRAINING_RECORD_FILE),
        Progress::File(ENCODER_JSON),
        Progress::Complete,
    ];

    #[test]
    fn a_write_that_fails_part_way_leaves_no_half_package_and_no_existing_one_disturbed() {
        let project = Project::new("fails");
        let paths = ProjectPaths::new(&project.root);
        let (model, table) = (
            fixture("spike-enc-b32-int8.onnx"),
            fixture("science-2.1.table.bin"),
        );
        let parts = |name: &'static str, input_mode| PackageParts {
            name,
            arch: EncoderArch::V1,
            input_mode,
            model_int8: &model,
            table: &table,
            training_record: b"{}",
            model_f32: None,
        };

        // Into an empty project: after a failure at any step nothing is left.
        for step in STEPS {
            let mut seen = Vec::new();
            let error = EncoderPackage::write_observed(
                &paths,
                &parts("first", InputMode::Whole),
                &mut |progress| {
                    seen.push(progress);
                    if progress == step {
                        Err(L2dError::Runtime {
                            detail: format!("the disk is full at {step:?}"),
                        })
                    } else {
                        Ok(())
                    }
                },
            )
            .unwrap_err();
            assert!(error.to_string().contains("the disk is full"), "{error}");
            assert_eq!(seen.last(), Some(&step));
            assert_eq!(entries(&paths.encoders_dir()), Vec::<String>::new());
        }
        // The observer saw every step in order on the way to the last one.
        let mut seen = Vec::new();
        let first = EncoderPackage::write_observed(
            &paths,
            &parts("first", InputMode::Whole),
            &mut |progress| {
                seen.push(progress);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(first.outcome, WriteOutcome::Written);
        let first = first.package;
        assert_eq!(seen, STEPS);
        let completed = snapshot(&paths.models_dir());

        // Beside a completed package: a failed write of another package, and of the same
        // name under another revision, leaves the completed one exactly as it was.
        for name in ["second", "first"] {
            for step in STEPS {
                EncoderPackage::write_observed(
                    &paths,
                    &parts(name, InputMode::Payload),
                    &mut |progress| {
                        if progress == step {
                            Err(L2dError::Runtime {
                                detail: "interrupted".into(),
                            })
                        } else {
                            Ok(())
                        }
                    },
                )
                .unwrap_err();
                assert_eq!(snapshot(&paths.models_dir()), completed, "{name} {step:?}");
            }
        }
        let reopened = EncoderPackage::open(&paths, first.dir()).unwrap();
        assert_eq!(reopened.identity(), first.identity());
    }

    #[test]
    fn a_process_that_dies_while_writing_leaves_only_its_temporary_directory() {
        let project = Project::new("dies");
        let paths = ProjectPaths::new(&project.root);
        let (model, table) = (
            fixture("spike-enc-b32-int8.onnx"),
            fixture("science-2.1.table.bin"),
        );
        let parts = |name: &'static str, input_mode| PackageParts {
            name,
            arch: EncoderArch::V1,
            input_mode,
            model_int8: &model,
            table: &table,
            training_record: b"{}",
            model_f32: None,
        };
        let first = EncoderPackage::write(&paths, &parts("first", InputMode::Whole))
            .unwrap()
            .package;
        let completed = snapshot(first.dir());

        // A panic in the observer unwinds through the write as a kill would stop it: the
        // clean-up of a failed write does not run.
        let mut leftovers = Vec::new();
        for (at, step) in STEPS.into_iter().enumerate() {
            let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                EncoderPackage::write_observed(
                    &paths,
                    &parts("second", InputMode::Payload),
                    &mut |progress| {
                        if progress == step {
                            panic!("killed at {step:?} (a simulated kill, expected)");
                        }
                        Ok(())
                    },
                )
            }));
            assert!(died.is_err());

            // Exactly one more entry: a temporary directory with the files so far.
            let now = entries(&paths.encoders_dir());
            let new: Vec<&String> = now
                .iter()
                .filter(|name| !leftovers.contains(*name) && **name != entries_name(&first))
                .collect();
            assert_eq!(new.len(), 1, "{now:?}");
            let leftover = new[0].clone();
            assert!(leftover.starts_with(".second-"), "{leftover}");
            assert!(leftover.contains(".writing-"), "{leftover}");
            let expected_files = match at {
                0 => 0,
                1..=4 => at,
                _ => 4,
            };
            assert_eq!(
                entries(&paths.encoders_dir().join(&leftover)).len(),
                expected_files,
                "{step:?}"
            );
            // The leftover does not open as a package, complete or not.
            let opened = EncoderPackage::open(&paths, &paths.encoders_dir().join(&leftover));
            assert!(opened.is_err(), "{step:?}");
            leftovers.push(leftover);

            // No package directory of the second encoder exists, and the first is as it
            // was.
            let revision =
                EncoderPackage::revision_of(&EncoderArch::V1, InputMode::Payload, &table, &model);
            let second_dir = paths.encoder_dir("second", &revision);
            assert!(matches!(
                EncoderPackage::open(&paths, &second_dir),
                Err(L2dError::Missing { .. })
            ));
            assert_eq!(snapshot(first.dir()), completed);
        }

        // The next run writes the package beside the leftovers and leaves them alone.
        let second = EncoderPackage::write(&paths, &parts("second", InputMode::Payload)).unwrap();
        assert_eq!(second.outcome, WriteOutcome::Written);
        assert_eq!(second.package.identity().name, "second");
        assert_eq!(entries(&paths.encoders_dir()).len(), 2 + leftovers.len());
        assert_eq!(snapshot(first.dir()), completed);
        EncoderPackage::open(&paths, first.dir()).unwrap();
    }

    #[test]
    fn a_temporary_name_taken_by_a_leftover_is_passed_over() {
        let project = Project::new("temporary");
        let encoders = project.root.join("models").join("encoders");
        fs::create_dir_all(&encoders).unwrap();
        let name = |count: u64| {
            encoders.join(format!(
                ".enc-0123456789ab.writing-{}-{count}",
                std::process::id()
            ))
        };
        // A process that had this process id died at counts 0, 1 and 2; one leftover is
        // a file, which is in the way just the same.
        fs::create_dir(name(0)).unwrap();
        fs::create_dir(name(1)).unwrap();
        fs::write(name(2), b"").unwrap();
        let counter = AtomicU64::new(0);
        let made = create_temporary(&encoders, "enc-0123456789ab", &counter).unwrap();
        assert_eq!(made, name(3));
        assert!(made.is_dir());
        assert_eq!(counter.load(Ordering::Relaxed), 4);
        // The leftovers were not touched.
        assert!(name(0).is_dir() && name(1).is_dir() && name(2).is_file());

        // Bounded: with every name of the next tries taken, the write ends with what to
        // remove, and makes nothing.
        for count in 4..4 + u64::from(TEMPORARY_TRIES) {
            fs::create_dir(name(count)).unwrap();
        }
        let before = entries(&encoders).len();
        match create_temporary(&encoders, "enc-0123456789ab", &counter) {
            Err(L2dError::Runtime { detail }) => {
                assert!(detail.contains(".enc-0123456789ab.writing-*"), "{detail}");
                assert!(detail.contains(&name(4).display().to_string()), "{detail}");
            }
            other => panic!("expected a failure, got {other:?}"),
        }
        assert_eq!(entries(&encoders).len(), before);
        // The count went on, so the next write is past the leftovers.
        let made = create_temporary(&encoders, "enc-0123456789ab", &counter).unwrap();
        assert_eq!(made, name(4 + u64::from(TEMPORARY_TRIES)));
    }

    /// The place holds a verified package whose whole revision is another than the one
    /// to be written (they share the first twelve characters): not `Unchanged`.
    #[test]
    fn a_package_of_another_whole_revision_in_the_place_is_refused_naming_both() {
        let project = Project::new("collision");
        let paths = ProjectPaths::new(&project.root);
        let (model, table) = (
            fixture("spike-enc-b32-int8.onnx"),
            fixture("science-2.1.table.bin"),
        );
        let first = EncoderPackage::write(
            &paths,
            &PackageParts {
                name: "first",
                arch: EncoderArch::V1,
                input_mode: InputMode::Whole,
                model_int8: &model,
                table: &table,
                training_record: b"{}",
                model_f32: None,
            },
        )
        .unwrap()
        .package;
        let before = snapshot(first.dir());
        let stored = fs::read(first.dir().join(ENCODER_JSON)).unwrap();
        let wanted: Manifest = serde_json::from_slice(&stored).unwrap();
        // The very same description is the same package.
        let same = EncoderPackage::existing(&paths, first.dir(), &wanted).unwrap();
        assert_eq!(same.outcome, WriteOutcome::Unchanged);

        // Another revision with the same first twelve characters and the same files.
        let mut other = wanted.clone();
        let found = wanted.identity.revision.clone();
        let last = if found.ends_with('0') { "1" } else { "0" };
        other.identity.revision = format!("{}{last}", &found[..39]);
        match EncoderPackage::existing(&paths, first.dir(), &other) {
            Err(L2dError::Refused { reason }) => {
                assert!(reason.contains(&found), "{reason}");
                assert!(reason.contains(&other.identity.revision), "{reason}");
                assert!(reason.contains("left unchanged"), "{reason}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert_eq!(snapshot(first.dir()), before);
    }

    /// Makes `link` a directory link to `target`: a symbolic link, or on Windows, where
    /// that needs a privilege, a junction.
    fn link_dir(link: &Path, target: &Path) {
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_dir(target, link).is_ok() {
                return;
            }
            let made = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(link)
                .arg(target)
                .output()
                .unwrap();
            assert!(made.status.success(), "no directory link could be made");
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, link).unwrap();
    }

    /// The comparison of resolved paths on its own, without the check for links that
    /// runs before it in `open` and `write`.
    #[test]
    fn a_directory_that_resolves_elsewhere_is_refused_by_the_resolved_path() {
        let project = Project::new("resolved");
        let paths = ProjectPaths::new(&project.root);
        let encoders = paths.encoders_dir();
        fs::create_dir_all(encoders.join("real-0123456789ab")).unwrap();
        require_resolved(&paths, &encoders.join("real-0123456789ab")).unwrap();

        // A link in `models/encoders/` to a directory outside it.
        let outside = project.root.join("elsewhere").join("linked-0123456789ab");
        fs::create_dir_all(&outside).unwrap();
        let link = encoders.join("linked-0123456789ab");
        link_dir(&link, &outside);
        match require_resolved(&paths, &link) {
            Err(L2dError::Refused { reason }) => {
                assert!(reason.contains("resolves to"), "{reason}");
                assert!(reason.contains("elsewhere"), "{reason}");
                assert!(reason.contains(&encoders.display().to_string()), "{reason}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // A link to a sibling package: the same parent, another name.
        let alias = encoders.join("alias-0123456789ab");
        link_dir(&alias, &encoders.join("real-0123456789ab"));
        match require_resolved(&paths, &alias) {
            Err(L2dError::Refused { reason }) => {
                assert!(reason.contains("resolves to"), "{reason}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // `models/` reached through a link resolves to the same place on both sides:
        // that case is the link check's, not this one's.
        let other_root = project.root.join("other-root");
        fs::create_dir_all(&other_root).unwrap();
        link_dir(&other_root.join("models"), &paths.models_dir());
        let through = ProjectPaths::new(&other_root);
        require_resolved(&through, &through.encoders_dir().join("real-0123456789ab")).unwrap();

        // Where the file system does not tell upper case from lower case, the other
        // spelling is the existing directory, and the refusal says so.
        let shouted = encoders.join("REAL-0123456789ab");
        if shouted.is_dir() {
            match require_resolved(&paths, &shouted) {
                Err(L2dError::Refused { reason }) => {
                    assert!(reason.contains("`real-0123456789ab`"), "{reason}");
                    assert!(reason.contains("upper case"), "{reason}");
                }
                other => panic!("expected a refusal, got {other:?}"),
            }
        }
    }

    fn entries_name(package: &EncoderPackage) -> String {
        package
            .dir()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    }
}
