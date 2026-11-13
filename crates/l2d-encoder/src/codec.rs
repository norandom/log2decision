//! `ByteCodec`: a log text becomes the encoder's input (task 2.5, requirements 1.5, 3.1).
//!
//! # The mapping
//!
//! The encoder reads bytes. Of the part of the text the [`InputMode`] selects, the first
//! `input_bytes` UTF-8 bytes are the byte ids; the rest of the ids is [`PAD_BYTE`] (0).
//! The graph input of one text is `x[c, 0, t] = table[id[t]][c]`: channel after channel,
//! `input_bytes` values each. A batch is `batch` such inputs one after the other, which
//! is the layout `[batch, channels, 1, input_bytes]` in row-major order that the model
//! takes (`[32, 64, 1, 256]` float32 for architecture v1, [`InputArch::V1`]).
//!
//! This is the mapping the trained model was fitted with (`byte_ids` and `table_input`
//! in `tools/science/l2dsci.py`); `tests/byte_codec.rs` holds the two together bit for
//! bit.
//!
//! # Truncation (requirement 1.5)
//!
//! A text is **truncated** when the part that is encoded is longer than `input_bytes`
//! **bytes**: the bytes after any input mode, not characters and not the bytes of the
//! whole text. In [`InputMode::Payload`] a long routing part therefore never truncates,
//! and a text of exactly `input_bytes` payload bytes is not truncated.
//!
//! The cut is at the byte. When it falls inside a multi-byte UTF-8 character, the bytes
//! of that character before the cut stay in the input and the others are dropped; the
//! codec does not back up to a character boundary, because the model was trained on raw
//! bytes cut this way.
//!
//! # Input modes
//!
//! The mode is given when the codec is made; there is no default. It is part of what an
//! encoder is: a model fitted in one mode must be fed in that mode, so the mode belongs
//! in the encoder identity (task 2.4).
//!
//! - [`InputMode::Whole`]: every byte of the text.
//! - [`InputMode::Payload`]: the payload part only, as the baseline's strip mode:
//!   everything after the first `?` or line feed
//!   ([`l2d_core::adapters::csic2010::split_routing`]). **A text without `?` and without
//!   line feed has no payload: its input is empty.** That is every ordinary log line
//!   outside the CSIC text layout.
//!
//! # The empty input
//!
//! [`TextInput::empty`] says that no byte other than padding reaches the model: the part
//! is empty, or every byte that is read is a NUL byte. The input is then all zeros, the
//! same for every such text, and so is the vector the model returns for it. Callers
//! count these logs as the baseline counts its logs without features.
//!
//! # NUL bytes
//!
//! A NUL byte in a text has the id of the padding byte, so the model sees padding at
//! its position (a trailing NUL byte is indistinguishable from a shorter text). The
//! codec does not refuse such a text (the store does); it counts the NUL bytes it read
//! in [`TextInput::nul_bytes`].
//!
//! # Memory and speed
//!
//! Nothing is allocated per text or per batch. A codec holds the table
//! ([`ByteCodec::held_bytes`]: 65,536 bytes for v1). An [`InputBatch`] is allocated once
//! and reused for every batch ([`InputBatch::held_bytes`]: 2 MiB of values for v1, plus
//! the flags of its texts, about 1 KiB). Neither grows with the number or the length of
//! the texts. [`InputBatch::values`] is one contiguous slice in the model's layout, so
//! a session can lend it to the runtime as the input tensor without a copy.
//!
//! # Determinism and untrusted data
//!
//! The input is a pure function of the text bytes, the table and the mode: values are
//! copied from the table, no arithmetic is done on them. Log text can hold attack
//! payloads; its bytes are used as table indices only.

use std::path::Path;

use l2d_core::L2dError;
use l2d_core::adapters::csic2010::split_routing;
use serde::{Deserialize, Serialize};

/// The number of rows of a byte table: one per byte value.
pub const BYTE_VALUES: usize = 256;

/// The id that fills an input after the last byte of the text. Row 0 of a byte table is
/// all zeros, so padding is zero input.
pub const PAD_BYTE: u8 = 0;

/// Largest accepted input length in bytes.
pub const MAX_INPUT_BYTES: u32 = 65_536;
/// Largest accepted number of values per byte.
pub const MAX_CHANNELS: u32 = 4_096;
/// Largest accepted batch size.
pub const MAX_BATCH: u32 = 4_096;
/// Largest accepted size of one batch of inputs ([`InputArch::batch_bytes`]): 256 MiB.
///
/// The three maxima above bound each field alone; together they would allow a batch of
/// 4 TiB. This ceiling bounds the buffer [`ByteCodec::new_batch`] allocates. It is 128
/// times the 2 MiB of architecture v1, which leaves room for v1 batches up to
/// [`MAX_BATCH`] texts (exactly 256 MiB) when larger batches are tried.
pub const MAX_BATCH_BYTES: u64 = 256 * 1024 * 1024;

/// Which part of a text the encoder reads (see the module documentation).
///
/// The names `whole` and `payload` ([`InputMode::as_str`], and the JSON form) are what an
/// encoder identity records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputMode {
    /// The whole text.
    Whole,
    /// The payload part only: what follows the first `?` or line feed. A text without
    /// either is an empty input.
    Payload,
}

impl InputMode {
    /// Both modes.
    pub const ALL: [InputMode; 2] = [InputMode::Whole, InputMode::Payload];

    /// `whole` or `payload`.
    pub fn as_str(self) -> &'static str {
        match self {
            InputMode::Whole => "whole",
            InputMode::Payload => "payload",
        }
    }

    /// The mode with exactly this name; any other name is [`L2dError::InvalidInput`].
    pub fn parse(name: &str) -> Result<Self, L2dError> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.as_str() == name)
            .ok_or_else(|| L2dError::InvalidInput {
                detail: format!(
                    "unknown encoder input mode `{name}`: the modes are `whole` and `payload`"
                ),
            })
    }

    /// The part of `text` that is encoded in this mode.
    pub fn select(self, text: &str) -> &str {
        match self {
            InputMode::Whole => text,
            InputMode::Payload => split_routing(text).payload,
        }
    }
}

/// The part of an encoder's architecture the codec needs: the shape of the model input.
///
/// The byte table belongs to it as a file of `256 x channels` values ([`ByteTable`]).
/// Task 2.4 makes this description part of the encoder package and its revision.
///
/// A value read through serde is validated ([`InputArch::validate`]) and refuses unknown
/// fields, so a description that comes out of a file is usable or an error. A value
/// written as a struct literal is checked when a [`ByteTable`] or a [`ByteCodec`] is made
/// from it. Because unknown fields are refused, it cannot be flattened into a larger
/// structure with `#[serde(flatten)]`: nest it, or build it from the fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "InputArchFields")]
pub struct InputArch {
    /// Bytes of a text the encoder reads; the last dimension of the model input.
    pub input_bytes: u32,
    /// Values per byte in the byte table; the channel dimension of the model input.
    pub channels: u32,
    /// Texts per inference; the model's batch dimension is fixed.
    pub batch: u32,
}

/// The serialized form of [`InputArch`], before validation.
#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename = "InputArch")]
struct InputArchFields {
    input_bytes: u32,
    channels: u32,
    batch: u32,
}

impl TryFrom<InputArchFields> for InputArch {
    type Error = L2dError;

    fn try_from(fields: InputArchFields) -> Result<Self, L2dError> {
        let arch = InputArch {
            input_bytes: fields.input_bytes,
            channels: fields.channels,
            batch: fields.batch,
        };
        arch.validate()?;
        Ok(arch)
    }
}

impl InputArch {
    /// Architecture v1: 256 bytes, 64 values per byte, batches of 32.
    pub const V1: InputArch = InputArch {
        input_bytes: 256,
        channels: 64,
        batch: 32,
    };

    /// Refuses a description with a zero or a value above the accepted maximum
    /// ([`MAX_INPUT_BYTES`], [`MAX_CHANNELS`], [`MAX_BATCH`]), naming the field, and one
    /// whose batch of inputs would be larger than [`MAX_BATCH_BYTES`].
    pub fn validate(&self) -> Result<(), L2dError> {
        for (name, value, max) in [
            ("input_bytes", self.input_bytes, MAX_INPUT_BYTES),
            ("channels", self.channels, MAX_CHANNELS),
            ("batch", self.batch, MAX_BATCH),
        ] {
            if value == 0 || value > max {
                return Err(L2dError::InvalidInput {
                    detail: format!(
                        "encoder input architecture: {name} is {value}, outside the range 1 to \
                         {max}"
                    ),
                });
            }
        }
        // Each factor is at most 65,536, so the product fits in 64 bits.
        let bytes =
            u64::from(self.batch) * u64::from(self.channels) * u64::from(self.input_bytes) * 4;
        if bytes > MAX_BATCH_BYTES {
            return Err(L2dError::InvalidInput {
                detail: format!(
                    "encoder input architecture: a batch of {} x {} x {} 32-bit values is \
                     {bytes} bytes, above the ceiling of {MAX_BATCH_BYTES} bytes (256 MiB)",
                    self.batch, self.channels, self.input_bytes
                ),
            });
        }
        Ok(())
    }

    /// The size of the byte table file: 256 rows of `channels` 32-bit floats.
    pub fn table_file_bytes(&self) -> u64 {
        BYTE_VALUES as u64 * self.channels as u64 * 4
    }

    /// The number of values of one text's input: `channels x input_bytes`.
    pub fn row_values(&self) -> usize {
        self.channels as usize * self.input_bytes as usize
    }

    /// The number of values of a batch: `batch x channels x input_bytes`.
    pub fn batch_values(&self) -> usize {
        self.batch as usize * self.row_values()
    }

    /// The bytes of a batch of inputs (32-bit floats): 2,097,152 for v1.
    pub fn batch_bytes(&self) -> u64 {
        self.batch_values() as u64 * 4
    }

    /// The dimensions of the model input: `[batch, channels, 1, input_bytes]`.
    pub fn dims(&self) -> [i64; 4] {
        [
            i64::from(self.batch),
            i64::from(self.channels),
            1,
            i64::from(self.input_bytes),
        ]
    }
}

/// The byte table of an encoder: for each of the 256 byte values, `channels` values.
///
/// The file is 256 rows of `channels` little-endian 32-bit floats in byte order. Row 0,
/// the row of the padding byte, must be all positive zeros; every value must be finite.
#[derive(Debug, Clone, PartialEq)]
pub struct ByteTable {
    channels: u32,
    /// `by_channel[c * 256 + byte]`: one 256-entry lookup row per channel, the order in
    /// which an input is written.
    by_channel: Box<[f32]>,
}

impl ByteTable {
    /// The table held in the bytes of a table file, for the architecture `arch`.
    ///
    /// [`L2dError::Refused`], naming the reason, when the size is not
    /// [`InputArch::table_file_bytes`], when row 0 holds anything but positive zeros, or
    /// when a value is not a finite number.
    pub fn from_le_bytes(bytes: &[u8], arch: &InputArch) -> Result<Self, L2dError> {
        Self::parse(bytes, arch, "byte table")
    }

    /// Reads a table file. A file that does not exist is [`L2dError::Missing`]; a file
    /// of the wrong size is refused before it is read.
    pub fn read(path: &Path, arch: &InputArch) -> Result<Self, L2dError> {
        arch.validate()?;
        let what = format!("byte table {}", path.display());
        let failed = |error: std::io::Error| L2dError::Runtime {
            detail: format!("reading the {what}: {error}"),
        };
        let metadata = match std::fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(L2dError::Missing {
                    item: "byte table of the encoder".into(),
                    expected_at: path.to_path_buf(),
                });
            }
            Err(error) => return Err(failed(error)),
        };
        if metadata.len() != arch.table_file_bytes() {
            return Err(wrong_size(&what, metadata.len(), arch));
        }
        let bytes = std::fs::read(path).map_err(failed)?;
        Self::parse(&bytes, arch, &what)
    }

    fn parse(bytes: &[u8], arch: &InputArch, what: &str) -> Result<Self, L2dError> {
        arch.validate()?;
        if bytes.len() as u64 != arch.table_file_bytes() {
            return Err(wrong_size(what, bytes.len() as u64, arch));
        }
        let channels = arch.channels as usize;
        let mut by_channel = vec![0.0f32; BYTE_VALUES * channels].into_boxed_slice();
        for (at, value) in bytes.as_chunks::<4>().0.iter().enumerate() {
            let value = f32::from_le_bytes(*value);
            let (row, channel) = (at / channels, at % channels);
            if row == PAD_BYTE as usize && value.to_bits() != 0 {
                return Err(L2dError::Refused {
                    reason: format!(
                        "{what}: row 0 is the row of the padding byte and must be all zeros, \
                         so that padding is zero input, but its channel {channel} holds {value}"
                    ),
                });
            }
            if !value.is_finite() {
                return Err(L2dError::Refused {
                    reason: format!(
                        "{what}: row {row}, channel {channel} holds {value}, which is not a \
                         finite number"
                    ),
                });
            }
            by_channel[channel * BYTE_VALUES + row] = value;
        }
        Ok(Self {
            channels: arch.channels,
            by_channel,
        })
    }

    /// Values per byte.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// `table[byte][channel]`.
    ///
    /// # Panics
    /// When `channel` is not below [`ByteTable::channels`].
    pub fn value(&self, byte: u8, channel: usize) -> f32 {
        assert!(
            channel < self.channels as usize,
            "channel {channel} of a byte table with {} channels",
            self.channels
        );
        self.by_channel[channel * BYTE_VALUES + byte as usize]
    }

    /// The heap bytes the table holds: the size of its file.
    pub fn held_bytes(&self) -> u64 {
        self.by_channel.len() as u64 * 4
    }
}

fn wrong_size(what: &str, found: u64, arch: &InputArch) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "{what}: {found} bytes, but {BYTE_VALUES} rows of {} little-endian 32-bit floats \
             are {} bytes",
            arch.channels,
            arch.table_file_bytes()
        ),
    }
}

/// What the codec did with one text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextInput {
    /// Bytes of the part of the text the input mode selects.
    pub part_bytes: usize,
    /// Bytes of that part that are in the input: at most the input length.
    pub read_bytes: usize,
    /// NUL bytes among the bytes read; each reads as padding at its position.
    pub nul_bytes: usize,
    /// The part is longer than the input length, so its end is not in the input
    /// (requirement 1.5).
    pub truncated: bool,
    /// No byte other than padding is in the input: it is all zeros.
    pub empty: bool,
}

/// Maps texts to model inputs (see the module documentation).
#[derive(Debug, Clone)]
pub struct ByteCodec {
    arch: InputArch,
    table: ByteTable,
    mode: InputMode,
}

impl ByteCodec {
    /// A codec for the architecture, its table and the input mode the encoder was fitted
    /// with. A table of another width than `arch.channels` is [`L2dError::Refused`].
    pub fn new(arch: InputArch, table: ByteTable, mode: InputMode) -> Result<Self, L2dError> {
        arch.validate()?;
        if table.channels != arch.channels {
            return Err(L2dError::Refused {
                reason: format!(
                    "the byte table has {} values per byte, the encoder input has {} channels",
                    table.channels, arch.channels
                ),
            });
        }
        Ok(Self { arch, table, mode })
    }

    pub fn arch(&self) -> &InputArch {
        &self.arch
    }

    pub fn mode(&self) -> InputMode {
        self.mode
    }

    pub fn table(&self) -> &ByteTable {
        &self.table
    }

    /// The heap bytes the codec holds for as long as it lives: the table.
    pub fn held_bytes(&self) -> u64 {
        self.table.held_bytes()
    }

    /// The bytes of `text` that are read, and the flags.
    fn read<'t>(&self, text: &'t str) -> (&'t [u8], TextInput) {
        let part = self.mode.select(text).as_bytes();
        let read = &part[..part.len().min(self.arch.input_bytes as usize)];
        let nul_bytes = read.iter().filter(|byte| **byte == PAD_BYTE).count();
        let flags = TextInput {
            part_bytes: part.len(),
            read_bytes: read.len(),
            nul_bytes,
            truncated: part.len() > read.len(),
            empty: nul_bytes == read.len(),
        };
        (read, flags)
    }

    /// The flags of a text, without producing its input.
    pub fn describe(&self, text: &str) -> TextInput {
        self.read(text).1
    }

    /// Writes the byte ids of `text` into `ids`: the bytes read, then padding.
    ///
    /// # Panics
    /// When `ids` is not `input_bytes` long.
    pub fn byte_ids(&self, text: &str, ids: &mut [u8]) -> TextInput {
        assert_eq!(
            ids.len(),
            self.arch.input_bytes as usize,
            "the id buffer must hold the {} bytes of one input",
            self.arch.input_bytes
        );
        let (read, flags) = self.read(text);
        ids[..read.len()].copy_from_slice(read);
        ids[read.len()..].fill(PAD_BYTE);
        flags
    }

    /// Writes the input of `text` into `row`, `[channels, 1, input_bytes]` in row-major
    /// order. Every value of `row` is written.
    ///
    /// # Panics
    /// When `row` is not [`InputArch::row_values`] long.
    pub fn encode_text(&self, text: &str, row: &mut [f32]) -> TextInput {
        assert_eq!(
            row.len(),
            self.arch.row_values(),
            "the row buffer must hold the {} x {} values of one input",
            self.arch.channels,
            self.arch.input_bytes
        );
        let (read, flags) = self.read(text);
        self.write_row(read, row);
        flags
    }

    /// The table lookup: for each channel, one value per byte read, then zeros.
    fn write_row(&self, read: &[u8], row: &mut [f32]) {
        let length = self.arch.input_bytes as usize;
        let lookups = self.table.by_channel.as_chunks::<BYTE_VALUES>().0;
        for (lookup, out) in lookups.iter().zip(row.chunks_exact_mut(length)) {
            let (text, padding) = out.split_at_mut(read.len());
            for (value, byte) in text.iter_mut().zip(read) {
                *value = lookup[*byte as usize];
            }
            padding.fill(0.0);
        }
    }

    /// A batch buffer for this codec: all-zero inputs, no text. Allocate it once and
    /// reuse it; this is the only allocation of encoding.
    pub fn new_batch(&self) -> InputBatch {
        InputBatch {
            arch: self.arch,
            values: vec![0.0; self.arch.batch_values()].into_boxed_slice(),
            texts: vec![TextInput::default(); self.arch.batch as usize].into_boxed_slice(),
            rows: 0,
            written: 0,
        }
    }

    /// Adds `text` as the next row of `batch`. `None`, and nothing written, when the
    /// batch is full.
    ///
    /// # Panics
    /// When `batch` was made for another architecture.
    pub fn push(&self, batch: &mut InputBatch, text: &str) -> Option<TextInput> {
        assert_eq!(
            batch.arch, self.arch,
            "the batch was made for another architecture"
        );
        if batch.is_full() {
            return None;
        }
        let (read, flags) = self.read(text);
        let row = self.arch.row_values();
        let at = batch.rows;
        self.write_row(read, &mut batch.values[at * row..(at + 1) * row]);
        batch.texts[at] = flags;
        batch.rows += 1;
        batch.written = batch.written.max(batch.rows);
        Some(flags)
    }

    /// Starts `batch` anew and fills it with the next texts of `texts`, at most
    /// `batch` of them; returns how many it took (also [`InputBatch::rows`]). Fewer than
    /// a full batch means the texts ran out: the rest of the batch is all-zero inputs.
    /// Zero means there was no text left.
    ///
    /// Only the rows that held a text before and hold none now are zeroed, so a full
    /// batch costs nothing beyond its texts.
    ///
    /// # Panics
    /// When `batch` was made for another architecture.
    pub fn fill<'t, I>(&self, batch: &mut InputBatch, texts: &mut I) -> usize
    where
        I: Iterator<Item = &'t str>,
    {
        assert_eq!(
            batch.arch, self.arch,
            "the batch was made for another architecture"
        );
        // Rows up to `written` may hold values of the previous batch until the end of
        // this call; nothing reads them in between.
        batch.rows = 0;
        while !batch.is_full() {
            let Some(text) = texts.next() else { break };
            self.push(batch, text);
        }
        batch.zero_unused();
        batch.rows
    }
}

/// The reusable buffer of one batch of model inputs, with the flags of its texts.
///
/// Between calls, every row from [`InputBatch::rows`] on is an all-zero input, which is
/// how the last batch of a run is padded: the caller drops the model's output for those
/// rows.
#[derive(Debug, Clone)]
pub struct InputBatch {
    arch: InputArch,
    values: Box<[f32]>,
    texts: Box<[TextInput]>,
    /// Rows that hold a text.
    rows: usize,
    /// Rows that may hold non-zero values; at least `rows`.
    written: usize,
}

impl InputBatch {
    /// The architecture the batch was made for.
    pub fn arch(&self) -> &InputArch {
        &self.arch
    }

    /// The number of texts in the batch; rows from here on are padding.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The batch size of the architecture.
    pub fn capacity(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn is_full(&self) -> bool {
        self.rows == self.texts.len()
    }

    /// Removes the texts: every row is an all-zero input again.
    pub fn clear(&mut self) {
        self.rows = 0;
        self.zero_unused();
    }

    /// Zeroes the rows that hold no text but may hold values.
    fn zero_unused(&mut self) {
        let row = self.arch.row_values();
        self.values[self.rows * row..self.written * row].fill(0.0);
        self.written = self.rows;
    }

    /// The whole batch, always [`InputArch::batch_values`] values: the model input
    /// `[batch, channels, 1, input_bytes]` in row-major order, ready to be lent to the
    /// runtime as it is.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The dimensions of [`InputBatch::values`].
    pub fn dims(&self) -> [i64; 4] {
        self.arch.dims()
    }

    /// The flags of the texts in the batch, in the order they were added.
    pub fn texts(&self) -> &[TextInput] {
        &self.texts[..self.rows]
    }

    /// The heap bytes the buffer holds, whatever it contains: the values
    /// ([`InputArch::batch_bytes`]) and the flags of a full batch.
    pub fn held_bytes(&self) -> u64 {
        self.arch.batch_bytes() + std::mem::size_of_val::<[TextInput]>(&self.texts) as u64
    }
}
