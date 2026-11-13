//! The arithmetic around the model that task 2.12 measured on the NPU, for the session
//! of task 2.6 to lift:
//!
//! - [`ImageInput`]: log texts written as the rows of one image, the layout in which the
//!   NPU computes every text right (`onnx_edit::with_rows_as_height`), as floats or as
//!   the bytes the model's first node would make of them;
//! - [`RowImage`]: the image buffer of the session (task 2.6), the same values written
//!   channel by channel into one buffer that is reused;
//! - [`read_rows`]: the vectors of such an image's output, L2-normalized;
//! - [`quantize_uint8`] and [`dequantize_uint8`]: the rule of the 8-bit graph's first and
//!   last node, so that the tool can feed and read the bytes itself;
//! - [`l2_normalize`]: the step after the graph;
//! - [`provider_cache_key`]: the name of the provider's cache folder for a compiled model.
//!
//! Nothing here loads the runtime.

use sha2::{Digest, Sha256};

use crate::codec::{ByteCodec, TextInput};

/// What ONNX Runtime's `QuantizeLinear` computes for an unsigned 8-bit output:
/// `saturate(round_half_even(value / scale) + zero_point)`, with the division in 32-bit
/// floats and the sum in integers. `value` and `scale` must be finite and `scale`
/// positive, as the model reader requires of a model's scale.
pub fn quantize_uint8(value: f32, scale: f32, zero_point: u8) -> u8 {
    let steps = (value / scale).round_ties_even();
    // The cast saturates at the ends of the 32-bit range, far outside what a byte holds.
    let shifted = i64::from(steps as i32) + i64::from(zero_point);
    shifted.clamp(0, 255) as u8
}

/// What ONNX Runtime's `DequantizeLinear` computes: `(byte - zero_point) * scale`.
pub fn dequantize_uint8(byte: u8, scale: f32, zero_point: u8) -> f32 {
    (i32::from(byte) - i32::from(zero_point)) as f32 * scale
}

/// Scales `vector` to length 1 and returns the length it had. A vector shorter than
/// `1e-12` is divided by `1e-12`, as the science gate's `l2n` does: an all-zero vector
/// stays all zeros.
pub fn l2_normalize(vector: &mut [f32]) -> f32 {
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt() as f32;
    let divisor = norm.max(1e-12);
    for value in vector.iter_mut() {
        *value /= divisor;
    }
    norm
}

/// How many of the `256 * channels` table values of `codec` lie outside the range a
/// model input quantized with `scale` and `zero_point` can express and are clipped.
pub fn clipped_table_values(codec: &ByteCodec, scale: f32, zero_point: u8) -> usize {
    let channels = codec.arch().channels as usize;
    (0..=255u8)
        .flat_map(|byte| (0..channels).map(move |channel| (byte, channel)))
        .filter(|(byte, channel)| {
            let steps = (codec.table().value(*byte, *channel) / scale).round_ties_even();
            !(0.0..=255.0).contains(&(steps + f32::from(zero_point)))
        })
        .count()
}

/// Writes log texts as the rows of one image `[1, channels, rows, input_bytes]`:
/// `image[c * rows * input_bytes + n * input_bytes + t] = table[byte[n, t]][c]` for the
/// text in row `n`, and zero for padding. With one row this is [`ByteCodec::encode_text`]'s
/// layout. `T` is `f32` for the model's float input ([`ImageInput::float`]) or `u8` for
/// its quantized input fed directly ([`ImageInput::quantized`]).
///
/// The byte ids, the truncation rule and the flags are the codec's
/// ([`ByteCodec::byte_ids`]); this type only lays the looked-up values out differently.
#[derive(Debug, Clone)]
pub struct ImageInput<T> {
    /// `lookups[c * 256 + byte]`.
    lookups: Box<[T]>,
    channels: usize,
    input_bytes: usize,
    zero: T,
    ids: Box<[u8]>,
}

impl ImageInput<f32> {
    /// The float input, as the model's `QuantizeLinear` node reads it.
    pub fn float(codec: &ByteCodec) -> Self {
        Self::build(codec, 0.0, |value| value)
    }
}

impl ImageInput<u8> {
    /// The input in the bytes the model's `QuantizeLinear` node would produce with
    /// `scale` and `zero_point`: for a model whose input is those bytes
    /// (`onnx_edit::with_quantized_ends`).
    pub fn quantized(codec: &ByteCodec, scale: f32, zero_point: u8) -> Self {
        Self::build(codec, quantize_uint8(0.0, scale, zero_point), |value| {
            quantize_uint8(value, scale, zero_point)
        })
    }
}

impl<T: Copy> ImageInput<T> {
    fn build(codec: &ByteCodec, zero: T, convert: impl Fn(f32) -> T) -> Self {
        let arch = codec.arch();
        let channels = arch.channels as usize;
        let mut lookups = Vec::with_capacity(channels * 256);
        for channel in 0..channels {
            for byte in 0..=255u8 {
                lookups.push(convert(codec.table().value(byte, channel)));
            }
        }
        Self {
            lookups: lookups.into_boxed_slice(),
            channels,
            input_bytes: arch.input_bytes as usize,
            zero,
            ids: vec![0u8; arch.input_bytes as usize].into_boxed_slice(),
        }
    }

    /// The values of an image of `rows` texts.
    pub fn image_values(&self, rows: usize) -> usize {
        self.channels * rows * self.input_bytes
    }

    /// The value an all-zero (padding) input is made of.
    pub fn zero(&self) -> T {
        self.zero
    }

    /// Writes `text` as row `row` of an image of `rows` rows. Every value of that row is
    /// written, in every channel; the other rows are not touched. The flags are the
    /// codec's.
    ///
    /// # Panics
    /// When `image` is not [`ImageInput::image_values`] long or `row` is not below `rows`.
    pub fn write(
        &mut self,
        codec: &ByteCodec,
        text: &str,
        rows: usize,
        row: usize,
        image: &mut [T],
    ) -> TextInput {
        assert_eq!(image.len(), self.image_values(rows), "the image buffer");
        assert!(row < rows, "row {row} of an image of {rows} rows");
        let flags = codec.byte_ids(text, &mut self.ids);
        let read = &self.ids[..flags.read_bytes];
        let plane = rows * self.input_bytes;
        let lookups = self.lookups.as_chunks::<256>().0;
        for (lookup, plane) in lookups.iter().zip(image.chunks_exact_mut(plane)) {
            let line = &mut plane[row * self.input_bytes..(row + 1) * self.input_bytes];
            let (text, padding) = line.split_at_mut(read.len());
            for (value, byte) in text.iter_mut().zip(read) {
                *value = lookup[*byte as usize];
            }
            padding.fill(self.zero);
        }
        flags
    }

    /// Makes row `row` of an image an empty input (padding only).
    pub fn clear_row(&self, rows: usize, row: usize, image: &mut [T]) {
        assert_eq!(image.len(), self.image_values(rows), "the image buffer");
        for plane in image.chunks_exact_mut(rows * self.input_bytes) {
            plane[row * self.input_bytes..(row + 1) * self.input_bytes].fill(self.zero);
        }
    }
}

/// The image buffer of a session (task 2.6): `[1, channels, rows, input_bytes]` float
/// values, allocated once and written again for every image of a run.
///
/// It holds the same values [`ImageInput::write`] and [`ImageInput::clear_row`] produce,
/// bit for bit (a test holds the two together), and differs in the order of the work:
///
/// - the bytes of all texts of an image are taken first, then the image is written
///   channel by channel, each channel's plane from its first row to its last. The buffer
///   is walked once from start to end. [`ImageInput::write`] goes text by text and
///   touches one line in each of the channel planes per text, lines that lie a whole
///   plane apart (64 KiB at 64 rows);
/// - it remembers how many bytes each row holds, so that after a text only the positions
///   the previous text of that row filled and this one does not are set back to zero. A
///   row that holds no text any more is emptied the same way.
///
/// The other order was tried for speed (task 2.12 had measured its writer at 13.8 us per
/// log inside a run against 3 to 4 us for the codec's batch layout) and gains nothing:
/// on CSIC's 25,608 distinct texts, writers alone on one thread, 15 alternating passes,
/// 3.14 us per log against 3.21 us in `payload` mode and 3.75 against 3.71 in `whole`
/// mode, with the batch layout at 3.00 and 3.58
/// (`tests/byte_codec_speed.rs`, `csic_throughput_of_the_image_writers_on_one_thread`).
/// The layout of the image is not what made the writer slow in that measurement. The
/// type is kept as the session's buffer because it owns the image and the flags of its
/// texts and allocates nothing after it is made.
#[derive(Debug, Clone)]
pub struct RowImage {
    codec: ByteCodec,
    /// `lookups[c * 256 + byte]`.
    lookups: Box<[f32]>,
    rows: usize,
    input_bytes: usize,
    values: Box<[f32]>,
    /// The bytes read of each row's text, `rows * input_bytes`.
    ids: Box<[u8]>,
    /// `lengths[n]`: the values of row `n` from this position on are zero in every channel.
    lengths: Box<[usize]>,
    flags: Box<[TextInput]>,
    /// Rows that hold a text.
    used: usize,
}

impl RowImage {
    /// An all-zero image of `rows` rows for the codec's architecture.
    ///
    /// # Panics
    /// When `rows` is zero.
    pub fn new(codec: ByteCodec, rows: usize) -> Self {
        assert!(rows > 0, "an image has at least one row");
        let arch = *codec.arch();
        let channels = arch.channels as usize;
        let input_bytes = arch.input_bytes as usize;
        let mut lookups = Vec::with_capacity(channels * 256);
        for channel in 0..channels {
            for byte in 0..=255u8 {
                lookups.push(codec.table().value(byte, channel));
            }
        }
        Self {
            codec,
            lookups: lookups.into_boxed_slice(),
            rows,
            input_bytes,
            values: vec![0.0; channels * rows * input_bytes].into_boxed_slice(),
            ids: vec![0u8; rows * input_bytes].into_boxed_slice(),
            lengths: vec![0usize; rows].into_boxed_slice(),
            flags: vec![TextInput::default(); rows].into_boxed_slice(),
            used: 0,
        }
    }

    /// The codec whose mapping the image holds.
    pub fn codec(&self) -> &ByteCodec {
        &self.codec
    }

    /// The rows of the image.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The shape of the image as a tensor: `[1, channels, rows, input_bytes]`.
    pub fn dims(&self) -> [i64; 4] {
        [
            1,
            i64::from(self.codec.arch().channels),
            self.rows as i64,
            self.input_bytes as i64,
        ]
    }

    /// Makes `texts` the rows of the image, text `n` as row `n`; the rows after them are
    /// empty inputs. Returns the flags of the texts, in order.
    ///
    /// # Panics
    /// When there are more texts than rows.
    pub fn fill(&mut self, texts: &[&str]) -> &[TextInput] {
        assert!(
            texts.len() <= self.rows,
            "{} texts for an image of {} rows",
            texts.len(),
            self.rows
        );
        let width = self.input_bytes;
        for (row, text) in texts.iter().enumerate() {
            let (read, flags) = self.codec.read(text);
            self.ids[row * width..row * width + read.len()].copy_from_slice(read);
            self.flags[row] = flags;
        }
        let used = texts.len();
        let lookups = self.lookups.as_chunks::<256>().0;
        for (lookup, plane) in lookups
            .iter()
            .zip(self.values.chunks_exact_mut(self.rows * width))
        {
            for row in 0..used {
                let new = self.flags[row].read_bytes;
                let old = self.lengths[row];
                let line = &mut plane[row * width..(row + 1) * width];
                let ids = &self.ids[row * width..row * width + new];
                for (value, id) in line[..new].iter_mut().zip(ids) {
                    *value = lookup[*id as usize];
                }
                if old > new {
                    line[new..old].fill(0.0);
                }
            }
            for row in used..self.used {
                plane[row * width..row * width + self.lengths[row]].fill(0.0);
            }
        }
        for row in 0..used {
            self.lengths[row] = self.flags[row].read_bytes;
        }
        for row in used..self.used {
            self.lengths[row] = 0;
        }
        self.used = used;
        &self.flags[..used]
    }

    /// The values of the image, one contiguous slice in the tensor's order.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The flags of the texts the image holds.
    pub fn flags(&self) -> &[TextInput] {
        &self.flags[..self.used]
    }

    /// The heap bytes the buffer holds for as long as it lives, whatever the texts: the
    /// values (`channels * rows * input_bytes` floats), the codec's table and its copy in
    /// channel order, the bytes of one image's texts, and a length and the flags per row.
    pub fn held_bytes(&self) -> u64 {
        (std::mem::size_of_val(&*self.values)
            + std::mem::size_of_val(&*self.lookups)
            + std::mem::size_of_val(&*self.ids)
            + std::mem::size_of_val(&*self.lengths)
            + std::mem::size_of_val(&*self.flags)) as u64
            + self.codec.held_bytes()
    }
}

/// The vectors of an image's output `[1, dimension, rows, 1]`: for each of the first
/// `used` rows, `vector[d] = output[d * rows + n]`, L2-normalized, written to `vectors`
/// one after the other (`used * dimension` values).
///
/// # Panics
/// When the lengths do not fit.
pub fn read_rows(output: &[f32], rows: usize, used: usize, vectors: &mut [f32]) {
    assert!(used <= rows && rows > 0, "{used} of {rows} rows");
    assert_eq!(
        output.len() % rows,
        0,
        "the output is not an image of {rows} rows"
    );
    let dimension = output.len() / rows;
    assert_eq!(vectors.len(), used * dimension, "the vector buffer");
    for (n, vector) in vectors.chunks_exact_mut(dimension).enumerate() {
        for (d, value) in vector.iter_mut().enumerate() {
            *value = output[d * rows + n];
        }
        l2_normalize(vector);
    }
}

/// The name of the provider's cache folder for one compiled model: the first twelve
/// characters of the encoder revision, the runtime release, and sixteen hexadecimal
/// characters of a digest over what the provider compiles: `model`, the bytes of the
/// model file that is actually handed to the provider (after any rewrite: they carry
/// the layout and the rows), and `options`, every provider option with its value, in
/// the order given.
///
/// The provider looks at nothing but this name (task 2.12): under a name that is
/// reused it runs the old compile, without an error, and for another model of the
/// same shape that gives wrong vectors. Two sessions may share a compiled model only
/// when all three parts agree.
pub fn provider_cache_key(
    revision: &str,
    release: &str,
    model: &[u8],
    options: &[(&str, &str)],
) -> String {
    let mut digest = Sha256::new();
    digest.update((model.len() as u64).to_le_bytes());
    digest.update(Sha256::digest(model));
    for (name, value) in options {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    let digest: String = digest
        .finalize()
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let revision: String = revision.chars().take(12).collect();
    format!("{revision}-{release}-{digest}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{ByteTable, InputArch, InputMode};

    #[test]
    fn quantization_follows_the_runtime_rule() {
        let scale = 0.5f32;
        // Halves round to the even neighbour.
        assert_eq!(quantize_uint8(0.25, scale, 100), 100); // 0.5 -> 0
        assert_eq!(quantize_uint8(0.75, scale, 100), 102); // 1.5 -> 2
        assert_eq!(quantize_uint8(1.25, scale, 100), 102); // 2.5 -> 2
        assert_eq!(quantize_uint8(-0.25, scale, 100), 100); // -0.5 -> -0
        assert_eq!(quantize_uint8(-0.75, scale, 100), 98); // -1.5 -> -2
        assert_eq!(quantize_uint8(0.0, scale, 134), 134);
        // Saturation at both ends, also far outside.
        assert_eq!(quantize_uint8(1000.0, scale, 100), 255);
        assert_eq!(quantize_uint8(-1000.0, scale, 100), 0);
        assert_eq!(quantize_uint8(3.0e38, 1.0e-30, 100), 255);
        assert_eq!(quantize_uint8(-3.0e38, 1.0e-30, 100), 0);
        // With the published model's scale.
        let scale = f32::from_bits(0x3CFC_E010);
        assert_eq!(dequantize_uint8(134, scale, 134), 0.0);
        assert_eq!(dequantize_uint8(135, scale, 134), scale);
        assert_eq!(dequantize_uint8(0, scale, 134), -134.0 * scale);
        // A byte survives the round trip.
        for byte in 0..=255u8 {
            assert_eq!(
                quantize_uint8(dequantize_uint8(byte, scale, 134), scale, 134),
                byte
            );
        }
    }

    #[test]
    fn normalization_gives_unit_length_and_keeps_zero() {
        let mut vector = [3.0f32, 4.0, 0.0];
        assert_eq!(l2_normalize(&mut vector), 5.0);
        assert_eq!(vector, [0.6, 0.8, 0.0]);
        let mut zero = [0.0f32; 4];
        assert_eq!(l2_normalize(&mut zero), 0.0);
        assert_eq!(zero, [0.0; 4]);
    }

    fn small_codec() -> ByteCodec {
        let arch = InputArch {
            input_bytes: 6,
            channels: 2,
            batch: 2,
        };
        // Row `b` holds (b / 16, -b / 16); row 0 is zeros.
        let bytes: Vec<u8> = (0..256)
            .flat_map(|byte| {
                let value = byte as f32 / 16.0;
                // Row 0 must be positive zeros.
                [value, 0.0 - value]
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        let table = ByteTable::from_le_bytes(&bytes, &arch).unwrap();
        ByteCodec::new(arch, table, InputMode::Whole).unwrap()
    }

    const TEXTS: [&str; 6] = [
        "",
        "a",
        "abcdef",
        "abcdefgh",
        "\u{00e9}\u{00e9}\u{00e9}\u{00e9}",
        "a\0b",
    ];

    #[test]
    fn an_image_of_one_row_is_the_codecs_input() {
        let codec = small_codec();
        let mut image = ImageInput::float(&codec);
        assert_eq!(image.image_values(1), 12);
        assert_eq!(image.image_values(3), 36);
        for text in TEXTS {
            let mut expected = [9.0f32; 12];
            let flags = codec.encode_text(text, &mut expected);
            let mut written = [7.0f32; 12];
            assert_eq!(image.write(&codec, text, 1, 0, &mut written), flags);
            assert_eq!(written, expected, "{text:?}");
        }
    }

    #[test]
    fn the_rows_of_an_image_are_the_texts_channel_by_channel() {
        let codec = small_codec();
        let mut image = ImageInput::float(&codec);
        let rows = TEXTS.len();
        let mut values = vec![5.0f32; image.image_values(rows)];
        for (row, text) in TEXTS.iter().enumerate() {
            image.write(&codec, text, rows, row, &mut values);
        }
        for (row, text) in TEXTS.iter().enumerate() {
            let mut expected = [0.0f32; 12];
            codec.encode_text(text, &mut expected);
            for channel in 0..2 {
                let at = channel * rows * 6 + row * 6;
                assert_eq!(
                    values[at..at + 6],
                    expected[channel * 6..(channel + 1) * 6],
                    "row {row}, channel {channel}"
                );
            }
        }
        // Writing a row again leaves the others alone; clearing makes it padding.
        let before = values.clone();
        image.write(&codec, "zz", rows, 2, &mut values);
        image.write(&codec, TEXTS[2], rows, 2, &mut values);
        assert_eq!(values, before);
        image.clear_row(rows, 1, &mut values);
        assert_eq!(values[6..12], [0.0; 6]);
        assert_eq!(values[rows * 6 + 6..rows * 6 + 12], [0.0; 6]);
        assert_eq!(values[..6], before[..6]);
    }

    #[test]
    fn the_quantized_image_is_the_quantized_float_image() {
        let codec = small_codec();
        let (scale, zero) = (0.125f32, 100u8);
        let mut floats = ImageInput::float(&codec);
        let mut bytes = ImageInput::quantized(&codec, scale, zero);
        assert_eq!(bytes.zero(), 100);
        let rows = TEXTS.len();
        let mut float_image = vec![0.0f32; floats.image_values(rows)];
        let mut byte_image = vec![0u8; bytes.image_values(rows)];
        for (row, text) in TEXTS.iter().enumerate() {
            let flags = floats.write(&codec, text, rows, row, &mut float_image);
            assert_eq!(bytes.write(&codec, text, rows, row, &mut byte_image), flags);
        }
        let expected: Vec<u8> = float_image
            .iter()
            .map(|value| quantize_uint8(*value, scale, zero))
            .collect();
        assert_eq!(byte_image, expected);
        // `a` is byte 97: 97 / 16 / 0.125 = 48.5 -> 48 (even), and -48.5 -> -48.
        assert_eq!(byte_image[6], 148);
        assert_eq!(byte_image[rows * 6 + 6], 52);
        assert_eq!(byte_image[7..12], [100; 5]);
        // Clipped values are counted: 255 / 16 / 0.125 = 127.5 -> 128 -> 228 is inside,
        // while -b / 2 falls below -100 steps for the 54 bytes 202 to 255. With a coarse
        // scale nothing leaves the range.
        assert_eq!(clipped_table_values(&codec, scale, zero), 54);
        assert_eq!(clipped_table_values(&codec, 1.0, zero), 0);
    }

    #[test]
    fn the_rows_of_an_output_are_read_and_normalized() {
        // dimension 2, rows 3: output[d * rows + n].
        let output = [3.0f32, 0.0, 1.0, 4.0, 0.0, 1.0];
        let mut vectors = [9.0f32; 4];
        read_rows(&output, 3, 2, &mut vectors);
        assert_eq!(vectors, [0.6, 0.8, 0.0, 0.0]);
        let mut all = [0.0f32; 6];
        read_rows(&output, 3, 3, &mut all);
        let half = 0.5f32.sqrt();
        assert_eq!(all[4..], [half, half]);
    }

    #[test]
    fn the_cache_key_names_revision_release_model_and_options() {
        let revision = "099570ec739ce4d0548c2d15629e857cf516664f";
        let model = b"the bytes handed to the provider";
        let options = [("opt_level", "0"), ("target", "X2")];
        let key = provider_cache_key(revision, "1.9.0", model, &options);
        assert!(key.starts_with("099570ec739c-1.9.0-"), "{key}");
        assert_eq!(key.len(), 12 + 1 + 5 + 1 + 16);
        assert!(key[19..].chars().all(|c| c.is_ascii_hexdigit()), "{key}");
        assert_eq!(key, provider_cache_key(revision, "1.9.0", model, &options));
        for other in [
            provider_cache_key(
                "199570ec739ce4d0548c2d15629e857cf516664f",
                "1.9.0",
                model,
                &options,
            ),
            provider_cache_key(revision, "1.10.0", model, &options),
            // One byte of the model, its length, an option's value, an option less, and
            // a byte moved between a name and its value.
            provider_cache_key(
                revision,
                "1.9.0",
                b"The bytes handed to the provider",
                &options,
            ),
            provider_cache_key(revision, "1.9.0", &model[..31], &options),
            provider_cache_key(
                revision,
                "1.9.0",
                model,
                &[("opt_level", "3"), ("target", "X2")],
            ),
            provider_cache_key(revision, "1.9.0", model, &options[..1]),
            provider_cache_key(revision, "1.9.0", model, &[]),
            provider_cache_key(
                revision,
                "1.9.0",
                model,
                &[("opt_leve", "l0"), ("target", "X2")],
            ),
        ] {
            assert_ne!(key, other);
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// The image buffer of the session against `ImageInput`, the writer task 2.12
    /// measured with: the same values bit for bit, image after image in one buffer.
    #[test]
    fn the_image_buffer_holds_what_the_row_writer_writes() {
        let codec = small_codec();
        let rows = 4;
        let mut writer = ImageInput::float(&codec);
        let mut buffer = RowImage::new(codec.clone(), rows);
        assert_eq!(buffer.rows(), rows);
        assert_eq!(buffer.dims(), [1, 2, 4, 6]);
        assert_eq!(buffer.values().len(), writer.image_values(rows));
        assert!(buffer.values().iter().all(|value| value.to_bits() == 0));
        assert!(buffer.flags().is_empty());

        // Long after short and short after long in the same row, a NUL byte, an empty
        // text, images that are not full, an image without a text, a full one again.
        let images: [&[&str]; 8] = [
            &["abcdef", "abcdefgh", "a", ""],
            &["a", "ab", "abcdef", "abc"],
            &["abcdefgh", "a\0b"],
            &["\u{00e9}\u{00e9}\u{00e9}\u{00e9}"],
            &[],
            &["", "", ""],
            &["zzzzzz", "yyyyyy", "xxxxxx", "wwwwww"],
            &["a"],
        ];
        for (index, texts) in images.iter().enumerate() {
            let mut expected = vec![0.0f32; writer.image_values(rows)];
            let mut flags = Vec::new();
            for (row, text) in texts.iter().enumerate() {
                flags.push(writer.write(&codec, text, rows, row, &mut expected));
            }
            assert_eq!(buffer.fill(texts), &flags[..], "image {index}");
            assert_eq!(buffer.flags(), &flags[..], "image {index}");
            assert_eq!(bits(buffer.values()), bits(&expected), "image {index}");
        }
    }

    #[test]
    fn the_image_buffer_states_what_it_holds() {
        let codec = small_codec();
        // 2 channels, 3 rows, 6 bytes: 36 values; the table twice (2 x 256 values each);
        // 18 bytes of texts; a length and the flags per row.
        let buffer = RowImage::new(codec.clone(), 3);
        let per_row = std::mem::size_of::<usize>() + std::mem::size_of::<TextInput>();
        assert_eq!(
            buffer.held_bytes(),
            (36 * 4 + 2 * 256 * 4 + 18 + 3 * per_row) as u64 + codec.held_bytes()
        );
        assert_eq!(codec.held_bytes(), 2 * 256 * 4);
    }

    #[test]
    #[should_panic(expected = "5 texts for an image of 4 rows")]
    fn more_texts_than_rows_are_a_programming_error() {
        RowImage::new(small_codec(), 4).fill(&["a"; 5]);
    }
}
