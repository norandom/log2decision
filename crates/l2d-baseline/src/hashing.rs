//! Character n-gram feature hashing: the feature rows of the baseline scorer.
//!
//! # Feature rule
//!
//! A text is read as its UTF-8 bytes. Every window of 1, 2, … `ngram_max` consecutive
//! bytes (default 3) is one n-gram; a part shorter than `n` bytes has no n-grams of
//! length `n`. Each n-gram is hashed on its own with 64-bit FNV-1a ([`fnv1a_64`]):
//!
//! - **bucket** = (low 32 bits of the hash) modulo `buckets` (default 65,536, for which
//!   this is the low 16 bits),
//! - **sign** = `+1` when the high 32 bits of the hash hold an even number of one bits,
//!   `-1` when odd.
//!
//! Bucket and sign therefore come from disjoint halves of the hash. The sign uses the
//! parity of the whole high half rather than one fixed bit because single high bits of
//! FNV-1a are constant over all one-byte inputs.
//!
//! The signs of all n-grams that fall into one bucket are added up as integers. Buckets
//! whose sum is zero are dropped. The remaining sums are divided by the Euclidean norm of
//! the row, so a non-empty row has norm 1. A text without n-grams, or one whose sums all
//! cancel, gives an empty row.
//!
//! # Determinism
//!
//! The row is a pure function of the text bytes and the settings. Counts and the sum of
//! squares are exact integers; the only floating-point operations are one integer to
//! `f64` conversion, one square root and one division per entry, all of which IEEE 754
//! rounds correctly. Nothing depends on the process, on addresses, on map iteration
//! order, on the locale or on the byte order of the platform, so equal input gives
//! bit-equal rows in every run and process.
//!
//! # Routing context
//!
//! [`RoutingMode`] decides what happens to the routing part of a text (method and URL
//! path), so that a model can be kept from memorizing endpoints. The split is
//! [`l2d_core::adapters::csic2010::split_routing`]: the routing part is everything
//! before the first `?` or line feed, the payload part everything after it, and that one
//! delimiter byte belongs to neither part. No n-gram spans the two parts.
//!
//! **A text without `?` and without line feed is routing only.** With
//! [`RoutingMode::Strip`] its row is empty, and with [`RoutingMode::Isolate`] all of it
//! lands in the routing half. The split is meant for the CSIC text layout
//! (`METHOD URL[?QUERY][LF BODY]`); a text in another layout is cut at whatever `?` or
//! line feed it happens to contain.
//!
//! # Untrusted data
//!
//! Log text can hold attack payloads. It is hashed as bytes only: never executed and
//! never used to build paths or commands.

use l2d_core::adapters::csic2010::split_routing;

/// Number of buckets of the default feature space.
pub const DEFAULT_BUCKETS: u32 = 65_536;

/// Longest n-gram, in bytes, of the default feature rule.
pub const DEFAULT_NGRAM_MAX: u8 = 3;

/// Largest accepted bucket count; keeps the doubled index space of
/// [`RoutingMode::Isolate`] inside `u32`.
pub const MAX_BUCKETS: u32 = 1 << 30;

/// What to do with the routing part of a text (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RoutingMode {
    /// Hash the whole text, delimiter included. The plain feature rule.
    #[default]
    Keep,
    /// Hash the payload part only. A text without payload gives an empty row.
    Strip,
    /// Hash both parts into disjoint index ranges: payload n-grams into
    /// `0..buckets` (the same indices as with [`RoutingMode::Strip`]), routing n-grams
    /// into `buckets..2 * buckets`. The row is normalized as a whole.
    Isolate,
}

/// Settings of the feature rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HashingSettings {
    /// Number of hash buckets per feature space, `1..=MAX_BUCKETS`.
    pub buckets: u32,
    /// Longest n-gram in bytes, at least 1.
    pub ngram_max: u8,
    /// Treatment of the routing part.
    pub routing: RoutingMode,
}

impl Default for HashingSettings {
    fn default() -> Self {
        Self {
            buckets: DEFAULT_BUCKETS,
            ngram_max: DEFAULT_NGRAM_MAX,
            routing: RoutingMode::Keep,
        }
    }
}

/// Why a [`HashingSettings`] value was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashingError {
    /// `buckets` is zero or above [`MAX_BUCKETS`].
    BucketsOutOfRange { buckets: u32 },
    /// `ngram_max` is zero.
    NgramMaxZero,
}

impl std::fmt::Display for HashingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BucketsOutOfRange { buckets } => write!(
                f,
                "bucket count {buckets} is outside the range 1 to {MAX_BUCKETS}"
            ),
            Self::NgramMaxZero => write!(f, "the longest n-gram must be at least 1 byte"),
        }
    }
}

impl std::error::Error for HashingError {}

/// One feature row: bucket indices in strictly ascending order with their values.
///
/// No value is zero, and the values of a non-empty row have Euclidean norm 1. Values are
/// `f64` so a solver working in double precision can use them without conversion.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseRow {
    indices: Vec<u32>,
    values: Vec<f64>,
}

impl SparseRow {
    /// Bucket indices, strictly ascending.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The value at each index of [`SparseRow::indices`].
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the row has no entries.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// The entries as `(index, value)` pairs in ascending index order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }
}

/// 64-bit FNV-1a of `bytes` (offset basis `0xcbf29ce484222325`, prime `0x100000001b3`).
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Turns texts into feature rows under fixed, validated settings. The default hasher
/// uses [`HashingSettings::default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CharHasher {
    settings: HashingSettings,
}

impl CharHasher {
    /// A hasher for `settings`, or the reason the settings are not usable.
    pub fn new(settings: HashingSettings) -> Result<Self, HashingError> {
        if settings.buckets == 0 || settings.buckets > MAX_BUCKETS {
            return Err(HashingError::BucketsOutOfRange {
                buckets: settings.buckets,
            });
        }
        if settings.ngram_max == 0 {
            return Err(HashingError::NgramMaxZero);
        }
        Ok(Self { settings })
    }

    /// The settings this hasher was built with.
    pub fn settings(&self) -> HashingSettings {
        self.settings
    }

    /// Number of columns of the feature space: `2 * buckets` with
    /// [`RoutingMode::Isolate`], otherwise `buckets`. Every index of a row is below it.
    pub fn dimension(&self) -> u32 {
        match self.settings.routing {
            RoutingMode::Isolate => self.settings.buckets * 2,
            RoutingMode::Keep | RoutingMode::Strip => self.settings.buckets,
        }
    }

    /// First index of the routing half with [`RoutingMode::Isolate`]: indices at or above
    /// it are routing features, indices below it payload features. `None` in the other
    /// modes, whose rows have a single feature space.
    pub fn routing_offset(&self) -> Option<u32> {
        match self.settings.routing {
            RoutingMode::Isolate => Some(self.settings.buckets),
            RoutingMode::Keep | RoutingMode::Strip => None,
        }
    }

    /// The feature row of `text`.
    pub fn row(&self, text: &str) -> SparseRow {
        let buckets = self.settings.buckets;
        let mut features = Vec::new();
        match self.settings.routing {
            RoutingMode::Keep => self.hash_part(text.as_bytes(), 0, &mut features),
            RoutingMode::Strip => {
                self.hash_part(split_routing(text).payload.as_bytes(), 0, &mut features);
            }
            RoutingMode::Isolate => {
                let split = split_routing(text);
                self.hash_part(split.payload.as_bytes(), 0, &mut features);
                self.hash_part(split.routing.as_bytes(), buckets, &mut features);
            }
        }
        normalized(features)
    }

    /// Appends one `(index, sign)` pair per n-gram of `part`, with `offset` added to
    /// every bucket.
    fn hash_part(&self, part: &[u8], offset: u32, features: &mut Vec<(u32, i8)>) {
        let buckets = u64::from(self.settings.buckets);
        for n in 1..=usize::from(self.settings.ngram_max) {
            if part.len() < n {
                break;
            }
            features.reserve(part.len() - n + 1);
            for gram in part.windows(n) {
                let hash = fnv1a_64(gram);
                // Below `buckets`, which is at most `MAX_BUCKETS`: fits `u32`, and so
                // does the sum with an offset of `buckets`.
                let bucket = ((hash & 0xffff_ffff) % buckets) as u32;
                let sign = if (hash >> 32).count_ones().is_multiple_of(2) {
                    1
                } else {
                    -1
                };
                features.push((offset + bucket, sign));
            }
        }
    }
}

/// Adds up the signs per index, drops sums of zero and scales the rest to norm 1.
fn normalized(mut features: Vec<(u32, i8)>) -> SparseRow {
    // Integer addition is order-independent, so the unstable sort cannot change the sums.
    features.sort_unstable_by_key(|(index, _)| *index);

    let mut indices = Vec::new();
    let mut sums: Vec<i64> = Vec::new();
    let mut squares: u128 = 0;
    let mut position = 0;
    while position < features.len() {
        let index = features[position].0;
        let mut sum: i64 = 0;
        while position < features.len() && features[position].0 == index {
            sum += i64::from(features[position].1);
            position += 1;
        }
        if sum != 0 {
            indices.push(index);
            sums.push(sum);
            squares += u128::from(sum.unsigned_abs()).pow(2);
        }
    }

    let norm = (squares as f64).sqrt();
    let values = sums.into_iter().map(|sum| sum as f64 / norm).collect();
    SparseRow { indices, values }
}
