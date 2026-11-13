//! The baseline scorer: a bad/good ratio from character n-grams, without an encoder.
//!
//! # Model
//!
//! A text is hashed into a sparse feature row ([`crate::hashing`]) and scored by an
//! L2-penalized logistic regression ([`crate::logreg`]). The **ratio** of a text is the
//! logistic function of `row . weights + bias`: a value in `0.0..=1.0`, where higher means
//! more likely bad (requirement 2.2). The baseline never abstains.
//!
//! ## Texts without features
//!
//! A text can hash to an empty row: an empty text, a text whose n-gram signs all cancel,
//! and above all a text without payload when the routing mode is
//! [`RoutingMode::Strip`] (any text without `?` and line feed, for example an ordinary
//! log line). Its ratio is defined as the logistic function of the bias alone. The bias
//! is not penalized, so this is the model's answer to "nothing is known about the text":
//! a constant near the share of bad logs the fit saw, the same for every such text. The
//! value says nothing about the text itself. In a ranking all such texts tie at that
//! constant, above every text the model considers less suspicious than its prior and
//! below every text it considers more suspicious. The scores frame therefore carries the
//! column `no_features`, so a consumer can tell these rows apart; the fit report counts
//! them among the fitted rows ([`BaselineFitReport::no_feature_rows`]). When every fitted
//! row is without features the model is the bias alone and ranks nothing.
//!
//! # Fitting
//!
//! [`BaselineScorer::fit_with`] takes the logs frame and fits on its rows labelled `good`
//! or `bad`, in frame order; rows labelled `unknown` are ignored. The frame holds one row
//! per distinct text, so the counts below are counts of distinct labelled texts.
//!
//! - **Refusal** (requirement 2.3): fewer than 10 labelled rows, or fewer than 5 of either
//!   label, is [`FitError::TooFewLabels`] with the counts found.
//! - **Determinism** (requirement 2.4): equal frames (same rows in the same order) with
//!   equal settings give bit-equal weights, hence equal ratios and the same revision, in
//!   every run and process of one build on one machine. Row order is part of the input.
//! - **Termination**: a fit that stops at the iteration cap is not an error. How the fit
//!   ended, the iterations used and the gradient norm are kept in the
//!   [`BaselineFitReport`], which is saved with the scorer.
//!   [`BaselineScorer::gradient_converged`] says whether the gradient criterion was met.
//!
//! ## Memory
//!
//! Fitting needs all rows at once (requirement 12.5). Before the feature matrix is
//! built, its size is estimated and checked with
//! [`MemoryBudget::require_whole`]; a fit whose estimate exceeds what the limit leaves is
//! refused with the estimate and the limit, and nothing is allocated. The estimate is
//!
//! ```text
//! entries * 12 + (rows + 1) * 8          feature matrix (u32 index + f64 value per entry)
//! + rows                                 labels
//! + 48 * (dimension + 1) * 8             weights and the solver's vectors (L-BFGS history
//!                                        of ten pairs, line search, kept evaluations)
//! + 6 * rows * 8                         per-row vectors of the solver
//! + longest_text_bytes * ngram_max * 40  hashing one text
//! + rows * 16                            the hashes of the fitted texts (below)
//! ```
//!
//! `entries` comes from hashing a sample of up to [`ESTIMATE_SAMPLE_ROWS`] labelled rows
//! spread evenly over the frame: stored entries per text byte in the sample, times the
//! text bytes of all labelled rows, times 1.25, and never more than the number of n-grams
//! the texts can have. With at most that many labelled rows the sample is every row and
//! the count is exact. The matrix is allocated once with exactly the estimated room. If
//! the rows turn out to need more (a sample that is not typical of the frame), the rest
//! is counted exactly and the budget is asked again before the matrix grows.
//!
//! Hashing a row takes memory too, and the sample hashes rows. So the budget is asked
//! twice before the matrix exists: first, before any row is hashed, for every term but
//! the entries (with the longest labelled text, found by a scan that allocates
//! nothing); then for the whole estimate. A fit whose longest text alone cannot be
//! hashed within the limit is refused by the first question, with nothing allocated;
//! the sample hashes one row at a time and so stays within the hashing term.
//! `tests/scorer_memory.rs` counts the heap of such a refusal (a few hundred bytes for
//! a text of 2 MiB, where the sample used to take 64 MiB first).
//!
//! A load takes the files once: the weights are read in pieces of 64 KiB straight into
//! their final form and the revision is hashed over the pieces as they pass, so the
//! weights are never in memory twice ([`BaselineScorer::load_bytes`]; the same test
//! file holds the heap of a load to that figure). Saving and the revision of a fit go
//! over the weights in the same pieces.
//!
//! Measured on HTTP CSIC 2010 (25,608 rows, release build): keep is estimated at
//! 161.6 MiB, the heap grows by 154.0 MiB at its peak and the working set of the process
//! by 128 MiB; isolate is estimated at 193.6 MiB against 179.5 MiB of heap and 152 MiB
//! of working set. `tests/scorer_csic.rs` asserts on every run that the heap a fit
//! takes stays within its estimate.
//!
//! # Saved form
//!
//! [`BaselineScorer::save`] writes `models/baselines/<name>-<rev12>/` with three files
//! (two for a scorer that does not know the texts it was fitted on, see below):
//!
//! - `weights.bin`: the weights as IEEE 754 doubles, little-endian, 8 bytes each, in
//!   column order; `dimension * 8` bytes, where the dimension is
//!   [`CharHasher::dimension`] (`buckets`, or `2 * buckets` with
//!   [`RoutingMode::Isolate`]).
//! - `fitted-texts.bin`: one hash per distinct text the scorer was fitted on, 16 bytes
//!   each, sorted ascending as byte strings, each once (see "Fitted texts").
//! - `baseline.json`: format number, kind, revision, tool version, the settings, the
//!   dimension and the bias, the fit report, the fitting inputs (source path, content
//!   SHA-256, record and distinct-text counts), and under `fitted_texts` the name of
//!   the hash file, the hash rule, the number of hashes and the SHA-256 of the file.
//!   Every floating-point number is written
//!   as `{"value": 1.0, "bits": "3ff0000000000000"}`; `bits` is the exact bit pattern in
//!   hexadecimal and is what is read back, `value` is for the reader.
//!
//! The directory is written under a temporary name beside its final place and renamed
//! when complete, so a scorer directory is either complete or absent.
//!
//! Saving a scorer whose directory already exists with the same revision does not write
//! the model again. The fitting inputs are not part of the revision, so they can differ
//! (another file with the same distinct labelled texts in the same order gives the same
//! model). The stored inputs then become the union of the stored ones and the new
//! ones, stored first, deduplicated by content hash, and `baseline.json` is replaced
//! through a temporary file in the same directory. The stored fit report is kept. A
//! saved scorer therefore always names every input any saved copy of it was fitted on
//! (requirement 11.1).
//!
//! ## Fitted texts
//!
//! The whole-input hashes above tell an identical file apart and nothing else: the same
//! logs reordered, re-exported or cut into a subset have other bytes. So a fit also
//! keeps a hash of every text it was fitted on (the rows labelled good or bad; rows
//! labelled unknown are not fitted on), and calibration leaves out every calibration
//! log whose text is among them (requirement 11.1, owner decision of 2026-11-12,
//! task 1.11). The hash of a text is the first 16 bytes of the SHA-256 of its bytes
//! ([`text_hash`], rule name `sha256-128`); a text is compared as its bytes, nothing is
//! trimmed or folded. Two different texts share a hash with a probability of about
//! `n^2 / 2^129` for `n` texts, which no dataset reaches; the effect would be one
//! calibration log left out without need.
//!
//! The hashes are not part of the revision, like the fitting inputs: they describe
//! what the model was fitted on, not the model. The revision of a model is the same
//! with and without them, so calibrations made for a scorer stay valid when it gains
//! its hashes. `baseline.json` carries the SHA-256 of `fitted-texts.bin`, and a load
//! refuses a hash file that does not match it, has another length than its count, or
//! is not sorted; `baseline.json` itself is not covered by the revision, as before.
//!
//! **Format numbers.** Format 1 (tasks 1.7 to 1.10) has no `fitted_texts` entry and no
//! hash file. Format 2 (task 1.11 on) is format 1 plus the optional entry. Both load;
//! a scorer of format 1 has [`BaselineScorer::has_fitted_text_hashes`] false and
//! [`BaselineScorer::was_fitted_on_text`] `None`, and calibration falls back to the
//! whole-input check for it. Everything is written as format 2. A scorer without
//! hashes that is saved again stays without them (format 2 without the entry).
//!
//! On a save over a directory of the same revision the hashes are merged like the
//! fitting inputs: a directory without hashes gains those of the scorer being saved
//! (equal revisions mean bit-equal weights from equal settings, which this module
//! already takes to mean the same fit), and two sets become their union. The hash file
//! is replaced first and `baseline.json` second, each through a temporary file; when
//! the process dies between the two after a union changed an existing hash file, the
//! directory no longer loads (the file does not match the stated SHA-256) and the
//! scorer has to be fitted again. A scorer without hashes saved over a directory that
//! has them changes nothing, unless it names a fitting input the directory did not
//! have.
//!
//! **Complete and incomplete sets.** The entry carries `complete`. A set is complete
//! while it covers every fitting input the scorer names, and a fit's own set always
//! is. It becomes incomplete, for good, when hashes join fitting inputs that were saved
//! without hashes: a directory of format 1 gains the hashes of a new fit, but the
//! texts of its earlier inputs are not known, unless the new fit was fitted on every
//! one of those inputs (the same content hash means the same texts) and the directory
//! names at least one. The same holds the other way round, when a scorer without
//! hashes adds a fitting input to a directory that has them, and a union is complete
//! only when both sets are. [`BaselineScorer::fitted_text_hashes_complete`] says
//! which; for an incomplete set [`BaselineScorer::was_fitted_on_text`] still finds the
//! texts it has, but its no is not a proof, and calibration reports the check per log
//! as partial. Saving the refitted scorer under a new name gives a complete set.
//!
//! **What the check per log does and does not catch.** The text is hashed as the
//! adapter read it, before any routing rule, so the hash is the same under keep, strip
//! and isolate. A text that differs in one byte (a trailing space, another case,
//! another encoding of the same request) is another text, by design. Under strip a
//! calibration log with the same payload on another path is not left out, although the
//! model sees the two alike. A text that occurred with both labels is unlabelled, is
//! not fitted on and is not hashed.
//!
//! A loaded scorer keeps the hashes for as long as it lives, 16 bytes per fitted text
//! ([`BaselineScorer::held_bytes`]); a fit counts them in its estimate.
//!
//! ## Revision
//!
//! The revision is the first 40 hexadecimal characters of the SHA-256 of these bytes, in
//! this order (all integers and bit patterns little-endian):
//!
//! ```text
//! "l2d-baseline-scorer/1\n"     22 bytes
//! buckets                       u32
//! ngram_max                     u8
//! routing                       u8: 0 keep, 1 strip, 2 isolate
//! l2                            f64 bit pattern
//! max_iter                      u32
//! tolerance                     f64 bit pattern
//! dimension                     u64
//! weights                       dimension f64 bit patterns (the bytes of weights.bin)
//! bias                          f64 bit pattern
//! ```
//!
//! That is every setting that can change the model, and the model. The fit report and
//! the fitting inputs are not part of the revision. [`BaselineScorer::load`] computes the
//! revision again from the files and refuses a directory whose stored revision or
//! directory name does not match.
//!
//! # Untrusted data
//!
//! Log text is hashed as bytes only. The scorer name is the only caller-supplied part of
//! a path and is restricted to letters, digits, `.`, `_` and `-`. A stored source path
//! is information for the reader and is never opened. The file name under
//! `fitted_texts` must be `fitted-texts.bin`; any other name is refused and not opened.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use l2d_core::input::content_sha256;
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::types::short_revision;
use l2d_core::{
    InputSummary, L2dError, Label, LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::hashing::{CharHasher, HashingSettings, RoutingMode};
use crate::logreg::{LogReg, LogRegError, LogRegSettings, SparseMatrix, Termination};

/// Fewest labelled rows a fit accepts (requirement 2.3).
pub const MIN_LABELLED: u64 = 10;
/// Fewest rows of each label a fit accepts (requirement 2.3).
pub const MIN_PER_LABEL: u64 = 5;

/// Most labelled rows hashed for the memory estimate of a fit.
pub const ESTIMATE_SAMPLE_ROWS: usize = 1_024;

/// Default iteration cap of a fit: about five times what HTTP CSIC 2010 needs at the
/// default tolerance (see [`DEFAULT_TOLERANCE`]), so the cap is a safety limit and not
/// the usual way a fit ends.
pub const DEFAULT_MAX_ITER: u32 = 1_000;
/// Default bound on the gradient norm of the summed loss.
///
/// Measured on HTTP CSIC 2010 (25,608 distinct texts, 65,536 buckets, `l2` 1.0, release
/// build, `tests/scorer_csic.rs`):
///
/// | Tolerance | keep | strip | isolate | Largest ratio change against `1e-6` |
/// |-----------|------|-------|---------|--------------------------------------|
/// | `1e-1` | 99 iterations | 81 | 84 | 6.1e-4 |
/// | `1e-2` | 141 | 114 | 122 | 6.4e-5 |
/// | `1e-3` | 178 | 149 | 154 | 5.7e-6 |
/// | `1e-4` | 221 | 202 | 201 | 3.0e-7 |
/// | `1e-5`, `1e-6` | 236 | 206 | 219 | reference |
///
/// At `1e-3` every mode ends by the gradient criterion (norms 8.1e-4 to 1.0e-3). The
/// smallest gradient norm double precision reaches on this data is between 3.6e-5 and
/// 1.2e-4: at `1e-5` no mode meets the gradient criterion (keep and strip stop because
/// the objective no longer changes, isolate stops in the line search at 1.2e-4), and
/// `1e-4` sits on that floor. `1e-3` is ten times above the floor and changes no ratio
/// by more than 6e-6, far below what a ranking or a `f32` ratio can show. A fit of the
/// whole dataset takes about 5 to 15 s in a release build and about 95 s in a debug
/// build.
///
/// The bound is absolute on the summed loss, so the floor rises with the number of
/// rows; for much larger inputs check [`BaselineScorer::gradient_converged`].
pub const DEFAULT_TOLERANCE: f64 = 1e-3;
/// Value of the `scorer` column of the scores frame.
pub const SCORER_KIND: &str = "baseline";
/// Scores frame column: the log text (String).
pub const SCORES_TEXT: &str = "text";
/// Scores frame column: the scorer kind (String), always [`SCORER_KIND`].
pub const SCORES_SCORER: &str = "scorer";
/// Scores frame column: the bad/good ratio (Float32), never null for this scorer.
pub const SCORES_RATIO: &str = "ratio";
/// Scores frame column: whether the scorer abstained (Boolean), always `false`.
pub const SCORES_ABSTAINED: &str = "abstained";
/// Scores frame column owned by this scorer (Boolean): the text has no features, and its
/// ratio is the constant of the module documentation.
pub const SCORES_NO_FEATURES: &str = "no_features";

/// Name of the settings and report file in a scorer directory.
pub const MANIFEST_FILE: &str = "baseline.json";
/// Name of the weights file in a scorer directory.
pub const WEIGHTS_FILE: &str = "weights.bin";
/// Name of the file with the hashes of the fitted texts in a scorer directory.
pub const FITTED_TEXTS_FILE: &str = "fitted-texts.bin";
/// Memory of a load beside the files: the piece of the weights being read (64 KiB) and
/// what the description of the scorer takes. See [`BaselineScorer::load_bytes`].
pub const LOAD_FIXED_BYTES: u64 = 128 * 1024;
/// Bytes of the hash of one fitted text.
pub const TEXT_HASH_BYTES: usize = 16;
/// Name of the hash rule in `baseline.json`: the first 128 bits of the SHA-256.
pub const TEXT_HASH_NAME: &str = "sha256-128";

/// Longest accepted scorer name, in bytes.
pub const MAX_NAME_BYTES: usize = 64;

/// Characters of the revision that are kept.
const REVISION_CHARS: usize = 40;
/// First bytes of the hashed content; changes when the hashed layout changes.
const REVISION_DOMAIN: &[u8] = b"l2d-baseline-scorer/1\n";
/// Format number of `baseline.json` as it is written: format 1 and the optional entry
/// `fitted_texts`.
const FORMAT: u32 = 2;
/// The format of tasks 1.7 to 1.10, which is still read: no `fitted_texts`.
const FORMAT_WITHOUT_TEXT_HASHES: u32 = 1;

const ENTRY_BYTES: u64 = 12;
const SOLVER_VECTORS: u64 = 48;
const SOLVER_ROW_VECTORS: u64 = 6;
const HASHING_BYTES_PER_NGRAM: u64 = 40;
const ESTIMATE_MARGIN: f64 = 1.25;
const OPERATION: &str = "fitting the baseline scorer";

const CHECKED: &str = "the logs frame was checked on construction";

/// Everything that decides the model. All of it is saved and is part of the revision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineSettings {
    /// Hash buckets per feature space.
    pub buckets: u32,
    /// Longest n-gram in bytes.
    pub ngram_max: u8,
    /// Treatment of the routing part of a text (requirement 2.6).
    pub routing: RoutingMode,
    /// Strength of the L2 penalty on the summed loss; the bias is not penalized.
    pub l2: f64,
    /// Largest number of solver iterations.
    pub max_iter: u32,
    /// The fit has converged when the gradient norm of the summed loss is below this. It
    /// is an absolute bound, so it is stricter the more rows there are.
    pub tolerance: f64,
}

impl Default for BaselineSettings {
    fn default() -> Self {
        let hashing = HashingSettings::default();
        Self {
            buckets: hashing.buckets,
            ngram_max: hashing.ngram_max,
            routing: hashing.routing,
            l2: 1.0,
            max_iter: DEFAULT_MAX_ITER,
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

impl BaselineSettings {
    /// The feature rule of these settings.
    pub fn hashing(&self) -> HashingSettings {
        HashingSettings {
            buckets: self.buckets,
            ngram_max: self.ngram_max,
            routing: self.routing,
        }
    }

    /// The solver settings.
    pub fn logreg(&self) -> LogRegSettings {
        LogRegSettings {
            l2: self.l2,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
        }
    }

    /// The hasher of these settings, or why the settings are not usable.
    fn hasher(&self) -> Result<CharHasher, String> {
        let hasher = CharHasher::new(self.hashing()).map_err(|error| error.to_string())?;
        if !self.l2.is_finite() || self.l2 < 0.0 {
            return Err(format!(
                "the L2 penalty must be finite and not negative, not {}",
                self.l2
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(format!(
                "the tolerance must be finite and not negative, not {}",
                self.tolerance
            ));
        }
        if self.max_iter == 0 {
            return Err("the iteration cap must be at least 1".into());
        }
        Ok(hasher)
    }
}

/// Why a fit did not produce a scorer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FitError {
    /// Requirement 2.3: too few labelled logs, with the counts of distinct texts found.
    #[error(
        "fitting the baseline scorer needs at least {MIN_LABELLED} labelled logs and at least \
         {MIN_PER_LABEL} of each label; found {} labelled ({good} good, {bad} bad)",
        .good + .bad
    )]
    TooFewLabels { good: u64, bad: u64 },
    /// A setting is outside its range.
    #[error("unusable baseline settings: {detail}")]
    InvalidSettings { detail: String },
    /// The memory limit does not allow the fit (requirement 12.5), or the free memory
    /// could not be read. Carries the refusal of the memory budget unchanged.
    #[error("{0}")]
    Memory(L2dError),
    /// The solver or the feature matrix failed.
    #[error("the baseline fit failed: {0}")]
    Solver(LogRegError),
}

impl From<FitError> for L2dError {
    fn from(error: FitError) -> Self {
        match error {
            FitError::TooFewLabels { .. } => L2dError::Refused {
                reason: error.to_string(),
            },
            FitError::InvalidSettings { .. } => L2dError::InvalidInput {
                detail: error.to_string(),
            },
            FitError::Memory(inner) => inner,
            FitError::Solver(_) => L2dError::Runtime {
                detail: error.to_string(),
            },
        }
    }
}

/// How the fit of a scorer went. Saved with the scorer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineFitReport {
    /// Why the solver stopped. [`Termination::Converged`] also covers a stop because the
    /// objective no longer changed; compare `gradient_norm` with the tolerance when the
    /// gradient criterion matters ([`BaselineScorer::gradient_converged`]).
    pub termination: Termination,
    /// Solver iterations performed.
    pub iterations: u64,
    /// Value of the objective at the fitted model.
    pub objective: f64,
    /// Gradient norm of the objective at the fitted model.
    pub gradient_norm: f64,
    /// Labelled rows (distinct texts) fitted: `good + bad`.
    pub rows: u64,
    /// Rows labelled good.
    pub good: u64,
    /// Rows labelled bad.
    pub bad: u64,
    /// Rows labelled unknown, which were ignored.
    pub unknown_ignored: u64,
    /// Fitted rows whose text has no features (see the module documentation).
    pub no_feature_rows: u64,
    /// Entries of the feature matrix.
    pub stored_entries: u64,
    /// The memory estimate the fit was checked against, in bytes.
    pub estimated_bytes: u64,
}

/// One input a scorer was fitted on. Calibration refuses an input whose content hash is
/// among a scorer's fitting inputs (requirement 11.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FittingInput {
    /// Where the input was read from. Information only; never opened.
    pub source_path: String,
    /// SHA-256 of the input's bytes, as in [`InputSummary::content_sha256`].
    pub content_sha256: String,
    /// Records read from the input.
    pub records: u64,
    /// Distinct texts of the input.
    pub distinct_texts: u64,
}

impl From<&InputSummary> for FittingInput {
    fn from(summary: &InputSummary) -> Self {
        Self {
            source_path: summary.source_path.to_string_lossy().into_owned(),
            content_sha256: summary.content_sha256.clone(),
            records: summary.records,
            distinct_texts: summary.distinct_texts,
        }
    }
}

/// The hash of a text for the held-out check per log: the first [`TEXT_HASH_BYTES`]
/// bytes of the SHA-256 of its bytes (see "Fitted texts" in the module documentation).
pub fn text_hash(text: &str) -> [u8; TEXT_HASH_BYTES] {
    let digest = Sha256::digest(text.as_bytes());
    let mut hash = [0_u8; TEXT_HASH_BYTES];
    hash.copy_from_slice(&digest[..TEXT_HASH_BYTES]);
    hash
}

/// The hashes of the texts a scorer was fitted on: [`TEXT_HASH_BYTES`] bytes each, end
/// to end, sorted ascending, each once. This is also the content of
/// [`FITTED_TEXTS_FILE`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct FittedTexts {
    bytes: Vec<u8>,
    /// Whether the hashes cover every fitting input the scorer names. `false` after
    /// hashes were merged into a directory whose earlier fitting inputs came without
    /// hashes (module documentation, "Fitted texts").
    complete: bool,
}

impl FittedTexts {
    fn from_hashes(mut hashes: Vec<[u8; TEXT_HASH_BYTES]>, complete: bool) -> Self {
        hashes.sort_unstable();
        hashes.dedup();
        Self {
            bytes: hashes.into_flattened(),
            complete,
        }
    }

    /// The hashes of a file, or what is wrong with it.
    fn from_file_bytes(bytes: Vec<u8>, complete: bool) -> Result<Self, String> {
        let (hashes, rest) = bytes.as_chunks::<TEXT_HASH_BYTES>();
        if !rest.is_empty() {
            return Err(format!(
                "its {} bytes are not a whole number of {TEXT_HASH_BYTES}-byte hashes",
                bytes.len()
            ));
        }
        if !hashes.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err("its hashes are not sorted, or one occurs twice".into());
        }
        Ok(Self { bytes, complete })
    }

    fn hashes(&self) -> &[[u8; TEXT_HASH_BYTES]] {
        self.bytes.as_chunks::<TEXT_HASH_BYTES>().0
    }

    fn len(&self) -> u64 {
        self.hashes().len() as u64
    }

    fn contains(&self, hash: &[u8; TEXT_HASH_BYTES]) -> bool {
        self.hashes().binary_search(hash).is_ok()
    }

    fn union(&self, other: &Self) -> Self {
        let mut hashes = self.hashes().to_vec();
        hashes.extend_from_slice(other.hashes());
        Self::from_hashes(hashes, self.complete && other.complete)
    }

    fn file(&self) -> FittedTextsFile {
        FittedTextsFile {
            file: FITTED_TEXTS_FILE.into(),
            hash: TEXT_HASH_NAME.into(),
            count: self.len(),
            sha256: content_sha256(&self.bytes),
            complete: self.complete,
        }
    }
}

/// A fitted baseline scorer.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineScorer {
    settings: BaselineSettings,
    hasher: CharHasher,
    model: LogReg,
    revision: String,
    report: BaselineFitReport,
    fitting_inputs: Vec<FittingInput>,
    /// `None` for a scorer saved before task 1.11 (module documentation, "Fitted
    /// texts").
    fitted_texts: Option<FittedTexts>,
}

/// The labelled rows of a logs frame in frame order: the text, and whether it is bad.
fn labelled_rows(logs: &LogsFrame) -> impl Iterator<Item = (&str, bool)> + '_ {
    let frame = logs.frame();
    let texts = frame
        .column(LogsFrame::TEXT)
        .and_then(Column::str)
        .expect(CHECKED);
    let labels = frame
        .column(LogsFrame::LABEL)
        .and_then(Column::str)
        .expect(CHECKED);
    texts.iter().zip(labels.iter()).filter_map(|(text, label)| {
        match label.and_then(Label::from_name).expect(CHECKED) {
            Label::Good => Some((text.expect(CHECKED), false)),
            Label::Bad => Some((text.expect(CHECKED), true)),
            Label::Unknown => None,
        }
    })
}

/// Most entries a text of `bytes` bytes can have: its number of n-grams, and never more
/// than the dimension.
fn entry_bound(bytes: usize, ngram_max: u8, dimension: u32) -> u64 {
    let mut grams: u64 = 0;
    for n in 1..=usize::from(ngram_max) {
        if bytes < n {
            break;
        }
        grams += (bytes - n + 1) as u64;
    }
    grams.min(u64::from(dimension))
}

/// The memory estimate of a fit (see the module documentation).
#[derive(Debug, Clone, Copy)]
struct FitEstimate {
    /// Entries the feature matrix gets room for.
    entries: u64,
    /// Bytes checked against the memory budget.
    bytes: u64,
    longest_text: u64,
}

impl FitEstimate {
    fn bytes_for(entries: u64, rows: u64, hasher: &CharHasher, longest_text: u64) -> u64 {
        let matrix = entries
            .saturating_mul(ENTRY_BYTES)
            .saturating_add((rows + 1) * 8);
        let solver = SOLVER_VECTORS
            .saturating_mul(u64::from(hasher.dimension()) + 1)
            .saturating_mul(8)
            .saturating_add(SOLVER_ROW_VECTORS.saturating_mul(rows).saturating_mul(8));
        let hashing = longest_text
            .saturating_mul(u64::from(hasher.settings().ngram_max))
            .saturating_mul(HASHING_BYTES_PER_NGRAM);
        matrix
            .saturating_add(rows)
            .saturating_add(solver)
            .saturating_add(hashing)
            .saturating_add(rows.saturating_mul(TEXT_HASH_BYTES as u64))
    }

    /// Estimates the fit of the `rows` labelled rows of `logs` from a sample of them.
    fn sample(logs: &LogsFrame, hasher: &CharHasher, rows: usize) -> Self {
        let settings = hasher.settings();
        let sample = rows.min(ESTIMATE_SAMPLE_ROWS);
        let mut text_bytes: u64 = 0;
        let mut longest_text: u64 = 0;
        let mut bound: u64 = 0;
        let mut sample_bytes: u64 = 0;
        let mut sample_entries: u64 = 0;
        let mut taken = 0;
        for (index, (text, _)) in labelled_rows(logs).enumerate() {
            let bytes = text.len() as u64;
            text_bytes += bytes;
            longest_text = longest_text.max(bytes);
            bound = bound.saturating_add(entry_bound(
                text.len(),
                settings.ngram_max,
                hasher.dimension(),
            ));
            // Sample row `taken` is the labelled row `taken * rows / sample`.
            if taken < sample && index == taken * rows / sample {
                sample_entries += hasher.row(text).len() as u64;
                sample_bytes += bytes;
                taken += 1;
            }
        }
        let entries = if sample == rows {
            sample_entries
        } else if sample_bytes == 0 {
            bound
        } else {
            let per_byte = sample_entries as f64 / sample_bytes as f64;
            ((per_byte * text_bytes as f64 * ESTIMATE_MARGIN).ceil() as u64).min(bound)
        };
        Self {
            entries,
            bytes: Self::bytes_for(entries, rows as u64, hasher, longest_text),
            longest_text,
        }
    }

    /// The estimate after the matrix has to grow to `entries`: the new size, plus the
    /// old entries, which exist a second time while they are moved.
    fn regrown(&self, entries: u64, rows: u64, hasher: &CharHasher) -> Self {
        Self {
            entries,
            bytes: Self::bytes_for(entries, rows, hasher, self.longest_text)
                .saturating_add(self.entries.saturating_mul(ENTRY_BYTES)),
            longest_text: self.longest_text,
        }
    }
}

/// Computes the revision of a model piece by piece, so that the weights never have to
/// exist a second time as bytes: see "Revision" in the module documentation.
struct RevisionHasher(Sha256);

impl RevisionHasher {
    /// Starts with everything that comes before the weights.
    fn new(settings: &BaselineSettings, dimension: u64) -> Self {
        let mut hash = Sha256::new();
        hash.update(REVISION_DOMAIN);
        hash.update(settings.buckets.to_le_bytes());
        hash.update([settings.ngram_max, routing_code(settings.routing)]);
        hash.update(settings.l2.to_bits().to_le_bytes());
        hash.update(settings.max_iter.to_le_bytes());
        hash.update(settings.tolerance.to_bits().to_le_bytes());
        hash.update(dimension.to_le_bytes());
        Self(hash)
    }

    /// The next bytes of the weights, as they are in `weights.bin`.
    fn weight_bytes(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    /// Ends with the bias and returns the revision.
    fn finish(mut self, bias: f64) -> String {
        self.0.update(bias.to_bits().to_le_bytes());
        let mut revision: String = self
            .0
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        revision.truncate(REVISION_CHARS);
        revision
    }
}

/// How many weights are turned into bytes, or read, at a time: 64 KiB.
const WEIGHTS_AT_A_TIME: usize = 8 * 1024;

/// Hands the bytes of `weights` (little-endian doubles, as in `weights.bin`) to `take`
/// in pieces of at most [`WEIGHTS_AT_A_TIME`] weights.
fn weight_pieces<E>(
    weights: &[f64],
    mut take: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    let mut piece = [0_u8; WEIGHTS_AT_A_TIME * 8];
    for weights in weights.chunks(WEIGHTS_AT_A_TIME) {
        for (bytes, weight) in piece.as_chunks_mut::<8>().0.iter_mut().zip(weights) {
            *bytes = weight.to_bits().to_le_bytes();
        }
        take(&piece[..weights.len() * 8])?;
    }
    Ok(())
}

/// The revision of a model.
fn revision_of(settings: &BaselineSettings, weights: &[f64], bias: f64) -> String {
    let mut hash = RevisionHasher::new(settings, weights.len() as u64);
    let _ = weight_pieces(weights, |bytes| {
        hash.weight_bytes(bytes);
        Ok::<(), std::convert::Infallible>(())
    });
    hash.finish(bias)
}
fn routing_code(routing: RoutingMode) -> u8 {
    match routing {
        RoutingMode::Keep => 0,
        RoutingMode::Strip => 1,
        RoutingMode::Isolate => 2,
    }
}

fn routing_name(routing: RoutingMode) -> &'static str {
    match routing {
        RoutingMode::Keep => "keep",
        RoutingMode::Strip => "strip",
        RoutingMode::Isolate => "isolate",
    }
}

fn routing_from_name(name: &str) -> Option<RoutingMode> {
    match name {
        "keep" => Some(RoutingMode::Keep),
        "strip" => Some(RoutingMode::Strip),
        "isolate" => Some(RoutingMode::Isolate),
        _ => None,
    }
}

fn termination_name(termination: Termination) -> &'static str {
    match termination {
        Termination::Converged => "converged",
        Termination::IterationCap => "iteration_cap",
        Termination::LineSearchStopped => "line_search_stopped",
    }
}

fn termination_from_name(name: &str) -> Option<Termination> {
    match name {
        "converged" => Some(Termination::Converged),
        "iteration_cap" => Some(Termination::IterationCap),
        "line_search_stopped" => Some(Termination::LineSearchStopped),
        _ => None,
    }
}

/// Writes `weights.bin`: the weights piece by piece, never as a second copy in memory.
fn write_weights(path: &Path, weights: &[f64]) -> Result<(), L2dError> {
    let failed = |error: std::io::Error| io_failure("writing", path, &error);
    let mut file = fs::File::create(path).map_err(failed)?;
    weight_pieces(weights, |bytes| file.write_all(bytes)).map_err(failed)?;
    file.sync_all().map_err(failed)
}

/// Reads `weights.bin` of `dimension` weights straight into their final form, in
/// pieces of [`WEIGHTS_AT_A_TIME`] weights that are also handed to `hash`. A file of
/// another length is refused.
fn read_weights(
    dir: &Path,
    dimension: usize,
    hash: &mut RevisionHasher,
) -> Result<Vec<f64>, L2dError> {
    let path = dir.join(WEIGHTS_FILE);
    let mut file = fs::File::open(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            L2dError::Missing {
                item: format!("baseline scorer file {WEIGHTS_FILE}"),
                expected_at: path.clone(),
            }
        } else {
            io_failure("reading", &path, &error)
        }
    })?;
    let wrong_length = || {
        not_a_scorer(
            dir,
            format!(
                "{WEIGHTS_FILE} does not hold exactly the {} bytes of dimension {dimension}",
                dimension * 8
            ),
        )
    };
    let failed = |error: std::io::Error| match error.kind() {
        std::io::ErrorKind::UnexpectedEof => wrong_length(),
        _ => io_failure("reading", &path, &error),
    };
    let mut weights = Vec::with_capacity(dimension);
    let mut piece = [0_u8; WEIGHTS_AT_A_TIME * 8];
    while weights.len() < dimension {
        let bytes = &mut piece[..(dimension - weights.len()).min(WEIGHTS_AT_A_TIME) * 8];
        file.read_exact(bytes).map_err(failed)?;
        hash.weight_bytes(bytes);
        weights.extend(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|bits| f64::from_bits(u64::from_le_bytes(*bits))),
        );
    }
    match file.read(&mut piece[..1]) {
        Ok(0) => Ok(weights),
        Ok(_) => Err(wrong_length()),
        Err(error) => Err(io_failure("reading", &path, &error)),
    }
}

/// A floating-point number in `baseline.json`: readable, and exact.
#[derive(Debug, Serialize, Deserialize)]
struct Exact {
    /// For the reader; absent when the number is not finite.
    value: Option<f64>,
    /// The bit pattern, 16 hexadecimal digits. This is what is read back.
    bits: String,
}

impl Exact {
    fn of(number: f64) -> Self {
        Self {
            value: number.is_finite().then_some(number),
            bits: format!("{:016x}", number.to_bits()),
        }
    }

    fn get(&self) -> Option<f64> {
        if self.bits.len() != 16 {
            return None;
        }
        u64::from_str_radix(&self.bits, 16).ok().map(f64::from_bits)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SettingsFile {
    buckets: u32,
    ngram_max: u8,
    routing: String,
    l2: Exact,
    max_iter: u32,
    tolerance: Exact,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelFile {
    dimension: u64,
    bias: Exact,
}

#[derive(Debug, Serialize, Deserialize)]
struct FitFile {
    termination: String,
    iterations: u64,
    objective: Exact,
    gradient_norm: Exact,
    rows: u64,
    good: u64,
    bad: u64,
    unknown_ignored: u64,
    no_feature_rows: u64,
    stored_entries: u64,
    estimated_bytes: u64,
}

/// The content of `baseline.json`.
#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    format: u32,
    kind: String,
    revision: String,
    tool_version: String,
    settings: SettingsFile,
    model: ModelFile,
    fit: FitFile,
    fitting_inputs: Vec<FittingInput>,
    /// Absent in format 1 and for a scorer that does not know its fitted texts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fitted_texts: Option<FittedTextsFile>,
}

/// The entry `fitted_texts` of `baseline.json`.
#[derive(Debug, Serialize, Deserialize)]
struct FittedTextsFile {
    /// Always [`FITTED_TEXTS_FILE`]; any other name is refused, never opened.
    file: String,
    /// Always [`TEXT_HASH_NAME`].
    hash: String,
    /// The number of hashes in the file.
    count: u64,
    /// SHA-256 of the file's bytes.
    sha256: String,
    /// Whether the hashes cover every fitting input named in `fitting_inputs`; absent
    /// means yes.
    #[serde(default = "complete_by_default")]
    complete: bool,
}

fn complete_by_default() -> bool {
    true
}

/// Checks a scorer name: 1 to [`MAX_NAME_BYTES`] characters out of letters, digits, `.`,
/// `_` and `-`, not starting with a dot. That excludes path separators, `..`, drive
/// prefixes and anything a shell or the file system treats specially.
pub fn validate_name(name: &str) -> Result<(), L2dError> {
    let usable = !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if usable {
        Ok(())
    } else {
        Err(L2dError::InvalidInput {
            detail: format!(
                "the scorer name must be 1 to {MAX_NAME_BYTES} letters, digits, '.', '_' or \
                 '-' and must not start with a dot; got a name of {} bytes that is not",
                name.len()
            ),
        })
    }
}

fn io_failure(doing: &str, path: &Path, error: &std::io::Error) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

fn not_a_scorer(dir: &Path, why: impl std::fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!(
            "{} does not hold a usable baseline scorer: {why}",
            dir.display()
        ),
    }
}

/// Reads a file of a scorer directory; a file that is not there is
/// [`L2dError::Missing`].
fn read_scorer_file(dir: &Path, file: &str) -> Result<Vec<u8>, L2dError> {
    let path = dir.join(file);
    fs::read(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            L2dError::Missing {
                item: format!("baseline scorer file {file}"),
                expected_at: path.clone(),
            }
        } else {
            io_failure("reading", &path, &error)
        }
    })
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), L2dError> {
    let failed = |error: std::io::Error| io_failure("writing", path, &error);
    let mut file = fs::File::create(path).map_err(failed)?;
    file.write_all(bytes).map_err(failed)?;
    file.sync_all().map_err(failed)
}

/// Makes temporary directory names of one process distinct.
static SAVES: AtomicU64 = AtomicU64::new(0);

impl BaselineScorer {
    /// Fits a scorer under a memory budget observed now with the default settings (half
    /// of the free physical memory). See [`BaselineScorer::fit_with`].
    pub fn fit(
        logs: &LogsFrame,
        inputs: &[InputSummary],
        settings: &BaselineSettings,
    ) -> Result<Self, FitError> {
        let budget = MemoryBudget::observe(&MemorySettings::default()).map_err(FitError::Memory)?;
        Self::fit_with(logs, inputs, settings, &budget)
    }

    /// Fits a scorer on the rows of `logs` labelled good or bad, in frame order; rows
    /// labelled unknown are ignored. `inputs` are the summaries of the sources `logs`
    /// was read from; their content hashes are kept with the scorer.
    ///
    /// The fit is refused before the feature matrix is allocated when its estimated
    /// memory exceeds what `budget` leaves. See the module documentation for the
    /// refusal rule, determinism and the memory estimate.
    pub fn fit_with(
        logs: &LogsFrame,
        inputs: &[InputSummary],
        settings: &BaselineSettings,
        budget: &MemoryBudget,
    ) -> Result<Self, FitError> {
        let hasher = settings
            .hasher()
            .map_err(|detail| FitError::InvalidSettings { detail })?;

        let bad = labelled_rows(logs).filter(|(_, bad)| *bad).count() as u64;
        let good = labelled_rows(logs).count() as u64 - bad;
        if good + bad < MIN_LABELLED || good < MIN_PER_LABEL || bad < MIN_PER_LABEL {
            return Err(FitError::TooFewLabels { good, bad });
        }
        let rows = (good + bad) as usize;

        // Before a row is hashed: what the fit needs whatever its rows hash to (hashing
        // the longest text, the solver's vectors, the per-row vectors, the hashes of
        // the texts). A fit that cannot have even that is refused here, with nothing
        // allocated (requirement 12.5). The sample below hashes one row at a time, so
        // it stays within the hashing term that was just asked for.
        let longest_text = labelled_rows(logs)
            .map(|(text, _)| text.len() as u64)
            .max()
            .unwrap_or(0);
        budget
            .require_whole(
                FitEstimate::bytes_for(0, rows as u64, &hasher, longest_text),
                OPERATION,
            )
            .map_err(FitError::Memory)?;

        let mut estimate = FitEstimate::sample(logs, &hasher, rows);
        budget
            .require_whole(estimate.bytes, OPERATION)
            .map_err(FitError::Memory)?;

        let mut matrix =
            SparseMatrix::with_capacity(hasher.dimension(), rows, estimate.entries as usize);
        let mut labels = Vec::with_capacity(rows);
        let mut text_hashes = Vec::with_capacity(rows);
        let mut no_feature_rows = 0;
        for (index, (text, bad)) in labelled_rows(logs).enumerate() {
            text_hashes.push(text_hash(text));
            let row = hasher.row(text);
            if (matrix.stored() + row.len()) as u64 > estimate.entries {
                // The sample was not typical. Count the rest exactly and ask again.
                let rest: usize = labelled_rows(logs)
                    .skip(index + 1)
                    .map(|(text, _)| hasher.row(text).len())
                    .sum();
                let needed = (matrix.stored() + row.len() + rest) as u64;
                let regrown = estimate.regrown(needed, rows as u64, &hasher);
                budget
                    .require_whole(regrown.bytes, OPERATION)
                    .map_err(FitError::Memory)?;
                matrix.reserve_entries(row.len() + rest);
                estimate = regrown;
            }
            no_feature_rows += u64::from(row.is_empty());
            matrix
                .push_row(row.indices(), row.values())
                .map_err(FitError::Solver)?;
            labels.push(bad);
        }

        let stored_entries = matrix.stored() as u64;
        let (model, fit) =
            LogReg::fit_matrix(&matrix, &labels, &settings.logreg()).map_err(FitError::Solver)?;
        drop(matrix);

        let report = BaselineFitReport {
            termination: fit.termination,
            iterations: fit.iterations,
            objective: fit.objective,
            gradient_norm: fit.gradient_norm,
            rows: good + bad,
            good,
            bad,
            unknown_ignored: logs.len() as u64 - good - bad,
            no_feature_rows,
            stored_entries,
            estimated_bytes: estimate.bytes,
        };
        let revision = revision_of(settings, model.weights(), model.bias());
        Ok(Self {
            settings: *settings,
            hasher,
            model,
            revision,
            report,
            fitting_inputs: inputs.iter().map(FittingInput::from).collect(),
            fitted_texts: Some(FittedTexts::from_hashes(text_hashes, true)),
        })
    }

    /// The settings the scorer was fitted with.
    pub fn settings(&self) -> &BaselineSettings {
        &self.settings
    }

    /// The revision: 40 hexadecimal characters (see the module documentation).
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// How the fit went.
    pub fn fit_report(&self) -> &BaselineFitReport {
        &self.report
    }

    /// Whether the fit met the gradient criterion: the gradient norm at the fitted model
    /// is not above the tolerance of the settings. `false` for a fit that stopped at the
    /// iteration cap, in the line search, or because the objective no longer changed
    /// before the gradient was that small.
    pub fn gradient_converged(&self) -> bool {
        self.report.gradient_norm <= self.settings.tolerance
    }

    /// The inputs the scorer was fitted on.
    pub fn fitting_inputs(&self) -> &[FittingInput] {
        &self.fitting_inputs
    }

    /// Whether the scorer names at least one fitting input. A scorer fitted without
    /// input summaries has unknown provenance: [`BaselineScorer::was_fitted_on`] is then
    /// `false` for every input, which says nothing about what it was fitted on.
    pub fn has_known_fitting_inputs(&self) -> bool {
        !self.fitting_inputs.is_empty()
    }

    /// Whether an input with this content SHA-256 is among the fitting inputs
    /// (requirement 11.1). The comparison ignores the case of the hexadecimal digits.
    pub fn was_fitted_on(&self, content_sha256: &str) -> bool {
        self.fitting_inputs
            .iter()
            .any(|input| input.content_sha256.eq_ignore_ascii_case(content_sha256))
    }

    /// Whether the scorer knows the texts it was fitted on (module documentation,
    /// "Fitted texts"). `false` for a scorer saved before task 1.11: for it only the
    /// whole-input check of [`BaselineScorer::was_fitted_on`] exists.
    pub fn has_fitted_text_hashes(&self) -> bool {
        self.fitted_texts.is_some()
    }

    /// How many distinct texts the scorer knows it was fitted on; `None` when it does
    /// not know them.
    pub fn fitted_text_hashes(&self) -> Option<u64> {
        self.fitted_texts.as_ref().map(FittedTexts::len)
    }

    /// Whether the hashes cover every fitting input the scorer names: `Some(false)`
    /// after hashes were merged into a saved scorer whose earlier fitting inputs came
    /// without them, so that [`BaselineScorer::was_fitted_on_text`] can say yes for a
    /// text but its no is not a proof; `None` when the scorer has no hashes.
    pub fn fitted_text_hashes_complete(&self) -> Option<bool> {
        self.fitted_texts.as_ref().map(|texts| texts.complete)
    }

    /// Whether the scorer was fitted on a log with exactly this text (requirement
    /// 11.1, per log); `None` when the scorer does not know the texts it was fitted on.
    pub fn was_fitted_on_text(&self, text: &str) -> Option<bool> {
        self.fitted_texts
            .as_ref()
            .map(|texts| texts.contains(&text_hash(text)))
    }

    /// The memory the scorer keeps for as long as it lives, fitted or loaded, in bytes
    /// (requirement 12.9): the weights (8 bytes per column), the hashes of the fitted
    /// texts ([`TEXT_HASH_BYTES`] each), and the description of its fitting inputs.
    pub fn held_bytes(&self) -> u64 {
        let inputs: usize = self
            .fitting_inputs
            .iter()
            .map(|input| {
                size_of::<FittingInput>() + input.source_path.len() + input.content_sha256.len()
            })
            .sum();
        let texts = self
            .fitted_texts
            .as_ref()
            .map_or(0, |texts| texts.bytes.capacity());
        (size_of::<Self>()
            + size_of_val(self.model.weights())
            + texts
            + inputs
            + self.revision.len()) as u64
    }

    /// The memory [`BaselineScorer::load`] of `dir` takes at its peak, in bytes, from
    /// the sizes of the files and without reading them: the weights once (they are
    /// read in pieces of 64 KiB straight into their final form and hashed as they
    /// pass), the hash file once, `baseline.json` eight times (the text and what is
    /// parsed from it), and [`LOAD_FIXED_BYTES`] for the piece being read and the
    /// rest. 0 when no file is there; the load itself reports that.
    /// `tests/scorer_memory.rs` holds the heap of a load to this figure.
    pub fn load_bytes(dir: &Path) -> u64 {
        let size = |file: &str| fs::metadata(dir.join(file)).map_or(0, |found| found.len());
        let files = size(WEIGHTS_FILE)
            .saturating_add(size(FITTED_TEXTS_FILE))
            .saturating_add(size(MANIFEST_FILE).saturating_mul(8));
        match files {
            0 => 0,
            files => files.saturating_add(LOAD_FIXED_BYTES),
        }
    }

    /// The fitted weights, one per column of the feature space.
    pub fn weights(&self) -> &[f64] {
        self.model.weights()
    }

    /// The fitted bias.
    pub fn bias(&self) -> f64 {
        self.model.bias()
    }

    /// The memory hashing one text of `longest_text_bytes` bytes is estimated to take
    /// while it is scored, in bytes: the figure the fit's estimate uses for its longest
    /// text (40 bytes per n-gram, `ngram_max` n-grams per byte). Texts are scored one
    /// after another, so the longest text of a set decides (task 1.10, for the memory
    /// budget of triage).
    pub fn scoring_bytes(&self, longest_text_bytes: u64) -> u64 {
        longest_text_bytes
            .saturating_mul(u64::from(self.settings.ngram_max))
            .saturating_mul(HASHING_BYTES_PER_NGRAM)
    }

    /// The ratio of `text` and whether the text has no features.
    fn score_one(&self, text: &str) -> (f32, bool) {
        let row = self.hasher.row(text);
        let probability = self
            .model
            .probability_sparse(&row)
            .expect("rows of the scorer's hasher fit the scorer's model");
        (probability as f32, row.is_empty())
    }

    /// The bad/good ratio of one text: `0.0..=1.0`, higher is more likely bad. A text
    /// without features gets the constant of the module documentation.
    pub fn ratio(&self, text: &str) -> f32 {
        self.score_one(text).0
    }

    /// The scores frame of the rows of `logs`, in the same row order. Labels play no
    /// part. Columns: `text` String, `scorer` String ([`SCORER_KIND`]), `ratio` Float32
    /// (never null), `abstained` Boolean (always `false`), `no_features` Boolean.
    pub fn score(&self, logs: &LogsFrame) -> Result<DataFrame, L2dError> {
        let text = logs.frame().column(LogsFrame::TEXT).expect(CHECKED).clone();
        self.scores(text)
    }

    /// The scores frame of `texts`, in the same order; see [`BaselineScorer::score`].
    pub fn score_texts(&self, texts: &[&str]) -> Result<DataFrame, L2dError> {
        self.scores(Column::new(SCORES_TEXT.into(), texts))
    }

    fn scores(&self, text: Column) -> Result<DataFrame, L2dError> {
        let failed = |error: &dyn std::fmt::Display| L2dError::Runtime {
            detail: format!("building the scores frame failed: {error}"),
        };
        let texts = text.str().map_err(|error| failed(&error))?;
        let rows = texts.len();
        let mut ratios = Vec::with_capacity(rows);
        let mut no_features = Vec::with_capacity(rows);
        for (row, text) in texts.iter().enumerate() {
            let text = text.ok_or_else(|| L2dError::InvalidInput {
                detail: format!("row {row} has no text to score"),
            })?;
            let (ratio, empty) = self.score_one(text);
            ratios.push(ratio);
            no_features.push(empty);
        }
        DataFrame::new(vec![
            text.with_name(SCORES_TEXT.into()),
            Column::new(SCORES_SCORER.into(), vec![SCORER_KIND; rows]),
            Column::new(SCORES_RATIO.into(), ratios),
            Column::new(SCORES_ABSTAINED.into(), vec![false; rows]),
            Column::new(SCORES_NO_FEATURES.into(), no_features),
        ])
        .map_err(|error| failed(&error))
    }

    fn manifest(&self) -> Manifest {
        Manifest {
            format: FORMAT,
            kind: SCORER_KIND.into(),
            revision: self.revision.clone(),
            tool_version: env!("CARGO_PKG_VERSION").into(),
            settings: SettingsFile {
                buckets: self.settings.buckets,
                ngram_max: self.settings.ngram_max,
                routing: routing_name(self.settings.routing).into(),
                l2: Exact::of(self.settings.l2),
                max_iter: self.settings.max_iter,
                tolerance: Exact::of(self.settings.tolerance),
            },
            model: ModelFile {
                dimension: self.model.dimension() as u64,
                bias: Exact::of(self.model.bias()),
            },
            fit: FitFile {
                termination: termination_name(self.report.termination).into(),
                iterations: self.report.iterations,
                objective: Exact::of(self.report.objective),
                gradient_norm: Exact::of(self.report.gradient_norm),
                rows: self.report.rows,
                good: self.report.good,
                bad: self.report.bad,
                unknown_ignored: self.report.unknown_ignored,
                no_feature_rows: self.report.no_feature_rows,
                stored_entries: self.report.stored_entries,
                estimated_bytes: self.report.estimated_bytes,
            },
            fitting_inputs: self.fitting_inputs.clone(),
            fitted_texts: self.fitted_texts.as_ref().map(FittedTexts::file),
        }
    }

    /// Saves the scorer as `models/baselines/<name>-<rev12>/` and returns that
    /// directory. The directory appears complete or not at all.
    ///
    /// - A name that is not 1 to [`MAX_NAME_BYTES`] letters, digits, `.`, `_` or `-` is
    ///   [`L2dError::InvalidInput`].
    /// - A target that is not directly below the project's `models/baselines/` is
    ///   [`L2dError::Refused`] (this includes a project root written with `..`).
    /// - When the directory already holds this very revision, the model is the same and
    ///   `weights.bin` is not written again. The stored fitting inputs become the union
    ///   of the stored ones and this scorer's: stored inputs keep their place and their
    ///   entry, and this scorer's inputs whose content hash is not yet stored (compared
    ///   without regard to case) follow in their order. The model was fitted on all of
    ///   them, so [`BaselineScorer::was_fitted_on`] of the loaded scorer is true for
    ///   each. If that adds an input, `baseline.json` is replaced atomically; otherwise
    ///   no file is touched. Everything else in `baseline.json` stays as stored, the
    ///   fit report included: equal revisions mean bit-equal weights from the same
    ///   settings, so the stored report describes this model too.
    /// - Only the stored file is merged. This scorer in memory keeps its own inputs,
    ///   which are a subset of what the directory then holds; load the directory to get
    ///   the union.
    /// - When the directory holds anything else (another revision, or content that
    ///   does not load), the save is [`L2dError::Refused`] and the directory is left
    ///   unchanged.
    pub fn save(&self, paths: &ProjectPaths, name: &str) -> Result<PathBuf, L2dError> {
        validate_name(name)?;
        let parent = paths.baselines_dir();
        let dir = paths.baseline_dir(name, &self.revision);
        if !paths.is_under_models(&dir) || dir.parent() != Some(parent.as_path()) {
            return Err(L2dError::Refused {
                reason: format!(
                    "baseline scorers are written only below the model directory {}; {} is \
                     not accepted as a place below it",
                    paths.models_dir().display(),
                    dir.display()
                ),
            });
        }
        if dir.exists() {
            return match Self::load(&dir) {
                Ok(mut stored) if stored.revision == self.revision => {
                    // The same model: only the fitting inputs can differ. Stored
                    // inputs keep their place; inputs not yet stored follow.
                    let known = stored.fitting_inputs.len();
                    // Whether this scorer was fitted on every input the directory
                    // names (and it names one): then this scorer's hashes cover them.
                    let covers_stored = known > 0
                        && stored
                            .fitting_inputs
                            .iter()
                            .all(|input| self.was_fitted_on(&input.content_sha256));
                    for input in &self.fitting_inputs {
                        if !stored.was_fitted_on(&input.content_sha256) {
                            stored.fitting_inputs.push(input.clone());
                        }
                    }
                    // The hashes of the fitted texts likewise: a directory without
                    // them gains this scorer's, two sets become their union.
                    // A set is complete only while it covers every input named: hashes
                    // that join inputs saved without hashes, or inputs that join
                    // without hashes, leave it incomplete.
                    let adds_inputs = stored.fitting_inputs.len() != known;
                    let texts = match (&stored.fitted_texts, &self.fitted_texts) {
                        (None, Some(own)) => Some(FittedTexts {
                            bytes: own.bytes.clone(),
                            complete: own.complete && covers_stored,
                        }),
                        (Some(have), Some(own)) => Some(have.union(own)).filter(|all| all != have),
                        (Some(have), None) if adds_inputs && have.complete => Some(FittedTexts {
                            bytes: have.bytes.clone(),
                            complete: false,
                        }),
                        (_, None) => None,
                    };
                    let texts_changed = texts.is_some();
                    if let Some(texts) = texts {
                        let same_file = stored
                            .fitted_texts
                            .as_ref()
                            .is_some_and(|have| have.bytes == texts.bytes);
                        if !same_file {
                            replace_file(&dir, FITTED_TEXTS_FILE, &texts.bytes)?;
                        }
                        stored.fitted_texts = Some(texts);
                    }
                    if stored.fitting_inputs.len() != known || texts_changed {
                        stored.replace_manifest(&dir)?;
                    }
                    Ok(dir)
                }
                _ => Err(L2dError::Refused {
                    reason: format!(
                        "{} already exists and does not hold revision {} of this scorer; it \
                         was left unchanged",
                        dir.display(),
                        self.revision
                    ),
                }),
            };
        }

        fs::create_dir_all(&parent).map_err(|error| io_failure("creating", &parent, &error))?;
        let temporary = parent.join(format!(
            ".{name}-{}.saving-{}-{}",
            short_revision(&self.revision),
            std::process::id(),
            SAVES.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&temporary).map_err(|error| io_failure("creating", &temporary, &error))?;
        let written = self.write_files(&temporary).and_then(|()| {
            fs::rename(&temporary, &dir).map_err(|error| io_failure("completing", &dir, &error))
        });
        if written.is_err() {
            // Best effort: the failure itself is what the caller has to see.
            let _ = fs::remove_dir_all(&temporary);
        }
        written.map(|()| dir)
    }

    fn manifest_json(&self) -> Result<String, L2dError> {
        serde_json::to_string_pretty(&self.manifest()).map_err(|error| L2dError::Runtime {
            detail: format!("describing the baseline scorer failed: {error}"),
        })
    }

    fn write_files(&self, dir: &Path) -> Result<(), L2dError> {
        write_weights(&dir.join(WEIGHTS_FILE), self.model.weights())?;
        if let Some(texts) = &self.fitted_texts {
            write_file(&dir.join(FITTED_TEXTS_FILE), &texts.bytes)?;
        }
        write_file(&dir.join(MANIFEST_FILE), self.manifest_json()?.as_bytes())
    }

    /// Replaces `baseline.json` of the existing scorer directory `dir` with this
    /// scorer's: written under a temporary name in `dir`, then renamed over the old
    /// file, so the directory holds the old file or the new one, never a partial one.
    fn replace_manifest(&self, dir: &Path) -> Result<(), L2dError> {
        replace_file(dir, MANIFEST_FILE, self.manifest_json()?.as_bytes())
    }

    /// Loads a scorer directory after checking that `dir` lies below the project's
    /// model directory; anything else is [`L2dError::Refused`]. See
    /// [`BaselineScorer::load`].
    pub fn load_under(paths: &ProjectPaths, dir: &Path) -> Result<Self, L2dError> {
        if !paths.is_under_models(dir) {
            return Err(L2dError::Refused {
                reason: format!(
                    "baseline scorers are read only from below the model directory {}, not \
                     from {}",
                    paths.models_dir().display(),
                    dir.display()
                ),
            });
        }
        Self::load(dir)
    }

    /// Loads a saved scorer. The loaded scorer gives bit-equal ratios.
    ///
    /// A directory or file that is not there is [`L2dError::Missing`], naming the item
    /// and where it was expected. Content that does not verify is [`L2dError::Refused`]:
    /// a revision that differs from the one computed from the files, a directory name
    /// that does not end in the first twelve characters of the revision, a weights file
    /// of the wrong length, or an unreadable `baseline.json`.
    pub fn load(dir: &Path) -> Result<Self, L2dError> {
        if !dir.is_dir() {
            return Err(L2dError::Missing {
                item: "baseline scorer".into(),
                expected_at: dir.to_path_buf(),
            });
        }
        let manifest = read_scorer_file(dir, MANIFEST_FILE)?;
        let manifest: Manifest = serde_json::from_slice(&manifest).map_err(|error| {
            not_a_scorer(dir, format!("{MANIFEST_FILE} is not readable: {error}"))
        })?;
        let known_format =
            manifest.format == FORMAT || manifest.format == FORMAT_WITHOUT_TEXT_HASHES;
        if !known_format || manifest.kind != SCORER_KIND {
            return Err(not_a_scorer(
                dir,
                format!(
                    "{MANIFEST_FILE} has format {} and kind '{}', expected format \
                     {FORMAT_WITHOUT_TEXT_HASHES} or {FORMAT} and kind '{SCORER_KIND}'",
                    manifest.format, manifest.kind
                ),
            ));
        }
        if manifest.format == FORMAT_WITHOUT_TEXT_HASHES && manifest.fitted_texts.is_some() {
            return Err(not_a_scorer(
                dir,
                format!(
                    "{MANIFEST_FILE} has format {FORMAT_WITHOUT_TEXT_HASHES}, which has no \
                     entry fitted_texts, and yet has one"
                ),
            ));
        }

        let unreadable =
            |what: &str| not_a_scorer(dir, format!("{MANIFEST_FILE} has no usable {what}"));
        let settings = BaselineSettings {
            buckets: manifest.settings.buckets,
            ngram_max: manifest.settings.ngram_max,
            routing: routing_from_name(&manifest.settings.routing)
                .ok_or_else(|| unreadable("routing mode"))?,
            l2: manifest.settings.l2.get().ok_or_else(|| unreadable("l2"))?,
            max_iter: manifest.settings.max_iter,
            tolerance: manifest
                .settings
                .tolerance
                .get()
                .ok_or_else(|| unreadable("tolerance"))?,
        };
        let hasher = settings
            .hasher()
            .map_err(|detail| not_a_scorer(dir, detail))?;
        let bias = manifest
            .model
            .bias
            .get()
            .ok_or_else(|| unreadable("bias"))?;
        let report = BaselineFitReport {
            termination: termination_from_name(&manifest.fit.termination)
                .ok_or_else(|| unreadable("termination"))?,
            iterations: manifest.fit.iterations,
            objective: manifest
                .fit
                .objective
                .get()
                .ok_or_else(|| unreadable("objective"))?,
            gradient_norm: manifest
                .fit
                .gradient_norm
                .get()
                .ok_or_else(|| unreadable("gradient norm"))?,
            rows: manifest.fit.rows,
            good: manifest.fit.good,
            bad: manifest.fit.bad,
            unknown_ignored: manifest.fit.unknown_ignored,
            no_feature_rows: manifest.fit.no_feature_rows,
            stored_entries: manifest.fit.stored_entries,
            estimated_bytes: manifest.fit.estimated_bytes,
        };

        // The length is checked before the file is read, so a wrong file is never loaded.
        let weights_path = dir.join(WEIGHTS_FILE);
        let expected_bytes = u64::from(hasher.dimension()) * 8;
        match fs::metadata(&weights_path) {
            Ok(found) if found.len() == expected_bytes => {}
            Ok(found) => {
                return Err(not_a_scorer(
                    dir,
                    format!(
                        "{WEIGHTS_FILE} has {} bytes, but dimension {} needs {expected_bytes}",
                        found.len(),
                        hasher.dimension()
                    ),
                ));
            }
            Err(_) => {}
        }
        if manifest.model.dimension != u64::from(hasher.dimension()) {
            return Err(not_a_scorer(
                dir,
                format!(
                    "{MANIFEST_FILE} states dimension {}, but the settings give dimension {}",
                    manifest.model.dimension,
                    hasher.dimension()
                ),
            ));
        }
        // The weights are read once, straight into their final form, and hashed as
        // they pass: the load never holds them twice (module documentation, "Memory").
        let mut hash = RevisionHasher::new(&settings, u64::from(hasher.dimension()));
        let weights = read_weights(dir, hasher.dimension() as usize, &mut hash)?;
        let revision = hash.finish(bias);
        if revision != manifest.revision {
            return Err(L2dError::Refused {
                reason: format!(
                    "the baseline scorer in {} does not match its revision: {MANIFEST_FILE} \
                     states {}, the files give {revision}",
                    dir.display(),
                    manifest.revision
                ),
            });
        }
        let short = short_revision(&revision);
        let named = dir
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(&format!("-{short}")));
        if !named {
            return Err(L2dError::Refused {
                reason: format!(
                    "the baseline scorer in {} has revision {revision}, but the directory \
                     name does not end in -{short}",
                    dir.display()
                ),
            });
        }

        let model = LogReg::from_parts(weights, bias).map_err(|error| not_a_scorer(dir, error))?;
        let fitted_texts = manifest
            .fitted_texts
            .as_ref()
            .map(|stated| read_fitted_texts(dir, stated))
            .transpose()?;
        Ok(Self {
            settings,
            hasher,
            model,
            revision,
            report,
            fitting_inputs: manifest.fitting_inputs,
            fitted_texts,
        })
    }
}

/// Reads and verifies the hash file `baseline.json` describes under `fitted_texts`.
fn read_fitted_texts(dir: &Path, stated: &FittedTextsFile) -> Result<FittedTexts, L2dError> {
    let unusable = |why: String| {
        not_a_scorer(
            dir,
            format!("the entry fitted_texts of {MANIFEST_FILE} is not usable: {why}"),
        )
    };
    // The name is compared, never used: only the one file of this directory is opened.
    if stated.file != FITTED_TEXTS_FILE {
        return Err(unusable(format!(
            "it names a file other than {FITTED_TEXTS_FILE}"
        )));
    }
    if stated.hash != TEXT_HASH_NAME {
        return Err(unusable(format!(
            "it states the hash rule '{}', expected '{TEXT_HASH_NAME}'",
            stated.hash
        )));
    }
    let bytes = read_scorer_file(dir, FITTED_TEXTS_FILE)?;
    let found_sha256 = content_sha256(&bytes);
    let texts = FittedTexts::from_file_bytes(bytes, stated.complete)
        .map_err(|why| unusable(format!("{FITTED_TEXTS_FILE} does not verify: {why}")))?;
    if texts.len() != stated.count {
        return Err(unusable(format!(
            "it states {} fitted texts, but {FITTED_TEXTS_FILE} holds {}",
            stated.count,
            texts.len()
        )));
    }
    if !found_sha256.eq_ignore_ascii_case(&stated.sha256) {
        return Err(unusable(format!(
            "{FITTED_TEXTS_FILE} does not have the SHA-256 it states"
        )));
    }
    Ok(texts)
}

/// Replaces `file` of the existing scorer directory `dir`: written under a temporary
/// name in `dir`, then renamed over the old file, so the directory holds the old file
/// or the new one, never a partial one.
fn replace_file(dir: &Path, file: &str, bytes: &[u8]) -> Result<(), L2dError> {
    let target = dir.join(file);
    let temporary = dir.join(format!(
        ".{file}.saving-{}-{}",
        std::process::id(),
        SAVES.fetch_add(1, Ordering::Relaxed)
    ));
    let written = write_file(&temporary, bytes).and_then(|()| {
        fs::rename(&temporary, &target).map_err(|error| io_failure("replacing", &target, &error))
    });
    if written.is_err() {
        // Best effort: the failure itself is what the caller has to see.
        let _ = fs::remove_file(&temporary);
    }
    written
}
