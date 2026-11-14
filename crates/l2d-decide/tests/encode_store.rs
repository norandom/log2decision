//! Encoding logs into a store (task 2.7; requirements 4.1, 5.1, 5.6 and 12), everything
//! that needs no NPU.
//!
//! No model is run here, on any provider. The encoder behind the seam
//! ([`l2d_decide::LogEncoder`]) is the encoder crate's own `ImageEncoder` around a formula
//! of this file (as in `l2d-encoder/tests/encoder_session.rs`), with an identity and a
//! runtime release the test makes up. The dimensions are 4 and 6, the inputs 8 and 12
//! bytes, in both input modes: nothing here or in the code under test knows the published
//! encoder. The run on the device is in `npu_encode_store.rs`, on request.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    EncoderIdentity, InputAdapter, L2dError, Label, LabelledAdapter, LabelledFormat, LogRecord,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths, RunKind, RunRecord,
};
use l2d_decide::{FillCause, FillReport, LogEncoder, encode_into_store, fill_row_bytes};
use l2d_encoder::encode::RowImage;
use l2d_encoder::{
    ByteCodec, ByteTable, EncoderError, EncoderSession, ImageEncoder, ImageModel, InputArch,
    InputMode, TextInput,
};
use l2d_store::{
    ADD_FIXED_BYTES, ADD_TEXT_FACTOR, Filter, MAX_TEXT_BYTES, Store, StoreError, VectorOrigin,
};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const SOURCE: &str = "tests/fixtures/mini-labelled.csv";
const RELEASE: &str = "1.9.0";
const OTHER_RELEASE: &str = "1.10.0";
/// Texts per image of the stand-in encoder.
const ROWS: usize = 4;

// ---------------------------------------------------------------------------------------
// The stand-in encoder
// ---------------------------------------------------------------------------------------

/// Row `n` of the output is a fixed formula of row `n` of the image.
struct Formula {
    dimension: usize,
}

impl ImageModel for Formula {
    fn run(&mut self, image: &RowImage, output: &mut [f32]) -> Result<(), EncoderError> {
        let [_, channels, rows, width] = image.dims().map(|dim| dim as usize);
        let values = image.values();
        for n in 0..rows {
            for d in 0..self.dimension {
                let mut value = (d + 1) as f32 * 0.25;
                for c in 0..channels {
                    for t in 0..width {
                        let weight = ((d * 31 + c * 7 + t * 3) % 11) as f32 - 5.0;
                        value += values[c * rows * width + n * width + t] * weight;
                    }
                }
                output[d * rows + n] = value;
            }
        }
        Ok(())
    }
}

/// An encoder for the seam: the encoder crate's image encoder around the formula.
struct StandIn {
    encoder: ImageEncoder<Formula>,
    identity: EncoderIdentity,
    release: String,
    /// The number of texts of every call.
    calls: Vec<usize>,
    /// Fail the call with this number (from 0).
    fail_at: Option<usize>,
    /// Return a value that is not a number in the last vector of the call with this
    /// number (from 0): rows the store refuses.
    spoil_at: Option<usize>,
}

fn stand_in(dimension: u16, input_bytes: u32, mode: InputMode, release: &str) -> StandIn {
    let arch = InputArch {
        input_bytes,
        channels: 3,
        batch: 2,
    };
    let mut table = Vec::new();
    for byte in 0..256u32 {
        for channel in 0..arch.channels {
            let value = if byte == 0 {
                0.0f32
            } else {
                (byte * arch.channels + channel) as f32 / 1024.0 - 0.25
            };
            table.extend_from_slice(&value.to_le_bytes());
        }
    }
    let table = ByteTable::from_le_bytes(&table, &arch).unwrap();
    let codec = ByteCodec::new(arch, table, mode).unwrap();
    StandIn {
        encoder: ImageEncoder::new(
            codec,
            usize::from(dimension),
            ROWS,
            Formula {
                dimension: usize::from(dimension),
            },
        ),
        identity: EncoderIdentity {
            name: format!("stand-in-{dimension}-{input_bytes}-{}", mode.as_str()),
            revision: format!(
                "{dimension:02}{input_bytes:02}abcdef0123456789abcdef0123456789abcdef"
            ),
            dimension,
        },
        release: release.to_string(),
        calls: Vec::new(),
        fail_at: None,
        spoil_at: None,
    }
}

/// The encoder most tests use: dimension 4, 8 input bytes, the payload part.
fn small() -> StandIn {
    stand_in(4, 8, InputMode::Payload, RELEASE)
}

impl LogEncoder for StandIn {
    fn identity(&self) -> &EncoderIdentity {
        &self.identity
    }

    fn runtime_release(&self) -> &str {
        &self.release
    }

    fn encode_into(
        &mut self,
        texts: &[&str],
        vectors: &mut Vec<f32>,
        flags: &mut Vec<TextInput>,
    ) -> Result<(), EncoderError> {
        let call = self.calls.len();
        self.calls.push(texts.len());
        if self.fail_at == Some(call) {
            return Err(EncoderError::Failed {
                detail: format!("call {call} failed"),
            });
        }
        self.encoder.encode_into(texts, vectors, flags)?;
        if self.spoil_at == Some(call)
            && let Some(last) = vectors.last_mut()
        {
            *last = f32::NAN;
        }
        Ok(())
    }
}

impl StandIn {
    fn origin(&self) -> VectorOrigin<'_> {
        VectorOrigin::made_under(&self.identity, &self.release)
    }

    fn texts_encoded(&self) -> u64 {
        self.encoder.stats().texts
    }
}

/// What a fresh encoder of the same kind gives for each text on its own.
fn alone(dimension: u16, input_bytes: u32, mode: InputMode, texts: &[String]) -> Vec<Expected> {
    let mut encoder = stand_in(dimension, input_bytes, mode, RELEASE);
    texts
        .iter()
        .map(|text| {
            let encoded = encoder.encoder.encode(&[text.as_str()]).unwrap().remove(0);
            Expected {
                vector: encoded.vector.iter().map(|value| value.to_bits()).collect(),
                input: encoded.input,
            }
        })
        .collect()
}

struct Expected {
    vector: Vec<u32>,
    input: TextInput,
}

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-encode-store-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("the scratch root can be created");
        Self { root }
    }

    fn paths(&self) -> ProjectPaths {
        ProjectPaths::new(&self.root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A budget with a fixed free-memory reading, so the tests do not depend on the host.
fn budget_with(limit: Option<u64>, batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).expect("the budget is usable")
}

/// The 169 distinct logs of the mini fixture.
fn fixture_logs() -> LogsFrame {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv");
    let (logs, summary) = LabelledAdapter::new(LabelledFormat::Csv)
        .with_group_column("group")
        .read(&path)
        .expect("the fixture is readable");
    assert_eq!(summary.distinct_texts, 169);
    logs
}

fn record(text: &str, position: u64) -> LogRecord {
    LogRecord {
        text: text.to_string(),
        label: Label::Unknown,
        group: None,
        first_position: position,
        occurrences: 1,
    }
}

fn texts_of(logs: &LogsFrame) -> Vec<String> {
    logs.records().map(|record| record.text).collect()
}

/// One stored row, as the store gives it back.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    text: String,
    source: String,
    occurrences: u32,
    label: String,
    group: Option<String>,
    truncated: bool,
    text_longer_than_input: bool,
    empty: bool,
    vector: Vec<u32>,
}

/// Every stored row in order, read as vectors of the store's own encoder and release.
fn stored(store: &Store) -> Vec<Row> {
    let release = store
        .runtime_release()
        .expect("the stores of these tests record a release");
    let origin = VectorOrigin::made_under(store.identity(), release);
    let frame: DataFrame = store.vectors(origin, &Filter::default()).unwrap();
    let text = frame.column(l2d_store::TEXT).unwrap().str().unwrap();
    let source = frame.column(l2d_store::SOURCE).unwrap().str().unwrap();
    let occurrences = frame.column(l2d_store::OCCURRENCES).unwrap().u32().unwrap();
    let label = frame.column(l2d_store::LABEL).unwrap().str().unwrap();
    let group = frame.column(l2d_store::GROUP).unwrap().str().unwrap();
    let truncated = frame.column(l2d_store::TRUNCATED).unwrap().bool().unwrap();
    let longer = frame
        .column(l2d_store::TEXT_LONGER_THAN_INPUT)
        .unwrap()
        .bool()
        .unwrap();
    let empty = frame.column(l2d_store::EMPTY).unwrap().bool().unwrap();
    let vector = frame.column(l2d_store::VECTOR).unwrap().list().unwrap();
    (0..frame.height())
        .map(|row| Row {
            text: text.get(row).unwrap().to_string(),
            source: source.get(row).unwrap().to_string(),
            occurrences: occurrences.get(row).unwrap(),
            label: label.get(row).unwrap().to_string(),
            group: group.get(row).map(str::to_string),
            truncated: truncated.get(row).unwrap(),
            text_longer_than_input: longer.get(row).expect("a format-2 store knows the flag"),
            empty: empty.get(row).expect("a format-2 store knows the flag"),
            vector: vector
                .get_as_series(row)
                .unwrap()
                .f32()
                .unwrap()
                .into_no_null_iter()
                .map(f32::to_bits)
                .collect(),
        })
        .collect()
}

fn file(store: &Store) -> Vec<u8> {
    std::fs::read(store.path()).expect("the store file is readable")
}

fn fill(
    encoder: &mut StandIn,
    store: &mut Store,
    logs: &LogsFrame,
    budget: &mut MemoryBudget,
) -> FillReport {
    encode_into_store(encoder, store, logs, SOURCE, budget)
        .unwrap_or_else(|error| panic!("{error}"))
}

// ---------------------------------------------------------------------------------------
// The task's done-when
// ---------------------------------------------------------------------------------------

#[test]
fn the_fixture_logs_are_encoded_and_stored_under_the_encoders_identity_and_release() {
    let scratch = Scratch::new("fixture");
    let logs = fixture_logs();
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "fixture", encoder.origin()).unwrap();
    assert_eq!(store.identity(), &encoder.identity);
    assert_eq!(store.runtime_release(), Some(RELEASE));

    let mut budget = budget_with(None, None);
    let report = fill(&mut encoder, &mut store, &logs, &mut budget);
    assert_eq!(
        (report.logs, report.encoded, report.added, report.repeated),
        (169, 169, 169, 0)
    );
    assert!(report.skipped.is_empty());
    assert_eq!(encoder.texts_encoded(), 169);
    assert_eq!(store.len().unwrap(), 169);
    assert_eq!(
        budget.held_bytes(),
        0,
        "the budget holds what it held before"
    );

    // Every log is stored with the vector the encoder gives its text on its own, in the
    // order of the frame, with what the frame says about it.
    let rows = stored(&store);
    let texts = texts_of(&logs);
    let expected = alone(4, 8, InputMode::Payload, &texts);
    assert_eq!(rows.len(), 169);
    for ((row, record), expected) in rows.iter().zip(logs.records()).zip(&expected) {
        assert_eq!(row.text, record.text);
        assert_eq!(row.source, SOURCE);
        assert_eq!(row.occurrences, record.occurrences);
        assert_eq!(row.label, record.label.as_str());
        assert_eq!(row.group, record.group);
        assert_eq!(row.vector, expected.vector, "{}", row.text);
        assert_eq!(row.vector.len(), 4);
        // The three flags are stored as the encoder reports them for the text.
        assert_eq!(row.truncated, expected.input.truncated, "{}", row.text);
        assert_eq!(
            row.text_longer_than_input, expected.input.text_longer_than_input,
            "{}",
            row.text
        );
        assert_eq!(row.empty, expected.input.empty, "{}", row.text);
    }
    // In the payload mode the two flags of requirement 1.5 differ for some logs, and
    // the store keeps them apart.
    assert!(
        rows.iter()
            .any(|row| row.text_longer_than_input && !row.truncated)
    );

    // Both truncation flags of requirement 1.5 and the empty inputs are counted.
    let count = |flag: fn(&TextInput) -> bool| {
        expected.iter().filter(|text| flag(&text.input)).count() as u64
    };
    assert_eq!(report.truncated, count(|input| input.truncated));
    assert_eq!(
        report.longer_than_input,
        count(|input| input.text_longer_than_input)
    );
    assert_eq!(report.empty_inputs, count(|input| input.empty));
    assert!(report.truncated > 0 && report.truncated < report.longer_than_input);
}

#[test]
fn adding_the_same_logs_again_only_adds_their_occurrences() {
    let scratch = Scratch::new("again");
    let logs = fixture_logs();
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "fixture", encoder.origin()).unwrap();
    let mut budget = budget_with(None, None);
    fill(&mut encoder, &mut store, &logs, &mut budget);
    let first = stored(&store);

    let report = fill(&mut encoder, &mut store, &logs, &mut budget);
    assert_eq!(
        (report.logs, report.encoded, report.added, report.repeated),
        (169, 169, 0, 169)
    );
    assert_eq!(store.len().unwrap(), 169);
    let second = stored(&store);
    for (before, after) in first.iter().zip(&second) {
        assert_eq!(after.occurrences, 2 * before.occurrences, "{}", before.text);
        assert_eq!(
            Row {
                occurrences: before.occurrences,
                ..after.clone()
            },
            *before,
            "nothing but the count changed"
        );
    }
}

#[test]
fn a_store_of_another_runtime_release_is_refused_before_anything_is_encoded() {
    let scratch = Scratch::new("release");
    let logs = fixture_logs();
    let mut first = small();
    let mut store = Store::create(&scratch.paths(), "fixture", first.origin()).unwrap();
    let mut budget = budget_with(None, None);
    fill(&mut first, &mut store, &logs, &mut budget);
    let before = file(&store);

    // The same encoder package under another release of the AMD runtime.
    let mut later = stand_in(4, 8, InputMode::Payload, OTHER_RELEASE);
    assert_eq!(later.identity, first.identity);
    let error = encode_into_store(&mut later, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("vectors of another release");
    assert!(
        matches!(
            &*error.cause,
            FillCause::Store(StoreError::ReleaseMismatch { store, given, .. })
                if store.as_deref() == Some(RELEASE) && given.as_deref() == Some(OTHER_RELEASE)
        ),
        "{error}"
    );
    assert_eq!((error.logs, error.encoded), (169, 0));
    let message = error.to_string();
    assert!(
        message.contains(RELEASE) && message.contains(OTHER_RELEASE),
        "{message}"
    );
    assert!(later.calls.is_empty(), "nothing was encoded");
    assert_eq!(file(&store), before, "nothing was stored");
    assert!(matches!(L2dError::from(error), L2dError::Refused { .. }));
    assert_eq!(budget.held_bytes(), 0);
}

#[test]
fn a_store_of_another_encoder_is_refused_and_each_encoder_fills_its_own() {
    let scratch = Scratch::new("encoder");
    let paths = scratch.paths();
    let logs = fixture_logs();
    let mut small = small();
    // Another encoder altogether: six dimensions, twelve bytes, the whole text.
    let mut wide = stand_in(6, 12, InputMode::Whole, RELEASE);
    let mut store = Store::create(&paths, "small", small.origin()).unwrap();
    let mut budget = budget_with(None, None);

    let error = encode_into_store(&mut wide, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("vectors of another encoder");
    assert!(
        matches!(
            &*error.cause,
            FillCause::Store(StoreError::IdentityMismatch { .. })
        ),
        "{error}"
    );
    assert!(wide.calls.is_empty(), "nothing was encoded");
    assert!(store.is_empty().unwrap());

    fill(&mut small, &mut store, &logs, &mut budget);
    let mut wide_store = Store::create(&paths, "wide", wide.origin()).unwrap();
    let report = fill(&mut wide, &mut wide_store, &logs, &mut budget);
    assert_eq!(report.added, 169);
    let rows = stored(&wide_store);
    let expected = alone(6, 12, InputMode::Whole, &texts_of(&logs));
    for (row, expected) in rows.iter().zip(&expected) {
        assert_eq!(row.vector.len(), 6);
        assert_eq!(row.vector, expected.vector);
        assert_eq!(row.truncated, expected.input.truncated);
        assert_eq!(
            row.text_longer_than_input,
            expected.input.text_longer_than_input
        );
        assert_eq!(row.empty, expected.input.empty);
    }
    // In the whole mode the two flags are one.
    assert_eq!(report.truncated, report.longer_than_input);
}

// ---------------------------------------------------------------------------------------
// Memory (requirement 12)
// ---------------------------------------------------------------------------------------

#[test]
fn encoding_and_adding_run_in_batches_and_the_stored_rows_do_not_depend_on_them() {
    let scratch = Scratch::new("batches");
    let paths = scratch.paths();
    let logs = fixture_logs();

    let mut whole = small();
    let mut one = Store::create(&paths, "one", whole.origin()).unwrap();
    let report = fill(&mut whole, &mut one, &logs, &mut budget_with(None, None));
    assert_eq!(report.adds, 1);
    assert_eq!(whole.calls, [169]);

    let mut batched = small();
    let mut many = Store::create(&paths, "many", batched.origin()).unwrap();
    let mut budget = budget_with(None, Some(16));
    let report = fill(&mut batched, &mut many, &logs, &mut budget);
    assert_eq!(report.adds, 11, "ten batches of 16 and one of 9");
    assert_eq!(batched.calls, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 9]);
    assert_eq!((report.added, report.repeated), (169, 0));
    assert_eq!(report.plan.batch_rows, 16);
    assert_eq!(report.plan.max_rows_used, 16);
    assert_eq!(report.plan.min_rows_used, 9, "the last batch was one too");
    assert!(report.plan.overridden);
    assert_eq!(budget.held_bytes(), 0);

    assert_eq!(stored(&many), stored(&one));
}

#[test]
fn a_batch_is_planned_with_what_the_fill_holds_and_what_the_add_asks_per_row() {
    let scratch = Scratch::new("row-bytes");
    let logs = fixture_logs();
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "s", encoder.origin()).unwrap();
    let report = fill(
        &mut encoder,
        &mut store,
        &logs,
        &mut budget_with(None, None),
    );

    let logs_row = (logs.frame().estimated_size() as u64).div_ceil(169);
    assert_eq!(
        report.plan.bytes_per_row,
        fill_row_bytes(4, SOURCE.len() as u64, logs_row)
    );
    // Three copies of a vector are in memory while a batch is added: the encoder's
    // output, the frame's column and the store's.
    assert_eq!(
        fill_row_bytes(128, 10, 100) - fill_row_bytes(4, 10, 100),
        3 * 4 * 124
    );
    assert!(fill_row_bytes(4, 0, 0) >= l2d_store::ADD_ROW_BYTES);
    // The share the store holds for an add, as its report says.
    let longest = texts_of(&logs).iter().map(String::len).max().unwrap() as u64;
    assert_eq!(report.longest_text_bytes, longest);
    assert_eq!(
        report.add_held_bytes,
        ADD_FIXED_BYTES + ADD_TEXT_FACTOR * longest
    );
}

#[test]
fn the_adds_share_is_checked_before_anything_is_encoded() {
    let scratch = Scratch::new("share");
    let logs = fixture_logs();
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "s", encoder.origin()).unwrap();
    let before = file(&store);

    // 8 MiB are left for data: less than the 16 MiB an add holds for the database.
    let mut budget = budget_with(Some(40 * MIB), None);
    let error = encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("the add would not fit");
    assert!(matches!(&*error.cause, FillCause::Memory(_)), "{error}");
    let message = error.to_string();
    assert!(message.contains("nothing was encoded"), "{message}");
    assert!(message.contains("record"), "{message}");
    assert!(encoder.calls.is_empty());
    assert_eq!(file(&store), before);
    assert_eq!(budget.held_bytes(), 0);
    assert!(matches!(L2dError::from(error), L2dError::Refused { .. }));

    // What the caller holds counts: the same logs fit an empty budget of 64 MiB and do
    // not fit when the caller holds 50 MiB of it already.
    let mut budget = budget_with(Some(96 * MIB), None);
    budget.hold(50 * MIB);
    let error = encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("the caller's frames leave too little");
    assert!(matches!(&*error.cause, FillCause::Memory(_)), "{error}");
    assert_eq!(budget.held_bytes(), 50 * MIB, "the caller's figure is back");
    assert!(encoder.calls.is_empty());

    let mut budget = budget_with(Some(96 * MIB), None);
    fill(&mut encoder, &mut store, &logs, &mut budget);
}

#[test]
fn a_long_text_is_refused_by_its_share_before_anything_is_encoded() {
    let scratch = Scratch::new("long");
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "s", encoder.origin()).unwrap();
    // 512 KiB of text: the add holds 16 MiB + 192 x 512 KiB = 112 MiB.
    let long = format!("GET /a?{}", "x".repeat(512 * 1024));
    let logs = LogsFrame::from_records(&[record("GET /a?b=1", 1), record(&long, 2)]).unwrap();
    let mut budget = budget_with(Some(128 * MIB), None);
    let error = encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("the long text's share does not fit 96 MiB");
    assert!(matches!(&*error.cause, FillCause::Memory(_)), "{error}");
    let message = error.to_string();
    assert!(
        message.contains(&long.len().to_string()),
        "the longest text is named: {message}"
    );
    assert!(encoder.calls.is_empty());
    assert!(store.is_empty().unwrap());

    let mut budget = budget_with(Some(256 * MIB), None);
    let report = fill(&mut encoder, &mut store, &logs, &mut budget);
    assert_eq!(report.added, 2);
    assert_eq!(report.longest_text_bytes, long.len() as u64);
}

// ---------------------------------------------------------------------------------------
// Rows that cannot be stored, failures, the record
// ---------------------------------------------------------------------------------------

#[test]
fn texts_the_store_cannot_hold_are_taken_out_and_reported_by_position() {
    let scratch = Scratch::new("skipped");
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "s", encoder.origin()).unwrap();
    let too_long = "y".repeat(MAX_TEXT_BYTES as usize + 1);
    let logs = LogsFrame::from_records(&[
        record("GET /a?b=1", 3),
        record("GET /a?b=\0x", 7),
        record("GET /c?d=2", 8),
        record(&too_long, 11),
        record("GET /e?f=3", 12),
    ])
    .unwrap();
    let report = fill(
        &mut encoder,
        &mut store,
        &logs,
        &mut budget_with(None, None),
    );
    assert_eq!(
        (report.logs, report.encoded, report.added),
        (5, 3, 3),
        "{report:?}"
    );
    assert_eq!(
        report
            .skipped
            .iter()
            .map(|skipped| skipped.position)
            .collect::<Vec<_>>(),
        [7, 11]
    );
    assert!(report.skipped[0].reason.contains("NUL"), "{report:?}");
    assert!(
        report.skipped[1]
            .reason
            .contains(&MAX_TEXT_BYTES.to_string()),
        "{report:?}"
    );
    assert_eq!(encoder.texts_encoded(), 3, "the encoder never saw them");
    assert_eq!(
        stored(&store)
            .iter()
            .map(|row| row.text.as_str())
            .collect::<Vec<_>>(),
        ["GET /a?b=1", "GET /c?d=2", "GET /e?f=3"]
    );
    // The longest text that counts is the longest that is stored.
    assert_eq!(report.longest_text_bytes, 10);
}

#[test]
fn logs_without_a_row_store_nothing_and_ask_nothing_of_the_encoder() {
    let scratch = Scratch::new("empty");
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "s", encoder.origin()).unwrap();
    let report = fill(
        &mut encoder,
        &mut store,
        &LogsFrame::empty(),
        &mut budget_with(None, None),
    );
    assert_eq!(
        (report.logs, report.encoded, report.added, report.adds),
        (0, 0, 0, 0)
    );
    assert!(encoder.calls.is_empty());
    assert!(store.is_empty().unwrap());
}

/// `count` logs of `logs` from row `from`.
fn part(logs: &LogsFrame, from: i64, count: usize) -> LogsFrame {
    LogsFrame::new(logs.frame().slice(from, count)).unwrap()
}

fn hits_for(store: &Store, phrase: &str) -> usize {
    let filter = Filter {
        contains: vec![phrase.to_string()],
        ..Filter::default()
    };
    store.text_search(&filter, usize::MAX).unwrap().height()
}

/// Owner decision B of 2026-11-14: a fill is all or nothing.
#[test]
fn a_failure_in_a_later_batch_rolls_the_whole_fill_back_and_a_rerun_is_a_clean_fill() {
    let scratch = Scratch::new("failure");
    let paths = scratch.paths();
    let logs = fixture_logs();
    let texts = texts_of(&logs);
    // A log of the first batches whose text is in none of the 40 stored before.
    let probe = texts[..100]
        .iter()
        .find(|text| {
            let folded = text.to_lowercase();
            !texts[100..140]
                .iter()
                .any(|stored| stored.to_lowercase().contains(&folded))
        })
        .expect("one of the first hundred logs is unlike the forty")
        .clone();
    let mut budget = budget_with(None, Some(16));

    // A store that holds 40 of the logs already.
    let mut encoder = small();
    let mut store = Store::create(&paths, "s", encoder.origin()).unwrap();
    fill(&mut encoder, &mut store, &part(&logs, 100, 40), &mut budget);
    let rows_before = stored(&store);
    let file_before = file(&store);
    assert_eq!(rows_before.len(), 40);
    assert_eq!(hits_for(&store, &probe), 0, "the first log is not there");

    // The ninth batch fails: eight batches (128 logs: 100 new ones and 28 that are
    // stored already) were encoded and added before it.
    let mut failing = small();
    failing.fail_at = Some(8);
    let error = encode_into_store(&mut failing, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("the ninth batch fails");
    assert!(
        matches!(
            &*error.cause,
            FillCause::Encoder(EncoderError::Failed { .. })
        ),
        "{error}"
    );
    assert_eq!(failing.calls.len(), 9);
    assert_eq!(
        (error.logs, error.encoded),
        (169, 128),
        "eight batches were encoded for nothing"
    );
    let message = error.to_string();
    assert!(message.contains("call 8 failed"), "{message}");
    assert!(message.contains("rolled back"), "{message}");
    assert!(
        message.contains("nothing of these 169 logs was stored"),
        "{message}"
    );
    assert!(
        !message.contains("stay stored") && !message.contains("second time"),
        "{message}"
    );
    let as_tool_error = L2dError::from(error);
    assert!(
        matches!(&as_tool_error, L2dError::Runtime { detail } if detail.contains("rolled back")),
        "{as_tool_error}"
    );

    // The store is as before: the rows, their occurrences, the text index, the file.
    assert_eq!(store.len().unwrap(), 40);
    assert_eq!(stored(&store), rows_before);
    assert_eq!(
        hits_for(&store, &probe),
        0,
        "the text index has no new text"
    );
    assert_eq!(file(&store), file_before, "the file is as it was");
    assert_eq!(budget.held_bytes(), 0);

    // So a rerun is safe: it gives the store that a fill without a failure gives.
    let mut again = small();
    let report = fill(&mut again, &mut store, &logs, &mut budget);
    assert_eq!((report.added, report.repeated), (129, 40));
    let mut clean_encoder = small();
    let mut clean = Store::create(&paths, "clean", clean_encoder.origin()).unwrap();
    fill(
        &mut clean_encoder,
        &mut clean,
        &part(&logs, 100, 40),
        &mut budget,
    );
    fill(&mut clean_encoder, &mut clean, &logs, &mut budget);
    assert_eq!(stored(&store), stored(&clean));
    assert_eq!(hits_for(&store, &probe), hits_for(&clean, &probe));
    assert!(hits_for(&store, &probe) >= 1);

    // A failure in the first batch: the same, on an empty store.
    let mut encoder = small();
    encoder.fail_at = Some(0);
    let mut empty = Store::create(&paths, "empty", encoder.origin()).unwrap();
    let error = encode_into_store(&mut encoder, &mut empty, &logs, SOURCE, &mut budget)
        .expect_err("the first batch fails");
    assert!(error.to_string().contains("nothing"), "{error}");
    assert!(empty.is_empty().unwrap());
}

/// Review finding 1: a batch is sized by the bytes of its own texts, not by the average
/// row of the frame.
#[test]
fn a_late_batch_of_long_texts_is_sized_by_its_own_bytes() {
    let scratch = Scratch::new("tail");
    let paths = scratch.paths();
    let mut records = Vec::new();
    for index in 0..2048u64 {
        records.push(record(&format!("GET /s?i={index}"), index + 1));
    }
    for index in 0..24u64 {
        records.push(record(
            &format!("GET /l?i={index}&{}", "x".repeat(200_000)),
            5000 + index,
        ));
    }
    let logs = LogsFrame::from_records(&records).unwrap();
    let frame_bytes = logs.frame().estimated_size() as u64;
    let longest = texts_of(&logs).iter().map(String::len).max().unwrap() as u64;
    // What is held before a batch: the tool's own share, the caller's frame, the open
    // store and the store's share of an add.
    let held = l2d_core::memory::PROCESS_BASELINE_BYTES
        + frame_bytes
        + l2d_store::IDLE_BYTES
        + ADD_FIXED_BYTES
        + ADD_TEXT_FACTOR * longest;

    // 4 MiB are left for a batch: the 1,024 short texts of a batch fit, the 24 long ones
    // (4.8 MB) do not fit as one batch, twelve of them do.
    let mut encoder = small();
    let mut store = Store::create(&paths, "cut", encoder.origin()).unwrap();
    let mut budget = budget_with(Some(held + 4 * MIB), Some(1024));
    budget.hold(frame_bytes);
    let report = encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
        .unwrap_or_else(|error| panic!("calls {:?}: {error}", encoder.calls));
    assert_eq!(encoder.calls, [1024, 1024, 12, 12]);
    assert_eq!((report.encoded, report.added, report.adds), (2072, 2072, 4));
    assert_eq!(
        (report.plan.min_rows_used, report.plan.max_rows_used),
        (12, 1024),
        "the batches used are reported"
    );
    assert_eq!(report.batches_cut, 1, "the 24 long texts became 12 and 12");
    assert_eq!(store.len().unwrap(), 2072);
    assert_eq!(budget.held_bytes(), frame_bytes);

    // 150 KB are left for a batch: sixteen average rows fit, one long text does not.
    // That is known before the first text is encoded.
    let mut encoder = small();
    let mut store = Store::create(&paths, "refused", encoder.origin()).unwrap();
    let before = file(&store);
    let mut budget = budget_with(Some(held + 150_000), Some(16));
    budget.hold(frame_bytes);
    let error = encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("one long text does not fit");
    assert!(
        encoder.calls.is_empty(),
        "nothing was encoded: {} calls",
        encoder.calls.len()
    );
    assert!(matches!(&*error.cause, FillCause::Memory(_)), "{error}");
    let message = error.to_string();
    assert!(message.contains("nothing was encoded"), "{message}");
    assert!(
        message.contains(&texts_of(&logs)[2048].len().to_string()) && message.contains("row 2049"),
        "the text that does not fit is named: {message}"
    );
    assert_eq!(file(&store), before);
    assert_eq!(budget.held_bytes(), frame_bytes);
}

/// Review finding 2: a text that is left out does not end a batch.
#[test]
fn a_batch_is_the_next_usable_texts_and_the_report_names_the_batches_used() {
    let scratch = Scratch::new("usable");
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "s", encoder.origin()).unwrap();
    let mut records = Vec::new();
    for index in 0..40u64 {
        let text = if index % 5 == 2 {
            format!("GET /n?i={index}\0z")
        } else {
            format!("GET /n?i={index}")
        };
        records.push(record(&text, index + 1));
    }
    let logs = LogsFrame::from_records(&records).unwrap();
    let report = fill(
        &mut encoder,
        &mut store,
        &logs,
        &mut budget_with(None, Some(16)),
    );
    assert_eq!(encoder.calls, [16, 16], "32 usable texts in two batches");
    assert_eq!((report.encoded, report.adds), (32, 2));
    assert_eq!(report.skipped.len(), 8);
    assert_eq!(
        (report.plan.min_rows_used, report.plan.max_rows_used),
        (16, 16)
    );

    // The stored rows are the usable texts in the order of the frame, each with the
    // vector of its own text.
    let usable: Vec<String> = texts_of(&logs)
        .into_iter()
        .filter(|text| !text.contains('\0'))
        .collect();
    let expected = alone(4, 8, InputMode::Payload, &usable);
    let rows = stored(&store);
    assert_eq!(rows.len(), 32);
    for ((row, text), expected) in rows.iter().zip(&usable).zip(&expected) {
        assert_eq!(&row.text, text);
        assert_eq!(row.vector, expected.vector, "{text}");
    }

    // A last batch that is smaller is a batch that was used.
    let mut encoder = small();
    let mut short = Store::create(&scratch.paths(), "short", encoder.origin()).unwrap();
    let report = fill(
        &mut encoder,
        &mut short,
        &part(&logs, 0, 30),
        &mut budget_with(None, Some(16)),
    );
    assert_eq!(encoder.calls, [16, 8], "24 usable texts of 30");
    assert_eq!(
        (report.plan.min_rows_used, report.plan.max_rows_used),
        (8, 16)
    );
}

#[test]
fn the_run_record_takes_the_batch_plan_and_what_was_stored() {
    let scratch = Scratch::new("record");
    let logs = fixture_logs();
    let mut encoder = small();
    let mut store = Store::create(&scratch.paths(), "fixture", encoder.origin()).unwrap();
    let report = fill(
        &mut encoder,
        &mut store,
        &logs,
        &mut budget_with(None, Some(64)),
    );

    // The record a session's `finish` returns has an object as its settings.
    let mut record = RunRecord {
        kind: RunKind::Encode,
        started_utc: "2026-11-14T10:00:00Z".to_string(),
        duration_ms: 1,
        inputs: Vec::new(),
        settings: serde_json::json!({ "runtime": { "release": RELEASE } }),
        seed: None,
        encoder: Some(encoder.identity.clone()),
        device: None,
        driver_version: None,
        logs_per_second: None,
        placement: None,
    };
    report.record(&store, &mut record).unwrap();
    let settings = &record.settings;
    assert_eq!(
        settings["runtime"]["release"], RELEASE,
        "what was there stays"
    );
    assert_eq!(settings[l2d_core::BATCH_PLAN_KEY]["batch_rows"], 64);
    let stored = &settings["store"];
    assert_eq!(stored["name"], "fixture");
    assert_eq!(stored["runtime_release"], RELEASE);
    assert_eq!(stored["logs"], 169);
    assert_eq!(stored["added"], 169);
    assert_eq!(stored["repeated"], 0);
    assert_eq!(stored["adds"], 3);
    assert_eq!(stored["batches_cut"], 0);
    assert_eq!(stored["skipped"], 0);
    assert_eq!(stored["truncated"], report.truncated);
    assert_eq!(stored["longer_than_input"], report.longer_than_input);
    assert_eq!(stored["empty_inputs"], report.empty_inputs);
    assert_eq!(stored["add_held_bytes"], report.add_held_bytes);
}

/// A budget whose free-memory reading falls to `low` bytes after `readings` readings.
fn budget_that_runs_dry(batch_rows: u64, readings: u32, low: u64) -> MemoryBudget {
    let settings = MemorySettings {
        max_batch_rows: Some(batch_rows),
        ..MemorySettings::default()
    };
    let mut read = 0u32;
    MemoryBudget::observe_with(&settings, move || {
        read += 1;
        Ok(if read > readings { low } else { 8 * GIB })
    })
    .expect("the budget is usable")
}

/// Owner decision B: whatever stops a fill, the store is as before.
#[test]
fn a_memory_refusal_or_a_store_error_in_a_later_batch_rolls_the_fill_back_too() {
    let scratch = Scratch::new("later");
    let paths = scratch.paths();
    let logs = fixture_logs();
    let mut encoder = small();
    let mut store = Store::create(&paths, "s", encoder.origin()).unwrap();
    fill(
        &mut encoder,
        &mut store,
        &part(&logs, 100, 40),
        &mut budget_with(None, None),
    );
    let rows_before = stored(&store);
    let file_before = file(&store);

    // The free memory is gone after a few batches: the budget refuses the next one.
    let mut encoder = small();
    let mut budget = budget_that_runs_dry(16, 8, 1024);
    let error = encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("no memory is free for a later batch");
    assert!(
        matches!(
            &*error.cause,
            FillCause::Memory(_) | FillCause::Store(StoreError::Memory(_))
        ),
        "{error}"
    );
    assert!(
        error.encoded >= 16 && error.encoded < 169,
        "batches were encoded before it: {}",
        error.encoded
    );
    assert!(error.to_string().contains("rolled back"), "{error}");
    assert!(matches!(L2dError::from(error), L2dError::Refused { .. }));
    assert_eq!(stored(&store), rows_before);
    assert_eq!(file(&store), file_before);
    assert_eq!(budget.held_bytes(), 0);

    // The store refuses the rows of a later batch: a vector with a value that is not a
    // number.
    let mut encoder = small();
    encoder.spoil_at = Some(3);
    let mut budget = budget_with(None, Some(16));
    let error = encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("the store does not take the fourth batch");
    assert!(
        matches!(
            &*error.cause,
            FillCause::Store(StoreError::InvalidRows { .. })
        ),
        "{error}"
    );
    assert_eq!(error.encoded, 64, "the batch the store refused was encoded");
    assert!(error.to_string().contains("rolled back"), "{error}");
    assert_eq!(stored(&store), rows_before);
    assert_eq!(file(&store), file_before);
    assert_eq!(budget.held_bytes(), 0);

    // And the store takes a fill after both.
    let mut encoder = small();
    let report = fill(&mut encoder, &mut store, &logs, &mut budget);
    assert_eq!((report.added, report.repeated), (129, 40));
}

// ---------------------------------------------------------------------------------------
// What one long transaction costs (on request)
// ---------------------------------------------------------------------------------------

/// The largest sizes the store file and its rollback journal had while `run` ran, read
/// every few milliseconds from another thread.
fn watch_files<T>(store_file: &Path, run: impl FnOnce() -> T) -> (T, u64, u64) {
    use std::sync::atomic::AtomicBool;
    let journal = PathBuf::from(format!("{}-journal", store_file.display()));
    let wal = PathBuf::from(format!("{}-wal", store_file.display()));
    let stop = AtomicBool::new(false);
    let size = |path: &Path| std::fs::metadata(path).map_or(0, |metadata| metadata.len());
    std::thread::scope(|scope| {
        let watcher = scope.spawn(|| {
            let (mut file_peak, mut journal_peak) = (0u64, 0u64);
            loop {
                let done = stop.load(Ordering::Relaxed);
                file_peak = file_peak.max(size(store_file));
                journal_peak = journal_peak.max(size(&journal)).max(size(&wal));
                if done {
                    return (file_peak, journal_peak);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        });
        let result = run();
        stop.store(true, Ordering::Relaxed);
        let (file_peak, journal_peak) = watcher.join().expect("the watcher ran");
        (result, file_peak, journal_peak)
    })
}

fn synthetic_logs(from: u64, count: u64) -> LogsFrame {
    let records: Vec<LogRecord> = (from..from + count)
        .map(|index| {
            record(
                &format!(
                    "GET /shop/item/{}/view?session={:016x}&page={}&ref=search HTTP/1.1",
                    index % 977,
                    index.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    index % 13
                ),
                index + 1,
            )
        })
        .collect();
    LogsFrame::from_records(&records).unwrap()
}

/// Figures for the notes of task 2.7 (owner decision B): what it costs that all batches
/// of a fill are one transaction. Run on request, in release:
///
/// ```text
/// cargo test --release -p l2d-decide --test encode_store -- --ignored --nocapture one_transaction_measured
/// ```
#[test]
#[ignore = "measurement of one long transaction; run in release mode on request"]
fn one_transaction_measured() {
    let scratch = Scratch::new("measured");
    let paths = scratch.paths();
    let mib = |bytes: u64| bytes as f64 / MIB as f64;
    let measure = |what: &str,
                   encoder: &mut StandIn,
                   store: &mut Store,
                   logs: &LogsFrame,
                   budget: &mut MemoryBudget| {
        let before = std::fs::metadata(store.path()).unwrap().len();
        let path = store.path().to_path_buf();
        let started = std::time::Instant::now();
        let (report, file_peak, journal_peak) =
            watch_files(&path, || fill(encoder, store, logs, budget));
        let seconds = started.elapsed().as_secs_f64();
        let after = std::fs::metadata(&path).unwrap().len();
        println!(
            "one_transaction_measured: {what}: {} logs, dimension {}, {} adds in one \
             transaction (batches of {} to {} rows); store file {:.2} MiB before, at most \
             {:.2} MiB during, {:.2} MiB after; rollback journal or WAL at most {:.3} MiB \
             ({} bytes); the store's share of the budget per add {:.2} MiB, {} bytes per row \
             planned; {seconds:.2} s",
            report.encoded,
            encoder.identity.dimension,
            report.adds,
            report.plan.min_rows_used,
            report.plan.max_rows_used,
            mib(before),
            mib(file_peak),
            mib(after),
            mib(journal_peak),
            journal_peak,
            mib(report.add_held_bytes),
            report.plan.bytes_per_row,
        );
    };

    // The fixture, in eleven batches.
    let fixture = fixture_logs();
    let mut encoder = small();
    let mut store = Store::create(&paths, "fixture", encoder.origin()).unwrap();
    measure(
        "fixture, new store",
        &mut encoder,
        &mut store,
        &fixture,
        &mut budget_with(None, Some(16)),
    );
    measure(
        "fixture again (only occurrences change)",
        &mut encoder,
        &mut store,
        &fixture,
        &mut budget_with(None, Some(16)),
    );

    // A larger fill: 200,000 logs with vectors of 64 values, then 200,000 more into the
    // store that holds the first.
    let mut encoder = stand_in(64, 8, InputMode::Payload, RELEASE);
    let mut store = Store::create(&paths, "synthetic", encoder.origin()).unwrap();
    for (what, from) in [
        ("synthetic, new store", 0u64),
        ("synthetic, 200,000 more into the same store", 200_000),
    ] {
        let logs = synthetic_logs(from, 200_000);
        let mut budget = budget_with(None, Some(16_384));
        budget.hold(logs.frame().estimated_size() as u64);
        measure(what, &mut encoder, &mut store, &logs, &mut budget);
    }
    assert_eq!(store.len().unwrap(), 400_000);

    // The same larger fill, stopped in its last batch: what a rollback leaves.
    let logs = synthetic_logs(400_000, 200_000);
    let mut budget = budget_with(None, Some(16_384));
    budget.hold(logs.frame().estimated_size() as u64);
    let before = std::fs::metadata(store.path()).unwrap().len();
    encoder.calls.clear();
    encoder.fail_at = Some(12);
    let path = store.path().to_path_buf();
    let started = std::time::Instant::now();
    let (result, file_peak, journal_peak) = watch_files(&path, || {
        encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
    });
    let error = result.expect_err("the thirteenth batch fails");
    let after = std::fs::metadata(&path).unwrap().len();
    println!(
        "one_transaction_measured: synthetic, stopped in batch 13 of 13 after {} logs were \
         encoded: store file {:.2} MiB before, at most {:.2} MiB during, {:.2} MiB after the \
         rollback; rollback journal or WAL at most {:.3} MiB; {:.2} s",
        error.encoded,
        mib(before),
        mib(file_peak),
        mib(after),
        mib(journal_peak),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(after, before, "the rollback gave the file its length back");
    assert_eq!(store.len().unwrap(), 400_000);
}

/// The tool's one encoder for the seam is the session on the NPU.
#[test]
fn the_session_on_the_npu_is_the_tools_encoder() {
    fn is_an_encoder<E: LogEncoder>() {}
    is_an_encoder::<EncoderSession>();
}
