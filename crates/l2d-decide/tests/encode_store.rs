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
        self.encoder.encode_into(texts, vectors, flags)
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
    vector: Vec<u32>,
}

fn stored(store: &Store) -> Vec<Row> {
    let frame: DataFrame = store.vectors(&Filter::default()).unwrap();
    let text = frame.column(l2d_store::TEXT).unwrap().str().unwrap();
    let source = frame.column(l2d_store::SOURCE).unwrap().str().unwrap();
    let occurrences = frame.column(l2d_store::OCCURRENCES).unwrap().u32().unwrap();
    let label = frame.column(l2d_store::LABEL).unwrap().str().unwrap();
    let group = frame.column(l2d_store::GROUP).unwrap().str().unwrap();
    let truncated = frame.column(l2d_store::TRUNCATED).unwrap().bool().unwrap();
    let vector = frame.column(l2d_store::VECTOR).unwrap().list().unwrap();
    (0..frame.height())
        .map(|row| Row {
            text: text.get(row).unwrap().to_string(),
            source: source.get(row).unwrap().to_string(),
            occurrences: occurrences.get(row).unwrap(),
            label: label.get(row).unwrap().to_string(),
            group: group.get(row).map(str::to_string),
            truncated: truncated.get(row).unwrap(),
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
        assert_eq!(row.truncated, expected.input.truncated, "{}", row.text);
    }

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
            FillCause::Store(StoreError::ReleaseMismatch { store, given })
                if store.as_deref() == Some(RELEASE) && given.as_deref() == Some(OTHER_RELEASE)
        ),
        "{error}"
    );
    assert_eq!(error.stored_logs, 0);
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

#[test]
fn an_encoder_that_fails_part_way_leaves_the_batches_before_it_and_says_so() {
    let scratch = Scratch::new("failure");
    let logs = fixture_logs();
    let mut encoder = small();
    encoder.fail_at = Some(2);
    let mut store = Store::create(&scratch.paths(), "s", encoder.origin()).unwrap();
    let mut budget = budget_with(None, Some(16));
    let error = encode_into_store(&mut encoder, &mut store, &logs, SOURCE, &mut budget)
        .expect_err("the third batch fails");
    assert!(
        matches!(
            &*error.cause,
            FillCause::Encoder(EncoderError::Failed { .. })
        ),
        "{error}"
    );
    assert_eq!((error.stored_logs, error.logs), (32, 169));
    let message = error.to_string();
    assert!(message.contains("call 2 failed"), "{message}");
    assert!(message.contains("32 of the 169"), "{message}");
    assert_eq!(store.len().unwrap(), 32, "two batches of 16 stay stored");
    assert_eq!(budget.held_bytes(), 0);
    let as_tool_error = L2dError::from(error);
    assert!(
        matches!(&as_tool_error, L2dError::Runtime { detail } if detail.contains("32 of the 169")),
        "{as_tool_error}"
    );

    // A failure in the first batch stores nothing, and says that too.
    let mut encoder = small();
    encoder.fail_at = Some(0);
    let mut empty = Store::create(&scratch.paths(), "empty", encoder.origin()).unwrap();
    let error = encode_into_store(&mut encoder, &mut empty, &logs, SOURCE, &mut budget)
        .expect_err("the first batch fails");
    assert_eq!(error.stored_logs, 0);
    assert!(error.to_string().contains("nothing"), "{error}");
    assert!(empty.is_empty().unwrap());
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
    assert_eq!(stored["skipped"], 0);
    assert_eq!(stored["truncated"], report.truncated);
    assert_eq!(stored["longer_than_input"], report.longer_than_input);
    assert_eq!(stored["empty_inputs"], report.empty_inputs);
    assert_eq!(stored["add_held_bytes"], report.add_held_bytes);
}

/// The tool's one encoder for the seam is the session on the NPU.
#[test]
fn the_session_on_the_npu_is_the_tools_encoder() {
    fn is_an_encoder<E: LogEncoder>() {}
    is_an_encoder::<EncoderSession>();
}
