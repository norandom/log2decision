//! `ByteCodec` (task 2.5): no allocation per text or per batch, and its throughput.
//!
//! This binary counts the heap allocations of the calling thread, so the claim "encoding
//! allocates nothing once the batch buffer exists" is a test and not a reading of the
//! code.
//!
//! The throughput measurement runs on request, in release, on the 25,608 distinct texts
//! of HTTP CSIC 2010 (`data/csic-2010/csic_database.csv` of the project, or the file
//! `L2D_CSIC_CSV` names):
//!
//! ```text
//! $env:L2D_CSIC_CSV = "D:\Source\log2decision\data\csic-2010\csic_database.csv"
//! cargo test --release -p l2d-encoder --test byte_codec_speed -- --ignored --nocapture --test-threads=1
//! ```
//!
//! It prints logs per second on one thread for text to flags alone and for text to the
//! float batch `[32, 64, 1, 256]` (bytes plus table lookup), in both input modes, next
//! to the 30,000 logs per second the NPU ran the spike model at (task 2.2).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use l2d_core::{Csic2010Adapter, InputAdapter, ProjectPaths};
use l2d_encoder::{ByteCodec, ByteTable, InputArch, InputMode};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

// SAFETY: every call is passed on to the system allocator unchanged; the counter is a
// thread-local cell without a destructor.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Allocations and reallocations of this thread so far.
fn allocations() -> u64 {
    ALLOCATIONS.with(Cell::get)
}

fn codec(mode: InputMode) -> ByteCodec {
    let table = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/science-2.1.table.bin");
    let table = ByteTable::read(&table, &InputArch::V1).unwrap();
    ByteCodec::new(InputArch::V1, table, mode).unwrap()
}

/// What one pass over the texts saw.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
struct Pass {
    texts: u64,
    batches: u64,
    last_rows: usize,
    truncated: u64,
    empty: u64,
    read_bytes: u64,
    /// A sum over the values' bits of one position per row, so the batch is used.
    check: u64,
}

/// Encodes all texts batch after batch into the one buffer, as a session will.
fn encode_all(codec: &ByteCodec, batch: &mut l2d_encoder::InputBatch, texts: &[String]) -> Pass {
    let mut pass = Pass::default();
    let mut texts = texts.iter().map(String::as_str);
    let row = codec.arch().row_values();
    loop {
        let rows = codec.fill(batch, &mut texts);
        if rows == 0 {
            break;
        }
        pass.batches += 1;
        pass.last_rows = rows;
        pass.texts += rows as u64;
        for flags in batch.texts() {
            pass.truncated += u64::from(flags.truncated);
            pass.empty += u64::from(flags.empty);
            pass.read_bytes += flags.read_bytes as u64;
        }
        // The buffer a session would lend to the runtime here.
        let values = std::hint::black_box(batch.values());
        for at in 0..rows {
            pass.check = pass
                .check
                .wrapping_add(u64::from(values[at * row].to_bits()));
        }
    }
    pass
}

#[test]
fn encoding_allocates_nothing_per_text_or_per_batch() {
    // 1,000 texts of 0 to 2,999 bytes, some without payload, some with multi-byte
    // characters: 31 full batches and one of 8.
    let texts: Vec<String> = (0..1_000usize)
        .map(|at| match at % 4 {
            0 => format!("GET /a/{at}?{}", "p=\u{e9}&".repeat(at % 600)),
            1 => format!("POST /b/{at}\n{}", "x".repeat(at * 3)),
            2 => format!("line {at} without a delimiter"),
            _ => String::new(),
        })
        .collect();
    for mode in InputMode::ALL {
        let codec = codec(mode);
        let mut batch = codec.new_batch();
        let before = allocations();
        let first = encode_all(&codec, &mut batch, &texts);
        let second = encode_all(&codec, &mut batch, &texts);
        let allocated = allocations() - before;
        assert_eq!(
            allocated,
            0,
            "{}: allocations while encoding",
            mode.as_str()
        );
        assert_eq!(first, second);
        assert_eq!(
            (first.texts, first.batches, first.last_rows),
            (1_000, 32, 8)
        );
        // The single-text calls allocate nothing either.
        let mut row = vec![0.0f32; InputArch::V1.row_values()];
        let mut ids = [0u8; 256];
        let before = allocations();
        for text in &texts {
            let a = codec.encode_text(text, &mut row);
            let b = codec.byte_ids(text, &mut ids);
            assert!(a == b && a == codec.describe(text));
        }
        assert_eq!(allocations() - before, 0);
    }
    // The counter does count: the buffer itself is two allocations (values and flags).
    let codec = codec(InputMode::Whole);
    let before = allocations();
    let batch = codec.new_batch();
    assert_eq!(allocations() - before, 2);
    drop(batch);
}

fn csic_path() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let supplied = std::env::var_os("L2D_CSIC_CSV").map(PathBuf::from);
    Csic2010Adapter::locate(&ProjectPaths::new(root), supplied.as_deref())
        .unwrap_or_else(|error| panic!("{error} (set L2D_CSIC_CSV to csic_database.csv)"))
}

/// Logs per second of the passes: (median, best).
fn rate(texts: usize, seconds: &mut [f64]) -> (f64, f64) {
    seconds.sort_by(f64::total_cmp);
    let per = |s: f64| texts as f64 / s;
    (per(seconds[seconds.len() / 2]), per(seconds[0]))
}

#[test]
#[ignore = "on request, in release: throughput on the CSIC texts"]
fn csic_throughput_of_text_to_model_input_on_one_thread() {
    const NPU_LOGS_PER_SECOND: f64 = 30_000.0;
    const PASSES: usize = 15;
    let (logs, _) = Csic2010Adapter::new().read(&csic_path()).unwrap();
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    let total_bytes: usize = texts.iter().map(String::len).sum();
    println!(
        "byte_codec_speed: {} distinct texts, {} bytes, {} build",
        texts.len(),
        total_bytes,
        if cfg!(debug_assertions) {
            "DEBUG (figures mean nothing)"
        } else {
            "release"
        }
    );
    for mode in InputMode::ALL {
        let codec = codec(mode);
        let mut batch = codec.new_batch();
        let warm = encode_all(&codec, &mut batch, &texts);

        let before = allocations();
        let mut flags_seconds = Vec::with_capacity(PASSES);
        let mut input_seconds = Vec::with_capacity(PASSES);
        let after_buffers = allocations();
        for _ in 0..PASSES {
            let started = Instant::now();
            let mut truncated = 0u64;
            for text in &texts {
                truncated += u64::from(std::hint::black_box(codec.describe(text)).truncated);
            }
            flags_seconds.push(started.elapsed().as_secs_f64());
            assert_eq!(truncated, warm.truncated);

            let started = Instant::now();
            let pass = encode_all(&codec, &mut batch, &texts);
            input_seconds.push(started.elapsed().as_secs_f64());
            assert_eq!(pass, warm, "every pass gives the same batches");
        }
        assert_eq!(
            allocations() - after_buffers,
            0,
            "allocations in {PASSES} passes"
        );
        assert_eq!(after_buffers - before, 2, "the two lists of timings");

        let n = texts.len();
        let (flags_median, flags_best) = rate(n, &mut flags_seconds);
        let (input_median, input_best) = rate(n, &mut input_seconds);
        println!(
            "byte_codec_speed: mode {}: {} batches (last {} rows), truncated {} ({:.1} %), \
             empty {} ({:.1} %), mean bytes read {:.1}",
            mode.as_str(),
            warm.batches,
            warm.last_rows,
            warm.truncated,
            100.0 * warm.truncated as f64 / n as f64,
            warm.empty,
            100.0 * warm.empty as f64 / n as f64,
            warm.read_bytes as f64 / n as f64
        );
        println!(
            "byte_codec_speed: mode {}: text to flags {:.0} logs/s (best {:.0}); text to the \
             float batch {:.0} logs/s median of {PASSES} passes (best {:.0}), {:.1} us per \
             log, {:.1} times the NPU's {:.0} logs/s; {:.2} GiB/s of input written",
            mode.as_str(),
            flags_median,
            flags_best,
            input_median,
            input_best,
            1e6 / input_median,
            input_median / NPU_LOGS_PER_SECOND,
            NPU_LOGS_PER_SECOND,
            input_median * InputArch::V1.row_values() as f64 * 4.0 / (1u64 << 30) as f64
        );
        assert_eq!(warm.texts as usize, n);
        assert_eq!(warm.batches as usize, n.div_ceil(32));
        if !cfg!(debug_assertions) {
            assert!(
                input_median > NPU_LOGS_PER_SECOND,
                "the codec is slower than the NPU it feeds: {input_median:.0} logs/s"
            );
        }
    }

    // The worst case: every text fills the input, so every value is a table lookup.
    let full: Vec<String> = (0..texts.len())
        .map(|at| format!("{at:05}{}", "abcdefghij".repeat(30)))
        .collect();
    let codec = codec(InputMode::Whole);
    let mut batch = codec.new_batch();
    let warm = encode_all(&codec, &mut batch, &full);
    assert_eq!(warm.truncated as usize, full.len());
    let mut seconds = Vec::with_capacity(PASSES);
    for _ in 0..PASSES {
        let started = Instant::now();
        let pass = encode_all(&codec, &mut batch, &full);
        seconds.push(started.elapsed().as_secs_f64());
        assert_eq!(pass, warm);
    }
    let (median, best) = rate(full.len(), &mut seconds);
    println!(
        "byte_codec_speed: worst case, {} texts of 305 bytes (256 read): {median:.0} logs/s \
         median (best {best:.0}), {:.1} us per log, {:.1} times the NPU's rate",
        full.len(),
        1e6 / median,
        median / NPU_LOGS_PER_SECOND
    );
}
