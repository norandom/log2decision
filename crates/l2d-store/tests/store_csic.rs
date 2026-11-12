//! The store measured: the memory an add really takes against what it asks of the
//! budget (requirements 12.5, 12.9), on synthetic logs in every run and on HTTP CSIC
//! 2010 on request.
//!
//! Two heaps are read. This test binary counts every allocation of the Rust side (a
//! counting global allocator around the system allocator, as
//! `l2d-decide/tests/conformal_csic.rs` does). SQLite allocates through the C runtime,
//! which that counter does not see; its own counter (`sqlite3_memory_highwater`) is read
//! for it. The page cache, the text index and the statements are all in that figure.
//!
//! The CSIC measurement runs on request only, in release mode:
//!
//! ```text
//! cargo test --release -p l2d-store --test store_csic -- --ignored --nocapture --test-threads=1
//! ```
//!
//! It is skipped with a note when `data/csic-2010/csic_database.csv` is absent. The
//! vectors are synthetic (no encoder exists yet); the store is written below the system
//! temporary directory and removed afterwards.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    Csic2010Adapter, EncoderIdentity, InputAdapter, L2dError, Label, LogRecord, LogsFrame,
    MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_store::{
    ADD_FIXED_BYTES, AddReport, IDLE_BYTES, MAX_TEXT_BYTES, Store, StoreError, log_id, rows_frame,
};
use rusqlite::{Connection, OpenFlags};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every call is forwarded unchanged to the system allocator; the counters are
// only bookkeeping beside it.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            if new_size >= layout.size() {
                grew(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The measurements share the counters: one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const MIB: f64 = 1024.0 * 1024.0;
const REVISION: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// The bytes SQLite holds now, and the most it held since the last reset.
fn sqlite_heap(reset: bool) -> (u64, u64) {
    // SAFETY: both functions only read (and, on request, reset) SQLite's allocation
    // counters; they take no pointers and may be called at any time.
    let (used, peak) = unsafe {
        (
            rusqlite::ffi::sqlite3_memory_used(),
            rusqlite::ffi::sqlite3_memory_highwater(i32::from(reset)),
        )
    };
    (used as u64, peak as u64)
}

/// What `run` took at its peak.
struct Taken {
    /// By how much the Rust heap grew beyond what was held when `run` started (the
    /// caller's frames are held then and are not the add's).
    rust_bytes: u64,
    /// The most SQLite held during `run`, in all: a connection keeps its page cache
    /// between adds, so what it held before counts too.
    sqlite_bytes: u64,
}

fn measure<T>(run: impl FnOnce() -> T) -> (T, Taken) {
    let rust_before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(rust_before, Ordering::Relaxed);
    sqlite_heap(true);
    let result = run();
    let rust_peak = PEAK.load(Ordering::Relaxed);
    let (_, sqlite_peak) = sqlite_heap(false);
    (
        result,
        Taken {
            rust_bytes: rust_peak.saturating_sub(rust_before) as u64,
            sqlite_bytes: sqlite_peak,
        },
    )
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// A project root in the system temporary directory, removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("l2d-store-csic-{}-{name}", std::process::id()));
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

fn identity(dimension: u16) -> EncoderIdentity {
    EncoderIdentity {
        name: "synthetic".to_string(),
        revision: REVISION.to_string(),
        dimension,
    }
}

/// FNV-1a over the bytes of `text`: a fixed pseudo-random number per text.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Fixed unit vectors, one per log of `logs`.
fn vectors_of(logs: &LogsFrame, dimension: u16) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(logs.len() * usize::from(dimension));
    for record in logs.records() {
        let mut state = fnv(&record.text) | 1;
        let start = vectors.len();
        for _ in 0..dimension {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vectors.push(((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0);
        }
        let norm = vectors[start..].iter().map(|v| v * v).sum::<f32>().sqrt();
        for value in &mut vectors[start..] {
            *value /= norm;
        }
    }
    vectors
}

fn rows_of(logs: &LogsFrame, source: &str, dimension: u16) -> DataFrame {
    let truncated: Vec<bool> = logs
        .records()
        .map(|record| record.text.len() > 256)
        .collect();
    rows_frame(
        logs,
        source,
        &vectors_of(logs, dimension),
        dimension,
        &truncated,
    )
    .expect("the logs make a rows frame")
}

/// The ids of the rows whose text contains `phrase`, through the FTS5 index.
fn fts_ids(path: &Path, phrase: &str) -> Vec<String> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut statement = connection
        .prepare(
            "SELECT logs.id FROM logs_fts JOIN logs ON logs.seq = logs_fts.rowid \
             WHERE logs_fts MATCH ?1 ORDER BY logs.seq",
        )
        .unwrap();
    let quoted = format!("\"{}\"", phrase.replace('"', "\"\""));
    statement
        .query_map([quoted], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// By a plain scan, ignoring ASCII case only (the index folds Unicode case; the phrases
/// used here are ASCII).
fn scan_ids(logs: &LogsFrame, source: &str, phrase: &str) -> Vec<String> {
    let phrase = phrase.to_ascii_lowercase();
    logs.records()
        .filter(|record| record.text.to_ascii_lowercase().contains(&phrase))
        .map(|record| log_id(source, &record.text))
        .collect()
}

fn assert_fts_consistent(path: &Path) {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)
        .unwrap()
        .execute(
            "INSERT INTO logs_fts(logs_fts, rank) VALUES ('integrity-check', 1)",
            [],
        )
        .expect("the FTS5 index agrees with the logs table");
}

/// What the add asked of the budget: its own share (the fixed part and the part for its
/// longest text) and one batch.
fn asked(report: &AddReport) -> u64 {
    report.held_bytes + report.plan.bytes_per_row * report.plan.max_rows_used.min(report.rows)
}

fn print_taken(what: &str, report: &AddReport, taken: &Taken) {
    println!(
        "{what}: SQLite heap peak {:.2} MiB (fixed share {:.0} MiB), Rust heap peak {:.2} MiB, \
         together {:.2} MiB of the {:.2} MiB asked of the budget ({} bytes per row, batches of \
         up to {} rows)",
        taken.sqlite_bytes as f64 / MIB,
        ADD_FIXED_BYTES as f64 / MIB,
        taken.rust_bytes as f64 / MIB,
        (taken.sqlite_bytes + taken.rust_bytes) as f64 / MIB,
        asked(report) as f64 / MIB,
        report.plan.bytes_per_row,
        report.plan.max_rows_used,
    );
}

/// The fixed share covers what SQLite takes for short texts, and the whole add stays
/// within what it asked of the budget, also when one transaction is several times the
/// page cache. On request: 12,000 rows are slow in a debug build.
#[test]
#[ignore = "12,000-row measurement; run in release mode on request"]
fn an_add_takes_no_more_memory_than_it_asks_of_the_budget() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // 12,000 logs of about 400 bytes with 128-dimension vectors: a database file of
    // about 23 MiB, nearly three times the page cache, written in twelve batches.
    let records: Vec<LogRecord> = (0..12_000u32)
        .map(|index| LogRecord {
            text: format!(
                "POST http://localhost:8080/shop/pay.jsp?item={index}\n{}",
                (0..12)
                    .map(|field| format!("field{field}={:016x}", fnv(&format!("{index}/{field}"))))
                    .collect::<Vec<_>>()
                    .join("&")
            ),
            label: if index % 3 == 0 {
                Label::Bad
            } else {
                Label::Good
            },
            group: Some(format!("/shop/{}", index % 40)),
            first_position: u64::from(index) + 1,
            occurrences: 1,
        })
        .collect();
    let logs = LogsFrame::from_records(&records).unwrap();
    drop(records);
    let rows = rows_of(&logs, "synthetic", 128);

    {
        let batch_rows = Some(1_024);
        let scratch = Scratch::new("synthetic");
        let paths = scratch.paths();
        let mut store = Store::create(&paths, "s", &identity(128)).unwrap();
        let settings = MemorySettings {
            max_batch_rows: batch_rows,
            ..MemorySettings::default()
        };
        // A fixed reading in place of the host's: 8 GiB free, so a limit of 4 GiB.
        let mut budget =
            MemoryBudget::observe_with(&settings, || Ok(8 << 30)).expect("the budget is usable");

        // All or nothing, also when the transaction has long outgrown the page cache
        // and most of it is in the file already: the last of the 12,000 rows cannot be
        // stored, and the store is as it was.
        let empty = std::fs::read(paths.store_path("s")).unwrap();
        let mut vectors = vectors_of(&logs, 128);
        *vectors.last_mut().unwrap() = f32::NAN;
        let truncated = vec![false; logs.len()];
        let spoiled = rows_frame(&logs, "synthetic", &vectors, 128, &truncated).unwrap();
        drop(vectors);
        match store.add_with(&spoiled, &identity(128), &mut budget) {
            Err(StoreError::InvalidRows { detail }) => {
                assert!(detail.contains("row 12000"), "{detail}");
            }
            other => panic!("expected InvalidRows, got {other:?}"),
        }
        drop(spoiled);
        assert_eq!(store.len().unwrap(), 0);
        assert_eq!(std::fs::read(paths.store_path("s")).unwrap(), empty);

        let started = Instant::now();
        let (report, taken) = measure(|| store.add_with(&rows, &identity(128), &mut budget));
        let report = report.expect("the logs are added");
        let seconds = started.elapsed().as_secs_f64();
        drop(store);
        let file = std::fs::metadata(paths.store_path("s")).unwrap().len();
        print_taken(
            &format!("12,000 synthetic logs, batch {batch_rows:?}"),
            &report,
            &taken,
        );
        println!("  {seconds:.2} s, file {:.1} MiB", file as f64 / MIB);
        assert_eq!((report.added, report.repeated), (12_000, 0));
        assert!(
            file > 2 * l2d_store::CACHE_BYTES,
            "the test is meant to write more than the page cache holds"
        );
        assert!(
            taken.sqlite_bytes > 0,
            "SQLite's allocation counters are off; the measurement says nothing"
        );
        assert!(
            taken.sqlite_bytes <= ADD_FIXED_BYTES,
            "SQLite took {} bytes, the fixed share is {ADD_FIXED_BYTES}",
            taken.sqlite_bytes
        );
        assert!(
            taken.sqlite_bytes + taken.rust_bytes <= asked(&report),
            "the add took {} bytes and asked the budget for {}",
            taken.sqlite_bytes + taken.rust_bytes,
            asked(&report)
        );
        // Nothing of the frame is copied: the Rust side stays far below one batch.
        assert!(
            taken.rust_bytes <= 1024 * 1024,
            "the Rust side took {} bytes",
            taken.rust_bytes
        );
    }
}

/// The next number of a fixed pseudo-random sequence.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// What a long text is made of.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    /// Request lines from a small vocabulary, as a log has them: few distinct trigrams.
    Realistic,
    /// Random printable ASCII, 90 symbols: at most 729,000 distinct trigrams.
    Noisy,
    /// Random two-byte characters (U+0100 to U+07FF): nearly every trigram is new, and
    /// there is one for every two bytes.
    Distinct,
    /// The worst case for the text index: ASCII, one trigram per byte, and no trigram
    /// repeats for as long as there are new ones (101 symbols that differ after case
    /// folding: 1,030,301 trigrams). The text counts upwards in base 101, three symbols
    /// to a number.
    Counting,
}

fn long_text(kind: Kind, bytes: usize, seed: u64) -> String {
    const WORDS: [&str; 12] = [
        "id",
        "nombre",
        "precio",
        "cantidad",
        "login",
        "pwd",
        "modo",
        "entrar",
        "Vino+Rioja",
        "Jamon+Iberico",
        "B1=Anadir+al+carrito",
        "remember=on",
    ];
    let mut state = seed | 1;
    let mut text = String::with_capacity(bytes + 128);
    while text.len() < bytes {
        match kind {
            Kind::Realistic => {
                let n = next(&mut state);
                text.push_str(&format!(
                    "GET http://localhost:8080/tienda1/publico/anadir.jsp?{}={}&{}={}&{}\n",
                    WORDS[(n % 12) as usize],
                    n % 100_000,
                    WORDS[((n >> 8) % 12) as usize],
                    WORDS[((n >> 16) % 12) as usize],
                    WORDS[((n >> 24) % 12) as usize],
                ));
            }
            Kind::Noisy => text.push(char::from(b'!' + (next(&mut state) % 90) as u8)),
            Kind::Distinct => text.push(
                char::from_u32(0x100 + (next(&mut state) % 0x700) as u32)
                    .expect("the range holds characters only"),
            ),
            Kind::Counting => {
                // The bytes 1 to 127 without the capital letters, which fold to the small
                // ones.
                let symbol = |digit: usize| {
                    let byte = 1 + digit as u8;
                    char::from(if byte >= b'A' { byte + 26 } else { byte })
                };
                let number = text.len() / 3 + seed as usize;
                text.push(symbol(number / (101 * 101) % 101));
                text.push(symbol(number / 101 % 101));
                text.push(symbol(number % 101));
            }
        }
    }
    while text.len() > bytes {
        text.pop();
    }
    text
}

/// 200 short logs and then `long` texts of `bytes` bytes of `kind`, with 8-dimension
/// vectors.
fn rows_with_long_texts(kind: Kind, bytes: usize, long: usize) -> DataFrame {
    let mut records: Vec<LogRecord> = (0..200u64)
        .map(|index| LogRecord {
            text: format!(
                "GET /short/{index}?x={}",
                long_text(Kind::Noisy, 60, index + 7)
            ),
            label: Label::Good,
            group: None,
            first_position: index + 1,
            occurrences: 1,
        })
        .collect();
    for index in 0..long as u64 {
        records.push(LogRecord {
            text: long_text(kind, bytes, 1_000 + index),
            label: Label::Bad,
            group: None,
            first_position: 201 + index,
            occurrences: 1,
        });
    }
    let logs = LogsFrame::from_records(&records).unwrap();
    drop(records);
    rows_of(&logs, "long", 8)
}

/// A budget that does not depend on the host: 64 GiB free, so a limit of 32 GiB.
fn roomy_budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(64 << 30))
        .expect("the budget is usable")
}

/// The bytes SQLite holds now.
fn sqlite_used() -> u64 {
    sqlite_heap(false).0
}

/// 300 short logs from `source`, with 8-dimension vectors.
fn short_rows(source: &str) -> DataFrame {
    let records: Vec<LogRecord> = (0..300u64)
        .map(|index| LogRecord {
            text: format!(
                "GET /later/{index}?x={}",
                long_text(Kind::Noisy, 60, index + 11)
            ),
            label: Label::Good,
            group: None,
            first_position: index + 1,
            occurrences: 1,
        })
        .collect();
    rows_of(&LogsFrame::from_records(&records).unwrap(), source, 8)
}

/// An open store gives back what a long text took: after the add SQLite holds no more
/// than the store says it retains, and a later add of short texts on the same open
/// store stays within the share it held. (Before the fix the text index kept its table
/// of the long text until the store was dropped, counted by nobody.)
#[test]
fn an_open_store_gives_the_memory_of_a_long_text_back() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(
        sqlite_used(),
        0,
        "another connection is open in this process"
    );
    let scratch = Scratch::new("give-back");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
    let mut budget = roomy_budget();
    let left = budget.left_bytes();

    let long = rows_with_long_texts(Kind::Distinct, 256 * 1024, 1);
    let (first, taken) = measure(|| store.add_with(&long, &identity(8), &mut budget));
    let first = first.expect("the long text is added");
    let idle = sqlite_used();
    println!(
        "first add (one text of 256 KiB): SQLite heap peak {:.1} MiB, share held {:.1} MiB; \
         in use afterwards {:.2} MiB, the store says it retains at most {:.2} MiB",
        taken.sqlite_bytes as f64 / MIB,
        first.held_bytes as f64 / MIB,
        idle as f64 / MIB,
        store.retained_bytes() as f64 / MIB,
    );
    assert!(taken.sqlite_bytes <= first.held_bytes);
    assert!(
        idle <= store.retained_bytes(),
        "after the add SQLite holds {idle} bytes; the store says it retains at most {}",
        store.retained_bytes()
    );
    assert!(store.retained_bytes() <= IDLE_BYTES);
    assert_eq!(budget.left_bytes(), left);

    // The same open store, short texts only.
    let (second, taken) =
        measure(|| store.add_with(&short_rows("later"), &identity(8), &mut budget));
    let second = second.expect("the short texts are added");
    println!(
        "second add on the same open store (300 short texts): SQLite heap peak {:.2} MiB, \
         share held {:.2} MiB; in use afterwards {:.2} MiB",
        taken.sqlite_bytes as f64 / MIB,
        second.held_bytes as f64 / MIB,
        sqlite_used() as f64 / MIB,
    );
    assert!(second.held_bytes < ADD_FIXED_BYTES + 64 * 1024);
    assert!(
        taken.sqlite_bytes <= second.held_bytes,
        "the second add took {} bytes of SQLite's heap and held {}",
        taken.sqlite_bytes,
        second.held_bytes
    );
    assert!(sqlite_used() <= store.retained_bytes());
    assert_eq!(store.len().unwrap(), 501);

    // A failed add gives back as well: the long text is indexed, then the last row fails.
    let logs = LogsFrame::from_records(&[
        LogRecord {
            text: long_text(Kind::Distinct, 256 * 1024, 77),
            label: Label::Good,
            group: None,
            first_position: 1,
            occurrences: 1,
        },
        LogRecord {
            text: "the vector of this row is not finite".to_string(),
            label: Label::Good,
            group: None,
            first_position: 2,
            occurrences: 1,
        },
    ])
    .unwrap();
    let mut vectors = vectors_of(&logs, 8);
    *vectors.last_mut().unwrap() = f32::INFINITY;
    let spoiled = rows_frame(&logs, "spoiled", &vectors, 8, &[false, false]).unwrap();
    assert!(matches!(
        store.add_with(&spoiled, &identity(8), &mut budget),
        Err(StoreError::InvalidRows { .. })
    ));
    assert!(
        sqlite_used() <= store.retained_bytes(),
        "after the failed add SQLite holds {} bytes",
        sqlite_used()
    );
    assert_eq!(store.len().unwrap(), 501);

    drop(store);
    assert_eq!(sqlite_used(), 0);
    assert_fts_consistent(&paths.store_path("s"));
}

/// What an open connection keeps after it indexed one long text, and what each means
/// of release gives back: the figures behind the release at the end of every add.
///
/// The first columns are a plain connection that writes one row as the store does and
/// releases nothing: the store before the fix. The last columns are the store.
#[test]
#[ignore = "retained-memory measurement; run in release mode on request"]
fn retained_memory_measured() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    println!(
        "text                 | plain connection, MiB in use: after commit | after shrink_memory \
         | after statement cache flush | after close || store, MiB: after the add | says it \
         retains | peak of a later short add | share that add held"
    );
    let no_repeat = no_repeat_text((1 << 20) + 64);
    let cases: [(&str, String); 6] = [
        ("log-like 1 MiB", long_text(Kind::Realistic, 1 << 20, 1)),
        ("random ASCII 1 MiB", long_text(Kind::Noisy, 1 << 20, 1)),
        ("two-byte 1 MiB", long_text(Kind::Distinct, 1 << 20, 1)),
        ("no repeat 2^20+64", no_repeat),
        ("two-byte 8 MiB", long_text(Kind::Distinct, 8 << 20, 1)),
        ("short texts only", String::from("GET /a?x=1")),
    ];
    let mut failures = Vec::new();
    for (label, text) in &cases {
        // The plain connection.
        let scratch = Scratch::new("retained-plain");
        let paths = scratch.paths();
        drop(Store::create(&paths, "s", &identity(8)).unwrap());
        assert_eq!(sqlite_used(), 0);
        let connection =
            Connection::open_with_flags(paths.store_path("s"), OpenFlags::SQLITE_OPEN_READ_WRITE)
                .unwrap();
        connection
            .pragma_update(
                None,
                "cache_size",
                -((l2d_store::CACHE_BYTES / 1024) as i64),
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO logs (id, text, source, occurrences, label, group_key, truncated, \
                 vector, encoder_revision) VALUES ('x', ?1, 's', 1, 'good', NULL, 0, ?2, 'r')",
                rusqlite::params![text, vec![0u8; 32]],
            )
            .unwrap();
        let after_commit = sqlite_used();
        connection.execute_batch("PRAGMA shrink_memory").unwrap();
        let after_shrink = sqlite_used();
        connection.flush_prepared_statement_cache();
        let after_flush = sqlite_used();
        drop(connection);
        let after_close = sqlite_used();
        drop(scratch);

        // The store.
        let scratch = Scratch::new("retained-store");
        let paths = scratch.paths();
        let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
        let mut budget = roomy_budget();
        let logs = LogsFrame::from_records(&[LogRecord {
            text: text.clone(),
            label: Label::Good,
            group: None,
            first_position: 1,
            occurrences: 1,
        }])
        .unwrap();
        store
            .add_with(&rows_of(&logs, "long", 8), &identity(8), &mut budget)
            .expect("the text is added");
        let idle = sqlite_used();
        let retains = store.retained_bytes();
        let (later, taken) =
            measure(|| store.add_with(&short_rows("later"), &identity(8), &mut budget));
        let later = later.expect("the short texts are added");
        drop(store);
        println!(
            "{label:20} | {:42.2} | {:19.2} | {:27.2} | {:11.2} || {:26.2} | {:15.2} | {:25.2} | {:.2}",
            after_commit as f64 / MIB,
            after_shrink as f64 / MIB,
            after_flush as f64 / MIB,
            after_close as f64 / MIB,
            idle as f64 / MIB,
            retains as f64 / MIB,
            taken.sqlite_bytes as f64 / MIB,
            later.held_bytes as f64 / MIB,
        );
        assert_eq!(after_close, 0);
        // The whole table is printed before anything fails.
        if idle > retains || retains > IDLE_BYTES {
            failures.push(format!(
                "{label}: {idle} bytes in use after the add, the store says {retains}"
            ));
        }
        if taken.sqlite_bytes > later.held_bytes {
            failures.push(format!(
                "{label}: the later add took {} bytes and held {}",
                taken.sqlite_bytes, later.held_bytes
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The De Bruijn sequence B(k, 3) as symbol numbers: every triple of `k` symbols once,
/// read cyclically (the standard recursive construction from Lyndon words).
fn de_bruijn(k: usize) -> Vec<usize> {
    fn extend(t: usize, p: usize, k: usize, word: &mut [usize; 4], out: &mut Vec<usize>) {
        if t > 3 {
            if 3 % p == 0 {
                out.extend_from_slice(&word[1..=p]);
            }
        } else {
            word[t] = word[t - p];
            extend(t + 1, p, k, word, out);
            for symbol in word[t - p] + 1..k {
                word[t] = symbol;
                extend(t + 1, t, k, word, out);
            }
        }
    }
    let mut out = Vec::with_capacity(k * k * k);
    extend(1, 1, k, &mut [0; 4], &mut out);
    out
}

/// A text with `trigrams` trigrams of which none repeats, also after case folding: the
/// worst case for the text index. Single-byte characters for as long as there are new
/// trigrams of them (101 symbols, 1,030,301 trigrams), then two-byte letters without
/// case (at most 32,766 more).
fn no_repeat_text(trigrams: usize) -> String {
    let ascii = de_bruijn(101);
    let symbol = |digit: usize| {
        let byte = 1 + digit as u8;
        char::from(if byte >= b'A' { byte + 26 } else { byte })
    };
    let from_ascii = trigrams.min(ascii.len());
    let mut text = String::with_capacity(trigrams + 2);
    // n + 2 symbols in a row give n trigrams; the sequence is cyclic.
    for index in 0..from_ascii + 2 {
        text.push(symbol(ascii[index % ascii.len()]));
    }
    if trigrams > from_ascii {
        let more = trigrams - from_ascii;
        let letters = de_bruijn(32);
        assert!(
            more + 2 <= letters.len(),
            "no more than 32,766 further trigrams"
        );
        for digit in &letters[..more + 2] {
            text.push(char::from_u32(0x620 + *digit as u32).expect("an Arabic letter"));
        }
    }
    text
}

/// One long text among short ones: the text index builds a whole text in memory, so
/// SQLite's heap follows the longest text, not the average. The add's own share of the
/// budget covers it, where the fixed share alone would not.
#[test]
fn a_long_text_among_short_ones_stays_within_the_adds_own_share() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (kind, bytes) in [(Kind::Noisy, 256 * 1024), (Kind::Distinct, 256 * 1024)] {
        let rows = rows_with_long_texts(kind, bytes, 1);
        let scratch = Scratch::new("long-row");
        let paths = scratch.paths();
        let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
        let mut budget = roomy_budget();
        let (report, taken) = measure(|| store.add_with(&rows, &identity(8), &mut budget));
        let report = report.expect("the rows are added");
        drop(store);
        println!(
            "{kind:?}, one text of {bytes} bytes among 200 short: SQLite heap peak {:.1} MiB, \
             own share {:.1} MiB, with one batch {:.1} MiB",
            taken.sqlite_bytes as f64 / MIB,
            report.held_bytes as f64 / MIB,
            asked(&report) as f64 / MIB,
        );
        assert_eq!(report.added, 201);
        assert_eq!(report.longest_text_bytes, bytes as u64);
        assert!(
            taken.sqlite_bytes > ADD_FIXED_BYTES,
            "the test is meant for a text that the fixed share does not cover; SQLite took {}",
            taken.sqlite_bytes
        );
        assert!(
            taken.sqlite_bytes <= report.held_bytes,
            "SQLite took {} bytes; the add held {} of the budget",
            taken.sqlite_bytes,
            report.held_bytes
        );
        assert_fts_consistent(&paths.store_path("s"));
    }
}

/// SQLite's heap against the length of the longest text, for the three kinds of text:
/// the table behind `ADD_TEXT_FACTOR` and `MAX_TEXT_BYTES`.
#[test]
#[ignore = "long-text measurement; run in release mode on request"]
fn long_texts_measured() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    const KIB: usize = 1024;
    let longest = usize::try_from(MAX_TEXT_BYTES).unwrap();
    // One long text of each length up to the longest that is stored, and sixteen texts
    // of a mebibyte: what one long text takes is given back before the next.
    let ladder: Vec<(usize, usize)> = [
        400,
        4 * KIB,
        16 * KIB,
        64 * KIB,
        256 * KIB,
        KIB * KIB,
        4 * KIB * KIB,
        8 * KIB * KIB,
        16 * KIB * KIB,
        32 * KIB * KIB,
        64 * KIB * KIB,
    ]
    .into_iter()
    .filter(|bytes| *bytes <= longest)
    .map(|bytes| (bytes, 1))
    .chain([(KIB * KIB, 16)])
    .collect();
    println!(
        "kind      | long texts | text bytes | SQLite peak MiB | own share MiB | peak / share | \
         peak bytes per text byte | seconds"
    );
    for kind in [Kind::Realistic, Kind::Noisy, Kind::Distinct, Kind::Counting] {
        for &(bytes, count) in &ladder {
            let rows = rows_with_long_texts(kind, bytes, count);
            let scratch = Scratch::new("long");
            let paths = scratch.paths();
            let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
            let mut budget = roomy_budget();
            let started = Instant::now();
            let (report, taken) = measure(|| store.add_with(&rows, &identity(8), &mut budget));
            let seconds = started.elapsed().as_secs_f64();
            let report = report.expect("the rows are added");
            drop(store);
            println!(
                "{:9} | {count:10} | {bytes:10} | {:15.1} | {:13.1} | {:12.2} | {:24.1} | {seconds:.2}",
                format!("{kind:?}"),
                taken.sqlite_bytes as f64 / MIB,
                report.held_bytes as f64 / MIB,
                taken.sqlite_bytes as f64 / report.held_bytes as f64,
                taken.sqlite_bytes as f64 / bytes as f64,
            );
            assert!(
                taken.sqlite_bytes <= report.held_bytes,
                "{kind:?}, {count} texts of {bytes} bytes: SQLite took {} bytes; the add held \
                 {} of the budget",
                taken.sqlite_bytes,
                report.held_bytes
            );
        }
    }

    // The worst case: texts in which no trigram repeats, just below and just above a
    // power of two of trigrams. The index doubles its table when the trigrams of a text
    // pass a power of two, so the heap steps up just above one; the ladder of round
    // lengths above stops just short of each step.
    let mut worst: f64 = 0.0;
    for (trigrams, count) in [
        ((1usize << 14) + 64, 1usize),
        ((1 << 16) - 64, 1),
        ((1 << 16) + 64, 1),
        ((1 << 18) - 64, 1),
        ((1 << 18) + 64, 1),
        ((1 << 19) - 64, 1),
        ((1 << 19) + 64, 1),
        (1_030_301, 1),
        ((1 << 20) + 64, 1),
        ((1 << 18) + 64, 8),
    ] {
        let text = no_repeat_text(trigrams);
        let bytes = text.len();
        let mut records: Vec<LogRecord> = (0..200u64)
            .map(|index| LogRecord {
                text: format!("GET /short/{index}"),
                label: Label::Good,
                group: None,
                first_position: index + 1,
                occurrences: 1,
            })
            .collect();
        for index in 0..count {
            records.push(LogRecord {
                // The same trigrams in each, and a different text.
                text: format!("{text}{index}"),
                label: Label::Bad,
                group: None,
                first_position: 201 + index as u64,
                occurrences: 1,
            });
        }
        let rows = rows_of(&LogsFrame::from_records(&records).unwrap(), "long", 8);
        drop(records);
        let scratch = Scratch::new("long");
        let paths = scratch.paths();
        let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
        let mut budget = roomy_budget();
        let started = Instant::now();
        let (report, taken) = measure(|| store.add_with(&rows, &identity(8), &mut budget));
        let seconds = started.elapsed().as_secs_f64();
        let report = report.expect("the rows are added");
        drop(store);
        let used = taken.sqlite_bytes as f64 / report.held_bytes as f64;
        worst = worst.max(used);
        println!(
            "NoRepeat  | {count:10} | {bytes:10} | {:15.1} | {:13.1} | {used:12.2} | {:24.1} | {seconds:.2}",
            taken.sqlite_bytes as f64 / MIB,
            report.held_bytes as f64 / MIB,
            taken.sqlite_bytes as f64 / bytes as f64,
        );
        assert!(
            taken.sqlite_bytes <= report.held_bytes,
            "{count} texts of {trigrams} trigrams without a repeat: SQLite took {} bytes; \
             the add held {} of the budget",
            taken.sqlite_bytes,
            report.held_bytes
        );
    }
    println!("the most that was used of a share held, texts without a repeat: {worst:.3}");
}

/// The CSIC logs frame, or `None` with a note when the dataset is absent.
fn csic(test: &str) -> Option<LogsFrame> {
    let path = match Csic2010Adapter::locate(&ProjectPaths::new(repository()), None) {
        Ok(path) => path,
        Err(error @ L2dError::Missing { .. }) => {
            eprintln!("SKIPPED {test}: {error}");
            return None;
        }
        Err(other) => panic!("locating the dataset failed: {other:?}"),
    };
    let (logs, _) = Csic2010Adapter::new()
        .read(&path)
        .expect("the dataset is readable");
    Some(logs)
}

/// Populates a store with the 25,608 distinct logs of HTTP CSIC 2010 and synthetic
/// 128-dimension vectors, restarts, and prints time, file size and memory.
#[test]
#[ignore = "CSIC measurement; run in release mode on request"]
fn csic_populates_a_store_within_its_memory_share() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(logs) = csic("csic_populates_a_store_within_its_memory_share") else {
        return;
    };
    const SOURCE: &str = "data/csic-2010/csic_database.csv";
    const DIMENSION: u16 = 128;
    assert_eq!(logs.len(), 25_608);
    let text_bytes: usize = logs.records().map(|record| record.text.len()).sum();
    let rows = rows_of(&logs, SOURCE, DIMENSION);

    let scratch = Scratch::new("csic");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "csic", &identity(DIMENSION)).unwrap();
    let mut budget =
        MemoryBudget::observe(&MemorySettings::default()).expect("the host has free memory");

    let started = Instant::now();
    let (report, taken) = measure(|| store.add_with(&rows, &identity(DIMENSION), &mut budget));
    let report = report.expect("the logs are added");
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!(
        (report.rows, report.added, report.repeated),
        (25_608, 25_608, 0)
    );
    println!(
        "CSIC 2010: 25,608 logs ({:.1} MiB of text, {:.1} MiB of vectors) added in {seconds:.2} s \
         ({:.0} logs/s)",
        text_bytes as f64 / MIB,
        (25_608 * usize::from(DIMENSION) * 4) as f64 / MIB,
        25_608.0 / seconds
    );
    println!("  plan: {}", report.plan);
    println!(
        "  longest text {} bytes; own share of the budget {:.2} MiB",
        report.longest_text_bytes,
        report.held_bytes as f64 / MIB
    );
    print_taken("  first add", &report, &taken);
    assert!(
        taken.sqlite_bytes > 0,
        "SQLite's allocation counters are off"
    );
    assert!(taken.sqlite_bytes <= ADD_FIXED_BYTES);
    assert!(taken.sqlite_bytes + taken.rust_bytes <= asked(&report));

    // The same logs again: every id is stored already.
    let started = Instant::now();
    let (again, taken) = measure(|| store.add_with(&rows, &identity(DIMENSION), &mut budget));
    let again = again.expect("the logs are added again");
    println!(
        "  second add of the same logs: {:.2} s, all {} repeated",
        started.elapsed().as_secs_f64(),
        again.repeated
    );
    print_taken("  second add", &again, &taken);
    assert_eq!((again.added, again.repeated), (0, 25_608));
    assert!(taken.sqlite_bytes <= ADD_FIXED_BYTES);
    drop(store);

    let path = paths.store_path("csic");
    println!(
        "  file: {:.1} MiB",
        std::fs::metadata(&path).unwrap().len() as f64 / MIB
    );

    // Restart.
    let started = Instant::now();
    let store = Store::open(&paths, "csic").expect("the store opens again");
    assert_eq!(store.len().unwrap(), 25_608);
    assert_eq!(store.identity(), &identity(DIMENSION));
    drop(store);
    println!(
        "  reopened and counted in {:.3} s",
        started.elapsed().as_secs_f64()
    );

    let connection = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let (occurrences, labelled_bad): (i64, i64) = connection
        .query_row(
            "SELECT sum(occurrences), sum(label = 'bad') FROM logs",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    let wanted: i64 = logs
        .records()
        .map(|record| i64::from(record.occurrences))
        .sum();
    assert_eq!(occurrences, 2 * wanted, "each log was added twice");
    assert_eq!(labelled_bad, 15_964);
    drop(connection);

    for phrase in [
        "/tienda1/publico/anadir.jsp",
        "DROP+TABLE",
        "%3Cscript%3E",
        "alert%28",
    ] {
        let started = Instant::now();
        let found = fts_ids(&path, phrase);
        let seconds = started.elapsed().as_secs_f64();
        assert!(!found.is_empty(), "'{phrase}' matches nothing");
        assert_eq!(found, scan_ids(&logs, SOURCE, phrase), "'{phrase}'");
        println!(
            "  text index: '{phrase}' is in {} logs ({seconds:.3} s)",
            found.len()
        );
    }
    let started = Instant::now();
    assert_fts_consistent(&path);
    println!(
        "  text index agrees with the logs table (checked in {:.2} s)",
        started.elapsed().as_secs_f64()
    );
}
