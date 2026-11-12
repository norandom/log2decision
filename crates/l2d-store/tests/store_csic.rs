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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    Csic2010Adapter, EncoderIdentity, InputAdapter, L2dError, Label, LogRecord, LogsFrame,
    MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_store::{
    ADD_FIXED_BYTES, ADD_TEXT_FACTOR, AddReport, DISTANCE, Filter, IDLE_BYTES, INDEX_TEXT_FACTOR,
    MAX_TEXT_BYTES, Nearest, PHRASE_BYTE_BYTES, RELEASE_TEXT_BYTES, RETAINED_TEXT_FACTOR,
    SCHEMA_VERSION, SEARCH_FIXED_BYTES, Store, StoreError, TEXT, TEXT_READ_FACTOR, log_id,
    rows_frame,
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

/// The tests of this file share two sets of counters, the allocation counters above and
/// SQLite's own, so they run one at a time. That is enforced here, in one place: a
/// scratch directory (and so a store) and every reading of the counters need this guard,
/// and only [`alone`] makes one. A test added to this file cannot open a store or
/// measure beside a running measurement without it.
///
/// SQLite's figures are read as differences from what SQLite held when the guard was
/// taken, so that a connection left open elsewhere in the process shifts no figure.
struct Alone {
    _guard: MutexGuard<'static, ()>,
    /// What SQLite held when this test got its turn.
    sqlite_before: u64,
}

fn alone() -> Alone {
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
    let guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // SAFETY: reads SQLite's allocation counter; no pointers, callable at any time.
    let sqlite_before = unsafe { rusqlite::ffi::sqlite3_memory_used() } as u64;
    Alone {
        _guard: guard,
        sqlite_before,
    }
}

impl Alone {
    /// The bytes SQLite holds now, beyond what it held when this test started.
    fn sqlite_used(&self) -> u64 {
        sqlite_heap(self, false).0
    }
}

const MIB: f64 = 1024.0 * 1024.0;
const REVISION: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// The bytes SQLite holds now, and the most it held since the last reset, both beyond
/// what it held when the test started.
fn sqlite_heap(one: &Alone, reset: bool) -> (u64, u64) {
    // SAFETY: both functions only read (and, on request, reset) SQLite's allocation
    // counters; they take no pointers and may be called at any time.
    let (used, peak) = unsafe {
        (
            rusqlite::ffi::sqlite3_memory_used(),
            rusqlite::ffi::sqlite3_memory_highwater(i32::from(reset)),
        )
    };
    (
        (used as u64).saturating_sub(one.sqlite_before),
        (peak as u64).saturating_sub(one.sqlite_before),
    )
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

fn measure<T>(one: &Alone, run: impl FnOnce() -> T) -> (T, Taken) {
    let rust_before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(rust_before, Ordering::Relaxed);
    sqlite_heap(one, true);
    let result = run();
    let rust_peak = PEAK.load(Ordering::Relaxed);
    let (_, sqlite_peak) = sqlite_heap(one, false);
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
    /// Only a test that has its turn ([`alone`]) gets a place for a store.
    fn new(_one: &Alone, name: &str) -> Self {
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
    let one = alone();
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
        let scratch = Scratch::new(&one, "synthetic");
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
        let (report, taken) = measure(&one, || store.add_with(&rows, &identity(128), &mut budget));
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
    let one = alone();
    let scratch = Scratch::new(&one, "give-back");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
    let mut budget = roomy_budget();
    let left = budget.left_bytes();

    let long = rows_with_long_texts(Kind::Distinct, 256 * 1024, 1);
    let (first, taken) = measure(&one, || store.add_with(&long, &identity(8), &mut budget));
    let first = first.expect("the long text is added");
    let idle = one.sqlite_used();
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
    let (second, taken) = measure(&one, || {
        store.add_with(&short_rows("later"), &identity(8), &mut budget)
    });
    let second = second.expect("the short texts are added");
    println!(
        "second add on the same open store (300 short texts): SQLite heap peak {:.2} MiB, \
         share held {:.2} MiB; in use afterwards {:.2} MiB",
        taken.sqlite_bytes as f64 / MIB,
        second.held_bytes as f64 / MIB,
        one.sqlite_used() as f64 / MIB,
    );
    assert!(second.held_bytes < ADD_FIXED_BYTES + 64 * 1024);
    assert!(
        taken.sqlite_bytes <= second.held_bytes,
        "the second add took {} bytes of SQLite's heap and held {}",
        taken.sqlite_bytes,
        second.held_bytes
    );
    assert!(one.sqlite_used() <= store.retained_bytes());
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
        one.sqlite_used() <= store.retained_bytes(),
        "after the failed add SQLite holds {} bytes",
        one.sqlite_used()
    );
    assert_eq!(store.len().unwrap(), 501);

    drop(store);
    assert_eq!(one.sqlite_used(), 0);
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
    let one = alone();
    println!(
        "text                 | plain connection, MiB in use: after commit | after shrink_memory \
         | after statement cache flush | after close || store, MiB: after the add | says it \
         retains | peak of a later short add | share that add held"
    );
    let no_repeat = no_repeat_text((1 << 19) + 64);
    let cases: [(&str, String); 6] = [
        ("log-like 1 MiB", long_text(Kind::Realistic, 1 << 20, 1)),
        ("random ASCII 1 MiB", long_text(Kind::Noisy, 1 << 20, 1)),
        ("two-byte 1 MiB", long_text(Kind::Distinct, 1 << 20, 1)),
        ("no repeat 2^19+64", no_repeat),
        ("random ASCII 256 KiB", long_text(Kind::Noisy, 256 << 10, 1)),
        ("short texts only", String::from("GET /a?x=1")),
    ];
    let mut failures = Vec::new();
    for (label, text) in &cases {
        // The plain connection.
        let scratch = Scratch::new(&one, "retained-plain");
        let paths = scratch.paths();
        drop(Store::create(&paths, "s", &identity(8)).unwrap());
        assert_eq!(one.sqlite_used(), 0);
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
        let after_commit = one.sqlite_used();
        connection.execute_batch("PRAGMA shrink_memory").unwrap();
        let after_shrink = one.sqlite_used();
        connection.flush_prepared_statement_cache();
        let after_flush = one.sqlite_used();
        drop(connection);
        let after_close = one.sqlite_used();
        drop(scratch);

        // The store.
        let scratch = Scratch::new(&one, "retained-store");
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
        let idle = one.sqlite_used();
        let retains = store.retained_bytes();
        let (later, taken) = measure(&one, || {
            store.add_with(&short_rows("later"), &identity(8), &mut budget)
        });
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
    let one = alone();
    for (kind, bytes) in [(Kind::Noisy, 256 * 1024), (Kind::Distinct, 256 * 1024)] {
        let rows = rows_with_long_texts(kind, bytes, 1);
        let scratch = Scratch::new(&one, "long-row");
        let paths = scratch.paths();
        let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
        let mut budget = roomy_budget();
        let (report, taken) = measure(&one, || store.add_with(&rows, &identity(8), &mut budget));
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
    let one = alone();
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
            let scratch = Scratch::new(&one, "long");
            let paths = scratch.paths();
            let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
            let mut budget = roomy_budget();
            let started = Instant::now();
            let (report, taken) =
                measure(&one, || store.add_with(&rows, &identity(8), &mut budget));
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
        let scratch = Scratch::new(&one, "long");
        let paths = scratch.paths();
        let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
        let mut budget = roomy_budget();
        let started = Instant::now();
        let (report, taken) = measure(&one, || store.add_with(&rows, &identity(8), &mut budget));
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
    let one = alone();
    let Some(logs) = csic("csic_populates_a_store_within_its_memory_share") else {
        return;
    };
    const SOURCE: &str = "data/csic-2010/csic_database.csv";
    const DIMENSION: u16 = 128;
    assert_eq!(logs.len(), 25_608);
    let text_bytes: usize = logs.records().map(|record| record.text.len()).sum();
    let rows = rows_of(&logs, SOURCE, DIMENSION);

    let scratch = Scratch::new(&one, "csic");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "csic", &identity(DIMENSION)).unwrap();
    let mut budget =
        MemoryBudget::observe(&MemorySettings::default()).expect("the host has free memory");

    let started = Instant::now();
    let (report, taken) = measure(&one, || {
        store.add_with(&rows, &identity(DIMENSION), &mut budget)
    });
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
    let (again, taken) = measure(&one, || {
        store.add_with(&rows, &identity(DIMENSION), &mut budget)
    });
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

// ---------------------------------------------------------------------------------------
// The reopening that fails (follow-up of the 1.8 review)
// ---------------------------------------------------------------------------------------

/// After an add that indexed a long text the store opens its connection again. When
/// that fails the add stands, the store goes on with the connection it has, says what
/// that connection keeps, counts the long text in the share of later adds, and gives
/// the memory back with the first add after which the file opens again.
#[test]
fn a_reopening_that_fails_leaves_the_add_standing_and_what_is_kept_counted() {
    let one = alone();
    let scratch = Scratch::new(&one, "reopen-fails");
    let paths = scratch.paths();
    let path = paths.store_path("s");
    let mut store = Store::create(&paths, "s", &identity(8)).unwrap();
    let mut budget = roomy_budget();
    let left = budget.left_bytes();

    // Another connection gives the file a format number this tool does not open.
    let set_format = |format: u32| {
        Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_WRITE)
            .unwrap()
            .pragma_update(None, "user_version", format)
            .unwrap();
    };
    set_format(SCHEMA_VERSION + 1);

    let long = rows_with_long_texts(Kind::Distinct, 256 * 1024, 1);
    let first = store
        .add_with(&long, &identity(8), &mut budget)
        .expect("the add stands although the store could not be opened again");
    assert_eq!((first.added, first.repeated), (201, 0));
    assert_eq!(budget.left_bytes(), left);
    let longest = first.longest_text_bytes;
    assert!(longest > RELEASE_TEXT_BYTES);
    assert!(matches!(
        Store::open(&paths, "s"),
        Err(StoreError::Failed { .. })
    ));

    // The store says what its connection keeps, and that covers what SQLite holds.
    let kept = IDLE_BYTES + RETAINED_TEXT_FACTOR * longest;
    assert_eq!(store.retained_bytes(), kept);
    let used = one.sqlite_used();
    println!(
        "after the add whose reopening failed: SQLite holds {:.2} MiB, the store says it \
         retains {:.2} MiB",
        used as f64 / MIB,
        kept as f64 / MIB
    );
    assert!(
        used > IDLE_BYTES,
        "the table of the text index was given back after all ({used} bytes in use): the \
         reopening did not fail"
    );
    assert!(used <= kept);

    // A search counts what the connection keeps above the idle level in its own
    // share, and is refused one byte short of it.
    let explained = store.explain(&Filter::default()).unwrap();
    let kept_more = RETAINED_TEXT_FACTOR * longest;
    assert_eq!(explained.held_bytes, SEARCH_FIXED_BYTES + kept_more);
    let queries = query_vectors(1, 8);
    let limit = (32 << 20) + explained.held_bytes;
    let mut short = fixed_budget(Some(limit - 1), None);
    for error in [
        store
            .text_search_with(&Filter::default(), 10, &mut short)
            .unwrap_err(),
        store
            .matching_with(&Filter::default(), 10, &mut short)
            .unwrap_err(),
        store
            .nearest_with(&queries, &identity(8), 3, &Filter::default(), &mut short)
            .unwrap_err(),
    ] {
        let StoreError::Memory(L2dError::Refused { reason }) = &error else {
            panic!("expected a memory refusal, got {error:?}");
        };
        // The nearest search names a larger total: it holds its best hits as well.
        assert!(reason.contains(&kept_more.to_string()), "{reason}");
        assert!(reason.contains("still keeps"), "{reason}");
    }
    let mut enough = fixed_budget(Some(limit + (1 << 20)), None);
    assert_eq!(
        store
            .text_search_with(&Filter::default(), 10, &mut enough)
            .unwrap()
            .height(),
        10
    );
    let found = store
        .nearest_with(&queries, &identity(8), 3, &Filter::default(), &mut enough)
        .unwrap();
    assert!(found.held_bytes >= explained.held_bytes);
    assert!(one.sqlite_used() <= kept);

    // It stays usable, and the long text stays counted in the share of the next add,
    // which tries again and fails again.
    assert_eq!(store.len().unwrap(), 201);
    let second = store
        .add_with(&short_rows("second"), &identity(8), &mut budget)
        .expect("the store takes adds on the connection it kept");
    assert_eq!((second.added, second.repeated), (300, 0));
    assert_eq!(
        second.held_bytes,
        ADD_FIXED_BYTES + ADD_TEXT_FACTOR * longest
    );
    assert_eq!(store.retained_bytes(), kept);
    assert!(one.sqlite_used() <= kept);
    assert_eq!(budget.left_bytes(), left);

    // The file is right again: the next add still counts the text, because the table is
    // there while it runs, and ends by opening the store again.
    set_format(SCHEMA_VERSION);
    let third = store
        .add_with(&short_rows("third"), &identity(8), &mut budget)
        .unwrap();
    assert_eq!(
        third.held_bytes,
        ADD_FIXED_BYTES + ADD_TEXT_FACTOR * longest
    );
    assert_eq!(store.retained_bytes(), IDLE_BYTES);
    let used = one.sqlite_used();
    assert!(
        used <= IDLE_BYTES,
        "after the reopening SQLite still holds {used} bytes"
    );
    assert_eq!(
        store.explain(&Filter::default()).unwrap().held_bytes,
        SEARCH_FIXED_BYTES
    );
    let fourth = store
        .add_with(&short_rows("fourth"), &identity(8), &mut budget)
        .unwrap();
    assert!(fourth.held_bytes < ADD_FIXED_BYTES + 64 * 1024);
    assert_eq!(store.len().unwrap(), 201 + 900);

    drop(store);
    assert_eq!(one.sqlite_used(), 0);
    assert_fts_consistent(&path);
}

// ---------------------------------------------------------------------------------------
// Searching (task 1.9)
// ---------------------------------------------------------------------------------------

/// Whether a text is one the search under test has to return.
type Keep<'a> = Box<dyn Fn(&str) -> bool + 'a>;

/// A budget that does not depend on the host, with a limit or a batch size if given.
fn fixed_budget(limit: Option<u64>, batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(64 << 30)).expect("the budget is usable")
}

/// `rows` request lines, every other one bad, over seven groups.
fn search_logs(rows: u64) -> LogsFrame {
    let records: Vec<LogRecord> = (0..rows)
        .map(|index| LogRecord {
            text: format!(
                "GET http://localhost:8080/tienda1/publico/anadir.jsp?id={index}&nombre={}",
                long_text(Kind::Noisy, 40, index + 3)
            ),
            label: if index % 2 == 0 {
                Label::Bad
            } else {
                Label::Good
            },
            group: Some(format!("/tienda1/{}", index % 7)),
            first_position: index + 1,
            occurrences: 1,
        })
        .collect();
    LogsFrame::from_records(&records).unwrap()
}

/// `count` query vectors of `dimension`, one after the other.
fn query_vectors(count: u64, dimension: u16) -> Vec<f32> {
    let records: Vec<LogRecord> = (0..count)
        .map(|index| LogRecord {
            text: format!("a log that is searched for, number {index}"),
            label: Label::Unknown,
            group: None,
            first_position: index + 1,
            occurrences: 1,
        })
        .collect();
    vectors_of(&LogsFrame::from_records(&records).unwrap(), dimension)
}

fn phrase(text: &str) -> Filter {
    Filter {
        contains: vec![text.to_string()],
        ..Filter::default()
    }
}

/// What a nearest search asked of the budget: its own share (SQLite's, the texts it
/// reads, and its best hits), one batch of vectors, and the hits.
fn asked_by(found: &Nearest, stored: u64) -> u64 {
    let plan = found.plan.as_ref().expect("the search read the store");
    found.held_bytes + plan.bytes_per_row * plan.max_rows_used.min(stored) + found.hits_bytes
}

/// Asserts that a nearest search stayed within what it asked: SQLite within the share
/// of a search with its filter, the Rust side within the rest (and its part of what
/// is held for reading texts, the lower-case copy).
fn assert_nearest_within(
    what: &str,
    store: &Store,
    filter: &Filter,
    found: &Nearest,
    stored: u64,
    taken: &Taken,
) {
    let asked = asked_by(found, stored);
    let explained = store.explain(filter).unwrap();
    let sqlite_share = explained.held_bytes;
    let text_share = TEXT_READ_FACTOR * explained.longest_text_bytes;
    assert!(sqlite_share >= SEARCH_FIXED_BYTES && sqlite_share <= found.held_bytes);
    let plan = found.plan.as_ref().expect("the search read the store");
    println!(
        "{what}: {} hits of {} candidates; SQLite heap peak {:.2} MiB of {:.2} MiB held for \
         it, Rust heap peak {:.3} MiB, together {:.2} of the {:.2} MiB asked of the budget \
         ({} bytes per row, batches of up to {} rows)",
        found.hits.height(),
        found.candidates,
        taken.sqlite_bytes as f64 / MIB,
        sqlite_share as f64 / MIB,
        taken.rust_bytes as f64 / MIB,
        (taken.sqlite_bytes + taken.rust_bytes) as f64 / MIB,
        asked as f64 / MIB,
        plan.bytes_per_row,
        plan.max_rows_used,
    );
    assert!(
        taken.sqlite_bytes > 0,
        "SQLite's allocation counters are off"
    );
    assert!(
        taken.sqlite_bytes <= sqlite_share,
        "{what}: SQLite took {} bytes, {sqlite_share} were held for it",
        taken.sqlite_bytes,
    );
    assert!(
        taken.rust_bytes <= asked - sqlite_share + text_share,
        "{what}: the Rust side took {} bytes of the {} asked beside SQLite's share",
        taken.rust_bytes,
        asked - sqlite_share + text_share
    );
    assert!(taken.sqlite_bytes + taken.rust_bytes <= asked);
}

/// Reads the rows that match and asserts that it stayed within what was asked.
fn read_within(
    one: &Alone,
    what: &str,
    store: &Store,
    filter: &Filter,
    limit: Option<usize>,
    budget: &mut MemoryBudget,
) -> DataFrame {
    let matching = store
        .matching_with(filter, limit.unwrap_or(usize::MAX), budget)
        .expect("the matching rows are counted");
    let (frame, taken) = measure(one, || match limit {
        Some(limit) => store.text_search_with(filter, limit, budget),
        None => store.vectors_with(filter, budget),
    });
    let frame = frame.expect("the rows are read");
    let explained = store.explain(filter).unwrap();
    let sqlite_share = explained.held_bytes;
    let text_share = TEXT_READ_FACTOR * explained.longest_text_bytes;
    println!(
        "{what}: {} rows; SQLite heap peak {:.2} MiB of the {:.2} MiB held for it, Rust heap \
         peak {:.3} MiB of the {:.3} MiB asked for the rows; the frame takes {:.3} MiB",
        frame.height(),
        taken.sqlite_bytes as f64 / MIB,
        sqlite_share as f64 / MIB,
        taken.rust_bytes as f64 / MIB,
        matching.read_bytes as f64 / MIB,
        frame.estimated_size() as f64 / MIB,
    );
    assert_eq!(frame.height() as u64, matching.rows);
    assert!(
        taken.sqlite_bytes <= sqlite_share,
        "{what}: SQLite took {} bytes, {sqlite_share} were held for it",
        taken.sqlite_bytes
    );
    assert!(
        taken.rust_bytes <= matching.read_bytes + text_share,
        "{what}: the Rust side took {} bytes, {} were asked",
        taken.rust_bytes,
        matching.read_bytes + text_share
    );
    assert!(taken.sqlite_bytes + taken.rust_bytes <= sqlite_share + matching.read_bytes);
    frame
}

/// A text of `bytes` bytes that repeats `unit`.
fn repeated(unit: &str, bytes: usize) -> String {
    let mut text = unit.repeat(bytes / unit.len() + 1);
    while text.len() > bytes || !text.is_char_boundary(text.len()) {
        text.pop();
    }
    text
}

/// A store named `name` with 200 short logs and the one long `text`, 8-dimension
/// vectors.
fn store_with_long_text(scratch: &Scratch, name: &str, text: &str) -> Store {
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
    records.push(LogRecord {
        text: text.to_string(),
        label: Label::Bad,
        group: None,
        first_position: 201,
        occurrences: 1,
    });
    let mut store = Store::create(&scratch.paths(), name, &identity(8)).unwrap();
    store
        .add_with(
            &rows_of(&LogsFrame::from_records(&records).unwrap(), "long", 8),
            &identity(8),
            &mut roomy_budget(),
        )
        .expect("the long text is added");
    store
}

/// Searches `store` for `filter`, whose phrases occur in its long text, in the three
/// ways (counting, reading back, nearest) and asserts each time that SQLite's heap
/// stayed within what the search held for it (requirement 12.9) and the Rust side
/// within the rest. Returns the largest part of the held share that SQLite used.
fn assert_phrase_search_within(one: &Alone, what: &str, store: &Store, filter: &Filter) -> f64 {
    let explained = store.explain(filter).unwrap();
    let held = explained.held_bytes;
    let text_share = TEXT_READ_FACTOR * explained.longest_text_bytes;
    let mut budget = roomy_budget();
    let left = budget.left_bytes();
    let within = |way: &str, sqlite: u64| {
        assert!(
            sqlite <= held,
            "{what}, {way}: SQLite took {sqlite} bytes, the search held {held} ({:.2})",
            sqlite as f64 / held as f64
        );
    };

    let (matching, counting) =
        measure(one, || store.matching_with(filter, usize::MAX, &mut budget));
    let matching = matching.expect("the logs are counted");
    assert!(
        matching.rows >= 1,
        "{what}: the long text holds the phrases"
    );
    within("counting", counting.sqlite_bytes);
    // What is held already contains the share for the texts: both sides together.
    assert!(
        counting.sqlite_bytes + counting.rust_bytes <= held,
        "{what}, counting: SQLite and the Rust side took {} bytes together, {held} were held",
        counting.sqlite_bytes + counting.rust_bytes
    );
    assert!(
        counting.rust_bytes <= text_share,
        "{what}, counting: the Rust side took {} bytes",
        counting.rust_bytes
    );

    let (frame, reading) = measure(one, || {
        store.text_search_with(filter, usize::MAX, &mut budget)
    });
    let frame = frame.expect("the logs are read back");
    assert_eq!(frame.height() as u64, matching.rows);
    drop(frame);
    within("reading back", reading.sqlite_bytes);
    assert!(
        reading.sqlite_bytes + reading.rust_bytes <= held + matching.read_bytes,
        "{what}, reading back: SQLite and the Rust side took {} bytes together",
        reading.sqlite_bytes + reading.rust_bytes
    );
    assert!(
        reading.rust_bytes <= matching.read_bytes + text_share,
        "{what}, reading back: the Rust side took {} bytes",
        reading.rust_bytes
    );

    let queries = query_vectors(1, 8);
    let (found, nearest) = measure(one, || {
        store.nearest_with(&queries, &identity(8), 3, filter, &mut budget)
    });
    let found = found.expect("the nearest logs are found");
    assert_eq!(found.candidates, matching.rows);
    assert!(found.held_bytes >= held);
    within("nearest", nearest.sqlite_bytes);
    assert!(
        nearest.sqlite_bytes + nearest.rust_bytes <= asked_by(&found, 201),
        "{what}, nearest: SQLite and the Rust side took {} bytes together",
        nearest.sqlite_bytes + nearest.rust_bytes
    );
    assert!(
        nearest.rust_bytes <= asked_by(&found, 201) - held + text_share,
        "{what}, nearest: the Rust side took {} bytes",
        nearest.rust_bytes
    );
    drop(found);
    assert_eq!(budget.left_bytes(), left);
    assert!(one.sqlite_used() <= IDLE_BYTES);

    let most = counting
        .sqlite_bytes
        .max(reading.sqlite_bytes)
        .max(nearest.sqlite_bytes);
    println!(
        "{what}: pieces {:?}; held {:.2} MiB; SQLite heap peak {:.2} MiB counting, {:.2} \
         reading back, {:.2} nearest ({:.2} of held)",
        explained.index_pieces,
        held as f64 / MIB,
        counting.sqlite_bytes as f64 / MIB,
        reading.sqlite_bytes as f64 / MIB,
        nearest.sqlite_bytes as f64 / MIB,
        most as f64 / held as f64
    );
    most as f64 / held as f64
}

/// The lower-case copy of a text that is compared ignoring case is allocated once and
/// stays within what the share counts for it, twice the bytes of the longest stored
/// text: a text of `Ⱥ` (U+023A, two bytes), whose lower-case form has three bytes, and
/// a text of four-byte characters, each exactly as long as a store takes.
#[test]
fn the_lower_case_copy_of_a_text_stays_within_what_is_held_for_it() {
    let one = alone();
    let scratch = Scratch::new(&one, "lower-case");
    let longest = usize::try_from(MAX_TEXT_BYTES).unwrap();
    for (name, unit, grows) in [("stroke", "\u{23A}", 3), ("emoji", "\u{1F600}", 2)] {
        let text = repeated(unit, longest);
        assert_eq!(text.len(), longest);
        let store = store_with_long_text(&scratch, name, &text);
        // Two characters: answered by reading every text, each of which is lowered.
        let filter = phrase(&unit.repeat(2));
        let explained = store.explain(&filter).unwrap();
        assert!(!explained.text_index);
        let copy_share = 2 * MAX_TEXT_BYTES;
        assert_eq!(
            explained.held_bytes,
            SEARCH_FIXED_BYTES
                + TEXT_READ_FACTOR * MAX_TEXT_BYTES
                + l2d_store::fold(&unit.repeat(2)).len() as u64
        );
        let mut budget = roomy_budget();
        let (matching, taken) = measure(&one, || {
            store.matching_with(&filter, usize::MAX, &mut budget)
        });
        assert_eq!(matching.expect("the logs are counted").rows, 1);
        let copy = l2d_store::fold(&text).len() as u64;
        println!(
            "1 MiB of U+{:04X}: the lower-case copy has {:.2} MiB; Rust heap peak {:.2} MiB of \
             the {:.2} MiB counted for it; SQLite heap peak {:.2} MiB; held {:.2} MiB",
            u32::from(unit.chars().next().unwrap()),
            copy as f64 / MIB,
            taken.rust_bytes as f64 / MIB,
            copy_share as f64 / MIB,
            taken.sqlite_bytes as f64 / MIB,
            explained.held_bytes as f64 / MIB,
        );
        assert_eq!(copy, MAX_TEXT_BYTES * grows / 2);
        // One allocation of exactly the copy, and little beside it.
        assert!(taken.rust_bytes >= copy && taken.rust_bytes <= copy + 64 * 1024);
        assert!(taken.rust_bytes <= copy_share);
        assert!(taken.sqlite_bytes + taken.rust_bytes <= explained.held_bytes);
        // The other ways of searching, and phrases that go through the index.
        for (what, filter) in [
            ("two characters", filter.clone()),
            ("sixteen characters", phrase(&unit.repeat(16))),
        ] {
            assert_phrase_search_within(&one, &format!("1 MiB of {name}, {what}"), &store, &filter);
        }
        drop(store);
        assert_eq!(one.sqlite_used(), 0);
    }
}

/// The phrase searches of the memory tests over one long text: one phrase of 3, 4, 16
/// and 300 characters, four phrases and six, each ignoring case (given in upper case)
/// and case-exactly, and the long text itself as the exact text.
fn phrase_filters(text: &str) -> Vec<(String, Filter)> {
    let start = |characters: usize| -> String { text.chars().take(characters).collect() };
    let sets: Vec<(&str, Vec<String>)> = vec![
        ("one phrase of 3 characters", vec![start(3)]),
        ("one phrase of 4 characters", vec![start(4)]),
        ("one phrase of 16 characters", vec![start(16)]),
        ("one phrase of 300 characters", vec![start(300)]),
        ("four phrases", vec![start(5), start(6), start(7), start(8)]),
        (
            "six phrases",
            vec![start(4), start(5), start(6), start(7), start(8), start(9)],
        ),
    ];
    let mut filters = Vec::new();
    for (name, phrases) in sets {
        filters.push((
            format!("{name}, ignoring case"),
            Filter {
                contains: phrases.iter().map(|phrase| phrase.to_uppercase()).collect(),
                ..Filter::default()
            },
        ));
        filters.push((
            format!("{name}, case-exact"),
            Filter {
                contains: phrases,
                case_exact: true,
                ..Filter::default()
            },
        ));
    }
    filters.push((
        "the exact text".to_string(),
        Filter {
            equals: Some(text.to_string()),
            ..Filter::default()
        },
    ));
    filters
}

/// Searches stay within what they ask of the budget (requirement 12.9), hand the
/// budget back, and leave an open store at its idle level.
#[test]
fn a_search_takes_no_more_memory_than_it_asks_of_the_budget() {
    let one = alone();
    let scratch = Scratch::new(&one, "search-heap");
    let paths = scratch.paths();
    const DIMENSION: u16 = 64;
    const ROWS: u64 = 4_000;
    let logs = search_logs(ROWS);
    let mut store = Store::create(&paths, "s", &identity(DIMENSION)).unwrap();
    let mut budget = roomy_budget();
    store
        .add_with(
            &rows_of(&logs, "search", DIMENSION),
            &identity(DIMENSION),
            &mut budget,
        )
        .unwrap();
    drop(logs);
    let left = budget.left_bytes();
    let idle = |what: &str| {
        let used = one.sqlite_used();
        assert_eq!(store.retained_bytes(), IDLE_BYTES);
        assert!(
            used <= IDLE_BYTES,
            "{what}: SQLite holds {used} bytes; an idle store keeps at most {IDLE_BYTES}"
        );
    };
    idle("after the add");

    let queries = query_vectors(5, DIMENSION);
    let exact = Filter {
        case_exact: true,
        ..phrase("anadir.jsp?id=12")
    };
    let cases: [(&str, Filter, Option<u64>); 6] = [
        ("nearest, all logs", Filter::default(), None),
        ("nearest, batches of 1,024", Filter::default(), Some(1_024)),
        ("nearest, batches of 100", Filter::default(), Some(100)),
        ("nearest, a phrase through the index", phrase("ID=12"), None),
        ("nearest, a case-exact phrase", exact.clone(), None),
        ("nearest, a short phrase by a scan", phrase("=7"), None),
    ];
    for (what, filter, batch_rows) in &cases {
        let mut budget = fixed_budget(None, *batch_rows);
        let left = budget.left_bytes();
        let (found, taken) = measure(&one, || {
            store.nearest_with(&queries, &identity(DIMENSION), 10, filter, &mut budget)
        });
        let found = found.expect("the nearest logs are found");
        assert!(found.candidates > 0, "{what}");
        assert_nearest_within(what, &store, filter, &found, ROWS, &taken);
        assert_eq!(budget.left_bytes(), left);
        drop(found);
        idle(what);
    }

    let bad = Filter {
        label: Some(Label::Bad),
        ..Filter::default()
    };
    for (what, filter, limit) in [
        ("export, all logs", &Filter::default(), None),
        ("export, one label", &bad, None),
        ("export, a phrase", &phrase("id=12"), None),
        (
            "text search, a phrase, 50 rows",
            &phrase("nombre="),
            Some(50),
        ),
        ("text search, case-exact", &exact, Some(usize::MAX)),
        (
            "text search, a short phrase by a scan",
            &phrase("=7"),
            Some(usize::MAX),
        ),
    ] {
        let frame = read_within(&one, what, &store, filter, limit, &mut budget);
        assert!(frame.height() > 0, "{what}");
        assert_eq!(budget.left_bytes(), left);
        drop(frame);
        idle(what);
    }
    drop(store);
    assert_eq!(one.sqlite_used(), 0);
}

/// Requirement 12.9 where it is hardest: a phrase that occurs in a long text that
/// repeats itself. The text index reads, for that log, the list of all places of
/// every trigram it is given, and each list is as long as the text. The store's
/// longest text is exactly the longest a store takes (the worst case: the buffers
/// double just there), a run of one character and a cycle of three; every search
/// stays within what it held, and is refused up front one byte short of it.
#[test]
fn a_phrase_search_over_a_long_repetitive_text_stays_within_what_it_holds() {
    let one = alone();
    let scratch = Scratch::new(&one, "repetitive");
    let longest = usize::try_from(MAX_TEXT_BYTES).unwrap();
    let mut worst = 0.0f64;
    for (name, unit) in [("run", "a"), ("cycle", "abc")] {
        let text = repeated(unit, longest);
        let store = store_with_long_text(&scratch, name, &text);
        assert_eq!(store.longest_text_bytes().unwrap(), MAX_TEXT_BYTES);
        assert!(one.sqlite_used() <= IDLE_BYTES);
        for (what, filter) in phrase_filters(&text) {
            let what = format!("1 MiB of '{unit}', {what}");
            worst = worst.max(assert_phrase_search_within(&one, &what, &store, &filter));
        }

        // What is held, and the refusal one byte short of it.
        let filter = phrase(&text[..16].to_uppercase());
        let explained = store.explain(&filter).unwrap();
        // The three pieces of the phrase are one and the same: given once.
        assert_eq!(explained.index_pieces.len(), 1);
        assert_eq!(explained.longest_text_bytes, MAX_TEXT_BYTES);
        let index_share = 4 * PHRASE_BYTE_BYTES + INDEX_TEXT_FACTOR * 4 * MAX_TEXT_BYTES;
        // With the lower-case copy of the phrase, 16 bytes.
        let text_share = TEXT_READ_FACTOR * MAX_TEXT_BYTES + 16;
        assert_eq!(
            explained.held_bytes,
            SEARCH_FIXED_BYTES + text_share + index_share
        );
        let limit = (32 << 20) + explained.held_bytes;
        let queries = query_vectors(1, 8);
        let mut short = fixed_budget(Some(limit - 1), None);
        let left = short.left_bytes();
        for error in [
            store
                .matching_with(&filter, usize::MAX, &mut short)
                .unwrap_err(),
            store
                .text_search_with(&filter, usize::MAX, &mut short)
                .unwrap_err(),
            store
                .nearest_with(&queries, &identity(8), 3, &filter, &mut short)
                .unwrap_err(),
        ] {
            let StoreError::Memory(L2dError::Refused { reason }) = &error else {
                panic!("expected a memory refusal, got {error:?}");
            };
            for figure in [
                MAX_TEXT_BYTES,
                text_share,
                index_share,
                SEARCH_FIXED_BYTES,
                left,
                limit - 1,
            ] {
                assert!(
                    reason.contains(&figure.to_string()),
                    "{reason} lacks {figure}"
                );
            }
            assert!(reason.contains("nothing was read"), "{reason}");
        }
        assert_eq!(short.left_bytes(), left);
        let counted = store
            .matching_with(&filter, usize::MAX, &mut fixed_budget(Some(limit), None))
            .expect("the share itself is enough to count");
        assert_eq!(counted.rows, 1);
        drop(store);
        assert_eq!(one.sqlite_used(), 0);
    }
    println!("the most SQLite took of what a phrase search held: {worst:.2}");
    assert!(worst <= 1.0);
}

/// Searches over the 25,608 distinct logs of HTTP CSIC 2010 with synthetic
/// 128-dimension vectors: time, memory against what was asked, and the nearest logs
/// against a brute-force computation.
#[test]
#[ignore = "CSIC measurement; run in release mode on request"]
fn csic_searches_measured() {
    let one = alone();
    let Some(logs) = csic("csic_searches_measured") else {
        return;
    };
    const SOURCE: &str = "data/csic-2010/csic_database.csv";
    const DIMENSION: u16 = 128;
    const STORED: u64 = 25_608;
    let width = usize::from(DIMENSION);
    let scratch = Scratch::new(&one, "csic-search");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "csic", &identity(DIMENSION)).unwrap();
    let mut budget =
        MemoryBudget::observe(&MemorySettings::default()).expect("the host has free memory");
    {
        let rows = rows_of(&logs, SOURCE, DIMENSION);
        store
            .add_with(&rows, &identity(DIMENSION), &mut budget)
            .expect("the logs are added");
    }
    let stored_vectors = vectors_of(&logs, DIMENSION);
    let stored_texts: Vec<String> = logs.records().map(|record| record.text).collect();
    drop(logs);
    assert!(one.sqlite_used() <= IDLE_BYTES);
    println!("CSIC 2010: {STORED} logs with {DIMENSION}-dimension vectors; {budget:?}");

    // Brute force in double precision: the distances of the ten nearest, in order.
    let reference = |query: &[f32], keep: &dyn Fn(&str) -> bool| -> Vec<f64> {
        let mut all: Vec<f64> = stored_vectors
            .chunks_exact(width)
            .zip(&stored_texts)
            .filter(|(_, text)| keep(text))
            .map(|(vector, _)| {
                vector
                    .iter()
                    .zip(query)
                    .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                    .sum()
            })
            .collect();
        all.sort_by(f64::total_cmp);
        all.truncate(10);
        all
    };
    let check = |found: &Nearest, query_index: usize, expected: &[f64]| {
        let distances: Vec<f32> = found
            .hits
            .column(DISTANCE)
            .unwrap()
            .f32()
            .unwrap()
            .iter()
            .flatten()
            .skip(query_index * 10)
            .take(10)
            .collect();
        assert_eq!(distances.len(), expected.len());
        for (got, want) in distances.iter().zip(expected) {
            assert!(
                (f64::from(*got) - want).abs() <= 1e-5,
                "distance {got}, reference {want}"
            );
        }
    };

    // One query at a time: a stored log's own vector (found at the distance 0) and
    // vectors of logs that are not stored.
    let queries = query_vectors(100, DIMENSION);
    let mut seconds = Vec::new();
    let mut worst = (0u64, 0u64);
    for index in 0..20 {
        let query = if index == 0 {
            &stored_vectors[7_000 * width..7_001 * width]
        } else {
            &queries[index * width..(index + 1) * width]
        };
        let started = Instant::now();
        let (found, taken) = measure(&one, || {
            store.nearest_with(
                query,
                &identity(DIMENSION),
                10,
                &Filter::default(),
                &mut budget,
            )
        });
        seconds.push(started.elapsed().as_secs_f64());
        let found = found.expect("the nearest logs are found");
        if index == 0 {
            assert_nearest_within(
                "  one query, 10 nearest",
                &store,
                &Filter::default(),
                &found,
                STORED,
                &taken,
            );
            println!("  plan: {}", found.plan.as_ref().unwrap());
            let first = found
                .hits
                .column(TEXT)
                .unwrap()
                .str()
                .unwrap()
                .get(0)
                .unwrap();
            assert_eq!(first, stored_texts[7_000]);
        }
        assert_eq!(found.candidates, STORED);
        check(&found, 0, &reference(query, &|_| true));
        let asked = asked_by(&found, STORED);
        assert!(taken.sqlite_bytes <= SEARCH_FIXED_BYTES);
        assert!(taken.rust_bytes <= asked - SEARCH_FIXED_BYTES);
        worst = (
            worst.0.max(taken.sqlite_bytes),
            worst.1.max(taken.rust_bytes),
        );
        assert!(one.sqlite_used() <= IDLE_BYTES);
    }
    seconds.sort_by(f64::total_cmp);
    println!(
        "  one query per search, 20 searches: {:.1} ms at best, {:.1} ms median, {:.1} ms at \
         worst; SQLite heap peak {:.2} MiB, Rust heap peak {:.2} MiB at most",
        seconds[0] * 1e3,
        seconds[10] * 1e3,
        seconds[19] * 1e3,
        worst.0 as f64 / MIB,
        worst.1 as f64 / MIB
    );

    // A hundred queries in one pass over the store.
    let started = Instant::now();
    let (found, taken) = measure(&one, || {
        store.nearest_with(
            &queries,
            &identity(DIMENSION),
            10,
            &Filter::default(),
            &mut budget,
        )
    });
    let elapsed = started.elapsed().as_secs_f64();
    let found = found.expect("the nearest logs are found");
    assert_eq!(found.hits.height(), 1_000);
    assert_nearest_within(
        "  100 queries in one search",
        &store,
        &Filter::default(),
        &found,
        STORED,
        &taken,
    );
    println!(
        "  100 queries in one search: {:.1} ms, {:.2} ms per query",
        elapsed * 1e3,
        elapsed * 10.0
    );
    for index in [0usize, 49, 99] {
        let query = &queries[index * width..(index + 1) * width];
        check(&found, index, &reference(query, &|_| true));
    }
    drop(found);

    // Small batches give the same hits.
    let mut small = MemoryBudget::observe(&MemorySettings {
        max_batch_rows: Some(1_024),
        ..MemorySettings::default()
    })
    .unwrap();
    let query = &queries[..3 * width];
    let whole = store
        .nearest_with(
            query,
            &identity(DIMENSION),
            10,
            &Filter::default(),
            &mut budget,
        )
        .unwrap();
    let started = Instant::now();
    let (batched, taken) = measure(&one, || {
        store.nearest_with(
            query,
            &identity(DIMENSION),
            10,
            &Filter::default(),
            &mut small,
        )
    });
    let elapsed = started.elapsed().as_secs_f64();
    let batched = batched.unwrap();
    assert_nearest_within(
        "  3 queries, batches of 1,024",
        &store,
        &Filter::default(),
        &batched,
        STORED,
        &taken,
    );
    println!("  3 queries, batches of 1,024: {:.1} ms", elapsed * 1e3);
    assert!(batched.hits.equals_missing(&whole.hits));
    drop((whole, batched));

    // Combined with a phrase and a label.
    for (what, filter, keep) in [
        (
            "  nearest among the logs with a phrase",
            phrase("anadir.jsp"),
            (&|text: &str| text.to_ascii_lowercase().contains("anadir.jsp"))
                as &dyn Fn(&str) -> bool,
        ),
        (
            "  nearest among the logs with a case-exact phrase",
            Filter {
                case_exact: true,
                ..phrase("B1=Entrar")
            },
            &|text: &str| text.contains("B1=Entrar"),
        ),
        (
            "  nearest among the logs with a short phrase (scan)",
            phrase("PW"),
            &|text: &str| text.to_ascii_lowercase().contains("pw"),
        ),
    ] {
        let query = &queries[..width];
        let started = Instant::now();
        let (found, taken) = measure(&one, || {
            store.nearest_with(query, &identity(DIMENSION), 10, &filter, &mut budget)
        });
        let elapsed = started.elapsed().as_secs_f64();
        let found = found.expect("the nearest logs are found");
        assert_nearest_within(what, &store, &filter, &found, STORED, &taken);
        println!("{what}: {:.1} ms", elapsed * 1e3);
        let expected = stored_texts.iter().filter(|text| keep(text)).count();
        assert!(expected > 0, "{what}: the dataset has no such log");
        assert_eq!(found.candidates, expected as u64, "{what}");
        check(&found, 0, &reference(query, keep));
        assert!(one.sqlite_used() <= IDLE_BYTES);
    }

    // Text searches and exports.
    let longest = stored_texts.iter().max_by_key(|text| text.len()).unwrap();
    let bad = Filter {
        label: Some(Label::Bad),
        ..Filter::default()
    };
    let cases: Vec<(&str, Filter, Option<usize>, Keep<'_>)> = vec![
        (
            "  text search, a phrase through the index, first 100",
            phrase("anadir.jsp"),
            Some(100),
            Box::new(|text| text.to_ascii_lowercase().contains("anadir.jsp")),
        ),
        (
            "  text search, a phrase through the index, all",
            phrase("anadir.jsp"),
            Some(usize::MAX),
            Box::new(|text| text.to_ascii_lowercase().contains("anadir.jsp")),
        ),
        (
            "  text search, a rare phrase",
            phrase("drop+table"),
            Some(usize::MAX),
            Box::new(|text| text.to_ascii_lowercase().contains("drop+table")),
        ),
        (
            "  text search, case-exact",
            Filter {
                case_exact: true,
                ..phrase("B1=Entrar")
            },
            Some(usize::MAX),
            Box::new(|text| text.contains("B1=Entrar")),
        ),
        (
            "  text search, a short phrase by a scan",
            phrase("PW"),
            Some(usize::MAX),
            Box::new(|text| text.to_ascii_lowercase().contains("pw")),
        ),
        (
            "  text search, a short phrase by a scan, first 10",
            phrase("PW"),
            Some(10),
            Box::new(|text| text.to_ascii_lowercase().contains("pw")),
        ),
        (
            "  text search, a phrase no log holds",
            phrase("no log holds this"),
            Some(usize::MAX),
            Box::new(|_| false),
        ),
        (
            "  text search, the exact text of the longest log",
            Filter {
                equals: Some(longest.clone()),
                ..Filter::default()
            },
            Some(usize::MAX),
            Box::new(|text| text == longest),
        ),
        (
            "  export of all logs with their vectors",
            Filter::default(),
            None,
            Box::new(|_| true),
        ),
    ];
    for (what, filter, limit, keep) in &cases {
        let started = Instant::now();
        let frame = read_within(&one, what, &store, filter, *limit, &mut budget);
        // Counting and reading: the time of both passes and of the count before them.
        println!(
            "{what}: {:.1} ms with the count that the test makes first",
            started.elapsed().as_secs_f64() * 1e3
        );
        let started = Instant::now();
        let again = match limit {
            Some(limit) => store.text_search_with(filter, *limit, &mut budget),
            None => store.vectors_with(filter, &mut budget),
        }
        .unwrap();
        println!(
            "{what}: {:.1} ms for the search alone",
            started.elapsed().as_secs_f64() * 1e3
        );
        assert!(again.equals_missing(&frame));
        let expected: Vec<&String> = stored_texts
            .iter()
            .filter(|text| keep(text))
            .take(limit.unwrap_or(usize::MAX))
            .collect();
        let got: Vec<&str> = frame
            .column(TEXT)
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(got.len(), expected.len(), "{what}");
        assert!(got.iter().zip(&expected).all(|(a, b)| a == b), "{what}");
        assert!(one.sqlite_used() <= IDLE_BYTES);
    }
    let started = Instant::now();
    let frame = read_within(
        &one,
        "  export of the bad logs",
        &store,
        &bad,
        None,
        &mut budget,
    );
    println!(
        "  export of the bad logs: {:.1} ms with the count",
        started.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(frame.height(), 15_964);
    drop(frame);
    drop(store);
    assert_eq!(one.sqlite_used(), 0);
}

/// What the text index takes for the phrases of a search: SQLite's heap by the bytes of
/// one phrase, on a store filled by one add and on one filled by many (the index keeps
/// one segment or more per add until it merges them, and a search opens a reader per
/// trigram and segment). Since the remediation of task 1.9 the index is given short
/// pieces of a phrase only, so the figures no longer grow with the phrase; those behind
/// `PHRASE_BYTE_BYTES` were taken with this test when it was given whole phrases.
#[test]
#[ignore = "phrase-memory measurement; run in release mode on request"]
fn phrase_memory_measured() {
    let one = alone();
    let Some(logs) = csic("phrase_memory_measured") else {
        return;
    };
    const SOURCE: &str = "data/csic-2010/csic_database.csv";
    let rows = rows_of(&logs, SOURCE, 8);
    let longest = logs
        .records()
        .map(|record| record.text)
        .max_by_key(|text| text.len())
        .unwrap();
    drop(logs);
    let scratch = Scratch::new(&one, "phrase-memory");
    let paths = scratch.paths();
    let mut budget = roomy_budget();
    let mut worst = 0.0f64;
    for (name, adds) in [("one", 1usize), ("many", 16), ("more", 256)] {
        let mut store = Store::create(&paths, name, &identity(8)).unwrap();
        let step = rows.height().div_ceil(adds);
        for start in (0..rows.height()).step_by(step) {
            store
                .add_with(&rows.slice(start as i64, step), &identity(8), &mut budget)
                .unwrap();
        }
        assert_eq!(store.len().unwrap(), 25_608);
        println!("store filled by {adds} adds:");
        for bytes in [3usize, 8, 16, 32, 64, 128, 256, 512, 891] {
            let filter = phrase(&longest[..bytes.min(longest.len())]);
            let started = Instant::now();
            let (frame, taken) = measure(&one, || {
                store.text_search_with(&filter, usize::MAX, &mut budget)
            });
            let frame = frame.expect("the phrase is searched");
            let held = store.explain(&filter).unwrap().held_bytes;
            let share = taken.sqlite_bytes as f64 / held as f64;
            println!(
                "  phrase of {bytes:4} bytes: {:5} rows, SQLite heap peak {:6.2} MiB of {:6.2} \
                 MiB held ({share:.2}), {:7.1} ms",
                frame.height(),
                taken.sqlite_bytes as f64 / MIB,
                held as f64 / MIB,
                started.elapsed().as_secs_f64() * 1e3
            );
            assert!(frame.height() >= 1);
            worst = worst.max(share);
            assert!(one.sqlite_used() <= IDLE_BYTES);
        }
    }
    println!("the most SQLite took of what a search held for it: {worst:.2}");
    assert!(worst <= 1.0);
}

// ---------------------------------------------------------------------------------------
// What the text index takes by itself (remediation of task 1.9)
// ---------------------------------------------------------------------------------------

/// SQLite's heap peak for one `MATCH` of `pieces` on the store file `path`, through a
/// plain connection: `(rows matched, peak bytes)`.
fn raw_match(one: &Alone, path: &Path, pieces: &[&str]) -> (i64, u64) {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let query = pieces
        .iter()
        .map(|piece| format!("\"{}\"", piece.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" AND ");
    let (rows, taken) = measure(one, || {
        connection
            .query_row(
                "SELECT count(*) FROM logs_fts CROSS JOIN logs ON logs.seq = logs_fts.rowid \
                 WHERE logs_fts MATCH ?1",
                [&query],
                |row| row.get::<_, i64>(0),
            )
            .unwrap()
    });
    (rows, taken.sqlite_bytes)
}

/// The heap of the text index for the pieces of a phrase, by the length and kind of
/// the longest stored text and by the characters given to the index: the figures
/// behind `INDEX_TEXT_FACTOR`. First raw SQL, so that pieces of any length can be
/// tried; then every kind and length through the store, in every way of searching,
/// asserting that each search stays within what it held.
#[test]
#[ignore = "index-piece measurement; run in release mode on request"]
fn index_piece_memory_measured() {
    let one = alone();
    type Make = Box<dyn Fn(usize) -> String>;
    let kinds: [(&str, Make); 7] = [
        ("one character", Box::new(|bytes| repeated("a", bytes))),
        ("cycle of 2", Box::new(|bytes| repeated("ab", bytes))),
        ("cycle of 3", Box::new(|bytes| repeated("abc", bytes))),
        (
            "cycle of 16",
            Box::new(|bytes| repeated("0123456789abcdef", bytes)),
        ),
        (
            "one two-byte character",
            Box::new(|bytes| repeated("ä", bytes)),
        ),
        (
            "log-like",
            Box::new(|bytes| long_text(Kind::Realistic, bytes, 1)),
        ),
        (
            "random ASCII",
            Box::new(|bytes| long_text(Kind::Noisy, bytes, 1)),
        ),
    ];
    let longest = usize::try_from(MAX_TEXT_BYTES).unwrap();
    // Up to the longest text a store takes; just below it as well, because the
    // buffers of the index double at a power of two.
    let lengths = [64 << 10, 256 << 10, longest - 16, longest];
    let mut worst_rate = 0.0f64;
    let mut worst_share = 0.0f64;
    println!(
        "raw: heap peak in MiB (and what the long text adds, in bytes per byte of it and per \
         character given to the index)"
    );
    for (kind, make) in &kinds {
        for bytes in lengths {
            let scratch = Scratch::new(&one, "piece-memory");
            let text = make(bytes);
            let store = store_with_long_text(&scratch, "s", &text);
            let path = scratch.paths().store_path("s");
            // What a match that does not touch the long text takes: the connection and
            // its readers. The rates are of what the long text adds to that.
            let (_, base) = raw_match(&one, &path, &["/short/"]);
            let mut line = format!("{kind:24} {bytes:8} B:");
            for chars in [3usize, 4, 6, 8, 12, 16, 32] {
                let end = text
                    .char_indices()
                    .nth(chars)
                    .map_or(text.len(), |(at, _)| at);
                let (rows, peak) = raw_match(&one, &path, &[&text[..end]]);
                assert!(rows >= 1);
                let rate = peak.saturating_sub(base) as f64 / (bytes as f64 * chars as f64);
                // The store gives the index no piece longer than this.
                if chars <= l2d_store::INDEX_PIECE_CHARS {
                    worst_rate = worst_rate.max(rate);
                }
                line.push_str(&format!(" {chars}: {:.1} ({rate:.2})", peak as f64 / MIB));
            }
            // Several pieces at once: two, three and four pieces of four characters.
            let end = text.char_indices().nth(4).map_or(text.len(), |(at, _)| at);
            let piece = &text[..end];
            for count in [2usize, 3, 4] {
                let (_, peak) = raw_match(&one, &path, &vec![piece; count]);
                let rate = peak.saturating_sub(base) as f64 / (bytes as f64 * (4 * count) as f64);
                worst_rate = worst_rate.max(rate);
                line.push_str(&format!(" {count}x4: {:.1} ({rate:.2})", peak as f64 / MIB));
            }
            println!("{line}");

            // Through the store: what a search holds covers it.
            for (what, filter) in phrase_filters(&text) {
                let what = format!("  {kind}, {bytes} B, {what}");
                worst_share =
                    worst_share.max(assert_phrase_search_within(&one, &what, &store, &filter));
            }
            drop(store);
            assert_eq!(one.sqlite_used(), 0);
        }
    }
    println!(
        "raw, pieces as the store gives them (up to four characters, up to four pieces): the \
         long text adds at most {worst_rate:.2} bytes per byte of it and per character given to \
         the index (held: {INDEX_TEXT_FACTOR})"
    );
    println!("through the store: the most SQLite took of what a search held: {worst_share:.2}");
    assert!(worst_rate <= INDEX_TEXT_FACTOR as f64);
    assert!(worst_share <= 1.0);
}

/// How many logs a piece of a phrase leaves, by the length of the piece and by where in
/// the phrase it is taken: phrases are 24-character cuts of CSIC logs.
#[test]
#[ignore = "piece-selectivity measurement; run in release mode on request"]
fn index_piece_selectivity_measured() {
    let one = alone();
    let Some(logs) = csic("index_piece_selectivity_measured") else {
        return;
    };
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    let scratch = Scratch::new(&one, "piece-selectivity");
    let paths = scratch.paths();
    let mut store = Store::create(&paths, "csic", &identity(8)).unwrap();
    store
        .add_with(
            &rows_of(&logs, "data/csic-2010/csic_database.csv", 8),
            &identity(8),
            &mut roomy_budget(),
        )
        .unwrap();
    drop(store);
    drop(logs);
    let path = paths.store_path("csic");
    let connection = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let count = |piece: &str| -> (i64, f64) {
        let started = Instant::now();
        let rows = connection
            .query_row(
                "SELECT count(*) FROM logs_fts WHERE logs_fts MATCH ?1",
                [format!("\"{}\"", piece.replace('"', "\"\""))],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        (rows, started.elapsed().as_secs_f64() * 1e3)
    };
    // 200 phrases: 24 characters from a fixed place of every 128th log that is ASCII and
    // long enough; a third each from the start, the middle and the end of the log.
    let mut phrases: Vec<&str> = Vec::new();
    for (index, text) in texts.iter().enumerate().step_by(128) {
        if !text.is_ascii() || text.len() < 80 {
            continue;
        }
        let start = match index / 128 % 3 {
            0 => 0,
            1 => (text.len() - 24) / 2,
            _ => text.len() - 24,
        };
        phrases.push(&text[start..start + 24]);
    }
    println!(
        "{} phrases of 24 characters; logs that hold the whole phrase and logs that a piece leaves (median, mean, most), with the mean time of the index",
        phrases.len()
    );
    let whole: Vec<i64> = phrases
        .iter()
        .map(|phrase| texts.iter().filter(|text| text.contains(*phrase)).count() as i64)
        .collect();
    let summary = |name: &str, counts: &mut Vec<i64>, ms: f64| {
        counts.sort_unstable();
        println!(
            "  {name:28}: median {:6}, mean {:8.0}, most {:6}; {ms:.2} ms",
            counts[counts.len() / 2],
            counts.iter().sum::<i64>() as f64 / counts.len() as f64,
            counts[counts.len() - 1],
        );
    };
    summary("the whole phrase", &mut whole.clone(), 0.0);
    // Two and three pieces at once: the ends, and the ends with the middle.
    let count_all = |pieces: &[&str]| -> (i64, f64) {
        let started = Instant::now();
        let query = pieces
            .iter()
            .map(|piece| format!("\"{}\"", piece.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" AND ");
        let rows = connection
            .query_row(
                "SELECT count(*) FROM logs_fts WHERE logs_fts MATCH ?1",
                [query],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        (rows, started.elapsed().as_secs_f64() * 1e3)
    };
    for chars in [3usize, 4, 6, 8] {
        for middle in [false, true] {
            let mut counts = Vec::new();
            let mut ms = 0.0;
            for phrase in &phrases {
                let mut pieces = vec![&phrase[..chars], &phrase[phrase.len() - chars..]];
                if middle {
                    let start = (phrase.len() - chars) / 2;
                    pieces.push(&phrase[start..start + chars]);
                }
                let (rows, took) = count_all(&pieces);
                counts.push(rows);
                ms += took;
            }
            let name = if middle {
                format!("first, middle, last {chars}")
            } else {
                format!("first and last {chars}")
            };
            summary(&name, &mut counts, ms / phrases.len() as f64);
        }
    }
    for chars in [3usize, 4, 6, 8, 12, 16, 24] {
        for (place, pick) in [("first", 0usize), ("last", 1), ("middle", 2)] {
            let mut counts = Vec::new();
            let mut ms = 0.0;
            for phrase in &phrases {
                let start = match pick {
                    0 => 0,
                    1 => phrase.len() - chars,
                    _ => (phrase.len() - chars) / 2,
                };
                let (rows, took) = count(&phrase[start..start + chars]);
                counts.push(rows);
                ms += took;
            }
            let ms = ms / phrases.len() as f64;
            summary(&format!("{place} {chars} characters"), &mut counts, ms);
        }
    }
}
