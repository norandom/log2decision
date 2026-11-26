//! Triager task 3.1, requirement 6.2 measured: an occurrence projection, its re-scan,
//! the fields and histogram views and the field and time drills stay within the
//! memory limit and within what the budget admitted, and what they leave behind is
//! held.
//!
//! As `service_drill_memory.rs`, this test binary counts every heap allocation of the
//! process. One run goes through every operation of the task, keeps every reply
//! alive, and then compares: the heap may not have grown by more than the limit
//! leaves, nor by more than the peak the budget counted as admitted; what stays
//! allocated while the service lives is at most what it reports as held. The
//! projections keep a field of 20,000 values of 200 bytes (about 4 MB) and a table of
//! 48,000 occurrences in memory (about 1.5 MB each), each far larger than the
//! tolerance of the last comparison, so that one of them that is kept and not held
//! fails it; so do the drilled views beside them.

mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_triager::service::ADAPTER_FIXED_BYTES;
use serde_json::{Value, json};
use support::{BASELINE, HostOptions, MIB, start, write_archive};

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

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// Lines of each of the two entries; half of the second entry's lines repeat the
/// first's, so an original is read on two hosts and two days.
const LINES: usize = 24_000;
/// Distinct values of the wide field, each 200 bytes long.
const WIDE_VALUES: usize = 20_000;
/// What the job table keeps of one job beside its result.
const JOB_RECORD_BYTES: u64 = 512;
/// What a live service keeps beyond its artifacts whatever it ran: 4,552 to 4,622
/// bytes were measured in every run of this binary, also in one that committed
/// nothing (its open was refused); the three drilled views alone hold more than four
/// times this, the dictionaries and the table hundreds of times.
const SERVICE_RESIDUAL_BYTES: u64 = 8 * 1024;

fn entries() -> Vec<(String, Vec<String>)> {
    (0..2)
        .map(|entry| {
            let lines = (0..LINES)
                .map(|line| {
                    // The second entry repeats every other line of the first.
                    let key = if entry == 1 && line % 2 == 0 {
                        line
                    } else {
                        line + entry * LINES
                    };
                    format!(
                        "2026-01-0{}T{:02}:{:02}:00Z login user=u{:03} wide={:0>200} src=10.{}.{}",
                        1 + key % 3,
                        key % 24,
                        key % 60,
                        key % 400,
                        key % WIDE_VALUES,
                        key % 250,
                        key % 199
                    )
                })
                .collect();
            (
                format!("logs/host-{entry}/2026-01-0{}.log", entry + 1),
                lines,
            )
        })
        .collect()
}

fn refused(value: &Value) -> bool {
    value["status"] == "error" && value["error"]["code"] == "refused"
}

/// One run of every operation of the task under `usable` bytes; returns the steps
/// with what they gave, `refused` where the budget refused one.
fn run(dir: &std::path::Path, usable: u64, measured: bool) -> Vec<(String, String)> {
    let options = HostOptions {
        usable,
        ..HostOptions::default()
    };
    let _ = std::fs::remove_dir_all(dir.join("work"));
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let service = start(dir, "projection-memory", &options).unwrap();
    let usage = service.usage();
    let mut outcome: Vec<(String, String)> = Vec::new();
    let mut kept: Vec<String> = Vec::new();
    // The steps whose result only the job table keeps: the views.
    let mut unowned = 0u64;
    let mut call = |step: &str, tool: &str, args: Value, field: &str| -> Option<Value> {
        let reply = service.call(tool, &args);
        let value: Value = serde_json::from_str(&reply.json).unwrap();
        if refused(&value) {
            let reason: String = value["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect();
            eprintln!("  {step} refused under {} MiB: {reason}", usable / MIB);
            outcome.push((step.into(), "refused".into()));
            kept.push(reply.json);
            return None;
        }
        assert!(
            matches!(value["status"].as_str(), Some("ready" | "time_unavailable")),
            "{step}: {value}"
        );
        if tool == "triager_inspect" && args["view"] != "evidence" {
            unowned += reply.json.len() as u64;
        }
        outcome.push((
            step.into(),
            value
                .pointer(field)
                .cloned()
                .unwrap_or(Value::Null)
                .to_string(),
        ));
        kept.push(reply.json);
        Some(value)
    };
    let inspect = |handle: &Value, view: &str, extra: Value| {
        let mut args = json!({"v": 1, "handle": handle, "view": view});
        for (key, value) in extra.as_object().unwrap() {
            args[key] = value.clone();
        }
        args
    };
    (|| -> Option<()> {
        let opened = call(
            "open",
            "triager_open",
            json!({"v": 1, "source": {"ref": "private"}, "projections": {
                "fields": [{"name": "host", "from": "entry", "capture": "^logs/([^/]+)/"},
                           {"name": "user", "from": "kv"},
                           {"name": "wide", "from": "kv"}],
                "time": {"from": "text"},
                "max_values": WIDE_VALUES}}),
            "/projection/projection",
        )?;
        let snapshot = opened["snapshot"].clone();
        call(
            "rescan",
            "triager_open",
            json!({"v": 1, "snapshot": snapshot, "projections": {
                "fields": [{"name": "src", "from": "kv"}]}}),
            "/revision",
        )?;
        call(
            "fields",
            "triager_inspect",
            inspect(&snapshot, "fields", json!({})),
            "/fields",
        )?;
        call(
            "wide",
            "triager_inspect",
            inspect(&snapshot, "fields", json!({"field": "wide", "limit": 100})),
            "/field",
        )?;
        call(
            "hosts",
            "triager_inspect",
            inspect(&snapshot, "fields", json!({"field": "host"})),
            "/field",
        )?;
        call(
            "histogram",
            "triager_inspect",
            inspect(&snapshot, "histogram", json!({"bucket_seconds": 3_600})),
            "/time",
        )?;
        let host = call(
            "drill-host",
            "triager_refine",
            json!({"v": 1, "set": snapshot,
                   "action": {"kind": "field", "field": "host", "value": "host-1"}}),
            "/revision",
        )?;
        let view = host["view"].clone();
        let day = call(
            "drill-day",
            "triager_refine",
            json!({"v": 1, "set": view, "action": {"kind": "time",
                   "start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"}}),
            "/revision",
        )?;
        call(
            "drill-wide",
            "triager_refine",
            json!({"v": 1, "set": snapshot,
                   "action": {"kind": "field", "field": "wide", "value": format!("{:0>200}", 7)}}),
            "/revision",
        )?;
        call(
            "day-fields",
            "triager_inspect",
            inspect(
                &day["view"],
                "fields",
                json!({"field": "user", "limit": 100}),
            ),
            "/field",
        )?;
        call(
            "day-histogram",
            "triager_inspect",
            inspect(&day["view"], "histogram", json!({})),
            "/time",
        )?;
        call(
            "evidence",
            "triager_inspect",
            json!({"v": 1, "handle": view, "view": "evidence",
                   "record": "r-0000000000000001"}),
            "/record",
        )?;
        Some(())
    })();
    let replies: usize = kept.iter().map(String::capacity).sum();
    let retained = CURRENT
        .load(Ordering::Relaxed)
        .saturating_sub(before + replies) as u64;
    let held = service.held_bytes();
    let growth = PEAK.load(Ordering::Relaxed).saturating_sub(before) as u64;
    let admitted = usage.report().counted_peak_bytes.saturating_sub(BASELINE);
    let artifacts = held - ADAPTER_FIXED_BYTES;
    eprintln!(
        "projection usable {:>4} MiB: heap {:>6.2} MiB = {:.2} x admitted ({:>6.2} MiB); retained {} of {} bytes held, {} beyond the artifacts; {}",
        usable / MIB,
        growth as f64 / MIB as f64,
        growth as f64 / admitted.max(1) as f64,
        admitted as f64 / MIB as f64,
        retained,
        artifacts,
        retained as i64 - artifacts as i64 - unowned as i64,
        outcome
            .iter()
            .map(|(step, result)| format!(
                "{step}={}",
                if result == "refused" { "refused" } else { "ok" }
            ))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if !measured {
        return outcome;
    }
    assert!(
        growth <= usable,
        "the heap grew by {growth} bytes under a limit that leaves {usable} bytes"
    );
    assert!(
        growth <= admitted,
        "the heap grew by {growth} bytes, and the budget admitted a peak of {admitted} bytes"
    );
    assert!(
        retained
            <= artifacts
                + unowned
                + SERVICE_RESIDUAL_BYTES
                + JOB_RECORD_BYTES * outcome.len() as u64,
        "the live service keeps {retained} bytes; its artifacts hold {artifacts} and the job \
         table {unowned} of results"
    );
    drop(service);
    drop(kept);
    assert!(
        std::fs::read_dir(dir.join("work")).map_or(true, |mut left| left.next().is_none()),
        "the work directory is emptied with the service"
    );
    outcome
}

#[test]
fn projections_and_their_views_stay_within_the_limit_and_hold_what_they_keep() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let entries = entries();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    // What the process allocates once and keeps belongs to its own share.
    let warm = run(dir.path(), 512 * MIB, false);
    assert!(
        warm.iter().all(|(_, result)| result != "refused"),
        "{warm:?}"
    );
    assert_eq!(warm.len(), 12, "{warm:?}");
    let complete = run(dir.path(), 512 * MIB, true);
    assert_eq!(
        complete, warm,
        "the same run gives the same handles and revisions"
    );
    // Under tighter limits an operation completes with the same result or is refused
    // with the figures; in both cases the heap stayed within what was admitted.
    let mut refusals = 0;
    for usable in [192 * MIB, 128 * MIB, 96 * MIB, 64 * MIB, 48 * MIB, 32 * MIB] {
        let tight = run(dir.path(), usable, true);
        for (step, result) in &tight {
            if result == "refused" {
                refusals += 1;
            } else {
                let full = complete.iter().find(|(known, _)| known == step).unwrap();
                assert_eq!(result, &full.1, "{step} under {} MiB", usable / MIB);
            }
        }
    }
    assert!(refusals > 0, "no limit of the ladder refused an operation");
}
