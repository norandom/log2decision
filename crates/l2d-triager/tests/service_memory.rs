//! Triager task 1.2, requirements 1.2 and 6.2 measured: the service's operations stay
//! within the memory limit and within what the budget admitted, with everything they
//! return and retain still alive.
//!
//! As `scope_memory.rs`, this test binary counts every heap allocation of the process
//! (a counting global allocator around the system allocator), the service's worker and
//! the archive's reader threads included. A ladder of memory limits is tried at two
//! batch sizes, with deduplication in memory and on disk. Whether an operation
//! completes or is refused, the heap may not have grown by more than the limit leaves
//! beside the share kept for the process, nor by more than the peak the budget counted
//! as admitted; what stays allocated while the service lives is at most what it reports
//! as held. Every completed run gives the same handles, revisions and rows. A request
//! line far beyond the request limit is answered without being kept.

mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Read;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_core::resource::{BusyChoice, ResourceGuard};
use l2d_triager::mcp::{self, MAX_REQUEST_BYTES, MAX_REQUEST_VALUES};
use l2d_triager::scope::IndexPlacement;
use l2d_triager::service::{ADAPTER_FIXED_BYTES, TablePlacement};
use serde_json::{Value, json};
use support::{BASELINE, HostOptions, MIB, Orig, oracle, start, write_archive};

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

/// The measurements share the process's counters, so they run one after the other.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const ORIGINALS: usize = 90_000;
/// What the job table of one measured run may keep of the adapter's share beside the
/// results, which the artifacts hold a second time for it: the records of at most
/// 27 jobs. One membership table of a run takes 22,500 bytes, more than this: a
/// single table that is in memory and not held does not hide in it (the open test
/// gap of task 1.2, whose tables took 15,000 bytes each).
const JOB_TABLE_BYTES: u64 = 16 * 1024;
const _: () = assert!(ORIGINALS as u64 / 4 > JOB_TABLE_BYTES);
/// The steps of a run that nothing refuses.
const STEPS: usize = 27;

fn entries() -> Vec<(String, Vec<String>)> {
    (0..3)
        .map(|entry| {
            let mut lines: Vec<String> = (0..ORIGINALS / 3)
                .map(|line| {
                    format!(
                        "host-{entry} {} request {line} from 10.{entry}.{}.{} took {} ms",
                        if line % 4 == 0 { "login" } else { "fetch" },
                        line % 250,
                        line % 199,
                        line * 7 % 1000
                    )
                })
                .collect();
            // Repeats, so that occurrences and originals differ.
            for line in 0..2_000 {
                lines.push(lines[line * 3].clone());
            }
            (format!("logs/host-{entry}.log"), lines)
        })
        .collect()
}

/// What a completed run gave: per step its handle or revision, or `refused`.
type Outcome = Vec<(String, String)>;

fn refused(value: &Value) -> bool {
    value["status"] == "error" && value["error"]["code"] == "refused"
}

/// How one run of the operations is set up.
#[derive(Clone, Copy)]
struct Setup<'a> {
    usable: u64,
    batch_rows: Option<u64>,
    dedup_set_texts: Option<u64>,
    /// Where the sets keep their membership and scores, and the snapshot its relation.
    tables: TablePlacement,
    first: &'a Orig,
    /// The lines of the membership file `fixture`.
    fixture: &'a str,
    measured: bool,
}

fn run(dir: &std::path::Path, setup: &Setup<'_>) -> Outcome {
    let Setup {
        usable,
        batch_rows,
        dedup_set_texts,
        tables,
        first,
        fixture,
        measured,
    } = *setup;
    let options = HostOptions {
        usable,
        batch_rows,
        dedup_set_texts,
        tables,
        index: match tables {
            TablePlacement::Disk => IndexPlacement::Disk,
            TablePlacement::MemoryWhenAdmitted => IndexPlacement::MemoryWhenAdmitted,
        },
        memberships: vec![("fixture".into(), fixture.to_string())],
        ..HostOptions::default()
    };
    let _ = std::fs::remove_dir_all(dir.join("work"));
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let started = std::time::Instant::now();
    let service = match start(dir, "memory", &options) {
        Ok(service) => service,
        Err(l2d_core::L2dError::Refused { reason }) => {
            eprintln!(
                "service usable {:>3} MiB: not started: {reason}",
                usable / MIB
            );
            return vec![("start".into(), "refused".into())];
        }
        Err(other) => panic!("{other:?}"),
    };
    let usage = service.usage();
    let mut outcome = Outcome::new();
    // Everything a call returns stays alive until the end of the measurement.
    let mut kept: Vec<String> = Vec::new();
    let mut on_disk = 0;
    let mut call = |step: &str, tool: &str, args: Value, field: &str| -> Option<Value> {
        let reply = service.call(tool, &args);
        let value: Value = serde_json::from_str(&reply.json).unwrap();
        kept.push(reply.json);
        if refused(&value) {
            let reason = value["error"]["message"].as_str().unwrap_or_default();
            let shown: String = reason.chars().take(260).collect();
            eprintln!("  {step} refused under {} MiB: {shown}", usable / MIB);
            outcome.push((step.into(), "refused".into()));
            return None;
        }
        assert_eq!(value["status"], "ready", "{step}: {value}");
        if value["cost"]["tables_on_disk"] == true {
            on_disk += 1;
        }
        outcome.push((step.into(), value[field].to_string()));
        Some(value)
    };
    (|| {
        let opened = call(
            "open",
            "triager_open",
            json!({"v": 1, "source": {"ref": "private"}}),
            "snapshot",
        )?;
        let snapshot = opened["snapshot"].clone();
        let a = call(
            "score",
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {
                "kind": "score", "model": "sum", "top_k": 5000, "ties": "record_id_ascending"}}),
            "revision",
        );
        let b = call(
            "filter",
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {
                "kind": "filter", "syntax": "regex", "query": r"login request \d+7 "}}),
            "revision",
        );
        let (a, b) = (a?, b?);
        let view = call(
            "compose",
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot,
                   "sets": {"a": a["set"], "b": b["set"]}, "expr": {"difference": ["a", "b"]}}),
            "revision",
        )?;
        call(
            "regions",
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "regions",
                   "sets": {"a": a["set"], "b": b["set"]}}),
            "regions",
        )?;
        call(
            "records",
            "triager_inspect",
            json!({"v": 1, "handle": view["view"], "view": "records", "limit": 100,
                   "sets": {"a": a["set"], "b": b["set"]}}),
            "rows",
        )?;
        let evidence = call(
            "evidence",
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "evidence", "record": first.id(),
                   "sets": {"a": a["set"], "b": b["set"]},
                   "projection": ["id", "membership", "explanation", "excerpt"]}),
            "excerpt",
        )?;
        // A registered membership, joined by key; another rule from the scores of the
        // first; the filter and the membership again inside the view, from the sets
        // over the whole snapshot.
        let fixture = call(
            "membership",
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {"kind": "membership", "ref": "fixture"}}),
            "revision",
        )?;
        let derived = call(
            "threshold",
            "triager_select",
            json!({"v": 1, "scope": snapshot,
                   "producer": {"kind": "score", "model": "sum", "threshold": 0.9}}),
            "revision",
        )?;
        assert_eq!(derived["cost"]["archive_scans"], 0, "{derived}");
        let within = call(
            "filter-in-view",
            "triager_select",
            json!({"v": 1, "scope": view["view"], "producer": {
                "kind": "filter", "syntax": "regex", "query": r"login request \d+7 "}}),
            "revision",
        )?;
        assert_eq!(within["cost"]["archive_scans"], 0, "{within}");
        call(
            "regions-3",
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "regions",
                   "sets": {"a": a["set"], "b": b["set"], "c": fixture["set"]}}),
            "regions",
        )?;
        // More selections and views, so that what the membership tables take is far
        // more than the tolerance of the check below: twenty-two tables in all.
        for index in 1..=4 {
            let more = call(
                &format!("filter-{index}"),
                "triager_select",
                json!({"v": 1, "scope": snapshot, "producer": {
                    "kind": "filter", "syntax": "literal", "query": format!("request {index}7 from")}}),
                "revision",
            )?;
            for operator in ["union", "intersection", "difference"] {
                call(
                    &format!("{operator}-{index}"),
                    "triager_compose",
                    json!({"v": 1, "mode": "view", "snapshot": snapshot,
                           "sets": {"a": a["set"], "m": more["set"]},
                           "expr": {operator: ["a", "m"]}}),
                    "revision",
                )?;
            }
        }
        Some(evidence)
    })();
    // What stays allocated while the service lives is what it reports as held, the
    // replies kept here aside.
    let replies: usize = kept.iter().map(String::capacity).sum();
    let retained = CURRENT
        .load(Ordering::Relaxed)
        .saturating_sub(before + replies) as u64;
    let held = service.held_bytes();
    let growth = PEAK.load(Ordering::Relaxed).saturating_sub(before) as u64;
    let admitted = usage.report().counted_peak_bytes.saturating_sub(BASELINE);
    // With the tables on disk every set that was made says so, and nothing of a
    // table is in memory.
    let made = outcome
        .iter()
        .filter(|(step, result)| {
            result != "refused"
                && !["open", "regions", "regions-3", "records", "evidence"].contains(&step.as_str())
        })
        .count();
    if tables == TablePlacement::Disk {
        assert_eq!(on_disk, made, "{outcome:?}");
    }
    eprintln!(
        "service usable {:>3} MiB batch {:>5} set {:>5} tables {}: {on_disk} of {made} sets on disk; heap {:>6.2} MiB = {:.2} x usable, {:.2} x admitted ({:>6.2} MiB); retained {:.2} of {:.2} MiB held, {} bytes beyond the artifacts; {:.1?}; {}",
        usable / MIB,
        batch_rows.map_or("auto".into(), |rows| rows.to_string()),
        dedup_set_texts.map_or("auto".into(), |texts| texts.to_string()),
        if tables == TablePlacement::Disk {
            "disk"
        } else {
            "auto"
        },
        growth as f64 / MIB as f64,
        growth as f64 / usable as f64,
        growth as f64 / admitted.max(1) as f64,
        admitted as f64 / MIB as f64,
        retained as f64 / MIB as f64,
        held as f64 / MIB as f64,
        retained as i64 - (held - ADAPTER_FIXED_BYTES) as i64,
        started.elapsed(),
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
    // The artifacts are held byte for byte; of the adapter's fixed share only the
    // table of this run's few jobs and their results is in use between calls.
    let artifacts = held - ADAPTER_FIXED_BYTES;
    assert!(
        retained <= artifacts + JOB_TABLE_BYTES,
        "the live service keeps {retained} bytes, and its artifacts hold {artifacts}"
    );
    drop(service);
    drop(kept);
    assert!(
        std::fs::read_dir(dir.join("work")).map_or(true, |mut left| left.next().is_none()),
        "the work directory is emptied with the service"
    );
    outcome
}

/// Runs every operation once on a small archive, outside any measurement: what the
/// process allocates once and keeps (thread pools, the tables of the regex and frame
/// libraries) belongs to the share kept for the process itself, not to a run.
fn warm_up() {
    let dir = tempfile::tempdir().unwrap();
    let entries = support::auth_entries();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    let universe = oracle(&entries, ".log");
    let fixture = fixture(&universe);
    for (batch_rows, dedup_set_texts, tables) in [
        (None, None, TablePlacement::MemoryWhenAdmitted),
        (Some(64), Some(20), TablePlacement::MemoryWhenAdmitted),
        (Some(64), Some(20), TablePlacement::Disk),
    ] {
        let outcome = run(
            dir.path(),
            &Setup {
                usable: 128 * MIB,
                batch_rows,
                dedup_set_texts,
                tables,
                first: &universe[0],
                fixture: &fixture,
                measured: false,
            },
        );
        assert!(outcome.iter().all(|(_, result)| result != "refused"));
    }
}

/// A membership file over `universe`: every fourth original is assessed, in the
/// reverse order of the record IDs, a third of them in, out and abstained each.
fn fixture(universe: &[Orig]) -> String {
    let mut lines = String::new();
    for (index, original) in universe.iter().enumerate().rev().step_by(4) {
        let state = ["in", "out", "abstain"][index % 3];
        lines.push_str(&support::membership_line(&original.id(), state));
    }
    lines
}

#[test]
fn the_operations_stay_within_the_limit_and_the_admitted_peak() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let entries = entries();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries[..1]);
    let universe = oracle(&entries, ".log");
    assert_eq!(universe.len(), ORIGINALS);
    let first = universe[ORIGINALS / 2].clone();
    let fixture = fixture(&universe);
    drop(universe);
    drop(entries);
    warm_up();

    let mut complete: Option<Outcome> = None;
    let mut completed = 0;
    let mut completed_on_disk = 0;
    let mut refusals = 0;
    let (auto, disk) = (TablePlacement::MemoryWhenAdmitted, TablePlacement::Disk);
    for (usable, placements) in [
        (10 * MIB, &[auto, disk][..]),
        (16 * MIB, &[auto, disk]),
        (24 * MIB, &[auto]),
        (40 * MIB, &[auto, disk]),
        (128 * MIB, &[auto]),
    ] {
        for (batch_rows, dedup_set_texts, tables) in placements
            .iter()
            .flat_map(|tables| [(None, None, *tables), (Some(1_024), Some(4_000), *tables)])
        {
            let outcome = run(
                dir.path(),
                &Setup {
                    usable,
                    batch_rows,
                    dedup_set_texts,
                    tables,
                    first: &first,
                    fixture: &fixture,
                    measured: true,
                },
            );
            // A step that completed gave what it gives at every limit.
            if let Some(reference) = &complete {
                for (step, result) in &outcome {
                    if result != "refused" {
                        let expected = reference.iter().find(|(name, _)| name == step).unwrap();
                        assert_eq!(result, &expected.1, "{step} at {usable}");
                    }
                }
            }
            if outcome.iter().all(|(_, result)| result != "refused") && outcome.len() == STEPS {
                completed += 1;
                completed_on_disk += usize::from(tables == disk);
                complete.get_or_insert(outcome);
            } else {
                refusals += 1;
            }
        }
    }
    // The ladder shows both: limits that are refused and limits that complete, with
    // the tables in memory and with the tables on disk.
    assert!(completed >= 2, "{completed} runs completed");
    assert!(
        completed_on_disk >= 2,
        "{completed_on_disk} runs completed with the tables on disk"
    );
    assert!(refusals >= 1, "no limit of the ladder was refused");
}

/// What one selection under one limit gave, with the heap it took.
struct Attempt {
    selected: Value,
    /// The peak of the heap above its level before the service started.
    run_peak: u64,
    /// The peak of the heap during the selection above its level before it.
    select_peak: u64,
    /// The highest figure the budget admitted, the share of the process aside.
    admitted: u64,
    scored_by_the_server: u64,
}

/// Opens the archive under `usable` bytes and selects the top `top_k` of the test
/// scorer; `None` when the open or the selection is refused.
fn select_under(
    dir: &std::path::Path,
    usable: u64,
    tables: TablePlacement,
    top_k: usize,
) -> Option<Attempt> {
    // Small batches and deduplication on disk, so that the read takes little heap of
    // its own. What it keeps per original (3.7 MB here) is still about one list of
    // candidates: a second list made for an instant between the two reads would not
    // be the peak of the selection, and this test would not see it. That the
    // candidates are emptied without a second list is measured where it happens, in
    // `candidates_are_emptied_and_used_again_without_a_second_list` (`ops.rs`).
    let options = HostOptions {
        usable,
        tables,
        batch_rows: Some(1_024),
        dedup_set_texts: Some(2_000),
        ..HostOptions::default()
    };
    let _ = std::fs::remove_dir_all(dir.join("work"));
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let service = start(dir, "memory", &options).ok()?;
    let call = |tool: &str, args: Value| -> Option<Value> {
        let value: Value = serde_json::from_str(&service.call(tool, &args).json).unwrap();
        if refused(&value) {
            return None;
        }
        assert_eq!(value["status"], "ready", "{value}");
        Some(value)
    };
    let opened = call(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private"}}),
    )?;
    let open_peak = PEAK.load(Ordering::Relaxed).saturating_sub(before) as u64;
    let level = CURRENT.load(Ordering::Relaxed);
    PEAK.store(level, Ordering::Relaxed);
    let selected = call(
        "triager_select",
        json!({"v": 1, "scope": opened["snapshot"], "producer": {
            "kind": "score", "model": "sum", "top_k": top_k, "ties": "record_id_ascending"}}),
    )?;
    let peak = PEAK.load(Ordering::Relaxed);
    let overview = call(
        "triager_inspect",
        json!({"v": 1, "handle": opened["snapshot"], "view": "overview"}),
    )?;
    Some(Attempt {
        selected,
        run_peak: open_peak.max(peak.saturating_sub(before) as u64),
        select_peak: peak.saturating_sub(level) as u64,
        admitted: service
            .usage()
            .report()
            .counted_peak_bytes
            .saturating_sub(BASELINE),
        scored_by_the_server: overview["activity"]["rows_scored"].as_u64().unwrap(),
    })
}

/// Requirement 6.2 by admission, in the default suite and with the heap measured
/// (review round 1 of task 1.3): under the lowest memory limit at which a selection
/// completes, the read of the source is refused with the tables in memory, and the
/// selection is placed on disk and read again. The top-k is every original, so the
/// kept candidates are the largest thing the selection holds: they are charged once,
/// and the selection that was read again takes no more heap than one that the host
/// placed on disk from the start.
///
/// The limit is found by bisection between 12 MiB, which refuses, and 48 MiB, which
/// admits the tables in memory, to 64 KiB: about ten services over an archive of
/// 150,000 originals. The tables take 637,500 bytes in memory and 327,680 on disk, so
/// the limits at which only the disk form completes span more than four such steps.
#[test]
fn a_selection_the_read_refuses_in_memory_is_read_again_onto_disk_within_its_share() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    const ORIGINALS: usize = 150_000;
    let dir = tempfile::tempdir().unwrap();
    let entries: Vec<(String, Vec<String>)> = (0..3)
        .map(|entry| {
            let lines = (0..ORIGINALS / 3)
                .map(|line| format!("host-{entry} request {line} took {} ms", line * 7 % 1000))
                .collect();
            (format!("logs/host-{entry}.log"), lines)
        })
        .collect();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries[..1]);
    drop(entries);
    warm_up();
    let auto = TablePlacement::MemoryWhenAdmitted;
    let dir = dir.path();

    let (mut refused_at, mut completes_at) = (12 * MIB, 48 * MIB);
    assert!(select_under(dir, refused_at, auto, ORIGINALS).is_none());
    let roomy = select_under(dir, completes_at, auto, ORIGINALS).expect("48 MiB complete");
    assert_eq!(roomy.selected["cost"]["tables_on_disk"], false);
    assert_eq!(roomy.selected["counts"]["in"], ORIGINALS as u64);
    let mut lowest = None;
    while completes_at - refused_at > 64 * 1024 {
        let middle = refused_at + (completes_at - refused_at) / 2;
        match select_under(dir, middle, auto, ORIGINALS) {
            Some(attempt) => {
                completes_at = middle;
                lowest = Some(attempt);
            }
            None => refused_at = middle,
        }
    }
    let lowest = lowest.expect("a limit below 48 MiB completes");
    let cost = &lowest.selected["cost"];
    eprintln!(
        "the lowest limit that completes leaves {completes_at} bytes: tables on disk {}, read \
         again {}, archive scans {}, rows scored {}; heap {} of {} admitted, {} during the \
         selection",
        cost["tables_on_disk"],
        cost["read_again_for_disk"],
        cost["archive_scans"],
        cost["rows_scored"],
        lowest.run_peak,
        lowest.admitted,
        lowest.select_peak
    );
    // On disk, by admission: no option placed the tables there.
    assert_eq!(cost["tables_on_disk"], true);
    assert_eq!(cost["read_again_for_disk"], true);
    assert_eq!(cost["archive_scans"], 2);
    assert_eq!(lowest.selected["revision"], roomy.selected["revision"]);
    assert_eq!(lowest.selected["counts"], roomy.selected["counts"]);
    // The cost counts both reads, as the server's work counters do.
    assert_eq!(cost["rows_scored"], lowest.scored_by_the_server);
    assert!(cost["rows_scored"].as_u64().unwrap() >= ORIGINALS as u64);
    assert!(
        lowest.run_peak <= completes_at && lowest.run_peak <= lowest.admitted,
        "the heap grew by {} bytes; the limit leaves {completes_at} and the budget admitted {}",
        lowest.run_peak,
        lowest.admitted
    );
    // The same limit with the tables placed on disk from the start: at its peak the
    // selection that was read again holds what that one holds, and nothing of the
    // first read beside it (the tables it had in memory are 637,500 bytes).
    let placed = select_under(dir, completes_at, TablePlacement::Disk, ORIGINALS)
        .expect("the limit completes with the tables on disk");
    assert_eq!(placed.selected["cost"]["read_again_for_disk"], false);
    assert_eq!(placed.selected["revision"], roomy.selected["revision"]);
    eprintln!(
        "placed on disk from the start: {} bytes during the selection",
        placed.select_peak
    );
    assert!(
        lowest.select_peak <= placed.select_peak + 512 * 1024,
        "read again, the selection took {} bytes of heap, and {} when it was placed on disk \
         from the start",
        lowest.select_peak,
        placed.select_peak
    );
    // Two mebibytes above, the tables are admitted in memory beside the read.
    let above = select_under(dir, completes_at + 2 * MIB, auto, ORIGINALS).expect("above");
    assert_eq!(above.selected["cost"]["tables_on_disk"], false);
    assert_eq!(above.selected["revision"], roomy.selected["revision"]);
}

/// Yields one line of `length` bytes and then `tail`, without holding either.
struct LongLine {
    left: usize,
    tail: Vec<u8>,
    at: usize,
}

impl Read for LongLine {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.left > 0 {
            let take = self.left.min(out.len());
            out[..take].fill(b'x');
            self.left -= take;
            return Ok(take);
        }
        let take = (self.tail.len() - self.at).min(out.len());
        out[..take].copy_from_slice(&self.tail[self.at..self.at + take]);
        self.at += take;
        Ok(take)
    }
}

#[test]
fn a_request_beyond_the_limit_is_answered_without_being_held() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let entries = support::auth_entries();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    let service = start(dir.path(), "memory", &HostOptions::default()).unwrap();
    let opened: Value = serde_json::from_str(
        &service
            .call(
                "triager_open",
                &json!({"v": 1, "source": {"ref": "private"}}),
            )
            .json,
    )
    .unwrap();
    let largest_page = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "triager_inspect",
        "arguments": {"v": 1, "handle": opened["snapshot"], "view": "records", "limit": 100}}});
    let input = LongLine {
        left: 64 * MIB as usize,
        tail: format!("\n{largest_page}\n").into_bytes(),
        at: 0,
    };
    let mut output = Vec::with_capacity(256 * 1024);
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    mcp::serve(&service, input, &mut output).unwrap();
    let growth = PEAK.load(Ordering::Relaxed).saturating_sub(before) as u64;
    eprintln!(
        "a request line of 64 MiB and the largest page: the heap grew by {:.2} MiB of the {:.2} MiB kept for the adapter",
        growth as f64 / MIB as f64,
        ADAPTER_FIXED_BYTES as f64 / MIB as f64
    );
    assert!(
        growth <= ADAPTER_FIXED_BYTES,
        "the adapter grew the heap by {growth} bytes and {ADAPTER_FIXED_BYTES} are kept for it"
    );
    let replies: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(replies.len(), 2);
    assert_eq!(replies[0]["error"]["code"], -32600);
    assert!(
        replies[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains(&MAX_REQUEST_BYTES.to_string())
    );
    assert!(
        !replies[1]["result"]["structuredContent"]["rows"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

/// A tool call of exactly [`MAX_REQUEST_BYTES`] whose arguments hold `filler` and a
/// string that fills the rest of the line: as a JSON-RPC message and as a recipe line.
fn filled(snapshot: &str, id: usize, filler: &str) -> (String, String) {
    let arguments = |pad: usize| {
        format!(
            r#"{{"v":1,"handle":"{snapshot}","view":"overview","x":{filler},"pad":"{}"}}"#,
            "p".repeat(pad)
        )
    };
    let message = |pad: usize| {
        format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"tools/call","params":{{"name":"triager_inspect","arguments":{}}}}}"#,
            arguments(pad)
        )
    };
    let recipe = |pad: usize| format!(r#"{{"tool":"triager_inspect","args":{}}}"#, arguments(pad));
    let (message, recipe) = (
        message(MAX_REQUEST_BYTES - message(0).len()),
        recipe(MAX_REQUEST_BYTES - recipe(0).len()),
    );
    assert_eq!(message.len(), MAX_REQUEST_BYTES);
    assert_eq!(recipe.len(), MAX_REQUEST_BYTES);
    (message, recipe)
}

/// The shapes that take the most memory per value once parsed, each with about 900
/// values: below [`MAX_REQUEST_VALUES`] with the call around them.
fn dense_fillers() -> Vec<(&'static str, String)> {
    let list = |item: &str, count: usize| format!("[{}]", vec![item; count].join(","));
    let chain = |open: &str, close: &str, depth: usize| {
        format!("{}0{}", open.repeat(depth), close.repeat(depth))
    };
    vec![
        ("empty arrays", list("[]", 450)),
        ("arrays of one", list("[0]", 450)),
        ("objects of one", list(r#"{"a":0}"#, 450)),
        ("empty objects", list("{}", 450)),
        ("array chains", list(&chain("[", "]", 100), 9)),
        ("object chains", list(&chain(r#"{"a":"#, "}", 100), 9)),
        ("numbers", list("0", 900)),
        ("one string", "0".to_string()),
    ]
}

/// The adapter's share with everything it is said to cover at once: a full job table
/// (ended jobs with large results and idempotency keys, one job waiting for the host
/// and eight queued behind it with the largest arguments), the densest admitted
/// request through both adapters, the largest page and the largest job status.
#[test]
fn a_full_job_table_and_the_densest_requests_stay_within_the_adapters_share() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    warm_up();
    let dir = tempfile::tempdir().unwrap();
    let entries = support::auth_entries();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    let service = start(
        dir.path(),
        "memory",
        &HostOptions {
            inline_wait: Some(std::time::Duration::ZERO),
            ..HostOptions::default()
        },
    )
    .unwrap();
    let call = |tool: &str, args: Value| -> Value {
        serde_json::from_str(&service.call(tool, &args).json).unwrap()
    };
    // Waits for a job a call started and returns its status.
    let ended = |started: Value| -> Value {
        assert_eq!(started["status"], "running", "{started}");
        loop {
            let status = call(
                "triager_job",
                json!({"v": 1, "job": started["job"], "action": "status"}),
            );
            if !matches!(status["status"].as_str(), Some("waiting" | "running")) {
                return status;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    };
    let done = |started: Value| -> Value {
        let status = ended(started);
        assert_eq!(status["status"], "completed", "{status}");
        status["result"].clone()
    };

    // The artifacts the jobs below read: a snapshot, a set and three views whose
    // expressions make the evidence of a record large.
    let opened = done(call(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private"}}),
    ));
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let set = done(call(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "filter", "syntax": "literal", "query": "login"}}),
    ));
    let mut sets = serde_json::Map::new();
    for (at, name) in ["a", "b", "c"].iter().enumerate() {
        let view = done(call(
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"s": set["set"]},
                   "expr": {"union": vec![json!("s"); 60 - at]}}),
        ));
        sets.insert((*name).to_string(), view["view"].clone());
    }
    let page = call(
        "triager_inspect",
        json!({"v": 1, "handle": set["set"], "view": "records", "limit": 1}),
    );
    let record = page["rows"][0]["id"].clone();

    // What is sent below is built before the measurement, and its output has room.
    let fillers = dense_fillers();
    let mut messages: Vec<u8> = Vec::with_capacity(MIB as usize);
    let mut recipes = Vec::new();
    for (id, (_, filler)) in fillers.iter().enumerate() {
        let (message, recipe) = filled(&snapshot, id + 1, filler);
        messages.extend_from_slice(message.as_bytes());
        messages.push(b'\n');
        recipes.push(recipe.into_bytes());
    }
    // A line of the admitted length with too many values, in the shape that takes
    // the most memory once parsed (review round 1 measured 4.2 MB for it): chains
    // of arrays nested as deep as the parser goes, to the end of the line.
    let chain = format!("{}{}", "[".repeat(119), "]".repeat(119));
    let crowded = format!(
        "[{}]",
        vec![chain.as_str(); (MAX_REQUEST_BYTES - 2) / (chain.len() + 1)].join(",")
    );
    assert!(crowded.len() <= MAX_REQUEST_BYTES && crowded.len() > MAX_REQUEST_BYTES - 256);
    messages.extend_from_slice(crowded.as_bytes());
    messages.push(b'\n');
    let largest_page = json!({"jsonrpc": "2.0", "id": 100, "method": "tools/call", "params": {
        "name": "triager_inspect",
        "arguments": {"v": 1, "handle": snapshot, "view": "records", "limit": 100,
                      "sets": {"s": set["set"]}}}});
    messages.extend_from_slice(format!("{largest_page}\n").as_bytes());
    // The reads that go through the most blocks of committed tables: the regions of
    // three sets, and the evidence of a record with its rank among every score.
    let scored = done(call(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "score", "model": "sum", "top_k": 5, "ties": "record_id_ascending"}}),
    ));
    let regions = json!({"jsonrpc": "2.0", "id": 102, "method": "tools/call", "params": {
        "name": "triager_inspect",
        "arguments": {"v": 1, "handle": snapshot, "view": "regions",
                      "sets": {"s": set["set"], "t": scored["set"], "u": sets["a"]}}}});
    let ranked = json!({"jsonrpc": "2.0", "id": 103, "method": "tools/call", "params": {
        "name": "triager_inspect",
        "arguments": {"v": 1, "handle": snapshot, "view": "evidence", "record": record,
                      "sets": {"t": scored["set"]}}}});
    messages.extend_from_slice(format!("{regions}\n{ranked}\n").as_bytes());
    let mut output: Vec<u8> = Vec::with_capacity(MIB as usize);
    let mut recipe_output: Vec<u8> = Vec::with_capacity(MIB as usize);
    let queries: Vec<Value> = (0..10)
        .map(|index| {
            json!({"v": 1, "scope": snapshot, "idempotency_key": format!("queued-{index}"),
                   "producer": {"kind": "filter", "syntax": "literal",
                                "query": format!("{index}{}", "q".repeat(60_000))}})
        })
        .collect();
    let keys: Vec<String> = (0..64).map(|index| format!("{index:0>128}")).collect();
    let held_before = service.held_bytes();

    let before = CURRENT.load(Ordering::Relaxed);
    // Sixty-four ended jobs, each with a key of the longest length and a large result.
    let mut smallest = usize::MAX;
    let mut last_job = Value::Null;
    for key in &keys {
        let started = call(
            "triager_inspect",
            json!({"v": 1, "handle": set["set"], "view": "evidence", "record": record,
                   "sets": sets, "projection": ["id", "locator", "occurrences", "membership",
                                                "explanation", "excerpt"],
                   "idempotency_key": key}),
        );
        last_job = started["job"].clone();
        let result = done(started);
        smallest = smallest.min(result.to_string().len());
    }
    assert!(
        smallest > 8 * 1024,
        "a result of {smallest} bytes is not large"
    );
    // A replay names the kept result.
    let replayed = service.call(
        "triager_inspect",
        &json!({"v": 1, "handle": set["set"], "view": "evidence", "record": record,
                "sets": sets, "projection": ["id", "locator", "occurrences", "membership",
                                             "explanation", "excerpt"],
                "idempotency_key": keys[63]}),
    );
    assert!(replayed.json.len() > 8 * 1024 && !replayed.is_error);

    // The host is held elsewhere: one job waits for it, eight are queued behind it,
    // each with a query of 60,000 bytes, and a ninth is refused.
    let (release, held) = std::sync::mpsc::channel::<()>();
    let (ready, is_ready) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let guard =
            ResourceGuard::enter("the memory test holds the host", BusyChoice::Wait).unwrap();
        ready.send(()).unwrap();
        held.recv().unwrap();
        drop(guard);
    });
    is_ready.recv().unwrap();
    let first = call("triager_select", queries[0].clone());
    assert_eq!(first["status"], "running", "{first}");
    loop {
        let status = call(
            "triager_job",
            json!({"v": 1, "job": first["job"], "action": "status"}),
        );
        if status["phase"] == "waiting for the host" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    for query in &queries[1..9] {
        let queued = call("triager_select", query.clone());
        assert_eq!(queued["status"], "running", "{queued}");
    }
    let refused = call("triager_select", queries[9].clone());
    assert_eq!(refused["error"]["code"], "too_many_jobs", "{refused}");
    drop(refused);
    drop(first);

    // With the table full: every dense request, the crowded line, the largest page
    // and the largest job status, through the MCP adapter and as recipe lines.
    let status = json!({"jsonrpc": "2.0", "id": 101, "method": "tools/call", "params": {
        "name": "triager_job", "arguments": {"v": 1, "job": last_job, "action": "status"}}});
    messages.extend_from_slice(format!("{status}\n").as_bytes());
    let table = CURRENT.load(Ordering::Relaxed).saturating_sub(before) as u64;
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
    mcp::serve(&service, &messages[..], &mut output).unwrap();
    let mut recipe_errors = 0;
    for recipe in &recipes {
        if mcp::run_requests(&service, &recipe[..], &mut recipe_output).is_err() {
            recipe_errors += 1;
        }
    }
    let crowded_recipe = mcp::run_requests(&service, crowded.as_bytes(), &mut recipe_output);
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before) as u64;
    let retained = CURRENT.load(Ordering::Relaxed).saturating_sub(before) as u64;
    eprintln!(
        "job table full (55 ended, 1 waiting, 8 queued): {:.2} MiB kept; with the densest requests, the largest page and status the heap peaked {:.2} MiB above the start, of the {:.2} MiB kept for the adapter; {:.2} MiB stay",
        table as f64 / MIB as f64,
        peak as f64 / MIB as f64,
        ADAPTER_FIXED_BYTES as f64 / MIB as f64,
        retained as f64 / MIB as f64,
    );
    assert!(
        peak <= ADAPTER_FIXED_BYTES,
        "the adapter's heap peaked {peak} bytes above the start and {ADAPTER_FIXED_BYTES} are kept for it"
    );
    assert!(retained <= ADAPTER_FIXED_BYTES);
    // Nothing of this was an artifact: the budget holds what it held.
    assert_eq!(service.held_bytes(), held_before);

    // Every dense request was parsed and answered by the tool; the crowded line was
    // refused unparsed; the page and the status came whole.
    let replies: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(replies.len(), fillers.len() + 5);
    for (reply, (shape, _)) in replies.iter().zip(&fillers) {
        assert_eq!(
            reply["result"]["structuredContent"]["error"]["code"], "invalid_arguments",
            "{shape}: {reply}"
        );
    }
    let crowded_reply = &replies[fillers.len()];
    assert_eq!(crowded_reply["error"]["code"], -32600);
    assert!(
        crowded_reply["error"]["message"]
            .as_str()
            .unwrap()
            .contains(&MAX_REQUEST_VALUES.to_string())
    );
    assert!(
        !replies[fillers.len() + 1]["result"]["structuredContent"]["rows"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        replies[fillers.len() + 2]["result"]["structuredContent"]["regions"]
            .as_array()
            .unwrap()
            .len(),
        8
    );
    assert!(
        replies[fillers.len() + 3]["result"]["structuredContent"]["explanation"][0]["rank"]
            .is_u64()
    );
    assert_eq!(
        replies[fillers.len() + 4]["result"]["structuredContent"]["status"],
        "completed"
    );
    assert_eq!(
        recipe_errors,
        recipes.len(),
        "`x` is refused on every recipe line"
    );
    assert!(
        crowded_recipe
            .unwrap_err()
            .to_string()
            .contains(&MAX_REQUEST_VALUES.to_string())
    );

    release.send(()).unwrap();
    holder.join().unwrap();
    drop(service);
}
