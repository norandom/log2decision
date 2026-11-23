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
use l2d_triager::service::ADAPTER_FIXED_BYTES;
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

const ORIGINALS: usize = 60_000;
/// What the job table of one measured run may keep of the adapter's share beside the
/// results, which the artifacts hold a second time for it: the records of at most
/// 23 jobs. The membership tables of a complete run take 19 x 15,000 bytes, so a
/// table that is not held does not hide in this.
const JOB_TABLE_BYTES: u64 = 16 * 1024;
/// The steps of a run that nothing refuses.
const STEPS: usize = 23;

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

fn run(
    dir: &std::path::Path,
    usable: u64,
    batch_rows: Option<u64>,
    dedup_set_texts: Option<u64>,
    first: &Orig,
    measured: bool,
) -> Outcome {
    let options = HostOptions {
        usable,
        batch_rows,
        dedup_set_texts,
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
    let mut call = |step: &str, tool: &str, args: Value, field: &str| -> Option<Value> {
        let reply = service.call(tool, &args);
        let value: Value = serde_json::from_str(&reply.json).unwrap();
        kept.push(reply.json);
        if refused(&value) {
            outcome.push((step.into(), "refused".into()));
            return None;
        }
        assert_eq!(value["status"], "ready", "{step}: {value}");
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
        // More selections and views, so that what the membership tables take is far
        // more than the tolerance of the check below: nineteen tables in all.
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
    eprintln!(
        "service usable {:>3} MiB batch {:>5} set {:>5}: heap {:>6.2} MiB = {:.2} x usable, {:.2} x admitted ({:>6.2} MiB); retained {:.2} of {:.2} MiB held; {:.1?}; {}",
        usable / MIB,
        batch_rows.map_or("auto".into(), |rows| rows.to_string()),
        dedup_set_texts.map_or("auto".into(), |texts| texts.to_string()),
        growth as f64 / MIB as f64,
        growth as f64 / usable as f64,
        growth as f64 / admitted.max(1) as f64,
        admitted as f64 / MIB as f64,
        retained as f64 / MIB as f64,
        held as f64 / MIB as f64,
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
    let first = oracle(&entries, ".log")[0].clone();
    for (batch_rows, dedup_set_texts) in [(None, None), (Some(64), Some(20))] {
        let outcome = run(
            dir.path(),
            128 * MIB,
            batch_rows,
            dedup_set_texts,
            &first,
            false,
        );
        assert!(outcome.iter().all(|(_, result)| result != "refused"));
    }
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
    drop(universe);
    drop(entries);
    warm_up();

    let mut complete: Option<Outcome> = None;
    let mut completed = 0;
    let mut refusals = 0;
    for usable in [10 * MIB, 16 * MIB, 24 * MIB, 40 * MIB, 128 * MIB] {
        for (batch_rows, dedup_set_texts) in [(None, None), (Some(1_024), Some(4_000))] {
            let outcome = run(
                dir.path(),
                usable,
                batch_rows,
                dedup_set_texts,
                &first,
                true,
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
                complete.get_or_insert(outcome);
            } else {
                refusals += 1;
            }
        }
    }
    // The ladder shows both: limits that are refused and limits that complete.
    assert!(completed >= 2, "{completed} runs completed");
    assert!(refusals >= 1, "no limit of the ladder was refused");
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
    assert_eq!(replies.len(), fillers.len() + 3);
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
        replies[fillers.len() + 2]["result"]["structuredContent"]["status"],
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
