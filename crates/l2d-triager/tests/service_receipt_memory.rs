//! Triager task 2.2, requirement 6.2 measured: an allocation, a saved receipt and a
//! receipt reopened in another server stay within the memory limit and within what
//! the budget admitted, and what they keep is held.
//!
//! As `service_drill_memory.rs`, this test binary counts every heap allocation of the
//! process. A first server allocates a budget of 6,000 slots and saves a candidate of
//! about 70,000 originals and the final work set; a second server reopens the
//! candidate, composes a sibling, allocates again and saves. Every reply is kept
//! alive; the heap may not grow by more than the limit leaves nor by more than the
//! peak the budget admitted, and what stays allocated while a server lives is at
//! most what it reports as held. The reopened relation (about 3 MiB), the tables of
//! the reopened sets, the slots of each final set (about 220 KiB) and the model
//! judgments of a whole issued run (6,400 records with reasons of 200 bytes, about
//! 2.5 MiB) are each far larger than the tolerance of the last comparison, so that one
//! of them kept and not held fails it.

mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_triager::service::{ADAPTER_FIXED_BYTES, Service};
use serde_json::{Value, json};
use support::{BASELINE, HostOptions, MIB, Orig, membership_line, oracle, write_archive};

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

const ORIGINALS: usize = 100_000;
/// What the job table keeps of one job beside its result: its record, its handle and
/// its counters; part of the adapter's share.
const JOB_RECORD_BYTES: u64 = 512;
/// What the job table takes when its first job is added (its maps and its queue) and
/// the bounded message a refused job keeps: part of the adapter's share. A run that
/// only refuses has no artifact whose estimate covers it. Every share this test guards
/// (relation, tables, slots, judgments) is more than ten times larger.
const JOB_TABLE_BYTES: u64 = 8 * 1024;

fn entries() -> Vec<(String, Vec<String>)> {
    (0..2)
        .map(|entry| {
            let lines = (0..ORIGINALS / 2)
                .map(|line| {
                    format!(
                        "host-{entry} {} user=u{:03} event={line} from 10.{entry}.{}.{}",
                        ["login failed", "login ok", "fetch page", "session opened"][line % 4],
                        line % 400,
                        line % 250,
                        line % 199,
                    )
                })
                .collect();
            (format!("logs/host-{entry}.log"), lines)
        })
        .collect()
}

fn membership(universe: &[Orig]) -> String {
    universe
        .iter()
        .enumerate()
        .filter_map(|(index, original)| {
            [Some("in"), Some("out"), Some("abstain"), None][index % 4]
                .map(|state| membership_line(&original.id(), state))
        })
        .collect()
}

fn budget() -> Value {
    json!({"total": 6000, "duplicate_rule": "first_pool_wins", "shortfall": "leave_unfilled",
           "pools": [
               {"name": "general", "limit": 4000, "from": {"intersection": ["d", "c"]},
                "order": ["score_desc", "record_id_ascending"]},
               {"name": "unfamiliar", "limit": 1500,
                "from": {"intersection": ["d", {"state": {"set": "c", "is": "not_evaluated"}}]},
                "order": ["record_id_ascending"]},
               {"name": "audit", "limit": 500, "from": {"difference": ["d", "c"]},
                "order": ["seeded_hash", "record_id_ascending"], "seed": 5}]})
}

/// One call of a step: its name, the tool, the arguments and the result field kept.
type Call<'a> = &'a mut dyn FnMut(&str, &str, Value, &str) -> Option<Value>;

fn refused(value: &Value) -> bool {
    value["status"] == "error" && value["error"]["code"] == "refused"
}

/// One server's steps under `usable` bytes: `steps` makes the calls through `call`.
/// Returns what every step gave, `refused` where the budget refused one.
fn measure(
    service: Service,
    usable: u64,
    measured: bool,
    before: usize,
    steps: &mut dyn FnMut(Call<'_>),
    dir: &Path,
) -> Vec<(String, String)> {
    let usage = service.usage();
    let mut outcome: Vec<(String, String)> = Vec::new();
    let mut kept: Vec<String> = Vec::new();
    let mut held_after: Vec<u64> = vec![service.held_bytes()];
    {
        let mut call = |step: &str, tool: &str, args: Value, field: &str| -> Option<Value> {
            let reply = service.call(tool, &args);
            let value: Value = serde_json::from_str(&reply.json).unwrap();
            kept.push(reply.json);
            held_after.push(service.held_bytes());
            if refused(&value) {
                eprintln!(
                    "  {step} refused under {} MiB: {}",
                    usable / MIB,
                    value["error"]["message"].as_str().unwrap_or_default()
                );
                outcome.push((step.into(), "refused".into()));
                return None;
            }
            assert_eq!(value["status"], "ready", "{step}: {value}");
            outcome.push((step.into(), value[field].to_string()));
            Some(value)
        };
        steps(&mut call);
    }
    // Every step that keeps something holds at least what it visibly keeps: its result
    // twice, and a reopened receipt its relation (record ID, identity and count of
    // every retained original).
    for (index, ((step, result), reply)) in outcome.iter().zip(&kept).enumerate() {
        if result == "refused" {
            continue;
        }
        let grown = held_after[index + 1].saturating_sub(held_after[index]);
        let value: Value = serde_json::from_str(reply).unwrap();
        // A batch, a submission and a collection are task 2.1's, measured by
        // `service_drill_memory.rs`.
        let least = if step.starts_with("save")
            || step.starts_with("issue-")
            || step.starts_with("submit-")
        {
            0
        } else if step == "reopen" && value["cost"]["index_on_disk"] == false {
            2 * reply.len() as u64 + 44 * value["counts"]["distinct_originals"].as_u64().unwrap()
        } else {
            2 * reply.len() as u64
        };
        assert!(
            grown >= least,
            "{step}: the service holds {grown} bytes more after it, and it keeps at least {least}"
        );
    }
    let replies: usize = kept.iter().map(String::capacity).sum();
    let retained = CURRENT
        .load(Ordering::Relaxed)
        .saturating_sub(before + replies) as u64;
    let held = service.held_bytes();
    let growth = PEAK.load(Ordering::Relaxed).saturating_sub(before) as u64;
    let admitted = usage.report().counted_peak_bytes.saturating_sub(BASELINE);
    let artifacts = held - ADAPTER_FIXED_BYTES;
    // A save and a batch make no artifact: their result is kept once, in the job table.
    let unowned: u64 = outcome
        .iter()
        .zip(&kept)
        .filter(|((step, _), _)| step.starts_with("save") || step.starts_with("issue-"))
        .map(|(_, reply)| reply.len() as u64)
        .sum();
    eprintln!(
        "receipt usable {:>4} MiB: heap {:>6.2} MiB = {:.2} x admitted ({:>6.2} MiB); retained {} of {} bytes held, {} beyond the artifacts; {}",
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
    if measured {
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
                <= artifacts + unowned + JOB_TABLE_BYTES + JOB_RECORD_BYTES * outcome.len() as u64,
            "the live service keeps {retained} bytes; its artifacts hold {artifacts} and the job table {unowned} of results"
        );
    }
    drop(service);
    drop(kept);
    let _ = dir;
    outcome
}

fn options(dir: &Path, usable: u64) -> HostOptions {
    let universe = oracle(&entries(), ".log");
    let _ = dir;
    HostOptions {
        usable,
        memberships: vec![("c".into(), membership(&universe))],
        ..HostOptions::default()
    }
}

/// The first server: the reduction, an allocation and both saves. Returns the steps
/// and the receipt of the candidate.
fn first(dir: &Path, usable: u64, measured: bool) -> (Vec<(String, String)>, Option<String>) {
    let options = options(dir, usable);
    let _ = std::fs::remove_dir_all(dir.join("work"));
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let service = support::start(dir, "receipt-memory", &options).unwrap();
    let mut receipt = None;
    let outcome = measure(
        service,
        usable,
        measured,
        before,
        &mut |call| {
            let Some(opened) = call(
                "open",
                "triager_open",
                json!({"v": 1, "source": {"ref": "private"}}),
                "snapshot",
            ) else {
                return;
            };
            let snapshot = opened["snapshot"].clone();
            let select = |producer: Value| json!({"v": 1, "scope": snapshot, "producer": producer});
            let a = call(
                "a",
                "triager_select",
                select(json!({"kind": "score", "model": "sum", "top_k": 75_000,
                              "ties": "record_id_ascending"})),
                "revision",
            );
            let b = call(
                "b",
                "triager_select",
                select(json!({"kind": "filter", "syntax": "regex", "query": "user=u0[0-4]"})),
                "revision",
            );
            let c = call(
                "c",
                "triager_select",
                select(json!({"kind": "membership", "ref": "c"})),
                "revision",
            );
            let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                return;
            };
            let Some(d) = call(
                "d",
                "triager_compose",
                json!({"v": 1, "mode": "view", "snapshot": snapshot,
                       "sets": {"a": a["set"], "b": b["set"]}, "expr": {"difference": ["a", "b"]}}),
                "revision",
            ) else {
                return;
            };
            let f = call(
                "f",
                "triager_compose",
                json!({"v": 1, "mode": "allocate", "snapshot": snapshot,
                       "sets": {"d": d["view"], "c": c["set"]},
                       "expr": {"intersection": ["d", "c"]}, "budget": budget()}),
                "revision",
            );
            // A model client judges a whole run over D: 64 batches of 100 records, each
            // judgment with a reason of the greatest length. The receipt keeps them all.
            let issued = |producer: Value| {
                let mut producer = producer;
                producer["kind"] = json!("recommend");
                producer["method"] = json!("issued_batch");
                json!({"v": 1, "scope": d["view"], "producer": producer})
            };
            let mut judged = None;
            for index in 0..64 {
                let Some(batch) = call(
                    &format!("issue-{index}"),
                    "triager_select",
                    issued(json!({"action": "issue", "limit": 100})),
                    "batch",
                ) else {
                    break;
                };
                let items: Vec<Value> = batch["records"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(at, record)| {
                        json!({"id": record["id"],
                               "outcome": (["recommend", "pass", "abstain"][at % 3]),
                               "reason": "r".repeat(200)})
                    })
                    .collect();
                judged = call(
                    &format!("submit-{index}"),
                    "triager_select",
                    issued(json!({"action": "submit", "batch": batch["batch"],
                                  "expected_revision": 1, "model": "dev/model@1",
                                  "prompt_revision": "p1", "items": items})),
                    "revision",
                );
            }
            let mut beside = json!({"c": c["set"]});
            if let Some(judged) = &judged {
                beside["j"] = judged["set"].clone();
            }
            if let Some(saved) = call(
                "save-d",
                "triager_save",
                json!({"v": 1, "handle": d["view"], "sets": beside}),
                "receipt",
            ) {
                receipt = saved["receipt"].as_str().map(str::to_string);
            }
            if let Some(f) = f {
                call(
                    "save-f",
                    "triager_save",
                    json!({"v": 1, "handle": f["set"]}),
                    "receipt",
                );
            }
        },
        dir,
    );
    (outcome, receipt)
}

/// The second server: the candidate reopened, a sibling, an allocation, a save.
fn second(dir: &Path, receipt: &str, usable: u64, measured: bool) -> Vec<(String, String)> {
    let options = options(dir, usable);
    let work = dir.join("work-2");
    let _ = std::fs::remove_dir_all(&work);
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let mut config = support::config(dir, "receipt-memory", &options);
    config.work_root = work;
    let service = Service::start(
        config,
        support::budget(&options),
        Box::new(support::TestScorers {
            delay: std::time::Duration::ZERO,
        }),
    )
    .unwrap();
    measure(
        service,
        usable,
        measured,
        before,
        &mut |call| {
            let Some(opened) = call(
                "reopen",
                "triager_open",
                json!({"v": 1, "receipt": receipt}),
                "revision",
            ) else {
                return;
            };
            let handle = |kind: &str, producer: &str| -> Value {
                opened["sets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|set| set["kind"] == kind && set["producer"] == producer)
                    .map(|set| set["handle"].clone())
                    .unwrap()
            };
            let (d, c) = (
                handle("view", "expression"),
                handle("selection", "membership"),
            );
            let snapshot = opened["snapshot"].clone();
            call(
                "sibling",
                "triager_compose",
                json!({"v": 1, "mode": "view", "snapshot": snapshot, "scope": d,
                       "sets": {"d": d, "c": c},
                       "expr": {"intersection": ["d", {"state": {"set": "c", "is": "not_evaluated"}}]}}),
                "revision",
            );
            if let Some(f) = call(
                "f",
                "triager_compose",
                json!({"v": 1, "mode": "allocate", "snapshot": snapshot,
                       "sets": {"d": d, "c": c},
                       "expr": {"intersection": ["d", "c"]}, "budget": budget()}),
                "selected",
            ) {
                call(
                    "save-f",
                    "triager_save",
                    json!({"v": 1, "handle": f["set"]}),
                    "content_sha256",
                );
            }
        },
        dir,
    )
}

#[test]
fn allocation_receipts_and_reopening_stay_within_the_limit_and_hold_what_they_keep() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let entries = entries();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    // What the process allocates once and keeps belongs to its own share.
    let (warm, receipt) = first(dir.path(), 512 * MIB, false);
    assert!(
        warm.iter().all(|(_, result)| result != "refused"),
        "{warm:?}"
    );
    let receipt = receipt.unwrap();
    let _ = second(dir.path(), &receipt, 512 * MIB, false);

    let (complete, again) = first(dir.path(), 512 * MIB, true);
    assert_eq!(
        complete, warm,
        "the same run gives the same handles and revisions"
    );
    assert_eq!(again.as_deref(), Some(receipt.as_str()));
    let reopened = second(dir.path(), &receipt, 512 * MIB, true);
    assert!(
        reopened.iter().all(|(_, result)| result != "refused"),
        "{reopened:?}"
    );
    // Under tighter limits a step of the second server completes with the same result
    // or is refused with the figures; in both cases the heap stayed within what was
    // admitted.
    let mut refusals = 0;
    for usable in [64 * MIB, 24 * MIB, 12 * MIB, 8 * MIB, 6 * MIB] {
        for (step, result) in second(dir.path(), &receipt, usable, true) {
            if result == "refused" {
                refusals += 1;
            } else {
                let full = reopened.iter().find(|(known, _)| *known == step).unwrap();
                assert_eq!(result, full.1, "{step} under {} MiB", usable / MIB);
            }
        }
    }
    assert!(refusals > 0, "no limit of the ladder refused a step");
}
