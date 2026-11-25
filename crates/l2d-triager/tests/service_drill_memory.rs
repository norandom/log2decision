//! Triager task 2.1, requirement 6.2 measured: pattern refinement, neighbour search,
//! the Venn recommendation and the issued-batch protocol stay within the memory limit
//! and within what the budget admitted, and what they leave behind is held.
//!
//! As `service_memory.rs`, this test binary counts every heap allocation of the
//! process. One run goes through every operation of the task, keeps every reply
//! alive, and then compares: the heap may not have grown by more than the limit
//! leaves, nor by more than the peak the budget counted as admitted; what stays
//! allocated while the service lives is at most what it reports as held. The run
//! keeps six refinement revisions, three issued batches and three submissions with
//! reasons of the greatest length, each kind far larger than the tolerance of the
//! last comparison, so that one of them that is kept and not held fails it.

mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use l2d_triager::service::{ADAPTER_FIXED_BYTES, DisclosureProfile};
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

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const ORIGINALS: usize = 12_000;
/// What the job table of one run may keep of the adapter's share beside the results,
/// which the artifacts hold a second time for it. One submission of the run keeps
/// more than this, and so do its six refinements and its three batches together.
const JOB_TABLE_BYTES: u64 = 16 * 1024;
/// What the job table keeps of one job beside its result: its record, its handle and
/// its counters. 475 bytes were measured; the table is part of the adapter's share.
const JOB_RECORD_BYTES: u64 = 512;
/// The judgments of one submission, each with a reason of the greatest length.
const JUDGED: usize = 100;
/// The batches of the run that issues record handles only: 40 handles each, 320 bytes
/// of positions and the batch's own strings.
const ID_BATCHES: usize = 36;
const _: () = assert!((ID_BATCHES * 40 * 8) as u64 * 3 / 2 > JOB_TABLE_BYTES);
const _: () = assert!((JUDGED * 200) as u64 > JOB_TABLE_BYTES);

fn entries() -> Vec<(String, Vec<String>)> {
    (0..2)
        .map(|entry| {
            let lines = (0..ORIGINALS / 2)
                .map(|line| {
                    format!(
                        "host-{entry} {} user=u{:03} attempt={} from 10.{entry}.{}.{}",
                        ["login failed", "login ok", "fetch page", "session opened"][line % 4],
                        line % 400,
                        line % 9,
                        line % 250,
                        line % 199,
                    )
                })
                .collect();
            (format!("logs/host-{entry}.log"), lines)
        })
        .collect()
}

/// A valid saved policy report over `private.zip` in `dir` that decides `decided`.
fn write_policy(dir: &std::path::Path, content_sha256: &Value, decided: &[&Orig]) {
    let path = dir.join("private.zip");
    let path = path.to_string_lossy();
    let mut rows: Vec<(String, &Orig)> = decided
        .iter()
        .map(|original| (l2d_store::log_id(&path, &original.text), *original))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    let decisions: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(index, (id, original))| {
            json!({
                "input": {"id": id, "text": original.text, "group": null},
                "score": {"kind": "mark-set", "revision": "rev-1",
                          "comparison_class_assumed": true,
                          "diagnostic_ratio": 0.75, "actionable_ratio": null},
                "vector": null, "queue": "Review", "reason": "Unsupported",
                "explanation": "incompatible exploration profile; actionable ratio withheld",
                "allocation": match index { 0 => json!("UnknownConflict"),
                                            1 | 2 => json!("General"), _ => Value::Null },
            })
        })
        .collect();
    let zip = l2d_core::ZipInputSettings::default();
    let report = json!({
        "format": 1,
        "query_source": {
            "path": path,
            "content_sha256": content_sha256,
            "archive_entries": zip.entries,
            "interpretation": l2d_core::marks::InputInterpretation::archive(&zip),
        },
        "policy": {
            "version": "test-v1", "selection_provenance": "synthetic test", "low": 0.2, "high": 0.8,
            "corroboration": 1,
            "reach": {"radius": 0.2, "top_k": 2, "minimum_labelled_support": 1,
                      "expected_profile": "expected"},
            "require_calibration": false, "review_budget": 4, "unknown_slots": 1,
            "audit_slots": 1, "audit_seed": 5
        },
        "eligibility": l2d_decide::policy::ELIGIBILITY,
        "allocation_rule": l2d_decide::policy::ALLOCATION,
        "scorer_revision": "rev-1", "evidence_revision": "history-v1",
        "actual_profile": "incompatible",
        "calibration": "Missing",
        "qualification": "Exploratory; qualification incomplete. Policy mechanics and local support do not establish deployment quality.",
        "decisions": decisions,
        "unknown": {"configured": 1, "selected": 1, "unused": 0},
        "general": {"configured": 2, "selected": 2, "unused": 0},
        "audit": {"configured": 1, "selected": 0, "unused": 1},
        "deferred": decided.len() - 3
    });
    std::fs::write(
        dir.join("policy-v1.json"),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();
}

fn refused(value: &Value) -> bool {
    value["status"] == "error" && value["error"]["code"] == "refused"
}

/// One run of every operation of the task under `usable` bytes; returns the steps
/// with what they gave, `refused` where the budget refused one.
fn run(dir: &std::path::Path, usable: u64, measured: bool) -> Vec<(String, String)> {
    let options = HostOptions {
        usable,
        policies: vec!["v1".into()],
        profile: DisclosureProfile::PrivateLocal,
        // A catalogue reserves memory for every pattern it may make before it reads:
        // with the limits of `l2d discover search` 169 MiB, with these a fraction,
        // so that the steps after it run under tight limits too.
        patterns: l2d_triager::service::PatternLimits {
            max_patterns: 8,
            max_text_bytes: 256,
        },
        ..HostOptions::default()
    };
    let _ = std::fs::remove_dir_all(dir.join("work"));
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let service = start(dir, "drill-memory", &options).unwrap();
    let usage = service.usage();
    let mut outcome: Vec<(String, String)> = Vec::new();
    let mut kept: Vec<String> = Vec::new();
    // What the service holds after every step: what a step keeps, it holds.
    let mut held_after: Vec<u64> = vec![service.held_bytes()];
    let mut call = |step: &str, tool: &str, args: Value, field: &str| -> Option<Value> {
        let reply = service.call(tool, &args);
        let value: Value = serde_json::from_str(&reply.json).unwrap();
        kept.push(reply.json);
        held_after.push(service.held_bytes());
        if refused(&value) {
            let reason: String = value["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect();
            eprintln!("  {step} refused under {} MiB: {reason}", usable / MIB);
            outcome.push((step.into(), "refused".into()));
            return None;
        }
        assert_eq!(value["status"], "ready", "{step}: {value}");
        outcome.push((step.into(), value[field].to_string()));
        Some(value)
    };
    // Every group of steps goes as far as the budget lets it: a refused step ends
    // its group, not the run, so that under a tight limit every operation that still
    // fits is measured.
    if let Some(opened) = call(
        "open",
        "triager_open",
        json!({"v": 1, "source": {"ref": "private"}}),
        "snapshot",
    ) {
        let snapshot = opened["snapshot"].clone();
        let filter = |query: &str| {
            json!({"v": 1, "scope": snapshot,
                   "producer": {"kind": "filter", "syntax": "regex", "query": query}})
        };
        let b = call(
            "b",
            "triager_select",
            filter(r"login failed user=u0[0-4]"),
            "revision",
        );
        let wide = call(
            "wide",
            "triager_select",
            filter(r"login \w+ user=u0[0-9]"),
            "revision",
        );
        call(
            "venn",
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {
                "kind": "recommend", "method": "venn", "policy": "v1",
                "membership": "investigate_or_review"}}),
            "revision",
        );
        if let Some(wide) = &wide {
            let issued = |producer: Value| {
                let mut producer = producer;
                producer["kind"] = json!("recommend");
                producer["method"] = json!("issued_batch");
                json!({"v": 1, "scope": wide["set"], "producer": producer})
            };
            // Three batches with excerpts, each submitted in two halves, every
            // judgment with a reason of the greatest length; then the collection, and
            // a second run that issues record handles only.
            (|| -> Option<()> {
                for index in 0..3 {
                    let batch = call(
                        &format!("issue-{index}"),
                        "triager_select",
                        issued(json!({"action": "issue", "limit": JUDGED,
                                      "projection": ["excerpt"]})),
                        "batch",
                    )?;
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
                    for (half, items) in items.chunks(items.len().div_ceil(2)).enumerate() {
                        call(
                            &format!("submit-{index}-{half}"),
                            "triager_select",
                            issued(json!({"action": "submit", "batch": batch["batch"],
                                          "expected_revision": half + 1,
                                          "model": "dev/model@1", "prompt_revision": "p1",
                                          "items": items})),
                            "revision",
                        )?;
                    }
                }
                call(
                    "collect",
                    "triager_select",
                    issued(json!({"action": "collect"})),
                    "revision",
                )?;
                for index in 0..ID_BATCHES {
                    call(
                        &format!("ids-{index}"),
                        "triager_select",
                        issued(json!({"action": "issue", "run": "ids", "limit": 40})),
                        "batch",
                    )?;
                }
                Some(())
            })();
        }
        let refine = |target: &str, handle: &Value, action: Value| json!({"v": 1, target: handle, "action": action});
        let p = b.as_ref().and_then(|b| {
            call(
                "patterns",
                "triager_refine",
                refine("set", &b["set"], json!({"kind": "patterns"})),
                "revision",
            )
        });
        let pw = wide.as_ref().and_then(|wide| {
            call(
                "patterns-wide",
                "triager_refine",
                refine("set", &wide["set"], json!({"kind": "patterns"})),
                "revision",
            )
        });
        if let (Some(b), Some(p)) = (&b, &p) {
            let first = p["patterns"][0]["pattern"].clone();
            call(
                "list",
                "triager_refine",
                refine(
                    "refinement",
                    &p["refinement"],
                    json!({"kind": "list", "limit": 100, "projection": ["template"]}),
                ),
                "revision",
            );
            // Four more revisions, each split from the one before, then a merge of
            // the last two halves; a refused one ends the chain.
            let mut last = p.clone();
            (|| -> Option<()> {
                for (index, regex) in ["attempt=0", "attempt=1", "attempt=2", "attempt=3"]
                    .iter()
                    .enumerate()
                {
                    last = call(
                        &format!("split-{index}"),
                        "triager_refine",
                        refine(
                            "refinement",
                            &last["refinement"],
                            json!({"kind": "split", "pattern": first, "regex": regex}),
                        ),
                        "revision",
                    )?;
                }
                call(
                    "merge",
                    "triager_refine",
                    refine(
                        "refinement",
                        &last["refinement"],
                        json!({"kind": "merge",
                                  "patterns": [first, last["changed"]["new_pattern"]]}),
                    ),
                    "revision",
                )?;
                Some(())
            })();
            call(
                "preview",
                "triager_refine",
                refine(
                    "refinement",
                    &p["refinement"],
                    json!({"kind": "preview", "pattern": first, "regex": "attempt=[0-4]"}),
                ),
                "preview",
            );
            call(
                "pattern-set",
                "triager_select",
                json!({"v": 1, "scope": snapshot, "producer": {
                    "kind": "filter", "syntax": "pattern", "refinement": p["refinement"],
                    "pattern": first}}),
                "revision",
            );
            let members = call(
                "records",
                "triager_inspect",
                json!({"v": 1, "handle": b["set"], "view": "records", "limit": 1}),
                "rows",
            );
            if let (Some(members), Some(pw)) = (members, &pw) {
                call(
                    "neighbours",
                    "triager_refine",
                    refine(
                        "refinement",
                        &p["refinement"],
                        json!({"kind": "neighbours", "exemplar": members["rows"][0]["id"],
                                  "method": "character_trigram", "within": "all",
                                  "top_k": 200, "radius": 0.8, "target": pw["refinement"]}),
                    ),
                    "revision",
                );
            }
        }
    }
    // Every step that keeps something holds at least what it visibly keeps: a batch
    // the positions it issued, a submission the reasons it took, a refinement and a
    // set their result twice (once for the job table). A step that charged nothing
    // fails here, whatever the others over-estimate.
    for (index, ((step, result), reply)) in outcome.iter().zip(&kept).enumerate() {
        if result == "refused" {
            continue;
        }
        let grown = held_after[index + 1].saturating_sub(held_after[index]);
        let value: Value = serde_json::from_str(reply).unwrap();
        let least = if step.starts_with("issue-") || step.starts_with("ids-") {
            8 * value["records"].as_array().unwrap().len() as u64 + 64
        } else if step.starts_with("submit-") {
            200 * value["accepted"]["items"].as_u64().unwrap() + 2 * reply.len() as u64
        } else if ["list", "preview", "records", "open", "collect"].contains(&step.as_str()) {
            0
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
    // The result of a job that made no artifact is kept once, in the job table, which
    // is part of the adapter's share: a batch, a page of patterns, a preview.
    let unowned: u64 = outcome
        .iter()
        .zip(&kept)
        .filter(|((step, _), _)| {
            step.starts_with("issue-")
                || step.starts_with("ids-")
                || step == "list"
                || step == "preview"
        })
        .map(|(_, reply)| reply.len() as u64)
        .sum();
    eprintln!(
        "drill usable {:>4} MiB: heap {:>6.2} MiB = {:.2} x admitted ({:>6.2} MiB); retained {} of {} bytes held, {} beyond the artifacts; {}",
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
        retained <= artifacts + unowned + JOB_RECORD_BYTES * outcome.len() as u64,
        "the live service keeps {retained} bytes; its artifacts hold {artifacts} and the job table {unowned} of results"
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
fn drill_down_stays_within_the_limit_and_holds_what_it_keeps() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let entries = entries();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    let universe = oracle(&entries, ".log");
    // The policy needs the hash of the archive: one open outside any measurement.
    let content_sha256 = {
        let service = start(dir.path(), "drill-memory", &HostOptions::default()).unwrap();
        let reply = service.call(
            "triager_open",
            &json!({"v": 1, "source": {"ref": "private"}}),
        );
        let opened: Value = serde_json::from_str(&reply.json).unwrap();
        opened["source"]["content_sha256"].clone()
    };
    let decided: Vec<&Orig> = universe.iter().step_by(6).collect();
    write_policy(dir.path(), &content_sha256, &decided);
    // What the process allocates once and keeps belongs to its own share.
    let warm = run(dir.path(), 512 * MIB, false);
    assert!(
        warm.iter().all(|(_, result)| result != "refused"),
        "{warm:?}"
    );
    assert_eq!(warm.len(), 26 + ID_BATCHES, "{warm:?}");

    let complete = run(dir.path(), 512 * MIB, true);
    assert_eq!(
        complete, warm,
        "the same run gives the same handles and revisions"
    );
    // Under tighter limits an operation completes with the same result or is refused
    // with the figures; in both cases the heap stayed within what was admitted.
    let mut refusals = 0;
    let mut seen = std::collections::BTreeMap::new();
    for usable in [
        160 * MIB,
        96 * MIB,
        64 * MIB,
        48 * MIB,
        32 * MIB,
        24 * MIB,
        16 * MIB,
    ] {
        let tight = run(dir.path(), usable, true);
        for (step, result) in &tight {
            if result == "refused" {
                refusals += 1;
            } else {
                let full = complete.iter().find(|(known, _)| known == step).unwrap();
                assert_eq!(result, &full.1, "{step} under {} MiB", usable / MIB);
            }
        }
        seen.insert(usable / MIB, tight);
    }
    assert!(refusals > 0, "no limit of the ladder refused an operation");
    // Every operation of the task is measured at a limit near the least it runs
    // under: it completes at the limit named here and is refused at the next lower
    // one of the ladder (the issued-batch protocol at every limit that still reads
    // the archive for a selection). The figures are what this fixture measures.
    let state = |limit: u64, step: &str| -> &str {
        match seen[&limit].iter().find(|(known, _)| known == step) {
            Some((_, result)) if result == "refused" => "refused",
            Some(_) => "ok",
            None => "not run",
        }
    };
    assert_eq!(
        (state(160, "neighbours"), state(96, "neighbours")),
        ("ok", "refused")
    );
    for step in [
        "patterns",
        "list",
        "split-3",
        "merge",
        "preview",
        "pattern-set",
    ] {
        assert_eq!(state(96, step), "ok", "{step}");
    }
    assert_eq!(state(64, "patterns"), "refused");
    assert_eq!((state(48, "venn"), state(32, "venn")), ("ok", "refused"));
    for step in ["issue-2", "submit-2-1", "collect", "ids-35"] {
        assert_eq!(
            (state(24, step), state(16, step)),
            ("ok", "not run"),
            "{step}"
        );
    }
}
