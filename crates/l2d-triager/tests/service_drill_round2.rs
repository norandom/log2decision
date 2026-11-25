//! Triager task 2.1, the findings of review round 1 as tests: a batch cut by bytes is
//! issued whole or not at all, a collected set reads the same whatever its run does
//! later, nothing of a host file or a host path reaches a response, the dependency of
//! a Venn set does not depend on what was selected before it, and the bounds of the
//! protocol the first round left untested.

mod support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use l2d_core::{L2dError, MemoryBudget, ZipInputSettings};
use l2d_decide::exploration::ExplorationEncoder;
use l2d_triager::service::{
    DisclosureProfile, EncoderSource, Encoders, HostConfig, Service, TablePlacement,
};
use serde_json::{Value, json};
use support::{
    Host, HostOptions, MIB, Orig, TestScorers, auth_entries, count, oracle, write_archive,
};

/// One entry of sign-in failures whose texts are 60 to 200 bytes long.
fn long_entries() -> Vec<(String, Vec<String>)> {
    let lines = (0..700)
        .map(|index| {
            format!(
                "login failed user=u{index:03} detail={}",
                "x".repeat(30 + (index * 37) % 140)
            )
        })
        .collect();
    vec![("app/long.log".into(), lines)]
}

fn issued_args(scope: &str, mut producer: Value) -> Value {
    producer["kind"] = json!("recommend");
    producer["method"] = json!("issued_batch");
    json!({"v": 1, "scope": scope, "producer": producer})
}

fn issued(host: &Host, scope: &str, producer: Value) -> Value {
    host.ok("triager_select", issued_args(scope, producer))
}

fn batch_ids(batch: &Value) -> Vec<String> {
    batch["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| record["id"].as_str().unwrap().to_string())
        .collect()
}

fn ids_of(host: &Host, handle: &str) -> Vec<String> {
    host.all_records(handle, json!({}), 100)
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect()
}

fn disclosure(host: &Host, snapshot: &str) -> Value {
    host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "overview"}),
    )["disclosure"]
        .clone()
}

/// A host whose configuration `edit` changes before the service starts.
fn host_with(
    entries: &[(String, Vec<String>)],
    options: HostOptions,
    edit: impl FnOnce(&mut HostConfig, &Path),
) -> Host {
    let dir = tempfile::tempdir().unwrap();
    write_archive(&dir.path().join("private.zip"), entries);
    write_archive(&dir.path().join("public.zip"), entries);
    let mut config = support::config(dir.path(), "project-one", &options);
    edit(&mut config, dir.path());
    let service = Service::start(
        config,
        support::budget(&options),
        Box::new(TestScorers {
            delay: options.delay,
        }),
    )
    .unwrap();
    Host { dir, service }
}

/// A saved policy report as `l2d triage` writes one with an incompatible profile:
/// every decision is routed to review as unsupported. `source_path` is the path the
/// report says its IDs were derived from.
fn policy_report(
    source_path: &str,
    content_sha256: &Value,
    entries: &[String],
    decided: &[&Orig],
) -> Value {
    let mut rows: Vec<(String, &Orig)> = decided
        .iter()
        .map(|original| (l2d_store::log_id(source_path, &original.text), *original))
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
    let zip = ZipInputSettings {
        entries: entries.to_vec(),
        ..ZipInputSettings::default()
    };
    json!({
        "format": 1,
        "query_source": {
            "path": source_path,
            "content_sha256": content_sha256,
            "archive_entries": entries,
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
    })
}

fn venn_args(scope: &str, policy: &str) -> Value {
    json!({"v": 1, "scope": scope, "producer": {
        "kind": "recommend", "method": "venn", "policy": policy,
        "membership": "investigate_or_review"}})
}

/// Finding 1: a batch that the response limit cuts is returned, whatever the offset it
/// starts at and however long the name of its run is; the run goes on exactly after
/// the last record the client was given, and the allowance and the counters count
/// what was returned and nothing else.
#[test]
fn a_batch_cut_by_bytes_is_issued_at_every_offset_and_run_name_length() {
    const ALLOWANCE: u64 = 100 * MIB;
    let entries = long_entries();
    let options = HostOptions {
        profile: DisclosureProfile::PublicDevelopment,
        evidence_allowance_bytes: ALLOWANCE,
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let snapshot = host.open("public-fixture")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "literal", "login failed");
    let b_set = b["set"].as_str().unwrap();
    let ids = ids_of(&host, b_set);
    assert_eq!(ids.len(), 700);
    let (mut disclosed, mut rows, mut cut) = (0u64, 0u64, 0);
    for offset in 1..=40usize {
        let run = format!("{}{offset:02}", "n".repeat((offset * 13) % 62));
        let mut next = 0;
        for limit in [offset, 100, 3] {
            let batch = issued(
                &host,
                b_set,
                json!({"action": "issue", "run": run, "limit": limit, "projection": ["excerpt"]}),
            );
            let got = batch_ids(&batch);
            assert!(!got.is_empty() && got.len() <= limit, "{run} limit {limit}");
            assert_eq!(got, ids[next..next + got.len()], "{run} limit {limit}");
            if limit == 100 {
                assert_eq!(batch["clipping"]["rows_by_bytes"], true, "{run}");
                assert!(got.len() < 100);
                cut += 1;
            }
            next += got.len();
            assert_eq!(batch["issued"]["run_total"], next as u64, "{run}");
            let bytes: u64 = batch["records"]
                .as_array()
                .unwrap()
                .iter()
                .map(|record| record["excerpt"].as_str().unwrap().len() as u64)
                .sum();
            assert_eq!(batch["disclosure"]["excerpt_bytes"], bytes);
            disclosed += bytes;
            rows += got.len() as u64;
            assert_eq!(
                batch["disclosure"]["remaining_evidence_bytes"],
                ALLOWANCE - disclosed,
                "{run} limit {limit}"
            );
        }
    }
    assert_eq!(cut, 40);
    let activity = host.activity(&snapshot);
    assert_eq!(activity["genai_rows_issued"], rows);
    assert_eq!(activity["evidence_bytes_disclosed"], disclosed);
    assert_eq!(
        disclosure(&host, &snapshot)["remaining_evidence_bytes"],
        ALLOWANCE - disclosed
    );
}

/// Finding 1, the other half: an `issue` that fails changes nothing. The source is
/// replaced under the service, so the read for the excerpts fails; the allowance,
/// the counter, the run and what is collected from it are as before, and once the
/// source is back the run goes on after its first batch.
#[test]
fn a_failed_issue_leaves_the_allowance_the_counters_and_the_run_unchanged() {
    let entries = long_entries();
    let options = HostOptions {
        profile: DisclosureProfile::PublicDevelopment,
        evidence_allowance_bytes: MIB,
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let archive = host.dir.path().join("public.zip");
    let snapshot = host.open("public-fixture")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "literal", "login failed");
    let b_set = b["set"].as_str().unwrap();
    let ids = ids_of(&host, b_set);
    let issue = json!({"action": "issue", "limit": 5, "projection": ["excerpt"]});
    let first = issued(&host, b_set, issue.clone());
    assert_eq!(batch_ids(&first), ids[..5]);
    let before = (disclosure(&host, &snapshot), host.activity(&snapshot));

    let original = std::fs::read(&archive).unwrap();
    write_archive(&archive, &auth_entries());
    for limit in [5, 100] {
        let mut failing = issue.clone();
        failing["limit"] = json!(limit);
        let (code, message) = host.err("triager_select", issued_args(b_set, failing));
        assert_eq!(code, "refused", "{message}");
    }
    let after = (disclosure(&host, &snapshot), host.activity(&snapshot));
    assert_eq!(after.0, before.0);
    for counter in ["genai_rows_issued", "evidence_bytes_disclosed"] {
        assert_eq!(after.1[counter], before.1[counter], "{counter}");
    }
    std::fs::write(&archive, original).unwrap();
    let second = issued(&host, b_set, issue);
    assert_eq!(batch_ids(&second), ids[5..10]);
    assert_eq!(second["issued"]["batch_index"], 1);
    let collected = issued(
        &host,
        b_set,
        json!({"action": "submit", "batch": second["batch"], "expected_revision": 1,
               "model": "dev/model@1", "prompt_revision": "p1",
               "items": [{"id": ids[5], "outcome": "pass"}]}),
    );
    assert_eq!(collected["run_coverage"]["issued"], 10);
    assert_eq!(collected["producer"]["batches"], 2);
}

/// Finding 2: what a collected set says of a record is bound to the batches it was
/// collected over. A later batch of the run changes neither the evidence under the
/// earlier handle nor the set a retry of the earlier submission returns.
#[test]
fn a_collected_set_reads_the_same_after_its_run_issued_another_batch() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let snapshot = host.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "literal", "user=");
    let b_set = b["set"].as_str().unwrap();
    let ids = ids_of(&host, b_set);
    let batch = issued(&host, b_set, json!({"action": "issue", "limit": 4}));
    let submit = json!({"action": "submit", "batch": batch["batch"], "expected_revision": 1,
                        "model": "dev/model@1", "prompt_revision": "p1",
                        "items": [{"id": ids[0], "outcome": "recommend", "reason": "first"}]});
    let c1 = issued(&host, b_set, submit.clone());
    let c1_set = c1["set"].as_str().unwrap();
    let evidence = |record: &str| {
        let mut read = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": b_set, "view": "evidence", "record": record,
                   "sets": {"c": c1_set}}),
        );
        // What the server did so far is no part of what the set says.
        read.as_object_mut().unwrap().remove("cost");
        read
    };
    let before: Vec<Value> = [0, 2, 6].iter().map(|at| evidence(&ids[*at])).collect();
    assert_eq!(
        before[1]["explanation"][1]["recommendation"]["outcome"],
        "omitted"
    );
    assert_eq!(
        before[2]["explanation"][1]["recommendation"]["outcome"],
        "not_issued"
    );
    let later = issued(&host, b_set, json!({"action": "issue", "limit": 4}));
    assert_eq!(batch_ids(&later), ids[4..8]);
    let after: Vec<Value> = [0, 2, 6].iter().map(|at| evidence(&ids[*at])).collect();
    assert_eq!(after, before, "the earlier set changed under its handle");
    // The same payload again, now that the run has more batches: the set it made.
    let retry = issued(&host, b_set, submit);
    assert_eq!(retry["set"], c1_set);
    assert_eq!(retry["revision"], c1["revision"]);
    assert_eq!(host.activity(&snapshot)["genai_rows_submitted"], 1);
    // What is collected now covers both batches and is another set.
    let now = issued(&host, b_set, json!({"action": "collect"}));
    assert_ne!(now["set"], c1_set);
    assert_eq!(now["run_coverage"]["issued"], 8);
    assert_eq!(c1["run_coverage"]["issued"], 4);
}

const MARK: &str = "PRIVATEDIR-MARK";
const SECRET: &str = "PRIVATE-HOST-SECRET-STRING";

struct BrokenEncoders;

impl EncoderSource for BrokenEncoders {
    fn session_bytes(&self) -> u64 {
        64 * 1024
    }

    fn profile(&self) -> &str {
        "test-broken-v1"
    }

    fn open(&self, _budget: &mut MemoryBudget) -> Result<Box<dyn ExplorationEncoder>, L2dError> {
        Err(L2dError::Missing {
            item: "pinned encoder artifact byte.int8.onnx".into(),
            expected_at: PathBuf::from(format!(r"D:\{MARK}\profile\byte.int8.onnx")),
        })
    }
}

/// Finding 3: under the public-development profile no text of a registered policy
/// file, no path of it, of the work directory or of the encoder's profile reaches a
/// response, whatever goes wrong with them.
#[test]
fn no_text_or_path_of_a_host_file_reaches_a_development_client() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = HostOptions {
        profile: DisclosureProfile::PublicDevelopment,
        policies: vec!["v1".into(), "v2".into(), "v3".into(), "v4".into()],
        encoder: Some(Encoders(Arc::new(BrokenEncoders))),
        ..HostOptions::default()
    };
    let host = host_with(&entries, options, |config, dir| {
        let private = dir.join(MARK);
        std::fs::create_dir_all(&private).unwrap();
        for policy in &mut config.policies {
            policy.path = private.join(format!("{}.json", policy.name));
        }
    });
    let private = host.dir.path().join(MARK);
    let opened = host.open("public-fixture");
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let sha = &opened["source"]["content_sha256"];
    let entry_filter = vec!["*.log".to_string()];
    let decided: Vec<&Orig> = universe.iter().step_by(9).collect();
    let here = host.dir.path().join("public.zip");
    let valid = policy_report(&here.to_string_lossy(), sha, &entry_filter, &decided);
    // v1: no report at all, with a string where the policy is expected. v2: a report
    // that does not validate. v3: a report whose IDs come from another path. v4: no
    // file.
    std::fs::write(
        private.join("v1.json"),
        json!({"format": 1, "policy": format!("{SECRET} login failed user=u00")}).to_string(),
    )
    .unwrap();
    let mut tampered = valid.clone();
    tampered["deferred"] = json!(decided.len());
    tampered["policy"]["selection_provenance"] = json!(SECRET);
    std::fs::write(private.join("v2.json"), tampered.to_string()).unwrap();
    let elsewhere = format!(r"X:\{MARK}\{SECRET}\other.zip");
    std::fs::write(
        private.join("v3.json"),
        policy_report(&elsewhere, sha, &entry_filter, &decided).to_string(),
    )
    .unwrap();
    let work = host.work().to_string_lossy().into_owned();
    let root = host.dir.path().to_string_lossy().into_owned();
    let clean = |what: &str, message: &str| {
        for hidden in [
            MARK,
            SECRET,
            work.as_str(),
            root.as_str(),
            ".json",
            "public.zip",
        ] {
            assert!(
                !message.contains(hidden),
                "{what} shows `{hidden}`: {message}"
            );
        }
    };
    for (policy, needle) in [
        ("v1", "line 1"),
        ("v2", "validation"),
        ("v3", "another source path"),
        ("v4", "NotFound"),
    ] {
        let (code, message) = host.err("triager_select", venn_args(&snapshot, policy));
        assert_eq!(code, "refused", "{policy}: {message}");
        assert!(message.contains(needle), "{policy}: {message}");
        clean(policy, &message);
    }
    // The encoder's profile path stays on the host as well.
    let b = host.select_filter(&snapshot, "literal", "login failed");
    let p = host.ok(
        "triager_refine",
        json!({"v": 1, "set": b["set"], "action": {"kind": "patterns"}}),
    );
    let exemplar = ids_of(&host, b["set"].as_str().unwrap()).remove(0);
    let (code, message) = host.err(
        "triager_refine",
        json!({"v": 1, "refinement": p["refinement"], "action": {
            "kind": "neighbours", "exemplar": exemplar, "method": "vector",
            "top_k": 3, "radius": 0.5}}),
    );
    assert_eq!(code, "refused", "{message}");
    assert!(message.contains("encoder"), "{message}");
    clean("the encoder", &message);
    // A report that is one: read, and nothing of the host's paths in its result.
    std::fs::write(private.join("v4.json"), valid.to_string()).unwrap();
    let c = host.ok("triager_select", venn_args(&snapshot, "v4"));
    assert_eq!(count(&c, "counts", "in"), decided.len() as u64);
    clean("the selection", &c.to_string());
    clean("the refinement", &p.to_string());
}

/// Finding 4: what a Venn selection states does not depend on what was selected
/// before it. The scorer it depends on is part of its producer; the score sets of
/// that scorer are found when the selection is read.
#[test]
fn c_selected_before_a_states_its_dependency_when_it_is_read() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = HostOptions {
        policies: vec!["v1".into()],
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let decided: Vec<&Orig> = universe.iter().step_by(7).collect();
    let path = host.dir.path().join("private.zip");
    std::fs::write(
        host.dir.path().join("policy-v1.json"),
        policy_report(
            &path.to_string_lossy(),
            &opened["source"]["content_sha256"],
            &["*.log".to_string()],
            &decided,
        )
        .to_string(),
    )
    .unwrap();
    let overview = |handle: &str| {
        host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "overview"}),
        )
    };
    let c = host.ok("triager_select", venn_args(&snapshot, "v1"));
    let c_set = c["set"].as_str().unwrap();
    assert_eq!(c["producer"]["depends_on"][0]["revision"], "rev-1");
    assert!(c.get("depends_on").is_none(), "{c}");
    assert_eq!(overview(c_set)["dependencies"], json!([]));
    let a = host.select_top(&snapshot, 40);
    // Asked again after A exists, C is what it was.
    assert_eq!(host.ok("triager_select", venn_args(&snapshot, "v1")), c);
    let read = overview(c_set);
    assert_eq!(read["artifact"], c);
    assert_eq!(read["dependencies"].as_array().unwrap().len(), 1, "{read}");
    assert_eq!(read["dependencies"][0]["set"], a["set"]);
    assert_eq!(read["dependencies"][0]["model"], "sum");
    assert_eq!(read["dependencies"][0]["scorer_revision"], "rev-1");
    // The other order gives the same C and the same dependency.
    let other = Host::new(
        &entries,
        HostOptions {
            policies: vec!["v1".into()],
            ..HostOptions::default()
        },
    );
    let opened = other.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let path = other.dir.path().join("private.zip");
    std::fs::write(
        other.dir.path().join("policy-v1.json"),
        policy_report(
            &path.to_string_lossy(),
            &opened["source"]["content_sha256"],
            &["*.log".to_string()],
            &decided,
        )
        .to_string(),
    )
    .unwrap();
    let a = other.select_top(&snapshot, 40);
    let c_after = other.ok("triager_select", venn_args(&snapshot, "v1"));
    for field in ["counts", "revision", "policy", "coverage"] {
        assert_eq!(c_after[field], c[field], "{field}");
    }
    assert!(c_after.get("depends_on").is_none());
    let read = other.ok(
        "triager_inspect",
        json!({"v": 1, "handle": c_after["set"], "view": "overview"}),
    );
    assert_eq!(read["dependencies"][0]["set"], a["set"]);
}

/// The bounds the first round left untested: the 65th batch of a run, and a policy
/// report beyond the size that is read.
#[test]
fn the_65th_batch_and_a_report_over_16_mib_are_refused_with_the_figures() {
    let entries = auth_entries();
    let options = HostOptions {
        policies: vec!["big".into()],
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let snapshot = host.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "literal", "user=");
    let b_set = b["set"].as_str().unwrap();
    let ids = ids_of(&host, b_set);
    assert!(ids.len() > 65, "{}", ids.len());
    for index in 0..64 {
        let batch = issued(&host, b_set, json!({"action": "issue", "limit": 1}));
        assert_eq!(batch_ids(&batch), ids[index..=index]);
    }
    let before = host.activity(&snapshot);
    let (code, message) = host.err(
        "triager_select",
        issued_args(b_set, json!({"action": "issue", "limit": 1})),
    );
    assert_eq!(code, "refused");
    assert!(message.contains("64 batches"), "{message}");
    assert_eq!(
        host.activity(&snapshot)["genai_rows_issued"],
        before["genai_rows_issued"]
    );
    // Another run of the same scope is not affected.
    let other = issued(
        &host,
        b_set,
        json!({"action": "issue", "run": "second", "limit": 2}),
    );
    assert_eq!(batch_ids(&other), ids[..2]);

    let report = host.dir.path().join("policy-big.json");
    let file = std::fs::File::create(&report).unwrap();
    file.set_len(16 * MIB + 1).unwrap();
    drop(file);
    let (code, message) = host.err("triager_select", venn_args(&snapshot, "big"));
    assert_eq!(code, "refused");
    assert!(
        message.contains("16777217") && message.contains("16777216"),
        "{message}"
    );
}

/// A refine job and a recommendation job that are cancelled while they wait end as
/// cancelled and leave nothing: the refinement and the batch are made when they are
/// asked for again.
#[test]
fn a_cancelled_refine_or_recommend_job_commits_nothing() {
    let entries = auth_entries();
    // A host whose calls return their job at once and whose scorer takes its time:
    // jobs queue up behind a score.
    let options = HostOptions {
        inline_wait: Some(Duration::ZERO),
        delay: Duration::from_millis(40),
        batch_rows: Some(16),
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let opened = host.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private", "entries": ["*.log"]}}),
    );
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let b = host.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "filter", "syntax": "literal", "query": "login failed"}}),
    );
    let b_set = b["set"].as_str().unwrap().to_string();
    let slow = host.reply(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "score", "model": "sum", "top_k": 5, "ties": "record_id_ascending"}}),
    );
    let slow: Value = serde_json::from_str(&slow.json).unwrap();
    assert_eq!(slow["status"], "running", "{slow}");
    let refine = json!({"v": 1, "set": b_set, "action": {"kind": "patterns"}});
    let issue = issued_args(&b_set, json!({"action": "issue", "limit": 3}));
    let mut jobs = Vec::new();
    for (tool, args) in [("triager_refine", &refine), ("triager_select", &issue)] {
        let queued: Value = serde_json::from_str(&host.reply(tool, args.clone()).json).unwrap();
        assert_eq!(queued["status"], "running", "{queued}");
        let job = queued["job"].as_str().unwrap().to_string();
        assert_eq!(host.job(&job, "cancel")["status"], "cancel_requested");
        jobs.push(job);
    }
    for job in &jobs {
        let ended = host.wait_status(job);
        assert_eq!(ended["status"], "cancelled", "{ended}");
        assert!(ended.get("result").is_none());
    }
    host.wait(slow["job"].as_str().unwrap());
    let activity = host.activity(&snapshot);
    assert_eq!(activity["genai_rows_issued"], 0);
    // Asked again, both are made from the start.
    let p = host.ok("triager_refine", refine);
    assert_eq!(p["parent"], Value::Null);
    assert_eq!(p["cost"]["archive_scans"], 2, "{p}");
    let batch = host.ok("triager_select", issue);
    assert_eq!(batch["issued"]["batch_index"], 0);
}

/// With the tables on disk a pattern, a neighbourhood, a Venn selection and a
/// collected set say they are there and hold the members they hold in memory.
#[test]
fn listed_selections_on_disk_hold_the_members_they_hold_in_memory() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let decided: Vec<&Orig> = universe.iter().step_by(5).collect();
    let run = |tables: TablePlacement| -> Vec<(String, Vec<String>, Value)> {
        let options = HostOptions {
            tables,
            policies: vec!["v1".into()],
            ..HostOptions::default()
        };
        let host = Host::new(&entries, options);
        let opened = host.open("private");
        let snapshot = opened["snapshot"].as_str().unwrap().to_string();
        let path = host.dir.path().join("private.zip");
        std::fs::write(
            host.dir.path().join("policy-v1.json"),
            policy_report(
                &path.to_string_lossy(),
                &opened["source"]["content_sha256"],
                &["*.log".to_string()],
                &decided,
            )
            .to_string(),
        )
        .unwrap();
        let b = host.select_filter(&snapshot, "literal", "login failed");
        let b_set = b["set"].as_str().unwrap();
        let ids = ids_of(&host, b_set);
        let p = host.ok(
            "triager_refine",
            json!({"v": 1, "set": b_set, "action": {"kind": "patterns"}}),
        );
        let pattern = host.ok(
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {
                "kind": "filter", "syntax": "pattern", "refinement": p["refinement"],
                "pattern": p["patterns"][0]["pattern"]}}),
        );
        let near = host.ok(
            "triager_refine",
            json!({"v": 1, "refinement": p["refinement"], "action": {
                "kind": "neighbours", "exemplar": ids[2], "method": "character_trigram",
                "top_k": 7, "radius": 0.7}}),
        );
        let venn = host.ok("triager_select", venn_args(&snapshot, "v1"));
        let batch = issued(&host, b_set, json!({"action": "issue", "limit": 6}));
        let collected = issued(
            &host,
            b_set,
            json!({"action": "submit", "batch": batch["batch"], "expected_revision": 1,
                   "model": "dev/model@1", "prompt_revision": "p1",
                   "items": [{"id": ids[0], "outcome": "recommend"},
                             {"id": ids[1], "outcome": "pass"},
                             {"id": ids[3], "outcome": "abstain"}]}),
        );
        let on_disk = tables == TablePlacement::Disk;
        let mut made = Vec::new();
        for (name, set) in [
            ("pattern", pattern),
            ("neighbours", near),
            ("venn", venn),
            ("collected", collected),
        ] {
            assert_eq!(set["cost"]["tables_on_disk"], on_disk, "{name}");
            let handle = set["set"].as_str().unwrap();
            let read = host.ok(
                "triager_inspect",
                json!({"v": 1, "handle": handle, "view": "overview"}),
            );
            assert_eq!(read["tables"]["on_disk"], on_disk, "{name}");
            assert_eq!(
                read["tables"]["memory_bytes"].as_u64() == Some(0),
                on_disk,
                "{name}"
            );
            // The value kept beside a record reads the same from either placement.
            let first = ids_of(&host, handle).remove(0);
            let mut evidence = host.ok(
                "triager_inspect",
                json!({"v": 1, "handle": handle, "view": "evidence", "record": first}),
            );
            let explained = evidence["explanation"][0].take();
            made.push((
                name.to_string(),
                ids_of(&host, handle),
                json!({"counts": set["counts"], "revision": set["revision"],
                       "state": explained["state"], "distance": explained["distance"],
                       "venn": explained["venn"], "matched": explained["matched"],
                       "outcome": explained["recommendation"]["outcome"]}),
            ));
        }
        made
    };
    let memory = run(TablePlacement::MemoryWhenAdmitted);
    let disk = run(TablePlacement::Disk);
    assert_eq!(memory.len(), 4);
    assert_eq!(disk, memory);
}

/// Finding 3, the work directory: a refusal of a refine job for disk space names the
/// figures and "the work directory", not where the host keeps it.
#[test]
fn a_disk_refusal_of_a_refine_job_does_not_show_the_work_directory() {
    let entries = auth_entries();
    let host = host_with(&entries, HostOptions::default(), |config, _| {
        config.free_disk_bytes = Some(1024 * 1024);
    });
    let snapshot = host.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "literal", "login failed");
    let (code, message) = host.err(
        "triager_refine",
        json!({"v": 1, "set": b["set"], "action": {"kind": "patterns"}}),
    );
    assert_eq!(code, "refused", "{message}");
    assert!(message.contains("1048576"), "{message}");
    assert!(message.contains("<work directory>"), "{message}");
    let root = host.dir.path().to_string_lossy().into_owned();
    assert!(!message.contains(&root), "{message}");
}

/// Review round 2: the overview of a Venn selection lists at most eight of the score
/// sets it depends on, states how many there are, and says so when it lists fewer.
#[test]
fn the_dependencies_of_a_venn_selection_are_counted_and_a_cut_list_says_so() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = HostOptions {
        policies: vec!["v1".into()],
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let decided: Vec<&Orig> = universe.iter().step_by(7).collect();
    let path = host.dir.path().join("private.zip");
    std::fs::write(
        host.dir.path().join("policy-v1.json"),
        policy_report(
            &path.to_string_lossy(),
            &opened["source"]["content_sha256"],
            &["*.log".to_string()],
            &decided,
        )
        .to_string(),
    )
    .unwrap();
    let c = host.ok("triager_select", venn_args(&snapshot, "v1"));
    let c_set = c["set"].as_str().unwrap();
    let overview = || {
        host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": c_set, "view": "overview"}),
        )
    };
    let listed = |read: &Value| -> Vec<String> {
        read["dependencies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["set"].as_str().unwrap().to_string())
            .collect()
    };
    let none = overview();
    assert_eq!(none["dependencies_total"], 0);
    assert_eq!(none["clipping"]["clipped"], false);
    // Score sets of the policy's scorer revision, one after the other; a filter set
    // between them is no dependency.
    let mut handles: Vec<String> = Vec::new();
    host.select_filter(&snapshot, "literal", "login");
    for k in 1..=11u64 {
        handles.push(
            host.select_top(&snapshot, k)["set"]
                .as_str()
                .unwrap()
                .to_string(),
        );
        handles.sort();
        let read = overview();
        assert_eq!(read["dependencies_total"], k, "{read}");
        let expected: Vec<String> = handles.iter().take(8).cloned().collect();
        assert_eq!(listed(&read), expected, "after {k} score sets");
        assert_eq!(read["clipping"]["clipped"], k > 8, "after {k} score sets");
        if k > 8 {
            assert_eq!(read["clipping"]["fields"], json!(["dependencies"]));
        }
        assert!(
            read["dependencies"]
                .as_array()
                .unwrap()
                .iter()
                .all(|entry| entry["model"] == "sum" && entry["scorer_revision"] == "rev-1")
        );
    }
}

/// Review round 2: an `issue` that is refused because the evidence allowance does
/// not cover its first record changes nothing: the allowance, the counters and the
/// run are as before, and what is collected covers the first batch only.
#[test]
fn an_issue_the_allowance_refuses_leaves_everything_as_it_was() {
    let entries = long_entries();
    let options = HostOptions {
        profile: DisclosureProfile::PublicDevelopment,
        // The first three excerpts take 291 bytes; the fourth has 171.
        evidence_allowance_bytes: 300,
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let snapshot = host.open("public-fixture")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "literal", "login failed");
    let b_set = b["set"].as_str().unwrap();
    let ids = ids_of(&host, b_set);
    let issue = |limit: u64| json!({"action": "issue", "limit": limit, "projection": ["excerpt"]});
    let first = issued(&host, b_set, issue(3));
    assert_eq!(batch_ids(&first), ids[..3]);
    assert_eq!(first["disclosure"]["excerpt_bytes"], 291);
    assert_eq!(first["disclosure"]["remaining_evidence_bytes"], 9);
    let before = (disclosure(&host, &snapshot), host.activity(&snapshot));
    assert_eq!(before.1["genai_rows_issued"], 3);
    for limit in [1, 5, 100] {
        let (code, message) = host.err("triager_select", issued_args(b_set, issue(limit)));
        assert_eq!(code, "disclosure_refused", "{message}");
        assert!(message.contains("9 bytes left"), "{message}");
    }
    let after = (disclosure(&host, &snapshot), host.activity(&snapshot));
    assert_eq!(after.0, before.0);
    for counter in ["genai_rows_issued", "evidence_bytes_disclosed"] {
        assert_eq!(after.1[counter], before.1[counter], "{counter}");
    }
    assert_eq!(after.0["remaining_evidence_bytes"], 9);
    // The run holds its first batch and nothing of the refused ones.
    let collected = issued(
        &host,
        b_set,
        json!({"action": "submit", "batch": first["batch"], "expected_revision": 1,
               "model": "dev/model@1", "prompt_revision": "p1",
               "items": [{"id": ids[0], "outcome": "recommend"}]}),
    );
    assert_eq!(collected["producer"]["batches"], 1);
    assert_eq!(collected["run_coverage"]["issued"], 3);
    assert_eq!(collected["run_coverage"]["omitted"], 2);
    assert_eq!(
        collected["run_coverage"]["not_issued"],
        ids.len() as u64 - 3
    );
    let evidence = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": b_set, "view": "evidence", "record": ids[3],
               "sets": {"c": collected["set"]}}),
    );
    assert_eq!(
        evidence["explanation"][1]["recommendation"]["outcome"],
        "not_issued"
    );
}
