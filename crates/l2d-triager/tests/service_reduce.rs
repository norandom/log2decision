//! Triager task 1.3, requirements 2.3, 4.3, 6.1, 6.3 and 6.4 on the small fixture:
//! prepared results are reused, warm drilling reads no source and scores nothing, and
//! every job states its cost in the units of the scale gate. The same checks run on
//! the multi-million fixture through `tools/triager-scale`.

mod support;

use serde_json::{Value, json};
use support::{
    Host, HostOptions, Orig, auth_entries, count, ids, occurrences, oracle, score_of, top_k,
};

fn row_ids(rows: &[Value]) -> Vec<String> {
    rows.iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect()
}

/// The work counters of the server: what a warm call must leave unchanged.
fn work(host: &Host, snapshot: &str) -> Value {
    let activity = host.activity(snapshot);
    json!({
        "archive_scans": activity["archive_scans"],
        "bytes_unpacked": activity["bytes_unpacked"],
        "rows_scored": activity["rows_scored"],
        "scorer_loads": activity["scorer_loads"],
    })
}

#[test]
fn a_second_rule_on_a_prepared_scorer_reads_and_scores_nothing() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let universe = oracle(&entries, ".log");
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();

    let first = host.select_top(snapshot, 20);
    assert_eq!(first["cost"]["archive_scans"], 1);
    assert_eq!(first["cost"]["rows_scored"], universe.len() as u64);
    assert_eq!(first["cost"]["reused"], Value::Null);
    let before = host.activity(snapshot);
    assert_eq!(before["rows_scored"], universe.len() as u64);
    assert!(before["bytes_unpacked"].as_u64().unwrap() > 0, "{before}");

    // Another k and a threshold of the same scorer revision over the same population.
    let second = host.select_top(snapshot, 35);
    let third = host.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "score", "model": "sum", "threshold": 0.5}}),
    );
    for derived in [&second, &third] {
        assert_eq!(derived["cost"]["archive_scans"], 0, "{derived}");
        assert_eq!(derived["cost"]["rows_scored"], 0, "{derived}");
        assert_eq!(derived["cost"]["bytes_unpacked"], 0, "{derived}");
        assert_eq!(derived["cost"]["reused"]["kind"], "scores", "{derived}");
        assert_eq!(derived["cost"]["reused"]["set"], first["set"], "{derived}");
    }
    let after = host.activity(snapshot);
    for counter in ["archive_scans", "bytes_unpacked", "rows_scored"] {
        assert_eq!(after[counter], before[counter], "{counter}");
    }
    // The scorer was loaded to bind its revision, and nothing was scored with it.
    assert_eq!(
        after["scorer_loads"].as_u64().unwrap(),
        before["scorer_loads"].as_u64().unwrap() + 2
    );

    // The derived sets are the oracle's.
    let expected = top_k(&universe, 35);
    assert_eq!(count(&second, "counts", "in"), 35);
    assert_eq!(
        count(&second, "occurrence_counts", "in"),
        occurrences(&expected)
    );
    let second_set = second["set"].as_str().unwrap();
    assert_eq!(
        row_ids(&host.all_records(second_set, json!({}), 100)),
        ids(&expected)
    );
    let above: Vec<Orig> = universe
        .iter()
        .filter(|original| f64::from(score_of(&original.text)) >= 0.5)
        .cloned()
        .collect();
    assert!(!above.is_empty() && above.len() < universe.len());
    let third_set = third["set"].as_str().unwrap();
    assert_eq!(
        row_ids(&host.all_records(third_set, json!({}), 100)),
        ids(&above)
    );

    // A rule over a sub-population reuses the scores too: outside it nothing is
    // evaluated, and a rank counts its members only.
    let filtered = host.select_filter(snapshot, "literal", "login");
    let scope = filtered["set"].as_str().unwrap();
    let within: Vec<Orig> = universe
        .iter()
        .filter(|original| original.text.contains("login"))
        .cloned()
        .collect();
    let narrowed = host.select_top(scope, 5);
    assert_eq!(narrowed["cost"]["archive_scans"], 0, "{narrowed}");
    assert_eq!(narrowed["cost"]["rows_scored"], 0, "{narrowed}");
    assert_eq!(narrowed["cost"]["reused"]["kind"], "scores");
    let best = top_k(&within, 5);
    let narrowed_set = narrowed["set"].as_str().unwrap();
    assert_eq!(
        row_ids(&host.all_records(narrowed_set, json!({}), 100)),
        ids(&best)
    );
    assert_eq!(count(&narrowed, "counts", "parent"), within.len() as u64);
    assert_eq!(count(&narrowed, "counts", "unknown"), 0);
    assert_eq!(
        narrowed["outside_scope"]["distinct_originals"],
        (universe.len() - within.len()) as u64
    );
    let mut ranked: Vec<&Orig> = within.iter().collect();
    ranked.sort_by(|a, b| {
        score_of(&b.text)
            .total_cmp(&score_of(&a.text))
            .then(a.position.cmp(&b.position))
    });
    for place in [0, 4, ranked.len() - 1] {
        let evidence = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": scope, "view": "evidence", "record": ranked[place].id(),
                   "sets": {"a": narrowed_set}}),
        );
        let explained = &evidence["explanation"][1];
        assert_eq!(explained["set"], "a");
        // The rank read the score of every original, beside the row of the record.
        assert_eq!(
            evidence["cost"]["index_rows_read"],
            universe.len() as u64 + 1,
            "{evidence}"
        );
        assert_eq!(explained["rank"], place as u64 + 1, "{evidence}");
        assert_eq!(explained["ranked_among"], within.len() as u64);
        assert_eq!(
            explained["state"],
            if place < 5 { "in" } else { "out" },
            "{evidence}"
        );
    }
}

#[test]
fn a_filter_over_a_selection_is_derived_from_the_filter_over_the_snapshot() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let universe = oracle(&entries, ".log");
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let a = host.select_top(snapshot, 40);
    let b = host.select_filter(snapshot, "glob", "*login*");
    assert_eq!(b["cost"]["archive_scans"], 1);
    let before = work(&host, snapshot);

    let a_set = a["set"].as_str().unwrap();
    let within = host.select_filter(a_set, "glob", "*login*");
    assert_eq!(within["cost"]["archive_scans"], 0, "{within}");
    assert_eq!(within["cost"]["bytes_unpacked"], 0, "{within}");
    assert_eq!(within["cost"]["reused"]["kind"], "membership");
    assert_eq!(within["cost"]["reused"]["set"], b["set"]);
    assert_eq!(work(&host, snapshot), before);

    let top = top_k(&universe, 40);
    let expected: Vec<Orig> = top
        .iter()
        .filter(|original| original.text.contains("login"))
        .cloned()
        .collect();
    assert!(!expected.is_empty() && expected.len() < top.len());
    assert_eq!(count(&within, "counts", "parent"), 40);
    assert_eq!(count(&within, "counts", "in"), expected.len() as u64);
    assert_eq!(
        count(&within, "counts", "out"),
        (top.len() - expected.len()) as u64
    );
    assert_eq!(count(&within, "counts", "unknown"), 0);
    assert_eq!(
        within["outside_scope"]["distinct_originals"],
        (universe.len() - 40) as u64
    );
    let within_set = within["set"].as_str().unwrap();
    assert_eq!(
        row_ids(&host.all_records(within_set, json!({}), 7)),
        ids(&expected)
    );
    // Another filter was not prepared: it reads the source.
    let other = host.select_filter(a_set, "literal", "session");
    assert_eq!(other["cost"]["archive_scans"], 1);
    assert_eq!(other["cost"]["reused"], Value::Null);
}

#[test]
fn warm_drilling_reads_no_source_and_scores_nothing() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let universe = oracle(&entries, ".log");
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let a = host.select_top(snapshot, 60);
    let b = host.select_filter(snapshot, "glob", "*login*");
    let sets = json!({"a": a["set"], "b": b["set"]});
    let before = work(&host, snapshot);
    let session_before = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "overview"}),
    )["session"]
        .clone();

    // Thirty interactions on prepared sets: overlap, reduce, page forward at three
    // page sizes, explain one row, come back to the regions.
    let mut interactions = 0;
    let top = top_k(&universe, 60);
    let only_a: Vec<Orig> = top
        .iter()
        .filter(|original| !original.text.contains("login"))
        .cloned()
        .collect();
    for round in 0..3 {
        let regions = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "regions", "sets": sets}),
        );
        assert_eq!(regions["cost"]["archive_scans"], 0);
        interactions += 1;
        let view = host.ok(
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": sets,
                   "expr": {"difference": ["a", "b"]}}),
        );
        assert_eq!(view["cost"]["archive_scans"], 0);
        assert_eq!(view["cost"]["rows_scored"], 0);
        assert_eq!(count(&view, "counts", "in"), only_a.len() as u64);
        interactions += 1;
        let handle = view["view"].as_str().unwrap();
        for limit in [1 + round, 7, 100] {
            let mut cursor: Option<String> = None;
            let mut seen = Vec::new();
            loop {
                let mut args = json!({"v": 1, "handle": handle, "view": "records",
                                      "limit": limit, "sets": sets});
                if let Some(cursor) = &cursor {
                    args["cursor"] = json!(cursor);
                }
                let page = host.ok("triager_inspect", args);
                assert_eq!(page["cost"]["archive_scans"], 0);
                interactions += 1;
                seen.extend(row_ids(page["rows"].as_array().unwrap()));
                match page["page"]["next_cursor"].as_str() {
                    Some(next) => cursor = Some(next.to_string()),
                    None => break,
                }
            }
            assert_eq!(seen, ids(&only_a), "limit {limit}");
        }
        let evidence = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "evidence", "record": only_a[round].id(),
                   "sets": sets}),
        );
        assert_eq!(evidence["cost"]["archive_scans"], 0);
        interactions += 1;
    }
    assert!(interactions >= 30, "{interactions} interactions");
    assert_eq!(work(&host, snapshot), before);

    // What the session returned is counted, per call and in bytes.
    let session = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "overview"}),
    )["session"]
        .clone();
    let calls = |session: &Value| session["calls"].as_u64().unwrap();
    let bytes = |session: &Value| session["bytes_returned"].as_u64().unwrap();
    // The interactions, the three overviews of `work` and the one above.
    assert_eq!(calls(&session) - calls(&session_before), interactions + 2);
    assert!(bytes(&session) - bytes(&session_before) > 300 * interactions);
}

#[test]
fn every_job_states_its_cost_in_the_units_of_the_gate() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let universe = oracle(&entries, ".log");
    let records: u64 = entries[..2]
        .iter()
        .map(|(_, lines)| lines.len() as u64)
        .sum();

    let reply = host.reply(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private", "entries": ["*.log"]}}),
    );
    let opened: Value = serde_json::from_str(&reply.json).unwrap();
    let cost = &opened["cost"];
    assert_eq!(cost["archive_scans"], 1);
    assert_eq!(cost["entries_read"], 2);
    assert_eq!(cost["entries_skipped"], 1);
    assert_eq!(cost["records_scanned"], records);
    assert_eq!(cost["records_skipped"], 0);
    assert_eq!(cost["accepted_occurrences"], records);
    assert_eq!(cost["distinct_originals"], universe.len() as u64);
    assert_eq!(cost["rows_scored"], 0);
    let unpacked: u64 = entries[..2]
        .iter()
        .flat_map(|(_, lines)| lines.iter().map(|line| line.len() as u64 + 1))
        .sum();
    assert_eq!(cost["bytes_unpacked"], unpacked);
    assert!(cost["bytes_packed"].as_u64().unwrap() > 0, "{cost}");
    assert_eq!(cost["spilled_bytes"], 0);
    assert_eq!(cost["dedup_on_disk"], false);
    for counter in [
        "elapsed_ms",
        "queue_ms",
        "memory_limit_bytes",
        "server_counted_peak_bytes",
        "disk_bytes",
        "disk_peak_bytes",
    ] {
        assert!(cost[counter].is_u64(), "{counter}: {cost}");
    }
    // The process peak is observed where the operating system gives it, and a
    // counter that is not available says so.
    if cfg!(windows) {
        assert!(cost["server_process_peak_bytes"].as_u64().unwrap() > 1024 * 1024);
        assert_eq!(cost["unavailable"], json!([]));
    } else if cost["server_process_peak_bytes"].is_null() {
        assert_eq!(cost["unavailable"], json!(["server_process_peak_bytes"]));
    }
    assert_eq!(
        cost["response_bytes"],
        l2d_triager::service::transport_bytes(&reply.json) as u64
    );

    let snapshot = opened["snapshot"].as_str().unwrap();
    let reply = host.reply(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "score", "model": "sum", "top_k": 20, "ties": "record_id_ascending"}}),
    );
    let a: Value = serde_json::from_str(&reply.json).unwrap();
    let cost = &a["cost"];
    assert_eq!(cost["archive_scans"], 1);
    assert_eq!(cost["bytes_unpacked"], unpacked);
    assert_eq!(cost["records_scanned"], records);
    assert_eq!(cost["accepted_occurrences"], records);
    assert_eq!(cost["candidates"], universe.len() as u64);
    assert_eq!(cost["output"], 20);
    assert_eq!(cost["rows_scored"], universe.len() as u64);
    assert_eq!(cost["tables_on_disk"], false);
    assert_eq!(
        cost["response_bytes"],
        l2d_triager::service::transport_bytes(&reply.json) as u64
    );

    let b = host.select_filter(snapshot, "literal", "login");
    let reply = host.reply(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"a": a["set"], "b": b["set"]}, "expr": {"difference": ["a", "b"]}}),
    );
    let view: Value = serde_json::from_str(&reply.json).unwrap();
    let cost = &view["cost"];
    assert_eq!(cost["archive_scans"], 0);
    assert_eq!(cost["bytes_unpacked"], 0);
    assert_eq!(cost["rows_scored"], 0);
    assert_eq!(cost["candidates"], universe.len() as u64);
    assert_eq!(cost["output"], view["counts"]["in"]);
    assert_eq!(cost["index_rows_read"], universe.len() as u64);
    assert!(
        cost["queue_ms"].is_u64() && cost["server_process_peak_bytes"].is_u64() == cfg!(windows)
    );
    assert_eq!(
        cost["response_bytes"],
        l2d_triager::service::transport_bytes(&reply.json) as u64
    );

    // A page states what it read of the index and what it returns.
    let reply = host.reply(
        "triager_inspect",
        json!({"v": 1, "handle": view["view"], "view": "records", "limit": 3}),
    );
    let page: Value = serde_json::from_str(&reply.json).unwrap();
    assert_eq!(page["cost"]["archive_scans"], 0);
    assert_eq!(page["cost"]["rows_scored"], 0);
    assert_eq!(
        page["cost"]["response_bytes"],
        l2d_triager::service::transport_bytes(&reply.json) as u64
    );
}

/// The open test gap of task 1.2: an evidence request whose retained part alone
/// leaves no room for an excerpt is refused before the source is read.
#[test]
fn evidence_that_leaves_no_room_for_an_excerpt_is_refused_before_the_read() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let set = host.select_filter(snapshot, "literal", "login");
    let mut sets = serde_json::Map::new();
    for (at, name) in ["a", "b", "c", "d", "e"].iter().enumerate() {
        let view = host.ok(
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"s": set["set"]},
                   "expr": {"union": vec![json!("s"); 60 - at]}}),
        );
        sets.insert((*name).to_string(), view["view"].clone());
    }
    let page = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": set["set"], "view": "records", "limit": 1}),
    );
    let record = page["rows"][0]["id"].clone();
    let before = host.activity(snapshot);
    let request = |projection: Value| {
        json!({"v": 1, "handle": set["set"], "view": "evidence", "record": record,
               "sets": sets, "projection": projection})
    };
    // Without an excerpt the evidence of these sets fits.
    let retained = host.ok(
        "triager_inspect",
        request(json!([
            "id",
            "locator",
            "occurrences",
            "membership",
            "explanation"
        ])),
    );
    assert_eq!(retained["explanation"].as_array().unwrap().len(), 6);
    let (code, message) = host.err(
        "triager_inspect",
        request(json!([
            "id",
            "locator",
            "occurrences",
            "membership",
            "explanation",
            "excerpt"
        ])),
    );
    assert_eq!(code, "invalid_arguments");
    assert!(
        message.contains("leaves no room for an excerpt") && message.contains("was not read"),
        "{message}"
    );
    // Nothing was scheduled and nothing was read.
    let after = host.activity(snapshot);
    assert_eq!(after["archive_scans"], before["archive_scans"]);
    assert_eq!(after["jobs_started"], before["jobs_started"]);
    // With fewer sets the same record has room for its excerpt.
    let fewer = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": set["set"], "view": "evidence", "record": record,
               "sets": {"a": sets["a"]}, "projection": ["id", "explanation", "excerpt"]}),
    );
    assert!(fewer["excerpt"]["text"].as_str().unwrap().contains("login"));
}
