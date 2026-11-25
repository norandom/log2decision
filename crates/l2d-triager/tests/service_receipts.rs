//! Triager task 2.2, requirements 2.3, 3.1, 3.2, 3.3, 4.1, 4.3, 6.1 and 6.3: one
//! review budget allocated once over named pools against an independent oracle; a
//! candidate view and a final work set saved as receipts and reopened by other
//! servers without their source and without new work; atomic, checked receipts; the
//! recipe of a receipt replayed on the same and on a changed source; and a late job
//! that completes under the parent it was started in.

mod support;

use std::time::Duration;

use l2d_triager::service::Service;
use serde_json::{Value, json};
use support::{
    Host, HostOptions, Orig, auth_entries, budget, config, membership_line, oracle, record_id,
    score_of, top_k, write_archive,
};

/// What the stand-in for C says of the original at `index` of the universe.
fn c_state(index: usize) -> Option<&'static str> {
    match index % 4 {
        0 => Some("in"),
        1 => Some("out"),
        2 => Some("abstain"),
        _ => None,
    }
}

fn c_file(universe: &[Orig]) -> String {
    universe
        .iter()
        .enumerate()
        .rev()
        .filter_map(|(index, original)| {
            c_state(index).map(|state| membership_line(&original.id(), state))
        })
        .collect()
}

fn options(universe: &[Orig]) -> HostOptions {
    HostOptions {
        memberships: vec![("c".into(), c_file(universe))],
        ..HostOptions::default()
    }
}

/// Another server of the same project over the same sources and receipts, with a
/// work directory of its own: a second client session.
fn second(first: &Host, options: &HostOptions) -> Host {
    let work = tempfile::tempdir().unwrap();
    let mut config = config(first.dir.path(), "project-one", options);
    config.work_root = work.path().join("work");
    let service = Service::start(
        config,
        budget(options),
        Box::new(support::TestScorers {
            delay: options.delay,
        }),
    )
    .unwrap();
    Host { dir: work, service }
}

/// The handles of the sets of the recipe: A, B, C and D = A − B.
struct Sets {
    snapshot: String,
    a: String,
    b: String,
    c: String,
    d: String,
}

fn reduce(host: &Host) -> Sets {
    let snapshot = host.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let a = host.select_top(&snapshot, 40)["set"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "glob", "*login*")["set"]
        .as_str()
        .unwrap()
        .to_string();
    let c = host.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {"kind": "membership", "ref": "c"}}),
    )["set"]
        .as_str()
        .unwrap()
        .to_string();
    let d = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"a": a, "b": b}, "expr": {"difference": ["a", "b"]}}),
    )["view"]
        .as_str()
        .unwrap()
        .to_string();
    Sets {
        snapshot,
        a,
        b,
        c,
        d,
    }
}

/// The budget of the worked recipe at fixture scale: recommendations by score, the
/// unfamiliar by record ID, and a seeded audit of assessed passes outside the
/// candidate expression.
fn budget_of_recipe() -> Value {
    json!({"total": 7, "duplicate_rule": "first_pool_wins", "shortfall": "leave_unfilled",
           "pools": [
               {"name": "general", "limit": 3, "from": {"intersection": ["d", "c"]},
                "order": ["score_desc", "record_id_ascending"]},
               {"name": "unfamiliar", "limit": 2,
                "from": {"intersection": ["d", {"state": {"set": "c", "is": "not_evaluated"}}]},
                "order": ["record_id_ascending"]},
               {"name": "audit", "limit": 2, "from": {"difference": ["d", "c"]},
                "order": ["seeded_hash", "record_id_ascending"], "seed": 5}]})
}

fn allocate(host: &Host, snapshot: &str, d: &str, c: &str) -> Value {
    host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "allocate", "snapshot": snapshot, "sets": {"d": d, "c": c},
               "expr": {"intersection": ["d", "c"]}, "budget": budget_of_recipe()}),
    )
}

fn ids(rows: &[Value]) -> Vec<String> {
    rows.iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect()
}

/// The ordered work set, its pools and its counts: what two servers must agree on.
fn work_set(host: &Host, f: &str) -> Value {
    let rows = host.all_records(f, json!({}), 3);
    let overview = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": f, "view": "overview"}),
    );
    let allocation = overview["artifact"]["allocation"].clone();
    json!({
        "ids": ids(&rows),
        "allocations": rows.iter().map(|row| row["allocation"].clone()).collect::<Vec<_>>(),
        "allocation": allocation,
    })
}

#[test]
fn one_budget_fills_named_pools_once_with_reasons_and_deferrals() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let host = Host::new(&entries, options(&universe));
    let sets = reduce(&host);

    // The oracle: the populations of the pools, from the fixture alone.
    let a = top_k(&universe, 40);
    let d: Vec<(usize, &Orig)> = universe
        .iter()
        .enumerate()
        .filter(|(_, original)| {
            a.iter().any(|kept| kept.position == original.position)
                && !original.text.contains("login")
        })
        .collect();
    let with = |state: Option<&str>| -> Vec<&Orig> {
        d.iter()
            .filter(|(index, _)| match (c_state(*index), state) {
                (Some("abstain"), None) => true,
                (found, wanted) => found == wanted,
            })
            .map(|(_, original)| *original)
            .collect()
    };
    let mut general = with(Some("in"));
    let unfamiliar = with(None);
    let mut audit = with(Some("out"));
    assert!(
        general.len() > 3,
        "the fixture needs a deferred recommendation"
    );
    assert!(!unfamiliar.is_empty() && audit.len() > 2);
    general.sort_by(|x, y| {
        score_of(&y.text)
            .total_cmp(&score_of(&x.text))
            .then(x.position.cmp(&y.position))
    });
    audit.sort_by_key(|original| {
        (
            l2d_decide::policy::audit_key(&original.id(), 5),
            original.position,
        )
    });
    let mut expected: Vec<(String, &str)> = Vec::new();
    expected.extend(general.iter().take(3).map(|o| (o.id(), "general")));
    expected.extend(unfamiliar.iter().take(2).map(|o| (o.id(), "unfamiliar")));
    expected.extend(audit.iter().take(2).map(|o| (o.id(), "audit")));
    let occurrences =
        |originals: &[&Orig]| -> u64 { originals.iter().map(|o| o.occurrences).sum() };

    let f = allocate(&host, &sets.snapshot, &sets.d, &sets.c);
    assert_eq!(f["kind"], "final");
    assert_eq!(f["capacity"], 7);
    assert_eq!(f["selected"], expected.len() as u64);
    let allocation = &f["allocation"];
    assert_eq!(
        allocation["rule"],
        "named-pools-first-pool-wins-leave-unfilled-fnv1a-audit-v1"
    );
    assert_eq!(allocation["unused"], 7 - expected.len() as u64);
    let pools = allocation["pools"].as_array().unwrap();
    for (pool, (name, population)) in pools.iter().zip([
        ("general", &general),
        ("unfamiliar", &unfamiliar),
        ("audit", &audit),
    ]) {
        assert_eq!(pool["name"], name);
        assert_eq!(
            pool["eligible"],
            json!({"distinct_originals": population.len(), "occurrences": occurrences(population)})
        );
        let taken = population
            .len()
            .min(pool["limit"].as_u64().unwrap() as usize);
        assert_eq!(pool["selected"], taken as u64, "{name}");
        assert_eq!(
            pool["deferred"]["distinct_originals"],
            (population.len() - taken) as u64
        );
        assert_eq!(
            pool["deferred"]["occurrences"],
            occurrences(&population[taken..])
        );
    }
    // The audit and the unfamiliar are drawn outside the candidate expression D ∩ C
    // (C is out for the one, not evaluated for the other), and say so.
    assert_eq!(pools[0]["outside_candidate"], 0);
    assert_eq!(pools[1]["outside_candidate"], pools[1]["selected"]);
    assert_eq!(pools[2]["outside_candidate"], pools[2]["selected"]);
    assert_eq!(
        allocation["outside_candidate"]["distinct_originals"],
        pools[1]["selected"].as_u64().unwrap() + pools[2]["selected"].as_u64().unwrap()
    );
    assert_eq!(
        allocation["candidate"]["distinct_originals"],
        with(Some("in")).len() as u64
    );
    let eligible = general.len() + unfamiliar.len() + audit.len();
    assert_eq!(
        allocation["deferred"]["distinct_originals"],
        (eligible - expected.len()) as u64
    );
    // The final set's reduction within its parent, the snapshot.
    assert_eq!(f["counts"]["in"], expected.len() as u64);
    assert_eq!(f["counts"]["parent"], universe.len() as u64);

    // Its members in the order of the work set, with the pool of every slot.
    let handle = f["set"].as_str().unwrap();
    let rows = host.all_records(handle, json!({"d": sets.d, "c": sets.c}), 2);
    let found: Vec<(String, String)> = rows
        .iter()
        .map(|row| {
            (
                row["id"].as_str().unwrap().to_string(),
                row["allocation"]["pool"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let wanted: Vec<(String, String)> = expected
        .iter()
        .map(|(id, pool)| (id.clone(), pool.to_string()))
        .collect();
    assert_eq!(found, wanted);
    for (slot, row) in rows.iter().enumerate() {
        assert_eq!(row["allocation"]["slot"], slot as u64 + 1);
        assert_eq!(row["membership"]["d"], "in");
    }
    // By record ID on request, the same members.
    let by_id = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "records", "sort": ["record_id_ascending"],
               "limit": 100}),
    );
    let mut sorted: Vec<String> = expected.iter().map(|(id, _)| id.clone()).collect();
    sorted.sort();
    assert_eq!(ids(by_id["rows"].as_array().unwrap()), sorted);

    // Why a slot holds an audited original, and why a recommendation was deferred.
    let audited = &expected[expected.len() - 1].0;
    let evidence = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "evidence", "record": audited}),
    );
    let explained = &evidence["explanation"][0]["allocation"];
    assert_eq!(explained["allocation"], "audit");
    assert_eq!(explained["outside_candidate"], true);
    assert!(
        explained["reason"]
            .as_str()
            .unwrap()
            .contains("outside the candidate expression"),
        "{explained}"
    );
    let deferred = general[3].id();
    let evidence = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": sets.d, "view": "evidence", "record": deferred,
               "sets": {"f": handle}}),
    );
    let explained = &evidence["explanation"][1];
    assert_eq!(explained["state"], "out");
    assert_eq!(explained["allocation"]["deferred"], true);
    assert_eq!(explained["allocation"]["eligible_for"], json!(["general"]));
    // An allocation suggests an order of review: nothing of it is a reviewed mark.
    assert!(
        f["suggestion_only"]
            .as_str()
            .unwrap()
            .contains("no analyst mark")
    );

    // The same request is the same final set; it is allocated once.
    let again = allocate(&host, &sets.snapshot, &sets.d, &sets.c);
    assert_eq!(again["set"], f["set"]);
    assert_eq!(again["cost"], f["cost"]);

    // First pool wins and a short pool leaves its slots unused.
    let overlap = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "allocate", "snapshot": sets.snapshot,
               "sets": {"d": sets.d, "c": sets.c}, "expr": "d",
               "budget": {"total": 53, "duplicate_rule": "first_pool_wins",
                          "shortfall": "leave_unfilled", "pools": [
                    {"name": "first", "limit": 3, "from": "d",
                     "order": ["score_desc", "record_id_ascending"]},
                    {"name": "second", "limit": 50, "from": {"intersection": ["d", "c"]},
                     "order": ["record_id_ascending"]}]}}),
    );
    let first: Vec<String> = {
        let mut ranked: Vec<&Orig> = d.iter().map(|(_, original)| *original).collect();
        ranked.sort_by(|x, y| {
            score_of(&y.text)
                .total_cmp(&score_of(&x.text))
                .then(x.position.cmp(&y.position))
        });
        ranked.iter().take(3).map(|o| o.id()).collect()
    };
    let taken_twice = general.iter().filter(|o| first.contains(&o.id())).count() as u64;
    let pools = overlap["allocation"]["pools"].as_array().unwrap();
    assert_eq!(pools[1]["taken_by_other_pools"], taken_twice);
    assert_eq!(pools[1]["selected"], general.len() as u64 - taken_twice);
    assert_eq!(
        pools[1]["unused"],
        50 - (general.len() as u64 - taken_twice)
    );
    assert_eq!(overlap["selected"], 3 + general.len() as u64 - taken_twice);
}

#[test]
fn a_saved_candidate_reopens_in_another_server_and_allocates_there_as_here() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = options(&universe);
    let first = Host::new(&entries, options.clone());
    let sets = reduce(&first);
    let named = json!({"a": sets.a, "b": sets.b, "c": sets.c});
    let regions_here = first.ok(
        "triager_inspect",
        json!({"v": 1, "handle": sets.d, "view": "regions", "sets": named}),
    );
    let records_here = first.all_records(&sets.d, named.clone(), 7);
    let sibling = |host: &Host, snapshot: &str, d: &str, c: &str, is: Value| -> Value {
        host.ok(
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "scope": d,
                   "sets": {"d": d, "c": c}, "expr": {"intersection": ["d", is]}}),
        )
    };
    let recommended_here = sibling(&first, &sets.snapshot, &sets.d, &sets.c, json!("c"));
    let unknown_here = sibling(
        &first,
        &sets.snapshot,
        &sets.d,
        &sets.c,
        json!({"state": {"set": "c", "is": "not_evaluated"}}),
    );
    let saved = first.ok(
        "triager_save",
        json!({"v": 1, "handle": sets.d, "sets": {"c": sets.c}, "idempotency_key": "hunt-one"}),
    );
    let receipt = saved["receipt"].as_str().unwrap().to_string();
    assert_eq!(saved["kind"], "candidate");
    assert_eq!(
        saved["retained"]["distinct_originals"],
        records_here.len() as u64
    );
    assert_eq!(saved["retained"]["sets"], 4);
    assert!(saved["contains"].as_array().unwrap().len() > 5);
    assert!(
        saved["excludes"]
            .as_array()
            .unwrap()
            .contains(&json!("source text and excerpts"))
    );
    assert_eq!(saved["recipe"]["replayable"], true);
    // A second save of the same view is the same receipt; one without C is another.
    let again = first.ok(
        "triager_save",
        json!({"v": 1, "handle": sets.d, "sets": {"c": sets.c}}),
    );
    assert_eq!(again["receipt"], saved["receipt"]);
    assert_eq!(again["cost"]["receipt_published"], false);
    let without = first.ok("triager_save", json!({"v": 1, "handle": sets.d}));
    assert_ne!(without["receipt"], saved["receipt"]);
    assert_eq!(without["retained"]["sets"], 3);
    // The receipt holds no host path and no text of the source.
    let text = std::fs::read_to_string(
        first
            .dir
            .path()
            .join("receipts")
            .join(format!("{receipt}.receipt")),
    )
    .unwrap();
    let root = first.dir.path().to_string_lossy().replace('\\', "\\\\");
    assert!(!text.contains(&root), "a host path in the receipt");
    for original in &universe {
        assert!(!text.contains(&format!("\"{}\"", original.text)));
    }
    let here = allocate(&first, &sets.snapshot, &sets.d, &sets.c);
    let here_set = work_set(&first, here["set"].as_str().unwrap());

    // A second server reopens the candidate: no source read, no score, no file.
    let second = second(&first, &options);
    let opened = second.ok("triager_open", json!({"v": 1, "receipt": receipt}));
    assert_eq!(opened["kind"], "candidate");
    assert_eq!(opened["saved_as"], sets.d);
    assert_eq!(opened["origin"]["snapshot"], sets.snapshot);
    assert_eq!(
        opened["origin"]["distinct_originals"],
        universe.len() as u64
    );
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let reopened = |saved_as: &str| -> String {
        opened["sets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|set| set["saved_as"] == saved_as)
            .unwrap_or_else(|| panic!("{saved_as} in {opened}"))["handle"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let (a, b, c, d) = (
        reopened(&sets.a),
        reopened(&sets.b),
        reopened(&sets.c),
        reopened(&sets.d),
    );
    assert_eq!(opened["set"], d);
    // The source, the saved counts and the coverage of every set, as they were saved.
    let opened_here = first.ok(
        "triager_inspect",
        json!({"v": 1, "handle": sets.snapshot, "view": "overview"}),
    );
    assert_eq!(opened["source"], opened_here["artifact"]["source"]);
    assert_eq!(
        opened["origin"]["revision"],
        opened_here["artifact"]["revision"]
    );
    for (here, there) in [(&sets.c, &c), (&sets.d, &d)] {
        let made = first.ok(
            "triager_inspect",
            json!({"v": 1, "handle": here, "view": "overview"}),
        );
        let kept = second.ok(
            "triager_inspect",
            json!({"v": 1, "handle": there, "view": "overview"}),
        );
        assert_eq!(
            kept["artifact"]["retained_from"]["counts"],
            made["artifact"]["counts"]
        );
        assert_eq!(
            kept["artifact"]["retained_from"]["occurrence_counts"],
            made["artifact"]["occurrence_counts"]
        );
        assert_eq!(
            kept["artifact"]["retained_from"]["revision"],
            made["artifact"]["revision"]
        );
        assert_eq!(
            kept["artifact"]["saved_coverage"],
            made["artifact"]["coverage"]
        );
    }
    let activity = second.activity(&snapshot);
    for counter in [
        "archive_scans",
        "bytes_unpacked",
        "rows_scored",
        "scorer_loads",
        "membership_reads",
        "rows_encoded",
        "genai_rows_issued",
        "genai_rows_submitted",
    ] {
        assert_eq!(activity[counter], 0, "{counter}: {activity}");
    }
    assert_eq!(activity["receipts_opened"], 1);
    let renamed = json!({"a": a, "b": b, "c": c});
    let regions_there = second.ok(
        "triager_inspect",
        json!({"v": 1, "handle": d, "view": "regions", "sets": renamed}),
    );
    for field in ["regions", "unknown_regions", "unknown", "complete"] {
        assert_eq!(regions_there[field], regions_here[field], "{field}");
    }
    let records_there = second.all_records(&d, renamed.clone(), 5);
    assert_eq!(records_there, records_here);
    // The evidence the receipt retains: the score and its producer, a membership not
    // evaluated with its reason; no excerpt.
    let row = &records_here[0]["id"];
    let evidence = second.ok(
        "triager_inspect",
        json!({"v": 1, "handle": d, "view": "evidence", "record": row, "sets": renamed}),
    );
    let explained = evidence["explanation"].as_array().unwrap();
    assert!(explained[1]["score"].is_number(), "{evidence}");
    assert_eq!(explained[1]["producer"]["kind"], "score");
    assert!(
        evidence["excerpt_estimate"]["requires"]
            .as_str()
            .unwrap()
            .contains("does not retain")
    );
    // The sibling views under the reopened D: recommended and unknown, as here.
    let recommended = sibling(&second, &snapshot, &d, &c, json!("c"));
    let unknown = sibling(
        &second,
        &snapshot,
        &d,
        &c,
        json!({"state": {"set": "c", "is": "not_evaluated"}}),
    );
    for (there, here) in [(&recommended, &recommended_here), (&unknown, &unknown_here)] {
        assert_eq!(there["counts"], here["counts"]);
        assert_eq!(there["occurrence_counts"], here["occurrence_counts"]);
        assert_eq!(there["parent"], d);
    }
    // Allocated once from the reopened D: the work set allocated from D here.
    let there = allocate(&second, &snapshot, &d, &c);
    let f = there["set"].as_str().unwrap().to_string();
    assert_eq!(work_set(&second, &f), here_set);
    // What reads the source is refused by name, and changes nothing.
    for (tool, args) in [
        (
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {"kind": "score", "model": "sum",
                   "top_k": 3, "ties": "record_id_ascending"}}),
        ),
        (
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {"kind": "membership", "ref": "c"}}),
        ),
        (
            "triager_inspect",
            json!({"v": 1, "handle": d, "view": "evidence", "record": row,
                   "projection": ["id", "excerpt"]}),
        ),
        (
            "triager_refine",
            json!({"v": 1, "set": b, "action": {"kind": "patterns"}}),
        ),
    ] {
        let (code, message) = second.err(tool, args);
        assert_eq!(code, "source_not_retained", "{tool}: {message}");
        assert!(message.contains(&receipt), "{message}");
    }
    assert_eq!(second.activity(&snapshot)["archive_scans"], 0);

    // The final work set saved by the second server reopens in a third one: the same
    // ordered IDs, slots, reasons and deferred counts.
    let final_receipt = second.ok("triager_save", json!({"v": 1, "handle": f}));
    assert_eq!(final_receipt["kind"], "final");
    let third = second_of(&first, &options);
    let opened = third.ok(
        "triager_open",
        json!({"v": 1, "receipt": final_receipt["receipt"]}),
    );
    assert_eq!(opened["kind"], "final");
    let reopened_f = opened["set"].as_str().unwrap();
    assert_eq!(work_set(&third, reopened_f), here_set);
    let audited = &here_set["ids"][here_set["ids"].as_array().unwrap().len() - 1];
    let explained = third.ok(
        "triager_inspect",
        json!({"v": 1, "handle": reopened_f, "view": "evidence", "record": audited}),
    );
    assert_eq!(
        explained["explanation"][0]["allocation"]["allocation"],
        "audit"
    );
    let third_snapshot = opened["snapshot"].as_str().unwrap();
    assert_eq!(third.activity(third_snapshot)["archive_scans"], 0);
    assert_eq!(third.activity(third_snapshot)["rows_scored"], 0);
}

/// A third session, as [`second`].
fn second_of(first: &Host, options: &HostOptions) -> Host {
    second(first, options)
}

#[test]
fn receipts_are_published_whole_checked_when_reopened_and_bound_to_the_host() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = options(&universe);
    let host = Host::new(&entries, options.clone());
    let sets = reduce(&host);
    let saved = host.ok("triager_save", json!({"v": 1, "handle": sets.d}));
    let receipt = saved["receipt"].as_str().unwrap().to_string();
    let receipts = host.dir.path().join("receipts");
    let path = receipts.join(format!("{receipt}.receipt"));
    let bytes = std::fs::read(&path).unwrap();
    // A staging file a crash left behind is never read, and saving goes on.
    std::fs::write(receipts.join(".saving-1-1.tmp"), &bytes[..bytes.len() / 2]).unwrap();

    // A changed byte, a cut file and a file under another receipt's name: refused.
    let session = second(&host, &options);
    let mut changed = bytes.clone();
    let at = changed.len() - 300;
    changed[at] = if changed[at] == b'0' { b'1' } else { b'0' };
    for (case, content) in [
        ("changed", changed),
        ("cut", bytes[..bytes.len() - 40].to_vec()),
    ] {
        std::fs::write(&path, &content).unwrap();
        let (code, message) = session.err("triager_open", json!({"v": 1, "receipt": receipt}));
        assert_eq!(code, "refused", "{case}: {message}");
        assert!(message.contains("damaged"), "{case}: {message}");
        assert!(!message.contains(&host.dir.path().to_string_lossy().to_string()));
    }
    let renamed = format!("rcpt-{}", "f".repeat(32));
    std::fs::write(receipts.join(format!("{renamed}.receipt")), &bytes).unwrap();
    let (code, message) = session.err("triager_open", json!({"v": 1, "receipt": renamed}));
    assert_eq!(code, "refused", "{message}");
    assert!(message.contains("digests"), "{message}");
    std::fs::write(&path, &bytes).unwrap();
    let opened = session.ok("triager_open", json!({"v": 1, "receipt": receipt}));
    assert_eq!(opened["receipt"], receipt);
    // Reopened twice, one snapshot.
    let again = session.ok("triager_open", json!({"v": 1, "receipt": receipt}));
    assert_eq!(again["snapshot"], opened["snapshot"]);

    // Another project does not reopen it; the public-development profile does not
    // reopen a private source.
    let elsewhere = {
        let work = tempfile::tempdir().unwrap();
        let mut config = config(host.dir.path(), "project-two", &options);
        config.work_root = work.path().join("work");
        let service = Service::start(
            config,
            budget(&options),
            Box::new(support::TestScorers {
                delay: Duration::ZERO,
            }),
        )
        .unwrap();
        Host { dir: work, service }
    };
    let (code, _) = elsewhere.err("triager_open", json!({"v": 1, "receipt": receipt}));
    assert_eq!(code, "wrong_project");
    let public = second(
        &host,
        &HostOptions {
            profile: l2d_triager::service::DisclosureProfile::PublicDevelopment,
            ..options.clone()
        },
    );
    let (code, _) = public.err("triager_open", json!({"v": 1, "receipt": receipt}));
    assert_eq!(code, "disclosure_refused");

    // A save the disk does not admit is refused before anything is written.
    let full = {
        let work = tempfile::tempdir().unwrap();
        let mut config = config(host.dir.path(), "project-one", &options);
        config.work_root = work.path().join("work");
        config.receipts = Some(work.path().join("receipts"));
        // Enough for the scope's relation, written a chunk of 44 KiB at a time, and
        // less than a receipt of the view needs at most.
        config.free_disk_bytes = Some(80_000);
        let service = Service::start(
            config,
            budget(&options),
            Box::new(support::TestScorers {
                delay: Duration::ZERO,
            }),
        )
        .unwrap();
        Host { dir: work, service }
    };
    let sets_full = reduce(&full);
    let (code, message) = full.err("triager_save", json!({"v": 1, "handle": sets_full.d}));
    assert_eq!(code, "refused", "{message}");
    let left = std::fs::read_dir(full.dir.path().join("receipts"))
        .map(|entries| entries.count())
        .unwrap_or(0);
    assert_eq!(left, 0, "a refused save leaves nothing");
    // A host without a receipt directory saves nothing.
    let none = {
        let work = tempfile::tempdir().unwrap();
        let mut config = config(host.dir.path(), "project-one", &options);
        config.work_root = work.path().join("work");
        config.receipts = None;
        let service = Service::start(
            config,
            budget(&options),
            Box::new(support::TestScorers {
                delay: Duration::ZERO,
            }),
        )
        .unwrap();
        Host { dir: work, service }
    };
    let (code, _) = none.err("triager_save", json!({"v": 1, "handle": "view-x"}));
    assert_eq!(code, "receipts_not_configured");
}

/// Runs the recipe of `receipt` in `host` and returns the result of its save.
fn replay(host: &Host, receipt: &str) -> Value {
    let mut output = Vec::new();
    l2d_triager::mcp::run_recipe(&host.service, receipt, &mut output).unwrap();
    let last = String::from_utf8(output).unwrap();
    let last: Value = serde_json::from_str(last.lines().last().unwrap()).unwrap();
    assert_eq!(last["tool"], "triager_save", "{last}");
    last["result"].clone()
}

#[test]
fn a_recipe_replays_to_the_same_content_and_to_a_new_revision_on_a_changed_source() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = options(&universe);
    let host = Host::new(&entries, options.clone());
    let sets = reduce(&host);
    let f = allocate(&host, &sets.snapshot, &sets.d, &sets.c);
    let saved = host.ok("triager_save", json!({"v": 1, "handle": f["set"]}));
    let receipt = saved["receipt"].as_str().unwrap().to_string();
    assert_eq!(saved["recipe"]["replayable"], true);
    assert_eq!(saved["lineage"]["parent_receipt"], Value::Null);

    // On the same source another server makes the same content, with this receipt as
    // its parent.
    let same = second(&host, &options);
    let replayed = replay(&same, &receipt);
    {
        let read = |receipt: &Value| -> Vec<String> {
            let path = host
                .dir
                .path()
                .join("receipts")
                .join(format!("{}.receipt", receipt.as_str().unwrap()));
            std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .map(str::to_string)
                .collect()
        };
        let (old, new) = (read(&saved["receipt"]), read(&replayed["receipt"]));
        let mut old_header: Value = serde_json::from_str(&old[1]).unwrap();
        let mut new_header: Value = serde_json::from_str(&new[1]).unwrap();
        old_header["lineage"] = Value::Null;
        new_header["lineage"] = Value::Null;
        assert_eq!(new_header, old_header, "the headers but their lineage");
        assert_eq!(new[2..new.len() - 1], old[2..old.len() - 1]);
    }
    assert_eq!(replayed["content_sha256"], saved["content_sha256"]);
    assert_eq!(replayed["rows_sha256"], saved["rows_sha256"]);
    assert_eq!(replayed["lineage"]["parent_receipt"], receipt);
    assert_ne!(replayed["receipt"], saved["receipt"]);
    assert_eq!(
        replayed["set"], f["set"],
        "the same inputs give the same handles"
    );

    // The source changes: the replay is a new revision with the same lineage.
    let mut changed = entries.clone();
    changed[0]
        .1
        .extend((0..40).map(|index| format!("login failed user=x{index:02} late")));
    write_archive(&host.dir.path().join("private.zip"), &changed);
    let revised = second(&host, &options);
    let replayed = replay(&revised, &receipt);
    assert_ne!(replayed["content_sha256"], saved["content_sha256"]);
    assert_ne!(replayed["snapshot"], saved["snapshot"]);
    assert_eq!(replayed["lineage"]["parent_receipt"], receipt);
    assert_eq!(replayed["recipe"]["replayable"], true);
    let header = |receipt: &Value| -> Value {
        let path = host
            .dir
            .path()
            .join("receipts")
            .join(format!("{}.receipt", receipt.as_str().unwrap()));
        let text = std::fs::read_to_string(path).unwrap();
        serde_json::from_str(text.lines().nth(1).unwrap()).unwrap()
    };
    let old = header(&saved["receipt"]);
    let new = header(&replayed["receipt"]);
    assert_ne!(
        old["source"]["content_sha256"],
        new["source"]["content_sha256"]
    );
    assert_eq!(new["lineage"]["parent_receipt"], receipt);

    // A recipe is not run from a receipt whose bytes are not those of its handle: a
    // step changed on disk (a wider top-k) is refused before any call is made.
    let path = host
        .dir
        .path()
        .join("receipts")
        .join(format!("{receipt}.receipt"));
    let original = std::fs::read_to_string(&path).unwrap();
    assert!(original.contains("\"top_k\":40"));
    std::fs::write(&path, original.replacen("\"top_k\":40", "\"top_k\":41", 1)).unwrap();
    let tampered = second(&host, &options);
    let mut output = Vec::new();
    let refused = l2d_triager::mcp::run_recipe(&tampered.service, &receipt, &mut output)
        .expect_err("a tampered recipe is refused");
    assert!(refused.to_string().contains("damaged"), "{refused}");
    assert!(output.is_empty(), "no call of the recipe was made");
    std::fs::write(&path, original).unwrap();
    let mut output = Vec::new();
    l2d_triager::mcp::run_recipe(&tampered.service, &receipt, &mut output).unwrap();
}

/// A job keeps the parent it was started under: the client moves to another branch
/// while it runs, and the late result names its own parent and leaves the branch the
/// client is on unchanged.
#[test]
fn a_late_job_completes_under_its_original_parent() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = HostOptions {
        delay: Duration::from_millis(40),
        batch_rows: Some(8),
        inline_wait: Some(Duration::from_millis(1)),
        ..options(&universe)
    };
    let host = Host::new(&entries, options);
    let snapshot = host.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "glob", "*login*")["set"]
        .as_str()
        .unwrap()
        .to_string();
    let c = host.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {"kind": "membership", "ref": "c"}}),
    )["set"]
        .as_str()
        .unwrap()
        .to_string();
    let view = |expr: Value| -> String {
        host.ok(
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot,
                   "sets": {"b": b, "c": c}, "expr": expr}),
        )["view"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let left = view(json!({"difference": ["b", "c"]}));
    let right = view(json!({"intersection": ["b", "c"]}));
    let right_before = host.all_records(&right, json!({"c": c}), 50);
    // A score inside the left branch reads the source, slowly.
    let started = host.reply(
        "triager_select",
        json!({"v": 1, "scope": left, "producer": {"kind": "score", "model": "sum",
               "top_k": 4, "ties": "record_id_ascending"}}),
    );
    let started: Value = serde_json::from_str(&started.json).unwrap();
    assert_eq!(started["status"], "running", "{started}");
    let job = started["job"].as_str().unwrap().to_string();
    // The client is on the right branch meanwhile.
    let right_during = host.all_records(&right, json!({"c": c}), 50);
    assert_eq!(right_during, right_before);
    let status = host.job(&job, "status");
    assert!(
        ["running", "waiting"].contains(&status["status"].as_str().unwrap()),
        "{status}"
    );
    let late = host.wait(&job);
    assert_eq!(late["scope"], left);
    assert_eq!(late["counts"]["parent"], {
        let overview = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": left, "view": "overview"}),
        );
        overview["members"]["distinct_originals"].clone()
    });
    // The late result is that of its branch: within the left view, and every member
    // of it is a member of the left view.
    let members = host.all_records(
        late["set"].as_str().unwrap(),
        json!({"l": left, "r": right}),
        50,
    );
    assert_eq!(members.len(), 4);
    assert!(members.iter().all(|row| row["membership"]["l"] == "in"));
    assert!(members.iter().all(|row| row["membership"]["r"] == "out"));
    assert_eq!(host.all_records(&right, json!({"c": c}), 50), right_before);
    let _ = record_id(0);
}

/// A saved policy report as `l2d triage` writes one with an incompatible profile: every
/// decision is routed to review as unsupported, and the first three hold slots of the
/// policy's own allocation, so that the route codes of the records differ.
fn policy_report(source_path: &str, content_sha256: &Value, decided: &[&Orig]) -> Value {
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
    let entries = vec!["*.log".to_string()];
    let zip = l2d_core::ZipInputSettings {
        entries: entries.clone(),
        ..l2d_core::ZipInputSettings::default()
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

/// What an explanation says of a record that a receipt must keep: the state and its
/// reason, the Venn route, the distance, the pattern, and the model judgments.
fn kept_evidence(explained: &Value) -> Value {
    json!({
        "state": explained["state"],
        "reason": explained["reason"],
        "venn": explained["venn"],
        "distance": explained["distance"],
        "pattern": explained["pattern"],
        "matched": explained["matched"],
        "outcome": explained["recommendation"]["outcome"],
        "judgments": explained["recommendation"]["judgments"],
    })
}

/// Requirements 2.2 and 3.3: the evidence of every producer survives a receipt. A view
/// saved with a Venn C, a pattern set, a neighbour set and an issued-batch C beside it
/// reopens in another server with the same Venn routes (from the f32 route codes), the
/// same distances and pattern membership, and the same model judgments with their
/// reasons, the caller-reported model and prompt revision; nothing is inferred again,
/// and the recipe says why it cannot be replayed.
#[test]
fn producer_evidence_and_model_judgments_survive_a_receipt() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = HostOptions {
        policies: vec!["v1".into()],
        ..options(&universe)
    };
    let first = Host::new(&entries, options.clone());
    let opened = first.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let path = first.dir.path().join("private.zip");
    // Decisions over the members of B only, so that the ones the policy allocated are
    // among the records the saved view retains, whatever order their IDs take.
    let decided: Vec<&Orig> = universe
        .iter()
        .filter(|original| original.text.contains("login failed"))
        .step_by(2)
        .collect();
    std::fs::write(
        first.dir.path().join("policy-v1.json"),
        policy_report(
            &path.to_string_lossy(),
            &opened["source"]["content_sha256"],
            &decided,
        )
        .to_string(),
    )
    .unwrap();
    let venn = first.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "recommend", "method": "venn", "policy": "v1",
            "membership": "investigate_or_review"}}),
    )["set"]
        .clone();
    let b = first.select_filter(&snapshot, "literal", "login failed")["set"].clone();
    let members: Vec<String> = ids(&first.all_records(b.as_str().unwrap(), json!({}), 100));
    let refinement = first.ok(
        "triager_refine",
        json!({"v": 1, "set": b, "action": {"kind": "patterns"}}),
    );
    let pattern = first.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "filter", "syntax": "pattern", "refinement": refinement["refinement"],
            "pattern": refinement["patterns"][0]["pattern"]}}),
    )["set"]
        .clone();
    let near = first.ok(
        "triager_refine",
        json!({"v": 1, "refinement": refinement["refinement"], "action": {
            "kind": "neighbours", "exemplar": members[2], "method": "character_trigram",
            "top_k": 7, "radius": 0.7}}),
    )["set"]
        .clone();
    let issue = |producer: Value| {
        let mut producer = producer;
        producer["kind"] = json!("recommend");
        producer["method"] = json!("issued_batch");
        first.ok(
            "triager_select",
            json!({"v": 1, "scope": b, "producer": producer}),
        )
    };
    let batch = issue(json!({"action": "issue", "limit": 6}));
    let issued: Vec<String> = batch["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| record["id"].as_str().unwrap().to_string())
        .collect();
    issue(
        json!({"action": "submit", "batch": batch["batch"], "expected_revision": 1,
                 "model": "dev/model@1", "prompt_revision": "p1",
                 "items": [{"id": issued[0], "outcome": "recommend", "reason": "repeated failures"},
                           {"id": issued[1], "outcome": "pass", "reason": "a known user"},
                           {"id": issued[3], "outcome": "abstain"}]}),
    );
    let judged = issue(json!({"action": "collect"}))["set"].clone();
    let view = first.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"b": b}, "expr": "b"}),
    )["view"]
        .clone();
    let sets = json!({"v": venn, "p": pattern, "n": near, "c": judged});
    let evidence = |host: &Host, handle: &Value, sets: &Value, record: &str| -> Vec<Value> {
        let read = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "evidence", "record": record,
                   "sets": sets}),
        );
        read["explanation"].as_array().unwrap()[1..]
            .iter()
            .map(kept_evidence)
            .collect()
    };
    let here: Vec<Vec<Value>> = members
        .iter()
        .map(|record| evidence(&first, &view, &sets, record))
        .collect();
    // The fixture reaches every kind of evidence the receipt must keep.
    let flat: Vec<&Value> = here.iter().flatten().collect();
    assert!(
        flat.iter()
            .any(|kept| kept["venn"]["allocation"] == "general")
    );
    assert!(
        flat.iter()
            .any(|kept| kept["venn"]["deferred_by_policy"] == true)
    );
    assert!(flat.iter().any(|kept| kept["distance"].is_number()));
    assert!(flat.iter().any(|kept| kept["matched"] == true));
    assert!(
        flat.iter()
            .any(|kept| kept["judgments"][0]["reason"] == "repeated failures")
    );
    assert!(flat.iter().any(|kept| kept["outcome"] == "omitted"));

    let saved = first.ok(
        "triager_save",
        json!({"v": 1, "handle": view, "sets": {"v": venn, "p": pattern, "n": near, "c": judged}}),
    );
    assert_eq!(saved["recipe"]["replayable"], false);
    assert!(
        saved["recipe"]["reason"]
            .as_str()
            .unwrap()
            .contains("model client's judgments"),
        "{saved}"
    );
    assert!(
        saved["contains"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item.as_str().unwrap().starts_with("model judgments"))
    );

    let second = second(&first, &options);
    let reopened = second.ok("triager_open", json!({"v": 1, "receipt": saved["receipt"]}));
    assert_eq!(reopened["recipe"]["replayable"], false);
    let handle = |saved_as: &Value| -> Value {
        reopened["sets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|set| set["saved_as"] == *saved_as)
            .unwrap()["handle"]
            .clone()
    };
    let sets_there = json!({"v": handle(&venn), "p": handle(&pattern), "n": handle(&near),
                            "c": handle(&judged)});
    let view_there = handle(&view);
    let there: Vec<Vec<Value>> = members
        .iter()
        .map(|record| evidence(&second, &view_there, &sets_there, record))
        .collect();
    assert_eq!(there, here);
    // The run of the judgments, as it was reported.
    let read = second.ok(
        "triager_inspect",
        json!({"v": 1, "handle": view_there, "view": "evidence", "record": issued[0],
               "sets": {"c": handle(&judged)}}),
    );
    let run = &read["explanation"][1]["recommendation"]["run"];
    assert_eq!(run["model"], "dev/model@1");
    assert_eq!(run["prompt_revision"], "p1");
    assert_eq!(run["model_identity"], "caller-reported");
    assert_eq!(
        read["explanation"][1]["recommendation"]["kind"],
        "model suggestion; not a reviewed outcome"
    );
    let activity = second.activity(reopened["snapshot"].as_str().unwrap());
    for counter in [
        "archive_scans",
        "rows_scored",
        "rows_encoded",
        "genai_rows_issued",
        "genai_rows_submitted",
    ] {
        assert_eq!(activity[counter], 0, "{counter}");
    }
    // The receipt saved again from the reopened sets keeps the same judgments.
    let again = second.ok(
        "triager_save",
        json!({"v": 1, "handle": view_there, "sets": sets_there}),
    );
    let third = second_of(&first, &options);
    let reopened = third.ok("triager_open", json!({"v": 1, "receipt": again["receipt"]}));
    let handle = |saved_as: &Value| -> Value {
        reopened["sets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|set| set["saved_as"] == *saved_as)
            .unwrap()["handle"]
            .clone()
    };
    let sets_third = json!({"v": handle(&sets_there["v"]), "p": handle(&sets_there["p"]),
                            "n": handle(&sets_there["n"]), "c": handle(&sets_there["c"])});
    let again_there: Vec<Vec<Value>> = members
        .iter()
        .map(|record| evidence(&third, &handle(&view_there), &sets_third, record))
        .collect();
    assert_eq!(again_there, here);
}

/// The error of an allocation is shown without the host's paths: a table of an operand
/// that can no longer be read names the work directory as `<work directory>`.
#[test]
fn an_allocation_error_does_not_show_the_work_directory() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let host = Host::new(
        &entries,
        HostOptions {
            tables: l2d_triager::service::TablePlacement::Disk,
            ..options(&universe)
        },
    );
    let sets = reduce(&host);
    // Every table of a set on disk is cut short behind the server's back.
    let mut cut = 0;
    for entry in std::fs::read_dir(host.work()).unwrap() {
        let dir = entry.unwrap().path();
        let states = dir.join("states.bin");
        if states.is_file() {
            std::fs::OpenOptions::new()
                .write(true)
                .open(&states)
                .unwrap()
                .set_len(0)
                .unwrap();
            cut += 1;
        }
    }
    assert!(cut >= 3, "{cut} tables");
    let (code, message) = host.err(
        "triager_compose",
        json!({"v": 1, "mode": "allocate", "snapshot": sets.snapshot,
               "sets": {"d": sets.d, "c": sets.c},
               "expr": {"intersection": ["d", "c"]}, "budget": budget_of_recipe()}),
    );
    assert_eq!(code, "runtime_failure", "{message}");
    assert!(message.contains("<work directory>"), "{message}");
    let root = host.dir.path().to_string_lossy().into_owned();
    assert!(!message.contains(&root), "{message}");
}
