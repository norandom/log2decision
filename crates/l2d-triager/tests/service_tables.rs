//! Triager task 1.3, requirements 1.2, 2.3 and 6.2: membership and scores kept on
//! disk give the reduction they give in memory, across batch sizes, spilled
//! deduplication and a relation on disk; a deterministic fixture membership gives the
//! eight known regions and the unknown combinations of an independent oracle.

mod support;

use std::collections::BTreeMap;

use l2d_triager::scope::IndexPlacement;
use l2d_triager::service::TablePlacement;
use serde_json::{Value, json};
use support::{Host, HostOptions, Orig, auth_entries, occurrences, oracle, top_k};

/// What the fixture producer says of the original at `index` of the universe: it
/// did not assess every third original nor any rejected challenge, abstained on every
/// fifth of the rest and recommends every fifth.
fn fixture_state(index: usize, original: &Orig) -> Option<&'static str> {
    if index % 3 == 2 || original.text.starts_with("challenge") {
        return None;
    }
    Some(match index % 5 {
        0 => "in",
        1 => "abstain",
        _ => "out",
    })
}

fn fixture(universe: &[Orig]) -> String {
    let mut lines = String::new();
    // Not in the order of the record IDs: the file is joined by key.
    for (index, original) in universe.iter().enumerate().rev() {
        if let Some(state) = fixture_state(index, original) {
            lines.push_str(&format!("{}\t{state}\n", original.id()));
        }
    }
    lines
}

fn options(universe: &[Orig]) -> HostOptions {
    let first = universe[0].id();
    HostOptions {
        memberships: vec![
            ("fixture-c".into(), fixture(universe)),
            (
                "foreign".into(),
                format!("{first}\tin\nr-00000000deadbeef\tout\n"),
            ),
            ("conflict".into(), format!("{first}\tin\n{first}\tout\n")),
            ("malformed".into(), format!("{first}\tmaybe\n")),
        ],
        ..HostOptions::default()
    }
}

fn row_ids(rows: &[Value]) -> Vec<String> {
    rows.iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect()
}

/// The regions of a result as a map from the membership of a region to its counts.
fn regions_of(result: &Value) -> BTreeMap<String, (u64, u64)> {
    let mut regions = BTreeMap::new();
    for list in ["regions", "unknown_regions"] {
        for region in result[list].as_array().unwrap() {
            let key: Vec<String> = ["a", "b", "c"]
                .iter()
                .filter_map(|name| region["membership"][name].as_str())
                .map(str::to_string)
                .collect();
            let counts = (
                region["distinct_originals"].as_u64().unwrap(),
                region["occurrences"].as_u64().unwrap(),
            );
            if list == "regions" || counts.0 > 0 {
                assert!(regions.insert(key.join("/"), counts).is_none());
            }
        }
    }
    regions
}

/// One reduction from the archive to A, B, the fixture C, D = A - B and E = D and C,
/// with everything about it that does not depend on how it was computed.
fn reduce(options: HostOptions, on_disk: bool) -> Value {
    let entries = auth_entries();
    let host = Host::new(&entries, options);
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    eprintln!(
        "relation ordered on disk: {}, entries read at once: {}, deduplication on disk: {}",
        opened["cost"]["index_ordered_on_disk"],
        opened["cost"]["entries_at_once"],
        opened["cost"]["dedup_on_disk"]
    );
    let a = host.select_top(snapshot, 30);
    let b = host.select_filter(snapshot, "glob", "*login*");
    let c = host.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {"kind": "membership", "ref": "fixture-c"}}),
    );
    let sets = json!({"a": a["set"], "b": b["set"], "c": c["set"]});
    let d = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": sets,
               "expr": {"difference": ["a", "b"]}}),
    );
    let e = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot, "scope": d["view"],
               "sets": {"d": d["view"], "c": c["set"]}, "expr": {"intersection": ["d", "c"]}}),
    );
    for made in [&a, &b, &c, &d, &e] {
        assert_eq!(made["cost"]["tables_on_disk"], on_disk, "{made}");
        assert_eq!(
            made["cost"]["disk_bytes"].as_u64().unwrap() > 0,
            on_disk,
            "{made}"
        );
    }
    if on_disk {
        // The tables are files of the work directory while the service lives.
        let files: Vec<String> = std::fs::read_dir(host.work())
            .unwrap()
            .flat_map(|dir| std::fs::read_dir(dir.unwrap().path()).unwrap())
            .map(|file| file.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            files.iter().filter(|name| *name == "states.bin").count(),
            5,
            "{files:?}"
        );
        assert_eq!(files.iter().filter(|name| *name == "scores.bin").count(), 1);
    }
    let regions = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions", "sets": sets}),
    );
    let within_d = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": d["view"], "view": "regions", "sets": {"c": c["set"]}}),
    );
    let d_rows = host.all_records(d["view"].as_str().unwrap(), sets.clone(), 4);
    let e_rows = host.all_records(e["view"].as_str().unwrap(), sets.clone(), 100);
    let c_rows = host.all_records(c["set"].as_str().unwrap(), sets.clone(), 9);
    let evidence = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": d["view"], "view": "evidence", "record": d_rows[0]["id"],
               "sets": sets}),
    );
    let stable = |made: &Value| {
        json!({"handle": made.get("set").or(made.get("view")), "revision": made["revision"],
               "counts": made["counts"], "occurrence_counts": made["occurrence_counts"],
               "coverage": made["coverage"]})
    };
    let mut outcome = json!({
        "counts": opened["counts"],
        "a": stable(&a), "b": stable(&b), "c": stable(&c), "d": stable(&d), "e": stable(&e),
        "regions": {"known": regions["regions"], "unknown": regions["unknown_regions"],
                    "total_unknown": regions["unknown"]},
        "within_d": {"known": within_d["regions"], "unknown": within_d["unknown_regions"]},
        "d_rows": d_rows, "e_rows": e_rows, "c_rows": c_rows,
        "explanation": evidence["explanation"], "membership": evidence["membership"],
    });
    // Every host has a directory of its own, and a handle binds the path of its
    // source: the handles are left out, the revisions of the memberships are not.
    fn scrub(value: &mut Value) {
        match value {
            Value::String(text) => {
                for kind in ["set-", "view-", "snap-"] {
                    if text.starts_with(kind) {
                        *text = kind.to_string();
                    }
                }
            }
            Value::Array(items) => items.iter_mut().for_each(scrub),
            Value::Object(fields) => fields.values_mut().for_each(scrub),
            _ => {}
        }
    }
    scrub(&mut outcome);
    let Host { dir, service } = host;
    drop(service);
    assert!(
        std::fs::read_dir(dir.path().join("work")).map_or(true, |mut left| left.next().is_none()),
        "the work directory is emptied with the service"
    );
    outcome
}

#[test]
fn a_fixture_membership_gives_the_eight_regions_and_the_unknown_combinations() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let host = Host::new(&entries, options(&universe));
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let a = host.select_top(snapshot, 30);
    let b = host.select_filter(snapshot, "glob", "*login*");
    let c = host.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {"kind": "membership", "ref": "fixture-c"}}),
    );

    // The oracle: every original with its state in A, B and C.
    let top: Vec<u64> = top_k(&universe, 30).iter().map(|o| o.position).collect();
    let state = |index: usize, original: &Orig| -> [&'static str; 3] {
        [
            if top.contains(&original.position) {
                "in"
            } else {
                "out"
            },
            if original.text.contains("login") {
                "in"
            } else {
                "out"
            },
            match fixture_state(index, original) {
                Some("in") => "in",
                Some("out") => "out",
                _ => "not_evaluated",
            },
        ]
    };
    let mut expected: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for a in ["in", "out"] {
        for b in ["in", "out"] {
            for c in ["in", "out"] {
                expected.insert(format!("{a}/{b}/{c}"), (0, 0));
            }
        }
    }
    let (mut abstained, mut uncovered) = ((0u64, 0u64), (0u64, 0u64));
    let mut in_c = Vec::new();
    for (index, original) in universe.iter().enumerate() {
        let counts = expected
            .entry(state(index, original).join("/"))
            .or_default();
        counts.0 += 1;
        counts.1 += original.occurrences;
        match fixture_state(index, original) {
            Some("abstain") => {
                abstained.0 += 1;
                abstained.1 += original.occurrences;
            }
            None => {
                uncovered.0 += 1;
                uncovered.1 += original.occurrences;
            }
            Some("in") => in_c.push(original.clone()),
            _ => {}
        }
    }
    assert!(abstained.0 > 0 && uncovered.0 > 0 && !in_c.is_empty());
    // The fixture has every known region and unknown combinations beside them.
    assert_eq!(expected.len(), 12);
    assert!(expected.values().all(|counts| counts.0 > 0), "{expected:?}");

    // C states what it assessed, and why the rest is not evaluated.
    assert_eq!(c["producer"]["kind"], "membership");
    assert_eq!(c["producer"]["ref"], "fixture-c");
    assert_eq!(c["producer"]["content_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(c["counts"]["in"], in_c.len() as u64);
    assert_eq!(c["occurrence_counts"]["in"], occurrences(&in_c));
    assert_eq!(c["counts"]["unknown"], abstained.0 + uncovered.0);
    assert_eq!(c["coverage"]["not_evaluated"], abstained.0 + uncovered.0);
    assert_eq!(
        c["coverage"]["reasons"],
        json!([
            {"reason": "abstained", "distinct_originals": abstained.0, "occurrences": abstained.1},
            {"reason": "outside_producer_coverage", "distinct_originals": uncovered.0,
             "occurrences": uncovered.1},
        ])
    );
    assert_eq!(c["complete"], false);
    assert_eq!(c["cost"]["archive_scans"], 0);
    assert_eq!(c["cost"]["rows_scored"], 0);

    let sets = json!({"a": a["set"], "b": b["set"], "c": c["set"]});
    let regions = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions", "sets": sets}),
    );
    assert_eq!(regions["regions"].as_array().unwrap().len(), 8);
    assert_eq!(regions_of(&regions), expected);
    let unknown: (u64, u64) = expected
        .iter()
        .filter(|(key, _)| key.contains("not_evaluated"))
        .fold((0, 0), |sum, (_, counts)| {
            (sum.0 + counts.0, sum.1 + counts.1)
        });
    assert_eq!(regions["unknown"]["distinct_originals"], unknown.0);
    assert_eq!(regions["unknown"]["occurrences"], unknown.1);
    assert_eq!(regions["complete"], false);

    // "What did C miss": assessed negatives apart from what it never assessed.
    let missed = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": sets,
               "expr": {"difference": [{"union": ["a", "b"]}, "c"]}}),
    );
    let candidates = |c: &str| -> u64 {
        expected
            .iter()
            .filter(|(key, _)| key.ends_with(c) && !key.starts_with("out/out"))
            .map(|(_, counts)| counts.0)
            .sum()
    };
    assert_eq!(missed["counts"]["in"], candidates("/out"));
    assert_eq!(missed["counts"]["unknown"], candidates("not_evaluated"));
    assert_eq!(
        missed["coverage"]["reasons"][0]["reason"],
        "operand_not_evaluated"
    );

    // The evidence of a record names the reason it is not evaluated.
    for (wanted, reason) in [
        (Some("abstain"), "abstained"),
        (None, "outside_producer_coverage"),
    ] {
        let (_, original) = universe
            .iter()
            .enumerate()
            .find(|(index, original)| fixture_state(*index, original) == wanted)
            .unwrap();
        let evidence = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "evidence", "record": original.id(),
                   "sets": {"c": c["set"]}}),
        );
        assert_eq!(evidence["membership"]["c"]["state"], "not_evaluated");
        assert_eq!(evidence["explanation"][0]["reason"], reason, "{evidence}");
    }

    // Within a population the fixture is evaluated only there.
    let narrowed = host.ok(
        "triager_select",
        json!({"v": 1, "scope": b["set"], "producer": {"kind": "membership", "ref": "fixture-c"}}),
    );
    let within: u64 = expected
        .iter()
        .filter(|(key, _)| key.contains("/in/") && !key.ends_with("not_evaluated"))
        .map(|(_, counts)| counts.0)
        .sum();
    assert_eq!(narrowed["coverage"]["evaluated"], within);
    assert_eq!(narrowed["cost"]["reused"]["set"], c["set"]);
    // The file is read once to bind its content and once more for its lines; a
    // derived selection still reads it once, and says so.
    let bytes = c["producer"]["bytes"].as_u64().unwrap();
    assert_eq!(bytes, fixture(&universe).len() as u64);
    assert_eq!(c["cost"]["membership_bytes_read"], 2 * bytes);
    assert_eq!(narrowed["cost"]["membership_bytes_read"], bytes);

    // What is no membership of this snapshot is refused, and nothing is committed.
    for (reference, code, needle) in [
        ("absent", "unknown_membership", "registered no membership"),
        ("foreign", "refused", "line 2"),
        ("conflict", "refused", "line 2"),
        ("malformed", "refused", "line 1"),
    ] {
        let (found, message) = host.err(
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {"kind": "membership", "ref": reference}}),
        );
        assert_eq!(found, code, "{reference}: {message}");
        assert!(message.contains(needle), "{reference}: {message}");
    }
}

#[test]
fn tables_on_disk_give_the_reduction_they_give_in_memory() {
    let universe = oracle(&auth_entries(), ".log");
    let reference = reduce(options(&universe), false);
    assert_eq!(reference["a"]["counts"]["in"], 30);
    assert!(reference["d_rows"].as_array().unwrap().len() > 4);
    assert!(!reference["e_rows"].as_array().unwrap().is_empty());
    assert!(
        row_ids(reference["c_rows"].as_array().unwrap())
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    );
    for (tables, index, batch_rows, dedup_set_texts) in [
        (
            TablePlacement::Disk,
            IndexPlacement::MemoryWhenAdmitted,
            None,
            None,
        ),
        (
            TablePlacement::Disk,
            IndexPlacement::Disk,
            Some(64),
            Some(20),
        ),
        (
            TablePlacement::Disk,
            IndexPlacement::Disk,
            Some(1_024),
            None,
        ),
        (
            TablePlacement::MemoryWhenAdmitted,
            IndexPlacement::Disk,
            Some(64),
            Some(20),
        ),
        (
            TablePlacement::MemoryWhenAdmitted,
            IndexPlacement::MemoryWhenAdmitted,
            Some(7),
            Some(3),
        ),
    ] {
        let outcome = reduce(
            HostOptions {
                tables,
                index,
                batch_rows,
                dedup_set_texts,
                ..options(&universe)
            },
            tables == TablePlacement::Disk,
        );
        assert_eq!(
            difference(&outcome, &reference, "outcome"),
            None,
            "{tables:?} {index:?} batch {batch_rows:?} set {dedup_set_texts:?}"
        );
    }
}

/// Requirement 6.2 without the host's switch: under a memory limit that does not
/// admit the states and scores of a selection beside its read of the source, the
/// selection goes to disk and completes with the membership it has in memory, where
/// task 1.2 refused it.
///
/// The limits are tried from below in steps of 256 KiB, each with a service of its
/// own over an archive of 300,000 originals, whose score tables take 1.3 MB and whose
/// disk form takes 0.3 MB. Long in a debug build:
///
/// `cargo test -j 4 -p l2d-triager --test service_tables -- --ignored --nocapture`
#[test]
#[ignore = "opens an archive of 300,000 originals at up to 48 memory limits"]
fn a_selection_the_memory_does_not_admit_goes_to_disk_and_completes() {
    const ORIGINALS: usize = 300_000;
    let entries: Vec<(String, Vec<String>)> = (0..3)
        .map(|entry| {
            let lines = (0..ORIGINALS / 3)
                .map(|line| format!("host-{entry} request {line} took {} ms", line * 7 % 1000))
                .collect();
            (format!("logs/host-{entry}.log"), lines)
        })
        .collect();
    let select = |host: &Host| -> Result<Value, String> {
        let opened = host.reply(
            "triager_open",
            json!({"v": 1, "source": {"ref": "private"}}),
        );
        let opened: Value = serde_json::from_str(&opened.json).unwrap();
        if opened["status"] == "error" {
            return Err(format!("open: {}", opened["error"]["message"]));
        }
        let selected = host.reply(
            "triager_select",
            json!({"v": 1, "scope": opened["snapshot"], "producer": {
                "kind": "score", "model": "sum", "top_k": 1000, "ties": "record_id_ascending"}}),
        );
        let selected: Value = serde_json::from_str(&selected.json).unwrap();
        if selected["status"] == "error" {
            return Err(format!("select: {}", selected["error"]["message"]));
        }
        Ok(selected)
    };
    let reference = select(&Host::new(&entries, HostOptions::default())).unwrap();
    assert_eq!(reference["cost"]["tables_on_disk"], false);
    assert_eq!(reference["counts"]["in"], 1000);

    let mut on_disk = None;
    for step in 0..48 {
        let usable = 12 * support::MIB + step * 256 * 1024;
        let host = Host::new(
            &entries,
            HostOptions {
                usable,
                ..HostOptions::default()
            },
        );
        match select(&host) {
            Err(refused) => {
                let shown: String = refused.chars().take(160).collect();
                eprintln!("usable {usable}: {shown}");
            }
            Ok(selected) => {
                eprintln!(
                    "usable {usable}: completed, tables on disk: {}, read again for disk: {}, \
                     archive scans {}, counted peak {}",
                    selected["cost"]["tables_on_disk"],
                    selected["cost"]["read_again_for_disk"],
                    selected["cost"]["archive_scans"],
                    selected["cost"]["server_counted_peak_bytes"]
                );
                // The first limit that completes does so on disk: in memory the
                // tables take a megabyte more than their disk form.
                assert_eq!(selected["cost"]["tables_on_disk"], true, "{usable}");
                // A read that was refused with the tables in memory is counted too.
                let again = selected["cost"]["read_again_for_disk"] == true;
                assert_eq!(selected["cost"]["archive_scans"], if again { 2 } else { 1 });
                assert_eq!(selected["revision"], reference["revision"]);
                assert_eq!(selected["counts"], reference["counts"]);
                assert_eq!(
                    selected["occurrence_counts"],
                    reference["occurrence_counts"]
                );
                on_disk = Some(usable);
                break;
            }
        }
    }
    let on_disk = on_disk.expect("no limit up to 24 MiB completed the selection");
    // A megabyte and a half above it the tables are admitted in memory again.
    let above = Host::new(
        &entries,
        HostOptions {
            usable: on_disk + 3 * support::MIB / 2,
            ..HostOptions::default()
        },
    );
    let in_memory = select(&above).unwrap();
    assert_eq!(in_memory["cost"]["tables_on_disk"], false);
    assert_eq!(in_memory["revision"], reference["revision"]);
}

/// A refusal about a registered membership names it as the client did. Where the
/// host keeps the file is the host's: under the public-development profile the
/// client is a hosted agent, and no message shows it the path.
#[test]
fn a_membership_refusal_names_the_membership_and_never_the_hosts_path() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let host = Host::new(
        &entries,
        HostOptions {
            profile: l2d_triager::service::DisclosureProfile::PublicDevelopment,
            ..options(&universe)
        },
    );
    let opened = host.open("public-fixture");
    let directory = host.dir.path().to_string_lossy().into_owned();
    // The last one after its file is gone: the host removed it while the server ran.
    std::fs::remove_file(host.dir.path().join("membership-fixture-c.tsv")).unwrap();
    for (reference, needle) in [
        ("foreign", "line 2 of the membership `foreign`"),
        ("conflict", "line 2 of the membership `conflict`"),
        ("malformed", "line 1 of the membership `malformed`"),
        ("fixture-c", "registered as the membership `fixture-c`"),
    ] {
        let (code, message) = host.err(
            "triager_select",
            json!({"v": 1, "scope": opened["snapshot"],
                   "producer": {"kind": "membership", "ref": reference}}),
        );
        assert_eq!(code, "refused", "{reference}: {message}");
        assert!(message.contains(needle), "{reference}: {message}");
        for hidden in [directory.as_str(), "membership-", ".tsv", "\\", "/"] {
            assert!(
                !message.contains(hidden),
                "{reference} shows `{hidden}`: {message}"
            );
        }
    }
}

/// A selection inside a population is one result, whether it is read directly or
/// derived from the prepared set over the whole snapshot (review round 1 of task
/// 1.3: the same request under the same handle gave another `coverage`). Everything
/// but the cost is equal for a membership, a filter and a score rule.
#[test]
fn a_selection_inside_a_population_is_the_same_derived_or_read_directly() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let dir = tempfile::tempdir().unwrap();
    support::write_archive(&dir.path().join("private.zip"), &entries);
    support::write_archive(&dir.path().join("public.zip"), &entries);
    let producers = [
        json!({"kind": "membership", "ref": "fixture-c"}),
        json!({"kind": "filter", "syntax": "glob", "query": "*login*"}),
        json!({"kind": "score", "model": "sum", "top_k": 7, "ties": "record_id_ascending"}),
    ];
    let run = |prepared: bool| -> Vec<Value> {
        let service = support::start(dir.path(), "project-one", &options(&universe)).unwrap();
        let call = |tool: &str, args: Value| -> Value {
            let reply = service.call(tool, &args);
            let value: Value = serde_json::from_str(&reply.json).unwrap();
            assert!(!reply.is_error, "{tool} {args}: {value}");
            value
        };
        let opened = call(
            "triager_open",
            json!({"v": 1, "source": {"ref": "private", "entries": ["*.log"]}}),
        );
        let snapshot = opened["snapshot"].clone();
        let population = call(
            "triager_select",
            json!({"v": 1, "scope": snapshot,
                   "producer": {"kind": "filter", "syntax": "literal", "query": "user=u0"}}),
        );
        assert!(population["counts"]["in"].as_u64().unwrap() > 10);
        producers
            .iter()
            .map(|producer| {
                if prepared {
                    call(
                        "triager_select",
                        json!({"v": 1, "scope": snapshot, "producer": producer}),
                    );
                }
                let mut within = call(
                    "triager_select",
                    json!({"v": 1, "scope": population["set"], "producer": producer}),
                );
                let cost = within.as_object_mut().unwrap().remove("cost").unwrap();
                assert_eq!(cost["reused"].is_object(), prepared, "{producer}: {cost}");
                assert_eq!(
                    cost["archive_scans"] == 0,
                    prepared || producer["kind"] == "membership"
                );
                within
            })
            .collect()
    };
    let (direct, derived) = (run(false), run(true));
    for ((direct, derived), producer) in direct.iter().zip(&derived).zip(&producers) {
        assert!(
            direct["outside_scope"]["distinct_originals"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(difference(derived, direct, "derived"), None, "{producer}");
    }
}

/// Where two results first differ, with the two values there.
fn difference(found: &Value, expected: &Value, at: &str) -> Option<String> {
    match (found, expected) {
        (Value::Object(found), Value::Object(expected)) => {
            found.keys().chain(expected.keys()).find_map(|key| {
                difference(
                    found.get(key).unwrap_or(&Value::Null),
                    expected.get(key).unwrap_or(&Value::Null),
                    &format!("{at}.{key}"),
                )
            })
        }
        (Value::Array(found), Value::Array(expected)) if found.len() == expected.len() => found
            .iter()
            .zip(expected)
            .enumerate()
            .find_map(|(index, (found, expected))| {
                difference(found, expected, &format!("{at}[{index}]"))
            }),
        (Value::Array(found), Value::Array(expected)) => Some(format!(
            "{at}: {} items and {} expected",
            found.len(),
            expected.len()
        )),
        _ if found == expected => None,
        _ => Some(format!("{at}: {found} and {expected} expected")),
    }
}
