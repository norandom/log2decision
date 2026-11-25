//! Triager task 1.2: the shared operations behind the MCP adapter and the CLI
//! (requirements 1.2, 2.1, 2.3, 4.1, 4.2, 4.3, 4.4, 6.1 and 6.2), driven through
//! `Service::call` as both adapters drive them, and compared with an oracle computed
//! from the archive's lines and with the existing selection search.

mod support;

use std::collections::BTreeSet;
use std::time::Duration;

use l2d_core::resource::{BusyChoice, ResourceGuard};
use l2d_core::selections::{Member, SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{MemoryBudget, MemorySettings, ZipInputSettings};
use l2d_triager::scope::IndexPlacement;
use l2d_triager::service::{
    DisclosureProfile, MAX_EXCERPT_BYTES, MAX_RESULT_BYTES, MAX_ROWS, transport_bytes,
};
use serde_json::{Value, json};
use support::{
    GIB, Host, HostOptions, Orig, auth_entries, count, ids, occurrences, oracle, record_id,
    score_of, top_k,
};

fn row_ids(rows: &[Value]) -> Vec<String> {
    rows.iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect()
}

fn minus(a: &[Orig], b: &[Orig]) -> Vec<Orig> {
    let b: BTreeSet<u64> = b.iter().map(|original| original.position).collect();
    a.iter()
        .filter(|original| !b.contains(&original.position))
        .cloned()
        .collect()
}

fn both(a: &[Orig], b: &[Orig]) -> Vec<Orig> {
    let b: BTreeSet<u64> = b.iter().map(|original| original.position).collect();
    a.iter()
        .filter(|original| b.contains(&original.position))
        .cloned()
        .collect()
}

/// What one reduction gave, to compare between batch sizes, spill and index placement.
#[derive(Debug, PartialEq)]
struct Reduction {
    /// The revisions of A, of B and of the view: digests of each membership. The
    /// handles and the snapshot's relation also bind the archive's path.
    revisions: [String; 3],
    d_ids: Vec<String>,
    regions: Value,
}

fn reduce(options: HostOptions) -> Reduction {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let a = top_k(&universe, 40);
    let b: Vec<Orig> = universe
        .iter()
        .filter(|original| original.text.contains("login"))
        .cloned()
        .collect();
    let host = Host::new(&entries, options);

    // Open: both denominators, and what the entry filter left out, with its unit.
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    assert_eq!(opened["status"], "ready");
    assert_eq!(
        opened["counts"]["distinct_originals"],
        universe.len() as u64
    );
    assert_eq!(opened["counts"]["occurrences"], occurrences(&universe));
    assert_eq!(opened["completeness"]["complete"], true);
    assert_eq!(opened["completeness"]["entries_read"], 2);
    assert_eq!(opened["completeness"]["entries_excluded"], 1);
    assert_eq!(opened["completeness"]["reasons"][0]["unit"], "entries");
    assert_eq!(opened["source"]["entries"], json!(["*.log"]));

    // A and B with their scorer and filter revisions.
    let set_a = host.select_top(&snapshot, 40);
    assert_eq!(set_a["producer"]["scorer"]["revision"], "rev-1");
    assert_eq!(set_a["producer"]["ties"], "record_id_ascending");
    assert_eq!(count(&set_a, "counts", "parent"), universe.len() as u64);
    assert_eq!(count(&set_a, "counts", "in"), 40);
    assert_eq!(count(&set_a, "counts", "out"), universe.len() as u64 - 40);
    assert_eq!(count(&set_a, "counts", "unknown"), 0);
    assert_eq!(count(&set_a, "occurrence_counts", "in"), occurrences(&a));
    assert_eq!(
        count(&set_a, "occurrence_counts", "parent"),
        occurrences(&universe)
    );
    let set_b = host.select_filter(&snapshot, "glob", "*login*");
    assert_eq!(set_b["producer"]["query"], "*login*");
    assert_eq!(count(&set_b, "counts", "in"), b.len() as u64);
    assert_eq!(count(&set_b, "occurrence_counts", "in"), occurrences(&b));
    let (handle_a, handle_b) = (
        set_a["set"].as_str().unwrap().to_string(),
        set_b["set"].as_str().unwrap().to_string(),
    );
    assert_eq!(
        row_ids(&host.all_records(&handle_a, json!({}), 25)),
        ids(&a)
    );
    assert_eq!(
        row_ids(&host.all_records(&handle_b, json!({}), 100)),
        ids(&b)
    );

    // The overlap: counts first, in both units, reconciling to the parent.
    let regions = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions",
               "sets": {"a": handle_a, "b": handle_b}}),
    );
    let region = |a_state: &str, b_state: &str| -> (u64, u64) {
        let found: Vec<&Value> = regions["regions"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["membership"]["a"] == a_state && row["membership"]["b"] == b_state)
            .collect();
        assert_eq!(found.len(), 1, "{regions}");
        (
            found[0]["distinct_originals"].as_u64().unwrap(),
            found[0]["occurrences"].as_u64().unwrap(),
        )
    };
    let overlap = both(&a, &b);
    let a_only = minus(&a, &b);
    let b_only = minus(&b, &a);
    assert!(!overlap.is_empty() && !a_only.is_empty() && !b_only.is_empty());
    assert_eq!(
        region("in", "in"),
        (overlap.len() as u64, occurrences(&overlap))
    );
    assert_eq!(
        region("in", "out"),
        (a_only.len() as u64, occurrences(&a_only))
    );
    assert_eq!(
        region("out", "in"),
        (b_only.len() as u64, occurrences(&b_only))
    );
    let none = universe.len() - overlap.len() - a_only.len() - b_only.len();
    assert_eq!(region("out", "out").0, none as u64);
    assert_eq!(regions["regions"].as_array().unwrap().len(), 4);
    assert_eq!(regions["unknown_regions"], json!([]));
    assert_eq!(
        regions["parent"]["distinct_originals"],
        universe.len() as u64
    );
    assert_eq!(regions["snapshot"], snapshot.as_str());

    // A minus B as a view: the reduction receipt in both units.
    let view = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"a": handle_a, "b": handle_b}, "expr": {"difference": ["a", "b"]}}),
    );
    let handle_d = view["view"].as_str().unwrap().to_string();
    assert_eq!(view["parent"], snapshot.as_str());
    assert_eq!(view["counts"]["unit"], "distinct_originals");
    assert_eq!(count(&view, "counts", "parent"), universe.len() as u64);
    assert_eq!(count(&view, "counts", "in"), a_only.len() as u64);
    assert_eq!(
        count(&view, "counts", "out"),
        (universe.len() - a_only.len()) as u64
    );
    assert_eq!(count(&view, "counts", "unknown"), 0);
    assert_eq!(
        count(&view, "occurrence_counts", "in"),
        occurrences(&a_only)
    );
    assert_eq!(
        count(&view, "occurrence_counts", "parent"),
        occurrences(&universe)
    );
    assert_eq!(view["complete"], true);
    let share = view["counts"]["retained_share"].as_f64().unwrap();
    assert!((share - a_only.len() as f64 / universe.len() as f64).abs() < 1e-9);

    // Its members, paged by an opaque cursor, with their memberships.
    let rows = host.all_records(&handle_d, json!({"a": handle_a, "b": handle_b}), 7);
    assert_eq!(row_ids(&rows), ids(&a_only));
    for (row, original) in rows.iter().zip(&a_only) {
        assert_eq!(row["membership"], json!({"a": "in", "b": "out"}));
        assert_eq!(row["occurrences"], original.occurrences);
        assert_eq!(row["locator"]["entry"], original.entry.as_str());
        assert_eq!(row["locator"]["line"], original.line);
    }

    // One explanatory row: why it is in A, why not in B, and its text.
    let first = &a_only[0];
    let evidence = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": handle_d, "view": "evidence", "record": first.id(),
               "sets": {"a": handle_a, "b": handle_b},
               "projection": ["id", "locator", "occurrences", "membership", "explanation", "excerpt"]}),
    );
    assert_eq!(evidence["record"]["id"], first.id().as_str());
    assert_eq!(evidence["record"]["occurrences"], first.occurrences);
    assert_eq!(evidence["membership"]["a"]["state"], "in");
    assert_eq!(evidence["membership"]["b"]["state"], "out");
    assert_eq!(evidence["excerpt"]["text"], first.text.as_str());
    assert_eq!(evidence["excerpt"]["clipped"], false);
    let explained = evidence["explanation"].as_array().unwrap();
    let of_a = explained.iter().find(|item| item["set"] == "a").unwrap();
    assert!((of_a["score"].as_f64().unwrap() - f64::from(score_of(&first.text))).abs() < 1e-6);
    let mut ranked: Vec<&Orig> = universe.iter().collect();
    ranked.sort_by(|x, y| {
        score_of(&y.text)
            .total_cmp(&score_of(&x.text))
            .then(x.position.cmp(&y.position))
    });
    let rank = ranked
        .iter()
        .position(|original| original.position == first.position)
        .unwrap()
        + 1;
    assert_eq!(of_a["rank"], rank as u64);
    let of_b = explained.iter().find(|item| item["set"] == "b").unwrap();
    assert_eq!(of_b["matched"], false);
    assert_eq!(of_b["producer"]["query"], "*login*");

    let revision = |value: &Value| value["revision"].as_str().unwrap().to_string();
    Reduction {
        revisions: [revision(&set_a), revision(&set_b), revision(&view)],
        d_ids: row_ids(&rows),
        regions: regions["regions"].clone(),
    }
}

#[test]
fn the_reduction_to_a_minus_b_matches_the_oracle_at_every_batch_size_and_spill() {
    let reference = reduce(HostOptions::default());
    for (batch_rows, dedup_set_texts, index) in [
        (Some(64), None, IndexPlacement::MemoryWhenAdmitted),
        (Some(64), Some(50), IndexPlacement::Disk),
        (None, Some(10), IndexPlacement::MemoryWhenAdmitted),
        (Some(7), Some(100), IndexPlacement::Disk),
    ] {
        let other = reduce(HostOptions {
            batch_rows,
            dedup_set_texts,
            index,
            ..HostOptions::default()
        });
        assert_eq!(
            other, reference,
            "batch {batch_rows:?}, set {dedup_set_texts:?}, {index:?}"
        );
    }
}

#[test]
fn a_filter_selects_what_the_existing_selection_search_selects() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let zip = ZipInputSettings {
        entries: vec!["*.log".into()],
        ..ZipInputSettings::default()
    };
    for (syntax, query, native) in [
        (
            "literal",
            "login failed",
            SearchQuery::literal("login failed"),
        ),
        ("glob", "*login*", SearchQuery::glob("*login*")),
        (
            "glob",
            "GET /page/1? 200",
            SearchQuery::glob("GET /page/1? 200"),
        ),
        (
            "regex",
            r"user=u0[1-3]\b",
            SearchQuery::regex(r"user=u0[1-3]\b"),
        ),
    ] {
        let mut budget =
            MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB)).unwrap();
        let (selection, summary) = SelectionSet::search_archive(
            "native",
            native,
            SearchSettings::default(),
            &host.dir.path().join("private.zip"),
            &zip,
            &mut budget,
        )
        .unwrap();
        let expected: Vec<String> = selection
            .members()
            .iter()
            .map(|member| match member {
                Member::Position(position) => record_id(*position),
                Member::StoreId(id) => panic!("a store member {id}"),
            })
            .collect();
        assert!(!expected.is_empty(), "{syntax} {query}");
        let set = host.select_filter(snapshot, syntax, query);
        let rows = host.all_records(set["set"].as_str().unwrap(), json!({}), 100);
        assert_eq!(row_ids(&rows), expected, "{syntax} {query}");
        // The resource figures of the scan are those of the existing search.
        assert_eq!(set["cost"]["records_scanned"], summary.input.records);
        assert_eq!(
            set["cost"]["distinct_originals"],
            summary.input.distinct_texts
        );
        assert_eq!(
            set["cost"]["records_skipped"],
            summary.input.skipped.len() as u64
        );
        assert_eq!(set["cost"]["entries_read"], summary.read().count() as u64);
        assert_eq!(set["cost"]["archive_scans"], 1);
    }
}

#[test]
fn a_threshold_selects_every_score_at_or_above_it() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let set = host.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "score", "model": "sum", "threshold": 0.75}}),
    );
    let expected: Vec<Orig> = universe
        .iter()
        .filter(|original| score_of(&original.text) >= 0.75)
        .cloned()
        .collect();
    assert_eq!(set["producer"]["threshold"], 0.75);
    assert_eq!(
        row_ids(&host.all_records(set["set"].as_str().unwrap(), json!({}), 100)),
        ids(&expected)
    );
}

#[test]
fn a_set_made_inside_another_keeps_what_it_did_not_evaluate_unknown() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let a = top_k(&universe, 40);
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let set_a = host.select_top(snapshot, 40);
    let handle_a = set_a["set"].as_str().unwrap();

    // B evaluated only inside A: its parent is A, and outside A it is not evaluated.
    let inside = host.select_filter(handle_a, "glob", "*login*");
    let handle_inside = inside["set"].as_str().unwrap();
    let matching: Vec<Orig> = a
        .iter()
        .filter(|original| original.text.contains("login"))
        .cloned()
        .collect();
    assert_eq!(inside["scope"], handle_a);
    assert_eq!(count(&inside, "counts", "parent"), 40);
    assert_eq!(count(&inside, "counts", "in"), matching.len() as u64);
    assert_eq!(
        count(&inside, "counts", "out"),
        (a.len() - matching.len()) as u64
    );
    assert_eq!(
        inside["outside_scope"]["distinct_originals"],
        (universe.len() - 40) as u64
    );
    assert_eq!(inside["outside_scope"]["reason"], "outside_producer_scope");

    // A wider score set minus that B: `in - not_evaluated` stays unknown.
    let wide = top_k(&universe, 90);
    let set_wide = host.select_top(snapshot, 90);
    let handle_wide = set_wide["set"].as_str().unwrap();
    let view = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"w": handle_wide, "b": handle_inside},
               "expr": {"difference": ["w", "b"]}}),
    );
    let in_a: BTreeSet<u64> = a.iter().map(|original| original.position).collect();
    let known_in: Vec<Orig> = wide
        .iter()
        .filter(|original| in_a.contains(&original.position) && !original.text.contains("login"))
        .cloned()
        .collect();
    let unknown: Vec<Orig> = wide
        .iter()
        .filter(|original| !in_a.contains(&original.position))
        .cloned()
        .collect();
    assert_eq!(unknown.len(), 50);
    assert_eq!(count(&view, "counts", "in"), known_in.len() as u64);
    assert_eq!(count(&view, "counts", "unknown"), 50);
    assert_eq!(
        count(&view, "counts", "out"),
        (universe.len() - known_in.len() - 50) as u64
    );
    assert_eq!(
        count(&view, "occurrence_counts", "unknown"),
        occurrences(&unknown)
    );
    assert_eq!(view["complete"], false);
    assert_eq!(view["coverage"]["not_evaluated"], 50);
    assert_eq!(
        row_ids(&host.all_records(view["view"].as_str().unwrap(), json!({}), 100)),
        ids(&known_in)
    );

    // Union and intersection follow the same three-valued rules.
    let union = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"w": handle_wide, "b": handle_inside},
               "expr": {"union": ["w", "b"]}}),
    );
    // In W is in; out of W and outside A is unknown, since B was not evaluated there.
    assert_eq!(count(&union, "counts", "in"), 90);
    assert_eq!(
        count(&union, "counts", "unknown"),
        (universe.len() - 90) as u64
    );
    let intersection = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"w": handle_wide, "b": handle_inside},
               "expr": {"intersection": ["w", "b"]}}),
    );
    assert_eq!(count(&intersection, "counts", "in"), matching.len() as u64);
    assert_eq!(count(&intersection, "counts", "unknown"), 50);

    // Inverting a region within the declared population keeps the unknowns apart:
    // the receipts of the region and of its complement add up to the parent.
    let inverted = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"w": handle_wide, "b": handle_inside},
               "expr": {"complement": [{"difference": ["w", "b"]}]}}),
    );
    assert_eq!(
        count(&inverted, "counts", "in"),
        (universe.len() - known_in.len() - 50) as u64
    );
    assert_eq!(count(&inverted, "counts", "out"), known_in.len() as u64);
    assert_eq!(count(&inverted, "counts", "unknown"), 50);
    assert_eq!(
        inverted["operation"]["expr"]["complement"][0]["difference"][0],
        handle_wide
    );

    // The regions keep known overlap and unknown coverage apart, and reconcile.
    let regions = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions",
               "sets": {"w": handle_wide, "b": handle_inside}}),
    );
    let known: u64 = regions["regions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["distinct_originals"].as_u64().unwrap())
        .sum();
    let unknown_rows = regions["unknown_regions"].as_array().unwrap();
    let not_known: u64 = unknown_rows
        .iter()
        .map(|row| row["distinct_originals"].as_u64().unwrap())
        .sum();
    assert_eq!(known + not_known, universe.len() as u64);
    assert_eq!(not_known, (universe.len() - 40) as u64);
    assert!(
        unknown_rows.iter().any(|row| row["membership"]
            == json!({"w": "in", "b": "not_evaluated"})
            && row["distinct_originals"] == 50),
        "{regions}"
    );
    assert_eq!(regions["complete"], false);

    // Three sets: the eight fully known regions, "none" included, and the unknown
    // combinations beside them.
    let three = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions",
               "sets": {"a": handle_a, "w": handle_wide, "b": handle_inside}}),
    );
    let eight = three["regions"].as_array().unwrap();
    assert_eq!(eight.len(), 8);
    assert_eq!(
        eight[0]["membership"],
        json!({"a": "in", "b": "in", "w": "in"})
    );
    assert_eq!(eight[0]["distinct_originals"], matching.len() as u64);
    assert_eq!(
        eight[7]["membership"],
        json!({"a": "out", "b": "out", "w": "out"})
    );
    // Outside A the filter was not evaluated, so no region there is fully known.
    assert_eq!(eight[7]["distinct_originals"], 0);
    let all: u64 = eight
        .iter()
        .chain(three["unknown_regions"].as_array().unwrap())
        .map(|row| row["distinct_originals"].as_u64().unwrap())
        .sum();
    assert_eq!(all, universe.len() as u64);
    assert_eq!(
        three["unknown"]["distinct_originals"],
        (universe.len() - 40) as u64
    );
    let (code, _) = host.err(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions",
               "sets": {"a": handle_a, "w": handle_wide, "b": handle_inside, "d": handle_a}}),
    );
    assert_eq!(code, "invalid_arguments");

    // A view is a scope: the share of the parent it keeps is measured against it.
    let child = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot, "scope": handle_a,
               "sets": {"b": handle_inside}, "expr": "b"}),
    );
    assert_eq!(child["parent"], handle_a);
    assert_eq!(count(&child, "counts", "parent"), 40);
    assert_eq!(count(&child, "counts", "in"), matching.len() as u64);
    assert_eq!(child["root"]["distinct_originals"], universe.len() as u64);

    // Evidence of a record the view does not hold is refused, with its state.
    let outside = &unknown[0];
    let (code, message) = host.err(
        "triager_inspect",
        json!({"v": 1, "handle": view["view"], "view": "evidence", "record": outside.id()}),
    );
    assert_eq!(code, "record_not_in_scope");
    assert!(message.contains("not_evaluated"), "{message}");
}

#[test]
fn requests_are_validated_before_anything_is_scheduled() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let before = host.activity(snapshot);
    let set = host.select_filter(snapshot, "literal", "login");
    let handle = set["set"].as_str().unwrap();
    let scans = host.activity(snapshot)["archive_scans"].as_u64().unwrap();
    assert_eq!(scans, before["archive_scans"].as_u64().unwrap() + 1);
    let jobs = host.activity(snapshot)["jobs_started"].as_u64().unwrap();
    let first_login = host.all_records(handle, json!({}), 1)[0]["id"].clone();

    let score = |producer: Value| json!({"v": 1, "scope": snapshot, "producer": producer});
    for (tool, args, code, needle) in [
        (
            "triager_open",
            json!({"source": {"ref": "private"}}),
            "invalid_arguments",
            "`v`",
        ),
        (
            "triager_open",
            json!({"v": 2, "source": {"ref": "private"}}),
            "unsupported_version",
            "1",
        ),
        (
            "triager_open",
            json!({"v": 1, "source": {"ref": "private"}, "profile": "private_local"}),
            "invalid_arguments",
            "profile",
        ),
        (
            "triager_open",
            json!({"v": 1, "source": {"ref": "private", "disclosure": "public"}}),
            "invalid_arguments",
            "disclosure",
        ),
        (
            "triager_open",
            json!({"v": 1, "source": {"ref": "nowhere"}}),
            "unknown_source",
            "nowhere",
        ),
        (
            "triager_open",
            json!({"v": 1, "source": {"ref": "private", "path": "x.zip"}}),
            "invalid_arguments",
            "one of",
        ),
        (
            "triager_open",
            json!({"v": 1, "source": {"ref": "private", "adapter": "lines"}}),
            "unsupported_operation",
            "lines",
        ),
        (
            "triager_open",
            json!({"v": 1, "receipt": "H1"}),
            "unsupported_operation",
            "receipt",
        ),
        (
            "triager_open",
            json!({"v": 1, "source": {"ref": "private", "max_record_bytes": 0}}),
            "invalid_arguments",
            "max_record_bytes",
        ),
        (
            "triager_select",
            score(json!({"kind": "score", "model": "sum", "top_k": 5})),
            "invalid_arguments",
            "ties",
        ),
        (
            "triager_select",
            score(
                json!({"kind": "score", "model": "sum", "top_k": 5, "threshold": 0.5,
                      "ties": "record_id_ascending"}),
            ),
            "invalid_arguments",
            "either",
        ),
        (
            "triager_select",
            score(
                json!({"kind": "score", "model": "sum", "top_k": 0, "ties": "record_id_ascending"}),
            ),
            "invalid_arguments",
            "top_k",
        ),
        (
            "triager_select",
            score(json!({"kind": "score", "model": "sum", "top_k": 5, "ties": "random"})),
            "invalid_arguments",
            "record_id_ascending",
        ),
        (
            "triager_open",
            json!({"v": 1, "source": {"ref": "private", "entries": ["a".repeat(600), "b".repeat(600)]}}),
            "invalid_arguments",
            "source.entries",
        ),
        (
            "triager_open",
            json!({"v": 1, "source": {"ref": "private", "entries": ["\u{1}".repeat(400)]}}),
            "invalid_arguments",
            "source.entries",
        ),
        (
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "evidence", "record": first_login,
                   "idempotency_key": "no-job"}),
            "invalid_arguments",
            "idempotency_key",
        ),
        (
            "triager_select",
            score(json!({"kind": "filter", "syntax": "pattern", "query": "P1"})),
            "invalid_arguments",
            "producer.query",
        ),
        (
            "triager_select",
            score(
                json!({"kind": "recommend", "method": "venn", "policy": "V1",
                         "membership": "investigate"}),
            ),
            "unknown_policy",
            "V1",
        ),
        (
            "triager_select",
            score(json!({"kind": "recommend", "method": "local_model"})),
            "unsupported_operation",
            "local_model",
        ),
        (
            "triager_select",
            json!({"v": 1, "scope": "snap-0000", "producer": {"kind": "filter", "syntax": "glob", "query": "*"}}),
            "unknown_handle",
            "snap-0000",
        ),
        (
            "triager_compose",
            json!({"v": 1, "mode": "allocate", "snapshot": snapshot, "sets": {"a": handle},
                "expr": "a", "budget": {"total": 3}}),
            "unsupported_operation",
            "allocate",
        ),
        (
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": handle, "sets": {"a": handle}, "expr": "a"}),
            "wrong_handle_kind",
            "snapshot",
        ),
        (
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"a": handle},
                "expr": {"difference": ["a", "b"]}}),
            "invalid_arguments",
            "`b`",
        ),
        (
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"a": handle},
                "expr": {"difference": ["a"]}}),
            "invalid_arguments",
            "two",
        ),
        (
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"a": handle},
                "expr": {"state": {"set": "a", "is": "not_evaluated"}}}),
            "unsupported_operation",
            "state",
        ),
        (
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "records", "limit": 101}),
            "invalid_arguments",
            "100",
        ),
        (
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "histogram"}),
            "unsupported_operation",
            "histogram",
        ),
        (
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "records", "cursor": "zz"}),
            "stale_cursor",
            "cursor",
        ),
        (
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "records", "sort": ["score_desc"]}),
            "unsupported_operation",
            "score_desc",
        ),
        (
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "evidence", "record": "R1"}),
            "invalid_arguments",
            "record",
        ),
        (
            "triager_job",
            json!({"v": 1, "job": "job-none", "action": "status"}),
            "unknown_handle",
            "job-none",
        ),
        (
            "triager_job",
            json!({"v": 1, "job": "job-none", "action": "restart"}),
            "invalid_arguments",
            "action",
        ),
        (
            "triager_refine",
            json!({"v": 1, "set": handle, "action": {"kind": "field"}}),
            "unsupported_operation",
            "field",
        ),
        (
            "triager_save",
            json!({"v": 1, "handle": handle, "idempotency_key": "k"}),
            "unsupported_operation",
            "triager_save",
        ),
    ] {
        let (found, message) = host.err(tool, args.clone());
        assert_eq!(found, code, "{tool} {args}: {message}");
        assert!(message.contains(needle), "{tool} {args}: {message}");
    }
    // None of them was scheduled, read the archive or left anything behind.
    assert_eq!(host.activity(snapshot)["jobs_started"], jobs);
    assert_eq!(host.activity(snapshot)["archive_scans"], scans);

    // The entry patterns are bounded so that the result that states them fits: the
    // most that is taken opens, with the patterns stated whole.
    let patterns = vec!["*.log".to_string(), "x".repeat(1_019)];
    let opened = host.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private", "entries": patterns}}),
    );
    assert_eq!(opened["source"]["entries"], json!(patterns));

    // An excerpt must fit beside what the evidence states of the named sets: when it
    // cannot, the call is refused before the source is read.
    let mut sets = serde_json::Map::new();
    for (at, name) in ["a", "b", "c", "d", "e", "f", "g", "h"].iter().enumerate() {
        // Sixty-three operands: the largest expression, each stated in the evidence.
        let mut operands = vec![json!("s"); 62];
        operands.push(json!({"complement": [{"union": ["s", "s"]}]}));
        operands.truncate(60 - at);
        let view = host.ok(
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"s": handle},
                   "expr": {"union": operands}}),
        );
        sets.insert((*name).to_string(), view["view"].clone());
    }
    let started = host.activity(snapshot);
    let (code, message) = host.err(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "evidence", "record": first_login,
               "sets": sets, "projection": ["id", "explanation", "excerpt"]}),
    );
    assert_eq!(code, "invalid_arguments", "{message}");
    assert!(message.contains("fewer sets"), "{message}");
    let ended = host.activity(snapshot);
    assert_eq!(ended["jobs_started"], started["jobs_started"]);
    assert_eq!(ended["archive_scans"], started["archive_scans"]);
}

/// A pattern is compiled and a scorer is loaded where the budget is: at the start of
/// the job. A pattern that does not compile and a scorer that does not exist end
/// the job there, before the source is read.
#[test]
fn what_needs_the_budget_is_checked_at_the_start_of_its_job_before_the_source_is_read() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let before = host.activity(snapshot);
    for (producer, code, needle) in [
        (
            json!({"kind": "score", "model": "absent", "top_k": 5, "ties": "record_id_ascending"}),
            "missing",
            "absent",
        ),
        (
            json!({"kind": "filter", "syntax": "regex", "query": "("}),
            "invalid_arguments",
            "selection query",
        ),
    ] {
        let (found, message) = host.err(
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": producer}),
        );
        assert_eq!(found, code, "{message}");
        assert!(message.contains(needle), "{message}");
    }
    let after = host.activity(snapshot);
    assert_eq!(
        after["jobs_started"].as_u64(),
        Some(before["jobs_started"].as_u64().unwrap() + 2)
    );
    assert_eq!(after["archive_scans"], before["archive_scans"]);
}

#[test]
fn a_cursor_is_bound_to_what_it_was_issued_for() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let set = host.select_filter(snapshot, "literal", "login");
    let handle = set["set"].as_str().unwrap();

    // A cursor is bound to its handle, its membership columns and its projection.
    let page = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "records", "limit": 2}),
    );
    let cursor = page["page"]["next_cursor"].as_str().unwrap();
    for other in [
        json!({"v": 1, "handle": snapshot, "view": "records", "limit": 2, "cursor": cursor}),
        json!({"v": 1, "handle": handle, "view": "records", "limit": 2, "cursor": cursor,
               "sets": {"a": handle}}),
        json!({"v": 1, "handle": handle, "view": "records", "limit": 2, "cursor": cursor,
               "projection": ["id"]}),
    ] {
        assert_eq!(host.err("triager_inspect", other).0, "stale_cursor");
    }
    host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "records", "limit": 5, "cursor": cursor}),
    );
}

#[test]
fn handles_resolve_only_in_their_project_and_snapshot() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let first = host.open("private");
    let snapshot = first["snapshot"].as_str().unwrap();
    let set = host.select_filter(snapshot, "literal", "login");

    // The same bytes at another place are another snapshot, and their sets do not mix.
    let other = host.ok(
        "triager_open",
        json!({"v": 1, "source": {"path": host.dir.path().join("allowed").join("mine.zip"),
                                  "entries": ["*.log"]}}),
    );
    let other_snapshot = other["snapshot"].as_str().unwrap();
    assert_ne!(other_snapshot, snapshot);
    let other_set = host.select_filter(other_snapshot, "literal", "login");
    let (code, message) = host.err(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"a": set["set"], "b": other_set["set"]},
               "expr": {"union": ["a", "b"]}}),
    );
    assert_eq!(code, "mixed_snapshots", "{message}");

    // Another entry filter is another snapshot too.
    let all = host.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private"}}),
    );
    assert_ne!(all["snapshot"], first["snapshot"]);
    assert_eq!(all["completeness"]["entries_excluded"], 0);

    // A path outside the allowed roots is refused, whatever it holds.
    let (code, _) = host.err(
        "triager_open",
        json!({"v": 1, "source": {"path": host.dir.path().join("private.zip")}}),
    );
    assert_eq!(code, "path_not_allowed");

    // The same archive under another project gives other handles.
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        host.dir.path().join("private.zip"),
        dir.path().join("private.zip"),
    )
    .unwrap();
    std::fs::copy(
        host.dir.path().join("public.zip"),
        dir.path().join("public.zip"),
    )
    .unwrap();
    let second = support::start(dir.path(), "project-two", &HostOptions::default()).unwrap();
    let reply = second.call(
        "triager_inspect",
        &json!({"v": 1, "handle": snapshot, "view": "overview"}),
    );
    assert!(reply.is_error);
    assert!(reply.json.contains("unknown_handle"), "{}", reply.json);
}

#[test]
fn rows_bytes_and_excerpts_are_bounded_and_clipping_is_reported() {
    let mut entries = auth_entries();
    // A long record with terminal controls, and many originals with long entry names.
    let long = format!("\u{1b}[31mlogin \u{202e}{}\u{7}", "é".repeat(3000));
    let wide: Vec<String> = (0..400)
        .map(|index| format!("login probe {index}"))
        .collect();
    entries.insert(0, (format!("{}/wide.log", "d".repeat(200)), wide));
    entries[1].1.push(long.clone());
    let universe = oracle(&entries, ".log");
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let set = host.select_filter(snapshot, "glob", "*login*");
    let handle = set["set"].as_str().unwrap();
    let total = count(&set, "counts", "in");
    assert!(total > 400);

    // The default is 25 rows; the most is 100, cut earlier where the bytes run out.
    let page = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "records"}),
    );
    assert_eq!(page["rows"].as_array().unwrap().len(), 25);
    assert_eq!(page["applied_limits"]["max_rows"], MAX_ROWS as u64);
    assert_eq!(
        page["applied_limits"]["max_result_bytes"],
        MAX_RESULT_BYTES as u64
    );
    let reply = host.reply(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "records", "limit": 100,
               "sets": {"b": handle}}),
    );
    assert!(transport_bytes(&reply.json) <= MAX_RESULT_BYTES);
    let page: Value = serde_json::from_str(&reply.json).unwrap();
    let returned = page["rows"].as_array().unwrap().len();
    assert!(returned < 100, "the long entry names do not fit 100 rows");
    assert_eq!(page["clipping"]["rows_by_bytes"], true);
    assert_eq!(page["clipping"]["clipped"], true);
    // The total is the set's, never the page's.
    assert_eq!(page["page"]["total"]["distinct_originals"], total);
    assert!(page["page"]["next_cursor"].is_string());
    let every = host.all_records(handle, json!({"b": handle}), 100);
    assert_eq!(every.len() as u64, total);

    // A long query is kept whole and shown in part, named by its length and hash.
    let query = format!("(login){}", "|(a text that occurs nowhere)".repeat(200));
    let same = host.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "filter", "syntax": "regex", "query": query}}),
    );
    assert_eq!(count(&same, "counts", "in"), total);
    assert_eq!(same["revision"], set["revision"]);
    assert_eq!(same["producer"]["query_clipped"], true);
    assert_eq!(same["producer"]["query_bytes"], query.len() as u64);
    assert_eq!(same["producer"]["query"].as_str().unwrap().len(), 512);
    assert_eq!(same["producer"]["query_sha256"].as_str().unwrap().len(), 64);

    // The excerpt is cut on a character boundary, with the controls escaped.
    let original = universe
        .iter()
        .find(|original| original.text == long)
        .unwrap();
    let evidence = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "evidence", "record": original.id(),
               "projection": ["id", "excerpt"]}),
    );
    let text = evidence["excerpt"]["text"].as_str().unwrap();
    assert!(text.len() <= MAX_EXCERPT_BYTES, "{} bytes", text.len());
    assert!(text.starts_with("\\u{1b}[31mlogin \\u{202e}é"), "{text}");
    assert!(!text.contains('\u{1b}') && !text.contains('\u{202e}'));
    assert_eq!(evidence["excerpt"]["clipped"], true);
    assert_eq!(evidence["excerpt"]["source_bytes"], long.len() as u64);
    assert_eq!(evidence["clipping"]["clipped"], true);
    assert!(evidence.get("membership").is_none(), "{evidence}");
}

#[test]
fn disclosure_follows_the_host_profile_and_the_source_class() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let host = Host::new(
        &entries,
        HostOptions {
            profile: DisclosureProfile::PublicDevelopment,
            evidence_allowance_bytes: 40,
            ..HostOptions::default()
        },
    );
    // Private evidence does not reach this profile: neither a registered private
    // source nor a path the client names.
    for source in [
        json!({"ref": "private"}),
        json!({"path": host.dir.path().join("allowed").join("mine.zip")}),
    ] {
        let (code, message) = host.err("triager_open", json!({"v": 1, "source": source}));
        assert_eq!(code, "disclosure_refused", "{message}");
    }
    let opened = host.open("public-fixture");
    assert_eq!(opened["disclosure"]["profile"], "public_development");
    assert_eq!(opened["disclosure"]["source_class"], "public_fixture");
    assert_eq!(opened["disclosure"]["remaining_evidence_bytes"], 40);
    let snapshot = opened["snapshot"].as_str().unwrap();

    // Excerpts count against the session's allowance, which is shown and enforced.
    let first = &universe[0];
    let evidence = |record: &Orig| {
        json!({"v": 1, "handle": snapshot, "view": "evidence", "record": record.id(),
               "projection": ["id", "excerpt"]})
    };
    let shown = host.ok("triager_inspect", evidence(first));
    assert_eq!(shown["excerpt"]["text"], first.text.as_str());
    assert_eq!(
        shown["disclosure"]["excerpt_bytes"],
        first.text.len() as u64
    );
    assert_eq!(
        shown["disclosure"]["remaining_evidence_bytes"],
        40 - first.text.len() as u64
    );
    let second = host.ok("triager_inspect", evidence(&universe[1]));
    assert_eq!(second["excerpt"]["clipped"], true);
    assert_eq!(second["disclosure"]["remaining_evidence_bytes"], 0);
    let (code, message) = host.err("triager_inspect", evidence(&universe[2]));
    assert_eq!(code, "disclosure_refused", "{message}");
    // Counts and handles stay available.
    host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "evidence", "record": universe[2].id()}),
    );

    // The private profile shows excerpts of a private source without an allowance.
    let local = Host::new(&entries, HostOptions::default());
    let opened = local.open("private");
    assert_eq!(opened["disclosure"]["profile"], "private_local");
    assert_eq!(opened["disclosure"]["source_class"], "private");
    assert_eq!(
        opened["disclosure"]["remaining_evidence_bytes"],
        Value::Null
    );
    let shown = local.ok(
        "triager_inspect",
        json!({"v": 1, "handle": opened["snapshot"], "view": "evidence",
               "record": first.id(), "projection": ["excerpt"]}),
    );
    assert_eq!(shown["excerpt"]["text"], first.text.as_str());
}

#[test]
fn a_replayed_request_returns_its_original_result_without_a_second_run() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let open = json!({"v": 1, "source": {"ref": "private", "entries": ["*.log"]},
                      "idempotency_key": "open-1"});
    let opened = host.reply("triager_open", open.clone());
    let again = host.reply("triager_open", open);
    assert_eq!(opened.json, again.json);
    let snapshot = serde_json::from_str::<Value>(&opened.json).unwrap()["snapshot"]
        .as_str()
        .unwrap()
        .to_string();

    let select = json!({"v": 1, "scope": snapshot, "idempotency_key": "a-1", "producer": {
        "kind": "score", "model": "sum", "top_k": 40, "ties": "record_id_ascending"}});
    let first = host.reply("triager_select", select.clone());
    assert!(!first.is_error, "{}", first.json);
    let scans = host.activity(&snapshot)["archive_scans"].as_u64().unwrap();
    let jobs = host.activity(&snapshot)["jobs_started"].as_u64().unwrap();
    let replay = host.reply("triager_select", select.clone());
    assert_eq!(replay.json, first.json);
    assert_eq!(host.activity(&snapshot)["archive_scans"], scans);
    assert_eq!(host.activity(&snapshot)["jobs_started"], jobs);

    // The key with another payload conflicts, and nothing runs.
    let mut other = select.clone();
    other["producer"]["top_k"] = json!(41);
    let (code, message) = host.err("triager_select", other);
    assert_eq!(code, "idempotency_conflict", "{message}");
    assert_eq!(host.activity(&snapshot)["archive_scans"], scans);

    // Without a key the committed set is found by its identity: no second scan.
    let mut keyless = select;
    keyless.as_object_mut().unwrap().remove("idempotency_key");
    let found = host.ok("triager_select", keyless);
    let first: Value = serde_json::from_str(&first.json).unwrap();
    assert_eq!(found["set"], first["set"]);
    assert_eq!(found["revision"], first["revision"]);
    assert_eq!(host.activity(&snapshot)["archive_scans"], scans);

    // An excerpt is read by a job, so it takes a key too: the replay returns the
    // result that was read and reads nothing again.
    let universe = oracle(&entries, ".log");
    let excerpt = json!({"v": 1, "handle": snapshot, "view": "evidence",
        "record": universe[3].id(), "projection": ["id", "excerpt"], "idempotency_key": "row-3"});
    let shown = host.reply("triager_inspect", excerpt.clone());
    assert!(!shown.is_error, "{}", shown.json);
    assert_eq!(host.activity(&snapshot)["archive_scans"], scans + 1);
    assert_eq!(
        host.reply("triager_inspect", excerpt.clone()).json,
        shown.json
    );
    assert_eq!(host.activity(&snapshot)["archive_scans"], scans + 1);
    let mut other = excerpt;
    other["record"] = json!(universe[4].id());
    assert_eq!(host.err("triager_inspect", other).0, "idempotency_conflict");
}

/// Many distinct lines in three entries, so that deduplication goes on on disk.
fn spill_entries() -> Vec<(String, Vec<String>)> {
    (0..3)
        .map(|entry| {
            let lines = (0..20_000)
                .map(|line| format!("event {entry} login step {line} of the long run"))
                .collect();
            (format!("run/part-{entry}.log"), lines)
        })
        .collect()
}

#[test]
fn a_job_interrupted_during_spill_leaves_committed_revisions_usable() {
    let entries = spill_entries();
    let host = Host::new(
        &entries,
        HostOptions {
            inline_wait: Some(Duration::ZERO),
            batch_rows: Some(1_024),
            dedup_set_texts: Some(2_000),
            delay: Duration::from_millis(4),
            ..HostOptions::default()
        },
    );
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    assert_eq!(opened["counts"]["distinct_originals"], 60_000);
    assert_eq!(opened["cost"]["dedup_on_disk"], true);
    let committed = host.select_filter(snapshot, "glob", "*step 1999 of*");
    let handle = committed["set"].as_str().unwrap();
    assert_eq!(count(&committed, "counts", "in"), 3);

    // A long score run is started, seen spilling, and cancelled.
    let select = json!({"v": 1, "scope": snapshot, "idempotency_key": "long-a", "producer": {
        "kind": "score", "model": "sum", "top_k": 500, "ties": "record_id_ascending"}});
    let started: Value =
        serde_json::from_str(&host.reply("triager_select", select.clone()).json).unwrap();
    assert_eq!(started["status"], "running", "{started}");
    let job = started["job"].as_str().unwrap();
    // The same key and payload names the job in progress.
    let same: Value =
        serde_json::from_str(&host.reply("triager_select", select.clone()).json).unwrap();
    assert_eq!(same["job"], job);
    let begun = std::time::Instant::now();
    loop {
        let status = host.job(job, "status");
        // Other tests of this binary may hold the host for a moment.
        assert!(
            matches!(status["status"].as_str(), Some("waiting" | "running")),
            "{status}"
        );
        assert_eq!(status["tool"], "triager_select");
        if status["progress"]["spilled"] == true && status["progress"]["batches"].as_u64() > Some(2)
        {
            break;
        }
        assert!(begun.elapsed() < Duration::from_secs(60), "{status}");
        std::thread::sleep(Duration::from_millis(1));
    }
    let cancel = host.job(job, "cancel");
    assert!(
        matches!(
            cancel["status"].as_str().unwrap(),
            "cancel_requested" | "cancelled"
        ),
        "{cancel}"
    );
    let ended = host.wait_status(job);
    assert_eq!(ended["status"], "cancelled", "{ended}");
    assert!(ended.get("result").is_none(), "no partial set: {ended}");
    assert_eq!(ended["progress"]["spilled"], true);

    // No partial final set, nothing left in the work directory beside the snapshot.
    let left: Vec<String> = std::fs::read_dir(host.work())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        left.iter().all(|name| name.starts_with("snapshot-")),
        "{left:?}"
    );
    // The committed revisions answer as before.
    let rows = host.all_records(handle, json!({}), 25);
    assert_eq!(rows.len(), 3);
    let overview = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": handle, "view": "overview"}),
    );
    assert_eq!(overview["artifact"]["revision"], committed["revision"]);
    // Cancelling again reports the state it ended in.
    assert_eq!(host.job(job, "cancel")["status"], "cancelled");

    // The key of the cancelled job starts the run again, which now completes.
    let rerun: Value = serde_json::from_str(&host.reply("triager_select", select).json).unwrap();
    let rerun_job = rerun["job"].as_str().unwrap();
    assert_ne!(rerun_job, job);
    let done = host.wait(rerun_job);
    assert_eq!(count(&done, "counts", "in"), 500);
    assert_eq!(done["cost"]["dedup_on_disk"], true);
    // A cancel that arrives after completion reports the committed result.
    let late = host.job(rerun_job, "cancel");
    assert_eq!(late["status"], "completed");
    assert_eq!(late["result"]["set"], done["set"]);
}

#[test]
fn a_job_that_ends_abnormally_is_interrupted_and_commits_nothing() {
    let entries = auth_entries();
    let host = Host::new(
        &entries,
        HostOptions {
            inline_wait: Some(Duration::ZERO),
            ..HostOptions::default()
        },
    );
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let committed = host.select_top(snapshot, 10);
    let started: Value = serde_json::from_str(
        &host
            .reply(
                "triager_select",
                json!({"v": 1, "scope": snapshot, "producer": {
                    "kind": "score", "model": "boom", "top_k": 10, "ties": "record_id_ascending"}}),
            )
            .json,
    )
    .unwrap();
    let ended = host.wait_status(started["job"].as_str().unwrap());
    assert_eq!(ended["status"], "interrupted", "{ended}");
    assert_eq!(ended["error"]["code"], "interrupted");
    assert!(ended.get("result").is_none());
    // The worker goes on, the budget holds what it held, and nothing is left behind.
    let rows = host.all_records(committed["set"].as_str().unwrap(), json!({}), 25);
    assert_eq!(rows.len(), 10);
    let again = host.select_top(snapshot, 11);
    assert_eq!(count(&again, "counts", "in"), 11);
    let left: Vec<String> = std::fs::read_dir(host.work())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        left.iter().all(|name| name.starts_with("snapshot-")),
        "{left:?}"
    );
}

#[test]
fn a_changed_archive_is_refused_and_the_snapshot_keeps_its_counts() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap();
    let mut changed = entries.clone();
    changed[0].1.push("login from a new place".into());
    support::write_archive(&host.dir.path().join("private.zip"), &changed);
    let (code, message) = host.err(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "filter", "syntax": "glob", "query": "*login*"}}),
    );
    assert_eq!(code, "refused");
    assert!(message.contains("changed"), "{message}");
    let overview = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "overview"}),
    );
    assert_eq!(overview["artifact"]["counts"], opened["counts"]);
}

#[test]
fn a_busy_host_refuses_with_the_holder_or_waits() {
    let entries = auth_entries();
    let host = Host::new(
        &entries,
        HostOptions {
            busy: BusyChoice::Refuse,
            ..HostOptions::default()
        },
    );
    let (release, held) = std::sync::mpsc::channel::<()>();
    let (ready, is_ready) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let guard = ResourceGuard::enter("a test that holds the host", BusyChoice::Wait).unwrap();
        ready.send(()).unwrap();
        held.recv().unwrap();
        drop(guard);
    });
    is_ready.recv().unwrap();
    let (code, message) = host.err(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private"}}),
    );
    release.send(()).unwrap();
    holder.join().unwrap();
    assert_eq!(code, "refused");
    assert!(message.contains("a test that holds the host"), "{message}");
    // Once the host is free the same request is admitted. Another test of this
    // binary may hold the host meanwhile: that is refused with its holder too.
    let started = std::time::Instant::now();
    loop {
        let reply = host.reply(
            "triager_open",
            json!({"v": 1, "source": {"ref": "private"}}),
        );
        if !reply.is_error {
            break;
        }
        assert!(reply.json.contains("\"refused\""), "{}", reply.json);
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "{}",
            reply.json
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
