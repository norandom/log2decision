//! Triager task 3.2, requirements 4.1, 5.1, 5.3, 5.4 and 6.3: the terminal workbench
//! over the service. Driven by keys in-process, it makes the tool calls an MCP client
//! makes and gets the same counts, ordered IDs and receipts; drawing at any size
//! makes no call; a running job is cancelled with Esc and commits nothing.

mod support;

use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use l2d_triager::bench::{Start, Workbench};
use l2d_triager::mcp;
use l2d_triager::service::Service;
use serde_json::{Value, json};
use support::{HostOptions, Orig, auth_entries, membership_line, oracle, start, write_archive};

fn c_file(universe: &[Orig]) -> String {
    universe
        .iter()
        .enumerate()
        .filter_map(|(index, original)| {
            let state = match index % 4 {
                0 => "in",
                1 => "out",
                2 => "abstain",
                _ => return None,
            };
            Some(membership_line(&original.id(), state))
        })
        .collect()
}

fn options(universe: &[Orig]) -> HostOptions {
    HostOptions {
        memberships: vec![("c".into(), c_file(universe))],
        inline_wait: Some(Duration::from_millis(20)),
        ..HostOptions::default()
    }
}

const RECIPE: &str = r#"{"tool":"triager_open","args":{"v":1,"source":{"ref":"private","entries":["*.log"]},"projections":{"fields":[{"name":"user","from":"kv"}]}},"bind":{"S":"snapshot"}}
{"tool":"triager_select","args":{"v":1,"scope":"$S","producer":{"kind":"score","model":"sum","top_k":40,"ties":"record_id_ascending"}},"bind":{"A":"set"}}
{"tool":"triager_select","args":{"v":1,"scope":"$S","producer":{"kind":"filter","syntax":"glob","query":"*login*"}},"bind":{"B":"set"}}
{"tool":"triager_select","args":{"v":1,"scope":"$S","producer":{"kind":"membership","ref":"c"}},"bind":{"C":"set"}}
"#;

/// Calls the service as an MCP client does and waits for a job to end.
fn call(service: &Service, tool: &str, args: Value) -> Value {
    let reply = mcp::settle(service, service.call(tool, &args));
    let value: Value = serde_json::from_str(&reply.json).unwrap();
    assert!(!reply.is_error, "{tool} {args}: {value}");
    value
}

fn press(bench: &mut Workbench, code: KeyCode) {
    assert!(!bench.press(KeyEvent::new(code, KeyModifiers::NONE)));
}

fn type_text(bench: &mut Workbench, text: &str) {
    for ch in text.chars() {
        press(bench, KeyCode::Char(ch));
    }
}

/// Every ID of a view, page by page, as an MCP client reads them.
fn ids(service: &Service, handle: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let mut args = json!({"v": 1, "handle": handle, "view": "records", "limit": 100});
        if let Some(cursor) = &cursor {
            args["cursor"] = json!(cursor);
        }
        let page = call(service, "triager_inspect", args);
        found.extend(
            page["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["id"].as_str().unwrap().to_string()),
        );
        match page["page"]["next_cursor"].as_str() {
            Some(next) => cursor = Some(next.to_string()),
            None => return found,
        }
    }
}

fn work(service: &Service, snapshot: &str) -> (Value, u64) {
    let overview = call(
        service,
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "overview"}),
    );
    (
        overview["activity"].clone(),
        overview["session"]["calls"].as_u64().unwrap(),
    )
}

fn opened(dir: &std::path::Path, options: &HostOptions) -> (Service, Vec<(String, Value)>) {
    let service = start(dir, "project-one", options).unwrap();
    let bound = mcp::run_requests_bound(&service, RECIPE.as_bytes(), std::io::sink()).unwrap();
    (service, bound)
}

#[test]
fn the_keyboard_walk_makes_the_calls_of_an_mcp_client_and_drawing_makes_none() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = options(&universe);
    let dir = tempfile::tempdir().unwrap();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    let (service, bound) = opened(dir.path(), &options);
    let get = |name: &str| {
        bound
            .iter()
            .find(|(bound, _)| bound == name)
            .unwrap()
            .1
            .as_str()
            .unwrap()
            .to_string()
    };
    let (s, a, b, c) = (get("S"), get("A"), get("B"), get("C"));
    let start_at = Start::from_bindings(&bound).unwrap();
    assert_eq!(start_at.named.len(), 3);
    let trace = dir.path().join("trace.jsonl");
    let mut bench = Workbench::new(service, start_at, Some(&trace)).unwrap();
    bench.settle();
    assert_eq!(bench.active_handle(), s);
    let root = bench.render_text(120, 32);
    assert!(
        root.contains(&format!("{} originals", universe.len())),
        "{root}"
    );

    // A − B by key, against the same composition made as an MCP client.
    press(&mut bench, KeyCode::Char('1'));
    bench.settle();
    let d = bench.active_handle().to_string();
    let direct = call(
        bench.service(),
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": s, "sets": {"a": a, "b": b},
               "expr": {"intersection": [{"state": {"set": "a", "is": "in"}},
                                         {"state": {"set": "b", "is": "out"}}]}}),
    );
    assert_eq!(direct["view"], d, "the key composes the same view");
    let difference = call(
        bench.service(),
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": s, "sets": {"a": a, "b": b},
               "expr": {"difference": ["a", "b"]}}),
    );
    assert_eq!(direct["counts"], difference["counts"]);
    let counts = &direct["counts"];
    let screen = bench.render_text(120, 32);
    let header = format!(
        "{}/{} originals",
        counts["in"].as_u64().unwrap(),
        counts["parent"].as_u64().unwrap()
    );
    assert!(screen.contains(&header), "{header} in\n{screen}");
    let page = bench.summary();
    let shown: Vec<&str> = page["page"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    let all = ids(bench.service(), &d);
    assert_eq!(shown, all[..shown.len()]);

    // Drawing at other sizes, resizing the panes and redrawing make no call.
    let (before, calls_before) = work(bench.service(), &s);
    for (width, height) in [(80, 24), (200, 60), (60, 18), (120, 32)] {
        bench.render_text(width, height);
    }
    for key in [']', ']', '[', '+', '+', '-'] {
        press(&mut bench, KeyCode::Char(key));
        bench.render_text(80, 24);
    }
    assert!(!bench.busy());
    let (after, calls_after) = work(bench.service(), &s);
    assert_eq!(after, before, "work counters after resizing");
    assert_eq!(
        calls_after,
        calls_before + 1,
        "only the second overview was called"
    );

    // The siblings inside D: C in, then back and C not evaluated.
    press(&mut bench, KeyCode::Char('i'));
    bench.settle();
    let recommended = bench.active_handle().to_string();
    assert_ne!(recommended, d);
    press(&mut bench, KeyCode::Char('b'));
    assert_eq!(bench.active_handle(), d, "back returns to D");
    press(&mut bench, KeyCode::Char('u'));
    bench.settle();
    let unknown = bench.summary();
    let direct_unknown = call(
        bench.service(),
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": s, "scope": d, "sets": {"c": c},
               "expr": {"state": {"set": "c", "is": "not_evaluated"}}}),
    );
    assert_eq!(unknown["view"]["handle"], direct_unknown["view"]);
    press(&mut bench, KeyCode::Char('b'));

    // A search inside D: a filter selection scoped to D, then its overlap with D.
    press(&mut bench, KeyCode::Char('/'));
    type_text(&mut bench, "*failed*");
    press(&mut bench, KeyCode::Enter);
    bench.settle();
    let searched = bench.active_handle().to_string();
    let selection = call(
        bench.service(),
        "triager_select",
        json!({"v": 1, "scope": d, "producer": {"kind": "filter", "syntax": "glob",
               "query": "*failed*"}}),
    );
    let overlap = call(
        bench.service(),
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": s, "scope": d,
               "sets": {"p": d, "s": selection["set"]}, "expr": {"intersection": ["p", "s"]}}),
    );
    assert_eq!(overlap["view"], searched);
    assert!(
        ids(bench.service(), &searched)
            .iter()
            .all(|id| all.contains(id))
    );
    press(&mut bench, KeyCode::Char('b'));

    // A field drill from the sidebar: the values of `user`, then one value.
    press(&mut bench, KeyCode::Tab);
    press(&mut bench, KeyCode::Tab);
    let fields = bench.render_text(120, 32);
    assert!(fields.contains("user"), "{fields}");
    // The built-in `source` comes first, then the declared `user`.
    press(&mut bench, KeyCode::Down);
    press(&mut bench, KeyCode::Enter);
    bench.settle();
    let menu = bench.render_text(120, 32);
    assert!(
        menu.contains("user · ") && menu.contains("values by occurrences"),
        "{menu}"
    );
    press(&mut bench, KeyCode::Enter);
    bench.settle();
    let drilled = bench.summary();
    assert_eq!(drilled["view"]["kind"], "view", "{drilled}");
    let title = drilled["view"]["title"].as_str().unwrap();
    let action = match title.strip_prefix("user = ") {
        Some(value) => json!({"kind": "field", "field": "user", "value": value}),
        None => {
            assert_eq!(title, "user missing", "{drilled}");
            json!({"kind": "field", "field": "user", "missing": true})
        }
    };
    let refined = call(
        bench.service(),
        "triager_refine",
        json!({"v": 1, "set": d, "action": action}),
    );
    assert_eq!(drilled["view"]["handle"], refined["view"], "{refined}");
    assert_eq!(
        drilled["members"]["distinct_originals"],
        refined["counts"]["in"]
    );
    press(&mut bench, KeyCode::Char('b'));
    press(&mut bench, KeyCode::Tab);

    // Evidence of the first original of D, then its excerpt from the source.
    press(&mut bench, KeyCode::Enter);
    bench.settle();
    let evidence = bench.render_text(120, 40);
    assert!(evidence.contains("Why it is here"), "{evidence}");
    let (scans_before, _) = work(bench.service(), &s);
    press(&mut bench, KeyCode::Char('x'));
    bench.settle();
    let excerpt = bench.render_text(120, 40);
    assert!(excerpt.contains("Excerpt ("), "{excerpt}");
    let (scans_after, _) = work(bench.service(), &s);
    assert_eq!(
        scans_after["archive_scans"].as_u64().unwrap(),
        scans_before["archive_scans"].as_u64().unwrap() + 1,
        "an excerpt is one read of the source"
    );
    press(&mut bench, KeyCode::Esc);

    // Allocate once from D: the ordered IDs of the same allocation as an MCP client
    // (requirement 5.4).
    press(&mut bench, KeyCode::Char('f'));
    for _ in 0..8 {
        press(&mut bench, KeyCode::Backspace);
    }
    type_text(&mut bench, "3 2 2 7");
    press(&mut bench, KeyCode::Enter);
    bench.settle();
    let terminal_final = bench.active_handle().to_string();
    assert!(terminal_final.starts_with("final-"), "{terminal_final}");
    let mcp_final = call(
        bench.service(),
        "triager_compose",
        json!({"v": 1, "mode": "allocate", "snapshot": s, "sets": {"d": d, "c": c},
               "expr": {"intersection": ["d", "c"]}, "budget": {"total": 7,
               "duplicate_rule": "first_pool_wins", "shortfall": "leave_unfilled", "pools": [
               {"name": "general", "limit": 3, "from": {"intersection": ["d", "c"]},
                "order": ["score_desc", "record_id_ascending"]},
               {"name": "unfamiliar", "limit": 2, "from": {"intersection": ["d",
                 {"state": {"set": "c", "is": "not_evaluated"}}]}, "order": ["record_id_ascending"]},
               {"name": "audit", "limit": 2, "from": {"difference": ["d", "c"]},
                "order": ["seeded_hash", "record_id_ascending"], "seed": 7}]}}),
    );
    assert_eq!(mcp_final["set"], terminal_final);
    let ordered = ids(bench.service(), &terminal_final);
    assert_eq!(ordered.len(), 7);
    let final_screen = bench.render_text(120, 32);
    assert!(final_screen.contains("audit #"), "{final_screen}");
    press(&mut bench, KeyCode::Char('f'));
    assert!(
        bench.summary()["notice"]
            .as_str()
            .unwrap()
            .contains("Already allocated")
    );

    // Save, then reopen the receipt in a second server: the same ordered IDs.
    press(&mut bench, KeyCode::Char('s'));
    let saved = bench.summary();
    let notice = saved["notice"].as_str().unwrap();
    let receipt = notice
        .split_whitespace()
        .find(|word| word.starts_with("rcpt-"))
        .unwrap()
        .to_string();
    let mut second_options = options.clone();
    second_options.inline_wait = Some(Duration::from_millis(20));
    let work_dir = tempfile::tempdir().unwrap();
    let mut config = support::config(dir.path(), "project-one", &second_options);
    config.work_root = work_dir.path().join("work");
    let second = Service::start(
        config,
        support::budget(&second_options),
        Box::new(support::TestScorers {
            delay: Duration::ZERO,
        }),
    )
    .unwrap();
    let reopened = call(&second, "triager_open", json!({"v": 1, "receipt": receipt}));
    let mut there = Workbench::new(second, Start::from_reopened(&reopened).unwrap(), None).unwrap();
    let summary = there.summary();
    assert_eq!(summary["view"]["kind"], "final");
    let reopened_ids: Vec<String> = summary["page"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(reopened_ids, ordered);
    let (activity, _) = work(there.service(), reopened["snapshot"].as_str().unwrap());
    assert_eq!(activity["archive_scans"], 0);
    assert_eq!(activity["rows_scored"], 0);
    // The excerpt of a reopened record comes from the registered source.
    press(&mut there, KeyCode::Enter);
    there.settle();
    press(&mut there, KeyCode::Char('x'));
    there.settle();
    assert!(there.render_text(120, 40).contains("Excerpt ("));

    // The trace keeps layout and drawing apart from the calls.
    let trace = std::fs::read_to_string(&trace).unwrap();
    let events: Vec<Value> = trace
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(events.iter().any(|event| event["event"] == "call"));
    assert!(events.iter().any(|event| event["event"] == "key"));
}

/// Lines with a leading UTC timestamp, a host and a user, repeated across two entries,
/// so the projection has fields and time buckets.
fn timed_entries() -> Vec<(String, Vec<String>)> {
    let mut first = Vec::new();
    let mut second = Vec::new();
    for index in 0..300u32 {
        let line = format!(
            "2026-01-01T{:02}:{:02}:00Z host=h{} user=u{:02} {}",
            index % 6,
            index % 60,
            index % 4,
            index % 25,
            if index % 3 == 0 {
                "login failed"
            } else {
                "page served"
            }
        );
        if index % 2 == 0 {
            first.push(line);
        } else {
            second.push(line.clone());
            second.push(line);
        }
    }
    vec![("app/a.log".into(), first), ("app/b.log".into(), second)]
}

const TIMED_RECIPE: &str = r#"{"tool":"triager_open","args":{"v":1,"source":{"ref":"private","entries":["*.log"]},"projections":{"fields":[{"name":"host","from":"kv"},{"name":"user","from":"kv"}],"time":{"from":"text"}}},"bind":{"S":"snapshot"}}
{"tool":"triager_select","args":{"v":1,"scope":"$S","producer":{"kind":"score","model":"sum","top_k":60,"ties":"record_id_ascending"}},"bind":{"A":"set"}}
{"tool":"triager_select","args":{"v":1,"scope":"$S","producer":{"kind":"filter","syntax":"glob","query":"*login*"}},"bind":{"B":"set"}}
"#;

fn timed(options: &HostOptions) -> (tempfile::TempDir, Service, Vec<(String, Value)>) {
    let entries = timed_entries();
    let dir = tempfile::tempdir().unwrap();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    std::fs::create_dir_all(dir.path().join("allowed")).unwrap();
    write_archive(&dir.path().join("allowed").join("mine.zip"), &entries);
    let service = start(dir.path(), "project-one", options).unwrap();
    let bound =
        mcp::run_requests_bound(&service, TIMED_RECIPE.as_bytes(), std::io::sink()).unwrap();
    (dir, service, bound)
}

/// `start` (`2026-01-01THH:MM:SSZ`, within the fixture's one day) plus `seconds`.
fn plus_seconds(start: &str, seconds: u64) -> String {
    let at = |from: usize| start[from..from + 2].parse::<u64>().unwrap();
    let total = at(11) * 3600 + at(14) * 60 + at(17) + seconds;
    format!(
        "2026-01-01T{:02}:{:02}:{:02}Z",
        total / 3600,
        total % 3600 / 60,
        total % 60
    )
}

#[test]
fn a_time_bucket_drill_is_the_refine_an_mcp_client_makes() {
    let options = HostOptions {
        inline_wait: Some(Duration::from_millis(20)),
        ..HostOptions::default()
    };
    let (_dir, service, bound) = timed(&options);
    let mut bench = Workbench::new(service, Start::from_bindings(&bound).unwrap(), None).unwrap();
    bench.settle();
    press(&mut bench, KeyCode::Char('1'));
    bench.settle();
    let d = bench.active_handle().to_string();
    let screen = bench.render_text(120, 36);
    let histogram = call(
        bench.service(),
        "triager_inspect",
        json!({"v": 1, "handle": d, "view": "histogram"}),
    );
    let first = &histogram["time"]["buckets"][0];
    let width = histogram["time"]["bucket_seconds"].as_u64().unwrap();
    assert!(
        screen.contains(&format!("Time UTC · {width} s buckets")),
        "{screen}"
    );
    press(&mut bench, KeyCode::Char('t'));
    let menu = bench.render_text(120, 36);
    assert!(
        menu.contains(&format!("Time buckets of {width} s (UTC)")),
        "{menu}"
    );
    press(&mut bench, KeyCode::Enter);
    bench.settle();
    let start_text = first["start"].as_str().unwrap();
    let refined = call(
        bench.service(),
        "triager_refine",
        json!({"v": 1, "set": d, "action": {"kind": "time", "start": start_text,
               "end": plus_seconds(start_text, width)}}),
    );
    assert_eq!(bench.active_handle(), refined["view"], "{refined}");
    let summary = bench.summary();
    assert_eq!(
        summary["members"]["distinct_originals"],
        refined["counts"]["in"]
    );
    assert_eq!(
        summary["members"]["distinct_originals"], first["distinct_originals"],
        "a bucket's originals are the drilled view's"
    );
    assert!(
        bench.render_text(120, 36).contains("matching occ."),
        "the drilled view states its matching occurrences apart"
    );
}

#[test]
fn a_cancelled_or_dropped_pane_loads_again() {
    let options = HostOptions {
        inline_wait: Some(Duration::from_millis(20)),
        // Every scored batch takes this long: the slow job of another client.
        delay: Duration::from_millis(1500),
        ..HostOptions::default()
    };
    let (_dir, service, bound) = timed(&options);
    let mut bench = Workbench::new(service, Start::from_bindings(&bound).unwrap(), None).unwrap();
    bench.settle();
    let other = call(
        bench.service(),
        "triager_open",
        json!({"v": 1, "source": {"ref": "public-fixture", "entries": ["*.log"]}}),
    );
    // A − B opens; its records and regions are inline reads, its fields and histogram
    // jobs. Another client then holds the worker, so the fields job waits.
    press(&mut bench, KeyCode::Char('1'));
    let root = bench.active_handle().to_string();
    while bench.active_handle() == root {
        bench.tick();
    }
    let slow = bench.service().call(
        "triager_select",
        &json!({"v": 1, "scope": other["snapshot"], "producer": {"kind": "score",
               "model": "sum", "top_k": 5, "ties": "record_id_ascending"}}),
    );
    assert!(slow.json.contains("\"running\""), "{}", slow.json);
    for _ in 0..20 {
        if bench.render_text(120, 36).contains("Running fields") {
            break;
        }
        bench.tick();
    }
    assert!(bench.render_text(120, 36).contains("Running fields"));
    press(&mut bench, KeyCode::Esc);
    bench.settle();
    let cancelled = bench.render_text(120, 36);
    assert!(cancelled.contains("Cancelled fields"), "{cancelled}");
    assert!(
        cancelled.contains("cancelled · r loads it"),
        "a cancelled pane says so:\n{cancelled}"
    );
    // Nothing of the view is stuck: `r` loads what was cancelled or dropped.
    press(&mut bench, KeyCode::Char('r'));
    bench.settle();
    let screen = bench.render_text(120, 36);
    assert!(!screen.contains("loading…"), "{screen}");
    assert!(screen.contains(" s buckets"), "{screen}");
    assert!(screen.contains("host"), "{screen}");
}

/// Starts a slow job of another client on the public snapshot, so jobs of the
/// workbench wait behind it.
fn hold_the_worker(bench: &Workbench, source: Value) -> Value {
    let other = call(
        bench.service(),
        "triager_open",
        json!({"v": 1, "source": source}),
    );
    let slow = bench.service().call(
        "triager_select",
        &json!({"v": 1, "scope": other["snapshot"], "producer": {"kind": "score",
               "model": "sum", "top_k": 5, "ties": "record_id_ascending"}}),
    );
    let slow: Value = serde_json::from_str(&slow.json).unwrap();
    assert_eq!(slow["status"], "running", "{slow}");
    slow
}

#[test]
fn a_view_that_ends_after_the_analyst_moved_on_does_not_take_over() {
    let options = HostOptions {
        inline_wait: Some(Duration::from_millis(20)),
        delay: Duration::from_millis(1500),
        ..HostOptions::default()
    };
    let (dir, service, bound) = timed(&options);
    let mut bench = Workbench::new(service, Start::from_bindings(&bound).unwrap(), None).unwrap();
    bench.settle();
    press(&mut bench, KeyCode::Char('1'));
    bench.settle();
    let d = bench.active_handle().to_string();

    // A drill that waits behind another client's job; the analyst goes back meanwhile.
    hold_the_worker(
        &bench,
        json!({"ref": "public-fixture", "entries": ["*.log"]}),
    );
    press(&mut bench, KeyCode::Char('2'));
    bench.tick();
    assert!(bench.render_text(120, 36).contains("Running"));
    press(&mut bench, KeyCode::Char('b'));
    let root = bench.active_handle().to_string();
    assert_ne!(root, d);
    bench.settle();
    assert_eq!(
        bench.active_handle(),
        root,
        "the finished drill does not pull the analyst"
    );
    let notice = bench.summary()["notice"].as_str().unwrap().to_string();
    assert!(
        notice.contains("ready") && notice.contains("v lists"),
        "{notice}"
    );
    // The view is kept: `v` lists it.
    press(&mut bench, KeyCode::Char('v'));
    let views = bench.render_text(120, 36);
    assert!(views.contains("a in · b in"), "{views}");
    press(&mut bench, KeyCode::Esc);

    // A drill that ends while the analyst types a search: the draft and the mode stay.
    press(&mut bench, KeyCode::Char('1'));
    bench.settle();
    hold_the_worker(
        &bench,
        json!({"path": dir.path().join("allowed").join("mine.zip"), "entries": ["*.log"]}),
    );
    press(&mut bench, KeyCode::Char('3'));
    bench.tick();
    press(&mut bench, KeyCode::Char('/'));
    type_text(&mut bench, "q1sb");
    bench.settle();
    let screen = bench.render_text(120, 36);
    assert!(
        screen.contains("Enter runs"),
        "still editing the search:\n{screen}"
    );
    assert!(screen.contains("q1sb"), "{screen}");
    assert_eq!(
        bench.active_handle(),
        d,
        "the finished drill did not switch views"
    );
    press(&mut bench, KeyCode::Esc);
    assert!(
        !bench.summary()["notice"]
            .as_str()
            .unwrap()
            .contains("Saved")
    );
}

#[test]
fn esc_cancels_a_running_job_and_nothing_is_committed() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let mut options = options(&universe);
    // Every scored batch takes this long: the slow job another client starts.
    options.delay = Duration::from_millis(1500);
    let dir = tempfile::tempdir().unwrap();
    write_archive(&dir.path().join("private.zip"), &entries);
    write_archive(&dir.path().join("public.zip"), &entries);
    let (service, bound) = opened(dir.path(), &options);
    let mut bench = Workbench::new(service, Start::from_bindings(&bound).unwrap(), None).unwrap();
    bench.settle();
    let at = bench.active_handle().to_string();
    let views_before = bench.render_text(120, 32);
    // Another client of the session holds the worker: it scores another snapshot. The
    // workbench's search job waits behind it, and Esc cancels the search.
    let other = call(
        bench.service(),
        "triager_open",
        json!({"v": 1, "source": {"ref": "public-fixture", "entries": ["*.log"]}}),
    );
    let slow = bench.service().call(
        "triager_select",
        &json!({"v": 1, "scope": other["snapshot"], "producer": {"kind": "score",
               "model": "sum", "top_k": 5, "ties": "record_id_ascending"}}),
    );
    let slow: Value = serde_json::from_str(&slow.json).unwrap();
    assert_eq!(slow["status"], "running", "{slow}");
    press(&mut bench, KeyCode::Char('/'));
    type_text(&mut bench, "*rejected*");
    press(&mut bench, KeyCode::Enter);
    bench.tick();
    let running = bench.render_text(120, 32);
    assert!(running.contains("Running search"), "{running}");
    press(&mut bench, KeyCode::Esc);
    assert!(bench.render_text(120, 32).contains("cancel requested"));
    bench.settle();
    let summary = bench.summary();
    let notice = summary["notice"].as_str().unwrap();
    assert!(notice.contains("Cancelled search"), "{notice}");
    assert_eq!(
        bench.active_handle(),
        at,
        "a cancelled search opens no view"
    );
    let _ = views_before;
    // The slow job of the other client is not affected, and the next search works.
    let other_done = loop {
        let status = call(
            bench.service(),
            "triager_job",
            json!({"v": 1, "job": slow["job"], "action": "status"}),
        );
        if !matches!(status["status"].as_str(), Some("waiting" | "running")) {
            break status;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(other_done["status"], "completed", "{other_done}");
    // On the service's side the cancelled search committed nothing: the same search
    // made now as an MCP client reads the archive again and reuses no result.
    let again = call(
        bench.service(),
        "triager_select",
        json!({"v": 1, "scope": at, "producer": {"kind": "filter", "syntax": "glob",
               "query": "*rejected*"}}),
    );
    assert_eq!(again["cost"]["archive_scans"], 1, "{again}");
    assert!(again["cost"]["reused"].is_null(), "{again}");
    press(&mut bench, KeyCode::Char('/'));
    press(&mut bench, KeyCode::Enter);
    bench.settle();
    assert_ne!(
        bench.active_handle(),
        at,
        "the same search, not cancelled, opens a view"
    );
}
