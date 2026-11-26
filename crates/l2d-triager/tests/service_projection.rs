//! Triager task 3.1: occurrence-aware field, time and record views (requirements 1.3,
//! 2.4, 4.3, 5.2, 6.1, 6.3 and 6.4), driven through `Service::call` and compared with
//! an oracle computed from the archive's lines without the product's extraction.
//!
//! One original is read from several hosts and on several days; records miss a
//! field, carry an overlong value, an unreadable or a missing timestamp; a field has
//! more values than it retains; and a field or time view drills back to every
//! occurrence of an original in the wider scope.

mod support;

use std::collections::{BTreeMap, BTreeSet};

use l2d_core::ArchivePosition;
use l2d_triager::scope::IndexPlacement;
use l2d_triager::service::{DisclosureProfile, TablePlacement};
use serde_json::{Value, json};
use support::{Host, HostOptions, write_archive};

/// The archive of the hosts: `logs/HOST/DAY.log`, the same text on several hosts and
/// days, an entry whose name is no date and one whose date does not exist.
fn hosts() -> Vec<(String, Vec<String>)> {
    let repeated = "login ok user=u1 status=200".to_string();
    vec![
        (
            "logs/alpha/2026-01-01.log".into(),
            vec![
                repeated.clone(),
                "login failed user=u2 status=500".into(),
                "cron run status=ok".into(),
                repeated.clone(),
            ],
        ),
        (
            "logs/beta/2026-01-01.log".into(),
            vec![repeated.clone(), "disk warn free=3".into()],
        ),
        (
            "logs/alpha/2026-01-02.log".into(),
            vec![
                repeated.clone(),
                "login failed user=u3 status=500".into(),
                format!("login ok user={} status=200", "x".repeat(300)),
            ],
        ),
        (
            "logs/gamma/undated.log".into(),
            vec![repeated.clone(), "cron run status=ok".into()],
        ),
        ("logs/beta/2026-02-30.log".into(), vec![repeated]),
        ("notes.txt".into(), vec!["ignored user=zz status=1".into()]),
    ]
}

/// One occurrence as the oracle reads it.
#[derive(Debug, Clone)]
struct Occ {
    original: u64,
    entry: String,
    host: Option<String>,
    day: Option<Result<i64, ()>>,
    text: String,
}

/// The value of `key` in a record of space-separated `key=value` tokens.
fn token(text: &str, key: &str) -> Option<String> {
    text.split(' ')
        .find_map(|part| part.strip_prefix(&format!("{key}=")))
        .map(str::to_string)
}

fn days(year: i64, month: i64, day: i64) -> Option<i64> {
    let lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=12).contains(&month) || day < 1 || day > lengths[month as usize - 1] {
        return None;
    }
    // Days since 1970-01-01, for the years of the fixture.
    let mut total = 0;
    for y in 1970..year {
        total += if y % 4 == 0 { 366 } else { 365 };
    }
    for m in 1..month {
        total += lengths[m as usize - 1] + i64::from(m == 2 && year % 4 == 0);
    }
    Some(total + day - 1)
}

fn occurrences(entries: &[(String, Vec<String>)]) -> Vec<Occ> {
    let mut first: BTreeMap<&str, u64> = BTreeMap::new();
    let mut found = Vec::new();
    for (index, (name, lines)) in entries.iter().enumerate() {
        if !name.ends_with(".log") {
            continue;
        }
        for (line, text) in lines.iter().enumerate() {
            let position = ArchivePosition {
                entry: index as u32,
                line: line as u64 + 1,
            }
            .pack();
            let original = *first.entry(text.as_str()).or_insert(position);
            let file = name.rsplit('/').next().unwrap();
            let day = file.strip_suffix(".log").and_then(|stem| {
                let parts: Vec<&str> = stem.split('-').collect();
                (parts.len() == 3 && parts[0].len() == 4).then(|| {
                    let number = |part: &str| part.parse::<i64>().unwrap();
                    days(number(parts[0]), number(parts[1]), number(parts[2]))
                        .map(|day| day * 86_400)
                        .ok_or(())
                })
            });
            found.push(Occ {
                original,
                entry: name.clone(),
                host: name.split('/').nth(1).map(str::to_string),
                day,
                text: text.clone(),
            });
        }
    }
    found
}

/// Occurrences and distinct originals of `occurrences` that `keep`.
fn counted(occurrences: &[Occ], keep: impl Fn(&Occ) -> bool) -> (u64, u64) {
    let kept: Vec<&Occ> = occurrences.iter().filter(|occ| keep(occ)).collect();
    let originals: BTreeSet<u64> = kept.iter().map(|occ| occ.original).collect();
    (originals.len() as u64, kept.len() as u64)
}

fn pair(value: &Value) -> (u64, u64) {
    (
        value["distinct_originals"]
            .as_u64()
            .unwrap_or_else(|| panic!("{value}")),
        value["occurrences"]
            .as_u64()
            .unwrap_or_else(|| panic!("{value}")),
    )
}

const HOSTS_PROJECTION: &str = r#"{
    "fields": [{"name": "host", "from": "entry", "capture": "^logs/([^/]+)/"},
               {"name": "user", "from": "kv"}],
    "time": {"from": "entry", "capture": "(\\d{4}-\\d{2}-\\d{2})\\.log$", "format": "date"}
}"#;

fn open_projected(host: &Host, projections: &Value) -> Value {
    host.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private", "entries": ["*.log"]},
               "projections": projections}),
    )
}

fn inspect(host: &Host, handle: &str, view: &str, extra: Value) -> Value {
    let mut args = json!({"v": 1, "handle": handle, "view": view});
    for (key, value) in extra.as_object().unwrap() {
        args[key] = value.clone();
    }
    host.ok("triager_inspect", args)
}

fn field<'v>(summary: &'v Value, name: &str) -> &'v Value {
    summary["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|field| field["field"] == name)
        .unwrap_or_else(|| panic!("{name} in {summary}"))
}

fn values(page: &Value) -> BTreeMap<String, (u64, u64)> {
    page["field"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| (row["value"].as_str().unwrap().to_string(), pair(row)))
        .collect()
}

fn drill(host: &Host, set: &str, action: Value) -> Value {
    host.ok(
        "triager_refine",
        json!({"v": 1, "set": set, "action": action}),
    )
}

/// What a run gives that must not depend on batch sizes, spill or placement: every
/// count and revision of every view, without costs and placements. Handles bind the
/// archive's path, which is another temporary directory in every run, and so does the
/// relation's revision and a native ID.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                // The revision of a snapshot is its relation's.
                .filter(|(key, _)| {
                    !(*key == "revision" && fields.get("kind") == Some(&json!("snapshot")))
                })
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "cost"
                            | "table"
                            | "disclosure"
                            | "tables"
                            | "activity"
                            | "session"
                            | "native_id"
                            | "artifact"
                    )
                })
                .map(|(key, value)| (key.clone(), canonical(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        Value::String(text)
            if ["snap-", "set-", "view-", "proj-", "c1."]
                .iter()
                .any(|prefix| text.starts_with(prefix)) =>
        {
            json!("<handle>")
        }
        other => other.clone(),
    }
}

/// The views of the hosts archive under `options`, checked against the oracle, and
/// their canonical form.
fn hosts_run(options: HostOptions) -> Vec<Value> {
    let entries = hosts();
    let oracle = occurrences(&entries);
    let host = Host::new(&entries, options);
    let projections: Value = serde_json::from_str(HOSTS_PROJECTION).unwrap();
    let opened = open_projected(&host, &projections);
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    assert_eq!(
        opened["counts"]["occurrences"],
        oracle.len() as u64,
        "{opened}"
    );
    let projection = &opened["projection"];
    assert_eq!(projection["occurrences"], oracle.len() as u64, "{opened}");
    assert_eq!(projection["cost"]["archive_scans"], 1, "{opened}");
    let mut seen = vec![
        opened["counts"].clone(),
        projection["revision"].clone(),
        projection["declared"].clone(),
    ];
    let before = host.activity(&snapshot);

    // The fields of the snapshot: typed counts, missing and invalid values.
    let summary = inspect(&host, &snapshot, "fields", json!({}));
    assert_eq!(summary["status"], "ready", "{summary}");
    assert_eq!(
        pair(&summary["population"]["counted"]),
        counted(&oracle, |_| true)
    );
    let user = field(&summary, "user");
    let overlong = |occ: &Occ| token(&occ.text, "user").is_some_and(|user| user.len() > 256);
    assert_eq!(
        pair(&user["missing"]),
        counted(&oracle, |occ| token(&occ.text, "user").is_none())
    );
    assert_eq!(pair(&user["invalid"]), counted(&oracle, overlong));
    assert_eq!(
        pair(&user["with_value"]),
        counted(&oracle, |occ| token(&occ.text, "user").is_some()
            && !overlong(occ))
    );
    assert_eq!(user["per_occurrence"], false);
    assert_eq!(user["distinct_values"]["count"], 3);
    assert_eq!(user["distinct_values"]["exact"], true);
    let host_field = field(&summary, "host");
    assert_eq!(host_field["per_occurrence"], true);
    assert_eq!(pair(&host_field["with_value"]), counted(&oracle, |_| true));
    assert_eq!(field(&summary, "source")["distinct_values"]["count"], 5);
    seen.push(canonical(&summary));

    // One original on several hosts: counted once in each host it was read on.
    let by_host = inspect(&host, &snapshot, "fields", json!({"field": "host"}));
    let expected: BTreeMap<String, (u64, u64)> = ["alpha", "beta", "gamma"]
        .iter()
        .map(|name| {
            (
                name.to_string(),
                counted(&oracle, |occ| occ.host.as_deref() == Some(name)),
            )
        })
        .collect();
    assert_eq!(values(&by_host), expected);
    let sum: u64 = expected.values().map(|(originals, _)| originals).sum();
    assert!(
        sum > counted(&oracle, |_| true).0,
        "a repeated original counts on each host"
    );
    assert_eq!(
        by_host["field"]["order"],
        "occurrences descending, then value ascending"
    );
    seen.push(canonical(&by_host));

    // The histogram: two days, the day that does not exist and the undated entry
    // counted apart.
    // The smallest width that draws the two days in at most 100 buckets: 15 minutes.
    let automatic = inspect(&host, &snapshot, "histogram", json!({}));
    assert_eq!(automatic["time"]["bucket_seconds"], 900, "{automatic}");
    assert_eq!(automatic["time"]["buckets"].as_array().unwrap().len(), 97);
    let histogram = inspect(
        &host,
        &snapshot,
        "histogram",
        json!({"bucket_seconds": 86_400}),
    );
    assert_eq!(histogram["status"], "ready", "{histogram}");
    let time = &histogram["time"];
    assert_eq!(time["field"], "_time");
    assert_eq!(time["timezone"], "UTC");
    assert_eq!(time["bucket_seconds"], 86_400);
    assert_eq!(time["range"]["start"], "2026-01-01T00:00:00Z");
    assert_eq!(time["range"]["end"], "2026-01-03T00:00:00Z");
    let day1 = days(2026, 1, 1).unwrap() * 86_400;
    let buckets = time["buckets"].as_array().unwrap();
    assert_eq!(buckets.len(), 2);
    for (at, bucket) in buckets.iter().enumerate() {
        let start = day1 + at as i64 * 86_400;
        assert_eq!(
            pair(bucket),
            counted(&oracle, |occ| occ.day == Some(Ok(start))),
            "{bucket}"
        );
    }
    assert_eq!(
        pair(&time["missing_time"]),
        counted(&oracle, |occ| occ.day.is_none())
    );
    assert_eq!(
        pair(&time["invalid_time"]),
        counted(&oracle, |occ| occ.day == Some(Err(())))
    );
    let bucketed: u64 = buckets.iter().map(|bucket| pair(bucket).1).sum();
    assert_eq!(
        bucketed + pair(&time["missing_time"]).1 + pair(&time["invalid_time"]).1,
        oracle.len() as u64
    );
    seen.push(canonical(&histogram));

    // Drill into one host: the members with an occurrence there; their matching
    // occurrences apart from all their occurrences.
    let beta = drill(
        &host,
        &snapshot,
        json!({"kind": "field", "field": "host", "value": "beta"}),
    );
    let on_beta = |occ: &Occ| occ.host.as_deref() == Some("beta");
    let (beta_originals, beta_occurrences) = counted(&oracle, on_beta);
    assert_eq!(beta["counts"]["in"], beta_originals, "{beta}");
    assert_eq!(beta["counts"]["parent"], counted(&oracle, |_| true).0);
    assert_eq!(beta["matching_occurrences"]["in"], beta_occurrences);
    assert_eq!(beta["matching_occurrences"]["parent"], oracle.len() as u64);
    let beta_members: BTreeSet<u64> = oracle
        .iter()
        .filter(|occ| on_beta(occ))
        .map(|occ| occ.original)
        .collect();
    assert_eq!(
        beta["occurrence_counts"]["in"],
        oracle
            .iter()
            .filter(|occ| beta_members.contains(&occ.original))
            .count() as u64
    );
    let beta_view = beta["view"].as_str().unwrap().to_string();
    seen.push(canonical(&beta));
    // Its histogram counts only the occurrences on beta.
    let beta_time = inspect(
        &host,
        &beta_view,
        "histogram",
        json!({"bucket_seconds": 86_400}),
    );
    assert_eq!(beta_time["time"]["buckets"].as_array().unwrap().len(), 1);
    assert_eq!(
        pair(&beta_time["time"]["buckets"][0]),
        counted(&oracle, |occ| on_beta(occ) && occ.day == Some(Ok(day1)))
    );
    assert_eq!(
        pair(&beta_time["time"]["invalid_time"]),
        counted(&oracle, |occ| on_beta(occ) && occ.day == Some(Err(())))
    );
    assert_eq!(
        pair(&beta_time["population"]["counted"]),
        (beta_originals, beta_occurrences)
    );
    seen.push(canonical(&beta_time));

    // And a time bucket below it: the second level of the filter.
    let beta_day1 = drill(
        &host,
        &beta_view,
        json!({"kind": "time", "start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"}),
    );
    let both = |occ: &Occ| on_beta(occ) && occ.day == Some(Ok(day1));
    assert_eq!(beta_day1["counts"]["in"], counted(&oracle, both).0);
    assert_eq!(beta_day1["counts"]["parent"], beta_originals);
    assert_eq!(
        beta_day1["matching_occurrences"]["in"],
        counted(&oracle, both).1
    );
    assert_eq!(
        beta_day1["matching_occurrences"]["parent"],
        beta_occurrences
    );
    assert_eq!(beta_day1["occurrence_filter"].as_array().unwrap().len(), 2);
    seen.push(canonical(&beta_day1));
    // Occurrences without a valid time, across the snapshot.
    let undated = drill(&host, &snapshot, json!({"kind": "time", "no_time": true}));
    assert_eq!(
        undated["matching_occurrences"]["in"],
        counted(&oracle, |occ| !matches!(occ.day, Some(Ok(_)))).1
    );
    seen.push(canonical(&undated));
    // Records without a user, and a value no record has.
    let anonymous = drill(
        &host,
        &snapshot,
        json!({"kind": "field", "field": "user", "missing": true}),
    );
    assert_eq!(
        anonymous["counts"]["in"],
        counted(&oracle, |occ| token(&occ.text, "user").is_none()).0
    );
    let nobody = drill(
        &host,
        &snapshot,
        json!({"kind": "field", "field": "user", "value": "u9"}),
    );
    assert_eq!(nobody["counts"]["in"], 0);
    // The entry distribution as a field.
    let entry = drill(
        &host,
        &snapshot,
        json!({"kind": "field", "field": "source", "value": "logs/alpha/2026-01-02.log"}),
    );
    assert_eq!(
        entry["matching_occurrences"]["in"],
        counted(&oracle, |occ| occ.entry == "logs/alpha/2026-01-02.log").1
    );
    seen.push(canonical(&entry));

    // Drill back from the view to the original in the wider scope: every occurrence,
    // which of them are on beta.
    let repeated = oracle
        .iter()
        .find(|occ| occ.text == "login ok user=u1 status=200")
        .unwrap()
        .original;
    let evidence = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": beta_view, "view": "evidence",
               "record": format!("r-{repeated:016x}")}),
    );
    let listed = &evidence["record"]["occurrence_list"];
    assert_eq!(evidence["record"]["occurrence_locators"], "projected");
    let all_of = |keep: &dyn Fn(&Occ) -> bool| {
        oracle
            .iter()
            .filter(|occ| occ.original == repeated && keep(occ))
            .count() as u64
    };
    assert_eq!(listed["total"], all_of(&|_| true), "{evidence}");
    assert_eq!(listed["in_view"], all_of(&on_beta));
    let rows = listed["rows"].as_array().unwrap();
    assert_eq!(rows.len() as u64, all_of(&|_| true));
    let hosts_listed: Vec<&str> = rows
        .iter()
        .map(|row| row["fields"]["host"].as_str().unwrap())
        .collect();
    assert_eq!(
        hosts_listed,
        ["alpha", "alpha", "beta", "alpha", "gamma", "beta"]
    );
    assert_eq!(
        rows.iter().filter(|row| row["in_view"] == true).count() as u64,
        all_of(&on_beta)
    );
    assert_eq!(rows[4]["time_state"], "missing");
    assert_eq!(rows[5]["time_state"], "invalid");
    assert_eq!(rows[0]["time"], "2026-01-01T00:00:00Z");
    seen.push(canonical(&evidence["record"]));
    // The overview of the view states both units.
    let overview = inspect(&host, &beta_view, "overview", json!({}));
    assert_eq!(
        overview["occurrences"]["matching_occurrences"]["in"],
        beta_occurrences
    );

    // Nothing above read the source, scored or encoded.
    let after = host.activity(&snapshot);
    for counter in [
        "archive_scans",
        "bytes_unpacked",
        "rows_scored",
        "rows_encoded",
    ] {
        assert_eq!(after[counter], before[counter], "{counter}");
    }
    seen
}

#[test]
fn field_time_and_record_views_count_every_occurrence_and_drill_back() {
    hosts_run(HostOptions::default());
}

#[test]
fn views_are_the_same_whatever_the_batches_spill_and_placement() {
    let reference = hosts_run(HostOptions::default());
    for options in [
        HostOptions {
            batch_rows: Some(2),
            dedup_set_texts: Some(2),
            ..HostOptions::default()
        },
        HostOptions {
            tables: TablePlacement::Disk,
            index: IndexPlacement::Disk,
            ..HostOptions::default()
        },
        HostOptions {
            batch_rows: Some(3),
            dedup_set_texts: Some(1),
            tables: TablePlacement::Disk,
            index: IndexPlacement::Disk,
            ..HostOptions::default()
        },
    ] {
        assert_eq!(hosts_run(options), reference);
    }
}

#[test]
fn timestamps_of_the_text_count_missing_and_unreadable_values_apart() {
    let lines: Vec<String> = [
        "2026-01-01T00:10:00Z login ok user=a",
        "2026-01-01T00:50:00Z login ok user=b",
        "2026-01-01T02:05:00+01:00 login failed user=a",
        "2026-01-01 00:20:00 login ok user=c",
        "login without time user=d",
        "2026-01-01T00:10:00Z login ok user=a",
        "[2026-01-01 00:59:59Z] bracketed user=e",
    ]
    .iter()
    .map(|line| line.to_string())
    .collect();
    let entries = vec![("app/text.log".to_string(), lines)];
    let host = Host::new(&entries, HostOptions::default());
    let opened = open_projected(
        &host,
        &json!({"fields": [{"name": "user", "from": "kv"}], "time": {"from": "text"}}),
    );
    let snapshot = opened["snapshot"].as_str().unwrap();
    let histogram = inspect(
        &host,
        snapshot,
        "histogram",
        json!({"bucket_seconds": 3_600}),
    );
    let time = &histogram["time"];
    let buckets = time["buckets"].as_array().unwrap();
    assert_eq!(buckets.len(), 2, "{histogram}");
    assert_eq!(buckets[0]["start"], "2026-01-01T00:00:00Z");
    // a at 00:10 twice (one original), b at 00:50 and e at 00:59:59.
    assert_eq!(pair(&buckets[0]), (3, 4));
    assert_eq!(pair(&buckets[1]), (1, 1));
    assert_eq!(pair(&time["missing_time"]), (1, 1));
    assert_eq!(pair(&time["invalid_time"]), (1, 1));
    assert_eq!(time["observed"]["earliest"], "2026-01-01T00:10:00Z");
    assert_eq!(time["observed"]["latest"], "2026-01-01T01:05:00Z");
    // Automatic width: the smallest of the widths that draws the range in 100 buckets.
    let automatic = inspect(&host, snapshot, "histogram", json!({}));
    assert_eq!(automatic["time"]["bucket_seconds"], 60);
    assert_eq!(automatic["time"]["buckets"].as_array().unwrap().len(), 56);
    // A width that draws more than 100 buckets is refused with the one that fits.
    let (code, message) = host.err(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "histogram", "bucket_seconds": 1}),
    );
    assert_eq!(code, "too_many_buckets");
    assert!(message.contains("60 s"), "{message}");
    // A bucket of the histogram as a child view.
    let first = drill(
        &host,
        snapshot,
        json!({"kind": "time", "start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z"}),
    );
    assert_eq!(first["counts"]["in"], 3);
    assert_eq!(first["matching_occurrences"]["in"], 4);
    let (code, _) = host.err(
        "triager_refine",
        json!({"v": 1, "set": snapshot,
               "action": {"kind": "time", "start": "2026-01-01T00:00:00", "end": "2026-01-01T01:00:00Z"}}),
    );
    assert_eq!(
        code, "invalid_arguments",
        "a time without a zone is no bucket boundary"
    );
}

#[test]
fn a_field_with_more_values_than_it_retains_counts_the_rest_as_overflow() {
    let lines: Vec<String> = (0..30)
        .map(|at| format!("request user=u{at:02} status={}", 200 + at % 3))
        .collect();
    let entries = vec![("app/many.log".to_string(), lines)];
    let host = Host::new(&entries, HostOptions::default());
    let opened = open_projected(
        &host,
        &json!({"fields": [{"name": "user", "from": "kv"}, {"name": "status", "from": "kv"}],
                "max_values": 10}),
    );
    let snapshot = opened["snapshot"].as_str().unwrap();
    let summary = inspect(&host, snapshot, "fields", json!({}));
    let user = field(&summary, "user");
    assert_eq!(pair(&user["with_value"]), (30, 30));
    assert_eq!(pair(&user["overflow"]), (20, 20));
    assert_eq!(user["distinct_values"]["count"], 10);
    assert_eq!(user["distinct_values"]["exact"], false);
    assert_eq!(user["value_counts_exact"], false);
    let status = field(&summary, "status");
    assert_eq!(status["distinct_values"]["exact"], true);
    assert_eq!(pair(&status["types"]["number"]), (30, 30));
    // The values of a field, a page at a time, with the overflow counted apart.
    let first = inspect(
        &host,
        snapshot,
        "fields",
        json!({"field": "user", "limit": 4}),
    );
    assert_eq!(first["field"]["values"].as_array().unwrap().len(), 4);
    assert_eq!(pair(&first["field"]["overflow"]), (20, 20));
    assert_eq!(first["field"]["distinct_values"]["exact"], false);
    // With values beyond the limit the order is among the retained ones only.
    assert_eq!(first["field"]["ranking_exact"], false);
    assert!(
        first["field"]["ranking"]
            .as_str()
            .unwrap()
            .contains("among retained values"),
        "{first}"
    );
    let statuses = inspect(&host, snapshot, "fields", json!({"field": "status"}));
    assert_eq!(statuses["field"]["ranking_exact"], true);
    let mut listed: Vec<String> = values(&first).into_keys().collect();
    let mut cursor = first["page"]["next_cursor"].as_str().unwrap().to_string();
    loop {
        let next = inspect(
            &host,
            snapshot,
            "fields",
            json!({"field": "user", "limit": 4, "cursor": cursor}),
        );
        listed.extend(values(&next).into_keys());
        match next["page"]["next_cursor"].as_str() {
            Some(more) => cursor = more.to_string(),
            None => break,
        }
    }
    listed.sort();
    assert_eq!(
        listed,
        (0..10).map(|at| format!("u{at:02}")).collect::<Vec<_>>()
    );
    // A cursor of another field is refused.
    let (code, _) = host.err(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "fields", "field": "status",
               "cursor": first["page"]["next_cursor"]}),
    );
    assert_eq!(code, "stale_cursor");
    // A retained value drills exactly; one beyond the limit is refused by name.
    let retained = drill(
        &host,
        snapshot,
        json!({"kind": "field", "field": "user", "value": "u03"}),
    );
    assert_eq!(retained["counts"]["in"], 1);
    let (code, message) = host.err(
        "triager_refine",
        json!({"v": 1, "set": snapshot,
               "action": {"kind": "field", "field": "user", "value": "u25"}}),
    );
    assert_eq!(code, "value_not_retained");
    assert!(message.contains("first 10 values"), "{message}");
}

#[test]
fn a_field_no_projection_declares_states_the_scan_and_an_explicit_rescan_provides_it() {
    let entries = hosts();
    let oracle = occurrences(&entries);
    let host = Host::new(&entries, HostOptions::default());
    // Before any projection: what is missing, and what a scan would take.
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let missing = inspect(&host, &snapshot, "histogram", json!({}));
    assert_eq!(missing["status"], "projection_missing", "{missing}");
    assert_eq!(missing["missing"]["time"], true);
    assert_eq!(missing["estimate"]["archive_scans"], 1);
    assert_eq!(
        missing["estimate"]["records_to_scan"],
        opened["completeness"]["records_scanned"]
    );
    assert_eq!(missing["estimate"]["rows_scored"], 0);
    let (code, _) = host.err(
        "triager_refine",
        json!({"v": 1, "set": snapshot,
               "action": {"kind": "field", "field": "user", "value": "u1"}}),
    );
    assert_eq!(code, "projection_missing");
    // A first projection, then a field it does not declare.
    let projections: Value = serde_json::from_str(HOSTS_PROJECTION).unwrap();
    let first = host.ok(
        "triager_open",
        json!({"v": 1, "snapshot": snapshot, "projections": projections}),
    );
    assert_eq!(first["status"], "ready", "{first}");
    let before = host.activity(&snapshot);
    let wanted = inspect(&host, &snapshot, "fields", json!({"field": "status"}));
    assert_eq!(wanted["status"], "projection_missing", "{wanted}");
    assert_eq!(wanted["missing"]["fields"], json!(["status"]));
    assert_eq!(
        wanted["missing"]["coverage"]["not_projected"],
        oracle.len() as u64
    );
    assert_eq!(wanted["projections"]["projection"], first["projection"]);
    assert_eq!(
        wanted["estimate"]["bytes_unpacked"],
        opened["cost"]["bytes_unpacked"]
    );
    // Asking scheduled nothing.
    assert_eq!(
        host.activity(&snapshot)["archive_scans"],
        before["archive_scans"]
    );
    // The request the estimate names runs the scan, keeping what was declared.
    let rescan = wanted["rescan"].clone();
    assert_eq!(rescan["tool"], "triager_open");
    let second = host.ok("triager_open", rescan["args"].clone());
    assert_eq!(second["status"], "ready", "{second}");
    assert_eq!(second["previous"], first["projection"]);
    assert_eq!(second["cost"]["archive_scans"], 1);
    let names: Vec<&str> = second["declared"]["fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["host", "status", "user"]);
    assert_eq!(
        host.activity(&snapshot)["archive_scans"],
        before["archive_scans"].as_u64().unwrap() + 1
    );
    // The same request again is the same projection, with no scan.
    let again = host.ok("triager_open", rescan["args"].clone());
    assert_eq!(again["projection"], second["projection"]);
    assert_eq!(
        host.activity(&snapshot)["archive_scans"],
        before["archive_scans"].as_u64().unwrap() + 1
    );
    // Reconciled with the oracle.
    let status = inspect(&host, &snapshot, "fields", json!({"field": "status"}));
    let expected: BTreeMap<String, (u64, u64)> = ["200", "500", "ok"]
        .iter()
        .map(|value| {
            (
                value.to_string(),
                counted(&oracle, |occ| {
                    token(&occ.text, "status").as_deref() == Some(value)
                }),
            )
        })
        .collect();
    assert_eq!(values(&status), expected);
    // Another declaration under a declared name is refused by name.
    let (code, _) = host.err(
        "triager_open",
        json!({"v": 1, "snapshot": snapshot,
               "projections": {"fields": [{"name": "user", "from": "entry", "capture": "x"}]}}),
    );
    assert_eq!(code, "field_defined");
}

#[test]
fn a_changed_source_or_reading_never_supplies_a_projection_silently() {
    let entries = hosts();
    let host = Host::new(&entries, HostOptions::default());
    let projections: Value = serde_json::from_str(HOSTS_PROJECTION).unwrap();
    let opened = open_projected(&host, &projections);
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    // Another reader setting is another snapshot, with no projection.
    let other = host.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private", "entries": ["*.log"], "max_record_bytes": 4000}}),
    );
    let other_snapshot = other["snapshot"].as_str().unwrap();
    assert_ne!(other_snapshot, snapshot);
    let missing = inspect(&host, other_snapshot, "fields", json!({}));
    assert_eq!(missing["status"], "projection_missing", "{missing}");
    assert_eq!(
        missing["missing"]["coverage"]["reason"],
        "the snapshot has no occurrence projection"
    );
    // A changed archive is not read for a projection of the snapshot it was.
    let mut changed = entries.clone();
    changed[0].1.push("login ok user=u7 status=200".into());
    write_archive(&host.dir.path().join("private.zip"), &changed);
    let (code, message) = host.err(
        "triager_open",
        json!({"v": 1, "snapshot": snapshot,
               "projections": {"fields": [{"name": "status", "from": "kv"}]}}),
    );
    assert_eq!(code, "refused", "{message}");
    assert!(message.contains("changed"), "{message}");
    // The committed projection stays usable, and its views read no source.
    let summary = inspect(&host, &snapshot, "fields", json!({}));
    assert_eq!(summary["status"], "ready");
}

#[test]
fn a_declared_timestamp_without_a_valid_value_offers_the_entries_as_a_labelled_alternative() {
    let entries = hosts();
    let oracle = occurrences(&entries);
    let host = Host::new(&entries, HostOptions::default());
    // The records of the hosts archive have no leading timestamp: the declared one is
    // missing on every occurrence.
    let opened = open_projected(
        &host,
        &json!({"fields": [{"name": "user", "from": "kv"}], "time": {"from": "text"}}),
    );
    let snapshot = opened["snapshot"].as_str().unwrap();
    let histogram = inspect(&host, snapshot, "histogram", json!({"limit": 2}));
    assert_eq!(histogram["status"], "time_unavailable", "{histogram}");
    assert_eq!(histogram["time"]["available"], false);
    assert_eq!(histogram["time"]["declared"]["from"], "text");
    assert_eq!(
        histogram["time"]["reason"],
        "no counted occurrence has a valid timestamp"
    );
    assert_eq!(
        pair(&histogram["time"]["missing_time"]),
        counted(&oracle, |_| true)
    );
    assert_eq!(pair(&histogram["time"]["with_time"]), (0, 0));
    let alternative = &histogram["alternative"];
    assert_eq!(alternative["kind"], "entry_order");
    assert!(alternative["label"].as_str().unwrap().contains("not time"));
    let mut rows: Vec<Value> = alternative["rows"].as_array().unwrap().clone();
    let mut cursor = histogram["page"]["next_cursor"]
        .as_str()
        .map(str::to_string);
    while let Some(next) = cursor {
        let page = inspect(
            &host,
            snapshot,
            "histogram",
            json!({"limit": 2, "cursor": next}),
        );
        rows.extend(
            page["alternative"]["rows"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
        cursor = page["page"]["next_cursor"].as_str().map(str::to_string);
    }
    let names: Vec<&str> = rows
        .iter()
        .map(|row| row["entry"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "logs/alpha/2026-01-01.log",
            "logs/beta/2026-01-01.log",
            "logs/alpha/2026-01-02.log",
            "logs/gamma/undated.log",
            "logs/beta/2026-02-30.log"
        ]
    );
    for row in &rows {
        let name = row["entry"].as_str().unwrap();
        assert_eq!(pair(row), counted(&oracle, |occ| occ.entry == name));
    }
}

#[test]
fn a_timestamp_no_projection_declares_states_the_scan_that_provides_it() {
    let entries = hosts();
    let host = Host::new(&entries, HostOptions::default());
    let opened = open_projected(&host, &json!({"fields": [{"name": "user", "from": "kv"}]}));
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let before = host.activity(&snapshot);
    // A projection that declares no timestamp has no time to draw: what is missing,
    // the scan and the request that provides it, and nothing scheduled.
    let histogram = inspect(&host, &snapshot, "histogram", json!({}));
    assert_eq!(histogram["status"], "projection_missing", "{histogram}");
    assert_eq!(histogram["missing"]["time"], true);
    assert_eq!(histogram["missing"]["fields"], json!([]));
    assert_eq!(histogram["estimate"]["archive_scans"], 1);
    assert_eq!(
        histogram["projections"]["projection"],
        opened["projection"]["projection"]
    );
    let args = histogram["rescan"]["args"].clone();
    assert_eq!(histogram["rescan"]["tool"], "triager_open");
    assert_eq!(args["snapshot"], snapshot);
    assert_eq!(args["projections"]["time"]["from"], "text");
    assert_eq!(args["projections"]["fields"][0]["name"], "user");
    assert_eq!(
        host.activity(&snapshot)["archive_scans"],
        before["archive_scans"]
    );
    // A time drill is refused the same way, by name.
    let (code, _) = host.err(
        "triager_refine",
        json!({"v": 1, "set": snapshot, "action": {"kind": "time", "no_time": true}}),
    );
    assert_eq!(code, "projection_missing");
    // The named request declares it; the archive's records have no timestamp, so the
    // declared one is then missing everywhere.
    let rescanned = host.ok("triager_open", args);
    assert_eq!(rescanned["cost"]["archive_scans"], 1);
    let after = inspect(&host, &snapshot, "histogram", json!({}));
    assert_eq!(after["status"], "time_unavailable", "{after}");
    assert_eq!(after["time"]["declared"]["from"], "text");
}

#[test]
fn a_drill_into_a_value_longer_than_a_field_keeps_is_refused_by_name() {
    let entries = hosts();
    let host = Host::new(&entries, HostOptions::default());
    let projections: Value = serde_json::from_str(HOSTS_PROJECTION).unwrap();
    let opened = open_projected(&host, &projections);
    let snapshot = opened["snapshot"].as_str().unwrap();
    // The fixture has one: counted as invalid, never as a value.
    let (code, message) = host.err(
        "triager_refine",
        json!({"v": 1, "set": snapshot,
               "action": {"kind": "field", "field": "user", "value": "x".repeat(300)}}),
    );
    assert_eq!(code, "value_too_long", "{message}");
    assert!(message.contains("256"), "{message}");
}

#[test]
fn values_from_record_text_count_against_the_public_development_allowance() {
    let entries = hosts();
    let options = HostOptions {
        profile: DisclosureProfile::PublicDevelopment,
        evidence_allowance_bytes: 1_000,
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let projections: Value = serde_json::from_str(HOSTS_PROJECTION).unwrap();
    let opened = host.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "public-fixture", "entries": ["*.log"]},
               "projections": projections}),
    );
    let snapshot = opened["snapshot"].as_str().unwrap();
    // Entry names are shown as locators are: they cost nothing.
    let by_host = inspect(&host, snapshot, "fields", json!({"field": "host"}));
    assert_eq!(by_host["disclosure"]["remaining_evidence_bytes"], 1_000);
    // A value read from the record's text is evidence.
    let users = inspect(&host, snapshot, "fields", json!({"field": "user"}));
    let shown: u64 = values(&users).keys().map(|value| value.len() as u64).sum();
    assert_eq!(
        users["disclosure"]["remaining_evidence_bytes"],
        1_000 - shown
    );
    // A private source is not opened under this profile, with or without projections.
    let (code, _) = host.err(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private"}, "projections": projections}),
    );
    assert_eq!(code, "disclosure_refused");
}

#[test]
fn a_field_view_saves_and_reopens_without_its_projection() {
    let entries = hosts();
    let host = Host::new(&entries, HostOptions::default());
    let projections: Value = serde_json::from_str(HOSTS_PROJECTION).unwrap();
    let opened = open_projected(&host, &projections);
    let snapshot = opened["snapshot"].as_str().unwrap();
    let alpha = drill(
        &host,
        snapshot,
        json!({"kind": "field", "field": "host", "value": "alpha"}),
    );
    let saved = host.ok(
        "triager_save",
        json!({"v": 1, "handle": alpha["view"], "idempotency_key": "alpha"}),
    );
    assert_eq!(saved["status"], "ready", "{saved}");
    let reopened = host.ok("triager_open", json!({"v": 1, "receipt": saved["receipt"]}));
    let set = reopened["set"].as_str().unwrap();
    let overview = inspect(&host, set, "overview", json!({}));
    assert_eq!(
        overview["members"]["distinct_originals"],
        alpha["counts"]["in"]
    );
    // A reopened receipt retains no source and so no occurrences.
    let (code, _) = host.err(
        "triager_inspect",
        json!({"v": 1, "handle": set, "view": "fields"}),
    );
    assert_eq!(code, "source_not_retained");
}
