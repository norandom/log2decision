//! CLI marking contracts (task 1.25).
use l2d_core::marks::MarkSet;
use l2d_core::{
    InputAdapter, LabelledAdapter, LabelledFormat, MemoryBudget, MemorySettings, ProjectPaths,
};
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("l2d-cli-mark-{}-{name}", std::process::id()));
        fs::create_dir_all(dir.join("project")).unwrap();
        Self(dir)
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_l2d"))
            .arg("--project")
            .arg(self.0.join("project"))
            .args(args)
            .output()
            .unwrap()
    }
    fn json(&self, args: &[&str]) -> Value {
        let mut all = vec!["--json"];
        all.extend_from_slice(args);
        let out = self.run(&all);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
    fn zip(&self) -> String {
        let path = self.0.join("logs.zip");
        let mut zip = zip::ZipWriter::new(fs::File::create(&path).unwrap());
        for (name, text) in [
            ("a.log", "safe\nhit one\nhit one\n"),
            ("b.log", "hit two\nsafe\nhit three\n"),
        ] {
            zip.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(text.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
        path.to_string_lossy().into_owned()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mini-labelled.csv")
        .to_string_lossy()
        .into_owned()
}
fn positions(v: &Value) -> Vec<u64> {
    v["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["first_position"].as_u64().unwrap())
        .collect()
}
#[test]
fn fixture_phrase_regex_positions_and_text() {
    let s = Scratch::new("fixture");
    for (option, query, expected) in [
        ("--phrase", "UNION SELECT", vec![7, 57, 83, 188]),
        (
            "--regex",
            r"(?i)drop(\s|\+)+table",
            vec![8, 27, 46, 52, 65, 84, 170, 189],
        ),
    ] {
        let v = s.json(&["mark", "--input", &fixture(), option, query]);
        assert_eq!(positions(&v), expected);
        assert_eq!(v["count"], expected.len());
        assert!(
            v["matches"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| !r["text"].as_str().unwrap().is_empty())
        );
    }
    let out = s.run(&["mark", "--input", &fixture(), "--phrase", "UNION SELECT"]);
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.contains("UNION SELECT") && text.contains("188") && text.contains("4 distinct"),
        "{text}"
    );
}
#[test]
fn saved_set_loads_and_changed_source_is_refused() {
    let s = Scratch::new("saved");
    let path = s.0.join("input.csv");
    fs::copy(fixture(), &path).unwrap();
    s.json(&[
        "mark",
        "--input",
        path.to_str().unwrap(),
        "--phrase",
        "UNION SELECT",
        "--name",
        "union",
        "--label",
        "good",
    ]);
    let mut budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let set = MarkSet::load(&ProjectPaths::new(s.0.join("project")), "union", &budget).unwrap();
    assert_eq!(set.positions(), [7, 57, 83, 188]);
    assert_eq!(set.label(), l2d_core::Label::Good);
    let (_, original, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .read_with(&path, &mut budget)
        .unwrap();
    set.require_input(&original, None).unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"\nchanged,good\n")
        .unwrap();
    let (_, changed, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .read_with(&path, &mut budget)
        .unwrap();
    assert!(set.require_input(&changed, None).is_err());
    let saved = fs::read(s.0.join("project/marks/union.json")).unwrap();
    let out = s.run(&[
        "--json",
        "mark",
        "--input",
        path.to_str().unwrap(),
        "--phrase",
        "UNION SELECT",
        "--name",
        "union",
        "--label",
        "good",
    ]);
    assert_eq!(out.status.code(), Some(4));
    for line in String::from_utf8(out.stdout).unwrap().lines() {
        serde_json::from_str::<Value>(line).unwrap();
    }
    assert_eq!(
        fs::read(s.0.join("project/marks/union.json")).unwrap(),
        saved
    );
}
#[test]
fn archive_parallel_positions_count_and_selection() {
    let s = Scratch::new("archive");
    let path = s.zip();
    let mut previous = None;
    for n in ["1", "2", "3"] {
        let v = s.json(&[
            "mark",
            "--input",
            &path,
            "--regex",
            "^hit",
            "--entries-at-once",
            n,
            "--batch-rows",
            "1",
        ]);
        assert_eq!(v["count"], 3);
        let rows = v["matches"].clone();
        assert_eq!(rows[0]["entry"], "a.log");
        assert_eq!(rows[0]["line"], 2);
        assert_eq!(rows[1]["entry"], "b.log");
        assert_eq!(rows[1]["line"], 1);
        assert_eq!(rows[2]["line"], 3);
        if let Some(p) = previous {
            assert_eq!(rows, p);
        }
        previous = Some(rows);
    }
    let v = s.json(&[
        "mark", "--input", &path, "--phrase", "hit", "--entry", "b.log", "--name", "hits",
    ]);
    assert_eq!(v["count"], 2);
    let budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let set = MarkSet::load(&ProjectPaths::new(s.0.join("project")), "hits", &budget).unwrap();
    assert_eq!(
        set.input().unwrap().archive_entries.as_deref(),
        Some(["b.log".to_string()].as_slice())
    );
}
#[test]
fn errors_are_whole_json_and_memory_is_refused() {
    let s = Scratch::new("errors");
    for (args, code) in [
        (vec!["mark", "--input", &fixture(), "--regex", "["], 2),
        (
            vec!["mark", "--input", "missing-file.log", "--phrase", "x"],
            3,
        ),
        (
            vec![
                "mark",
                "--input",
                &fixture(),
                "--regex",
                "x",
                "--memory-limit",
                "40MiB",
            ],
            4,
        ),
    ] {
        let mut all = vec!["--json"];
        all.extend(args);
        let out = s.run(&all);
        assert_eq!(
            out.status.code(),
            Some(code),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(!text.is_empty());
        for line in text.lines() {
            serde_json::from_str::<Value>(line).unwrap();
        }
    }
}
#[test]
fn damaged_archive_reports_other_matches_as_whole_documents_without_saving() {
    let s = Scratch::new("damaged");
    let path = s.0.join("damaged.zip");
    let mut writer = zip::ZipWriter::new(fs::File::create(&path).unwrap());
    for (name, text) in [
        ("a.log", "hit one\n".to_string()),
        ("broken.log", "broken text\n".repeat(50)),
        ("z.log", "hit two\n".to_string()),
    ] {
        writer
            .start_file(
                name,
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
        writer.write_all(text.as_bytes()).unwrap();
    }
    writer.finish().unwrap();
    let mut bytes = fs::read(&path).unwrap();
    let at = bytes.windows(10).position(|w| w == b"broken.log").unwrap();
    bytes[at + 10..at + 14].fill(0xff);
    fs::write(&path, bytes).unwrap();
    let out = s.run(&[
        "--json",
        "mark",
        "--input",
        path.to_str().unwrap(),
        "--phrase",
        "hit",
        "--name",
        "broken",
    ]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let docs: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0]["count"], 2);
    assert_eq!(docs[0]["input"]["entries_not_read"]["damaged"], 1);
    assert_eq!(docs[1]["error"]["exit_code"], 2);
    assert!(!s.0.join("project/marks/broken.json").exists());
}
#[test]
fn saved_cli_marks_are_usable_by_redaction_and_fitting() {
    use l2d_baseline::{
        BaselineSettings,
        markscore::{MarkFitSettings, fit_from_marks},
        redact::{RedactionInput, RedactionSettings, propose_redactions},
    };
    let s = Scratch::new("consumers");
    s.json(&[
        "mark",
        "--input",
        &fixture(),
        "--regex",
        r"(?i)drop(\s|\+)+table",
        "--name",
        "drop-table",
    ]);
    let mut budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let set = MarkSet::load(
        &ProjectPaths::new(s.0.join("project")),
        "drop-table",
        &budget,
    )
    .unwrap();
    let (logs, summary, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .read_with(Path::new(&fixture()), &mut budget)
        .unwrap();
    let proposals = propose_redactions(
        &set,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &RedactionSettings::default(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(proposals.marked_logs, 8);
    let fit = fit_from_marks(
        &set,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &MarkFitSettings {
            baseline: BaselineSettings {
                buckets: 1024,
                ..Default::default()
            },
            ..Default::default()
        },
        &mut budget,
    )
    .unwrap();
    assert_eq!(fit.counts.marked, 8);
}
#[test]
fn late_save_error_closes_json_and_preserves_existing_archive_marks() {
    let s = Scratch::new("late-save");
    let path = s.zip();
    let initial = s.json(&[
        "mark", "--input", &path, "--phrase", "hit", "--name", "hits",
    ]);
    assert_eq!(initial["memory"]["operations"][0]["entries_at_once"], 2);
    let saved = fs::read(s.0.join("project/marks/hits.json")).unwrap();
    let out = s.run(&[
        "--json", "mark", "--input", &path, "--phrase", "safe", "--name", "hits",
    ]);
    assert_eq!(out.status.code(), Some(4));
    let docs: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0]["incomplete"], true);
    assert_eq!(docs[1]["error"]["kind"], "refused");
    assert_eq!(
        fs::read(s.0.join("project/marks/hits.json")).unwrap(),
        saved
    );
}
#[test]
fn archive_text_includes_entry_summary_and_memory_refusal_is_json() {
    let s = Scratch::new("archive-text");
    let path = s.zip();
    let out = s.run(&["mark", "--input", &path, "--phrase", "hit"]);
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.contains("a.log:2") && text.contains("b.log:3") && text.contains("entries: 2"),
        "{text}"
    );
    let out = s.run(&[
        "--json",
        "mark",
        "--input",
        &path,
        "--regex",
        "hit",
        "--memory-limit",
        "40MiB",
        "--name",
        "tight",
    ]);
    assert_eq!(out.status.code(), Some(4));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["error"]["kind"], "refused");
    assert!(!s.0.join("project/marks/tight.json").exists());
}
#[test]
fn saved_archive_marks_feed_redaction_and_fitting() {
    use l2d_baseline::{
        BaselineSettings,
        markscore::{MarkFitSettings, fit_from_marks},
        redact::{RedactionInput, RedactionSettings, propose_redactions},
    };
    let s = Scratch::new("archive-consumers");
    let path = s.0.join("fit.zip");
    let mut zip = zip::ZipWriter::new(fs::File::create(&path).unwrap());
    zip.start_file("host.log", zip::write::SimpleFileOptions::default())
        .unwrap();
    for i in 0..20 {
        writeln!(
            zip,
            "host common {} record {i}",
            if i < 10 { "attack" } else { "normal" }
        )
        .unwrap();
    }
    zip.finish().unwrap();
    s.json(&[
        "mark",
        "--input",
        path.to_str().unwrap(),
        "--phrase",
        "attack",
        "--name",
        "attacks",
    ]);
    let mut budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let set = MarkSet::load(&ProjectPaths::new(s.0.join("project")), "attacks", &budget).unwrap();
    let settings = l2d_core::ZipInputSettings::default();
    let input = || RedactionInput::Archive {
        path: &path,
        settings: &settings,
    };
    let proposals =
        propose_redactions(&set, input(), &RedactionSettings::default(), &mut budget).unwrap();
    assert_eq!(proposals.marked_logs, 10);
    let fit = fit_from_marks(
        &set,
        None,
        input(),
        &MarkFitSettings {
            baseline: BaselineSettings {
                buckets: 1024,
                ..Default::default()
            },
            ..Default::default()
        },
        &mut budget,
    )
    .unwrap();
    assert_eq!(fit.counts.marked, 10);
}

#[test]
fn analyst_workflow_pins_rules_score_history_and_replays_policy() {
    let s = Scratch::new("analyst-policy");
    let input = s.0.join("training.txt");
    let text = (0..12)
        .flat_map(|i| {
            [
                format!("host alarm denied {i}"),
                format!("host request accepted {i}"),
            ]
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&input, &text).unwrap();
    let input = input.to_str().unwrap();
    for (phrase, label) in [("denied", "bad"), ("accepted", "good")] {
        let v = s.json(&[
            "mark", "--input", input, "--phrase", phrase, "--label", label, "--name", label,
        ]);
        assert_eq!(v["count"], 12);
    }
    let group = s.json(&[
        "group",
        "--mark-set",
        "bad",
        "--method",
        "character-trigram",
        "--radius",
        "1",
        "--deselect-position",
        "1",
    ]);
    assert_eq!(group["selected"], 11);
    assert!(group["groups"][0]["boundary"]["text"].is_string());
    let human = s.run(&[
        "group",
        "--mark-set",
        "bad",
        "--method",
        "character-trigram",
        "--radius",
        "1",
    ]);
    assert!(human.status.success());
    assert!(
        String::from_utf8(human.stdout)
            .unwrap()
            .contains("representative")
    );
    let proposal = s.json(&["redact", "--mark-set", "bad", "--counter-slots", "256"]);
    assert!(
        proposal["proposals"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["fragment"] == "host")
    );
    assert!(proposal["confirmed_rules"].as_array().unwrap().is_empty());
    let rules = s.0.join("rules.json");
    fs::write(
        &rules,
        r#"[{"kind":"literal","fragment":"host","placeholder":""}]"#,
    )
    .unwrap();
    let redacted = s.json(&[
        "redact",
        "--mark-set",
        "bad",
        "--rules",
        rules.to_str().unwrap(),
        "--counter-slots",
        "256",
    ]);
    let bundle = s.0.join("evidence");
    let fit = s.json(&[
        "score",
        "fit",
        "--mark-set",
        "bad",
        "--other-mark-set",
        "good",
        "--name",
        "fixture",
        "--out",
        bundle.to_str().unwrap(),
    ]);
    assert_eq!(redacted["rules_sha256"], fit["rules_sha256"]);
    assert_eq!(fit["counts"]["deselected"], 1);
    assert_eq!(fit["scorer"]["kind"], "mark-set");
    let stored: Value =
        serde_json::from_slice(&fs::read(bundle.join("score-redaction.json")).unwrap()).unwrap();
    assert_eq!(stored["scorer"], fit["scorer"]);
    let queries = s.0.join("queries.txt");
    fs::write(
        &queries,
        "host alarm denied 1\ncompletely novel\nhost request accepted 1\n",
    )
    .unwrap();
    let q = queries.to_str().unwrap();
    let common = [
        "triage",
        "--input",
        q,
        "--baseline",
        "fixture",
        "--mark-set",
        "bad",
        "--method",
        "character-trigram",
    ];
    let close = s.json(
        &[
            &common[..],
            &["--radius", "0", "--top-k", "3", "--min-support", "1"],
        ]
        .concat(),
    );
    assert_eq!(close["decisions"][1]["evidence"]["close_enough"], false);
    assert!(close["decisions"][1]["evidence"]["ratio"].is_null());
    assert!(close["decisions"][0]["evidence"]["ratio"].is_number());
    let policy = s.0.join("policy.json");
    fs::write(&policy,serde_json::to_vec(&serde_json::json!({"version":"test-1","selection_provenance":"synthetic fixture","low":0.45,"high":0.55,
        "corroboration":1,"reach":{"radius":0.0,"top_k":3,"minimum_labelled_support":1,"expected_profile":close["decisions"][0]["evidence"]["profile"]},
        "require_calibration":false,"review_budget":2,"unknown_slots":1,"audit_slots":1,"audit_seed":5})).unwrap()).unwrap();
    let saved = s.0.join("report.json");
    let config = ["--policy", policy.to_str().unwrap()];
    let first = s.json(&[&common[..], &config, &["--out", saved.to_str().unwrap()]].concat());
    let policy = &first["policy_report"];
    assert_eq!(policy["decisions"].as_array().unwrap().len(), 3);
    assert_eq!(policy["unknown"]["configured"], 1);
    assert_eq!(policy["audit"]["configured"], 1);
    let unsupported = policy["decisions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["input"]["text"] == "completely novel")
        .unwrap();
    assert!(unsupported["score"]["actionable_ratio"].is_null());
    assert_eq!(unsupported["queue"], "Review");
    let open = ["tui", "--report", saved.to_str().unwrap()];
    let imported = s.json(&open);
    assert_eq!(imported["counts"]["originals"], 3);
    assert_eq!(imported["counts"]["occurrences"], Value::Null);
    assert_eq!(imported["regions"]["unknown"]["originals"], 3);
    let digest = l2d_core::input::content_sha256(&fs::read(&saved).unwrap());
    assert_eq!(imported["source_revision"], digest);
    let decisions = policy["decisions"].as_array().unwrap();
    let expected_in = decisions
        .iter()
        .filter(|r| matches!(r["queue"].as_str(), Some("Investigate" | "Review")))
        .count();
    let routed = s.json(&[&open[..], &["--policy-recommendations"]].concat());
    assert_eq!(routed["regions"]["recommended"]["originals"], expected_in);
    assert_eq!(routed["regions"]["pass"]["originals"], 3 - expected_in);
    assert_eq!(routed["regions"]["unknown"]["originals"], 0);
    let projected_ids: std::collections::BTreeSet<_> = routed["preview"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        projected_ids,
        decisions
            .iter()
            .map(|r| r["input"]["id"].as_str().unwrap())
            .collect()
    );
    let replay = s.json(&[&common[..], &config, &["--replay", saved.to_str().unwrap()]].concat());
    assert_eq!(replay["replay_verified"], true);
    assert_eq!(policy["query_source"]["path"], q);
    // Same originals and decisions, different query snapshot.
    let qtext = fs::read_to_string(q).unwrap();
    fs::write(q, format!("{qtext}\n")).unwrap();
    let mismatch = s.run(&[&common[..], &config, &["--replay", saved.to_str().unwrap()]].concat());
    assert_eq!(mismatch.status.code(), Some(4));
    // Opening retained evidence remains possible after the live source changes.
    assert_eq!(s.json(&open)["source_revision"], digest);
    fs::write(q, qtext).unwrap();
    fs::write(input, format!("{text}\nnew source row")).unwrap();
    let stale = s.run(&[&common[..], &config].concat());
    assert_eq!(stale.status.code(), Some(4));
}

#[test]
fn grouped_archive_history_uses_only_marked_entries_and_real_host() {
    let s = Scratch::new("analyst-archive");
    let zip = s.zip();
    let marks = s.json(&[
        "mark", "--input", &zip, "--entry", "a.log", "--phrase", "hit", "--name", "selected",
    ]);
    assert_eq!(marks["count"], 1);
    let groups = s.json(&[
        "group",
        "--mark-set",
        "selected",
        "--method",
        "character-trigram",
        "--radius",
        "1",
    ]);
    assert_eq!(groups["groups"][0]["size"], 1);
    assert_eq!(groups["groups"][0]["occurrences"], 2);
    assert_eq!(groups["source"]["archive_entries"][0], "a.log");
    let remove = [
        "group",
        "--mark-set",
        "selected",
        "--method",
        "character-trigram",
        "--radius",
        "1",
        "--deselect-group",
        "0",
        "--expect-grouping",
    ];
    let bad = s.run(&[&remove[..], &["stale"]].concat());
    assert_eq!(bad.status.code(), Some(4));
    let changed = s.json(
        &[
            &remove[..],
            &[groups["grouping_revision"].as_str().unwrap()],
        ]
        .concat(),
    );
    assert_eq!(changed["selected"], 0);
}

#[test]
fn remediation_history_interpretation_is_bound_to_the_review() {
    let s = Scratch::new("interpretation");
    let path = s.0.join("columns.csv");
    fs::write(&path,"text,other,label,endpoint,alias\nhit one,changed one,bad,h1,x1\nhit one,changed two,bad,h1,x2\nsafe,changed three,good,h2,x3\n").unwrap();
    s.json(&[
        "mark",
        "--input",
        path.to_str().unwrap(),
        "--group-column",
        "endpoint",
        "--phrase",
        "hit",
        "--name",
        "review",
    ]);
    let group = [
        "group",
        "--mark-set",
        "review",
        "--method",
        "character-trigram",
        "--radius",
        "1",
    ];
    let valid = s.json(&[&group[..], &["--history-group-column", "endpoint"]].concat());
    assert_eq!(valid["groups"][0]["occurrences"], 2);
    for options in [
        vec![
            "--history-text-column",
            "other",
            "--history-group-column",
            "endpoint",
        ],
        vec!["--history-group-column", "alias"],
    ] {
        assert_eq!(
            s.run(&[&group[..], &options].concat()).status.code(),
            Some(4)
        );
    }
    let redaction = s.run(&[
        "redact",
        "--mark-set",
        "review",
        "--history-text-column",
        "other",
    ]);
    assert_eq!(redaction.status.code(), Some(4));
    let marks = s.0.join("project/marks/review.json");
    let mut legacy: Value = serde_json::from_slice(&fs::read(&marks).unwrap()).unwrap();
    legacy["source"]
        .as_object_mut()
        .unwrap()
        .remove("interpretation");
    fs::write(&marks, serde_json::to_vec(&legacy).unwrap()).unwrap();
    let refused = s.run(&[&group[..], &["--history-group-column", "endpoint"]].concat());
    assert_eq!(refused.status.code(), Some(4));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("mark"));
}

#[test]
fn remediation_archive_rechecks_the_record_length_used_for_review() {
    let s = Scratch::new("archive-interpretation");
    let zip = s.zip();
    s.json(&[
        "mark",
        "--input",
        &zip,
        "--phrase",
        "hit",
        "--name",
        "review",
        "--max-record-bytes",
        "20",
    ]);
    let group = [
        "group",
        "--mark-set",
        "review",
        "--method",
        "character-trigram",
        "--radius",
        "1",
    ];
    let v = s.json(&[&group[..], &["--history-max-record-bytes", "20"]].concat());
    assert_eq!(v["selected"], 3);
    assert_eq!(
        s.run(&[&group[..], &["--history-max-record-bytes", "7"]].concat())
            .status
            .code(),
        Some(4)
    );
}

#[test]
fn remediation_store_group_warning_and_named_mark_lifecycle() {
    use l2d_core::{EncoderIdentity, Label, LogRecord, LogsFrame};
    use l2d_store::{LogFlags, Store, VectorOrigin, rows_frame};
    let s = Scratch::new("stored-group-warning");
    let endpoints: Vec<String> = (0..30)
        .map(|i| {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(format!("endpoint-{i}")))[..10].to_owned()
        })
        .collect();
    let mut records = Vec::new();
    for endpoint in &endpoints {
        for i in 0..18 {
            records.push(LogRecord {
                text: format!("GET /{endpoint}/message?id={i}"),
                group: Some(endpoint.clone()),
                label: Label::Unknown,
                first_position: records.len() as u64 + 1,
                occurrences: 1,
            });
        }
    }
    let mut budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let identity = EncoderIdentity {
        name: "fixture".into(),
        revision: "a".repeat(64),
        dimension: 2,
    };
    let origin = VectorOrigin::without_release(&identity);
    let mut store = Store::create(&ProjectPaths::new(s.0.join("project")), "logs", origin).unwrap();
    let logs = LogsFrame::from_records(&records).unwrap();
    let rows = rows_frame(
        &logs,
        "fixture",
        &vec![1.; records.len() * 2],
        2,
        &vec![LogFlags::default(); records.len()],
    )
    .unwrap();
    store.add_with(&rows, origin, &mut budget).unwrap();
    drop(store);
    let pattern = endpoints[..15].join("|");
    let mark = [
        "mark", "--store", "logs", "--regex", &pattern, "--name", "review",
    ];
    let matches = s.json(&mark);
    let fit = s.json(&[
        "score",
        "fit",
        "--mark-set",
        "review",
        "--name",
        "score",
        "--margin",
        "0",
    ]);
    let report = &fit["held_out_report"];
    assert!(report["difference"].as_f64().unwrap() > 0.);
    assert!(
        report["warning"]
            .as_str()
            .unwrap()
            .contains(report["group_key"].as_str().unwrap()),
        "{report}"
    );
    let id = matches["matches"][0]["id"].as_str().unwrap();
    s.json(&[
        "group",
        "--mark-set",
        "review",
        "--method",
        "character-trigram",
        "--radius",
        "1",
        "--deselect-id",
        id,
    ]);
    let path = s.0.join("project/marks/review.json");
    let saved = fs::read(&path).unwrap();
    let _ = s.run(&mark);
    assert_eq!(saved, fs::read(path).unwrap());
}

#[test]
fn remediation_repeating_input_marks_preserves_confirmed_rules() {
    let s = Scratch::new("repeat-marks");
    let zip = s.zip();
    let file = s.0.join("lines.log");
    fs::write(&file, "hit one\nhit two\n").unwrap();
    for (name, input) in [("file", file.to_str().unwrap()), ("archive", &zip)] {
        let mark = ["mark", "--input", input, "--phrase", "hit", "--name", name];
        s.json(&mark);
        s.json(&mark);
        let rules = s.0.join("rules.json");
        fs::write(
            &rules,
            r#"[{"kind":"literal","fragment":"hit","placeholder":""}]"#,
        )
        .unwrap();
        s.json(&[
            "redact",
            "--mark-set",
            name,
            "--rules",
            rules.to_str().unwrap(),
        ]);
        let p = s.0.join(format!("project/marks/{name}.json"));
        let saved = fs::read(&p).unwrap();
        let _ = s.run(&mark);
        assert_eq!(saved, fs::read(p).unwrap());
    }
}

fn in_directory(scratch: &Scratch, cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_l2d"))
        .current_dir(cwd)
        .arg("--project")
        .arg(scratch.0.join("project"))
        .arg("--json")
        .args(args)
        .output()
        .unwrap()
}

fn successful(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn saved_source_paths_survive_a_different_working_directory() {
    let s = Scratch::new("relocation");
    let elsewhere = s.0.join("elsewhere");
    fs::create_dir_all(&elsewhere).unwrap();
    fs::write(
        s.0.join("rows.csv"),
        "text,label\nhit one,bad\nhit two,bad\n",
    )
    .unwrap();
    s.zip();
    for (source, name) in [("rows.csv", "rows"), ("logs.zip", "archive")] {
        let mark = successful(in_directory(
            &s,
            &s.0,
            &["mark", "--input", source, "--phrase", "hit", "--name", name],
        ));
        let group = [
            "group",
            "--mark-set",
            name,
            "--method",
            "character-trigram",
            "--radius",
            "1",
        ];
        assert_eq!(
            successful(in_directory(&s, &s.0, &group))["selected"],
            mark["count"]
        );
        assert_eq!(
            successful(in_directory(&s, &elsewhere, &group))["selected"],
            mark["count"]
        );
        // A source replacement still fails even though the path is now anchored.
        fs::write(s.0.join(source), "different bytes").unwrap();
        assert!(!in_directory(&s, &elsewhere, &group).status.success());
    }
}

#[test]
fn saved_source_paths_pair_relative_and_absolute_marks_for_the_same_input() {
    let s = Scratch::new("paired-paths");
    let alias = s.0.join("elsewhere");
    fs::create_dir_all(&alias).unwrap();
    let source = s.0.join("rows.csv");
    let mut csv = "text,label\n".to_owned();
    for i in 0..10 {
        csv.push_str(&format!(
            "request denied endpoint {i},bad\nrequest accepted endpoint {i},good\n"
        ));
    }
    fs::write(&source, &csv).unwrap();
    let first = successful(in_directory(
        &s,
        &s.0,
        &[
            "mark",
            "--input",
            "./rows.csv",
            "--phrase",
            "denied",
            "--name",
            "bad",
        ],
    ));
    let other = alias.join("..").join("rows.csv");
    let good = successful(in_directory(
        &s,
        &alias,
        &[
            "mark",
            "--input",
            other.to_str().unwrap(),
            "--phrase",
            "accepted",
            "--label",
            "good",
            "--name",
            "good",
        ],
    ));
    let fit = [
        "score",
        "fit",
        "--mark-set",
        "bad",
        "--other-mark-set",
        "good",
        "--name",
        "score",
    ];
    assert_eq!(
        successful(in_directory(&s, &s.0, &fit))["counts"]["marked"],
        10
    );
    assert_eq!(first["source"], good["source"]);
    // Old absolute spellings retain enough context to verify the same file.
    let saved = s.0.join("project/marks/good.json");
    let mut legacy: Value = serde_json::from_slice(&fs::read(&saved).unwrap()).unwrap();
    legacy["source"]["path"] = Value::String(other.to_str().unwrap().to_owned());
    fs::write(&saved, serde_json::to_vec(&legacy).unwrap()).unwrap();
    assert_eq!(
        successful(in_directory(&s, &alias, &fit))["counts"]["marked"],
        10
    );
    // A legacy relative name does not establish which directory held the original.
    legacy["source"]["path"] = Value::String("rows.csv".into());
    fs::write(&saved, serde_json::to_vec(&legacy).unwrap()).unwrap();
    let refused = in_directory(
        &s,
        &s.0,
        &[
            "group",
            "--mark-set",
            "good",
            "--method",
            "character-trigram",
            "--radius",
            "1",
        ],
    );
    assert_eq!(refused.status.code(), Some(4));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("working directory"));
    // A byte-identical copy is still a different source.
    fs::write(alias.join("copy.csv"), csv).unwrap();
    successful(in_directory(
        &s,
        &alias,
        &[
            "mark", "--input", "copy.csv", "--phrase", "accepted", "--label", "good", "--name",
            "copy",
        ],
    ));
    assert_eq!(
        in_directory(
            &s,
            &alias,
            &[
                "score",
                "fit",
                "--mark-set",
                "bad",
                "--other-mark-set",
                "copy",
                "--name",
                "different"
            ]
        )
        .status
        .code(),
        Some(4)
    );
}

#[test]
fn saved_source_paths_anchor_discovery_files_and_archives() {
    let s = Scratch::new("discovery-paths");
    let cwd = s.0.join("elsewhere");
    fs::create_dir_all(&cwd).unwrap();
    fs::write(s.0.join("rows.log"), "hit one\nhit two\n").unwrap();
    s.zip();
    for (source, name) in [("rows.log", "file"), ("logs.zip", "zip")] {
        let bundle = s.0.join(name);
        successful(in_directory(
            &s,
            &s.0,
            &[
                "discover",
                "search",
                "--input",
                source,
                "--phrase",
                "hit",
                "--out",
                bundle.to_str().unwrap(),
            ],
        ));
        let selection: Value =
            serde_json::from_slice(&fs::read(bundle.join("selection.json")).unwrap()).unwrap();
        let recorded = Path::new(selection["source"]["path"].as_str().unwrap());
        assert!(recorded.is_absolute(), "{recorded:?}");
        assert_eq!(
            recorded.canonicalize().unwrap(),
            s.0.join(source).canonicalize().unwrap()
        );
        let artifact = bundle.join("refinement.json");
        let out = cwd.join(name);
        successful(in_directory(
            &s,
            &cwd,
            &[
                "discover",
                "explore",
                "--artifact",
                artifact.to_str().unwrap(),
                "--exemplar-row",
                "0",
                "--method",
                "character-trigram",
                "--out",
                out.to_str().unwrap(),
            ],
        ));
    }
}
