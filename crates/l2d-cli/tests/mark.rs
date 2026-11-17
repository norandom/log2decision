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
