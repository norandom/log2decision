//! The wired commands of task 1.11, run as the built binary against the mini fixture in
//! a temporary project: `l2d ingest`, `l2d baseline fit`, `l2d baseline score`,
//! `l2d calibrate`, `l2d triage`.
//!
//! Requirements 1.1, 1.2, 1.6, 2.1, 2.2, 6.4 to 6.6, 10.3 to 10.7, 11.1, 11.5 to 11.7,
//! 11.9, 11.10, 12.2, 12.3, 12.6, 12.8.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::Value;

const NOTE: &str = "The error rate holds only for logs that resemble the calibration logs";

// --- Helpers -------------------------------------------------------------------------------

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv")
}

/// A temporary project root with the fixture cut into three files, removed when dropped.
struct Project {
    root: PathBuf,
}

impl Project {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-cli-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("in")).unwrap();
        let project = Self { root };
        // Data row i (from 1) goes to part i mod 3: 1 fit, 2 calibration, 0 triage.
        let fixture = std::fs::read_to_string(fixture_path()).unwrap();
        let mut lines = fixture.lines();
        let header = lines.next().unwrap();
        let mut parts = [
            vec![header.to_string()],
            vec![header.to_string()],
            vec![header.to_string()],
        ];
        for (index, line) in lines.enumerate() {
            // The group key becomes the method alone (GET, POST): two groups, large
            // enough for lists of their own in a third of the fixture. The malformed
            // row keeps its two fields.
            let line = match line.rsplit_once(',') {
                Some((head, group)) => {
                    format!("{head},{}", group.split(' ').next().unwrap_or(group))
                }
                None => line.to_string(),
            };
            parts[(index + 1) % 3].push(line);
        }
        let [triage, fit, calibration] = parts;
        project.write("fit.csv", &fit);
        project.write("calibration.csv", &calibration);
        project.write("triage.csv", &triage);
        project
    }

    fn write(&self, name: &str, lines: &[String]) -> PathBuf {
        let path = self.input(name);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        path
    }

    /// A file outside the project's data directory, given to the tool as a path.
    fn input(&self, name: &str) -> PathBuf {
        self.root.join("in").join(name)
    }

    fn arg(&self, name: &str) -> String {
        self.input(name).to_string_lossy().into_owned()
    }

    fn project(&self) -> PathBuf {
        self.root.join("project")
    }

    fn run(&self, args: &[&str]) -> Output {
        std::fs::create_dir_all(self.project()).unwrap();
        Command::new(env!("CARGO_BIN_EXE_l2d"))
            .arg("--project")
            .arg(self.project())
            .args(args)
            .output()
            .expect("the l2d binary runs")
    }

    /// Runs a command that must succeed with `--json` and returns its document.
    fn json(&self, args: &[&str]) -> Value {
        let mut all = vec!["--json"];
        all.extend_from_slice(args);
        let output = self.run(&all);
        assert_eq!(
            output.status.code(),
            Some(0),
            "l2d {}: {}",
            args.join(" "),
            stderr(&output)
        );
        assert_eq!(stderr(&output), "", "l2d {}", args.join(" "));
        let out = stdout(&output);
        assert!(out.ends_with('\n'));
        serde_json::from_str(&out)
            .unwrap_or_else(|error| panic!("l2d {}: {error}: {out}", args.join(" ")))
    }

    /// Runs a command that must succeed and returns its text output.
    fn text(&self, args: &[&str]) -> String {
        let output = self.run(args);
        assert_eq!(
            output.status.code(),
            Some(0),
            "l2d {}: {}",
            args.join(" "),
            stderr(&output)
        );
        assert_eq!(stderr(&output), "", "l2d {}", args.join(" "));
        stdout(&output)
    }

    /// Runs a command that must fail with `code` and returns its one line of
    /// diagnostics. Nothing may be printed on standard output.
    fn fails(&self, args: &[&str], code: i32) -> String {
        let output = self.run(args);
        assert_eq!(
            output.status.code(),
            Some(code),
            "l2d {}: {}{}",
            args.join(" "),
            stdout(&output),
            stderr(&output)
        );
        assert_eq!(stdout(&output), "", "l2d {}", args.join(" "));
        let err = stderr(&output);
        let lines: Vec<&str> = err.lines().collect();
        assert_eq!(lines.len(), 1, "one line states what failed: {err:?}");
        assert!(lines[0].starts_with("error: "), "{err}");
        lines[0].to_string()
    }

    /// Fits the baseline `base` on the fit part (keep mode, small feature space).
    fn fit(&self) -> Value {
        let fit = self.arg("fit.csv");
        self.json(&[
            "baseline",
            "fit",
            "--input",
            &fit,
            "--group-column",
            "group",
            "--name",
            "base",
            "--buckets",
            "1024",
        ])
    }

    /// Calibrates `base` on the calibration part as `cal`.
    fn calibrate(&self) -> Value {
        let calibration = self.arg("calibration.csv");
        self.json(&[
            "calibrate",
            "--input",
            &calibration,
            "--group-column",
            "group",
            "--baseline",
            "base",
            "--name",
            "cal",
            "--min-per-group",
            "10",
            "--error-rate",
            "0.1",
        ])
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

fn rows<'a>(value: &'a Value, key: &str) -> &'a Vec<Value> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("`{key}` is a list in {value}"))
}

fn count(value: &Value) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("a count, got {value}"))
}

/// Every file below `dir`, relative to it, with forward slashes.
fn files_below(dir: &Path) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                found.insert(
                    path.strip_prefix(dir)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    found
}

// --- l2d ingest ----------------------------------------------------------------------------

#[test]
fn ingesting_the_fixture_displays_the_summary() {
    let project = Project::new("ingest");
    let fixture = fixture_path().to_string_lossy().into_owned();
    let args = ["ingest", "--input", &fixture, "--group-column", "group"];

    let text = project.text(&args);
    for expected in [
        "records read: 200 (1 skipped)",
        "distinct texts: 169 (96 good, 72 bad, 1 unknown)",
        "texts with conflicting labels: 1",
        "group keys: 8",
        "content SHA-256: 25156ef523229c5933659f205ec3ffe3c12c28e9500071bfd0ec32efcef8f0a6",
        "position 121:",
        "adapter: csv",
        "memory: free memory ",
        "memory limit ",
        "reserve ",
        "reading the input: 200 records in batches of at most ",
        " rows (chosen from the free memory)",
        "the batch size did not change",
    ] {
        assert!(text.contains(expected), "`{expected}` missing in:\n{text}");
    }
    assert!(!text.contains("lines"), "records are not lines:\n{text}");

    let value = project.json(&args);
    assert_eq!(value["command"], "ingest");
    let input = &value["input"];
    assert_eq!(input["adapter"], "csv");
    assert_eq!(input["records"], 200);
    assert_eq!(input["distinct_texts"], 169);
    assert_eq!(input["label_conflicts"], 1);
    assert_eq!(input["skipped_count"], 1);
    assert_eq!(input["skipped"][0]["position"], 121);
    assert!(input["skipped"][0]["reason"].is_string());
    assert_eq!(input["labels"]["good"], 96);
    assert_eq!(input["labels"]["bad"], 72);
    assert_eq!(input["labels"]["unknown"], 1);
    assert_eq!(input["group_keys"], 8);
    assert_eq!(
        input["content_sha256"],
        "25156ef523229c5933659f205ec3ffe3c12c28e9500071bfd0ec32efcef8f0a6"
    );
    assert!(
        input["source_path"]
            .as_str()
            .unwrap()
            .ends_with("mini-labelled.csv")
    );

    // Requirement 12.3 and 12.8: what was observed, applied and used.
    let memory = &value["memory"];
    let free = count(&memory["free_bytes"]);
    let limit = count(&memory["limit_bytes"]);
    assert!(free > 0 && limit > 0 && limit <= free);
    assert!(count(&memory["reserve_bytes"]) > 0);
    assert_eq!(memory["limit_set_by_analyst"], false);
    assert_eq!(memory["limit_share"], 0.5);
    assert_eq!(memory["reserve_share"], 0.25);
    let read = &memory["operations"][0];
    assert_eq!(read["operation"], "read");
    assert_eq!(read["batches"]["free_bytes"], free);
    assert_eq!(read["batches"]["limit_bytes"], limit);
    assert!(count(&read["batches"]["batch_rows"]) >= 200);
    assert!(count(&read["batches"]["min_rows_used"]) > 0);
    assert_eq!(
        read["batches"]["min_rows_used"],
        read["batches"]["max_rows_used"]
    );
    assert!(read["batches"]["changes"].as_array().unwrap().is_empty());

    // Nothing is written by reading.
    assert!(files_below(&project.project()).is_empty());
}

#[test]
fn an_input_is_read_from_the_path_given_or_from_the_data_directory() {
    let project = Project::new("data-dir");
    let data = project.project().join("data").join("mini");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::copy(fixture_path(), data.join("logs.csv")).unwrap();
    let value = project.json(&["ingest", "--input", "mini/logs.csv"]);
    assert_eq!(value["input"]["records"], 200);
    assert_eq!(value["input"]["group_keys"], 0, "no group column was named");

    // A plain text file: one log per line, unlabelled, positions are line numbers.
    let lines = project.write(
        "plain.log",
        &[
            "GET /a".to_string(),
            String::new(),
            "GET /b".to_string(),
            "GET /a".to_string(),
        ],
    );
    let value = project.json(&["ingest", "--input", &lines.to_string_lossy()]);
    assert_eq!(value["input"]["adapter"], "lines");
    assert_eq!(value["input"]["records"], 4);
    assert_eq!(value["input"]["distinct_texts"], 2);
    assert_eq!(value["input"]["labels"]["unknown"], 2);
    assert_eq!(value["input"]["skipped"][0]["position"], 2);

    // Requirement 10.5: what is missing, and where it was expected.
    let line = project.fails(&["ingest", "--input", "mini/absent.csv"], 3);
    assert!(
        line.contains("mini/absent.csv") || line.contains("mini\\absent.csv"),
        "{line}"
    );
    let expected = project.project().join("data");
    assert!(
        line.contains(&*expected.file_name().unwrap().to_string_lossy())
            && line.contains("expected at"),
        "{line}"
    );
    let absolute = project.arg("absent.csv");
    let line = project.fails(&["ingest", "--input", &absolute], 3);
    assert!(line.contains(&absolute), "{line}");

    // Analyst errors: exit code 2, naming the option.
    let fixture = fixture_path().to_string_lossy().into_owned();
    for (args, named) in [
        (
            &["ingest", "--input", &fixture, "--adapter", "xml"][..],
            "--adapter",
        ),
        (&["ingest"][..], "--input"),
        (
            &[
                "ingest",
                "--input",
                &fixture,
                "--adapter",
                "lines",
                "--group-column",
                "g",
            ][..],
            "--group-column",
        ),
        (
            &["ingest", "--input", &fixture, "--group-column", "nope"][..],
            "nope",
        ),
        (
            &[
                "ingest",
                "--input",
                &fixture,
                "--no-labels",
                "--label-column",
                "label",
            ][..],
            "--no-labels",
        ),
    ] {
        let line = project.fails(args, 2);
        assert!(line.starts_with("error: invalid input: "), "{line}");
        assert!(line.contains(named), "{}: {line}", args.join(" "));
    }
    let value = project.json(&["ingest", "--input", &fixture, "--no-labels"]);
    assert_eq!(value["input"]["labels"]["unknown"], 169);

    // A project directory that does not exist is named.
    let output = Command::new(env!("CARGO_BIN_EXE_l2d"))
        .args(["--project"])
        .arg(project.root.join("no-such-project"))
        .args(["ingest", "--input", &fixture])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    assert!(
        stderr(&output).contains("no-such-project"),
        "{}",
        stderr(&output)
    );
}

// --- Memory options (requirement 12.2, owner decision on the shares) ------------------------

#[test]
fn the_memory_shares_and_the_limit_are_applied_and_reported() {
    let project = Project::new("memory");
    let fixture = fixture_path().to_string_lossy().into_owned();
    let ingest = |options: &[&str]| {
        let mut args = options.to_vec();
        args.extend_from_slice(&["ingest", "--input", &fixture, "--group-column", "group"]);
        project.json(&args)
    };

    let value = ingest(&["--memory-share", "0.25", "--reserve-share", "0.125"]);
    let memory = &value["memory"];
    let free = count(&memory["free_bytes"]) as f64;
    assert_eq!(memory["limit_share"], 0.25);
    assert_eq!(memory["reserve_share"], 0.125);
    assert_eq!(count(&memory["limit_bytes"]), (free * 0.25) as u64);
    assert_eq!(count(&memory["reserve_bytes"]), (free * 0.125) as u64);
    assert_eq!(memory["limit_set_by_analyst"], false);

    let value = ingest(&["--memory-limit", "64MB"]);
    assert_eq!(value["memory"]["limit_bytes"], 64 * 1024 * 1024);
    assert_eq!(value["memory"]["limit_set_by_analyst"], true);
    assert!(value["memory"]["limit_share"].is_null());

    // The analyst's batch size is used and reported, and the result is the same (12.6).
    let default = ingest(&[]);
    let small = ingest(&["--batch-rows", "16"]);
    let batches = &small["memory"]["operations"][0]["batches"];
    assert_eq!(batches["batch_rows"], 16);
    assert_eq!(batches["max_rows_used"], 16);
    assert_eq!(batches["overridden"], true);
    assert_eq!(small["input"], default["input"]);
    let text = project.text(&[
        "--batch-rows",
        "16",
        "ingest",
        "--input",
        &fixture,
        "--group-column",
        "group",
    ]);
    assert!(
        text.contains("200 records in batches of at most 16 rows (set by the analyst)"),
        "{text}"
    );
    assert_eq!(small["memory"]["operations"][0]["rows"], 200);
}

#[test]
fn shares_outside_zero_to_one_and_a_reserve_that_leaves_nothing_are_rejected() {
    let project = Project::new("shares");
    let fixture = fixture_path().to_string_lossy().into_owned();
    for option in ["--memory-share", "--reserve-share"] {
        for value in ["0", "1", "1.5", "-0.1", "abc", "NaN", "inf", "50%", ""] {
            let line = project.fails(
                &["ingest", "--input", &fixture, &format!("{option}={value}")],
                2,
            );
            assert!(line.starts_with("error: invalid input: "), "{line}");
            assert!(line.contains(option), "{option}={value}: {line}");
        }
    }
    // A reserve of nearly all the free memory leaves nothing the tool could use.
    let line = project.fails(
        &[
            "ingest",
            "--input",
            &fixture,
            "--reserve-share",
            "0.9999999",
        ],
        2,
    );
    assert!(line.contains("--reserve-share"), "{line}");
    assert!(line.contains("leaves"), "{line}");

    // A share and a limit together: the share would mean nothing.
    let line = project.fails(
        &[
            "ingest",
            "--input",
            &fixture,
            "--memory-share",
            "0.3",
            "--memory-limit",
            "1GB",
        ],
        2,
    );
    assert!(
        line.contains("--memory-share") || line.contains("--memory-limit"),
        "{line}"
    );

    // A limit the tool itself uses up, and one that leaves nothing for a batch, say so
    // before anything is read.
    let line = project.fails(
        &["ingest", "--input", &fixture, "--memory-limit", "32MB"],
        2,
    );
    assert!(
        line.contains("--memory-limit") || line.contains("memory limit"),
        "{line}"
    );
    for limit in ["33MB", "35MB", "36MB", "37000KB"] {
        let line = project.fails(&["ingest", "--input", &fixture, "--memory-limit", limit], 2);
        assert!(line.contains("nothing for a batch"), "{limit}: {line}");
        assert!(!line.contains("larger than 1 B"), "{line}");
    }
    let value = project.json(&["ingest", "--input", &fixture, "--memory-limit", "37200KB"]);
    assert_eq!(value["input"]["records"], 200);

    // The options are on every command, stubs included.
    for command in [&["shift"][..], &["store", "add"][..], &["triage"][..]] {
        let mut args = command.to_vec();
        args.push("--help");
        let help = project.text(&args);
        for option in ["--memory-share <SHARE>", "--reserve-share <SHARE>"] {
            assert!(
                help.contains(option),
                "{option} missing in l2d {command:?} --help"
            );
        }
    }
}

// --- l2d baseline fit / score --------------------------------------------------------------

#[test]
fn fitting_then_scoring_displays_ratios() {
    let project = Project::new("baseline");
    let fit = project.arg("fit.csv");
    let text = project.text(&[
        "baseline",
        "fit",
        "--input",
        &fit,
        "--group-column",
        "group",
        "--name",
        "base",
        "--buckets",
        "1024",
    ]);
    for expected in [
        "baseline scorer: base",
        "revision: ",
        "saved to: ",
        "fitted on: ",
        "good, ",
        "text hashes saved: ",
        "settings: 1024 buckets, n-grams up to 3 bytes, routing keep",
        "memory: free memory ",
        "fitting: the whole dataset at once (",
    ] {
        assert!(text.contains(expected), "`{expected}` missing in:\n{text}");
    }

    // The same fit again: the same model in the same place (requirement 2.4).
    let value = project.fit();
    assert_eq!(value["command"], "baseline fit");
    let scorer = &value["scorer"];
    assert_eq!(scorer["kind"], "baseline");
    assert_eq!(scorer["name"], "base");
    let revision = scorer["revision"].as_str().unwrap();
    assert_eq!(revision.len(), 40);
    assert!(text.contains(revision), "{text}");
    let directory = PathBuf::from(scorer["directory"].as_str().unwrap());
    assert_eq!(
        directory.file_name().unwrap().to_string_lossy(),
        format!("base-{}", &revision[..12])
    );
    assert_eq!(scorer["settings"]["buckets"], 1024);
    assert_eq!(scorer["settings"]["routing"], "keep");
    let fitted = &value["fit"];
    let labelled = count(&fitted["good"]) + count(&fitted["bad"]);
    assert_eq!(count(&fitted["rows"]), labelled);
    assert_eq!(scorer["fitted_text_hashes"], labelled);
    assert!(fitted["termination"].is_string());
    assert!(fitted["gradient_converged"].is_boolean());
    assert_eq!(value["input"]["records"], 67);
    let fitting = &value["memory"]["operations"][1];
    assert_eq!(fitting["operation"], "fit");
    assert_eq!(fitting["whole_dataset_bytes"], fitted["estimated_bytes"]);

    // Requirement 10.3: the model is written below the project's model directory only.
    assert_eq!(
        files_below(&project.project()),
        BTreeSet::from(
            ["baseline.json", "fitted-texts.bin", "weights.bin"]
                .map(|file| format!("models/baselines/base-{}/{file}", &revision[..12]))
        )
    );

    // Scoring: a ratio between 0 and 1 for every distinct text, in input order.
    let triage = project.arg("triage.csv");
    let args = [
        "baseline",
        "score",
        "--input",
        &triage,
        "--group-column",
        "group",
        "--baseline",
        "base",
    ];
    let value = project.json(&args);
    assert_eq!(value["command"], "baseline score");
    assert_eq!(value["scorer"]["revision"], revision);
    assert_eq!(value["scorer"]["name"], "base");
    let scores = rows(&value, "scores");
    assert_eq!(
        scores.len() as u64,
        count(&value["input"]["distinct_texts"])
    );
    assert_eq!(value["scored"], scores.len());
    let mut positions = Vec::new();
    for row in scores {
        let ratio = row["ratio"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&ratio), "{row}");
        assert_eq!(row["scorer"], "baseline");
        assert_eq!(row["abstained"], false);
        assert_eq!(row["no_features"], false);
        assert!(row["text"].is_string() && row["label"].is_string());
        positions.push(count(&row["first_position"]));
    }
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "input order"
    );
    // The scorer separates the labels it was fitted for: bad logs score higher on average.
    let mean = |label: &str| {
        let ratios: Vec<f64> = scores
            .iter()
            .filter(|row| row["label"] == label)
            .map(|row| row["ratio"].as_f64().unwrap())
            .collect();
        ratios.iter().sum::<f64>() / ratios.len() as f64
    };
    assert!(
        mean("bad") > mean("good") + 0.05,
        "{} {}",
        mean("bad"),
        mean("good")
    );
    let scoring = &value["memory"]["operations"][1];
    assert_eq!(scoring["operation"], "score");
    assert!(count(&scoring["batches"]["max_rows_used"]) >= scores.len() as u64);

    // The same ratios whatever the batch size (12.6), with the batches reported.
    let batched = project.json(&[
        "--batch-rows",
        "7",
        "baseline",
        "score",
        "--input",
        &triage,
        "--group-column",
        "group",
        "--baseline",
        "base",
    ]);
    assert_eq!(batched["scores"], value["scores"]);
    assert_eq!(
        batched["memory"]["operations"][1]["batches"]["max_rows_used"],
        7
    );

    // Text: one line per log with its ratio, at most `--limit` of them.
    let text = project.text(&args);
    assert!(text.contains("scorer: baseline base"), "{text}");
    assert!(text.contains(&revision[..12]), "{text}");
    let shown: Vec<&str> = text
        .lines()
        .filter(|line| line.trim_start().starts_with("0.") || line.trim_start().starts_with("1.0"))
        .collect();
    assert_eq!(shown.len(), 20, "the default limit:\n{text}");
    assert!(
        text.contains(&format!("{} more", scores.len() - 20)),
        "{text}"
    );
    let text = project.text(&[
        "baseline",
        "score",
        "--input",
        &triage,
        "--group-column",
        "group",
        "--baseline",
        "base",
        "--limit",
        "3",
    ]);
    assert!(
        text.contains(&format!("{} more", scores.len() - 3)),
        "{text}"
    );
    let first = format!("{:.4}", scores[0]["ratio"].as_f64().unwrap());
    assert!(text.contains(&first), "{first} missing in:\n{text}");
    let limited = project.json(&[
        "baseline",
        "score",
        "--input",
        &triage,
        "--group-column",
        "group",
        "--baseline",
        "base",
        "--limit",
        "3",
    ]);
    assert_eq!(rows(&limited, "scores").len(), 3);
    assert_eq!(limited["scored"], scores.len());
}

#[test]
fn fit_and_score_failures_state_what_failed() {
    let project = Project::new("baseline-failures");
    let fit = project.arg("fit.csv");
    // Requirement 2.3: too few labelled logs, with the counts found.
    let plain = project.write("plain.log", &["GET /a".to_string(), "GET /b".to_string()]);
    let line = project.fails(
        &[
            "baseline",
            "fit",
            "--input",
            &plain.to_string_lossy(),
            "--name",
            "x",
        ],
        4,
    );
    assert!(line.contains("0 good, 0 bad"), "{line}");
    // A name that could leave the model directory is an analyst error, found before
    // anything is read.
    for name in ["../x", "a/b", ".hidden", ""] {
        let line = project.fails(
            &[
                "baseline",
                "fit",
                "--input",
                "no-such-input.csv",
                "--name",
                name,
            ],
            2,
        );
        assert!(line.contains("name"), "{name}: {line}");
    }
    for (option, value) in [("--routing", "drop"), ("--buckets", "0"), ("--l2", "-1")] {
        let line = project.fails(
            &[
                "baseline", "fit", "--input", &fit, "--name", "x", option, value,
            ],
            2,
        );
        assert!(line.starts_with("error: invalid input: "), "{line}");
    }
    assert!(files_below(&project.project()).is_empty());

    // Requirement 10.5: a baseline that is not there, with its expected location.
    let line = project.fails(
        &["baseline", "score", "--input", &fit, "--baseline", "base"],
        3,
    );
    assert!(line.contains("baseline scorer 'base'"), "{line}");
    assert!(line.contains("baselines"), "{line}");
    assert!(line.contains("expected at"), "{line}");
    let line = project.fails(
        &[
            "baseline",
            "score",
            "--input",
            &fit,
            "--baseline",
            "../base",
        ],
        2,
    );
    assert!(line.contains("--baseline"), "{line}");

    // With `--json` the failure is one document on standard output.
    let output = project.run(&[
        "--json",
        "baseline",
        "score",
        "--input",
        &fit,
        "--baseline",
        "b",
    ]);
    assert_eq!(output.status.code(), Some(3));
    let value: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(value["error"]["kind"], "missing");
    assert_eq!(value["error"]["exit_code"], 3);
    assert_eq!(stderr(&output).lines().count(), 1);

    // Two fits under one name: the name alone no longer says which.
    project.fit();
    let calibration = project.arg("calibration.csv");
    project.json(&[
        "baseline",
        "fit",
        "--input",
        &calibration,
        "--group-column",
        "group",
        "--name",
        "base",
        "--buckets",
        "1024",
    ]);
    let line = project.fails(
        &["baseline", "score", "--input", &fit, "--baseline", "base"],
        2,
    );
    assert!(line.contains("2 baseline scorers"), "{line}");
    assert_eq!(line.matches("base-").count(), 2, "{line}");
}

// --- l2d calibrate -------------------------------------------------------------------------

/// The texts of a part and their labels, as `l2d baseline score` lists them.
fn texts_of(project: &Project, part: &str) -> Vec<(String, String)> {
    let path = project.arg(part);
    let value = project.json(&["baseline", "score", "--input", &path, "--baseline", "base"]);
    rows(&value, "scores")
        .iter()
        .map(|row| {
            (
                row["text"].as_str().unwrap().to_string(),
                row["label"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

#[test]
fn calibrating_on_a_held_out_part_of_the_fixture_succeeds() {
    let project = Project::new("calibrate");
    let fitted = project.fit();
    let revision = fitted["scorer"]["revision"].as_str().unwrap();

    // What the check per log must find: texts of the calibration part that are also
    // labelled texts of the fit part (the fixture repeats 27 texts).
    let fit_texts: BTreeSet<String> = texts_of(&project, "fit.csv")
        .into_iter()
        .filter(|(_, label)| label != "unknown")
        .map(|(text, _)| text)
        .collect();
    let calibration_texts = texts_of(&project, "calibration.csv");
    let shared = calibration_texts
        .iter()
        .filter(|(text, _)| fit_texts.contains(text))
        .count();
    assert!(shared > 0, "the parts share repeated texts");
    let kept_labelled = calibration_texts
        .iter()
        .filter(|(text, label)| !fit_texts.contains(text) && label != "unknown")
        .count();

    let value = project.calibrate();
    assert_eq!(value["command"], "calibrate");
    let held_out = &value["held_out"];
    assert_eq!(held_out["check"], "per_log");
    assert_eq!(held_out["logs"], calibration_texts.len());
    assert_eq!(held_out["dropped"], shared);
    assert_eq!(held_out["kept"], calibration_texts.len() - shared);
    assert!(held_out["note"].is_null());

    let calibration = &value["calibration"];
    assert_eq!(calibration["name"], "cal");
    let calibration_revision = calibration["revision"].as_str().unwrap();
    assert_eq!(calibration["scorer"]["kind"], "baseline");
    assert_eq!(calibration["scorer"]["revision"], revision);
    assert_eq!(calibration["min_per_group"], 10);
    assert_eq!(
        count(&calibration["good"]) + count(&calibration["bad"]),
        kept_labelled as u64
    );
    let strata = rows(calibration, "strata");
    assert_eq!(strata[0]["stratum"], "pooled");
    assert_eq!(strata[0]["logs"], calibration["good"]);
    assert!(
        strata.iter().any(|stratum| stratum["stratum"] == "group"
            && stratum["group"].is_string()
            && count(&stratum["logs"]) >= 10),
        "{strata:?}"
    );
    assert_eq!(value["error_rate"], 0.1);
    assert!(
        calibration["smallest_supported_error_rate"]
            .as_f64()
            .unwrap()
            <= 0.1
    );
    assert!(value["note"].as_str().unwrap().starts_with(NOTE));

    // Requirement 10.3: written below the model directory, beside the scorer.
    let files = files_below(&project.project());
    for file in ["calibration.json", "scores.bin"] {
        let path = format!(
            "models/calibrations/cal-{}/{file}",
            &calibration_revision[..12]
        );
        assert!(files.contains(&path), "{path} missing in {files:?}");
    }
    assert_eq!(files.len(), 5);

    // The same calibration on repeat (requirement 11.10), and the text form.
    let calibration_path = project.arg("calibration.csv");
    let text = project.text(&[
        "calibrate",
        "--input",
        &calibration_path,
        "--group-column",
        "group",
        "--baseline",
        "base",
        "--name",
        "cal",
        "--min-per-group",
        "10",
        "--error-rate",
        "0.1",
    ]);
    for expected in [
        format!("calibration: cal\n  revision: {calibration_revision}"),
        format!("for the baseline scorer base, revision {revision}"),
        format!(
            "held-out check per log: {shared} of {} calibration logs left out",
            calibration_texts.len()
        ),
        "smallest error rate supported: ".to_string(),
        "error rate 0.1: supported".to_string(),
        "pooled".to_string(),
        NOTE.to_string(),
        "calibrating: the whole dataset at once".to_string(),
    ] {
        assert!(text.contains(&expected), "`{expected}` missing in:\n{text}");
    }
}

#[test]
fn calibration_refuses_logs_the_scorer_was_fitted_on() {
    let project = Project::new("calibrate-refusals");
    project.fit();
    let fit = project.arg("fit.csv");
    let calibrate = |input: &str, more: &[&str]| {
        let mut args = vec![
            "calibrate",
            "--input",
            input,
            "--group-column",
            "group",
            "--baseline",
            "base",
            "--name",
            "cal",
        ];
        args.extend_from_slice(more);
        args.into_iter().map(str::to_string).collect::<Vec<_>>()
    };
    let fails = |args: Vec<String>, code: i32| {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        project.fails(&args, code)
    };

    // The fitting file itself: the whole-file check.
    let line = fails(calibrate(&fit, &[]), 4);
    assert!(line.contains("an input the scorer was fitted on"), "{line}");

    // The same logs in another order: other bytes, so only the check per log finds them.
    let text = std::fs::read_to_string(project.input("fit.csv")).unwrap();
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    lines[1..].reverse();
    let reordered = project.write("reordered.csv", &lines);
    let line = fails(calibrate(&reordered.to_string_lossy(), &[]), 4);
    assert!(line.contains("left out"), "{line}");
    assert!(line.contains("the scorer was fitted on"), "{line}");
    assert!(line.contains("too few"), "{line}");

    // Too few left for the error rate: the counts and the smallest supported rate.
    let calibration = project.arg("calibration.csv");
    let line = fails(calibrate(&calibration, &["--error-rate", "0.001"]), 4);
    assert!(line.contains("too few"), "{line}");
    assert!(line.contains("good") && line.contains("bad"), "{line}");
    assert!(line.contains("support error rates from"), "{line}");
    assert!(line.contains("left out"), "{line}");

    // Analyst errors.
    for (more, named) in [
        (&["--error-rate", "0"][..], "error rate"),
        (&["--error-rate", "1.5"][..], "error rate"),
        (&["--min-per-group", "0"][..], "--min-per-group"),
    ] {
        let line = fails(calibrate(&calibration, more), 2);
        assert!(line.contains(named), "{line}");
    }
    let line = project.fails(
        &[
            "calibrate",
            "--input",
            &calibration,
            "--baseline",
            "base",
            "--name",
            "../c",
        ],
        2,
    );
    assert!(line.contains("name"), "{line}");
    // The store has no vectors without an encoder: said plainly, not as a usage error.
    let line = project.fails(&["calibrate", "--store", "s", "--name", "c"], 4);
    assert!(
        line.contains("--store") && line.contains("encoder"),
        "{line}"
    );

    // Nothing but the scorer was written.
    assert_eq!(files_below(&project.project()).len(), 3);
}

#[test]
fn a_scorer_saved_without_text_hashes_falls_back_to_the_whole_file_check() {
    let project = Project::new("earlier-scorer");
    // The scorer saved by the code before task 1.11 (fitted on the whole fixture).
    let saved = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("l2d-baseline")
        .join("tests")
        .join("fixtures")
        .join("scorer-1.7")
        .join("models")
        .join("baselines")
        .join("before-1.11-53c19878ccb7");
    let target = project
        .project()
        .join("models")
        .join("baselines")
        .join("before-1.11-53c19878ccb7");
    std::fs::create_dir_all(&target).unwrap();
    for file in ["baseline.json", "weights.bin"] {
        std::fs::copy(saved.join(file), target.join(file)).unwrap();
    }

    // The fitting file itself is refused.
    let fixture = fixture_path().to_string_lossy().into_owned();
    let args = |input: &str| {
        vec![
            "calibrate".to_string(),
            "--input".into(),
            input.to_string(),
            "--group-column".into(),
            "group".into(),
            "--baseline".into(),
            "before-1.11".into(),
            "--name".into(),
            "cal".into(),
        ]
    };
    let owned = args(&fixture);
    let line = project.fails(&owned.iter().map(String::as_str).collect::<Vec<_>>(), 4);
    assert!(line.contains("an input the scorer was fitted on"), "{line}");

    // A part of it under other bytes is not detected, and the command says so.
    let owned = args(&project.arg("calibration.csv"));
    let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
    let value = project.json(&borrowed);
    assert_eq!(value["held_out"]["check"], "whole_file");
    assert_eq!(value["held_out"]["dropped"], 0);
    let note = value["held_out"]["note"].as_str().unwrap();
    assert!(note.contains("identical"), "{note}");
    assert_eq!(value["scorer"]["fitted_text_hashes"], Value::Null);
    let text = project.text(&borrowed);
    assert!(text.contains("held-out check by whole file only"), "{text}");
    assert!(text.contains(note), "{text}");

    // The earlier scorer scores and triages like any other.
    let triage = project.arg("triage.csv");
    let value = project.json(&[
        "triage",
        "--input",
        &triage,
        "--baseline",
        "before-1.11",
        "--calibration",
        "cal",
    ]);
    assert_eq!(
        value["scorers"][0]["revision"],
        "53c19878ccb777b54e64170818dec20fee792a77"
    );
}

// --- l2d triage ----------------------------------------------------------------------------

const ROW_KEYS: [&str; 18] = [
    "row",
    "text",
    "label",
    "group",
    "first_position",
    "occurrences",
    "scorer",
    "ratio",
    "abstained",
    "list",
    "rank",
    "scored_down_by",
    "abstain_reason",
    "label_set",
    "p_good",
    "p_bad",
    "good_stratum",
    "bad_stratum",
];

fn assert_row_shape(row: &Value) {
    let object = row.as_object().unwrap();
    for key in ROW_KEYS {
        assert!(object.contains_key(key), "`{key}` missing in {row}");
    }
    assert!(object["ratios"].is_object(), "{row}");
    assert!(object["evidence"].is_object(), "{row}");
    assert_eq!(object.len(), ROW_KEYS.len() + 2, "{row}");
}

#[test]
fn triage_without_a_calibration_ranks_and_cuts_at_the_budget() {
    let project = Project::new("triage-plain");
    let fitted = project.fit();
    let triage = project.arg("triage.csv");
    let base = [
        "triage",
        "--input",
        &triage,
        "--group-column",
        "group",
        "--baseline",
        "base",
    ];

    let value = project.json(&base);
    assert_eq!(value["command"], "triage");
    assert_eq!(value["primary"], "baseline");
    assert_eq!(value["scorers"][0]["kind"], "baseline");
    assert_eq!(value["scorers"][0]["name"], "base");
    assert_eq!(
        value["scorers"][0]["revision"],
        fitted["scorer"]["revision"]
    );
    assert!(value["calibration"].is_null());
    assert!(value["review_budget"].is_null());
    assert_eq!(
        value["evidence_columns"],
        serde_json::json!(["no_features"])
    );
    let logs = count(&value["counts"]["logs"]);
    assert_eq!(logs, count(&value["input"]["distinct_texts"]));
    let review = rows(&value, "review");
    assert_eq!(review.len() as u64, logs);
    assert!(rows(&value, "scored_down").is_empty());
    assert!(rows(&value, "abstained").is_empty());
    let mut last = f64::INFINITY;
    for (index, row) in review.iter().enumerate() {
        assert_row_shape(row);
        assert_eq!(row["rank"], index + 1);
        assert_eq!(row["list"], "review");
        assert_eq!(row["scorer"], "baseline");
        assert!(row["label_set"].is_null() && row["p_good"].is_null());
        assert_eq!(row["evidence"]["no_features"], false);
        let ratio = row["ratio"].as_f64().unwrap();
        assert!(ratio <= last, "highest ratio first");
        last = ratio;
    }
    let triaging = &value["memory"]["operations"][1];
    assert_eq!(triaging["operation"], "triage");
    assert!(count(&triaging["batches"]["max_rows_used"]) >= logs);

    // Requirement 6.5: the budget keeps that many and marks the rest as scored down.
    let mut args = base.to_vec();
    args.extend_from_slice(&["--budget", "10"]);
    let budgeted = project.json(&args);
    assert_eq!(budgeted["review_budget"], 10);
    assert_eq!(rows(&budgeted, "review")[..], review[..10]);
    let scored_down = rows(&budgeted, "scored_down");
    assert_eq!(scored_down.len() as u64, logs - 10);
    for (row, ranked) in scored_down.iter().zip(&review[10..]) {
        assert_eq!(row["scored_down_by"], "budget_only");
        assert_eq!(row["text"], ranked["text"]);
        assert_eq!(row["rank"], ranked["rank"]);
    }
    assert_eq!(budgeted["counts"]["review"], 10);
    assert_eq!(budgeted["counts"]["scored_down"], logs - 10);
    assert_eq!(budgeted["counts"]["scored_down_by_budget"], logs - 10);
    assert_eq!(budgeted["counts"]["scored_down_by_calibration"], 0);

    // The text form: the three lists, the scorer, the counts; `--limit` rows per list.
    args.extend_from_slice(&["--limit", "4"]);
    let text = project.text(&args);
    for expected in [
        format!("triage of {logs} logs by the baseline scorer base"),
        "review budget: 10 logs".to_string(),
        format!("counts: 10 for review, {} scored down", logs - 10),
        "review (10 logs, highest ratio first; the first 4 shown)".to_string(),
        format!("scored down ({} logs", logs - 10),
        "abstained (0 logs".to_string(),
        "6 more".to_string(),
        "budget only".to_string(),
        format!("triage: {logs} logs in batches of at most "),
    ] {
        assert!(text.contains(&expected), "`{expected}` missing in:\n{text}");
    }
    assert!(
        !text.contains("error rate"),
        "no calibration, no error rate:\n{text}"
    );

    // Requirement 6.9: triage writes nothing.
    assert_eq!(files_below(&project.project()).len(), 3);
}

#[test]
fn triage_with_a_calibration_displays_the_three_lists_with_label_sets() {
    let project = Project::new("triage-calibrated");
    project.fit();
    let calibrated = project.calibrate();
    let triage = project.arg("triage.csv");
    let base = [
        "triage",
        "--input",
        &triage,
        "--group-column",
        "group",
        "--baseline",
        "base",
        "--calibration",
        "cal",
        "--error-rate",
        "0.1",
    ];
    let value = project.json(&base);

    let calibration = &value["calibration"];
    assert_eq!(calibration["name"], "cal");
    assert_eq!(
        calibration["revision"],
        calibrated["calibration"]["revision"]
    );
    assert_eq!(calibration["error_rate"], 0.1);
    assert!(calibration["note"].as_str().unwrap().starts_with(NOTE));

    // Every log is in exactly one list, and the lists are what the label sets say
    // (requirements 11.6, 11.7).
    let counts = &value["counts"];
    let logs = count(&counts["logs"]);
    let review = rows(&value, "review");
    let scored_down = rows(&value, "scored_down");
    let abstained = rows(&value, "abstained");
    assert_eq!(
        (review.len() + scored_down.len() + abstained.len()) as u64,
        logs
    );
    assert_eq!(counts["review"], review.len());
    assert_eq!(counts["scored_down"], scored_down.len());
    assert_eq!(counts["abstained"], abstained.len());
    let mut seen = BTreeSet::new();
    for row in review.iter().chain(scored_down).chain(abstained) {
        assert_row_shape(row);
        assert!(seen.insert(count(&row["row"])), "row twice: {row}");
        assert!(
            row["p_good"].is_number() && row["p_bad"].is_number(),
            "{row}"
        );
        for stratum in ["good_stratum", "bad_stratum"] {
            assert!(row[stratum] == "group" || row[stratum] == "pooled", "{row}");
        }
    }
    assert!(!review.is_empty() && !scored_down.is_empty(), "{counts}");
    for row in review {
        assert_eq!(row["label_set"], "bad");
        assert!(row["rank"].is_number());
    }
    for row in scored_down {
        assert_eq!(row["label_set"], "good");
        assert_eq!(row["scored_down_by"], "calibration");
    }
    for row in abstained {
        let reason = row["abstain_reason"].as_str().unwrap();
        match row["label_set"].as_str().unwrap() {
            "both" => assert_eq!(reason, "uncertain"),
            "neither" => assert_eq!(reason, "unlike_calibration"),
            other => panic!("{other}: {row}"),
        }
    }
    let sets = &calibration["sets"];
    assert_eq!(sets["bad"], review.len());
    assert_eq!(sets["good"], scored_down.len());
    assert_eq!(
        count(&sets["both"]) + count(&sets["neither"]),
        abstained.len() as u64
    );
    assert_eq!(counts["uncertain"], sets["both"]);
    assert_eq!(counts["unlike_calibration"], sets["neither"]);
    // Logs judged on a group's own list against the pooled list, per label.
    for label in ["good", "bad"] {
        let lists = &calibration["lists"][label];
        assert_eq!(
            count(&lists["group"]) + count(&lists["pooled"]),
            logs,
            "{lists}"
        );
        let own = review
            .iter()
            .chain(scored_down)
            .chain(abstained)
            .filter(|row| row[format!("{label}_stratum")] == "group")
            .count();
        assert_eq!(lists["group"], own);
    }
    assert!(
        count(&calibration["lists"]["good"]["group"]) > 0,
        "{calibration}"
    );

    // With a budget, calibration and budget are told apart on every scored-down log.
    let mut args = base.to_vec();
    args.extend_from_slice(&["--budget", "5"]);
    let budgeted = project.json(&args);
    assert_eq!(rows(&budgeted, "review").len(), 5);
    let by_budget = rows(&budgeted, "scored_down")
        .iter()
        .filter(|row| row["scored_down_by"] == "budget_only")
        .count();
    assert_eq!(by_budget, review.len() - 5);
    assert_eq!(budgeted["counts"]["scored_down_by_budget"], by_budget);
    assert_eq!(
        budgeted["counts"]["scored_down_by_calibration"],
        scored_down.len()
    );

    // The same lists on repeat and at another batch size (11.10, 12.6).
    let again = project.json(&base);
    for part in [
        "review",
        "scored_down",
        "abstained",
        "counts",
        "calibration",
        "input",
    ] {
        assert_eq!(again[part], value[part], "{part}");
    }
    let mut batched = vec!["--batch-rows", "7"];
    batched.extend_from_slice(&base);
    let batched = project.json(&batched);
    for list in [
        "review",
        "scored_down",
        "abstained",
        "counts",
        "calibration",
    ] {
        assert_eq!(batched[list], value[list], "{list}");
    }
    assert_eq!(
        batched["memory"]["operations"][1]["batches"]["max_rows_used"],
        7
    );

    // The text form (the done-criterion): the three lists with label sets and reasons,
    // the scorer, the statement of requirement 11.9, the counts.
    let text = project.text(&base);
    for expected in [
        format!("triage of {logs} logs by the baseline scorer base"),
        "calibration: cal".to_string(),
        "error rate 0.1".to_string(),
        NOTE.to_string(),
        format!(
            "label sets: {} good, {} bad, {} both, {} neither",
            sets["good"], sets["bad"], sets["both"], sets["neither"]
        ),
        format!(
            "judged on a group's own list / on the pooled list: good {} / {}, bad {} / {}",
            calibration["lists"]["good"]["group"],
            calibration["lists"]["good"]["pooled"],
            calibration["lists"]["bad"]["group"],
            calibration["lists"]["bad"]["pooled"]
        ),
        format!("review ({} logs", review.len()),
        format!("scored down ({} logs", scored_down.len()),
        format!("abstained ({} logs", abstained.len()),
        "by calibration".to_string(),
        " bad ".to_string(),
        " good ".to_string(),
    ] {
        assert!(text.contains(&expected), "`{expected}` missing in:\n{text}");
    }
    if !abstained.is_empty() {
        assert!(
            text.contains("uncertain") || text.contains("unlike the calibration logs"),
            "{text}"
        );
    }
}

#[test]
fn triage_refusals_and_option_errors() {
    let project = Project::new("triage-failures");
    project.fit();
    project.calibrate();
    let triage = project.arg("triage.csv");
    let with = |more: &[&str]| {
        let mut args = vec!["triage", "--input", &triage, "--baseline", "base"];
        args.extend_from_slice(more);
        args.into_iter().map(str::to_string).collect::<Vec<_>>()
    };
    let fails = |args: Vec<String>, code: i32| {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        project.fails(&args, code)
    };

    let line = fails(with(&["--error-rate", "0.1"]), 2);
    assert!(
        line.contains("--error-rate") && line.contains("--calibration"),
        "{line}"
    );
    let line = fails(with(&["--calibration", "cal", "--error-rate", "2"]), 2);
    assert!(line.contains("error rate"), "{line}");
    let line = fails(with(&["--budget", "many"]), 2);
    assert!(line.contains("--budget"), "{line}");
    let line = fails(with(&["--primary", "linear"]), 2);
    assert!(line.contains("--primary"), "{line}");

    // Requirement 11.5: an error rate the calibration logs do not support.
    let line = fails(with(&["--calibration", "cal", "--error-rate", "0.001"]), 4);
    assert!(line.contains("too few"), "{line}");
    assert!(line.contains("support error rates from"), "{line}");

    // Requirement 10.5.
    let line = fails(with(&["--calibration", "absent"]), 3);
    assert!(line.contains("calibration 'absent'"), "{line}");
    assert!(
        line.contains("calibrations") && line.contains("expected at"),
        "{line}"
    );
    let line = project.fails(&["triage", "--input", &triage, "--baseline", "absent"], 3);
    assert!(line.contains("baseline scorer 'absent'"), "{line}");
    let line = project.fails(
        &["triage", "--input", "absent.csv", "--baseline", "base"],
        3,
    );
    assert!(line.contains("absent.csv"), "{line}");

    // What needs an encoder is refused with the reason, not mistaken for a typing error.
    let line = project.fails(&["triage", "--store", "s", "--baseline", "base"], 4);
    assert!(
        line.contains("--store") && line.contains("encoder"),
        "{line}"
    );
    let line = fails(with(&["--primary", "neighbour"]), 4);
    assert!(
        line.contains("neighbour") && line.contains("encoder"),
        "{line}"
    );
    let line = project.fails(&["triage", "--input", &triage], 2);
    assert!(line.contains("--baseline"), "{line}");

    // A calibration made for another scorer is refused (a second fit under another name).
    let calibration = project.arg("calibration.csv");
    project.json(&[
        "baseline",
        "fit",
        "--input",
        &calibration,
        "--group-column",
        "group",
        "--name",
        "other",
        "--buckets",
        "1024",
    ]);
    let line = project.fails(
        &[
            "triage",
            "--input",
            &triage,
            "--baseline",
            "other",
            "--calibration",
            "cal",
        ],
        4,
    );
    assert!(
        line.contains("not for the baseline scorer of revision"),
        "{line}"
    );

    // A memory limit that does not leave what triage holds: refused with the figures.
    // 5,000 logs take more than the 6 MiB that 38 MiB leave beside the tool itself.
    let many: Vec<String> = (0..5_000)
        .map(|index| format!("GET /tienda1/publico/pagina.jsp?id={index:06}"))
        .collect();
    let many = project.write("many.log", &many);
    let many = many.to_string_lossy().into_owned();
    let line = project.fails(
        &[
            "triage",
            "--input",
            &many,
            "--baseline",
            "base",
            "--memory-limit",
            "38MB",
        ],
        4,
    );
    assert!(line.contains("memory limit"), "{line}");
    assert!(line.contains("bytes"), "{line}");
    // With room the same logs are triaged; `--limit` bounds the rows printed.
    let value = project.json(&[
        "triage",
        "--input",
        &many,
        "--baseline",
        "base",
        "--memory-limit",
        "64MB",
        "--limit",
        "2",
    ]);
    assert_eq!(value["counts"]["review"], 5_000);
    assert_eq!(rows(&value, "review").len(), 2);
    assert_eq!(value["rows_omitted"]["review"], 4_998);
}

#[test]
fn triage_of_an_empty_input_has_the_same_shape_and_logs_without_features_are_marked() {
    let project = Project::new("triage-edges");
    let fit = project.arg("fit.csv");
    // Strip mode: a text without payload has no features and gets the constant ratio.
    project.json(&[
        "baseline",
        "fit",
        "--input",
        &fit,
        "--group-column",
        "group",
        "--name",
        "strip",
        "--routing",
        "strip",
        "--buckets",
        "1024",
    ]);

    let empty = project.write("empty.csv", &["text,label,group".to_string()]);
    let empty = empty.to_string_lossy().into_owned();
    let value = project.json(&["triage", "--input", &empty, "--baseline", "strip"]);
    assert_eq!(value["counts"]["logs"], 0);
    for list in ["review", "scored_down", "abstained"] {
        assert!(rows(&value, list).is_empty());
    }
    // The documented shape for no logs: the lists are empty and no evidence column is
    // named, because the scorer was not asked for any.
    assert_eq!(value["evidence_columns"], serde_json::json!([]));
    assert_eq!(value["scorers"][0]["name"], "strip");
    let text = project.text(&["triage", "--input", &empty, "--baseline", "strip"]);
    assert!(text.contains("triage of 0 logs"), "{text}");
    assert!(text.contains("review (0 logs"), "{text}");

    let plain = project.write(
        "plain.log",
        &[
            "GET http://localhost:8080/tienda1/index.jsp".to_string(),
            "GET http://localhost:8080/tienda1/publico/buscar.jsp?q=%27+OR+1%3D1".to_string(),
            "no payload at all".to_string(),
            "esc \u{1b}[31m red".to_string(),
        ],
    );
    let plain = plain.to_string_lossy().into_owned();
    let value = project.json(&["triage", "--input", &plain, "--baseline", "strip"]);
    let review = rows(&value, "review");
    assert_eq!(review.len(), 4);
    let flagged: Vec<&str> = review
        .iter()
        .filter(|row| row["evidence"]["no_features"] == true)
        .map(|row| row["text"].as_str().unwrap())
        .collect();
    assert_eq!(flagged.len(), 3, "{flagged:?}");
    assert_eq!(value["counts"]["no_features"], 3);
    let text = project.text(&["triage", "--input", &plain, "--baseline", "strip"]);
    assert_eq!(
        text.matches("no features").count(),
        3 + 1,
        "three rows and the count:\n{text}"
    );
    // Log text is data: a control character is shown escaped, never sent to the terminal.
    assert!(!text.contains('\u{1b}'), "{text:?}");
    assert!(text.contains("\\u{1b}[31m"), "{text}");
}

// --- Remediation of the first review -------------------------------------------------------

/// Runs `l2d` with `args`, reads `keep` bytes of its standard output, closes the pipe
/// and returns the exit code and what was written to standard error.
fn closing_the_pipe_early(project: &Project, args: &[&str], keep: usize) -> (Option<i32>, String) {
    use std::io::Read;
    use std::process::Stdio;

    let mut child = Command::new(env!("CARGO_BIN_EXE_l2d"))
        .arg("--project")
        .arg(project.project())
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the l2d binary runs");
    let mut out = child.stdout.take().unwrap();
    let mut start = vec![0_u8; keep];
    out.read_exact(&mut start)
        .expect("the output is longer than what is kept");
    drop(out);
    let mut err = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut err)
        .unwrap();
    (child.wait().unwrap().code(), err)
}

#[test]
fn a_reader_that_closes_the_output_early_gets_exit_5_and_one_line() {
    let project = Project::new("closed-pipe");
    project.fit();
    // 20,000 logs: several megabytes of output in either form, far more than a pipe holds.
    let many: Vec<String> = (0..20_000)
        .map(|index| format!("GET /tienda1/publico/pagina.jsp?id={index:06}&q=vino+tinto"))
        .collect();
    let many = project.write("many.log", &many);
    let many = many.to_string_lossy().into_owned();
    for args in [
        &[
            "--json",
            "baseline",
            "score",
            "--input",
            &many,
            "--baseline",
            "base",
        ][..],
        &["--json", "triage", "--input", &many, "--baseline", "base"][..],
        &[
            "baseline",
            "score",
            "--input",
            &many,
            "--baseline",
            "base",
            "--limit",
            "20000",
        ][..],
        &[
            "triage",
            "--input",
            &many,
            "--baseline",
            "base",
            "--limit",
            "20000",
        ][..],
    ] {
        let (code, err) = closing_the_pipe_early(&project, args, 200);
        assert_eq!(code, Some(5), "l2d {}: {err}", args.join(" "));
        let lines: Vec<&str> = err.lines().collect();
        assert_eq!(lines.len(), 1, "l2d {}: {err:?}", args.join(" "));
        assert!(
            lines[0].starts_with("error: runtime failure: writing the output failed"),
            "{err}"
        );
        assert!(!err.contains("panicked"), "{err}");
    }
}

#[test]
fn a_typed_share_that_cannot_be_honoured_is_an_analyst_error() {
    let project = Project::new("typed-shares");
    let fixture = fixture_path().to_string_lossy().into_owned();
    // A limit share so small that the limit is below what the tool itself needs.
    let line = project.fails(
        &["ingest", "--input", &fixture, "--memory-share", "0.000001"],
        2,
    );
    assert!(line.contains("--memory-share"), "{line}");

    // Shares that add up to more than 1: the limit could never be used.
    // Both typed: an analyst error that says what to type.
    for options in [
        &["--memory-share", "0.9", "--reserve-share", "0.9"][..],
        &["--memory-share", "0.8", "--reserve-share", "0.25"][..],
    ] {
        let mut args = vec!["ingest", "--input", &fixture];
        args.extend_from_slice(options);
        let line = project.fails(&args, 2);
        assert!(
            line.contains("--memory-share") && line.contains("--reserve-share"),
            "{line}"
        );
        assert!(line.contains("more than 1"), "{line}");
        assert!(line.contains("before"), "it says why: {line}");
        assert!(line.contains("at most"), "it says what to type: {line}");
    }
    // One typed: the other is lowered to what is left of 1, and the output says so.
    let ingest = |options: &[&str]| {
        let mut args = vec!["ingest", "--input", fixture.as_str()];
        args.extend_from_slice(options);
        (project.json(&args), project.text(&args))
    };
    let (value, text) = ingest(&["--memory-share", "0.8"]);
    let memory = &value["memory"];
    assert_eq!(memory["limit_share"], 0.8);
    assert_eq!(memory["reserve_share"], 0.2);
    assert_eq!(memory["reserve_share_adjusted"], true);
    assert_eq!(memory["limit_share_adjusted"], false);
    let free = count(&memory["free_bytes"]) as f64;
    assert_eq!(
        count(&memory["reserve_bytes"]),
        (free * f64::from(0.2_f32)) as u64
    );
    assert!(
        text.contains("reserve ") && text.contains("(share 0.2, lowered from the default 0.25"),
        "{text}"
    );
    let (value, text) = ingest(&["--reserve-share", "0.6"]);
    let memory = &value["memory"];
    assert_eq!(memory["reserve_share"], 0.6);
    assert_eq!(memory["limit_share"], 0.4);
    assert_eq!(memory["limit_share_adjusted"], true);
    assert_eq!(memory["reserve_share_adjusted"], false);
    assert_eq!(
        count(&memory["limit_bytes"]),
        (free_of(memory) * f64::from(0.4_f32)) as u64
    );
    assert!(
        text.contains("(share 0.4 of the free memory, lowered from the default 0.5"),
        "{text}"
    );
    // Nothing is adjusted when the shares fit.
    let (value, text) = ingest(&["--memory-share", "0.6"]);
    assert_eq!(value["memory"]["reserve_share"], 0.25);
    assert_eq!(value["memory"]["reserve_share_adjusted"], false);
    assert!(!text.contains("lowered"), "{text}");
    // Exactly 1 is the edge that is still accepted, and a limit in bytes is not a share.
    for options in [
        &["--memory-share", "0.75"][..],
        &["--memory-share", "0.5", "--reserve-share", "0.5"][..],
        &["--memory-limit", "64MB", "--reserve-share", "0.9"][..],
    ] {
        let mut args = vec!["ingest", "--input", &fixture];
        args.extend_from_slice(options);
        assert_eq!(project.json(&args)["input"]["records"], 200);
    }
}

#[test]
fn a_directory_given_as_input_is_an_analyst_error_not_a_missing_file() {
    let project = Project::new("directory-input");
    let directory = project.root.join("in").to_string_lossy().into_owned();
    for adapter in ["auto", "lines", "csv"] {
        let line = project.fails(&["ingest", "--input", &directory, "--adapter", adapter], 2);
        assert!(line.contains("--input"), "{line}");
        assert!(line.contains("is a directory"), "{line}");
    }
    // Below the data directory too.
    std::fs::create_dir_all(project.project().join("data").join("folder")).unwrap();
    let line = project.fails(&["ingest", "--input", "folder"], 2);
    assert!(line.contains("is a directory"), "{line}");
}

#[test]
fn format_characters_in_log_text_do_not_reach_the_terminal() {
    let project = Project::new("format-characters");
    project.fit();
    let texts = [
        "GET /a?file=\u{202e}gpj.exe".to_string(),
        "GET /b?x=\u{2066}y\u{2069}\u{200b}\u{200f}\u{feff}z".to_string(),
        "GET /c?plain=1".to_string(),
    ];
    let logs = project.write("format.log", &texts);
    let logs = logs.to_string_lossy().into_owned();
    for args in [
        &["baseline", "score", "--input", &logs, "--baseline", "base"][..],
        &["triage", "--input", &logs, "--baseline", "base"][..],
    ] {
        let text = project.text(args);
        for character in [
            '\u{202e}', '\u{2066}', '\u{2069}', '\u{200b}', '\u{200f}', '\u{feff}',
        ] {
            assert!(
                !text.contains(character),
                "U+{:04X} reached the output of l2d {}",
                character as u32,
                args.join(" ")
            );
        }
        assert!(text.contains("file=\\u{202e}gpj.exe"), "{text}");
        assert!(
            text.contains("\\u{2066}y\\u{2069}\\u{200b}\\u{200f}\\u{feff}z"),
            "{text}"
        );
        assert!(text.contains("GET /c?plain=1"), "{text}");
    }
    // JSON carries the text as it is.
    let value = project.json(&["baseline", "score", "--input", &logs, "--baseline", "base"]);
    assert_eq!(value["scores"][0]["text"], texts[0].as_str());
}

#[test]
fn a_scorer_that_does_not_fit_the_memory_limit_is_refused_before_it_is_loaded() {
    let project = Project::new("load-limit");
    let fit = project.arg("fit.csv");
    // 524,288 columns: a weights file of 4 MiB.
    project.json(&[
        "baseline",
        "fit",
        "--input",
        &fit,
        "--group-column",
        "group",
        "--name",
        "wide",
        "--buckets",
        "262144",
        "--routing",
        "isolate",
        "--max-iter",
        "2",
    ]);
    let triage = project.arg("triage.csv");
    // 39 MiB leave 7 MiB beside the tool itself, of which the read holds more than 4:
    // the scorer's 4 MiB of weights do not fit.
    for command in [&["baseline", "score"][..], &["triage"][..]] {
        let mut args = command.to_vec();
        args.extend_from_slice(&[
            "--input",
            &triage,
            "--baseline",
            "wide",
            "--memory-limit",
            "39MB",
        ]);
        let line = project.fails(&args, 4);
        assert!(line.contains("loading the baseline scorer"), "{line}");
        assert!(line.contains("nothing was loaded"), "{line}");
        // The figure asked for is about the files: above the weights, below twice them.
        let asked: u64 = line
            .split("estimated ")
            .nth(1)
            .and_then(|rest| rest.split('(').nth(1))
            .and_then(|rest| rest.split(' ').next())
            .and_then(|bytes| bytes.parse().ok())
            .unwrap_or_else(|| panic!("no estimate in: {line}"));
        assert!(
            asked > 4 * 1024 * 1024 && asked < 5 * 1024 * 1024,
            "{asked}: {line}"
        );
    }
    // With 4 MiB more the scorer fits and the logs are scored.
    let value = project.json(&[
        "baseline",
        "score",
        "--input",
        &triage,
        "--baseline",
        "wide",
        "--memory-limit",
        "44MB",
    ]);
    assert_eq!(value["scored"], value["input"]["distinct_texts"]);
}

fn free_of(memory: &Value) -> f64 {
    count(&memory["free_bytes"]) as f64
}

#[test]
fn a_fit_reports_the_scorer_as_it_is_saved_after_a_merge_into_an_earlier_directory() {
    let project = Project::new("fit-merge");
    // The scorer saved by the code before task 1.11: 512 buckets, fitted on the fixture.
    let saved = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("l2d-baseline")
        .join("tests")
        .join("fixtures")
        .join("scorer-1.7")
        .join("models")
        .join("baselines")
        .join("before-1.11-53c19878ccb7");
    let target = project
        .project()
        .join("models")
        .join("baselines")
        .join("before-1.11-53c19878ccb7");
    std::fs::create_dir_all(&target).unwrap();
    for file in ["baseline.json", "weights.bin"] {
        std::fs::copy(saved.join(file), target.join(file)).unwrap();
    }

    // The same distinct texts in the same order in another file (one row repeated at
    // the end): the same model, so the fit is saved into the earlier directory.
    let fixture = std::fs::read_to_string(fixture_path()).unwrap();
    let mut lines: Vec<String> = fixture.lines().map(str::to_string).collect();
    lines.push(lines[1].clone());
    let other = project.write("other.csv", &lines);
    let other = other.to_string_lossy().into_owned();
    let args = [
        "baseline",
        "fit",
        "--input",
        &other,
        "--group-column",
        "group",
        "--name",
        "before-1.11",
        "--buckets",
        "512",
    ];
    let value = project.json(&args);
    let scorer = &value["scorer"];
    assert_eq!(
        scorer["revision"],
        "53c19878ccb777b54e64170818dec20fee792a77"
    );
    assert!(
        scorer["directory"]
            .as_str()
            .unwrap()
            .ends_with("before-1.11-53c19878ccb7")
    );
    assert_eq!(scorer["fitted_text_hashes"], 168);
    // The directory also names the earlier input, whose texts came without hashes.
    assert_eq!(scorer["fitted_text_hashes_complete"], false);
    let text = project.text(&args);
    assert!(
        text.contains("text hashes saved: 168 (incomplete"),
        "{text}"
    );

    // Calibration then says that its check per log is partial.
    let mut calibration = vec!["text,label,group".to_string()];
    for index in 0..30 {
        calibration.push(format!(
            "GET /nuevo/pagina.jsp?id={index}&q=vino,good,GET /nuevo/pagina.jsp"
        ));
        calibration.push(format!(
            "GET /nuevo/pagina.jsp?id={index}&q=%27+OR+1%3D1+--,bad,GET /nuevo/pagina.jsp"
        ));
    }
    calibration.push(lines[1].clone());
    let calibration = project.write("new.csv", &calibration);
    let calibration = calibration.to_string_lossy().into_owned();
    let value = project.json(&[
        "calibrate",
        "--input",
        &calibration,
        "--group-column",
        "group",
        "--baseline",
        "before-1.11",
        "--name",
        "cal",
    ]);
    assert_eq!(value["held_out"]["check"], "per_log_partial");
    assert_eq!(value["held_out"]["dropped"], 1);
    assert!(value["held_out"]["note"].as_str().unwrap().contains("part"));
    assert_eq!(value["scorer"]["fitted_text_hashes_complete"], false);
}
