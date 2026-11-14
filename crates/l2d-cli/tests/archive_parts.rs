//! `l2d baseline score` on a zip archive (requirements 13.11 and 13.12; task 1.17), run
//! as the built binary: the ratios go to Parquet parts in a work directory, one row per
//! distinct log, and the deduplication goes on on disk when it outgrows the memory.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_core::parts::read_part;
use l2d_core::polars::prelude::Column;
use serde_json::Value;

/// A temporary directory with a project in it that holds a fitted baseline `web`,
/// removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-cli-parts-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("project")).unwrap();
        let scratch = Self { root };
        let mut labelled = String::from("text,label\n");
        for index in 0..40 {
            labelled.push_str(&format!("GET /page/{index} 200,good\n"));
            labelled.push_str(&format!("GET /page/{index}?id=1%27%20OR%201=1-- 500,bad\n"));
        }
        let labelled = scratch.write("labelled.csv", labelled.as_bytes());
        scratch.text(&["baseline", "fit", "--input", &labelled, "--name", "web"]);
        scratch
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn arg(&self, name: &str) -> String {
        self.path(name).to_string_lossy().into_owned()
    }

    fn write(&self, name: &str, bytes: &[u8]) -> String {
        std::fs::write(self.path(name), bytes).unwrap();
        self.arg(name)
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_l2d"))
            .arg("--project")
            .arg(self.root.join("project"))
            .args(args)
            .output()
            .expect("the l2d binary runs")
    }

    /// Runs a command that must succeed with `--json` and returns its document.
    fn json(&self, args: &[&str]) -> Value {
        let mut all = vec!["--json"];
        all.extend_from_slice(args);
        let output = self.run(&all);
        assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
        assert_eq!(stderr(&output), "");
        serde_json::from_str(&stdout(&output)).unwrap()
    }

    /// Runs a command that must succeed and returns its text output.
    fn text(&self, args: &[&str]) -> String {
        let output = self.run(args);
        assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
        assert_eq!(stderr(&output), "");
        stdout(&output)
    }

    /// Runs a command that must fail with `code` before it prints a result, and
    /// returns its one line of diagnostics.
    fn fails(&self, args: &[&str], code: i32) -> String {
        let output = self.run(args);
        assert_eq!(output.status.code(), Some(code), "{}", stderr(&output));
        assert_eq!(stdout(&output), "");
        let diagnostics = stderr(&output);
        assert_eq!(diagnostics.lines().count(), 1, "{diagnostics}");
        diagnostics.trim_end().to_string()
    }

    /// Scores the archive `archive` into the work directory `run` with further options.
    fn score(&self, archive: &str, run: &str, more: &[&str]) -> Value {
        let work = self.arg(run);
        let mut args = more.to_vec();
        args.extend_from_slice(&[
            "baseline",
            "score",
            "--input",
            archive,
            "--baseline",
            "web",
            "--work-dir",
            &work,
        ]);
        self.json(&args)
    }
}

impl Drop for Scratch {
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

fn gzip(content: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(content).unwrap();
    encoder.finish().unwrap()
}

/// An archive with the entries `(name, content)`; a name ending in `/` is a directory.
fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    for (name, content) in entries {
        if name.ends_with('/') {
            writer.add_directory(*name, options).unwrap();
        } else {
            writer.start_file(*name, options).unwrap();
            writer.write_all(content).unwrap();
        }
    }
    writer.finish().unwrap().into_inner()
}

/// The file names in `dir`, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap()
}

fn table<'a>(manifest: &'a Value, name: &str) -> &'a Value {
    manifest["tables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|table| table["name"] == name)
        .unwrap_or_else(|| panic!("no table {name} in {manifest}"))
}

/// One row of the scores table of a run.
#[derive(Debug, Clone, PartialEq)]
struct Scored {
    row: u64,
    entry: u32,
    line: u64,
    text: String,
    ratio: f32,
    no_features: bool,
    occurrences: u32,
}

/// Reads the parts the manifest of `dir` names, checks them against it, and returns
/// the rows of the scores table with their occurrence counts.
fn scored(dir: &Path) -> Vec<Scored> {
    let manifest = manifest(dir);
    let mut rows = Vec::new();
    let mut listed = vec!["manifest.json".to_string()];
    let scores = table(&manifest, "scores");
    for part in scores["parts"].as_array().unwrap() {
        let file = part["file"].as_str().unwrap();
        listed.push(file.to_string());
        let path = dir.join(file);
        assert_eq!(part["bytes"], std::fs::metadata(&path).unwrap().len());
        let frame = read_part(&path).unwrap();
        assert_eq!(part["rows"], frame.height() as u64);
        let names: Vec<&str> = frame
            .get_column_names()
            .into_iter()
            .map(|name| name.as_str())
            .collect();
        assert_eq!(
            names,
            ["row", "entry", "line", "text", "ratio", "no_features"]
        );
        let row = frame.column("row").and_then(Column::u64).unwrap();
        let entry = frame.column("entry").and_then(Column::u32).unwrap();
        let line = frame.column("line").and_then(Column::u64).unwrap();
        let text = frame.column("text").and_then(Column::str).unwrap();
        let ratio = frame.column("ratio").and_then(Column::f32).unwrap();
        let flag = frame.column("no_features").and_then(Column::bool).unwrap();
        for index in 0..frame.height() {
            rows.push(Scored {
                row: row.get(index).unwrap(),
                entry: entry.get(index).unwrap(),
                line: line.get(index).unwrap(),
                text: text.get(index).unwrap().to_string(),
                ratio: ratio.get(index).unwrap(),
                no_features: flag.get(index).unwrap(),
                occurrences: 0,
            });
        }
    }
    assert_eq!(scores["rows"], rows.len() as u64);

    let occurrences = table(&manifest, "occurrences");
    let mut at = 0;
    for part in occurrences["parts"].as_array().unwrap() {
        let file = part["file"].as_str().unwrap();
        listed.push(file.to_string());
        let frame = read_part(&dir.join(file)).unwrap();
        let row = frame.column("row").and_then(Column::u64).unwrap();
        let count = frame.column("occurrences").and_then(Column::u32).unwrap();
        for index in 0..frame.height() {
            assert_eq!(row.get(index).unwrap(), at as u64);
            rows[at].occurrences = count.get(index).unwrap();
            at += 1;
        }
    }
    assert_eq!(at, rows.len(), "one count per row");
    assert_eq!(occurrences["rows"], rows.len() as u64);
    assert_eq!(manifest["rows"], rows.len() as u64);

    // The directory holds the manifest and the parts it names, nothing else.
    listed.sort();
    assert_eq!(names(dir), listed);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.row, index as u64, "rows are numbered in order");
    }
    rows
}

/// Two logs (one of them gzip), a packet capture and a directory.
fn sample() -> Vec<u8> {
    let rotated = gzip(b"GET /b 200\nGET /a 200\nGET /page/3?id=1%27%20OR%201=1-- 500\n");
    let mut capture = b"\xd4\xc3\xb2\xa1".to_vec();
    capture.extend_from_slice(&[0; 64]);
    archive(&[
        ("logs/", b""),
        ("logs/access.log", b"GET /a 200\n\nGET /a 200\nGET /c 404\n"),
        ("logs/access.log.1.gz", &rotated),
        ("capture/eth0.pcap", &capture),
    ])
}

#[test]
fn scoring_an_archive_writes_one_row_per_distinct_log_into_parts() {
    let scratch = Scratch::new("score");
    let path = scratch.write("sample.zip", &sample());
    let document = scratch.score(&path, "run", &[]);
    let run = scratch.path("run");

    assert_eq!(document["command"], "baseline score");
    assert_eq!(document["input"]["adapter"], "zip");
    assert_eq!(document["input"]["records"], 7);
    assert_eq!(document["input"]["distinct_texts"], 4);
    assert_eq!(document["input"]["entries_read"], 2);
    assert_eq!(document["scored"], 4);
    let parts = &document["parts"];
    assert_eq!(parts["directory"], run.to_string_lossy().as_ref());
    assert_eq!(
        parts["manifest"],
        run.join("manifest.json").to_string_lossy().as_ref()
    );
    assert_eq!(parts["rows"], 4);
    assert_eq!(parts["deduplication"]["on_disk"], false);
    assert!(document["memory"].is_object());

    // One row per distinct log, in order of first occurrence, with where it was first
    // seen and how often it occurred in the archive.
    let rows = scored(&run);
    let seen: Vec<(&str, u32, u64, u32)> = rows
        .iter()
        .map(|row| (row.text.as_str(), row.entry, row.line, row.occurrences))
        .collect();
    assert_eq!(
        seen,
        [
            ("GET /a 200", 1, 1, 3),
            ("GET /c 404", 1, 4, 1),
            ("GET /b 200", 2, 1, 1),
            ("GET /page/3?id=1%27%20OR%201=1-- 500", 2, 3, 1),
        ]
    );
    assert!(rows[3].ratio > rows[0].ratio, "{rows:?}");

    // The ratios are those the same baseline gives the same texts read from a file.
    let lines: String = rows.iter().map(|row| format!("{}\n", row.text)).collect();
    let plain = scratch.write("distinct.log", lines.as_bytes());
    let reference = scratch.json(&["baseline", "score", "--input", &plain, "--baseline", "web"]);
    let reference = reference["scores"].as_array().unwrap();
    assert_eq!(reference.len(), rows.len());
    for (row, expected) in rows.iter().zip(reference) {
        assert_eq!(expected["text"], row.text.as_str());
        assert_eq!(expected["ratio"].as_f64().unwrap() as f32, row.ratio);
        assert_eq!(expected["no_features"], row.no_features);
    }

    // The rows printed are the first rows of the parts.
    let printed = document["scores"].as_array().unwrap();
    assert_eq!(printed.len(), 4);
    for (row, shown) in rows.iter().zip(printed) {
        assert_eq!(shown["row"], row.row);
        assert_eq!(shown["text"], row.text.as_str());
        assert_eq!(
            shown["entry"],
            ["", "logs/access.log", "logs/access.log.1.gz"][row.entry as usize]
        );
        assert_eq!(shown["line"], row.line);
        assert_eq!(shown["ratio"].as_f64().unwrap() as f32, row.ratio);
    }

    // The manifest records the archive, its hash and every entry (requirement 13.13).
    let manifest = manifest(&run);
    assert_eq!(manifest["command"], "baseline score");
    assert_eq!(manifest["scorer"]["name"], "web");
    assert_eq!(manifest["input"]["source_path"], path.as_str());
    assert_eq!(
        manifest["input"]["content_sha256"],
        document["input"]["content_sha256"]
    );
    assert_eq!(manifest["input"]["records"], 7);
    assert_eq!(manifest["input"]["skipped_count"], 1);
    let entries = manifest["input"]["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 4);
    assert_eq!(entries[1]["path"], "logs/access.log");
    assert_eq!(entries[1]["records"], 4);
    assert_eq!(entries[3]["skipped"], "not_text");
    assert_eq!(
        table(&manifest, "scores")["rows_sha256"],
        parts["tables"][0]["rows_sha256"]
    );

    // The text output says where the parts are.
    let text = scratch.text(&[
        "baseline",
        "score",
        "--input",
        &path,
        "--baseline",
        "web",
        "--work-dir",
        &scratch.arg("run-text"),
        "--limit",
        "2",
    ]);
    assert!(text.contains("  records read: 7 (1 skipped)\n"), "{text}");
    assert!(text.contains("  ... 2 more"), "{text}");
    assert!(
        text.contains(&format!(
            "result parts: {}\n",
            scratch.path("run-text").display()
        )),
        "{text}"
    );
    assert!(text.contains("  scores: 4 rows in 1 part"), "{text}");
    assert_eq!(scored(&scratch.path("run-text")), rows);
}

#[test]
fn without_a_work_directory_the_parts_go_below_the_projects_work_directory() {
    let scratch = Scratch::new("default");
    let path = scratch.write("sample.zip", &sample());
    let document = scratch.json(&["baseline", "score", "--input", &path, "--baseline", "web"]);
    let work = scratch.path("project").join("work");
    let runs = names(&work);
    assert_eq!(runs.len(), 1, "{runs:?}");
    let run = &runs[0];
    // baseline-score-<date>T<time>Z
    let stamp = run.strip_prefix("baseline-score-").expect(run);
    assert_eq!(stamp.len(), 16, "{run}");
    assert!(
        stamp.bytes().enumerate().all(|(at, byte)| match at {
            8 => byte == b'T',
            15 => byte == b'Z',
            _ => byte.is_ascii_digit(),
        }),
        "{run}"
    );
    let directory = document["parts"]["directory"].as_str().unwrap();
    assert!(
        Path::new(directory).ends_with(Path::new("work").join(run)),
        "{directory}"
    );
    assert_eq!(scored(&work.join(run)).len(), 4);
}

/// 600 lines over three entries; 140 distinct texts that repeat within and across
/// the entries.
fn repeating() -> Vec<u8> {
    let entry = |step: u32| {
        (0..200u32)
            .map(|line| format!("GET /page/{}?step={} 200\n", (line * step) % 47, line % 3))
            .collect::<String>()
    };
    let (first, second, third) = (entry(1), entry(5), entry(11));
    archive(&[
        ("a/first.log", first.as_bytes()),
        ("a/second.log", second.as_bytes()),
        ("third.log", &gzip(third.as_bytes())),
    ])
}

fn rows_hashes(document: &Value) -> Vec<String> {
    document["parts"]["tables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|table| table["rows_sha256"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn the_rows_do_not_depend_on_the_batch_size_and_a_repeated_run_gives_the_same_bytes() {
    let scratch = Scratch::new("batches");
    let path = scratch.write("repeating.zip", &repeating());
    let whole = scratch.score(&path, "whole", &[]);
    let rows = scored(&scratch.path("whole"));
    assert_eq!(whole["input"]["records"], 600);
    let texts: BTreeSet<&str> = rows.iter().map(|row| row.text.as_str()).collect();
    assert_eq!(texts.len(), rows.len(), "one row per distinct log");
    assert_eq!(whole["input"]["distinct_texts"], rows.len() as u64);
    assert_eq!(
        rows.iter()
            .map(|row| u64::from(row.occurrences))
            .sum::<u64>(),
        600
    );
    assert_eq!(whole["parts"]["tables"][0]["parts"], 1);

    for batch_rows in ["1", "7", "64"] {
        let run = format!("rows-{batch_rows}");
        let document = scratch.score(&path, &run, &["--batch-rows", batch_rows]);
        assert_eq!(rows_hashes(&document), rows_hashes(&whole), "{batch_rows}");
        assert_eq!(scored(&scratch.path(&run)), rows, "{batch_rows}");
        if batch_rows == "7" {
            assert!(
                document["parts"]["tables"][0]["parts"].as_u64().unwrap() > 1,
                "{document}"
            );
        }
    }

    // The same run again: the same manifest and the same part files, byte for byte.
    scratch.score(&path, "again", &["--batch-rows", "7"]);
    let (first, second) = (scratch.path("rows-7"), scratch.path("again"));
    assert_eq!(names(&first), names(&second));
    for name in names(&first) {
        assert_eq!(
            std::fs::read(first.join(&name)).unwrap(),
            std::fs::read(second.join(&name)).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn an_archive_whose_texts_outgrow_the_memory_limit_is_scored_with_deduplication_on_disk() {
    let scratch = Scratch::new("outgrown");
    // 300,000 distinct lines, each twice: the second time in another entry, in reverse.
    let lines = 300_000u32;
    let mut forward = String::new();
    let mut backward = String::new();
    for line in 0..lines {
        forward.push_str(&format!("line {line}\n"));
        backward.push_str(&format!("line {}\n", lines - 1 - line));
    }
    let path = scratch.write(
        "many.zip",
        &archive(&[
            ("forward.log", forward.as_bytes()),
            ("backward.log", backward.as_bytes()),
        ]),
    );
    const LIMIT: &str = "50MB";

    // `ingest` reads it under this limit on disk as well (task 1.20) and leaves
    // nothing behind.
    let ingested = scratch.json(&["--memory-limit", LIMIT, "ingest", "--input", &path]);
    assert_eq!(ingested["work"]["deduplication"]["on_disk"], true);
    assert!(!scratch.path("project").join("work").exists());

    // Scoring it completes under the same limit, on disk, with the rows of a run that
    // has room for the whole set.
    let short = scratch.score(&path, "short", &["--memory-limit", LIMIT]);
    let roomy = scratch.score(&path, "roomy", &["--memory-limit", "2GB"]);
    let dedup = &short["parts"]["deduplication"];
    assert_eq!(dedup["on_disk"], true, "{dedup}");
    let in_memory = dedup["set_texts"].as_u64().unwrap();
    assert!(in_memory > 0 && in_memory < u64::from(lines), "{dedup}");
    assert_eq!(
        dedup["spilled_distinct"].as_u64().unwrap() + in_memory,
        u64::from(lines)
    );
    assert_eq!(roomy["parts"]["deduplication"]["on_disk"], false);
    assert_eq!(short["scored"], lines);
    assert_eq!(short["input"]["distinct_texts"], lines);
    assert_eq!(short["input"]["records"], 2 * lines);
    assert_eq!(rows_hashes(&short), rows_hashes(&roomy));
    // The counts of `ingest` and of `baseline score`, on disk and in memory.
    assert_eq!(ingested["input"], short["input"]);
    assert_eq!(ingested["input"], roomy["input"]);

    let rows = scored(&scratch.path("short"));
    assert_eq!(rows.len(), lines as usize);
    let mut by_ratio: BTreeMap<u32, u32> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.text, format!("line {index}"));
        assert_eq!(
            (row.entry, row.line, row.occurrences),
            (0, index as u64 + 1, 2)
        );
        *by_ratio.entry(row.ratio.to_bits()).or_default() += 1;
    }
    assert!(by_ratio.len() > 1, "the texts are scored one by one");
}

#[test]
fn a_work_directory_in_use_and_options_for_archives_on_other_inputs_are_refused() {
    let scratch = Scratch::new("refused");
    let path = scratch.write("sample.zip", &sample());
    std::fs::create_dir_all(scratch.path("used")).unwrap();
    let kept = scratch.write("used/notes.txt", b"mine");
    let line = scratch.fails(
        &[
            "baseline",
            "score",
            "--input",
            &path,
            "--baseline",
            "web",
            "--work-dir",
            &scratch.arg("used"),
        ],
        4,
    );
    assert!(line.contains("not an empty directory"), "{line}");
    assert_eq!(std::fs::read(&kept).unwrap(), b"mine");
    assert_eq!(names(&scratch.path("used")), ["notes.txt"]);

    let plain = scratch.write("plain.log", b"GET /a 200\n");
    for option in [
        &["--work-dir", "somewhere"][..],
        &["--entry", "*.log"][..],
        &["--max-record-bytes", "1KB"][..],
    ] {
        let mut args = vec!["baseline", "score", "--input", &plain, "--baseline", "web"];
        args.extend_from_slice(option);
        let line = scratch.fails(&args, 2);
        assert!(line.contains(option[0]), "{line}");
        assert!(line.contains("zip archives"), "{line}");
    }
    assert!(!scratch.path("project").join("work").exists());
    assert!(!Path::new("somewhere").exists());

    // An entry pattern selects what is scored.
    let document = scratch.score(&path, "pattern", &[]);
    assert_eq!(document["scored"], 4);
    let work = scratch.arg("pattern-2");
    let document = scratch.json(&[
        "baseline",
        "score",
        "--input",
        &path,
        "--baseline",
        "web",
        "--work-dir",
        &work,
        "--entry",
        "*.log",
    ]);
    assert_eq!(document["scored"], 2);
    assert_eq!(document["input"]["entries_not_read"]["no_pattern_match"], 3);
}

/// On request: checks the parts of a finished run against its manifest, for a run too
/// large to read into memory at once. `L2D_PARTS_DIR` names the work directory.
///
/// ```text
/// $env:L2D_PARTS_DIR = "D:\BuildArtifacts\log2decision\task-1.17\work\ait-all-512MB"
/// cargo test --release -j 4 -p l2d-cli --test archive_parts -- --ignored --nocapture
/// ```
///
/// Checked: every part has the rows, bytes and SHA-256 the manifest states; the rows
/// are numbered from 0 without a gap; no text occurs twice (by the first 128 bits of
/// its SHA-256); there is one occurrence count per row, none of them 0, and they add
/// up to the records read less the skipped ones.
#[test]
#[ignore = "on request: reads the parts of a run named by L2D_PARTS_DIR; run in release"]
fn the_parts_of_a_run_hold_one_row_per_distinct_log() {
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;

    let dir = PathBuf::from(std::env::var_os("L2D_PARTS_DIR").expect("L2D_PARTS_DIR is set"));
    let manifest = manifest(&dir);
    let part_frames = |name: &str| {
        let dir = dir.clone();
        table(&manifest, name)["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(move |part| {
                let path = dir.join(part["file"].as_str().unwrap());
                let bytes = std::fs::read(&path).unwrap();
                assert_eq!(part["bytes"], bytes.len() as u64, "{}", path.display());
                let hash: String = Sha256::digest(&bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect();
                assert_eq!(part["sha256"], hash.as_str(), "{}", path.display());
                let frame = read_part(&path).unwrap();
                assert_eq!(part["rows"], frame.height() as u64, "{}", path.display());
                frame
            })
    };

    let mut seen: HashSet<[u8; 16]> = HashSet::new();
    let (mut rows, mut text_bytes) = (0u64, 0u64);
    for frame in part_frames("scores") {
        let row = frame.column("row").and_then(Column::u64).unwrap();
        let text = frame.column("text").and_then(Column::str).unwrap();
        for (row, text) in row.iter().zip(text.iter()) {
            assert_eq!(row, Some(rows));
            let text = text.expect("a row has a text");
            let mut key = [0u8; 16];
            key.copy_from_slice(&Sha256::digest(text.as_bytes())[..16]);
            assert!(seen.insert(key), "row {rows} repeats the text {text:?}");
            rows += 1;
            text_bytes += text.len() as u64;
        }
    }
    let (mut counted, mut occurrences) = (0u64, 0u64);
    for frame in part_frames("occurrences") {
        let row = frame.column("row").and_then(Column::u64).unwrap();
        let count = frame.column("occurrences").and_then(Column::u32).unwrap();
        for (row, count) in row.iter().zip(count.iter()) {
            assert_eq!(row, Some(counted));
            let count = count.expect("a row has a count");
            assert!(count > 0, "row {counted} has no occurrence");
            counted += 1;
            occurrences += u64::from(count);
        }
    }
    let input = &manifest["input"];
    eprintln!(
        "{}: {rows} rows with {text_bytes} bytes of text, all distinct; {counted} counts that \
         add up to {occurrences}; the manifest states {} rows, {} distinct texts, {} records, \
         {} skipped",
        dir.display(),
        manifest["rows"],
        input["distinct_texts"],
        input["records"],
        input["skipped_count"]
    );
    assert_eq!(manifest["rows"], rows);
    assert_eq!(input["distinct_texts"], rows);
    assert_eq!(counted, rows);
    assert_eq!(
        occurrences,
        input["records"].as_u64().unwrap() - input["skipped_count"].as_u64().unwrap()
    );
}

#[test]
fn a_damaged_entry_fails_the_command_after_the_parts_are_written() {
    let scratch = Scratch::new("damaged");
    let mut bytes = archive(&[
        ("a.log", b"first\n"),
        (
            "broken.log",
            "a line that is repeated\n".repeat(50).as_bytes(),
        ),
        ("z.log", b"last\n"),
    ]);
    // Break the deflate stream of the second entry: its data follows its name.
    let name_at = bytes
        .windows(b"broken.log".len())
        .position(|window| window == b"broken.log")
        .unwrap();
    for byte in &mut bytes[name_at + 10..name_at + 14] {
        *byte = 0xff;
    }
    let path = scratch.write("damaged.zip", &bytes);
    let work = scratch.arg("run");
    let output = scratch.run(&[
        "--json",
        "baseline",
        "score",
        "--input",
        &path,
        "--baseline",
        "web",
        "--work-dir",
        &work,
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    let out = stdout(&output);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2, "{out}");
    let report: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(report["scored"], 2);
    assert_eq!(report["input"]["entries_not_read"]["damaged"], 1);
    let envelope: Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(envelope["error"]["exit_code"], 2);

    let rows = scored(&scratch.path("run"));
    let texts: Vec<&str> = rows.iter().map(|row| row.text.as_str()).collect();
    assert_eq!(texts, ["first", "last"]);
    let manifest = manifest(&scratch.path("run"));
    assert!(manifest["input"]["entries"][1]["skipped"]["damaged"].is_string());
}

// --- Task 1.20 -----------------------------------------------------------------------------

/// `count` distinct lines whose keys all fall into partition 0 of deduplication on
/// disk: the partition of a text is chosen by the first byte of its key, which is the
/// first byte of its SHA-256.
fn lines_of_one_partition(count: usize) -> String {
    use sha2::{Digest, Sha256};
    let mut lines = String::new();
    let mut found = 0;
    let mut index = 0u64;
    while found < count {
        let text = format!("h{index}");
        if Sha256::digest(text.as_bytes())[0] % 64 == 0 {
            lines.push_str(&text);
            lines.push('\n');
            found += 1;
        }
        index += 1;
    }
    lines
}

#[test]
fn a_partition_whose_keys_do_not_fit_is_refused_and_leaves_whole_documents() {
    let scratch = Scratch::new("partition");
    let path = scratch.write(
        "one-partition.zip",
        &archive(&[("hostile.log", lines_of_one_partition(200_000).as_bytes())]),
    );
    let refusal = |line: &str| {
        assert!(line.starts_with("error: refused: "), "{line}");
        for part in ["partition 0 of 64", "needs", "is left", " bytes)"] {
            assert!(line.contains(part), "{part}: {line}");
        }
    };

    // `ingest`: exit code 4, no report, and nothing left of its work directory.
    let ingest_work = scratch.arg("ingest-run");
    let line = scratch.fails(
        &[
            "--memory-limit",
            "44MB",
            "ingest",
            "--input",
            &path,
            "--work-dir",
            &ingest_work,
        ],
        4,
    );
    refusal(&line);
    assert!(!scratch.path("ingest-run").exists());
    let line = scratch.fails(&["--memory-limit", "44MB", "ingest", "--input", &path], 4);
    refusal(&line);
    assert!(!scratch.path("project").join("work").exists());

    // `baseline score` with `--json`: exit code 4, and standard output holds whole
    // documents only, the last of them the error envelope.
    let work = scratch.arg("score-run");
    let output = scratch.run(&[
        "--json",
        "--memory-limit",
        "53MB",
        "baseline",
        "score",
        "--input",
        &path,
        "--baseline",
        "web",
        "--work-dir",
        &work,
    ]);
    assert_eq!(output.status.code(), Some(4), "{}", stderr(&output));
    refusal(stderr(&output).trim_end());
    let out = stdout(&output);
    let documents: Vec<Value> = out
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|error| panic!("{error}: {line}")))
        .collect();
    assert_eq!(documents.len(), 2, "{out}");
    assert_eq!(documents[0]["command"], "baseline score");
    assert_eq!(documents[0]["incomplete"], true, "{out}");
    assert!(documents[0]["scores"].is_array(), "{out}");
    assert!(documents[0].get("parts").is_none(), "{out}");
    assert_eq!(documents[1]["error"]["exit_code"], 4);
    assert_eq!(documents[1]["error"]["kind"], "refused");
    // What a run that failed may leave: the parts written until then, without a
    // manifest; nothing of the deduplication on disk.
    let left = names(&scratch.path("score-run"));
    assert!(
        !left.is_empty(),
        "the texts of the set in memory were scored"
    );
    for name in &left {
        assert!(
            name.starts_with("scores-") && name.ends_with(".parquet"),
            "{left:?}"
        );
    }
}

#[test]
fn the_disk_estimate_holds_for_an_archive_of_short_lines() {
    let scratch = Scratch::new("short-lines");
    // 400,000 distinct lines of 1 to 6 digits: a row on disk takes 32 bytes beside a
    // text of about 6.
    let lines = 400_000u64;
    let content: String = (0..lines).map(|line| format!("{line}\n")).collect();
    let path = scratch.write("short.zip", &archive(&[("short.log", content.as_bytes())]));

    let scored = scratch.score(&path, "scored", &["--memory-limit", "64MB"]);
    let parts = &scored["parts"];
    let dedup = &parts["deduplication"];
    assert_eq!(dedup["on_disk"], true, "{parts}");
    assert_eq!(scored["input"]["distinct_texts"], lines);
    let spilled = dedup["spilled_bytes"].as_u64().unwrap();
    // Every row that went to disk is counted with its 32 bytes.
    assert!(
        spilled >= 32 * dedup["spilled_records"].as_u64().unwrap(),
        "{dedup}"
    );
    let written = spilled + parts["bytes"].as_u64().unwrap();
    let estimated = parts["disk_estimated_bytes"].as_u64().unwrap();
    assert!(
        written <= estimated,
        "{written} bytes were written and {estimated} estimated: {parts}"
    );

    let ingested = scratch.json(&["--memory-limit", "50MB", "ingest", "--input", &path]);
    let work = &ingested["work"];
    assert_eq!(work["deduplication"]["on_disk"], true, "{work}");
    assert_eq!(ingested["input"]["distinct_texts"], lines);
    let spilled = work["deduplication"]["spilled_bytes"].as_u64().unwrap();
    let estimated = work["disk_estimated_bytes"].as_u64().unwrap();
    assert!(
        spilled <= estimated,
        "{spilled} bytes were written and {estimated} estimated: {work}"
    );
}

#[test]
fn a_work_directory_the_run_made_is_removed_when_the_run_is_refused() {
    let scratch = Scratch::new("made");
    let path = scratch.write("sample.zip", &sample());
    // The text output has its first lines by the time the first batch is refused, so
    // only the exit code and the diagnostics are looked at.
    let score = |limit: &'static str, work: &str, code: i32| {
        let output = scratch.run(&[
            "--memory-limit",
            limit,
            "baseline",
            "score",
            "--input",
            &path,
            "--baseline",
            "web",
            "--work-dir",
            work,
        ]);
        assert_eq!(output.status.code(), Some(code), "{}", stderr(&output));
        stderr(&output).trim_end().to_string()
    };

    // Under this limit the scorer fits and the first batch does not: the run is
    // refused after its work directory was made.
    let made = scratch.path("made").join("run");
    let line = score("41MB", &made.to_string_lossy(), 4);
    assert!(line.contains("reading the archive needs"), "{line}");
    assert!(!scratch.path("made").exists(), "{:?}", names(&scratch.root));

    // A directory that was there before stays, empty as it was.
    std::fs::create_dir(scratch.path("there")).unwrap();
    score("41MB", &scratch.arg("there"), 4);
    assert_eq!(names(&scratch.path("there")), Vec::<String>::new());

    // Below a file: the analyst's error, said plainly.
    let file = scratch.write("plain.txt", b"x");
    let below = Path::new(&file).join("run");
    let line = score("2GB", &below.to_string_lossy(), 2);
    assert!(line.contains("is a file"), "{line}");
    assert!(!line.contains("os error"), "{line}");
    assert_eq!(std::fs::read(&file).unwrap(), b"x");
}
