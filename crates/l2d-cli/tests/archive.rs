//! `l2d ingest` on a zip archive (requirement 13; task 1.16), run as the built binary
//! against archives written into a temporary directory.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::Value;

/// A temporary directory with an empty project in it, removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-cli-archive-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("project")).unwrap();
        Self { root }
    }

    fn write(&self, name: &str, bytes: &[u8]) -> String {
        let path = self.root.join(name);
        std::fs::write(&path, bytes).unwrap();
        path.to_string_lossy().into_owned()
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

    /// What is in the directory beside the project, sorted.
    fn files(&self) -> Vec<String> {
        fn walk(dir: &Path, found: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, found);
                }
                found.push(path.to_string_lossy().into_owned());
            }
        }
        let mut found = Vec::new();
        walk(&self.root, &mut found);
        found.sort();
        found
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

/// Two logs (one of them gzip), a packet capture and a directory.
fn sample() -> Vec<u8> {
    let rotated = gzip(b"GET /b 200\nGET /a 200\n");
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
fn ingesting_an_archive_displays_its_entries_and_records() {
    let scratch = Scratch::new("text");
    let path = scratch.write("sample.zip", &sample());
    let before = scratch.files();
    let out = scratch.text(&["ingest", "--input", &path]);
    for expected in [
        "  adapter: zip\n",
        "  records read: 6 (1 skipped)\n",
        "  distinct texts: 3 (0 good, 0 bad, 3 unknown)\n",
        "  content SHA-256: ",
        "  entries: 4 in the archive, 2 read, 2 not read\n",
        "    not read: 1 not text, 1 directories\n",
        "  entries read: 2\n",
        "    logs/access.log: 4 records\n",
        "    logs/access.log.1.gz: 2 records\n",
        "  entries not read: 1\n",
        "    capture/eth0.pcap: not text\n",
        "  skipped records: 1\n",
        "    logs/access.log line 2: empty line\n",
        "memory: free memory ",
        "  reading the input: 6 records in batches of at most ",
    ] {
        assert!(out.contains(expected), "{expected:?} is not in:\n{out}");
    }
    assert!(out.starts_with(&format!("input: {path}\n")), "{out}");
    assert_eq!(scratch.files(), before, "nothing is written");
}

#[test]
fn ingesting_an_archive_with_json_names_every_entry() {
    let scratch = Scratch::new("json");
    let bytes = sample();
    let path = scratch.write("sample.zip", &bytes);
    let value = scratch.json(&["ingest", "--input", &path]);
    assert_eq!(value["command"], "ingest");
    let input = &value["input"];
    assert_eq!(input["source_path"], path.as_str());
    assert_eq!(input["adapter"], "zip");
    assert_eq!(input["records"], 6);
    assert_eq!(input["distinct_texts"], 3);
    assert_eq!(input["label_conflicts"], 0);
    assert_eq!(input["labels"]["unknown"], 3);
    assert_eq!(input["content_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(input["skipped_count"], 1);
    assert_eq!(
        input["skipped"],
        serde_json::json!([{
            "position": (1u64 << 40) | 2,
            "entry": "logs/access.log",
            "line": 2,
            "reason": "empty line"
        }])
    );
    assert_eq!(input["entries_total"], 4);
    assert_eq!(input["entries_read"], 2);
    assert_eq!(input["entries_not_read"]["not_text"], 1);
    assert_eq!(input["entries_not_read"]["directory"], 1);
    assert_eq!(input["entries_not_read"]["damaged"], 0);
    assert_eq!(
        input["entries"],
        serde_json::json!([
            {"path": "logs/", "records": 0, "skipped": "directory"},
            {"path": "logs/access.log", "records": 4, "skipped": null},
            {"path": "logs/access.log.1.gz", "records": 2, "skipped": null},
            {"path": "capture/eth0.pcap", "records": 0, "skipped": "not_text"},
        ])
    );
    let read = &value["memory"]["operations"][0];
    assert_eq!(read["operation"], "read");
    assert_eq!(read["rows"], 6);

    // The same report whatever the batch size, and by `--adapter zip` under any name.
    let small = scratch.json(&["--batch-rows", "1", "ingest", "--input", &path]);
    assert_eq!(small["input"], value["input"]);
    assert_eq!(small["memory"]["operations"][0]["batches"]["batch_rows"], 1);
    let renamed = scratch.write("sample.bin", &bytes);
    let other = scratch.json(&["ingest", "--input", &renamed, "--adapter", "zip"]);
    assert_eq!(other["input"]["entries"], value["input"]["entries"]);
    assert_eq!(
        other["input"]["content_sha256"],
        value["input"]["content_sha256"]
    );
}

#[test]
fn entry_patterns_and_the_record_length_are_options_of_ingest() {
    let scratch = Scratch::new("options");
    let path = scratch.write("sample.zip", &sample());
    let value = scratch.json(&["ingest", "--input", &path, "--entry", "*.log"]);
    assert_eq!(value["input"]["records"], 4);
    assert_eq!(value["input"]["entries_read"], 1);
    assert_eq!(value["input"]["entries_not_read"]["no_pattern_match"], 3);
    let value = scratch.json(&[
        "ingest", "--input", &path, "--entry", "*.gz", "--entry", "*.pcap",
    ]);
    assert_eq!(value["input"]["records"], 2);
    assert_eq!(value["input"]["entries_not_read"]["not_text"], 1);

    // Records longer than 9 bytes are skipped and reported with their length.
    let value = scratch.json(&["ingest", "--input", &path, "--max-record-bytes", "9"]);
    assert_eq!(value["input"]["records"], 6);
    assert_eq!(value["input"]["distinct_texts"], 0);
    assert_eq!(value["input"]["skipped_count"], 6);
    let reason = value["input"]["skipped"][0]["reason"].as_str().unwrap();
    assert!(
        reason.contains("10 bytes") && reason.contains("9 bytes"),
        "{reason}"
    );

    for (args, expected) in [
        (
            &["ingest", "--input", &path, "--entry", "logs/[a"][..],
            "logs/[a",
        ),
        (
            &["ingest", "--input", &path, "--max-record-bytes", "2MB"][..],
            "--max-record-bytes",
        ),
        (
            &["ingest", "--input", &path, "--max-record-bytes", "0"][..],
            "--max-record-bytes",
        ),
        (
            &["ingest", "--input", &path, "--text-column", "message"][..],
            "zip adapter",
        ),
    ] {
        let line = scratch.fails(args, 2);
        assert!(line.starts_with("error: invalid input: "), "{line}");
        assert!(line.contains(expected), "{line}");
    }

    // The options of an archive on an input that is none.
    let lines = scratch.write("plain.log", b"one\n");
    for option in [
        &["--entry", "*.log"][..],
        &["--max-record-bytes", "1KB"][..],
    ] {
        let mut args = vec!["ingest", "--input", &lines];
        args.extend_from_slice(option);
        let line = scratch.fails(&args, 2);
        assert!(
            line.contains(option[0]) && line.contains("zip archives"),
            "{line}"
        );
    }
}

#[test]
fn a_damaged_entry_is_reported_and_the_command_fails() {
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

    let output = scratch.run(&["ingest", "--input", &path]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    let out = stdout(&output);
    assert!(out.contains("  records read: 2 (0 skipped)\n"), "{out}");
    assert!(
        out.contains("  entries: 3 in the archive, 2 read, 1 not read\n"),
        "{out}"
    );
    assert!(out.contains("    broken.log: damaged: "), "{out}");
    let diagnostics = stderr(&output);
    assert_eq!(diagnostics.lines().count(), 1, "{diagnostics}");
    assert!(
        diagnostics.starts_with("error: invalid input: 1 of the 3 entries of "),
        "{diagnostics}"
    );
    assert!(diagnostics.contains("`broken.log`"), "{diagnostics}");

    // With `--json`: the report on the first line, the error envelope on the last.
    let output = scratch.run(&["--json", "ingest", "--input", &path]);
    assert_eq!(output.status.code(), Some(2));
    let out = stdout(&output);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2, "{out}");
    let report: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(report["input"]["records"], 2);
    assert_eq!(report["input"]["entries_not_read"]["damaged"], 1);
    assert!(report["input"]["entries"][1]["skipped"]["damaged"].is_string());
    let envelope: Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(envelope["error"]["kind"], "invalid_input");
    assert_eq!(envelope["error"]["exit_code"], 2);
}

#[test]
fn entry_names_are_shown_as_data() {
    let scratch = Scratch::new("names");
    let path = scratch.write(
        "names.zip",
        &archive(&[
            ("../../escaped.log", b"one\n"),
            ("red\x1b[31m.log", b"two\n"),
            ("C:\\escaped.log", b"three\n"),
        ]),
    );
    let before = scratch.files();
    let out = scratch.text(&["ingest", "--input", &path]);
    assert!(out.contains("  records read: 3 (0 skipped)\n"), "{out}");
    assert!(out.contains("    ../../escaped.log: 1 records\n"), "{out}");
    assert!(
        !out.contains('\x1b'),
        "control characters are escaped: {out:?}"
    );
    assert_eq!(scratch.files(), before, "nothing is written");
    assert!(!scratch.root.parent().unwrap().join("escaped.log").exists());
}

#[test]
fn a_file_that_is_no_archive_and_a_limit_that_is_too_small_fail_with_the_reason() {
    let scratch = Scratch::new("refusals");
    let not_zip = scratch.write("not.zip", b"this is a text file\n");
    let line = scratch.fails(&["ingest", "--input", &not_zip], 2);
    assert!(line.contains("is not a zip archive"), "{line}");

    let line = scratch.fails(&["ingest", "--input", "absent.zip"], 3);
    assert!(line.starts_with("error: missing "), "{line}");

    // 120,000 distinct lines need a deduplication set that 38 MiB do not hold.
    let mut content = String::new();
    for index in 0..120_000 {
        content.push_str(&format!("distinct line {index}\n"));
    }
    let path = scratch.write("distinct.zip", &archive(&[("all.log", content.as_bytes())]));
    let line = scratch.fails(&["ingest", "--input", &path, "--memory-limit", "38MB"], 4);
    assert!(line.starts_with("error: refused: "), "{line}");
    assert!(line.contains("deduplication set"), "{line}");
    assert!(line.contains("(39845888 bytes)"), "{line}");
    let value = scratch.json(&["ingest", "--input", &path, "--memory-limit", "64MB"]);
    assert_eq!(value["input"]["distinct_texts"], 120_000);
}

#[test]
fn a_line_that_starts_with_a_byte_order_mark_is_read_at_any_batch_size() {
    let scratch = Scratch::new("marks");
    let path = scratch.write(
        "marks.zip",
        &archive(&[("a.log", "a\n\u{feff}b\nc\n\u{feff}b\n".as_bytes())]),
    );
    let lines = scratch.write("marks.log", "a\n\u{feff}b\nc\n\u{feff}b\n".as_bytes());
    let mut first = None;
    for batch_rows in ["1", "2", "65536"] {
        let value = scratch.json(&["--batch-rows", batch_rows, "ingest", "--input", &path]);
        assert_eq!(value["input"]["records"], 4, "batch size {batch_rows}");
        assert_eq!(value["input"]["distinct_texts"], 3);
        assert_eq!(value["input"]["skipped_count"], 0);
        assert_eq!(first.get_or_insert(value["input"].clone()), &value["input"]);
        // The same file unpacked, by the lines adapter.
        let plain = scratch.json(&["--batch-rows", batch_rows, "ingest", "--input", &lines]);
        assert_eq!(plain["input"]["records"], 4, "batch size {batch_rows}");
        assert_eq!(plain["input"]["distinct_texts"], 3);
    }
}

#[test]
fn the_other_commands_do_not_read_an_archive_yet() {
    let scratch = Scratch::new("others");
    let path = scratch.write("sample.zip", &sample());
    for args in [
        &["baseline", "fit", "--input", &path, "--name", "x"][..],
        &[
            "baseline",
            "fit",
            "--input",
            &path,
            "--name",
            "x",
            "--adapter",
            "zip",
        ][..],
    ] {
        let line = scratch.fails(args, 4);
        assert!(line.contains("does not read zip archives yet"), "{line}");
        assert!(line.contains("l2d ingest"), "{line}");
    }
}

// --- Deduplication on disk (requirement 13.12; task 1.20) -----------------------------------

/// `lines` distinct lines, each twice: the second time in another entry, in reverse.
fn twice(lines: u32) -> Vec<u8> {
    let mut forward = String::new();
    let mut backward = String::new();
    for line in 0..lines {
        forward.push_str(&format!("line {line}\n"));
        backward.push_str(&format!("line {}\n", lines - 1 - line));
    }
    archive(&[
        ("forward.log", forward.as_bytes()),
        ("backward.log", backward.as_bytes()),
    ])
}

/// The limit under which the set of the 100,000 texts of `twice` does not fit and the
/// buffers of deduplication on disk do fit beside it when it is frozen: at 40 MiB the
/// read is refused at the freeze, at 42 MiB the set fits. It depends on the constants
/// of the memory model.
const SHORT: &str = "41MB";

#[test]
fn an_archive_whose_texts_outgrow_the_memory_limit_is_ingested_with_deduplication_on_disk() {
    let scratch = Scratch::new("on-disk");
    let lines = 100_000u64;
    let path = scratch.write("many.zip", &twice(lines as u32));
    let before = scratch.files();

    let roomy = scratch.json(&["--memory-limit", "2GB", "ingest", "--input", &path]);
    assert_eq!(roomy["input"]["records"], 2 * lines);
    assert_eq!(roomy["input"]["distinct_texts"], lines);
    // In memory the report is the one of task 1.16: no `work`.
    assert!(roomy.get("work").is_none(), "{roomy}");

    // Under a limit the set does not fit: the same report, by way of the disk, at any
    // batch size.
    for batch_rows in [None, Some("3000")] {
        let mut args = vec!["--memory-limit", SHORT];
        if let Some(rows) = batch_rows {
            args.extend_from_slice(&["--batch-rows", rows]);
        }
        args.extend_from_slice(&["ingest", "--input", &path]);
        let short = scratch.json(&args);
        let work = &short["work"];
        let dedup = &work["deduplication"];
        assert_eq!(dedup["on_disk"], true, "{batch_rows:?}: {work}");
        let in_memory = dedup["set_texts"].as_u64().unwrap();
        assert!(in_memory > 0 && in_memory < lines, "{dedup}");
        assert_eq!(
            dedup["spilled_distinct"].as_u64().unwrap() + in_memory,
            lines
        );
        assert_eq!(short["input"], roomy["input"], "{batch_rows:?}");
        // What was written stayed within what the run was estimated to need.
        assert!(
            dedup["spilled_bytes"].as_u64().unwrap()
                <= work["disk_estimated_bytes"].as_u64().unwrap(),
            "{work}"
        );
        // The disk was checked at the freeze, for deduplication on disk alone: the
        // unpacked bytes and 32 bytes for every 6 of them.
        let unpacked = 2
            * (0..lines)
                .map(|line| format!("line {line}\n").len() as u64)
                .sum::<u64>();
        assert_eq!(
            work["disk_estimated_bytes"].as_u64().unwrap(),
            unpacked + 32 * (unpacked / 6),
            "{work}"
        );
        assert!(work["disk_free_bytes"].as_u64().unwrap() > 0, "{work}");
        // The directory the files were in: one of this run below the project's.
        let directory = Path::new(work["directory"].as_str().unwrap());
        assert!(
            directory
                .parent()
                .unwrap()
                .ends_with(Path::new("project").join("work")),
            "{work}"
        );
        let name = directory.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("ingest-2"), "{name}");
    }
    // Without `--work-dir` the files were below the project's work directory, and
    // nothing of it is left.
    assert_eq!(scratch.files(), before, "nothing is left behind");

    // A directory that is named and absent is made and removed again, with the
    // directories made above it.
    let given = scratch.root.join("given").join("run");
    let given_arg = given.to_string_lossy().into_owned();
    let text = scratch.text(&[
        "--memory-limit",
        SHORT,
        "ingest",
        "--input",
        &path,
        "--work-dir",
        &given_arg,
    ]);
    for expected in [
        "  records read: 200000 (0 skipped)\n",
        "  distinct texts: 100000 (0 good, 0 bad, 100000 unknown)\n",
        "  entries: 2 in the archive, 2 read, 0 not read\n",
        "  deduplication: on disk after ",
    ] {
        assert!(text.contains(expected), "{expected:?} is not in:\n{text}");
    }
    assert_eq!(scratch.files(), before, "nothing is left behind");

    // A directory that was there and empty is used and left as it was.
    std::fs::create_dir(scratch.root.join("empty")).unwrap();
    let with_empty = scratch.files();
    let empty_arg = scratch.root.join("empty").to_string_lossy().into_owned();
    let short = scratch.json(&[
        "--memory-limit",
        SHORT,
        "ingest",
        "--input",
        &path,
        "--work-dir",
        &empty_arg,
    ]);
    assert_eq!(short["input"], roomy["input"]);
    assert_eq!(short["work"]["directory"], empty_arg.as_str());
    assert_eq!(scratch.files(), with_empty);
}

/// Remediation of task 1.20 after the review of d03b1db: `ingest` uses the disk only
/// when the set of distinct texts outgrows the memory, as at task 1.16 until then.
#[test]
fn an_archive_whose_texts_fit_in_memory_is_ingested_without_a_work_directory() {
    let scratch = Scratch::new("no-disk");
    let path = scratch.write("sample.zip", &sample());
    // A file is where the project's work directory would be: it cannot be made.
    let blocker = scratch.root.join("project").join("work");
    std::fs::write(&blocker, b"not a directory").unwrap();
    let before = scratch.files();

    let value = scratch.json(&["ingest", "--input", &path]);
    assert_eq!(value["input"]["distinct_texts"], 3);
    assert!(value.get("work").is_none(), "{value}");
    let text = scratch.text(&["ingest", "--input", &path]);
    assert!(text.contains("  distinct texts: 3 "), "{text}");
    assert!(!text.contains("deduplication"), "{text}");
    assert!(!text.contains("  disk: "), "{text}");
    // Nothing is held for deduplication on disk beside a set that fits: under this
    // limit the buffers for it (3.4 MiB) would not fit beside a batch.
    let value = scratch.json(&["--memory-limit", "40MB", "ingest", "--input", &path]);
    assert_eq!(value["input"]["records"], 6);
    assert_eq!(scratch.files(), before, "nothing is written");
    assert_eq!(std::fs::read(&blocker).unwrap(), b"not a directory");

    // When the set does outgrow the memory, the directory is made then: here it
    // cannot be, which is said plainly, after the read began and with no report.
    let many = scratch.write("many.zip", &twice(100_000));
    let line = scratch.fails(&["--memory-limit", SHORT, "ingest", "--input", &many], 2);
    assert!(line.contains("is a file"), "{line}");
    let output = scratch.run(&[
        "--json",
        "--memory-limit",
        SHORT,
        "ingest",
        "--input",
        &many,
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    let out = stdout(&output);
    let documents: Vec<Value> = out
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|error| panic!("{error}: {line}")))
        .collect();
    assert_eq!(documents.len(), 1, "{out}");
    assert_eq!(documents[0]["error"]["exit_code"], 2, "{out}");
    std::fs::remove_file(&many).unwrap();
    assert_eq!(scratch.files(), before, "nothing is written");
}

#[test]
fn the_work_directory_of_ingest_is_checked_before_anything_is_read() {
    let scratch = Scratch::new("work-dir");
    let path = scratch.write("sample.zip", &sample());

    // Below a file: the analyst's error, said plainly.
    let file = scratch.write("plain.txt", b"x");
    let before = scratch.files();
    let below = Path::new(&file).join("run").to_string_lossy().into_owned();
    let line = scratch.fails(&["ingest", "--input", &path, "--work-dir", &below], 2);
    assert!(line.contains("is a file"), "{line}");
    assert!(!line.contains("os error"), "{line}");

    // In use: refused, and left alone.
    std::fs::create_dir(scratch.root.join("used")).unwrap();
    let kept = scratch.write("used/notes.txt", b"mine");
    let used = scratch.root.join("used").to_string_lossy().into_owned();
    let line = scratch.fails(&["ingest", "--input", &path, "--work-dir", &used], 4);
    assert!(line.contains("not an empty directory"), "{line}");
    assert_eq!(std::fs::read(&kept).unwrap(), b"mine");

    // On an input that is not an archive the option is an analyst error.
    let plain = scratch.write("plain.log", b"GET /a 200\n");
    let line = scratch.fails(&["ingest", "--input", &plain, "--work-dir", "somewhere"], 2);
    assert!(line.contains("--work-dir"), "{line}");
    assert!(line.contains("zip archives"), "{line}");
    assert!(!Path::new("somewhere").exists());
    std::fs::remove_file(&plain).unwrap();
    std::fs::remove_dir_all(scratch.root.join("used")).unwrap();
    assert_eq!(scratch.files(), before);
}
