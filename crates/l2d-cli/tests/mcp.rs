//! Triager task 1.2: `l2d mcp` driven as the built binary by a real MCP stdio client,
//! and compared with the existing command paths (`baseline score`, `discover search`)
//! on the same archive: resource figures, native memberships and results.

mod support;

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Output, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use l2d_core::selections::{Member, SelectionSet};
use l2d_core::{ArchivePosition, MemoryBudget, MemorySettings, ZipInputSettings};
use l2d_triager::scope::{NativeKey, Scope, ScopeSettings, ScoreManifest};
use serde_json::{Value, json};

const GIB: u64 = 1024 * 1024 * 1024;
const MAX_RESULT_BYTES: usize = 32 * 1024;

/// A temporary directory with a project that holds the fitted scorer `web`.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-cli-mcp-{}-{}-{name}",
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
        std::fs::write(scratch.path("labelled.csv"), labelled).unwrap();
        scratch
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn arg(&self, name: &str) -> String {
        self.path(name).to_string_lossy().into_owned()
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_l2d"));
        command.arg("--project").arg(self.root.join("project"));
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        self.command()
            .args(args)
            .output()
            .expect("the l2d binary runs")
    }

    /// Runs a command that must succeed with `--json` and returns its document.
    fn json(&self, args: &[&str]) -> Value {
        let mut all = vec!["--json"];
        all.extend_from_slice(args);
        let output = self.run(&all);
        let diagnostics = support::diagnostics(&output);
        assert_eq!(output.status.code(), Some(0), "{diagnostics}");
        assert_eq!(diagnostics, "");
        serde_json::from_slice(&output.stdout).unwrap()
    }

    /// Writes an archive with the entries `(name, lines)`.
    fn archive(&self, name: &str, entries: &[(String, Vec<String>)]) -> String {
        let path = self.path(name);
        let mut writer = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        for (entry, lines) in entries {
            writer
                .start_file(entry.as_str(), zip::write::SimpleFileOptions::default())
                .unwrap();
            for line in lines {
                writer.write_all(line.as_bytes()).unwrap();
                writer.write_all(b"\n").unwrap();
            }
        }
        writer.finish().unwrap();
        path.to_string_lossy().into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn fitted(name: &str) -> Scratch {
    let scratch = Scratch::new(name);
    let labelled = scratch.arg("labelled.csv");
    let output = scratch.run(&["baseline", "fit", "--input", &labelled, "--name", "web"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        support::diagnostics(&output)
    );
    scratch
}

/// An MCP client over the standard streams of a running `l2d mcp`.
struct Client {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next: u64,
    stderr: PathBuf,
}

impl Client {
    fn spawn(scratch: &Scratch, global: &[&str], args: &[&str]) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let stderr = scratch.path(&format!(
            "mcp-stderr-{}.txt",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut child = scratch
            .command()
            .args(global)
            .arg("mcp")
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(std::fs::File::create(&stderr).unwrap())
            .spawn()
            .expect("the l2d binary runs");
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut client = Self {
            child,
            stdin,
            stdout,
            next: 0,
            stderr,
        };
        let initialized = client.request(
            "initialize",
            json!({"protocolVersion": "2025-11-25", "capabilities": {},
                   "clientInfo": {"name": "l2d-cli tests", "version": "0"}}),
        );
        assert_eq!(initialized["result"]["protocolVersion"], "2025-11-25");
        assert_eq!(
            initialized["result"]["capabilities"]["tools"]["listChanged"],
            false
        );
        client.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        client
    }

    fn send(&mut self, message: &Value) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{message}").unwrap();
        stdin.flush().unwrap();
    }

    /// Sends a request and reads its reply: one line of JSON-RPC with the same ID.
    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next += 1;
        let id = self.next;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line).unwrap();
        assert!(
            read > 0,
            "the server closed its output: {}",
            self.diagnostics()
        );
        let reply: Value = serde_json::from_str(&line)
            .unwrap_or_else(|error| panic!("standard output holds {line:?}: {error}"));
        assert_eq!(reply["jsonrpc"], "2.0", "{reply}");
        assert_eq!(reply["id"], id, "{reply}");
        reply
    }

    /// Calls a tool and returns its structured result and whether it is an error.
    fn tool(&mut self, tool: &str, arguments: Value) -> (Value, bool) {
        let reply = self.request("tools/call", json!({"name": tool, "arguments": arguments}));
        assert!(reply.get("error").is_none(), "{reply}");
        let result = &reply["result"];
        let bytes = serde_json::to_string(result).unwrap().len();
        assert!(
            bytes <= MAX_RESULT_BYTES,
            "{tool}: a result of {bytes} bytes"
        );
        let structured = result["structuredContent"].clone();
        let text: Value =
            serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, structured);
        assert_eq!(structured["v"], 1, "{structured}");
        (structured, result["isError"] == true)
    }

    /// Calls a tool that must succeed; a job that is still running is waited for.
    fn ok(&mut self, tool: &str, arguments: Value) -> Value {
        let (value, is_error) = self.tool(tool, arguments.clone());
        assert!(!is_error, "{tool} {arguments}: {value}");
        if value["status"] == "running" {
            let job = value["job"].as_str().unwrap().to_string();
            let ended = self.wait(&job);
            assert_eq!(ended["status"], "completed", "{ended}");
            return ended["result"].clone();
        }
        value
    }

    fn job(&mut self, job: &str, action: &str) -> Value {
        let (value, is_error) =
            self.tool("triager_job", json!({"v": 1, "job": job, "action": action}));
        assert!(!is_error, "{value}");
        value
    }

    fn wait(&mut self, job: &str) -> Value {
        let started = Instant::now();
        loop {
            let status = self.job(job, "status");
            let state = status["status"].as_str().unwrap().to_string();
            if !matches!(state.as_str(), "waiting" | "running" | "cancel_requested") {
                return status;
            }
            assert!(started.elapsed() < Duration::from_secs(300), "{status}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Every record of `handle`, page by page.
    fn records(&mut self, handle: &str, sets: Value) -> Vec<Value> {
        let mut rows = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut args = json!({"v": 1, "handle": handle, "view": "records", "sets": sets});
            if let Some(cursor) = &cursor {
                args["cursor"] = json!(cursor);
            }
            let page = self.ok("triager_inspect", args);
            rows.extend(page["rows"].as_array().unwrap().iter().cloned());
            match page["page"]["next_cursor"].as_str() {
                Some(next) => cursor = Some(next.to_string()),
                None => return rows,
            }
        }
    }

    fn activity(&mut self, snapshot: &str) -> Value {
        self.ok(
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "overview"}),
        )["activity"]
            .clone()
    }

    fn diagnostics(&self) -> String {
        std::fs::read_to_string(&self.stderr).unwrap_or_default()
    }

    /// Closes the server's input; it must end with exit code 0, nothing more on
    /// standard output and no diagnostics beside waits for the host.
    fn close(mut self) {
        drop(self.stdin.take());
        let mut rest = String::new();
        std::io::Read::read_to_string(&mut self.stdout, &mut rest).unwrap();
        assert_eq!(rest, "", "output after the last reply");
        let status = self.child.wait().unwrap();
        let diagnostics: String = self
            .diagnostics()
            .lines()
            .filter(|line| !line.starts_with("waiting: "))
            .collect();
        assert_eq!(status.code(), Some(0), "{diagnostics}");
        assert_eq!(diagnostics, "");
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn web_entries() -> Vec<(String, Vec<String>)> {
    let mut access = Vec::new();
    for index in 0..120 {
        access.push(format!("GET /page/{} 200", index % 45));
        if index % 6 == 0 {
            access.push(format!("GET /page/{}?id=1%27%20OR%201=1-- 500", index % 20));
        }
        if index % 8 == 0 {
            access.push(format!("POST /login?user=u{} 302", index % 12));
        }
    }
    let mut auth = Vec::new();
    for index in 0..90 {
        auth.push(format!("login ok user=u{}", index % 12));
        if index % 5 == 0 {
            auth.push(format!(
                "GET /login?next=/page/{}%27%20OR%201=1-- 500",
                index % 9
            ));
        }
        if index % 9 == 0 {
            // Also in the first entry: its first position stays there.
            auth.push(format!("GET /page/{} 200", index % 45));
        }
    }
    vec![
        ("logs/access.log".into(), access),
        ("logs/auth.log".into(), auth),
        ("notes/readme.txt".into(), vec!["login notes".into()]),
    ]
}

fn record_id(position: u64) -> String {
    format!("r-{position:016x}")
}

fn ids(rows: &[Value]) -> Vec<String> {
    rows.iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn an_mcp_client_reduces_an_archive_to_a_minus_b_and_explains_one_row() {
    let scratch = fitted("reduce");
    let entries = web_entries();
    let archive = scratch.archive("logs.zip", &entries);

    // The existing command paths on the same archive and entry filter.
    let scored = scratch.json(&[
        "baseline",
        "score",
        "--input",
        &archive,
        "--baseline",
        "web",
        "--entry",
        "*.log",
        "--limit",
        "100000",
        "--work-dir",
        &scratch.arg("scored"),
    ]);
    let entry_index = |path: &str| entries.iter().position(|(name, _)| name == path).unwrap();
    // (position, ratio, text), in the order of the top-k rule.
    let mut ranked: Vec<(u64, f32, String)> = scored["scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let position = ArchivePosition {
                entry: entry_index(row["entry"].as_str().unwrap()) as u32,
                line: row["line"].as_u64().unwrap(),
            }
            .pack();
            (
                position,
                row["ratio"].as_f64().unwrap() as f32,
                row["text"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(scored["rows_omitted"], 0);
    let originals = ranked.len() as u64;
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let k = 25;
    let mut native_a: Vec<u64> = ranked[..k].iter().map(|row| row.0).collect();
    native_a.sort_unstable();
    scratch.json(&[
        "discover",
        "search",
        "--input",
        &archive,
        "--entry",
        "*.log",
        "--glob",
        "*login*",
        "--out",
        &scratch.arg("selected"),
    ]);
    let mut budget =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB)).unwrap();
    let selection =
        SelectionSet::load(&scratch.path("selected").join("selection.json"), &budget).unwrap();
    let native_b: Vec<u64> = selection
        .members()
        .iter()
        .map(|member| match member {
            Member::Position(position) => *position,
            Member::StoreId(id) => panic!("a store member {id}"),
        })
        .collect();
    let native_d: Vec<u64> = native_a
        .iter()
        .copied()
        .filter(|position| !native_b.contains(position))
        .collect();
    let overlap = native_a.len() - native_d.len();
    assert!(
        overlap > 0 && !native_d.is_empty(),
        "the fixture has both regions"
    );

    // The same reduction from an MCP client.
    let mut client = Client::spawn(
        &scratch,
        &["--memory-limit", "512MB"],
        &["--source", &format!("logs={archive}")],
    );
    let listed = client.request("tools/list", json!({}));
    assert_eq!(listed["result"]["tools"].as_array().unwrap().len(), 7);
    let opened = client.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "logs", "entries": ["*.log"]}}),
    );
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    // Resource figures and source identity are those of the existing command.
    assert_eq!(opened["counts"]["distinct_originals"], originals);
    assert_eq!(
        opened["counts"]["distinct_originals"],
        scored["input"]["distinct_texts"]
    );
    assert_eq!(
        opened["completeness"]["records_scanned"],
        scored["input"]["records"]
    );
    assert_eq!(
        opened["completeness"]["skipped_records"],
        scored["input"]["skipped_count"]
    );
    assert_eq!(
        opened["completeness"]["entries_read"],
        scored["input"]["entries_read"]
    );
    assert_eq!(
        opened["source"]["content_sha256"],
        scored["input"]["content_sha256"]
    );
    assert_eq!(opened["counts"]["occurrences"], scored["input"]["records"]);

    let select_a = json!({"v": 1, "scope": snapshot, "idempotency_key": "set-a", "producer": {
        "kind": "score", "model": "web", "top_k": k, "ties": "record_id_ascending"}});
    let set_a = client.ok("triager_select", select_a.clone());
    assert_eq!(
        set_a["producer"]["scorer"]["revision"],
        scored["scorer"]["revision"]
    );
    assert_eq!(
        set_a["producer"]["scorer"]["kind"],
        scored["scorer"]["kind"]
    );
    assert_eq!(set_a["cost"]["records_scanned"], scored["input"]["records"]);
    assert_eq!(set_a["cost"]["rows_scored"], scored["scored"]);
    let set_b = client.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "filter", "syntax": "glob", "query": "*login*"}}),
    );
    let (a, b) = (
        set_a["set"].as_str().unwrap().to_string(),
        set_b["set"].as_str().unwrap().to_string(),
    );
    let expect = |positions: &[u64]| positions.iter().map(|p| record_id(*p)).collect::<Vec<_>>();
    assert_eq!(ids(&client.records(&a, json!({}))), expect(&native_a));
    assert_eq!(ids(&client.records(&b, json!({}))), expect(&native_b));

    // Counts first: the overlap, then A minus B.
    let regions = client.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions", "sets": {"a": a, "b": b}}),
    );
    let both = regions["regions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["membership"] == json!({"a": "in", "b": "in"}))
        .unwrap();
    assert_eq!(both["distinct_originals"], overlap as u64);
    let view = client.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"a": a, "b": b},
               "expr": {"difference": ["a", "b"]}}),
    );
    let d = view["view"].as_str().unwrap().to_string();
    assert_eq!(view["counts"]["parent"], originals);
    assert_eq!(view["counts"]["in"], native_d.len() as u64);
    assert_eq!(view["counts"]["unknown"], 0);
    assert_eq!(
        view["occurrence_counts"]["parent"],
        scored["input"]["records"]
    );
    let rows = client.records(&d, json!({"a": a, "b": b}));
    assert_eq!(ids(&rows), expect(&native_d));

    // One explanatory row, with the score and the text the existing command gave.
    let first = ranked
        .iter()
        .find(|row| row.0 == native_d[0])
        .unwrap()
        .clone();
    let evidence = client.ok(
        "triager_inspect",
        json!({"v": 1, "handle": d, "view": "evidence", "record": record_id(first.0),
               "sets": {"a": a, "b": b},
               "projection": ["id", "locator", "occurrences", "membership", "explanation", "excerpt"]}),
    );
    assert_eq!(evidence["excerpt"]["text"], first.2.as_str());
    assert_eq!(
        evidence["membership"],
        json!({"a": {"state": "in"}, "b": {"state": "out"}})
    );
    let explained = evidence["explanation"].as_array().unwrap();
    let of_a = explained.iter().find(|item| item["set"] == "a").unwrap();
    assert!(
        (of_a["score"].as_f64().unwrap() as f32 - first.1).abs() < 1e-6,
        "{of_a}"
    );
    let rank = ranked.iter().position(|row| row.0 == first.0).unwrap() + 1;
    assert_eq!(of_a["rank"], rank as u64);
    let position = ArchivePosition::unpack(first.0);
    assert_eq!(
        evidence["record"]["locator"]["entry"],
        entries[position.entry as usize].0.as_str()
    );
    assert_eq!(evidence["record"]["locator"]["line"], position.line);

    // A completed request replayed with its key: the same result, no second run.
    let before = client.activity(&snapshot);
    let (replayed, _) = client.tool("triager_select", select_a);
    assert_eq!(replayed, set_a);
    assert_eq!(client.activity(&snapshot), before);
    client.close();

    // The manifest of the existing score run names the same source: its rows join to
    // the same originals in a scope over the archive, and another archive is refused.
    let manifest =
        ScoreManifest::load(&scratch.path("scored").join("manifest.json"), &budget).unwrap();
    assert_eq!(
        manifest.scorer.revision,
        scored["scorer"]["revision"].as_str().unwrap()
    );
    assert_eq!(manifest.rows, originals);
    let zip = ZipInputSettings {
        entries: vec!["*.log".into()],
        ..ZipInputSettings::default()
    };
    let (scope, _summary) = Scope::open_archive(
        Path::new(&archive),
        &zip,
        &ScopeSettings::new(scratch.path("scope")),
        &mut budget,
    )
    .unwrap();
    let mut joined = Vec::new();
    scope
        .join_keys(
            &manifest.source,
            native_a
                .iter()
                .map(|position| NativeKey::Position(*position)),
            &mut budget,
            |_, original| {
                joined.push(record_id(original.id.0));
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(joined, expect(&native_a));
    let mut changed = entries.clone();
    changed[0].1.push("GET /page/new 200".into());
    let other = scratch.archive("other.zip", &changed);
    let (other_scope, _summary) = Scope::open_archive(
        Path::new(&other),
        &zip,
        &ScopeSettings::new(scratch.path("other-scope")),
        &mut budget,
    )
    .unwrap();
    let refused = other_scope
        .join_keys(
            &manifest.source,
            [NativeKey::Position(native_a[0])],
            &mut budget,
            |_, _| Ok(()),
        )
        .unwrap_err();
    assert!(
        refused.to_string().contains("the source changed"),
        "{refused}"
    );
}

#[test]
fn an_mcp_client_interrupts_a_job_during_spill_and_keeps_what_was_committed() {
    let scratch = fitted("interrupt");
    let entries: Vec<(String, Vec<String>)> = (0..3)
        .map(|entry| {
            let lines = (0..20_000)
                .map(|line| format!("GET /page/{entry}/{line}?login={} 200", line % 13))
                .collect();
            (format!("run/part-{entry}.log"), lines)
        })
        .collect();
    let archive = scratch.archive("long.zip", &entries);
    let mut client = Client::spawn(
        &scratch,
        &["--memory-limit", "512MB", "--batch-rows", "1024"],
        &[
            "--source",
            &format!("long={archive}"),
            "--inline-wait-ms",
            "0",
            "--dedup-set-texts",
            "2000",
            "--work-dir",
            &scratch.arg("work"),
        ],
    );
    let opened = client.ok("triager_open", json!({"v": 1, "source": {"ref": "long"}}));
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    assert_eq!(opened["counts"]["distinct_originals"], 60_000);
    assert_eq!(opened["cost"]["dedup_on_disk"], true);
    let committed = client.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "filter", "syntax": "glob", "query": "*/1999?login=*"}}),
    );
    let handle = committed["set"].as_str().unwrap().to_string();
    assert_eq!(committed["counts"]["in"], 3);

    let (started, is_error) = client.tool(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "idempotency_key": "long-a", "producer": {
            "kind": "score", "model": "web", "top_k": 500, "ties": "record_id_ascending"}}),
    );
    assert!(!is_error, "{started}");
    assert_eq!(started["status"], "running");
    let job = started["job"].as_str().unwrap().to_string();
    let begun = Instant::now();
    loop {
        let status = client.job(&job, "status");
        // Another test's command may hold the host for a moment.
        assert!(
            matches!(status["status"].as_str(), Some("waiting" | "running")),
            "{status}"
        );
        if status["progress"]["spilled"] == true {
            break;
        }
        assert!(begun.elapsed() < Duration::from_secs(120), "{status}");
        std::thread::sleep(Duration::from_millis(1));
    }
    let cancel = client.job(&job, "cancel");
    assert!(
        matches!(
            cancel["status"].as_str().unwrap(),
            "cancel_requested" | "cancelled"
        ),
        "{cancel}"
    );
    let ended = client.wait(&job);
    assert_eq!(ended["status"], "cancelled", "{ended}");
    assert!(ended.get("result").is_none(), "no partial set: {ended}");

    // What was committed before stays usable, and nothing of the job is left.
    let rows = client.records(&handle, json!({}));
    assert_eq!(rows.len(), 3);
    let left: Vec<String> = std::fs::read_dir(scratch.path("work"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        left.iter().all(|name| name.starts_with("snapshot-")),
        "{left:?}"
    );
    client.close();
    assert!(
        std::fs::read_dir(scratch.path("work")).map_or(true, |mut left| left.next().is_none()),
        "the work directory is emptied when the server ends"
    );
}

#[test]
fn the_cli_recipe_and_the_disclosure_profile_use_the_same_operations() {
    let scratch = fitted("recipe");
    let archive = scratch.archive("logs.zip", &web_entries());
    let recipe = [
        json!({"tool": "triager_open", "bind": {"S": "snapshot"},
               "args": {"v": 1, "source": {"ref": "logs", "entries": ["*.log"]}}}),
        json!({"tool": "triager_select", "bind": {"A": "set"},
               "args": {"v": 1, "scope": "$S", "producer": {
                   "kind": "score", "model": "web", "top_k": 25, "ties": "record_id_ascending"}}}),
        json!({"tool": "triager_select", "bind": {"B": "set"},
               "args": {"v": 1, "scope": "$S", "producer": {
                   "kind": "filter", "syntax": "glob", "query": "*login*"}}}),
        json!({"tool": "triager_compose", "bind": {"D": "view"},
               "args": {"v": 1, "mode": "view", "snapshot": "$S",
                        "sets": {"a": "$A", "b": "$B"}, "expr": {"difference": ["a", "b"]}}}),
        json!({"tool": "triager_inspect",
               "args": {"v": 1, "handle": "$D", "view": "records", "limit": 5,
                        "sets": {"a": "$A", "b": "$B"}}}),
    ];
    let text: String = recipe.iter().map(|line| format!("{line}\n")).collect();
    std::fs::write(scratch.path("recipe.jsonl"), text).unwrap();
    let source = format!("logs={archive}");
    let output = scratch.run(&[
        "--memory-limit",
        "512MB",
        "mcp",
        "--source",
        &source,
        "--requests",
        &scratch.arg("recipe.jsonl"),
    ]);
    let diagnostics = support::diagnostics(&output);
    assert_eq!(output.status.code(), Some(0), "{diagnostics}");
    assert_eq!(diagnostics, "");
    let lines: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 5);

    // The MCP server gives the same handles, revisions, counts and rows.
    let mut client = Client::spawn(
        &scratch,
        &["--memory-limit", "512MB"],
        &["--source", &source],
    );
    let opened = client.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "logs", "entries": ["*.log"]}}),
    );
    assert_eq!(opened["snapshot"], lines[0]["result"]["snapshot"]);
    let snapshot = opened["snapshot"].clone();
    let a = client.ok(
        "triager_select",
        recipe[1]["args"].clone().tap_scope(&snapshot),
    );
    let b = client.ok(
        "triager_select",
        recipe[2]["args"].clone().tap_scope(&snapshot),
    );
    for (mine, theirs) in [(&a, &lines[1]["result"]), (&b, &lines[2]["result"])] {
        for field in ["set", "revision", "counts", "occurrence_counts", "producer"] {
            assert_eq!(mine[field], theirs[field], "{field}");
        }
    }
    let view = client.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"a": a["set"], "b": b["set"]}, "expr": {"difference": ["a", "b"]}}),
    );
    assert_eq!(view["view"], lines[3]["result"]["view"]);
    let page = client.ok(
        "triager_inspect",
        json!({"v": 1, "handle": view["view"], "view": "records", "limit": 5,
               "sets": {"a": a["set"], "b": b["set"]}}),
    );
    assert_eq!(page, lines[4]["result"]);
    // No argument widens what the host configured.
    let (refused, is_error) = client.tool(
        "triager_open",
        json!({"v": 1, "source": {"ref": "logs"}, "profile": "public_development"}),
    );
    assert!(is_error);
    assert_eq!(refused["error"]["code"], "invalid_arguments");
    client.close();

    // Under the public-development profile only the declared public fixture opens.
    let fixture = format!("fixture={archive}");
    let mut hosted = Client::spawn(
        &scratch,
        &["--memory-limit", "512MB"],
        &[
            "--disclosure",
            "public-development",
            "--source",
            &source,
            "--public-fixture",
            &fixture,
        ],
    );
    let (refused, is_error) =
        hosted.tool("triager_open", json!({"v": 1, "source": {"ref": "logs"}}));
    assert!(is_error);
    assert_eq!(refused["error"]["code"], "disclosure_refused");
    let opened = hosted.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "fixture"}}),
    );
    assert_eq!(opened["disclosure"]["profile"], "public_development");
    assert_eq!(opened["disclosure"]["source_class"], "public_fixture");
    hosted.close();
}

trait TapScope {
    fn tap_scope(self, snapshot: &Value) -> Value;
}

impl TapScope for Value {
    /// The arguments of a recipe line with the alias of the snapshot replaced.
    fn tap_scope(mut self, snapshot: &Value) -> Value {
        self["scope"] = snapshot.clone();
        self
    }
}

/// Triager task 1.3 through the real server, on the small archive: the reduction
/// with its relation, membership and scores kept on disk (`--tables-on-disk`,
/// deduplication forced to disk, another batch size) is the reduction with them in
/// memory; a registered membership stands in for C with its unknown states; and
/// thirty warm interactions on the prepared sets read no source and score nothing.
/// `tools/triager-scale` runs the same checks on the multi-million fixture.
#[test]
fn an_mcp_client_drills_prepared_sets_on_disk_as_in_memory_without_another_read() {
    let scratch = fitted("drill");
    let entries = web_entries();
    let archive = scratch.archive("logs.zip", &entries);
    // The stand-in for C: of the originals of the first entry, by first occurrence,
    // a quarter each is in, out, abstained and not assessed.
    let mut seen = std::collections::HashSet::new();
    let mut membership = String::new();
    let mut assessed = [0u64; 3];
    for (line, text) in entries[0].1.iter().enumerate() {
        if !seen.insert(text.as_str()) || seen.len() % 4 == 3 {
            continue;
        }
        let outcome = seen.len() % 4;
        assessed[outcome] += 1;
        let state = ["in", "out", "abstain"][outcome];
        membership.push_str(&format!(
            "{}\n",
            json!({"id": record_id(line as u64 + 1), "state": state})
        ));
    }
    std::fs::write(scratch.path("c.jsonl"), membership).unwrap();
    let source = format!("logs={archive}");
    let registered = format!("c={}", scratch.arg("c.jsonl"));

    let mut outcomes = Vec::new();
    for on_disk in [false, true] {
        let mut args = vec![
            "--source",
            source.as_str(),
            "--membership",
            registered.as_str(),
        ];
        let mut global = vec!["--memory-limit", "512MB"];
        if on_disk {
            args.extend(["--tables-on-disk", "--dedup-set-texts", "20"]);
            global.extend(["--batch-rows", "64"]);
        }
        let mut client = Client::spawn(&scratch, &global, &args);
        let opened = client.ok(
            "triager_open",
            json!({"v": 1, "source": {"ref": "logs", "entries": ["*.log"]}}),
        );
        let snapshot = opened["snapshot"].as_str().unwrap().to_string();
        assert_eq!(opened["cost"]["index_on_disk"], on_disk);
        assert_eq!(opened["cost"]["dedup_on_disk"], on_disk);
        assert!(opened["cost"]["bytes_unpacked"].as_u64().unwrap() > 0);
        let a = client.ok(
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {
                "kind": "score", "model": "web", "top_k": 25, "ties": "record_id_ascending"}}),
        );
        let b = client.ok(
            "triager_select",
            json!({"v": 1, "scope": snapshot,
                   "producer": {"kind": "filter", "syntax": "glob", "query": "*login*"}}),
        );
        let c = client.ok(
            "triager_select",
            json!({"v": 1, "scope": snapshot, "producer": {"kind": "membership", "ref": "c"}}),
        );
        assert_eq!(c["counts"]["in"], assessed[0]);
        assert_eq!(c["counts"]["out"], assessed[1]);
        assert_eq!(c["coverage"]["reasons"][0]["reason"], "abstained");
        assert_eq!(
            c["coverage"]["reasons"][0]["distinct_originals"],
            assessed[2]
        );
        assert_eq!(
            c["coverage"]["reasons"][1]["reason"],
            "outside_producer_coverage"
        );
        let sets = json!({"a": a["set"], "b": b["set"], "c": c["set"]});
        let d = client.ok(
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "sets": sets,
                   "expr": {"difference": ["a", "b"]}}),
        );
        for made in [&a, &b, &c, &d] {
            assert_eq!(made["cost"]["tables_on_disk"], on_disk, "{made}");
            assert!(made["cost"]["server_process_peak_bytes"].is_u64() || !cfg!(windows));
        }
        assert_eq!(a["cost"]["dedup_on_disk"], on_disk);
        assert_eq!(a["cost"]["spilled_bytes"].as_u64().unwrap() > 0, on_disk);

        // Warm: the overlap with its unknown combinations, pages, a row, another
        // rule of the prepared scorer. Nothing of it reads the source.
        let before = client.activity(&snapshot);
        let d_view = d["view"].as_str().unwrap().to_string();
        let mut interactions = 0;
        let mut regions = Value::Null;
        let mut rows = Vec::new();
        for round in 0..6 {
            regions = client.ok(
                "triager_inspect",
                json!({"v": 1, "handle": snapshot, "view": "regions", "sets": sets}),
            );
            assert_eq!(regions["regions"].as_array().unwrap().len(), 8);
            assert!(!regions["unknown_regions"].as_array().unwrap().is_empty());
            rows = client.records(&d_view, sets.clone());
            let page = client.ok(
                "triager_inspect",
                json!({"v": 1, "handle": c["set"], "view": "records", "limit": 3 + round}),
            );
            assert_eq!(page["cost"]["archive_scans"], 0);
            let evidence = client.ok(
                "triager_inspect",
                json!({"v": 1, "handle": d_view, "view": "evidence", "record": rows[0]["id"],
                       "sets": sets}),
            );
            assert!(evidence["explanation"][1]["rank"].is_u64(), "{evidence}");
            let derived = client.ok(
                "triager_select",
                json!({"v": 1, "scope": snapshot, "producer": {
                    "kind": "score", "model": "web", "top_k": 10 + round,
                    "ties": "record_id_ascending"}}),
            );
            assert_eq!(derived["cost"]["archive_scans"], 0, "{derived}");
            assert_eq!(derived["cost"]["rows_scored"], 0, "{derived}");
            assert_eq!(derived["cost"]["reused"]["set"], a["set"]);
            interactions += 5;
        }
        assert!(interactions >= 30);
        let after = client.activity(&snapshot);
        for counter in ["archive_scans", "bytes_unpacked", "rows_scored"] {
            assert_eq!(after[counter], before[counter], "{counter}");
        }
        outcomes.push(json!({
            "counts": opened["counts"],
            "revisions": [a["revision"], b["revision"], c["revision"], d["revision"]],
            "counts_of": [a["counts"], b["counts"], c["counts"], d["counts"]],
            "regions": regions["regions"], "unknown_regions": regions["unknown_regions"],
            "rows": rows,
        }));
        client.close();
    }
    assert_eq!(outcomes[1], outcomes[0], "on disk and in memory");
}

/// Triager task 2.1 through the real binary: a development client on an explicit
/// public fixture lists the patterns `l2d discover search` builds, takes one as a
/// set, and plays the model client of the issued-batch protocol with a partial, a
/// duplicate and a foreign submission. The private registration of the same archive
/// cannot be opened, so nothing of it can be issued.
#[test]
fn an_mcp_client_on_a_public_fixture_refines_patterns_and_submits_to_issued_batches() {
    let scratch = Scratch::new("drill");
    let entries = web_entries();
    let archive = scratch.archive("logs.zip", &entries);
    // The existing command path on the same archive, entry filter and query.
    let seed = scratch.arg("seed");
    let searched = scratch.run(&[
        "discover",
        "search",
        "--input",
        &archive,
        "--entry",
        "*.log",
        "--glob",
        "*login*",
        "--name",
        "triager",
        "--max-members",
        "100000",
        "--out",
        &seed,
    ]);
    assert_eq!(
        searched.status.code(),
        Some(0),
        "{}",
        support::diagnostics(&searched)
    );
    let native: Value = serde_json::from_slice(
        &std::fs::read(scratch.path("seed").join("refinement.json")).unwrap(),
    )
    .unwrap();
    let native_groups = native["groups"].as_array().unwrap();
    assert!(native_groups.len() >= 2, "{}", native_groups.len());

    let source = format!("logs={archive}");
    let fixture = format!("fixture={archive}");
    let mut client = Client::spawn(
        &scratch,
        &["--memory-limit", "1GB"],
        &[
            "--disclosure",
            "public-development",
            "--source",
            &source,
            "--public-fixture",
            &fixture,
            "--evidence-allowance-bytes",
            "4096",
        ],
    );
    let listed = client.request("tools/list", json!({}));
    let names: Vec<&str> = listed["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"triager_refine"), "{names:?}");
    let (refused, is_error) =
        client.tool("triager_open", json!({"v": 1, "source": {"ref": "logs"}}));
    assert!(is_error);
    assert_eq!(refused["error"]["code"], "disclosure_refused");
    let opened = client.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "fixture", "entries": ["*.log"]}}),
    );
    let snapshot = opened["snapshot"].clone();
    let b = client.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot,
               "producer": {"kind": "filter", "syntax": "glob", "query": "*login*"}}),
    );
    let members = b["counts"]["in"].as_u64().unwrap();
    assert_eq!(members, native["rows"].as_array().unwrap().len() as u64);

    // The refinement is the one the command saved, revision for revision.
    let p = client.ok(
        "triager_refine",
        json!({"v": 1, "set": b["set"], "action": {"kind": "patterns"}}),
    );
    assert_eq!(p["revision"], native["revision"], "{p}");
    assert_eq!(p["groups"], native_groups.len() as u64);
    let first = native_groups[0]["id"].as_u64().unwrap();
    let selected = client.ok(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "filter", "syntax": "pattern", "refinement": p["refinement"],
            "pattern": first}}),
    );
    let mut expected: Vec<String> = native["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["pattern"] == first)
        .map(|row| record_id(row["member"]["value"].as_u64().unwrap()))
        .collect();
    expected.sort();
    assert_eq!(
        ids(&client.records(selected["set"].as_str().unwrap(), json!({}))),
        expected
    );
    assert_eq!(selected["counts"]["out"], members - expected.len() as u64);

    // The model client: one batch, a duplicate and a foreign submission refused
    // whole, a partial one taken.
    let issued = |producer: Value| {
        let mut producer = producer;
        producer["kind"] = json!("recommend");
        producer["method"] = json!("issued_batch");
        json!({"v": 1, "scope": b["set"], "producer": producer})
    };
    let batch = client.ok(
        "triager_select",
        issued(json!({"action": "issue", "limit": 4, "projection": ["excerpt"]})),
    );
    let records = batch["records"].as_array().unwrap();
    assert_eq!(records.len(), 4);
    assert!(
        records
            .iter()
            .all(|record| record["excerpt"].as_str().unwrap().contains("login"))
    );
    assert_eq!(batch["disclosure"]["source_class"], "public_fixture");
    let disclosed = batch["disclosure"]["excerpt_bytes"].as_u64().unwrap();
    assert_eq!(
        batch["disclosure"]["remaining_evidence_bytes"],
        4096 - disclosed
    );
    let all = ids(&client.records(b["set"].as_str().unwrap(), json!({})));
    let submit = |items: Value| {
        issued(
            json!({"action": "submit", "batch": batch["batch"], "expected_revision": 1,
                      "model": "dev-provider/model@rev", "prompt_revision": "p1",
                      "items": items}),
        )
    };
    let (refused, is_error) = client.tool(
        "triager_select",
        submit(json!([{"id": records[0]["id"], "outcome": "recommend"},
                      {"id": records[0]["id"], "outcome": "pass"}])),
    );
    assert!(is_error);
    assert_eq!(refused["error"]["code"], "duplicate_item");
    let (refused, is_error) = client.tool(
        "triager_select",
        submit(json!([{"id": records[0]["id"], "outcome": "recommend"},
                      {"id": all[5], "outcome": "recommend"}])),
    );
    assert!(is_error);
    assert_eq!(refused["error"]["code"], "foreign_item");
    let c = client.ok(
        "triager_select",
        submit(json!([{"id": records[0]["id"], "outcome": "recommend",
                       "reason": "Repeated sign-in failures"},
                      {"id": records[1]["id"], "outcome": "pass"}])),
    );
    assert_eq!(c["counts"]["in"], 1);
    assert_eq!(c["counts"]["out"], 1);
    assert_eq!(c["counts"]["unknown"], members - 2);
    assert_eq!(c["run_coverage"]["omitted"], 2);
    assert_eq!(c["producer"]["model"]["identity"], "caller_reported");
    assert_eq!(
        ids(&client.records(c["set"].as_str().unwrap(), json!({}))),
        [records[0]["id"].as_str().unwrap()]
    );
    let activity = client.activity(snapshot.as_str().unwrap());
    assert_eq!(activity["genai_rows_issued"], 4);
    assert_eq!(activity["genai_rows_submitted"], 2);
    client.close();
    // A suggestion creates no analyst mark (requirement 3.2): after a session that
    // took a recommendation the project holds no mark set.
    assert!(
        !scratch.root.join("project").join("marks").exists(),
        "a recommendation wrote into the project's marks"
    );
}
