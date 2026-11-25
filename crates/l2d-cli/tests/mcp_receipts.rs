//! Triager task 2.2 through the built binary (requirements 3.3, 4.1 and 6.3): a
//! candidate saved by one `l2d mcp` process is reopened by another process with its
//! own work directory, drilled into both C branches, allocated once and saved; a
//! third process reopens the final work set. The same calls run as a recipe file
//! (`--requests`) save the same receipt byte for byte, and `--replay` runs a
//! receipt's recipe to the same content with that receipt as its parent.

mod support;

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-cli-receipts-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("project")).unwrap();
        let mut labelled = String::from("text,label\n");
        for index in 0..40 {
            labelled.push_str(&format!("GET /page/{index} 200,good\n"));
            labelled.push_str(&format!("GET /page/{index}?id=1%27%20OR%201=1-- 500,bad\n"));
        }
        std::fs::write(root.join("labelled.csv"), labelled).unwrap();
        let scratch = Self { root };
        let output = scratch
            .command()
            .args([
                "baseline",
                "fit",
                "--input",
                &scratch.arg("labelled.csv"),
                "--name",
                "web",
            ])
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(0),
            "{}",
            support::diagnostics(&output)
        );
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

    fn receipts(&self) -> PathBuf {
        self.root.join("project").join("receipts")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
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
    fn spawn(scratch: &Scratch, args: &[&str]) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let stderr = scratch.path(&format!(
            "mcp-stderr-{}.txt",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut child = scratch
            .command()
            .args(["--memory-limit", "512MB", "mcp"])
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(std::fs::File::create(&stderr).unwrap())
            .spawn()
            .unwrap();
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
                   "clientInfo": {"name": "receipt tests", "version": "0"}}),
        );
        assert_eq!(initialized["result"]["protocolVersion"], "2025-11-25");
        client
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next += 1;
        let id = self.next;
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
        )
        .unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line).unwrap();
        assert!(
            read > 0,
            "the server closed its output: {}",
            std::fs::read_to_string(&self.stderr).unwrap_or_default()
        );
        let reply: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(reply["id"], id, "{reply}");
        reply
    }

    fn tool(&mut self, tool: &str, arguments: Value) -> (Value, bool) {
        let reply = self.request("tools/call", json!({"name": tool, "arguments": arguments}));
        let result = &reply["result"];
        (
            result["structuredContent"].clone(),
            result["isError"] == true,
        )
    }

    fn ok(&mut self, tool: &str, arguments: Value) -> Value {
        let (value, is_error) = self.tool(tool, arguments.clone());
        assert!(!is_error, "{tool} {arguments}: {value}");
        if value["status"] == "running" {
            let job = value["job"].as_str().unwrap().to_string();
            let started = Instant::now();
            loop {
                let (status, _) = self.tool(
                    "triager_job",
                    json!({"v": 1, "job": job, "action": "status"}),
                );
                match status["status"].as_str().unwrap() {
                    "waiting" | "running" | "cancel_requested" => {}
                    "completed" => return status["result"].clone(),
                    _ => panic!("{status}"),
                }
                assert!(started.elapsed() < Duration::from_secs(300));
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        value
    }

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

    fn close(mut self) {
        drop(self.stdin.take());
        let status = self.child.wait().unwrap();
        let diagnostics: String = std::fs::read_to_string(&self.stderr)
            .unwrap_or_default()
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
    }
    vec![
        ("logs/access.log".into(), access),
        ("logs/auth.log".into(), auth),
    ]
}

fn write_archive(path: &Path, entries: &[(String, Vec<String>)]) {
    let mut writer = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
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
}

/// The calls of the recipe up to the save of D with C beside it, as an MCP client
/// makes them and as a recipe file writes them.
fn calls() -> Vec<Value> {
    vec![
        json!({"tool": "triager_open", "bind": {"S": "snapshot"},
               "args": {"v": 1, "source": {"ref": "logs", "entries": ["*.log"]}}}),
        json!({"tool": "triager_select", "bind": {"A": "set"},
               "args": {"v": 1, "scope": "$S", "producer": {"kind": "score", "model": "web",
                        "top_k": 30, "ties": "record_id_ascending"}}}),
        json!({"tool": "triager_select", "bind": {"B": "set"},
               "args": {"v": 1, "scope": "$S",
                        "producer": {"kind": "filter", "syntax": "glob", "query": "*login*"}}}),
        json!({"tool": "triager_select", "bind": {"C": "set"},
               "args": {"v": 1, "scope": "$S", "producer": {"kind": "membership", "ref": "c"}}}),
        json!({"tool": "triager_compose", "bind": {"D": "view"},
               "args": {"v": 1, "mode": "view", "snapshot": "$S", "sets": {"a": "$A", "b": "$B"},
                        "expr": {"difference": ["a", "b"]}}}),
        json!({"tool": "triager_save",
               "args": {"v": 1, "handle": "$D", "sets": {"c": "$C"}}}),
    ]
}

/// Substitutes `$NAME` as `l2d mcp --requests` does.
fn bound(value: &Value, names: &std::collections::BTreeMap<String, Value>) -> Value {
    match value {
        Value::String(text) if text.starts_with('$') => names[&text[1..]].clone(),
        Value::Array(items) => Value::Array(items.iter().map(|item| bound(item, names)).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, item)| (key.clone(), bound(item, names)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn budget() -> Value {
    json!({"total": 6, "duplicate_rule": "first_pool_wins", "shortfall": "leave_unfilled",
           "pools": [
               {"name": "general", "limit": 3, "from": {"intersection": ["d", "c"]},
                "order": ["score_desc", "record_id_ascending"]},
               {"name": "unfamiliar", "limit": 2,
                "from": {"intersection": ["d", {"state": {"set": "c", "is": "not_evaluated"}}]},
                "order": ["record_id_ascending"]},
               {"name": "audit", "limit": 1, "from": {"difference": ["d", "c"]},
                "order": ["seeded_hash", "record_id_ascending"], "seed": 5}]})
}

#[test]
fn a_receipt_passes_between_server_processes_and_the_cli_saves_the_same_one() {
    let scratch = Scratch::new("handoff");
    let archive = scratch.path("logs.zip");
    write_archive(&archive, &web_entries());
    // C: of the first entry's originals, by first occurrence, a quarter each in, out,
    // abstained and not assessed.
    let mut seen = std::collections::HashSet::new();
    let mut membership = String::new();
    for (line, text) in web_entries()[0].1.iter().enumerate() {
        if !seen.insert(text.clone()) || seen.len() % 4 == 3 {
            continue;
        }
        let state = ["in", "out", "abstain"][seen.len() % 4];
        membership.push_str(&format!(
            "{}\n",
            json!({"id": format!("r-{:016x}", line + 1), "state": state})
        ));
    }
    std::fs::write(scratch.path("c.jsonl"), membership).unwrap();
    let source = format!("logs={}", archive.to_string_lossy());
    let registered = format!("c={}", scratch.arg("c.jsonl"));
    let host = [
        "--source",
        source.as_str(),
        "--membership",
        registered.as_str(),
    ];

    // The first process: the hunt up to D, saved with C beside it.
    let mut first = Client::spawn(&scratch, &host);
    let mut names = std::collections::BTreeMap::new();
    let mut saved = Value::Null;
    for call in calls() {
        let args = bound(&call["args"], &names);
        let result = first.ok(call["tool"].as_str().unwrap(), args);
        if let Some(binds) = call["bind"].as_object() {
            for (name, field) in binds {
                names.insert(name.clone(), result[field.as_str().unwrap()].clone());
            }
        }
        saved = result;
    }
    let receipt = saved["receipt"].as_str().unwrap().to_string();
    let named = json!({"d": names["D"], "c": names["C"]});
    let here = first.records(names["D"].as_str().unwrap(), json!({"c": names["C"]}));
    let allocated_here = first.ok(
        "triager_compose",
        json!({"v": 1, "mode": "allocate", "snapshot": names["S"], "sets": named,
               "expr": {"intersection": ["d", "c"]}, "budget": budget()}),
    );
    let final_here = first.records(allocated_here["set"].as_str().unwrap(), json!({}));
    first.close();
    let bytes = std::fs::read(scratch.receipts().join(format!("{receipt}.receipt"))).unwrap();

    // The second process, with a work directory of its own: reopen, both C branches,
    // allocate once, save the final work set.
    let mut second = Client::spawn(&scratch, &host);
    let opened = second.ok("triager_open", json!({"v": 1, "receipt": receipt}));
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let reopened = |saved_as: &Value| -> String {
        opened["sets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|set| set["saved_as"] == *saved_as)
            .unwrap()["handle"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let (d, c) = (reopened(&names["D"]), reopened(&names["C"]));
    assert_eq!(second.records(&d, json!({"c": c})), here);
    for is in [
        json!("c"),
        json!({"state": {"set": "c", "is": "not_evaluated"}}),
    ] {
        let branch = second.ok(
            "triager_compose",
            json!({"v": 1, "mode": "view", "snapshot": snapshot, "scope": d,
                   "sets": {"d": d, "c": c}, "expr": {"intersection": ["d", is]}}),
        );
        assert_eq!(branch["parent"], d);
    }
    let allocated = second.ok(
        "triager_compose",
        json!({"v": 1, "mode": "allocate", "snapshot": snapshot, "sets": {"d": d, "c": c},
               "expr": {"intersection": ["d", "c"]}, "budget": budget()}),
    );
    let final_there = second.records(allocated["set"].as_str().unwrap(), json!({}));
    assert_eq!(final_there, final_here);
    assert_eq!(allocated["allocation"], allocated_here["allocation"]);
    let overview = second.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "overview"}),
    );
    for counter in [
        "archive_scans",
        "bytes_unpacked",
        "rows_scored",
        "scorer_loads",
        "membership_reads",
    ] {
        assert_eq!(overview["activity"][counter], 0, "{counter}");
    }
    let final_receipt = second.ok("triager_save", json!({"v": 1, "handle": allocated["set"]}));
    second.close();

    // The third process reopens the final work set: the same ordered slots.
    let mut third = Client::spawn(&scratch, &host);
    let opened = third.ok(
        "triager_open",
        json!({"v": 1, "receipt": final_receipt["receipt"]}),
    );
    assert_eq!(opened["kind"], "final");
    assert_eq!(
        third.records(opened["set"].as_str().unwrap(), json!({})),
        final_here
    );
    third.close();

    // The same calls as a recipe file save the same receipt, byte for byte.
    std::fs::remove_file(scratch.receipts().join(format!("{receipt}.receipt"))).unwrap();
    let recipe: String = calls().iter().map(|call| format!("{call}\n")).collect();
    std::fs::write(scratch.path("recipe.jsonl"), recipe).unwrap();
    let output = scratch
        .command()
        .args(["--memory-limit", "512MB", "mcp"])
        .args(host)
        .args(["--requests", &scratch.arg("recipe.jsonl")])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        support::diagnostics(&output)
    );
    let lines: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.last().unwrap()["result"]["receipt"], receipt);
    assert_eq!(
        std::fs::read(scratch.receipts().join(format!("{receipt}.receipt"))).unwrap(),
        bytes,
        "CLI and MCP write the same receipt"
    );

    // `--replay` runs the receipt's recipe: the same content, this receipt its parent.
    let output = scratch
        .command()
        .args(["--memory-limit", "512MB", "mcp"])
        .args(host)
        .args(["--replay", &receipt])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        support::diagnostics(&output)
    );
    let last: Value = serde_json::from_str(
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(last["tool"], "triager_save");
    assert_eq!(last["result"]["content_sha256"], saved["content_sha256"]);
    assert_eq!(last["result"]["lineage"]["parent_receipt"], receipt);
}
