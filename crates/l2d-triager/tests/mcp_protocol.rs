//! Triager task 1.2: the MCP stdio adapter and the line-oriented CLI adapter over the
//! shared operations (requirements 4.1, 4.2 and 4.3). The adapter is driven here with
//! in-memory streams; `l2d-cli`'s `tests/mcp.rs` drives the built binary.

mod support;

use l2d_triager::mcp::{self, MAX_REQUEST_BYTES, PROTOCOL_VERSION};
use l2d_triager::service::MAX_RESULT_BYTES;
use serde_json::{Value, json};
use support::{Host, HostOptions, auth_entries, oracle};

/// Sends the lines and returns the messages the server wrote, one per line.
fn exchange(host: &Host, lines: &[String]) -> Vec<Value> {
    let mut input = Vec::new();
    for line in lines {
        input.extend_from_slice(line.as_bytes());
        input.push(b'\n');
    }
    let mut output = Vec::new();
    mcp::serve(&host.service, &input[..], &mut output).unwrap();
    let text = String::from_utf8(output).unwrap();
    assert!(text.is_empty() || text.ends_with('\n'));
    text.lines()
        .map(|line| {
            assert!(!line.is_empty());
            serde_json::from_str(line).unwrap_or_else(|error| panic!("{error}: {line}"))
        })
        .collect()
}

fn request(id: u64, method: &str, params: Value) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string()
}

fn call(id: u64, tool: &str, arguments: Value) -> String {
    request(
        id,
        "tools/call",
        json!({"name": tool, "arguments": arguments}),
    )
}

#[test]
fn the_server_initializes_lists_typed_tools_and_answers_only_requests() {
    let host = Host::new(&auth_entries(), HostOptions::default());
    let replies = exchange(
        &host,
        &[
            request(
                1,
                "initialize",
                json!({"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                       "clientInfo": {"name": "test", "version": "0"}}),
            ),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}).to_string(),
            request(2, "ping", json!({})),
            request(3, "tools/list", json!({})),
            json!({"jsonrpc": "2.0", "method": "notifications/cancelled",
                   "params": {"requestId": 2}})
            .to_string(),
            request(4, "resources/list", json!({})),
            json!({"jsonrpc": "2.0", "id": "five", "method": "initialize",
                   "params": {"protocolVersion": "1999-01-01", "capabilities": {},
                              "clientInfo": {"name": "old", "version": "0"}}})
            .to_string(),
        ],
    );
    assert_eq!(replies.len(), 5, "notifications get no reply: {replies:?}");
    let initialized = &replies[0];
    assert_eq!(initialized["jsonrpc"], "2.0");
    assert_eq!(initialized["id"], 1);
    assert_eq!(initialized["result"]["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(
        initialized["result"]["capabilities"],
        json!({"tools": {"listChanged": false}})
    );
    assert_eq!(initialized["result"]["serverInfo"]["name"], "l2d-triager");
    assert_eq!(replies[1], json!({"jsonrpc": "2.0", "id": 2, "result": {}}));

    let tools = replies[2]["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "triager_open",
            "triager_select",
            "triager_compose",
            "triager_inspect",
            "triager_job"
        ]
    );
    for tool in tools {
        for schema in ["inputSchema", "outputSchema"] {
            assert_eq!(tool[schema]["type"], "object", "{tool}");
            assert!(tool[schema]["properties"]["v"].is_object(), "{tool}");
            assert!(
                tool[schema]["required"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("v")),
                "{tool}"
            );
        }
        assert_eq!(tool["inputSchema"]["additionalProperties"], false);
        assert_eq!(tool["inputSchema"]["properties"]["v"]["const"], 1);
        assert!(tool["description"].as_str().unwrap().len() > 20);
    }
    assert!(replies[2]["result"].get("nextCursor").is_none());

    assert_eq!(replies[3]["id"], 4);
    assert_eq!(replies[3]["error"]["code"], -32601);
    // An unknown protocol version is answered with the one the server speaks.
    assert_eq!(replies[4]["id"], "five");
    assert_eq!(replies[4]["result"]["protocolVersion"], PROTOCOL_VERSION);
}

#[test]
fn malformed_and_oversized_messages_are_answered_without_being_kept() {
    let host = Host::new(&auth_entries(), HostOptions::default());
    let oversized = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\",\"params\":{{\"pad\":\"{}\"}}}}",
        "x".repeat(MAX_REQUEST_BYTES)
    );
    let replies = exchange(
        &host,
        &[
            "{not json".to_string(),
            oversized,
            json!([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]).to_string(),
            json!({"jsonrpc": "1.0", "id": 2, "method": "ping"}).to_string(),
            json!({"jsonrpc": "2.0", "id": 3}).to_string(),
            request(
                4,
                "tools/call",
                json!({"name": "triager_delete", "arguments": {}}),
            ),
            request(5, "tools/call", json!({"arguments": {}})),
            request(6, "ping", json!({})),
        ],
    );
    let codes: Vec<Value> = replies
        .iter()
        .map(|reply| reply["error"]["code"].clone())
        .collect();
    assert_eq!(
        codes,
        [
            json!(-32700),
            json!(-32600),
            json!(-32600),
            json!(-32600),
            json!(-32600),
            json!(-32602),
            json!(-32602),
            Value::Null
        ]
    );
    assert_eq!(replies[0]["id"], Value::Null);
    assert!(
        replies[1]["error"]["message"]
            .as_str()
            .unwrap()
            .contains(&MAX_REQUEST_BYTES.to_string())
    );
    assert!(
        replies[5]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("triager_delete")
    );
    // The server goes on after each of them.
    assert_eq!(replies[7], json!({"jsonrpc": "2.0", "id": 6, "result": {}}));
}

#[test]
fn a_tool_result_is_structured_has_its_text_form_and_stays_within_the_byte_limit() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let host = Host::new(&entries, HostOptions::default());
    let replies = exchange(
        &host,
        &[call(
            1,
            "triager_open",
            json!({"v": 1, "source": {"ref": "private", "entries": ["*.log"]}}),
        )],
    );
    let result = &replies[0]["result"];
    assert_eq!(result["isError"], false);
    let structured = &result["structuredContent"];
    assert_eq!(structured["status"], "ready");
    assert_eq!(
        structured["counts"]["distinct_originals"],
        universe.len() as u64
    );
    let content = result["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    let text: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(&text, structured);
    let snapshot = structured["snapshot"].as_str().unwrap();

    // The largest page: the serialized result, text form included, fits the limit.
    let lines = [
        call(
            2,
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "records", "limit": 100}),
        ),
        // A tool error is a result the model can read, not a protocol error.
        call(
            3,
            "triager_inspect",
            json!({"v": 1, "handle": "view-none", "view": "records"}),
        ),
        call(
            4,
            "triager_inspect",
            json!({"handle": snapshot, "view": "records"}),
        ),
    ];
    let mut input = Vec::new();
    for line in &lines {
        input.extend_from_slice(line.as_bytes());
        input.push(b'\n');
    }
    let mut output = Vec::new();
    mcp::serve(&host.service, &input[..], &mut output).unwrap();
    let text = String::from_utf8(output).unwrap();
    let raw: Vec<&str> = text.lines().collect();
    assert_eq!(raw.len(), 3);
    for line in &raw {
        let reply: Value = serde_json::from_str(line).unwrap();
        let result = serde_json::to_string(&reply["result"]).unwrap();
        assert!(result.len() <= MAX_RESULT_BYTES, "{} bytes", result.len());
    }
    let page: Value = serde_json::from_str(raw[0]).unwrap();
    assert!(
        !page["result"]["structuredContent"]["rows"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    for (line, code) in [(raw[1], "unknown_handle"), (raw[2], "invalid_arguments")] {
        let reply: Value = serde_json::from_str(line).unwrap();
        assert!(reply.get("error").is_none(), "{reply}");
        assert_eq!(reply["result"]["isError"], true);
        assert_eq!(reply["result"]["structuredContent"]["error"]["code"], code);
    }
}

#[test]
fn the_cli_adapter_runs_a_recipe_of_tool_calls_over_the_same_operations() {
    let entries = auth_entries();
    let host = Host::new(&entries, HostOptions::default());
    let recipe = [
        json!({"tool": "triager_open", "bind": {"S": "snapshot"},
               "args": {"v": 1, "source": {"ref": "private", "entries": ["*.log"]}}}),
        json!({"tool": "triager_select", "bind": {"A": "set"},
               "args": {"v": 1, "scope": "$S", "producer": {
                   "kind": "score", "model": "sum", "top_k": 40, "ties": "record_id_ascending"}}}),
        json!({"tool": "triager_select", "bind": {"B": "set"},
               "args": {"v": 1, "scope": "$S", "producer": {
                   "kind": "filter", "syntax": "glob", "query": "*login*"}}}),
        json!({"tool": "triager_compose", "bind": {"D": "view"},
               "args": {"v": 1, "mode": "view", "snapshot": "$S",
                        "sets": {"a": "$A", "b": "$B"}, "expr": {"difference": ["a", "b"]}}}),
        json!({"tool": "triager_inspect",
               "args": {"v": 1, "handle": "$D", "view": "records", "limit": 3}}),
    ];
    let input: String = recipe.iter().map(|line| format!("{line}\n")).collect();
    let mut output = Vec::new();
    mcp::run_requests(&host.service, input.as_bytes(), &mut output).unwrap();
    let lines: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 5);
    assert_eq!(lines[3]["tool"], "triager_compose");

    // The MCP adapter gives the same handles, counts and rows for the same calls.
    let snapshot = host.open("private");
    let a = host.select_top(snapshot["snapshot"].as_str().unwrap(), 40);
    let b = host.select_filter(snapshot["snapshot"].as_str().unwrap(), "glob", "*login*");
    assert_eq!(lines[0]["result"]["snapshot"], snapshot["snapshot"]);
    assert_eq!(lines[1]["result"]["set"], a["set"]);
    assert_eq!(lines[1]["result"]["revision"], a["revision"]);
    assert_eq!(lines[2]["result"]["counts"], b["counts"]);
    let replies = exchange(
        &host,
        &[call(
            1,
            "triager_inspect",
            json!({"v": 1, "handle": lines[3]["result"]["view"], "view": "records", "limit": 3}),
        )],
    );
    assert_eq!(
        replies[0]["result"]["structuredContent"],
        lines[4]["result"]
    );

    // A failing call ends the recipe with its error on the last line.
    let failing = format!(
        "{}\n{}\n",
        json!({"tool": "triager_inspect", "args": {"v": 1, "handle": "$Z", "view": "records"}}),
        json!({"tool": "triager_inspect", "args": {"v": 1, "handle": "x", "view": "records"}}),
    );
    let mut output = Vec::new();
    let error = mcp::run_requests(&host.service, failing.as_bytes(), &mut output).unwrap_err();
    assert!(error.to_string().contains("$Z"), "{error}");
    assert!(output.is_empty());
}
