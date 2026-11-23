//! The two adapters in front of the shared operations of [`crate::service`]
//! (triager requirement 4.1): a local MCP server over stdio, and a line-oriented
//! reader of tool calls for a noninteractive CLI.
//!
//! # MCP
//!
//! [`serve`] speaks JSON-RPC 2.0 as the MCP stdio transport frames it: one message
//! per line, no embedded newlines, requests on the input and nothing but messages on
//! the output. It is a tools-only server: `initialize`, `ping`, `tools/list` and
//! `tools/call`; notifications are taken without a reply, and every other method is
//! answered as not found. A tool result carries the structured result and the same
//! JSON as its text form; an error of a tool, a refused argument included, is a
//! result with `isError`, so that the client's model can read it. Only a tool that
//! does not exist is a protocol error.
//!
//! A message is read into at most [`MAX_REQUEST_BYTES`] bytes: a longer line is
//! passed over without being kept and answered with an error. No batch arrays.
//!
//! # CLI
//!
//! [`run_requests`] reads one JSON object per line, `{"tool", "args", "bind"}`, calls
//! the same operation, and writes `{"tool", "result"}` per line. `bind` names
//! fields of the result (`{"S": "snapshot"}`); a later argument string `"$S"` is
//! replaced by the bound value, so a recipe can be written before its handles exist.

use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Read, Write};

use l2d_core::L2dError;
use serde_json::{Value, json};

use crate::service::{MAX_RESULT_BYTES, Service};

/// The MCP revision this server speaks.
pub const PROTOCOL_VERSION: &str = "2025-11-25";
/// Earlier revisions whose tools-only stdio exchange is the same.
const ALSO_SPOKEN: [&str; 1] = ["2025-06-18"];
/// The most bytes of one message on the input, its line end not counted.
pub const MAX_REQUEST_BYTES: usize = 64 * 1024;
/// The most bytes of a request ID that is echoed.
const MAX_ID_BYTES: usize = 128;
/// The most JSON values one message may hold, counted before it is parsed. A parsed
/// value takes many times the bytes that write it (an empty array is two bytes on the
/// line and a table of its own in memory), so the length of a line alone does not
/// bound its parsed form. A tool call has a few dozen values; with this limit the
/// parsed form of the densest admitted line is measured in `tests/service_memory.rs`.
pub const MAX_REQUEST_VALUES: usize = 1024;

/// Whether `line` holds more than [`MAX_REQUEST_VALUES`] values, by its structural
/// characters outside strings: every value of well-formed JSON follows an opening
/// bracket, an opening brace or a comma, or is the message itself. Nothing is
/// allocated. A line that is not JSON may be miscounted; it is refused by the parser.
fn too_many_values(line: &[u8]) -> bool {
    let (mut values, mut in_string, mut escaped) = (1usize, false, false);
    for byte in line {
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
        } else {
            match byte {
                b'"' => in_string = true,
                b'[' | b'{' | b',' => values += 1,
                _ => {}
            }
        }
    }
    values > MAX_REQUEST_VALUES
}
/// The most values a recipe binds.
const MAX_BINDINGS: usize = 64;

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

enum Line {
    /// The input has ended.
    End,
    /// A line longer than [`MAX_REQUEST_BYTES`]; it was passed over.
    TooLong,
    /// The line is in the buffer.
    Read,
}

/// Reads one line into `line` (cleared first), without its line end, keeping at most
/// [`MAX_REQUEST_BYTES`] bytes: what a longer line has beyond them is consumed and
/// dropped.
fn read_line(input: &mut impl BufRead, line: &mut Vec<u8>) -> io::Result<Line> {
    line.clear();
    let mut too_long = false;
    let mut any = false;
    loop {
        let available = match input.fill_buf() {
            Ok(available) => available,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if available.is_empty() {
            return Ok(match (any, too_long) {
                (false, _) => Line::End,
                (true, true) => Line::TooLong,
                (true, false) => Line::Read,
            });
        }
        any = true;
        let (part, ended) = match available.iter().position(|byte| *byte == b'\n') {
            Some(at) => (&available[..at], true),
            None => (available, false),
        };
        if !too_long {
            if line.len() + part.len() > MAX_REQUEST_BYTES {
                too_long = true;
                line.clear();
            } else {
                line.extend_from_slice(part);
            }
        }
        let consumed = part.len() + usize::from(ended);
        input.consume(consumed);
        if ended {
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(if too_long { Line::TooLong } else { Line::Read });
        }
    }
}

fn error(id: &Value, code: i64, message: &str) -> String {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}).to_string()
}

fn result(id: &Value, result: &Value) -> String {
    json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string()
}

/// The reply to one message; `None` for a notification.
fn reply(service: &Service, line: &[u8]) -> Option<String> {
    if too_many_values(line) {
        return Some(error(
            &Value::Null,
            INVALID_REQUEST,
            &format!(
                "a message holds at most {MAX_REQUEST_VALUES} JSON values; this one was not \
                 parsed"
            ),
        ));
    }
    let message: Value = match serde_json::from_slice(line) {
        Ok(message) => message,
        Err(_) => return Some(error(&Value::Null, PARSE_ERROR, "the message is not JSON")),
    };
    let Some(fields) = message.as_object() else {
        return Some(error(
            &Value::Null,
            INVALID_REQUEST,
            "a message is one JSON-RPC object; batches are not taken",
        ));
    };
    let id = match fields.get("id") {
        None => None,
        Some(id @ (Value::String(_) | Value::Number(_)))
            if id.to_string().len() <= MAX_ID_BYTES =>
        {
            Some(id)
        }
        Some(_) => {
            return Some(error(
                &Value::Null,
                INVALID_REQUEST,
                "a request ID is a string or a number of at most 128 bytes",
            ));
        }
    };
    let method = match (fields.get("jsonrpc"), fields.get("method")) {
        (Some(Value::String(version)), Some(Value::String(method))) if version == "2.0" => method,
        _ => {
            return Some(error(
                id.unwrap_or(&Value::Null),
                INVALID_REQUEST,
                "a message carries `jsonrpc` \"2.0\" and a `method`",
            ));
        }
    };
    // A notification is never answered, whatever it names.
    let id = id?;
    let params = fields.get("params");
    Some(match method.as_str() {
        "initialize" => {
            let asked = params
                .and_then(|params| params.get("protocolVersion"))
                .and_then(Value::as_str);
            let spoken = match asked {
                Some(asked) if ALSO_SPOKEN.contains(&asked) => asked,
                _ => PROTOCOL_VERSION,
            };
            result(
                id,
                &json!({
                    "protocolVersion": spoken,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {
                        "name": "l2d-triager",
                        "title": "log2decision triager",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "instructions": "Reduce a local log archive by comparing named sets: \
                        open, select A and B, inspect counts and regions, compose a view, \
                        then read bounded pages and one record's evidence. Counts come \
                        first; every result states its denominators and limits.",
                }),
            )
        }
        "ping" => result(id, &json!({})),
        "tools/list" => result(id, &json!({"tools": Service::tools()})),
        "tools/call" => {
            let name = params
                .and_then(|params| params.get("name"))
                .and_then(Value::as_str);
            let Some(name) = name.filter(|name| Service::knows(name)) else {
                let named: String = name.unwrap_or("").chars().take(64).collect();
                return Some(error(id, INVALID_PARAMS, &format!("Unknown tool: {named}")));
            };
            let empty = json!({});
            let arguments = params
                .and_then(|params| params.get("arguments"))
                .unwrap_or(&empty);
            let reply = service.call(name, arguments);
            tool_result(id, &reply.json, reply.is_error)
        }
        _ => error(id, METHOD_NOT_FOUND, "the method is not served"),
    })
}

/// The reply that carries a tool result: the payload as structured content and, for
/// clients that read only text, the same JSON as text. The payload is written as it
/// is, not parsed again.
fn tool_result(id: &Value, payload: &str, is_error: bool) -> String {
    let text = Value::String(payload.to_string()).to_string();
    let result = format!(
        "{{\"content\":[{{\"type\":\"text\",\"text\":{text}}}],\"structuredContent\":{payload},\
         \"isError\":{is_error}}}"
    );
    if result.len() > MAX_RESULT_BYTES {
        // The operations size their results; this is the last line of that limit.
        return error(
            id,
            -32603,
            "the tool result exceeded the response limit and was not sent",
        );
    }
    format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{result}}}")
}

/// Serves MCP on `input` and `output` until the input ends: see the module
/// description. Nothing but JSON-RPC messages is written to `output`.
pub fn serve(service: &Service, input: impl Read, mut output: impl Write) -> io::Result<()> {
    let mut input = BufReader::with_capacity(8 * 1024, input);
    let mut line = Vec::new();
    loop {
        let answer = match read_line(&mut input, &mut line)? {
            Line::End => return Ok(()),
            Line::TooLong => Some(error(
                &Value::Null,
                INVALID_REQUEST,
                &format!(
                    "a message is at most {MAX_REQUEST_BYTES} bytes; this one was passed over"
                ),
            )),
            Line::Read if line.iter().all(u8::is_ascii_whitespace) => None,
            Line::Read => reply(service, &line),
        };
        if let Some(answer) = answer {
            output.write_all(answer.as_bytes())?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
}

fn invalid(detail: String) -> L2dError {
    L2dError::InvalidInput { detail }
}

/// Replaces every argument string `"$NAME"` by the value bound to `NAME`.
fn substitute(value: &mut Value, bound: &BTreeMap<String, Value>) -> Result<(), L2dError> {
    match value {
        Value::String(text) => {
            if let Some(name) = text.strip_prefix('$') {
                *value = bound
                    .get(name)
                    .ok_or_else(|| {
                        invalid(format!("the recipe uses `${name}` before a line binds it"))
                    })?
                    .clone();
            }
        }
        Value::Array(items) => {
            for item in items {
                substitute(item, bound)?;
            }
        }
        Value::Object(fields) => {
            for item in fields.values_mut() {
                substitute(item, bound)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Runs a recipe of tool calls: see the module description. Every call waits for its
/// job when the service was configured to. The first failing call ends the recipe
/// with its error, after its line `{"tool", "error"}` was written; a line that cannot
/// be read ends it before anything is called for it.
pub fn run_requests(
    service: &Service,
    input: impl Read,
    mut output: impl Write,
) -> Result<(), L2dError> {
    let failed = |error: io::Error| L2dError::Runtime {
        detail: format!("the recipe's input or output failed: {error}"),
    };
    let mut input = BufReader::with_capacity(8 * 1024, input);
    let mut line = Vec::new();
    let mut bound: BTreeMap<String, Value> = BTreeMap::new();
    let mut number = 0u64;
    loop {
        number += 1;
        match read_line(&mut input, &mut line).map_err(failed)? {
            Line::End => return Ok(()),
            Line::TooLong => {
                return Err(invalid(format!(
                    "line {number} of the recipe is longer than {MAX_REQUEST_BYTES} bytes"
                )));
            }
            Line::Read if line.iter().all(u8::is_ascii_whitespace) => continue,
            Line::Read => {}
        }
        if too_many_values(&line) {
            return Err(invalid(format!(
                "line {number} of the recipe holds more than {MAX_REQUEST_VALUES} JSON values"
            )));
        }
        let mut call: Value = serde_json::from_slice(&line).map_err(|error| {
            invalid(format!("line {number} of the recipe is not JSON: {error}"))
        })?;
        let known = call.as_object().is_some_and(|fields| {
            fields
                .keys()
                .all(|field| matches!(field.as_str(), "tool" | "args" | "bind"))
        });
        let tool = call
            .get("tool")
            .and_then(Value::as_str)
            .filter(|tool| known && Service::knows(tool))
            .map(str::to_string)
            .ok_or_else(|| {
                invalid(format!(
                    "line {number} of the recipe is `{{\"tool\", \"args\", \"bind\"}}` with a \
                     triager tool"
                ))
            })?;
        let binds: Vec<(String, String)> = match call.get("bind") {
            None => Vec::new(),
            Some(Value::Object(binds)) => binds
                .iter()
                .map(|(name, field)| match field.as_str() {
                    Some(field) if !name.is_empty() && name.len() <= 32 => {
                        Ok((name.clone(), field.to_string()))
                    }
                    _ => Err(invalid(format!(
                        "`bind` on line {number} maps names of at most 32 bytes to result fields"
                    ))),
                })
                .collect::<Result<_, _>>()?,
            Some(_) => {
                return Err(invalid(format!(
                    "`bind` on line {number} maps names to result fields"
                )));
            }
        };
        let mut args = call
            .get_mut("args")
            .map(Value::take)
            .unwrap_or_else(|| json!({}));
        substitute(&mut args, &bound)
            .map_err(|error| invalid(format!("line {number}: {error}")))?;
        let reply = service.call(&tool, &args);
        if reply.is_error {
            writeln!(output, "{{\"tool\":\"{tool}\",\"error\":{}}}", reply.json).map_err(failed)?;
            let payload: Value = serde_json::from_str(&reply.json).unwrap_or_default();
            let message = payload["error"]["message"]
                .as_str()
                .unwrap_or("")
                .to_string();
            return Err(match payload["error"]["code"].as_str() {
                Some("invalid_arguments" | "unsupported_version" | "unsupported_operation") => {
                    invalid(format!("line {number}, `{tool}`: {message}"))
                }
                Some("runtime_failure") => L2dError::Runtime {
                    detail: format!("line {number}, `{tool}`: {message}"),
                },
                code => L2dError::Refused {
                    reason: format!(
                        "line {number}, `{tool}`: {} ({})",
                        message,
                        code.unwrap_or("error")
                    ),
                },
            });
        }
        if !binds.is_empty() {
            let payload: Value = serde_json::from_str(&reply.json).unwrap_or_default();
            for (name, field) in binds {
                let value = payload
                    .get(&field)
                    // A handle or another short scalar: what a later line can name.
                    .filter(|value| match value {
                        Value::String(text) => text.len() <= 256,
                        Value::Number(_) | Value::Bool(_) => true,
                        _ => false,
                    })
                    .ok_or_else(|| {
                        invalid(format!(
                            "line {number} binds `{name}` to the field `{field}`, which the \
                             result of `{tool}` does not have as a string of at most 256 \
                             bytes, a number or a truth value"
                        ))
                    })?;
                if bound.len() >= MAX_BINDINGS && !bound.contains_key(&name) {
                    return Err(invalid(format!(
                        "the recipe binds more than {MAX_BINDINGS} names"
                    )));
                }
                bound.insert(name, value.clone());
            }
        }
        writeln!(output, "{{\"tool\":\"{tool}\",\"result\":{}}}", reply.json).map_err(failed)?;
        output.flush().map_err(failed)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &[u8]) -> Vec<Option<Vec<u8>>> {
        let mut input = BufReader::with_capacity(16, text);
        let mut line = Vec::new();
        let mut found = Vec::new();
        loop {
            match read_line(&mut input, &mut line).unwrap() {
                Line::End => return found,
                Line::TooLong => found.push(None),
                Line::Read => found.push(Some(line.clone())),
            }
        }
    }

    #[test]
    fn lines_are_read_whole_and_long_ones_are_passed_over() {
        assert_eq!(
            lines(b"one\r\n\ntwo two two two two two\nlast"),
            [
                Some(b"one".to_vec()),
                Some(Vec::new()),
                Some(b"two two two two two two".to_vec()),
                Some(b"last".to_vec())
            ]
        );
        let mut long = vec![b'x'; MAX_REQUEST_BYTES + 1];
        long.extend_from_slice(b"\nnext\n");
        assert_eq!(lines(&long), [None, Some(b"next".to_vec())]);
        let mut fits = vec![b'x'; MAX_REQUEST_BYTES];
        fits.push(b'\n');
        assert_eq!(lines(&fits).len(), 1);
        assert!(lines(&fits)[0].is_some());
        assert!(lines(b"").is_empty());
    }

    #[test]
    fn values_are_counted_outside_strings_before_a_line_is_parsed() {
        let list = |count: usize| format!("[{}]", vec!["[]"; count].join(","));
        // An array of n empty arrays holds n + 1 values and is counted as 2n + 1: the
        // count is an upper bound of the values.
        assert!(!too_many_values(
            list(MAX_REQUEST_VALUES / 2 - 1).as_bytes()
        ));
        assert!(too_many_values(list(MAX_REQUEST_VALUES / 2).as_bytes()));
        // Brackets and commas inside strings are text, also after an escaped quote.
        let text = format!(
            r#"{{"a":"{}\"{}"}}"#,
            "[,".repeat(2_000),
            "{,".repeat(2_000)
        );
        assert!(serde_json::from_str::<Value>(&text).is_ok());
        assert!(!too_many_values(text.as_bytes()));
        let members = format!(
            "{{{}}}",
            (0..MAX_REQUEST_VALUES - 2)
                .map(|index| format!(r#""k{index}":0"#))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(!too_many_values(members.as_bytes()));
        let one_more = format!(r#"[{members},0]"#);
        assert!(too_many_values(one_more.as_bytes()));
    }

    #[test]
    fn a_tool_result_is_one_line_with_both_forms() {
        let payload = json!({"v": 1, "status": "ready", "text": "a \"b\"\n\\"}).to_string();
        let written = tool_result(&json!("id-1"), &payload, false);
        assert!(!written.contains('\n'));
        let reply: Value = serde_json::from_str(&written).unwrap();
        assert_eq!(reply["id"], "id-1");
        assert_eq!(reply["result"]["isError"], false);
        assert_eq!(reply["result"]["structuredContent"]["text"], "a \"b\"\n\\");
        assert_eq!(reply["result"]["content"][0]["text"], payload.as_str());
        let result = serde_json::to_string(&reply["result"]).unwrap();
        assert!(result.len() <= crate::service::transport_bytes(&payload));
    }

    #[test]
    fn a_recipe_substitutes_only_bound_names() {
        let mut bound = BTreeMap::new();
        bound.insert("S".to_string(), json!("snap-1"));
        let mut args = json!({"scope": "$S", "sets": {"a": "$S"}, "list": ["$S", "S", 1]});
        substitute(&mut args, &bound).unwrap();
        assert_eq!(
            args,
            json!({"scope": "snap-1", "sets": {"a": "snap-1"}, "list": ["snap-1", "S", 1]})
        );
        let mut unbound = json!({"scope": "$T"});
        assert!(substitute(&mut unbound, &bound).is_err());
    }
}
