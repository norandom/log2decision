//! The tools as `tools/list` advertises them: name, description and JSON Schema for
//! the arguments and for the structured result.
//!
//! The input schemas state what the service checks itself before it schedules
//! anything; they take no argument for the project or the disclosure profile, which
//! belong to the host. Every result is an object with `v` and `status`; an error
//! result carries `error` with a code, a bounded message and whether it was clipped.

use serde_json::{Value, json};

use super::state::MAX_KEY_BYTES;
use super::wire::{DEFAULT_ROWS, MAX_ROWS};

fn version() -> Value {
    json!({"const": 1, "description": "Version of the interface; every request carries it."})
}

fn key() -> Value {
    json!({
        "type": "string", "minLength": 1, "maxLength": MAX_KEY_BYTES,
        "description": "Idempotency key: the same key and payload return the original \
                        result or the job in progress; another payload conflicts."
    })
}

fn handle(description: &str) -> Value {
    json!({"type": "string", "minLength": 1, "maxLength": 80, "description": description})
}

fn sets(description: &str) -> Value {
    json!({
        "type": "object",
        "description": description,
        "propertyNames": {"pattern": "^[a-z][a-z0-9_]{0,15}$"},
        "additionalProperties": {"type": "string"},
        "maxProperties": 8
    })
}

fn counts(unit: &str) -> Value {
    json!({
        "type": "object",
        "description": "A reduction within the parent population: in + out + unknown = parent.",
        "properties": {
            "unit": {"const": unit},
            "parent": {"type": "integer"},
            "in": {"type": "integer"},
            "out": {"type": "integer"},
            "unknown": {"type": "integer"},
            "retained_share": {"type": ["number", "null"],
                               "description": "in / parent; null for an empty parent."}
        },
        "required": ["unit", "parent", "in", "out", "unknown", "retained_share"]
    })
}

/// The result schema of a tool: `v`, `status`, the tool's own fields, and the error
/// and job forms every tool can end in.
fn output(statuses: &[&str], own: Value) -> Value {
    let mut properties = json!({
        "v": {"const": 1},
        "status": {"enum": statuses},
        "job": {"type": "string",
                "description": "With status running: poll or cancel it with triager_job."},
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string", "maxLength": 1024},
                "clipped": {"type": "boolean"}
            },
            "required": ["code", "message", "clipped"]
        }
    });
    if let (Some(properties), Some(own)) = (properties.as_object_mut(), own.as_object()) {
        for (name, schema) in own {
            properties.insert(name.clone(), schema.clone());
        }
    }
    json!({"type": "object", "properties": properties, "required": ["v", "status"]})
}

fn tool(name: &str, title: &str, description: &str, input: Value, output: Value) -> Value {
    json!({
        "name": name,
        "title": title,
        "description": description,
        "inputSchema": input,
        "outputSchema": output,
    })
}

pub(crate) fn tools() -> Value {
    let job_statuses = ["ready", "running", "error"];
    let open = tool(
        "triager_open",
        "Open a source as a snapshot",
        "Open one local ZIP archive as an immutable snapshot: a source the host registered \
         (`ref`) or an archive below a directory the host allowed (`path`), with optional \
         entry patterns. Returns the snapshot handle, both denominators (distinct originals \
         and occurrences), what was skipped with units and reasons, and the capabilities. \
         A large scan returns a job.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "source"],
            "properties": {
                "v": version(),
                "source": {
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Exactly one of `ref` and `path`.",
                    "properties": {
                        "ref": {"type": "string", "description": "A source the host registered."},
                        "path": {"type": "string",
                                 "description": "Absolute path of an archive below an allowed directory."},
                        "adapter": {"const": "zip"},
                        "entries": {
                            "type": "array", "maxItems": 64,
                            "items": {"type": "string", "minLength": 1, "maxLength": 1024},
                            "description": "Glob patterns over entry paths, at most 1,024 bytes together; without them every entry."
                        },
                        "max_record_bytes": {"type": "integer", "minimum": 1, "maximum": 1_048_576}
                    }
                },
                "idempotency_key": key()
            }
        }),
        output(
            &job_statuses,
            json!({
                "snapshot": {"type": "string"},
                "revision": {"type": "string"},
                "source": {"type": "object"},
                "counts": {
                    "type": "object",
                    "properties": {
                        "distinct_originals": {"type": "integer"},
                        "occurrences": {"type": "integer"}
                    }
                },
                "completeness": {"type": "object"},
                "capabilities": {"type": "object"},
                "disclosure": {"type": "object"},
                "cost": {"type": "object"}
            }),
        ),
    );
    let select = tool(
        "triager_select",
        "Derive a named set",
        "Derive an immutable set inside a snapshot, a selection or a view. `score` takes a \
         frozen scorer with `top_k` and the tie rule `record_id_ascending`, or a \
         `threshold`. `filter` takes a `literal`, a whole-record `glob` or a `regex`. \
         Outside its scope a set is not evaluated. Returns member counts in both units, \
         coverage and the producer revision; reads the source once, as a job.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "scope", "producer"],
            "properties": {
                "v": version(),
                "scope": handle("Snapshot, selection or view the producer is evaluated in."),
                "producer": {
                    "oneOf": [
                        {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["kind", "model"],
                            "properties": {
                                "kind": {"const": "score"},
                                "version": {"const": 1},
                                "model": {"type": "string", "minLength": 1, "maxLength": 128},
                                "top_k": {"type": "integer", "minimum": 1},
                                "threshold": {"type": "number"},
                                "ties": {"const": "record_id_ascending"}
                            }
                        },
                        {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["kind", "syntax", "query"],
                            "properties": {
                                "kind": {"const": "filter"},
                                "version": {"const": 1},
                                "syntax": {"enum": ["literal", "glob", "regex"]},
                                "query": {"type": "string", "minLength": 1},
                                "case_insensitive": {"type": "boolean"}
                            }
                        }
                    ]
                },
                "idempotency_key": key()
            }
        }),
        output(
            &job_statuses,
            json!({
                "set": {"type": "string"},
                "snapshot": {"type": "string"},
                "scope": {"type": "string"},
                "producer": {"type": "object"},
                "revision": {"type": "string"},
                "counts": counts("distinct_originals"),
                "occurrence_counts": counts("occurrences"),
                "coverage": {"type": "object"},
                "outside_scope": {"type": "object"},
                "complete": {"type": "boolean"},
                "evidence": {"type": "object"},
                "cost": {"type": "object"}
            }),
        ),
    );
    let expr = json!({
        "description": "A set name, or one of union and intersection (two or more operands), \
                        difference (exactly two) and complement (one, within the parent); \
                        three-valued where a set was not evaluated.",
        "oneOf": [
            {"type": "string"},
            {"type": "object", "additionalProperties": false, "required": ["union"],
             "properties": {"union": {"type": "array", "minItems": 2}}},
            {"type": "object", "additionalProperties": false, "required": ["intersection"],
             "properties": {"intersection": {"type": "array", "minItems": 2}}},
            {"type": "object", "additionalProperties": false, "required": ["difference"],
             "properties": {"difference": {"type": "array", "minItems": 2, "maxItems": 2}}},
            {"type": "object", "additionalProperties": false, "required": ["complement"],
             "properties": {"complement": {"type": "array", "minItems": 1, "maxItems": 1}}}
        ]
    });
    let compose = tool(
        "triager_compose",
        "Compose a candidate view",
        "Reduce a population by a set expression over named sets of one snapshot. \
         `mode=view` returns an immutable candidate view with its parent and the \
         reduction in both units: parent, in, out, unknown and the retained share. The \
         parent is `scope`, which defaults to the snapshot.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "mode", "snapshot", "sets", "expr"],
            "properties": {
                "v": version(),
                "mode": {"const": "view"},
                "snapshot": handle("The snapshot every set belongs to."),
                "scope": handle("The parent population; defaults to the snapshot."),
                "sets": sets("Names for the selections and views the expression uses."),
                "expr": expr,
                "idempotency_key": key()
            }
        }),
        output(
            &job_statuses,
            json!({
                "view": {"type": "string"},
                "snapshot": {"type": "string"},
                "parent": {"type": "string"},
                "operation": {"type": "object"},
                "revision": {"type": "string"},
                "counts": counts("distinct_originals"),
                "occurrence_counts": counts("occurrences"),
                "coverage": {"type": "object"},
                "complete": {"type": "boolean"},
                "root": {"type": "object"},
                "cost": {"type": "object"}
            }),
        ),
    );
    let inspect = tool(
        "triager_inspect",
        "Inspect counts, regions, records or evidence",
        "Read a snapshot, selection or view. `overview` states what it is and what the \
         server did so far; `regions` counts every combination of up to three named sets \
         in both units, known and not evaluated apart; `records` pages members by record \
         ID with an opaque cursor; `evidence` explains one record and, when `excerpt` is \
         projected, reads its text from the source as a job. Results state the snapshot, \
         revision, units, coverage, the applied limits, clipping and the page.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "handle", "view"],
            "properties": {
                "v": version(),
                "handle": handle("Snapshot, selection or view."),
                "view": {"enum": ["overview", "regions", "records", "evidence"]},
                "sets": sets("Named sets: the regions to count, or the membership columns."),
                "projection": {
                    "type": "array", "uniqueItems": true, "minItems": 1,
                    "items": {"enum": ["id", "locator", "occurrences", "membership",
                                       "explanation", "excerpt"]},
                    "description": "records: id, locator, occurrences, membership. evidence: \
                                    those, explanation and excerpt (not projected by default)."
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS,
                          "default": DEFAULT_ROWS},
                "cursor": {"type": "string", "maxLength": 64,
                           "description": "From `page.next_cursor` of the same request."},
                "sort": {"const": ["record_id_ascending"]},
                "idempotency_key": key(),
                "record": {"type": "string", "pattern": "^r-[0-9a-f]{16}$",
                           "description": "evidence: a record handle from the records view."}
            }
        }),
        output(
            &job_statuses,
            json!({
                "view": {"type": "string"},
                "handle": {"type": "string"},
                "kind": {"enum": ["snapshot", "selection", "view"]},
                "snapshot": {"type": "string"},
                "revision": {"type": "string"},
                "coverage": {"type": "object"},
                "applied_limits": {"type": "object"},
                "clipping": {"type": "object"},
                "page": {"type": ["object", "null"]},
                "rows": {"type": "array", "maxItems": MAX_ROWS},
                "regions": {"type": "array"},
                "unknown_regions": {"type": "array"},
                "record": {"type": "object"},
                "membership": {"type": "object"},
                "explanation": {"type": "array"},
                "excerpt": {"type": "object"},
                "disclosure": {"type": "object"},
                "artifact": {"type": "object"},
                "activity": {"type": "object"},
                "cost": {"type": "object"}
            }),
        ),
    );
    let job = tool(
        "triager_job",
        "Follow or cancel a job",
        "Report the state of a job of this server, or ask it to stop. A completed job \
         carries its committed result; a cancelled, failed or interrupted one carries \
         none, and what was committed before stays usable.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "job", "action"],
            "properties": {
                "v": version(),
                "job": handle("The job a running call returned."),
                "action": {"enum": ["status", "cancel"]}
            }
        }),
        output(
            &[
                "waiting",
                "running",
                "cancel_requested",
                "cancelled",
                "completed",
                "interrupted",
                "failed",
                "error",
            ],
            json!({
                "tool": {"type": "string"},
                "phase": {"type": "string"},
                "progress": {"type": "object"},
                "admitted": {"type": "object"},
                "result": {"type": "object"}
            }),
        ),
    );
    json!([open, select, compose, inspect, job])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_served_tool_is_listed_with_closed_arguments() {
        let tools = tools();
        let listed: Vec<&str> = tools
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(listed, crate::service::TOOLS);
        for tool in tools.as_array().unwrap() {
            let input = &tool["inputSchema"];
            assert_eq!(input["additionalProperties"], false, "{tool}");
            let properties = input["properties"].as_object().unwrap();
            // The project and the disclosure profile belong to the host.
            for name in ["project", "profile", "disclosure"] {
                assert!(!properties.contains_key(name), "{tool}");
            }
        }
    }
}
