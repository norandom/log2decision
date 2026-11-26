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
         A large scan returns a job. With `receipt` instead of `source`, reopens a saved \
         candidate view or final work set without its source: a retained snapshot with \
         every set the receipt retained, their lineage, coverage, ordering and reasons; \
         nothing is read from the archive, scored or inferred. With `projections`, the same \
         job also reads the declared timestamp and field values of every occurrence (one \
         more read of the source); with `snapshot` and `projections` instead of `source`, it \
         is an explicit re-scan of an open snapshot that keeps every earlier declaration.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v"],
            "properties": {
                "v": version(),
                "snapshot": {"type": "string",
                             "description": "An explicit re-scan: the open snapshot to project; exclusive with `source` and `receipt`."},
                "projections": {
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Occurrence projections: fields from `kv` (the record's key=value, keyed by the name) or `entry` (with `capture`, a regex over the entry path), and a timestamp from `text` (the leading one), `kv` (with `key`) or `entry` (with `capture`), read as `iso8601` with Z or an offset or as a `date` at 00:00 UTC. `source` is built in.",
                    "properties": {
                        "fields": {
                            "type": "array", "maxItems": 8,
                            "items": {
                                "type": "object", "additionalProperties": false,
                                "required": ["name", "from"],
                                "properties": {
                                    "name": {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,64}$"},
                                    "from": {"enum": ["kv", "entry"]},
                                    "capture": {"type": "string", "minLength": 1, "maxLength": 256}
                                }
                            }
                        },
                        "time": {
                            "type": "object", "additionalProperties": false,
                            "required": ["from"],
                            "properties": {
                                "from": {"enum": ["text", "kv", "entry"]},
                                "key": {"type": "string"},
                                "capture": {"type": "string", "minLength": 1, "maxLength": 256},
                                "format": {"enum": ["iso8601", "date"], "default": "iso8601"},
                                "timezone": {"const": "UTC"}
                            }
                        },
                        "max_values": {"type": "integer", "minimum": 1, "maximum": 100_000,
                                       "default": 10_000,
                                       "description": "Values a field retains in read order; the rest are counted as overflow."}
                    }
                },
                "receipt": {"type": "string", "pattern": "^rcpt-[0-9a-f]{32}$",
                            "description": "A receipt `triager_save` returned; exclusive with `source`."},
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
                "receipt": {"type": "string"},
                "kind": {"enum": ["candidate", "final"]},
                "set": {"type": "string", "description": "receipt: the reopened saved set."},
                "sets": {"type": "array", "description": "receipt: every reopened set with the handle it was saved as."},
                "retained": {"type": "object"},
                "origin": {"type": "object"},
                "lineage": {"type": "object"},
                "recipe": {"type": "object"},
                "projection": {"type": ["object", "string"],
                               "description": "projections: the projection with its revision, declarations and table; a re-scan returns its handle here."},
                "declared": {"type": "object"},
                "previous": {"type": ["string", "null"]},
                "fields": {"type": "object"},
                "time": {"type": ["object", "null"]},
                "cost": {"type": "object"}
            }),
        ),
    );
    let pattern_producer = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["kind", "syntax", "refinement", "pattern"],
        "description": "The exact members of one pattern of a refinement revision: in for its members, out for the other members of the refined selection, not evaluated elsewhere.",
        "properties": {
            "kind": {"const": "filter"},
            "version": {"const": 1},
            "syntax": {"const": "pattern"},
            "refinement": {"type": "string"},
            "pattern": {"type": "integer", "minimum": 0}
        }
    });
    let venn_producer = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["kind", "method", "policy", "membership"],
        "description": "The routes of a saved Venn policy report the host registered. A route is a suggestion, never a review.",
        "properties": {
            "kind": {"const": "recommend"},
            "version": {"const": 1},
            "method": {"const": "venn"},
            "policy": {"type": "string"},
            "membership": {"enum": ["investigate_or_review", "investigate"]}
        }
    });
    let issued_producer = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["kind", "method", "action"],
        "description": "Issued batches for a model client. `issue` returns the next batch of the run over `scope` (a selection or view); `submit` takes judgments for the records of one batch under the revision the client saw and returns the set collected so far; `collect` returns the set of everything submitted for the run. A foreign or repeated ID, an unknown outcome, another model or prompt, or a stale revision refuses the whole submission.",
        "properties": {
            "kind": {"const": "recommend"},
            "version": {"const": 1},
            "method": {"const": "issued_batch"},
            "action": {"enum": ["issue", "submit", "collect"]},
            "run": {"type": "string", "pattern": "^[A-Za-z0-9._-]{1,64}$",
                    "description": "issue, collect: the run within the scope; `default` when absent."},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS,
                      "default": DEFAULT_ROWS},
            "projection": {"type": "array", "uniqueItems": true, "minItems": 1,
                           "items": {"enum": ["id", "excerpt"]}},
            "batch": {"type": "string"},
            "expected_revision": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "minLength": 1, "maxLength": 128,
                      "description": "Recorded as caller-reported."},
            "prompt_revision": {"type": "string", "minLength": 1, "maxLength": 64},
            "items": {
                "type": "array", "minItems": 1, "maxItems": MAX_ROWS,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "outcome"],
                    "properties": {
                        "id": {"type": "string", "pattern": "^r-[0-9a-f]{16}$"},
                        "outcome": {"enum": ["recommend", "pass", "abstain"]},
                        "reason": {"type": "string", "maxLength": 200}
                    }
                }
            },
            "membership": {"const": "recommend"},
            "submissions_ref": {"type": "string",
                                "description": "collect: the submissions the client expects the run to hold."}
        }
    });
    let select = tool(
        "triager_select",
        "Derive a named set",
        "Derive an immutable set inside a snapshot, a selection or a view. `score` takes a \
         frozen scorer with `top_k` and the tie rule `record_id_ascending`, or a \
         `threshold`. `filter` takes a `literal`, a whole-record `glob` or a `regex`. \
         `membership` takes a membership file the host registered: assessed in, assessed \
         out, abstained, and not assessed for every record it does not name. `filter` \
         with the syntax `pattern` takes one pattern of a refinement. `recommend` takes a \
         saved Venn policy report (`venn`) or the issued-batch protocol for a model client \
         (`issued_batch`); a recommendation is a suggestion and never a review. Outside its \
         scope a set is not evaluated. Returns member counts in both units, coverage, the \
         producer revision and the cost. A job: it reads the source once, unless a \
         prepared set of the same scorer revision, filter or membership supplies it.",
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
                        },
                        {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["kind", "ref"],
                            "properties": {
                                "kind": {"const": "membership"},
                                "version": {"const": 1},
                                "ref": {"type": "string", "description": "A membership file the host registered."}
                            }
                        },
                        pattern_producer,
                        venn_producer,
                        issued_producer
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
                "policy": {"type": "object"},
                "batch": {"type": "string"},
                "run": {"type": "object"},
                "records": {"type": "array", "maxItems": MAX_ROWS},
                "issued": {"type": "object"},
                "run_coverage": {"type": "object"},
                "disclosure": {"type": "object"},
                "cost": {"type": "object"}
            }),
        ),
    );
    let refine = tool(
        "triager_refine",
        "Drill into patterns, neighbours, field values and time buckets",
        "Group the members of a filter selection into patterns and refine them. `patterns` \
         (with `set`) builds the project's pattern catalogue for a literal, glob or regex \
         selection over a whole snapshot and returns a refinement revision with its groups. \
         The other actions take `refinement`: `list` pages the patterns, with templates when \
         projected; `split` (by a regex) and `merge` return a new revision with its parent \
         and the changed counts; `preview` counts what a proposed or given regex matches \
         among the members of a pattern and among the other members; `neighbours` returns \
         the selection of the members nearest to an exemplar, within its refinement or, \
         with `target`, also within the refinement of a wider selection. A pattern becomes a \
         set through `triager_select` with the filter syntax `pattern`. `field` and `time` \
         take `set` (a snapshot, selection or view) and make a child view of its members \
         with an occurrence that has a field value (`value`, or `missing: true`; the field \
         `source` is the archive entry) or falls in a time bucket (`start` and `end` as the \
         histogram states them, or `no_time: true`); they read the snapshot's occurrence \
         projection, never the source, and count matching occurrences apart from all \
         occurrences of the members (`matching_occurrences`).",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "action"],
            "properties": {
                "v": version(),
                "set": handle("patterns: the filter selection to group; field and time: the snapshot, selection or view to drill into."),
                "refinement": handle("Every other action: a refinement revision."),
                "action": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["kind"],
                    "properties": {
                        "kind": {"enum": ["patterns", "list", "split", "merge", "preview",
                                          "neighbours", "field", "time"]},
                        "field": {"type": "string", "minLength": 1, "maxLength": 64,
                                  "description": "field: a projected field, or `source`."},
                        "value": {"type": "string", "maxLength": 1024,
                                  "description": "field: a value the fields view listed; at most 256 bytes for a declared field (`value_too_long`), 1,024 for `source`."},
                        "missing": {"type": "boolean",
                                    "description": "field: the occurrences without a value."},
                        "start": {"type": "string",
                                  "description": "time: ISO 8601 with Z or an offset, inclusive."},
                        "end": {"type": "string",
                                "description": "time: ISO 8601 with Z or an offset, exclusive."},
                        "no_time": {"type": "boolean",
                                    "description": "time: the occurrences without a valid timestamp."},
                        "after": {"type": "integer", "minimum": 0,
                                  "description": "list: from `page.next_after`."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS,
                                  "default": DEFAULT_ROWS},
                        "projection": {"const": ["template"],
                                       "description": "list: show templates; they count as evidence."},
                        "pattern": {"type": "integer", "minimum": 0},
                        "patterns": {"type": "array", "minItems": 2, "maxItems": 256,
                                     "items": {"type": "integer", "minimum": 0}},
                        "regex": {"type": "string", "minLength": 1},
                        "exemplar": {"type": "string", "pattern": "^r-[0-9a-f]{16}$"},
                        "method": {"enum": ["character_trigram", "vector"]},
                        "within": {"enum": ["pattern", "other_patterns", "all"], "default": "all"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 1000},
                        "radius": {"type": "number", "minimum": 0, "maximum": 2},
                        "target": {"type": "string",
                                   "description": "neighbours: the refinement of a wider selection to search as well."}
                    }
                },
                "idempotency_key": key()
            }
        }),
        output(
            &job_statuses,
            json!({
                "refinement": {"type": "string"},
                "set": {"type": "string"},
                "snapshot": {"type": "string"},
                "revision": {"type": "string"},
                "parent": {"type": ["string", "null"]},
                "action": {"type": "string"},
                "groups": {"type": "integer"},
                "members": {"type": "object"},
                "ungrouped": {"type": "object"},
                "method": {"type": "object"},
                "patterns": {"type": "array", "maxItems": MAX_ROWS},
                "page": {"type": "object"},
                "changed": {"type": ["object", "null"]},
                "preview": {"type": "object"},
                "neighbours": {"type": "object"},
                "producer": {"type": "object"},
                "view": {"type": "string", "description": "field and time: the child view."},
                "operation": {"type": "object"},
                "matching_occurrences": {
                    "type": "object",
                    "description": "field and time: the parent's counted occurrences that pass \
                                    the drill (parent, in, out, retained_share)."
                },
                "occurrence_filter": {"type": "array"},
                "complete": {"type": "boolean"},
                "root": {"type": "object"},
                "counts": counts("distinct_originals"),
                "occurrence_counts": counts("occurrences"),
                "coverage": {"type": "object"},
                "clipping": {"type": "object"},
                "disclosure": {"type": "object"},
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
             "properties": {"complement": {"type": "array", "minItems": 1, "maxItems": 1}}},
            {"type": "object", "additionalProperties": false, "required": ["state"],
             "description": "In where the persisted state of the named set is `is`, out elsewhere: selects what a producer did not evaluate.",
             "properties": {"state": {
                 "type": "object", "additionalProperties": false, "required": ["set", "is"],
                 "properties": {"set": {"type": "string"},
                                "is": {"enum": ["in", "out", "not_evaluated"]}}}}}
        ]
    });
    let budget = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["total", "duplicate_rule", "shortfall", "pools"],
        "description": "allocate: one review budget over named pools, filled in their order. \
                        An original takes one slot, in the first pool that reaches it; a pool \
                        whose population runs short leaves its slots unused. The limits add up \
                        to `total`.",
        "properties": {
            "total": {"type": "integer", "minimum": 1, "maximum": 10_000},
            "duplicate_rule": {"const": "first_pool_wins"},
            "shortfall": {"const": "leave_unfilled"},
            "pools": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["name", "limit", "from", "order"],
                    "properties": {
                        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$"},
                        "limit": {"type": "integer", "minimum": 1},
                        "from": {"description": "The pool's population: an expression over `sets`; it may reach beyond the candidate expression."},
                        "order": {"type": "array", "minItems": 1, "maxItems": 3,
                                  "description": "Keys `score_desc` (or {\"score_desc\": NAME}), `seeded_hash` (the policy's FNV-1a key with `seed`) and `record_id_ascending`, which ends every order."},
                        "seed": {"type": "integer", "minimum": 0}
                    }
                }
            }
        }
    });
    let compose = tool(
        "triager_compose",
        "Compose a candidate view or allocate once",
        "Reduce a population by a set expression over named sets of one snapshot. \
         `mode=view` returns an immutable candidate view with its parent and the \
         reduction in both units: parent, in, out, unknown and the retained share. The \
         parent is `scope`, which defaults to the snapshot. `mode=allocate` spends one \
         review budget once: `expr` is the candidate expression, `budget` names the pools; \
         it returns a final work set with the pool, slot and reason of every member, the \
         deferred counts and the audit drawn outside the candidate expression.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "mode", "snapshot", "sets", "expr"],
            "properties": {
                "v": version(),
                "mode": {"enum": ["view", "allocate"]},
                "snapshot": handle("The snapshot every set belongs to."),
                "scope": handle("The parent population; defaults to the snapshot."),
                "sets": sets("Names for the selections and views the expression uses."),
                "expr": expr,
                "budget": budget,
                "idempotency_key": key()
            }
        }),
        output(
            &job_statuses,
            json!({
                "view": {"type": "string"},
                "set": {"type": "string", "description": "allocate: the final work set."},
                "kind": {"const": "final"},
                "snapshot": {"type": "string"},
                "parent": {"type": "string"},
                "operation": {"type": "object"},
                "revision": {"type": "string"},
                "counts": counts("distinct_originals"),
                "occurrence_counts": counts("occurrences"),
                "coverage": {"type": "object"},
                "complete": {"type": "boolean"},
                "root": {"type": "object"},
                "capacity": {"type": "integer"},
                "selected": {"type": "integer"},
                "allocation": {"type": "object",
                               "description": "allocate: the rule, every pool with its limit, selected, unused, eligible and deferred counts, the candidate and the audit outside it."},
                "cost": {"type": "object"}
            }),
        ),
    );
    let save = tool(
        "triager_save",
        "Save a receipt",
        "Save a candidate view or a final work set as a durable receipt in the host's \
         receipt directory, atomically: the source identity, the set and every set it was \
         made from with their producers, revisions, counts and coverage, every retained \
         original's record handle, locator, occurrences, states and slot, the allocation \
         and the recipe that makes it again. No source text and no host path. The \
         receipt is named by its content; `triager_open` with `receipt` reopens it in \
         this or another server.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "handle"],
            "properties": {
                "v": version(),
                "handle": handle("A candidate view or a final work set."),
                "sets": sets("Sets to retain beside it over its originals, such as a C to drill into after reopening."),
                "parent_receipt": {"type": "string", "pattern": "^rcpt-[0-9a-f]{32}$",
                                   "description": "The receipt whose recipe this set replays: recorded as lineage."},
                "idempotency_key": key()
            }
        }),
        output(
            &job_statuses,
            json!({
                "receipt": {"type": "string"},
                "kind": {"enum": ["candidate", "final"]},
                "snapshot": {"type": "string"},
                "set": {"type": "string"},
                "content_sha256": {"type": "string"},
                "rows_sha256": {"type": "string"},
                "retained": {"type": "object"},
                "recipe": {"type": "object"},
                "lineage": {"type": "object"},
                "contains": {"type": "array"},
                "excludes": {"type": "array"},
                "reopen": {"type": "object"},
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
         projected, reads its text from the source as a job, and lists every occurrence of \
         the record once the snapshot is projected. `fields` counts the projected fields of \
         the population (typed counts, missing, invalid and overflowing values), or with \
         `field` its values a page at a time; `histogram` draws the timestamp in at most 100 \
         buckets with its field, zone, range and width and the occurrences without a time, or \
         returns `time_unavailable` with the occurrences by archive entry as a labelled \
         alternative. Both count occurrences and their distinct originals, run as jobs over \
         the occurrence projection and read no source; a field or timestamp no projection \
         declares returns `projection_missing` with the scan that would provide it and the \
         request that runs it. Results state the snapshot, revision, units, coverage, the \
         applied limits, clipping and the page.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["v", "handle", "view"],
            "properties": {
                "v": version(),
                "handle": handle("Snapshot, selection or view."),
                "view": {"enum": ["overview", "regions", "records", "evidence", "fields",
                                  "histogram"]},
                "sets": sets("Named sets: the regions to count, or the membership columns."),
                "field": {"type": "string", "minLength": 1, "maxLength": 64,
                          "description": "fields: the field whose values are paged, or `source`."},
                "bucket_seconds": {"enum": [1, 60, 300, 900, 3600, 21600, 86400, 604800,
                                            2592000, 31536000],
                                   "description": "histogram: the bucket width; without it the smallest that draws the range in 100 buckets."},
                "projection": {
                    "type": "array", "uniqueItems": true, "minItems": 1,
                    "items": {"enum": ["id", "locator", "occurrences", "membership",
                                       "allocation", "explanation", "excerpt"]},
                    "description": "records: id, locator, occurrences, membership, and of a \
                                    final work set allocation. evidence: id, locator, \
                                    occurrences, membership, explanation and excerpt (not \
                                    projected by default)."
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS,
                          "default": DEFAULT_ROWS},
                "cursor": {"type": "string", "maxLength": 64,
                           "description": "From `page.next_cursor` of the same request."},
                "sort": {"enum": [["record_id_ascending"], ["allocation"]],
                         "description": "A final work set is paged by `allocation` (its slots in order) unless record IDs are asked for."},
                "idempotency_key": key(),
                "record": {"type": "string", "pattern": "^r-[0-9a-f]{16}$",
                           "description": "evidence: a record handle from the records view."}
            }
        }),
        output(
            &[
                "ready",
                "running",
                "error",
                "time_unavailable",
                "projection_missing",
            ],
            json!({
                "view": {"type": "string"},
                "handle": {"type": "string"},
                "kind": {"enum": ["snapshot", "selection", "view", "final"]},
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
                "excerpt_estimate": {
                    "type": "object",
                    "description": "evidence without an excerpt: what reading one takes at most."
                },
                "disclosure": {"type": "object"},
                "artifact": {"type": "object"},
                "dependencies_total": {"type": "integer",
                                       "description": "overview of a Venn selection: how many score sets it depends on; at most eight are listed, and clipping says when fewer are."},
                "dependencies": {"type": "array",
                                 "description": "regions, evidence and the overview of a Venn selection: the score sets made by the scorer revision the policy scored with."},
                "activity": {
                    "type": "object",
                    "description": "overview: the work the server did so far. Reads of \
                                    prepared sets leave archive_scans, bytes_unpacked and \
                                    rows_scored unchanged."
                },
                "session": {
                    "type": "object",
                    "description": "overview: calls answered, bytes returned, calls refused \
                                    as not served."
                },
                "tables": {
                    "type": "object",
                    "description": "overview of a set: whether its membership and scores \
                                    are kept on disk, and what they take."
                },
                "occurrences": {"type": "object",
                                "description": "overview of a field or time view: its matching occurrences and its occurrence filter."},
                "projection": {"type": "object"},
                "population": {"type": "object",
                               "description": "fields and histogram: the members and the occurrences counted."},
                "fields": {"type": "array"},
                "field": {"type": "object"},
                "time": {"type": "object"},
                "alternative": {"type": "object",
                                "description": "time_unavailable: the occurrences by archive entry, labelled as not time."},
                "missing": {"type": "object"},
                "projections": {"type": ["object", "null"]},
                "estimate": {"type": "object",
                             "description": "projection_missing: the scan that would provide it; nothing is scheduled."},
                "rescan": {"type": "object"},
                "unit": {"type": "string"},
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
    json!([open, select, refine, compose, inspect, save, job])
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
