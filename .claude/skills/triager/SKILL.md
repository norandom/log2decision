---
name: triager
description: Reduce a large local log archive to a reviewable work set through the `l2d mcp` triager tools. Use when an agent or analyst asks which records a search missed, what two signals agree on, what a recommendation could not assess, or wants to allocate a review budget, save a hunt and hand it to another client. Counts first, bounded evidence second, one allocation last.
allowed-tools: Read, Bash, Grep, Glob
argument-hint: <question about a registered source>
---

# triager

## Overview

The triager service answers set questions over one immutable snapshot of a local ZIP
archive: A (a frozen score), B (a filter), C (a recommendation or a membership file)
and their regions. Every call returns handles, counts in two units (distinct
originals and occurrences) and the cost of the work; pages and excerpts are bounded.
The archive, the score stream and full result tables never enter the conversation.

The sequence is always **count → compare regions → reduce → inspect evidence →
allocate → save**. Stop when the analyst's question is answered or the review budget
is spent; record what was excluded or not assessed at each step.

Contract and limits: `.kiro/specs/triager/interface.md`. Worked example with sizes:
`.kiro/specs/triager/recipe.md`.

## When to Use

- "Which high-scoring records did my search miss?" (A − B)
- "What did C recommend inside that difference, and what could it not assess?"
- "Give me 24 records to review today: recommendations first, some unfamiliar ones,
  and a seeded audit of what C passed."
- "Save this hunt so a colleague can reopen it" or "replay last week's hunt on the
  new archive".

Do not use it to read whole files, to train or fit models, or to make a review
decision: a recommendation or an allocation is a suggestion and never an analyst mark.

## Starting the server

The host, not the agent, decides sources, profile and directories:

```
l2d --memory-limit 512MB mcp --source NAME=ARCHIVE.zip [--public-fixture NAME=ZIP] \
    [--membership C=FILE.jsonl] [--policy V=REPORT.json] [--receipts DIR] \
    [--disclosure public-development]
```

- A membership file is JSON lines, one `{"id": "r-<16 hex>", "state": "in"|"out"|"abstain"}`
  per assessed record, at most 128 bytes a line. A record it does not name is not
  assessed. The tab-separated form is refused.
- Receipts are saved in `--receipts` (default `receipts/` below the project); it is
  separate from the work directory, which is emptied when the server ends.
- `--requests FILE` runs a recipe of tool calls (`{"tool","args","bind"}` per line,
  `"$NAME"` replaced by a bound result field); `--replay RECEIPT` runs the recipe a
  receipt retains.

## Procedure

Every request carries `"v": 1`. A call that reads the source returns
`{"status": "running", "job": ...}` when it outlasts the inline wait: poll it with
`triager_job` (`status` or `cancel`). A failed or cancelled job changes nothing that
was committed.

1. **Open.** `triager_open {"source": {"ref": NAME, "entries": ["*.log"]}}`. Note the
   denominators (`counts.distinct_originals`, `counts.occurrences`) and what was
   skipped (`completeness`). Read `capabilities` and `disclosure` before asking for
   text.
2. **A and B.** `triager_select` with `{"kind": "score", "model": M, "top_k": K,
   "ties": "record_id_ascending"}` and `{"kind": "filter", "syntax": "glob", "query":
   "*login*"}`. Report in, out and not evaluated, and the `cost` (archive scans, rows
   scored, reuse). A second rule of the same scorer revision is derived without a read.
3. **Regions first.** `triager_inspect {"view": "regions", "handle": S, "sets":
   {"a": A, "b": B}}`: the eight fully evaluated regions and, apart from them, the
   combinations with a not-evaluated set. Say which region answers the question before
   opening any row.
4. **Reduce.** `triager_compose {"mode": "view", "snapshot": S, "sets": {"a": A, "b":
   B}, "expr": {"difference": ["a", "b"]}}` gives D with its parent and the retained
   share in both units. Use `scope` to make D the parent of the next step. Operators:
   `union`, `intersection`, `difference`, `complement` and the `state` selector
   `{"state": {"set": "c", "is": "not_evaluated"}}`, which selects unknowns on purpose.
   Three-valued: `in − not_evaluated` stays unknown.
5. **C where it pays, on a bounded scope.** A saved Venn report (`{"kind":
   "recommend", "method": "venn", "policy": V, "membership":
   "investigate_or_review"}`), a registered membership file, or issued batches for a
   model client (`issue`, `submit`, `collect`) with `scope` set to D. Outside its scope
   C is not evaluated, never negative. Venn's dependence on A's scorer is stated in
   `dependencies`.
6. **Patterns before rows.** `triager_refine {"set": B, "action": {"kind":
   "patterns"}}` groups a filter selection; `split`, `merge`, `preview` and
   `neighbours` refine it.
7. **Field and time slices from a projection.** Declare what to project when opening
   (`triager_open {"source": ..., "projections": {"fields": [{"name": "user", "from":
   "kv"}, {"name": "host", "from": "entry", "capture": "^gather/([^/]+)/"}], "time":
   {"from": "text"}}}`) or later as an explicit re-scan (`triager_open {"snapshot": S,
   "projections": {...}}`); either is one more admitted read of the source. Then
   `triager_inspect {"view": "fields", "handle": H}` (with `"field": F` for its values,
   paged) and `{"view": "histogram", "handle": H}` count occurrences and their distinct
   originals of any snapshot, selection or view, with missing, invalid and overflowing
   values apart and `source` (the archive entry) built in. A timestamp needs `Z` or an
   offset. A projection without a declared timestamp gives `projection_missing` for
   the histogram; a declared one with no valid value gives `time_unavailable`, which
   offers the occurrences by entry as a labelled alternative that is not time.
   `triager_refine
   {"set": H, "action": {"kind": "field", "field": F, "value": V}}` or `{"kind": "time",
   "start": ..., "end": ...}` makes a child view: report its `counts` (originals) and
   its `matching_occurrences` (the occurrences that pass) apart. A field nobody
   projected returns `projection_missing` with the estimate and the exact `rescan`
   request: show the cost before running it. A value beyond the retained `max_values`
   is `value_not_retained`, one longer than 256 bytes `value_too_long`; with overflow
   a page of values ranks among retained values only (`ranking_exact: false`). None of
   these views reads the source; `evidence` under a field or time view lists every
   occurrence of the record with `in_view`.
8. **Inspect evidence sparingly.** `records` pages at most 100 rows (default 25) by an
   opaque cursor; `evidence` explains one record: its memberships, the producer's
   reason, a score and rank, a Venn route. An excerpt (`"projection": [..,
   "excerpt"]`) is one admitted re-read of the source and counts against the evidence
   allowance under the public-development profile. Never infer a total from a page.
9. **Siblings.** Compare `D ∩ C` (recommended) with `D ∩ state(c, not_evaluated)`
   (unknown) under `scope: D`; both are children of D and keep their own handles.
10. **Allocate once.** `triager_compose {"mode": "allocate", "snapshot": S, "sets":
   {"d": D, "c": C}, "expr": {"intersection": ["d", "c"]}, "budget": {"total": 24,
   "duplicate_rule": "first_pool_wins", "shortfall": "leave_unfilled", "pools": [
   {"name": "general", "limit": 18, "from": {"intersection": ["d", "c"]}, "order":
   ["score_desc", "record_id_ascending"]},
   {"name": "unfamiliar", "limit": 4, "from": {"intersection": ["d", {"state": {"set":
   "c", "is": "not_evaluated"}}]}, "order": ["record_id_ascending"]},
   {"name": "audit", "limit": 2, "from": {"difference": ["d", "c"]}, "order":
   ["seeded_hash", "record_id_ascending"], "seed": 5}]}}`.
   Limits add up to `total`; every order ends with `record_id_ascending`;
   `seeded_hash` is the parent policy's FNV-1a audit key and needs `seed`. The result
   states every pool's eligible, selected, unused and deferred counts, and the members
   drawn outside the candidate expression. `records` of the final set page its slots
   in order with their pool; `evidence` of a member says why its slot was filled, and
   evidence under D with `"sets": {"f": F}` says why another original was deferred.
11. **Save and hand off.** `triager_save {"handle": D or F, "sets": {"c": C}}` writes a
    receipt atomically, named by its content. Pass in `sets` what the next client must
    drill into that is not part of the saved set's lineage (such as C). The receipt
    states what it `contains` and `excludes`: no source text, and no host path except
    one case: under the private-local profile, a source the client opened by its
    `path` is retained with that path (its `contains` says so); a registered `ref` is
    retained by name only. The receipt keeps the producers' evidence: scores, Venn
    routes and reasons, distances, pattern membership and the model judgments of
    issued batches (outcome, reason, caller-reported model and prompt revision). Another
    client reopens it with `triager_open {"receipt": R}`: a retained snapshot with every
    saved set; regions, records, evidence, compose, allocate and save work; anything
    that needs the source (select, refine, excerpts) is refused with
    `source_not_retained`. Reopening reads nothing of the archive and scores nothing:
    check `activity` in the overview.
12. **Replay.** `l2d mcp --replay R` (or the calls in the reopen result's
    `recipe.calls`, with `parent_receipt: R` on the save) runs the recipe against the
    registered source. The same input gives the same `content_sha256`; a changed source
    gives a new revision whose lineage names R.

## Reporting

At each step report the parent, in, out and not-evaluated counts in both units, the
retained share with its denominator, and the cost. Name the refusal code and figures
when a call is refused (`refused`, `disclosure_refused`, `source_not_retained`,
`projection_missing`, `value_not_retained`, `unsupported_operation`), and do not
retry an `unsupported_operation`. Keep the distinction between a suggested priority
(recommendation, allocation) and a reviewed outcome.

## Limits worth remembering

- 64 KiB and 1,024 JSON values per request; 32 KiB per tool result; 100 rows a page;
  1 KiB an excerpt.
- At most 8 named sets per request, 3 for regions; 64 expression nodes, depth 8.
- A budget has at most 10,000 slots and 8 pools; a receipt at most 16 sets and
  1,000,000 retained originals.
- A projection declares at most 8 fields and 4 expressions over entry paths; a field
  retains 10,000 values by default and at most 100,000 (`max_values`); a value of more
  than 256 bytes is invalid; a histogram has at most 100 buckets.
- `local_model` recommendations, the refine action `region` and lineage views are
  refused by name in this revision. A field or time view saved in a receipt reopens
  with its members, but its occurrences are not retained: its fields and histogram are
  `source_not_retained`, and its recipe step is not replayable.
