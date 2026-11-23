# triager implementation plan

The owner approved starting the native terminal on 18 November using the MCP walkthrough as its behavioral contract. The parent project's archive readers, scoring and Venn policy already supply evidence. The terminal starts with retained reports. Direct archive-scale operation and interoperability with MCP need the service milestones below. [Requirements](requirements.md) · [Worked example](recipe.md)

Use the [shared end-to-end cases](../../../docs/qualification/triager-e2e.md) to compare the [design walkthrough](analyst-flow.md), native terminal and parent CLI. A discrepancy in population, coverage or order reopens the corresponding case in both specs. Task 1.2 adds a real MCP stdio client to these cases. Task 2.2 extends handoff to independent MCP and terminal clients.

- [x] 3.2a Deliver the native retained-evidence workflow. From `l2d tui`, drill A − B, inspect C and occurrence evidence, revisit a sibling, resize at 80×24, hand off an unallocated candidate and reopen the ordered final work set. Browse the full retained text in any populated view and return to the same result or evidence detail. Validate the parent CLI's saved policy format at the same entry point. See the [cases and evidence](../../../docs/qualification/triager-e2e.md).

- [x] 3.2b Bring the native search mode close to Splunk SPL.
  - Owner decision of 18 November: use Splunk token semantics for terms, and support `search` plus the commands that filter, extract, order or truncate originals. Commands that build tables or new fields stay out until a statistics result type exists.
  - Implemented in `crates/l2d-triager/src/flow/query.rs` and `query/{field,search,eval}.rs`. Unit tests cover segmentation, `TERM()` and `CASE()`, fields and automatic extraction, `where` functions, `regex`, `rex` and `fields`, the ordering commands, occurrence masks, refusals and limits. A flow test covers an ordered view, receipt replay and set B refusals. Through `l2d tui --demo --query`, the README example keeps 3 originals and 4 occurrences, and the order of a piped preview follows the pipeline.
  - Independent review, round 1: three blocking findings. `substr` could overflow; `replace` growth was not estimated; the pipeline budget check ignored what the caller held. All three are fixed, along with owned sort and dedup keys, leading-star scans, quoted backslashes and IP ordering in automatic sort. Round 2: approved. Its one non-blocking note, host length in the text estimate, is fixed. A thread-local counting-allocator test compares measured peaks with the admitted estimate. `cargo test -p l2d-triager` passes 44 of 44, and Clippy (all targets, `-D warnings`) and fmt are clean. Both ConPTY drivers pass on SHA256 `E72B2310…5D85`: `native-pty-fin2p7vi` and `policy-pty-_np94b92`. [Language reference](../../../docs/2026-11-18-triager-search-language.md)
  - _Requirements: 5.1, 5.4, 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 1.1 Establish shared source and membership identity.
  - Completed 2026-11-23 in `crates/l2d-triager/src/scope.rs`, `scope/index.rs` and `scope/join.rs`. A file, selected ZIP entries or a store opens as one scope whose handle is a SHA-256 over source identity, revision, entry filter, reader interpretation, policy path and the record relation. Selection, policy and score-position keys resolve to one scoped original; each incompatibility has its own refusal. Handle, counts and joins are equal across 24 combinations of batch size, reader count, dedup placement and index placement. Independent review: round 1 REJECTED because the reader's summary was dropped from the budget while the index was admitted (reproduced: 49.2 MB of growth under a limit leaving 46.1 MB); round 2 APPROVED. Workspace: 1,017 passed, 0 failed, 56 ignored; Clippy with warnings denied and fmt clean. Evidence: `D:\BuildArtifacts\log2decision\triager-1.1\`.
  - Recorded deviations: a scope keeps the first locator and the exact occurrence count per original, and per-occurrence positions need the admitted pass of task 3.1. The relation lives in a work directory for the life of the scope, with no manifest and no reopen from another process; that belongs to 1.3 and 2.2. The refusal when a store is written during an open has no test. The second archive hash pass and the on-disk table build are unmeasured at scale.
  - Implement the Rust domain boundary for files, ZIP selections and stores. Reuse existing readers, resource leases and snapshot identities. Resolve existing score, pattern-selection and policy keys to one scoped original with occurrence locators.
  - Exercise duplicate texts across archive entries, identical text in unrelated snapshots, quoted multiline CSV, changed options and changed files. Prove exact joins across batch sizes and disk spill. Surface incompatible mappings.
  - _Requirements: 1.1, 1.2, 1.3, 6.2_

- [ ] 1.2 Ship the first MCP reduction from archive to A and B.
  - Implement local stdio open, deterministic score/filter selection, expression views, count-first inspection and job status/cancel. Connect the CLI to the shared Rust operations. Add schema, byte bounds, distinct/occurrence denominators and disclosure profiles.
  - From an MCP client open a ZIP, compute A and B, inspect their overlap, select A minus B, then retrieve one explanatory row. Compare resource metrics, native memberships and results against the existing command paths. Replay a completed request and interrupt another during spill.
  - _Depends: 1.1_
  - _Requirements: 1.2, 2.1, 2.3, 4.1, 4.2, 4.3, 4.4, 6.1, 6.2_

- [ ] 1.3 Qualify reduction and paging on a multi-million-record archive.
  - Implement disk-backed membership intersections and differences, cost reporting, result reuse and cold/warm measurements. Run the A/B, spill and region-paging legs of the scale gate. Use deterministic fixture memberships to check unknowns. Demonstrate an exact region overview and repeated warm drilling through MCP.
  - Record zero reinflations or repeated inference for prepared views. Report refused optional operations and compare measured end-to-end cost with the corresponding native work. [Scale acceptance](performance.md)
  - _Depends: 1.2_
  - _Requirements: 1.2, 2.3, 4.2, 4.3, 6.1, 6.2, 6.3, 6.4_

- [ ] 2.1 Add pattern, vector and recommendation drill-down.
  - Bind current catalogue split/merge, preview and scoped nearest-neighbour operations to immutable views. Add C from the existing Venn policy, preserving score dependencies, uncertainty, native reasons and suggested routes.
  - Implement the optional GenAI recommendation protocol. Test local inference on an on-host provider separately. During development use issued batches on an explicit public fixture with a developer-selected client, including partial, duplicate and foreign responses. Verify that private evidence cannot reach this mode.
  - _Depends: 1.2_
  - _Requirements: 2.2, 2.3, 2.4, 3.2, 4.4, 6.2_

- [ ] 2.2 Deliver saved workflows, final allocation and the `triager` skill.
  - Implement all A/B/C regions, three-valued expressions, repeated reduction, named reserve and audit populations, deterministic ordering, atomic receipts and recipe replay on identical or revised inputs.
  - Through two independent client sessions save and reopen D before allocation, inspect the recommended and unknown sibling views, and allocate once. Compare parent counts in both units, ordered final IDs and pool reasons. Test that a late job completes under its original parent when the user has moved to another branch.
  - Ship the project skill named `triager`. Demonstrate the worked recipe through MCP, including sizes at every step, a scoped C, unfamiliar cases and an audit outside the candidate expression. Complete the cold AIT scale gate with actual bounded C and the warm saved-region checks. Verify reopening retains source, selected IDs, coverage, deferrals and evidence with no new inference. CLI and MCP produce the same receipt content.
  - _Depends: 1.3, 2.1_
  - _Requirements: 2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 6.1, 6.3_

- [ ] 3.1 Build occurrence-aware field, time and record views.
  - Implement declared timestamp and field projections during an admitted scan or explicit re-scan. Test one repeated original across hosts and time buckets, missing timestamps, mixed valid/invalid values, high-cardinality facets and selection-to-original drill-back.
  - Serve exact region totals, covered/unknown totals, count units, bucket boundaries and a truthful no-time alternative. Complete the warm facet/time and explicit re-scan legs of the scale gate through MCP before rendering them in the TUI.
  - _Depends: 1.3_
  - _Requirements: 1.3, 2.4, 4.3, 5.2, 6.1, 6.3, 6.4_

- [ ] 3.2 Complete SSH, MCP interoperability and scale acceptance for the terminal workflow.
  - Build the selected Taffy layout, Ratatui widgets and Crossterm event loop against the typed service. Verify the conversion from rounded shared pane edges to clipped terminal rectangles and reuse those rectangles for drawing and hit testing. Implement search, histogram, fields, region selector, results, status and evidence.
  - Over a real SSH PTY, resize sidebar width and search height with the keyboard, shrink to 80×24 and expand again. Test optional pointer drag, backtracking, cancellation and terminal restoration. Reopen an MCP receipt and check identical ordered IDs, evidence and counts after filtering. Record layout and redraw time separately from indexed retrieval. Verify resize counters show zero archive re-reads, rescoring or model calls. [Renderer research](research.md) · [Performance gate](performance.md)
  - _Depends: 2.2, 3.1_
  - _Requirements: 3.3, 4.1, 5.1, 5.2, 5.3, 5.4, 6.3_

## Implementation Notes

- 1.1: A scope's record ID is the original's first native position (file record number, packed archive position, store `seq`), because the archive reader hands out rows in another order once deduplication spills. Policy IDs derive from the path as given to the reader, selections record the canonical path, and the scope binds both: a report made under another spelling is refused with `PolicyPath`. A join streams to its callback, so on `Err` the caller must discard what it received. The summary returned by `open_file` and `open_archive` is the caller's to hold beside `Scope::held_bytes()`; `input_summary_bytes` and `archive_summary_bytes` give the figure. `join_keys` takes the source a score manifest records (`source_path`, `content_sha256`, `settings.entries`, `max_record_bytes`); reading that manifest into a source claim is left to 1.2. `MarkedInput::source_location` in `l2d-core` is now `pub`.
