# Large-data execution and acceptance

An analyst opening a multi-gigabyte ZIP should pay for the scan once per agreed scope and projection. Opening a saved overlap, paging through results or resizing the terminal should operate on retained local evidence. This is the proposed Triager contract for the Rust service, MCP adapter and terminal client.

## Existing measurements

The public AIT `russellmitchell.zip` measured 7,132,670,599 bytes. The existing all-entry runs observed 30,916,764 records in 11,870 entries, skipped 357,123 and retained **30,559,641 occurrences of 8,276,854 distinct originals**. These are reference measurements of the parent commands. Triager indexing and query costs require their own measurements. [Archive design and manifest](../npu-log-decision/design.md#zipinput-and-result-parts) · [Task 1.17 and 1.20 measurement records](../npu-log-decision/tasks.md#implementation-notes)

| Existing release measurement under `--memory-limit 512MB` | Elapsed | Counted heap peak | Process working-set peak |
| --- | ---: | ---: | ---: |
| Ingest all entries with disk deduplication | 57.5 s | 374.4 MiB | 379.3 MiB |
| Baseline score all entries, including writing result parts | 487.8 s | 355.0 MiB | 359.1 MiB |

Scoring produced 386 score parts (277,133,027 bytes) and 128 occurrence parts (9,577,139 bytes). In-memory and spilled runs produced identical logical row hashes for both tables. A separate entry-filter test reading only `*.log` saw 22,880,272 records and 5,642,082 distinct texts. [Archived benchmark summary](../../handover.md#results-worth-remembering)

## Carry the four performance criteria into the service

1. **Batch by available memory.** Charge readers, decompressor windows, queued batches, join buffers, result pages and serialization before allocating. Batch sizes and reader counts may change cost, while memberships, logical ordering and exact counts must agree.
2. **Spill under one global limit.** Admit joins, facet aggregation, rank ordering and growing membership indexes against the same budget. Offer a bounded disk form or a refusal with required and available memory/disk. Record counted allocations and observed process peaks separately. Include model-runtime memory observations when available.
3. **Stream inputs and outputs.** Write immutable index/evidence parts incrementally and publish a checked manifest last. Cancellation, interruption or disk exhaustion must leave the last committed revision usable. A whole-history operation that lacks a streaming form must declare and pass its existing admission check.
4. **Admit one heavy operation at a time.** Use the existing cooperating host lease and NPU lease across MCP, CLI and terminal processes. Return a wait or refusal with the reason. Lightweight reads from a committed revision can proceed under their own bounded read budget. [Parent operating criteria](../npu-log-decision/design.md#operating-criteria-field-machines-not-labs)

## Read and index only what the investigation needs

| Source | First materialization | Reuse on the same immutable revision |
| --- | --- | --- |
| File | Stream records with the chosen parser, actual logical positions and requested occurrence projections. Bind the source fingerprint and adapter settings. | Resolve IDs through the local locator index. Use an admitted bounded evidence read for an original whose excerpt was omitted. |
| ZIP | Apply entry-name filters against the directory before inflation. Stream selected entries with the existing reader, limits, integrity checks, skipped/damaged accounting and optional disk dedup. Build requested time/field projections before identical text is collapsed. | Reuse matching checked score/occurrence parts and stored projections. Bind archive digest, entry filter and parsing settings. A requested new field or raw excerpt can require another pass through a compressed entry. Estimate and expose that work before starting a job. |
| Store | Bind the store snapshot and stable record IDs. Use compatible native indexes for eligible queries and verify the exact predicate on retrieved candidates. | Reuse immutable membership and value indexes. Report missing occurrence history or field/timestamp coverage when the store cannot supply it. |

Keep a disk-backed mapping from native locator to scope-local record ID. Index membership and assessed coverage for each producer revision by that ID. Store explicit reasons for unknowns, scores/rank keys and references to supporting receipts. Keep occurrence-level field/time postings alongside distinct-ID membership. Use a snapshot-pinned ordered index or bounded sorted parts for cursor paging. Every retained index records the projection, extractor, scorer and policy revisions it depends on.

Push a scope or predicate earlier only when its evaluation universe and semantics are preserved. An entry-name glob can skip ZIP entries. A content glob or regex generally needs a record scan unless an index supplies a guaranteed superset and the exact match is verified. A precomputed global top-k or a background audit must keep its declared population when a user drills into B. If C ran only over B, preserve C as unevaluated outside B. A missing C decision cannot become an evaluated negative through pushdown or caching.

For each job report source/entry bytes read and inflated, entries skipped/read, records scanned/skipped, accepted occurrences, distinct originals, candidate and output counts, and rows actually scored, encoded or submitted to GenAI. Record scanned/reused partitions, spill bytes and disk peak, elapsed/queue time, charged memory peak, observed process peak and response bytes. Report unavailable counters explicitly. Show the reduction as input and output with their units and denominator. For example, “540 of 8,000 distinct candidates assessed” lets the analyst distinguish narrowing their hunt from evaluating its remainder.

## Proposed scale gate

| Exercise | Evidence required to pass |
| --- | --- |
| Deterministic multi-million-record fixture | Generate at least 4 million occurrences and 2 million distinct originals across multiple entries, with duplicate text across time bins, missing fields, damaged or skipped input and deliberately partial C coverage. Compare streamed and forced-spill results with an independently generated oracle for eight known regions, unknown states, accepted/skipped counts, bin counts, ordered final IDs, allocations and deferrals. Vary batch size and page size. |
| Cold public-archive acceptance | On the reference host, run all-entry AIT ingestion, score reuse or generation and A/B composition at the reference 512 MB setting. For bounded C, record the producer, its separately admitted memory limit and scope. The current NPU session reserves 1 GiB before other work. Record actual C coverage, index build time/size, resource counters and input, part and result hashes. Recover the reference counts for the same adapter and scope. Demonstrate spill and clean refusal when storage is insufficient. [Product NPU bounds](../../../docs/qualification/vector-exploration-contract.md#resource-limits) |
| Warm analyst drill-down | Reopen the committed archive work set, inspect an overlap, page forward/back, select a retained field and time bin, then resize and return. For this sequence require zero archive inflation, source scans, re-scoring, NPU calls and GenAI calls. Count index reads and bytes returned. Measure first-page and drill-down p50/p95 over at least 30 interactions on the reference host. Proposed usability target is p95 ≤ 2 s for an indexed 25-row page or a previously materialized aggregate. Record machine, revision and cache state. |
| Explicit rescan and expiry | Request a field absent from retained projections. Show the missing coverage, estimated affected scope and scan cost, then run an explicit admitted job against the verified source. Reconcile the derived view with an independent full computation. Change one source/adapter/producer revision and verify incompatible artifacts cannot silently supply results. Repainting, sorting, paging and cancelling an inspection must never schedule producer inference. |

Keep machine-readable run receipts and semantic output hashes alongside the workstation timings. The performance gate covers resource use, interactivity and reproducibility. Security effectiveness on future logs is evaluated separately in parent task 2.22. [Interface limits](interface.md#drill-down-and-limits) · [Parent roadmap](../npu-log-decision/tasks.md#active-roadmap--owner-alignment-2026-11-17)
