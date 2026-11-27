# triager design

The unit of work is a reduction from one named population to another. The analyst asks what two signals agree on, what one found alone, and what remains unevaluated. The proposed system computes those answers locally, even when the archive contains millions of rows. The agent and the terminal see the same counts, choosing one region at a time. [Worked recipe](recipe.md) · [Execution and scale gate](performance.md) · [Current vectorization assessment](../../../docs/2026-11-18-vectorization-assessment.md)

## Shared operations and client adapters

The [analyst interaction design](analyst-flow.md) defines the shared operations. The owner chose to start native implementation from this contract while the MCP stdio adapter is pending. The Rust client and its machine-readable CLI projection share one workflow implementation. The design walkthrough supplies independent expected counts and ordered IDs. [Cross-spec acceptance](../../../docs/qualification/triager-e2e.md)

| Component | Responsibility |
|---|---|
| Existing Rust readers and producers | Admit and stream files, ZIP entries and stores. Reuse fitted scoring, typed queries, catalogue refinement, NPU exploration, Venn decisions and resource accounting. |
| Shared Rust investigation service | Own snapshot identities, membership, occurrence projections, set expressions, reduction counts, allocation and saved receipts. Use disk-backed state when needed. |
| Local MCP adapter and `triager` skill | MCP exposes bounded, typed operations. The skill teaches the sequence **count → compare regions → reduce → inspect evidence → allocate**. A CLI adapter offers machine-readable parity. |
| Native terminal client | Render the shared Rust view on the analysis host. The first version opens a retained report or the synthetic walkthrough and shares native receipts between processes. The workbench over the service (task 3.2) makes the MCP tool calls on a service in its own process. It starts from a recipe or a saved receipt, and its views, allocations and receipts are those of an MCP client. It is accepted over a real SSH PTY. |

One distinction matters throughout. The existing Venn *policy* is one possible producer of C. The Venn *drill-down* in this feature is set algebra over independently named inputs. An analyst can reduce A and B before any NPU or GenAI expense. Adding C records the places it ran and the score or previous selections it consumed. [Existing Venn policy](../../../docs/qualification/venn-policy-contract.md) · [Proposed interface](interface.md)

## What the foundation already supplies

The archive reader, adaptive batches, disk-backed deduplication, score parts, host/NPU admission and immutable producer artifacts already exist. Use their native operations and capability checks. The existing policy can require admitted whole-history work. Reduce to an explicit scope before invoking it when appropriate. The current CLI produces scores, filtered selections and recommendations independently. The missing link is an exact, reusable join across their identities. [Existing CLI](../../../docs/qualification/analyst-cli.md) · [Measured archive operations](../../../README.md#what-works-today)

| Native evidence | How to join it |
|---|---|
| File selection with first original position | Resolve through that parser's occurrence map. Quoted multiline CSV illustrates why physical line numbers are insufficient. |
| ZIP entry and entry position, score part row | Resolve under the pinned archive digest, entry selection, parser and manifest. Keep locations for repeated occurrences. |
| Policy decision on a file or archive query | Resolve the query ID derived from the source path and original text, even when a historical MarkSet comes from a store. Verify the snapshot and preserve the native key. A path change can affect policy ordering and requires a new producer receipt or replay with the original path. |
| Store original or stored selection member | Resolve its native store ID within the pinned snapshot. A producer accepting only file or archive queries must advertise that capability. Any future store-query adapter must verify its mapping. |

Maintain a local relation between accepted distinct originals and their occurrences. Report malformed or skipped input separately in scope completeness. A skipped item with no valid original has no record ID to include in a Venn region. A record carries the durable source reference, exact count and retrievable locators. Occurrence projections carry any selected timestamp and typed facet values. If the same original appears on two hosts or in two time buckets, filtering to one host or bucket selects the matching occurrences and their distinct parent originals. Show both units. A drill back to the original can inspect its other occurrences in the wider scope. Sources which retained only a first position and total need an admitted pass to build the projection. [Current input representation](../../../crates/l2d-core/src/input.rs)

## Membership, cost and evidence

For each named set keep known-in, known-out and indeterminate membership within a fixed universe. Reasons distinguish abstention, unsupported input, skipped scope, incomplete execution and missing model output. Union is true if either operand is true. Intersection is false if either is false. Difference means intersection with the right operand's complement. Complement is relative to the declared parent and preserves unknowns. Counts for the eight fully evaluated Boolean regions plus unknown combinations must reconcile to the entire parent. Separate known-negative from unknown when asking “what did C miss?”

Persist the expression tree and the parent/child counts for each step. Display retained percentage with its denominator, including the out and unknown counts. On a zero-size parent, percentage is undefined. Keep branch history when the analyst broadens or backtracks. Rank within a named region using the declared sort and a source-bound tie-break. Only the final allocation consumes review slots. Apply reservations, skip duplicate slot assignments and retain explicit evidence for audited records outside the candidate expression.

Derive repeated region counts, result pages and prepared field/time distributions from a local membership index. Expensive scoring, inference or reinflation gets its own estimate and receipt. Existing artifacts can be reused only when identities and settings agree. The MCP payload has separate limits from engine admission. A large scan is an ordinary server job even if it returns three integers. Limit evidence disclosure cumulatively for a model batch or agent session as well as per response, with visible remaining allowance. [Performance contract](performance.md) · [MCP limits](interface.md)

## Terminal workbench

Illustrative state from the recipe. The numbers are example counts of distinct originals. In the histogram, horizontal position represents time and bar height represents occurrences.

~~~text
Search [glob]  *login*                       [edit / resize]
U > A minus B > recommended           1,200 of 8,000 | 15%
Time UTC    ▁▂▇▂▁▃▂▁                     missing time: 16
Fields               | Results                  A  B  C
  host               | record      score        +  -  +
  source             | ... open row for reason and evidence
  pattern            |
Regions / coverage   | Low score | unknown | audit | deferred
Status  snapshot S   | exact counts | 24 review slots | ready
~~~

Search accepts a broad literal, whole-record glob or regex and can grow vertically for a multiline edit. A dedicated region selector shows overlap, differences, “none” and unknowns as a count table usable without color. The analyst can select a field value, histogram bucket or region to create a child view, open a row for native evidence and return to the same scroll position. Status shows the current scope and units, exact/partial state, job progress or refusal, and remaining review budget.

Use Taffy to lay out the visible panes in terminal-cell units. Keep a small layout tree whose inputs are the frame size, sidebar width, search-editor height and pane visibility. Convert the solved edges to clipped integer Ratatui rectangles. Round each shared boundary once so adjacent panes meet at the same cell. Use the same rectangles for widgets, divider hit testing and focus. Keep the result table virtualized to the current page so layout cost depends on visible panes and rows.

Crossterm handles focus cycling, Enter to drill, Escape to go back and a discoverable keyboard mode for resizing sidebar columns and search rows. Mouse dragging can adjust the same dimensions. Collapse the fields panel on narrow terminals. Window, divider and focus changes redraw from retained view state. Persist personal layout separately from the evidence recipe and display source text with terminal controls escaped. Test through an ordinary SSH PTY. [Framework decision and SSH test](research.md)

## Search language

The SPL search mode follows Splunk as closely as a view of originals allows (Requirement 7). The [language reference](../../../docs/2026-11-18-triager-search-language.md) lists the syntax, built-in fields and every documented difference from Splunk.

- **Pipeline.** A query is split at `|` outside quotes after removing ```` ``` ```` comments. The first stage is always a search (`search` is optional), as in Splunk. Later stages are compiled in order. `fields` and `rex` change which fields later stages can read when the query is compiled, so evaluation never consults stage history.
- **Terms.** A bare term or quoted phrase matches `_raw` the way Splunk's segmenter does. It must begin and end at a major or minor breaker, unless its own edge character is a breaker. `*` stays inside one major segment. `TERM()` requires major breakers and `CASE()` is case-sensitive. The matcher is written by hand and uses no regex, so a term holds no compiled program and no budget share.
- **Fields.** Built-in fields come first: `_raw`, `id`, `score`, `c`, `reason`, `punct`, `linecount`, and per occurrence `host` and `_time`. Other names come from `rex` captures or from Splunk-style automatic `key=value` extraction over `_raw`. Field names are case-sensitive. Values compare numerically when both sides are numbers. `search` matches case-insensitively and `where` case-sensitively. A missing field fails both `f=v` and `f!=v`.
- **Rows and occurrences.** Rows are originals. A filter stage that reads `host` or `_time` runs once per scoped occurrence and clears the bits of rejected occurrences in the branch's occurrence mask. Any stage that drops an original clears all of its bits. `sort`, `dedup`, `rex` and `head (...)` refuse occurrence fields, because their meaning depends on a single value per original.
- **Order.** `sort`, `head`, `tail` and `dedup` make view order part of the result. The receipt digest already covers order, so replaying a saved receipt checks it.
- **Set B.** Set B must be a membership predicate over whole originals. It refuses occurrence fields and ordering commands.
- **Determinism.** Relative times and functions such as `now()` and `random()` are refused, because a saved receipt must replay identically.
- **Resources.**
  - Parsing reserves 256 KiB, and each regex (`regex`, `rex`, `match`, `replace`) counts as one `REGEX_HELD_BYTES` share. A query has at most 16 regexes, 32 stages and 8 KiB of text.
  - A run is admitted with one `require_whole` covering it and what the caller holds beside it: scratch, the compiled query, an occurrence mask and the input copy. The run's share covers working rows, the dropped-row list, capture slots, the result, sort keys and order, and dedup keys with owned text. It also covers per-row string scratch, computed for each eval node from the longest text field: `replace` may insert its replacement at every position, and case mapping may triple text. A query whose bound exceeds the limit is refused before it runs. A counting-allocator test checks measured peaks against the estimate. The run holds nothing after it returns.
- **Out of scope.** Commands that build tables or new fields (`stats`, `eval`, `table`, `timechart`, ...) are refused with the list of supported commands. They need a result type other than a view of originals, plus a statistics pane.

## Acceptance and ownership

The bounded terminal exercises A/B reduction, named C, allocation and saved receipts today. Next, feed the same operations from source-verified archive membership on disk and drive them over MCP stdio. Time and field coverage must come from actual occurrences. The scale gate measures this full path. Parent qualification measures how well a frozen decision rule transfers to future cases. [Implementation tasks](tasks.md) · [Parent qualification](../npu-log-decision/tasks.md)
