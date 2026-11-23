# triager requirements

The purpose of `triager` is to reduce large log populations by comparing sets. An analyst starts with millions of occurrences and asks “Which records did my search miss, why are they here, and which can I review today?” The agent and analyst see counts first, choose an overlap or difference, and repeat until they have a manageable work set with evidence. The MCP workflow defines the interaction contract. The native terminal implements its first slice over retained evidence, with SSH and archive-scale acceptance tracked separately. [Worked recipe](recipe.md) · [Design](design.md) · [MCP interface](interface.md) · [Scale and performance](performance.md) · [Implementation plan](tasks.md)

### Requirement 1. Open one bounded local investigation

1.1 When an analyst opens a file, selected ZIP entries or a local store snapshot, the service shall bind the dataset revision, inclusion filters, parsing settings and source identity to an immutable scope handle.

1.2 When scanning or joining records, the service shall use existing memory and disk admission, archive streaming and resource leases. It shall report scanned, skipped, incomplete and unsupported counts with units and reasons. Each operation shall spill where supported or return a bounded refusal.

1.3 When records from score, selection and recommendation artifacts are joined, the service shall verify source and interpretation, preserve native locators and occurrences, detect ambiguity or drift, and expose stable keys within the scope.

### Requirement 2. Build and inspect named evidence sets

2.1 The service shall derive `set_a` by an explicit score threshold or top-k and stable tie rule, and `set_b` by literal, whole-record glob, regex, or selected refined pattern membership. It shall retain the scorer revision and the exact filter or refinement revision.

2.2 For `set_c`, the service shall record which producer recommended a record, which records it assessed, the decision predicate, reasons and evidence. Supported producers shall include the existing Venn policy and a GenAI suggestion adapter. It shall distinguish an assessed negative, an abstention and a record outside assessed coverage. Venn’s dependence on A’s scorer shall be visible when both use that scorer.

2.3 The service shall support union, intersection and difference on compatible revisions, an overview of the eight fully known A/B/C regions including “none,” and separate counts for combinations containing unknowns. It shall retain each original membership and drill down from any region to its members and evidence.

2.4 When an analyst changes a filter, time bucket, field value, template or exemplar, the service shall create a child view with its parent and scope. It shall support exact pattern membership, examples and exceptions, previews of revised queries, and deliberate vector-neighbour expansion within a view or back to the wider scope.

### Requirement 3. Allocate one review budget and retain evidence

3.1 When materializing `set_final`, the service shall apply one declared budget to deduplicated originals. Its receipt shall state candidate expression, ordering, reservations for unfamiliar or conflicting records, audit population, seed, overlap handling, shortfall and deferred counts. An audit outside the candidate union shall carry that reason.

3.2 The analyst shall be able to inspect why an original was selected or deferred, the producer decisions and underlying source locator, and the distinction between suggested priority and independently reviewed outcome. A model or queue suggestion shall never create an analyst mark.

3.3 Saving shall retain either an unallocated candidate or a final work set, together with its recipe and immutable evidence locally. A second client shall be able to reopen the saved kind, parent lineage, memberships, ordering and reasons, then continue drilling or inspect the allocation. Applying the recipe to a changed source or invoking a model again shall produce a new revision with lineage.

### Requirement 4. Expose a bounded MCP interface and an agent skill

4.1 The local stdio MCP server shall expose typed operations for scope, selection, refinement, composition, inspection, saving and job control. A noninteractive CLI and the TUI shall share the same Rust operations and receipt schema.

4.2 The `triager` skill shall guide an agent from counts to disagreements, patterns or field/time slices, and finally to selected evidence. For large archives the service shall return handles, aggregates and budgeted pages or excerpts. A client shall be able to finish the recipe without loading a ZIP, an unbounded score stream or a full result table into context.

4.3 Inspect results shall state snapshot, view revision, count unit, coverage, applied response limits, clipping and pagination. Jobs shall expose progress and cancellation. Failed, interrupted, concurrent or retried calls shall preserve committed revisions.

4.4 Private data processing and optional local-model inference shall work on the analysis host. When developing on explicitly designated public datasets, the developer’s hosted agent may consume bounded eligible evidence and supply suggestions without a local model installation. Dataset eligibility and caller disclosure policy shall be enforced on the host.

### Requirement 5. Provide a terminal drill-down over SSH

5.1 The terminal shall provide an editable, height-resizable search area, a histogram, a width-resizable field sidebar, a main result list with A/B/C membership, and a status line. An analyst shall be able to drill from an overlap or exception to a record and its evidence, then go back with their prior scope, filters and selection intact.

5.2 Time buckets and field counts shall identify their population and unit. A time histogram shall account for actual occurrence timestamps, repeated originals, timezone and missing or unparseable values. A source without a usable timestamp shall offer a clearly labelled alternative distribution.

5.3 An analyst with a keyboard and an SSH pseudo-terminal shall be able to complete the workflow, resize both panes, switch regions, inspect evidence and reopen a saved recipe. Mouse dragging may supplement keyboard controls. Across a terminal resize the UI shall preserve state and restore the shell’s terminal modes on normal exit and handled errors.

5.4 Given the same scope, recipe, producer receipts and budget, the MCP and terminal paths shall return identical ordered work-set IDs, membership states and explanations.

### Requirement 6. Reduce archive-scale populations interactively

6.1 For each Venn drill-down, the service shall return parent and child distinct-original and occurrence counts, retained share, excluded and indeterminate counts, completeness and the operation responsible. The analyst shall be able to select or invert a region within the declared population and backtrack without losing either receipt.

6.2 Given an archive larger than the memory limit, the service shall compute exact scored and filtered membership and supported set expressions through bounded batches, disk-backed membership and shared admission. It shall admit expensive vector or GenAI work against a declared scope, report uncovered records, and preserve the option to explore outside the seed search.

6.3 Once exact membership and requested time/field projections are materialized, inspecting a known region, paging its members or resizing the terminal shall require zero new archive decompressions, scoring or model invocations. New projections or a changed producer revision shall indicate any new scan in the work estimate.

6.4 The scale gate shall exercise millions of occurrences under a declared memory limit, verify multi-stage reduction against reference counts, and record elapsed time, throughput, peak accounted and process memory, spill space, cold and warm costs and bytes returned to the agent. Saved exact results shall be invariant under admitted batch sizes and disk spill.

### Requirement 7. Search in Splunk's language

7.1 The SPL search mode shall behave as close to Splunk's `search` command as a view of originals allows. Bare terms and quoted phrases shall match `_raw` by Splunk's default major and minor breakers, so `fail` does not match `failed` and `fail*` does. `TERM()`, `CASE()`, `IN`, `*` wildcards, implicit AND, and NOT, OR, AND with Splunk precedence shall be supported. Field names shall be case-sensitive, values case-insensitive, and comparisons numeric when both sides are numbers. A missing field shall fail both `field=value` and `field!=value`.

7.2 Fields shall include `_raw`, `id`, `score`, `c`, `reason`, `punct`, `linecount`, occurrence `host` and `_time`, automatic `key=value` extraction from `_raw`, and `rex` captures. `earliest` and `latest` shall bound occurrence time with absolute values. Relative times and other non-reproducible inputs shall be refused with a reason, because a saved view must replay identically.

7.3 A query shall be able to pipe results through `search`, `where` (eval expressions), `regex`, `rex`, `fields`, `head`, `tail`, `sort` and `dedup` with Splunk's defaults and options. Commands that build tables or new fields shall be refused, naming the supported commands. Set B shall accept only membership predicates over whole originals.

7.4 A filter on an occurrence field shall keep only the matching occurrences of each original, and ordering commands shall refuse such fields. View order produced by a query shall be part of the receipt and replay. Compiling and running a query shall stay within the memory budget, with bounded query size, stages and regular expressions.

7.5 Every intended difference from Splunk shall be documented in the language reference.
