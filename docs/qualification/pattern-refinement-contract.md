# Pattern refinement and discovery CLI

Task 1.28 implements Requirements 17.4, 17.5, 17.6 and 19.7 under the memory and resource rules of Requirements 12 and 15. The design boundaries are `PatternRefinement`, Discovery CLI and the MarkSet bridge. Existing labelled `l2d mark` behavior is unchanged.

## Analyst workflow

Use one `l2d discover` command group. Every command accepts global `--json` and memory options. Human output prints indented fields; JSON output is one complete document. Search supports explicit literal (`--phrase`), whole-record content glob (`--glob`), regular expression (`--regex`), or store-only FTS5 trigram expression (`--fts`). Content searches are case-sensitive unless `--insensitive`; FTS5 retains its intrinsically insensitive semantics. `--entry` filters archive paths independently of content matching.

Example PowerShell session (choose new destinations for every write):

```powershell
l2d discover search --input .\logs.txt --glob '*login*' --out .\discovery\seed
l2d discover patterns --artifact .\discovery\seed\refinement.json
l2d discover details --artifact .\discovery\seed\refinement.json --pattern 0
l2d discover preview --artifact .\discovery\seed\refinement.json --pattern 0 --out .\discovery\preview.json
l2d discover save-query --artifact .\discovery\seed\refinement.json --preview .\discovery\preview.json --out .\discovery\query.json
```

Search writes `selection.json`, the task-1.27 `catalogue/`, and `refinement.json`. It never uses dataset/store labels as analyst reviews. `--max-members`, `--max-patterns` and `--max-text-bytes` expose limits. Pattern overflow remains an explicit ungrouped remainder; inspect it using `details --ungrouped`. All exact saved records, occurrence counts and reviews are retained.

To refine a group, use `split --pattern 0 --regex 'user=alice' --out split.json` or `merge --patterns 0,3 --out merged.json`, with `--artifact` naming the previous refinement. A split must leave both sides nonempty; duplicate/unknown merge IDs refuse. New revisions preserve exact membership, source/catalogue identity, parent revision and action. Group IDs remain stable except that the selected split side receives a fresh ID and a merge keeps the smallest ID. A merge keeps that ID's priority. Exceptions remain in exact rows; summary examples and differing literal tokens make outcome collisions visible. Summaries are rederived from rows after refinement. `priority --pattern 0 --priority promoted|hidden|normal --out next.json` changes only priority and lineage.

## Query previews

With no `--regex`, `preview` proposes an anchored regex from escaped literal tokens. Only explicit discovery-mask positions or positions whose original tokens differ become wildcards. Literal metacharacters (including an actual `<*>`) remain escaped. Variable token counts refuse automatic generation and ask for an explicit regex to preview. `--regex` previews an analyst-edited alternative.

The preview lists exact matching/missing member IDs, plus matching/nonmatching IDs in **all other saved selection members, including ungrouped members**. This background is complete within that stated scope; it is neither an outside-source sample nor a corpus-wide precision estimate. Every preview pins the selection and refinement revisions. `save-query` recomputes and compares the entire saved preview; changing priorities, membership, reviews, query or source invalidates acceptance. The preview receipt is saved separately from the explicit query-save action. Templates are discovery summaries, and no whole-corpus matcher accuracy is claimed.

## Explicit review and export

Preview a batch before applying it. The preview lists affected original rows, their existing states, the proposed state, analyst/action and a review revision binding the target, proposed state, analyst/action, selection and refinement. Apply explicitly with that review revision just inspected:

```powershell
l2d discover review --artifact .\discovery\seed\refinement.json --selection .\discovery\seed\selection.json --pattern 0 --state bad --analyst alice --action 'Reviewed failed logins'
l2d discover review --artifact .\discovery\seed\refinement.json --selection .\discovery\seed\selection.json --pattern 0 --state bad --analyst alice --action 'Reviewed failed logins' --apply --expect-revision <preview-revision> --out .\discovery\reviewed
l2d discover export --artifact .\discovery\reviewed\refinement.json --selection .\discovery\reviewed\selection.json --label bad --name failed-logins --out .\discovery\export
```

The states are `good`, `bad`, `unresolved` and `conflict`. A partial review preserves all prior decisions outside the target pattern. Previously unreviewed other members remain unresolved. Each applied review writes a new SelectionSet receipt and matching refinement together; earlier immutable receipt artifacts supply the history. Subsequent reviews must use both new files.

Export writes an immutable bundle containing `marks/<name>.json` (MarkSet format 3), `selection.json` (detailed action receipt), and `refinement.json`. The requested resolved label is selected. Unresolved/conflicting members remain de-selected exclusions, so fitting cannot silently treat them as the assumed opposite class. The other resolved label is not included in the requested class. The bundle itself can be passed as the existing fitting workflow's `--project`; the MarkSet contract and redaction/fitting APIs are unchanged.

## Persistence, source checks and limits

Refinement revisions include the original catalogue and source, selection revision, complete rows, derived counts, priorities, parent and action. Load checks revision hashes, canonical member/group order, exact counts, source catalogue version and settings. Selection binding also checks source, revision, all identities and review states. Original input bytes are rehashed before publishing source-dependent actions. Stores are checked against the task-1.27 occurrence-aware snapshot, so inserting duplicate IDs with increased counts also invalidates reuse.

Saved snapshots can still be listed and inspected after the live source changes. A new split/merge/priority, preview, review or export refuses changed live sources. This separates historical inspection from source-dependent writes.

Selection/catalogue persistence keeps its existing disk form. Refinement uses an explicitly bounded whole-membership snapshot for arbitrary split/merge and exact preview differences. It has no disk refinement execution form: metadata-derived admission occurs before loading, and an oversized operation refuses with estimated/available memory. The CLI holds an explicit conservative reservation for retained JSON reports and serialization across the complete command, including two revisions and the review report. Native-heap tests measure admission/refusal as well as successful processing. Resource guards serialize heavy operations and record usage.

Single files publish atomically without overwrite. Search/review/export construct a unique sibling staging directory, complete all checks and files, then publish the directory; errors remove only that staging directory. Existing destinations refuse. Receipt publication is indivisible with export, and old task-1.26/1.27 artifact formats are unchanged.

## Verification

- `crates/l2d-core/tests/refinement.rs`: exact membership/counts, split/merge lineage, invalid split/merge, query differences and escaped Unicode/multiline literals, priority neutrality, stale preview, immutable replay, partial-review preservation, supervised exclusions, corruption/source refusal.
- `crates/l2d-core/tests/refinement_memory.rs`: native Rust heap during create/split/preview/save/load and refusal before parsing.
- `crates/l2d-store/tests/marks.rs`: retained mixed review/conflict states and occurrence-only store mutation invalidation.
- `crates/l2d-cli/tests/discover.rs`: actual binary broad search to reviewed examples without a prewritten regex; human/JSON output, replay, split/merge, generated/edited query previews, explicit two-step review, format-3 receipt export, stale/invalid/low-memory/changed-input refusals; archive-entry scope and store-count refusal.

Execution evidence is recorded under `D:\BuildArtifacts\log2decision\task-1.28`. These fixtures validate the workflow and contracts; they are not deployment qualification of labels or query generalization.


### Batch and resource reporting

Every discovery result includes the existing common memory report: observed free memory, applied limit/reserve, counted peak and spill status. File searches report their real input BatchPlan. Archive selection and catalogue scans report their respective actual plans, including chosen size, observed min/max, and changes with free-memory observations. The original archive APIs delegate through new observer-capable wrappers; observers do not change batching or persisted revisions. CLI reports reserve 1 MiB before collecting at most two plans; more than 1,024 change events in one plan refuses explicitly before copying metadata.

Store search and catalogue scans report the actual delivered frame count and min/max rows, alongside the configured maximum and an explanation that byte-budget cuts and the final short frame are included. These are observed frame sizes, not fabricated archive-style BatchPlans. FTS5 index selection uses a cursor and explicitly reports no selection frame batches; the subsequent catalogue still reports its real frames. Saved-artifact-only commands have no input batch plan; their common report and ResourceGuard record cover bounded whole-artifact work. ResourceGuard records the wait decision as well as the final peak/limit/spill figures.
