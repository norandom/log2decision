# Vector exploration (task 2.19)

`l2d discover explore` retrieves original saved members around one analyst-selected
exemplar. It makes no classification or deployment qualification claim. The baseline
remains the classifier. Both routes expose human and JSON output and atomically publish
a new `report.json` bundle; existing selections, refinements and labels are unchanged.

## Using the command

Use `discover details --artifact seed/refinement.json --pattern 0` to find
`exploration_exemplars[].exemplar_row`. The row number refers to complete saved
membership, not the filtered details array. For example:

```powershell
l2d discover explore --artifact seed/refinement.json --exemplar-row 0 `
  --method character-trigram --scope all-patterns --top-k 20 --radius 0.2 `
  --out lexical-result

l2d --json --memory-limit 2GB discover explore `
  --artifact seed/refinement.json --exemplar-row 0 --method npu `
  --profile-dir D:\BuildArtifacts\log2decision\task-2.17\npu\f0-s0-mask `
  --scope all-patterns --top-k 5 --radius 0.2 `
  --expand-artifact scope/refinement.json --out npu-result
```

Set `L2D_RUNTIME_DIR` to the provisioned runtime and `L2D_NPU_WORK_DIR` to a
disposable directory under the project's build-artifact directory. Without the latter,
the adapter uses the developer shell's temporary directory. No Python is used.

Outside-query expansion requires `--expand-artifact`, another explicitly searched and
saved discovery scope. It may be an input, selected archive entries, or a store.
The command checks both source snapshots before retrieval and again before publishing.
Overlapping members of the same source are not scanned twice as evidence. New
suggestions have no review label, even when their originating artifact was reviewed;
their source, member ID, selection revision and refinement revision are retained.
Source receipt labels remain in the original artifact. Expansion never edits it.

`within-pattern` filters to the exemplar's pattern; `across-patterns` excludes it;
`all-patterns` includes both. Filters precede top-k. Outside-query expansion is refused
with `within-pattern`. Ties use refinement revision and member identity. Replay with
the previous report's `profile` in `--expect-profile`; an incompatible profile names
both identities and refuses. Changing the source also refuses.

## Methods and evidence

The NPU profile is the accepted 2.17 f0-s0-mask hierarchy: immutable SHA-256 checks
of `byte.int8.onnx`, `line.int8.onnx`, `table.bin`, `exp-table.f64` and `layout.json`;
Ryzen AI 1.9.0 and NPU driver 32.0.20102.4480. These are **not** the older single-graph
`EncoderSession` package. The two graphs use the existing placement and first-run
execution checks, allowing only boundary quantize/dequantize arithmetic off the NPU.
The 8-bit byte logits determine entropy cuts; byte features are pooled into patches;
the stride-1 line graph is pooled over real patches. Cosine uses float64 normalization
and sums. No templates, text masks, CPU encoder fallback, or model retraining occur.

Original UTF-8 input must be nonempty and at most 1,024 bytes. Longer originals are
unsupported, never truncated. Candidate failures appear in `unsupported`; an invalid
exemplar refuses the operation. Zero, non-finite or differently sized vectors are
unsupported. A new driver/model/runtime requires a new verified profile, not implicit
reuse. Store originals are re-encoded; old stored vectors are not silently mixed in.

`character-trigram` is the existing exact Unicode character-trigram/Jaccard arithmetic,
shared with lexical pattern discovery, applied to original text without discovery
masks. It does not load an encoder or require an NPU. Its distance is `1-Jaccard`;
the NPU route uses cosine distance. The two radii have different meanings.

The representative is the real analyst-selected exemplar, not a computed medoid.
Boundary and distance range describe the returned top-k. Support counts cover every
eligible member within the radius **before** top-k, with occurrences separate from
globally distinct original texts. Copies of the exemplar are not independent support.
Group identity includes source snapshot, refinement revision and pattern ID. Same- and
cross-group distinct counts may overlap when a text appears in both. Ungrouped records
are not group-diversity evidence. Endpoint/host independence is unavailable and must
not be inferred from these counts. Connectivity through neighbours is not computed:
all distances represent direct exemplar reach.

## Resource limits

This first implementation explicitly holds bounded refinement memberships and reports
in memory, admits them before loading, and refuses oversized operations. It has no
disk form for retrieval reports. Reservations include repeated source metadata,
distinct-text trees, invalid records and output copies. No all-pairs vector matrix is
built. One original per fixed-shape NPU image is used; memory is rechecked for every
candidate and the actual one-row plan is recorded. This favors auditable bounds and
does not claim the gate's batched throughput.

The NPU session reserves a conservative 1 GiB estimate, informed by the gate's measured
524 MiB peak. Native provider allocations are not fully measured by the Rust allocator;
this is not a hard native allocation bound. Host/NPU guards serialize cooperating
operations; outside-tool NPU utilization remains unknown. Resource records retain the
process memory limit, counted peak, waiting and spilling status.

## Frozen engineering comparison, 2026-11-17

[`vector-exploration-fixture.json`](vector-exploration-fixture.json) was frozen before
retrieval. SHA-256: `f90ea9926bcaef8f2971c47d98732dfd4b56ed69a06901df424acfdebde3383a`.
It is a synthetic engineering relevance review authored by the implementer, **not**
independent human security labels or a qualification holdout. Nine distinct texts:
two seed matches, three relevant failures missed by `login failed`, four irrelevant
records. Settings were frozen at exemplar row 0, top-k 5, radius 0.2; no tuning followed.

| Route | Returned seed neighbours | Returned expansion | Relevant missed examples found | Expansion precision | Missed-example recall | Unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable NPU cosine | 1 | 4 | 3 | 3/4 | 3/3 | 0/9 |
| Exact character trigram/Jaccard | 0 | 0 | 0 | undefined (no results) | 0/3 | 0/9 |

The NPU ranked `authentication rejected` first (distance 0.00656664), then the
**misleading `login succeeded`** record (0.01030973), the other seed (0.01297332),
`login unsuccessful` (0.01326303), and `login denied` (0.01460773). All eight eligible
records fell inside its radius; top-k, not the radius, excluded the remaining three
irrelevant records. Thus radius-only expansion would include all four irrelevant
records. The lexical route's same numeric radius returned nothing. Different distance
scales and nine synthetic examples cannot establish general superiority, classification
quality, uncertainty bounds or deployment readiness.

For reproduction, write the fixture's `records[].text` in order to a UTF-8 line file;
save a `discover search --phrase "login failed"` scope and a `--glob "*"` scope;
run both commands with the frozen settings. Compare only hits with
`outside_saved_search=true` against the pre-existing relevance annotations.

Validation evidence is in `D:\BuildArtifacts\log2decision\task-2.19`:
`red.log`, `green-initial.log`, `affected-final.log`, `npu-adapter-final.log`, `memory-final.log`, `workspace-final.log`,
`npu-retrieval.json`, `lexical-retrieval.json`, `smoke-final-result.json`, and the immutable `smoke` bundles.
The hardware test reproduces a saved gate vector bit-for-bit, repeats it, and refuses
empty and over-limit originals. The allocator test includes repeated archive scope
metadata, report retention and JSON copies: 3,700,932 bytes peak growth against
99,373,247 bytes admitted; preallocation refusal grew the heap by 1,265 bytes.
