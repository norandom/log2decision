# Task 2.11: outlier distance and the effect of k on CSIC

Larger k did **not** separate the constructed rare/new proxy better in this development experiment. Keep the specified default k=5; this experiment does not select an operational threshold or qualify a deployment.

## Product contract

`l2d_decide::outlier::rank_outliers` returns the squared L2 distance to the k-th nearest stored record, largest first, with deterministic input-order ties. It calls `Store::nearest_with`, preserving the Store's arithmetic. This is a distance, not a probability or a classifier. MarkSet scoring remains the classifier.

The caller supplies original text, source and vector under a named encoder identity/runtime release. Source/text derive the stored ID. An exact stored match must have bit-identical supplied/stored vectors before its ID is excluded; a distinct record with an equal vector remains a neighbour. A small store is refused with the stored and needed counts (k+1 for a stored query). Occurrences do not multiply neighbours. Per-label counts and mean squared distances describe the actual k rows used, including unknown as a separate category; they do not establish independent analyst support. The existing close-enough path reuses the same LabelEvidence type and retains its own named metric and support rules.

Original-record flags must be known and indicate no empty, partial or over-limit record. Zero/nonfinite vectors, incompatible origins and distance overflow are unsupported. This evidence-specific check does not change general Store search semantics. Store-owned validation streams vectors, and a connection-local generation guard refuses the whole ranking if another connection commits any change during it. The token is not a portable content revision. Results include origin, store path, generation and resource-record location.

**External-query API precondition:** the caller must supply the supported original-record vector under its actual pinned encoder/preprocessing profile. Ranking checks vector shape, finite/nonzero values and declared origin; it does not independently prove how an external vector was encoded. Generic EncoderIdentity does not expose an input-length limit. The task 2.9 encoder adapter must enforce its encoder's real flags and limits before calling this API, rather than infer a universal 1,024-byte limit. Stored-query bit checks and stored-candidate flag checks do not attest external-query provenance.

Inputs, compact sorted outputs, JSON allowance and scan diagnostics are preflighted. Queries run one at a time, stored vectors in adaptive batches. Returned results remain reserved until the caller drops them and restores its incoming budget reservation. Errors restore the incoming hold. ResourceGuard controls admission and writes peak-counted-memory evidence. The whole output needs memory for sorting; it is refused before loading when its estimate does not fit. This is not a claim of full process-peak qualification under requirement 15.2.

## Frozen development comparison

Source: task 2.18 `run01`, 25,608 distinct reconstructed CSIC originals. Seed 20261117. The frozen f0/s0/mask actual-NPU vectors are reused, with no encoding, model fitting or operational tuning here. The script checks all provenance hashes, the vector hash against the encoding audit, every original text against mapping, and every raw row index.

**Layout is 128-dimensional little-endian f32, raw/redacted interleaved; raw original i is vector row 2*i.** Producer proof: `tools/science/transfer218.py` writes both arms for each record; `transfer218_verify.py` compares `[::2]` with historical actual-NPU vectors. Earlier inspection guesses of 256 dimensions or contiguous halves were superseded before any distance calculation. No figures use those guesses.

The original proposed 8+8 endpoint panel was impossible: only eight endpoint groups have at least 18 distinct originals. Before distances, the panel was reduced to three rare and three new endpoint groups, 16 queries per group (48+48), against 2,048 reference originals. Exactly two distinct originals of each rare endpoint appear in reference; no original of a new endpoint appears there. Remaining reference originals come from other endpoint groups. Query/reference originals are disjoint. These are deliberately constructed endpoint-presence proxies, **not** temporal novelty, true frequency rarity, attack relevance or new human ground truth.

The encoder training cohort includes 32/48 rare queries and 48/48 new queries. This is previously inspected development data, with representation-training overlap. There is no untouched final cohort, independent new human labelling, or temporal holdout. There are only three endpoint clusters per class.

Candidates were fixed to k={1,5,11,21}. Metrics were fixed before distances: new-positive ROC AUC, per-proxy median distance, top-32 new fraction, and 2,000 paired endpoint-cluster bootstrap differences against k=5, seed 20261117. The resampling ranges below are descriptive and severely limited by three clusters per class; they are not qualification confidence intervals.

| k | AUC, new vs rare | Median rare distance | Median new distance | New fraction, top 32 | Descriptive 2.5–97.5% AUC difference vs k=5 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.89019 | 0.03501 | 0.57006 | 0.90625 | −0.07292 to 0.11719 |
| 5 | 0.87370 | 0.11177 | 0.82977 | 0.90625 | 0 to 0 |
| 11 | 0.85113 | 0.14312 | 0.92130 | 0.87500 | −0.06858 to 0.00521 |
| 21 | 0.85938 | 0.20203 | 1.05653 | 0.87500 | −0.06510 to 0.01302 |

The point estimates favour smaller k on this panel. With exactly two represented originals, increasing k crosses that endpoint's local reference support by construction. Neither that mechanism nor these estimates establishes semantic novelty detection or broad superiority of a particular k. The specified default is unchanged.

## Evidence and replay

Evidence root: `D:\BuildArtifacts\log2decision\task-2.11`.

- `csic-v2/plan.json`: pre-distance settings, source hashes, full original IDs, endpoint memberships, training overlap and imported profile. SHA-256 `3bfd74fd83fe817c173eed79a8ca94829a5b44d4e87be0e7e2e732382269c727`.
- `csic-v2/frozen.json`, `panel.jsonl`, `k{1,5,11,21}.json`, `report.json`: frozen fixture hashes, actual production results, memory plans and comparisons. Vector source SHA-256 `b06efa1d5e31263c8166edcbc0328f793f5232f1a06120479cc5204b6a544f95`.
- `csic-first.log` preserves the first pre-distance failure: a descriptive runtime string with spaces was invalid for Store. The v2 plan corrected only the runtime schema to release `1.9.0`, with separate ONNX Runtime `1.30.0` and driver `32.0.20102.4480`. Panel IDs and comparisons did not change; the original failed plan is retained.
- The Rust fixture creates a disposable Store, fills in batches of 64, invokes production ranking for all four k values and writes JSON. It imports every vector bit unchanged, uses no all-pairs matrix, and leaves original artifacts untouched.

Replay preparation into a **new** directory:

```powershell
& D:\BuildArtifacts\log2decision\science-venv\Scripts\python.exe tools/science/outlier211.py prepare D:\BuildArtifacts\log2decision\task-2.18\run01 <new-evidence-directory>
$env:L2D_OUTLIER_PANEL = '<new-evidence-directory>'
# Run the exact final outlier_csic executable recorded in validation.md:
& <outlier_csic.exe> --ignored --exact frozen_csic_k_comparison --nocapture
& D:\BuildArtifacts\log2decision\science-venv\Scripts\python.exe tools/science/outlier211.py report <new-evidence-directory>
```

Preparation explicitly admits a bounded whole-JSON exception under a 1 GiB estimated allowance and maps the immutable vectors read-only. The Rust replay uses a 1 GiB budget. Final binary hashes, canonical tests and exact replay commands are recorded in the evidence-root validation report, not inferred from an earlier targeted binary.
