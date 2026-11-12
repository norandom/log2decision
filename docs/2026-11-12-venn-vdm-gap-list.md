# Venn's operations against the log2decision spec

Date: 2026-11-12. Purpose: decide how far the Vector Decision Model part of
`l2d` should go once the NPU encoder delivers vectors. Nothing in the spec is
changed by this document.

## Summary

- The spec already covers the three Venn operations that fit log triage:
  a ratio from the nearest labelled logs with abstention, the evidence shown
  with it, and the population-shift test.
- Three Venn operations are missing and would fit: a fitted linear head on the
  vectors, a ranking of outliers, and two more evidence figures.
- Four are missing and should wait or be left out: the trained residual head,
  heads compiled from an outcome store, numeric outcomes, and the generic
  text head.
- Owner's steer of 2026-11-12: the NPU encoder comes first, and the baseline
  score and the vector decision model are brought together later, at the point
  where the decision to investigate is made. They are not merged now.

## What was read

Seven pages of https://markrussinovich.github.io/venn-vdm/ on 2026-11-12: home,
concepts, API reference, outcome stores, task recipes, limitations, model
card.

Not read: getting started, installation, adapters, design, performance,
benchmark, evidence, reproduction, red-team, licences, examples, and the
source repository's README. The gaps this leaves are named where they matter.

## What Venn is, in its own terms

- A **reader** (a frozen embedding model; Venn 1 uses EmbeddingGemma 2, 768
  dimensions, normalised) encodes content once.
- A small **decision layer** answers typed questions over the vectors.
- A **profile** pins reader, revision, dimension and prompt. "Vectors are only
  comparable within one profile."
- An **outcome store** holds past cases with numerical outcomes.
- A **family** is a custom predictor registered with the service: a
  classifier, a numeric predictor, or a head compiled from an outcome store.
- Three routes to a decision: described options (the generic head), historical
  cases (the outcome store), and fitted families.
- Maturity: "Venn is a research preview." The outcome API is "experimental."

## Operation by operation

| Venn operation | What it does | In the spec | Verdict |
| --- | --- | --- | --- |
| `nearest_outcome` | Mean outcome of the k nearest past cases per option; filters before top-k; an option below `min_support` is unsupported; answer `none` if no option qualifies; evidence ids | Task 2.8 (neighbour scorer: bad share among neighbours within reach, abstain below minimum support, evidence, same-group against cross-group counts) on the search of task 1.9 | Covered for good/bad labels |
| `nearest_outcome`, evidence figures | Per option: mean, count, spread, mean distance | Evidence has labels, distances and counts | Small gap: spread and mean distance |
| `nearest_reference` | Label of the closest past case, with a margin | Search of task 1.9 returns nearest logs with distances | Covered by the search; no separate scorer needed |
| `population_shift` | Permutation p-value between two vector populations; "not a posterior" | Task 3.1 (kernel two-sample test, seeded permutations, refusal below 30 logs, p-value described as a test result) | Covered. Venn's kernel and permutation count are not in the pages read |
| Profiles and tags | Vectors of different readers must not be mixed | Encoder identity on packages and stores; mismatch refused (requirement 5.6) | Covered |
| Grouped cross-validation (`group_id`) | Keeps a group on one side of every split | Grouped splits by endpoint (requirement 9.2) | Covered |
| Abstention routed to review | "Route a no-match answer to review"; "nothing in an answer authorizes an action by itself" | Abstained list in triage; the tool ranks and never acts (requirement 6.9) | Covered |
| Calibration | Softmax temperature; a `calibrated` flag needs 500 report items and a bound on calibration error | Conformal label sets at a stated error rate (requirement 11) | Covered by a different method that makes a stronger statement |
| Linear family | Multinomial logistic regression on single vectors; penalty chosen by 5-fold cross-validation; L-BFGS; no abstention | Not in any task | **Gap, cheap**: the regression of task 1.6 already takes dense input |
| `set_membership` | Outliers of a set: a centroid outlier, or a ranking by distance to the fifth nearest neighbour; no probabilities | Not in any task. The conformal set "neither" says "unlike the calibration logs" for scores, not for vectors | **Gap, cheap**: a ranking of logs unlike anything stored, on the search of task 1.9 |
| `index_apply` | One prepared question applied to every stored vector | Triage over a set of logs | Covered in effect |
| Residual family | Trained head: rank-16 bilinear scorer below 1,000 examples, otherwise a small network; can answer "none" | Not in any task | Gap, costly: needs the training crate |
| Outcome-compiled head | A head trained from an outcome store (at least 320 cases), served without lookup; imitation or advantage-weighted | Not in any task | Gap, costly, and needs outcomes that are utilities, which the tool does not have |
| Numeric family and numeric outcomes | Ridge or kernel ridge with an interval; outcomes as utilities, goal maximise or minimise, exploration | Not in any task; labels are good, bad, unknown | Not needed for a bad/good ratio |
| Generic head (`Choice`, `Score`, `Noul`) | Answers questions described in text, from a text embedding with a prompt | Not in any task | Does not apply: the log encoder is a byte encoder, not a text reader with prompts |

Venn's documentation, as read, gives no numeric rule for when to act, defer or
escalate. It gives the abstention signal and leaves the policy to the
application. That matches the owner's steer: the decision to investigate is
where score and decision model meet, and it is ours to define.

## Proposal

After the NPU encoder (order unchanged):

1. **As specified:** neighbour scorer with evidence and abstention (2.8),
   calibration on top of it (already built for any scorer), shift test (3.1).
2. **Add by amendment, small:**
   - a fitted linear head on the vectors, as a third scorer, using the
     regression that exists, with the penalty chosen by grouped
     cross-validation, evaluated side by side with the baseline and the
     neighbour scorer in milestone 3;
   - an outlier ranking (distance to the k-th nearest stored log) as a signal
     for "unlike anything stored";
   - spread and mean distance in the evidence.
3. **Later, as its own piece of work:** the decision to investigate, which
   brings together the baseline ratio, the vector scorers, the label sets and
   the outlier signal. This is where "score and decision model align".
4. **Not planned:** the residual and outcome-compiled heads until a trained
   encoder exists and the linear head has shown what a head adds; numeric
   outcomes; the generic text head.

## Decided on 2026-11-12

The owner chose point 2 (the three small additions) and confirmed point 3 as
the reading of "score and decision model align later". The amendment draft is
`.kiro/specs/npu-log-decision/amendment-2026-11-12-vector-decision-heads.md`.
It replaces "spread" by per-label counts, because the spread of good/bad labels
follows from the ratio.

## Open

- Venn's defaults for `k`, `min_support`, the shift kernel and the permutation
  count were not in the pages read. The spec has its own (199 permutations,
  Gaussian kernel with the median bandwidth).
- The benchmark, evidence and design pages may change the verdict on the
  residual head. They were not read.
