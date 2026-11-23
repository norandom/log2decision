# Score order with reserved slots for unsupported records: not shown to help, on a panel that could show little

Date: 2026-11-23. Task 2.25. Exploratory development evidence. No pass or fail threshold, no deployment claim, final holdout untouched.

On a second fresh panel of 1,000 held CSIC records with 100 positives, reserving review slots for unsupported records was not shown to improve on score alone. The planned test had almost no power: only three positives lay within the candidate's reach, and it found them. A control that fills the same reservation from the rarest endpoints, with no encoder, was not distinguishable from the candidate. The unsupported flag cannot be separated from the endpoint on this fold. The policy again found fewer positives than score alone.

| Reviews offered | Score alone | Candidate, 1/6 reserved | Candidate, 1/3 reserved | Control, 1/6 | Control, 1/3 | Policy, own unsupported slots |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 12 | 12 | 12 | 12 | 9 |
| 25 | 25 | 25 | 25 | 25 | 25 | 20 |
| 50 | 45 | 45 | 45 | 47 | 48 | 34 |
| 100 | 61 | 61 | 63 | 62 | 63 | 47 |
| 200 | 70 | 72 | 72 | 72 | 72 | 48 |

Cells are positives found, of 100. Score, candidate and control always review the full budget. The policy reviewed 12, 25, 50, 74 and 91 records. The score's first negative is at rank 42, so budgets 12 and 25 cannot separate any of the score-based arms.

## Primary test: candidate against score alone

Positives found by one and not the other, exact two-sided sign test. All ten planned results. The budgets are nested.

| Reviews offered | Reservation | Candidate only | Score only | p |
|---:|---|---:|---:|---:|
| 12, 25, 50 | 1/6 and 1/3 | 0 | 0 | identical selections |
| 100 | 1/6 | 0 | 0 | identical selection |
| 100 | 1/3 | 3 | 1 | 0.63 |
| 200 | 1/6 | 3 | 1 | 0.63 |
| 200 | 1/3 | 3 | 1 | 0.63 |

The candidate found 63 against 61 and 72 against 70 where it differed. That is within chance, and it is one comparison seen three times: the same eight records swapped in, three of them positive, for eight displaced records, one of them positive each time.

**The test could not have shown a gain.** The reservation takes unsupported records in score order, and the highest-scoring ones are already inside the score's own top of the list. Score alone reviewed 5, 11, 16, 19 and 19 unsupported records at the five budgets, against reservations of 2, 4, 8, 16 and 33 at one sixth and 4, 8, 16, 33 and 66 at one third. The reservation changes the selection only once it is larger than what the score already includes. Only eight unsupported records lie below rank 77, at 226, 373, 636, 770, 860, 997, 999 and 1,000, and three of them are positive. The largest possible gain was therefore three positives. Three gained and none lost would give p = 0.25. No outcome on this panel could have reached a conventional level.

This is a property of the arm as planned. It should have been seen when the plan was written, and the task 2.24 panel already showed it. The power figures were worked out after outcomes were joined.

## Control test: candidate against the rule without an encoder

| Reviews offered | Reservation | Candidate only | Control only | p |
|---:|---|---:|---:|---:|
| 12 | 1/6 | 2 | 2 | 1.00 |
| 12 | 1/3 | 4 | 4 | 1.00 |
| 25 | 1/6 | 4 | 4 | 1.00 |
| 25 | 1/3 | 8 | 8 | 1.00 |
| 50 | 1/6 | 1 | 3 | 0.63 |
| 50 | 1/3 | 2 | 5 | 0.45 |
| 100 | 1/6 | 1 | 2 | 1.00 |
| 100 | 1/3 | 0 | 0 | same positives |
| 200 | 1/6 and 1/3 | 0 | 0 | same positives |

The two never differ beyond chance. Where the counts differ, the control has the higher one. As planned for this outcome: the vector evidence is not shown to add anything over the endpoint on this fold.

This test has little power too. At large reservations the two hold largely the same records, 23 of 27 at one third of 100 and at one sixth of 200, because 22 of the 27 unsupported records are off the main endpoint. At 50 the reservations overlap little, 0 of 8 and 7 of 16, and the low power comes from the small counts.

The control against score alone was not a planned test: three gained and one lost at 50 with one sixth (p = 0.63), five and two with one third (p = 0.45).

## Replication of the unsupported question

Planned with 40 score bins, 9,999 permutations within bins, seed 225.

| Population | Unsupported | Positive | Expected from supported records in the same bins | One-sided p |
|---|---:|---:|---:|---:|
| Whole panel | 27 | 22 | 15.9 | 0.0002 |
| Main endpoint only | 5 | 4 | 4.0 | 1.00 |
| Other endpoints, counts only | 22 | 18 | too few supported records for bins | |

- **Whole panel.** The excess replicates, about six positives.
- **Main endpoint.** The test is uninformative. Four of the five unsupported records sit in the first bin, where all twenty supported records are positive, so they can never count as an excess. The fifth is a negative at rank 750 on that endpoint. With these five records the test returns p = 1.0 whatever the vectors are worth. This was seen after outcomes were joined.
- **Other endpoints.** 24 records, 20 of them positive. 22 of the 24 are unsupported, and the two supported ones are both positive.

All of the excess on the whole panel sits off the main endpoint. There nearly every record is unsupported and nearly every record is positive, so the flag and the endpoint cannot be told apart. The panel does not show that the vectors carry the signal, and it does not show that they carry none.

## The policy against score alone

These comparisons were not among the planned tests. They are nested and uncorrected.

| Reviews offered | Policy only | Score only | p | Policy reviewed |
|---:|---:|---:|---:|---:|
| 12 | 3 | 6 | 0.51 | 12 |
| 25 | 8 | 13 | 0.38 | 25 |
| 50 | 9 | 20 | 0.06 | 50 |
| 100 | 4 | 18 | 0.004 | 74 |
| 200 | 5 | 27 | 0.0001 | 91 |

The last two rows compare 74 and 91 policy reviews with 100 and 200 score reviews. At equal effort, computed after outcomes were joined:

| Records reviewed by both | Policy | Score's top that many | Policy only | Score only | p |
|---:|---:|---:|---:|---:|---:|
| 74 | 47 | 54 | 7 | 14 | 0.19 |
| 91 | 48 | 60 | 5 | 17 | 0.02 |

On the task 2.24 panel the equal-effort comparison at 77 reviews was a tie. Across both panels the policy is never ahead, and at equal effort it is behind beyond chance on one post-hoc look.

Two things hold the policy back, and neither is the encoder as such. It routes 942 records to low priority, 54 of them positive. And it spends 2 to 33 slots on the low-priority audit, which exists to check that route and not to find positives.

| Route | Records | Positive |
|---|---:|---:|
| Investigate | 11 | 11 |
| Review, uncertain | 4 | 4 |
| Review, unsupported | 27 | 22 |
| Review, disagreement | 16 | 9 |
| Low priority | 942 | 54 |

The score's ranking on this panel: ROC AUC 0.852, average precision 0.670, first negative at rank 42.

## What did not replicate from task 2.24

Task 2.24 motivated the candidate with ten unsupported positives that score alone never reached. Here score alone missed three unsupported positives at 100 and 200 reviews, and it found 17 of the 20 positives off the main endpoint by 100 reviews, against 7 of 17 in task 2.24. The gap the candidate was meant to close was much smaller on this panel.

## What follows

- **Score alone stays the review order.** On two panels the policy is never ahead of it.
- **A reservation for unsupported records is not shown to help.** This panel could not have shown it, so the question is open, not closed.
- **Whether the vectors add to ranking is undecided on this fold,** because unsupported records and off-endpoint records are nearly the same set. The rare-endpoint rule profits from the same peculiarity and is not a finding either.
- **Nothing here supports spending the NPU on ranking.** That agrees with the owner's earlier decision that the score classifies and the encoder serves search and grouping. It does not test the encoder beyond the unsupported flag.
- **Fold zero is used up for panels of this shape.** 180 fresh good records remain. Evidence on ranking needs data where the endpoint and the outcome are not tied, such as a second dataset or the protected cohort of task 2.22.
- **Automation stays parked.**

## Limits

- One fold. 976 of 1,000 records are on `/tienda1/publico/autenticar.jsp`. The other 13 endpoints hold 24 records, 20 of them positive. No across-fold or later-time evidence.
- This panel shares fold, history, scorer, thresholds and endpoints with task 2.24. It is a second draw, not an independent replication.
- The primary test could gain at most three positives, and the main-endpoint replication could not register an excess. Both are stated above.
- The candidate as planned could differ from score alone only at large reservations. A reservation aimed at low-scoring unsupported records would be a different arm and was not planned.
- The control needs the endpoint counts of the whole batch before it can order anything.
- The policy tests, the control against score alone and all equal-effort figures were not planned.
- The benchmark label drew the two strata by seeded hash. No score was read for the draw, and every selection was frozen before outcomes were joined.
- The product ran once, at budget 12 under the shared rule. The policy arm at each budget was recomputed from its evidence and validated by the product, not run end to end.
- The history holds 64 records and the thresholds are untuned engineering settings, as in task 2.24.
- CSIC development material has appeared in earlier experiments. Outcomes are benchmark labels, not analyst reviews.

## Design

- **Panel.** 100 positive and 900 good records drawn by seeded hash from the held fold-zero records used by neither the [first](2026-11-18-triage-workflow-comparison.md) nor the [second panel](2026-11-23-triage-low-share-comparison.md). 14 endpoints. 1,080 fresh good records existed before the draw. History, marks, thresholds and NPU profile are unchanged. The scorer's weights file is byte-identical to the earlier ones.
- **Arms.** Score alone takes the first *b* records in score order. The candidate reserves *n* slots for records the product routed as unsupported, in score order, and fills the rest in score order. The control fills the reservation from the endpoints with the fewest panel records, then score order. *n* is one sixth or one third of the budget, rounded down. The policy arm is the product policy with its own unsupported slots.
- **Order of work.** Manifest with arms and tests, product run and verified replay, frozen selections for every budget, then outcomes. The evaluation script was saved 39 seconds after the manifest, while the product run was starting, and 27 minutes before outcomes were joined; its hash is not in the manifest. The product validated the five recomputed policy reports and refused a tampered one with "inconsistent saved allocations or deferred count".
- **Independent review.** A reviewer recomputed every figure with separate code and confirmed the plan, the order of work and that the panel is disjoint from both earlier ones. The review rejected the first draft for reading the main-endpoint test as a negative finding, for wording an underpowered test as "no gain" and for policy comparisons without uncertainty. The power limits and the equal-effort tests come from that review and were recomputed with `tools/science/triage225_posthoc.py`.

## Cost

| Step | Wall time | Counted memory peak (limit 2 GiB) |
|---|---:|---:|
| Policy run, 65,000 encodings on the NPU | 820 s | 1.23 GiB |
| Replay, a second run of the same work | 807 s | 1.55 GiB |

About 79 and 81 encodings per second of wall time. The candidate needs the policy run, which encodes every record together with the history; score alone and the control need no encoding. Timings on this laptop vary with power management. The two NPU runs were sequential and nothing else used the NPU.

## Evidence and reproduction

Run directory: `D:\BuildArtifacts\log2decision\task-2.25\run01`. [Manifest](qualification/triage225/manifest.json) · [Frozen selections](qualification/triage225/frozen-decisions.json) · [Comparison](qualification/triage225/comparison.json) · [Policy decisions](qualification/triage225/policy-decisions.json)

| Identity | Value |
|---|---|
| Panel manifest SHA-256 | 7ebf3f5645dd59584acae1fa5be2ddf7aa68b6405520cb30977d7e61e38fb4bb |
| Frozen selections SHA-256 | 154eed329bb42b5f4729078c93cb2da8257ea5ca17fee68fbf007eee9e5d507f |
| Comparison SHA-256 | 4bb3c45f9da4185cfa21d572735bad5b3c447a904c7409ef158c4f6f9f07e42b |
| Persisted policy report SHA-256 | 360da8e3768d46d8f40e1e6c74c99a78dbd20c52f183904b779c03e7de2ebcab |
| CLI SHA-256 (built from commit 88a754d) | fec0d638e19b02addf22fbe0baf56868d1404a6012f1d3d0444ee375c4e52d57 |
| MarkSet scorer revision | fb16fcd53b6c9a74615f9e8461f4db7b295a482b |

~~~powershell
$artifacts = 'D:\BuildArtifacts\log2decision'
$run = Join-Path $artifacts 'task-2.25\replay01'
$cli = Join-Path $run 'l2d-fec0d638.exe'
uv run --no-project --python 3.12 python -I -B tools/science/triage225_panel.py "$artifacts\task-2.18\run01" "$artifacts\foreground-2.22\run01" "$artifacts\task-2.24\run01" $run
Copy-Item "$artifacts\task-2.25\run01\l2d-fec0d638.exe" $cli
uv run --no-project --python 3.12 python -I -B tools/science/triage224_policy.py $run --cli-sha256 fec0d638e19b02addf22fbe0baf56868d1404a6012f1d3d0444ee375c4e52d57
uv run --no-project --python 3.12 python -I -B tools/science/triage225_compare.py freeze $run --cli $cli
uv run --no-project --python 3.12 python -I -B tools/science/triage225_compare.py evaluate $run --run218 "$artifacts\task-2.18\run01"
uv run --no-project --python 3.12 python -I -B tools/science/triage225_posthoc.py $run "$artifacts\task-2.18\run01"
~~~

Policy IDs include the absolute query path, so a replay in another directory can differ in tie-breaks and audit samples. The power figures, the equal-effort comparison, the unsupported rank list and the off-endpoint counts for score alone are not part of `comparison.json`.
