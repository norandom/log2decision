# Low positive share, five review budgets: the Venn policy does not beat score alone

Date: 2026-11-23. Task 2.24. Exploratory development evidence. No pass or fail threshold, no deployment claim, final holdout untouched.

On 1,000 fresh held CSIC records with 100 positives, the Venn policy found fewer positives than score alone at every budget offered. Only the difference at 200 reviews is beyond chance on its own test, and that test is one of five nested looks. The policy sent 53 of the 100 positives to low priority, where only the audit sample can reach them. Unsupported records were positive more often than their score rank predicts, but on this panel "unsupported" nearly coincides with "not on the main endpoint", so the panel cannot say whether the vector evidence or the unfamiliar endpoint carries that signal.

| Reviews offered | Score alone | Policy, shared reservation | Policy, own unsupported slots | Patterns | Policy actually reviewed |
|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 10 | 10 | 3 | 12 |
| 25 | 25 | 20 | 21 | 4 | 25 |
| 50 | 42 | 31 | 31 | 6 | 50 |
| 100 | 56 | 49 | 49 | 12 | 77 |
| 200 | 67 | 51 | 51 | 27 | 94 |

Cells are positives found, of 100. At 100 and 200 the policy left capacity unused: only 61 records were routed outside low priority, and unused reservations are not reassigned.

## How sure the difference is

Positives found by one workflow and not the other, with an exact two-sided sign test. Shared reservation; the separate rule differs only at 12 (6 against 4, p = 0.75) and 25 (14 against 10, p = 0.54).

| Reviews offered | Score only | Policy only | p |
|---:|---:|---:|---:|
| 12 | 5 | 3 | 0.73 |
| 25 | 16 | 11 | 0.44 |
| 50 | 20 | 9 | 0.06 |
| 100 | 19 | 12 | 0.28 |
| 200 | 28 | 12 | 0.02 |

The budgets are nested, so these are not five independent results. As a count, score alone is ahead everywhere. As evidence, the panel shows no advantage for the policy, and an advantage for the score that passes its own uncorrected test only at 200.

At equal effort the picture is closer. At 100 offered the policy reviewed 77 records and found 49 positives; the score's top 77 also hold 49. At 200 offered the policy reviewed 94 and found 51; the score's top 94 hold 56.

The two workflows find different positives. Of the 17 positives off the main endpoint, the policy with its own unsupported slots found 3, 4, 10, 17 and 17 at the five budgets, the policy with the shared reservation 2, 2, 10, 17 and 17, and score alone 5, 5, 5, 7 and 7.

## Where the policy loses positives

| Route | Records | Positive | Share |
|---|---:|---:|---:|
| Investigate (high score, bad neighbours) | 8 | 8 | 100 % |
| Review, uncertain | 6 | 6 | 100 % |
| Review, unsupported | 27 | 22 | 81 % |
| Review, disagreement | 20 | 11 | 55 % |
| Low priority (low score, good neighbours) | 939 | 53 | 5.6 % |
| Panel | 1,000 | 100 | 10 % |

The 61 records outside low priority hold 47 positives. Low priority is cleaner than the panel, 5.6 % against 10 %, but it holds more than half of all positives.

The score's own ranking on this panel: ROC AUC 0.817, average precision 0.625, first negative at rank 33.

## Unsupported records: the planned test, and why it says less than it seems

This question was fixed in the manifest before outcomes were joined. The ranking by score was cut into ten bins of 100. Within each bin that holds both kinds, unsupported records were compared with the positive rate of supported records.

| Score bin (1 = highest) | Supported | Positive | Unsupported | Positive |
|---:|---:|---:|---:|---:|
| 1 | 88 | 44 | 12 | 12 |
| 2 | 100 | 11 | 0 | 0 |
| 3 | 98 | 4 | 2 | 2 |
| 4 | 97 | 4 | 3 | 2 |
| 5 | 100 | 2 | 0 | 0 |
| 6 | 98 | 2 | 2 | 1 |
| 7 | 98 | 6 | 2 | 2 |
| 8 | 99 | 1 | 1 | 1 |
| 9 | 98 | 1 | 2 | 1 |
| 10 | 97 | 3 | 3 | 1 |

**Planned result.** 22 positives among 27 unsupported records, against 6.5 expected from the supported rate in the same bins. One-sided p = 0.0001 from 9,999 permutations of the unsupported flag within bins, seed 224. That is the smallest value this test can return.

**The bins are too coarse at the top.** Ten of the twelve unsupported records in bin 1 sit at ranks 1 to 21, where every record is positive, while the bin as a whole is half positive. With finer bins the expected count rises and the excess shrinks. These finer cuts were made after outcomes were joined:

| Bins | Expected positives | Observed |
|---:|---:|---:|
| 10 (planned) | 6.5 | 22 |
| 20 | 9.0 | 22 |
| 40 | 10.9 | 22 |
| 100 | 11.6 | 22 |

The independent review reported p = 0.0005 for each of the finer cuts, the smallest value its 1,999 permutations can return. An excess remains, about ten positives rather than fifteen.

**The excess is confounded with the endpoint.** Looked at after outcomes were joined:

- Off the main endpoint there are 20 records, 17 of them positive, and 18 of the 20 are unsupported. Unsupported records there are positive at 15 of 18, the same rate as the stratum. Fold zero puts 2,000 of its 2,007 held good records on the main endpoint, so nearly everything elsewhere is an attack.
- On the main endpoint, 7 of 9 unsupported records are positive. All seven sit within the top 21 ranks, where every record is positive. The other two, at ranks 366 and 557, are negative. Here the unsupported flag adds nothing beyond the rank.

So on this panel a rule without any encoder, "the endpoint is rare in the batch under review", would select much of what the unsupported route selects: 18 of the 20 records off the main endpoint are unsupported, and 18 of the 27 unsupported records are off it. The other 9 are on the main endpoint, where the flag adds nothing beyond the rank. Every held endpoint is absent from the history by the fold's construction, so absence from the history alone does not distinguish them.

## What the separate reservation changes

It guarantees that unsupported records are reviewed at small budgets and does little for the yield.

| Reviews offered | Unsupported reviewed, shared | Unsupported reviewed, own slots | Positives, shared | Positives, own slots |
|---:|---:|---:|---:|---:|
| 12 | 0 | 2 | 10 | 10 |
| 25 | 0 | 4 | 20 | 21 |
| 50 | 14 | 14 | 31 | 31 |
| 100 | 27 | 27 | 49 | 49 |
| 200 | 27 | 27 | 51 | 51 |

From 50 reviews on, the general pool reaches the unsupported records under either rule. Score alone reviewed 5, 10, 10, 12 and 12 unsupported records at the five budgets, all positive, and never reached the other 10 unsupported positives.

## Patterns

None of the 16 templates learned from the training marks matched any of the 1,000 records. The pattern arm is therefore the fallback order by text hash, and its yield follows the base rate.

## What follows

- **The policy should not replace the score as the review order.** Nothing here shows an advantage, and its low-priority route holds 53 positives among 939 records.
- **Something finds off-endpoint positives that the score ranks low.** Ten of them sit below rank 200. Whether that takes vectors or only the endpoint name is open. A next comparison has to include the encoder-free rule as a control.
- **Automation stays parked.** The concept's condition for a triager skill with an automated flow was that triage beats score alone. On this panel it does not.

## Limits

- One fold, one panel. 980 of 1,000 records are on `/tienda1/publico/autenticar.jsp`, because nearly all held good records of fold zero are on that endpoint. No across-fold or later-time evidence.
- For the same reason "unsupported" and "off the main endpoint" cannot be separated here.
- The planned ten-bin test overstates the excess of unsupported positives, as shown above.
- The benchmark label drew the two strata by seeded hash. No score was read for the draw, and every selection was frozen before outcomes were joined.
- The product ran once, at budget 12 under the shared rule. The other budgets and the separate rule were recomputed from its evidence and validated by the product, not run end to end.
- The history holds 64 records: 16 marked bad, 16 marked good, 32 unmarked. Thresholds are the engineering settings of the earlier panel, not tuned values. A larger history or other thresholds can change the low-priority route.
- CSIC development material has appeared in earlier experiments.
- Outcomes are benchmark labels, not independent analyst reviews.

## Design

- **Panel.** 100 positive and 900 good records drawn by seeded hash from the 5,275 held fold-zero records not used by the [earlier panel](2026-11-18-triage-workflow-comparison.md). 12 endpoints. History, marks and policy thresholds are the earlier panel's. The scorer was refitted from the same history and marks; its weights file is byte-identical to the earlier one, and its revision differs because the source path is part of it.
- **Policy.** NPU profile f0-s0-mask, radius 0.05, nearest count 3, minimum labelled support 1, corroboration 1, thresholds 0.45 and 0.55, audit seed 5, no calibration. One product run at budget 12 under the shared rule, then a product replay that verified it. Both graphs ran on the NPU.
- **Budgets.** 12, 25, 50, 100 and 200. Unsupported, conflict and audit reservations are each one sixth of the budget, rounded down. The shared rule has no unsupported reservation.
- **Other budgets and the second rule.** Recomputed from the product run's persisted evidence. The restated allocation first reproduced the product's own allocation. The product then validated each of the ten recomputed reports with its own allocation code and refused a tampered one.
- **Order of work.** Manifest, product run, pattern decisions, frozen selections for every budget, then outcomes.
- **Independent review.** A reviewer recomputed every figure from the product report, the pattern decisions and the labels with separate code. The review supplied the finer-bin check, the endpoint confound, the equal-effort comparison and the sign tests. The expected counts, the endpoint split, the equal-effort figures and the sign tests were recomputed again with `tools/science/triage224_posthoc.py` before they entered this report. The p-values of the finer cuts are the reviewer's alone.

## Cost

| Step | Wall time | Counted memory peak (limit 2 GiB) |
|---|---:|---:|
| Policy run, 65,000 encodings on the NPU | 716 s | 1.23 GiB |
| Replay | 817 s | 1.55 GiB |

About 91 and 80 encodings per second of wall time. Each of the 1,000 queries re-encodes the 64 history records. Timings on this laptop vary with power management. The NPU job ran alone.

## Evidence and reproduction

Run directory: `D:\BuildArtifacts\log2decision\task-2.24\run01`. [Manifest](qualification/triage224/manifest.json) · [Frozen selections](qualification/triage224/frozen-decisions.json) · [Comparison](qualification/triage224/comparison.json) · [Policy decisions](qualification/triage224/policy-decisions.json) · [Pattern decisions](qualification/triage224/pattern-decisions.json)

| Identity | Value |
|---|---|
| Panel manifest SHA-256 | dd5d52fdf9cd81010bd1a0ad1e0ba26ed1e34e136f61b4672c50f9528fd07836 |
| Frozen selections SHA-256 | c7577df6ff92438567f6afac2179db9fcec08bfddf18e731df99a3f80debb5d5 |
| Comparison SHA-256 | f6dd497acde7790dfbd341b604d606bdae7b72e3386d6fc0b52c9452e30d5eb8 |
| Persisted policy report SHA-256 | 8e4e4a1dc5e1aaa2204fc275c89f080cf1fbd18425ab65b25e749684dd0e0c44 |
| CLI SHA-256 (built from commit 88a754d) | fec0d638e19b02addf22fbe0baf56868d1404a6012f1d3d0444ee375c4e52d57 |
| MarkSet scorer revision | c04f9199d5bf43e0cee65230f023f719e1967443 |

~~~powershell
$artifacts = 'D:\BuildArtifacts\log2decision'
$run = Join-Path $artifacts 'task-2.24\replay01'
$cli = Join-Path $run 'l2d-fec0d638.exe'
uv run --no-project --python 3.12 python -I -B tools/science/triage224_panel.py "$artifacts\task-2.18\run01" "$artifacts\foreground-2.22\run01" $run
Copy-Item "$artifacts\task-2.24\run01\l2d-fec0d638.exe" $cli
uv run --no-project --python 3.12 python -I -B tools/science/triage224_policy.py $run --cli-sha256 fec0d638e19b02addf22fbe0baf56868d1404a6012f1d3d0444ee375c4e52d57
uv run --no-project --python 3.12 python -I -B tools/science/triage222_patterns.py --run $run --cli $cli
uv run --no-project --python 3.12 python -I -B tools/science/triage224_compare.py freeze $run --cli $cli
uv run --no-project --python 3.12 python -I -B tools/science/triage224_compare.py evaluate $run --run218 "$artifacts\task-2.18\run01"
~~~

Policy IDs include the absolute query path, so a replay in another directory can differ in tie-breaks and audit samples. The pattern catalogue's hash also includes paths. The finer-bin check, the endpoint split and the sign tests are not part of `comparison.json`.
