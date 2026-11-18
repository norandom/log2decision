# Three review workflows on held CSIC endpoints

With 12 reviews, score alone found 12 benchmark positives, the Venn policy found 11, and the pattern workflow found 8. The score ranking reached the maximum possible yield at this budget. The policy recovered three positives that score alone missed. Four positives selected by score alone fell outside the policy allocation.

| Workflow | Reviewed | Positives found (of 69) | Precision | Recall | Positives missed | Found beyond the seed |
|---|---:|---:|---:|---:|---:|---:|
| MarkSet score | 12 | 12 | 100.0% | 17.4% | 57 | 6 |
| Training patterns and deterministic fallback | 12 | 8 | 66.7% | 11.6% | 61 | 6 |
| Score with NPU evidence and Venn allocation | 12 | 11 | 91.7% | 15.9% | 58 | 5 |

**The pattern workflow’s eight hits came entirely from the fallback.** Native discovery on 16 marked-bad training originals produced 16 singleton templates. Each generated regex matched one marked-bad training original and zero marked-good originals. All 16 missed every held original. Under the frozen rule, ties went to the lowest original-text SHA-256. This tests automatic template transfer before analyst refinement. The contribution of analyst-edited masks, merges and query rules remains unmeasured.

**The low-priority queue contained 34 positives among 61 records.** Its two audit slots found one positive. All 17 unsupported records were benchmark positives and all 17 were deferred. The two slots shared by unknown and conflicting cases went to disagreements. The eight general slots went to corroborated high-score records, all positive. A shared reservation can therefore fill while leaving every unsupported record unreviewed. The score-only diagnostic ratio was available on all 96 records. Its ROC AUC was 0.813 and its average precision was 0.935, against a panel positivity rate of 0.719. The policy produced actionable scores on 79/96 records (82.3%).

## Comparison design

- A salted hash selected 96 distinct originals from historical CSIC fold-zero held endpoints without using outcomes or scores. Fold zero matches the product’s pinned encoder. All 96 were outside encoder training and its byte, line and threshold calibration sets.
- A separate 64-original history from that fold’s training partition supplied 16 seed-matched benchmark-bad originals, 16 benchmark-good originals and 32 unmarked originals. The benchmark-labelled training groups simulate prior analyst review. The unmarked originals supplied background context. Score and policy used one newly fitted, unredacted MarkSet model with identical diagnostic ratios on all queries. The earlier five-fold comparison used a different scorer fitted against assumed background.
- The pattern procedure used only training records to derive templates. Matching queries took priority by their best template’s training bad fraction and bad support. The original-text digest broke ties. The catalogue and generated regexes were frozen before application to held records.
- The product policy used NPU profile f0-s0-mask with radius 0.05, nearest count 3, minimum labelled support 1, corroboration 1 and low/high score thresholds 0.45/0.55. These engineering settings were fixed before reading panel outcomes. The diagnostic budget of 12 reserved two unknown/conflict and two audit slots, with audit seed 5. Calibration was absent. Both graphs executed on the NPU in the original run and replay.
- All three selections and the full score and pattern orderings were frozen before joining the evaluation labels. Offered capacity and actual review count both equalled 12 for every arm. Each unreviewed positive counts once, including deferred unsupported positives. The comparison artifact lists individual discordant positive IDs.

One endpoint supplied 84 of 96 queries and 57 of 69 positives. On those 84 records, the score, pattern and policy queues recovered 9, 6 and 8 positives respectively. The remaining 12 records were all positive. Training and query endpoint sets were disjoint. The uneven group sizes and single fold limit inference about transfer. Across-fold variation and later-time performance remain unmeasured.

## Evidence and reproduction

The original run directory is `D:\BuildArtifacts\log2decision\foreground-2.22\run01`. Its training catalogue, CLI output, scorer and NPU receipts bind the selections to this run. The live policy replay verified the persisted report. An independent baseline command reproduced every diagnostic ratio bit-for-bit. Independent review also reconstructed the training and calibration populations and reproduced the paired outcomes from the original benchmark labels. [Panel manifest](qualification/triage222/manifest.json) · [Frozen selections and rankings](qualification/triage222/frozen-decisions.json) · [Retrieval metrics, endpoint breakdown and paired differences](qualification/triage222/comparison.json)

| Identity | Value |
|---|---|
| Panel manifest SHA-256 | e098d4df5ef4ea623d37789c3347fa4b22845b431ec1e0b0100a7dcb3c43a71e |
| Frozen selections SHA-256 (original file) | 3d7c190bfc919403ccbf2694d4090805fcd440695601291354010705fdcbc0c4 |
| MarkSet scorer revision | 7a2496ab449ca0836281982f89658f5172c31444 |
| Frozen native pattern catalogue SHA-256 | 280d6e045eaaeb16181377a2184dc52f073593f7e6bdb31a6631d387f468a3f1 |
| Persisted policy report SHA-256 | b2dde5561c71c1d3551f0c34c857ea8d1b418121d997475befecb7a9b9e6344c |

A lighter replay needs the original 2.18 development export and the two retained decision files. Run the panel preparation below, copy both decision files into the new run directory, then run the freeze and evaluate commands. On the workstation this reproduced the manifest, frozen selections and comparison byte-for-byte. [Per-record pattern decisions](qualification/triage222/pattern-decisions.json) · [Per-record policy scores and allocations](qualification/triage222/policy-decisions.json)

These commands exercise the procedure from a fresh artifact directory using the pinned workstation CLI and NPU profile. Policy IDs incorporate the absolute query path, so moving the CSV can change tie-breaks and audit samples. To reproduce the original policy selections, keep the original input path and direct new receipts elsewhere using the policy runner’s `--output` option. Pattern provenance includes paths and the runner script’s file hash. Freeze and evaluate each replay together even when its pattern ordering matches the original.

~~~powershell
$artifacts = 'D:\BuildArtifacts\log2decision'
$py = Join-Path $artifacts 'science-venv\Scripts\python.exe'
$cli = Join-Path $artifacts 'task-1.26-target\debug\l2d.exe'
$source = Join-Path $artifacts 'task-2.18\run01'
$run = Join-Path $artifacts 'foreground-2.22\replay01'
& $py -B tools/science/triage222_panel.py $source $run
& $py -B tools/science/triage222_policy.py $run
& $py -B tools/science/triage222_patterns.py --run $run --cli $cli
& $py -B tools/science/triage222_compare.py freeze $run
& $py -B tools/science/triage222_compare.py evaluate $run --run218 $source
~~~

Task 2.22 now has a paired development comparison. Final qualification still requires a separately protected cohort, independent good/bad reviews beyond the seed, and operational capacity, missed-case and allocation targets set before evaluation. The CSIC development material had appeared in previous experiments. Task 2.21 remains dependent on 2.22.
