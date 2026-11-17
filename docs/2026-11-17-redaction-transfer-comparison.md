# Exploratory redaction transfer comparison — 17 November 2026

**Confirmed host/path redactions did not demonstrate agnostic transfer of the MarkSet score.** Across the five CSIC endpoint folds, the baseline's mean held-endpoint ROC AUC was **0.6163 without redaction and 0.6158 with it**. Two redacted folds exceeded the diagnostic 0.05 fitted-minus-held margin, versus one without redaction. The small or negative mean gap does not establish usefulness: both sides rank at only about 0.6 AUC.

The frozen NPU vectors' diagnostic head improved mean held AUC from 0.6643 to 0.6737, but its gap change was uncertain and one endpoint fold remained weak. This experiment does not introduce a production vector classifier. The baseline still classifies; vectors serve search and grouping.

**Recommendation:** retain that separation, and do not treat these redactions or an average gap below 0.05 as qualification. Continue the planned conservative evidence/support work. Before final policy qualification, obtain independently reviewed examples beyond the seed search, an untouched report cohort and owner-specified review capacity/missed-case targets. A future preregistered budget curve would be more discriminating than the single diagnostic budget here. No settings were selected from these results.

## Status and scope

This is the owner's exploratory-first task 2.18 comparison, not deployment qualification. CSIC was repeatedly examined in earlier gates. **A truly untouched final dataset is not allocated/available in this experiment; no final holdout was examined.** Operational targets remain unset. The fixed budget of **100 distinct records per fold and report partition**, shared by all four arms, is a diagnostic setting only.

CSIC's supplied `classification` labels are independent of the regex and include 15,964 bad and 9,644 good distinct texts. They are benchmark annotations, **not newly human-reviewed examples**. The seed matches 538 bad and zero good texts; the other 15,426 positives remain in evaluation. Fitting uses regex marks against assumed unmarked background, not these evaluation labels. Many actual attacks therefore belong to the assumed comparison class. This is a limitation of the fitted question, not evidence that unmarked means good. Source duplicate-label conflicts or unknown classification values refuse preparation; none occurred here.

Pattern-assisted review and the finished score-plus-vector-evidence policy were not measured: those policies are not delivered by this measurement. Requirement 19.5's final workflow comparison, independent review under 19.1, and final qualification under 19.2/19.6 remain incomplete. There are no suitable event timestamps for a later-time split; temporal generalization is **untested**.

## Frozen experiment

The source is the same 29,539,583-byte CSV as task 2.17: 61,065 occurrences, 25,608 distinct reconstructed log texts. Preparation checks a 64 MiB source-size ceiling and streams its SHA-256 against the historical manifest before reading the bounded corpus. A 4 GiB Windows process-commit limit covers this whole-object preparation exception; large arbitrary datasets are not admitted by this experiment.

The historical five URL-path folds are reused exactly. For each fold, the other four endpoint folds supply training: texts whose SHA-256 first 32 bits modulo five equal zero instead become the fitted-endpoint report partition. Exact original duplicates always share identity/partition; all repeated source occurrences were collapsed before splitting. Held endpoints never occur in that fold's scorer/head training.

| Fold | MarkSet/head training | Fitted-endpoint report | Held-endpoint report | Training regex marks |
|---|---:|---:|---:|---:|
| 0 | 16,254 | 3,983 | 5,371 | 334 |
| 1 | 16,336 | 4,001 | 5,271 | 338 |
| 2 | 16,325 | 4,019 | 5,264 | 324 |
| 3 | 16,523 | 4,160 | 4,925 | 358 |
| 4 | 16,698 | 4,133 | 4,777 | 338 |

The seed was frozen before scoring:

```regex
(?i)(union(\s|\+|%20)+select|drop(\s|\+|%20)+table|<script|%3cscript)
```

The actual product APIs create the MarkSet, propose redactions and fit both baseline scorers. Proposal settings are the frozen product defaults: at least 5% of marked and 1% of unmarked logs, with a floor of two on each side. The simulated confirmation policy accepts only proposed `localhost:8080` or an exact training URL path: five literal rules per fold, with counts and rejected proposals retained. It is a documented simulation of analyst confirmation, not a claim of human review. No held-endpoint text or evaluation label selects a rule. A literal path learned from training is not a general rule that strips an unseen path.

Baseline settings are the product defaults: 65,536 hash buckets, byte n-grams through length three, routing kept, L2=1, maximum 1,000 iterations, tolerance 0.001. All ten fits converged. The paired inputs and all ten saved scorer revisions are frozen on disk. The report also verifies the Python regex membership used for diagnostic heads against every product MarkSet position, not just its count.

These are not directly comparable to the earlier supervised-label classifier figures in 2.17: the fitting labels here are narrow regex marks versus assumed background. The lower absolute quality is not a measured regression of the earlier supervised scorer.

Vectors use the original **fold-specific `f0`–`f4`, seed-zero, mask-objective** 2.17 weights, byte tables, quantization and training-calibrated patch thresholds. Their graph hashes match the historical records. Training/calibration indices are reconstructed from the original deterministic code and checked against each held fold. No encoder is retrained or recalibrated. The production adapter's f0-only pins are unchanged; using that one encoder across all five folds would have leaked endpoint training, so the scientific harness runs the five verified graph pairs separately.

There is an explicit asymmetry: the frozen unsupervised encoders had already seen the fitted-endpoint report texts during 2.17 pretraining. The baseline and diagnostic head do not fit those report records, but this is **not a fully disjoint encoder fitted-side evaluation**. The held-endpoint side was excluded from that encoder's training. Fitted-report records recur in four fold contexts, so their pooled observations are not independent. This is development evidence, not a new clean holdout.

Each vector arm fits only a diagnostic L2 logistic head on normalized NPU vectors and the same regex/assumed-background labels: C=1, maximum 1,000 iterations, fixed seed 20261117, no model/penalty search. Coefficients, training identities and settings are persisted before scoring. Raw and redacted arms use the same frozen encoder and paired membership. Empty or original UTF-8 lengths over 1,024 bytes are unsupported even if redaction would shorten them; neither truncation nor partial vectors are allowed. This CSIC cohort has no such unsupported records and no invalid vector pairs: coverage is 100%, abstentions/unsupported positives are zero in all arms.

## Main figures

Values are mean ± sample SD across five folds. Gap means fitted-report AUC minus held-endpoint AUC; only a positive gap above 0.05 triggers the diagnostic dependence warning. Negative gaps do not establish agnosticism.

| Arm | Fitted ROC AUC | Held ROC AUC | Gap | Folds with gap >0.05 | Held AP |
|---|---:|---:|---:|---:|---:|
| Baseline, raw | 0.5942 ± 0.0203 | 0.6163 ± 0.0611 | −0.0221 ± 0.0805 | 1/5 | 0.7789 ± 0.0497 |
| Baseline, redacted | 0.6118 ± 0.0267 | 0.6158 ± 0.0648 | −0.0040 ± 0.0901 | 2/5 | 0.7774 ± 0.0508 |
| NPU vectors + diagnostic head, raw | 0.6749 ± 0.0309 | 0.6643 ± 0.0782 | 0.0106 ± 0.0771 | 1/5 | 0.8044 ± 0.0617 |
| NPU vectors + diagnostic head, redacted | 0.6548 ± 0.0344 | 0.6737 ± 0.0910 | −0.0190 ± 0.0696 | 1/5 | 0.8061 ± 0.0685 |

AP is average precision, the step-integral definition of PR area. All main partitions contain both classes, so no main ROC/AP metric is undefined. The reporter explicitly returns null with a reason for single-class/empty supported partitions.

The baseline's redacted-minus-raw gap change is **+0.0181**, SD 0.0117, paired-fold bootstrap interval **[0.0087, 0.0262]**: fitted performance improves while held performance does not. The vector diagnostic change is **−0.0296**, SD 0.0536, interval **[−0.0753, 0.0100]**. Its apparent mean gap reduction includes a 0.0201 decline on the fitted side and only a 0.0094 rise on the held side. These intervals use 10,000 fixed-seed paired fold resamples; five folds with overlapping training sets provide descriptive uncertainty, not an independent qualification confidence statement. Encoder-seed variation is not estimated.

| Fold | Baseline raw held AUC / gap | Baseline redacted held AUC / gap | Vector diagnostic raw held AUC / gap | Vector diagnostic redacted held AUC / gap |
|---|---:|---:|---:|---:|
| 0 | 0.6829 / −0.1007 | 0.6956 / −0.1011 | 0.7433 / −0.0992 | 0.7804 / −0.0976 |
| 1 | 0.5616 / 0.0489 | 0.5569 / 0.0795 | 0.6599 / 0.0377 | 0.6760 / −0.0261 |
| 2 | 0.5458 / 0.0750 | 0.5449 / 0.0987 | 0.6557 / 0.0497 | 0.6813 / −0.0549 |
| 3 | 0.6655 / −0.0920 | 0.6594 / −0.0763 | 0.5420 / 0.0976 | 0.5291 / 0.0889 |
| 4 | 0.6259 / −0.0417 | 0.6222 / −0.0207 | 0.7205 / −0.0327 | 0.7018 / −0.0051 |

| Arm | Pooled fitted ROC / AP | Pooled held ROC / AP |
|---|---:|---:|
| Baseline, raw | 0.5903 / 0.7407 | 0.5765 / 0.7320 |
| Baseline, redacted | 0.6068 / 0.7439 | 0.5764 / 0.7297 |
| Vector diagnostic, raw | 0.6706 / 0.7943 | 0.6200 / 0.7634 |
| Vector diagnostic, redacted | 0.6528 / 0.7831 | 0.6063 / 0.7393 |

Pooled held records are distinct, but their scores come from five different fitted models. Their cross-fold scales are not calibrated as one production policy. The redacted vector diagnostic's pooled ranking is worse despite its better mean fold AUC; the two summaries must not be substituted for one another.

### Beyond the seed and duplicate sensitivity

Removing all seed-query matches from evaluation gives held AUC means of **0.6029 → 0.6024** for the baseline and **0.6528 → 0.6627** for the vector diagnostic. Thus the modest absolute quality is not hidden by treating the query's own matches as independent truth.

Redaction creates exact transformed-text collisions with training in 71 held records (12/13/17/18/11 by fold) and 47 fitted-report contexts (12/6/10/10/9). Main figures retain the frozen original-identity cohort and disclose this non-disjoint transformed representation. The sensitivity removes the **union of raw/redacted collisions from both arms**, preserving paired comparison: held AUC means become **0.6191 → 0.6186** for the baseline and **0.6664 → 0.6756** for the vector diagnostic. The conclusion is unchanged. These collisions cannot be treated as independent redaction-transfer evidence. Per-fold sensitivity ROC/AP, budgets, counts and exclusions are in `report.json`.

### Fixed review budget and missed cases

Every arm achieves precision **1.000** at 100 reviewed held records per fold. Mean recall is **3.1378% ± 0.1523 percentage points**. The five held partitions contain 3,364/3,264/3,257/2,980/3,099 positives; missed positives are consequently 3,264/3,164/3,157/2,880/2,999, totaling **15,464 of 15,964**. Unsupported positives are zero and are included in the reporter's missed-case accounting when present.

This roughly 96.9% unreviewed positive fraction is principally the **budget ceiling**: with precision already one, 100 reviews cannot find more than 100 of several thousand positives. It is not a model-quality failure by itself, and this budget cannot discriminate the four workflows. ROC/AP, endpoint spread and beyond-seed results carry the transfer comparison. No larger budget was selected after inspecting results. Fitted-report precision is also one and mean recall 3.9528% ± 0.1009 points. The report additionally provides a clearly labeled pooled 500-record ranking; it does not enforce per-fold quotas.

## NPU and resource evidence

All **256,080** paired vectors were newly encoded on the NPU: 51,216 per frozen fold profile. Runtime is Ryzen AI 1.9.0, ONNX Runtime 1.30.0, driver 32.0.20102.4480. Both graph placement reports and first-run execution profiles passed the existing strict gate: convolutional work on NPU, only allowed boundary QuantizeLinear/DequantizeLinear operations on the host, no CPU encoder fallback. Entropy cuts and pooling are host arithmetic. Scientific 64-row batches reuse the stable graph layout; this is not a claim about production throughput.

For each fold the first 64 rows were bit-identical batched versus individually encoded, with **64 distinct texts and 64 distinct vectors**, so this check did not pass vacuously through collapse. All 25,608 raw f0 vectors are bit-identical to the prior actual NPU run. Existing CPU-reference vectors were read for diagnostics only; no CPU encoder was executed:

| Fold | Raw exact CPU-reference matches /25,608 | Minimum cosine | 1st-percentile cosine |
|---|---:|---:|---:|
| 0 | 24,984 | 0.999920 | 0.99999884 |
| 1 | 23,783 | 0.999871 | 0.99998630 |
| 2 | 25,462 | 0.999848 | approximately 1 |
| 3 | 23,248 | 0.996016 | 0.99998602 |
| 4 | 25,590 | 0.952138 | approximately 1 |

The rare f4 discrepancy is retained, not corrected by changing weights/cuts or switching execution provider. All reported classification diagnostics use the actual new NPU vectors. Complete CPU bit identity was never the historical contract: an initially over-strict audit assertion was corrected to report these observations. The actual-NPU replay and batch/individual exact checks remain strict.

Jobs ran sequentially under the existing host/NPU guards. No guard waited and no application spill occurred. Baseline counted peaks were 279.3–298.6 MB; NPU counted peaks 336.6–337.0 MB against a 4 GiB admission budget, with OS peak working sets 341.4–344.1 MB. Native provider memory remains outside exact reservation accounting, and outside-process NPU utilization is unknown. Baseline OS/native peaks were not measured. The report process and its replay peaked at up to 1.872 GiB committed/0.418 GiB working set under the 4 GiB commit limit. NPU debug-harness wall times were 93.5–128.3 seconds per fold, not production speed measurements. This does **not** close all of Requirement 15.2's native-accounting concerns.

## Reproduction and evidence

Run from this repository; generated data stays under `D:\BuildArtifacts\log2decision`. The reusable Cargo target remains `task-1.26-target`, with debug symbols/incremental compilation disabled and at most four jobs. The [replay script](../tools/science/transfer218.ps1) checks target/free-space admission and invokes the actual product scorer plus the opt-in NPU test:

```powershell
# Complete replay into a fresh directory (requires the existing dataset, models and NPU runtime).
.\tools\science\transfer218.ps1 -Root D:\BuildArtifacts\log2decision\task-2.18-replay
# Cheap independent encoding/provenance audit or report replay of this measured run.
.\tools\science\transfer218.ps1 -Stage audit -Root D:\BuildArtifacts\log2decision\task-2.18
.\tools\science\transfer218.ps1 -Stage report -Root D:\BuildArtifacts\log2decision\task-2.18
```

Measured evidence lives in `D:\BuildArtifacts\log2decision\task-2.18`:

- `run01/plan.json`, `records.jsonl`, `f*/split.json`: frozen settings, source identities and membership.
- `f*/frozen.json`, `f*/scorers/models/baselines/`, `head-*.json/npz`: proposals, confirmations, actual MarkSet scorer artifacts and diagnostic head provenance (all below `run01/f*/`).
- `run01/f*/npu/provenance.json`, `encoder-training-manifest.json`, `encoding-audit.json`: immutable graph/file identities, reconstructed training/calibration separation, actual vector parity and diversity.
- `run01/report.json`, `predictions.jsonl`: all per-fold/pooled ROC/AP, fixed-budget precision/recall, coverage, missed positives, beyond-query and paired collision sensitivity.
- `logs/`: baseline/NPU, resource, helper-test, audit and replay evidence. The Python-on-PATH hardware refusal is preserved; the successful native run used a process-local minimal PATH, without skipping the hardware checks.
- `sources-during-npu.json`, `analysis-source-before-results.json`, `artifact-hashes.json`, `validation.md`: exact source/binary/runtime/output provenance and the validation index.

The strengthened preparation replay reproduced all 12 initial files byte-for-byte. All five baseline runs were repeated after adding retained-fit memory accounting: all 15 score/pair/frozen artifacts were bit-identical. Scientific helper tests cover deterministic separation, unsupported-positive recall, undefined metrics, deterministic score ties, pre-read source refusal and UTF-8/no-shortening support. No product behavior, encoder architecture, production profile pins, task checkbox or global checkpoint is changed by this measurement.
