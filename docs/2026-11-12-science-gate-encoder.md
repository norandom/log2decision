# Science gate: does a trained byte-level encoder earn its place on the NPU?

Date: 2026-11-12 · Status: measured, task 2.1 · Spec: `.kiro/specs/npu-log-decision/` · Scripts: [`tools/science/`](../tools/science/README.md)

## Review, read this first

An independent review on the same day confirmed every figure below from the
run records, found no leakage, and reproduced a run byte for byte. It also
found two things this document does not contain, and they take back the lead
over the baseline:

- **Twin endpoints.** `registro.jsp` and `editar.jsp` have the same
  13-parameter form. In folds 1 and 2 one is held out while the other is
  fitted, and those are the two folds where the encoder beats the baseline.
  With both held out together (one seed): head 0.639, neighbour ratio 0.504,
  baseline 0.762.
- **The baseline's penalty.** The baseline's 0.83 on those folds belongs to its
  default L2 of 1.0. At 0.1 the baseline averages 0.955 over the five folds,
  level with the head's 0.959. That value was chosen with held-out figures in
  view, so it shows sensitivity, not a setting.

What stands: the linear head clears the 0.90 bar on grouped folds, with every
objective variant and every seed. What does not: "better than the baseline".
It is about level at best, and behind on one large endpoint, on the earlier
three-endpoint split, and on endpoints of a form it has not seen. The gate
decision is the owner's. Conditions the review names before an encoder package
is frozen: repeat against task 1.12 with the baseline's penalty settled, add a
fold that holds both twin endpoints out, and run the hardware check on this
model.

## Answer

As written by the implementer, before the review:

Yes for one use of the vectors, no for the other, and not with the training
objective the design names.

- **Linear head on the 8-bit vectors: passes.** Cross-endpoint ROC AUC 0.959
  (mean of 5 grouped folds and 3 seeds; seeds 0.967, 0.960, 0.949), against the
  gate's 0.90 and the baseline's 0.923 on the same folds. It wins by about 0.11
  on two endpoints where the baseline is weak, ties on two, and loses by 0.04 on
  one. On the single split of the 1.7 note it loses: 0.962 against 0.989.
- **Nearest-neighbour ratio at the design's defaults (k = 11): does not pass.**
  0.893 on average (seeds 0.927, 0.895, 0.859), below the baseline, and on one
  endpoint between 0.55 and 0.87 depending on the seed.
- **The design's objective does not work.** Cross-entropy on a classifier after
  the raw 128-value output puts the signal into the length of the vector. The tool
  normalizes the length away, and 8-bit quantization destroys short vectors
  (float against 8-bit cosine 0.66). Two changes fixed it: the classifier reads the
  normalized vector, and a penalty keeps the raw length near 1. With both, float
  and 8-bit agree (cosine 0.9999, ROC AUC within 0.0006 for the head).
- The 8-bit model has the same interface and the same operators, node for node,
  as the random-weight spike that ran on the NPU. It was not run on the NPU here.

What this means for the plan: the encoder is worth deploying as the source of
vectors for the linear head (task 2.10). The neighbour scorer (task 2.8) needs
work before it can be the primary scorer. The comparison must be repeated
against the output of task 1.12, which does not exist yet; the baseline figures
below come from a Python re-implementation.

## Setup

| Item | Value |
| --- | --- |
| Data | HTTP CSIC 2010, `data/csic-2010/csic_database.csv`, SHA-256 `c420f0bc…e44e637`, 61,065 records, 25,608 distinct texts (9,644 good, 15,964 bad) |
| Groups | 1,632 URL paths; 8 hold both labels, of which 5 are large (4,453 to 4,783 texts) and 3 are tiny (165 to 250 texts, 2 to 4 good) |
| Folds | 5, by path. The 8 mixed paths are dealt from largest to smallest, so each fold holds one large endpoint; single-label paths round-robin in byte order |
| 1.7 split | Mixed paths 2, 5 and 8 in name order held out (`anadir.jsp`, `entrar.jsp`, `vaciar.jsp`; 4,783 texts), all else fitted (20,825) |
| Encoder | Architecture v1 of the design: 256 bytes, byte table 256 × 64, five convolutions (kernel 5), global average pooling, linear to 128. 976,000 parameters in the graph, 16,384 in the table |
| Input text | `strip`: the payload only (after the first `?` or line feed). `keep`: the whole text. First 256 bytes, padded with byte 0, whose table row is zero |
| Training | PyTorch 2.15.0 on the CPU, AdamW, learning rate 0.001 with a one-cycle schedule, weight decay 0.01, batch 64, 8 epochs, seeds 0, 1, 2. 4 or 8 threads |
| Epoch choice | The epoch with the best neighbour-ratio ROC AUC on a validation part of the fitting side: one large endpoint and every fifth single-label path. The encoder does not train on that part |
| Export | ONNX opset 17, static `[32, 64, 1, 256]` to `[32, 128]`; ONNX Runtime 1.31.1 `quantize_static`, QDQ, uint8 activations, int8 weights, per tensor, min/max over 512 training texts |
| Scorers | (a) share of bad among the 11 nearest fitting-side vectors, cosine on unit vectors, no reach limit; (b) logistic regression on unit vectors, penalty chosen from 0.01 to 100 by grouped cross-validation inside the fitting side |
| Baseline | Python re-implementation of `l2d-baseline`: byte 1- to 3-grams, FNV-1a, 65,536 buckets, signed, unit rows, L2 penalty 1.0, bias not penalized |
| Metric | ROC AUC and average precision on the held-out logs of paths that hold both labels, pooled per fold (requirement 9.8) |

There are 61 run records: 55 encoder runs and 6 baseline fits, about 7.5 hours
of run time in total. An encoder run takes 5 to 12 minutes; training alone 215 to 640 s for 8 epochs,
because other jobs shared the machine (about 20 s per epoch when it was quiet).

## Results on the five folds

Method and path stripped, 8-bit model, objective `bce-norm-unit` (described
below). Encoder cells are the mean of three seeds with the lowest and highest
seed.

### Cross-endpoint ROC AUC

| Fold | Held-out endpoints | Baseline | Encoder, linear head | Encoder, neighbour ratio | Head minus baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | `autenticar`, `caracteristicas` | 0.976 | 0.974 (0.969 to 0.978) | 0.904 (0.833 to 0.950) | -0.003 |
| 1 | `registro`, `entrar` | 0.830 | 0.935 (0.926 to 0.943) | 0.907 (0.890 to 0.920) | +0.105 |
| 2 | `editar`, `vaciar` | 0.829 | 0.949 (0.944 to 0.952) | 0.919 (0.917 to 0.922) | +0.120 |
| 3 | `pagar` | 0.991 | 0.991 (0.991 to 0.992) | 0.991 (0.990 to 0.992) | +0.001 |
| 4 | `anadir` | 0.988 | 0.945 (0.907 to 0.975) | 0.745 (0.545 to 0.874) | -0.043 |
| Mean | | 0.923 | 0.959 | 0.893 | +0.036 |

| Mean over folds | Seed 0 | Seed 1 | Seed 2 | Mean | Spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear head, 8-bit | 0.967 | 0.960 | 0.949 | 0.959 | 0.017 |
| Linear head, float | 0.967 | 0.960 | 0.949 | 0.959 | 0.017 |
| Neighbour ratio, 8-bit | 0.927 | 0.895 | 0.859 | 0.893 | 0.068 |
| Neighbour ratio, float | 0.926 | 0.895 | 0.859 | 0.894 | 0.067 |
| Baseline, strip (keep: 0.923) | | | | 0.923 | 0 |

### Cross-endpoint average precision

| Fold | Baseline | Encoder, linear head | Encoder, neighbour ratio |
| --- | ---: | ---: | ---: |
| 0 | 0.987 | 0.983 | 0.899 |
| 1 | 0.888 | 0.965 | 0.916 |
| 2 | 0.887 | 0.972 | 0.927 |
| 3 | 0.995 | 0.996 | 0.994 |
| 4 | 0.995 | 0.974 | 0.785 |
| Mean | 0.950 | 0.978 | 0.904 |

The head beats the baseline's mean in every seed. The lead rests on two folds;
with five endpoints that is a pattern, not a significance test. The lowest single
run of the head is 0.907 (fold 4, seed 2), still above the gate.

## The 1.7 split, like for like

The Python baseline reproduces the Rust figures of the 1.7 note: 0.9889
stripped and 0.9962 kept (note: 0.989 and 0.996), with the same row counts.

| Scorer, 8-bit, stripped | ROC AUC, seeds 0, 1, 2 | Mean | Average precision, mean |
| --- | --- | ---: | ---: |
| Baseline | | 0.989 | 0.995 |
| Encoder, linear head | 0.938, 0.986, 0.963 | 0.962 | 0.984 |
| Encoder, neighbour ratio | 0.592, 0.709, 0.780 | 0.694 | 0.755 |

On this split the encoder loses to the baseline with both scorers. It is the
`anadir` endpoint again (4,453 of the 4,783 held-out texts).

## Method and path kept

Seed 0 only, same objective, 8-bit.

| | Baseline keep | Encoder keep, head | Encoder keep, neighbours | Encoder strip, head (seed 0) |
| --- | ---: | ---: | ---: | ---: |
| Five folds, mean ROC AUC | 0.923 | 0.928 | 0.900 | 0.967 |
| Fold 1 (`registro`) | 0.817 | 0.833 | 0.814 | 0.943 |
| Fold 2 (`editar`) | 0.817 | 0.848 | 0.824 | 0.950 |
| Fold 4 (`anadir`) | 0.996 | 0.994 | 0.944 | 0.975 |
| 1.7 split | 0.996 | 0.843 | 0.839 | 0.938 |

Kept input is level with the baseline on the folds and worse than stripped
input. The cause is the 256-byte limit: with the method, host and path in front,
91 % of the texts of the two long-form endpoints are cut (45 % when stripped; 38 %
and 19 % over all texts). The encoder does not win by memorizing endpoints; it
does better without them.

## Float against 8-bit

| Objective | Runs | Cosine, mean (lowest run) | Logs below cosine 0.99 | Neighbour ratio, 8-bit minus float ROC AUC | Linear head, 8-bit minus float |
| --- | ---: | ---: | ---: | --- | --- |
| `bce` (design), strip | 5 | 0.656 (0.594) | 73 % | -0.336 (-0.484 to -0.236) | +0.112 (-0.052 to +0.605) |
| `bce-norm`, strip | 15 | 0.841 (0.509) | 59 % | -0.099 (-0.489 to +0.112) | -0.006 (-0.020 to -0.001) |
| `bce-norm-unit`, strip | 15 | 0.9999 (0.9999) | 0 % | -0.0002 (-0.0035 to +0.0040) | +0.0001 (-0.0001 to +0.0006) |
| `bce-norm-unit`, keep | 5 | 0.9999 (0.9998) | 0 % | -0.0002 (-0.0099 to +0.0104) | +0.0001 (-0.0001 to +0.0004) |

The float ONNX model matches PyTorch to 3e-7 (largest absolute difference,
`bce-norm-unit`; 2e-4 for the long vectors of `bce`).

## Same model shape as the spike

Every 8-bit model of every run was compared with
`D:\BuildArtifacts\log2decision\npu-cnn-spike\enc_b32_int8.onnx` (the model the
hardware gate ran): 55 of 55 identical in all of these.

| Property | Spike and trained model |
| --- | --- |
| Input | `x`, float32, `[32, 64, 1, 256]` |
| Output | `e`, float32, `[32, 128]` |
| Opset | 17 |
| Operators | 5 Conv (kernel `[1, 5]`, pads `[0, 2, 0, 2]`, strides 1, 2, 2, 2, 1), 1 GlobalAveragePool, 1 Flatten, 1 Gemm, 9 QuantizeLinear, 21 DequantizeLinear, in the same order with the same attributes; no Relu node (the quantizer folds it into the uint8 range) |
| Quantization | uint8 activations (18 zero points), int8 symmetric weights (6), int32 biases (6), no per-channel scale |
| Initializers | 52, same shapes and types |

Weights and ranges differ, so placement on the NPU is expected to be the same
but is not proven. To check with the test of the `gate-2.2` branch: the node
counts placed on the NPU and the CPU are those of the spike, the provider does
not reject a scale, and the NPU output matches the CPU output of the same 8-bit
model (cosine at least 0.99, requirement 4.6).

## What was tried

| Variant | What it is | Five folds, 8-bit: head / neighbours | Verdict |
| --- | --- | --- | --- |
| `bce` | The design: cross-entropy on a linear layer after the raw output. Seed 0 | 0.940 / 0.542 | Vectors are one-dimensional: 99.99 % of the variance in one direction, lengths from 0.9 to 2,565, good logs short and bad logs long. Unusable after normalization and after quantization. The float head scored 0.385 on one fold |
| `bce-norm` | The classifier reads 16 × the unit vector. Seeds 0, 1, 2 | 0.946 / 0.750 | Better vectors, but lengths still span 0.5 to 155, so the 8-bit output range crushes the short ones; neighbour ratio falls to 0.50 in several runs |
| `bce-norm-unit` | `bce-norm` plus the penalty mean((length - 1)²). Seeds 0, 1, 2 | 0.959 / 0.893 | Lengths 0.8 to 1.4 (1st to 99th percentile, all runs); float and 8-bit agree. The reported result |
| `supcon` | Supervised contrastive loss, batch 256. One run on the 1.7 split | 0.769 / 0.500 (that split) | Clusters by label on seen endpoints; every held-out log lands next to bad logs |
| Neighbours, k = 5 and 51 | Sensitivity only, `bce-norm-unit` | k = 5: 0.849; k = 51: 0.921 | A larger k helps; not chosen in advance, so not the reported figure |
| Reach 0.5, minimum support 3 | The design's defaults | Same as without reach; no log abstained | With unit-length training all neighbours are close; the reach does not act |
| Kept input | See above | 0.928 / 0.900 | Worse than stripped |

`bce-norm` was chosen over `supcon` on the validation figure of the pilot run.
The length penalty was added after diagnosing the quantization loss, which needs
no held-out label. But its results were looked at after the earlier variants'
held-out results were known, so the 0.959 is not a blind figure.

A repeat of one run (same seed, same settings, 4 threads) gave byte-identical
model files and identical metrics.

Other figures from the same runs (8-bit, stripped, `bce-norm-unit`): overall
ROC AUC on all held-out logs 0.953 for the head, 0.871 for the neighbour ratio,
0.929 for the baseline. Each fold holds about 327 logs of single-label paths (20
of them good over five folds). With a review budget equal to the number of bad
held-out logs, the head ranks 85 % of the bad ones inside it, the neighbour ratio
53 %, the baseline 100 %. Stripped, these logs have no payload, so every scorer
gives them one constant value; the figure says little.

## Artifacts

`models/science-2.1/` in the main checkout (not tracked). The model is the
seed 0 run of the 1.7 split, stripped input, `bce-norm-unit` (8-bit head 0.938,
neighbours 0.592 on its held-out endpoints): the weakest of the three seeds, taken
because the seed was fixed before the results.

| File | SHA-256 |
| --- | --- |
| `split17-f0-s0-strip-bce-norm-unit.int8.onnx` | `45f1810b553ae618a4ffe49a3ad96d4067f7ad1bbb60a181d212ec9082e4f43a` |
| `split17-f0-s0-strip-bce-norm-unit.f32.onnx` | `df1d385c700154baab11eeeafb4254e3b98c9b0c39d0a7e92f384163a94a150d` |
| `split17-f0-s0-strip-bce-norm-unit.table.bin` (256 × 64 float32, little-endian, row = byte) | `f90c5f6812989e6bd087a67ea72bf6d5f3600ed37a7899e623fce30ac65af58e` |
| `record.json` (data hash, split, seed, settings, metrics, versions, durations, fold summaries) | `655ae7542bac5035d4004133034a024dcac17596a85e87b9a5f1281ca10ee5e5` |

The run records and models of all runs are in
`D:\BuildArtifacts\log2decision\science-2.1\runs\`.

## Limits

- One dataset, one generated application, five endpoints of real size. Fold
  means rest on five numbers.
- The baseline is a Python re-implementation (scikit-learn L-BFGS in place of
  `argmin`). It matches the Rust figures of the 1.7 note to three decimals; the
  Rust code was not run here. Task 1.12 is not done: its folds may differ, and the
  comparison must be repeated against its output.
- The validation endpoint is taken out of encoder training, so each encoder saw
  three large endpoints, not four. A final model trained on everything was not
  made and could not be evaluated.
- Kept input and the design's objective were run with one seed.
- No run on the NPU. No Rust code read the model.
- The review budget of requirement 9.1 is not defined for this gate; recall at a
  budget is not reported.
- 256 bytes cut 19 % of the stripped texts. A longer input was not tried; it
  would change the model the hardware gate ran.
- Training used PyTorch, not Burn. Whether Burn reproduces this is task 4.4.
