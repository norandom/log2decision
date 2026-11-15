# Entropy-patch encoder stability gate: are the vectors stable, and do they classify on endpoints they have not seen?

2026-11-14 to 2026-11-15. Task 2.17 of the spec `npu-log-decision`, the gate that follows the
[hierarchical encoder gate](2026-11-14-hierarchical-encoder-gate.md) (task 2.16).
A development spike: training and analysis in Python through `uv`
(`tools/science/hg17.py`, `hg17_run.py`, `hg17_sweep.ps1`, `memguard.py`), the
qualification with the scripts of task 2.15 (`tools/embedding-qualification/`), the
hardware check in Rust through `ort` (`crates/l2d-encoder/tests/npu_hierarchical_gate.rs`),
on the reference host (NPU driver 32.0.20102.4480, AMD Ryzen AI 1.9.0). It is the
first spike run under Requirement 15: alone on the machine, one step at a time, every
step under a memory limit with its peak recorded. Every figure is in a named file
under `D:\BuildArtifacts\log2decision\task-2.17\`; the section "Where the figures are"
lists them.

The owner's decisions stand and were not reopened: entropy patches after Meta's Byte
Latent Transformer, the convolutional architecture, the hierarchy bytes, then patches,
then one vector per line, no cutting of the line, n dimensions.

## The answer

**The vectors are now stable and reproducible, but they still do not classify on
endpoints they have not seen.** With five folds by endpoint and three seeds (15
encoders, each trained without the held-out endpoints), a classifier fitted
afterwards on the 8-bit vectors reaches a ROC AUC of **0.722 ± 0.070** (linear; range
0.54 to 0.81) and **0.688 ± 0.061** (boosted trees) on unseen endpoints, against 0.948
with boosted trees on seen ones. On the same folds the **baseline scorer reaches 0.923
± 0.089** and the **published encoder 0.969 ± 0.025** (which saw most of these
endpoints' labels in its training; on the one large endpoint it never saw, 0.933,
the baseline 0.996, this encoder 0.740). The task says to stop here and bring the
measurements to the owner; this document does.

1. **(a) Randomness is ruled out.** The same encoder gives the same bits twice in a
   process, in another process, in another batch composition and in other image rows,
   in float, in 8 bits and on the NPU (three NPU processes, every vector and patch
   start identical); a training repeated with the same seed in new processes gives
   the same weights and files, SHA-256 for SHA-256.
2. **(b) Stability passes.** The cut is now the 8-bit byte graph's, for every
   precision: cuts identical for 100 % of lines float to 8 bits (by construction) and
   8 bits to NPU (25,680 of 25,680, measured); median cosine float to 8 bits 0.9999
   (all 15 encoders), 8 bits to NPU 1.000000 (1st percentile 0.999999, 97.6 %
   bit-identical). The float model's own cut agrees with the 8-bit cut on only 29 % of
   the lines on average, so it cannot be the cut. 2.16 had 0.24 and 0.97.
3. **(c) The detail-keeping objective helps, but not enough.** Predicting the bytes of
   blanked and visible patches from the line vector (no label) beats 2.16's
   contrastive objective under the same stable layout and folds: 0.722 against 0.626
   (linear, unseen), and the good/bad direction learned on known endpoints keeps its
   sign on unseen ones for 0.94 ± 0.11 of its weight against 0.46 ± 0.26.
4. **(d) Why it does not carry over, measured.** The attack is in the vector (an
   attack moves it further than a benign change of one value: ROC AUC 0.95 over
   2,204 CSIC pairs), but the vector is dominated by the endpoint (three principal
   components for 95 % of the variance; the endpoint is recovered from the vector at
   99.6 % accuracy), and the good/bad difference turns between endpoints (cosine 0.74
   between seen and unseen). The contrastive control spreads over 38 components but
   keeps no stable good/bad direction. Neither gives what the gate asks.
5. **NPU.** Both graphs compile and run on the NPU only; every row is right (1,118
   comparisons, 0 differ); 2,037 to 2,166 logs per second over three processes with
   nothing else running (requirement 1,000), 2,091 to 2,143 in the owner's low-power
   mode. The session's placement rule and the provider's duplicated dequantize node of
   2.16 are unchanged and stated below.
6. **Requirement 15.** Every step ran alone, under a 6 GiB process memory limit, with
   its peak recorded: at most 2.55 GiB committed for the training and evaluation
   steps, 3.94 GiB for one qualification script; no run was stopped by the host and no
   figure comes from a resumed run.

**Recommendation** (section "Recommendation"): keep the stable layout and the
baseline scorer as the first scorer; before any further training, test on the
existing vectors whether removing the endpoint (centring each vector on its
endpoint's unlabelled mean) closes the gap; if not, stop using these vectors for
classification on new endpoints.

## What changed against 2.16, and why

| 2.16 | 2.17 | Why |
|---|---|---|
| Each precision cut a line with its own entropy model: float with the float model, 8 bits with the 8-bit graph | One cut for all: the 8-bit byte graph's, computed in the Rust side's arithmetic (steps of the 8-bit logit, a table of 256 exponentials written into the NPU fixture, sums in class order). The float path pools its float byte features at those cuts | The jump rule reads differences of entropies; the 8-bit graph moves them by 0.033 nats on average (99th percentile 0.18), and the float model cuts only 9,786 of 25,608 CSIC lines exactly as the 8-bit graph does (fold 0, seed 0). Putting only the float model's output on the 8-bit grid (a quantization-aware cut that reads the output) gives 9,931: the cut moves because of the 8-bit arithmetic inside the graph, not at its output. A float cut that matches the 8-bit cut for 99 % of lines is out of reach this way, so the 8-bit graph's cut is the cut |
| Line graph: convolutions of stride 1, 2, 2, 1 and an average over all 128 positions, padding included, inside the graph | Four convolutions `[1, 5]` of stride 1 and a 1 x 1 convolution; the graph gives one output per patch position `[1, 128, rows, 128]`; the mean over the line's real patches is plain arithmetic after the graph, as the pooling of bytes into patches is before it | 2.16's line graph on the NPU was not ONNX Runtime's arithmetic (median cosine 0.96 to the reference on the same patch image, 6 % of values equal). The byte graph, of stride 1, was bit-identical. The mean over real patches also keeps the padding of a short line out of its vector |
| Objective: NT-Xent between two views of a line with 30 % of the patches blanked (it trains the vector to ignore a blanked patch, where an attack may sit) | Patch-level byte prediction: 30 % of the patches blanked; from the line vector, the patch position and the byte position, a small decoder (training only) predicts every byte of each blanked patch (part `masked`) and of as many visible ones (part `visible`). No label | Requirement (c): the vector must hold what is in a patch. 2.16's contrastive objective stays as the control, under the new cut and line graph |
| Whole-dataset arrays of byte features (1.9 GB) and pooled patches (1.3 GB); checkpoint and resume, numpy's random state saved but not torch's | The texts are one memory-mapped byte file; byte features are never stored; the pooled patch features are appended to a file as they are made and read back memory-mapped by the batch maker; no checkpoint, no resume | Requirement 15 and the brief: no whole-dataset array, no figure from a resumed run |
| Encoder trained once, on split17's fitting side; the five folds scored with an encoder that had seen four of the five large endpoints | One encoder per fold and seed, trained on the fitting side of that fold only (byte model and line model); five folds by URL path x three seeds | (d): "unseen" must be unseen by the encoder too |
| One seed for most variants | Three seeds (0, 1, 2) per fold and variant; Python, numpy and torch seeded per run, `torch.use_deterministic_algorithms(True)`, no DataLoader workers, 12 threads | (a) and (d) |

Kept from 2.16: the byte model (four causal convolutions `[1, 9]`, 96 channels, the
features of layer 2, next-byte logits as relu(logit + 12)), its objective and
training (8 epochs), the jump rule with its threshold calibrated so that a training
line gets as many patches as with a stride of 8 bytes (on 4,096 training lines, with
the 8-bit graph's entropies), at most 32 bytes per patch and 128 patches per line,
1,024 bytes per line read, 128 dimensions, 12 epochs for the line model, batch 256,
AdamW with a one-cycle schedule, the 8-bit quantization (static QDQ, uint8
activations, int8 weights, per tensor, min/max calibration), 16 rows per byte image
and 64 per line image.

## (a) Randomness: ruled out

Fold 0, seed 0, the masked-byte variant (`enc\f0-s0\mask\determinism.json`,
`logs\determinism-f0-s0-mask.log`). Tolerance zero bits throughout.

| What | Result |
|---|---|
| Float vectors of all 25,608 CSIC texts, twice in one process | 25,608 of 25,608 bit-identical |
| The same, in another process (the training step's own vectors against the check's) | 25,608 of 25,608 |
| The same, in another batch composition (shuffled order, batches of 37 instead of 256) | 25,608 of 25,608 |
| 8-bit vectors (ONNX Runtime on the CPU), twice in one process; in another process; with every text in another row of the image beside other texts | 25,608 of 25,608 each |
| 8-bit byte graph, the 16 rows of an image in reverse order | features and logits identical |
| The training repeated with the same seed, in new processes (byte model, 8-bit byte graph, cuts, pooled patch features, line model, 8-bit line graph, vectors; tag `repeat`) | every file has the same SHA-256; largest difference of any weight 0.0 |
| On the NPU: five further passes over the 25,680 texts in one process | the same bits as the first (asserted by the test) |
| On the NPU: a text in another row of either image, beside other texts or alone, 1,118 comparisons | 0 differ |
| On the NPU: a second and a third process | raw vectors and every patch start of all 25,680 texts identical to the first process (SHA-256 of `npu-raw.f32` and `npu-starts.u32`) |

So nothing in the generation of a vector is random, and a training is reproduced bit
for bit by its seed on this host. The vectors of different seeds differ, as they
should: seed changes the initial weights, the batch order, the masks and the
calibration sample. What 2.16 called "seeds differed by up to 0.27" is a property of
the training, not noise in the encoding; it is measured below as the spread over
seeds. The 2.16 defect that would have broken this (a resumed run restored numpy's
random state but not torch's) is gone with the resume itself.

## (b) Stability: cuts and vectors in float, 8 bits and on the NPU

**Pass.** The condition: median cosine float to 8 bits and 8 bits to NPU at least
0.99, cuts identical for at least 99 % of lines.

| Pair | Cuts identical | Vectors |
|---|---|---|
| Float to 8 bits, masked-byte variant, all 15 encoders (5 folds x 3 seeds) | 100 % by construction: one cut, the 8-bit graph's | median cosine 0.99986 to 0.99993 per encoder; 1st percentile 0.9992 to 0.9998 |
| Float to 8 bits, contrastive control, all 15 encoders | 100 % by construction | median 0.99903 to 0.99936; 1st percentile 0.988 to 0.997 |
| 8 bits (ONNX Runtime, CPU) to NPU, fold 0 seed 0, masked-byte variant, 25,680 texts | 25,680 of 25,680 lines, every patch start | 25,054 bit-identical; median 1.000000, 1st percentile 0.999999 |
| 2.16 for comparison: entropy variant, float to 8 bits; 8 bits to NPU | 180 of 25,608 lines (0.7 %) | median 0.24; 0.97 |

**Why the float model's own cut cannot pass, measured.** Over the 15 byte models the
float model cuts between 155 and 10,769 of the 25,608 CSIC lines (mean 7,306, 29 %)
exactly as its 8-bit graph does; with only its output put on the 8-bit grid, between
151 and 10,892 (mean 8,028). The 8-bit graph moves an entropy by 0.028 to 0.040 nats
on average (99th percentile 0.14 to 0.26), and the jump rule compares a difference of
two entropies with a threshold of 0.33 to 0.52 at every byte of a line of about 190:
one crossing anywhere changes the cut. A quantization-aware entropy model could bring
the float entropies closer to the 8-bit ones but not make every one of about five
million comparisons fall on the same side, so it was not trained; the 8-bit graph's
cut is the definition of the cut, and the float path is a model of the 8-bit path, not
the reverse. Boundary drop (2.16) is not needed for stability any more and was not
used.

The float and 8-bit vectors of one encoder give the same figures in (d) to the
second or third decimal (the tables there show both for the means).

## (c) An objective that keeps local detail

**The objective.** A line is seen with 30 % of its patches blanked (set to zero
before the line graph). The unit line vector, a learned embedding (64 values) of a
patch's position and one (256 values) of a byte's position inside the patch go
through a small decoder (training only, about 150,000 weights) that predicts each
byte of the patch, a cross-entropy over the 256 byte values. Queried are all blanked
patches (masked-patch prediction: what is in a patch, from the others) and as many
visible patches drawn at random (the vector must hold what the line says). Plus 2.16's
term mean((|e| - 1)^2) on the raw vector, which keeps the 8-bit output range from
being set by a few long vectors. No label. The two parts are reported apart.

**A first version failed and was replaced before any figure was taken.** It predicted
the pooled byte features of a blanked patch and picked them among those of all queried
patches of the batch (InfoNCE, as the `masked` objective of 2.16, which 2.16 never
trained). Most patches (method, host, path) are the same in every line of an endpoint,
so the target was ambiguous, the loss stayed near chance (7.47 against about 8.3) and
the vectors collapsed: one principal component held 97.7 % of the variance, two random
CSIC texts had a median cosine of 0.92 (smoke run with a byte model of one epoch,
`logs\smoke-line.log`). Bytes as the target have no such ambiguity.

**What the training shows**, fold 0, seed 0 (`enc\f0-s0\mask\line.json`): the loss
falls from 4.43 to 2.10 nats per byte over 12 epochs; a decoder that reads only the
positions reaches about 2.6 in the smoke run, so the vector carries content. On the
held-out fold (texts the encoder never saw; no label read) it is 2.77. The two parts
stay level (masked 2.095, visible 2.098 in the last epoch): the vector says about as
much about a blanked patch as about a visible one. Over the 15 encoders the last
epoch ends at 1.79 to 2.12 nats per byte, and the held-out fold at 2.28 to 3.02, the
two parts level in every run. The contrastive control ends at 0.79 to 1.14 on its
own scale (NT-Xent) and at 1.65 to 4.92 on the held-out fold.

**The label.** No variant reported here reads it, so there is no label term to
report. A variant with the label as a secondary term is in the code (`mask-label`,
weight 0.5) and was not trained: the brief says to stop once (d) shows the vectors do
not carry over, and a label term on the fitting side's endpoints is the very thing
that does not carry over (2.16: weight 0.5 gave 0.71).

## (d) Evaluation over five folds by endpoint and three seeds

The folds are the science gate's five folds by URL path (`l2dsci.cv5_fold_of_group`):
each holds one of the five large endpoints with both labels, folds 0 to 2 also one of
the three tiny ones (165 to 250 texts, 2 to 4 of them good). For every fold and seed
the byte model and the line model were trained on the four other folds only; the
classifiers were fitted afterwards on the same fitting side. **Unseen**: ROC AUC over
the held-out fold's logs of the endpoints with both labels (4,453 to 5,033 logs per
fold), endpoints neither the encoder nor the classifier saw. **Seen**: five stratified
folds inside the fitting side, every endpoint on both sides, out-of-fold scores, ROC
AUC over the fitting side's logs of endpoints with both labels. Linear: the science
gate's head (logistic regression on unit vectors, penalty by grouped cross-validation);
boosted: scikit-learn's histogram gradient boosting, 200 rounds, seeded. Means over 15
runs (5 folds x 3 seeds) with the standard deviation and the range. The published
encoder's NPU vectors (task 2.15) and the baseline scorer (hashed byte n-grams and
logistic regression, `l2dsci.hashed_rows`, whole text) are fitted on the same folds
(`refs\refs.json`, `summary\summary.json`).

**The published encoder is not a clean reference on these folds.** It was trained
once, with the labels, on split17's fitting side, which holds every endpoint with
both labels except `anadir.jsp`, `entrar.jsp` and `vaciar.jsp`. On folds 0 and 3 its
"unseen" endpoints were seen with their labels in its training; only `anadir.jsp`
(fold 4), `entrar.jsp` and `vaciar.jsp` (folds 1 and 2, both tiny) are unseen by it
too. The baseline scorer and the encoders of this task are fitted per fold.

**Classification, ROC AUC good against bad**

| Vectors | Unseen, linear | Unseen, boosted | Unseen, 11 neighbours | Seen, linear | Seen, boosted |
|---|---|---|---|---|---|
| Masked-byte variant, 8 bits | **0.722 ± 0.070** (0.542 to 0.808) | 0.688 ± 0.061 (0.569 to 0.794) | 0.598 ± 0.095 | 0.786 ± 0.032 | **0.948 ± 0.016** |
| Masked-byte variant, float | 0.723 ± 0.067 | 0.698 ± 0.062 | 0.597 ± 0.096 | 0.787 ± 0.032 | 0.961 ± 0.016 |
| Contrastive control, 8 bits | 0.626 ± 0.080 (0.503 to 0.804) | 0.664 ± 0.070 (0.490 to 0.773) | 0.596 ± 0.081 | 0.791 ± 0.026 | 0.948 ± 0.017 |
| Contrastive control, float | 0.629 ± 0.078 | 0.661 ± 0.071 | 0.598 ± 0.080 | 0.791 ± 0.026 | 0.950 ± 0.017 |
| Published encoder (NPU), see above | 0.969 ± 0.025 (0.933 to 0.996) | 0.966 ± 0.026 | 0.941 ± 0.058 | 0.973 ± 0.009 | 0.991 ± 0.001 |
| Baseline scorer | 0.923 ± 0.089 (0.817 to 0.998) | | | 0.961 ± 0.011 | |

The spread of the baseline is between folds only (it has no seed); the encoders'
spread is over folds and seeds. Per fold, mean of three seeds (8 bits):

| Fold (held-out endpoints with both labels) | Masked-byte, linear | Masked-byte, boosted | Contrastive, linear | Contrastive, boosted | Published, linear | Baseline |
|---|---|---|---|---|---|---|
| 0 (`autenticar`, `caracteristicas`) | 0.795 | 0.740 | 0.691 | 0.722 | 0.993 | 0.985 |
| 1 (`entrar`, `registro`) | 0.734 | 0.694 | 0.576 | 0.660 | 0.974 | 0.817 |
| 2 (`editar`, `vaciar`) | 0.728 | 0.715 | 0.571 | 0.669 | 0.950 | 0.817 |
| 3 (`pagar`) | 0.613 | 0.590 | 0.637 | 0.566 | 0.996 | 0.998 |
| 4 (`anadir`) | 0.740 | 0.702 | 0.658 | 0.702 | 0.933 | 0.996 |

Per endpoint (`refs\per-endpoint.json`, 8 bits, mean of three seeds and range), the one
large endpoint none of the encoders saw, `anadir.jsp`: masked-byte 0.740 (0.682 to
0.803) linear and 0.702 (0.684 to 0.719) boosted, contrastive 0.658 and 0.702,
published encoder 0.933, baseline 0.996. The tiny endpoints swing with the seed
(`entrar.jsp`, 2 good texts: boosted 0.371 to 0.957 for the masked-byte variant) and
carry no weight.

**Geometry, the good/bad direction, attacks and folding** (means over the 15 runs, 8
bits; the good/bad direction inside the endpoints with both labels: the difference of
the class means on the fitting side, in the fitting side's eigen-basis, against the
same difference on the held-out fold)

| | Masked-byte | Contrastive | Published |
|---|---|---|---|
| Principal components for 95 % of the variance | 3.5 (3 to 4) | 38.6 (32 to 44) | 2.6 |
| Spectrum entropy (bits of 7.00) | 1.56 ± 0.28 | 5.26 ± 0.12 | 0.81 |
| Share of the good/bad difference whose sign holds on the unseen endpoints | **0.94 ± 0.11** (0.56 to 1.00) | 0.46 ± 0.26 | 1.00 |
| Cosine between the good/bad difference seen and unseen | 0.74 ± 0.12 | 0.10 ± 0.13 | 0.97 |
| Attack and its benign request (2,204 pairs that differ in one parameter value): median cosine | 0.906 ± 0.029 | 0.306 ± 0.184 | 0.745 |
| Two benign requests that differ in one value (5,000 pairs): median cosine | 1.000 | 0.899 ± 0.065 | 1.000 |
| An attack moves the vector further than a benign change of one value (ROC AUC over the pairs) | 0.954 ± 0.029 | 0.888 ± 0.067 | 0.980 |
| The same, pairs of the held-out fold only (9 of 15 runs have both kinds) | 0.865 ± 0.079 | 0.915 ± 0.036 | 0.980 |
| Distinct vectors of 25,608 distinct texts (8 bits) | 25,608 | 25,608 | 11,052 |

**The qualification scripts of task 2.15** (q1 to q5, unchanged, `--input whole
--input-bytes 1024`), on fold 0, seed 0: the masked-byte variant in 8 bits and from
the NPU, and the contrastive control in 8 bits (`qualification\f0-s0-*\`). Their
"held-out" is the fold's two endpoints with both labels; their "validation" is only
the scripts' own label for every fifth single-label path (2.17 has no validation
part). The NPU's figures equal the 8-bit ones to the third decimal in every line below.

| | Masked-byte, 8 bits (NPU the same) | Contrastive, 8 bits | Published (2.15) |
|---|---|---|---|
| q1: distinct vectors of 25,608 texts; random CSIC pairs, median cosine | 25,608; 0.40 | 25,608; 0.06 | 11,052 (payload only, cut at 256 bytes); 0.83 |
| q2 (probes): one character / one value changed, median cosine | 1.0000 / 0.9997 | 0.996 / 0.938 | |
| q2: attack put into a benign request, median cosine (25 pairs) | 0.83 | 0.20 | |
| q3: k-means, k = 8, against the endpoint: NMI, adjusted Rand | 0.66, 0.57 | 0.49, 0.36 | |
| q3: k-means against the label: NMI | 0.07 | 0.09 | |
| q4: components for 90 / 95 / 99 % of the variance; participation ratio | 2 / 3 / 4; 1.7 | 27 / 33 / 50; 18 | 95 %: 2; 1.3 |
| q5: spectrum entropy, features scaled (bits of 7.00) | 1.60 | 4.95 | 0.75 |
| q5 ablation, all components: held-out label ROC AUC; endpoint accuracy | 0.761; 0.996 | 0.785; 0.968 | |
| q5 ablation: the first 3 components only; all but the first 20 | label 0.785, endpoint 0.786; label 0.570, endpoint 0.869 | | |

q5 says plainly what the vectors of the masked-byte variant hold: the endpoint, to
99.6 % accuracy and 99 % of its entropy, and in every group of components; the label
mostly in the first three components (0.785 from three, 0.57 from all but the first
20). Folding: none in float, 8 bits or on the NPU (every distinct text its own
vector).

**What the measurements say about the question of the task.**

1. Randomness is not the cause ((a)): nothing in the encoding is random and a
   training is reproduced bit for bit by its seed. Instability is not the cause any
   more ((b)): cuts and vectors agree across float, 8 bits and the NPU, and the 8-bit
   figures equal the float ones. 2.16's float-to-8-bit cosine of 0.24 was a real
   fault and is gone; with it went the drop from about 0.70 (float) to 0.57 to 0.59
   (8 bits) that the diagnostic after 2.16 saw.
2. The objective matters ((c)): under the same cuts, the same line graph and the same
   folds, the masked-byte objective beats the contrastive control by about 0.10 with
   the linear head on unseen endpoints (0.722 against 0.626) and turns the good/bad
   direction from a coin toss (0.46 of it keeps its sign) into a stable one (0.94).
   Against 2.16's contrastive entropy variant scored the same way after 2.16 (0.57 to
   0.70 on three endpoints, one seed), it is a step.
3. **But the vectors still do not classify on unseen endpoints.** Seen endpoints:
   0.95 with boosted trees; unseen: 0.69 to 0.72, with 0.54 to 0.81 over the 15 runs.
   The baseline scorer on the same folds: 0.92 (0.82 to 1.00); the published encoder
   0.97, partly from labels of those endpoints. On the clean case, `anadir.jsp`:
   0.74 against 0.93 and 1.00. The gap seen to unseen (0.95 to 0.69 boosted) is
   larger than any spread over seeds (0.06).
4. Where the information goes: the attack is in the vector (an attack moves it far
   more than a benign change of one value, ROC AUC 0.95; the direction from good to
   bad keeps its sign, 0.94), but the masked-byte vectors use about three
   directions for 95 % of their variance, the vectors hold the endpoint above all
   (q5: endpoint accuracy 0.996 from them, from every group of components), and the
   attack sits in directions whose
   orientation turns from one endpoint to the next (cosine 0.74 between the
   good/bad difference on seen and on unseen endpoints, 0.97 for the published
   encoder). A classifier fitted on four endpoints draws its boundary at the wrong
   place for the fifth. The contrastive control has the dimensions (38 components)
   but no stable direction. Neither objective gives both.

## On the NPU

`crates/l2d-encoder/tests/npu_hierarchical_gate.rs`, release build, NPU only, no CPU
provider, on the encoder of fold 0, seed 0, masked-byte variant (fixture
`npu\f0-s0-mask\`, `logs\npu-f0-s0-mask-1.log`). The test is 2.16's, taught the
"positions" layout of the line graph (`"line_output": "positions"` in `layout.json`):
the mean over the real patches after the graph, raw vectors compared, every line's
patch starts written out, and the 256 exponentials read from the fixture.

- **Both graphs compile and run on the NPU**, by the provider's placement report and
  ONNX Runtime's profile of a first run: byte graph 25 of 28 listed nodes on the NPU,
  the input's `QuantizeLinear` and two output `DequantizeLinear` on the CPU, ready in
  3.9 s compiled fresh; line graph 25 of 27 on the NPU, the input's quantize and the
  output's dequantize on the CPU, 2.4 s.
- **The session's rules, stated and not changed.** Placement: the report must list
  exactly the nodes of the model, all on the NPU except the quantize node at the input
  and the dequantize nodes at the outputs, and the profile of a first run must show
  the compiled part executed through the provider. **The duplicated dequantize node of
  2.16 is still there:** the byte graph has 27 nodes in the file and the report lists
  28, because the provider splits the dequantize node of `h`, which is a model output
  and feeds the third convolution, into `h` and `h/duplicated`; the test allows one
  more listed node per output beyond the first, as in 2.16. The line graph has one
  output and no such node.
- **Every row is right**: 1,118 comparisons (texts reversed, rotated by 5 and 37, in
  steps of 3, 66 alone, one text in all 64 rows) differ in no bit.
- **Against the Python 8-bit reference**, 25,608 CSIC texts and 72 probes
  (`npu\f0-s0-mask\compare.json`): the cuts are identical for 25,680 of 25,680 lines
  (every patch start, not only the count); 25,054 raw vectors are bit-identical; the
  cosine is at least 0.999999 for 99 % of the non-empty lines, median 1.000000. The
  626 vectors that differ do so by at most 0.0029 in a component. Where: the byte graph
  on the NPU is bit-identical to the reference (features and logits of texts 0 to 15
  and 64 to 79, the window holding the first two vectors that differ), the patch
  image is bit-identical, and the line graph on that same patch image gives the
  reference's value at 99.9 % of the real patch positions and one 8-bit step off at
  0.1 % (`logs\npu-f0-s0-mask-stages.log`). 2.16's line graph was off at 94 % of its
  values; stride 1 and the mean outside the graph remove almost all of it.
- **Logs per second, from text to vector**, five passes over 25,680 texts after one
  untimed, per process. Measured three times, each with no job of this task beside
  it: **2,037 to 2,166 logs per second** over the three processes. Run 1 (the owner's
  own programs still open, no training running): 2,094 to 2,166, median 2,122. Runs 2
  and 3, after the owner had handed the whole laptop over and with the training chain
  stopped for the measurement: 2,037 to 2,060 (median 2,050) and 2,049 to 2,054
  (median 2,053). The requirement is 1,000. Byte graph 56 % of the time, entropy,
  cuts and pooling 33 to 35 %, line graph 7 %, writing the image 2 %; process CPU 255
  to 278 s per million logs; working set 322 to 363 MiB, peak 326 to 524 MiB
  (`logs\npu-f0-s0-mask-1.log`, `-2.log`, `-3.log`). A fourth run, measured with the
  laptop in the owner's low-power / silent mode for the night: 2,091 to 2,143 (median
  2,114), the same bits (`logs\npu-f0-s0-mask-4-low-power.log`). The published encoder runs at
  16,000 to 42,000; this layout reads four times the bytes with a byte model of four
  layers and is about ten times slower, as in 2.16.

## Resources (Requirement 15)

Every Python step ran as its own process under a **process memory limit of 6 GiB**
(a Windows job object, `tools/science/memguard.py`; an allocation beyond it fails in
the process instead of the host's guard ending it), took the project's lock for hungry
processes, and appended its peak to `runs.jsonl`. The qualification scripts ran under
the same limit (`L2D_MEMORY_GIB`, `eq.py`). One step at a time; the NPU runs and the
Rust build ran with nothing of this task beside them. Physical memory 29.6 GiB; the
host's load at the end of a step was 53 to 64 %.

| Step | Runs | Peak committed (what the limit counts) | Peak working set | Time per run |
|---|---|---|---|---|
| `bytes` (byte model, 8-bit graph, cuts, pooled patches of 25,680 texts) | 17 | at most 2.49 GiB | at most 1.37 GiB | 6.5 to 11.4 min |
| `line` (line model, 8-bit graph, vectors) | 34 | at most 1.74 GiB | at most 1.40 GiB | 6 to 12 min (12 epochs) |
| `eval` (classifiers, spectrum, pairs, folding) | 31 | at most 2.55 GiB | at most 1.10 GiB | 37 to 66 s |
| `refs`, `per-endpoint` | 1 each | 2.45, 2.03 GiB | 1.00, 0.55 GiB | 6, 8 min |
| `determinism` | 1 | 1.23 GiB | 0.95 GiB | 65 s |
| qualification scripts q0 to q5 (three vector sets) | 18, and 2 attempts that stopped at a missing record field before loading anything | at most 3.94 GiB (q3, UMAP); q1 2.12, q2 2.10, q5 1.76, q4 1.43, q0 0.83 | | about 8 min per set |
| NPU test (Rust, release) | 4 processes | | working set 322 to 363 MiB, peak at most 524 MiB | 76 to 82 s |

The committed peak includes about 0.75 GiB that a Python process with numpy and
OpenBLAS reserves before it does anything (`prepare`, which reads the CSV in a second,
peaks at 0.76 GiB). No run came near the limit, none was stopped by the host, and no
figure comes from a resumed run: there is no resume. What failed and was fixed:

- The first version of the masked objective trained but collapsed (section (c)); it
  was replaced before any figure was taken.
- The first chain of runs was stopped by hand during `eval f1 s0 mask` (a one-minute
  step) to free the host for the NPU measurement when the owner handed the laptop
  over; that step was run again from the start, and the chain went on with the steps
  not yet done (`chain2.ps1`). Steps are separate processes, so nothing was resumed.
- `npu-compare` first failed on an index type (offsets as float), and the export for
  the qualification scripts lacked the fields `q0` and `q3` read from a training
  record; both fixed and run again.
- The two Rust test runs of the stage outputs and the NPU measurements each needed an
  incremental release build (6 min 39 s the first time, `-j 4`, target directory
  `D:\BuildArtifacts\log2decision\task-2.17-target`).

The night runs (seeds 1 and 2 of most folds) were made with the laptop in low-power
mode; they took longer, which changes nothing in their results (same code, same seeds,
same arithmetic).

## What was not verified

- The NPU was run on one encoder only (fold 0, seed 0, masked-byte variant). The
  other 29 encoders' NPU vectors are inferred from that one and from the bit-identical
  byte graph; their 8-bit reference vectors were used for (d).
- A quantization-aware entropy model was not trained; why it cannot make the float
  cut equal the 8-bit cut on 99 % of lines is argued from the measured entropy
  differences, not shown by a training.
- The label as a secondary term was not trained (section (c)).
- Only the jump rule of BLT was used; the global rule and other budgets of patches
  were not tried. The line model's width, depth, receptive field (17 patches) and the
  12 epochs were not tuned; no validation part was used to choose anything.
- No source but CSIC: the AIT sample of 2.16 was not encoded. The baseline scorer is
  `l2dsci`'s re-implementation, not the Rust binary.
- The published encoder's reference rows are leaky on folds 0 and 3 and partly 1 and
  2 (section (d)).
- The NPU logs per second are those of the test harness, which is not the tool's
  `EncoderSession`.

## Recommendation

The gate's done condition is met on (a) and (b) and answered on (d): the vectors do
not classify on unseen endpoints. Per the task, this stops here and goes to the owner
with the measurements; nothing in requirements, design or tasks 2.4 to 2.7 was
amended, and tasks 2.8, 2.10 and 2.11 should keep waiting.

Keep in any case, whatever is decided next: the stable layout of this task (the
8-bit graph's cut as the one cut, a line graph of stride 1 with the mean over real
patches after it, the Rust side's arithmetic between the graphs, tested bit for bit);
it removes the instability and the randomness from the question for good, runs at
about 2,100 logs per second on the NPU, and its vectors fold nothing. Also keep the
baseline scorer as the first scorer: on the same folds it is at 0.92, the encoders at
0.69 to 0.72.

Options for the owner:

1. **Stop the encoder line for classification; the baseline scorer triages.** Use
   the encoder (this layout, masked-byte objective) only where its measured strengths
   are wanted: endpoint structure (k-means NMI 0.66, endpoint accuracy 0.996) and
   "this request moved further than a benign change" (pair ROC AUC 0.95), for
   grouping and near-duplicate views, not as the input of a classifier on new
   endpoints. Tasks 2.8 and 2.10 (scorers on vectors) are then re-scoped or dropped.
2. **One more spike on the cause measured here: the endpoint dominates the vector.**
   Both are cheap to test on the vectors that exist before any training: (a) centre
   each vector on its endpoint's mean (unlabelled logs of the same path, available in
   the store at scoring time) and fit the classifiers again on the same folds; (b) if
   that helps, train the line model with the endpoint removed from the target (the
   decoder told the endpoint, or predicting the parameter values only), so that the
   vector spends its few strong directions on what varies inside an endpoint. Pass
   condition as here: unseen ROC AUC with spread, against the baseline's 0.92.
3. **Combine the two objectives.** The contrastive control spreads over 38
   components but keeps no stable good/bad direction; the masked-byte objective keeps
   the direction in three components. A weighted sum (or masked bytes plus a
   variance/covariance term that spreads the vector over its dimensions) would test
   whether "n dimensions" and a stable direction can be had together. This keeps the
   owner's "n dimensions" in view; it does not address the endpoint dominance
   directly, so it is the weaker of the two spikes.

My recommendation: option 2 (a) first, as it costs minutes on existing files and
decides whether option 2 (b) is worth a training sweep; if it does not close most of
the gap to 0.92, option 1.

## Where the figures are

Below `D:\BuildArtifacts\log2decision\task-2.17\`; model weights below
`D:\Source\log2decision\models\task-2.17\f<fold>-s<seed>\` (byte model, 8-bit byte
graph, byte table; per variant the line model and its graphs).

| What | File |
|---|---|
| Every step: command, seconds, memory limit, peaks | `runs.jsonl` |
| Byte model, cuts, float-against-8-bit cut counts, per fold and seed | `enc\f<fold>-s<seed>\bytes.json`, `logs\bytes-*.log` |
| Line model, training history, held-out objective, float-against-8-bit cosine, file hashes | `enc\f<fold>-s<seed>\<variant>\line.json`, `logs\line-*.log` |
| Raw vectors (float, 8 bits; NPU for f0-s0 mask) | `enc\f<fold>-s<seed>\<variant>\csic-*.f32`, `probes-*.f32` |
| Classifiers, spectrum, sign stability, pairs, folding per encoder | `enc\f<fold>-s<seed>\<variant>\eval.json` |
| Means and spread over folds and seeds; reference rows | `summary\summary.json`, `logs\summary.log`, `refs\refs.json` |
| Per held-out endpoint | `refs\per-endpoint.json`, `logs\per-endpoint.log` |
| (a) | `enc\f0-s0\mask\determinism.json`, `logs\determinism-f0-s0-mask.log`, `enc\f0-s0-repeat\` |
| NPU | `npu\f0-s0-mask\` (fixture, `compare.json`), `logs\npu-f0-s0-mask-1.log` to `-4-low-power.log`, `logs\npu-f0-s0-mask-stages.log` |
| Qualification (q0 to q5, pictures) | `qualification\f0-s0-mask-int8\`, `-mask-npu\`, `-contrastive-int8\`; vectors in `vectors\` |
| The collapsed first objective | `logs\smoke-line.log` |
| Runs in order | `chain.ps1`, `chain2.ps1`, `qual.ps1`, `logs\chain.log`, `logs\chain2.log`, `logs\qual.log` |
