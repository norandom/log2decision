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

**The vectors are now stable and reproducible, but they do not classify on endpoints
they have not seen: the baseline scorer beats them on every fold.** Five folds by
endpoint and three seeds (15 encoders, each trained without its held-out endpoints);
classifiers fitted afterwards on the 8-bit vectors; ROC AUC good against bad:

| Scorer | Seen endpoints | Unseen endpoints | Seen to unseen |
|---|---|---|---|
| Masked-byte encoder, linear head | 0.786 ± 0.032 | **0.722 ± 0.070** (0.54 to 0.81) | -0.06 |
| Masked-byte encoder, boosted trees | 0.948 ± 0.016 | 0.688 ± 0.061 | -0.26 |
| Baseline scorer (no seed; n = 5 folds) | 0.961 | **0.923 ± 0.097** | -0.04 |
| Published encoder, linear head (n = 5 folds; saw most of these endpoints' labels in its training) | 0.973 | 0.969 ± 0.027 | |

The baseline is ahead of the masked-byte encoder's linear head on every fold, by
+0.19, +0.08, +0.09, +0.39 and +0.26. On the one large endpoint none of them saw,
`anadir.jsp`: encoder 0.740 (seeds 0.68 to 0.80), published 0.933, baseline 0.996.
The spread of the encoder's figures is over folds and seeds; over seeds alone, inside
a fold, it is about 0.03 (linear and boosted). The task says to stop here and bring
the measurements to the owner; this document does.

1. **(a) Randomness is ruled out.** The same encoder gives the same bits twice in a
   process, in another process, in another batch composition and in other image rows,
   in float, in 8 bits and on the NPU (three NPU processes, every vector and patch
   start identical); a training repeated with the same seed in new processes gives
   the same weights and files, SHA-256 for SHA-256.
2. **(b) Stability passes, by a definition and one measurement.** The cut is now the
   8-bit byte graph's, for every precision, so float and 8 bits cut alike by
   definition; the evidence is 8 bits on the CPU against the NPU: 25,680 of 25,680
   lines cut identically, median cosine 1.000000, lowest 0.99992 over the non-empty
   lines, 97.6 % of the vectors bit-identical. Float to 8 bits: median cosine 0.9999
   over all 15 encoders. The float model's own cut agrees with the 8-bit cut on only
   29 % of the lines on average, so it cannot be the cut; what this choice hides is in
   section (b). 2.16 had 0.24 and 0.97.
3. **(c) The new objective is better than the contrastive one with a linear head, not
   by much otherwise, and it does not keep the detail of the line.** Predicting the
   bytes of blanked and visible patches from the line vector (no label) beats 2.16's
   contrastive objective under the same layout and folds by about 0.10 with the
   linear head (0.722 against 0.626) but by 0.024 with boosted trees (0.688 against
   0.664), and the contrastive control wins fold 3 with the linear head. The vectors
   fall back to **3 to 4 principal components for 95 % of the variance** (mean 3.47;
   2.16's label-free variants: 12 to 32; the contrastive control: 38.6; the published
   encoder: 2.6), which 2.16 treated as a failure and which is against the owner's
   "n dimensions". A benign change of one value leaves the vector where it was
   (median cosine 1.000 over 5,000 pairs; a one-character probe 1.0000), two random
   CSIC texts sit at 0.40 (2.16: 0.02 to 0.31), and the byte-prediction loss is the
   same for blanked and visible patches (2.095 and 2.098). The vector holds the
   endpoint's template more than the line's content.
4. **(d) What the figures say about the cause.** Compared head by head, the linear
   head loses only 0.06 from seen to unseen endpoints, and the good/bad direction
   learned on known endpoints mostly carries over (it keeps its sign for 0.94 of its
   weight; cosine 0.74 between the seen and the unseen difference). The linear head
   falls short of the baseline (0.92 unseen, 0.96 seen) mainly because **the vectors
   hold little label signal that a linear head can use, on seen endpoints already
   (0.786)**. The large drop, 0.26, belongs to the boosted trees alone, and their
   seen figure comes from random stratified splits inside endpoints, which flatters
   it. The endpoint dominates the vector (q5: the endpoint is read back at 99.6 %
   accuracy). An attack does move a vector further than a benign one-value change,
   but on held-out endpoints only at ROC AUC 0.865 ± 0.079 (9 runs), less than the
   contrastive control's 0.915.
5. **NPU.** Both graphs compile and run on the NPU only; every row is right (1,118
   comparisons, 0 differ); 2,037 to 2,166 logs per second over three processes with
   nothing else running (requirement 1,000), 2,091 to 2,143 in the owner's low-power
   mode. The session's placement rule and the provider's duplicated dequantize node of
   2.16 are unchanged and stated below.
6. **Requirement 15.** Every step ran alone, under a 6 GiB process memory limit, with
   its peak recorded: at most 2.55 GiB committed for the training and evaluation
   steps, 3.94 GiB for one qualification script; no run was stopped by the host and no
   figure comes from a resumed run. Whole-corpus structures that remain are small and
   named (section "Resources").

**Recommendation** (section "Recommendation"): keep the stable layout and keep the
baseline scorer as the first scorer; stop using these vectors for classification on
new endpoints (option 1). The reviewer's quick test of centring each vector on its
endpoint's unlabelled mean (option 2(a)) did not close the gap: unseen linear 0.693 ±
0.095.

## What changed against 2.16, and why

| 2.16 | 2.17 | Why |
|---|---|---|
| Each precision cut a line with its own entropy model: float with the float model, 8 bits with the 8-bit graph | One cut for all: the 8-bit byte graph's, computed in the Rust side's arithmetic (steps of the 8-bit logit, a table of 256 exponentials written into the NPU fixture, sums in class order). The float path pools its float byte features at those cuts | The jump rule reads differences of entropies; the 8-bit graph moves them by 0.033 nats on average (99th percentile 0.18), and the float model cuts only 9,786 of 25,608 CSIC lines exactly as the 8-bit graph does (fold 0, seed 0). Putting only the float model's output on the 8-bit grid (a quantization-aware cut that reads the output) gives 9,931: the cut moves because of the 8-bit arithmetic inside the graph, not at its output. A float cut that matches the 8-bit cut for 99 % of lines is out of reach this way, so the 8-bit graph's cut is the cut |
| Line graph: convolutions of stride 1, 2, 2, 1 and an average over all 128 positions, padding included, inside the graph | Four convolutions `[1, 5]` of stride 1 and a 1 x 1 convolution; the graph gives one output per patch position `[1, 128, rows, 128]`; the mean over the line's real patches is plain arithmetic after the graph, as the pooling of bytes into patches is before it | 2.16's line graph on the NPU was not ONNX Runtime's arithmetic (median cosine 0.96 to the reference on the same patch image, 6 % of values equal). The byte graph, of stride 1, was bit-identical. The mean over real patches also keeps the padding of a short line out of its vector |
| Objective: NT-Xent between two views of a line with 30 % of the patches blanked (it trains the vector to ignore a blanked patch, where an attack may sit) | Patch-level byte prediction: 30 % of the patches blanked; from the line vector, the patch position and the byte position, a small decoder (training only) predicts every byte of each blanked patch (part `masked`) and of as many visible ones (part `visible`). No label | Requirement (c): the vector must hold what is in a patch. 2.16's contrastive objective stays as the control, under the new cut and line graph |
| Whole-dataset arrays of byte features (1.9 GB) and pooled patches (1.3 GB); checkpoint and resume, numpy's random state saved but not torch's | The texts are one memory-mapped byte file; byte features are never stored; the pooled patch features are appended to a file as they are made and read back memory-mapped by the batch maker; no checkpoint, no resume. Still held whole, as stated and bounded needs: the vectors of the corpus while they are made and evaluated (`float_vectors`, `onnx_vectors`, `eval`; 25,608 x 128 float32, about 13 MB), the patch starts (`PatchBytes`, about 3 MB) and the baseline's sparse feature rows in `refs` | Requirement 15 and the brief: the 1.3 to 1.9 GB whole-dataset arrays of 2.16 are gone, what is left is small and named; no figure from a resumed run |
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
the training, not noise in the encoding; it is measured in (d): inside a fold the
standard deviation over seeds is about 0.03. The 2.16 defect that would have broken this (a resumed run restored numpy's
random state but not torch's) is gone with the resume itself.

## (b) Stability: cuts and vectors in float, 8 bits and on the NPU

**Pass.** The condition: median cosine float to 8 bits and 8 bits to NPU at least
0.99, cuts identical for at least 99 % of lines. The brief allows the way it passes
("train with the cuts the 8-bit entropy model makes"), but the pass on the cuts rests
on one measurement, not two:

- Float to 8 bits, "100 %" of cuts identical is true **by definition**: the float path
  is given the 8-bit graph's cuts. The real evidence is 8 bits on the CPU against the
  NPU, measured on one encoder (below).
- The cut now depends on the 8-bit graph and so on its **quantization calibration**
  (256 training lines, min/max): another calibration sample is another cut, and with
  it another encoder.
- It holds **only while the byte graph stays bit-identical on the NPU**. Float and
  8-bit cuts agree on only 29 % of the lines, so the cut is fragile: one 8-bit step at
  one byte can move it. Every new NPU driver or Ryzen AI release must re-check the bit
  identity of the byte graph (features and logits) before its vectors are mixed with
  older ones.

| Pair | Cuts identical | Vectors |
|---|---|---|
| Float to 8 bits, masked-byte variant, all 15 encoders (5 folds x 3 seeds) | 100 % by definition: one cut, the 8-bit graph's | median cosine 0.99986 to 0.99993 per encoder; 1st percentile 0.9992 to 0.9998 |
| Float to 8 bits, contrastive control, all 15 encoders | 100 % by definition | median 0.99903 to 0.99936; 1st percentile 0.988 to 0.997 |
| 8 bits (ONNX Runtime, CPU) to NPU, fold 0 seed 0, masked-byte variant, 25,680 texts | 25,680 of 25,680 lines, every patch start (measured) | 25,054 bit-identical; the other 626 differ by at most 0.00292 in a component; median 1.000000, 1st percentile 0.999999, lowest 0.99992 over the non-empty lines (the `cosine_min` of 0.0 in `compare.json` is the one empty probe, whose vector is zero on both sides) |
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

The float and 8-bit vectors give the same figures in (d) **as means** over the 15
runs, to the second or third decimal. Single runs differ more: by up to 0.013 with the
linear head and up to 0.091 with boosted trees on unseen endpoints (fold 4, seed 0:
0.775 float, 0.684 in 8 bits), and the NPU's vectors, nearly bit-identical to the
8-bit ones, moved the boosted figure of fold 0, seed 0 from 0.794 to 0.770. The
boosted unseen figure of one run is noisy; read it as a mean.

## (c) An objective meant to keep local detail

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
positions reached about 2.6 in a smoke run (with a byte model of one epoch, so not
a clean comparison), so the vector tells the decoder something beyond the positions.
On the
held-out fold (texts the encoder never saw; no label read) it is 2.77. The two parts
stay level (masked 2.095, visible 2.098 in the last epoch): the vector says no more
about a visible patch, whose bytes went into it, than about a blanked one, whose bytes
did not. A vector that held the line's own content would make the visible part
easier. Together with what (d) measures (3 to 4 components for 95 % of the variance,
a benign one-value change at cosine 1.000, random pairs at 0.40), this says the
decoder is fed mostly the endpoint's template, the typical bytes at each position of
that endpoint, and not the detail of the line. The objective was meant to keep local
detail; these figures do not show that it does. Over the 15 encoders the last
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
| Published encoder (NPU), see above | 0.969 ± 0.027 (0.933 to 0.996; n = 5) | 0.966 ± 0.026 | 0.941 (n = 5) | 0.973 (n = 5) | 0.991 ± 0.001 |
| Baseline scorer | 0.923 ± 0.097 (0.817 to 0.998; n = 5) | | | 0.961 ± 0.011 | |

**Spreads.** The encoders' "±" is the standard deviation over all 15 runs, folds and
seeds together; over seeds alone, the standard deviation inside a fold is about 0.03
on average (masked-byte, 8 bits: 0.029 linear, 0.029 boosted; contrastive: 0.069
linear, 0.043 boosted), and the folds' means differ by 0.067 (linear). The baseline
and the published encoder's linear head and neighbours do not depend on the seed: one
value per fold, so their spread is over the 5 folds (n = 5). Their boosted and seen
boosted figures take the seed and keep n = 15. (`summary\summary.json`, corrected in
`hg17_run.py summary` after the review; the first version counted the seed-free
figures 15 times, 0.089 and 0.025.)

**Head by head.** The linear head loses 0.06 from seen to unseen endpoints (0.786 to
0.722); boosted trees lose 0.26 (0.948 to 0.688). The boosted trees' seen figure comes
from random stratified splits inside endpoints, where near-copies of a request sit on
both sides, which flatters it. The baseline loses 0.04 (0.961 to 0.923).

**Against the contrastive control** the masked-byte objective wins by about 0.10 with
the linear head (0.722 against 0.626) but by 0.024 with boosted trees (0.688 against
0.664); the contrastive control wins fold 3 with the linear head (0.637 against
0.613).

Per fold, mean of three seeds (8 bits); the baseline is ahead of the masked-byte
linear head on every fold, by +0.19, +0.08, +0.09, +0.39 and +0.26:

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
| Principal components for 95 % of the variance (2.16's label-free variants: 12 to 32) | **3.47 (3 to 4)**: a fall back to few dimensions | 38.6 (32 to 44) | 2.6 |
| Spectrum entropy (bits of 7.00) | 1.56 ± 0.28 | 5.26 ± 0.12 | 0.81 |
| Share of the good/bad difference whose sign holds on the unseen endpoints | **0.94 ± 0.11** (0.56 to 1.00) | 0.46 ± 0.26 | 1.00 |
| Cosine between the good/bad difference seen and unseen | 0.74 ± 0.12 | 0.10 ± 0.13 | 0.97 |
| Attack and its benign request (2,204 pairs that differ in one parameter value): median cosine | 0.906 ± 0.029 | 0.306 ± 0.184 | 0.745 |
| Two benign requests that differ in one value (5,000 pairs): median cosine | 1.000 | 0.899 ± 0.065 | 1.000 |
| An attack moves the vector further than a benign change of one value (ROC AUC over all pairs, mostly on endpoints the encoder trained on) | 0.954 ± 0.029 | 0.888 ± 0.067 | 0.980 |
| **The same, pairs of the held-out fold only** (9 of 15 runs have both kinds) | **0.865 ± 0.079** | **0.915 ± 0.036** | 0.980 |
| Distinct vectors of 25,608 distinct texts (8 bits) | 25,608 | 25,608 | 11,052 |

Two cautions on this table. With about three effective directions, the share of the
good/bad difference that keeps its sign is close to all or nothing per run (0.56 to
1.00 over the 15) and does not compare fairly with the contrastive control's 38
directions, where many small directions each carry a little of the difference. And
the pair figure on all pairs mostly uses endpoints the encoder trained on; on the
held-out fold the masked-byte encoder is at 0.865, below the contrastive control.

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
   means equal the float ones. 2.16's float-to-8-bit cosine of 0.24 was a real fault
   and is gone; with it went the drop from about 0.70 (float) to 0.57 to 0.59 (8 bits)
   that the diagnostic after 2.16 saw.
2. **The vectors do not classify on unseen endpoints well enough.** Unseen: 0.722 ±
   0.070 linear, 0.688 ± 0.061 boosted, 0.54 to 0.81 over the 15 runs. The baseline
   scorer on the same folds: 0.923 ± 0.097 (n = 5), ahead on every fold; the
   published encoder 0.969, partly from labels of those endpoints. On the clean case,
   `anadir.jsp`: 0.74 against 0.93 and 1.00.
3. **The main shortfall is little linearly usable label signal, not a failure to
   carry over.** With the linear head the vectors reach only 0.786 on seen endpoints,
   and lose 0.06 from there to unseen ones; the good/bad direction learned on known
   endpoints mostly carries over (sign kept 0.94, cosine 0.74 between seen and
   unseen). Against the baseline (0.96 seen, 0.92 unseen) the gap is there before any
   endpoint is new. The large seen-to-unseen drop (0.26) belongs to the boosted trees
   alone, measured against a seen figure that random stratified splits inside
   endpoints flatter. That the attack direction turns between endpoints is a minor
   part of what these figures show.
4. **Why the signal is small: the vectors fall back to the endpoint's template.**
   3 to 4 principal components hold 95 % of the variance (mean 3.47; 2.16's label-free
   variants 12 to 32, which 2.16 counted as its success; a fall back to few
   dimensions is what 2.16 treated as a failure and is against the owner's "n
   dimensions"). The endpoint is read back from the vector at 99.6 % accuracy (q5); a
   benign one-value change leaves the vector at cosine 1.000; the byte-prediction
   loss is the same for blanked and visible patches. On held-out endpoints an attack
   moves the vector further than a benign one-value change only at ROC AUC 0.865 ±
   0.079, below the contrastive control's 0.915.
5. The contrastive control has the dimensions (38.6 components) but less label signal
   for a linear head (0.626) and no stable good/bad direction (sign kept 0.46). Neither
   objective gives what the gate asks.

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
figure comes from a resumed run: there is no resume.

Not every structure is streamed. Held whole, as stated and bounded needs: the vectors
of the corpus while they are made and evaluated (`hg17.float_vectors`,
`hg17.onnx_vectors`, the `eval` step; 25,608 x 128 float32, about 13 MB per set), the
patch starts (`PatchBytes`, about 3 MB), the labels and groups, and the baseline's
sparse feature rows in `refs` and `per-endpoint`. They are small and fixed by the
corpus size; the 1.3 to 1.9 GB whole-dataset arrays of byte features and pooled
patches that 2.16 built are gone.

What failed and was fixed:

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
patches after it, the Rust side's arithmetic between the graphs, tested bit for bit),
with the re-check it needs (section (b): the byte graph's bit identity on every new
driver or Ryzen AI release); it removes randomness and instability from the question,
runs at about 2,100 logs per second on the NPU, and its vectors fold nothing. Also
keep the baseline scorer as the first scorer: on the same folds it is at 0.92, ahead
of the encoders (0.69 to 0.72) on every fold.

Options for the owner:

1. **Stop using these vectors to classify new endpoints; the baseline scorer
   triages.** Use the encoder (this layout, masked-byte objective) at most where its
   measured strengths are wanted: grouping by endpoint (k-means NMI 0.66, endpoint
   accuracy 0.996) and, with care, "this request moved further than a benign change"
   (ROC AUC 0.865 ± 0.079 on held-out endpoints, 0.95 only on endpoints it trained
   on; the contrastive control is better on held-out pairs, 0.915). Tasks 2.8 and 2.10
   (scorers on vectors) are then re-scoped or dropped. **This is the option the
   figures support best, and the reviewer's judgement.**
2. **Take the endpoint out of the vector.** (a) Centre each vector on the mean of its
   endpoint's logs and fit the classifiers again. Using a new endpoint's own
   unlabelled logs is transductive, test-time adaptation; no label is read, so it is
   acceptable for batch triage, with three caveats: it needs a warm-up batch of logs
   per endpoint before that endpoint can be scored; the mean includes the attacks
   (CSIC's large mixed endpoints are 58 to 62 % attacks, and an endpoint hit mostly by
   a scanner would be centred on the attack); and it does not compare like for like
   with the baseline, which scores each log alone. **The reviewer ran it** on the 15
   masked-byte encoders (centring on each endpoint's mean and renormalising, the
   held-out endpoints with their own logs; `review\option2a-mask.json`): unseen linear
   0.693 ± 0.095, boosted 0.707 ± 0.054, seen linear 0.799, `anadir.jsp` linear 0.70
   to 0.79. Done this way it does not close the gap to 0.92. What is left of this
   option is (b), a training with the endpoint removed from the target (the decoder
   told the endpoint, or predicting the parameter values only), a full sweep with no
   evidence yet that it would help.
3. **Combine the two objectives.** The contrastive control spreads over 38
   components but has less linear label signal and no stable good/bad direction; the
   masked-byte objective falls back to about three components. A weighted sum (or
   masked bytes plus a variance/covariance term that spreads the vector over its
   dimensions) would test whether "n dimensions" and usable label signal can be had
   together. It keeps the owner's "n dimensions" in view but addresses neither
   measured shortfall directly.

My recommendation: option 1. If the owner wants one more try at the encoder for
classification, option 2 (b) or 3 as one sweep, judged as here (unseen ROC AUC with
its spread over folds and seeds, against the baseline's 0.92 on the same folds).

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
| The review's own figures, option 2(a) | `review\option2a-mask.json`, `review\review-figures.json` |
| Runs in order | `chain.ps1`, `chain2.ps1`, `qual.ps1`, `logs\chain.log`, `logs\chain2.log`, `logs\qual.log` |
