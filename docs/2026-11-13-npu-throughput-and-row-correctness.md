# The trained encoder on the NPU: are all rows right, how fast is it, how good are the vectors

2026-11-13. Task 2.12 of the spec `npu-log-decision`. Measured on the reference
host (AMD Ryzen AI 9 HX 370, NPU driver 32.0.20102.4480, AMD Ryzen AI 1.9.0
with ONNX Runtime 1.30.0, files from AMD's NuGet package), from Rust, with no
Python on `PATH`. No model was run on the CPU provider for comparison.

## The answer

1. **Is every row right?** Not in the layout the model was exported in. With
   a batch along the first dimension, only the first text of a batch is
   computed right on the NPU. Every other row is wrong, by about as much as two
   unrelated texts differ, and some rows change with the texts of other rows.
   The cause is not the pooling step: the rows are already wrong after the
   first convolution. No provider option changed it.
2. **It can be made right without retraining.** The model file is rewritten so
   that the texts are the rows of one image (input `[1, 64, rows, 256]`,
   output `[1, 128, rows, 1]`). In that layout every row is bit-identical to
   the text encoded alone, for 1 to 256 rows. Requirement 4.6 then holds
   exactly, with a tolerance of zero output steps. The hardware gate asserts it
   on the trained model.
3. **Throughput, end to end from log text to unit vector.** One session, 64
   rows per image, float input: 16,600 logs per second (13,900 in a slower
   run), sustained over 102,432 texts 17,200 (16,000). That is 14 to 17 times
   the 1,000 of requirement 4.2. The fastest setting measured is four sessions
   on four threads, 37,600 logs per second. Feeding the quantized bytes
   directly gives 21,600 with one session and bit-identical vectors.
4. **Start-up.** Ready to encode 4.9 s after a cold start (the provider
   compiles) and 1.0 s from the provider's cache in a fresh process.
   Requirement 4.7 allows 30 s.
5. **Quality of the vectors from the NPU.** The same as the science gate
   recorded for the same 15 fold models. Linear head: cross-endpoint ROC AUC
   0.958 from the NPU's vectors against 0.959 recorded (average precision
   0.978 against 0.978). Neighbour ratio at k = 11: 0.899 against 0.893; at
   k = 51: 0.921 against 0.921. The head clears the 0.90 bar; the neighbour
   ratio at k = 11 still does not.

## Part 1: rows

The model is the published wiring model of task 2.1
(`split17-f0-s0-strip-bce-norm-unit`), packaged with `EncoderPackage` as
`csic-science-2.1-099570ec739c`, opened from the bytes `read_model()` returns.
Inputs come from `ByteCodec` in `payload` mode on CSIC texts with a non-empty
input (23,974 of 25,608). The reference for a text is the text encoded alone
through the same model rewritten for a batch of one. One output step is
0.002527 (the output's scale). Distances are squared L2 between unit vectors.

Where the figures come from. Rows without a mark are printed by the committed
tests (`tests/npu_rows.rs`). Rows marked **[E1]** to **[E6]** are exploratory:
they were measured once with a scratch test that is not committed, and what
remains of them is the log of that run,
`D:\BuildArtifacts\log2decision\task-2.12\logs\explore-1.log` to
`explore-6.log` (outside the repository). The committed gate reproduces the
core of them (first row right, other rows wrong, wrong after the first
convolution, right as rows of an image), not every figure.

### The batch layout, `[32, 64, 1, 256]`

| What was fed | Result |
| --- | --- |
| The same text in all 32 rows, 16 texts | Row 0 equals the text alone, bit for bit, in 16 of 16 batches. All 496 other rows differ from it, by up to 87 output steps; lowest cosine 0.626. **[E1]** With 16 other texts: up to 132 steps, lowest cosine -0.30 |
| 32 different texts, 2 batches | Distance between a row's vector and the text's own, 62 rows other than the first: minimum 0.037, 10 % 0.162, median 0.504, 90 % 1.141, maximum 1.754 |
| For scale: distance from a text's vector to the nearest other text's, 2,000 texts, right vectors | 10 % 0.00009, median 0.00053, 90 % 0.0032, maximum 0.037 |
| **[E1]** One text placed in each of the 32 rows among 31 fixed other texts, 4 texts | 16 distinct vectors per text; lowest cosine against row 0 between 0.09 and 0.77 |
| **[E1]** One text in a fixed row, 16 different sets of all other texts (row 0 included) | Row 0: one vector. Row 1: 15 distinct vectors (lowest cosine 0.38). Row 15: 4. Row 31: 4 |
| Rows 2 to 31 exchanged, rows 0 and 1 kept | Row 1 keeps its vector (2 batches) |
| **[E1]** One text alone in each row, the other rows empty | 8 distinct vectors, lowest cosine 0.981 |
| **[E2]** Batches of 2, 4 and 8 (the file rewritten) | Every row but the first differs from it for identical inputs (lowest cosine 0.76 to 0.98) |
| Batch of 1 against row 0 of the batch of 32 | Bit-identical (16 of 16 batches) |

What the dependence on other rows amounts to: a row's vector changed when all
other rows changed, and row 1 did not change when only rows 2 to 31 were
exchanged. That fits a row depending on the rows before it; which rows a row
depends on was not worked out.

So the median error of a wrong row is about a thousand times the median
distance to the nearest other text. The decision part works on those nearest
distances; vectors from rows other than the first are unusable.

### The cause

The owner's first candidate was the pooling step. It is not the cause. Models
cut at inner tensors (`onnx_edit::with_output_at`), all 32 rows fed the same
text:

| Output taken | Rows that differ from row 0 |
| --- | --- |
| **[E2]** After the first convolution | Rows 1 to 31, all alike, at all 256 positions (8,019 to 8,902 of 32,768 values) |
| **[E2]** After the second convolution | Two groups: rows 1 to 7, rows 8 to 31 |
| **[E2]** Before the pool (after the fifth convolution) | Row 1, and rows 2 to 31 alike |
| **[E1]** After the pool | 31 distinct rows among the 32 |

At the output of the first convolution, with an all-zero input, only row 0 of
the 32 rows shows the response of the bias (**[E3]**: 0.018 in channels 0 and
2, where rows 1 to 31 show 0.000). One input value set to 3.0 in row 0 moves
302 output values; the same value set in row 1 moves none (the committed test
prints both; **[E3]**: five positions and about 60 channels each in row 0, and
none for rows 2 and 31 as well). So for rows other than the first, the first
convolution does not compute what the model says. What it computes instead was
not established from outside the provider.

What was tried and changed nothing, all exploratory (**[E2]**: same 32
distinct rows, same 51 steps, same cosine 0.798 on one text): `opt_level` 1, 2
and 3; `target` `X2-Benchmark` and `PSO`; `enable_preemption` 0; and
(**[E4]**) feeding the quantized uint8 input directly. Three more observations
that are not the output of a test:

- `vaip_config.json` is not among the five runtime files. It was read from
  AMD's NuGet package, `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg`,
  entry `runtimes/win-x64/native/vaip_config.json`: its pass
  `fuse_DPU_VAIML-x2.0` (the pass of the default target `X2`) lists a sub-pass
  `dynamic_input_batch`, and no target or option in it names the batch layout.
- The compiled model differs with the batch size. Sizes of
  `compiled.AMD_AIE2P_4x8_CMC_Overlay.xmodel` in the cache folders of the
  exploration, as they are now: 1,435,236 bytes for a batch of 1, 1,438,380
  for 2, 1,439,104 for 4, 1,439,992 for 8, 1,440,195 for 32. The size also
  moves by some tens of bytes between two compiles of the same model under
  another cache key (1,435,173 and 1,440,129 in the gate's folders).
- **[E4]** A run of the batch of 32 costs 0.61 ms against 0.30 ms for a batch
  of 1, session only.

### The layout that is right

`onnx_edit::with_rows_as_height(model, rows)` rewrites the file, weights
untouched:

- input `[1, 64, rows, 256]`: text `n` is row `n` of an image, channel by
  channel; every convolution has a kernel of height 1, so texts stay apart;
- `GlobalAveragePool` becomes `AveragePool` with kernel `[1, 32]` and strides
  `[1, 1]` written out (without the strides the provider's compiler fails);
- `Flatten` and the quantize pair behind it are removed;
- `Gemm` becomes a `Conv` with a 1 x 1 kernel; its stored weight keeps its
  bytes and gets the dimensions `[128, 256, 1, 1]`;
- output `[1, 128, rows, 1]`.

On the NPU: 33 of 35 nodes, the two on the CPU being the input's
`QuantizeLinear` and the output's `DequantizeLinear`, one compiled subgraph.

| Rows per image | Rows bit-identical to the text alone |
| --- | --- |
| 64 (the gate) | all |
| **[E6]** 1, 2, 8, 32, 128, 256 | all (256 of 256 at 256 rows) |

`with_rows_as_height` refuses a file for which the rewrite would not be the
same arithmetic: a convolution whose kernel, padding, stride or dilation
reaches along the height, a projection that is not `y = x W^T + b` with one
bias value per output, a `Flatten` at another axis, and a quantize pair behind
`Flatten` whose scale or zero point differs from the pair before it
(`tests/onnx_edit.rs`).

The gate (`tests/npu_rows.rs`, 64 rows): identical rows in, 8 texts x 64 rows,
0 rows differ; a text in each of the 64 rows among other texts that change
with the row, 4 texts, 0 vectors differ; 256 different texts in full images
and one image of 3 texts, 0 differ; the same image twice gives the same bits.

**Verdict for requirement 4.6:** it does not hold in the batch layout for any
batch above one. It holds exactly, tolerance zero, with the texts as rows of
one image, and trivially with a batch of one (1,300 logs per second end to
end).

One thing found on the way (**[E5]**): when the provider's compiler failed on the first
version of the rewritten file, the session still opened, in spite of
`error_on_failure`, with 0 nodes on the NPU, and ran everything on the CPU.
Its vectors were within 4 output steps of the NPU's for the same texts
(cosine at least 0.9993). This was an accident, not a comparison that was
looked for, and it is the reason the gate asserts the placement.

## Part 2: throughput

From log text to L2-normalised vector, CSIC's 25,608 distinct texts (4.9 MB),
`payload` mode, one untimed pass and five timed passes per setting, release
build, one process. Two full runs; the machine is shared with another agent
that compiles Rust, and run A was slower throughout. "CPU" is the process's
CPU time (kernel and user) per million logs. The sustained run encodes the
same 25,608 distinct texts four times over in one go (102,432 texts): it is
sustained in time, not a run over 100,000 distinct texts. Every setting gave the same
vectors, bit for bit (checked by comparing all 3.3 million values).

| Setting | Logs/s, run B: median (range) | Logs/s, run A: median | CPU s per million logs (B) | Working set (MiB) |
| --- | --- | --- | --- | --- |
| 1 row per image (a batch of one) | 1,311 (1,223 to 1,324) | 1,327 | 294 | 256 |
| 8 rows | 7,393 (6,404 to 7,508) | 6,973 | 65 | 278 |
| 16 rows | 9,710 (9,054 to 10,907) | 9,123 | 52 | 282 |
| 32 rows | 13,281 (12,477 to 13,836) | 10,556 | 38 | 292 |
| 48 rows | 15,196 (14,240 to 15,856) | 12,839 | 34 | 295 |
| **64 rows** | **16,636 (16,172 to 17,685)** | 13,928 | 33 | 300 |
| 96 rows | 16,853 (15,371 to 17,980) | 14,081 | 32 | 310 |
| 128 rows | 17,003 (16,080 to 17,597) | 13,908 | 30 | 319 |
| 256 rows | 20,091 (19,979 to 21,642) | 16,737 | 27 | 338 |
| 64 rows, uint8 input | 21,147 (20,065 to 22,136) | 17,639 | 20 | 304 |
| 64 rows, uint8 input and output | 21,572 (21,278 to 21,633) | 18,452 | 18 | 307 |
| 64 rows, input and output bound once | 15,928 (14,944 to 17,774) | 14,287 | 36 | 293 |
| 64 rows, 1 intra-op thread | 17,299 (15,764 to 17,374) | 13,898 | 34 | 310 |
| 64 rows, 4 intra-op threads | 17,040 (15,711 to 17,617) | 13,878 | 34 | 311 |
| 64 rows, 2 sessions on 2 threads | 26,486 (25,462 to 27,685) | 24,468 | 45 | 379 |
| 64 rows, 3 sessions on 3 threads | 35,678 (33,469 to 36,652) | 26,717 | 53 | 400 |
| 64 rows, 4 sessions on 4 threads | 37,613 (37,197 to 38,796) | not run | 64 | 421 |
| 64 rows, sustained over 102,432 texts | 17,161 (16,547 to 18,000) | 15,971 | 32 | 352 |
| 64 rows, the run alone (one image repeated) | 23,974 (22,072 to 24,214) | not run | 18 | 300 |

A third run of the 64-row settings alone, after the last change to the code,
gave 15,886 (float input), 23,589 (uint8 input), 22,864 (uint8 input and
output), 27,344, 33,676 and 39,796 (two, three and four sessions) and 17,173
sustained, with the same vectors.

All settings: 33 nodes on the NPU; 2 on the CPU with float input, 1 with
uint8 input, 0 with uint8 input and output. The peak working set of the whole
process, all settings one after the other, was 479 MiB; private commit stayed
between 216 and 387 MiB.

### Where the time goes (64 rows, one session, run B)

59.6 microseconds per log.

| Stage | Share | Per log |
| --- | --- | --- |
| Text to bytes and table lookup, written into the image | 23.1 % | 13.8 us |
| Handing the buffer to the runtime (a tensor that borrows it, no copy) | 0.5 % | 0.3 us |
| The run: the provider's two CPU nodes and the NPU | 75.5 % | 45.0 us |
| Reading the output: rows out of the image, normalising | 1.0 % | 0.6 us |

- The run alone, the same image again and again, costs 2.67 ms per image of
  64 (41.7 us per log). Inside it, the input's `QuantizeLinear` runs on the
  CPU over 1,048,576 floats per image: removing it (uint8 input) takes the
  whole path from 59.6 to 47.3 us per log.
- Writing the image costs 13.8 us per log. `ByteCodec`'s own batch layout
  writes a text in 3 to 4 us (268,000 to 331,000 logs per second, task 2.5).
  The image layout writes 64 separate lines of 1 KiB, 64 KiB apart, per text.
  This is the next thing to make faster and was not optimised here.
- More rows per image help little beyond 64, and compiling takes longer (4.3 s
  at 64 rows, 5.4 s at 128 and 256).

### What each change gave

- **Buffers.** The image buffer is lent to the runtime without a copy and
  reused; the output is read in place. IO binding works with this provider
  and gives the same vectors, and no gain (15,928 against 16,636). As
  measured ("bound once" in the table): the input is a tensor the runtime
  allocated once, written in place and bound again before each run (no copy);
  the output tensor is allocated and bound once. Handing over costs 0.5 %
  either way.
- **Rows per image.** 1 row: 1,311 logs per second. 64 rows: 16,636. 256 rows:
  20,091. All nodes stay on the NPU at every size.
- **The two CPU nodes.** A model without the input's `QuantizeLinear` is made
  by `onnx_edit::with_quantized_ends` and accepted by the provider. The tool
  then writes uint8 with the model's scale (0.030868560075759888) and zero
  point (134), dividing in float32, rounding half to even, saturating
  (`encode::ImageInput::quantized`). The vectors are bit-identical to the
  float-input path over all 25,608 texts, with the input alone and with input
  and output both as bytes. Gain: 27 % with the input, 30 % with both, and
  CPU time per log falls from 33 to 18 s per million.
- **Sessions and threads.** The intra-op thread setting changes nothing. Two
  sessions on two threads give 1.6 times one session, three 2.1 times, four
  2.3 times, with the same vectors; CPU time rises from 33 to 64 s per million
  logs.
- **Provider options.** `opt_level` and `target` had no effect on the outputs
  in part 1; their effect on speed was not measured.

### Start-up and the provider's cache (64 rows)

| Start | Session ready | First 64 vectors |
| --- | --- | --- |
| Cold: the provider compiles | 4.94 s | 4.94 s |
| Warm, fresh process: compiled model from the cache | 0.98 s | 0.99 s |
| Warm, same process (3 times) | 0.44 to 0.60 s | 0.44 to 0.60 s |

Loading the runtime library takes 0.11 s. From the start of the test process
(reading CSIC, verifying the package, loading the runtime) to the first
vectors: 5.5 s cold, 1.6 s warm.

What the cache key must hold, by experiment. The provider looks at nothing but
the key:

| The key of one compiled model reused for | What happened |
| --- | --- |
| The same model at 32 rows instead of 64 | The session opened without compiling; the first run failed |
| Another model of the same shape (a fold model of task 2.1) | The session opened without compiling and returned vectors that are neither the other model's nor the first model's. No error |
| The same model with the provider option `opt_level=3` | Not compiled again; the option was not applied |

So a stale compile is used silently, and with another model of the same shape
it gives wrong vectors. The 2.4 notes asked for revision, runtime release and
a digest of the options. That is confirmed, and one thing is added: the key
must also cover the model file that is actually handed to the provider (the
rewritten file: layout, rows, the version of the rewrite).
`encode::provider_cache_key` builds such a key.

### The best configuration

- Fastest measured: four sessions on four threads, 64 rows, float input:
  37,613 logs per second, 64 CPU seconds per million logs, 421 MiB working
  set.
- Recommended for task 2.6: one session, 64 rows per image, float input, the
  image buffer lent to the runtime and reused, the session created from the
  package's verified bytes rewritten in memory. 16,636 logs per second (13,928
  in the slower run), 33 CPU seconds per million logs (about half a core
  while encoding), 300 MiB working set, 4.9 s cold and 1.0 s warm. It keeps the
  package format as it is.
- Against the codec's own rate (268,000 to 331,000 logs per second in its
  batch layout): the NPU run is the bottleneck, then the image writer.

## Part 3: quality of the vectors from the NPU

The 15 fold models of the science gate (stripped input, objective
`bce-norm-unit`, folds 0 to 4, seeds 0 to 2), each with its own byte table.
For each, the vectors of all 25,608 distinct CSIC texts were computed on the
NPU (64 rows per image, 33 of 35 nodes on the NPU, 14,600 to 18,800 logs per
second) and scored in Rust:

- the folds are the science gate's (`cv5_fold_of_group`), rebuilt in Rust and
  checked against each run's record: number of held-out paths, their SHA-256,
  number of held-out texts, and the hashes of the model and table files;
- linear head: the deterministic logistic regression of `l2d-baseline` (dense
  path) on the fitting side's unit vectors, penalty chosen from 0.01, 0.1, 1,
  10, 100 by five-fold cross-validation on log loss over the fitting side
  only, folds by a hash of the path;
- neighbour ratio: the bad share among the k nearest fitting-side vectors;
- cross-endpoint: the held-out texts of paths that hold both labels, pooled.

"Recorded" is what the science gate's record holds for the same run and the
8-bit model (computed there in Python on the CPU). Each cell is NPU / recorded.

| Fold | Seed | Head ROC AUC | Head AP | k = 11 ROC AUC | k = 11 AP | k = 51 ROC AUC | Head penalty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.9748 / 0.9749 | 0.9845 / 0.9846 | 0.9413 / 0.9291 | 0.9391 / 0.9217 | 0.9536 / 0.9533 | 10 / 10 |
| 1 | 0 | 0.9428 / 0.9429 | 0.9685 / 0.9685 | 0.9170 / 0.9201 | 0.9247 / 0.9285 | 0.9313 / 0.9314 | 10 / 10 |
| 2 | 0 | 0.9498 / 0.9499 | 0.9732 / 0.9733 | 0.9163 / 0.9175 | 0.9213 / 0.9250 | 0.9347 / 0.9323 | 10 / 10 |
| 3 | 0 | 0.9909 / 0.9909 | 0.9955 / 0.9955 | 0.9921 / 0.9923 | 0.9945 / 0.9942 | 0.9922 / 0.9906 | 1 / 1 |
| 4 | 0 | 0.9723 / 0.9746 | 0.9876 / 0.9885 | 0.8675 / 0.8735 | 0.8704 / 0.8732 | 0.8795 / 0.8799 | 0.01 / 0.01 |
| 0 | 1 | 0.9690 / 0.9685 | 0.9787 / 0.9785 | 0.8769 / 0.8331 | 0.8672 / 0.8246 | 0.9483 / 0.9468 | 1 / 1 |
| 1 | 1 | 0.9356 / 0.9361 | 0.9661 / 0.9663 | 0.9097 / 0.9110 | 0.9209 / 0.9207 | 0.9225 / 0.9225 | 10 / 10 |
| 2 | 1 | 0.9520 / 0.9521 | 0.9746 / 0.9746 | 0.9237 / 0.9223 | 0.9299 / 0.9276 | 0.9366 / 0.9373 | 100 / 100 |
| 3 | 1 | 0.9912 / 0.9912 | 0.9955 / 0.9956 | 0.9910 / 0.9909 | 0.9940 / 0.9940 | 0.9891 / 0.9906 | 1 / 1 |
| 4 | 1 | 0.9488 / 0.9532 | 0.9757 / 0.9780 | 0.8105 / 0.8155 | 0.8316 / 0.8348 | 0.8821 / 0.8905 | 0.1 / 0.1 |
| 0 | 2 | 0.9780 / 0.9779 | 0.9863 / 0.9862 | 0.9541 / 0.9504 | 0.9571 / 0.9498 | 0.9657 / 0.9642 | 10 / 10 |
| 1 | 2 | 0.9264 / 0.9263 | 0.9603 / 0.9602 | 0.8898 / 0.8903 | 0.8978 / 0.8984 | 0.9021 / 0.9015 | 100 / 100 |
| 2 | 2 | 0.9439 / 0.9438 | 0.9685 / 0.9684 | 0.9167 / 0.9170 | 0.9263 / 0.9272 | 0.9282 / 0.9267 | 10 / 10 |
| 3 | 2 | 0.9919 / 0.9920 | 0.9961 / 0.9961 | 0.9900 / 0.9900 | 0.9928 / 0.9921 | 0.9920 / 0.9919 | 10 / 10 |
| 4 | 2 | 0.9077 / 0.9071 | 0.9554 / 0.9550 | 0.5890 / 0.5453 | 0.6685 / 0.6456 | 0.6589 / 0.6600 | 0.1 / 0.1 |
| mean | seed 0 | 0.9661 / 0.9666 | 0.9819 / 0.9821 | 0.9268 / 0.9265 | 0.9300 / 0.9285 | 0.9383 / 0.9375 | |
| mean | seed 1 | 0.9593 / 0.9602 | 0.9781 / 0.9786 | 0.9024 / 0.8946 | 0.9087 / 0.9003 | 0.9357 / 0.9375 | |
| mean | seed 2 | 0.9496 / 0.9494 | 0.9733 / 0.9732 | 0.8679 / 0.8586 | 0.8885 / 0.8827 | 0.8894 / 0.8889 | |
| **mean** | **all 15** | **0.9583 / 0.9588** | **0.9778 / 0.9779** | **0.8990 / 0.8932** | **0.9091 / 0.9038** | **0.9211 / 0.9213** | |

- **Linear head:** the NPU's vectors give the quality the science gate
  measured. The largest difference in one run is 0.0044 in ROC AUC and 0.0023
  in average precision; the chosen penalty is the recorded one in all 15 runs;
  none of the 390 fits stopped at the iteration cap.
- **Neighbour ratio, k = 51:** the same on average (0.9211 against 0.9213),
  largest difference in one run 0.0085.
- **Neighbour ratio, k = 11:** 0.899 against 0.893 on average. Twelve runs
  agree within 0.006; three differ more: fold 0 seed 1 (0.877 against 0.833),
  fold 4 seed 2 (0.589 against 0.545) and fold 0 seed 0 (0.941 against 0.929).
  At k = 11 the ratio takes twelve values and many 8-bit vectors are equal or
  equally far, so which of several equally near neighbours is counted matters.
  Here ties go to the earlier stored vector; the science gate's selection
  leaves them to the order of a partial sort. A difference in single output
  steps between the two computations would act the same way. The two were not
  told apart, because that needs the CPU's vectors.
- Against the bar of 0.90: the head passes on every seed mean (0.966, 0.959,
  0.950). The neighbour ratio at k = 11 misses it on average (0.899) and on
  seed 2 (0.868), as recorded; `/tienda1/publico/anadir.jsp` (fold 4) is where
  it fails (0.59 to 0.87).
- The fold that holds both twin endpoints out, which the review of 2.1 asked
  for, is not among these runs: no such model was trained.

## How it was measured

- Code: `crates/l2d-encoder/tests/npu_rows.rs` (the gate),
  `npu_throughput.rs`, `npu_quality.rs`, shared code in `tests/npu_support/`;
  the model rewrites in `src/onnx_edit.rs` (tested without hardware in
  `tests/onnx_edit.rs`), the input layout, quantization rule, normalisation
  and cache key in `src/encode.rs`.
- All hardware tests are ignored by default, fail when the runtime directory,
  the NPU or the trained model is missing, and skip only with
  `L2D_GATE_ALLOW_SKIP=1`. They ran with
  `PATH = C:\Windows\System32;C:\Windows;C:\Users\mariu\.cargo\bin`,
  `L2D_RUNTIME_DIR` = the main checkout's `runtime\ryzenai`, and every cache
  and output under `D:\BuildArtifacts\log2decision\task-2.12\`:

  ```
  cargo test --release -p l2d-encoder --test npu_rows -- --ignored --nocapture --test-threads=1
  cargo test --release -p l2d-encoder --test npu_throughput -- --ignored --nocapture --test-threads=1
  cargo test --release -p l2d-encoder --test npu_quality -- --ignored --nocapture --test-threads=1
  ```

- Every session is opened with a fresh cache folder unless the test is about
  the cache, and the placement is read from the provider's report of that
  compilation.
- The first exploration of part 1 (single values, cuts, options, batch sizes)
  was run with a scratch test whose findings the gate's second test,
  `in_the_batch_layout_only_the_first_row_is_right`, reproduces in short. That
  test fails when the batch layout becomes right, so a later runtime release
  that repairs it is noticed.

## Limits and what was not tried

- One host, one NPU, one driver, one runtime release. The defect of the batch
  layout was seen with the random-weight spike model in task 2.3 as well; why
  the provider does it is not known.
- The machine was shared. Run A is 15 to 25 % below run B in most rows. A
  first exploratory timing of the run alone gave 1.7 ms per image of 64 rows;
  both full runs gave 2.7 ms. The difference is not explained.
- The image writer was not optimised (13.8 us per log).
- uint8 input with several sessions, a text-encoding thread beside the NPU
  thread, 256 rows with several sessions: not tried.
- Provider options were tried for their effect on the rows, not on speed.
  `config_file` with a `vaip_config.json` was not passed; `target=X1` was not
  tried (its overlay file is not among the runtime files).
- Peak memory was read for the process as a whole, not per setting in a
  process of its own.
- Rows above 256 per image, a cache directory with spaces or non-ASCII
  characters, two processes compiling into one cache folder, and a damaged
  cache folder were not tried.
- The four table values that the model's input range clips (2.5 notes) were
  left as they are: the vectors from the NPU are what the model gives with
  them, and part 3 measures the quality with them in.
