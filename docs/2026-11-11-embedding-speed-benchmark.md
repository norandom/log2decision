# Embedding speed and separability on CSIC 2010

Date: 2026-11-11 · Status: measured with one-off scripts · Spec: `.kiro/specs/npu-log-decision/`

## Result

EmbeddingGemma 300m on the NPU is too slow for log triage. It manages about 2.3
logs per second on real requests, and neither batching, parallel requests nor
the power mode moves that number.

It is not better for being slow, either. On the same data, every general text
embedding model we tried separates attacks worse than a plain character n-gram
count that needs no model and runs thousands of times faster.

That changed the question. We had been asking which embedding model is fast
enough. The better question is whether a general text embedding is the right
representation for this kind of log at all, and on this evidence it is not.

## What we decided

- The bad/good ratio comes from character-level features. That is the baseline
  scorer in the spec.
- A small encoder trained on log text is the research question. It runs on the
  NPU (see the last section) and has to beat the baseline to earn its place.
- Larger models, such as a language model that explains a decision, come later
  in the triage flow and only see the logs that survive the cheap stages.

## NPU speed

`flm serve --embed embed-gemma:300m --quiet`, FastFlowLM 1.0.12, AMD Ryzen AI 9
HX 370. The request text is `METHOD URL`, with the body on a second line.

| Case | `performance` mode | `turbo` mode |
| --- | ---: | ---: |
| Model load (first request) | 5.0 s | 3.9 s |
| Short line, one per request | 342 ms/log | 354 ms/log |
| CSIC requests, one per request | 441 ms/log | 505 ms/log |
| CSIC requests, 8 per request | 429 ms/log | 502 ms/log |
| CSIC requests, 32 per request | 435 ms/log | 488 ms/log |
| Two requests of 8 in parallel | 2.1 logs/s | 1.8 logs/s |

A request with several inputs costs the same per input as single requests; the
server works through them one after another. A very short line still takes
about 340 ms, so most of the cost has nothing to do with the length of the
text. And `turbo` was, if anything, slower.

Deduplication helps without rescuing it. CSIC 2010 has 25,608 distinct request
texts among 61,065 rows (42 %). At 2.3 logs/s that is still about three hours.

## Speed and separability

The sample is distinct request texts from CSIC 2010, shuffled with a fixed
seed. After deduplication 62 % of them are anomalous, because normal requests
repeat far more often. Separability is the ROC AUC and average precision of a
logistic regression on the vectors, with 5-fold cross-validation. The CPU runs
used 4,000 texts.

| Representation | Where | Dimensions | Logs/s | ROC AUC | Average precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Character 1-3 grams, hashed (no model) | CPU | 65,536 sparse | 6,743 | 0.962 | 0.975 |
| potion-base-8M (static embeddings) | CPU | 256 | 16,810 | 0.878 | 0.926 |
| bge-small-en-v1.5 | CPU, ONNX | 384 | 18 | 0.789 | 0.875 |
| all-MiniLM-L6-v2 | CPU, ONNX | 384 | 45 | 0.777 | 0.870 |

The NPU model ran on the first 1,000 texts of the same shuffled sample, because
4,000 would have taken half an hour. The baseline is repeated on those 1,000 so
the rows compare like with like.

| Representation (1,000 texts) | Where | Dimensions | Logs/s | ROC AUC | Average precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Character 1-3 grams, hashed (no model) | CPU | 65,536 sparse | 6,994 | 0.910 | 0.946 |
| EmbeddingGemma 300m | NPU | 768 | 2.4 | 0.792 | 0.879 |
| EmbeddingGemma 300m, first 256 dimensions | NPU | 256 | 2.4 | 0.770 | 0.863 |
| EmbeddingGemma 300m, first 128 dimensions | NPU | 128 | 2.4 | 0.716 | 0.829 |

EmbeddingGemma lands level with the small CPU transformers and well below the
baseline. Cutting its vectors to 256 dimensions loses little; 128 loses more.

## Why the sentence models lose

The models trained on natural-language sentences do worst. An attack in a web
request is a few characters inside an otherwise ordinary line, and these models
are trained to look past exactly that kind of surface detail.

The static model does better and is the fastest of all, but it still trails the
character count by a wide margin.

The CPU transformer figures (18 to 45 logs/s) are a floor. They ran through a
Python ONNX wrapper with default settings and nobody tuned them.

## Limits

- One dataset: generated traffic against one application.
- The cross-validation was not grouped by URL. A scorer can do well here by
  learning which endpoints get attacked instead of what an attack looks like.
  That flatters every row in the tables, the baseline included. The spec's
  evaluation holds out whole endpoints for this reason.
- A linear model only. A different one could change the order.
- No energy measurement.

## Follow-up: a small encoder of our own runs fast on the NPU

With the decision made, one thing could still have sunk the encoder idea: maybe
nothing but a vendor-packaged model runs on this NPU. It does run.

The spike was a convolutional encoder over bytes with 976,000 parameters and
random weights. It takes 256 bytes per log, each byte looked up in a 64-value
table on the CPU side, and returns a 128-dimension vector. It was exported to
ONNX with fixed shapes, quantized to 8-bit, and run through ONNX Runtime 1.30
with the Vitis AI provider (`onnxruntime_vitisai`, `voe` 1.9.0, from the Ryzen
AI environment in `../memguard_alpha/tools/oga-venv`).

| Run, 32 logs per call, 15 s | Logs/s | Process CPU |
| --- | ---: | ---: |
| Vitis AI provider | 36,713 | 27 % of one core |
| CPU provider, same 8-bit model | 4,467 | 484 % of one core |

- With one log per call: about 1,800 to 2,500 logs/s on the Vitis AI provider.
- First load, including compilation, took 3 to 5 seconds.
- The outputs match the CPU run of the same model: cosine 1.000 for one log per
  call, at least 0.992 across a batch of 32.
- Placement is inferred, not observed. The provider's partition log names the
  NPU target and sends only the final dequantize step to the CPU, the work left
  the CPU (27 % against 484 %), and no GPU engine was busy. We did not get a
  direct NPU utilization reading.

The spike says nothing about quality. The weights were random. Whether a
trained encoder of this shape beats the character baseline is the open
question, and training cannot run on the NPU: it needs the CPU, the integrated
GPU or another machine.

One more unknown for the Rust implementation is whether the Rust `ort` crate
can load this provider. The spec makes that a gate before any encoder code.

## Reproduction

The measurements came from one-off Python scripts run with `uv`. They are not
in the repository. One ran the CPU models with `fastembed`, `model2vec` and
`scikit-learn`; the other called `POST /v1/embeddings` on the local FastFlowLM
server.
