# Embedding speed and separability on CSIC 2010

Date: 2026-11-11 · Status: measured, one-off scripts · Spec: `.kiro/specs/npu-log-decision/`

## Result

EmbeddingGemma 300m on the NPU is too slow for log triage: about 2.3 logs per
second on real requests, and batching, concurrency and the power mode do not
change that. On the same data, every general-purpose text embedding model that
was tried separates attacks worse than a plain character n-gram baseline that
needs no model and runs thousands of times faster.

This changes the question. It is no longer which embedding model is fast
enough. It is whether a general text embedding is the right representation for
this kind of log at all.

## NPU speed

`flm serve --embed embed-gemma:300m --quiet`, FastFlowLM 1.0.12, AMD Ryzen AI 9
HX 370. Request text is `METHOD URL` plus the body on a second line.

| Case | `performance` mode | `turbo` mode |
| --- | ---: | ---: |
| Model load (first request) | 5.0 s | 3.9 s |
| Short line, one per request | 342 ms/log | 354 ms/log |
| CSIC requests, one per request | 441 ms/log | 505 ms/log |
| CSIC requests, 8 per request | 429 ms/log | 502 ms/log |
| CSIC requests, 32 per request | 435 ms/log | 488 ms/log |
| Two requests of 8 in parallel | 2.1 logs/s | 1.8 logs/s |

- A request with several inputs costs the same per input as single requests.
  The server works through them one after another.
- A very short line still takes about 340 ms, so most of the cost does not
  depend on the length of the text.
- `turbo` was not faster than `performance`.

Deduplication helps but does not rescue it: CSIC 2010 has 25,608 distinct
request texts among 61,065 rows (42 %). At 2.3 logs/s that is still about three
hours.

## Speed and separability

Sample: distinct request texts from CSIC 2010, shuffled with a fixed seed.
After deduplication 62 % of them are anomalous, because normal requests repeat
far more often. Separability is the ROC AUC and average precision of a logistic
regression on the vectors, 5-fold cross-validated. CPU runs used 4,000 texts.

| Representation | Where | Dimensions | Logs/s | ROC AUC | Average precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Character 1-3 grams, hashed (no model) | CPU | 65,536 sparse | 6,743 | 0.962 | 0.975 |
| potion-base-8M (static embeddings) | CPU | 256 | 16,810 | 0.878 | 0.926 |
| bge-small-en-v1.5 | CPU, ONNX | 384 | 18 | 0.789 | 0.875 |
| all-MiniLM-L6-v2 | CPU, ONNX | 384 | 45 | 0.777 | 0.870 |

The NPU model was run on the first 1,000 texts of the same shuffled sample,
because 4,000 would take half an hour. The baseline is repeated on those 1,000
so the two rows compare like with like.

| Representation (1,000 texts) | Where | Dimensions | Logs/s | ROC AUC | Average precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Character 1-3 grams, hashed (no model) | CPU | 65,536 sparse | 6,994 | 0.910 | 0.946 |
| EmbeddingGemma 300m | NPU | 768 | 2.4 | 0.792 | 0.879 |
| EmbeddingGemma 300m, first 256 dimensions | NPU | 256 | 2.4 | 0.770 | 0.863 |
| EmbeddingGemma 300m, first 128 dimensions | NPU | 128 | 2.4 | 0.716 | 0.829 |

The slow model is not better for it: it lands level with the small CPU
transformers and well below the baseline. Cutting its vectors loses little at
256 dimensions and more at 128.

## Reading the numbers

- The models trained on natural-language sentences do worst. An attack in a
  web request is a few characters inside an otherwise ordinary line, and these
  models are trained to ignore such surface detail.
- The static model does better and is the fastest of all, but it still trails
  the character baseline by a wide margin.
- The CPU transformer runs were also slow (18 to 45 logs/s). They ran through
  a Python ONNX wrapper with default settings and are a lower bound, not a
  tuned figure.

## Limits

- One dataset, generated traffic against one application.
- Cross-validation was not grouped by URL, so the head may have learned which
  endpoints are attacked rather than what an attack looks like. That favours
  every representation, the baseline included.
- A linear head only. A different head could change the order.
- No energy measurement.

## Consequence for the project

Three ways forward, to be decided:

1. Keep embeddings for search and evidence only, and base the bad/good ratio on
   character-level features. The store then holds both.
2. Train or fine-tune a small encoder on log text, so the vectors carry the
   character-level signal. This is the thesis idea (own tokenizer, own
   encoder) with a model that is actually trained.
3. Drop the NPU requirement for embedding and reserve the NPU for a later
   language-model step, such as explaining a triage decision.

## Follow-up: a small encoder of our own runs fast on the NPU

Decision after the results above: character-level features give the baseline
ratio, and a small encoder trained on log text is the research question. The
blocker was whether such an encoder can run on the NPU at all. It can.

Spike: a convolutional encoder over bytes, 976,000 parameters, random weights.
Input is 256 bytes per log, each byte looked up in a 64-value table on the CPU
side; output is a 128-dimension vector. Exported to ONNX with fixed shapes,
quantized to 8-bit, run through ONNX Runtime 1.30 with the Vitis AI provider
(`onnxruntime_vitisai`, `voe` 1.9.0 from the Ryzen AI environment in
`../memguard_alpha/tools/oga-venv`).

| Run, 32 logs per call, 15 s | Logs/s | Process CPU |
| --- | ---: | ---: |
| Vitis AI provider | 36,713 | 27 % of one core |
| CPU provider, same 8-bit model | 4,467 | 484 % of one core |

- One log per call: about 1,800 to 2,500 logs/s on the Vitis AI provider.
- First load including compilation took 3 to 5 seconds.
- Outputs match the CPU run of the same model (cosine 1.000 for one log per
  call, at least 0.992 across a batch of 32).
- Placement: the provider's partition log names the NPU target and sends only
  the final dequantize step to the CPU, and the work left the CPU (27 % against
  484 %). No GPU engine was busy. A direct NPU utilization reading was not
  obtained.

What this does not show: quality. The weights were random. Whether a trained
encoder of this shape beats the character baseline is the open research
question. Training itself cannot run on the NPU; it needs the CPU, the
integrated GPU or another machine.

Open for the Rust implementation: whether the Rust `ort` crate can load this
provider. If not, the encoder runs behind a small local service, as FastFlowLM
does today.

## Reproduction

One-off Python scripts run with `uv` (not part of the repository): a CPU run
with `fastembed`, `model2vec` and `scikit-learn`, and an NPU run that calls
`POST /v1/embeddings` on the local FastFlowLM server.
