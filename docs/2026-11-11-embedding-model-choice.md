# Embedding model choice

Date: 2026-11-11 · Status: closed for general embedding models; open for our own encoder · Spec: `.kiro/specs/npu-log-decision/`

## Outcome

No general text embedding model was chosen. The working default,
EmbeddingGemma 300m on the NPU, was dropped the same day for two reasons: 2.3
logs per second, and worse separation of attacks than a character n-gram count.
The small CPU models we then measured were faster, and the best of them still
trailed the character count. The numbers are in the
[benchmark](2026-11-11-embedding-speed-benchmark.md).

What remains open is a different question: whether a small encoder trained on
log text can beat the character baseline. That is decided by the evaluation in
the spec, not here.

The rest of this document is the reasoning from before the benchmark. It is
kept because the candidate list and the notes on the thesis models are still
useful.

## What "faster" can mean

Three different things can get faster, and they are easy to mix up.

1. Fewer embedding calls. Identical log lines are embedded once. This needs no
   model change, and on repetitive logs it is probably the largest gain.
2. Shorter vectors. EmbeddingGemma's vectors can be cut to 512, 256 or 128
   dimensions. Storage and search get cheaper. Embedding time does not change.
3. A smaller model. Embedding itself gets faster, and some quality goes.

## Candidates

| Candidate | Size, dimensions | Runs where | What we learned |
| --- | --- | --- | --- |
| EmbeddingGemma 300m (`embed-gemma:300m`) | about 300M parameters, 768 | NPU via FastFlowLM | 2.3 logs/s, ROC AUC 0.79. Dropped |
| Same model, vectors cut to 256 or 128 | 256 or 128 | NPU via FastFlowLM | ROC AUC 0.77 and 0.72; no faster |
| all-MiniLM-L6-v2 | about 22M, 384 | CPU | 45 logs/s untuned, ROC AUC 0.78 |
| bge-small-en-v1.5 | about 33M, 384 | CPU | 18 logs/s untuned, ROC AUC 0.79 |
| Static embeddings (potion-base-8M) | a few million, 256 | CPU, no transformer | 16,810 logs/s, ROC AUC 0.88 |
| gte-large-en-v1.5 | about 430M, 1024 | NPU via Ryzen AI (AMD lists an example) | Not tried; larger than the default |

The sizes and dimensions of the models we did not run ourselves are from memory
of their model cards.

FastFlowLM's catalog (version 1.0.12, as listed on this machine) has exactly
one embedding model, so a smaller NPU model cannot come from there.

The other NPU route is ONNX Runtime with AMD's Vitis AI provider. Before the
benchmark it looked doubtful for three reasons:

- One user reports that BERT compilation hangs on this same processor (Ryzen AI
  1.6, issue 312 in amd/RyzenAI-SW). It is a single report, about a year old.
- AMD's examples are Python and C++. Rust access was unconfirmed.
- A model has to be quantized and compiled for the NPU first.

The benchmark's follow-up spike took some of that doubt away: a small
convolutional encoder compiled in seconds and ran at about 36,700 logs/s
through that provider. Loading it from Rust is still unproven.

## The thesis models

| Model in the thesis | What it is | Usable now? |
| --- | --- | --- |
| Linformer (`linformer-pytorch`) with a custom BPE tokenizer | An efficient-attention architecture. The notebooks appear to use it without pretrained weights, producing 30,000-dimensional vectors | No. There are no pretrained weights to deploy, and the vectors are far too large |
| LongFormer | A 2020 encoder for long inputs, not trained to produce sentence embeddings | No. Not an embedding model; NPU support unknown |
| `bert-base-uncased` | A 2018 encoder, 768 dimensions, mean-pooled in the notebooks | No. Not trained for similarity |
| `BAAI/bge-large-en-v1.5` | A 2023 embedding model, 1024 dimensions | Still a sound model, but larger and slower than what was measured here |

Long input was the reason for Linformer and LongFormer (700 tokens). The
FastFlowLM model card states chunks of up to 2,048 tokens for EmbeddingGemma, so
that reason no longer applies.

The thesis idea itself, a tokenizer and an encoder built for log text, is the
one that survives. What it lacked was a trained model.

## How a model would be judged

This was the plan for comparing candidates, and it carries over to the encoder.
Each candidate embeds the same deduplicated sample, and the assessment tools in
the spec answer:

| Question | Measure |
| --- | --- |
| Does triage still work? | Ranking quality of the bad/good ratio: ROC AUC, average precision, recall at a fixed review budget |
| Does it separate attacks on the same endpoint? | Separability within one URL |
| How much information is lost? | Neighbour overlap against a reference; PCA explained variance; intrinsic dimension |
| Do the vectors cluster? | Silhouette score, projection plots |
| Are the vectors well behaved? | Cosine distribution, anisotropy, norms, duplicate rate |
| What does it cost? | Logs per second, time to first vector, an energy proxy, storage per log |

## Log

| Date | Entry |
| --- | --- |
| 2026-11-11 | Document created. Default measured at about 3 logs/s, 768 dimensions. No alternative measured. |
| 2026-11-11 | Benchmark run. The NPU default is rejected on speed: 2.3 logs/s on real requests, unchanged by batching, parallel requests or power mode. The small CPU models separate attacks worse than a character n-gram count. The question is now the representation, not the model size. |
| 2026-11-11 | Direction agreed: character-level features for the baseline ratio; a small encoder trained on log text as the research question; larger models only later in the triage flow. A 1M-parameter byte-level convolutional encoder with random weights runs at about 36,700 logs/s through the Vitis AI provider. Quality untested. |

## Sources

- FastFlowLM EmbeddingGemma model card: https://fastflow-page.pages.dev/docs/models/embeddinggemma
- EmbeddingGemma overview: https://ai.google.dev/gemma/docs/embeddinggemma
- BERT compilation report: https://github.com/amd/RyzenAI-SW/issues/312
- Ryzen AI model deployment: https://ryzenai.docs.amd.com/en/1.5/modelrun.html
- ONNX Runtime Vitis AI provider: https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html
- Rust `ort` crate feature list (old version): https://docs.rs/crate/ort/1.14.1
