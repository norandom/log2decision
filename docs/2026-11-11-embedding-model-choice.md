# Embedding model choice

Date: 2026-11-11 · Status: open; NPU default rejected on speed the same day, see the log at the end · Spec: `.kiro/specs/npu-log-decision/`

## Position today

EmbeddingGemma 300m through FastFlowLM is the only embedding model that is known
to run on this NPU, and it is the working default. No faster NPU model is
available without new work. Faster options exist, but each is either a CPU model
or needs an NPU route that nobody has proven on this machine. The vector-set
analysis decides whether a smaller or faster option is good enough.

## What "faster" can mean

Three different things get faster, and they should not be mixed up:

1. Fewer embedding calls. Identical log lines are embedded once. This needs no
   model change and is likely the largest gain on repetitive logs.
2. Shorter vectors. EmbeddingGemma's vectors can be cut to 512, 256 or 128
   dimensions. Storage and search get cheaper; embedding time does not change.
3. A smaller model. Embedding itself gets faster, at some cost in quality.

## Candidates

| Candidate | Size, dimensions | Runs where | Status on this machine |
| --- | --- | --- | --- |
| EmbeddingGemma 300m (`embed-gemma:300m`) | about 300M parameters, 768 | NPU via FastFlowLM | Measured: works, about 3 logs/s unoptimized |
| Same model, vectors cut to 256 or 128 | 256 or 128 | NPU via FastFlowLM | Not measured; no new model needed |
| all-MiniLM-L6-v2 | about 22M, 384 | CPU; NPU unproven | Not tried |
| bge-small-en-v1.5 | about 33M, 384 | CPU; NPU unproven | Not tried |
| Static embeddings (model2vec family) | a few million, 256 | CPU only, no transformer | Not tried |
| gte-large-en-v1.5 | about 430M, 1024 | NPU via Ryzen AI (AMD lists an example) | Not tried; larger than the default, so not faster |

Sizes and dimensions for the untried models are from memory of their model
cards and need checking before use.

FastFlowLM's catalog (version 1.0.12, listed on this machine) contains one
embedding model only, so a smaller NPU model cannot come from that route.

The other NPU route is ONNX Runtime with AMD's Vitis AI provider. It is open for
three reasons:

- One user reports that BERT compilation hangs on the same processor (Ryzen AI
  1.6, issue 312 in amd/RyzenAI-SW). It is a single report, about a year old.
- AMD's examples are Python and C++. The Rust `ort` crate documents its `vitis`
  feature for other hardware, so Rust access is unconfirmed.
- The model has to be quantized and compiled for the NPU first.

Small models are fast on a CPU. For a 22M-parameter model the CPU may use less
energy per log than the NPU does for a 300M-parameter one. That is a guess until
measured, and it is the comparison that matters most.

## The thesis models

| Model in the thesis | What it is | Usable now? |
| --- | --- | --- |
| Linformer (`linformer-pytorch`) with a custom BPE tokenizer | An efficient-attention architecture; the notebooks appear to use it without pretrained weights, producing 30,000-dimensional vectors | No. There are no pretrained weights to deploy, and the vectors are far too large |
| LongFormer | 2020 encoder for long inputs, not trained to produce sentence embeddings | No. Not an embedding model; NPU support unknown |
| `bert-base-uncased` | 2018 encoder, 768 dimensions, mean-pooled in the notebooks | No. Not trained for similarity; superseded by the small models above |
| `BAAI/bge-large-en-v1.5` | 2023 embedding model, 1024 dimensions | Still a sound model, but larger and slower than the default. A CPU reference at most |

Long input was the reason for Linformer and LongFormer (700 tokens). The
FastFlowLM model card states chunks of up to 2,048 tokens for EmbeddingGemma, so
that reason no longer applies.

## How the analysis decides

Each candidate embeds the same stratified, deduplicated sample of CSIC 2010.
The comparison uses the assessment tools from the spec:

| Question | Measure |
| --- | --- |
| Does triage still work? | Ranking quality of the bad/good ratio head (ROC AUC, average precision, recall at a fixed review budget) |
| Does it separate attacks on the same endpoint? | Separability within one URL |
| How much information is lost? | Neighbour overlap against the 768-dimension default; PCA explained variance; intrinsic dimension |
| Do the vectors cluster? | Silhouette score, projection plots |
| Are the vectors well behaved? | Cosine distribution, anisotropy, norms, duplicate rate |
| What does it cost? | Logs per second, time to first vector, energy proxy, storage per log |

Decision rule: take the cheapest candidate whose triage ranking is within an
agreed margin of the default. The margin is not set yet.

## Order of work

1. Deduplication and batching on the default model.
2. Vector cuts to 256 and 128 dimensions on the default model.
3. One small model on the CPU from Rust, as the cost and quality comparison.
4. The Vitis AI route, only if steps 1 to 3 leave a gap worth closing.

## Log

| Date | Entry |
| --- | --- |
| 2026-11-11 | Document created. Default measured at about 3 logs/s, 768 dimensions. No alternative measured. |
| 2026-11-11 | Benchmark run, see [speed and separability](2026-11-11-embedding-speed-benchmark.md). The NPU default is rejected on speed: 2.3 logs/s on real requests, unchanged by batching, concurrency or power mode. The small CPU models were also measured and separate attacks worse than a character n-gram baseline. The position and order of work above are superseded; the open question is now the representation, not the model size. |

| 2026-11-11 | Direction agreed: character-level features for the baseline ratio; a small encoder trained on log text as the research question; larger models only later in the triage flow. Spike: a 1M-parameter byte-level convolutional encoder (random weights) runs at about 36,700 logs/s through the Vitis AI provider. Quality untested. |

## Sources

- FastFlowLM EmbeddingGemma model card: https://fastflow-page.pages.dev/docs/models/embeddinggemma
- EmbeddingGemma overview: https://ai.google.dev/gemma/docs/embeddinggemma
- BERT compilation report: https://github.com/amd/RyzenAI-SW/issues/312
- Ryzen AI model deployment: https://ryzenai.docs.amd.com/en/1.5/modelrun.html
- ONNX Runtime Vitis AI provider: https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html
- Rust `ort` crate feature list (old version): https://docs.rs/crate/ort/1.14.1
