# Feasibility: triaging text logs with NPU embeddings and small decision heads

Date: 2026-11-11 · Status: research, first measurements · Spec: `.kiro/specs/npu-log-decision/`

> Update, same day: the embedding model measured here is too slow for triage
> (2.3 logs/s on real requests), and general text embeddings separate attacks
> worse than a character n-gram baseline. See
> [Embedding speed and separability on CSIC 2010](2026-11-11-embedding-speed-benchmark.md).
> The verdicts "Works" and "Tight" in the table below describe the route, not
> its fitness for this workload.

## Summary

The plan is feasible on this laptop. An embedding model already runs on the NPU
and answers over local HTTP, so a Rust program can use it without any Python.
Venn's published decision head cannot read those vectors, so the decision part
has to be built here; the methods behind it are small. The largest open risk is
not the hardware. It is that raw embedding similarity follows the URL of a web
request more than the attack inside it, so the bad/good ratio needs a fitted
head.

| Part | Verdict | Basis |
| --- | --- | --- |
| Embeddings on the NPU | Works | Measured on this machine |
| Vector assessment tools | Feasible | Methods exist in the thesis notebooks; port needed |
| Vector plus text database | Feasible | LanceDB recommended; not yet tried |
| Venn as shipped, on NPU vectors | Not feasible | Venn documentation |
| Venn's method, rebuilt in Rust | Feasible | Methods are logistic regression, nearest neighbours, a permutation test |
| Speed | Tight | About 3 logs/s unoptimized |

## Why this report exists

The log2ml thesis classified Sysmon logs with a Linformer encoder and TPOT in
Python notebooks. The goal now is different in three ways. The scope is any text
log (web, attack traffic, CTF material). The output is a ratio of bad to good
per log, used to score logs down for triage, not a verdict. And the work should
run locally at low energy, in Rust.

The question was whether an NPU-only embedding route, a vector database with
text search, and the Vector Decision Model idea from Venn fit together.

## 1. Embeddings on the NPU work today

`../memguard_alpha` has two NPU routes on this machine (AMD Ryzen AI 9 HX 370,
NPU driver 32.0.20102.3930). Both had only been used for chat models.

| Route | How a model is loaded | Use here |
| --- | --- | --- |
| FastFlowLM 1.0.12 | `flm serve` starts a local OpenAI-compatible server | Chosen: has an `--embed` option |
| Ryzen AI OGA 0.18.0 | `og.Model(dir)` in a Python environment | Not used: generative runtime, Python only |

Measured on 2026-11-11 with `flm serve --embed embed-gemma:300m` on port 11560:

- The embedder runs without a chat model next to it.
- `POST /v1/embeddings` returns 768 dimensions, norm 0.997, and the same vector
  on repeat.
- The server log shows an NPU lock and release for each request.
- 40 single requests took 13.3 s, about 3 logs/s. The client was PowerShell and
  nothing was tuned.

Not measured: NPU utilization, power draw, batch throughput, long inputs.

The model is stored in `./models` (615 MB), set through `FLM_MODEL_PATH`.

## 2. Raw similarity follows the endpoint, not the attack

Four sample web requests were embedded in the same run:

| Pair | Cosine |
| --- | ---: |
| Normal request and its SQL-injection variant, same URL | 0.90 |
| Normal request and a normal request to another page | 0.40 |
| SQL injection and cross-site scripting, different URLs | 0.34 |

Four lines are an observation, not a result. But the direction matters: a
plain nearest-neighbour vote would find the neighbours of an injected request
among the normal requests to the same page. The ratio therefore has to come
from a head fitted on labelled examples, or from comparing a request with
others on the same endpoint. The assessment tools in section 3 should test this
first on the full dataset.

## 3. Vector assessment: what the thesis has and what is missing

The thesis notebooks contain these checks as notebook cells:

| Group | In the notebooks |
| --- | --- |
| Information loss | PCA explained-variance ratio; StandardScaler versus QuantileTransformer before PCA; reduction from 30,000 to 100 dimensions before classification; fixed 700-token truncation |
| Clustering capability | KMeans with silhouette score; UMAP projection; t-SNE (GCN/BERT prototype) |
| Properties | Cosine-similarity distribution (Linformer versus LongFormer); raw versus normalized vectors in FAISS; shape checks |
| Downstream | TPOT classification with recall, F1, ROC AUC, confusion matrix, imbalance handling |

To add for this project:

- Loss from shortening the vector: EmbeddingGemma is trained so that the first
  512, 256 or 128 dimensions can be used alone. Measure what each cut costs.
- Loss from quantization: compare the NPU vectors with a float reference of the
  same model (neighbour overlap, cosine).
- Label separability, overall and within one endpoint (follows from section 2).
- Intrinsic dimension, anisotropy, duplicate rate.

UMAP and t-SNE have weak Rust support. Projection plots can stay a small Python
script run with `uv`; the numeric checks belong in Rust.

## 4. Storage: LanceDB first

The store has to answer vector searches and text queries over the same rows.

| Store | Vector and text search | Fit |
| --- | --- | --- |
| LanceDB | Both, in one embedded store | Recommended: Rust-native, no server, Venn has an adapter |
| Qdrant | Both | Separate server; good Rust client |
| OpenSearch | Both | Closest to the thesis's Elasticsearch work; heavy |
| OpenObserve | Text yes; vector unknown | Two searches found no evidence of vector support |

Nothing has been loaded into any store yet.

## 5. Decisions: rebuild Venn's method, do not use its head

Venn calls itself a Vector Decision Model: a frozen embedding model plus small
local heads that route, score and compare populations. It is a research
preview in Python with PyTorch.

Why it cannot be used directly:

- Its published head is tied to EmbeddingGemma 2 in float32. The documentation
  says other vectors "cannot be mixed with `venn-1` stores or used with the
  published generic head".
- It has no NPU, ONNX or quantized backend.
- Only its in-process Python outcome store accepts vectors from another model.

What to build instead, in Rust, on the NPU vectors:

1. A ratio head: logistic regression on labelled good and bad logs. Its output
   is the bad/good ratio used to rank and score down logs. It can abstain when
   a log is far from everything it has seen.
2. Evidence: the nearest labelled logs from the store, shown with the score.
3. Population shift: an RBF MMD permutation test between two groups of logs,
   for example today's logs against a known-good baseline.

Venn on CPU with its own reader can serve as a reference on the same samples.
That is optional and would be the only Python component.

For later, if prompts to a local language model are needed (for example to
explain a triage decision), `dspy-rs` is a candidate. Version 0.7.3 has the
COPRO, MIPROv2 and GEPA optimizers; its README warns that the API changes
between versions. It has not been tried here.

## 6. Data: CSIC 2010 first, AIT V2 second

HTTP CSIC 2010 is downloaded to `data/csic-2010/` (Kaggle upload by ispangler).

- 61,065 requests: 36,000 normal, 25,065 anomalous.
- 43,088 GET, 17,580 POST, 397 PUT; 13,498 distinct URLs.
- Columns include method, URL, headers, body and a 0/1 label.
- The requests are generated, not captured, and target one shop application.
  Results on it will not carry over to other applications without a second set.

At 3 logs/s the full set takes about 5.7 hours to embed. Deduplicating
identical requests and batching should come before any full run.

AIT Log Data Set V2 (Zenodo) is step 2: eight testbeds, several log types,
attack labels per line. Splunk BOTS v1 was considered as a CTF source and set
aside, because its answer key is given on request only and its answers are not
per-log labels.

## What would prove this wrong

- A fitted ratio head on CSIC 2010 does not separate attacks from normal
  requests on the same endpoint. Then the embedding model, the input format, or
  the approach has to change.
- NPU throughput stays near 3 logs/s after batching and deduplication, and a
  small model on the CPU reaches the same quality with less energy. Then the
  NPU is the wrong place for this workload. See
  [the model choice document](2026-11-11-embedding-model-choice.md).
- The quantized NPU vectors differ enough from the float reference to change
  triage rankings.

## Next steps

1. Generate requirements for the spec (`/kiro-spec-requirements npu-log-decision`).
2. Embed a stratified, deduplicated CSIC sample and run the separability checks
   from section 3.
3. Fit the first ratio head and report ranking quality.
4. Use the same vectors to settle the model choice.

## Sources

- Local evidence: `../memguard_alpha/scripts/serve_local_flm.ps1`,
  `../memguard_alpha/scripts/score_documents_oga.py`,
  `../memguard_alpha/docs/results/2026-11-08-gemma4-document-likelihood-npu.md`,
  notebooks under `log2ml/`.
- Venn VDM: https://markrussinovich.github.io/venn-vdm/
- FastFlowLM EmbeddingGemma model card: https://fastflow-page.pages.dev/docs/models/embeddinggemma
- HTTP CSIC 2010: https://www.tic.itefi.csic.es/dataset/ and
  https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks
- AIT Log Data Set V2.0: https://zenodo.org/records/5789064
- Splunk BOTS v1: https://github.com/splunk/botsv1
- dspy-rs: https://docs.rs/crate/dspy-rs/0.7.3
