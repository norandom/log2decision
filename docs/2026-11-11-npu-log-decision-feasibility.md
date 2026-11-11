# Feasibility: triaging text logs with NPU embeddings and small decision heads

Date: 2026-11-11 · Status: superseded in part, see the first section · Spec: `.kiro/specs/npu-log-decision/`

This is where the project started: could a laptop NPU produce embeddings for
log lines, and could a small decision layer on top rank those logs for an
analyst? The short answer on the day was yes. Within hours the benchmark showed
the embedding model was the wrong tool, and the plan changed. The document is
kept as the record of the starting point.

## What became of this plan

| Part | Verdict that morning | What happened |
| --- | --- | --- |
| Embeddings on the NPU | Works | The route works, but at 2.3 logs/s it is useless for triage. Dropped |
| General text embeddings as the representation | Assumed | They separate attacks worse than character counts. Dropped as the basis of the score |
| Vector assessment tools | Feasible | Kept; in the spec as requirement 8 |
| Vector plus text database | Feasible, LanceDB | Kept, but the store became SQLite with its text-search extension |
| Venn as shipped | Not feasible | Unchanged |
| Venn's method, rebuilt in Rust | Feasible | Kept: nearest labelled logs as evidence, plus a calibration layer and a shift test |
| Speed | Tight | The deciding problem, see the [benchmark](2026-11-11-embedding-speed-benchmark.md) |

What replaced the embedding model is a character-level baseline scorer and,
behind a gate, a small encoder trained on log text.

## Why the question came up

The log2ml thesis classified Sysmon logs with a Linformer encoder and TPOT in
Python notebooks. Three things are different now. The scope is any text log:
web requests, attack traffic, CTF material. The output is a ratio of bad to
good per log, used to score logs down, not a verdict. And it should run locally
at low energy, in Rust.

So the question was whether three pieces fit together: an embedding route that
uses only the NPU, a database that searches both vectors and text, and the
Vector Decision Model idea from Venn.

## 1. Embeddings on the NPU

`../memguard_alpha` has two NPU routes on this machine (AMD Ryzen AI 9 HX 370,
NPU driver 32.0.20102.3930). Both had only been used for chat models.

| Route | How a model is loaded | Use here |
| --- | --- | --- |
| FastFlowLM 1.0.12 | `flm serve` starts a local OpenAI-compatible server | Chosen: has an `--embed` option |
| Ryzen AI OGA 0.18.0 | `og.Model(dir)` in a Python environment | Not used: generative runtime, Python only |

Measured with `flm serve --embed embed-gemma:300m` on port 11560:

- The embedder runs without a chat model next to it.
- `POST /v1/embeddings` returns 768 dimensions, norm 0.997, and the same vector
  on repeat.
- The server log shows an NPU lock and release for each request.
- 40 single requests took 13.3 s, about 3 logs/s. The client was PowerShell and
  nothing was tuned.

NPU utilization, power draw, batch throughput and long inputs were not
measured. The model is 615 MB and lives in `./models`, set through
`FLM_MODEL_PATH`.

## 2. Raw similarity follows the endpoint, not the attack

Four sample web requests went through the embedder in the same run:

| Pair | Cosine |
| --- | ---: |
| Normal request and its SQL-injection variant, same URL | 0.90 |
| Normal request and a normal request to another page | 0.40 |
| SQL injection and cross-site scripting, different URLs | 0.34 |

Four lines are an observation and nothing more. But they point the wrong way
for a nearest-neighbour vote: the neighbours of an injected request are the
normal requests to the same page. This was the first sign of the problem the
benchmark later measured properly, and it is why the spec now evaluates on
endpoints the scorer has never seen.

## 3. Vector assessment: what the thesis has and what is missing

The thesis notebooks contain these checks, as notebook cells:

| Group | In the notebooks |
| --- | --- |
| Information loss | PCA explained-variance ratio; StandardScaler versus QuantileTransformer before PCA; reduction from 30,000 to 100 dimensions before classification; fixed 700-token truncation |
| Clustering capability | KMeans with silhouette score; UMAP projection; t-SNE (GCN/BERT prototype) |
| Properties | Cosine-similarity distribution (Linformer versus LongFormer); raw versus normalized vectors in FAISS; shape checks |
| Downstream | TPOT classification with recall, F1, ROC AUC, confusion matrix, imbalance handling |

Missing, and added to the spec:

- What shortening a vector costs.
- What quantization costs: NPU vectors against a float reference of the same
  model.
- Label separability, overall and within one endpoint.
- Intrinsic dimension, anisotropy, duplicate rate.

## 4. Storage

The store has to answer vector searches and text queries over the same rows.

| Store | Vector and text search | Fit |
| --- | --- | --- |
| LanceDB | Both, in one embedded store | Recommended that day: Rust-native, no server |
| Qdrant | Both | Separate server; good Rust client |
| OpenSearch | Both | Closest to the thesis's Elasticsearch work; heavy |
| OpenObserve | Text yes; vector unknown | Two searches found no evidence of vector support |

Nothing was loaded into any store. The spec later settled on SQLite, which is
smaller and synchronous.

## 5. Decisions: rebuild Venn's method, do not use its head

Venn calls itself a Vector Decision Model: a frozen embedding model plus small
local heads that route, score and compare populations. It is a research preview
in Python with PyTorch, and it cannot be used here as shipped:

- Its published head is tied to EmbeddingGemma 2 in float32. The documentation
  says other vectors "cannot be mixed with `venn-1` stores or used with the
  published generic head".
- It has no NPU, ONNX or quantized backend.
- Only its in-process Python outcome store accepts vectors from another model.

The methods behind it are small, though, and can be rebuilt:

1. A ratio per log, learned from labelled good and bad logs, with the option to
   abstain.
2. Evidence: the nearest labelled logs, shown with the score.
3. A population-shift test between two groups of logs, for example today's
   logs against a known-good baseline.

If prompts to a local language model are needed later, for instance to explain
a decision, `dspy-rs` is a candidate. Version 0.7.3 has the COPRO, MIPROv2 and
GEPA optimizers, and its README warns that the API changes between versions. It
has not been tried.

## 6. Data: CSIC 2010 first, AIT V2 second

HTTP CSIC 2010 is in `data/csic-2010/` (Kaggle upload by ispangler).

- 61,065 requests: 36,000 normal, 25,065 anomalous.
- 43,088 GET, 17,580 POST, 397 PUT.
- Columns include method, URL, headers, body and a 0/1 label.
- The requests are generated, not captured, and target one shop application.
  Results will not carry over to other applications without a second dataset.

AIT Log Data Set V2 (Zenodo) is step 2: eight testbeds, several log types,
attack labels per line. Splunk BOTS v1 was considered as a CTF source and set
aside. Its answer key is given on request only, and its answers are not per-log
labels.

## What would have proved this wrong, and what did

Three conditions were written down that morning.

- A fitted ratio on CSIC 2010 fails to separate attacks from normal requests on
  the same endpoint. Not tested in this form; the spec now tests it with
  grouped splits.
- NPU throughput stays near 3 logs/s after batching and deduplication, and a
  small model on the CPU does as well for less. This is what happened. Batching
  changed nothing, and the cheapest method tried was also the best.
- Quantized NPU vectors differ enough from a float reference to change
  rankings. Never reached; the model was dropped first.

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
