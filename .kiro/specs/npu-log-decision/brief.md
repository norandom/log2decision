# Brief: npu-log-decision

Input and decision record for this spec, moved here from `requirements.md` when the requirements were generated on 2026-11-11. Later sections supersede earlier ones where they conflict; "Direction after the benchmark" is the current direction.

### Who has the problem
The author of the log2ml master thesis ("Development and Evaluation of Software
for Forensic Log-Analysis Using Machine Learning and Genetic Programming"),
acting as a forensic log analyst who wants to triage text logs locally on a
low-energy laptop NPU: each log gets a bad-versus-good ratio, and logs that look
good are scored down so attention goes to the rest. The scope is general text
logs (web, attack traffic, CTF material), no longer only Sysmon.

### Current situation
- `log2ml/` holds the thesis as Python notebooks: a custom BPE tokenizer and an
  untrained-for-the-task Linformer/LongFormer encoder produce 30,000-dimensional
  vectors (13,455 rows, labels `good`/`bad`, heavily imbalanced), classified with
  TPOT AutoML, searched with FAISS through LangChain. Everything runs on CPU/GPU
  in Colab-style notebooks; nothing is reusable as a tool.
- Vector assessment exists only as notebook cells: PCA explained variance, UMAP
  and t-SNE projection, KMeans silhouette score, cosine-similarity distribution,
  StandardScaler versus QuantileTransformer scaling, normalized versus raw
  vectors in FAISS.
- `../memguard_alpha` has two working NPU routes on this machine (AMD Ryzen AI 9
  HX 370, NPU driver 32.0.20102.3930): FastFlowLM 1.0.12 (`flm serve`,
  OpenAI-compatible HTTP on 127.0.0.1) and Ryzen AI OGA 0.18.0
  (`onnxruntime-genai-directml-ryzenai`). Both were used for chat/logit scoring
  only; no embedding model has been run on the NPU yet. FastFlowLM lists
  `embed-gemma:300m` (not downloaded) and a `--embed` serve option.

### What should change
A reusable, local pipeline, implemented in Rust, that:

1. Produces text embeddings for log messages on the NPU (first candidate:
   FastFlowLM `--embed embed-gemma:300m`, `POST /v1/embeddings`; fallback: an
   ONNX embedding model through the Ryzen AI runtime), and records model
   identity, runtime, driver and observed NPU placement for each run.
2. Provides reusable vector-assessment tools in three groups:
   - information loss: PCA explained variance and intrinsic dimension, effect of
     scaling and normalization, truncation of long messages, drift of the
     quantized NPU vectors against a float reference;
   - clustering capability: KMeans silhouette, 2-D projection, separability of
     the labelled classes;
   - properties: cosine-similarity distribution, norms, anisotropy, duplicate
     rate, dimensionality.
3. Stores embeddings together with raw log text and metadata in a database that
   answers both vector searches and text search queries (first candidate:
   LanceDB, embedded and Rust-native; alternatives Qdrant and OpenSearch;
   OpenObserve only if vector support is confirmed).
4. Applies the Vector Decision Model approach of Venn
   (https://markrussinovich.github.io/venn-vdm/): a frozen embedding reader plus
   small local decision heads. From labelled samples of good and bad logs it
   gives every new log a bad-versus-good ratio used to rank and score down logs
   for triage, returns the nearest labelled logs as evidence, can abstain when
   neither class is close, and tests population shift between groups of logs
   (RBF MMD permutation test). The Python Venn package on CPU with its pinned
   reader is an optional reference baseline, since its published head cannot
   consume NPU vectors from another reader.
5. Reports feasibility: throughput, latency, an energy proxy, and triage quality
   (ranking and imbalance-aware metrics) on a public labelled dataset.

### Decisions (2026-11-11)
- Output is a bad-versus-good ratio per log for triage, not a hard verdict.
- Rust is the implementation language.
- The thesis Sysmon dataset is not the evaluation set. A public labelled log
  dataset is used instead; candidates: HTTP CSIC 2010 (Kaggle upload, web
  requests labelled normal/anomalous), AIT Log Data Set V2 (Zenodo, line-level
  attack labels across host logs), Splunk BOTS v1 (CTF, CC0; answer key on
  request only).

- HTTP CSIC 2010 is step 1 and is downloaded to `data/csic-2010/` (61,065
  rows: 36,000 normal, 25,065 anomalous). AIT Log Data Set V2 is step 2.
- The embedding model is chosen by measurement with the assessment tools; see
  `docs/2026-11-11-embedding-model-choice.md`. EmbeddingGemma 300m on the NPU
  is the working default.
- Python is not developed here; `uv` may run one-off analysis scripts.
  `dspy-rs` is a candidate if prompts to a local language model are needed
  later.
- Documents go to `docs/` with a date prefix and are indexed in `README.md`;
  datasets go to `data/<dataset>/` and are registered in `data/README.md`.

### Direction after the benchmark (2026-11-11)
See `docs/2026-11-11-embedding-speed-benchmark.md`. This supersedes the
EmbeddingGemma route in "What should change" item 1.
- EmbeddingGemma on the NPU is rejected: 2.3 logs/s, and general text
  embeddings separate attacks worse than character n-grams (ROC AUC 0.79
  against 0.91 to 0.96 on CSIC 2010).
- Baseline: the bad/good ratio comes from character-level features.
- Research question: a small encoder trained on log text, run on the NPU
  through ONNX Runtime with the Vitis AI provider. A 1M-parameter byte-level
  convolutional encoder with random weights reached about 36,700 logs/s there.
  Its quality after training is unknown.
- Larger models (language models on the NPU) come later in the triage flow, for
  the logs that survive the cheap stages.
- The encoder is meant to act as the frozen reader of a Vector Decision Model:
  fixed revision, fixed dimensions, small heads and nearest-outcome evidence on
  top.

- The tools have no Python dependency at all at runtime. Python is allowed during
  development in a project `uv` environment to train, validate model architectures
  and produce quantized ONNX artifacts before committing to NPU deployment.
- The encoder is deployed as a verified artifact (`model.int8.onnx`, `table.bin`,
  `encoder.json`) loaded and executed in Rust via `ort` with the Vitis AI execution provider.
  In-Rust training via Burn is an optional post-baseline feature, not a blocker for
  runtime triage.
- Storage uses an embedded SQLite database (`rusqlite` with FTS5) combined with
  in-memory / SIMD exact vector search via `faer`, eliminating heavy async runtimes (LanceDB/Tokio).
- The Venn operations that are needed are ported to Rust: nearest-outcome
  ratio with evidence, fitted heads, population shift. The Python Venn package
  is at most a development-time cross-check.
- Agnostic threat scoring barrier: Log features and decision heads must not hardcode
  or overfit to specific training endpoints, hostnames, or site topography. The tool
  must enforce cross-endpoint generalization (grouped holdouts) and separate routing
  context from payload anomaly signatures so threat scores measure threat characteristics
  rather than training set specifics.

### Open points to settle in the requirements phase
- Agnostic threat scoring: mechanism to isolate or normalize endpoint topology so scores
  reflect attack payload and syntactic anomaly patterns across unseen endpoints.
- Model architecture validation: verify that the small convolutional encoder matches or
  exceeds baseline ROC AUC on cross-endpoint holdout splits before final NPU freezing.
- Whether Rust can load the Vitis AI provider directly (ONNX Runtime DLLs from
  the Ryzen AI packages) or needs a local service.
- The package index for the Ryzen AI runtime is not recorded in
  `../memguard_alpha`; it has to be found before the project environment is
  built.
- The licence of the CSIC 2010 Kaggle upload has not been checked.
- Whether the ratio is also reported per batch or source (share of bad logs in
  a window), in addition to per log.

### Host rules and first measurement (2026-11-11)
- Agents use PowerShell only on this host: no Git Bash, no `wsl` commands.
  Containers run through `docker` / `docker compose` (Podman backend). The NPU
  is reachable only natively on Windows.
- Model weights live in `./models` (`FLM_MODEL_PATH`), never on C:.
- Spike: `flm serve --embed embed-gemma:300m` runs without a chat model and
  answers `POST /v1/embeddings` with 768-dimensional, near unit-norm,
  repeatable vectors; the server log shows the NPU lock per request. About
  3 logs/s with one short request line per call, unoptimized. NPU utilization
  and energy were not measured.
- In that spike a normal request and its SQL-injection variant on the same URL
  had cosine 0.90, while two attacks on different URLs had 0.34: raw similarity
  follows the endpoint more than the attack. The ratio must come from a fitted
  head or a per-endpoint comparison, not from raw nearest neighbours alone.
