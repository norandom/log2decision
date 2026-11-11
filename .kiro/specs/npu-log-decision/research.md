# Research & Design Decisions

## Summary
- **Feature**: `npu-log-decision`
- **Discovery Scope**: New Feature (full discovery)
- **Key Findings**:
  - A small convolutional encoder runs on this laptop's NPU through ONNX Runtime with AMD's Vitis AI provider at about 36,700 logs/s (measured, random weights). The Rust `ort` crate exposes that provider and can load AMD's runtime DLL, but nobody has documented the combination; it must be proven first.
  - Burn 0.22.1 can train the encoder in Rust. Its ONNX exporter is experimental, opset 18 only, and has no quantization, so this project writes the quantized ONNX file itself.
  - LanceDB's Rust API covers vector search, text-filtered search, scalar filters and exact search in one embedded store. The build is heavy and releases are frequent, so the version is pinned and a Windows build check comes early.

Skills consulted: `kiro-spec-design` rules (design principles, full discovery, synthesis, review gate). No steering files exist.

## Research Log

### Measurements on this host (2026-11-11)
- **Context**: Whether NPU embedding is fast enough, and whether embeddings help.
- **Sources Consulted**: `docs/2026-11-11-embedding-speed-benchmark.md`.
- **Findings**:
  - EmbeddingGemma 300m through FastFlowLM: 2.3 logs/s; batching, concurrency and power mode do not help.
  - ROC AUC of a logistic regression on distinct CSIC 2010 requests: hashed character 1-3 grams 0.962 (4,000 texts) and 0.910 (1,000 texts); EmbeddingGemma 0.792; small CPU transformers 0.78 to 0.79.
  - Byte-level convolutional encoder, 976,000 parameters, input `[32, 64, 1, 256]`, 8-bit QDQ, `onnxruntime-vitisai` 1.30.0: 36,713 logs/s at 27 % of one core; CPU provider 4,467 logs/s at 484 %. First load 3 to 5 s. NPU and CPU outputs agree at cosine 0.992 or better.
  - Plain ONNX Runtime static quantization (uint8 activations, int8 weights, per tensor) was accepted by the provider.
- **Implications**: The character-level baseline is a product feature, not only a comparison. The encoder is convolutional, fixed-shape and 8-bit. Cross-validation in those runs was not grouped by URL; the design evaluates with grouped splits.

### Vitis AI provider from Rust
- **Context**: The tools must run without Python (10.1).
- **Sources Consulted**: docs.rs `ort` 2.0.0-rc.15 (`ort::ep::Vitis`, feature list), pykeio/ort sources (`src/ep/mod.rs`, `src/ep/vitis.rs`, linking and multiversion docs), AMD Ryzen AI docs (installation, application development, model deployment, quantization).
- **Findings**:
  - `ort::ep::Vitis` exists behind the `vitis` feature, with `with_cache_dir`, `with_cache_key`, `with_cache_in_memory`, `with_target`, and arbitrary key/value options. Registration is a by-name call into the loaded runtime: `VitisAIExecutionProvider` with feature `api-22` or higher, `VitisAI` below.
  - `load-dynamic` plus `ORT_DYLIB_PATH` (or `init_from`) loads a supplied `onnxruntime.dll`. `ort` supports runtime versions 1.17 to 1.31 and refuses a DLL older than the selected `api-NN` feature.
  - Provider registration fails silently by default; `error_on_failure()` makes it an error.
  - AMD lists eight DLLs to ship with an application for 8-bit models and offers them as a NuGet package (`RyzenAI_Deployment.1.9.0.nupkg`) besides the MSI. On this processor family no `xclbin` option is set and `target` defaults to `X2`.
  - The placement report needs the cache on disk and the environment variable `XLNX_ONNX_EP_REPORT_FILE`.
  - AMD accepts third-party QDQ quantization for 8-bit activations and weights with float scales.
  - Not found: a documented package index for AMD's Python wheels, and the exact ONNX Runtime version AMD's DLL reports.
- **Implications**: No Python is needed at runtime or for the runtime files. The `ort` feature level is chosen from the DLL's reported version during the first task. The NuGet package content is unverified and is checked in that task.

### Training and ONNX writing in Rust
- **Context**: Rust training is preferred; the NPU needs a fixed-shape 8-bit ONNX file.
- **Sources Consulted**: Burn 0.22 book and release notes, burn-store and burn-onnx docs, Candle docs, ONNX operator specifications (QuantizeLinear, DequantizeLinear, QLinearConv), ONNX Runtime quantization docs, crates.io metadata.
- **Findings**:
  - Burn 0.22.1 has Embedding, Conv2d, pooling, Linear, cross-entropy, AdamW, autodiff, a pure-Rust CPU backend (`Flex`) and GPU backends through wgpu and Vulkan. Weights can be read out by name through burn-store.
  - Open reports describe wrong or non-repeatable results on the Vulkan path. No evidence either way for the Radeon 890M.
  - burn-onnx 0.22 exports ONNX experimentally: opset 18 only, no QDQ.
  - Candle trains on CPU only on this hardware and has no exporter. tch-rs needs libtorch and cannot export ONNX without Python.
  - `tract-onnx` exposes prost types for `ModelProto` that serialize; generating types from `onnx.proto` with prost is the lean alternative.
  - QDQ rules: weights as int8 initializer followed by DequantizeLinear; bias as int32 with scale `input_scale * weight_scale` and zero point 0; activations wrapped in QuantizeLinear and DequantizeLinear; opset 13 definitions apply at opset 17.
- **Implications**: Train on the CPU backend for repeatability (3.5); GPU training is an opt-in feature. The project owns a small ONNX writer and a quantizer for one fixed architecture.

### Store and analysis crates
- **Context**: Requirements 5 and 8 in pure Rust on Windows.
- **Sources Consulted**: docs.rs `lancedb` 0.41.0, LanceDB docs (full-text search, hybrid search, scalar indexes), LanceDB CI workflow, linfa, faer, bhtsne, sqlite-vec, DuckDB vss, Microsoft Energy Meter Interface docs.
- **Findings**:
  - LanceDB: `nearest_to` with `distance_type` and `_distance`; `only_if` filters applied before the vector search by default; `bypass_vector_index` for exact search; `Index::NGram` accelerates `contains(text, '...')`; `Index::Bitmap` for low-cardinality columns; BM25 search through `Index::FTS`. The API is async. Its CI installs `protoc` on Windows.
  - linfa provides KMeans, PCA, silhouette and L2 logistic regression in pure Rust; its logistic regression is dense only. No crate offers a character n-gram hashing vectorizer or a maintained TwoNN estimator.
  - UMAP crates are immature; `bhtsne` provides t-SNE.
  - Process CPU time is always readable without elevation. The Energy Meter Interface may give real energy on Windows 11, but availability on this laptop is unconfirmed.
- **Implications**: LanceDB is the store. The hashing vectorizer, a sparse logistic regression and TwoNN are written in the project. The projection uses t-SNE on a sample. The energy proxy is CPU seconds.

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Layered workspace of library crates plus one CLI | One crate per responsibility, one-way dependencies | Parallel work per crate; heavy dependencies (store, training) isolated; each crate testable alone | More manifests | Selected |
| Single crate with modules | Everything in one package | Least ceremony | Every change recompiles Burn and LanceDB; boundaries not enforced by the compiler | Rejected |
| Local service plus thin clients | A long-running process owning the NPU | Model stays loaded | A second process to manage; not needed at 3 to 5 s load | Rejected; revisit only if load time matters |

## Design Decisions

### Decision: One generalized "scorer" contract for baseline and encoder-based ratios
- **Context**: Requirements 2, 6 and 9 all need a bad/good ratio per log, from two different sources.
- **Alternatives Considered**:
  1. Separate code paths for baseline and encoder-based triage.
  2. One `Scorer` contract returning a ratio or an abstention, with two implementations.
- **Selected Approach**: Option 2. Ranking, review budget and evaluation consume the contract.
- **Rationale**: Evaluation compares scorers on equal terms (9.5); triage can report one or both (6.6).
- **Trade-offs**: Evidence exists only for the encoder-based scorer; the contract makes evidence optional.
- **Follow-up**: None.

### Decision: Encoder-based ratio is the mean outcome of nearest labelled logs
- **Context**: Requirement 6 and the Venn outcome-store operation.
- **Alternatives Considered**:
  1. A fitted linear head on the vectors.
  2. Mean label of the k nearest labelled logs within a reach radius, with minimum support.
- **Selected Approach**: Option 2. The encoder is trained with labels, so the vector space is shaped for the task and neighbours carry the signal.
- **Rationale**: It yields the ratio, the evidence and the abstention rule from one operation, and matches Venn's nearest-outcome semantics.
- **Trade-offs**: Quality depends on the encoder; a general encoder would follow the endpoint, as measured with EmbeddingGemma.
- **Follow-up**: Evaluate within-group separability (8.4) before trusting it.

### Decision: Supervised training objective for the first encoder
- **Context**: Requirement 3 leaves the objective open.
- **Alternatives Considered**:
  1. Self-supervised objective on unlabelled logs.
  2. Binary classification on labelled logs; the layer before the classifier is the vector.
- **Selected Approach**: Option 2.
- **Rationale**: Smallest thing that can beat the baseline on CSIC 2010; one loss, one dataset.
- **Trade-offs**: The encoder is specific to the labelled data. A self-supervised objective is a later change and a revalidation trigger.
- **Follow-up**: Train only on the fitting side of each evaluation split to avoid leakage.

### Decision: Build the ONNX writer and quantizer; adopt everything else
- **Context**: Build-versus-adopt review.
- **Selected Approach**: Adopt Burn (training), ONNX Runtime with AMD's provider (inference), LanceDB (store), linfa and faer (analysis), bhtsne (projection), clap and serde (CLI). Build: ONNX writer and QDQ quantizer for the one architecture, hashing vectorizer, sparse logistic regression, nearest-outcome ratio, MMD permutation test, TwoNN.
- **Rationale**: Each built part is small and has no fitting crate; the exporter in burn-onnx cannot produce the required file.
- **Trade-offs**: The writer is tied to one architecture. A different architecture means extending it.
- **Follow-up**: Validate written files by loading them with the runtime in tests.

### Decision: Simplifications
- No BM25 ranking: text search is containment (5.3), served by `contains` filters.
- No approximate vector index: exact search up to one million rows of 128 dimensions.
- One store holds vectors of one encoder identity.
- No Energy Meter Interface probe: CPU seconds is the proxy (9.4).
- No GPU training by default.
- No local service process.

## Risks & Mitigations
- Rust cannot load AMD's provider, or the DLL version does not fit an `ort` feature level — first task is a spike with a pass/fail gate; if it fails, the design returns to the user before further encoder work.
- The NuGet package lacks a needed DLL — the same spike checks it; the MSI install is the documented alternative source.
- A trained encoder does not beat the baseline — the baseline scorer is a complete product path on its own; the evaluation states the difference (9.5).
- Hand-written quantization loses accuracy or lands nodes on the CPU — tests compare float and 8-bit outputs and read the placement report.
- Burn's CPU backend trains too slowly — measure in the first training task; GPU feature is the fallback, with the repeatability criterion checked again.
- LanceDB build problems on Windows — early build task; SQLite with trigram text search and a flat vector scan is the recorded fallback.
- CSIC 2010 Kaggle upload licence unchecked — data stays untracked.

## References
- [ort crate, execution providers](https://docs.rs/ort/latest/ort/ep/index.html) — Vitis provider type and options
- [ort linking and multiversion docs](https://github.com/pykeio/ort/tree/main/docs/content/setup) — `load-dynamic`, API levels
- [Ryzen AI model deployment](https://ryzenai.docs.amd.com/en/latest/modelrun.html) — provider options, placement report
- [Ryzen AI application development](https://ryzenai.docs.amd.com/en/latest/app_development.html) — DLLs to ship
- [Ryzen AI quantization](https://ryzenai.docs.amd.com/en/latest/model_quantization.html) — accepted quantization types
- [Burn 0.22 book](https://burn.dev/books/burn/building-blocks/backend.html) — backends, modules, optimizers
- [burn-onnx export](https://docs.rs/burn-onnx/latest/burn_onnx/export/index.html) — exporter limits
- [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html), [DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html), [QLinearConv](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) — QDQ rules
- [lancedb crate](https://docs.rs/lancedb/latest/lancedb/) — query, filter and index API
- [LanceDB scalar indexes](https://docs.lancedb.com/indexing/scalar-index) — NGram and Bitmap
- [linfa](https://github.com/rust-ml/linfa) — KMeans, PCA, silhouette, logistic regression
- [Venn VDM](https://markrussinovich.github.io/venn-vdm/) — nearest-outcome and population-shift semantics
