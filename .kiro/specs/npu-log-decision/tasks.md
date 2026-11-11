# Implementation Plan: npu-log-decision

## Milestone 1: Working Baseline Triage & Core Infrastructure

- [ ] 1.1 Set up the Rust workspace and the shared core types
  - Create the Cargo workspace with crates: `l2d-core`, `l2d-baseline`, `l2d-store`, `l2d-decide`, `l2d-assess`, `l2d-eval`, `l2d-encoder`, `l2d-cli`, and optional `l2d-train`
  - Define shared record, label (`Good`, `Bad`, `Unknown`), encoder identity, run record, and error types (`L2dError`) with exit-code mapping
  - Configure project paths for models (`models/`), data (`data/`), stores (`stores/`), runtime files (`runtime/ryzenai/`), and reports (`reports/`)
  - Set `CARGO_TARGET_DIR` off-repo and ignore runtime, store, and report directories in `.gitignore`
  - Done when workspace builds, tests verify exit code mappings, and missing items report their expected location
  - _Requirements: 10.3, 10.4, 10.5, 10.6_

- [ ] 1.2 Add the command-line skeleton and the test fixture
  - Implement CLI entry point in `l2d-cli` with subcommands stubbed out, `--json` global flag, and stderr diagnostics
  - Create synthetic labelled fixture (`tests/fixtures/mini-labelled.csv`) with 200 request lines: at least 6 group keys with both labels, duplicates, 1 malformed row, and 1 conflicting label
  - Done when `l2d --help` lists subcommands, stub exits non-zero with informative error, and tests verify fixture properties
  - _Requirements: 10.6, 10.7_

- [ ] 1.3 Read, deduplicate, and summarize text logs and labelled datasets
  - Implement `InputAdapter` in `l2d-core` for plain line-based logs and labelled CSV/JSONL with configurable text, label, and group columns
  - Exact text deduplication: keep occurrence counts and first position; conflicting labels become `Label::Unknown`
  - Skip malformed records and report their count and positions in `InputSummary`
  - Compute SHA-256 content hash and record count
  - Done when fixture tests show correct deduplication, malformed row skipping, and summary counts matching total rows
  - _Boundary: Input adapters and dedup_
  - _Depends: 1.2_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6_

- [ ] 1.4 Read HTTP CSIC 2010 with group-key extraction for agnostic splitting
  - Implement CSIC 2010 adapter: format text as `METHOD URL\nBODY`, extract label from classification column, derive group key from `METHOD /path` (excluding query params)
  - Ensure routing structure is separable from payload for agnostic threat analysis
  - Read from `data/csic-2010/` or specified path
  - Done when adapter reports 61,065 records (36,000 good, 25,065 bad) before dedup, and tests verify group key isolation on sample rows
  - _Requirements: 1.2, 9.7, 10.4_

- [ ] 1.5 Hash character n-grams into sparse feature rows with routing isolation option
  - Implement signed FNV-1a hashing of character 1-, 2-, and 3-grams over text bytes into 65,536 buckets with L2 normalization in `l2d-baseline`
  - Provide an option to isolate or strip routing tokens (method/path) to prevent the baseline from memorizing endpoint URLs
  - Done when same text yields bit-identical sparse vectors across processes, and test verifies row values on fixed examples
  - _Boundary: BaselineScorer_
  - _Depends: 1.1_
  - _Requirements: 2.1, 2.4_

- [ ] 1.6 Fit deterministic logistic regression via numerical library
  - Implement L2-penalized logistic regression in `l2d-baseline` using robust numerical solver (`faer` / `linfa`), supporting sparse and dense rows
  - Guarantee bit-identical weights for identical inputs and settings without stochastic line-search instability
  - Done when two fits on the same rows yield bit-equal weights and toy separable data is classified with 100% accuracy
  - _Requirements: 2.1, 2.4_

- [ ] 1.7 Fit, apply, save, and load the baseline scorer
  - Implement `BaselineScorer` in `l2d-baseline`: compute bad/good ratio (0.0 to 1.0)
  - Refusal rule: refuse fitting if fewer than 10 labelled logs or fewer than 5 of either label are provided
  - Save and load scorer with SHA-256 revision hash under `models/baselines/<name>-<rev12>/`
  - Done when fitting succeeds on the fixture without any runtime DLLs, reloaded model reproduces exact ratios, and refusal cases are tested
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 1.8 Create, open, and populate SQLite store with FTS5 and exact vector storage
  - Implement embedded SQLite storage in `l2d-store` using `rusqlite` (bundled, feature `fts5`): `stores/<name>.db`
  - Tables: `logs` (id, text, source, occurrences, label, group_key, truncated, vector BLOB, encoder_revision), `logs_fts` (FTS5 index on text), `meta` (encoder identity)
  - Enforce single encoder identity per store; refuse adding vectors with different identity
  - Adding an existing `id` increments occurrences while leaving text, label, and vector untouched
  - Completely synchronous operation without Tokio async runtime
  - Done when store survives restart, FTS5 queries match text, and identity mismatch is refused with both identities named
  - _Boundary: Store_
  - _Depends: 1.1_
  - _Requirements: 5.1, 5.6, 5.7, 6.7_

- [ ] 1.9 Search by text, metadata, and exact SIMD vector search
  - Implement FTS5 text search and containment query in `l2d-store`
  - Implement exact squared L2 vector distance search using `faer` SIMD dot-products over stored vector BLOBs
  - Combine text containment with distance ranking, and apply SQL index filters for `label` and `source`
  - Implement full vector export for assessment and shift testing
  - Done when tests verify exact nearest search, text search with special characters, combined search, and filtered vector exports
  - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 1.10 Triage ranking, abstention, and review budget
  - Implement triage engine in `l2d-decide`: rank logs descending by ratio, separate abstained logs, name scorer
  - Review budget: return top $N$ logs as review items and mark remaining logs as scored down
  - Postcondition: every input record lands in exactly one list (review, scored_down, abstained), and stored rows are never modified
  - Done when triage on the fixture partitions logs correctly according to budget and scorer attribution
  - _Boundary: Scorer, Triage_
  - _Depends: 1.7, 1.9_
  - _Requirements: 6.4, 6.5, 6.6, 6.7_

- [ ] 1.11 Wire Baseline CLI commands
  - Connect CLI commands in `l2d-cli`: `l2d ingest`, `l2d baseline fit`, `l2d baseline score`, `l2d triage`
  - Support both human-readable text and `--json` machine-readable output
  - Done when ingesting fixture displays summary, fitting then scoring displays ratios, and triage displays ranked review and scored-down lists
  - _Requirements: 1.1, 1.2, 1.6, 2.1, 2.2, 6.4, 6.5, 10.6, 10.7_

- [ ] 1.12 Baseline validation: Grouped cross-validation on CSIC 2010
  - Run 5-fold grouped cross-validation (grouped by endpoint/URL path) using the baseline scorer on CSIC 2010
  - Verify that the baseline achieves cross-endpoint ROC AUC >= 0.90 without memorizing static URLs
  - Done when baseline cross-endpoint benchmark is completed and results are recorded in `reports/`
  - _Requirements: 2.1, 9.1, 9.2, 9.7_

---

## Milestone 2: Model Architecture Validation & NPU Inference

- [ ] 2.1 Science Gate: Offline model architecture validation via development spike
  - In a project-local `uv` environment, train the candidate 1M byte-level CNN on CSIC 2010 using grouped endpoint holdouts
  - Export to ONNX and quantize to 8-bit QDQ using ONNX Runtime tools
  - Evaluate cross-endpoint ROC AUC against the baseline benchmark from 1.12
  - Gate Decision: If the 1M byte CNN achieves cross-endpoint ROC AUC >= 0.90, proceed to NPU deployment. If not, refine model architecture before writing Rust NPU code
  - Done when validation metrics are documented in `docs/` and the verified 8-bit model artifact is placed in `models/`
  - _Requirements: 3.1, 9.1, 9.2, 9.5_

- [ ] 2.2 Prove NPU execution from Rust via `ort` (Hardware gate)
  - Extract AMD Ryzen AI 1.9 runtime files into `runtime/ryzenai/` and record required DLL list in `runtime/README.md`
  - Copy verified 8-bit spike ONNX model into `l2d-encoder` test fixtures
  - In a Rust test, dynamically load the runtime via `ort`, register Vitis AI execution provider with `error_on_failure`, and run inference on batch of 32
  - Done when test passes on the reference host, printing non-zero NPU node count and driver version with no Python on PATH
  - If provider fails or 0 nodes placed on NPU, halt encoder tasks and report runtime blocker
  - _Boundary: EncoderSession, RuntimeFiles_
  - _Requirements: 4.3, 10.1_

- [ ] 2.3 Check runtime files, NPU device presence, and driver query
  - Implement `RuntimeFiles::check` to verify DLL presence against manifest and name any missing file with expected path
  - Query Windows device subsystem for AMD NPU presence and driver version
  - Done when missing DLL raises `L2dError::Missing` with file path, and present host reports NPU status
  - _Requirements: 4.3, 10.5_

- [ ] 2.4 Encoder packaging and revision verification
  - Define package layout under `models/encoders/<name>-<rev12>/`: `encoder.json`, `table.bin`, `model.int8.onnx`, `training-record.json`
  - Compute 40-character revision hash from SHA-256 over `model.int8.onnx`, `table.bin`, and architecture parameters
  - Verify revision on `EncoderPackage::open`; refuse opening if any byte is altered
  - Atomic packaging: write to temporary sibling directory and rename into place
  - Done when package round-trip verifies identical identity, modified file triggers refusal, and packages outside `models/` are rejected
  - _Boundary: EncoderPackage_
  - _Requirements: 3.2, 3.6, 10.3_

- [ ] 2.5 Text-to-bytes encoding and truncation flag
  - Implement `ByteCodec` in `l2d-encoder`: map text to fixed 256 byte IDs with truncation flag and byte-table embedding lookup
  - Done when 300-byte text sets truncated flag, 256-byte text does not, and produced input matches model dimensions
  - _Boundary: ByteCodec_
  - _Requirements: 1.5, 3.1_

- [ ] 2.6 NPU execution session via `ort` with CPU fallback permission
  - Implement `EncoderSession` in `l2d-encoder`: open package on NPU with disk cache keyed by revision
  - If NPU is unavailable, state reason and fall back to CPU provider only if `allow_cpu: true`
  - Process batches of 32 with padding; return one L2-normalized vector per input text in input order
  - Record execution metrics: device used, driver version, logs per second, and placement node counts
  - Done when encoding fixture with CPU allowed returns 128-dim vectors, CPU-disallowed open fails when NPU missing, and NPU execution logs placement
  - _Boundary: EncoderSession_
  - _Depends: 2.2, 2.4, 2.5_
  - _Requirements: 4.1, 4.3, 4.4, 4.5, 4.6, 4.7_

- [ ] 2.7 Integration: Encode logs and populate SQLite store
  - Connect `l2d-core` input records, `l2d-encoder` session, and `l2d-store`
  - Encode distinct texts, insert rows with vectors under the store's encoder identity
  - Done when fixture logs are encoded and stored, and re-inserting same logs only increments occurrence counts
  - _Boundary: Store + Encoder integration_
  - _Depends: 1.8, 2.6_
  - _Requirements: 4.1, 5.1_

- [ ] 2.8 Nearest-neighbor decision scorer with evidence diversity
  - Implement `NeighbourScorer` in `l2d-decide`: query $k$ nearest labelled logs within reach distance
  - Calculate ratio as bad share among valid neighbors; abstain if valid neighbors < `min_support`
  - Track evidence diversity: record `same_group_count` vs `cross_group_count` in `Evidence`
  - Agnostic threat scoring barrier: if nearest neighbors match only on static benign endpoint without payload match, avoid false threat suppression
  - Done when test store with known neighbors yields expected ratio and evidence, and isolated log triggers abstention
  - _Boundary: Scorer, NeighbourScorer_
  - _Depends: 1.9, 2.7_
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 2.9 Wire NPU & Encoder CLI commands
  - Wire CLI commands in `l2d-cli`: `l2d runtime check`, `l2d encoder info`, `l2d encoder encode`, `l2d store add`
  - Include `--allow-cpu` flag and display truncation flags and placement evidence
  - Done when CLI commands execute against fixture and print run records in human and JSON format
  - _Requirements: 1.5, 3.2, 4.3, 4.4, 4.5, 10.5, 10.7_

---

## Milestone 3: Vector Assessment & Comprehensive Evaluation

- [ ] 3.1 Kernel MMD population shift test
  - Implement RBF MMD two-sample test in `l2d-decide` (`shift.rs`) with median-distance bandwidth and seeded permutations
  - Refuse groups with fewer than 30 logs; subsample groups > 2,000 using seed
  - Done when identical groups yield $p > 0.05$, shifted groups yield $p < 0.01$, seed reproduces exact result, and 29-log group is refused
  - _Boundary: ShiftTest_
  - _Depends: 1.1_
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 3.2 Vector set abstraction and shared measures
  - Implement `VectorSet` in `l2d-assess` with optional labels and group keys
  - Implement nearest-neighbor set overlap and cross-validated label separability
  - Done when set overlap with itself is 1.0, with shuffled copy is near 0.0, and separability is near 1.0 for separated clusters
  - _Boundary: VectorSet_
  - _Depends: 1.6_
  - _Requirements: 8.2_

- [ ] 3.3 Measure information loss and vector shortening
  - Implement `assess_loss`: PCA explained variance, TwoNN intrinsic dimension, and neighbor overlap / separability after shortening vectors
  - Done when synthetic dataset with known rank correctly reports variance profile and intrinsic dimension
  - _Requirements: 8.1_

- [ ] 3.4 Measure clustering capability and within-group separability
  - Implement `assess_clusters`: KMeans silhouette score and separability overall and within groups sharing a key
  - Done when well-separated synthetic clusters give silhouette > 0.5, and within-group separability evaluates on valid groups
  - _Boundary: ClusterAssessment_
  - _Depends: 3.2_
  - _Requirements: 8.3, 8.4_

- [ ] 3.5 Report vector properties and write 2D projection
  - Implement `assess_properties`: pairwise cosine distribution, norm quantiles, anisotropy, duplicate rate, dimension
  - Implement `project_to_csv`: 2D projection (t-SNE / PCA) written to CSV with coordinates, labels, and groups
  - Done when unit-norm random vectors report norms 1.0 and anisotropy near 0.0, and projection CSV outputs all sample rows
  - _Boundary: PropertiesAssessment, Projection_
  - _Depends: 3.2_
  - _Requirements: 8.5, 8.6_

- [ ] 3.6 Compare two vector sets side by side
  - Implement `compare` in `l2d-assess`: run loss, clustering, and properties on two vector sets from identical logs and format side by side
  - Done when comparing a vector set with a shortened version correctly shows both columns and reduced neighbor overlap
  - _Requirements: 8.7_

- [ ] 3.7 Grouped evaluation harness with quality and cost metrics
  - Implement grouped K-fold cross-validation in `l2d-eval`: all logs of a group key stay on one side of each fold
  - Compute ROC AUC (overall and cross-endpoint), average precision, recall at budget, abstention rate, and cost (logs/s, CPU seconds)
  - Explicitly report cross-endpoint ROC AUC to verify that threat scoring is agnostic to training-set endpoints
  - Done when small synthetic test ensures no group key crosses fold boundaries and hand-calculated metrics match output
  - _Boundary: GroupedSplit, Metrics, Cost_
  - _Depends: 1.10, 2.8_
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 3.8 Write evaluation reports and comparison
  - Generate evaluation reports in JSON and Markdown under `reports/<date>-<dataset>-<encoder>.json` and `.md`
  - Compare baseline scorer and encoder-based scorer: state which has higher ROC AUC and by how much
  - Done when report file outputs all settings, dataset hashes, scorer metrics, and comparison summary
  - _Requirements: 9.5, 9.6_

- [ ] 3.9 Wire Assessment, Shift, and Evaluation CLI commands
  - Wire CLI commands in `l2d-cli`: `l2d shift`, `l2d assess`, `l2d eval`
  - Output human-readable summaries and machine-readable JSON
  - Verify that `l2d shift` output explicitly describes the p-value as a test result rather than probability of shift
  - Done when all assessment and evaluation subcommands run against fixture data and generate expected report files
  - _Requirements: 7.5, 8.7, 9.6, 10.7_

---

## Milestone 4: End-to-End Validation & Optional In-Rust Training

- [ ] 4.1 End-to-end integration tests on the fixture
  - Run full triage workflow with Baseline Scorer with zero runtime DLLs present
  - Run full triage workflow with NPU Encoder Scorer with CPU allowed fallback
  - Verify refusal on missing stores, missing CPU fallback permission, and encoder identity mismatches
  - Confirm entire workspace has zero network dependencies and runs with no Python on PATH
  - Done when full automated integration test suite passes cleanly
  - _Requirements: 2.5, 4.4, 9.6, 10.1, 10.2, 10.5, 10.6_

- [ ] 4.2 Hardware repeatability tests on reference AMD NPU
  - Benchmark 10,000 distinct logs on NPU: verify sustained >= 1,000 logs/sec with NPU placement confirmed
  - Compare NPU and CPU vectors for 1,000 logs: verify cosine similarity >= 0.99 for all logs
  - Measure initialization time: verify ready to encode within 30 seconds for cold and warm cache
  - Done when performance figures pass required thresholds and are printed to test log
  - _Requirements: 4.2, 4.3, 4.6, 4.7_

- [ ] 4.3 Full evaluation on HTTP CSIC 2010 and cross-endpoint generalization check
  - Run grouped evaluation across all 61,065 rows of CSIC 2010 for both Baseline and NPU Encoder scorers
  - Compare cross-endpoint ROC AUC to confirm threat score agnosticism
  - Repeat training twice with fixed seed to check repeatability within 0.01 ROC AUC
  - Done when final evaluation report is generated, saved in `reports/`, and indexed in `README.md`
  - _Requirements: 3.5, 9.5, 9.7_

- [ ] 4.4 (Optional / Follow-up) Trainable model definition and Burn training module
  - Define Burn module mirroring `EncoderArch` with classifier head in `l2d-train`
  - Implement CPU training loop with seeded shuffled batches and calibration activation tracking
  - Export trained weights into `EncoderPackage` format
  - Done when Burn model output matches exported model output within 1e-4
  - _Requirements: 3.1, 3.3, 3.4_
