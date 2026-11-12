# Implementation Plan: npu-log-decision

## Milestone 1: Working Baseline Triage & Core Infrastructure

- [x] 1.1 Set up the Rust workspace and the shared core types
  - Create the Cargo workspace with crates: `l2d-core`, `l2d-baseline`, `l2d-store`, `l2d-decide`, `l2d-assess`, `l2d-eval`, `l2d-encoder`, `l2d-cli`, and optional `l2d-train`
  - Define shared record, label (`Good`, `Bad`, `Unknown`), encoder identity, run record, and error types (`L2dError`) with exit-code mapping
  - Configure project paths for models (`models/`), data (`data/`), stores (`stores/`), runtime files (`runtime/ryzenai/`), and reports (`reports/`)
  - Set `CARGO_TARGET_DIR` off-repo and ignore runtime, store, and report directories in `.gitignore`
  - Done when workspace builds, tests verify exit code mappings, and missing items report their expected location
  - _Requirements: 10.3, 10.4, 10.5, 10.6_

- [x] 1.2 Add the command-line skeleton and the test fixture
  - Implement CLI entry point in `l2d-cli` with subcommands stubbed out, `--json` global flag, and stderr diagnostics
  - Create synthetic labelled fixture (`tests/fixtures/mini-labelled.csv`) with 200 request lines: at least 6 group keys with both labels, duplicates, 1 malformed row, and 1 conflicting label
  - Done when `l2d --help` lists subcommands, stub exits non-zero with informative error, and tests verify fixture properties
  - _Requirements: 10.6, 10.7_

- [x] 1.3 Read, deduplicate, and summarize text logs and labelled datasets
  - Implement `InputAdapter` in `l2d-core` for plain line-based logs and labelled CSV/JSONL with configurable text, label, and group columns
  - Exact text deduplication: keep occurrence counts and first position; conflicting labels become `Label::Unknown`
  - Skip malformed records and report their count and positions in `InputSummary`
  - Compute SHA-256 content hash and record count
  - Done when fixture tests show correct deduplication, malformed row skipping, and summary counts matching total rows
  - _Boundary: Input adapters and dedup_
  - _Depends: 1.2_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6_

- [x] 1.4 Read HTTP CSIC 2010 with group-key extraction for agnostic splitting
  - Implement CSIC 2010 adapter: format text as `METHOD URL\nBODY`, extract label from classification column, derive group key from `METHOD /path` (excluding query params)
  - Ensure routing structure is separable from payload for agnostic threat analysis
  - Read from `data/csic-2010/` or specified path
  - Done when adapter reports 61,065 records (36,000 good, 25,065 bad) before dedup, and tests verify group key isolation on sample rows
  - _Requirements: 1.2, 9.7, 10.4_

- [x] 1.5 Hash character n-grams into sparse feature rows with routing isolation option
  - Implement signed FNV-1a hashing of character 1-, 2-, and 3-grams over text bytes into 65,536 buckets with L2 normalization in `l2d-baseline`
  - Provide an option to isolate or strip routing tokens (method/path) to prevent the baseline from memorizing endpoint URLs
  - Done when same text yields bit-identical sparse vectors across processes, and test verifies row values on fixed examples
  - _Boundary: BaselineScorer_
  - _Depends: 1.1_
  - _Requirements: 2.1, 2.4_

- [x] 1.6 Fit deterministic logistic regression via numerical library
  - Implement L2-penalized logistic regression in `l2d-baseline` using robust numerical solver (`faer` / `linfa`), supporting sparse and dense rows
  - Guarantee bit-identical weights for identical inputs and settings without stochastic line-search instability
  - Done when two fits on the same rows yield bit-equal weights and toy separable data is classified with 100% accuracy
  - _Requirements: 2.1, 2.4_

- [x] 1.7 Fit, apply, save, and load the baseline scorer
  - Implement `BaselineScorer` in `l2d-baseline`: compute bad/good ratio (0.0 to 1.0)
  - Refusal rule: refuse fitting if fewer than 10 labelled logs or fewer than 5 of either label are provided
  - Save and load scorer with SHA-256 revision hash under `models/baselines/<name>-<rev12>/`
  - Record the content hashes of the fitting inputs in the saved scorer, so calibration can refuse the same input (11.1)
  - Done when fitting succeeds on the fixture without any runtime DLLs, reloaded model reproduces exact ratios, and refusal cases are tested
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 1.14 Move the data layer to Polars (runs after 1.6, before 1.7)
  - Add `polars` to `l2d-core` with default features off and only the CSV, JSON, lazy and `ndarray`-conversion features; confirm with the dependency tree that no cloud or HTTP client is linked, and record the added packages and the clean build time
  - Rework the line, labelled CSV/JSONL and CSIC 2010 adapters so that parsing uses Polars and each returns the logs frame of the design's "Data frames" section plus the `InputSummary`; keep `LogRecord` as the row view
  - Keep skipped-record reporting with position and reason through a validation pass before Polars reads the accepted rows; deduplication, occurrence counts, first positions and label conflicts become frame operations in first-occurrence order
  - Change the CSIC group key to the URL path alone (no method), so GET and POST copies of a payload share a group; `group_key` and the routing split are updated to match
  - Done when every existing input test passes unchanged in its expectations except the CSIC group key: fixture 200 records, 1 skipped at position 121, 169 distinct, 1 conflict; CSIC 61,065 records (36,000 good, 25,065 bad), 25,608 distinct texts, 1,632 path groups of which 8 hold both labels, the same content hashes; and the frames have the documented schema and row order
  - _Boundary: Input adapters and dedup_
  - _Depends: 1.3, 1.4_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 9.7, 10.2, 10.4_

- [x] 1.15 Size batches to the free memory before loading a dataset (runs after 1.14, before 1.7)
  - Read the host's free physical memory once before loading and derive a memory limit: the analyst's limit if given, otherwise a share of the free memory (default one half)
  - Estimate bytes per row from the file size and a sample of the first rows, and choose the batch size from the limit; the analyst's batch size overrides it
  - Read datasets in batches in file order: the validation pass walks the source in batches and Polars parses each batch of accepted records, with deduplication state kept across batches (`Dedup::push` already takes batches), so that the logs frame and the input summary are identical for any batch size; the source file is no longer loaded whole
  - Add a guard test that fails when `Cargo.lock` names a network client crate (tokio, reqwest, hyper, rustls, object_store, mio, socket2, h2), so a later feature change cannot link one unnoticed
  - Scale dynamically: start from the maximum batch size, read the free memory again before every batch, halve the next batch when free memory is below the reserve and double it (up to the maximum) when memory has recovered; record each change with the free memory that caused it, and report the smallest and largest batch used
  - Make the free-memory reading injectable so tests can simulate memory falling and recovering during a run
  - Refuse when the smallest allowed batch does not fit; provide the whole-dataset check for operations that need all rows at once, refusing before loading with the estimate and the limit
  - Print the plan (free memory observed, limit, bytes per row, batch size, whether overridden) and make it available for run records; add the global `--batch-rows` and `--memory-limit` options to the command-line skeleton
  - Done when reading the fixture and CSIC 2010 with batch sizes of 1,024 rows, 10,000 rows and the whole file gives identical frames and summaries, an artificially low memory limit triggers both refusals with the stated figures, a simulated drop and recovery of free memory shrinks and regrows the batch with the changes recorded and the results unchanged, and the plan shows the free memory read from the host
  - _Boundary: MemoryBudget, Input adapters and dedup_
  - _Depends: 1.14_
  - _Requirements: 10.2, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9_

- [x] 1.13 Conformal calibration layer over any scorer (runs after 1.7, before 1.10)
  - (Superseded by task 1.18, which replaced the crate with own code.) Add the `conformal-prediction` crate and wrap it in `l2d-decide` (`conformal.rs`) behind the design's contract; confirm it links no network client and gives identical results on repeat, and record in Implementation Notes which parts of the method the crate covers and which the wrapper adds
  - Label-conditional split conformal calibration on scorer ratios: p-value per label, label set (good, bad, both, neither) at a chosen error rate (default 0.05), no randomization
  - Per-group strata where a group holds at least `min_per_group` calibration logs of a label, pooled otherwise; each decision names its stratum
  - Refuse an error rate that the calibration counts do not support, stating the counts and the smallest supported rate
  - Save and load under `models/calibrations/<name>-<rev12>/` with scorer kind and revision
  - Done when a synthetic test with known score distributions achieves the chosen error rate per label within sampling tolerance, repeated calibration gives identical label sets, the too-few-logs refusal is tested, and a calibration fitted on the fixture's held-out part loads back identically
  - _Boundary: ConformalCalibrator_
  - _Depends: 1.7_
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.10_

- [x] 1.8 Create, open, and populate SQLite store with FTS5 and exact vector storage
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
  - With a calibrated primary scorer, place logs by label set: bad alone to the review ranking, good alone scored down by calibration, both or neither to the abstained list with the reason (uncertain, unlike the calibration logs); logs beyond the budget are scored down by budget only, and each scored-down log says which applied
  - Owner decision of 2026-11-12 on the 1.13 CSIC result (the error rate is missed for good logs on endpoints the calibration logs do not cover): triage places logs by label set as specified and does not treat uncovered endpoints specially; every triage result with a calibration states that the error rate holds only for logs that resemble the calibration logs (requirement 11.9), and the result reports how many logs used a group's own list and how many the pooled list. No per-endpoint normalisation of ratios in milestone 1
  - A loaded calibration is counted against the memory budget (1.13 note, requirement 12.9)
  - Done when triage on the fixture partitions logs correctly according to budget and scorer attribution, with and without a calibration
  - _Boundary: Scorer, Triage_
  - _Depends: 1.7, 1.9, 1.13_
  - _Requirements: 6.4, 6.5, 6.6, 6.7, 11.6, 11.7_

- [ ] 1.11 Wire Baseline CLI commands
  - Connect CLI commands in `l2d-cli`: `l2d ingest`, `l2d baseline fit`, `l2d baseline score`, `l2d calibrate`, `l2d triage`
  - Support both human-readable text and `--json` machine-readable output
  - `l2d calibrate` refuses an input whose content hash is among the scorer's fitting inputs; `l2d triage` takes a calibration and an error rate, and its output states that the error rate holds only for logs that resemble the calibration logs
  - Add options for the memory shares (owner decision of 2026-11-12): the share of free memory used as the limit (default one half) and the share kept as the reserve (default one quarter), on every command that takes `--memory-limit`; reject values outside (0, 1) and a reserve that leaves no usable memory, and test both
  - Done when ingesting fixture displays summary, fitting then scoring displays ratios, calibrating on a held-out part of the fixture succeeds, and triage displays ranked review, scored-down and abstained lists with label sets
  - _Requirements: 1.1, 1.2, 1.6, 2.1, 2.2, 6.4, 6.5, 10.6, 10.7, 11.1, 11.6, 11.7, 11.9_

- [ ] 1.12 Baseline validation: Grouped cross-validation on CSIC 2010
  - Run 5-fold grouped cross-validation (grouped by URL path alone) using the baseline scorer on CSIC 2010
  - Compute the cross-endpoint ROC AUC on the 8 path groups that hold both labels; report the logs of single-label groups (about 1,600 bad-only paths) separately with their count and the share ranked within the review budget
  - Verify that the baseline achieves cross-endpoint ROC AUC >= 0.90 without memorizing static URLs
  - In each fold, calibrate on a part of the fitting side the scorer did not see, and report the achieved error rate against the chosen rate per label, overall and per group, and the share of each label set
  - Done when baseline cross-endpoint benchmark is completed and results, including the calibration coverage, are recorded in `reports/`
  - _Requirements: 2.1, 9.1, 9.2, 9.7, 9.8, 11.8_

- [x] 1.18 Replace the `conformal-prediction` crate with own code (runs after 1.8, before 1.10)
  - Owner decision of 2026-11-12: the crate only stores a list's scores and counts `score >= candidate` by linear scan, and links `lean-agentic`, `rand` 0.8 and a second `ndarray` into the tool
  - In `l2d-decide/conformal.rs`, compute the p-value `(count + 1) / (n + 1)` on the sorted lists by binary search, ties counting as greater or equal, no randomness; remove the dependency from the workspace and from `l2d-decide`
  - The public contract, the label sets, the calibration revision and the saved files stay exactly as in 1.13: add a test that loads a calibration saved by the 1.13 code (kept as a fixture) and gets identical label sets, and keep the test that compares the p-value with a direct count
  - Re-run the CSIC evaluation in release and confirm the figures of the 1.13 note to the printed digit
  - Done when `Cargo.lock` no longer names `conformal-prediction` or `lean-agentic`, all 1.13 tests pass unchanged in what they assert, and the CSIC figures are unchanged
  - _Depends: 1.13_
  - _Requirements: 11.2, 11.3, 11.10, 10.2_

- [ ] 1.16 Read zip archives in batches under the memory limit (runs after 1.12)
  - `ZipInput` in `l2d-core`: entry walk in central-directory order, entry patterns, text check, gzip entries, maximum record length, entry reports, archive content hash, positions naming entry and line; other nested archives are skipped and reported
  - Add `zip` (default features off, `deflate` only) and re-check `cargo tree` and the lock file for network crates
  - Batches through `Batcher`; the deduplication set is held against the budget, with a refusal that states the memory needed and available when it does not fit
  - `l2d ingest` accepts an archive and prints the entry report
  - Tests first, on archives built in a temporary directory: equality with the unpacked files at several batch sizes; a binary entry, an encrypted entry, a damaged entry and a gzip entry; an over-long record; a zip64 archive
  - On-request test in release on `data/ait-lds-v2/russellmitchell.zip` with a log pattern: records read, time, and the heap peak against the limit
  - Done when the fixture archive gives the same summary as its unpacked files, and the AIT archive is read within the memory limit
  - _Depends: 1.11, 1.15_
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10, 13.12, 13.13, 12.4, 12.9_

- [ ] 1.17 Write per-log results in parts and deduplicate on disk
  - Add the `polars-io` features `parquet` and `zstd` and re-check `cargo tree` and the lock file for network crates; confirm that the C parts build on the reference host
  - Parquet parts per batch with a manifest, `--work-dir` (default `<project>/work/<run>/`, ignored by git), and a free-disk check before the run
  - Hash-partitioned deduplication on disk when the in-memory set would exceed the budget; equality with the in-memory path asserted on the fixture
  - `l2d baseline score` on an archive writes parts
  - Done when scoring an archive larger than the memory limit completes within the limit and its parts hold one row per distinct log
  - _Depends: 1.16_
  - _Requirements: 13.11, 13.12, 12.6, 12.9_

- [ ] 1.19 Read archive entries in parallel
  - Read several entries at once under one shared memory budget, with a defined record order so that results equal those of sequential reading
  - Measure against sequential reading on an AIT archive in release; keep it only if it is faster within the same memory limit
  - Done when parallel and sequential reading give identical summaries and parts on the fixture and the measurement is recorded
  - _Depends: 1.17_
  - _Requirements: 13.8, 13.10, 12.6, 12.9_

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
  - For calibrated scorers, hold out a calibration part of each fold's fitting side and report achieved against chosen error rate per label, overall and per group, and the share of each label set
  - Done when small synthetic test ensures no group key crosses fold boundaries and hand-calculated metrics match output
  - _Boundary: GroupedSplit, Metrics, Cost_
  - _Depends: 1.10, 2.8_
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.8, 11.8_

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

## Implementation Notes

- Data layer (decision of 2026-11-11): Polars is the one tabular data layer. From task 1.14 on, tabular data crosses crate boundaries as Polars frames with the schemas in the design's "Data frames" section; tasks written before that decision that speak of records or lists (1.7, 1.9, 1.10, 1.12, 2.7, 3.7) take and return those frames. Numeric code gets slices, arrays or sparse rows from the frames in row order and stays out of Polars' thread pool.
- Memory-aware batching (decision of 2026-11-11): requirement 12, task 1.15. Every operation that loads a dataset plans its batch size from the free memory first; operations that need all rows at once (baseline fit, calibration, shift test, pairwise assessments) call the whole-dataset check before loading. Later tasks (1.7, 1.12, 2.6, 2.7, 3.x) use it; encoding batches for the NPU stay at the model's fixed 32 and are fed from these data batches.
- Conformal layer (decision of 2026-11-11): the `conformal-prediction` crate is adopted and wrapped in `l2d-decide` (task 1.13, requirement 11). Superseded on 2026-11-12: the owner had the crate replaced by own code (task 1.18); the tool no longer depends on it.

- 1.1: Build output relies on the host's `CARGO_TARGET_DIR`; `l2d-core/build.rs` only warns if output lands in the repo. Run `cargo fmt --all` before each commit.
- 1.1: `Placement` and `PlacementSource` live in `l2d-core` (the core `RunRecord` needs them); `l2d-encoder` re-exports them and must not redefine them.
- 1.1: `Label` serializes in lower case (`"good"`, `"bad"`, `"unknown"`); keep the CLI JSON consistent with that.
- 1.1: `l2d-train` is a workspace member outside `default-members`; `l2d-cli` and `l2d-eval` do not depend on it.
- 1.1: Store location is undecided in the design (`stores/<name>.db` at the technology stack, `stores/<name>/` in the file plan). Task 1.8 uses `stores/<name>.db` and adds the path helper to `ProjectPaths`.
- 1.2: Unwired commands exit 4 (`Refused`, "not implemented yet"); with `--json` the error envelope `{"error":{"kind","message","exit_code"}}` goes to standard output and the one-line diagnostic to standard error. Stubs declare no options yet.
- 1.2: Fixture `tests/fixtures/mini-labelled.csv` (header `text,label,group`): 200 data rows = 199 well-formed (115 good, 84 bad) + 1 malformed at data row 121 (two fields only, group missing); 169 distinct texts (96 good-only, 72 bad-only, 1 conflicting); 27 texts repeat; 8 group keys `METHOD /path`, each with 12 good and 9 bad distinct texts. Exact counts are pinned by `crates/l2d-cli/tests/fixture.rs`.
- 1.2: The malformed row is malformed only by field count. A reader must treat a wrong field count as a skipped record (expected: 200 read, 1 skipped at position 121, 169 distinct, 1 unknown label); a lenient CSV reader would wrongly accept it.
- 1.2: One group key holds only 21 to 22 distinct texts, below the 30 minimum of the shift test; fixture shift tests must compare good against bad, not group against group.
- 1.3: Input code is in `l2d-core`: `input.rs` (`InputAdapter`, `Dedup`, `read_source`, `content_sha256`, `strip_bom`, `split_lines`) and `adapters/{lines,labelled}.rs`. New adapters reuse `Dedup` and `read_source`; they do not reimplement deduplication.
- 1.3: Position is the 1-based record index: line number for lines and JSONL, data-row index without the header for CSV. CSV blank lines take no position, so `records` counts CSV data rows, not file lines; `l2d ingest` must not present it as a line count.
- 1.3: Labels: `good`/`bad` case-insensitive; empty, `unknown`, null or absent means unlabelled and never conflicts; any other value skips the record. On duplicate text the first group key is kept. Any read failure maps to `Missing`.
- 1.3: `LabelledAdapter::new` defaults the label column to `Some("label")`; unlabelled CSV/JSONL needs `label_column = None` set by the caller (1.11).
- 1.3: A CSV data row starting with an unbalanced quote absorbs the following rows into one skipped record (standard CSV behaviour). Adapters for real datasets must check the total row count against the dataset's known size.
- 1.3: Open follow-ups, not blocking: no test pins that the content hash covers raw bytes including a byte-order mark; input tests leave scratch directories in the system temp folder.
- 1.7: `l2d_baseline::scorer::BaselineScorer`: `fit(&LogsFrame, &[InputSummary], &BaselineSettings)`, `fit_with(.., &MemoryBudget)`, `ratio`, `score(&LogsFrame)` and `score_texts` (Scores frame: `text`, `scorer` = "baseline", `ratio` Float32, `abstained` false, `no_features`), `save`, `load`, `load_under`, `was_fitted_on(hash)`, `has_known_fitting_inputs()`, `fit_report()`, `gradient_converged()`. Settings: buckets, ngram_max, routing, l2, max_iter (1,000), tolerance (1e-3), all in the revision.
- 1.7: For 1.13: refuse calibration when `was_fitted_on(input hash)` is true, and when `has_known_fitting_inputs()` is false (unknown provenance). Fitting inputs are merged on re-save of the same revision; they are outside the revision, so a hand-edited list is not detected; two processes saving the same model at the same moment can lose one added entry (no lock, accepted for a single-analyst tool).
- 1.7: For 1.10 and 1.11: rows flagged `no_features` all tie at one constant ratio and carry no evidence; triage must decide how to place them (in strip mode 1,634 of 20,825 CSIC fitting rows have no features). Pass the budget used for reading to `fit_with`, so the logs frame counts against the limit. Use `load_under` in commands. A project root written with `..` is refused on save; commands pass a normalized root. A fit in which every row is featureless is not refused.
- 1.7: For 1.12: a whole-CSIC fit takes 5 to 15 s in release and about 96 s in debug; run the evaluation in release. Sanity run with three path groups held out (4,783 rows): ROC AUC strip 0.989, keep 0.996, isolate 0.998. No held-out payload occurs in the fitting rows, but 28 % of held-out bad rows contain a parameter value seen only in bad fitting rows (CSIC reuses a small attack dictionary). Held-out good rows average 0.65 to 0.68, so a fixed 0.5 threshold does not transfer across endpoints; calibration per group matters. 1,604 of 1,632 groups are bad-only.
- 1.7: On-request release tests: `cargo test --release -p l2d-baseline --test scorer_csic -- --ignored --nocapture --test-threads=1`. Open follow-ups: no failure-injection test for the atomic save steps; the memory estimate has a thin margin (heap peak 0.95 of the estimate on CSIC keep mode).
- 1.15: `l2d_core::memory`: `MemoryBudget::{observe, observe_with, plan, next_batch_rows, require_whole, require_held, hold, usable_bytes, left_bytes}`, `Batcher`, `BatchCost { expansion, row_bytes, batch_bytes }`, `BatchPlan` (Display, `RunRecord::record_batch_plan` under `settings["batch_plan"]`), `BatchChange { cause, available_bytes, .. }`, `parse_memory_size`, `PROCESS_BASELINE_BYTES` (32 MiB). Free memory via `windows-sys` `GlobalMemoryStatusEx`; reading injectable.
- 1.15: Adapter contract: required `read_from(source, &mut MemoryBudget) -> (LogsFrame, InputSummary, BatchPlan)`; provided `read_with(path, budget)` and `read(path)`. `Csic2010Adapter::read_counted_with`. `Source` reads 64 KiB pieces and hashes incrementally; `read_source`, `strip_bom`, `split_lines` no longer exist. Batches under 4 MiB are parsed by Polars on one thread; `Dedup::push` copies text and group strings out of the parse buffers.
- 1.15: Memory limit is kept in reality (reviewer measured peak working set at most 0.78 of the limit over 12 shapes, completed or refused). Operations that hold memory must declare a `BatchCost` and count what they keep through `hold`; operations needing all rows call `require_whole` first. Constants (parse 6.0 x bytes + 500 per row + 4 MiB per batch; held 200 per row + 2.6 x estimated size; baseline 32 MiB) are calibrated on this host with Polars 0.52.0: re-run `cargo test --release -p l2d-core --test memory_limit -- --ignored --nocapture` (about 2.5 min) on any Polars change and at feature validation.
- 1.15: The accounting is cautious: reads can be refused that would have fitted (2,000,000 distinct 120-byte lines peak at about 830 MiB but need a limit of about 1.2 GB). 1,024-row batches are slow at scale (about 47 to 60 s against 3 s for 2,000,000 distinct lines). Follow-up: a deduplication that does not copy its state.
- 1.15: For 1.11: no command prints the batch plan yet; `--memory-limit` at or below 32 MiB is only rejected when a budget is observed, so wired commands must observe the budget before doing anything else; a limit between 32 and about 36 MiB gives a misleading refusal ("record larger than 1 B") and should say the limit leaves nothing for a batch. `--memory-limit` units are multiples of 1,024.
- 1.15: `crates/l2d-core/tests/no_network_crates.rs` fails if `Cargo.lock` names a network client crate. `crates/l2d-core/examples/read_batched.rs` reads a file with a given limit and batch size for memory measurements.
- 1.13: `l2d_decide::conformal::ConformalCalibrator`: `fit(&DataFrame, ScorerKind, scorer_revision, &CalibrationSettings)`, `fit_with(.., &MemoryBudget)`, `fit_baseline(&BaselineScorer, &LogsFrame, &[InputSummary], &CalibrationSettings, &MemoryBudget)`, `predict(ratio, group, error_rate) -> Calibrated`, `require_error_rate`, `require_scorer`, `smallest_supported`, `strata`, `good`, `bad`, `save`, `load`, `load_under`; `require_held_out(scorer, inputs)`, `points_frame`, `points_schema`, `estimated_bytes(points, group_bytes)`, `DEFAULT_ERROR_RATE` (0.05), `DEFAULT_MIN_PER_GROUP` (100), `EXCHANGEABILITY_NOTE` (the sentence of requirement 11.9 for 1.11). `l2d_decide::ScorerKind` (`baseline`, `neighbour`) is defined at the crate root; 1.10 uses it. `ProjectPaths::{calibrations_dir, calibration_dir}` were added to `l2d-core`.
- 1.13: `conformal-prediction` is pinned `=2.0.1`. The crate covers: keeping the calibration scores of a list and the p-value `(count of scores >= candidate + 1) / (n + 1)` with ties counted (`ConformalPredictor::calibrate` and `predict`); a unit test compares it with a direct count. The wrapper adds: the nonconformity of a ratio for a label (the crate's measures are for regression; its `NonconformityMeasure` hook passes the wrapper's value through), one list per label and per group, the choice of list, the label set at an error rate chosen per call (the crate fixes one level per predictor, so predictors are created at the smallest positive level and return every p-value), the refusal rule, the check against the fitting inputs, the memory check, saving and loading. The crate's `cpd`, `pcp` and `streaming` modules are not used; they are the only places that draw random numbers.
- 1.13: Dependencies: `Cargo.lock` grows by `conformal-prediction`, `lean-agentic` 0.1.0 and a second `ndarray` (0.17.2), to 199 packages. `rand` 0.8.8 with `getrandom` 0.2 was in the lock for the tests of `l2d-baseline` only and is now linked into the tool, unused. `lean-agentic` has no dependencies and no file, process or network access, but its arena, symbol table and environment are created for every list. No network crate; the guard test passes. Results are identical on repeat within and across processes. The owner has not reviewed these crates.
- 1.13: Deviations from the design contract: `fit` takes the calibration points frame, not `&[CalibrationPoint]` (data layer decision); `Calibrated` has `good_stratum` and `bad_stratum` in place of one `stratum`, because a group can have its own list for one label and the pooled list for the other (requirement 11.4 is per label); `CalibrationError` has more variants (`UnknownFittingInputs`, `UnknownCalibrationInputs`, `InvalidErrorRate`, `InvalidRatio`, `InvalidPoints`, `InvalidSettings`, `OtherScorer`, `Memory`, `Scorer`, `Internal`). The design text should be brought in line.
- 1.13: Rules to know. An error rate below `1 / (n + 1)` of the shortest list in use (pooled or a group's) is refused for every log, with the lengths of the shortest good and bad lists; a calibration without good or without bad logs is refused at fit. P-values and error rates are compared as 32-bit floats, and `Calibrated` reports those values. Rows labelled `unknown` are ignored. The pooled list of a label holds the logs of all groups. Saved as `calibration.json` and `scores.bin` (sorted nonconformity lists as little-endian doubles); the revision covers scorer kind and revision, `min_per_group` and the lists, and does not depend on the order of the calibration logs. Calibration inputs are not stored with the calibration.
- 1.13: For 1.10, 1.11, 1.12: the provenance check compares content hashes of whole inputs. Folds cut from one file all carry that file's hash, so `fit_baseline` refuses them; the evaluation harness keeps its parts disjoint itself and calls `fit_with` with the points. Call `require_scorer` before judging ratios and `require_error_rate` once before a batch. A decision scans the lists it uses (the crate has no sorted lookup), so its time grows with the number of calibration logs: 4,783 decisions against 5,254 calibration logs took about 0.01 s in release; large triage runs against large calibrations were not measured. A scorer's abstention has no ratio and must not be passed to `predict`.
- 1.13: Measured on CSIC 2010 at error rate 0.05 (release, three mixed path groups held out, 4,783 logs of which 4,453 are one endpoint, `/tienda1/publico/anadir.jsp`): the target is missed for good logs on held-out endpoints. Calibrated on other logs of the endpoints the scorer was fitted on: good logs wrongly ruled out 79.1 % (keep) and 56.7 % (strip), bad logs 0.0 %. Calibrated on two endpoints kept from the scorer: good 32.0 % (keep) and 31.9 % (strip), bad 0.0 %. On logs of the seen endpoints, which are exchangeable with the calibration logs, the rate is reached: good 5.2 % and 5.1 %, bad 4.5 % and 4.6 %. Cause: the scorer's ratios shift on unseen endpoints (good logs average 0.39 on calibration and 0.69 held out), which is the exchangeability condition failing; no setting of the calibration layer repairs it. From the same run (confirmed by the reviewer): on held-out endpoints no log receives the set "good" in any of the four arrangements, so calibration scores nothing down there; a higher error rate makes the good-log miss worse (70 % to 99.6 % at 0.1, 99.6 % to 100 % at 0.2); the failure is on the safe side, bad logs being missed 0.0 % at 0.05 and at most 0.5 % at 0.2. The pooled list also holds the logs of groups that have their own list, so a log of a small or unseen group is judged against a mixture dominated by the large groups. Command: `cargo test --release -p l2d-decide --test conformal_csic -- --ignored --nocapture --test-threads=1` (about 8 s).
- 1.13: Memory: `fit_with` asks `require_whole` for `1 MiB + 200 bytes per point + 2 bytes per byte of group key`; the heap peak was at most 69 % of that in the first runs and 73 to 80 % in the runs made for task 1.18 on the same code (it varies by run, the peak being Polars' group-by), over seven shapes (the worst is a group per point) in `tests/conformal_csic.rs`, which asserts it on every run; the points frame itself is the caller's and is not in the estimate. `save` and `load` take no budget. A loaded calibration is not transient: it keeps about 16 bytes per stored value for its lifetime, and while loading the peak is roughly 24 to 40 bytes per value of the list under construction (reviewer's reading of `List::new`, not measured). Tasks 1.10 and 1.11 must count a loaded calibration against the memory budget (requirement 12.9). Open follow-ups: no failure-injection test for the atomic save; the group-key check on load does not verify that a group list is a subset of the pooled list. For 1.11 (provenance gaps): the same logs in a file with different bytes (reordered, a subset, re-exported) pass the whole-input hash check; nothing binds the `inputs` argument of `fit_baseline` to its `logs`; `require_held_out` takes a `BaselineScorer` only, so the neighbour scorer needs an equivalent. Test weaknesses noted by the reviewer: the memory-refusal test in `tests/conformal.rs` does not assert the estimate or the limit it names; the abstained-row filter in `fit_baseline` is never exercised; the synthetic tolerance of 0.03 at 0.05 is loose.
- 1.18: The `conformal-prediction` crate is gone (owner decision of 2026-11-12, which reverses the decision of 2026-11-11 above); the 1.13 notes that describe the crate are history. `conformal.rs` now computes the p-value itself: each list is kept once, sorted ascending (the form it is saved and hashed in), the number of values at least the new log's is the list length less `partition_point(value < candidate)` (binary search; ties count, nothing is random), and the p-value is `(count as f64 + 1.0) / (n as f64 + 1.0)` rounded to `f32`, the crate's arithmetic in the crate's order. The public contract is unchanged: same types, signatures, error variants and messages; only the doc text of `CalibrationError::Internal` changed (it is now Polars failing, or lists that do not have the shape of a calibration). `predict` can no longer return `Internal`. Revisions and saved files are unchanged.
- 1.18: Dependencies: `Cargo.lock` goes from 206 to 203 packages; `conformal-prediction` 2.0.1, `lean-agentic` 0.1.0 and `ndarray` 0.17.2 left, nothing else changed. `rand` 0.8.8 (with `rand_chacha` 0.3.1, `rand_core` 0.6.4) stays in the lock file through `linfa`, a test-only dependency of `l2d-baseline`, and is no longer linked into the tool (`cargo tree -i rand@0.8.8 -e all`). `getrandom` 0.2.17 is still linked into the tool through Polars, as before 1.13. The tool still links `rand` 0.9.5 and `ndarray` 0.16.1 through Polars and the regression, as before 1.13.
- 1.18: Proof of equality: `crates/l2d-decide/tests/fixtures/conformal-1.13/` holds files written by the crate-based code before it was changed: a saved calibration (`saved/crate-2.0.1-2a1be4d97ed8/`, 180 calibration logs, six lists: both pooled lists, a group with a list for each label, a group with a list for good only, a group with a list for bad only, ties, both zeros, the smallest positive ratio, the largest below 1, and 1), its revision, and `decisions.tsv` with 250 ratio and group pairs at six error rates (p-values as `f32` bit patterns, lists, label sets) and 36 refusals (error rate too small, not an error rate, not a ratio). `tests/conformal_fixture.rs` loads the saved calibration, fits the same points again in both orders, and requires the same revision, bit-identical p-values, identical sets, lists and refusal texts, and byte-identical files on saving. The manifest names the tool version (`0.1.0`), so the byte comparison of `calibration.json` puts the present version in that one place. The fixture directory is stored without line-ending conversion (`tests/fixtures/.gitattributes`; the host has `core.autocrlf` on). To regenerate: the reference is the crate-based code, so check out the last commit whose `Cargo.lock` names `conformal-prediction`, copy `tests/conformal_fixture.rs` into it, set `L2D_CONFORMAL_FIXTURE_OUT` to an empty directory, run `cargo test -p l2d-decide --test conformal_fixture -- --ignored regenerate`, and replace the fixture directory with what was written. On the present code the same command only reproduces the committed files.
- 1.18: Measured (release). CSIC evaluation, same command as in the 1.13 note: every printed figure is the same before and after (good wrongly ruled out 79.05 % keep and 56.67 % strip on the fitted endpoints, 32.02 % and 31.90 % with two endpoints kept from the scorer, bad 0.00 %; control 5.21 % and 5.06 % good, 4.52 % and 4.62 % bad); the only lines that differ apart from timings are three `heap_peak` values of the calibration (0.86, 0.97, 0.82 MiB before; 0.84, 0.73, 0.86 MiB after), which vary from run to run. Speed of a decision (`cargo test --release -p l2d-decide --test conformal_fixture -- --ignored decision_speed --nocapture`, 20,000 decisions, no groups): against 5,000 calibration logs 0.91 microseconds before and 0.05 after; against 200,000 calibration logs 32.8 microseconds before and 0.08 after. A decision no longer grows with the number of calibration logs in any way that matters, which settles the open point of the 1.13 note for large triage runs.
- 1.18: Memory: `estimated_bytes` is unchanged (`1 MiB + 200 bytes per point + 2 bytes per byte of group key`) and so are the heap assertions. The heap peak of a calibration as a share of the estimate over the seven shapes of `tests/conformal_csic.rs`: 27, 50, 77, 44, 57, 22, 35 % before; 28, 49, 73, 44, 58, 22, 26 % after, and 27, 51, 67, 44, 56, 22, 22 % on a repeat. The peak is the counting in Polars, which runs on several threads and varies by a few points from run to run, so it hardly moved; the worst shape is still a group per point. What did change is what a calibration keeps: 8 bytes per stored value (one sorted list) where the 1.13 note says about 16, and no second copy while a list is built. The kept figure follows from the representation; it was not measured separately.
- 1.8: `l2d_store::Store`: `create(paths, name, identity)`, `open(paths, name)`, `add(&DataFrame, identity)`, `add_with(.., &mut MemoryBudget) -> AddReport { rows, added, repeated, held_bytes, longest_text_bytes, plan }`, `require_identity(given)`, `retained_bytes()`, `len`, `is_empty`, `name`, `path`, `identity`, `created_at`, `tool_version`. Free functions `log_id(source, text)`, `rows_schema()`, `rows_frame(&LogsFrame, source, vectors: &[f32], dimension, truncated: &[bool])`; column constants `TEXT`, `SOURCE`, `OCCURRENCES`, `LABEL`, `GROUP`, `TRUNCATED`, `VECTOR`; `MAX_NAME_BYTES` (64), `SCHEMA_VERSION` (1), `APPLICATION_ID`, `CACHE_BYTES` (8 MiB), `ADD_FIXED_BYTES` (16 MiB), `ADD_TEXT_FACTOR` (192), `MAX_TEXT_BYTES` (8 MiB), `ADD_ROW_BYTES` (256), `IDLE_BYTES` (1 MiB), `RELEASE_TEXT_BYTES` (16 KiB), `RETAINED_TEXT_FACTOR` (32). `ProjectPaths::store_path(name)` = `stores/<name>.db` was added to `l2d-core`. Code is in `lib.rs` (facade, rows frame, add) and `sqlite.rs` (schema, create, open); `vector.rs` and `query.rs` of the design's file plan are left to 1.9.
- 1.8: Schema as built (format 1, `PRAGMA user_version`; `PRAGMA application_id` 0x4C324453). `meta` (one row: `id` = 1, `encoder_name`, `encoder_revision`, `encoder_dimension`, `created_at` UTC `YYYY-MM-DDTHH:MM:SSZ`, `tool_version`). `logs` (`seq` INTEGER PRIMARY KEY = insertion order from 1, `id` TEXT UNIQUE, `text`, `source`, `occurrences`, `label`, `group_key`, `truncated`, `vector` BLOB, `encoder_revision`), indexes `logs_label` and `logs_source`. Both tables are STRICT with CHECK constraints: label one of the three names, occurrences 1 to 4,294,967,295, truncated 0 or 1, vector length exactly `4 * dimension` bytes. `logs_fts`: FTS5 over `logs.text` as an external-content table (`content = 'logs'`, `content_rowid = 'seq'`, so the text is stored once) with the `trigram` tokenizer, kept in step by insert, delete and update triggers. Vectors are little-endian 32-bit floats, `dimension` of them, no header. Journal mode is SQLite's default (rollback journal), so a closed store is its one file.
- 1.8: For 1.9. The trigram index answers containment of any phrase of three or more characters, whatever the characters (`logs_fts MATCH '"<phrase with " doubled>"'`, join `logs.seq = logs_fts.rowid`); it ignores case, so a case-exact containment needs a second condition (`instr(text, ?) > 0`), and a phrase of one or two characters cannot use the index and needs a scan. Order by `seq` for insertion order. `Store::require_identity` is the check `nearest` needs. Reading rows back (`StoredLog`, `vectors`, `text_search`) does not exist yet; the tests read the file through their own `rusqlite` connection. The connection is opened read-write; a read-only open for triage (requirement 6.9) is not offered yet.
- 1.8: Rules to know. The id is computed by the store: SHA-256 (lower-case hex) of the byte length of the source as 64 bits little-endian, the source, the text. The same text from another source is another log. Re-adding an id adds the occurrences and changes nothing else of the row; this also holds inside one frame. An add is one transaction: an unusable row (named by its 1-based row number), a memory refusal or a database failure leaves the stored rows as they were (the file is byte-identical after a refused add and after a 12,000-row add that fails on its last row when most of it was already written to the file). A text that holds a NUL character, or is longer than `MAX_TEXT_BYTES`, is refused (`InvalidRows`). An occurrence count beyond 32 bits fails the add (`Failed`, from the CHECK constraint). `create` refuses an existing file of that name whatever it holds (`Exists`); `open` of an empty file (what a killed `create` leaves), a non-database, a foreign database or another format is `Failed` and leaves the file unchanged. Vectors must be finite; their norm is not checked. `StoreError` maps to `L2dError`: `IdentityMismatch` and `Exists` to `Refused`, `Missing` to `Missing` (item `store <name>`), `InvalidName` and `InvalidIdentity` to `InvalidInput`, `InvalidRows` and `Failed` to `Runtime`, `Memory` to the inner error.
- 1.8: Memory. `add_with` walks the caller's frame in batches from `MemoryBudget::plan` and `next_batch_rows` (free memory read before each batch, changes recorded in the returned plan) and binds each row straight from the frame's buffers, so nothing is kept per row. It holds its own share on top of what the caller's budget already holds (`ADD_FIXED_BYTES` plus `ADD_TEXT_FACTOR` times the bytes of the longest text of the frame; reported as `AddReport::held_bytes`), restores the caller's figure afterwards, and asks for the frame's bytes per row plus 256 for a batch. It is refused before anything is written when less than its own share is left; a refusal between batches (free memory fell) rolls back what was written, and its message says so and names the store's share. `Store::add` observes a fresh budget in which nothing is held, so the caller's frame is counted nowhere on that path: commands that hold a frame must use `add_with` with the budget the frame was read under. The frame is the caller's and is not counted by the store: 1.11 and the encoder integration must count it (as 1.7 does for the logs frame) and add a dataset that exceeds the budget frame by frame, one `add` per frame (each is its own transaction). Measured (release, `cargo test --release -p l2d-store --test store_csic -- --ignored --nocapture --test-threads=1`): CSIC's 25,608 distinct logs with synthetic 128-dimension vectors (4.7 MiB of text, 12.5 MiB of vectors) are added in 3.3 to 3.4 s (about 7,600 logs/s), the file is 45.7 MiB, SQLite's heap peaks at 8.5 MiB (the longest CSIC text has 891 bytes; the figure is bounded by the 8 MiB page cache and does not grow with the number of rows, but it does grow with the length of the longest text, see the next note), the Rust side allocates under 0.01 MiB; re-adding all of them takes 0.45 to 0.60 s (the later runs, with the release at the end of every add, gave 0.56 to 0.60 s); reopening takes 2 to 3 ms; a containment query through the index takes 3 to 25 ms. The peak working set of that whole test process was 73 MiB, which is the peak of reading CSIC itself. Debug builds are slow (about 4,000 logs/s: the bundled C source is compiled without optimization), so the 12,000-row measurement (which also checks the rollback of an add that has outgrown the page cache) and the long-text table run on request only, with the same command; the on-request tests together take about eight minutes, nearly all of it the long-text table. The default run keeps one heap test (a 256 KiB text among 200 short ones, under a fixed budget, under a second).
- 1.8: Memory of long texts (review round 1; the first version asked for the frame's average row and was wrong). The text index builds the entry of one text whole in memory, so SQLite's heap follows the longest text of the frame, not the number of rows and not the average. Peak of SQLite's heap in MiB when one long text is added among 200 short ones (release, `long_texts_measured`; the small test database itself takes 0.4 to 0.5 MiB):

  | Text | 400 B | 4 KiB | 16 KiB | 64 KiB | 256 KiB | 1 MiB | 4 MiB | 8 MiB | 16 x 1 MiB | 16 MiB | 32 MiB | 64 MiB |
  |------|-------|-------|--------|--------|---------|-------|-------|-------|------------|--------|--------|--------|
  | request lines, as in a log | 0.4 | 0.5 | 0.6 | 0.9 | 2.1 | 6.6 | 24.2 | 47.4 | 13.5 | 86.3 | 164.1 | 319.7 |
  | random printable ASCII, 90 symbols | 0.5 | 1.0 | 2.6 | 9.4 | 26.6 | 42.0 | 62.8 | 96.0 | 45.1 | 155.5 | 285.9 | not run |
  | random two-byte characters | 0.4 | 0.7 | 1.6 | 5.8 | 22.0 | 80.4 | 304.4 | 607.3 | 83.4 | not run | not run | not run |
  | ASCII counting upwards (few trigrams repeat below 1 MiB) | 0.5 | 1.0 | 2.7 | 10.6 | 39.3 | 113.5 | 158.4 | 174.2 | 116.5 | not run | not run | not run |
  | own share held, factor 192 | 16.1 | 16.8 | 19.0 | 28.0 | 64.0 | 208.0 | 784.0 | 1,552.0 | 208.0 | refused | refused | refused |

  With the page cache filled (8.5 MiB, as on CSIC) the fixed 16 MiB alone stops being enough between 16 KiB and 64 KiB for text without repeating trigrams, and between 1 MiB and 4 MiB for log-like text. The worst case is single-byte text in which no trigram repeats, just above a power of two of trigrams, where the index doubles its table of slots (review round 2: the round lengths of the table stop just short of each step). Measured with texts from a De Bruijn sequence (heap in bytes per byte of text, and as a part of the share held): 2^16 - 64 trigrams 171 and 0.38; 2^16 + 64: 186 and 0.42; 2^18 - 64: 168 and 0.66; 2^18 + 64: 184 and 0.72; 2^19 + 64: 181 and 0.81; all 1,030,301 single-byte trigrams: 153 and 0.73; 2^20 + 64 (1,066,985 bytes): 177 and 0.85; eight texts of 2^18 + 64: 0.77. So the worst rate is about 186 bytes per byte; `ADD_TEXT_FACTOR` is 192, 3 % above it, and the rest of the margin is the 7.5 MiB that a filled page cache leaves of the fixed 16 MiB. Random ASCII levels off when its 729,000 possible trigrams are used up; multi-byte text does not (about 76 bytes per byte at any length). The most that was measured of a share held is 0.85 (in a small database; about 0.89 with a filled page cache); for long log-like text the share is about 30 times what is used, which is the price of not knowing the kind of a text. Indexing time for one worst-case text: 1 MiB 3 to 8 s, 4 MiB 14 to 40 s, 8 MiB 31 to 78 s; log-like text 0.4 s for 8 MiB. `MAX_TEXT_BYTES` is 8 MiB: the longest length at which every kind in the table is measured and re-checked by the on-request test (which asserts heap <= share for all of them); above it a row is refused (`InvalidRows`, row and limit named, before anything is written) because the kinds that take the most were not measured there and one row would take minutes. The figures above 8 MiB were taken before the limit existed.
- 1.8: Memory an open store keeps (review round 2). Before the fix an open store kept memory after an add that nobody counted. What kept it: the hash table of the text index (FTS5). Its entries are freed when a text is written, but its table of slots stays at the size it grew to for the longest text indexed on the connection (two to four slots of 8 bytes per distinct trigram) until the connection is closed; the page cache (up to 8 MiB) stays as well. Measured in release (`retained_memory_measured`, MiB of SQLite's heap in use after one text was written):

  | Text | plain connection after commit (the store before the fix) | after `PRAGMA shrink_memory` | after releasing prepared statements | after close | store now, after the add | peak of a later add of 300 short texts on the same open store (share held 16.0) |
  |------|------|------|------|------|------|------|
  | log-like 1 MiB | 2.71 | 0.12 | 0.12 | 0.00 | 0.08 | 3.25 |
  | random ASCII 1 MiB | 9.88 | 4.09 | 4.09 | 0.00 | 0.08 | 8.46 |
  | two-byte characters 1 MiB | 15.11 | 8.09 | 8.09 | 0.00 | 0.08 | 8.46 |
  | no repeating trigram, 2^20 + 64 trigrams | 40.35 | 32.09 | 32.09 | 0.00 | 0.08 | 8.48 |
  | two-byte characters 8 MiB | 72.35 | 64.09 | 64.09 | 0.00 | 0.08 | 8.51 |
  | short texts only | 0.13 | 0.09 | 0.09 | 0.00 | 0.09 | 0.53 |

  `PRAGMA shrink_memory` gives back the page cache only; what is left is exactly the slot table (4, 8, 32, 64 MiB = 2^19 to 2^23 slots); releasing prepared statements gives back nothing; only closing the connection frees the table. The fix: every add ends, on success and on failure, with a release. If it indexed a text longer than `RELEASE_TEXT_BYTES` (16 KiB) the store closes its connection and opens it again (2 to 3 ms); otherwise it runs `PRAGMA shrink_memory` and keeps the connection. An idle open store then holds 0.08 to 0.10 MiB as measured, and at most `IDLE_BYTES` (1 MiB) by construction: a table for a text of up to 16 KiB is at most 0.5 MiB. Because that small table does depend on earlier adds, the store remembers the longest text indexed on its connection and computes an add's share from the larger of that and the frame's longest, so after a 10,000-byte text later adds on the same open store hold 16 MiB + 192 x 10,000 bytes until the store is reopened. `Store::retained_bytes()` reports what an open store keeps: `IDLE_BYTES`, or, only if the reopening failed (the add itself stands), `IDLE_BYTES + 32 x` the longest text, which later adds then count in their share and try again to give back. Callers that keep a store open (2.7: one store, one add per frame) count `retained_bytes()` against their limit between adds. A side effect: the page cache is cold at the start of every add; the CSIC add took 3.2 to 3.8 s in the runs after the change against 3.2 to 3.4 s before, which is within what those runs vary by.
- 1.8: Dependency. `rusqlite` 0.32.1 with `default-features = false, features = ["bundled"]`. Release 0.32 has no `fts5` feature (the task and the design name one); `bundled` compiles SQLite 3.46.0 with `SQLITE_ENABLE_FTS5`, and a test asserts the compile option and the trigram tokenizer. The test compares the linked SQLite with the bundled header's version and pins 3.46.0; the build variables `LIBSQLITE3_SYS_USE_PKG_CONFIG` and `LIBSQLITE3_FLAGS` can change what is linked (see the workspace manifest). `Cargo.lock` grows from 199 to 206 packages: `rusqlite`, `libsqlite3-sys`, `hashlink`, `hashbrown` 0.14.5 (a third copy), `fallible-iterator`, and the build dependencies `pkg-config` and `vcpkg` (not consulted with `bundled` unless `LIBSQLITE3_SYS_USE_PKG_CONFIG` is set). No network crate; the guard test passes. `faer` (named for the store in the design) is not added; it belongs to the vector search of 1.9.
- 1.8: Deviations from the design contract. `add` takes the store rows frame, not `&[StoredLog]` (data layer decision), and returns `AddReport`, not `u64`; the caller does not supply `id` or `encoder_revision` (the store derives both); `StoredLog`, `Filter` and `Hit` are not defined yet (1.9 needs them for reading); `StoreError` has more variants (`Exists`, `InvalidName`, `InvalidIdentity`, `InvalidRows`, `Memory`) and `Missing` also carries the name. The design's file plan still says `stores/<name>/` in one place and its physical data model still speaks of an n-gram index and bitmap indexes; the store is `stores/<name>.db`, the column is `group_key`, and the indexes are ordinary SQLite indexes. Every vector is required: a store cannot hold a log without a vector, so milestone 1 (no encoder) can only fill a store with vectors from elsewhere.
- 1.8: For 1.9, 1.11 and 2.7 (review round 1). (a) NUL: the input layer of `l2d-core` accepts a NUL character in a text and keeps it (lines, CSV and JSONL; pinned by `crates/l2d-core/tests/input_adapters.rs`, the records `a\0b`), but SQLite's text index reads a text only up to its first NUL, so the rest would be stored and hashed yet never found. `add` therefore refuses a text with a NUL (`InvalidRows`, row and byte named), and so one such row fails the whole add. 1.11 (and the encoder integration) should take such rows out before the add and report them by position and reason, as the adapters report skipped records; the same goes for a text longer than `MAX_TEXT_BYTES`. `source` and `group` are not checked for NUL. (b) Case (the reviewer's probe): the trigram index folds case beyond ASCII (`ÄÖ` matches `äö`), but it does not expand characters (`STRASSE` does not match `straße`); a case-exact search needs the second condition named above. (c) Never write to `logs` with `INSERT OR REPLACE` (or `REPLACE`): its implicit delete does not fire the delete trigger unless recursive triggers are on, and the text index then drifts from the table; use the insert-then-update of `add`. (d) The id depends on the `source` string byte for byte: `data/x.log`, `data\x.log`, `./data/x.log` and an absolute path give four ids for the same log. 1.11 and 2.7 must settle one spelling of a source (separators, relative or absolute, case on Windows) before they call `log_id` or `rows_frame`.
- 1.8: For 1.11 and 2.7, the price of long records (review round 2). The add's own share grows with the longest text whatever its content: at the design's default maximum record length of 1 MiB (requirement 13) an add holds about 208 MiB of the budget (16 MiB + 192 x 1 MiB), at the store's limit of 8 MiB about 1.5 GiB. Reading is charged far less per byte than that, so a read can succeed under a budget that the later add refuses, after the encoder has already run. 1.11 and 2.7 should work out the add's share (`ADD_FIXED_BYTES + ADD_TEXT_FACTOR x longest text`, plus what `Store::retained_bytes` reports for an open store) from the logs frame and check it against the budget before encoding, and should say in the refusal that a lower record limit makes the add fit.
- 1.8: Open follow-ups, not blocking: no test kills a process during an add or a create (SQLite's journal recovery is relied on, not exercised); two processes writing one store were not tested (a statement waits 5 s for the lock, then the add fails); a store on a network share was not considered; adds from a second connection while an add runs are not prevented by the tool. The add is not fast (trigram indexing and one synchronous commit); one million logs would take about two minutes at the measured rate, which was not tried.
- 1.14: Polars is pinned `=0.52.0` (facade features `lazy`, `ndarray`; readers via a direct `polars-io` dependency with `csv`, `json`). Do not enable the facade `csv`/`json`/`strings` features and do not upgrade without re-checking `cargo tree` for network crates: 0.53+ links tokio with networking. Import Polars only as `l2d_core::polars`.
- 1.14: `InputAdapter::read` returns `(LogsFrame, InputSummary)`. `LogsFrame` (in `l2d-core` `types.rs`) validates the Logs schema and offers `frame()`, `records()`, `to_records()`, `from_records()`; it does not check text uniqueness. Parsing is a validation pass (skips, positions, per-record digest) followed by Polars parsing the accepted records; `Dedup::push` fails with `Runtime` if Polars' rows disagree with the validation pass. Labels, groups and CSIC text are derived in Rust over Polars string columns.
- 1.14: Behaviour changes in JSONL input: a record repeating the text, label or group key, and a record Polars cannot parse (for example an integer beyond 64 bits in any key), are skipped with a reason. `InputSummary.skipped` is sorted by position.
- 1.14: CSIC group key is the URL path alone: 1,632 path groups, 8 with both labels (supersedes the `METHOD /path` figures in the 1.4 notes). CSIC read: about 2.5 to 3.3 s debug, 290 ms release.
- 1.14: Workspace manifest has `[profile.dev.package."*"] debug = false`: with debug info on dependencies the linker failed intermittently (Polars debug files of about 1 GB). Clean debug build is about 110 s; `Cargo.lock` has 196 packages.
- 1.14: Shared build folder hazard: a scratch copy of the workspace built into the shared `CARGO_TARGET_DIR` leaves test binaries with the scratch path baked in, which the real workspace then reuses (seen as "fixture ... cannot find the path"). Scratch copies must build with their own `--target-dir`; the cure is `cargo clean -p <the l2d crates>`.
- 1.4: CSIC text is `METHOD URL` plus a line feed and the body when the body is non-empty; the URL keeps `http://localhost:8080` and drops ` HTTP/1.1`. Group key was `METHOD /path` until task 1.14 (now the path alone); paths are not normalized or case-folded. `l2d_core::adapters::csic2010::{split_routing, group_key}` split a text at the first `?` or line feed into routing and payload; valid for this layout only (the mini fixture puts a space before the body). `Csic2010Adapter::read_counted` gives label counts before deduplication.
- 1.4: CSIC totals on the real file: 61,065 records (36,000 good, 25,065 bad), 25,608 distinct texts (9,644 good, 15,964 bad), 1,648 group keys, no skipped rows, read in about 2 s (debug).
- 1.4: Group distribution (distinct texts): 16 groups hold both labels (9,624 good, 14,082 bad) and are 8 endpoints under GET and POST; 20 good-only groups with one text each; 1,612 bad-only groups (1,882 texts), 1,605 of them singletons. So nearly all good texts sit in 16 groups, and about 1,600 bad paths never occur as good.
- 1.4: DECIDED 2026-11-11: the CSIC group key becomes the URL path alone (task 1.14 changes the adapter); single-label groups are reported separately (requirement 9.8); default conformal error rate 0.05. Background: 11,853 of 12,094 distinct (path, payload) pairs occur under more than one method (same parameters as GET query and as POST body). Folds grouped by `METHOD /path` leak payloads between fitting and testing; grouping by path alone gives 1,632 groups, 8 of them mixed. The group definition for cross-endpoint evaluation must be decided before 1.12.
- 1.5: `l2d_baseline::hashing`: `CharHasher::row(&str) -> SparseRow` (strictly ascending `u32` indices, `f64` values, unit L2 norm, empty text gives an empty row). FNV-1a 64-bit per n-gram; bucket = low 32 bits modulo `buckets`; sign = parity of the high 32 bits. `HashingSettings { buckets, ngram_max, routing }` with `RoutingMode::{Keep, Strip, Isolate}`.
- 1.5: Row dimension is `CharHasher::dimension()`: `buckets` for keep and strip, `2 * buckets` (131,072 by default) for isolate, where payload features are in `0..buckets` and routing features in `buckets..2*buckets` (`routing_offset()`). Size weight vectors from `dimension()`, not from `buckets`. Isolate normalizes the whole row; masking the routing half needs renormalization.
- 1.5: Strip and isolate are meaningful only for texts in the CSIC adapter layout. A text without `?` or line feed is routing only: strip gives an empty row. That includes every POST row of the mini fixture (space before the body) and any ordinary log line. Task 1.7 must define the ratio for an empty row and must store the routing mode in the saved settings and the revision; fixture tests use keep.
- 1.6: `l2d_baseline::logreg`: `LogReg::{fit_sparse, fit_matrix, fit_dense, from_parts, probability_sparse, probability_dense}`, `SparseMatrix` (CSR builder), `LogRegSettings { l2, max_iter, tolerance }` (defaults 1.0 / 100 / 1e-4), fits return `(LogReg, FitReport)`; labels are `&[bool]` with `true` = bad. Objective: sum of log-losses plus `l2/2 * |w|^2`, bias not penalized.
- 1.6: Solver deviation from the wording of requirement 2.7: `linfa-logistic` 0.8.1 fits dense arrays only, so the code calls `argmin` 0.11 L-BFGS with More-Thuente line search directly (the optimizer `linfa-logistic` uses, same settings, zero start) on sparse and dense rows. `faer` is not used; `linfa-logistic` is a dev-dependency used as a test reference. `argmin` and `ndarray` were added to the design's allowed dependencies on 2026-11-11; the texts "via `faer` / `linfa`" in requirement 2.7 and task 1.6 are met in intent, not literally.
- 1.6: At CSIC fold size (20,000 x 65,536) the default settings stop at the 100-iteration cap (gradient norm 1.8e-4 against 1e-4, training accuracy 0.988; release 1.7 s, debug about 32 s). The tolerance is absolute on the summed-loss gradient, so it tightens with row count. `Termination::Converged` also covers the cost-change stop, so check `gradient_norm` when the gradient criterion matters. Task 1.7 must fix or store `max_iter` and tolerance, include them in the revision, and surface the termination.
- 1.6: Weights are bit-equal across runs and processes of one build on one machine, not across toolchains or machines (platform `exp` and `ln_1p`). Row order is part of the input.
- 1.6: Decisions of 2026-11-11: requirement 2.7 was reworded to name the `argmin` optimizer, so code and spec agree. The CSIC-size scale test is to run on request only (ignored in the normal run, run in release mode with an explicit command); this change is made after task 1.15 lands. Polars stays pinned at 0.52.0. The shared build folder is left as it is.
- 1.6: Open follow-ups, not blocking: no test drives the fit to very large margins (a naive softplus would pass); the scale test adds about 30 to 45 s to every debug test run and asserts a wall-time bound that could flake on a loaded machine.
- 1.1: The README layout table (design "Modified Files") is not owned by any task; update it when the CLI is wired (1.11).
- 2.2: GATE PASSED on 2026-11-12 (reference host, AMD Ryzen AI 9 HX 370, device `PCI\VEN_1022&DEV_17F0`). The spike's 8-bit model runs on the NPU from Rust: `ort` loads AMD's `onnxruntime.dll` at run time, the Vitis AI provider is registered with `error_on_failure`, a batch of 32 is inferred, and the provider's report places 36 of the model's 38 nodes on the NPU and 2 on the CPU (the `QuantizeLinear` of the input and the `DequantizeLinear` of the output). Same result in every run (more than ten, cold each time). Test: `crates/l2d-encoder/tests/npu_gate.rs`, ignored by default; run with `$env:L2D_RUNTIME_DIR = "<main checkout>\runtime\ryzenai"; cargo test -p l2d-encoder --test npu_gate -- --ignored --nocapture` (about 6 to 11 s, debug build is enough). The test is asked for explicitly, so it fails when it cannot prove what it is for: without the runtime directory, without one of its files, without an AMD NPU device, and with a Python interpreter on `PATH`. Only with `L2D_GATE_ALLOW_SKIP=1` does it print `SKIPPED` with the reason and pass (runtime files or device missing), or report the interpreter and go on. There is no other switch (`L2D_GATE_STRICT` of the first version is gone). The branch "no NPU device" could not be exercised on this host.
- 2.2: Runtime files (`runtime/README.md` has sizes and hashes). Five DLLs, 713,594,768 bytes: `onnxruntime.dll` (1.30.0), `onnxruntime_providers_shared.dll`, `onnxruntime_providers_vitisai.dll`, `onnxruntime_vitisai_ep.dll` (1.9.0, 518 MB), `dyn_dispatch_core.dll` (1.9.0, 170 MB). Found by removal: the gate passes without `DirectML.dll`, `onnxruntime_providers_ryzenai.dll`, `onnxruntime_vitis_ai_custom_ops.dll`, `waic_inmem.dll`, `waic_mlir.dll` (each alone and all together). There is no Ryzen AI installation on the host (no `C:\Program Files\RyzenAI`, no `RYZEN_AI_INSTALLATION_PATH`); the files are those of the Python wheel `onnxruntime-vitisai` 1.30.0 with `voe` 1.9.0 in `D:\Source\memguard_alpha\tools\oga-venv\Lib\site-packages\onnxruntime\capi\`, the environment the spike used, copied unchanged. They sit in the main checkout's `runtime\ryzenai\` (ignored by git). From the driver the process also loads `C:\Windows\System32\xrt_coreutil.dll` and `xrt_core.dll` from `DriverStore\FileRepository\kipudrv.inf_amd64_*` (both 32.00.20102.4481); nothing else from outside `System32` and the runtime directory, and no Python module (module list of the running test process, read once from outside).
- 2.2: For 2.3, the reason `RuntimeFiles::check` must run before anything is loaded: with `dyn_dispatch_core.dll` missing the process dies with an access violation (exit code `0xC0000005`) while the provider is registered. No error comes back, `error_on_failure` does not help, and the test harness prints nothing after the load line. An `onnxruntime.dll` of the wrong API level is refused by `ort::init_from` with an error (tried, next note). Not tried: an absent `onnxruntime.dll` (the test checks the files first and skips), and removing the other three one by one.
- 2.2: `ort` is pinned `=2.0.0-rc.15` (a release candidate; there is no stable 2.x) with `default-features = false, features = ["std", "load-dynamic", "vitis", "api-30"]`, a dev-dependency of `l2d-encoder` for now (2.6 makes it a normal one). `api-30` is the level of AMD's ONNX Runtime 1.30.0; with `api-31` (the crate's default) the load is refused: "ort 2.0.0-rc.15 is not compatible with the ONNX Runtime binary found at ...; expected version >= '1.31.x', but got '1.30.0'" (tried). The default features must stay off: they include `download-binaries` and `tls-native` (`ureq`, `native-tls` in the build script of `ort-sys`). `Cargo.lock` grows from 203 to 209 packages: `ort`, `ort-sys`, `libloading` 0.9.0, and `ndarray` 0.17.2, `tracing` 0.1.44, `tracing-core` 0.1.36. The last three are optional dependencies named by the feature `std` (`ndarray?/std`, `tracing?/std`); Cargo writes them into the lock file, but they are not in the dependency tree and are not compiled (`cargo tree -i ndarray@0.17.2 -e all --target all` prints nothing). So the lock file again names a second `ndarray` that nothing links. No network crate; `no_network_crates` passes. The test also uses `sha2`, `serde_json` and `windows-sys` (features `Win32_Foundation`, `Win32_Devices_DeviceAndDriverInstallation`, `Win32_Devices_Properties`, added on the dev-dependency) from the workspace.
- 2.2: API of `ort` rc.15 as used: `ort::init_from(<absolute path of onnxruntime.dll>)?.with_name(..).build()?` gives an `Environment`; `Session::builder(&environment)?.with_execution_providers([Vitis::default()...build().error_on_failure()])?.commit_from_file(model)?`; input `TensorRef::from_array_view((shape_vec, &data[..]))?` (no copy, no `ndarray` feature); `session.run(ort::inputs![name => tensor])?`; output `outputs[0].try_extract_tensor::<f32>()?` gives `(&Shape, &[f32])`. `session.inputs()[0].name()` and `.dtype().tensor_shape()` give the model's interface. No entry on `PATH` and no `SetDllDirectory` is needed: the runtime finds its provider DLLs next to `onnxruntime.dll`. The gate ran with `PATH` = `C:\Windows\System32;C:\Windows` plus the cargo directory.
- 2.2: Provider options used: `cache_dir` = `<runtime dir>\..\cache` (so `runtime/cache/`), `cache_key` = `gate-2.2-<first 12 hex of the model's SHA-256>`, `enable_cache_file_io_in_mem` = `0` (`with_cache_in_memory(false)`: the compiled model is kept on disk). Nothing else: no `config_file` (`vaip_config.json`), no `xclbin`, no `target`, no `opt_level`. The provider picks the target `AMD_AIE2P_4x8_CMC_Overlay` itself. The paths are plain drive paths (`D:\...`); the `\\?\` form that `canonicalize` returns was stripped and not tried. The cache folder `runtime/cache/<key>/` then holds `compiled.AMD_AIE2P_4x8_CMC_Overlay.xmodel` (1.4 MB), `context.json` (0.3 MB) and `vitisai_ep_report.json` (24 KB). Nothing was written to `C:\temp` or to the working directory.
- 2.2: How placement is read. The environment variable `XLNX_ONNX_EP_REPORT_FILE` = `vitisai_ep_report.json` must be set before the session is created; `std::env::set_var` in the process, before the runtime is loaded, is enough (it is `unsafe` in edition 2024: do it before any thread is started). Without the variable no report is written (tried). The report lands in the model's cache folder. Its `nodeStat` list has one entry per model node with `device` = `NPU` or `CPU`; `deviceStat` has totals named `all`, `NPU` and `VITIS_EP_CPU` (not `CPU`). The test counts `nodeStat` and requires agreement with `deviceStat`. The design's risk is real: the report is written only when the provider compiles; a session that loads the compiled model from the cache writes none (tried in a fresh process and in a second session of one process). But the report of the compilation stays in the cache folder next to the compiled model, so 2.6 can read `runtime/cache/<revision>/vitisai_ep_report.json` on a warm start and needs no `placement.json` of its own; it should treat a cache folder without the report as not compiled. The gate test removes its own cache folder first, so every run compiles and reads a report written by that run.
- 2.2: Driver version: the Windows configuration manager, through `windows-sys` (already in the workspace; the design names the `windows` crate): `CM_Get_Device_ID_ListW` with the filter class `{F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}` (`ComputeAccelerator`) and present devices, `CM_Locate_DevNodeW` for each, then `CM_Get_DevNode_PropertyW` for `DEVPKEY_Device_DriverVersion` and `DEVPKEY_Device_DeviceDesc`. The test prints every present device of that class and takes the first whose instance id starts with `PCI\VEN_1022&` (AMD), saying which. It does not compare the device id. For 2.3: on this host there is one such device, `PCI\VEN_1022&DEV_17F0&SUBSYS_17F01022&REV_10`, so `DEV_17F0` is the NPU of the Ryzen AI 9 HX 370; other processor generations have other ids, and whether another AMD device can appear in this class was not looked into, so 2.3 should decide between the vendor match and a list of known device ids. On this host: "NPU Compute Accelerator Device", driver 32.0.20102.4480 (the documents of 2026-11-11 say 32.0.20102.3930; the driver has been updated since). No process is started and no PowerShell is needed. 2.3 can move the roughly 90 lines from the test into `RuntimeFiles`.
- 2.2: Model interface of the fixture `crates/l2d-encoder/tests/fixtures/spike-enc-b32-int8.onnx` (989,260 bytes, SHA-256 `6901c42a8d1187b71b986eaaae6b7306914e9da675a17a0f4e2776719f49a638`, a copy of `D:\BuildArtifacts\log2decision\npu-cnn-spike\enc_b32_int8.onnx`, random weights): input `x`, 32-bit float, shape `[32, 64, 1, 256]` (batch, table width, 1, bytes); output `e`, 32-bit float, `[32, 128]`. So the byte-table lookup is done on the CPU before the model and the layout is channels first with a dummy height of 1; 2.5 must produce exactly this layout. The spike's Python script is not kept anywhere (searched `D:\BuildArtifacts` and `memguard_alpha`), so its provider options are not known; the options above are what works.
- 2.2: Measured (debug build of the test, machine shared with other builds at the time, so the spread is wide). Start to ready with compilation (runtime load, provider registration, compile): 2.9 to 5.7 s. Ready from the cache: 0.65 s in a fresh process (one measurement), 0.27 to 1.8 s for a second session in the same process. First inference 1.4 to 6.4 ms. 300 batches of 32 after 5 warm-up batches: 0.88 to 1.32 ms per batch, 24,200 to 36,400 logs/s (the spike measured 36,713). CPU provider of the same DLL, same model: 7.4 to 25.7 ms per batch, 1,200 to 4,300 logs/s (the CPU figure suffers most from the other builds). Lowest cosine between NPU and CPU output over the 32 rows: 0.9958 (the same in every run). The NPU output is bit-identical on repeat, in a second session and across runs. The input is copied by no one and the output is copied once per batch; text encoding, padding and normalization are not in these figures. Requirement 4.7 (ready within 30 s) and the 1,000 logs/s of 4.2 have wide margins.
- 2.2: For 2.6. The provider writes to standard error on its own: "WARNING: Logging before InitGoogleLogging() is written to STDERR" and two `register_ops` lines at every load; the spike also saw a partition warning. The CLI's one-line diagnostic on standard error will have these lines around it; standard output is untouched. One `Environment` served three sessions in one process (NPU, NPU from the cache, CPU). The CPU session is a session without providers on the same `Environment`. Not tried: a path with spaces or non-ASCII characters for the cache or the model, two processes compiling into one cache folder, a read-only cache, what a stale or damaged cache folder does, `commit_from_memory`, and whether the driver version alone identifies a compatible driver.
- 2.2: Deviations and design lines that no longer match. (a) Task text "Extract AMD Ryzen AI 1.9 runtime files": taken from the wheel in the spike's Python environment, there being no installation; the DLLs report 1.9.0. (b) Design "Allowed Dependencies": `ort` needs `std` and `api-30` besides `load-dynamic` and `vitis`, with default features off and the version pinned to a release candidate; the device query uses `windows-sys`, not `windows`. (c) Design, EncoderSession notes: "the `api-NN` feature ... recorded in the workspace manifest" is now `api-30`; the risk note about `placement.json` can be replaced by "read the provider's report kept in the cache folder"; "Unproven from Rust; task 1 gates it" is proven. (d) Design `Placement.cpu_nodes`: the provider's own name for that device is `VITIS_EP_CPU`. (e) The gate's cache key is not an encoder revision (there is no package yet). (f) `docs/2026-11-11-npu-log-decision-feasibility.md` names the older driver version. (g) The runtime directory is found through `L2D_RUNTIME_DIR` in the test; the design has no such variable, and 2.3 must decide whether the tool reads it. (h) The test compares with the CPU provider, opens a second session and profiles it, which the task does not ask for; they are evidence for the gate and for 2.6, not encoder code.
- 2.2: Found on the way, not part of the gate: in a fresh checkout on this host (`core.autocrlf` is on) `tests/fixtures/mini-labelled.csv` is written with CRLF line ends, and `fixture_row_with_wrong_field_count_is_skipped_at_position_121` in `crates/l2d-core/tests/input_adapters.rs` then fails (the reason names line 121, the test expects line 122). The main checkout has the file with LF and passes. The gate's worktree restored LF in its working copy to run the suite (no change to the committed file). Fixed in the review round: `tests/fixtures/.gitattributes` has `*.csv -text` (the CSV is the only file there), and the test passes on a fresh checkout of the file in the worktree. The main checkout's LF copy is unaffected.
- 2.2: Review round 1 (2026-11-12), what the gate test asserts now. (1) Placement, the compiler's plan: the report's `nodeStat` may name only `NPU` and `CPU` (another device string is an error that names it); the counts must equal the `deviceStat` totals `NPU` and `VITIS_EP_CPU` (an absent total counts as 0) and `all`, and the two must add up to `all`; NPU nodes must be more than 0. (2) Execution, from ONNX Runtime's own profile: the second NPU session is opened with `with_profiling(<prefix>)`; after one inference `session.end_profiling()` returns the file name; the file is a JSON list of events, and events with `cat` = `Node` and a name ending in `_kernel_time` are executed nodes with `args.provider`. On this model the profile has 3 such events, all `VitisAIExecutionProvider`, none of the CPU provider: the provider's compiled subgraphs appear as its nodes, and the two nodes the report gives to `VITIS_EP_CPU` run inside the provider too, so the runtime's CPU provider executes nothing. The test requires at least one node of `VitisAIExecutionProvider`. This tells who executed the nodes, not on which hardware the provider ran them; the hardware evidence remains the report, the timing and the driver modules. 2.6 can use the same profile for its placement log. (3) Output: the lowest cosine between NPU and CPU output must be at least 0.99 (requirement 4.6; measured 0.9958, so the margin is small and a trained model must be re-measured), and the NPU output must not be bit-identical to the CPU output (it is deterministic and differs, since the NPU computes in its own 8-bit arithmetic; identical numbers would mean the CPU provider served the session). The profile file is written next to the cache folder (`runtime/cache/<key>-profile_<time>.json`) and removed by the test.
- 2.2: Licence of the runtime files, found in the review round and not settled: the wheel's metadata declares `License-Expression: MIT`, but its licence file says "By using this wheel file, you agree to the EULAs given below: AMD EULA and Third Party Notices - https://ryzenai.docs.amd.com/en/latest/licenses.html". ONNX Runtime's own MIT licence and third-party notices ship in the wheel under `onnxruntime/`. The repository does not redistribute the DLLs (git-ignored, obtained by the user); `runtime/README.md` says so. Nobody has read AMD's terms for this project; the owner should, before the tool is handed to others.
