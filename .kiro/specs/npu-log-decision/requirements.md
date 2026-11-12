# Requirements Document

## Introduction

The Log Triage Tool helps a forensic log analyst decide which text logs deserve
attention. It runs locally on a laptop. For every log it gives a ratio of bad to
good, drawn from logs whose outcome is already known, and it ranks logs so that
those that look good are scored down. A small encoder trained on log text runs
on the laptop's NPU and turns logs into vectors; the vectors are stored next to
the log text so that both similarity search and text search work. The tool also
measures the vectors themselves, so that a smaller or faster encoder can be
judged by evidence.

Background, measurements and decisions are in `brief.md` and in
`docs/2026-11-11-embedding-speed-benchmark.md`. Two results shape these
requirements: a general text-embedding model on the NPU managed 2.3 logs per
second, and general embeddings separated attacks worse than character-level
features on HTTP CSIC 2010.

## Boundary Context

- **In scope**:
  - Reading text logs and labelled log datasets from local files.
  - Reading log records from zip archives in batches, without extracting them.
  - A fitted linear head on the encoder's vectors as a further scorer, and a ranking of logs unlike anything stored.
  - A baseline bad/good ratio from character-level features, fitted deterministically.
  - Agnostic threat scoring barrier: isolating routing context (URLs, endpoints, hosts) from payload anomaly signals so threat scores derive from general syntactic anomalies and attack patterns rather than hardcoded training-set site topography.
  - Pre-validating neural encoder architecture quality offline via `uv` on held-out grouped splits before NPU deployment.
  - Packaging the encoder as an identified artifact (`model.int8.onnx`, `table.bin`, `encoder.json`) and running it on the NPU via ONNX Runtime.
  - Storing logs with their vectors in an embedded SQLite database with full-text search (FTS5) and exact SIMD vector search (`faer`), without async or external database runtimes.
  - Triage output per log: ratio, evidence with endpoint diversity metrics, abstention, ranking.
  - Comparing two groups of logs for population shift (MMD permutation test).
  - Assessing a set of vectors: information loss, clustering capability, properties.
  - Grouped evaluation and feasibility reports on HTTP CSIC 2010 measuring cross-endpoint generalization.
- **Out of scope**:
  - Language-model stages, such as explaining a triage decision. They come
    later in the triage flow and get their own spec.
  - Evaluation on the AIT Log Data Set V2 (step 2) and on the thesis Sysmon
    data.
  - Live or streaming ingestion, alerting, and any graphical interface.
  - A bad/good share per source or time window. Only the per-log ratio is
    required here.
  - Blocking, deleting or otherwise acting on logs. The tool ranks; the analyst
    decides.
- **Adjacent expectations**:
  - The host provides a working NPU and its driver. The tool reports when it
    cannot use the NPU; it does not install or repair drivers.
  - Datasets are obtained by the analyst and placed under `data/<dataset>/`.
    The tool does not download them.
  - The Venn project is the model for the decision operations. Its software is
    not a dependency of the tool.

## Requirements

### Requirement 1: Log input
**Objective:** As a log analyst, I want to load text logs and labelled datasets from local files, so that the same tool works on web logs, attack traffic and CTF material.

#### Acceptance Criteria
1. When the analyst supplies a file of text logs, the Log Triage Tool shall read each log record as one item with its text and its position in the source.
2. When the analyst supplies a labelled dataset, the Log Triage Tool shall read each record's label as good or bad together with its text.
3. When the same log text occurs more than once in an input, the Log Triage Tool shall process the text once and keep the number of occurrences.
4. If a record cannot be read, then the Log Triage Tool shall skip it, continue with the next record, and report the count and positions of skipped records.
5. If a log text is longer than the encoder's input length, then the Log Triage Tool shall mark the record as truncated in its output.
6. The Log Triage Tool shall record for each loaded input its source path, record count, distinct text count, and a content hash.

### Requirement 2: Baseline ratio from character-level features
**Objective:** As a log analyst, I want a bad/good ratio that needs no trained encoder, so that triage works from the first day and every later model has a baseline to beat.

#### Acceptance Criteria
1. When labelled logs are supplied for fitting, the Log Triage Tool shall fit a baseline scorer on character-level features of the log text.
2. When a log is scored with the baseline scorer, the Log Triage Tool shall return a bad/good ratio between 0 and 1, where higher means more likely bad.
3. If fewer than ten labelled logs, or fewer than five of either label, are supplied for fitting, then the Log Triage Tool shall refuse to fit and state the counts it found.
4. The Log Triage Tool shall produce the same baseline ratios when fitted twice on the same labelled logs with the same settings.
5. The Log Triage Tool shall make the baseline scorer usable when no NPU and no trained encoder are available.
6. The Log Triage Tool shall provide an option to separate or normalize routing context (e.g. method and URL path) from payload query/body tokens during feature extraction, ensuring that baseline ratios assess payload anomalies agnostically rather than memorizing endpoint identities.
7. The Log Triage Tool shall solve the baseline logistic regression deterministically with a library optimizer (the `argmin` L-BFGS with its library line search, as used by `linfa-logistic`), without stochastic methods and without a hand-written line search.

### Requirement 3: Training the log encoder
**Objective:** As a log analyst, I want to train a small encoder on log text, so that the vectors carry the character-level signal that general text models miss.

#### Acceptance Criteria
1. Prior to freezing an encoder package for NPU deployment, the candidate encoder architecture shall be validated on held-out grouped splits (ensuring unseen endpoints) to verify that its cross-endpoint ROC AUC matches or exceeds the baseline threshold.
2. When the analyst starts training with a set of log texts, the Log Triage Tool shall produce an encoder with a fixed input length and a fixed output dimension.
3. When training finishes, the Log Triage Tool shall assign the encoder an identity consisting of a name, a revision derived from its content, and its output dimension.
4. When training finishes, the Log Triage Tool shall write a training record with the input datasets and their hashes, the settings, the random seed, and the duration.
5. The Log Triage Tool shall train without using the NPU.
6. When training is repeated with the same inputs, settings and seed on the same host, the Log Triage Tool shall produce an encoder whose triage quality on the evaluation set differs by no more than 0.01 ROC AUC.
7. If training is interrupted, then the Log Triage Tool shall leave any previously completed encoder unchanged.

### Requirement 4: Encoding on the NPU
**Objective:** As a log analyst, I want logs encoded on the laptop's NPU, so that triage is fast and uses little energy.

#### Acceptance Criteria
1. When logs are submitted for encoding, the Log Triage Tool shall return one vector per distinct log text, with the dimension stated in the encoder identity.
2. While the NPU is available, the Log Triage Tool shall encode at least 1,000 distinct logs per second, sustained over 10,000 logs, on the reference host.
3. When an encoding run completes, the Log Triage Tool shall record the encoder identity, the driver version, the logs per second, and evidence of whether the work ran on the NPU.
4. If the NPU cannot be used, then the Log Triage Tool shall state the reason and encode on the CPU only when the analyst has allowed that fallback.
5. If encoding ran on the CPU, then the Log Triage Tool shall mark the run record as a CPU run.
6. The Log Triage Tool shall return vectors for the same log text and encoder identity that agree with a cosine similarity of at least 0.99 between the NPU and the CPU.
7. When the encoder is loaded for the first time, the Log Triage Tool shall be ready to encode within 30 seconds.

### Requirement 5: Storage and search
**Objective:** As a log analyst, I want logs stored with their vectors in a lightweight embedded database, searchable both by similarity and by text, without requiring external database services or async runtimes.

#### Acceptance Criteria
1. When logs are stored, the Log Triage Tool shall persist them in an embedded SQLite database keeping for each log its text, source, occurrence count, label if known, vector, and the identity of the encoder that produced the vector.
2. The Log Triage Tool shall index log text with SQLite FTS5 to answer exact and containment text queries.
3. When the analyst searches by a log or a vector, the Log Triage Tool shall compute exact L2 distances against stored vectors using SIMD matrix operations (`faer`) and return the nearest stored logs with their distances.
4. When the analyst combines a text search with a similarity search, the Log Triage Tool shall filter rows through FTS5 before distance ordering.
5. Where a search is restricted by label or by source, the Log Triage Tool shall apply SQL index restrictions.
6. If a vector from one encoder identity is used against a store built with another, then the Log Triage Tool shall refuse the operation and name both identities.
7. The Log Triage Tool shall keep stored logs available after it is restarted, with zero background service processes and no asynchronous runtime requirements.

### Requirement 6: Triage decision per log
**Objective:** As a log analyst, I want each log to carry a bad/good ratio with the evidence behind it, so that I can score down logs that look good and spend my time on the rest.

#### Acceptance Criteria
1. When a log is triaged, the Log Triage Tool shall return a bad/good ratio between 0 and 1 derived from stored logs with known labels.
2. When a log is triaged, the Log Triage Tool shall return the evidence for the ratio: the nearest labelled logs, their labels, their distances, and how many were used.
3. If fewer labelled logs than the configured minimum support are within reach of a log, then the Log Triage Tool shall abstain for that log and say so in place of a ratio.
4. When a set of logs is triaged, the Log Triage Tool shall return them ranked from highest to lowest ratio, with abstained logs listed separately.
5. When the analyst sets a review budget as a number of logs, the Log Triage Tool shall return that many highest-ranked logs and mark the remaining logs as scored down.
6. When a log is triaged, the Log Triage Tool shall state which scorer produced the ratio: the baseline scorer, the encoder-based neighbour scorer, the linear head (Requirement 14), or several of them.
7. When evidence is reported for a triage decision, the Log Triage Tool shall report the count of nearest neighbors sharing the same endpoint/group key versus distinct endpoints.
8. If nearest neighbors are solely composed of benign logs from the identical endpoint while payload anomaly indicators are absent, the decision logic shall avoid falsely suppressing the bad/good ratio.
9. The Log Triage Tool shall not delete, block or alter logs as a result of triage.

### Requirement 7: Population shift between groups of logs
**Objective:** As a log analyst, I want to test whether two groups of logs differ as populations, so that I can tell whether today's logs still look like a known-good baseline.

#### Acceptance Criteria
1. When the analyst supplies two groups of logs, the Log Triage Tool shall return a test statistic and a p-value for the hypothesis that both groups come from the same population.
2. When a shift test completes, the Log Triage Tool shall report the size of each group and the number of permutations used.
3. If either group has fewer than 30 logs, then the Log Triage Tool shall refuse the test and state the group sizes.
4. The Log Triage Tool shall return the same result when the test is repeated on the same groups with the same seed.
5. The Log Triage Tool shall describe the p-value in its output as a test result and not as the probability that a shift occurred.

### Requirement 8: Assessment of a vector set
**Objective:** As a log analyst, I want reusable measurements of a set of vectors, so that I can judge whether a smaller or faster encoder still fits.

#### Acceptance Criteria
1. When the analyst requests an information-loss assessment, the Log Triage Tool shall report the variance explained by leading components, an intrinsic-dimension estimate, and the effect of shortening the vectors to requested lengths.
2. When the analyst supplies a reference vector set for the same logs, the Log Triage Tool shall report how much the nearest-neighbour sets of the two vector sets overlap.
3. When the analyst requests a clustering assessment, the Log Triage Tool shall report a silhouette score for a requested number of clusters and, where labels exist, how well the labels separate.
4. Where labels and a group key exist, the Log Triage Tool shall report label separability within groups that share the same key.
5. When the analyst requests a properties assessment, the Log Triage Tool shall report the distribution of pairwise cosine similarity, vector norms, anisotropy, duplicate rate, and dimension.
6. When the analyst requests a two-dimensional projection, the Log Triage Tool shall write the projected coordinates with labels to a file.
7. When two encoders are compared, the Log Triage Tool shall report the same measurements for both on the same logs, side by side.

### Requirement 9: Evaluation and feasibility report
**Objective:** As a log analyst, I want triage quality and cost reported on a public labelled dataset, so that I can decide whether the encoder earns its place over the baseline.

#### Acceptance Criteria
1. When an evaluation is run on a labelled dataset, the Log Triage Tool shall report ROC AUC, average precision, and recall at the configured review budget for each scorer.
2. The Log Triage Tool shall evaluate on distinct log texts using strict grouped splits (where all logs of an endpoint/group key fall on one side of every split between fitting and testing), and shall explicitly report both overall ROC AUC and cross-endpoint ROC AUC to verify that threat scores generalize agnostically without overfitting to training set specifics.
3. When an evaluation is run, the Log Triage Tool shall report the abstention rate and the quality measured on the logs that were not abstained.
4. When an evaluation is run, the Log Triage Tool shall report logs per second, time to the first result, and an energy proxy for each scorer.
5. When the encoder-based scorer and the baseline scorer are both evaluated, the Log Triage Tool shall state which has the higher ROC AUC and by how much.
6. When an evaluation completes, the Log Triage Tool shall write a report that names the dataset and its hash, the encoder identity, the settings, and the date.
7. The Log Triage Tool shall support HTTP CSIC 2010 as the first evaluation dataset.
8. When an evaluation is run, the Log Triage Tool shall compute the cross-endpoint ROC AUC only on groups that hold both good and bad logs, and shall report the logs of groups that hold a single label separately, with their count and the share of them ranked within the review budget.

### Requirement 10: Local operation and housekeeping
**Objective:** As a log analyst, I want the tool to run on my laptop without extra runtimes or network access, so that log data stays local and the tool is simple to keep.

#### Acceptance Criteria
1. The Log Triage Tool shall run on the reference host without a Python interpreter installed.
2. The Log Triage Tool shall perform all operations without sending log data, vectors or labels over the network.
3. The Log Triage Tool shall read and write encoders only under the project's model directory.
4. The Log Triage Tool shall read datasets only from the project's data directory or from a path the analyst supplies.
5. If a required encoder, dataset or store is missing, then the Log Triage Tool shall name the missing item and its expected location.
6. When an operation fails, the Log Triage Tool shall exit with a non-zero status and a message that states what failed.
7. The Log Triage Tool shall offer every operation from the command line with machine-readable output on request.

### Requirement 11: Calibrated decisions with a stated error rate
**Objective:** As a log analyst, I want each triage decision to carry a stated error rate, so that scoring a log down is a bounded risk and not only a ranking.

#### Acceptance Criteria
1. When labelled calibration logs are supplied for a fitted scorer, the Log Triage Tool shall calibrate that scorer using only logs that were not used to fit it.
2. When a calibrated scorer triages a log at a chosen error rate, the Log Triage Tool shall return the set of labels it cannot rule out at that error rate: good, bad, both, or neither.
3. The Log Triage Tool shall calibrate separately for each label, so that the chosen error rate applies to good logs and to bad logs individually.
4. Where group keys exist and a group holds at least the configured minimum number of calibration logs of a label, the Log Triage Tool shall calibrate that group separately, and shall state for each decision whether the group's or the pooled calibration was used.
5. If the calibration logs of a label are too few to support the chosen error rate, then the Log Triage Tool shall refuse that error rate and state the counts found and the smallest error rate they support.
6. When a set of logs is triaged with a calibrated scorer, the Log Triage Tool shall mark as scored down by calibration only the logs whose label set is good alone, shall rank for review the logs whose label set is bad alone, and shall list the logs whose label set is both or neither separately with the reason: uncertain, or unlike the calibration logs.
7. When a review budget marks further logs as scored down, the Log Triage Tool shall state for each scored-down log whether calibration or only the budget scored it down.
8. When an evaluation is run with a calibrated scorer, the Log Triage Tool shall report, for each label, the achieved error rate against the chosen error rate, overall and per group, and the share of logs in each label set.
9. The Log Triage Tool shall state in its output that the error rate holds only for logs that resemble the calibration logs.
10. The Log Triage Tool shall return the same label sets when calibration and triage are repeated on the same logs with the same settings.

### Requirement 12: Batches sized to the available memory
**Objective:** As a log analyst, I want the tool to size its batches to the memory that is free on my laptop, so that large datasets load without exhausting memory and small ones are not slowed down.

#### Acceptance Criteria
1. When a dataset is about to be loaded, the Log Triage Tool shall read the free physical memory of the host and choose a batch size whose estimated memory use stays within the configured share of that free memory.
2. Where the analyst sets a batch size or a memory limit, the Log Triage Tool shall use the analyst's value in place of its own choice.
3. When an operation processes a dataset in batches, the Log Triage Tool shall report the free memory it observed, the memory limit it applied, and the batch size it chose.
4. If the smallest allowed batch does not fit within the memory limit, then the Log Triage Tool shall refuse the operation and state the memory it needs and the memory available.
5. If an operation needs the whole dataset in memory at once and its estimated memory use exceeds the memory limit, then the Log Triage Tool shall refuse before loading and state the estimate and the limit.
6. The Log Triage Tool shall produce the same results for a dataset regardless of the batch size used.
7. While a dataset is processed in batches, the Log Triage Tool shall read the free physical memory again before each batch, shall reduce the batch size when the free memory has fallen below the configured reserve, and shall raise it again, up to the configured maximum batch size, when the free memory has recovered.
8. When the batch size changes during an operation, the Log Triage Tool shall record the change with the free memory that caused it, and shall report the smallest and largest batch size used when the operation ends.
9. The Log Triage Tool shall keep its own memory use within the memory limit for the whole operation, so that other programs on the host are not starved of memory.

### Requirement 13: Zip archives as input, in batches
**Objective:** As a log analyst, I want to point the tool at a zip archive of logs that is larger than my laptop's memory, so that I can work on large datasets without extracting them and without the tool exhausting memory.

Added on 2026-11-12 (owner approval of `amendment-2026-11-12-zip-batch.md`). Reading the label files of the AIT Log Data Set is not part of this requirement.

#### Acceptance Criteria
1. When the analyst supplies a zip archive, the Log Triage Tool shall read log records from the entries of the archive without extracting the archive to disk.
2. Where the analyst gives entry patterns, the Log Triage Tool shall read only the entries whose path matches a pattern.
3. When an archive has been read, the Log Triage Tool shall report the number of entries read and, for every entry it did not read, the entry's path and the reason.
4. If an entry is not text, is encrypted, is a nested archive other than a gzip-compressed text file, or uses a compression method the tool does not support, then the Log Triage Tool shall skip the entry, continue with the next entry, and report it.
5. When an entry is a gzip-compressed text file, the Log Triage Tool shall read the log records inside it.
6. If an entry is damaged or fails its checksum, then the Log Triage Tool shall stop reading that entry, keep the records read from other entries, report the entry, and exit with a non-zero status.
7. The Log Triage Tool shall give each record read from an archive a position that names the entry path and the line within the entry.
8. While an archive is read, the Log Triage Tool shall process it in batches under Requirement 12, so that its memory use stays within the memory limit whatever the size of the archive or of a single entry.
9. If a single log record is longer than the configured maximum record length, then the Log Triage Tool shall skip the record and report its position and length.
10. The Log Triage Tool shall produce the same records, occurrence counts and input summary for an archive as for the same files read unpacked, whatever the batch size.
11. When an operation writes per-log results for an input that is processed in batches, the Log Triage Tool shall write the results to files in parts as the batches complete, and shall state where the parts are.
12. If the deduplication state of an input would exceed the memory limit, then the Log Triage Tool shall either continue with deduplication on disk or refuse and state the memory needed and the memory available; it shall not exceed the limit.
13. The Log Triage Tool shall record for an archive input the archive path, the archive's content hash, and for each entry read its path and record count.

### Requirement 14: Fitted linear head and outlier ranking on vectors
**Objective:** As a log analyst, I want a second decision model on the encoder's vectors that is fitted to my labelled logs, and a signal for logs unlike anything stored, so that I can compare it with the nearest-neighbour ratio and the baseline, and see what is new.

Added on 2026-11-12 (owner approval of `amendment-2026-11-12-vector-decision-heads.md`, after comparing Venn's operations with this spec in `docs/2026-11-12-venn-vdm-gap-list.md`). The scorers stay separate; the decision to investigate, which brings their results together, is not part of this requirement.

#### Acceptance Criteria
1. When labelled stored logs are supplied for fitting, the Log Triage Tool shall fit a linear head on their vectors that returns a bad/good ratio between 0 and 1 for a vector, where higher means more likely bad.
2. If fewer than ten labelled logs, or fewer than five of either label, are supplied for fitting, then the Log Triage Tool shall refuse to fit and state the counts it found.
3. Where group keys exist, the Log Triage Tool shall choose the head's penalty by cross-validation in which all logs of a group fall on one side of every split, and shall report the penalty chosen and the score of each candidate.
4. The Log Triage Tool shall produce the same head and the same ratios when fitting is repeated on the same logs with the same settings.
5. If a vector from one encoder identity is given to a head fitted on another, then the Log Triage Tool shall refuse and name both identities.
6. When a head is saved, the Log Triage Tool shall give it a revision derived from its content and record the encoder identity and the logs it was fitted on, so that calibration can exclude them.
7. When a log is triaged with the linear head, the Log Triage Tool shall name the linear head as the scorer that produced the ratio.
8. The Log Triage Tool shall calibrate the linear head under Requirement 11 in the same way as any other scorer.
9. When an evaluation is run, the Log Triage Tool shall report the linear head next to the baseline scorer and the neighbour scorer on the same splits.
10. When the analyst requests an outlier ranking for a set of logs against a store, the Log Triage Tool shall return for each log the distance to its k-th nearest stored log and rank the logs from the largest distance down.
11. The Log Triage Tool shall describe the outlier figure in its output as a distance and not as a probability.
12. If the store holds fewer logs than the outlier ranking needs, then the Log Triage Tool shall refuse and state the number stored and the number needed.
13. When evidence is reported for a neighbour-based decision, the Log Triage Tool shall report, for each label, the number of neighbours used and their mean distance.
14. The Log Triage Tool shall keep fitting a head and ranking outliers within the memory limit of Requirement 12, refusing before loading when the vectors needed do not fit.
