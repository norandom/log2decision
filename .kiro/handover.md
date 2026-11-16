# Handover: npu-log-decision

Written 2026-11-15, end of session. Branch `main`, pushed (this file's commit
follows `f931482`). Remote: https://github.com/norandom/log2decision

This file describes the state as it is now. It replaces the handover of
2026-11-14.

## Where things stand

The spec `npu-log-decision` is in implementation
(`.kiro/specs/npu-log-decision/`). Each finished task was implemented by one
agent, checked by an independent reviewer, then committed and pushed.

Tests on `main` (last run after the 2.17 merge, `28c69b3`; only documents have
changed since): 711 passing, 0 failing, 47 ignored, in 69 test binaries.
`cargo fmt` and `clippy` are clean.

What works from the command line: a file or a zip archive of logs is read,
scored, calibrated and triaged with the baseline scorer; `baseline score` on an
archive writes Parquet parts with a manifest. The encoder runs on the NPU as a
tested session, and logs can be encoded into the store (2.7, library code); no
command uses the encoder yet.

**Update 2026-11-16: tasks 1.19, 1.21 and 1.22 finished, reviewed (APPROVED)
and merged. Next is 1.23 (see "Then, in order"); read the `- 1.22, review`
note in tasks.md first, it carries two points into 1.23.**

### Finished tasks

| Task | What it is | Commit |
| --- | --- | --- |
| 1.1 | Rust workspace, shared core types | `92f2714` |
| 1.2 | Command-line skeleton (stubs), test fixture | `fd64eb9` |
| 1.3 | Readers for lines, CSV, JSONL; deduplication | `5895ba7` |
| 1.4 | HTTP CSIC 2010 reader | `90da638` |
| 1.5 | Character n-gram hashing with routing modes | `ff89b83` |
| 1.6 | Deterministic logistic regression, sparse and dense | `3675d61` |
| 1.14 | Polars as the data layer; CSIC group key = URL path | `b1e9198` |
| 1.15 | Batches sized to free memory; memory limit that holds | `786ff1b` |
| 1.7 | Baseline scorer: fit, score, save, load | `d5d0be7` |
| 1.13 | Conformal calibration over any scorer | `fc1f253` |
| 1.8 | SQLite store with a text index and the vectors | `5f4a9b5` |
| 1.18 | Own conformal code; the `conformal-prediction` crate is gone | `a60f981` |
| 1.9 | Search by phrase, label, source and nearest vectors | `b0def83` |
| 1.10 | Triage: ranking, review budget, placement by label set | `52918fc` |
| 2.2 | Hardware gate: the 8-bit spike model runs on the NPU from Rust | `bdabad2` |
| 2.3 | Runtime file check, NPU device query, `L2D_RUNTIME_DIR` | `3b12989` |
| 2.13 | Provisioning from AMD's NuGet ZIP (release 1.8.0) | `e3413ef` |
| 2.14 | Provisioning release 1.9.0 from AMD's NuGet package | `bb2f4f1` |
| 2.5 | Log text to the encoder's input (`ByteCodec`) | `9ced82b` |
| 1.11 | Commands `ingest`, `baseline fit`, `baseline score`, `calibrate`, `triage` | `7e70f43` |
| 2.4 | Encoder package with a verified revision; the 1.8.0 build switch removed | `150cef6` |
| 1.12 | Baseline gate: grouped evaluation on CSIC in `l2d-eval` | `775b08a` |
| 2.12 | Rows right and fast on the NPU: the texts as rows of one image | `51aa056` |
| 2.1 | Science gate: decided by the owner on 2026-11-13 | `51aa056` |
| 1.16 | Zip archives read in batches under the memory limit | `8ba2e95` |
| 2.6 | Encoder session on the NPU, NPU only by evidence | `0491f8d` |
| 2.15 | Gate: embedding qualification with diagrams (did not pass for the published encoder) | `a713fd8` |
| 1.17 | Results in Parquet parts on disk, deduplication on disk | `e7cbbce` |
| 2.16 | Gate: hierarchical byte encoder, entropy patches against two controls (record) | `acb9767` |
| 2.7 | Encode logs and store the vectors, one transaction, release recorded | `7e1bc77` |
| 1.20 | `l2d ingest` deduplicates on disk only when needed; partition room by distinct keys | `15ee402` |
| 2.17 | Gate: stable entropy-patch encoder; vectors do not classify on unseen endpoints; owner decision | `28c69b3`, `8fded88` |
| 1.19 | Archive entries read at once: `ingest` 2 by default (faster), `baseline score` 1 (not faster) | merge of `6088015` (2026-11-16) |
| 1.21 | Mark sets on a file, archive or store; store logs by `log_id`, store hash over sorted ids (owner decision 2026-11-16) | merge of `a53708e` (2026-11-16) |
| 1.22 | Redaction: proposals with counts, confirmed rules before hashing; on CSIC host and five endpoint paths proposed | merge of `cfd2ade` (2026-11-16) |

Spec changes on 2026-11-15:
- **Requirement 16**, the analyst flow (amendment
  `amendment-2026-11-15-analyst-flow.md`, approved and merged, `f931482`):
  - 6.1 and 6.6 changed; 14.1 to 14.9 (linear head) deferred;
  - design section "MarkSet, Redaction and the mark-set score";
  - tasks 1.21 to 1.23 and gate 2.18 added; 2.8 re-scoped; 2.10 deferred; 2.9
    extended.
- **Requirement 15** (resource management on a busy machine): wording approved
  by the owner as written.

## Open at the end of the session

### Task 1.19, parallel archive entries: DONE 2026-11-16 (record below kept for history)

Finished from `fa48bc1`, reviewed independently (APPROVED), merged into
`main`. The failing CLI test was a test limit, not the accounting; the
on-disk tests now take their limit from the readers' measured need. Measured
(512 MB, all entries, ranges): `ingest` 74–81 s at 1, 50–71 s at 2;
`baseline score` 434–482 s at 1, 376–458 s at 2, 451–469 s at 4 (not
consistently faster, more memory). Decision by the task's rule: `ingest`
defaults to 2 when the readers fit, `baseline score` to 1. Details in the
`- 1.19:` notes of tasks.md. On 2026-11-16 the left-over build folders listed
under "Things that will bite" were removed with `cargo clean` (about 49 GiB) because
the all-entries `ingest` was refused for disk space (88.3 GiB asked, 84.3 GiB
free).

Record as of 2026-11-15:

- **State:** the implementer was stopped by the owner's end of session while
  it measured. Its work is committed as `fa48bc1` ("wip") on local branch
  `worktree-agent-a0eff62fb80c1b7d1` in the worktree
  `.claude/worktrees/agent-a0eff62fb80c1b7d1` (not pushed, base `8fded88`).
  Not reviewed. The checks were not run on this state. Its last test log
  (`task-1.19\green-cli.txt`) has one failing CLI test.
- **Files:** new `crates/l2d-core/src/zip_input/parallel.rs`,
  `crates/l2d-core/tests/zip_parallel.rs`; changed `zip_input.rs`,
  `zip_directory.rs`, `l2d-cli/src/cmd/{common,ingest,baseline_archive}.rs`,
  `l2d-cli/tests/archive_parts.rs`. No `- 1.19:` notes in tasks.md yet.
- **Measurements so far** (`D:\BuildArtifacts\log2decision\task-1.19\ait-measurement.log`,
  `russellmitchell.zip`, `*.log` entries, 512 MB, release, single runs per
  setting):

  | Command | Entries at once | Wall time | Peak working set |
  | --- | --- | --- | --- |
  | `ingest` | 1 | 32.9–34.3 s | 350–355 MiB |
  | `ingest` | 2 | 27.9–30.6 s | 356–398 MiB |
  | `ingest` | 4 | 27.4–31.8 s | 362–416 MiB |
  | `baseline score` | 1 | 171–195 s | 323 MiB |
  | `baseline score` | 2 | 187–190 s | 351–363 MiB |
  | `baseline score` | 4 | 188 s | 410 MiB |

  Counts are identical in every run (22,880,272 records, 5,642,082 distinct),
  and so are the `rows_sha256` of `baseline score`. This is a reading of
  unfinished runs, not a result. `ingest` gains about 10 to 20 %, `baseline
  score` none. The all-entries series was not run.
- **Next:** resume the implementer in that worktree, or start a new one from
  `fa48bc1`. It must:
  - fix the failing CLI test and finish the measurement (all entries, two
    rounds), then run the checks and write the notes;
  - apply the task's rule: keep the change only where it is faster within the
    same memory limit; for `baseline score` the reading so far says it is not.

  Then the independent review, the merge with `git merge --no-ff --no-commit
  worktree-agent-a0eff62fb80c1b7d1` (tasks.md notes: keep both sides), checks,
  tick, README row, push, and removing the worktree.

### Then, in order (one heavy job at a time)

1. 1.23 the score fitted from a mark set with the held-out-group report
   (1.21 mark sets and 1.22 redaction done 2026-11-16). These need neither
   the NPU nor the encoder. For 1.22 read the `- 1.21:` notes: a mark set of
   an archive must be checked with the entry patterns it was made with
   (`require_input(summary, archive_entries)`), and a store mark set's ids are
   held in memory, refused rather than spilled.
   Then 1.24 (added 2026-11-16 by the owner): guards for a busy machine
   (15.6 to 15.10), reader robustness tests, 128 reader pieces measured,
   store mark sets on disk, a real concurrent-writer test.
2. Gate 2.18: does redaction make scores agnostic? On the 2.17 folds, run
   alone.
3. 2.8 (re-scoped: grouping of a mark set and "close enough"), 2.11 outlier
   ranking, 2.9 command wiring.
4. Milestones 3 and 4.

## Decisions waiting for the owner

None. 2.17, the analyst-flow amendment and Requirement 15 were decided on
2026-11-15.

## Decisions in force

From 2026-11-11:

- Output is a bad/good ratio per log for triage, not a verdict.
- Rust only. No Python in the tool; Python only during development, in a `uv`
  environment that belongs to this project.
- General text embeddings were dropped (EmbeddingGemma: 2.3 logs/s on the NPU,
  worse than character n-grams).
- The baseline scorer is character n-grams plus logistic regression.
- Polars is the one data layer, pinned to 0.52.0 (later releases link
  networking code).
- No crate that can open a network connection. A test guards the lock file.
- CSIC is grouped by URL path alone. Single-label groups are reported
  separately from the cross-endpoint score.
- Memory (requirement 12): read free memory before loading, size and rescale
  batches, refuse instead of running out.
- The store is SQLite with its text-search extension.

From 2026-11-12:

- Conformal calibration is own code.
- Triage places logs by label set and states that the error rate holds only for
  logs that resemble the calibration logs.
- A single log text in the store may be at most 1 MiB. `faer` stays for the
  vector search.
- Requirement 13: zip archives read in batches without extracting; gzip the
  only nested format; results to `<project>/work/<run>/`; maximum record length
  64 KiB.
- The runtime files are provisioned from AMD's packages by the method of the
  sibling project `npu_rag_from_folder`.
- The project code is under the MIT licence.
- Design text is brought in line after each approved task.

From 2026-11-13:

- **Decide on measured metrics, not estimates.** AMD release 1.9.0, "always
  forward".
- **NPU only.** No CPU fallback, no NPU-against-CPU comparison. 4.5 withdrawn;
  4.6: the same vector for the same text whatever its place in a batch. A host
  without an NPU uses the baseline scorer.
- **Baseline gate figure: pooled,** the mean over folds beside it. L2 penalty
  stays at 1 until repeated on AIT.
- **NPU input stays float.** Requirement 1.5 has two truncation flags.
- **Zip input (1.16):** own zip-directory reader; lines before damage kept;
  exit code 2 for a damaged entry.

From 2026-11-14:

- **Encoder direction:** convolutional architecture; bytes, then patches, then
  one vector per line; entropy patches after BLT; no cutting; n dimensions.
  Vector sets are qualified for classification by distinct feature ablation
  (information view).
- **No `blaze`.**
- **Zip and parts (1.17):** `flate2` on `zlib-rs` (built at `opt-level = 3` in
  dev); five Parquet codecs; disk estimate 6 bytes per row until a second
  archive is measured; `statvfs` later.
- **1.20:**
  - `ingest` touches the disk only when the set outgrows memory;
  - partition room is judged by distinct keys;
  - `baseline score` checks its parts and a bound for `keys.bin` before the
    read, and deduplication at the freeze;
  - buffers are sized at the freeze.
- **Store (2.7):** both truncation flags plus "empty"; a fill is one
  transaction; the store stays usable after a refused fill.
- **Field machines (requirement 15, design's operating criteria):**
  - batching;
  - spilling datatypes and one global memory limit;
  - streaming where possible;
  - one hungry operation at a time.

  The reference host kills processes at 92 % of memory. Development pipelines
  follow the same rules.

From 2026-11-15:

- **Gate 2.17, option 1:** the baseline scorer classifies logs. The encoder, in
  the stable layout of 2.17 (cut from the 8-bit byte graph, line graph of
  stride-1 convolutions), serves search, grouping and near-duplicate views.
- **The score is agnostic, not dataset-specific.** A score that works because
  it learned certain hosts or endpoints is against the idea. The flow
  (Requirement 16):
  1. the analyst marks n samples (phrase, regex, positions);
  2. they are grouped by string similarity and the analyst de-selects;
  3. hardcoded fragments are redacted, proposed by the tool and confirmed by
     the analyst;
  4. the baseline scorer is fitted on the redacted texts;
  5. Venn decides "close enough", and a log that is not close enough gets no
     ratio.

  CSIC and AIT labels measure agnosticism only (they stand in for an analyst's
  marks); they are not the source of a product score.
- **Requirement 16 details:**
  - the other class is the unmarked logs of the same input by default (good
    marks allowed);
  - the group key is per kind of log and settable (CSIC: URL path; AIT: host);
  - the margin for "depends on seen groups" is 0.05 ROC AUC;
  - without an NPU, grouping is by character n-gram similarity.
- **Requirement 15 wording approved as written.**

### Follow-ups, not yet tasks

- `statvfs` free-disk query on systems other than Windows.
- A measured disk check at the freeze (replaces the 6-byte estimate; would also
  tighten the `keys.bin` bound, about three quarters of the limit under the
  default limit).
- `"incomplete": true` in the JSON of `triage` and `baseline score` on plain
  inputs.
- `ingest` reads its frames back from disk although only their number is
  needed.
- The mid-batch freeze for lack of memory has no test (with readers: in 1.24).
- The hungry-operation gate and an NPU lock in `l2d-core` (requirement 15.6
  to 15.10): now task 1.24.
- The session's placement rule against the provider's duplicated dequantize
  node: stated, not changed.
- The 8-bit cut of 2.17 holds only while the byte graph stays bit-identical on
  the NPU: re-check on every new driver or Ryzen AI release.
- 2.17 record, minor: `hg17_run.py:741` treats the published encoder's
  `seen_linear` as seed-free (0.973, n = 5; over 15 runs 0.9726 ± 0.0089).
- From the 2.7 review: the refusal message wording; `VectorOrigin`'s public
  fields; the message of `Fill::store`; a free-disk check in 2.9; the
  `npu_gate` cache directory under `L2D_NPU_WORK_DIR`.
- Remove the CPU provider comparison from `tests/npu_gate.rs`; repeat the
  penalty comparison on AIT; process CPU seconds as the energy proxy
  (requirement 9.4); AIT labels (a separate amendment later).

The task file's `## Implementation Notes` section holds what each finished task
left for the next ones. Read the notes of the tasks a new task depends on
before starting it.

## How the work is run

Autonomous mode, review required:

1. One heavy task at a time on this machine. Dispatch an implementer agent with
   the task text, the spec sections, the implementation notes and the host
   rules. Tests first.
2. Dispatch an independent reviewer (`.claude/skills/kiro-review/SKILL.md`).
   Give it the code and the implementer's claims to verify, not the conclusion.
   On a rejection, send the findings back, then re-review. At most two
   remediation rounds.
3. Re-run the checks yourself, mark the task `[x]`, bring stale design text in
   line, update the README status table.
4. Commit only that task's files as `feat(npu-log-decision): ...`, then push
   (the owner asked for a push after each approved task). Commit messages end
   with the `Co-Authored-By` line.

A task in a worktree commits on a local branch there. Merge it with
`git merge --no-ff --no-commit <branch>`; `tasks.md` usually conflicts in the
notes section, resolved by keeping both sides. Run the checks in the main
checkout, commit, push, then remove the worktree and delete its branch.

Checks, from the repository root, with a target folder under
`D:\BuildArtifacts\log2decision\`:

```
cargo fmt --all -- --check
cargo test --workspace -j 4
cargo clippy --workspace --all-targets -j 4 -- -D warnings
```

Hardware tests are ignored by default and run on request, in release, one job
on the NPU at a time, with a reduced `PATH` to show that no Python is involved:

```
$env:PATH = 'C:\Windows\System32;C:\Windows;C:\Users\mariu\.cargo\bin'
$env:L2D_RUNTIME_DIR = 'D:\Source\log2decision\runtime\ryzenai'
cargo test --release -j 4 -p l2d-encoder --test npu_rows -- --ignored --nocapture
```

Others: `npu_gate`, `npu_throughput`, `npu_quality`, `npu_session`,
`npu_embedding_export` (2.15), `npu_hierarchical_gate` (2.16, 2.17).

With the owner: answer the question asked first and briefly, say plainly
whether anything needs attention, and bring open decisions as a menu (choices
with a recommendation first).

## Results worth remembering

- **CSIC 2010:** 61,065 records, 25,608 distinct texts, 1,632 path groups, 8 of
  them with both labels.
- **Baseline gate (1.12):** cross-endpoint ROC AUC 0.914 pooled, 0.919 mean
  over five folds by URL path, routing stripped; bar 0.90.
- **Calibration on CSIC:** the 5 % error rate holds on endpoints the
  calibration logs cover, not on others.
- **NPU, rows (2.12):** texts as the rows of one image; every row
  bit-identical to the text encoded alone.
- **NPU, speed:** 16,000 to 42,000 logs/s end to end for the published encoder
  over recorded runs; 2,037 to 2,166 logs/s for the hierarchical encoder of
  2.17 (two graphs); requirement 1,000. Ranges; the laptop's variation is not
  pursued.
- **NPU session (2.6):** NPU only by evidence (placement report plus a profile
  of the first run). A damaged provider cache can run on the CPU while its
  report claims the NPU.
- **Embedding qualification (2.15):** the published encoder's vectors are nearly
  one-dimensional (first component 87.9 %, participation ratio 1.3 of 128).
  The cause is a label-only objective.
- **Hierarchical encoder (2.16):** 12 to 32 components for 95 %; classification
  on unseen endpoints unsettled.
- **Stability gate (2.17)**, `docs/2026-11-15-entropy-encoder-stability-gate.md`:
  - **Randomness ruled out:** bit for bit in and across processes, in float, 8
    bits and on the NPU; training again with the same seed gives identical
    weights.
  - **Cuts stable:** with the 8-bit byte graph's cut as the one cut, cuts are
    identical from 8 bits to the NPU for 25,680 of 25,680 lines, cosine at
    least 0.99992.
  - **Five folds by endpoint × three seeds:** the masked-byte encoder's linear
    head reaches 0.786 seen and 0.722 ± 0.070 unseen. The baseline reaches
    0.961 and 0.923 ± 0.097, and wins every fold.
  - **Why it falls short:** the vectors hold little linearly usable label
    signal; they fall back to 3 to 4 components for 95 % and mostly hold the
    endpoint's template.
  - **Attack pairs on held-out endpoints:** 0.865, with the contrastive control
    better at 0.915.
  - **Centring on the endpoint's mean does not close the gap** (0.69 to 0.71).
- **AIT `russellmitchell.zip` (7.1 GB), 512 MB:** `ingest` of all entries (30.9
  million lines, 8.3 million distinct) on disk in 58 s, peak 374 MiB;
  `baseline score` of all entries in about 8 minutes, peak 359 MiB, 273 MiB of
  parts. Single runs.
- **Store:** CSIC's 25,608 logs added in about 3.5 s into a 46 MiB file;
  nearest-vector search about 36 ms; store format 2 records vector origin and
  runtime release.

## Things that will bite

- **The host kills processes at 92 % of physical memory** (29.6 GiB). Exit code
  -536870911 (0xE0000001) means killed for memory. A died process is never a
  mystery: say which jobs ran together and how much each held. Cures:
  batching, one heavy job at a time. "Commit memory" was never a measured
  cause; do not repeat it.
- **One heavy job at a time:** one training, one measurement, one large build,
  one job on the NPU. Builds use `-j 4`.
- **Spike scripts must stream.** `tools/science/hg17.py` (2.17) still holds the
  vectors (about 13 MB), `PatchBytes` (about 3 MB) and the baseline's feature
  rows whole: bounded, stated needs. The 2.16 arrays of 1.3 to 1.9 GB are gone.
  `tools/science/memguard.py` runs a step under an OS-enforced memory limit and
  records its peak.
- **Shared build folder.** `CARGO_TARGET_DIR` is `D:\BuildArtifacts\cargo-target`
  for every Rust project on this machine. Worktrees, reviewers and scratch
  copies build with their own target under `D:\BuildArtifacts\log2decision\`.
- **Left-over build folders,** safe to remove when nothing runs:
  none at the moment under `D:\BuildArtifacts\log2decision\` (all finished
  tasks' build folders, up to 1.21, were cleaned on 2026-11-16). Keep the `task-*` and `review-*` log folders as evidence.
  The disk check of an all-entries AIT run asks about 88 to 99 GiB free on D:.
- **Debug builds are slow where it matters.** Run evaluations and large archive
  cases in release.
- **Timings on this host vary** (laptop power management; the owner sometimes
  runs it in low-power, silent mode overnight). State ranges; do not
  investigate.
- **Polars.** Import only as `l2d_core::polars`. Do not enable its `csv`,
  `json`, `strings` or `streaming` facade features or upgrade it without
  checking `cargo tree` for network crates. It drops a byte-order mark at the
  start of any buffer it parses (handled since 1.16).
- **Runtime directory.** `runtime\ryzenai\` holds the five files of release
  1.9.0, provisioned from
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` with
  `uv run --no-project --python 3.12 python tools/runtime/provision.py --nupkg <path>`.
  Do not provision or run hardware tests while another process has the DLLs
  loaded. A missing DLL ends the process without an error.
- **NPU provider traps:** `error_on_failure` and the placement report do not
  make a session NPU only; the compiled-model cache trusts its key alone.
- **What reviews reject.** Figures and causes stated but not measured;
  comparisons across different heads or splits (2.17 round 1); code that
  accepts input it cannot handle. Every task that touched memory was rejected
  at least once on a case the implementer had not measured.
- **Host rules.** PowerShell only, no Git Bash, no `wsl`; containers through
  `docker` and `docker compose`. Models in `./models` (never C:), datasets in
  `./data/<dataset>/`, documents in `./docs/` with a date prefix and a README
  row. Caches and temp under `D:\BuildArtifacts\`, never a drive root.
- **Large files.** A hook blocks reading a file of 150 lines or more in one go;
  read in ranges or through `smart_read`.
- **Untracked on purpose.** `.openidea/` is IDE state; `.claude/worktrees/`
  holds agent worktrees. `log2ml/`, `models/`, `reports/`, `runtime/ryzenai/`,
  `runtime/cache/` and the dataset files are ignored.

## Data and artifacts outside git

- **AIT Log Data Set V2:** all eight testbeds, 121.7 GB, in `data/ait-lds-v2/`,
  verified against their MD5 sums, read in place.
- **Science gate:** runs and models in
  `D:\BuildArtifacts\log2decision\science-2.1\runs\`; the published model in
  `models/science-2.1/` (the weakest of three seeds).
- **Encoder gates:** `D:\BuildArtifacts\log2decision\task-2.15\`, `task-2.16\`,
  `task-2.17\` (with the review's figures and the option 2(a) result in
  `task-2.17\review\`); 2.17 model weights in `models/task-2.17/`.
- **Task 1.19 measurements:** `D:\BuildArtifacts\log2decision\task-1.19\`
  (review logs in `review-1.19\`).
- **Baseline gate reports:** `reports/2026-11-13-csic-2010-baseline.{json,md,run.json}`.
- **Evidence of the tasks:** the `task-*` and `review-*` folders under
  `D:\BuildArtifacts\log2decision\`.
- **AMD packages on the owner's machine:**
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` and
  `C:\Users\mariu\Downloads\ryzen_ai_nuget_1.8.0 (1).zip`.

## Other records

- Session memory for agents: `mmhmem` project `log2decision` holds a checkpoint
  of the same state.
- Research documents: `docs/`, indexed in `README.md`. New on 2026-11-15: the
  entropy-patch encoder stability gate (2.17).
- Amendment records in `.kiro/specs/npu-log-decision/`:
  `amendment-2026-11-12-zip-batch.md`,
  `amendment-2026-11-12-vector-decision-heads.md`,
  `amendment-2026-11-15-analyst-flow.md`. All merged; where they differ from the
  spec, the spec wins.
- Spec input and early decisions: `brief.md` and `research.md`. Where they
  disagree with `design.md`, the design wins.
