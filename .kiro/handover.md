# Handover: npu-log-decision

Written 2026-11-14, end of day. Branch `main`, pushed at `15ee402`. Remote:
https://github.com/norandom/log2decision

This file describes the state as it is now. It replaces the handover of
2026-11-13.

## Where things stand

The spec `npu-log-decision` is in implementation
(`.kiro/specs/npu-log-decision/`). Each finished task was implemented by one
agent, checked by an independent reviewer, then committed and pushed.

Tests on `main` at `15ee402`: 711 passing, 0 failing, 47 ignored, in 69 test
binaries. `cargo fmt` and `clippy` are clean.

What works from the command line: a file or a zip archive of logs is read,
scored, calibrated and triaged with the baseline scorer; `baseline score` on an
archive writes Parquet parts with a manifest. The encoder runs on the NPU as a
tested session, and logs can be encoded into the store (2.7, library code); no
command uses the encoder yet.

**One gate is running: 2.17, alone on the machine. Read "Open at the end of the
day" before doing anything else.**

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

Also on 2026-11-14, documentation only: requirement 15 (resource management on
a busy machine, `a49e4d2`), the design section "Operating Criteria: Field
Machines, Not Labs" (`d2dece8`) and the host's 92 % memory guard in it
(`f0769df`).

## Open at the end of the day

### Gate 2.17, running alone

- **What it is:** task 2.17 in `tasks.md`. The question is why a classifier
  fitted on the entropy-patch encoder's vectors does not carry over to
  endpoints it has not seen. In order: (a) rule out randomness, (b) the same
  cuts and vectors in float, 8 bits and on the NPU (pass: median cosine at
  least 0.99, cuts identical for at least 99 % of lines), (c) an objective
  that keeps local detail (masked-patch or next-byte prediction), (d)
  evaluation over five folds grouped by endpoint and at least three seeds,
  beside the published encoder and the baseline scorer. Then NPU throughput
  with nothing else running and a dated document.
- **State:** one implementer agent, in the worktree
  `.claude/worktrees/agent-a0c9e82b9927f0853` (branch
  `worktree-agent-a0c9e82b9927f0853`, locked). Its first commit is `8fa23cc`
  (stability gate scripts). Outputs in `D:\BuildArtifacts\log2decision\task-2.17\`,
  build folder `task-2.17-target`. It has a memory limit of 6 GiB per run and
  records the peak. The owner handed over the whole laptop for it.
- **It runs alone.** No other build, agent, training or measurement while it
  runs (requirement 15; see "Things that will bite").
- **Its instructions:** leave 2.17 unticked, add `- 2.17:` notes, do not amend
  requirements, design or tasks 2.4 to 2.7. If the vectors still do not
  classify on unseen endpoints, stop and bring the measurements to the owner.
- **Next, when it reports:** an independent reviewer checks it (code, figures,
  the memory record of each run). Then merge with `git merge --no-ff
  --no-commit worktree-agent-a0c9e82b9927f0853`, run the checks, and bring the
  conclusion to the owner as a menu with a recommendation. The owner decides
  whether 2.17 passes and what the spec takes from it.
- **If the session was lost:** look at the worktree's commits and at
  `D:\BuildArtifacts\log2decision\task-2.17\` before restarting anything. Do not
  resume a run around a host kill; no figure comes from a resumed run.

### The label and classifier question (state of knowledge)

Not solved, narrowed down. Do not report it as solved or as a myth.

- The label information is in the vectors: with every endpoint seen in
  fitting, boosted trees on the entropy encoder's 8-bit vectors reach ROC AUC
  0.922, a linear model 0.799.
- The failure is transfer to unseen endpoints: 0.57 to 0.59 in 8 bits, about
  0.70 in float. The published encoder reaches 0.94 there, the baseline scorer
  0.989.
- Within one model and one precision the vectors are deterministic (NPU rows
  bit-identical in 1,118 comparisons).
- Measured contributor: quantization moves the entropy model's patch cuts
  (median cosine float to 8 bits 0.24; 8 bits to NPU 0.97). The good/bad
  direction keeps its sign on unseen endpoints for 76 % of its weight in float,
  34 % in 8 bits. Float alone still fails, so this is not the only cause.
- Open: the training objective (it trained the vector to ignore any single
  blanked patch, which is where an attack sits; from the construction, not
  measured), run-to-run randomness (not yet tested across processes or
  repeated training) and evaluation noise (three held-out CSIC endpoints,
  mostly one seed; two seeds differed by up to 0.27).
- Diagnostic scripts used: `diag_classify.py` and `diag_eigen.py` in the
  session scratchpad
  `D:\BuildArtifacts\Temp\claude\f993b7fd18aae7a9\ebba0a50-bc4e-4b97-95b0-e1d9f89c4987\scratchpad\`;
  logs under `D:\BuildArtifacts\log2decision\task-2.16\logs\`. Gate 2.17
  replaces these figures.

## Decisions waiting for the owner

- **Requirement 15 wording** (resource management on a busy machine, 12
  acceptance criteria) is written into `requirements.md` and is with the owner
  for review.
- **Gate 2.17** outcome, when it reports.

## Decisions in force

From 2026-11-11:

- Output is a bad/good ratio per log for triage, not a verdict.
- Rust only. No Python in the tool; Python only during development, in a `uv`
  environment that belongs to this project.
- General text embeddings were dropped: EmbeddingGemma on the NPU managed 2.3
  logs/s and separated attacks worse than character n-gram counts.
- The baseline scorer is character n-grams plus logistic regression.
- Polars is the one data layer: parsing and frames between crates. It is
  pinned to 0.52.0 because later releases link networking code.
- No crate that can open a network connection. A test guards the lock file.
- CSIC is grouped by URL path alone. Groups with a single label are reported
  separately from the cross-endpoint score.
- Memory (requirement 12): the tool reads free memory before loading, sizes
  and rescales its batches, and refuses instead of running out.
- The store is SQLite with its text-search extension.

From 2026-11-12:

- Conformal calibration is own code.
- Triage places logs by label set as specified and states that the error rate
  holds only for logs that resemble the calibration logs.
- A single log text in the store may be at most 1 MiB.
- `faer` stays for the vector search.
- Requirement 13: zip archives are read in batches without extracting them.
  Gzip is the only nested format read. Results go to `<project>/work/<run>/`
  by default. The default maximum record length is 64 KiB.
- Requirement 14: a fitted linear head on the vectors and an outlier ranking,
  after the comparison with Venn (`docs/2026-11-12-venn-vdm-gap-list.md`).
- The runtime files are provisioned from AMD's packages on the owner's
  machine, by the method of the sibling project `npu_rag_from_folder`.
- The project code is under the MIT licence.
- Design text is brought in line after each approved task.

From 2026-11-13:

- **Decide on measured metrics, not estimates.** The AMD release is 1.9.0,
  "always forward"; the 1.8.0 build switch is gone.
- **NPU only.** There is no CPU fallback and no comparison of NPU against CPU
  vectors. Requirement 4.5 is withdrawn; 4.6 asks for the same vector for the
  same text whatever its place in a batch. A host without an NPU uses the
  baseline scorer.
- **Science gate (2.1): passed, with the encoder as the vector source.** The
  baseline stays the first scorer.
- **Baseline gate figure: pooled.** The mean over the folds is reported beside
  it. The baseline's L2 penalty stays at 1 until the comparison is repeated on
  AIT.
- **NPU input stays float.** 8-bit input is a recorded option.
- **Requirement 1.5 has two flags** (the part the encoder reads was cut; the
  whole text is longer than the encoder's input).
- **Zip input (1.16):** own zip-directory reader; lines read before damage are
  kept and the entry reported as damaged; exit code 2 for a run with a
  damaged entry.

From 2026-11-14:

- **Encoder direction.** The convolutional architecture stays. Hierarchy:
  bytes, then patches, then one vector per line. Entropy-based patches after
  Meta's BLT (byte latent encoding). No cutting: cutting loses information.
  n dimensions; the number of principal components for 95 % is not a goal in
  itself, the features must carry meaning. The 2.16 result (12 to 32
  components for 95 % instead of one) is read by the owner as a success of the
  geometry; the classification question goes to 2.17.
- **Qualification is information-based.** A vector set is qualified for
  classification by distinct feature ablation; collapsed vectors lose
  information (Shannon framing; log2ml charts in the research log).
- **No `blaze`** in the project.
- **Zip and parts (1.17):** `flate2` on the `zlib-rs` backend, accepted; the
  five codecs the Parquet feature links, accepted; `zlib-rs` built at
  `opt-level = 3` in the dev profile. The disk estimate stays (6 bytes per
  row) until a second archive is measured. `statvfs` for other systems is a
  later follow-up; until then an archive run stops at the disk check there.
- **1.20:** `ingest` touches the disk only when the set of distinct texts
  outgrows memory; the partition room is judged by distinct keys.
  `baseline score` checks its parts and a bound for `keys.bin` (16 bytes per
  21-byte slot the memory limit holds) before the read, and deduplication at
  the freeze. Write buffers are sized at the freeze (32 KiB down to 4 KiB).
- **Store (2.7):** both truncation flags plus "empty" per log; a fill is one
  transaction; the store stays usable after a refused fill.
- **Field machines (requirement 15 and the design's operating criteria):**
  batching; datatypes with memory spilling and one global memory limit;
  streaming where possible; a resource-aware process that runs one hungry
  operation (a training or similar) at a time. The reference host kills
  processes at 92 % of physical memory. Resource management is a key design
  objective: the users have busy machines. Development pipelines (spikes,
  trainings, measurements) follow the same rules.

## Order of the remaining work

1. **Gate 2.17** (running). Its outcome decides how 2.8 onward use vectors.
2. **1.19** entries read in parallel. Starts only after 2.17 has finished, so
   that 2.17 has the machine alone.
3. **2.8** neighbour scorer, **2.10** linear head, **2.11** outlier ranking,
   then **2.9** command wiring for milestone 2; all after the owner's decision
   on 2.17.
4. Milestones 3 and 4: shift test, assessment, evaluation (3.7 and 3.9 reuse
   `l2d-eval` and the pooled gate figure), end-to-end and hardware tests.

The owner's order: the NPU encoder first, then the decision part after Venn.
The baseline ratio and the vector scorers stay separate scorers.

### Follow-ups, not yet tasks

- `statvfs` free-disk query on systems other than Windows.
- A measured disk check at the freeze (replaces the 6-byte estimate; would
  also tighten the `keys.bin` bound, which under the default limit is about
  three quarters of the limit and can refuse a `baseline score` on a nearly
  full disk).
- `"incomplete": true` in the JSON of `triage` and `baseline score` on plain
  inputs.
- `ingest` reads its frames back from disk although only their number is
  needed.
- The mid-batch freeze for lack of memory has no test.
- The hungry-operation gate and an NPU lock in `l2d-core` (requirement 15.6,
  15.7, 15.10): a project lock plus an NPU lock; trainers read batches.
- The session's placement rule against the provider's duplicated dequantize
  node (seen in 2.16): stated, not changed.
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

1. One heavy task at a time on this machine. Dispatch an implementer agent
   with the task text, the spec sections, the implementation notes and the
   host rules. Tests first.
2. Dispatch an independent reviewer (`.claude/skills/kiro-review/SKILL.md`).
   Give it the code and the implementer's claims to verify, not the
   conclusion. On a rejection, send the findings back, then re-review. At most
   two remediation rounds.
3. Re-run the checks yourself, mark the task `[x]`, bring stale design text in
   line, update the README status table.
4. Commit only that task's files as `feat(npu-log-decision): ...`, then push.
   The owner asked for a push after each approved task. Commit messages end
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
`npu_embedding_export` (2.15). The task notes name each on-request test.

With the owner: answer the question asked first and briefly, say plainly
whether anything needs attention, and bring open decisions as a menu (choices
with a recommendation first).

## Results worth remembering

- **CSIC 2010:** 61,065 records, 25,608 distinct texts, 1,632 path groups, 8 of
  them with both labels.
- **Baseline gate (1.12):** cross-endpoint ROC AUC 0.914 pooled, 0.919 mean
  over five folds by URL path, routing stripped; bar 0.90
  (`docs/2026-11-13-baseline-gate.md`).
- **Calibration on CSIC:** the 5 % error rate holds on endpoints the
  calibration logs cover, not on others.
- **NPU, rows (2.12):** texts as the rows of one image `[1, 64, rows, 256]`;
  every row bit-identical to the text encoded alone.
- **NPU, speed:** 16,000 to 42,000 logs/s end to end over recorded runs on
  this host (requirement 1,000); ready in 3 to 5 s cold, 0.6 to 1 s warm.
  Figures are stated as ranges; the variation belongs to the laptop and is not
  pursued.
- **NPU session (2.6):** NPU only by evidence: the placement report plus a
  profile of the first run, `disable_cpu_ep_fallback`, and an error on a zero
  or non-finite output. A damaged provider cache can run on the CPU while its
  report claims the NPU; the profile catches it.
- **Embedding qualification (2.15):** the published encoder's vectors are
  nearly one-dimensional: the first component carries 87.9 % of the variance,
  participation ratio 1.3 of 128. Cause: a label-only training objective.
  `docs/2026-11-14-embedding-qualification.md`.
- **Hierarchical encoder (2.16):** bytes to patches to one vector per line,
  entropy patches against fixed-stride and delimiter controls; 12 to 32
  components for 95 % of the variance, a flat spectrum (23 eigenvalues above
  1 %). Classification on unseen endpoints unsettled (see above).
  `docs/2026-11-14-hierarchical-encoder-gate.md`, and the research log
  `docs/2026-11-14-research-log-encoder-direction.md` (with log2ml charts and
  the information view, `tools/embedding-qualification/q5_information.py`).
- **AIT archive `russellmitchell.zip` (7.1 GB), limit 512 MB:** `ingest` of the
  272 `*.log` entries (22.9 million lines, 5.6 million distinct) in memory in
  about 40 s, peak 349 MiB; all entries (30.9 million lines, 8.3 million
  distinct) on disk in 58 s, peak 374 MiB (1.20); `baseline score` of all
  entries in about 8 minutes, peak 359 MiB, 273 MiB of parts. Single runs.
- **Store:** CSIC's 25,608 logs added in about 3.5 s into a 46 MiB file;
  nearest-vector search about 36 ms. Store format 2 records the vector origin
  and the AMD runtime release and refuses another release.
- **Runtime sources:** the five files of AMD's 1.9.0 NuGet package; NPU output
  bit-identical to the wheel's builds.

## Things that will bite

- **The host kills processes at 92 % of physical memory** (29.6 GiB). Exit
  code -536870911 (0xE0000001) means killed for memory. A died process is
  never a mystery: say which jobs ran together and how much each held. The
  cures are batching and running one heavy job at a time. Earlier handovers
  blamed "commit memory" for failed builds; that was never measured, do not
  repeat it as a cause.
- **One heavy job at a time.** One training, one measurement, one large build,
  one job on the NPU. Several agents training or measuring in parallel on this
  laptop gave killed processes on 2026-11-14. Builds use `-j 4`.
- **Spike scripts must stream.** `tools/science/hg.py` (2.16) built
  whole-dataset arrays (around lines 335 and 446) and its checkpoint saved the
  numpy random state but not torch's. Gate 2.17 is to fix that; check before
  reusing the 2.16 code.
- **Shared build folder.** `CARGO_TARGET_DIR` is `D:\BuildArtifacts\cargo-target`
  for every Rust project on this machine. Worktrees, reviewers and scratch
  copies build with their own target under `D:\BuildArtifacts\log2decision\`.
- **Left-over build folders,** safe to remove when nothing runs:
  `task-1.20-target`, `review-1.20-target`, `task-1.20-review2-target`,
  `review-2.12-probe-target` under `D:\BuildArtifacts\log2decision\`. Keep
  `task-2.17-target` until 2.17 is merged, and keep the `task-*` and
  `review-*` log folders as evidence.
- **Debug builds are slow where it matters.** Run evaluations and large
  archive cases in release.
- **Timings on this host vary** (laptop power management). State ranges over
  recorded runs; the owner does not want the variation investigated.
- **Polars.** Import it only as `l2d_core::polars`. Do not enable its `csv`,
  `json`, `strings` or `streaming` facade features and do not upgrade it
  without checking `cargo tree` for network crates. It drops a byte-order
  mark at the start of any buffer it parses (handled since 1.16).
- **Runtime directory.** `runtime\ryzenai\` holds the five files of release
  1.9.0, provisioned from
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` with
  `uv run --no-project --python 3.12 python tools/runtime/provision.py --nupkg <path>`.
  Do not provision or run hardware tests while another process has the DLLs
  loaded. A missing DLL ends the process without an error.
- **NPU provider traps:** `error_on_failure` and the placement report do not
  make a session NPU only; the compiled-model cache trusts its key alone.
- **Store limits.** A log text may be at most 1 MiB and may not hold a NUL
  byte.
- **What reviews reject.** Figures and causes stated but not measured, and code
  that accepts input it cannot handle. Every task that touched memory was
  rejected at least once on a case the implementer had not measured.
- **Host rules.** PowerShell only, no Git Bash, no `wsl`; containers through
  `docker` and `docker compose`. Models go in `./models` (never C:), datasets
  in `./data/<dataset>/`, documents in `./docs/` with a date prefix and a row
  in the README. Caches and temp files under `D:\BuildArtifacts\`, never a
  drive root.
- **Large files.** A hook blocks reading a file of 150 lines or more in one
  go; read in ranges or through `smart_read`.
- **Untracked on purpose.** `.openidea/` is IDE state; `.claude/worktrees/`
  holds agent worktrees. `log2ml/`, `models/`, `reports/`, `runtime/ryzenai/`,
  `runtime/cache/` and the dataset files are ignored.

## Data and artifacts outside git

- **AIT Log Data Set V2:** all eight testbeds, 121.7 GB, in
  `data/ait-lds-v2/`, verified against their MD5 sums, read in place.
- **Science gate:** runs and models in
  `D:\BuildArtifacts\log2decision\science-2.1\runs\`. The published model is in
  `models/science-2.1/`; it is the weakest of three seeds.
- **Encoder gates:** 2.15 vectors and logs in
  `D:\BuildArtifacts\log2decision\task-2.15\`; 2.16 vectors, qualification and
  logs in `task-2.16\`; 2.17 in `task-2.17\`.
- **Baseline gate reports:** `reports/2026-11-13-csic-2010-baseline.{json,md,run.json}`.
- **Evidence of the tasks:** the `task-*` and `review-*` folders under
  `D:\BuildArtifacts\log2decision\`.
- **AMD packages on the owner's machine:**
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` and
  `C:\Users\mariu\Downloads\ryzen_ai_nuget_1.8.0 (1).zip`.

## Other records

- Session memory for agents: `mmhmem` project `log2decision` holds a
  checkpoint of the same state.
- Research documents: `docs/`, indexed in `README.md`. New on 2026-11-14: the
  embedding qualification (2.15), the research log on the encoder direction,
  and the hierarchical encoder gate (2.16).
- Amendment records: `.kiro/specs/npu-log-decision/amendment-2026-11-12-zip-batch.md`
  and `amendment-2026-11-12-vector-decision-heads.md`. Both are merged into the
  spec; where they differ from it, the spec wins.
- Spec input and early decisions: `.kiro/specs/npu-log-decision/brief.md` and
  `research.md`. Where they disagree with `design.md`, the design wins.
