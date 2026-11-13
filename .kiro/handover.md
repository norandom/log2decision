# Handover: npu-log-decision

Written 2026-11-13, end of day. Branch `main`, pushed at `51aa056`. Remote:
https://github.com/norandom/log2decision

This file describes the state as it is now. It replaces the handover of
2026-11-12.

## Where things stand

The spec `npu-log-decision` is in implementation
(`.kiro/specs/npu-log-decision/`). Each finished task was implemented by one
agent, checked by an independent reviewer, then committed and pushed.

Tests on `main` at `51aa056`: 563 passing, 0 failing, 28 ignored. `cargo fmt`
and `clippy` are clean.

A file of logs can be read, scored, calibrated and triaged from the command
line with the simple scorer. The encoder runs on the NPU as tested library
code; no command uses it yet.

**Two tasks are open in the middle of their review. Read "Open at the end of
the day" before doing anything else.**

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
| 2.1 | Science gate: decided by the owner on 2026-11-13, see below | `51aa056` |

## Open at the end of the day

No agent is running. Two pieces of work are unfinished.

### Task 1.16, zip input: uncommitted in the main checkout

- **State:** implemented, reviewed once, rejected on one finding, and the
  implementer finished the fix after this handover was first written. The
  working tree of the main checkout holds the work uncommitted and complete.
  The fix has **not** been re-reviewed. The implementer reports 608 passed,
  0 failed, 30 ignored, fmt and clippy clean; run the checks again before
  trusting that.
- **The fix as reported:** `Accepted::push_raw` in `input.rs` puts a second
  mark in front when a batch buffer starts with one, for Polars to drop. The
  lines and CSV adapters had the same fault and are fixed by it; JSONL was not
  affected. A mark at the very start of a file or entry is dropped once;
  anywhere else it is text. All advisory items were done, including a test
  over 1,500 mutated archives (none panicked, heap growth under 1 MiB) and a
  read failure of the archive file now giving exit code 5. `data/README.md`
  was corrected.
- **Files:** new `crates/l2d-core/src/{zip_input.rs,zip_directory.rs}`,
  `crates/l2d-core/tests/{zip_input.rs,zip_memory.rs,byte_order_mark.rs}`,
  `crates/l2d-cli/tests/archive.rs`; changed `Cargo.toml`, `Cargo.lock`,
  `crates/l2d-core/{Cargo.toml,src/lib.rs,src/input.rs,src/memory.rs,src/adapters/lines.rs}`,
  `crates/l2d-cli/{Cargo.toml,src/cmd/ingest.rs,src/cmd/common.rs,src/output.rs}`,
  `README.md` (one row), `tasks.md` (`1.16:` notes).
- **The blocking finding:** a log line that starts with a byte-order mark
  (U+FEFF) and is the first record of a batch aborts the whole read, and the
  result depends on the batch size. Polars drops a mark at the start of the
  buffer it parses. The cause is in the shared path in `input.rs`, so the
  lines adapter on a plain file fails the same way. The fix belongs in the
  shared place, with tests for both adapters at batch sizes 1 and default and
  for a double mark at an entry start.
- **Advisory items sent with it:** wording of the 128-bit key comment; README
  figures that have no log; the entry count in `data/README.md` (14,911
  directory entries, 2,696 directories, 12,215 files, 11,992 of them
  non-empty); the parser's known differentials in the notes; a real I/O error
  mid-entry reported as damage; JSON `position` above 2^53; `grow_for`
  reserving one step early; design lines 1126 and 1128 in the stale list; a
  property test over mutated archives.
- **Next:** run the checks, have an independent reviewer confirm the fix,
  write the owner's three zip decisions into the notes and design.md, bring
  design.md in line (stale lines 53, 449, 457, 464-478, 1126, 1128), tick,
  commit, push.
- **What held up in the review:** the own zip reader survived every hostile
  archive the reviewer built; the memory accounting is exact; the AIT figures
  reproduce. Measured on `russellmitchell.zip` (7.1 GB) with `*.log` under
  512 MB: 272 entries, 22,880,272 records, 5,642,082 distinct, heap peak
  334.4 MiB, about 27 s to read plus 5 to 9 s to open. Reading every entry
  under 512 MB is refused at about 7.3 million distinct texts; that is the
  limit task 1.17 removes.

### Task 2.6, encoder session: committed on a branch, rejected once

- **State:** implemented and committed as `6b929bb` on local branch `task-2.6`
  in the worktree `.claude/worktrees/agent-a46f05b739bed075d` (not pushed).
  Reviewed once and rejected on two small findings. The findings have **not**
  been sent back yet.
- **Blocking 1:** `ExecutedNodes::npu_only`
  (`crates/l2d-encoder/src/session.rs` about 375-393) is looser than the
  notes say. Any operator through the Vitis provider other than the quantize
  pair counts as a compiled part, and the two ends are accepted on any other
  provider. Fix: through the Vitis provider accept only the two ends and
  nodes named `vitis_ai_ep_*`, with at least one of the latter; elsewhere
  accept only `CPUExecutionProvider` with at most one `QuantizeLinear` and one
  `DequantizeLinear`; refuse everything else. Add default tests for `Conv`
  through Vitis and for an unknown provider. Correct the note that says a new
  node name would be refused.
- **Blocking 2:** the `2.6:` throughput note explains the gap to 2.12's
  16,636 logs/s by host load. That is not supported: 16,636 was 2.12's
  quieter run, and process CPU per million logs fell from 33 s to 11 to 15 s.
  State the range (medians 27,044 to 42,217 logs/s over ten runs, the
  reviewer's included) and note the owner's reading of 2026-11-13: the host
  is a laptop with power management, the difference is related to the system
  itself, it was not measured and is not pursued.
- **Advisory:** stale lines missing from the list (design.md 15, 295, 296;
  tasks.md 267 and 419); the comment in `tests/npu_session.rs` that no CPU
  provider runs anywhere, while `tests/npu_gate.rs` still has one from task
  2.2 (removing it is a follow-up); try `session.disable_cpu_ep_fallback=1`;
  no check for a zero or non-finite model output; device and driver are not
  in the cache key; `plain()` breaks a path on a network share; the dropped
  `deviceStat` cross-check.
- **Next:** a new implementer in that worktree gets these findings; the
  reviewer re-checks; then merge with `git merge --no-ff --no-commit
  task-2.6`, checks, design.md in line, tick, README, push, remove the
  worktree and the target directory
  `D:\BuildArtifacts\log2decision\task-2.6-target`.
- **What held up, on the committed code, run by the reviewer on the NPU:**
  every vector is identical whatever row a text is in (0 of 2,708 differ);
  cold start 3.0 s, warm 0.6 s; a damaged cache folder is refused; 587 tests
  pass. A finding beyond the brief: with a damaged `context.json` in the
  provider's cache the provider opens without error and runs on the CPU while
  its report still says 33 of 35 nodes on the NPU. The session therefore
  profiles its first run and refuses on that evidence.

## Decisions waiting for the owner

None. Everything put to the owner on 2026-11-13 was answered; see the next
section.

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
  vectors. Requirement 4.5 is withdrawn; 4.6 now asks for the same vector for
  the same text whatever its place in a batch. The aim on the NPU is
  end-to-end throughput. A host without an NPU uses the baseline scorer.
- **Science gate (2.1): passed, with the encoder as the vector source.** The
  encoder does not clearly beat the baseline, so the baseline stays the first
  scorer. The encoder goes ahead because search, the linear head and the
  Venn-style decision part need vectors.
- **Baseline gate figure: pooled,** as the design says: all held-out
  predictions as one ranking. The mean over the folds is reported beside it.
- **The baseline's L2 penalty stays at 1.** The comparison of penalties is
  repeated on the AIT data before anything is changed.
- **NPU input stays float.** 8-bit input gave bit-identical vectors and 27 %
  to 48 % more throughput; it is a recorded option that needs the revision's
  domain string to change.
- **Requirement 1.5 has two flags:** the part the encoder reads was cut, and
  the whole log text is longer than the encoder's input. Built in task 2.6.
- **The store records the AMD runtime release** beside the encoder identity
  and refuses vectors of another release. Planned into task 2.7.
- **Zip input (1.16):** the own zip-directory reader stays, and the `zip`
  crate is used in tests only. Lines read from an entry before its damage is
  found are kept and the entry is reported as damaged. A run with a damaged
  entry exits with code 2.
- Memory shares: one typed share lowers the other automatically; both typed
  above 1 in sum is exit code 2.

## Order of the remaining work

Two lanes, one implementer and one reviewer at a time in each.

- **Main lane, in the main checkout:**
  1. finish 1.16 (see above)
  2. 1.17 results in parts on disk, 1.19 entries read in parallel
- **Second lane, in a worktree:**
  1. finish 2.6 (see above)
  2. 2.7 encode and store, with the runtime release in the store
  3. 2.8 neighbour scorer, 2.10 linear head, 2.11 outlier ranking
  4. 2.9 command wiring for milestone 2
- **Then** milestones 3 and 4: shift test, assessment, evaluation (3.7 and 3.9
  reuse `l2d-eval` and use the pooled gate figure), end-to-end and hardware
  tests.
- **Follow-ups noted, not yet tasks:** remove the CPU provider comparison from
  `tests/npu_gate.rs`; repeat the penalty comparison on AIT; process CPU
  seconds as the energy proxy (requirement 9.4); AIT labels (a separate
  amendment later).

The owner's order: the NPU encoder first, then the decision part after Venn
(neighbour scorer, linear head, outlier ranking, shift test). The baseline
ratio and the vector scorers stay separate scorers.

The task file's `## Implementation Notes` section holds what each finished task
left for the next ones. Read the notes of the tasks a new task depends on
before starting it.

## How the work is run

Autonomous mode, review required:

1. One task at a time per lane. Dispatch an implementer agent with the task
   text, the spec sections, the implementation notes and the host rules. Tests
   first; it does not commit in the main checkout.
2. Dispatch an independent reviewer (`.claude/skills/kiro-review/SKILL.md`).
   Give it the code and the implementer's claims to verify, not the
   conclusion. On a rejection, send the findings back to the same implementer,
   then re-review. At most two remediation rounds.
3. Re-run the checks yourself, mark the task `[x]`, bring stale design text in
   line, update the README status table.
4. Commit only that task's files (`git add <paths>`, never `git add -A`) as
   `feat(npu-log-decision): ...`, then push. The owner asked for a push after
   each approved task.

A task in a worktree commits on a local branch there. Merge it with
`git merge --no-ff --no-commit <branch>`, run the checks in the main checkout,
commit, push, then remove the worktree with `git worktree remove -f -f` and
delete its branches.

Checks, from the repository root:

```
cargo fmt --all -- --check
cargo test --workspace -j 4
cargo clippy --workspace --all-targets -j 4 -- -D warnings
```

Hardware tests are ignored by default and run on request, in release, one
agent on the NPU at a time, with a reduced `PATH` to show that no Python is
involved:

```
$env:PATH = 'C:\Windows\System32;C:\Windows;C:\Users\mariu\.cargo\bin'
$env:L2D_RUNTIME_DIR = 'D:\Source\log2decision\runtime\ryzenai'
cargo test --release -j 4 -p l2d-encoder --test npu_rows -- --ignored --nocapture
```

The other hardware tests are `npu_gate`, `npu_throughput`, `npu_quality` and,
on the 2.6 branch, `npu_session`. The task notes name each on-request test.

With the owner: answer the question asked first and briefly, say plainly
whether anything needs attention, and bring open decisions as choices with a
recommendation.

## Results worth remembering

- **CSIC 2010:** 61,065 records, 25,608 distinct texts, 1,632 path groups, 8 of
  them with both labels.
- **Baseline gate (1.12),** five folds by URL path, routing stripped, default
  penalty: cross-endpoint ROC AUC 0.914 pooled, 0.919 as the mean over the
  folds; bar 0.90. With the whole text kept: 0.897 pooled, below the bar. The
  two endpoints that share one form (`registro.jsp`, `editar.jsp`) score
  0.82, and 0.76 when both are held out; the others 0.97 to 0.99. Lighter
  penalties score 0.953 to 0.956, mostly through transfer between those two
  endpoints. The report is byte-reproducible
  (`docs/2026-11-13-baseline-gate.md`).
- **Calibration on CSIC:** the 5 % error rate is reached on endpoints the
  calibration logs cover and missed for both labels on endpoints they do not.
- **NPU, rows (2.12):** in the exported batch layout `[N, 64, 1, 256]` the NPU
  gets every row but the first wrong under Ryzen AI 1.9.0, for any N above 1.
  The pooling step is not the cause; what the provider computes there was not
  established. With the texts as the rows of one image `[1, 64, rows, 256]`
  (`onnx_edit::with_rows_as_height`) every row is bit-identical to the text
  encoded alone.
- **NPU, speed:** from log text to unit vector, one session at 64 rows:
  16,000 to 17,000 logs/s in 2.12's runs and 27,000 to 42,000 in 2.6's runs
  on the same host. The owner's reading: the host is a laptop with power
  management, so the difference belongs to the system, not to the code. It
  was not measured and is left as it is; figures are stated as ranges.
  Four sessions: 35,000 to
  40,000 in 2.12's runs. The requirement is 1,000. Ready in 3 to 5 s cold and
  0.6 to 1 s warm; the requirement allows 30 s.
- **NPU, quality:** the vectors off the NPU reproduce the science gate:
  linear head 0.958 against 0.959 recorded, neighbour ratio at k = 11 0.899
  against 0.893.
- **NPU, traps:** registering the provider with `error_on_failure` does not
  make a session NPU only; neither does the provider's placement report. The
  provider's compiled-model cache trusts its key alone: a reused key gave a
  failing run for another row count and silently wrong vectors for another
  model of the same shape.
- **Store:** CSIC's 25,608 logs are added in about 3.5 s into a 46 MiB file. A
  nearest-vector search takes about 36 ms, the first 100 phrase matches about
  1.5 ms.
- **Commands on CSIC:** fit in about 2 s, triage of 8,644 logs in under half a
  second.
- **Runtime sources:** the five files of AMD's 1.9.0 NuGet package are the
  wheel's builds signed separately, and the NPU output is bit-identical
  between them.

## Things that will bite

- **Memory.** A build can fail with "only metadata stub found", "required to
  be available in rlib format", "The paging file is too small" or linker error
  LNK1102, and a test run can end with exit code -1 and no failing test. The
  cause is commit memory running out when several compilers run beside other
  heavy work. Build with `-j 4` and rerun once.
- **Shared build folder.** `CARGO_TARGET_DIR` is `D:\BuildArtifacts\cargo-target`
  for every Rust project on this machine. Worktrees, reviewers and scratch
  copies must build with their own `--target-dir` under
  `D:\BuildArtifacts\log2decision\`. The cure for wrong paths baked into test
  binaries is `cargo clean -p` for the `l2d-*` crates.
- **Left-over build folders** from today, safe to remove when nothing runs:
  `task-2.12-target`, `review-2.12-target`, `review-1.12-target`,
  `review-1.16-target`, `review-2.6-target` under
  `D:\BuildArtifacts\log2decision\`. Keep `task-2.6-target` until 2.6 is
  merged, and keep the `task-*` and `review-*` log folders as evidence.
- **Debug builds are slow where it matters.** Run evaluations in release. The
  full baseline evaluation is about 30 minutes.
- **Timings on this host vary.** Fit times differed by 30 % to 50 % and NPU
  throughput by a factor of two between runs. The host is a laptop with
  power management; the owner attributes the variation to the system itself
  and does not want it pursued. State ranges over recorded runs, and mark a
  cause that was not measured as a reading, not a finding.
- **Polars.** Import it only as `l2d_core::polars`. Do not enable its `csv`,
  `json`, `strings` or `streaming` facade features and do not upgrade it
  without checking `cargo tree` for network crates. It drops a byte-order
  mark at the start of any buffer it parses (see task 1.16).
- **Runtime directory.** `runtime\ryzenai\` holds the five files of release
  1.9.0, provisioned from
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` with
  `uv run --no-project --python 3.12 python tools/runtime/provision.py --nupkg <path>`.
  Do not provision or run hardware tests while another process has the DLLs
  loaded. A missing DLL ends the process without an error, which is why the
  file check runs before anything is loaded. Provisioning from AMD's package
  index is outstanding and not needed.
- **Store limits.** A log text may be at most 1 MiB and may not hold a NUL
  byte. An add holds 16 MiB plus 192 times the longest text. A phrase search
  holds up to about 59 MiB and is refused when that does not fit.
- **Triage and evidence.** Triage keeps only the primary scorer's evidence and
  takes text, yes/no and numeric columns only.
- **What reviews reject.** Figures that were stated and not measured, causes
  that were stated and not measured, and code that accepts input it cannot
  handle. Every task that touched memory was rejected at least once on a case
  the implementer had not measured.
- **Host rules.** PowerShell only, no Git Bash, no `wsl`; containers through
  `docker` and `docker compose`. Models go in `./models`, datasets in
  `./data/<dataset>/`, documents in `./docs/` with a date prefix and a row in
  the README.
- **Large files.** A hook blocks reading a file of 150 lines or more in one
  go; read in ranges or through `smart_read`.
- **Untracked on purpose.** `.openidea/` is IDE state. `log2ml/`, `models/`,
  `reports/`, `runtime/ryzenai/`, `runtime/cache/` and the dataset files are
  ignored.

## Data and artifacts outside git

- **AIT Log Data Set V2:** all eight testbeds, 121.7 GB, in
  `data/ait-lds-v2/`, verified against their MD5 sums. Not unpacked. Task
  1.16 reads `russellmitchell.zip` in place.
- **Science gate:** all runs and models in
  `D:\BuildArtifacts\log2decision\science-2.1\runs\`. The published model for
  wiring is in `models/science-2.1/`; it is the weakest of three seeds and not
  a model trained on all data.
- **Baseline gate reports:** `reports/2026-11-13-csic-2010-baseline.{json,md,run.json}`.
- **Evidence logs of today's tasks:** `D:\BuildArtifacts\log2decision\task-2.12\logs\`,
  `task-2.6\logs\`, `task-1.16\`, and the reviewers' `review-*` folders.
- **AMD packages on the owner's machine:**
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` and
  `C:\Users\mariu\Downloads\ryzen_ai_nuget_1.8.0 (1).zip`.

## Other records

- Session memory for agents: `mmhmem` project `log2decision` holds a
  checkpoint of the same state.
- Research documents: `docs/`, indexed in `README.md`. New on 2026-11-13: the
  baseline gate and the NPU throughput and row correctness document.
- Amendment records: `.kiro/specs/npu-log-decision/amendment-2026-11-12-zip-batch.md`
  and `amendment-2026-11-12-vector-decision-heads.md`. Both are merged into the
  spec; where they differ from it, the spec wins.
- Spec input and early decisions: `.kiro/specs/npu-log-decision/brief.md` and
  `research.md`. Where they disagree with `design.md`, the design wins.
