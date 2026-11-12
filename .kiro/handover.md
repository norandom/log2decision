# Handover: npu-log-decision

Written 2026-11-12, end of day. Branch `main`, pushed. Remote:
https://github.com/norandom/log2decision

This file describes the state as it is now. It replaces the handover of
2026-11-11; nothing from that day is still pending.

## Where things stand

The spec `npu-log-decision` is in implementation
(`.kiro/specs/npu-log-decision/`). Requirements, design and tasks are approved
and were extended on 2026-11-12 (requirements 13 and 14). Each finished task
was implemented by one agent, checked by an independent reviewer, then
committed and pushed.

Tests on `main`: 377 passing, 19 ignored. `cargo fmt` and `clippy` are clean.

Nothing can be triaged from the command line yet. The library parts exist; the
commands are stubs until task 1.11.

No agent, download or worktree is running.

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

### Measured, not closed

Task 2.1, the science gate, is measured, reviewed and merged as a record
(`4656e66`: `tools/science/`, `docs/2026-11-12-science-gate-encoder.md`). Its
box is not ticked, because the gate decision is the owner's. See "Decisions
waiting for the owner".

## Decisions waiting for the owner

1. **The science gate (2.1).** Proceed with the encoder, or refine the model
   first.
   - A linear head on the encoder's 8-bit vectors reaches a cross-endpoint ROC
     AUC of 0.959 on five folds grouped by URL path and three seeds. That
     clears the 0.90 bar with every training variant and every seed.
   - The nearest-neighbour ratio at the design's k = 11 reaches 0.893 and
     misses it.
   - The review took back the lead over the baseline. Two CSIC endpoints share
     one form; where the encoder won, one twin was held out and the other
     fitted. With both held out: head 0.639, neighbour ratio 0.504, baseline
     0.762. With a lighter penalty the baseline averages 0.955, level with the
     head.
   - So the encoder is about level with the baseline at best, and behind on
     endpoints of a form it has not seen.
   - Conditions the review names before an encoder package is frozen: repeat
     against task 1.12 with the baseline's penalty settled; add a fold that
     holds both twin endpoints out, three seeds; run the hardware check on the
     trained model.
   - Tasks 2.5 and 2.4 do not depend on the decision.
2. **What the gate carries with it.**
   - The training objective. The design's does not survive 8-bit
     quantisation; the one that worked reads the normalised vector and keeps
     its length near 1.
   - Input of the payload only. It is specific to the CSIC layout and has to
     go into the encoder identity.
   - Padding with byte 0.
   - PyTorch for the spike, where the design names Burn.
   - The baseline's default penalty.
3. **AMD release 1.8.0 or 1.9.0.** The tool is built for 1.9.0. The choice
   waits for task 2.12 on a trained model.

## Order of the remaining work

Two lanes, one implementer and one reviewer at a time in each.

- **Main lane, in the main checkout:**
  1. 1.11 command wiring, with options for the memory shares and the per-log
     held-out check
  2. 1.12 baseline gate on CSIC in release: cross-endpoint ROC AUC of at least
     0.90 on the 8 path groups that hold both labels; calibration coverage is
     reported, not a pass condition
  3. 1.16 zip archives read in batches, 1.17 results in parts on disk, 1.19
     entries read in parallel
- **Second lane, in a worktree:**
  1. 2.5 text to bytes, 2.4 encoder package
  2. 2.12 the batch-position effect on the NPU (needs a trained model)
  3. 2.6 encoder session, 2.7 encode and store
  4. 2.8 neighbour scorer, 2.10 linear head, 2.11 outlier ranking
  5. 2.9 command wiring for milestone 2
- **Then** milestones 3 and 4: shift test, assessment, evaluation, end-to-end
  and hardware tests.

The owner's order: the NPU encoder first, then the decision part after Venn
(neighbour scorer, linear head, outlier ranking, shift test). The baseline
ratio and the vector scorers stay separate scorers. The decision to
investigate, which brings them together, is later work with its own spec or
amendment.

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
   then re-review. At most two remediation rounds. Advisory findings after an
   approval go back for a short pass.
3. Re-run the checks yourself, mark the task `[x]`, bring stale design text in
   line and list what changed, update the README status table.
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

The hardware gate, on request, with a reduced `PATH` to show that no Python is
involved:

```
$env:PATH = 'C:\Windows\System32;C:\Windows;C:\Users\mariu\.cargo\bin'
$env:L2D_RUNTIME_DIR = 'D:\Source\log2decision\runtime\ryzenai'
cargo test -j 4 -p l2d-encoder --test npu_gate -- --ignored --nocapture
```

Other on-request tests run in release with `-- --ignored --nocapture
--test-threads=1`; the task notes name each one.

## Decisions in force

From 2026-11-11:

- Output is a bad/good ratio per log for triage, not a verdict.
- Rust only. No Python in the tool; Python only during development, in a `uv`
  environment that belongs to this project.
- General text embeddings were dropped: EmbeddingGemma on the NPU managed 2.3
  logs/s and separated attacks worse than character n-gram counts.
- The baseline scorer is character n-grams plus logistic regression. A small
  encoder trained on log text is the research question.
- Polars is the one data layer: parsing and frames between crates. It is
  pinned to 0.52.0 because later releases link networking code.
- No crate that can open a network connection. A test guards the lock file.
- CSIC is grouped by URL path alone. Groups with a single label are reported
  separately from the cross-endpoint score.
- Memory (requirement 12): the tool reads free memory before loading, sizes
  and rescales its batches, and refuses instead of running out.
- The store is SQLite with its text-search extension.

From 2026-11-12:

- Conformal calibration is own code; the `conformal-prediction` crate was
  adopted in 1.13 and removed in 1.18.
- Triage places logs by label set as specified, with no special treatment of
  endpoints the calibration logs do not cover, and states that the error rate
  holds only for logs that resemble the calibration logs.
- A single log text in the store may be at most 1 MiB.
- `faer` stays for the vector search, although the store uses one function of
  it.
- Requirement 13: zip archives are read in batches without extracting them.
  Gzip is the only nested format read. Results go to
  `<project>/work/<run>/` by default. The default maximum record length is
  64 KiB.
- Requirement 14: a fitted linear head on the vectors and an outlier ranking,
  after the comparison with Venn (`docs/2026-11-12-venn-vdm-gap-list.md`).
  Venn's trained residual heads, heads compiled from an outcome store, numeric
  outcomes and its generic text head are not planned.
- The held-out check for calibration becomes per log in task 1.11: the scorer
  saves a hash of each text it was fitted on.
- The memory shares get command-line options in task 1.11.
- The runtime files are provisioned from AMD's packages on the owner's
  machine, by the method of the sibling project `npu_rag_from_folder`.
- The project code is under the MIT licence.
- Design text is brought in line after each approved task, with the changes
  listed in the report.

## Results worth remembering

- **CSIC 2010:** 61,065 records, 25,608 distinct texts, 1,632 path groups, 8 of
  them with both labels.
- **Baseline on three held-out endpoints** (one split, not the gate): ROC AUC
  0.989 with the method and path stripped, 0.996 with the whole text. CSIC
  reuses a small attack dictionary across endpoints.
- **Calibration on CSIC:** the 5 % error rate is reached on endpoints the
  calibration logs cover. On endpoints they do not cover, 32 % to 79 % of good
  logs are wrongly ruled out and none is called good; bad logs are missed 0 %.
- **Store:** CSIC's 25,608 logs are added in about 3.5 s into a 46 MiB file. A
  nearest-vector search takes about 36 ms, the first 100 phrase matches about
  1.5 ms.
- **Triage** of CSIC's 25,608 logs takes under a second.
- **NPU:** 36 of 38 nodes on the NPU, 22,000 to 36,000 logs per second in
  batches of 32, against about 4,000 on the CPU provider. No Python on `PATH`.
- **The gate's cosine says little.** The spike model has random weights and
  its output is nearly the same for every input, so 0.9958 against the CPU
  does not show that every row is right.
- **Batch position:** the same row repeated 32 times gives 32 different rows
  on the NPU under 1.9.0 and two under 1.8.0. The CPU gives identical rows.
  Row 0 is right. The first candidate is the pooling step (task 2.12).
- **Runtime sources:** the five files of AMD's 1.9.0 NuGet package are the
  wheel's builds signed separately, and the NPU output is bit-identical
  between them. Release 1.8.0 needs six files.
- **Science gate:** see "Decisions waiting for the owner".

## Things that will bite

- **Memory.** A build can fail with "only metadata stub found", "required to
  be available in rlib format", "The paging file is too small" or linker error
  LNK1102, and a test run can end with exit code -1 and no failing test. The
  cause is commit memory running out when several compilers run beside other
  heavy work. Build with `-j 4` and rerun once.
- **Shared build folder.** `CARGO_TARGET_DIR` is `D:\BuildArtifacts\cargo-target`
  for every Rust project on this machine. A second copy of this workspace
  built into it leaves test binaries with the wrong paths baked in. Worktrees
  and scratch copies must build with their own `--target-dir` under
  `D:\BuildArtifacts\log2decision\`. The cure is `cargo clean -p` for the
  `l2d-*` crates.
- **Debug builds are slow where it matters.** A whole-CSIC fit is about 96 s
  in debug and 5 to 15 s in release. Run evaluations in release.
- **Polars.** Import it only as `l2d_core::polars`. Do not enable its `csv`,
  `json`, `strings` or `streaming` facade features and do not upgrade it
  without checking `cargo tree` for network crates.
- **Runtime directory.** `runtime\ryzenai\` holds the five files of release
  1.9.0, provisioned from
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` with
  `uv run --no-project --python 3.12 python tools/runtime/provision.py --nupkg <path>`.
  Do not provision or run the hardware gate while another process has the
  DLLs loaded. A missing DLL ends the process without an error, which is why
  the file check runs before anything is loaded. Provisioning from AMD's
  package index is outstanding: the index answered 403 all day.
- **Release switch.** The release is a cargo feature of `l2d-encoder`
  (`ryzenai-1-9` by default, `ryzenai-1-8` for trials). Exactly one must be
  on, so `--all-features` does not compile.
- **Store limits.** A log text may be at most 1 MiB and may not hold a NUL
  byte. An add holds 16 MiB plus 192 times the longest text. A phrase search
  holds up to about 59 MiB and is refused when that does not fit.
- **Triage and evidence.** Triage keeps only the primary scorer's evidence and
  takes text, yes/no and numeric columns only. The three list views are not
  counted against the memory budget.
- **Memory accounting in reviews.** Every task that touched it was rejected at
  least once on a case the implementer had not measured: long texts, a
  repetitive text, text evidence at small batch sizes.
- **Host rules.** PowerShell only, no Git Bash, no `wsl`; containers through
  `docker` and `docker compose`. Models go in `./models`, datasets in
  `./data/<dataset>/`, documents in `./docs/` with a date prefix and a row in
  the README.
- **Large files.** A hook blocks reading a file of 150 lines or more in one
  go; read in ranges or through `smart_read`.
- **Untracked on purpose.** `.openidea/` is IDE state. `log2ml/`, `models/`,
  `runtime/ryzenai/`, `runtime/cache/` and the dataset files are ignored.

## Data and artifacts outside git

- **AIT Log Data Set V2:** all eight testbeds, 121.7 GB, in
  `data/ait-lds-v2/`, verified against their MD5 sums. Not unpacked; nothing
  reads them yet. The first archive holds mostly files that are not logs,
  packet captures of about 1 GB, and 108 nested archives.
- **Science gate:** all runs and models in
  `D:\BuildArtifacts\log2decision\science-2.1\runs\`. The published model for
  wiring is in `models/science-2.1/`; it is the weakest of three seeds and not
  a model trained on all data.
- **AMD packages on the owner's machine:**
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` and
  `C:\Users\mariu\Downloads\ryzen_ai_nuget_1.8.0 (1).zip`.

## Other records

- Session memory for agents: `mmhmem` project `log2decision` holds a
  checkpoint of the same state.
- Research documents: `docs/`, indexed in `README.md`. New on 2026-11-12: the
  Venn comparison and the science gate.
- Amendment records: `.kiro/specs/npu-log-decision/amendment-2026-11-12-zip-batch.md`
  and `amendment-2026-11-12-vector-decision-heads.md`. Both are merged into the
  spec; where they differ from it, the spec wins.
- Spec input and early decisions: `.kiro/specs/npu-log-decision/brief.md` and
  `research.md`. Where they disagree with `design.md`, the design wins.
