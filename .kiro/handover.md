# Handover: npu-log-decision

Written 2026-11-11, end of day. Branch `main`, last pushed commit `d5d0be7`
(plus the commit that adds this file). Remote: https://github.com/norandom/log2decision

## State at the end of 2026-11-12 (read this first; the rest is the state of 2026-11-11)

Everything below this section that speaks of adopting, pinning or reviewing
the `conformal-prediction` crate, or of task 1.13 being interrupted, is
history.

### Done today, each reviewed, committed and pushed

| Task | What it is |
| --- | --- |
| 1.13 | Conformal calibration over any scorer |
| 1.18 | Own conformal code; the `conformal-prediction` crate is gone |
| 1.8 | SQLite store with a text index and the vectors |
| 1.9 | Search by phrase, label, source and nearest vectors |
| 1.10 | Triage: ranking, review budget, placement by label set |
| 2.2 | Hardware gate: the 8-bit spike model runs on the NPU from Rust |
| 2.3 | Runtime file check, NPU device query, `L2D_RUNTIME_DIR` |
| 2.13 | Provisioning from AMD's NuGet ZIP (release 1.8.0) |
| 2.14 | Provisioning release 1.9.0 from AMD's NuGet package |

Task 2.1 (science gate) is measured, reviewed and merged as a record; its box
is not ticked, because the gate decision is the owner's (see below).

Tests on `main`: 377 passing, 19 ignored. `cargo fmt` and `clippy` are clean.

### New in the spec today

- Requirement 13: zip archives read in batches under the memory limit, results
  in parts on disk. Tasks 1.16, 1.17, 1.19, after 1.12.
- Requirement 14: a fitted linear head on the vectors and an outlier ranking,
  from the comparison with Venn (`docs/2026-11-12-venn-vdm-gap-list.md`).
  Tasks 2.10 and 2.11, after 2.8.
- Task 2.12: on the NPU a row's output depends on its position in the batch.
  To be settled before the encoder session (2.6).

### Order of the remaining work

- Main lane: 1.11 command wiring (with options for the memory shares and the
  per-log held-out check), 1.12 baseline gate on CSIC, 1.16, 1.17, 1.19.
- Second lane, in a worktree: 2.5 text to bytes, 2.4 encoder package, 2.12,
  then 2.6, 2.7, 2.8, 2.10, 2.11, 2.9.
- Then milestones 3 and 4.

The owner's order: the NPU encoder first, then the decision part after Venn
(neighbour scorer, linear head, outlier ranking, shift test). The baseline
ratio and the vector scorers stay separate scorers; the decision to
investigate, which brings them together, is later work.

### Decisions waiting for the owner

1. **The science gate (2.1).** A linear head on the encoder's 8-bit vectors
   clears the 0.90 bar on grouped folds (0.959). The neighbour ratio at k = 11
   does not (0.893). The review took back the lead over the baseline: about
   level at best, behind on endpoints of a form it has not seen. Read the
   review section of `docs/2026-11-12-science-gate-encoder.md`. Conditions the
   review names before an encoder package is frozen: repeat against 1.12 with
   the baseline's penalty settled; a fold that holds both twin endpoints out;
   the hardware check on the trained model.
2. **What the gate carries with it:** the training objective (the design's
   does not survive 8-bit quantisation), input of the payload only, padding
   with byte 0, PyTorch for the spike where the design names Burn, the
   baseline's default penalty.
3. **AMD release 1.8.0 or 1.9.0.** The tool is built for 1.9.0. The choice
   waits for task 2.12 on a trained model.

### Results worth remembering from today

- Calibration on CSIC: the 5 % error rate is reached on endpoints the
  calibration logs cover and missed for good logs on endpoints they do not
  cover (32 % to 79 % wrongly ruled out, none called good; bad logs missed
  0 %). Triage places logs by label set as specified and states the limit.
- Store: a log text may be at most 1 MiB and may not hold a NUL byte. An add
  holds 16 MiB plus 192 times the longest text. A phrase search holds up to
  about 59 MiB and is refused when that does not fit.
- NPU: 36 of 38 nodes on the NPU, 22,000 to 36,000 logs per second in batches
  of 32, against about 4,000 on the CPU provider. The cosine of 0.9958 against
  the CPU says little, because the spike model's output is nearly constant.
- Batch position: the same row repeated 32 times gives 32 different rows on
  the NPU under 1.9.0 and two under 1.8.0; the CPU gives identical rows. First
  candidate: the pooling step (task 2.12).
- Release 1.8.0 needs six runtime files, 1.9.0 five.

### Things that will bite (new today)

- **Memory.** A build can fail with "only metadata stub found", "required to
  be available in rlib format", "The paging file is too small" or linker error
  LNK1102, and a test run can end with exit code -1 and no failing test. The
  cause is commit memory running out when several compilers run beside other
  heavy work. Build with `-j 4` and rerun once.
- **Worktrees.** A second copy of the workspace must build into its own
  `--target-dir` under `D:\BuildArtifacts\log2decision\`.
- **Runtime directory.** `runtime\ryzenai\` holds the five files of release
  1.9.0, provisioned from
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` with
  `uv run --no-project --python 3.12 python tools/runtime/provision.py --nupkg <path>`.
  Do not provision while a process has the DLLs loaded. Provisioning from
  AMD's package index is outstanding: the index answered 403 all day.
- **Triage and evidence.** Triage keeps only the primary scorer's evidence and
  takes text, yes/no and numeric columns only.
- **Review rounds.** Every task today that touched memory accounting was
  rejected at least once on a case the implementer had not measured. Give
  reviewers the code and the claims, not the conclusion.

### Data

- AIT Log Data Set V2: all eight testbeds, 121.7 GB, in `data/ait-lds-v2/`,
  verified against their MD5 sums. Not unpacked; nothing reads them yet.
- Science gate runs and models: `D:\BuildArtifacts\log2decision\science-2.1\runs\`;
  the published wiring model is in `models/science-2.1/`.

## Where things stand

The spec `npu-log-decision` is in implementation. Requirements, design and
tasks are approved (`.kiro/specs/npu-log-decision/`). Nine tasks of milestone 1
are done, each implemented by one agent, checked by an independent reviewer,
then committed and pushed.

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

Tests: 197 passing, 5 ignored (scale and large-memory variants that run on
request in release mode).

Nothing can be triaged from the command line yet. The commands are stubs until
task 1.11.

## Interrupted work: task 1.13 (conformal calibration)

The implementer for 1.13 was stopped part-way when the day was wrapped up. What
it left in the working tree, uncommitted:

- `Cargo.toml`, `Cargo.lock`, `crates/l2d-decide/Cargo.toml`: the crate
  `conformal-prediction` pinned to `=2.0.1`.
- No `conformal.rs` and no tests yet.

What it had found about the crate (from its comment in `Cargo.toml`, not
reviewed by anyone):

- The crate has no features. It brings `lean-agentic`, `rand` 0.8 and a second
  copy of `ndarray` (0.17, next to the 0.16 that Polars and the regression use).
- It plans to use only the crate's deterministic `ConformalPredictor` p-value
  path, never the parts that draw random numbers.
- The workspace still builds with the change, and the network guard test
  passes.

To resume: either keep the three manifest changes and re-dispatch 1.13 from the
task text, or discard them with `git checkout -- Cargo.toml Cargo.lock crates/l2d-decide/Cargo.toml`
and start clean. The implementer's instructions were: adopt the crate and wrap
it; if the crate cannot be used, stop and report instead of writing the method
by hand; write the tests before the code; report the achieved error rate on
held-out CSIC endpoints as it comes out.

## Order of the remaining work

1. 1.13 conformal calibration (after 1.7, before 1.10)
2. 1.8 SQLite store, 1.9 search
3. 1.10 triage, placing logs by label set
4. 1.11 command wiring, including `l2d calibrate`
5. 1.12 grouped evaluation of the baseline on CSIC: the gate (cross-endpoint
   ROC AUC of at least 0.90 on the 8 path groups that hold both labels)
6. Milestone 2: 2.1 science gate (train the byte-level encoder in a
   project-local `uv` environment), 2.2 hardware gate (run it on the NPU from
   Rust), then the encoder tasks
7. Milestones 3 and 4: assessment, evaluation, end-to-end and hardware tests

The task file's `## Implementation Notes` section holds what each finished task
left for the next ones. Read it before starting any task.

## How the work is run

The project's autonomous mode, review required
(`.claude/skills/kiro-impl/SKILL.md`):

1. One sub-task at a time. Dispatch an implementer agent with the task text,
   the spec sections, the implementation notes and the host rules.
2. Dispatch an independent reviewer (`.claude/skills/kiro-review/SKILL.md`).
   On a rejection, send the findings back to the same implementer, then
   re-review. At most two remediation rounds, then a debug agent.
3. Re-run the checks yourself, mark the task `[x]`, add implementation notes,
   bring stale design text in line, update the README status table.
4. Commit only that task's files (`git add <paths>`, never `git add -A`) as
   `feat(npu-log-decision): ...`, then push. The owner asked for a push after
   each approved task.

Checks, from the repository root:

```
cargo fmt --all -- --check
cargo test --workspace --no-fail-fast -- --nocapture
cargo clippy --workspace --all-targets
cargo run -q -p l2d-cli -- --help
```

On request, in release mode:

```
cargo test --release -p l2d-baseline --test logreg -- --ignored --nocapture
cargo test --release -p l2d-core --test memory_limit -- --ignored --nocapture
cargo test --release -p l2d-baseline --test scorer_csic -- --ignored --nocapture --test-threads=1
```

## Decisions made today

- Output is a bad/good ratio per log for triage, not a verdict.
- Rust only. No Python in the tool; Python only during development, in a `uv`
  environment that belongs to this project.
- General text embeddings were dropped: EmbeddingGemma on the NPU managed 2.3
  logs/s and separated attacks worse than character n-gram counts.
- The baseline scorer is character n-grams plus logistic regression. A small
  encoder trained on log text is the research question, gated behind the
  baseline evaluation.
- Polars is the one data layer: parsing and frames between crates. It is
  pinned to 0.52.0 because later releases link networking code.
- No crate that can open a network connection. A test guards the lock file.
- Conformal prediction layer (requirement 11) with the `conformal-prediction`
  crate; default error rate 5 %; per-group calibration from 100 logs per label.
- CSIC is grouped by URL path alone. Groups with a single label are reported
  separately from the cross-endpoint score.
- Memory (requirement 12): the tool reads free memory before loading, sizes and
  rescales its batches, and refuses instead of running out. The analyst can set
  `--memory-limit` and `--batch-rows`.
- The store is SQLite with its text-search extension (owner's choice in the
  amended design).
- Requirement 2.7 names the `argmin` optimizer; `linfa-logistic` cannot take
  sparse input.
- The shared Rust build folder is left as it is.

## Results worth remembering

- CSIC 2010: 61,065 records, 25,608 distinct texts, 1,632 path groups, 8 of
  them with both labels. Read and deduplicated in about 230 ms at 73 MiB
  (release).
- Memory limit: over 12 input shapes the reviewer measured a peak of at most
  0.78 of the limit, for completed and refused runs.
- Baseline on three held-out CSIC endpoints (one split, not the gate): ROC AUC
  0.989 with the method and path stripped, 0.996 with the whole text.
- Caveats on that number: CSIC reuses a small attack dictionary across
  endpoints, and good logs on unseen endpoints average 0.65 to 0.68, so a fixed
  threshold does not transfer. Per-group calibration is meant to address that.
- A 1M-parameter byte-level convolutional encoder with random weights ran at
  about 36,700 logs/s through AMD's Vitis AI provider (Python spike). Quality
  after training is unknown, and loading the provider from Rust is unproven.

## Things that will bite

- Shared build folder: `CARGO_TARGET_DIR` is `D:\BuildArtifacts\cargo-target`
  for every Rust project on this machine. A scratch copy of this workspace
  built into it leaves test binaries with the wrong paths baked in, which the
  real workspace then reuses. Scratch copies must build with their own
  `--target-dir` under `D:\BuildArtifacts\log2decision\`. The cure is
  `cargo clean -p` for the `l2d-*` crates.
- Debug builds are slow where it matters: a whole-CSIC fit is about 96 s in
  debug and 5 to 15 s in release. Run evaluations in release.
- Polars: import it only as `l2d_core::polars`. Do not enable its `csv`, `json`
  or `strings` facade features and do not upgrade it without checking
  `cargo tree` for network crates.
- Dependencies build without debug info
  (`[profile.dev.package."*"] debug = false`); with it the linker failed
  intermittently.
- Host rules: PowerShell only, no Git Bash, no `wsl`; containers through
  `docker` and `docker compose`. Models go in `./models`, datasets in
  `./data/<dataset>/`, documents in `./docs/` with a date prefix and a row in
  the README.
- `.openidea/` is IDE state and stays untracked. `log2ml/`, `models/` and the
  dataset files are ignored.

## Open points for the owner

Decided by the owner on 2026-11-12:

- Task 1.13 restarted clean (the three manifest changes were discarded) and is
  now done and reviewed; the section "Interrupted work" above is history. Its
  result on CSIC: the error rate holds on seen endpoints and is missed for good
  logs on unseen ones. See the 1.13 lines in the task file's Implementation
  Notes before starting 1.10.
- The memory shares (half of free memory as the limit, a quarter as the
  reserve) get command-line options in task 1.11.
- The CSIC 2010 licence is not pursued. The project code is under the MIT
  licence (`LICENSE`).
- AIT Log Data Set V2 is being downloaded into `data/ait-lds-v2/`, starting
  with `russellmitchell.zip`. The full set is 121.7 GB zipped and is under
  CC BY-NC-SA 4.0; see `data/README.md`.

Still open:

- The second copy of `ndarray` and the `rand` and `lean-agentic` crates that
  `conformal-prediction` brings have not been reviewed.
- Whether to download the other seven AIT testbeds (about 115 GB).

## Other records

- Session memory for agents: `mmhmem` project `log2decision` holds a
  checkpoint of the same state.
- Research documents: `docs/`, indexed in `README.md`.
- Spec input and early decisions: `.kiro/specs/npu-log-decision/brief.md` and
  `research.md` (written before the owner amended the design; where they
  disagree with `design.md`, the design wins).
