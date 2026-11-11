# Handover: npu-log-decision

Written 2026-11-11, end of day. Branch `main`, last pushed commit `d5d0be7`
(plus the commit that adds this file). Remote: https://github.com/norandom/log2decision

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

- The memory shares (half of free memory as the limit, a quarter as the
  reserve) are fixed in code. Options for them can be added in task 1.11 if
  wanted.
- The second copy of `ndarray` and the `rand` and `lean-agentic` crates that
  `conformal-prediction` brings have not been reviewed.
- The licence of the CSIC 2010 Kaggle upload has not been checked.
- AIT Log Data Set V2 is the planned second dataset and is not downloaded.

## Other records

- Session memory for agents: `mmhmem` project `log2decision` holds a
  checkpoint of the same state.
- Research documents: `docs/`, indexed in `README.md`.
- Spec input and early decisions: `.kiro/specs/npu-log-decision/brief.md` and
  `research.md` (written before the owner amended the design; where they
  disagree with `design.md`, the design wins).
