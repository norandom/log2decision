# log2decision

![Before: an operator buried in paper and alarms at a control desk. After: the same operator at ease in front of a short checked list.](docs/images/before-after.png)

`l2d` reads text logs on your own machine and tells you which ones to look at
first. Each log gets a score between 0 and 1 for how likely it is bad, learned
from logs you have already labelled. Logs that look good are scored down so
your time goes to the rest.

It is built for a security analyst with a laptop and too many logs. Nothing is
sent anywhere, and the tool is designed to stay out of the way of whatever else
the machine is doing.

Status, 2026-11-12: under construction. The reading, scoring, calibration,
store, search and triage parts exist as tested library code. The commands that tie them together are not wired
yet, so you cannot triage a file from the command line today. See
[what works today](#what-works-today).

## What you will get

For a file of logs, `l2d triage` returns three lists:

| List | Meaning |
| --- | --- |
| Review | Logs ranked from most to least likely bad, up to the number you say you have time for |
| Scored down | Logs the tool considers good, with the error rate that claim carries |
| Uncertain | Logs it will not call either way, or that look unlike anything it was calibrated on |

Each decision shows what it rests on: which scorer produced it and, once the
encoder exists, the nearest labelled logs.

Three things it will not do. It does not block, delete or change logs; it ranks
and you decide. It does not send log data over a network; no network code is
even linked into the binary. And it does not take the machine down: before it
loads a file it reads the free memory, sizes its batches to fit, and refuses
with a clear message when a job would not fit.

## How it decides

The first scorer is deliberately simple. It counts short character sequences in
each log (think `' OR`, `%27`, `../`) and learns which ones show up in bad logs.
That is the technique behind classic spam filters, and on web attack data it
beat every general text-embedding model we tried, by a wide margin. The
numbers are in the [benchmark](docs/2026-11-11-embedding-speed-benchmark.md).

On top of the score sits a calibration step (conformal prediction). It turns
"0.8" into a statement you can act on: at a 5 % error rate, this log is good,
bad, or too close to call. Without it you have a ranking; with it you have a
bounded risk when you skip a log.

A small encoder trained on log text, running on the laptop's NPU, is the next
stage. It has to beat the simple scorer on endpoints it has never seen before
it is allowed in.

## What works today

| Part | State |
| --- | --- |
| Reading logs: one per line, labelled CSV or JSONL, HTTP CSIC 2010 | Done, tested |
| Deduplication, with skipped rows reported by position and reason | Done, tested |
| Batches sized to free memory, with a memory limit that holds | Done, measured on files up to 564 MB |
| Character-sequence features and the scoring model | Done, tested |
| Baseline scorer (fit, score, save, load) | Done, tested |
| Calibrated decisions (label sets at a chosen error rate) | Library done and tested, not yet wired to a command. The error rate holds on endpoints the calibration logs cover (about 5 % measured on CSIC); on endpoints it has never seen, 32 % to 79 % of good logs were wrongly ruled out and none were called good |
| Store: one SQLite file per store with a text index and the vectors | Library done and tested, not yet wired to a command. CSIC's 25,608 distinct logs are added in about 3.5 s into a 46 MiB file. A single log text may be at most 1 MiB |
| Search in a store: by phrase (with or without case), by label and source, and by nearest vectors | Library done and tested, not yet wired to a command. On CSIC's 25,608 logs a nearest-vector search takes about 36 ms and the first 100 phrase matches about 1.5 ms. A search that does not fit the memory limit is refused |
| Triage: ranking, review budget, and with a calibration the split into review, scored down, uncertain and unlike | Library done and tested, not yet wired to a command. Triage of CSIC's 25,608 logs takes under a second |
| Evaluation | Specified, not built |
| Commands (`ingest`, `baseline`, `triage`, ...) | Present as stubs that say "not implemented yet" |
| Encoder on the NPU | Hardware gate passed: an 8-bit model with random weights runs on the AMD NPU from Rust, 36 of 38 nodes on the NPU, at 22,000 to 36,000 logs per second. Open: on the NPU a row's output depends on its position in the batch, which the test model could not reveal; this is settled before the encoder is built. The runtime files are checked against a manifest before anything is loaded. Whether a trained encoder beats the simple scorer is being measured; the encoder itself is not built |

A measured example: HTTP CSIC 2010 (61,065 requests, 28 MB) is read and
deduplicated to 25,608 distinct logs in about a quarter of a second, with a
peak of 73 MiB of memory. In small batches the peak drops to about 35 MiB.

## Try it

You need Windows 11 and Rust 1.98 (the repository pins the toolchain). Linux
should work for everything except the NPU, but has not been tested.

```
cargo build --release
cargo test --workspace
cargo run --release -p l2d-cli -- --help
```

The help lists every command and the two options that apply to all of them:

| Option | Effect |
| --- | --- |
| `--memory-limit 2GB` | Cap the memory the tool may use. Default: half of what is free |
| `--batch-rows 10000` | Cap the batch size. Default: 65,536 rows, fewer when memory is short |

To see the reader and the memory limit at work on your own file:

```
cargo run --release -p l2d-core --example read_batched -- lines path\to\your.log --memory-limit 512MB
```

For the CSIC dataset, download it as described in [data/README.md](data/README.md)
and replace `lines` with `csic`.

## Documents

Every document in `docs/` is listed here, newest first. File names start with
the date. These are records of what was known on that day; where later work
overturned something, the document says so at the top.

| Date | Document | What it settles |
| --- | --- | --- |
| 2026-11-12 | [Science gate: does a trained byte-level encoder earn its place on the NPU?](docs/2026-11-12-science-gate-encoder.md) | A linear head on the 8-bit vectors reaches cross-endpoint ROC AUC 0.959 on five grouped folds and clears the 0.90 bar; the neighbour ratio (0.893) and the design's training objective do not. The review found the lead over the simple scorer does not hold: about level at best, behind on endpoints of a form it has not seen |
| 2026-11-11 | [Embedding speed and separability on CSIC 2010](docs/2026-11-11-embedding-speed-benchmark.md) | General embeddings are too slow on the NPU and worse than character features; a small encoder of our own runs fast there |
| 2026-11-11 | [Embedding model choice](docs/2026-11-11-embedding-model-choice.md) | Which embedding models were considered and why the default was dropped |
| 2026-11-11 | [Feasibility: triaging text logs with NPU embeddings and small decision heads](docs/2026-11-11-npu-log-decision-feasibility.md) | The starting plan and what became of each part |
| 2026-11-12 | [Venn's operations against the log2decision spec](docs/2026-11-12-venn-vdm-gap-list.md) | Which Vector Decision Model operations the spec covers, which are missing, and a proposal for after the NPU encoder |

## For contributors

The work follows a written specification:
[requirements](.kiro/specs/npu-log-decision/requirements.md),
[design](.kiro/specs/npu-log-decision/design.md) and
[tasks](.kiro/specs/npu-log-decision/tasks.md). The task file also holds the
notes each finished task left for the next one.

| Path | Content |
| --- | --- |
| `crates/` | The Rust workspace: `l2d-core` (input, memory), `l2d-baseline` (features, scorer), `l2d-cli`, and crates reserved for the store, decisions, assessment, evaluation and encoder |
| `tests/fixtures/` | A small labelled fixture used by the tests |
| `docs/` | Dated research and decision documents, indexed above |
| `data/<dataset>/` | One directory per dataset, registered in [data/README.md](data/README.md); the data is not tracked |
| `models/` | Fitted scorers and model weights; not tracked |
| `.kiro/specs/` | Specification |
| `log2ml/` | Local reference copy of the [thesis repository](https://github.com/norandom/log2ml) this project grew out of; not tracked |

Rules of the house:

- Rust only. Python is not part of the tool; `uv` may run one-off analysis
  scripts during development.
- Polars is the one data layer. Tabular data moves between crates as Polars
  frames.
- No crate that can open a network connection. A test fails if one appears in
  the lock file.
- New documents go to `docs/` with a date prefix and get a row in the table
  above. Datasets go to `data/<dataset>/` and get a row in `data/README.md`.
- On the development host, agents use PowerShell: no Git Bash, no `wsl`.
