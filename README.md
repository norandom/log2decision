# log2decision

![Before: an operator buried in paper and alarms at a control desk. After: the same operator at ease in front of a short checked list.](docs/images/before-after.png)

`l2d` reads text logs on your own machine and tells you which ones to look at
first. Each log gets a score between 0 and 1 for how likely it is bad, learned
from logs you have already labelled. Logs that look good are scored down so
your time goes to the rest.

It is built for a security analyst with a laptop and too many logs. Nothing is
sent anywhere, and the tool is designed to stay out of the way of whatever else
the machine is doing.

Status, 2026-11-13: under construction. You can read, score, calibrate and
triage a file of logs from the command line with the simple scorer. The store,
the search and the encoder on the NPU exist as tested library code and are not
wired to commands yet. See
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
| Zip archives as input: entries read without unpacking, in batches, within the memory limit | `l2d ingest` reads an archive and reports every entry and its records; `l2d baseline score` scores one into Parquet parts on disk, one row per distinct log; the other commands do not read archives yet. Measured on a 7.1 GB archive of the AIT data set under a limit of 512 MB: `ingest` reads its 272 `*.log` entries (22.9 million lines, 5.6 million distinct) in 32 s with a peak of 343 MiB, and refuses all of its entries together with the memory they would need; `baseline score` scores all of its entries (30.9 million lines, 8.3 million distinct) in about 8 minutes with a peak of 359 MiB, deduplicating on disk, and writes 273 MiB of parts |
| Character-sequence features and the scoring model | Done, tested |
| Baseline scorer (fit, score, save, load) | Done, tested |
| Calibrated decisions (label sets at a chosen error rate) | Library done and tested, not yet wired to a command. The error rate holds on endpoints the calibration logs cover (about 5 % measured on CSIC); on endpoints it has never seen, 32 % to 79 % of good logs were wrongly ruled out and none were called good |
| Store: one SQLite file per store with a text index and the vectors | Library done and tested, not yet wired to a command. CSIC's 25,608 distinct logs are added in about 3.5 s into a 46 MiB file. A single log text may be at most 1 MiB |
| Search in a store: by phrase (with or without case), by label and source, and by nearest vectors | Library done and tested, not yet wired to a command. On CSIC's 25,608 logs a nearest-vector search takes about 36 ms and the first 100 phrase matches about 1.5 ms. A search that does not fit the memory limit is refused |
| Triage: ranking, review budget, and with a calibration the split into review, scored down, uncertain and unlike | Library done and tested, not yet wired to a command. Triage of CSIC's 25,608 logs takes under a second |
| Evaluation | The simple scorer's evaluation is built and run as library code: on CSIC, with whole endpoints held out, it ranks attacks at ROC AUC 0.914 (bar 0.90). It is weaker on two endpoints that share one form (0.82, and 0.76 when both are unseen). The `eval` command and the encoder's side are not built |
| Commands `ingest`, `baseline fit`, `baseline score`, `calibrate`, `triage` | Work from the command line, with text and `--json` output. On CSIC: fit in about 2 s, triage of 8,644 logs in under half a second, each within the memory limit it reports |
| Other commands (`store`, `encoder`, `shift`, `assess`, `eval`, `runtime`) | Present as stubs that say "not implemented yet" |
| Encoder on the NPU | A trained 8-bit model runs on the AMD NPU from Rust, measured as library and test code: from log text to vector at 16,000 to 42,000 logs per second with one session over the recorded runs (the laptop's power management moves the figure from run to run) and 35,000 to 40,000 with four, ready in 3 to 5 s the first time and about 1 s after. In the usual batch layout the NPU gets every text but the first of a batch wrong; the tool lays the texts out as rows of one image, and then each vector is bit-identical to the text encoded alone. The vectors that come off the NPU give the triage quality measured in training (0.958 against 0.959). The runtime files are checked against a manifest before anything is loaded. The encoder session is built: it runs on the NPU only and refuses, with the reason, unless both the provider's placement report and the first run show the model on the NPU; there is no CPU fallback. Each vector comes with two flags that say whether the text was cut. Not built yet: the commands that use the session, storing the vectors, and training in Rust. The encoder is kept as the source of vectors for search and the decision part; the simple scorer stays the first scorer |

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

The help lists every command and the options that apply to all of them:

| Option | Effect |
| --- | --- |
| `--project DIR` | The project directory, where models are kept under `models/`. Default: the current directory |
| `--json` | Machine-readable output |
| `--memory-limit 2GB` | Cap the memory the tool may use. Default: half of what is free |
| `--memory-share 0.5`, `--reserve-share 0.25` | The same as shares of the free memory: how much to use, and how much to keep free |
| `--batch-rows 10000` | Cap the batch size. Default: 65,536 rows, fewer when memory is short |

A first run on a labelled file, here a CSV with the columns `text`, `label`
and `group`, cut into three parts so that nothing is judged on logs the scorer
has already seen:

```
l2d ingest --input all.csv --group-column group
l2d baseline fit --input fit.csv --group-column group --name demo
l2d calibrate --input cal.csv --group-column group --baseline demo --name demo-cal
l2d triage --input new.csv --group-column group --baseline demo --calibration demo-cal --budget 50
```

`triage` prints the logs to review, the logs it scored down and the logs it
will not call either way. `calibrate` leaves out any log whose text the scorer
was fitted on and says how many. For the CSIC dataset, download it as
described in [data/README.md](data/README.md) and use `--adapter csic2010`;
`l2d ingest --help` lists the adapters and their options. `l2d` is the binary
that `cargo build --release` puts in the build folder; `cargo run --release -p
l2d-cli --` in front of the arguments does the same.

Exit codes: 0 success, 2 a mistake in the options, 3 something missing (the
message says where it was expected), 4 a refusal such as too little memory,
5 an input or output failure.

## Documents

Every document in `docs/` is listed here, newest first. File names start with
the date. These are records of what was known on that day; where later work
overturned something, the document says so at the top.

| Date | Document | What it settles |
| --- | --- | --- |
| 2026-11-14 | [Hierarchical encoder gate: do patches and a content objective give vectors that keep the line?](docs/2026-11-14-hierarchical-encoder-gate.md) | No variant passes. Encoders trained without the label on bytes, patches and one vector per line spread over 17 to 31 principal components (published encoder: 2), keep unrelated lines near cosine 0 and edits near 1, and tell all 25,608 CSIC lines apart in float. The two-graph layout runs on the NPU only, every row right, at 1,200 to 1,500 logs per second with entropy patches and about 3,000 with fixed stride. But entropy patches do not beat fixed stride and delimiter patches, and the vectors rank attacks on held-out endpoints at 0.58 to 0.76 where the published encoder's head reaches 0.94 and the baseline scorer 0.99. Two of the research log's six "we are wrong if" conditions hold; the owner decides |
| 2026-11-14 | [Research log: why the encoder's vectors are one line, and what replaces it](docs/2026-11-14-research-log-encoder-direction.md) | The day's findings and decisions in one place: the vectors are a good/bad score because the encoder was trained on the label alone; the owner's direction for its replacement (convolutional, hierarchical, entropy-based patches, no cutting, one vector per line); the objections, and what would prove the direction wrong. The scorers on vectors wait for the gate of task 2.16 |
| 2026-11-14 | [Embedding qualification: what is in the vectors the NPU encoder gives?](docs/2026-11-14-embedding-qualification.md) | Not noise and not the one-point collapse of the spike model, but close to one-dimensional: the first principal component holds 88 % of the variance, two hold 95 %, and the vectors are a good/bad score on one curve. Small edits stay near (cosine 0.995 or more), attacks move away from benign requests, endpoints group although the path is not read. On three endpoints the encoder was not fitted on, the linear head ranks at ROC AUC 0.939 (the first component alone 0.936), while every good text has a bad majority among its 11 nearest stored texts (0.79, and 0.55 with repeated inputs). Suitable with limits for the head, not shown suitable for neighbours across endpoints, outlier ranking not checked |
| 2026-11-13 | [Baseline gate: does the simple scorer rank attacks on endpoints it has not seen?](docs/2026-11-13-baseline-gate.md) | The simple scorer passes its gate on five folds by URL path (cross-endpoint ROC AUC 0.914 with all held-out logs as one ranking, 0.919 as the mean over the folds; bar 0.90), but only three of five endpoints are strong: the two that share one form score 0.82, and 0.76 when both are unseen. The Rust tool confirms the science gate's baseline figures. Two honest rules for choosing the L2 penalty disagree; the default stays and the comparison is repeated on a second dataset |
| 2026-11-13 | [The trained encoder on the NPU: are all rows right, how fast is it, how good are the vectors](docs/2026-11-13-npu-throughput-and-row-correctness.md) | In the exported batch layout only the first text of a batch is computed right on the NPU; with the texts as rows of one image every row is bit-identical to the text alone. End to end 16,600 logs per second with one session (37,600 with four), ready in 4.9 s cold and 1.0 s warm. The NPU's vectors give the quality the science gate recorded: linear head 0.958, neighbour ratio 0.899 |
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
