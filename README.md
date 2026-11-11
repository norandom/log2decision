# log2decision

Local, low-energy triage of text logs: embeddings on the laptop NPU, a vector
and text store, and small decision heads that give each log a bad/good ratio.
Successor to the [log2ml](https://github.com/norandom/log2ml) master thesis.
Implementation language: Rust.

Status: research and specification. No application code yet.

## Documents

Every document in `docs/` is listed here, newest first. File names start with
the date.

| Date | Document | Status |
| --- | --- | --- |
| 2026-11-11 | [Embedding speed and separability on CSIC 2010](docs/2026-11-11-embedding-speed-benchmark.md) | Measured; NPU embedding too slow, general embeddings trail a character baseline |
| 2026-11-11 | [Feasibility: triaging text logs with NPU embeddings and small decision heads](docs/2026-11-11-npu-log-decision-feasibility.md) | Research, first measurements |
| 2026-11-11 | [Embedding model choice](docs/2026-11-11-embedding-model-choice.md) | Open; NPU default rejected on speed |

## Specifications

| Spec | Phase |
| --- | --- |
| [npu-log-decision](.kiro/specs/npu-log-decision/requirements.md) | Requirements and [design](.kiro/specs/npu-log-decision/design.md) approved; [tasks](.kiro/specs/npu-log-decision/tasks.md) generated, awaiting approval |

## Layout

| Path | Content |
| --- | --- |
| `docs/` | Dated research and decision documents, indexed above |
| `data/<dataset>/` | One directory per dataset, registered in [data/README.md](data/README.md); the data itself is not tracked |
| `models/` | Model weights (`FLM_MODEL_PATH`); never on C:; not tracked |
| `.kiro/specs/` | Specifications |
| `log2ml/` | Local reference copy of the thesis repository; not tracked |

## Project rules

- Rust is the implementation language. Python is not developed here; `uv` may
  run one-off analysis scripts.
- On this host, agents use PowerShell. No Git Bash and no `wsl` commands.
  Containers run through `docker` and `docker compose`.
- New documents go to `docs/` with a date prefix and get a row in the table
  above.
- Datasets go to `data/<dataset>/` and get a row in `data/README.md`.
- Models go to `models/`.
