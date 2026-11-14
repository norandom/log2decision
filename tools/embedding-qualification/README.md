# Embedding qualification scripts (task 2.15)

Development only. The `l2d` tool is Rust and never runs this code. The scripts read the
vectors the NPU gave for CSIC's distinct texts and for a set of probe texts, and write
the pictures and figures of
[`docs/2026-11-14-embedding-qualification.md`](../../docs/2026-11-14-embedding-qualification.md).

## Step 1: the vectors, from the NPU (Rust)

From the repository root, in a shell whose `PATH` has no Python interpreter:

```powershell
$env:PATH = 'C:\Windows\System32;C:\Windows;C:\Users\<you>\.cargo\bin'
$env:L2D_RUNTIME_DIR = 'D:\Source\log2decision\runtime\ryzenai'
$env:L2D_SCIENCE_DIR = 'D:\Source\log2decision\models\science-2.1'
$env:L2D_CSIC_CSV    = 'D:\Source\log2decision\data\csic-2010\csic_database.csv'
$env:L2D_NPU_WORK_DIR = 'D:\BuildArtifacts\log2decision\task-2.15'
$env:L2D_EXPORT_DIR   = 'D:\BuildArtifacts\log2decision\task-2.15\vectors'
cargo test --release -p l2d-encoder --test npu_embedding_export -- --ignored --nocapture --test-threads=1
```

It writes `csic.f32`, `csic.jsonl`, `probes.f32`, `probes.jsonl` and `export.json` to
`L2D_EXPORT_DIR` (the layout is in the header of
`crates/l2d-encoder/tests/npu_embedding_export.rs`, and so are the probe texts).

## Step 2: the pictures and figures (Python through `uv`)

Python 3.12; numpy, scikit-learn and scipy pinned as in `tools/science`, plus matplotlib
and umap-learn (`uv.lock`). The virtual environment is `.venv` here (git-ignored).
In a fresh shell, from this directory:

```powershell
uv sync
pwsh -NoProfile -File run_all.ps1 `
    -Vectors D:\BuildArtifacts\log2decision\task-2.15\vectors `
    -Record  D:\Source\log2decision\models\science-2.1\record.json `
    -Logs    D:\BuildArtifacts\log2decision\task-2.15\logs
```

| Script | Question | Log | Pictures |
|---|---|---|---|
| `q1_collapse.py` | Do different strings give different vectors? | `q1.log` | `q1-*.png` |
| `q2_similarity.py` | Do similar strings land near each other? | `q2.log` | `q2-*.png` |
| `q3_structure.py` | Do labels and endpoints show structure? | `q3.log` | `q3-*.png` |
| `q4_dimensions.py` | How many dimensions carry the variance? | `q4.log` | `q4-*.png` |

The pictures go to `docs/images/2026-11-14-embedding-qualification/`. Every script uses
the seed in `eq.py`; `q3_structure.py` takes about four minutes (UMAP twice, k-means).

## Another encoder (task 2.16)

The scripts qualify any encoder whose vectors are in the layout of step 1. Three options
of every script (and of `run_all.ps1`) say how that encoder reads a text and where the
vectors come from; without them a run is the published model's, as before:

| Option | `run_all.ps1` | Meaning |
|---|---|---|
| `--input payload` or `whole` | `-InputMode` | `payload`: what follows the first `?` or line feed (the published model). `whole`: the whole line |
| `--input-bytes N` | `-InputBytes` | the bytes the encoder reads at most (256; 1024 for the encoders of task 2.16) |
| `--source "..."` | `-Source` | the words after "vectors" in legends and log lines (`from the NPU`) |

With `whole`, the analysis set stays the published encoder's (the first text of every
distinct non-empty payload, 12,078 texts), so the figures compare like with like; the
folding counts of `q1_collapse.py` are over all 25,608 texts, whose whole lines are all
distinct. `-Information` adds `q5_information.py` to the run.

```powershell
$v = 'D:\BuildArtifacts\log2decision\task-2.16\vectors\entropy-s0-int8'
$q = 'D:\BuildArtifacts\log2decision\task-2.16\qualification\entropy-s0-int8'
pwsh -NoProfile -File run_all.ps1 -Vectors $v -Record "$v\record.json" -Logs $q -Out $q `
    -InputMode whole -InputBytes 1024 -Source 'in 8 bits (Python reference)' -Information
# The comparison table and the pictures of docs/2026-11-14-hierarchical-encoder-gate.md
uv run python hg_report.py --root D:\BuildArtifacts\log2decision\task-2.16 `
    --out ..\..\docs\images\2026-11-14-hierarchical-encoder-gate
```

## The information view

`q5_information.py` is run by itself and is not part of `run_all.ps1` unless
`-Information` is given. It gives the
information view of the
[research log](../../docs/2026-11-14-research-log-encoder-direction.md): the variance
spectrum beside log2ml's encoders, its Shannon entropy, and the feature ablation.

```powershell
uv run python q5_information.py `
    --vectors D:\BuildArtifacts\log2decision\task-2.15\vectors `
    --record  D:\Source\log2decision\models\science-2.1\record.json `
    --out     ..\..\docs\images\2026-11-14-research-log 2>&1 |
    Out-File D:\BuildArtifacts\log2decision\task-2.15\logs\q5.log -Encoding utf8
```
