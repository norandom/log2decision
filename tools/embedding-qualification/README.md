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

`q5_information.py` is run by itself and is not part of `run_all.ps1`. It gives the
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
