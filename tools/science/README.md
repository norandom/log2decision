# Science gate scripts (task 2.1)

Development only. The `l2d` tool is Rust and never runs this code; nothing here is
needed at run time. The scripts train the candidate byte-level encoder on HTTP CSIC
2010, export it to ONNX, quantize it to 8 bits, and compare it with the character
n-gram baseline on held-out endpoints. The result is in
[`docs/2026-11-12-science-gate-encoder.md`](../../docs/2026-11-12-science-gate-encoder.md).

## Environment

Python 3.12 through `uv`; versions are pinned in `pyproject.toml` and `uv.lock`
(torch 2.15.0 CPU, onnx 1.23.2, onnxruntime 1.31.1, scikit-learn 1.9.1, numpy 2.5.4).
The virtual environment is `tools/science/.venv` (git-ignored, disposable). `uv` uses
the cache the shell configures (`UV_CACHE_DIR`). The only network use is the package
download by `uv sync`.

Run everything from `tools\science` in PowerShell:

```powershell
uv sync
$data = 'D:\Source\log2decision\data\csic-2010\csic_database.csv'
$out  = 'D:\BuildArtifacts\log2decision\science-2.1\runs'   # run records and models, outside the repository
```

## Commands

```powershell
# The dataset figures and the folds
uv run python describe_data.py $data

# Baseline on the five grouped folds and on the split of the 1.7 note (about 20 s each)
pwsh -NoProfile -File run_all.ps1 -Data $data -Out $out -Batch baseline

# Encoder, objective bce-norm, method and path stripped: 5 folds x seeds 0, 1, 2
pwsh -NoProfile -File run_all.ps1 -Data $data -Out $out -Batch main
# The reported objective, bce-norm-unit (bce-norm plus the length penalty), same runs
pwsh -NoProfile -File run_all.ps1 -Data $data -Out $out -Batch penalty
# Method and path kept, seed 0: bce-norm, then bce-norm-unit
pwsh -NoProfile -File run_all.ps1 -Data $data -Out $out -Batch keep -Seeds 0
pwsh -NoProfile -File run_all.ps1 -Data $data -Out $out -Batch keep-unit
# The design's objective (bce), seed 0
pwsh -NoProfile -File run_all.ps1 -Data $data -Out $out -Batch objectives
# The split of the 1.7 note: all objectives
pwsh -NoProfile -File run_all.ps1 -Data $data -Out $out -Batch split17

# One run by hand
uv run python run_gate.py --data $data --out $out --split cv5 --fold 0 --seed 0 --routing strip --objective bce-norm-unit

# Tables (Markdown) for one objective
uv run python report.py $out --variant bce-norm-unit

# Describe an ONNX model (interface, operators, quantization parameters)
uv run python inspect_onnx.py D:\BuildArtifacts\log2decision\npu-cnn-spike\enc_b32_int8.onnx

# Copy one run's models into models\science-2.1 with a record and hashes
uv run python publish.py $out D:\Source\log2decision\models\science-2.1 --run split17-f0-s0-strip-bce-norm-unit
```

An encoder run takes 5 to 12 minutes on the CPU with 4 to 8 threads (`-Threads`), depending on what else the
machine does. A run whose record exists in `$out` is skipped by `run_all.ps1`.

## Files

| File | Content |
| --- | --- |
| `l2dsci.py` | Data reading (the text layout and group key of the Rust CSIC adapter), splits, the baseline re-implementation, the encoder, training, ONNX export and quantization, the two scorers on vectors, metrics |
| `run_gate.py` | One run: one split, fold, seed, routing mode and objective; writes one JSON record |
| `run_all.ps1` | The batches of runs |
| `report.py` | Tables from the run records |
| `inspect_onnx.py` | Interface and operators of an ONNX model; used for the comparison with the spike model |
| `describe_data.py` | Dataset figures and folds |
| `publish.py` | Copies a run's models to `models/science-2.1/` and writes `record.json` |

## Leakage rules the code keeps

- Texts are deduplicated before any split.
- Every split is by URL path: all texts of a path are on one side.
- The encoder trains on the fitting side only. The epoch is chosen on a validation
  part of the fitting side, which is itself cut by path (one endpoint with both labels
  and every fifth single-label path).
- The calibration sample of the quantizer is drawn from the training part.
- The neighbour store and the linear head hold fitting-side vectors only; the head's
  penalty is chosen by grouped cross-validation inside the fitting side.
- Held-out labels are read in one function, `metrics`.
