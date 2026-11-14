"""Shared by the embedding qualification scripts (task 2.15): loading the vector files
the Rust export wrote, the training record's split, the plot style and small helpers.

Development only; the l2d tool never runs this code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261114
INPUT_BYTES = 256

# Colours: one fixed categorical order (blue, orange, aqua, yellow, magenta, green,
# violet, red), grey for "other" and for the random baseline.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, BAD = SERIES[0], SERIES[1]
MUTED = "#898781"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
            "figure.dpi": 100,
            "savefig.dpi": 140,
        }
    )


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--vectors", required=True, type=Path, help="directory of the Rust export")
    parser.add_argument("--record", required=True, type=Path, help="record.json of the published model")
    parser.add_argument("--out", required=True, type=Path, help="directory for the PNG files")
    return parser.parse_args()


def encoder_input(text: str) -> bytes:
    """The bytes the encoder reads in `payload` mode: what follows the first `?` or line
    feed, cut at 256 bytes (the rule of `l2d_core::adapters::csic2010::split_routing`)."""
    raw = text.encode("utf-8")
    cuts = [at for at in (raw.find(b"?"), raw.find(b"\n")) if at >= 0]
    return raw[min(cuts) + 1 :][:INPUT_BYTES] if cuts else b""


@dataclass
class Csic:
    vectors: np.ndarray  # all texts, float32, unit length
    rows: list[dict]
    bad: np.ndarray  # bool
    group: np.ndarray  # str
    split: np.ndarray  # "training" | "validation" | "held-out"
    inputs: list[bytes]
    empty: np.ndarray  # bool
    keep: np.ndarray  # indices: first text of every distinct non-empty input
    record: dict


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


def read_vectors(path: Path, rows: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    assert values.size % rows == 0, f"{path}: {values.size} values for {rows} texts"
    return values.reshape(rows, -1)


def load_csic(vectors_dir: Path, record_path: Path) -> Csic:
    rows = read_jsonl(vectors_dir / "csic.jsonl")
    vectors = read_vectors(vectors_dir / "csic.f32", len(rows))
    models = json.loads(record_path.read_text(encoding="utf-8"))["models"]
    assert len(models) == 1, "one published model is expected in the record"
    record = models[0]
    held_out = set(record["split"]["held_out_mixed_paths"])
    group = np.array([row["group"] for row in rows])
    # The validation part as tools/science/l2dsci.py `validation_groups` forms it: one
    # path with both labels (named in the record) and every fifth single-label path of
    # the fitting side in byte order (fold 0).
    labels_of: dict[str, set[str]] = {}
    for row in rows:
        labels_of.setdefault(row["group"], set()).add(row["label"])
    single = sorted((g for g, labels in labels_of.items() if len(labels) == 1 and g not in held_out), key=lambda g: g.encode())
    validation = set(record["settings"]["validation_mixed_paths"]) | set(single[0::5])
    split = np.array(
        ["held-out" if g in held_out else "validation" if g in validation else "training" for g in group]
    )
    inputs = [encoder_input(row["text"]) for row in rows]
    for row, data in zip(rows, inputs):
        assert row["read_bytes"] == len(data), "the input rule here is not the codec's"
    empty = np.array([row["empty"] for row in rows])
    seen: set[bytes] = set()
    keep = []
    for i, data in enumerate(inputs):
        if not empty[i] and data not in seen:
            seen.add(data)
            keep.append(i)
    return Csic(
        vectors=vectors,
        rows=rows,
        bad=np.array([row["label"] == "bad" for row in rows]),
        group=group,
        split=split,
        inputs=inputs,
        empty=empty,
        keep=np.array(keep),
        record=record,
    )


def load_probes(vectors_dir: Path) -> tuple[np.ndarray, list[dict]]:
    rows = read_jsonl(vectors_dir / "probes.jsonl")
    return read_vectors(vectors_dir / "probes.f32", len(rows)), rows


def random_unit(rows: int, dimension: int, seed: int = SEED) -> np.ndarray:
    """The baseline: points drawn uniformly from the unit sphere."""
    values = np.random.default_rng(seed).standard_normal((rows, dimension)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def random_pairs(count: int, pairs: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, count, pairs)
    b = rng.integers(0, count, pairs)
    different = a != b
    return a[different], b[different]


def pair_cosine(vectors: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", vectors[a].astype(np.float64), vectors[b].astype(np.float64))


def short(text: str, width: int = 110) -> str:
    text = text.replace("\n", " \\n ").replace("http://localhost:8080", "")
    return text if len(text) <= width else text[: width - 3] + "..."
