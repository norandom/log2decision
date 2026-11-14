"""Question 1: do different strings give different vectors? (No collapse.)

Counts distinct vectors, compares the cosine of random pairs with the same for random
unit vectors and with the collapse reference (0.9998, the random-weight spike model),
and shows the spread per dimension.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import eq

COLLAPSE = 0.9998


def describe(name: str, values: np.ndarray) -> None:
    q = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
    print(
        f"{name}: pairs {values.size}, mean {values.mean():.4f}, p01 {q[0]:.4f}, p25 {q[1]:.4f}, "
        f"median {q[2]:.4f}, p75 {q[3]:.4f}, p99 {q[4]:.4f}, min {values.min():.4f}, max {values.max():.4f}, "
        f"share above 0.99 {np.mean(values > 0.99):.5f}, share above {COLLAPSE} {np.mean(values > COLLAPSE):.5f}"
    )


def main() -> None:
    args = eq.arguments(__doc__)
    eq.style()
    csic = eq.load_csic(args.vectors, args.record)
    vectors = csic.vectors
    rows, dimension = vectors.shape
    print(f"texts {rows}, dimension {dimension}")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    print(f"vector length: min {norms.min():.7f}, max {norms.max():.7f} (the session normalises; the raw length is not in the file)")
    print(f"values not finite: {int((~np.isfinite(vectors)).sum())}")

    # Distinct vectors against distinct inputs.
    as_bytes = [row.tobytes() for row in vectors]
    distinct_vectors = len(set(as_bytes))
    distinct_inputs = len(set(csic.inputs))
    empty = int(csic.empty.sum())
    print(f"distinct texts {rows}; distinct encoder inputs {distinct_inputs} (of them 1 is the empty input, shared by {empty} texts)")
    print(f"distinct vectors (bit for bit) {distinct_vectors}")
    by_input: dict[bytes, set[bytes]] = {}
    by_vector: dict[bytes, set[bytes]] = {}
    for data, vector in zip(csic.inputs, as_bytes):
        by_input.setdefault(data, set()).add(vector)
        by_vector.setdefault(vector, set()).add(data)
    print(f"inputs with more than one vector: {sum(len(v) > 1 for v in by_input.values())}")
    shared = [inputs for inputs in by_vector.values() if len(inputs) > 1]
    print(f"vectors shared by different inputs: {len(shared)} (inputs involved: {sum(len(s) for s in shared)})")
    for inputs in shared[:5]:
        print("   shared vector, inputs: " + " | ".join(repr(i[:80]) for i in sorted(inputs)[:3]))
    sizes = Counter(csic.inputs)
    print(f"texts per input: largest groups {sorted(sizes.values(), reverse=True)[:5]}")
    print(
        f"analysis set (first text of every distinct non-empty input): {csic.keep.size} texts, "
        f"good {int((~csic.bad[csic.keep]).sum())}, bad {int(csic.bad[csic.keep].sum())}"
    )

    # Cosine of random pairs.
    kept = vectors[csic.keep]
    a, b = eq.random_pairs(kept.shape[0], 200_000)
    cosine = eq.pair_cosine(kept, a, b)
    baseline = eq.random_unit(kept.shape[0], dimension)
    cosine_random = eq.pair_cosine(baseline, a, b)
    describe("cosine, random pairs of CSIC vectors", cosine)
    describe("cosine, random pairs of random unit vectors", cosine_random)
    bad = csic.bad[csic.keep]
    for name, mask in [
        ("good-good", ~bad[a] & ~bad[b]),
        ("bad-bad", bad[a] & bad[b]),
        ("good-bad", bad[a] != bad[b]),
    ]:
        describe(f"cosine, random pairs, {name}", cosine[mask])
    mean_vector = kept.astype(np.float64).mean(axis=0)
    print(
        f"length of the mean vector: CSIC {np.linalg.norm(mean_vector):.4f}, random unit vectors "
        f"{np.linalg.norm(baseline.astype(np.float64).mean(axis=0)):.4f} (1 would be one point, 0 no common direction)"
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    grid = np.linspace(-1, 1, 801)
    bins = np.linspace(-1, 1, 161)
    ax.hist(cosine, bins=bins, density=True, color=eq.SERIES[0], alpha=0.35, label="CSIC vectors from the NPU, histogram")
    ax.plot(grid, gaussian_kde(cosine[:50_000])(grid), color=eq.SERIES[0], lw=2, label="CSIC vectors, density estimate")
    ax.plot(grid, gaussian_kde(cosine_random[:50_000])(grid), color=eq.MUTED, lw=2, label="random unit vectors, 128 dimensions")
    ax.axvline(COLLAPSE, color=eq.SERIES[7], lw=2, ls="--", label=f"collapse reference (spike model): {COLLAPSE}")
    ax.set_xlabel("cosine similarity of a pair")
    ax.set_ylabel("density")
    ax.set_xlim(-1, 1.02)
    ax.set_title(f"Cosine similarity of {cosine.size:,} random pairs of distinct inputs")
    ax.legend(loc="upper left")
    eq.save(fig, args.out, "q1-pairwise-cosine.png")

    # Spread per dimension.
    std = kept.astype(np.float64).std(axis=0)
    std_random = baseline.astype(np.float64).std(axis=0)
    mean = kept.astype(np.float64).mean(axis=0)
    print(
        f"standard deviation per dimension: min {std.min():.4f}, median {np.median(std):.4f}, max {std.max():.4f}; "
        f"random unit vectors: min {std_random.min():.4f}, median {np.median(std_random):.4f}, max {std_random.max():.4f}"
    )
    print(f"dimensions with standard deviation below 0.001: {int((std < 0.001).sum())}; below 0.01: {int((std < 0.01).sum())}")
    print(f"total variance (sum over dimensions): {np.sum(std**2):.4f} (1 minus the squared length of the mean vector)")
    print(f"mean per dimension: min {mean.min():.4f}, max {mean.max():.4f}, largest absolute {np.abs(mean).max():.4f}")
    order = np.argsort(-std)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(np.arange(dimension), std[order], width=0.8, color=eq.SERIES[0], label="CSIC vectors from the NPU (dimensions sorted)")
    ax.axhline(1 / np.sqrt(dimension), color=eq.MUTED, lw=2, label=f"random unit vectors: 1/sqrt(128) = {1 / np.sqrt(dimension):.3f}")
    ax.set_xlabel("dimension, sorted by standard deviation")
    ax.set_ylabel("standard deviation of the value")
    ax.set_title("Spread of each of the 128 dimensions over the distinct inputs")
    ax.set_xlim(-1, dimension)
    ax.set_ylim(0, 0.115)
    ax.legend(loc="upper right")
    eq.save(fig, args.out, "q1-dimension-spread.png")


if __name__ == "__main__":
    main()
