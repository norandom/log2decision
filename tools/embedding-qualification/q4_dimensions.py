"""Question 4: how many dimensions carry the variance?

PCA spectrum (scree and cumulative) beside random unit vectors, the number of components
for 90, 95 and 99 % of the variance, the participation ratio, dead dimensions, and how
many distinct values each dimension and each vector takes (the model is 8-bit quantised).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import eq


def spectrum(vectors: np.ndarray) -> np.ndarray:
    """Eigenvalues of the covariance, largest first."""
    centred = vectors.astype(np.float64) - vectors.astype(np.float64).mean(axis=0)
    values = np.linalg.svd(centred, compute_uv=False) ** 2 / (vectors.shape[0] - 1)
    return values


def summary(name: str, values: np.ndarray) -> np.ndarray:
    ratio = values / values.sum()
    cumulative = np.cumsum(ratio)
    needed = [int(np.searchsorted(cumulative, level) + 1) for level in (0.90, 0.95, 0.99)]
    participation = values.sum() ** 2 / (values**2).sum()
    print(
        f"{name}: components for 90 / 95 / 99 % of the variance: {needed[0]} / {needed[1]} / {needed[2]}; "
        f"participation ratio {participation:.1f}; first component {ratio[0]:.3f}, first five {cumulative[4]:.3f}, first ten {cumulative[9]:.3f}; "
        f"components above 1 % of the variance: {int((ratio > 0.01).sum())}; above 0.1 %: {int((ratio > 0.001).sum())}; "
        f"smallest eigenvalue over the largest {values[-1] / values[0]:.2e}"
    )
    return ratio


def grid_step(vector: np.ndarray) -> tuple[float, float, int]:
    """The step of the value grid of one vector, if its values are whole multiples of
    one step (an 8-bit output scaled to unit length): the smallest gap between two
    distinct values. Returns the step, the largest distance of a value from the grid in
    steps, and the largest absolute level."""
    values = np.unique(vector.astype(np.float64))
    if values.size < 2:
        return 0.0, 0.0, 0
    step = np.diff(values).min()
    levels = vector.astype(np.float64) / step
    return step, float(np.abs(levels - np.round(levels)).max()), int(np.abs(np.round(levels)).max())


def main() -> None:
    args = eq.arguments(__doc__)
    eq.style()
    csic = eq.load_csic(args.vectors, args.record)
    vectors = csic.vectors[csic.keep]
    bad = csic.bad[csic.keep]
    held = csic.split[csic.keep] == "held-out"
    count, dimension = vectors.shape
    baseline = eq.random_unit(count, dimension)

    ratio = summary(f"CSIC vectors {eq.SOURCE} ({count} distinct inputs)", spectrum(vectors))
    ratio_random = summary("random unit vectors, same number", spectrum(baseline))
    summary(f"good texts only ({int((~bad).sum())})", spectrum(vectors[~bad]))
    summary(f"bad texts only ({int(bad.sum())})", spectrum(vectors[bad]))
    summary(f"held-out texts only ({int(held.sum())})", spectrum(vectors[held]))
    summary(f"all {csic.vectors.shape[0]} texts, duplicates of an input included", spectrum(csic.vectors))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    index = np.arange(1, dimension + 1)
    axes[0].plot(index, ratio, color=eq.SERIES[0], lw=2, marker="o", ms=3, label=f"CSIC vectors {eq.SOURCE}")
    axes[0].plot(index, ratio_random, color=eq.MUTED, lw=2, label="random unit vectors")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("share of the variance (log scale)")
    axes[0].set_title("Scree plot")
    axes[0].legend(loc="upper right")
    cumulative = np.cumsum(ratio)
    axes[1].plot(index, cumulative, color=eq.SERIES[0], lw=2, label=f"CSIC vectors {eq.SOURCE}")
    axes[1].plot(index, np.cumsum(ratio_random), color=eq.MUTED, lw=2, label="random unit vectors")
    for level in (0.90, 0.95, 0.99):
        needed = int(np.searchsorted(cumulative, level) + 1)
        axes[1].plot([needed, needed], [0, level], color=eq.INK2, lw=0.8, ls=":")
        axes[1].annotate(f"{level:.0%}: {needed}", (needed, level), textcoords="offset points", xytext=(6, -11), fontsize=8, color=eq.INK)
    axes[1].set_xlabel("number of principal components")
    axes[1].set_ylabel("cumulative share of the variance")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("Cumulative variance")
    axes[1].legend(loc="lower right")
    fig.suptitle(f"PCA of {count:,} distinct CSIC inputs, 128 dimensions", fontsize=12, fontweight="bold", color=eq.INK)
    fig.tight_layout()
    eq.save(fig, args.out, "q4-pca-variance.png")

    # Dead dimensions and distinct values.
    std = vectors.astype(np.float64).std(axis=0)
    distinct = np.array([np.unique(vectors[:, d]).size for d in range(dimension)])
    zeros = (vectors == 0).mean(axis=0)
    print(f"dimensions that are constant over all texts: {int((distinct == 1).sum())}; with standard deviation below 0.001: {int((std < 0.001).sum())}; below 0.01: {int((std < 0.01).sum())}; below half the median: {int((std < np.median(std) / 2).sum())}")
    print(f"distinct values per dimension (in the unit vectors, over {count} texts): min {distinct.min()}, median {int(np.median(distinct))}, max {distinct.max()}")
    print(f"share of exact zeros per dimension: min {zeros.min():.4f}, median {np.median(zeros):.4f}, max {zeros.max():.4f}; over all values {(vectors == 0).mean():.4f}")

    # The 8-bit grid, per vector: the session divides the dequantised output by its length.
    per_vector = np.array([np.unique(v).size for v in vectors])
    print(f"distinct values within one vector (128 values): min {per_vector.min()}, median {int(np.median(per_vector))}, max {per_vector.max()}")
    steps = [grid_step(v) for v in vectors]
    off = np.array([s[1] for s in steps])
    levels = np.array([s[2] for s in steps])
    on_grid = off < 0.02
    if not on_grid.any():
        # Float vectors (an encoder that was not run in 8 bits) lie on no such grid.
        print(f"vectors whose values are whole multiples of their smallest gap (within 0.02 of a step): 0 of {count}; no 8-bit grid to recover")
        return
    print(
        f"vectors whose values are whole multiples of their smallest gap (within 0.02 of a step): {int(on_grid.sum())} of {count} ({on_grid.mean():.4f}); "
        f"largest absolute level among those: min {levels[on_grid].min()}, median {int(np.median(levels[on_grid]))}, max {levels[on_grid].max()}"
    )
    integer = np.stack([np.round(v.astype(np.float64) / s[0]) for v, s in zip(vectors[on_grid], steps_on(steps, on_grid))])
    distinct_levels = np.array([np.unique(integer[:, d]).size for d in range(dimension)])
    print(
        f"on that grid: levels from {int(integer.min())} to {int(integer.max())}; distinct levels per dimension: min {distinct_levels.min()}, "
        f"median {int(np.median(distinct_levels))}, max {distinct_levels.max()}; distinct levels over all dimensions {np.unique(integer).size}"
    )
    length = np.sqrt((integer**2).sum(axis=1))
    print(f"length of the output before normalising, in steps: p01 {np.quantile(length, 0.01):.1f}, median {np.median(length):.1f}, p99 {np.quantile(length, 0.99):.1f}, min {length.min():.1f}, max {length.max():.1f}")

    order = np.argsort(-distinct_levels)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    axes[0].bar(np.arange(dimension), distinct_levels[order], width=0.8, color=eq.SERIES[0])
    axes[0].set_xlabel("dimension, sorted")
    axes[0].set_ylabel("distinct 8-bit levels used")
    axes[0].set_xlim(-1, dimension)
    axes[0].set_title("Distinct output levels per dimension")
    values, counts = np.unique(integer, return_counts=True)
    axes[1].bar(values, counts / counts.sum(), width=0.9, color=eq.SERIES[0])
    axes[1].set_xlabel("output level (steps of the 8-bit output)")
    axes[1].set_ylabel("share of all values")
    axes[1].set_title("Output levels over all dimensions and texts")
    fig.suptitle("The 8-bit output recovered from the unit vectors", fontsize=12, fontweight="bold", color=eq.INK)
    fig.tight_layout()
    eq.save(fig, args.out, "q4-output-levels.png")


def steps_on(steps: list, mask: np.ndarray) -> list:
    return [s for s, keep in zip(steps, mask) if keep]


if __name__ == "__main__":
    main()
