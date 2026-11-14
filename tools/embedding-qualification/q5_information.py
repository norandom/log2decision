"""Information in the vectors: the variance spectrum against log2ml's encoders, its
Shannon entropy, and what a classifier loses when principal components are removed.

Added for the research log of 2026-11-14 (after task 2.15). Three parts:

1. log2ml's own check (notebook `Comparison_of_Encoder_Models.ipynb`): scale every
   feature to unit variance, run PCA, count the components that hold 95 % of the
   variance. The figures of log2ml's two encoders are copied from the printed output of
   that notebook (cells 41 and 44) and drawn beside this encoder's.
2. The Shannon entropy of the variance spectrum, in bits: the shares of the variance
   per principal component are a distribution; its entropy says over how many
   directions the vectors really spread (2 ** entropy is the effective rank).
3. Feature ablation: fit a classifier on all components, on the first k only, and on
   everything but the first k, for a task the encoder was trained on (good or bad, on
   the held-out endpoints) and one it was not (which endpoint a good text belongs to).
   Beside the score, the share of the class entropy the classifier recovers
   (1 - cross-entropy / entropy of the classes): a lower bound on the information the
   vectors carry about the classes, as a ratio.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import eq

# Printed by log2ml's notebook: explained variance ratios of the components that hold
# 95 % of the variance, features scaled with StandardScaler.
LOG2ML = {
    "log2ml LongFormer (768 features)": [
        0.61036086, 0.20625743, 0.0349943, 0.02877296, 0.01811446, 0.01343036,
        0.01034578, 0.00865575, 0.00768904, 0.0061328, 0.00457506, 0.00375366,
    ],
    "log2ml Linformer (30,000 features)": [
        0.7322417, 0.10258386, 0.01919063, 0.01368955, 0.01286633, 0.01047001,
        0.00869492, 0.00842645, 0.00745646, 0.00658019, 0.00584541, 0.00514998,
        0.00491944, 0.00477372, 0.00384961, 0.00355314,
    ],
}
KEPT = [1, 2, 3, 5, 10, 20, 50]
MIN_ENDPOINT_TEXTS = 30


def spectrum(values: np.ndarray) -> np.ndarray:
    """Shares of the variance per principal component, largest first."""
    centred = values.astype(np.float64) - values.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centred.T @ centred / (len(values) - 1))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvalues / eigenvalues.sum()


def entropy_bits(shares: np.ndarray) -> float:
    shares = shares[shares > 0]
    return float(-(shares * np.log2(shares)).sum())


def for_share(shares: np.ndarray, target: float) -> int:
    return int(np.searchsorted(np.cumsum(shares), target) + 1)


def class_entropy_bits(classes: np.ndarray) -> float:
    _, counts = np.unique(classes, return_counts=True)
    return entropy_bits(counts / counts.sum())


def head() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=5000)


def scaled(fit_on: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Every retained component at unit variance, so that a component with little
    variance is not ignored by the penalty: the question is whether the information is
    there at all, not whether it is large."""
    scaler = StandardScaler().fit(fit_on)
    return [scaler.transform(part) for part in (fit_on, *others)]


def label_task(scores: np.ndarray, bad: np.ndarray, fitting: np.ndarray) -> tuple[float, float]:
    """Fit on the fitting side, test on the held-out endpoints: ROC AUC, and the share
    of the label entropy of the held-out texts that the head recovers."""
    held = ~fitting
    train, test = scaled(scores[fitting], scores[held])
    model = head().fit(train, bad[fitting])
    auc = roc_auc_score(bad[held], model.decision_function(test))
    bits = log_loss(bad[held], model.predict_proba(test), labels=[False, True]) / np.log(2)
    return float(auc), 1.0 - bits / class_entropy_bits(bad[held])


def endpoint_task(scores: np.ndarray, endpoint: np.ndarray) -> tuple[float, float]:
    """Five stratified folds over the good texts: accuracy, and the share of the
    endpoint entropy the head recovers."""
    classes = np.unique(endpoint)
    right, bits = 0, 0.0
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=eq.SEED)
    for train_rows, test_rows in folds.split(scores, endpoint):
        train, test = scaled(scores[train_rows], scores[test_rows])
        model = head().fit(train, endpoint[train_rows])
        right += int((model.predict(test) == endpoint[test_rows]).sum())
        bits += log_loss(endpoint[test_rows], model.predict_proba(test), labels=classes, normalize=False) / np.log(2)
    return right / len(endpoint), 1.0 - bits / len(endpoint) / class_entropy_bits(endpoint)


def main() -> None:
    args = eq.arguments(__doc__.splitlines()[0])
    eq.style()
    warnings.filterwarnings("ignore", category=FutureWarning)
    csic = eq.load_csic(args.vectors, args.record)
    keep = csic.keep
    vectors = csic.vectors[keep].astype(np.float64)
    bad = csic.bad[keep]
    group = csic.group[keep]
    fitting = csic.split[keep] != "held-out"
    baseline = eq.random_unit(len(vectors), vectors.shape[1]).astype(np.float64)
    print(f"analysis set: {len(vectors)} distinct non-empty inputs, dimension {vectors.shape[1]}; held-out {int((~fitting).sum())}")

    # 1. log2ml's check: features at unit variance, components for 95 % of the variance.
    ours_scaled = spectrum(StandardScaler().fit_transform(vectors))
    ours_raw = spectrum(vectors)
    random_scaled = spectrum(StandardScaler().fit_transform(baseline))
    print("components that hold 95 % of the variance, features scaled to unit variance (log2ml's check):")
    print(f"    this encoder {eq.SOURCE} (128 features): {for_share(ours_scaled, 0.95)}; first component {ours_scaled[0]:.3f}, second {ours_scaled[1]:.3f}, third {ours_scaled[2]:.3f}")
    print(f"    this encoder, vectors as they are (not scaled): {for_share(ours_raw, 0.95)}; first component {ours_raw[0]:.3f}")
    for name, shares in LOG2ML.items():
        print(f"    {name}: {len(shares)}; first component {shares[0]:.3f} (printed by log2ml's notebook, Sysmon logs, pretrained models without training on labels)")
    print(f"    random unit vectors (128 features): {for_share(random_scaled, 0.95)}; first component {random_scaled[0]:.3f}")

    # 2. Shannon entropy of the variance spectrum.
    limit = np.log2(vectors.shape[1])
    rows = [
        ("this encoder, scaled features", ours_scaled),
        ("this encoder, vectors as they are", ours_raw),
        ("random unit vectors", random_scaled),
    ]
    print(f"Shannon entropy of the variance spectrum (bits; the most that 128 directions can give is {limit:.2f}):")
    for name, shares in rows:
        bits = entropy_bits(shares)
        print(f"    {name}: {bits:.2f} bits, effective rank {2 ** bits:.1f}, {bits / limit:.1%} of the most")
    for name, shares in LOG2ML.items():
        lower = entropy_bits(np.array(shares))
        print(f"    {name}: at least {lower:.2f} bits from the {len(shares)} printed components (the rest of the spectrum was not printed, so this is a lower bound)")
    print(f"distinct vectors among the {len(vectors)} inputs: {len(np.unique(csic.vectors[keep], axis=0))} ({np.log2(len(np.unique(csic.vectors[keep], axis=0))):.2f} of {np.log2(len(vectors)):.2f} bits if every input were told apart)")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.7, 1]})
    shown = 20
    ax = axes[0]
    series = [
        (f"this encoder {eq.SOURCE}", ours_scaled, eq.SERIES[0]),
        *[(name, np.array(shares), colour) for (name, shares), colour in zip(LOG2ML.items(), eq.SERIES[1:3])],
        ("random unit vectors", random_scaled, eq.MUTED),
    ]
    for name, shares, colour in series:
        cumulative = np.cumsum(shares)[:shown]
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=colour, linewidth=2, marker="o", markersize=4, label=name)
        if cumulative[-1] >= 0.95:
            at = for_share(shares, 0.95)
            ax.annotate(f"{at}", (at, np.cumsum(shares)[at - 1]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9.5, color=eq.INK)
    ax.axhline(0.95, color=eq.INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(shown + 0.3, 0.95, "95 %", va="center", fontsize=8.5, color=eq.INK2)
    ax.set_xlim(0.5, shown + 0.5)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([1, 2, 5, 10, 12, 16, 20])
    ax.set_xlabel("number of principal components (features scaled to unit variance)")
    ax.set_ylabel("cumulative share of the variance")
    ax.set_title("Components needed for 95 % of the variance")
    ax.legend(loc="center right")

    ax = axes[1]
    names = [f"this encoder\n{eq.SOURCE}", "random\nunit vectors"]
    bits = [entropy_bits(ours_scaled), entropy_bits(random_scaled)]
    bars = ax.bar(names, bits, color=[eq.SERIES[0], eq.MUTED], width=0.5)
    for bar, value in zip(bars, bits):
        inside = value > limit / 2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.15 if inside else value + 0.12,
            f"{value:.2f} bits",
            ha="center",
            va="top" if inside else "bottom",
            fontsize=9.5,
            color="#ffffff" if inside else eq.INK,
        )
    ax.axhline(limit, color=eq.INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(0.5, limit + 0.1, f"the most 128 directions can give: {limit:.0f} bits", va="bottom", ha="center", fontsize=8.5, color=eq.INK2)
    ax.set_ylim(0, limit + 1.0)
    ax.set_ylabel("Shannon entropy of the variance spectrum (bits)")
    ax.set_title("Over how many directions the vectors spread")
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"{len(vectors):,} distinct CSIC inputs: the variance spectrum against log2ml's encoders and against noise", fontsize=12, fontweight="bold")
    fig.tight_layout()
    eq.save(fig, args.out, "q5-variance-against-log2ml.png")

    # 3. Feature ablation.
    parts = PCA(n_components=vectors.shape[1], random_state=eq.SEED).fit(vectors[fitting])
    scores = parts.transform(vectors)
    good = ~bad
    names_good, counts = np.unique(group[good], return_counts=True)
    large = set(names_good[counts >= MIN_ENDPOINT_TEXTS])
    in_task = good & np.isin(group, list(large))
    endpoint = group[in_task]
    print(f"feature ablation; principal components of the fitting side ({int(fitting.sum())} texts); every retained component scaled to unit variance; logistic regression, C = 1")
    print(f"    label task: fitted on the fitting side, tested on the {int((~fitting).sum())} held-out texts; entropy of the label there {class_entropy_bits(bad[~fitting]):.3f} bits")
    print(f"    endpoint task: {int(in_task.sum())} good texts of the {len(large)} endpoints with {MIN_ENDPOINT_TEXTS} or more of them, five stratified folds; entropy of the endpoint {class_entropy_bits(endpoint):.3f} bits; the largest endpoint holds {np.unique(endpoint, return_counts=True)[1].max() / len(endpoint):.3f}")

    def both(name: str, columns: np.ndarray) -> tuple[float, float, float, float]:
        auc, label_share = label_task(scores[:, columns], bad, fitting)
        accuracy, endpoint_share = endpoint_task(scores[in_task][:, columns], endpoint)
        print(f"    {name}: label ROC AUC {auc:.3f}, share of the label entropy recovered {label_share:+.3f}; endpoint accuracy {accuracy:.3f}, share of the endpoint entropy recovered {endpoint_share:+.3f}")
        return auc, label_share, accuracy, endpoint_share

    everything = np.arange(vectors.shape[1])
    full = both("all 128 components", everything)
    first = [both(f"the first {k} only", everything[:k]) for k in KEPT]
    without = [both(f"all but the first {k}", everything[k:]) for k in KEPT]
    random_scores = PCA(n_components=vectors.shape[1], random_state=eq.SEED).fit(baseline[fitting]).transform(baseline)
    scores, kept_scores = random_scores, scores
    noise = both("random unit vectors, all 128 components", everything)
    scores = kept_scores
    chance = np.unique(endpoint, return_counts=True)[1].max() / len(endpoint)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, column, title, ylabel, floor, floor_name, side in (
        (axes[0], 0, "Good or bad, on the held-out endpoints", "ROC AUC", 0.5, "chance", "bottom"),
        (axes[1], 2, "Which endpoint a good text belongs to", "accuracy, five folds", chance, "largest endpoint", "top"),
    ):
        ax.plot(KEPT, [row[column] for row in first], color=eq.SERIES[0], linewidth=2, marker="o", markersize=5, label="the first k components only")
        ax.plot(KEPT, [row[column] for row in without], color=eq.SERIES[1], linewidth=2, marker="o", markersize=5, label="all but the first k components")
        ax.axhline(full[column], color=eq.INK2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(KEPT[-1] * 1.12, full[column], f"all 128: {full[column]:.3f}", va=side, ha="right", fontsize=8.5, color=eq.INK2)
        ax.axhline(floor, color=eq.MUTED, linewidth=0.9, linestyle=(0, (1, 2)))
        ax.text(KEPT[-1] * 1.12, floor, f"{floor_name}: {floor:.2f}", va="bottom", ha="right", fontsize=8.5, color=eq.INK2)
        ax.set_xscale("log")
        ax.set_xticks(KEPT, [str(k) for k in KEPT])
        ax.minorticks_off()
        ax.set_ylim(min(floor, min(row[column] for row in first + without)) - 0.05, 1.03)
        ax.set_xlabel("k, principal components in order of their variance")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(loc="center", bbox_to_anchor=(0.5, 0.68))
    fig.suptitle("Feature ablation: what a linear classifier keeps when components are removed", fontsize=12, fontweight="bold")
    fig.tight_layout()
    eq.save(fig, args.out, "q5-ablation.png")
    print(f"random unit vectors for scale: label ROC AUC {noise[0]:.3f}, endpoint accuracy {noise[2]:.3f}")


if __name__ == "__main__":
    main()
