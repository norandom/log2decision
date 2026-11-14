"""Question 3: do good and bad logs, and the endpoints, show structure?

Projections (PCA, UMAP) coloured by label and by endpoint, k-means against label and
endpoint, silhouette, nearest-neighbour agreement, each beside a control (shuffled
labels, random unit vectors) and separately for the texts the encoder was not fitted on.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="n_jobs value")
import umap
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

import eq

K = 11


def purity(clusters: np.ndarray, classes: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(clusters):
        _, counts = np.unique(classes[clusters == cluster], return_counts=True)
        total += counts.max()
    return total / clusters.size


def against(name: str, clusters: np.ndarray, classes: np.ndarray, rng: np.random.Generator) -> None:
    shuffled = rng.permutation(classes)
    _, counts = np.unique(classes, return_counts=True)
    print(
        f"    {name} (n {classes.size}, {counts.size} classes, largest class {counts.max() / classes.size:.3f}): "
        f"adjusted Rand {adjusted_rand_score(classes, clusters):.3f}, NMI {normalized_mutual_info_score(classes, clusters):.3f}, "
        f"purity {purity(clusters, classes):.3f}; shuffled classes: adjusted Rand {adjusted_rand_score(shuffled, clusters):.3f}, "
        f"NMI {normalized_mutual_info_score(shuffled, clusters):.3f}, purity {purity(clusters, shuffled):.3f}"
    )


def neighbours(points: np.ndarray, queries: np.ndarray, k: int, drop_self: bool) -> np.ndarray:
    index = NearestNeighbors(n_neighbors=k + int(drop_self), metric="cosine", algorithm="brute").fit(points)
    found = index.kneighbors(queries, return_distance=False)
    return found[:, 1:] if drop_self else found


def agreement(vectors: np.ndarray, classes: np.ndarray) -> float:
    """Mean share of the K nearest other texts that have the text's own class."""
    found = neighbours(vectors, vectors, K, drop_self=True)
    return float((classes[found] == classes[:, None]).mean())


def chance(classes: np.ndarray) -> float:
    _, counts = np.unique(classes, return_counts=True)
    share = counts / counts.sum()
    return float((share**2).sum())


def scatter(ax, points: np.ndarray, colours: np.ndarray, title: str, handles: list, rng: np.random.Generator, legend_columns: int = 1) -> None:
    order = rng.permutation(points.shape[0])
    ax.scatter(points[order, 0], points[order, 1], c=colours[order], s=3, alpha=0.55, linewidth=0, rasterized=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if handles:
        ax.legend(handles=handles, loc="best", ncols=legend_columns, markerscale=1.0, fontsize=7.5, handletextpad=0.2, columnspacing=0.8)


def handle(colour: str, label: str) -> Line2D:
    return Line2D([], [], marker="o", ls="", color=colour, markersize=6, label=label)


def main() -> None:
    args = eq.arguments(__doc__)
    eq.style()
    rng = np.random.default_rng(eq.SEED)
    csic = eq.load_csic(args.vectors, args.record)
    record = csic.record

    # The split, checked against the training record.
    for name, expected in [
        ("held-out", record["split"]["held_out_logs"]),
        ("validation", record["settings"]["validation_logs"]),
        ("training", record["settings"]["training_logs"]),
    ]:
        got = int((csic.split == name).sum())
        print(f"all texts, {name}: {got} (training record: {expected})")
        assert got == expected, "the split here is not the training record's"
    print(f"held-out paths: {record['split']['held_out_mixed_paths']}; validation path: {record['settings']['validation_mixed_paths']}")
    print(f"model: epoch {record['training']['chosen_epoch']} of {record['settings']['epochs']}, chosen on the validation path; objective {record['settings']['objective']}, routing {record['settings']['routing']}")

    keep = csic.keep
    vectors = csic.vectors[keep]
    bad = csic.bad[keep]
    group = csic.group[keep]
    split = csic.split[keep]
    fitting = split != "held-out"
    held = ~fitting
    count, dimension = vectors.shape
    for name in ["training", "validation", "held-out"]:
        mask = split == name
        print(f"analysis set, {name}: {int(mask.sum())} texts, good {int((~bad[mask]).sum())}, bad {int(bad[mask].sum())}")
    baseline = eq.random_unit(count, dimension)

    # Endpoints with both labels get a colour each; the rest is "other".
    mixed = sorted(g for g in np.unique(group) if bad[group == g].any() and (~bad[group == g]).any())
    print(f"endpoints in the analysis set: {np.unique(group).size}; with both labels: {len(mixed)}")
    for g in mixed:
        mask = group == g
        print(f"    {g}: {int(mask.sum())} texts, bad {int(bad[mask].sum())}, split {split[mask][0]}")
    in_mixed = np.isin(group, mixed)
    print(f"texts on other endpoints: {int((~in_mixed).sum())} (bad {int(bad[~in_mixed].sum())}, good {int((~bad[~in_mixed]).sum())}) on {np.unique(group[~in_mixed]).size} paths")
    assert len(mixed) <= len(eq.SERIES)
    # A fixed colour per endpoint. `editar.jsp` and `registro.jsp` take the same form and
    # lie on top of each other, so they get the two colours that are easiest to tell apart.
    order = ["editar", "registro", "autenticar", "pagar", "anadir", "caracteristicas", "entrar", "vaciar"]
    mixed = sorted(mixed, key=lambda g: next((i for i, name in enumerate(order) if name in g), len(order)))
    endpoint_colour = {g: eq.SERIES[i] for i, g in enumerate(mixed)}
    by_label = np.where(bad, eq.BAD, eq.GOOD)
    by_endpoint = np.array([endpoint_colour.get(g, eq.MUTED) for g in group])
    label_handles = [handle(eq.GOOD, "good"), handle(eq.BAD, "bad")]
    endpoint_handles = [handle(endpoint_colour[g], g.replace("/tienda1/", "")) for g in mixed] + [handle(eq.MUTED, "other paths")]

    # --- Projections ---
    pca = PCA(n_components=2, random_state=eq.SEED).fit(vectors)
    flat = pca.transform(vectors)
    print(f"PCA to 2 dimensions keeps {pca.explained_variance_ratio_.sum():.3f} of the variance ({pca.explained_variance_ratio_[0]:.3f} + {pca.explained_variance_ratio_[1]:.3f})")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    scatter(axes[0], flat, by_label, "PCA, first two components, by label", label_handles, rng)
    scatter(axes[1], flat, by_endpoint, "PCA, first two components, by endpoint", endpoint_handles, rng)
    for ax in axes:
        ax.set_xlabel(f"component 1 ({pca.explained_variance_ratio_[0]:.0%} of the variance)")
        ax.set_ylabel(f"component 2 ({pca.explained_variance_ratio_[1]:.0%})")
    fig.suptitle(f"{count:,} distinct CSIC inputs, vectors from the NPU", fontsize=12, fontweight="bold", color=eq.INK)
    fig.tight_layout()
    eq.save(fig, args.out, "q3-pca.png")

    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=eq.SEED)
    low = reducer.fit_transform(vectors)
    low_random = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=eq.SEED).fit_transform(baseline)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.5))
    scatter(axes[0][0], low, by_label, "UMAP, by label (all distinct inputs)", label_handles, rng)
    scatter(axes[0][1], low, by_endpoint, "UMAP, by endpoint", endpoint_handles, rng)
    axes[1][0].scatter(low[fitting, 0], low[fitting, 1], c="#d9d8d2", s=3, linewidth=0, rasterized=True)
    scatter(
        axes[1][0],
        low[held],
        by_label[held],
        "UMAP, only the endpoints the encoder was not fitted on, by label",
        label_handles + [handle("#d9d8d2", "fitted on (training, validation)")],
        rng,
    )
    scatter(axes[1][1], low_random, np.where(bad, eq.BAD, eq.GOOD), "Baseline: UMAP of random unit vectors, same labels", label_handles, rng)
    fig.suptitle(f"UMAP (cosine, 15 neighbours) of {count:,} distinct CSIC inputs", fontsize=12, fontweight="bold", color=eq.INK)
    fig.tight_layout()
    eq.save(fig, args.out, "q3-umap.png")

    # --- k-means ---
    print("k-means on all distinct inputs (10 starts):")
    for k in [2, 8, 20, 50]:
        clusters = KMeans(n_clusters=k, n_init=10, random_state=eq.SEED).fit_predict(vectors)
        clusters_random = KMeans(n_clusters=k, n_init=10, random_state=eq.SEED).fit_predict(baseline)
        sizes = np.sort(np.bincount(clusters))[::-1]
        print(f"  k = {k}: cluster sizes {sizes[:8].tolist()}{' ...' if k > 8 else ''}")
        against("label", clusters, bad, rng)
        against("label, held-out texts", clusters[held], bad[held], rng)
        against("endpoint (texts on the endpoints with both labels)", clusters[in_mixed], group[in_mixed], rng)
        against("endpoint, good texts only", clusters[in_mixed & ~bad], group[in_mixed & ~bad], rng)
        against("label, clusters of random unit vectors", clusters_random, bad, rng)
        against("endpoint, clusters of random unit vectors", clusters_random[in_mixed], group[in_mixed], rng)
        if k == 8:
            clusters8 = clusters
    order = np.argsort([-bad[clusters8 == c].mean() for c in range(8)])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    goods = [int((~bad[clusters8 == c]).sum()) for c in order]
    bads = [int(bad[clusters8 == c].sum()) for c in order]
    ys = np.arange(8)
    axes[0].barh(ys, goods, color=eq.GOOD, label="good", height=0.7)
    axes[0].barh(ys, bads, left=goods, color=eq.BAD, label="bad", height=0.7, edgecolor=eq.SURFACE, linewidth=1)
    for y, (g, b) in enumerate(zip(goods, bads)):
        axes[0].text(g + b + 60, y, f"{b / (g + b):.0%} bad", va="center", fontsize=8, color=eq.INK2)
    axes[0].set_yticks(ys, [f"cluster {i + 1}" for i in range(8)])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("texts")
    axes[0].set_xlim(0, max(g + b for g, b in zip(goods, bads)) * 1.2)
    axes[0].set_title("k-means, 8 clusters: labels in each cluster")
    axes[0].legend(loc="center right")
    columns = mixed + ["other paths"]
    table = np.array([[int(((group == g) if g in mixed else ~in_mixed)[clusters8 == c].sum()) for g in columns] for c in order], dtype=float)
    share = table / table.sum(axis=0, keepdims=True)
    image = axes[1].imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for y in range(8):
        for x in range(len(columns)):
            if share[y, x] >= 0.005:
                axes[1].text(x, y, f"{share[y, x]:.2f}", ha="center", va="center", fontsize=7.5, color="#ffffff" if share[y, x] > 0.55 else eq.INK)
    axes[1].set_xticks(range(len(columns)), [c.replace("/tienda1/", "") for c in columns], rotation=35, ha="right", fontsize=7.5)
    axes[1].set_yticks(ys, [f"cluster {i + 1}" for i in range(8)])
    axes[1].grid(False)
    axes[1].set_title("Share of each endpoint's texts in each cluster (columns sum to 1)")
    fig.colorbar(image, ax=axes[1], fraction=0.04, pad=0.02).set_label("share of the endpoint")
    fig.tight_layout()
    eq.save(fig, args.out, "q3-kmeans.png")

    # --- Silhouette (cosine), on a sample ---
    sample = rng.choice(count, 6000, replace=False)
    print("silhouette (cosine distance; sample of 6,000 distinct inputs unless said otherwise):")

    def silhouette(name: str, points: np.ndarray, classes: np.ndarray) -> None:
        if points.shape[0] > 6000:
            pick = rng.choice(points.shape[0], 6000, replace=False)
            points, classes = points[pick], classes[pick]
        real = silhouette_score(points, classes, metric="cosine")
        shuffled = silhouette_score(points, rng.permutation(classes), metric="cosine")
        print(f"    {name} (n {points.shape[0]}): {real:.3f}; shuffled classes {shuffled:.3f}")

    silhouette("by label", vectors[sample], bad[sample])
    silhouette("by label, held-out texts", vectors[held], bad[held])
    silhouette("by label, fitting side", vectors[fitting], bad[fitting])
    silhouette("by endpoint (endpoints with both labels)", vectors[in_mixed], group[in_mixed])
    silhouette("by endpoint, good texts only", vectors[in_mixed & ~bad], group[in_mixed & ~bad])
    silhouette("by endpoint, bad texts only", vectors[in_mixed & bad], group[in_mixed & bad])
    silhouette("by label, random unit vectors", baseline[sample], bad[sample])
    silhouette("by endpoint, random unit vectors", baseline[in_mixed], group[in_mixed])
    silhouette("by k-means cluster, k = 8", vectors[sample], clusters8[sample])

    # --- Nearest neighbours ---
    print(f"agreement with the {K} nearest other texts (mean share with the text's own class):")
    rows = []

    def agree(name: str, points: np.ndarray, classes: np.ndarray, random_points: np.ndarray) -> None:
        real = agreement(points, classes)
        shuffled = agreement(points, rng.permutation(classes))
        random = agreement(random_points, classes)
        print(f"    {name} (n {classes.size}): {real:.3f}; shuffled classes {shuffled:.3f}; random unit vectors {random:.3f}; chance by class sizes {chance(classes):.3f}")
        rows.append((name, real, shuffled, random))

    agree("label, all", vectors, bad, baseline)
    agree("label, held-out texts among themselves", vectors[held], bad[held], baseline[held])
    agree("endpoint, endpoints with both labels", vectors[in_mixed], group[in_mixed], baseline[in_mixed])
    agree("endpoint, good texts only", vectors[in_mixed & ~bad], group[in_mixed & ~bad], baseline[in_mixed & ~bad])
    agree("endpoint, bad texts only", vectors[in_mixed & bad], group[in_mixed & bad], baseline[in_mixed & bad])

    fig, ax = plt.subplots(figsize=(9, 4))
    ys = np.arange(len(rows))
    height = 0.26
    for offset, (column, colour, label) in enumerate([(1, eq.SERIES[0], "vectors from the NPU"), (2, eq.SERIES[3], "same vectors, classes shuffled"), (3, eq.MUTED, "random unit vectors")]):
        values = [row[column] for row in rows]
        ax.barh(ys + (offset - 1) * height, values, height=height - 0.03, color=colour, label=label)
        for y, value in zip(ys, values):
            ax.text(value + 0.01, y + (offset - 1) * height, f"{value:.2f}", va="center", fontsize=7.5, color=eq.INK2)
    ax.set_yticks(ys, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(f"mean share of the {K} nearest other texts with the same class")
    ax.set_title("Nearest-neighbour agreement, with two controls")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
    eq.save(fig, args.out, "q3-neighbour-agreement.png")

    # Held-out texts against the fitting side: what the neighbour scorer would see.
    print(f"held-out texts ({int(held.sum())}) scored by their {K} nearest texts of the fitting side ({int(fitting.sum())}):")
    found = neighbours(vectors[fitting], vectors[held], K, drop_self=False)
    share = bad[fitting][found].mean(axis=1)
    found_random = neighbours(baseline[fitting], baseline[held], K, drop_self=False)
    share_random = bad[fitting][found_random].mean(axis=1)
    shuffles = [rng.permutation(bad[fitting]) for _ in range(20)]
    shuffled_auc = np.array([roc_auc_score(bad[held], s[found].mean(axis=1)) for s in shuffles])
    print(
        f"    ROC AUC of the bad share: {roc_auc_score(bad[held], share):.3f}; fitting labels shuffled, 20 times: mean {shuffled_auc.mean():.3f}, "
        f"min {shuffled_auc.min():.3f}, max {shuffled_auc.max():.3f}; random unit vectors {roc_auc_score(bad[held], share_random):.3f}"
    )
    print(f"    bad share among the neighbours: held-out good texts mean {share[~bad[held]].mean():.3f}, min {share[~bad[held]].min():.3f}; held-out bad texts mean {share[bad[held]].mean():.3f}")
    vote = share > 0.5
    print(f"    majority vote: accuracy {np.mean(vote == bad[held]):.3f} (always the larger class: {max(bad[held].mean(), 1 - bad[held].mean()):.3f}); bad found {np.mean(vote[bad[held]]):.3f}, good kept {np.mean(~vote[~bad[held]]):.3f}")
    for g in sorted(np.unique(group[held])):
        mask = group[held] == g
        if bad[held][mask].any() and (~bad[held][mask]).any():
            print(f"    {g}: n {int(mask.sum())}, ROC AUC of the bad share {roc_auc_score(bad[held][mask], share[mask]):.3f}")
    training = split == "training"
    found_training = neighbours(vectors[training], vectors[held], K, drop_self=False)
    print(f"    neighbours among the training texts only ({int(training.sum())}): ROC AUC {roc_auc_score(bad[held], bad[training][found_training].mean(axis=1)):.3f}")
    # The same on all texts, an input as often as CSIC has distinct texts with it (the
    # count the earlier documents used), the texts with an empty input included.
    all_held = csic.split == "held-out"
    found_all = neighbours(csic.vectors[~all_held], csic.vectors[all_held], K, drop_self=False)
    print(f"    all {int(all_held.sum())} held-out texts against all {int((~all_held).sum())} fitting-side texts, repeated inputs included: ROC AUC {roc_auc_score(csic.bad[all_held], csic.bad[~all_held][found_all].mean(axis=1)):.3f}")
    # How much of it the first principal components carry.
    parts = PCA(n_components=10, random_state=eq.SEED).fit(vectors[fitting])
    reduced = parts.transform(vectors)
    first = roc_auc_score(bad[held], reduced[held, 0])
    print(f"    first principal component of the fitting side alone as the score: ROC AUC on the held-out texts {max(first, 1 - first):.3f}, on the fitting side {max(roc_auc_score(bad[fitting], reduced[fitting, 0]), 1 - roc_auc_score(bad[fitting], reduced[fitting, 0])):.3f}")
    for n in [1, 2, 5, 10]:
        index = NearestNeighbors(n_neighbors=K, algorithm="brute").fit(reduced[fitting, :n])
        near = index.kneighbors(reduced[held, :n], return_distance=False)
        print(f"    neighbours by the first {n} principal components only (Euclidean): ROC AUC {roc_auc_score(bad[held], bad[fitting][near].mean(axis=1)):.3f}")
    head = LogisticRegression(C=1.0, max_iter=5000).fit(vectors[fitting], bad[fitting])
    shuffled_head = np.array([roc_auc_score(bad[held], LogisticRegression(C=1.0, max_iter=5000).fit(vectors[fitting], s).decision_function(vectors[held])) for s in shuffles])
    head_random = LogisticRegression(C=1.0, max_iter=5000).fit(baseline[fitting], bad[fitting])
    print(
        f"    linear head (logistic regression, C = 1) fitted on the fitting side: ROC AUC on the held-out texts {roc_auc_score(bad[held], head.decision_function(vectors[held])):.3f}; "
        f"labels shuffled, 20 times: mean {shuffled_head.mean():.3f}, min {shuffled_head.min():.3f}, max {shuffled_head.max():.3f}, further than 0.2 from 0.5 in {int((np.abs(shuffled_head - 0.5) > 0.2).sum())}; random unit vectors {roc_auc_score(bad[held], head_random.decision_function(baseline[held])):.3f}; "
        f"on the fitting side itself {roc_auc_score(bad[fitting], head.decision_function(vectors[fitting])):.3f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    bins = (np.arange(K + 2) - 0.5) / K
    for ax, values, title in [(axes[0], share, "vectors from the NPU"), (axes[1], share_random, "baseline: random unit vectors")]:
        width = 0.4 / K
        centres = np.arange(K + 1) / K
        good_counts = np.histogram(values[~bad[held]], bins=bins)[0] / (~bad[held]).sum()
        bad_counts = np.histogram(values[bad[held]], bins=bins)[0] / bad[held].sum()
        ax.bar(centres - width / 2, good_counts, width=width * 0.92, color=eq.GOOD, label="held-out good texts")
        ax.bar(centres + width / 2, bad_counts, width=width * 0.92, color=eq.BAD, label="held-out bad texts")
        ax.set_xlabel(f"bad share among the {K} nearest texts of the fitting side")
        ax.set_title(title)
        ax.set_xticks(centres, [f"{i}/{K}" for i in range(K + 1)], fontsize=7.5)
    axes[0].set_ylabel("share of the texts of that label")
    axes[0].legend(loc="upper center")
    fig.suptitle("Texts of the three endpoints the encoder was not fitted on, scored by their neighbours", fontsize=12, fontweight="bold", color=eq.INK)
    fig.tight_layout()
    eq.save(fig, args.out, "q3-held-out-neighbours.png")


if __name__ == "__main__":
    main()
