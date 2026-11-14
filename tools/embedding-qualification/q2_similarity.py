"""Question 2: do similar strings land near each other and unrelated ones apart?

Probe pairs by kind, the relation between a text distance and the vector distance on
CSIC pairs (with a rank correlation), and nearest neighbours shown as text.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

import eq


def ngrams(data: bytes, n: int = 3) -> set[bytes]:
    return {data[i : i + n] for i in range(max(len(data) - n + 1, 1))}


def jaccard_distance(a: bytes, b: bytes) -> float:
    x, y = ngrams(a), ngrams(b)
    return 1.0 - len(x & y) / len(x | y)


def edit_distance(a: bytes, b: bytes) -> float:
    """Levenshtein distance over the bytes, divided by the longer length."""
    if not a or not b:
        return 1.0 if (a or b) else 0.0
    x = np.frombuffer(a, dtype=np.uint8)
    y = np.frombuffer(b, dtype=np.uint8)
    positions = np.arange(y.size + 1)
    previous = positions.copy()
    for i, value in enumerate(x, start=1):
        best = np.minimum(previous[1:] + 1, previous[:-1] + (y != value))
        # Insertions run along the row: a running minimum of (cost - position).
        previous = np.minimum.accumulate(np.concatenate(([i], best)) - positions) + positions
    return float(previous[-1]) / max(x.size, y.size)


def probe_pairs(rows: list[dict]) -> list[tuple[str, int, int]]:
    """(kind of pair, probe, probe) for the comparisons of the document."""
    at = {row["name"]: i for i, row in enumerate(rows)}
    pairs = []
    for i, row in enumerate(rows):
        if row["base"]:
            pairs.append((row["relation"], i, at[row["base"]]))
    attacks = [row for row in rows if row["kind"].startswith("attack")]
    for i, a in enumerate(attacks):
        for b in attacks[i + 1 :]:
            value_a, value_b = a["name"].rsplit("-", 1)[0], b["name"].rsplit("-", 1)[0]
            request_a, request_b = a["name"].rsplit("-", 1)[1], b["name"].rsplit("-", 1)[1]
            if value_a == value_b:
                pairs.append(("same attack value, other request", at[a["name"]], at[b["name"]]))
            elif request_a == request_b and a["kind"] != b["kind"]:
                pairs.append(("other attack class, same request", at[a["name"]], at[b["name"]]))
            elif request_a == request_b:
                pairs.append(("same attack class, other value, same request", at[a["name"]], at[b["name"]]))
    benign = [row["name"] for row in rows if row["kind"] == "benign"]
    randoms = [row["name"] for row in rows if row["kind"] == "random"]
    for i, a in enumerate(benign):
        for b in benign[i + 1 :]:
            pairs.append(("different benign requests", at[a], at[b]))
    for i, a in enumerate(randoms):
        for b in randoms[i + 1 :]:
            pairs.append(("two random texts", at[a], at[b]))
        for b in benign:
            pairs.append(("random text and benign request", at[a], at[b]))
    return pairs


ORDER = [
    "duplicate",
    "same payload, other endpoint",
    "differs after byte 256",
    "one character",
    "one value",
    "parameter order",
    "case",
    "attack in a benign request",
    "same attack value, other request",
    "same attack class, other value, same request",
    "other attack class, same request",
    "different benign requests",
    "random text and benign request",
    "two random texts",
]


def main() -> None:
    args = eq.arguments(__doc__)
    eq.style()
    csic = eq.load_csic(args.vectors, args.record)
    probes, rows = eq.load_probes(args.vectors)
    kept = csic.vectors[csic.keep]
    kept_inputs = [csic.inputs[i] for i in csic.keep]
    count = kept.shape[0]

    # The scale: cosine of random CSIC pairs.
    a, b = eq.random_pairs(count, 200_000)
    reference = np.sort(eq.pair_cosine(kept, a, b))
    percentile = lambda value: 100.0 * np.searchsorted(reference, value) / reference.size

    # --- Probes ---
    print(f"probe texts: {len(rows)}")
    for row in rows:
        print(
            f"  probe {row['name']} [{row['kind']}] input bytes {row['read_bytes']}, truncated {row['truncated']}, "
            f"longer {row['text_longer_than_input']}, empty {row['empty']}: {eq.short(row['text'], 150)}"
        )
    pairs = probe_pairs(rows)
    cosines: dict[str, list[float]] = {}
    for kind, i, j in pairs:
        value = float(np.dot(probes[i].astype(np.float64), probes[j].astype(np.float64)))
        cosines.setdefault(kind, []).append(value)
        identical = bool(np.array_equal(probes[i], probes[j]))
        print(
            f"  pair [{kind}] {rows[i]['name']} ~ {rows[j]['name']}: cosine {value:.4f} "
            f"(percentile among random CSIC pairs {percentile(value):.1f}){' identical bits' if identical else ''}"
        )
    assert set(cosines) <= set(ORDER), set(cosines) - set(ORDER)
    for kind in ORDER:
        values = np.array(cosines.get(kind, []))
        if values.size:
            print(f"probe pairs, {kind}: n {values.size}, min {values.min():.4f}, median {np.median(values):.4f}, max {values.max():.4f}")
    empties = [i for i, row in enumerate(rows) if row["empty"]]
    same = all(np.array_equal(probes[empties[0]], probes[i]) for i in empties)
    print(f"probes with an empty input: {[rows[i]['name'] for i in empties]}; all the same vector: {same}")
    if csic.empty.any():
        csic_empty = csic.vectors[np.flatnonzero(csic.empty)[0]]
        print(f"the same vector as CSIC's texts with an empty input: {bool(np.array_equal(csic_empty, probes[empties[0]]))}")
    else:
        print("no CSIC text has an empty input")

    kinds = [kind for kind in ORDER if kind in cosines]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    rng = np.random.default_rng(eq.SEED)
    for y, kind in enumerate(kinds):
        values = np.array(cosines[kind])
        ax.scatter(values, y + rng.uniform(-0.18, 0.18, values.size), s=26, color=eq.SERIES[0], alpha=0.75, edgecolor=eq.SURFACE, linewidth=0.6)
        ax.plot([np.median(values)] * 2, [y - 0.32, y + 0.32], color=eq.INK, lw=1.6)
    q = np.quantile(reference, [0.05, 0.5, 0.95])
    ax.axvspan(q[0], q[2], color=eq.MUTED, alpha=0.15, label="random CSIC pairs, 5th to 95th percentile")
    ax.axvline(q[1], color=eq.MUTED, lw=1.5, label=f"random CSIC pairs, median {q[1]:.2f}")
    ax.set_yticks(range(len(kinds)), [f"{kind} (n = {len(cosines[kind])})" for kind in kinds])
    ax.invert_yaxis()
    ax.set_xlabel("cosine similarity of the pair (black bar: median)")
    ax.set_xlim(-0.6, 1.03)
    ax.set_title("Probe pairs by kind: cosine similarity of the two vectors")
    ax.legend(loc="lower left")
    eq.save(fig, args.out, "q2-probe-pairs.png")

    # Heatmap of the probes.
    names = [row["name"] for row in rows]
    matrix = probes.astype(np.float64) @ probes.astype(np.float64).T
    cmap = LinearSegmentedColormap.from_list("poles", [eq.SERIES[1], "#f0efec", eq.SERIES[0]])
    fig, ax = plt.subplots(figsize=(13.5, 12.5))
    image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(names)), names, fontsize=6.5)
    ax.grid(False)
    ax.set_title("Cosine similarity between all probe texts (same order on both axes)")
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("cosine similarity")
    eq.save(fig, args.out, "q2-probe-heatmap.png")

    # --- Text distance against vector distance on CSIC pairs ---
    rng = np.random.default_rng(eq.SEED)
    pairs_random = list(zip(*eq.random_pairs(count, 4_000, seed=eq.SEED + 1)))
    group = csic.group[csic.keep]
    bad = csic.bad[csic.keep]
    members: dict[str, np.ndarray] = {}
    for name in np.unique(group):
        index = np.flatnonzero(group == name)
        if index.size >= 50:
            members[name] = index
    weights = np.array([m.size for m in members.values()], dtype=float)
    names_of = list(members)
    pairs_same = []
    while len(pairs_same) < 4_000:
        index = members[names_of[rng.choice(len(names_of), p=weights / weights.sum())]]
        i, j = rng.choice(index, 2, replace=False)
        pairs_same.append((int(i), int(j)))
    random_vectors = eq.random_unit(count, kept.shape[1])
    figure_data = {}
    for label, pair_list in [("random pairs", pairs_random), ("pairs within one endpoint", pairs_same)]:
        i = np.array([p[0] for p in pair_list])
        j = np.array([p[1] for p in pair_list])
        cosine_distance = 1.0 - eq.pair_cosine(kept, i, j)
        jaccard = np.array([jaccard_distance(kept_inputs[x], kept_inputs[y]) for x, y in pair_list])
        edit = np.array([edit_distance(kept_inputs[x], kept_inputs[y]) for x, y in pair_list])
        figure_data[label] = (jaccard, edit, cosine_distance)
        control = 1.0 - eq.pair_cosine(random_vectors, i, j)
        for name, text_distance in [("3-gram Jaccard distance", jaccard), ("normalised edit distance", edit)]:
            rho = spearmanr(text_distance, cosine_distance).statistic
            rho_control = spearmanr(text_distance, control).statistic
            print(f"{label} (n {i.size}), {name} against cosine distance: Spearman {rho:.3f}; with random unit vectors {rho_control:.3f}")
            for which, mask in [("good-good", ~bad[i] & ~bad[j]), ("bad-bad", bad[i] & bad[j]), ("good-bad", bad[i] != bad[j])]:
                if mask.sum() > 30:
                    print(f"    {which} (n {int(mask.sum())}): Spearman {spearmanr(text_distance[mask], cosine_distance[mask]).statistic:.3f}")
        print(f"{label}: Spearman between the two text distances {spearmanr(jaccard, edit).statistic:.3f}")
        # Bins of the text distance: median cosine distance in each.
        edges = np.linspace(0, 1, 6)
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (jaccard >= low) & ((jaccard < high) | (high == 1.0))
            if mask.sum():
                print(f"    Jaccard distance {low:.1f} to {high:.1f}: n {int(mask.sum())}, median cosine distance {np.median(cosine_distance[mask]):.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.2), sharey=True)
    for row_at, (label, (jaccard, edit, cosine_distance)) in enumerate(figure_data.items()):
        for column, (name, text_distance) in enumerate([("3-gram Jaccard distance of the inputs", jaccard), ("normalised edit distance of the inputs", edit)]):
            ax = axes[row_at][column]
            ax.scatter(text_distance, cosine_distance, s=5, color=eq.SERIES[0], alpha=0.25, linewidth=0)
            edges = np.linspace(0, 1, 11)
            centres, medians = [], []
            for low, high in zip(edges[:-1], edges[1:]):
                mask = (text_distance >= low) & ((text_distance < high) | (high == 1.0))
                if mask.sum() >= 20:
                    centres.append((low + high) / 2)
                    medians.append(np.median(cosine_distance[mask]))
            ax.plot(centres, medians, color=eq.INK, lw=2, marker="o", ms=4, label="median per tenth (20 pairs or more)")
            rho = spearmanr(text_distance, cosine_distance).statistic
            ax.set_title(f"{label}: Spearman {rho:.2f}")
            ax.set_xlabel(name)
            ax.set_xlim(-0.02, 1.02)
            if column == 0:
                ax.set_ylabel("cosine distance of the vectors")
            ax.legend(loc="upper left")
    fig.suptitle("Text distance against vector distance, 4,000 CSIC pairs per row", fontsize=12, fontweight="bold", color=eq.INK)
    fig.tight_layout()
    eq.save(fig, args.out, "q2-text-vs-vector-distance.png")

    # --- Nearest neighbours as text ---
    index = NearestNeighbors(n_neighbors=4, metric="cosine", algorithm="brute").fit(kept)
    split = csic.split[csic.keep]
    examples = []
    for want_bad, want_split, needle in [
        (False, "held-out", b""),
        (True, "held-out", b"DROP"),
        (True, "held-out", b"script"),
        (True, "training", b"etc"),
        (False, "training", b"registro"),
        (True, "training", b"SELECT"),
    ]:
        for i in range(count):
            if bad[i] == want_bad and split[i] == want_split and needle in kept_inputs[i] and i not in examples:
                examples.append(i)
                break
    print("nearest neighbours of CSIC examples (the example itself excluded):")
    for i in examples:
        distance, neighbour = index.kneighbors(kept[i : i + 1])
        row = csic.rows[csic.keep[i]]
        print(f"  QUERY [{row['label']}, {split[i]}] {eq.short(row['text'], 170)}")
        for d, n in zip(distance[0][1:], neighbour[0][1:]):
            other = csic.rows[csic.keep[n]]
            print(f"     {1 - d:.3f} [{other['label']}, {split[n]}] {eq.short(other['text'], 170)}")
    print("nearest CSIC neighbours of probes:")
    wanted = ["cart", "login", "search", "sqli-drop-cart", "xss-script-login", "traversal-passwd-pay", "cmdi-cat-cart", "random-printable-0", "random-percent-bytes-0", "short-payload-a1", "long-a"]
    at = {row["name"]: i for i, row in enumerate(rows)}
    bad_share = []
    for name in wanted:
        distance, neighbour = index.kneighbors(probes[at[name]][None, :], n_neighbors=3)
        print(f"  PROBE {name} [{rows[at[name]]['kind']}] {eq.short(rows[at[name]]['text'], 170)}")
        for d, n in zip(distance[0], neighbour[0]):
            other = csic.rows[csic.keep[n]]
            print(f"     {1 - d:.3f} [{other['label']}, {split[n]}] {eq.short(other['text'], 170)}")
    # The bad share among the 11 nearest CSIC texts, per probe kind.
    index11 = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute").fit(kept)
    distance, neighbour = index11.kneighbors(probes)
    print("bad share among the 11 nearest CSIC texts and cosine to the nearest, per probe:")
    by_kind: dict[str, list[float]] = {}
    for i, row in enumerate(rows):
        share = float(bad[neighbour[i]].mean())
        by_kind.setdefault(row["kind"], []).append(share)
        bad_share.append(share)
        print(f"  {row['name']} [{row['kind']}]: bad share {share:.2f}, nearest cosine {1 - distance[i][0]:.3f}")
    for kind, shares in by_kind.items():
        print(f"bad share among the 11 nearest, {kind}: n {len(shares)}, mean {np.mean(shares):.2f}, min {min(shares):.2f}, max {max(shares):.2f}")
    print(f"bad share of the analysis set: {bad.mean():.3f}")


if __name__ == "__main__":
    main()
