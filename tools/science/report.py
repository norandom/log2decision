"""Collects the run records of `run_gate.py` into the tables of the gate document.

Usage: uv run python report.py <runs dir> [--variant bce] [--summary summary.json]
Prints Markdown. Figures are cross-endpoint (held-out paths that hold both labels, pooled).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def load(folder: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def variant_of(record: dict) -> str:
    # name = <split>-f<fold>-s<seed>-<routing>-<variant>
    return record["name"].split("-", 4)[4]


def cell(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{np.mean(values):.4f} ({min(values):.4f} to {max(values):.4f})"


def main() -> None:
    folder = sys.argv[1]
    variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "bce"
    records = load(folder)
    baselines = {(r["split"]["name"], r["split"]["fold"]): r for r in records if "baseline" in r}
    runs = [r for r in records if "settings" in r and variant_of(r) == variant]
    summary: dict = {"variant": variant, "splits": {}}
    uses = [("float", "knn_k11"), ("int8", "knn_k11"), ("float", "linear_head"), ("int8", "linear_head")]

    for split in ("cv5", "split17"):
        folds = sorted({r["split"]["fold"] for r in runs if r["split"]["name"] == split})
        if not folds:
            continue
        seeds = sorted({r["settings"]["seed"] for r in runs if r["split"]["name"] == split})
        print(f"\n## {split}: variant `{variant}`, seeds {seeds}\n")
        out = summary["splits"].setdefault(split, {"seeds": seeds, "folds": folds})
        for metric, label in (("roc_auc", "ROC AUC"), ("average_precision", "average precision")):
            print(f"\n### Cross-endpoint {label}; mean over seeds (lowest to highest seed)\n")
            print("| Fold | Held-out mixed paths | Baseline strip | Baseline keep | "
                  + " | ".join(f"{routing} {kind} {use}" for routing in ("strip", "keep") for kind, use in uses) + " |")
            print("|---|---|---:|---:|" + "---:|" * (2 * len(uses)))
            for fold in folds:
                base = baselines.get((split, fold))
                paths = ", ".join(p.rsplit("/", 1)[1] for p in
                                  next(r for r in runs if r["split"]["name"] == split and r["split"]["fold"] == fold)["split"]["held_out_mixed_paths"])
                row = [str(fold), paths]
                for routing in ("strip", "keep"):
                    row.append(f"{base['baseline'][routing]['cross_endpoint'][metric]:.4f}" if base else "n/a")
                for routing in ("strip", "keep"):
                    for kind, use in uses:
                        values = [r[kind][use]["cross_endpoint"][metric] for r in runs
                                  if r["split"]["name"] == split and r["split"]["fold"] == fold
                                  and r["settings"]["routing"] == routing]
                        row.append(cell(values))
                print("| " + " | ".join(row) + " |")

            # Mean over folds per seed, then the spread over seeds; and over folds.
            print(f"\n### {label}: mean over folds\n")
            print("| Scorer | Mean over folds, per seed | Mean of seeds | Seed spread (max - min) | Lowest fold (seed mean) | Highest fold (seed mean) |")
            print("|---|---|---:|---:|---:|---:|")
            for routing in ("strip", "keep"):
                values = [baselines[(split, f)]["baseline"][routing]["cross_endpoint"][metric]
                          for f in folds if (split, f) in baselines]
                if values:
                    print(f"| baseline {routing} | deterministic | {np.mean(values):.4f} | 0 | {min(values):.4f} | {max(values):.4f} |")
                    out.setdefault(metric, {})[f"baseline {routing}"] = {"mean": float(np.mean(values)), "folds": values}
            for routing in ("strip", "keep"):
                for kind, use in uses:
                    per_seed, per_fold = [], {}
                    for seed in seeds:
                        got = {r["split"]["fold"]: r[kind][use]["cross_endpoint"][metric] for r in runs
                               if r["split"]["name"] == split and r["settings"]["routing"] == routing
                               and r["settings"]["seed"] == seed}
                        if len(got) == len(folds):
                            per_seed.append(float(np.mean(list(got.values()))))
                            for f, v in got.items():
                                per_fold.setdefault(f, []).append(v)
                    if not per_seed:
                        continue
                    fold_means = [float(np.mean(v)) for v in per_fold.values()]
                    print(f"| encoder {routing} {kind} {use} | {', '.join(f'{v:.4f}' for v in per_seed)} | "
                          f"{np.mean(per_seed):.4f} | {max(per_seed) - min(per_seed):.4f} | "
                          f"{min(fold_means):.4f} | {max(fold_means):.4f} |")
                    out.setdefault(metric, {})[f"encoder {routing} {kind} {use}"] = {
                        "per_seed": per_seed, "mean": float(np.mean(per_seed)),
                        "seed_spread": max(per_seed) - min(per_seed),
                        "per_fold_seed_mean": {str(f): float(np.mean(v)) for f, v in sorted(per_fold.items())},
                        "per_fold_per_seed": {str(f): v for f, v in sorted(per_fold.items())}}

        chosen = [r for r in runs if r["split"]["name"] == split]
        print("\n### Float against 8-bit, training, export\n")
        print("| Routing | Runs | Cosine mean (lowest run) | Cosine 1st percentile (lowest run) | Share of logs below 0.99 (highest run) | "
              "8-bit minus float ROC AUC, neighbour ratio (mean, range) | same, linear head | Same operators as spike | "
              "Chosen epoch (range) | Training s (mean) | Run s (mean) |")
        print("|---|---:|---:|---:|---:|---|---|---|---|---:|---:|")
        for routing in ("strip", "keep"):
            part = [r for r in chosen if r["settings"]["routing"] == routing]
            if not part:
                continue
            cos = [r["export"]["float_vs_int8_cosine"] for r in part]
            delta = {use: [r["int8"][use]["cross_endpoint"]["roc_auc"] - r["float"][use]["cross_endpoint"]["roc_auc"]
                           for r in part] for use in ("knn_k11", "linear_head")}
            epochs = [r["training"]["chosen_epoch"] for r in part]
            same = sum(bool(r["export"]["spike_comparison"].get("identical")) for r in part)
            print(f"| {routing} | {len(part)} | {np.mean([c['all']['mean'] for c in cos]):.4f} ({min(c['all']['mean'] for c in cos):.4f}) | "
                  f"{np.mean([c['all']['p01'] for c in cos]):.4f} ({min(c['all']['p01'] for c in cos):.4f}) | "
                  f"{np.mean([c['share_below_0.99'] for c in cos]):.2%} ({max(c['share_below_0.99'] for c in cos):.2%}) | "
                  + " | ".join(f"{np.mean(delta[u]):+.4f} ({min(delta[u]):+.4f} to {max(delta[u]):+.4f})" for u in delta)
                  + f" | {same} of {len(part)} | {min(epochs)} to {max(epochs)} | "
                  f"{np.mean([r['training']['seconds'] for r in part]):.0f} | {np.mean([r['seconds'] for r in part]):.0f} |")
            out.setdefault("float_vs_int8", {})[routing] = {
                "cosine_mean": float(np.mean([c["all"]["mean"] for c in cos])),
                "cosine_mean_lowest_run": float(min(c["all"]["mean"] for c in cos)),
                "roc_auc_change": {u: {"mean": float(np.mean(delta[u])), "min": float(min(delta[u])), "max": float(max(delta[u]))} for u in delta},
                "same_operators_as_spike": f"{same} of {len(part)}"}

        print("\n### Other figures (8-bit model, mean over runs)\n")
        print("| Routing | Scorer | Overall ROC AUC (all held-out logs) | Cross-endpoint ROC AUC | Abstained share | "
              "Single-label logs | Bad single-label logs inside the budget | Good single-label logs inside the budget |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for routing in ("strip", "keep"):
            part = [r for r in chosen if r["settings"]["routing"] == routing]
            for use in ("knn_k5", "knn_k11", "knn_k51", "knn_k11_reach0.5", "linear_head"):
                if not part:
                    continue
                blocks = [r["int8"][use] for r in part]
                mean = lambda f: np.mean([v for v in map(f, blocks) if v is not None])
                print(f"| {routing} | {use} | {mean(lambda b: b['overall'].get('roc_auc')):.4f} | "
                      f"{mean(lambda b: b['cross_endpoint'].get('roc_auc')):.4f} | "
                      f"{mean(lambda b: b['overall']['abstained'] / b['overall']['logs']):.2%} | "
                      f"{mean(lambda b: b['single_label']['logs']):.0f} | "
                      f"{mean(lambda b: b['single_label']['bad_share_in_budget']):.2%} | "
                      f"{mean(lambda b: b['single_label']['good_share_in_budget']):.2%} |")
            if part and "training_classifier_reference" in part[0]:
                ref = [r["training_classifier_reference"]["cross_endpoint"]["roc_auc"] for r in part]
                print(f"| {routing} | training classifier (float, not in the tool) | | {np.mean(ref):.4f} | | | | |")
        for routing in ("strip", "keep"):
            part = [baselines[k] for k in baselines if k[0] == split]
            if part:
                blocks = [b["baseline"][routing] for b in part]
                print(f"| {routing} | baseline | {np.mean([b['overall']['roc_auc'] for b in blocks]):.4f} | "
                      f"{np.mean([b['cross_endpoint']['roc_auc'] for b in blocks]):.4f} | 0.00% | "
                      f"{np.mean([b['single_label']['logs'] for b in blocks]):.0f} | "
                      f"{np.mean([b['single_label']['bad_share_in_budget'] for b in blocks if b['single_label']['bad_share_in_budget'] is not None]):.2%} | "
                      f"{np.mean([b['single_label']['good_share_in_budget'] for b in blocks if b['single_label']['good_share_in_budget'] is not None]):.2%} |")

    if "--summary" in sys.argv:
        with open(sys.argv[sys.argv.index("--summary") + 1], "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
