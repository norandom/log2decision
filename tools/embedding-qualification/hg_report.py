"""The comparison table and the pictures of the hierarchical encoder gate (task 2.16).

Reads the logs `q1.log` to `q5.log` the qualification scripts wrote for every variant
(one directory per variant and precision), the training records of the variants, the
reference rows and the patching report of `tools/science/hg_run.py`, and writes one
table (Markdown and JSON) and the pictures.

    uv run python hg_report.py --root D:\\BuildArtifacts\\log2decision\\task-2.16 `
        --out ..\\..\\docs\\images\\<date>-hierarchical-encoder-gate

Development only; the l2d tool never runs this code.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eq

PUBLISHED_LOGS = Path(r"D:\BuildArtifacts\log2decision\task-2.15\logs")
# Fixed order of the variants. One colour per patcher (blue fixed stride, orange
# delimiter, aqua entropy), yellow for the flat control; grey for the two reference rows.
VARIANTS = [
    ("fixed", "fixed stride", eq.SERIES[0]),
    ("delimiter", "delimiter", eq.SERIES[1]),
    ("entropy", "entropy", eq.SERIES[2]),
    ("flat", "flat control", eq.SERIES[3]),
    ("entropy-label", "entropy + label", eq.SERIES[2]),
    ("fixed-jitter", "fixed stride, boundary drop", eq.SERIES[0]),
    ("delimiter-jitter", "delimiter, boundary drop", eq.SERIES[1]),
    ("entropy-jitter", "entropy, boundary drop", eq.SERIES[2]),
    ("entropy-global", "entropy (global rule)", eq.SERIES[2]),
    ("entropy-masked", "entropy, masked patches", eq.SERIES[2]),
]
PATCHERS = [("fixed", "fixed stride", eq.SERIES[0]), ("delimiter", "delimiter", eq.SERIES[1]),
            ("entropy-jump", "entropy (jump)", eq.SERIES[2])]


def find(pattern: str, text: str, cast=float, group: int = 1):
    match = re.search(pattern, text)
    return cast(match.group(group)) if match else None


def read_logs(directory: Path) -> dict | None:
    if not (directory / "q5.log").exists():
        return None
    q = {name: (directory / f"{name}.log").read_text(encoding="utf-8") for name in ("q1", "q2", "q3", "q4", "q5")}
    q["agreement"] = q["q3"][q["q3"].index("agreement with the 11 nearest"):]
    row = {
        "all_inputs": find(r"distinct encoder inputs (\d+)", q["q1"], int),
        "all_distinct_vectors": find(r"distinct vectors \(bit for bit\) (\d+)", q["q1"], int),
        "shared_vectors": find(r"vectors shared by different inputs: (\d+)", q["q1"], int),
        "inputs_sharing": find(r"inputs involved: (\d+)", q["q1"], int),
        "set_inputs": find(r"distinct vectors among the (\d+) inputs", q["q5"], int),
        "set_distinct_vectors": find(r"distinct vectors among the \d+ inputs: (\d+)", q["q5"], int),
        "cosine_median": find(r"cosine, random pairs of CSIC vectors: .*?median (-?[\d.]+)", q["q1"]),
        "cosine_above_099": find(r"cosine, random pairs of CSIC vectors: .*?share above 0.99 ([\d.]+)", q["q1"]),
        "components_95": find(r"vectors as they are \(not scaled\): (\d+)", q["q5"], int),
        "components_95_scaled": find(r"this encoder .*?\(128 features\): (\d+)", q["q5"], int),
        "first_component": find(r"vectors as they are \(not scaled\): \d+; first component ([\d.]+)", q["q5"]),
        "entropy_bits": find(r"this encoder, vectors as they are: ([\d.]+) bits", q["q5"]),
        "entropy_bits_scaled": find(r"this encoder, scaled features: ([\d.]+) bits", q["q5"]),
        "participation": find(r"CSIC vectors .*?participation ratio ([\d.]+)", q["q4"]),
        "one_character": find(r"probe pairs, one character: .*?median (-?[\d.]+)", q["q2"]),
        "one_value": find(r"probe pairs, one value: .*?median (-?[\d.]+)", q["q2"]),
        "attack_in_benign": find(r"probe pairs, attack in a benign request: .*?median (-?[\d.]+)", q["q2"]),
        "different_benign": find(r"probe pairs, different benign requests: .*?median (-?[\d.]+)", q["q2"]),
        "two_random": find(r"probe pairs, two random texts: .*?median (-?[\d.]+)", q["q2"]),
        "spearman_jaccard": find(r"random pairs \(n \d+\), 3-gram Jaccard distance against cosine distance: Spearman (-?[\d.]+)", q["q2"]),
        "label_agreement": find(r"label, all \(n \d+\): ([\d.]+)", q["agreement"]),
        "endpoint_agreement_good": find(r"endpoint, good texts only \(n \d+\): ([\d.]+)", q["agreement"]),
        "held_out_neighbours": find(r"ROC AUC of the bad share: ([\d.]+)", q["q3"]),
        "held_out_head": find(r"fitted on the fitting side: ROC AUC on the held-out texts ([\d.]+)", q["q3"]),
        "held_out_first_component": find(r"first principal component of the fitting side alone as the score: ROC AUC on the held-out texts ([\d.]+)", q["q3"]),
        "ablation_label_all": find(r"all 128 components: label ROC AUC ([\d.]+)", q["q5"]),
        "ablation_endpoint_all": find(r"all 128 components: .*?endpoint accuracy ([\d.]+)", q["q5"]),
        "ablation_label_first1": find(r"the first 1 only: label ROC AUC ([\d.]+)", q["q5"]),
        "ablation_endpoint_first1": find(r"the first 1 only: .*?endpoint accuracy ([\d.]+)", q["q5"]),
        "ablation_label_first5": find(r"the first 5 only: label ROC AUC ([\d.]+)", q["q5"]),
        "ablation_endpoint_first5": find(r"the first 5 only: .*?endpoint accuracy ([\d.]+)", q["q5"]),
        "ablation_label_without5": find(r"all but the first 5: label ROC AUC ([\d.]+)", q["q5"]),
        "ablation_endpoint_without5": find(r"all but the first 5: .*?endpoint accuracy ([\d.]+)", q["q5"]),
        "ablation_label_without20": find(r"all but the first 20: label ROC AUC ([\d.]+)", q["q5"]),
        "ablation_endpoint_without20": find(r"all but the first 20: .*?endpoint accuracy ([\d.]+)", q["q5"]),
        "ablation_label_without50": find(r"all but the first 50: label ROC AUC ([\d.]+)", q["q5"]),
        "ablation_endpoint_without50": find(r"all but the first 50: .*?endpoint accuracy ([\d.]+)", q["q5"]),
    }
    return row


def random_row(q: dict[str, str]) -> dict:
    """The random-unit-vector figures the scripts print beside every encoder."""
    agreement = q["q3"][q["q3"].index("agreement with the 11 nearest"):]
    return {
        "cosine_median": find(r"cosine, random pairs of random unit vectors: .*?median (-?[\d.]+)", q["q1"]),
        "components_95": find(r"random unit vectors, same number: components for 90 / 95 / 99 % of the variance: \d+ / (\d+)", q["q4"], int),
        "components_95_scaled": find(r"random unit vectors \(128 features\): (\d+)", q["q5"], int),
        "entropy_bits_scaled": find(r"random unit vectors: ([\d.]+) bits", q["q5"]),
        "participation": find(r"random unit vectors, same number: .*?participation ratio ([\d.]+)", q["q4"]),
        "held_out_neighbours": find(r"ROC AUC of the bad share: .*?random unit vectors ([\d.]+)", q["q3"]),
        "held_out_head": find(r"fitted on the fitting side: ROC AUC on the held-out texts .*?random unit vectors ([\d.]+)", q["q3"]),
        "ablation_label_all": find(r"random unit vectors, all 128 components: label ROC AUC ([\d.]+)", q["q5"]),
        "ablation_endpoint_all": find(r"random unit vectors, all 128 components: .*?endpoint accuracy ([\d.]+)", q["q5"]),
        "label_agreement": find(r"label, all \(n \d+\): .*?random unit vectors ([\d.]+)", agreement),
        "endpoint_agreement_good": find(r"endpoint, good texts only \(n \d+\): .*?random unit vectors ([\d.]+)", agreement),
    }


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lm-seed", type=int, default=0)
    args = parser.parse_args()
    eq.style()
    root: Path = args.root

    rows: list[dict] = []
    published = read_logs(PUBLISHED_LOGS)
    q_published = {name: (PUBLISHED_LOGS / f"{name}.log").read_text(encoding="utf-8") for name in ("q1", "q2", "q3", "q4", "q5")}
    reference = json.loads((root / "reference" / "triage.json").read_text(encoding="utf-8"))
    published.update(name="published encoder", precision="8-bit, NPU", key="published", colour=eq.MUTED,
                     triage=reference["published_npu_vectors"])
    rows.append(published)
    rows.append(dict(random_row(q_published), name="random unit vectors", precision="float", key="random",
                     colour="#c3c2b7", triage=reference["random_unit_vectors"]))
    for directory in sorted((root / "qualification").iterdir()) if (root / "qualification").exists() else []:
        match = re.fullmatch(r"(.+)-s(\d+)-(f32|int8|npu)", directory.name)
        logs = read_logs(directory)
        if not match or logs is None:
            continue
        key, seed, precision = match.group(1), int(match.group(2)), match.group(3)
        label = next((label for k, label, _ in VARIANTS if k == key), key)
        colour = next((c for k, _, c in VARIANTS if k == key), eq.SERIES[7])
        record = json.loads((root / "variants" / f"{key}-s{seed}" / "record.json").read_text(encoding="utf-8"))
        logs.update(name=label, key=key, seed=seed, colour=colour,
                    precision={"f32": "float", "int8": "8-bit", "npu": "8-bit, NPU"}[precision],
                    triage=None if precision == "npu" else record.get("triage_int8" if precision == "int8" else "triage_f32"),
                    float_8bit_cosine=record.get("float_against_8bit_cosine_min_p01_median"))
        rows.append(logs)
    order = {key: i for i, (key, _, _) in enumerate(VARIANTS)}
    precision_order = {"float": 0, "8-bit": 1, "8-bit, NPU": 2}
    rows[2:] = sorted(rows[2:], key=lambda r: (order.get(r["key"], 99), r["seed"], precision_order[r["precision"]]))

    def triage(row, *keys):
        value = row.get("triage")
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value

    first = [
        ("Vectors", lambda r: r["name"] + (f", seed {r['seed']}" if r.get("seed") else "")),
        ("Precision", lambda r: r["precision"]),
    ]
    tables = {
        "Table 1: dimensions, and near and apart": [
            ("Components for 95 % of the variance (features scaled)", lambda r: f"{fmt(r.get('components_95'))} ({fmt(r.get('components_95_scaled'))})"),
            ("Spectrum entropy, bits (scaled)", lambda r: f"{fmt(r.get('entropy_bits'), 2)} ({fmt(r.get('entropy_bits_scaled'), 2)})"),
            ("Random pairs, median cosine", lambda r: fmt(r.get("cosine_median"), 2)),
            ("One character changed", lambda r: fmt(r.get("one_character"), 4)),
            ("One value changed", lambda r: fmt(r.get("one_value"), 3)),
            ("Different benign requests", lambda r: fmt(r.get("different_benign"), 2)),
            ("Text distance against vector distance, Spearman", lambda r: fmt(r.get("spearman_jaccard"), 2)),
            ("Float against 8-bit vector, median cosine", lambda r: fmt((r.get("float_8bit_cosine") or [None] * 3)[2] if r["precision"] == "8-bit" else None, 3)),
        ],
        "Table 2: structure, the held-out endpoints, feature ablation": [
            ("Label shared with the 11 nearest", lambda r: fmt(r.get("label_agreement"))),
            ("Endpoint shared with the 11 nearest (good texts)", lambda r: fmt(r.get("endpoint_agreement_good"))),
            ("Held-out linear head, ROC AUC", lambda r: fmt(r.get("held_out_head"))),
            ("Held-out 11 neighbours, ROC AUC", lambda r: fmt(r.get("held_out_neighbours"))),
            ("Ablation, all 128 components: label ROC AUC / endpoint accuracy", lambda r: f"{fmt(r.get('ablation_label_all'))} / {fmt(r.get('ablation_endpoint_all'))}"),
            ("First component only", lambda r: f"{fmt(r.get('ablation_label_first1'))} / {fmt(r.get('ablation_endpoint_first1'))}"),
            ("Without the first 5", lambda r: f"{fmt(r.get('ablation_label_without5'))} / {fmt(r.get('ablation_endpoint_without5'))}"),
            ("Without the first 20", lambda r: f"{fmt(r.get('ablation_label_without20'))} / {fmt(r.get('ablation_endpoint_without20'))}"),
        ],
        "Table 3: folding, and the triage quality of the science gate": [
            ("Distinct vectors of the 12,078 texts of the analysis set", lambda r: fmt(r.get("set_distinct_vectors"))),
            ("Distinct vectors of all distinct inputs", lambda r: f"{fmt(r.get('all_distinct_vectors'))} of {fmt(r.get('all_inputs'))}"),
            ("Vectors shared by different inputs (inputs that share)", lambda r: f"{fmt(r.get('shared_vectors'))} ({fmt(r.get('inputs_sharing'))})"),
            ("Split of the 1.7 note, all texts: head / neighbours", lambda r: f"{fmt(triage(r, 'split17', 'linear_head', 'roc_auc'))} / {fmt(triage(r, 'split17', 'knn_k11', 'roc_auc'))}"),
            ("Five folds by path, pooled: head / neighbours", lambda r: f"{fmt(triage(r, 'cv5', 'linear_head_pooled'))} / {fmt(triage(r, 'cv5', 'knn_k11_pooled'))}"),
        ],
    }
    lines = []
    for title, columns in tables.items():
        columns = first + columns
        lines += ["", title, "", "| " + " | ".join(name for name, _ in columns) + " |", "|" + "---|" * len(columns)]
        for row in rows:
            lines.append("| " + " | ".join(cell(row) for _, cell in columns) + " |")
    args.out.mkdir(parents=True, exist_ok=True)
    table_dir = root / "report"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (table_dir / "comparison.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"baseline scorer on the split17 held-out endpoints (Python re-implementation): {json.dumps(reference['baseline'])}")

    # --- Picture 1: the variants side by side (8-bit bars, float as a mark) ---
    panels = [
        ("entropy_bits", "Spectrum entropy, bits (7.00 at most)", None),
        ("cosine_median", "Median cosine of random pairs", None),
        ("endpoint_agreement_good", "Endpoint shared with the 11 nearest (good texts)", None),
        ("held_out_head", "Held-out endpoints: linear head, ROC AUC", 0.5),
        ("held_out_neighbours", "Held-out endpoints: 11 neighbours, ROC AUC", 0.5),
        ("inputs_sharing", "Inputs that share a vector (published: of 12,079; others: of 25,608)", None),
    ]
    shown = [r for r in rows if r["precision"] in ("8-bit", "8-bit, NPU") and r.get("seed", 0) == 0 and not (r["precision"] == "8-bit, NPU" and r["key"] != "published")]
    shown.insert(1, dict(rows[1], entropy_bits=rows[1].get("entropy_bits_scaled")))
    float_of = {(r["key"], r.get("seed", 0)): r for r in rows if r["precision"] == "float"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax, (key, title, floor) in zip(axes.flat, panels):
        names = [r["name"] for r in shown]
        values = [r.get(key) if r.get(key) is not None else 0 for r in shown]
        y = np.arange(len(shown))
        ax.barh(y, values, color=[r["colour"] for r in shown], height=0.62)
        marks = [(i, float_of[r["key"], 0].get(key)) for i, r in enumerate(shown)
                 if r["key"] not in ("published", "random") and (r["key"], 0) in float_of and float_of[r["key"], 0].get(key) is not None]
        if marks:
            ax.scatter([v for _, v in marks], [i for i, _ in marks], marker="D", s=26, color=eq.INK, zorder=3, label="float")
        mark_at = dict(marks)
        for i, (r, value) in enumerate(zip(shown, values)):
            text = "not measured" if r.get(key) is None else (f"{value:,}" if isinstance(value, int) else f"{value:.2f}")
            # The label goes right of the bar and of the float mark, whichever is further.
            ax.annotate(text, (max(value, mark_at.get(i, value)), i), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8, color=eq.INK2)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title)
        if floor is not None:
            ax.axvline(floor, color=eq.MUTED, lw=1, ls="--")
        ax.set_xlim(min(0, min(values)) * 1.15, max(values) * 1.22 if max(values) > 0 else 1)
        ax.grid(axis="y", visible=False)
        if marks:
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.09), borderaxespad=0.0)
    fig.suptitle("The variants on the published encoder's analysis set (bars: 8-bit vectors; diamonds: float; grey: references)",
                 fontsize=12, fontweight="bold", color=eq.INK)
    fig.tight_layout()
    eq.save(fig, args.out, "comparison.png")

    # --- Picture 2: where the patchers cut, on example lines ---
    patching = json.loads((root / "features" / f"s{args.lm_seed}" / "patching.json").read_text(encoding="utf-8"))
    examples = patching["examples"]
    for index, example in enumerate(examples):
        text = bytes(example["bytes"]).decode("latin-1")
        n = len(text)
        fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.105 * n + 1.6), 4.3), height_ratios=[1.25, 1.75], sharex=True)
        top, strip = axes
        entropy = np.array(example["entropy"])
        # entropy[t] belongs to the prediction of byte t + 1: draw it at that byte.
        top.step(np.arange(1, n), entropy[: n - 1], where="mid", color=eq.SERIES[2], lw=1.4)
        top.set_ylabel("entropy of the\nnext-byte prediction (nats)")
        top.set_title(f"{example['entry']}: {n} bytes; patches by fixed stride {len(example['starts']['fixed'])}, "
                      f"delimiter {len(example['starts']['delimiter'])}, entropy {len(example['starts']['entropy-jump'])}")
        top.set_ylim(0, max(3.0, float(entropy.max()) * 1.05))
        for start in example["starts"]["entropy-jump"][1:]:
            top.axvline(start - 0.5, color=eq.SERIES[2], lw=0.6, alpha=0.5)
        for level, (key, label, colour) in enumerate(PATCHERS):
            starts = example["starts"][key]
            for patch, start in enumerate(starts):
                end = starts[patch + 1] if patch + 1 < len(starts) else n
                strip.add_patch(plt.Rectangle((start - 0.5, level - 0.42), end - start, 0.84, color=colour,
                                              alpha=0.32 if patch % 2 == 0 else 0.12, linewidth=0))
                strip.plot([start - 0.5, start - 0.5], [level - 0.42, level + 0.42], color=colour, lw=0.9)
            for t, char in enumerate(text):
                shown_char = char if 32 < ord(char) < 127 else ("␣" if char == " " else "·")
                strip.text(t, level, shown_char, ha="center", va="center", fontsize=6.3, family="DejaVu Sans Mono", color=eq.INK)
        strip.set_yticks(range(len(PATCHERS)), [label for _, label, _ in PATCHERS])
        strip.set_ylim(len(PATCHERS) - 0.4, -0.6)
        strip.set_xlim(-1, n)
        strip.set_xlabel("byte position in the line (shading alternates per patch)")
        strip.grid(False)
        fig.tight_layout()
        eq.save(fig, args.out, f"patches-example-{index + 1}-{example['source']}.png")

    # --- Picture 3: bytes per patch and the model's surprise, per source ---
    sources = [("CSIC, training part", "csic", "patches_mean_training"), ("CSIC, held-out endpoints", "csic", "patches_mean_held-out")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    labels = ["CSIC, all lines"] + ["AIT " + entry.split("/")[-1].replace("2022-01-21-", "logstash ") + f" ({entry.split('/')[1]})" for entry in patching["ait_entries"]]
    y = np.arange(len(labels))
    for offset, (key, label, colour) in zip((-0.24, 0.0, 0.24), PATCHERS):
        values = [patching["stats"][f"csic/{key}"]["patch_bytes_mean"]]
        values += [patching["stats"][f"ait/{key}"]["by_entry"][entry]["bytes_per_patch"] for entry in patching["ait_entries"]]
        axes[0].barh(y + offset, values, height=0.22, color=colour, label=label)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean bytes per patch")
    axes[0].set_title("Patch length by source (AIT: up to 2,000 lines per file)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    axes[0].grid(axis="y", visible=False)
    surprise = [patching["entropy_nats_mean"]["csic"]] + [patching["entropy_nats_mean_ait_by_entry"][entry] for entry in patching["ait_entries"]]
    axes[1].barh(y, surprise, height=0.5, color=[eq.SERIES[2]] + [eq.MUTED] * len(patching["ait_entries"]))
    for i, value in enumerate(surprise):
        axes[1].annotate(f"{value:.2f}", (value, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=eq.INK2)
    axes[1].set_xlabel("mean entropy of the next-byte prediction (nats per byte)")
    axes[1].set_title("How surprised the byte model is (trained on CSIC only)")
    axes[1].grid(axis="y", visible=False)
    axes[1].set_xlim(0, max(surprise) * 1.15)
    fig.tight_layout()
    eq.save(fig, args.out, "patching-by-source.png")

    # --- Picture 4: patches per line, CSIC against the AIT sample ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (key, label, colour) in zip(axes, PATCHERS):
        for source, name, tone in (("csic", "CSIC", colour), ("ait", "AIT sample", eq.MUTED)):
            counts = np.array(patching["stats"][f"{source}/{key}"].get("counts_histogram", []))
            name = f"{name} ({int(counts.sum()):,} lines)"
            if counts.size:
                ax.step(np.arange(counts.size), counts / counts.sum(), where="mid", color=tone, lw=1.6, label=name)
        ax.set_title(label)
        ax.set_xlabel("patches in a line (limit 128)")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("share of the lines")
    fig.tight_layout()
    eq.save(fig, args.out, "patches-per-line.png")


if __name__ == "__main__":
    main()
