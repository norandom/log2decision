"""Freeze all three sets of development decisions, then join CSIC outcomes in a separate step."""

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

from triage222_panel import digest, sha, write_json


def read_json(path):
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError(f"File exceeds comparison reader limit: {path}")
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_rows(rows, expected):
    mapped = {r["id"]: r for r in rows}
    require(len(mapped) == len(rows) and set(mapped) == expected, "Decision panel differs")
    return mapped


def frozen_document(run):
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["input_hashes"].items():
        require(sha(run / name) == expected, f"Input changed: {name}")
    expected = set(manifest["query_ids"])
    policy_path, pattern_path = run / "policy-decisions.json", run / "pattern-decisions.json"
    policy = index_rows(read_json(policy_path), expected)
    patterns = index_rows(read_json(pattern_path), expected)
    require(all(math.isfinite(r["score"]) and 0 <= r["score"] <= 1 for r in policy.values()),
            "Invalid diagnostic score")
    for row in policy.values():
        require(type(row["supported"]) is bool and type(row["selected"]) is bool
                and row["selected"] == (row["allocation"] is not None), "Invalid allocation")
    require(all(type(r["matched"]) is bool for r in patterns.values()), "Invalid match flag")
    score_order = sorted(expected, key=lambda i: (-policy[i]["score"], i))
    pattern_order = sorted(expected, key=lambda i: patterns[i]["rank"])
    ranks = [patterns[i]["rank"] for i in pattern_order]
    require(ranks == list(range(1, len(ranks) + 1)), "Pattern ranks must be unique and 1-based")
    policy_ids = sorted(i for i in expected if policy[i]["selected"])
    budget = manifest["budget"]
    require(len(policy_ids) <= budget, "Policy exceeded review capacity")
    return {
        "manifest_sha256": sha(run / "manifest.json"),
        "decision_hashes": {p.name: sha(p) for p in (policy_path, pattern_path)},
        "capacity": budget,
        "panel": sorted(expected),
        "policy_selected": policy_ids,
        "score_order": score_order,
        "pattern_order": pattern_order,
    }


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def ranking_metrics(labels, scores):
    # Aggregate equal ratios at the same threshold for AP and half-credit ROC ties.
    counts = Counter(labels.values())
    if not counts[1] or not counts[0]:
        return {"roc_auc": None, "average_precision": None, "reason": "single-class panel"}
    order = sorted(labels, key=lambda i: -scores[i])
    tp = fp = 0
    auc_twice = ap = 0.
    for _, group in itertools.groupby(order, key=scores.__getitem__):
        chunk = [labels[i] for i in group]
        positive = sum(chunk)
        negative = len(chunk) - positive
        auc_twice += negative * (2 * tp + positive)
        tp += positive
        fp += negative
        ap += positive * tp / (tp + fp)
    return {"roc_auc": auc_twice / (2 * counts[0] * counts[1]),
            "average_precision": ap / counts[1]}


def evaluate(run, source):
    saved = read_json(run / "frozen-decisions.json")
    require(saved == frozen_document(run), "Decisions differ from frozen artifacts")
    manifest = read_json(run / "manifest.json")
    expected = set(saved["panel"])
    label_file = source / "records.jsonl"
    require(sha(label_file) == manifest["source_hashes"]["records.jsonl"], "Outcome artifact changed")
    seed = re.compile(manifest["seed_query"])
    labels, groups, outside = {}, {}, set()
    with label_file.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            i = row["id"]
            if i not in expected:
                continue
            require(i not in labels and row["bad"] in (0, 1) and row["fold"] == 0
                    and digest(row["text"]) == i, "Invalid outcome identity")
            labels[i], groups[i] = row["bad"], row["group"]
            if not seed.search(row["text"]):
                outside.add(i)
    require(set(labels) == expected, "Missing outcomes")
    positives = {i for i in expected if labels[i]}
    policy = index_rows(read_json(run / "policy-decisions.json"), expected)
    patterns = index_rows(read_json(run / "pattern-decisions.json"), expected)
    unsupported = {i for i in expected if not policy[i]["supported"]}
    audit = {i for i in expected if policy[i]["allocation"] == "LowPriorityAudit"}
    selected_policy = set(saved["policy_selected"])

    def measurements(selected):
        hits = selected & positives
        missed = positives - selected
        return {"reviewed": len(selected), "hits": len(hits), "missed": len(missed),
                "precision": fraction(len(hits), len(selected)), "recall": fraction(len(hits), len(positives)),
                "hits_beyond_seed": len(hits & outside),
                "missed_beyond_seed": len(missed & outside)}

    comparisons = {}
    for name, cap in [("offered_capacity", saved["capacity"]), ("matched_review_count", len(selected_policy))]:
        chosen = {"score": set(saved["score_order"][:cap]),
                  "patterns": set(saved["pattern_order"][:cap]), "policy": selected_policy}
        pairs = {}
        for left, right in itertools.combinations(chosen, 2):
            a, b = chosen[left] & positives, chosen[right] & positives
            pairs[f"{left}_vs_{right}"] = {"positive_both": len(a & b), "positive_left_only": len(a - b),
                                           "positive_right_only": len(b - a),
                                           "left_only_positive_ids": sorted(a - b),
                                           "right_only_positive_ids": sorted(b - a),
                                           "per_endpoint_net_hits": {
                                               g: sum(groups[i] == g for i in a - b) - sum(groups[i] == g for i in b - a)
                                               for g in sorted(set(groups.values()))}}
        comparisons[name] = {"arm": {arm: measurements(ids) for arm, ids in chosen.items()}, "pairwise": pairs}
    endpoint_breakdown = []
    for group in sorted(set(groups.values())):
        members = {i for i in expected if groups[i] == group}
        chosen = {"score": saved["score_order"][:saved["capacity"]],
                  "patterns": saved["pattern_order"][:saved["capacity"]], "policy": selected_policy}
        endpoint_breakdown.append({"endpoint": group, "records": len(members), "positive": len(members & positives),
                                   "arms": {arm: {"reviewed": len(members.intersection(ids)),
                                                  "hits": len((members & positives).intersection(ids))}
                                            for arm, ids in chosen.items()}})
    report = {
        "frozen_decisions_sha256": sha(run / "frozen-decisions.json"), "panel": len(expected),
        "positive": len(positives), "positive_beyond_seed": len(positives & outside),
        "endpoint_groups": len(set(groups.values())), "seed_matches": len(expected - outside),
        "score_ranking": ranking_metrics(labels, {i: policy[i]["score"] for i in expected}),
        "pattern_matched": sum(r["matched"] for r in patterns.values()),
        "policy_supported": len(expected - unsupported), "unsupported_positives": len(unsupported & positives),
        "unsupported_positives_selected": len(unsupported & positives & selected_policy),
        "unsupported_positives_missed": len((unsupported & positives) - selected_policy),
        "audit_reviews": len(audit), "audit_positives": len(audit & positives),
        "allocations": Counter(policy[i]["allocation"] for i in selected_policy),
        "routes": Counter(r["reason"] for r in policy.values()),
        "comparisons": comparisons, "endpoint_breakdown": endpoint_breakdown,
    }
    write_json(run / "comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("run", type=Path)
    parser.add_argument("--run218", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        write_json(args.run / "frozen-decisions.json", frozen_document(args.run))
    elif args.run218 is None:
        parser.error("evaluate requires --run218")
    else:
        evaluate(args.run, args.run218)
