"""Freeze and evaluate the task 2.25 comparison: score order with a reservation for
unsupported records, against score alone and an encoder-free control.

freeze   Computes every selection for every budget from the product's persisted evidence and
         writes frozen-decisions.json. Reads no outcome.
evaluate Joins the held benchmark outcomes and writes comparison.json.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage224_compare as base  # noqa: E402

BINS, SEED = 40, 225
SIZES = ("sixth", "third")


def reserved(order, front, budget, slots):
    """Take up to `slots` records from the front list, then fill to `budget` in score order."""
    taken = front[:slots]
    chosen = set(taken)
    rest = [i for i in order if i not in chosen][:budget - len(taken)]
    return {"reserved": taken, "selected": sorted(taken + rest)}


def freeze(run, cli):
    manifest = base.read_json(run / "manifest.json")
    for name, expected in manifest["input_hashes"].items():
        base.require(base.sha(run / name) == expected, f"Frozen input differs: {name}")
    evidence, report, rows = base.load_rows(run, manifest)
    policy = report["policy"]
    seed = policy["audit_seed"]
    persisted = {r["product_id"]: r["allocation"] for r in rows if r["allocation"]}
    base.require(base.allocate(rows, policy["review_budget"], policy["unknown_slots"], policy["audit_slots"], seed)
                 == persisted and "unsupported_slots" not in policy,
                 "Restated allocation does not reproduce the product's")
    to_panel = {r["product_id"]: r["id"] for r in rows}
    by_id = {r["id"]: r for r in rows}
    order = [r["id"] for r in sorted(rows, key=lambda r: (-r["diagnostic"], r["id"]))]
    endpoint_size = Counter(r["group"] for r in rows)
    unsupported_front = [i for i in order if by_id[i]["reason"] == "Unsupported"]
    position = {i: n for n, i in enumerate(order)}
    rare_front = sorted(order, key=lambda i: (endpoint_size[by_id[i]["group"]], position[i]))
    work = run / "recomputed-reports"
    work.mkdir(exist_ok=False)
    selections, validated = {}, []
    for budget in manifest["budgets"]:
        key = str(budget)
        arms = {"score": {"selected": sorted(order[:budget])}}
        for size in SIZES:
            slots = manifest["arms"][f"reservation_{size}"][key]
            arms[f"candidate_{size}"] = reserved(order, unsupported_front, budget, slots)
            arms[f"control_{size}"] = reserved(order, rare_front, budget, slots)
        slots = manifest["arms"]["policy_separate_slots"][key]
        chosen = base.allocate(rows, budget, slots["unknown_slots"], slots["audit_slots"], seed,
                               slots["unsupported_slots"])
        path = work / f"policy_separate-{budget}.json"
        base.publish(path, base.recomputed_report(report, chosen, budget, slots["unknown_slots"],
                                                  slots["audit_slots"], slots["unsupported_slots"]), indent=None)
        result = subprocess.run([str(cli), "--project", str(work), "--json", "tui", "--report", str(path)],
                                capture_output=True, check=False)
        base.require(result.returncode == 0,
                     f"Product refused {path.name}: {result.stderr.decode(errors='replace')}")
        validated.append({"report": path.name, "sha256": base.sha(path), "product_exit": 0})
        arms["policy_separate"] = {"selected": sorted(to_panel[i] for i in chosen)}
        selections[key] = arms
    control = base.read_json(work / f"policy_separate-{manifest['budgets'][0]}.json")
    moved = next(d for d in control["decisions"] if d["allocation"] == "Unsupported")
    moved["allocation"] = "UnknownConflict"
    base.publish(work / "control-tampered.json", control, indent=None)
    refused = subprocess.run([str(cli), "--project", str(work), "--json", "tui", "--report",
                              str(work / "control-tampered.json")], capture_output=True, check=False)
    base.require(refused.returncode != 0, "Product accepted a tampered allocation")
    (work / "control-tampered.stderr").write_bytes(refused.stderr)
    base.publish(run / "frozen-decisions.json", {
        "manifest_sha256": base.sha(run / "manifest.json"),
        "policy_report_sha256": base.sha(evidence / "policy-report.json"), "cli_sha256": base.sha(cli),
        "panel": sorted(r["id"] for r in rows), "score_order": order,
        "endpoint_sizes": dict(sorted(endpoint_size.items())),
        "routes": {r["id"]: {"queue": r["queue"], "reason": r["reason"], "supported": r["actionable"] is not None,
                             "score": r["diagnostic"], "group": r["group"]} for r in rows},
        "selections": selections, "product_validation": validated, "tampered_control_refused": True,
        "outcomes_joined": False,
    })
    print(json.dumps({"frozen": base.sha(run / "frozen-decisions.json"), "validated_reports": len(validated)}))


def sign_test(left, right):
    n, k = left + right, max(left, right)
    if n == 0:
        return None
    return min(1.0, 2 * sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def evaluate(run, source):
    frozen = base.read_json(run / "frozen-decisions.json")
    manifest = base.read_json(run / "manifest.json")
    base.require(frozen["manifest_sha256"] == base.sha(run / "manifest.json")
                 and frozen["policy_report_sha256"] == base.sha(base.evidence_dir(run) / "policy-report.json"),
                 "Decisions differ from frozen artifacts")
    label_file = source / "records.jsonl"
    base.require(base.sha(label_file) == manifest["source_hashes"]["records.jsonl"], "Outcome artifact changed")
    expected = set(frozen["panel"])
    labels = {}
    with label_file.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in expected:
                base.require(row["id"] not in labels and row["bad"] in (0, 1) and row["fold"] == 0
                             and base.digest(row["text"]) == row["id"], "Invalid outcome identity")
                labels[row["id"]] = row["bad"]
    base.require(set(labels) == expected, "Missing outcomes")
    positives = {i for i in expected if labels[i]}
    base.require(len(positives) == manifest["panel"]["positive_stratum"], "Positive stratum differs from the plan")
    routes, order = frozen["routes"], frozen["score_order"]
    unsupported = {i for i in expected if routes[i]["reason"] == "Unsupported"}
    main_group = max(frozen["endpoint_sizes"], key=lambda g: (frozen["endpoint_sizes"][g], g))
    on_main = {i for i in expected if routes[i]["group"] == main_group}

    def arm(selection, budget):
        ids = set(selection["selected"])
        hits = ids & positives
        out = {"offered": budget, "reviewed": len(ids), "positive": len(hits),
               "precision": len(hits) / len(ids) if ids else None,
               "positive_on_main_endpoint": len(hits & on_main), "positive_elsewhere": len(hits - on_main),
               "unsupported_reviewed": len(ids & unsupported),
               "unsupported_positive_missed": len((positives & unsupported) - ids)}
        if "reserved" in selection:
            taken = set(selection["reserved"])
            out["reservation"] = {"filled": len(taken), "positive": len(taken & positives),
                                  "unsupported": len(taken & unsupported), "off_main_endpoint": len(taken - on_main)}
        return out

    def paired(left, right):
        a, b = set(left["selected"]) & positives, set(right["selected"]) & positives
        return {"left_only": len(a - b), "right_only": len(b - a), "both": len(a & b),
                "two_sided_p": sign_test(len(a - b), len(b - a))}

    budgets = {}
    for key, arms in frozen["selections"].items():
        result = {name: arm(selection, int(key)) for name, selection in arms.items()}
        for size in SIZES:
            result[f"candidate_{size}_vs_score"] = paired(arms[f"candidate_{size}"], arms["score"])
            result[f"candidate_{size}_vs_control_{size}"] = paired(arms[f"candidate_{size}"],
                                                                   arms[f"control_{size}"])
            result[f"control_{size}_vs_score"] = paired(arms[f"control_{size}"], arms["score"])
            same = set(arms[f"candidate_{size}"]["reserved"]) & set(arms[f"control_{size}"]["reserved"])
            result[f"reservation_overlap_{size}"] = len(same)
        result["policy_separate_vs_score"] = paired(arms["policy_separate"], arms["score"])
        budgets[key] = result

    base.BINS, base.SEED = BINS, SEED
    whole = base.question(frozen, labels)
    main_only = base.question({"score_order": [i for i in order if i in on_main], "routes": routes}, labels)
    off = expected - on_main
    by_reason = {}
    for i in expected:
        entry = by_reason.setdefault(routes[i]["reason"], {"records": 0, "positive": 0})
        entry["records"] += 1
        entry["positive"] += labels[i]
    base.publish(run / "comparison.json", {
        "frozen_decisions_sha256": base.sha(run / "frozen-decisions.json"), "panel": len(expected),
        "positive": len(positives),
        "main_endpoint": {"name": main_group, "records": len(on_main), "positive": len(on_main & positives)},
        "score_ranking": base.ranking_metrics(labels, order) | {
            "first_negative_rank": next(n + 1 for n, i in enumerate(order) if not labels[i])},
        "routes": dict(sorted(by_reason.items())),
        "budgets": budgets,
        "replication": {
            "bins": BINS, "whole_panel": whole, "main_endpoint": main_only,
            "other_endpoints": {"records": len(off), "positive": len(off & positives),
                                "unsupported": len(off & unsupported),
                                "unsupported_positive": len(off & unsupported & positives),
                                "supported": len(off - unsupported),
                                "supported_positive": len((off - unsupported) & positives)},
        },
        "qualification": manifest["qualification"],
    })
    print(json.dumps({"comparison": base.sha(run / "comparison.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("freeze")
    one.add_argument("run", type=Path)
    one.add_argument("--cli", required=True, type=Path)
    two = commands.add_parser("evaluate")
    two.add_argument("run", type=Path)
    two.add_argument("--run218", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.run.resolve(strict=True), args.cli.resolve(strict=True))
    else:
        evaluate(args.run.resolve(strict=True), args.run218.resolve(strict=True))
