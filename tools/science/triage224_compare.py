"""Freeze and evaluate the task 2.24 comparison: low positive share, several review budgets.

freeze   Computes every selection for every budget from the product's persisted evidence,
         has the product validate each recomputed policy report, and writes
         frozen-decisions.json. Reads no outcome.
evaluate Joins the held benchmark outcomes and writes comparison.json.
"""
import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

PRIORITY = {"HighCorroborated": 0, "Disagreement": 1, "Unsupported": 2, "Uncertain": 3, "LowCorroborated": 4}
SEPARATE = "reserved-unsupported-and-conflict-no-reassignment-general-priority-fnv1a-audit-v2"
SHARED = "reserved-no-reassignment-general-priority-fnv1a-audit-v1"
FNV, PRIME, MASK = 0xCBF29CE484222325, 0x100000001B3, (1 << 64) - 1
PERMUTATIONS, SEED, BINS = 9999, 224, 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def publish(path, value, indent=2):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=indent, allow_nan=False)
        stream.write("\n")


def audit_key(identity, seed):
    value = FNV ^ seed
    for byte in identity.encode():
        value = ((value ^ byte) * PRIME) & MASK
    return value


def allocate(rows, budget, conflict, audit, seed, unsupported=None):
    """The product's allocation, restated. The product validates every result."""
    chosen = {}
    ratio = lambda r: r["actionable"] if r["actionable"] is not None else -1.0
    order = sorted(rows, key=lambda r: (PRIORITY[r["reason"]], -ratio(r), r["product_id"]))
    if unsupported is not None:
        for r in [r for r in order if r["reason"] == "Unsupported"][:unsupported]:
            chosen[r["product_id"]] = "Unsupported"
    eligible = {"Disagreement"} if unsupported is not None else {"Unsupported", "Disagreement"}
    for r in [r for r in order if r["reason"] in eligible and r["product_id"] not in chosen][:conflict]:
        chosen[r["product_id"]] = "UnknownConflict"
    general = budget - conflict - audit - (unsupported or 0)
    for r in [r for r in order if r["product_id"] not in chosen and r["queue"] != "LowPriority"][:general]:
        chosen[r["product_id"]] = "General"
    low = sorted((r for r in rows if r["queue"] == "LowPriority"),
                 key=lambda r: (audit_key(r["product_id"], seed), r["product_id"]))
    for r in low[:audit]:
        chosen[r["product_id"]] = "LowPriorityAudit"
    return chosen


def recomputed_report(report, chosen, budget, conflict, audit, unsupported):
    out = json.loads(json.dumps(report))
    out["policy"].update(review_budget=budget, unknown_slots=conflict, audit_slots=audit)
    out["policy"].pop("unsupported_slots", None)
    out.pop("unsupported", None)
    count = lambda configured, pool: {
        "configured": configured, "selected": sum(v == pool for v in chosen.values()),
        "unused": configured - sum(v == pool for v in chosen.values())}
    if unsupported is None:
        out["allocation_rule"] = SHARED
    else:
        out["policy"]["unsupported_slots"] = unsupported
        out["allocation_rule"] = SEPARATE
        out["unsupported"] = count(unsupported, "Unsupported")
    for decision in out["decisions"]:
        decision["allocation"] = chosen.get(decision["input"]["id"])
    out["unknown"] = count(conflict, "UnknownConflict")
    out["general"] = count(budget - conflict - audit - (unsupported or 0), "General")
    out["audit"] = count(audit, "LowPriorityAudit")
    out["deferred"] = len(out["decisions"]) - len(chosen)
    return out


def evidence_dir(run):
    found = sorted(run.glob("policy-cli-*"))
    require(len(found) == 1, "Expected exactly one policy evidence directory")
    return found[0]


def load_rows(run, manifest):
    evidence = evidence_dir(run)
    report = read_json(evidence / "policy-report.json")
    rows = [{"product_id": d["input"]["id"], "id": digest(d["input"]["text"]), "group": d["input"]["group"],
             "diagnostic": d["score"]["diagnostic_ratio"], "actionable": d["score"]["actionable_ratio"],
             "queue": d["queue"], "reason": d["reason"], "allocation": d["allocation"]} for d in report["decisions"]]
    require({r["id"] for r in rows} == set(manifest["query_ids"]) and len(rows) == len(manifest["query_ids"]),
            "Policy report and panel differ")
    return evidence, report, rows


def freeze(run, cli):
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["input_hashes"].items():
        require(sha(run / name) == expected, f"Frozen input differs: {name}")
    evidence, report, rows = load_rows(run, manifest)
    policy = report["policy"]
    seed = policy["audit_seed"]
    persisted = {r["product_id"]: r["allocation"] for r in rows if r["allocation"]}
    require(allocate(rows, policy["review_budget"], policy["unknown_slots"], policy["audit_slots"], seed) == persisted
            and "unsupported_slots" not in policy, "Restated allocation does not reproduce the product's")
    patterns = read_json(run / "pattern-decisions.json")
    require({p["id"] for p in patterns} == {r["id"] for r in rows}, "Pattern and policy panels differ")
    to_panel = {r["product_id"]: r["id"] for r in rows}
    score_order = [r["id"] for r in sorted(rows, key=lambda r: (-r["diagnostic"], r["id"]))]
    pattern_order = [p["id"] for p in sorted(patterns, key=lambda p: p["rank"])]
    work = run / "recomputed-reports"
    work.mkdir(exist_ok=False)
    selections, validated = {}, []
    for budget in manifest["budgets"]:
        key = str(budget)
        shared = manifest["reservations"]["shared_rule"][key]
        separate = manifest["reservations"]["separate_rule"][key]
        arms = {"score": score_order[:budget], "patterns": pattern_order[:budget]}
        for name, slots in (("policy_shared", shared), ("policy_separate", separate)):
            unsupported = slots.get("unsupported_slots")
            chosen = allocate(rows, budget, slots["unknown_slots"], slots["audit_slots"], seed, unsupported)
            path = work / f"{name}-{budget}.json"
            publish(path, recomputed_report(report, chosen, budget, slots["unknown_slots"], slots["audit_slots"],
                                            unsupported), indent=None)
            result = subprocess.run([str(cli), "--project", str(work), "--json", "tui", "--report", str(path)],
                                    capture_output=True, check=False)
            require(result.returncode == 0, f"Product refused {path.name}: {result.stderr.decode(errors='replace')}")
            validated.append({"report": path.name, "sha256": sha(path), "product_exit": 0})
            arms[name] = {"selected": sorted(to_panel[i] for i in chosen),
                          "pools": {to_panel[i]: pool for i, pool in sorted(chosen.items())}, "slots": slots}
        selections[key] = arms
    # Negative control: the product must refuse a recomputed report with one allocation moved.
    control = read_json(work / f"policy_separate-{manifest['budgets'][0]}.json")
    moved = next(d for d in control["decisions"] if d["allocation"] == "Unsupported")
    moved["allocation"] = "UnknownConflict"
    publish(work / "control-tampered.json", control, indent=None)
    refused = subprocess.run([str(cli), "--project", str(work), "--json", "tui", "--report",
                              str(work / "control-tampered.json")], capture_output=True, check=False)
    require(refused.returncode != 0, "Product accepted a tampered allocation")
    (work / "control-tampered.stderr").write_bytes(refused.stderr)
    publish(run / "frozen-decisions.json", {
        "manifest_sha256": sha(run / "manifest.json"), "policy_report_sha256": sha(evidence / "policy-report.json"),
        "pattern_decisions_sha256": sha(run / "pattern-decisions.json"), "cli_sha256": sha(cli),
        "panel": sorted(r["id"] for r in rows), "score_order": score_order, "pattern_order": pattern_order,
        "routes": {r["id"]: {"queue": r["queue"], "reason": r["reason"], "supported": r["actionable"] is not None,
                             "score": r["diagnostic"], "group": r["group"]} for r in rows},
        "selections": selections, "product_validation": validated, "tampered_control_refused": True,
        "outcomes_joined": False,
    })
    print(json.dumps({"frozen": sha(run / "frozen-decisions.json"), "validated_reports": len(validated)}))


def ranking_metrics(labels, order):
    positives = sum(labels.values())
    negatives = len(labels) - positives
    tp = fp = 0
    below = ap = 0.0
    for i in order:
        if labels[i]:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
            below += tp
    return {"roc_auc": below / (positives * negatives), "average_precision": ap / positives}


def question(frozen, labels):
    order, routes = frozen["score_order"], frozen["routes"]
    size = len(order) // BINS
    bins = [order[n * size:(n + 1) * size] if n < BINS - 1 else order[n * size:] for n in range(BINS)]
    table, usable = [], []
    for n, members in enumerate(bins):
        supported = [i for i in members if routes[i]["supported"]]
        unsupported = [i for i in members if not routes[i]["supported"]]
        row = {"bin": n + 1, "ranks": [n * size + 1, n * size + len(members)],
               "supported": len(supported), "supported_positive": sum(labels[i] for i in supported),
               "unsupported": len(unsupported), "unsupported_positive": sum(labels[i] for i in unsupported)}
        row["used"] = bool(supported and unsupported)
        table.append(row)
        if row["used"]:
            usable.append(members)

    def statistic(flags):
        observed = expected = 0.0
        for members in usable:
            marked = [i for i in members if flags[i]]
            rest = [i for i in members if not flags[i]]
            observed += sum(labels[i] for i in marked)
            expected += len(marked) * sum(labels[i] for i in rest) / len(rest)
        return observed, expected

    flags = {i: not routes[i]["supported"] for i in order}
    observed, expected = statistic(flags)
    rng = random.Random(SEED)
    at_least = 0
    for _ in range(PERMUTATIONS):
        shuffled = {}
        for members in usable:
            values = [flags[i] for i in members]
            rng.shuffle(values)
            shuffled.update(zip(members, values))
        o, e = statistic(shuffled)
        at_least += (o - e) >= (observed - expected) - 1e-12
    counted = sum(row["unsupported"] for row in table if row["used"])
    return {"bins": table, "unsupported_in_used_bins": counted,
            "unsupported_outside_used_bins": sum(row["unsupported"] for row in table if not row["used"]),
            "observed_positive": observed, "expected_positive_from_supported_rate": expected,
            "excess": observed - expected,
            "observed_rate": observed / counted if counted else None,
            "expected_rate": expected / counted if counted else None,
            "one_sided_p": (1 + at_least) / (1 + PERMUTATIONS), "permutations": PERMUTATIONS, "seed": SEED}


def evaluate(run, source):
    frozen = read_json(run / "frozen-decisions.json")
    manifest = read_json(run / "manifest.json")
    require(frozen["manifest_sha256"] == sha(run / "manifest.json")
            and frozen["pattern_decisions_sha256"] == sha(run / "pattern-decisions.json")
            and frozen["policy_report_sha256"] == sha(evidence_dir(run) / "policy-report.json"),
            "Decisions differ from frozen artifacts")
    label_file = source / "records.jsonl"
    require(sha(label_file) == manifest["source_hashes"]["records.jsonl"], "Outcome artifact changed")
    expected = set(frozen["panel"])
    labels = {}
    with label_file.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in expected:
                require(row["id"] not in labels and row["bad"] in (0, 1) and row["fold"] == 0
                        and digest(row["text"]) == row["id"], "Invalid outcome identity")
                labels[row["id"]] = row["bad"]
    require(set(labels) == expected, "Missing outcomes")
    positives = {i for i in expected if labels[i]}
    require(len(positives) == manifest["panel"]["positive_stratum"], "Positive stratum differs from the plan")
    routes = frozen["routes"]
    unsupported = {i for i in expected if routes[i]["reason"] == "Unsupported"}
    main_group = Counter(routes[i]["group"] for i in expected).most_common(1)[0][0]

    def arm(ids, capacity):
        ids = set(ids)
        hits = ids & positives
        return {"offered": capacity, "reviewed": len(ids), "positive": len(hits),
                "precision": len(hits) / len(ids) if ids else None, "recall": len(hits) / len(positives),
                "missed": len(positives - ids),
                "unsupported_reviewed": len(ids & unsupported),
                "unsupported_positive_reviewed": len(hits & unsupported),
                "unsupported_positive_missed": len((positives & unsupported) - ids),
                "positive_on_main_endpoint": sum(routes[i]["group"] == main_group for i in hits),
                "positive_elsewhere": sum(routes[i]["group"] != main_group for i in hits)}

    budgets = {}
    for key, arms in frozen["selections"].items():
        budget = int(key)
        result = {"score": arm(arms["score"], budget), "patterns": arm(arms["patterns"], budget)}
        for name in ("policy_shared", "policy_separate"):
            result[name] = arm(arms[name]["selected"], budget)
            pools = Counter(arms[name]["pools"].values())
            result[name]["pools"] = {pool: {"selected": n, "positive": sum(
                labels[i] for i, p in arms[name]["pools"].items() if p == pool)} for pool, n in sorted(pools.items())}
        score = set(arms["score"]) & positives
        for name in ("policy_shared", "policy_separate"):
            policy = set(arms[name]["selected"]) & positives
            result[f"score_vs_{name}"] = {"positive_both": len(score & policy), "score_only": len(score - policy),
                                          "policy_only": len(policy - score)}
        budgets[key] = result
    by_reason = {}
    for i in expected:
        entry = by_reason.setdefault(routes[i]["reason"], {"records": 0, "positive": 0})
        entry["records"] += 1
        entry["positive"] += labels[i]
    order = frozen["score_order"]
    first_negative = next(n + 1 for n, i in enumerate(order) if not labels[i])
    publish(run / "comparison.json", {
        "frozen_decisions_sha256": sha(run / "frozen-decisions.json"), "panel": len(expected),
        "positive": len(positives), "positive_share": len(positives) / len(expected),
        "main_endpoint": {"name": main_group, "records": sum(routes[i]["group"] == main_group for i in expected),
                          "positive": sum(routes[i]["group"] == main_group for i in positives)},
        "score_ranking": ranking_metrics(labels, order) | {"first_negative_rank": first_negative},
        "routes": dict(sorted(by_reason.items())),
        "policy_supported": len(expected - unsupported),
        "budgets": budgets, "question": question(frozen, labels),
        "qualification": manifest["qualification"],
    })
    print(json.dumps({"comparison": sha(run / "comparison.json")}))


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
