"""Prepare the task 2.24 panel: fresh held fold-zero records at a low positive share.

The benchmark label is read here for one purpose: to draw the two strata by seeded hash.
No score is read. History, marks and policy settings are those of the task 2.22 panel.
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

PREFIX = "l2d-224-low-share-v1"
POSITIVE, NEGATIVE = 100, 900
BUDGETS = [12, 25, 50, 100, 200]
REUSED = ("history.csv", "training-marks.json")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def sample(rows, name, count):
    ordered = sorted(rows, key=lambda row: digest(f"{PREFIX}:{name}:{row['id']}"))
    if len(ordered) < count:
        raise ValueError(f"{name}: need {count}, found {len(ordered)}")
    return ordered[:count]


def prepare(source, earlier, dest):
    old = json.loads((earlier / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in old["source_hashes"].items():
        path = source / name
        if path.stat().st_size > 16 * 1024**2 or sha(path) != expected:
            raise ValueError(f"Unexpected input {name}")
    for name in REUSED:
        if sha(earlier / name) != old["input_hashes"][name]:
            raise ValueError(f"Earlier panel input changed: {name}")
    used = set(old["query_ids"]) | set(old["history_ids"])
    split = json.loads((source / "f0/split.json").read_bytes())
    held, ids = [], set()
    with (source / "records.jsonl").open(encoding="utf-8") as stream:
        for line, position in zip(stream, split, strict=True):
            raw = json.loads(line)
            if (raw["id"] != position["id"] or raw["i"] != position["i"]
                    or digest(raw["text"]) != raw["id"] or raw["id"] in ids or raw["bad"] not in (0, 1)):
                raise ValueError("Source identity or split order mismatch")
            ids.add(raw["id"])
            if position["partition"] == "held" and raw["id"] not in used:
                held.append({"id": raw["id"], "text": raw["text"], "group": raw["group"], "stratum": raw["bad"]})
    if {r["group"] for r in held} & set(old["history_groups"]):
        raise ValueError("Endpoint overlap with history")
    drawn = (sample((r for r in held if r["stratum"] == 1), "positive-stratum", POSITIVE)
             + sample((r for r in held if r["stratum"] == 0), "negative-stratum", NEGATIVE))
    # File order is a second hash, so position carries nothing about the stratum.
    queries = sample(drawn, "queries", len(drawn))
    dest.mkdir(parents=True, exist_ok=False)
    for name in REUSED:
        shutil.copyfile(earlier / name, dest / name)
    with (dest / "queries.csv").open("x", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["text", "endpoint"])
        writer.writerows((r["text"], r["group"]) for r in queries)
    sixth = {b: max(1, b // 6) for b in BUDGETS}
    write_json(dest / "manifest.json", {
        "version": PREFIX,
        "scope": "historical CSIC development, held endpoint fold zero, fresh records only",
        "source_hashes": old["source_hashes"], "seed_query": old["seed_query"],
        "earlier_panel_manifest_sha256": sha(earlier / "manifest.json"),
        "history_ids": old["history_ids"], "history_groups": old["history_groups"],
        "query_ids": [r["id"] for r in queries], "query_groups": sorted({r["group"] for r in queries}),
        "input_hashes": {name: sha(dest / name) for name in ("history.csv", "queries.csv", "training-marks.json")},
        "training": old["training"],
        "panel": {"records": len(queries), "positive_stratum": POSITIVE, "negative_stratum": NEGATIVE,
                  "excluded": "the 96 queries and 64 history records of the task 2.22 panel",
                  "label_use": "The benchmark label selected the two strata by seeded hash. No score was read. "
                               "Decisions are frozen before outcomes are joined for evaluation.",
                  "known_limit": "Nearly all held good records are on one endpoint, so it dominates the panel."},
        "budgets": BUDGETS,
        "reservations": {
            "shared_rule": {str(b): {"unknown_slots": n, "audit_slots": n} for b, n in sixth.items()},
            "separate_rule": {str(b): {"unsupported_slots": n, "unknown_slots": n, "audit_slots": n}
                              for b, n in sixth.items()},
            "basis": "one sixth of the budget each, rounded down, at least one; as 2 of 12 on the earlier panel",
        },
        "score_order": old["score_order"], "pattern_order": old["pattern_order"],
        "policy": old["policy"],
        "allocation": "One product run at budget 12 under the shared rule supplies routes and evidence. Other budgets "
                      "and the separate rule are recomputed from that evidence and each recomputed report is "
                      "validated by the product.",
        "question": {
            "text": "Are unsupported records positive at a higher rate than their score rank predicts?",
            "method": "Rank all records by diagnostic ratio descending, then original SHA-256. Cut the ranking into "
                      "ten equal bins. Statistic: positives among unsupported records minus the sum over bins of "
                      "(unsupported in bin) x (positive rate of supported records in that bin). Bins without "
                      "supported records are reported and excluded. One-sided p-value from 9,999 permutations of "
                      "the unsupported flag within bins, seed 224.",
        },
        "selection": "Freeze all selections for every budget before joining held benchmark outcomes.",
        "qualification": "exploratory; no operational targets, no pass/fail threshold, no final holdout",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run218", type=Path)
    parser.add_argument("earlier_panel", type=Path)
    parser.add_argument("fresh_output", type=Path)
    args = parser.parse_args()
    prepare(args.run218, args.earlier_panel, args.fresh_output)
