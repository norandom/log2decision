"""Prepare the task 2.25 panel: the last fresh held fold-zero records at a low positive share.

Same shape as task 2.24 and disjoint from both earlier panels. The benchmark label is read
for one purpose: to draw the two strata by seeded hash. No score is read.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage224_panel as base  # noqa: E402

PREFIX = "l2d-225-score-with-reserve-v1"
POSITIVE, NEGATIVE = 100, 900
BUDGETS = [12, 25, 50, 100, 200]


def prepare(source, first, second, dest):
    base.PREFIX = PREFIX
    old = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    oldest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    if base.sha(first / "manifest.json") != old["earlier_panel_manifest_sha256"]:
        raise ValueError("Earlier panels do not form one chain")
    for name, expected in old["source_hashes"].items():
        path = source / name
        if path.stat().st_size > 16 * 1024**2 or base.sha(path) != expected:
            raise ValueError(f"Unexpected input {name}")
    for name in base.REUSED:
        if base.sha(second / name) != old["input_hashes"][name]:
            raise ValueError(f"Earlier panel input changed: {name}")
    used = set(old["query_ids"]) | set(old["history_ids"]) | set(oldest["query_ids"]) | set(oldest["history_ids"])
    split = json.loads((source / "f0/split.json").read_bytes())
    held, ids = [], set()
    with (source / "records.jsonl").open(encoding="utf-8") as stream:
        for line, position in zip(stream, split, strict=True):
            raw = json.loads(line)
            if (raw["id"] != position["id"] or raw["i"] != position["i"]
                    or base.digest(raw["text"]) != raw["id"] or raw["id"] in ids or raw["bad"] not in (0, 1)):
                raise ValueError("Source identity or split order mismatch")
            ids.add(raw["id"])
            if position["partition"] == "held" and raw["id"] not in used:
                held.append({"id": raw["id"], "text": raw["text"], "group": raw["group"], "stratum": raw["bad"]})
    if {r["group"] for r in held} & set(old["history_groups"]):
        raise ValueError("Endpoint overlap with history")
    remaining_good = sum(r["stratum"] == 0 for r in held)
    drawn = (base.sample((r for r in held if r["stratum"] == 1), "positive-stratum", POSITIVE)
             + base.sample((r for r in held if r["stratum"] == 0), "negative-stratum", NEGATIVE))
    queries = base.sample(drawn, "queries", len(drawn))
    dest.mkdir(parents=True, exist_ok=False)
    for name in base.REUSED:
        shutil.copyfile(second / name, dest / name)
    with (dest / "queries.csv").open("x", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["text", "endpoint"])
        writer.writerows((r["text"], r["group"]) for r in queries)
    sixth = {str(b): max(1, b // 6) for b in BUDGETS}
    third = {str(b): max(1, b // 3) for b in BUDGETS}
    base.write_json(dest / "manifest.json", {
        "version": PREFIX,
        "scope": "historical CSIC development, held endpoint fold zero, records used by neither earlier panel",
        "source_hashes": old["source_hashes"], "seed_query": old["seed_query"],
        "earlier_panel_manifests_sha256": [base.sha(first / "manifest.json"), base.sha(second / "manifest.json")],
        "history_ids": old["history_ids"], "history_groups": old["history_groups"],
        "query_ids": [r["id"] for r in queries], "query_groups": sorted({r["group"] for r in queries}),
        "input_hashes": {name: base.sha(dest / name)
                         for name in ("history.csv", "queries.csv", "training-marks.json")},
        "training": old["training"],
        "panel": {"records": len(queries), "positive_stratum": POSITIVE, "negative_stratum": NEGATIVE,
                  "fresh_good_records_before_this_draw": remaining_good,
                  "label_use": "The benchmark label selected the two strata by seeded hash. No score was read. "
                               "Decisions are frozen before outcomes are joined for evaluation.",
                  "known_limit": "Nearly all held good records are on one endpoint, so it dominates the panel, and "
                                 "records elsewhere are mostly positive. Every held endpoint is absent from the "
                                 "history by the fold's construction."},
        "budgets": BUDGETS,
        "policy": old["policy"],
        "score_order": "diagnostic MarkSet ratio descending, then original SHA256",
        "arms": {
            "score": "the first b records in score order",
            "candidate": "Reservation of n slots for records the product routed as Unsupported, taken in score "
                         "order. The remaining b - n slots, and any reservation slots left empty, are filled in "
                         "score order from the records not yet taken. Always b records.",
            "control": "As the candidate, with the reservation filled without any encoder: records ordered by the "
                       "number of panel records on their endpoint ascending, then score order.",
            "policy_separate": "the product policy with its own unsupported slots, as in task 2.24",
            "reservation_sixth": sixth, "reservation_third": third,
            "policy_separate_slots": {b: {"unsupported_slots": n, "unknown_slots": n, "audit_slots": n}
                                      for b, n in sixth.items()},
        },
        "allocation": "One product run at budget 12 under the shared rule supplies routes and evidence. The policy "
                      "arm at each budget is recomputed from that evidence and validated by the product.",
        "tests": {
            "primary": "At each budget and for both reservation sizes: positives found by the candidate and not by "
                       "score alone against positives found by score alone and not by the candidate. Exact "
                       "two-sided sign test on these discordant positives. All ten results are reported. Budgets "
                       "are nested.",
            "control": "The same paired comparison between the candidate and the control. If they do not differ, "
                       "the vector evidence is not shown to add anything over the endpoint on this fold.",
            "replication": "The task 2.24 question and statistic with 40 score bins, 9,999 permutations of the "
                           "unsupported flag within bins, seed 225: once on the whole panel and once on the "
                           "records of the endpoint with the most panel records. For the records on other "
                           "endpoints only counts are reported, because they are too few for bins.",
        },
        "selection": "Freeze all selections for every budget before joining held benchmark outcomes.",
        "qualification": "exploratory; no operational targets, no pass/fail threshold, no final holdout",
    })
    print(json.dumps({"fresh_good_before_draw": remaining_good, "queries": len(queries)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run218", type=Path)
    parser.add_argument("first_panel", type=Path, help="Task 2.22 panel directory")
    parser.add_argument("second_panel", type=Path, help="Task 2.24 panel directory")
    parser.add_argument("fresh_output", type=Path)
    args = parser.parse_args()
    prepare(args.run218, args.first_panel, args.second_panel, args.fresh_output)
