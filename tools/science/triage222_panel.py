"""Prepare the fixed fold-zero development panel without using held outcomes."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

SOURCE_HASHES = {
    "records.jsonl": "ac850585603e82b9755c8455306337aa142a55fafca6a0af00c26921539b4270",
    "f0/split.json": "a4165de03606258c36a56f9b8bd412bbad34a5b7b897538d1bf844a7c39ff449",
    "plan.json": "23971a68d635c1ce85685a779804b3aeacf38bab02b61cea56b50eb2a239b36e",
}
PREFIX = "l2d-222-foreground-v1"


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


def prepare(source, dest):
    # The source is the previously inspected 2.18 development export. Cap admission
    # before decoding JSON; original corpus sizes are also pinned by the digests.
    for name, expected in SOURCE_HASHES.items():
        path = source / name
        if path.stat().st_size > 16 * 1024**2 or sha(path) != expected:
            raise ValueError(f"Unexpected input {name}")
    split = json.loads((source / "f0/split.json").read_bytes())
    seed = json.loads((source / "plan.json").read_bytes())["seed_query"]
    regex = re.compile(seed)
    train, held = [], []
    ids = set()
    with (source / "records.jsonl").open(encoding="utf-8") as stream:
        for line, position in zip(stream, split, strict=True):
            raw = json.loads(line)
            if (raw["id"] != position["id"] or raw["i"] != position["i"]
                    or digest(raw["text"]) != raw["id"] or raw["id"] in ids):
                raise ValueError("Source identity or split order mismatch")
            ids.add(raw["id"])
            row = {field: raw[field] for field in ("id", "text", "group")}
            partition = position["partition"]
            if partition == "held":
                held.append(row)
            elif partition == "train":
                row["training_label"] = raw["bad"]
                train.append(row)
    if {r["group"] for r in train} & {r["group"] for r in held}:
        raise ValueError("Endpoint overlap")
    bad = sample((r for r in train if r["training_label"] == 1 and regex.search(r["text"])),
                 "seed-bad", 16)
    good = sample((r for r in train if r["training_label"] == 0), "reviewed-good", 16)
    marked = {r["id"] for r in bad + good}
    background = sample((r for r in train if r["id"] not in marked), "background", 32)
    history = sorted(bad + good + background, key=lambda r: r["id"])
    queries = sample(held, "queries", 96)
    dest.mkdir(parents=True, exist_ok=False)
    for name, rows in [("history.csv", history), ("queries.csv", queries)]:
        with (dest / name).open("x", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(["text", "endpoint"])
            writer.writerows((r["text"], r["group"]) for r in rows)
    write_json(dest / "training-marks.json",
               {"bad": [r["text"] for r in bad], "good": [r["text"] for r in good]})
    write_json(dest / "manifest.json", {
        "version": PREFIX, "scope": "historical CSIC development, held endpoint fold zero",
        "source_hashes": SOURCE_HASHES, "seed_query": seed, "history_ids": [r["id"] for r in history],
        "query_ids": [r["id"] for r in queries], "history_groups": sorted({r["group"] for r in history}),
        "query_groups": sorted({r["group"] for r in queries}),
        "input_hashes": {name: sha(dest / name) for name in ("history.csv", "queries.csv", "training-marks.json")},
        "training": "16 seed-matched benchmark bad, 16 benchmark good, 32 unmarked; simulated review",
        "budget": 12, "score_order": "diagnostic MarkSet ratio descending, then original SHA256",
        "pattern_order": "training template marked bad fraction descending, bad support descending, SHA256 tie; unmatched last",
        "policy": {"low": .45, "high": .55, "radius": .05, "top_k": 3, "minimum_labelled_support": 1,
                   "corroboration": 1, "unknown_slots": 2, "audit_slots": 2, "audit_seed": 5,
                   "method": "pinned f0-s0-mask NPU", "require_calibration": False},
        "selection": "Freeze decisions before joining held benchmark outcomes. Compare at offered budget and policy fill.",
        "uncertainty": "Report paired gains/losses and discordant records, one held fold; no across-fold or temporal uncertainty estimate.",
        "qualification": "exploratory; no operational targets or final holdout",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run218", type=Path)
    parser.add_argument("fresh_output", type=Path)
    args = parser.parse_args()
    prepare(args.run218, args.fresh_output)
