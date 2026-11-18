"""Freeze native discovery queries on training originals, then rank a label-free panel."""

import argparse
import csv
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def load(path):
    return json.loads(path.read_bytes())


def frame(path, ids, expected_hash):
    if sha(path) != expected_hash:
        raise ValueError(f"Input hash mismatch: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["text", "endpoint"]:
            raise ValueError(f"Expected label-free text,endpoint CSV: {path.name}")
        rows = list(reader)
    by_id = {text_id(row["text"]): row for row in rows}
    if len(by_id) != len(rows) or len(ids) != len(rows) or set(by_id) != set(ids):
        raise ValueError(f"Distinct original identities differ from manifest: {path.name}")
    return by_id


def product(cli, project, *args, allow_variable_structure=False):
    command = [cli, "--project", project, "--json", "--memory-limit", "2GB", *args]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        diagnostic = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        if allow_variable_structure and "variable token structure: provide an explicit regex" in diagnostic:
            return None
        raise RuntimeError(f"Native {args[:2]} failed ({result.returncode}): {diagnostic}")
    return json.loads(result.stdout)


def matches(cli, project, source, regex, expected_ids):
    # Without --name this command previews only. Consume original texts; ignore
    # the command's default mark label. No suggestion or held mark is saved.
    result = product(cli, project, "mark", "--input", source, "--no-labels",
                     "--group-column", "endpoint", "--regex", regex)
    ids = {text_id(row["text"]) for row in result["matches"]}
    if len(ids) != result["count"] or not ids <= expected_ids or result.get("saved"):
        raise ValueError("Native regex preview returned unexpected identities or a saved mark")
    return ids


def freeze(cli, run, evidence, manifest):
    history_path = run / "history.csv"
    hashes = manifest["input_hashes"]
    history = frame(history_path, manifest["history_ids"], hashes["history.csv"])
    mark_path = run / "training-marks.json"
    if sha(mark_path) != hashes[mark_path.name]:
        raise ValueError("Training marks do not match manifest")
    marks = load(mark_path)
    labelled = {state: {text_id(t) for t in marks[state]} for state in ("bad", "good")}
    bad, good = labelled["bad"], labelled["good"]
    if (not bad or bad & good or not (bad | good) <= history.keys() or
            any(len(labelled[s]) != len(marks[s]) for s in labelled)):
        raise ValueError("Training marks must be distinct, disjoint history members")
    seeds = evidence / "training-bad.csv"
    with seeds.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "endpoint"])
        writer.writeheader()
        writer.writerows(history[record_id] for record_id in sorted(bad))

    native = evidence / "native"
    product(cli, evidence, "discover", "search", "--input", seeds, "--no-labels",
            "--group-column", "endpoint", "--regex", "(?s:.*)", "--out", native)
    refinement = native / "refinement.json"
    listing = product(cli, evidence, "discover", "patterns", "--artifact", refinement)
    refined = load(refinement)
    members = {row["member"]["value"]: text_id(row["text"]) for row in refined["rows"]
               if row["member"]["kind"] == "position"}
    if set(members.values()) != bad or len(members) != len(bad):
        raise ValueError("Native discovery did not preserve the exact training originals")
    templates = []
    for pattern in listing["patterns"]:
        index = pattern["id"]
        candidate = {"pattern": index, "template": pattern["template"],
                     "catalogue_member_ids": sorted(text_id(row["text"]) for row in refined["rows"]
                                                    if row["pattern"] == index)}
        preview = product(cli, evidence, "discover", "preview", "--artifact", refinement,
                          "--pattern", str(index), allow_variable_structure=True)
        if preview is None:
            candidate["exclusion"] = "Native variable-token refusal; no manual replacement"
        else:
            regex = preview["regex"]
            matched = matches(cli, evidence, history_path, regex, history.keys())
            preview_ids = {members[m["value"]] for key in ("member_matches", "background_matches")
                           for m in preview[key]}
            if preview_ids != matched & bad:
                raise ValueError("Native preview and full-history regex disagree on marked originals")
            save(evidence / f"training-preview-{index}.json", preview)
            candidate.update(regex=regex, training_match_ids=sorted(matched),
                             bad_ids=sorted(matched & bad), good_ids=sorted(matched & good))
            if not candidate["bad_ids"]:
                candidate["exclusion"] = "Zero matching marked-bad originals"
        templates.append(candidate)

    return {
        "version": 1, "source": "Native Drain-style catalogue on training bad originals",
        "training_marks": "Training-only benchmark simulation; assumed/unmarked excluded from purity",
        "ranking": "Any eligible match first, maximum bad/(bad+good), then bad count, then original SHA-256. Smallest pattern ID resolves explanatory ties only.",
        "manifest_sha256": sha(run / "manifest.json"), "history_sha256": sha(history_path),
        "training_marks_sha256": sha(mark_path), "training_seeds_sha256": sha(seeds),
        "cli_sha256": sha(cli), "script_sha256": sha(Path(__file__)),
        "native_refinement_revision": listing["revision"], "native_settings": refined["catalogue"]["settings"],
        "native_artifact_hashes": {p.relative_to(native).as_posix(): sha(p)
                                   for p in sorted(native.rglob("*")) if p.is_file()},
        "ungrouped_bad_ids": sorted(text_id(r["text"]) for r in refined["rows"] if r["pattern"] is None),
        "templates": templates,
    }


def rank(cli, run, evidence, manifest, frozen, digest):
    queries_path = run / "queries.csv"
    queries = frame(queries_path, manifest["query_ids"], manifest["input_hashes"][queries_path.name])
    if set(queries) & set(manifest["history_ids"]):
        raise ValueError("Query identities overlap training history")
    by_id = {record_id: [] for record_id in queries}
    quality = {}
    for t in frozen["templates"]:
        if "exclusion" in t:
            continue
        index = t["pattern"]
        bad, good = len(t["bad_ids"]), len(t["good_ids"])
        quality[index] = (Fraction(bad, bad + good), bad)
        for record_id in matches(cli, evidence, queries_path, t["regex"], queries.keys()):
            by_id[record_id].append(index)
    decisions = []
    for record_id, candidates in by_id.items():
        best = max(candidates, key=lambda i: (*quality[i], -i)) if candidates else None
        decisions.append(dict(id=record_id, matched=bool(candidates), best_pattern=best,
                              matching_patterns=sorted(candidates), catalogue_sha256=digest))

    def sort_key(d):
        purity, support = quality.get(d["best_pattern"], (0, 0))
        return not d["matched"], -purity, -support, d["id"]

    decisions.sort(key=sort_key)
    for position, decision in enumerate(decisions, 1):
        decision["rank"] = position
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Frozen run input directory")
    parser.add_argument("--cli", required=True, type=Path, help="Natively compiled l2d executable")
    parser.add_argument("--output", type=Path, help="New evidence and decisions parent (default: run)")
    args = parser.parse_args()
    run, cli = args.run.resolve(strict=True), args.cli.resolve(strict=True)
    output = (args.output or run).resolve()
    manifest = load(run / "manifest.json")
    if (output / "pattern-decisions.json").exists():
        raise FileExistsError("Use a new output directory to replay an existing result")
    evidence = output / "pattern-evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    catalogue = evidence / "frozen-catalogue.json"
    save(catalogue, freeze(cli, run, evidence, manifest))
    digest = sha(catalogue)
    save(evidence / "freeze.json", {"catalogue_sha256": digest, "query_matching_started": False})
    decisions = rank(cli, run, evidence, manifest, load(catalogue), digest)
    if sha(catalogue) != digest:
        raise ValueError("Frozen catalogue changed during query matching")
    save(output / "pattern-decisions.json", decisions)
    print(json.dumps(dict(queries=len(decisions), matched=sum(d["matched"] for d in decisions),
                          frozen_catalogue_sha256=digest, decisions_sha256=sha(output / "pattern-decisions.json"))))


if __name__ == "__main__":
    main()
