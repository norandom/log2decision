"""What-if for task 2.23 on the frozen 96-record panel. Mirrors the allocation rule on the
persisted policy decisions. A diagnostic, not a product report: no evidence is recomputed."""
import hashlib, json, sys
from pathlib import Path

run, source, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
frozen = json.loads((run / "frozen-decisions.json").read_text(encoding="utf-8"))
manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
report_file = next(run.glob("policy-cli-*/policy-report.json"))
report = json.loads(report_file.read_text(encoding="utf-8"))
digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
# Product IDs bind the query path; the panel uses the text digest. Keep both.
rows = [{"id": d["input"]["id"], "panel": digest(d["input"]["text"]), "score": d["score"]["actionable_ratio"],
         "supported": d["score"]["actionable_ratio"] is not None, "queue": d["queue"], "reason": d["reason"],
         "allocation": d["allocation"]} for d in report["decisions"]]
panel = set(frozen["panel"])
label_file = source / "records.jsonl"
assert hashlib.sha256(label_file.read_bytes()).hexdigest() == manifest["source_hashes"]["records.jsonl"]
labels = {}
with label_file.open(encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if row["id"] in panel:
            labels[row["id"]] = row["bad"]
assert set(labels) == panel == {r["panel"] for r in rows}
labels = {r["id"]: labels[r["panel"]] for r in rows}
to_panel = {r["id"]: r["panel"] for r in rows}
panel = set(labels)

PRIORITY = {"HighCorroborated": 0, "Disagreement": 1, "Unsupported": 2, "Uncertain": 3, "LowCorroborated": 4}
FNV, PRIME, MASK = 0xCBF29CE484222325, 0x100000001B3, (1 << 64) - 1


def audit_key(i, seed):
    h = FNV ^ seed
    for b in i.encode():
        h = ((h ^ b) * PRIME) & MASK
    return h


def allocate(budget, conflict, audit, seed, unsupported=None):
    chosen = {}
    ratio = lambda r: r["score"] if r["supported"] else -1.0
    order = sorted(rows, key=lambda r: (PRIORITY[r["reason"]], -ratio(r), r["id"]))
    if unsupported is not None:
        for r in [r for r in order if r["reason"] == "Unsupported"][:unsupported]:
            chosen[r["id"]] = "Unsupported"
    eligible = {"Disagreement"} if unsupported is not None else {"Unsupported", "Disagreement"}
    for r in [r for r in order if r["reason"] in eligible and r["id"] not in chosen][:conflict]:
        chosen[r["id"]] = "UnknownConflict"
    general = budget - conflict - audit - (unsupported or 0)
    for r in [r for r in order if r["id"] not in chosen and r["queue"] != "LowPriority"][:general]:
        chosen[r["id"]] = "General"
    low = sorted((r for r in rows if r["queue"] == "LowPriority"), key=lambda r: (audit_key(r["id"], seed), r["id"]))
    for r in low[:audit]:
        chosen[r["id"]] = "LowPriorityAudit"
    return chosen


def summary(chosen):
    by = {}
    reason = {r["id"]: r["reason"] for r in rows}
    for i, pool in chosen.items():
        entry = by.setdefault(pool, {"selected": 0, "positive": 0})
        entry["selected"] += 1
        entry["positive"] += labels[i]
    unsupported = [i for i in panel if reason[i] == "Unsupported"]
    return {"reviewed": len(chosen), "positive": sum(labels[i] for i in chosen), "pools": dict(sorted(by.items())),
            "unsupported_reviewed": sum(i in chosen for i in unsupported),
            "unsupported_deferred": sum(i not in chosen for i in unsupported)}


shared = allocate(12, 2, 2, 5)
saved = {r["id"]: r["allocation"] for r in rows if r["allocation"]}
assert shared == saved, "mirror does not reproduce the persisted allocation"
assert {to_panel[i] for i in shared} == set(frozen["policy_selected"])
policy = report["policy"]
assert (policy["review_budget"], policy["unknown_slots"], policy["audit_slots"], policy["audit_seed"]) == (12, 2, 2, 5)
result = {"mirror_reproduces_persisted_allocation": True, "shared_v1": summary(shared), "separate_v2": {}}
for n in (1, 2, 4):
    separate = allocate(12, 2, 2, 5, n)
    s = summary(separate)
    left = set(shared) - set(separate)
    s["entered"] = sorted(to_panel[i] for i in set(separate) - set(shared))
    s["left"] = sorted(to_panel[i] for i in left)
    s["left_reasons"] = sorted({r["reason"] for r in rows if r["id"] in left})
    s["left_positive"] = sum(labels[i] for i in left)
    # A report in the product's saved form, for the product's own validation.
    v2 = json.loads(json.dumps(report))
    v2["policy"]["unsupported_slots"] = n
    v2["allocation_rule"] = "reserved-unsupported-and-conflict-no-reassignment-general-priority-fnv1a-audit-v2"
    for d in v2["decisions"]:
        d["allocation"] = separate.get(d["input"]["id"])
    count = lambda configured, pool: {"configured": configured, "selected": sum(v == pool for v in separate.values()),
                                      "unused": configured - sum(v == pool for v in separate.values())}
    v2["unsupported"] = count(n, "Unsupported")
    v2["unknown"] = count(2, "UnknownConflict")
    v2["general"] = count(12 - 4 - n, "General")
    v2["audit"] = count(2, "LowPriorityAudit")
    v2["deferred"] = len(rows) - len(separate)
    (out.parent / f"policy-report-unsupported-{n}.json").write_text(json.dumps(v2), encoding="utf-8")
    result["separate_v2"][f"unsupported_slots={n}"] = s
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: (v if k != "separate_v2" else {a: {x: y for x, y in b.items() if x not in ("entered", "left")} for a, b in v.items()}) for k, v in result.items()}, indent=1))
