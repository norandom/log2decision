"""Checks made after outcomes were joined on the task 2.24 panel. Descriptive, not part of the frozen plan:
endpoint split of unsupported records, expected positives under finer score bins, exact sign tests on
discordant positives, and the equal-effort comparison. Arguments: run directory, task 2.18 export."""
import json, sys
from math import comb
from pathlib import Path

run, source = Path(sys.argv[1]), Path(sys.argv[2])
frozen = json.loads((run / "frozen-decisions.json").read_text(encoding="utf-8"))
comparison = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
routes, order = frozen["routes"], frozen["score_order"]
labels = {}
with (source / "records.jsonl").open(encoding="utf-8") as s:
    for line in s:
        r = json.loads(line)
        if r["id"] in routes:
            labels[r["id"]] = r["bad"]
rank = {i: n + 1 for n, i in enumerate(order)}
main = comparison["main_endpoint"]["name"]
uns = [i for i in order if routes[i]["reason"] == "Unsupported"]
on = [(rank[i], labels[i]) for i in uns if routes[i]["group"] == main]
print("unsupported on main endpoint (rank, positive):", on)
print("top 32 all positive:", all(labels[i] for i in order[:32]))
off = [i for i in order if routes[i]["group"] != main]
print("off main: records", len(off), "positive", sum(labels[i] for i in off),
      "unsupported", sum(routes[i]["reason"] == "Unsupported" for i in off),
      "unsupported positive", sum(labels[i] for i in off if routes[i]["reason"] == "Unsupported"))
print("off-main ranks of positives:", sorted(rank[i] for i in off if labels[i]))


def expected(bins):
    size = len(order) / bins
    obs = exp = 0.0
    for n in range(bins):
        members = order[round(n * size):round((n + 1) * size)]
        sup = [i for i in members if routes[i]["reason"] != "Unsupported"]
        un = [i for i in members if routes[i]["reason"] == "Unsupported"]
        if sup and un:
            obs += sum(labels[i] for i in un)
            exp += len(un) * sum(labels[i] for i in sup) / len(sup)
    return obs, round(exp, 2)


for bins in (10, 20, 40, 100):
    print("bins", bins, "observed, expected:", expected(bins))


def sign(a, b):
    n, k = a + b, max(a, b)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)


for budget, result in comparison["budgets"].items():
    for name in ("policy_shared", "policy_separate"):
        d = result[f"score_vs_{name}"]
        print(budget, name, "score only", d["score_only"], "policy only", d["policy_only"],
              "two-sided p", round(sign(d["score_only"], d["policy_only"]), 3))
    reviewed = result["policy_separate"]["reviewed"]
    print("  score top", reviewed, "holds", sum(labels[i] for i in order[:reviewed]), "policy found",
          result["policy_separate"]["positive"],
          "| off-main positives: score", result["score"]["positive_elsewhere"], "policy",
          result["policy_separate"]["positive_elsewhere"],
          "| unsupported positives missed: score", result["score"]["unsupported_positive_missed"], "policy",
          result["policy_separate"]["unsupported_positive_missed"])
