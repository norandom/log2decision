"""Checks made after outcomes were joined on the task 2.25 panel. Descriptive, not part of the
frozen plan: what the planned tests could have shown, and the policy against score alone at
equal effort. Arguments: run directory, task 2.18 export."""
import json
import sys
from math import comb
from pathlib import Path

run, source = Path(sys.argv[1]), Path(sys.argv[2])
frozen = json.loads((run / "frozen-decisions.json").read_text(encoding="utf-8"))
routes, order = frozen["routes"], frozen["score_order"]
labels = {}
with (source / "records.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if row["id"] in routes:
            labels[row["id"]] = row["bad"]
rank = {i: n + 1 for n, i in enumerate(order)}
main = max(frozen["endpoint_sizes"], key=lambda g: (frozen["endpoint_sizes"][g], g))
unsupported = [i for i in order if routes[i]["reason"] == "Unsupported"]


def sign(a, b):
    n, k = a + b, max(a, b)
    return None if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)


# 1. What the primary test could have shown.
outside = [i for i in unsupported if rank[i] > 77]
print("unsupported records below rank 77:", len(outside), "positive", sum(labels[i] for i in outside),
      "at ranks", [rank[i] for i in outside if labels[i]])
print("smallest two-sided p for 3 gained, 0 lost:", sign(3, 0), "| for 4 discordant:", sign(4, 0))
for budget, size in ((100, "third"), (200, "sixth"), (200, "third")):
    arms = frozen["selections"][str(budget)]
    candidate, score = set(arms[f"candidate_{size}"]["selected"]), set(arms["score"]["selected"])
    print(budget, size, "swapped in:", len(candidate - score), "positive", sum(labels[i] for i in candidate - score),
          "| displaced:", len(score - candidate), "positive", sum(labels[i] for i in score - candidate))

# 2. What the main-endpoint replication could have shown.
on_main = [i for i in order if routes[i]["group"] == main]
main_rank = {i: n + 1 for n, i in enumerate(on_main)}
size = len(on_main) // 40
for i in unsupported:
    if i in main_rank:
        n = min((main_rank[i] - 1) // size, 39)
        members = on_main[n * size:(n + 1) * size] if n < 39 else on_main[n * size:]
        supported = [m for m in members if routes[m]["reason"] != "Unsupported"]
        print("main-endpoint unsupported: rank", main_rank[i], "positive", labels[i], "| bin", n + 1,
              "supported", len(supported), "positive", sum(labels[m] for m in supported))

# 3. Reservation overlap between candidate and control.
for budget in ("50", "100", "200"):
    for size in ("sixth", "third"):
        arms = frozen["selections"][budget]
        a, b = set(arms[f"candidate_{size}"]["reserved"]), set(arms[f"control_{size}"]["reserved"])
        print("reservation overlap", budget, size, len(a & b), "of", len(a), "and", len(b))

# 4. Policy against score alone at equal effort.
positives = {i for i in order if labels[i]}
for budget in ("12", "25", "50", "100", "200"):
    policy = set(frozen["selections"][budget]["policy_separate"]["selected"])
    score = set(order[:len(policy)])
    a, b = policy & positives, score & positives
    print("equal effort", len(policy), "policy", len(a), "score", len(b), "policy only", len(a - b),
          "score only", len(b - a), "p", sign(len(a - b), len(b - a)))
