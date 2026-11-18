"""Regressions for comparison units, equal review counts, ties and artifact freezing."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triage222_compare import evaluate, frozen_document, ranking_metrics
from triage222_panel import digest, sha, write_json


class ComparisonTests(unittest.TestCase):
    def test_equal_scores_get_half_credit_for_tied_pairs(self):
        metrics = ranking_metrics({"a": 1, "b": 0, "c": 1}, {"a": 0.8, "b": 0.8, "c": 0.1})
        self.assertEqual(metrics["roc_auc"], 0.25)
        self.assertAlmostEqual(metrics["average_precision"], (0.5 + 2 / 3) / 2)

    def test_reserved_review_and_deferred_unsupported_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [digest(t) for t in ("x", "y", "z")]
            with (root / "records.jsonl").open("w", encoding="utf-8") as out:
                for t, i, bad in zip(("x", "y", "z"), ids, (1, 1, 0)):
                    out.write(json.dumps(dict(id=i, text=t, fold=0, group="endpoint", bad=bad)) + "\n")
            write_json(root / "manifest.json", {"input_hashes": {}, "query_ids": ids, "budget": 2,
                                                "seed_query": "seed", "source_hashes": {
                                                    "records.jsonl": sha(root / "records.jsonl")}})
            write_json(root / "pattern-decisions.json",
                       [dict(id=i, rank=n, matched=False) for n, i in enumerate(ids, 1)])
            write_json(root / "policy-decisions.json", [
                dict(id=i, score=s, selected=n == 0, supported=n == 2, allocation="UnknownConflict"
                     if n == 0 else None, reason="Unsupported" if n < 2 else "Uncertain")
                for n, (i, s) in enumerate(zip(ids, (0.2, 0.5, 0.9)))])
            write_json(root / "frozen-decisions.json", frozen_document(root))
            r = evaluate(root, root)
            self.assertEqual((r["unsupported_positives_selected"], r["unsupported_positives_missed"]), (1, 1))
            self.assertEqual(r["comparisons"]["offered_capacity"]["arm"]["score"]["reviewed"], 2)
            self.assertEqual(r["comparisons"]["matched_review_count"]["arm"]["score"]["reviewed"], 1)
            self.assertEqual(r["comparisons"]["offered_capacity"]["arm"]["policy"]["missed"], 1)
            with (root / "policy-decisions.json").open("a") as out:
                out.write("\n")
            with self.assertRaisesRegex(ValueError, "differ from frozen"):
                evaluate(root, root)


if __name__ == "__main__":
    unittest.main()
