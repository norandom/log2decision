"""Check the native terminal against the decisions in a retained l2d policy report."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from run import Workbench


def inspect(binary, *args):
    child = subprocess.run([binary, "--json", "tui", *args], capture_output=True,
                           check=True, timeout=20, encoding="utf-8")
    return json.loads(child.stdout)


def verify(binary, report, run):
    if report.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("the native retained-report limit is 16 MiB")
    raw = report.read_bytes()
    decisions = json.loads(raw)["decisions"]
    expected = {r["input"]["id"] for r in decisions if r["queue"] in {"Investigate", "Review"}}
    digest = hashlib.sha256(raw).hexdigest()
    default = inspect(binary, "--report", report)
    assert default["source_revision"] == digest
    assert default["regions"]["unknown"] == {"originals": len(decisions), "occurrences": None}
    if default["counts"]["originals"] != len(decisions):
        raise AssertionError("the parent report and native root disagree on the population")

    query = "c=recommended | search score>=0.5"
    matching = sorted(r["input"]["id"] for r in decisions
                      if r["input"]["id"] in expected and r["score"]["diagnostic_ratio"] >= 0.5)
    search = inspect(binary, "--report", report, "--policy-recommendations", "--query", query)
    assert search["counts"] == {"originals": len(matching), "occurrences": None}
    assert [r["id"] for r in search["preview"]] == matching[:16]

    receipt = run / "policy-view.json"
    args = [str(binary), "tui", "--report", str(report), "--session", str(receipt)]
    with Workbench([*args, "--policy-recommendations"], show_screen_on_error=False) as app:
        app.expect("occurrences unavailable")
        app.drive("h", "Occurrence locations are unavailable")
        app.drive("i", "parent #0")
        if expected:
            first_id = min(expected)
            first_text = next(row["input"]["text"] for row in decisions if row["input"]["id"] == first_id)
            app.drive("r", f"Original 1 of {len(expected)}")
            app.expect("Raw logs")
            # Compare displayed source text directly when its prefix requires no terminal escaping.
            prefix = first_text[:32]
            if prefix.isascii() and prefix.isprintable() and prefix.strip():
                app.expect(prefix)
            app.drive("b", "Regions & fields")
        app.drive("s", "Saved")
        app.quit()
    # A second process recovers the chosen producer and scoped population from the receipt.
    resumed = inspect(binary, "--report", report, "--session", receipt)
    assert resumed["counts"] == {"originals": len(expected), "occurrences": None}
    assert {row["id"] for row in resumed["preview"]} <= expected
    assert len(resumed["preview"]) == min(16, len(expected))
    assert resumed["parent"] == 0 and not resumed["allocated"]
    saved = json.loads(receipt.read_bytes())
    assert saved["counts"]["occurrences"] is None
    # Native v1 frames each ID, pool, occurrence count and projection terminator.
    # Build the expected digest directly from the parent decisions, including rows off screen.
    ids_digest = hashlib.sha256()
    for identity in sorted(expected):
        encoded = identity.encode("utf-8")
        ids_digest.update(len(encoded).to_bytes(8, "little") + encoded)
        ids_digest.update(bytes(9) + b"\xff" * 8)
    assert saved["result_sha256"] == ids_digest.hexdigest()

    with tempfile.TemporaryDirectory(dir=run) as scratch:
        changed = Path(scratch) / "changed-report.json"
        changed.write_bytes(raw + b"\n")
        refusal = subprocess.run([binary, "--json", "tui", "--report", changed, "--session", receipt],
                                 capture_output=True, timeout=20)
        assert refusal.returncode != 0, "a receipt reopened against revised report bytes"
        assert b"different retained dataset" in refusal.stdout + refusal.stderr
    return {"report_sha256": digest, "distinct": len(decisions), "policy_c_in": len(expected),
            "policy_c_out": len(decisions) - len(expected), "occurrences": None,
            "search_recommended_score_gte_half": len(matching),
            "reopen_counted_peak_bytes": resumed["memory"]["counted_peak_bytes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(dir=args.evidence, prefix="policy-pty-"))
    result = verify(args.binary.resolve(strict=True), args.report.resolve(strict=True), run)
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Passed report PTY replay. Evidence at {run}")
