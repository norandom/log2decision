"""Summarize repeated runs of the triager scale gate as ranges.

Reads the `report.json` of every `leg-NAME-RUN` and `native-RUN` directory below
`--out` (written by `run.py`) and prints, per leg, the lowest and highest value of
each figure over the runs. Timing on a laptop varies with its power management, so a
figure is a range, never one number.

    python summarize.py --out D:\\BuildArtifacts\\log2decision\\triager-1.3\\scale

Standard library only.
"""

import argparse
import json
import sys
from pathlib import Path

MIB = 1024 * 1024


def spread(values, unit="", scale=1.0, digits=1):
    values = [value / scale for value in values if value is not None]
    if not values:
        return "not available"
    low, high = min(values), max(values)
    if low == high:
        return f"{low:.{digits}f}{unit}"
    return f"{low:.{digits}f} to {high:.{digits}f}{unit}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = {}

    natives = [json.loads(path.read_text()) for path in sorted(args.out.glob("native-*/report.json"))]
    if natives:
        summary["native"] = {"runs": len(natives)}
        for step in ("score", "search", "search_1gb"):
            found = [run[step] for run in natives if step in run]
            summary["native"][step] = {
                "exit_codes": sorted({run["exit_code"] for run in found}),
                "seconds": spread([run["seconds"] for run in found], " s", digits=2),
                "peak_working_set": spread(
                    [run["peak_working_set_bytes"] for run in found], " MiB", MIB),
            }

    legs = {}
    for path in sorted(args.out.glob("leg-*/report.json")):
        report = json.loads(path.read_text())
        legs.setdefault(report["leg"], []).append(report)
    for name, runs in legs.items():
        passed = [run for run in runs if run["passed"]]
        leg = summary[name] = {
            "runs": len(runs), "passed": len(passed),
            "failures": [run.get("failure") for run in runs if not run["passed"]],
            "options": runs[0]["global"] + runs[0]["mcp"],
            "l2d_sha256": sorted({run.get("l2d_sha256") for run in runs}),
            "harness_sha256": runs[0].get("harness_sha256"),
            "harness_same_in_every_run": all(
                run.get("harness_sha256") == runs[0].get("harness_sha256")
                and run.get("harness_unchanged") for run in runs),
            "oracle_sha256": sorted({run.get("oracle_sha256") for run in runs}),
            "started_ended": [[run.get("started"), run.get("ended")] for run in runs],
        }
        if not passed:
            continue
        leg["cold_seconds_open_to_last_view"] = spread(
            [run["cold_seconds"] for run in passed], " s", digits=2)
        leg["cold_ms"] = {
            step: spread([run["cold_ms"].get(step) for run in passed], " ms", digits=0)
            for step in passed[0]["cold_ms"] if step != "compose again"
        }
        leg["server_peak_working_set"] = spread(
            [run["server_peak_working_set_bytes"] for run in passed], " MiB", MIB)
        costs = {}
        for step in ("open", "select a", "select b", "select c", "compose d", "compose m",
                     "select a_threshold", "select a_in_b", "select c_in_b"):
            found = [run["costs"][step] for run in passed]
            first = found[0]
            costs[step] = {
                "archive_scans": first["archive_scans"],
                "bytes_unpacked": first["bytes_unpacked"],
                "rows_scored": first["rows_scored"],
                "reused": (first["reused"] or {}).get("kind"),
                "tables_on_disk": first["tables_on_disk"],
                "dedup_on_disk": first["dedup_on_disk"],
                "spilled_bytes": sorted({cost["spilled_bytes"] for cost in found}),
                "disk_bytes": first["disk_bytes"],
                "disk_peak_bytes": sorted({cost["disk_peak_bytes"] for cost in found}),
                "server_counted_peak": spread([cost["server_counted_peak_bytes"] for cost in found], " MiB", MIB),
                "server_process_peak": spread([cost["server_process_peak_bytes"] for cost in found], " MiB", MIB),
                "response_bytes": first["response_bytes"],
            }
            if step == "open":
                costs[step]["index_on_disk"] = first["index_on_disk"]
                costs[step]["entries_at_once"] = first["entries_at_once"]
            if "read_again_for_disk" in first:
                costs[step]["read_again_for_disk"] = sorted(
                    {cost["read_again_for_disk"] for cost in found})
        leg["costs"] = costs
        leg["memory_limit_bytes"] = passed[0]["costs"]["open"]["memory_limit_bytes"]
        warm = [run["warm"] for run in passed]
        leg["warm"] = {
            "interactions_per_run": warm[0]["interactions"],
            "all_p50": spread([w["all"]["p50_ms"] for w in warm], " ms", digits=2),
            "all_p95": spread([w["all"]["p95_ms"] for w in warm], " ms", digits=2),
            "all_max": spread([w["all"]["max_ms"] for w in warm], " ms", digits=2),
            "by_kind": {
                kind: {"n_per_run": warm[0]["by_kind"][kind]["n"],
                       "p50": spread([w["by_kind"][kind]["p50_ms"] for w in warm], " ms", digits=2),
                       "p95": spread([w["by_kind"][kind]["p95_ms"] for w in warm], " ms", digits=2)}
                for kind in warm[0]["by_kind"]
            },
            "bytes_returned": sorted({w["bytes_returned"] for w in warm}),
            "work_counters_unchanged": all(
                w["work_counters_before"] == w["work_counters_after"] for w in warm),
        }
        leg["paging"] = {key: value["pages"] for key, value in passed[0]["paging"].items()}
        leg["regions"] = passed[0]["regions"]
        leg["refused"] = sorted({f"{item['tool']}: {item['code']}" for item in passed[0]["refused"]})
        leg["refused_count"] = len(passed[0]["refused"])
        leg["excerpts"] = [
            {"record": excerpt["record"],
             "originals_read": excerpt["cost"]["distinct_originals_read"],
             "ms_to_result": spread([run["excerpts"][at]["ms_to_result"] for run in passed],
                                    " ms", digits=0),
             "estimate": excerpt["estimate"]}
            for at, excerpt in enumerate(passed[0]["excerpts"])
        ]
        leg["activity_at_end"] = passed[0]["activity"]
        leg["session_at_end"] = {
            "calls": sorted({run["session"]["calls"] for run in passed}),
            "bytes_returned": sorted({run["session"]["bytes_returned"] for run in passed}),
        }
        leg["tables"] = passed[0]["tables"]
        leg["work_dir_left_after_exit"] = sorted({len(run["work_dir_left"]) for run in passed})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
