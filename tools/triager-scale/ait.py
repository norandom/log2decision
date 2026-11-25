"""The cold public-archive and warm saved-region legs of the triager scale gate.

Triager task 2.2 (`.kiro/specs/triager/performance.md`, "Proposed scale gate"). A
cold server opens every entry of the AIT archive under the reference limit of 512
MB, scores it with the frozen baseline scorer of parent task 1.17 (A, top 10,000),
filters it (B), composes D = A - B, takes a bounded C, counts the regions, allocates
one budget of 24 slots from D and saves D and the final work set as receipts. It
then drills the prepared sets thirty times and more. A second, warm server reopens
both receipts without the archive and drills the saved regions; its work counters
must stay at zero.

C is made first by a server of its own under its own limit (`C_LIMIT`): the lexical
neighbours of an exemplar within the pattern catalogue of a selective filter X, the
service's own producer. The catalogue reads the archive with its deduplication in
memory, which 512 MB does not admit for this archive. C reaches the gate as a
registered membership file; outside X it is not evaluated. The exemplar lies inside
D: the record handles of the receipt of D an earlier run saved (`--d-receipt`) are
read, and that server takes the first of them that X holds, so that C and D overlap.
The cold run checks that they do: some of D recommended, and at least one audited
slot drawn from D outside the candidate D ∩ C.

    uv run --no-project python ait.py --l2d L2D --project P --archive ZIP --out O \\
        --d-receipt EARLIER/receipts/rcpt-....receipt

The reference counts are those of parent tasks 1.16, 1.17 and 1.20 for the same
adapter and scope (all entries, the default record limit). Standard library only;
the client of `run.py` is reused.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from run import Failed, Server, check, sha256_of

REFERENCE = {
    "records_scanned": 30_916_764,
    "entries_read": 11_870,
    "skipped_records": 357_123,
    "occurrences": 30_559_641,
    "distinct_originals": 8_276_854,
}
WORK_COUNTERS = ("archive_scans", "bytes_unpacked", "rows_scored", "scorer_loads",
                 "membership_reads", "rows_encoded", "genai_rows_issued",
                 "genai_rows_submitted")
# A literal of members of D (the SSH moduli lines), from a bounded look at the earlier
# D's texts; the first candidate whose members include one of D's and that has
# patterns and neighbours is X. The DNS tunnel queries of D ("3x6-.") are 268,026
# distinct originals, beyond the bound of 100,000 a bounded C takes. A moduli line is up
# to about 2 KiB (an 8192-bit modulus in hex): the catalogue takes texts of 4 KiB.
X_CANDIDATES = [" 2 6 100 "]
# The separately admitted limit of the C producer: its catalogue reads the archive with
# deduplication in memory, 666 to 672 MiB on this archive (refused under 512 MB).
C_LIMIT = "1536MB"


def budget():
    return {"total": 24, "duplicate_rule": "first_pool_wins", "shortfall": "leave_unfilled",
            "pools": [
                {"name": "general", "limit": 18, "from": {"intersection": ["d", "c"]},
                 "order": ["score_desc", "record_id_ascending"]},
                {"name": "unfamiliar", "limit": 4,
                 "from": {"intersection": ["d", {"state": {"set": "c", "is": "not_evaluated"}}]},
                 "order": ["record_id_ascending"]},
                {"name": "audit", "limit": 2, "from": {"difference": ["d", "c"]},
                 "order": ["seeded_hash", "record_id_ascending"], "seed": 5}]}


def counts(result):
    return {"counts": result.get("counts"), "occurrence_counts": result.get("occurrence_counts")}


def timed(server, steps, name, tool, arguments):
    started = time.perf_counter()
    result = server.ok(tool, arguments, name)
    steps.append({"step": name, "seconds": round(time.perf_counter() - started, 3),
                  "cost": result.get("cost"), **counts(result)})
    print(f"{name}: {steps[-1]['seconds']} s", file=sys.stderr, flush=True)
    return result


def pages(server, handle, sets, limit, most=None):
    rows, cursor, calls = [], None, 0
    while True:
        arguments = {"v": 1, "handle": handle, "view": "records", "limit": limit, "sets": sets}
        if cursor:
            arguments["cursor"] = cursor
        page = server.ok("triager_inspect", arguments, "page")
        calls += 1
        rows.extend(page["rows"])
        cursor = page["page"]["next_cursor"]
        if not cursor or (most and len(rows) >= most):
            return rows, calls


def percentiles(values):
    ordered = sorted(values)
    at = lambda share: ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]
    return {"n": len(ordered), "p50_ms": round(statistics.median(ordered), 3),
            "p95_ms": round(at(0.95), 3), "max_ms": round(ordered[-1], 3)}


def drill(server, label, d, sets, c_handle, snapshot):
    """Thirty and more warm interactions on prepared or reopened sets: the overlap,
    pages forward and back, one record's evidence, the siblings and the overview."""
    first_page, drill_down = [], []
    index_rows = 0
    opening = server.overview(snapshot)
    before, session_before = opening["activity"], opening["session"]

    def call(tool, arguments, into):
        nonlocal index_rows
        result = server.ok(tool, arguments)
        into.append(server.last["ms_to_result"])
        index_rows += (result.get("cost") or {}).get("index_rows_read") or 0
        return result

    for round_ in range(8):
        call("triager_inspect", {"v": 1, "handle": d, "view": "regions", "sets": sets},
             drill_down)
        page = call("triager_inspect", {"v": 1, "handle": d, "view": "records", "limit": 25,
                                        "sets": sets}, first_page)
        cursor = page["page"]["next_cursor"]
        if cursor:
            call("triager_inspect", {"v": 1, "handle": d, "view": "records", "limit": 25,
                                     "sets": sets, "cursor": cursor}, drill_down)
            # Back: the first page again, by starting over.
            again = call("triager_inspect", {"v": 1, "handle": d, "view": "records",
                                             "limit": 25, "sets": sets}, first_page)
            check(again["rows"], page["rows"], f"{label}: the first page again")
        record = page["rows"][round_ % max(1, len(page["rows"]))]["id"]
        call("triager_inspect", {"v": 1, "handle": d, "view": "evidence", "record": record,
                                 "sets": sets}, drill_down)
        for is_ in ("c", {"state": {"set": "c", "is": "not_evaluated"}}):
            call("triager_compose", {"v": 1, "mode": "view", "snapshot": snapshot,
                                     "scope": d, "sets": {"d": d, "c": c_handle},
                                     "expr": {"intersection": ["d", is_]}}, drill_down)
    closing = server.overview(snapshot)
    after, session_after = closing["activity"], closing["session"]
    return {
        "interactions": len(first_page) + len(drill_down),
        "index_rows_read": index_rows,
        "bytes_returned": session_after["bytes_returned"] - session_before["bytes_returned"],
        "calls": session_after["calls"] - session_before["calls"],
        "first_page": percentiles(first_page),
        "drill_down": percentiles(drill_down),
        "work_counters_before": {key: before[key] for key in WORK_COUNTERS},
        "work_counters_after": {key: after[key] for key in WORK_COUNTERS},
        "unchanged": all(before[key] == after[key] for key in WORK_COUNTERS),
    }


def attempt(server, steps, name, tool, arguments):
    """A call that may be refused: its result or its refusal, recorded."""
    started = time.perf_counter()
    payload, error, _ = server.tool(tool, arguments, name)
    while not error and payload.get("status") == "running":
        time.sleep(0.2)
        status, error, _ = server.tool("triager_job", {
            "v": 1, "job": payload["job"], "action": "status"}, "poll")
        if status["status"] == "completed":
            payload = status["result"]
        elif status["status"] not in ("running", "waiting", "cancel_requested"):
            error, payload = True, status
    record = {"step": name, "seconds": round(time.perf_counter() - started, 3)}
    record.update({"refused": payload} if error else
                  {"cost": payload.get("cost"), **counts(payload)})
    steps.append(record)
    print(f"{name}: {record['seconds']} s{' refused' if error else ''}",
          file=sys.stderr, flush=True)
    return None if error else payload


def produce_c(args, report):
    """The bounded C, made by a server of its own under its own limit: the lexical
    neighbours of an exemplar within the pattern catalogue of a selective filter X. The
    catalogue reads the archive with its deduplication in memory, which the reference
    limit of 512 MB does not admit for this archive; C is written as a membership file
    (in near the exemplar, out for the rest of X, abstained where the method cannot
    represent a member) and is not evaluated anywhere else."""
    out = args.out
    # The D of an earlier run of this gate, as its receipt kept it: the same archive,
    # scorer, filter and composition give the same D, so its members choose an
    # exemplar inside D before the cold run makes D again (and checks the overlap).
    # The receipt is read as the file it is: its row lines name the retained originals,
    # the members of the saved view (the receipt format of that run is older than this
    # binary's, which also keeps model judgments).
    earlier_ids = []
    with args.d_receipt.open(encoding="utf-8") as lines:
        header = None
        for number, line in enumerate(lines):
            item = json.loads(line)
            if number == 1:
                header = item
            if "r" in item:
                earlier_ids.append(item["r"][0])
    check(header["retained"]["population"], "the members of the saved view",
          "what the earlier receipt retained")
    check(len(earlier_ids), header["retained"]["distinct_originals"], "the earlier D's rows")
    report["earlier_d"] = {"receipt": args.d_receipt.stem,
                           "receipt_sha256": sha256_of(args.d_receipt),
                           "saved": header["saved"], "members": len(earlier_ids)}
    server = Server(args.l2d, args.project, ["--memory-limit", C_LIMIT], [
        "--source", f"ait={args.archive}", "--receipts", str(out / "receipts-c"),
        "--work-dir", str(out / "work-c"), "--inline-wait-ms", "500",
        "--max-patterns", "64", "--max-text-bytes", "4096"], out / "c-stderr.txt")
    steps = report.setdefault("c_steps", [])
    try:
        opened = timed(server, steps, "c_open", "triager_open",
                       {"v": 1, "source": {"ref": "ait"}})
        snapshot = opened["snapshot"]
        x = c = None
        for query in X_CANDIDATES:
            candidate = timed(server, steps, f"x_filter_{query}", "triager_select",
                              {"v": 1, "scope": snapshot, "producer": {
                                  "kind": "filter", "syntax": "literal", "query": query}})
            if not 2 <= candidate["counts"]["in"] <= 100_000:
                continue
            # An exemplar inside D: the first member of the earlier D that X holds,
            # asked one record at a time (bounded evidence, no excerpt).
            exemplar, asked = None, 0
            for record in earlier_ids:
                asked += 1
                read = server.ok("triager_inspect", {
                    "v": 1, "handle": snapshot, "view": "evidence", "record": record,
                    "sets": {"x": candidate["set"]}}, "x membership")
                if read["membership"]["x"]["state"] == "in":
                    exemplar = record
                    break
            steps.append({"step": f"x_exemplar_in_d_{query}", "asked": asked,
                          "exemplar": exemplar})
            if exemplar is None:
                continue
            refinement = attempt(server, steps, f"x_patterns_{query}", "triager_refine", {
                "v": 1, "set": candidate["set"], "action": {"kind": "patterns"}})
            if refinement is None:
                continue
            # About half of X near the exemplar: in for them, out for the rest of X.
            top_k = min(1000, max(1, candidate["counts"]["in"] // 2))
            c = attempt(server, steps, f"c_neighbours_{query}", "triager_refine", {
                "v": 1, "refinement": refinement["refinement"], "action": {
                    "kind": "neighbours", "exemplar": exemplar, "method": "character_trigram",
                    "within": "all", "top_k": top_k, "radius": 0.6}})
            if c is not None:
                x = candidate
                break
        if c is None:
            raise Failed("no candidate X gave patterns and neighbours")
        rows, _ = pages(server, x["set"], {"c": c["set"]}, 100)
        states = {"in": "in", "out": "out", "not_evaluated": "abstain"}
        lines = [json.dumps({"id": row["id"], "state": states[row["membership"]["c"]]},
                            separators=(",", ":")) for row in rows]
        (out / "c.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        report["c"] = {
            "producer": c["producer"], "coverage": c["coverage"], "counts": c["counts"],
            "occurrence_counts": c["occurrence_counts"], "cost": c["cost"],
            "memory_limit": C_LIMIT, "exemplar": exemplar,
            "exemplar_from": "the first member of the earlier D that X holds",
            "scope": {"filter": x["producer"]["query"], "members": x["counts"]["in"],
                      "meaning": "the members of X's pattern refinement; C is not "
                                 "evaluated outside it"},
            "membership_file": {"lines": len(lines), "bytes": (out / "c.jsonl").stat().st_size,
                                "sha256": sha256_of(out / "c.jsonl")},
            "snapshot": snapshot, "snapshot_revision": opened["revision"],
        }
    finally:
        report["c_peak_working_set_bytes"] = server.close()


def cold(args, report):
    out = args.out
    server = Server(args.l2d, args.project, ["--memory-limit", "512MB"], [
        "--source", f"ait={args.archive}", "--receipts", str(out / "receipts"),
        "--membership", f"c={out / 'c.jsonl'}",
        "--work-dir", str(out / "work-cold"), "--inline-wait-ms", "500"],
        out / "cold-stderr.txt")
    steps = report.setdefault("cold_steps", [])
    started = time.perf_counter()
    try:
        opened = timed(server, steps, "open", "triager_open",
                       {"v": 1, "source": {"ref": "ait"}})
        snapshot = opened["snapshot"]
        found = {
            "records_scanned": opened["completeness"]["records_scanned"],
            "entries_read": opened["completeness"]["entries_read"],
            "skipped_records": opened["completeness"]["skipped_records"],
            "occurrences": opened["counts"]["occurrences"],
            "distinct_originals": opened["counts"]["distinct_originals"],
        }
        report["reference_counts"] = {"expected": REFERENCE, "found": found,
                                      "recovered": found == REFERENCE}
        report["archive_sha256_by_service"] = opened["source"]["content_sha256"]
        report["open_cost"] = opened["cost"]
        report["open_completeness"] = opened["completeness"]
        select = lambda producer: {"v": 1, "scope": snapshot, "producer": producer}
        a = timed(server, steps, "a_score_top_10000", "triager_select", select(
            {"kind": "score", "model": "csic", "top_k": 10_000, "ties": "record_id_ascending"}))
        b = timed(server, steps, "b_filter", "triager_select", select(
            {"kind": "filter", "syntax": "literal", "query": "CRON"}))
        timed(server, steps, "regions_ab", "triager_inspect",
              {"v": 1, "handle": snapshot, "view": "regions",
               "sets": {"a": a["set"], "b": b["set"]}})
        d = timed(server, steps, "d_a_minus_b", "triager_compose",
                  {"v": 1, "mode": "view", "snapshot": snapshot,
                   "sets": {"a": a["set"], "b": b["set"]}, "expr": {"difference": ["a", "b"]}})
        # C: the membership file the separately admitted producer wrote, joined by key.
        c = timed(server, steps, "c_membership", "triager_select", select(
            {"kind": "membership", "ref": "c"}))
        report["c_in_gate"] = {"producer": c["producer"], "coverage": c["coverage"],
                               "counts": c["counts"], "occurrence_counts": c["occurrence_counts"],
                               "cost": c["cost"]}
        sets = {"a": a["set"], "b": b["set"], "c": c["set"]}
        report["regions_abc_in_d"] = timed(server, steps, "regions_abc_in_d", "triager_inspect",
                                           {"v": 1, "handle": d["view"], "view": "regions",
                                            "sets": sets})
        siblings = {}
        for name, is_ in (("recommended", "c"),
                          ("unknown", {"state": {"set": "c", "is": "not_evaluated"}})):
            sibling = timed(server, steps, f"sibling_{name}", "triager_compose",
                            {"v": 1, "mode": "view", "snapshot": snapshot, "scope": d["view"],
                             "sets": {"d": d["view"], "c": c["set"]},
                             "expr": {"intersection": ["d", is_]}})
            siblings[name] = counts(sibling)
        report["siblings"] = siblings
        f = timed(server, steps, "allocate_24", "triager_compose",
                  {"v": 1, "mode": "allocate", "snapshot": snapshot,
                   "sets": {"d": d["view"], "c": c["set"]}, "expr": {"intersection": ["d", "c"]},
                   "budget": budget()})
        report["allocation"] = f["allocation"]
        final_rows, _ = pages(server, f["set"], {}, 100)
        report["final_ids"] = [row["id"] for row in final_rows]
        report["final_slots"] = [row["allocation"] for row in final_rows]
        pools = {pool["name"]: pool for pool in f["allocation"]["pools"]}
        report["overlap"] = {
            "d": d["counts"]["in"],
            "recommended_sibling": siblings["recommended"]["counts"]["in"],
            "unknown_sibling": siblings["unknown"]["counts"]["in"],
            "fill": {name: pool["selected"] for name, pool in pools.items()},
            "limits": {name: pool["limit"] for name, pool in pools.items()},
            "eligible": {name: pool["eligible"] for name, pool in pools.items()},
            "outside_candidate": {name: pool["outside_candidate"]
                                  for name, pool in pools.items()},
            "audited_outside_candidate": [
                slot for slot in report["final_slots"]
                if slot["pool"] == "audit" and slot["outside_candidate"]],
        }
        saved_d = timed(server, steps, "save_d", "triager_save",
                        {"v": 1, "handle": d["view"], "sets": {"c": c["set"]}})
        saved_f = timed(server, steps, "save_f", "triager_save", {"v": 1, "handle": f["set"]})
        report["receipts"] = {"d": saved_d, "f": saved_f}
        for key, saved in (("d", saved_d), ("f", saved_f)):
            path = out / "receipts" / f"{saved['receipt']}.receipt"
            report["receipts"][key]["file"] = {"bytes": path.stat().st_size,
                                               "sha256": sha256_of(path)}
        report["cold_seconds"] = round(time.perf_counter() - started, 2)
        report["warm_on_cold_server"] = drill(server, "prepared", d["view"], sets, c["set"],
                                              snapshot)
        report["cold_activity"] = server.overview(snapshot)["activity"]
        report["handles"] = {"snapshot": snapshot, "a": a["set"], "b": b["set"],
                             "c": c["set"], "d": d["view"], "f": f["set"]}
        overlap = report["overlap"]
        if not (overlap["recommended_sibling"] > 0 and overlap["fill"]["general"] > 0
                and overlap["audited_outside_candidate"]):
            raise Failed(f"C does not overlap D as the gate needs: {json.dumps(overlap)[:600]}")
    finally:
        report["cold_peak_working_set_bytes"] = server.close()
    return report


def warm(args, report):
    out = args.out
    server = Server(args.l2d, args.project, ["--memory-limit", "512MB"], [
        "--receipts", str(out / "receipts"), "--work-dir", str(out / "work-warm"),
        "--inline-wait-ms", "500"], out / "warm-stderr.txt")
    try:
        started = time.perf_counter()
        opened = server.ok("triager_open", {"v": 1, "receipt": report["receipts"]["d"]["receipt"]})
        report["warm_reopen_seconds"] = round(time.perf_counter() - started, 3)
        report["warm_reopen_cost"] = opened["cost"]
        snapshot = opened["snapshot"]
        handle = {set_["saved_as"]: set_["handle"] for set_ in opened["sets"]}
        cold = report["handles"]
        d, c = handle[cold["d"]], handle[cold["c"]]
        sets = {"a": handle[cold["a"]], "b": handle[cold["b"]], "c": c}
        regions = server.ok("triager_inspect", {"v": 1, "handle": d, "view": "regions",
                                                "sets": sets})
        for field in ("regions", "unknown_regions"):
            check(regions[field], report["regions_abc_in_d"][field],
                  f"the reopened regions: {field}")
        report["warm_drill"] = drill(server, "reopened", d, sets, c, snapshot)
        f = server.ok("triager_compose", {"v": 1, "mode": "allocate", "snapshot": snapshot,
                                          "sets": {"d": d, "c": c},
                                          "expr": {"intersection": ["d", "c"]},
                                          "budget": budget()})
        rows, _ = pages(server, f["set"], {}, 100)
        check([row["id"] for row in rows], report["final_ids"],
              "the work set allocated from the reopened D")
        reopened_f = server.ok("triager_open",
                               {"v": 1, "receipt": report["receipts"]["f"]["receipt"]})
        rows, _ = pages(server, reopened_f["set"], {}, 100)
        check([row["id"] for row in rows], report["final_ids"], "the reopened final work set")
        check([row["allocation"] for row in rows], report["final_slots"],
              "the slots of the reopened final work set")
        activity = server.overview(snapshot)["activity"]
        report["warm_activity"] = activity
        check({key: activity[key] for key in WORK_COUNTERS}, {key: 0 for key in WORK_COUNTERS},
              "the warm server did no work on the archive")
    finally:
        report["warm_peak_working_set_bytes"] = server.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2d", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--d-receipt", required=True, type=Path,
                        help="the receipt of D an earlier run of this gate saved")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    report = {
        "gate": "triager task 2.2, cold public archive and warm saved regions",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "l2d": str(args.l2d), "l2d_sha256": sha256_of(args.l2d),
        "harness_sha256": {name: sha256_of(here / name) for name in ("ait.py", "run.py")},
        "archive": args.archive.name, "archive_bytes": args.archive.stat().st_size,
        "passed": False,
    }
    started = time.perf_counter()
    report["archive_sha256"] = sha256_of(args.archive)
    report["archive_hash_seconds"] = round(time.perf_counter() - started, 1)
    try:
        produce_c(args, report)
        cold(args, report)
        warm(args, report)
        check(report["archive_sha256_by_service"], report["archive_sha256"], "archive hash")
        report["passed"] = report["reference_counts"]["recovered"]
    except Failed as error:
        report["failure"] = str(error)
    finally:
        report["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["harness_sha256_at_end"] = {name: sha256_of(here / name)
                                           for name in ("ait.py", "run.py")}
        (args.out / "ait-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in (
        "passed", "failure", "reference_counts", "cold_seconds",
        "cold_peak_working_set_bytes", "warm_peak_working_set_bytes")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
