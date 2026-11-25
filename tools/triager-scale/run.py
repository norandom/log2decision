"""Run the legs of the triager scale gate through MCP stdio and check them.

Triager task 1.3. A leg starts `l2d mcp` as a child process, reduces the fixture
archive to A, B and the stand-in for C, composes and pages the reductions, and
compares every count, membership digest, region and ordered record ID with the
oracle (`oracle.py`). It records what each job cost as the service states it, the
time of every call as this client saw it, the bytes it was returned and the peak
working set of the server.

    python run.py native   --l2d L2D --project P --fixture F --out O
    python run.py leg NAME --l2d L2D --project P --fixture F --oracle O/oracle.json --out O

`native` fits the baseline scorer and runs the existing commands the legs are compared
with (`l2d baseline score`, `l2d discover search`). The legs are:

    streamed  the reference limit of 512 MB, tables where the budget admits them
    spill     256 MB, relation, membership and scores on disk, deduplication forced
              to disk, another batch size
    tight     192 MB, placement left to the budget

Standard library only. Windows gives the peak working set; elsewhere it is omitted.
"""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

LEGS = {
    "streamed": (["--memory-limit", "512MB"], []),
    "spill": (["--memory-limit", "256MB", "--batch-rows", "8192"],
              ["--tables-on-disk", "--dedup-set-texts", "200000"]),
    "tight": (["--memory-limit", "192MB", "--batch-rows", "16384"], []),
    # The limit is given with `--limit`: one at which the budget, not an option, puts
    # the tables of a selection on disk (found with the `probe` command).
    "admission": (["--memory-limit", None], []),
}
HARNESS = ("generate.py", "oracle.py", "run.py", "summarize.py", "requirements.txt")


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def harness_hashes() -> dict:
    """The SHA-256 of every file of this harness, as it is on disk now."""
    here = Path(__file__).resolve().parent
    return {name: sha256_of(here / name) for name in HARNESS}
WORK_COUNTERS = ("archive_scans", "bytes_unpacked", "rows_scored", "rows_encoded",
                 "genai_rows_submitted", "membership_reads")


def peak_working_set(process) -> int | None:
    """The peak working set of a child process; it may have ended already."""
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(int(process._handle), ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize


class Failed(Exception):
    pass


def check(found, expected, what):
    if found != expected:
        raise Failed(f"{what}: {json.dumps(found)[:300]} and {json.dumps(expected)[:300]} expected")


class Server:
    """An MCP client over the standard streams of a running `l2d mcp`."""

    def __init__(self, l2d, project, global_args, mcp_args, stderr_path):
        self.stderr = open(stderr_path, "wb")
        self.process = subprocess.Popen(
            [str(l2d), "--project", str(project), *global_args, "mcp", *mcp_args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr)
        self.next = 0
        self.calls = []
        initialized = self.request("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "triager-scale", "version": "1"}})
        check(initialized["result"]["protocolVersion"], "2025-11-25", "protocol version")
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def send(self, message):
        self.process.stdin.write(json.dumps(message).encode() + b"\n")
        self.process.stdin.flush()

    def request(self, method, params):
        self.next += 1
        self.send({"jsonrpc": "2.0", "id": self.next, "method": method, "params": params})
        line = self.process.stdout.readline()
        if not line:
            raise Failed(f"the server ended during {method}")
        self.last_bytes = len(line)
        reply = json.loads(line)
        check(reply.get("id"), self.next, "reply id")
        return reply

    def tool(self, name, arguments, label=None):
        """One tool call: its payload, whether it is an error, and its record."""
        started = time.perf_counter()
        reply = self.request("tools/call", {"name": name, "arguments": arguments})
        elapsed = (time.perf_counter() - started) * 1000
        result = reply["result"]
        payload = result["structuredContent"]
        record = {"tool": name, "label": label or arguments.get("view") or name,
                  "ms": round(elapsed, 3), "line_bytes": self.last_bytes,
                  "error": bool(result.get("isError"))}
        self.calls.append(record)
        return payload, bool(result.get("isError")), record

    def ok(self, name, arguments, label=None):
        """A call that must succeed; a job that outlives the inline wait is polled,
        and the time of the call is the time until its result."""
        started = time.perf_counter()
        payload, error, record = self.tool(name, arguments, label)
        if error:
            raise Failed(f"{name} {json.dumps(arguments)[:200]}: {json.dumps(payload)[:400]}")
        while payload.get("status") == "running":
            time.sleep(0.05)
            status, error, _ = self.tool(
                "triager_job", {"v": 1, "job": payload["job"], "action": "status"}, "poll")
            if error or status["status"] in ("failed", "cancelled", "interrupted"):
                raise Failed(f"{name}: {json.dumps(status)[:400]}")
            if status["status"] == "completed":
                payload = status["result"]
        record["ms_to_result"] = round((time.perf_counter() - started) * 1000, 3)
        self.last = record
        return payload

    def overview(self, handle):
        return self.ok("triager_inspect", {"v": 1, "handle": handle, "view": "overview"},
                       "overview")

    def close(self):
        peak = peak_working_set(self.process)
        self.process.stdin.close()
        code = self.process.wait(timeout=120)
        self.stderr.close()
        if code != 0:
            raise Failed(f"the server ended with code {code}")
        return peak


def tally_of(result):
    return {"counts": {key: result["counts"][key] for key in ("parent", "in", "out", "unknown")},
            "occurrence_counts": {key: result["occurrence_counts"][key]
                                  for key in ("parent", "in", "out", "unknown")},
            "revision": result["revision"]}


def expected_tally(oracle, name):
    found = oracle["sets"][name]
    return {key: found[key] for key in ("counts", "occurrence_counts", "revision")}


def page_through(server, handle, limit, most=None, sets=None, label="page"):
    """The record IDs of `handle` in page order, as a digest with its ends."""
    digest = hashlib.sha256()
    ids, cursor, listed, pages = [], None, 0, 0
    first_cursor = None
    while most is None or listed < most:
        arguments = {"v": 1, "handle": handle, "view": "records", "limit": limit,
                     "projection": ["id", "occurrences", "membership"], "sets": sets or {}}
        if cursor:
            arguments["cursor"] = cursor
        page = server.ok("triager_inspect", arguments, label)
        check(page["cost"]["archive_scans"], 0, "a page reads no source")
        check(page["cost"]["rows_scored"], 0, "a page scores nothing")
        rows = page["rows"]
        if most is not None:
            rows = rows[: most - listed]
        for row in rows:
            digest.update(row["id"].encode() + b"\n")
            ids.append(row["id"])
        listed += len(rows)
        pages += 1
        cursor = page["page"]["next_cursor"]
        first_cursor = first_cursor or cursor
        if not cursor:
            break
    return {"listed": listed, "sha256": digest.hexdigest(), "first": ids[:3],
            "last": ids[-3:], "pages": pages, "first_cursor": first_cursor,
            "total": page["page"]["total"]["distinct_originals"]}


def reduce(server, fixture, oracle, costs):
    """Open, select and compose; every result is compared with the oracle."""
    counts = oracle["counts"]
    opened = server.ok("triager_open", {"v": 1, "source": {
        "ref": "fixture", "entries": oracle["source"]["entries"],
        "max_record_bytes": oracle["source"]["max_record_bytes"]}}, "open")
    costs["open"] = opened["cost"]
    snapshot = opened["snapshot"]
    check(opened["source"]["content_sha256"], oracle["source"]["sha256"], "archive hash")
    check(opened["counts"], {"distinct_originals": counts["distinct_originals"],
                             "occurrences": counts["accepted_occurrences"]}, "snapshot counts")
    completeness = opened["completeness"]
    check({key: completeness[key] for key in (
        "records_scanned", "skipped_records", "entries_read", "entries_unsupported",
        "entries_excluded", "complete")},
        {"records_scanned": counts["records"], "skipped_records": counts["skipped_records"],
         "entries_read": counts["entries_read"],
         "entries_unsupported": counts["entries_unsupported"],
         "entries_excluded": counts["entries_excluded"], "complete": False}, "completeness")
    check(opened["cost"]["bytes_unpacked"], counts["bytes_unpacked_selected"], "bytes unpacked")
    handles = {"snapshot": snapshot}

    def select(name, scope, producer):
        result = server.ok("triager_select", {"v": 1, "scope": handles[scope],
                                              "producer": producer}, f"select {name}")
        check(tally_of(result), expected_tally(oracle, name), f"set {name}")
        check(result["outside_scope"]["distinct_originals"],
              oracle["sets"][name]["outside_scope"][0], f"outside the scope of {name}")
        handles[name] = result["set"]
        costs[f"select {name}"] = result["cost"]
        return result

    def compose(name, scope, sets, expr):
        result = server.ok("triager_compose", {
            "v": 1, "mode": "view", "snapshot": snapshot, "scope": handles[scope],
            "sets": {key: handles[value] for key, value in sets.items()}, "expr": expr},
            f"compose {name}")
        check(tally_of(result), expected_tally(oracle, name), f"view {name}")
        check(result["cost"]["archive_scans"], 0, f"composing {name} reads no source")
        handles[name] = result["view"]
        costs[f"compose {name}"] = result["cost"]
        return result

    rules = oracle["score_rules"]
    model = fixture["model"]
    a = select("a", "snapshot", {"kind": "score", "model": model, "top_k": rules["top_k"],
                                 "ties": "record_id_ascending"})
    # A read that is refused with the tables in memory is made again onto disk; the
    # cost then counts both reads, in every counter.
    reads = 2 if a["cost"]["read_again_for_disk"] else 1
    check(a["cost"]["archive_scans"], reads, "A reads the source once, or again onto disk")
    if not rules["scored"] <= a["cost"]["rows_scored"] <= reads * rules["scored"]:
        raise Failed(f"rows scored for A: {a['cost']['rows_scored']} in {reads} reads")
    scored_by_server = server.overview(snapshot)["activity"]["rows_scored"]
    check(a["cost"]["rows_scored"], scored_by_server, "the cost of A and the server's counter")
    for name in ("b", "b_regex", "b_literal"):
        b = select(name, "snapshot", {"kind": "filter", **oracle["filters"][name]})
        check(b["cost"]["archive_scans"], 2 if b["cost"]["read_again_for_disk"] else 1,
              f"{name} reads the source once, or again onto disk")
        check(b["cost"]["rows_scored"], 0, f"{name} scores nothing")
    c = select("c", "snapshot", {"kind": "membership", "ref": "fixture-c"})
    check(c["cost"]["archive_scans"], 0, "C reads no source")
    check(c["cost"]["membership_bytes_read"], 2 * c["producer"]["bytes"],
          "C reads its file to bind it and again for its lines")
    abstained = oracle["sets"]["c"]["abstained"]
    unknown = oracle["sets"]["c"]["counts"]["unknown"]
    check(c["coverage"]["reasons"], [
        {"reason": "abstained", "distinct_originals": abstained[0], "occurrences": abstained[1]},
        {"reason": "outside_producer_coverage", "distinct_originals": unknown - abstained[0],
         "occurrences": oracle["sets"]["c"]["occurrence_counts"]["unknown"] - abstained[1]}],
        "why C is not evaluated")
    compose("d", "snapshot", {"a": "a", "b": "b"}, {"difference": ["a", "b"]})
    compose("e", "d", {"d": "d", "c": "c"}, {"intersection": ["d", "c"]})
    compose("m", "snapshot", {"a": "a", "b": "b", "c": "c"},
            {"difference": [{"union": ["a", "b"]}, "c"]})
    # Prepared results are reused: another rule of the scorer, the scorer within B,
    # the membership within B. None reads the source or scores a row.
    before = server.overview(snapshot)["activity"]
    for name, scope, producer in (
        ("a_threshold", "snapshot", {"kind": "score", "model": model,
                                     "threshold": rules["threshold"]}),
        ("a_in_b", "b", {"kind": "score", "model": model, "top_k": rules["top_k_within_b"],
                         "ties": "record_id_ascending"}),
        ("c_in_b", "b", {"kind": "membership", "ref": "fixture-c"}),
    ):
        derived = select(name, scope, producer)
        check([derived["cost"][key] for key in ("archive_scans", "rows_scored", "bytes_unpacked")],
              [0, 0, 0], f"{name} is derived from a prepared set")
        check(derived["cost"]["reused"]["set"], handles["c" if name == "c_in_b" else "a"],
              f"what {name} reuses")
        if name == "c_in_b":
            check(derived["cost"]["membership_bytes_read"], derived["producer"]["bytes"],
                  "a derived membership still reads its file once to bind it")
            check(sorted(derived["coverage"]), ["evaluated", "not_evaluated", "reasons", "unit"],
                  "the coverage of a derived membership")
    compose("g", "snapshot", {"t": "a_threshold", "r": "b_regex"}, {"intersection": ["t", "r"]})
    after = server.overview(snapshot)["activity"]
    check({key: after[key] for key in ("archive_scans", "bytes_unpacked", "rows_scored")},
          {key: before[key] for key in ("archive_scans", "bytes_unpacked", "rows_scored")},
          "reuse leaves the work counters unchanged")
    return handles


def regions(server, handles, oracle):
    def as_table(result, names):
        table = {}
        for region in result["regions"] + result["unknown_regions"]:
            key = "/".join(region["membership"][name] for name in names)
            if region["distinct_originals"] or region in result["regions"]:
                table[key] = [region["distinct_originals"], region["occurrences"]]
        return table

    whole = server.ok("triager_inspect", {
        "v": 1, "handle": handles["snapshot"], "view": "regions",
        "sets": {name: handles[name] for name in "abc"}}, "regions")
    expected = dict(oracle["regions"]["snapshot_abc"])
    for a in ("in", "out"):
        for b in ("in", "out"):
            for c in ("in", "out"):
                expected.setdefault(f"{a}/{b}/{c}", [0, 0])
    check(as_table(whole, "abc"), expected, "the regions of A, B and C")
    check(len(whole["regions"]), 8, "eight known regions")
    within = server.ok("triager_inspect", {
        "v": 1, "handle": handles["d"], "view": "regions", "sets": {"c": handles["c"]}},
        "regions")
    expected = dict(oracle["regions"]["d_c"])
    for c in ("in", "out"):
        expected.setdefault(c, [0, 0])
    check(as_table(within, "c"), expected, "the regions of C within D")
    unknown = sum(cell[0] for key, cell in oracle["regions"]["snapshot_abc"].items()
                  if "not_evaluated" in key)
    check(whole["unknown"]["distinct_originals"], unknown, "unknown combinations")
    return {"known": len(whole["regions"]), "unknown_regions": len(whole["unknown_regions"]),
            "unknown_originals": unknown}


def paging(server, handles, oracle, page_sizes):
    """Every member of the reductions in order, at several page sizes."""
    report = {}
    membership = {name: handles[name] for name in "abc"}
    for name in ("a", "d", "e", "m", "a_in_b", "g", "c"):
        expected = oracle["ordered"][name]
        for limit in page_sizes:
            if expected["members"] / limit > 4000:
                continue
            found = page_through(server, handles[name], limit, sets=membership,
                                 label=f"page {limit}")
            check({key: found[key] for key in ("listed", "sha256", "first", "last")},
                  {"listed": expected["members"], "sha256": expected["sha256"],
                   "first": expected["first"], "last": expected["last"]},
                  f"the ordered members of {name} at {limit} rows a page")
            check(found["total"], expected["members"], f"total of {name}")
            report[f"{name}@{limit}"] = {"members": found["listed"], "pages": found["pages"]}
    for name, key in (("b", "b_first"), ("snapshot", "snapshot_first")):
        expected = oracle["ordered"][key] if name == "b" else oracle[key]
        found = page_through(server, handles[name], 100, most=expected["listed"],
                             label="page 100")
        check((found["listed"], found["sha256"]), (expected["listed"], expected["sha256"]),
              f"the first members of {name}")
        report[f"{name}@100 first"] = {"members": found["listed"], "pages": found["pages"]}
        # Back: the cursor of an earlier page gives that page again.
        again = server.ok("triager_inspect", {
            "v": 1, "handle": handles[name], "view": "records", "limit": 100,
            "projection": ["id", "occurrences", "membership"], "sets": {},
            "cursor": found["first_cursor"]}, "page back")
        check(len(again["rows"]), 100, f"going back in {name}")
    return report


def warm(server, handles, oracle):
    """Repeated drilling on prepared sets: timed, and no work counter moves."""
    snapshot = handles["snapshot"]
    before = server.overview(snapshot)
    sets = {name: handles[name] for name in "abc"}
    start = len(server.calls)
    for round_ in range(12):
        server.ok("triager_inspect", {"v": 1, "handle": snapshot, "view": "regions",
                                      "sets": sets}, "regions")
        for name in ("d", "e", "m", "b", "c", "g"):
            first = server.ok("triager_inspect", {
                "v": 1, "handle": handles[name], "view": "records", "limit": 25,
                "sets": sets}, "first page")
            cursor = first["page"]["next_cursor"]
            if cursor:
                server.ok("triager_inspect", {
                    "v": 1, "handle": handles[name], "view": "records", "limit": 25,
                    "sets": sets, "cursor": cursor}, "next page")
        server.ok("triager_compose", {
            "v": 1, "mode": "view", "snapshot": snapshot,
            "sets": {"a": handles["a"], "b": handles["b"]},
            "expr": {"difference": ["a", "b"]}}, "compose again")
        for record in oracle["evidence"]:
            evidence = server.ok("triager_inspect", {
                "v": 1, "handle": handles["d"], "view": "evidence", "record": record["id"],
                "sets": sets}, "evidence")
            explained = {item["set"]: item for item in evidence["explanation"]}
            check(evidence["record"]["occurrences"], record["occurrences"], "occurrences")
            check({name: evidence["membership"][name]["state"] for name in "abc"},
                  record["membership"], "membership of a record")
            check((explained["a"]["rank"], explained["a"]["ranked_among"]),
                  (record["rank"], record["ranked_among"]), "rank of a record")
            check(evidence["cost"]["index_rows_read"],
                  oracle["counts"]["distinct_originals"] + 1,
                  "a rank reads the score of every original")
            if abs(explained["a"]["score"] - record["score"]) > 1e-6 * abs(record["score"]):
                raise Failed(f"score of {record['id']}")
    calls = server.calls[start:]
    after = server.overview(snapshot)
    check({key: after["activity"][key] for key in WORK_COUNTERS},
          {key: before["activity"][key] for key in WORK_COUNTERS},
          "warm drilling leaves the work counters unchanged")
    check(after["activity"]["scorer_loads"], before["activity"]["scorer_loads"],
          "warm drilling loads no scorer")

    def spread(label=None):
        times = sorted(call["ms"] for call in calls if label is None or call["label"] == label)
        if not times:
            return None
        return {"n": len(times), "p50_ms": round(statistics.median(times), 3),
                "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))], 3),
                "max_ms": times[-1]}

    return {
        "interactions": len(calls),
        "all": spread(),
        "by_kind": {label: spread(label) for label in (
            "regions", "first page", "next page", "compose again", "evidence")},
        "bytes_returned": after["session"]["bytes_returned"] - before["session"]["bytes_returned"],
        "calls_counted_by_server": after["session"]["calls"] - before["session"]["calls"],
        "work_counters_before": {key: before["activity"][key] for key in WORK_COUNTERS},
        "work_counters_after": {key: after["activity"][key] for key in WORK_COUNTERS},
    }


def refusals(server, handles):
    """The optional operations this revision refuses, each with its code."""
    snapshot, d = handles["snapshot"], handles["d"]
    asked = [
        ("triager_refine", {"v": 1, "set": handles["b"], "action": {"kind": "patterns"}}),
        ("triager_save", {"v": 1, "handle": d, "idempotency_key": "scale"}),
        ("triager_open", {"v": 1, "receipt": "H1"}),
        ("triager_select", {"v": 1, "scope": snapshot, "producer": {
            "kind": "recommend", "method": "venn", "policy": "V1"}}),
        ("triager_select", {"v": 1, "scope": snapshot, "producer": {
            "kind": "filter", "syntax": "pattern", "query": "P"}}),
        ("triager_compose", {"v": 1, "mode": "allocate", "snapshot": snapshot,
                             "sets": {"d": d}, "expr": "d"}),
        ("triager_compose", {"v": 1, "mode": "view", "snapshot": snapshot, "sets": {"d": d},
                             "expr": {"state": {"set": "d", "is": "not_evaluated"}}}),
        ("triager_inspect", {"v": 1, "handle": d, "view": "histogram"}),
        ("triager_inspect", {"v": 1, "handle": d, "view": "fields"}),
        ("triager_inspect", {"v": 1, "handle": d, "view": "lineage"}),
        ("triager_inspect", {"v": 1, "handle": d, "view": "records",
                             "sort": ["score_desc", "record_id_ascending"]}),
    ]
    before = server.overview(snapshot)
    found = []
    for tool, arguments in asked:
        payload, error, _ = server.tool(tool, arguments, "refused")
        check(error, True, f"{tool} is refused")
        found.append({"tool": tool, "asked": json.dumps(arguments)[:120],
                      "code": payload["error"]["code"],
                      "message": payload["error"]["message"][:160]})
    after = server.overview(snapshot)
    check(after["session"]["unsupported_refusals"] - before["session"]["unsupported_refusals"],
          len(asked), "refusals counted by the server")
    check(after["activity"]["archive_scans"], before["activity"]["archive_scans"],
          "a refusal reads nothing")
    return found


def excerpts(server, handles, oracle):
    """What an excerpt costs: one read of the source up to the record."""
    found = []
    for record in (oracle["evidence"][0], oracle["evidence"][-1]):
        retained = server.ok("triager_inspect", {
            "v": 1, "handle": handles["d"], "view": "evidence", "record": record["id"]},
            "evidence")
        evidence = server.ok("triager_inspect", {
            "v": 1, "handle": handles["d"], "view": "evidence", "record": record["id"],
            "projection": ["id", "excerpt"]}, "excerpt")
        found.append({"record": record["id"], "estimate": retained["excerpt_estimate"],
                      "cost": evidence["cost"], "bytes": evidence["excerpt"]["bytes"],
                      "ms_to_result": server.last["ms_to_result"]})
    return found


def leg(args):
    fixture = {"model": args.model}
    oracle = json.loads(args.oracle.read_text())
    global_args, mcp_args = LEGS[args.name]
    if None in global_args:
        if not args.limit:
            raise SystemExit(f"the leg `{args.name}` takes its memory limit with --limit")
        global_args = [args.limit if item is None else item for item in global_args]
    out = args.out / f"leg-{args.name}-{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    # What ran: the binary and every file of this harness, hashed before the server
    # starts and again when the leg ends. A harness edited meanwhile shows.
    harness = harness_hashes()
    report = {"leg": args.name, "run": args.run, "global": global_args, "mcp": mcp_args,
              "l2d": str(args.l2d), "l2d_sha256": sha256_of(args.l2d),
              "harness_sha256": harness, "oracle_sha256": sha256_of(args.oracle),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "passed": False}
    server = Server(args.l2d, args.project, global_args, [
        "--source", f"fixture={args.fixture / 'fixture.zip'}",
        "--membership", f"fixture-c={args.fixture / 'membership-c.tsv'}",
        "--work-dir", str(work), "--inline-wait-ms", "200", *mcp_args],
        out / "server-stderr.txt")
    costs = {}
    started = time.perf_counter()
    try:
        handles = reduce(server, fixture, oracle, costs)
        report["cold_seconds"] = round(time.perf_counter() - started, 2)
        report["regions"] = regions(server, handles, oracle)
        report["paging"] = paging(server, handles, oracle, args.page_sizes)
        report["warm"] = warm(server, handles, oracle)
        report["refused"] = refusals(server, handles)
        report["excerpts"] = excerpts(server, handles, oracle)
        final = server.overview(handles["snapshot"])
        report["activity"] = final["activity"]
        report["session"] = final["session"]
        report["tables"] = {name: server.overview(handles[name])["tables"]
                            for name in ("a", "b", "c", "d")}
        report["passed"] = True
    except Failed as failure:
        report["failure"] = str(failure)
    finally:
        report["costs"] = costs
        report["cold_ms"] = {call["label"]: call.get("ms_to_result", call["ms"])
                             for call in server.calls
                             if call["label"].startswith(("open", "select", "compose "))}
        report["calls"] = len(server.calls)
        report["line_bytes_returned"] = sum(call["line_bytes"] for call in server.calls)
        try:
            report["server_peak_working_set_bytes"] = server.close()
        except Exception as error:  # noqa: BLE001 - the report says how the server ended
            report["server_end"] = str(error)
        report["work_dir_left"] = sorted(path.name for path in work.iterdir()) if work.exists() else []
        report["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["harness_unchanged"] = harness_hashes() == harness
        if not report["harness_unchanged"]:
            report["passed"] = False
            report["failure"] = "the harness was edited while the leg ran"
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    summary = {key: report.get(key) for key in (
        "leg", "run", "passed", "failure", "cold_seconds", "server_peak_working_set_bytes")}
    print(json.dumps(summary))
    return 0 if report["passed"] else 1


def timed(command, out_path, stdout_path=None):
    started = time.perf_counter()
    stdout = open(stdout_path, "wb") if stdout_path else subprocess.DEVNULL
    with open(out_path, "wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        code = process.wait()
        peak = peak_working_set(process)
    if stdout_path:
        stdout.close()
    return {"command": " ".join(map(str, command[1:])), "exit_code": code,
            "seconds": round(time.perf_counter() - started, 2), "peak_working_set_bytes": peak}


def native(args):
    """The existing command paths on the same archive, entry filter and limits."""
    out = args.out / f"native-{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    l2d = [str(args.l2d), "--project", str(args.project), "--memory-limit", "512MB"]
    archive = str(args.fixture / "fixture.zip")
    common = ["--input", archive, "--entry", "*.log", "--max-record-bytes", "4096"]
    report = {"l2d": str(args.l2d), "run": args.run}
    if not (args.project / "models").exists() or args.fit:
        report["fit"] = timed([*l2d, "baseline", "fit", "--input",
                               str(args.fixture / "labelled.csv"), "--name", args.model],
                              out / "fit-stderr.txt")
    report["score"] = timed([*l2d, "--json", "baseline", "score", *common, "--baseline",
                             args.model, "--limit", "5", "--work-dir", str(out / "scored")],
                            out / "score-stderr.txt", out / "score.json")
    # The existing search keeps at most `--max-members` members (10,000 by default);
    # the glob selects far more of the fixture.
    report["search"] = timed([*l2d, "--json", "discover", "search", *common, "--glob",
                              "*login*", "--max-members", "1000000",
                              "--out", str(out / "selected")],
                             out / "search-stderr.txt", out / "search.json")
    # At 512 MB the existing search refuses the fixture (its catalogue of the selected
    # texts is held whole); with a limit that admits it, it gives the comparison.
    roomy = [str(args.l2d), "--project", str(args.project), "--memory-limit", "1GB"]
    report["search_1gb"] = timed([*roomy, "--json", "discover", "search", *common, "--glob",
                                  "*login*", "--max-members", "1000000",
                                  "--out", str(out / "selected-1gb")],
                                 out / "search-1gb-stderr.txt", out / "search-1gb.json")
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    return 0 if report["score"]["exit_code"] == 0 else 1


def probe(args):
    """Whether the selection of A completes under `--limit`, and where its tables
    went: for finding a limit at which the budget places them on disk."""
    oracle = json.loads(args.oracle.read_text())
    out = args.out / f"probe-{args.limit}"
    out.mkdir(parents=True, exist_ok=True)
    server = Server(args.l2d, args.project, ["--memory-limit", args.limit], [
        "--source", f"fixture={args.fixture / 'fixture.zip'}",
        "--work-dir", str(out / "work"), "--inline-wait-ms", "200"],
        out / "server-stderr.txt")
    found = {"limit": args.limit, "completed": False}
    started = time.perf_counter()
    try:
        opened = server.ok("triager_open", {"v": 1, "source": {
            "ref": "fixture", "entries": oracle["source"]["entries"],
            "max_record_bytes": oracle["source"]["max_record_bytes"]}}, "open")
        found["index_on_disk"] = opened["cost"]["index_on_disk"]
        selected = server.ok("triager_select", {"v": 1, "scope": opened["snapshot"], "producer": {
            "kind": "score", "model": args.model, "top_k": oracle["score_rules"]["top_k"],
            "ties": "record_id_ascending"}}, "select a")
        found.update(completed=True, revision_matches=(
            selected["revision"] == oracle["sets"]["a"]["revision"]),
            **{key: selected["cost"][key] for key in (
                "tables_on_disk", "read_again_for_disk", "archive_scans", "dedup_on_disk")})
    except Failed as failure:
        found["refused"] = str(failure)[:400]
    found["seconds"] = round(time.perf_counter() - started, 1)
    try:
        server.close()
    except Exception as error:  # noqa: BLE001 - the probe says how the server ended
        found["server_end"] = str(error)
    (out / "probe.json").write_text(json.dumps(found, indent=1) + "\n")
    print(json.dumps(found))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("native", "leg", "probe"):
        command = commands.add_parser(name)
        if name == "probe":
            command.add_argument("--oracle", required=True, type=Path)
            command.add_argument("--limit", required=True)
        elif name == "leg":
            command.add_argument("name", choices=LEGS)
            command.add_argument("--limit", help="the memory limit of the leg `admission`")
            command.add_argument("--oracle", required=True, type=Path)
            command.add_argument("--page-sizes", type=int, nargs="+", default=[100, 25, 7])
        else:
            command.add_argument("--fit", action="store_true")
        command.add_argument("--l2d", required=True, type=Path)
        command.add_argument("--project", required=True, type=Path)
        command.add_argument("--fixture", required=True, type=Path)
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--model", default="fixture")
        command.add_argument("--run", default="1")
    args = parser.parse_args()
    return {"leg": leg, "native": native, "probe": probe}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
