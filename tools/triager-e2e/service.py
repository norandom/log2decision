"""Drive the workbench over the triager service in a PTY, locally or over SSH (task 3.2).

The same recipe runs twice: as an MCP recipe (`l2d mcp --requests`), which reduces,
allocates and saves a receipt, and in the terminal (`l2d tui --requests`), where the
keyboard reduces, inspects, resizes, cancels, allocates and saves. The ordered work-set
IDs of both receipts, and of the MCP receipt reopened in the terminal, must be equal
(requirement 5.4). Work counters read on screen before and after resizing must not
change (requirement 6.3). The caller's terminal modes are checked after every session,
on both ends of an SSH connection (requirement 5.3).

Local:   python service.py --binary l2d.exe --project DIR --fixture DIR --evidence DIR
SSH:     python service.py --ssh-config FILE --ssh-host triager-ssh --remote-binary
         /target/release/l2d --remote-project /data/project --remote-fixture
         /data/fixtures/NAME --remote-dir /data/runs --evidence DIR

The fixture directory holds `fixture.zip` and `membership-c.jsonl` (tools/triager-scale/
generate.py); the project holds a fitted baseline scorer named by --model.
"""

import argparse
import json
import os
import queue
from pathlib import Path, PurePosixPath
import re
import shlex
import statistics
import subprocess
import tempfile
import time
import uuid

import run as pty


class ServiceTarget(pty.Target):
    """A local or SSH host with the binary, a project, a fixture and a run directory."""

    def __init__(self, binary, directory, project, fixture, ssh_host=None, ssh_config=None):
        super().__init__(binary, directory, ssh_host)
        self.project = project
        self.fixture = fixture
        self.ssh_config = ssh_config
        self.processes = 0

    def ssh(self, argv, *, pty=False):
        transport = ["-tt"] if pty else ["-T"]
        config = ["-F", str(self.ssh_config)] if self.ssh_config else []
        return ["ssh", *config, *transport, "-e", "none", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
                "--", self.ssh_host, shlex.join(argv)]

    def path(self, *parts):
        return self.directory.joinpath(*parts)

    def write(self, name, text):
        path = self.path(name)
        if self.ssh_host is None:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        else:
            subprocess.run(self.ssh(["sh", "-c", 'cat > "$1"', "write", str(path)]),
                           input=text.encode("utf-8"), check=True, timeout=60)
        return path

    def read(self, name):
        return self.read_receipt(name).decode("utf-8")

    def host_args(self):
        self.processes += 1
        return ["--source", f"fixture={self.fixture / 'fixture.zip'}",
                "--membership", f"fixture-c={self.fixture / 'membership-c.jsonl'}",
                "--receipts", str(self.path("receipts")),
                "--work-dir", str(self.path(f"work-{self.processes}"))]

    def l2d(self, *args, memory="512MB"):
        return ["--project", str(self.project), "--memory-limit", memory, *args]

    def run(self, *args, timeout=3600):
        argv = self.command(*args)
        if self.ssh_host is not None:
            argv = [a for a in argv]
        started = time.monotonic()
        result = subprocess.run(argv, capture_output=True, timeout=timeout)
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise AssertionError(f"{args[:6]} failed ({result.returncode}): "
                                 f"{result.stderr.decode('utf-8', 'replace')[-2000:]}")
        return result.stdout.decode("utf-8"), elapsed

    def tui(self, *args):
        argv = self.command(*args, pty=True)
        return pty.Workbench(argv, remote=self.ssh_host is not None)


def open_call(timestamps):
    projections = {"fields": [{"name": "host", "from": "kv"}, {"name": "user", "from": "kv"},
                              {"name": "event", "from": "kv"}]}
    if timestamps:
        projections["time"] = {"from": "text"}
    return {"tool": "triager_open", "args": {"v": 1, "source": {"ref": "fixture", "entries": ["*.log"]},
                                             "projections": projections},
            "bind": {"S": "snapshot"}}


def sets_calls(model, top_k):
    return [
        {"tool": "triager_select", "args": {"v": 1, "scope": "$S", "producer": {
            "kind": "score", "model": model, "top_k": top_k, "ties": "record_id_ascending"}},
         "bind": {"A": "set"}},
        {"tool": "triager_select", "args": {"v": 1, "scope": "$S", "producer": {
            "kind": "filter", "syntax": "glob", "query": "*login*"}}, "bind": {"B": "set"}},
        {"tool": "triager_select", "args": {"v": 1, "scope": "$S", "producer": {
            "kind": "membership", "ref": "fixture-c"}}, "bind": {"C": "set"}},
    ]


BUDGET = (18, 4, 2, 5)


def allocate_call():
    general, unfamiliar, audit, seed = BUDGET
    return {"tool": "triager_compose", "args": {
        "v": 1, "mode": "allocate", "snapshot": "$S", "sets": {"d": "$D", "c": "$C"},
        "expr": {"intersection": ["d", "c"]},
        "budget": {"total": general + unfamiliar + audit, "duplicate_rule": "first_pool_wins",
                   "shortfall": "leave_unfilled", "pools": [
                       {"name": "general", "limit": general, "from": {"intersection": ["d", "c"]},
                        "order": ["score_desc", "record_id_ascending"]},
                       {"name": "unfamiliar", "limit": unfamiliar, "from": {"intersection": [
                           "d", {"state": {"set": "c", "is": "not_evaluated"}}]},
                        "order": ["record_id_ascending"]},
                       {"name": "audit", "limit": audit, "from": {"difference": ["d", "c"]},
                        "order": ["seeded_hash", "record_id_ascending"], "seed": seed}]}},
        "bind": {"F": "set"}}


def recipe(lines):
    return "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)


def mcp_leg(target, model, top_k, timestamps):
    calls = [open_call(timestamps), *sets_calls(model, top_k),
             {"tool": "triager_compose", "args": {"v": 1, "mode": "view", "snapshot": "$S",
              "sets": {"a": "$A", "b": "$B"}, "expr": {"intersection": [
                  {"state": {"set": "a", "is": "in"}}, {"state": {"set": "b", "is": "out"}}]}},
              "bind": {"D": "view"}},
             {"tool": "triager_compose", "args": {"v": 1, "mode": "view", "snapshot": "$S", "scope": "$D",
              "sets": {"c": "$C"}, "expr": {"state": {"set": "c", "is": "in"}}}, "bind": {"DC": "view"}},
             allocate_call(),
             {"tool": "triager_inspect", "args": {"v": 1, "handle": "$F", "view": "records", "limit": 100,
              "projection": ["id", "allocation"]}},
             {"tool": "triager_save", "args": {"v": 1, "handle": "$F",
              "sets": {"a": "$A", "b": "$B", "c": "$C"}}, "bind": {"H": "receipt"}}]
    path = target.write("mcp.jsonl", recipe(calls))
    out, elapsed = target.run(*target.l2d("mcp", *target.host_args(), "--requests", str(path)))
    results = [json.loads(line) for line in out.splitlines() if line.strip()]
    by_tool = [(r["tool"], r["result"]) for r in results]
    compose = [r for tool, r in by_tool if tool == "triager_compose"]
    d, dc, f = compose[0], compose[1], compose[2]
    records = [r for tool, r in by_tool if tool == "triager_inspect"][-1]
    save = [r for tool, r in by_tool if tool == "triager_save"][-1]
    return {"elapsed_s": elapsed, "d": d["view"], "d_counts": d["counts"], "d_occurrences": d["occurrence_counts"],
            "dc_counts": dc["counts"], "final": f["set"], "ordered": [row["id"] for row in records["rows"]],
            "pools": [row["allocation"]["pool"] for row in records["rows"]], "receipt": save["receipt"],
            "snapshot_cost": results[0]["result"]["cost"]}


def header(counts):
    return f"{counts['in']:,}/{counts['parent']:,} originals"


WORK = re.compile(r"work: scans ([\d,]+) · unpacked ([\d,]+) · scored ([\d,]+) · encoded ([\d,]+) · "
                  r"genai ([\d,]+) · excerpt bytes ([\d,]+)")


def drain(app, seconds):
    """Feeds what the terminal writes for `seconds` into the screen."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            chunk = app.pending.get(timeout=0.05)
        except queue.Empty:
            continue
        if chunk is None:
            return
        app.stream.feed(chunk)


def work(app):
    # A notice (such as a saved receipt) takes the status line; Esc in browse mode
    # would go back, so the counters are asked for and the screen is read after the
    # overview has been drawn.
    app.child.write("w")
    drain(app, 1.0 if not app.remote else 2.0)
    app.until(lambda screen: WORK.search(" ".join(screen.split())) is not None)
    match = WORK.search(" ".join(app.text().split()))
    names = ["archive_scans", "bytes_unpacked", "rows_scored", "rows_encoded", "genai_rows_submitted",
             "evidence_bytes_disclosed"]
    return {name: int(value.replace(",", "")) for name, value in zip(names, match.groups())}


def wait_for(app, text, seconds):
    """Waits up to `seconds` for `text`: for the steps whose cost grows with the
    archive (the recipe before the first view, an excerpt, a receipt)."""
    words = " ".join(text.split())
    deadline = time.monotonic() + seconds
    while words not in " ".join(app.text().split()):
        if time.monotonic() > deadline:
            raise AssertionError(f"Terminal did not show {text!r}:\n{app.text()}")
        try:
            chunk = app.pending.get(timeout=0.2)
        except queue.Empty:
            continue
        if chunk is None:
            raise AssertionError(f"The terminal ended before {text!r}:\n{app.text()}")
        app.stream.feed(chunk)


def busy(app):
    """Whether the status line shows a call or job still under way."""
    status = " ".join(app.text().splitlines()[-2:])
    return "Running " in status or "Working…" in status


def slow(app, keys, text, seconds=600, absent=None):
    """Presses `keys` and waits until `text` is on screen, `absent` is not (so a view
    really changed, not merely was drawn before the key took effect), and no call or
    job is under way."""
    words = " ".join(text.split())
    gone = " ".join(absent.split()) if absent else None
    app.child.write(keys)
    deadline = time.monotonic() + seconds
    while True:
        shown = " ".join(app.text().split())
        if words in shown and (gone is None or gone not in shown) and not busy(app):
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"Terminal did not show {text!r} without {absent!r}:\n{app.text()}")
        try:
            chunk = app.pending.get(timeout=0.2)
        except queue.Empty:
            continue
        if chunk is None:
            raise AssertionError(f"The terminal ended before {text!r}:\n{app.text()}")
        app.stream.feed(chunk)


def press_until(app, key, text, most=8):
    """Presses `key` until `text` is on screen: PgDn moves 10 rows, and a page is
    fetched when the selection is already on the last row of the page."""
    words = " ".join(text.split())
    for _ in range(most):
        app.child.write(key)
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            try:
                chunk = app.pending.get(timeout=0.1)
            except Exception:
                chunk = ""
            if chunk is None:
                break
            if chunk:
                app.stream.feed(chunk)
            if words in " ".join(app.text().split()):
                return
    app.expect(text)


def sgr(button, column, row, release=False):
    return f"\x1b[<{button};{column + 1};{row + 1}{'m' if release else 'M'}"


def tui_leg(target, mcp, evidence, *, cancel):
    tui_recipe = target.write("tui.jsonl", recipe([open_call(mcp["timestamps"]),
                                                   *sets_calls(mcp["model"], mcp["top_k"])]))
    trace = target.path("trace-1.jsonl")
    checks = {}
    with target.tui(*target.l2d("tui", "--requests", str(tui_recipe), "--trace", str(trace),
                                *target.host_args())) as app:
        wait_for(app, "Regions of a b c", 3600)
        wait_for(app, "Results 1–", 600)
        slow(app, "1", header(mcp["d_counts"]), 600)
        app.expect("> a in · b out")
        (evidence / "tui-d.txt").write_text(app.text(), encoding="utf-8")
        before = work(app)
        # Resize both panes with the keyboard, shrink to 80×24 and widen again.
        for key in "]]][++-":
            app.child.write(key)
            time.sleep(0.15)
        app.resize(24, 80)
        app.expect("Results")
        app.expect(header(mcp["d_counts"]))
        (evidence / "tui-80x24.txt").write_text(app.text(), encoding="utf-8")
        checks["fits_80x24"] = all(len(line.rstrip()) <= 80 for line in app.text().splitlines())
        app.resize(36, 120)
        app.expect("Results")
        # Optional pointer drag of the fields pane's edge (38 columns after ]]][).
        time.sleep(0.3)
        body_row = 1 + 4 + 4 + 3
        app.child.write(sgr(0, 37, body_row) + sgr(32, 49, body_row) + sgr(0, 49, body_row, release=True))
        time.sleep(0.5)
        after = work(app)
        checks["work_unchanged_by_resize"] = before == after
        assert before == after, (before, after)
        checks["work_before_resize"] = before
        # The sibling C in inside D, its evidence and the excerpt read from the source.
        slow(app, "i", header(mcp["dc_counts"]), absent=header(mcp["d_counts"]))
        app.drive("\r", "Why it is here")
        slow(app, "x", "Excerpt (", 600)
        (evidence / "tui-evidence.txt").write_text(app.text(), encoding="utf-8")
        app.child.write("\x1b")
        time.sleep(0.2)
        slow(app, "b", header(mcp["d_counts"]), absent=header(mcp["dc_counts"]))
        # Warm drill-down: siblings of D and back, and paging, at least 30 interactions
        # over committed views (performance.md, "Warm analyst drill-down").
        warm = 0
        warm_start = work(app)
        checks["work_after_excerpt"] = warm_start
        for _ in range(5):
            slow(app, "i", header(mcp["dc_counts"]), absent=header(mcp["d_counts"]))
            slow(app, "b", header(mcp["d_counts"]), absent="> c in")
            slow(app, "u", "> c not evaluated", absent=header(mcp["d_counts"]))
            slow(app, "b", header(mcp["d_counts"]), absent="> c not evaluated")
            warm += 4
        total = mcp["d_counts"]["in"]
        pages = min(12, max(0, (total - 1) // 25))
        for page in range(1, pages + 1):
            press_until(app, "\x1b[6~", f"Results {page * 25 + 1:,}–")
            warm += 1
        for page in range(pages - 1, -1, -1):
            press_until(app, "\x1b[5~", f"Results {page * 25 + 1:,}–")
            warm += 1
        # A retained field value and a time bin, drilled and backed out of, inside the
        # same zero-work check (performance.md, "Warm analyst drill-down").
        app.child.write("\t\t")
        time.sleep(0.2)
        app.child.write("\x1b[B\x1b[B")
        slow(app, "\r", "values by occurrences", 120)
        slow(app, "\r", "> host = ", 600, absent=header(mcp["d_counts"]))
        (evidence / "tui-field.txt").write_text(app.text(), encoding="utf-8")
        slow(app, "b", header(mcp["d_counts"]), 120, absent="> host = ")
        app.child.write("\t")
        time.sleep(0.2)
        if mcp["timestamps"]:
            slow(app, "t", "Time buckets of", 120)
            slow(app, "\r", "> time 20", 600, absent=header(mcp["d_counts"]))
            (evidence / "tui-time.txt").write_text(app.text(), encoding="utf-8")
            slow(app, "b", header(mcp["d_counts"]), 120, absent="> time 20")
            warm += 1
        warm += 1
        checks["warm_interactions"] = warm
        after_warm = work(app)
        # Field values shown count as disclosed evidence (task 3.1); that is no archive,
        # scoring or model work, so it is recorded apart.
        work_keys = ["archive_scans", "bytes_unpacked", "rows_scored", "rows_encoded",
                     "genai_rows_submitted"]
        unchanged = all(after_warm[key] == warm_start[key] for key in work_keys)
        checks["work_unchanged_by_warm_drill"] = unchanged
        checks["evidence_bytes_disclosed_by_warm_drill"] = (
            after_warm["evidence_bytes_disclosed"] - warm_start["evidence_bytes_disclosed"])
        assert unchanged, (warm_start, after_warm)
        excerpt_scans = warm_start["archive_scans"] - after["archive_scans"]
        checks["excerpt_archive_scans"] = excerpt_scans
        assert excerpt_scans == 1, (after, warm_start)
        # Cancel a search while its job runs: nothing is committed, D stays.
        if cancel:
            app.child.write("/")
            app.child.write("*no such text in the fixture*")
            # The search runs as a job: wait for it to show as running, not to end.
            app.child.write("\r")
            wait_for(app, "Running search", 120)
            slow(app, "\x1b", "Cancelled search", 300)
            app.expect(header(mcp["d_counts"]))
            checks["cancelled_search"] = True
        # Allocate once with the MCP recipe's budget, then save.
        app.drive("f", "Allocate final work set")
        app.child.write("\x7f" * 10)
        app.child.write(" ".join(str(n) for n in BUDGET))
        slow(app, "\r", "Analyst work set", 600)
        app.expect("general #1")
        (evidence / "tui-final.txt").write_text(app.text(), encoding="utf-8")
        slow(app, "s", "Saved receipt", 600)
        found = re.search(r"Saved receipt (rcpt-[0-9a-f]+)", " ".join(app.text().split()))
        assert found, app.text()
        checks["tui_receipt"] = found.group(1)
        app.quit()
    checks["restored"] = True
    return checks, trace


def reopened(target, receipt):
    out, elapsed = target.run(*target.l2d("--json", "tui", "--receipt", receipt, *target.host_args()))
    summary = json.loads(out)
    return summary, elapsed


def reopen_leg(target, mcp, evidence):
    trace = target.path("trace-2.jsonl")
    with target.tui(*target.l2d("tui", "--receipt", mcp["receipt"], "--trace", str(trace),
                                *target.host_args())) as app:
        app.expect(f"Receipt {mcp['receipt']}"[:28])
        app.expect("general #1")
        start = work(app)
        app.drive("\r", "Why it is here")
        slow(app, "x", "Excerpt (", 600)
        app.child.write("\x1b")
        time.sleep(0.2)
        end = work(app)
        (evidence / "tui-reopened.txt").write_text(app.text(), encoding="utf-8")
        # Counts after filtering the reopened hunt: its retained originals are D's, so
        # a in · b out keeps all of them and c in inside it is MCP's D ∩ C.
        d_in, dc_in = mcp["d_counts"]["in"], mcp["dc_counts"]["in"]
        app.drive("b", "Regions of a b c")
        slow(app, "1", f"{d_in:,}/{d_in:,} originals", 120)
        slow(app, "i", f"{dc_in:,}/{d_in:,} originals", 120, absent=f"{d_in:,}/{d_in:,} originals")
        filtered = work(app)
        app.quit()
    return {"work_at_reopen": start, "work_after_excerpt": end, "work_after_filtering": filtered}, trace


def percentile(values, share):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]


def summarize(trace_text):
    events = [json.loads(line) for line in trace_text.splitlines() if line.strip()]
    layout = [e["layout_us"] for e in events if e["event"] == "frame" and e.get("layout_us") is not None]
    draw = [e["draw_us"] for e in events if e["event"] == "frame"]
    calls = {}
    for e in events:
        if e["event"] == "call" and e["tool"] != "triager_job":
            key = f"{e['tool']} {e['view'] or ''}".strip()
            calls.setdefault(key, []).append(e["ms"])
    # Operations from their call to their result, jobs included, by kind.
    fixed = {"work counters", "records page", "regions", "fields", "histogram", "evidence",
             "excerpt (a read of the source)", "saving a receipt"}
    ops, service, stored = {}, {}, {}
    for e in events:
        if e["event"] == "op" and e["outcome"] == "ready":
            label = e["op"]
            key = label if label in fixed else (
                "search" if label.startswith("search") else
                "values" if label.startswith("values of") else
                "allocate" if label.startswith("Analyst work set") else "drill (child view)")
            ops.setdefault(key, []).append(e["ms"])
            # The service's own time of the work, where its result states one. A result
            # the service returns from store keeps the cost of its first computation:
            # a stated time above what the client saw (whole milliseconds) was not spent
            # by this call, so it is counted as stored, not as service time.
            if e.get("service_ms") is not None:
                if e["service_ms"] <= e["ms"] + 1:
                    service.setdefault(key, []).append(e["service_ms"])
                else:
                    stored[key] = stored.get(key, 0) + 1
    # Calls between the first pane resize and the work counters read after it.
    keys = [i for i, e in enumerate(events) if e["event"] == "key"]
    first = next((i for i in keys if events[i]["code"] == "Char(']')"), None)
    after = next((i for i in keys if first is not None and i > first and events[i]["code"] == "Char('w')"), None)
    between = [] if first is None or after is None else [
        e for e in events[first:after] if e["event"] == "call"]
    sizes = [(e["width"], e["height"], e["sidebar"], e["search_rows"]) for e in events if e["event"] == "frame"]
    return {
        "frames": len(draw),
        "layout_us": {"solved": len(layout), "p50": percentile(layout, 0.5), "p95": percentile(layout, 0.95),
                      "max": max(layout) if layout else None},
        "draw_us": {"p50": percentile(draw, 0.5), "p95": percentile(draw, 0.95), "max": max(draw) if draw else None},
        "calls_ms": {key: {"n": len(v), "p50": percentile(v, 0.5), "p95": percentile(v, 0.95),
                           "max": max(v)} for key, v in sorted(calls.items())},
        "ops_ms": {key: {"n": len(v), "p50": percentile(v, 0.5), "p95": percentile(v, 0.95),
                         "max": max(v)} for key, v in sorted(ops.items())},
        "service_ms": {key: {"n": len(v), "p50": percentile(v, 0.5), "p95": percentile(v, 0.95),
                             "max": max(v)} for key, v in sorted(service.items())},
        "returned_from_store": stored,
        "calls_during_resize": len(between),
        "resizes": sum(1 for e in events if e["event"] == "resize"),
        "pane_sizes_seen": sorted(set(sizes)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--remote-binary", type=PurePosixPath)
    parser.add_argument("--remote-project", type=PurePosixPath)
    parser.add_argument("--remote-fixture", type=PurePosixPath)
    parser.add_argument("--remote-dir", type=PurePosixPath)
    parser.add_argument("--model", default="fixture")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--timestamps", action="store_true", help="the fixture's lines start with a timestamp")
    parser.add_argument("--cancel", action="store_true", help="cancel a search job (needs a fixture large "
                                                                  "enough for a scan to outlast a key press)")
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(dir=args.evidence, prefix="service-pty-"))
    if args.ssh_host:
        remote = args.remote_dir / f"service-pty-{uuid.uuid4().hex}"
        target = ServiceTarget(args.remote_binary, remote, args.remote_project, args.remote_fixture,
                               args.ssh_host, args.ssh_config)
        subprocess.run(target.ssh(["sh", "-c", 'test -x "$1" && mkdir -m 700 "$2"', "setup",
                                   str(args.remote_binary), str(remote)]), check=True, timeout=30)
        print(f"Remote run at {args.ssh_host}:{remote}", flush=True)
    else:
        target = ServiceTarget(args.binary.resolve(strict=True), run, args.project.resolve(),
                               args.fixture.resolve())
    started = time.monotonic()
    mcp = mcp_leg(target, args.model, args.top_k, args.timestamps)
    mcp.update(model=args.model, top_k=args.top_k, timestamps=args.timestamps)
    print(f"MCP recipe: D {mcp['d_counts']['in']:,} originals, final {len(mcp['ordered'])} slots, "
          f"receipt {mcp['receipt']} in {mcp['elapsed_s']:.1f} s", flush=True)
    checks, trace = tui_leg(target, mcp, run, cancel=args.cancel)
    print(f"Terminal: receipt {checks['tui_receipt']}", flush=True)
    tui_final, _ = reopened(target, checks["tui_receipt"])
    mcp_final, _ = reopened(target, mcp["receipt"])
    ids = lambda summary: [row["id"] for row in summary["page"]["rows"]]
    assert ids(tui_final) == mcp["ordered"], (ids(tui_final), mcp["ordered"])
    assert ids(mcp_final) == mcp["ordered"], (ids(mcp_final), mcp["ordered"])
    assert tui_final["view"]["kind"] == "final" and mcp_final["view"]["kind"] == "final"
    reopen, trace2 = reopen_leg(target, mcp, run)
    assert reopen["work_at_reopen"]["archive_scans"] == 0, reopen
    assert reopen["work_at_reopen"]["rows_scored"] == 0, reopen
    assert reopen["work_after_excerpt"]["archive_scans"] == 1, reopen
    traces = {"session": summarize(target.read("trace-1.jsonl")),
              "reopened": summarize(target.read("trace-2.jsonl"))}
    for name in ["trace-1.jsonl", "trace-2.jsonl", "mcp.jsonl", "tui.jsonl"]:
        (run / name).write_text(target.read(name), encoding="utf-8")
    assert traces["session"]["calls_during_resize"] == 0, traces["session"]
    summary = {"ssh_host": args.ssh_host, "remote_run": str(target.directory) if args.ssh_host else None,
               "elapsed_s": time.monotonic() - started, "mcp": mcp, "checks": checks,
               "ordered_ids_equal": {"terminal_receipt": True, "mcp_receipt_in_terminal": True},
               "reopen": reopen, "traces": traces}
    (run / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Passed the service PTY walkthrough. Evidence at {run}")


if __name__ == "__main__":
    main()
