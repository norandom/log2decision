"""Drive the shipped l2d binary in a PTY using the Kiro walkthrough as the oracle."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import pyte

if os.name == "nt":
    from winpty import PtyProcess as Console
else:
    from ptyprocess import PtyProcessUnicode as Console


ORACLE = Path(__file__).resolve().parents[2] / ".kiro/specs/triager/prototype/walkthrough.json"
EXPECT = {step["name"]: step["expect"] for step in json.loads(ORACLE.read_text())["steps"]}

# The local probe checks the client PTY. On SSH, separately check the remote PTY that
# l2d actually owns. The binary and its arguments are shell positional parameters.
REMOTE_PROBE = """
before=$(stty -g) || exit 1
"$@"
code=$?
after=$(stty -g) || exit 1
if [ "$before" != "$after" ]; then
    echo 'l2d left the remote PTY input mode changed' >&2
    exit 1
fi
printf '\\r\\nTRIAGER_REMOTE_PTY_RESTORED\\r\\n'
exit "$code"
"""


class Target:
    def __init__(self, binary, directory, ssh_host=None):
        self.binary = str(binary)
        self.directory = directory
        self.ssh_host = ssh_host

    def ssh(self, argv, *, pty=False):
        transport = ["-tt"] if pty else ["-T", "-n"]
        return ["ssh", *transport, "-e", "none", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
                "--", self.ssh_host, shlex.join(argv)]

    def command(self, *args, pty=False):
        argv = [self.binary, *args]
        if self.ssh_host is None:
            return argv
        if pty:
            argv = ["sh", "-c", REMOTE_PROBE, "triager-probe", *argv]
        return self.ssh(argv, pty=pty)

    def read_receipt(self, name):
        path = self.directory / name
        if self.ssh_host is None:
            return path.read_bytes()
        return subprocess.run(self.ssh(["cat", str(path)]), check=True,
                              stdout=subprocess.PIPE, timeout=30).stdout

    def inspect(self, name):
        args = self.command("--json", "tui", "--demo", "--session", str(self.directory / name))
        result = subprocess.run(args, capture_output=True, check=True, timeout=30, encoding="utf-8")
        return json.loads(result.stdout)

    def open(self, name):
        argv = self.command("tui", "--demo", "--session", str(self.directory / name), pty=True)
        return Workbench(argv, remote=self.ssh_host is not None)


class Workbench:
    def __init__(self, args, remote=False, show_screen_on_error=True):
        self.args = args
        self.remote = remote
        self.show_screen_on_error = show_screen_on_error
        self.screen = pyte.Screen(120, 36)
        self.stream = pyte.Stream(self.screen)
        self.pending = queue.Queue()

    def __enter__(self):
        probe = str(Path(__file__).with_name("probe.py"))
        self.child = Console.spawn([sys.executable, probe, *self.args], dimensions=(36, 120),
                                   env={**os.environ, "TERM": "xterm-256color"})
        self.reader = threading.Thread(target=self.read_output, daemon=True)
        self.reader.start()
        return self

    def read_output(self):
        try:
            while True:
                chunk = self.child.read(16384)
                if not chunk:
                    break
                self.pending.put(chunk)
        except (EOFError, OSError):
            pass
        finally:
            self.pending.put(None)

    def __exit__(self, *_):
        self.child.close(force=True)
        self.reader.join(timeout=2)

    def text(self):
        return "\n".join(self.screen.display)

    def until(self, predicate):
        deadline = time.monotonic() + (30 if self.remote else 12)
        while not predicate(self.text()):
            left = deadline - time.monotonic()
            try:
                chunk = self.pending.get(timeout=max(0, left))
            except queue.Empty:
                break
            if chunk is None:
                break
            self.stream.feed(chunk)
            if time.monotonic() > deadline:
                break
        if not predicate(self.text()):
            detail = self.text() if self.show_screen_on_error else "retained report view"
            raise AssertionError(f"Terminal did not reach the expected view:\n{detail}")

    def expect(self, text):
        words = " ".join(text.split())
        self.until(lambda screen: words in " ".join(screen.split()))

    def drive(self, keys, expected):
        self.child.write(keys)
        self.expect(expected)

    def resize(self, rows, columns):
        self.screen.resize(lines=rows, columns=columns)
        self.child.setwinsize(rows, columns)

    def quit(self):
        self.child.write("q")
        if self.remote:
            self.expect("TRIAGER_REMOTE_PTY_RESTORED")
        self.expect("TRIAGER_PTY_RESTORED")
        deadline = time.monotonic() + 8
        while self.child.isalive() and time.monotonic() < deadline:
            time.sleep(0.03)
        if self.child.isalive() or self.child.wait() != 0:
            raise AssertionError("l2d failed to exit cleanly on q")


def scope(step, state="in"):
    counts = EXPECT[step]
    originals, events = counts[f"counts.{state}"], counts[f"occurrence_counts.{state}"]
    return f"{originals} original{'s' if originals != 1 else ''} · {events} occurrence{'s' if events != 1 else ''}"


def verify_search(target, evidence):
    # R01, R02 and R03 match on edge-01. R03 has two matching locations
    # and a third on another host. The latter must stay out of evidence.
    expression = "host=edge-01 (login OR challenge) score>=0.9"
    with target.open("search.json") as app:
        app.expect("12 originals")
        app.drive("/", "Enter apply")
        app.drive(expression + "\r", "3 originals · 4 occurrences")
        app.until(lambda screen: "universe 3/12 originals" in screen
                  and "authentication challenge rejected" in screen
                  and "idp-01" not in screen and "session verification declined" not in screen)
        (evidence / "search.txt").write_text(app.text(), encoding="utf-8")
        app.drive("1", "1 original · 2 occurrences")
        app.drive("\r", "Matching occurrences  2")
        app.expect("edge-01")
        app.drive("b", "1 original · 2 occurrences")
        app.drive("s", "Saved")
        app.quit()
    result = target.inspect("search.json")
    assert result["counts"] == {"originals": 1, "occurrences": 2}
    assert result["preview"] == [{"id": "R03", "pool": None}]
    return result["counts"]


def verify(target, evidence):
    difference = EXPECT["Reduce to A minus B"]
    counts_d = {"originals": difference["counts.in"], "occurrences": difference["occurrence_counts.in"]}
    scope_d = scope("Reduce to A minus B")
    hour = scope("Choose a UTC hour at D")
    with target.open("candidate.json") as app:
        app.expect(scope("Reduce to A minus B", "parent"))
        app.drive("1", scope_d)
        app.drive("c", "Prepared C coverage")
        app.drive("i", scope("Reduce D to recommendations"))
        app.drive("h", "Choose host")
        app.drive("\r", scope("Drill on a host in the recommended region"))
        app.drive("\r", "Matching occurrences  2")
        app.expect("R03")
        app.expect("edge-01")
        app.drive("b", "parent #2 1/3 originals")
        app.drive("b", "parent #1 3/6 originals")
        app.drive("\r", "Matching occurrences  3")
        app.expect("idp-01")
        app.drive("b", "parent #1 3/6 originals")
        app.drive("b", "parent #0 6/12 originals")
        app.drive("u", scope("Branch from D to its unassessed original"))
        app.drive("\r", EXPECT["Inspect the branch reason"]["reason"])
        app.expect("R08")
        app.drive("b", "parent #1 1/6 originals")
        app.drive("b", "parent #0 6/12 originals")
        app.drive("t", "Choose UTC hour")
        app.drive("\r", hour)
        app.drive("/", "Enter apply")
        app.drive("no matching fixture original\r", "No matching originals")
        app.drive("r", "No originals in this view.")
        app.drive("b", hour)
        app.drive("b", "parent #0 6/12 originals")
        app.drive("s", "Saved")
        before_browsing = target.read_receipt("candidate.json")
        # Start on the second original in D, move within the raw reader, then
        # reopen evidence to verify that the results cursor stayed on R04.
        app.drive("jr", "Original 2 of 6")
        app.expect("Raw logs")
        app.expect("session verification declined")
        app.drive("k", "Original 1 of 6")
        app.expect("authentication challenge rejected")
        app.drive("j", "Original 2 of 6")
        app.drive("j", "Original 3 of 6")
        app.expect("identity check expired")
        (evidence / "raw.txt").write_text(app.text(), encoding="utf-8")
        app.drive("b", "Regions & fields")
        app.drive("\r", "Record evidence")
        app.expect("Reviewed session-verification failures provide a lead.")
        app.drive("r", "Original 2 of 6")
        app.drive("\x1b", "Record evidence")
        app.expect("Reviewed session-verification failures provide a lead.")
        app.drive("b", scope_d)
        (evidence / "wide.txt").write_text(app.text(), encoding="utf-8")
        app.resize(24, 80)
        app.drive("?", "Keyboard")
        app.child.write("?")
        app.until(lambda screen: "Keyboard" not in screen and "Regions & fields" in screen)
        (evidence / "80x24.txt").write_text(app.text(), encoding="utf-8")
        app.resize(36, 120)
        app.child.write("]+[-")
        app.drive("?", "Keyboard")
        app.child.write("?")
        app.until(lambda screen: "Keyboard" not in screen)
        app.drive("s", "Saved")
        assert target.read_receipt("candidate.json") == before_browsing, "Raw browsing or pane changes mutated the investigation"
        app.quit()
    candidate = target.inspect("candidate.json")
    assert candidate["counts"] == counts_d and not candidate["allocated"], candidate
    coverage = EXPECT["Collect prepared recommendations scoped to D"]["coverage"]
    for name, key in [("recommended", "in"), ("pass", "out"), ("unknown", "not_evaluated")]:
        assert candidate["regions"][name]["originals"] == coverage[key]

    # The analyst can share the draft independently of the final handoff.
    with target.open("candidate.json") as app:
        app.expect(scope_d)
        app.drive("v", "Retained branches")
        app.expect("C unknown")
        app.drive("\r", "parent #0 6/12 originals")
        app.drive("f", "Allocate final work set")
        app.drive("\r", "Analyst work set")
        app.expect("[unfamiliar]")
        app.expect("[audit]")
        app.drive("f", "Already allocated")
        app.drive("S", "Save receipt as")
        app.drive(str(target.directory / "final.json") + "\r", "Saved")
        app.quit()

    final = target.inspect("final.json")
    allocation = EXPECT["Allocate once from the three disjoint pools"]
    expected_rows = EXPECT["Read the original order and attribution"]["rows"]
    assert final["allocated"] and final["counts"] == {
        "originals": allocation["selected"], "occurrences": allocation["occurrence_counts.in"]}
    assert final["preview"] == [{"id": row["id"], "pool": row["allocation"]} for row in expected_rows]
    assert target.inspect("candidate.json")["counts"] == counts_d
    with target.open("final.json") as app:
        app.expect("Analyst work set")
        app.expect("[audit]")
        app.drive("r", "Original 1 of 4")
        app.expect("authentication challenge rejected")
        app.drive("j", "Original 2 of 4")
        app.expect("session verification declined")
        app.drive("j", "Original 3 of 4")
        app.expect("identity response format changed")
        app.drive("j", "Original 4 of 4")
        app.expect("service account rotation complete")
        app.drive("\x1b", "Regions & fields")
        app.expect("Analyst work set")
        app.quit()
    summary = {"candidate": candidate["counts"], "coverage": coverage, "ordered_final": final["preview"],
               "search_then_a_minus_b": verify_search(target, evidence)}
    if target.ssh_host is not None:
        for name in ["candidate.json", "final.json", "search.json"]:
            (evidence / name).write_bytes(target.read_receipt(name))
        summary["ssh_host"] = target.ssh_host
        summary["remote_run"] = str(target.directory)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--binary", type=Path, help="local l2d executable")
    where.add_argument("--ssh-host", help="configured, known POSIX SSH host with matching l2d installed")
    parser.add_argument("--remote-binary", type=PurePosixPath, help="absolute path to installed remote l2d")
    parser.add_argument("--remote-dir", type=PurePosixPath, help="existing remote scratch directory")
    parser.add_argument("--evidence", type=Path, required=True, help="directory for per-run synthetic test receipts")
    args = parser.parse_args()
    if args.ssh_host is not None:
        if not args.ssh_host or args.ssh_host.startswith("-") or any(
                c.isspace() or not c.isprintable() for c in args.ssh_host):
            parser.error("--ssh-host requires a host name, SSH alias or user@host")
        for path in [args.remote_binary, args.remote_dir]:
            if (not path or not path.is_absolute() or path.parent == path or ".." in path.parts
                    or not str(path).isprintable()):
                parser.error("--ssh-host requires absolute, non-root --remote-binary and --remote-dir paths")
    elif args.remote_binary or args.remote_dir:
        parser.error("--remote-binary and --remote-dir require --ssh-host")
    args.evidence.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(dir=args.evidence, prefix="native-pty-"))
    if args.ssh_host is None:
        target = Target(args.binary.resolve(strict=True), run)
    else:
        remote_run = args.remote_dir / f"native-pty-{uuid.uuid4().hex}"
        target = Target(args.remote_binary, remote_run, args.ssh_host)
        # No -p: the operator's scratch directory must already exist.
        setup = ["sh", "-c", 'test -f "$1" && test -x "$1" && mkdir -m 700 "$2"',
                 "triager-setup", str(args.remote_binary), str(remote_run)]
        subprocess.run(target.ssh(setup), check=True, timeout=30)
        print(f"Remote evidence at {args.ssh_host}:{remote_run}", flush=True)
    summary = verify(target, run)
    (run / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Passed native PTY walkthrough. Evidence at {run}")


if __name__ == "__main__":
    main()
