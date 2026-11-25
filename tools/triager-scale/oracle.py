"""Compute the expected results of the triager scale gate without the product.

Reads the fixture archive with Python's `zipfile`, splits and judges its lines as
`.kiro/specs/triager/performance.md` and the archive reader's documentation state
them, deduplicates the texts, and derives every count the scale run compares:
accepted and skipped records, distinct originals, the membership of the filter sets,
the membership file that stands in for C with its unknown states, the eight regions
of A, B and C with the unknown combinations, the reductions D = A - B, E = D and C
within D and "what C missed", their ordered record IDs, and the digest (`revision`)
the service states for each membership.

A is a score rule. The scores are not recomputed here: they are read from the parts
the existing command `l2d baseline score` wrote (`--scores`), and the top-k rule with
its tie rule, the threshold rule, the ranks and everything composed from A are
computed here from those scores. Without `--scores` everything that needs A is left
out.

    uv run --no-project --with-requirements requirements.txt python oracle.py \\
        --zip fixture.zip --membership membership-c.tsv --scores scored --out oracle.json

`polars` is used for one thing: reading the Parquet parts of `--scores`.

This file imports nothing of `generate.py` and nothing of the product.
"""

import argparse
import array
import bisect
import fnmatch
import hashlib
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path

LINE_BITS = 40
OUT, IN, UNKNOWN, ABSTAINED = 0, 1, 2, 3
MASK = (1 << 128) - 1
REGEX = r"event=http_(get|post) .* status=5\d\d "
LITERAL = "status=404"


def record_id(position: int) -> str:
    return f"r-{position:016x}"


def read_archive(path: Path, patterns, max_record_bytes: int):
    """The distinct originals of the selected text entries, in order of first
    occurrence: first positions, occurrence counts and filter flags."""
    index = {}
    first = array.array("Q")
    count = array.array("L")
    flags = bytearray()
    regex = re.compile(REGEX)
    report = {
        "entries_read": 0, "entries_excluded": 0, "entries_unsupported": 0,
        "records": 0, "accepted_occurrences": 0,
        "skipped": {"empty line": 0, "line is not valid UTF-8": 0,
                    "longer than the maximum record length": 0},
        "bytes_unpacked_selected": 0, "bytes_packed_selected": 0,
        "entry_records": {},
    }
    with zipfile.ZipFile(path) as archive:
        for entry, info in enumerate(archive.infolist()):
            name = info.filename
            if info.is_dir() or not any(fnmatch.fnmatchcase(name, p) for p in patterns):
                report["entries_excluded"] += 1
                continue
            report["bytes_unpacked_selected"] += info.file_size
            report["bytes_packed_selected"] += info.compress_size
            with archive.open(info) as member:
                head = member.read(8192)
            text_head = b"\0" not in head
            if text_head:
                try:
                    head.decode("utf-8")
                except UnicodeDecodeError as error:
                    # A character cut by the end of the head is no evidence.
                    text_head = error.start >= len(head) - 3 and len(head) == 8192
            if not text_head:
                report["entries_unsupported"] += 1
                continue
            report["entries_read"] += 1
            records = 0
            with archive.open(info) as member:
                for number, raw in enumerate(member, start=1):
                    records += 1
                    line = raw[:-1] if raw.endswith(b"\n") else raw
                    if line.endswith(b"\r"):
                        line = line[:-1]
                    if not line:
                        report["skipped"]["empty line"] += 1
                        continue
                    if len(line) > max_record_bytes:
                        report["skipped"]["longer than the maximum record length"] += 1
                        continue
                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError:
                        report["skipped"]["line is not valid UTF-8"] += 1
                        continue
                    report["accepted_occurrences"] += 1
                    digest = hashlib.blake2b(line, digest_size=16).digest()
                    at = index.get(digest)
                    if at is None:
                        index[digest] = len(first)
                        first.append(entry << LINE_BITS | number)
                        count.append(1)
                        flags.append(
                            ("login" in text)
                            | (regex.search(text) is not None) << 1
                            | (LITERAL in text) << 2
                        )
                    else:
                        count[at] += 1
            report["entry_records"][name] = records
            report["records"] += records
    report["distinct_originals"] = len(first)
    report["skipped_records"] = sum(report["skipped"].values())
    return first, count, flags, report


def row_of(first, position: int) -> int:
    at = bisect.bisect_left(first, position)
    if at == len(first) or first[at] != position:
        raise SystemExit(f"{record_id(position)} names no original of the archive")
    return at


def read_scores(directory: Path, first):
    """The ratio `l2d baseline score` wrote for every original; NaN where none."""
    import polars

    scores = array.array("d", [math.nan]) * len(first)
    seen = 0
    for part in sorted(directory.glob("scores-*.parquet")):
        frame = polars.read_parquet(part, columns=["entry", "line", "ratio"])
        for entry, line, ratio in zip(
            frame["entry"].to_list(), frame["line"].to_list(), frame["ratio"].to_list()
        ):
            scores[row_of(first, entry << LINE_BITS | line)] = (
                math.nan if ratio is None else ratio
            )
            seen += 1
    if seen != len(first):
        raise SystemExit(f"the score parts hold {seen} rows and the archive {len(first)} originals")
    return scores


def read_membership(path: Path, first):
    codes = bytearray([UNKNOWN]) * len(first)
    with path.open("rb") as lines:
        for number, raw in enumerate(lines, start=1):
            handle, _, outcome = raw.rstrip(b"\r\n").decode().partition("\t")
            code = {"in": IN, "out": OUT, "abstain": ABSTAINED}[outcome]
            row = row_of(first, int(handle[2:], 16))
            if codes[row] != UNKNOWN and codes[row] != code:
                raise SystemExit(f"line {number} of {path} conflicts with a line before it")
            codes[row] = code
    return codes


def state(code: int) -> int:
    return UNKNOWN if code == ABSTAINED else code


def both(a: int, b: int) -> int:
    if a == OUT or b == OUT:
        return OUT
    return IN if a == IN and b == IN else UNKNOWN


def either(a: int, b: int) -> int:
    if a == IN or b == IN:
        return IN
    return OUT if a == OUT and b == OUT else UNKNOWN


def negate(a: int) -> int:
    return (IN, OUT, UNKNOWN)[a]


def selection(codes, parent):
    """A producer's codes within `parent`: outside it nothing is evaluated."""
    if parent is None:
        return bytearray(codes)
    return bytearray(c if p == IN else UNKNOWN for c, p in zip(codes, parent))


def view(evaluated, parent):
    """An expression within `parent`: outside it nothing is in."""
    if parent is None:
        return bytearray(evaluated)
    return bytearray(
        OUT if state(p) == OUT else both(state(p), e) for e, p in zip(evaluated, parent)
    )


def tally(codes, parent, first, count):
    """Counts among the members of `parent`, what lies outside it, and the digest
    over every original that is not out."""
    originals = [0, 0, 0]
    occurrences = [0, 0, 0]
    abstained = [0, 0]
    outside = [0, 0]
    total = 0
    for row, code in enumerate(codes):
        s = state(code)
        if parent is None or parent[row] == IN:
            originals[s] += 1
            occurrences[s] += count[row]
            if code == ABSTAINED:
                abstained[0] += 1
                abstained[1] += count[row]
        else:
            outside[0] += 1
            outside[1] += count[row]
        if s != OUT:
            part = hashlib.sha256(first[row].to_bytes(8, "little") + bytes([s])).digest()[:16]
            total = (total + int.from_bytes(part, "little")) & MASK
    return {
        "revision": f"{total:032x}",
        "counts": {"parent": sum(originals), "in": originals[IN], "out": originals[OUT],
                   "unknown": originals[UNKNOWN]},
        "occurrence_counts": {"parent": sum(occurrences), "in": occurrences[IN],
                              "out": occurrences[OUT], "unknown": occurrences[UNKNOWN]},
        "abstained": abstained,
        "outside_scope": outside,
    }


def ordered(codes, first, limit=None):
    """The record IDs of the members in order, as a digest with its ends."""
    digest = hashlib.sha256()
    ids = []
    members = 0
    for row, code in enumerate(codes):
        if code == IN:
            members += 1
            if limit is None or members <= limit:
                handle = record_id(first[row])
                digest.update(handle.encode() + b"\n")
                ids.append(handle)
    return {"members": members, "listed": len(ids), "sha256": digest.hexdigest(),
            "first": ids[:3], "last": ids[-3:]}


def regions(names, sets, parent, count):
    table = {}
    for row in range(len(count)):
        if parent is not None and parent[row] != IN:
            continue
        key = "/".join(("out", "in", "not_evaluated")[state(sets[name][row])] for name in names)
        cell = table.setdefault(key, [0, 0])
        cell[0] += 1
        cell[1] += count[row]
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--entry", action="append", default=None)
    parser.add_argument("--max-record-bytes", type=int, default=4096)
    parser.add_argument("--scores", type=Path, help="work directory of `l2d baseline score`")
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--threshold-rank", type=int, default=20000)
    parser.add_argument("--top-k-within-b", type=int, default=500)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    patterns = args.entry or ["*.log"]
    started = time.monotonic()

    first, count, flags, report = read_archive(args.zip, patterns, args.max_record_bytes)
    if any(first[at] >= first[at + 1] for at in range(len(first) - 1)):
        raise SystemExit("the first positions are not in order of first occurrence")
    size = len(first)
    occurrences_total = sum(count)
    sets = {
        "b": bytearray(flag & 1 for flag in flags),
        "b_regex": bytearray(flag >> 1 & 1 for flag in flags),
        "b_literal": bytearray(flag >> 2 & 1 for flag in flags),
        "c": read_membership(args.membership, first),
    }
    parents = {name: None for name in sets}
    oracle = {
        "source": {"zip": args.zip.name, "entries": patterns,
                   "max_record_bytes": args.max_record_bytes,
                   "sha256": hashlib.file_digest(args.zip.open("rb"), "sha256").hexdigest()},
        "counts": report,
        "occurrences": occurrences_total,
        "filters": {"b": {"syntax": "glob", "query": "*login*"},
                    "b_regex": {"syntax": "regex", "query": REGEX},
                    "b_literal": {"syntax": "literal", "query": LITERAL}},
        "snapshot_first": {"listed": min(size, 2500),
                           "sha256": hashlib.sha256("".join(
                               record_id(first[row]) + "\n" for row in range(min(size, 2500))
                           ).encode()).hexdigest()},
    }
    # C within B: what the same membership gives inside a population.
    sets["c_in_b"] = selection(sets["c"], sets["b"])
    parents["c_in_b"] = sets["b"]

    if args.scores:
        scores = read_scores(args.scores, first)
        finite = [row for row in range(size) if math.isfinite(scores[row])]
        ranked = sorted(finite, key=lambda row: (-scores[row], first[row]))
        rank = {row: place + 1 for place, row in enumerate(ranked[: 4 * args.threshold_rank])}
        base = bytearray(OUT if math.isfinite(scores[row]) else UNKNOWN for row in range(size))
        top = bytearray(base)
        for row in ranked[: args.top_k]:
            top[row] = IN
        threshold = scores[ranked[min(args.threshold_rank, len(ranked)) - 1]]
        above = bytearray(base)
        for row in ranked:
            if scores[row] < threshold:
                break
            above[row] = IN
        sets["a"], parents["a"] = top, None
        sets["a_threshold"], parents["a_threshold"] = above, None
        # The top-k of the same scorer among the members of B.
        within = [row for row in ranked if sets["b"][row] == IN]
        a_in_b = bytearray(
            (OUT if math.isfinite(scores[row]) else UNKNOWN) if sets["b"][row] == IN else UNKNOWN
            for row in range(size)
        )
        for row in within[: args.top_k_within_b]:
            a_in_b[row] = IN
        sets["a_in_b"], parents["a_in_b"] = a_in_b, sets["b"]
        # The reductions.
        a, b, c = sets["a"], sets["b"], sets["c"]
        sets["d"] = view(bytearray(both(x, negate(y)) for x, y in zip(a, b)), None)
        parents["d"] = None
        sets["e"] = view(
            bytearray(both(state(x), state(y)) for x, y in zip(sets["d"], c)), sets["d"]
        )
        parents["e"] = sets["d"]
        sets["m"] = view(
            bytearray(both(either(x, y), negate(state(z))) for x, y, z in zip(a, b, c)), None
        )
        parents["m"] = None
        sets["g"] = view(
            bytearray(both(x, y) for x, y in zip(sets["a_threshold"], sets["b_regex"])), None
        )
        parents["g"] = None
        oracle["score_rules"] = {
            "top_k": args.top_k, "threshold": threshold, "threshold_rank": args.threshold_rank,
            "top_k_within_b": args.top_k_within_b, "scored": len(finite),
            "distinct_scores": len({scores[row] for row in finite}),
        }
        oracle["regions"] = {
            "snapshot_abc": regions(["a", "b", "c"], sets, None, count),
            "d_c": regions(["c"], sets, sets["d"], count),
        }
        members = [row for row in range(size) if sets["d"][row] == IN]
        picks = sorted({members[0], members[len(members) // 2], members[-1]})
        oracle["evidence"] = [
            {"id": record_id(first[row]), "occurrences": count[row], "score": scores[row],
             "rank": rank[row], "ranked_among": len(finite),
             "membership": {name: ("out", "in", "not_evaluated")[state(sets[name][row])]
                            for name in ("a", "b", "c")}}
            for row in picks
        ]
        oracle["ordered"] = {name: ordered(sets[name], first)
                             for name in ("a", "d", "e", "m", "a_in_b", "g")}
    oracle.setdefault("ordered", {})["b_first"] = ordered(sets["b"], first, 2500)
    oracle["ordered"]["c"] = ordered(sets["c"], first)
    oracle["sets"] = {
        name: tally(codes, parents[name], first, count) for name, codes in sets.items()
    }
    oracle["oracle_seconds"] = round(time.monotonic() - started, 1)
    args.out.write_text(json.dumps(oracle, indent=1) + "\n")
    summary = {name: found["counts"] for name, found in oracle["sets"].items()}
    print(json.dumps({"counts": {key: report[key] for key in (
        "records", "accepted_occurrences", "skipped_records", "distinct_originals",
        "entries_read", "entries_excluded", "entries_unsupported")}, "sets": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
