"""Compute the expected results of the triager scale gate without the product.

Reads the fixture archive with Python's `zipfile`, splits and judges its lines as
`.kiro/specs/triager/performance.md` and the archive reader's documentation state
them, deduplicates the texts, and derives every count the scale run compares:
accepted and skipped records, distinct originals, the membership of the filter sets,
the membership file that stands in for C with its unknown states, the eight regions
of A, B and C with the unknown combinations, the reductions D = A - B, E = D and C
within D and "what C missed", their ordered record IDs, and the digest (`revision`)
the service states for each membership. Triager task 2.2 adds the final allocation of
one budget (`BUDGET`) over the candidate D ∩ C: the record IDs of its slots in order
with their pool, and the counts of every pool and of the deferrals in originals and
occurrences.

A is a score rule. The scores are not recomputed here: they are read from the parts
the existing command `l2d baseline score` wrote (`--scores`), and the top-k rule with
its tie rule, the threshold rule, the ranks and everything composed from A are
computed here from those scores. Without `--scores` everything that needs A is left
out.

    uv run --no-project --with-requirements requirements.txt python oracle.py \\
        --zip fixture.zip --membership membership-c.jsonl --scores scored --out oracle.json

`polars` is used for one thing: reading the Parquet parts of `--scores`.

This file imports nothing of `generate.py` and nothing of the product.
"""

import argparse
import array
import bisect
import calendar
import fnmatch
import hashlib
import json
import math
import re
import struct
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
    """JSON lines: one {"id": record handle, "state": in|out|abstain} object a line."""
    codes = bytearray([UNKNOWN]) * len(first)
    with path.open("rb") as lines:
        for number, raw in enumerate(lines, start=1):
            line = json.loads(raw)
            if not isinstance(line, dict) or set(line) != {"id", "state"}:
                raise SystemExit(f"line {number} of {path} is no membership line")
            handle, outcome = line["id"], line["state"]
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


def audit_key(handle: str, seed: int) -> int:
    """`l2d_decide::policy::audit_key`: 64-bit FNV-1a whose basis is XOR `seed`."""
    value = 0xCBF29CE484222325 ^ seed
    for byte in handle.encode():
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def as_f32(value: float) -> float:
    """The score as the server keeps it: rounded to the nearest f32."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


# The budget the allocation leg submits: three named pools over D and C, the seed of
# the fixture for the audit. `run.py` reads it from the oracle.
ALLOCATION_SEED = 20261125
BUDGET = {
    "total": 200, "duplicate_rule": "first_pool_wins", "shortfall": "leave_unfilled",
    "pools": [
        {"name": "general", "limit": 120, "from": {"intersection": ["d", "c"]},
         "order": ["score_desc", "record_id_ascending"]},
        {"name": "unfamiliar", "limit": 40,
         "from": {"intersection": ["d", {"state": {"set": "c", "is": "not_evaluated"}}]},
         "order": ["record_id_ascending"]},
        {"name": "audit", "limit": 40, "from": {"difference": ["d", "c"]},
         "order": ["seeded_hash", "record_id_ascending"], "seed": ALLOCATION_SEED},
    ],
}


def allocation(sets, scores, first, count):
    """The final work set of BUDGET over the candidate D ∩ C, from the codes alone:
    every pool in order takes its first `limit` members in its own order that no
    pool before it took (first pool wins; a shortfall stays unfilled)."""
    d = [state(code) for code in sets["d"]]
    c = [state(code) for code in sets["c"]]
    candidate = [both(x, y) for x, y in zip(d, c)]
    populations = [
        [both(x, y) for x, y in zip(d, c)],
        [both(x, IN if y == UNKNOWN else OUT) for x, y in zip(d, c)],
        [both(x, negate(y)) for x, y in zip(d, c)],
    ]
    keys = [
        lambda row: (-as_f32(scores[row]), first[row]),
        lambda row: first[row],
        lambda row: (audit_key(record_id(first[row]), ALLOCATION_SEED), first[row]),
    ]
    taken = {}
    slots = []
    pools = []
    for index, (pool, population, key) in enumerate(zip(BUDGET["pools"], populations, keys)):
        members = sorted((row for row, code in enumerate(population) if code == IN), key=key)
        picked = []
        for row in members:
            if len(picked) == pool["limit"]:
                break
            if row not in taken:
                taken[row] = index
                picked.append(row)
        slots.extend((row, pool["name"]) for row in picked)
        pools.append({"name": pool["name"], "members": members, "picked": picked})
    outcomes = []
    for index, pool in enumerate(pools):
        members = pool["members"]
        deferred = [row for row in members if row not in taken]
        outcomes.append({
            "name": pool["name"], "limit": BUDGET["pools"][index]["limit"],
            "selected": len(pool["picked"]),
            "eligible": {"distinct_originals": len(members),
                         "occurrences": sum(count[row] for row in members)},
            "taken_by_other_pools": sum(1 for row in members if taken.get(row, index) != index),
            "deferred": {"distinct_originals": len(deferred),
                         "occurrences": sum(count[row] for row in deferred)},
            "outside_candidate": sum(1 for row in pool["picked"] if candidate[row] != IN),
        })
    eligible = [row for row in range(len(first))
                if candidate[row] == IN or any(p[row] == IN for p in populations)]
    deferred = [row for row in eligible if row not in taken]
    final = bytearray(OUT for _ in first)
    for row, _ in slots:
        final[row] = IN
    digest = hashlib.sha256()
    for row, pool in slots:
        digest.update(f"{record_id(first[row])} {pool}\n".encode())
    unknown = [row for row in range(len(first)) if candidate[row] == UNKNOWN]
    found = {
        "budget": BUDGET,
        "capacity": BUDGET["total"], "selected": len(slots),
        "unused": BUDGET["total"] - len(slots),
        "pools": outcomes,
        "candidate": {"distinct_originals": candidate.count(IN),
                      "occurrences": sum(count[row] for row, code in enumerate(candidate)
                                         if code == IN),
                      "not_evaluated": {"distinct_originals": len(unknown),
                                        "occurrences": sum(count[row] for row in unknown)}},
        "deferred": {"distinct_originals": len(deferred),
                     "occurrences": sum(count[row] for row in deferred)},
        "outside_candidate": sum(outcome["outside_candidate"] for outcome in outcomes),
        "slots": {"listed": len(slots), "sha256": digest.hexdigest(),
                  "first": [[record_id(first[row]), pool] for row, pool in slots[:3]],
                  "last": [[record_id(first[row]), pool] for row, pool in slots[-3:]]},
    }
    return found, final


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
        # The allocation of one budget over D and C: the final set, ordered by slot.
        oracle["allocation"], sets["f"] = allocation(sets, scores, first, count)
        parents["f"] = None
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


# --- Triager task 3.1: field, time and record views -------------------------------------
#
# `python oracle.py facets --zip fixture.zip --out facets.json` reads every accepted
# occurrence (not only the first of each text), attributes it to its original by text,
# and computes what the fields, histogram and drill views of the service state: for
# every projected field the occurrences and distinct originals with a value, missing
# and beyond the retained limit (the first `MAX_VALUES` values in read order), the
# first page of values, the occurrences by archive entry (the no-time alternative: the
# fixture's records carry no timestamp), the drills the scale leg makes and the
# occurrences of one repeated original. A field's value is the `key=value` token of the
# record split at spaces, which is what Splunk's automatic extraction gives on the
# fixture's records; nothing of the product is imported.

FIELDS = ("event", "host", "src", "svc", "user")
RESCAN_FIELD = "status"
MAX_VALUES = 10_000
MAX_VALUE_BYTES = 256
PAGE = 25
NUMBER = re.compile(r"^[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)$")


def tokens(text: str) -> dict:
    found = {}
    for part in text.split(" "):
        key, sep, value = part.partition("=")
        if sep and value and key not in found:
            found[key] = value
    return found


def read_occurrences(path: Path, patterns, max_record_bytes: int, times=None):
    """Every accepted occurrence: the original it repeats (by first position order) and
    its entry, with the values of every field per original, interned per field in read
    order up to `MAX_VALUES` (`-1` missing, `-2` beyond the limit, `-3` too long)."""
    index = {}
    first = array.array("Q")
    count = array.array("L")
    occ_original = array.array("L")
    occ_entry = array.array("H")
    names = {}
    fields = (*FIELDS, RESCAN_FIELD)
    ids = {field: {} for field in fields}
    values_of = {field: array.array("l") for field in fields}
    report = {"records": 0, "accepted_occurrences": 0}
    with zipfile.ZipFile(path) as archive:
        for entry, info in enumerate(archive.infolist()):
            name = info.filename
            if info.is_dir() or not any(fnmatch.fnmatchcase(name, p) for p in patterns):
                continue
            with archive.open(info) as member:
                head = member.read(8192)
            text_head = b"\0" not in head
            if text_head:
                try:
                    head.decode("utf-8")
                except UnicodeDecodeError as error:
                    text_head = error.start >= len(head) - 3 and len(head) == 8192
            if not text_head:
                continue
            names[entry] = name
            with archive.open(info) as member:
                for number, raw in enumerate(member, start=1):
                    report["records"] += 1
                    line = raw[:-1] if raw.endswith(b"\n") else raw
                    if line.endswith(b"\r"):
                        line = line[:-1]
                    if not line or len(line) > max_record_bytes:
                        continue
                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    report["accepted_occurrences"] += 1
                    digest = hashlib.blake2b(line, digest_size=16).digest()
                    at = index.get(digest)
                    if at is None:
                        at = len(first)
                        index[digest] = at
                        first.append(entry << LINE_BITS | number)
                        count.append(0)
                        found = tokens(text)
                        for field in fields:
                            value = found.get(field)
                            if value is None:
                                code = -1
                            elif len(value.encode()) > MAX_VALUE_BYTES:
                                code = -3
                            elif value in ids[field]:
                                code = ids[field][value]
                            elif len(ids[field]) < MAX_VALUES:
                                code = ids[field][value] = len(ids[field])
                            else:
                                code = -2
                            values_of[field].append(code)
                        if times is not None:
                            times.append(leading_time(text))
                    count[at] += 1
                    occ_original.append(at)
                    occ_entry.append(entry)
    del index
    report["distinct_originals"] = len(first)
    return first, count, occ_original, occ_entry, names, ids, values_of, report


def pair(originals, occurrences):
    return {"distinct_originals": originals, "occurrences": occurrences}


def facets_main(argv) -> int:
    parser = argparse.ArgumentParser(prog="oracle.py facets")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--entry", action="append", default=None)
    parser.add_argument("--max-record-bytes", type=int, default=4096)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    patterns = args.entry or ["*.log"]
    started = time.monotonic()
    first, count, occ_original, occ_entry, names, ids, values_of, report = read_occurrences(
        args.zip, patterns, args.max_record_bytes)
    size = len(first)
    total = len(occ_original)
    value_text = {field: {code: value for value, code in ids[field].items()} for field in ids}

    def summary(field, members=None):
        """The fields-view figures of `field` over the originals `members` (all)."""
        codes = values_of[field]
        with_value = missing = invalid = overflow = number = string = (0, 0)
        seen = set()

        def add(sum_, original):
            return sum_[0] + 1, sum_[1] + count[original]

        for original in range(size) if members is None else members:
            code = codes[original]
            if code == -1:
                missing = add(missing, original)
            elif code == -3:
                invalid = add(invalid, original)
            else:
                with_value = add(with_value, original)
                if code == -2:
                    overflow = add(overflow, original)
                else:
                    seen.add(code)
                    if NUMBER.match(value_text[field][code].strip()):
                        number = add(number, original)
                    else:
                        string = add(string, original)
        return {"with_value": pair(*with_value), "missing": pair(*missing),
                "invalid": pair(*invalid), "overflow": pair(*overflow),
                "types": {"number": pair(*number), "string": pair(*string)},
                "distinct_values": len(seen), "exact": overflow[1] == 0}

    def first_page(field, members=None, occurrences_of=None):
        """The first page of values: most occurrences first, then value ascending. A
        text-derived value is the same for every occurrence of an original, so an
        original counts once with all its counted occurrences."""
        codes = values_of[field]
        per = {}
        for original in range(size) if members is None else members:
            code = codes[original]
            if code < 0:
                continue
            counted = count[original] if occurrences_of is None else occurrences_of[original]
            originals, occurrences = per.get(code, (0, 0))
            per[code] = (originals + 1, occurrences + counted)
        ranked = sorted(per.items(), key=lambda item: (-item[1][1], value_text[field][item[0]]))
        return {"values_total": len(per),
                "rows": [{"value": value_text[field][code], **pair(*figures)}
                         for code, figures in ranked[:PAGE]]}

    # The snapshot: every field, the first pages and the entries.
    out = {
        "source": {"zip": args.zip.name, "entries": patterns,
                   "max_record_bytes": args.max_record_bytes},
        "counts": {**report, "occurrences": total},
        "fields": {field: summary(field) for field in FIELDS},
        "pages": {field: first_page(field) for field in FIELDS},
    }
    entries_seen = {}
    last_entry = array.array("h", [-1]) * size
    for at, original in enumerate(occ_original):
        entry = occ_entry[at]
        figures = entries_seen.setdefault(entry, [0, 0])
        figures[1] += 1
        if last_entry[original] != entry:
            last_entry[original] = entry
            figures[0] += 1
    out["entries"] = [{"entry": names[entry], "entry_index": entry, **pair(*entries_seen[entry])}
                      for entry in sorted(entries_seen)]
    out["source_field"] = {"distinct_values": len(entries_seen)}

    # Drill 1: the occurrences of one entry, the time bin of the fixture.
    day = next(entry for entry, name in names.items() if name == "logs/day-03.log")
    in_day = array.array("L", [0]) * size
    for at, original in enumerate(occ_original):
        if occ_entry[at] == day:
            in_day[original] += 1
    members = [original for original in range(size) if in_day[original]]
    out["drill_day"] = {
        "entry": "logs/day-03.log",
        "counts_in": len(members),
        "occurrence_counts_in": sum(count[original] for original in members),
        "matching_in": sum(in_day[original] for original in members),
        "matching_parent": total,
        "user_page": first_page("user", members, in_day),
        "user_summary_counted": pair(len(members), sum(in_day[o] for o in members)),
    }
    # Drill 2, below it: the most frequent user of the day.
    top_user = out["drill_day"]["user_page"]["rows"][0]["value"]
    code = ids["user"][top_user]
    nested = [original for original in members if values_of["user"][original] == code]
    out["drill_day_user"] = {
        "user": top_user,
        "counts_in": len(nested),
        "counts_parent": len(members),
        "matching_in": sum(in_day[original] for original in nested),
        "matching_parent": out["drill_day"]["matching_in"],
    }
    # Drill 3: a host of the snapshot.
    host = ids["host"]["h042"]
    hosted = [original for original in range(size) if values_of["host"][original] == host]
    out["drill_host"] = {"host": "h042", "counts_in": len(hosted),
                         "matching_in": sum(count[original] for original in hosted)}
    # The evidence of one repeated original of the day: every occurrence by position.
    repeated = next(original for original in members if count[original] >= 3
                    and in_day[original] < count[original])
    positions = []
    for at, original in enumerate(occ_original):
        if original == repeated:
            positions.append(occ_entry[at])
    out["evidence"] = {"id": record_id(first[repeated]), "total": count[repeated],
                       "in_view": in_day[repeated],
                       "entries": [names[entry] for entry in positions]}
    # The re-scan: the field the first projection does not declare.
    out["rescan"] = {"field": RESCAN_FIELD, "summary": summary(RESCAN_FIELD),
                     "page": first_page(RESCAN_FIELD)}
    out["oracle_seconds"] = round(time.monotonic() - started, 1)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"counts": out["counts"], "drill_day": {
        key: out["drill_day"][key] for key in ("counts_in", "matching_in")},
        "seconds": out["oracle_seconds"]}))
    return 0


# --- Triager task 3.1, round 2: the timestamped fixture (`generate.py --timestamps`) ---
#
# `python oracle.py time --zip fixture.zip --out time.json` computes the histogram of
# the records' leading timestamps and of the entries' days, the drills of the scale leg
# and their counts in both units, from the archive alone. A record's timestamp is the
# leading `YYYY-MM-DDTHH:MM:SSZ` token; one that begins like a date and is not that is
# invalid (the generator writes `YYYY-MM-DD HH:MM:SS` without a zone), any other record
# has none. An entry's day is the `YYYY-MM-DD` of its name; `logs/late.log` has none.

TIME_MISSING = -(1 << 62)
TIME_INVALID = TIME_MISSING + 1
WIDTHS = (1, 60, 300, 900, 3_600, 21_600, 86_400, 604_800, 2_592_000, 31_536_000)
STAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z(?: |$)")
DATED = re.compile(r"\d{4}-")
ENTRY_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})\.log$")


def leading_time(text: str) -> int:
    if not DATED.match(text):
        return TIME_MISSING
    found = STAMP.match(text)
    if not found:
        return TIME_INVALID
    return calendar.timegm(tuple(int(part) for part in found.groups()))


def iso(seconds: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def histogram(times, count, selected, width):
    """Buckets of `width` over the originals `selected`: a timestamp read from the text
    is the same for every occurrence of an original, so an original falls in one bucket
    with all its occurrences."""
    valid = [o for o in selected if times[o] > TIME_INVALID]
    low = min(times[o] for o in valid)
    high = max(times[o] for o in valid)
    first = low // width * width
    buckets = high // width - low // width + 1
    originals = [0] * buckets
    occurrences = [0] * buckets
    for o in valid:
        at = (times[o] - first) // width
        originals[at] += 1
        occurrences[at] += count[o]
    return {"bucket_seconds": width,
            "range": {"start": iso(first), "end": iso(first + buckets * width)},
            "observed": {"earliest": iso(low), "latest": iso(high)},
            "buckets": [{"start": iso(first + at * width), "distinct_originals": originals[at],
                         "occurrences": occurrences[at]} for at in range(buckets)]}


def auto_width(low, high):
    return next(width for width in WIDTHS if high // width - low // width + 1 <= 100)


def time_main(argv) -> int:
    parser = argparse.ArgumentParser(prog="oracle.py time")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--entry", action="append", default=None)
    parser.add_argument("--max-record-bytes", type=int, default=4096)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    patterns = args.entry or ["*.log"]
    started = time.monotonic()
    times = array.array("q")
    first, count, occ_original, occ_entry, names, ids, values_of, report = read_occurrences(
        args.zip, patterns, args.max_record_bytes, times)
    size = len(first)
    total = len(occ_original)
    user_text = {code: value for value, code in ids["user"].items()}
    every = range(size)

    def classes(selected):
        figures = {"with_time": [0, 0], "missing_time": [0, 0], "invalid_time": [0, 0]}
        for o in selected:
            when = times[o]
            name = ("missing_time" if when == TIME_MISSING
                    else "invalid_time" if when == TIME_INVALID else "with_time")
            figures[name][0] += 1
            figures[name][1] += count[o]
        return {name: pair(*value) for name, value in figures.items()}

    valid = [times[o] for o in every if times[o] > TIME_INVALID]
    width = auto_width(min(valid), max(valid))
    del valid
    out = {"source": {"zip": args.zip.name, "entries": patterns,
                      "max_record_bytes": args.max_record_bytes},
           "counts": {**report, "occurrences": total},
           "text": {"auto": histogram(times, count, every, width),
                    "day": histogram(times, count, every, 86_400), **classes(every)}}
    # The bucket drill: the bucket of the automatic width with the most occurrences, the
    # earliest of them; then the most frequent user in it, and the records without time.
    buckets = out["text"]["auto"]["buckets"]
    chosen = max(range(len(buckets)), key=lambda at: (buckets[at]["occurrences"], -at))
    start = calendar.timegm(time.strptime(buckets[chosen]["start"], "%Y-%m-%dT%H:%M:%SZ"))
    end = start + width
    inside = [o for o in every if start <= times[o] < end]
    per_user = {}
    for o in inside:
        code = values_of["user"][o]
        if code >= 0:
            originals, occurrences = per_user.get(code, (0, 0))
            per_user[code] = (originals + 1, occurrences + count[o])
    ranked = sorted(per_user.items(), key=lambda item: (-item[1][1], user_text[item[0]]))
    top_code = ranked[0][0]
    nested = [o for o in inside if values_of["user"][o] == top_code]
    untimed = [o for o in every if times[o] <= TIME_INVALID]
    inside_occurrences = sum(count[o] for o in inside)
    out["drill_bucket"] = {
        "start": iso(start), "end": iso(end), "counts_in": len(inside),
        "occurrence_counts_in": inside_occurrences,
        "matching_in": inside_occurrences, "matching_parent": total,
        "hourly": histogram(times, count, inside, 3_600)["buckets"],
        "user_page": [{"value": user_text[code], **pair(*figures)}
                      for code, figures in ranked[:PAGE]],
    }
    out["drill_bucket_user"] = {"user": user_text[top_code], "counts_in": len(nested),
                                "counts_parent": len(inside),
                                "matching_in": sum(count[o] for o in nested),
                                "matching_parent": inside_occurrences}
    out["drill_no_time"] = {"counts_in": len(untimed),
                            "matching_in": sum(count[o] for o in untimed)}
    # The entries' days: a time of every occurrence, so a repeated original falls on
    # several days. Every dated entry is one day, so a day's figures are its entry's.
    day_of = {}
    for entry, name in names.items():
        found = ENTRY_DAY.search(name)
        day_of[entry] = (calendar.timegm((*(int(part) for part in found.groups()), 0, 0, 0))
                         if found else TIME_MISSING)
    per_entry = {entry: [0, 0] for entry in names}
    last = array.array("h", [-1]) * size
    days_seen = array.array("B", [0]) * size
    for at in range(total):
        o, entry = occ_original[at], occ_entry[at]
        figures = per_entry[entry]
        figures[1] += 1
        if last[o] != entry:
            last[o] = entry
            figures[0] += 1
            if day_of[entry] != TIME_MISSING and days_seen[o] < 255:
                days_seen[o] += 1
    dated = sorted(entry for entry in names if day_of[entry] != TIME_MISSING)
    low, high = day_of[dated[0]], day_of[dated[-1]]
    days = {"bucket_seconds": 86_400,
            "range": {"start": iso(low), "end": iso(high + 86_400)},
            "buckets": []}
    for day in range(low, high + 86_400, 86_400):
        entry = next((e for e in dated if day_of[e] == day), None)
        figures = per_entry[entry] if entry is not None else [0, 0]
        days["buckets"].append({"start": iso(day), **pair(*figures)})
    late = [entry for entry in names if day_of[entry] == TIME_MISSING]
    days["missing_time"] = pair(sum(per_entry[e][0] for e in late),
                                sum(per_entry[e][1] for e in late))
    # Entries are read in order and each dated one is its own day, so an original read
    # from two dated entries is on two days.
    days["originals_on_several_days"] = sum(1 for o in every if days_seen[o] >= 2)
    # What a day histogram counts beyond the originals with a day: one more for every
    # further day an original is read on.
    days["extra_day_counts"] = sum(days_seen[o] - 1 for o in every if days_seen[o] >= 2)
    out["entry_days"] = days
    # The day drill: the busiest day, its occurrences and their distinct originals.
    rows = days["buckets"]
    busiest = max(range(len(rows)), key=lambda at: (rows[at]["occurrences"], -at))
    entry = next(e for e in dated if iso(day_of[e]) == rows[busiest]["start"])
    members = {occ_original[at] for at in range(total) if occ_entry[at] == entry}
    out["drill_day"] = {"start": rows[busiest]["start"],
                        "end": iso(day_of[entry] + 86_400),
                        "counts_in": len(members), "matching_in": rows[busiest]["occurrences"],
                        "occurrence_counts_in": sum(count[o] for o in members)}
    out["oracle_seconds"] = round(time.monotonic() - started, 1)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"counts": out["counts"], "auto_width": width,
                      "text_classes": {key: out["text"][key] for key in
                                       ("with_time", "missing_time", "invalid_time")},
                      "drill_bucket": out["drill_bucket"]["counts_in"],
                      "several_days": days["originals_on_several_days"],
                      "seconds": out["oracle_seconds"]}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "facets":
        sys.exit(facets_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        sys.exit(time_main(sys.argv[2:]))
    sys.exit(main())
