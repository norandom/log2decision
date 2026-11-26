"""Generate the deterministic multi-million-record fixture of the triager scale gate.

Triager task 1.3 (`.kiro/specs/triager/performance.md`, "Proposed scale gate"): at
least 4 million occurrences of at least 2 million distinct originals across several
archive entries, with the same text in several entries (the time bins), records that
miss a field, skipped records, an entry that is no text, an entry outside the entry
filter, and a membership file with deliberately partial coverage that stands in for C.

Everything follows from `--seed` and `--scale`; nothing is random. The fixture is not
committed: `manifest.json` beside this script records what the full scale gives.

    python generate.py --out D:\\BuildArtifacts\\log2decision\\triager-1.3\\fixture

Standard library only. `oracle.py` does not import this file: it reads the archive.
"""

import argparse
import array
import hashlib
import json
import platform
import sys
import time
import zipfile
import zlib
from pathlib import Path

SCALES = {
    # entries (time bins), record slots per entry
    "full": (8, 550_000),
    "small": (4, 6_000),
}
FIXED_TIME = (2026, 1, 1, 0, 0, 0)
LINE_BITS = 40

EVENTS = [
    ("auth", "login_ok"), ("auth", "login_failed"), ("auth", "login_locked"),
    ("auth", "session_open"), ("auth", "session_close"),
    ("web", "http_get"), ("web", "http_get"), ("web", "http_post"),
    ("web", "http_get"), ("web", "http_post"),
    ("edge", "challenge_rejected"), ("sched", "cron_run"), ("node", "disk_warn"),
    ("app", "trace"), ("app", "trace"), ("app", "trace"),
]
STATUS = [200, 200, 200, 404, 500, 302]


def text_of(key: int) -> str:
    """The text of the original `key`; two keys never give one text."""
    svc, event = EVENTS[key % 16]
    parts = [f"host=h{key % 211:03d}", f"svc={svc}", f"event={event}"]
    if key % 11:  # every eleventh record misses the field
        parts.append(f"user=u{key * 7 % 9973:04d}")
    parts.append(f"src=10.{key >> 16 & 255}.{key >> 8 & 255}.{key & 255}")
    parts.append(f"req={key:07x}")
    if svc == "web":
        attack = "?id=1%27%20OR%201=1--" if key % 97 == 0 else ""
        parts.append(f"path=/api/v{key % 3}/item/{key % 1000}{attack}")
        parts.append(f"status={STATUS[key % 6]}")
        parts.append(f"ms={key % 977}")
    elif event == "login_failed":
        parts.append(f"attempt={key % 9}")
    elif event == "disk_warn":
        parts.append(f"free_pct={key % 23}")
    return " ".join(parts)


def stamped(key: int, day: int, text: str) -> str:
    """Triager task 3.1, `--timestamps`: the record of the original `key` first read on
    day `day` of January 2026, with a leading ISO 8601 time that varies within the day.
    One original in 13 has none (missing), one in 29 has one without a zone (invalid,
    as the service reads timestamps). A repeat is the same text, so it carries the time
    of its original: the entries it is read from are other days."""
    if key % 13 == 5:
        return text
    seconds = key * 7_919 % 86_400
    clock = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    if key % 29 == 7:
        return f"2026-01-{day:02d} {clock} {text}"
    return f"2026-01-{day:02d}T{clock}Z {text}"


def record(key: int, state: dict) -> str:
    """The text of the original `key` as the archive holds it."""
    if not state["timestamps"]:
        return text_of(key)
    first = state["first"]
    day = first[key] >> LINE_BITS if key < len(first) else state["late_entry"]
    return stamped(key, day, text_of(key))


def fixture_state(key: int):
    """What the stand-in for C says of the original `key`; None is not assessed."""
    if key % 4 == 3 or key % 16 not in (0, 1, 2, 5, 6, 7, 8, 9, 10):
        return None
    return ("in", "abstain")[key % 7] if key % 7 < 2 else "out"


def entry_lines(entry: int, slots: int, seed: int, state: dict):
    """Yields the physical lines of time bin `entry` (1-based) as bytes, and notes
    the first position of every new original and what was skipped."""
    slot = 0
    line = 0
    while slot < slots:
        line += 1
        if line % 100_003 == 0 or (line == 1_917 and entry == 1):
            state["skipped"]["empty line"] += 1
            yield b""
            continue
        # Far behind the first 8 KiB of the entry, which decide whether it is text.
        if line % 250_007 == 0 or (line == 2_917 and entry == 2):
            state["skipped"]["line is not valid UTF-8"] += 1
            yield b"\xff\xfe not text"
            continue
        if line == 123_457 % (2 * slots) and entry in (2, 3):
            state["skipped"]["longer than the maximum record length"] += 1
            yield b"x" * 5_000
            continue
        if slot % 2 == 0:
            key = state["keys"]
            state["keys"] += 1
            state["first"].append(entry << LINE_BITS | line)
        else:
            # A repeat of an original of this or an earlier time bin.
            key = (slot * 2_654_435_761 + entry * 40_503 + seed) % state["keys"]
        slot += 1
        yield record(key, state).encode()


def write_entry(archive: zipfile.ZipFile, name: str, lines) -> int:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    count = 0
    with archive.open(info, "w", force_zip64=True) as out:
        chunk = []
        for line in lines:
            chunk.append(line)
            count += 1
            if len(chunk) == 20_000:
                out.write(b"\n".join(chunk) + b"\n")
                chunk.clear()
        if chunk:
            out.write(b"\n".join(chunk) + b"\n")
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scale", choices=SCALES, default="full")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--timestamps", action="store_true",
                        help="task 3.1: a leading ISO 8601 time per record and entries named by day")
    args = parser.parse_args()
    bins, slots = SCALES[args.scale]
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    state = {
        "keys": 0,
        "timestamps": args.timestamps,
        "late_entry": bins + 2,
        "first": array.array("Q"),
        "skipped": {
            "empty line": 0,
            "line is not valid UTF-8": 0,
            "longer than the maximum record length": 0,
        },
    }
    names = ["notes/README.txt"]
    if args.timestamps:
        names += [f"logs/2026-01-{day:02d}.log" for day in range(1, bins + 1)]
    else:
        names += [f"logs/day-{day:02d}.log" for day in range(1, bins + 1)]
    names += ["logs/core-dump.log", "logs/late.log"]
    archive_path = args.out / "fixture.zip"
    physical = {}
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        # Entry 0 is outside the entry filter `*.log`.
        physical[names[0]] = write_entry(
            archive, names[0], [b"The fixture of the triager scale gate.", b"login notes"]
        )
        for day in range(1, bins + 1):
            physical[names[day]] = write_entry(
                archive, names[day], entry_lines(day, slots, args.seed, state)
            )
        # No text: a NUL byte within the first 8 KiB.
        core = bytes(range(256)) * 64
        info = zipfile.ZipInfo(names[bins + 1], date_time=FIXED_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, core)
        # A last, short text entry after it: three repeats and one new original.
        late = [record(0, state).encode(), record(1, state).encode(), record(0, state).encode()]
        late.append(record(state["keys"], state).encode())
        state["first"].append((bins + 2) << LINE_BITS | 4)
        state["keys"] += 1
        physical[names[bins + 2]] = write_entry(archive, names[bins + 2], late)

    # The stand-in for C, keyed by record handle, in descending order of the keys:
    # JSON lines, one {"id", "state"} object a line (task 2.2; the tab-separated form
    # of task 1.3 is no longer read).
    membership_path = args.out / "membership-c.jsonl"
    assessed = {"in": 0, "out": 0, "abstain": 0}
    with membership_path.open("w", newline="\n") as out:
        for key in range(state["keys"] - 1, -1, -1):
            outcome = fixture_state(key)
            if outcome is None:
                continue
            assessed[outcome] += 1
            out.write(f'{{"id":"r-{state["first"][key]:016x}","state":"{outcome}"}}\n')

    # What a baseline scorer is fitted on: texts of keys the archive does not hold.
    labelled_path = args.out / "labelled.csv"
    with labelled_path.open("w", newline="\n") as out:
        out.write("text,label\n")
        base = 16_000_000
        good = bad = 0
        key = base
        while good < 600 or bad < 600:
            text = text_of(key)
            is_bad = "%27" in text or "login_failed" in text or "challenge" in text
            if is_bad and bad < 600:
                out.write(f"{text},bad\n")
                bad += 1
            elif not is_bad and good < 600:
                out.write(f"{text},good\n")
                good += 1
            key += 1

    skipped = sum(state["skipped"].values())
    accepted = bins * slots + 4
    manifest = {
        "fixture": "triager scale gate, task 1.3",
        "scale": args.scale,
        **({"timestamps": True} if args.timestamps else {}),
        "seed": args.seed,
        "generator": {
            "python": platform.python_version(),
            "zlib": zlib.ZLIB_VERSION,
            "note": "the archive's bytes depend on the zlib that deflated it; its "
                    "content does not",
        },
        "entries": names,
        "entry_filter": ["*.log"],
        "max_record_bytes": 4096,
        "by_construction": {
            "entries_selected_text": bins + 1,
            "entries_excluded": 1,
            "entries_unsupported": 1,
            "records": accepted + skipped,
            "accepted_occurrences": accepted,
            "skipped_records": skipped,
            "skipped_by_reason": state["skipped"],
            "distinct_originals": state["keys"],
            "membership_c": assessed,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (archive_path, membership_path, labelled_path)
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["by_construction"]))
    print(f"written to {args.out} in {time.monotonic() - started:.1f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
