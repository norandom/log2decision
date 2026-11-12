"""Copies the models of chosen runs into a `models/science-2.1/` directory and writes
`record.json` beside them: data hash, split, seed, settings, metrics, package versions,
duration, and the SHA-256 of every file.

Usage: uv run python publish.py <runs dir> <target dir> --run <name> [--run <name> ...]
           [--summary <summary.json> ...]
"""

from __future__ import annotations

import json
import os
import shutil
import sys

from l2dsci import file_sha256


def main() -> None:
    runs_dir, target = sys.argv[1], sys.argv[2]
    names = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--run"]
    summaries = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--summary"]
    os.makedirs(target, exist_ok=True)
    record = {"task": "npu-log-decision 2.1 (science gate)", "models": [], "summaries": []}
    for name in names:
        with open(os.path.join(runs_dir, f"{name}.json"), encoding="utf-8") as handle:
            run = json.load(handle)
        files = {}
        for kind, suffix in (("float_onnx", "f32.onnx"), ("int8_onnx", "int8.onnx"), ("table", "table.bin")):
            source = os.path.join(runs_dir, f"{name}.{suffix}")
            copy = os.path.join(target, f"{name}.{suffix}")
            shutil.copyfile(source, copy)
            digest = file_sha256(copy)
            if digest != run["export"]["sha256"][kind]:
                raise SystemExit(f"{copy}: hash differs from the run record")
            files[os.path.basename(copy)] = {"sha256": digest, "bytes": os.path.getsize(copy)}
        run["files"] = files
        record["models"].append(run)
    for path in summaries:
        with open(path, encoding="utf-8") as handle:
            record["summaries"].append(json.load(handle))
    all_runs = [f for f in os.listdir(runs_dir) if f.endswith(".json")]
    seconds = 0.0
    for f in all_runs:
        with open(os.path.join(runs_dir, f), encoding="utf-8") as handle:
            seconds += json.load(handle).get("seconds", 0.0)
    record["all_runs"] = {"count": len(all_runs), "seconds_total": seconds, "directory": runs_dir}
    out = os.path.join(target, "record.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    for name, info in sorted((n, i) for m in record["models"] for n, i in m["files"].items()):
        print(f"{info['sha256']}  {name}")
    print(f"{file_sha256(out)}  record.json")


if __name__ == "__main__":
    main()
