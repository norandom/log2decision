"""Prints the dataset figures the splits rest on, and the folds.

Usage: uv run python describe_data.py <csic_database.csv>
"""

import sys
from collections import Counter

import numpy as np

import l2dsci as s

data = s.read_csic(sys.argv[1])
print(f"sha256 {data.sha256}")
print(f"records {data.records}  distinct texts {len(data.texts)}  good {(~data.bad).sum()}  bad {data.bad.sum()}")
print(f"path groups {len(set(data.groups))}  with both labels {len(data.mixed_groups)}")
groups = np.array(data.groups)
for mode in ("keep", "strip"):
    ids, truncated = s.byte_ids(data.texts, mode)
    lengths = (ids != 0).sum(axis=1)
    print(f"{mode}: truncated at 256 bytes {truncated.sum()} ({truncated.mean():.1%}); empty input {(lengths == 0).sum()}; "
          f"median length {int(np.median(lengths))}; distinct inputs {len({r.tobytes() for r in ids})}")
    by_input: dict[bytes, set] = {}
    for row, b in zip(ids, data.bad):
        by_input.setdefault(row.tobytes(), set()).add(bool(b))
    print(f"{mode}: inputs that occur with both labels {sum(len(v) == 2 for v in by_input.values())}")
fold = s.cv5_fold_of_group(data)
held17 = s.held_out_groups(data, "split17", 0)
for g in data.mixed_groups:
    m = groups == g
    print(f"mixed {g}: good {(~data.bad[m]).sum()} bad {data.bad[m].sum()} cv5 fold {fold[g]} split17 held out {g in held17}")
for f in range(5):
    held = s.held_out_groups(data, "cv5", f)
    m = np.isin(groups, sorted(held))
    val = s.validation_groups(data, held, f)
    v = np.isin(groups, sorted(val))
    print(f"cv5 fold {f}: held-out logs {m.sum()} (good {(~data.bad[m]).sum()}, bad {data.bad[m].sum()}), groups {len(held)}; "
          f"validation mixed path {sorted(set(val) & set(data.mixed_groups))} logs {v.sum()}")
m = np.isin(groups, sorted(held17))
val = s.validation_groups(data, held17, 0)
print(f"split17: held-out logs {m.sum()} fitting {(~m).sum()}; validation mixed path {sorted(set(val) & set(data.mixed_groups))}")
print(Counter(len(t.encode()) > 256 for t in data.texts))
