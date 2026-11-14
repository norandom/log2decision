"""Before the questions: how CSIC's distinct texts map to encoder inputs.

In `payload` mode the encoder reads what follows the first `?` or line feed, at most 256
bytes, so several texts can have one input. Counts them.
"""

from __future__ import annotations

import collections

import eq


def main() -> None:
    args = eq.arguments(__doc__)
    csic = eq.load_csic(args.vectors, args.record)
    by_input: dict[bytes, list[int]] = collections.defaultdict(list)
    for i, data in enumerate(csic.inputs):
        by_input[data].append(i)
    sizes = collections.Counter(len(texts) for texts in by_input.values())
    print(f"texts {len(csic.rows)}, distinct inputs {len(by_input)}")
    print(f"texts per input (texts, inputs): {sorted(sizes.items())}")
    two = [texts for texts in by_input.values() if len(texts) == 2]
    methods = collections.Counter(tuple(sorted(csic.rows[i]["text"].split(" ")[0] for i in texts)) for texts in two)
    print(f"inputs with two texts, by the methods of the two: {dict(methods)}")
    print(f"of them with both texts on one path: {sum(csic.group[t[0]] == csic.group[t[1]] for t in two)}")
    print(f"inputs that carry both labels: {sum(1 for texts in by_input.values() if len({bool(csic.bad[i]) for i in texts}) == 2)}")
    empty = [i for i in range(len(csic.rows)) if csic.empty[i]]
    print(
        f"texts with an empty input: {len(empty)} (bad {int(csic.bad[empty].sum())}, good {int((~csic.bad[empty]).sum())}) on "
        f"{len(set(csic.group[empty]))} paths; held-out among them {int((csic.split[empty] == 'held-out').sum())}"
    )
    for name in ["training", "validation", "held-out"]:
        mask = csic.split == name
        print(f"all texts, {name}: {int(mask.sum())}, good {int((~csic.bad[mask]).sum())}, bad {int(csic.bad[mask].sum())}")
    for flag in ["truncated", "text_longer_than_input", "empty"]:
        print(f"flag {flag}: {sum(row[flag] for row in csic.rows)} of all texts; {sum(csic.rows[i][flag] for i in csic.keep)} of the analysis set")


if __name__ == "__main__":
    main()
