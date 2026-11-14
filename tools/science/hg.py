"""Library of the hierarchical encoder gate (task 2.16): bytes, then patches, then one
vector per log line.

Development only. The `l2d` tool is Rust and never runs this code. It reuses the data
loading, the split and the scoring of the science gate (`l2dsci.py`).

The layout, as two fixed-shape graphs with plain arithmetic between them:

1. byte graph: the byte table's rows of a line as one row of an image
   `[1, 64, rows, 1024]`; causal convolutions (a kernel `[1, 9]` whose taps to the right
   of the centre are zero) give byte features `h` `[1, 96, rows, 1024]` and next-byte
   logits `y` `[1, 256, rows, 1024]` (as relu(logit + 12), see `LOGIT_FLOOR`). `y[t]`
   predicts byte `t + 1`; its entropy is what the entropy patcher reads.
2. between the graphs: the patcher cuts the line into at most 128 patches; the byte
   features of a patch are averaged (`pool`).
3. patch graph: `[1, 96, rows, 128]` patch features to `[1, 128, rows, 1]`, L2-normalised
   outside the graph.

Leakage rules: the byte model and every patch model see the training part of the fitting
side only (no held-out endpoint, no validation path); the label is read only by the
variant that says so, and by the scorers of `triage`, which hold fitting-side labels only.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import l2dsci

L_MAX = 1024          # bytes of a line the byte graph reads
P_MAX = 128           # patches of a line the patch graph reads
TABLE_WIDTH = 64      # values per byte in the byte table
FEATURES = 96         # channels of the byte features
OUTPUT_DIM = 128      # vector dimension
STRIDE = 8            # bytes per patch of the fixed-stride patcher: the budget
MAX_PATCH = 32        # no patch is longer (BLT's `max_patch_length`)
LM_LAYERS = 4
LM_KERNEL = 9         # causal: taps t-4 .. t
FEATURE_LAYER = 2     # the byte features are the output of this layer of the byte model
# The graph gives the next-byte logits as relu(logit + LOGIT_FLOOR). The byte model is
# trained so that its logits are close to log-probabilities (`NORMALISER_WEIGHT`), so the
# output lies in about [0, 12] and its 8-bit step is about 0.05 nats; a byte the model
# gives a probability below exp(-12) counts as exp(-12). Without this the min/max range
# of the raw logits was about 160 wide and the 8-bit step 0.63 nats (first run of the
# spike): too coarse for an entropy.
LOGIT_FLOOR = 12.0
NORMALISER_WEIGHT = 0.1
PATCH_WIDTH = 192
ROWS_BYTE = 16        # rows per image of the exported byte graph
ROWS_PATCH = 64       # rows per image of the exported patch graph
DELIMITERS = b" \t\n/?&=:;,|\"'()[]{}<>"

ROOT = os.environ.get("L2D_HG_DIR", r"D:\BuildArtifacts\log2decision\task-2.16")
CSIC = os.environ.get("L2D_CSIC_CSV", r"D:\Source\log2decision\data\csic-2010\csic_database.csv")
AIT_ZIP = os.environ.get("L2D_AIT_ZIP", r"D:\Source\log2decision\data\ait-lds-v2\russellmitchell.zip")
PROBES = os.environ.get("L2D_PROBES", r"D:\BuildArtifacts\log2decision\task-2.15\vectors\probes.jsonl")
AIT_ENTRIES = [
    "gather/mail/logs/horde/horde-access.log",
    "gather/mail/logs/audit/audit.log",
    "gather/inet-dns/logs/dnsmasq.log",
    "gather/cloud_share/logs/auth.log",
    "gather/inet-firewall/logs/suricata/fast.log",
    "gather/vpn/logs/openvpn.log",
    "gather/mail/logs/mail.log",
    "gather/monitoring/logs/logstash/intranet-server/2022-01-21-system.syslog.log",
]
AIT_LINES_PER_ENTRY = 2000


def path(*parts: str) -> str:
    full = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

class Lines:
    """Lines as bytes, cut at `L_MAX`, in one padded array."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        raw = [t.encode("utf-8") for t in texts]
        self.full_bytes = np.array([len(r) for r in raw], dtype=np.int64)
        self.lens = np.minimum(self.full_bytes, L_MAX)
        self.ids = np.zeros((len(raw), L_MAX), dtype=np.uint8)
        for row, r in enumerate(raw):
            n = self.lens[row]
            self.ids[row, :n] = np.frombuffer(r[:n], dtype=np.uint8)
        self.offsets = np.concatenate([[0], np.cumsum(self.lens)])

    def __len__(self) -> int:
        return len(self.texts)

    def raw(self, row: int) -> np.ndarray:
        return self.ids[row, :self.lens[row]]


class Split:
    def __init__(self):
        data = l2dsci.read_csic(CSIC)
        groups = np.array(data.groups)
        held_groups = l2dsci.held_out_groups(data, "split17", 0)
        val_groups = l2dsci.validation_groups(data, held_groups, 0)
        self.data = data
        self.groups = groups
        self.held_groups, self.val_groups = held_groups, val_groups
        self.held = np.isin(groups, sorted(held_groups))
        self.val = np.isin(groups, sorted(val_groups))
        self.train = ~self.held & ~self.val
        self.fit = ~self.held
        self.lines = Lines(data.texts)

    def record(self) -> dict:
        mixed = set(self.data.mixed_groups)
        return {
            "data": {"path": CSIC, "sha256": self.data.sha256, "records": self.data.records,
                     "distinct_texts": len(self.data.texts)},
            "split": {"name": "split17", "fold": 0,
                      "held_out_mixed_paths": sorted(self.held_groups & mixed),
                      "held_out_paths": len(self.held_groups), "held_out_logs": int(self.held.sum()),
                      "fitting_logs": int(self.fit.sum())},
            "validation_mixed_paths": sorted(self.val_groups & mixed),
            "validation_logs": int(self.val.sum()), "training_logs": int(self.train.sum()),
        }


def read_probes() -> list[dict]:
    with open(PROBES, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def ait_sample() -> tuple[list[str], list[str]]:
    """(lines, source entry per line): the first `AIT_LINES_PER_ENTRY` lines of a few
    `*.log` entries of the archive, read from the zip without unpacking it; kept as a file
    under the task directory."""
    cache = path("ait", "sample.jsonl")
    if not os.path.exists(cache):
        import zipfile

        with zipfile.ZipFile(AIT_ZIP) as archive, open(cache, "w", encoding="utf-8") as out:
            for entry in AIT_ENTRIES:
                with archive.open(entry) as handle:
                    taken = 0
                    for raw in handle:
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        if not line:
                            continue
                        out.write(json.dumps({"entry": entry, "text": line}, ensure_ascii=False) + "\n")
                        taken += 1
                        if taken == AIT_LINES_PER_ENTRY:
                            break
    with open(cache, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return [r["text"] for r in rows], [r["entry"] for r in rows]


# --------------------------------------------------------------------------------------
# The byte model: a causal convolutional next-byte model
# --------------------------------------------------------------------------------------

def build_byte_model():
    import torch
    from torch import nn

    class ByteLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Embedding(256, TABLE_WIDTH, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv2d(TABLE_WIDTH if i == 0 else FEATURES, FEATURES, (1, LM_KERNEL),
                          padding=(0, LM_KERNEL // 2)) for i in range(LM_LAYERS)])
            self.out = nn.Conv2d(FEATURES, 256, 1)
            mask = torch.zeros(1, 1, 1, LM_KERNEL)
            mask[..., :LM_KERNEL // 2 + 1] = 1.0          # taps t-4 .. t; none to the right
            self.register_buffer("mask", mask)

        def body(self, x):
            """x `[n, 64, rows, L]` -> (byte features, logits)."""
            feature = None
            for i, conv in enumerate(self.convs):
                x = torch.relu(torch.nn.functional.conv2d(
                    x, conv.weight * self.mask, conv.bias, padding=(0, LM_KERNEL // 2)))
                if i + 1 == FEATURE_LAYER:
                    feature = x
            return feature, self.out(x)

        def forward_ids(self, ids):
            """ids `[n, L]` int64 -> (features `[n, 96, L]`, raw logits `[n, 256, L]`)."""
            x = self.table(ids).transpose(1, 2).unsqueeze(2)
            feature, logits = self.body(x)
            return feature.squeeze(2), logits.squeeze(2)

        def outputs_ids(self, ids):
            """What the exported graph gives: (features, relu(logits + LOGIT_FLOOR))."""
            feature, logits = self.forward_ids(ids)
            return feature, torch.relu(logits + LOGIT_FLOOR)

    class ByteGraph(nn.Module):
        """The exported graph: plain convolutions with the masked weights written in."""

        def __init__(self, lm: ByteLM, with_logits: bool):
            super().__init__()
            self.with_logits = with_logits
            layers = LM_LAYERS if with_logits else FEATURE_LAYER
            self.convs = nn.ModuleList()
            for conv in list(lm.convs)[:layers]:
                plain = nn.Conv2d(conv.in_channels, conv.out_channels, (1, LM_KERNEL),
                                  padding=(0, LM_KERNEL // 2))
                plain.weight.data = (conv.weight * lm.mask).detach().clone()
                plain.bias.data = conv.bias.detach().clone()
                self.convs.append(plain)
            self.out = nn.Conv2d(FEATURES, 256, 1)
            self.out.weight.data = lm.out.weight.detach().clone()
            self.out.bias.data = lm.out.bias.detach().clone() + LOGIT_FLOOR

        def forward(self, x):
            feature = None
            for i, conv in enumerate(self.convs):
                x = torch.relu(conv(x))
                if i + 1 == FEATURE_LAYER:
                    feature = x
            if self.with_logits:
                return feature, torch.relu(self.out(x))
            return feature

    return ByteLM, ByteGraph


def length_batches(lens: np.ndarray, rows: np.ndarray, batch: int, rng: np.random.Generator):
    """Batches of rows with similar lengths, in random order."""
    order = rng.permutation(rows)
    batches = []
    for start in range(0, len(order), batch * 50):
        chunk = order[start:start + batch * 50]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        batches += [chunk[i:i + batch] for i in range(0, len(chunk), batch)]
    return [batches[i] for i in rng.permutation(len(batches))]


def lm_loss(model, lines: Lines, rows: np.ndarray):
    """(sum of the cross-entropy in nats, sum of the squared log-normaliser, positions) of
    predicting byte t+1 from the bytes up to t; after the last byte the model predicts
    byte 0 (the padding, as an end mark)."""
    import torch

    width = int(lines.lens[rows].max()) + 1
    width = min(width, L_MAX)
    ids = torch.from_numpy(lines.ids[rows, :width].astype(np.int64))
    _, logits = model.forward_ids(ids)
    target = torch.zeros_like(ids)
    target[:, :-1] = ids[:, 1:]
    lens = torch.from_numpy(lines.lens[rows])
    valid = torch.arange(width)[None, :] < lens[:, None]
    loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    normaliser = torch.logsumexp(logits, dim=1) ** 2
    return (loss * valid).sum(), (normaliser * valid).sum(), int(valid.sum())


def train_byte_model(split: Split, *, seed: int, epochs: int, batch: int, lr: float,
                     weight_decay: float, threads: int, log=print):
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    ByteLM, _ = build_byte_model()
    model = ByteLM()
    lines = split.lines
    train_rows = np.flatnonzero(split.train & (lines.lens > 0))
    val_rows = np.flatnonzero(split.val)
    held_rows = np.flatnonzero(split.held)
    rng = np.random.default_rng(seed)
    steps = epochs * ((len(train_rows) + batch - 1) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    def bits(rows):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for part in np.array_split(rows[np.argsort(lines.lens[rows])], max(1, len(rows) // 256)):
                s, _, n = lm_loss(model, lines, part)
                total += float(s)
                count += n
        return total / count / np.log(2)

    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for rows in length_batches(lines.lens, train_rows, batch, rng):
            s, z, n = lm_loss(model, lines, rows)
            # The second term keeps the logits near log-probabilities (self-normalisation).
            loss = (s + NORMALISER_WEIGHT * z) / n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(s.detach())
            count += n
        row = {"epoch": epoch, "train_bits_per_byte": total / count / np.log(2),
               "validation_bits_per_byte": bits(val_rows),
               "held_out_bits_per_byte": bits(held_rows),
               "seconds": time.perf_counter() - started}
        history.append(row)
        log(f"    byte model epoch {epoch}: bits per byte training {row['train_bits_per_byte']:.3f}, "
            f"validation {row['validation_bits_per_byte']:.3f}, held-out endpoints "
            f"{row['held_out_bits_per_byte']:.3f}; {row['seconds']:.0f} s")
    model.eval()
    return model, history


def entropy_nats(logits: np.ndarray) -> np.ndarray:
    """`[n, 256]` logits -> `[n]` entropies of the softmax, natural log (as BLT's `entropy`),
    in float64."""
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    total = p.sum(axis=1)
    return np.log(total) - (p * z).sum(axis=1) / total


def float_byte_outputs(model, lines: Lines, log=print):
    """Byte features `[total bytes, 96]` float32 (rows of line i at offsets[i]:offsets[i+1])
    and the entropy per byte position `[total bytes]` float64, from the float model."""
    import torch

    feats = np.zeros((int(lines.offsets[-1]), FEATURES), dtype=np.float32)
    ent = np.zeros(int(lines.offsets[-1]), dtype=np.float64)
    order = np.argsort(lines.lens, kind="stable")
    with torch.no_grad():
        for part in np.array_split(order, max(1, len(order) // 128)):
            width = int(lines.lens[part].max())
            if width == 0:
                continue
            f, y = model.outputs_ids(torch.from_numpy(lines.ids[part, :width].astype(np.int64)))
            f, y = f.numpy(), y.numpy()
            for k, row in enumerate(part):
                n, at = int(lines.lens[row]), int(lines.offsets[row])
                feats[at:at + n] = f[k, :, :n].T
                ent[at:at + n] = entropy_nats(y[k, :, :n].T)
    return feats, ent


# --------------------------------------------------------------------------------------
# Patchers. Each returns the start index of every patch of a line (first is 0).
# --------------------------------------------------------------------------------------

def finish_starts(starts: np.ndarray, n: int) -> np.ndarray:
    """Splits patches longer than `MAX_PATCH` into chunks of that length, then keeps the
    first `P_MAX` starts: the bytes of all later patches fall into the last one, so no
    byte of the line is dropped."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.append(starts[1:], n)
    if (ends - starts).max() > MAX_PATCH:
        parts = [np.arange(s, e, MAX_PATCH) for s, e in zip(starts, ends)]
        starts = np.concatenate(parts)
    return starts[:P_MAX].astype(np.int64)


def starts_fixed(n: int) -> np.ndarray:
    return finish_starts(np.arange(0, n, STRIDE), n)


IS_DELIMITER = np.zeros(256, dtype=bool)
IS_DELIMITER[list(DELIMITERS)] = True


def starts_delimiter(raw: np.ndarray, min_len: int) -> np.ndarray:
    """A patch ends after a delimiter byte, once it holds at least `min_len` bytes."""
    n = len(raw)
    starts, last = [0], 0
    for t in np.flatnonzero(IS_DELIMITER[raw[:max(n - 1, 0)]]):
        if t + 1 - last >= min_len:
            starts.append(int(t) + 1)
            last = int(t) + 1
    return finish_starts(np.array(starts, dtype=np.int64), n)


def starts_entropy(ent: np.ndarray, threshold: float, rule: str) -> np.ndarray:
    """`ent[t]` is the entropy of the model's prediction of byte t+1. `jump` (BLT's
    monotonic rule): a patch starts at byte t+1 where ent[t] - ent[t-1] > threshold.
    `global`: where ent[t] > threshold."""
    n = len(ent)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if rule == "jump":
        hit = np.flatnonzero(ent[1:n - 1] - ent[:n - 2] > threshold) + 2
    else:
        hit = np.flatnonzero(ent[:n - 1] > threshold) + 1
    return finish_starts(np.concatenate([[0], hit]), n)


def all_starts(lines: Lines, patcher: str, setting, ent: np.ndarray | None = None) -> list[np.ndarray]:
    out = []
    for row in range(len(lines)):
        n = int(lines.lens[row])
        if patcher == "fixed":
            out.append(starts_fixed(n))
        elif patcher == "delimiter":
            out.append(starts_delimiter(lines.raw(row), setting))
        else:
            at = int(lines.offsets[row])
            out.append(starts_entropy(ent[at:at + n], setting, patcher.split("-")[1]))
    return out


def calibrate_threshold(lines: Lines, rows: np.ndarray, ent: np.ndarray, rule: str, target: float) -> float:
    """The threshold at which the mean patch count of `rows` equals `target` (bisection)."""
    def mean_count(threshold):
        return float(np.mean([len(starts_entropy(
            ent[int(lines.offsets[r]):int(lines.offsets[r]) + int(lines.lens[r])], threshold, rule))
            for r in rows]))
    low, high = (-10.0, 10.0) if rule == "jump" else (0.0, 6.0)
    for _ in range(40):
        mid = (low + high) / 2
        if mean_count(mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def pool(feats: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """Mean of the byte features of each patch: `[n, C]` -> `[P_MAX, C]`, zero rows after
    the last patch. float32 sums in byte order, the arithmetic the Rust side repeats."""
    out = np.zeros((P_MAX, feats.shape[1]), dtype=np.float32)
    if n == 0:
        return out
    sums = np.add.reduceat(feats[:n].astype(np.float64), starts, axis=0)
    sizes = np.diff(np.append(starts, n))[:, None]
    out[:len(starts)] = (sums / sizes).astype(np.float32)
    return out


def pooled(lines: Lines, feats: np.ndarray, starts: list[np.ndarray]) -> np.ndarray:
    """`[lines, P_MAX, C]` float32."""
    out = np.zeros((len(lines), P_MAX, feats.shape[1]), dtype=np.float32)
    for row in range(len(lines)):
        n, at = int(lines.lens[row]), int(lines.offsets[row])
        out[row] = pool(feats[at:at + n], starts[row], n)
    return out


# --------------------------------------------------------------------------------------
# The patch model, and the flat control
# --------------------------------------------------------------------------------------

def build_patch_model():
    import torch
    from torch import nn

    class LineNet(nn.Module):
        """Convolutions along the patch axis (or, for the flat control, along the byte
        axis), one average over what is left, a 1 x 1 convolution to the vector."""

        def __init__(self, strides: tuple[int, ...], positions: int):
            super().__init__()
            layers, channels = [], FEATURES
            for stride in strides:
                layers += [nn.Conv2d(channels, PATCH_WIDTH, (1, 5), stride=(1, stride), padding=(0, 2)),
                           nn.ReLU()]
                channels = PATCH_WIDTH
                positions = (positions + 4 - 5) // stride + 1
            self.f = nn.Sequential(*layers)
            self.pool = nn.AvgPool2d((1, positions), stride=(1, positions))
            self.head = nn.Conv2d(PATCH_WIDTH, OUTPUT_DIM, 1)
            self.classifier = nn.Linear(OUTPUT_DIM, 1)   # training only (label variant)

        def forward(self, x):
            """`[n, 96, rows, positions]` -> `[n, 128, rows, 1]`."""
            return self.head(self.pool(self.f(x)))

    return LineNet


HIER_STRIDES = (1, 2, 2, 1)   # 128 patches -> 32 positions -> one average
FLAT_STRIDES = (2, 2, 2, 1)   # 1024 bytes -> 128 positions -> one average


def nt_xent(a, b, temperature: float):
    """The normalised-temperature cross-entropy of SimCLR: two views of a line are the
    positives, every other view in the batch a negative."""
    import torch

    z = torch.cat([a, b])
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(len(z), dtype=torch.bool), float("-inf"))
    n = len(a)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return torch.nn.functional.cross_entropy(sim, target)


def train_line_model(make_batch, counts, labels, rows_train, rows_val, *, flat: bool, seed: int,
                     epochs: int, batch: int, lr: float, weight_decay: float, temperature: float,
                     mask_rate: float, label_weight: float, objective: str, threads: int, log=print,
                     checkpoint: str | None = None, lengths: np.ndarray | None = None,
                     boundary_drop: float = 0.0):
    """`make_batch(rows)` gives `[b, 96, 1, positions]` float32 (patch features, or byte
    features for the flat control); `counts[row]` is the number of real positions.

    `contrastive`: two views of each line, each with a random `mask_rate` of its patches
    set to zero (flat control: of its 8-byte blocks); NT-Xent on the unit vectors.
    `masked`: one view with masked patches; the line vector, through a small decoder that
    also gets the position, must pick the pooled features of each masked patch among those
    of all masked patches of the batch (InfoNCE). The decoder is training-only.
    Both add mean((length - 1)^2) on the raw vector, as the published model, so that the
    8-bit output range is not set by a few long vectors. `label_weight` > 0 adds a
    binary cross-entropy on a classifier that reads 16 x the unit vector.

    `boundary_drop` > 0 (with `lengths`, the bytes of every patch): each view also
    loses that share of its patch boundaries at random; the two patches beside a dropped
    boundary become one, with the mean of their bytes (`merge_patches`). The vector is
    then trained to stay put when the cut of a line changes a little.

    With `checkpoint`, the state after every epoch is written to that file and a run that
    finds the file goes on from it (a run that was stopped by the host), with the random
    states restored, so that the result is that of an uninterrupted run."""
    import torch
    from torch import nn

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    LineNet = build_patch_model()
    positions = L_MAX if flat else P_MAX
    model = LineNet(FLAT_STRIDES if flat else HIER_STRIDES, positions)
    unit = STRIDE if flat else 1          # positions masked together
    slots = positions // unit
    decoder = None
    params = list(model.parameters())
    if objective == "masked":
        position_table = nn.Embedding(slots, 64)
        decoder = nn.Sequential(nn.Linear(OUTPUT_DIM + 64, 256), nn.ReLU(), nn.Linear(256, FEATURES))
        params += list(position_table.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = epochs * ((len(rows_train) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    order = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    bce = nn.BCEWithLogitsLoss()

    def masks(rows):
        keep = rng.random((len(rows), slots)) >= mask_rate
        real = np.arange(slots)[None, :] < np.ceil(counts[rows] / unit)[:, None]
        return keep, real

    def expand(keep):
        m = torch.from_numpy(np.repeat(keep, unit, axis=1).astype(np.float32))
        return m[:, None, None, :]

    def step_loss(rows):
        batch_x = make_batch(rows)
        x = torch.from_numpy(batch_x)
        parts = {}
        if objective == "contrastive":
            raw = []
            for _ in range(2):
                keep, _ = masks(rows)
                view = x if boundary_drop <= 0 else torch.from_numpy(
                    merge_patches(batch_x, lengths[rows], rng, boundary_drop))
                raw.append(model(view * expand(keep)).flatten(1))
            e = torch.cat(raw)
            z = torch.nn.functional.normalize(e, dim=1)
            parts["objective"] = nt_xent(z[:len(rows)], z[len(rows):], temperature)
            y = np.concatenate([labels[rows], labels[rows]])
        else:
            keep, real = masks(rows)
            e = model(x * expand(keep)).flatten(1)
            z = torch.nn.functional.normalize(e, dim=1)
            line, slot = np.nonzero(~keep & real)
            if unit == 1:
                target = x[line, :, 0, slot]
            else:
                target = x.reshape(len(rows), FEATURES, slots, unit).mean(3)[line, :, slot]
            query = decoder(torch.cat([z[line], position_table(torch.from_numpy(slot))], dim=1))
            sim = torch.nn.functional.normalize(query, dim=1) @ torch.nn.functional.normalize(target, dim=1).T
            parts["objective"] = torch.nn.functional.cross_entropy(sim / temperature, torch.arange(len(line)))
            y = labels[rows]
        parts["length"] = ((e.norm(dim=1) - 1.0) ** 2).mean()
        loss = parts["objective"] + parts["length"]
        if label_weight > 0:
            parts["label"] = bce(model.classifier(16.0 * z).squeeze(1), torch.from_numpy(y.astype(np.float32)))
            loss = loss + label_weight * parts["label"]
        return loss, parts

    history = []
    extra = [m for m in (decoder, position_table if decoder is not None else None) if m is not None]
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(state["model"])
        for module, saved in zip(extra, state["extra"]):
            module.load_state_dict(saved)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        order.set_state(state["order"])
        rng.bit_generator.state = state["rng"]
        history = state["history"]
        log(f"    resumed after epoch {len(history)} from {checkpoint}")
    for epoch in range(len(history) + 1, epochs + 1):
        started = time.perf_counter()
        model.train()
        sums, seen = {}, 0
        for index in torch.randperm(len(rows_train), generator=order).split(batch):
            if len(index) < 8:
                continue
            rows = rows_train[index.numpy()]
            loss, parts = step_loss(rows)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for name, value in parts.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * len(rows)
            seen += len(rows)
        model.eval()
        val_sums, val_seen = {}, 0
        with torch.no_grad():
            for rows in np.array_split(rows_val, max(1, len(rows_val) // batch)):
                _, parts = step_loss(rows)
                for name, value in parts.items():
                    val_sums[name] = val_sums.get(name, 0.0) + float(value) * len(rows)
                val_seen += len(rows)
        row = {"epoch": epoch, "train": {k: v / seen for k, v in sums.items()},
               "validation": {k: v / val_seen for k, v in val_sums.items()},
               "seconds": time.perf_counter() - started}
        history.append(row)
        log(f"    epoch {epoch}: training {json.dumps({k: round(v, 4) for k, v in row['train'].items()})} "
            f"validation {json.dumps({k: round(v, 4) for k, v in row['validation'].items()})} {row['seconds']:.0f} s")
        if checkpoint:
            torch.save({"model": model.state_dict(), "extra": [m.state_dict() for m in extra],
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "order": order.get_state(),
                        "rng": rng.bit_generator.state, "history": history}, checkpoint + ".tmp")
            os.replace(checkpoint + ".tmp", checkpoint)
    model.eval()
    return model, history


def merge_patches(x: np.ndarray, lengths: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """`x` `[b, C, 1, P]` patch features, `lengths` `[b, P]` bytes per patch (0 after the
    last). Drops each patch boundary with probability `rate`; patches that are no longer
    separated become one patch with the mean of their bytes."""
    out = np.zeros_like(x)
    for k in range(len(x)):
        n = int((lengths[k] > 0).sum())
        if n < 2:
            out[k] = x[k]
            continue
        keep = np.concatenate([[True], rng.random(n - 1) >= rate])
        starts = np.flatnonzero(keep)
        weight = lengths[k, :n].astype(np.float32)
        sums = np.add.reduceat(x[k, :, 0, :n] * weight, starts, axis=1)
        out[k, :, 0, :len(starts)] = sums / np.add.reduceat(weight, starts)
    return out


def line_vectors(model, make_batch, count: int) -> np.ndarray:
    """Raw (not normalised) vectors of rows 0..count from the float model."""
    import torch

    out = np.zeros((count, OUTPUT_DIM), dtype=np.float32)
    with torch.no_grad():
        for rows in np.array_split(np.arange(count), max(1, count // 256)):
            out[rows] = model(torch.from_numpy(make_batch(rows))).flatten(1).numpy()
    return out


# --------------------------------------------------------------------------------------
# ONNX: export in the image layout, 8-bit QDQ, runs through ONNX Runtime on the CPU
# (the Python 8-bit reference the NPU vectors are compared with)
# --------------------------------------------------------------------------------------

def export_onnx(module, shape: tuple[int, ...], outputs: list[str], onnx_path: str) -> None:
    import torch

    torch.onnx.export(module, torch.zeros(*shape), onnx_path, input_names=["x"],
                      output_names=outputs, opset_version=17, dynamo=False, do_constant_folding=True)


def quantize(float_path: str, int8_path: str, images: list[np.ndarray]) -> None:
    """As `l2dsci.quantize_qdq`: static QDQ, uint8 activations, int8 symmetric weights,
    per tensor, min/max over the calibration images."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.images = iter(images)

        def get_next(self):
            image = next(self.images, None)
            return None if image is None else {"x": image}

    quantize_static(float_path, int8_path, Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=False, calibrate_method=CalibrationMethod.MinMax)


def session(onnx_path: str, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])


def byte_image(table: np.ndarray, ids: np.ndarray, rows: int) -> np.ndarray:
    """`[1, 64, rows, 1024]`: line n of `ids` is row n; rows after the last line are zero."""
    image = np.zeros((1, TABLE_WIDTH, rows, L_MAX), dtype=np.float32)
    image[0, :, :len(ids), :] = table[ids].transpose(2, 0, 1)
    return image


def onnx_byte_outputs(sess, table: np.ndarray, lines: Lines, with_logits: bool, log=print):
    """As `float_byte_outputs`, through an exported byte graph."""
    feats = np.zeros((int(lines.offsets[-1]), FEATURES), dtype=np.float32)
    ent = np.zeros(int(lines.offsets[-1]), dtype=np.float64) if with_logits else None
    rows = sess.get_inputs()[0].shape[2]
    started = time.perf_counter()
    for start in range(0, len(lines), rows):
        part = np.arange(start, min(start + rows, len(lines)))
        outputs = sess.run(None, {"x": byte_image(table, lines.ids[part], rows)})
        for k, row in enumerate(part):
            n, at = int(lines.lens[row]), int(lines.offsets[row])
            feats[at:at + n] = outputs[0][0, :, k, :n].T
            if with_logits:
                ent[at:at + n] = entropy_nats(outputs[1][0, :, k, :n].T)
    log(f"    byte graph over {len(lines)} lines: {time.perf_counter() - started:.0f} s")
    return feats, ent


def onnx_line_vectors(sess, make_batch, count: int) -> np.ndarray:
    """Raw vectors through an exported line graph `[1, 96, rows, positions]` -> `[1, 128, rows, 1]`."""
    rows = sess.get_inputs()[0].shape[2]
    out = np.zeros((count, OUTPUT_DIM), dtype=np.float32)
    for start in range(0, count, rows):
        part = np.arange(start, min(start + rows, count))
        x = make_batch(part)                               # [b, 96, 1, positions]
        image = np.zeros((1, FEATURES, rows, x.shape[3]), dtype=np.float32)
        image[0, :, :len(part), :] = x[:, :, 0, :].transpose(1, 0, 2)
        out[part] = sess.run(None, {"x": image})[0][0, :, :len(part), 0].T
    return out


# --------------------------------------------------------------------------------------
# Triage quality, with the scorers of the science gate
# --------------------------------------------------------------------------------------

def triage(vectors: np.ndarray, split: Split, cv5: bool) -> dict:
    """Linear head and 11-neighbour ratio of `l2dsci`, fitted on the fitting side and
    scored on the three held-out endpoints (`split17`); with `cv5` also the five folds
    grouped by URL path of the science gate, all held-out predictions pooled."""
    from sklearn.metrics import roc_auc_score

    data = split.data
    unit = l2dsci.l2n(vectors)
    mixed = set(data.mixed_groups)

    def score(fit, held):
        store_groups = [g for g, f in zip(data.groups, fit) if f]
        groups = [g for g, h in zip(data.groups, held) if h]
        ratio, _ = l2dsci.knn_ratio(unit[fit], data.bad[fit], unit[held], 11)
        probability, head = l2dsci.linear_head(unit[fit], data.bad[fit], store_groups, unit[held])
        return ratio, probability, groups, head

    def short(block):
        return {"roc_auc": block["cross_endpoint"].get("roc_auc"),
                "average_precision": block["cross_endpoint"].get("average_precision"),
                "per_group": {g: b.get("roc_auc") for g, b in block["per_group"].items()}}

    ratio, probability, groups, head = score(split.fit, split.held)
    out = {"split17": {
        "linear_head": short(l2dsci.metrics(probability, data.bad[split.held], groups, mixed)),
        "knn_k11": short(l2dsci.metrics(ratio, data.bad[split.held], groups, mixed)),
        "head_penalty": head["penalty"]}}
    if cv5:
        fold_of = l2dsci.cv5_fold_of_group(data)
        fold = np.array([fold_of[g] for g in data.groups])
        in_mixed = np.isin(split.groups, sorted(mixed))
        pooled_head = np.zeros(len(unit))
        pooled_knn = np.zeros(len(unit))
        per_fold = []
        for f in range(5):
            held = fold == f
            ratio, probability, groups, _ = score(~held, held)
            pooled_head[held], pooled_knn[held] = probability, ratio
            use = held & in_mixed
            per_fold.append({"fold": f,
                             "linear_head": float(roc_auc_score(data.bad[use], pooled_head[use])),
                             "knn_k11": float(roc_auc_score(data.bad[use], pooled_knn[use]))})
        out["cv5"] = {
            "linear_head_pooled": float(roc_auc_score(data.bad[in_mixed], pooled_head[in_mixed])),
            "knn_k11_pooled": float(roc_auc_score(data.bad[in_mixed], pooled_knn[in_mixed])),
            "linear_head_mean_of_folds": float(np.mean([p["linear_head"] for p in per_fold])),
            "knn_k11_mean_of_folds": float(np.mean([p["knn_k11"] for p in per_fold])),
            "per_fold": per_fold}
    return out
