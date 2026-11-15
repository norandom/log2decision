"""Library of the stability gate of the entropy-patch encoder (task 2.17).

Development only; the `l2d` tool is Rust and never runs this code. It keeps the layout of
task 2.16 (`hg.py`: a causal convolutional byte model whose next-byte entropy cuts a line
into patches; the byte features of a patch are averaged; a convolutional line model
turns the patches into one vector per line, the whole line read) and changes what 2.16
measured as unstable or flawed:

- The cut of a line is the 8-bit byte graph's (`int8_entropies`, the arithmetic of the
  Rust side, bit for bit): the float path pools its float byte features at the same
  cuts, so float, 8 bits and the NPU cut a line at the same bytes.
- The line graph has convolutions of stride 1 only and gives one output per patch
  position; the mean over the line's real patches is plain arithmetic after the graph
  (`masked_mean`), as the pooling of bytes into patches is before it. 2.16's line graph
  (strided convolutions and an average over all 128 positions, padding included) was
  not the same arithmetic on the NPU as in ONNX Runtime on the CPU.
- The objective (`masked`) is patch-level byte prediction: 30 % of a line's patches
  are blanked; from the unit line vector, an embedding of the patch position and one
  of the byte position inside the patch, a small training-only decoder predicts every
  byte (at most 32) of each blanked patch and of as many visible ones, cross-entropy
  over 256 byte values, no label. (A first version picked pooled patch features by
  InfoNCE among the batch's patches; it collapsed and was replaced, see
  `train_line_model`.) 2.16's contrastive loss stays as the control.
- Every dataset is streamed: the texts are memory-mapped bytes; byte features are never
  stored, only the pooled patch features, appended to a file as they are produced and
  read back memory-mapped by the batch maker. No checkpoint and no resume. Held whole,
  as stated and bounded needs: the vectors of a corpus (`float_vectors`,
  `onnx_vectors`, the evaluation; 25,608 x 128 float32, about 13 MB), the patch starts
  (`PatchBytes`, about 3 MB), and the baseline's sparse feature rows in `refs`.
- Seeds: numpy and torch are seeded per run, deterministic torch algorithms are asked
  for, no DataLoader workers exist, the thread count is fixed and recorded.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import hg
import l2dsci

L_MAX, P_MAX, FEATURES, OUTPUT_DIM = hg.L_MAX, hg.P_MAX, hg.FEATURES, hg.OUTPUT_DIM
TABLE_WIDTH, MAX_PATCH = hg.TABLE_WIDTH, hg.MAX_PATCH
ROWS_BYTE, ROWS_PATCH = hg.ROWS_BYTE, hg.ROWS_PATCH
PATCH_WIDTH = hg.PATCH_WIDTH
LINE_LAYERS = 4
LINE_KERNEL = 5

ROOT = os.environ.get("L2D_HG17_DIR", r"D:\BuildArtifacts\log2decision\task-2.17")
MODELS = os.environ.get("L2D_HG17_MODELS", r"D:\Source\log2decision\models\task-2.17")
CSIC = hg.CSIC
PROBES = hg.PROBES


def path(*parts: str) -> str:
    full = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full


def write_json(file_path: str, value) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False)
    os.replace(file_path + ".tmp", file_path)


def read_json(file_path: str):
    with open(file_path, encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed: int, threads: int) -> dict:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn") and hasattr(torch.backends.mkldnn, "deterministic"):
        torch.backends.mkldnn.deterministic = True
    return {"seed": seed, "threads": threads, "torch_deterministic_algorithms": True,
            "mkldnn_deterministic": bool(getattr(getattr(torch.backends, "mkldnn", None), "deterministic", False)),
            "dataloader_workers": 0, "python_random": seed, "numpy_global": seed, "torch_manual_seed": seed}


# --------------------------------------------------------------------------------------
# The corpus: memory-mapped bytes
# --------------------------------------------------------------------------------------

class Corpus:
    """Texts as one memory-mapped byte file with offsets (each cut at `L_MAX` bytes;
    no CSIC text is longer). `prepare` writes it once."""

    def __init__(self, name: str):
        base = path("data", name)
        self.name = name
        self.offsets = np.fromfile(base + ".offsets.i64", dtype="<i8")
        self.lens = np.diff(self.offsets)
        self.bytes = np.memmap(base + ".bytes.u8", dtype=np.uint8, mode="r") if self.offsets[-1] > 0 else np.zeros(0, np.uint8)
        meta = read_json(base + ".meta.json")
        self.full_bytes = np.array(meta["full_bytes"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.lens)

    def raw(self, row: int) -> np.ndarray:
        return np.asarray(self.bytes[self.offsets[row]:self.offsets[row + 1]])

    def ids(self, rows: np.ndarray, width: int) -> np.ndarray:
        out = np.zeros((len(rows), width), dtype=np.uint8)
        for k, row in enumerate(rows):
            n = min(int(self.lens[row]), width)
            out[k, :n] = self.bytes[self.offsets[row]:self.offsets[row] + n]
        return out


def write_corpus(name: str, texts: list[str]) -> None:
    base = path("data", name)
    lens, full = [], []
    with open(base + ".bytes.u8", "wb") as handle:
        for text in texts:
            raw = text.encode("utf-8")
            full.append(len(raw))
            handle.write(raw[:L_MAX])
            lens.append(min(len(raw), L_MAX))
    np.concatenate([[0], np.cumsum(lens)]).astype("<i8").tofile(base + ".offsets.i64")
    write_json(base + ".meta.json", {"texts": len(texts), "full_bytes": full})


class Labels:
    """Labels, endpoints and folds of the CSIC corpus (small arrays)."""

    def __init__(self):
        meta = read_json(path("data", "csic.labels.json"))
        self.bad = np.array(meta["bad"], dtype=bool)
        self.groups = meta["groups"]
        self.group_array = np.array(self.groups)
        self.mixed_groups = meta["mixed_groups"]
        self.mixed = np.isin(self.group_array, self.mixed_groups)
        self.fold = np.array(meta["fold"], dtype=np.int64)
        self.sha256 = meta["sha256"]
        self.records = meta["records"]
        self.split17_held = np.isin(self.group_array, meta["split17_held_groups"])

    def fit(self, fold: int) -> np.ndarray:
        return self.fold != fold

    def held(self, fold: int) -> np.ndarray:
        return self.fold == fold


# --------------------------------------------------------------------------------------
# The byte model (hg.py's), trained on a corpus
# --------------------------------------------------------------------------------------

def lm_loss(model, corpus: Corpus, rows: np.ndarray):
    import torch

    width = min(int(corpus.lens[rows].max()) + 1, L_MAX)
    ids = torch.from_numpy(corpus.ids(rows, width).astype(np.int64))
    _, logits = model.forward_ids(ids)
    target = torch.zeros_like(ids)
    target[:, :-1] = ids[:, 1:]
    lens = torch.from_numpy(corpus.lens[rows])
    valid = torch.arange(width)[None, :] < lens[:, None]
    loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    normaliser = torch.logsumexp(logits, dim=1) ** 2
    return (loss * valid).sum(), (normaliser * valid).sum(), int(valid.sum())


def train_byte_model(corpus: Corpus, train_rows: np.ndarray, held_rows: np.ndarray, *, seed: int,
                     epochs: int, batch: int, lr: float, weight_decay: float, log=print):
    import torch

    ByteLM, _ = hg.build_byte_model()
    torch.manual_seed(seed)
    model = ByteLM()
    rng = np.random.default_rng(seed)
    steps = epochs * sum(1 for _ in hg.length_batches(corpus.lens, train_rows, batch, np.random.default_rng(0)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    def bits(rows):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            order = rows[np.argsort(corpus.lens[rows], kind="stable")]
            for part in np.array_split(order, max(1, len(order) // 256)):
                s, _, n = lm_loss(model, corpus, part)
                total += float(s)
                count += n
        return total / count / np.log(2)

    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for rows in hg.length_batches(corpus.lens, train_rows, batch, rng):
            s, z, n = lm_loss(model, corpus, rows)
            loss = (s + hg.NORMALISER_WEIGHT * z) / n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(s.detach())
            count += n
        row = {"epoch": epoch, "train_bits_per_byte": total / count / np.log(2), "seconds": time.perf_counter() - started}
        if epoch == epochs:
            row["held_out_bits_per_byte"] = bits(held_rows)
        history.append(row)
        log(f"    byte model epoch {epoch}: bits per byte training {row['train_bits_per_byte']:.3f}"
            + (f", held-out endpoints {row['held_out_bits_per_byte']:.3f}" if "held_out_bits_per_byte" in row else "")
            + f"; {row['seconds']:.0f} s")
    model.eval()
    return model, history


# --------------------------------------------------------------------------------------
# ONNX Runtime on the CPU (the Python 8-bit reference)
# --------------------------------------------------------------------------------------

def session(onnx_path: str, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.use_deterministic_compute = True
    return ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])


def dequantize_scale(onnx_path: str, tensor: str) -> tuple[float, int]:
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(onnx_path).graph
    constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    (node,) = [n for n in graph.node if n.op_type == "DequantizeLinear" and n.output[0] == tensor]
    return float(constants[node.input[1]]), int(constants[node.input[2]])


def exp_table(scale: float) -> np.ndarray:
    """`[256, 2]` float64: exp(q s) and q exp(q s) for the 256 steps q of an 8-bit logit.
    Written into the NPU fixture, so that the Rust side uses the same numbers."""
    q = np.arange(256, dtype=np.float64)
    e = np.exp(q * np.float64(np.float32(scale)))
    return np.stack([e, q * e], axis=1)


def int8_entropies(y: np.ndarray, scale: float, table: np.ndarray) -> np.ndarray:
    """The entropy (nats) at every position of the 8-bit byte graph's logit output `y`
    `[256, ...]` (relu(logit + 12) in steps of `scale`, zero point 0): the arithmetic of
    `Between::entropies` on the Rust side, per position in the same order (classes 0 to
    255, float64). Returns `y.shape[1:]`."""
    inverse = np.float32(np.float32(1.0) / np.float32(scale))
    total = np.zeros(y.shape[1:], dtype=np.float64)
    weighted = np.zeros(y.shape[1:], dtype=np.float64)
    for c in range(256):
        steps = (y[c] * inverse + np.float32(0.5)).astype(np.int64)
        np.minimum(steps, 255, out=steps)
        total += table[steps, 0]
        weighted += table[steps, 1]
    return np.log(total) - np.float64(np.float32(scale)) * weighted / total


def fake_quant_entropies(raw_logits: np.ndarray, scale: float, table: np.ndarray) -> np.ndarray:
    """The float model's logits `[256, ...]` put on the 8-bit graph's output grid
    (relu(logit + 12), round to the step, at most 255 steps), then `int8_entropies`:
    what a quantization-aware cut would read if only the output were quantized."""
    y = np.maximum(raw_logits + np.float32(hg.LOGIT_FLOOR), 0).astype(np.float32)
    q = np.minimum(np.rint(y / np.float32(scale)), 255).astype(np.float32) * np.float32(scale)
    return int8_entropies(q, scale, table)


def float_entropies(logits: np.ndarray) -> np.ndarray:
    """`[256, n]` raw float logits -> entropy per position (hg.entropy_nats)."""
    return hg.entropy_nats(logits.T)


def starts_jump(ent: np.ndarray, threshold: float) -> np.ndarray:
    return hg.starts_entropy(ent, threshold, "jump")


def pool(feats: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """`feats` `[C, >= n]` byte features of one line -> `[patches, C]` float32: the mean of
    each patch, float64 sums in byte order, as `Between::pool` on the Rust side."""
    if n == 0:
        return np.zeros((0, feats.shape[0]), dtype=np.float32)
    ends = np.append(starts[1:], n)
    sizes = ends - starts
    acc = np.zeros((len(starts), feats.shape[0]), dtype=np.float64)
    f = feats[:, :n].T.astype(np.float64)
    for k in range(int(sizes.max())):
        use = sizes > k
        acc[use] += f[starts[use] + k]
    return (acc / sizes[:, None]).astype(np.float32)


def masked_mean(e: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """`e` `[rows, D, P]` per-position outputs -> `[rows, D]` float32: the mean over the
    first `counts[row]` positions, float64 sums in position order (the Rust side's)."""
    acc = np.zeros(e.shape[:2], dtype=np.float64)
    for p in range(int(counts.max()) if len(counts) else 0):
        use = counts > p
        acc[use] += e[use, :, p].astype(np.float64)
    out = acc / np.maximum(counts, 1)[:, None]
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Patch features on disk: ragged, appended, read back memory-mapped
# --------------------------------------------------------------------------------------

class PatchWriter:
    def __init__(self, base: str):
        self.base = base
        self.handle = open(base + ".f32", "wb")
        self.counts = []

    def add(self, pooled: np.ndarray) -> None:
        self.handle.write(np.ascontiguousarray(pooled, dtype="<f4").tobytes())
        self.counts.append(len(pooled))

    def close(self) -> None:
        self.handle.close()
        np.array(self.counts, dtype="<i4").tofile(self.base + ".counts.i32")


class PatchBytes:
    """The bytes of each patch of a corpus, from the memory-mapped text bytes and the
    patch starts (`<name>-starts.i32`, ragged, with the patch counts)."""

    def __init__(self, corpus: Corpus, starts_path: str, counts: np.ndarray):
        self.corpus = corpus
        self.starts = np.fromfile(starts_path, dtype="<i4").astype(np.int64)
        self.counts = counts
        self.offsets = np.concatenate([[0], np.cumsum(counts)])

    def get(self, lines: np.ndarray, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`[q, MAX_PATCH]` uint8 bytes (zero after the patch) and `[q]` sizes (at most
        MAX_PATCH; the last patch of a line at the patch limit can hold more)."""
        content = np.zeros((len(lines), MAX_PATCH), dtype=np.uint8)
        sizes = np.zeros(len(lines), dtype=np.int64)
        for q, (line, slot) in enumerate(zip(lines, slots)):
            at = self.offsets[line] + slot
            start = self.starts[at]
            end = self.starts[at + 1] if slot + 1 < self.counts[line] else self.corpus.lens[line]
            n = int(min(end - start, MAX_PATCH))
            base = int(self.corpus.offsets[line] + start)
            content[q, :n] = self.corpus.bytes[base:base + n]
            sizes[q] = n
        return content, sizes


class Patches:
    """Pooled patch features of a corpus, `[total patches, 96]` memory-mapped."""

    def __init__(self, base: str):
        self.counts = np.fromfile(base + ".counts.i32", dtype="<i4").astype(np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self.values = np.memmap(base + ".f32", dtype="<f4", mode="r", shape=(int(self.offsets[-1]), FEATURES))

    def batch(self, rows: np.ndarray, width: int | None = None) -> np.ndarray:
        """`[b, 96, 1, width]` float32, zero after a line's last patch."""
        width = width or max(1, int(self.counts[rows].max()))
        x = np.zeros((len(rows), FEATURES, 1, width), dtype=np.float32)
        for k, row in enumerate(rows):
            a, b = int(self.offsets[row]), int(self.offsets[row + 1])
            x[k, :, 0, :b - a] = self.values[a:b].T
        return x


# --------------------------------------------------------------------------------------
# The line model: stride 1, one output per patch position, the mean after the graph
# --------------------------------------------------------------------------------------

def build_line_model():
    import torch
    from torch import nn

    class LineNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers, channels = [], FEATURES
            for _ in range(LINE_LAYERS):
                layers += [nn.Conv2d(channels, PATCH_WIDTH, (1, LINE_KERNEL), padding=(0, LINE_KERNEL // 2)), nn.ReLU()]
                channels = PATCH_WIDTH
            self.f = nn.Sequential(*layers)
            self.head = nn.Conv2d(PATCH_WIDTH, OUTPUT_DIM, 1)
            self.classifier = nn.Linear(OUTPUT_DIM, 1)       # training only (label term)

        def positions(self, x):
            """`[n, 96, rows, P]` -> `[n, 128, rows, P]`: what the exported graph gives."""
            return self.head(self.f(x))

        def forward(self, x, counts):
            """`x` `[b, 96, 1, P]`, `counts` `[b]` -> raw line vectors `[b, 128]`."""
            e = self.positions(x)[:, :, 0, :]
            mask = (torch.arange(e.shape[2])[None, :] < counts[:, None]).to(e.dtype)
            return (e * mask[:, None, :]).sum(2) / counts.clamp(min=1)[:, None].to(e.dtype)

    class Graph(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.f, self.head = net.f, net.head

        def forward(self, x):
            return self.head(self.f(x))

    return LineNet, Graph


def nt_xent(a, b, temperature: float):
    return hg.nt_xent(a, b, temperature)


def train_line_model(patches: Patches, labels: np.ndarray, rows_train: np.ndarray, *, seed: int, epochs: int,
                     batch: int, lr: float, weight_decay: float, temperature: float, mask_rate: float,
                     label_weight: float, objective: str, patch_bytes=None, log=print):
    """`masked`: each line is seen with `mask_rate` of its patches set to zero. From the
    unit line vector, a learned embedding of the patch position and one of the byte
    position inside the patch, a small decoder predicts every byte of a queried patch
    (cross-entropy over 256 byte values; `patch_bytes.get` gives the bytes). Queried are
    all masked patches (part `masked`: masked-patch prediction) and as many visible ones
    drawn at random (part `visible`: the vector must hold what the line says). The
    decoder is training-only. A first version predicted the pooled byte features of a
    patch by InfoNCE among the batch's patches; most patches (method, host, path) are
    the same in every line, the loss stayed near chance and the vectors collapsed to one
    direction (smoke run, `logs/smoke-line.log`), so the bytes are the target.
    `contrastive`: 2.16's NT-Xent between two masked views of a line (the control).
    Both add mean((|e| - 1)^2) on the raw vector. `label_weight` > 0 adds a binary
    cross-entropy (part `label`, reported separately)."""
    import torch
    from torch import nn

    LineNet, _ = build_line_model()
    torch.manual_seed(seed)
    model = LineNet()
    params = list(model.parameters())
    position_table = decoder = None
    if objective == "masked":
        position_table = nn.Embedding(P_MAX, 64)
        decoder = nn.ModuleDict({"patch": nn.Sequential(nn.Linear(OUTPUT_DIM + 64, 256), nn.ReLU(), nn.Linear(256, 256)),
                                 "byte_position": nn.Embedding(MAX_PATCH, 256),
                                 "out": nn.Linear(256, 256)})
        params += list(position_table.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    per_epoch = sum(1 for i in range(0, len(rows_train), batch) if len(rows_train[i:i + batch]) >= 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch, pct_start=0.1)
    order = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    bce = nn.BCEWithLogitsLoss()

    def step_loss(rows):
        counts_np = patches.counts[rows]
        x = torch.from_numpy(patches.batch(rows))
        counts = torch.from_numpy(counts_np)
        width = x.shape[3]
        real = np.arange(width)[None, :] < counts_np[:, None]
        parts = {}
        if objective == "contrastive":
            raw = []
            for _ in range(2):
                keep = rng.random((len(rows), width)) >= mask_rate
                raw.append(model(x * torch.from_numpy(keep.astype(np.float32))[:, None, None, :], counts))
            e = torch.cat(raw)
            z = torch.nn.functional.normalize(e, dim=1)
            parts["objective"] = nt_xent(z[:len(rows)], z[len(rows):], temperature)
            y = np.concatenate([labels[rows], labels[rows]])
        else:
            keep = rng.random((len(rows), width)) >= mask_rate
            masked = ~keep & real
            visible = keep & real
            e = model(x * torch.from_numpy(keep.astype(np.float32))[:, None, None, :], counts)
            z = torch.nn.functional.normalize(e, dim=1)
            m_line, m_slot = np.nonzero(masked)
            v_line, v_slot = np.nonzero(visible)
            take = rng.permutation(len(v_line))[:max(len(m_line), 1)]
            v_line, v_slot = v_line[take], v_slot[take]
            line_np = np.concatenate([m_line, v_line])
            slot_np = np.concatenate([m_slot, v_slot])
            # The bytes of each queried patch (at most MAX_PATCH), predicted from the line
            # vector and the positions alone: patch-level byte prediction.
            content, sizes = patch_bytes.get(rows[line_np], slot_np)
            q_index, k_index = np.nonzero(np.arange(MAX_PATCH)[None, :] < sizes[:, None])
            hidden = decoder["patch"](torch.cat([z[torch.from_numpy(line_np)],
                                                 position_table(torch.from_numpy(slot_np))], dim=1))
            per_byte = torch.relu(hidden[torch.from_numpy(q_index)] + decoder["byte_position"](torch.from_numpy(k_index)))
            logits = decoder["out"](per_byte)
            loss_byte = torch.nn.functional.cross_entropy(
                logits, torch.from_numpy(content[q_index, k_index].astype(np.int64)), reduction="none")
            in_masked = torch.from_numpy(q_index < len(m_line))
            parts["masked"] = loss_byte[in_masked].mean() if bool(in_masked.any()) else loss_byte.sum() * 0
            parts["visible"] = loss_byte[~in_masked].mean()
            parts["objective"] = loss_byte.mean()
            y = labels[rows]
        parts["length"] = ((e.norm(dim=1) - 1.0) ** 2).mean()
        loss = parts["objective"] + parts["length"]
        if label_weight > 0:
            parts["label"] = bce(model.classifier(16.0 * z).squeeze(1), torch.from_numpy(y.astype(np.float32)))
            loss = loss + label_weight * parts["label"]
        return loss, parts

    history = []
    for epoch in range(1, epochs + 1):
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
        row = {"epoch": epoch, "train": {k: v / seen for k, v in sums.items()}, "seconds": time.perf_counter() - started}
        history.append(row)
        log(f"    epoch {epoch}: training {json.dumps({k: round(v, 4) for k, v in row['train'].items()})} {row['seconds']:.0f} s")
    model.eval()
    extra = {"decoder": decoder, "position_table": position_table}

    def held_loss(rows_held):
        """The objective on lines the model never saw (no label read): measured once."""
        sums, seen = {}, 0
        with torch.no_grad():
            for i in range(0, len(rows_held), batch):
                rows = rows_held[i:i + batch]
                if len(rows) < 8:
                    continue
                _, parts = step_loss(rows)
                for name, value in parts.items():
                    if name != "label":
                        sums[name] = sums.get(name, 0.0) + float(value) * len(rows)
                seen += len(rows)
        return {k: v / max(seen, 1) for k, v in sums.items()}

    return model, history, extra, held_loss


def float_vectors(model, patches: Patches, rows: np.ndarray, batch: int = 256) -> np.ndarray:
    """Raw float vectors, every line at the full width of `P_MAX` positions."""
    import torch

    out = np.zeros((len(rows), OUTPUT_DIM), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            part = rows[i:i + batch]
            x = torch.from_numpy(patches.batch(part, P_MAX))
            out[i:i + len(part)] = model(x, torch.from_numpy(patches.counts[part])).numpy()
    return out


def onnx_vectors(sess, patches: Patches, rows: np.ndarray) -> np.ndarray:
    """Raw 8-bit vectors: the exported line graph on images of `ROWS_PATCH` lines (lines in
    order, rows after the last line zero), then `masked_mean`."""
    rows_per_image = sess.get_inputs()[0].shape[2]
    out = np.zeros((len(rows), OUTPUT_DIM), dtype=np.float32)
    for i in range(0, len(rows), rows_per_image):
        part = rows[i:i + rows_per_image]
        x = patches.batch(part, P_MAX)
        image = np.zeros((1, FEATURES, rows_per_image, P_MAX), dtype=np.float32)
        image[0, :, :len(part), :] = x[:, :, 0, :].transpose(1, 0, 2)
        e = sess.run(None, {"x": image})[0][0, :, :len(part), :].transpose(1, 0, 2)   # [rows, 128, P]
        out[i:i + len(part)] = masked_mean(e, patches.counts[part])
    return out
