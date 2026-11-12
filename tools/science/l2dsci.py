"""Library of the science gate (task 2.1): data, splits, baseline, encoder, export, metrics.

Development only. The `l2d` tool is Rust and never runs this code.

Leakage rules kept here (the reviewer should check these first):
- texts are deduplicated before any split;
- every split is by URL path (the group key of the Rust CSIC adapter): all texts of a path
  are on one side;
- the encoder, its early stopping, the calibration sample of the quantizer, the neighbour
  store, the linear head and its penalty search see fitting-side texts only;
- held-out labels are read in exactly one place: `metrics`.
"""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass

import numpy as np

INPUT_BYTES = 256      # bytes per log the encoder reads
TABLE_WIDTH = 64       # values per byte in the byte table
OUTPUT_DIM = 128       # vector dimension
BATCH = 32             # static batch of the ONNX graph
PAD_BYTE = 0           # padding byte id; its table row is fixed at zero


# --------------------------------------------------------------------------------------
# Data: the text layout and group key of `l2d_core::adapters::csic2010`
# --------------------------------------------------------------------------------------

def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_routing(text: str) -> tuple[str, str]:
    """(routing, payload): cut at the first `?` or line feed, which belongs to neither."""
    cut = len(text)
    for mark in ("?", "\n"):
        at = text.find(mark)
        if at != -1:
            cut = min(cut, at)
    if cut == len(text):
        return text, ""
    return text[:cut], text[cut + 1:]


def group_key(text: str) -> str:
    routing, _ = split_routing(text)
    target = routing.split(" ", 1)[1] if " " in routing else routing
    if "://" not in target:
        return target
    scheme, rest = target.split("://", 1)
    ok = (scheme[:1].isascii() and scheme[:1].isalpha()
          and all((c.isascii() and c.isalnum()) or c in "+-." for c in scheme))
    if not ok:
        return target
    at = rest.find("/")
    return rest[at:] if at != -1 else "/"


@dataclass
class Dataset:
    path: str
    sha256: str
    records: int
    texts: list[str]          # distinct, in order of first occurrence
    bad: np.ndarray           # bool per distinct text
    groups: list[str]         # URL path per distinct text
    mixed_groups: list[str]   # paths holding both labels, byte order


def read_csic(path: str) -> Dataset:
    """Reads `csic_database.csv` the way the Rust adapter does and deduplicates the texts."""
    seen: dict[str, int] = {}
    labels: list[bool] = []
    records = 0
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        at = {name: header.index(name) for name in ("Method", "URL", "content", "classification")}
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"row {records + 1}: {len(row)} fields, header has {len(header)}")
            records += 1
            method = row[at["Method"]].strip()
            url = row[at["URL"]].strip()
            head, _, version = url.rpartition(" ")
            if head and version.startswith("HTTP/"):
                url = head
            body = row[at["content"]]
            label = {"0": False, "1": True}[row[at["classification"]].strip()]
            text = f"{method} {url}" if body == "" else f"{method} {url}\n{body}"
            known = seen.get(text)
            if known is None:
                seen[text] = len(labels)
                labels.append(label)
            elif labels[known] != label:
                raise ValueError(f"row {records}: text occurs with both labels")
    texts = list(seen)
    groups = [group_key(t) for t in texts]
    bad = np.array(labels, dtype=bool)
    by_group: dict[str, set] = {}
    for g, b in zip(groups, bad):
        by_group.setdefault(g, set()).add(bool(b))
    mixed = sorted((g for g, s in by_group.items() if len(s) == 2), key=lambda g: g.encode())
    return Dataset(path, file_sha256(path), records, texts, bad, groups, mixed)


# --------------------------------------------------------------------------------------
# Splits by URL path
# --------------------------------------------------------------------------------------

def cv5_fold_of_group(data: Dataset) -> dict[str, int]:
    """Five folds of path groups. The paths that hold both labels are dealt round-robin
    from the largest to the smallest (distinct texts; ties in byte order), so each fold
    holds one of the five large endpoints; three of the eight are tiny (165 to 250 texts,
    2 to 4 of them good) and would give a fold without a usable ROC AUC on their own.
    The single-label paths are dealt round-robin in byte order."""
    size = {g: 0 for g in data.mixed_groups}
    for g in data.groups:
        if g in size:
            size[g] += 1
    mixed = sorted(data.mixed_groups, key=lambda g: (-size[g], g.encode()))
    single = sorted(set(data.groups) - set(mixed), key=lambda g: g.encode())
    fold = {g: i % 5 for i, g in enumerate(mixed)}
    fold.update({g: i % 5 for i, g in enumerate(single)})
    return fold


def held_out_groups(data: Dataset, split: str, fold: int) -> set[str]:
    if split == "split17":
        # The sanity run of task 1.7: every third of the mixed paths in name order,
        # starting at the second; every other path is on the fitting side.
        return set(data.mixed_groups[1::3])
    if split == "cv5":
        return {g for g, f in cv5_fold_of_group(data).items() if f == fold}
    raise ValueError(split)


def validation_groups(data: Dataset, held: set[str], fold: int) -> set[str]:
    """A part of the fitting side, by path, for early stopping: one path with both labels
    (of the fitting-side mixed paths with at least 100 good and 100 bad texts, in byte
    order, entry `fold` modulo their number) and every fifth single-label fitting path."""
    good = {g: 0 for g in data.mixed_groups}
    bad = dict(good)
    for g, b in zip(data.groups, data.bad):
        if g in good:
            (bad if b else good)[g] += 1
    mixed = [g for g in data.mixed_groups if g not in held and min(good[g], bad[g]) >= 100]
    single = sorted(set(data.groups) - set(data.mixed_groups) - held, key=lambda g: g.encode())
    return {mixed[fold % len(mixed)]} | set(single[fold % 5::5])


# --------------------------------------------------------------------------------------
# Baseline: a re-implementation of `l2d-baseline` (hashing.rs, logreg.rs objective)
# --------------------------------------------------------------------------------------

FNV_BASIS = np.uint64(0xCBF29CE484222325)
FNV_PRIME = np.uint64(0x100000001B3)


def hashed_rows(texts: list[str], routing: str, buckets: int = 65_536, ngram_max: int = 3):
    """CSR matrix of the baseline's feature rows: FNV-1a 64 per byte n-gram of length
    1..ngram_max, bucket = low 32 bits mod buckets, sign = parity of the high 32 bits,
    signs summed per bucket, zero sums dropped, row scaled to unit L2 norm."""
    from scipy.sparse import csr_matrix

    indptr, indices, values = [0], [], []
    with np.errstate(over="ignore"):
        for text in texts:
            part = text if routing == "keep" else split_routing(text)[1]
            data = np.frombuffer(part.encode("utf-8"), dtype=np.uint8).astype(np.uint64)
            keys, signs = [], []
            for n in range(1, ngram_max + 1):
                if len(data) < n:
                    break
                h = np.full(len(data) - n + 1, FNV_BASIS, dtype=np.uint64)
                for j in range(n):
                    h = (h ^ data[j:len(data) - n + 1 + j]) * FNV_PRIME
                keys.append(((h & np.uint64(0xFFFFFFFF)) % np.uint64(buckets)).astype(np.int64))
                high = (h >> np.uint64(32)).astype(np.uint32)
                parity = np.bitwise_count(high) & 1
                signs.append(np.where(parity == 0, 1, -1).astype(np.int64))
            if keys:
                sums = np.bincount(np.concatenate(keys), weights=np.concatenate(signs),
                                   minlength=buckets)
                nz = np.flatnonzero(sums)
                if len(nz):
                    v = sums[nz]
                    indices.append(nz)
                    values.append(v / np.sqrt(np.sum(v * v)))
                    indptr.append(indptr[-1] + len(nz))
                    continue
            indptr.append(indptr[-1])
    idx = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
    val = np.concatenate(values) if values else np.zeros(0)
    return csr_matrix((val, idx, np.array(indptr)), shape=(len(texts), buckets))


def fit_logreg(x, y, l2: float):
    """Minimizes sum of log-losses + l2/2 |w|^2 with an unpenalized bias: the objective of
    `l2d_baseline::logreg`. scikit-learn's L-BFGS minimizes C * sum + 1/2 |w|^2, the same
    minimizer for C = 1 / l2."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0 / l2, solver="lbfgs", max_iter=5000, tol=1e-6)
    model.fit(x, y)
    return model


# --------------------------------------------------------------------------------------
# Encoder input
# --------------------------------------------------------------------------------------

def byte_ids(texts: list[str], routing: str) -> tuple[np.ndarray, np.ndarray]:
    """(ids [n, 256] uint8, truncated [n] bool). `keep`: the UTF-8 bytes of the text;
    `strip`: of the payload part only. The first 256 bytes, padded with byte 0."""
    ids = np.full((len(texts), INPUT_BYTES), PAD_BYTE, dtype=np.uint8)
    truncated = np.zeros(len(texts), dtype=bool)
    for row, text in enumerate(texts):
        part = text if routing == "keep" else split_routing(text)[1]
        raw = part.encode("utf-8")
        truncated[row] = len(raw) > INPUT_BYTES
        raw = raw[:INPUT_BYTES]
        ids[row, :len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return ids, truncated


def table_input(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """The graph input `[n, 64, 1, 256]` float32: x[n, c, 0, t] = table[ids[n, t], c]."""
    return np.ascontiguousarray(table[ids].transpose(0, 2, 1)[:, :, None, :], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Encoder (architecture v1 of the design, the shape of the NPU spike)
# --------------------------------------------------------------------------------------

def build_modules():
    import torch
    from torch import nn

    class Body(nn.Module):
        """The exported graph. Module names match the spike model (`f`, `head`)."""

        def __init__(self):
            super().__init__()
            plan = [(64, 128, 1), (128, 128, 2), (128, 256, 2), (256, 256, 2), (256, 256, 1)]
            self.f = nn.Sequential(*[
                nn.Sequential(nn.Conv2d(i, o, (1, 5), stride=(1, s), padding=(0, 2)), nn.ReLU())
                for i, o, s in plan])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(256, OUTPUT_DIM)

        def forward(self, x):
            return self.head(torch.flatten(self.pool(self.f(x)), 1))

    class Trainee(nn.Module):
        """Byte table + body + the training-only layers that are discarded at export."""

        def __init__(self, objective: str):
            super().__init__()
            self.objective = objective
            self.table = nn.Embedding(256, TABLE_WIDTH, padding_idx=PAD_BYTE)
            self.body = Body()
            self.classifier = nn.Linear(OUTPUT_DIM, 1)

        def embed(self, ids):
            x = self.table(ids)                               # [n, 256, 64]
            return self.body(x.transpose(1, 2).unsqueeze(2))  # [n, 64, 1, 256] -> [n, 128]

        def logits(self, e):
            if self.objective.startswith("bce-norm"):
                # The classifier sees what the tool stores: the L2-normalized vector.
                e = 16.0 * torch.nn.functional.normalize(e, dim=1)
            return self.classifier(e).squeeze(1)

    return Body, Trainee


def supcon_loss(e, y, temperature: float = 0.1):
    """Supervised contrastive loss on L2-normalized vectors (Khosla et al. 2020)."""
    import torch

    z = torch.nn.functional.normalize(e, dim=1)
    sim = z @ z.T / temperature
    eye = torch.eye(len(y), dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positive = (y[:, None] == y[None, :]) & ~eye
    count = positive.sum(1)
    use = count > 0
    loss = -(log_prob.masked_fill(~positive, 0.0).sum(1)[use] / count[use])
    return loss.mean()


def train_encoder(ids_train, y_train, ids_val, y_val, *, objective: str, seed: int, epochs: int,
                  batch: int, lr: float, weight_decay: float, threads: int, keep_last: bool = False,
                  log=print):
    """Trains on the training part and returns the weights of the epoch with the highest
    ROC AUC on the validation part (both parts are fitting-side). The validation figure is
    the ROC AUC of the 11-nearest-neighbour ratio of the validation vectors over the
    training vectors: the use the tool makes of the vectors."""
    import torch
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    _, Trainee = build_modules()
    model = Trainee(objective)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * ((len(ids_train) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    x = torch.from_numpy(ids_train.astype(np.int64))
    y = torch.from_numpy(y_train.astype(np.float32))
    xv = torch.from_numpy(ids_val.astype(np.int64))
    order = torch.Generator().manual_seed(seed)
    bce = torch.nn.BCEWithLogitsLoss()
    history, best = [], None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total, seen = 0.0, 0
        for index in torch.randperm(len(x), generator=order).split(batch):
            e = model.embed(x[index])
            loss = supcon_loss(e, y[index]) if objective == "supcon" else bce(model.logits(e), y[index])
            if objective == "bce-norm-unit":
                # Keep the length of the raw vector near 1, so that the 8-bit output range
                # is not set by a few long vectors (see the gate document).
                loss = loss + ((e.norm(dim=1) - 1.0) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        model.eval()
        with torch.no_grad():
            ev = torch.cat([model.embed(part) for part in xv.split(512)])
            et = torch.cat([model.embed(part) for part in x.split(512)])
            score = knn_ratio(l2n(et.numpy()), y_train.astype(bool), l2n(ev.numpy()), 11)[0]
            head = None if objective == "supcon" else model.logits(ev).numpy()
        both = 0 < y_val.sum() < len(y_val)
        val_auc = float(roc_auc_score(y_val, score)) if both else float("nan")
        head_auc = float(roc_auc_score(y_val, head)) if both and head is not None else None
        norms = np.linalg.norm(et.numpy(), axis=1)
        history.append({"epoch": epoch, "train_loss": total / seen, "val_roc_auc": val_auc,
                        "val_roc_auc_training_classifier": head_auc,
                        "vector_norm_p01_p50_p99": [float(v) for v in np.percentile(norms, [1, 50, 99])],
                        "seconds": time.perf_counter() - started})
        log(f"    epoch {epoch}: loss {total / seen:.4f}  validation ROC AUC neighbours {val_auc:.4f}"
            f"  classifier {head_auc if head_auc is None else round(head_auc, 4)}  norms "
            f"{[round(v, 2) for v in history[-1]['vector_norm_p01_p50_p99']]}  {history[-1]['seconds']:.0f} s")
        if keep_last or best is None or val_auc > best[0]:
            best = (val_auc, epoch, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    model.eval()
    return model, {"history": history, "chosen_epoch": best[1], "chosen_val_roc_auc": best[0]}


def float_vectors(model, ids: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        parts = [model.embed(torch.from_numpy(p.astype(np.int64))).numpy()
                 for p in np.array_split(ids, max(1, len(ids) // 512))]
    return np.concatenate(parts)


# --------------------------------------------------------------------------------------
# ONNX export, 8-bit QDQ, inference through ONNX Runtime
# --------------------------------------------------------------------------------------

def export_float(model, path: str) -> None:
    """Static shapes of the spike: input `x` [32, 64, 1, 256] float32, output `e` [32, 128]."""
    import torch

    torch.onnx.export(model.body, torch.zeros(BATCH, TABLE_WIDTH, 1, INPUT_BYTES), path,
                      input_names=["x"], output_names=["e"], opset_version=17, dynamo=False,
                      do_constant_folding=True)


def quantize_qdq(float_path: str, int8_path: str, table: np.ndarray, calibration_ids: np.ndarray) -> None:
    """Static 8-bit QDQ with ONNX Runtime's quantizer: uint8 activations, int8 symmetric
    weights, per tensor, min/max calibration over `calibration_ids` (fitting side)."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    assert len(calibration_ids) % BATCH == 0

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = iter(np.split(calibration_ids, len(calibration_ids) // BATCH))

        def get_next(self):
            ids = next(self.batches, None)
            return None if ids is None else {"x": table_input(table, ids)}

    quantize_static(float_path, int8_path, Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=False, calibrate_method=CalibrationMethod.MinMax)


def onnx_vectors(path: str, table: np.ndarray, ids: np.ndarray, threads: int) -> np.ndarray:
    """Runs a static-batch model on the CPU provider; the last batch is padded with empty
    inputs and the padded outputs are dropped, as `EncoderSession` will do."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    out = np.empty((len(ids), OUTPUT_DIM), dtype=np.float32)
    for start in range(0, len(ids), BATCH):
        chunk = ids[start:start + BATCH]
        n = len(chunk)
        if n < BATCH:
            chunk = np.concatenate([chunk, np.full((BATCH - n, INPUT_BYTES), PAD_BYTE, np.uint8)])
        out[start:start + n] = session.run(["e"], {"x": table_input(table, chunk)})[0][:n]
    return out


# --------------------------------------------------------------------------------------
# Scores from vectors, the way the tool will use them
# --------------------------------------------------------------------------------------

def l2n(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(norm, 1e-12)).astype(np.float32)


def knn_ratio(store: np.ndarray, store_bad: np.ndarray, query: np.ndarray, k: int,
              reach: float | None = None, min_support: int = 3):
    """Task 2.8: the share of bad labels among the k nearest stored vectors (unit vectors,
    squared L2 distance = 2 - 2 cos). With `reach`, only neighbours within that squared
    distance count and a query with fewer than `min_support` of them abstains (NaN)."""
    ratios = np.empty(len(query), dtype=np.float64)
    abstained = np.zeros(len(query), dtype=bool)
    bad = store_bad.astype(np.float64)
    for start in range(0, len(query), 1024):
        sim = query[start:start + 1024] @ store.T
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.arange(len(top))[:, None]
        if reach is None:
            ratios[start:start + len(top)] = bad[top].mean(axis=1)
            continue
        valid = (2.0 - 2.0 * sim[rows, top]) <= reach
        found = valid.sum(axis=1)
        share = (bad[top] * valid).sum(axis=1) / np.maximum(found, 1)
        out = found < min_support
        ratios[start:start + len(top)] = np.where(out, np.nan, share)
        abstained[start:start + len(top)] = out
    return ratios, abstained


def group_hash_fold(group: str, folds: int = 5) -> int:
    return int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "little") % folds


def linear_head(store: np.ndarray, store_bad: np.ndarray, store_groups: list[str], query: np.ndarray):
    """Task 2.10 / design `LinearHead`: logistic regression on the unit vectors; the penalty
    is chosen from 0.01, 0.1, 1, 10, 100 by 5-fold grouped cross-validation on log loss
    over the fitting side, folds from a hash of the group key."""
    from sklearn.metrics import log_loss

    folds = np.array([group_hash_fold(g) for g in store_groups])
    x = store.astype(np.float64)
    losses = {}
    for l2 in (0.01, 0.1, 1.0, 10.0, 100.0):
        parts = []
        for fold in range(5):
            fit, test = folds != fold, folds == fold
            if test.sum() == 0 or len(set(store_bad[fit])) < 2:
                continue
            p = fit_logreg(x[fit], store_bad[fit], l2).predict_proba(x[test])[:, 1]
            parts.append(log_loss(store_bad[test], p, labels=[False, True]) * test.sum())
        losses[l2] = float(sum(parts) / len(x))
    chosen = min(losses, key=lambda l2: (losses[l2], l2))
    model = fit_logreg(x, store_bad, chosen)
    return model.predict_proba(query.astype(np.float64))[:, 1], {"penalty": chosen, "cv_log_loss": losses}


# --------------------------------------------------------------------------------------
# Metrics (requirements 9.1, 9.2, 9.8). The only place held-out labels are read.
# --------------------------------------------------------------------------------------

def metrics(score: np.ndarray, bad: np.ndarray, groups: list[str], mixed: set[str],
            abstained: np.ndarray | None = None) -> dict:
    """`cross_endpoint`: the held-out logs of paths that hold both labels, pooled.
    `overall`: all held-out logs. `single_label`: the logs of single-label paths, reported
    separately with the share of them inside a review budget equal to the number of bad
    held-out logs. Abstained logs are left out and counted."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    groups = np.array(groups)
    in_mixed = np.isin(groups, sorted(mixed))
    if abstained is None:
        abstained = np.zeros(len(score), dtype=bool)

    def block(mask):
        use = mask & ~abstained
        out = {"logs": int(mask.sum()), "bad": int(bad[mask].sum()),
               "abstained": int((mask & abstained).sum())}
        if 0 < bad[use].sum() < use.sum():
            out["roc_auc"] = float(roc_auc_score(bad[use], score[use]))
            out["average_precision"] = float(average_precision_score(bad[use], score[use]))
        return out

    result = {"cross_endpoint": block(in_mixed), "overall": block(np.ones(len(score), dtype=bool)),
              "per_group": {g: block(groups == g) for g in sorted(set(groups[in_mixed]))}}
    single = ~in_mixed
    budget = int(bad.sum())
    filled = np.where(abstained, -np.inf, score)
    order = np.argsort(-filled, kind="stable")
    in_budget = np.zeros(len(score), dtype=bool)
    in_budget[order[:budget]] = True
    result["single_label"] = {
        "logs": int(single.sum()), "bad": int(bad[single].sum()),
        "good": int((~bad[single]).sum()), "abstained": int((single & abstained).sum()),
        "review_budget": budget,
        "bad_share_in_budget": float(in_budget[single & bad].mean()) if (single & bad).any() else None,
        "good_share_in_budget": float(in_budget[single & ~bad].mean()) if (single & ~bad).any() else None,
    }
    return result
