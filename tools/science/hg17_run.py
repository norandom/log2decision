"""Steps of the stability gate of the entropy-patch encoder (task 2.17). Development only.

Every step is one process, runs under a memory limit (`--memory-gib`, default 6) as a
Windows job object, takes the project's lock for hungry processes, and records its peak
memory. Run alone on the machine, one step at a time (`hg17_sweep.ps1`).

    uv run python hg17_run.py prepare
    uv run python hg17_run.py bytes   --fold 0 --seed 0 [--tag repeat]
    uv run python hg17_run.py line    --fold 0 --seed 0 --variant mask [--tag repeat]
    uv run python hg17_run.py eval    --fold 0 --seed 0 --variant mask
    uv run python hg17_run.py refs
    uv run python hg17_run.py determinism --fold 0 --seed 0 --variant mask
    uv run python hg17_run.py npu-fixture --fold 0 --seed 0 --variant mask
    uv run python hg17_run.py npu-compare --fold 0 --seed 0 --variant mask
    uv run python hg17_run.py export  --fold 0 --seed 0 --variant mask   (layout of the q-scripts)
    uv run python hg17_run.py summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from importlib.metadata import version

import numpy as np

import hg
import hg17
import l2dsci
import memguard

VARIANTS = {
    # name: (objective, label weight)
    "mask": ("masked", 0.0),
    "contrastive": ("contrastive", 0.0),
    "mask-label": ("masked", 0.5),
}
LM = {"epochs": 8, "batch": 64, "lr": 2e-3, "weight_decay": 0.01}
LINE = {"epochs": 12, "batch": 256, "lr": 1e-3, "weight_decay": 0.01, "temperature": 0.1, "mask_rate": 0.3}


def sha256_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    return {"packages": {n: version(n) for n in ("numpy", "torch", "onnx", "onnxruntime", "scikit-learn", "scipy")},
            "python": platform.python_version(), "host": platform.processor(), "host_memory_at_start": memguard.host_memory()}


def enc_dir(fold: int, seed: int, tag: str = "") -> str:
    return hg17.path("enc", f"f{fold}-s{seed}" + (f"-{tag}" if tag else ""), "x")[:-2]


def model_dir(fold: int, seed: int, tag: str = "") -> str:
    out = os.path.join(hg17.MODELS, f"f{fold}-s{seed}" + (f"-{tag}" if tag else ""))
    os.makedirs(out, exist_ok=True)
    return out


# --------------------------------------------------------------------------------------
# prepare: the corpora as memory-mapped bytes, labels, folds, attack/benign pairs
# --------------------------------------------------------------------------------------

def params_of(text: str):
    routing, payload = l2dsci.split_routing(text)
    items = [kv.split("=", 1) if "=" in kv else [kv, ""] for kv in payload.replace("\n", "&").split("&") if kv]
    return routing.split(" ")[0], tuple(k for k, _ in items), tuple(v for _, v in items)


def step_prepare(args) -> dict:
    # Reading the CSV builds the set of distinct texts (25,608, about 10 MB): the one
    # whole-dataset structure of the spike, needed for deduplication, written to disk at once.
    data = l2dsci.read_csic(hg17.CSIC)
    hg17.write_corpus("csic", data.texts)
    probes = hg.read_probes()
    hg17.write_corpus("probes", [r["text"] for r in probes])
    with open(hg17.path("data", "probes.jsonl"), "w", encoding="utf-8") as handle:
        for row in probes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fold_of = l2dsci.cv5_fold_of_group(data)
    fold = [fold_of[g] for g in data.groups]
    # Pairs: an attack and a benign request of the same endpoint, method and parameter
    # names that differ in exactly one parameter value; the control: two benign requests
    # that differ in exactly one value.
    index: dict = {}
    for i, text in enumerate(data.texts):
        method, keys, _ = params_of(text)
        index.setdefault((data.groups[i], method, keys), []).append(i)
    attack, control = [], []
    for rows in index.values():
        goods = [r for r in rows if not data.bad[r]]
        bads = [r for r in rows if data.bad[r]]
        by_rest: dict = {}
        for g in goods:
            values = params_of(data.texts[g])[2]
            for j in range(len(values)):
                key = (j, values[:j] + values[j + 1:])
                if key in by_rest and by_rest[key] != g and len(control) < 200_000:
                    control.append((by_rest[key], g))
                by_rest.setdefault(key, g)
        for b in bads:
            values = params_of(data.texts[b])[2]
            for j in range(len(values)):
                key = (j, values[:j] + values[j + 1:])
                if key in by_rest:
                    attack.append((by_rest[key], b))
                    break
    rng = np.random.default_rng(20261114)
    control = sorted(set(control))
    if len(control) > 5000:
        control = [control[i] for i in sorted(rng.choice(len(control), 5000, replace=False))]
    hg17.write_json(hg17.path("data", "csic.labels.json"), {
        "sha256": data.sha256, "records": data.records, "bad": data.bad.astype(int).tolist(), "groups": data.groups,
        "mixed_groups": data.mixed_groups, "fold": fold, "split17_held_groups": sorted(l2dsci.held_out_groups(data, "split17", 0)),
        "attack_pairs": attack, "control_pairs": control})
    sizes = {f: {"texts": int(sum(1 for x in fold if x == f)),
                 "mixed_paths": sorted(g for g in data.mixed_groups if fold_of[g] == f)} for f in range(5)}
    return {"texts": len(data.texts), "probes": len(probes), "records": data.records, "folds": sizes,
            "attack_pairs": len(attack), "control_pairs": len(control)}


# --------------------------------------------------------------------------------------
# bytes: the byte model of one fold and seed, its 8-bit graph, the cuts, the patch features
# --------------------------------------------------------------------------------------

def step_bytes(args) -> dict:
    import torch

    settings = hg17.seed_everything(args.seed, args.threads)
    corpus, probes, labels = hg17.Corpus("csic"), hg17.Corpus("probes"), hg17.Labels()
    out, mdir = enc_dir(args.fold, args.seed, args.tag), model_dir(args.fold, args.seed, args.tag)
    fit = labels.fit(args.fold)
    train_rows = np.flatnonzero(fit & (corpus.lens > 0))
    held_rows = np.flatnonzero(~fit)
    model, history = hg17.train_byte_model(corpus, train_rows, held_rows, seed=args.seed, **LM)
    torch.save(model.state_dict(), os.path.join(mdir, "byte.pt"))
    table = model.table.weight.detach().numpy().astype("<f4")
    assert not table[0].any()
    table.tofile(os.path.join(mdir, "table.bin"))

    # The exported byte graph with both outputs, and its 8-bit form.
    _, ByteGraph = hg.build_byte_model()
    float_path, int8_path = os.path.join(mdir, "byte.f32.onnx"), os.path.join(mdir, "byte.int8.onnx")
    hg.export_onnx(ByteGraph(model, True).eval(), (1, hg17.TABLE_WIDTH, hg17.ROWS_BYTE, hg17.L_MAX), ["h", "y"], float_path)
    rng = np.random.default_rng(args.seed)
    calibration = np.sort(rng.choice(train_rows, size=256, replace=False))
    images = [hg.byte_image(table, corpus.ids(calibration[i:i + hg17.ROWS_BYTE], hg17.L_MAX), hg17.ROWS_BYTE)
              for i in range(0, len(calibration), hg17.ROWS_BYTE)]
    hg.quantize(float_path, int8_path, images)
    del images
    scale, zero = hg17.dequantize_scale(int8_path, "y")
    assert zero == 0
    exp_table = hg17.exp_table(scale)
    sess = hg17.session(int8_path, args.threads)

    def run_image(c: hg17.Corpus, rows: np.ndarray):
        ids = c.ids(rows, hg17.L_MAX)
        h, y = sess.run(None, {"x": hg.byte_image(table, ids, hg17.ROWS_BYTE)})
        return h[0], y[0]                                    # [96, R, L], [256, R, L]

    # The budget of 2.16: the mean patch count of the fixed stride on the training lines.
    target = float(np.mean([len(hg.starts_fixed(int(n))) for n in corpus.lens[train_rows]]))
    sample = np.sort(np.random.default_rng(args.seed + 1000).choice(train_rows, size=4096, replace=False))
    sample_ent = []
    for i in range(0, len(sample), hg17.ROWS_BYTE):
        rows = sample[i:i + hg17.ROWS_BYTE]
        _, y = run_image(corpus, rows)
        ent = hg17.int8_entropies(y, scale, exp_table)
        sample_ent += [ent[k, :int(corpus.lens[r])] for k, r in enumerate(rows)]
    low, high = -10.0, 10.0
    for _ in range(40):
        mid = (low + high) / 2
        if np.mean([len(hg17.starts_jump(e, mid)) for e in sample_ent]) > target:
            low = mid
        else:
            high = mid
    threshold = (low + high) / 2
    del sample_ent

    # One pass over every text: the 8-bit graph's features and entropies, the cuts, the
    # float model's features pooled at the same cuts; and, for the measurement only, the
    # cuts the float model would make itself (raw, and with its output on the 8-bit grid).
    report = {"lines": {}, "same_cuts_float_model": {}, "same_cuts_float_output_on_8bit_grid": {},
              "patches_mean": {}, "lines_at_patch_limit": {}}
    entropy_diff = []
    with torch.no_grad():
        for c in (corpus, probes):
            w32 = hg17.PatchWriter(os.path.join(out, f"{c.name}-patches-f32"))
            w8 = hg17.PatchWriter(os.path.join(out, f"{c.name}-patches-int8"))
            starts_file = open(os.path.join(out, f"{c.name}-starts.i32"), "wb")
            same_float = same_fq = 0
            for i in range(0, len(c), hg17.ROWS_BYTE):
                rows = np.arange(i, min(i + hg17.ROWS_BYTE, len(c)))
                h, y = run_image(c, rows)
                ent8 = hg17.int8_entropies(y, scale, exp_table)
                width = max(1, int(c.lens[rows].max()))
                f, raw = model.forward_ids(torch.from_numpy(c.ids(rows, width).astype(np.int64)))
                f, raw = f.numpy(), raw.numpy()
                ent_f = hg17.float_entropies(raw.transpose(1, 0, 2).reshape(256, -1)).reshape(len(rows), width)
                ent_fq = hg17.fake_quant_entropies(raw.transpose(1, 0, 2), scale, exp_table)
                for k, row in enumerate(rows):
                    n = int(c.lens[row])
                    starts = hg17.starts_jump(ent8[k, :n], threshold)
                    w8.add(hg17.pool(h[:, k, :], starts, n))
                    w32.add(hg17.pool(f[k], starts, n))
                    starts_file.write(starts.astype("<i4").tobytes())
                    same_float += np.array_equal(starts, hg17.starts_jump(ent_f[k, :n], threshold))
                    same_fq += np.array_equal(starts, hg17.starts_jump(ent_fq[k, :n], threshold))
                    if c is corpus and row % 8 == 0 and n:
                        entropy_diff.append(np.abs(ent8[k, :n] - ent_f[k, :n]))
            w32.close()
            w8.close()
            starts_file.close()
            counts = np.array(w8.counts)
            report["lines"][c.name] = len(c)
            report["same_cuts_float_model"][c.name] = int(same_float)
            report["same_cuts_float_output_on_8bit_grid"][c.name] = int(same_fq)
            report["patches_mean"][c.name] = float(counts.mean())
            report["lines_at_patch_limit"][c.name] = int((counts == hg17.P_MAX).sum())
            hg17.Patches(os.path.join(out, f"{c.name}-patches-int8"))   # readable
    diff = np.concatenate(entropy_diff)
    record = {"task": "npu-log-decision 2.17", "step": "bytes", "fold": args.fold, "seed": args.seed, "tag": args.tag,
              **environment(), "determinism": settings, "settings": {**LM, "optimizer": "AdamW, one-cycle (10 % warm-up)",
              "objective": "next-byte cross-entropy + 0.1 x squared log-normaliser, no label",
              "trained_on": "every CSIC text of the fitting side of the fold (the four other folds)"},
              "training_lines": int(len(train_rows)), "history": history,
              "patching": {"rule": "jump (BLT monotonic)", "threshold": threshold, "budget_patches_per_training_line": target,
                           "calibrated_on": "4,096 training lines, entropies of the 8-bit graph",
                           "logit_scale": scale, **report,
                           "entropy_8bit_minus_float_abs_mean_p99": [float(diff.mean()), float(np.percentile(diff, 99))]},
              "files": {"byte_int8_sha256": sha256_file(int8_path), "byte_f32_sha256": sha256_file(float_path),
                        "byte_pt_sha256": sha256_file(os.path.join(mdir, "byte.pt")),
                        "table_sha256": sha256_file(os.path.join(mdir, "table.bin")),
                        "patches_f32_sha256": sha256_file(os.path.join(out, "csic-patches-f32.f32")),
                        "patches_int8_sha256": sha256_file(os.path.join(out, "csic-patches-int8.f32")),
                        "starts_sha256": sha256_file(os.path.join(out, "csic-starts.i32"))}}
    hg17.write_json(os.path.join(out, "bytes.json"), record)
    print(f"bytes f{args.fold} s{args.seed}{args.tag}: threshold {threshold:.4f}, patches {report['patches_mean']}, "
          f"same cuts float model {report['same_cuts_float_model']}, float output on 8-bit grid {report['same_cuts_float_output_on_8bit_grid']}")
    return record


# --------------------------------------------------------------------------------------
# line: the line model of one fold, seed and variant; its 8-bit graph; the vectors
# --------------------------------------------------------------------------------------

def step_line(args) -> dict:
    import torch

    settings = hg17.seed_everything(args.seed, args.threads)
    objective, label_weight = VARIANTS[args.variant]
    labels = hg17.Labels()
    edir = enc_dir(args.fold, args.seed, args.tag)
    out = os.path.join(edir, args.variant)
    mdir = os.path.join(model_dir(args.fold, args.seed, args.tag), args.variant)
    os.makedirs(out, exist_ok=True)
    os.makedirs(mdir, exist_ok=True)
    p32 = hg17.Patches(os.path.join(edir, "csic-patches-f32"))
    p8 = hg17.Patches(os.path.join(edir, "csic-patches-int8"))
    fit = labels.fit(args.fold)
    rows_train = np.flatnonzero(fit & (p32.counts > 0))
    patch_bytes = hg17.PatchBytes(hg17.Corpus("csic"), os.path.join(edir, "csic-starts.i32"), p32.counts)
    model, history, extra, held_loss = hg17.train_line_model(
        p32, labels.bad, rows_train, seed=args.seed, objective=objective, label_weight=label_weight,
        patch_bytes=patch_bytes, **LINE)
    torch.save({"model": model.state_dict(), **{k: v.state_dict() for k, v in extra.items() if v is not None}},
               os.path.join(mdir, "line.pt"))
    held = held_loss(np.flatnonzero(~fit))
    print(f"  objective on the held-out fold (no label read): {json.dumps({k: round(v, 4) for k, v in held.items()})}")

    _, Graph = hg17.build_line_model()
    float_path, int8_path = os.path.join(mdir, "line.f32.onnx"), os.path.join(mdir, "line.int8.onnx")
    hg.export_onnx(Graph(model).eval(), (1, hg17.FEATURES, hg17.ROWS_PATCH, hg17.P_MAX), ["e"], float_path)
    rng = np.random.default_rng(args.seed)
    calibration = np.sort(rng.choice(rows_train, size=512, replace=False))
    images = []
    for i in range(0, len(calibration), hg17.ROWS_PATCH):
        x = p8.batch(calibration[i:i + hg17.ROWS_PATCH], hg17.P_MAX)
        image = np.zeros((1, hg17.FEATURES, hg17.ROWS_PATCH, hg17.P_MAX), dtype=np.float32)
        image[0, :, :len(x), :] = x[:, :, 0, :].transpose(1, 0, 2)
        images.append(image)
    hg.quantize(float_path, int8_path, images)
    del images
    sess = hg17.session(int8_path, args.threads)
    files = {"line_pt_sha256": sha256_file(os.path.join(mdir, "line.pt")), "line_f32_sha256": sha256_file(float_path),
             "line_int8_sha256": sha256_file(int8_path)}
    for name in ("csic", "probes"):
        f32 = hg17.Patches(os.path.join(edir, f"{name}-patches-f32"))
        i8 = hg17.Patches(os.path.join(edir, f"{name}-patches-int8"))
        rows = np.arange(len(f32.counts))
        for precision, vectors in (("f32", hg17.float_vectors(model, f32, rows)), ("int8", hg17.onnx_vectors(sess, i8, rows))):
            file_path = os.path.join(out, f"{name}-{precision}.f32")
            vectors.astype("<f4").tofile(file_path)
            files[f"{name}_{precision}_sha256"] = sha256_file(file_path)
    a = np.fromfile(os.path.join(out, "csic-f32.f32"), dtype="<f4").reshape(-1, hg17.OUTPUT_DIM)
    b = np.fromfile(os.path.join(out, "csic-int8.f32"), dtype="<f4").reshape(-1, hg17.OUTPUT_DIM)
    cosine = np.sum(l2dsci.l2n(a) * l2dsci.l2n(b), axis=1)
    record = {"task": "npu-log-decision 2.17", "step": "line", "fold": args.fold, "seed": args.seed, "variant": args.variant,
              "tag": args.tag, **environment(), "determinism": settings,
              "settings": {**LINE, "objective": objective, "label_weight": label_weight,
                           "optimizer": "AdamW, one-cycle (10 % warm-up)", "layers": hg17.LINE_LAYERS,
                           "kernel": [1, hg17.LINE_KERNEL], "strides": 1, "width": hg17.PATCH_WIDTH,
                           "input": "float byte features pooled at the 8-bit graph's cuts",
                           "pooling": "mean over the line's real patches, after the graph",
                           "trained_on": "every CSIC text of the fitting side of the fold", "calibration_lines": 512},
              "training_lines": int(len(rows_train)), "history": history, "held_out_fold_objective": held,
              "parameters": int(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("classifier"))),
              "float_against_8bit_cosine_min_p01_median": [float(cosine.min()), float(np.percentile(cosine, 1)), float(np.median(cosine))],
              "raw_length_p01_p50_p99_int8": [float(v) for v in np.percentile(np.linalg.norm(b, axis=1), [1, 50, 99])],
              "files": files}
    hg17.write_json(os.path.join(out, "line.json"), record)
    print(f"line f{args.fold} s{args.seed} {args.variant}{args.tag}: float against 8-bit cosine median {np.median(cosine):.4f}, "
          f"p01 {np.percentile(cosine, 1):.4f}")
    return record


# --------------------------------------------------------------------------------------
# eval: what the vectors of one encoder say, on seen and unseen endpoints
# --------------------------------------------------------------------------------------

def classify(x: np.ndarray, labels: hg17.Labels, fold: int, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    bad, mixed = labels.bad, labels.mixed
    fit, held = labels.fit(fold), labels.held(fold)
    fit_rows = np.flatnonzero(fit)
    groups_fit = [labels.groups[i] for i in fit_rows]
    out = {"unseen": {}, "seen": {}}
    use = held & mixed
    probability, head = l2dsci.linear_head(x[fit], bad[fit], groups_fit, x[held])
    score = np.zeros(len(x))
    score[held] = probability
    out["unseen"]["linear"] = float(roc_auc_score(bad[use], score[use]))
    out["linear_penalty"] = head["penalty"]
    boost = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(x[fit], bad[fit])
    score[held] = boost.predict_proba(x[held])[:, 1]
    out["unseen"]["boosted"] = float(roc_auc_score(bad[use], score[use]))
    ratio, _ = l2dsci.knn_ratio(x[fit], bad[fit], x[held], 11)
    score[held] = ratio
    out["unseen"]["knn11"] = float(roc_auc_score(bad[use], score[use]))
    # Seen: five stratified folds inside the fitting side; every endpoint on both sides.
    oof = {"linear": np.zeros(len(fit_rows)), "boosted": np.zeros(len(fit_rows))}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(fit_rows, bad[fit_rows]):
        ra, rb = fit_rows[a], fit_rows[b]
        oof["linear"][b] = l2dsci.fit_logreg(x[ra], bad[ra], head["penalty"]).predict_proba(x[rb])[:, 1]
        oof["boosted"][b] = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(x[ra], bad[ra]).predict_proba(x[rb])[:, 1]
    m = mixed[fit_rows]
    for name, s in oof.items():
        out["seen"][name] = float(roc_auc_score(bad[fit_rows][m], s[m]))
    return out


def spectrum(x: np.ndarray, labels: hg17.Labels, fold: int) -> dict:
    fit, held = labels.fit(fold), labels.held(fold)
    mean = x[fit].mean(axis=0)
    centred = x[fit] - mean
    values, vectors = np.linalg.eigh(centred.T @ centred / (fit.sum() - 1))
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    share = values / values.sum()
    k95 = int(np.searchsorted(np.cumsum(share), 0.95) + 1)
    nz = share[share > 0]
    # The good/bad difference inside the endpoints that hold both labels.
    p_fit = (x[fit & labels.mixed] - mean) @ vectors
    b_fit = labels.bad[fit & labels.mixed]
    p_held = (x[held & labels.mixed] - mean) @ vectors
    b_held = labels.bad[held & labels.mixed]
    diff = p_fit[b_fit].mean(axis=0) - p_fit[~b_fit].mean(axis=0)
    diff_h = p_held[b_held].mean(axis=0) - p_held[~b_held].mean(axis=0)
    weight = diff ** 2 / (diff ** 2).sum()
    agree = np.sign(diff) == np.sign(diff_h)
    within = 0.5 * (p_fit[b_fit].var(axis=0) + p_fit[~b_fit].var(axis=0))
    fisher = diff ** 2 / np.maximum(within, 1e-30)
    top = np.argsort(fisher)[::-1][:10]
    return {"components_95": k95, "share_top10": [float(s) for s in share[:10]],
            "spectrum_entropy_bits": float(-(nz * np.log2(nz)).sum()),
            "eigenvalues_above_1pct": int((share > 0.01).sum()),
            "sign_kept_weighted": float((weight * agree).sum()),
            "sign_kept_top10_fisher": int(agree[top].sum()),
            "cosine_of_differences": float(diff @ diff_h / (np.linalg.norm(diff) * np.linalg.norm(diff_h) + 1e-30))}


def pairs(x: np.ndarray, labels: hg17.Labels, fold: int, meta: dict) -> dict:
    from sklearn.metrics import roc_auc_score

    attack = np.array(meta["attack_pairs"], dtype=np.int64)
    control = np.array(meta["control_pairs"], dtype=np.int64)
    out = {}
    for part, keep_a, keep_c in (("all", np.ones(len(attack), bool), np.ones(len(control), bool)),
                                 ("held_fold", labels.held(fold)[attack[:, 0]], labels.held(fold)[control[:, 0]])):
        ca = np.sum(x[attack[keep_a, 0]] * x[attack[keep_a, 1]], axis=1)
        cc = np.sum(x[control[keep_c, 0]] * x[control[keep_c, 1]], axis=1)
        block = {"attack_pairs": int(keep_a.sum()), "control_pairs": int(keep_c.sum())}
        if len(ca):
            block["attack_cosine_median"] = float(np.median(ca))
        if len(cc):
            block["control_cosine_median"] = float(np.median(cc))
        if len(ca) and len(cc):
            # Does an attack move a vector further than a benign change of one value?
            block["auc_attack_further"] = float(roc_auc_score(np.r_[np.ones(len(ca)), np.zeros(len(cc))], -np.r_[ca, cc]))
        out[part] = block
    return out


def folding(raw: np.ndarray) -> dict:
    _, inverse, counts = np.unique(raw.view(np.dtype((np.void, raw.dtype.itemsize * raw.shape[1]))).ravel(),
                                   return_inverse=True, return_counts=True)
    return {"distinct_vectors": int(len(counts)), "inputs_that_share_a_vector": int(counts[counts > 1].sum())}


def evaluate(raw: np.ndarray, labels: hg17.Labels, fold: int, seed: int, meta: dict) -> dict:
    x = l2dsci.l2n(raw).astype(np.float64)
    return {"classify": classify(x, labels, fold, seed), "spectrum": spectrum(x, labels, fold),
            "pairs": pairs(x, labels, fold, meta), "folding": folding(raw)}


def step_eval(args) -> dict:
    hg17.seed_everything(args.seed, args.threads)
    labels = hg17.Labels()
    meta = hg17.read_json(hg17.path("data", "csic.labels.json"))
    out = os.path.join(enc_dir(args.fold, args.seed, args.tag), args.variant)
    result = {"fold": args.fold, "seed": args.seed, "variant": args.variant}
    for precision in ("f32", "int8", "npu"):
        file_path = os.path.join(out, f"csic-{precision}.f32")
        if not os.path.exists(file_path):
            continue
        raw = np.fromfile(file_path, dtype="<f4").reshape(-1, hg17.OUTPUT_DIM)
        result[precision] = evaluate(raw, labels, args.fold, args.seed, meta)
        c = result[precision]["classify"]
        print(f"eval f{args.fold} s{args.seed} {args.variant} {precision}: unseen linear {c['unseen']['linear']:.3f} "
              f"boosted {c['unseen']['boosted']:.3f} knn {c['unseen']['knn11']:.3f}; seen linear {c['seen']['linear']:.3f} "
              f"boosted {c['seen']['boosted']:.3f}; sign kept {result[precision]['spectrum']['sign_kept_weighted']:.3f}")
    hg17.write_json(os.path.join(out, "eval.json"), result)
    return result


def step_refs(args) -> dict:
    """The published encoder (its NPU vectors of task 2.15) and the baseline scorer, on
    the same folds; seeds as the encoders (they change only the boosted trees and the
    seen folds)."""
    labels = hg17.Labels()
    meta = hg17.read_json(hg17.path("data", "csic.labels.json"))
    published_dir = r"D:\BuildArtifacts\log2decision\task-2.15\vectors"
    with open(os.path.join(published_dir, "csic.jsonl"), encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in handle]
    corpus = hg17.Corpus("csic")
    assert len(texts) == len(corpus) and all(t.encode("utf-8")[:hg17.L_MAX].__eq__(bytes(corpus.raw(i))) for i, t in enumerate(texts))
    published = np.fromfile(os.path.join(published_dir, "csic.f32"), dtype="<f4").reshape(len(texts), -1)
    out = {"published": {}, "baseline": {}}
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    features = l2dsci.hashed_rows(texts, "keep")
    for fold in range(5):
        for seed in (0, 1, 2):
            out["published"][f"f{fold}-s{seed}"] = evaluate(published, labels, fold, seed, meta)
            fit, held = labels.fit(fold), labels.held(fold)
            model = l2dsci.fit_logreg(features[fit], labels.bad[fit], l2=1.0)
            use = held & labels.mixed
            unseen = float(roc_auc_score(labels.bad[use], model.predict_proba(features[use])[:, 1]))
            fit_rows = np.flatnonzero(fit)
            oof = np.zeros(len(fit_rows))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(fit_rows, labels.bad[fit_rows]):
                oof[b] = l2dsci.fit_logreg(features[fit_rows[a]], labels.bad[fit_rows[a]], l2=1.0).predict_proba(features[fit_rows[b]])[:, 1]
            m = labels.mixed[fit_rows]
            out["baseline"][f"f{fold}-s{seed}"] = {"unseen": unseen, "seen": float(roc_auc_score(labels.bad[fit_rows][m], oof[m]))}
            print(f"refs f{fold} s{seed}: published unseen linear {out['published'][f'f{fold}-s{seed}']['classify']['unseen']['linear']:.3f}, "
                  f"baseline unseen {unseen:.3f} seen {out['baseline'][f'f{fold}-s{seed}']['seen']:.3f}")
    hg17.write_json(hg17.path("refs", "refs.json"), out)
    return {"written": hg17.path("refs", "refs.json")}


# --------------------------------------------------------------------------------------
# (a) determinism
# --------------------------------------------------------------------------------------

def step_determinism(args) -> dict:
    """In this process: the vectors of one encoder computed twice, and in another batch
    composition, against each other and against the files the `line` step wrote in its
    own process. Across trainings: the `repeat` encoder (same fold, seed and variant,
    trained again in new processes) against the first, file by file."""
    import torch

    hg17.seed_everything(args.seed, args.threads)
    edir = enc_dir(args.fold, args.seed)
    out = os.path.join(edir, args.variant)
    mdir = os.path.join(model_dir(args.fold, args.seed), args.variant)
    LineNet, _ = hg17.build_line_model()
    model = LineNet()
    model.load_state_dict(torch.load(os.path.join(mdir, "line.pt"))["model"])
    model.eval()
    sess = hg17.session(os.path.join(mdir, "line.int8.onnx"), args.threads)
    p32 = hg17.Patches(os.path.join(edir, "csic-patches-f32"))
    p8 = hg17.Patches(os.path.join(edir, "csic-patches-int8"))
    rows = np.arange(len(p32.counts))
    saved = {p: np.fromfile(os.path.join(out, f"csic-{p}.f32"), dtype="<f4").reshape(-1, hg17.OUTPUT_DIM) for p in ("f32", "int8")}
    result = {}

    def compare(a, b):
        same = (a == b).all(axis=1)
        cosine = np.sum(l2dsci.l2n(a) * l2dsci.l2n(b), axis=1)
        return {"bit_identical_rows": int(same.sum()), "rows": len(a), "cosine_min": float(cosine.min()),
                "max_abs_difference": float(np.abs(a.astype(np.float64) - b).max())}

    first = hg17.float_vectors(model, p32, rows)
    second = hg17.float_vectors(model, p32, rows)
    shuffled_order = np.random.default_rng(7).permutation(rows)
    other = np.zeros_like(first)
    other[shuffled_order] = hg17.float_vectors(model, p32, shuffled_order, batch=37)
    result["float"] = {"twice_in_one_process": compare(first, second), "against_the_line_step_process": compare(first, saved["f32"]),
                       "other_batch_composition_37_shuffled": compare(first, other)}
    first = hg17.onnx_vectors(sess, p8, rows)
    second = hg17.onnx_vectors(sess, p8, rows)
    other = np.zeros_like(first)
    other[shuffled_order] = hg17.onnx_vectors(sess, p8, shuffled_order)
    result["int8"] = {"twice_in_one_process": compare(first, second), "against_the_line_step_process": compare(first, saved["int8"]),
                      "other_rows_in_the_image_shuffled": compare(first, other)}
    # The byte graph in 8 bits, two runs of the same image and the same text in another row.
    corpus = hg17.Corpus("csic")
    table = np.fromfile(os.path.join(model_dir(args.fold, args.seed), "table.bin"), dtype="<f4").reshape(256, hg17.TABLE_WIDTH)
    bsess = hg17.session(os.path.join(model_dir(args.fold, args.seed), "byte.int8.onnx"), args.threads)
    rows16 = np.arange(16)
    h1, y1 = bsess.run(None, {"x": hg.byte_image(table, corpus.ids(rows16, hg17.L_MAX), 16)})
    h2, y2 = bsess.run(None, {"x": hg.byte_image(table, corpus.ids(rows16[::-1], hg17.L_MAX), 16)})
    result["byte_graph_int8"] = {"reversed_rows_identical_h": bool((h1[0] == h2[0][:, ::-1]).all()),
                                 "reversed_rows_identical_y": bool((y1[0] == y2[0][:, ::-1]).all())}
    # Same-seed training: the `repeat` encoder against the first.
    rdir, rmdir = enc_dir(args.fold, args.seed, "repeat"), model_dir(args.fold, args.seed, "repeat")
    if os.path.exists(os.path.join(rdir, args.variant, "line.json")):
        a_b, b_b = hg17.read_json(os.path.join(edir, "bytes.json")), hg17.read_json(os.path.join(rdir, "bytes.json"))
        a_l = hg17.read_json(os.path.join(out, "line.json"))
        b_l = hg17.read_json(os.path.join(rdir, args.variant, "line.json"))
        result["same_seed_training"] = {
            "bytes": {k: a_b["files"][k] == b_b["files"][k] for k in a_b["files"]},
            "line": {k: a_l["files"][k] == b_l["files"][k] for k in a_l["files"]}}
        sa, sb = torch.load(os.path.join(mdir, "line.pt")), torch.load(os.path.join(rmdir, args.variant, "line.pt"))
        result["same_seed_training"]["line_weights_max_abs_difference"] = max(
            float((sa["model"][k] - sb["model"][k]).abs().max()) for k in sa["model"])
        ba, bb = torch.load(os.path.join(model_dir(args.fold, args.seed), "byte.pt")), torch.load(os.path.join(rmdir, "byte.pt"))
        result["same_seed_training"]["byte_weights_max_abs_difference"] = max(float((ba[k] - bb[k]).abs().max()) for k in ba)
    hg17.write_json(os.path.join(out, "determinism.json"), result)
    print(json.dumps(result, indent=1))
    return result


# --------------------------------------------------------------------------------------
# NPU: the fixture of the Rust test, and the comparison of what it wrote
# --------------------------------------------------------------------------------------

def step_npu_fixture(args) -> dict:
    edir = enc_dir(args.fold, args.seed)
    mdir = model_dir(args.fold, args.seed)
    out = hg17.path("npu", f"f{args.fold}-s{args.seed}-{args.variant}", "x")[:-2]
    shutil.copyfile(os.path.join(mdir, "byte.int8.onnx"), os.path.join(out, "byte.int8.onnx"))
    shutil.copyfile(os.path.join(mdir, "table.bin"), os.path.join(out, "table.bin"))
    shutil.copyfile(os.path.join(mdir, args.variant, "line.int8.onnx"), os.path.join(out, "line.int8.onnx"))
    texts = []
    for name in ("csic", "probes"):
        c = hg17.Corpus(name)
        texts += [bytes(c.raw(i)).decode("utf-8") for i in range(len(c))]
    with open(os.path.join(out, "texts.jsonl"), "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps(text, ensure_ascii=False) + "\n")
    both = np.concatenate([np.fromfile(os.path.join(edir, args.variant, f"{n}-int8.f32"), dtype="<f4") for n in ("csic", "probes")])
    both.astype("<f4").tofile(os.path.join(out, "reference-raw.f32"))
    with open(os.path.join(out, "reference-starts.i32"), "wb") as handle:
        for n in ("csic", "probes"):
            handle.write(open(os.path.join(edir, f"{n}-starts.i32"), "rb").read())
    counts = np.concatenate([np.fromfile(os.path.join(edir, f"{n}-patches-int8.counts.i32"), dtype="<i4") for n in ("csic", "probes")])
    counts.astype("<u4").tofile(os.path.join(out, "reference-patches.u32"))
    record = hg17.read_json(os.path.join(edir, "bytes.json"))
    scale, zero = hg17.dequantize_scale(os.path.join(out, "byte.int8.onnx"), "y")
    hg17.exp_table(scale).astype("<f8").tofile(os.path.join(out, "exp-table.f64"))
    layout = {"variant": f"f{args.fold}-s{args.seed}-{args.variant}", "patcher": "entropy-jump",
              "patcher_setting": record["patching"]["threshold"], "input_bytes": hg17.L_MAX, "patch_limit": hg17.P_MAX,
              "max_patch_bytes": hg17.MAX_PATCH, "stride": hg.STRIDE, "delimiters": hg.DELIMITERS.decode("ascii"),
              "table_width": hg17.TABLE_WIDTH, "features": hg17.FEATURES, "dimension": hg17.OUTPUT_DIM,
              "byte_rows": hg17.ROWS_BYTE, "patch_rows": hg17.ROWS_PATCH, "texts": len(texts),
              "csic_texts": len(hg17.Corpus("csic")), "byte_outputs": ["h", "y"], "logit_scale": scale,
              "logit_zero_point": zero, "line_output": "positions"}
    hg17.write_json(os.path.join(out, "layout.json"), layout)
    print(f"NPU fixture in {out}")
    return layout


def step_npu_compare(args) -> dict:
    out = hg17.path("npu", f"f{args.fold}-s{args.seed}-{args.variant}", "x")[:-2]
    layout = hg17.read_json(os.path.join(out, "layout.json"))
    n, c = layout["texts"], layout["csic_texts"]
    npu = np.fromfile(os.path.join(out, "npu-raw.f32"), dtype="<f4").reshape(n, -1)
    ref = np.fromfile(os.path.join(out, "reference-raw.f32"), dtype="<f4").reshape(n, -1)
    cosine = np.sum(l2dsci.l2n(npu).astype(np.float64) * l2dsci.l2n(ref), axis=1)
    identical = int((npu == ref).all(axis=1).sum())
    ref_counts = np.fromfile(os.path.join(out, "reference-patches.u32"), dtype="<u4")
    npu_counts = np.fromfile(os.path.join(out, "npu-patches.u32"), dtype="<u4")
    ref_starts = np.fromfile(os.path.join(out, "reference-starts.i32"), dtype="<i4")
    npu_starts = np.fromfile(os.path.join(out, "npu-starts.u32"), dtype="<u4").astype(np.int64)
    same_cuts = 0
    if len(npu_starts) == len(ref_starts) and (npu_counts == ref_counts).all():
        offsets = np.concatenate([[0], np.cumsum(ref_counts.astype(np.int64))])
        same_cuts = int(sum(np.array_equal(ref_starts[offsets[i]:offsets[i + 1]], npu_starts[offsets[i]:offsets[i + 1]]) for i in range(n)))
    else:
        same_cuts = int((npu_counts == ref_counts).sum())     # an upper bound
    result = {"texts": n, "bit_identical_raw_vectors": identical, "cosine_min": float(cosine.min()),
              "cosine_p01": float(np.percentile(cosine, 1)), "cosine_median": float(np.median(cosine)),
              "lines_with_identical_cuts": same_cuts, "same_patch_count": int((npu_counts == ref_counts).sum())}
    hg17.write_json(os.path.join(out, "compare.json"), result)
    edir = os.path.join(enc_dir(args.fold, args.seed), args.variant)
    npu[:c].astype("<f4").tofile(os.path.join(edir, "csic-npu.f32"))
    npu[c:].astype("<f4").tofile(os.path.join(edir, "probes-npu.f32"))
    print(json.dumps(result, indent=1))
    return result


def step_export(args) -> dict:
    """The layout the qualification scripts read (`hg_run.write_export`), unit vectors."""
    edir = enc_dir(args.fold, args.seed)
    labels = hg17.Labels()
    probe_rows = [json.loads(line) for line in open(hg17.path("data", "probes.jsonl"), encoding="utf-8")]
    corpus, probes = hg17.Corpus("csic"), hg17.Corpus("probes")
    counts = {n: np.fromfile(os.path.join(edir, f"{n}-patches-int8.counts.i32"), dtype="<i4") for n in ("csic", "probes")}
    line_record = hg17.read_json(os.path.join(edir, args.variant, "line.json"))
    written = []
    for precision in ("f32", "int8", "npu"):
        src = os.path.join(edir, args.variant, f"csic-{precision}.f32")
        if not os.path.exists(src):
            continue
        out = hg17.path("vectors", f"f{args.fold}-s{args.seed}-{args.variant}-{precision}", "x")[:-2]
        for n in ("csic", "probes"):
            raw = np.fromfile(os.path.join(edir, args.variant, f"{n}-{precision}.f32"), dtype="<f4").reshape(-1, hg17.OUTPUT_DIM)
            l2dsci.l2n(raw).astype("<f4").tofile(os.path.join(out, f"{n}.f32"))
        with open(os.path.join(out, "csic.jsonl"), "w", encoding="utf-8") as handle:
            for i in range(len(corpus)):
                long = bool(corpus.full_bytes[i] > hg17.L_MAX)
                handle.write(json.dumps({"i": i, "text": bytes(corpus.raw(i)).decode("utf-8"),
                                         "label": "bad" if labels.bad[i] else "good", "group": labels.groups[i],
                                         "empty": bool(corpus.lens[i] == 0), "part_bytes": int(corpus.full_bytes[i]),
                                         "read_bytes": int(corpus.lens[i]), "truncated": long, "text_longer_than_input": long,
                                         "patches": int(counts["csic"][i])}, ensure_ascii=False) + "\n")
        with open(os.path.join(out, "probes.jsonl"), "w", encoding="utf-8") as handle:
            for i, row in enumerate(probe_rows):
                long = bool(probes.full_bytes[i] > hg17.L_MAX)
                handle.write(json.dumps(dict(row, empty=bool(probes.lens[i] == 0), part_bytes=int(probes.full_bytes[i]),
                                             read_bytes=int(probes.lens[i]), truncated=long, text_longer_than_input=long,
                                             patches=int(counts["probes"][i])), ensure_ascii=False) + "\n")
        # The qualification scripts read the held-out endpoints and the validation paths
        # from the record: here the fold's endpoints with both labels; 2.17 has no
        # validation part (the scripts still mark every fifth single-label path so).
        # The counts are the scripts' own partition (q3 checks them): held-out = the
        # fold's mixed endpoints, "validation" = every fifth single-label path in byte
        # order, the rest "training"; labels only, the encoder trained on the whole
        # fitting side of the fold.
        held_mixed = sorted(set(labels.group_array[labels.held(args.fold) & labels.mixed]))
        single = sorted({g for g in labels.groups if g not in set(labels.mixed_groups)}, key=lambda g: g.encode())
        in_held = np.isin(labels.group_array, held_mixed)
        in_val = np.isin(labels.group_array, single[0::5])
        record = dict(line_record, precision=precision,
                      split={"name": "cv5", "fold": args.fold, "held_out_mixed_paths": held_mixed, "held_out_logs": int(in_held.sum())},
                      training={"chosen_epoch": line_record["settings"]["epochs"]})
        record["settings"] = dict(line_record["settings"], validation_mixed_paths=[], validation_logs=int(in_val.sum()),
                                  training_logs=int((~in_held & ~in_val).sum()), routing="keep (the whole line)")
        hg17.write_json(os.path.join(out, "record.json"), {"models": [record]})
        written.append(out)
    return {"written": written}


# --------------------------------------------------------------------------------------
# summary: means and spread over folds and seeds
# --------------------------------------------------------------------------------------

def step_summary(args) -> dict:
    rows = []
    enc_root = hg17.path("enc", "x")[:-2]
    for name in sorted(os.listdir(enc_root)):
        if name.count("-") != 1:
            continue
        fold, seed = int(name.split("-")[0][1:]), int(name.split("-")[1][1:])
        for variant in VARIANTS:
            file_path = os.path.join(enc_root, name, variant, "eval.json")
            if os.path.exists(file_path):
                rows.append((variant, fold, seed, hg17.read_json(file_path)))
    refs = hg17.read_json(hg17.path("refs", "refs.json")) if os.path.exists(hg17.path("refs", "refs.json")) else None

    def stats(values):
        v = np.array(values, dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}

    def pick(block):
        c, s, p = block["classify"], block["spectrum"], block["pairs"]
        return {"unseen_linear": c["unseen"]["linear"], "unseen_boosted": c["unseen"]["boosted"], "unseen_knn11": c["unseen"]["knn11"],
                "seen_linear": c["seen"]["linear"], "seen_boosted": c["seen"]["boosted"],
                "components_95": s["components_95"], "spectrum_entropy_bits": s["spectrum_entropy_bits"],
                "sign_kept_weighted": s["sign_kept_weighted"], "cosine_of_differences": s["cosine_of_differences"],
                "attack_cosine_median": p["all"].get("attack_cosine_median"), "control_cosine_median": p["all"].get("control_cosine_median"),
                "pairs_auc_attack_further": p["all"].get("auc_attack_further"),
                "pairs_auc_attack_further_held_fold": p["held_fold"].get("auc_attack_further"),
                "distinct_vectors": block["folding"]["distinct_vectors"]}

    summary = {}
    for variant in VARIANTS:
        for precision in ("f32", "int8", "npu"):
            picked = [(f, s, pick(r[precision])) for v, f, s, r in rows if v == variant and precision in r]
            if not picked:
                continue
            keys = picked[0][2].keys()
            block = {k: stats([p[k] for _, _, p in picked if p[k] is not None]) for k in keys}
            block["per_fold_unseen_boosted_mean"] = {f"f{f}": float(np.mean([p["unseen_boosted"] for ff, _, p in picked if ff == f]))
                                                      for f in sorted({f for f, _, _ in picked})}
            block["per_fold_unseen_linear_mean"] = {f"f{f}": float(np.mean([p["unseen_linear"] for ff, _, p in picked if ff == f]))
                                                     for f in sorted({f for f, _, _ in picked})}
            summary[f"{variant}/{precision}"] = block
    if refs:
        picked = [pick(v) for v in refs["published"].values()]
        summary["published (NPU vectors of task 2.15)"] = {k: stats([p[k] for p in picked if p[k] is not None]) for k in picked[0]}
        summary["published (NPU vectors of task 2.15)"]["per_fold_unseen_linear_mean"] = {
            f"f{f}": float(np.mean([refs["published"][f"f{f}-s{s}"]["classify"]["unseen"]["linear"] for s in range(3)])) for f in range(5)}
        summary["baseline scorer"] = {"unseen": stats([v["unseen"] for v in refs["baseline"].values()]),
                                      "seen": stats([v["seen"] for v in refs["baseline"].values()]),
                                      "per_fold_unseen": {f"f{f}": refs["baseline"][f"f{f}-s0"]["unseen"] for f in range(5)}}
    hg17.write_json(hg17.path("summary", "summary.json"), summary)
    for key, block in summary.items():
        if "unseen_linear" in block:
            print(f"{key}: unseen linear {block['unseen_linear']['mean']:.3f} ± {block['unseen_linear']['std']:.3f}, "
                  f"boosted {block['unseen_boosted']['mean']:.3f} ± {block['unseen_boosted']['std']:.3f}; "
                  f"seen linear {block['seen_linear']['mean']:.3f}, boosted {block['seen_boosted']['mean']:.3f} (n {block['unseen_linear']['n']})")
        else:
            print(f"{key}: unseen {block['unseen']['mean']:.3f} ± {block['unseen']['std']:.3f}, seen {block['seen']['mean']:.3f}")
    return {"written": hg17.path("summary", "summary.json")}


def step_per_endpoint(args) -> dict:
    """ROC AUC per held-out endpoint (paths with both labels), for the scorers fitted on
    the fitting side of each fold: the encoders' 8-bit vectors (linear and boosted, each
    seed), the published encoder's vectors and the baseline scorer. The published
    encoder was trained with the labels of split17's fitting side, so only split17's
    three held-out endpoints (anadir, entrar, vaciar) are unseen by it as well."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    labels = hg17.Labels()
    published_dir = r"D:\BuildArtifacts\log2decision\task-2.15\vectors"
    published = l2dsci.l2n(np.fromfile(os.path.join(published_dir, "csic.f32"), dtype="<f4").reshape(len(labels.bad), -1)).astype(np.float64)
    corpus = hg17.Corpus("csic")
    features = l2dsci.hashed_rows([bytes(corpus.raw(i)).decode("utf-8") for i in range(len(corpus))], "keep")
    out = {}

    def per_group(score, held):
        result = {}
        for g in sorted(set(labels.group_array[held & labels.mixed])):
            use = held & (labels.group_array == g)
            result[g] = float(roc_auc_score(labels.bad[use], score[use]))
        return result

    for fold in range(5):
        fit, held = labels.fit(fold), labels.held(fold)
        groups_fit = [labels.groups[i] for i in np.flatnonzero(fit)]
        block = {}
        score = np.zeros(len(labels.bad))
        score[held] = l2dsci.fit_logreg(features[fit], labels.bad[fit], l2=1.0).predict_proba(features[held])[:, 1]
        block["baseline"] = per_group(score, held)
        sources = [("published", 0, published)]
        for variant in ("mask", "contrastive"):
            for seed in (0, 1, 2):
                raw = np.fromfile(os.path.join(enc_dir(fold, seed), variant, "csic-int8.f32"), dtype="<f4").reshape(-1, hg17.OUTPUT_DIM)
                sources.append((f"{variant}-s{seed}", seed, l2dsci.l2n(raw).astype(np.float64)))
        for name, seed, x in sources:
            probability, _ = l2dsci.linear_head(x[fit], labels.bad[fit], groups_fit, x[held])
            score[held] = probability
            block[f"{name}/linear"] = per_group(score, held)
            score[held] = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(x[fit], labels.bad[fit]).predict_proba(x[held])[:, 1]
            block[f"{name}/boosted"] = per_group(score, held)
        out[f"f{fold}"] = block
        print(f"per-endpoint f{fold}: " + json.dumps({k: {g.rsplit('/', 1)[1]: round(v, 3) for g, v in b.items()} for k, b in block.items()}))
    hg17.write_json(hg17.path("refs", "per-endpoint.json"), out)
    return {"written": hg17.path("refs", "per-endpoint.json")}


STEPS = {"per-endpoint": step_per_endpoint, "prepare": step_prepare, "bytes": step_bytes, "line": step_line, "eval": step_eval, "refs": step_refs,
         "determinism": step_determinism, "npu-fixture": step_npu_fixture, "npu-compare": step_npu_compare,
         "export": step_export, "summary": step_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=list(STEPS))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="mask", choices=list(VARIANTS))
    parser.add_argument("--tag", default="")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--memory-gib", type=float, default=6.0)
    parser.add_argument("--lm-epochs", type=int, default=None, help="smoke tests only")
    parser.add_argument("--line-epochs", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()
    if args.lm_epochs:
        LM["epochs"] = args.lm_epochs
    if args.line_epochs:
        LINE["epochs"] = args.line_epochs
    if os.path.splitdrive(os.getcwd())[1] in ("\\", "/", ""):
        raise SystemExit("refusing to run from a drive root")
    memguard.limit(args.memory_gib)
    name = f"{args.step} f{args.fold} s{args.seed} {args.variant} {args.tag}".strip()
    started = time.perf_counter()
    with memguard.hungry(f"task 2.17 {name}"):
        result = STEPS[args.step](args)
    if args.step == "prepare":
        print(json.dumps(result, indent=1))
    run ={"step": name, "argv": sys.argv[1:], "seconds": round(time.perf_counter() - started, 1),
           "memory": memguard.peaks(), "host_memory_at_end": memguard.host_memory(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(hg17.path("runs.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(run) + "\n")
    print(f"RUN {name}: {run['seconds']:.0f} s, peak committed {run['memory']['peak_committed_gib']} GiB, "
          f"peak working set {run['memory']['peak_working_set_gib']} GiB, limit {args.memory_gib} GiB")


if __name__ == "__main__":
    main()
