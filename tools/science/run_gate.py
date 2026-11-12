"""One run of the science gate: one split, one fold, one seed, one routing mode.

Trains the encoder on the fitting side, exports it, quantizes it to 8 bits, and scores the
held-out endpoints from the vectors with the nearest-neighbour ratio and with a linear
head, for the float and the 8-bit model. With `--baseline` it fits the character n-gram
baseline on the same split instead. Writes one JSON record per run into `--out`.

Usage (see README.md):
  uv run python run_gate.py --data <csv> --out <dir> --split cv5 --fold 0 --seed 0 --routing strip
  uv run python run_gate.py --data <csv> --out <dir> --split cv5 --fold 0 --baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from importlib.metadata import version

import numpy as np

import l2dsci as s
from inspect_onnx import describe

SPIKE_INT8 = r"D:\BuildArtifacts\log2decision\npu-cnn-spike\enc_b32_int8.onnx"


def sha256_of(path: str) -> str:
    return s.file_sha256(path)


def compare_with_spike(int8_path: str, spike_path: str) -> dict:
    """Same interface and same operators, node for node, as the spike model that passed
    the hardware gate. Weights and quantization ranges differ by construction."""
    if not os.path.exists(spike_path):
        return {"compared": False, "reason": f"{spike_path} not found"}
    ours, spike = describe(int8_path), describe(spike_path)
    checks = {key: ours[key] == spike[key] for key in
              ("opsets", "inputs", "outputs", "op_counts", "op_sequence", "node_signatures",
               "zero_point_dtypes", "per_channel_scales", "initializer_count")}
    strip = lambda w: {k: w[k] for k in ("dtype", "shape", "zero_point")}
    checks["quantized_initializers"] = ([strip(w) for w in ours["quantized_initializers"]]
                                        == [strip(w) for w in spike["quantized_initializers"]])
    return {"compared": True, "spike": spike_path, "spike_sha256": sha256_of(spike_path),
            "identical": all(checks.values()), "checks": checks,
            "ours": {k: ours[k] for k in ("opsets", "inputs", "outputs", "op_counts",
                                          "zero_point_dtypes", "per_channel_scales")}}


def score_vectors(vectors: np.ndarray, data: s.Dataset, fit: np.ndarray, held: np.ndarray) -> dict:
    """Both uses of the vectors. The store and the head hold fitting-side vectors only."""
    unit = s.l2n(vectors)
    store, store_bad = unit[fit], data.bad[fit]
    store_groups = [g for g, f in zip(data.groups, fit) if f]
    query, bad = unit[held], data.bad[held]
    groups = [g for g, h in zip(data.groups, held) if h]
    mixed = set(data.mixed_groups)
    out = {}
    for k in (5, 11, 51):
        ratio, _ = s.knn_ratio(store, store_bad, query, k)
        out[f"knn_k{k}"] = s.metrics(ratio, bad, groups, mixed)
    ratio, abstained = s.knn_ratio(store, store_bad, query, 11, reach=0.5, min_support=3)
    out["knn_k11_reach0.5"] = s.metrics(np.nan_to_num(ratio), bad, groups, mixed, abstained)
    probability, head = s.linear_head(store, store_bad, store_groups, query)
    out["linear_head"] = s.metrics(probability, bad, groups, mixed)
    out["linear_head"]["head"] = head
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["cv5", "split17"], required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--routing", choices=["strip", "keep"], default="strip")
    p.add_argument("--objective", choices=["bce", "bce-norm", "bce-norm-unit", "supcon"], default="bce")
    p.add_argument("--no-validation", action="store_true",
                   help="train on the whole fitting side for all epochs and keep the last epoch")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--calibration", type=int, default=512)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    started = time.perf_counter()
    os.makedirs(args.out, exist_ok=True)

    data = s.read_csic(args.data)
    groups = np.array(data.groups)
    held_groups = s.held_out_groups(data, args.split, args.fold)
    held = np.isin(groups, sorted(held_groups))
    fit = ~held
    mixed = set(data.mixed_groups)
    record = {
        "task": "npu-log-decision 2.1",
        "data": {"path": args.data, "sha256": data.sha256, "records": data.records,
                 "distinct_texts": len(data.texts)},
        "split": {"name": args.split, "fold": args.fold,
                  "held_out_mixed_paths": sorted(held_groups & mixed),
                  "held_out_paths": len(held_groups), "held_out_logs": int(held.sum()),
                  "fitting_logs": int(fit.sum()),
                  "held_out_paths_sha256": hashlib.sha256("\n".join(sorted(held_groups)).encode()).hexdigest()},
        "packages": {name: version(name) for name in ("numpy", "torch", "onnx", "onnxruntime", "scikit-learn", "scipy")},
        "python": platform.python_version(), "host": platform.processor(),
    }
    held_groups_list = [g for g, h in zip(data.groups, held) if h]

    if args.baseline:
        name = f"{args.split}-f{args.fold}-baseline"
        record["baseline"] = {}
        for routing in ("strip", "keep"):
            t0 = time.perf_counter()
            rows = s.hashed_rows(data.texts, routing)
            model = s.fit_logreg(rows[fit], data.bad[fit], l2=1.0)
            ratio = model.predict_proba(rows[held])[:, 1]
            result = s.metrics(ratio, data.bad[held], held_groups_list, mixed)
            result["settings"] = {"buckets": 65536, "ngram_max": 3, "l2": 1.0, "routing": routing,
                                  "iterations": int(model.n_iter_[0]),
                                  "fitting_rows_without_features": int((np.diff(rows[fit].indptr) == 0).sum())}
            result["seconds"] = time.perf_counter() - t0
            record["baseline"][routing] = result
            print(f"{name} {routing}: cross-endpoint ROC AUC {result['cross_endpoint'].get('roc_auc'):.4f} "
                  f"AP {result['cross_endpoint'].get('average_precision'):.4f}")
    else:
        variant = args.objective + ("-noval" if args.no_validation else "") + (f"-{args.tag}" if args.tag else "")
        name = f"{args.split}-f{args.fold}-s{args.seed}-{args.routing}-{variant}"
        print(name)
        ids, truncated = s.byte_ids(data.texts, args.routing)
        if args.no_validation:
            val = np.zeros(len(groups), dtype=bool)
            val_groups: set[str] = set()
        else:
            val_groups = s.validation_groups(data, held_groups, args.fold)
            val = np.isin(groups, sorted(val_groups))
        assert not (val & held).any()
        train = fit & ~val
        # Identical encoder inputs on both sides of the split (possible in strip mode,
        # where two paths can carry the same payload).
        fit_inputs = {row.tobytes() for row in ids[fit]}
        held_mixed = held & np.isin(groups, sorted(mixed))
        repeated = sum(row.tobytes() in fit_inputs for row in ids[held_mixed])
        record["settings"] = {
            "routing": args.routing, "objective": args.objective, "seed": args.seed,
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "weight_decay": args.weight_decay, "optimizer": "AdamW, one-cycle schedule (10 % warm-up)",
            "threads": args.threads, "device": "cpu", "validation": not args.no_validation,
            "validation_mixed_paths": sorted(val_groups & mixed), "validation_logs": int(val.sum()),
            "training_logs": int(train.sum()), "calibration_texts": args.calibration,
            "input_bytes": s.INPUT_BYTES, "pad_byte": s.PAD_BYTE,
            "truncated_share_held_out": float(truncated[held].mean()),
            "held_out_mixed_inputs_identical_to_a_fitting_input": int(repeated),
        }
        t0 = time.perf_counter()
        if args.no_validation:
            # No selection: a dummy validation part of the training texts, last epoch kept.
            model, training = s.train_encoder(
                ids[train], data.bad[train], ids[train][:512], data.bad[train][:512],
                objective=args.objective, seed=args.seed, epochs=args.epochs, batch=args.batch,
                lr=args.lr, weight_decay=args.weight_decay, threads=args.threads, keep_last=True)
        else:
            model, training = s.train_encoder(
                ids[train], data.bad[train], ids[val], data.bad[val],
                objective=args.objective, seed=args.seed, epochs=args.epochs, batch=args.batch,
                lr=args.lr, weight_decay=args.weight_decay, threads=args.threads)
        training["seconds"] = time.perf_counter() - t0
        record["training"] = training
        record["parameters"] = {
            "graph": int(sum(p.numel() for p in model.body.parameters())),
            "byte_table": int(model.table.weight.numel()),
        }

        table = model.table.weight.detach().numpy().astype(np.float32)
        assert not table[s.PAD_BYTE].any()
        float_path = os.path.join(args.out, f"{name}.f32.onnx")
        int8_path = os.path.join(args.out, f"{name}.int8.onnx")
        table_path = os.path.join(args.out, f"{name}.table.bin")
        table.astype("<f4").tofile(table_path)
        s.export_float(model, float_path)
        # Calibration sample: a seeded draw from the training part of the fitting side.
        rng = np.random.default_rng(args.seed)
        calibration = rng.choice(np.flatnonzero(train), size=args.calibration, replace=False)
        s.quantize_qdq(float_path, int8_path, table, ids[np.sort(calibration)])

        torch_vectors = s.float_vectors(model, ids)
        t0 = time.perf_counter()
        f32_vectors = s.onnx_vectors(float_path, table, ids, args.threads)
        int8_vectors = s.onnx_vectors(int8_path, table, ids, args.threads)
        encode_seconds = time.perf_counter() - t0
        cosine = np.sum(s.l2n(f32_vectors) * s.l2n(int8_vectors), axis=1)
        record["export"] = {
            "float_onnx": float_path, "int8_onnx": int8_path, "table": table_path,
            "sha256": {"float_onnx": sha256_of(float_path), "int8_onnx": sha256_of(int8_path),
                       "table": sha256_of(table_path)},
            "torch_vs_float_onnx_max_abs_difference": float(np.abs(torch_vectors - f32_vectors).max()),
            "float_vs_int8_cosine": {
                "all": {"mean": float(cosine.mean()), "min": float(cosine.min()),
                        "p01": float(np.quantile(cosine, 0.01)), "p50": float(np.median(cosine))},
                "held_out": {"mean": float(cosine[held].mean()), "min": float(cosine[held].min()),
                             "p01": float(np.quantile(cosine[held], 0.01))},
                "share_below_0.99": float((cosine < 0.99).mean()),
            },
            "onnx_cpu_encode_seconds_both_models": encode_seconds,
            "spike_comparison": compare_with_spike(int8_path, SPIKE_INT8),
        }
        record["float"] = score_vectors(f32_vectors, data, fit, held)
        record["int8"] = score_vectors(int8_vectors, data, fit, held)
        # The classifier layer used in training, for reference only: the tool does not have it.
        if args.objective != "supcon":
            import torch
            with torch.no_grad():
                logit = model.logits(torch.from_numpy(torch_vectors[held])).numpy()
            record["training_classifier_reference"] = s.metrics(logit, data.bad[held], held_groups_list, mixed)
        for kind in ("float", "int8"):
            for use in ("knn_k11", "linear_head"):
                m = record[kind][use]["cross_endpoint"]
                print(f"  {kind} {use}: cross-endpoint ROC AUC {m['roc_auc']:.4f}  AP {m['average_precision']:.4f}")
        print(f"  float vs int8 cosine: mean {cosine.mean():.4f}  min {cosine.min():.4f}; "
              f"same operators and interface as the spike: {record['export']['spike_comparison'].get('identical')}")

    record["name"] = name
    record["seconds"] = time.perf_counter() - started
    with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    print(f"{name}: {record['seconds']:.0f} s")


if __name__ == "__main__":
    main()
