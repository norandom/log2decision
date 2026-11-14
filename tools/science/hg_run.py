"""Steps of the hierarchical encoder gate (task 2.16). Development only.

    uv run python hg_run.py bytelm   --seed 0
    uv run python hg_run.py features --seed 0
    uv run python hg_run.py variant  --name entropy --patcher entropy-jump --seed 0
    uv run python hg_run.py reference
    uv run python hg_run.py npu-fixture --variant entropy-s0
    uv run python hg_run.py npu-stages  --variant entropy-s0   (after the Rust test ran)
    uv run python hg_run.py npu-compare --variant entropy-s0

Everything is written below `hg.ROOT` (`D:\\BuildArtifacts\\log2decision\\task-2.16`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from importlib.metadata import version

import numpy as np

import hg
import l2dsci


def sha256_of(file_path: str) -> str:
    with open(file_path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def environment() -> dict:
    return {"packages": {name: version(name) for name in
                         ("numpy", "torch", "onnx", "onnxruntime", "scikit-learn", "scipy")},
            "python": platform.python_version(), "host": platform.processor()}


def write_json(file_path: str, value) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False)


def read_json(file_path: str):
    with open(file_path, encoding="utf-8") as handle:
        return json.load(handle)


def lm_dir(seed: int) -> str:
    return hg.path("bytelm", f"s{seed}", "x")[:-2]


def load_byte_model(seed: int):
    import torch

    ByteLM, _ = hg.build_byte_model()
    model = ByteLM()
    model.load_state_dict(torch.load(os.path.join(lm_dir(seed), "model.pt")))
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# Step 1: the byte model
# --------------------------------------------------------------------------------------

def step_bytelm(args) -> None:
    import torch

    started = time.perf_counter()
    split = hg.Split()
    settings = {"seed": args.seed, "epochs": args.epochs, "batch": 64, "lr": 2e-3, "weight_decay": 0.01,
                "optimizer": "AdamW, one-cycle schedule (10 % warm-up)", "threads": args.threads,
                "device": "cpu", "layers": hg.LM_LAYERS, "kernel": [1, hg.LM_KERNEL],
                "causal_taps": hg.LM_KERNEL // 2 + 1, "channels": hg.FEATURES,
                "feature_layer": hg.FEATURE_LAYER, "table_width": hg.TABLE_WIDTH,
                "input_bytes": hg.L_MAX, "objective": "next-byte cross-entropy plus 0.1 x squared log-normaliser, no label",
                "logit_floor": hg.LOGIT_FLOOR, "normaliser_weight": hg.NORMALISER_WEIGHT,
                "trained_on": "training part of the fitting side"}
    model, history = hg.train_byte_model(split, seed=args.seed, epochs=args.epochs, batch=64, lr=2e-3,
                                         weight_decay=0.01, threads=args.threads)
    out = lm_dir(args.seed)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    table = model.table.weight.detach().numpy().astype("<f4")
    assert not table[0].any()
    table.tofile(os.path.join(out, "table.bin"))

    _, ByteGraph = hg.build_byte_model()
    rng = np.random.default_rng(args.seed)
    calibration = np.sort(rng.choice(np.flatnonzero(split.train), size=256, replace=False))
    files = {}
    for name, with_logits, outputs in (("byte-ent", True, ["h", "y"]), ("byte-feat", False, ["h"])):
        float_path = os.path.join(out, f"{name}.f32.onnx")
        int8_path = os.path.join(out, f"{name}.int8.onnx")
        hg.export_onnx(ByteGraph(model, with_logits).eval(),
                       (1, hg.TABLE_WIDTH, hg.ROWS_BYTE, hg.L_MAX), outputs, float_path)
        images = [hg.byte_image(table, split.lines.ids[calibration[i:i + hg.ROWS_BYTE]], hg.ROWS_BYTE)
                  for i in range(0, len(calibration), hg.ROWS_BYTE)]
        hg.quantize(float_path, int8_path, images)
        files[name] = {"f32_sha256": sha256_of(float_path), "int8_sha256": sha256_of(int8_path),
                       "int8_bytes": os.path.getsize(int8_path)}
    record = {"task": "npu-log-decision 2.16", "step": "byte model", **split.record(), **environment(),
              "settings": settings, "history": history,
              "parameters": {"graph": int(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("table"))),
                             "byte_table": int(model.table.weight.numel())},
              "calibration_lines": 256, "files": files, "table_sha256": sha256_of(os.path.join(out, "table.bin")),
              "seconds": time.perf_counter() - started}
    write_json(os.path.join(out, "record.json"), record)
    print(f"byte model seed {args.seed}: {record['parameters']}, {record['seconds']:.0f} s")


# --------------------------------------------------------------------------------------
# Step 2: byte features and entropies (float and 8-bit), the patchers, patch statistics
# --------------------------------------------------------------------------------------

def feature_dir(seed: int) -> str:
    return hg.path("features", f"s{seed}", "x")[:-2]


def patch_stats(lines: hg.Lines, starts: list[np.ndarray]) -> dict:
    counts = np.array([len(s) for s in starts])
    lengths = np.concatenate([np.diff(np.append(s, n)) for s, n in zip(starts, lines.lens) if n > 0])
    uncapped_last = np.array([n - s[-1] for s, n in zip(starts, lines.lens) if n > 0])
    return {"lines": len(lines), "patches_mean": float(counts.mean()),
            "counts_histogram": np.bincount(counts, minlength=hg.P_MAX + 1).tolist(),
            "patches_p50_p95_max": [float(np.percentile(counts, 50)), float(np.percentile(counts, 95)), int(counts.max())],
            "patch_bytes_mean": float(lengths.mean()),
            "patch_bytes_p05_p50_p95_max": [float(v) for v in np.percentile(lengths, [5, 50, 95])] + [int(lengths.max())],
            "lines_at_patch_limit": int((counts == hg.P_MAX).sum()),
            "lines_whose_last_patch_exceeds_max_patch": int((uncapped_last > hg.MAX_PATCH).sum()),
            "lines_longer_than_input": int((lines.full_bytes > hg.L_MAX).sum()),
            "line_bytes_p50_p95_p99_max": [float(v) for v in np.percentile(lines.full_bytes, [50, 95, 99])] + [int(lines.full_bytes.max())]}


def step_features(args) -> None:
    import torch

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    split = hg.Split()
    model = load_byte_model(args.seed)
    out = feature_dir(args.seed)
    table = np.fromfile(os.path.join(lm_dir(args.seed), "table.bin"), dtype="<f4").reshape(256, hg.TABLE_WIDTH)
    probe_rows = hg.read_probes()
    ait_texts, ait_entries = hg.ait_sample()
    sets = {"csic": split.lines, "probes": hg.Lines([r["text"] for r in probe_rows]), "ait": hg.Lines(ait_texts)}

    ent = {}
    # Computed once; a second call reads the files (delete them to compute again).
    sess = None
    for name, precision in [(n, "f32") for n in sets] + [("csic", "int8"), ("probes", "int8")]:
        feat_path = os.path.join(out, f"{name}-feat-{precision}.npy")
        ent_path = os.path.join(out, f"{name}-ent-{precision}.npy")
        if os.path.exists(feat_path) and os.path.exists(ent_path):
            ent[name, precision] = np.load(ent_path)
            assert len(ent[name, precision]) == int(sets[name].offsets[-1])
            continue
        if precision == "f32":
            feats, ent[name, precision] = hg.float_byte_outputs(model, sets[name])
        else:
            sess = sess or hg.session(os.path.join(lm_dir(args.seed), "byte-ent.int8.onnx"), args.threads)
            feats, ent[name, precision] = hg.onnx_byte_outputs(sess, table, sets[name], True)
        np.save(feat_path, feats)
        np.save(ent_path, ent[name, precision])
        print(f"byte features of {name}, {precision}: {feats.shape}")
        del feats

    # The budget: the mean patch count of the fixed-stride patcher on the training lines.
    train_rows = np.flatnonzero(split.train)
    csic = sets["csic"]
    target = float(np.mean([len(hg.starts_fixed(int(n))) for n in csic.lens[train_rows]]))
    settings = {"fixed": hg.STRIDE}
    for rule in ("jump", "global"):
        settings[f"entropy-{rule}"] = hg.calibrate_threshold(csic, train_rows, ent["csic", "f32"], rule, target)
    means = {m: float(np.mean([len(hg.starts_delimiter(csic.raw(r), m)) for r in train_rows])) for m in range(1, 17)}
    settings["delimiter"] = min(means, key=lambda m: abs(means[m] - target))
    print(f"budget: {target:.2f} patches per training line; settings {settings}; delimiter means {means}")

    report = {"task": "npu-log-decision 2.16", "step": "patching", "seed": args.seed,
              "budget_patches_per_training_line": target, "settings": settings,
              "delimiters": hg.DELIMITERS.decode("ascii"), "delimiter_mean_by_min_len": means,
              "max_patch_bytes": hg.MAX_PATCH, "patch_limit": hg.P_MAX, "input_bytes": hg.L_MAX,
              "ait_entries": hg.AIT_ENTRIES, "ait_lines_per_entry": hg.AIT_LINES_PER_ENTRY, "stats": {}}
    patchers = ("fixed", "delimiter", "entropy-jump", "entropy-global")
    starts = {}
    for name, lines in sets.items():
        for patcher in patchers:
            starts[name, patcher] = hg.all_starts(lines, patcher, settings[patcher], ent[name, "f32"])
            block = patch_stats(lines, starts[name, patcher])
            if name == "csic":
                for part, mask in (("training", split.train), ("held-out", split.held)):
                    block[f"patches_mean_{part}"] = float(np.mean([len(starts[name, patcher][r]) for r in np.flatnonzero(mask)]))
            if name == "ait":
                block["by_entry"] = {}
                for entry in hg.AIT_ENTRIES:
                    rows = [i for i, e in enumerate(ait_entries) if e == entry]
                    counts = np.array([len(starts[name, patcher][i]) for i in rows])
                    sizes = lines.lens[rows]
                    block["by_entry"][entry] = {
                        "lines": len(rows), "bytes_mean": float(sizes.mean()),
                        "patches_mean": float(counts.mean()),
                        "bytes_per_patch": float(sizes.sum() / max(counts.sum(), 1)),
                        "lines_longer_than_input": int((lines.full_bytes[rows] > hg.L_MAX).sum()),
                        "lines_at_patch_limit": int((counts == hg.P_MAX).sum())}
            report["stats"][f"{name}/{patcher}"] = block
            print(f"{name} {patcher}: {block['patches_mean']:.2f} patches per line, {block['patch_bytes_mean']:.2f} bytes per patch, "
                  f"at the patch limit {block['lines_at_patch_limit']}, longer than the input {block['lines_longer_than_input']}")
    # Entropy of the float model per source: how surprised it is.
    report["entropy_nats_mean"] = {name: float(ent[name, "f32"].mean()) for name in sets}
    report["entropy_nats_mean_csic_training"] = float(np.concatenate(
        [ent["csic", "f32"][int(csic.offsets[r]):int(csic.offsets[r + 1])] for r in train_rows]).mean())
    report["entropy_nats_mean_csic_held_out"] = float(np.concatenate(
        [ent["csic", "f32"][int(csic.offsets[r]):int(csic.offsets[r + 1])] for r in np.flatnonzero(split.held)]).mean())
    report["entropy_nats_mean_ait_by_entry"] = {
        entry: float(np.concatenate([ent["ait", "f32"][int(sets["ait"].offsets[i]):int(sets["ait"].offsets[i + 1])]
                                     for i, e in enumerate(ait_entries) if e == entry]).mean())
        for entry in hg.AIT_ENTRIES}
    # 8-bit against float: the same boundaries?
    for rule in ("jump", "global"):
        patcher = f"entropy-{rule}"
        int8 = hg.all_starts(csic, patcher, settings[patcher], ent["csic", "int8"])
        same = sum(np.array_equal(a, b) for a, b in zip(starts["csic", patcher], int8))
        report["stats"][f"csic/{patcher}"]["lines_with_the_same_boundaries_from_the_8bit_graph"] = int(same)
        report["stats"][f"csic/{patcher}"]["patches_mean_from_the_8bit_graph"] = float(np.mean([len(s) for s in int8]))
        print(f"{patcher}: {same} of {len(csic)} CSIC lines have the same boundaries from the 8-bit graph")
    diff = np.abs(ent["csic", "int8"] - ent["csic", "f32"])
    report["entropy_8bit_minus_float_abs_mean_p99"] = [float(diff.mean()), float(np.percentile(diff, 99))]

    # Example lines for the pictures.
    examples = []
    wanted = [("csic", int(np.flatnonzero(split.train & ~split.data.bad & (csic.lens > 90) & (csic.lens < 130))[0])),
              ("csic", int(np.flatnonzero(split.held & split.data.bad & (csic.lens > 110) & (csic.lens < 170))[0]))]
    seen_entries = set()
    for i, entry in enumerate(ait_entries):
        if entry not in seen_entries and 60 < sets["ait"].lens[i] < 190:
            seen_entries.add(entry)
            if entry.endswith(("horde-access.log", "audit.log", "auth.log", "dnsmasq.log")):
                wanted.append(("ait", i))
    for name, row in wanted:
        lines = sets[name]
        at, n = int(lines.offsets[row]), int(lines.lens[row])
        examples.append({"source": name, "entry": ait_entries[row] if name == "ait" else "CSIC 2010",
                         "bytes": lines.raw(row).tolist(), "entropy": ent[name, "f32"][at:at + n].tolist(),
                         "starts": {p: starts[name, p][row].tolist() for p in patchers}})
    report["examples"] = examples
    report["seconds"] = time.perf_counter() - started
    write_json(os.path.join(out, "patching.json"), report)
    print(f"features and patching: {report['seconds']:.0f} s")


# --------------------------------------------------------------------------------------
# Step 3: one variant
# --------------------------------------------------------------------------------------

def ragged_batches(lines: hg.Lines, feats: np.ndarray):
    def make(rows):
        x = np.zeros((len(rows), hg.FEATURES, 1, hg.L_MAX), dtype=np.float32)
        for k, row in enumerate(rows):
            n, at = int(lines.lens[row]), int(lines.offsets[row])
            x[k, :, 0, :n] = feats[at:at + n].T
        return x
    return make


def pooled_batches(x: np.ndarray):
    return lambda rows: np.ascontiguousarray(x[rows].transpose(0, 2, 1)[:, :, None, :])


def write_export(out: str, split: hg.Split, probe_rows: list[dict], probe_lines: hg.Lines,
                 csic_vectors: np.ndarray, probe_vectors: np.ndarray, csic_patches, probe_patches,
                 record: dict) -> None:
    """The directory layout the qualification scripts read (header of
    `crates/l2d-encoder/tests/npu_embedding_export.rs`)."""
    os.makedirs(out, exist_ok=True)
    l2dsci.l2n(csic_vectors).astype("<f4").tofile(os.path.join(out, "csic.f32"))
    l2dsci.l2n(probe_vectors).astype("<f4").tofile(os.path.join(out, "probes.f32"))
    lines = split.lines
    with open(os.path.join(out, "csic.jsonl"), "w", encoding="utf-8") as handle:
        for i, text in enumerate(lines.texts):
            long = bool(lines.full_bytes[i] > hg.L_MAX)
            handle.write(json.dumps({
                "i": i, "text": text, "label": "bad" if split.data.bad[i] else "good", "group": split.data.groups[i],
                "empty": bool(lines.lens[i] == 0), "part_bytes": int(lines.full_bytes[i]),
                "read_bytes": int(lines.lens[i]), "truncated": long, "text_longer_than_input": long,
                "patches": int(csic_patches[i])}, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "probes.jsonl"), "w", encoding="utf-8") as handle:
        for i, row in enumerate(probe_rows):
            long = bool(probe_lines.full_bytes[i] > hg.L_MAX)
            row = dict(row, empty=bool(probe_lines.lens[i] == 0), part_bytes=int(probe_lines.full_bytes[i]),
                       read_bytes=int(probe_lines.lens[i]), truncated=long, text_longer_than_input=long,
                       patches=int(probe_patches[i]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(os.path.join(out, "record.json"), record)


def step_variant(args) -> None:
    import torch

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    name = f"{args.name}-s{args.seed}"
    out = hg.path("variants", name, "x")[:-2]
    log_lines = []

    def log(text):
        print(text, flush=True)
        log_lines.append(text)

    split = hg.Split()
    probe_rows = hg.read_probes()
    sets = {"csic": split.lines, "probes": hg.Lines([r["text"] for r in probe_rows])}
    fdir = feature_dir(args.lm_seed)
    patching = read_json(os.path.join(fdir, "patching.json"))
    flat = args.patcher == "flat"
    setting = None if flat else patching["settings"][args.patcher]
    log(f"variant {name}: patcher {args.patcher} (setting {setting}), objective {args.objective}, label weight {args.label_weight}")

    def load(which: str, precision: str):
        """(batch maker, positions per line) of one set at one precision. The byte
        features stay on disk (memory-mapped); one set's patch features are held at a
        time, so that several variants can train side by side."""
        lines = sets[which]
        feats = np.load(os.path.join(fdir, f"{which}-feat-{precision}.npy"), mmap_mode="r")
        if flat:
            return ragged_batches(lines, feats), np.array(lines.lens)
        ent = np.load(os.path.join(fdir, f"{which}-ent-{precision}.npy"))
        starts = hg.all_starts(lines, args.patcher, setting, ent)
        if which == "csic" and precision == "f32":
            for row, s in enumerate(starts):
                patch_bytes[row, :len(s)] = np.diff(np.append(s, lines.lens[row]))
        return pooled_batches(hg.pooled(lines, feats, starts)), np.array([len(s) for s in starts])

    patch_bytes = np.zeros((len(sets["csic"]), hg.P_MAX), dtype=np.int32)   # bytes per patch, float cut
    make, counts = load("csic", "f32")
    settings = {"patcher": args.patcher, "patcher_setting": setting, "objective": args.objective,
                "label_weight": args.label_weight, "seed": args.seed, "byte_model_seed": args.lm_seed,
                "epochs": args.epochs, "batch": 256, "lr": 1e-3, "weight_decay": 0.01,
                "optimizer": "AdamW, one-cycle schedule (10 % warm-up)", "temperature": args.temperature,
                "mask_rate": args.mask_rate, "boundary_drop": args.boundary_drop, "threads": args.threads, "device": "cpu",
                "routing": "keep (the whole line)", "input_bytes": hg.L_MAX, "patch_limit": hg.P_MAX,
                "strides": list(hg.FLAT_STRIDES if flat else hg.HIER_STRIDES), "width": hg.PATCH_WIDTH,
                "validation": "loss only; the last epoch is kept", "validation_mixed_paths": split.record()["validation_mixed_paths"],
                "validation_logs": int(split.val.sum()), "training_logs": int(split.train.sum()),
                "calibration_texts": 512}
    model, history = hg.train_line_model(
        make, counts, split.data.bad, np.flatnonzero(split.train), np.flatnonzero(split.val), flat=flat,
        seed=args.seed, epochs=args.epochs, batch=256, lr=1e-3, weight_decay=0.01,
        temperature=args.temperature, mask_rate=args.mask_rate, label_weight=args.label_weight,
        objective=args.objective, threads=args.threads, log=log,
        checkpoint=os.path.join(out, "checkpoint.pt"), lengths=patch_bytes, boundary_drop=args.boundary_drop)
    train_seconds = float(sum(row["seconds"] for row in history))
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))

    # Export and 8-bit.
    positions = hg.L_MAX if flat else hg.P_MAX
    rows = 8 if flat else hg.ROWS_PATCH
    float_path, int8_path = os.path.join(out, "line.f32.onnx"), os.path.join(out, "line.int8.onnx")

    class Graph(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.f, self.pool, self.head = net.f, net.pool, net.head

        def forward(self, x):
            return self.head(self.pool(self.f(x)))

    hg.export_onnx(Graph(model).eval(), (1, hg.FEATURES, rows, positions), ["e"], float_path)
    float_results = {"csic": (hg.line_vectors(model, make, len(sets["csic"])), counts)}
    make_p, counts_p = load("probes", "f32")
    float_results["probes"] = (hg.line_vectors(model, make_p, len(sets["probes"])), counts_p)
    del make, make_p
    rng = np.random.default_rng(args.seed)
    calibration = np.sort(rng.choice(np.flatnonzero(split.train), size=512, replace=False))
    make8, counts8 = load("csic", "int8")
    images = []
    for i in range(0, len(calibration), rows):
        x = make8(calibration[i:i + rows])
        image = np.zeros((1, hg.FEATURES, rows, positions), dtype=np.float32)
        image[0, :, :x.shape[0], :] = x[:, :, 0, :].transpose(1, 0, 2)
        images.append(image)
    hg.quantize(float_path, int8_path, images)
    sess = hg.session(int8_path, args.threads)

    record_model = {
        "task": "npu-log-decision 2.16", "variant": name, **{k: v for k, v in split.record().items() if k in ("data", "split")},
        **environment(), "settings": settings,
        "training": {"history": history, "chosen_epoch": args.epochs, "seconds": train_seconds},
        "parameters": {"line_graph": int(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("classifier"))),
                       **read_json(os.path.join(lm_dir(args.lm_seed), "record.json"))["parameters"]},
        "files": {"line_f32_sha256": sha256_of(float_path), "line_int8_sha256": sha256_of(int8_path),
                  "line_int8_bytes": os.path.getsize(int8_path)},
    }
    vectors = {}
    for precision in ("f32", "int8"):
        if precision == "f32":
            result = float_results
        else:
            result = {"csic": (hg.onnx_line_vectors(sess, make8, len(sets["csic"])), counts8)}
            make_p, counts_p = load("probes", "int8")
            result["probes"] = (hg.onnx_line_vectors(sess, make_p, len(sets["probes"])), counts_p)
        vectors[precision] = result
        norms = np.linalg.norm(result["csic"][0], axis=1)
        record_model[f"raw_length_p01_p50_p99_{precision}"] = [float(v) for v in np.percentile(norms, [1, 50, 99])]
        # Kept as a file, so that a run the host stopped does not score twice.
        triage_path = os.path.join(out, f"triage-{precision}.json")
        if not os.path.exists(triage_path):
            write_json(triage_path, {"vectors_sha256": hashlib.sha256(result["csic"][0].tobytes()).hexdigest(),
                                     "triage": hg.triage(result["csic"][0], split, cv5=True)})
        saved = read_json(triage_path)
        assert saved["vectors_sha256"] == hashlib.sha256(result["csic"][0].tobytes()).hexdigest(), \
            f"{triage_path} belongs to other vectors; delete it"
        record_model[f"triage_{precision}"] = saved["triage"]
        t = record_model[f"triage_{precision}"]
        log(f"  {precision}: split17 head {t['split17']['linear_head']['roc_auc']:.4f} neighbours {t['split17']['knn_k11']['roc_auc']:.4f}; "
            f"cv5 pooled head {t['cv5']['linear_head_pooled']:.4f} neighbours {t['cv5']['knn_k11_pooled']:.4f}")
    cosine = np.sum(l2dsci.l2n(vectors["f32"]["csic"][0]) * l2dsci.l2n(vectors["int8"]["csic"][0]), axis=1)
    record_model["float_against_8bit_cosine_min_p01_median"] = [float(cosine.min()), float(np.percentile(cosine, 1)), float(np.median(cosine))]
    record_model["cv5_note"] = ("the encoder was fitted once, on the training part of split17; in the cv5 folds the "
                                "scorers hold no label of the held-out fold, but the encoder has seen unlabelled texts of "
                                "four of the five large endpoints" + ("; this variant also saw their labels" if args.label_weight > 0 else ""))
    record_model["seconds"] = time.perf_counter() - started
    log(f"  float against 8-bit cosine: min {cosine.min():.4f}, p01 {np.percentile(cosine, 1):.4f}, median {np.median(cosine):.4f}")
    for precision in ("f32", "int8"):
        write_export(hg.path("vectors", f"{name}-{precision}", "x")[:-2], split, probe_rows, sets["probes"],
                     vectors[precision]["csic"][0], vectors[precision]["probes"][0],
                     vectors[precision]["csic"][1], vectors[precision]["probes"][1],
                     {"models": [dict(record_model, precision=precision)]})
    write_json(os.path.join(out, "record.json"), record_model)
    with open(os.path.join(out, "train.log"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")
    log(f"variant {name}: {record_model['seconds']:.0f} s")


# --------------------------------------------------------------------------------------
# Step 4: the reference rows (published encoder, random unit vectors, baseline)
# --------------------------------------------------------------------------------------

def step_reference(args) -> None:
    split = hg.Split()
    data = split.data
    published_dir = r"D:\BuildArtifacts\log2decision\task-2.15\vectors"
    with open(os.path.join(published_dir, "csic.jsonl"), encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert [r["text"] for r in rows] == data.texts, "the export of task 2.15 is in another order"
    published = np.fromfile(os.path.join(published_dir, "csic.f32"), dtype="<f4").reshape(len(rows), -1)
    rng = np.random.default_rng(20261114)
    random = rng.standard_normal(published.shape).astype(np.float32)
    out = {"published_npu_vectors": hg.triage(published, split, cv5=False),
           "random_unit_vectors": hg.triage(random, split, cv5=True), "baseline": {}}
    mixed = set(data.mixed_groups)
    held_groups = [g for g, h in zip(data.groups, split.held) if h]
    for routing in ("strip", "keep"):
        feature_rows = l2dsci.hashed_rows(data.texts, routing)
        model = l2dsci.fit_logreg(feature_rows[split.fit], data.bad[split.fit], l2=1.0)
        result = l2dsci.metrics(model.predict_proba(feature_rows[split.held])[:, 1], data.bad[split.held], held_groups, mixed)
        out["baseline"][routing] = {"roc_auc": result["cross_endpoint"]["roc_auc"],
                                    "per_group": {g: b.get("roc_auc") for g, b in result["per_group"].items()}}
    write_json(hg.path("reference", "triage.json"), out)
    print(json.dumps(out, indent=1))


# --------------------------------------------------------------------------------------
# Step 5: the files of the NPU test, and the comparison of what it wrote
# --------------------------------------------------------------------------------------

def step_npu_fixture(args) -> None:
    variant = hg.path("variants", args.variant, "x")[:-2]
    record = read_json(os.path.join(variant, "record.json"))
    seed = record["settings"]["byte_model_seed"]
    patcher = record["settings"]["patcher"]
    out = hg.path("npu", args.variant, "x")[:-2]
    byte_graph = "byte-ent" if patcher.startswith("entropy") else "byte-feat"
    shutil.copyfile(os.path.join(lm_dir(seed), f"{byte_graph}.int8.onnx"), os.path.join(out, "byte.int8.onnx"))
    shutil.copyfile(os.path.join(lm_dir(seed), "table.bin"), os.path.join(out, "table.bin"))
    shutil.copyfile(os.path.join(variant, "line.int8.onnx"), os.path.join(out, "line.int8.onnx"))
    split = hg.Split()
    probe_rows = hg.read_probes()
    texts = split.data.texts + [r["text"] for r in probe_rows]
    with open(os.path.join(out, "texts.jsonl"), "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps(text, ensure_ascii=False) + "\n")
    reference = hg.path("vectors", f"{args.variant}-int8", "x")[:-2]
    both = np.concatenate([np.fromfile(os.path.join(reference, f"{which}.f32"), dtype="<f4") for which in ("csic", "probes")])
    both.astype("<f4").tofile(os.path.join(out, "reference.f32"))
    layout = {"variant": args.variant, "patcher": patcher, "patcher_setting": record["settings"]["patcher_setting"],
              "input_bytes": hg.L_MAX, "patch_limit": hg.P_MAX, "max_patch_bytes": hg.MAX_PATCH,
              "stride": hg.STRIDE, "delimiters": hg.DELIMITERS.decode("ascii"),
              "table_width": hg.TABLE_WIDTH, "features": hg.FEATURES, "dimension": hg.OUTPUT_DIM,
              "byte_rows": hg.ROWS_BYTE, "patch_rows": hg.ROWS_PATCH, "texts": len(texts),
              "csic_texts": len(split.data.texts), "byte_outputs": ["h", "y"] if byte_graph == "byte-ent" else ["h"]}
    if byte_graph == "byte-ent":
        # The step of the 8-bit logits: the Rust side takes the exponentials from a table.
        import onnx
        from onnx import numpy_helper

        graph = onnx.load(os.path.join(out, "byte.int8.onnx")).graph
        constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
        (node,) = [n for n in graph.node if n.op_type == "DequantizeLinear" and n.output[0] == "y"]
        layout["logit_scale"] = float(constants[node.input[1]])
        layout["logit_zero_point"] = int(constants[node.input[2]])
    write_json(os.path.join(out, "layout.json"), layout)
    print(f"wrote the NPU fixture of {args.variant} to {out}: {layout}")


def step_npu_compare(args) -> None:
    """Compares the vectors and patch boundaries the Rust test wrote with the Python
    8-bit reference, and writes an export directory of the NPU vectors for the
    qualification scripts."""
    out = hg.path("npu", args.variant, "x")[:-2]
    layout = read_json(os.path.join(out, "layout.json"))
    n, c = layout["texts"], layout["csic_texts"]
    npu = np.fromfile(os.path.join(out, "npu.f32"), dtype="<f4").reshape(n, -1)
    ref = np.fromfile(os.path.join(out, "reference.f32"), dtype="<f4").reshape(n, -1)
    cosine = np.sum(npu.astype(np.float64) * ref.astype(np.float64), axis=1)
    identical = int((npu == ref).all(axis=1).sum())
    print(f"NPU against the Python 8-bit reference, {n} texts: bit-identical vectors {identical}; cosine min {cosine.min():.6f}, "
          f"p01 {np.percentile(cosine, 1):.6f}, median {np.median(cosine):.6f}; below 0.999: {int((cosine < 0.999).sum())}, below 0.99: {int((cosine < 0.99).sum())}")
    reference = hg.path("vectors", f"{args.variant}-int8", "x")[:-2]
    with open(os.path.join(reference, "csic.jsonl"), encoding="utf-8") as handle:
        ref_patches = [json.loads(line)["patches"] for line in handle]
    with open(os.path.join(reference, "probes.jsonl"), encoding="utf-8") as handle:
        ref_patches += [json.loads(line)["patches"] for line in handle]
    npu_patches = np.fromfile(os.path.join(out, "npu-patches.u32"), dtype="<u4")
    same = int((npu_patches == np.array(ref_patches)).sum())
    print(f"lines with the same patch count as the reference: {same} of {n}")
    target = hg.path("vectors", f"{args.variant}-npu", "x")[:-2]
    for file_name in ("csic.jsonl", "probes.jsonl", "record.json"):
        shutil.copyfile(os.path.join(reference, file_name), os.path.join(target, file_name))
    npu[:c].astype("<f4").tofile(os.path.join(target, "csic.f32"))
    npu[c:].astype("<f4").tofile(os.path.join(target, "probes.f32"))
    write_json(os.path.join(out, "compare.json"), {
        "texts": n, "bit_identical_vectors": identical, "cosine_min": float(cosine.min()),
        "cosine_p01": float(np.percentile(cosine, 1)), "cosine_median": float(np.median(cosine)),
        "below_0.999": int((cosine < 0.999).sum()), "below_0.99": int((cosine < 0.99).sum()),
        "same_patch_count": same})


def step_npu_stages(args) -> None:
    """Where do the NPU and ONNX Runtime's CPU run of the same 8-bit graphs part? Reads
    the stage outputs the Rust test wrote (`stage_outputs_for_the_comparison`) for the
    first lines of the fixture and runs the same graphs on the same inputs here."""
    out = hg.path("npu", args.variant, "x")[:-2]
    layout = read_json(os.path.join(out, "layout.json"))
    rows_b, rows_p = layout["byte_rows"], layout["patch_rows"]
    table = np.fromfile(os.path.join(out, "table.bin"), dtype="<f4").reshape(256, hg.TABLE_WIDTH)
    with open(os.path.join(out, "texts.jsonl"), encoding="utf-8") as handle:
        texts = [json.loads(line) for line in handle][:rows_p]
    lines = hg.Lines(texts)
    byte = hg.session(os.path.join(out, "byte.int8.onnx"), args.threads)
    line = hg.session(os.path.join(out, "line.int8.onnx"), args.threads)
    import onnx
    from onnx import numpy_helper

    def step_of(model_path, tensor):
        graph = onnx.load(model_path).graph
        constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
        (node,) = [n for n in graph.node if n.op_type == "DequantizeLinear" and n.output[0] == tensor]
        return float(constants[node.input[1]])

    def compare(name, npu, cpu, step):
        steps = np.abs(npu.astype(np.float64) - cpu.astype(np.float64)) / step
        print(f"{name}: {npu.size} values, 8-bit step {step:.5f}; equal {float((steps < 0.01).mean()):.4f}, "
              f"one step apart {float(((steps >= 0.01) & (steps < 1.5)).mean()):.4f}, further {float((steps >= 1.5).mean()):.4f}; "
              f"largest difference {steps.max():.1f} steps, mean {steps.mean():.3f} steps")
        return {"values": int(npu.size), "step": step, "equal": float((steps < 0.01).mean()),
                "one_step": float(((steps >= 0.01) & (steps < 1.5)).mean()), "further": float((steps >= 1.5).mean()),
                "largest_steps": float(steps.max()), "mean_steps": float(steps.mean())}

    report = {}
    cpu = byte.run(None, {"x": hg.byte_image(table, lines.ids[:rows_b], rows_b)})
    valid = np.zeros((1, 1, rows_b, hg.L_MAX), dtype=bool)
    for row in range(rows_b):
        valid[0, 0, row, :lines.lens[row]] = True
    for index, name in enumerate(layout["byte_outputs"]):
        npu = np.fromfile(os.path.join(out, f"npu-stage-{name}.f32"), dtype="<f4").reshape(cpu[index].shape)
        step = step_of(os.path.join(out, "byte.int8.onnx"), name)
        report[f"byte graph, output {name}, all positions"] = compare(f"byte graph, output {name}, all positions", npu, cpu[index], step)
        mask = np.broadcast_to(valid, npu.shape)
        report[f"byte graph, output {name}, positions of the lines"] = compare(
            f"byte graph, output {name}, positions of the lines", npu[mask], cpu[index][mask], step)
    patch_image = np.fromfile(os.path.join(out, "npu-stage-p.f32"), dtype="<f4").reshape(1, hg.FEATURES, rows_p, hg.P_MAX)
    npu_e = np.fromfile(os.path.join(out, "npu-stage-e.f32"), dtype="<f4").reshape(1, hg.OUTPUT_DIM, rows_p, 1)
    cpu_e = line.run(None, {"x": patch_image})[0]
    report["line graph on the NPU's patch image"] = compare(
        "line graph on the NPU's patch image", npu_e, cpu_e, step_of(os.path.join(out, "line.int8.onnx"), "e"))
    cosine = np.sum(l2dsci.l2n(npu_e[0, :, :, 0].T) * l2dsci.l2n(cpu_e[0, :, :, 0].T), axis=1)
    print(f"line graph on the same patch image, NPU against CPU: cosine min {cosine.min():.6f}, median {np.median(cosine):.6f}")
    report["line graph cosine min and median"] = [float(cosine.min()), float(np.median(cosine))]
    write_json(os.path.join(out, "stages.json"), report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["bytelm", "features", "variant", "reference", "npu-fixture", "npu-compare", "npu-stages"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--name")
    parser.add_argument("--patcher", choices=["fixed", "delimiter", "entropy-jump", "entropy-global", "flat"])
    parser.add_argument("--objective", choices=["contrastive", "masked"], default="contrastive")
    parser.add_argument("--label-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--boundary-drop", type=float, default=0.0,
                        help="share of the patch boundaries each view loses at random (not for the flat control)")
    parser.add_argument("--variant")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 8 if args.step == "bytelm" else 12
    {"bytelm": step_bytelm, "features": step_features, "variant": step_variant, "reference": step_reference,
     "npu-fixture": step_npu_fixture, "npu-compare": step_npu_compare, "npu-stages": step_npu_stages}[args.step](args)


if __name__ == "__main__":
    main()
