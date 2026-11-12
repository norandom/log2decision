# Runtime files

`l2d` runs its encoder through ONNX Runtime with AMD's Vitis AI execution provider.
The runtime is AMD's own build and is loaded at run time from this directory. It is
not part of the repository and is not downloaded by the build.

| Directory | Content | In git |
| --- | --- | --- |
| `runtime/ryzenai/` | AMD Ryzen AI runtime DLLs (table below) | no (ignored) |
| `runtime/cache/` | Compiled models of the provider, one folder per cache key | no (ignored) |

## Required files in `runtime/ryzenai/`

AMD Ryzen AI 1.9, as shipped in the Python wheel `onnxruntime-vitisai` 1.30.0 (AMD,
MIT licence; file versions as the DLLs report them). All five must be present. Proven
by the hardware gate of task 2.2 on 2026-11-12 with exactly these five files in the
directory.

| File | Bytes | File version | SHA-256 |
| --- | ---: | --- | --- |
| `onnxruntime.dll` | 25,442,896 | 1.30.0 | `152d62cf8d52eff8c203bdb19aa0ecc8ab698c10425782d84926b8550d4f96fd` |
| `onnxruntime_providers_shared.dll` | 117,840 | 1.30.0 | `a9ee9253bcecc4c8c6e3010bbf80f85350f30b0d566b4e44db4668750f92fdb4` |
| `onnxruntime_providers_vitisai.dll` | 466,000 | 1.30.0 | `941193bb53ab056967ec62def14b36334ac6032d37dc8f0c0e755d4517aadc03` |
| `onnxruntime_vitisai_ep.dll` | 517,920,336 | 1.9.0 | `416ada2dd0c79638ed9aa50ba29205e80785882faa8931d8125cf805df73f74a` |
| `dyn_dispatch_core.dll` | 169,647,696 | 1.9.0 | `60073393b6a4dcbbea454012104217de064c5e5948c1e68a917a5ca52d0dfc34` |

Total: 713,594,768 bytes (680.5 MiB).

What each is for:

- `onnxruntime.dll`: ONNX Runtime 1.30.0, the library `ort` loads. Also holds the CPU
  provider.
- `onnxruntime_providers_shared.dll`, `onnxruntime_providers_vitisai.dll`: the bridge
  through which ONNX Runtime loads the Vitis AI provider.
- `onnxruntime_vitisai_ep.dll`: the provider itself (graph partitioning, compiler,
  NPU execution).
- `dyn_dispatch_core.dll`: loaded by the provider when it is registered. If it is
  missing, the process dies with an access violation (exit code `0xC0000005`) and no
  error is returned. The files must therefore be checked before the runtime is loaded.

Not needed (the wheel also holds them; the gate passes with each of them, and with all
of them, removed): `DirectML.dll`, `onnxruntime_providers_ryzenai.dll`,
`onnxruntime_vitis_ai_custom_ops.dll`, `waic_inmem.dll`, `waic_mlir.dll`, and the
Python module `onnxruntime_pybind11_state.pyd`. No `vaip_config.json` and no `.xclbin`
file is passed to the provider.

## Where the files come from on the reference host

Copied unchanged (hashes compared) from

    D:\Source\memguard_alpha\tools\oga-venv\Lib\site-packages\onnxruntime\capi\

which is the Python environment the spike of 2026-11-11 ran in. Python is only the
packaging the files arrived in; nothing of Python is loaded at run time.

To use another location, set `L2D_RUNTIME_DIR` to the directory that holds the five
files (the gate test reads it; default `runtime/ryzenai/` of the workspace).

## What must be on the host besides these files

- An AMD NPU with its driver. The provider talks to the NPU through `xrt_coreutil.dll`
  in `C:\Windows\System32` and `xrt_core.dll` in the driver's folder under
  `C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*`; both are
  installed by the NPU driver (seen at file version 32.00.20102.4481 with driver
  32.0.20102.4480).
- Nothing else: no Ryzen AI installation, no Python, no entry on `PATH`. The gate was
  run with `PATH` set to `C:\Windows\System32;C:\Windows` and the Rust toolchain.

## Checking

    $env:L2D_RUNTIME_DIR = "<path>\runtime\ryzenai"   # only if not the default
    $env:L2D_GATE_STRICT = "1"
    cargo test -p l2d-encoder --test npu_gate -- --ignored --nocapture

The test compiles the spike's 8-bit model for the NPU, runs batches of 32, and prints
the NPU driver version and the number of model nodes placed on the NPU and on the CPU.
Its compiled model goes to `runtime/cache/gate-2.2-<hash>/` next to the runtime
directory (or under `L2D_GATE_CACHE_DIR`).
