# Runtime files

`l2d` runs its encoder through ONNX Runtime with AMD's Vitis AI execution provider.
The runtime is AMD's own build and is loaded at run time from this directory. It is
not part of the repository and is not downloaded by the build.

| Directory | Content | In git |
| --- | --- | --- |
| `runtime/ryzenai/` | AMD Ryzen AI runtime DLLs (table below) | no (ignored) |
| `runtime/cache/` | Compiled models of the provider, one folder per cache key | no (ignored) |

## Required files in `runtime/ryzenai/`

AMD Ryzen AI 1.9, as shipped in the Python wheel `onnxruntime-vitisai` 1.30.0 (author
Advanced Micro Devices, Inc.; file versions as the DLLs report them). All five must be
present. Proven by the hardware gate of task 2.2 on 2026-11-12 with exactly these five
files in the directory.

The files are not in the repository: `runtime/ryzenai/` is ignored by git, and each
user obtains them from AMD's package and places them here with the provisioning step
below.

The table is also the tracked manifest `runtime/ryzenai.manifest.json`, which the tool
has compiled in and checks the directory against (a test holds the two together).

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

## Where the files come from

AMD's wheel `onnxruntime-vitisai` 1.30.0 of the index for Ryzen AI 1.9.0 (section
"Provisioning"). Python is only the packaging the files arrive in; nothing of Python is
loaded at run time. On the reference host the files of `runtime/ryzenai/` were copied by
hand on 2026-11-12, unchanged, from the environment that pip had installed that wheel
into for the spike of 2026-11-11:

    D:\Source\memguard_alpha\tools\oga-venv\Lib\site-packages\onnxruntime\capi\

## Another location: `L2D_RUNTIME_DIR`

The tool reads the files from `runtime/ryzenai/` of the project. Set `L2D_RUNTIME_DIR`
to read them from another directory instead; an empty value counts as not set. The
files are 714 MB and the same for every checkout on a host, so one copy can serve
several working copies. It is the only location an environment variable moves:
`models/`, `data/` and the provider's cache `runtime/cache/` stay in the project. The
directory is checked against the same manifest wherever it is.

Precedence: a directory given by an option of a command, then `L2D_RUNTIME_DIR`, then
`runtime/ryzenai/`. A relative value is relative to the current directory of the
process, not to the project.

## Provisioning (development step, needs `uv`)

    uv run --project tools/runtime python tools/runtime/provision.py
    uv run --project tools/runtime python tools/runtime/provision.py --target <directory>

`tools/runtime/` is a `uv` project with one dependency: AMD's wheel `onnxruntime-vitisai`
1.30.0, pinned by its URL on AMD's index for Ryzen AI 1.9.0,
`https://pypi.amd.com/ryzenai_llm/1.9.0/windows/simple/`. All five files are in that
wheel, in `onnxruntime/capi/`. What the wheel declares as requirements is not installed
(`voe` 1.9.0 would bring `torch` and `transformers`; it holds no DLL), and nothing of
the environment is imported or needed to run `l2d`. The environment is
`tools/runtime/.venv` (ignored by git); uv's cache is where `UV_CACHE_DIR` points.

`provision.py` requires the installed package to be the manifest's version from the
manifest's index, compares the wheel's `RECORD` and the SHA-256 of each file with the
manifest before it copies anything, refuses a target that holds a DLL the manifest does
not name, copies what is missing or different, and writes `provisioned.json` into the
target (release, source, and for each file its size, SHA-256 and modification time). A
refusal names what was found and what was expected and changes nothing. Its tests:

    uv run --no-project --python 3.12 python -m unittest discover -s tools/runtime

State of 2026-11-12: `pypi.amd.com` answered every request from the reference host with
403 "Access Denied" (the URL above and the 1.8.0 URL of the sibling project alike), so
the wheel was not fetched through this environment, there is no `tools/runtime/uv.lock`
yet, and the command above has not run by its default route. That the index serves the
wheel under that URL is known from pip's cache on the host: it holds the download of
2026-11-08 under the key of exactly that URL (307,779,608 bytes, SHA-256
`0f4b98625c4217832d4001acd4beb4306e65123c679afc5e5bb3282fc9064c8b`), and the five files
inside it have the sizes and hashes of the table. uv does not read pip's cache, so
`uv lock` has to wait for the index and should then record that hash. The step itself
was run with `--site-packages` pointing at the environment named under "Where the files
come from" and `--accept-unrecorded-origin` (pip records no URL for a package from an
index) into a scratch directory; the five files it placed are byte for byte those of
`runtime/ryzenai/`.

AMD's NuGet package `RyzenAI_Deployment.1.9.0.nupkg` holds five files of the same names
and sizes, but they are other builds of them (their CRC-32 values differ from the
wheel's files), so they do not pass the check against this manifest.

## What must be on the host besides these files

- An AMD NPU with its driver. The provider talks to the NPU through `xrt_coreutil.dll`
  in `C:\Windows\System32` and `xrt_core.dll` in the driver's folder under
  `C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*`; both are
  installed by the NPU driver (seen at file version 32.00.20102.4481 with driver
  32.0.20102.4480).
- Nothing else: no Ryzen AI installation, no Python, no entry on `PATH`. The gate was
  run with `PATH` set to `C:\Windows\System32;C:\Windows` and the Rust toolchain.

## Checking

The tool checks the directory before it loads anything from it
(`l2d_encoder::RuntimeFiles::check`). The rule: a file is used without being hashed only
when the record `provisioned.json` in the directory says it was hashed with the
modification time it has now.

- At every start: the directory exists; every file of the manifest is there, is a file
  and has its size; no other DLL is in the directory; the record, if there is one, has
  a known format and names release 1.9.0; a file whose modification time differs from
  the record, or that the record does not name, is hashed. With a record and unchanged
  files this takes 0.3 to 2.2 ms.
- In a directory without a record (filled by hand): every file is hashed (0.84 s), and
  when all are right the check writes the record itself, marked `"written_by": "check"`
  and without a source, so that later starts are fast. If the directory cannot be
  written to, every start hashes, and the tool says so.
- On request (the full check, 0.84 to 0.89 s): the SHA-256 of every file, record or not.

The record of the provisioning step (`"written_by": "provisioning"`, with the source)
and the record of the check are used and invalidated alike.

A missing file is reported with its expected path (exit code 3). A file of the wrong
size or content, a folder in the place of a file, a DLL that is not in the table, and a
directory whose record names another release or format are refused (exit code 4); the
message names each file, what was found and what was expected, and the file version a
wrong DLL reports. Not found at a start: a file replaced by another of the same size
whose modification time was set back to the recorded one, and a record written by hand;
the full check finds both.

    cargo test -p l2d-encoder --test runtime_files -- --ignored --nocapture

runs the checks against the directory and prints their times (in a directory without a
record it writes the record). The hardware gate:

    $env:L2D_RUNTIME_DIR = "<path>\runtime\ryzenai"   # only if not the default
    cargo test -p l2d-encoder --test npu_gate -- --ignored --nocapture

The test runs the full file check first (and writes the record into a directory that has none). It fails when the runtime files or the NPU device are missing, and when a Python
interpreter is on `PATH`. With `L2D_GATE_ALLOW_SKIP=1` it prints `SKIPPED` and passes
in the first two cases and only reports the interpreter in the third.

The test compiles the spike's 8-bit model for the NPU, runs batches of 32, and prints
the NPU driver version and the number of model nodes placed on the NPU and on the CPU.
Its compiled model goes to `runtime/cache/gate-2.2-<hash>/` next to the runtime
directory (or under `L2D_GATE_CACHE_DIR`).
