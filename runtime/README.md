# Runtime files

`l2d` runs its encoder through ONNX Runtime with AMD's Vitis AI execution provider.
The runtime is AMD's own build and is loaded at run time from this directory. It is
not part of the repository and is not downloaded by the build.

| Directory | Content | In git |
| --- | --- | --- |
| `runtime/ryzenai/` | AMD Ryzen AI runtime DLLs (table below) | no (ignored) |
| `runtime/cache/` | Compiled models of the provider, one folder per cache key | no (ignored) |

## Required files in `runtime/ryzenai/`

The tool is built for **AMD Ryzen AI 1.9.0** (ONNX Runtime 1.30.0).

That is the one release feature of `crates/l2d-encoder/Cargo.toml` (`ryzenai-1-9`, on
by default). It stands for the manifests the tool has compiled in and sets the API
level `ort` asks the runtime for. The line above and the tables below are held to it
and to the manifests by tests. The tool has the manifests of this release only. A
directory that holds another release is refused: when its record names that release,
the message names both; without a record it names this release and every file that is
not this release's, with the size or hash found and expected, the file version a DLL
of another release reports, and each DLL this release does not have.

A manifest is for one release from one source:
`runtime/ryzenai-<release>.<source>.manifest.json`, the source being `wheel` (AMD's
Python wheel) or `nuget` (AMD's NuGet package). It names the package and gives for each
file its size, SHA-256 and file version. AMD ships the same builds of release 1.9.0 in
both packages, signed separately: the files have the same names, sizes and versions and
other hashes. The tool accepts a directory that matches either manifest of its built
release with all its files, and says which source it matched; a directory with some
files of each is refused as a mix, naming which file is from which source.

A build of `l2d-encoder` with `--no-default-features` and without that feature does not
compile, and the error says why: there is no build without a release. A build with
`--all-features` compiles.

Release 1.8.0 was tried on 2026-11-12 and is not pursued (owner, 2026-11-13: the tool
stays on 1.9.0). Its build switch, manifest and table were removed; the trial's result
and how to bring a release back are in the notes of tasks 2.3, 2.13 and 2.4 in
`.kiro/specs/npu-log-decision/tasks.md`.

The files are not in the repository: `runtime/ryzenai/` is ignored by git, and each
user obtains them from AMD's package and places them here (section "Provisioning").

### AMD Ryzen AI 1.9.0, wheel (the built release)

`runtime/ryzenai-1.9.0.wheel.manifest.json`. As shipped in the Python wheel `onnxruntime-vitisai` 1.30.0 (author Advanced Micro
Devices, Inc.; file versions as the DLLs report them). All five must be present, and no
other DLL. Proven by the hardware gate of task 2.2 on 2026-11-12 with exactly these five
files in the directory.

| File | Bytes | File version | SHA-256 |
| --- | ---: | --- | --- |
| `onnxruntime.dll` | 25,442,896 | 1.30.0 | `152d62cf8d52eff8c203bdb19aa0ecc8ab698c10425782d84926b8550d4f96fd` |
| `onnxruntime_providers_shared.dll` | 117,840 | 1.30.0 | `a9ee9253bcecc4c8c6e3010bbf80f85350f30b0d566b4e44db4668750f92fdb4` |
| `onnxruntime_providers_vitisai.dll` | 466,000 | 1.30.0 | `941193bb53ab056967ec62def14b36334ac6032d37dc8f0c0e755d4517aadc03` |
| `onnxruntime_vitisai_ep.dll` | 517,920,336 | 1.9.0 | `416ada2dd0c79638ed9aa50ba29205e80785882faa8931d8125cf805df73f74a` |
| `dyn_dispatch_core.dll` | 169,647,696 | 1.9.0 | `60073393b6a4dcbbea454012104217de064c5e5948c1e68a917a5ca52d0dfc34` |

Total: 713,594,768 bytes (680.5 MiB).

### AMD Ryzen AI 1.9.0, nuget (the built release)

`runtime/ryzenai-1.9.0.nuget.manifest.json`. As shipped in AMD's NuGet package
`RyzenAI_Deployment.1.9.0.nupkg` (289,269,284 bytes, SHA-256
`b96db85a612f8ac5579a6684ef8f254b27c6e181df3ab6d86bc98f2c97ffc3e9`; directory
`runtimes/win-x64/native/`). The same five files and no other DLL.

| File | Bytes | File version | SHA-256 |
| --- | ---: | --- | --- |
| `onnxruntime.dll` | 25,442,896 | 1.30.0 | `a495db24b3bcf337c83b5145a6cfc49c81354fe58eced433a573cf7c25d40d5c` |
| `onnxruntime_providers_shared.dll` | 117,840 | 1.30.0 | `86322cd2d6665a37a0b1eecb2c99ebcebd9b0b25944cea4d8b471387db5068ed` |
| `onnxruntime_providers_vitisai.dll` | 466,000 | 1.30.0 | `d33a827b2c9eeb82d87c483198a381dc3979963e09ab5bfa5f94f98a49eeee73` |
| `onnxruntime_vitisai_ep.dll` | 517,920,336 | 1.9.0 | `a92cef8e451b1e0a404f9dacdd8d15d6af7ea7bf8be82dea2715243c22ef0d35` |
| `dyn_dispatch_core.dll` | 169,647,696 | 1.9.0 | `32ca7381bdeceae210ce2d7b106f2aa2db46c852e669a669a05c5cb75a0ef7c9` |

Total: 713,594,768 bytes (680.5 MiB).

Each of the five is the wheel's file with another signature: with the Authenticode
signature and the checksum field of the header left out, the wheel's and the package's
file are byte for byte the same (compared on 2026-11-12; the signature block has the
same place and size in both). The tool does not rely on that; it compares whole files
with the manifest it matched.

## What the files are for (1.9.0)


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

Release 1.9.0 comes from AMD in two packages that hold the same builds, signed
separately: the Python wheel `onnxruntime-vitisai` 1.30.0 on AMD's index for Ryzen AI
1.9.0, and the NuGet package `RyzenAI_Deployment.1.9.0.nupkg`. Either is a source the
tool accepts (section "Provisioning"). Python and NuGet are only the packaging the files
arrive in; nothing of either is loaded at run time.

On the reference host the files of `runtime/ryzenai/` were copied by hand on 2026-11-12,
unchanged, from the environment that pip had installed the wheel into for the spike of
2026-11-11 (`D:\Source\memguard_alpha\tools\oga-venv\Lib\site-packages\onnxruntime\capi\`);
they are the wheel's files and match the wheel manifest. Provisioning from the NuGet
package replaces them by that package's files, with a record that names the package.

## Another location: `L2D_RUNTIME_DIR`

The tool reads the files from `runtime/ryzenai/` of the project. Set `L2D_RUNTIME_DIR`
to read them from another directory instead; an empty value counts as not set. The
files are 714 MB and the same for every checkout on a host, so one copy can serve
several working copies. It is the only location an environment variable moves:
`models/`, `data/` and the provider's cache `runtime/cache/` stay in the project. The
directory is checked against the same manifests wherever it is.

Precedence: a directory given by an option of a command, then `L2D_RUNTIME_DIR`, then
`runtime/ryzenai/`. A relative value is relative to the current directory of the
process, not to the project.

## Provisioning (development step, needs `uv`)

`tools/runtime/provision.py` places the files of a release in the runtime directory and
writes the record `provisioned.json` there. The release is the one the tool is built
for, unless `--release` names another; the source is the NuGet package when `--nupkg`
or `--nuget-zip` is given, and the wheel otherwise; the two together name the manifest.
Both routes hash every file against the manifest before anything is copied, refuse a
target that holds a DLL the manifest does not name, place each file through a temporary
file that is hashed before it takes its name, and refuse the other kind of manifest.
Refused as a target: a drive root, the directory the package or ZIP lies in, and any
directory inside the repository other than `runtime/ryzenai/`. A refusal names what
was found and what was expected and changes nothing (exit code 1); a copy that fails
ends with one line and exit code 2. The tests of both routes:

    uv run --no-project --python 3.12 python -m unittest discover -s tools/runtime

### From AMD's NuGet package (release 1.9.0, the built release; needs no index)

    uv run --no-project --python 3.12 python tools/runtime/provision.py --nupkg "<path>\RyzenAI_Deployment.1.9.0.nupkg" [--target <directory>]

The package is the file AMD publishes
(`https://download.amd.com/opendownload/RyzenAI/1.9.0/RyzenAI_Deployment.1.9.0.nupkg`,
289,269,284 bytes); the user downloads it, the script does not. It is read where it
lies and is not copied or changed. The script compares the file's size with the
manifest's before it reads it, hashes it as it reads it and compares the SHA-256,
requires the id and version stated inside to be `RyzenAI_Deployment` 1.9.0, and reads
each of the five files by the name the manifest gives it. No name or path found in the
package is used to read or to write. The record names the package (file, path, size,
SHA-256) as the source. No environment is made; `uv` only supplies a Python 3.12.

Run on 2026-11-12 on the reference host into a scratch directory: five files,
713,594,768 bytes, in 11 s. The tool's full check passes on it and reports the source
`nuget`. The hardware gate passes on it with the default build, with the same result
as on the wheel's files: 36 of 38 nodes on the NPU, cosine 0.9958 against the CPU
provider, and the NPU output and the CPU output of the same input are bit for bit the
same from both directories (two runs each).

### From a ZIP that holds AMD's NuGet package (needs no index)

    uv run --no-project --python 3.12 python tools/runtime/provision.py --nuget-zip "<path of the ZIP>" [--target <directory>]

AMD publishes release 1.9.0 as a bare package, so the command above is the usual one.
Where the package lies inside a ZIP (AMD published earlier releases that way), the ZIP
is read where it lies: the script opens `RyzenAI_Deployment.1.9.0.nupkg` at the top of
the ZIP by that name, the name the manifest gives, and then does what it does for a
package file, with the same checks of size, SHA-256, id and version. The record names
the ZIP (name, path, size, SHA-256) and the package inside it. A ZIP that holds another
package is refused, naming the package that was expected.

This route was run for real on 2026-11-12 with AMD's ZIP of release 1.8.0, whose
manifest has since been removed. With release 1.9.0 it is covered by the script's tests
on small ZIPs made for them, and by one that runs the command line against the tracked
1.9.0 manifest up to the refusal of a ZIP that does not hold AMD's package; it has not
been run on a ZIP that holds AMD's real 1.9.0 package.

### From AMD's wheel (release 1.9.0, the built release; needs AMD's index)

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
target (release, source, and for each file its size, SHA-256 and modification time).

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

- Which manifest: the built release has one per source. The directory must match one of
  them with all its files; the one whose hashes the record holds is tried first, and
  each file is hashed at most once. The tool says which source it matched. A directory
  with files of both sources is refused as a mix.
- At every start: the directory exists; every file of the manifest is there, is a file
  and has its size; no other DLL is in the directory; the record, if there is one, has
  a known format and names the built release; a file whose modification time differs
  from the record, that the record does not name, or that the record has with another
  size or SHA-256 than the manifest, is hashed (the last case: a directory that was
  verified against another source of the same release). When it is right, the
  record gets its new time and a note `reverified` (by the check, when, which files),
  and keeps who wrote it and where the files came from; the start after it is fast
  again. With a record and unchanged files a start takes 0.3 to 2.2 ms.
- In a directory without a record (filled by hand): every file is hashed (0.84 s), and
  when all are right the check writes the record itself, marked `"written_by": "check"`
  and without a source, so that later starts are fast. If the directory cannot be
  written to, every start hashes, and the tool says so.
- On request (the full check, 0.84 to 0.89 s): the SHA-256 of every file, record or not.

The record of the provisioning step (`"written_by": "provisioning"`, with the source)
and the record of the check are used and invalidated alike.

A record stays bound to the manifest it was made against: if the files are right for
the other source of the release (they were replaced by that source's), the record's
hashes are not that manifest's, every file is hashed, and a new record of the check
replaces the old one, whose source would no longer be true.

A record that cannot be read (not JSON, not the shape of a record) is treated as no
record: every file is hashed, and when all are right a new record replaces it, which
the tool says. A record that is there but cannot be opened at the moment (another
process has it open, access is denied) is not replaced: every file is hashed for this
start and the record is left as it is. Nothing has to be repaired by hand; the hashes decide, not the record. A
record of another format is different: it can be read, a later version of the tool may
have written it, and it is refused with the advice to remove it or to provision again.
Temporary record files that an interrupted write left behind are removed by the next
write that succeeds.

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
