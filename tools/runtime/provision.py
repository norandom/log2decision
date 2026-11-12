"""Copy the AMD runtime files of `l2d` from AMD's wheel into the runtime directory.

Development only. `l2d` is Rust and needs no Python; this script is the one step that
places AMD's native files where the tool loads them from (`runtime/README.md`).

    uv run --project tools/runtime python tools/runtime/provision.py

`uv run` first makes the environment of `tools/runtime/pyproject.toml`, which holds one
package: AMD's wheel `onnxruntime-vitisai`, pinned by its URL on AMD's release-specific
index and by its SHA-256 in `uv.lock`. The script imports nothing from it. It

1. reads the tracked manifest `runtime/ryzenai.manifest.json` (release, source, and for
   each file its size and SHA-256);
2. requires the installed package to be the manifest's version, installed from the
   manifest's index, and its `RECORD` to list the manifest's sizes and hashes;
3. hashes the files in the environment and requires the manifest's hashes, before
   anything is copied;
4. refuses a target that holds a DLL the manifest does not name (a file of another
   release, or one the tool was not proven with);
5. copies what is missing or different, through a temporary file that is hashed before
   it takes the final name;
6. writes `provisioned.json` into the target: release, source and for each file its
   size, SHA-256 and modification time. The tool's fast check uses the times to hash
   only files that changed since.

A refusal names what was found and what was expected, and changes nothing (exit code
1). A copy or write that fails ends with one line and exit code 2. The script
uses the standard library only and never downloads: fetching the wheel is uv's part.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import shutil
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "runtime" / "ryzenai.manifest.json"
DEFAULT_TARGET = REPO_ROOT / "runtime" / "ryzenai"
RECEIPT_FILE = "provisioned.json"
MANIFEST_FORMAT = 1
RECEIPT_FORMAT = 1


class ProvisioningError(RuntimeError):
    """A refusal: what was found, what was expected. Nothing was changed."""


@dataclass(frozen=True)
class ManifestFile:
    name: str
    bytes: int
    sha256: str
    version: str


@dataclass(frozen=True)
class Manifest:
    release: str
    onnxruntime: str
    index: str
    package: str
    version: str
    wheel: str
    directory: str
    files: tuple[ManifestFile, ...]

    @property
    def wheel_url(self) -> str:
        return f"{self.index}{self.package}/{self.wheel}"

    @property
    def release_text(self) -> str:
        return f"AMD Ryzen AI {self.release} (ONNX Runtime {self.onnxruntime})"


def load_manifest(path: Path) -> Manifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProvisioningError(f"the manifest {path} cannot be read: {error}") from error
    if data.get("format") != MANIFEST_FORMAT:
        raise ProvisioningError(
            f"the manifest {path} has format {data.get('format')!r}, this script reads "
            f"format {MANIFEST_FORMAT}"
        )
    try:
        source = data["source"]
        files = tuple(
            ManifestFile(
                str(entry["name"]), int(entry["bytes"]), str(entry["sha256"]), str(entry["version"])
            )
            for entry in data["files"]
        )
        manifest = Manifest(
            release=str(data["release"]),
            onnxruntime=str(data["onnxruntime"]),
            index=str(source["index"]),
            package=str(source["package"]),
            version=str(source["version"]),
            wheel=str(source["wheel"]),
            directory=str(source["directory"]),
            files=files,
        )
    except KeyError as error:
        raise ProvisioningError(f"the manifest {path} lacks the key {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise ProvisioningError(f"the manifest {path} is not of the expected shape: {error}") from error
    if not files:
        raise ProvisioningError(f"the manifest {path} names no file")
    for entry in files:
        if Path(entry.name).name != entry.name:
            raise ProvisioningError(f"the manifest {path} names {entry.name!r}, not a plain file name")
    return manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _normalised(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_")


def find_dist_info(site_packages: Path, manifest: Manifest) -> Path:
    """The installed package's metadata directory; its version must be the manifest's."""
    if not site_packages.is_dir():
        raise ProvisioningError(
            f"the environment's package directory {site_packages} does not exist; make the "
            "environment first: uv sync --project tools/runtime"
        )
    wanted = _normalised(manifest.package)
    found: list[tuple[str, Path]] = []
    for candidate in sorted(site_packages.glob("*.dist-info")):
        name, _, version = candidate.name.removesuffix(".dist-info").rpartition("-")
        if _normalised(name) == wanted:
            found.append((version, candidate))
    if not found:
        raise ProvisioningError(
            f"the package {manifest.package} is not installed in {site_packages}; make the "
            "environment first: uv sync --project tools/runtime"
        )
    versions = [version for version, _ in found]
    if versions != [manifest.version]:
        raise ProvisioningError(
            f"{site_packages} holds {manifest.package} {', '.join(versions)}; "
            f"{manifest.release_text} is {manifest.package} {manifest.version}. Versions are "
            "not mixed: sync the environment of tools/runtime, which pins that release"
        )
    return found[0][1]


def installed_from(dist_info: Path) -> str | None:
    """The URL the installer recorded for a package installed from a URL, if any."""
    record = dist_info / "direct_url.json"
    if not record.is_file():
        return None
    try:
        url = json.loads(record.read_text(encoding="utf-8")).get("url")
    except ValueError as error:
        raise ProvisioningError(f"{record} cannot be read: {error}") from error
    return url if isinstance(url, str) else None


def check_origin(dist_info: Path, manifest: Manifest, accept_unrecorded: bool) -> str | None:
    url = installed_from(dist_info)
    if url is None:
        if accept_unrecorded:
            return None
        raise ProvisioningError(
            f"{dist_info} does not record where the package was installed from "
            f"(no direct_url.json), so it cannot be told to be {manifest.wheel_url}. Use the "
            "environment of tools/runtime, or pass --accept-unrecorded-origin to rely on the "
            "file hashes alone"
        )
    if url != manifest.wheel_url:
        raise ProvisioningError(
            f"the package in {dist_info.parent} was installed from {url}; "
            f"{manifest.release_text} comes from {manifest.wheel_url}"
        )
    return url


def recorded_files(dist_info: Path) -> dict[str, tuple[str, int]]:
    """`RECORD` of the installed wheel: path -> (hexadecimal SHA-256, size)."""
    record = dist_info / "RECORD"
    try:
        lines = record.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProvisioningError(f"{record} cannot be read: {error}") from error
    entries: dict[str, tuple[str, int]] = {}
    for line in lines:
        path, _, rest = line.rpartition(",sha256=")
        digest, _, size = rest.partition(",")
        if not path or not size.isdigit():
            continue
        padded = digest + "=" * (-len(digest) % 4)
        entries[path.replace("\\", "/")] = (base64.urlsafe_b64decode(padded).hex(), int(size))
    return entries


def verify_source(site_packages: Path, dist_info: Path, manifest: Manifest) -> list[Path]:
    """The files to copy, after they were found to be the manifest's. Names every
    difference."""
    recorded = recorded_files(dist_info)
    source_dir = site_packages.joinpath(*manifest.directory.split("/"))
    problems: list[str] = []
    sources: list[Path] = []
    for entry in manifest.files:
        key = f"{manifest.directory}/{entry.name}"
        if key not in recorded:
            problems.append(f"the wheel's RECORD does not list {key}")
        elif recorded[key] != (entry.sha256, entry.bytes):
            problems.append(
                f"the wheel's RECORD lists {key} with {recorded[key][1]} bytes, SHA-256 "
                f"{recorded[key][0]}; the manifest has {entry.bytes} bytes, SHA-256 {entry.sha256}"
            )
        path = source_dir / entry.name
        if not path.is_file():
            problems.append(f"{path} is missing")
            continue
        size = path.stat().st_size
        if size != entry.bytes:
            problems.append(f"{path} has {size} bytes, the manifest has {entry.bytes}")
            continue
        found = file_sha256(path)
        if found != entry.sha256:
            problems.append(f"{path} has SHA-256 {found}, the manifest has {entry.sha256}")
            continue
        sources.append(path)
    if problems:
        raise ProvisioningError(
            f"the package in {site_packages} is not {manifest.release_text} as the manifest "
            "describes it; nothing was copied: " + "; ".join(problems)
        )
    return sources


def refuse_unusable_target(target: Path, manifest: Manifest) -> None:
    resolved = target.resolve()
    if resolved == Path(resolved.anchor):
        raise ProvisioningError(f"the target {target} is a drive root")
    if target.exists() and not target.is_dir():
        raise ProvisioningError(f"the target {target} is not a directory")
    if not target.is_dir():
        return
    named = {entry.name.lower() for entry in manifest.files}
    stray = sorted(
        path.name
        for path in target.iterdir()
        if path.suffix.lower() == ".dll" and path.name.lower() not in named
    )
    if stray:
        raise ProvisioningError(
            f"the target {target} holds DLLs that are not part of {manifest.release_text}: "
            f"{', '.join(stray)}. Files of different releases are not mixed; remove them or "
            "choose an empty directory. Nothing was copied"
        )


def previous_release(target: Path) -> str | None:
    """The release a provisioning record in the target names, if there is one."""
    try:
        data = json.loads((target / RECEIPT_FILE).read_text(encoding="utf-8"))
        return f"{data['release']} (ONNX Runtime {data['onnxruntime']})"
    except (OSError, ValueError, KeyError, TypeError):
        return None


def copy_verified(source: Path, destination: Path, entry: ManifestFile) -> bool:
    """Copies one file unless the destination already is it. True when it copied."""
    if (
        destination.is_file()
        and destination.stat().st_size == entry.bytes
        and file_sha256(destination) == entry.sha256
    ):
        return False
    partial = destination.with_name(f".{destination.name}.partial")
    try:
        shutil.copyfile(source, partial)
        found = file_sha256(partial)
        if found != entry.sha256:
            raise ProvisioningError(
                f"the copy of {source} has SHA-256 {found}, expected {entry.sha256}"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return True


def write_receipt(
    target: Path, manifest: Manifest, site_packages: Path, origin: str | None
) -> Path:
    receipt = {
        "format": RECEIPT_FORMAT,
        "release": manifest.release,
        "onnxruntime": manifest.onnxruntime,
        # The tool's check writes a record of its own kind ("check") into a directory
        # that has none; this one also says where the files came from.
        "written_by": "provisioning",
        "provisioned_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": {
            "index": manifest.index,
            "package": manifest.package,
            "version": manifest.version,
            "wheel": manifest.wheel,
            # The URL the installer recorded; null when it recorded none.
            "installed_from": origin,
            "site_packages": str(site_packages),
        },
        "files": [
            {
                "name": entry.name,
                "bytes": entry.bytes,
                "sha256": entry.sha256,
                "modified_unix_ns": (target / entry.name).stat().st_mtime_ns,
            }
            for entry in manifest.files
        ],
    }
    path = target / RECEIPT_FILE
    partial = path.with_name(f".{RECEIPT_FILE}.partial")
    partial.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)
    return path


def provision(
    *,
    manifest_path: Path,
    site_packages: Path,
    target: Path,
    accept_unrecorded_origin: bool = False,
) -> list[str]:
    """Runs the steps of the module text. Returns the lines of its report."""
    manifest = load_manifest(manifest_path)
    dist_info = find_dist_info(site_packages, manifest)
    origin = check_origin(dist_info, manifest, accept_unrecorded_origin)
    refuse_unusable_target(target, manifest)
    sources = verify_source(site_packages, dist_info, manifest)

    report = [
        f"release: {manifest.release_text}",
        f"source: {manifest.package} {manifest.version} in {site_packages}, installed from "
        + (origin or "an origin the installer did not record (accepted on request)"),
    ]
    before = previous_release(target) if target.is_dir() else None
    if before is not None and before != f"{manifest.release} (ONNX Runtime {manifest.onnxruntime})":
        report.append(f"the target was provisioned for {before}; its files are replaced")
    target.mkdir(parents=True, exist_ok=True)
    for source, entry in zip(sources, manifest.files, strict=True):
        copied = copy_verified(source, target / entry.name, entry)
        report.append(
            f"{'copied ' if copied else 'present'} {entry.name}  {entry.bytes} bytes  "
            f"sha256 {entry.sha256}"
        )
    receipt = write_receipt(target, manifest, site_packages, origin)
    total = sum(entry.bytes for entry in manifest.files)
    report.append(f"{len(manifest.files)} files, {total} bytes in {target}; record: {receipt}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy the AMD runtime files of l2d from AMD's wheel into the runtime directory."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="runtime directory to fill (default: runtime/ryzenai of this checkout)",
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path(sysconfig.get_paths()["purelib"]),
        help="package directory that holds AMD's wheel (default: the running environment's)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest to provision (default: runtime/ryzenai.manifest.json)",
    )
    parser.add_argument(
        "--accept-unrecorded-origin",
        action="store_true",
        help="go on when the installer did not record the wheel's URL; the hashes still decide",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = provision(
            manifest_path=arguments.manifest,
            site_packages=arguments.site_packages,
            target=arguments.target,
            accept_unrecorded_origin=arguments.accept_unrecorded_origin,
        )
    except ProvisioningError as error:
        print(f"provision: refused: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        # A copy or a write that failed (disk full, a file in use, no permission). Files
        # already copied are whole, since each takes its name only after it was hashed;
        # the record is written last, so the target has none for an unfinished run.
        print(f"provision: failed: {error}", file=sys.stderr)
        return 2
    for line in report:
        print(f"provision: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
