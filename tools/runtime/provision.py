"""Place the AMD runtime files of `l2d` in the runtime directory, from AMD's package.

Development only. `l2d` is Rust and needs no Python; this script is the one step that
places AMD's native files where the tool loads them from (`runtime/README.md`). Which
files, of which release and from which package, is said by a tracked manifest,
`runtime/ryzenai-<release>.manifest.json`; its `source.kind` decides the route.

Route `wheel` (AMD's wheel on its release-specific package index; needs the index):

    uv run --project tools/runtime python tools/runtime/provision.py

`uv run` first makes the environment of `tools/runtime/pyproject.toml`, which holds one
package: AMD's wheel `onnxruntime-vitisai`, pinned by its URL and by its SHA-256 in
`uv.lock`. The script imports nothing from it. It requires the installed package to be
the manifest's version, installed from the manifest's index, and its `RECORD` to list
the manifest's sizes and hashes.

Route `nuget` (AMD's NuGet package inside the ZIP AMD publishes; needs no index and no
environment, only the ZIP on disk):

    uv run --no-project --python 3.12 python tools/runtime/provision.py --release 1.8.0 --nuget-zip <path of the ZIP>

The ZIP is read where it is. The package inside it is opened by the name the manifest
gives, its SHA-256 is compared with the manifest's, and its id and version (the
`.nuspec` inside) must be the manifest's. Each file is read by the name the manifest
gives for it. No name or path found in the archive is used to read or to write: a
member with a path that points elsewhere is never looked at.

Both routes then do the same:

- hash every file of the source and require the manifest's sizes and hashes, before
  anything is copied;
- refuse a target that holds a DLL the manifest does not name (a file of another
  release, or one the tool was not proven with);
- place what is missing or different, through a temporary file that is hashed before
  it takes the final name;
- write `provisioned.json` into the target: release, source and for each file its
  size, SHA-256 and modification time. The tool's fast check uses it to hash only files
  that changed since.

A refusal names what was found and what was expected, and changes nothing (exit code
1). A copy or write that fails ends with one line and exit code 2. The script uses the
standard library only and never downloads: fetching the wheel is uv's part, and the ZIP
is the user's.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import io
import json
import os
import shutil
import sys
import sysconfig
import tomllib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = REPO_ROOT / "runtime" / "ryzenai"
RECEIPT_FILE = "provisioned.json"
MANIFEST_FORMAT = 1
RECEIPT_FORMAT = 1
#: What a manifest's source must name besides kind, package, version and directory.
SOURCE_KEYS = {"wheel": ("index", "wheel"), "nuget": ("archive", "nupkg", "nupkg_sha256")}


class ProvisioningError(RuntimeError):
    """A refusal: what was found, what was expected. Nothing was changed."""


def built_release(repo_root: Path = REPO_ROOT) -> str:
    """The release of the AMD runtime the tool is built for: the one default feature of
    `crates/l2d-encoder/Cargo.toml` (`ryzenai-1-9` is release 1.9.0). The manifest of
    that release is the one provisioned unless `--release` or `--manifest` names another."""
    path = repo_root / "crates" / "l2d-encoder" / "Cargo.toml"
    try:
        default = tomllib.loads(path.read_text(encoding="utf-8"))["features"]["default"]
    except (OSError, ValueError, KeyError) as error:
        raise ProvisioningError(f"the built release cannot be read from {path}: {error}") from error
    releases = [name for name in default if name.startswith("ryzenai-")]
    if len(releases) != 1:
        raise ProvisioningError(
            f"{path} must name exactly one release among its default features, found {releases}"
        )
    major, _, minor = releases[0].removeprefix("ryzenai-").partition("-")
    return f"{major}.{minor}.0"


def manifest_of(release: str, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "runtime" / f"ryzenai-{release}.manifest.json"


def default_manifest(repo_root: Path = REPO_ROOT) -> Path:
    return manifest_of(built_release(repo_root), repo_root)


@dataclass(frozen=True)
class ManifestFile:
    name: str
    bytes: int
    sha256: str
    version: str


@dataclass(frozen=True)
class Manifest:
    path: Path
    release: str
    onnxruntime: str
    kind: str
    package: str
    version: str
    directory: str
    files: tuple[ManifestFile, ...]
    #: By kind: `index` and `wheel`, or `archive`, `nupkg` and `nupkg_sha256`.
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def wheel_url(self) -> str:
        return f"{self.detail['index']}{self.package}/{self.detail['wheel']}"

    @property
    def release_text(self) -> str:
        return f"AMD Ryzen AI {self.release} (ONNX Runtime {self.onnxruntime})"


def load_manifest(path: Path) -> Manifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProvisioningError(f"the manifest {path} cannot be read: {error}") from error
    if not isinstance(data, dict) or data.get("format") != MANIFEST_FORMAT:
        found = data.get("format") if isinstance(data, dict) else None
        raise ProvisioningError(
            f"the manifest {path} has format {found!r}, this script reads format {MANIFEST_FORMAT}"
        )
    try:
        source = data["source"]
        kind = str(source["kind"])
        if kind not in SOURCE_KEYS:
            raise ProvisioningError(
                f"the manifest {path} names a source of the kind {kind!r}; this script knows "
                f"{' and '.join(repr(known) for known in SOURCE_KEYS)}"
            )
        files = tuple(
            ManifestFile(
                str(entry["name"]), int(entry["bytes"]), str(entry["sha256"]), str(entry["version"])
            )
            for entry in data["files"]
        )
        manifest = Manifest(
            path=path,
            release=str(data["release"]),
            onnxruntime=str(data["onnxruntime"]),
            kind=kind,
            package=str(source["package"]),
            version=str(source["version"]),
            directory=str(source["directory"]),
            files=files,
            detail={key: str(source[key]) for key in SOURCE_KEYS[kind]},
        )
    except KeyError as error:
        raise ProvisioningError(f"the manifest {path} lacks the key {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise ProvisioningError(f"the manifest {path} is not of the expected shape: {error}") from error
    if not files:
        raise ProvisioningError(f"the manifest {path} names no file")
    for entry in files:
        if Path(entry.name).name != entry.name or entry.name in (".", ".."):
            raise ProvisioningError(f"the manifest {path} names {entry.name!r}, not a plain file name")
    return manifest


def require_kind(manifest: Manifest, kind: str) -> None:
    """The route asked for must be the one the manifest's source is of."""
    if manifest.kind == kind:
        return
    if kind == "nuget":
        raise ProvisioningError(
            f"--nuget-zip was given, but the manifest {manifest.path} ({manifest.release_text}) "
            f"names a source of the kind {manifest.kind!r}: its files come from AMD's wheel, "
            "whose builds differ from the NuGet package's. Name a release whose manifest is "
            "of the kind 'nuget' with --release, or leave --nuget-zip away"
        )
    raise ProvisioningError(
        f"the manifest {manifest.path} ({manifest.release_text}) names a source of the kind "
        f"{manifest.kind!r}: its files come from AMD's NuGet package "
        f"{manifest.detail.get('nupkg')} inside {manifest.detail.get('archive')}, not from a "
        "wheel. Pass --nuget-zip with the path of that ZIP"
    )


def sha256_of_stream(handle: io.BufferedIOBase) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(1 << 20):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_of_stream(handle)[0]


# --- Route `wheel` ------------------------------------------------------------------------


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


# --- Route `nuget` ------------------------------------------------------------------------


@dataclass
class NugetSource:
    """AMD's NuGet package, opened inside the ZIP it was published in."""

    archive: Path
    archive_bytes: int
    archive_sha256: str
    nupkg: str
    nupkg_bytes: int
    nupkg_sha256: str
    package: zipfile.ZipFile

    def member(self, manifest: Manifest, entry: ManifestFile) -> str:
        """The name of a file inside the package, built from the manifest alone."""
        return f"{manifest.directory}/{entry.name}"


def _nuspec_identity(text: str) -> tuple[str | None, str | None]:
    """Package id and version of a `.nuspec`, whatever its XML namespace."""
    root = ElementTree.fromstring(text)
    found: dict[str, str] = {}
    for element in root.iter():
        tag = element.tag.rpartition("}")[2]
        if tag in ("id", "version") and tag not in found and element.text:
            found[tag] = element.text.strip()
    return found.get("id"), found.get("version")


def open_nuget(archive: Path, manifest: Manifest) -> NugetSource:
    """Opens the package of the manifest inside `archive` and requires it to be that
    package: its name, its SHA-256, and the id and version it states."""
    if not archive.is_file():
        raise ProvisioningError(f"the ZIP {archive} does not exist")
    nupkg = manifest.detail["nupkg"]
    try:
        with archive.open("rb") as handle:
            archive_sha256, archive_bytes = sha256_of_stream(handle)
        with zipfile.ZipFile(archive) as outer:
            # Opened by the manifest's name, never by a name read from the archive.
            try:
                data = outer.read(nupkg)
            except KeyError:
                packages = sorted(name for name in outer.namelist() if name.lower().endswith(".nupkg"))
                raise ProvisioningError(
                    f"the ZIP {archive} does not hold {nupkg}, the package of "
                    f"{manifest.release_text}; it holds: {', '.join(packages) or 'no package'}. "
                    "Releases are not mixed"
                ) from None
        found = hashlib.sha256(data).hexdigest()
        if found != manifest.detail["nupkg_sha256"]:
            raise ProvisioningError(
                f"{nupkg} in {archive} has SHA-256 {found}, the manifest has "
                f"{manifest.detail['nupkg_sha256']}: another build of the package"
            )
        package = zipfile.ZipFile(io.BytesIO(data))
        try:
            identity = _nuspec_identity(package.read(f"{manifest.package}.nuspec").decode("utf-8"))
        except KeyError:
            raise ProvisioningError(
                f"{nupkg} in {archive} holds no {manifest.package}.nuspec: it is not the "
                f"package {manifest.package}"
            ) from None
    except (zipfile.BadZipFile, ElementTree.ParseError, UnicodeDecodeError) as error:
        raise ProvisioningError(f"the ZIP {archive} cannot be read as AMD's NuGet ZIP: {error}") from error
    if identity != (manifest.package, manifest.version):
        raise ProvisioningError(
            f"{nupkg} in {archive} states the package {identity[0]} {identity[1]}; "
            f"{manifest.release_text} is {manifest.package} {manifest.version}. Releases are "
            "not mixed"
        )
    return NugetSource(
        archive=archive,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        nupkg=nupkg,
        nupkg_bytes=len(data),
        nupkg_sha256=found,
        package=package,
    )


def verify_nuget(source: NugetSource, manifest: Manifest) -> None:
    """Every file of the manifest is in the package with its size and hash. Names every
    difference; nothing has been written when this is called."""
    problems: list[str] = []
    for entry in manifest.files:
        member = source.member(manifest, entry)
        try:
            info = source.package.getinfo(member)
        except KeyError:
            problems.append(f"{member} is missing in the package")
            continue
        if info.file_size != entry.bytes:
            problems.append(f"{member} has {info.file_size} bytes, the manifest has {entry.bytes}")
            continue
        try:
            with source.package.open(info) as handle:
                found, _ = sha256_of_stream(handle)
        except zipfile.BadZipFile as error:
            problems.append(f"{member} cannot be read: {error}")
            continue
        if found != entry.sha256:
            problems.append(f"{member} has SHA-256 {found}, the manifest has {entry.sha256}")
    if problems:
        raise ProvisioningError(
            f"{source.nupkg} in {source.archive} is not {manifest.release_text} as the manifest "
            "describes it; nothing was copied: " + "; ".join(problems)
        )


def extract_member(source: NugetSource, member: str, partial: Path) -> None:
    """Writes one file of the package to `partial`, a path this script chose."""
    with source.package.open(member) as handle, partial.open("wb") as output:
        shutil.copyfileobj(handle, output, 1 << 20)


# --- Both routes --------------------------------------------------------------------------


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
    """The release a record in the target names, if there is one."""
    try:
        data = json.loads((target / RECEIPT_FILE).read_text(encoding="utf-8"))
        return f"{data['release']} (ONNX Runtime {data['onnxruntime']})"
    except (OSError, ValueError, KeyError, TypeError):
        return None


def place_verified(
    write: Callable[[Path], None], origin: str, destination: Path, entry: ManifestFile
) -> bool:
    """Places one file unless the destination already is it. `write` fills the temporary
    file, which is hashed before it takes the final name. True when it placed the file."""
    if (
        destination.is_file()
        and destination.stat().st_size == entry.bytes
        and file_sha256(destination) == entry.sha256
    ):
        return False
    partial = destination.with_name(f".{destination.name}.partial")
    try:
        write(partial)
        found = file_sha256(partial)
        if found != entry.sha256:
            raise ProvisioningError(
                f"the copy of {origin} has SHA-256 {found}, expected {entry.sha256}"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return True


def write_receipt(target: Path, manifest: Manifest, source: dict[str, object]) -> Path:
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
        "source": source,
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
    partial = path.with_name(f".{RECEIPT_FILE}.{os.getpid()}.partial")
    try:
        partial.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def provision(
    *,
    manifest_path: Path,
    target: Path,
    site_packages: Path | None = None,
    accept_unrecorded_origin: bool = False,
    nuget_zip: Path | None = None,
) -> list[str]:
    """Runs the steps of the module text. Returns the lines of its report."""
    manifest = load_manifest(manifest_path)
    report = [f"release: {manifest.release_text}"]
    writers: list[tuple[Callable[[Path], None], str]]
    if nuget_zip is not None:
        require_kind(manifest, "nuget")
        nuget = open_nuget(nuget_zip, manifest)
        refuse_unusable_target(target, manifest)
        verify_nuget(nuget, manifest)
        source: dict[str, object] = {
            "kind": "nuget",
            "package": manifest.package,
            "version": manifest.version,
            "directory": manifest.directory,
            "archive": {
                "name": nuget.archive.name,
                "path": str(nuget.archive.resolve()),
                "bytes": nuget.archive_bytes,
                "sha256": nuget.archive_sha256,
            },
            "nupkg": {
                "name": nuget.nupkg,
                "bytes": nuget.nupkg_bytes,
                "sha256": nuget.nupkg_sha256,
            },
        }
        report.append(
            f"source: {manifest.package} {manifest.version} ({nuget.nupkg}, {nuget.nupkg_bytes} "
            f"bytes, sha256 {nuget.nupkg_sha256}) in {nuget.archive} ({nuget.archive_bytes} "
            f"bytes, sha256 {nuget.archive_sha256})"
        )
        writers = [
            (
                lambda partial, member=nuget.member(manifest, entry): extract_member(
                    nuget, member, partial
                ),
                f"{nuget.member(manifest, entry)} of {nuget.nupkg}",
            )
            for entry in manifest.files
        ]
    else:
        require_kind(manifest, "wheel")
        if site_packages is None:
            site_packages = Path(sysconfig.get_paths()["purelib"])
        dist_info = find_dist_info(site_packages, manifest)
        origin = check_origin(dist_info, manifest, accept_unrecorded_origin)
        refuse_unusable_target(target, manifest)
        sources = verify_source(site_packages, dist_info, manifest)
        source = {
            "kind": "wheel",
            "index": manifest.detail["index"],
            "package": manifest.package,
            "version": manifest.version,
            "wheel": manifest.detail["wheel"],
            # The URL the installer recorded; null when it recorded none.
            "installed_from": origin,
            "site_packages": str(site_packages),
        }
        report.append(
            f"source: {manifest.package} {manifest.version} in {site_packages}, installed from "
            + (origin or "an origin the installer did not record (accepted on request)")
        )
        writers = [
            (lambda partial, path=path: shutil.copyfile(path, partial), str(path))
            for path in sources
        ]

    before = previous_release(target) if target.is_dir() else None
    if before is not None and before != f"{manifest.release} (ONNX Runtime {manifest.onnxruntime})":
        report.append(f"the target was provisioned for {before}; its files are replaced")
    target.mkdir(parents=True, exist_ok=True)
    for (write, origin_name), entry in zip(writers, manifest.files, strict=True):
        placed = place_verified(write, origin_name, target / entry.name, entry)
        report.append(
            f"{'copied ' if placed else 'present'} {entry.name}  {entry.bytes} bytes  "
            f"sha256 {entry.sha256}"
        )
    receipt = write_receipt(target, manifest, source)
    total = sum(entry.bytes for entry in manifest.files)
    report.append(f"{len(manifest.files)} files, {total} bytes in {target}; record: {receipt}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Place the AMD runtime files of l2d in the runtime directory, from AMD's "
        "wheel or from AMD's NuGet ZIP."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="runtime directory to fill (default: runtime/ryzenai of this checkout)",
    )
    chosen = parser.add_mutually_exclusive_group()
    chosen.add_argument(
        "--release",
        help="release to provision, for example 1.8.0: its manifest is "
        "runtime/ryzenai-<release>.manifest.json (default: the release the tool is built for)",
    )
    chosen.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest to provision, in place of --release",
    )
    parser.add_argument(
        "--nuget-zip",
        type=Path,
        default=None,
        help="AMD's NuGet ZIP (for example ryzen_ai_nuget_1.8.0.zip), read where it is; "
        "for a release whose manifest names a source of the kind 'nuget'",
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=None,
        help="wheel route: package directory that holds AMD's wheel (default: the running "
        "environment's)",
    )
    parser.add_argument(
        "--accept-unrecorded-origin",
        action="store_true",
        help="wheel route: go on when the installer did not record the wheel's URL; the "
        "hashes still decide",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.manifest is not None:
            manifest_path = arguments.manifest
        elif arguments.release is not None:
            manifest_path = manifest_of(arguments.release)
        else:
            manifest_path = default_manifest()
        report = provision(
            manifest_path=manifest_path,
            target=arguments.target,
            site_packages=arguments.site_packages,
            accept_unrecorded_origin=arguments.accept_unrecorded_origin,
            nuget_zip=arguments.nuget_zip,
        )
    except ProvisioningError as error:
        print(f"provision: refused: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        # A copy or a write that failed (disk full, a file in use, no permission). Files
        # already placed are whole, since each takes its name only after it was hashed;
        # the record is written last, so the target has none for an unfinished run.
        print(f"provision: failed: {error}", file=sys.stderr)
        return 2
    for line in report:
        print(f"provision: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
