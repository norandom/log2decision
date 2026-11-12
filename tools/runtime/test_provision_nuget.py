"""Tests of the NuGet ZIP route of the provisioning step, on a small synthetic ZIP built
in a temporary directory (standard library only).

    uv run --no-project --python 3.12 python -m unittest discover -s tools/runtime -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import provision

PACKAGE = "RuntimePackage"
NUPKG = "RuntimePackage.2.0.0.nupkg"
DIRECTORY = "runtimes/win-x64/native"
FILES = {
    "onnxruntime.dll": b"runtime of release two",
    "provider.dll": b"the provider, a longer file",
}
ALTERED = b"the provider, a LONGER file"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def nuspec(package_id: str, version: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">\n'
        f"  <metadata><id>{package_id}</id><version>{version}</version>"
        "<authors>Vendor</authors></metadata>\n</package>\n"
    ).encode()


def build_nupkg(
    *,
    package_id: str = PACKAGE,
    version: str = "2.0.0",
    files: dict[str, bytes] | None = None,
    extra: dict[str, bytes] | None = None,
) -> bytes:
    """A package as AMD's: a `.nuspec` at the root, the files under `runtimes/...`."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(f"{PACKAGE}.nuspec", nuspec(package_id, version))
        for name, data in (FILES if files is None else files).items():
            package.writestr(f"{DIRECTORY}/{name}", data)
        for name, data in (extra or {}).items():
            package.writestr(name, data)
    return buffer.getvalue()


class NugetScratch(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="l2d-provision-test-")
        self.addCleanup(self._temporary.cleanup)
        # One level down, so that a file written one level up would still be seen.
        self.root = Path(self._temporary.name) / "work"
        self.archive = self.root / "downloads" / "runtime_nuget_2.0.0 (1).zip"
        self.archive.parent.mkdir(parents=True)
        self.manifest_path = self.root / "manifest.json"
        self.target = self.root / "runtime" / "ryzenai"
        self.write_zip(build_nupkg())

    def write_zip(
        self,
        nupkg: bytes,
        *,
        name: str = NUPKG,
        outer_extra: dict[str, bytes] | None = None,
        manifest_hash: str | None = None,
        manifest_bytes: int | None = None,
    ) -> None:
        """The ZIP with the package inside, and a manifest that describes it."""
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_STORED) as outer:
            outer.writestr("Other.Managed.1.0.0.nupkg", b"not looked at")
            outer.writestr(name, nupkg)
            for extra_name, data in (outer_extra or {}).items():
                outer.writestr(extra_name, data)
        manifest = {
            "format": 1,
            "release": "2.0.0",
            "onnxruntime": "1.30.0",
            "source": {
                "kind": "nuget",
                "archive": "runtime_nuget_2.0.0.zip",
                "nupkg": NUPKG,
                "nupkg_bytes": str(len(nupkg) if manifest_bytes is None else manifest_bytes),
                "nupkg_sha256": manifest_hash or sha256(nupkg),
                "package": PACKAGE,
                "version": "2.0.0",
                "directory": DIRECTORY,
            },
            "files": [
                {"name": name, "bytes": len(data), "sha256": sha256(data), "version": "2.0.0"}
                for name, data in FILES.items()
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.nupkg = nupkg

    def run_provision(self, **overrides: object) -> list[str]:
        arguments: dict[str, object] = {
            "manifest_path": self.manifest_path,
            "target": self.target,
            "nuget_zip": self.archive,
        }
        arguments.update(overrides)
        return provision.provision(**arguments)  # type: ignore[arg-type]

    def assert_refused(self, *fragments: str, **overrides: object) -> str:
        with self.assertRaises(provision.ProvisioningError) as caught:
            self.run_provision(**overrides)
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    def every_file_of_the_scratch(self) -> list[str]:
        top = self.root.parent
        return sorted(path.relative_to(top).as_posix() for path in top.rglob("*") if path.is_file())


class NugetTests(NugetScratch):
    def test_the_files_are_placed_and_the_record_names_the_zip(self) -> None:
        archive_before = self.archive.read_bytes()
        report = self.run_provision()
        for name, data in FILES.items():
            self.assertEqual((self.target / name).read_bytes(), data)
        self.assertEqual(sum(line.startswith("copied") for line in report), 2)
        # The ZIP was read where it is and is unchanged.
        self.assertEqual(self.archive.read_bytes(), archive_before)

        receipt = json.loads((self.target / provision.RECEIPT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(receipt["format"], 1)
        self.assertEqual(receipt["release"], "2.0.0")
        self.assertEqual(receipt["onnxruntime"], "1.30.0")
        self.assertEqual(receipt["written_by"], "provisioning")
        source = receipt["source"]
        self.assertEqual(source["kind"], "nuget")
        self.assertEqual(source["package"], PACKAGE)
        self.assertEqual(source["version"], "2.0.0")
        self.assertEqual(source["directory"], DIRECTORY)
        self.assertEqual(
            source["archive"],
            {
                "name": "runtime_nuget_2.0.0 (1).zip",
                "path": str(self.archive.resolve()),
                "bytes": len(archive_before),
                "sha256": sha256(archive_before),
            },
        )
        self.assertEqual(
            source["nupkg"],
            {"name": NUPKG, "bytes": len(self.nupkg), "sha256": sha256(self.nupkg)},
        )
        self.assertNotIn("installed_from", source)
        self.assertEqual(
            [(entry["name"], entry["bytes"], entry["sha256"]) for entry in receipt["files"]],
            [(name, len(data), sha256(data)) for name, data in FILES.items()],
        )
        for entry in receipt["files"]:
            self.assertEqual(
                entry["modified_unix_ns"], (self.target / entry["name"]).stat().st_mtime_ns
            )
        self.assertEqual(
            sorted(path.name for path in self.target.iterdir()),
            sorted([*FILES, provision.RECEIPT_FILE]),
        )

    def test_a_second_run_copies_nothing(self) -> None:
        self.run_provision()
        report = self.run_provision()
        self.assertEqual(sum(line.startswith("present") for line in report), 2)
        self.assertEqual(sum(line.startswith("copied") for line in report), 0)

    def test_a_package_with_another_id_is_refused_with_both(self) -> None:
        self.write_zip(build_nupkg(package_id="SomethingElse"))
        self.assert_refused("SomethingElse 2.0.0", f"{PACKAGE} 2.0.0", "not mixed")
        self.assertFalse(self.target.exists())

    def test_a_package_of_another_version_is_refused_with_both(self) -> None:
        self.write_zip(build_nupkg(version="1.9.0"))
        self.assert_refused(f"{PACKAGE} 1.9.0", f"{PACKAGE} 2.0.0", "not mixed")
        self.assertFalse(self.target.exists())

    def test_a_zip_of_another_release_is_refused_naming_what_it_holds(self) -> None:
        self.write_zip(build_nupkg(version="1.9.0"), name="RuntimePackage.1.9.0.nupkg")
        self.assert_refused(NUPKG, "RuntimePackage.1.9.0.nupkg", "not mixed")
        self.assertFalse(self.target.exists())

    def test_another_build_of_the_package_is_refused_with_both_hashes(self) -> None:
        self.write_zip(build_nupkg(), manifest_hash="0" * 64)
        self.assert_refused(sha256(self.nupkg), "0" * 64)
        self.assertFalse(self.target.exists())

    def test_a_package_without_its_nuspec_is_refused(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as package:
            for name, data in FILES.items():
                package.writestr(f"{DIRECTORY}/{name}", data)
        self.write_zip(buffer.getvalue())
        self.assert_refused(f"{PACKAGE}.nuspec")
        self.assertFalse(self.target.exists())

    def test_what_is_not_a_zip_is_refused(self) -> None:
        self.archive.write_bytes(b"not an archive")
        self.assert_refused("cannot be read")
        self.assert_refused("does not exist", nuget_zip=self.root / "absent.zip")
        self.assertFalse(self.target.exists())

    def test_members_with_paths_that_point_elsewhere_are_never_written(self) -> None:
        # In the package and in the ZIP: members whose names climb out of the directory,
        # are absolute, or put a required name under another directory.
        first = next(iter(FILES))
        climbing = "/".join([DIRECTORY, "..", "..", "..", "climbed.dll"])
        escaping = {
            climbing: b"escaped",
            "../climbed-too.dll": b"escaped",
            "//absolute.dll": b"escaped",
            "C:/drive.dll": b"escaped",
            f"elsewhere/{first}": b"the right name in the wrong place",
            f"../{first}": b"the right name one level up",
        }
        self.write_zip(
            build_nupkg(extra=escaping),
            outer_extra={"../outside.nupkg": b"escaped", f"../{NUPKG}": b"escaped"},
        )
        before = self.every_file_of_the_scratch()
        self.run_provision()
        # Exactly the manifest's files and the record were written, all in the target.
        added = sorted(set(self.every_file_of_the_scratch()) - set(before))
        self.assertEqual(
            added,
            sorted(f"work/runtime/ryzenai/{name}" for name in [*FILES, provision.RECEIPT_FILE]),
        )
        for name, data in FILES.items():
            self.assertEqual((self.target / name).read_bytes(), data)

    def test_a_manifest_that_names_a_path_is_refused(self) -> None:
        for name in ("../onnxruntime.dll", "sub/onnxruntime.dll", "..", "C:\\onnxruntime.dll"):
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            data["files"][0]["name"] = name
            self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
            self.assert_refused("not a plain file name")
        self.assertFalse(self.target.exists())

    def test_a_file_missing_in_the_package_is_named_and_nothing_is_copied(self) -> None:
        partial = dict(FILES)
        del partial["provider.dll"]
        self.write_zip(build_nupkg(files=partial))
        message = self.assert_refused(f"{DIRECTORY}/provider.dll is missing in the package")
        self.assertIn("nothing was copied", message)
        self.assertFalse(self.target.exists())

    def test_a_file_with_another_hash_is_refused_before_anything_is_copied(self) -> None:
        self.assertEqual(len(ALTERED), len(FILES["provider.dll"]))
        self.write_zip(build_nupkg(files=dict(FILES, **{"provider.dll": ALTERED})))
        message = self.assert_refused(
            f"{DIRECTORY}/provider.dll", sha256(ALTERED), sha256(FILES["provider.dll"])
        )
        self.assertIn("nothing was copied", message)
        self.assertFalse(self.target.exists())

        self.write_zip(build_nupkg(files=dict(FILES, **{"provider.dll": b"short"})))
        self.assert_refused("has 5 bytes, the manifest has 27")
        self.assertFalse(self.target.exists())

    def test_an_interrupted_copy_leaves_no_half_file_and_no_record(self) -> None:
        real = provision.extract_member
        calls: list[str] = []

        def interrupted(source: object, member: str, partial: Path) -> None:
            calls.append(member)
            if len(calls) == 2:
                partial.write_bytes(b"half")
                raise OSError(28, "No space left on device")
            real(source, member, partial)  # type: ignore[arg-type]

        arguments = [
            "--manifest", str(self.manifest_path),
            "--nuget-zip", str(self.archive),
            "--target", str(self.target),
        ]
        stderr = io.StringIO()
        with (
            mock.patch.object(provision, "extract_member", interrupted),
            contextlib.redirect_stderr(stderr),
        ):
            code = provision.main(arguments)
        self.assertEqual(code, 2)
        self.assertTrue(stderr.getvalue().startswith("provision: failed: "), stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        # The first file is whole, the second is absent, and there is no record.
        first = next(iter(FILES))
        self.assertEqual([path.name for path in self.target.iterdir()], [first])
        self.assertEqual((self.target / first).read_bytes(), FILES[first])

        # The next run finishes the work.
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(provision.main(arguments), 0)
        self.assertIn(f"present {first}", stdout.getvalue())
        self.assertIn("copied  provider.dll", stdout.getvalue())
        self.assertTrue((self.target / provision.RECEIPT_FILE).is_file())

    def test_a_copy_that_does_not_hash_right_is_not_given_its_name(self) -> None:
        def corrupting(source: object, member: str, partial: Path) -> None:
            partial.write_bytes(b"x" * len(FILES["onnxruntime.dll"]))

        with mock.patch.object(provision, "extract_member", corrupting):
            self.assert_refused("the copy of", sha256(FILES["onnxruntime.dll"]))
        self.assertEqual(list(self.target.iterdir()), [])

    def test_a_target_with_a_dll_of_another_release_is_refused(self) -> None:
        self.target.mkdir(parents=True)
        (self.target / "aiecompiler_client.dll").write_bytes(b"from another release")
        self.assert_refused("aiecompiler_client.dll", "not mixed", "Nothing was copied")
        self.assertEqual([path.name for path in self.target.iterdir()], ["aiecompiler_client.dll"])

    def test_each_route_refuses_the_other_kind_of_manifest(self) -> None:
        # A manifest of the kind `nuget` without the ZIP: the wheel route refuses it.
        message = self.assert_refused("'nuget'", "--nuget-zip", NUPKG, nuget_zip=None)
        self.assertIn("runtime_nuget_2.0.0.zip", message)
        # A manifest of the kind `wheel` with a ZIP: the ZIP route refuses it.
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        data["source"] = {
            "kind": "wheel",
            "index": "https://example.invalid/2.0.0/simple/",
            "package": "runtime-package",
            "version": "1.30.0",
            "wheel": "runtime_package-1.30.0-py3-none-win_amd64.whl",
            "directory": "onnxruntime/capi",
        }
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        self.assert_refused("'wheel'", "a NuGet package was given")
        self.assertFalse(self.target.exists())

    def test_the_command_line_takes_the_release_and_refuses_the_wrong_zip(self) -> None:
        # Without --release the built release is meant (1.9.0), and with a package given
        # its manifest of the kind `nuget`; this ZIP is not AMD's 1.9.0 either.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = provision.main(["--nuget-zip", str(self.archive), "--target", str(self.target)])
        self.assertEqual(code, 1)
        self.assertIn("RyzenAI_Deployment.1.9.0.nupkg", stderr.getvalue())
        # A wheel manifest named outright is refused with a package.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = provision.main(
                [
                    "--manifest", str(provision.manifest_of("1.9.0", "wheel")),
                    "--nupkg", str(self.archive),
                    "--target", str(self.target),
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("'wheel'", stderr.getvalue())
        # --release picks the manifest of that release; this ZIP is not AMD's 1.8.0.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = provision.main(
                ["--release", "1.8.0", "--nuget-zip", str(self.archive), "--target", str(self.target)]
            )
        self.assertEqual(code, 1)
        self.assertIn("RyzenAI_Deployment.1.8.0.nupkg", stderr.getvalue())
        self.assertFalse(self.target.exists())


class BarePackageTests(NugetScratch):
    """The package file itself, without a ZIP around it (`--nupkg`)."""

    def setUp(self) -> None:
        super().setUp()
        self.package = self.root / "downloads" / "RuntimePackage.2.0.0 (1).nupkg"
        self.package.write_bytes(self.nupkg)

    def bare(self, **overrides: object) -> list[str]:
        return self.run_provision(nuget_zip=None, nupkg=self.package, **overrides)

    def test_the_files_are_placed_and_the_record_names_the_package(self) -> None:
        before = self.package.read_bytes()
        report = self.bare()
        for name, data in FILES.items():
            self.assertEqual((self.target / name).read_bytes(), data)
        self.assertEqual(sum(line.startswith("copied") for line in report), 2)
        self.assertEqual(self.package.read_bytes(), before)

        receipt = json.loads((self.target / provision.RECEIPT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(receipt["written_by"], "provisioning")
        source = receipt["source"]
        self.assertEqual(source["kind"], "nuget")
        self.assertEqual(source["package"], PACKAGE)
        self.assertEqual(source["version"], "2.0.0")
        self.assertEqual(
            source["nupkg"],
            {
                "name": NUPKG,
                "bytes": len(before),
                "sha256": sha256(before),
                "file": "RuntimePackage.2.0.0 (1).nupkg",
                "path": str(self.package.resolve()),
            },
        )
        self.assertNotIn("archive", source)
        self.assertEqual(
            [(entry["name"], entry["sha256"]) for entry in receipt["files"]],
            [(name, sha256(data)) for name, data in FILES.items()],
        )
        self.assertEqual(
            sorted(path.name for path in self.target.iterdir()),
            sorted([*FILES, provision.RECEIPT_FILE]),
        )
        # A second run copies nothing; the same record comes from the ZIP route too.
        self.assertEqual(sum(line.startswith("present") for line in self.bare()), 2)
        self.assertEqual(sum(line.startswith("present") for line in self.run_provision()), 2)

    def test_a_package_of_another_size_is_refused_before_it_is_read(self) -> None:
        self.package.write_bytes(self.nupkg + b"more")
        opened: list[object] = []
        real_open = Path.open

        def watching(path: Path, *arguments: object, **keywords: object):
            if path == self.package:
                opened.append(path)
            return real_open(path, *arguments, **keywords)  # type: ignore[arg-type]

        with mock.patch.object(Path, "open", watching):
            message = self.assert_refused(
                f"{len(self.nupkg) + 4} bytes", f"the manifest has {len(self.nupkg)}",
                "It was not read", nuget_zip=None, nupkg=self.package,
            )
        self.assertIn(str(self.package), message)
        self.assertEqual(opened, [], "the package was opened although its size is wrong")
        self.assertFalse(self.target.exists())

    def test_a_package_in_a_zip_of_another_size_is_refused_before_it_is_read(self) -> None:
        self.write_zip(self.nupkg, manifest_bytes=len(self.nupkg) - 1)
        with mock.patch.object(provision.tempfile, "TemporaryFile") as spooled:
            self.assert_refused(f"the manifest has {len(self.nupkg) - 1}", "It was not read")
        spooled.assert_not_called()
        self.assertFalse(self.target.exists())

    def test_a_package_with_another_hash_or_identity_is_refused(self) -> None:
        other = build_nupkg(version="1.9.0")
        self.package.write_bytes(other)
        self.write_zip(other)  # the manifest now describes this package: hash and size
        self.assert_refused(
            f"{PACKAGE} 1.9.0", f"{PACKAGE} 2.0.0", "not mixed", nuget_zip=None, nupkg=self.package
        )
        # Same size, other bytes: the hash decides.
        flipped = bytearray(self.nupkg)
        flipped[-1] ^= 0xFF
        self.package.write_bytes(bytes(flipped))
        self.write_zip(self.nupkg)
        self.assert_refused(sha256(bytes(flipped)), sha256(self.nupkg), nuget_zip=None, nupkg=self.package)
        self.assertFalse(self.target.exists())

    def test_both_a_package_and_a_zip_are_refused(self) -> None:
        self.assert_refused("name one source", nupkg=self.package)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            provision.main(["--nupkg", str(self.package), "--nuget-zip", str(self.archive)])

    def test_a_manifest_without_the_packages_size_is_refused(self) -> None:
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        del data["source"]["nupkg_bytes"]
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        self.assert_refused("lacks the key 'nupkg_bytes'", nuget_zip=None, nupkg=self.package)
        data["source"]["nupkg_bytes"] = "0"
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        self.assert_refused("nupkg_bytes", nuget_zip=None, nupkg=self.package)


class NamesAndTargetsTests(NugetScratch):
    def test_names_that_are_not_plainly_a_file_are_refused(self) -> None:
        """With `second.dll:stream` the script once wrote into a stream of another file;
        `CON.dll` was placed as a file. Both are refused now, before anything is read."""
        for name in (
            "second.dll:stream",
            "onnxruntime.dll::$DATA",
            "C:onnxruntime.dll",
            "trailing.dll.",
            "trailing.dll ",
            "CON",
            "CON.dll",
            "nul.dll",
            "COM1.dll",
            "lpt9.txt.dll",
            "what?.dll",
            "star*.dll",
            "tab\t.dll",
            "sub/onnxruntime.dll",
            "..",
            "",
        ):
            self.assertFalse(provision.is_plain_file_name(name), name)
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            data["files"][0]["name"] = name
            self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
            self.assert_refused("not a plain file name")
            self.assertFalse(self.target.exists(), name)
        for name in ("onnxruntime.dll", "onnxruntime-genai.dll", "a b.dll", "CONSOLE.dll", "x.y.z"):
            self.assertTrue(provision.is_plain_file_name(name), name)
        # Nothing was written anywhere, not even beside the target.
        self.assertEqual(
            self.every_file_of_the_scratch(),
            ["work/downloads/runtime_nuget_2.0.0 (1).zip", "work/manifest.json"],
        )

    def test_a_package_name_in_the_manifest_must_be_plain_too(self) -> None:
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        data["source"]["nupkg"] = "../RuntimePackage.2.0.0.nupkg"
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        self.assert_refused("not a plain file name")

    def test_the_directory_of_the_package_or_zip_is_refused_as_target(self) -> None:
        self.assert_refused("is the directory that holds", target=self.archive.parent)
        package = self.archive.parent / NUPKG
        package.write_bytes(self.nupkg)
        self.assert_refused(
            "is the directory that holds", NUPKG,
            nuget_zip=None, nupkg=package, target=self.archive.parent,
        )
        self.assertEqual(
            sorted(path.name for path in self.archive.parent.iterdir()),
            [NUPKG, "runtime_nuget_2.0.0 (1).zip"],
        )

    def test_a_target_in_the_tracked_tree_is_refused_except_the_runtime_directory(self) -> None:
        repository = self.root / "repository"
        for inside in (
            repository,
            repository / "crates" / "l2d-encoder",
            repository / "runtime",
            repository / "runtime" / "ryzenai" / "deeper",
            repository / "runtime" / "cache",
        ):
            self.assert_refused("lies inside the repository", target=inside, repo_root=repository)
            self.assertFalse(inside.exists(), inside)
        allowed = repository / "runtime" / "ryzenai"
        report = self.run_provision(target=allowed, repo_root=repository)
        self.assertEqual(sum(line.startswith("copied") for line in report), 2)
        # The real repository: its own tree is refused, from the command line too.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = provision.main(
                [
                    "--manifest", str(self.manifest_path),
                    "--nuget-zip", str(self.archive),
                    "--target", str(provision.REPO_ROOT / "tools" / "runtime" / "never"),
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("lies inside the repository", stderr.getvalue())
        self.assertFalse((provision.REPO_ROOT / "tools" / "runtime" / "never").exists())

    def test_an_archive_that_fails_while_placing_ends_with_one_line(self) -> None:
        calls: list[str] = []
        real = provision.extract_member

        def failing(source: object, member: str, partial: Path) -> None:
            calls.append(member)
            if len(calls) == 2:
                partial.write_bytes(b"half")
                raise zipfile.BadZipFile("Bad CRC-32 for file 'provider.dll'")
            real(source, member, partial)  # type: ignore[arg-type]

        stderr = io.StringIO()
        with (
            mock.patch.object(provision, "extract_member", failing),
            contextlib.redirect_stderr(stderr),
        ):
            code = provision.main(
                [
                    "--manifest", str(self.manifest_path),
                    "--nuget-zip", str(self.archive),
                    "--target", str(self.target),
                ]
            )
        self.assertEqual(code, 2)
        self.assertTrue(stderr.getvalue().startswith("provision: failed: "), stderr.getvalue())
        self.assertIn("Bad CRC-32", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(stderr.getvalue().count("\n"), 1)
        first = next(iter(FILES))
        self.assertEqual([path.name for path in self.target.iterdir()], [first])


if __name__ == "__main__":
    unittest.main()

