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
        self.assert_refused("'wheel'", "--nuget-zip was given")
        self.assertFalse(self.target.exists())

    def test_the_command_line_takes_the_release_and_refuses_the_wrong_zip(self) -> None:
        # The default manifest is the built release's (1.9.0, a wheel).
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = provision.main(["--nuget-zip", str(self.archive), "--target", str(self.target)])
        self.assertEqual(code, 1)
        self.assertIn("1.9.0", stderr.getvalue())
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


if __name__ == "__main__":
    unittest.main()
