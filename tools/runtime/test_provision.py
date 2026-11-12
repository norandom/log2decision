"""Tests of the provisioning step against temporary directories (standard library only).

    uv run --no-project --python 3.12 python -m unittest discover -s tools/runtime -v
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import provision

INDEX = "https://example.invalid/2.0.0/simple/"
WHEEL = "runtime_package-1.30.0-py3-none-win_amd64.whl"
FILES = {
    "onnxruntime.dll": b"runtime of release two",
    "provider.dll": b"the provider, a longer file",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_hash(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


class Scratch(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="l2d-provision-test-")
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.manifest_path = self.root / "manifest.json"
        self.site_packages = self.root / "env" / "Lib" / "site-packages"
        self.target = self.root / "runtime" / "ryzenai"
        self.write_manifest()
        self.install()

    def write_manifest(self, release: str = "2.0.0", version: str = "1.30.0") -> None:
        manifest = {
            "format": 1,
            "release": release,
            "onnxruntime": "1.30.0",
            "source": {
                "kind": "wheel",
                "index": INDEX,
                "package": "runtime-package",
                "version": version,
                "wheel": WHEEL,
                "directory": "onnxruntime/capi",
            },
            "files": [
                {"name": name, "bytes": len(data), "sha256": sha256(data), "version": "2.0.0"}
                for name, data in FILES.items()
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def install(
        self,
        version: str = "1.30.0",
        url: str | None = f"{INDEX}runtime-package/{WHEEL}",
        files: dict[str, bytes] | None = None,
    ) -> None:
        """A package directory as an installer leaves it."""
        files = FILES if files is None else files
        capi = self.site_packages / "onnxruntime" / "capi"
        capi.mkdir(parents=True, exist_ok=True)
        dist_info = self.site_packages / f"runtime_package-{version}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        record = []
        for name, data in files.items():
            (capi / name).write_bytes(data)
            record.append(f"onnxruntime/capi/{name},sha256={record_hash(data)},{len(data)}")
        record.append(f"runtime_package-{version}.dist-info/RECORD,,")
        (dist_info / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")
        if url is not None:
            (dist_info / "direct_url.json").write_text(json.dumps({"url": url}), encoding="utf-8")

    def run_provision(self, **overrides: object) -> list[str]:
        arguments: dict[str, object] = {
            "manifest_path": self.manifest_path,
            "site_packages": self.site_packages,
            "target": self.target,
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


class ProvisionTests(Scratch):
    def test_the_files_are_copied_and_recorded(self) -> None:
        report = self.run_provision()
        for name, data in FILES.items():
            self.assertEqual((self.target / name).read_bytes(), data)
        self.assertEqual(sum(line.startswith("copied") for line in report), 2)

        receipt = json.loads((self.target / provision.RECEIPT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(receipt["format"], 1)
        self.assertEqual(receipt["release"], "2.0.0")
        self.assertEqual(receipt["onnxruntime"], "1.30.0")
        self.assertEqual(receipt["source"]["installed_from"], f"{INDEX}runtime-package/{WHEEL}")
        self.assertEqual(receipt["source"]["version"], "1.30.0")
        self.assertEqual(
            [(entry["name"], entry["bytes"], entry["sha256"]) for entry in receipt["files"]],
            [(name, len(data), sha256(data)) for name, data in FILES.items()],
        )
        for entry in receipt["files"]:
            self.assertEqual(
                entry["modified_unix_ns"], (self.target / entry["name"]).stat().st_mtime_ns
            )
        # No temporary file is left behind.
        self.assertEqual(
            sorted(path.name for path in self.target.iterdir()),
            sorted([*FILES, provision.RECEIPT_FILE]),
        )

    def test_a_second_run_copies_nothing(self) -> None:
        self.run_provision()
        before = {name: (self.target / name).stat().st_mtime_ns for name in FILES}
        report = self.run_provision()
        self.assertEqual(sum(line.startswith("copied") for line in report), 0)
        self.assertEqual(sum(line.startswith("present") for line in report), 2)
        self.assertEqual(before, {name: (self.target / name).stat().st_mtime_ns for name in FILES})

    def test_a_damaged_file_in_the_target_is_replaced(self) -> None:
        self.run_provision()
        (self.target / "provider.dll").write_bytes(b"the provider, a LONGER file")
        report = self.run_provision()
        self.assertEqual(sum(line.startswith("copied") for line in report), 1)
        self.assertEqual((self.target / "provider.dll").read_bytes(), FILES["provider.dll"])

    def test_a_package_of_another_version_is_refused_with_both_versions(self) -> None:
        self.write_manifest(release="2.1.0", version="1.31.0")
        self.assert_refused("1.30.0", "1.31.0", "2.1.0", "not mixed")
        self.assertFalse(self.target.exists())

    def test_a_package_that_is_not_installed_is_refused(self) -> None:
        self.site_packages = self.root / "empty"
        self.site_packages.mkdir()
        self.assert_refused("runtime-package is not installed", "uv sync")

    def test_a_package_from_another_index_is_refused(self) -> None:
        self.install(url=f"https://example.invalid/1.8.0/simple/runtime-package/{WHEEL}")
        self.assert_refused("/1.8.0/", "/2.0.0/")
        self.assertFalse(self.target.exists())

    def test_an_unrecorded_origin_is_refused_unless_accepted(self) -> None:
        (self.site_packages / "runtime_package-1.30.0.dist-info" / "direct_url.json").unlink()
        self.assert_refused("direct_url.json", "--accept-unrecorded-origin")
        self.assertFalse(self.target.exists())

        report = self.run_provision(accept_unrecorded_origin=True)
        self.assertTrue(any("did not record" in line for line in report))
        receipt = json.loads((self.target / provision.RECEIPT_FILE).read_text(encoding="utf-8"))
        self.assertIsNone(receipt["source"]["installed_from"])

    def test_a_changed_source_file_is_refused_before_anything_is_copied(self) -> None:
        altered = b"the provider, a LONGER file"
        (self.site_packages / "onnxruntime" / "capi" / "provider.dll").write_bytes(altered)
        message = self.assert_refused("provider.dll", sha256(altered), sha256(FILES["provider.dll"]))
        self.assertIn("nothing was copied", message)
        self.assertFalse(self.target.exists())

    def test_a_wheel_whose_record_differs_from_the_manifest_is_refused(self) -> None:
        other = dict(FILES, **{"provider.dll": b"a provider of another build"})
        self.install(files=other)
        self.assert_refused("RECORD", "onnxruntime/capi/provider.dll", sha256(other["provider.dll"]))
        self.assertFalse(self.target.exists())

    def test_a_missing_source_file_is_named(self) -> None:
        (self.site_packages / "onnxruntime" / "capi" / "onnxruntime.dll").unlink()
        self.assert_refused("onnxruntime.dll is missing")

    def test_a_target_with_a_dll_of_another_release_is_refused(self) -> None:
        self.target.mkdir(parents=True)
        (self.target / "Old_Provider.DLL").write_bytes(b"from another release")
        (self.target / "notes.txt").write_text("kept", encoding="utf-8")
        self.assert_refused("Old_Provider.DLL", "not mixed", "Nothing was copied")
        self.assertEqual(
            sorted(path.name for path in self.target.iterdir()), ["Old_Provider.DLL", "notes.txt"]
        )

    def test_a_target_provisioned_for_another_release_is_replaced_and_said_so(self) -> None:
        self.target.mkdir(parents=True)
        (self.target / "onnxruntime.dll").write_bytes(b"the runtime of release one")
        (self.target / provision.RECEIPT_FILE).write_text(
            json.dumps({"format": 1, "release": "1.8.0", "onnxruntime": "1.27.0", "files": []}),
            encoding="utf-8",
        )
        report = self.run_provision()
        self.assertTrue(any("1.8.0 (ONNX Runtime 1.27.0)" in line for line in report))
        receipt = json.loads((self.target / provision.RECEIPT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(receipt["release"], "2.0.0")
        self.assertEqual((self.target / "onnxruntime.dll").read_bytes(), FILES["onnxruntime.dll"])

    def test_a_drive_root_is_refused_as_target(self) -> None:
        self.assert_refused("drive root", target=Path(self.root.anchor))

    def test_a_manifest_of_another_format_is_refused(self) -> None:
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        data["format"] = 2
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        self.assert_refused("format 2")

    def test_a_manifest_that_lacks_a_key_is_refused_by_name(self) -> None:
        for key in ("release", "source", "files"):
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            del data[key]
            self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
            self.assert_refused("lacks the key", repr(key))
            self.write_manifest()
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        del data["source"]["wheel"]
        del data["files"][0]["sha256"]
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        self.assert_refused("lacks the key")
        data["files"] = [{"name": "a.dll", "bytes": "many", "sha256": "0", "version": "1"}]
        data["source"]["wheel"] = WHEEL
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        self.assert_refused("not of the expected shape")
        # From the command line: one line and exit code 1, no traceback.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = provision.main(["--manifest", str(self.manifest_path)])
        self.assertEqual(code, 1)
        self.assertTrue(stderr.getvalue().startswith("provision: refused: "))
        self.assertEqual(stderr.getvalue().count("\n"), 1)

    def test_a_failed_copy_ends_with_one_line_and_exit_code_2(self) -> None:
        def failing_copy(source: object, destination: object) -> None:
            Path(str(destination)).write_bytes(b"half")
            raise OSError(28, "No space left on device")

        stderr = io.StringIO()
        with (
            mock.patch.object(provision.shutil, "copyfile", failing_copy),
            contextlib.redirect_stderr(stderr),
        ):
            code = provision.main(
                [
                    "--manifest", str(self.manifest_path),
                    "--site-packages", str(self.site_packages),
                    "--target", str(self.target),
                ]
            )
        self.assertEqual(code, 2)
        self.assertTrue(stderr.getvalue().startswith("provision: failed: "), stderr.getvalue())
        self.assertIn("No space left", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        # Neither a half file nor a record is left.
        self.assertEqual(list(self.target.iterdir()), [])

    def test_the_record_says_it_was_written_by_provisioning(self) -> None:
        self.run_provision()
        receipt = json.loads((self.target / provision.RECEIPT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(receipt["written_by"], "provisioning")

    def test_the_command_line_reports_a_refusal_with_exit_code_1(self) -> None:
        code = provision.main(
            [
                "--manifest", str(self.manifest_path),
                "--site-packages", str(self.root / "nowhere"),
                "--target", str(self.target),
            ]
        )
        self.assertEqual(code, 1)
        self.assertEqual(
            provision.main(
                [
                    "--manifest", str(self.manifest_path),
                    "--site-packages", str(self.site_packages),
                    "--target", str(self.target),
                ]
            ),
            0,
        )


class PinningTests(unittest.TestCase):
    """The environment pins exactly the wheel the tracked manifest names."""

    def test_the_default_manifest_is_that_of_the_built_release(self) -> None:
        cargo = tomllib.loads(
            (provision.REPO_ROOT / "crates" / "l2d-encoder" / "Cargo.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(cargo["features"]["default"], ["ryzenai-1-9"])
        self.assertEqual(provision.built_release(), "1.9.0")
        self.assertEqual(provision.default_manifest().name, "ryzenai-1.9.0.manifest.json")

    def test_a_manifest_whose_source_is_not_a_wheel_is_refused(self) -> None:
        other = provision.REPO_ROOT / "runtime" / "ryzenai-1.8.0.manifest.json"
        with self.assertRaises(provision.ProvisioningError) as caught:
            provision.load_manifest(other)
        self.assertIn("'nuget'", str(caught.exception))
        self.assertIn("1.8.0", str(caught.exception))

    def test_pyproject_pins_the_wheel_of_the_tracked_manifest(self) -> None:
        manifest = provision.load_manifest(provision.default_manifest())
        project = tomllib.loads(
            (Path(__file__).parent / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            project["project"]["dependencies"], [f"{manifest.package} @ {manifest.wheel_url}"]
        )
        self.assertIn(f"/{manifest.release}/", manifest.index)
        self.assertIn(manifest.version, manifest.wheel)
        # Nothing the wheel asks for is installed: it is unpacked, never imported.
        self.assertIn("voe", project["tool"]["uv"]["exclude-dependencies"])

    def test_the_tracked_manifest_names_the_five_files(self) -> None:
        manifest = provision.load_manifest(provision.default_manifest())
        self.assertEqual(manifest.release, "1.9.0")
        self.assertEqual(len(manifest.files), 5)
        self.assertEqual(sum(entry.bytes for entry in manifest.files), 713_594_768)


if __name__ == "__main__":
    unittest.main()
