"""Exercise main's resource finalization without importing training dependencies."""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "hg17_run.py"
sys.path.insert(0, str(SOURCE.parent))
import memguard


class ResourceFinalization(unittest.TestCase):
    def invoke(self, root, step):
        main = next(node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        namespace = dict(argparse=argparse, os=os, time=time, sys=sys, json=json,
                         memguard=memguard, hg17=types.SimpleNamespace(path=lambda name: root / name),
                         STEPS={"line": step}, VARIANTS={"mask": None}, LM={}, LINE={}, __doc__="test")
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(SOURCE), "exec"), namespace)
        with mock.patch.object(memguard, "LOCK", str(root / "host.lock")), \
             mock.patch.object(memguard, "limit"), \
             mock.patch.object(memguard, "peaks", return_value={"limit_gib": 6.0,
                 "peak_committed_gib": 0.125, "peak_working_set_gib": 0.25}), \
             mock.patch.object(sys, "argv", ["hg17_run.py", "line"]):
            try:
                namespace["main"]()
            finally:
                with memguard.hungry("after training"):
                    pass

    def test_failure_is_recorded_and_the_original_exception_survives(self):
        original = KeyboardInterrupt("injected training failure")
        def fail(args):
            raise original
        with tempfile.TemporaryDirectory(prefix="l2d-hg17-finalize-") as directory:
            root = Path(directory)
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.invoke(root, fail)
            self.assertIs(caught.exception, original)
            run = json.loads((root / "runs.jsonl").read_text())
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["error"]["type"], "KeyboardInterrupt")
            self.assertEqual(run["memory"]["peak_committed_gib"], 0.125)
            self.assertTrue(run["resource_admission"]["gate_held"])
            self.assertFalse(run["resource_admission"]["spilled"])

    def test_success_records_completion(self):
        with tempfile.TemporaryDirectory(prefix="l2d-hg17-finalize-") as directory:
            root = Path(directory)
            self.invoke(root, lambda args: None)
            run = json.loads((root / "runs.jsonl").read_text())
            self.assertEqual(run["status"], "completed")
            self.assertIsNone(run["error"])

    def test_failed_record_write_does_not_replace_training_exception(self):
        original = RuntimeError("training failure remains primary")
        def fail(args):
            raise original
        with tempfile.TemporaryDirectory(prefix="l2d-hg17-finalize-") as directory:
            root = Path(directory)
            (root / "runs.jsonl").mkdir()
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics), self.assertRaises(RuntimeError) as caught:
                self.invoke(root, fail)
            self.assertIs(caught.exception, original)
            self.assertIn("could not finalize resource record", diagnostics.getvalue())


if __name__ == "__main__":
    unittest.main()
