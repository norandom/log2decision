"""Run the frozen fold-zero score and NPU policy on the task 2.24 panel.

Reuses the task 2.22 runner unchanged, pinned to this panel's manifest and to the CLI built
from the task 2.23 source. The product run uses the earlier policy at budget 12.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage222_policy as runner  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Panel directory from triage224_panel.py')
    parser.add_argument('--cli-sha256', required=True, help='SHA-256 of l2d-<first 8 hex>.exe archived in the run')
    parser.add_argument('--runtime', type=Path, default=Path('D:/Source/log2decision/runtime/ryzenai'))
    args = parser.parse_args()
    runner.INTEGRATED_CLI_SHA256 = args.cli_sha256
    runner.FROZEN_MANIFEST_SHA256 = runner.sha(args.run / 'manifest.json')
    archived = args.run / f'l2d-{args.cli_sha256[:8]}.exe'
    runner.require(archived.is_file(), f'Archive the CLI as {archived} first')
    runner.run_policy(args.run, args.runtime)
