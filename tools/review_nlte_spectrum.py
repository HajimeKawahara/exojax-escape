#!/usr/bin/env python3
"""Review known observer-model questions using an external MCEG installation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ASSESSMENTS = ("prescribed_field_review", "atmospheric_review")


def review(root: Path, python: str) -> int:
    """Inspect both scoped records without importing the scientific package."""
    contract = root / "mceg/contracts/escape.nlte_spectrum.yaml"
    failed = False
    for assessment in ASSESSMENTS:
        print(f"Reviewing {assessment}", flush=True)
        command = [
            python, "-m", "mceg", "check", str(contract),
            "--assessment", assessment, "--root", str(root),
            "--require-handling", "--plain", "--verbose",
        ]
        result = subprocess.run(command, check=False)
        failed |= result.returncode != 0
    print("This audit checks recorded handling of known questions. "
          "Unknown scientific support remains unknown; this is not approval "
          "for physical inference from observations.")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Provider checkout whose recorded hashes are checked.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter with MCEG's --require-handling support.")
    args = parser.parse_args(argv)
    try:
        return review(args.root.resolve(), args.python)
    except OSError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
