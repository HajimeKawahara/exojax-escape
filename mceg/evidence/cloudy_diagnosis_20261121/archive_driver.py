#!/usr/bin/env python3
"""Archive selected Cloudy runs without changing their receipts or raw bytes.

Large text outputs use deterministic gzip. Auxiliary files over the declared
size limit are listed with their hashes, not silently treated as retained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(runs, destination, max_auxiliary_bytes=2_000_000):
    destination = Path(destination)
    if not runs or any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in runs):
        raise ValueError("Supply runs with plain, unique names.")
    destination.mkdir(parents=True, exist_ok=False)
    report = {"archived_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "driver_sha256": digest(Path(__file__).read_bytes()),
              "scope": "Retention only; no scientific support or completion is inferred.",
              "max_auxiliary_bytes": max_auxiliary_bytes, "runs": {}}
    for name, source in runs.items():
        source = Path(source).resolve()
        target = destination / name
        target.mkdir()
        row = {"source_directory": str(source), "files": [], "omitted": []}
        for path in sorted(source.iterdir()):
            if not path.is_file():
                raise ValueError(f"Expected flat Cloudy outputs; found {path}.")
            data = path.read_bytes()
            entry = {"original_name": path.name, "original_bytes": len(data), "original_sha256": digest(data)}
            if len(data) > max_auxiliary_bytes and path.suffix != ".out":
                row["omitted"].append({**entry, "reason": "Auxiliary output exceeds the declared retention size limit."})
                continue
            compress = path.suffix == ".out"
            retained = target / (path.name + ".gz" if compress else path.name)
            stored = gzip.compress(data, mtime=0) if compress else data
            retained.write_bytes(stored)
            row["files"].append({**entry, "retained": retained.relative_to(destination).as_posix(),
                                 "compression": "gzip" if compress else "none", "stored_sha256": digest(stored)})
        report["runs"][name] = row
    (destination / "archive_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=DIRECTORY; repeatable.")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--max-auxiliary-bytes", type=int, default=2_000_000)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        name, separator, path = value.partition("=")
        if not separator or name in runs:
            parser.error("Use unique NAME=DIRECTORY run arguments.")
        runs[name] = Path(path)
    report = archive(runs, args.record, args.max_auxiliary_bytes)
    print(json.dumps({"runs": len(report["runs"]), "record": str(args.record)}))


if __name__ == "__main__":
    main()
