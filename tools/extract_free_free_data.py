"""Extract the three C23 free-free tables used by the low-charge atmosphere.

Offline only; the runtime package does not import or execute Cloudy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloudy_data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tables, hashes = [], {}
    grid = None
    for charge in (1, 2, 3):
        path = args.cloudy_data/f"gauntff_merged_Z{charge:02d}.dat"
        lines = path.read_text().splitlines()
        if int(lines[0].split()[0]) != 20140510 or int(lines[5].split()[0]) != charge:
            raise ValueError(f"Unexpected C23 table: {path}")
        ng, nu = map(int, lines[1].split()[:2])
        current_grid = [float(lines[i].split()[0]) for i in (2, 3, 4)]
        if grid is not None and grid != current_grid:
            raise ValueError("Inconsistent Gaunt-factor grids.")
        grid = current_grid
        table = np.loadtxt(lines[8:8+nu])
        if table.shape != (nu, ng) or not np.all(table > 0):
            raise ValueError("Invalid Gaunt factors.")
        tables.append(table)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output/"free_free.npz"
    np.savez_compressed(output, gaunt=np.stack(tables), grid=grid)
    provenance = {
        "source": "Cloudy C23.01, gauntff_merged_Z01/Z02/Z03.dat",
        "references": ["van Hoof et al. 2014, MNRAS 444, 420",
                       "van Hoof et al. 2015, MNRAS 449, 2112"],
        "license": "../_licenses/Cloudy.txt",
        "source_sha256": hashes,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "changes": "Retain the positive Gaunt-factor grid; omit the unused uncertainty table. No fitted values or stellar spectrum included.",
    }
    (args.output/"free_free_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    main()
