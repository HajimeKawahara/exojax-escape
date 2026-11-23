#!/usr/bin/env python3
"""Retain multi-zone Cloudy hydrogen references for coupled-transfer checks.

This external tool runs a fixed-density, fixed-temperature slab. Successful
execution is not agreement with escape or validation of a planetary wind.
Cloudy's escape-probability treatment differs from complete-redistribution
moment transfer; geometry, boundary illumination and atom must be compared too.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from cloudy_reference import (
    artifact_bytes, artifact_sha256, hydrogen_input, sha256, store_artifact, write_json,
)


def slab_input(config):
    """Use the established finite atom with explicit column and illumination."""
    column, zones = config["column_cm2"], config["zones"]
    if not math.isfinite(column) or column <= 0:
        raise ValueError("Column density must be finite and positive.")
    if not isinstance(zones, int) or isinstance(zones, bool) or zones < 0:
        raise ValueError("Use a nonnegative integer zone count (zero means adaptive).")
    if config["illumination"] not in ("beam", "isotropic"):
        raise ValueError("Illumination must be beam or isotropic.")
    original = {key: config[key] for key in (
        "nmax", "temperature_k", "density_cm3", "radiation_temperature_k", "dilution", "resolution",
    )}
    original["thickness_cm"] = column / config["density_cm3"] / max(zones, 1)
    text = hydrogen_input(original)
    text = text.replace("Pure hydrogen thin reference", "Pure hydrogen slab reference")
    text = text.replace("print h-like population hydrogen\n", "")
    illumination = "0" if config["illumination"] == "beam" else "isotropic"
    text = text.replace("hden ", f"illumination forward {illumination}\nhden ", 1)
    if zones:
        text = text.replace("stop zone 1\n", f"stop zone {zones}\n")
    else:
        text = re.sub(r"^set dr .*\n", "", text, flags=re.MULTILINE)
        text = text.replace("stop zone 1\n", f"stop column density {math.log10(column):.17g}\nstop zone 5000\n")
    return text


def _rows(path):
    values = np.loadtxt(io.StringIO(artifact_bytes(path).decode()), ndmin=2)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"Missing or nonfinite saved values in {path.name}.")
    return values


def parse_profile(directory, nmax):
    """Return final-zone arrays in inward depth order, retaining measured charge."""
    directory = Path(directory)
    output = artifact_bytes(directory / "model.out").decode(errors="replace")
    if "Cloudy exited OK" not in output:
        raise ValueError("Cloudy did not report successful completion.")
    levels = _rows(directory / "model.levels")
    conditions = _rows(directory / "model.conditions")
    electrons = _rows(directory / "model.electrons")
    protons = _rows(directory / "model.protons")
    size = nmax + 3
    depth = levels[:, 0]
    if (levels.shape[1] < size + 1 or np.any(levels[:, size + 1:] != 0)
            or np.any(np.diff(depth) <= 0) or np.any(depth <= 0)):
        raise ValueError("Saved populations do not match the declared finite atom or depth grid.")
    for values in (conditions, electrons, protons):
        if values.shape[0] != depth.size or not np.array_equal(values[:, 0], depth):
            raise ValueError("Saved species and condition grids differ.")
    if conditions.shape[1] < 6 or electrons.shape[1] != 2 or protons.shape[1] != 2:
        raise ValueError("Unexpected condition or charge-density columns.")
    bound = levels[:, 1:size + 1]
    if (np.any(bound < 0) or np.any(conditions[:, 1:3] <= 0)
            or np.any(electrons[:, 1] < 0) or np.any(protons[:, 1] < 0)):
        raise ValueError("Unphysical saved populations or plasma conditions.")
    return dict(depth_cm=depth, bound_cm3=bound, temperature_k=conditions[:, 1],
                n_h_cm3=conditions[:, 2], electron_cm3=electrons[:, 1],
                proton_cm3=protons[:, 1], n2_cm3=bound[:, 1:3].sum(axis=1),
                n3_cm3=bound[:, 3:6].sum(axis=1))


def run(executable, destination, config, *, timeout=900., provenance=None):
    """Keep every attempt, raw hashes and derived profile without support labels."""
    input_text = slab_input(config)
    executable, destination = Path(executable).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "model.in").write_text(input_text)
    (destination / "driver.py").write_bytes(Path(__file__).read_bytes())
    receipt = dict(
        configuration=config, started_utc=datetime.now(timezone.utc).isoformat(),
        executable=str(executable), executable_sha256=sha256(executable),
        tool_source_sha256={name: sha256(Path(__file__).with_name(name))
                            for name in (Path(__file__).name, "cloudy_reference.py")},
        python=sys.version, numpy_version=np.__version__,
        command=[str(executable), "-r", "model"], status="running",
        geometry="plane-parallel slab; depth increases inward from the illuminated face",
        boundary="Forward-only blackbody; Cloudy normalization is 4 W sigma T^4 = 4 pi J_inc integrated over frequency, not hemispheric normal flux.",
        scope="Reference execution only. No escape agreement or scientific Assessment is inferred.",
        limitations=["No hydrodynamics, energy balance, metals, molecules or observational processing.",
                     "Cloudy's line escape probabilities and redistribution are not identical to moment transfer.",
                     "Surface zones and reference continuum resolution require independent refinement.",
                     "Executable source/data origin is not inferred from a version banner or hash."],
    )
    if provenance is not None:
        raw = Path(provenance).read_bytes()
        json.loads(raw)
        (destination / "provenance.json").write_bytes(raw)
    write_json(destination / "receipt.json", receipt)
    try:
        with (destination / "stdout.log").open("w") as stdout, (destination / "stderr.log").open("w") as stderr:
            process = subprocess.run(receipt["command"], cwd=destination, stdout=stdout,
                                     stderr=stderr, timeout=timeout, check=False)
        receipt["exit_code"] = process.returncode
        if process.returncode:
            raise ValueError(f"Cloudy exited with code {process.returncode}.")
        profile = parse_profile(destination, config["nmax"])
        output = (destination / "model.out").read_text(errors="replace")
        stops = [line.strip() for line in output.splitlines()
                 if "Calculation stopped because" in line]
        expected_stop = "NZONE reached" if config["zones"] else "H column dens reached"
        if (not stops or expected_stop not in stops[-1] or "not converged" in stops[-1]
                or (config["zones"] and len(profile["depth_cm"]) != config["zones"])):
            raise ValueError("Cloudy did not converge at the requested stopping condition.")
        receipt["final_stopping_condition"] = stops[-1]
        np.savez(destination / "profile.npz", **profile)
        receipt.update(status="completed", zones=len(profile["depth_cm"]),
                       cloudy_banner=next(line.strip() for line in output.splitlines() if re.search(r"Cloudy \d+\.", line)),
                       diagnostic_messages=sorted({line.strip() for line in output.splitlines()
                                                   if line.lstrip().startswith(("C-", "W-", "PROBLEM"))}))
        charge_error = np.max(np.abs(profile["electron_cm3"] - profile["proton_cm3"]) / profile["n_h_cm3"])
        receipt["saved_charge_relative_error_maximum"] = float(charge_error)
    except (OSError, ValueError, subprocess.TimeoutExpired, StopIteration) as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["artifacts"] = {}
    for path in sorted(destination.iterdir()):
        if not path.is_file() or path.name == "receipt.json":
            continue
        receipt["artifacts"][path.name] = artifact_sha256(path)
        if path.stat().st_size > 100_000 and path.suffix not in (".npz", ".gz"):
            store_artifact(path, path.read_bytes())
            path.unlink()
    write_json(destination / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="External source/data provenance JSON retained with the attempt.")
    parser.add_argument("--timeout", type=float, default=900.)
    parser.add_argument("--nmax", type=int, default=15)
    parser.add_argument("--temperature-k", type=float, default=8000.)
    parser.add_argument("--density-cm3", type=float, default=1e12)
    parser.add_argument("--radiation-temperature-k", type=float, default=10000.)
    parser.add_argument("--dilution", type=float, default=.025)
    parser.add_argument("--column-cm2", type=float, default=1e18)
    parser.add_argument("--zones", type=int, default=0, help="Zero selects Cloudy's adaptive zoning.")
    parser.add_argument("--resolution", type=float, default=.05)
    parser.add_argument("--illumination", choices=("beam", "isotropic"), default="beam")
    args = parser.parse_args()
    config = {name: getattr(args, name) for name in (
        "nmax", "temperature_k", "density_cm3", "radiation_temperature_k", "dilution",
        "column_cm2", "zones", "resolution", "illumination",
    )}
    try:
        receipt = run(args.executable, args.record, config, timeout=args.timeout, provenance=args.provenance)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({key: receipt[key] for key in ("status", "zones", "error") if key in receipt}, indent=2))
    return int(receipt["status"] != "completed")


if __name__ == "__main__":
    sys.exit(main())
