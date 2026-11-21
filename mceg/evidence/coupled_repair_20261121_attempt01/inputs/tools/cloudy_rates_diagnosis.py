#!/usr/bin/env python3
"""Reconstruct saved Cloudy operators and perform controlled rate-block swaps.

This is an independent NumPy/SciPy diagnostic, not a physical rate provider.
Extracted coefficients are frozen at the recorded plasma state. Re-solving
charge conservation does not re-evaluate their density-dependent physics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize import brentq


RATE_KEYS = ("radiative_bound_bound_s", "electron_bound_bound_cm3_s", "photoionization_s",
             "radiative_recombination_cm3_s", "electron_ionization_cm3_s",
             "three_body_recombination_cm6_s")


def read_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return stream.read()
    return path.read_text()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_operator(path):
    """Read the final complete instrumented H solve, rejecting partial outputs."""
    text = read_text(path)
    if "Cloudy exited OK" not in text:
        raise ValueError("Cloudy did not report successful completion.")
    blocks, block = [], None
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("MCEG_"):
            continue
        tag = fields[0]
        if tag == "MCEG_BEGIN":
            if block is not None:
                raise ValueError("Overlapping operator readouts.")
            block = {"header": list(map(float, fields[1:])), "fb": [], "bb": [],
                     "matrix": [], "two_photon": []}
        elif block is None:
            raise ValueError("Operator record outside a complete readout.")
        elif tag in ("MCEG_FB", "MCEG_BB", "MCEG_2PH"):
            key = {"MCEG_FB": "fb", "MCEG_BB": "bb", "MCEG_2PH": "two_photon"}[tag]
            block[key].append(list(map(float, fields[1:])))
        elif tag == "MCEG_MATRIX":
            if int(fields[1]) != len(block["matrix"]):
                raise ValueError("Matrix rows are missing or reordered.")
            block["matrix"].append(list(map(float, fields[2:])))
        elif tag == "MCEG_SOLUTION":
            block["solution"] = list(map(float, fields[1:]))
        elif tag == "MCEG_END":
            blocks.append(block)
            block = None
        else:
            raise ValueError(f"Unknown operator readout {tag}.")
    if block is not None or not blocks:
        raise ValueError("No final complete operator readout.")
    result = blocks[-1]
    header = result["header"]
    if len(header) != 8 or int(header[2]) != header[2]:
        raise ValueError("Unexpected operator header.")
    size = int(header[2])
    if (size < 6 or np.shape(result["fb"]) != (size, 14)
            or np.shape(result["matrix"]) != (size, size)
            or len(result.get("solution", [])) != size
            or len(result["bb"]) != size * (size - 1) // 2
            or not np.array_equal(np.asarray(result["fb"])[:, 0], np.arange(size))):
        raise ValueError("Incomplete finite-atom operator data.")
    for key in ("header", "fb", "bb", "matrix", "solution", "two_photon"):
        if not np.isfinite(np.asarray(result[key])).all():
            raise ValueError(f"Nonfinite operator data: {key}.")
    if header[3] <= 0 or header[4] <= 0 or header[7] <= 0:
        raise ValueError("Operator requires positive temperature, electrons and protons.")
    result["number_of_readouts"] = len(blocks)
    return result


def rate_blocks(operator):
    """Translate the saved local Cloudy process rates into a common convention."""
    temperature, electron, corrected_electron, _, _ = operator["header"][3:]
    continuum = np.asarray(operator["fb"])
    weights, energy_kelvin = continuum[:, 3], continuum[:, 4]
    size = len(continuum)
    radiative = np.zeros((size, size))
    collisional = np.zeros_like(radiative)
    for row in operator["bb"]:
        upper, lower = map(int, row[:2])
        einstein_a, loss_probability, pump, collision_down = row[2:6]
        radiative[upper, lower] = einstein_a * loss_probability + pump * weights[lower] / weights[upper]
        radiative[lower, upper] = pump
        collisional[upper, lower] = collision_down / electron
        collisional[lower, upper] = (collisional[upper, lower] * weights[upper] / weights[lower]
                                     * np.exp(-(energy_kelvin[upper] - energy_kelvin[lower]) / temperature))
    for upper, lower, spontaneous, induced_up, induced_down in operator["two_photon"]:
        radiative[int(upper), int(lower)] += spontaneous + induced_down
        radiative[int(lower), int(upper)] += induced_up
    ionization = continuum[:, 6] * corrected_electron / electron
    three_body = ionization * continuum[:, 7]
    recombination = continuum[:, 8] * continuum[:, 9] + continuum[:, 10] * continuum[:, 7]
    return dict(zip(RATE_KEYS, (radiative, collisional, continuum[:, 5], recombination,
                                ionization, three_body)))


def matrix(rates, electron):
    """Return destruction matrix [s^-1] and source per proton [s^-1]."""
    transitions = rates[RATE_KEYS[0]] + electron * rates[RATE_KEYS[1]]
    ionization = rates[RATE_KEYS[2]] + electron * rates[RATE_KEYS[4]]
    destruction = np.diag(transitions.sum(axis=1) + ionization) - transitions.T
    source = electron * rates[RATE_KEYS[3]] + electron**2 * rates[RATE_KEYS[5]]
    return destruction, source


def solve_frozen(rates, hydrogen_nuclei_cm3, electron=None):
    """Solve a frozen-coefficient system, optionally imposing pure-H charge."""
    if electron is None:
        def conservation(log_electron):
            trial = np.exp(log_electron)
            destruction, source = matrix(rates, trial)
            return trial * (1 + np.linalg.solve(destruction, source).sum()) / hydrogen_nuclei_cm3 - 1
        electron = np.exp(brentq(conservation, np.log(hydrogen_nuclei_cm3 * 1e-15),
                                np.log(hydrogen_nuclei_cm3), xtol=1e-13))
    destruction, source = matrix(rates, electron)
    bound = electron * np.linalg.solve(destruction, source)
    if not np.isfinite(bound).all() or np.any(bound <= 0):
        raise ValueError("The frozen diagnostic has no positive finite solution.")
    return {"electron_cm3": float(electron), "bound_cm3": bound.tolist(),
            "n2_cm3": float(bound[1:3].sum()), "n3_cm3": float(bound[3:6].sum())}


def interventions(legacy, cloudy):
    """Replace paired forward/inverse blocks while keeping every other block."""
    cases = {"legacy": legacy, "cloudy_full_operator": cloudy}
    for name in ("n3_mixing", "n2_mixing", "all_mixing", "bound_free", "ionization",
                 "bb_collision", "bb_radiation", "all_continuum", "nonmixing_collision",
                 "n3_mixing_and_bound_free", "n3_mixing_and_ionization",
                 "n3_mixing_and_bf_and_ionization"):
        rates = {key: value.copy() for key, value in legacy.items()}
        if "n3_mixing" in name or name == "all_mixing":
            rates[RATE_KEYS[1]][3:6, 3:6] = cloudy[RATE_KEYS[1]][3:6, 3:6]
        if name in ("n2_mixing", "all_mixing"):
            rates[RATE_KEYS[1]][1:3, 1:3] = cloudy[RATE_KEYS[1]][1:3, 1:3]
        if name in ("bound_free", "all_continuum") or "bound_free" in name or "_bf_" in name:
            for key in RATE_KEYS[2:4]:
                rates[key] = cloudy[key].copy()
        if name in ("ionization", "all_continuum") or "_ionization" in name:
            for key in RATE_KEYS[4:]:
                rates[key] = cloudy[key].copy()
        if name == "bb_collision":
            rates[RATE_KEYS[1]] = cloudy[RATE_KEYS[1]].copy()
        if name == "nonmixing_collision":
            rates[RATE_KEYS[1]] = cloudy[RATE_KEYS[1]].copy()
            rates[RATE_KEYS[1]][1:3, 1:3] = legacy[RATE_KEYS[1]][1:3, 1:3]
            rates[RATE_KEYS[1]][3:6, 3:6] = legacy[RATE_KEYS[1]][3:6, 3:6]
        if name == "bb_radiation":
            rates[RATE_KEYS[0]] = cloudy[RATE_KEYS[0]].copy()
        cases[name] = rates
    rates = {key: value.copy() for key, value in cloudy.items()}
    rates[RATE_KEYS[1]][3:6, 3:6] = legacy[RATE_KEYS[1]][3:6, 3:6]
    cases["cloudy_with_legacy_n3_mixing"] = rates
    return cases


def diagnose(operator, legacy, hydrogen_nuclei_cm3):
    """Check the operator extraction before using it for causal interventions."""
    cloudy = rate_blocks(operator)
    for key in RATE_KEYS:
        if legacy[key].shape != cloudy[key].shape or not np.isfinite(legacy[key]).all() or np.any(legacy[key] < 0):
            raise ValueError(f"Invalid legacy rate block {key}.")
    electron, proton = operator["header"][4], operator["header"][7]
    destruction, source = matrix(cloudy, electron)
    saved_matrix = np.asarray(operator["matrix"])
    saved_source = np.asarray(operator["fb"])[:, -1]
    checks = {
        "matrix_relative_max": float(np.max(np.abs(destruction - saved_matrix)) / np.max(np.abs(saved_matrix))),
        "source_relative_max": float(np.max(np.abs(source * proton - saved_source)) / np.max(np.abs(saved_source))),
        "fixed_ne_independent_solution_relative_max": float(np.max(np.abs(
            np.linalg.solve(destruction, source) * proton / np.asarray(operator["solution"]) - 1))),
    }
    # This is an extraction/linear-algebra requirement, not a physical tolerance.
    if any(value > 1e-9 for value in checks.values()):
        raise ValueError(f"Cloudy operator reconstruction failed: {checks}")
    cases = interventions(legacy, cloudy)
    result = {"operator_reconstruction": checks, "results": {},
              "cloudy_rates": {key: value.tolist() for key, value in cloudy.items()},
              "legacy_rates": {key: value.tolist() for key, value in legacy.items()},
              "limitations": [
                  "Fixed-ne interventions isolate rate blocks at the measured reference state.",
                  "The second diagnostic re-solves pure-H charge with frozen coefficients; it does not re-evaluate density-dependent mixing or Cloudy physics.",
                  "A successful block replacement identifies a contributor, not a production implementation of that physics."]}
    for name, density in (("fixed_ne", electron), ("frozen_coefficients_charge_closure", None)):
        result["results"][name] = {case: solve_frozen(rates, hydrogen_nuclei_cm3, density)
                                   for case, rates in cases.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudy-output", type=Path, required=True, help="Instrumented .out or .out.gz.")
    parser.add_argument("--legacy-rates", type=Path, required=True, help="Archived escape_reference.json.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--no-mixing-output", type=Path)
    parser.add_argument("--nmax", type=int, default=5)
    args = parser.parse_args()
    args.record.mkdir(parents=True, exist_ok=False)
    (args.record / "driver.py").write_bytes(Path(__file__).read_bytes())
    inputs = {"cloudy_output": args.cloudy_output, "legacy_rates": args.legacy_rates,
              "protocol": args.protocol}
    if args.no_mixing_output:
        inputs["no_mixing_output"] = args.no_mixing_output
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
               "driver_sha256": sha256(__file__), "status": "running", "inputs": {},
               "python": sys.version, "package_versions": {
                   name: importlib.metadata.version(name) for name in ("numpy", "scipy")}}
    try:
        for name, path in inputs.items():
            retained = name + (".out.gz" if path.suffix == ".gz" else path.suffix)
            shutil.copyfile(path, args.record / retained)
            receipt["inputs"][name] = {"source": str(path.resolve()), "retained": retained, "sha256": sha256(path)}
        write_json(args.record / "receipt.json", receipt)
        operator = parse_operator(args.cloudy_output)
        archive = json.loads(args.legacy_rates.read_text())
        legacy = {key: np.asarray(value) for key, value in archive["models"][str(args.nmax)]["rates"].items()}
        result = diagnose(operator, legacy, archive["conditions"]["n_h_cm3"])
        if args.no_mixing_output:
            native = parse_operator(args.no_mixing_output)
            bound = np.asarray(native["solution"])
            result["cloudy_native_no_mixing"] = {"bound_cm3": bound.tolist(),
                                                "n2_cm3": float(bound[1:3].sum()),
                                                "n3_cm3": float(bound[3:6].sum())}
        write_json(args.record / "operator.json", operator)
        write_json(args.record / "diagnosis.json", result)
        receipt["status"] = "completed"
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["artifacts"] = {path.name: sha256(path) for path in sorted(args.record.iterdir())
                            if path.is_file() and path.name != "receipt.json"}
    write_json(args.record / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "error") if key in receipt}))
    return int(receipt["status"] != "completed")


if __name__ == "__main__":
    sys.exit(main())
