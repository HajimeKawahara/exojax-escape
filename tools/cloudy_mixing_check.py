#!/usr/bin/env python3
"""Compare retained Cloudy n=3 mixing readouts with two explicit conventions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax

from cloudy_mixing_convention import cloudy_psm20_convention
from cloudy_rates_diagnosis import parse_operator
from escape.hydrogen_mixing import proton_l_mixing_psm20

jax.config.update("jax_enable_x64", True)


def analyze(root):
    rows = []
    paths = sorted([*root.glob("rates_T*_n*/*.out"), *root.glob("rates_T*_n*/*.out.gz")])
    if not paths:
        raise ValueError("No recorded rate-grid outputs were found.")
    for path in paths:
        operator = parse_operator(path)
        temperature, electron = operator["header"][3:5]
        transitions = operator["bb"]
        for upper, lower, ell in ((4, 3, 1), (5, 4, 2)):
            saved = next(row for row in transitions if row[:2] == [upper, lower])
            lifetime = 1 / sum(row[2] for row in transitions if row[0] == upper)
            reference = saved[8]  # C23 collider slot 1 is proton.
            shortcut, combined, crossover = cloudy_psm20_convention(temperature, electron, ell, ell - 1, lifetime)
            runtime = float(proton_l_mixing_psm20(temperature, electron, 3, ell, ell - 1, lifetime))
            rows.append({"run": path.parent.name, "temperature_k": temperature,
                         "electron_cm3": electron, "upper": upper, "lower": lower,
                         "lifetime_s": lifetime, "cloudy_q_cm3_s": reference,
                         "cloudy_convention_q_cm3_s": shortcut,
                         "continuous_cloudy_mass_q_cm3_s": combined,
                         "runtime_q_cm3_s": runtime, "cutoff_ratio": crossover,
                         "cloudy_convention_relative_error": shortcut / reference - 1,
                         "runtime_relative_difference": runtime / reference - 1,
                         "stored_output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"source": "Actual Cloudy C23.01 proton coefficients with read-only operator instrumentation.",
            "interpretation": "The exact-convention diagnostic retains the Cloudy cutoff shortcut. The runtime uses a continuous cutoff integral and physical H-1/proton mass; differences are intentional.",
            "rows": rows,
            "max_cloudy_convention_relative_error": max(abs(row["cloudy_convention_relative_error"]) for row in rows),
            "max_runtime_relative_difference": max(abs(row["runtime_relative_difference"]) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key.startswith("max_")}))


if __name__ == "__main__":
    main()
