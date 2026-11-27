"""Render accepted continuation exports sequentially; CPU qualification is separate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(directory, state_review, control):
    """Record bounds and D1 differences without changing the observer receipt."""
    path = directory / "review.json"
    review = json.loads(path.read_text())
    stages = {}
    for selection, cases in review["selections"].items():
        candidate = cases["candidate"]
        bounds = np.array([review["nuisance_bounds"][key] for key in ("rv_kms", "log_amplitude")])
        parameters = np.array(candidate["profiled"]["parameters"])
        distances = np.column_stack((parameters - bounds[:, 0], bounds[:, 1] - parameters))
        fractions = distances.min(axis=1) / np.diff(bounds).ravel()
        stages[selection] = dict(
            nuisance_success=candidate["profiled"]["nuisance_success"],
            fitted_RV_kms_and_log_amplitude=parameters.tolist(),
            distance_to_lower_and_upper_bounds=distances.tolist(),
            minimum_distance_as_fraction_of_bound_width=fractions.tolist(),
            within_one_percent_of_any_bound=bool(np.any(fractions < .01)),
            relative_to_D1={mode: dict(
                Q_improvement=control["selections"][selection]["candidate"][mode]["Q"] - candidate[mode]["Q"],
                target_model_plus_GP_change_BCR_ppm=[
                    control["selections"][selection]["candidate"][mode]["target"]["all"][band]["residual_ppm"]
                    - candidate[mode]["target"]["all"][band]["residual_ppm"]
                    for band in ("blue", "core", "red")]) for mode in ("fixed", "profiled")})
    result = dict(
        status="provisional_until_independent_CPU_check",
        scope="Observer-only numerical receipt. The GPU stage passed its own fresh residual/guards; final physical acceptance requires the separate independent CPU check.",
        source_sha256=digest(__file__), observer_review_sha256=digest(path),
        state_review=state_review, selections=stages,
        limitations=["Boundary distances are diagnostic, not a guarantee of a global optimum.",
                     "Changes in Q and target residuals are descriptive under fixed GP hyperparameters."])
    (directory / "stage_assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    control = json.loads((args.output / "D1/review.json").read_text())
    for factor in (2, 5, 10, 20):
        name = f"D{factor}"
        source = args.attempt / name
        summary_path = source / "summary.json"
        print(f"Waiting for accepted {name} export in {args.attempt}", flush=True)
        while not summary_path.exists():
            if (args.attempt / "summary.json").exists():
                raise RuntimeError(f"Continuation finished without {name}; no observation result fabricated")
            time.sleep(30)
        summary = json.loads(summary_path.read_text())
        if not summary.get("accepted_for_observation") or not summary.get("converged"):
            raise RuntimeError(f"Rejected physical stage {name}; observer skipped")
        state = source / "fixed_inputs.npz"
        if digest(state) != summary["fixed_inputs_sha256"]:
            raise RuntimeError(f"Physical export hash mismatch for {name}")
        configuration = json.loads((args.attempt / "configuration.json").read_text())
        for path, expected in configuration["source_and_input_sha256"].items():
            if digest(path) != expected:
                raise RuntimeError(f"Physical stage source/input changed: {path}")
        destination = args.output / name
        if (destination / "review.json").exists():
            saved = json.loads((destination / "review.json").read_text())
            if saved["inputs"].get(str(state)) != digest(state):
                raise RuntimeError(f"Previously rendered {name} has a different physical input")
        else:
            command = [sys.executable, "examples/halpha_dense_observe.py", "--state-npz", str(state),
                       "--state-review", str(summary_path), "--raw-output", str(args.raw / name),
                       "--output", str(destination)]
            print(f"Rendering provisional {name}", flush=True)
            subprocess.run(command, check=True)
        assess(destination, dict(path=str(summary_path), sha256=digest(summary_path)), control)
        print(f"Observer {name} complete; independent CPU qualification remains separate", flush=True)


if __name__ == "__main__":
    main()
