"""Attach an independent CPU check to an existing observer case receipt."""
import argparse
import json
from pathlib import Path

import numpy as np

from watch_candidates import assess, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-review", required=True, type=Path)
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    cpu = json.loads(args.cpu_review.read_text())
    if not cpu["passed"] or not cpu["checks"]["passed"]:
        raise ValueError("Independent CPU qualification did not pass")
    source = Path(cpu["source_directory"])
    destination = args.output / source.name
    state_path, checkpoint_path = source / "fixed_inputs.npz", source / "checkpoint.npz"
    cpu_arrays = Path(cpu["output"]["path"])
    if digest(cpu_arrays) != cpu["output"]["sha256"]:
        raise ValueError("Independent CPU array hash mismatch")
    review = json.loads((destination / "review.json").read_text())
    if review["inputs"][str(state_path)] != digest(state_path):
        raise ValueError("Observer evaluated a different state")
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    if not summary["accepted_for_observation"] or summary["fixed_inputs_sha256"] != digest(state_path):
        raise ValueError("Unaccepted or changed GPU export")
    physical, gpu, checkpoint = np.load(cpu_arrays), np.load(state_path), np.load(checkpoint_path)
    np.testing.assert_array_equal(physical["theta"], checkpoint["theta"])
    np.testing.assert_array_equal(physical["log_density_factor"], checkpoint["log_density_factor"])
    for key in ("temperature_k", "bound_cm3", "radial_velocity_kms", "continuum_extinction", "continuum_emission_hz"):
        np.testing.assert_allclose(physical[key], gpu[key], rtol=1e-10, atol=1e-25)
    assess(destination, dict(path=str(summary_path), sha256=digest(summary_path)),
           json.loads((args.output / "D1/review.json").read_text()))
    path = destination / "stage_assessment.json"
    result = json.loads(path.read_text())
    result.update(status="independent_CPU_verified",
        case_classification="Additional exploratory case" if args.exploratory else "Predeclared observer density factor",
        scope="Observer receipt for a supplied equilibrium snapshot independently checked on CPU. A prescribed supporting force is still required; this is not a closed MHD solution.",
        qualification_source_sha256=digest(__file__),
        cpu_qualification=dict(receipt_path=str(args.cpu_review), receipt_sha256=digest(args.cpu_review),
            cpu_arrays_path=str(cpu_arrays), cpu_arrays_sha256=digest(cpu_arrays),
            checkpoint_path=str(checkpoint_path), checkpoint_sha256=digest(checkpoint_path),
            theta_exactly_equal=True, used_physical_fields_allclose_rtol=1e-10,
            used_physical_fields_allclose_atol=1e-25,
            residual_infinity=cpu["residual_infinity"],
            CPU_GPU_residual_difference_infinity=cpu["residual_difference_infinity"]))
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Independent CPU qualification attached: {destination.name}")


if __name__ == "__main__":
    main()
