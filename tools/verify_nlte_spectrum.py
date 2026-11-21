#!/usr/bin/env python3
"""Record observer-spectrum numerical checks without running a fit or NUTS.

The bundled observation errors supply a numerical scale only. The synthetic
objective below is neither an observational likelihood nor a posterior model.
The parent process records the actual worker exit code and source hashes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ROOT / "data/zenodo_20707255/KELT9b_w6565_avg_p0.30-0.45.dat"
SIGMA_TOLERANCE = 0.1
GRADIENT_RTOL = 3e-3
GRADIENT_ATOL = 1e-8
SPECTRAL_GRADIENT_ATOL = 1e-10
DIFFERENCE_STEP = 1e-4
PARAMETER_NAMES = ("log_mass_loss", "log_dilution", "temperature_top_over_1e4",
                   "rotation_over_10kms", "zonal_over_10kms")
CASES = {
    "baseline": (1.0, 1.0, 10000.0),
    "low_mass": (0.3, 1.0, 10000.0),
    "high_mass": (3.0, 1.0, 10000.0),
    "low_dilution": (1.0, 0.5, 10000.0),
    "high_dilution": (1.0, 2.0, 10000.0),
    "cool": (1.0, 1.0, 8000.0),
    "hot": (1.0, 1.0, 12000.0),
}
REFINEMENTS = ("nmax", "layers", "disk", "wavelength", "continuum", "parker")
# Unresolved H I 2->3 NIST row retrieved 2026-11-16. Use the same recorded
# input for every run, without an implicit network request or a pytest fixture.
NIST_ROW = {
    "wav": [15233.146], "A": [44101000.0], "El": [82259.158],
    "Eu": [97492.304], "gl": [8.0], "gu": [18.0], "jl": [3.5],
    "ju": [8.5], "species": ["H_I"],
}


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        return None if isinstance(item, float) and not np.isfinite(item) else item

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def source_snapshot():
    """Hash all local model and ExoJAX Python sources and the numerical scale."""
    spec = importlib.util.find_spec("exojax")
    exojax_path = Path(next(iter(spec.submodule_search_locations)))
    paths = [Path(__file__).resolve(), OBSERVATIONS]
    paths.extend((ROOT / "src/escape").rglob("*.py"))
    paths.extend(exojax_path.rglob("*.py"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def uncertainty_scaled_difference(reference, candidate, uncertainty):
    """Reject nonfinite predictions rather than accidentally passing a gate."""
    reference, candidate, uncertainty = map(np.asarray, (reference, candidate, uncertainty))
    if reference.shape != candidate.shape or reference.shape != uncertainty.shape:
        raise ValueError("Spectra and uncertainties must have identical shapes.")
    if not np.isfinite(uncertainty).all() or not np.all(uncertainty > 0):
        raise ValueError("Every uncertainty must be finite and positive.")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"passed": False, "max_absolute_difference_over_sigma": None,
                "reason": "nonfinite spectrum"}
    difference = np.abs(candidate - reference) / uncertainty
    maximum = float(np.max(difference))
    return {"passed": maximum < SIGMA_TOLERANCE,
            "max_absolute_difference_over_sigma": maximum,
            "rms_difference_over_sigma": float(np.sqrt(np.mean(difference**2))),
            "threshold": SIGMA_TOLERANCE}


def require_observer_coverage(model_velocity, observer_velocity):
    """Do not silently extrapolate through jnp.interp endpoint clipping."""
    model_velocity, observer_velocity = map(np.asarray, (model_velocity, observer_velocity))
    if (not np.isfinite(model_velocity).all() or not np.isfinite(observer_velocity).all()
            or not np.all(np.diff(model_velocity) < 0)
            or observer_velocity.min() < model_velocity[-1]
            or observer_velocity.max() > model_velocity[0]):
        raise ValueError("The descending model velocity grid must cover every observation pixel.")


def gradient_comparison(first, second):
    first, second = np.asarray(first), np.asarray(second)
    scale = np.maximum(np.abs(first), np.abs(second))
    allowed = GRADIENT_ATOL + GRADIENT_RTOL * scale
    finite = np.isfinite(first).all() and np.isfinite(second).all()
    return {"passed": bool(finite and np.all(np.abs(first - second) <= allowed)),
            "maximum_error_over_tolerance":
                float(np.max(np.abs(first - second) / allowed)) if finite else None}


def worker(args):
    # The scientific imports occur in the child after the parent snapshots inputs.
    import jax
    import jax.numpy as jnp
    import pandas as pd
    from unittest.mock import patch

    from escape.nlte_forward import HydrogenNLTEForwardModel
    from exojax.utils.constants import c

    jax.config.update("jax_enable_x64", True)
    configuration = {
        "nmax": args.nmax, "nlayer": args.layers, "nlon": args.longitude,
        "nlat": args.latitude, "nnu": args.wavelength,
        "continuum_points": args.continuum, "wind_ngrid": args.parker,
    }
    parameters = {
        name: jnp.array([np.log(CASES[name][0]), np.log(CASES[name][1]),
                         CASES[name][2] / 1e4, args.rotation / 10, args.zonal / 10])
        for name in args.cases
    }
    wavelength, _, sigma = np.loadtxt(OBSERVATIONS, unpack=True)
    report = {
        "scope": "Prescribed-field spectrum numerics; no observation fit or NUTS.",
        "configuration": configuration, "cases": {name: np.asarray(value).tolist()
                                                   for name, value in parameters.items()},
        "parameter_names": PARAMETER_NAMES, "checks_requested": args.checks,
        "parameter_mapping": {"mass_loss_g_s": "1e13 * exp(theta[0])",
                              "radiation_dilution": "0.025 * exp(theta[1])",
                              "temperature_top_k": "1e4 * theta[2]",
                              "rotation_kms": "10 * theta[3]", "zonal_kms": "10 * theta[4]"},
        "sigma_tolerance": SIGMA_TOLERANCE,
        "gradient_rtol": GRADIENT_RTOL, "gradient_atol": GRADIENT_ATOL,
        "spectral_gradient_atol": SPECTRAL_GRADIENT_ATOL,
        "finite_difference_step": DIFFERENCE_STEP,
        "synthetic_target_case": args.cases[0],
        "uncertainty_scope": "Numerical scale only; confidence level not verified.",
        "nist_input": {"row": NIST_ROW, "source":
                       "https://physics.nist.gov/PhysRefData/ASD/lines_form.html"},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "jax_backend": jax.default_backend(),
                        "jax_devices": [str(device) for device in jax.devices()],
                        "jax_enable_x64": jax.config.jax_enable_x64,
                        "cpu_affinity": sorted(os.sched_getaffinity(0)),
                        "versions": {name: importlib.metadata.version(name)
                                     for name in ("jax", "jaxlib", "numpy", "exojax")}},
        "domain": {}, "refinements": {}, "gradients": {}, "timing": {},
    }
    arrays = {"observed_wavelength_angstrom": wavelength, "uncertainty": sigma}
    passed = True

    def save_progress(*, completed=False):
        report["completed"] = completed
        report["passed"] = bool(passed and completed)
        np.savez_compressed(args.record / "spectra.npz", **arrays)
        write_json(args.record / "metrics.json", report)

    def build(config):
        with patch("exojax.database.nist.api.fetch_nist_lines",
                   lambda *a, **kw: (pd.DataFrame(NIST_ROW), [])):
            model = HydrogenNLTEForwardModel(**config)
        velocity = c * (wavelength / (1e8 / model.line_center_cm) - 1)
        require_observer_coverage(model.velocity_kms, velocity)

        def keywords(theta):
            return {"mass_loss_g_s": 1e13 * jnp.exp(theta[0]),
                    "radiation_dilution": 0.025 * jnp.exp(theta[1]),
                    "temperature_top_k": 1e4 * theta[2],
                    "rotation_kms": 10 * theta[3], "zonal_kms": 10 * theta[4]}

        @jax.jit
        def predict(theta):
            flux = model.spectrum(**keywords(theta))
            return jnp.interp(velocity, model.velocity_kms[::-1], flux[::-1])

        return model, predict, keywords, velocity

    def elapsed(function, argument):
        start = time.perf_counter()
        result = jax.block_until_ready(function(argument))
        return result, time.perf_counter() - start

    print("Baseline spectra and domain diagnostics", flush=True)
    model, predict, keywords, velocity = build(configuration)
    arrays["observed_velocity_kms"] = velocity
    baseline = {}
    for name, theta in parameters.items():
        spectrum, seconds = elapsed(predict, theta)
        baseline[name] = np.asarray(spectrum)
        arrays[f"baseline_{name}"] = baseline[name]
        if name == args.cases[0]:
            report["timing"]["forward_first_call_seconds"] = seconds
        profile_keywords = keywords(theta)
        del profile_keywords["rotation_kms"], profile_keywords["zonal_kms"]
        profiles = jax.device_get(model.population_profiles(**profile_keywords))
        valid = bool(np.all(profiles["valid"]) and np.all(profiles["converged"])
                     and np.isfinite(baseline[name]).all())
        density = profiles["n_h_cm3"]
        report["domain"][name] = {
            "passed": valid, "all_local_solves_converged": bool(np.all(profiles["converged"])),
            "positive_integrated_extinction": bool(np.all(profiles["positive_extinction"])),
            "finite_spectrum": bool(np.isfinite(baseline[name]).all()),
            "temperature_min_k": float(np.min(profiles["temperature_k"])),
            "temperature_max_k": float(np.max(profiles["temperature_k"])),
            "density_min_cm3": float(np.min(density)),
            "density_max_cm3": float(np.max(density)),
            "max_nuclei_relative_error": float(np.max(np.abs(
                np.sum(profiles["bound_cm3"], axis=1)
                + profiles["proton_density_cm3"] - density) / density)),
            "max_charge_relative_error": float(np.max(np.abs(
                profiles["proton_density_cm3"] - profiles["electron_density_cm3"]) / density)),
            "max_rate_residual_norm": float(np.max(profiles["rate_residual_norm"])),
            "max_residual_norm": float(np.max(profiles["residual_norm"])),
        }
        passed &= valid
    save_progress()

    if "gradients" in args.checks or "timing" in args.checks:
        # This deterministic offset keeps derivatives nonzero at the baseline;
        # it is deliberately unrelated to the measured residual flux values.
        target = jnp.asarray(0.9 * baseline[args.cases[0]]
                             + 0.1 * sigma * np.exp(-((velocity - 15) / 30)**2))

        @jax.jit
        def objective(theta):
            return 0.5 * jnp.mean(((predict(theta) - target) / sigma)**2)

        reverse = jax.jit(jax.value_and_grad(objective))
        forward = jax.jit(jax.jacfwd(objective))
        first = parameters[args.cases[0]]
        _, seconds = elapsed(reverse, first)
        report["timing"]["value_and_grad_first_call_seconds"] = seconds
        if "timing" in args.checks:
            for label, function in (("forward", predict), ("value_and_grad", reverse)):
                samples = [elapsed(function, first)[1] for _ in range(args.repeats)]
                report["timing"][f"{label}_warm_seconds"] = samples
                report["timing"][f"{label}_warm_median_seconds"] = float(np.median(samples))
        if "gradients" in args.checks:
            print("Forward/reverse gradients and two finite-difference steps", flush=True)
            for name, theta in parameters.items():
                value, reverse_gradient = jax.device_get(reverse(theta))
                forward_gradient = np.asarray(forward(theta))
                differences = []
                for step in (DIFFERENCE_STEP, DIFFERENCE_STEP / 2):
                    derivatives = []
                    for index in range(len(PARAMETER_NAMES)):
                        delta = jnp.eye(len(PARAMETER_NAMES))[index] * step
                        derivatives.append(float((objective(theta + delta)
                                                  - objective(theta - delta)) / (2 * step)))
                    differences.append(derivatives)
                comparisons = {
                    "forward_vs_reverse": gradient_comparison(forward_gradient, reverse_gradient),
                    "reverse_vs_difference": gradient_comparison(reverse_gradient, differences[0]),
                    "reverse_vs_half_step": gradient_comparison(reverse_gradient, differences[1]),
                    "difference_step_stability": gradient_comparison(*differences),
                }
                checks_passed = all(item["passed"] for item in comparisons.values())
                passed &= checks_passed
                report["gradients"][name] = {
                    "passed": checks_passed, "objective": float(value),
                    "reverse": reverse_gradient.tolist(), "forward": forward_gradient.tolist(),
                    "finite_differences": differences, "comparisons": comparisons,
                }
            # A scalar objective can hide cancellation between pixels. Also
            # test every observer pixel at the first case, normalized by each
            # parameter's peak spectral derivative rather than by wing zeros.
            jacobian = np.asarray(jax.jit(jax.jacfwd(predict))(first))
            columns = []
            for index in range(len(PARAMETER_NAMES)):
                delta = jnp.eye(len(PARAMETER_NAMES))[index] * (DIFFERENCE_STEP / 2)
                columns.append(np.asarray((predict(first + delta) - predict(first - delta))
                                          / DIFFERENCE_STEP))
            finite_difference = np.stack(columns, axis=1)
            finite = np.isfinite(jacobian).all() and np.isfinite(finite_difference).all()
            derivative_scale = np.maximum(np.max(np.abs(jacobian), axis=0),
                                          np.max(np.abs(finite_difference), axis=0))
            error = np.max(np.abs(jacobian - finite_difference), axis=0)
            allowed = SPECTRAL_GRADIENT_ATOL + GRADIENT_RTOL * derivative_scale
            spectrum_passed = bool(finite and np.all(error <= allowed))
            report["spectral_gradients"] = {
                "case": args.cases[0], "passed": spectrum_passed,
                "max_error_over_tolerance_by_parameter": (error / allowed).tolist(),
                "peak_absolute_derivative_by_parameter": derivative_scale.tolist(),
            }
            arrays["spectral_jacobian"] = jacobian
            arrays["spectral_jacobian_finite_difference"] = finite_difference
            passed &= spectrum_passed
        del objective, reverse, forward
        save_progress()
    del model, predict
    jax.clear_caches()

    if "refinement" in args.checks:
        for axis in args.refinements:
            refined = dict(configuration)
            updates = {
                "nmax": {"nmax": 20}, "layers": {"nlayer": 2 * args.layers},
                "disk": {"nlon": 2 * args.longitude, "nlat": 2 * args.latitude},
                "wavelength": {"nnu": 2 * (args.wavelength - 1) + 1},
                "continuum": {"continuum_points": 2 * args.continuum},
                "parker": {"wind_ngrid": 2 * (args.parker - 1) + 1},
            }
            refined.update(updates[axis])
            print(f"Refinement: {axis} {updates[axis]}", flush=True)
            model, predict, _, _ = build(refined)
            comparisons = {}
            for name, theta in parameters.items():
                candidate = np.asarray(predict(theta))
                arrays[f"{axis}_{name}"] = candidate
                comparisons[name] = uncertainty_scaled_difference(baseline[name], candidate, sigma)
                passed &= comparisons[name]["passed"]
            report["refinements"][axis] = {"configuration": refined, "cases": comparisons}
            save_progress()
            del model, predict
            jax.clear_caches()

    save_progress(completed=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True, help="New evidence directory.")
    parser.add_argument("--checks", nargs="+", choices=("refinement", "gradients", "timing"),
                        default=["refinement", "gradients", "timing"])
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--refinements", nargs="+", choices=REFINEMENTS, default=list(REFINEMENTS))
    for name, default in (("nmax", 15), ("layers", 64), ("longitude", 12), ("latitude", 6),
                          ("wavelength", 601), ("continuum", 48), ("parker", 129), ("repeats", 3)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--rotation", type=float, default=5.0)
    parser.add_argument("--zonal", type=float, default=10.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
    if ("refinement" in args.checks and "nmax" in args.refinements
            and not 3 <= args.nmax < 20):
        parser.error("The nmax refinement requires 3 <= --nmax < 20.")
    args.record = args.record.resolve()
    if args.worker:
        try:
            return worker(args)
        except Exception:
            traceback.print_exc()
            return 2
    args.record.mkdir(parents=True, exist_ok=False)
    before = source_snapshot()
    write_json(args.record / "source_before.json", before)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    started = datetime.now(timezone.utc).isoformat()
    with (args.record / "stdout.txt").open("w") as stdout, (args.record / "stderr.txt").open("w") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    after = source_snapshot()
    write_json(args.record / "source_after.json", after)
    write_json(args.record / "execution.json", {
        "command": command, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "worker_exit_code": result.returncode, "source_unchanged": before == after,
        "environment": {"python": sys.executable, "platform": platform.platform(),
                        "variables": {name: os.environ.get(name) for name in (
                            "PYTHONPATH", "JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS",
                            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "passed": result.returncode == 0 and before == after,
    })
    print(f"Evidence: {args.record}; worker exit: {result.returncode}; source unchanged: {before == after}")
    return result.returncode if result.returncode else (0 if before == after else 1)


if __name__ == "__main__":
    raise SystemExit(main())
