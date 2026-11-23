#!/usr/bin/env python3
"""Verify the finite PHOENIX irradiation experiment, not physical adequacy.

The supplied observational errors define a numerical scale only. Historical
continuum failures are retained when a finer quadrature is selected.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from escape.hydrogen_forward import SYNCHRONOUS_ROTATION_KMS
from escape.nlte_forward import HydrogenNLTEForwardModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from generate_spectra import load_observations
from stellar_halpha import load_surface, occupation_fields
from verify_nlte_spectrum import NIST_ROW, uncertainty_scaled_difference, write_json


CASES = {
    "previous_fit": dict(mass_loss_g_s=209185986389574.34,
                         temperature_top_k=6105.231332735623,
                         rotation_kms=SYNCHRONOUS_ROTATION_KMS, zonal_kms=11.6),
    "hot_manual": dict(mass_loss_g_s=1e15, temperature_top_k=14000.,
                       rotation_kms=0., zonal_kms=0.),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength", required=True, type=Path)
    parser.add_argument("--flux", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/stellar_verification"))
    parser.add_argument("--max-continuum", type=int, default=1536)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    surface = load_surface(args.wavelength, args.flux)
    paths = [Path(__file__), ROOT / "examples/stellar_halpha.py",
             ROOT / "examples/generate_spectra.py", ROOT / "tools/verify_nlte_spectrum.py",
             *sorted((ROOT / "src/escape").glob("*.py"))]
    report = dict(scope="Finite prescribed PHOENIX-field numerical checks; no fit, internal radiation or physical validation.",
                  surface=surface[2], cases=CASES, continuum_runs={}, continuum_comparisons={},
                  refinements={}, gradients={}, complete=False, passed=False,
                  source_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                                 for p in paths},
                  baseline_policy="No baseline is refitted for numerical differences.")
    arrays = {}
    runs = {}

    def save():
        write_json(args.output / "metrics.json", report)
        np.savez_compressed(args.output / "arrays.npz", **arrays)

    def build(points, **extra):
        with patch("exojax.database.nist.api.fetch_nist_lines",
                   lambda *a, **kw: (pd.DataFrame(NIST_ROW), [])):
            model = HydrogenNLTEForwardModel(**(dict(nlayer=128, nmax=15,
                continuum_points=points, nlon=12, nlat=6, nnu=601,
                tidal_coefficient_s2=3 * (2*np.pi/(1.48*86400))**2) | extra))
        fields = occupation_fields(model, surface)
        return model, fields

    def evaluate(model, fields, label):
        velocity, _, uncertainty = map(np.asarray, load_observations(model.line_center_cm))
        records, predictions = {}, {}
        transfer = jax.jit(model._transfer_line_coefficients)
        for case, parameters in CASES.items():
            start = time.monotonic()
            pop_kwargs = {k: v for k, v in parameters.items()
                          if k not in ("rotation_kms", "zonal_kms")}
            profiles = model.population_profiles(**pop_kwargs, **fields)
            spectrum = transfer(profiles["halpha_branch_extinction_integral"],
                profiles["halpha_branch_emissivity_integral"], profiles["radial_velocity_kms"],
                parameters["rotation_kms"], parameters["zonal_kms"], profiles["temperature_k"])
            prediction = np.interp(velocity, np.asarray(model.velocity_kms)[::-1],
                                   np.asarray(spectrum)[::-1])
            records[case] = dict(valid=bool(jnp.all(profiles["valid"])),
                converged=bool(jnp.all(profiles["converged"])),
                finite_spectrum=bool(np.isfinite(prediction).all()),
                max_rate_residual=float(jnp.max(profiles["rate_residual_norm"])),
                elapsed_seconds=time.monotonic()-start)
            predictions[case] = prediction
            arrays[label + "_" + case] = prediction
            print(label, case, json.dumps(records[case]), flush=True)
        arrays["observation_velocity_kms"] = velocity
        arrays["uncertainty"] = uncertainty
        return records, predictions, uncertainty

    points = 48
    while points <= args.max_continuum:
        model, fields = build(points)
        records, predictions, uncertainty = evaluate(model, fields, f"continuum{points}")
        runs[points] = (model, fields, predictions)
        report["continuum_runs"][str(points)] = records
        if len(runs) > 1:
            previous = sorted(runs)[-2]
            comparisons = {case: uncertainty_scaled_difference(
                runs[previous][2][case], predictions[case], uncertainty) for case in CASES}
            report["continuum_comparisons"][f"{previous}to{points}"] = comparisons
            print(f"continuum {previous}->{points}", json.dumps(comparisons), flush=True)
        save()
        if points >= 192:
            eligible = [candidate for candidate in sorted(runs)[:-1] if all(
                uncertainty_scaled_difference(runs[candidate][2][case], runs[finer][2][case],
                                              uncertainty)["passed"]
                for finer in runs if finer > candidate for case in CASES)]
            if eligible:
                selected = min(eligible)
                break
        points *= 2
    else:
        report["failure"] = "Continuum quadrature did not meet the declared criterion."
        report["complete"] = True
        save()
        return 1

    report["selected_continuum_points"] = selected
    report["selected_against_finer"] = {str(finer): {
        case: uncertainty_scaled_difference(runs[selected][2][case], runs[finer][2][case], uncertainty)
        for case in CASES} for finer in runs if finer > selected}
    save()
    model, fields, base = runs[selected]
    for label, extra in [("layers128to256", {"nlayer": 256}), ("nmax15to20", {"nmax": 20})]:
        refined_model, refined_fields = build(selected, **extra)
        records, predictions, _ = evaluate(refined_model, refined_fields, label)
        report["refinements"][label] = {case: records[case] | uncertainty_scaled_difference(
            base[case], predictions[case], uncertainty) for case in CASES}
        print(label, json.dumps(report["refinements"][label]), flush=True)
        save()

    # The hot case sits on the atomic upper-temperature boundary. Test both
    # sides of finite differences at the interior previous-fit atmosphere.
    parameters = CASES["previous_fit"]
    initial = jnp.array([0., 0., np.log(parameters["mass_loss_g_s"] / 1e13),
                         parameters["temperature_top_k"] / 1e4])
    observation_velocity = jnp.asarray(arrays["observation_velocity_kms"])

    @jax.jit
    def prediction(theta):
        spectrum = model.spectrum(
            line_occupation=fields["line_occupation"] * jnp.exp(theta[0]),
            continuum_occupation=fields["continuum_occupation"] * jnp.exp(theta[1]),
            mass_loss_g_s=1e13*jnp.exp(theta[2]), temperature_top_k=1e4*theta[3],
            rotation_kms=parameters["rotation_kms"], zonal_kms=parameters["zonal_kms"])
        return jnp.interp(observation_velocity, model.velocity_kms[::-1], spectrum[::-1]) / uncertainty

    automatic = np.asarray(jax.jit(jax.jacfwd(prediction))(initial))
    step = 1e-4
    finite = np.stack([np.asarray((prediction(initial + step*direction)
                                  - prediction(initial - step*direction)) / (2*step))
                       for direction in np.eye(4)], axis=-1)
    allowed = 1e-7 + .003*np.maximum(np.abs(automatic), np.abs(finite))
    reverse = np.asarray(jax.jit(jax.grad(lambda theta: jnp.mean(prediction(theta))))(initial))
    expected_reverse = automatic.mean(axis=0)
    reverse_allowed = 1e-7 + .003*np.maximum(abs(reverse), abs(expected_reverse))
    report["gradients"] = dict(case="previous_fit",
        coordinates=["log_line_amplitude", "log_continuum_amplitude", "log_mass_over_1e13", "temperature_top_over_1e4"],
        step=step, rtol=.003, atol=1e-7,
        finite=bool(np.isfinite(automatic).all() and np.isfinite(finite).all() and np.isfinite(reverse).all()),
        maximum_error_over_tolerance=float(np.max(abs(automatic-finite)/allowed)),
        reverse_error_over_tolerance=float(np.max(abs(reverse-expected_reverse)/reverse_allowed)),
        passed=bool(np.isfinite(automatic).all() and np.isfinite(finite).all()
                    and np.all(abs(automatic-finite) <= allowed)
                    and np.all(abs(reverse-expected_reverse) <= reverse_allowed)))
    arrays["forward_jacobian_over_sigma"] = automatic
    arrays["finite_difference_jacobian_over_sigma"] = finite
    arrays["reverse_mean_gradient_over_sigma"] = reverse
    report["complete"] = True
    report["passed"] = bool(report["gradients"]["passed"] and all(
        rec["passed"] and rec["valid"] and rec["converged"]
        for group in report["refinements"].values() for rec in group.values())
        and all(rec["valid"] and rec["converged"] and rec["finite_spectrum"]
                for rec in report["continuum_runs"][str(selected)].values()))
    save()
    print("final", json.dumps({"selected_continuum_points": selected,
                               "gradients": report["gradients"], "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
