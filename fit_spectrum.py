"""Test whether existing atmospheric parameters can reproduce the red-peak width.

This local least-squares diagnostic is not a posterior or a self-consistent
atmosphere. It treats supplied uncertainties as independent fitting weights;
the processed data may be correlated. No velocity offset or new broadening is
introduced. Fits use a coarse grid and are reevaluated at the original resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import curve_fit, least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_manual import MANUAL_PARAMETERS
from generate_spectra import OBSERVATION_PATH, load_observations
from hydrogen_forward import HydrogenForwardModel

TEMPERATURES_K = (12000.0, 15000.0, 9000.0)
FIT_GRID = dict(nlayer=32, nlon=12, nlat=6, nnu=601)
CHECK_GRID = dict(nlayer=64, nlon=24, nlat=12, nnu=1201)
# Coordinates: ln(N2 / 1e14), ln(b_min), Twind / 1e4, vzonal / 10, baseline / 1e-4.
LOWER = np.array([np.log(0.01), np.log(0.001), 0.6, 0.0, -5.0])
UPPER = np.array([np.log(30.0), np.log(0.8), 2.0, 3.0, 5.0])


def atmosphere(parameters):
    """Decode optimizer coordinates without breaking JAX derivatives."""
    return {
        **MANUAL_PARAMETERS,
        "n2_column": 1e14 * jnp.exp(parameters[0]),
        "departure_floor": jnp.exp(parameters[1]),
        "wind_temperature_k": 1e4 * parameters[2],
        "zonal_kms": 10.0 * parameters[3],
    }


def fit_at_temperature(temperature, data, initial, fixed_baseline=None):
    """Adjust four atmospheric inputs and optionally a constant baseline."""
    model = HydrogenForwardModel(temperature_top_k=temperature, **FIT_GRID)
    velocity, observed, error = data
    count = 5 if fixed_baseline is None else 4

    @jax.jit
    def residual(parameters):
        baseline = parameters[4] * 1e-4 if count == 5 else fixed_baseline
        spectrum = model.prescribed_spectrum(**atmosphere(parameters)) + baseline
        sampled = jnp.interp(velocity, model.velocity_kms[::-1], spectrum[::-1])
        return (sampled - observed) / error

    jacobian = jax.jit(jax.jacfwd(residual))
    result = least_squares(
        lambda p: np.asarray(residual(p)), initial[:count],
        jac=lambda p: np.asarray(jacobian(p)), bounds=(LOWER[:count], UPPER[:count]),
        max_nfev=60 if count == 5 else 40, x_scale="jac",
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )
    report = {
        "temperature_top_k": temperature,
        "parameters": {key: float(value) for key, value in atmosphere(result.x).items()},
        "baseline_fraction": float(result.x[4] * 1e-4 if count == 5 else fixed_baseline),
        "coarse_residual_rms": float(np.sqrt(np.mean(result.fun**2))),
        "success": bool(result.success), "message": result.message,
        "nfev": int(result.nfev), "active_bounds": result.active_mask.tolist(),
    }
    print(json.dumps(report), flush=True)
    return report, result.x


def red_width(data, flux, baseline):
    """Descriptive Gaussian proxy on the same observed samples and weights."""
    velocity, _, error = data
    use = (velocity > 5.0) & (velocity < 60.0)

    def gaussian(x, amplitude, center, sigma):
        return baseline + amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)

    parameters, _ = curve_fit(
        gaussian, velocity[use], flux[use], sigma=error[use],
        p0=[0.0015, 27.0, 8.0], bounds=([0.0, 10.0, 1.0], [0.005, 45.0, 30.0]),
    )
    return {"center_kms": float(parameters[1]),
            "fwhm_kms": float(np.sqrt(8 * np.log(2)) * parameters[2])}


def plot_comparison(data, velocity, spectra, cases, destination):
    """Keep the previous manual example visible alongside the adjusted model."""
    observed_velocity, observed, error = data
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].errorbar(
        observed_velocity, 100 * observed, yerr=100 * error, fmt=".",
        color="black", markersize=4, elinewidth=0.7, alpha=0.7,
        label="Keck/KPF observation (phases 0.30-0.45)",
    )
    for name, label, color, style in (
        ("manual", "Previous hand choices", "C3", "--"),
        ("adjusted", "Same model, adjusted inputs", "C0", "-"),
    ):
        case, spectrum = cases[name], spectra[name]
        label += (f" (Ttop={case['temperature_top_k']:,.0f} K); "
                  f"RMS {case['residual_rms']:.2f}")
        axes[0].plot(velocity, 100 * spectrum, color=color, ls=style, label=label)
        residual = (np.interp(observed_velocity, velocity[::-1], spectrum[::-1]) - observed) / error
        axes[1].plot(observed_velocity, residual, color=color, ls=style, marker=".", markersize=3)
    axes[0].set(ylabel="Line contrast [%]", title="Red-peak width within the existing forward model")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].axhspan(-1, 1, color="0.5", alpha=0.1)
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set(xlim=(-105, 105), xlabel="Velocity from NIST H-alpha [km/s]",
                ylabel="(Model - data) / error")
    fig.suptitle("Exploratory least-squares diagnostic; fixed rest center, no added broadening", fontsize=13)
    fig.supxlabel(
        "Data: Yapeng Zhang (2026), doi:10.5281/zenodo.20707255; CC BY 4.0\n"
        "Prescribed atmosphere at phase 0.375; RMS is diagnostic, not a calibrated fit statistic.",
        fontsize=8,
    )
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fit"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    manual_model = HydrogenForwardModel(temperature_top_k=12000.0, **CHECK_GRID)
    data = load_observations(manual_model.line_center_cm)
    velocity, observed, error = data
    outside = np.abs(velocity) > 50.0
    baseline = float(np.average(observed[outside], weights=error[outside]**-2))
    initial = np.array([
        np.log(MANUAL_PARAMETERS["n2_column"] / 1e14),
        np.log(MANUAL_PARAMETERS["departure_floor"]),
        MANUAL_PARAMETERS["wind_temperature_k"] / 1e4,
        MANUAL_PARAMETERS["zonal_kms"] / 10.0, baseline / 1e-4,
    ])
    fits = [fit_at_temperature(t, data, initial) for t in TEMPERATURES_K]
    successful = [fit for fit in fits if fit[0]["success"]]
    if not successful:
        raise RuntimeError("No temperature trial converged; inspect the printed fit reports.")
    best, best_coordinates = min(successful, key=lambda fit: fit[0]["coarse_residual_rms"])
    fixed, _ = fit_at_temperature(best["temperature_top_k"], data, best_coordinates, baseline)
    if not fixed["success"]:
        raise RuntimeError("The fixed-baseline control did not converge.")
    cases = {
        "manual": {"temperature_top_k": 12000.0, "parameters": MANUAL_PARAMETERS,
                   "baseline_fraction": baseline},
        "adjusted": dict(best), "fixed_baseline": dict(fixed),
    }
    spectra = {}
    for name, case in cases.items():
        model = (manual_model if name == "manual" else HydrogenForwardModel(
            temperature_top_k=case["temperature_top_k"], **CHECK_GRID))
        spectrum = np.asarray(model.prescribed_spectrum(**case["parameters"])) + case["baseline_fraction"]
        sampled = np.interp(velocity, model.velocity_kms[::-1], spectrum[::-1])
        residual = (sampled - observed) / error
        case["residual_rms"] = float(np.sqrt(np.mean(residual**2)))
        case["regional_rms"] = {
            region: float(np.sqrt(np.mean(residual[mask]**2))) for region, mask in (
                ("blue", velocity < -5), ("center", np.abs(velocity) <= 5), ("red", velocity > 5))
        }
        case["red_gaussian_proxy"] = red_width(data, sampled, case["baseline_fraction"])
        spectra[name] = spectrum
    report = {
        "note": "Local least-squares diagnostic, not NUTS or a physical atmosphere solution; correlated errors unmodeled.",
        "data_doi": "10.5281/zenodo.20707255", "data_file": OBSERVATION_PATH.name,
        "data_sha256": hashlib.sha256(OBSERVATION_PATH.read_bytes()).hexdigest(),
        "versions": {package: version(package) for package in ("exojax", "jax", "numpy", "scipy")},
        "model_phase": 0.375, "observation_phase_range": [0.30, 0.45],
        "wavelength_model": "nist", "rest_wavelength_angstrom": 1e8 / manual_model.line_center_cm,
        "planet_mass_mj": manual_model.planet_mass_mj,
        "fit_grid": FIT_GRID, "verification_grid": CHECK_GRID,
        "temperature_trials_k": TEMPERATURES_K,
        "optimizer_coordinates": ["ln(N2/1e14)", "ln(b_min)", "Twind/1e4", "vzonal/10", "baseline/1e-4"],
        "initial_coordinates": initial.tolist(), "lower_bounds": LOWER.tolist(), "upper_bounds": UPPER.tolist(),
        "baseline_cutoff_kms": 50.0, "red_proxy_window_kms": [5.0, 60.0],
        "observation_red_gaussian_proxy": red_width(data, observed, baseline),
        "coarse_fits": [fit[0] for fit in fits], "full_resolution": cases,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(
        args.output / "spectra.csv", np.column_stack([manual_model.velocity_kms, *spectra.values()]),
        delimiter=",", header="velocity_kms," + ",".join(spectra), comments="",
    )
    plot_comparison(data, manual_model.velocity_kms, spectra, cases, args.output / "fit_comparison.png")
    print(json.dumps(report, indent=2))
    print(f"Wrote parameters.json, spectra.csv, and fit_comparison.png to {args.output}")


if __name__ == "__main__":
    main()
