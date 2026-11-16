"""Fit three population closures with the same Parker prescription and data.

All cases use local least squares, several deterministic starts, then full-grid
refinement. Supplied errors are treated as independent weights, not a calibrated
likelihood for the processed data. This is not NUTS or a global optimum proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from generate_spectra import OBSERVATION_PATH, load_observations
from hydrogen_forward import HydrogenForwardModel, SYNCHRONOUS_ROTATION_KMS
from hydrogen_populations import COLLISION_STRENGTH_23

IONIZED_FRACTION = 0.5
MEAN_MOLECULAR_WEIGHT = 1 / (1 + IONIZED_FRACTION)
FIT_GRID = dict(nlayer=32, nlon=12, nlat=6, nnu=601)
FULL_GRID = dict(nlayer=64, nlon=24, nlat=12, nnu=1201)
LABELS = {
    1: "Model 1: exponential + sigmoid",
    2: "Model 2: continuity + sigmoid",
    3: "Model 3: continuity + rates",
}


def bounds(model_id, mass_loss_max=1e16):
    """Coordinates: ln(N2/1e14), ln(b or Mdot/1e13), T/1e4, Tw/1e4, v/10, c/1e-4."""
    control = (np.log(1e-4), 0.0) if model_id < 3 else (
        np.log(1e11 / 1e13), np.log(mass_loss_max / 1e13),
    )
    return (
        np.array([np.log(0.01), control[0], 0.6, 0.6, 0.0, -5.0]),
        np.array([np.log(30.0), control[1], 3.0, 3.0, 3.0, 5.0]),
    )


def inputs(model, model_id, coordinates):
    """Decode the same common parameters and one closure-specific parameter."""
    p = coordinates
    common = dict(
        temperature_top_k=p[2] * 1e4, wind_temperature_k=p[3] * 1e4,
        rotation_kms=SYNCHRONOUS_ROTATION_KMS, zonal_kms=p[4] * 10,
    )
    column = jnp.exp(p[0]) * 1e14
    if model_id < 3:
        return dict(
            **common, n2_column=column, departure_floor=jnp.exp(p[1]),
            mean_molecular_weight=MEAN_MOLECULAR_WEIGHT,
        )
    populations = dict(
        mass_loss_g_s=jnp.exp(p[1]) * 1e13,
        ionized_fraction=IONIZED_FRACTION, radiation_dilution=0.025,
        collision_strength=COLLISION_STRENGTH_23,
    )
    unit = model.population_profiles(
        **populations, excited_fraction=1.0,
        temperature_top_k=common["temperature_top_k"],
        wind_temperature_k=common["wind_temperature_k"],
    )
    fraction = column / jnp.sum(unit["n2_cm3"] * model.thickness_cm)
    return dict(**common, **populations, excited_fraction=fraction)


def evaluate(model, model_id, coordinates):
    """Use the existing transfer methods; only the population closure differs."""
    method = {1: model.prescribed_spectrum, 2: model.continuity_spectrum, 3: model.spectrum}[model_id]
    return method(**inputs(model, model_id, coordinates)) + coordinates[5] * 1e-4


def fit_runner(model, model_id, data, mass_loss_max=1e16):
    """Compile one residual/Jacobian pair for all starts on this grid."""
    velocity, observed, error = data

    @jax.jit
    def residual(p):
        spectrum = evaluate(model, model_id, p)
        prediction = jnp.interp(velocity, model.velocity_kms[::-1], spectrum[::-1])
        return (prediction - observed) / error

    jacobian = jax.jit(jax.jacfwd(residual))

    def run(initial, max_nfev):
        result = least_squares(
            lambda p: np.asarray(residual(p)), initial,
            jac=lambda p: np.asarray(jacobian(p)), bounds=bounds(model_id, mass_loss_max),
            x_scale="jac", max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        report = dict(
            coordinates=result.x.tolist(), success=bool(result.success),
            message=result.message, nfev=int(result.nfev),
            active_bounds=result.active_mask.tolist(),
            residual_rms=float(np.sqrt(np.mean(result.fun**2))),
            optimality=float(result.optimality),
        )
        print(json.dumps({"model": model_id, **report}), flush=True)
        return report

    return run


def starts(model_id, baseline):
    if model_id < 3:
        physical = [
            (0.8654, 0.00557, 1.5, 1.8107, 1.9625),
            (1.5, 0.03, 1.2, 1.6, 0.6),
            (3.0, 0.08, 0.9, 1.2, 1.16),
            (0.5, 0.001, 2.0, 2.4, 2.5),
        ]
    else:
        physical = [
            (0.8654, 10.0, 1.5, 0.9, 1.96),
            (3.0, 100.0, 1.5, 1.4, 1.2),
            (1.5, 1.0, 1.2, 1.2, 0.6),
            (1.0, 1000.0, 2.4, 1.8, 2.2),
        ]
    return [np.array([np.log(n), np.log(q), t, w, z, baseline / 1e-4])
            for n, q, t, w, z in physical]


def profiles(model, model_id, coordinates):
    """Return comparable n2, n3, excitation ratio, and Parker velocity."""
    kw = inputs(model, model_id, coordinates)
    for key in ("rotation_kms", "zonal_kms"):
        kw.pop(key)
    if model_id == 2:
        return model.continuity_profiles(**kw)
    if model_id == 3:
        return model.population_profiles(**kw)
    ratio = model.prescribed_ratio(kw["departure_floor"], kw["temperature_top_k"])
    lower = model.n2_reference * (kw["n2_column"] / 3e14)
    return dict(
        n2_cm3=lower, n3_cm3=lower * ratio, ratio=ratio,
        radial_velocity_kms=model.radial_velocity(
            kw["wind_temperature_k"], kw["mean_molecular_weight"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/three_models"))
    parser.add_argument("--check-convergence", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Reuse saved fit coordinates to regenerate spectra and figures.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    coarse = HydrogenForwardModel(**FIT_GRID)
    full = HydrogenForwardModel(**FULL_GRID)
    data = load_observations(full.line_center_cm)
    velocity, observed, error = data
    outside = np.abs(velocity) > 50
    baseline = float(np.average(observed[outside], weights=error[outside]**-2))
    report = dict(
        note="Local multistart least squares; not NUTS or formal model selection. Processed error correlations omitted.",
        data_doi="10.5281/zenodo.20707255",
        data_sha256=hashlib.sha256(OBSERVATION_PATH.read_bytes()).hexdigest(),
        phase=0.375, observation_phase_range=[0.30, 0.45],
        rest_wavelength_angstrom=1e8 / full.line_center_cm,
        planet_mass_mj=full.planet_mass_mj, ionized_fraction=IONIZED_FRACTION,
        coordinate_names=["ln(N2/1e14)", "ln(b_min) or ln(Mdot/1e13)",
                          "Ttop/1e4", "Twind/1e4", "zonal/10", "baseline/1e-4"],
        mean_molecular_weight=MEAN_MOLECULAR_WEIGHT,
        fit_grid=FIT_GRID, final_grid=FULL_GRID, models={},
    )
    if args.plot_only:
        saved = json.loads((args.output / "parameters.json").read_text())
        if saved["data_sha256"] != report["data_sha256"] or set(saved["models"]) != {"1", "2", "3"}:
            raise ValueError("Saved fits must contain all three models and the same data.")
        if saved["final_grid"] != FULL_GRID or saved["mean_molecular_weight"] != MEAN_MOLECULAR_WEIGHT:
            raise ValueError("Saved fits use different grids or mean molecular weight.")
        saved["coordinate_names"] = report["coordinate_names"]
        report = saved
    else:
        for model_id in LABELS:
            initial = starts(model_id, baseline)
            fit = fit_runner(coarse, model_id, data)
            trials = [fit(p, 160) for p in initial]
            successful = sorted((r for r in trials if r["success"]), key=lambda r: r["residual_rms"])
            if not successful:
                raise RuntimeError(f"No converged coarse trial for model {model_id}.")
            # Refine up to two distinct solutions rather than repeating the same minimum.
            chosen = [successful[0]]
            for candidate in successful[1:]:
                if np.linalg.norm(np.array(candidate["coordinates"]) - chosen[0]["coordinates"]) > 0.01:
                    chosen.append(candidate)
                    break
            refine = fit_runner(full, model_id, data)
            final_trials = [refine(np.array(r["coordinates"]), 120) for r in chosen]
            valid = [r for r in final_trials if r["success"]]
            if not valid:
                raise RuntimeError(f"No converged full-grid refinement for model {model_id}.")
            best = min(valid, key=lambda r: r["residual_rms"])
            p = np.array(best["coordinates"])
            best["spectrum_inputs"] = {k: float(v) for k, v in inputs(full, model_id, p).items()}
            best["n2_column_cm2"] = float(np.exp(p[0]) * 1e14)
            best["baseline_fraction"] = float(p[5] * 1e-4)
            entry = dict(label=LABELS[model_id], initial_coordinates=[p.tolist() for p in initial],
                         lower_bounds=bounds(model_id)[0].tolist(), upper_bounds=bounds(model_id)[1].tolist(),
                         coarse_trials=trials, full_trials=final_trials, best=best)
            if model_id == 3:
                assert 0 <= best["spectrum_inputs"]["excited_fraction"] <= 1
                if best["spectrum_inputs"]["mass_loss_g_s"] > 1e14:
                    limited = fit_runner(full, model_id, data, mass_loss_max=1e14)
                    seed = p.copy()
                    seed[1] = min(seed[1], np.log(10.0))
                    entry["mass_loss_cap_1e14"] = limited(seed, 160)
            report["models"][str(model_id)] = entry
            (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    spectra, atmosphere_profiles = {}, {}
    profile_columns = {"radius_rp": np.asarray(full.radius_rp)}
    for model_id in LABELS:
        key = str(model_id)
        p = np.array(report["models"][key]["best"]["coordinates"])
        spectra[key] = np.asarray(evaluate(full, model_id, p))
        atmosphere_profiles[key] = {k: np.asarray(v) for k, v in profiles(full, model_id, p).items()}
        atmosphere_profiles[key]["temperature_k"] = np.asarray(full.temperature_profile(p[2] * 1e4))
        profile_columns.update({f"model{key}_{k}": v for k, v in atmosphere_profiles[key].items()})
    if args.check_convergence:
        report["convergence"] = {}
        for label, grids in (("double_layers", dict(nlayer=128)), ("double_disk", dict(nlon=48, nlat=24))):
            finer = HydrogenForwardModel(**grids)
            for model_id in LABELS:
                p = np.array(report["models"][str(model_id)]["best"]["coordinates"])
                difference = np.asarray(evaluate(finer, model_id, p)) - spectra[str(model_id)]
                report["convergence"].setdefault(str(model_id), {})[label] = float(np.max(np.abs(difference)))
        report["convergence_note"] = "Fitted coordinates, including N2, fixed on finer grids; model3 f23 is derived on each grid."
    (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(args.output / "spectra.csv", np.column_stack([full.velocity_kms, *spectra.values()]),
               delimiter=",", header="velocity_kms,model1,model2,model3", comments="")
    np.savetxt(args.output / "profiles.csv", np.column_stack(list(profile_columns.values())),
               delimiter=",", header=",".join(profile_columns), comments="")
    # The plot helper is defined below to keep the diagnostic self-contained.
    plot_results(full, data, spectra, atmosphere_profiles, report, args.output / "three_models.png")
    print(json.dumps({k: v["best"] for k, v in report["models"].items()}, indent=2))


def plot_results(model, data, spectra, profiles, report, destination):
    """Plot fitted spectra and the atmospheric structure of all three models."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from exojax.utils.constants import hcperk

    styles = {
        "1": ("Model 1", "C0", "--"),
        "2": ("Model 2", "C1", "-."),
        "3": ("Model 3", "C3", "-"),
    }
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.55])
    temperature_ax = fig.add_subplot(grid[0, 0])
    density_ax = fig.add_subplot(grid[1, 0], sharex=temperature_ax)
    velocity_ax = fig.add_subplot(grid[2, 0], sharex=temperature_ax)
    spectrum_ax = fig.add_subplot(grid[:2, 1])
    residual_ax = fig.add_subplot(grid[2, 1], sharex=spectrum_ax)
    observed_velocity, observed, error = data
    velocity = np.asarray(model.velocity_kms)
    order = np.argsort(velocity)
    radius = np.asarray(model.radius_rp)
    energy_k = hcperk * model.line_center_cm
    weight_ratio = float(model.adb.gupper[0] / model.adb.glower[0])

    spectrum_ax.errorbar(
        observed_velocity, 100 * observed, yerr=100 * error, fmt=".",
        color="black", markersize=4, elinewidth=0.7, alpha=0.65,
        label="Keck/KPF observations", zorder=5,
    )
    for key, (label, color, linestyle) in styles.items():
        profile = profiles[key]
        with np.errstate(divide="ignore", invalid="ignore"):
            excitation = energy_k / np.log(weight_ratio / profile["ratio"])
        temperature_ax.plot(
            radius, profile["temperature_k"], color=color, ls=":", lw=1.3,
        )
        temperature_ax.plot(
            radius, excitation, color=color, ls=linestyle, label=label,
        )
        density_ax.semilogy(
            radius, profile["n2_cm3"], color=color, ls=linestyle,
        )
        velocity_ax.plot(
            radius, profile["radial_velocity_kms"], color=color, ls=linestyle,
        )
        rms = report["models"][key]["best"]["residual_rms"]
        spectrum = np.asarray(spectra[key])
        spectrum_ax.plot(
            velocity, 100 * spectrum, color=color, ls=linestyle,
            label=f"{label}; RMS {rms:.2f}",
        )
        sampled = np.interp(observed_velocity, velocity[order], spectrum[order])
        residual_ax.plot(
            observed_velocity, (sampled - observed) / error,
            color=color, ls=linestyle, lw=1.1, marker=".", markersize=2.5,
        )

    handles, labels = temperature_ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.3", ls=":", lw=1.3))
    labels.append("Kinetic T (dotted)")
    temperature_ax.legend(handles, labels, fontsize=8, ncol=2, loc="upper right")
    temperature_ax.set(title="Kinetic and excitation temperatures", ylabel="Temperature [K]")
    density_ax.set(title="Hydrogen n=2 density", ylabel="Number density [cm$^{-3}$]")
    velocity_ax.set(title="Parker radial velocity", ylabel="Velocity [km/s]", xlabel="Radius / Rp")
    for axis in (temperature_ax, density_ax, velocity_ax):
        axis.set_xlim(1, 3)
    for axis in (temperature_ax, density_ax):
        axis.tick_params(labelbottom=False)
    spectrum_ax.set(title="Observed spectrum and fitted models", ylabel="Line contrast [%]")
    spectrum_ax.legend(fontsize=9, loc="upper left")
    spectrum_ax.tick_params(labelbottom=False)
    residual_ax.axhspan(-1, 1, color="0.5", alpha=0.1)
    residual_ax.axhline(0, color="0.5", lw=0.8)
    residual_ax.set(
        xlim=(-105, 105), xlabel="Velocity from NIST H-alpha [km/s]",
        ylabel="(Model - data) / error",
    )
    fig.suptitle("Fits of three population models; all radial winds follow Parker solutions", fontsize=14)
    fig.supxlabel(
        "Common mu=2/3; model phase 0.375; data phases 0.30-0.45. "
        "Local least squares; RMS is diagnostic, not a calibrated posterior statistic.\n"
        "Data: Yapeng Zhang (2026), doi:10.5281/zenodo.20707255; CC BY 4.0.",
        fontsize=8,
    )
    fig.savefig(destination, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
