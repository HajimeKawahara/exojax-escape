"""Fit within-night log-spectrum changes without constructing a nightly master.

The archived ratio 1+r contains a shared multiplicative nightly reference.
One common log value per native wavelength is profiled analytically. Planet
parameters are shared or phase-dependent, with independent night prediction
checks. Supplied diagonal errors are a diagnostic proxy, not archive covariance.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from escape.observation import C_KMS
from escape.time_contrast import profile_time_residuals
from cloudy_halpha import stellar_reference
from phase_halpha import PHASE_EDGES, REST_ANGSTROM


def load_time_data(paths, phase_range=(0.30, 0.435), stellar_range=(110.0, 390.0)):
    """Select a fixed native stellar window before any fitting or de-orbiting."""
    groups = []
    for night, path in enumerate(paths):
        with h5py.File(path) as source:
            phase, wave, residual, error = (
                source[key][:] for key in ("phase", "wave", "flux", "err")
            )
        keep_time = (phase >= phase_range[0]) & (phase < phase_range[1])
        coordinate = C_KMS * (wave / REST_ANGSTROM - 1)
        keep_pixel = (coordinate >= stellar_range[0]) & (coordinate <= stellar_range[1])
        ratio = 1 + residual[keep_time][:, keep_pixel]
        errors = error[keep_time][:, keep_pixel]
        usable = np.isfinite(ratio) & (ratio > 0) & np.isfinite(errors) & (errors > 0)
        pixel_support = usable.sum(axis=0) >= 2
        ratio, errors, usable = (
            value[:, pixel_support] for value in (ratio, errors, usable)
        )
        log_data = np.log(np.where(usable, ratio, 1.0))
        weight = np.where(usable, (ratio / np.where(usable, errors, 1.0)) ** 2, 0.0)
        groups.append(
            dict(
                night=night,
                label=Path(path).stem,
                phase=phase[keep_time],
                stellar_velocity=coordinate[keep_pixel][pixel_support],
                order=np.broadcast_to(np.arange(len(wave))[:, None], wave.shape)[
                    keep_pixel
                ][pixel_support],
                log_data=log_data,
                weight=weight,
                source=str(path),
                input_hash=sha256(Path(path).read_bytes()).hexdigest(),
            )
        )
    return groups


def select_times(group, keep):
    """Change only the retained exposure set; native columns remain fixed."""
    result = dict(group)
    for key in ("phase", "log_data", "weight"):
        result[key] = group[key][keep]
    return result


def make_native_flux(model_velocity_kms, surface, group):
    """Evaluate continuum-normalized flux at the native pixels before logarithms."""
    velocity = jnp.asarray(model_velocity_kms)
    coordinate = group["stellar_velocity"]
    native_star = jnp.asarray(stellar_reference(coordinate, surface, 110.0))
    normalization_grid = np.linspace(-600.0, 600.0, 2401)
    normalization_grid = normalization_grid[np.abs(normalization_grid) > 400]
    normalization_star = jnp.asarray(
        stellar_reference(normalization_grid, surface, 110.0)
    )
    orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
    native_query = jnp.asarray(
        C_KMS * ((1 + coordinate[None, :] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1)
    )
    continuum_query = jnp.asarray(
        C_KMS
        * (
            (1 + normalization_grid[None, :] / C_KMS) / (1 + orbital[:, None] / C_KMS)
            - 1
        )
    )

    @jax.jit
    def flux(spectra, offset_kms):
        offsets = jnp.broadcast_to(offset_kms, (len(group["phase"]),))
        shifted = jax.vmap(lambda row, rv: jnp.interp(velocity - rv, velocity, row))(
            spectra, offsets
        )
        # Constant endpoint extension is the declared continuum approximation.
        native_planet = jax.vmap(lambda q, row: jnp.interp(q, velocity, row))(
            native_query, shifted
        )
        continuum_planet = jax.vmap(lambda q, row: jnp.interp(q, velocity, row))(
            continuum_query, shifted
        )
        normalization = (normalization_star[None, :] + continuum_planet).mean(axis=1)
        return (native_star[None, :] + native_planet) / normalization[:, None]

    return flux


def make_time_objective(groups, model_velocity_kms, surface, intrinsic, *, drift=False):
    """Return a JAX objective and native model-flux callback for supplied spectra.

    ``intrinsic(parameters, phase_array)`` returns spectra(E,V), offsets(E).
    Optional drift profiles a time-linear grey term and stellar-wavelength
    slope per night; it is a declared sensitivity control, not a calibration.
    """
    fluxes = [make_native_flux(model_velocity_kms, surface, group) for group in groups]
    bases = []
    for group in groups:
        if not drift:
            bases.append(None)
            continue
        time = group["phase"] - np.mean(group["phase"])
        time /= max(np.ptp(group["phase"]), 1e-12)
        coordinate = (group["stellar_velocity"] - 250) / 140
        patterns = [
            np.broadcast_to(time[:, None], group["log_data"].shape),
            time[:, None] * coordinate[None, :],
        ]
        columns = np.stack(
            [
                np.asarray(
                    profile_time_residuals(
                        pattern, np.zeros_like(pattern), group["weight"]
                    ).residual
                ).ravel()
                for pattern in patterns
            ],
            axis=1,
        )
        q, r = np.linalg.qr(columns)
        rank = np.sum(np.abs(np.diag(r)) > 1e-10)
        bases.append(jnp.asarray(q[:, :rank]))

    def native_models(parameters):
        return tuple(
            flux(*intrinsic(parameters, jnp.asarray(group["phase"])))
            for flux, group in zip(fluxes, groups)
        )

    @jax.jit
    def objective(parameters):
        values = []
        for group, model, basis in zip(groups, native_models(parameters), bases):
            projected = profile_time_residuals(
                group["log_data"], jnp.log(model), group["weight"]
            ).residual.ravel()
            if basis is not None:
                projected = projected - basis @ (basis.T @ projected)
            values.append(projected)
        return jnp.concatenate(values)

    return objective, jax.jit(native_models)


def saved_intrinsic(model, mode, active_bins=(0, 1, 2)):
    """Build a shared or supported phase-parameter callback from saved spectra."""
    phases = jnp.asarray(model["phase"])
    spectra = jnp.asarray(model["fine_intrinsic_phases"])
    rv0 = float(model["best_rv_kms"])
    indices = jnp.asarray(active_bins)

    def intrinsic(parameters, exposure_phase):
        sampled = jax.vmap(
            lambda column: jnp.interp(exposure_phase, phases, column),
            in_axes=1,
            out_axes=1,
        )(spectra)
        if mode == "shared":
            rv, amplitude = parameters[0], parameters[1]
        else:
            count = len(active_bins)
            rvs = jnp.full(3, rv0).at[indices].set(parameters[:count])
            amplitudes = jnp.ones(3).at[indices].set(parameters[count:])
            phase_bin = jnp.clip(
                jnp.searchsorted(jnp.asarray(PHASE_EDGES), exposure_phase, side="right")
                - 1,
                0,
                2,
            )
            rv, amplitude = rvs[phase_bin], amplitudes[phase_bin]
        return sampled * jnp.broadcast_to(amplitude, exposure_phase.shape)[
            :, None
        ], jnp.broadcast_to(rv, exposure_phase.shape)

    return intrinsic


def fit_objective(objective, initial, lower, upper):
    """Optimize with JAX Jacobians and retain a local finite-difference check."""
    jacobian = jax.jit(jax.jacfwd(objective))
    initial = np.asarray(initial)
    rv_count = len(initial) // 2
    seeds = [initial, np.r_[initial[:rv_count] - 2.0, initial[rv_count:]]]
    fits = [
        least_squares(
            objective,
            seed,
            jac=jacobian,
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=100,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-7,
        )
        for seed in seeds
    ]
    result = min(fits, key=lambda item: np.dot(item.fun, item.fun))
    probe = result.x.copy()
    probe[:rv_count] += 0.017
    automatic = np.asarray(jacobian(probe))
    step = 1e-4
    finite = np.column_stack(
        [
            (
                np.asarray(objective(probe + step * axis))
                - np.asarray(objective(probe - step * axis))
            )
            / (2 * step)
            for axis in np.eye(len(initial))
        ]
    )
    record = dict(
        parameters=result.x.tolist(),
        chi2=float(result.fun @ result.fun),
        success=bool(result.success),
        message=str(result.message),
        derivative_max_scaled_error=float(
            np.max(abs(automatic - finite) / (1 + abs(finite)))
        ),
        boundary_hit=bool(
            np.any(result.x - np.asarray(lower) < 0.02)
            or np.any(np.asarray(upper) - result.x < 0.02)
        ),
    )
    return record


def plot_summary(report, output):
    """Show fitted time variation and prediction checks without error claims."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {
        (row["selection"], row["mode"], row["drift"]): row for row in report["results"]
    }
    shared = results["pooled", "shared", False]
    phase = results["pooled", "phase", False]
    shared_drift = results["pooled", "shared", True]
    phase_drift = results["pooled", "phase", True]
    bins = np.asarray(phase["active_phase_bins"])
    upper = np.minimum(PHASE_EDGES[1:], report["science_phase_range"][1])
    centers = ((upper + PHASE_EDGES[:-1]) / 2)[bins]
    count = len(bins)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9))
    for row, label, color in [
        (phase, "Phase coefficients", "#c55e36"),
        (phase_drift, "+ smooth time nuisance", "#2b797b"),
    ]:
        parameters = np.asarray(row["parameters"])
        axes[0].plot(centers, parameters[count:], "o-", label=label, color=color)
        axes[1].plot(centers, parameters[:count], "o-", color=color)
    for axis, index in zip(axes[:2], [1, 0]):
        axis.axhline(
            shared["parameters"][index], color=".4", ls="--", label="Shared fit"
        )
        axis.set(xlabel="Orbital phase", xticks=centers)
        axis.grid(alpha=0.2)
    axes[0].set(ylabel="Amplitude relative to fixed atmosphere")
    axes[1].set(ylabel="Velocity offset (km/s)")
    axes[0].legend(fontsize=8, frameon=False)
    labels = ["Pooled", "Pooled\n+ nuisance", "Aug 8\ntraining", "Aug 20\nheld out"]
    train_shared = results["night0", "shared", False]
    train_phase = results["night0", "phase", False]
    gains = [
        shared["chi2"] - phase["chi2"],
        shared_drift["chi2"] - phase_drift["chi2"],
        train_shared["chi2"] - train_phase["chi2"],
        train_shared["held_out"]["chi2"] - train_phase["held_out"]["chi2"],
    ]
    axes[2].bar(labels, gains, color=["#c55e36", "#2b797b", "#c55e36", "#71678e"])
    axes[2].axhline(0, color=".4", lw=0.8)
    axes[2].set(ylabel="Score gain from phase parameters")
    axes[2].tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "Within-night log contrasts: the unknown common master cancels", fontsize=12
    )
    fig.text(
        0.5,
        0.015,
        "Fixed native pixels; diagonal error proxy only. Positive gain favors phase coefficients; no significance claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output / "time_invariants.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs=2, type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stellar-surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase-max",
        type=float,
        default=0.435,
        help="Exclusive upper science phase; default avoids stellar overlap of a 3 Rp region including 150 s exposures.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    model, surface = dict(np.load(args.model)), dict(np.load(args.stellar_surface))
    groups = load_time_data(args.data, phase_range=(0.30, args.phase_max))
    if any(
        group["phase"].min() < model["phase"].min()
        or group["phase"].max() > model["phase"].max()
        for group in groups
    ):
        raise ValueError(
            "Saved phase spectra must cover every selected science exposure."
        )
    rv0 = float(model["best_rv_kms"])
    records, arrays = [], {}
    reference_objective, _ = make_time_objective(
        groups, model["model_velocity_kms"], surface, saved_intrinsic(model, "shared")
    )
    reference = np.asarray(reference_objective(jnp.array([rv0, 1.0])))
    no_planet = np.asarray(reference_objective(jnp.array([rv0, 0.0])))
    for selection, selected in [
        ("pooled", groups),
        ("night0", [groups[0]]),
        ("night1", [groups[1]]),
    ]:
        active_bins = tuple(
            sorted(
                set(
                    np.clip(
                        np.searchsorted(
                            PHASE_EDGES,
                            np.concatenate([g["phase"] for g in selected]),
                            side="right",
                        )
                        - 1,
                        0,
                        2,
                    )
                )
            )
        )
        for mode in ["shared", "phase"]:
            for drift in [False, True] if selection == "pooled" else [False]:
                count = 1 if mode == "shared" else len(active_bins)
                intrinsic = saved_intrinsic(model, mode, active_bins)
                objective, native = make_time_objective(
                    selected,
                    model["model_velocity_kms"],
                    surface,
                    intrinsic,
                    drift=drift,
                )
                result = fit_objective(
                    objective,
                    [rv0] * count + [1.0] * count,
                    [-15.0] * count + [0.0] * count,
                    [15.0] * count + [4.0] * count,
                )
                result.update(
                    selection=selection,
                    mode=mode,
                    drift=drift,
                    active_phase_bins=[int(x) for x in active_bins],
                    drift_nuisance_rank=2 * len(selected) if drift else 0,
                    supported_time_rank=int(
                        sum(
                            np.maximum((g["weight"] > 0).sum(axis=0) - 1, 0).sum()
                            for g in selected
                        )
                    ),
                )
                name = f"{selection}_{mode}" + ("_drift" if drift else "")
                records.append(result)
                for n, (group, flux) in enumerate(
                    zip(selected, native(jnp.asarray(result["parameters"])))
                ):
                    # Compact per-exposure contrast metrics retain time dependence.
                    arrays[f'{name}_night{group["night"]}_log_model'] = np.asarray(
                        jnp.log(flux)
                    )
                if selection != "pooled":
                    train = selected[0]
                    other = groups[1 - train["night"]]
                    low, high = (
                        max(train["phase"].min(), other["phase"].min()),
                        min(train["phase"].max(), other["phase"].max()),
                    )
                    test = select_times(
                        other, (other["phase"] >= low) & (other["phase"] <= high)
                    )
                    held, _ = make_time_objective(
                        [test], model["model_velocity_kms"], surface, intrinsic
                    )
                    held_residual = np.asarray(held(jnp.asarray(result["parameters"])))
                    result["held_out"] = dict(
                        night=test["night"],
                        phase_range=[float(low), float(high)],
                        exposures=len(test["phase"]),
                        chi2=float(held_residual @ held_residual),
                        rank=int(
                            np.maximum((test["weight"] > 0).sum(axis=0) - 1, 0).sum()
                        ),
                        scope="Planet parameters frozen; a common native log-spectrum is profiled on the held-out night for scoring. Only common phase coverage is scored.",
                    )
                print(json.dumps(result), flush=True)
                (args.output / "progress.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
    report = dict(
        scope=__doc__,
        stellar_velocity_window_kms=[110.0, 390.0],
        science_phase_range=[0.3, args.phase_max],
        groups=[
            dict(
                night=g["night"],
                exposures=len(g["phase"]),
                wavelength_columns=len(g["stellar_velocity"]),
                retained_samples=int(np.sum(g["weight"] > 0)),
                contrast_rank=int(
                    np.maximum((g["weight"] > 0).sum(axis=0) - 1, 0).sum()
                ),
            )
            for g in groups
        ],
        results=records,
        fixed_reference=dict(parameters=[rv0, 1.0], chi2=float(reference @ reference)),
        no_planet=dict(chi2=float(no_planet @ no_planet)),
        limitations=[
            "This profiles arbitrary time-constant spectra separately for each night/order/native wavelength; absolute spectral levels are not recovered.",
            "Log errors use a first-order transformation of supplied residual errors. Missing extraction, original-master and cross-order covariance prevent likelihood calibration.",
            "No nightly master is constructed; master-only late phases are not needed under the shared multiplicative-master assumption.",
            "Per-exposure normalization and time-dependent stellar/calibration terms remain. The optional two drift modes are sensitivity controls.",
            "Shared and phase fits use exactly the same fixed native pixel window. Held-out comparisons use common phase coverage and do not infer unobserved phase parameters.",
            "The atmosphere and velocity field are fixed; phase amplitude coefficients are not new thermal or level-population solutions.",
                "The primary midpoint phase cutoff 0.435 excludes stellar overlap of a 3 Rp emitting region including 150 s exposures under the adopted eclipse geometry/phase convention; a larger window is a sensitivity check with occultation omitted.",
        ],
        sources={
            str(p): sha256(p.read_bytes()).hexdigest()
            for p in [*args.data, args.model, args.stellar_surface, Path(__file__)]
        },
    )
    for g in groups:
        for key in ["phase", "stellar_velocity", "order", "log_data", "weight"]:
            arrays[f'night{g["night"]}_{key}'] = g[key]
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    (args.output / "review.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot_summary(report, args.output)


if __name__ == "__main__":
    main()
