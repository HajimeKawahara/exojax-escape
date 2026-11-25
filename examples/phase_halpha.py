"""Fit saved H-alpha spectra by phase, retaining each complete nightly master.

Inputs are a saved atmosphere's phase spectra, two Zenodo residual HDF5 files,
and a local PHOENIX surface-flux NPZ. No atmosphere is solved. The archived
observations are already residuals: only the model undergoes master division.
Use --plot-only on the compact output NPZ to regenerate the explanation plot.
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

from escape.observation import C_KMS, process_phase_spectra
from escape.phase_observation import coadd_residual_pixels, sample_stellar_residuals
from cloudy_halpha import NIST_ROW, stellar_reference


PHASE_EDGES = np.array([0.30, 0.35, 0.40, 0.45])
OBSERVATIONS = Path(__file__).resolve().parents[1] / "data/zenodo_20707255/KELT9b_w6565_avg_p0.30-0.45.dat"
REST_ANGSTROM = 1e8 / NIST_ROW["wav"][0]


def load_pixels(paths, observed_wavelength):
    """Preserve all exposures; select native pixels only for the final bins."""
    edges = np.r_[observed_wavelength[0] - np.diff(observed_wavelength)[0] / 2,
                  (observed_wavelength[:-1] + observed_wavelength[1:]) / 2,
                  observed_wavelength[-1] + np.diff(observed_wavelength)[-1] / 2]
    phase, nights = [], []
    exposure, coordinate, value, weight, group = [], [], [], [], []
    bins = len(observed_wavelength)
    for night, path in enumerate(paths):
        with h5py.File(path) as source:
            ph, wave, flux, err = (source[key][:] for key in ("phase", "wave", "flux", "err"))
        if wave.ndim != 2 or flux.shape != err.shape or flux.shape != (len(ph), *wave.shape):
            raise ValueError("Expected wave(order,pixel) and flux/err(exposure,order,pixel).")
        offset = len(phase)
        phase.extend(ph)
        nights.extend([night] * len(ph))
        for index, phi in enumerate(ph):
            phase_bin = np.searchsorted(PHASE_EDGES, phi, side="right") - 1
            if not 0 <= phase_bin < 3:
                continue
            orbital = 244.8 * np.sin(2 * np.pi * phi)
            for order in range(len(wave)):
                stellar_velocity = C_KMS * (wave[order] / REST_ANGSTROM - 1)
                bin_index = np.searchsorted(edges, wave[order] / (1 + orbital / C_KMS), side="right") - 1
                keep = ((bin_index >= 0) & (bin_index < bins)
                        & (np.abs(stellar_velocity) >= 110) & (np.abs(stellar_velocity) < 599)
                        & np.isfinite(flux[index, order]) & np.isfinite(err[index, order])
                        & (err[index, order] > 0))
                exposure.extend(np.full(keep.sum(), offset + index))
                coordinate.extend(stellar_velocity[keep])
                value.extend(flux[index, order, keep])
                weight.extend(1 / err[index, order, keep] ** 2)
                group.extend((phase_bin * len(paths) + night) * bins + bin_index[keep])
    return {key: np.asarray(values) for key, values in dict(
        phase=phase, night=nights, exposure=exposure, coordinate=coordinate,
        value=value, weight=weight, group=group).items()}


def make_predictor(model, surface, pixels, output_velocity):
    """Use the existing JAX operator, then sample the same observed native pixels."""
    velocity = jnp.asarray(model["model_velocity_kms"])
    phase = pixels["phase"]
    source_phase = model["phase"]
    if phase.min() < source_phase.min() or phase.max() > source_phase.max():
        raise ValueError("Saved spectra must cover ALL exposures, including master-only phases.")
    sampled = jnp.asarray(np.stack([
        np.interp(phase, source_phase, column) for column in model["fine_intrinsic_phases"].T
    ], axis=1))
    stellar_velocity = np.linspace(-600., 600., 2401)
    reference = stellar_reference(stellar_velocity, surface, 110.)
    phase_bin = np.searchsorted(PHASE_EDGES, phase, side="right") - 1
    active = (phase_bin >= 0) & (phase_bin < 3)
    phase_bin = jnp.asarray(np.clip(phase_bin, 0, 2))
    rv0 = float(model["best_rv_kms"])
    count = len(phase)
    shape = (count, len(stellar_velocity))
    # Integer interpolation nodes map each already-sampled exposure exactly.
    # Physical phases have already been used above; no exposure is dropped.
    nodes = np.arange(count, dtype=float)
    if count < 2:
        raise ValueError("This phase example needs at least two exposures.")
    operator = dict(
        model_phase=nodes, exposure_phase=nodes,
        model_velocity_kms=np.r_[-1000., velocity, 1000.],
        orbital_velocity_kms=244.8 * np.sin(2 * np.pi * phase),
        night_index=pixels["night"], stellar_velocity_kms=stellar_velocity,
        output_velocity_kms=output_velocity, stellar_reference=reference,
        continuum_weights=np.broadcast_to(np.abs(stellar_velocity) > 400, shape),
        master_weights=np.ones(shape), stellar_keep=np.ones(shape, dtype=bool),
        coadd_weights=np.zeros((count, len(output_velocity))),
    )
    # The operator's final coadd is unused: the native-pixel mask and weights
    # below are identical for the data and model. Master weights remain nonzero.
    num_bins = 3 * len(np.unique(pixels["night"])) * len(output_velocity)

    @jax.jit
    def observation_result(rv, amplitude):
        shifts = jnp.where(active, rv[phase_bin], rv0)
        scales = jnp.where(active, amplitude[phase_bin], 1.)
        adjusted = jax.vmap(lambda row, shift, scale: jnp.interp(
            velocity - shift, velocity, row) * scale)(sampled, shifts, scales)
        # Explicit constant endpoint extension retains the modeled continuum.
        padded = jnp.column_stack((adjusted[:, 0], adjusted, adjusted[:, -1]))
        return process_phase_spectra(model_contrast=padded, **operator)

    @jax.jit
    def exposures(rv, amplitude):
        return observation_result(rv, amplitude).stellar_residual

    @jax.jit
    def predict(rv, amplitude):
        values = sample_stellar_residuals(
            stellar_velocity, exposures(rv, amplitude), pixels["exposure"], pixels["coordinate"])
        return coadd_residual_pixels(values, pixels["group"], pixels["weight"], num_bins=num_bins).contrast

    return predict, exposures, stellar_velocity, rv0, observation_result


def fit_cases(predict, data, weight, rv0, n_nights, n_velocity):
    """Conditional least squares, with JAX derivatives of the observation path."""
    valid = np.isfinite(data) & (weight > 0)
    target, sqrt_weight = jnp.asarray(data[valid]), jnp.sqrt(weight[valid])
    night_index = np.tile(np.repeat(np.arange(n_nights), n_velocity), 3)
    results, curves = [], {}
    for kind in ("common_rv", "phase_rv", "common_rv_amplitude", "phase_rv_amplitude"):
        for offsets in (False, True):
            nr = 1 if kind.startswith("common") else 3
            na = nr if kind.endswith("amplitude") else 0
            initial = np.array([rv0] * nr + [1.] * na + ([0.] * n_nights if offsets else []))
            lower = [-15.] * nr + [0.] * na + ([-3000.] * n_nights if offsets else [])
            upper = [15.] * nr + [4.] * na + ([3000.] * n_nights if offsets else [])

            def unpack(parameters):
                rv = jnp.repeat(parameters[:nr], 3) if nr == 1 else parameters[:nr]
                amplitude = parameters[nr:nr + na] if na else jnp.ones(3)
                if na == 1:
                    amplitude = jnp.repeat(amplitude, 3)
                zero = parameters[-n_nights:] if offsets else jnp.zeros(n_nights)
                return rv, amplitude, zero

            def prediction(parameters):
                rv, amplitude, zero = unpack(parameters)
                return predict(rv, amplitude) + 1e-6 * zero[night_index]

            objective = jax.jit(lambda parameters: (prediction(parameters)[valid] - target) * sqrt_weight)
            jacobian = jax.jit(jax.jacfwd(objective))
            fits = [least_squares(objective, seed, jac=jacobian, bounds=(lower, upper),
                                  x_scale="jac", max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-7)
                    for seed in (initial, np.r_[initial[:nr] - 3., initial[nr:]])]
            fitted = min(fits, key=lambda fit: np.dot(fit.fun, fit.fun))
            rv, amplitude, zero = map(np.asarray, unpack(fitted.x))
            name = kind + ("_offsets" if offsets else "")
            curves[name] = np.asarray(prediction(fitted.x)).reshape(3, n_nights, n_velocity)
            result = dict(name=name, chi2=float(np.dot(fitted.fun, fitted.fun)),
                          parameters=len(initial), success=bool(fitted.success),
                          rv_kms=rv.tolist(), amplitudes=amplitude.tolist(), night_offset_ppm=zero.tolist(),
                          boundary_hit=bool(np.any(fitted.x - lower < 0.02) or np.any(upper - fitted.x < 0.02)))
            # Check away from interpolation knots; this is not atmosphere AD.
            probe = fitted.x.copy()
            probe[:nr] += 0.017
            step = 1e-4
            finite = np.column_stack([(np.asarray(objective(probe + step * axis))
                                       - np.asarray(objective(probe - step * axis))) / (2 * step)
                                      for axis in np.eye(len(initial))])
            auto = np.asarray(jacobian(probe))
            result["derivative_max_scaled_error"] = float(np.max(np.abs(finite - auto) / (1 + np.abs(finite))))
            result["derivative_probe"] = probe.tolist()
            results.append(result)
            print(json.dumps(result), flush=True)
    return results, curves


def plot_arrays(path, output):
    """Plot compact derived arrays without the source spectra or HDF5 files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(path) as source:
        arrays = dict(source)
    velocity, weights = arrays["velocity_kms"], arrays["weight"]
    total = weights.sum(axis=1)

    def combined(value):
        return np.divide(np.sum(np.nan_to_num(value) * weights, axis=1), total,
                         out=np.full_like(total, np.nan), where=total > 0)

    data = combined(arrays["data"])
    error = np.divide(1., np.sqrt(total), out=np.full_like(total, np.nan), where=total > 0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [2, 1]})
    for phase_bin in range(3):
        ax = axes[0, phase_bin]
        ax.errorbar(velocity, 1e6 * data[phase_bin], yerr=1e6 * error[phase_bin], fmt=".",
                    color="0.4", ms=3, elinewidth=.5, label="Re-binned archived residuals")
        for name, style, color, label in [
            ("common_rv", "--", "0.5", "One RV, fixed strength"),
            ("phase_rv", "-", "C0", "Per-phase RV"),
            ("phase_rv_amplitude", "-", "C1", "Per-phase RV + strength"),
        ]:
            ax.plot(velocity, 1e6 * combined(arrays[name])[phase_bin], ls=style, color=color, label=label)
        ax.set_title(f"Phase {PHASE_EDGES[phase_bin]:.2f}-{PHASE_EDGES[phase_bin + 1]:.2f}")
        for night in range(weights.shape[1]):
            take = weights[phase_bin, night] > 0
            delta = arrays["data"][phase_bin, night] - arrays["phase_rv_amplitude"][phase_bin, night]
            axes[1, phase_bin].errorbar(velocity[take], 1e6 * delta[take],
                                       yerr=1e6 / np.sqrt(weights[phase_bin, night, take]),
                                       fmt=".", ms=3, elinewidth=.5, label=f"Night {night + 1}")
        for row in axes[:, phase_bin]:
            row.axhline(0, color="k", lw=.4)
            row.grid(alpha=.2)
            row.set_xlim(-95, 95)
        axes[1, phase_bin].set_xlabel("Planet-rest velocity (km/s)")
    axes[0, 0].set_ylabel("Residual contrast (ppm)")
    axes[1, 0].set_ylabel("Data - per-phase fit\n(ppm)")
    axes[0, 0].legend(fontsize=7)
    axes[1, -1].legend(fontsize=8)
    fig.suptitle("Fixed atmosphere; fits retain night information and all exposures in each master")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs=2, help="Two residual HDF5 files, in night order")
    parser.add_argument("--model", type=Path, help="NPZ: phase, model_velocity_kms, fine_intrinsic_phases, best_rv_kms")
    parser.add_argument("--stellar-surface", type=Path, help="Local PHOENIX frequency_hz/surface_flux_nu NPZ")
    parser.add_argument("--observations", type=Path, default=OBSERVATIONS, help="Published coadd used ONLY for wavelength-bin centers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-only", type=Path, help="Derived phase_arrays.npz; other inputs not required")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_arrays(args.plot_only, args.output / "phase_fit.png")
        return
    if not args.data or not args.model or not args.stellar_surface:
        parser.error("Provide --data, --model and --stellar-surface, or use --plot-only.")
    jax.config.update("jax_enable_x64", True)
    model, surface = dict(np.load(args.model)), dict(np.load(args.stellar_surface))
    wavelength = np.loadtxt(args.observations)[:, 0]
    output_velocity = C_KMS * (wavelength / REST_ANGSTROM - 1)
    pixels = load_pixels(args.data, wavelength)
    n_nights, n_velocity = len(args.data), len(wavelength)
    result = coadd_residual_pixels(pixels["value"], pixels["group"], pixels["weight"],
                                  num_bins=3 * n_nights * n_velocity)
    data, weight = np.asarray(result.contrast), np.asarray(result.weight)
    predict, exposures, stellar_velocity, rv0, observation_result = make_predictor(model, surface, pixels, output_velocity)
    reference = np.asarray(exposures(jnp.full(3, rv0), jnp.ones(3)))
    reference_pixels = sample_stellar_residuals(
        stellar_velocity, reference, pixels["exposure"], pixels["coordinate"])
    if not np.isfinite(reference_pixels).all():
        raise ValueError("The model must cover every selected observed pixel.")
    reference_coadd = coadd_residual_pixels(reference_pixels, pixels["group"], pixels["weight"],
                                          num_bins=3 * n_nights * n_velocity)
    np.testing.assert_array_equal(reference_coadd.weight, weight)
    results, curves = fit_cases(predict, data, weight, rv0, n_nights, n_velocity)
    perturbed = np.asarray(exposures(jnp.full(3, rv0), jnp.array([1.5, 1., .8])))
    baseline_result = observation_result(jnp.full(3, rv0), jnp.ones(3))
    changed_result = observation_result(jnp.full(3, rv0), jnp.array([1.5, 1., .8]))
    fixed_master = np.asarray(changed_result.normalized_stellar_flux / baseline_result.nightly_master - 1)
    np.savez_compressed(args.output / "processed_exposures.npz", phase=pixels["phase"], night=pixels["night"],
                        planet_velocity_kms=244.8 * np.sin(2 * np.pi * pixels["phase"]),
                        stellar_velocity_kms=stellar_velocity, reference_residuals=reference,
                        perturbed_residuals=perturbed, fixed_master_residuals=fixed_master)
    np.savez_compressed(args.output / "phase_arrays.npz", velocity_kms=output_velocity,
                        data=data.reshape(3, n_nights, n_velocity), weight=weight.reshape(3, n_nights, n_velocity), **curves)
    report = dict(scope="Conditional fixed-atmosphere phase/night diagnostic; not a calibrated likelihood or atmosphere fit",
                  exposures=len(pixels["phase"]), samples=int(np.sum(weight > 0)), native_pixels=len(pixels["value"]),
                  preserved_full_nightly_master=True, identical_data_model_pixel_support=True,
                  gradient_method="JAX observer Jacobian, checked with central differences away from interpolation knots",
                  results=results, limits=[
                      "Archived HDF5 is already residuals; no second stellar/master division is applied to observations.",
                      "Fixed circular de-orbiting, native bins, and supplied independent pixel errors are a processing proxy.",
                      "Exact author coaddition, frame metadata and shared-master/extraction covariance remain unresolved.",
                      "The contrast multiplier includes line and continuum, after phase projection and before master division.",
                      "Master-only late phases retain reference strength/RV; eclipse and coupled horizontal structure are not solved.",
                      "Night constants are diagnostic residual parameters, not a known continuum correction.",
                  ], sources={str(path): sha256(path.read_bytes()).hexdigest() for path in
                              [*args.data, args.model, args.stellar_surface, args.observations, Path(__file__)]})
    (args.output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_arrays(args.output / "phase_arrays.npz", args.output / "phase_fit.png")


if __name__ == "__main__":
    main()
