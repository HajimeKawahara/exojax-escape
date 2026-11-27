"""Audit the current native H-alpha observation operator with known signals.

Archived pixels supply support and fixed weights, not synthetic signal values.
This does not reconstruct the publication's unknown upstream reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from gp_halpha import make_native_flux_fast, PERIOD_MINUTES
from time_halpha import make_native_flux
from cloudy_halpha import stellar_reference
from phase_halpha import REST_ANGSTROM
from halpha_geometry_fit import setup, projections, fixed_fit, PHASE, WINDOWS

jax.config.update("jax_enable_x64", True)
C = 299792.458


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_phases(spectra, phases, science):
    sampled = np.array([np.interp(phases, PHASE, column) for column in spectra.T]).T
    return np.where(science[:, None], sampled, 0.)


def project_native(problem, models, edges, offsets=None):
    """Use the unchanged production display projector and GP posterior route."""
    groups = []
    for i, (group, model) in enumerate(zip(problem["groups"], models)):
        groups.append(dict(group, log_data=np.asarray(model) + (0. if offsets is None else offsets[i])))
    synthetic = dict(problem, groups=groups, models=lambda spectra, rv: tuple(models))
    return projections(synthetic, np.zeros((len(PHASE), 2)), [0., 0.], edges)


def morphology(row):
    means = {}
    for name, (left, right) in WINDOWS.items():
        take = (row["velocity"] >= left) & (row["velocity"] < right)
        means[name] = float(np.average(row["planet"][take], weights=1 / row["error"][take]**2)) * 1e6
    means["blue_minus_core_ppm"] = means["blue"] - means["core"]
    means["shoulders_minus_core_ppm"] = (means["blue"] + means["red"]) / 2 - means["core"]
    return means


def integrated_models(problem, spectra, velocity, surface, pixel_widths, *, exposure_seconds=0., pixels=False):
    """Integrate flux and its continuum estimator before division/logarithms."""
    nodes, weights = np.polynomial.legendre.leggauss(5 if exposure_seconds else 3)
    continuum_grid = np.linspace(-600., 600., 2401)
    continuum_grid = continuum_grid[abs(continuum_grid) > 400]
    continuum_star = np.mean(stellar_reference(continuum_grid, surface, 110.))
    results = []
    for group, width in zip(problem["groups"], pixel_widths):
        total = np.zeros_like(group["log_data"])
        total_normalization = np.zeros(len(group["phase"]))
        for node, weight in zip(nodes, weights / 2):
            phase = group["phase"] + node * exposure_seconds / (2 * PERIOD_MINUTES * 60)
            coordinate = group["stellar_velocity"] + (node * width / 2 if pixels else 0.)
            altered = dict(group, phase=phase, stellar_velocity=coordinate)
            sampled = sample_phases(spectra, phase, group["science"])
            flux = make_native_flux_fast(velocity, surface, altered)
            orbital = 244.8 * np.sin(2 * np.pi * phase)
            query = C * ((1 + continuum_grid[None] / C) / (1 + orbital[:, None] / C) - 1)
            normalization = continuum_star + np.mean(np.array([
                np.interp(q, velocity, row) for q, row in zip(query, sampled)]), axis=1)
            total += weight * np.asarray(flux(sampled, 0.)) * normalization[:, None]
            total_normalization += weight * normalization
        results.append(np.log(total / total_normalization[:, None]))
    return tuple(results)


def run(args):
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *[root / "examples" / name for name in
        ("gp_halpha.py", "time_halpha.py", "phase_halpha.py", "cloudy_halpha.py", "halpha_geometry_fit.py")],
        *[root / "src/escape" / name for name in
          ("observation.py", "phase_observation.py", "time_contrast.py", "time_gp.py")]]
    paths = [*args.observations, args.stellar, args.noise, args.bins, args.baseline, *sources]
    pins = {str(path.resolve()): digest(path) for path in paths}
    surface = dict(np.load(args.stellar))
    noise = {key: np.median(value, axis=(0, 1)) for key, value in dict(np.load(args.noise)).items()}
    saved = dict(np.load(args.baseline))
    velocity, edges = saved["model_velocity_kms"], np.load(args.bins)["edges"]
    problem = setup(PHASE, velocity, args.observations, surface, noise)
    shape = (len(PHASE), len(velocity))
    gaussian = 500e-6 * np.exp(-.5 * (velocity / 20)**2)
    cases = dict(null=np.zeros(shape), flat=np.full(shape, 800e-6),
        gaussian20=np.broadcast_to(gaussian, shape),
        gaussian50=np.broadcast_to(500e-6 * np.exp(-.5 * (velocity / 50)**2), shape),
        changing_gaussian=gaussian[None] * (1 + 2 * (PHASE[:, None] - .375)))
    offsets = [(.03 * np.sin(group["stellar_velocity"] / 17)
                + .02 * np.cos(group["stellar_velocity"] / 51))[None]
               for group in problem["groups"]]
    arrays, review_cases = {}, {}
    null_models = problem["models"](cases["null"], 0.)
    interpolation_error, offset_error = 0., 0.
    for name, spectra in cases.items():
        models = problem["models"](spectra, 0.)
        rows = project_native(problem, models, edges)
        changed = project_native(problem, models, edges, offsets)
        offset_error = max(offset_error, max(float(np.max(np.abs(changed[k]["observed"] - rows[k]["observed"]))) for k in rows))
        for group, model in zip(problem["groups"], models):
            slow = make_native_flux(velocity, surface, group)
            comparison = np.log(np.asarray(slow(sample_phases(spectra, group["phase"], group["science"]), 0.)))
            interpolation_error = max(interpolation_error, float(np.max(np.abs(comparison - model))))
        review_cases[name] = dict(bands_ppm={key: morphology(row) for key, row in rows.items()},
            maximum_matched_display_residual=float(max(np.max(np.abs(row["residual"])) for row in rows.values())))
        arrays[name + "_intrinsic"] = np.asarray(spectra)
        for label, row in rows.items():
            arrays.update({name + "__" + label + "__" + key: value for key, value in row.items()})
        print(name, review_cases[name]["bands_ppm"]["all"], flush=True)

    # A common arbitrary log master is removed without estimating its weights.
    # The fit uses the current two-parameter optimizer and fixed GP covariance.
    injected_parameters = np.array([3.25, np.log(1.25)])
    recovery = setup(PHASE, velocity, args.observations, surface, noise)
    injected = recovery["models"](cases["gaussian20"] * np.exp(injected_parameters[1]), injected_parameters[0])
    for group, model, offset in zip(recovery["groups"], injected, offsets):
        group["log_data"] = np.asarray(model) + offset
    fit = fixed_fit(recovery, cases["gaussian20"])
    fit["injected_parameters"] = injected_parameters.tolist()
    fit["parameter_error"] = (np.asarray(fit["parameters"]) - injected_parameters).tolist()

    # Pixel widths refer to the full archive grid before stride-two selection.
    widths, support, orbital_span = [], [], []
    for group in problem["groups"]:
        with h5py.File(args.observations[group["night"]]) as source:
            native = C * (source["wave"][group["order"]] / REST_ANGSTROM - 1)
        width = np.gradient(native)[group["pixel_index"]]
        widths.append(width)
        phase = group["phase"][group["science"]]
        half_phase = 150 / (2 * PERIOD_MINUTES * 60)
        span = 244.8 * (np.sin(2 * np.pi * (phase + half_phase)) - np.sin(2 * np.pi * (phase - half_phase)))
        orbital_span.extend(abs(span))
        orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
        query = C * ((1 + group["stellar_velocity"][None] / C) / (1 + orbital[:, None] / C) - 1)
        roundtrip = C * ((1 + query / C) * (1 + orbital[:, None] / C) - 1)
        support.append(dict(label=group["label"], science_exposures=int(group["science"].sum()),
            eclipse_exposures=int(group["eclipse"].sum()), native_pixels=len(width),
            native_pixel_width_kms=[float(width.min()), float(width.max())],
            science_orbital_velocity_kms=[float(orbital[group["science"]].min()), float(orbital[group["science"]].max())],
            model_endpoint_extension_fraction=float(np.mean((query < velocity[0]) | (query > velocity[-1]))),
            science_endpoint_extension_fraction=float(np.mean(
                ((query < velocity[0]) | (query > velocity[-1]))[group["science"]])),
            doppler_roundtrip_max_error_kms=float(np.max(abs(roundtrip - group["stellar_velocity"][None])))))
    integrations = {}
    for name, spectra in (("gaussian20", cases["gaussian20"]), ("baseline", saved["fine_intrinsic_phases"][:len(PHASE)])):
        midpoint = problem["models"](spectra, 0.)
        original = project_native(problem, midpoint, edges)["all"]
        integrations[name] = {}
        for label, options in (("exposure_150s", dict(exposure_seconds=150.)), ("native_pixel", dict(pixels=True))):
            integrated = integrated_models(problem, spectra, velocity, surface, widths, **options)
            row = project_native(problem, integrated, edges)["all"]
            delta = row["planet"] - original["planet"]
            center = abs(row["velocity"]) < 80
            integrations[name][label] = dict(maximum_native_log_flux_change_ppm=1e6 * max(float(np.max(abs(a - b))) for a, b in zip(integrated, midpoint)),
                maximum_display_change_within_80kms_ppm=float(np.max(abs(delta[center]))) * 1e6,
                band_change_ppm={key: morphology(row)[key] - morphology(original)[key] for key in WINDOWS})
            arrays[name + "__" + label + "__delta_ppm"] = delta * 1e6
            print(name, label, integrations[name][label], flush=True)

    # Positive RV means a redward displacement, independently of display bins.
    group = problem["groups"][0]
    row_index = int(np.argmin(abs(group["phase"] - .35)))
    flux = make_native_flux_fast(velocity, surface, group)
    sampled = sample_phases(cases["gaussian20"], group["phase"], group["science"])
    star = np.asarray(flux(np.zeros_like(sampled), 0.))[row_index]
    centroids = []
    for rv in (-3.25, 3.25):
        line = np.asarray(flux(sampled, rv))[row_index] - star
        centroids.append(float(np.sum(line * group["stellar_velocity"]) / line.sum()))
    sign_shift = centroids[1] - centroids[0]
    expected_shift = 6.5 * (1 + 244.8 * np.sin(2 * np.pi * group["phase"][row_index]) / C)
    null_max = float(np.max(abs(arrays["null__all__planet"])))
    unchanged = all(digest(path) == expected for path, expected in pins.items())
    checks = dict(input_unchanged=unchanged, null_below_1e_12=null_max < 1e-12,
        common_master_invariance_below_1e_12=offset_error < 1e-12,
        native_sampler_parity_below_1e_12=interpolation_error < 1e-12,
        positive_rv_is_red=sign_shift > 0,
        rv_centroid_error_below_0p01kms=abs(sign_shift - expected_shift) < .01,
        injection_parameter_error_below_1e_3=bool(np.max(np.abs(fit["parameter_error"])) < 1e-3),
        injection_Q_below_1e_6=fit["Q"] < 1e-6)
    checks = {key: bool(value) for key, value in checks.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    arrays.update(model_velocity_kms=velocity, phase=PHASE, display_edges_kms=edges)
    raw = args.output / "arrays.npz"
    np.savez_compressed(raw, **arrays)
    review = dict(scope=__doc__.strip(), source_and_input_sha256=pins, checks=checks, passed=all(checks.values()),
        conventions=dict(rest_angstrom=float(REST_ANGSTROM), wavelength="Vacuum model convention; archived wave coordinates are used as supplied without an additional barycentric or systemic correction.",
            orbital="Circular 244.8*sin(2*pi*phase) km/s; positive is red. P=1.48 days; RV is an additional planet-frame redshift.",
            stellar="PHOENIX/pi B(10000 K), uniform-disk rotation 110 km/s; planet contrast uses the same blackbody continuum. Star and planet are continuum-normalized together before logarithms.",
            master="The current native GP path profiles one common log spectrum per night/order/wavelength. It does not apply the older example's invented nightly-master weights to archived data.",
            endpoints="The model is interpolated on its saved grid with constant endpoint extension, including in the continuum normalization windows |v_star|>400 km/s.",
            display="Same native-bin coefficients and common-reference projection for synthetic data and model; fixed GP covariance. Display white errors are weighting devices, not a calibrated covariance."),
        support=support, cases=review_cases, injection_recovery=fit,
        numerical_controls=dict(null_maximum_abs=null_max, arbitrary_common_master_maximum_change=offset_error,
            fast_vs_reference_native_sampler_maximum_difference=interpolation_error,
            rv_centroid_shift_kms=sign_shift, expected_rv_centroid_shift_kms=expected_shift),
        finite_integration=integrations, orbital_span_during_150s_kms=[float(min(orbital_span)), float(max(orbital_span))],
        raw_arrays=dict(path=str(raw), sha256=digest(raw)),
        limitations=["No atmospheric solve, inferred covariance calibration, or claim to reconstruct the authors' master/coadd is made.",
            "Time-constant multiplicative references cancel in log projection; time-variable extraction, telluric, wavelength or stellar errors need separate controls.",
            "Synthetic values use actual archive support and weights frozen from the real data. This noiseless test checks implementation and shape transfer, not statistical coverage.",
            "The 150 s duration follows the paper as supplied in the audit; no timestamps/durations were present in these HDF5 products. Smearing uses the adopted circular orbit and interpolated saved phase spectra.",
            "Native pixel integration assumes top-hat widths inferred from adjacent archive wavelengths; the original pixel response and upstream interpolation are unknown.",
            "Projection can distort a positive line and create negative side structures. Comparing an intrinsic model directly with projected data is inconsistent even when the matched projected residual is zero."])
    args.review.mkdir(parents=True, exist_ok=True)
    (args.review / "review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    plot(arrays, args.review / "injection_shapes.png")
    print(json.dumps(dict(passed=review["passed"], checks=checks, recovery=fit)), flush=True)
    if not review["passed"]:
        raise RuntimeError("An operator audit control failed; inspect the receipt.")


def plot(arrays, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, name in zip(axes.flat, ("gaussian20", "gaussian50", "flat", "changing_gaussian")):
        ax.plot(arrays["model_velocity_kms"], 1e6 * arrays[name + "_intrinsic"].mean(axis=0), "--", label="Intrinsic phase mean")
        ax.plot(arrays[name + "__all__velocity"], 1e6 * arrays[name + "__all__planet"], label="Matched common-reference display")
        ax.axhline(0, color="0.5", lw=.5)
        ax.set(title=name, xlim=(-80, 80), ylabel="Signal (ppm)", xlabel="Planet-rest velocity (km/s)")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Noiseless known signals on the actual native support; fixed weights")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", nargs=2, type=Path, default=[Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")])
    parser.add_argument("--stellar", type=Path, default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--noise", type=Path, default=Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz"))
    parser.add_argument("--bins", type=Path, default=Path("/tmp/escape-gp-combined-20261125/combined_arrays.npz"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/escape-halpha-operator-audit"))
    parser.add_argument("--review", type=Path, default=Path("mceg/evidence/halpha_data_audit_20261127/operator"))
    run(parser.parse_args())
