"""Observe supplied Cloudy profiles with an explicit, approximate data operator.

This example does not run Cloudy, solve the atmosphere, or fit observations.
Supply profiles.npz and its run receipt from tools/cloudy_atmosphere.py, the
archived cadence_weights.npz, and optionally a local PHOENIX surface-flux NPZ.
Raw stellar data are neither copied nor included in the derived artifacts.
"""

from __future__ import annotations

import argparse
from functools import partial
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from escape.hydrogen_forward import PLANET_RADIUS_CM, SYNCHRONOUS_ROTATION_KMS
from escape.hydrogen_continuum import continuum_coefficients
from escape.hydrogen_bound_free import _orbital_cross_sections
from escape.nlte_forward import HydrogenNLTEForwardModel, halpha_coefficients
from escape.observation import process_phase_spectra
from generate_spectra import OBSERVATION_PATH, load_observations

C_KMS = 299792.458
ROOT = Path(__file__).resolve().parents[1]
# Same unresolved H I 2->3 row retained by tools/verify_nlte_spectrum.py;
# retrieved from NIST ASD on 2026-11-16. Construction makes no network request.
NIST_ROW = {
    "wav": [15233.146], "A": [44101000.0], "El": [82259.158],
    "Eu": [97492.304], "gl": [8.0], "gu": [18.0], "jl": [3.5],
    "ju": [8.5], "species": ["H_I"],
}


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def split_observer_populations(bound):
    """Keep the native n<=20 atom and identify additional supplied shells."""
    bound = np.asarray(bound)
    if bound.ndim != 2:
        raise ValueError("Use a two-dimensional layer/population array.")
    nmax = bound.shape[1]-3
    if not 3 <= nmax <= 25:
        raise ValueError("Use packed resolved n<=3 and shell populations through n<=25.")
    stop = min(bound.shape[1], 23)
    return bound[:, :stop], np.arange(21, nmax+1), bound[:, stop:]


def validate_high_n_fixture(fixture):
    """Check the independent C23 fixture against the native overlap shell n=20."""
    frequency, principal, sigma = (np.asarray(fixture[key]) for key in
                                   ("frequency_hz", "principal_n", "cross_section_cm2"))
    if (frequency.ndim != 1 or np.any(np.diff(frequency) <= 0)
            or not np.array_equal(principal, np.arange(20, 26))
            or sigma.shape != (6, len(frequency)) or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)):
        raise ValueError("Provide positive C23 cross sections for n=20..25 on an increasing frequency grid.")
    indices = np.array([0, len(frequency)//2, len(frequency)-1])
    relative = 6.62607015e-27*frequency[indices]/(13.5984345*1.602176634e-12/20**2)
    native = np.asarray(jax.jit(lambda energy: jnp.sum(
        ((2*jnp.arange(20)+1)/20**2)[:, None]*_orbital_cross_sections(20, energy), axis=0,
    ))(jnp.asarray(relative)))
    error = float(np.max(abs(sigma[0, indices]/native-1)))
    if error > 1e-5:
        raise ValueError(f"The n=20 cross-section overlap fails: relative error {error:g}.")
    return error


def high_n_continuum(profile, frequency, fixture):
    """Add supplied n=21..25 populations with the same Milne convention.

    This is an external reference observer, not a larger native SE atom.
    No populations, photoionization rates or thermal solution are recomputed.
    """
    _, principal, bound = split_observer_populations(profile["bound_cm3"])
    frequency = np.asarray(frequency)
    if not len(principal):
        zero = np.zeros((len(profile["temperature_k"]), len(frequency)))
        return zero, zero
    if fixture is None:
        raise ValueError("A profile above n=20 requires --high-n-cross-sections.")
    source_frequency = fixture["frequency_hz"]
    rounding = 8*np.finfo(float).eps*source_frequency.max()
    if (frequency.min() < source_frequency.min()-rounding
            or frequency.max() > source_frequency.max()+rounding):
        raise ValueError("The high-n cross-section fixture must cover the observer frequencies.")
    sigma = np.stack([np.interp(frequency, source_frequency,
                               fixture["cross_section_cm2"][int(n)-20]) for n in principal])
    temperature = profile["temperature_k"][:, None, None]
    h, k, c, me = 6.62607015e-27, 1.380649e-16, 2.99792458e10, 9.1093837015e-28
    thermal_volume = (h*h/(2*np.pi*me*k))**1.5
    threshold = 13.5984345*1.602176634e-12/principal**2
    inverse = (profile["electron_cm3"]*profile["n_h_cm3"]
               * profile["ionized_fraction"])[:, None, None]
    inverse = inverse * principal[None, :, None]**2 * thermal_volume/temperature**1.5 * np.exp(
        -(h*frequency[None, None, :]-threshold[None, :, None])/(k*temperature))
    extinction = np.sum(sigma[None, :, :]*(bound[:, :, None]-inverse), axis=1)
    emissivity = 2*h/c**2*frequency[None, :]**3*np.sum(sigma[None, :, :]*inverse, axis=1)
    return extinction, emissivity


def interpolate_profiles(data, radius):
    """Use log-linear interpolation for positive populations and linear T,v.

    Input and output radii must overlap fully; no atmosphere extrapolation.
    The supplied Cloudy level order is 1s,2s,2p,3s,3p,3d,n4,...,nmax.
    """
    source = data["radius_cm"]
    if (not np.all(np.diff(source) > 0) or radius.min() < source.min()
            or radius.max() > source.max()):
        raise ValueError("Reference profiles must cover the complete 1--3 Rp observer grid.")
    result = {}
    for key in ("temperature_k", "velocity_cm_s", "electron_cm3", "n_h_cm3",
                "ionized_fraction", "mean_molecular_weight", "bound_cm3"):
        values = data[key]
        logarithmic = key in ("bound_cm3", "electron_cm3", "n_h_cm3")
        if not np.isfinite(values).all() or (logarithmic and np.any(values <= 0)):
            raise ValueError(f"Nonfinite or nonpositive reference field: {key}.")
        def interpolate(column):
            if logarithmic:
                return np.exp(np.interp(radius, source, np.log(column)))
            return np.interp(radius, source, column)
        result[key] = (interpolate(values) if values.ndim == 1 else
                       np.column_stack([interpolate(column) for column in values.T]))
    return result


def stellar_reference(velocity, surface, rotation_kms):
    """PHOENIX / pi B(10000 K), broadened by a uniform stellar disk.

    The reference continuum is the same blackbody used by the planet observer.
    The subsequent continuum normalization rescales star and planet together.
    No limb darkening, spots, or time-variable stellar spectrum is assumed.
    """
    if surface is None:
        return np.ones_like(velocity)
    frequency, flux = surface["frequency_hz"], surface["surface_flux_nu"]
    center_hz = NIST_ROW["wav"][0] * 2.99792458e10
    x, weights = np.polynomial.legendre.leggauss(128)
    weights *= np.sqrt(1-x*x)
    weights /= weights.sum()
    query = center_hz / (1 + (velocity[:, None] - rotation_kms*x) / C_KMS)
    broadened = np.sum(np.interp(query, frequency, flux) * weights, axis=1)
    nu = center_hz / (1 + velocity / C_KMS)
    blackbody = (2*np.pi*6.62607015e-27*nu**3 / 2.99792458e10**2
                 / np.expm1(6.62607015e-27*nu/(1.380649e-16*10000.)))
    return broadened / blackbody


def apply_observation(phases, velocity, spectra, cadence, reference, stellar_velocity,
                      *, zero_after_phase=None):
    """Retain broad continuum using an explicitly constant endpoint extension."""
    if zero_after_phase is not None:
        samples = np.unique(cadence["phase"])
        spectra = np.asarray([np.interp(samples, phases, column) for column in spectra.T]).T
        spectra[samples > zero_after_phase] = 0.
        phases = samples
    padded_velocity = np.r_[-1000., velocity, 1000.]
    padded = np.column_stack((spectra[:, 0], spectra, spectra[:, -1]))
    shape = (len(cadence["phase"]), len(stellar_velocity))
    return process_phase_spectra(
        phases, padded_velocity, padded, cadence["phase"], cadence["planet_velocity"],
        cadence["night"], stellar_velocity, cadence["velocity"],
        stellar_reference=reference,
        continuum_weights=np.broadcast_to(abs(stellar_velocity) > 400, shape),
        master_weights=np.ones(shape),
        stellar_keep=np.broadcast_to(abs(stellar_velocity) >= 110, shape),
        coadd_weights=cadence["coadd_weight"],
    )


@partial(jax.jit, static_argnums=0, static_argnames=("include_hminus",))
def transfer_with_controls(model, profile, lower_mask, rotation, zonal, tail_chi, tail_eta,
                           *, include_hminus):
    """Remove lower H-alpha only; preserve the baseline continuum and flow.

    This diagnostic does not re-solve the atmosphere. Insensitivity cannot
    bound the lower thermal/ionization law's feedback on the entire wind.
    """
    bound, temperature = profile["bound_cm3"], profile["temperature_k"]
    electron = profile["electron_cm3"]
    proton = profile["n_h_cm3"]*profile["ionized_fraction"]
    alpha, emission = halpha_coefficients(bound, model.line_center_cm, sum_branches=False)
    alpha, emission = (jnp.where(lower_mask[:, None], 0., value) for value in (alpha, emission))
    chi, eta = jax.vmap(lambda t, n, ne, np_: continuum_coefficients(
        model.atom, t, n, ne, np_, jnp.asarray(model.wavenumber_cm)*2.99792458e10,
        include_hminus=include_hminus,
    ))(temperature, bound, electron, proton)
    return model._transfer_line_coefficients(
        alpha, emission, profile["velocity_cm_s"]/1e5, rotation, zonal, temperature,
        continuum_extinction=chi+tail_chi, continuum_emission=(eta+tail_eta)*2.99792458e10,
    )


def plot_results(arrays, records, destination):
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for index, record in enumerate(records):
        prefix, label = f"case{index}_", record["label"]
        color = colors[index % len(colors)]
        r = arrays[prefix+"radius_cm"] / PLANET_RADIUS_CM
        for axis, field, divisor in ((axes[0, 0], "temperature_k", 1),
                                    (axes[0, 1], "mean_molecular_weight", 1),
                                    (axes[0, 2], "velocity_cm_s", 1e5)):
            axis.plot(r, arrays[prefix+field]/divisor, color=color, label=label)
        axes[1, 0].semilogy(r, arrays[prefix+"n2_cm3"], color=color, label=label+", n=2")
        axes[1, 0].semilogy(r, arrays[prefix+"n3_cm3"], "--", color=color, label=label+", n=3")
        axes[1, 1].semilogy(r, arrays[prefix+"n3_cm3"] / arrays[prefix+"n2_cm3"],
                            color=color, label=label)
        mask = arrays[prefix+"cloudy_mask"]
        balance = arrays[prefix+"total_heating_cooling_ratio"].copy()
        if np.all(mask):
            balance[0] = np.nan  # The fixed-temperature base is not an energy equation.
        axes[1, 2].semilogy(r, balance, color=color, label=label)
        if np.any(~mask):
            for axis in axes.flat:
                axis.axvspan(1., r[np.flatnonzero(mask)[0]], color=color, alpha=.08)
    titles = ("Temperature", "Mean molecular weight", "Radial wind",
              "Hydrogen populations", "Excitation ratio", "Thermal balance incl. flow")
    labels = ("T [K]", r"$\mu$ [proton masses]", "v [km/s]",
              r"$n_i$ [cm$^{-3}$]", r"$n_3/n_2$", "Total heating / cooling")
    for axis, title, ylabel in zip(axes.flat, titles, labels):
        axis.set(title=title, ylabel=ylabel, xlabel=r"Radius / $R_p$", xlim=(1, 3))
        axis.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=7)
    axes[1, 2].axhline(1, color=".4", ls=":")
    fig.suptitle("Conditional Cloudy solar snapshots; not a validated native atmosphere", fontsize=13)
    fig.supxlabel("Shading: prescribed dense lower patch, where present. Fixed base excluded from thermal check.\n"
                  "Convergence and patch status are recorded separately.", fontsize=9)
    fig.savefig(destination / "atmosphere_profiles.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.3), constrained_layout=True)
    axis.errorbar(arrays["observed_velocity_kms"], 1e6*arrays["observed_contrast"],
                  yerr=1e6*arrays["observed_error"], fmt=".", color=".25", ms=4,
                  elinewidth=.7, alpha=.6, label="Keck/KPF, phases 0.30–0.45")
    for index, record in enumerate(records):
        prefix, color = f"case{index}_", colors[index % len(colors)]
        axis.plot(arrays["output_velocity_kms"], 1e6*arrays[prefix+"processed_contrast"],
                  color=color, label=record["label"]+": observation proxy")
        axis.plot(arrays["output_velocity_kms"], 1e6*arrays[prefix+"unprocessed_average"],
                  color=color, ls="--", alpha=.65, label=record["label"]+": before processing")
    axis.axhline(0, color=".5", ls=":")
    axis.set(xlim=(-105, 105), xlabel="Velocity from NIST H-alpha center [km/s]",
             ylabel="Residual contrast [ppm]",
             title="Conditional Cloudy snapshots → planet spectrum → observation proxy; no fit")
    axis.legend(fontsize=8, loc="best")
    axis.grid(alpha=.16)
    fig.supxlabel("Approximate cadence weights; equal nightly-master weights; stellar mask |v| < 110 km/s.\n"
                  "Data: Yapeng Zhang (2026), doi:10.5281/zenodo.20707255, CC BY 4.0. "
                  "Error bars are the supplied column.", fontsize=9)
    fig.savefig(destination / "halpha_observation.png", dpi=180)
    plt.close(fig)
    if any(record["lower_line_control_max_difference_over_supplied_error"] > .1 for record in records):
        fig, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
        for index, record in enumerate(records):
            prefix = f"case{index}_"
            difference = arrays[prefix+"lower_halpha_removed_contrast"]-arrays[prefix+"processed_contrast"]
            axis.plot(arrays["output_velocity_kms"], 1e6*difference, label=record["label"])
        axis.axhline(0, color=".5", ls=":")
        axis.set(xlim=(-105, 105), xlabel="Velocity from NIST H-alpha center [km/s]",
                 ylabel="Removal control minus baseline [ppm]",
                 title="Remove prescribed lower H-alpha opacity/emission; retain all continuum")
        axis.legend(fontsize=8)
        fig.supxlabel("Diagnostic only: populations, temperature and wind stay fixed. "
                      "This does not bound the lower thermal law's feedback.", fontsize=9)
        fig.savefig(destination / "lower_patch_control.png", dpi=180)
        plt.close(fig)
    else:
        (destination / "lower_patch_control.png").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, action="append", required=True,
                        metavar=("LABEL", "PROFILES_NPZ", "RECEIPT_JSON"))
    parser.add_argument("--cadence", type=Path, required=True)
    parser.add_argument("--stellar-surface", type=Path)
    parser.add_argument("--high-n-cross-sections", type=Path,
                        help="Independent C23 n20..25 cross-section fixture; required for default-atom profiles.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nlayer", type=int, default=128)
    parser.add_argument("--nlon", type=int, default=24)
    parser.add_argument("--nlat", type=int, default=12)
    parser.add_argument("--nnu", type=int, default=601)
    parser.add_argument("--rotation-kms", type=float, default=SYNCHRONOUS_ROTATION_KMS)
    parser.add_argument("--zonal-kms", type=float, default=11.6)
    parser.add_argument("--stellar-rotation-kms", type=float, default=110.)
    parser.add_argument("--without-hminus", action="store_true")
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    cadence = dict(np.load(args.cadence))
    if np.any(cadence["coadd_weight"][cadence["phase"] > .45] > 0):
        raise ValueError("Late-exposure control requires zero coadd weight after phase 0.45.")
    phases = np.linspace(.30, .50, 5)
    if cadence["phase"].min() < phases[0] or cadence["phase"].max() > phases[-1]:
        raise ValueError("Model phases do not cover the supplied nightly masters.")
    stellar_velocity = np.linspace(-600, 600, 2401)
    surface = None if args.stellar_surface is None else dict(np.load(args.stellar_surface))
    high_n = None if args.high_n_cross_sections is None else dict(np.load(args.high_n_cross_sections))
    overlap_error = None if high_n is None else validate_high_n_fixture(high_n)
    reference = stellar_reference(stellar_velocity, surface, args.stellar_rotation_kms)
    exojax_path = Path(next(iter(importlib.util.find_spec("exojax").submodule_search_locations)))
    arrays = dict(output_velocity_kms=cadence["velocity"], model_phase=phases,
                  exposure_phase=cadence["phase"], exposure_night=cadence["night"],
                  exposure_orbital_velocity_kms=cadence["planet_velocity"],
                  supplied_coadd_weight=cadence["coadd_weight"],
                  stellar_velocity_kms=stellar_velocity, stellar_reference=reference)
    report = {
        "scope": "External conditional reference snapshots; neither a fit, native agreement, nor paper reproduction.",
        "command": [sys.executable, *sys.argv],
        "command_configuration": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "observation_operator": {
            "cadence": f"{len(cadence['phase'])} archived exposures; approximate inverse-variance coadd weights and validity",
            "master": "Equal pixel weights within each night; all supplied phases included",
            "unprocessed_curve": "Planet contrasts averaged with the same supplied coadd weights before stellar/master division",
            "continuum": "Mean stellar-frame flux over |v|>400 km/s; constant planet endpoints outside +/-120 km/s",
            "mask": "Discard stellar-frame |v|<110 km/s before planet-frame coaddition",
            "stellar_reference": "PHOENIX / pi B(10000 K), uniform-disk vsini broadening" if surface is not None else "Unit continuum",
            "not_recovered": "Original master weights, exposure/bin integration, occultation, and exact publication reduction",
            "late_signal_control": "Zero planet signal at master exposures phase>0.45; an extreme sensitivity, not eclipse geometry",
        },
        "observer": "1--3 Rp plane-parallel dayside columns, fixed projected area, 4500 K lower boundary; R=98000",
        "continuum": "Hydrogen bound/free continuum plus LTE trace H-minus" if not args.without_hminus else "Hydrogen bound/free continuum; no H-minus",
        "hminus_limit": "Observer H-minus is not the Cloudy chemical abundance or full easychem chemistry",
        "lower_line_control": "Remove only H-alpha opacity/emission below Cloudy join; preserve continuum, T, populations and wind. This does not bound the lower thermal law's global feedback.",
        "high_n_control": "Native observer through n20 plus external C23 n21..25 continuum/Milne contribution; no native SE extension. Also save the n20-only spectrum.",
        "high_n_overlap_relative_error": overlap_error,
        "profile_interpolation": "Linear T,mu,v,ionization; log-linear positive populations; no radial extrapolation",
        "nist": {"row": NIST_ROW, "source": "https://physics.nist.gov/PhysRefData/ASD/lines_form.html"},
        "data": {"source": "https://doi.org/10.5281/zenodo.20707255", "license": "CC BY 4.0"},
        "input_sha256": {str(path): file_hash(path) for path in
                          [Path(__file__), args.cadence, OBSERVATION_PATH] +
                          ([] if args.stellar_surface is None else [args.stellar_surface]) +
                          ([] if args.high_n_cross_sections is None else [args.high_n_cross_sections])},
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path)
                          for path in sorted((ROOT/"src/escape").glob("*.py"))},
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "numpy", "exojax")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "exojax_source_sha256": {str(path.relative_to(exojax_path)): file_hash(path)
                                 for path in sorted(exojax_path.rglob("*.py"))},
        "cases": [],
    }
    for name in ("cadence_summary.json", "data_reconstruction.json"):
        source = args.cadence.with_name(name)
        if source.exists():
            report[name.removesuffix(".json")] = json.loads(source.read_text())
            report["input_sha256"][str(source)] = file_hash(source)
    for index, (label, profile_path, receipt_path) in enumerate(args.case):
        print(f"Observing {label}: {profile_path}", flush=True)
        receipt_bytes = Path(receipt_path).read_bytes()
        data, receipt = dict(np.load(profile_path)), json.loads(receipt_bytes)
        original_nmax = data["bound_cm3"].shape[1]-3
        nmax = min(original_nmax, 20)
        if original_nmax > 20 and high_n is None:
            raise ValueError("A profile above n=20 requires --high-n-cross-sections.")
        iteration = int(Path(profile_path).parent.name.removeprefix("iteration_"))
        step = next(item for item in receipt["iterations"] if item["iteration"] == iteration)
        record = {"label": label, "profile_path": profile_path, "profile_sha256": file_hash(profile_path),
                  "receipt_path": receipt_path, "receipt_sha256": sha256(receipt_bytes).hexdigest(),
                  "selected_iteration": step,
                  "selected_iteration_converged": bool(receipt["converged"] and
                      iteration == receipt["iterations"][-1]["iteration"]),
                  "reference_receipt": receipt}
        report["cases"].append(record)
        prefix = f"case{index}_"
        for key in ("radius_cm", "temperature_k", "mean_molecular_weight", "velocity_cm_s",
                    "n_h_cm3", "electron_cm3", "ionized_fraction", "bound_cm3", "n2_cm3",
                    "n3_cm3", "total_heating_cooling_ratio"):
            arrays[prefix+key] = data[key]
        arrays[prefix+"cloudy_mask"] = data.get("cloudy_mask", np.ones_like(data["radius_cm"], dtype=bool))
        for key in ("principal_n", "angular_l", "statistical_weights"):
            if key in data:
                arrays[prefix+key] = data[key]
        spectra, removed_spectra, truncated_spectra = [], [], []
        cloudy_mask = arrays[prefix+"cloudy_mask"]
        join_radius = data["radius_cm"][np.flatnonzero(cloudy_mask)[0]]
        for phase in phases:
            with patch("exojax.database.nist.api.fetch_nist_lines", lambda *a, **kw: (pd.DataFrame(NIST_ROW), [])):
                model = HydrogenNLTEForwardModel(nlayer=args.nlayer, nmax=nmax, nlon=args.nlon,
                                                nlat=args.nlat, nnu=args.nnu, phase=float(phase))
            profile = interpolate_profiles(data, np.asarray(model.radius_rp)*PLANET_RADIUS_CM)
            tail_chi, tail_eta = high_n_continuum(
                profile, np.asarray(model.wavenumber_cm)*2.99792458e10, high_n)
            observer_profile = dict(profile, bound_cm3=split_observer_populations(profile["bound_cm3"])[0])
            truncated = model.spectrum_from_profiles(
                observer_profile["bound_cm3"], profile["temperature_k"], profile["velocity_cm_s"]/1e5,
                electron_cm3=profile["electron_cm3"],
                proton_cm3=profile["n_h_cm3"]*profile["ionized_fraction"],
                rotation_kms=args.rotation_kms, zonal_kms=args.zonal_kms,
                include_continuum=True, include_hminus=not args.without_hminus,
            )
            truncated_spectra.append(np.asarray(truncated)[::-1])
            spectrum = (transfer_with_controls(
                model, observer_profile, np.zeros(args.nlayer, dtype=bool),
                args.rotation_kms, args.zonal_kms, tail_chi, tail_eta,
                include_hminus=not args.without_hminus) if original_nmax > 20 else truncated)
            spectra.append(np.asarray(spectrum)[::-1])
            lower_mask = np.asarray(model.radius_rp)*PLANET_RADIUS_CM < join_radius
            removed = (transfer_with_controls(model, observer_profile, lower_mask, args.rotation_kms, args.zonal_kms,
                                          tail_chi, tail_eta,
                                          include_hminus=not args.without_hminus)
                       if np.any(lower_mask) else spectrum)
            removed_spectra.append(np.asarray(removed)[::-1])
        spectra = np.asarray(spectra)
        if not np.isfinite(spectra).all():
            raise ValueError("Nonfinite spectrum; inspect observer temperature/opacity domain, do not plot as agreement.")
        velocity = np.asarray(model.velocity_kms)[::-1]
        result = apply_observation(phases, velocity, spectra, cadence, reference, stellar_velocity)
        unit = apply_observation(phases, velocity, spectra, cadence, np.ones_like(reference), stellar_velocity)
        late_zero = apply_observation(phases, velocity, spectra, cadence, reference, stellar_velocity,
                                      zero_after_phase=.45)
        lower_removed = apply_observation(phases, velocity, np.asarray(removed_spectra), cadence,
                                          reference, stellar_velocity)
        tail_removed = apply_observation(phases, velocity, np.asarray(truncated_spectra), cadence,
                                         reference, stellar_velocity)
        raw = np.asarray([np.interp(cadence["phase"], phases, column) for column in spectra.T]).T
        unprocessed = np.asarray([np.interp(cadence["velocity"], velocity, row) for row in raw])
        weights = cadence["coadd_weight"]
        arrays.update(model_velocity_kms=velocity)
        arrays[prefix+"phase_spectra"] = spectra
        arrays[prefix+"unprocessed_average"] = (unprocessed*weights).sum(axis=0)/weights.sum(axis=0)
        arrays[prefix+"processed_contrast"] = np.asarray(result.contrast)
        arrays[prefix+"unit_reference_contrast"] = np.asarray(unit.contrast)
        arrays[prefix+"late_master_signal_zero_contrast"] = np.asarray(late_zero.contrast)
        arrays[prefix+"lower_halpha_removed_contrast"] = np.asarray(lower_removed.contrast)
        arrays[prefix+"high_n_continuum_removed_contrast"] = np.asarray(tail_removed.contrast)
        arrays[prefix+"effective_weight"] = np.asarray(result.weight)
        observed_v, observed, error = load_observations(model.line_center_cm)
        if (observed_v.min() < cadence["velocity"].min()
                or observed_v.max() > cadence["velocity"].max()
                or not np.all(np.asarray(result.valid))):
            raise ValueError("The processed output must cover every observation without missing weights.")
        arrays.update(observed_velocity_kms=observed_v, observed_contrast=observed, observed_error=error)
        prediction = np.interp(observed_v, cadence["velocity"], np.asarray(result.contrast))
        late_difference = np.asarray(late_zero.contrast-result.contrast)
        late_scaled = np.interp(observed_v, cadence["velocity"], late_difference)/error
        lower_difference = np.asarray(lower_removed.contrast-result.contrast)
        lower_scaled = np.interp(observed_v, cadence["velocity"], lower_difference)/error
        tail_difference = np.asarray(tail_removed.contrast-result.contrast)
        tail_scaled = np.interp(observed_v, cadence["velocity"], tail_difference)/error
        record.update(
            maximum_phx_vs_unit_reference_difference_ppm=float(np.nanmax(abs(result.contrast-unit.contrast))*1e6),
            late_signal_control_max_difference_ppm=float(np.nanmax(abs(late_difference))*1e6),
            late_signal_control_max_difference_over_supplied_error=float(np.max(abs(late_scaled))),
            late_signal_control_rms_difference_over_supplied_error=float(np.sqrt(np.mean(late_scaled**2))),
            lower_line_control_max_difference_ppm=float(np.nanmax(abs(lower_difference))*1e6),
            lower_line_control_max_difference_over_supplied_error=float(np.max(abs(lower_scaled))),
            lower_line_control_rms_difference_over_supplied_error=float(np.sqrt(np.mean(lower_scaled**2))),
            high_n_continuum_control_max_difference_ppm=float(np.max(abs(tail_difference))*1e6),
            high_n_continuum_control_max_difference_over_supplied_error=float(np.max(abs(tail_scaled))),
            observation_residual_rms_over_supplied_error=float(np.sqrt(np.mean(((prediction-observed)/error)**2))),
            spectrum_endpoint_range_ppm=(1e6*spectra[:, [0,-1]]).tolist(),
            observer_population_grid_radius_cm=(np.asarray(model.radius_rp)*PLANET_RADIUS_CM).tolist(),
        )
        print(f"  RMS residual / supplied error = {record['observation_residual_rms_over_supplied_error']:.3f}", flush=True)
    np.savez_compressed(args.output / "derived_profiles_spectra.npz", **arrays)
    plot_results(arrays, report["cases"], args.output)
    report["artifact_sha256"] = {name: file_hash(args.output/name) for name in
                                ("derived_profiles_spectra.npz", "atmosphere_profiles.png", "halpha_observation.png",
                                 "lower_patch_control.png") if (args.output/name).exists()}
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
