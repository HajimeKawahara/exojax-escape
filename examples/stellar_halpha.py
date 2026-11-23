"""Compare PHOENIX and blackbody irradiation at fixed Parker atmospheres.

Supply the official vacuum wavelength FITS and 10000 K/logg=4/solar surface
flux FITS. Astropy and Matplotlib are example dependencies. No stellar data
are downloaded or bundled. This changes incident excitation only: internal
radiation feedback and the observer's stellar reference remain separate work.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from escape.hydrogen_forward import SYNCHRONOUS_ROTATION_KMS
from escape.hydrogen_rates import BOLTZMANN_CGS, LIGHT_SPEED_CGS, PLANCK_CGS, planck_occupation
from escape.nlte_forward import HydrogenNLTEForwardModel
from escape.stellar_radiation import surface_flux_occupation
from generate_spectra import load_observations


def blackbody_flux(frequency, temperature=10000.):
    """Surface pi B_nu, in cgs per Hz."""
    return (2 * np.pi * PLANCK_CGS * frequency**3 / LIGHT_SPEED_CGS**2
            / np.expm1(PLANCK_CGS * frequency / (BOLTZMANN_CGS * temperature)))


def load_surface(wavelength_path, flux_path, *, uv_tail="matched", ir_tail="matched"):
    """Complete missing PHOENIX tails explicitly; do not normalize luminosity."""
    from astropy.io import fits

    wavelength = np.asarray(fits.getdata(wavelength_path), dtype=float) * 1e-8
    flux_lambda, header = fits.getdata(flux_path, header=True)
    flux_lambda = np.asarray(flux_lambda, dtype=float)
    if header.get("BUNIT") != "erg/s/cm^2/cm" or header.get("PHXTEFF") != 10000:
        raise ValueError("This example requires a 10000 K PHOENIX surface flux per cm.")
    if (wavelength.shape != flux_lambda.shape or not np.all(np.diff(wavelength) > 0)
            or not np.isfinite(flux_lambda).all() or np.any(flux_lambda < 0)):
        raise ValueError("Invalid PHOENIX wavelength/flux arrays.")
    frequency = (LIGHT_SPEED_CGS / wavelength)[::-1]
    flux = (flux_lambda * wavelength**2 / LIGHT_SPEED_CGS)[::-1]
    infrared = np.geomspace(1e10, frequency[0], 512, endpoint=False)
    ultraviolet = np.geomspace(frequency[-1], 3e16, 512)[1:]
    factors = flux[[0, -1]] / blackbody_flux(frequency[[0, -1]])
    uv_factor = {"matched": factors[1], "zero": 0., "blackbody": 1.}[uv_tail]
    ir_factor = factors[0] if ir_tail == "matched" else 1.
    # The first outside sample is adjacent to the data boundary, so a zero/BB
    # tail does not create a long artificial interpolation ramp inside coverage.
    ultraviolet = np.r_[np.nextafter(frequency[-1], np.inf), ultraviolet]
    infrared = np.r_[infrared, np.nextafter(frequency[0], 0.)]
    info = dict(teff=header["PHXTEFF"], logg=header["PHXLOGG"],
                metallicity=header["PHXM_H"], native_range_angstrom=(wavelength[[0, -1]] * 1e8).tolist(),
                uv_tail=uv_tail, ir_tail=ir_tail, tail_factors=factors.tolist(),
                input_sha256={str(p): sha256(Path(p).read_bytes()).hexdigest()
                              for p in (wavelength_path, flux_path)})
    return (np.r_[infrared, frequency, ultraviolet],
            np.r_[ir_factor * blackbody_flux(infrared), flux,
                  uv_factor * blackbody_flux(ultraviolet)], info)


def occupation_fields(model, surface, *, rotation_kms=0., shift_kms=0., sigma_kms=0.):
    """Return shared occupations; velocity/smoothing options are sensitivities.

    Rotation uses a uniformly bright disk kernel (no limb darkening). Its
    projected speed as seen by the planet is not known from observer v sin i.
    Gaussian smoothing is a fixed-width sensitivity, not layer-dependent
    profile integration. Continuum occupations use the unbroadened spectrum.
    """
    frequency, flux = surface[:2]
    line = np.asarray(model.atom.line_frequency_hz)
    if rotation_kms:
        x, w = np.polynomial.legendre.leggauss(64)
        offsets, weights = rotation_kms * x, w * np.sqrt(1 - x*x)
        weights /= weights.sum()
    else:
        offsets, weights = np.array([0.]), np.array([1.])
    if sigma_kms:
        x, w = np.polynomial.hermite.hermgauss(16)
        offsets = (offsets[:, None] + np.sqrt(2) * sigma_kms * x).ravel()
        weights = (weights[:, None] * w / np.sqrt(np.pi)).ravel()
    query = line[:, None] * (1 + (offsets[None, :] + shift_kms) / (LIGHT_SPEED_CGS / 1e5))
    occupations = surface_flux_occupation(query, frequency, flux)
    line_field = jnp.sum(occupations * weights[None, :], axis=1)
    continuum_field = surface_flux_occupation(model.continuum_frequency_hz, frequency, flux)
    return dict(line_occupation=line_field, continuum_occupation=continuum_field)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength", required=True, type=Path)
    parser.add_argument("--flux", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/stellar_halpha"))
    parser.add_argument("--continuum-points", type=int, default=48)
    parser.add_argument("--nlayer", type=int, default=128)
    parser.add_argument("--nmax", type=int, default=15)
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    model = HydrogenNLTEForwardModel(nlayer=args.nlayer, nmax=args.nmax,
        continuum_points=args.continuum_points, nlon=12, nlat=6, nnu=601,
        tidal_coefficient_s2=3 * (2*np.pi/(1.48*86400))**2)
    surface = load_surface(args.wavelength, args.flux)
    phoenix = occupation_fields(model, surface)
    bb = dict(line_occupation=planck_occupation(model.atom.line_frequency_hz, 10000., .025),
              continuum_occupation=planck_occupation(model.continuum_frequency_hz, 10000., .025))
    fields = {"blackbody": bb, "PHOENIX": phoenix,
              "stellar_lines_only": phoenix | {"continuum_occupation": bb["continuum_occupation"]},
              "stellar_continuum_only": bb | {"continuum_occupation": phoenix["continuum_occupation"]},
              "rotation110": occupation_fields(model, surface, rotation_kms=110.),
              "smooth12": occupation_fields(model, surface, sigma_kms=12.),
              "shift_minus30": occupation_fields(model, surface, shift_kms=-30.),
              "shift_plus30": occupation_fields(model, surface, shift_kms=30.),
              "uv_zero": occupation_fields(model, load_surface(args.wavelength, args.flux, uv_tail="zero")),
              "uv_blackbody": occupation_fields(model, load_surface(args.wavelength, args.flux, uv_tail="blackbody")),
              "ir_blackbody": occupation_fields(model, load_surface(args.wavelength, args.flux, ir_tail="blackbody"))}
    cases = {"previous_fit": dict(mass_loss_g_s=209185986389574.34,
                                temperature_top_k=6105.231332735623,
                                rotation_kms=SYNCHRONOUS_ROTATION_KMS, zonal_kms=11.6),
             "hot_manual": dict(mass_loss_g_s=1e15, temperature_top_k=14000.,
                                rotation_kms=0., zonal_kms=0.)}
    velocity = np.asarray(model.velocity_kms)
    obs_v, observed, error = map(np.asarray, load_observations(model.line_center_cm))
    arrays = dict(velocity_kms=velocity, radius_rp=np.asarray(model.radius_rp),
                  observation_velocity=obs_v, observed=observed, error=error)
    report = dict(scope="Fixed atmospheres, prescribed incident SED; no internal radiation feedback or refit.",
                  surface=surface[2], nlayer=args.nlayer, nmax=args.nmax,
                  continuum_points=args.continuum_points, cases={}, fields={})
    for key, field in fields.items():
        ratio = np.asarray(field["line_occupation"] / bb["line_occupation"])
        report["fields"][key] = dict(Lyalpha_to_bb=float(ratio[0]), Lybeta_to_bb=float(ratio[1]),
                                      Halpha_to_bb=ratio[2:5].tolist())
    def save():
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(args.output / "arrays.npz", **arrays)
    transfer = jax.jit(model._transfer_line_coefficients)
    for case, parameters in cases.items():
        records = report["cases"][case] = {}
        pop_kwargs = {k: v for k, v in parameters.items() if k not in ("rotation_kms", "zonal_kms")}
        for name, field in fields.items():
            prof = model.population_profiles(**pop_kwargs, **field)
            good = bool(jnp.all(prof["converged"] & prof["valid"]))
            if not good:
                records[name] = dict(valid=False, reason="Local population root invalid or unconverged.")
                save()
                continue
            spectrum = np.asarray(transfer(prof["halpha_branch_extinction_integral"],
                prof["halpha_branch_emissivity_integral"], prof["radial_velocity_kms"],
                parameters["rotation_kms"], parameters["zonal_kms"], prof["temperature_k"]))
            if not np.isfinite(spectrum).all():
                records[name] = dict(valid=False, reason="Observer transfer is invalid.")
                save()
                continue
            pred = np.interp(obs_v, velocity[::-1], spectrum[::-1])
            baseline = float(np.average(observed-pred, weights=error**-2))
            phi = jax.vmap(model._line_profiles)(prof["temperature_k"], jnp.zeros(args.nlayer))[:, :, len(velocity)//2]
            source = np.asarray(jnp.sum(prof["halpha_branch_emissivity_integral"]*phi, axis=1)
                                / jnp.sum(prof["halpha_branch_extinction_integral"]*phi, axis=1))
            planck_scale = 2*PLANCK_CGS*LIGHT_SPEED_CGS**2*model.line_center_cm**3
            tex = PLANCK_CGS*LIGHT_SPEED_CGS*model.line_center_cm/BOLTZMANN_CGS / np.log1p(planck_scale/source)
            prefix = case+"_"+name
            arrays[prefix+"_spectrum"] = spectrum
            arrays[prefix+"_source_temperature"] = tex
            for key in ("n2_cm3", "n3_cm3", "n_h_cm3", "bound_cm3", "temperature_k", "radial_velocity_kms"):
                arrays[prefix+"_"+key] = np.asarray(prof[key])
            records[name] = dict(valid=True, parameters=parameters, baseline=baseline,
                rms=float(np.sqrt(np.mean(((pred+baseline-observed)/error)**2))),
                max_rate_residual=float(jnp.max(prof["rate_residual_norm"])),
                n2_column_cm2=float(jnp.sum(prof["n2_cm3"]*model.thickness_cm)),
                source_1p8_k=float(np.interp(1.8, arrays["radius_rp"][::-1], tex[::-1])))
            print(case, name, records[name], flush=True)
            save()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    # Keep the physically consistent input comparisons readable. The deliberately
    # mixed line/continuum interventions remain in metrics.json and arrays.npz.
    colors = {"blackbody": "k", "PHOENIX": "C0", "rotation110": "C1"}
    for row, case in enumerate(cases):
        axes[row, 0].errorbar(obs_v, observed*100, error*100, fmt=".", color=".65", label="Zenodo observation")
        for name, color in colors.items():
            rec = report["cases"][case][name]
            if not rec["valid"]:
                continue
            prefix = case+"_"+name
            axes[row, 0].plot(velocity, 100*(arrays[prefix+"_spectrum"]+rec["baseline"]), color=color,
                              label=f'{name.replace("_", " ")} ({rec["rms"]:.2f})')
            axes[row, 1].plot(arrays["radius_rp"], arrays[prefix+"_source_temperature"], color=color)
            axes[row, 2].semilogy(arrays["radius_rp"], arrays[prefix+"_n2_cm3"], color=color)
        axes[row, 0].set(xlim=(-100, 100), xlabel="Velocity [km/s]", ylabel="Contrast [%]", title=case.replace("_", " "))
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].set(xlim=(1, 2.5), xlabel="Radius / Rp", ylabel="Source brightness T [K]")
        axes[row, 1].axhline(4500, color=".6", ls=":")
        axes[row, 2].set(xlim=(1, 2.5), xlabel="Radius / Rp", ylabel="n2 [cm^-3]")
    fig.suptitle("Replace the incident spectrum; hold density, temperature and velocity fixed\nLabels show residual RMS after one constant offset; no physical refit or internal radiation feedback")
    fig.supxlabel("Observation: Zhang (2026), Zenodo 20707255, CC BY 4.0. Stellar model: Husser et al. (2013), PHOENIX.", fontsize=9)
    fig.savefig(args.output / "stellar_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, idx, label in zip(axes, [0, 1, 2], ["Ly-alpha", "Ly-beta", "H-alpha"]):
        center = float(model.atom.line_frequency_hz[idx])
        v = np.linspace(-1000, 1000, 2001)
        frequency = center*(1-v/(LIGHT_SPEED_CGS/1e5))
        ratio = np.asarray(surface_flux_occupation(frequency, *surface[:2]) / planck_occupation(frequency, 10000., .025))
        ax.semilogy(v, ratio)
        ax.axhline(1, ls=":", color="k", label="10000 K blackbody")
        ax.set(xlabel="Offset from atomic center [km/s]", ylabel="PHOENIX / blackbody", title=label)
        ax.legend(fontsize=8)
    fig.suptitle("Stellar photospheric absorption before any planetary absorption\n10000 K, log g=4, solar composition; vacuum surface flux, unbroadened")
    fig.savefig(args.output / "stellar_absorption.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
