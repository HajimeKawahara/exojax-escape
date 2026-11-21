"""Generate a prescribed-field NLTE spectrum and population diagnostics.

This is a forward example, not a fit or a Cloudy reproduction. Optional NumPyro
sampling uses synthetic data to smoke-test the differentiable spectrum path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from escape.hydrogen_atom import DEFAULT_PRINCIPAL_N
from escape.hydrogen_rates import LEVEL_NAMES
from escape.nlte_forward import HydrogenNLTEForwardModel
from generate_spectra import load_observations


def nuts_smoke(model):
    """Exercise NUTS on synthetic spectral pixels; not a posterior validation."""
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    signal = model.spectrum()
    sigma = float(jnp.max(jnp.abs(signal))) / 10
    observations = signal + sigma * jax.random.normal(jax.random.key(19), signal.shape)

    def probability_model():
        log_mass = numpyro.sample("log_mass_scale", dist.Normal(0., 0.2))
        mean = model.spectrum(mass_loss_g_s=1e13 * jnp.exp(log_mass))
        numpyro.sample("synthetic_spectrum", dist.Normal(mean, sigma), obs=observations)

    sampler = MCMC(
        NUTS(probability_model, target_accept_prob=0.85, max_tree_depth=5),
        num_warmup=32, num_samples=32, num_chains=1, progress_bar=False,
    )
    sampler.run(jax.random.key(20))
    samples = np.asarray(sampler.get_samples()["log_mass_scale"])
    return {
        "purpose": "AD and sampling smoke test only; no convergence claim",
        "synthetic_truth_log_mass_scale": 0.0,
        "num_warmup": 32, "num_samples": 32, "num_chains": 1,
        "divergences": int(np.sum(sampler.get_extra_fields()["diverging"])),
        "all_samples_finite": bool(np.isfinite(samples).all()),
        "mean_log_mass_scale": float(samples.mean()),
        "std_log_mass_scale": float(samples.std()),
        "numpyro_version": numpyro.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/nlte"))
    parser.add_argument("--nlayer", type=int, default=128)
    parser.add_argument("--nmax", type=int, default=DEFAULT_PRINCIPAL_N)
    parser.add_argument("--nuts-smoke", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    model = HydrogenNLTEForwardModel(
        nlayer=args.nlayer, nmax=args.nmax, nlon=12, nlat=6, nnu=601,
    )
    level_names = list(LEVEL_NAMES) + [f"n{n}" for n in range(4, args.nmax + 1)]
    profiles = model.population_profiles()
    spectrum = np.asarray(model.spectrum())
    if not np.all(profiles["converged"]) or not np.isfinite(spectrum).all():
        raise RuntimeError("NLTE or wind solution failed; inspect the model diagnostics.")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "spectrum.csv", np.column_stack((model.velocity_kms, spectrum)),
               delimiter=",", header="velocity_kms,contrast", comments="")
    np.savetxt(args.output / "profiles.csv", np.column_stack((
        model.radius_rp, profiles["temperature_k"], profiles["radial_velocity_kms"],
        profiles["n_h_cm3"], profiles["bound_cm3"], profiles["electron_density_cm3"],
        profiles["ionized_fraction"], profiles["rate_residual_norm"],
    )), delimiter=",", comments="", header=(
        "radius_rp,temperature_k,velocity_kms,n_h_cm3," + ",".join(level_names)
        + ",electron_cm3,ionized_fraction,scaled_rate_residual"
    ))
    report = {
        "scope": "extended local SE with prescribed T,nH,external radiation; not a fit",
        "nmax": args.nmax, "bound_states": len(level_names),
        "mass_loss_g_s": 1e13, "radiation_temperature_k": 10000.,
        "radiation_dilution": 0.025, "wind_mean_molecular_weight": 0.5,
        "temperature_base_k": 6000., "temperature_top_k": 10000.,
        "nlayer": args.nlayer, "nlon": 12, "nlat": 6, "nnu": 601,
        "max_conservation_error": float(jnp.max(profiles["conservation_error"])),
        "max_scaled_rate_residual": float(jnp.max(profiles["rate_residual_norm"])),
        "equilibrium_scope": "local statistical equilibrium is assumed; no advective"
                             " population or relaxation-timescale validation is implied",
        "positive_integrated_extinction": bool(jnp.all(profiles["positive_extinction"])),
        "ionized_fraction_range": [float(jnp.min(profiles["ionized_fraction"])),
                                   float(jnp.max(profiles["ionized_fraction"]))],
        "contrast_range": [float(spectrum.min()), float(spectrum.max())],
        "jax_version": jax.__version__, "jax_enable_x64": jax.config.x64_enabled,
    }
    if args.nuts_smoke:
        smoke_model = HydrogenNLTEForwardModel(
            nlayer=8, nmax=args.nmax, nlon=4, nlat=2, nnu=101, wind_ngrid=65,
        )
        report["nuts_smoke"] = nuts_smoke(smoke_model)
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    radius = np.asarray(model.radius_rp)
    axes[0, 0].plot(radius, profiles["temperature_k"])
    axes[0, 0].set(ylabel="Gas temperature [K]", title="Prescribed atmosphere")
    axes[1, 0].plot(radius, profiles["radial_velocity_kms"])
    axes[1, 0].set(ylabel="Radial velocity [km/s]", title="Nonisothermal Parker wind")
    for index, label in enumerate(level_names):
        axes[0, 1].semilogy(radius, np.asarray(profiles["bound_cm3"])[:, index], label=label)
    axes[0, 1].semilogy(radius, profiles["electron_density_cm3"], "k--", label="e = p")
    axes[0, 1].set(ylabel="Population [cm$^{-3}$]", title="Solved populations")
    axes[0, 1].legend(fontsize=6, ncol=3)
    axes[1, 1].plot(radius, profiles["ionized_fraction"])
    axes[1, 1].set(ylabel=r"$n_p/n_{\rm H}$", title="Solved ionization", ylim=(0, 1))
    for ax in axes[:, :2].flat:
        ax.set_xlabel("Radius / Rp")
    velocity, residual, error = load_observations(model.line_center_cm)
    axes[0, 2].errorbar(velocity, 100 * residual, yerr=100 * error,
                        fmt=".", color="0.5", markersize=2, label="Zenodo observation")
    axes[0, 2].plot(model.velocity_kms, 100 * spectrum, label="NLTE example (not fitted)")
    axes[0, 2].set(ylabel="Line contrast [%]", title="Observation for scale only")
    axes[0, 2].legend(fontsize=8)
    axes[1, 2].plot(model.velocity_kms, 1e6 * spectrum)
    axes[1, 2].set(ylabel="Line contrast [ppm]", title="Generated spectrum, enlarged")
    for ax in axes[:, 2]:
        ax.set(xlabel="Velocity from H-alpha rest center [km/s]", xlim=(-100, 100))
        ax.axhline(0, color="0.5", ls=":")
    fig.suptitle(f"Hydrogen through n={args.nmax}: prescribed radiation and temperature")
    fig.supxlabel("Data: Y. Zhang (2026), doi:10.5281/zenodo.20707255; CC BY 4.0", fontsize=8)
    fig.savefig(args.output / "nlte.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
