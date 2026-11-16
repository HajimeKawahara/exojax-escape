"""Compare prescribed-temperature and isothermal Parker winds, without fitting."""

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

from exojax.utils.constants import RJ, kB
from escape.nonisothermal import nonisothermal_parker_velocity
from escape.parker import PROTON_MASS_CGS, parker_velocity

RADIUS_CM = 1.9 * RJ
MASS_MJ = 2.17
MU = 2 / 3
TOP_K = 14000.0
MASS_LOSS_G_S = 1e13


def temperature(radius_cm, top_k):
    """The same smooth heating profile used by the spectral examples."""
    return 4500 + (top_k - 4500) * (
        1 - jnp.exp(-(radius_cm / RADIUS_CM - 1) / 0.12)
    )


def velocity(radius_cm, top_k, ngrid=129, return_info=False):
    return nonisothermal_parker_velocity(
        radius_cm, temperature, top_k, MASS_MJ, MU,
        base_radius_cm=RADIUS_CM, outer_radius_cm=32 * RADIUS_CM,
        ngrid=ngrid, return_info=return_info,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/parker_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    radius = jnp.asarray(np.geomspace(1, 20, 400)) * RADIUS_CM
    wind, info = velocity(radius, TOP_K, return_info=True)
    initial = parker_velocity(radius, TOP_K, MASS_MJ, MU)
    finer = velocity(radius, TOP_K, ngrid=257)
    derivative = jax.jacfwd(lambda top: velocity(radius, top))(TOP_K)
    step = 0.1
    finite_difference = (velocity(radius, TOP_K + step) - velocity(radius, TOP_K - step)) / (2 * step)
    density = MASS_LOSS_G_S / (4 * jnp.pi * radius**2 * wind * 1e5 * PROTON_MASS_CGS)
    density_initial = MASS_LOSS_G_S / (4 * jnp.pi * radius**2 * initial * 1e5 * PROTON_MASS_CGS)
    columns = np.column_stack([
        radius / RADIUS_CM, temperature(radius, TOP_K), initial, wind,
        density_initial, density, derivative, finite_difference,
    ])
    np.savetxt(args.output / "profiles.csv", columns, delimiter=",", comments="",
               header="radius_rp,temperature_k,isothermal_kms,nonisothermal_kms,"
                      "isothermal_nH_cm3,nonisothermal_nH_cm3,dv_dT,finite_difference_dv_dT")
    report = {
        "note": "Prescribed temperature, pure hydrogen, no tides or energy equation; not a fit.",
        "temperature_top_k": TOP_K, "mean_molecular_weight": MU,
        "planet_mass_mj": MASS_MJ, "mass_loss_g_s": MASS_LOSS_G_S,
        "ngrid": 129, "sonic_radius_rp": float(info["sonic_radius_cm"] / RADIUS_CM),
        "max_relative_velocity_grid_change": float(jnp.max(jnp.abs(finer / wind - 1))),
        "max_relative_gradient_difference": float(jnp.max(jnp.abs(derivative / finite_difference - 1))),
    }
    (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    radius_rp = np.asarray(radius / RADIUS_CM)
    sonic_rp = report["sonic_radius_rp"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(radius_rp, temperature(radius, TOP_K), color="C3", label="Prescribed T(r)")
    axes[0, 0].axhline(TOP_K, color="C0", ls="--", label="Isothermal starting solution")
    axes[0, 0].set(xlim=(1, 3), xlabel="Radius / Rp", ylabel="Temperature [K]",
                   title="One temperature profile for pressure and line formation")
    axes[0, 0].legend(fontsize=8)
    for flux, label, color, style in ((initial, "Isothermal", "C0", "--"),
                                     (wind, "Nonisothermal", "C3", "-")):
        axes[0, 1].plot(radius_rp, flux, color=color, ls=style, label=label)
    sound_speed = jnp.sqrt(kB * temperature(radius, TOP_K) / (MU * PROTON_MASS_CGS)) / 1e5
    axes[0, 1].plot(radius_rp, sound_speed, color="0.4", ls="-.", lw=1,
                    label="Local sound speed")
    axes[0, 1].axvline(sonic_rp, color="0.4", ls=":", label=f"Sonic point: {sonic_rp:.2f} Rp")
    axes[0, 1].axvspan(1, 3, color="0.5", alpha=0.1, label="Spectral model region")
    axes[0, 1].set(xlabel="Radius / Rp", ylabel="Radial velocity [km/s]", xlim=(1, 20),
                   xscale="log", yscale="log",
                   title="Smooth transition through the sonic point")
    axes[0, 1].set_xticks([1, 2, 3, 6, 10, 20], labels=["1", "2", "3", "6", "10", "20"])
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].semilogy(radius_rp, density_initial, color="C0", ls="--", label="Isothermal")
    axes[1, 0].semilogy(radius_rp, density, color="C3", label="Nonisothermal")
    axes[1, 0].set(xlim=(1, 3), xlabel="Radius / Rp", ylabel="Hydrogen nuclei [cm$^{-3}$]",
                   title="Same mass loss; density follows 1 / (r² v)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(radius_rp, derivative, color="C3", label="Implicit derivative")
    sample = np.arange(0, len(radius_rp), 16)
    axes[1, 1].plot(radius_rp[sample], np.asarray(finite_difference)[sample], "o",
                    color="black", ms=3, fillstyle="none", label="Finite difference")
    axes[1, 1].set(xlabel="Radius / Rp", ylabel="dv / dTtop [km/s/K]", xlim=(1, 20),
                   title="Differentiate the converged wind")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Nonisothermal Parker wind from a prescribed temperature profile")
    fig.supxlabel("M = 2.17 Mj; Rp = 1.9 Rj; mu = 2/3; mass loss = 1e13 g/s. No stellar tides.", fontsize=9)
    fig.savefig(args.output / "parker_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
