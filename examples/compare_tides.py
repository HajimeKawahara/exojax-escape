"""Compare the same prescribed-temperature wind with and without linear tides."""

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

from exojax.utils.constants import G, MJ, kB
from escape.nonisothermal import nonisothermal_parker_velocity
from escape.parker import PROTON_MASS_CGS
from compare_parker import MASS_LOSS_G_S, MASS_MJ, MU, RADIUS_CM, TOP_K, temperature

# Kepler's law in the Mplanet << Mstar limit, for KELT-9 b's 1.48-day orbit.
TIDAL_COEFFICIENT_S2 = 3 * (2 * np.pi / (1.48 * 86400)) ** 2


def velocity(radius_cm, coefficient, ngrid=129, return_info=False):
    return nonisothermal_parker_velocity(
        radius_cm, temperature, TOP_K, MASS_MJ, MU,
        base_radius_cm=RADIUS_CM, outer_radius_cm=32 * RADIUS_CM,
        tidal_coefficient_s2=coefficient, ngrid=ngrid, return_info=return_info,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/tidal_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    radius = jnp.asarray(np.geomspace(1, 6, 300)) * RADIUS_CM
    coefficient = TIDAL_COEFFICIENT_S2
    original, original_info = velocity(radius, 0.0, return_info=True)
    tidal, tidal_info = velocity(radius, coefficient, return_info=True)
    finer = velocity(radius, coefficient, ngrid=257)
    derivative = jax.jacfwd(lambda scale: velocity(radius, coefficient * scale))(1.0)
    step = 1e-5
    difference = (velocity(radius, coefficient * (1 + step))
                  - velocity(radius, coefficient * (1 - step))) / (2 * step)
    density_factor = MASS_LOSS_G_S / (4 * jnp.pi * radius**2 * 1e5 * PROTON_MASS_CGS)
    density_original, density_tidal = density_factor / original, density_factor / tidal
    inward_gravity = G * MJ * MASS_MJ / radius**2
    effective_gravity = inward_gravity - coefficient * radius
    hill_radius = (G * MJ * MASS_MJ / coefficient) ** (1 / 3) / RADIUS_CM
    report = {
        "note": "Same prescribed T(r) and mass loss; linear 1D tides, not full Roche geometry or a fit.",
        "tidal_coefficient_s2": coefficient, "orbital_period_days": 1.48,
        "coefficient_approximation": "3*(2*pi/P)^2, neglecting planet/star mass ratio",
        "temperature_top_k": TOP_K, "mean_molecular_weight": MU,
        "planet_mass_mj": MASS_MJ, "planet_radius_cm": RADIUS_CM,
        "mass_loss_g_s": MASS_LOSS_G_S, "ngrid": 129,
        "sonic_radius_no_tide_rp": float(original_info["sonic_radius_cm"] / RADIUS_CM),
        "sonic_radius_tidal_rp": float(tidal_info["sonic_radius_cm"] / RADIUS_CM),
        "hill_radius_rp": hill_radius,
        "max_relative_velocity_grid_change": float(jnp.max(jnp.abs(finer / tidal - 1))),
        "max_relative_gradient_difference": float(jnp.max(jnp.abs(derivative / difference - 1))),
    }
    (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(args.output / "profiles.csv", np.column_stack([
        radius / RADIUS_CM, temperature(radius, TOP_K), original, tidal,
        density_original, density_tidal, effective_gravity, derivative, difference,
    ]), delimiter=",", comments="",
        header="radius_rp,temperature_k,no_tide_kms,tidal_kms,no_tide_nH_cm3,"
               "tidal_nH_cm3,effective_gravity_cm_s2,dv_dscale,finite_difference_dv_dscale")
    x = np.asarray(radius / RADIUS_CM)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(x, inward_gravity, "C0--", label="Planet gravity")
    axes[0, 0].plot(x, effective_gravity, "C3", label="Effective inward gravity")
    axes[0, 0].axhline(0, color="0.6", lw=0.8)
    axes[0, 0].axvline(hill_radius, color="0.4", ls=":", label=f"Hill scale: {hill_radius:.2f} Rp")
    axes[0, 0].set(xlim=(1, 3), ylabel="Inward acceleration [cm/s²]",
                   title="Linear tides weaken planetary gravity")
    for values, color, style, label, info in (
        (original, "C0", "--", "No tides", original_info),
        (tidal, "C3", "-", "With tides", tidal_info),
    ):
        sonic = float(info["sonic_radius_cm"] / RADIUS_CM)
        axes[0, 1].semilogy(x, values, color=color, ls=style,
                            label=f"{label}; sonic point {sonic:.2f} Rp")
        axes[0, 1].axvline(sonic, color=color, ls=":", alpha=0.6)
    sound = jnp.sqrt(kB * temperature(radius, TOP_K) / (MU * PROTON_MASS_CGS)) / 1e5
    axes[0, 1].semilogy(x, sound, color="0.4", ls="-.", label="Local sound speed")
    axes[0, 1].set(xlim=(1, 6), ylabel="Radial velocity [km/s]",
                   title="The sonic point moves inward")
    axes[1, 0].semilogy(x, density_original, "C0--", label="No tides")
    axes[1, 0].semilogy(x, density_tidal, "C3", label="With tides")
    axes[1, 0].set(xlim=(1, 3), ylabel="Hydrogen nuclei [cm$^{-3}$]",
                   title="Same mass loss: faster flow, lower density")
    axes[1, 1].plot(x, derivative, "C3", label="Implicit derivative")
    sample = np.arange(0, len(x), 12)
    axes[1, 1].plot(x[sample], np.asarray(difference)[sample], "ko", ms=3,
                    fillstyle="none", label="Finite difference")
    axes[1, 1].set(xlim=(1, 6), ylabel="dv / d(tidal scale) [km/s]",
                   title="Gradients include the moving sonic point")
    for axis in axes.flat:
        axis.set_xlabel("Radius / Rp")
        axis.legend(fontsize=8)
    fig.suptitle("Prescribed-temperature Parker wind with linear stellar tides")
    fig.supxlabel(
        "M = 2.17 Mj; Rp = 1.9 Rj; mu = 2/3; T rises from 4500 to 14000 K; mass loss = 1e13 g/s.\n"
        "1D approximation along the star-planet direction; outer curves illustrate the model continuation.",
        fontsize=9,
    )
    fig.savefig(args.output / "tidal_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
