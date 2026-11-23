"""Pure-hydrogen continuity and an isolated two-level excitation closure.

This is not an ionization or multilevel NLTE solver. The radiation field is
externally specified; line trapping and population feedback from RT are absent.
"""

from __future__ import annotations

import jax.numpy as jnp

from exojax.utils.constants import hcperk
from .parker import PROTON_MASS_CGS

# Sum of the 15 n=3 -> n=2 effective strengths at kT=1 eV, assuming
# statistical populations within each shell: Lehmann et al. (2020), Eq. 49
# and Table B.3 (HTML Table 7), using Anderson et al. (2002) data.
# https://arxiv.org/html/2010.01042v1#A2
# Frozen at 1 eV: this is not a temperature-dependent collision calculation.
COLLISION_STRENGTH_23 = 35.52


def hydrogen_nuclei_density(radius_cm, velocity_kms, mass_loss_g_s):
    """Return nH [cm^-3] for steady spherical pure-H flow, with v > 0."""
    return mass_loss_g_s / (
        4 * jnp.pi * radius_cm**2 * velocity_kms * 1e5 * PROTON_MASS_CGS
    )


def diluted_blackbody_occupation(wavenumber_cm, temperature_k, dilution):
    """Return photon occupation for J = dilution * B(T), with positive T."""
    return dilution / jnp.expm1(hcperk * wavenumber_cm / temperature_k)


def two_level_ratio(
    temperature_k, electron_density_cm3, wavenumber_cm, einstein_a,
    g_lower, g_upper, collision_strength, photon_occupation,
):
    """Return n_upper/n_lower from electron collisions and Einstein rates.

    The Maxwellian de-excitation coefficient is 8.629e-6 Upsilon/(g_u sqrt(T))
    in cm^3/s. Excitation follows detailed balance. Stimulated emission is
    included. Require T,A,g > 0 and ne,Upsilon,occupation >= 0; no clipping
    is applied. The result is finite at zero density and has no inversion.
    """
    collision_down = (
        electron_density_cm3 * 8.629e-6 * collision_strength
        / (g_upper * jnp.sqrt(temperature_k))
    )
    boltzmann = jnp.exp(-hcperk * wavenumber_cm / temperature_k)
    return (g_upper / g_lower) * (
        collision_down * boltzmann + einstein_a * photon_occupation
    ) / (collision_down + einstein_a * (1 + photon_occupation))
