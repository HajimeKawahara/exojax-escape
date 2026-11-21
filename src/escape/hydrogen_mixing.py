"""Density-dependent proton mixing of the resolved hydrogen n=2,3 sublevels.

The PSM20 impact-parameter model is Badnell et al. (2021), MNRAS 507, 2922,
https://doi.org/10.1093/mnras/stab2266, equations 6--12 and 22. This is an
altered JAX adaptation of Cloudy C23.01 ``CS_l_mixing_PS64_expI`` (retained
notice: escape/_licenses/Cloudy.txt). Both Debye and lifetime cutoffs are
integrated at all densities. Unlike Cloudy's shortcut switching to a pure
Debye expression, this avoids an artificial jump at equal cutoff radii.
The target is neutral protium and the collider a proton; Cloudy instead uses
its standard hydrogen atomic weight. No numerical coefficient is fitted to
Cloudy populations. Other shells and electron l-changing collisions are not
added here. For n=2 this deliberately replaces the alternative Seaton/Huang
electron-plus-proton closure retained in ``hydrogen_rates``. It is a choice
of atomic model, not a correction to the published Huang coefficients.
Reciprocal rates use the actual finite-atom energy differences.
"""

from __future__ import annotations

from math import factorial

import jax
import jax.numpy as jnp
from jax.custom_batching import sequential_vmap
from jax.scipy.special import erf, exp1

from .hydrogen_rates import (
    BOLTZMANN_CGS, ELECTRON_MASS_CGS, LIGHT_SPEED_CGS, PLANCK_CGS,
)
from .parker import PROTON_MASS_CGS

BOHR_RADIUS_CM = 5.29177210903e-9
BOHR_TIME_S = 2.41889e-17
RYDBERG_TEMPERATURE_K = PLANCK_CGS * LIGHT_SPEED_CGS * 109737.3156816 / BOLTZMANN_CGS
HYDROGEN_PROTON_REDUCED_MASS_G = (
    PROTON_MASS_CGS * (PROTON_MASS_CGS + ELECTRON_MASS_CGS)
    / (2 * PROTON_MASS_CGS + ELECTRON_MASS_CGS)
)


@jax.custom_jvp
@sequential_vmap
def _exponential_integral_e1(x):
    """E1 with its analytic derivative, valid in nested implicit-root AD."""
    # Preserve scalar branches: vmap(exp1) also runs inactive slow iterations.
    return exp1(x)


@_exponential_integral_e1.defjvp
def _exponential_integral_e1_jvp(primals, tangents):
    (x,), (tangent,) = primals, tangents
    # Avoid differentiating expn's discrete order inside nested JAX transforms.
    return _exponential_integral_e1(x), -jnp.exp(-x) / x * tangent


def _debye_low_energy_term(x):
    """sqrt(pi)*erf(sqrt(x))/(2*x**1.5)-exp(-x)/x without cancellation."""
    small = jnp.minimum(x, .01)
    series = sum((-1)**k * 2 * small**k / (factorial(k) * (2 * k + 3))
                 for k in range(9))
    large = jnp.maximum(x, .01)
    direct = jnp.sqrt(jnp.pi) * erf(jnp.sqrt(large)) / (2 * large**1.5) - jnp.exp(-large) / large
    return jnp.where(x < .01, series, direct)


def _gamma3_scaled(x):
    """4*P(3,x)/x**3 with a stable small-x series, including its derivative."""
    small = jnp.minimum(x, .1)
    series = sum(2 * (-1)**k * small**k / (factorial(k) * (k + 3))
                 for k in range(11))
    large = jnp.maximum(x, .1)
    # Integer-order identity avoids a general incomplete-gamma AD graph.
    direct = 4 * (-jnp.expm1(-large) - jnp.exp(-large) * large * (1 + large / 2)) / large**3
    return jnp.where(x < .1, series, direct)


def proton_l_mixing_psm20(temperature_k, electron_cm3, n, l, l_lower, lifetime_s,
                         *, reduced_mass_g=HYDROGEN_PROTON_REDUCED_MASS_G):
    """Return the nl -> n(l-1) proton rate coefficient [cm^3/s].

    Electrons and protons have the same temperature. Electron density sets
    Debye screening; multiply the coefficient by proton density for a rate.
    The lifetime is that of the departing nl state. The combined Maxwellian
    integral uses min(R_D,R_lifetime(E)) at every collision energy. Gas-state
    and density derivatives therefore belong inside the population root.
    Positive scalar T, ne and lifetime and integer n>l>=1 are required.
    The rate model treats same-n states as degenerate for the cross section;
    the matrix helper supplies finite-energy detailed balance for the reverse.
    """
    temperature = jnp.asarray(temperature_k)
    density = jnp.asarray(electron_cm3)
    thermal_rydberg = temperature / RYDBERG_TEMPERATURE_K
    velocity = jnp.sqrt(thermal_rydberg * ELECTRON_MASS_CGS / reduced_mass_g)
    strength = 4.5 * n**2 * l * (n**2 - l**2)
    dipole = 2 * strength / (3 * (2 * l + 1))
    # P1=1/4 for a final-state-resolved transition; Cloudy's R12 is doubled.
    matching_radius2 = 8 * strength / (3 * velocity**2 * (2 * l + 1))
    debye_radius2 = thermal_rydberg / (8 * jnp.pi * density * BOHR_RADIUS_CM**3)
    lifetime_radius2 = (.72 * lifetime_s * velocity / BOHR_TIME_S)**2
    um = matching_radius2 / debye_radius2
    uc = debye_radius2 / lifetime_radius2
    # Compute directly to avoid cancellation of ne in sqrt(um*uc).
    uml = jnp.sqrt(matching_radius2 / lifetime_radius2)
    lifetime_integral = (_gamma3_scaled(uml) + 2 * _exponential_integral_e1(uml)
                         - _exponential_integral_e1(uc))
    # Extension for Uc<Uml, evaluated safely even when the other branch is used.
    uc_dense = jnp.minimum(uc, uml)
    um_dense = jnp.maximum(um, uml)
    dense_integral = (
        _gamma3_scaled(uc_dense) * (uc_dense / uml)**3
        + jnp.exp(-uc_dense) * (uc_dense / uml)**2 / uml
        - jnp.sqrt(jnp.pi) * erf(jnp.sqrt(uc_dense)) / (2 * um_dense**1.5)
        + _debye_low_energy_term(um_dense) + _exponential_integral_e1(um_dense)
    )
    bracket = jnp.where(uc >= uml, lifetime_integral, dense_integral)
    coefficient = 2 * BOHR_RADIUS_CM**3 * jnp.sqrt(jnp.pi) / (velocity * BOHR_TIME_S) * dipole * bracket
    valid = (jnp.isfinite(temperature) & (temperature > 0)
             & jnp.isfinite(density) & (density > 0)
             & jnp.isfinite(lifetime_s) & (lifetime_s > 0)
             & (n > l) & (l >= 1) & (l_lower == l - 1)
             & (n == jnp.floor(n)) & (l == jnp.floor(l))
             & jnp.isfinite(reduced_mass_g) & (reduced_mass_g > 0)
             & jnp.isfinite(coefficient) & (coefficient >= 0))
    return jnp.where(valid, coefficient, jnp.nan)


def mixing_bound_bound_cm3_s(atom, temperature_k, electron_cm3):
    """Return PSM20 n=2,3 proton coefficients in [from,to] order.

    The first six atom entries are the resolved 1s,2s,2p,3s,3p,3d states.
    Clear any existing 2s--2p block before adding this matrix: these entries
    replace the alternative Seaton/Huang coefficients, rather than supplement
    them. In pure hydrogen np=ne, so these proton coefficients join the
    effective electron matrix with their Debye-density dependence explicit.
    """
    size = atom.statistical_weights.size
    lifetime_rate = jnp.zeros(size).at[atom.line_upper].add(atom.einstein_a_s)
    upper, lower = jnp.array([2, 4, 5]), jnp.array([1, 3, 4])
    down = jax.vmap(lambda n, ell, tau: proton_l_mixing_psm20(
        temperature_k, electron_cm3, n, ell, ell - 1, tau,
    ))(jnp.array([2, 3, 3]), jnp.array([1, 1, 2]), 1 / lifetime_rate[upper])
    up = down * atom.statistical_weights[upper] / atom.statistical_weights[lower] * jnp.exp(
        -(atom.excitation_energy_erg[upper] - atom.excitation_energy_erg[lower])
        / (BOLTZMANN_CGS * temperature_k)
    )
    return jnp.zeros((size, size)).at[upper, lower].set(down).at[lower, upper].set(up)


__all__ = ["proton_l_mixing_psm20", "mixing_bound_bound_cm3_s"]
