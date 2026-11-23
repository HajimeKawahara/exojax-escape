"""Anderson n=4,5 data and optional Lebedev higher-shell electron collisions.

Data: Anderson et al. (2000), J. Phys. B 33, 1255, corrected in (2002),
J. Phys. B 35, 1613. Source: Cloudy C23.01 data/h_coll_str.dat, SHA256
f461454d45c2add1afee3350673351ad8fd3de4d84c51e63c3e0598b5875a70d.
The 13 rows sum every upper orbital sublevel, and every lower sublevel when
n_lower=4, following hydrocollid.cpp:HlikeCSInterp. These are collision
strength sums, independent of radiative branching ratios.

This is altered Cloudy data/code, not the original implementation.
Cloudy (C)1978--2023 Gary J. Ferland and others; see _licenses/Cloudy.txt.
The three original temperature columns cover the existing 4000--14000 K
model domain. Below 5802 K, linear extrapolation to Upsilon(0)=0 reproduces
Cloudy's approximation; it is not additional validated atomic data.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .hydrogen_rates import BOLTZMANN_CGS

# C23 uses these rounded temperature knots and linear interpolation.
_TEMPERATURE_K = np.array([5802., 11604., 34812.])
# Columns: upper state, lower state, Upsilon at the three temperatures.
_DATA = np.array([
    [6, 0, 0.1049, 0.1167, 0.1568],
    [6, 1, 2.381, 2.783, 4.368],
    [6, 2, 9.156, 10.551, 15.998],
    [6, 3, 20.37, 30.96, 68.5],
    [6, 4, 64.87, 97.78, 222.83],
    [6, 5, 117.71, 177.87, 434.05],
    [7, 0, 0.07605, 0.08407, 0.09667],
    [7, 1, 1.682, 1.933, 2.431],
    [7, 2, 6.407, 7.319, 9.1],
    [7, 3, 13.74, 17.54, 25.45],
    [7, 4, 46.32, 58.75, 84.91],
    [7, 5, 86.19, 111.6, 169.89],
    [7, 6, 1398.74, 2349.1, 4969.56],
])
_COLLISION_CONSTANT = 8.6291321808518402e-6  # C23 COLL_CONST, cgs
_ELECTRON_MASS_AMU = float(np.float32(9.1093837015e-28 * 6.02214076e23))
_HYDROGEN_MASS_AMU = float(np.float32(1.00794))
_REDUCED_MASS_G = (_ELECTRON_MASS_AMU * _HYDROGEN_MASS_AMU
                   / (_ELECTRON_MASS_AMU + _HYDROGEN_MASS_AMU) * 1.66053906660e-24)
_C23_RATE_MASS_FACTOR = (9.1093837015e-28 / _REDUCED_MASS_G)**1.5


def validate_shell_collision_model(model):
    if model not in ("vs80", "anderson_n45", "anderson_lebedev"):
        raise ValueError("shell_collision_model must be 'vs80', 'anderson_n45', or 'anderson_lebedev'.")


def lebedev_collision_strength(temperature_k, n_upper, n_lower, binding_lower_ryd):
    """C23 Lebedev--Beigman n-changing electron collision strength.

    Altered from hydro_vs_rates.cpp:hydro_Lebedev_deexcit and its C23 physical
    constants. The source attributes Eq. 8.30 of *Physics of Highly Excited
    Atoms and Ions* (Lebedev & Beigman, Springer, 1998). This shell formula
    is used only above n=5; it is not resolved low-level atomic data.
    ``binding_lower_ryd`` is the lower level's ionization energy in Rydbergs.
    The returned Upsilon is summed over both shells, before CSresolver.
    """
    temperature, hi, lo = map(jnp.asarray, (temperature_k, n_upper, n_lower))
    delta, total = hi - lo, hi + lo
    rydberg_temperature = 1.57887512402035907e5
    x = binding_lower_ryd * rydberg_temperature / temperature
    # The same Abramowitz--Stegun fits as C23 thirdparty.cpp:e1. Preserve
    # their piecewise approximation at x=1, rather than silently replacing
    # the reference kernel by another numerical special function.
    small = jnp.clip(x, 1e-300, 1.0)
    polynomial = jnp.polyval(
        jnp.array(
            [0.00107857, -0.00976004, 0.05519968, -0.24991055, 0.99999193, -0.57721566]
        ),
        small,
    )
    small_e1 = jnp.exp(small) * (polynomial - jnp.log(small))
    large = jnp.maximum(x, 1.0)
    numerator = jnp.polyval(
        jnp.array([1.0, 8.5733287401, 18.0590169730, 8.6347608925, 0.2677737343]), large
    )
    denominator = jnp.polyval(
        jnp.array([1.0, 9.5733223454, 25.6329561486, 21.0996530827, 3.9584969228]),
        large,
    )
    scaled_e1 = jnp.where(x < 1, small_e1, numerator / denominator / large)
    phi = 2 * (hi * lo / total**2) ** 2 / delta * (4 * delta - 1) * scaled_e1 / delta
    phi += (
        (2 * lo / total) ** 3
        * total
        * (delta - 0.6)
        * (1.33 + lo**2 * delta)
        * (1 - x * scaled_e1)
        / (delta * lo**2 * hi**2)
    )
    theta = temperature / rydberg_temperature
    root = jnp.sqrt(theta)
    factor = jnp.log1p(lo * theta / (delta * root + 2.5)) / jnp.log1p(lo * root / delta)
    rate_x_exp = (
        2
        * (5.29177210613775259e-9) ** 2
        * 7.29735256530521237e-3
        * 2.99792458e10
        * jnp.sqrt(jnp.pi)
    )
    rate_x_exp *= lo * (hi / delta) ** 3 / root * phi * factor
    strength = rate_x_exp / _COLLISION_CONSTANT * jnp.sqrt(temperature) * (2 * lo**2)
    valid = (
        (temperature >= 4000)
        & (temperature <= 14000)
        & (hi > lo)
        & (lo >= 1)
        & (binding_lower_ryd > 0)
    )
    return jnp.where(valid, strength, jnp.nan)


def replace_lebedev_high_shells(atom, temperature_k, coefficients):
    """Replace n>5 electron rates with C23's shell formula and CSresolver.

    A collapsed upper shell distributes collision strength among resolved
    lower l states as (2*l+1)/n_lower**2, not by radiative A branching.
    Upward rates use the actual selected atom energies for detailed balance.
    The selected model also applies C23 iso_collide.cpp's electron/atom
    reduced-mass correction to all n>=4 excitation rates (including Anderson).
    """
    upper, lower = atom.line_upper[5:], atom.line_lower[5:]
    hi, lo = atom.principal_n[upper], atom.principal_n[lower]
    # Same infinite-mass Rydberg as C23; binding energies remain those of the
    # selected atom (including NIST low-level differences).
    rydberg_erg = 2.17987236110358478e-11
    strength = lebedev_collision_strength(
        temperature_k,
        hi,
        lo,
        atom.ionization_energy_erg[lower] / rydberg_erg,
    )
    resolved = jnp.where(lo <= 3, atom.statistical_weights[lower] / (2 * lo**2), 1.0)
    down = (
        _COLLISION_CONSTANT
        * strength
        * resolved
        / (atom.statistical_weights[upper] * jnp.sqrt(temperature_k))
    )
    up = (
        down
        * atom.statistical_weights[upper]
        / atom.statistical_weights[lower]
        * jnp.exp(
            -(atom.excitation_energy_erg[upper] - atom.excitation_energy_erg[lower])
            / (BOLTZMANN_CGS * temperature_k)
        )
    )
    coefficients = coefficients.at[upper, lower].set(
        jnp.where(hi > 5, down, coefficients[upper, lower])
    )
    coefficients = coefficients.at[lower, upper].set(
        jnp.where(hi > 5, up, coefficients[lower, upper])
    )
    added = (atom.principal_n[:, None] >= 4) | (atom.principal_n[None, :] >= 4)
    return jnp.where(added, coefficients * _C23_RATE_MASS_FACTOR, coefficients)


def replace_anderson_n45(atom, temperature_k, coefficients):
    """Replace n=4,5 rates, retaining existing low and higher-shell models."""
    selected = _DATA[_DATA[:, 0] < len(atom.principal_n)]
    upper, lower = selected[:, 0].astype(int), selected[:, 1].astype(int)
    table = jnp.asarray(selected[:, 2:])
    strengths = jnp.asarray([jnp.interp(temperature_k, _TEMPERATURE_K, row) for row in table])
    strengths = jnp.where(temperature_k < _TEMPERATURE_K[0],
                          table[:, 0] * temperature_k / _TEMPERATURE_K[0], strengths)
    down = (_COLLISION_CONSTANT * strengths
            / (atom.statistical_weights[upper] * jnp.sqrt(temperature_k)))
    up = down * atom.statistical_weights[upper] / atom.statistical_weights[lower] * jnp.exp(
        -(atom.excitation_energy_erg[upper] - atom.excitation_energy_erg[lower])
        / (BOLTZMANN_CGS * temperature_k)
    )
    return coefficients.at[upper, lower].set(down).at[lower, upper].set(up)
