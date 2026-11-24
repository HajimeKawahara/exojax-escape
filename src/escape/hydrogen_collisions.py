"""Anderson n=4,5 data and optional Lebedev higher-shell electron collisions.

Data: Anderson et al. (2000), J. Phys. B 33, 1255, corrected in (2002),
J. Phys. B 35, 1613. Source: Cloudy C23.01 data/h_coll_str.dat, SHA256
f461454d45c2add1afee3350673351ad8fd3de4d84c51e63c3e0598b5875a70d.
The 13 rows sum every upper orbital sublevel, and every lower sublevel when
n_lower=4, following hydrocollid.cpp:HlikeCSInterp. These are collision
strength sums, independent of radiative branching ratios. The 74 original
n=4,5 orbital rows below support explicitly resolved atoms without imposing
statistical orbital populations.

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
# Original n4/5 nl data: same source hash and Cloudy license as the sums above.
# Quantum numbers followed by Upsilon at 5802, 11604, 34812 K.
_N45_ORBITAL_DATA = np.array([
    [4, 0, 1, 0, 0.0223, 0.0255, 0.0319],
    [4, 0, 2, 0, 0.392, 0.388, 0.508],
    [4, 0, 2, 1, 0.736, 0.771, 0.758],
    [4, 0, 3, 0, 2.62, 4.31, 10.8],
    [4, 0, 3, 1, 4.07, 4.88, 7.63],
    [4, 0, 3, 2, 4.21, 4.27, 4.35],
    [4, 1, 1, 0, 0.0403, 0.0479, 0.074],
    [4, 1, 2, 0, 0.77, 0.903, 1.45],
    [4, 1, 2, 1, 2.16, 2.26, 2.69],
    [4, 1, 3, 0, 4.5, 5.95, 15.2],
    [4, 1, 3, 1, 14.1, 20.7, 44.9],
    [4, 1, 3, 2, 13.7, 15.5, 19.7],
    [4, 2, 1, 0, 0.03, 0.0319, 0.0404],
    [4, 2, 2, 0, 0.654, 0.754, 1.12],
    [4, 2, 2, 1, 3.54, 4.31, 7.62],
    [4, 2, 3, 0, 6.41, 10.1, 22.9],
    [4, 2, 3, 1, 21.3, 31.3, 82.4],
    [4, 2, 3, 2, 34.4, 46.1, 83],
    [4, 3, 1, 0, 0.0123, 0.0114, 0.0105],
    [4, 3, 2, 0, 0.565, 0.738, 1.29],
    [4, 3, 2, 1, 2.72, 3.21, 4.93],
    [4, 3, 3, 0, 6.84, 10.6, 19.6],
    [4, 3, 3, 1, 25.4, 40.9, 87.9],
    [4, 3, 3, 2, 65.4, 112, 327],
    [5, 0, 1, 0, 0.0145, 0.0172, 0.0192],
    [5, 0, 2, 0, 0.201, 0.213, 0.233],
    [5, 0, 2, 1, 0.463, 0.51, 0.475],
    [5, 0, 3, 0, 1.04, 1.59, 2.91],
    [5, 0, 3, 1, 2.25, 2.58, 3.01],
    [5, 0, 3, 2, 2.78, 2.92, 2.53],
    [5, 0, 4, 0, 6.73, 15.8, 43.4],
    [5, 0, 4, 1, 11, 18.1, 29.5],
    [5, 0, 4, 2, 12.7, 16, 19],
    [5, 0, 4, 3, 10.1, 10.8, 9.56],
    [5, 1, 1, 0, 0.0269, 0.0315, 0.0404],
    [5, 1, 2, 0, 0.42, 0.518, 0.714],
    [5, 1, 2, 1, 1.32, 1.4, 1.37],
    [5, 1, 3, 0, 2.61, 3.46, 5.57],
    [5, 1, 3, 1, 7.47, 9.77, 14],
    [5, 1, 3, 2, 8.21, 8.88, 8.76],
    [5, 1, 4, 0, 9.81, 17.3, 42.5],
    [5, 1, 4, 1, 39.2, 75.4, 175],
    [5, 1, 4, 2, 41.9, 60.5, 84.1],
    [5, 1, 4, 3, 34.8, 40.5, 39.7],
    [5, 2, 1, 0, 0.0208, 0.0222, 0.0247],
    [5, 2, 2, 0, 0.435, 0.487, 0.556],
    [5, 2, 2, 1, 2.07, 2.48, 3.44],
    [5, 2, 3, 0, 3.38, 4.26, 6.14],
    [5, 2, 3, 1, 11.5, 15.3, 26.7],
    [5, 2, 3, 2, 16.9, 21, 27.3],
    [5, 2, 4, 0, 14.3, 25.4, 61.6],
    [5, 2, 4, 1, 41.2, 70.5, 186],
    [5, 2, 4, 2, 89.7, 159, 335],
    [5, 2, 4, 3, 75.1, 103, 132],
    [5, 3, 1, 0, 0.00919, 0.00914, 0.00952],
    [5, 3, 2, 0, 0.391, 0.461, 0.676],
    [5, 3, 2, 1, 1.71, 2.02, 2.9],
    [5, 3, 3, 0, 2.94, 3.41, 4.12],
    [5, 3, 3, 1, 10.4, 12.5, 16.5],
    [5, 3, 3, 2, 28.5, 40.1, 78.1],
    [5, 3, 4, 0, 22.2, 34.8, 63],
    [5, 3, 4, 1, 69.1, 115, 241],
    [5, 3, 4, 2, 107, 195, 537],
    [5, 3, 4, 3, 167, 271, 528],
    [5, 4, 1, 0, 0.00466, 0.00403, 0.00285],
    [5, 4, 2, 0, 0.235, 0.254, 0.252],
    [5, 4, 2, 1, 0.844, 0.909, 0.915],
    [5, 4, 3, 0, 3.77, 4.82, 6.71],
    [5, 4, 3, 1, 14.7, 18.6, 24.7],
    [5, 4, 3, 2, 29.8, 38.7, 53.2],
    [5, 4, 4, 0, 20.3, 27.6, 41.2],
    [5, 4, 4, 1, 69.6, 99.4, 163],
    [5, 4, 4, 2, 170, 261, 479],
    [5, 4, 4, 3, 387, 733, 1760],
])
_COLLISION_CONSTANT = 8.6291321808518402e-6  # C23 COLL_CONST, cgs
_ELECTRON_MASS_AMU = float(np.float32(9.1093837015e-28 * 6.02214076e23))
_HYDROGEN_MASS_AMU = float(np.float32(1.00794))
_REDUCED_MASS_G = (_ELECTRON_MASS_AMU * _HYDROGEN_MASS_AMU
                   / (_ELECTRON_MASS_AMU + _HYDROGEN_MASS_AMU) * 1.66053906660e-24)
_C23_RATE_MASS_FACTOR = (9.1093837015e-28 / _REDUCED_MASS_G)**1.5


def validate_shell_collision_model(model, *, resolved_nmax=3):
    if model not in ("vs80", "anderson_n45", "anderson_lebedev"):
        raise ValueError("shell_collision_model must be 'vs80', 'anderson_n45', or 'anderson_lebedev'.")
    if isinstance(resolved_nmax, int) and resolved_nmax > 3 and model == "vs80":
        raise ValueError("Resolved n4/5 requires Anderson n4/5 collision data.")


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
    resolved_lower = lo <= 3 if atom.resolved_nmax == 3 else atom.angular_l[lower] >= 0
    resolved = jnp.where(resolved_lower, atom.statistical_weights[lower] / (2 * lo**2), 1.0)
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
    if atom.resolved_nmax == 3:
        selected = _DATA[_DATA[:, 0] < len(atom.principal_n)]
        upper, lower = selected[:, 0].astype(int), selected[:, 1].astype(int)
        table = jnp.asarray(selected[:, 2:])
    else:
        data = {tuple(map(int, row[:4])): row[4:] for row in _N45_ORBITAL_DATA}
        upper, lower, values = [], [], []
        members = atom.orbital_members
        for high, high_members in enumerate(members):
            nh = high_members[0][0]
            if not 4 <= nh <= 5:
                continue
            for low in range(high):
                low_members = members[low]
                nl = low_members[0][0]
                if nl == nh:
                    continue
                upper.append(high)
                lower.append(low)
                values.append(sum(data[(nh, lh, nl, ll)]
                                  for _, lh in high_members for _, ll in low_members))
        upper, lower, table = jnp.asarray(upper), jnp.asarray(lower), jnp.asarray(values)
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
