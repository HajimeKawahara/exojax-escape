"""Anderson electron collision strengths for the collapsed H n=4,5 shells.

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


def validate_shell_collision_model(model):
    if model not in ("vs80", "anderson_n45"):
        raise ValueError("shell_collision_model must be 'vs80' or 'anderson_n45'.")


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
