"""Independent SciPy evaluation of the Cloudy C23 PSM20 cutoff convention.

This altered diagnostic adaptation of CS_l_mixing_PS64_expI retains Cloudy's
thermal-cutoff shortcut and standard hydrogen atomic weight. Its purpose is
to explain differences from the continuous H-1 runtime model, not to fit them.
See Badnell et al. (2021), MNRAS507,2922, doi:10.1093/mnras/stab2266;
the retained Cloudy notice is src/escape/_licenses/Cloudy.txt.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf, exp1, gammainc

BOHR_RADIUS_CM = 5.29177210903e-9
BOHR_TIME_S = 2.41889e-17
RYDBERG_TEMPERATURE_K = 157887.512402
ELECTRON_MASS_G = 9.1093837015e-28
ATOMIC_MASS_G = 1.66053906660e-24
HYDROGEN_MASS_U = 1.00794
PROTON_MASS_U = HYDROGEN_MASS_U - 5.48579909065e-4
REDUCED_MASS_G = HYDROGEN_MASS_U * PROTON_MASS_U / (HYDROGEN_MASS_U + PROTON_MASS_U) * ATOMIC_MASS_G


def cloudy_psm20_convention(temperature_k, electron_cm3, ell, lower_ell, lifetime_s):
    """Return (Cloudy shortcut, combined integral) [cm^3/s] and cutoff ratio."""
    n = 3
    velocity = np.sqrt(temperature_k / RYDBERG_TEMPERATURE_K / (REDUCED_MASS_G / ELECTRON_MASS_G))
    strength = 9 * n**2 * max(ell, lower_ell) * (n**2 - max(ell, lower_ell)**2) / 2
    impact_squared = 2 / (3 * .5 * velocity**2 * (2 * ell + 1)) * strength * 2
    debye_squared = temperature_k / RYDBERG_TEMPERATURE_K / (8 * np.pi * electron_cm3 * BOHR_RADIUS_CM**3)
    lifetime_squared = (.72 * lifetime_s / BOHR_TIME_S * velocity)**2
    lower = impact_squared / debye_squared
    crossover = debye_squared / lifetime_squared
    lifetime_lower = np.sqrt(lower * crossover)
    small_energy = (np.sqrt(np.pi / lower**3) * erf(np.sqrt(lower)) / 2 - np.exp(-lower) / lower
                    if lower > 1e-6 else 2 / 3)
    if crossover >= lifetime_lower:
        combined = 4 * gammainc(3, lifetime_lower) / lifetime_lower**3 + 2 * exp1(lifetime_lower) - exp1(crossover)
    else:
        combined = (4 * (gammainc(3, crossover) + .25 * np.exp(-crossover) * crossover**2) / lifetime_lower**3
                    - np.sqrt(np.pi / lower**3) * erf(np.sqrt(crossover)) / 2 + small_energy + exp1(lower))
    debye = small_energy + exp1(lower)
    scale = 2 * BOHR_RADIUS_CM**3 * np.sqrt(np.pi) / velocity / BOHR_TIME_S * 2 * strength / (3 * (2 * ell + 1))
    return scale * (debye if crossover <= 1 else combined), scale * combined, crossover
