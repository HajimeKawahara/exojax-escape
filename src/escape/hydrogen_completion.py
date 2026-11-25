"""Optional C23 highest-shell completion for a fixed hydrogen atom.

Altered from C23.01 iso_radiative_recomb.cpp:TempInterp and the hydrogen
TotalRecomb row in h_iso_recomb.dat. Cloudy license remains in escape.
The total is a finite n<=1000 completion convention, not exact infinity.
"""
from __future__ import annotations

import jax.numpy as jnp

_LOG_RR = (
    -9.892013, -10.031207, -10.17326, -10.31763, -10.463905, -10.611841,
    -10.761361, -10.912512, -11.065429, -11.220287, -11.37728, -11.536621,
    -11.698562, -11.863421, -12.031618, -12.203715, -12.380468, -12.56288,
    -12.752252, -12.950228, -13.158768, -13.380004, -13.615901, -13.867858,
    -14.1364, -14.421125, -14.720866, -15.033979, -15.358633, -15.693025,
    -16.035508, -16.384652, -16.739244, -17.098273, -17.460895, -17.826402,
    -18.194196, -18.56376, -18.934648, -19.306465, -19.693764,
)


def total_radiative_recombination(temperature_k):
    """C23 four-point interpolation of log RR in log temperature."""
    x = jnp.log10(temperature_k)
    start = jnp.clip(jnp.floor(x / .25).astype(int) - 1, 0, 37)
    index = start + jnp.arange(4)
    knots = index * .25 + jnp.where(index == 40, .009999999776482582, 0.)
    weights = jnp.ones(4)
    for i in range(4):
        for k in range(4):
            if i != k:
                weights = weights.at[i].multiply((x - knots[k]) / (knots[i] - knots[k]))
    table = jnp.asarray(_LOG_RR, dtype=x.dtype)
    rate = 10**jnp.sum(table[index] * weights)
    return jnp.where((temperature_k >= 4000) & (temperature_k <= 14000), rate, jnp.nan)


def highest_shell_scale(temperature_k, spontaneous_rr_cm3_s):
    """Scale the top BF kernel using its same finite-grid spontaneous RR.

    Every BF rate, opacity, Milne inverse and kinetic excess-energy integral
    must use this factor. Applying this function only to RR is invalid.
    A separate highest CI/three-body reservoir and heat exclusion are needed
    for the complete C23 topoff convention. Pressure-lowered cells must be
    rejected with critical_principal_n.
    """
    alpha = jnp.asarray(spontaneous_rr_cm3_s)
    deficit = jnp.maximum(total_radiative_recombination(temperature_k) - alpha.sum(), 0.)
    return jnp.ones_like(alpha).at[-1].set(1 + deficit / alpha[-1])


def critical_principal_n(temperature_k, electron_cm3, nuclei_cm3, ion_stark_cm3):
    """C23 continuum-lowering limit; use as a validity boundary, not truncation.

    ion_stark_cm3 is n(H+) + n(He+) for the solar case. A finite atom with
    highest shell nmax requires floor(n_critical) > nmax for topoff.
    """
    packing = jnp.sqrt(1.8887e8 / jnp.cbrt(nuclei_cm3))
    debye = 2.6e7 * (temperature_k / electron_cm3)**.25
    stark = 3171 * (electron_cm3 + ion_stark_cm3)**(-.1333)
    return jnp.floor(jnp.minimum(packing, jnp.minimum(debye, stark)))
