"""Spontaneous hydrogen 2s two-photon continuum, including its photon energy.

Altered JAX adaptation of Cloudy C23.01 atmdat_2photon.cpp and two_photon.cpp;
see _licenses/Cloudy.txt. The nine hydrogen shape samples are Goldman & Drake
(1981), Phys. Rev. A 24, 183, as retained by C23. Its natural cubic spline is
symmetric about half the transition frequency and normalized on the actual
quadrature. C23 disables induced two-photon transitions by default; this
module deliberately implements that spontaneous-only model.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .hydrogen_rates import PLANCK_CGS, TWO_PHOTON_A_S

_SHAPE = np.array(
    [0.0, 2.03239, 3.15792, 3.84452, 4.28435, 4.56958, 4.74855, 4.84732, 4.87892]
)
# Natural cubic coefficients; computed from the small published table, not fit.
_STEP = 1 / 16
_MATRIX = np.diag(np.full(7, 4.0)) + np.diag(np.ones(6), 1) + np.diag(np.ones(6), -1)
_SECOND = np.r_[0.0, np.linalg.solve(_MATRIX, 6 * np.diff(_SHAPE, n=2) / _STEP**2), 0.0]


def hydrogen_two_photon_shape(fraction):
    """Unnormalized C23 shape versus nu/nu_2s, zero outside 0<nu<nu_2s."""
    x = jnp.asarray(fraction)
    y = jnp.clip(jnp.minimum(x, 1 - x), 0.0, 0.5)
    index = jnp.minimum(jnp.floor(y / _STEP).astype(int), 7)
    b = (y - index * _STEP) / _STEP
    a = 1 - b
    shape, second = jnp.asarray(_SHAPE), jnp.asarray(_SECOND)
    value = (
        a * shape[index]
        + b * shape[index + 1]
        + ((a**3 - a) * second[index] + (b**3 - b) * second[index + 1]) * _STEP**2 / 6
    )
    return jnp.where((x > 0) & (x < 1), value, 0.0)


def two_photon_emissivity(frequency_hz, weight_hz, transition_frequency_hz, upper_cm3):
    """Return spontaneous j_nu [erg/s/cm^3/sr/Hz] for supplied 2s population.

    The integral of photon emissivity is exactly 2 n_2s A_2gamma on the chosen
    quadrature. Energy converges to n_2s A_2gamma h nu_2s as the symmetric
    spectrum is resolved; this is checked separately from normalization.
    """
    frequency, weights = jnp.asarray(frequency_hz), jnp.asarray(weight_hz)
    shape = hydrogen_two_photon_shape(frequency / transition_frequency_hz)
    density_per_hz = shape / jnp.sum(shape * weights)
    return (
        jnp.asarray(upper_cm3)[..., None]
        * 2
        * TWO_PHOTON_A_S
        * density_per_hz
        * PLANCK_CGS
        * frequency
        / (4 * jnp.pi)
    )
