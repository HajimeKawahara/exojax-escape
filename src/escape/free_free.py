"""Shared electron-ion/neutral bremsstrahlung with C23 Gaunt factors.

The packaged van Hoof et al. tables and cubic log interpolation are adapted
from Cloudy C23.01 atmdat_gaunt.cpp. See data/free_free_provenance.json and
_licenses/Cloudy.txt. Electron-electron bremsstrahlung is not included.
"""

from __future__ import annotations

from importlib.resources import files

import jax.numpy as jnp
import numpy as np

from .hydrogen_continuum import hydrogen_free_free_coefficients, planck_intensity_hz
from .hydrogen_rates import BOLTZMANN_CGS, PLANCK_CGS

RYDBERG_ERG = 2.1798723611035e-11


class FreeFreeContinuum:
    """Common absorption/emission for charges 1--3, including neutral H."""

    def __init__(self):
        with files("escape").joinpath("data/free_free.npz").open("rb") as stream:
            with np.load(stream, allow_pickle=False) as data:
                self.log_gaunt = jnp.asarray(np.log(data["gaunt"]))
                self.gamma_min, self.u_min, self.step = map(float, data["grid"])

    def gaunt_factor(self, charge, temperature_k, frequency_hz):
        """Cubic interpolation of log(gff) on the original rectangular grid."""
        if charge not in (1, 2, 3):
            raise ValueError("The packaged free-free tables cover charges 1--3.")
        table = self.log_gaunt[charge-1]
        gamma = jnp.log10(charge**2*RYDBERG_ERG/(BOLTZMANN_CGS*temperature_k))
        u = jnp.log10(PLANCK_CGS*jnp.asarray(frequency_hz)/(BOLTZMANN_CGS*temperature_k))

        def stencil(coordinate, minimum, size):
            scaled = (coordinate-minimum)/self.step
            start = jnp.clip(jnp.floor(scaled).astype(int)-1, 0, size-4)
            index = start[..., None]+jnp.arange(4)
            offset = scaled[..., None]-index
            weights = jnp.stack([
                offset[..., 1]*offset[..., 2]*offset[..., 3]/-6,
                offset[..., 0]*offset[..., 2]*offset[..., 3]/2,
                offset[..., 0]*offset[..., 1]*offset[..., 3]/-2,
                offset[..., 0]*offset[..., 1]*offset[..., 2]/6,
            ], axis=-1)
            valid = (scaled >= 0) & (scaled <= size-1)
            return index, weights, valid

        ig, wg, vg = stencil(gamma, self.gamma_min, table.shape[1])
        iu, wu, vu = stencil(u, self.u_min, table.shape[0])
        value = jnp.exp(jnp.sum(table[iu[..., :, None], ig]*wu[..., :, None]*wg, axis=(-2, -1)))
        return jnp.where(vg & vu, value, jnp.nan)

    def coefficients(self, temperature_k, electron_cm3, ion_charge_cm3,
                     neutral_h_cm3, frequency_hz):
        """Return absorption [cm-1] and emissivity [erg/s/cm3/sr/Hz].

        ion_charge_cm3[q-1] sums ALL ions with charge q; include protons once.
        Neutral-H free-free follows C23's ratio 0.1175 sqrt(E/Ryd) to H+.
        H-minus bound-free belongs to the separate molecular network.
        Kirchhoff's law includes stimulated emission for each contribution.
        """
        if jnp.shape(ion_charge_cm3) != (3,):
            raise ValueError("Supply the total density for each of charges 1, 2, 3.")
        frequency = jnp.asarray(frequency_hz)
        gaunt = jnp.stack([self.gaunt_factor(q, temperature_k, frequency) for q in (1, 2, 3)])
        weighted = jnp.sum(jnp.asarray(ion_charge_cm3)[:, None]*jnp.array([1., 4., 9.])[:, None]*gaunt, axis=0)
        weighted += neutral_h_cm3*.1175*jnp.sqrt(PLANCK_CGS*frequency/RYDBERG_ERG)*gaunt[0]
        reference, _ = hydrogen_free_free_coefficients(temperature_k, electron_cm3, 1., frequency)
        absorption = reference*weighted
        return absorption, absorption*planck_intensity_hz(frequency, temperature_k)


__all__ = ["FreeFreeContinuum"]
