"""Convert a tabulated stellar surface flux to a prescribed photon field."""

from __future__ import annotations

import jax.numpy as jnp

from .hydrogen_rates import LIGHT_SPEED_CGS, PLANCK_CGS


def surface_flux_occupation(frequency_hz, surface_frequency_hz, surface_flux_nu,
                            *, dilution=0.025):
    """Interpolate surface F_nu and return occupation for J_nu = W F_nu/pi.

    Frequencies are vacuum Hz; surface flux is erg s^-1 cm^-2 Hz^-1, already
    integrated over the outward hemisphere. This adopts a uniformly bright
    stellar disk with I_nu=F_nu/pi and geometrical dilution W. It does not
    attenuate the incident field or include atmospheric diffuse radiation.
    The surface grid must be strictly increasing. Outside coverage returns
    NaN: completing the UV/IR tails is an explicit caller choice. The result
    remains differentiable in flux and dilution; line-profile averaging, if
    needed, must be performed by the caller.
    """
    frequency = jnp.asarray(frequency_hz)
    grid, flux = jnp.asarray(surface_frequency_hz), jnp.asarray(surface_flux_nu)
    if grid.ndim != 1 or grid.size < 2 or flux.shape != grid.shape:
        raise ValueError("Surface frequency and flux require matching 1-D grids.")
    valid = (jnp.all(jnp.isfinite(grid)) & jnp.all(grid > 0)
             & jnp.all(jnp.diff(grid) > 0)
             & jnp.all(jnp.isfinite(flux)) & jnp.all(flux >= 0)
             & jnp.isfinite(dilution) & (dilution >= 0)
             & jnp.all(jnp.isfinite(frequency)) & jnp.all(frequency > 0))
    sampled = jnp.interp(frequency, grid, flux, left=jnp.nan, right=jnp.nan)
    occupation = dilution * LIGHT_SPEED_CGS**2 * sampled / (
        2 * jnp.pi * PLANCK_CGS * frequency**3
    )
    return jnp.where(valid, occupation, jnp.nan)
