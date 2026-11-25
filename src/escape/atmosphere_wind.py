"""Connect sampled temperature and composition to the existing Parker solver.

Momentum depends on T/mu through P/rho = kB T/(mu mH). This module supplies
that profile to the prescribed-temperature solver; it does not solve an energy
or ionization equation. Mean molecular weight includes free electrons.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from exojax.utils.constants import kB

from .nonisothermal import nonisothermal_parker_velocity
from .parker import PROTON_MASS_CGS

__all__ = ["WindProfiles", "FlowTerms", "wind_from_profiles", "thermal_flow_terms"]


class WindProfiles(NamedTuple):
    velocity_kms: jax.Array
    density_g_cm3: jax.Array


class FlowTerms(NamedTuple):
    """Signed advective and expansion heating in erg cm^-3 s^-1."""

    advection: jax.Array
    expansion: jax.Array


def _effective_temperature(radius_cm, parameters):
    """C1 Hermite interpolation of log(T/mu), constant beyond either endpoint."""
    log_radius, log_temperature = parameters
    interior = ((log_temperature[2:] - log_temperature[:-2])
                / (log_radius[2:] - log_radius[:-2]))
    slope = jnp.concatenate((jnp.zeros(1), interior, jnp.zeros(1)))
    query = jnp.clip(jnp.log(radius_cm), log_radius[0], log_radius[-1])
    index = jnp.clip(jnp.searchsorted(log_radius, query, side="right") - 1,
                     0, log_radius.size - 2)
    width = log_radius[index + 1] - log_radius[index]
    fraction = (query - log_radius[index]) / width
    value = ((2 * fraction**3 - 3 * fraction**2 + 1) * log_temperature[index]
             + (fraction**3 - 2 * fraction**2 + fraction) * width * slope[index]
             + (-2 * fraction**3 + 3 * fraction**2) * log_temperature[index + 1]
             + (fraction**3 - fraction**2) * width * slope[index + 1])
    return jnp.exp(value)


def _profile_arrays(radius_cm, *profiles):
    radius = jnp.asarray(radius_cm)
    values = tuple(jnp.asarray(value) for value in profiles)
    if (radius.ndim != 1 or radius.size < 2
            or any(value.shape != radius.shape for value in values)):
        raise ValueError("Use matching one-dimensional profiles on at least two radii.")
    valid = (jnp.all(jnp.isfinite(radius)) & jnp.all(radius > 0)
             & jnp.all(jnp.diff(radius) > 0))
    for value in values:
        valid = valid & jnp.all(jnp.isfinite(value)) & jnp.all(value > 0)
    return radius, values, valid


@partial(jax.jit, static_argnames=("ngrid", "return_info"))
def wind_from_profiles(
    radius_cm, temperature_k, mean_molecular_weight, mass_loss_g_s, planet_mass_mj,
    *, base_radius_cm, outer_radius_cm, tidal_coefficient_s2=0.0,
    ngrid=129, return_info=False,
):
    """Return velocity [km/s] and mass density [g/cm^3] at ascending sample radii.

    Interpolate log(T/mu) with a cubic Hermite curve against log radius. Endpoint
    slopes are zero; the value stays constant outside the sampled interval up
    to the supplied wind-domain boundaries. This extension is an assumption,
    especially if the samples do not reach the sonic point. It is not a solved
    outer-atmosphere temperature. The entire wind domain must contain a single
    regular sonic point as required by ``nonisothermal_parker_velocity``.

    The existing implicit solver receives T/mu and mu=1, retaining the spatial
    derivative of composition in the pressure force. Density follows spherical
    mass conservation, rho = Mdot/(4 pi r^2 v), independent of composition.
    Invalid physical inputs or failed solves yield NaN profiles. Shape errors
    raise ValueError. With ``return_info=True``, return ``(profiles, info)``.
    """
    radius, (temperature, mu), valid = _profile_arrays(
        radius_cm, temperature_k, mean_molecular_weight,
    )
    parameters = (jnp.log(radius), jnp.log(temperature / mu))
    velocity, info = nonisothermal_parker_velocity(
        radius, _effective_temperature, parameters, planet_mass_mj, 1.0,
        base_radius_cm=base_radius_cm, outer_radius_cm=outer_radius_cm,
        tidal_coefficient_s2=tidal_coefficient_s2, ngrid=ngrid, return_info=True,
    )
    valid = valid & jnp.isfinite(mass_loss_g_s) & (mass_loss_g_s > 0)
    valid = valid & info["valid"]
    velocity = jnp.where(valid, velocity, jnp.nan)
    density = mass_loss_g_s / (4 * jnp.pi * radius**2 * velocity * 1e5)
    profiles = WindProfiles(velocity, density)
    if return_info:
        return profiles, {**info, "valid": valid}
    return profiles


def thermal_flow_terms(
    radius_cm, density_g_cm3, temperature_k, mean_molecular_weight, velocity_cm_s,
    *, log_velocity_gradient_cm1=None, sound_squared_gradient_cm_s2=None,
):
    """Return sunbather's signed advective and expansion thermal terms in cgs.

    For a monatomic gas (gamma=5/3), q_adv=-rho v d[3 kB T/(2 mu mH)]/dr
    and q_exp=[kB T v/(mu mH)] d(rho)/dr. A decreasing density produces negative
    expansion heating, i.e. cooling. Radiative thermal heating/cooling must
    satisfy q_adv + q_exp + Gamma - Lambda = 0 at equilibrium.

    For a mass-conserving Parker wind, supply both spatial derivatives from
    ``wind_from_profiles(..., return_info=True)``. Then d(rho)/dr follows
    rho * [-2/r - d(log v)/dr], and the sound-speed derivative uses the same
    interpolant as momentum. This avoids differencing a density that can fall
    by orders of magnitude between coarse atmosphere cells.

    Without those derivatives, retain three-point central differences on the
    nonuniform grid and one-sided endpoint differences for general sampled
    profiles. These terms include the changing particle count through mu but
    omit advected excitation/ionization energy; they match a local statistical-
    equilibrium thermal closure, not ``reactive_flow``'s chemical-energy model.
    """
    radius, (density, temperature, mu), valid = _profile_arrays(
        radius_cm, density_g_cm3, temperature_k, mean_molecular_weight,
    )
    velocity = jnp.asarray(velocity_cm_s)
    if velocity.shape != radius.shape:
        raise ValueError("Use matching one-dimensional profiles on at least two radii.")
    valid = valid & jnp.all(jnp.isfinite(velocity))
    sound_squared = kB * temperature / (mu * PROTON_MASS_CGS)
    if (log_velocity_gradient_cm1 is None) != (sound_squared_gradient_cm_s2 is None):
        raise ValueError("Supply both wind spatial derivatives, or neither.")
    if log_velocity_gradient_cm1 is None:
        sound_gradient = jnp.gradient(sound_squared, radius)
        density_gradient = jnp.gradient(density, radius)
    else:
        log_velocity_gradient = jnp.asarray(log_velocity_gradient_cm1)
        sound_gradient = jnp.asarray(sound_squared_gradient_cm_s2)
        if log_velocity_gradient.shape != radius.shape or sound_gradient.shape != radius.shape:
            raise ValueError("Wind spatial derivatives must match the profile shape.")
        valid &= jnp.all(jnp.isfinite(log_velocity_gradient) & jnp.isfinite(sound_gradient))
        density_gradient = -density * (2 / radius + log_velocity_gradient)
    advection = -1.5 * density * velocity * sound_gradient
    expansion = sound_squared * velocity * density_gradient
    return FlowTerms(jnp.where(valid, advection, jnp.nan),
                     jnp.where(valid, expansion, jnp.nan))
