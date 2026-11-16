"""Differentiable isothermal, spherical, transonic Parker wind.

The velocity follows steady mass and momentum conservation with constant
temperature and mean molecular weight. Stellar tides are omitted. See
https://p-winds.readthedocs.io/en/latest/_modules/parker.html for the physical
equation and an independent implementation. No velocity rescaling is applied.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from exojax.utils.constants import G, MJ, kB

PROTON_MASS_CGS = 1.67262192369e-24

__all__ = ["parker_velocity", "sonic_radius_cm"]


def _sonic_series(t):
    """Log Mach number near the sonic point, with t = log(r / rs)."""
    return t * (1 + t * (-0.5 + t * (0.25 + t * (-7 / 60 + t * 23 / 480))))


@jax.custom_jvp
def _log_mach(t):
    """Solve exp(2z) - 2z - 1 = 4[t + exp(-t) - 1] on the wind branch."""
    target = 4 * (t + jnp.expm1(-t))
    sign = jnp.where(t < 0, -1.0, 1.0)
    # Solve for |z|. Both selected branches increase monotonically in |z|.
    lower = jnp.zeros_like(t)
    upper = jnp.where(t < 0, (target + 1) / 2, jnp.log(2 * (target + 1)) / 2)

    def bisect(_, bounds):
        low, high = bounds
        midpoint = (low + high) / 2
        z = sign * midpoint
        value = jnp.expm1(2 * z) - 2 * z
        below = value < target
        return jnp.where(below, midpoint, low), jnp.where(below, high, midpoint)

    lower, upper = jax.lax.fori_loop(0, 64, bisect, (lower, upper))
    root = sign * (lower + upper) / 2
    # Avoid cancellation and the degenerate implicit root at r = rs.
    return jnp.where(jnp.abs(t) < 1e-3, _sonic_series(t), root)


@_log_mach.defjvp
def _log_mach_jvp(primals, tangents):
    (t,), (t_dot,) = primals, tangents
    z = _log_mach(t)
    near_sonic = jnp.abs(t) < 1e-3
    series_derivative = 1 + t * (-1 + t * (0.75 + t * (-7 / 15 + t * 23 / 96)))
    # dz/dt = 2(1 - exp(-t)) / (exp(2z) - 1), with limit 1 at t = 0.
    denominator = jnp.where(near_sonic, 1.0, jnp.expm1(2 * z))
    implicit_derivative = -2 * jnp.expm1(-t) / denominator
    derivative = jnp.where(near_sonic, series_derivative, implicit_derivative)
    return z, derivative * t_dot


@jax.jit
def sonic_radius_cm(temperature_k, planet_mass_mj, mean_molecular_weight):
    """Return the sonic radius in cm for positive inputs.

    Temperature is in K, planet mass in Jupiter masses, and mean molecular
    weight in proton-mass units. Mean molecular weight includes free electrons.
    """
    sound_speed_squared = kB * temperature_k / (mean_molecular_weight * PROTON_MASS_CGS)
    return G * MJ * planet_mass_mj / (2 * sound_speed_squared)


@jax.jit
def parker_velocity(radius_cm, temperature_k, planet_mass_mj, mean_molecular_weight):
    """Return radial velocity in km/s on the accelerating transonic branch.

    Inputs must be positive; radius is in cm, temperature in K, planet mass
    in Jupiter masses, and mean molecular weight in proton-mass units. Scalars
    and broadcast-compatible arrays are supported. Enable JAX float64 before
    use. The solution is subsonic below rs, sonic at rs, and supersonic above
    rs; its derivatives remain finite at the sonic point.
    """
    sound_speed = jnp.sqrt(kB * temperature_k / (mean_molecular_weight * PROTON_MASS_CGS))
    radius_sonic = sonic_radius_cm(temperature_k, planet_mass_mj, mean_molecular_weight)
    t = jnp.log(jnp.asarray(radius_cm) / radius_sonic)
    return sound_speed * jnp.exp(_log_mach(t)) / 1e5
