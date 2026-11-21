"""Transonic spherical winds with a prescribed, differentiable temperature.

Solve mass and momentum conservation for constant mean molecular weight.
Temperature is supplied, not determined by an energy equation. An optional
linear stellar tide reduces planetary gravity. Both the critical point and
the discretized wind use implicit differentiation through ``custom_root``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from exojax.utils.constants import G, MJ, kB

from .parker import PROTON_MASS_CGS, parker_velocity

__all__ = ["nonisothermal_parker_velocity"]


def _chebyshev_grid(size):
    """Lobatto nodes, differentiation matrix, and polynomial transform."""
    index = np.arange(size)
    nodes = (1 - np.cos(np.pi * index / (size - 1))) / 2
    weights = (-1.0) ** index
    weights[[0, -1]] *= 0.5
    difference = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(difference, 1)
    derivative = weights[None, :] / weights[:, None] / difference
    np.fill_diagonal(derivative, 0)
    np.fill_diagonal(derivative, -derivative.sum(axis=1))
    transform = 2 / (size - 1) * np.cos(
        np.pi * np.outer(index, index) / (size - 1)
    ) * (-1.0) ** index[:, None]
    transform[:, [0, -1]] *= 0.5
    transform[[0, -1], :] *= 0.5
    return tuple(jnp.asarray(value) for value in (nodes, derivative, transform))


def _polynomial(coefficients, coordinate):
    """Evaluate a Chebyshev series with Clenshaw's recurrence."""
    zeros = jnp.zeros_like(coordinate)

    def step(carry, coefficient):
        following, after = carry
        current = coefficient + 2 * coordinate * following - after
        return (current, following), None

    (first, second), _ = jax.lax.scan(step, (zeros, zeros), coefficients[:0:-1])
    return coefficients[0] + coordinate * first - second


def _bisect(function, initial, lower, upper):
    """Find the negative-to-positive root in a supplied bracket."""
    del initial

    def step(_, bracket):
        left, right = bracket
        midpoint = (left + right) / 2
        negative = function(midpoint) < 0
        return jnp.where(negative, midpoint, left), jnp.where(negative, right, midpoint)

    lower, upper = jax.lax.fori_loop(0, 64, step, (lower, upper))
    return (lower + upper) / 2


def _newton(function, initial):
    """Damped Newton steps for a square residual, with deterministic stopping."""
    def iteration(_, values):
        residual = function(values)
        norm = jnp.max(jnp.abs(residual))

        def update(values):
            correction = jnp.linalg.solve(jax.jacfwd(function)(values), -residual)

            def backtrack(index, best):
                candidate = values + correction * (0.5 ** index)
                candidate_norm = jnp.max(jnp.abs(function(candidate)))
                accept = candidate_norm < best[1]
                return (
                    jnp.where(accept, candidate, best[0]),
                    jnp.where(accept, candidate_norm, best[1]),
                )

            return jax.lax.fori_loop(0, 8, backtrack, (values, norm))[0]

        return jax.lax.cond(norm > 1e-9, update, lambda value: value, values)

    return jax.lax.fori_loop(0, 16, iteration, initial)


@partial(jax.jit, static_argnames=("temperature_fn", "ngrid", "return_info"))
def nonisothermal_parker_velocity(
    radius_cm,
    temperature_fn,
    temperature_params,
    planet_mass_mj,
    mean_molecular_weight,
    *,
    base_radius_cm,
    outer_radius_cm,
    tidal_coefficient_s2=0.0,
    ngrid=129,
    return_info=False,
):
    """Return a prescribed-temperature transonic wind in km/s.

    ``temperature_fn(radius_cm, temperature_params)`` must be a smooth scalar
    JAX function returning positive temperatures in K. Radius and the two
    domain bounds are in cm; mass is in Jupiter masses and mean molecular
    weight is in proton-mass units. Enable JAX float64 before use.

    ``tidal_coefficient_s2`` is the nonnegative coefficient kappa = 3 G Mstar
    / a_orbit**3 in s**-2: effective inward gravity is G Mp / r**2 - kappa r.
    This is the linear tide along the star-planet axis, used with spherical
    mass conservation, not a full Roche geometry. Zero recovers no tides.

    The supported domain has one regular wind critical point between the
    positive bounds, and the no-tide outer-temperature isothermal critical
    point must also lie in that interval. The temperature must be defined
    throughout this domain, even when requested radii lie below the sonic point.
    Inputs outside this domain or failed solves return NaN velocities.

    ``ngrid`` is an odd, static total node count, split at the critical point.
    No-tide isothermal initial values and eight continuation steps in both
    temperature and tidal strength select the wind branch. There is no state
    carried between calls. Derivatives follow the
    converged equations, including the critical radius and moving grid, not
    the Newton iterations. With ``return_info=True``, return ``(velocity,
    diagnostics)``; the residual is dimensionless. Grid convergence still
    needs checking for each temperature family.
    """
    if ngrid < 17 or ngrid % 2 != 1:
        raise ValueError("ngrid must be an odd integer of at least 17.")
    count = (ngrid + 1) // 2
    nodes, derivative, transform = _chebyshev_grid(count)
    reference_temperature = temperature_fn(outer_radius_cm, temperature_params)
    reference_speed = jnp.sqrt(
        kB * reference_temperature / (mean_molecular_weight * PROTON_MASS_CGS)
    )
    gravity = G * MJ * planet_mass_mj / (base_radius_cm * reference_speed**2)
    tide = tidal_coefficient_s2 * base_radius_cm**2 / reference_speed**2
    upper = jnp.log(outer_radius_cm / base_radius_cm)

    def sound_squared(log_radius, blend):
        temperature = temperature_fn(base_radius_cm * jnp.exp(log_radius), temperature_params)
        return 1 + blend * (temperature / reference_temperature - 1)

    def numerator(log_radius, blend):
        sound = sound_squared(log_radius, blend)
        slope = jax.grad(sound_squared, argnums=0)(log_radius, blend)
        return (
            2 - slope / sound - gravity * jnp.exp(-log_radius) / sound
            + blend * tide * jnp.exp(2 * log_radius) / sound
        )

    def critical_point(blend):
        equation = lambda value: numerator(value, blend)
        return jax.lax.custom_root(
            equation, jnp.log(gravity / 2),
            lambda function, initial: _bisect(function, initial, jnp.zeros_like(upper), upper),
            lambda linear, rhs: rhs / linear(jnp.ones_like(rhs)),
        )

    def grid(critical):
        inner = critical * nodes
        outer = critical + (upper - critical) * nodes
        return jnp.concatenate((inner, outer[1:]))

    def residual_function(critical, blend):
        radii = grid(critical)
        sound = jax.vmap(sound_squared, in_axes=(0, None))(radii, blend)
        right = jax.vmap(numerator, in_axes=(0, None))(radii, blend)
        sound_gradient = jax.vmap(
            jax.grad(sound_squared, argnums=0), in_axes=(0, None)
        )(radii, blend)
        beta = sound_gradient / sound
        critical_derivative = jax.grad(numerator, argnums=0)(critical, blend)
        sonic_beta = beta[count - 1]
        sonic_slope = (-sonic_beta + jnp.sqrt(sonic_beta**2 + 8 * critical_derivative)) / 4

        def residual(log_mach):
            inner_slope = derivative @ log_mach[:count] / critical
            outer_slope = derivative @ log_mach[count - 1:] / (upper - critical)
            slope = jnp.concatenate((inner_slope, outer_slope[1:]))
            # q = log(v/a), beta = d log(a^2)/d log(r).
            result = jnp.expm1(2 * log_mach) * (slope + beta / 2) - right
            # The sonic ODE row would be identically zero; impose its value.
            result = result.at[count - 1].set(log_mach[count - 1])
            # Use the two remaining boundary rows for regularity at the
            # sonic point, so both polynomial pieces have the same slope.
            result = result.at[0].set(inner_slope[-1] - sonic_slope)
            return result.at[-1].set(outer_slope[0] - sonic_slope)

        return residual

    isothermal_critical = jnp.log(gravity / 2)
    initial_radius = base_radius_cm * jnp.exp(grid(isothermal_critical))
    initial = jnp.log(
        1e5 * parker_velocity(
            initial_radius, reference_temperature, planet_mass_mj, mean_molecular_weight,
        )
        / reference_speed
    )
    critical = critical_point(1.0)
    residual = residual_function(critical, 1.0)

    def solve(function, guess):
        del function

        def continue_wind(index, values):
            blend = (index + 1) / 8
            return _newton(residual_function(critical_point(blend), blend), values)

        return jax.lax.fori_loop(0, 8, continue_wind, guess)

    log_mach = jax.lax.custom_root(
        residual, initial, solve,
        lambda linear, rhs: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs),
    )
    log_radius = jnp.log(jnp.asarray(radius_cm) / base_radius_cm)
    inside = log_radius <= critical
    # Inactive polynomials are evaluated at the sonic endpoint, avoiding
    # extrapolation overflow while retaining smooth one-sided derivatives.
    inner_coordinate = 2 * jnp.where(inside, log_radius, critical) / critical - 1
    outer_coordinate = (
        2 * (jnp.where(inside, critical, log_radius) - critical) / (upper - critical) - 1
    )
    inner_values = _polynomial(transform @ log_mach[:count], inner_coordinate)
    outer_values = _polynomial(transform @ log_mach[count - 1:], outer_coordinate)
    query_sound = jax.vmap(sound_squared, in_axes=(0, None))(
        log_radius.reshape(-1), 1.0,
    ).reshape(log_radius.shape)
    velocity = (
        reference_speed * jnp.sqrt(query_sound)
        * jnp.exp(jnp.where(inside, inner_values, outer_values)) / 1e5
    )

    residual_norm = jnp.max(jnp.abs(residual(log_mach)))
    radii = grid(critical)
    sound = jax.vmap(sound_squared, in_axes=(0, None))(radii, 1.0)
    mach = jnp.exp(log_mach)
    critical_slope = jax.grad(numerator, argnums=0)(critical, 1.0)
    valid = (
        (base_radius_cm > 0) & (outer_radius_cm > base_radius_cm)
        & (planet_mass_mj > 0) & (mean_molecular_weight > 0)
        & jnp.isfinite(tidal_coefficient_s2) & (tidal_coefficient_s2 >= 0)
        & (reference_temperature > 0) & jnp.all(sound > 0)
        & (isothermal_critical > 0) & (isothermal_critical < upper)
        & (numerator(jnp.zeros_like(upper), 1.0) < 0) & (numerator(upper, 1.0) > 0)
        & (critical_slope > 0)
        & jnp.all(mach[:count - 1] < 1) & jnp.all(mach[count:] > 1)
        & jnp.all(log_radius >= 0) & jnp.all(log_radius <= upper)
    )
    converged = jnp.isfinite(residual_norm) & (residual_norm < 1e-8)
    velocity = jnp.where(valid & converged, velocity, jnp.nan)
    if return_info:
        return velocity, {
            "sonic_radius_cm": base_radius_cm * jnp.exp(critical),
            "critical_slope": critical_slope,
            "residual_norm": residual_norm,
            "subsonic_margin": jnp.min(1 - mach[:count - 1]),
            "supersonic_margin": jnp.min(mach[count:] - 1),
            "converged": converged,
            "valid": valid,
        }
    return velocity
