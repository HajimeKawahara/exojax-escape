"""Local pure-hydrogen statistical equilibrium with prescribed radiation.

The electron density is solved from charge and nucleus conservation. Bound
populations are linear at fixed electron density, leaving one scalar root.
This module supplies no atomic data, radiation transfer or energy equation.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class RateCoefficients(NamedTuple):
    """One cell's nonnegative rates; bound-bound arrays use [from, to].

    Bound-bound diagonals must be zero. The electron collision coefficient
    may include proton collisions because this pure-H model has ne = np.
    Radiation rates are prescribed independently of the solved populations.
    """

    radiative_bound_bound_s: jax.Array
    electron_bound_bound_cm3_s: jax.Array
    photoionization_s: jax.Array
    radiative_recombination_cm3_s: jax.Array
    electron_ionization_cm3_s: jax.Array
    three_body_recombination_cm6_s: jax.Array


class PopulationSolution(NamedTuple):
    bound_cm3: jax.Array
    proton_cm3: jax.Array
    electron_cm3: jax.Array
    ionized_fraction: jax.Array
    residual_norm: jax.Array
    conservation_error: jax.Array
    rate_residual_norm: jax.Array
    root_slope: jax.Array
    converged: jax.Array
    valid: jax.Array


def _bound_system(electron_density, rates):
    transition = (rates.radiative_bound_bound_s
                  + electron_density * rates.electron_bound_bound_cm3_s)
    ionization = (rates.photoionization_s
                  + electron_density * rates.electron_ionization_cm3_s)
    source = electron_density * (
        rates.radiative_recombination_cm3_s
        + electron_density * rates.three_body_recombination_cm6_s
    )
    loss = jnp.sum(transition, axis=1) + ionization
    matrix = jnp.diag(loss) - transition.T
    return matrix, source, transition, loss


def _bound_ratios(electron_density, rates):
    """Solve for b_i = n_i / np; each rate-equation row is scaled by loss."""
    matrix, source, _, loss = _bound_system(electron_density, rates)
    return jnp.linalg.solve(matrix / loss[:, None], source / loss)


def _bisect(function, initial, iterations):
    del initial
    lower, upper = jnp.asarray(-80.0), jnp.asarray(0.0)

    def step(_, interval):
        left, right = interval
        midpoint = (left + right) / 2
        below = function(midpoint) < 0
        return jnp.where(below, midpoint, left), jnp.where(below, right, midpoint)

    lower, upper = jax.lax.fori_loop(0, iterations, step, (lower, upper))
    return (lower + upper) / 2


@partial(jax.jit, static_argnames=("iterations",))
def solve_statistical_equilibrium(n_h_cm3, rates: RateCoefficients, *, iterations=80):
    """Solve one cell and differentiate the converged rate equations.

    Require float64, positive finite nH, nonnegative finite rates, and at
    least one positive photoionization rate. The bound-state loss matrix must
    be nonsingular throughout the bracket. The supported ionized fraction is
    exp(-80) <= np/nH <= 1. Exact zero-irradiation or disconnected systems are
    outside this initial domain; they can have neutral absorbing branches.

    ``rates`` contains two (N,N) arrays followed by four (N,) arrays. Use
    ``jax.vmap`` over independent cells. No density or rate is clipped.
    Invalid inputs or failed solves return NaN populations with
    ``converged=False``. Root and balance diagnostics remain available.

    For ne=np=nH*exp(z), solve M(ne)b=ne*alpha+ne**2*q3, where b_i=n_i/np.
    Nuclear conservation is F(z)=z+log(1+sum(b))=0. ``custom_root`` differentiates
    this residual, including all ne-dependent rates and the linear solve,
    rather than the bisection iterations. The diagnostics are dimensionless;
    ``rate_residual_norm`` scales each net bound rate by its total turnover.
    ``valid`` reports input/bracket checks, while ``converged`` also checks the
    final populations, conservation, balance and locally positive root slope.
    """
    rates = RateCoefficients(*(jnp.asarray(value, dtype=jnp.float64) for value in rates))
    return _solve_equilibrium_with_rates(n_h_cm3, lambda _: rates, iterations=iterations)


def _solve_equilibrium_with_rates(n_h_cm3, rates_at_density, *, iterations=80):
    """Shared scalar charge root; the callback is evaluated inside its Jacobian.

    Prescribed rates are a constant callback. Density-dependent screening or
    orbital mixing must instead use the trial electron density in every call.
    The caller supplies a JAX-traceable, side-effect-free coefficient function.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before solving statistical equilibrium.")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive static integer.")
    density = jnp.asarray(n_h_cm3, dtype=jnp.float64)
    if density.ndim != 0:
        raise ValueError("n_h_cm3 must be scalar; use jax.vmap for multiple cells.")
    rates = RateCoefficients(*(jnp.asarray(value, dtype=jnp.float64)
                               for value in rates_at_density(density)))
    size = rates.photoionization_s.size
    shapes = [(size, size)] * 2 + [(size,)] * 4
    if size < 1 or any(value.shape != shape for value, shape in zip(rates, shapes)):
        raise ValueError("Use two (N,N) bound-bound arrays and four (N,) continuum arrays.")

    def residual(log_ionized_fraction):
        electron_density = density * jnp.exp(log_ionized_fraction)
        ratios = _bound_ratios(electron_density, rates_at_density(electron_density))
        return log_ionized_fraction + jnp.log1p(jnp.sum(ratios))

    log_fraction = jax.lax.custom_root(
        residual, jnp.asarray(-1.0),
        lambda function, initial: _bisect(function, initial, iterations),
        lambda linear, rhs: rhs / linear(jnp.ones_like(rhs)),
    )
    ionized_fraction = jnp.exp(log_fraction)
    electron_density = density * ionized_fraction
    rates = rates_at_density(electron_density)
    ratios = _bound_ratios(electron_density, rates)
    bound = electron_density * ratios
    matrix, source, transition, loss = _bound_system(electron_density, rates)
    net = source - matrix @ ratios
    turnover = source + transition.T @ ratios + loss * ratios
    scaled_net = net / jnp.where(turnover > 0, turnover, 1.0)
    rate_residual = jnp.max(jnp.abs(scaled_net))
    conservation_error = jnp.abs(jnp.sum(bound) / density + ionized_fraction - 1)
    residual_norm = jnp.abs(residual(log_fraction))
    root_slope = jax.grad(residual)(log_fraction)
    finite_nonnegative = jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(value) & (value >= 0)) for value in rates
    ]))
    left, right = residual(jnp.asarray(-80.0)), residual(jnp.asarray(0.0))
    valid = (
        jnp.isfinite(density) & (density > 0) & finite_nonnegative
        & jnp.all(jnp.diag(rates.radiative_bound_bound_s) == 0)
        & jnp.all(jnp.diag(rates.electron_bound_bound_cm3_s) == 0)
        & jnp.any(rates.photoionization_s > 0)
        & jnp.isfinite(left) & jnp.isfinite(right) & (left <= 0) & (right >= 0)
    )
    converged = (
        valid & jnp.all(jnp.isfinite(bound)) & jnp.all(bound >= 0)
        & jnp.isfinite(root_slope) & (root_slope > 0)
        & (residual_norm < 1e-10) & (conservation_error < 1e-10)
        & (rate_residual < 1e-10)
    )
    value = lambda array: jnp.where(converged, array, jnp.nan)
    return PopulationSolution(
        value(bound), value(electron_density), value(electron_density),
        value(ionized_fraction), residual_norm, conservation_error,
        rate_residual, root_slope, converged, valid,
    )


@jax.jit
def relaxation_time_s(n_h_cm3, bound_cm3, rates: RateCoefficients):
    """Return the slowest local kinetic damping time at a supplied state.

    Pass converged bound populations and the same prescribed rates used for
    statistical equilibrium. Bound densities are the independent variables;
    ne=np=nH-sum(n_i) is differentiated inside the kinetic Jacobian. This
    removes the redundant nuclear-conservation mode and retains electron
    feedback in collisional ionization and both recombination processes.

    If every eigenvalue has a negative real part, return -1/max(Re(lambda))
    in seconds. Return infinity for a nondecaying or unstable mode, and NaN
    for invalid inputs or a nonfinite eigensolve. Near-zero eigenvalues can
    be numerically ill-conditioned. This optional diagnostic is separate
    from spectrum evaluation and does not establish local equilibrium along
    a flow; compare it with the relevant population-gradient transport time.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before evaluating relaxation times.")
    density = jnp.asarray(n_h_cm3, dtype=jnp.float64)
    bound = jnp.asarray(bound_cm3, dtype=jnp.float64)
    rates = RateCoefficients(*(jnp.asarray(value, dtype=jnp.float64) for value in rates))
    size = bound.size
    shapes = [(size, size)] * 2 + [(size,)] * 4
    if density.ndim != 0 or bound.shape != (size,) or size < 1:
        raise ValueError("Use one scalar density and a nonempty bound-population vector.")
    if any(value.shape != shape for value, shape in zip(rates, shapes)):
        raise ValueError("Rates must match the bound-population vector.")

    def kinetic_rhs(populations):
        electron_density = density - jnp.sum(populations)
        matrix, source, _, _ = _bound_system(electron_density, rates)
        return electron_density * source - matrix @ populations

    jacobian = jax.jacfwd(kinetic_rhs)(bound)
    eigenvalues = jnp.linalg.eigvals(jacobian)
    slowest = jnp.max(jnp.real(eigenvalues))
    stable = slowest < 0
    timescale = jnp.where(stable, -1 / jnp.where(stable, slowest, -1.0), jnp.inf)
    valid = (
        jnp.isfinite(density) & (density > 0)
        & jnp.all(jnp.isfinite(bound) & (bound >= 0)) & (jnp.sum(bound) <= density)
        & jnp.all(jnp.stack([jnp.all(jnp.isfinite(value) & (value >= 0)) for value in rates]))
        & jnp.all(jnp.diag(rates.radiative_bound_bound_s) == 0)
        & jnp.all(jnp.diag(rates.electron_bound_bound_cm3_s) == 0)
        & jnp.all(jnp.isfinite(eigenvalues))
    )
    return jnp.where(valid, timescale, jnp.nan)


__all__ = [
    "RateCoefficients", "PopulationSolution", "solve_statistical_equilibrium", "relaxation_time_s",
]
