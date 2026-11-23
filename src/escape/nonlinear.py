"""Damped Newton roots using exact JAX Jacobian products and implicit AD.

No dense population Jacobian is formed. A failed linear solve or line search
is reported, never replaced by an unconverged physical state silently.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


class NewtonStep(NamedTuple):
    value: object
    residual: object
    linear_residual_norm: object
    accepted: object


class RootSolution(NamedTuple):
    value: object
    residual_norm: object
    iterations: object
    converged: object


def newton_krylov_step(
    function,
    value,
    *,
    restart=40,
    krylov_iterations=20,
    linear_rtol=1e-4,
    max_backtracks=24,
):
    """One exact-JVP Newton step, suitable for an external checkpointing loop.

    ``function`` and ``value`` are a flat residual and state vector. GMRES
    tolerance controls the Newton correction, not the final nonlinear root.
    Backtracking requires a strict decrease in the maximum residual norm.
    """
    residual, linear = jax.linearize(function, value)
    correction, _ = gmres(
        linear,
        -residual,
        tol=linear_rtol,
        atol=0.0,
        restart=restart,
        maxiter=krylov_iterations,
        solve_method="incremental",
    )
    linear_error = jnp.linalg.norm(linear(correction) + residual)
    linear_scale = jnp.linalg.norm(residual)
    linear_valid = jnp.all(jnp.isfinite(correction)) & (
        linear_error <= 5 * linear_rtol * linear_scale + 1e-13
    )
    norm = jnp.max(jnp.abs(residual))

    def condition(state):
        iteration, _, _, accepted = state
        return (iteration < max_backtracks) & ~accepted & linear_valid

    def backtrack(state):
        iteration, best, previous, _ = state
        candidate = value + 0.5**iteration * correction
        current = function(candidate)
        score = jnp.max(jnp.abs(current))
        accepted = jnp.isfinite(score) & (score < norm)
        return (
            iteration + 1,
            jnp.where(accepted, candidate, best),
            jnp.where(accepted, current, previous),
            accepted,
        )

    _, updated, final_residual, accepted = jax.lax.while_loop(
        condition,
        backtrack,
        (jnp.asarray(0), value, residual, jnp.asarray(False)),
    )
    return NewtonStep(updated, final_residual, linear_error, accepted)


def newton_krylov(
    function,
    initial,
    *,
    iterations=40,
    tolerance=1e-9,
    restart=40,
    krylov_iterations=20,
    linear_rtol=1e-4,
):
    """Native JAX primal solve; returns an explicit nonlinear convergence flag."""

    def condition(state):
        iteration, _, residual, progress = state
        return (
            (iteration < iterations)
            & progress
            & jnp.all(jnp.isfinite(residual))
            & (jnp.max(jnp.abs(residual)) > tolerance)
        )

    def advance(state):
        iteration, value, _, _ = state
        result = newton_krylov_step(
            function,
            value,
            restart=restart,
            krylov_iterations=krylov_iterations,
            linear_rtol=linear_rtol,
        )
        return iteration + 1, result.value, result.residual, result.accepted

    count, value, residual, _ = jax.lax.while_loop(
        condition,
        advance,
        (jnp.asarray(0), initial, function(initial), jnp.asarray(True)),
    )
    norm = jnp.max(jnp.abs(residual))
    return RootSolution(value, norm, count, jnp.isfinite(norm) & (norm <= tolerance))


def implicit_newton_krylov(
    function,
    initial,
    *,
    iterations=40,
    tolerance=1e-9,
    restart=40,
    krylov_iterations=20,
    linear_rtol=1e-4,
    tangent_rtol=1e-10,
):
    """Differentiate the complete root equation, not the Newton iterations.

    Tangents solve J dx = b with the same exact linear operator. Failed primal
    or tangent residual checks produce NaNs, so inference cannot accept an
    unconverged derivative. Call ``newton_krylov`` for detailed primal failure
    diagnostics and an inspectable unconverged state.
    """

    def solve(residual, guess):
        result = newton_krylov(
            residual,
            guess,
            iterations=iterations,
            tolerance=tolerance,
            restart=restart,
            krylov_iterations=krylov_iterations,
            linear_rtol=linear_rtol,
        )
        return jnp.where(result.converged, result.value, jnp.nan)

    def tangent_solve(linear, rhs):
        def checked_solve(operator, vector):
            value, _ = gmres(
                operator,
                vector,
                tol=tangent_rtol,
                atol=0.0,
                restart=restart,
                maxiter=krylov_iterations,
                solve_method="incremental",
            )
            norm = jnp.linalg.norm(operator(value) - vector)
            valid = jnp.isfinite(norm) & (
                norm <= 5 * tangent_rtol * jnp.linalg.norm(vector) + 1e-13
            )
            return jnp.where(valid, value, jnp.nan)

        # Keep the residual check inside the linear-solve primitive. An outer
        # where(condition(rhs), value, NaN) is not a linear tangent map and
        # cannot be transposed correctly for reverse-mode inference.
        return jax.lax.custom_linear_solve(
            linear,
            rhs,
            solve=checked_solve,
            transpose_solve=checked_solve,
        )

    return jax.lax.custom_root(function, initial, solve, tangent_solve)
