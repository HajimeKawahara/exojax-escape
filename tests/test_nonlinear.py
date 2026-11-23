"""Nonlinear coupled roots and their implicit derivatives without dense matrices."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.nonlinear import implicit_newton_krylov, newton_krylov

jax.config.update("jax_enable_x64", True)


def test_manufactured_nonlinear_boundary_value_problem_and_gradient():
    size = 96
    grid = jnp.linspace(0.0, 1.0, size)
    target = 0.2 * jnp.sin(jnp.pi * grid)

    def operator(x):
        # Diffusion plus nonlinear local reaction, with zero ghost boundaries.
        laplacian = 2 * x - jnp.pad(x[:-1], (1, 0)) - jnp.pad(x[1:], (0, 1))
        return laplacian + jnp.sinh(x)

    @jax.jit
    def solution(amplitude):
        rhs = operator(amplitude * target)
        return implicit_newton_krylov(
            lambda x: operator(x) - rhs,
            jnp.zeros(size),
            restart=24,
            krylov_iterations=8,
            tolerance=1e-10,
        )

    actual = solution(1.0)
    np.testing.assert_allclose(actual, target, atol=2e-11)
    derivative = jax.jacfwd(solution)(1.0)
    np.testing.assert_allclose(derivative, target, rtol=3e-9, atol=2e-11)
    # Reverse-mode NUTS path must use the same implicit equation.
    gradient = jax.grad(lambda a: jnp.sum(solution(a) ** 2))(1.0)
    np.testing.assert_allclose(gradient, 2 * jnp.sum(target**2), rtol=3e-9)


def test_failed_nonlinear_or_linear_solve_is_not_marked_converged():
    result = newton_krylov(lambda x: x**2 + 1.0, jnp.array([1.0]), iterations=1)
    assert not result.converged
    assert result.residual_norm >= 1.0
    failed = implicit_newton_krylov(
        lambda x: x**2 + 1.0, jnp.array([1.0]), iterations=1
    )
    assert np.isnan(failed).all()


def test_exact_jvp_solver_matches_dense_hydrogen_population_root():
    from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
    from escape.hydrogen_nlte import _bound_ratios

    model = HydrogenEscapeProbabilityModel(
        [0.0, 1e4, 3e4], nmax=3, quadrature_order=2, angle_order=2
    )
    temperature = jnp.array([8000.0, 9000.0])
    incoming = model.boundary_blackbody(10000.0, 0.025)
    initial = jnp.tile(
        jnp.log(jnp.array([7e11, 1e5, 3e5, 1e3, 3e3, 5e3]) / 3e11), (2, 1)
    )

    def residual(flat, scale):
        state = flat.reshape(2, 6)
        fractions = jax.nn.softmax(jnp.column_stack((state, jnp.zeros(2))), axis=1)
        bound, electron = 1e12 * fractions[:, :-1], 1e12 * fractions[:, -1]
        rates = model.population_state(
            temperature * scale, bound, electron, electron, incoming
        )["rates"]
        return (state - jnp.log(jax.vmap(_bound_ratios)(electron, rates))).ravel()

    @jax.jit
    def root(scale):
        return implicit_newton_krylov(
            lambda x: residual(x, scale),
            initial.ravel(),
            restart=12,
            krylov_iterations=6,
            tolerance=1e-9,
        )

    iterative = root(1.0)
    dense = model.solve(1e12, temperature, incoming, initial)
    assert dense["converged"]
    np.testing.assert_allclose(iterative, dense["state"].ravel(), rtol=0.0, atol=2e-8)
    automatic = jax.grad(lambda scale: jnp.mean(root(scale)))(1.0)
    step = 1e-4
    finite = (jnp.mean(root(1 + step)) - jnp.mean(root(1 - step))) / (2 * step)
    assert np.isfinite(automatic) and abs(finite) > 0.01
    np.testing.assert_allclose(automatic, finite, rtol=3e-5, atol=2e-7)
