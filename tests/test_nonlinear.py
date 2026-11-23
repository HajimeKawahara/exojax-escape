"""Nonlinear coupled roots and their implicit derivatives without dense matrices."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.nonlinear import implicit_newton_krylov, newton_krylov, newton_krylov_step

jax.config.update("jax_enable_x64", True)


def test_correction_merit_handles_a_weak_equation_and_nonlinear_curvature():
    # The exact root is (0, 0). A full Newton update first increases the raw
    # residual, but reduces the correction needed for the weak first row.
    def residual(x):
        return jnp.array([1e-6*x[0], x[1]+x[0]**2])

    initial = jnp.array([.1, -.01])
    step = jax.jit(lambda x: newton_krylov_step(
        residual, x, merit="correction", restart=2, krylov_iterations=2,
        linear_rtol=1e-8))(initial)
    assert step.accepted
    np.testing.assert_allclose(step.value, [0., .01], atol=2e-10)
    assert jnp.max(jnp.abs(step.residual)) > jnp.max(jnp.abs(residual(initial)))
    result = jax.jit(lambda x: newton_krylov(
        residual, x, merit="correction", iterations=3, tolerance=1e-12,
        restart=2, krylov_iterations=2, linear_rtol=1e-8))(initial)
    assert result.converged
    np.testing.assert_allclose(result.value, [0., 0.], atol=2e-10)


def test_correction_merit_preserves_implicit_forward_and_reverse_derivatives():
    def solution(parameter):
        return implicit_newton_krylov(
            lambda x: jnp.array([1e-6*(x[0]-parameter), x[1]+x[0]**2]),
            jnp.array([.1, -.01]), merit="correction", iterations=4,
            tolerance=1e-12, restart=2, krylov_iterations=2, linear_rtol=1e-8)

    parameter = .03
    np.testing.assert_allclose(jax.jit(solution)(parameter),
                               [parameter, -parameter**2], atol=2e-10)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(solution))(parameter),
                               [1., -2*parameter], rtol=2e-8)
    np.testing.assert_allclose(jax.jit(jax.jacrev(solution))(parameter),
                               [1., -2*parameter], rtol=2e-8)


def test_correction_merit_does_not_accept_an_invalid_domain_trial():
    def residual(x):
        result = jnp.array([1e-6*x[0], x[1]+x[0]**2])
        return jnp.where(x[1] <= 0., result, jnp.nan)

    initial = jnp.array([.1, -.01])
    step = jax.jit(lambda x: newton_krylov_step(
        residual, x, merit="correction", restart=2, krylov_iterations=2,
        linear_rtol=1e-8))(initial)
    assert step.accepted and step.value[1] <= 0.
    assert np.isfinite(step.residual).all()


def test_zero_iterations_only_evaluates_the_initial_residual():
    @jax.custom_jvp
    def residual(value):
        return value**2 - 4.0

    @residual.defjvp
    def unused_jvp(primals, tangents):
        raise AssertionError("An initial-residual check needs no Newton Jacobian.")

    check = jax.jit(lambda x: newton_krylov(residual, x, iterations=0))
    accepted = check(jnp.array([2.0]))
    assert accepted.converged and accepted.iterations == 0
    rejected = check(jnp.array([1.0]))
    assert not rejected.converged and rejected.residual_norm == 3.0


def test_zero_step_warm_root_has_implicit_not_initial_guess_derivatives():
    def solution(scale):
        return implicit_newton_krylov(
            lambda x: x**2 - scale * jnp.array([4.0, 9.0]),
            jnp.array([2.0, 3.0]), iterations=0, tolerance=1e-10,
            restart=2, krylov_iterations=2,
        )

    np.testing.assert_allclose(jax.jit(jax.jacfwd(solution))(1.0), [1.0, 1.5],
                               rtol=2e-12)
    derivative = jax.jit(jax.grad(lambda s: jnp.vdot(solution(s), jnp.array([1., 2.]))))
    np.testing.assert_allclose(derivative(1.0), 4.0, rtol=2e-12)
    assert np.isnan(jax.jit(solution)(1.01)).all()


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
