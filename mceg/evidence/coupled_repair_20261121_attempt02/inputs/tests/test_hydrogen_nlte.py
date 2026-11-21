"""Independent conservation, equilibrium and derivative checks for the SE solver."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_nlte import (
    RateCoefficients, relaxation_time_s, solve_statistical_equilibrium,
)


jax.config.update("jax_enable_x64", True)


def one_level(gamma=0.2, alpha=1e-12, ionization=0.0, three_body=0.0):
    return RateCoefficients(
        jnp.zeros((1, 1)), jnp.zeros((1, 1)),
        jnp.asarray([gamma]), jnp.asarray([alpha]),
        jnp.asarray([ionization]), jnp.asarray([three_body]),
    )


def thermal_rates(temperature=10000.0):
    """Synthetic rates with exact Saha/Boltzmann balance, not an atomic dataset."""
    electron_mass, planck, boltzmann, electron_volt = (
        9.1093837139e-28, 6.62607015e-27, 1.380649e-16, 1.602176634e-12,
    )
    weights = np.array([2.0, 8.0, 18.0])
    binding = np.array([13.6, 3.4, 13.6 / 9]) * electron_volt
    saha = weights / 2 * (planck**2 / (2 * np.pi * electron_mass * boltzmann * temperature))**1.5
    saha *= np.exp(binding / (boltzmann * temperature))
    collisions = np.zeros((3, 3))
    for upper in range(1, 3):
        for lower in range(upper):
            collisions[upper, lower] = (1 + upper + lower) * 1e-8
            collisions[lower, upper] = collisions[upper, lower] * saha[upper] / saha[lower]
    gamma = np.array([0.2, 1.5, 4.0])
    ionization = np.array([1e-11, 2e-9, 3e-8])
    return RateCoefficients(
        jnp.zeros((3, 3)), jnp.asarray(collisions), jnp.asarray(gamma),
        jnp.asarray(saha * gamma), jnp.asarray(ionization), jnp.asarray(saha * ionization),
    ), saha


@pytest.mark.parametrize("density", [1e2, 1e8, 1e12, 1e18])
def test_one_level_photoionization_matches_analytic_quadratic(density):
    gamma, alpha = 0.2, 1e-12
    result = solve_statistical_equilibrium(density, one_level(gamma, alpha))
    x = 2 / (1 + np.sqrt(1 + 4 * alpha * density / gamma))
    assert result.converged and result.valid
    np.testing.assert_allclose(result.ionized_fraction, x, rtol=2e-14)
    np.testing.assert_allclose(result.bound_cm3[0], density * x**2 * density * alpha / gamma, rtol=4e-14)
    np.testing.assert_allclose(result.proton_cm3, result.electron_cm3, rtol=0, atol=0)
    assert result.conservation_error < 2e-14
    assert result.rate_residual_norm < 2e-14


def test_ionization_and_three_body_terms_match_independent_cubic_root():
    density, gamma, alpha, ionization, three_body = 4e11, 0.2, 1e-12, 3e-13, 1e-25
    result = solve_statistical_equilibrium(density, one_level(gamma, alpha, ionization, three_body))
    roots = np.roots([
        density**2 * three_body,
        density * (alpha + ionization), gamma - density * ionization, -gamma,
    ])
    expected = [r.real for r in roots if abs(r.imag) < 1e-14 and 0 < r.real < 1]
    assert len(expected) == 1 and result.converged
    np.testing.assert_allclose(result.ionized_fraction, expected[0], rtol=2e-14)


@pytest.mark.parametrize("temperature", [6000.0, 10000.0, 14000.0])
@pytest.mark.parametrize("density", [1e8, 1e12, 1e16])
def test_multilevel_detailed_balance_recovers_saha_boltzmann(temperature, density):
    rates, saha = thermal_rates(temperature)
    result = solve_statistical_equilibrium(density, rates)
    x = 2 / (1 + np.sqrt(1 + 4 * density * saha.sum()))
    expected = density**2 * x**2 * saha
    assert result.converged
    assert np.all(np.asarray(result.bound_cm3) > 0)
    np.testing.assert_allclose(result.ionized_fraction, x, rtol=2e-10)
    np.testing.assert_allclose(result.bound_cm3, expected, rtol=2e-10)
    assert result.residual_norm < 1e-10
    assert result.rate_residual_norm < 1e-10


def test_jit_vmap_keeps_each_cells_density_and_rates():
    density = jnp.asarray([1e8, 1e10, 1e12, 1e14])
    gamma = jnp.asarray([0.01, 0.1, 1.0, 10.0])
    batch = jax.jit(jax.vmap(lambda n, g: solve_statistical_equilibrium(n, one_level(g))))
    result = batch(density, gamma)
    expected = 2 / (1 + np.sqrt(1 + 4e-12 * np.asarray(density / gamma)))
    assert np.all(result.converged)
    np.testing.assert_allclose(result.ionized_fraction, expected, rtol=2e-14)
    assert result.bound_cm3.shape == (4, 1)


def test_implicit_jvp_vjp_match_independent_finite_differences():
    parameters = jnp.log(jnp.asarray([4e11, 0.2, 1e-12, 3e-13, 1e-25]))

    def outputs(values):
        density, gamma, alpha, ionization, three_body = jnp.exp(values)
        result = solve_statistical_equilibrium(density, one_level(gamma, alpha, ionization, three_body))
        return jnp.log(jnp.concatenate((result.bound_cm3, result.proton_cm3[None])))

    jacobian = np.asarray(jax.jacfwd(outputs)(parameters))
    step = 1e-4
    finite_difference = np.column_stack([
        (np.asarray(outputs(parameters.at[i].add(step)))
         - np.asarray(outputs(parameters.at[i].add(-step)))) / (2 * step)
        for i in range(len(parameters))
    ])
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2e-6, atol=2e-9)
    direction = jnp.asarray([0.3, -0.2, 0.1, -0.4, 0.25])
    _, tangent = jax.jvp(outputs, (parameters,), (direction,))
    np.testing.assert_allclose(tangent, jacobian @ direction, rtol=1e-12)
    weights = jnp.asarray([0.2, 0.8])
    gradient = jax.grad(lambda values: jnp.dot(weights, outputs(values)))(parameters)
    np.testing.assert_allclose(gradient, weights @ finite_difference, rtol=2e-6, atol=2e-9)


def test_implicit_multilevel_gradients_include_electron_feedback_and_conserve_nuclei():
    rates, _ = thermal_rates()

    def fractions(log_density):
        density = jnp.exp(log_density)
        result = solve_statistical_equilibrium(density, rates)
        return jnp.concatenate((result.bound_cm3 / density, result.ionized_fraction[None]))

    point, step = jnp.log(1e12), 1e-4
    derivative = jax.jacfwd(fractions)(point)
    reference = (fractions(point + step) - fractions(point - step)) / (2 * step)
    np.testing.assert_allclose(derivative, reference, rtol=3e-6, atol=1e-11)
    np.testing.assert_allclose(jnp.sum(derivative), 0, atol=2e-13)


def test_bound_transition_and_continuum_gradients_follow_the_same_residual():
    def log_populations(parameters):
        scale = jnp.exp(parameters)
        rates = RateCoefficients(
            jnp.asarray([[0.0, 2.0], [1e8, 0.0]]) * scale[0],
            jnp.asarray([[0.0, 1e-10], [1e-8, 0.0]]) * scale[1],
            jnp.asarray([0.2, 300.0]) * scale[2],
            jnp.asarray([2e-13, 1e-13]) * scale[3],
            jnp.asarray([1e-16, 2e-10]) * scale[4],
            jnp.asarray([1e-31, 2e-27]) * scale[5],
        )
        result = solve_statistical_equilibrium(1e11, rates)
        return jnp.log(jnp.concatenate((result.bound_cm3, result.proton_cm3[None])))

    point = jnp.zeros(6)
    direction = jnp.asarray([0.2, -0.1, 0.4, -0.3, 0.15, -0.25])
    _, tangent = jax.jvp(log_populations, (point,), (direction,))
    step = 1e-4
    reference = (log_populations(point + step * direction)
                 - log_populations(point - step * direction)) / (2 * step)
    assert np.isfinite(tangent).all()
    np.testing.assert_allclose(tangent, reference, rtol=2e-6, atol=2e-9)


def test_six_level_rate_model_closes_over_sampled_physical_inputs():
    from escape.hydrogen_rates import blackbody_hydrogen_rates

    cases = jnp.asarray(list(itertools.product(
        [6000.0, 9000.0, 14000.0], [1e8, 1e12, 1e16], [1e-4, 0.025, 1.0],
    )))
    batch = jax.jit(jax.vmap(lambda temperature, density, dilution:
        solve_statistical_equilibrium(
            density, blackbody_hydrogen_rates(temperature, 10000.0, dilution),
        )))
    result = batch(*cases.T)
    assert result.bound_cm3.shape == (27, 6)
    assert np.all(result.valid) and np.all(result.converged)
    assert np.all(np.asarray(result.bound_cm3) >= 0)
    assert np.max(result.conservation_error) < 1e-10
    assert np.max(result.rate_residual_norm) < 1e-10


def test_six_level_collision_limit_and_diluted_non_lte_populations():
    from escape.hydrogen_rates import blackbody_hydrogen_rates, saha_factors

    temperature, density = 10000.0, 1e12
    rates = blackbody_hydrogen_rates(temperature, temperature, 1.0)
    saha = np.asarray(saha_factors(temperature))
    x = 2 / (1 + np.sqrt(1 + 4 * density * saha.sum()))
    expected = density**2 * x**2 * saha
    scale = 1e8
    collision_dominated = rates._replace(
        electron_bound_bound_cm3_s=rates.electron_bound_bound_cm3_s * scale,
        electron_ionization_cm3_s=rates.electron_ionization_cm3_s * scale,
        three_body_recombination_cm6_s=rates.three_body_recombination_cm6_s * scale,
    )
    result = solve_statistical_equilibrium(density, collision_dominated)
    assert result.converged
    np.testing.assert_allclose(result.bound_cm3, expected, rtol=1e-8)
    np.testing.assert_allclose(result.ionized_fraction, x, rtol=1e-8)

    diluted = solve_statistical_equilibrium(
        density, blackbody_hydrogen_rates(temperature, temperature, 0.025),
    )
    assert diluted.converged
    departure = np.asarray(diluted.bound_cm3) / (float(diluted.electron_cm3)**2 * saha)
    assert np.max(np.abs(departure - 1)) > 0.01


@pytest.mark.parametrize("density,rates", [
    (0.0, one_level()), (-1.0, one_level()), (np.inf, one_level()),
    (1e12, one_level(gamma=-1.0)), (1e12, one_level(alpha=-1.0)),
    (1e12, one_level(gamma=np.nan)),
    (1e12, one_level(gamma=0.0, ionization=1e-12)),
    (1e12, one_level(gamma=0.0)),
    (1.0, one_level(gamma=1e-100, alpha=1.0)),
])
def test_invalid_or_unbracketed_models_are_explicitly_unsolved(density, rates):
    result = solve_statistical_equilibrium(density, rates)
    assert not result.valid and not result.converged
    assert np.isnan(result.bound_cm3).all()
    assert np.isnan(result.electron_cm3)


def test_finite_iteration_failure_is_not_returned_as_a_population():
    result = solve_statistical_equilibrium(1e12, one_level(), iterations=1)
    assert result.valid and not result.converged
    assert result.residual_norm > 1e-10
    assert np.isnan(result.bound_cm3).all()


def test_diagonal_bound_bound_rates_are_not_silently_removed():
    rates = one_level()._replace(radiative_bound_bound_s=jnp.ones((1, 1)))
    result = solve_statistical_equilibrium(1e12, rates)
    assert not result.valid and not result.converged


def test_shapes_and_iteration_count_have_actionable_errors():
    with pytest.raises(ValueError, match="scalar"):
        solve_statistical_equilibrium(jnp.ones(2), one_level())
    with pytest.raises(ValueError, match="two .* arrays"):
        solve_statistical_equilibrium(1e12, one_level()._replace(photoionization_s=jnp.ones(2)))
    with pytest.raises(ValueError, match="positive"):
        solve_statistical_equilibrium(1e12, one_level(), iterations=0)


@pytest.mark.parametrize("ionization,three_body", [(0.0, 0.0), (3e-13, 1e-25)])
def test_relaxation_time_includes_charge_feedback_in_every_kinetic_term(ionization, three_body):
    density, gamma, alpha = 4e11, 0.2, 1e-12
    rates = one_level(gamma, alpha, ionization, three_body)
    result = solve_statistical_equilibrium(density, rates)
    electron, bound = float(result.electron_cm3), float(result.bound_cm3[0])
    decay_rate = gamma + 2 * alpha * electron + 3 * three_body * electron**2
    decay_rate += ionization * (electron - bound)
    assert decay_rate > 0
    timescale = relaxation_time_s(density, result.bound_cm3, rates)
    np.testing.assert_allclose(timescale, 1 / decay_rate, rtol=3e-14)


def test_relaxation_time_flags_unstable_and_nondecaying_modes():
    density = 1e12
    # With no photons or seed electrons this is an absorbing neutral state,
    # but collisional ionization makes its linearized electron perturbation grow.
    unstable = one_level(gamma=0.0, ionization=1e-12)
    assert np.isinf(relaxation_time_s(density, jnp.asarray([density]), unstable))
    no_reactions = one_level(gamma=0.0, alpha=0.0)
    assert np.isinf(relaxation_time_s(density, jnp.asarray([density]), no_reactions))
    assert np.isnan(relaxation_time_s(density, jnp.asarray([2 * density]), one_level()))


def test_six_level_relaxation_diagnostic_is_finite_without_redundant_conservation_mode():
    from escape.hydrogen_rates import blackbody_hydrogen_rates

    density = 1e12
    rates = blackbody_hydrogen_rates(9000.0, 10000.0, 0.025)
    result = solve_statistical_equilibrium(density, rates)
    timescale = relaxation_time_s(density, result.bound_cm3, rates)
    assert result.converged
    assert np.isfinite(timescale) and timescale > 0
