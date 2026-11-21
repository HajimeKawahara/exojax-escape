"""Atomic-data anchors, thermodynamic identities and differentiable rates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_rates import (
    BOLTZMANN_CGS,
    EV_CGS,
    EXCITATION_ENERGY_ERG,
    IONIZATION_ENERGY_ERG,
    LINE_EINSTEIN_A_S,
    LINE_FREQUENCY_HZ,
    LINE_LOWER,
    LINE_UPPER,
    MIN_TEMPERATURE_K,
    PLANCK_CGS,
    STATISTICAL_WEIGHTS,
    blackbody_hydrogen_rates,
    bound_bound_radiative_rates,
    bound_free_rates,
    collision_rates,
    continuum_frequency_grid,
    hydrogen_rate_coefficients,
    photoionization_cross_section,
    planck_occupation,
    saha_factors,
)

jax.config.update("jax_enable_x64", True)


def test_table_anchors_and_fine_structure_aggregation():
    # Independent published entries: 1s--2s Upsilon=.296 at kT=1 eV;
    # n=2 -> n=3 Upsilon sums to 35.52 in Lehmann et al. Table 7.
    temperature = EV_CGS / BOLTZMANN_CGS
    q, _, _ = collision_rates(temperature)
    assert float(q[1, 0]) == pytest.approx(8.629e-6 * .296 / (2 * np.sqrt(temperature)))
    upper = np.array([3, 4, 5])
    recovered_strength = np.sum(
        np.asarray(q)[upper, 1:3] * STATISTICAL_WEIGHTS[upper, None]
    ) * np.sqrt(temperature) / 8.629e-6
    assert recovered_strength == pytest.approx(35.52)
    np.testing.assert_allclose(LINE_EINSTEIN_A_S[2:], [6.3143e6, 2.244833333333e7, 6.46514e7])


@pytest.mark.parametrize("temperature", [6000., 9000., 13500.])
def test_electron_and_proton_collision_detailed_balance(temperature):
    bound_bound, ionization, three_body = collision_rates(temperature)
    boltzmann = STATISTICAL_WEIGHTS * np.exp(
        -EXCITATION_ENERGY_ERG / (BOLTZMANN_CGS * temperature)
    )
    flux = boltzmann[:, None] * np.asarray(bound_bound)
    np.testing.assert_allclose(flux, flux.T, rtol=2e-13, atol=1e-30)
    np.testing.assert_allclose(three_body / ionization, saha_factors(temperature), rtol=2e-13)
    assert np.all(np.diag(bound_bound) == 0)


def test_bound_free_saha_balance_and_one_photon_einstein_balance():
    temperature = 8500.
    frequency, weight = continuum_frequency_grid()
    gamma, alpha = bound_free_rates(
        temperature, planck_occupation(frequency, temperature), frequency, weight,
    )
    np.testing.assert_allclose(alpha / gamma, saha_factors(temperature), rtol=3e-13)
    rates = bound_bound_radiative_rates(planck_occupation(LINE_FREQUENCY_HZ, temperature))
    expected = (STATISTICAL_WEIGHTS[LINE_UPPER] / STATISTICAL_WEIGHTS[LINE_LOWER]
                * np.exp(-PLANCK_CGS * LINE_FREQUENCY_HZ / (BOLTZMANN_CGS * temperature)))
    np.testing.assert_allclose(
        rates[LINE_LOWER, LINE_UPPER] / rates[LINE_UPPER, LINE_LOWER], expected,
        rtol=3e-13,
    )
    # Spontaneous-only two-photon closure is deliberately not an Einstein line.
    assert rates[1, 0] == 8.2
    assert rates[0, 1] == 0


def test_continuum_threshold_and_quadrature_convergence():
    threshold = IONIZATION_ENERGY_ERG[:, None] / PLANCK_CGS
    sigma = photoionization_cross_section(jnp.concatenate((.999 * threshold, threshold), axis=1))
    np.testing.assert_array_equal(sigma[:, 0], np.zeros(6))
    assert float(sigma[0, 1]) == pytest.approx(7.9e-18, rel=.02)
    coarse = blackbody_hydrogen_rates(6000., 20000., .03, quadrature_points=64)
    fine = blackbody_hydrogen_rates(6000., 20000., .03, quadrature_points=128)
    extended = blackbody_hydrogen_rates(
        6000., 20000., .03, quadrature_points=256, maximum_energy_ev=200.,
    )
    for field in ("photoionization_s", "radiative_recombination_cm3_s"):
        np.testing.assert_allclose(getattr(coarse, field), getattr(fine, field), rtol=1e-10)
        np.testing.assert_allclose(getattr(fine, field), getattr(extended, field), rtol=1e-10)


def test_radiation_scaling_and_zero_external_field():
    dark = blackbody_hydrogen_rates(8000., 10000., 0.)
    low = blackbody_hydrogen_rates(8000., 10000., .01)
    high = blackbody_hydrogen_rates(8000., 10000., .02)
    np.testing.assert_array_equal(dark.photoionization_s, np.zeros(6))
    assert np.all(np.asarray(dark.radiative_recombination_cm3_s) > 0)
    np.testing.assert_allclose(high.photoionization_s, 2 * low.photoionization_s)
    # Stimulated recombination adds linearly to spontaneous recombination.
    np.testing.assert_allclose(
        high.radiative_recombination_cm3_s,
        2 * low.radiative_recombination_cm3_s - dark.radiative_recombination_cm3_s,
        rtol=2e-13,
    )


def test_temperature_domain_and_radiation_validation():
    for temperature in (MIN_TEMPERATURE_K - 1, 14001., np.nan):
        rates = blackbody_hydrogen_rates(temperature, 10000., .01)
        assert all(np.all(np.isnan(item)) for item in rates)
    for radiation_temperature, dilution in ((10000., -.01), (-1., 0.), (0., .01)):
        rates = blackbody_hydrogen_rates(8000., radiation_temperature, dilution)
        assert all(np.all(np.isnan(item)) for item in rates)
    frequency, weight = continuum_frequency_grid(16)
    with pytest.raises(ValueError, match="shape"):
        hydrogen_rate_coefficients(8000., jnp.zeros(6), frequency * 0, frequency, weight)


def test_jit_vmap_and_temperature_radiation_gradients():
    frequency, weight = continuum_frequency_grid(64)

    def log_rate(parameters):
        temperature, radiation_temperature, dilution = jnp.exp(parameters)
        rates = hydrogen_rate_coefficients(
            temperature,
            planck_occupation(LINE_FREQUENCY_HZ, radiation_temperature, dilution),
            planck_occupation(frequency, radiation_temperature, dilution),
            frequency, weight,
        )
        return (jnp.log(rates.radiative_recombination_cm3_s[1])
                + jnp.log(rates.photoionization_s[0])
                + jnp.log(rates.electron_bound_bound_cm3_s[0, 1]))

    point = jnp.log(jnp.array([8500., 10000., .03]))
    gradient = np.asarray(jax.jit(jax.grad(log_rate))(point))
    step = 1e-4
    numeric = np.array([
        (log_rate(point + step * jnp.eye(3)[i])
         - log_rate(point - step * jnp.eye(3)[i])) / (2 * step)
        for i in range(3)
    ])
    np.testing.assert_allclose(gradient, numeric, rtol=1e-7)
    results = jax.jit(jax.vmap(log_rate))(jnp.stack((point, point + .01)))
    assert np.all(np.isfinite(results))
