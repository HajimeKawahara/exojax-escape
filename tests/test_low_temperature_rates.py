"""Explicit low-T extrapolation, retained defaults and external Cloudy rates."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_atom import (
    atom_collision_rates, atom_rate_coefficients, make_hydrogen_atom,
    solve_atom_populations,
)
from escape.hydrogen_rates import (
    BOLTZMANN_CGS, EXCITATION_ENERGY_ERG, MIN_TEMPERATURE_K,
    STATISTICAL_WEIGHTS, collision_rates, minimum_rate_temperature,
    planck_occupation,
)

jax.config.update("jax_enable_x64", True)

# Actual electron de-excitation readouts from Cloudy C23.01, 2026-11-23.
# Pure H, nH=1e12 cm^-3, BB10000 K W=.025, nmax15 (resolved3, collapsed12),
# fixed T, thickness1e-4 cm. MCEG_BB rate_coef_ul()[ipELECTRON]; no proton mixing.
# Reproduction receipts and raw hashes are in the PR's low-temperature evidence.
# Cloudy uses Anderson nl values and rounded5802 K; our retained Lehmann
# fine-level sums and exact0.5eV/kB differ by up to0.279%, so agreement is0.3%.
_PAIRS = np.array([(1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 0),
                   (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)])
_CLOUDY_DOWN = np.array([
    [1.2238190905122792e-8, 6.7310055307245875e-9, 3.0642548498605038e-9,
     6.49565521371187e-8, 9.602272399433787e-8, 1.7572787669450918e-9,
     3.859737418565601e-8, 1.1390932286303723e-7, 5.846089723426928e-10,
     1.9675246321071427e-8, 1.2426471585058607e-7],
    [1.2980561514552846e-8, 7.1393086997773025e-9, 3.250133051579951e-9,
     6.889683135113955e-8, 1.0184747623043222e-7, 1.8638754983868466e-9,
     4.093869776511383e-8, 1.2081908096206462e-7, 6.200714386395134e-10,
     2.0868750283398185e-8, 1.3180263636058953e-7],
    [1.4739273801467334e-8, 8.106600724033797e-9, 3.6904874711751745e-9,
     7.823152969876901e-8, 1.1564661406282539e-7, 2.116408603924924e-9,
     4.648540397056422e-8, 1.3718863556277938e-7, 7.040837912697384e-10,
     2.369621654452203e-8, 1.4966032324537082e-7],
])


@pytest.mark.parametrize("index, temperature", enumerate([4000., 4500., 5802.]))
def test_explicit_extrapolation_matches_cloudy_resolved_electron_rates(index, temperature):
    rates = collision_rates(temperature, low_temperature_extrapolation="cloudy")[0]
    np.testing.assert_allclose(rates[_PAIRS[:, 0], _PAIRS[:, 1]],
                               _CLOUDY_DOWN[index], rtol=.003)


@pytest.mark.parametrize("temperature", [MIN_TEMPERATURE_K, 6000., 10000., 14000.])
def test_all_default_rates_remain_unchanged_in_the_tabulated_domain(temperature):
    default = collision_rates(temperature)
    extended = collision_rates(temperature, low_temperature_extrapolation="cloudy")
    for before, after in zip(default, extended):
        np.testing.assert_array_equal(before, after)


def test_extension_is_opt_in_and_keeps_the_ionization_fit_boundaries():
    assert np.isnan(collision_rates(4500.)[0]).all()
    for temperature in (3999., 14001.):
        assert np.isnan(collision_rates(temperature, low_temperature_extrapolation="cloudy")[0]).all()
    assert minimum_rate_temperature() == MIN_TEMPERATURE_K
    assert minimum_rate_temperature("cloudy") == 4000.
    with pytest.raises(ValueError, match="reject.*cloudy"):
        collision_rates(4500., low_temperature_extrapolation="constant")


def test_reverse_balance_and_low_temperature_derivative():
    temperature = 4500.
    rates = collision_rates(temperature, low_temperature_extrapolation="cloudy")[0]
    upper, lower = _PAIRS.T
    expected = (STATISTICAL_WEIGHTS[upper] / STATISTICAL_WEIGHTS[lower]
                * np.exp(-(EXCITATION_ENERGY_ERG[upper] - EXCITATION_ENERGY_ERG[lower])
                         / (BOLTZMANN_CGS * temperature)))
    np.testing.assert_allclose(rates[lower, upper] / rates[upper, lower], expected, rtol=3e-15)
    derivative = jax.grad(lambda log_temperature: jnp.log(collision_rates(
        jnp.exp(log_temperature), low_temperature_extrapolation="cloudy",
    )[0][2, 0]))(jnp.log(temperature))
    # Upsilon proportional to T makes the de-excitation coefficient proportional to sqrt(T).
    np.testing.assert_allclose(derivative, .5, rtol=3e-14)
    before = collision_rates(MIN_TEMPERATURE_K - 1e-4, low_temperature_extrapolation="cloudy")[0]
    after = collision_rates(MIN_TEMPERATURE_K + 1e-4, low_temperature_extrapolation="cloudy")[0]
    np.testing.assert_allclose(before[upper, lower], after[upper, lower], rtol=2e-8)


def test_extended_atom_population_root_and_gradient_accept_explicit_low_temperature():
    atom = make_hydrogen_atom(5)
    frequency = jnp.geomspace(1e13, 2.4e16, 128)
    weight = jnp.gradient(frequency)
    lines = planck_occupation(atom.line_frequency_hz, 10000., .025)
    continuum = planck_occupation(frequency, 10000., .025)
    rates = atom_rate_coefficients(atom, 4500., lines, continuum, frequency, weight,
                                   electron_cm3=1e11, low_temperature_extrapolation="cloudy")
    assert all(np.isfinite(value).all() for value in rates)
    assert np.isnan(atom_collision_rates(atom, 4500.)[0]).all()
    def solve(temperature):
        return solve_atom_populations(1e12, atom, temperature, lines, continuum, frequency,
                                       weight, low_temperature_extrapolation="cloudy")
    answer = solve(4500.)
    assert bool(answer.converged)
    np.testing.assert_allclose(answer.bound_cm3.sum() + answer.proton_cm3, 1e12, rtol=1e-9)
    evaluate = lambda log_temperature: jnp.log(solve(jnp.exp(log_temperature)).bound_cm3[3:6].sum())
    derivative = jax.grad(evaluate)(jnp.log(4500.))
    step = 1e-5
    finite = (evaluate(jnp.log(4500.) + step) - evaluate(jnp.log(4500.) - step)) / (2 * step)
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, finite, rtol=2e-5, atol=1e-7)
