"""Independent impact-parameter quadrature, detailed balance and mixing AD."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from escape.hydrogen_atom import make_hydrogen_atom
from escape.hydrogen_mixing import (
    BOHR_RADIUS_CM, BOHR_TIME_S, HYDROGEN_PROTON_REDUCED_MASS_G,
    RYDBERG_TEMPERATURE_K, mixing_bound_bound_cm3_s, proton_l_mixing_psm20,
)
from escape.hydrogen_rates import BOLTZMANN_CGS, ELECTRON_MASS_CGS

jax.config.update("jax_enable_x64", True)


def independent_cross_section_integral(temperature, density, ell, lifetime, n=3):
    """Integrate the published piecewise cross section, not the closed formula."""
    speed = np.sqrt(temperature / RYDBERG_TEMPERATURE_K
                    * ELECTRON_MASS_CGS / HYDROGEN_PROTON_REDUCED_MASS_G)
    strength = 4.5 * n**2 * ell * (n**2 - ell**2)
    match2 = 8 * strength / (3 * speed**2 * (2 * ell + 1))
    debye2 = (temperature / RYDBERG_TEMPERATURE_K
              / (8 * np.pi * density * BOHR_RADIUS_CM**3))
    life2 = (.72 * lifetime * speed / BOHR_TIME_S)**2
    um, uc, uml = match2 / debye2, debye2 / life2, np.sqrt(match2 / life2)

    def integrand(energy_over_kt):
        ratio2 = min(energy_over_kt / um, (energy_over_kt / uml)**2)
        kernel = 2 / 3 + np.log(ratio2) if ratio2 >= 1 else 2 / 3 * ratio2**1.5
        return np.exp(-energy_over_kt) * kernel

    edges = np.unique([0., *[u for u in (um, uc, uml) if 0 < u < 100], 100.])
    integral = sum(quad(integrand, a, b, epsabs=1e-11, epsrel=1e-11)[0]
                   for a, b in zip(edges[:-1], edges[1:]))
    units = 2 * BOHR_RADIUS_CM**3 * np.sqrt(np.pi) / (speed * BOHR_TIME_S)
    return units * 2 * strength / (3 * (2 * ell + 1)) * integral


@pytest.mark.parametrize("temperature,density", [
    (6000., 1e8), (9000., 1e10), (14000., 1e12), (9000., 1e15),
])
def test_combined_cutoffs_match_independent_maxwell_cross_section(temperature, density):
    for n, ell, lifetime in ((2, 1, 1 / 6.26489344e8),
                             (3, 1, 1 / (1.6724736e8 + 2.2448808e7)),
                             (3, 2, 1 / 6.4650552e7)):
        expected = independent_cross_section_integral(temperature, density, ell, lifetime, n)
        actual = proton_l_mixing_psm20(temperature, density, n, ell, ell - 1, lifetime)
        np.testing.assert_allclose(actual, expected, rtol=2e-10)


def test_cloudy_readout_anchor_with_declared_cutoff_and_mass_difference():
    # Actual C23.01 operator readout, T=9000K, ne=8.215510376031405e11.
    # Continuous combined cutoffs versus Cloudy's pure-Debye shortcut differ;
    # the physical H-1 reduced mass also differs from its standard atomic weight.
    density = 8.215510376031405e11
    for ell, lifetime, cloudy in (
        (1, 1 / (1.6724736e8 + 2.2448808e7), 1.6646620351821184e-3),
        (2, 1 / 6.4650552e7, 1.2913137907162309e-3),
    ):
        actual = proton_l_mixing_psm20(9000., density, 3, ell, ell - 1, lifetime)
        assert abs(float(actual) / cloudy - 1) < .002


def test_mixing_matrix_preserves_pairwise_detailed_balance_and_ground_state():
    atom, temperature = make_hydrogen_atom(5), 9000.
    matrix = mixing_bound_bound_cm3_s(atom, temperature, 1e12)
    assert np.count_nonzero(matrix) == 6
    assert np.all(np.asarray(matrix)[0] == 0)
    weights = atom.statistical_weights * jnp.exp(-atom.excitation_energy_erg / (BOLTZMANN_CGS * temperature))
    flux = weights[:, None] * matrix
    np.testing.assert_allclose(flux, flux.T, rtol=2e-14, atol=1e-40)
    assert float(matrix[4, 3]) > 1e-3


def test_n2_rate_matches_cloudy_dilute_readout_with_declared_mass_difference():
    actual = proton_l_mixing_psm20(9000., 9974420448., 2, 1, 0, 1 / 6.26489344e8)
    assert float(actual) == pytest.approx(3.8447085535153747e-4, rel=1e-4)


def test_density_and_temperature_derivatives_across_the_cloudy_cutoff_boundary():
    lifetime, temperature = 1 / (1.6724736e8 + 2.2448808e7), 9000.
    speed = np.sqrt(temperature / RYDBERG_TEMPERATURE_K
                    * ELECTRON_MASS_CGS / HYDROGEN_PROTON_REDUCED_MASS_G)
    critical_density = (temperature / RYDBERG_TEMPERATURE_K
                        / (8 * np.pi * BOHR_RADIUS_CM**3 * (.72 * lifetime * speed / BOHR_TIME_S)**2))
    def observable(log_parameters):
        temp = temperature * jnp.exp(log_parameters[0])
        density = critical_density * jnp.exp(log_parameters[1])
        return jnp.log(proton_l_mixing_psm20(temp, density, 3, 1, 0, lifetime))
    point = jnp.zeros(2)
    gradient = jax.jit(jax.grad(observable))(point)
    step = 1e-4
    finite = jnp.array([(observable(point + step * unit) - observable(point - step * unit)) / (2 * step)
                        for unit in jnp.eye(2)])
    np.testing.assert_allclose(gradient, finite, rtol=2e-7, atol=2e-10)
    left, right = observable(jnp.array([0., -1e-7])), observable(jnp.array([0., 1e-7]))
    assert abs(float(right - left)) < 1e-7
    assert float(gradient[1]) < 0


def test_weak_density_limit_is_finite_through_the_charge_root_bracket():
    # Charge closure samples ne far below its physical root while bracketing.
    lifetime = 1 / (1.6724736e8 + 2.2448808e7)
    observable = lambda log_ne: jnp.log(proton_l_mixing_psm20(
        9000., jnp.exp(log_ne), 3, 1, 0, lifetime,
    ))
    values = jax.vmap(observable)(jnp.log(jnp.array([1e-30, 1e-23, 1.])))
    slopes = jax.vmap(jax.grad(observable))(jnp.log(jnp.array([1e-30, 1e-23, 1.])))
    np.testing.assert_allclose(values, values[0], rtol=1e-14)
    np.testing.assert_allclose(slopes, 0., atol=1e-14)
    assert np.isfinite(values).all()


@pytest.mark.parametrize("temperature,density,lifetime", [(0., 1e12, 1e-8), (9000., -1., 1e-8), (9000., 1e12, 0.)])
def test_invalid_mixing_inputs_are_visible(temperature, density, lifetime):
    assert np.isnan(proton_l_mixing_psm20(temperature, density, 3, 1, 0, lifetime))


def test_canonical_population_gradients_include_screening_and_charge_feedback():
    """Check the NUTS-facing solve and distinguish omitted screening feedback."""
    from escape.hydrogen_atom import atom_rate_coefficients, solve_atom_populations
    from escape.hydrogen_nlte import solve_statistical_equilibrium
    from escape.hydrogen_rates import EV_CGS, PLANCK_CGS, planck_occupation

    atom = make_hydrogen_atom(5)
    # A fixed threshold-split continuum isolates population/root derivatives.
    edges = np.unique(np.r_[np.asarray(atom.ionization_energy_erg) / PLANCK_CGS,
                            100 * EV_CGS / PLANCK_CGS])
    nodes, weights = np.polynomial.legendre.leggauss(24)
    widths = np.diff(np.log(edges))
    frequency = edges[:-1, None] * np.exp(widths[:, None] * (nodes + 1) / 2)
    quadrature = jnp.asarray((frequency * widths[:, None] * weights / 2).ravel())
    frequency = jnp.asarray(frequency.ravel())
    line = planck_occupation(atom.line_frequency_hz, 10000., .025)
    continuum = planck_occupation(frequency, 10000., .025)

    def solve(log_parameters):
        temperature, density = 9000 * jnp.exp(log_parameters[0]), 1e12 * jnp.exp(log_parameters[1])
        return solve_atom_populations(density, atom, temperature, line, continuum,
                                      frequency, quadrature)

    def observables(solution):
        return jnp.log(jnp.r_[solution.electron_cm3, solution.bound_cm3[3:6],
                             jnp.sum(solution.bound_cm3[3:6])])

    point, step = jnp.zeros(2), 2e-4
    reference = solve(point)
    assert reference.converged
    observable = lambda parameters: observables(solve(parameters))
    automatic = jax.jit(jax.jacrev(observable))(point)
    compiled = jax.jit(observable)
    finite = jnp.stack([(compiled(point + step * unit) - compiled(point - step * unit)) / (2 * step)
                        for unit in jnp.eye(2)], axis=1)
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-8)

    def frozen_screening(parameters):
        temperature = 9000 * jnp.exp(parameters[0])
        rates = atom_rate_coefficients(atom, temperature, line, continuum, frequency, quadrature,
                                       electron_cm3=reference.electron_cm3)
        return observables(solve_statistical_equilibrium(1e12 * jnp.exp(parameters[1]), rates))

    # At the reference point, frozen screening has identical populations but
    # different derivatives. This control rules out a base-rate-only AD test.
    np.testing.assert_allclose(frozen_screening(point), observable(point), rtol=1e-12)
    frozen_gradient = jax.jit(jax.jacrev(frozen_screening))(point)
    assert np.max(np.abs(np.asarray(automatic - frozen_gradient))) > 1e-5
