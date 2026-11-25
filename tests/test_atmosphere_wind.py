"""Composition-pressure coupling, mass conservation and audited flow-energy signs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.utils.constants import RJ, kB

from escape.atmosphere_wind import (
    _effective_temperature, thermal_flow_terms, wind_from_profiles,
)
from escape.parker import PROTON_MASS_CGS, parker_velocity

jax.config.update("jax_enable_x64", True)
BASE = 1.9 * RJ
RADII = jnp.geomspace(BASE, 3 * BASE, 24)
MASS = 2.17


def wind(temperature, mu, mass_loss=1e13, radii=RADII, **kwargs):
    return wind_from_profiles(
        radii, temperature, mu, mass_loss, MASS,
        base_radius_cm=BASE, outer_radius_cm=32 * BASE, **kwargs,
    )


def test_constant_profiles_recover_analytic_wind_and_mass_conservation():
    result, info = wind(jnp.full_like(RADII, 10000.), jnp.full_like(RADII, .7),
                        return_info=True)
    assert bool(info["valid"]) and bool(info["converged"])
    np.testing.assert_allclose(result.velocity_kms, parker_velocity(RADII, 10000., MASS, .7), rtol=2e-9)
    np.testing.assert_allclose(4 * np.pi * RADII**2 * result.density_g_cm3 * result.velocity_kms * 1e5,
                               1e13, rtol=2e-15)


def test_variable_mu_enters_the_pressure_gradient_as_temperature_over_mu():
    mu = .85 - .2 * (RADII / BASE - 1) / 2
    temperature = 9000 * (1 + .15 * jnp.sin(jnp.log(RADII / BASE)))
    result = wind(temperature, mu)
    effective = wind(temperature / mu, jnp.ones_like(mu))
    assert np.isfinite(result.velocity_kms).all()
    np.testing.assert_allclose(result.velocity_kms, effective.velocity_kms, rtol=2e-13)
    # If temperature tracks the particle mass, the gas pressure/rho is constant.
    compensated = wind(14000 * mu, mu)
    np.testing.assert_allclose(compensated.velocity_kms, parker_velocity(RADII, 14000., MASS, 1.), rtol=2e-9)


def test_profile_interpolation_has_zero_endpoint_slopes_and_constant_extension():
    values = jnp.log(jnp.array([8000., 12000., 9000.]))
    radius = jnp.array([1., 2., 3.]) * BASE
    parameters = jnp.log(radius), values
    evaluate = lambda q: _effective_temperature(q, parameters)
    np.testing.assert_allclose(jax.vmap(evaluate)(radius), jnp.exp(values), rtol=2e-15)
    np.testing.assert_allclose(jax.vmap(jax.grad(evaluate))(radius[jnp.array([0, 2])]), 0., atol=1e-20)
    np.testing.assert_allclose(jax.vmap(evaluate)(jnp.array([.5, 4.]) * BASE), [8000., 9000.], rtol=2e-15)


def test_implicit_gradient_through_temperature_mu_and_density_matches_difference():
    def evaluate(parameters):
        temperature = jnp.full_like(RADII, 10000.) * jnp.exp(parameters[0])
        mu = (.8 - .1 * jnp.log(RADII / BASE)) * jnp.exp(parameters[1])
        result = wind(temperature, mu, mass_loss=1e13 * jnp.exp(parameters[2]))
        return jnp.log(result.velocity_kms[10]) + .3 * jnp.log(result.density_g_cm3[15])
    parameters = jnp.zeros(3)
    derivative = jax.grad(evaluate)(parameters)
    step = 1e-5
    finite = jnp.array([(evaluate(parameters + step * axis) - evaluate(parameters - step * axis)) / (2 * step)
                        for axis in jnp.eye(3)])
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, finite, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(derivative[2], .3, rtol=2e-12)


def test_flow_terms_match_nonuniform_polynomial_derivatives_and_expansion_sign():
    radius = jnp.array([1., 1.3, 2., 3., 5.]) * BASE
    density = 1e-12 * (6 - radius / BASE)
    velocity = jnp.full_like(radius, 1e6)
    mu = .9 - .03 * radius / BASE
    effective_temperature = 8000. + 300. * (radius / BASE)**2
    terms = thermal_flow_terms(radius, density, effective_temperature * mu, mu, velocity)
    factor = kB / PROTON_MASS_CGS
    expected_advection = -1.5 * density * velocity * factor * 600 * radius / BASE**2
    expected_expansion = factor * effective_temperature * velocity * (-1e-12 / BASE)
    np.testing.assert_allclose(terms.advection[1:-1], expected_advection[1:-1], rtol=2e-14)
    np.testing.assert_allclose(terms.expansion, expected_expansion, rtol=2e-14)
    assert np.all(np.asarray(terms.expansion) < 0)
    static = thermal_flow_terms(radius, density, effective_temperature * mu, mu,
                                jnp.zeros_like(radius))
    np.testing.assert_array_equal(static.advection, 0.)
    np.testing.assert_array_equal(static.expansion, 0.)


def test_invalid_profiles_are_reported_and_shape_errors_raise():
    temperature = jnp.full_like(RADII, 10000.)
    mu = jnp.full_like(RADII, .7)
    result, info = wind(temperature, mu, mass_loss=-1., return_info=True)
    assert not bool(info["valid"])
    assert np.isnan(result.velocity_kms).all() and np.isnan(result.density_g_cm3).all()
    assert np.isnan(thermal_flow_terms(RADII, -jnp.ones_like(RADII), temperature, mu,
                                      jnp.ones_like(RADII)).advection).all()
    with pytest.raises(ValueError, match="matching one-dimensional"):
        wind(temperature[:-1], mu)
