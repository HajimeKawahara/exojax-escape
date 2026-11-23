"""Regular wind derivatives and energy terms on a coarse atmospheric grid."""
from __future__ import annotations

from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.utils.constants import G, MJ, RJ, kB

from escape.atmosphere_wind import (
    _effective_temperature, thermal_flow_terms, wind_from_profiles,
)
from escape.nonisothermal import nonisothermal_parker_velocity
from escape.parker import PROTON_MASS_CGS
from escape.solar import SolarAtmosphereModel

jax.config.update("jax_enable_x64", True)
BASE, MASS = 1.9*RJ, 2.17


def constant_temperature(radius, effective_temperature):
    return effective_temperature+jnp.zeros_like(radius)


def test_isothermal_spatial_derivative_is_regular_at_the_sonic_point():
    sound = kB*10000./(.7*PROTON_MASS_CGS)
    sonic = G*MJ*MASS/(2*sound)
    radius = sonic*jnp.array([.8, .999999, 1., 1.000001, 1.2])
    velocity, info = nonisothermal_parker_velocity(
        radius, constant_temperature, 10000./.7, MASS, 1.,
        base_radius_cm=BASE, outer_radius_cm=32*BASE, ngrid=65, return_info=True)
    # Independent momentum equation, with the analytic regular sonic limit.
    expected = (2*sound/radius-G*MJ*MASS/radius**2)/((velocity*1e5)**2-sound)
    expected = expected.at[2].set(1/sonic)
    selected = jnp.array([0, 2, 4])
    assert info["converged"]
    assert np.isfinite(info["log_velocity_gradient_cm1"]).all()
    np.testing.assert_allclose(info["log_velocity_gradient_cm1"][selected],
                               expected[selected], rtol=2e-9)
    np.testing.assert_array_equal(info["sound_squared_gradient_cm_s2"], 0.)


def test_spatial_derivative_matches_independent_queries_with_fixed_knots():
    knots = BASE*jnp.array([1., 1.05, 1.3, 2.3, 4.])
    parameters = jnp.log(knots), jnp.log(jnp.array([6000., 7000., 11000., 14000., 12000.]))
    radius = BASE*jnp.array([1.07, 1.2, 1.8, 3., 5.])

    def evaluate(query, return_info=False):
        return nonisothermal_parker_velocity(
            query, _effective_temperature, parameters, MASS, 1.,
            base_radius_cm=BASE, outer_radius_cm=32*BASE, ngrid=65,
            tidal_coefficient_s2=7e-9, return_info=return_info)

    _, info = evaluate(radius, True)
    step = 1e-5
    numerical = (jnp.log(evaluate(radius*jnp.exp(step)))
                 -jnp.log(evaluate(radius*jnp.exp(-step))))/(2*step*radius)
    assert info["converged"]
    np.testing.assert_allclose(info["log_velocity_gradient_cm1"], numerical, rtol=2e-7)


def test_flow_terms_use_mass_conservation_even_across_a_steep_density_fall():
    radius = BASE*jnp.array([1., 1.3, 2., 4.])
    power = 5.
    velocity = 1e4*(radius/BASE)**power
    density = 1e13/(4*jnp.pi*radius**2*velocity)
    temperature, mu = jnp.full_like(radius, 8000.), jnp.full_like(radius, .8)
    result = thermal_flow_terms(radius, density, temperature, mu, velocity,
        log_velocity_gradient_cm1=power/radius,
        sound_squared_gradient_cm_s2=jnp.zeros_like(radius))
    expected = -(2+power)*density*velocity*kB*temperature/(mu*PROTON_MASS_CGS*radius)
    np.testing.assert_allclose(result.expansion, expected, rtol=2e-15)
    np.testing.assert_array_equal(result.advection, 0.)
    coarse = thermal_flow_terms(radius, density, temperature, mu, velocity)
    assert abs(coarse.expansion[-1]/expected[-1]) > 10
    with pytest.raises(ValueError, match="both wind"):
        thermal_flow_terms(radius, density, temperature, mu, velocity,
                           log_velocity_gradient_cm1=power/radius)
    with pytest.raises(ValueError, match="profile shape"):
        thermal_flow_terms(radius, density, temperature, mu, velocity,
            log_velocity_gradient_cm1=power/radius[:-1],
            sound_squared_gradient_cm_s2=jnp.zeros_like(radius))


def test_flow_gradient_parameter_derivatives_include_the_implicit_wind():
    radius = BASE*jnp.array([1., 1.3, 2., 3.])
    mu = jnp.array([.9, .85, .8, .75])

    def observable(parameters):
        temperature = jnp.array([8000., 9000., 10000., 11000.])*jnp.exp(parameters[0])
        wind, info = wind_from_profiles(radius, temperature, mu,
            1e13*jnp.exp(parameters[1]), MASS, base_radius_cm=BASE,
            outer_radius_cm=32*BASE, ngrid=65, return_info=True)
        terms = thermal_flow_terms(radius, wind.density_g_cm3, temperature, mu,
            wind.velocity_kms*1e5,
            log_velocity_gradient_cm1=info["log_velocity_gradient_cm1"],
            sound_squared_gradient_cm_s2=info["sound_squared_gradient_cm_s2"])
        return jnp.log(-jnp.sum(terms.advection+terms.expansion))

    parameters = jnp.zeros(2)
    derivative = jax.jit(jax.grad(observable))(parameters)
    step = 1e-5
    numerical = jnp.array([(observable(parameters+step*axis)
                           -observable(parameters-step*axis))/(2*step)
                          for axis in jnp.eye(2)])
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, numerical, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(derivative[1], 1., rtol=2e-12)


def test_solar_energy_uses_wind_gradients_after_removing_the_lower_endpoint():
    # Isolate the actual flow/energy plumbing from the separately tested atomic
    # solver. The composition closure is a fixed, monatomic particle count.
    model = object.__new__(SolarAtmosphereModel)
    model.radius_cm = BASE*jnp.array([1.05, 1.4, 2.5])
    model.cells, model.state_size, model.mass_per_h = 3, 1, PROTON_MASS_CGS
    model.dense_lower_cells, model.lower_temperature_k = 1, 6000.
    model.extend_wind_to_edge = False
    model.atomic_lte_mask = jnp.array([True, False, False])
    model.planet_mass_mj, model.tidal_coefficient_s2, model.wind_ngrid = MASS, 7e-9, 65
    model.unpack = MethodType(lambda self, state, density: {
        "mean_molecular_weight": jnp.full(self.cells, .8)}, model)

    def composition(self, state, density, temperature, incident, **kwargs):
        del density, temperature, incident, kwargs
        return state, dict(valid=jnp.array(True),
            mean_molecular_weight=jnp.full(self.cells, .8),
            thermal_net=jnp.zeros(self.cells), thermal_scale=jnp.ones(self.cells))

    model.evaluate = MethodType(composition, model)
    temperature = jnp.array([6500., 8000., 10000.])
    state = jnp.zeros((3, 2))
    residual, physical = model.evaluate_atmosphere(state.ravel(), temperature, 1e13, 0.)
    assert np.isfinite(residual).all()
    info = physical["wind_info"]
    assert info["log_velocity_gradient_cm1"].shape == (3,)
    assert info["sound_squared_gradient_cm_s2"].shape == (3,)
    # The last Hermite knot has exactly zero sound-speed slope. A coarse
    # temperature difference would give nonzero advective cooling here.
    assert abs(float(physical["thermal_advection"][-1])) < 1e-20
    assert np.all(np.asarray(physical["thermal_expansion"]) < 0)
