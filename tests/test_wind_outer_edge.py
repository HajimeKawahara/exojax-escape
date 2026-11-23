"""An explicit outer pressure knot preserves the flow equations and their AD."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.utils.constants import kB

from escape.atmosphere_wind import thermal_flow_terms
from escape.hydrogen_forward import PLANET_RADIUS_CM
from escape.parker import PROTON_MASS_CGS, parker_velocity
from escape.solar import SolarAtmosphereModel

jax.config.update("jax_enable_x64", True)


def model(*, extend=True, lower_cells=1):
    # Exercise actual wind/energy plumbing without compiling atomic SE.
    result = object.__new__(SolarAtmosphereModel)
    result.radius_cm = PLANET_RADIUS_CM*jnp.array([1.05, 1.4, 2., 2.65])
    result.hydrogen = SimpleNamespace(edges_cm=PLANET_RADIUS_CM*jnp.array([1., 3.]))
    result.cells = 4
    result.dense_lower_cells = lower_cells
    result.lower_temperature_k = 10000.
    result.extend_wind_to_edge = extend
    result.wind_outer_radius_rp = 32.
    result.planet_mass_mj, result.tidal_coefficient_s2, result.wind_ngrid = 2.17, 0., 65
    result.unpack = MethodType(lambda self, state, density: {
        "mean_molecular_weight": state}, result)
    return result


@pytest.mark.parametrize("lower_cells", [0, 1])
def test_outer_pressure_knot_preserves_isothermal_solution_and_shapes(lower_cells):
    atmosphere = model(lower_cells=lower_cells)
    temperature, mu = jnp.full(4, 10000.), jnp.full(4, .8)
    wind, info = atmosphere._wind(mu, temperature, 1e13)
    assert info["valid"] and info["converged"]
    assert wind.velocity_kms.shape == wind.density_g_cm3.shape == (4,)
    assert info["log_velocity_gradient_cm1"].shape == (4,)
    assert info["sound_squared_gradient_cm_s2"].shape == (4,)
    np.testing.assert_allclose(wind.velocity_kms,
        parker_velocity(atmosphere.radius_cm, 10000., 2.17, .8), rtol=2e-9)
    np.testing.assert_allclose(4*jnp.pi*atmosphere.radius_cm**2
        *wind.density_g_cm3*wind.velocity_kms*1e5, 1e13, rtol=2e-15)
    np.testing.assert_allclose(info["sound_squared_gradient_cm_s2"], 0., atol=1e-12)


def test_last_thermal_cell_has_the_extrapolated_pressure_slope():
    temperature = jnp.array([10000., 10500., 11000., 11500.])
    mu = jnp.array([.85, .8, .75, .7])
    extended, original = model(), model(extend=False)
    wind, info = extended._wind(mu, temperature, 1e13)
    _, original_info = original._wind(mu, temperature, 1e13)
    slope = (np.log(float((temperature[-1]/mu[-1])/(temperature[-2]/mu[-2])))
             / np.log(float(extended.radius_cm[-1]/extended.radius_cm[-2])))
    expected = kB*temperature[-1]/(mu[-1]*PROTON_MASS_CGS)*slope/extended.radius_cm[-1]
    assert info["valid"] and info["converged"]
    np.testing.assert_allclose(info["sound_squared_gradient_cm_s2"][-1], expected, rtol=3e-14)
    np.testing.assert_allclose(original_info["sound_squared_gradient_cm_s2"][-1], 0., atol=1e-12)
    terms = thermal_flow_terms(extended.radius_cm, wind.density_g_cm3, temperature, mu,
        wind.velocity_kms*1e5,
        log_velocity_gradient_cm1=info["log_velocity_gradient_cm1"],
        sound_squared_gradient_cm_s2=info["sound_squared_gradient_cm_s2"])
    np.testing.assert_allclose(terms.advection[-1],
        -1.5*wind.density_g_cm3[-1]*wind.velocity_kms[-1]*1e5*expected, rtol=3e-14)
    assert terms.advection[-1] < 0


def test_pressure_extension_derivative_includes_temperature_mu_and_implicit_wind():
    atmosphere = model()

    def observable(parameters):
        temperature = jnp.array([10000., 10500., 11000., 11500.])
        temperature = temperature.at[-1].multiply(jnp.exp(parameters[0]))
        mu = jnp.array([.85, .8, .75, .7]).at[-1].multiply(jnp.exp(parameters[1]))
        wind, info = atmosphere._wind(mu, temperature, 1e13)
        return (jnp.log(wind.velocity_kms[1])
                + .2*jnp.log(info["sound_squared_gradient_cm_s2"][-1]))

    parameters = jnp.zeros(2)
    derivative = jax.jit(jax.grad(observable))(parameters)
    step = 1e-5
    numerical = jnp.array([(observable(parameters+step*axis)
                           -observable(parameters-step*axis))/(2*step)
                          for axis in jnp.eye(2)])
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, numerical, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(derivative[0], -derivative[1], rtol=2e-12)
