"""Tidal first integral and the transonic branch in a compact wind domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from exojax.utils.constants import G, MJ, kB

from escape.hydrogen_forward import PLANET_RADIUS_CM
from escape.nonisothermal import _tidal_isothermal_seed, nonisothermal_parker_velocity
from escape.parker import PROTON_MASS_CGS
from escape.solar import SolarAtmosphereModel

jax.config.update("jax_enable_x64", True)
BASE, MASS = PLANET_RADIUS_CM, 2.17
TIDE = 3*(2*np.pi/(1.48*86400))**2


def constant_temperature(radius, temperature):
    return temperature+jnp.zeros_like(radius)


def test_tidal_wind_has_correct_branches_when_no_tide_sonic_point_is_outside():
    temperature = 5000.
    sound_squared = kB*temperature/PROTON_MASS_CGS
    gravity = G*MJ*MASS
    sonic = brentq(lambda r: 2*sound_squared*r-gravity+TIDE*r**3, BASE, 8*BASE)
    assert gravity/(2*sound_squared) > 8*BASE
    radius = np.array([BASE, 1.4*BASE, sonic*(1-1e-6), sonic,
                       sonic*(1+1e-6), 3*BASE, 7*BASE])
    velocity, info = nonisothermal_parker_velocity(radius, constant_temperature,
        temperature, MASS, 1., base_radius_cm=BASE, outer_radius_cm=8*BASE,
        tidal_coefficient_s2=TIDE, ngrid=129, return_info=True)
    expected = []
    for r in radius:
        if r == sonic:
            expected.append(1.)
            continue
        t = np.log(r/sonic)
        forcing = (4*(t+np.expm1(-t))
                   + TIDE*sonic**2/sound_squared*(2*np.expm1(-t)+np.expm1(2*t)))
        root = brentq(lambda z: np.expm1(2*z)-2*z-forcing,
                      *([-100., 0.] if r < sonic else [0., 20.]), xtol=1e-15)
        expected.append(np.exp(root))
    assert info["valid"] and info["converged"]
    assert info["subsonic_margin"] > 0 and info["supersonic_margin"] > 0
    np.testing.assert_allclose(velocity, np.sqrt(sound_squared)*np.array(expected)/1e5, rtol=2e-8)
    np.testing.assert_allclose(info["sonic_radius_cm"], sonic, rtol=2e-12)
    regular_slope = np.sqrt(1+1.5*TIDE*sonic**2/sound_squared)/sonic
    np.testing.assert_allclose(info["log_velocity_gradient_cm1"][3], regular_slope, rtol=2e-8)


@pytest.mark.parametrize("strength", [0., 1., 20., 100.])
def test_initializer_keeps_tiny_sonic_values_and_the_regular_slope(strength):
    offsets = jnp.array([-1e-15, -1e-12, 0., 1e-12, 1e-15])
    values = np.asarray(_tidal_isothermal_seed(offsets, strength))
    assert np.isfinite(values).all()
    assert values[2] == 0.
    np.testing.assert_array_equal(np.sign(values), np.sign(offsets))
    keep = np.asarray(offsets) != 0
    np.testing.assert_allclose(values[keep]/np.asarray(offsets)[keep],
                               np.sqrt(1+1.5*strength), rtol=1e-10)


def test_compact_tidal_wind_implicit_derivative_matches_resolved_difference():
    radii = BASE*jnp.array([1.2, 1.8, 2.5, 3.])

    def value(log_temperature):
        wind = nonisothermal_parker_velocity(radii, constant_temperature,
            5000*jnp.exp(log_temperature), MASS, 1., base_radius_cm=BASE,
            outer_radius_cm=8*BASE, tidal_coefficient_s2=TIDE, ngrid=65)
        return jnp.log(wind[1])+.3*jnp.log(wind[-1])

    derivative = jax.jit(jax.grad(value))(0.)
    step = 1e-5
    numerical = (value(step)-value(-step))/(2*step)
    assert np.isfinite(derivative) and np.isfinite(numerical)
    np.testing.assert_allclose(derivative, numerical, rtol=2e-6)


@pytest.mark.parametrize("outer", [np.nan, np.inf, -1., 2.9])
def test_solar_wind_domain_must_cover_the_rt_edge(outer):
    with pytest.raises(ValueError, match="must cover the RT outer edge"):
        SolarAtmosphereModel([1., 1.5, 2., 3.], wind_outer_radius_rp=outer)
