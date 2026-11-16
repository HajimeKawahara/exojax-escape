"""Independent solutions, physical equations, and derivatives of Parker flow."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import lambertw

from exojax.utils.constants import G, MJ, kB
from escapeh.parker import parker_velocity, sonic_radius_cm

jax.config.update("jax_enable_x64", True)

TEMPERATURE_K = 12000.0
PLANET_MASS_MJ = 2.17
MEAN_MOLECULAR_WEIGHT = 0.5
PROTON_MASS_G = 1.67262192369e-24
SOUND_SPEED_CGS = np.sqrt(kB * TEMPERATURE_K / (MEAN_MOLECULAR_WEIGHT * PROTON_MASS_G))
SONIC_RADIUS_CM = G * MJ * PLANET_MASS_MJ / (2 * SOUND_SPEED_CGS**2)


def velocity(radius_cm):
    return parker_velocity(radius_cm, TEMPERATURE_K, PLANET_MASS_MJ, MEAN_MOLECULAR_WEIGHT)


def test_velocity_matches_independent_lambert_w_solution():
    radius_ratio = np.array([0.05, 0.1, 0.3, 0.8, 0.99, 1.01, 1.2, 2.0, 5.0, 20.0])
    argument = -radius_ratio**-4 * np.exp(3 - 4 / radius_ratio)
    branches = np.where(radius_ratio < 1, 0, -1)
    expected = SOUND_SPEED_CGS / 1e5 * np.sqrt(-lambertw(argument, branches).real)
    np.testing.assert_allclose(velocity(SONIC_RADIUS_CM * radius_ratio), expected, rtol=5e-11)
    np.testing.assert_allclose(
        sonic_radius_cm(TEMPERATURE_K, PLANET_MASS_MJ, MEAN_MOLECULAR_WEIGHT),
        SONIC_RADIUS_CM, rtol=1e-14,
    )


def test_sonic_value_and_derivatives_are_regular():
    # The accelerating solution has u = 1 + delta - delta^2/2 + delta^3/4 + ... .
    delta = np.array([-1e-6, -1e-8, 0.0, 1e-8, 1e-6])
    expected = 1 + delta - 0.5 * delta**2 + 0.25 * delta**3
    mach = velocity(SONIC_RADIUS_CM * (1 + delta)) / (SOUND_SPEED_CGS / 1e5)
    np.testing.assert_allclose(mach, expected, rtol=2e-12)

    radius_derivative = jax.vmap(jax.grad(velocity))(
        jnp.asarray(SONIC_RADIUS_CM * (1 + delta))
    )
    expected_derivative = SOUND_SPEED_CGS / 1e5 / SONIC_RADIUS_CM
    assert np.isfinite(radius_derivative).all()
    np.testing.assert_allclose(radius_derivative, expected_derivative, rtol=2e-6)

    def log_speed(log_parameters):
        return jnp.log(parker_velocity(*jnp.exp(log_parameters)))

    parameters = np.log([SONIC_RADIUS_CM, TEMPERATURE_K, PLANET_MASS_MJ, MEAN_MOLECULAR_WEIGHT])
    # At fixed remaining inputs: d ln(v)/d ln(r,T,M,mu) = (1,3/2,-1,-3/2).
    gradient = jax.grad(log_speed)(jnp.asarray(parameters))
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient, [1.0, 1.5, -1.0, -1.5], rtol=2e-10)


def test_physical_parameter_gradients_match_finite_differences():
    @jax.jit
    def log_speed(log_parameters):
        return jnp.log(parker_velocity(*jnp.exp(log_parameters)))

    gradient = jax.jit(jax.grad(log_speed))
    for radius_ratio in (0.3, 0.8, 1.4, 3.0):
        parameters = np.log([
            radius_ratio * SONIC_RADIUS_CM, TEMPERATURE_K,
            PLANET_MASS_MJ, MEAN_MOLECULAR_WEIGHT,
        ])
        automatic = np.asarray(gradient(parameters))
        step = 1e-5
        finite = np.empty(4)
        for index in range(4):
            displacement = np.eye(4)[index] * step
            finite[index] = (log_speed(parameters + displacement)
                             - log_speed(parameters - displacement)) / (2 * step)
        assert np.isfinite(automatic).all()
        np.testing.assert_allclose(automatic, finite, rtol=2e-7, atol=1e-8)


def test_wind_satisfies_momentum_equation_and_accelerates():
    def mach(radius_ratio):
        return velocity(SONIC_RADIUS_CM * radius_ratio) / (SOUND_SPEED_CGS / 1e5)

    radii = jnp.asarray([0.1, 0.3, 0.7, 1.3, 2.0, 10.0])
    speed = jax.vmap(mach)(radii)
    derivative = jax.vmap(jax.grad(mach))(radii)
    np.testing.assert_allclose(
        (speed - 1 / speed) * derivative,
        2 / radii - 2 / radii**2,
        rtol=2e-10, atol=1e-11,
    )
    assert (np.diff(speed) > 0).all()
    assert (np.asarray(speed)[np.asarray(radii) < 1] < 1).all()
    assert (np.asarray(speed)[np.asarray(radii) > 1] > 1).all()
