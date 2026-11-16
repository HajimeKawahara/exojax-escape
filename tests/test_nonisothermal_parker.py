"""Independent sonic shooting, isothermal limits, and implicit derivatives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from exojax.utils.constants import G, MJ, RJ, kB

from escape.nonisothermal import nonisothermal_parker_velocity
from escape.parker import PROTON_MASS_CGS, parker_velocity

jax.config.update("jax_enable_x64", True)

RADIUS_CM = 1.9 * RJ
MASS_MJ = 2.17
MU = 2 / 3
TIDAL_COEFFICIENT_S2 = 3 * (2 * np.pi / (1.48 * 86400)) ** 2
RADII = np.array([1.0, 1.03, 1.3, 2.5, 4.0, 10.0]) * RADIUS_CM


def temperature(radius_cm, upper_temperature):
    return 4500 + (upper_temperature - 4500) * (
        -jnp.expm1(-(radius_cm / RADIUS_CM - 1) / 0.12)
    )


def constant_temperature(radius_cm, temperature_k):
    return temperature_k + jnp.zeros_like(radius_cm)


def powerlaw_temperature(radius_cm, parameters):
    temperature_k, exponent = parameters
    return temperature_k * (radius_cm / RADIUS_CM) ** (-exponent)


def wind(radii=RADII, top=22149.0, mass=MASS_MJ, mu=MU, **kwargs):
    return nonisothermal_parker_velocity(
        radii, temperature, top, mass, mu,
        base_radius_cm=RADIUS_CM, outer_radius_cm=32 * RADIUS_CM, **kwargs,
    )


def shooting_reference(radii, top, tidal_coefficient_s2=0.0):
    """DOP853 integration from the regular sonic slope, independent of JAX."""
    speed_squared = kB * top / (MU * PROTON_MASS_CGS)
    gravity = G * MJ * MASS_MJ / (RADIUS_CM * speed_squared)
    tide = tidal_coefficient_s2 * RADIUS_CM**2 / speed_squared

    def quantities(x):
        exponential = np.exp(-(x - 1) / 0.12)
        sound = 1 - (1 - 4500 / top) * exponential
        first = (1 - 4500 / top) * exponential / 0.12
        second = -first / 0.12
        beta = x * first / sound
        escape = gravity / (x * sound)
        tidal_force = tide * x**2 / sound
        numerator = 2 - beta - escape + tidal_force
        derivative = (
            -(beta + x**2 * second / sound - beta**2)
            + escape * (1 + beta) + tidal_force * (2 - beta)
        )
        return sound, beta, numerator, derivative

    critical = brentq(lambda x: quantities(x)[2], 1, 32, xtol=1e-13)
    sound, beta, _, derivative = quantities(critical)
    log_critical = np.log(critical)
    log_speed = 0.5 * np.log(sound)
    slope = (beta + np.sqrt(beta**2 + 8 * derivative)) / 4

    def equation(log_radius, values):
        sound, _, numerator, _ = quantities(np.exp(log_radius))
        return [numerator / (np.exp(2 * values[0]) / sound - 1)]

    displacement = 1e-6
    inner = solve_ivp(
        equation, (log_critical - displacement, 0),
        [log_speed - slope * displacement], method="DOP853",
        rtol=1e-11, atol=1e-12, dense_output=True,
    )
    outer = solve_ivp(
        equation, (log_critical + displacement, np.log(32)),
        [log_speed + slope * displacement], method="DOP853",
        rtol=1e-11, atol=1e-12, dense_output=True,
    )
    assert inner.success and outer.success
    query = np.log(np.asarray(radii) / RADIUS_CM)
    solution = np.where(query < log_critical, inner.sol(query)[0], outer.sol(query)[0])
    return np.sqrt(speed_squared) * np.exp(solution) / 1e5, critical * RADIUS_CM


@pytest.mark.parametrize("top", [6000.0, 13177.0, 22149.0, 30000.0])
def test_matches_independent_sonic_shooting(top):
    velocity, info = wind(top=top, return_info=True)
    expected, critical = shooting_reference(RADII, top)
    assert bool(info["valid"]) and bool(info["converged"])
    assert float(info["residual_norm"]) < 1e-8
    np.testing.assert_allclose(velocity, expected, rtol=2e-7, atol=1e-11)
    np.testing.assert_allclose(info["sonic_radius_cm"], critical, rtol=1e-12)


def test_constant_temperature_recovers_analytic_parker_solution():
    velocity, info = nonisothermal_parker_velocity(
        RADII, constant_temperature, 14000.0, MASS_MJ, MU,
        base_radius_cm=RADIUS_CM, outer_radius_cm=32 * RADIUS_CM,
        return_info=True,
    )
    expected = parker_velocity(RADII, 14000.0, MASS_MJ, MU)
    assert bool(info["valid"]) and bool(info["converged"])
    np.testing.assert_allclose(velocity, expected, rtol=2e-9)


def test_implicit_parameter_derivatives_match_finite_differences():
    def evaluate(log_parameters):
        top, mass, mu = jnp.exp(log_parameters)
        return jnp.log(wind(top=top, mass=mass, mu=mu))

    parameters = jnp.log(jnp.array([22149.0, MASS_MJ, MU]))
    jacobian = jax.jacfwd(evaluate)(parameters)
    assert np.isfinite(jacobian).all()
    step = 1e-5
    differences = np.column_stack([
        (np.asarray(evaluate(parameters + step * axis))
         - np.asarray(evaluate(parameters - step * axis))) / (2 * step)
        for axis in np.eye(3)
    ])
    np.testing.assert_allclose(jacobian, differences, rtol=3e-6, atol=2e-7)
    gradient = jax.grad(lambda inputs: jnp.sum(evaluate(inputs)))(parameters)
    np.testing.assert_allclose(gradient, np.asarray(jacobian).sum(axis=0), rtol=2e-10)


def test_velocity_and_derivative_are_regular_at_sonic_point():
    _, info = wind(return_info=True)
    critical = info["sonic_radius_cm"]
    top = 22149.0
    critical_sound = jnp.sqrt(kB * temperature(critical, top) / (MU * PROTON_MASS_CGS)) / 1e5
    derivative = jax.grad(lambda log_radius: jnp.log(wind(jnp.exp(log_radius))))
    log_critical = jnp.log(critical)
    slopes = jax.vmap(derivative)(log_critical + jnp.array([-1e-6, 0.0, 1e-6]))
    velocities = wind(critical * jnp.array([1 - 1e-6, 1.0, 1 + 1e-6]))
    assert np.isfinite(slopes).all()
    np.testing.assert_allclose(velocities[1], critical_sound, rtol=1e-10)
    beta = jax.grad(lambda value: jnp.log(temperature(jnp.exp(value), top)))(log_critical)
    expected = (beta + jnp.sqrt(beta**2 + 8 * info["critical_slope"])) / 4
    np.testing.assert_allclose(slopes, expected, rtol=3e-6)


def test_grid_refinement_preserves_velocities_and_gradients():
    def evaluate(top, ngrid):
        return jnp.log(wind(top=top, ngrid=ngrid))

    coarse = evaluate(22149.0, 129)
    fine = evaluate(22149.0, 257)
    np.testing.assert_allclose(coarse, fine, rtol=0, atol=2e-8)
    coarse_derivative = jax.jacfwd(evaluate)(22149.0, 129)
    fine_derivative = jax.jacfwd(evaluate)(22149.0, 257)
    np.testing.assert_allclose(coarse_derivative, fine_derivative, rtol=2e-6, atol=1e-10)


def test_cooling_powerlaw_has_analytic_critical_point_and_regular_slope():
    exponent = 0.25
    base_temperature = 12000.0

    def cooling_wind(radii, temperature_k=base_temperature, return_info=False):
        return nonisothermal_parker_velocity(
            radii, powerlaw_temperature, (temperature_k, exponent), MASS_MJ, MU,
            base_radius_cm=RADIUS_CM, outer_radius_cm=32 * RADIUS_CM,
            return_info=return_info,
        )

    _, info = cooling_wind(RADII, return_info=True)
    gravity = G * MJ * MASS_MJ * MU * PROTON_MASS_CGS / (kB * base_temperature * RADIUS_CM)
    critical = RADIUS_CM * (gravity / (2 + exponent)) ** (1 / (1 - exponent))
    assert bool(info["valid"]) and bool(info["converged"])
    np.testing.assert_allclose(info["sonic_radius_cm"], critical, rtol=1e-12)
    expected_slope = (-exponent + np.sqrt(exponent**2 + 8 * (1 - exponent) * (2 + exponent))) / 4
    slope = jax.grad(lambda value: jnp.log(cooling_wind(jnp.exp(value))))(np.log(critical))
    np.testing.assert_allclose(slope, expected_slope, rtol=2e-9)
    radius_derivative = jax.grad(
        lambda log_temperature: jnp.log(cooling_wind(
            RADII, jnp.exp(log_temperature), return_info=True,
        )[1]["sonic_radius_cm"])
    )(np.log(base_temperature))
    np.testing.assert_allclose(radius_derivative, -1 / (1 - exponent), rtol=1e-12)


def test_parameter_gradient_is_continuous_when_sonic_point_crosses_query():
    query = 3 * RADIUS_CM
    gravity_k = G * MJ * MASS_MJ * MU * PROTON_MASS_CGS / (kB * RADIUS_CM)
    exponential = np.exp(-2 / 0.12)
    crossing = brentq(
        lambda top: 2 * float(temperature(query, top)) / 3
        - (top - 4500) * exponential / 0.12 - gravity_k / 9,
        20000, 35000,
    )
    evaluate = lambda top: jnp.log(wind(query, top=top))
    derivative = jax.vmap(jax.grad(evaluate))(
        crossing + jnp.array([-1e-3, 0.0, 1e-3])
    )
    finite_difference = (evaluate(crossing + 0.01) - evaluate(crossing - 0.01)) / 0.02
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-6)


def test_invalid_domain_is_reported_instead_of_returning_a_wind():
    velocity, info = nonisothermal_parker_velocity(
        RADII, temperature, 9000.0, MASS_MJ, MU,
        base_radius_cm=RADIUS_CM, outer_radius_cm=3 * RADIUS_CM,
        return_info=True,
    )
    assert not bool(info["valid"])
    assert np.isnan(velocity).all()
    with pytest.raises(ValueError, match="odd integer"):
        wind(ngrid=64)


@pytest.mark.parametrize("top", [6000.0, 14000.0, 30000.0])
def test_tidal_wind_matches_independent_sonic_shooting(top):
    velocity, info = wind(
        top=top, tidal_coefficient_s2=TIDAL_COEFFICIENT_S2, return_info=True,
    )
    expected, critical = shooting_reference(RADII, top, TIDAL_COEFFICIENT_S2)
    assert bool(info["valid"]) and bool(info["converged"])
    np.testing.assert_allclose(velocity, expected, rtol=2e-7, atol=1e-11)
    np.testing.assert_allclose(info["sonic_radius_cm"], critical, rtol=1e-12)


def test_isothermal_tidal_wind_matches_integrated_momentum_equation():
    temperature_k = 14000.0
    sound_squared = kB * temperature_k / (MU * PROTON_MASS_CGS)
    gravity = G * MJ * MASS_MJ / (RADIUS_CM * sound_squared)
    tide = TIDAL_COEFFICIENT_S2 * RADIUS_CM**2 / sound_squared
    critical = brentq(lambda x: tide * x**3 + 2 * x - gravity, 1, 32)
    expected = []
    for x in RADII / RADIUS_CM:
        integrated_force = (
            4 * np.log(x / critical) + 2 * gravity * (1 / x - 1 / critical)
            + tide * (x**2 - critical**2)
        )
        equation = lambda q: np.expm1(2 * q) - 2 * q - integrated_force
        bracket = (-100, 0) if x < critical else (0, 20)
        expected.append(np.exp(brentq(equation, *bracket)))
    velocity, info = nonisothermal_parker_velocity(
        RADII, constant_temperature, temperature_k, MASS_MJ, MU,
        base_radius_cm=RADIUS_CM, outer_radius_cm=32 * RADIUS_CM,
        tidal_coefficient_s2=TIDAL_COEFFICIENT_S2, return_info=True,
    )
    assert bool(info["valid"]) and bool(info["converged"])
    np.testing.assert_allclose(
        velocity, np.sqrt(sound_squared) * np.asarray(expected) / 1e5, rtol=2e-9,
    )
    np.testing.assert_allclose(info["sonic_radius_cm"] / RADIUS_CM, critical, rtol=1e-12)


def test_tidal_parameter_and_sonic_radius_derivatives():
    def evaluate(parameters):
        top, mass, mu, tide = jnp.exp(parameters)
        velocity, info = wind(
            top=top, mass=mass, mu=mu, tidal_coefficient_s2=tide, return_info=True,
        )
        return jnp.log(jnp.append(velocity, info["sonic_radius_cm"] / RADIUS_CM))

    parameters = jnp.log(jnp.array([14000.0, MASS_MJ, MU, TIDAL_COEFFICIENT_S2]))
    jacobian = jax.jacfwd(evaluate)(parameters)
    step = 1e-5
    differences = np.column_stack([
        (evaluate(parameters + step * axis) - evaluate(parameters - step * axis)) / (2 * step)
        for axis in np.eye(4)
    ])
    assert np.isfinite(jacobian).all()
    np.testing.assert_allclose(jacobian, differences, rtol=3e-6, atol=2e-7)
    gradient = jax.grad(lambda inputs: jnp.sum(evaluate(inputs)))(parameters)
    np.testing.assert_allclose(gradient, np.asarray(jacobian).sum(axis=0), rtol=2e-10)


def test_tidal_sonic_regularity_and_grid_convergence():
    def evaluate(radii, log_tide=0.0, ngrid=129, return_info=False):
        return wind(
            radii, top=14000.0,
            tidal_coefficient_s2=TIDAL_COEFFICIENT_S2 * jnp.exp(log_tide),
            ngrid=ngrid, return_info=return_info,
        )

    coarse, info = evaluate(RADII, return_info=True)
    fine = evaluate(RADII, ngrid=257)
    np.testing.assert_allclose(coarse, fine, rtol=2e-8)
    coarse_gradient = jax.jacfwd(lambda k: jnp.log(evaluate(RADII, k)))(0.0)
    fine_gradient = jax.jacfwd(lambda k: jnp.log(evaluate(RADII, k, 257)))(0.0)
    np.testing.assert_allclose(coarse_gradient, fine_gradient, rtol=2e-6)
    critical = info["sonic_radius_cm"]
    critical_sound = jnp.sqrt(kB * temperature(critical, 14000.0) / (MU * PROTON_MASS_CGS)) / 1e5
    np.testing.assert_allclose(evaluate(critical), critical_sound, rtol=1e-10)
    derivative = jax.grad(lambda x: jnp.log(evaluate(jnp.exp(x))))
    slopes = jax.vmap(derivative)(jnp.log(critical) + jnp.array([-1e-6, 0, 1e-6]))
    beta = jax.grad(lambda x: jnp.log(temperature(jnp.exp(x), 14000.0)))(jnp.log(critical))
    expected = (beta + jnp.sqrt(beta**2 + 8 * info["critical_slope"])) / 4
    np.testing.assert_allclose(slopes, expected, rtol=3e-6)


@pytest.mark.parametrize("tide", [-TIDAL_COEFFICIENT_S2, np.nan, np.inf, 1e-4])
def test_invalid_tides_or_missing_tidal_critical_point_are_reported(tide):
    velocity, info = wind(tidal_coefficient_s2=tide, return_info=True)
    assert not bool(info["valid"])
    assert np.isnan(velocity).all()
