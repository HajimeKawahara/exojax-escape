"""Conservation, continuum limits and implicit derivatives of reactive flow."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from escape.hydrogen_nlte import RateCoefficients
from escape.reactive_flow import (
    ADIABATIC_INDEX, BOLTZMANN_CGS, PROTON_MASS_CGS,
    decode_flow_state, flow_domain_valid, reactive_flow_residual,
    solve_reactive_flow, species_production_s, spherical_shell_volume,
)

jax.config.update("jax_enable_x64", True)
CHI = 2.178709417e-11


def no_sources(radius, density, temperature, fractions, params):
    del radius, density, temperature, fractions, params
    zero = jnp.zeros(1)
    return RateCoefficients(jnp.zeros((1, 1)), jnp.zeros((1, 1)), zero, zero, zero, zero), jnp.asarray(0.)


def irradiated_sources(radius, density, temperature, fractions, params):
    del radius
    gamma, alpha0 = params
    alpha = alpha0 * (temperature / 9000.)**-.7
    zero = jnp.zeros(1)
    rates = RateCoefficients(
        jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.atleast_1d(gamma),
        jnp.atleast_1d(alpha), zero, zero,
    )
    electron = density * (1 - jnp.sum(fractions))
    # A deliberately simple photon energy closure for solver verification.
    # Each photoionization absorbs chi+2 kTstar; recombination emits chi+1.5kT.
    heating = density * fractions[0] * gamma * (CHI + 2 * BOLTZMANN_CGS * 12000.)
    heating -= electron**2 * alpha * (CHI + 1.5 * BOLTZMANN_CGS * temperature)
    return rates, heating


def solve(radius, velocity=3e6, source=no_sources, params=None, density=1e8, tide=0., iterations=32):
    return solve_reactive_flow(
        radius, density, 9000., velocity, jnp.array([.4]), jnp.array([0.]), CHI,
        source, params, 1e21, tidal_coefficient_s2=tide, iterations=iterations,
    )


@pytest.mark.parametrize("velocity", [5e5, 3e6])
def test_source_free_branch_conserves_mass_chemical_fractions_and_total_energy(velocity):
    radius = jnp.linspace(1e10, 1.04e10, 41)
    result = solve(radius, velocity)
    assert result.valid and result.converged
    profile = result.profile
    mass_flux = 4 * np.pi * radius**2 * PROTON_MASS_CGS * profile.n_h_cm3 * profile.velocity_cm_s
    np.testing.assert_allclose(mass_flux, result.mass_loss_g_s, rtol=5e-15)
    np.testing.assert_allclose(profile.bound_fraction, .4, atol=1e-14)
    np.testing.assert_allclose(profile.mean_molecular_weight, 1 / 1.6, atol=1e-14)
    np.testing.assert_allclose(result.bernoulli_erg_g, result.bernoulli_erg_g[0], rtol=2e-12)
    assert np.ptp(profile.temperature_k) > 10  # Temperature is actually solved.
    assert result.residual_norm < 1e-10


def continuum_solution(radius, velocity, params=None, tide=0.):
    """Independent primitive-variable IVP, with no discretized solver calls."""
    density_base, temperature_base, bound_base, gravity = 1e8, 9000., .4, 1e21

    def rhs(r, y):
        speed, temperature, bound = y
        density = density_base * (radius[0] / r)**2 * velocity / speed
        ionized = 1 - bound
        sound = BOLTZMANN_CGS * temperature * (1 + ionized) / PROTON_MASS_CGS
        derivative_potential = gravity / r**2 - tide * r
        if params is None:
            production, heating = 0., 0.
        else:
            gamma, alpha0 = params
            alpha = alpha0 * (temperature / 9000)**-.7
            production = density * ionized**2 * alpha - bound * gamma
            heating = density * bound * gamma * (CHI + 2 * BOLTZMANN_CGS * 12000.)
            heating -= (density * ionized)**2 * alpha * (CHI + 1.5 * BOLTZMANN_CGS * temperature)
        bound_gradient = production / speed
        chemical_gradient = -CHI * bound_gradient
        thermal_source = heating / (PROTON_MASS_CGS * density * speed) - chemical_gradient / PROTON_MASS_CGS
        speed_gradient = (
            2 * ADIABATIC_INDEX * sound / r - derivative_potential
            - (ADIABATIC_INDEX - 1) * thermal_source
        ) / (speed - ADIABATIC_INDEX * sound / speed)
        sound_gradient = (thermal_source - derivative_potential - speed * speed_gradient) / 2.5
        temperature_gradient = (
            sound_gradient * PROTON_MASS_CGS / BOLTZMANN_CGS + temperature * bound_gradient
        ) / (1 + ionized)
        return speed_gradient, temperature_gradient, bound_gradient

    solution = solve_ivp(
        rhs, [radius[0], radius[-1]], [velocity, temperature_base, bound_base],
        rtol=2e-11, atol=1e-10, dense_output=True,
    )
    assert solution.success
    return solution.sol(radius)


@pytest.mark.parametrize("velocity", [5e5, 3e6])
@pytest.mark.parametrize("tide", [0., 1e-9])
def test_grid_convergence_to_independent_momentum_energy_ivp(velocity, tide):
    errors = []
    for count in [33, 65, 129]:
        radius = jnp.linspace(1e10, 1.05e10, count)
        result = solve(radius, velocity, tide=tide)
        assert result.converged
        reference = continuum_solution(np.asarray(radius), velocity, tide=tide)
        values = np.stack((result.profile.velocity_cm_s, result.profile.temperature_k))
        errors.append(np.max(np.abs(values / reference[:2] - 1)))
    assert errors[-1] < 2e-4
    assert .45 < errors[1] / errors[0] < .55
    assert .45 < errors[2] / errors[1] < .55


def test_reaction_advection_variable_mu_and_exact_discrete_radiative_energy_budget():
    radius = jnp.linspace(1e10, 1.08e10, 65)
    params = jnp.array([3e-4, 1e-12])
    result = solve(radius, source=irradiated_sources, params=params)
    assert result.converged
    profile = result.profile
    rates, heating = jax.vmap(irradiated_sources, in_axes=(0, 0, 0, 0, None))(
        radius, profile.n_h_cm3, profile.temperature_k, profile.bound_fraction, params,
    )
    volumes = spherical_shell_volume(radius)
    kinetic_flux_change = result.mass_loss_g_s / (4 * np.pi) * jnp.diff(result.bernoulli_erg_g)
    np.testing.assert_allclose(kinetic_flux_change, volumes * heating[1:], rtol=5e-9)
    production, _ = species_production_s(profile, rates)
    nucleus_flux = profile.n_h_cm3[0] * profile.velocity_cm_s[0] * radius[0]**2
    np.testing.assert_allclose(
        nucleus_flux * jnp.diff(profile.bound_fraction, axis=0),
        volumes[:, None] * profile.n_h_cm3[1:, None] * production[1:], rtol=5e-9,
    )
    assert abs(profile.bound_fraction[-1, 0] - .4) > .01
    assert np.ptp(profile.mean_molecular_weight) > .001
    reference = continuum_solution(np.asarray(radius), 3e6, np.asarray(params))
    np.testing.assert_allclose(profile.velocity_cm_s, reference[0], rtol=3e-4)
    np.testing.assert_allclose(profile.temperature_k, reference[1], rtol=3e-4)
    np.testing.assert_allclose(profile.bound_fraction[:, 0], reference[2], rtol=3e-4)


def test_collision_ionization_uses_stored_latent_energy_without_external_cooling():
    def collision_source(radius, density, temperature, fractions, params):
        rates, heating = no_sources(radius, density, temperature, fractions, params)
        return rates._replace(electron_ionization_cm3_s=jnp.array([2e-12])), heating

    radius = jnp.linspace(1e10, 1.03e10, 33)
    collision = solve(radius, source=collision_source)
    frozen = solve(radius)
    assert collision.converged
    assert collision.profile.ionized_fraction[-1] > frozen.profile.ionized_fraction[-1]
    assert collision.profile.temperature_k[-1] < frozen.profile.temperature_k[-1]
    np.testing.assert_allclose(collision.bernoulli_erg_g, collision.bernoulli_erg_g[0], rtol=2e-12)


def test_global_residual_matches_standalone_cells_and_implicit_jvp_vjp():
    radius = jnp.linspace(1e10, 1.04e10, 17)
    params = jnp.array([3e-4, 1e-12])
    result = solve(radius, source=irradiated_sources, params=params)
    profile = decode_flow_state(result.state, radius, 1e8, 9000., 3e6)
    rates, heating = jax.vmap(irradiated_sources, in_axes=(0, 0, 0, 0, None))(
        radius, profile.n_h_cm3, profile.temperature_k, profile.bound_fraction, params,
    )
    residual = reactive_flow_residual(
        result.state, radius, 1e8, 9000., 3e6, jnp.array([.4]), jnp.array([0.]), CHI,
        rates, heating, 1e21,
    )
    assert np.max(np.abs(residual)) < 1e-10
    assert flow_domain_valid(profile, rates, heating)

    def observable(log_params):
        value = solve(radius, source=irradiated_sources, params=jnp.exp(log_params))
        return jnp.stack((jnp.log(value.profile.temperature_k[-1]), value.profile.ionized_fraction[-1]))

    point = jnp.log(params)
    direction = jnp.array([.7, -.3])
    _, derivative = jax.jvp(observable, (point,), (direction,))
    step = 1e-4
    finite_difference = (
        observable(point + step * direction) - observable(point - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-6, atol=2e-9)
    weights = jnp.array([.4, -.9])
    vjp = jax.grad(lambda p: observable(p) @ weights)(point)
    np.testing.assert_allclose(vjp @ direction, derivative @ weights, rtol=1e-10, atol=1e-12)


def test_batched_jit_keeps_inflow_parameters_independent():
    radius = jnp.linspace(1e10, 1.03e10, 9)
    batched = jax.jit(jax.vmap(lambda velocity: solve(radius, velocity)))
    result = batched(jnp.array([2.5e6, 3e6]))
    assert np.all(result.converged)
    assert result.profile.temperature_k.shape == (2, 9)
    assert result.profile.temperature_k[0, -1] != result.profile.temperature_k[1, -1]


def test_stiff_six_level_atom_with_full_photon_energy_solves_without_temperature_clipping():
    from escape.hydrogen_atom import atom_rate_coefficients, make_hydrogen_atom, solve_atom_populations
    from escape.hydrogen_continuum import total_radiative_energy
    from escape.hydrogen_rates import EV_CGS, PLANCK_CGS, planck_occupation

    atom = make_hydrogen_atom(3)
    frequency = jnp.asarray(np.geomspace(.1, 100, 160) * EV_CGS / PLANCK_CGS)
    weights = jnp.concatenate((
        jnp.diff(frequency)[:1] / 2, (frequency[2:] - frequency[:-2]) / 2,
        jnp.diff(frequency)[-1:] / 2,
    ))
    continuum_occupation = planck_occupation(frequency, 10000., .025)
    line_occupation = planck_occupation(atom.line_frequency_hz, 10000., .025)
    base = solve_atom_populations(
        1e9, atom, 9000., line_occupation, continuum_occupation, frequency, weights,
    )
    assert base.converged

    def source(radius, density, temperature, fractions, params):
        del radius, params
        electron = density * (1 - jnp.sum(fractions))
        rates = atom_rate_coefficients(
            atom, temperature, line_occupation, continuum_occupation, frequency, weights,
            electron_cm3=electron,
        )
        heating = total_radiative_energy(
            atom, temperature, density * fractions, electron, electron,
            line_occupation, continuum_occupation, frequency, weights,
        )
        return rates, heating

    result = solve_reactive_flow(
        jnp.linspace(1e10, 1.001e10, 9), 1e9, 9000., 3e6, base.bound_cm3 / 1e9,
        atom.excitation_energy_erg, atom.ionization_energy_erg[0], source, None, 1e21,
    )
    assert result.converged and result.residual_norm < 1e-10
    assert np.all(np.asarray(result.profile.bound_fraction) > 0)
    assert 8900 < result.profile.temperature_k[-1] < 9000
    assert np.max(atom.einstein_a_s) * (1e7 / 3e6) > 1e9


def test_domain_check_rejects_a_sonic_branch_crossing():
    radius = jnp.array([1e10, 1.1e10])
    state = jnp.array([[0., 0., jnp.log(.4 / .6)], [2., 0., jnp.log(.4 / .6)]])
    profile = decode_flow_state(state, radius, 1e8, 9000., 5e5)
    rates, heating = jax.vmap(no_sources, in_axes=(0, 0, 0, 0, None))(
        radius, profile.n_h_cm3, profile.temperature_k, profile.bound_fraction, None,
    )
    assert profile.mach_number[0] < 1 < profile.mach_number[1]
    assert not flow_domain_valid(profile, rates, heating)


@pytest.mark.parametrize("invalid_case", ["density", "grid", "sonic", "source", "iterations"])
def test_invalid_domains_or_unconverged_cells_fail_explicitly(invalid_case):
    radius = jnp.linspace(1e10, 1.05e10, 9)
    options = {}
    if invalid_case == "density":
        options["density"] = -1.
    elif invalid_case == "grid":
        radius = radius[::-1]
    elif invalid_case == "sonic":
        options["velocity"] = np.sqrt(ADIABATIC_INDEX * BOLTZMANN_CGS * 9000 * 1.6 / PROTON_MASS_CGS)
    elif invalid_case == "source":
        def invalid_source(r, n, t, f, p):
            rates, _ = no_sources(r, n, t, f, p)
            return rates, jnp.asarray(jnp.nan)
        options["source"] = invalid_source
    else:
        options["iterations"] = 1
    result = solve(radius, **options)
    assert not result.converged
    assert np.all(np.isnan(result.state))
    assert np.all(np.isnan(result.profile.temperature_k))


def test_no_temperature_clipping_when_callback_rejects_domain():
    def restricted_source(r, n, t, f, p):
        rates, _ = no_sources(r, n, t, f, p)
        return rates, jnp.where(t > 8999., 0., jnp.nan)

    result = solve(jnp.linspace(1e10, 1.01e10, 9), source=restricted_source)
    assert not result.converged
    assert np.all(np.isnan(result.profile.temperature_k))
