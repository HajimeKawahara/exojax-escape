"""Joint radiation/flow conservation, implicit gradients and domain failures."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.coupled_hydrogen import CoupledHydrogenModel, atom_line_profiles
from escape.hydrogen_atom import atom_rate_coefficients
from escape.hydrogen_nlte import solve_statistical_equilibrium
from escape.hydrogen_rates import LIGHT_SPEED_CGS, PLANCK_CGS
from escape.parker import PROTON_MASS_CGS
from escape.radiation import moment_geometry
from escape.reactive_flow import ADIABATIC_INDEX, BOLTZMANN_CGS

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def flow_experiment():
    """A short subsonic branch within the atomic temperature-data domain."""
    model = CoupledHydrogenModel(
        np.linspace(1e10, 1.0001e10, 4), nmax=3, quadrature_order=2,
    )
    inner = jnp.zeros_like(model.grid.frequency_hz)
    outer = model.boundary_blackbody(10000., .05)
    base = model._initial_fractions(
        jnp.full(3, 1e9), jnp.full(3, 9000.), inner, outer,
    )[0]
    assert np.all(np.isfinite(base))
    return model, inner, outer, base


def run_flow(experiment, density=1e9, velocity=1e6, gravity=1e21):
    model, inner, outer, base = experiment
    return model.solve_flow(
        jnp.asarray(density), 9000., jnp.asarray(velocity), base, inner, outer,
        jnp.asarray(gravity),
    )


def test_joint_flow_closes_material_and_boundary_energy_budgets(flow_experiment, record_property):
    model, _, _, _ = flow_experiment
    result = run_flow(flow_experiment)
    assert result.converged and result.radiation.converged
    profile = result.profile
    area, volume = moment_geometry(model.edges_cm)
    mass_per_sr = PROTON_MASS_CGS * profile.n_h_cm3 * profile.velocity_cm_s * area
    np.testing.assert_allclose(mass_per_sr, mass_per_sr[0], rtol=5e-15)
    gas_change = mass_per_sr[0] * (result.bernoulli_erg_g[-1] - result.bernoulli_erg_g[0])
    integrated_heating = volume @ result.radiation.heating_erg_cm3_s
    material_error = abs(gas_change - integrated_heating) / (abs(gas_change) + abs(integrated_heating))
    record_property("material_energy_error", float(material_error))
    record_property("boundary_budget_error", float(result.energy_balance_error))
    record_property("exchange_budget_error", float(result.energy_exchange_error))
    np.testing.assert_allclose(material_error, result.material_energy_error, rtol=1e-3, atol=1e-10)
    assert material_error < 5e-8
    assert result.energy_balance_error < 1e-12
    assert result.residual_norm < 1e-9
    assert np.all(profile.mach_number < 1)
    assert abs(profile.temperature_k[-1] - profile.temperature_k[0]) > .1
    assert abs(profile.mean_molecular_weight[-1] - profile.mean_molecular_weight[0]) > 1e-10


def test_joint_flow_implicit_density_gradient_matches_finite_difference(flow_experiment, record_property):
    def observable(log_density):
        result = run_flow(flow_experiment, density=jnp.exp(log_density))
        return jnp.log(result.profile.temperature_k[-1])

    point, step = jnp.log(1e9), 1e-4
    derivative = jax.grad(observable)(point)
    plus, minus = observable(point + step), observable(point - step)
    finite_difference = (plus - minus) / (2 * step)
    relative_error = abs(derivative - finite_difference) / abs(finite_difference)
    record_property("implicit_gradient_relative_error", float(relative_error))
    assert np.all(np.isfinite([derivative, plus, minus, finite_difference]))
    assert abs(derivative) > 1e-6
    assert relative_error < 2e-5


def test_joint_flow_rejects_negative_gravity_and_critical_inflow(flow_experiment):
    _, _, _, base = flow_experiment
    invalid_gravity = run_flow(flow_experiment, gravity=-1e21)
    assert not invalid_gravity.converged
    assert np.all(np.isnan(invalid_gravity.profile.temperature_k))
    critical_speed = jnp.sqrt(
        ADIABATIC_INDEX * BOLTZMANN_CGS * 9000. * (2 - jnp.sum(base)) / PROTON_MASS_CGS,
    )
    critical = run_flow(flow_experiment, velocity=critical_speed)
    assert not critical.converged
    assert np.all(np.isnan(critical.profile.velocity_cm_s))


def test_joint_flow_spatial_refinement_reduces_population_and_temperature_changes(
    flow_experiment, record_property,
):
    """Refine space only, retaining one analytic-vacuum inflow population."""
    reference, inner, outer, _ = flow_experiment
    radii = np.array([1e10, 1.0001e10])
    # Transparent spherical Eddington transfer has constant J. This formula
    # fixes the same inflow for every spatial grid, without a discretized RT
    # solve or a grid-specific population initial condition.
    mean = (radii[0]**2 * inner + radii[1]**2 * outer) / np.sum(radii**2)
    profiles, _ = atom_line_profiles(reference.atom, 9000., reference.grid)
    continuum = LIGHT_SPEED_CGS**2 * mean / (2 * PLANCK_CGS * reference.grid.frequency_hz**3)
    line_mean = profiles @ (mean * reference.grid.weight_hz)
    lines = LIGHT_SPEED_CGS**2 * line_mean / (2 * PLANCK_CGS * reference.atom.line_frequency_hz**3)
    rates = atom_rate_coefficients(
        reference.atom, 9000., lines, continuum, reference.grid.frequency_hz, reference.grid.weight_hz,
    )
    equilibrium = solve_statistical_equilibrium(1e9, rates)
    assert equilibrium.converged
    base = equilibrium.bound_cm3 / 1e9
    populations, temperatures = [], []
    for count in [4, 7, 13]:
        model = reference if count == 4 else CoupledHydrogenModel(
            np.linspace(*radii, count), nmax=3, quadrature_order=2,
        )
        np.testing.assert_array_equal(model.grid.frequency_hz, reference.grid.frequency_hz)
        result = run_flow((model, inner, outer, base))
        assert result.converged, (
            count, result.residual_norm, result.energy_balance_error, result.material_energy_error,
        )
        n3 = result.profile.n_h_cm3[-1] * jnp.sum(
            result.profile.bound_fraction[-1] * (model.atom.principal_n == 3),
        )
        temperature = result.profile.temperature_k[-1]
        populations.append(float(n3))
        temperatures.append(float(temperature))
        record_property(f"joint_spatial_n3_cm3_{count}", float(n3))
        record_property(f"joint_spatial_temperature_k_{count}", float(temperature))

    population_changes = np.abs(np.diff(populations)) / np.asarray(populations[1:])
    temperature_changes = np.abs(np.diff(temperatures)) / np.asarray(temperatures[1:])
    record_property("joint_spatial_population_change", float(population_changes[-1]))
    record_property("joint_spatial_temperature_change", float(temperature_changes[-1]))
    record_property(
        "joint_spatial_population_reduction", float(population_changes[1] / population_changes[0]),
    )
    record_property(
        "joint_spatial_temperature_reduction", float(temperature_changes[1] / temperature_changes[0]),
    )
    assert population_changes[-1] < .01
    assert temperature_changes[-1] < .01
    if population_changes[0] > 1e-8:
        assert population_changes[1] < .9 * population_changes[0]
    if temperature_changes[0] > 1e-10:
        assert temperature_changes[1] < .9 * temperature_changes[0]
