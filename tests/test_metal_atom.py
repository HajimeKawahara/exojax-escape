"""Packaged multilevel rates conserve particles and recover collisional LTE."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.metal_atom import (
    C,
    PLANCK,
    KB,
    electron_collision_coefficients,
    kinetic_line_heating,
    load_metal_atom,
    populations,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "species,size",
    [
        ("Fe II", 80),
        ("Ca II", 15),
        ("Mg II", 15),
        ("Si II", 15),
        ("S II", 72),
        ("C I", 15),
        ("Al II", 15),
        ("Mn II", 45),
        ("C II", 15),
        ("N I", 50),
        ("O I", 29),
        ("Fe III", 70),
    ],
)
def test_collision_only_lte_and_zero_heat(species, size):
    atom = load_metal_atom(species)
    temperature, electron, abundance = 5481.75, 4e10, 2e8
    rate = electron * electron_collision_coefficients(atom, temperature)
    population = populations(rate, abundance)
    lte = atom.weight * jnp.exp(
        -PLANCK * C * atom.energy_wavenumber / (KB * temperature)
    )
    lte = abundance * lte / lte.sum()
    np.testing.assert_allclose(population, lte, rtol=2e-12, atol=1e-20)
    assert float(jnp.min(population)) >= 0
    assert (
        abs(float(kinetic_line_heating(population, rate, atom.energy_wavenumber)))
        < 1e-12
    )


def test_general_rate_solution_and_gradient():
    base = jnp.array([[0.0, 3.0, 0.2], [2.0, 0.0, 4.0], [1.0, 5.0, 0.0]])

    def solve(parameter):
        rates = base.at[0, 2].set(jnp.exp(parameter))
        return populations(rates, 8.0)

    result = solve(0.0)
    rate = base.at[0, 2].set(1.0)
    residual = result @ rate - result * rate.sum(axis=1)
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)
    np.testing.assert_allclose(jnp.sum(result), 8.0, rtol=1e-14)
    derivative = jax.jacfwd(solve)(0.0)
    finite = (solve(1e-4) - solve(-1e-4)) / 2e-4
    np.testing.assert_allclose(derivative, finite, rtol=3e-8, atol=1e-12)


@pytest.mark.parametrize("species,size", [("Fe II", 80), ("C I", 15)])
def test_collision_temperature_gradient_is_finite(species, size):
    atom = load_metal_atom(species)
    function = lambda log_t: electron_collision_coefficients(atom, jnp.exp(log_t))
    derivative = jax.jacfwd(function)(jnp.log(5481.75))
    assert np.all(np.isfinite(derivative))


def test_radiative_centers_keep_chianti_wavelengths_separate_from_level_energies():
    atom = load_metal_atom("Mg II")
    expected = np.float32(1e8 / float(np.float32(2803.531)))
    np.testing.assert_equal(float(atom.radiative_wavenumber[1, 0]), float(expected))
    assert float(atom.radiative_wavenumber[1, 0]) != float(
        atom.energy_wavenumber[1] - atom.energy_wavenumber[0]
    )
    iron = load_metal_atom("Fe II")
    np.testing.assert_array_equal(
        iron.radiative_wavenumber,
        np.asarray(iron.energy_wavenumber[:,None]-iron.energy_wavenumber[None,:]).astype(np.float32).astype(float),
    )


@pytest.mark.parametrize("levels", [371, 716])
def test_large_iron_atom_conserves_particles_and_recovers_collision_lte(levels):
    atom = load_metal_atom("Fe II", levels=levels)
    assert len(atom.weight) == levels
    assert np.all(np.asarray(atom.upper) < levels)
    temperature, abundance = 6500.0, 1e8
    rate = 1e10 * electron_collision_coefficients(atom, temperature)
    solution = populations(rate, abundance)
    expected = atom.weight * jnp.exp(-PLANCK * C * atom.energy_wavenumber / (KB * temperature))
    expected *= abundance / expected.sum()
    np.testing.assert_allclose(solution, expected, rtol=2e-11, atol=1e-30)
    np.testing.assert_allclose(solution.sum(), abundance, rtol=3e-15)
    assert np.isfinite(float(kinetic_line_heating(solution, rate, atom.energy_wavenumber)))


def test_stiff_rate_derivatives_match_independent_three_state_tree_formula():
    rate = jnp.array([[0., 1e-12, 1e6], [1e-14, 0., 1e-20], [1e9, 1e-20, 0.]])
    direction = jnp.asarray(np.random.default_rng(1).normal(size=(3, 3)))
    weight = jnp.array([.17, 1.17, 2.17])

    def exact(parameter):
        a = rate * jnp.exp(parameter * direction)
        tree = jnp.array([
            a[1, 0]*a[2, 0] + a[1, 0]*a[2, 1] + a[1, 2]*a[2, 0],
            a[0, 1]*a[2, 1] + a[0, 1]*a[2, 0] + a[0, 2]*a[2, 1],
            a[0, 2]*a[1, 2] + a[0, 1]*a[1, 2] + a[0, 2]*a[1, 0],
        ])
        return jnp.exp(.2 * parameter) * tree / tree.sum()

    def solve(parameter):
        return populations(
            rate * jnp.exp(parameter * direction), jnp.exp(.2 * parameter)
        )
    value, tangent = jax.jvp(solve, (0.,), (1.,))
    expected, expected_tangent = jax.jvp(exact, (0.,), (1.,))
    reverse = jax.jit(jax.grad(lambda parameter: jnp.dot(solve(parameter), weight)))(0.)
    np.testing.assert_allclose(value, expected, rtol=2e-14)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-13)
    np.testing.assert_allclose(reverse, expected_tangent @ weight, rtol=2e-13)


def test_adas_placeholders_are_removed_but_physical_weak_lines_are_retained():
    atom = load_metal_atom("Fe II", levels=716)
    rates = np.asarray(atom.spontaneous)
    # C23's STOUT parser excludes its documented 1e-30 ADAS placeholders.
    assert np.count_nonzero(rates) == 168725
    assert not np.any((rates > 0) & (rates < 1e-29))
    # Real source transitions far below ordinary weak-line cutoffs remain.
    assert np.any((rates > 0) & (rates < 1e-25))
