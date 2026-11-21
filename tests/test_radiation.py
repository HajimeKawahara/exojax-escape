"""Independent transport limits and exact finite-volume energy accounting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from escape.radiation import moment_geometry, solve_moment_transfer

jax.config.update("jax_enable_x64", True)


def test_uniform_blackbody_is_stationary_for_any_optical_depth():
    edges = np.linspace(1., 2., 11)
    chi = np.broadcast_to(np.array([0., 1e-5, 1., 1e5]), (10, 4))
    planck = np.array([.1, 1., 3., 2.])
    answer = solve_moment_transfer(edges, chi, chi * planck, planck, planck)
    assert answer.converged
    np.testing.assert_allclose(answer.mean_intensity, np.broadcast_to(planck, chi.shape), rtol=1e-9)
    np.testing.assert_allclose(answer.flux_moment_edges, 0, atol=1e-13)


def test_transparent_sphere_conserves_luminosity_without_opacity_floor():
    edges = np.geomspace(1e10, 3e10, 9)
    zero = np.zeros((8, 1))
    result = solve_moment_transfer(edges, zero, zero, jnp.array([.1]), jnp.array([2.]))
    expected_j = (edges[0]**2 * .1 + edges[-1]**2 * 2) / (edges[0]**2 + edges[-1]**2)
    assert result.converged
    np.testing.assert_allclose(result.mean_intensity, expected_j, rtol=1e-13)
    luminosity = edges**2 * np.asarray(result.flux_moment_edges)[:, 0]
    np.testing.assert_allclose(luminosity, luminosity[0], rtol=1e-13)


def test_slab_refinement_converges_to_analytic_two_stream_absorption():
    errors = []
    for count in (16, 32, 64):
        edges = np.linspace(0, 1, count + 1)
        centers = (edges[1:] + edges[:-1]) / 2
        result = solve_moment_transfer(edges, np.ones((count, 1)), np.zeros((count, 1)),
                                       jnp.ones(1), jnp.zeros(1), spherical=False)
        assert result.converged
        exact = .5 * np.exp(-np.sqrt(3) * centers)
        errors.append(np.max(np.abs(np.asarray(result.mean_intensity)[:, 0] - exact)))
    assert errors[2] < errors[1] / 3 < errors[0] / 9
    assert errors[-1] < 6e-5


def test_spherical_radiation_energy_budget_and_negative_control():
    edges = np.geomspace(1e10, 2e10, 13)
    chi = np.linspace(.1, 10., 24).reshape(12, 2) / 1e10
    eta = chi * np.array([2., 3.])
    result = solve_moment_transfer(edges, chi, eta, jnp.array([1., 2.]), jnp.array([.2, .3]))
    area, volume = moment_geometry(edges)
    radiative_heating = 4 * np.pi * np.sum(volume[:, None] * (chi * result.mean_intensity - eta), axis=0)
    flux_change = 4 * np.pi * (area[-1] * result.flux_moment_edges[-1] - area[0] * result.flux_moment_edges[0])
    np.testing.assert_allclose(radiative_heating + flux_change, 0, atol=1e-12 * np.max(np.abs(flux_change)))
    # Omitting emission is an intentional negative control for the budget gate.
    wrong_heating = 4 * np.pi * np.sum(volume[:, None] * chi * result.mean_intensity, axis=0)
    assert np.max(np.abs(wrong_heating + flux_change)) > np.max(np.abs(flux_change))


def test_transparent_emission_and_implicit_gradient():
    edges = jnp.linspace(1., 2., 9)
    def emergent(scale):
        result = solve_moment_transfer(edges, jnp.full((8, 1), scale),
                                       jnp.full((8, 1), .1), jnp.ones(1), jnp.zeros(1), spherical=False)
        return result.flux_moment_edges[-1, 0]
    assert np.isfinite(emergent(0.))
    step = 1e-4
    finite = (emergent(1. + step) - emergent(1. - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(emergent)(1.), finite, rtol=1e-7)


def test_excessive_gain_is_an_explicit_failure():
    result = solve_moment_transfer(jnp.array([1., 2.]), -jnp.ones((1, 1)),
                                   jnp.zeros((1, 1)), jnp.ones(1), jnp.zeros(1))
    assert not result.valid and not result.converged
    assert np.isnan(result.mean_intensity).all()


def test_weak_gain_matches_independent_formal_solution_and_refines():
    for emissivity in (0., .2):
        errors = []
        for count in (16, 32, 64):
            edges = np.linspace(0., 1., count + 1)
            centers = (edges[1:] + edges[:-1]) / 2
            opacity = -.1
            solution = solve_moment_transfer(
                edges, np.full((count, 1), opacity), np.full((count, 1), emissivity),
                jnp.array([1.]), jnp.array([.3]), spherical=False,
            )
            assert solution.converged
            source = emissivity / opacity
            plus = np.exp(-np.sqrt(3) * opacity * centers)
            plus = plus + source * (1 - plus)
            attenuation = np.exp(-np.sqrt(3) * opacity * (1 - centers))
            minus = .3 * attenuation + source * (1 - attenuation)
            exact = (plus + minus) / 2
            errors.append(np.max(np.abs(np.asarray(solution.mean_intensity)[:, 0] - exact)))
        assert errors[2] < errors[1] / 3 < errors[0] / 9
        assert errors[2] < 1e-5


def test_weak_gain_derivative_matches_finite_difference():
    def observable(gain):
        result = solve_moment_transfer(jnp.linspace(0., 1., 9), -gain * jnp.ones((8, 1)),
                                       jnp.full((8, 1), .2), jnp.ones(1), jnp.zeros(1), spherical=False)
        return result.flux_moment_edges[-1, 0]
    step = 1e-5
    finite = (observable(.1 + step) - observable(.1 - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(observable)(.1), finite, rtol=1e-8)


def test_wide_outgoing_vacuum_detects_invalid_angular_closure():
    result = solve_moment_transfer(jnp.linspace(1., 3., 9), jnp.zeros((8, 1)),
                                   jnp.zeros((8, 1)), jnp.array([2.]), jnp.array([.1]))
    assert result.valid  # Inputs are allowed; the angular closure fails here.
    assert not result.converged
    assert result.realizability_margin < -.1
