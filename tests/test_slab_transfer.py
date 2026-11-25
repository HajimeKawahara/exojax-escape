"""Formal internal radiation: limits, photon energy, and implicit inputs."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.radiation import solve_slab_transfer

jax.config.update("jax_enable_x64", True)


def test_thermal_radiation_is_stationary_across_thin_and_thick_cells():
    chi = jnp.array([[0., 1e-10, 2.], [0., 2e-5, 1e9]])
    intensity = jnp.array([1., 2., 3.])
    result = solve_slab_transfer(jnp.array([0., 1., 3.]), chi, chi * intensity,
                                 intensity, intensity)
    assert result.converged
    np.testing.assert_allclose(result.mean_intensity, np.broadcast_to(intensity, (2, 3)), rtol=2e-14)
    np.testing.assert_allclose(result.flux_moment_edges, 0., atol=2e-15)


def test_flux_loss_matches_absorption_for_illuminated_inhomogeneous_slab():
    edges = jnp.array([0., .1, 1., 10.])
    chi = jnp.array([[.01], [1.], [100.]])
    emissivity = chi * jnp.array([[1.], [3.], [2.]])
    result = solve_slab_transfer(edges, chi, emissivity, jnp.array([4.]), jnp.array([.2]), angle_order=8)
    assert result.converged
    assert result.balance_error < 1e-13
    np.testing.assert_allclose(jnp.diff(result.flux_moment_edges, axis=0),
                               jnp.diff(edges)[:, None] * (emissivity - chi * result.mean_intensity),
                               rtol=1e-12, atol=1e-12)


def test_transparent_emission_and_derivative_are_finite():
    def observable(opacity):
        return solve_slab_transfer(jnp.array([0., 1.]), jnp.full((1, 1), opacity),
                                   jnp.ones((1, 1)), jnp.zeros(1), jnp.zeros(1)).mean_intensity[0, 0]
    assert np.isfinite(observable(0.))
    derivative = jax.grad(observable)(0.)
    # One-sided derivative stays inside the nonnegative-opacity domain.
    step = 1e-6
    finite = (-3 * observable(0.) + 4 * observable(step) - observable(2 * step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite, rtol=2e-8)


def test_unsupported_gain_is_visible():
    answer = solve_slab_transfer(jnp.array([0., 1.]), -jnp.ones((1, 1)),
                                 jnp.ones((1, 1)), jnp.zeros(1), jnp.zeros(1))
    assert not answer.converged
    assert np.isnan(answer.mean_intensity).all()


def test_direct_beam_has_correct_mean_and_normal_flux():
    edges = jnp.array([0., 1., 3.])
    chi = jnp.array([[.2], [.3]])
    incident = jnp.array([2.])
    mu = .7
    result = solve_slab_transfer(edges, chi, jnp.zeros_like(chi), jnp.zeros(1), jnp.zeros(1),
                                 beam_mean_intensity=incident, beam_mu=mu)
    assert result.converged
    depths = np.array([.2, .6]) / mu
    expected_faces = 2 * np.exp(-np.r_[depths.sum(), depths[1], 0.])
    np.testing.assert_allclose(result.flux_moment_edges[:, 0], -mu * expected_faces, rtol=2e-14)
    np.testing.assert_allclose(np.diff(result.flux_moment_edges[:, 0]),
                               -np.diff(edges) * chi[:, 0] * result.mean_intensity[:, 0], rtol=2e-14)


def test_beam_retains_overlying_column_next_to_extremely_thick_cell():
    edges = jnp.array([0., 1., 2.])
    bottom_depth = 1e16
    zeros = jnp.zeros(1)

    def solve(top_depth):
        return solve_slab_transfer(
            edges, jnp.array([[bottom_depth], [top_depth]]), jnp.zeros((2, 1)),
            zeros, zeros, beam_mean_intensity=jnp.ones(1),
        )

    result = solve(1.)
    assert result.converged
    expected = jnp.array([[jnp.exp(-1.) / bottom_depth], [1 - jnp.exp(-1.)]])
    np.testing.assert_allclose(result.mean_intensity, expected, rtol=2e-14)
    assert result.balance_error < 1e-14
    # Scaling the small lower-cell mean checks the derivative of its finite
    # overlying optical column, rather than a numerically negligible signal.
    derivative = jax.grad(lambda top: solve(top).mean_intensity[0, 0] * bottom_depth)(1.)
    np.testing.assert_allclose(derivative, -jnp.exp(-1.), rtol=2e-14)
