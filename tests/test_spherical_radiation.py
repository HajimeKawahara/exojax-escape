"""Independent finite-sphere limits, reciprocity and signed-extinction AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from escape.spherical_radiation import (
    solve_spherical_transfer,
    spherical_shell_geometry,
)

jax.config.update("jax_enable_x64", True)


def test_transparent_core_angular_dilution_and_luminosity():
    edges = np.array([1., 1.02, 1.3, 3.])
    geometry = spherical_shell_geometry(edges)
    result = solve_spherical_transfer(
        geometry, np.zeros((3, 1)), np.zeros((3, 1)), np.array([2.]), np.array([.3]),
    )
    visible = 1 - np.diff((edges**2 - edges[0]**2)**1.5) / np.diff(edges**3)
    expected = .3 + (2. - .3) * visible / 2
    assert result.converged
    np.testing.assert_allclose(result.mean_intensity[:, 0], expected, rtol=2e-11)
    np.testing.assert_allclose(edges**2 * result.flux_moment_edges[:, 0],
                               edges[0]**2 * (2. - .3) / 4, rtol=2e-12)


@pytest.mark.parametrize("opacity", [0., 1e-8, 1., 1e8])
def test_uniform_thermal_field_at_any_depth(opacity):
    geometry = spherical_shell_geometry([1., 1.02, 1.3, 3.])
    chi = np.full((3, 1), opacity)
    result = solve_spherical_transfer(geometry, chi, chi * 2., np.array([2.]), np.array([2.]))
    assert result.converged
    np.testing.assert_allclose(result.mean_intensity, 2., rtol=2e-11)
    np.testing.assert_allclose(result.flux_moment_edges, 0., atol=1e-14)


@pytest.mark.parametrize("opacity", [-.2, .3, 20.])
def test_homogeneous_full_sphere_against_independent_chord_integrals(opacity):
    radius, boundary, emission = 1.7, .3, .8
    geometry = spherical_shell_geometry([0., radius])
    result = solve_spherical_transfer(
        geometry, np.array([[opacity]]), np.array([[emission]]),
        np.array([0.]), np.array([boundary]),
    )

    def mean_on_chord(mu):
        length = 2 * radius * mu
        attenuation_average = -np.expm1(-opacity * length) / (opacity * length)
        return boundary * attenuation_average + emission / opacity * (1 - attenuation_average)

    def outgoing(mu):
        depth = 2 * radius * mu * opacity
        return boundary * np.exp(-depth) - emission / opacity * np.expm1(-depth)

    mean = quad(lambda mu: 3 * mu**2 * mean_on_chord(mu), 0, 1,
                epsabs=1e-12, epsrel=1e-12)[0]
    flux = quad(lambda mu: mu * (outgoing(mu) - boundary) / 2, 0, 1,
                epsabs=1e-12, epsrel=1e-12)[0]
    assert result.converged
    np.testing.assert_allclose(result.mean_intensity[0, 0], mean, rtol=2e-11)
    np.testing.assert_allclose(result.flux_moment_edges[-1, 0], flux, rtol=2e-11)
    np.testing.assert_allclose(radius**2 * result.flux_moment_edges[-1, 0],
                               radius**3 / 3 * (emission - opacity * mean), rtol=2e-11)


def test_transparent_emission_and_analytic_opacity_derivative():
    radius, boundary, emission = 1.7, .3, .8
    geometry = spherical_shell_geometry([0., radius])

    def intensity(opacity):
        return solve_spherical_transfer(
            geometry, jnp.reshape(opacity, (1, 1)), jnp.array([[emission]]),
            jnp.zeros(1), jnp.array([boundary]),
        ).mean_intensity[0, 0]

    np.testing.assert_allclose(intensity(0.), boundary + 3 * emission * radius / 4, rtol=1e-13)
    exact = -3 * boundary * radius / 4 - 2 * emission * radius**2 / 5
    np.testing.assert_allclose(jax.grad(intensity)(0.), exact, rtol=1e-12)
    for step in (1e-4, 1e-5):
        np.testing.assert_allclose((intensity(step) - intensity(-step)) / (2 * step),
                                   exact, rtol=1e-8)


def test_green_operator_reciprocity_with_thick_and_inverted_shells():
    geometry = spherical_shell_geometry([1., 1.05, 1.3, 3.])
    chi = jnp.array([[1e3], [-.002], [3.]])
    zero = jnp.zeros(1)

    def response(source):
        return solve_spherical_transfer(geometry, chi, source[:, None], zero, zero).mean_intensity[:, 0]

    green = jax.vmap(response)(jnp.eye(3)).T
    assert np.isfinite(green).all() and (np.asarray(green) >= 0).all()
    weighted = np.asarray(geometry.volume_cm3)[:, None] * green
    np.testing.assert_allclose(weighted, weighted.T, rtol=2e-11, atol=1e-16)


def test_geometry_error_is_exposed_and_requires_refinement():
    coarse = spherical_shell_geometry([1., 1.02, 1.3, 3.], angle_order=1)
    fine = spherical_shell_geometry([1., 1.02, 1.3, 3.], angle_order=32)
    assert coarse.volume_relative_error > 1e-3
    assert fine.volume_relative_error < 1e-10
    result = solve_spherical_transfer(coarse, np.zeros((3, 1)), np.zeros((3, 1)),
                                      np.ones(1), np.ones(1))
    assert not result.converged
    assert np.isnan(result.mean_intensity).all()


def test_overflowing_gain_is_not_silently_clipped():
    geometry = spherical_shell_geometry([0., 1.])
    result = solve_spherical_transfer(geometry, np.array([[-1000.]]), np.ones((1, 1)),
                                      np.zeros(1), np.ones(1))
    assert result.valid and not result.converged
    assert np.isnan(result.mean_intensity).all()


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError, match="edges"):
        spherical_shell_geometry([2., 1.])
    with pytest.raises(ValueError, match="angle_order"):
        spherical_shell_geometry([1., 2.], angle_order=0)
