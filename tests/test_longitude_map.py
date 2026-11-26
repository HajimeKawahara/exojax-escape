"""Independent geometry, source-transfer and differentiable map checks."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.longitude_map import (
    chord_segments,
    longitude_spectrum,
    longitude_weights,
    rebin_longitude_kernel,
    source_longitude_kernel,
)

jax.config.update("jax_enable_x64", True)


def test_map_normalization_and_gradient():
    nodes, weights = np.polynomial.legendre.leggauss(128)
    longitude = nodes * np.pi / 2
    for a, b in [(0.0, 0.0), (-3.0, 3.0), (3.0, -3.0), (2.0, 1.0)]:
        value = np.asarray(longitude_weights(longitude, a, b))
        np.testing.assert_allclose(np.sum(value * weights) / 2, 1, atol=2e-14)
        assert np.min(value) > 0

    def f(p):
        return longitude_weights(jnp.array([-0.4, 0.7]), p[0], p[1])

    point = np.array([0.3, -0.6])
    analytic = jax.jacfwd(f)(point)
    numeric = np.column_stack(
        [
            (np.asarray(f(point + 1e-5 * e)) - np.asarray(f(point - 1e-5 * e))) / 2e-5
            for e in np.eye(2)
        ]
    )
    np.testing.assert_allclose(analytic, numeric, rtol=2e-9, atol=1e-10)


def test_uniform_and_nested_partition():
    nodes = np.linspace(-np.pi / 2, np.pi / 2, 97)
    kernel = np.random.default_rng(3).normal(size=(97, 21))
    coarse = rebin_longitude_kernel(kernel, nodes, nodes[::2])
    np.testing.assert_allclose(kernel.sum(0), coarse.sum(0), atol=1e-13)
    np.testing.assert_allclose(
        longitude_spectrum(kernel, nodes, 0.0, 0.0), kernel.sum(0), atol=1e-13
    )


def test_local_longitude_weights_move_line_centroid():
    velocity = np.linspace(-30, 30, 101)
    nodes = np.array([-np.pi / 2, 0.0, np.pi / 2])
    kernel = np.exp(
        -0.5 * ((velocity[None] - np.array([-12, 0, 12])[:, None]) / 3) ** 2
    )

    def centroid(b):
        spectrum = np.asarray(longitude_spectrum(kernel, nodes, 0.0, b))
        return np.sum(velocity * spectrum) / spectrum.sum()

    assert centroid(1.0) > 5
    np.testing.assert_allclose(centroid(1.0), -centroid(-1.0), atol=1e-13)


def test_uniform_transparent_shell_matches_volume_integral():
    # At full dayside view only the near hemisphere emits; the opaque base
    # blocks no additional dayside volume. Constant eta integrates to volume.
    grid = dict(nlon=40, nlat=20, nrad_per_interval=12, naz=64, subsegments=2)
    edges = np.array([1.0, 1.3])

    def coefficient(cell, velocity):
        return np.zeros(1), np.ones(1)

    kernel = source_longitude_kernel(
        0.5,
        edges,
        np.linspace(-np.pi / 2, np.pi / 2, 49),
        coefficient,
        np.zeros(1),
        [0.0],
        1.0,
        0.0,
        0.0,
        grid,
    )
    analytic = 2 * np.pi / 3 * (edges[-1] ** 3 - 1)
    np.testing.assert_allclose(kernel.sum(), analytic, rtol=2e-3)


def test_absorbing_boundary_subtraction_and_dark_segments():
    grid = dict(nlon=80, nlat=40, nrad_per_interval=4, naz=32, subsegments=2)

    def coefficient(cell, velocity):
        return np.full(1, 1e6), np.zeros(1)

    kernel = source_longitude_kernel(
        0.5,
        [1.0, 1.3],
        np.linspace(-np.pi / 2, np.pi / 2, 49),
        coefficient,
        np.ones(1),
        [0.0],
        1.0,
        0.0,
        0.0,
        grid,
    )
    np.testing.assert_allclose(kernel.sum(), -np.pi, rtol=2e-4)
    assert (
        chord_segments(
            np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), [1.0, 1.3]
        )
        == []
    )
