"""Independent contrasts and common-master cancellation under declared noise."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares

from escape.time_contrast import profile_time_residuals

jax.config.update("jax_enable_x64", True)


def test_profiled_objective_matches_independent_whitened_contrasts():
    rng = np.random.default_rng(91)
    data, model = rng.normal(size=(2, 7, 4))
    sigma = rng.uniform(0.1, 2.0, size=(7, 4))
    weight = 1 / sigma**2
    weight[2, 1] = 0.0
    actual = profile_time_residuals(data, model, weight)
    expected = 0.0
    for pixel in range(4):
        keep = weight[:, pixel] > 0
        scale = np.sqrt(weight[keep, pixel])
        orthogonal, _ = np.linalg.qr(
            np.column_stack((scale, np.eye(len(scale)))), mode="complete"
        )
        contrast = orthogonal[:, 1:].T @ (scale * (data - model)[keep, pixel])
        expected += contrast @ contrast
        assert actual.rank[pixel] == len(scale) - 1
    np.testing.assert_allclose(np.sum(actual.residual**2), expected, rtol=3e-15)


def test_unknown_shared_master_cancels_even_for_selected_science_exposures():
    grid = np.linspace(-4.0, 4.0, 51)
    flux = 1 + 0.04 * np.exp(-0.5 * (grid[None, :] - np.arange(9)[:, None] / 3) ** 2)
    shared = 0.85 + 0.04 * np.cos(grid)
    ratio = flux / shared
    # Original master membership may be larger than the selected science set.
    take = [0, 2, 3, 5]
    weights = np.ones((len(take), len(grid)))
    result = profile_time_residuals(np.log(ratio[take]), np.log(flux[take]), weights)
    np.testing.assert_allclose(result.residual, 0.0, atol=2e-16)
    np.testing.assert_allclose(result.common_log_offset, -np.log(shared), atol=2e-16)
    assert int(result.rank.sum()) == (len(take) - 1) * len(grid)


def test_missing_singleton_and_invalid_samples_are_distinguished():
    data = np.array([[1.0, np.nan, 3.0, 1.0], [2.0, 2.0, 4.0, np.nan]])
    weight = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    result = profile_time_residuals(data, np.zeros_like(data), weight)
    np.testing.assert_array_equal(result.rank, [1, 0, 0, 0])
    np.testing.assert_array_equal(result.valid, [True, False, False, False])
    np.testing.assert_array_equal(result.residual[:, 1:3], 0.0)
    assert np.isnan(result.residual[:, 3]).all()
    bad = profile_time_residuals(np.ones((2, 1)), np.zeros((2, 1)), -np.ones((2, 1)))
    assert np.isnan(bad.residual).all()
    with pytest.raises(ValueError, match="matching"):
        profile_time_residuals(np.ones(3), np.ones(3), np.ones(3))


def test_common_mode_is_removed_but_time_signal_and_derivatives_remain():
    phase = jnp.linspace(0.0, 1.0, 8)[:, None]
    profile = jnp.exp(-(jnp.linspace(-2.0, 2.0, 13)[None, :] ** 2))
    data = 0.3 * phase * profile
    weights = jnp.broadcast_to(jnp.linspace(1.0, 3.0, 8)[:, None], data.shape)

    def objective(parameters):
        model = parameters[0] * phase * profile + parameters[1] * profile
        result = profile_time_residuals(data, model, weights)
        return jnp.sum(result.residual**2)

    point = np.array([0.1, 0.7])
    gradient = np.asarray(jax.grad(objective)(point))
    step = 1e-5
    finite = np.array(
        [
            (objective(point + step * axis) - objective(point - step * axis))
            / (2 * step)
            for axis in np.eye(2)
        ]
    )
    np.testing.assert_allclose(gradient, finite, atol=2e-11, rtol=1e-9)
    assert abs(gradient[0]) > 0.1
    assert abs(gradient[1]) < 1e-13


def test_phase_parameters_recover_with_an_unknown_structured_master():
    velocity = jnp.linspace(-8.0, 8.0, 65)
    orbital = jnp.linspace(-4.0, 4.0, 12)
    phase_bin = jnp.repeat(jnp.arange(2), 6)
    truth = np.array([0.35, -0.7, 0.025, 0.04])

    def log_flux(parameters):
        center = orbital + parameters[:2][phase_bin]
        amplitude = parameters[2:][phase_bin]
        return jnp.log1p(
            amplitude[:, None]
            * jnp.exp(-0.5 * (velocity[None, :] - center[:, None]) ** 2)
        )

    # A deep, wavelength-dependent positive master changes absolute line levels.
    master = 0.7 + 0.2 * jnp.exp(-(velocity**2) / 9) + 0.05 * jnp.cos(velocity)
    data = log_flux(truth) - jnp.log(master)
    weight = jnp.broadcast_to(jnp.linspace(2.0, 3.0, 12)[:, None], data.shape)

    @jax.jit
    def residual(parameters):
        return profile_time_residuals(
            data, log_flux(parameters), weight
        ).residual.ravel()

    result = least_squares(
        residual,
        [0.1, -0.1, 0.02, 0.02],
        jac=jax.jit(jax.jacfwd(residual)),
        gtol=1e-12,
        xtol=1e-12,
        ftol=1e-12,
    )
    np.testing.assert_allclose(result.x, truth, atol=2e-8)
    assert np.sum(result.fun**2) < 1e-20


def test_exposure_normalization_and_variable_stellar_lines_do_not_cancel():
    time = np.linspace(-1.0, 1.0, 9)[:, None]
    velocity = np.linspace(-3.0, 3.0, 31)[None, :]
    planet = np.log1p(0.01 * np.exp(-((velocity - 2 * time) ** 2)))
    shared_master = 0.8 + 0.05 * np.cos(velocity)
    base = planet - np.log(shared_master)
    grey = 0.001 * time
    stellar = 0.002 * time * np.exp(-(velocity**2))
    weight = np.ones_like(planet)
    clean = profile_time_residuals(base, planet, weight).residual
    grey_result = profile_time_residuals(base + grey, planet, weight).residual
    line_result = profile_time_residuals(base + stellar, planet, weight).residual
    np.testing.assert_allclose(clean, 0.0, atol=1e-16)
    np.testing.assert_allclose(
        grey_result, np.broadcast_to(grey, planet.shape), atol=1e-16
    )
    np.testing.assert_allclose(line_result, stellar, atol=1e-16)
    assert np.sum(grey_result**2) > 0
    assert np.sum(line_result**2) > 0
