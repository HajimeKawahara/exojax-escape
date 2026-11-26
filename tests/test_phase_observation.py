"""Native-pixel tests independent of the observed atmosphere or fitted parameters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.observation import process_phase_spectra
from escape.phase_observation import coadd_residual_pixels, sample_stellar_residuals

jax.config.update("jax_enable_x64", True)


def test_native_sampling_matches_linear_numpy_reference_and_masks():
    grid = np.array([-5., -1., 2., 6.])
    residual = np.array([[2., 4., 8., 16.], [-3., 1., 2., 5.]])
    exposure = np.array([0, 1, 0, 1, 0, 0, 1, -1, 2])
    query = np.array([-3., 0., 6., -5., -6., np.nan, 3., 0., 0.])
    expected = np.array([np.interp(q, grid, residual[e]) if 0 <= e < 2 and -5 <= q <= 6
                         else np.nan for e, q in zip(exposure, query)])
    actual = sample_stellar_residuals(grid, residual, exposure, query)
    np.testing.assert_allclose(actual, expected, atol=1e-15, equal_nan=True)
    keep = np.ones(residual.shape, dtype=bool)
    keep[0, 2] = False
    # An exact retained node remains valid; interpolation across the hole does not.
    masked = sample_stellar_residuals(grid, residual, np.array([0, 0, 0]),
                                     np.array([-1., 0., 2.]), stellar_keep=keep)
    np.testing.assert_allclose(masked, [4., np.nan, np.nan], equal_nan=True)
    residual[0, 2] = np.nan
    np.testing.assert_allclose(
        sample_stellar_residuals(grid, residual, np.array([0, 0]), np.array([-1., 0.])),
        [4., np.nan], equal_nan=True,
    )


def test_weighted_groups_match_numpy_and_expose_missing_samples():
    values = np.array([1., 3., -2., np.nan, 8., 20., 30.])
    group = np.array([0, 0, 2, 2, 2, -1, 4])
    weights = np.array([1., 3., 2., 100., 0., 1., 1.])
    result = coadd_residual_pixels(values, group, weights, num_bins=4)
    means, totals = [], []
    for label in range(4):
        take = (group == label) & np.isfinite(values) & (weights > 0)
        totals.append(weights[take].sum())
        means.append(np.average(values[take], weights=weights[take]) if take.any() else np.nan)
    np.testing.assert_allclose(result.contrast, means, equal_nan=True)
    np.testing.assert_allclose(result.weight, totals)
    np.testing.assert_array_equal(result.valid, np.array(totals) > 0)
    empty = coadd_residual_pixels(np.array([]), np.array([], dtype=int), np.array([]), num_bins=2)
    assert not empty.valid.any() and np.isnan(empty.contrast).all()


def test_phase_selection_keeps_full_nightly_master():
    velocity = jnp.linspace(-300., 300., 601)
    stellar = jnp.linspace(-180., 180., 361)
    phases = jnp.arange(3, dtype=float)
    line = .003 * jnp.exp(-.5 * (velocity / 8.) ** 2)
    args = dict(
        model_phase=phases, model_velocity_kms=velocity,
        model_contrast=jnp.stack([line, line, line]), exposure_phase=phases,
        orbital_velocity_kms=jnp.array([-40., 0., 40.]), night_index=jnp.zeros(3, dtype=int),
        stellar_velocity_kms=stellar, output_velocity_kms=jnp.array([0.]),
        stellar_reference=jnp.ones(stellar.size),
        continuum_weights=jnp.broadcast_to((abs(stellar) > 150.).astype(float), (3, stellar.size)),
        master_weights=jnp.ones((3, stellar.size)),
        stellar_keep=jnp.ones((3, stellar.size), dtype=bool), coadd_weights=jnp.ones((3, 1)),
    )
    full = process_phase_spectra(**args)
    selected = process_phase_spectra(**{**args, "coadd_weights": jnp.array([[1.], [0.], [0.]])})
    np.testing.assert_array_equal(selected.nightly_master, full.nightly_master)
    sample = sample_stellar_residuals(stellar, full.stellar_residual, jnp.array([0]), jnp.array([-40.]))
    coadd = coadd_residual_pixels(sample, jnp.array([0]), jnp.array([1.]), num_bins=1)
    np.testing.assert_allclose(coadd.contrast, selected.contrast, rtol=1e-12)
    # Incorrectly rebuilding the master from only the selected exposure erases it.
    wrong = process_phase_spectra(**{**args, "master_weights": args["master_weights"].at[1:].set(0.)})
    assert float(coadd.contrast[0]) > .0019
    np.testing.assert_allclose(wrong.stellar_residual[0], 0., atol=1e-15)


def test_sampling_and_grouping_gradients_match_finite_differences():
    grid = jnp.array([-3., -1., 1., 4.])
    exposure = jnp.array([0, 1, 0, 1])
    bins = jnp.array([0, 0, 1, 1])
    weights = jnp.array([1., 2., 3., 1.])

    def observable(point):
        amplitude, offset = point
        residual = amplitude * jnp.stack([jnp.exp(-grid ** 2 / 4.), .7 * jnp.exp(-grid ** 2 / 4.)])
        query = jnp.array([-.3, .4, 1.5, 2.3]) + offset
        samples = sample_stellar_residuals(grid, residual, exposure, query)
        result = coadd_residual_pixels(samples, bins, weights, num_bins=2)
        return jnp.dot(result.contrast, jnp.array([1., -.4]))

    point = np.array([.002, .03])
    step = 1e-6
    finite = np.array([(observable(point + step * axis) - observable(point - step * axis)) / (2 * step)
                       for axis in np.eye(2)])
    automatic = np.asarray(jax.grad(observable)(point))
    assert np.isfinite(automatic).all() and (abs(automatic) > 1e-6).all()
    np.testing.assert_allclose(automatic, finite, rtol=1e-8, atol=1e-12)


def test_invalid_inputs_are_not_silently_accepted():
    grid = jnp.array([0., 1.])
    with pytest.raises(ValueError, match="integer"):
        sample_stellar_residuals(grid, jnp.ones((1, 2)), jnp.array([0.]), jnp.array([.5]))
    with pytest.raises(ValueError, match="boolean"):
        sample_stellar_residuals(grid, jnp.ones((1, 2)), jnp.array([0]), jnp.array([.5]),
                                stellar_keep=jnp.ones((1, 2)))
    assert np.isnan(sample_stellar_residuals(grid[::-1], jnp.ones((1, 2)),
                                           jnp.array([0]), jnp.array([.5]))).all()
    for weights in [jnp.array([-1., 1.]), jnp.array([jnp.nan, 1.]), jnp.array([jnp.inf, 1.])]:
        result = coadd_residual_pixels(jnp.array([1., 2.]), jnp.array([0, 1]), weights, num_bins=2)
        assert not result.valid.any() and np.isnan(result.contrast).all()
    with pytest.raises(ValueError, match="integer"):
        coadd_residual_pixels(jnp.array([1.]), jnp.array([0.]), jnp.array([1.]), num_bins=2)
