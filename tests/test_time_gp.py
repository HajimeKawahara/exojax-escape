"""Independent dense references for the small sufficient-statistic GP system."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag, null_space

from escape.time_gp import (
    prepare_time_gp,
    time_gp_loglike,
    time_gp_posterior,
    time_gp_predict,
    time_gp_statistics,
)

jax.config.update("jax_enable_x64", True)


def _example():
    time = np.array([0.0, 1.2, 3.1, 5.0, 8.0])
    weight = np.array(
        [
            [100.0, 25.0, 10.0, 0.0],
            [120.0, 0.0, 0.0, 0.0],
            [80.0, 30.0, 0.0, 0.0],
            [90.0, 40.0, 0.0, 0.0],
            [110.0, 20.0, 0.0, 0.0],
        ]
    )
    basis = np.array([[1.0, 1.0, 1.0, 1.0], [-0.8, 0.7, 0.0, 1.0]])
    data = np.array(
        [
            [0.2, -0.1, 9.0, np.nan],
            [0.3, np.nan, np.nan, np.nan],
            [0.4, -0.2, np.nan, np.nan],
            [0.1, 0.2, np.nan, np.nan],
            [-0.2, 0.1, np.nan, np.nan],
        ]
    )
    return data, weight, basis, time


def _dense(data, weight, basis, time, amplitudes, lengths, scale):
    """Explicit orthonormal contrasts; intentionally no core preparation use."""
    exposure_count, pixels = weight.shape
    modes = len(basis)
    weight = np.where(((weight > 0).sum(0) >= 2)[None], weight, 0.0)
    rows = []
    for pixel in range(pixels):
        retained = np.flatnonzero(weight[:, pixel] > 0)
        if len(retained) < 2:
            continue
        null = null_space(np.sqrt(weight[retained, pixel])[None])
        for column in null.T:
            row = np.zeros(exposure_count * pixels)
            row[retained * pixels + pixel] = column
            rows.append(row)
    contrast = np.array(rows)
    whitened = np.sqrt(weight) * np.where(weight > 0, data, 0.0)
    target = contrast @ whitened.ravel()
    design = np.zeros((exposure_count * pixels, modes * exposure_count))
    for exposure in range(exposure_count):
        for pixel in range(pixels):
            for mode in range(modes):
                design[exposure * pixels + pixel, mode * exposure_count + exposure] = (
                    np.sqrt(weight[exposure, pixel]) * basis[mode, pixel]
                )
    design = contrast @ design
    matrices = []
    for amplitude, length in zip(amplitudes, np.broadcast_to(lengths, (modes,))):
        x = np.sqrt(3.0) * abs(time[:, None] - time[None]) / length
        matrices.append(
            amplitude**2 * ((1 + x) * np.exp(-x) + 1e-10 * np.eye(exposure_count))
        )
    prior = block_diag(*matrices)
    covariance = scale**2 * np.eye(len(target)) + design @ prior @ design.T
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    inverse_target = np.linalg.solve(covariance, target)
    ll = -0.5 * (target @ inverse_target + logdet + len(target) * np.log(2 * np.pi))
    mean = prior @ design.T @ inverse_target
    posterior = prior - prior @ design.T @ np.linalg.solve(covariance, design @ prior)
    return ll, mean.reshape(modes, exposure_count), posterior, design, target


def test_marginal_likelihood_and_posterior_match_dense_missing_sample_reference():
    data, weight, basis, time = _example()
    state = prepare_time_gp(weight, basis)
    amplitudes, lengths, scale = np.array([0.11, 0.05]), np.array([2.4, 4.0]), 0.8
    expected = _dense(data, weight, basis, time, amplitudes, lengths, scale)
    actual = time_gp_loglike(data, state, time, amplitudes, lengths, scale)
    posterior = time_gp_posterior(data, state, time, amplitudes, lengths, scale)
    np.testing.assert_allclose(actual, expected[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(posterior.mean, expected[1], rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(
        posterior.covariance, expected[2], rtol=2e-12, atol=2e-14
    )
    np.testing.assert_allclose(state.gram, expected[3].T @ expected[3], atol=2e-13)
    norm, product = time_gp_statistics(data, state)
    np.testing.assert_allclose(norm, expected[4] @ expected[4], atol=2e-13)
    np.testing.assert_allclose(product, expected[3].T @ expected[4], atol=2e-13)
    assert int(state.rank) == 7


def test_arbitrary_common_master_cancels_in_likelihood_and_posterior():
    data, weight, basis, time = _example()
    state = prepare_time_gp(weight, basis)
    common = np.array([2.1, -1.4, 9.0, 0.0])
    args = (state, time, np.array([0.1, 0.2]), 3.0, 1.1)
    np.testing.assert_allclose(
        time_gp_loglike(data, *args), time_gp_loglike(data + common, *args), atol=2e-13
    )
    np.testing.assert_allclose(
        time_gp_posterior(data, *args).mean,
        time_gp_posterior(data + common, *args).mean,
        atol=2e-14,
    )


def test_zero_gp_retains_white_scale_normalization_and_supported_rank():
    data, weight, basis, time = _example()
    state = prepare_time_gp(weight, basis)
    norm, _ = time_gp_statistics(data, state)
    scale = 1.7
    expected = -0.5 * (norm / scale**2 + int(state.rank) * np.log(2 * np.pi * scale**2))
    actual = time_gp_loglike(data, state, time, np.zeros(2), 3.0, scale)
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    derivative = jax.grad(
        lambda log_s: time_gp_loglike(
            data, state, time, jnp.zeros(2), 3.0, jnp.exp(log_s)
        )
    )(np.log(scale))
    np.testing.assert_allclose(
        derivative, norm / scale**2 - int(state.rank), atol=2e-13
    )
    posterior = time_gp_posterior(data, state, time, np.zeros(2), 3.0, scale)
    np.testing.assert_array_equal(posterior.mean, np.zeros((2, 5)))
    np.testing.assert_array_equal(posterior.covariance, np.zeros((10, 10)))


def test_jit_gradients_include_signal_gp_and_white_parameters():
    data, weight, basis, time = _example()
    state = prepare_time_gp(weight, basis)
    signal = np.sin(time[:, None] / 3.0) * np.array([0.1, -0.2, 0.0, 0.0])[None]

    def objective(parameters):
        return time_gp_loglike(
            data - parameters[0] * signal,
            state,
            time,
            jnp.exp(parameters[1:3]),
            jnp.exp(parameters[3]),
            jnp.exp(parameters[4]),
        )

    point = np.array([0.7, -2.1, -2.8, 1.2, -0.15])
    compiled = jax.jit(jax.value_and_grad(objective))
    value, derivative = compiled(point)
    assert np.isfinite(value)
    step = 1e-5
    finite = np.array(
        [
            (objective(point + step * x) - objective(point - step * x)) / (2 * step)
            for x in np.eye(5)
        ]
    )
    np.testing.assert_allclose(derivative, finite, rtol=3e-8, atol=3e-8)


def test_no_temporal_information_returns_prior_without_invented_samples():
    state = prepare_time_gp([[1.0, 0.0], [0.0, 2.0]], [[1.0, 1.0]])
    data = np.array([[5.0, np.nan], [np.nan, 3.0]])
    time = np.array([0.0, 2.0])
    assert int(state.rank) == 0
    assert float(time_gp_loglike(data, state, time, np.array([0.3]), 2.0, 0.9)) == 0.0
    posterior = time_gp_posterior(data, state, time, np.array([0.3]), 2.0, 0.9)
    np.testing.assert_array_equal(posterior.mean, np.zeros((1, 2)))
    x = np.sqrt(3.0) * abs(time[:, None] - time[None]) / 2
    np.testing.assert_allclose(
        posterior.covariance, 0.3**2 * ((1 + x) * np.exp(-x) + 1e-10 * np.eye(2))
    )


def test_invalid_inputs_are_not_silently_treated_as_measurements():
    with pytest.raises(ValueError):
        prepare_time_gp([[1.0, -1.0], [2.0, 3.0]], [[1.0, 1.0]])
    with pytest.raises(ValueError):
        prepare_time_gp([[1.0, 2.0], [2.0, 3.0]], [[np.nan, 1.0]])
    with pytest.raises(ValueError):
        prepare_time_gp(np.ones((3, 2)), np.ones((2, 3)))
    state = prepare_time_gp(np.ones((3, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        time_gp_loglike(np.ones((3, 1)), state, np.arange(3.0), np.ones(1), 2.0, 1.0)
    with pytest.raises(ValueError):
        time_gp_loglike(np.ones((3, 2)), state, np.arange(2.0), np.ones(1), 2.0, 1.0)
    assert np.isnan(
        time_gp_loglike(np.ones((3, 2)), state, np.arange(3.0), np.ones(1), 2.0, -1.0)
    )
    bad = np.ones((3, 2))
    bad[0, 0] = np.nan
    assert np.isnan(time_gp_loglike(bad, state, np.arange(3.0), np.ones(1), 2.0, 1.0))


def test_new_time_prediction_matches_dense_conditional_and_zero_amplitude():
    data, weight, basis, time = _example()
    state = prepare_time_gp(weight, basis)
    prediction = np.array([-2.0, 0.0, 2.0, 6.0, 14.0])
    amplitudes, lengths, scale = np.array([0.11, 0.05]), np.array([2.4, 4.0]), 0.8
    _, _, _, design, target = _dense(
        data, weight, basis, time, amplitudes, lengths, scale
    )
    train, cross, prior = [], [], []
    for amplitude, length in zip(amplitudes, lengths):

        def kernel(left, right):
            x = np.sqrt(3.0) * abs(left[:, None] - right[None]) / length
            return amplitude**2 * (1 + x) * np.exp(-x)

        train.append(kernel(time, time) + amplitude**2 * 1e-10 * np.eye(len(time)))
        cross.append(kernel(prediction, time))
        prior.append(kernel(prediction, prediction))
    training, cross, prior = map(
        lambda blocks: block_diag(*blocks), (train, cross, prior)
    )
    observed = scale**2 * np.eye(len(target)) + design @ training @ design.T
    expected_mean = cross @ design.T @ np.linalg.solve(observed, target)
    expected_covariance = prior - cross @ design.T @ np.linalg.solve(
        observed, design @ cross.T
    )
    actual = time_gp_predict(data, state, time, amplitudes, lengths, scale, prediction)
    np.testing.assert_allclose(actual.mean.ravel(), expected_mean, atol=3e-14)
    np.testing.assert_allclose(actual.covariance, expected_covariance, atol=3e-14)
    zero = time_gp_predict(data, state, time, np.zeros(2), lengths, scale, prediction)
    np.testing.assert_array_equal(zero.mean, np.zeros((2, len(prediction))))
    np.testing.assert_array_equal(
        zero.covariance, np.zeros((2 * len(prediction), 2 * len(prediction)))
    )
