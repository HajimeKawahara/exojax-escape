"""Independent GLS check of held-out GP gains with unknown pixel intercepts."""

import jax
import numpy as np
from scipy.linalg import block_diag

from escape.time_gp import prepare_time_gp, time_gp_loglike

jax.config.update("jax_enable_x64", True)


def dense_flat_intercept_loglike(data, weight, basis, time, amplitudes, length, scale):
    """Integrate pixel intercepts against the same flat prior in raw coordinates."""
    exposures, pixels = data.shape
    design = np.zeros((exposures * pixels, len(basis) * exposures))
    for mode, row in enumerate(basis):
        for exposure in range(exposures):
            design[
                exposure * pixels : (exposure + 1) * pixels, mode * exposures + exposure
            ] = row
    distance = np.sqrt(3) * abs(time[:, None] - time[None, :]) / length
    correlation = (1 + distance) * np.exp(-distance) + 1e-10 * np.eye(exposures)
    prior = block_diag(*(amplitude**2 * correlation for amplitude in amplitudes))
    covariance = np.diag(scale**2 / weight.ravel()) + design @ prior @ design.T
    intercept = np.tile(np.eye(pixels), (exposures, 1))
    precision_data = np.linalg.solve(covariance, data.ravel())
    precision_intercept = np.linalg.solve(covariance, intercept)
    intercept_information = intercept.T @ precision_intercept
    product = intercept.T @ precision_data
    quadratic = data.ravel() @ precision_data - product @ np.linalg.solve(
        intercept_information, product
    )
    logdet = (
        np.linalg.slogdet(covariance)[1] + np.linalg.slogdet(intercept_information)[1]
    )
    return -0.5 * (quadratic + logdet + (data.size - pixels) * np.log(2 * np.pi))


def test_heldout_gp_gain_retains_intercept_uncertainty():
    rng = np.random.default_rng(71)
    time = np.array([0.0, 1.0, 2.5, 5.0, 7.0, 8.0])
    weight = rng.uniform(30, 100, (6, 4))
    basis = np.array([np.ones(4), np.linspace(-1, 1, 4)])
    data = rng.normal(0, 0.1, weight.shape) + np.array([0.5, -0.2, 0.1, 0.8])
    train = np.array([True, True, False, False, True, True])
    parameters = [(np.zeros(2), 2.0, 1.2), (np.array([0.04, 0.03]), 3.0, 0.9)]
    conditional = {"dense": [], "core": []}
    for amplitudes, length, scale in parameters:
        values = {"dense": [], "core": []}
        for keep in (np.ones(6, dtype=bool), train):
            values["dense"].append(
                dense_flat_intercept_loglike(
                    data[keep],
                    weight[keep],
                    basis,
                    time[keep],
                    amplitudes,
                    length,
                    scale,
                )
            )
            state = prepare_time_gp(weight[keep], basis)
            values["core"].append(
                float(
                    time_gp_loglike(
                        data[keep], state, time[keep], amplitudes, length, scale
                    )
                )
            )
        for name, pair in values.items():
            conditional[name].append(pair[0] - pair[1])
    # Each individual score has a different fixed coordinate normalization.
    # The difference between candidate models cancels that normalization.
    np.testing.assert_allclose(
        np.diff(conditional["core"]), np.diff(conditional["dense"]), rtol=0, atol=2e-12
    )
