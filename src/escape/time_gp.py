"""Marginal time GPs after removing an unknown common native log spectrum.

The likelihood is measured in fixed, whitened, orthonormal time contrasts.
Only constants independent of every fitted parameter are omitted. Each GP
coefficient multiplies a supplied spectral basis, with independent Matern-3/2
priors. No dense covariance over all exposure/pixel samples is constructed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np


class TimeGPState(NamedTuple):
    """Fixed weights and the projected design Gram matrix, in mode/time order."""

    weight: jax.Array
    basis: jax.Array
    total_weight: jax.Array
    gram: jax.Array
    rank: jax.Array


class TimeGPPosterior(NamedTuple):
    """Latent coefficients; absolute common levels remain prior-dependent."""

    mean: jax.Array
    covariance: jax.Array


def prepare_time_gp(weight, basis):
    """Prepare one night/order outside JIT; arrays have shapes (E,P), (M,P).

    Zero weights exclude samples. Columns with fewer than two retained
    exposures carry no temporal information and are dropped. The basis and
    weights must be finite, and weights must be nonnegative. The supported
    contrast rank is sum_p(E_valid,p - 1), not the number of input samples.
    """
    weight, basis = (np.asarray(value, dtype=float) for value in (weight, basis))
    if weight.ndim != 2 or basis.ndim != 2 or basis.shape[1] != weight.shape[1]:
        raise ValueError("Use weights (exposure,pixel) and basis (mode,pixel).")
    if not weight.shape[0] or not weight.shape[1] or not basis.shape[0]:
        raise ValueError("Weights and basis must have nonempty axes.")
    if (
        not np.isfinite(weight).all()
        or np.any(weight < 0)
        or not np.isfinite(basis).all()
    ):
        raise ValueError("Use finite nonnegative weights and a finite basis.")
    count = (weight > 0).sum(axis=0)
    weight = np.where(count[None, :] >= 2, weight, 0.0)
    total = weight.sum(axis=0)
    inverse_total = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
    weighted_basis = basis[:, None, :] * weight[None, :, :]
    common = np.einsum(
        "mep,nfp,p->menf", weighted_basis, weighted_basis, inverse_total, optimize=True
    )
    diagonal = np.einsum("ep,mp,np->mne", weight, basis, basis, optimize=True)
    gram = np.einsum("mne,ef->menf", diagonal, np.eye(weight.shape[0])) - common
    size = basis.shape[0] * weight.shape[0]
    gram = gram.reshape(size, size)
    gram = (gram + gram.T) / 2
    return TimeGPState(
        jnp.asarray(weight),
        jnp.asarray(basis),
        jnp.asarray(total),
        jnp.asarray(gram),
        jnp.asarray(np.maximum(count - 1, 0).sum()),
    )


def time_gp_statistics(log_residual, state):
    """Return projected residual norm and design product without a large matrix."""
    data = jnp.asarray(log_residual)
    if data.shape != state.weight.shape:
        raise ValueError("Residuals must match the prepared exposure/pixel weights.")
    retained = state.weight > 0
    safe_data = jnp.where(retained, data, 0.0)
    common = (state.weight * safe_data).sum(axis=0) / jnp.where(
        state.total_weight > 0, state.total_weight, 1.0
    )
    centered = jnp.where(retained, safe_data - common[None, :], 0.0)
    norm = jnp.sum(state.weight * centered**2)
    product = jnp.einsum("ep,ep,mp->me", state.weight, centered, state.basis).ravel()
    return norm, product


def _components(
    log_residual, state, time_minutes, amplitudes, length_scale_minutes, white_scale
):
    time = jnp.asarray(time_minutes)
    amplitudes = jnp.asarray(amplitudes)
    modes, exposures = state.basis.shape[0], state.weight.shape[0]
    if time.shape != (exposures,) or amplitudes.shape != (modes,):
        raise ValueError("Use one time per exposure and one amplitude per mode.")
    length = jnp.broadcast_to(jnp.asarray(length_scale_minutes), (modes,))
    scale = jnp.asarray(white_scale)
    if scale.ndim != 0:
        raise ValueError("Use one white-noise scale for this night/order.")
    distance = jnp.sqrt(3.0) * jnp.abs(time[:, None] - time[None, :])
    scaled = distance[None, :, :] / length[:, None, None]
    correlation = (1 + scaled) * jnp.exp(-scaled)
    # Fixed relative regularization, scaled by amplitude squared; zero GP is exact.
    correlation = correlation + 1e-10 * jnp.eye(exposures)[None, :, :]
    factors = amplitudes[:, None, None] * jnp.linalg.cholesky(correlation)
    factor = jnp.einsum("mef,mn->menf", factors, jnp.eye(modes)).reshape(
        modes * exposures, modes * exposures
    )
    norm, product = time_gp_statistics(log_residual, state)
    system = jnp.eye(modes * exposures) + factor.T @ state.gram @ factor / scale**2
    cholesky = jnp.linalg.cholesky((system + system.T) / 2)
    rhs = factor.T @ product / scale**2
    solved = solve_triangular(cholesky, rhs, lower=True)
    invalid = (
        jnp.any(~jnp.isfinite(time))
        | jnp.any(~jnp.isfinite(amplitudes) | (amplitudes < 0))
        | jnp.any(~jnp.isfinite(length) | (length <= 0))
        | ~jnp.isfinite(scale)
        | (scale <= 0)
    )
    return norm, factor, cholesky, solved, scale, invalid


def time_gp_loglike(
    log_residual, state, time_minutes, amplitudes, length_scale_minutes, white_scale
):
    """Marginal log likelihood, including GP and white-scale determinants.

    The state defines a fixed whitened contrast measure. Parameter-independent
    constants involving the original supplied errors are omitted. Do not use
    differences across changing sample masks as absolute predictive densities
    without their measure constants. Gaussian GP coefficients are integrated
    out, while the arbitrary common spectrum is removed by the same contrast
    operation on both the mean and covariance.
    """
    norm, _, chol, solved, scale, invalid = _components(
        log_residual, state, time_minutes, amplitudes, length_scale_minutes, white_scale
    )
    quadratic = norm / scale**2 - jnp.dot(solved, solved)
    logdet = 2 * jnp.log(jnp.diag(chol)).sum() + 2 * state.rank * jnp.log(scale)
    value = -0.5 * (quadratic + logdet + state.rank * jnp.log(2 * jnp.pi))
    return jnp.where(invalid, jnp.nan, value)


def time_gp_posterior(
    log_residual, state, time_minutes, amplitudes, length_scale_minutes, white_scale
):
    """Conditional GP coefficients at input times, in mode-major ordering.

    Mean has shape (M,E), covariance (M*E,M*E). A time-constant GP component
    cannot be separated from the removed common spectrum. Plot projected
    corrections or differences, not this unidentifiable absolute level.
    """
    _, factor, chol, solved, _, invalid = _components(
        log_residual, state, time_minutes, amplitudes, length_scale_minutes, white_scale
    )
    posterior_factor = solve_triangular(chol, factor.T, lower=True).T
    mean = (posterior_factor @ solved).reshape(
        state.basis.shape[0], state.weight.shape[0]
    )
    covariance = posterior_factor @ posterior_factor.T
    return TimeGPPosterior(
        jnp.where(invalid, jnp.nan, mean), jnp.where(invalid, jnp.nan, covariance)
    )


def time_gp_predict(
    log_residual,
    state,
    time_minutes,
    amplitudes,
    length_scale_minutes,
    white_scale,
    prediction_time_minutes,
):
    """Predict smooth GP coefficients at new times, retaining their covariance.

    The tiny input-correlation regularizer is training-only; predictions refer
    to the smooth Matern process. Thus predictions at training times differ
    from ``time_gp_posterior`` only at that regularization scale. Absolute
    common levels remain unidentified; retain cross-covariances when plotting
    differences relative to a reference interval. Zero amplitudes are allowed.
    """
    _, _, chol, solved, _, invalid = _components(
        log_residual, state, time_minutes, amplitudes, length_scale_minutes, white_scale
    )
    time, prediction = map(jnp.asarray, (time_minutes, prediction_time_minutes))
    if prediction.ndim != 1 or not prediction.shape[0]:
        raise ValueError("Use a nonempty vector of prediction times.")
    modes, exposures = state.basis.shape[0], state.weight.shape[0]
    amplitudes = jnp.asarray(amplitudes)
    length = jnp.broadcast_to(jnp.asarray(length_scale_minutes), (modes,))

    def correlation(left, right):
        x = jnp.sqrt(3.0) * jnp.abs(left[:, None] - right[None, :])
        x = x[None, :, :] / length[:, None, None]
        return (1 + x) * jnp.exp(-x)

    training_factor = jnp.linalg.cholesky(
        correlation(time, time) + 1e-10 * jnp.eye(exposures)[None]
    )
    cross = (
        jax.vmap(lambda lower, value: solve_triangular(lower, value.T, lower=True).T)(
            training_factor, correlation(prediction, time)
        )
        * amplitudes[:, None, None]
    )
    count = len(prediction)
    cross = jnp.einsum("mte,mn->mtne", cross, jnp.eye(modes)).reshape(
        modes * count, modes * exposures
    )
    prior = amplitudes[:, None, None] ** 2 * correlation(prediction, prediction)
    prior = jnp.einsum("mtu,mn->mtnu", prior, jnp.eye(modes)).reshape(
        modes * count, modes * count
    )
    posterior_factor = solve_triangular(chol, cross.T, lower=True).T
    mean = (posterior_factor @ solved).reshape(modes, count)
    covariance = prior - cross @ cross.T + posterior_factor @ posterior_factor.T
    invalid = invalid | jnp.any(~jnp.isfinite(prediction))
    return TimeGPPosterior(
        jnp.where(invalid, jnp.nan, mean),
        jnp.where(invalid, jnp.nan, (covariance + covariance.T) / 2),
    )


__all__ = [
    "TimeGPState",
    "TimeGPPosterior",
    "prepare_time_gp",
    "time_gp_statistics",
    "time_gp_loglike",
    "time_gp_posterior",
    "time_gp_predict",
]
