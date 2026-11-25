"""Profile a shared log-spectrum before fitting within-night time variation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TimeContrast(NamedTuple):
    """Whitened projected residuals, fitted common term and supported ranks."""

    residual: jax.Array
    common_log_offset: jax.Array
    rank: jax.Array
    valid: jax.Array


@jax.jit
def profile_time_residuals(log_data, log_model, inverse_variance):
    """Remove one common log-spectrum value per wavelength within one night.

    Arrays have shape (exposure, native wavelength). The objective
    ``sum(result.residual**2)`` equals weighted least squares after profiling
    an arbitrary time-constant offset at each wavelength. For E retained
    exposures the residual projector has rank E-1, not E independent samples.
    No pairwise differences are counted as independent observations.

    This is a diagonal log-error model supplied by the caller, not a recovered
    covariance of a reduced archive. Weights must be finite and nonnegative;
    zero weights exclude samples. Nonfinite data/model at positive weight or
    invalid weights produce NaN residuals in that wavelength column. Columns
    with fewer than two retained exposures have zero residual and rank.
    """
    data, model, weight = map(jnp.asarray, (log_data, log_model, inverse_variance))
    if data.ndim != 2 or model.shape != data.shape or weight.shape != data.shape:
        raise ValueError("Use matching (exposure, wavelength) arrays.")
    bad = jnp.any(
        ~jnp.isfinite(weight)
        | (weight < 0)
        | ((weight > 0) & (~jnp.isfinite(data) | ~jnp.isfinite(model))),
        axis=0,
    )
    retained = jnp.isfinite(weight) & (weight > 0)
    safe_weight = jnp.where(retained, weight, 0.0)
    difference = jnp.where(
        retained & jnp.isfinite(data) & jnp.isfinite(model), data - model, 0.0
    )
    total = safe_weight.sum(axis=0)
    mean = (safe_weight * difference).sum(axis=0) / jnp.where(total > 0, total, 1.0)
    count = retained.sum(axis=0)
    residual = jnp.sqrt(safe_weight) * (difference - mean[None, :])
    residual = jnp.where(count[None, :] >= 2, residual, 0.0)
    residual = jnp.where(bad[None, :], jnp.nan, residual)
    rank = jnp.where(bad, 0, jnp.maximum(count - 1, 0))
    common = jnp.where((total > 0) & ~bad, mean, jnp.nan)
    return TimeContrast(residual, common, rank, (count >= 2) & ~bad)


__all__ = ["TimeContrast", "profile_time_residuals"]
