"""Native-pixel sampling and grouping after the full nightly observation model.

Use ``process_phase_spectra(...).stellar_residual`` before selecting phase or
night groups. Selecting those groups must not redefine the nightly masters.
Archived residual data are already processed and need only the grouping step.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class PixelCoadd(NamedTuple):
    """Weighted residuals; weights alone do not describe shared-master covariance."""

    contrast: jax.Array
    weight: jax.Array
    valid: jax.Array


@jax.jit
def sample_stellar_residuals(
    stellar_velocity_kms, stellar_residual, pixel_exposure_index,
    pixel_velocity_kms, *, stellar_keep=None,
):
    """Sample modeled stellar-frame residuals at selected native-pixel coordinates.

    The increasing velocity grid has shape (S,), residuals have shape (E, S),
    and integer exposure indices and pixel velocities have shape (N,). An
    optional boolean mask has shape (E, S). Interpolation requires finite,
    retained neighbors with nonzero interpolation weight; an exact grid point
    needs only that point. Unsupported coordinates or exposure indices return
    NaN, as do invalid grids. This is point sampling, not pixel integration.

    All exposures must already have contributed to their declared nightly
    masters. These indices select samples from that completed calculation.
    Derivatives are piecewise smooth for fixed interpolation cells and masks.
    """
    grid, residual, exposure, query = map(
        jnp.asarray,
        (stellar_velocity_kms, stellar_residual, pixel_exposure_index, pixel_velocity_kms),
    )
    if (grid.ndim != 1 or grid.size < 2 or residual.ndim != 2
            or residual.shape[0] < 1 or residual.shape[1] != grid.size):
        raise ValueError("Use a velocity grid (S,) and residuals (E, S), with S >= 2.")
    if (exposure.ndim != 1 or query.shape != exposure.shape
            or exposure.dtype.kind not in "iu"):
        raise ValueError("Pixel exposure indices must be integer (N,); velocities need (N,).")
    keep = jnp.ones(residual.shape, dtype=bool) if stellar_keep is None else jnp.asarray(stellar_keep)
    if keep.shape != residual.shape or keep.dtype.kind != "b":
        raise ValueError("stellar_keep must be boolean with shape (E, S).")

    valid_grid = jnp.all(jnp.isfinite(grid)) & jnp.all(jnp.diff(grid) > 0)
    supported = (jnp.isfinite(query) & (query >= grid[0]) & (query <= grid[-1])
                 & (exposure >= 0) & (exposure < residual.shape[0]) & valid_grid)
    safe_query = jnp.where(jnp.isfinite(query), query, grid[0])
    left = jnp.clip(jnp.searchsorted(grid, safe_query, side="right") - 1, 0, grid.size - 2)
    right = left + 1
    width = grid[right] - grid[left]
    fraction = (safe_query - grid[left]) / jnp.where(width > 0, width, 1.)
    row = jnp.clip(exposure, 0, residual.shape[0] - 1)
    available = jnp.isfinite(residual) & keep
    supported &= (available[row, left] | (fraction == 1.))
    supported &= (available[row, right] | (fraction == 0.))
    values = jnp.where(available, residual, 0.)
    sampled = (1. - fraction) * values[row, left] + fraction * values[row, right]
    return jnp.where(supported, sampled, jnp.nan)


@partial(jax.jit, static_argnames=("num_bins",))
def coadd_residual_pixels(residual, pixel_bin_index, pixel_weight, *, num_bins):
    """Apply identical supplied pixel groups and weights to data or model samples.

    All inputs have shape (N,); integer bin labels refer to a flattened set of
    phase/night/wavelength groups. Out-of-range labels and nonfinite residuals
    are excluded. Empty bins return zero weight and NaN contrast. Negative or
    nonfinite weights invalidate the whole result, rather than being hidden.
    Weights are supplied, not estimated: inverse-variance weights do not account
    for correlations from a shared master or spectral resampling.
    """
    values, group, weights = map(jnp.asarray, (residual, pixel_bin_index, pixel_weight))
    if (values.ndim != 1 or group.shape != values.shape or weights.shape != values.shape
            or group.dtype.kind not in "iu" or num_bins < 1):
        raise ValueError("Use residuals, integer bin labels and weights with shape (N), and num_bins >= 1.")
    valid_weights = jnp.all(jnp.isfinite(weights) & (weights >= 0))
    selected = (jnp.isfinite(values) & (group >= 0) & (group < num_bins)
                & jnp.isfinite(weights) & (weights >= 0) & valid_weights)
    effective = jnp.where(selected, weights, 0.)
    safe_values = jnp.where(selected, values, 0.)
    safe_group = jnp.clip(group, 0, num_bins - 1)
    total = jnp.zeros(num_bins, dtype=effective.dtype).at[safe_group].add(effective)
    numerator = jnp.zeros(num_bins, dtype=(effective * safe_values).dtype).at[safe_group].add(
        effective * safe_values
    )
    valid = total > 0
    contrast = numerator / jnp.where(valid, total, 1.)
    return PixelCoadd(jnp.where(valid, contrast, jnp.nan), total, valid)


__all__ = ["PixelCoadd", "sample_stellar_residuals", "coadd_residual_pixels"]
