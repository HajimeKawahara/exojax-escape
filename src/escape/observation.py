"""Explicit phase sampling, nightly division and masked spectral coaddition.

The ordering follows the observation audit of Zhang et al. (2026),
https://arxiv.org/abs/2606.23899v1, and the public exposure products at
https://doi.org/10.5281/zenodo.20707255. Doppler sampling follows the earlier
NumPy reconstruction experiment: lambda_stellar = lambda_planet * (1 + v/c).
That experiment did not recover the publication's master weights or exact
coaddition. This module applies supplied arrays; it does not recover a pipeline.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

C_KMS = 299792.458


class ObservationResult(NamedTuple):
    """Dimensionless residuals and the effective supplied coaddition weights."""

    contrast: jax.Array
    weight: jax.Array
    valid: jax.Array
    exposure_contrast: jax.Array
    normalized_stellar_flux: jax.Array
    nightly_master: jax.Array
    stellar_residual: jax.Array


@jax.jit
def process_phase_spectra(
    model_phase, model_velocity_kms, model_contrast, exposure_phase,
    orbital_velocity_kms, night_index, stellar_velocity_kms, output_velocity_kms,
    *, stellar_reference, continuum_weights, master_weights, stellar_keep,
    coadd_weights,
):
    """Apply a declared observation operator with differentiable interpolation.

    ``model_contrast`` has shape (P, V): planet line flux divided by the stellar
    continuum, not by the stellar absorption-line profile. ``stellar_reference``
    is normalized to that same continuum, with shape (S,) or (E, S). Spectral
    coordinates are v=c*(lambda/lambda0-1), positive to the red. All three
    velocity grids and the unwrapped model phases must be strictly ascending.
    ``orbital_velocity_kms`` is the planet-to-stellar shift for each exposure;
    no orbital, systemic or barycentric velocity is inferred here.

    Exposure phases, velocities and integer night labels have shape (E,).
    ``continuum_weights``, ``master_weights`` and boolean ``stellar_keep`` have
    shape (E, S). Each spectrum is divided by its continuum-weighted mean;
    its master is the pixel-weighted mean of all normalized exposures with the
    same night label. The supplied stellar mask is applied after this division.
    ``coadd_weights`` has shape (E, O), on the output planet grid. Zero an entire
    row to exclude a phase from the coadd while retaining it in the master.

    No cadence, noise model, master weighting or phase selection is assumed.
    Supply continuum/reference arrays appropriate to the declared experiment.
    Interpolation is linear in phase and wavelength coordinate. Unsupported
    phase or spectral samples are excluded, never extrapolated; explicitly pad
    a line-only model with zeros if that assumption is justified. Both adjacent
    stellar samples must be retained when interpolating across a mask edge.
    No contributing samples gives zero weight and NaN contrast. Invalid grids
    or negative/nonfinite weights yield an invalid result; shape errors raise.

    This operator omits exposure-time integration and wavelength-bin integration.
    The derivatives are piecewise smooth for fixed masks and interpolation cells.
    """
    phase, velocity, model = map(jnp.asarray, (model_phase, model_velocity_kms, model_contrast))
    exposures, shifts, nights = map(jnp.asarray, (exposure_phase, orbital_velocity_kms, night_index))
    stellar, output = map(jnp.asarray, (stellar_velocity_kms, output_velocity_kms))
    reference = jnp.asarray(stellar_reference)
    normalization, masters, weights = map(jnp.asarray, (continuum_weights, master_weights, coadd_weights))
    keep = jnp.asarray(stellar_keep)
    if (phase.ndim != 1 or phase.size < 2 or velocity.ndim != 1 or velocity.size < 2
            or exposures.ndim != 1 or exposures.size < 1 or stellar.ndim != 1 or stellar.size < 2
            or output.ndim != 1 or output.size < 1 or model.shape != (phase.size, velocity.size)):
        raise ValueError("Use ascending one-dimensional grids and model_contrast with shape (P, V).")
    count, size = exposures.size, stellar.size
    if shifts.shape != (count,) or nights.shape != (count,) or nights.dtype.kind not in "iu":
        raise ValueError("Exposure velocities and integer night labels must have shape (E,).")
    if reference.shape not in ((size,), (count, size)):
        raise ValueError("stellar_reference must have shape (S,) or (E, S).")
    if (normalization.shape != (count, size) or masters.shape != (count, size)
            or keep.shape != (count, size) or keep.dtype.kind != "b"
            or weights.shape != (count, output.size)):
        raise ValueError("Continuum/master weights and boolean mask need (E, S); coadd weights need (E, O).")

    valid = jnp.asarray(True)
    for grid in (phase, velocity, stellar, output):
        valid &= jnp.all(jnp.isfinite(grid)) & jnp.all(jnp.diff(grid) > 0)
    for weight in (normalization, masters, weights):
        valid &= jnp.all(jnp.isfinite(weight) & (weight >= 0))
    valid &= jnp.all(jnp.isfinite(model))
    valid &= jnp.all(jnp.isfinite(exposures)) & jnp.all(jnp.isfinite(shifts) & (shifts > -C_KMS))

    sampled = jax.vmap(lambda column: jnp.interp(exposures, phase, column),
                       in_axes=1, out_axes=1)(model)
    rest_query = C_KMS * ((1 + stellar[None, :] / C_KMS) / (1 + shifts[:, None] / C_KMS) - 1)
    planet = jax.vmap(lambda query, row: jnp.interp(query, velocity, row, left=0., right=0.))(
        rest_query, sampled)
    covered = ((exposures >= phase[0]) & (exposures <= phase[-1]))[:, None]
    covered &= (rest_query >= velocity[0]) & (rest_query <= velocity[-1])
    flux = reference + planet
    covered &= jnp.isfinite(flux) & (flux > 0) & jnp.isfinite(reference) & (reference > 0)
    norm_weights = jnp.where(covered, normalization, 0.)
    norm_sum = norm_weights.sum(axis=1)
    norm = (norm_weights * jnp.where(covered, flux, 0.)).sum(axis=1) / jnp.where(norm_sum > 0, norm_sum, 1.)
    normalized = jnp.where(covered, flux, 0.) / jnp.where(norm > 0, norm, 1.)[:, None]
    covered &= (norm_sum > 0)[:, None] & (norm > 0)[:, None] & valid

    same_night = (nights[:, None] == nights[None, :]).astype(normalized.dtype)
    master_weight = jnp.where(covered, masters, 0.)
    master_sum = same_night @ master_weight
    master = (same_night @ (master_weight * normalized)) / jnp.where(master_sum > 0, master_sum, 1.)
    available = covered & (master_sum > 0) & (master > 0)
    residual = normalized / jnp.where(master > 0, master, 1.) - 1
    usable = available & keep
    target = C_KMS * ((1 + output[None, :] / C_KMS) * (1 + shifts[:, None] / C_KMS) - 1)
    shifted = jax.vmap(lambda query, row: jnp.interp(query, stellar, row, left=0., right=0.))(
        target, jnp.where(usable, residual, 0.))
    coverage = jax.vmap(lambda query, row: jnp.interp(query, stellar, row, left=0., right=0.))(
        target, usable.astype(normalized.dtype))
    usable_output = coverage >= 1 - 1e-12
    effective = jnp.where(usable_output, weights, 0.)
    total = effective.sum(axis=0)
    coadd = (effective * shifted).sum(axis=0) / jnp.where(total > 0, total, 1.)
    return ObservationResult(
        jnp.where(total > 0, coadd, jnp.nan), total, total > 0,
        jnp.where(usable_output, shifted, jnp.nan),
        jnp.where(covered, normalized, jnp.nan), jnp.where(master_sum > 0, master, jnp.nan),
        jnp.where(available, residual, jnp.nan),
    )


__all__ = ["ObservationResult", "process_phase_spectra"]
