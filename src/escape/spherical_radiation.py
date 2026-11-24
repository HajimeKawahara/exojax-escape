"""Finite-chord formal transfer through static, homogeneous spherical shells.

Isotropic boundary intensities preserve spherical symmetry. This is not a
one-sided stellar irradiation model or a moving-medium transfer calculation.
Signed extinction is integrated on finite paths without an opacity floor.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .radiation import RadiationSolution, _cell_weights


class SphericalShellGeometry(NamedTuple):
    """Fixed geometry; numerical quadrature does not redefine shell volume."""

    path_cm: jax.Array
    hits_core: jax.Array
    mean_weights: jax.Array
    flux_weights: jax.Array
    area_cm2: jax.Array
    volume_cm3: jax.Array
    volume_relative_error: jax.Array


class SphericalScatteringSolution(NamedTuple):
    radiation: RadiationSolution
    linear_residual_norm: jax.Array
    emission_mean_intensity: jax.Array
    outer_boundary_response: jax.Array


def _energy_balance_error(geometry, opacity, emission, mean, flux, inner, outer):
    """Sum absolute cell defects relative to supplied power, per frequency.

    A cell-local relative ratio is undefined once an attenuated response
    underflows. Use gross incident/emitted power (including finite gain),
    without mixing frequencies or claiming accuracy of faint populations.
    """
    volume = geometry.volume_cm3[:, None]
    luminosity = geometry.area_cm2[:, None] * flux
    defect = jnp.diff(luminosity, axis=0) - volume * (emission - opacity * mean)
    supplied = (jnp.sum(volume * (emission + jnp.maximum(-opacity, 0) * mean), axis=0)
                + (geometry.area_cm2[0] * inner + geometry.area_cm2[-1] * outer) / 4)
    return jnp.max(jnp.sum(jnp.abs(defect), axis=0)
                   / jnp.where(supplied > 0, supplied, 1.))


def spherical_shell_geometry(edges_cm, *, angle_order=32):
    """Prepare impact-parameter panels split at every shell edge, on the host.

    On each panel [a,b], p^2=b^2-(b^2-a^2)t^2 and p dp=(b^2-a^2)t dt.
    The tangent square root is then smooth. ``angle_order`` is per panel.
    Geometry is static; derivatives below act on coefficients and boundaries.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 for radiative transfer.")
    edges = np.asarray(edges_cm, dtype=float)
    if (edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or not np.all(np.diff(edges) > 0)):
        raise ValueError("Use nonnegative, finite, increasing shell edges.")
    if (not isinstance(angle_order, int) or isinstance(angle_order, bool)
            or angle_order < 1):
        raise ValueError("angle_order must be a positive static integer.")
    panels = np.unique(np.r_[0., edges])
    nodes, weights = np.polynomial.legendre.leggauss(angle_order)
    t, weights = (nodes + 1) / 2, weights / 2
    delta = np.diff(panels) * (panels[1:] + panels[:-1])
    impact_squared = (panels[1:, None]**2 - delta[:, None] * t**2).ravel()
    impact_weights = (delta[:, None] * t * weights).ravel()
    z = np.sqrt(np.maximum(edges[None, :]**2 - impact_squared[:, None], 0.))
    # Rationalized subtraction retains thin-shell chord lengths.
    shell_area_difference = np.diff(edges) * (edges[1:] + edges[:-1])
    summed = z[:, 1:] + z[:, :-1]
    path = np.where(z[:, :-1] > 0,
                    shell_area_difference / np.where(summed > 0, summed, 1.),
                    z[:, 1:])
    volume = np.diff(edges) * (edges[1:]**2 + edges[1:] * edges[:-1]
                               + edges[:-1]**2) / 3
    mean_weights = impact_weights[:, None] * path / (2 * volume)
    area = edges**2
    flux_weights = np.where(impact_squared[:, None] < area,
                            impact_weights[:, None] / (2 * np.where(area > 0, area, 1.)),
                            0.)
    geometry_error = np.max(np.abs(2 * mean_weights.sum(axis=0) - 1))
    return SphericalShellGeometry(*map(jnp.asarray, (
        path, impact_squared < area[0], mean_weights, flux_weights,
        area, volume, geometry_error,
    )))


@jax.jit
def solve_spherical_transfer(
    geometry, extinction_cm1, emissivity, inner_intensity, outer_intensity,
):
    """Return shell-volume J and face H from the same finite-chord quadrature.

    Coefficients have shape [cell, frequency], boundaries [frequency], in the
    same units as ``solve_slab_transfer``. The core absorbs incoming rays;
    its outgoing intensity is prescribed. Rays missing it turn through the
    far hemisphere. Negative extinction is allowed, but overflow or failed
    conservation/geometry checks produce NaN fields. Saturation is not added.
    ``converged`` checks per-frequency energy conservation relative to gross
    supplied power, not relative accuracy in every faint cell or angular resolution.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 for radiative transfer.")
    chi, eta = jnp.asarray(extinction_cm1), jnp.asarray(emissivity)
    lower, upper = jnp.asarray(inner_intensity), jnp.asarray(outer_intensity)
    cells = geometry.path_cm.shape[1]
    if (chi.ndim != 2 or chi.shape[0] != cells or eta.shape != chi.shape
            or lower.shape != (chi.shape[1],) or upper.shape != lower.shape):
        raise ValueError("Coefficients, boundaries and shell geometry must agree.")

    def chord(path, hits_core):
        depth, emitted = chi * path[:, None], eta * path[:, None]
        first, second = _cell_weights(depth)

        def sweep(boundary, reverse):
            def segment(incoming, coefficients):
                tau, emission, w1, w2 = coefficients
                outgoing = incoming * jnp.exp(-tau) + emission * w1
                average = incoming * w1 + emission * w2
                return outgoing, (outgoing, average)
            final, (outgoing, mean) = jax.lax.scan(
                segment, boundary, (depth, emitted, first, second), reverse=reverse,
            )
            faces = (jnp.concatenate((outgoing, boundary[None])) if reverse
                     else jnp.concatenate((boundary[None], outgoing)))
            return final, faces, mean

        turn, inward, mean_in = sweep(upper, True)
        _, outward, mean_out = sweep(jnp.where(hits_core, lower, turn), False)
        gain = jnp.sum(jnp.maximum(-depth, 0), axis=0) * jnp.where(hits_core, 1, 2)
        return mean_in + mean_out, outward - inward, gain

    means, fluxes, gains = jax.vmap(chord)(geometry.path_cm, geometry.hits_core)
    mean = jnp.einsum("qc,qcf->cf", geometry.mean_weights, means)
    flux = jnp.einsum("qe,qef->ef", geometry.flux_weights, fluxes)
    error = _energy_balance_error(geometry, chi, eta, mean, flux, lower, upper)
    valid = jnp.all(jnp.isfinite(chi))
    for value in (eta, lower, upper):
        valid &= jnp.all(jnp.isfinite(value) & (value >= 0))
    converged = (valid & jnp.all(jnp.isfinite(mean)) & jnp.all(mean >= 0)
                 & jnp.all(jnp.isfinite(flux)) & (error < 1e-9)
                 & (geometry.volume_relative_error < 1e-9))
    return RadiationSolution(
        jnp.where(converged, mean, jnp.nan), jnp.where(converged, flux, jnp.nan),
        jnp.maximum(error, geometry.volume_relative_error), error,
        jnp.max(gains), jnp.asarray(0.), valid, converged,
    )


@jax.jit
def solve_spherical_scattering(
    geometry, absorption_cm1, emission, scattering_cm1, inner_intensity, outer_intensity,
):
    """Close isotropic coherent scattering in a positive continuum.

    Two linear right-hand sides separate emission/inner-boundary radiation
    from the response to unit outer intensity. This permits exact stellar
    line samples to multiply the same interpolated boundary response. The
    returned flux balance counts only true absorption/emission as heat.
    Frequency redistribution and electron energy exchange are not included.
    """
    absorption, eta = jnp.asarray(absorption_cm1), jnp.asarray(emission)
    scattering = jnp.broadcast_to(scattering_cm1, absorption.shape)
    inner, outer = jnp.asarray(inner_intensity), jnp.asarray(outer_intensity)
    extinction = absorption + scattering
    cells, frequencies = absorption.shape
    zero = jnp.zeros(frequencies)
    sources = jnp.eye(cells)[:, :, None] * jnp.ones(frequencies)
    response = jax.vmap(lambda source: solve_spherical_transfer(
        geometry, extinction, source, zero, zero).mean_intensity)(sources)
    response = jnp.transpose(response, (2, 1, 0))
    emitted = solve_spherical_transfer(geometry, extinction, eta, inner, zero)
    boundary = solve_spherical_transfer(geometry, extinction, jnp.zeros_like(eta),
                                         zero, jnp.ones_like(zero))
    matrix = jnp.eye(cells)[None, :, :] - response * scattering.T[:, None, :]
    rhs = jnp.stack((emitted.mean_intensity.T, boundary.mean_intensity.T), axis=-1)
    solved = jnp.linalg.solve(matrix, rhs)
    defect = jnp.einsum("fcs,fsb->fcb", matrix, solved) - rhs
    rhs_norm = jnp.max(jnp.abs(rhs), axis=1)
    linear_error = jnp.max(jnp.max(jnp.abs(defect), axis=1)
                           / jnp.maximum(rhs_norm, jnp.finfo(jnp.float64).tiny))
    emission_mean, boundary_response = solved[..., 0].T, solved[..., 1].T
    mean = emission_mean + boundary_response * outer
    radiation = solve_spherical_transfer(
        geometry, extinction, eta + scattering * mean, inner, outer)
    # Coherent scattering is internal recycling, not supplied energy or heat.
    balance = _energy_balance_error(
        geometry, absorption, eta, radiation.mean_intensity,
        radiation.flux_moment_edges, inner, outer)
    valid = (radiation.valid & jnp.all(jnp.isfinite(scattering) & (scattering >= 0))
             & jnp.all(jnp.isfinite(absorption) & (absorption >= 0)))
    converged = (radiation.converged & valid & jnp.all(jnp.isfinite(solved))
                 & jnp.all(solved >= 0) & (linear_error < 1e-10) & (balance < 1e-9))
    radiation = radiation._replace(
        mean_intensity=jnp.where(converged, radiation.mean_intensity, jnp.nan),
        flux_moment_edges=jnp.where(converged, radiation.flux_moment_edges, jnp.nan),
        residual_norm=jnp.maximum(radiation.residual_norm, linear_error),
        balance_error=balance, valid=valid, converged=converged,
    )
    return SphericalScatteringSolution(
        radiation, linear_error, jnp.where(converged, emission_mean, jnp.nan),
        jnp.where(converged, boundary_response, jnp.nan),
    )


__all__ = ["SphericalShellGeometry", "SphericalScatteringSolution",
           "spherical_shell_geometry", "solve_spherical_transfer", "solve_spherical_scattering"]
