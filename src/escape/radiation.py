"""Conservative static Eddington moments, with spherical and slab geometries.

K=J/3 is an imposed angular closure. This solves an internally coupled mean
field, not exact angle-dependent transport or a moving-medium line problem.
All frequency bins are independent for prescribed extinction and emissivity.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class RadiationSolution(NamedTuple):
    mean_intensity: jax.Array
    flux_moment_edges: jax.Array
    residual_norm: jax.Array
    balance_error: jax.Array
    gain_optical_depth: jax.Array
    realizability_margin: jax.Array
    valid: jax.Array
    converged: jax.Array


def moment_geometry(edges_cm, spherical=True):
    """Return face areas and cell volumes per steradian (unit area for slab)."""
    edges = jnp.asarray(edges_cm)
    if spherical:
        area = edges**2
        volume = jnp.diff(edges) * (edges[1:]**2 + edges[1:] * edges[:-1]
                                  + edges[:-1]**2) / 3
    else:
        area, volume = jnp.ones_like(edges), jnp.diff(edges)
    return area, volume


@partial(jax.jit, static_argnames=("spherical",))
def solve_moment_transfer(
    edges_cm, extinction_cm1, emissivity, inner_intensity, outer_intensity,
    *, spherical=True,
):
    """Solve d(A H)/dr=A(j-chi J), dJ/dr=-3 chi H on a fixed grid.

    Inputs have N+1 radial edges [cm], two (N,M) coefficient arrays, and two
    (M,) incoming intensities [erg s^-1 cm^-2 Hz^-1 sr^-1]. Emissivity has an
    additional cm^-1. J is a finite-volume cell value; H is a face moment,
    with physical flux F=4*pi*H. Incoming characteristics are J+sqrt(3)H at
    the inner boundary and J-sqrt(3)H at the outer boundary. Half-cell optical
    depths close the boundary interpolation. No incoming flux is inferred
    from a dilution factor. Radial arrays run from inner to outer.

    The mixed J/H system never divides by opacity: transparent emitting cells
    are supported. Float64 and nonnegative emissivity are required. Signed
    extinction is allowed only for weak integrated gain: the maximum over
    frequency of sum(max(-chi,0)*dr) must not exceed 0.25. This numerical
    domain bound is not a validated maser saturation or atomic-model criterion. Linear
    solves use JAX implicit AD. Failed inputs/solutions return NaN fields.
    Reconstructed faces must also admit nonnegative two-stream intensities,
    J >= sqrt(3)*abs(H), up to rounding. In particular, a wide transparent
    sphere with strongly outward beaming can violate the fixed K=J/3 closure.
    The exact discrete energy balance does not validate the Eddington closure.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 for radiative transfer.")
    edges = jnp.asarray(edges_cm, dtype=jnp.float64)
    chi, emission = jnp.asarray(extinction_cm1), jnp.asarray(emissivity)
    incoming_inner, incoming_outer = jnp.asarray(inner_intensity), jnp.asarray(outer_intensity)
    if chi.ndim != 2 or emission.shape != chi.shape:
        raise ValueError("Extinction and emissivity must have shape (Ncell,Nfrequency).")
    count, frequencies = chi.shape
    if (count < 1 or edges.shape != (count + 1,)
            or incoming_inner.shape != (frequencies,)
            or incoming_outer.shape != (frequencies,)):
        raise ValueError("Edges and incoming intensities must match the coefficient arrays.")
    area, volume = moment_geometry(edges, spherical)
    # A common geometric rescaling prevents centimeter powers from dominating
    # the mixed system; it leaves every conservation equation unchanged.
    area_scale = area[0]
    area, volume = area / area_scale, volume / area_scale
    widths = jnp.diff(edges)
    indices = jnp.arange(count)

    def one_frequency(opacity, eta, lower, upper):
        matrix = jnp.zeros((2 * count + 1, 2 * count + 1))
        rhs = jnp.zeros(2 * count + 1)
        scale = area[:-1] + area[1:] + jnp.abs(opacity) * volume
        matrix = matrix.at[indices, indices].set(opacity * volume / scale)
        matrix = matrix.at[indices, count + indices].set(-area[:-1] / scale)
        matrix = matrix.at[indices, count + indices + 1].set(area[1:] / scale)
        rhs = rhs.at[indices].set(eta * volume / scale)
        if count > 1:
            faces = jnp.arange(1, count)
            optical_distance = (opacity[:-1] * widths[:-1]
                                + opacity[1:] * widths[1:]) / 2
            scale = 2 + 3 * jnp.abs(optical_distance)
            rows = count + faces - 1
            matrix = matrix.at[rows, faces - 1].set(-1 / scale)
            matrix = matrix.at[rows, faces].set(1 / scale)
            matrix = matrix.at[rows, count + faces].set(3 * optical_distance / scale)
        inner_factor = jnp.sqrt(3.) + 1.5 * opacity[0] * widths[0]
        outer_factor = jnp.sqrt(3.) + 1.5 * opacity[-1] * widths[-1]
        matrix = matrix.at[-2, 0].set(1 / inner_factor)
        matrix = matrix.at[-2, count].set(1.)
        rhs = rhs.at[-2].set(lower / inner_factor)
        matrix = matrix.at[-1, count - 1].set(1 / outer_factor)
        matrix = matrix.at[-1, -1].set(-1.)
        rhs = rhs.at[-1].set(upper / outer_factor)
        values = jnp.linalg.solve(matrix, rhs)
        norm = jnp.max(jnp.abs(matrix @ values - rhs)) / jnp.maximum(
            jnp.max(jnp.abs(rhs)), jnp.finfo(jnp.float64).tiny,
        )
        mean, flux = values[:count], values[count:]
        net = area[1:] * flux[1:] - area[:-1] * flux[:-1]
        exchange = volume * (eta - opacity * mean)
        budget = (jnp.abs(area[1:] * flux[1:]) + jnp.abs(area[:-1] * flux[:-1])
                  + volume * (eta + jnp.abs(opacity * mean)))
        error = jnp.max(jnp.abs(net - exchange) / jnp.where(budget > 0, budget, 1.))
        return mean, flux, norm, error

    mean, flux, residual, balance = jax.vmap(one_frequency, in_axes=(1, 1, 0, 0),
                                           out_axes=(1, 1, 0, 0))(
        chi, emission, incoming_inner, incoming_outer,
    )
    face_inner = mean[:1] + 1.5 * chi[:1] * widths[:1, None] * flux[:1]
    face_outer = mean[-1:] - 1.5 * chi[-1:] * widths[-1:, None] * flux[-1:]
    face_middle = (mean[:-1] + mean[1:] + 1.5 * flux[1:-1] * (
        chi[1:] * widths[1:, None] - chi[:-1] * widths[:-1, None]
    )) / 2
    face_mean = jnp.concatenate((face_inner, face_middle, face_outer), axis=0)
    intensity_scale = jnp.max(jnp.abs(face_mean), axis=0)
    directional = face_mean - jnp.sqrt(3.) * jnp.abs(flux)
    normalized_margin = directional / jnp.where(intensity_scale > 0, intensity_scale, 1.)
    margin = jnp.min(normalized_margin)
    valid = (jnp.all(jnp.isfinite(edges)) & jnp.all(widths > 0)
             & (jnp.all(edges > 0) if spherical else True))
    gain_depth = jnp.max(jnp.sum(jnp.maximum(-chi, 0) * widths[:, None], axis=0))
    valid = valid & jnp.all(jnp.isfinite(chi)) & (gain_depth <= .25)
    for value in (emission, incoming_inner, incoming_outer):
        valid = valid & jnp.all(jnp.isfinite(value) & (value >= 0))
    converged = (valid & jnp.all(jnp.isfinite(mean)) & jnp.all(mean >= 0)
                 & jnp.all(jnp.isfinite(flux)) & jnp.all(residual < 1e-9)
                 & jnp.all(balance < 1e-9) & (margin >= -128 * jnp.finfo(jnp.float64).eps))
    return RadiationSolution(
        jnp.where(converged, mean, jnp.nan), jnp.where(converged, flux, jnp.nan),
        jnp.max(residual), jnp.max(balance), gain_depth, margin, valid, converged,
    )


def _cell_weights(depth):
    """Stable exponential averages, including the transparent limit."""
    small = jnp.abs(depth) < 1e-3
    safe = jnp.where(small, 1., depth)
    first = -jnp.expm1(-safe) / safe
    second = (1 - first) / safe
    first = jnp.where(small, 1 - depth / 2 + depth**2 / 6 - depth**3 / 24
                      + depth**4 / 120 - depth**5 / 720, first)
    second = jnp.where(small, .5 - depth / 6 + depth**2 / 24 - depth**3 / 120
                       + depth**4 / 720 - depth**5 / 5040, second)
    return first, second


@partial(jax.jit, static_argnames=("angle_order", "beam_mu"))
def solve_slab_transfer(
    edges_cm, extinction_cm1, emissivity, inner_intensity, outer_intensity,
    *, angle_order=4, beam_mean_intensity=None, beam_mu=1.,
):
    """Exact cell-averaged formal transfer for a static homogeneous-cell slab.

    Coefficients and boundary units match solve_moment_transfer. Each incoming
    boundary is isotropic over its hemisphere. Gauss-Legendre quadrature on
    0 < mu < 1 supplies J and H; no fixed Eddington factor is imposed. This
    is plane parallel, with complete redistribution supplied by the caller's
    emissivities, and does not implement Cloudy's escape/PRD closure.
    An optional inward stellar beam adds J_nu=beam_mean_intensity at the top
    with direction cosine beam_mu. Its normal flux is 4*pi*beam_mu*J_nu,
    unlike the flux of an isotropic hemisphere with the same incident J.

    Cell-average intensity, rather than midpoint intensity, makes absorbed
    minus emitted energy agree with the face flux difference even in thick
    cells. Negative extinction is unsupported; transparent emission and its
    derivatives are defined by analytic limits. No opacity clipping is used.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 for radiative transfer.")
    if not isinstance(angle_order, int) or isinstance(angle_order, bool) or angle_order < 1:
        raise ValueError("angle_order must be a positive static integer.")
    if not 0 < beam_mu <= 1:
        raise ValueError("beam_mu must lie in (0,1].")
    edges = jnp.asarray(edges_cm, dtype=jnp.float64)
    chi, emission = jnp.asarray(extinction_cm1), jnp.asarray(emissivity)
    lower, upper = jnp.asarray(inner_intensity), jnp.asarray(outer_intensity)
    if chi.ndim != 2 or emission.shape != chi.shape:
        raise ValueError("Coefficients must share shape (Ncell, Nfrequency).")
    count, frequencies = chi.shape
    if count < 1 or edges.shape != (count + 1,) or lower.shape != (frequencies,) or upper.shape != lower.shape:
        raise ValueError("Edges and boundary spectra must match the coefficient grid.")
    nodes, weights = np.polynomial.legendre.leggauss(angle_order)
    mu, weights = jnp.asarray((nodes + 1) / 2), jnp.asarray(weights / 2)
    widths = jnp.diff(edges)

    def direction(cosine):
        depth = chi * widths[:, None] / cosine
        emitted = emission * widths[:, None] / cosine
        first, second = _cell_weights(depth)

        def sweep(boundary, reverse):
            def cell(incoming, coefficients):
                tau, eta, w1, w2 = coefficients
                outgoing = incoming * jnp.exp(-tau) + eta * w1
                average = incoming * w1 + eta * w2
                return outgoing, (outgoing, average)
            _, (outgoing, average) = jax.lax.scan(
                cell, boundary, (depth, emitted, first, second), reverse=reverse,
            )
            faces = (jnp.concatenate((outgoing, boundary[None]), axis=0) if reverse
                     else jnp.concatenate((boundary[None], outgoing), axis=0))
            return faces, average

        outward, mean_out = sweep(lower, False)
        inward, mean_in = sweep(upper, True)
        return (mean_out + mean_in) / 2, cosine * (outward - inward) / 2

    means, fluxes = jax.vmap(direction)(mu)
    mean = jnp.einsum("a,acf->cf", weights, means)
    flux = jnp.einsum("a,acf->cf", weights, fluxes)
    if beam_mean_intensity is not None:
        incident = jnp.asarray(beam_mean_intensity)
        if incident.shape != lower.shape:
            raise ValueError("The incident beam must match the frequency grid.")
        depth = chi * widths[:, None] / beam_mu
        # Exclusive sum retains a finite overlying column even when the local
        # cell is extremely thick; inclusive-sum minus local depth cancels it.
        above = jnp.concatenate((jnp.cumsum(depth[::-1], axis=0)[:-1][::-1],
                                 jnp.zeros_like(depth[:1])), axis=0)
        incoming = incident * jnp.exp(-above)
        mean = mean + incoming * _cell_weights(depth)[0]
        beam_faces = jnp.concatenate((incoming * jnp.exp(-depth), incident[None]), axis=0)
        flux = flux - beam_mu * beam_faces
    exchange = widths[:, None] * (emission - chi * mean)
    net = jnp.diff(flux, axis=0)
    scale = (jnp.abs(flux[:-1]) + jnp.abs(flux[1:])
             + widths[:, None] * (emission + jnp.abs(chi * mean)))
    error = jnp.max(jnp.abs(net - exchange) / jnp.where(scale > 0, scale, 1.))
    valid = jnp.all(jnp.isfinite(edges)) & jnp.all(widths > 0)
    for value in (chi, emission, lower, upper):
        valid &= jnp.all(jnp.isfinite(value) & (value >= 0))
    if beam_mean_intensity is not None:
        valid &= jnp.all(jnp.isfinite(incident) & (incident >= 0))
    converged = valid & jnp.all(jnp.isfinite(mean)) & jnp.all(mean >= 0) & (error < 1e-9)
    return RadiationSolution(
        jnp.where(converged, mean, jnp.nan), jnp.where(converged, flux, jnp.nan),
        error, error, jnp.max(jnp.sum(jnp.maximum(-chi, 0) * widths[:, None], axis=0)),
        jnp.asarray(0.), valid, converged,
    )


__all__ = ["RadiationSolution", "moment_geometry", "solve_moment_transfer", "solve_slab_transfer"]
