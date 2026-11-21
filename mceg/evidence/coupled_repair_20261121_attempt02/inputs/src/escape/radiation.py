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


__all__ = ["RadiationSolution", "moment_geometry", "solve_moment_transfer"]
