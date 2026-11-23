"""Convert a tabulated stellar surface flux to a prescribed photon field."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .hydrogen_rates import LIGHT_SPEED_CGS, PLANCK_CGS


def surface_flux_occupation(frequency_hz, surface_frequency_hz, surface_flux_nu,
                            *, dilution=0.025):
    """Interpolate surface F_nu and return occupation for J_nu = W F_nu/pi.

    Frequencies are vacuum Hz; surface flux is erg s^-1 cm^-2 Hz^-1, already
    integrated over the outward hemisphere. This adopts a uniformly bright
    stellar disk with I_nu=F_nu/pi and geometrical dilution W. It does not
    attenuate the incident field or include atmospheric diffuse radiation.
    The surface grid must be strictly increasing. Outside coverage returns
    NaN: completing the UV/IR tails is an explicit caller choice. The result
    remains differentiable in flux and dilution; line-profile averaging, if
    needed, must be performed by the caller.
    """
    frequency = jnp.asarray(frequency_hz)
    grid, flux = jnp.asarray(surface_frequency_hz), jnp.asarray(surface_flux_nu)
    if grid.ndim != 1 or grid.size < 2 or flux.shape != grid.shape:
        raise ValueError("Surface frequency and flux require matching 1-D grids.")
    valid = (jnp.all(jnp.isfinite(grid)) & jnp.all(grid > 0)
             & jnp.all(jnp.diff(grid) > 0)
             & jnp.all(jnp.isfinite(flux)) & jnp.all(flux >= 0)
             & jnp.isfinite(dilution) & (dilution >= 0)
             & jnp.all(jnp.isfinite(frequency)) & jnp.all(frequency > 0))
    sampled = jnp.interp(frequency, grid, flux, left=jnp.nan, right=jnp.nan)
    occupation = dilution * LIGHT_SPEED_CGS**2 * sampled / (
        2 * jnp.pi * PLANCK_CGS * frequency**3
    )
    return jnp.where(valid, occupation, jnp.nan)


def prepare_tabulated_boundary(frequency_edges_hz, weight_hz, quadrature_order,
                               surface_frequency_hz, surface_flux_nu, *,
                               line_frequency_hz=(), outside_zero=False):
    """Project a static surface spectrum onto continuum bins and line centers.

    Return surface F_nu at continuum nodes and exact line centers. Multiply
    both by F_incident/F_surface/(4*pi) to obtain a normal-beam mean intensity.
    This NumPy preparation is outside the differentiable atmospheric solve;
    the returned arrays can be scaled by differentiable radiation parameters.

    TABLE SED-style log/log interpolation is integrated analytically, including
    narrow stellar absorption between continuum nodes. Each Gauss interval
    is partitioned in proportion to its positive weights, preserving its
    total incident energy. Cross-section-weighted rates still need a continuum
    refinement check. Set outside_zero=True only for an explicit zero-tail
    boundary; otherwise incomplete coverage raises an error.
    """
    edges, weights = np.asarray(frequency_edges_hz), np.asarray(weight_hz)
    frequency, flux = np.asarray(surface_frequency_hz), np.asarray(surface_flux_nu)
    lines = np.asarray(line_frequency_hz)
    if (edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0)
            or frequency.ndim != 1 or len(frequency) < 2 or flux.shape != frequency.shape
            or np.any(np.diff(frequency) <= 0) or np.any(frequency <= 0)
            or np.any(flux <= 0) or not np.isfinite(frequency).all()
            or not np.isfinite(flux).all() or not np.isfinite(edges).all()
            or edges[0] <= 0 or lines.ndim != 1 or not np.isfinite(lines).all()
            or np.any(lines <= 0) or not isinstance(quadrature_order, int)
            or quadrature_order < 1 or weights.shape != ((len(edges)-1)*quadrature_order,)
            or np.any(weights <= 0) or not np.isfinite(weights).all()):
        raise ValueError("Use positive finite flux, increasing frequencies and matching quadrature weights.")
    if not outside_zero and (edges[0] < frequency[0] or edges[-1] > frequency[-1]
            or (lines.size and (lines.min() < frequency[0] or lines.max() > frequency[-1]))):
        raise ValueError("The supplied spectrum does not cover the boundary; choose tails explicitly.")
    block = weights.reshape(-1, quadrature_order)
    subedges = edges[:-1, None]+np.diff(edges)[:, None]*np.cumsum(block, axis=1)/block.sum(1)[:, None]
    subedges[:, -1] = edges[1:]
    subedges = np.r_[edges[0], subedges.ravel()]
    # Integrate merged source/target segments directly into bins: subtracting
    # global cumulative integrals loses the faint far-UV tail by cancellation.
    knots = np.unique(np.r_[subedges, frequency[(frequency > edges[0]) & (frequency < edges[-1])]])
    left, right = knots[:-1], knots[1:]
    midpoint = np.sqrt(left*right)
    covered = (midpoint >= frequency[0]) & (midpoint <= frequency[-1])
    index = np.clip(np.searchsorted(frequency, midpoint)-1, 0, len(frequency)-2)
    slope = np.log(flux[index+1]/flux[index])/np.log(frequency[index+1]/frequency[index])
    width = np.log(right/left)
    exponent = (slope+1)*width
    ratio = np.ones_like(exponent)
    np.divide(np.expm1(exponent), exponent, out=ratio, where=exponent != 0.)
    integral = flux[index]*(left/frequency[index])**slope*left*width*ratio
    integral = np.where(covered, integral, 0.)
    bins = np.clip(np.searchsorted(subedges, midpoint)-1, 0, len(weights)-1)
    continuum = np.bincount(bins, weights=integral, minlength=len(weights))/weights
    sampled = np.exp(np.interp(np.log(lines), np.log(frequency), np.log(flux)))
    sampled = np.where((lines >= frequency[0]) & (lines <= frequency[-1]), sampled, 0.)
    return continuum, sampled
