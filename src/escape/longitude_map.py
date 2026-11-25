"""Fixed-opacity longitude source redistribution for spherical diagnostics.

The map scales local emissivity and the emitting lower boundary, including
its subtracted continuum. It does not change opacity, populations or wind.
Longitude is measured from the substellar meridian; only x > 0 emits here.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)


def longitude_weights(longitude, cosine, sine):
    """Positive map with unit dayside surface-area mean (latitude independent)."""
    nodes = jnp.asarray(_NODES) * (jnp.pi / 2)
    log_mean = logsumexp(
        cosine * jnp.cos(nodes) + sine * jnp.sin(nodes),
        b=jnp.asarray(_WEIGHTS) / 2,
    )
    return jnp.exp(cosine * jnp.cos(longitude) + sine * jnp.sin(longitude) - log_mean)


def longitude_spectrum(kernels, longitude, cosine, sine):
    """Combine local-source kernels; their penultimate axis is longitude."""
    return jnp.einsum(
        "...lv,l->...v", kernels, longitude_weights(longitude, cosine, sine)
    )


def _hat(longitude, nodes):
    index = np.clip(np.searchsorted(nodes, longitude) - 1, 0, len(nodes) - 2)
    fraction = (longitude - nodes[index]) / (nodes[index + 1] - nodes[index])
    return index, float(np.clip(fraction, 0, 1))


def rebin_longitude_kernel(kernel, fine_nodes, coarse_nodes):
    """Project a nested fine hat partition onto a coarser hat partition."""
    result = np.zeros((len(coarse_nodes), kernel.shape[-1]))
    for node, row in zip(fine_nodes, kernel):
        index, fraction = _hat(node, coarse_nodes)
        result[index] += (1 - fraction) * row
        result[index + 1] += fraction * row
    return result


def observer_basis(phase):
    observer = np.array([-np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase), 0.0])
    return (
        observer,
        np.array([-observer[1], observer[0], 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )


def disk_rays(phase, nlon, nlat):
    """Visible base-sphere rays and projected area weights in units of Rp²."""
    sinlat, weights = np.polynomial.legendre.leggauss(nlat)
    day = -np.pi / 2 + (np.arange(nlon) + 0.5) * np.pi / nlon
    latitude, longitude = np.meshgrid(
        np.arcsin(sinlat), np.r_[day, day + np.pi], indexing="ij"
    )
    normal = np.stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    )
    observer, _, _ = observer_basis(phase)
    mu = normal @ observer
    for i, j in np.argwhere(mu > 0):
        yield (
            normal[i, j] - mu[i, j] * observer,
            mu[i, j] * weights[i] * np.pi / nlon,
            normal[i, j] if j < nlon else None,
        )


def halo_rays(phase, edges, nrad, naz):
    """Off-limb rays with impact integration split at each radial shell edge."""
    _, e1, e2 = observer_basis(phase)
    nodes, weights = np.polynomial.legendre.leggauss(nrad)
    angles = 2 * np.pi * (np.arange(naz) + 0.5) / naz
    for left, right in zip(edges[:-1], edges[1:]):
        squared_radii = (left**2 + right**2) / 2 + (right**2 - left**2) * nodes / 2
        radial_weights = (right**2 - left**2) * weights / 4
        for radius, weight in zip(np.sqrt(squared_radii), radial_weights):
            for angle in angles:
                yield (
                    radius * (np.cos(angle) * e1 + np.sin(angle) * e2),
                    weight * 2 * np.pi / naz,
                    None,
                )


def chord_segments(impact, observer, edges):
    """Far-to-near dayside segments; an opaque unit-radius planet blocks rays.

    Nightside opacity and emissivity are absent, matching this prescribed
    observer, rather than implying a transparent real nightside atmosphere.
    """
    edges = np.asarray(edges)
    if edges[0] != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("Shell edges must increase from the opaque radius 1.")
    impact2 = np.dot(impact, impact)
    if impact2 >= edges[-1] ** 2:
        return []
    outer = np.sqrt(edges[-1] ** 2 - impact2)
    start = np.sqrt(max(0.0, 1 - impact2)) if impact2 < 1 else -outer
    cuts = [start, outer]
    for radius in edges:
        if radius**2 > impact2:
            distance = np.sqrt(radius**2 - impact2)
            cuts.extend(x for x in (-distance, distance) if start < x < outer)
    if observer[0] != 0:
        terminator = -impact[0] / observer[0]
        if start < terminator < outer:
            cuts.append(terminator)
    cuts = np.unique(cuts)
    segments = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        position = impact + (left + right) / 2 * observer
        cell = np.searchsorted(edges, np.linalg.norm(position), side="right") - 1
        if position[0] > 0 and 0 <= cell < len(edges) - 1 and right > left:
            segments.append((cell, left, right))
    return segments


def projected_velocity(position, observer, radial, rotation, zonal, layer_radius):
    """Existing observer convention: radial wind, rotation and sin-longitude flow."""
    radius = np.linalg.norm(position)
    xy = np.hypot(position[0], position[1])
    if xy == 0:
        return -np.dot(position / radius, observer) * radial
    tangent = np.array([-position[1] / xy, position[0] / xy, 0.0])
    speed = rotation * xy / radius / layer_radius + zonal * position[1] / xy
    return (
        -np.dot(position / radius, observer) * radial
        - np.dot(tangent, observer) * speed
    )


def source_longitude_kernel(
    phase,
    edges,
    nodes,
    coefficient,
    boundary,
    radial_velocity,
    planet_radius_cm,
    rotation_kms,
    zonal_kms,
    grid,
):
    """Integrate local source terms at their longitude and Doppler velocity.

    ``coefficient(cell, los_kms)`` returns extinction [cm^-1] and emissivity
    per length on a fixed spectral grid. The output is projected intensity
    per longitude hat function, before instrumental convolution or stellar
    normalization. Front-to-back accumulation preserves attenuation without
    evolving a separate intensity ray for each longitude node.
    """
    observer, _, _ = observer_basis(phase)
    edges, nodes, boundary = map(np.asarray, (edges, nodes, boundary))
    if len(nodes) < 2 or nodes[0] != -np.pi / 2 or nodes[-1] != np.pi / 2:
        raise ValueError("Longitude nodes must span the complete dayside.")
    kernel = np.zeros((len(nodes), len(boundary)))
    rays = list(disk_rays(phase, grid["nlon"], grid["nlat"]))
    rays.extend(halo_rays(phase, edges, grid["nrad_per_interval"], grid["naz"]))
    radii = (edges[:-1] + edges[1:]) / 2
    for impact, area, base in rays:
        foreground = np.ones_like(boundary)
        for cell, left, right in reversed(chord_segments(impact, observer, edges)):
            width = (right - left) / grid["subsegments"]
            length = width * planet_radius_cm
            for step in reversed(range(grid["subsegments"])):
                position = impact + (left + (step + 0.5) * width) * observer
                velocity = projected_velocity(
                    position,
                    observer,
                    radial_velocity[cell],
                    rotation_kms,
                    zonal_kms,
                    radii[cell],
                )
                chi, eta = coefficient(cell, velocity)
                tau = chi * length
                factor = np.divide(
                    -np.expm1(-tau), chi, out=np.full_like(chi, length), where=chi != 0
                )
                source = area * foreground * eta * factor
                index, fraction = _hat(np.arctan2(position[1], position[0]), nodes)
                kernel[index] += (1 - fraction) * source
                kernel[index + 1] += fraction * source
                foreground *= np.exp(-tau)
        if base is not None:
            source = area * boundary * (foreground - 1)
            index, fraction = _hat(np.arctan2(base[1], base[0]), nodes)
            kernel[index] += (1 - fraction) * source
            kernel[index + 1] += fraction * source
    return kernel
