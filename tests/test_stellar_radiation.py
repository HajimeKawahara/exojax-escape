"""Radiance/flux units and explicit spectral coverage at the stellar boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_rates import LIGHT_SPEED_CGS, PLANCK_CGS, planck_occupation
from escape.stellar_radiation import prepare_tabulated_boundary, surface_flux_occupation

jax.config.update("jax_enable_x64", True)


def test_blackbody_surface_flux_recovers_diluted_planck_field():
    frequency = jnp.geomspace(1e13, 1e16, 50)
    radiance = (2 * PLANCK_CGS * frequency**3 / LIGHT_SPEED_CGS**2
                * planck_occupation(frequency, 10000., 1.))
    result = surface_flux_occupation(frequency, frequency, jnp.pi * radiance,
                                     dilution=0.025)
    np.testing.assert_allclose(result, planck_occupation(frequency, 10000., .025),
                               rtol=1e-14)


def test_coverage_and_invalid_spectra_are_not_silently_extrapolated():
    grid = jnp.array([1e14, 2e14, 3e14])
    result = surface_flux_occupation(jnp.array([.5e14, 1.5e14, 4e14]), grid,
                                     jnp.ones(3))
    assert np.isnan(result[0]) and np.isfinite(result[1]) and np.isnan(result[2])
    for invalid_grid, invalid_flux in [(grid[::-1], jnp.ones(3)),
                                        (grid, jnp.array([1., -1., 1.]))]:
        assert np.isnan(surface_flux_occupation(2e14, invalid_grid, invalid_flux))
    with pytest.raises(ValueError, match="matching 1-D"):
        surface_flux_occupation(2e14, grid, jnp.ones(2))


def test_dilution_gradient_preserves_flux_normalization():
    grid = jnp.array([1e14, 2e14, 3e14])
    flux = jnp.array([1e-5, 2e-5, 1e-5])
    signal = lambda scale: jnp.sum(surface_flux_occupation(
        grid, grid, flux, dilution=scale,
    ))
    np.testing.assert_allclose(jax.grad(signal)(.025), signal(.025) / .025,
                               rtol=1e-14)


def test_integrated_boundary_preserves_a_narrow_absorption_trough_and_faint_uv():
    # Analytic piecewise power laws; the absorption is between coarse nodes.
    frequency = np.array([1., 1.4, 1.5, 1.6, 2., 3., 4.])*1e15
    flux = np.array([1., 1., .01, 1., 1., 1e-20, 1e-30])
    edges = np.array([1., 2., 3., 4.])*1e15
    weights = np.repeat(np.diff(edges)/2, 2)
    actual, lines = prepare_tabulated_boundary(edges, weights, 2, frequency, flux,
                                              line_frequency_hz=[1.5e15, 3.5e15])
    slope = np.log(flux[1:]/flux[:-1])/np.log(frequency[1:]/frequency[:-1])
    integral = flux[:-1]*frequency[:-1]*np.expm1((slope+1)*np.log(frequency[1:]/frequency[:-1]))/(slope+1)
    np.testing.assert_allclose(np.sum(actual[:2]*weights[:2]), integral[:4].sum(), rtol=2e-15)
    np.testing.assert_allclose(np.sum(actual[-2:]*weights[-2:]), integral[-1], rtol=3e-14)
    assert lines[0] == pytest.approx(.01)
    assert actual[:2].mean() < .95
    with pytest.raises(ValueError, match="choose tails"):
        prepare_tabulated_boundary([.5e15, 4e15], [3.5e15], 1, frequency, flux)
    tail, _ = prepare_tabulated_boundary([4e15, 5e15], [1e15], 1, frequency, flux,
                                         outside_zero=True)
    assert tail[0] == 0.
