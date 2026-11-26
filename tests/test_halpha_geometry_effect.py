"""Independent photon-budget limits for the frozen H-alpha experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

_PATH = Path(__file__).resolve().parents[1] / "examples/halpha_geometry_effect.py"
_SPEC = importlib.util.spec_from_file_location("halpha_geometry_effect", _PATH)
geometry = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(geometry)


@pytest.mark.parametrize("shape", ["sphere", "slab"])
def test_grey_branch_fates_against_independent_escape_integral(shape):
    radius, total_chi = 1.7, .7
    wn = np.array([12000., 14000., 17000.])
    line_chi = np.broadcast_to(np.array([.2, .4])[None, :, None], (1, 2, 3))
    line_eta = np.broadcast_to(np.array([1., 3.])[None, :, None], (1, 2, 3))
    solved = geometry.photon_fates([0., radius], wn, line_chi, line_eta,
                                   np.full((1, 3), .1), shape, angle_order=64)
    tau = radius * total_chi
    if shape == "sphere":
        escaped = 3 / (2 * tau) * quad(lambda mu: mu * -np.expm1(-2 * tau * mu), 0, 1,
                                       epsabs=1e-13)[0]
        expected_core, expected_outer = 0., escaped
    else:
        escaped = quad(lambda mu: mu * -np.expm1(-tau / mu), 0, 1,
                        epsabs=1e-13)[0] / tau
        expected_core = expected_outer = escaped / 2
    np.testing.assert_allclose(solved["outer"], expected_outer, rtol=2e-11)
    np.testing.assert_allclose(solved["core"], expected_core, atol=1e-14, rtol=2e-11)
    np.testing.assert_allclose(solved["line"][0, :, 0],
                               np.broadcast_to((1 - escaped) * np.array([.2, .4]) / .7, (2, 2)),
                               rtol=2e-11)
    np.testing.assert_allclose(solved["continuum"], (1 - escaped) / 7, rtol=2e-11)
    assert solved["accounting_error"] < 1e-12


def test_spherical_cell_subdivision_preserves_photon_budget_and_mean():
    # Splitting a homogeneous full sphere changes no physical coefficient.
    wn = np.array([14000., 15000., 16000.])
    edges = np.array([0., .3, .8, 1.7])
    chi, eta = np.full((3, 3), .7), np.full((3, 3), .4)
    whole = geometry.formal_fields([0., 1.7], chi[:1], eta[:1], angle_order=64)
    divided = geometry.formal_fields(edges, chi, eta, angle_order=64)
    volume, _ = geometry.geometry_measure(edges, "sphere")
    np.testing.assert_allclose(np.sum(volume[:, None] * divided.mean_intensity, axis=0) / volume.sum(),
                               whole.mean_intensity[0], rtol=2e-12)
    np.testing.assert_allclose(divided.flux_moment_edges[-1], whole.flux_moment_edges[-1], rtol=2e-12)
    fates = geometry.photon_fates(edges, wn, chi[:, None], eta[:, None], np.zeros_like(chi),
                                   spectral_weights=np.array([200., 300., 500.]), angle_order=64)
    weighted_escape = np.sum(fates["outer"] * fates["emitted"]) / fates["emitted"].sum()
    expected = 1.7**2 * geometry.photon_integral(wn, whole.flux_moment_edges[-1]) / (
        1.7**3 / 3 * geometry.photon_integral(wn, eta[0]))
    np.testing.assert_allclose(weighted_escape, expected, rtol=2e-12)
    assert fates["accounting_error"] < 1e-12


def test_transparent_sources_have_no_absorption_and_conserve_photons():
    wn = np.array([14000., 15000., 16000.])
    eta = np.ones((2, 1, 3))
    fates = geometry.photon_fates([1., 1.3, 2.], wn, np.zeros_like(eta), eta, np.zeros((2, 3)))
    np.testing.assert_array_equal(fates["line"], 0.)
    np.testing.assert_array_equal(fates["continuum"], 0.)
    np.testing.assert_allclose(fates["core"] + fates["outer"], 1., rtol=1e-12)
    assert np.all(fates["core"] > 0)
    assert np.all(fates["outer"] > fates["core"])


def test_explicit_spectral_weights_preserve_photon_measure():
    wn, weights = np.array([1., 2., 7.]), np.array([.2, .7, .1])
    # A spectrum proportional to photon energy has a constant photon density.
    eta = geometry.H_CGS * geometry.C_CGS * wn * 3
    np.testing.assert_allclose(geometry.photon_integral(wn, eta, weights), 3.)
