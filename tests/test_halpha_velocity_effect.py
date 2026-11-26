"""Controls for the frozen-state moving H-alpha diagnostic."""

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.special import voigt_profile

from escape.spherical_radiation import spherical_shell_geometry, solve_spherical_transfer

jax.config.update("jax_enable_x64", True)
_path = Path(__file__).parents[1] / "examples" / "halpha_velocity_effect.py"
_spec = importlib.util.spec_from_file_location("halpha_velocity_effect", _path)
velocity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(velocity)


@pytest.fixture
def state():
    wn = np.linspace(14995., 15005., 401)
    return {"edges_cm": np.array([1., 1.2, 2.]) * 1e5,
            "wavenumber_cm": wn, "branch_centers": np.array([15000.]),
            "branch_sigma": np.array([[.3], [.4]]), "branch_gamma": np.array([.002]),
            "branch_alpha": np.array([[1e-5], [2e-7]]),
            "branch_eta": np.array([[1e-3], [2e-5]]),
            "continuum_extinction": np.full((2, wn.size), 1e-7),
            "continuum_emission_hz": np.full((2, wn.size), 1e-5 / velocity.C_CGS)}


@pytest.mark.parametrize("segments", [1, 3])
def test_zero_velocity_matches_static_solver_and_receiver_integral(state, segments):
    profile = voigt_profile(state["wavenumber_cm"][None, None, :]
                            - state["branch_centers"][None, :, None],
                            state["branch_sigma"][:, :, None],
                            state["branch_gamma"][None, :, None])
    line_chi = state["branch_alpha"][:, :, None] * profile
    chi = line_chi.sum(axis=1) + state["continuum_extinction"]
    eta = (state["branch_eta"][:, :, None] * profile).sum(axis=1)
    eta += state["continuum_emission_hz"] * velocity.C_CGS
    zero = np.zeros(chi.shape[1])
    geometry = spherical_shell_geometry(state["edges_cm"], angle_order=8)
    reference = solve_spherical_transfer(geometry, chi, eta, zero, zero)
    actual = velocity.moving_transfer(state, np.zeros(2), angle_order=8, segments=segments)
    assert bool(reference.converged)
    np.testing.assert_allclose(actual["mean_intensity"], reference.mean_intensity, rtol=2e-13)
    photon_absorption = 4 * np.pi * np.trapezoid(
        line_chi * np.asarray(reference.mean_intensity)[:, None, :]
        / (velocity.H_CGS * velocity.C_CGS * state["wavenumber_cm"]),
        state["wavenumber_cm"], axis=-1)
    np.testing.assert_allclose(actual["pump_by_source"].sum(axis=-1), photon_absorption,
                               rtol=2e-13)
    assert actual["geometry_relative_error"] < 1e-9


def test_common_los_boost_and_frequency_grid_shift_leave_absorption_invariant(state):
    original = velocity.moving_transfer(state, np.array([2., 13.]), segments=4)
    boosted = velocity.moving_transfer(state, np.array([2., 13.]), segments=4,
                                       uniform_los_boost_kms=21.)
    np.testing.assert_allclose(boosted["pump_by_source"], original["pump_by_source"],
                               rtol=2e-12)
    np.testing.assert_allclose(boosted["mean_intensity"], original["mean_intensity"],
                               rtol=2e-12)


def test_radial_flow_reversal_reflects_spectrum_but_preserves_narrow_band_pumping(state):
    outward = velocity.moving_transfer(state, np.array([0., 13.]), segments=8)
    inward = velocity.moving_transfer(state, np.array([0., -13.]), segments=8)
    np.testing.assert_allclose(outward["mean_intensity"], inward["mean_intensity"][:, ::-1],
                               rtol=2e-11)
    # The photon-energy denominator varies across the line; allow its O(v/c) effect.
    np.testing.assert_allclose(outward["pump_by_source"], inward["pump_by_source"], rtol=1e-4)
    static = velocity.moving_transfer(state, np.zeros(2), segments=8)
    assert outward["pump_by_source"][1, 0, 0] < static["pump_by_source"][1, 0, 0]


def test_rejects_missing_radial_velocity(state):
    with pytest.raises(ValueError, match="one finite radial velocity"):
        velocity.moving_transfer(state, np.zeros(1))
