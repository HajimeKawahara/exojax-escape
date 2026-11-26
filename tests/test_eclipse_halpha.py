"""Check the new reference derivative and profiled temporal covariance."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("matplotlib")

SPEC = spec_from_file_location("eclipse_halpha", Path(__file__).resolve().parents[1]
                              / "examples" / "eclipse_halpha.py")
EXAMPLE = module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def test_remaster_derivative_includes_reference_dependent_weights():
    phase = np.array([.41, .42, .48, .49])
    wave = EXAMPLE.REST_ANGSTROM * (1 + np.array([[130., 145., 160., 175.]]) / EXAMPLE.C_KMS)
    grid = EXAMPLE.REST_ANGSTROM * (1 + np.arange(-30., 76., 15.) / EXAMPLE.C_KMS)
    r = np.array([[[.02, .01, -.01, .03]], [[.01, .03, .01, .02]],
                  [[-.01, .01, .00, -.02]], [[.02, -.01, .01, .01]]])
    error = np.full_like(r, .002)
    error[3] *= 1.3
    science, reference = phase < .45, phase > .45
    result = EXAMPLE.remaster(wave, phase, r, error, science, reference, grid)
    coordinate = np.unravel_index(np.argmax(abs(result["derivative"])), result["derivative"].shape)
    j, order, pixel = coordinate
    step = 1e-6
    plus, minus = r.copy(), r.copy()
    plus[reference, order, pixel] += step
    minus[reference, order, pixel] -= step
    rp = EXAMPLE.remaster(wave, phase, plus, error, science, reference, grid)
    rm = EXAMPLE.remaster(wave, phase, minus, error, science, reference, grid)
    finite = (rp["value"][j] - rm["value"][j]) / (2 * step)
    np.testing.assert_allclose(finite, result["derivative"][coordinate], atol=2e-10)
    covariance = result["reference_covariance"]
    assert np.max(abs(covariance - np.diag(np.diag(covariance)))) > 0
    assert np.linalg.eigvalsh(covariance).min() > -1e-15
    # A changed old common master transforms flux AND its errors.
    master = np.array([[1.02, .98, 1.01, 1.04]])
    changed = EXAMPLE.remaster(wave, phase, (1 + r) / master - 1,
                               error / master, science, reference, grid)
    np.testing.assert_allclose(changed["value"], result["value"], atol=1e-14)
    np.testing.assert_allclose(changed["reference_covariance"], covariance, atol=1e-18)


def test_profiled_trend_removes_master_and_retains_shared_mean_covariance():
    phase = np.linspace(.471, .493, 15)
    wave = EXAMPLE.REST_ANGSTROM * (1 + np.linspace(115, 205, 12) / EXAMPLE.C_KMS)
    slope = -.035
    logflux = np.linspace(-.01, .02, len(wave))[None] + slope * (phase[:, None] - .48)
    sigma = .002
    r, error = np.exp(logflux) - 1, sigma * np.exp(logflux)
    result = EXAMPLE.time_control(phase, r, error, wave, 110, 210)
    np.testing.assert_allclose(result["slope"], slope, atol=1e-13)
    n = len(phase)
    expected = sigma**2 / len(wave) * (np.eye(n) - np.ones((n, n)) / n)
    np.testing.assert_allclose(result["covariance"], expected, atol=1e-20)
    assert np.linalg.matrix_rank(result["covariance"]) == n - 1
    master = np.exp(np.linspace(-.03, .03, len(wave)))
    changed = EXAMPLE.time_control(phase, (1 + r) / master - 1,
                                   error / master, wave, 110, 210)
    np.testing.assert_allclose(changed["centered"], result["centered"], atol=1e-15)
    np.testing.assert_allclose(changed["slope"], slope, atol=1e-13)
    for block in result["temporal_holdouts"]:
        assert block["heldout_weighted_sse_linear"] < 1e-20
