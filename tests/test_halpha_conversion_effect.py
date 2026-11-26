"""Conservation and recycling controls for the conditional photon network."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples"))
from halpha_conversion_effect import absorption_chain, network, oxygen_metadata


def test_repeated_lybeta_recycling_is_not_branching_destruction():
    branch_halpha, escape_halpha, escape_uv = .12, .03, 0.
    q = np.tile([branch_halpha*(1-escape_halpha),
                 (1-branch_halpha)*(1-escape_uv)], (2, 1))
    d = np.tile([branch_halpha*escape_halpha, (1-branch_halpha)*escape_uv], (2, 1))
    rewards = np.array([[1-branch_halpha, 0], [0, branch_halpha]])
    fate, events, audit = absorption_chain(q, d, rewards)
    np.testing.assert_allclose(fate, [[1, 0], [1, 0]], atol=1e-13)
    assert events[0, 0] > 20  # Many conversions in each direction, zero UV loss.
    np.testing.assert_allclose(events[0, 0]-events[0, 1], 0, atol=1e-12)
    assert audit["probability_error"] < 1e-12


def test_transparent_limit_and_continuum_sink_compete_conservatively():
    q = np.zeros((1, 1))
    d = np.array([[.12, .88]])
    fate, _, _ = absorption_chain(q, d, np.zeros((1, 1)))
    np.testing.assert_array_equal(fate, d)
    # A small true destruction chance accumulates over many repeated flights.
    q = np.array([[.99]])
    d = np.array([[.003, .007]])
    fate, _, _ = absorption_chain(q, d, np.zeros((1, 1)))
    np.testing.assert_allclose(fate, [[.3, .7]])


def test_unclosed_and_perfectly_trapped_networks_are_rejected():
    with pytest.raises(ValueError, match="Unclosed"):
        absorption_chain(np.array([[.8]]), np.array([[.1]]), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="trapped"):
        absorption_chain(np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))


def test_full_atomic_network_recycles_uv_and_preserves_color_accounting():
    rates = SimpleNamespace(electron_bound_bound_cm3_s=np.zeros((8, 18, 18)),
        radiative_bound_bound_s=np.zeros((8, 18, 18)),
        photoionization_s=np.zeros((8, 18)), electron_ionization_cm3_s=np.zeros((8, 18)))
    physical = dict(radiation=dict(rates=rates), electron_cm3=np.ones(8))
    def flights(branches):
        return dict(line=np.zeros((8, branches, 8, branches)),
            continuum=np.zeros((8, branches, 8)), core=np.zeros((8, branches)),
            outer=np.zeros((8, branches)))
    ha, uv = flights(3), flights(7)
    ha["outer"][:] = 1.
    for cell in range(8):
        uv["line"][cell, :, cell, 0] = 1.
    _, oxygen = oxygen_metadata()
    fate, events, audit = network(physical, ha, uv, oxygen,
        oxygen_cascade=False, atomic_losses=False, mixing=False)
    np.testing.assert_allclose(fate[..., 0], 1, atol=2e-14)
    np.testing.assert_allclose(fate[..., 1:], 0, atol=2e-14)
    # H-alpha origin: forward and reverse gross conversions cancel.
    np.testing.assert_allclose(events[:, 1, 0, 3]-events[:, 1, 0, 4], 0, atol=2e-14)
    # UV origin: one net conversion back to H-alpha.
    np.testing.assert_allclose(events[:, 1, 1, 3]-events[:, 1, 1, 4], -1, atol=2e-14)
    assert np.min(events[:, 1, 0, 1]) > 7
    assert audit["probability_error"] < 1e-12
