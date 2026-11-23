"""A nearly unit profile integral can still miss optically thick line escape."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expn, voigt_profile

from escape.coupled_hydrogen import CoupledHydrogenModel, frequency_grid
from escape.hydrogen_atom import atom_natural_width_hz, make_hydrogen_atom
from escape.hydrogen_rates import BOLTZMANN_CGS, LIGHT_SPEED_CGS
from escape.parker import PROTON_MASS_CGS

jax.config.update("jax_enable_x64", True)


def _uniform_slab_escape(tau):
    """Exact depth/angle mean escape for a uniform source, without scattering."""
    # Integrating E2(tau*x) over 0<x<1 gives [1/2-E3(tau)]/tau.
    safe = np.maximum(tau, 1e-7)
    return np.where(tau < 1e-7, 1., (.5 - expn(3, safe)) / safe)


def test_wing_offsets_recover_thick_voigt_escape_that_profile_normalization_misses():
    atom = make_hydrogen_atom(15)
    center = float(atom.line_frequency_hz[1])  # Ly beta
    sigma = center / LIGHT_SPEED_CGS * np.sqrt(BOLTZMANN_CGS * 8000 / PROTON_MASS_CGS)
    damping = float(atom_natural_width_hz(atom)[1]) / sigma
    peak = voigt_profile(0., 1., damping)
    tau_center = 9.7917857e6

    def integrand(log_offset):
        offset = np.exp(log_offset)
        profile = voigt_profile(offset, 1., damping)
        return 2 * offset * profile * _uniform_slab_escape(tau_center * profile / peak)

    reference = quad(integrand, -25, 25, epsabs=1e-14, epsrel=1e-8, limit=300)[0]
    results = []
    for order, offsets in [(4, None), (4, (30, 100, 300, 1000)),
                           (8, (30, 100, 300, 1000))]:
        frequency, weight = map(np.asarray, frequency_grid(atom, order=order,
                                                          line_wing_offsets=offsets))
        profile = voigt_profile((frequency - center) / sigma, 1., damping) / sigma
        beta = np.sum(weight * profile * _uniform_slab_escape(
            tau_center * profile / (peak / sigma),
        ))
        results.append((beta, np.sum(weight * profile)))
    # The original profile area looks excellent, yet its escape integral fails.
    assert abs(results[0][1] - 1) < 3e-4
    assert abs(results[0][0] / reference - 1) > .1
    # The finite physical frequency window differs slightly from the infinite
    # symmetric Voigt reference, so sub-percent agreement is the relevant test.
    np.testing.assert_allclose([item[0] for item in results[1:]], reference, rtol=.005)
    np.testing.assert_allclose(results[1][0], results[2][0], rtol=.001)


def test_constructor_forwards_wing_offsets_and_none_preserves_the_original_grid():
    ordinary = CoupledHydrogenModel([0., 1.], nmax=3, spherical=False)
    extended = CoupledHydrogenModel([0., 1.], nmax=3, spherical=False,
                                    line_wing_offsets=(30, 100, 300, 1000))
    explicit = frequency_grid(ordinary.atom, line_wing_offsets=None)
    np.testing.assert_array_equal(ordinary.grid.frequency_hz, explicit.frequency_hz)
    assert len(extended.grid.frequency_hz) > len(ordinary.grid.frequency_hz)
    assert extended.line_wing_offsets == (30, 100, 300, 1000)


@pytest.mark.parametrize("offsets", [(3, 30), (np.nan,), [[30, 100]]])
def test_invalid_wing_offsets_raise(offsets):
    with pytest.raises(ValueError, match="greater than 10"):
        frequency_grid(make_hydrogen_atom(3), line_wing_offsets=offsets)
