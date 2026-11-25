"""Hydrogen completion respects the solar composition and prescribed boundary."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.solar import SolarAtmosphereModel, SolarCompositionModel
from test_hydrogen_forward import offline_nist  # noqa: F401

jax.config.update("jax_enable_x64", True)


def test_prescribed_lte_boundary_keeps_its_finite_atom():
    options = dict(nmax=6, resolved_nmax=5, atoms=(("C I", 6, 0),),
                   quadrature_order=2, angle_order=2, oxygen_overlap=False)
    atmosphere = SolarAtmosphereModel(
        [1., 1.1, 1.5, 3.], hydrogen_topoff=True, dense_lower_cells=1,
        wind_ngrid=33, observer_options=dict(nnu=31, nlon=4, nlat=2), **options)
    h = atmosphere.hydrogen
    np.testing.assert_array_equal(h.topoff_mask, [False, True, True])
    assert atmosphere.observer.atom.resolved_nmax == h.atom.resolved_nmax
    temperature = jnp.array([4500., 8000., 9000.])
    scale = h._bound_free_scale(temperature)
    np.testing.assert_array_equal(scale[0], np.ones(scale.shape[1]))
    assert np.all(np.asarray(scale[1:, -1]) > 1.)
    # This is a declared boundary convention, not a dissolved-atom LTE model.
    with pytest.raises(ValueError, match="set by dense_lower_cells"):
        SolarAtmosphereModel([1., 1.1, 1.5, 3.], hydrogen_topoff=True,
                             hydrogen_topoff_mask=[False, True, True], **options)


def test_active_completion_cannot_overlap_prescribed_lte_cells():
    model = SolarCompositionModel(
        [0., 1e4, 3e4], nmax=6, quadrature_order=2, angle_order=2,
        atoms=(("C I", 6, 0),), oxygen_overlap=False, hydrogen_topoff=True,
        hydrogen_topoff_mask=[False, True],
        core_escape_model="continuous", shielding_model="continuous_core")
    temperature, density = jnp.array([8000., 9000.]), jnp.array([1e10, 2e10])
    state = model.initialize(density, temperature)
    incoming = model.hydrogen.boundary_blackbody(10000., .025)

    @jax.jit
    def evaluate(mask):
        target, physical = model.evaluate(
            state, density, temperature, incoming, atomic_lte_mask=mask)
        return target, physical["thermal_net"], physical["valid"]

    target, heat, valid = evaluate(jnp.array([True, False]))
    assert valid and np.isfinite(target).all() and np.isfinite(heat).all()
    assert not evaluate(jnp.array([True, True]))[-1]
