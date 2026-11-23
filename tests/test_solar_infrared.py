"""Fixed IR line replacement through a synthetic, complete composition state."""
from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.solar import SolarCompositionModel

jax.config.update("jax_enable_x64", True)


def test_fixed_infrared_groups_replace_hydrogen_rates_and_preserve_ots_accounting():
    model = SolarCompositionModel(
        [0., 1e5, 3e5], nmax=15, quadrature_order=2, angle_order=2,
        atoms=(("C I", 6, 0),), oxygen_overlap=False,
        core_escape_model="continuous", shielding_model="continuous_core",
        resolved_infrared_lines=((14, 10), (14, 11), (15, 8)),
        infrared_options=dict(quadrature_order=2, angle_order=4))
    baseline = copy(model)
    baseline.infrared_groups, baseline.infrared_indices = [], []
    temperature, density = jnp.array([8000., 8200.]), jnp.array([1e11, 2e11])
    state = model.initialize(density, temperature)
    incoming = model.hydrogen.boundary_blackbody(10000., .025)

    def evaluate(provider):
        _, p = provider.evaluate(state, density, temperature, incoming)
        return dict(valid=p["valid"], rates=p["radiation"]["rates"].radiative_bound_bound_s,
                    ots=p["radiation"]["ots_photon_flux_cm2_s"],
                    infrared=p["infrared"], thermal=p["thermal_net"], ledger=p["thermal_ledger"])
    enabled = jax.jit(lambda: evaluate(model))()
    disabled = jax.jit(lambda: evaluate(baseline))()
    assert enabled["valid"] and disabled["valid"]
    h = model.hydrogen.atom
    for index, group in zip(model.infrared_indices, enabled["infrared"]):
        lower, upper = h.line_lower[index], h.line_upper[index]
        np.testing.assert_allclose(enabled["rates"][:, lower, upper], group["pump_up_s"][:, 0])
        np.testing.assert_allclose(enabled["rates"][:, upper, lower], group["radiative_down_s"][:, 0])
        np.testing.assert_array_equal(enabled["ots"][:, index], 0.)
        assert group["background_linear_residual"] < 1e-10
    other = np.ones(len(h.line_lower), dtype=bool)
    other[model.infrared_indices] = False
    np.testing.assert_array_equal(enabled["ots"][:, other], disabled["ots"][:, other])
    np.testing.assert_allclose(enabled["thermal"], sum(enabled["ledger"].values()))
    assert "infrared_line_redistribution" in enabled["ledger"]
    assert all("scattering" not in key for key in enabled["ledger"])
    assert np.max(np.abs(enabled["rates"]-disabled["rates"])) > 0


def test_unavailable_or_unverified_infrared_selections_are_rejected():
    with pytest.raises(ValueError, match="retained"):
        SolarCompositionModel([0., 1e5], nmax=3, atoms=(("C I", 6, 0),),
                              resolved_infrared_lines=((14, 11),))
    with pytest.raises(ValueError, match="distinct IR"):
        SolarCompositionModel([0., 1e5], nmax=3, atoms=(("C I", 6, 0),),
                              resolved_infrared_lines=((3, 2),))
