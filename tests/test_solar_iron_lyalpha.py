"""Shared H/Fe photon exchange in the actual composition evaluation."""
from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_rates import LIGHT_SPEED_CGS, PLANCK_CGS
from escape.solar import SolarCompositionModel

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", params=["gauss", "optical_depth", "optical_depth_metals"])
def exchange_case(request):
    model = SolarCompositionModel(
        [0., 1e8, 2e8], nmax=3, iron_levels=371, atoms=(("Fe II", 26, 1),),
        quadrature_order=2, angle_order=2, line_sampling="average",
        line_quadrature_order=2, line_quadrature=(
            "gauss" if request.param == "gauss" else "optical_depth"),
        metal_line_quadrature=("optical_depth" if request.param == "optical_depth_metals" else "gauss"),
        hydrogen_line_quadrature_order=(4 if request.param != "gauss" else None),
        core_escape_model="continuous", iron_lyalpha=True)
    disabled = copy(model)
    disabled.iron_lyalpha = False
    temperature, density = jnp.array([8000., 8200.]), jnp.array([1e13, 9e12])
    state = model.initialize(density, temperature)
    incident = model.hydrogen.boundary_blackbody(10000., .025)

    def evaluate(provider):
        def call(mask):
            target, physical = provider.evaluate(
                state, density, temperature, incident, atomic_lte_mask=mask)
            return dict(
                target=target, valid=physical["valid"], thermal=physical["thermal_net"],
                hydrogen_rates=physical["radiation"]["rates"].radiative_bound_bound_s,
                hydrogen_ots=physical["radiation"]["ots_photon_flux_cm2_s"],
                metal_ots=physical["metal"]["ots_photon_flux_cm2_s"],
                metal_quadrature_error=physical["metal"]["line_quadrature_normalization_error"],
                exchange=physical["iron_lyalpha_exchange"])
        return jax.jit(call)

    enabled_call, disabled_call = evaluate(model), evaluate(disabled)
    mask = jnp.zeros(2, dtype=bool)
    return (model, model.unpack(state, density), enabled_call, disabled_call,
            enabled_call(mask), disabled_call(mask))


def test_exchange_changes_both_atoms_without_duplicate_continuum_photons(exchange_case):
    model, physical, _, _, on, off = exchange_case
    assert model.metal.line_quadrature_order == 2
    assert model.hydrogen.line_quadrature_order == (
        4 if model.hydrogen.line_quadrature == "optical_depth" else 2)
    assert on["valid"] and off["valid"]
    assert np.all(np.isfinite(on["metal_quadrature_error"]))
    assert np.all(on["metal_quadrature_error"] >= 0)
    exchange = on["exchange"]
    assert exchange["valid"]
    assert np.max(np.abs(exchange["net_photons_cm3_s"])) > 0
    n1, n2p = physical["bound_cm3"][:, 0], physical["bound_cm3"][:, 2]
    hnet = n2p*exchange["hydrogen_downward_s"]-n1*exchange["hydrogen_upward_s"]
    np.testing.assert_allclose(hnet, exchange["net_photons_cm3_s"].sum(1), rtol=3e-14)
    # The added events reach the actual H matrix and Fe SE target.
    np.testing.assert_allclose(
        on["hydrogen_rates"][:, 2, 0]-off["hydrogen_rates"][:, 2, 0],
        exchange["hydrogen_downward_s"], rtol=2e-8, atol=1e-12)
    assert np.max(np.abs(on["target"][:, :model.hydrogen_size]
                         -off["target"][:, :model.hydrogen_size])) > 1e-10
    assert np.max(np.abs(on["target"][:, model.metal_slice]
                         -off["target"][:, model.metal_slice])) > 1e-10
    np.testing.assert_array_equal(on["hydrogen_ots"], off["hydrogen_ots"])
    np.testing.assert_array_equal(on["metal_ots"], off["metal_ots"])


def test_exchange_thermal_ledger_closes_the_actual_atomic_energy_gaps(exchange_case):
    model, _, _, _, on, off = exchange_case
    exchange = on["exchange"]
    atom = model.metal.atoms[model.iron_index]
    upper, lower, *_ = model.metal.lines[model.iron_index]
    fe_gap = PLANCK_CGS*LIGHT_SPEED_CGS*(atom.energy_wavenumber[upper]-atom.energy_wavenumber[lower])
    h_gap = model.hydrogen.atom.excitation_energy_erg[2]-model.hydrogen.atom.excitation_energy_erg[0]
    net = exchange["net_photons_cm3_s"]
    expected = jnp.sum((h_gap-fe_gap)*net, axis=1)
    np.testing.assert_allclose(exchange["redistribution_heat_erg_cm3_s"], expected, rtol=3e-13)
    np.testing.assert_allclose(on["thermal"]-off["thermal"], expected, rtol=2e-7, atol=1e-15)


def test_prescribed_dense_boundary_disables_empirical_exchange_without_nans(exchange_case):
    _, _, enabled, disabled, _, _ = exchange_case
    mask = jnp.ones(2, dtype=bool)
    on, off = enabled(mask), disabled(mask)
    assert on["valid"] and off["valid"]
    np.testing.assert_array_equal(on["exchange"]["net_photons_cm3_s"], 0.)
    np.testing.assert_array_equal(on["exchange"]["hydrogen_downward_s"], 0.)
    np.testing.assert_array_equal(on["target"], off["target"])
    np.testing.assert_array_equal(on["thermal"], off["thermal"])
