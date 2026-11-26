"""Actual H-alpha rate, continuum-receiver and heat replacement controls."""

from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_rates import PLANCK_CGS
from escape.solar import SolarCompositionModel

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def halpha_case():
    model = SolarCompositionModel(
        [1e5, 2e5, 4e5], nmax=3, quadrature_order=2, angle_order=2,
        atoms=(("C I", 6, 0),), oxygen_overlap=False,
        core_escape_model="continuous", shielding_model="continuous_core",
        halpha_transfer="spherical", halpha_options=dict(quadrature_order=2, angle_order=32))
    baseline = copy(model)
    baseline.halpha_group = None
    temperature, density = jnp.array([8000., 8200.]), jnp.array([1e11, 2e11])
    state = model.initialize(density, temperature)
    incoming = model.hydrogen.boundary_blackbody(10000., .025)
    call = jax.jit(lambda strength: model.evaluate(
        state, density, temperature, incoming, halpha_blend=strength))
    results = [call(strength) for strength in (0., .5, 1.)]
    native = jax.jit(lambda: baseline.evaluate(state, density, temperature, incoming))()
    return model, temperature, native, results


def test_zero_strength_recovers_native_residual_rates_and_heat(halpha_case):
    _, _, (native_target, native), results = halpha_case
    zero_target, zero = results[0]
    assert native["valid"] and zero["valid"]
    np.testing.assert_allclose(zero_target, native_target, rtol=2e-13, atol=2e-13)
    for key in ("thermal_net", "thermal_scale"):
        np.testing.assert_allclose(zero[key], native[key], rtol=2e-13)
    for key, value in native["thermal_ledger"].items():
        np.testing.assert_allclose(zero["thermal_ledger"][key], value, rtol=2e-13)
    np.testing.assert_array_equal(zero["thermal_ledger"]["halpha_line_redistribution"], 0.)
    for key in ("ots_photon_flux_cm2_s", "line_rate_override_mask"):
        np.testing.assert_array_equal(zero["radiation"][key], native["radiation"][key])
    for actual, expected in zip(zero["hydrogen_complete_rates"], native["hydrogen_complete_rates"]):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-30)


def test_halpha_rates_ots_and_thermal_use_the_same_homotopy(halpha_case):
    model, _, (_, native), results = halpha_case
    atom, indices = model.hydrogen.atom, model.halpha_indices
    lower, upper = atom.line_lower[indices], atom.line_upper[indices]
    for strength, (_, result) in zip((0., .5, 1.), results):
        assert result["valid"]
        rates = result["radiation"]["rates"].radiative_bound_bound_s
        old = native["radiation"]["rates"].radiative_bound_bound_s
        group = result["halpha"]
        np.testing.assert_allclose(rates[:, lower, upper],
            (1-strength)*old[:, lower, upper]+strength*group["pump_up_s"], rtol=2e-13)
        np.testing.assert_allclose(rates[:, upper, lower],
            (1-strength)*old[:, upper, lower]+strength*group["radiative_down_s"], rtol=2e-13)
        np.testing.assert_allclose(result["radiation"]["ots_photon_flux_cm2_s"][:, indices],
            (1-strength)*native["radiation"]["ots_photon_flux_cm2_s"][:, indices], rtol=2e-13)
        np.testing.assert_allclose(result["thermal_net"], sum(result["thermal_ledger"].values()), rtol=2e-13)
    for key in native["thermal_ledger"]:
        np.testing.assert_allclose(results[1][1]["thermal_ledger"][key],
            (results[0][1]["thermal_ledger"][key]+results[2][1]["thermal_ledger"][key])/2,
            rtol=2e-12, atol=1e-28)


def test_full_field_packet_matches_line_rates_and_hydrogen_receiver(halpha_case):
    model, temperature, (_, native), results = halpha_case
    result = results[-1][1]
    group = result["halpha"]
    frequency, weight = model.halpha_group.frequency_hz, model.halpha_group.weight_hz
    photon_weight = 4*jnp.pi*weight/(PLANCK_CGS*frequency)
    packet = group["coupled_photon_flux_cm2_s"]
    np.testing.assert_allclose(group["represented_background_mean_intensity"]+packet/photon_weight,
        group["transfer"].mean_intensity, rtol=3e-13, atol=1e-30)
    np.testing.assert_allclose(packet,
        group["line_only_photon_flux_cm2_s"]+group["background_geometry_photon_flux_cm2_s"], rtol=2e-13)
    assert np.max(np.abs(group["background_geometry_photon_flux_cm2_s"])) > 0
    incoming = model.hydrogen.photoionization_from_photon_flux(temperature,
        result["bound_cm3"], result["proton_cm3"], result["electron_cm3"], frequency, packet)
    # Metal OTS photons are unchanged at the same frozen composition, while
    # the H-alpha packet enters the final H rates and its kinetic ledger once.
    np.testing.assert_allclose(result["hydrogen_receiving"]["photoionization_s"],
        native["hydrogen_receiving"]["photoionization_s"]+incoming["photoionization_s"], rtol=2e-12)
    np.testing.assert_allclose(result["hydrogen_receiving"]["net_erg_cm3_s"],
        native["hydrogen_receiving"]["net_erg_cm3_s"]+incoming["net_erg_cm3_s"], rtol=2e-12)
    np.testing.assert_allclose(result["hydrogen_complete_rates"].photoionization_s[:, 1:],
        result["radiation"]["rates"].photoionization_s[:, 1:]
        +result["hydrogen_receiving"]["photoionization_s"][:, 1:], rtol=2e-13)


def test_stellar_boundary_keeps_exact_halpha_samples_and_existing_slices(halpha_case):
    model, _, _, _ = halpha_case
    frequency = np.geomspace(1e10, 1e18, 2000)
    center = float(model.halpha_group.line_frequency_hz[0])
    surface = 1e-4*(1-.6*np.exp(-((frequency/center-1)/.02)**2))
    actual = model.stellar_boundary(frequency, surface, outside_zero=True)
    baseline = copy(model)
    baseline.halpha_group = None
    previous = baseline.stellar_boundary(frequency, surface, outside_zero=True)
    for key, value in previous.items():
        if value is not None:
            np.testing.assert_array_equal(actual[key], value)
    np.testing.assert_allclose(actual["halpha_intensity"],
        np.exp(np.interp(np.log(model.halpha_group.frequency_hz), np.log(frequency), np.log(surface)))
        *.1/(4*np.pi), rtol=2e-13)
