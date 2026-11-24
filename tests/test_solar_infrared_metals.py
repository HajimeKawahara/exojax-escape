"""Reciprocal H/Fe infrared transfer through an actual composition state."""

from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_rates import LIGHT_SPEED_CGS, PLANCK_CGS
from escape.solar import SolarCompositionModel

jax.config.update("jax_enable_x64", True)


def _model(**options):
    return SolarCompositionModel(
        [0.0, 1e5, 3e5], nmax=14, iron_levels=200,
        atoms=(("Fe II", 26, 1),), oxygen_overlap=False,
        quadrature_order=2, angle_order=2,
        core_escape_model="continuous", shielding_model="continuous_core",
        resolved_infrared_lines=((14, 7),),
        infrared_options=dict(quadrature_order=2, angle_order=4),
        **options,
    )


@pytest.fixture(scope="module")
def infrared_case():
    model = _model(resolved_infrared_metal_lines=(("Fe II", 188, 180),))
    baseline = copy(model)
    baseline.infrared_groups, baseline.infrared_indices = [], []
    temperature = jnp.array([8000.0, 8200.0])
    density = jnp.array([1e11, 2e11])
    state = model.initialize(density, temperature)
    incoming = model.hydrogen.boundary_blackbody(10000.0, 0.025)
    # Inspect arguments while still calling the real metal evaluator. These
    # diagnostics test the Solar boundary rather than replacing atomic physics.
    original = model.metal.evaluate

    def audit_metal(*args, **kwargs):
        result = original(*args, **kwargs)
        result["test_line_overrides"] = kwargs["line_rate_overrides"]
        result["test_signed_frequency"] = kwargs["additional_photon_frequency_hz"]
        result["test_signed_flux"] = kwargs["additional_photon_flux_cm2_s"]
        return result

    model.metal.evaluate = audit_metal
    try:
        def call(provider):
            return jax.jit(lambda: provider.evaluate(
                state, density, temperature, incoming))()
        enabled, disabled = call(model), call(baseline)
    finally:
        model.metal.evaluate = original
    return model, state, temperature, density, incoming, enabled, disabled


def test_both_atomic_populations_supply_group_opacity_and_emission(infrared_case):
    model, _, temperature, _, _, (_, physical), _ = infrared_case
    assert physical["valid"]
    assert model.infrared_group_metal_members == [[(0, int(
        np.flatnonzero((np.asarray(model.metal.lines[0][0]) == 188)
                       & (np.asarray(model.metal.lines[0][1]) == 180))[0]))]]
    group = model.infrared_groups[0]
    result = physical["infrared"][0]
    h = model.hydrogen.atom
    selected = model.infrared_indices[0]
    lower = jnp.column_stack((physical["bound_cm3"][:, h.line_lower[selected]],
                              physical["metal_levels_cm3"][0][:, 180]))
    upper = jnp.column_stack((physical["bound_cm3"][:, h.line_upper[selected]],
                              physical["metal_levels_cm3"][0][:, 188]))
    chi, eta, *_ = group.coefficients(temperature, lower, upper)
    # The independent eager reconstruction and fused model differ by rounding.
    np.testing.assert_allclose(result["line_opacity_cm1"], chi, rtol=3e-14, atol=0)
    np.testing.assert_allclose(result["line_emission"], eta, rtol=3e-14, atol=0)
    assert np.all(np.max(abs(chi), axis=(0, 2)) > 0)
    assert np.all(np.max(eta, axis=(0, 2)) > 0)
    # Optical line centers and kinetic atomic gaps are distinct source fields.
    atom = model.metal.atoms[0]
    np.testing.assert_allclose(group.transition_energy_erg[1],
        PLANCK_CGS*LIGHT_SPEED_CGS*(atom.energy_wavenumber[188]
                                  -atom.energy_wavenumber[180]), rtol=2e-15)


def test_both_rate_routes_replace_isolated_ots_once(infrared_case):
    model, _, _, _, _, (target, physical), (old_target, old) = infrared_case
    assert physical["valid"] and old["valid"]
    h, group = model.hydrogen.atom, physical["infrared"][0]
    selected = model.infrared_indices[0]
    lower, upper = h.line_lower[selected], h.line_upper[selected]
    matrix = physical["radiation"]["rates"].radiative_bound_bound_s
    np.testing.assert_array_equal(matrix[:, lower, upper], group["pump_up_s"][:, 0])
    np.testing.assert_array_equal(matrix[:, upper, lower], group["radiative_down_s"][:, 0])
    indices, up, down, photons = physical["metal"]["test_line_overrides"]["Fe II"]
    assert list(indices) == [model.infrared_group_metal_members[0][0][1]]
    np.testing.assert_array_equal(up, group["pump_up_s"][:, 1:])
    np.testing.assert_array_equal(down, group["radiative_down_s"][:, 1:])
    np.testing.assert_array_equal(photons, 0.0)
    for new_flux, old_flux, removed in (
        (physical["radiation"]["ots_photon_flux_cm2_s"],
         old["radiation"]["ots_photon_flux_cm2_s"], selected),
        (physical["metal"]["ots_photon_flux_cm2_s"],
         old["metal"]["ots_photon_flux_cm2_s"], int(indices[0])),
    ):
        np.testing.assert_array_equal(new_flux[:, removed], 0.0)
        keep = np.arange(new_flux.shape[1]) != removed
        np.testing.assert_allclose(new_flux[:, keep], old_flux[:, keep], rtol=3e-14, atol=0)
    assert np.max(abs(target[:, :model.hydrogen_size]
                      -old_target[:, :model.hydrogen_size])) > 0
    assert np.max(abs(target[:, model.metal_slice]-old_target[:, model.metal_slice])) > 0


def test_replacement_removes_an_active_metal_destruction_source(infrared_case):
    model, state, temperature, _, _, (_, physical), _ = infrared_case
    metal = copy(model.metal)
    # A separate, deliberately thick synthetic column makes destruction of
    # this weak transition nonzero. It is not a planetary-atmosphere control.
    metal.widths_cm = jnp.full_like(metal.widths_cm, 1e20)
    index = model.infrared_group_metal_members[0][0][1]
    group = physical["infrared"][0]
    replacement = {"Fe II": (jnp.array([index]), group["pump_up_s"][:, 1:],
        group["radiative_down_s"][:, 1:], jnp.zeros((2, 1)))}

    def evaluate(replace):
        output = metal.evaluate(
            temperature, state[:, model.metal_slice], physical["element_density_cm3"],
            physical["electron_cm3"], physical["bound_cm3"][:, 0],
            physical["proton_cm3"], physical["helium_cm3"][:, 0],
            physical["helium_cm3"][:, 1], physical["radiation"],
            model.hydrogen.atom.line_frequency_hz,
            line_continuum_opacity_cm1=jnp.full((2, len(metal.line_frequency_hz)), 1e-12),
            line_rate_overrides=replacement if replace else None)
        return output["ots_photon_flux_cm2_s"], output["valid"]

    old, old_valid = jax.jit(lambda: evaluate(False))()
    new, new_valid = jax.jit(lambda: evaluate(True))()
    assert old_valid and new_valid
    assert np.all(old[:, index] > 0)
    np.testing.assert_array_equal(new[:, index], 0.0)
    keep = np.arange(new.shape[1]) != index
    np.testing.assert_allclose(new[:, keep], old[:, keep], rtol=3e-14, atol=0)


def test_shared_signed_field_and_kinetic_energy_are_counted_once(infrared_case):
    model, _, temperature, _, _, (_, physical), (_, old) = infrared_case
    group = physical["infrared"][0]
    frequency = model.infrared_groups[0].frequency_hz
    signed = group["additional_photon_flux_cm2_s"]
    np.testing.assert_array_equal(physical["metal"]["test_signed_frequency"], frequency)
    np.testing.assert_array_equal(physical["metal"]["test_signed_flux"], signed)
    # Independently reconstruct the H receiver ledger: one metal OTS stream
    # plus one common group correction, after removing selected isolated OTS.
    extra = model.hydrogen.photoionization_from_photon_flux(
        temperature, physical["bound_cm3"], physical["proton_cm3"],
        physical["electron_cm3"],
        jnp.r_[model.metal.line_frequency_hz, frequency],
        jnp.concatenate((physical["metal"]["ots_photon_flux_cm2_s"], signed), axis=1))
    ledger = physical["thermal_ledger"]
    np.testing.assert_allclose(ledger["hydrogen"],
        physical["radiation"]["thermal_net_erg_cm3_s"]+extra["net_erg_cm3_s"], rtol=3e-14)
    np.testing.assert_array_equal(ledger["infrared_line_redistribution"],
                                  group["redistribution_heat_erg_cm3_s"])
    np.testing.assert_allclose(physical["thermal_net"], sum(ledger.values()), rtol=3e-14)
    assert all("scattering" not in name for name in ledger)
    assert old["infrared"] == []
    assert "infrared_line_redistribution" not in old["thermal_ledger"]
    np.testing.assert_array_equal(physical["radiation"]["mean_intensity"],
                                  old["radiation"]["mean_intensity"])


def test_emission_feedback_is_reciprocal_and_differentiable(infrared_case):
    model, _, temperature, _, incident, (_, physical), _ = infrared_case
    h, group = model.hydrogen.atom, model.infrared_groups[0]
    selected = model.infrared_indices[0]
    lower = jnp.column_stack((physical["bound_cm3"][:, h.line_lower[selected]],
                              physical["metal_levels_cm3"][0][:, 180]))
    upper = jnp.column_stack((physical["bound_cm3"][:, h.line_upper[selected]],
                              physical["metal_levels_cm3"][0][:, 188]))
    chi, eta = model._infrared_coefficients_at(
        temperature, physical, group.frequency_hz, group.weight_hz)
    boundary = jnp.interp(group.frequency_hz, model.grid.frequency_hz, incident)

    def pump(log_upper_scale):
        result = group.solve(
            temperature, lower, upper*jnp.exp(log_upper_scale)[None, :],
            chi, eta, boundary,
            scattering_opacity_cm1=physical["electron_cm3"][:, None]*6.6524587321e-25)
        return result["pump_up_s"]

    derivative = jax.jit(jax.jacfwd(pump))(jnp.zeros(2))
    assert np.isfinite(derivative).all()
    # Altering each emitter must change the other atom's excitation, not just
    # its own source. This exercises the shared transfer with physical lines.
    assert np.max(abs(derivative[:, 0, 1])) > 0
    assert np.max(abs(derivative[:, 1, 0])) > 0


@pytest.mark.parametrize("selection,match", [
    ((("Fe II", 188, 180), ("Fe II", 188, 180)), "distinct"),
    ((("Fe II", 200, 180),), "not retained"),
    ((("Fe II", 180, 188),), "zero-based"),
    ((("Fe II", 188.0, 180),), "zero-based"),
    ((("Fe II", True, 0),), "zero-based"),
    ((("Ca II", 188, 180),), "retained metal"),
    ((("Fe II", 1, 0),), "must overlap"),
])
def test_invalid_or_unconnected_metal_selections_fail(selection, match):
    with pytest.raises(ValueError, match=match):
        _model(resolved_infrared_metal_lines=selection)


def test_empty_metal_selection_preserves_hydrogen_group_metadata():
    default, explicit = _model(), _model(resolved_infrared_metal_lines=())
    assert default.infrared_group_metal_members == explicit.infrared_group_metal_members == [[]]
    assert default.infrared_indices == explicit.infrared_indices
    for name in ("frequency_hz", "weight_hz", "line_frequency_hz",
                 "einstein_a_s", "transition_energy_erg"):
        np.testing.assert_array_equal(getattr(default.infrared_groups[0], name),
                                      getattr(explicit.infrared_groups[0], name))
