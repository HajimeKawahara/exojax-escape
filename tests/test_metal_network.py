"""Shared-field metal conservation, chemistry feedback and thermal derivatives."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.metal_network import MetalNetwork, metal_frequency_edges
from escape.hydrogen_continuum import planck_intensity_hz

jax.config.update("jax_enable_x64", True)


def setup_model():
    # A smooth continuum control; real stellar line centers are supplied separately.
    nu = jnp.geomspace(2e13, 3e16, 90)
    weight = jnp.gradient(nu)
    model = MetalNetwork(
        nu, weight, jnp.array([1e6, 2e6]), atoms=(("C I", 6, 0), ("C II", 6, 1))
    )
    temperature = jnp.array([6000.0, 9000.0])
    abundance = jnp.array([[1e7], [2e7]])
    state = model.initialize(temperature, abundance)
    radiation = dict(
        mean_intensity=jnp.broadcast_to(
            0.02 * planck_intensity_hz(nu, 10000.0), (2, len(nu))
        ),
        pump_mean_intensity=jnp.broadcast_to(
            0.02 * planck_intensity_hz(nu, 10000.0), (2, len(nu))
        ),
        continuum_opacity_cm1=jnp.full((2, len(nu)), 1e-9),
        ots_photon_flux_cm2_s=jnp.zeros((2, 1)),
    )
    return model, temperature, abundance, state, radiation


def evaluate(model, t, abundance, state, radiation, **kwargs):
    return model.evaluate(
        t,
        state,
        abundance,
        jnp.array([1e10, 2e10]),
        jnp.array([1e12, 2e12]),
        jnp.array([1e10, 2e10]),
        jnp.array([1e11, 2e11]),
        jnp.array([1e5, 2e5]),
        radiation,
        jnp.array([2.47e15]),
        **kwargs,
    )


def test_charge_and_multiple_excitation_atoms_share_element_total():
    model, t, abundance, state, field = setup_model()
    ions, levels = model.unpack(state, abundance)
    np.testing.assert_allclose(ions.sum(axis=2), abundance, rtol=1e-14)
    np.testing.assert_allclose(levels[0].sum(axis=1), ions[:, 0, 0], rtol=1e-14)
    np.testing.assert_allclose(levels[1].sum(axis=1), ions[:, 0, 1], rtol=1e-14)
    out = evaluate(model, t, abundance, state, field)
    assert bool(out["valid"])
    assert out["target_state"].shape == state.shape
    assert np.all(np.isfinite(out["target_state"]))
    np.testing.assert_allclose(out["singly_ionized_density_cm3"], ions[:, :, 1])
    assert np.all(out["hydrogen_ct_ionization_s"] >= 0)
    assert np.all(out["helium_ct_recombination_s"] >= 0)
    extra = evaluate(model, t, abundance, state, field, additional_recombination_s=1e4)
    # Extra neutralization increases neutral/singly-ionized equilibrium ratio.
    assert np.all(extra["target_state"][:, 0] > out["target_state"][:, 0])


def test_local_temperature_derivative_reaches_excitation_and_thermal_rates():
    model, t, abundance, state, field = setup_model()

    def function(scale):
        return evaluate(model, t * jnp.exp(scale), abundance, state, field)[
            "energy_erg_cm3_s"
        ]["net_kinetic_heat"].sum()

    derivative = jax.jit(jax.grad(function))(0.0)
    step = 1e-5
    finite = (function(step) - function(-step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite, rtol=3e-6, atol=1e-13)


def test_exact_line_boundary_bypasses_sparse_stellar_grid():
    model, t, abundance, state, field = setup_model()
    kwargs = dict(
        line_incident_mean_intensity=0.02
        * planck_intensity_hz(model.line_frequency_hz, 10000.0),
        line_continuum_opacity_cm1=jnp.full((2, len(model.line_frequency_hz)), 1e-9),
        line_diffuse_mean_intensity=jnp.zeros((2, len(model.line_frequency_hz))),
    )
    first = evaluate(model, t, abundance, state, field, **kwargs)
    altered = dict(field, pump_mean_intensity=field["pump_mean_intensity"] * 100.0)
    second = evaluate(model, t, abundance, state, altered, **kwargs)
    np.testing.assert_allclose(
        first["target_state"], second["target_state"], rtol=1e-14
    )


def test_arbitrary_frequency_continuum_is_positive_and_temperature_dependent():
    model, t, abundance, state, _ = setup_model()
    ions, _ = model.unpack(state, abundance)
    chi, emission = model.continuum_coefficients_at(
        t, ions, jnp.array([1e10, 2e10]), model.frequency_hz
    )
    assert np.all(chi >= 0)
    assert np.all(emission >= 0)
    assert np.any(emission > 0)
    derivative = jax.jacfwd(
        lambda factor: model.continuum_coefficients_at(
            t * factor, ions, jnp.array([1e10, 2e10]), model.frequency_hz
        )[1]
    )(1.0)
    assert np.all(np.isfinite(derivative))
    assert np.any(derivative != 0)
    edges = metal_frequency_edges()
    assert np.all(np.diff(edges) > 0)


def test_recombination_cascade_deposition_conserves_photons_and_energy():
    model, t, abundance, state, _ = setup_model()
    ions, _ = model.unpack(state, abundance)
    electron = jnp.array([1e10, 2e10])
    _, ground = model.continuum_coefficients_at(t, ions, electron, model.frequency_hz)
    _, total = model.continuum_coefficients(t, ions, electron)
    frequency, photons = model.recombination_line_emission(t, ions, electron)
    from escape.metal_atom import PLANCK

    radiation = 4 * jnp.pi * (total - ground) * model.weight_hz
    np.testing.assert_allclose(
        jnp.sum(radiation / (PLANCK * model.frequency_hz), axis=1),
        photons.sum(axis=1),
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        jnp.sum(radiation, axis=1),
        jnp.sum(photons * PLANCK * frequency, axis=1),
        rtol=2e-14,
    )


def test_excited_photoionization_rates_match_the_actual_absorbers():
    nu = jnp.geomspace(1e14, 1e17, 151)
    model = MetalNetwork(
        nu,
        jnp.gradient(nu),
        jnp.array([1e6]),
        atoms=(("N I", 7, 0), ("O I", 8, 0), ("Mg II", 12, 1)),
    )
    t = jnp.array([10000.0])
    abundance = jnp.array([[1e7, 2e7, 3e6]])
    ions, levels = model.unpack(model.initialize(t, abundance), abundance)
    chi, _ = model.continuum_coefficients_at(
        t, ions, jnp.array([1e10]), nu, level_population_cm3=levels
    )
    flux = jnp.ones((1, len(nu))) * 1e12
    base, _ = model._photo_rates_and_heat(flux, nu)
    correction, _, _ = model._excited_photo_correction(ions, levels, flux, nu)
    np.testing.assert_allclose(
        jnp.sum(ions * (base + correction), axis=(1, 2)),
        jnp.sum(chi * flux, axis=1),
        rtol=3e-14,
    )
    assert np.all(chi >= 0)


def test_explicit_double_line_column_changes_escape_closure():
    model, t, abundance, state, field = setup_model()
    doubled = MetalNetwork(
        model.frequency_hz,
        model.weight_hz,
        model.widths_cm,
        atoms=model.atom_specs,
        line_optical_depth_scale=2.0,
    )
    baseline = evaluate(model, t, abundance, state, field)["target_state"]
    changed = evaluate(doubled, t, abundance, state, field)["target_state"]
    assert np.all(np.isfinite(changed))
    assert float(jnp.max(jnp.abs(changed - baseline))) > 1e-6


def test_calcium_lyalpha_absorption_and_photo_rates_count_the_same_photons():
    from escape.metal_atom import C, PLANCK

    nu = jnp.geomspace(2e13, 3e16, 90)
    model = MetalNetwork(
        nu, jnp.gradient(nu), jnp.array([1e6, 2e6]), atoms=(("Ca II", 20, 1),)
    )
    temperature = jnp.array([6000.0, 9000.0])
    abundance = jnp.array([[1e7], [2e7]])
    state = model.initialize(temperature, abundance)
    ions, levels = model.unpack(state, abundance)
    electron = jnp.array([1e10, 2e10])
    line_frequency = jnp.array([2.47e15])
    ordinary, _ = model.continuum_coefficients_at(
        temperature, ions, electron, line_frequency, level_population_cm3=levels
    )
    total = model.hydrogen_line_opacity_at(
        temperature, ions, electron, line_frequency, levels
    )
    field = dict(
        mean_intensity=jnp.zeros((2, len(nu))),
        pump_mean_intensity=jnp.zeros((2, len(nu))),
        continuum_opacity_cm1=jnp.ones((2, len(nu))) * 1e-9,
        ots_photon_flux_cm2_s=jnp.zeros((2, 1)),
    )
    before = evaluate(model, temperature, abundance, state, field)
    flux = jnp.array([[1e18], [2e18]])
    after = evaluate(model, temperature, abundance, state,
                     dict(field, ots_photon_flux_cm2_s=flux))
    events = ((total - ordinary) * flux).sum(axis=1)
    np.testing.assert_allclose(
        after["calcium_lyalpha_photoionization_events_cm3_s"], events, rtol=2e-14
    )
    np.testing.assert_allclose(
        (after["photoionization_s"] - before["photoionization_s"])[:, 0, 1]
        * ions[:, 0, 1], events, rtol=2e-14,
    )
    _, ordinary_heat = model._photo_rates_and_heat(flux, line_frequency)
    sigma = jnp.array([4.8e-18, 7e-18, 8.4e-19, 1.7e-18])
    threshold = model.threshold_erg[0, 1] - PLANCK * C * model.atoms[0].energy_wavenumber[1:5]
    calcium_heat = jnp.sum(levels[0][:, 1:5] * flux * sigma
                           * (PLANCK * line_frequency[0] - threshold), axis=1)
    np.testing.assert_allclose(
        after["energy_erg_cm3_s"]["photoelectron_heat"]
        - before["energy_erg_cm3_s"]["photoelectron_heat"],
        jnp.sum(ions * ordinary_heat, axis=(1, 2)) + calcium_heat, rtol=2e-14,
    )
    assert np.all(after["target_state"][:, 1] > before["target_state"][:, 1])


def test_group_override_replaces_isolated_radiation_and_its_destroyed_photons():
    model, temperature, abundance, state, field = setup_model()
    replacements = {}
    for spec, line in zip(model.atom_specs, model.lines):
        indices = jnp.arange(len(line[0]))
        zero = jnp.zeros((2, len(indices)))
        replacements[spec[0]] = (indices, zero, zero, zero)
    def operator(scale):
        dynamic = {label: (indices, up + scale, down + scale, photons)
                   for label, (indices, up, down, photons) in replacements.items()}
        return evaluate(model, temperature, abundance, state, field,
                        line_rate_overrides=dynamic)

    result = jax.jit(operator)(0.0)
    # Removing all radiative excitation leaves collision-only LTE, even though
    # the supplied continuum still changes the separately normalized ion ladder.
    for block in model.excitation_slices:
        np.testing.assert_allclose(result["target_state"][:, block], state[:, block],
                                   rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(result["ots_photon_flux_cm2_s"], 0.0)
    assert bool(result["valid"])


def test_signed_group_field_correction_does_not_count_continuum_twice():
    from escape.metal_atom import PLANCK

    model, temperature, abundance, state, field = setup_model()
    baseline = evaluate(model, temperature, abundance, state, field)
    extra = 0.2 * field["mean_intensity"]
    adjusted = dict(field, mean_intensity=field["mean_intensity"] + extra)
    correction = -4 * jnp.pi * extra * model.weight_hz / (PLANCK * model.frequency_hz)
    result = evaluate(
        model, temperature, abundance, state, adjusted,
        additional_photon_frequency_hz=model.frequency_hz,
        additional_photon_flux_cm2_s=correction,
    )
    np.testing.assert_allclose(result["target_state"], baseline["target_state"],
                               rtol=3e-14, atol=2e-14)
    np.testing.assert_allclose(result["energy_erg_cm3_s"]["net_kinetic_heat"],
                               baseline["energy_erg_cm3_s"]["net_kinetic_heat"],
                               rtol=3e-14)


def test_oxygen_group_preserves_fine_structure_centers_and_natural_widths():
    frequency = jnp.geomspace(1e14, 1e17, 30)
    model = MetalNetwork(frequency, jnp.gradient(frequency), jnp.array([1e6]),
                         atoms=(("O I", 8, 0),))
    group = model.oxygen_lybeta_group_metadata()
    np.testing.assert_array_equal(group["lower"], [0, 1, 2, 0, 1, 0])
    np.testing.assert_array_equal(group["upper"], [20, 20, 20, 21, 21, 22])
    assert len(np.unique(group["line_frequency_hz"])) == 6
    np.testing.assert_array_equal(group["line_frequency_hz"],
                                   model.line_frequency_hz[group["line_indices"]])
    assert np.all(group["damping_hz"] > group["einstein_a_s"] / (4 * jnp.pi))


def test_restricted_cascade_grid_accounts_only_for_its_declared_band():
    from escape.metal_atom import PLANCK

    model, temperature, abundance, state, _ = setup_model()
    ions, _ = model.unpack(state, abundance)
    electron = jnp.array([1e10, 2e10])
    frequency, photons = model.recombination_line_emission(temperature, ions, electron)
    center = float(frequency[0])
    grid = jnp.linspace(center * .99, center * 1.01, 11)
    weight = jnp.gradient(grid)
    emission = model.recombination_emissivity(
        temperature, ions, electron, grid, weight, partial_coverage=True)
    selected = (frequency >= grid[0]) & (frequency <= grid[-1])
    power = 4 * jnp.pi * emission * weight
    np.testing.assert_allclose(jnp.sum(power / (PLANCK * grid), axis=1),
                               jnp.sum(photons[:, selected], axis=1), rtol=2e-14)
    np.testing.assert_allclose(power.sum(axis=1),
                               jnp.sum(photons[:, selected] * PLANCK * frequency[selected], axis=1),
                               rtol=2e-14)
    assert np.any(np.isnan(model.recombination_emissivity(
        temperature, ions, electron, grid, weight)))
