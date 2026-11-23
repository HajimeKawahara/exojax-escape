"""Variable plasma, shared fields, photon feedback and implicit derivatives."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
from escape.hydrogen_mixing import mixing_bound_bound_cm3_s

jax.config.update("jax_enable_x64", True)


def _state():
    model = HydrogenEscapeProbabilityModel(
        [0.0, 1e4, 3e4], nmax=3, quadrature_order=2, angle_order=2
    )
    bound = jnp.array(
        [[7e11, 1e5, 3e5, 1e3, 3e3, 5e3], [8e11, 2e5, 6e5, 2e3, 6e3, 1e4]]
    )
    return (
        model,
        bound,
        jnp.array([8000.0, 9000.0]),
        jnp.array([3e11, 2e11]),
        jnp.array([3.1e11, 2.1e11]),
    )


def test_shared_continuum_uses_variable_temperature_and_actual_proton_collider():
    model, bound, temperature, proton, electron = _state()
    run = jax.jit(model.population_state)
    incoming = model.boundary_blackbody(10000.0, 0.025)
    state = run(temperature, bound, proton, electron, incoming)
    assert state["converged"]
    expected_mixing = jax.vmap(
        lambda t, ne: mixing_bound_bound_cm3_s(model.atom, t, ne)
    )(temperature, electron)
    np.testing.assert_allclose(
        electron * state["rates"].electron_bound_bound_cm3_s[:, 1, 2],
        proton * expected_mixing[:, 1, 2],
        rtol=3e-15,
    )
    opacity, emissivity = model.continuum_coefficients_at(
        temperature, bound, proton, electron, model.grid.frequency_hz
    )
    np.testing.assert_allclose(opacity, state["continuum_opacity_cm1"], rtol=2e-14)
    np.testing.assert_allclose(
        emissivity + state["two_photon_emission"],
        state["continuum_emission"],
        rtol=2e-14,
    )
    shaded = run(
        temperature,
        bound,
        proton,
        electron,
        incoming,
        additional_opacity=jnp.full_like(opacity, 1e-4),
    )
    assert np.all(
        shaded["rates"].photoionization_s[:, 0] < state["rates"].photoionization_s[:, 0]
    )
    assert np.isfinite(state["thermal_net_erg_cm3_s"]).all()
    np.testing.assert_allclose(
        state["thermal_net_erg_cm3_s"],
        state["thermal_heating_erg_cm3_s"] - state["thermal_cooling_erg_cm3_s"],
    )


def test_explicit_narrow_stellar_line_boundary_changes_only_selected_pump():
    model, bound, temperature, proton, electron = _state()
    run = jax.jit(model.population_state)
    incoming = model.boundary_blackbody(10000.0, 0.025)
    line_incoming = jnp.interp(
        model.atom.line_frequency_hz, model.grid.frequency_hz, incoming
    )
    baseline = run(
        temperature,
        bound,
        proton,
        electron,
        incoming,
        incident_line_mean_intensity=line_incoming,
    )
    dark_lyman_beta = line_incoming.at[1].set(0.0)
    shaded = run(
        temperature,
        bound,
        proton,
        electron,
        incoming,
        incident_line_mean_intensity=dark_lyman_beta,
    )
    assert np.all(
        shaded["line_operators"].pump_up_s[:, 1]
        < baseline["line_operators"].pump_up_s[:, 1]
    )
    mask = np.arange(len(line_incoming)) != 1
    np.testing.assert_allclose(
        shaded["line_operators"].pump_up_s[:, mask],
        baseline["line_operators"].pump_up_s[:, mask],
        rtol=1e-14,
    )
    np.testing.assert_array_equal(shaded["mean_intensity"], baseline["mean_intensity"])


def test_population_root_has_implicit_temperature_gradient():
    model, bound, temperature, proton, electron = _state()
    initial = jnp.log(bound / proton[:, None])
    incident = model.boundary_blackbody(10000.0, 0.025)

    @jax.jit
    def signal(log_temperature):
        solution = model.solve(
            1e12, temperature * jnp.exp(log_temperature), incident, initial
        )
        return jnp.log(jnp.mean(solution["bound_cm3"][:, 3:6].sum(axis=1))), solution[
            "residual_norm"
        ]

    value, residual = signal(0.0)
    assert residual < 1e-8 and np.isfinite(value)
    automatic = jax.grad(lambda t: signal(t)[0])(0.0)
    step = 1e-4
    finite = (signal(step)[0] - signal(-step)[0]) / (2 * step)
    assert np.isfinite(automatic) and abs(finite) > 0.01
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-7)


def test_external_ots_rates_and_heat_preserve_photon_energy():
    from escape.hydrogen_atom import atom_cross_sections, atom_milne_population
    from escape.hydrogen_rates import PLANCK_CGS

    model, bound, temperature, proton, electron = _state()
    model.include_free_free = False
    frequency = jnp.array([5e14, 2.4e15, 3.5e15])
    flux = jnp.array([[1e19, 2e19, 3e19], [3e19, 2e19, 1e19]])
    events = jax.jit(model.photoionization_from_photon_flux)(
        temperature, bound, proton, electron, frequency, flux
    )
    net_ionizations = (
        bound * events["photoionization_s"]
        - (electron * proton)[:, None] * events["recombination_cm3_s"]
    )
    latent_energy = net_ionizations @ model.atom.ionization_energy_erg
    inverse = jax.vmap(lambda t: atom_milne_population(model.atom, t, frequency))(
        temperature
    )
    net_absorption = jnp.sum(
        atom_cross_sections(model.atom, frequency)
        * (bound[:, :, None] - (electron * proton)[:, None, None] * inverse),
        axis=1,
    )
    photon_energy = jnp.sum(net_absorption * flux * PLANCK_CGS * frequency, axis=1)
    np.testing.assert_allclose(
        events["net_erg_cm3_s"] + latent_energy, photon_energy, rtol=3e-14
    )
