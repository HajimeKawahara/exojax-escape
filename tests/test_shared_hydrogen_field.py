"""Complete shared continuum replacements reach every hydrogen rate path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
from escape.hydrogen_rates import PLANCK_CGS

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "sampling,quadrature",
    [("front", "gauss"), ("average", "gauss"), ("average", "optical_depth")],
)
def test_complete_shared_mean_replaces_beam_and_reaches_rates_and_heat(
    sampling, quadrature
):
    m = HydrogenEscapeProbabilityModel(
        [0.0, 1e4, 3e4],
        nmax=3,
        quadrature_order=2,
        angle_order=2,
        line_sampling=sampling,
        line_quadrature=quadrature,
        line_quadrature_order=4,
    )
    t = jnp.array([8000.0, 9000.0])
    bound = jnp.array(
        [[7e11, 1e5, 3e5, 1e3, 3e3, 5e3], [8e11, 2e5, 6e5, 2e3, 6e3, 1e4]]
    )
    proton, electron = jnp.array([3e11, 2e11]), jnp.array([3.1e11, 2.1e11])
    incoming = m.boundary_blackbody(10000.0, 0.025)
    run = jax.jit(m.population_state)
    original = run(t, bound, proton, electron, incoming)
    frequency_mask = m.grid.frequency_hz > m.atom.ionization_energy_erg[0] / PLANCK_CGS
    line_mask = jnp.arange(len(m.atom.line_frequency_hz)) == 1
    continuum = 1.2 * original["mean_intensity"]
    line = 0.3 * original["line_pump_mean_intensity"]
    result = run(
        t,
        bound,
        proton,
        electron,
        incoming,
        continuum_mean_intensity_override=(frequency_mask, continuum),
        line_mean_intensity_override=(line_mask, line),
    )
    assert original["converged"] and result["converged"]
    np.testing.assert_array_equal(
        result["mean_intensity"][:, ~frequency_mask],
        original["mean_intensity"][:, ~frequency_mask],
    )
    np.testing.assert_allclose(
        result["line_pump_mean_intensity"][:, line_mask], line[:, line_mask], rtol=3e-15
    )
    uniform = run(
        t,
        bound,
        proton,
        electron,
        incoming,
        line_mean_intensity_override=(line_mask, original["line_pump_mean_intensity"]),
    )
    np.testing.assert_allclose(
        result["line_operators"].pump_up_s[:, line_mask],
        0.3 * uniform["line_operators"].pump_up_s[:, line_mask],
        rtol=3e-14,
    )
    np.testing.assert_array_equal(
        result["line_operators"].pump_up_s[:, ~line_mask],
        original["line_operators"].pump_up_s[:, ~line_mask],
    )
    flux = (
        4
        * jnp.pi
        * (result["mean_intensity"] - original["mean_intensity"])
        * m.grid.weight_hz
        / (PLANCK_CGS * m.grid.frequency_hz)
    )
    added = m.photoionization_from_photon_flux(
        t, bound, proton, electron, m.grid.frequency_hz, flux
    )

    def conserved_increment(actual, original, expected, gross):
        # OTS rates may exceed the changed continuum contribution by many
        # orders. Bound subtraction roundoff using the actual gross rates.
        tolerance = 32 * np.finfo(float).eps * np.asarray(gross) + 2e-12 * abs(
            np.asarray(expected)
        )
        assert np.all(abs(np.asarray(actual - original - expected)) <= tolerance)

    conserved_increment(
        result["rates"].photoionization_s,
        original["rates"].photoionization_s,
        added["photoionization_s"],
        original["rates"].photoionization_s + result["rates"].photoionization_s,
    )
    conserved_increment(
        result["rates"].radiative_recombination_cm3_s,
        original["rates"].radiative_recombination_cm3_s,
        added["recombination_cm3_s"],
        original["rates"].radiative_recombination_cm3_s
        + result["rates"].radiative_recombination_cm3_s,
    )
    conserved_increment(
        result["thermal_net_erg_cm3_s"],
        original["thermal_net_erg_cm3_s"],
        added["net_erg_cm3_s"],
        original["thermal_heating_erg_cm3_s"]
        + original["thermal_cooling_erg_cm3_s"]
        + result["thermal_heating_erg_cm3_s"]
        + result["thermal_cooling_erg_cm3_s"],
    )
    invalid = run(
        t,
        bound,
        proton,
        electron,
        incoming,
        line_mean_intensity_override=(line_mask, line.at[:, 1].set(jnp.nan)),
    )
    assert not invalid["converged"]
