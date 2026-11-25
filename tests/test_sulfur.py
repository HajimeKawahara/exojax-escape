"""Sulfur coefficients and excitation against an unchanged C23 diagnostic."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from escape.metal_atom import (
    C, KB, PLANCK, electron_collision_coefficients, load_metal_atom, populations,
)
from escape.metal_rates import (
    charge_exchange, dielectronic_suppression, load_metal_partitions,
    load_metal_rates, log_metal_partition, radiative_recombination,
    raw_dielectronic_recombination,
)

jax.config.update("jax_enable_x64", True)


def reference():
    return json.loads((Path(__file__).parent / "data/cloudy_sulfur_rates.json").read_text())


def test_sulfur_ionic_coefficients_match_source_diagnostic():
    data = load_metal_rates()
    for row in reference()["ionic_cases"]:
        t, ne, q = row["temperature"], row["electron_cm3"], row["charge"]
        actual = [radiative_recombination(t, 16, q, data),
                  raw_dielectronic_recombination(t, 16, q, data)
                  * dielectronic_suppression(t, ne, 16, q),
                  *charge_exchange(t, 16, q, data)]
        np.testing.assert_allclose(actual, row["expected"], rtol=3e-12, atol=1e-40)


def test_sulfur_line_strength_conversion_and_partition_anchors():
    atom = load_metal_atom("S II")
    assert len(atom.weight) == 72
    for row in reference()["radiative_cases"]:
        assert float(atom.spontaneous[row["upper"], row["lower"]]) == row["aul_s"]
    partitions = load_metal_partitions()
    for charge, expected in reference()["partition_at_4500K"].items():
        np.testing.assert_allclose(
            jnp.exp(log_metal_partition(4500., 16, int(charge), partitions)),
            expected, rtol=2e-14,
        )


def test_complete_sulfur_table_recovers_collisional_lte():
    atom = load_metal_atom("S II", levels=232)
    temperature = 6500.
    rates = 1e12 * electron_collision_coefficients(atom, temperature)
    actual = populations(rates, 1.)
    expected = atom.weight * jnp.exp(-PLANCK*C*atom.energy_wavenumber/(KB*temperature))
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-30)
    np.testing.assert_allclose(actual.sum(), 1., rtol=3e-15)


def test_sulfur_uses_shared_charge_radiation_and_thermal_network():
    from escape.hydrogen_continuum import planck_intensity_hz
    from escape.metal_network import MetalNetwork

    frequency = jnp.geomspace(2e13, 3e16, 90)
    model = MetalNetwork(frequency, jnp.gradient(frequency), jnp.array([1e5]),
                         atoms=(("S II", 16, 1),))
    t, abundance = jnp.array([8000.]), jnp.array([[1.84e7]])
    state = model.initialize(t, abundance)
    ions, levels = model.unpack(state, abundance)
    np.testing.assert_allclose(ions.sum(axis=2), abundance, rtol=2e-15)
    np.testing.assert_allclose(levels[0].sum(axis=1), ions[:, 0, 1], rtol=2e-15)
    field = .025 * planck_intensity_hz(frequency, 10000.)[None, :]
    radiation = dict(mean_intensity=field, pump_mean_intensity=field,
                     continuum_opacity_cm1=jnp.full_like(field, 1e-9),
                     ots_photon_flux_cm2_s=jnp.zeros((1, 1)))
    result = model.evaluate(t, state, abundance, jnp.array([1e10]),
                            jnp.array([1e12]), jnp.array([1e10]),
                            jnp.array([1e11]), jnp.array([1e5]), radiation,
                            jnp.array([2.47e15]))
    assert bool(result["valid"])
    assert np.all(np.isfinite(result["target_state"]))
    assert np.all(np.isfinite(result["energy_erg_cm3_s"]["net_kinetic_heat"]))
    assert np.all(result["hydrogen_ct_ionization_s"] > 0)
