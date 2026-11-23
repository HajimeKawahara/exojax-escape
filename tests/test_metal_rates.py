"""Physical limits and derivatives of the packaged ionic-rate primitives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.metal_rates import (
    KB,
    PLANCK,
    C,
    EV_ERG,
    load_metal_rates,
    photoionization_cross_sections,
    photoionization_rates,
    radiative_recombination,
    raw_dielectronic_recombination,
    dielectronic_suppression,
    three_body_recombination,
    charge_exchange,
    collisional_ionization,
    ground_recombination_continuum,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def data():
    return load_metal_rates()


@pytest.mark.parametrize("z", [6, 7, 8, 12, 13, 14, 20, 25, 26])
def test_ionic_coefficients_have_finite_temperature_and_density_gradients(data, z):
    def function(log_values):
        t, ne = jnp.exp(log_values)
        return jnp.array(
            [
                radiative_recombination(t, z, 1, data),
                raw_dielectronic_recombination(t, z, 1, data)
                * dielectronic_suppression(t, ne, z, 1),
                three_body_recombination(t, ne, 1, data["cota"]),
                collisional_ionization(t, z, 0, data),
                *charge_exchange(t, z, 1, data),
            ]
        )

    point = jnp.log(jnp.array([8000.0, 1e10]))
    values = function(point)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)
    derivative = jax.jacfwd(function)(point)
    finite = jnp.column_stack(
        [
            (
                function(point + jnp.eye(2)[i] * 1e-5)
                - function(point - jnp.eye(2)[i] * 1e-5)
            )
            / 2e-5
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(derivative, finite, rtol=2e-7, atol=1e-20)


def test_ground_photoionization_threshold_and_photon_linearity(data):
    fit = data["photo"][26, 0]
    threshold = fit.threshold_ev[fit.outer_shell - 1]
    energy = jnp.linspace(threshold * 1.01, threshold * 2.0, 256)
    cross = photoionization_cross_sections(fit, energy)
    assert np.all(cross >= 0)
    assert np.all(photoionization_cross_sections(fit, threshold * 0.99) == 0)
    rate, heat = photoionization_rates(fit, energy, jnp.ones_like(energy))
    double_rate, double_heat = photoionization_rates(
        fit, energy, 2 * jnp.ones_like(energy)
    )
    np.testing.assert_allclose(double_rate, 2 * rate)
    np.testing.assert_allclose(double_heat, 2 * heat)
    assert np.sum(heat) > 0


def test_ground_milne_emission_has_correct_saha_boltzmann_balance(data):
    fit = data["photo"][12, 0]
    ratio = np.prod(data["ground_weight"][12, 0] ** np.array([1.0, -1.0]))
    temperature, ne, ion = 8000.0, 1e10, 1e8
    nu = (
        jnp.linspace(float(fit.threshold_ev[fit.outer_shell - 1]) * 1.01, 20.0, 30)
        * EV_ERG
        / PLANCK
    )
    emissivity = ground_recombination_continuum(fit, ratio, temperature, ne, ion, nu)
    cross = photoionization_cross_sections(fit, PLANCK * nu / EV_ERG)[
        :, fit.outer_shell - 1
    ]
    saha = (
        ratio
        / 2
        * (PLANCK**2 / (2 * jnp.pi * 9.1093837015e-28 * KB * temperature)) ** 1.5
        * jnp.exp(fit.threshold_ev[fit.outer_shell - 1] * EV_ERG / (KB * temperature))
    )
    opacity = ne * ion * saha * cross * (-jnp.expm1(-PLANCK * nu / (KB * temperature)))
    planck = 2 * PLANCK * nu**3 / C**2 / jnp.expm1(PLANCK * nu / (KB * temperature))
    np.testing.assert_allclose(emissivity, opacity * planck, rtol=3e-14)


def test_helium_ground_rates_and_manganese_override(data):
    for q in (0, 1):
        assert float(collisional_ionization(8000.0, 2, q, data)) >= 0
        assert float(radiative_recombination(8000.0, 2, q + 1, data)) > 0
        assert np.all(np.array(charge_exchange(8000.0, 2, q + 1, data)) >= 0)
        assert np.all(photoionization_cross_sections(data["photo"][2, q], 100.0) >= 0)
    assert float(charge_exchange(8000.0, 25, 1, data)[0]) == 3e-9


def test_cota_rejects_unvalidated_domain(data):
    assert np.isnan(three_body_recombination(3999.0, 1e10, 1, data["cota"]))
    assert np.isnan(three_body_recombination(8000.0, 1e4, 1, data["cota"]))


def test_coefficients_against_unchanged_cloudy_diagnostic(data):
    from pathlib import Path
    import json

    reference = json.loads(
        (Path(__file__).parent / "data/cloudy_metal_rates.json").read_text()
    )
    for row in reference["cases"]:
        t, z = row["temperature"], row["z"]
        if row["kind"] == "ionic":
            q, ne = row["q"], row["ne"]
            actual = [
                radiative_recombination(t, z, q, data),
                raw_dielectronic_recombination(t, z, q, data)
                * dielectronic_suppression(t, ne, z, q),
                *charge_exchange(t, z, q, data),
            ]
            expected = [
                row["expected"][key]
                for key in ("RR", "DR", "Hion", "Hrec", "Heion", "Herec")
            ]
        elif row["kind"] == "helium_ground":
            actual = [collisional_ionization(t, 2, q, data) for q in (0, 1)]
            actual += [
                value for q in (1, 2) for value in charge_exchange(t, 2, q, data)[:2]
            ]
            expected = row["expected"]
        else:
            actual = [collisional_ionization(t, z, q, data) for q in range(3)]
            expected = row["expected"]
        np.testing.assert_allclose(
            actual, expected, rtol=8e-13, atol=0, err_msg=f"{row['kind']} Z={z}, T={t}"
        )


def test_oxygen_valence_cross_section_against_original_c23_body():
    from pathlib import Path
    from escape.metal_rates import oxygen_ground_valence_cross_section

    with np.load(Path(__file__).parent / "data/cloudy_oxygen_cross.npz") as reference:
        actual = oxygen_ground_valence_cross_section(reference["energy_ev"])
        np.testing.assert_allclose(
            actual, reference["cross_section_cm2"], rtol=6e-8, atol=0
        )
