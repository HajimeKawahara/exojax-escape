"""Conservation laws and thermal/radiative limits of the population closure."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from exojax.utils.constants import hcperk
from escapeh.hydrogen_populations import (
    diluted_blackbody_occupation,
    hydrogen_nuclei_density,
    two_level_ratio,
)
from escapeh.parker import PROTON_MASS_CGS, parker_velocity

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def transition():
    return {
        "wavenumber_cm": 15233.146,
        "einstein_a": 4.4101e7,
        "g_lower": 8.0,
        "g_upper": 18.0,
        "collision_strength": 10.0,
    }


def test_parker_density_conserves_mass_through_every_shell():
    radius = np.geomspace(1.4e10, 1.4e11, 23)
    velocity = parker_velocity(radius, 12000.0, 2.17, 0.6)
    mass_loss = 3.0e13
    number_density = hydrogen_nuclei_density(radius, velocity, mass_loss)
    flux = 4 * np.pi * radius**2 * number_density * PROTON_MASS_CGS
    np.testing.assert_allclose(flux * velocity * 1e5, mass_loss, rtol=2e-14)
    assert np.all(np.diff(number_density) < 0)


def test_collision_dominated_populations_approach_boltzmann(transition):
    temperature = np.array([3000.0, 8000.0, 20000.0])
    ratio = two_level_ratio(temperature, 1e24, **transition, photon_occupation=0.0)
    boltzmann = 18 / 8 * np.exp(-hcperk * transition["wavenumber_cm"] / temperature)
    np.testing.assert_allclose(ratio, boltzmann, rtol=2e-9)


def test_zero_electron_density_has_the_einstein_radiation_limit(transition):
    occupation = np.array([0.0, 0.001, 0.1, 2.0])
    ratio = two_level_ratio(9000.0, 0.0, **transition, photon_occupation=occupation)
    expected = 18 / 8 * occupation / (1 + occupation)
    np.testing.assert_allclose(ratio, expected, rtol=2e-14, atol=0)


def test_planck_field_at_gas_temperature_recovers_lte_at_any_density(transition):
    temperature = np.array([3500.0, 9000.0, 20000.0])[:, None]
    electron_density = np.array([0.0, 1e7, 1e13, 1e21])[None, :]
    occupation = diluted_blackbody_occupation(
        transition["wavenumber_cm"], temperature, 1.0,
    )
    ratio = two_level_ratio(
        temperature, electron_density, **transition, photon_occupation=occupation,
    )
    boltzmann = 18 / 8 * np.exp(-hcperk * transition["wavenumber_cm"] / temperature)
    np.testing.assert_allclose(ratio, np.broadcast_to(boltzmann, ratio.shape), rtol=2e-14)


def test_unpumped_collisionless_gas_has_no_upper_population(transition):
    occupation = diluted_blackbody_occupation(transition["wavenumber_cm"], 10000.0, 0.0)
    ratio = two_level_ratio(12000.0, 0.0, **transition, photon_occupation=occupation)
    assert float(ratio) == 0.0


def test_positive_rates_never_invert_the_two_levels(transition):
    temperature = np.geomspace(100.0, 1e6, 21)[:, None]
    electron_density = np.geomspace(1e2, 1e24, 19)[None, :]
    for occupation in (0.0, 0.01, 10.0):
        ratio = np.asarray(two_level_ratio(
            temperature, electron_density, **transition, photon_occupation=occupation,
        ))
        assert np.isfinite(ratio).all()
        assert np.all(ratio >= 0.0)
        assert np.all(ratio < 18 / 8)
