"""Physical limits and temperature sensitivity of an explicit IR continuum."""

import jax
import numpy as np
import pytest
from scipy.constants import Boltzmann, c, h

from escape.stellar_radiation import matched_photospheric_ir_flux

jax.config.update("jax_enable_x64", True)


def planck_surface_flux_cgs(frequency, temperature):
    # SI Planck radiance converted from W/m^2 to erg/s/cm^2, then hemisphere.
    return 1e3 * np.pi * 2 * h * frequency**3 / c**2 / np.expm1(h * frequency / (Boltzmann * temperature))


def test_planck_anchor_recovers_the_full_long_wavelength_spectrum():
    frequency = np.geomspace(1e10, 5.45e13, 200)
    temperature = 10000.
    flux = planck_surface_flux_cgs(frequency[-1], temperature)
    actual = matched_photospheric_ir_flux(frequency, frequency[-1], flux, temperature)
    np.testing.assert_allclose(actual, planck_surface_flux_cgs(frequency, temperature), rtol=3e-13)
    np.testing.assert_allclose(actual[-1], flux, rtol=2e-16)
    np.testing.assert_allclose(actual[0]/frequency[0]**2,
                               actual[1]/frequency[1]**2, rtol=2e-6)


def test_matched_flux_temperature_derivative_and_invalid_domain():
    def flux(temperature):
        return matched_photospheric_ir_flux(1e13, 5.45e13, 1e-5, temperature)

    finite = (flux(10001.) - flux(9999.)) / 2
    np.testing.assert_allclose(jax.grad(flux)(10000.), finite, rtol=3e-8)
    assert np.isnan(matched_photospheric_ir_flux(6e13, 5.45e13, 1e-5, 10000.))
    assert np.isnan(matched_photospheric_ir_flux(1e13, 5.45e13, 1e-5, -1.))
    with pytest.raises(ValueError, match="scalars"):
        matched_photospheric_ir_flux(1e13, [5.45e13], 1e-5, 10000.)
