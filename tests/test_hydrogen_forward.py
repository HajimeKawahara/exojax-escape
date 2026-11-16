"""Physical limits and derivatives of the prescribed H-alpha atmosphere."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from exojax.database.nist import api as nist_api
from hydrogen_forward import HydrogenForwardModel, disk_quadrature

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def offline_nist():
    """NIST unresolved H-alpha row, fetched 2026-11-16; no network in tests.

    Source: https://physics.nist.gov/PhysRefData/ASD/lines_form.html
    H I, n=2 to n=3; energies in cm^-1 and A in s^-1.
    """
    frame = pd.DataFrame({
        "wav": [15233.146], "A": [44101000.0],
        "El": [82259.158], "Eu": [97492.304],
        "gl": [8.0], "gu": [18.0], "jl": [3.5], "ju": [8.5],
        "species": ["H_I"],
    })
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nist_api, "fetch_nist_lines", lambda *args, **kwargs: (frame.copy(), []))
        yield


@pytest.fixture(scope="module")
def model():
    return HydrogenForwardModel(nlayer=32, nlon=8, nlat=4, nnu=401)


def test_zero_column_has_no_line_and_a_finite_derivative(model):
    spectrum = model.prescribed_spectrum(n2_column=0.0)
    np.testing.assert_allclose(spectrum, 0.0, atol=1e-15)
    derivative = jax.grad(
        lambda scale: jnp.sum(model.prescribed_spectrum(n2_column=3e14 * scale))
    )(0.0)
    assert np.isfinite(derivative)
    assert derivative != 0.0


def test_upper_population_reduction_and_outflow_change_line_shape(model):
    velocity = np.asarray(model.velocity_kms)
    center = np.argmin(np.abs(velocity))
    blue = (velocity > -60.0) & (velocity < -5.0)
    red = (velocity > 5.0) & (velocity < 60.0)
    boltzmann = np.asarray(model.prescribed_spectrum(departure_floor=1.0, radial_wind=False))
    static = np.asarray(model.prescribed_spectrum(radial_wind=False))
    outflow = np.asarray(model.prescribed_spectrum(wind_temperature_k=12000.0))

    assert static.shape == velocity.shape
    assert np.isfinite([boltzmann, static, outflow]).all()
    assert boltzmann[center] > 0.95 * boltzmann.max()
    assert static[center] < 0.1 * min(static[blue].max(), static[red].max())
    np.testing.assert_allclose(static[blue].max(), static[red].max(), rtol=2e-3)
    static_ratio = static[red].max() / static[blue].max()
    outflow_ratio = outflow[red].max() / outflow[blue].max()
    assert outflow_ratio > 1.01 * static_ratio


def test_disk_weights_preserve_projected_dayside_area():
    for phase in (0.0, 0.375, 0.5):
        _, weights, *_ = disk_quadrature(phase, nlon=64, nlat=32)
        expected_area = np.pi * (1.0 - np.cos(2.0 * np.pi * phase)) / 2.0
        np.testing.assert_allclose(np.sum(weights), expected_area, rtol=5e-4, atol=1e-15)


def test_parameter_gradients_match_finite_differences():
    model = HydrogenForwardModel(nlayer=16, nlon=6, nlat=4, nnu=201)
    weights = jnp.exp(-0.5 * ((model.velocity_kms - 10.0) / 40.0) ** 2)

    @jax.jit
    def weighted_flux(parameters):
        spectrum = model.prescribed_spectrum(
            n2_column=3e14 * parameters[0],
            departure_floor=parameters[1],
            wind_temperature_k=1e4 * parameters[2],
            mean_molecular_weight=parameters[3],
        )
        return jnp.mean(weights * spectrum)

    parameters = np.array([1.0, 0.08, 1.2, 0.5])
    automatic = np.asarray(jax.jit(jax.grad(weighted_flux))(parameters))
    finite = np.empty(len(parameters))
    for index in range(len(parameters)):
        step = 1e-4 * (1.0 + abs(parameters[index]))
        displacement = np.zeros(len(parameters))
        displacement[index] = step
        finite[index] = (
            weighted_flux(parameters + displacement)
            - weighted_flux(parameters - displacement)
        ) / (2.0 * step)

    assert np.isfinite(automatic).all()
    assert np.all(np.abs(automatic) > 1e-10)
    np.testing.assert_allclose(automatic, finite, rtol=2e-3, atol=1e-10)


def test_rydberg_comparison_preserves_oscillator_strength_and_moves_peak():
    nist = HydrogenForwardModel(nlayer=4, nlon=2, nlat=2, nnu=2401)
    rydberg = HydrogenForwardModel(
        nlayer=4, nlon=2, nlat=2, nnu=2401, wavelength_model="rydberg"
    )
    np.testing.assert_allclose(1e8 / nist.line_center_cm, 6564.63215149, atol=1e-7, rtol=0)
    np.testing.assert_allclose(1e8 / rydberg.line_center_cm, 6564.69606350, atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        nist.adb.A / nist.adb.nu_lines**2, rydberg.adb.A / rydberg.adb.nu_lines**2
    )
    for model in (nist, rydberg):
        np.testing.assert_allclose(model.adb.eupper - model.adb.elower, model.adb.nu_lines)
        profile = np.asarray(model.opa.line_profiles(8000.0))[0]
        peak = model.wavenumber_cm[np.argmax(profile)]
        np.testing.assert_allclose(peak, model.line_center_cm, atol=1e-8, rtol=0)


@pytest.fixture(scope="module")
def rate_model():
    return HydrogenForwardModel(
        temperature_top_k=12000.0, nlayer=16, nlon=6, nlat=4, nnu=201,
    )


def test_rate_profiles_conserve_hydrogen_and_link_charge_to_dynamics(rate_model):
    from hydrogen_forward import PLANET_RADIUS_CM
    from parker import PROTON_MASS_CGS

    mass_loss = 2e13
    radius = np.asarray(rate_model.radius_rp) * PLANET_RADIUS_CM
    for ionized, excited in ((0.0, 0.0), (0.3, 0.2), (0.8, 1.0), (1.0, 0.4)):
        profiles = rate_model.population_profiles(
            mass_loss_g_s=mass_loss, ionized_fraction=ionized,
            excited_fraction=excited, wind_temperature_k=12000.0,
        )
        density = np.asarray(profiles["n_h_cm3"])
        electron_density = np.asarray(profiles["electron_density_cm3"])
        excited_density = np.asarray(profiles["n2_cm3"] + profiles["n3_cm3"])
        neutral_density = (1 - ionized) * density
        np.testing.assert_allclose(electron_density, ionized * density, rtol=2e-14)
        np.testing.assert_allclose(excited_density, excited * neutral_density, rtol=2e-14)
        assert np.all(excited_density <= neutral_density * (1 + 1e-14))
        ground_density = (1 - excited) * neutral_density
        np.testing.assert_allclose(
            ground_density + excited_density + electron_density, density, rtol=2e-14,
        )
        velocity = np.asarray(profiles["radial_velocity_kms"])
        mass_flux = 4 * np.pi * radius**2 * density * PROTON_MASS_CGS * velocity * 1e5
        np.testing.assert_allclose(mass_flux, mass_loss, rtol=2e-14)
        np.testing.assert_allclose(
            velocity, rate_model.radial_velocity(12000.0, 1 / (1 + ionized)), rtol=2e-14,
        )


@pytest.mark.parametrize("parameter,scale", [("mass_loss_g_s", 1e13), ("excited_fraction", 1e-6)])
def test_rate_spectrum_zero_limits_have_finite_derivatives(rate_model, parameter, scale):
    spectrum = rate_model.spectrum(**{parameter: 0.0})
    np.testing.assert_allclose(spectrum, 0.0, atol=1e-15)
    derivative = jax.grad(
        lambda value: jnp.sum(rate_model.spectrum(**{parameter: value * scale}))
    )(0.0)
    assert np.isfinite(derivative)
    assert derivative != 0.0


def test_rate_spectrum_parameter_gradients_match_finite_differences(rate_model):
    weights = jnp.exp(-0.5 * ((rate_model.velocity_kms - 8.0) / 30.0)**2)

    @jax.jit
    def weighted_flux(parameters):
        spectrum = rate_model.spectrum(
            mass_loss_g_s=1e13 * parameters[0],
            excited_fraction=1e-6 * parameters[1],
            ionized_fraction=parameters[2],
            wind_temperature_k=1e4 * parameters[3],
            radiation_dilution=0.025 * parameters[4],
            rotation_kms=6.67, zonal_kms=10.0,
        )
        return jnp.mean(weights * spectrum)

    parameters = np.array([1.0, 1.0, 0.5, 1.2, 1.0])
    automatic = np.asarray(jax.jit(jax.grad(weighted_flux))(parameters))
    finite = np.empty(parameters.size)
    for index in range(parameters.size):
        step = 1e-4 * (1 + abs(parameters[index]))
        displacement = np.zeros_like(parameters)
        displacement[index] = step
        finite[index] = (
            weighted_flux(parameters + displacement) - weighted_flux(parameters - displacement)
        ) / (2 * step)
    assert np.isfinite(automatic).all()
    assert np.all(automatic != 0.0)
    np.testing.assert_allclose(automatic, finite, rtol=2e-3, atol=1e-12)


def test_continuity_profiles_conserve_the_reservoir_and_requested_column(rate_model):
    column = 8.7e13
    profiles = rate_model.continuity_profiles(
        n2_column=column, departure_floor=0.03,
        wind_temperature_k=15000.0, mean_molecular_weight=2 / 3,
        temperature_top_k=15000.0,
    )
    n2, n3 = np.asarray(profiles["n2_cm3"]), np.asarray(profiles["n3_cm3"])
    np.testing.assert_allclose(np.sum(n2 * rate_model.thickness_cm), column, rtol=2e-14)
    reservoir_flux = (n2 + n3) * rate_model.radius_rp**2 * profiles["radial_velocity_kms"]
    np.testing.assert_allclose(reservoir_flux, reservoir_flux[0], rtol=2e-14)
    np.testing.assert_allclose(n3 / n2, profiles["ratio"], rtol=2e-14)
    assert np.all(n2 > 0)
    assert np.all(n3 > 0)


def test_continuity_spectrum_has_finite_zero_column_limit(rate_model):
    assert np.isfinite(rate_model.continuity_spectrum()).all()
    np.testing.assert_allclose(rate_model.continuity_spectrum(n2_column=0.0), 0.0, atol=1e-15)
    derivative = jax.grad(
        lambda scale: jnp.sum(rate_model.continuity_spectrum(n2_column=1e14 * scale))
    )(0.0)
    assert np.isfinite(derivative)
    assert derivative != 0.0


@pytest.fixture(params=["prescribed_spectrum", "continuity_spectrum", "spectrum"])
def temperature_case(request):
    parameters = {"wind_temperature_k": 12000.0, "rotation_kms": 6.67, "zonal_kms": 10.0}
    if request.param == "spectrum":
        parameters.update(mass_loss_g_s=1e14, excited_fraction=1e-7, ionized_fraction=0.5)
    else:
        parameters.update(n2_column=1e14, departure_floor=0.03, mean_molecular_weight=2 / 3)
    return request.param, parameters


def test_temperature_override_matches_a_separately_constructed_model(rate_model, temperature_case):
    method_name, parameters = temperature_case
    warmer = HydrogenForwardModel(
        temperature_top_k=15000.0, nlayer=16, nlon=6, nlat=4, nnu=201,
    )
    spectrum = getattr(rate_model, method_name)
    overridden = spectrum(**parameters, temperature_top_k=15000.0)
    expected = getattr(warmer, method_name)(**parameters)
    np.testing.assert_allclose(overridden, expected, rtol=2e-12, atol=1e-15)
    assert np.max(np.abs(np.asarray(overridden - spectrum(**parameters)))) > 1e-8
    np.testing.assert_allclose(
        rate_model.temperature_profile(15000.0), warmer.temperature_k, rtol=2e-14,
    )


def test_temperature_spectrum_gradient_matches_finite_difference(rate_model, temperature_case):
    method_name, parameters = temperature_case
    spectrum = getattr(rate_model, method_name)
    weights = jnp.exp(-0.5 * ((rate_model.velocity_kms - 8.0) / 30.0)**2)

    @jax.jit
    def weighted_flux(scaled_temperature):
        return jnp.mean(weights * spectrum(
            **parameters, temperature_top_k=1e4 * scaled_temperature,
        ))

    automatic = jax.grad(weighted_flux)(1.5)
    step = 1e-4
    finite = (weighted_flux(1.5 + step) - weighted_flux(1.5 - step)) / (2 * step)
    assert np.isfinite(automatic)
    assert automatic != 0.0
    np.testing.assert_allclose(automatic, finite, rtol=2e-4, atol=1e-13)
