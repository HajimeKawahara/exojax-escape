"""Physical limits and derivatives of the prescribed H-alpha atmosphere."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydrogen_forward import HydrogenForwardModel, disk_quadrature

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def model():
    return HydrogenForwardModel(nlayer=32, nlon=8, nlat=4, nnu=401)


def test_zero_column_has_no_line_and_a_finite_derivative(model):
    spectrum = model.spectrum(n2_column=0.0)
    np.testing.assert_allclose(spectrum, 0.0, atol=1e-15)
    derivative = jax.grad(
        lambda scale: jnp.sum(model.spectrum(n2_column=3e14 * scale))
    )(0.0)
    assert np.isfinite(derivative)
    assert derivative != 0.0


def test_upper_population_reduction_and_outflow_change_line_shape(model):
    velocity = np.asarray(model.velocity_kms)
    center = np.argmin(np.abs(velocity))
    blue = (velocity > -60.0) & (velocity < -5.0)
    red = (velocity > 5.0) & (velocity < 60.0)
    boltzmann = np.asarray(model.spectrum(departure_floor=1.0, outflow_kms=0.0))
    static = np.asarray(model.spectrum(outflow_kms=0.0))
    outflow = np.asarray(model.spectrum(outflow_kms=10.0))

    assert static.shape == velocity.shape
    assert np.isfinite([boltzmann, static, outflow]).all()
    assert boltzmann[center] > 0.95 * boltzmann.max()
    assert static[center] < 0.1 * min(static[blue].max(), static[red].max())
    np.testing.assert_allclose(static[blue].max(), static[red].max(), rtol=2e-3)
    assert outflow[red].max() > 1.5 * outflow[blue].max()


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
        spectrum = model.spectrum(
            n2_column=3e14 * parameters[0],
            departure_floor=parameters[1],
            outflow_kms=parameters[2],
        )
        return jnp.mean(weights * spectrum)

    parameters = np.array([1.0, 0.08, 10.0])
    automatic = np.asarray(jax.jit(jax.grad(weighted_flux))(parameters))
    finite = np.empty(3)
    for index in range(3):
        step = 1e-4 * (1.0 + abs(parameters[index]))
        displacement = np.zeros(3)
        displacement[index] = step
        finite[index] = (
            weighted_flux(parameters + displacement)
            - weighted_flux(parameters - displacement)
        ) / (2.0 * step)

    assert np.isfinite(automatic).all()
    assert np.all(np.abs(automatic) > 1e-10)
    np.testing.assert_allclose(automatic, finite, rtol=2e-3, atol=1e-10)
