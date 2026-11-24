"""Static H cross sections preserve coupled values and physical derivatives."""

from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.solar import SolarCompositionModel

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def composition_case():
    model = SolarCompositionModel(
        [0., 1e5, 3e5], nmax=6, quadrature_order=2, angle_order=2,
        atoms=(("C I", 6, 0),), oxygen_overlap=False,
        core_escape_model="continuous", shielding_model="continuous_core",
        resolved_infrared_lines=((6, 5),),
        infrared_options=dict(quadrature_order=2, angle_order=4))
    temperature, density = jnp.array([8000., 8200.]), jnp.array([1e11, 2e11])
    state = model.initialize(density, temperature)
    incoming = model.hydrogen.boundary_blackbody(10000., .025)
    return model, state, temperature, density, incoming


@pytest.mark.parametrize("remove_groups", [False, True])
def test_complete_composition_and_group_removal_match_uncached(composition_case, remove_groups):
    original, state, temperature, density, incoming = composition_case
    model = copy(original)
    if remove_groups:
        model.infrared_groups, model.infrared_indices = [], []
    uncached = copy(model)
    uncached._hydrogen_cross_section_grids = ()

    def evaluate(provider):
        residual, physical = provider.evaluate(state, density, temperature, incoming)
        return residual, physical["thermal_net"], physical["valid"]

    cached = jax.jit(lambda: evaluate(model))()
    expected = jax.jit(lambda: evaluate(uncached))()
    assert cached[2] and expected[2]
    np.testing.assert_allclose(cached[0], expected[0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(cached[1], expected[1], rtol=2e-12, atol=0)


def test_temperature_populations_and_signed_photons_keep_both_derivative_modes(composition_case):
    model, state, temperature, density, _ = composition_case
    physical = model.unpack(state, density)
    h, frequency = model.hydrogen, model.metal.line_frequency_hz
    sigma = model._hydrogen_cross_sections_at(frequency)
    flux = 1e5*jnp.cos(jnp.arange(model.cells)[:, None]+jnp.log(frequency)[None, :])
    arguments = (temperature, physical["bound_cm3"], physical["proton_cm3"],
                 physical["electron_cm3"], flux)

    def cached(t, bound, proton, electron, photons):
        return h._photoionization_from_cross_sections(
            t, bound, proton, electron, frequency, photons, sigma)

    def ordinary(t, bound, proton, electron, photons):
        return h.photoionization_from_photon_flux(
            t, bound, proton, electron, frequency, photons)

    direction = tuple(value*.1*jnp.cos(jnp.arange(value.size).reshape(value.shape))
                      for value in arguments)
    expected, derivative = jax.jit(lambda *x: jax.jvp(ordinary, x, direction))(*arguments)
    actual, cached_derivative = jax.jit(lambda *x: jax.jvp(cached, x, direction))(*arguments)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=3e-12, atol=1e-100)
        np.testing.assert_allclose(cached_derivative[key], derivative[key],
                                   rtol=3e-12, atol=1e-100)
    cotangent = {key: jnp.ones_like(value)/(value.size*jnp.max(abs(value)))
                 for key, value in expected.items()}
    first = jax.jit(lambda *x: jax.vjp(ordinary, *x)[1](cotangent))(*arguments)
    second = jax.jit(lambda *x: jax.vjp(cached, *x)[1](cotangent))(*arguments)
    for old, new in zip(first, second):
        np.testing.assert_allclose(new, old, rtol=3e-12, atol=1e-100)
        assert np.isfinite(new).all()


def test_dynamic_frequency_and_replaced_grids_preserve_frequency_derivatives(composition_case):
    model, state, temperature, density, _ = composition_case
    physical = model.unpack(state, density)
    uncached = copy(model)
    uncached._hydrogen_cross_section_grids = ()
    frequency = model.metal.line_frequency_hz

    def evaluate(provider, nu):
        return provider.continuum_coefficients_at(temperature, physical, nu)

    # Same shape and even the original registered values must not freeze a
    # traced frequency argument. A shifted grid must also use its own sigma.
    for scale in (1., 1.007):
        value = frequency*scale
        old, tangent = jax.jit(lambda nu: jax.jvp(
            lambda f: evaluate(uncached, f), (nu,), (nu,)))(value)
        actual, derivative = jax.jit(lambda nu: jax.jvp(
            lambda f: evaluate(model, f), (nu,), (nu,)))(value)
        for a, b, da, db in zip(old, actual, tangent, derivative):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(da, db)
            assert np.isfinite(db).all() and np.any(db != 0)
        step = 1e-6
        plus, minus = evaluate(model, value*jnp.exp(step)), evaluate(model, value*jnp.exp(-step))
        for derivative_value, a, b in zip(derivative, plus, minus):
            finite = (a-b)/(2*step)
            error = np.max(abs(derivative_value-finite))/np.max(abs(derivative_value))
            assert error < 2e-7
