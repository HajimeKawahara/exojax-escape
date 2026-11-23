"""The observer must consume the coupled root, not a separate local-SE path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.coupled_forward import CoupledHydrogenForwardModel
from test_hydrogen_forward import offline_nist  # noqa: F401

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def model():
    return CoupledHydrogenForwardModel(nlayer=4, nmax=3, quadrature_order=4,
                                       nnu=61, nlon=4, nlat=2)


def test_spectrum_consumes_the_same_nonlocal_populations(model, record_property):
    density, temperature = jnp.full(4, 1e10), jnp.full(4, 8000.)
    inner = model.radiation.boundary_blackbody(4500.)
    outer = model.radiation.boundary_blackbody(10000., .05)
    solution = model.population_profiles(density, temperature, inner, outer)
    assert solution["converged"]
    uncoupled = model.radiation._initial_fractions(density[::-1], temperature[::-1], inner, outer)
    change = float(jnp.max(jnp.abs(solution["bound_cm3"][::-1] / density[:, None] / uncoupled - 1)))
    assert change > .1
    velocity = jnp.linspace(5., 1., 4)
    direct = model.observer.spectrum_from_profiles(
        solution["bound_cm3"], solution["temperature_k"], velocity,
        electron_cm3=solution["electron_cm3"], proton_cm3=solution["proton_cm3"], include_continuum=True,
    )
    spectrum = model.spectrum(density, temperature, velocity, inner, outer)
    assert np.isfinite(spectrum).all()
    np.testing.assert_allclose(spectrum, direct, rtol=1e-12, atol=1e-15)
    record_property("feedback_population_change", change)
    record_property("coupled_root_residual", float(solution["residual_norm"]))


def test_observer_derivative_includes_the_radiation_population_root(model, record_property):
    inner = model.radiation.boundary_blackbody(4500.)
    outer = model.radiation.boundary_blackbody(10000., .05)

    @jax.jit
    def observable(log_density):
        spectrum = model.spectrum(jnp.full(4, 1e10 * jnp.exp(log_density)),
                                   jnp.full(4, 8000.), jnp.linspace(5., 1., 4), inner, outer)
        return jnp.sum(spectrum**2)

    step = 2e-4
    automatic = jax.grad(observable)(0.)
    finite = (observable(step) - observable(-step)) / (2 * step)
    assert np.isfinite(automatic) and abs(float(finite)) > 0
    np.testing.assert_allclose(automatic, finite, rtol=2e-4, atol=1e-18)
    record_property("spectrum_gradient_relative_error", float(abs(automatic / finite - 1)))
