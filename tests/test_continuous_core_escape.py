"""Independent angular/frequency integration for the explicit K2 control."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expn

from escape.line_escape_probability import crd_wing_escape, doppler_core_escape

jax.config.update("jax_enable_x64", True)


def _integrated_core(tau):
    # A normalized Doppler profile averaged over the outward hemisphere.
    scale = np.sqrt(max(0., np.log(tau)))
    edges = sorted(set([0., max(0., scale - 2), scale + 2, 20.]))
    return sum(quad(
        lambda x: 2 / np.sqrt(np.pi) * np.exp(-x*x)
        * expn(2, tau * np.exp(-x*x)), left, right,
        epsabs=1e-30, epsrel=1e-10,
    )[0] for left, right in zip(edges[:-1], edges[1:]))


def test_positive_control_matches_independent_escape_integral():
    t = np.r_[np.geomspace(1e-10, 1e14, 101), np.linspace(8., 14., 101)]
    tau = t / np.sqrt(np.pi)
    expected = np.array([_integrated_core(value) for value in tau])
    actual = np.asarray(doppler_core_escape(
        jnp.asarray(tau), core_escape_model="continuous"))
    # This remains an approximation, with a measured maximum near t=11.2.
    assert np.max(abs(actual / expected - 1)) < .0066
    thin = t < .011
    np.testing.assert_allclose(actual[thin], expected[thin], rtol=3e-10)
    assert np.all((actual > 0) & (actual <= 1))


def test_positive_joins_are_c1_and_automatic_derivative_matches_difference():
    function = lambda t: doppler_core_escape(t / jnp.sqrt(jnp.pi),
                                             core_escape_model="continuous")
    derivative = jax.grad(function)
    for join in (.01, 10., 11., 12.):
        epsilon = 1e-7
        np.testing.assert_allclose(function(join-epsilon), function(join+epsilon),
                                   rtol=4e-7, atol=1e-12)
        np.testing.assert_allclose(derivative(join-epsilon), derivative(join+epsilon),
                                   rtol=1e-5, atol=1e-12)
    for point in (1e-6, .005, .01, 1., 10., 11., 12., 100.):
        step = point * 1e-5
        numerical = (function(point+step)-function(point-step)) / (2*step)
        np.testing.assert_allclose(derivative(point), numerical,
                                   rtol=2e-5, atol=1e-11)
    assert function(0.) == 1.
    assert np.isfinite(derivative(0.))
    assert np.isfinite(derivative(jnp.asarray(0., dtype=jnp.float32)))
    # The positive logarithmic singularity is physical, not globally smoothed.
    assert abs(derivative(1e-12)) > abs(derivative(1e-6))


def test_default_high_depth_and_gain_prescriptions_are_retained():
    tau = jnp.r_[jnp.linspace(-29.9, -.001, 31), 0.,
                  jnp.geomspace(12.01/np.sqrt(np.pi), 1e14, 101)]
    for function in (doppler_core_escape, lambda x, **kw: crd_wing_escape(x, .01, **kw)):
        np.testing.assert_array_equal(function(tau), function(tau, core_escape_model="cloudy"))
        np.testing.assert_allclose(function(tau), function(tau, core_escape_model="continuous"),
                                   rtol=3e-15, atol=0.)
    assert np.isnan(doppler_core_escape(-30., core_escape_model="continuous"))
    with pytest.raises(ValueError, match="Doppler core"):
        doppler_core_escape(1., core_escape_model="unknown")


def test_actual_solar_path_selects_the_shared_core_variant():
    from escape.hydrogen_continuum import planck_intensity_hz
    from escape.solar import SolarCompositionModel

    options = dict(nmax=3, quadrature_order=2, angle_order=2,
                   atoms=(("C I", 6, 0),), line_sampling="average")
    source = SolarCompositionModel([0., 1e2, 3e2], **options)
    variant = SolarCompositionModel([0., 1e2, 3e2],
        core_escape_model="continuous", **options)
    assert variant.hydrogen.core_escape_model == variant.metal.core_escape_model == "continuous"
    density, temperature = jnp.array([1e10, 2e10]), jnp.array([7000., 8000.])
    state = source.initialize(density, temperature)
    incident = .025 * planck_intensity_hz(source.grid.frequency_hz, 10000.)

    def evaluate(model):
        target, physical = model.evaluate(state, density, temperature, incident)
        return target, physical["valid"], physical["radiation"]["line_operators"]

    before, valid_before, old = jax.jit(lambda: evaluate(source))()
    after, valid_after, new = jax.jit(lambda: evaluate(variant))()
    assert valid_before and valid_after
    assert np.isfinite(before).all() and np.isfinite(after).all()
    assert np.max(abs(np.asarray(new.escape_probability-old.escape_probability))) > 1e-8
    # Pump shielding is a separate selector and retains its source behavior.
    np.testing.assert_array_equal(new.pump_up_s, old.pump_up_s)
