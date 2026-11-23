"""Synthetic physical limits for an explicit infrared scattering reservoir."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_continuum import planck_intensity_hz
from escape.line_overlap import LineOverlapGroup

jax.config.update("jax_enable_x64", True)
H, C, KB, MH = 6.62607015e-27, 2.99792458e10, 1.380649e-16, 1.67262192369e-24


def model(**options):
    return LineOverlapGroup(
        [0., 1e6, 3e6], [1e13], [6000.], [2.], [4.], [MH], [1e4],
        quadrature_order=2, angle_order=4, **options,
    )


def fields(m):
    t = jnp.full(2, 8000.)
    lower = jnp.array([[1000.], [2000.]])
    upper = lower * 2 * jnp.exp(-H * 1e13 / (KB * t[:, None]))
    blackbody = planck_intensity_hz(m.frequency_hz, 8000.)
    opacity = jnp.full((2, len(m.frequency_hz)), 1e-6)
    return t, lower, upper, blackbody, opacity


def test_background_scattering_preserves_planck_field_and_detailed_balance():
    m = model(scattering_model="background", allow_inverted_lines=True)
    t, lower, upper, b, opacity = fields(m)
    result = jax.jit(m.solve)(t, lower, upper, opacity, opacity * b, jnp.zeros_like(b),
                             inner_intensity=b, outer_intensity=b,
                             scattering_opacity_cm1=2e-7)
    assert result["converged"]
    np.testing.assert_allclose(result["transfer"].mean_intensity,
                               jnp.broadcast_to(b, opacity.shape), rtol=3e-14)
    np.testing.assert_allclose(lower * result["pump_up_s"],
                               upper * result["radiative_down_s"], rtol=3e-14)
    energy_scale = 4 * np.pi * 2e-7 * jnp.sum(b * m.weight_hz)
    assert np.max(np.abs(result["scattering_energy_removed_erg_cm3_s"])) < 1e-14 * energy_scale
    assert result["background_linear_residual"] < 1e-14


def test_zero_net_line_opacity_has_unit_bracket_and_smooth_derivative():
    m = model(scattering_model="background", allow_inverted_lines=True)
    # A one-frequency synthetic operator gives an exact zero-opacity limit,
    # independently of profile quadrature or frequency redistribution.
    m.frequency_hz, m.weight_hz = jnp.array([1e13]), jnp.array([1e9])
    t, lower, _, b, opacity = fields(m)
    def solve(offset):
        upper = 2 * lower * jnp.exp(offset)
        result = m.solve(t, lower, upper, opacity, opacity * b, jnp.zeros_like(b),
                         scattering_opacity_cm1=2e-7)
        return result, upper
    result, upper = jax.jit(solve)(0.)
    assert result["converged"]
    _, _, absorption, stimulated, _ = m.coefficients(t, lower, upper)
    gross_opacity = lower[:, :, None]*absorption + upper[:, :, None]*stimulated
    assert np.max(np.abs(result["line_opacity_cm1"])/gross_opacity) < 1e-15
    np.testing.assert_allclose(-result["net_excitation_cm3_s"] / (upper * 6000),
                               1., rtol=3e-15)
    def bracket(offset):
        state, population = solve(offset)
        return -state["net_excitation_cm3_s"][1, 0] / (population[1, 0] * 6000)
    derivative = jax.jit(jax.grad(bracket))(0.)
    for step in [1e-3, 1e-4]:
        expected = (bracket(step) - bracket(-step)) / (2 * step)
        np.testing.assert_allclose(derivative, expected, rtol=2e-6)
    # Individual inversion is permitted only while total extinction is positive.
    assert solve(1e-3)[0]["converged"]
    assert not solve(10.)[0]["converged"]


def test_signed_photon_energy_ledgers_and_return_bound():
    m = model(scattering_model="background", allow_inverted_lines=True)
    t, lower, upper, b, opacity = fields(m)
    upper *= jnp.array([[1.2], [.7]])
    state = jax.jit(m.solve)(t, lower, upper, opacity, opacity * b, .2 * b,
                            scattering_opacity_cm1=2e-7)
    assert state["converged"]
    weight = 4 * np.pi * m.weight_hz / (H * m.frequency_hz)
    mean = state["transfer"].mean_intensity
    baseline = state["continuum_transfer"].mean_intensity
    direct = jnp.sum((state["line_opacity_cm1"] * mean[:, None, :]
                     - state["line_emission"]) * weight, axis=-1)
    np.testing.assert_allclose(state["net_excitation_cm3_s"], direct, rtol=3e-13)
    continuum = jnp.sum(opacity * (mean - b) * weight, axis=1)
    scatter = jnp.sum(2e-7 * (mean - baseline) * weight, axis=1)
    faces = jnp.sum(state["transfer"].flux_moment_edges * weight, axis=1)
    np.testing.assert_allclose(faces[-1] - faces[0],
        -jnp.sum(m.widths_cm * (direct.sum(1) + continuum + scatter)), rtol=2e-12)
    energy = 4 * np.pi * jnp.sum(2e-7 * (mean - baseline) * m.weight_hz, axis=1)
    np.testing.assert_allclose(state["scattering_energy_removed_erg_cm3_s"], energy)
    assert np.any(np.asarray(energy) < 0), "Scattering is signed, not always heating."
    assert 0 < state["scattering_photon_norm_bound"] < 1
    np.testing.assert_allclose(state["omitted_scattering_return_photons_cm2_s"],
        state["absolute_line_excess_photons_cm2_s"] * state["scattering_return_factor"])


def test_default_scattering_path_is_unchanged_and_options_are_explicit():
    default = model()
    explicit = model(scattering_model="attenuation", allow_inverted_lines=False)
    t, lower, upper, b, opacity = fields(default)
    args = (t, lower, upper, opacity, opacity*b, .2*b)
    a, z = default.solve(*args), explicit.solve(*args)
    for key in ("pump_up_s", "radiative_down_s", "additional_photon_flux_cm2_s"):
        np.testing.assert_array_equal(a[key], z[key])
    with pytest.raises(ValueError, match="scattering"):
        model(scattering_model="unrecognized")
    with pytest.raises(ValueError, match="background"):
        model(allow_inverted_lines=True)


def test_zero_line_has_no_background_only_receiver_correction():
    m = model(scattering_model="background", allow_inverted_lines=True)
    t, lower, upper, b, opacity = fields(m)
    result = jax.jit(m.solve)(t, jnp.zeros_like(lower), jnp.zeros_like(upper),
        opacity, opacity*b, .2*b, scattering_opacity_cm1=2e-7)
    assert result["converged"]
    photon_scale = 4 * np.pi * result["continuum_transfer"].mean_intensity * m.weight_hz / (H*m.frequency_hz)
    assert np.max(np.abs(result["additional_photon_flux_cm2_s"])/photon_scale) < 3e-15
    energy_scale = 4 * np.pi * 2e-7 * jnp.sum(b*m.weight_hz)
    assert np.max(np.abs(result["scattering_energy_removed_erg_cm3_s"])) < 3e-15*energy_scale
    assert result["omitted_scattering_return_photons_cm2_s"] == 0.
