"""Finite geometry for explicitly selected, possibly inverted IR groups."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.hydrogen_continuum import planck_intensity_hz
from escape.line_overlap import LineOverlapGroup

jax.config.update("jax_enable_x64", True)
H, KB, MH = 6.62607015e-27, 1.380649e-16, 1.67262192369e-24


def group(geometry="spherical", **options):
    m = LineOverlapGroup(
        [0.0, 1e6, 3e6],
        [1e13],
        [6000.0],
        [2.0],
        [4.0],
        [MH],
        [1e4],
        geometry=geometry,
        angle_order=32,
        scattering_model="background",
        allow_inverted_lines=True,
        **options,
    )
    m.frequency_hz, m.weight_hz = jnp.array([1e13]), jnp.array([1e9])
    return m


def test_spherical_group_preserves_planck_field_and_rejects_direct_beam():
    m = group()
    t = jnp.array([8000.0, 8000.0])
    lower = jnp.array([[1e3], [2e3]])
    upper = 2 * lower * jnp.exp(-H * 1e13 / (KB * t[:, None]))
    b = planck_intensity_hz(m.frequency_hz, 8000.0)
    opacity = jnp.array([[1e-3], [1e-10]])
    call = jax.jit(
        lambda beam: m.solve(
            t,
            lower,
            upper,
            opacity,
            opacity * b,
            beam,
            inner_intensity=b,
            outer_intensity=b,
            scattering_opacity_cm1=1e-10,
        )
    )
    state = call(jnp.zeros(1))
    assert state["converged"]
    np.testing.assert_allclose(
        state["transfer"].mean_intensity, jnp.broadcast_to(b, opacity.shape), rtol=1e-10
    )
    np.testing.assert_allclose(
        lower * state["pump_up_s"], upper * state["radiative_down_s"], rtol=1e-10
    )
    assert not call(b)["converged"], "Do not silently reinterpret a collimated beam."


def test_finite_signed_group_has_same_field_photon_budget_and_positive_rates():
    m = group()
    t = jnp.array([8000.0, 8000.0])
    lower = jnp.array([[1e3], [2e3]])
    upper = lower * jnp.array([[1.5], [2.2]])
    b = planck_intensity_hz(m.frequency_hz, 8000.0)
    opacity = jnp.array([[1e-3], [1e-10]])
    state = jax.jit(m.solve)(
        t,
        lower,
        upper,
        opacity,
        opacity * b,
        jnp.zeros(1),
        scattering_opacity_cm1=1e-10,
    )
    assert state["converged"] and state["transfer"].gain_optical_depth > 0
    assert np.all(state["pump_up_s"] >= 0) and np.all(state["radiative_down_s"] >= 0)
    mean, baseline = (
        state["transfer"].mean_intensity,
        state["continuum_transfer"].mean_intensity,
    )
    weight = 4 * np.pi * m.weight_hz / (H * m.frequency_hz)
    direct = jnp.sum(
        (state["line_opacity_cm1"] * mean[:, None, :] - state["line_emission"])
        * weight,
        axis=-1,
    )
    np.testing.assert_allclose(state["net_excitation_cm3_s"], direct, rtol=2e-12)
    continuum = jnp.sum(opacity * (mean - b) * weight, axis=1)
    scattering = jnp.sum(1e-10 * (mean - baseline) * weight, axis=1)
    geo = m.shell_geometry
    faces = jnp.sum(state["transfer"].flux_moment_edges * weight, axis=1) * geo.area_cm2
    np.testing.assert_allclose(
        faces[-1] - faces[0],
        -jnp.sum(geo.volume_cm3 * (direct.sum(1) + continuum + scattering)),
        rtol=2e-10,
    )
    assert 0 < state["scattering_photon_norm_bound"] < 1
    assert not group("slab").solve(
        t,
        lower,
        upper,
        opacity,
        opacity * b,
        jnp.zeros(1),
        scattering_opacity_cm1=1e-10,
    )["converged"]


def test_spherical_zero_line_extinction_derivative_is_continuous():
    m = group()
    t = jnp.array([8000.0, 8000.0])
    lower = jnp.array([[1e3], [2e3]])
    opacity = jnp.full((2, 1), 1e-10)
    b = planck_intensity_hz(m.frequency_hz, 8000.0)

    def bracket(offset):
        upper = 2 * lower * jnp.exp(offset)
        s = m.solve(
            t,
            lower,
            upper,
            opacity,
            opacity * b,
            jnp.zeros(1),
            scattering_opacity_cm1=1e-10,
        )
        return -s["net_excitation_cm3_s"][-1, 0] / (upper[-1, 0] * 6000.0)

    value, derivative = jax.jit(lambda x: jax.jvp(bracket, (x,), (jnp.array(1.0),)))(
        0.0
    )
    np.testing.assert_allclose(value, 1.0, rtol=2e-13)
    for step in (1e-3, 1e-4):
        numerical = (bracket(step) - bracket(-step)) / (2 * step)
        np.testing.assert_allclose(derivative, numerical, rtol=2e-6)


def test_shell_averaged_stellar_mapping_and_zero_line_background():
    average = group(incident_geometry="spherical_shell_average")
    isotropic = group()
    temperature = jnp.array([8000.0, 8000.0])
    zero = jnp.zeros((2, 1))
    opacity = jnp.full((2, 1), 1e-8)
    b = planck_intensity_hz(average.frequency_hz, 8000.0)

    def evaluate(provider, beam, outer):
        return provider.solve(
            temperature,
            zero,
            zero,
            opacity,
            opacity * b,
            beam,
            outer_intensity=outer,
            scattering_opacity_cm1=1e-9,
        )

    mapped = jax.jit(lambda: evaluate(average, 2 * b, jnp.zeros(1)))()
    explicit = jax.jit(lambda: evaluate(isotropic, jnp.zeros(1), 2 * b))()
    assert mapped["converged"] and explicit["converged"]
    np.testing.assert_array_equal(
        mapped["transfer"].mean_intensity, explicit["transfer"].mean_intensity
    )
    np.testing.assert_allclose(
        mapped["transfer"].mean_intensity,
        mapped["continuum_transfer"].mean_intensity,
        rtol=2e-14,
    )
    np.testing.assert_array_equal(mapped["net_excitation_cm3_s"], 0.0)
    np.testing.assert_array_equal(mapped["additional_photon_flux_cm2_s"], 0.0)
