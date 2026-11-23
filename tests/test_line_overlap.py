"""Physical LTE, nonlocal photon balance and AD for overlapping CRD lines."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.hydrogen_continuum import planck_intensity_hz
from escape.line_overlap import LineOverlapGroup

jax.config.update("jax_enable_x64", True)
H, C, KB, MH = 6.62607015e-27, 2.99792458e10, 1.380649e-16, 1.67262192369e-24


def group(order=4, lines=2):
    frequency = np.array([2.923e15, 2.923e15 * (1 - 11e5 / C)])[:lines]
    return LineOverlapGroup(
        [0, 1e3, 1e4, 3e4],
        frequency,
        np.array([1.67e8, 7.66e7])[:lines],
        np.array([2.0, 5.0])[:lines],
        np.array([6.0, 7.0])[:lines],
        np.array([MH, 16 * MH])[:lines],
        np.array([1.9e8, 8e7])[:lines] / (4 * np.pi),
        quadrature_order=order,
    )


def test_actual_finite_width_coefficients_and_rates_satisfy_lte():
    model = group()
    temperature = jnp.full(3, 8000.0)
    lower = jnp.array([[1e12, 3e8], [2e12, 4e8], [8e11, 2e8]])
    upper = (
        lower
        * model.upper_weight
        / model.lower_weight
        * jnp.exp(-H * model.line_frequency_hz / (KB * temperature[:, None]))
    )
    chi, eta, *_ = model.coefficients(temperature, lower, upper)
    blackbody = planck_intensity_hz(model.frequency_hz, 8000.0)
    np.testing.assert_allclose(
        eta / chi, jnp.broadcast_to(blackbody, eta.shape), rtol=1e-14
    )
    opacity = jnp.full((3, len(model.frequency_hz)), 1e-5)
    state = jax.jit(model.solve)(
        temperature,
        lower,
        upper,
        opacity,
        opacity * blackbody,
        jnp.zeros_like(blackbody),
        inner_intensity=blackbody,
        outer_intensity=blackbody,
    )
    assert state["converged"]
    np.testing.assert_allclose(
        state["transfer"].mean_intensity,
        jnp.broadcast_to(blackbody, opacity.shape),
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        lower * state["pump_up_s"], upper * state["radiative_down_s"], rtol=3e-13
    )
    assert np.max(np.abs(state["additional_photon_flux_cm2_s"])) < 1e-12 * np.max(
        4 * np.pi * blackbody * model.weight_hz / (H * model.frequency_hz)
    )


def solve_state(model, scale=1.0):
    lower = jnp.array([[1e12, 3e8], [2e12, 4e8], [8e11, 2e8]])
    upper = jnp.array([[1e3, 2.0], [3e3, 1.0], [1e4, 4.0]])
    lower = lower.at[:, 1].multiply(scale)
    upper = upper.at[:, 1].multiply(scale)
    opacity = jnp.full((3, len(model.frequency_hz)), 2e-5)
    beam = 0.02 * planck_intensity_hz(model.frequency_hz, 10000.0)
    state = model.solve(
        jnp.array([8000.0, 9000.0, 7000.0]),
        lower,
        upper,
        opacity,
        jnp.zeros_like(opacity),
        beam,
    )
    return state, lower, upper, opacity


def test_nonlocal_photon_budget_and_stable_rates_equal_formal_transfer():
    model = group()
    state, lower, upper, opacity = jax.jit(lambda: solve_state(model))()
    assert state["converged"]
    mean = state["transfer"].mean_intensity
    weight = 4 * np.pi * model.weight_hz / (H * model.frequency_hz)
    direct = jnp.sum(
        (state["line_opacity_cm1"] * mean[:, None, :] - state["line_emission"])
        * weight,
        axis=-1,
    )
    np.testing.assert_allclose(state["net_excitation_cm3_s"], direct, rtol=2e-11)
    faces = jnp.sum(state["transfer"].flux_moment_edges * weight, axis=1)
    continuum_absorption = jnp.sum(opacity * mean * weight, axis=1)
    np.testing.assert_allclose(
        faces[-1] - faces[0],
        -jnp.sum(model.widths_cm * (direct.sum(axis=1) + continuum_absorption)),
        rtol=3e-12,
    )
    assert np.any(
        state["additional_photon_flux_cm2_s"] < 0
    ), "Incident screening must be retained."


def test_zero_oxygen_recovers_same_single_line_formal_closure():
    model = group()
    state, lower, upper, opacity = solve_state(model, 0.0)
    # Use exactly the same quadrature to separate zero-species physics from
    # quadrature convergence; this is not a claim of legacy C23 CORE parity.
    single = group(lines=1)
    single.frequency_hz, single.weight_hz = model.frequency_hz, model.weight_hz
    reference = single.solve(
        jnp.array([8000.0, 9000.0, 7000.0]),
        lower[:, :1],
        upper[:, :1],
        opacity,
        jnp.zeros_like(opacity),
        0.02 * planck_intensity_hz(model.frequency_hz, 10000.0),
    )
    np.testing.assert_allclose(
        state["transfer"].mean_intensity,
        reference["transfer"].mean_intensity,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        state["radiative_down_s"][:, 0], reference["radiative_down_s"][:, 0], rtol=2e-14
    )


def test_overlap_derivative_and_frequency_quadrature_convergence():
    model = group()

    def observable(log_scale):
        state, *_ = solve_state(model, jnp.exp(log_scale))
        return jnp.sum(state["net_excitation_cm3_s"][:, 1] * model.widths_cm)

    derivative = jax.jit(jax.grad(observable))(0.0)
    step = 1e-4
    expected = (observable(step) - observable(-step)) / (2 * step)
    np.testing.assert_allclose(derivative, expected, rtol=2e-7)
    coarse, *_ = solve_state(model)
    fine, *_ = solve_state(group(order=8))
    np.testing.assert_allclose(
        coarse["net_excitation_cm3_s"], fine["net_excitation_cm3_s"], rtol=3e-3
    )


def test_scattering_removal_is_separate_from_true_continuum_absorption():
    model = group()
    _, lower, upper, opacity = solve_state(model)
    temperature = jnp.array([8000.0, 9000.0, 7000.0])
    state = jax.jit(model.solve)(
        temperature,
        lower,
        upper,
        opacity,
        jnp.zeros_like(opacity),
        0.02 * planck_intensity_hz(model.frequency_hz, 10000.0),
        scattering_opacity_cm1=jnp.full_like(opacity, 3e-5),
    )
    assert state["converged"]
    photon_weight = 4 * np.pi * model.weight_hz / (H * model.frequency_hz)
    absorbed = jnp.sum(
        (opacity + state["scattering_removal_cm1"])
        * state["transfer"].mean_intensity
        * photon_weight,
        axis=1,
    )
    faces = jnp.sum(state["transfer"].flux_moment_edges * photon_weight, axis=1)
    np.testing.assert_allclose(
        faces[-1] - faces[0],
        -jnp.sum(
            model.widths_cm * (state["net_excitation_cm3_s"].sum(axis=1) + absorbed)
        ),
        rtol=3e-12,
    )
