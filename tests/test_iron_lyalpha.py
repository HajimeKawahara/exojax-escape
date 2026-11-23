"""Scoped photon/energy and derivative checks for the empirical H/Fe closure."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from escape.iron_lyalpha import (
    _certified_bisection,
    _extra_rates,
    _prefix_probability,
    iron_lyalpha_exchange,
    static_lyalpha_width,
)

jax.config.update("jax_enable_x64", True)


def test_original_cloudy_width_and_pump_primitives():
    fixture = json.loads(
        (Path(__file__).parent / "data/cloudy_iron_lyalpha.json").read_text()
    )
    width, pump = map(np.asarray, (fixture["width"], fixture["pump"]))
    actual = static_lyalpha_width(*map(jnp.asarray, width[:, :7].T))
    np.testing.assert_allclose(actual, width[:, 7], rtol=2e-6, atol=1e-100)
    occupation = (1 - pump[:, 6]) / (3e12 / 1e5 - 1)
    up, down, _ = jax.vmap(
        lambda w, occ, wn, col, aa, gl, gu: _extra_rates(
            jnp.array([w]),
            jnp.array([occ]),
            jnp.array([aa]),
            jnp.array([gu / gl]),
            jnp.array([wn]),
            jnp.array([[col]]),
        )
    )(
        *map(
            jnp.asarray,
            (
                pump[:, 0],
                occupation,
                pump[:, 1],
                pump[:, 2],
                pump[:, 3],
                pump[:, 4],
                pump[:, 5],
            ),
        )
    )
    # The native offset-first trapezoid avoids subtraction around 82259;
    # the fixture records the independent C++ arithmetic-order control.
    np.testing.assert_allclose(
        np.asarray(up).ravel(), pump[:, 7], rtol=1e-10, atol=1e-100
    )
    np.testing.assert_allclose(
        np.asarray(down).ravel(), pump[:, 8], rtol=1e-10, atol=1e-100
    )


def arguments():
    h = 6.62607015e-27 * 2.99792458e10
    return dict(
        lower_population_cm3=jnp.array([[3e10, 2e10, 1e10]]),
        upper_population_cm3=jnp.array([[3e5, 2e5, 1e5]]),
        einstein_a_s=jnp.array([1e8, 2e8, 3e8]),
        weight_ratio=jnp.array([1.0, 2.0, 3.0]),
        wavenumber_cm=jnp.array([82274.0, 82304.0, 82349.0]),
        transition_energy_erg=h * jnp.array([82274.0, 82304.0, 82349.0]),
        total_optical_depth=jnp.array([[2.0, 3.0, 4.0]]),
        hydrogen_ground_cm3=jnp.array([1e12]),
        hydrogen_2p_cm3=jnp.array([1e5]),
        hydrogen_einstein_a_s=6.265e8,
        hydrogen_transition_energy_erg=h * 82259.0,
        hydrogen_tau_in=jnp.array([1e7]),
        hydrogen_tau_total=jnp.array([2e7]),
        hydrogen_damping=jnp.array([4e-4]),
        hydrogen_doppler_width_cm_s=jnp.array([1e6]),
        electron_cm3=jnp.array([1e8]),
        hydrogen_escape_probability=jnp.array([1e-4]),
        electron_escape_probability=jnp.array([1e-7]),
        continuum_destruction_probability=jnp.array([0.01]),
    )


def base_width(values):
    return static_lyalpha_width(
        values["hydrogen_tau_in"],
        values["hydrogen_tau_total"],
        values["hydrogen_damping"],
        values["hydrogen_doppler_width_cm_s"],
        values["electron_cm3"],
        values["electron_escape_probability"],
        values["continuum_destruction_probability"],
    )


def test_coupled_photon_and_atomic_energy_ledger():
    data = arguments()
    result = iron_lyalpha_exchange(**data)
    assert np.all(result.valid)
    assert np.max(abs(result.width_residual)) < 1e-12
    assert np.all(result.width_cm_s < base_width(data))
    photons = result.net_photons_cm3_s.sum(axis=1)
    hydrogen_loss = (
        data["hydrogen_2p_cm3"] * result.hydrogen_downward_s
        - data["hydrogen_ground_cm3"] * result.hydrogen_upward_s
    )
    np.testing.assert_allclose(hydrogen_loss, photons, rtol=3e-15)
    iron_gain = (result.net_photons_cm3_s * data["transition_energy_erg"]).sum(axis=1)
    hydrogen_energy = photons * data["hydrogen_transition_energy_erg"]
    np.testing.assert_allclose(
        iron_gain + result.redistribution_heat_erg_cm3_s,
        hydrogen_energy,
        rtol=3e-15,
    )
    # These exchanged photons have already been assigned to Fe. The helper
    # deliberately has no second continuum-OTS photon output.
    assert "ots_photon_flux_cm2_s" not in result._fields


def test_reverse_stimulated_exchange_is_signed_and_not_clipped():
    data = arguments()
    data.update(
        lower_population_cm3=jnp.full((1, 3), 1e7),
        upper_population_cm3=jnp.full((1, 3), 1e8),
    )
    result = iron_lyalpha_exchange(**data)
    assert np.all(result.valid)
    assert np.all(result.net_photons_cm3_s[:, :2] < 0)
    assert np.all(result.fe_destruction_probability < 0)
    assert np.all(result.width_cm_s > base_width(data))
    np.testing.assert_allclose(
        data["hydrogen_2p_cm3"] * result.hydrogen_downward_s
        - data["hydrogen_ground_cm3"] * result.hydrogen_upward_s,
        result.net_photons_cm3_s.sum(axis=1),
        rtol=3e-15,
    )
    data["continuum_destruction_probability"] = jnp.zeros(1)
    invalid = iron_lyalpha_exchange(**data)
    assert not np.any(invalid.valid)
    assert np.isnan(invalid.upward_s).all()


def test_zero_iron_and_declared_source_optical_depth_selection():
    data = arguments()
    data.update(
        lower_population_cm3=jnp.zeros((1, 3)), upper_population_cm3=jnp.zeros((1, 3))
    )
    result = iron_lyalpha_exchange(**data)
    np.testing.assert_allclose(result.width_cm_s, base_width(data), rtol=3e-15)
    np.testing.assert_array_equal(result.net_photons_cm3_s, 0)
    np.testing.assert_array_equal(result.hydrogen_downward_s, 0)
    data = arguments()
    data["total_optical_depth"] = jnp.full((1, 3), 0.5)
    thin = iron_lyalpha_exchange(**data)
    np.testing.assert_array_equal(thin.upward_s, 0)
    np.testing.assert_allclose(thin.selection_distance, 0.5)
    # This is the declared C23 thick-line approximation, not an optically
    # thin transfer model or a globally differentiable threshold treatment.


def test_implicit_width_derivative_matches_finite_difference():
    data = arguments()

    def heat(log_abundance):
        return iron_lyalpha_exchange(
            **(
                data
                | {
                    "lower_population_cm3": data["lower_population_cm3"]
                    * jnp.exp(log_abundance),
                    "upper_population_cm3": data["upper_population_cm3"]
                    * jnp.exp(log_abundance),
                }
            )
        ).redistribution_heat_erg_cm3_s.sum()

    value, derivative = jax.jit(jax.value_and_grad(heat))(0.0)
    finite_difference = (heat(1e-4) - heat(-1e-4)) / 2e-4
    assert np.isfinite(value) and np.isfinite(derivative)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-7)


def test_hydrogen_inversion_is_outside_the_trapped_prd_domain():
    data = arguments()
    data["hydrogen_2p_cm3"] = 4 * data["hydrogen_ground_cm3"]
    result = iron_lyalpha_exchange(**data)
    assert not np.any(result.valid)
    assert np.isnan(result.upward_s).all()


def _original_bisection(function, initial):
    def step(_, pair):
        lo, hi = pair
        middle = (lo + hi) / 2
        negative = function(middle) < 0
        return jnp.where(negative, middle, lo), jnp.where(negative, hi, middle)

    lo, hi = jax.lax.fori_loop(
        0, 48, step, (jnp.zeros_like(initial), jnp.ones_like(initial))
    )
    return (lo + hi) / 2


def test_signed_prefix_cancellation_does_not_change_the_solved_equation():
    coefficients = jnp.array([[1e16, -1e16, 1.]])
    offsets = jnp.array([1., 2., 3.])
    cumulative = jnp.pad(jnp.cumsum(coefficients, axis=1), ((0, 0), (1, 0)))
    moments = jnp.pad(jnp.cumsum(coefficients * offsets, axis=1), ((0, 0), (1, 0)))
    # Prefix moment subtraction loses this half-photon contribution. The
    # original signed sum remains the equation and rejects the wrong candidate.
    direct = jnp.sum(coefficients * jnp.clip(2 - offsets / 2, 0, 1), axis=1)
    approximate = _prefix_probability(
        jnp.array([2 / (1.37194e-6 * 2 / 3)]), offsets, cumulative, moments
    )
    np.testing.assert_array_equal(direct, .5)
    assert not np.array_equal(direct, approximate)
    def function(x):
        return x - .4 + .1 * (direct - .5)

    def candidate(x):
        return x - .4 + .1 * (approximate - .5)
    np.testing.assert_array_equal(
        _certified_bisection(function, candidate, jnp.zeros(1)),
        _original_bisection(function, jnp.zeros(1)),
    )


def test_certification_preserves_original_clip_knot_roots_and_derivatives():
    offsets, coefficients = jnp.array([.25, .5]), jnp.array([.01, -.002])

    def probability(width):
        return jnp.sum(coefficients * jnp.clip(2 - offsets / width, 0, 1))

    def root(target, use_candidate):
        def function(x):
            return x - target - probability(x[0])

        def solve(fn, initial):
            if use_candidate:
                return _certified_bisection(fn, lambda x: fn(x) + 1e-16, initial)
            return _original_bisection(fn, initial)

        return jax.lax.custom_root(
            function, jnp.zeros(1), solve,
            lambda linear, value: value / linear(jnp.ones_like(value)),
        )[0]

    for knot in (.125, .25, .5):
        for displacement in (-1e-12, 0., 1e-12):
            desired = knot + displacement
            target = desired - probability(desired)
            original = jax.value_and_grad(lambda t: root(t, False))(target)
            accelerated = jax.value_and_grad(lambda t: root(t, True))(target)
            np.testing.assert_allclose(original[0], desired, rtol=0, atol=2**-48)
            np.testing.assert_array_equal(accelerated, original)


def test_batched_nonfinite_candidate_falls_back_without_changing_guards():
    def solve(target):
        return _certified_bisection(
            lambda x: x - target,
            lambda x: jnp.where(target < .5, jnp.nan, x - .9),
            jnp.zeros(1),
        )

    target = jnp.array([.1, .4, .9])
    result = jax.jit(jax.vmap(solve))(target)
    np.testing.assert_allclose(result[:, 0], target, rtol=0, atol=2**-48)


def test_host_prepared_line_order_preserves_the_exchange():
    data = arguments()
    order = np.argsort(np.abs(np.asarray(data["wavenumber_cm"]) - 82259), kind="stable")
    expected = iron_lyalpha_exchange(**data)
    actual = iron_lyalpha_exchange(**data, wavenumber_order=order)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
