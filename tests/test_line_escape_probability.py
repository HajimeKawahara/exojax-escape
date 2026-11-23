"""Independent C23 line operators and photon/energy/derivative identities."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from escape import line_escape_probability as ep
from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
from escape.hydrogen_rates import PLANCK_CGS, TWO_PHOTON_A_S
from escape.hydrogen_two_photon import hydrogen_two_photon_shape, two_photon_emissivity

jax.config.update("jax_enable_x64", True)
_DATA = Path(__file__).with_name("data")


def test_independent_cloudy_line_kernels():
    data = np.load(_DATA / "cloudy_line_escape.npz")
    t, a, beta, albedo, delta = map(jnp.asarray, data["kernel_inputs"].T)
    lya, destroyed = ep.lya_one_side(t, beta, albedo)
    actual = np.stack(
        (
            ep.doppler_core_escape(t),
            ep.crd_wing_escape(t, a),
            ep.prd_escape(t, a),
            lya,
            destroyed,
            ep.federman_shield(t, a),
            ep.federman_shield(t, a, delta),
        ),
        axis=1,
    )
    expected = data["kernel_c23"]
    np.testing.assert_allclose(actual[:, :4], expected[:, :4], rtol=2e-6, atol=1e-30)
    np.testing.assert_allclose(actual[:, 4], expected[:, 4], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual[:, 5], expected[:, 5], rtol=2e-6, atol=1e-30)
    stable = (delta == 0) | (delta / (1 + t) >= 1e-7)
    np.testing.assert_allclose(
        actual[stable, 6], expected[stable, 6], rtol=2e-6, atol=1e-30
    )
    t, a, _, _, delta = map(jnp.asarray, data["inverted_kernel_inputs"].T)
    negative = np.stack(
        (
            ep.doppler_core_escape(t),
            ep.crd_wing_escape(t, a),
            ep.prd_escape(t, a),
            ep.federman_shield(t, a),
            ep.federman_shield(t, a, delta),
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        negative, data["inverted_kernel_c23"][:, [0, 1, 2, 5, 6]], rtol=1e-12
    )
    np.testing.assert_allclose(
        ep.hydrogen_rayleigh_cross_section(data["rayleigh_energy_ryd"]),
        data["rayleigh_c23"],
        rtol=1e-12,
    )


def test_complete_line_operator_against_original_cloudy():
    data = np.load(_DATA / "cloudy_line_escape.npz")
    args = data["operator_inputs"]
    k, ti, tt, tp, a, kl, klo, km, kc, ne, aa, gl, gu, ni, no, dz, nu, al = map(
        jnp.asarray, args.T
    )
    result = ep.line_operators(
        k,
        tau_in=ti,
        tau_total=tt,
        tau_pump=tp,
        damping=a,
        line_extinction_cm1=kl,
        lower_absorption_cm1=klo,
        multiplet_extinction_cm1=km,
        continuum_absorption_cm1=kc,
        electron_cm3=ne,
        einstein_a_s=aa,
        lower_weight=gl,
        upper_weight=gu,
        incident_occupation=ni,
        outward_continuum_occupation=no,
        zone_width_cm=dz,
        upper_population_cm3=nu,
        albedo=al,
    )
    actual, expected = np.asarray(result).T, data["operator_c23"]
    # C23 stores escape probabilities in float32; propagate its cancellation
    # error through destruction and electron escape rather than loosening rates.
    eps = np.finfo(np.float32).eps
    effective = np.where(args[:, 7] > 0, args[:, 7], args[:, 5])
    beta = args[:, 8] / (np.sqrt(np.pi) * effective + args[:, 8])
    fit = np.minimum(1e-3, 8.5 * beta)
    dest_budget = np.where(
        args[:, 0] == 0, 2 * eps * (1 - args[:, -1]), eps * fit / (1 + fit)
    )
    electron = args[:, 9] * ep.SIGMA_THOMSON
    eratio = electron / (electron + np.where(args[:, 7] > 0, args[:, 7], args[:, 6]))
    budget = np.zeros_like(actual)
    budget[:, 1] = dest_budget
    budget[:, 2] = eratio * (eps + dest_budget)
    budget[:, 6] = dest_budget * args[:, 10] * args[:, 16]
    stable = ((kl + kc) * dz <= 1e-3) | ((kl + kc) * dz / (1 + tp) >= 1e-7)
    assert np.all(
        np.abs(actual[stable] - expected[stable])
        <= budget[stable] + 2e-6 * np.abs(expected[stable]) + 1e-35
    )


def test_destroyed_photons_are_partitioned_and_only_excess_energy_heats():
    result = ep.on_the_spot_return(
        12.0,
        3e15,
        jnp.array([2.0, 1.0]),
        jnp.array([4.0, 5.0]),
        jnp.array([1e-11, 1.5e-11]),
        4.0,
    )
    np.testing.assert_allclose(result.absorbed_photons_cm3_s, [6.0, 3.0])
    np.testing.assert_allclose(result.unresolved_photons_cm3_s, 3.0)
    np.testing.assert_allclose(
        result.photoelectron_heating_erg_cm3_s,
        jnp.array([6.0, 3.0]) * (PLANCK_CGS * 3e15 - jnp.array([1e-11, 1.5e-11])),
    )


def test_two_photon_shape_and_photon_energy_conservation():
    rows = np.loadtxt(_DATA / "cloudy_two_photon.csv", delimiter=",")
    np.testing.assert_allclose(
        hydrogen_two_photon_shape(rows[:, 0]), rows[:, 1], rtol=2e-14
    )
    model = HydrogenEscapeProbabilityModel([0.0, 1.0], nmax=3, quadrature_order=6)
    nu = model.grid.frequency_hz
    transition = model.atom.excitation_energy_erg[1] / PLANCK_CGS
    upper = 3e5
    emission = two_photon_emissivity(nu, model.grid.weight_hz, transition, upper)
    photons = 4 * np.pi * np.sum(emission * model.grid.weight_hz / (PLANCK_CGS * nu))
    energy = 4 * np.pi * np.sum(emission * model.grid.weight_hz)
    np.testing.assert_allclose(photons, 2 * upper * TWO_PHOTON_A_S, rtol=3e-15)
    np.testing.assert_allclose(
        energy, upper * TWO_PHOTON_A_S * PLANCK_CGS * transition, rtol=2e-8
    )


def test_piecewise_operators_have_finite_temperature_and_column_derivatives():
    def values(log_tau):
        tau = jnp.exp(log_tau)
        return jnp.array(
            [
                ep.doppler_core_escape(tau),
                ep.crd_wing_escape(tau, 1e-3),
                ep.prd_escape(tau, 1e-3),
                ep.federman_shield(tau, 1e-3, 0.1 * tau),
            ]
        )

    point, step = jnp.log(300.0), 1e-5
    auto = jax.jacfwd(values)(point)
    finite = (values(point + step) - values(point - step)) / (2 * step)
    assert np.isfinite(auto).all()
    np.testing.assert_allclose(auto, finite, rtol=3e-6, atol=1e-12)


def test_inverted_metal_line_zone_average_matches_cloudy_negative_branch():
    # Original C23 avg_shield (rt_continuum_shield_fcn.cpp:165-203), compiled
    # with realnum=float, gives exactly zero for these positive-depth cases:
    # log(1 + MAX2(delta/(1+tau), SMALLFLOAT))/MAX2(..., SMALLFLOAT).
    tau = jnp.array([-29.0, -2.0, -1.0095126033145931, -1.000001])
    depth = jnp.array([0.0011, 0.1, 1.0, 1e8])
    actual = ep.federman_shield(tau[:, None], 1e-3, depth[None, :])
    np.testing.assert_array_equal(actual, np.zeros((4, 4)))
    # The exact C23 expression is 0/0 at tau=-1; the declared continuous
    # extension is zero. A zone whose average is disabled keeps its pump.
    assert ep.federman_shield(-1.0, 1e-3, 0.1) == 0.0
    np.testing.assert_allclose(
        ep.federman_shield(tau, 1e-3, 1e-4),
        ep.federman_shield(tau, 1e-3, 0.1, average_zone=False),
    )
    # Preserve the independently evaluated ordinary AVG_DEPTH branch.
    positive = ep.federman_shield(-0.5, 1e-3, 0.1)
    point = ep.federman_shield(-0.5, 1e-3, average_zone=False)
    np.testing.assert_allclose(positive / point, 0.91160778396977293, rtol=1e-15)
    for point in (-1.1, -0.9):
        value, derivative = jax.value_and_grad(
            lambda x: ep.federman_shield(x, 1e-3, 0.1)
        )(point)
        assert np.isfinite(value) and np.isfinite(derivative)
