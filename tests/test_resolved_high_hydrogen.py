"""Full nl collision graphs, independent kernels, and implicit derivatives."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_atom import (
    atom_collision_rates,
    atom_cross_sections,
    atom_saha_factors,
    make_hydrogen_atom,
    solve_atom_populations,
)
from escape.hydrogen_mixing import mixing_bound_bound_cm3_s, proton_l_mixing_psm20
from escape.hydrogen_rates import BOLTZMANN_CGS, PLANCK_CGS
from test_hydrogen_mixing import independent_cross_section_integral
from test_hydrogen_forward import offline_nist  # noqa: F401

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nmax,resolved", [(15, 6), (15, 10), (20, 10)])
def test_all_pairs_aggregate_and_obey_detailed_balance(nmax, resolved):
    base = make_hydrogen_atom(nmax, resolved_nmax=5)
    atom = make_hydrogen_atom(nmax, resolved_nmax=resolved)
    members = base.orbital_members
    old_index = [
        next(i for i, group in enumerate(members) if set(new).issubset(group))
        for new in atom.orbital_members
    ]
    summation = np.eye(len(members))[old_index]
    weights = (
        summation.T
        * np.asarray(atom.statistical_weights)
        / np.asarray(base.statistical_weights)[:, None]
    )

    def radiation(selected):
        matrix = np.zeros((len(selected.principal_n),) * 2)
        matrix[np.asarray(selected.line_upper), np.asarray(selected.line_lower)] = (
            selected.einstein_a_s
        )
        return matrix

    np.testing.assert_allclose(
        weights @ radiation(atom) @ summation, radiation(base), rtol=1e-12, atol=0
    )
    frequency = jnp.geomspace(1e11, 1e17, 32)
    np.testing.assert_allclose(
        weights @ atom_cross_sections(atom, frequency),
        atom_cross_sections(base, frequency),
        rtol=1e-12,
        atol=0,
    )
    for t in [4000.0, 8000.0, 14000.0]:
        options = dict(
            low_temperature_extrapolation="cloudy",
            shell_collision_model="anderson_lebedev",
        )
        bb, ci, three = atom_collision_rates(atom, t, **options)
        oldbb, oldci, oldthree = atom_collision_rates(base, t, **options)
        np.testing.assert_allclose(weights @ bb @ summation, oldbb, rtol=1e-12, atol=0)
        np.testing.assert_allclose(weights @ ci, oldci, rtol=1e-12, atol=0)
        np.testing.assert_allclose(summation.T @ three, oldthree, rtol=1e-12, atol=0)
        saha = atom_saha_factors(atom, t)
        np.testing.assert_allclose(
            saha[:, None] * bb, (saha[:, None] * bb).T, rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(three, ci * saha, rtol=1e-12, atol=0)
        pn = np.asarray(atom.principal_n)
        expected = (pn[:, None] > 5) & (pn[:, None] > pn[None, :])
        assert np.all(np.asarray(bb)[expected] > 0)
    i6s = atom.orbital_members.index(((6, 0),))
    i4s = atom.orbital_members.index(((4, 0),))
    assert bb[i6s, i4s] > 0 and radiation(atom)[i6s, i4s] == 0


def test_selected_rates_match_linked_c23_kernel_and_resolver():
    rows = np.load(Path(__file__).parent / "data/cloudy_resolved_high_collisions.npz")[
        "rows"
    ]
    atom = make_hydrogen_atom(15, resolved_nmax=10)
    lookup = {
        member[0] if len(member) == 1 else (member[0][0], -1): i
        for i, member in enumerate(atom.orbital_members)
    }
    for t in np.unique(rows[:, 0]):
        bb, ci, _ = atom_collision_rates(
            atom,
            t,
            low_temperature_extrapolation="cloudy",
            shell_collision_model="anderson_lebedev",
        )
        for row in rows[rows[:, 0] == t]:
            _, nu, lu, nl, ll, _, _, _, ion, rate = row
            hi, lo = lookup[int(nu), int(lu)], lookup[int(nl), int(ll)]
            np.testing.assert_allclose(bb[hi, lo], rate, rtol=5e-14)
            np.testing.assert_allclose(ci[hi], ion, rtol=5e-14)


def test_high_orbital_mixing_and_independent_integral():
    atom = make_hydrogen_atom(15, resolved_nmax=10)
    matrix = jax.jit(mixing_bound_bound_cm3_s)(atom, 8000.0, 1e12)
    assert np.count_nonzero(matrix) == 90
    saha = atom_saha_factors(atom, 8000.0)
    np.testing.assert_allclose(
        saha[:, None] * matrix, (saha[:, None] * matrix).T, rtol=1e-12, atol=0
    )
    decay = np.zeros(len(atom.principal_n))
    np.add.at(decay, np.asarray(atom.line_upper), np.asarray(atom.einstein_a_s))
    for n, ell in [(6, 1), (6, 5), (10, 1), (10, 9)]:
        i = atom.orbital_members.index(((n, ell),))
        lifetime = 1 / decay[i]
        for ne in [1e8, 1e12, 1e15]:
            expected = independent_cross_section_integral(8000.0, ne, ell, lifetime, n)
            actual = proton_l_mixing_psm20(8000.0, ne, n, ell, ell - 1, lifetime)
            np.testing.assert_allclose(actual, expected, rtol=2e-10)


def test_explicit_high_atom_implicit_population_gradient():
    atom = make_hydrogen_atom(15, resolved_nmax=10)
    edges = jnp.geomspace(1e11, 1e17, 33)
    frequency = jnp.sqrt(edges[1:] * edges[:-1])
    weight = jnp.diff(edges)
    continuum = 0.025 / jnp.expm1(PLANCK_CGS * frequency / (BOLTZMANN_CGS * 10000.0))
    line = 0.025 / jnp.expm1(
        PLANCK_CGS * atom.line_frequency_hz / (BOLTZMANN_CGS * 10000.0)
    )

    @jax.jit
    def observable(selected, log_t):
        result = solve_atom_populations(
            1e12,
            selected,
            8000 * jnp.exp(log_t),
            line,
            continuum,
            frequency,
            weight,
            low_temperature_extrapolation="cloudy",
            shell_collision_model="anderson_lebedev",
        )
        return jnp.log(result.bound_cm3[3:6].sum()), result.converged

    value, converged = observable(atom, 0.0)
    assert converged and np.isfinite(value)
    gradient = jax.grad(lambda point: observable(atom, point)[0])(0.0)
    h = 1e-4
    finite = (observable(atom, h)[0] - observable(atom, -h)[0]) / (2 * h)
    np.testing.assert_allclose(gradient, finite, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("model", ["vs80", "anderson_n45"])
def test_high_resolution_rejects_unimplemented_collision_models(model):
    atom = make_hydrogen_atom(10, resolved_nmax=6)
    with pytest.raises(ValueError, match="anderson_lebedev"):
        atom_collision_rates(atom, 8000.0, shell_collision_model=model)


@pytest.mark.parametrize("resolved,count", [(6, 116), (10, 374)])
def test_high_resolution_public_observer_and_ir_routing(resolved, count):
    from escape.coupled_forward import CoupledHydrogenForwardModel
    from escape.solar import SolarAtmosphereModel, _hydrogen_infrared_indices

    base = make_hydrogen_atom(15)
    selected = np.flatnonzero(
        np.asarray(base.line_frequency_hz) * PLANCK_CGS <= 4000 * BOLTZMANN_CGS
    )
    pairs = list(
        zip(
            np.asarray(base.principal_n[base.line_upper[selected]]),
            np.asarray(base.principal_n[base.line_lower[selected]]),
        )
    )
    options = dict(nnu=21, nlon=2, nlat=2, continuum_points=2)
    model = SolarAtmosphereModel(
        [1.0, 1.5, 2.0, 3.0],
        nmax=15,
        resolved_nmax=resolved,
        atoms=(("C I", 6, 0),),
        quadrature_order=2,
        angle_order=2,
        wind_ngrid=33,
        observer_options=options,
    )
    assert (
        model.observer.shell_collision_model
        == model.hydrogen.shell_collision_model
        == "anderson_lebedev"
    )
    np.testing.assert_array_equal(
        model.observer.atom.angular_l, model.hydrogen.atom.angular_l
    )
    assert len(_hydrogen_infrared_indices(model.hydrogen.atom, pairs)) == count
    density, temperature = jnp.full(3, 1e10), jnp.full(3, 8000.0)
    state = model.initialize(density, temperature)
    physical = model.unpack(state, density)
    # A supplied-profile observer fixture, not an atmospheric convergence test.
    solution = dict(
        **physical,
        temperature_k=temperature,
        radial_velocity_kms=jnp.array([1.0, 3.0, 8.0]),
        converged=True,
    )
    spectrum = model.spectrum_from_solution(solution)
    assert spectrum.shape == (21,) and np.isfinite(spectrum).all()
    failed = model.spectrum_from_solution(dict(solution, converged=False))
    assert np.isnan(failed).all()
    alternate = CoupledHydrogenForwardModel(
        nlayer=3,
        nmax=15,
        resolved_nmax=resolved,
        shell_collision_model="anderson_lebedev",
        quadrature_order=2,
        angle_order=2,
        wind_ngrid=33,
        **options,
    )
    assert (
        alternate.observer.shell_collision_model
        == alternate.radiation.shell_collision_model
        == "anderson_lebedev"
    )
    np.testing.assert_array_equal(
        alternate.observer.atom.angular_l, alternate.radiation.atom.angular_l
    )
    emitted = alternate.observer.spectrum_from_profiles(
        physical["bound_cm3"][::-1], temperature, solution["radial_velocity_kms"][::-1]
    )
    assert emitted.shape == (21,) and np.isfinite(emitted).all()
