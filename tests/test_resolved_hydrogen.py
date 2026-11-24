"""Orbital resolution preserves aggregate data and works inside implicit AD."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_atom import (
    atom_collision_rates, atom_cross_sections, atom_saha_factors,
    make_hydrogen_atom, solve_atom_populations,
)
from escape.hydrogen_rates import BOLTZMANN_CGS, PLANCK_CGS

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize('resolved', [4, 5])
@pytest.mark.parametrize('shell_model', ['anderson_n45', 'anderson_lebedev'])
def test_resolved_atom_aggregates_atomic_data(resolved, shell_model):
    base = make_hydrogen_atom(8)
    atom = make_hydrogen_atom(8, resolved_nmax=resolved)
    n, g = np.asarray(atom.principal_n), np.asarray(atom.statistical_weights)
    indices = [i if i < 6 else int(n[i]) + 2 for i in range(len(n))]
    summation = np.eye(len(base.principal_n))[indices]
    weights = summation.T * g / np.asarray(base.statistical_weights)[:, None]

    def radiative_matrix(selected):
        result = np.zeros((len(selected.principal_n),) * 2)
        result[np.asarray(selected.line_upper), np.asarray(selected.line_lower)] = selected.einstein_a_s
        return result

    np.testing.assert_allclose(weights @ radiative_matrix(atom) @ summation,
                               radiative_matrix(base), rtol=1e-12, atol=0)
    frequency = jnp.geomspace(1e11, 1e17, 32)
    np.testing.assert_allclose(weights @ atom_cross_sections(atom, frequency),
                               atom_cross_sections(base, frequency), rtol=1e-12, atol=0)
    for temperature in [4000., 8000., 14000.]:
        kwargs = dict(low_temperature_extrapolation='cloudy', shell_collision_model=shell_model)
        bb, ci, q3 = atom_collision_rates(atom, temperature, **kwargs)
        old_bb, old_ci, old_q3 = atom_collision_rates(base, temperature, **kwargs)
        np.testing.assert_allclose(weights @ bb @ summation, old_bb, rtol=1e-12, atol=0)
        np.testing.assert_allclose(weights @ ci, old_ci, rtol=1e-12, atol=0)
        np.testing.assert_allclose(summation.T @ q3, old_q3, rtol=1e-12, atol=0)
        lte = atom_saha_factors(atom, temperature)
        np.testing.assert_allclose(lte[:, None] * bb, (lte[:, None] * bb).T, rtol=1e-12, atol=0)
        np.testing.assert_allclose(q3, ci * lte, rtol=1e-12, atol=0)
    # A radiatively forbidden electron collision must still be represented.
    i4s = next(i for i, member in enumerate(atom.orbital_members) if member == ((4, 0),))
    assert bb[i4s, 3] > 0  # 4s -> 3s
    assert radiative_matrix(atom)[i4s, 3] == 0


@pytest.mark.parametrize('value', [True, 4.0, 2, 11, None])
def test_invalid_resolved_limit(value):
    with pytest.raises(ValueError, match='resolved_nmax'):
        make_hydrogen_atom(15, resolved_nmax=value)


def test_resolved_limit_and_collision_option():
    with pytest.raises(ValueError, match='resolved_nmax'):
        make_hydrogen_atom(4, resolved_nmax=5)
    atom = make_hydrogen_atom(5, resolved_nmax=5)
    with pytest.raises(ValueError, match='Anderson'):
        atom_collision_rates(atom, 8000., shell_collision_model='vs80')
    with pytest.raises(ValueError, match='shell_collision_model'):
        atom_collision_rates(atom, 8000., shell_collision_model='invalid')


def test_explicit_atom_pytree_through_implicit_population_gradient():
    atom = make_hydrogen_atom(15, resolved_nmax=5)
    edges = jnp.geomspace(1e11, 1e17, 33)
    frequency = jnp.sqrt(edges[1:] * edges[:-1])
    occupation = .025 / jnp.expm1(PLANCK_CGS * frequency / (BOLTZMANN_CGS * 10000.))
    line_occupation = .025 / jnp.expm1(
        PLANCK_CGS * atom.line_frequency_hz / (BOLTZMANN_CGS * 10000.))

    def observable(selected_atom, log_temperature):
        state = solve_atom_populations(
            1e10, selected_atom, jnp.exp(log_temperature), line_occupation,
            occupation, frequency, jnp.diff(edges),
            low_temperature_extrapolation='cloudy', shell_collision_model='anderson_lebedev')
        return jnp.log(state.bound_cm3[3:6].sum()), state

    compiled = jax.jit(observable)
    log_t = jnp.log(8000.)
    value, state = compiled(atom, log_t)
    assert state.converged
    assert state.rate_residual_norm < 1e-10
    derivative = jax.jit(jax.grad(lambda a, t: observable(a, t)[0], argnums=1))(atom, log_t)
    step = 1e-4
    finite = (compiled(atom, log_t + step)[0] - compiled(atom, log_t - step)[0]) / (2 * step)
    np.testing.assert_allclose(derivative, finite, rtol=5e-5, atol=1e-8)
    assert np.isfinite(value)
