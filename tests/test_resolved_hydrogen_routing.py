"""Selected orbital resolution reaches radiation, IR receivers and observers."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.coupled_forward import CoupledHydrogenForwardModel
from escape.hydrogen_atom import make_hydrogen_atom
from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
from escape.hydrogen_rates import BOLTZMANN_CGS, PLANCK_CGS
from escape.nlte_forward import HydrogenNLTEForwardModel
from escape.solar import SolarAtmosphereModel, SolarCompositionModel, _hydrogen_infrared_indices
from test_hydrogen_forward import offline_nist  # noqa: F401

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('resolved, count', [(3, 49), (4, 52), (5, 67)])
def test_principal_pair_selection_retains_all_orbital_lines(resolved, count):
    base = make_hydrogen_atom(15)
    selected = np.flatnonzero(np.asarray(base.line_frequency_hz) * PLANCK_CGS <= 4000 * BOLTZMANN_CGS)
    pairs = list(zip(np.asarray(base.principal_n[base.line_upper[selected]]),
                     np.asarray(base.principal_n[base.line_lower[selected]])))
    atom = make_hydrogen_atom(15, resolved_nmax=resolved)
    indices = _hydrogen_infrared_indices(atom, pairs)
    expected = np.flatnonzero(np.asarray(atom.line_frequency_hz) * PLANCK_CGS <= 4000 * BOLTZMANN_CGS)
    assert len(indices) == count
    np.testing.assert_array_equal(np.sort(indices), expected)
    if resolved == 3:
        np.testing.assert_array_equal(indices, selected)
    with pytest.raises(ValueError, match='distinct'):
        _hydrogen_infrared_indices(atom, [(5, 4), (5, 4)])
    with pytest.raises(ValueError, match='retained'):
        _hydrogen_infrared_indices(atom, [(16, 15)])


def test_solar_composition_uses_all_seven_5_to_4_receivers():
    model = SolarCompositionModel(
        [0., 1e4, 3e4], nmax=5, resolved_nmax=5, quadrature_order=2, angle_order=2,
        atoms=(("C I", 6, 0),), oxygen_overlap=False,
        core_escape_model='continuous', shielding_model='continuous_core',
        resolved_infrared_lines=((5, 4),),
        infrared_options=dict(quadrature_order=2, angle_order=4))
    assert model.hydrogen.atom.resolved_nmax == 5
    assert len(model.infrared_indices) == 7
    assert len(model.infrared_groups) == 1
    temperature, density = jnp.array([8000., 8200.]), jnp.array([1e10, 2e10])
    state = model.initialize(density, temperature)
    incoming = model.hydrogen.boundary_blackbody(10000., .025)

    def evaluate():
        target, p = model.evaluate(state, density, temperature, incoming)
        return target, p['valid'], p['radiation']['rates'].radiative_bound_bound_s, p['infrared'][0]

    target, valid, rates, group = jax.jit(evaluate)()
    assert valid
    assert np.isfinite(target).all()
    atom = model.hydrogen.atom
    indices = jnp.asarray(model.infrared_group_indices[0])
    np.testing.assert_allclose(rates[:, atom.line_lower[indices], atom.line_upper[indices]],
                               group['pump_up_s'], rtol=1e-12)
    np.testing.assert_allclose(rates[:, atom.line_upper[indices], atom.line_lower[indices]],
                               group['radiative_down_s'], rtol=1e-12)


def test_solar_and_alternate_observer_atoms_match():
    options = dict(nnu=31, nlon=4, nlat=2, continuum_points=2)
    model = SolarAtmosphereModel(
        [1., 1.5, 2., 3.], nmax=5, resolved_nmax=5,
        atoms=(("C I", 6, 0),), quadrature_order=2, angle_order=2,
        wind_ngrid=33, observer_options=options)
    np.testing.assert_array_equal(model.observer.atom.principal_n, model.hydrogen.atom.principal_n)
    np.testing.assert_array_equal(model.observer.atom.angular_l, model.hydrogen.atom.angular_l)
    alternate = CoupledHydrogenForwardModel(
        nlayer=3, nmax=5, resolved_nmax=4, quadrature_order=2, angle_order=2,
        wind_ngrid=33, **options)
    np.testing.assert_array_equal(alternate.observer.atom.principal_n,
                                  alternate.radiation.atom.principal_n)
    np.testing.assert_array_equal(alternate.observer.atom.angular_l,
                                  alternate.radiation.atom.angular_l)
    with pytest.raises(ValueError, match='same resolved_nmax'):
        SolarAtmosphereModel(
            [1., 1.5, 2., 3.], nmax=5, resolved_nmax=5, atoms=(("C I", 6, 0),),
            quadrature_order=2, angle_order=2, observer_options={'resolved_nmax': 3})


def test_resolved_constructor_collision_option_errors_are_explicit():
    with pytest.raises(ValueError, match='Anderson'):
        HydrogenEscapeProbabilityModel([0., 1e4], nmax=5, resolved_nmax=4,
                                       shell_collision_model='vs80')
    with pytest.raises(ValueError, match='Anderson'):
        HydrogenNLTEForwardModel(nmax=5, resolved_nmax=4, shell_collision_model='vs80')
    # The existing default orbital structure still supports the legacy option.
    assert HydrogenEscapeProbabilityModel([0., 1e4], nmax=5,
        shell_collision_model='vs80').atom.resolved_nmax == 3


@pytest.mark.parametrize('value', [None, '4', True, 4.0])
@pytest.mark.parametrize('collision', ['anderson_n45', 'vs80'])
def test_invalid_resolution_reaches_constructor_value_error(value, collision):
    with pytest.raises(ValueError, match='resolved_nmax'):
        HydrogenEscapeProbabilityModel([0., 1e4], nmax=5, resolved_nmax=value,
                                       shell_collision_model=collision)
