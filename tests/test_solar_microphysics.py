"""Independent reference anchors and conserved composition/continuum checks."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from escape.free_free import FreeFreeContinuum, RYDBERG_ERG
from escape.hydrogen_continuum import planck_intensity_hz
from escape.hydrogen_molecules import (
    MolecularPhotoRates, hminus_cross_section, hminus_equilibrium_factor,
    molecular_continuum_coefficients, molecular_reaction_balance,
)
from escape.hydrogen_rates import PLANCK_CGS
from escape.solar import SolarCompositionModel

jax.config.update("jax_enable_x64", True)


def test_gaunt_anchors_from_unmodified_c23_function():
    # C23.01 t_gaunt::gauntff calls, not a second implementation of interpolation.
    # The 900-point independent comparison and source/binary hashes are archived.
    inputs = [(1, 11125.895155025457, .47203926717932476),
              (1, 5226.643638454249, .0795701347178302),
              (2, 13292.715722562465, .0006479840988285749),
              (3, 14263.277349046142, .004506970987392428),
              (3, 9351.159845533633, .05274074472475319)]
    expected = [1.1283423062284914, 1.1187243029969978, 2.0204357895598233,
                1.4037823248234162, 1.0944892121223242]
    model = FreeFreeContinuum()
    actual = [model.gaunt_factor(q, t, energy*RYDBERG_ERG/PLANCK_CGS)
              for q, t, energy in inputs]
    np.testing.assert_allclose(actual, expected, rtol=2e-13)


def test_shared_free_free_kirchhoff_and_temperature_derivative():
    model = FreeFreeContinuum()
    frequency = jnp.geomspace(1e12, 1e16, 29)
    ion = jnp.array([1e11, 2e10, 1e9])
    chi, eta = model.coefficients(8000., 2e11, ion, 1e12, frequency)
    np.testing.assert_allclose(eta/chi, planck_intensity_hz(frequency, 8000.), rtol=2e-15)
    assert np.all(chi > 0)
    def observable(logt):
        return jnp.log(model.coefficients(8000*jnp.exp(logt), 2e11, ion, 1e12, frequency)[0])
    derivative = jax.jacfwd(observable)(0.)
    numerical = (observable(1e-5)-observable(-1e-5))/2e-5
    np.testing.assert_allclose(derivative, numerical, rtol=2e-7, atol=1e-8)


def test_hminus_inverse_process_recovers_its_declared_lte_continuum():
    frequency = jnp.array([.06, .1, .3, 1.])*RYDBERG_ERG/PLANCK_CGS
    t, neutral, electron = 7000., 1e12, 1e10
    density = neutral*electron*hminus_equilibrium_factor(t)
    chi, eta = molecular_continuum_coefficients(t, neutral, electron, density, 0., frequency)
    np.testing.assert_allclose(eta/chi, planck_intensity_hz(frequency, t), rtol=3e-15)
    assert hminus_cross_section(.05) == 0.
    assert hminus_cross_section(2.) == 0.


def test_chemical_reactions_preserve_hydrogen_nuclei_and_temperature_derivatives():
    molecules = jnp.array([1e3, 1e3, 5e5, 6e4])
    photo = MolecularPhotoRates(1e5, 1e-16, 1e-7, 1e-28, 1e5, 1e-7)
    def evaluate(logt):
        return molecular_reaction_balance(5500*jnp.exp(logt), 1e13, 1e13, 1e7, 1e5,
            1e10, 1.1e10, molecules, photo, jnp.array([1., 1., 1.]),
            neutral_he_cm3=1e12, singly_charged_metals_cm3=1e9)
    result = evaluate(0.)
    net = result.production_cm3_s-result.destruction_cm3_s
    atomic = (result.hydrogen_production_cm3_s-result.hydrogen_destruction_cm3_s
              +result.proton_production_cm3_s-result.proton_destruction_cm3_s)
    scale = result.production_cm3_s.sum()+result.destruction_cm3_s.sum()
    assert abs(atomic+jnp.dot(net, jnp.array([1., 2., 2., 2.]))) < 2e-15*scale
    function = lambda value: jnp.log(evaluate(value).production_cm3_s)
    np.testing.assert_allclose(jax.jacfwd(function)(0.),
        (function(1e-5)-function(-1e-5))/2e-5, rtol=2e-7, atol=1e-8)


def test_composition_uses_actual_electrons_molecules_and_conserved_elements():
    model = SolarCompositionModel([0., 1e4, 3e4], nmax=3, quadrature_order=2,
                                   atoms=(("C I", 6, 0),))
    n = jnp.array([1e12, 2e12])
    state = model.initialize(n, jnp.array([7000., 8000.]))
    result = model.unpack(state, n)
    h = (result["bound_cm3"].sum(1)+result["proton_cm3"]
         +result["molecule_cm3"]@jnp.array([1., 2., 2., 2.]))
    np.testing.assert_allclose(h, n, rtol=3e-15)
    np.testing.assert_allclose(result["metal_ion_cm3"].sum(2), n[:, None]*model.abundance, rtol=3e-15)
    np.testing.assert_allclose(result["helium_cm3"].sum(1), .1*n, rtol=3e-15)
    electron = (result["proton_cm3"]+result["helium_cm3"]@jnp.array([0., 1., 2.])
                +(result["metal_ion_cm3"]@jnp.array([0., 1., 2.])).sum(1)
                +result["molecule_cm3"][:, 1]-result["molecule_cm3"][:, 0])
    np.testing.assert_allclose(result["electron_cm3"], electron, rtol=3e-15)
