"""Scoped checks of optional H completion, not atmosphere/reference equivalence.

The small population anchors use unmodified C23.01, resolved n<=3 plus
collapsed n=4..15, T=9000 K, nH=1e8 cm^-3, blackbody 10000 K diluted by
0.025, thickness 1e-4 cm, no molecules, six iterations, continuum resolution
0.05, and electron convergence 1e-6. Only topoff differs between the runs.
Binary SHA256: 3001d9a71499dba72d3cf532538f34345b5df2de56cbfedf84b3970b13002305.
These anchors do not assert agreement of the nearly cancelling line heat.
"""

from __future__ import annotations

import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape import hydrogen_completion
from escape.hydrogen_atom import atom_milne_population, atom_saha_factors
from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
from escape.hydrogen_nlte import RateCoefficients
from escape.hydrogen_rates import BOLTZMANN_CGS, LIGHT_SPEED_CGS, PLANCK_CGS

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def completion_case():
    model = HydrogenEscapeProbabilityModel(
        [0., 1e10, 2e10], nmax=6, topoff=True,
        quadrature_order=8, angle_order=4, include_free_free=False)
    temperature = jnp.array([8200., 9300.])
    electron = jnp.array([1.3e8, 2.1e8])
    proton = electron*1.1
    lte = jax.vmap(lambda t: atom_saha_factors(model.atom, t))(temperature)
    lte *= (electron*proton)[:, None]
    return model, temperature, electron, proton, lte


def test_table_precision_does_not_depend_on_import_configuration():
    enabled = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        imported = runpy.run_path(hydrogen_completion.__file__)
        jax.config.update("jax_enable_x64", True)
        reference = runpy.run_path(hydrogen_completion.__file__)
        temperature = jnp.array(9000., dtype=jnp.float64)
        actual = imported["total_radiative_recombination"](temperature)
        expected = reference["total_radiative_recombination"](temperature)
    finally:
        jax.config.update("jax_enable_x64", enabled)
    np.testing.assert_array_equal(actual, expected)


def test_default_and_explicit_off_continuum_and_receivers_agree(completion_case):
    _, t, ne, proton, lte = completion_case
    options = dict(nmax=6, quadrature_order=2, include_free_free=False)
    default = HydrogenEscapeProbabilityModel([0., 1e10, 2e10], **options)
    explicit = HydrogenEscapeProbabilityModel([0., 1e10, 2e10], topoff=False, **options)
    frequency = jnp.array([1e14, 1e15, 5e15])
    photons = jnp.ones((2, 3))*1e10

    def evaluate(model):
        return (model.continuum_coefficients_at(t, lte*2, proton, ne, frequency),
                model.photoionization_from_photon_flux(
                    t, lte*2, proton, ne, frequency, photons))

    for actual, expected in zip(jax.tree.leaves(evaluate(default)),
                                jax.tree.leaves(evaluate(explicit))):
        np.testing.assert_array_equal(actual, expected)


def test_lte_and_signed_photon_number_and_energy_ledgers(completion_case):
    model, t, ne, proton, lte = completion_case
    # Only the highest shell can absorb between these adjacent thresholds.
    thresholds = model.atom.ionization_energy_erg/PLANCK_CGS
    frequency = jnp.array([(thresholds[-1]+thresholds[-2])/2])
    chi, eta = model.continuum_coefficients_at(t, lte, proton, ne, frequency)
    planck = (2*PLANCK_CGS/LIGHT_SPEED_CGS**2*frequency**3
              / jnp.expm1(PLANCK_CGS*frequency/(BOLTZMANN_CGS*t[:, None])))
    np.testing.assert_allclose(eta, chi*planck, rtol=3e-12, atol=0)

    photons = jnp.array([[2e10], [-3e10]])
    bound = lte*2
    receiver = model.photoionization_from_photon_flux(
        t, bound, proton, ne, frequency, photons)
    chi, _ = model.continuum_coefficients_at(t, bound, proton, ne, frequency)
    events = (bound*receiver["photoionization_s"]
              -(ne*proton)[:, None]*receiver["recombination_cm3_s"])
    np.testing.assert_allclose(events.sum(1), (chi*photons).sum(1), rtol=3e-12, atol=0)
    energy = PLANCK_CGS*frequency[0]-model.atom.ionization_energy_erg[-1]
    np.testing.assert_allclose(receiver["net_erg_cm3_s"], events.sum(1)*energy,
                               rtol=3e-12, atol=0)


@pytest.mark.parametrize("parameter", [0, 1, 2])
def test_temperature_density_and_signed_flux_derivatives(completion_case, parameter):
    model, t, ne, proton, lte = completion_case
    thresholds = model.atom.ionization_energy_erg/PLANCK_CGS
    frequency = jnp.array([1.08*thresholds[-1], .98*thresholds[-2],
                           1.03*thresholds[-2]])
    photons = jnp.array([[2e10, -1e10, 3e10], [-3e10, 2e10, 1e10]])

    @jax.jit
    def evaluate(z):
        return model.photoionization_from_photon_flux(
            t*jnp.exp(z[0]), lte*2, proton, ne*jnp.exp(z[1]),
            frequency, photons*jnp.exp(z[2]))

    zero, direction = jnp.zeros(3), jnp.eye(3)[parameter]
    _, tangent = jax.jvp(evaluate, (zero,), (direction,))
    step = 1e-4
    plus, minus = evaluate(step*direction), evaluate(-step*direction)
    # Compare each physical output separately. A large photo-rate block must
    # not hide an incorrect, much smaller kinetic-energy derivative.
    for name, derivative in tangent.items():
        finite_difference = (plus[name]-minus[name])/(2*step)
        magnitude = float(jnp.max(abs(derivative)))
        assert np.isfinite(derivative).all() and np.isfinite(finite_difference).all()
        if magnitude == 0:
            np.testing.assert_array_equal(finite_difference, derivative)
        else:
            assert float(jnp.max(abs(derivative-finite_difference)))/magnitude < 5e-5


def test_artificial_top_collision_pair_is_excluded_from_heat(completion_case):
    model, t, ne, proton, lte = completion_case
    levels = len(model.atom.principal_n)
    zero = jnp.zeros((2, levels))
    square = jnp.zeros((2, levels, levels))
    rates = RateCoefficients(square, square, zero, zero, zero, zero)
    inverse = jax.vmap(lambda temp: atom_milne_population(
        model.atom, temp, model.grid.frequency_hz))(t)
    inverse_line = jax.vmap(lambda temp: atom_milne_population(
        model.atom, temp, model.atom.line_frequency_hz))(t)
    occupation = jnp.zeros((2, len(model.grid.frequency_hz)))
    photons = jnp.zeros((2, len(model.atom.line_frequency_hz)))

    @jax.jit
    def thermal(coefficients):
        return model._thermal_exchange(
            t, lte*2, proton, ne, coefficients, occupation, inverse, photons, inverse_line)

    baseline = thermal(rates)
    enhanced = rates._replace(electron_ionization_cm3_s=zero.at[:, -1].set(1e-3),
                             three_body_recombination_cm6_s=zero.at[:, -1].set(1e-10))
    np.testing.assert_array_equal(jnp.stack(thermal(enhanced)), jnp.stack(baseline))
    lower = rates._replace(electron_ionization_cm3_s=zero.at[:, -2].set(1e-3))
    assert jnp.all(thermal(lower)[1] > baseline[1])


def test_pressure_lowered_active_atom_is_rejected(completion_case):
    model, t, ne, proton, lte = completion_case
    assert model._full_atom_valid(t, lte, proton, ne)
    assert not model._full_atom_valid(t, lte*1e25, proton, ne)
    with pytest.raises(ValueError, match="collapsed top shell"):
        HydrogenEscapeProbabilityModel([0., 1e5], nmax=3, topoff=True)


# Final unmodified C23 species tables (rounded by its save command).
# Off/on model.levels SHA256 respectively:
# 554b9efd982fb4afa8d4b19d10390c05dea820e36ac2af0a4534876c3f0051f2
# f22689cd7d0f3f689f3a854840c76d39952fd4dce6643076db6ed695da9d0c8f
_CLOUDY_BOUND = {
    False: [2851.29, .0115597, .00156156, 5.18801e-5, .00017085, 4.45558e-5,
            .000247489, .000250903, .000284441, .000332696, .000391884,
            .000459105, .000532899, .000611176, .000692782, .00077578,
            .000858758, .000943579],
    True: [2891.98, .0117237, .00158385, 5.26027e-5, .000173289, 4.5221e-5,
           .000251101, .000254715, .000289045, .000338581, .000399672,
           .000469717, .000547903, .000633621, .000729592, .000844908,
           .00101222, .0013933],
}


@pytest.mark.parametrize("topoff", [False, True])
def test_small_radiative_cloudy_population_anchor(topoff):
    model = HydrogenEscapeProbabilityModel(
        [0., 1e-4], nmax=15, topoff=topoff, quadrature_order=8, angle_order=8,
        shell_collision_model="anderson_lebedev", line_overlap_model="coincident")
    expected = np.asarray(_CLOUDY_BOUND[topoff])
    electron = 99997100. if topoff else 99997200.
    # Reference values initialize Newton; every evaluated rate/residual is native.
    initial = jnp.log(jnp.asarray(expected/electron))[None, :]
    solution = model.solve(1e8, 9000., model.boundary_blackbody(10000., .025),
                           initial, iterations=8)
    assert solution["converged"] and solution["residual_norm"] < 1e-9
    np.testing.assert_allclose(solution["bound_cm3"][0], expected, rtol=.05, atol=0)
    np.testing.assert_allclose(solution["electron_cm3"][0], electron, rtol=.05, atol=0)
