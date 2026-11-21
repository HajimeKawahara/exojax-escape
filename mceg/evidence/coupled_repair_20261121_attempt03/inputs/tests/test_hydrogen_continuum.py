"""Radiative energy accounting, LTE identities and optional trace continuum."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_atom import (
    atom_cross_sections, atom_milne_population, atom_rate_coefficients,
    atom_saha_factors, make_hydrogen_atom,
)
from escape.hydrogen_continuum import (
    continuum_coefficients, hminus_lte_coefficients, hminus_trace_fraction,
    hydrogen_free_free_coefficients, planck_intensity_hz, total_radiative_energy,
)
from escape.hydrogen_rates import (
    BOLTZMANN_CGS, LIGHT_SPEED_CGS, PLANCK_CGS, TWO_PHOTON_A_S, planck_occupation,
)
from test_hydrogen_atom import continuum_grid

jax.config.update("jax_enable_x64", True)


def test_ion_free_free_bolometric_emission_matches_analytic_integral():
    temperature, electrons, protons = 9000., 1e10, 2e10
    x, w = np.polynomial.legendre.leggauss(128)
    # Integrate h*nu/(kT) from 0 to 50, including the low-energy free-free continuum.
    nu = jnp.asarray((x + 1) * 25 * BOLTZMANN_CGS * temperature / PLANCK_CGS)
    weights = jnp.asarray(w * 25 * BOLTZMANN_CGS * temperature / PLANCK_CGS)
    alpha, emission = hydrogen_free_free_coefficients(temperature, electrons, protons, nu)
    np.testing.assert_allclose(emission, alpha * planck_intensity_hz(nu, temperature), rtol=2e-14)
    cooling = 4 * np.pi * np.sum(np.asarray(emission * weights))
    # Standard hydrogen free-free integrated coefficient with g_B=1.
    assert cooling == pytest.approx(1.426e-27 * np.sqrt(temperature) * electrons * protons, rel=5e-4)
    assert np.all(np.asarray(alpha) > 0)


def test_bound_free_and_free_free_kirchhoff_at_saha_populations():
    atom = make_hydrogen_atom(5)
    nu, weights = continuum_grid(atom)
    temperature, electron = 8500., 1e11
    bound = electron**2 * atom_saha_factors(atom, temperature)
    alpha, emission = continuum_coefficients(atom, temperature, bound, electron, electron, nu)
    np.testing.assert_allclose(emission, alpha * planck_intensity_hz(nu, temperature), rtol=3e-13)
    total = total_radiative_energy(
        atom, temperature, bound, electron, electron,
        planck_occupation(atom.line_frequency_hz, temperature),
        planck_occupation(nu, temperature), nu, weights,
    )
    two_photon_cooling = bound[1] * TWO_PHOTON_A_S * atom.excitation_energy_erg[1]
    assert abs(float(total + two_photon_cooling)) < float(two_photon_cooling) * 1e-5


def test_full_photon_energy_matches_independent_event_accounting():
    atom = make_hydrogen_atom(5)
    nu, weights = continuum_grid(atom)
    temperature, electron = 9000., 2e10
    bound = jnp.asarray([1e12, 1e5, 2e5, 500., 300., 600., 200., 100.])
    line_eta = planck_occupation(atom.line_frequency_hz, 10000., .025)
    continuum_eta = planck_occupation(nu, 10000., .025)
    rates = atom_rate_coefficients(atom, temperature, line_eta, continuum_eta, nu, weights)
    # Bound-bound events transfer their level energy to matter, including 2 gamma.
    radiation = rates.radiative_bound_bound_s
    population_change = bound @ radiation - bound * jnp.sum(radiation, axis=1)
    line_energy = jnp.dot(atom.excitation_energy_erg, population_change)
    # Each photoionization/recombination event deposits/removes the FULL photon.
    photon_kernel = 8 * jnp.pi / LIGHT_SPEED_CGS**2 * nu[None, :]**2 * atom_cross_sections(atom, nu)
    net_events = photon_kernel * (
        bound[:, None] * continuum_eta
        - electron**2 * atom_milne_population(atom, temperature, nu) * (1 + continuum_eta)
    )
    continuum_energy = jnp.sum(net_events * PLANCK_CGS * nu * weights)
    total = total_radiative_energy(
        atom, temperature, bound, electron, electron, line_eta, continuum_eta, nu, weights,
        include_free_free=False,
    )
    np.testing.assert_allclose(total, line_energy + continuum_energy, rtol=3e-13)
    # An ionization-threshold-only energy would drop photoelectron/recombination
    # kinetic energy and is not the radiative input to the total-energy equation.
    threshold_only = jnp.sum(net_events * atom.ionization_energy_erg[:, None] * weights)
    assert abs(float((continuum_energy - threshold_only) / continuum_energy)) > 1e-3


def test_hminus_optional_trace_domain_and_kirchhoff():
    nu = LIGHT_SPEED_CGS / jnp.array([.3e-4, .6564e-4, 1.e-4, 2.e-4])
    alpha, emission = hminus_lte_coefficients(8000., 1e12, 1e11, 1.1e12, nu)
    assert np.isfinite(alpha).all() and np.asarray(alpha > 0).all()
    np.testing.assert_allclose(emission, alpha * planck_intensity_hz(nu, 8000.), rtol=1e-14)
    assert hminus_trace_fraction(8000., 1e12, 1e11, 1.1e12) < 1e-3
    assert np.isnan(hminus_lte_coefficients(11000., 1e12, 1e11, 1.1e12, nu)[0]).all()
    assert np.isnan(hminus_lte_coefficients(6000., 1e20, 1e11, 1e20, nu)[0]).all()


def test_total_radiative_energy_jit_gradient():
    atom = make_hydrogen_atom(4)
    nu, weights = continuum_grid(atom, 24)
    bound = jnp.asarray([1e12, 1e5, 2e5, 500., 300., 600., 200.])

    def energy(parameters):
        temperature, dilution = 9000 * parameters[0], .025 * parameters[1]
        return total_radiative_energy(
            atom, temperature, bound, 2e10, 2e10,
            planck_occupation(atom.line_frequency_hz, 10000., dilution),
            planck_occupation(nu, 10000., dilution), nu, weights,
            include_hminus=True,
        )

    point, step = jnp.ones(2), 1e-4
    automatic = jax.jit(jax.grad(energy))(point)
    finite = jnp.array([(energy(point + step * d) - energy(point - step * d)) / (2 * step)
                        for d in jnp.eye(2)])
    assert np.isfinite(automatic).all()
    np.testing.assert_allclose(automatic, finite, rtol=2e-6, atol=1e-12)
