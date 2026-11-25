"""Thermal energy is independent of the large trapped radiative turnover."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.coupled_hydrogen import frequency_grid
from escape.hydrogen_atom import make_hydrogen_atom, atom_rate_coefficients, solve_atom_populations
from escape.hydrogen_continuum import total_radiative_energy, thermal_energy_exchange
from escape.hydrogen_rates import planck_occupation

jax.config.update("jax_enable_x64", True)


def test_thermal_and_photon_energy_differ_only_by_atomic_energy_production():
    atom = make_hydrogen_atom(5)
    grid = frequency_grid(atom, order=8)
    temperature, electron = 8000., 1e12
    line = planck_occupation(atom.line_frequency_hz, 10000., .025)
    continuum = planck_occupation(grid.frequency_hz, 10000., .025)
    rates = atom_rate_coefficients(atom, temperature, line, continuum,
                                   grid.frequency_hz, grid.weight_hz, electron_cm3=electron)
    # Intentionally out of equilibrium; the internal energy term cannot vanish.
    bound = jnp.array([1e12, 1e5, 3e5, 1e4, 3e4, 5e4, 2e4, 1e4])
    photon = total_radiative_energy(atom, temperature, bound, electron, electron,
                                    line, continuum, grid.frequency_hz, grid.weight_hz)
    thermal = thermal_energy_exchange(atom, temperature, bound, electron, electron, rates,
                                      continuum, grid.frequency_hz, grid.weight_hz)
    matrix = rates.radiative_bound_bound_s + electron * rates.electron_bound_bound_cm3_s
    production = (bound @ matrix - bound * matrix.sum(1)
                  - bound * (rates.photoionization_s + electron * rates.electron_ionization_cm3_s)
                  + electron**2 * (rates.radiative_recombination_cm3_s
                                   + electron * rates.three_body_recombination_cm6_s))
    atomic_energy = -production @ atom.ionization_energy_erg
    np.testing.assert_allclose(photon - thermal.net_erg_cm3_s, atomic_energy, rtol=3e-13)
    np.testing.assert_allclose(thermal.heating_erg_cm3_s - thermal.cooling_erg_cm3_s,
                               thermal.net_erg_cm3_s, rtol=2e-14)


def test_thermal_and_radiative_energy_agree_in_statistical_equilibrium():
    atom = make_hydrogen_atom(5)
    grid = frequency_grid(atom, order=8)
    line = planck_occupation(atom.line_frequency_hz, 10000., .025)
    continuum = planck_occupation(grid.frequency_hz, 10000., .025)
    solution = solve_atom_populations(1e12, atom, 9000., line, continuum,
                                      grid.frequency_hz, grid.weight_hz)
    assert solution.converged
    electron = solution.electron_cm3
    rates = atom_rate_coefficients(atom, 9000., line, continuum, grid.frequency_hz,
                                   grid.weight_hz, electron_cm3=electron)
    thermal = thermal_energy_exchange(atom, 9000., solution.bound_cm3, electron, electron,
                                      rates, continuum, grid.frequency_hz, grid.weight_hz)
    photon = total_radiative_energy(atom, 9000., solution.bound_cm3, electron, electron,
                                    line, continuum, grid.frequency_hz, grid.weight_hz)
    np.testing.assert_allclose(thermal.net_erg_cm3_s, photon, rtol=2e-9)
