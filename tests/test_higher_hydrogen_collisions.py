"""Independent C23 anchors and derivative checks for n=4,5 collisions."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.atmosphere import HydrogenAtmosphereModel
from escape.hydrogen_atom import atom_collision_rates, make_hydrogen_atom, solve_atom_populations
from escape.hydrogen_rates import BOLTZMANN_CGS, planck_occupation
from test_hydrogen_atom import continuum_grid
from test_hydrogen_forward import offline_nist  # noqa: F401

jax.config.update("jax_enable_x64", True)


def test_default_matches_independently_extracted_cloudy_table():
    rows = np.loadtxt(Path(__file__).with_name("data") / "cloudy_n45_collisions.csv",
                      delimiter=",", skiprows=1)
    atom = make_hydrogen_atom(5)
    rates = jax.jit(jax.vmap(lambda t: atom_collision_rates(
        atom, t, low_temperature_extrapolation="cloudy",
    )[0]))(jnp.asarray(rows[:, 0]))
    lower_indices = {(1, 0): 0, (2, 0): 1, (2, 1): 2,
                     (3, 0): 3, (3, 1): 4, (3, 2): 5, (4, -1): 6}
    for i, (temperature, upper_n, lower_n, lower_l, strength) in enumerate(rows):
        upper, lower = int(upper_n) + 2, lower_indices[int(lower_n), int(lower_l)]
        expected = 8.6291321808518402e-6 * strength / (2 * upper_n**2 * np.sqrt(temperature))
        np.testing.assert_allclose(rates[i, upper, lower], expected, rtol=3e-7)


def test_default_changes_only_n45_and_preserves_detailed_balance():
    atom = make_hydrogen_atom(15)
    revised, ion, three = atom_collision_rates(atom, 8000.)
    legacy, old_ion, old_three = atom_collision_rates(atom, 8000., shell_collision_model="vs80")
    allowed = np.zeros(revised.shape, bool)
    for upper in (6, 7):
        allowed[upper, :upper] = allowed[:upper, upper] = True
    np.testing.assert_array_equal(np.asarray(revised)[~allowed], np.asarray(legacy)[~allowed])
    np.testing.assert_array_equal(ion, old_ion)
    np.testing.assert_array_equal(three, old_three)
    assert revised[7, 5] > 15 * legacy[7, 5]
    populations = atom.statistical_weights * jnp.exp(-atom.excitation_energy_erg / (BOLTZMANN_CGS * 8000.))
    flux = populations[:, None] * revised
    np.testing.assert_allclose(flux, flux.T, rtol=3e-14, atol=1e-30)
    for nmax in (3, 4):
        assert np.isfinite(atom_collision_rates(make_hydrogen_atom(nmax), 8000.)[0]).all()
    with pytest.raises(ValueError, match="shell_collision_model"):
        atom_collision_rates(atom, 8000., shell_collision_model="unknown")


def test_new_collision_data_flow_through_implicit_population_derivative():
    atom = make_hydrogen_atom(5)
    frequency, weights = continuum_grid(atom, 8)
    line = planck_occupation(atom.line_frequency_hz, 10000., .025)
    continuum = planck_occupation(frequency, 10000., .025)

    @jax.jit
    def signal(log_temperature):
        result = solve_atom_populations(1e12, atom, 8000 * jnp.exp(log_temperature),
                                        line, continuum, frequency, weights)
        return jnp.log(jnp.sum(result.bound_cm3[3:6]))

    step = 1e-4
    automatic = jax.grad(signal)(0.)
    finite = (signal(step) - signal(-step)) / (2 * step)
    assert np.isfinite(automatic) and abs(finite) > .01
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-7)


def test_atmosphere_and_observer_use_selected_shell_collisions():
    kwargs = dict(nlayer=2, nmax=5, nnu=21, nlon=2, nlat=2, wind_ngrid=17,
                  quadrature_order=2, angle_order=2)
    model = HydrogenAtmosphereModel(**kwargs)
    legacy = HydrogenAtmosphereModel(**kwargs, shell_collision_model="vs80")
    assert model.observer.shell_collision_model == model.radiation.shell_collision_model == "anderson_n45"
    assert legacy.observer.shell_collision_model == legacy.radiation.shell_collision_model == "vs80"
    density, temperature = jnp.full(2, 1e10), jnp.full(2, 8000.)
    fractions = jnp.broadcast_to(jnp.array([.2, 1e-6, 3e-6, 1e-8, 3e-8, 5e-8, 1e-8, 1e-8]), (2, 8))
    inner = model.radiation.boundary_blackbody(4500.)
    outer = model.radiation.boundary_blackbody(10000., .05)
    # This is the actual radiation/rate evaluation used by the atmosphere root.
    revised = jax.jit(model.radiation.radiation_state)(density, temperature, fractions, inner, outer)
    previous = jax.jit(legacy.radiation.radiation_state)(density, temperature, fractions, inner, outer)
    assert np.isfinite(revised.rates.electron_bound_bound_cm3_s).all()
    assert np.all(revised.rates.electron_bound_bound_cm3_s[:, 7, 5]
                  > 15 * previous.rates.electron_bound_bound_cm3_s[:, 7, 5])
