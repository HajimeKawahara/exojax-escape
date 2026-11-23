"""Independent C23 shell kernels, actual rates and an implicit population root."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from escape.hydrogen_atom import (
    atom_collision_rates,
    make_hydrogen_atom,
    solve_atom_populations,
)
from escape.hydrogen_collisions import lebedev_collision_strength
from escape.hydrogen_rates import BOLTZMANN_CGS, planck_occupation
from test_hydrogen_atom import continuum_grid

jax.config.update("jax_enable_x64", True)


def test_independent_c23_kernel_and_resolved_distribution():
    data = np.load(Path(__file__).with_name("data") / "cloudy_lebedev_collisions.npz")
    rows = data["kernel"]
    strength = jax.jit(lebedev_collision_strength)(
        rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 4]
    )
    np.testing.assert_allclose(strength, rows[:, 5], rtol=5e-14)
    factor = np.where(rows[:, 2] <= 3, (2 * rows[:, 3] + 1) / rows[:, 2] ** 2, 1.0)
    np.testing.assert_allclose(factor, rows[:, 6], rtol=2e-15)
    actual = data["operator_electron_rates"]
    atom = make_hydrogen_atom(15)
    rates = atom_collision_rates(
        atom, 8000.0, shell_collision_model="anderson_lebedev"
    )[0]
    upper, lower = actual[:, 0].astype(int), actual[:, 1].astype(int)
    # The executable's atom uses degenerate low-level energies, whereas ours
    # retains the selected NIST energies. Kernel identity above is separate.
    np.testing.assert_allclose(rates[upper, lower], actual[:, 2], rtol=1e-5)


def test_selected_shell_model_preserves_low_rates_and_detailed_balance():
    atom = make_hydrogen_atom(15)
    previous = atom_collision_rates(atom, 8000.0)
    revised = atom_collision_rates(
        atom, 8000.0, shell_collision_model="anderson_lebedev"
    )
    np.testing.assert_array_equal(revised[0][:6, :6], previous[0][:6, :6])
    np.testing.assert_array_equal(revised[1], previous[1])
    np.testing.assert_array_equal(revised[2], previous[2])
    populations = atom.statistical_weights * jnp.exp(
        -atom.excitation_energy_erg / (BOLTZMANN_CGS * 8000.0)
    )
    flux = populations[:, None] * revised[0]
    np.testing.assert_allclose(flux, flux.T, rtol=3e-14, atol=1e-30)
    assert (
        np.max(
            np.abs(
                np.asarray(revised[0]) / np.where(previous[0] > 0, previous[0], 1.0) - 1
            )
        )
        > 0.1
    )
    assert np.isfinite(
        atom_collision_rates(
            make_hydrogen_atom(3), 8000.0, shell_collision_model="anderson_lebedev"
        )[0]
    ).all()


def test_new_shell_rates_pass_through_implicit_population_derivative():
    atom = make_hydrogen_atom(6)
    frequency, weights = continuum_grid(atom, 8)
    line = planck_occupation(atom.line_frequency_hz, 10000.0, 0.025)
    continuum = planck_occupation(frequency, 10000.0, 0.025)

    @jax.jit
    def signal(log_temperature):
        result = solve_atom_populations(
            1e12,
            atom,
            8000 * jnp.exp(log_temperature),
            line,
            continuum,
            frequency,
            weights,
            shell_collision_model="anderson_lebedev",
        )
        return jnp.log(jnp.sum(result.bound_cm3[3:6]))

    step = 1e-4
    automatic = jax.grad(signal)(0.0)
    finite = (signal(step) - signal(-step)) / (2 * step)
    assert np.isfinite(automatic) and abs(finite) > 0.01
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-7)
