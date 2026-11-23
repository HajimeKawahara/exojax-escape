"""Differentiable multilevel metal excitation from selected C23.01 atomic data.

This is an altered JAX implementation, not Cloudy. Interpolation and g-bar
fallbacks follow species2.cpp, atmdat_chianti.cpp and transition.cpp. The
original license and data-source identities accompany the packaged tables.
Element abundances, radiation and ionic closure remain caller-supplied.
"""

from __future__ import annotations

from importlib.resources import files
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

KB = 1.380649e-16
PLANCK = 6.62607015e-27
C = 2.99792458e10
ME = 9.1093837015e-28
RYDBERG = 1.0973731568160e5
ALPHA = 7.2973525693e-3
COLLISION_CONSTANT = (PLANCK**2 / (2 * np.pi * KB * ME)) ** 1.5 * KB / PLANCK
A_CONSTANT = 4 * np.pi * PLANCK * ALPHA / ME


class Atom(NamedTuple):
    energy_wavenumber: jax.Array
    weight: jax.Array
    spontaneous: jax.Array
    lower: jax.Array
    upper: jax.Array
    collision_kind: jax.Array
    collision_energy_ryd: jax.Array
    collision_scale: jax.Array
    collision_x: jax.Array
    collision_y: jax.Array
    collision_count: jax.Array
    ion_stage: jax.Array
    radiative_wavenumber: jax.Array


def load_metal_atom(species, *, levels=None):
    """Load a declared truncation of a packaged atom without external files.

    Fe II defaults to the historical 80-level control. Its packaged 716 levels
    allow explicit atomic-convergence studies; selecting more levels does not
    itself demonstrate convergence for a stellar spectrum or atmosphere.
    """
    prefix = species.replace(" ", "_") + "__"
    resource = files("escape").joinpath("data/metal_atoms.npz")
    with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
        if prefix + "weight" not in data:
            raise ValueError(f"Unsupported packaged metal atom: {species}")
        values = {name: data[prefix + name] for name in Atom._fields}
    retained = (80 if species == "Fe II" else len(values["weight"])) if levels is None else levels
    if (not isinstance(retained, (int, np.integer)) or isinstance(retained, bool)
            or not 2 <= retained <= len(values["weight"])):
        raise ValueError("Requested level count is outside the packaged atom.")
    pair = (values["lower"] < retained) & (values["upper"] < retained)
    for name, value in values.items():
        if name in ("energy_wavenumber", "weight"):
            values[name] = value[:retained]
        elif name in ("spontaneous", "radiative_wavenumber"):
            values[name] = value[:retained, :retained]
        elif name != "ion_stage":
            values[name] = value[pair]
    return Atom(*(jnp.asarray(values[name]) for name in Atom._fields))


def load_metal_colliders(species, *, levels=None):
    """Load available non-electron STOUT de-excitation tables by collider name."""
    prefix = species.replace(" ", "_") + "__"
    result = {}
    resource = files("escape").joinpath("data/metal_atoms.npz")
    with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith(prefix) and len(key.split("__")) == 3:
                _, collider, field = key.split("__")
                result.setdefault(collider, {})[field] = data[key]
    retained = 80 if species == "Fe II" and levels is None else levels
    for table in result.values():
        keep = np.ones(len(table["lower"]), dtype=bool) if retained is None else (
            (table["lower"] < retained) & (table["upper"] < retained)
        )
        for name, value in table.items():
            table[name] = jnp.asarray(value[keep])
    return result


def tabulated_collision_coefficients(atom, tables, temperature):
    """Non-electron rates [from,to], with detailed balance and no g-bar guess."""

    def interpolate(x, y, count):
        location = jnp.clip(
            jnp.searchsorted(x, temperature, side="right") - 1, 0, count - 2
        )
        fraction = jnp.clip(
            (temperature - x[location]) / (x[location + 1] - x[location]), 0.0, 1.0
        )
        return y[location] * (1 - fraction) + y[location + 1] * fraction

    value = jax.vmap(interpolate)(
        tables["collision_x"], tables["collision_y"], tables["collision_count"]
    )
    lo, hi = tables["lower"], tables["upper"]
    difference = atom.energy_wavenumber[hi] - atom.energy_wavenumber[lo]
    inverse = (
        value
        * atom.weight[hi]
        / atom.weight[lo]
        * jnp.exp(-PLANCK * C * difference / (KB * temperature))
    )
    rate = jnp.zeros_like(atom.spontaneous).at[hi, lo].set(value)
    return rate.at[lo, hi].set(inverse)


@jax.jit
def electron_collision_coefficients(atom, temperature):
    """Return q[from,to], including declared Cloudy g-bar/tiny-CS fallbacks.

    STOUT interpolates linearly in T and holds endpoint values outside its table.
    CHIANTI uses Burgess-Tully coordinates and linear, not cubic, interpolation
    as in C23.01. These conventions are not new atomic data or validation.
    """
    difference = atom.energy_wavenumber[:, None] - atom.energy_wavenumber[None, :]
    safe_difference = jnp.where(difference > 0, difference, 1.0)
    y = safe_difference * PLANCK * C / (KB * temperature)
    gf = atom.spontaneous * atom.weight[:, None] / (A_CONSTANT * safe_difference**2)
    neutral = jnp.where(
        y < 0.01,
        0.29 * (jnp.log1p(1 / y) - 0.4 / (y + 1) ** 2) * jnp.exp(-y),
        jnp.where(
            y > 10,
            0.066 / jnp.sqrt(y),
            0.015819068
            + 1.3018207 * jnp.exp(-0.5 * (jnp.log(y / 0.002689623) / 2.5486007) ** 2),
        ),
    )
    ionic = 0.6 + 0.28 * (jnp.log1p(1 / y) - 0.4 / (y + 1) ** 2)
    gaunt = jnp.where(gf >= 1e-8, jnp.where(atom.ion_stage == 1, neutral, ionic), 0.15)
    missing_cs = jnp.where(
        gf > 0, gaunt * 14.5104 * RYDBERG * gf / safe_difference, 1e-10
    )
    down = jnp.where(
        difference > 0,
        COLLISION_CONSTANT
        * missing_cs
        / (atom.weight[:, None] * jnp.sqrt(temperature)),
        0.0,
    )
    if atom.lower.size == 0:
        inverse = (
            down
            * atom.weight[:, None]
            / atom.weight[None, :]
            * jnp.exp(-jnp.maximum(difference, 0) * PLANCK * C / (KB * temperature))
        )
        return down + inverse.T
    kind, scale = atom.collision_kind, atom.collision_scale
    reduced_temperature = temperature / atom.collision_energy_ryd / 1.57888e5
    logarithmic = (kind == 1) | (kind == 4)
    coordinate = jnp.where(
        kind <= 0,
        temperature,
        jnp.where(
            logarithmic,
            1 - jnp.log(scale) / jnp.log(reduced_temperature + scale),
            reduced_temperature / (reduced_temperature + scale),
        ),
    )

    def interpolate(x, y, count, query):
        location = jnp.clip(jnp.searchsorted(x, query, side="right") - 1, 0, count - 2)
        fraction = jnp.clip(
            (query - x[location]) / (x[location + 1] - x[location]), 0.0, 1.0
        )
        return y[location] * (1 - fraction) + y[location + 1] * fraction

    scaled = jax.vmap(interpolate)(
        atom.collision_x, atom.collision_y, atom.collision_count, coordinate
    )
    upsilon = jnp.where(
        kind == 1,
        scaled * jnp.log(reduced_temperature + jnp.e),
        jnp.where(
            kind == 3,
            scaled / (reduced_temperature + 1),
            jnp.where(
                kind == 4,
                scaled * jnp.log(reduced_temperature + scale),
                jnp.where(
                    kind == 5,
                    scaled / reduced_temperature,
                    jnp.where(
                        kind == 6, 10 ** jnp.where(kind == 6, scaled, 0.0), scaled
                    ),
                ),
            ),
        ),
    )
    coefficients = jnp.where(
        kind == -1,
        scaled,
        COLLISION_CONSTANT
        * upsilon
        / (atom.weight[atom.upper] * jnp.sqrt(temperature)),
    )
    down = down.at[atom.upper, atom.lower].set(coefficients)
    inverse = (
        down
        * atom.weight[:, None]
        / atom.weight[None, :]
        * jnp.exp(-jnp.maximum(difference, 0) * PLANCK * C / (KB * temperature))
    )
    return down + inverse.T


@jax.jit
def populations(total_rate, species_density):
    """Solve closed local SE by positive GTH elimination, rates [from,to].

    Grassmann, Taksar & Heyman (1985), Operations Research 33, 1107;
    the same rearrangement is used in C23.01 atom_leveln.cpp:gthsolve.
    This JAX implementation avoids subtracting large outgoing sums in a
    nearly singular normalized rate matrix. Irreducible nonnegative rates
    are required; externally supplied ionization sources/sinks are excluded.
    """
    rate = total_rate - jnp.diag(jnp.diag(total_rate))
    size = rate.shape[0]
    indices = jnp.arange(size)

    def eliminate(step, state):
        reduced, inverse = state
        current = size - 1 - step
        active = indices < current
        reciprocal = 1 / jnp.sum(jnp.where(active, reduced[current, :], 0.0))
        update = reduced[:, current, None] * reduced[current, None, :] * reciprocal
        reduced = reduced + jnp.where(active[:, None] & active[None, :], update, 0.0)
        return reduced, inverse.at[current].set(reciprocal)

    reduced, inverse = jax.lax.fori_loop(0, size - 1, eliminate, (rate, jnp.ones(size)))

    def substitute(current, values):
        incoming = jnp.sum(
            jnp.where(indices < current, values * reduced[:, current], 0.0)
        )
        return values.at[current].set(inverse[current] * incoming)

    relative = jax.lax.fori_loop(1, size, substitute, jnp.zeros(size).at[0].set(1.0))
    return species_density * relative / relative.sum()


@jax.jit
def kinetic_line_heating(population, collision_rate, energy_wavenumber):
    """Signed collisional heat, positive for de-excitation heating [erg/cm3/s]."""
    energy = PLANCK * C * energy_wavenumber
    return jnp.sum(
        population[:, None] * collision_rate * (energy[:, None] - energy[None, :])
    )


@jax.jit
def solve_species(
    atom,
    temperature,
    electron_density,
    species_density,
    pump_up,
    loss_probability,
    additional_radiative_rate,
    additional_collision_rate,
):
    """Conditional atom kernel with explicit external closure inputs.

    pump_up has only upper-triangular entries [lower,upper]; spontaneous and
    loss_probability are lower triangular [upper,lower]. Additional radiative
    terms include explicitly supplied overlap/continuum-destruction effects.
    Cloudy supplies these quantities for the first controlled validation only.
    Their fixed-input derivatives are not atmospheric implicit derivatives.
    """
    collision = (
        electron_density * electron_collision_coefficients(atom, temperature)
        + additional_collision_rate
    )
    stimulated = pump_up.T * atom.weight[None, :] / atom.weight[:, None]
    radiative = (
        pump_up
        + stimulated
        + atom.spontaneous * loss_probability
        + additional_radiative_rate
    )
    population = populations(radiative + collision, species_density)
    return population, kinetic_line_heating(
        population, collision, atom.energy_wavenumber
    )
