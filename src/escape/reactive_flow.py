"""Conservative finite-domain hydrogen flow with advected populations.

This is a prescribed-inflow branch, not a transonic mass-loss eigenproblem.
Radiation supplies full photon energy, while excitation and ionization energy
are stored in the gas. A callback may supply local radiation; the public
residual also supports a joint solve with nonlocal radiation transport.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .hydrogen_nlte import RateCoefficients
from .parker import PROTON_MASS_CGS

BOLTZMANN_CGS = 1.380649e-16
ADIABATIC_INDEX = 5.0 / 3.0


class FlowState(NamedTuple):
    n_h_cm3: jax.Array
    temperature_k: jax.Array
    velocity_cm_s: jax.Array
    bound_fraction: jax.Array
    ionized_fraction: jax.Array
    mean_molecular_weight: jax.Array
    mach_number: jax.Array


class FlowSolution(NamedTuple):
    state: jax.Array
    profile: FlowState
    mass_loss_g_s: jax.Array
    bernoulli_erg_g: jax.Array
    residual_norm: jax.Array
    valid: jax.Array
    converged: jax.Array


def spherical_shell_volume(radius_cm):
    """Return shell volumes per steradian: (r_outer**3-r_inner**3)/3."""
    radius = jnp.asarray(radius_cm)
    left, right = radius[:-1], radius[1:]
    return (right - left) * (right**2 + right * left + left**2) / 3


def _decode(state, radius, density_base, temperature_base, velocity_base, radius_base):
    velocity = velocity_base * jnp.exp(state[..., 0])
    temperature = temperature_base * jnp.exp(state[..., 1])
    fractions = jax.nn.softmax(jnp.concatenate(
        (state[..., 2:], jnp.zeros_like(state[..., :1])), axis=-1
    ), axis=-1)
    bound, ionized = fractions[..., :-1], fractions[..., -1]
    density = density_base * (radius_base / radius)**2 * velocity_base / velocity
    sound_squared = BOLTZMANN_CGS * temperature * (1 + ionized) / PROTON_MASS_CGS
    return FlowState(
        density, temperature, velocity, bound, ionized,
        1 / (1 + ionized), velocity / jnp.sqrt(ADIABATIC_INDEX * sound_squared),
    )


def decode_flow_state(
    state, radius_cm, n_h_base_cm3, temperature_base_k, velocity_base_cm_s,
):
    """Decode log(v/v0), log(T/T0), log(f_i/x) at each radial node.

    The proton fraction x and bound fractions share a softmax, so charge and
    nucleus conservation hold identically. Density follows exact spherical
    mass conservation. Mach number uses the frozen-composition adiabatic
    sound speed sqrt((5/3) P/rho), not the isothermal sound speed.
    """
    radius = jnp.asarray(radius_cm)
    return _decode(
        jnp.asarray(state), radius, n_h_base_cm3, temperature_base_k,
        velocity_base_cm_s, radius[0],
    )


def _base_state(bound_fraction_base):
    bound = jnp.asarray(bound_fraction_base)
    return jnp.concatenate((jnp.zeros(2), jnp.log(bound / (1 - jnp.sum(bound)))))


def species_production_s(profile, rates):
    """Return net bound-fraction production and total turnover in s^-1.

    Electron and proton densities are x*nH; every collisional and inverse
    rate therefore responds to the advected ionization fraction.
    """
    electron = profile.ionized_fraction * profile.n_h_cm3
    transition = (rates.radiative_bound_bound_s
                  + electron[..., None, None] * rates.electron_bound_bound_cm3_s)
    loss = (jnp.sum(transition, axis=-1) + rates.photoionization_s
            + electron[..., None] * rates.electron_ionization_cm3_s)
    source = (electron**2 / profile.n_h_cm3)[..., None] * (
        rates.radiative_recombination_cm3_s
        + electron[..., None] * rates.three_body_recombination_cm6_s
    )
    incoming = jnp.einsum("...ij,...i->...j", transition, profile.bound_fraction)
    outgoing = loss * profile.bound_fraction
    return source + incoming - outgoing, source + incoming + outgoing


def bernoulli_energy(
    profile, radius_cm, excitation_energy_erg, ionization_energy_erg,
    gravitational_parameter_cm3_s2, tidal_coefficient_s2=0.0,
):
    """Return kinetic + enthalpy + potential energy per mass [erg/g].

    The ground state is the zero of chemical energy. The scalar ionization
    energy is measured from that ground state; bound excitation energies use
    the same reference. Thermal enthalpy includes electrons and heavy H.
    """
    radius = jnp.asarray(radius_cm)
    sound_squared = BOLTZMANN_CGS * profile.temperature_k * (1 + profile.ionized_fraction) / PROTON_MASS_CGS
    chemical = (profile.bound_fraction @ jnp.asarray(excitation_energy_erg)
                + profile.ionized_fraction * ionization_energy_erg)
    potential = -gravitational_parameter_cm3_s2 / radius - tidal_coefficient_s2 * radius**2 / 2
    return profile.velocity_cm_s**2 / 2 + 2.5 * sound_squared + chemical / PROTON_MASS_CGS + potential


def _cell_residual(
    left, right, radius_left, radius_right, rates, heating, excitation_energy,
    ionization_energy, gravity, tide, momentum_scale, energy_scale,
):
    sound_left = BOLTZMANN_CGS * left.temperature_k * (1 + left.ionized_fraction) / PROTON_MASS_CGS
    sound_right = BOLTZMANN_CGS * right.temperature_k * (1 + right.ionized_fraction) / PROTON_MASS_CGS
    width = radius_right - radius_left
    area_difference = width * (radius_right + radius_left)
    volume = width * (radius_right**2 + radius_right * radius_left + radius_left**2) / 3
    effective_width = volume / radius_right**2
    fourth_difference = area_difference * (radius_right**2 + radius_left**2)
    momentum_source = (
        sound_right * area_difference - gravity * width + tide * fourth_difference / 4
    ) / (radius_right**2 * right.velocity_cm_s)
    momentum = (
        right.velocity_cm_s + sound_right / right.velocity_cm_s
        - left.velocity_cm_s - sound_left / left.velocity_cm_s - momentum_source
    ) / momentum_scale
    energy_left = bernoulli_energy(left, radius_left, excitation_energy, ionization_energy, gravity, tide)
    energy_right = bernoulli_energy(right, radius_right, excitation_energy, ionization_energy, gravity, tide)
    energy = (energy_right - energy_left - effective_width * heating / (
        PROTON_MASS_CGS * right.n_h_cm3 * right.velocity_cm_s
    )) / energy_scale
    production, turnover = species_production_s(right, rates)
    travel_time = effective_width / right.velocity_cm_s
    scale = left.bound_fraction + travel_time[..., None] * turnover
    species = (
        right.bound_fraction - left.bound_fraction - travel_time[..., None] * production
    ) / jnp.where(scale > 0, scale, 1.0)
    return jnp.concatenate((momentum[..., None], energy[..., None], species), axis=-1)


def reactive_flow_residual(
    state, radius_cm, n_h_base_cm3, temperature_base_k, velocity_base_cm_s,
    bound_fraction_base, excitation_energy_erg, ionization_energy_erg,
    rates_at_nodes: RateCoefficients, heating_at_nodes,
    gravitational_parameter_cm3_s2, *, tidal_coefficient_s2=0.0,
):
    """Return a square residual for joint flow/radiation implicit solves.

    All units are cgs. State is (Nnode,Nbound+2); rate arrays have a leading
    Nnode axis. Source row zero is unused. Node zero fixes inflow nH,T,v,f;
    each later node is a right-endpoint cell approximation (first order).
    Energy and species sources use exact spherical shell volumes, enabling
    exact cancellation against a conservative radiation energy flux.

    ``heating_at_nodes`` is FULL absorbed minus emitted photon energy per
    volume and time, including thresholds and line energies. Do not supply
    excess-energy photoheating plus collisional cooling: excitation and
    ionization energy already enter gas enthalpy and species advection.

    Momentum integrates the spherical pressure term and the potential
    Phi=-GM/r-kappa*r**2/2 with right-cell pressure and mass density.
    Residual ordering is momentum, total energy, then species. Scaling does
    not change the physical roots; the first row imposes dimensionless logs.
    """
    radius = jnp.asarray(radius_cm)
    profile = decode_flow_state(state, radius, n_h_base_cm3, temperature_base_k, velocity_base_cm_s)
    base_sound = BOLTZMANN_CGS * temperature_base_k * (2 - jnp.sum(bound_fraction_base)) / PROTON_MASS_CGS
    cells = _cell_residual(
        jax.tree.map(lambda value: value[:-1], profile),
        jax.tree.map(lambda value: value[1:], profile), radius[:-1], radius[1:],
        jax.tree.map(lambda value: value[1:], rates_at_nodes),
        jnp.asarray(heating_at_nodes)[1:], jnp.asarray(excitation_energy_erg),
        ionization_energy_erg, gravitational_parameter_cm3_s2, tidal_coefficient_s2,
        velocity_base_cm_s + base_sound / velocity_base_cm_s, base_sound,
    )
    return jnp.concatenate(((state[0] - _base_state(bound_fraction_base))[None, :], cells), axis=0)


def flow_domain_valid(profile, rates_at_nodes, heating_at_nodes, *, sonic_margin=1e-3):
    """Check positive finite states/rates and one branch away from the sonic point.

    This does not establish uniqueness or convergence. A failed check must
    invalidate a solve; no temperature, density or population is clipped.
    The source callback must return nonfinite values outside its data domain.
    """
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in profile]))
    positive = (
        jnp.all(profile.n_h_cm3 > 0) & jnp.all(profile.temperature_k > 0)
        & jnp.all(profile.velocity_cm_s > 0) & jnp.all(profile.bound_fraction > 0)
        & jnp.all(profile.ionized_fraction > 0)
    )
    rate_valid = jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(value) & (value >= 0)) for value in rates_at_nodes
    ]))
    rate_valid &= jnp.all(jnp.diagonal(rates_at_nodes.radiative_bound_bound_s, axis1=-2, axis2=-1) == 0)
    rate_valid &= jnp.all(jnp.diagonal(rates_at_nodes.electron_bound_bound_cm3_s, axis1=-2, axis2=-1) == 0)
    sonic_difference = profile.mach_number**2 - 1
    branch = jnp.all(jnp.abs(sonic_difference) > sonic_margin)
    branch &= jnp.all(sonic_difference * sonic_difference[0] > 0)
    return finite & positive & rate_valid & jnp.all(jnp.isfinite(heating_at_nodes)) & branch


def _newton(function, initial, iterations):
    """Bounded Newton/backtracking; derivatives belong to the enclosing root.

    Stop evaluating the expensive nonlocal radiation residual once a root
    converges. Accept the first improving step, as in ordinary backtracking,
    rather than evaluating every shorter step after success.
    """
    def keep(state):
        index, _, residual, progress = state
        return ((index < iterations) & progress & jnp.all(jnp.isfinite(residual))
                & (jnp.max(jnp.abs(residual)) > 2e-12))

    def step(state):
        index, values, residual, _ = state
        norm = jnp.max(jnp.abs(residual))
        correction = jnp.linalg.solve(jax.jacfwd(function)(values), -residual)

        def search(state):
            trial, _, _, accepted = state
            return (trial < 12) & ~accepted

        def backtrack(state):
            trial, best, best_residual, _ = state
            candidate = values + (0.5**trial) * correction
            candidate_residual = function(candidate)
            score = jnp.max(jnp.abs(candidate_residual))
            accept = jnp.isfinite(score) & (score < norm)
            return (trial + 1, jnp.where(accept, candidate, best),
                    jnp.where(accept, candidate_residual, best_residual), accept)

        _, updated, updated_residual, accepted = jax.lax.while_loop(
            search, backtrack, (jnp.asarray(0), values, residual, jnp.asarray(False)),
        )
        return index + 1, updated, updated_residual, accepted

    return jax.lax.while_loop(
        keep, step, (jnp.asarray(0), initial, function(initial), jnp.asarray(True)),
    )[1]


@partial(jax.jit, static_argnames=("source_fn", "iterations"))
def solve_reactive_flow(
    radius_cm, n_h_base_cm3, temperature_base_k, velocity_base_cm_s,
    bound_fraction_base, excitation_energy_erg, ionization_energy_erg,
    source_fn, source_params, gravitational_parameter_cm3_s2,
    *, tidal_coefficient_s2=0.0, iterations=32,
):
    """Solve a finite spherical branch with prescribed inflow and local sources.

    ``source_fn(r,nH,T,bound_fraction,params)`` must be a differentiable JAX
    function returning ``(RateCoefficients, Qrad_erg_cm3_s)``. Use a joint
    ``reactive_flow_residual`` solve for nonlocal radiation feedback instead.
    Require float64, increasing positive radii, positive inflow variables,
    and strictly positive bound and proton fractions. No critical point may
    be crossed; this solver neither selects mass loss nor constructs a full
    transonic escape solution. Grid convergence must be checked separately.

    Each conservative backward-Euler cell is solved by damped Newton and
    differentiated implicitly with ``custom_root``. Derivatives therefore
    include rate, electron, pressure and latent-energy feedback without
    differentiating Newton iterations. Invalid/failed solves return NaN
    state/profile fields plus diagnostics; no unphysical clipping occurs.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before solving reactive flow.")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive static integer.")
    radius = jnp.asarray(radius_cm, dtype=jnp.float64)
    bound = jnp.asarray(bound_fraction_base, dtype=jnp.float64)
    excitation = jnp.asarray(excitation_energy_erg, dtype=jnp.float64)
    if radius.ndim != 1 or radius.size < 2:
        raise ValueError("Use at least two radial nodes.")
    if bound.ndim != 1 or bound.size < 1 or excitation.shape != bound.shape:
        raise ValueError("Use matching nonempty vectors for bound fractions and excitation energies.")
    base_sound = BOLTZMANN_CGS * temperature_base_k * (2 - jnp.sum(bound)) / PROTON_MASS_CGS
    initial = _base_state(bound)

    def advance(previous, radii):
        left_radius, right_radius = radii
        left = _decode(
            previous, left_radius, n_h_base_cm3, temperature_base_k, velocity_base_cm_s, radius[0],
        )

        def residual(candidate):
            right = _decode(
                candidate, right_radius, n_h_base_cm3, temperature_base_k, velocity_base_cm_s, radius[0],
            )
            rates, heating = source_fn(
                right_radius, right.n_h_cm3, right.temperature_k, right.bound_fraction, source_params,
            )
            return _cell_residual(
                left, right, left_radius, right_radius, rates, heating,
                excitation, ionization_energy_erg, gravitational_parameter_cm3_s2,
                tidal_coefficient_s2, velocity_base_cm_s + base_sound / velocity_base_cm_s, base_sound,
            )

        current = jax.lax.custom_root(
            residual, previous,
            lambda function, guess: _newton(function, guess, iterations),
            lambda linear, rhs: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs),
        )
        return current, current

    _, cells = jax.lax.scan(advance, initial, (radius[:-1], radius[1:]))
    state = jnp.concatenate((initial[None, :], cells), axis=0)
    profile = decode_flow_state(state, radius, n_h_base_cm3, temperature_base_k, velocity_base_cm_s)
    rates, heating = jax.vmap(source_fn, in_axes=(0, 0, 0, 0, None))(
        radius, profile.n_h_cm3, profile.temperature_k, profile.bound_fraction, source_params,
    )
    residual = reactive_flow_residual(
        state, radius, n_h_base_cm3, temperature_base_k, velocity_base_cm_s,
        bound, excitation, ionization_energy_erg, rates, heating,
        gravitational_parameter_cm3_s2, tidal_coefficient_s2=tidal_coefficient_s2,
    )
    residual_norm = jnp.max(jnp.abs(residual))
    valid = (
        jnp.all(jnp.isfinite(radius) & (radius > 0)) & jnp.all(jnp.diff(radius) > 0)
        & jnp.isfinite(n_h_base_cm3) & (n_h_base_cm3 > 0)
        & jnp.isfinite(temperature_base_k) & (temperature_base_k > 0)
        & jnp.isfinite(velocity_base_cm_s) & (velocity_base_cm_s > 0)
        & jnp.all(jnp.isfinite(bound) & (bound > 0)) & (jnp.sum(bound) < 1)
        & jnp.all(jnp.isfinite(excitation) & (excitation >= 0))
        & jnp.isfinite(ionization_energy_erg) & (ionization_energy_erg > 0)
        & jnp.all(excitation < ionization_energy_erg)
        & jnp.isfinite(gravitational_parameter_cm3_s2) & (gravitational_parameter_cm3_s2 >= 0)
        & jnp.isfinite(tidal_coefficient_s2) & (tidal_coefficient_s2 >= 0)
        & flow_domain_valid(profile, rates, heating)
    )
    converged = valid & (residual_norm < 1e-9)
    value = lambda array: jnp.where(converged, array, jnp.nan)
    mass_loss = 4 * jnp.pi * radius[0]**2 * PROTON_MASS_CGS * n_h_base_cm3 * velocity_base_cm_s
    energy = bernoulli_energy(
        profile, radius, excitation, ionization_energy_erg,
        gravitational_parameter_cm3_s2, tidal_coefficient_s2,
    )
    return FlowSolution(
        value(state), jax.tree.map(value, profile), value(mass_loss), value(energy),
        residual_norm, valid, converged,
    )


__all__ = [
    "FlowState", "FlowSolution", "decode_flow_state", "spherical_shell_volume",
    "species_production_s", "bernoulli_energy", "reactive_flow_residual",
    "flow_domain_valid", "solve_reactive_flow",
]
