"""Joint atom/radiation and finite-inflow hydrogen experiments.

These are closed reduced models with explicit boundaries and diagnostics,
not a Cloudy-equivalent atmosphere. Static Eddington/CRD transfer omits bulk
Doppler shifts. All derivatives follow the complete selected residual.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity.lpf.lpf import vvoigt

from .hydrogen_atom import (
    DEFAULT_PRINCIPAL_N, atom_natural_width_hz, atom_rate_coefficients,
    make_hydrogen_atom, solve_atom_populations,
)
from .hydrogen_collisions import validate_shell_collision_model
from .hydrogen_continuum import continuum_coefficients, planck_intensity_hz, thermal_energy_exchange
from .hydrogen_nlte import _bound_ratios
from .hydrogen_rates import (
    BOLTZMANN_CGS, LIGHT_SPEED_CGS, PLANCK_CGS, TWO_PHOTON_A_S, minimum_rate_temperature,
)
from .parker import PROTON_MASS_CGS
from .radiation import moment_geometry, solve_moment_transfer, solve_slab_transfer
from .reactive_flow import (
    _newton, bernoulli_energy, decode_flow_state, flow_domain_valid,
    reactive_flow_residual,
)


class FrequencyGrid(NamedTuple):
    frequency_hz: jax.Array
    weight_hz: jax.Array


class RadiationState(NamedTuple):
    rates: object
    heating_erg_cm3_s: jax.Array
    power_scale_erg_cm3_s: jax.Array
    two_photon_cooling: jax.Array
    mean_intensity: jax.Array
    flux_moment_edges: jax.Array
    profile_integrals: jax.Array
    transfer_residual: jax.Array
    radiation_balance_error: jax.Array
    gain_optical_depth: jax.Array
    converged: jax.Array
    thermal_heating_erg_cm3_s: jax.Array
    thermal_cooling_erg_cm3_s: jax.Array
    thermal_net_erg_cm3_s: jax.Array


class EquilibriumSolution(NamedTuple):
    temperature_k: jax.Array
    bound_cm3: jax.Array
    electron_cm3: jax.Array
    ionized_fraction: jax.Array
    radiation: RadiationState
    state: jax.Array
    residual_norm: jax.Array
    converged: jax.Array


class CoupledFlowSolution(NamedTuple):
    profile: object
    radiation: RadiationState
    state: jax.Array
    bernoulli_erg_g: jax.Array
    residual_norm: jax.Array
    energy_balance_error: jax.Array
    energy_exchange_error: jax.Array
    material_energy_error: jax.Array
    converged: jax.Array


def frequency_grid(atom, *, order=4, reference_temperature_k=10000.,
                   minimum_energy_ev=1e-4, maximum_energy_ev=100.,
                   line_wing_offsets=None):
    """One nonoverlapping quadrature for lines, edges and continua.

    Fixed log-frequency intervals split at ionization edges and line cores
    (0, +/-1,3,10 reference Gaussian sigma widths). ``line_wing_offsets``
    adds positive offsets beyond 10 sigma on both sides of every line; e.g.
    (30, 100, 300, 1000) resolves optically thick Lorentz wings. None retains
    the original grid. Every interval uses the same Gauss-Legendre order.
    Finite-window profiles are normalized on this grid;
    their pre-normalization integrals are returned as resolution diagnostics.
    Change order/window/atom explicitly and check convergence before inference.
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 2:
        raise ValueError("Quadrature order must be an integer >=2.")
    ev = 1.602176634e-12
    if not 0 < minimum_energy_ev < .754 < 13.6 < maximum_energy_ev:
        raise ValueError("The frequency window must cover H-minus and H thresholds.")
    limits = np.array([minimum_energy_ev, maximum_energy_ev]) * ev / PLANCK_CGS
    lines = np.asarray(atom.line_frequency_hz)
    widths = lines / LIGHT_SPEED_CGS * np.sqrt(BOLTZMANN_CGS * reference_temperature_k / PROTON_MASS_CGS)
    offsets = np.array([-10., -3., -1., 0., 1., 3., 10.])
    if line_wing_offsets is not None:
        extra = np.asarray(line_wing_offsets, dtype=float)
        if extra.ndim != 1 or not np.all(np.isfinite(extra)) or np.any(extra <= 10):
            raise ValueError("Line wing offsets must be finite positive sigma offsets greater than 10.")
        offsets = np.concatenate((offsets, -extra, extra))
    edges = np.concatenate((np.geomspace(*limits, 49), np.asarray(atom.ionization_energy_erg) / PLANCK_CGS,
                            [LIGHT_SPEED_CGS / 1.6419e-4],
                            (lines[:, None] + widths[:, None] * offsets).ravel()))
    edges = np.unique(edges[(edges >= limits[0]) & (edges <= limits[1])])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    logarithms = np.log(edges)
    width = np.diff(logarithms)
    frequency = np.exp(logarithms[:-1, None] + width[:, None] * (nodes + 1) / 2)
    weight = frequency * width[:, None] * weights / 2
    return FrequencyGrid(jnp.asarray(frequency.ravel()), jnp.asarray(weight.ravel()))


def atom_line_profiles(atom, temperature_k, grid):
    """Return CRD Voigt profiles [Hz^-1] and their unnormalized integrals.

    Natural width uses BOTH level lifetimes, including the spontaneous 2s
    channel. No pressure/Stark broadening or bulk velocity is added.
    """
    frequency = atom.line_frequency_hz
    gamma = atom_natural_width_hz(atom)
    sigma = frequency / LIGHT_SPEED_CGS * jnp.sqrt(BOLTZMANN_CGS * temperature_k / PROTON_MASS_CGS)
    raw = vvoigt(grid.frequency_hz[None, :] - frequency[:, None], sigma, gamma)
    integral = raw @ grid.weight_hz
    return raw / integral[:, None], integral


class CoupledHydrogenModel:
    """Fixed atomic/radial/frequency grids; independent root solves per call.

    Incoming boundary arguments are I_nu, not J_nu or flux. Use explicit
    spectra, e.g. planck_intensity_hz(grid.frequency_hz,T). Constant Eddington
    factor, complete redistribution and static transport are approximations.
    H-minus is an optional LTE trace closure, disabled by default.
    Equilibrium initialization requires nonzero incident ionizing radiation;
    a purely collisionally ionized, unilluminated branch is not initialized.
    """

    def __init__(self, edges_cm, *, nmax=DEFAULT_PRINCIPAL_N, resolved_nmax=3,
                 quadrature_order=4, spherical=True,
                 include_free_free=True, include_hminus=False,
                 minimum_energy_ev=1e-4, maximum_energy_ev=100.,
                 bound_free_model="hydrogenic", include_orbital_mixing=True,
                 transfer="moment", angle_order=4, low_temperature_extrapolation="reject",
                 stellar_beam=False, line_wing_offsets=None, shell_collision_model="anderson_n45"):
        edges = np.asarray(edges_cm, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValueError("Use finite strictly increasing radial edges.")
        if spherical and not np.all(edges > 0):
            raise ValueError("Spherical radii must be positive.")
        self.edges_cm = jnp.asarray(edges)
        self.atom = make_hydrogen_atom(nmax, resolved_nmax=resolved_nmax)
        self.grid = frequency_grid(self.atom, order=quadrature_order,
                                   minimum_energy_ev=minimum_energy_ev, maximum_energy_ev=maximum_energy_ev,
                                   line_wing_offsets=line_wing_offsets)
        self.line_wing_offsets = None if line_wing_offsets is None else tuple(line_wing_offsets)
        self.spherical = bool(spherical)
        if transfer not in ("moment", "formal") or (transfer == "formal" and spherical):
            raise ValueError("Use moment transfer, or formal transfer with spherical=False.")
        if not isinstance(angle_order, int) or isinstance(angle_order, bool) or angle_order < 1:
            raise ValueError("angle_order must be a positive integer.")
        self.transfer = transfer
        self.angle_order = angle_order
        if stellar_beam and transfer != "formal":
            raise ValueError("A directed stellar beam requires formal slab transfer.")
        self.stellar_beam = bool(stellar_beam)
        minimum_rate_temperature(low_temperature_extrapolation)
        self.low_temperature_extrapolation = low_temperature_extrapolation
        validate_shell_collision_model(shell_collision_model, resolved_nmax=self.atom.resolved_nmax)
        self.shell_collision_model = shell_collision_model
        self.include_free_free = bool(include_free_free)
        self.include_hminus = bool(include_hminus)
        if bound_free_model not in ("hydrogenic", "kramers"):
            raise ValueError("Bound-free model must be 'hydrogenic' or 'kramers'.")
        self.bound_free_model = bound_free_model
        self.include_orbital_mixing = bool(include_orbital_mixing)

    def boundary_blackbody(self, temperature_k, intensity_scale=1.):
        return intensity_scale * planck_intensity_hz(self.grid.frequency_hz, temperature_k)

    def _transfer(self, extinction, emission, inner, outer):
        if self.transfer == "formal":
            # Keep the same transparent-limit J=outer/2 for both illumination
            # options, while a normal beam carries twice the hemisphere flux.
            beam = {"beam_mean_intensity": outer / 2} if self.stellar_beam else {}
            incident = jnp.zeros_like(outer) if self.stellar_beam else outer
            return solve_slab_transfer(self.edges_cm, extinction, emission, inner, incident,
                                       angle_order=self.angle_order, **beam)
        return solve_moment_transfer(self.edges_cm, extinction, emission, inner, outer,
                                     spherical=self.spherical)

    def _occupations(self, mean, profiles):
        frequency = self.grid.frequency_hz
        continuum = LIGHT_SPEED_CGS**2 * mean / (2 * PLANCK_CGS * frequency**3)
        averaged = jnp.einsum("clf,cf,f->cl", profiles, mean, self.grid.weight_hz)
        lines = LIGHT_SPEED_CGS**2 * averaged / (2 * PLANCK_CGS * self.atom.line_frequency_hz**3)
        return lines, continuum

    def radiation_state(self, density, temperature, fractions, inner, outer, coupling=1.,
                        ionized_fraction=None):
        """Use identical line/continuum kernels for rates and photon energy."""
        atom, grid = self.atom, self.grid
        bound = density[:, None] * fractions
        ionized = 1 - jnp.sum(fractions, axis=1) if ionized_fraction is None else ionized_fraction
        electron = density * ionized
        phi, coverage = jax.vmap(lambda t: atom_line_profiles(atom, t, grid))(temperature)
        chi_cont, j_cont = jax.vmap(lambda t, n, ne: continuum_coefficients(
            atom, t, n, ne, ne, grid.frequency_hz,
            include_free_free=self.include_free_free, include_hminus=self.include_hminus,
            bound_free_model=self.bound_free_model,
        ))(temperature, bound, electron)
        u, l = atom.line_upper, atom.line_lower
        a, g, nu = atom.einstein_a_s, atom.statistical_weights, atom.line_frequency_hz
        extinction_weight = LIGHT_SPEED_CGS**2 * a / (8 * jnp.pi * nu**2) * (
            g[u] / g[l] * bound[:, l] - bound[:, u]
        )
        emission_weight = PLANCK_CGS * nu * a / (4 * jnp.pi) * bound[:, u]
        chi = chi_cont + jnp.einsum("cl,clf->cf", extinction_weight, phi)
        emission = j_cont + jnp.einsum("cl,clf->cf", emission_weight, phi)
        transfer = self._transfer(coupling * chi, coupling * emission, inner, outer)
        eta_line, eta_cont = self._occupations(transfer.mean_intensity, phi)
        rates = jax.vmap(lambda t, ne, line, cont: atom_rate_coefficients(
            atom, t, line, cont, grid.frequency_hz, grid.weight_hz,
            electron_cm3=ne if self.include_orbital_mixing else None,
            bound_free_model=self.bound_free_model,
            low_temperature_extrapolation=self.low_temperature_extrapolation,
            shell_collision_model=self.shell_collision_model,
        ))(temperature, electron, eta_line, eta_cont)
        two_photon = bound[:, 1] * TWO_PHOTON_A_S * atom.excitation_energy_erg[1]
        absorbed = 4 * jnp.pi * jnp.abs(chi * transfer.mean_intensity) @ grid.weight_hz
        emitted = 4 * jnp.pi * emission @ grid.weight_hz
        heating = 4 * jnp.pi * ((chi * transfer.mean_intensity - emission) @ grid.weight_hz) - two_photon
        thermal = jax.vmap(lambda t, n, ne, rate, eta: thermal_energy_exchange(
            atom, t, n, ne, ne, rate, eta, grid.frequency_hz, grid.weight_hz,
            include_free_free=self.include_free_free, include_hminus=self.include_hminus,
            bound_free_model=self.bound_free_model,
        ))(temperature, bound, electron, rates, eta_cont)
        return RadiationState(rates, heating, absorbed + emitted + two_photon,
                              two_photon, transfer.mean_intensity, transfer.flux_moment_edges,
                              coverage, transfer.residual_norm, transfer.balance_error,
                              transfer.gain_optical_depth, transfer.converged,
                              thermal.heating_erg_cm3_s, thermal.cooling_erg_cm3_s,
                              thermal.net_erg_cm3_s)

    def _initial_fractions(self, density, temperature, inner, outer):
        count = len(self.edges_cm) - 1
        zeros = jnp.zeros((count, len(self.grid.frequency_hz)))
        mean = self._transfer(zeros, zeros, inner, outer).mean_intensity
        phi, _ = jax.vmap(lambda t: atom_line_profiles(self.atom, t, self.grid))(temperature)
        line, continuum = self._occupations(mean, phi)
        solution = jax.vmap(lambda n, t, ll, cc: solve_atom_populations(
            n, self.atom, t, ll, cc, self.grid.frequency_hz, self.grid.weight_hz,
            bound_free_model=self.bound_free_model,
            include_orbital_mixing=self.include_orbital_mixing,
            low_temperature_extrapolation=self.low_temperature_extrapolation,
            shell_collision_model=self.shell_collision_model,
        ))(density, temperature, line, continuum)
        return solution.bound_cm3 / density[:, None]

    @partial(jax.jit, static_argnums=0, static_argnames=(
        "thermal", "iterations", "continuation_steps", "continuation_start",
    ))
    def solve_equilibrium(self, n_h_cm3, temperature_k, inner_intensity, outer_intensity,
                          *, thermal=False, iterations=24, continuation_steps=4,
                          initial_state=None, continuation_start=None):
        """Solve populations/mean field and optionally local radiative equilibrium.

        Density remains prescribed. thermal=False fixes T; thermal=True solves
        net kinetic heating = 0, equal to full photon heating at converged SE.
        The supplied T is then only an initial guess. Invalid roots return
        NaN populations/temperatures and converged=False. No range clipping.
        initial_state optionally supplies log population/proton ratios (and
        log(T/T_initial) as the first column for thermal=True). An optional
        static continuation_start in (0,1] uses logarithmic opacity continuation;
        None preserves the original linear schedule. One step solves full opacity.
        """
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (iterations, continuation_steps)):
            raise ValueError("Iterations and continuation steps must be positive static integers.")
        if continuation_start is not None and not 0 < continuation_start <= 1:
            raise ValueError("continuation_start must lie in (0,1].")
        count = len(self.edges_cm) - 1
        density = jnp.broadcast_to(jnp.asarray(n_h_cm3), (count,))
        temperature = jnp.broadcast_to(jnp.asarray(temperature_k), (count,))
        if initial_state is None:
            initial_f = self._initial_fractions(density, temperature, inner_intensity, outer_intensity)
            initial = jnp.log(initial_f / (1 - jnp.sum(initial_f, axis=1))[:, None])
            if thermal:
                initial = jnp.concatenate((jnp.zeros((count, 1)), initial), axis=1)
        else:
            initial = jnp.asarray(initial_state)
            if initial.shape != (count, len(self.atom.statistical_weights) + int(thermal)):
                raise ValueError("initial_state must match the equilibrium log-coordinate grid.")
        shape = initial.shape

        def decode(flat):
            values = flat.reshape(shape)
            temp = temperature * jnp.exp(values[:, 0]) if thermal else temperature
            ratios = values[:, 1:] if thermal else values
            all_f = jax.nn.softmax(jnp.concatenate((ratios, jnp.zeros((count, 1))), axis=1), axis=1)
            return temp, all_f[:, :-1], ratios, all_f[:, -1]

        def residual(flat, blend=1.):
            temp, fractions, log_ratios, ionized = decode(flat)
            radiation = self.radiation_state(density, temp, fractions, inner_intensity, outer_intensity,
                                             blend, ionized_fraction=ionized)
            ne = density * ionized
            target = jax.vmap(_bound_ratios)(ne, radiation.rates)
            balance = log_ratios - jnp.log(target)
            if thermal:
                heat = radiation.thermal_net_erg_cm3_s / (
                    radiation.thermal_heating_erg_cm3_s + radiation.thermal_cooling_erg_cm3_s
                )
                balance = jnp.concatenate((heat[:, None], balance), axis=1)
            return balance.ravel()

        def solve(function, guess):
            del function
            def step(index, state):
                fraction = (index + 1) / continuation_steps
                blend = (fraction if continuation_start is None
                         else continuation_start ** (1 - fraction))
                return _newton(lambda x: residual(x, blend), state, iterations)
            return jax.lax.fori_loop(0, continuation_steps, step, guess)

        root = jax.lax.custom_root(residual, initial.ravel(), solve,
                                  lambda linear, rhs: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs))
        temp, fractions, _, ionized = decode(root)
        radiation = self.radiation_state(density, temp, fractions, inner_intensity, outer_intensity,
                                         ionized_fraction=ionized)
        norm = jnp.max(jnp.abs(residual(root)))
        converged = radiation.converged & jnp.isfinite(norm) & (norm < 1e-8) & jnp.all(density > 0)
        bound = density[:, None] * fractions
        electron = density * ionized
        mask = lambda x: jnp.where(converged, x, jnp.nan)
        return EquilibriumSolution(mask(temp), mask(bound), mask(electron), mask(electron / density),
                                   radiation, root.reshape(shape), norm, converged)

    @partial(jax.jit, static_argnums=0, static_argnames=("iterations", "continuation_steps"))
    def solve_flow(self, n_h_base_cm3, temperature_base_k, velocity_base_cm_s,
                   bound_fraction_base, inner_intensity, outer_intensity,
                   gravitational_parameter_cm3_s2, *, tidal_coefficient_s2=0.,
                   iterations=32, continuation_steps=4):
        """Joint radiation, advected populations, temperature and velocity.

        Only a specified-inflow branch away from sonic crossing is supported.
        Radius edges are flow nodes, with right-endpoint material cells. Mass
        flux is fixed by the inflow; no transonic eigenvalue is selected.
        """
        if not self.spherical:
            raise ValueError("Reactive flow uses spherical radiation geometry.")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (iterations, continuation_steps)):
            raise ValueError("Iterations and continuation steps must be positive static integers.")
        f0 = jnp.asarray(bound_fraction_base)
        first = jnp.concatenate((jnp.zeros(2), jnp.log(f0 / (1 - jnp.sum(f0)))))
        initial = jnp.broadcast_to(first, (len(self.edges_cm), len(first)))
        shape = initial.shape

        def evaluate(flat, blend=1.):
            state = flat.reshape(shape)
            profile = decode_flow_state(state, self.edges_cm, n_h_base_cm3, temperature_base_k, velocity_base_cm_s)
            radiation = self.radiation_state(profile.n_h_cm3[1:], profile.temperature_k[1:],
                                             profile.bound_fraction[1:], inner_intensity, outer_intensity, blend,
                                             ionized_fraction=profile.ionized_fraction[1:])
            rates = jax.tree.map(lambda x: jnp.concatenate((x[:1], x), axis=0), radiation.rates)
            heat = jnp.concatenate((radiation.heating_erg_cm3_s[:1], radiation.heating_erg_cm3_s))
            result = reactive_flow_residual(
                state, self.edges_cm, n_h_base_cm3, temperature_base_k, velocity_base_cm_s,
                f0, self.atom.excitation_energy_erg, self.atom.ionization_energy_erg[0],
                rates, heat, gravitational_parameter_cm3_s2,
                tidal_coefficient_s2=tidal_coefficient_s2,
            )
            return result.ravel(), profile, radiation, rates, heat

        residual = lambda x: evaluate(x)[0]
        def solve(function, guess):
            del function
            def step(index, state):
                blend = (index + 1) / continuation_steps
                return _newton(lambda x: evaluate(x, blend)[0], state, iterations)
            return jax.lax.fori_loop(0, continuation_steps, step, guess)

        root = jax.lax.custom_root(residual, initial.ravel(), solve,
                                  lambda linear, rhs: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs))
        result, profile, radiation, rates, heat = evaluate(root)
        norm = jnp.max(jnp.abs(result))
        bernoulli = bernoulli_energy(profile, self.edges_cm, self.atom.excitation_energy_erg,
                                     self.atom.ionization_energy_erg[0], gravitational_parameter_cm3_s2,
                                     tidal_coefficient_s2)
        area, volume = moment_geometry(self.edges_cm)
        mass_per_sr = PROTON_MASS_CGS * n_h_base_cm3 * velocity_base_cm_s * self.edges_cm[0]**2
        gas_change = mass_per_sr * (bernoulli[-1] - bernoulli[0])
        spectral_inner = area[0] * radiation.flux_moment_edges[0]
        spectral_outer = area[-1] * radiation.flux_moment_edges[-1]
        radiation_change = 4 * jnp.pi * ((spectral_outer - spectral_inner) @ self.grid.weight_hz)
        escaped = volume @ radiation.two_photon_cooling
        budget = gas_change + radiation_change + escaped
        exchange_scale = jnp.abs(gas_change) + jnp.abs(radiation_change) + jnp.abs(escaped)
        exchange_error = jnp.abs(budget) / jnp.where(exchange_scale > 0, exchange_scale, 1.)
        # Net exchange can be far below the transparent photon throughput;
        # cancellation there must not turn nearby valid roots into NaNs.
        # Report both measures, and gate conservation against the carried flux.
        throughput = 4 * jnp.pi * ((jnp.abs(spectral_inner) + jnp.abs(spectral_outer)) @ self.grid.weight_hz)
        scale = exchange_scale + throughput
        budget_error = jnp.abs(budget) / jnp.where(scale > 0, scale, 1.)
        integrated_heating = volume @ radiation.heating_erg_cm3_s
        material_defect = jnp.abs(gas_change - integrated_heating)
        material_scale = jnp.abs(gas_change) + jnp.abs(integrated_heating)
        material_error = material_defect / jnp.where(material_scale > 0, material_scale, 1.)
        material_throughput = mass_per_sr * (jnp.abs(bernoulli[0]) + jnp.abs(bernoulli[-1]))
        roundoff = 128 * jnp.finfo(jnp.float64).eps
        budget_valid = jnp.abs(budget) <= 1e-8 * exchange_scale + roundoff * (throughput + material_throughput)
        material_valid = material_defect <= 1e-8 * material_scale + roundoff * material_throughput
        parameters_valid = (jnp.isfinite(gravitational_parameter_cm3_s2)
                            & (gravitational_parameter_cm3_s2 >= 0)
                            & jnp.isfinite(tidal_coefficient_s2) & (tidal_coefficient_s2 >= 0))
        converged = (parameters_valid & radiation.converged & flow_domain_valid(profile, rates, heat)
                     & jnp.isfinite(norm) & (norm < 1e-8) & budget_valid & material_valid)
        valid_profile = jax.tree.map(lambda x: jnp.where(converged, x, jnp.nan), profile)
        return CoupledFlowSolution(valid_profile, radiation, root.reshape(shape), bernoulli,
                                   norm, budget_error, exchange_error, material_error, converged)


__all__ = ["CoupledHydrogenModel", "frequency_grid", "atom_line_profiles"]
