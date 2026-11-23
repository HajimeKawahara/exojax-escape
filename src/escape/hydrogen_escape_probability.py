"""Hydrogen populations with an explicit Cloudy-style line-escape closure.

This alternative to the full-J CRD model omits the trapped local line field
from pumping. Continuum transport and line-destruction feedback are solved
with the populations; no Cloudy installation or saved Cloudy state is needed.
The finite atom, continuum transport and spatial convergence remain explicit
model choices, not an assertion of equivalence to a solar-abundance atmosphere.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from exojax.opacity.lpf.lpf import vvoigt

from .coupled_hydrogen import FrequencyGrid
from .hydrogen_atom import (
    atom_collision_rates,
    atom_cross_sections,
    atom_milne_population,
    make_hydrogen_atom,
)
from .hydrogen_collisions import validate_shell_collision_model
from .hydrogen_continuum import hydrogen_free_free_coefficients, planck_intensity_hz
from .hydrogen_mixing import mixing_bound_bound_cm3_s
from .hydrogen_nlte import RateCoefficients, _bound_ratios
from .hydrogen_rates import (
    BOLTZMANN_CGS,
    LIGHT_SPEED_CGS,
    PLANCK_CGS,
    TWO_PHOTON_A_S,
    minimum_rate_temperature,
)
from .hydrogen_two_photon import two_photon_emissivity
from .line_escape_probability import (
    CORE,
    LYA,
    WING,
    hydrogen_rayleigh_cross_section,
    hydrogen_stark_escape,
    line_operators,
)
from .parker import PROTON_MASS_CGS
from .radiation import _cell_weights, solve_slab_transfer
from .reactive_flow import _newton

_RYDBERG_HZ = LIGHT_SPEED_CGS * 109737.3156816


def _column_above(cell_depth):
    """Exclusive column from the illuminated (last-array) face."""
    return jnp.concatenate(
        (
            jnp.cumsum(cell_depth[::-1], axis=0)[:-1][::-1],
            jnp.zeros_like(cell_depth[:1]),
        ),
        axis=0,
    )


class HydrogenEscapeProbabilityModel:
    """Fixed slab grid with variable plasma state and implicit population solve.

    Arrays run from the dark/inner face to the illuminated/outer face. T, ne,
    and np have shape [cell], bound populations [cell,level], continuum
    coefficients [cell,frequency]. Incident boundary is a normal-beam J_nu;
    inner boundary is isotropic incoming I_nu. All quantities use cgs units.

    C23-style per-level suppression of bound-free continuum gain is part of
    this closure. The attenuation approximation for continuum scattering is
    explicit and never counted as thermal absorption. No LTE H-minus term is
    added: a molecular network must supply its own opacity and emissivity.
    """

    def __init__(
        self,
        edges_cm,
        *,
        nmax=15,
        quadrature_order=4,
        angle_order=4,
        extra_frequency_edges=(),
        minimum_energy_ev=1e-4,
        maximum_energy_ev=100.0,
        low_temperature_extrapolation="cloudy",
        shell_collision_model="anderson_n45",
        scattering_attenuation=True,
        include_free_free=True,
        line_optical_depth_scale=1.0,
    ):
        edges = np.asarray(edges_cm, dtype=float)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError("Use finite, strictly increasing slab edges.")
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in (quadrature_order, angle_order)
        ):
            raise ValueError("Quadrature and angle orders must be positive integers.")
        if not 0 < minimum_energy_ev < 0.754 < 13.6 < maximum_energy_ev:
            raise ValueError(
                "Frequency range must include hydrogen and H-minus thresholds."
            )
        extra = np.asarray(extra_frequency_edges, dtype=float)
        if extra.ndim != 1 or not np.isfinite(extra).all() or np.any(extra <= 0):
            raise ValueError("Extra frequency edges must be a positive finite vector.")
        minimum_rate_temperature(low_temperature_extrapolation)
        validate_shell_collision_model(shell_collision_model)
        self.low_temperature_extrapolation = low_temperature_extrapolation
        self.shell_collision_model = shell_collision_model
        self.scattering_attenuation = bool(scattering_attenuation)
        self.include_free_free = bool(include_free_free)
        if line_optical_depth_scale not in (1.0, 2.0):
            raise ValueError(
                "Use line_optical_depth_scale=1 for a finite slab or 2 for the explicit doubled-column approximation."
            )
        self.line_optical_depth_scale = float(line_optical_depth_scale)
        self.edges_cm = jnp.asarray(edges)
        self.widths_cm = jnp.diff(self.edges_cm)
        self.cells = len(edges) - 1
        self.angle_order = angle_order
        self.atom = atom = make_hydrogen_atom(nmax)
        limits = (
            np.array([minimum_energy_ev, maximum_energy_ev])
            * 1.602176634e-12
            / PLANCK_CGS
        )
        splits = np.unique(
            np.concatenate(
                (
                    np.geomspace(*limits, 49),
                    np.asarray(atom.ionization_energy_erg) / PLANCK_CGS,
                    np.asarray(atom.line_frequency_hz),
                    extra,
                    np.array([0.05, 0.646, 0.74859, 0.74959, 0.75059, 1.0])
                    * _RYDBERG_HZ,
                    [LIGHT_SPEED_CGS / 1.6419e-4],
                    (
                        np.arange(1, 16)
                        / 16
                        * float(atom.excitation_energy_erg[1])
                        / PLANCK_CGS
                    ),
                )
            )
        )
        splits = splits[(splits >= limits[0]) & (splits <= limits[1])]
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        delta = np.diff(np.log(splits))[:, None]
        frequency = splits[:-1, None] * np.exp(delta * (nodes + 1) / 2)
        self.grid = FrequencyGrid(
            jnp.asarray(frequency.ravel()),
            jnp.asarray((frequency * delta * weights / 2).ravel()),
        )
        self._sigma = jax.jit(lambda nu: atom_cross_sections(atom, nu))(
            self.grid.frequency_hz
        )
        self._sigma_line = jax.jit(lambda nu: atom_cross_sections(atom, nu))(
            atom.line_frequency_hz
        )
        self._photo_kernel = (
            8
            * jnp.pi
            / LIGHT_SPEED_CGS**2
            * self.grid.frequency_hz**2
            * self._sigma
            * self.grid.weight_hz
        )
        upper, lower = atom.line_upper, atom.line_lower
        self._kind = jnp.where(lower == 0, CORE, WING).at[0].set(LYA)
        decay = (
            jnp.zeros_like(atom.statistical_weights).at[upper].add(atom.einstein_a_s)
        )
        decay = decay.at[1].add(TWO_PHOTON_A_S)
        # C23 internal closure uses the upper lifetime only; observer unchanged.
        self._gamma_hz = decay[upper] / (4 * jnp.pi)
        n = np.asarray(atom.principal_n)
        self._multiplets = jnp.asarray(
            (n[np.asarray(upper), None] == n[np.asarray(upper)][None, :])
            & (n[np.asarray(lower), None] == n[np.asarray(lower)][None, :])
        )

    def boundary_blackbody(self, temperature_k, dilution=1.0):
        """Return an incident beam mean intensity J_nu, not a hemisphere I_nu."""
        return dilution * planck_intensity_hz(self.grid.frequency_hz, temperature_k)

    def _continuum(self, temperature, bound, proton, electron, frequency, sigma):
        inverse = jax.vmap(lambda t: atom_milne_population(self.atom, t, frequency))(
            temperature
        )
        inverse_population = (electron * proton)[:, None, None] * inverse
        net = sigma[None, :, :] * (bound[:, :, None] - inverse_population)
        # Altered C23 opacity_add1subshell.cpp: suppress gain per level.
        absorption = jnp.maximum(net, 0.0).sum(axis=1)
        emission = (
            2
            * PLANCK_CGS
            / LIGHT_SPEED_CGS**2
            * frequency**3
            * jnp.sum(sigma[None, :, :] * inverse_population, axis=1)
        )
        if self.include_free_free:
            ff, ff_emission = jax.vmap(
                lambda t, ne, np_: hydrogen_free_free_coefficients(
                    t, ne, np_, frequency
                )
            )(temperature, electron, proton)
            absorption, emission = absorption + ff, emission + ff_emission
        return absorption, emission, inverse

    def continuum_coefficients_at(
        self, temperature_k, bound_cm3, proton_cm3, electron_cm3, frequency_hz
    ):
        """Return H absorption [cm^-1] and emissivity [erg/s/cm^3/sr/Hz].

        This evaluates cross sections at the requested frequencies, preserving
        edges. Additional species and continuum scattering are not included.
        """
        frequency = jnp.asarray(frequency_hz)
        sigma = atom_cross_sections(self.atom, frequency)
        return self._continuum(
            temperature_k, bound_cm3, proton_cm3, electron_cm3, frequency, sigma
        )[:2]

    def photoionization_from_photon_flux(
        self,
        temperature_k,
        bound_cm3,
        proton_cm3,
        electron_cm3,
        frequency_hz,
        photon_flux_cm2_s,
    ):
        """H rates and kinetic energy from externally supplied line/OTS photons.

        Flux is integrated over each event frequency, not per Hz. This helper
        neither produces nor transports photons; the supplying network owns
        that accounting. It includes induced recombination through the same
        Milne inverse, and optional H free-free absorption. Sum its ledger
        with the returned H continuum/line ledger exactly once.
        """
        frequency = jnp.asarray(frequency_hz)
        flux, bound = jnp.asarray(photon_flux_cm2_s), jnp.asarray(bound_cm3)
        temperature = jnp.broadcast_to(temperature_k, (self.cells,))
        sigma = atom_cross_sections(self.atom, frequency)
        inverse = jax.vmap(lambda t: atom_milne_population(self.atom, t, frequency))(
            temperature
        )
        photo = flux @ sigma.T
        recombination = jnp.einsum("cf,lf,clf->cl", flux, sigma, inverse)
        excess = PLANCK_CGS * frequency - self.atom.ionization_energy_erg[:, None]
        positive = bound[:, :, None] * flux[:, None, :] * sigma * excess
        negative = (
            -(electron_cm3 * proton_cm3)[:, None, None]
            * inverse
            * flux[:, None, :]
            * sigma
            * excess
        )
        heating = jnp.maximum(positive, 0.0).sum(axis=(1, 2)) + jnp.maximum(
            negative, 0.0
        ).sum(axis=(1, 2))
        cooling = -jnp.minimum(positive, 0.0).sum(axis=(1, 2)) - jnp.minimum(
            negative, 0.0
        ).sum(axis=(1, 2))
        if self.include_free_free:
            chi_ff, _ = jax.vmap(
                lambda t, ne, np_: hydrogen_free_free_coefficients(
                    t, ne, np_, frequency
                )
            )(temperature, electron_cm3, proton_cm3)
            heating = heating + (chi_ff * flux * PLANCK_CGS * frequency).sum(axis=1)
        return dict(
            photoionization_s=photo,
            recombination_cm3_s=recombination,
            heating_erg_cm3_s=heating,
            cooling_erg_cm3_s=cooling,
            net_erg_cm3_s=heating - cooling,
        )

    def scattering_coefficients_at(self, bound_cm3, electron_cm3, frequency_hz):
        """Return Thomson plus neutral-H Rayleigh opacity [cm^-1]."""
        rayleigh = hydrogen_rayleigh_cross_section(
            jnp.asarray(frequency_hz) / _RYDBERG_HZ
        )
        return (
            electron_cm3[:, None] * 6.6524587321e-25 + bound_cm3[:, 0, None] * rayleigh
        )

    def _forward_diffuse(self, extinction, emission):
        """Cell-average J from emission moving away from the illuminating star."""
        x, w = np.polynomial.legendre.leggauss(self.angle_order)

        def direction(mu):
            tau = extinction * self.widths_cm[:, None] / mu
            emitted = emission * self.widths_cm[:, None] / mu
            first, second = _cell_weights(tau)

            def cell(incoming, values):
                t, e, f, s = values
                return incoming * jnp.exp(-t) + e * f, incoming * f + e * s

            return jax.lax.scan(
                cell,
                jnp.zeros_like(self.grid.frequency_hz),
                (tau, emitted, first, second),
                reverse=True,
            )[1]

        means = jax.vmap(direction)(jnp.asarray((x + 1) / 2))
        return jnp.einsum("a,acf->cf", jnp.asarray(w / 4), means)

    def population_state(
        self,
        temperature_k,
        bound_cm3,
        proton_cm3,
        electron_cm3,
        incident_mean_intensity,
        *,
        additional_opacity=0.0,
        additional_emission=0.0,
        additional_line_opacity=None,
        inner_intensity=0.0,
        incident_line_mean_intensity=None,
    ):
        """Return SE rates, common continuum fields and local line OTS events.

        Additional line opacity is evaluated at atom.line_frequency_hz; supply
        it explicitly when another absorber has edges near a hydrogen line.
        Supply incident_line_mean_intensity directly for a stellar spectrum
        with narrow absorption features; continuum-grid interpolation would
        otherwise smear those features. OTS flux [photons/cm^2/s] is integrated
        over each line, with no implicit
        bin width. A coupled composition model must distribute it to *all*
        absorbers using their same cross sections. This method adds the H part.
        """
        atom, grid = self.atom, self.grid
        t = jnp.broadcast_to(temperature_k, (self.cells,))
        bound = jnp.asarray(bound_cm3)
        ne, proton = jnp.asarray(electron_cm3), jnp.asarray(proton_cm3)
        chi, emission, inverse = self._continuum(
            t, bound, proton, ne, grid.frequency_hz, self._sigma
        )
        chi = chi + additional_opacity
        two_photon_emission = two_photon_emissivity(
            grid.frequency_hz,
            grid.weight_hz,
            atom.excitation_energy_erg[1] / PLANCK_CGS,
            bound[:, 1],
        )
        emission = emission + additional_emission + two_photon_emission
        scattering = self.scattering_coefficients_at(bound, ne, grid.frequency_hz)
        transport = chi + (
            jnp.log1p(scattering * self.widths_cm[:, None]) / self.widths_cm[:, None]
            if self.scattering_attenuation
            else 0.0
        )
        zero = jnp.zeros_like(grid.frequency_hz)
        transfer = solve_slab_transfer(
            self.edges_cm,
            transport,
            emission,
            zero + inner_intensity,
            zero,
            angle_order=self.angle_order,
            beam_mean_intensity=incident_mean_intensity,
        )
        prefactor = 2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * grid.frequency_hz**3
        occupation = transfer.mean_intensity / prefactor
        depth = transport * self.widths_cm[:, None]
        incident = (
            incident_mean_intensity
            * jnp.exp(-_column_above(depth))
            * _cell_weights(depth)[0]
        )
        forward_diffuse = self._forward_diffuse(transport, emission)
        pump_mean = incident + forward_diffuse
        line_chi, _, inverse_line = self._continuum(
            t, bound, proton, ne, atom.line_frequency_hz, self._sigma_line
        )
        if additional_line_opacity is None:
            extra = jnp.broadcast_to(additional_opacity, chi.shape)
            additional_line_opacity = jax.vmap(
                lambda row: jnp.interp(atom.line_frequency_hz, grid.frequency_hz, row)
            )(extra)
        line_chi = line_chi + additional_line_opacity
        upper, lower, g = atom.line_upper, atom.line_lower, atom.statistical_weights
        width = (
            atom.line_frequency_hz[None, :]
            / LIGHT_SPEED_CGS
            * jnp.sqrt(2 * BOLTZMANN_CGS * t[:, None] / PROTON_MASS_CGS)
        )
        damping = self._gamma_hz / width
        factor = (
            LIGHT_SPEED_CGS**2
            * atom.einstein_a_s
            / (8 * jnp.pi * atom.line_frequency_hz**2)
            / (jnp.sqrt(jnp.pi) * width)
        )
        kappa = factor * (bound[:, lower] * g[upper] / g[lower] - bound[:, upper])
        lower_opacity = factor * bound[:, lower] * g[upper] / g[lower]
        fine_matrix = jax.vmap(
            lambda w: vvoigt(
                atom.line_frequency_hz[None, :] - atom.line_frequency_hz[:, None],
                w / jnp.sqrt(2),
                self._gamma_hz,
            )
            * jnp.sqrt(jnp.pi)
            * w[:, None]
        )(width)
        fine = jnp.einsum("cl,clk->ck", kappa, fine_matrix)
        multiplet = kappa @ self._multiplets
        line_depth = fine * self.widths_cm[:, None]
        tau_in, tau_total = (
            _column_above(line_depth),
            self.line_optical_depth_scale * line_depth.sum(axis=0),
        )
        line_scattering = self.scattering_coefficients_at(
            bound, ne, atom.line_frequency_hz
        )
        albedo = line_scattering / (line_scattering + line_chi)
        line_transport = line_chi + (
            jnp.log1p(line_scattering * self.widths_cm[:, None])
            / self.widths_cm[:, None]
            if self.scattering_attenuation
            else 0.0
        )
        line_depth_continuum = line_transport * self.widths_cm[:, None]
        if incident_line_mean_intensity is None:
            incident_line_mean_intensity = jnp.interp(
                atom.line_frequency_hz, grid.frequency_hz, incident_mean_intensity
            )
        line_incident = (
            incident_line_mean_intensity
            * jnp.exp(-_column_above(line_depth_continuum))
            * _cell_weights(line_depth_continuum)[0]
        )
        line_diffuse = jax.vmap(
            lambda row: jnp.interp(atom.line_frequency_hz, grid.frequency_hz, row)
        )(forward_diffuse)
        line_pump_mean = line_incident + line_diffuse
        pump = line_pump_mean / (
            2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * atom.line_frequency_hz**3
        )
        operators = line_operators(
            self._kind,
            tau_in=tau_in,
            tau_total=tau_total,
            tau_pump=tau_in,
            damping=damping,
            line_extinction_cm1=kappa,
            lower_absorption_cm1=lower_opacity,
            multiplet_extinction_cm1=multiplet,
            continuum_absorption_cm1=line_chi,
            electron_cm3=ne[:, None],
            einstein_a_s=atom.einstein_a_s,
            lower_weight=g[lower],
            upper_weight=g[upper],
            incident_occupation=pump,
            outward_continuum_occupation=0.0,
            zone_width_cm=self.widths_cm[:, None],
            upper_population_cm3=bound[:, upper],
            albedo=albedo,
            stark_escape_probability=hydrogen_stark_escape(
                atom.principal_n[upper],
                atom.principal_n[lower],
                ne[:, None],
                t[:, None],
                tau_in,
            ),
        )
        ots_flux = operators.destroyed_photons_cm3_s / jnp.where(
            line_chi > 0, line_chi, 1.0
        )
        photo_ots = ots_flux @ self._sigma_line.T
        recombination_ots = jnp.einsum(
            "cf,lf,clf->cl", ots_flux, self._sigma_line, inverse_line
        )
        photo = occupation @ self._photo_kernel.T + photo_ots
        alpha = (
            jnp.einsum("cf,lf,clf->cl", 1 + occupation, self._photo_kernel, inverse)
            + recombination_ots
        )
        radiation = jnp.zeros((self.cells, len(g), len(g)))
        radiation = radiation.at[:, lower, upper].add(operators.pump_up_s)
        radiation = radiation.at[:, upper, lower].add(operators.radiative_down_s)
        radiation = radiation.at[:, 1, 0].set(TWO_PHOTON_A_S)
        collision, ion, q3 = jax.vmap(
            lambda temp: atom_collision_rates(
                atom,
                temp,
                low_temperature_extrapolation=self.low_temperature_extrapolation,
                shell_collision_model=self.shell_collision_model,
            )
        )(t)
        collision = collision.at[:, 1, 2].set(0.0).at[:, 2, 1].set(0.0)
        mixing = jax.vmap(
            lambda temp, electron: mixing_bound_bound_cm3_s(atom, temp, electron)
        )(t, ne)
        # RateCoefficients multiplies this field by ne; the collider is np.
        collision = collision + mixing * (proton / ne)[:, None, None]
        rates = RateCoefficients(radiation, collision, photo, alpha, ion, q3)
        thermal = self._thermal_exchange(
            t, bound, proton, ne, rates, occupation, inverse, ots_flux, inverse_line
        )
        return dict(
            rates=rates,
            thermal_heating_erg_cm3_s=thermal[0],
            thermal_cooling_erg_cm3_s=thermal[1],
            thermal_net_erg_cm3_s=thermal[0] - thermal[1],
            line_operators=operators,
            continuum=transfer,
            mean_intensity=transfer.mean_intensity,
            pump_mean_intensity=pump_mean,
            line_pump_mean_intensity=line_pump_mean,
            continuum_opacity_cm1=chi,
            continuum_emission=emission,
            two_photon_emission=two_photon_emission,
            photoionization_ots_s=photo_ots,
            recombination_ots_cm3_s=recombination_ots,
            ots_photon_flux_cm2_s=ots_flux,
            line_continuum_absorption_cm1=line_chi,
            scattering_cm1=scattering,
            transport_extinction_cm1=transport,
            tau_in=tau_in,
            tau_total=tau_total,
            line_extinction_cm1=kappa,
            multiplet_extinction_cm1=multiplet,
            line_albedo=albedo,
            pump_occupation=pump,
            converged=(
                transfer.converged
                & jnp.all(jnp.isfinite(photo + alpha))
                & jnp.all(jnp.isfinite(collision))
                & jnp.all(jnp.isfinite(ion + q3))
                & jnp.all((ne > 0) & (proton >= 0))
                & jnp.all(bound >= 0)
            ),
        )

    def _thermal_exchange(
        self,
        temperature,
        bound,
        proton,
        electron,
        rates,
        occupation,
        inverse,
        ots_flux,
        inverse_line,
    ):
        """Hydrogen kinetic energy only; destroyed photons are allocated, not heat.

        Extra-species emission/opacity affects J but its heating belongs to the
        supplying network. C23 continuum gain suppression is not a general
        signed-transfer energy identity. The returned terms remain the actual
        selected microscopic H rates, including induced recombination.
        """
        atom, grid = self.atom, self.grid
        pair = electron * proton
        photon = PLANCK_CGS * grid.frequency_hz
        excess = photon[None, :] - atom.ionization_energy_erg[:, None]
        photo = bound[:, :, None] * occupation[:, None, :] * self._photo_kernel * excess
        recombination = (
            -pair[:, None, None]
            * inverse
            * (1 + occupation[:, None, :])
            * self._photo_kernel
            * excess
        )
        collisional = (
            electron[:, None, None]
            * bound[:, :, None]
            * rates.electron_bound_bound_cm3_s
            * (
                atom.excitation_energy_erg[:, None]
                - atom.excitation_energy_erg[None, :]
            )
        )
        ionization = (
            -electron[:, None]
            * bound
            * rates.electron_ionization_cm3_s
            * atom.ionization_energy_erg
        )
        three_body = (
            electron[:, None] ** 2
            * proton[:, None]
            * rates.three_body_recombination_cm6_s
            * atom.ionization_energy_erg
        )
        line_excess = (
            PLANCK_CGS * atom.line_frequency_hz[None, :]
            - atom.ionization_energy_erg[:, None]
        )
        ots_photo = (
            bound[:, :, None] * self._sigma_line * ots_flux[:, None, :] * line_excess
        )
        ots_recombination = (
            -pair[:, None, None]
            * inverse_line
            * self._sigma_line
            * ots_flux[:, None, :]
            * line_excess
        )
        prefactor = 2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * grid.frequency_hz**3
        chi_ff, eta_ff = jax.vmap(
            lambda t, ne, np_: hydrogen_free_free_coefficients(
                t, ne, np_, grid.frequency_hz
            )
        )(temperature, electron, proton)
        free_free = (
            4 * jnp.pi * (chi_ff * prefactor * occupation - eta_ff) * grid.weight_hz
        )
        ff_line, _ = jax.vmap(
            lambda t, ne, np_: hydrogen_free_free_coefficients(
                t, ne, np_, atom.line_frequency_hz
            )
        )(temperature, electron, proton)
        ots_free_free = ff_line * ots_flux * PLANCK_CGS * atom.line_frequency_hz
        terms = (
            photo,
            recombination,
            collisional,
            ionization,
            three_body,
            ots_photo,
            ots_recombination,
        )
        if self.include_free_free:
            terms = terms + (free_free, ots_free_free)
        heating = sum(
            jnp.maximum(value, 0.0).reshape(self.cells, -1).sum(axis=1)
            for value in terms
        )
        cooling = -sum(
            jnp.minimum(value, 0.0).reshape(self.cells, -1).sum(axis=1)
            for value in terms
        )
        return heating, cooling

    @partial(jax.jit, static_argnums=0, static_argnames=("iterations",))
    def solve(
        self,
        density_cm3,
        temperature_k,
        incident_mean_intensity,
        initial,
        *,
        inner_intensity=0.0,
        iterations=40,
    ):
        """Implicit pure-H population root; a composition network supplies its own root."""
        density = jnp.broadcast_to(density_cm3, (self.cells,))
        temperature = jnp.broadcast_to(temperature_k, (self.cells,))
        shape = initial.shape

        def evaluate(flat):
            ratios = flat.reshape(shape)
            fractions = jax.nn.softmax(
                jnp.column_stack((ratios, jnp.zeros(self.cells))), axis=1
            )
            bound, electron = (
                density[:, None] * fractions[:, :-1],
                density * fractions[:, -1],
            )
            result = self.population_state(
                temperature,
                bound,
                electron,
                electron,
                incident_mean_intensity,
                inner_intensity=inner_intensity,
            )
            target = jax.vmap(_bound_ratios)(electron, result["rates"])
            return (ratios - jnp.log(target)).ravel(), bound, electron, result

        def residual(flat):
            return evaluate(flat)[0]

        root = jax.lax.custom_root(
            residual,
            initial.ravel(),
            lambda fn, guess: _newton(fn, guess, iterations),
            lambda linear, rhs: jnp.linalg.solve(
                jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs
            ),
        )
        residual_value, bound, electron, radiation = evaluate(root)
        norm = jnp.max(jnp.abs(residual_value))
        return dict(
            state=root.reshape(shape),
            bound_cm3=bound,
            electron_cm3=electron,
            radiation=radiation,
            residual_norm=norm,
            converged=radiation["converged"] & jnp.isfinite(norm) & (norm < 1e-8),
        )
