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
from .hydrogen_completion import critical_principal_n, highest_shell_scale
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
    cell_averaged_line_operators,
    hydrogen_rayleigh_cross_section,
    hydrogen_stark_escape,
    line_cell_quadrature,
    line_operators,
    optical_depth_cell_quadrature,
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



def _replace_complete_mean(original, override):
    """Select a complete continuum field, preserving unselected frequencies."""
    if override is None:
        return original, jnp.zeros(original.shape[1], dtype=bool), jnp.asarray(True)
    mask, value = map(jnp.asarray, override)
    if mask.shape != (original.shape[1],) or value.shape != original.shape:
        raise ValueError("A complete mean override needs mask[frequency], J[cell,frequency].")
    if mask.dtype != jnp.bool_:
        raise ValueError("A complete mean override requires a boolean frequency mask.")
    valid = jnp.all(jnp.where(mask[None, :], jnp.isfinite(value) & (value >= 0), True))
    return jnp.where(mask[None, :], value, original), mask, valid


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
        topoff=False,
        topoff_mask=None,
        resolved_nmax=3,
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
        line_overlap_model="resolved",
        line_sampling="front",
        line_quadrature_order=4,
        line_quadrature="gauss",
        shielding_model="cloudy",
        core_escape_model="cloudy",
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
            for v in (quadrature_order, angle_order, line_quadrature_order)
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
        validate_shell_collision_model(shell_collision_model, resolved_nmax=resolved_nmax)
        self.low_temperature_extrapolation = low_temperature_extrapolation
        self.shell_collision_model = shell_collision_model
        self.scattering_attenuation = bool(scattering_attenuation)
        self.include_free_free = bool(include_free_free)
        if line_sampling not in ("front", "average"):
            raise ValueError("Choose front or average line sampling.")
        self.line_sampling = line_sampling
        self.line_quadrature_order = line_quadrature_order
        if line_quadrature not in ("gauss", "optical_depth"):
            raise ValueError("Choose gauss or optical_depth line quadrature.")
        if line_quadrature == "optical_depth" and line_sampling != "average":
            raise ValueError("Optical-depth line quadrature requires average sampling.")
        self.line_quadrature = line_quadrature
        if shielding_model not in ("cloudy", "continuous_core"):
            raise ValueError("Choose cloudy or continuous_core shielding.")
        self.shielding_model = shielding_model
        if core_escape_model not in ("cloudy", "continuous"):
            raise ValueError("Choose cloudy or continuous Doppler core escape.")
        self.core_escape_model = core_escape_model
        if line_optical_depth_scale not in (1.0, 2.0):
            raise ValueError(
                "Use line_optical_depth_scale=1 for a finite slab or 2 for the explicit doubled-column approximation."
            )
        self.line_optical_depth_scale = float(line_optical_depth_scale)
        if line_overlap_model not in ("resolved", "coincident"):
            raise ValueError("line_overlap_model must be resolved or coincident.")
        self.line_overlap_model = line_overlap_model
        self.edges_cm = jnp.asarray(edges)
        self.widths_cm = jnp.diff(self.edges_cm)
        self.cells = len(edges) - 1
        self.angle_order = angle_order
        if not isinstance(topoff, bool) or (topoff and nmax <= resolved_nmax):
            raise ValueError("Topoff requires a boolean flag and a collapsed top shell.")
        self.topoff = topoff
        if topoff_mask is None:
            topoff_mask = np.ones(len(edges)-1, dtype=bool)
        else:
            topoff_mask = np.asarray(topoff_mask)
            if topoff_mask.dtype != bool or topoff_mask.shape != (len(edges)-1,):
                raise ValueError("The topoff mask must be a boolean per-cell vector.")
        self.topoff_mask = jnp.asarray(topoff_mask)
        self.atom = atom = make_hydrogen_atom(nmax, resolved_nmax=resolved_nmax)
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
        self.frequency_edges_hz = jnp.asarray(splits)
        self.quadrature_order = quadrature_order
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        delta = np.diff(np.log(splits))[:, None]
        frequency = splits[:-1, None] * np.exp(delta * (nodes + 1) / 2)
        self.grid = FrequencyGrid(
            jnp.asarray(frequency.ravel()),
            jnp.asarray((frequency * delta * weights / 2).ravel()),
        )
        # Only these continuum samples enter interpolation at H line centers.
        # A remote bound-free edge must not set the line spatial node scale.
        right = np.searchsorted(frequency.ravel(), np.asarray(atom.line_frequency_hz))
        self._line_grid_indices = np.unique(np.clip(
            np.r_[right-1, right], 0, frequency.size-1))
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

    def _bound_free_scale(self, temperature):
        """One dynamic completion factor for all forward/inverse BF paths."""
        inverse = jax.vmap(lambda t: atom_milne_population(
            self.atom, t, self.grid.frequency_hz))(jnp.asarray(temperature))
        spontaneous = jnp.einsum("lf,clf->cl", self._photo_kernel, inverse)
        scale = jax.vmap(highest_shell_scale)(temperature, spontaneous)
        return jnp.where(self.topoff_mask[:, None], scale, 1.)

    def _full_atom_valid(self, temperature, bound, proton, electron):
        nuclei = bound.sum(axis=1) + proton
        critical = critical_principal_n(temperature, electron, nuclei, proton)
        return jnp.all(~self.topoff_mask | (critical > self.atom.nmax))

    def _continuum(self, temperature, bound, proton, electron, frequency, sigma):
        inverse = jax.vmap(lambda t: atom_milne_population(self.atom, t, frequency))(
            temperature
        )
        inverse_population = (electron * proton)[:, None, None] * inverse
        sigma_cell = sigma[None, :, :]
        if self.topoff:
            sigma_cell = sigma_cell * self._bound_free_scale(temperature)[:, :, None]
        net = sigma_cell * (bound[:, :, None] - inverse_population)
        # Altered C23 opacity_add1subshell.cpp: suppress gain per level.
        absorption = jnp.maximum(net, 0.0).sum(axis=1)
        emission = (
            2
            * PLANCK_CGS
            / LIGHT_SPEED_CGS**2
            * frequency**3
            * jnp.sum(sigma_cell * inverse_population, axis=1)
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

        Flux is integrated over each event frequency, not per Hz. Signed
        corrections from an overlap-group minus continuum-only solve are
        also accepted. This helper
        neither produces nor transports photons; the supplying network owns
        that accounting. It includes induced recombination through the same
        Milne inverse, and optional H free-free absorption. Sum its ledger
        with the returned H continuum/line ledger exactly once.
        """
        frequency = jnp.asarray(frequency_hz)
        sigma = atom_cross_sections(self.atom, frequency)
        return self._photoionization_from_cross_sections(
            temperature_k, bound_cm3, proton_cm3, electron_cm3,
            frequency, photon_flux_cm2_s, sigma)

    def _photoionization_from_cross_sections(
        self, temperature_k, bound_cm3, proton_cm3, electron_cm3,
        frequency, photon_flux_cm2_s, sigma,
    ):
        """Use cross sections evaluated at exactly the supplied event frequencies."""
        flux, bound = jnp.asarray(photon_flux_cm2_s), jnp.asarray(bound_cm3)
        temperature = jnp.broadcast_to(temperature_k, (self.cells,))
        inverse = jax.vmap(lambda t: atom_milne_population(self.atom, t, frequency))(
            temperature
        )
        photo = flux @ sigma.T
        recombination = jnp.einsum("cf,lf,clf->cl", flux, sigma, inverse)
        if self.topoff:
            scale = self._bound_free_scale(temperature)
            photo, recombination = photo * scale, recombination * scale
        excess = PLANCK_CGS * frequency - self.atom.ionization_energy_erg[:, None]
        positive = bound[:, :, None] * flux[:, None, :] * sigma * excess
        negative = (
            -(electron_cm3 * proton_cm3)[:, None, None]
            * inverse
            * flux[:, None, :]
            * sigma
            * excess
        )
        if self.topoff:
            positive = positive * scale[:, :, None]
            negative = negative * scale[:, :, None]
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
            ff_exchange = chi_ff * flux * PLANCK_CGS * frequency
            heating = heating + jnp.maximum(ff_exchange, 0.0).sum(axis=1)
            cooling = cooling - jnp.minimum(ff_exchange, 0.0).sum(axis=1)
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

    def forward_diffuse_at_fraction(self, extinction, emission, fraction):
        """One-way continuum J at a position inside every uniform cell.

        Fraction is scalar or [cell]. Zero is the illuminated face and one is the dark face.
        The angular quadrature and dark outer diffuse boundary match
        _forward_diffuse; this does not add a new bottom-side pump source.
        """
        x, w = np.polynomial.legendre.leggauss(self.angle_order)
        fractions = jnp.broadcast_to(jnp.asarray(fraction), (self.cells,))

        def direction(mu):
            tau = extinction * self.widths_cm[:, None] / mu
            emitted = emission * self.widths_cm[:, None] / mu
            full_weight = _cell_weights(tau)[0]
            point_weight = _cell_weights(fractions[:, None] * tau)[0]

            def cell(incoming, values):
                depth, source, full, point, location = values
                outgoing = incoming * jnp.exp(-depth) + source * full
                inside = (incoming * jnp.exp(-location * depth)
                          + location * source * point)
                return outgoing, inside

            return jax.lax.scan(
                cell, jnp.zeros_like(extinction[0]),
                (tau, emitted, full_weight, point_weight, fractions), reverse=True,
            )[1]

        values = jax.vmap(direction)(jnp.asarray((x + 1) / 2))
        return jnp.einsum("a,acf->cf", jnp.asarray(w / 4), values)

    def forward_diffuse_field(self, extinction, emission):
        """Prepare one-way continuum rays for arbitrary per-line cell nodes.

        Retain both bracketing frequencies until after their pointwise formal
        solutions are evaluated. Interpolating cell-average intensities would
        not give the radiation at state-dependent line quadrature positions.
        """
        x, w = np.polynomial.legendre.leggauss(self.angle_order)
        mu = jnp.asarray((x+1)/2)
        depth = extinction[None, ...]*self.widths_cm[None, :, None]/mu[:, None, None]
        emitted = emission[None, ...]*self.widths_cm[None, :, None]/mu[:, None, None]

        def direction(tau, source):
            def cell(incoming, values):
                t, e = values
                return incoming*jnp.exp(-t)+e*_cell_weights(t)[0], incoming
            return jax.lax.scan(cell, jnp.zeros_like(extinction[0]),
                                (tau, source), reverse=True)[1]

        return dict(frequency_hz=self.grid.frequency_hz,
                    incoming=jax.vmap(direction)(depth, emitted),
                    optical_depth=depth, emitted=emitted,
                    angular_weight=jnp.asarray(w/4))

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
        line_rate_overrides=None,
        retain_line_nodes=False,
        continuum_mean_intensity_override=None,
        line_mean_intensity_override=None,
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
        Optional line_rate_overrides=(indices, pump_up, radiative_down,
        destroyed_photons) replaces selected rates and old OTS source events.
        An optional fifth entry supplies a weight broadcastable to [cell,line]
        in [0,1], blending all three replacements with the original closure.
        A zero weight retains the original entries even if the replacement is
        invalid; this supports a continuous transition between closed models.
        The returned probability/stimulated fields describe the original
        isolated closure at overridden indices; only up/down/photons there
        are replacements. line_rate_override_mask identifies those entries.
        retain_line_nodes exposes the original point probabilities/columns
        so a coupled process can use the identical finite-cell quadrature.
        Optional continuum/line mean overrides are (frequency_mask, complete_J)
        with complete_J[cell, frequency]. They replace, never add to, the old
        direct-plus-diffuse field. Line overrides use one cell-volume mean at
        every line-depth node; this is an explicit unresolved-cell reduction.
        All continuum rates and kinetic terms use the overridden mean. The
        returned ``continuum`` retains the native slab transport diagnostic;
        ``mean_intensity`` is the actual shared field used in these rates.
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
        continuum_mean, continuum_override_mask, continuum_override_valid = (
            _replace_complete_mean(transfer.mean_intensity, continuum_mean_intensity_override)
        )
        prefactor = 2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * grid.frequency_hz**3
        occupation = continuum_mean / prefactor
        depth = transport * self.widths_cm[:, None]
        incident = (
            incident_mean_intensity
            * jnp.exp(-_column_above(depth))
            * _cell_weights(depth)[0]
        )
        forward_diffuse = self._forward_diffuse(transport, emission)
        pump_mean, _, _ = _replace_complete_mean(
            incident + forward_diffuse, continuum_mean_intensity_override)
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
        if self.line_overlap_model == "coincident":
            # Diagnostic C23-like principal-shell degeneracy in internal
            # column opacity only. NIST energies and observer are unchanged.
            fine = multiplet
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
        line_pump_mean, pump_override_mask, pump_override_valid = _replace_complete_mean(
            line_incident + line_diffuse, line_mean_intensity_override)
        pump = line_pump_mean / (
            2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * atom.line_frequency_hz**3
        )
        operator_arguments = dict(
            shielding_model=self.shielding_model,
            core_escape_model=self.core_escape_model,
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
        normalization_error = jnp.zeros(self.cells)
        if self.line_sampling == "average":
            positions, weights = line_cell_quadrature(self.line_quadrature_order)
            supplied_quadrature = {}
            if self.line_quadrature == "optical_depth":
                angle_nodes = np.polynomial.legendre.leggauss(self.angle_order)[0]
                minimum_mu = (angle_nodes[0]+1)/2
                beta_lya = line_chi[:, 0]/(jnp.sqrt(jnp.pi)*kappa[:, 0]+line_chi[:, 0])
                relevant_depth = (transport*self.widths_cm[:, None])[:, self._line_grid_indices]
                # Smooth conservative norm, including the direct exact-line
                # field and the diffuse field's actual interpolation samples.
                # The positive unit term regularizes the zero-opacity limit;
                # it changes node placement only, never an opacity or rate.
                continuum_scale = jnp.sqrt(1+jnp.sum(relevant_depth**2, axis=1)
                    +jnp.sum(line_depth_continuum**2, axis=1))/minimum_mu
                positions, weights, normalization_error = optical_depth_cell_quadrature(
                    tau_in[:, 0], tau_total[0], line_depth[:, 0],
                    continuum_scale, beta_lya, self.line_quadrature_order)
                supplied_quadrature = dict(positions=positions[..., None],
                                           weights=weights[..., None])
            location = positions[:, None, None] if positions.ndim == 1 else positions[..., None]
            diffuse_nodes = jax.vmap(
                lambda position: self.forward_diffuse_at_fraction(
                    transport, emission, position)
            )(positions)
            diffuse_lines = jax.vmap(jax.vmap(
                lambda row: jnp.interp(atom.line_frequency_hz, grid.frequency_hz, row)
            ))(diffuse_nodes)
            incident_nodes = incident_line_mean_intensity * jnp.exp(
                -_column_above(line_depth_continuum)[None, :, :]
                - location * line_depth_continuum[None, :, :]
            )
            complete_nodes = incident_nodes + diffuse_lines
            if line_mean_intensity_override is not None:
                complete_nodes = jnp.where(pump_override_mask[None, None, :],
                    line_pump_mean[None, ...], complete_nodes)
            occupation_nodes = complete_nodes / (
                2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * atom.line_frequency_hz**3)
            stark_nodes = hydrogen_stark_escape(
                atom.principal_n[upper], atom.principal_n[lower],
                ne[None, :, None], t[None, :, None],
                tau_in[None, :, :] + location * line_depth[None, :, :],
            )
            sampled_operators = cell_averaged_line_operators(
                self._kind, line_optical_depth_increment=line_depth,
                quadrature_order=self.line_quadrature_order,
                node_incident_occupation=occupation_nodes,
                node_stark_escape_probability=stark_nodes,
                return_nodes=retain_line_nodes, **operator_arguments,
                **supplied_quadrature,
            )
            if retain_line_nodes:
                operators, point_operators = sampled_operators
            else:
                operators = sampled_operators
            line_pump_mean = (jnp.tensordot(weights, complete_nodes, axes=1)
                if weights.ndim == 1 else
                jnp.sum(weights[..., None]*complete_nodes, axis=0))
            pump = line_pump_mean / (
                2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * atom.line_frequency_hz**3)
        else:
            operators = line_operators(self._kind, **operator_arguments)
            positions, weights = jnp.zeros(1), jnp.ones(1)
            if retain_line_nodes:
                point_operators = jax.tree.map(lambda value: value[None, ...], operators)
        line_nodes = None
        if retain_line_nodes:
            location = positions[:, None, None] if positions.ndim == 1 else positions[..., None]
            line_nodes = dict(
                operators=point_operators, weights=weights,
                positions=positions, weight_normalization_error=normalization_error,
                tau_in=tau_in[None, ...]+location*line_depth[None, ...],
                damping=damping,
                doppler_width_cm_s=jnp.sqrt(2*BOLTZMANN_CGS*t/PROTON_MASS_CGS),
            )
        override_valid = jnp.asarray(True)
        override_mask = jnp.zeros(len(atom.line_frequency_hz), dtype=bool)
        if line_rate_overrides is not None:
            if len(line_rate_overrides) not in (4, 5):
                raise ValueError("Line overrides require four entries and an optional weight.")
            indices, up, down, photons = line_rate_overrides[:4]
            indices = jnp.asarray(indices, dtype=int)
            shape = (self.cells, len(indices))
            if indices.ndim != 1 or any(
                jnp.shape(a) != shape for a in (up, down, photons)
            ):
                raise ValueError("Line overrides must have shape [cell,line].")
            weight = jnp.broadcast_to(
                1. if len(line_rate_overrides) == 4 else line_rate_overrides[4], shape)
            def blend(original, replacement):
                old = original[:, indices]
                replacement = jnp.where((weight == 0) & ~jnp.isfinite(replacement),
                                        old, replacement)
                old = jnp.where((weight == 1) & ~jnp.isfinite(old), replacement, old)
                return (1-weight)*old+weight*replacement
            up = blend(operators.pump_up_s, up)
            down = blend(operators.radiative_down_s, down)
            photons = blend(operators.destroyed_photons_cm3_s, photons)
            override_valid = jnp.all((indices >= 0) & (indices < len(override_mask)))
            override_valid &= jnp.all(jnp.isfinite(weight) & (weight >= 0) & (weight <= 1))
            override_valid &= jnp.all(jnp.diff(jnp.sort(indices)) > 0)
            override_valid &= jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(a) & (a >= 0)) for a in (up, down, photons)]
                )
            )
            override_mask = override_mask.at[indices].set(jnp.any(weight > 0, axis=0))
            operators = operators._replace(
                pump_up_s=operators.pump_up_s.at[:, indices].set(up),
                radiative_down_s=operators.radiative_down_s.at[:, indices].set(down),
                destroyed_photons_cm3_s=operators.destroyed_photons_cm3_s.at[
                    :, indices
                ].set(photons),
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
        if self.topoff:
            scale = self._bound_free_scale(t)
            photo, alpha = photo * scale, alpha * scale
            photo_ots, recombination_ots = photo_ots * scale, recombination_ots * scale
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
        if self.topoff:
            ion = ion.at[:, -1].multiply(jnp.where(self.topoff_mask, 100., 1.))
            q3 = q3.at[:, -1].multiply(jnp.where(self.topoff_mask, 100., 1.))
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
            line_nodes=line_nodes,
            line_quadrature_normalization_error=normalization_error,
            line_rate_override_mask=override_mask,
            continuum=transfer,
            mean_intensity=continuum_mean,
            continuum_mean_override_mask=continuum_override_mask,
            line_mean_override_mask=pump_override_mask,
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
                & (self._full_atom_valid(t, bound, proton, ne) if self.topoff else True)
                & continuum_override_valid & pump_override_valid
                & override_valid
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
        if self.topoff:
            scale = self._bound_free_scale(temperature)
            photo = photo * scale[:, :, None]
            recombination = recombination * scale[:, :, None]
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
        if self.topoff:
            # C23 excludes the complete enhanced top CI/three-body pair.
            # The default off model retains its physical highest-level heat;
            # C23 itself omits that level even with topoff off.
            ionization = ionization.at[:, -1].multiply(~self.topoff_mask)
            three_body = three_body.at[:, -1].multiply(~self.topoff_mask)
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
        if self.topoff:
            ots_photo = ots_photo * scale[:, :, None]
            ots_recombination = ots_recombination * scale[:, :, None]
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
