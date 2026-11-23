"""Conservative formal CRD transfer for a selected group of overlapping lines.

This is an explicit alternative to isolated escape-probability operators.
The selected operators and their old OTS events must be replaced, not added.
No Cloudy overlap prescription, rest-frequency coincidence, or velocity
gradient is assumed. Positive net opacity and a static slab are required.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from exojax.opacity.lpf.lpf import vvoigt

from .hydrogen_rates import BOLTZMANN_CGS as KB
from .hydrogen_rates import LIGHT_SPEED_CGS as C
from .hydrogen_rates import PLANCK_CGS as H
from .radiation import _cell_weights, solve_slab_transfer


class LineOverlapGroup:
    """Frequency-resolved CRD with stable elimination of local self-emission.

    Line metadata have shape [line], populations [cell,line], and continuum
    coefficients [cell,frequency]. Damping is the Lorentz HWHM in Hz; mass is
    in grams. Edges run from the inner to the illuminated outer boundary.
    The grid is fixed; reference_temperature_k sets quadrature spacing only.

    Emission profiles integrate to unity on this quadrature. The absorption
    profile contains exp[(h*nu-transition_energy)/(kT)], the thermodynamic correction
    obtained by imposing Kirchhoff's law with Boltzmann level populations.
    Thus the actual finite-width coefficients, not just their line-center
    limit, have source function B_nu in LTE. This is a declared CRD closure;
    coherent/partial redistribution and pressure wings are not supplied.
    """

    def __init__(
        self,
        edges_cm,
        line_frequency_hz,
        einstein_a_s,
        lower_weight,
        upper_weight,
        mass_g,
        damping_hz,
        *,
        transition_energy_erg=None,
        quadrature_order=4,
        angle_order=8,
        reference_temperature_k=8000.0,
        wing_extent=1000.0,
    ):
        edges = np.asarray(edges_cm, dtype=float)
        arrays = [
            np.asarray(a, dtype=float)
            for a in (
                line_frequency_hz,
                einstein_a_s,
                lower_weight,
                upper_weight,
                mass_g,
                damping_hz,
            )
        ]
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError("Use finite, increasing cell edges.")
        if (
            any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays)
            or not len(arrays[0])
            or any(not np.all(np.isfinite(a) & (a > 0)) for a in arrays)
        ):
            raise ValueError("Line metadata must be positive arrays of equal length.")
        if (
            quadrature_order < 1
            or angle_order < 1
            or reference_temperature_k <= 0
            or wing_extent < 30
        ):
            raise ValueError("Use positive quadrature orders, T, and wing extent >=30.")
        self.edges_cm = jnp.asarray(edges)
        self.widths_cm = jnp.diff(self.edges_cm)
        (
            self.line_frequency_hz,
            self.einstein_a_s,
            self.lower_weight,
            self.upper_weight,
            self.mass_g,
            self.damping_hz,
        ) = map(jnp.asarray, arrays)
        energy = (
            H * arrays[0]
            if transition_energy_erg is None
            else np.asarray(transition_energy_erg)
        )
        if energy.shape != arrays[0].shape or not np.all(
            np.isfinite(energy) & (energy > 0)
        ):
            raise ValueError(
                "Transition energies must be positive and match the line count."
            )
        self.transition_energy_erg = jnp.asarray(energy)
        self.angle_order = angle_order
        offsets = np.unique(
            np.r_[0, 0.25, 0.5, 1, 2, 3, 4, 6, 10, np.geomspace(30, wing_extent, 8)]
        )
        offsets = np.r_[-offsets[:0:-1], offsets]
        widths = arrays[0] / C * np.sqrt(2 * KB * reference_temperature_k / arrays[4])
        splits = np.unique((arrays[0][:, None] + widths[:, None] * offsets).ravel())
        if np.any(splits <= 0):
            raise ValueError("Line grid must remain at positive frequencies.")
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        half = np.diff(splits) / 2
        self.frequency_hz = jnp.asarray(
            ((splits[1:] + splits[:-1])[:, None] / 2 + half[:, None] * nodes).ravel()
        )
        self.weight_hz = jnp.asarray((half[:, None] * weights).ravel())

    def coefficients(self, temperature_k, lower_population_cm3, upper_population_cm3):
        """Return physical line chi, eta, and per-particle absorption coefficients."""
        temperature = jnp.broadcast_to(temperature_k, self.widths_cm.shape)
        nu = self.frequency_hz
        width = (
            self.line_frequency_hz
            * jnp.sqrt(2 * KB * temperature[:, None] / self.mass_g)
            / C
        )
        profile = jax.vmap(
            lambda w: vvoigt(
                nu[None, :] - self.line_frequency_hz[:, None],
                w / jnp.sqrt(2),
                self.damping_hz,
            )
        )(width)
        profile = profile / jnp.sum(profile * self.weight_hz, axis=-1, keepdims=True)
        stimulated = (
            C**2 / (8 * jnp.pi * nu**2) * self.einstein_a_s[None, :, None] * profile
        )
        absorption = stimulated * (self.upper_weight / self.lower_weight)[None, :, None]
        absorption *= jnp.exp(
            (H * nu[None, None, :] - self.transition_energy_erg[None, :, None])
            / (KB * temperature[:, None, None])
        )
        chi = (
            lower_population_cm3[:, :, None] * absorption
            - upper_population_cm3[:, :, None] * stimulated
        )
        eta = (
            H
            * nu
            / (4 * jnp.pi)
            * upper_population_cm3[:, :, None]
            * self.einstein_a_s[None, :, None]
            * profile
        )
        return chi, eta, absorption, stimulated, profile

    def _incoming_and_local_response(self, chi, emission, lower, upper, beam):
        """Avoid subtracting a large trapped local source from the total mean."""
        nodes, weights = np.polynomial.legendre.leggauss(self.angle_order)
        mu, weights = jnp.asarray((nodes + 1) / 2), jnp.asarray(weights / 2)

        def direction(cosine):
            depth = chi * self.widths_cm[:, None] / cosine
            emitted = emission * self.widths_cm[:, None] / cosine
            first, second = _cell_weights(depth)

            def sweep(boundary, reverse):
                def cell(incoming, values):
                    tau, eta, coefficient = values
                    return incoming * jnp.exp(
                        -tau
                    ) + eta * coefficient, incoming * coefficient

                return jax.lax.scan(
                    cell, boundary, (depth, emitted, first), reverse=reverse
                )[1]

            incoming = (sweep(lower, False) + sweep(upper, True)) / 2
            response = self.widths_cm[:, None] / cosine * second
            return incoming, response, first

        incoming, response, escape = jax.vmap(direction)(mu)
        incoming = jnp.einsum("a,acf->cf", weights, incoming)
        response = jnp.einsum("a,acf->cf", weights, response)
        escape = jnp.einsum("a,acf->cf", weights, escape)
        depth = chi * self.widths_cm[:, None]
        above = jnp.concatenate(
            (jnp.cumsum(depth[::-1], axis=0)[:-1][::-1], jnp.zeros_like(depth[:1])),
            axis=0,
        )
        incoming += beam * jnp.exp(-above) * _cell_weights(depth)[0]
        return incoming, response, escape

    def solve(
        self,
        temperature_k,
        lower_population_cm3,
        upper_population_cm3,
        continuum_opacity_cm1,
        continuum_emission,
        incident_mean_intensity,
        *,
        inner_intensity=0.0,
        outer_intensity=0.0,
        scattering_opacity_cm1=0.0,
    ):
        """Return replacement rates and a signed continuum-field correction.

        additional_photon_flux_cm2_s = 4*pi*(J_group-J_continuum)*dnu/(h*nu).
        Supply this to every species' photoionization/induced-recombination
        and kinetic-energy ledgers once. Negative entries represent screening
        of continuum photons. The continuum-only solve uses identical grid,
        coefficients and boundaries, so no incident field is counted twice.

        The caller must use the same continuum interpolation/integration for
        its baseline field; coarse baseline grids can leave a quadrature
        mismatch. Boundary/grid/spatial convergence remain independent checks.
        Optional scattering uses the H continuum attenuation approximation,
        log(1+kappa_s*dz)/dz. It is transport removal, never true absorption.
        """
        lower_pop, upper_pop = (
            jnp.asarray(lower_population_cm3),
            jnp.asarray(upper_population_cm3),
        )
        chi, eta, absorption, stimulated, profile = self.coefficients(
            temperature_k, lower_pop, upper_pop
        )
        scattering = (
            jnp.log1p(scattering_opacity_cm1 * self.widths_cm[:, None])
            / self.widths_cm[:, None]
        )
        continuum_transport = continuum_opacity_cm1 + scattering
        total_chi = continuum_transport + chi.sum(axis=1)
        total_eta = continuum_emission + eta.sum(axis=1)
        zero = jnp.zeros_like(self.frequency_hz)
        lower, upper = zero + inner_intensity, zero + outer_intensity
        kwargs = dict(
            angle_order=self.angle_order, beam_mean_intensity=incident_mean_intensity
        )
        transfer = solve_slab_transfer(
            self.edges_cm, total_chi, total_eta, lower, upper, **kwargs
        )
        baseline = solve_slab_transfer(
            self.edges_cm,
            continuum_transport,
            continuum_emission,
            lower,
            upper,
            **kwargs,
        )
        incoming, local_response, local_escape = self._incoming_and_local_response(
            total_chi, total_eta, lower, upper, incident_mean_intensity
        )
        # Sum other emitters directly: total_eta-eta_i loses tiny contributions
        # when one trapped source dominates. The static line count is small.
        other_eta = jnp.stack(
            [
                continuum_emission + jnp.sum(jnp.delete(eta, i, axis=1), axis=1)
                for i in range(len(self.line_frequency_hz))
            ],
            axis=1,
        )
        other_chi = jnp.stack(
            [
                continuum_transport + jnp.sum(jnp.delete(chi, i, axis=1), axis=1)
                for i in range(len(self.line_frequency_hz))
            ],
            axis=1,
        )
        other_mean = incoming[:, None, :] + local_response[:, None, :] * other_eta
        photon_weight = 4 * jnp.pi * self.weight_hz / (H * self.frequency_hz)
        pump = jnp.sum(absorption * other_mean * photon_weight, axis=-1)
        induced = jnp.sum(stimulated * other_mean * photon_weight, axis=-1)
        safe_total = jnp.where(total_chi > 0, total_chi, 1.0)
        loss = (other_chi + local_escape[:, None, :] * chi) / safe_total[:, None, :]
        loss = jnp.where(total_chi[:, None, :] > 0, loss, 1.0)
        downward = induced + self.einstein_a_s * jnp.sum(
            profile * loss * self.weight_hz, axis=-1
        )
        net_excitation = lower_pop * pump - upper_pop * downward
        radiative_energy = jnp.sum(
            4
            * jnp.pi
            * (chi * transfer.mean_intensity[:, None, :] - eta)
            * self.weight_hz,
            axis=-1,
        )
        redistribution_heat = (
            radiative_energy - self.transition_energy_erg * net_excitation
        )
        valid = (
            transfer.converged
            & baseline.converged
            & jnp.all(chi >= 0)
            & jnp.all(jnp.asarray(scattering_opacity_cm1) >= 0)
            & jnp.all(jnp.asarray(continuum_opacity_cm1) >= 0)
        )
        return dict(
            pump_up_s=jnp.where(valid, pump, jnp.nan),
            radiative_down_s=jnp.where(valid, downward, jnp.nan),
            stimulated_down_s=jnp.where(valid, induced, jnp.nan),
            net_excitation_cm3_s=net_excitation,
            additional_photon_flux_cm2_s=(
                transfer.mean_intensity - baseline.mean_intensity
            )
            * photon_weight,
            redistribution_heat_erg_cm3_s=redistribution_heat.sum(axis=1),
            line_opacity_cm1=chi,
            line_emission=eta,
            transfer=transfer,
            continuum_transfer=baseline,
            scattering_removal_cm1=scattering,
            converged=valid,
        )


__all__ = ["LineOverlapGroup"]
