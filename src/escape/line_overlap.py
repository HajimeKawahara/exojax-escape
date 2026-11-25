"""Conservative formal CRD transfer for a selected group of overlapping lines.

This is an explicit alternative to isolated escape-probability operators.
The selected operators and their old OTS events must be replaced, not added.
No Cloudy overlap prescription, rest-frequency coincidence, or velocity
gradient is assumed. The default static slab requires positive total opacity.
Optional finite spherical chords also support signed total opacity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from exojax.opacity.lpf.lpf import hjert

from escape.hydrogen_rates import BOLTZMANN_CGS as KB
from escape.hydrogen_rates import LIGHT_SPEED_CGS as C
from escape.hydrogen_rates import PLANCK_CGS as H
from escape.radiation import _cell_weights, solve_slab_transfer
from escape.spherical_radiation import (
    spherical_shell_geometry, solve_spherical_scattering, solve_spherical_transfer,
)


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
        scattering_model="attenuation",
        allow_inverted_lines=False,
        geometry="slab",
        incident_geometry="direct",
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
        if geometry not in ("slab", "spherical"):
            raise ValueError("Choose slab or spherical line-transfer geometry.")
        self.geometry = geometry
        if incident_geometry not in ("direct", "spherical_shell_average"):
            raise ValueError("Choose direct or spherical_shell_average incident geometry.")
        if incident_geometry == "spherical_shell_average" and geometry != "spherical":
            raise ValueError("Shell-averaged illumination requires spherical geometry.")
        self.incident_geometry = incident_geometry
        self.shell_geometry = (
            spherical_shell_geometry(edges, angle_order=angle_order)
            if geometry == "spherical" else None
        )
        self.photon_measure_cm = (
            self.widths_cm if self.shell_geometry is None else
            self.shell_geometry.volume_cm3 / self.shell_geometry.area_cm2[-1]
        )
        if scattering_model not in ("attenuation", "background"):
            raise ValueError("Choose attenuation or background scattering.")
        self.scattering_model = scattering_model
        self.allow_inverted_lines = bool(allow_inverted_lines)
        if self.allow_inverted_lines and scattering_model != "background":
            raise ValueError("Inverted lines require the background scattering closure.")
        offsets = np.unique(
            np.r_[0, 0.25, 0.5, 1, 2, 3, 4, 6, 10, np.geomspace(30, wing_extent, 8)]
        )
        offsets = np.r_[-offsets[:0:-1], offsets]
        widths = arrays[0] / C * np.sqrt(2 * KB * reference_temperature_k / arrays[4])
        splits = np.unique((arrays[0][:, None] + widths[:, None] * offsets).ravel())
        if np.any(splits <= 0):
            raise ValueError("Line grid must remain at positive frequencies.")
        self.frequency_edges_hz = jnp.asarray(splits)
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
        def dimensionless_profile(w):
            # The positive Voigt prefactor is common to this line's frequency
            # nodes and cancels exactly from the discrete normalization.
            sfac = 1 / (jnp.sqrt(2.0) * (w / jnp.sqrt(2)))
            offset = nu[None, :] - self.line_frequency_hz[:, None]
            return jax.vmap(
                lambda d, s, a: jax.vmap(hjert, (0, None))(s * d, s * a)
            )(offset, sfac, self.damping_hz)

        profile = jax.vmap(dimensionless_profile)(width)
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
        if self.geometry == "spherical":
            response = self._emission_response(chi)
            local = jnp.diagonal(response, axis1=1, axis2=2).T
            other = response * (1 - jnp.eye(len(self.widths_cm)))[None, ...]
            boundary = self._transfer(chi, jnp.zeros_like(emission), lower, upper, beam)
            incoming = boundary.mean_intensity + jnp.einsum("fcs,sf->cf", other, emission)
            return incoming, local, 1 - local * chi
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

    def _transfer(self, opacity, emission, lower, upper, beam):
        if self.geometry == "spherical":
            # Rotational averaging of a collimated incident field equals an
            # isotropic outer intensity for volume means at prescribed radial
            # coefficients. This is not an angularly resolved SE solution.
            if self.incident_geometry == "spherical_shell_average":
                upper = upper + beam
            return solve_spherical_transfer(
                self.shell_geometry, opacity, emission, lower, upper)
        return solve_slab_transfer(
            self.edges_cm, opacity, emission, lower, upper,
            angle_order=self.angle_order, beam_mean_intensity=beam)

    def _emission_response(self, opacity):
        """Positive cell-mean Green operator [frequency, receiver, source]."""
        cells, frequencies = opacity.shape
        zero = jnp.zeros(frequencies)
        sources = jnp.eye(cells)[:, :, None] * jnp.ones(frequencies)
        response = jax.vmap(
            lambda source: self._transfer(opacity, source, zero, zero, zero).mean_intensity
        )(sources)
        return jnp.transpose(response, (2, 1, 0))

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
        With scattering_model='background', solve the continuum-only isotropic
        scattering field and retain kappa_s*J_background as the source. Return
        signed scattered line energy separately; it is not kinetic heating.
        Frequency-redistributed line photons are omitted, with a discrete
        positive-Green photon-norm bound, not a bound on the coupled atmosphere.
        Spherical geometry uses shell-volume fields and finite signed-opacity
        chords. By default nonzero incident beam is rejected. The explicit
        spherical_shell_average incident geometry maps it to isotropic outer
        intensity, valid for shell-volume means at prescribed radial opacity.
        Absolute photon bounds are per outer spherical area.
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
        if self.scattering_model == "background":
            scattering = jnp.broadcast_to(scattering_opacity_cm1, chi[:, 0, :].shape)
        continuum_transport = continuum_opacity_cm1 + scattering
        zero = jnp.zeros_like(self.frequency_hz)
        lower, upper = zero + inner_intensity, zero + outer_intensity
        background_source = continuum_emission
        if self.scattering_model == "background":
            if self.geometry == "spherical":
                boundary = upper + (incident_mean_intensity
                    if self.incident_geometry == "spherical_shell_average" else 0.)
                coherent = solve_spherical_scattering(
                    self.shell_geometry, continuum_opacity_cm1+zero,
                    continuum_emission+zero, scattering, lower, boundary)
                background_mean = coherent.radiation.mean_intensity
                background_error = coherent.linear_residual_norm
                background_source = continuum_emission + scattering * background_mean
            else:
                response = self._emission_response(continuum_transport)
                unscattered = self._transfer(
                    continuum_transport, continuum_emission, lower, upper,
                    incident_mean_intensity)
                matrix = jnp.eye(len(self.widths_cm))[None, :, :] - (
                    response * scattering.T[:, None, :]
                )
                background_mean = jnp.linalg.solve(
                    matrix, unscattered.mean_intensity.T[..., None]
                )[..., 0].T
                linear_residual = (
                    jnp.einsum("fcs,sf->cf", matrix, background_mean)
                    - unscattered.mean_intensity
                )
                background_error = jnp.max(jnp.abs(linear_residual)) / jnp.maximum(
                    jnp.max(jnp.abs(unscattered.mean_intensity)), jnp.finfo(jnp.float64).tiny
                )
                background_source = continuum_emission + scattering * background_mean
        total_chi = continuum_transport + chi.sum(axis=1)
        total_eta = background_source + eta.sum(axis=1)
        transfer = self._transfer(
            total_chi, total_eta, lower, upper, incident_mean_intensity)
        baseline = self._transfer(
            continuum_transport, background_source, lower, upper, incident_mean_intensity)
        incoming, local_response, local_escape = self._incoming_and_local_response(
            total_chi, total_eta, lower, upper, incident_mean_intensity
        )
        # Sum other emitters directly: total_eta-eta_i loses tiny contributions
        # when one trapped source dominates. The static line count is small.
        other_eta = jnp.stack(
            [
                background_source + jnp.sum(jnp.delete(eta, i, axis=1), axis=1)
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
        if self.geometry == "spherical":
            # The same shell also appears on the far side of tangent rays.
            # Its complete diagonal Green response replaces the slab-local one.
            loss = 1 - local_response[:, None, :] * chi
        downward = induced + self.einstein_a_s * jnp.sum(
            profile * loss * self.weight_hz, axis=-1
        )
        net_excitation = lower_pop * pump - upper_pop * downward
        # Use the same self-eliminated photon events as the rates. Subtract
        # binding energy before integration, avoiding both the trapped local
        # source and two large integrated energy budgets.
        photon_energy_defect = (
            H * self.frequency_hz[None, :]
            - self.transition_energy_erg[:, None]
        )
        occupation = (
            C**2 * other_mean / (2 * H * self.frequency_hz**3)
        )
        temperature = jnp.broadcast_to(temperature_k, self.widths_cm.shape)
        excitation_density = (
            lower_pop[:, :, None]
            * (self.upper_weight / self.lower_weight)[None, :, None]
            * jnp.exp(photon_energy_defect[None, :, :]
                      / (KB * temperature[:, None, None]))
        )
        net_density = (
            excitation_density * occupation
            - upper_pop[:, :, None] * (occupation + loss)
        )
        energy_weight = self.weight_hz[None, :] * photon_energy_defect
        redistribution_heat = jnp.sum(
            self.einstein_a_s[None, :, None] * profile * net_density
            * energy_weight[None, :, :],
            axis=-1,
        )
        valid = (
            transfer.converged
            & baseline.converged
            & (self.allow_inverted_lines | jnp.all(chi >= 0))
            & jnp.all(jnp.asarray(scattering_opacity_cm1) >= 0)
            & jnp.all(jnp.asarray(continuum_opacity_cm1) >= 0)
        )
        if self.geometry == "spherical" and self.incident_geometry == "direct":
            valid &= jnp.all(jnp.asarray(incident_mean_intensity) == 0)
        diagnostics = {}
        if self.scattering_model == "background":
            # The finite-path Green operator is positive even with an inverted
            # individual line or a finite spherical gain column. No negative
            # opacity is interpreted as an absorption-event probability here.
            response = self._emission_response(total_chi)
            removal = jnp.einsum(
                "fcs,cf,c->fs", response, scattering, self.photon_measure_cm
            ) / self.photon_measure_cm
            band_norm = jnp.max(removal)
            # Outside this band assume nonnegative extinction other than the
            # same electron opacity (frequency independent in the caller).
            # This is not a bound for extra scattering or other inverted lines.
            # Pure scattering maximizes the response under these assumptions.
            # This is a cell-averaged bound for this discrete angular operator.
            outside = self._emission_response(scattering)
            outside_norm = jnp.max(jnp.einsum(
                "fcs,cf,c->fs", outside, scattering, self.photon_measure_cm
            ) / self.photon_measure_cm)
            norm = jnp.maximum(band_norm, outside_norm)
            excess = eta.sum(axis=1) - chi.sum(axis=1) * baseline.mean_intensity
            absolute_source = jnp.sum(
                jnp.abs(excess) * photon_weight * self.photon_measure_cm[:, None]
            )
            scattered_energy = 4 * jnp.pi * jnp.sum(
                scattering * (transfer.mean_intensity - baseline.mean_intensity)
                * self.weight_hz, axis=1,
            )
            if self.geometry == "slab":
                valid &= jnp.all(total_chi > 0)
            valid &= (jnp.all(jnp.isfinite(background_mean))
                      & jnp.all(background_mean >= 0) & (norm < 1)
                      & (background_error < 1e-10)
                      & jnp.all(pump >= 0) & jnp.all(downward >= 0))
            diagnostics = dict(
                background_linear_residual=background_error,
                scattering_energy_removed_erg_cm3_s=scattered_energy,
                scattering_photon_norm_bound=norm,
                scattering_band_photon_norm=band_norm,
                scattering_return_factor=norm / (1 - norm),
                absolute_line_excess_photons_cm2_s=absolute_source,
                omitted_scattering_return_photons_cm2_s=(
                    absolute_source * norm / (1 - norm)
                ),
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
            **diagnostics,
        )


__all__ = ["LineOverlapGroup"]
