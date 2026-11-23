"""Coupled native radiation, composition and flow for a solar-abundance proxy.

This is a reduced C23-style model, not a call to Cloudy or a recovered author
configuration. Its retained species, empirical thermal closures and omitted
processes are explicit. Scientific adequacy requires the external comparison
and refinement checks; convergence of this root is only a numerical result.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .atmosphere_wind import thermal_flow_terms, wind_from_profiles
from .free_free import FreeFreeContinuum
from .helium import GroundHelium
from .hydrogen_atom import atom_natural_width_hz, atom_saha_factors
from .hydrogen_continuum import planck_intensity_hz
from .hydrogen_escape_probability import HydrogenEscapeProbabilityModel, _column_above
from .hydrogen_forward import PLANET_MASS_MJ, PLANET_RADIUS_CM
from .hydrogen_molecules import (
    HMINUS_THRESHOLD_RYD, MolecularPhotoRates, RYDBERG_ERG, SAHA, attachment_cooling,
    h2_small_photo_rates, h2_small_thermal_exchange, hminus_equilibrium_factor,
    molecular_continuum_coefficients, molecular_photo_rates,
    molecular_reaction_balance,
)
from .hydrogen_nlte import _bound_ratios
from .hydrogen_rates import BOLTZMANN_CGS, LIGHT_SPEED_CGS, PLANCK_CGS
from .iron_lyalpha import iron_lyalpha_exchange
from .line_escape_probability import line_cell_quadrature
from .metal_network import ATOMS, MetalNetwork, metal_frequency_edges
from .metal_rates import load_metal_partitions, log_metal_partition
from .line_overlap import LineOverlapGroup
from .nlte_forward import HydrogenNLTEForwardModel
from .nonlinear import implicit_newton_krylov, newton_krylov
from .parker import PROTON_MASS_CGS
from .reactive_flow import _newton
from .stellar_radiation import prepare_tabulated_boundary

# Explicit retained subset of C23 default.abn, not a metallicity calibration.
SOLAR_ABUNDANCE = {6: 2.45e-4, 7: 8.51e-5, 8: 4.90e-4, 12: 3.47e-5,
                   13: 2.95e-6, 14: 3.47e-5, 16: 1.84e-5, 20: 2.29e-6,
                   25: 2.88e-7, 26: 2.82e-5}
HELIUM_ABUNDANCE = .1
ATOMIC_MASS_G = 1.6605390666e-24


def _relax_composition(residual, initial, iterations=12):
    """Numerical seed only: accept damped fixed-point steps that reduce error."""
    def step(_, value):
        before = residual(value)
        candidate = value-.5*before
        after = residual(candidate)
        accepted = jnp.all(jnp.isfinite(after)) & (jnp.max(jnp.abs(after)) < jnp.max(jnp.abs(before)))
        return jnp.where(accepted, candidate, value)
    return jax.lax.fori_loop(0, iterations, step, initial)


class SolarCompositionModel:
    """One radiation/SE/chemistry root with conserved nuclei and charge.

    Cells run from the inner to the illuminated outer face. H bound levels,
    H+, H-, H2+, H2 and H2* share a hydrogen-nucleus softmax; helium and each
    retained metal have their own conserved totals. Temperature and density
    are inputs here; SolarAtmosphereModel closes them with energy and wind.
    """

    def __init__(self, edges_cm, *, nmax=15, quadrature_order=4, angle_order=4,
                 atoms=ATOMS, extra_frequency_edges=(), line_optical_depth_scale=2.,
                 linear_solver="krylov", oxygen_overlap=True, overlap_options=None,
                 shell_collision_model="anderson_lebedev", iron_levels=716, sulfur_levels=72,
                 line_sampling="front", line_quadrature_order=4, shielding_model="cloudy",
                 core_escape_model="cloudy", iron_lyalpha=False,
                 newton_merit="residual"):
        if linear_solver not in ("dense", "krylov"):
            raise ValueError("Choose dense or krylov Newton linear solves.")
        self.linear_solver = linear_solver
        if newton_merit not in ("residual", "correction"):
            raise ValueError("Choose residual or correction Newton merit.")
        self.newton_merit = newton_merit
        molecular_edges = np.r_[
            np.array([.052, HMINUS_THRESHOLD_RYD, .502, 1.03, 1.807139])*RYDBERG_ERG/PLANCK_CGS,
            LIGHT_SPEED_CGS/(np.array([912., 1000., 1110.])*1e-8),
        ]
        helium_edges = np.array([24.5874, 54.4178])*1.602176634e-12/PLANCK_CGS
        self.hydrogen = HydrogenEscapeProbabilityModel(
            edges_cm, nmax=nmax, quadrature_order=quadrature_order, angle_order=angle_order,
            extra_frequency_edges=np.unique(np.r_[metal_frequency_edges(), molecular_edges,
                                                  helium_edges, extra_frequency_edges]),
            include_free_free=False,
            shell_collision_model=shell_collision_model,
            line_optical_depth_scale=line_optical_depth_scale,
            line_sampling=line_sampling, line_quadrature_order=line_quadrature_order,
            shielding_model=shielding_model,
            core_escape_model=core_escape_model,
        )
        h = self.hydrogen
        self.grid, self.widths_cm, self.cells = h.grid, h.widths_cm, h.cells
        self.radius_cm = (h.edges_cm[:-1]+h.edges_cm[1:])/2
        line_options = dict(line_sampling=line_sampling,
                            line_quadrature_order=line_quadrature_order,
                            shielding_model=shielding_model,
                            core_escape_model=core_escape_model)
        self.metal = MetalNetwork(self.grid.frequency_hz, self.grid.weight_hz, self.widths_cm,
                                 atoms=atoms, line_optical_depth_scale=line_optical_depth_scale,
                                 iron_levels=iron_levels, sulfur_levels=sulfur_levels,
                                 **line_options)
        self.helium = GroundHelium(self.grid.frequency_hz, self.grid.weight_hz)
        self.free_free = FreeFreeContinuum()
        self.partition_data = load_metal_partitions()
        self.abundance = jnp.array([SOLAR_ABUNDANCE[z] for z in self.metal.atomic_number])
        self.mass_per_h = ATOMIC_MASS_G*(1.00794+4.002602*HELIUM_ABUNDANCE
                                        +sum(m*SOLAR_ABUNDANCE[z] for z, m in self.metal.elements))
        self.hydrogen_size = len(h.atom.statistical_weights)
        self.helium_slice = slice(self.hydrogen_size+4, self.hydrogen_size+6)
        self.metal_slice = slice(self.hydrogen_size+6, None)
        self.state_size = self.hydrogen_size+6+self.metal.state_size
        self.molecule_nuclei = jnp.array([1., 2., 2., 2.])
        self.iron_lyalpha = bool(iron_lyalpha)
        if self.iron_lyalpha:
            if ("Fe II", 26, 1) not in self.metal.atom_specs:
                raise ValueError("Ly-alpha/Fe exchange requires a retained Fe II atom.")
            self.iron_index = self.metal.atom_specs.index(("Fe II", 26, 1))
            selected = np.flatnonzero((np.asarray(h.atom.line_lower) == 0)
                                     & (np.asarray(h.atom.line_upper) == 2))
            if len(selected) != 1:
                raise ValueError("Ly-alpha/Fe exchange requires a resolved H 2p transition.")
            self.hydrogen_lyalpha_index = int(selected[0])
        self.overlap = None
        if oxygen_overlap and ("O I", 8, 0) in self.metal.atom_specs:
            self.oxygen_group = meta = self.metal.oxygen_lybeta_group_metadata()
            selected = np.flatnonzero((np.asarray(h.atom.line_lower) == 0)
                & (np.asarray(h.atom.principal_n)[np.asarray(h.atom.line_upper)] == 3))
            if len(selected) != 1:
                raise ValueError("The H/O overlap requires a resolved H Ly-beta transition.")
            self.hydrogen_lybeta_index = index = int(selected[0])
            upper, lower = h.atom.line_upper[index], h.atom.line_lower[index]
            options = {} if overlap_options is None else dict(overlap_options)
            self.overlap = LineOverlapGroup(edges_cm,
                jnp.r_[h.atom.line_frequency_hz[index], meta["line_frequency_hz"]],
                jnp.r_[h.atom.einstein_a_s[index], meta["einstein_a_s"]],
                jnp.r_[h.atom.statistical_weights[lower], meta["lower_weight"]],
                jnp.r_[h.atom.statistical_weights[upper], meta["upper_weight"]],
                jnp.r_[PROTON_MASS_CGS, meta["mass_g"]],
                jnp.r_[atom_natural_width_hz(h.atom)[index], meta["damping_hz"]],
                transition_energy_erg=jnp.r_[
                    h.atom.excitation_energy_erg[upper]-h.atom.excitation_energy_erg[lower],
                    meta["transition_energy_erg"]], **options)

    def unpack(self, state, density_cm3):
        """Return the same physical species used in opacity, energy and pressure."""
        density = jnp.asarray(density_cm3)
        fractions = jax.nn.softmax(jnp.column_stack((state[:, :self.hydrogen_size+4],
                                                     jnp.zeros(self.cells))), axis=1)
        bound = density[:, None]*fractions[:, :self.hydrogen_size]
        molecule = density[:, None]*fractions[:, self.hydrogen_size:-1]/self.molecule_nuclei
        proton = density*fractions[:, -1]
        helium = self.helium.unpack(state[:, self.helium_slice], HELIUM_ABUNDANCE*density)
        elements = density[:, None]*self.abundance
        metal, levels = self.metal.unpack(state[:, self.metal_slice], elements)
        electron = (proton+helium[:, 1]+2*helium[:, 2]
                    +jnp.einsum("csq,q->c", metal, jnp.arange(3))+molecule[:, 1]-molecule[:, 0])
        particles = bound.sum(1)+proton+molecule.sum(1)+helium.sum(1)+elements.sum(1)+electron
        mu = density*self.mass_per_h/(PROTON_MASS_CGS*particles)
        return dict(bound_cm3=bound, proton_cm3=proton, electron_cm3=electron,
                    molecule_cm3=molecule, helium_cm3=helium, metal_ion_cm3=metal,
                    metal_levels_cm3=levels,
                    element_density_cm3=elements, mean_molecular_weight=mu)

    def initialize(self, density_cm3, temperature_k):
        """Saha/LTE numerical seed; the solved model is not constrained to LTE."""
        density, temperature = jnp.asarray(density_cm3), jnp.asarray(temperature_k)
        saha = jax.vmap(lambda t: atom_saha_factors(self.hydrogen.atom, t))(temperature)
        ion = 2/(1+jnp.sqrt(1+4*density*saha.sum(1)))
        hydrogen = jnp.log(density[:, None]*ion[:, None]*saha)
        molecules = jnp.broadcast_to(jnp.log(jnp.array([1e-10, 1e-10, 1e-8, 1e-9])), (self.cells, 4))
        electron_seed = density*ion
        def inverse_charge_ratio(z, q, threshold):
            weights = self.metal.data["ground_weight"][z, q]
            return (jnp.log(electron_seed*SAHA/temperature**1.5*weights[0]/(2*weights[1]))
                    +threshold/(BOLTZMANN_CGS*temperature))
        he0 = inverse_charge_ratio(2, 0, self.helium.threshold[0])
        he1 = inverse_charge_ratio(2, 1, self.helium.threshold[1])
        helium = jnp.column_stack((-he0, -he0-he1))
        metals = self.metal.initialize(temperature, density[:, None]*self.abundance)
        # Equal charge fractions create a large artificial hard-UV recombination
        # field in cool gas. A Saha charge seed avoids that numerical transient;
        # all final charges still solve the non-LTE rates and shared radiation.
        for index, z in enumerate(self.metal.atomic_number):
            metals = metals.at[:, 2*index].set(inverse_charge_ratio(z, 0, self.metal.threshold_erg[index, 0]))
            metals = metals.at[:, 2*index+1].set(-inverse_charge_ratio(z, 1, self.metal.threshold_erg[index, 1]))
        return jnp.column_stack((hydrogen, molecules, helium, metals))

    def stellar_boundary(self, surface_frequency_hz, surface_flux_nu, *,
                         incident_flux_ratio=.1, outside_zero=False):
        """Prepare static stellar tables, keeping normalization differentiable.

        Return keyword arguments for solve/evaluate. Pass the spectrum's two
        coverage endpoints as extra_frequency_edges at construction to resolve
        an explicit cutoff. The input surface F_nu is hemisphere-integrated;
        the normal stellar beam has J_inc=incident_flux_ratio*F_nu/(4*pi).
        Original stellar arrays are neither stored nor exported by this model.
        """
        hnu, mnu = self.hydrogen.atom.line_frequency_hz, self.metal.line_frequency_hz
        group = np.array([]) if self.overlap is None else np.asarray(self.overlap.frequency_hz)
        continuum, lines = prepare_tabulated_boundary(
            self.hydrogen.frequency_edges_hz, self.grid.weight_hz, self.hydrogen.quadrature_order,
            surface_frequency_hz, surface_flux_nu, line_frequency_hz=np.r_[hnu, mnu, group],
            outside_zero=outside_zero)
        ratio = jnp.asarray(incident_flux_ratio)
        scale = jnp.where(jnp.isfinite(ratio) & (ratio >= 0), ratio/(4*jnp.pi), jnp.nan)
        lines = scale*jnp.asarray(lines)
        return dict(incident_mean_intensity=scale*jnp.asarray(continuum),
            hydrogen_line_intensity=lines[:len(hnu)],
            metal_line_intensity=lines[len(hnu):len(hnu)+len(mnu)],
            overlap_intensity=None if self.overlap is None else lines[len(hnu)+len(mnu):])

    def lte_atomic_target(self, state, density_cm3, temperature_k):
        """Atomic/ionic LTE ratios using actual electrons and retained partitions.

        This is a declared dense-boundary prescription. H-minus follows its
        Saha relation, giving a thermal optical continuum. Other molecular
        coordinates retain their chemical equations.
        Conservation and charge neutrality continue to use ``unpack``.
        """
        t = jnp.asarray(temperature_k)
        physical = self.unpack(state, density_cm3)
        ne = physical["electron_cm3"]
        saha = jax.vmap(lambda temp: atom_saha_factors(self.hydrogen.atom, temp))(t)
        target = state.at[:, :self.hydrogen_size].set(jnp.log(ne[:, None]*saha))
        target = target.at[:, self.hydrogen_size].set(jnp.log(
            ne*hminus_equilibrium_factor(t)*physical["bound_cm3"].sum(1)
            /physical["proton_cm3"]))
        metal = state[:, self.metal_slice]
        partitions = {}
        for atom, (_, z, q), block in zip(self.metal.atoms, self.metal.atom_specs,
                                         self.metal.excitation_slices):
            boltzmann = (jnp.log(atom.weight)[None, :]
                -PLANCK_CGS*LIGHT_SPEED_CGS*atom.energy_wavenumber[None, :]
                    /(BOLTZMANN_CGS*t[:, None]))
            partitions[z, q] = jax.scipy.special.logsumexp(boltzmann, axis=1)
            metal = metal.at[:, block].set(boltzmann[:, 1:]-boltzmann[:, :1])

        def log_inverse_charge(z, q, threshold):
            weights = self.metal.data["ground_weight"][z, q]
            def partition(charge, weight):
                if z == 2:
                    return jnp.log(weight)
                if (z, charge) in partitions:
                    return partitions[z, charge]
                return log_metal_partition(t, z, charge, self.partition_data)
            lower = partition(q, weights[0])
            upper = partition(q+1, weights[1])
            return (jnp.log(ne*SAHA/(2*t**1.5))+lower-upper
                    +threshold/(BOLTZMANN_CGS*t))

        he0 = log_inverse_charge(2, 0, self.helium.threshold[0])
        he1 = log_inverse_charge(2, 1, self.helium.threshold[1])
        target = target.at[:, self.helium_slice].set(jnp.column_stack((-he0, -he0-he1)))
        for index, z in enumerate(self.metal.atomic_number):
            metal = metal.at[:, 2*index].set(log_inverse_charge(z, 0, self.metal.threshold_erg[index, 0]))
            metal = metal.at[:, 2*index+1].set(-log_inverse_charge(z, 1, self.metal.threshold_erg[index, 1]))
        return target.at[:, self.metal_slice].set(metal)

    def _ion_charge_density(self, physical):
        helium, metal = physical["helium_cm3"], physical["metal_ion_cm3"]
        return jnp.column_stack((physical["proton_cm3"]+helium[:, 1]+metal[:, :, 1].sum(1)
                                  +physical["molecule_cm3"][:, 1],
                                 helium[:, 2]+metal[:, :, 2].sum(1), jnp.zeros(self.cells)))

    def other_continuum(self, temperature_k, physical, frequency_hz, *, quadrature=False):
        """All non-atomic-H absorption/emission, including chemistry and free-free."""
        t, f = jnp.asarray(temperature_k), jnp.asarray(frequency_hz)
        ne, neutral = physical["electron_cm3"], physical["bound_cm3"].sum(1)
        metal = self.metal.continuum_coefficients if quadrature else self.metal.continuum_coefficients_at
        args = () if quadrature else (f,)
        chi, eta = metal(t, physical["metal_ion_cm3"], ne, *args,
                         level_population_cm3=physical["metal_levels_cm3"])
        mask = physical.get("atomic_lte_mask")
        if mask is not None:
            # Replace both ground Milne and approximate RR-cascade sources in
            # prescribed dense cells. Their upper-atmosphere C23 closure is
            # unchanged. Full-partition Saha with ground-only Milne would
            # otherwise violate Kirchhoff's law for LTE atomic ratios.
            eta = jnp.where(mask[:, None], chi*planck_intensity_hz(f, t[:, None]), eta)
        hchi, heta = self.helium.continuum_coefficients_at(t, physical["helium_cm3"], ne, f)
        molecules = physical["molecule_cm3"]
        mchi, meta = jax.vmap(lambda temp, h, e, hm, hp: molecular_continuum_coefficients(
            temp, h, e, hm, hp, f))(t, neutral, ne, molecules[:, 0], molecules[:, 1])
        ffchi, ffeta = jax.vmap(lambda temp, e, ions, h: self.free_free.coefficients(
            temp, e, ions, h, f))(t, ne, self._ion_charge_density(physical), neutral)
        return chi+hchi+mchi+ffchi, eta+heta+meta+ffeta

    def continuum_coefficients_at(self, temperature_k, physical, frequency_hz):
        chi, eta = self.other_continuum(temperature_k, physical, frequency_hz)
        hchi, heta = self.hydrogen.continuum_coefficients_at(
            temperature_k, physical["bound_cm3"], physical["proton_cm3"],
            physical["electron_cm3"], frequency_hz,
        )
        return chi+hchi, eta+heta

    def _overlap_state(self, temperature, physical, incident, inner, overlap_intensity):
        group, meta, h = self.overlap, self.oxygen_group, self.hydrogen
        frequency = group.frequency_hz
        chi, eta = self.continuum_coefficients_at(temperature, physical, frequency)
        cascade = self.metal.recombination_emissivity(temperature, physical["metal_ion_cm3"],
            physical["electron_cm3"], frequency, group.weight_hz, partial_coverage=True)
        mask = physical.get("atomic_lte_mask")
        eta += cascade if mask is None else jnp.where(mask[:, None], 0., cascade)
        if overlap_intensity is None:
            overlap_intensity = jnp.interp(frequency, self.grid.frequency_hz, incident)
        inner = jnp.asarray(inner)
        if inner.ndim:
            inner = jnp.interp(frequency, self.grid.frequency_hz, inner)
        oxygen = physical["metal_levels_cm3"][meta["atom_index"]]
        index = self.hydrogen_lybeta_index
        lower = jnp.column_stack((physical["bound_cm3"][:, h.atom.line_lower[index]],
                                  oxygen[:, meta["lower"]]))
        upper = jnp.column_stack((physical["bound_cm3"][:, h.atom.line_upper[index]],
                                  oxygen[:, meta["upper"]]))
        return group.solve(temperature, lower, upper, chi, eta, overlap_intensity,
            inner_intensity=inner, scattering_opacity_cm1=h.scattering_coefficients_at(
                physical["bound_cm3"], physical["electron_cm3"], frequency))

    def _iron_exchange(self, temperature, physical, radiation):
        """Average local PRD exchange at the same nodes as the H line closure."""
        h, m, index = self.hydrogen, self.metal, self.iron_index
        atom, levels = m.atoms[index], physical["metal_levels_cm3"][index]
        upper, lower, frequency, a, _, _ = m.lines[index]
        geometry = m.line_geometry(index, temperature, levels)
        lya = self.hydrogen_lyalpha_index
        hu, hl = h.atom.line_upper[lya], h.atom.line_lower[lya]
        bound, nodes = physical["bound_cm3"], radiation["line_nodes"]
        active = ~physical.get("atomic_lte_mask", jnp.zeros(self.cells, dtype=bool))
        # Dense prescribed LTE cells do not use this empirical non-LTE closure.
        # Supply a regular inactive state, rather than masking NaNs afterwards.
        safe = lambda value, fallback: jnp.where(active, value, fallback)
        common = dict(
            lower_population_cm3=jnp.where(active[:, None], levels[:, lower], 0.),
            upper_population_cm3=jnp.where(active[:, None], levels[:, upper], 0.),
            einstein_a_s=a, weight_ratio=atom.weight[upper]/atom.weight[lower],
            wavenumber_cm=frequency/LIGHT_SPEED_CGS,
            transition_energy_erg=PLANCK_CGS*LIGHT_SPEED_CGS*(
                atom.energy_wavenumber[upper]-atom.energy_wavenumber[lower]),
            total_optical_depth=jnp.broadcast_to(geometry["tau_total"], (self.cells, len(a))),
            hydrogen_ground_cm3=safe(bound[:, hl], 1.),
            hydrogen_2p_cm3=safe(bound[:, hu], 0.),
            hydrogen_einstein_a_s=h.atom.einstein_a_s[lya],
            hydrogen_transition_energy_erg=h.atom.excitation_energy_erg[hu]-h.atom.excitation_energy_erg[hl],
            hydrogen_tau_total=safe(jnp.broadcast_to(radiation["tau_total"][lya], (self.cells,)), 0.),
            hydrogen_damping=safe(nodes["damping"][:, lya], 1.),
            hydrogen_doppler_width_cm_s=safe(nodes["doppler_width_cm_s"], 1.),
            electron_cm3=safe(physical["electron_cm3"], 1.),
        )
        def point(tau, operators):
            return iron_lyalpha_exchange(
                hydrogen_tau_in=safe(tau[:, lya], 0.),
                hydrogen_escape_probability=safe(operators.escape_probability[:, lya], 1.),
                electron_escape_probability=safe(operators.electron_escape_probability[:, lya], 0.),
                continuum_destruction_probability=safe(operators.destruction_probability[:, lya], 0.),
                **common)
        result = jax.vmap(point)(nodes["tau_in"], nodes["operators"])
        average = lambda value: jnp.tensordot(nodes["weights"], value, axes=1)
        return dict(
            upward_s=average(result.upward_s),
            stimulated_downward_s=average(result.stimulated_downward_s),
            hydrogen_upward_s=average(result.hydrogen_upward_s),
            hydrogen_downward_s=average(result.hydrogen_downward_s),
            net_photons_cm3_s=average(result.net_photons_cm3_s),
            redistribution_heat_erg_cm3_s=average(result.redistribution_heat_erg_cm3_s),
            width_cm_s=average(result.width_cm_s),
            maximum_width_residual=jnp.max(jnp.abs(result.width_residual)),
            selection_distance=jnp.min(result.selection_distance, axis=0),
            active=active, valid=jnp.all(result.valid),
        )

    def evaluate(self, state, density_cm3, temperature_k, incident_mean_intensity, *,
                 inner_intensity=0., hydrogen_line_intensity=None, metal_line_intensity=None,
                 overlap_intensity=None, atomic_lte_mask=None):
        """Return the coupled fixed point, all species and a non-duplicated heat ledger."""
        h, m, grid = self.hydrogen, self.metal, self.grid
        t, density = jnp.asarray(temperature_k), jnp.asarray(density_cm3)
        physical = self.unpack(state, density)
        if atomic_lte_mask is not None:
            mask = jnp.asarray(atomic_lte_mask, dtype=bool)
            if mask.shape != (self.cells,):
                raise ValueError("The atomic LTE mask must have one entry per cell.")
            physical["atomic_lte_mask"] = mask
        bound, proton, ne = (physical[k] for k in ("bound_cm3", "proton_cm3", "electron_cm3"))
        molecules, helium = physical["molecule_cm3"], physical["helium_cm3"]
        neutral = bound.sum(1)
        extra_chi, extra_eta = self.other_continuum(t, physical, grid.frequency_hz, quadrature=True)
        hline_extra, _ = self.other_continuum(t, physical, h.atom.line_frequency_hz)
        metal_continuum, _ = m.continuum_coefficients_at(
            t, physical["metal_ion_cm3"], ne, h.atom.line_frequency_hz,
            level_population_cm3=physical["metal_levels_cm3"])
        hline_extra += m.hydrogen_line_opacity_at(
            t, physical["metal_ion_cm3"], ne, h.atom.line_frequency_hz,
            physical["metal_levels_cm3"])-metal_continuum
        overlap, h_override, metal_override = None, None, None
        group_frequency, group_flux = None, None
        if self.overlap is not None:
            overlap = self._overlap_state(t, physical, incident_mean_intensity,
                                          inner_intensity, overlap_intensity)
            up, down = overlap["pump_up_s"], overlap["radiative_down_s"]
            # Replace the isolated H/O rates and their OTS production together.
            h_override = (jnp.array([self.hydrogen_lybeta_index]), up[:, :1], down[:, :1],
                          jnp.zeros_like(up[:, :1]))
            metal_override = {"O I": (self.oxygen_group["line_indices"], up[:, 1:], down[:, 1:],
                                       jnp.zeros_like(up[:, 1:]))}
            group_frequency = self.overlap.frequency_hz
            group_flux = overlap["additional_photon_flux_cm2_s"]
        radiation = h.population_state(t, bound, proton, ne, incident_mean_intensity,
            additional_opacity=extra_chi, additional_emission=extra_eta,
            additional_line_opacity=hline_extra, inner_intensity=inner_intensity,
            incident_line_mean_intensity=hydrogen_line_intensity,
            line_rate_overrides=h_override, retain_line_nodes=self.iron_lyalpha)
        iron_exchange, metal_extra_rates = None, None
        if self.iron_lyalpha:
            iron_exchange = self._iron_exchange(t, physical, radiation)
            index = self.hydrogen_lyalpha_index
            lower, upper = h.atom.line_lower[index], h.atom.line_upper[index]
            rates = radiation["rates"]
            matrix = rates.radiative_bound_bound_s
            matrix = matrix.at[:, lower, upper].add(iron_exchange["hydrogen_upward_s"])
            matrix = matrix.at[:, upper, lower].add(iron_exchange["hydrogen_downward_s"])
            # Direct H/Fe exchange is not an extra continuum OTS source.
            radiation["rates"] = rates._replace(radiative_bound_bound_s=matrix)
            metal_extra_rates = {"Fe II": (iron_exchange["upward_s"],
                                            iron_exchange["stimulated_downward_s"])}
        mline_chi, _ = self.continuum_coefficients_at(t, physical, m.line_frequency_hz)
        diffuse = h._forward_diffuse(radiation["transport_extinction_cm1"], radiation["continuum_emission"])
        mline_diffuse = jax.vmap(lambda row: jnp.interp(m.line_frequency_hz, grid.frequency_hz, row))(diffuse)
        metal_node_options = {}
        metal_incident = metal_line_intensity
        if h.line_sampling == "average":
            if metal_incident is None:
                metal_incident = jnp.interp(m.line_frequency_hz, grid.frequency_hz,
                                            incident_mean_intensity)
            positions, _ = line_cell_quadrature(h.line_quadrature_order)
            point_diffuse = jax.vmap(lambda fraction: h.forward_diffuse_at_fraction(
                radiation["transport_extinction_cm1"], radiation["continuum_emission"],
                fraction))(positions)
            metal_node_options["line_node_diffuse_mean_intensity"] = jax.vmap(
                jax.vmap(lambda row: jnp.interp(m.line_frequency_hz,
                                              grid.frequency_hz, row)))(point_diffuse)
        chemical_rec = jnp.zeros((self.cells, len(m.elements), 2))
        chemical_rec = chemical_rec.at[:, :, 0].set((molecules[:, 0]*4e-6/jnp.sqrt(t))[:, None])
        metal = m.evaluate(t, state[:, self.metal_slice], physical["element_density_cm3"],
            ne, bound[:, 0], proton, helium[:, 0], helium[:, 1], radiation, h.atom.line_frequency_hz,
            additional_recombination_s=chemical_rec, line_incident_mean_intensity=metal_incident,
            line_continuum_opacity_cm1=mline_chi, line_diffuse_mean_intensity=mline_diffuse,
            line_scattering_cm1=h.scattering_coefficients_at(bound, ne, m.line_frequency_hz),
            hydrogen_neutral_cm3=neutral,
            additional_collider_density_cm3={"H2": molecules[:, 2]+molecules[:, 3]},
            line_rate_overrides=metal_override, additional_line_rates=metal_extra_rates,
            additional_photon_frequency_hz=group_frequency,
            additional_photon_flux_cm2_s=group_flux, **metal_node_options)
        ots_frequency = jnp.concatenate((h.atom.line_frequency_hz, m.line_frequency_hz))
        ots_flux = jnp.concatenate((radiation["ots_photon_flux_cm2_s"], metal["ots_photon_flux_cm2_s"]), axis=1)
        receiving_frequency, receiving_flux = m.line_frequency_hz, metal["ots_photon_flux_cm2_s"]
        if overlap is not None:
            ots_frequency = jnp.concatenate((ots_frequency, group_frequency))
            ots_flux = jnp.concatenate((ots_flux, group_flux), axis=1)
            receiving_frequency = jnp.concatenate((receiving_frequency, group_frequency))
            receiving_flux = jnp.concatenate((receiving_flux, group_flux), axis=1)
        he_ion = jnp.zeros((self.cells, 2)).at[:, 0].set(metal["helium_ct_ionization_s"])
        he_rec = jnp.zeros((self.cells, 2)).at[:, 0].set(metal["helium_ct_recombination_s"])
        he = self.helium.evaluate(t, helium, ne, bound[:, 0], proton, radiation["mean_intensity"],
            ots_frequency_hz=ots_frequency, ots_flux_cm2_s=ots_flux,
            extra_ionization_s=he_ion, extra_recombination_s=he_rec)
        flux = 4*jnp.pi*radiation["mean_intensity"]*grid.weight_hz/(PLANCK_CGS*grid.frequency_hz)
        continuous = jax.vmap(lambda temp, row: molecular_photo_rates(temp, grid.frequency_hz, row))(t, flux)
        events = jax.vmap(lambda temp, row: molecular_photo_rates(temp, ots_frequency, row))(t, ots_flux)
        photo = MolecularPhotoRates(*(a+b for a, b in zip(continuous, events)))
        # Elwert's band field uses the actual stellar boundary and attenuation at 1000 A.
        wavelength = LIGHT_SPEED_CGS/grid.frequency_hz*1e8
        band = (wavelength >= 912.) & (wavelength <= 1110.)
        face = jnp.sum(jnp.where(band, 4*jnp.pi*incident_mean_intensity*grid.weight_hz/(PLANCK_CGS*grid.frequency_hz), 0.))/(1.232e7*1.71)
        uv_depth = jax.vmap(lambda row: jnp.interp(LIGHT_SPEED_CGS/1e-5, grid.frequency_hz, row))(radiation["continuum_opacity_cm1"])*self.widths_cm
        local_field = face*jnp.exp(-_column_above(uv_depth)-uv_depth/2)
        molecule_column = _column_above(molecules[:, 2:]*self.widths_cm[:, None])+molecules[:, 2:]*self.widths_cm[:, None]/2
        h2_photo = jax.vmap(h2_small_photo_rates, in_axes=(0, 0, 0, None, 0))(
            t, molecule_column[:, 0], molecule_column[:, 1], face, local_field)
        molecular = jax.vmap(lambda temp, nh, n1, n2, n3, p, e, mol, pr, ph, he0, metal1:
            molecular_reaction_balance(temp, nh, n1, n2, n3, p, e, mol, pr, ph,
                neutral_he_cm3=he0, singly_charged_metals_cm3=metal1))(
            t, neutral, bound[:, 0], bound[:, 1:3].sum(1), bound[:, 3:6].sum(1),
            proton, ne, molecules, photo, h2_photo, helium[:, 0],
            physical["metal_ion_cm3"][:, :, 1].sum(1))
        hydrogen_extra = h.photoionization_from_photon_flux(t, bound, proton, ne,
            receiving_frequency, receiving_flux)
        photo_h = hydrogen_extra["photoionization_s"]
        alpha_h = hydrogen_extra["recombination_cm3_s"]
        ion_extra = metal["hydrogen_ct_ionization_s"]+he["hydrogen_ct_ionization_s"]
        ion_extra += molecular.proton_production_cm3_s/bound[:, 0]
        rec_extra = metal["hydrogen_ct_recombination_s"]+he["hydrogen_ct_recombination_s"]
        rec_extra += molecular.proton_destruction_cm3_s/proton
        photo_h = photo_h.at[:, 0].add(ion_extra)
        alpha_h = alpha_h.at[:, 0].add(rec_extra/ne)
        rates = radiation["rates"]._replace(
            photoionization_s=radiation["rates"].photoionization_s+photo_h,
            radiative_recombination_cm3_s=radiation["rates"].radiative_recombination_cm3_s+alpha_h)
        target_h = jnp.log(jax.vmap(_bound_ratios)(ne, rates))
        target_molecules = (state[:, self.hydrogen_size:self.hydrogen_size+4]
                            +jnp.log(molecular.production_cm3_s/molecular.destruction_cm3_s))
        target = jnp.column_stack((target_h, target_molecules, he["target_state"], metal["target_state"]))
        if atomic_lte_mask is not None:
            atomic = ((jnp.arange(self.state_size) <= self.hydrogen_size)
                      | (jnp.arange(self.state_size) >= self.hydrogen_size+4))
            target = jnp.where(mask[:, None] & atomic[None, :],
                               self.lte_atomic_target(state, density, t), target)
        # Extra OTS photons are allocated to each absorbing species exactly once.
        ff, ff_emission = jax.vmap(lambda temp, e, ions, nh: self.free_free.coefficients(temp, e, ions, nh, grid.frequency_hz))(
            t, ne, self._ion_charge_density(physical), neutral)
        ff_ots, _ = jax.vmap(lambda temp, e, ions, nh: self.free_free.coefficients(temp, e, ions, nh, ots_frequency))(
            t, ne, self._ion_charge_density(physical), neutral)
        ff_net = 4*jnp.pi*jnp.sum((ff*radiation["mean_intensity"]-ff_emission)*grid.weight_hz, axis=1)
        ff_net += jnp.sum(ff_ots*ots_flux*PLANCK_CGS*ots_frequency, axis=1)
        negative_cool, positive_cool = attachment_cooling(t, neutral, proton, ne)
        molecule_net = (molecules[:, 0]*photo.hminus_photo_heat_erg_s
                        -ne*neutral*photo.hminus_induced_cooling_erg_cm3_s
                        +molecules[:, 1]*photo.h2plus_photo_heat_erg_s-negative_cool-positive_cool)
        molecular_h2 = h2_small_thermal_exchange(density, neutral, face, molecules[:, 2:],
            h2_photo, molecular.h2_collision_exchange_erg_cm3_s)
        ledger = dict(hydrogen=radiation["thermal_net_erg_cm3_s"]+hydrogen_extra["net_erg_cm3_s"],
                      metals=metal["energy_erg_cm3_s"]["net_kinetic_heat"],
                      helium=he["thermal_net"], free_free=ff_net,
                      molecular=molecule_net+molecular_h2)
        if overlap is not None:
            ledger["line_redistribution"] = overlap["redistribution_heat_erg_cm3_s"]
        if iron_exchange is not None:
            ledger["iron_lyalpha_redistribution"] = iron_exchange["redistribution_heat_erg_cm3_s"]
        net = sum(ledger.values())
        # Scale by real channels, not the much larger trapped radiative line cycles.
        scale = radiation["thermal_heating_erg_cm3_s"]+radiation["thermal_cooling_erg_cm3_s"]
        scale += sum(jnp.abs(value) for value in ledger.values())
        valid = (radiation["converged"] & metal["valid"] & he["valid"]
                 & jnp.all(ne > 0) & jnp.all((t >= 4000) & (t <= 14000))
                 & jnp.all(jnp.isfinite(target)))
        if overlap is not None:
            valid &= overlap["converged"]
        if iron_exchange is not None:
            valid &= iron_exchange["valid"]
        return target, dict(**physical, temperature_k=t, n_h_cm3=density,
            thermal_net=net, thermal_scale=scale, thermal_ledger=ledger,
            radiation=radiation, metal=metal, helium=he, molecule_balance=molecular,
            overlap=overlap, iron_lyalpha_exchange=iron_exchange,
            valid=valid)

    @partial(jax.jit, static_argnums=0, static_argnames=("iterations",))
    def solve(self, density_cm3, temperature_k, incident_mean_intensity, initial_state, *,
              inner_intensity=0., hydrogen_line_intensity=None, metal_line_intensity=None,
              overlap_intensity=None, atomic_lte_mask=None, iterations=30):
        """Fixed-profile coupled root with implicit derivatives of its residual."""
        shape = (self.cells, self.state_size)
        if initial_state.shape != shape:
            raise ValueError("initial_state must match the composition grid.")
        def evaluate(flat):
            return self.evaluate(flat.reshape(shape), density_cm3, temperature_k, incident_mean_intensity,
                inner_intensity=inner_intensity, hydrogen_line_intensity=hydrogen_line_intensity,
                metal_line_intensity=metal_line_intensity, overlap_intensity=overlap_intensity,
                atomic_lte_mask=atomic_lte_mask)
        def residual(flat):
            target, physical = evaluate(flat)
            return jnp.where(physical["valid"], (flat.reshape(shape)-target).ravel(), jnp.nan)
        initial = jax.lax.stop_gradient(_relax_composition(residual, initial_state.ravel()))
        if self.linear_solver == "krylov":
            root = implicit_newton_krylov(residual, initial, iterations=iterations,
                                          merit=self.newton_merit)
        else:
            root = jax.lax.custom_root(residual, initial,
                lambda fn, guess: _newton(fn, guess, iterations),
                lambda linear, rhs: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs))
        _, physical = evaluate(root)
        norm = jnp.max(jnp.abs(residual(root)))
        return dict(**physical, state=root.reshape(shape), residual_norm=norm,
                    converged=physical["valid"] & (norm < 1e-8))


class SolarAtmosphereModel(SolarCompositionModel):
    """Close the native composition root with thermal balance and a tidal wind.

    The mass flux sets density; solved T/mu sets transonic momentum. Local SE
    omits population advection, while the kinetic energy equation includes
    sunbather's thermal advection and expansion. The innermost T is a supplied
    boundary unless an explicit dense lower patch is selected. That patch
    prescribes atomic LTE and a temperature extension, retaining chemistry,
    radiation and conserved composition. Results are ordered inner-to-outer.
    """

    def __init__(self, radius_edges_rp, *, planet_mass_mj=PLANET_MASS_MJ,
                 tidal_coefficient_s2=0., wind_ngrid=129, observer_options=None,
                 dense_lower_cells=0, lower_temperature_k=4500.,
                 **composition_options):
        edges = np.asarray(radius_edges_rp, dtype=float)
        if edges.ndim != 1 or len(edges) < 4 or edges[0] < 1.:
            raise ValueError("Use at least three atmospheric cells outside one planet radius.")
        super().__init__(edges*PLANET_RADIUS_CM, **composition_options)
        if (not isinstance(dense_lower_cells, (int, np.integer))
                or not 0 <= dense_lower_cells < self.cells-1):
            raise ValueError("Leave at least two non-LTE cells above the dense lower patch.")
        if not 4000 <= lower_temperature_k <= 14000:
            raise ValueError("The lower temperature must lie within the declared thermal domain.")
        self.dense_lower_cells = int(dense_lower_cells)
        self.lower_temperature_k = lower_temperature_k
        self.atomic_lte_mask = (None if dense_lower_cells == 0
                                else jnp.arange(self.cells) < dense_lower_cells)
        self.planet_mass_mj, self.tidal_coefficient_s2 = planet_mass_mj, tidal_coefficient_s2
        self.wind_ngrid = wind_ngrid
        options = {} if observer_options is None else dict(observer_options)
        self.observer = HydrogenNLTEForwardModel(
            radius_edges_rp=edges, nlayer=self.cells, nmax=self.hydrogen.atom.principal_n[-1].item(),
            planet_mass_mj=planet_mass_mj, tidal_coefficient_s2=tidal_coefficient_s2,
            wind_ngrid=wind_ngrid, **options)

    def _wind(self, state, temperature, mass_loss_g_s):
        # Particle ratios do not depend on the temporary unit hydrogen density.
        mu = self.unpack(state, jnp.ones(self.cells))["mean_molecular_weight"]
        radius = self.radius_cm
        if self.dense_lower_cells:
            radius = jnp.r_[PLANET_RADIUS_CM, radius]
            temperature = jnp.r_[self.lower_temperature_k, temperature]
            # The photospheric endpoint carries the innermost solved mu;
            # its small unresolved composition gradient is a boundary choice.
            mu = jnp.r_[mu[0], mu]
        wind, info = wind_from_profiles(radius, temperature, mu, mass_loss_g_s,
            self.planet_mass_mj, base_radius_cm=PLANET_RADIUS_CM,
            outer_radius_cm=32*PLANET_RADIUS_CM, tidal_coefficient_s2=self.tidal_coefficient_s2,
            ngrid=self.wind_ngrid, return_info=True)
        if self.dense_lower_cells:
            wind = wind._replace(velocity_kms=wind.velocity_kms[1:],
                                 density_g_cm3=wind.density_g_cm3[1:])
            info = {**info,
                    "log_velocity_gradient_cm1": info["log_velocity_gradient_cm1"][1:],
                    "sound_squared_gradient_cm_s2": info["sound_squared_gradient_cm_s2"][1:]}
        return wind, info

    def evaluate_atmosphere(self, flat, reference_t, mass_loss_g_s, incident_mean_intensity, *,
                            inner_intensity=0., hydrogen_line_intensity=None,
                            metal_line_intensity=None, overlap_intensity=None,
                            thermal_blend=1., flow_energy=True):
        """Return the complete dimensionless residual and its physical state.

        Exposed for inspectable, checkpointed continuation. A small residual
        is numerical evidence only; callers must still check physical adequacy.
        State rows contain log(T/reference_t) followed by composition coordinates.
        """
        values = flat.reshape((self.cells, self.state_size+1))
        temperature = reference_t*jnp.exp(values[:, 0])
        composition = values[:, 1:]
        wind, wind_info = self._wind(composition, temperature, mass_loss_g_s)
        density = wind.density_g_cm3/self.mass_per_h
        target, physical = self.evaluate(composition, density, temperature, incident_mean_intensity,
            inner_intensity=inner_intensity, hydrogen_line_intensity=hydrogen_line_intensity,
            metal_line_intensity=metal_line_intensity, overlap_intensity=overlap_intensity,
            atomic_lte_mask=self.atomic_lte_mask)
        terms = thermal_flow_terms(self.radius_cm, wind.density_g_cm3, temperature,
            physical["mean_molecular_weight"], wind.velocity_kms*1e5,
            log_velocity_gradient_cm1=wind_info["log_velocity_gradient_cm1"],
            sound_squared_gradient_cm_s2=wind_info["sound_squared_gradient_cm_s2"])
        flow = terms.advection+terms.expansion if flow_energy else jnp.zeros(self.cells)
        scale = physical["thermal_scale"]+jnp.abs(terms.advection)+jnp.abs(terms.expansion)
        # Cooling minus heating has the same sign as a positive temperature
        # displacement on a stable thermal branch; this avoids reversing the
        # continuation Jacobian merely because of a residual sign convention.
        thermal = -thermal_blend*(physical["thermal_net"]+flow)/scale+(1-thermal_blend)*values[:, 0]
        if self.dense_lower_cells:
            join = self.dense_lower_cells
            fraction = ((self.radius_cm[:join]-PLANET_RADIUS_CM)
                        /(self.radius_cm[join]-PLANET_RADIUS_CM))
            lower_t = self.lower_temperature_k+fraction*(temperature[join]-self.lower_temperature_k)
            prescribed = jnp.log(temperature[:join]/lower_t)
            thermal = thermal.at[:join].set(
                thermal_blend*prescribed+(1-thermal_blend)*values[:join, 0])
        else:
            thermal = thermal.at[0].set(values[0, 0])
        residual = jnp.column_stack((thermal, composition-target)).ravel()
        valid = physical["valid"] & wind_info["valid"] & jnp.all(jnp.isfinite(thermal))
        result = dict(**physical, radial_velocity_kms=wind.velocity_kms,
                      thermal_advection=terms.advection, thermal_expansion=terms.expansion,
                      wind_info=wind_info, flow_energy_included=flow_energy)
        pressure = (wind.density_g_cm3*BOLTZMANN_CGS*temperature
                    /(physical["mean_molecular_weight"]*PROTON_MASS_CGS))
        upper = slice(self.dense_lower_cells, None)
        result["pressure_dyn_cm2"] = pressure
        result["thermal_balance_required"] = jnp.arange(self.cells) >= max(1, self.dense_lower_cells)
        result["reference_domain_valid"] = jnp.all(
            (density[upper] <= 1e15) & (pressure[upper] <= 1e3))
        return jnp.where(valid, residual, jnp.nan), result


    @partial(jax.jit, static_argnums=0, static_argnames=("iterations", "continuation_steps", "flow_energy"))
    def solve_atmosphere(self, mass_loss_g_s, temperature_initial_k,
                         incident_mean_intensity, initial_state, *, inner_intensity=0.,
                         hydrogen_line_intensity=None, metal_line_intensity=None,
                         overlap_intensity=None, iterations=30, continuation_steps=8, flow_energy=True):
        """Solve temperatures and all populations, then implicitly differentiate.

        initial_state is a composition seed on this same radial grid. The
        supplied T profile initializes the branch. Without a dense patch,
        its bottom value is fixed. With one, lower cells join the explicitly
        supplied photospheric temperature to the first non-LTE temperature;
        no thermal-balance claim is made for those prescribed cells.
        flow_energy=False selects radiative equilibrium as a declared
        comparison; it does not remove wind/pressure/density feedback.
        """
        reference_t = jnp.asarray(temperature_initial_k)
        if reference_t.shape != (self.cells,) or initial_state.shape != (self.cells, self.state_size):
            raise ValueError("Initial T and composition must match this atmospheric grid.")
        if iterations < 1 or continuation_steps < 1:
            raise ValueError("Iteration counts must be positive.")
        shape = (self.cells, self.state_size+1)
        guess = jnp.column_stack((jnp.zeros(self.cells), initial_state)).ravel()

        def evaluate(flat, thermal_blend=1.):
            return self.evaluate_atmosphere(flat, reference_t, mass_loss_g_s, incident_mean_intensity,
                inner_intensity=inner_intensity, hydrogen_line_intensity=hydrogen_line_intensity,
                metal_line_intensity=metal_line_intensity, thermal_blend=thermal_blend,
                overlap_intensity=overlap_intensity, flow_energy=flow_energy)

        residual = lambda flat: evaluate(flat)[0]

        def solve_root(function, initial):
            del function
            def primal(fn, value):
                if self.linear_solver == "krylov":
                    return newton_krylov(fn, value, iterations=iterations,
                                         merit=self.newton_merit).value
                return _newton(fn, value, iterations)
            fixed_temperature = lambda flat: evaluate(flat, 0.)[0]
            initial = _relax_composition(fixed_temperature, initial)
            initial = primal(fixed_temperature, initial)
            def step(index, current):
                return primal(lambda flat: evaluate(flat, (index+1)/continuation_steps)[0], current)
            return jax.lax.fori_loop(0, continuation_steps, step, initial)

        if self.linear_solver == "krylov":
            warm = jax.lax.stop_gradient(solve_root(residual, guess))
            root = implicit_newton_krylov(residual, warm, iterations=iterations,
                                          merit=self.newton_merit)
        else:
            root = jax.lax.custom_root(residual, guess, solve_root,
                lambda linear, rhs: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs))
        remainder, physical = evaluate(root)
        norm = jnp.max(jnp.abs(remainder))
        return dict(**physical, state=root.reshape(shape)[:, 1:], atmosphere_state=root.reshape(shape),
                    residual_norm=norm, converged=physical["valid"] & (norm < 1e-8),
                    applicable=physical["valid"] & (norm < 1e-8) & physical["reference_domain_valid"])

    @partial(jax.jit, static_argnums=0)
    def spectrum_from_solution(self, solution, *, rotation_kms=0., zonal_kms=0.):
        """Observe the actual solved species with their own continuum coefficients."""
        frequency = jnp.asarray(self.observer.wavenumber_cm)*LIGHT_SPEED_CGS
        chi, eta = self.continuum_coefficients_at(solution["temperature_k"], solution, frequency)
        contrast = self.observer.spectrum_from_profiles(
            solution["bound_cm3"][::-1], solution["temperature_k"][::-1],
            solution["radial_velocity_kms"][::-1], rotation_kms=rotation_kms, zonal_kms=zonal_kms,
            continuum_extinction=chi[::-1], continuum_emission=eta[::-1]*LIGHT_SPEED_CGS)
        return jnp.where(solution["converged"] & solution.get("reference_domain_valid", True), contrast, jnp.nan)


__all__ = ["SolarCompositionModel", "SolarAtmosphereModel", "SOLAR_ABUNDANCE", "HELIUM_ABUNDANCE"]
