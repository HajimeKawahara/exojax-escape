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
from .hydrogen_atom import atom_cross_sections, atom_milne_population, atom_saha_factors
from .hydrogen_escape_probability import HydrogenEscapeProbabilityModel, _column_above
from .hydrogen_forward import PLANET_MASS_MJ, PLANET_RADIUS_CM
from .hydrogen_molecules import (
    HMINUS_THRESHOLD_RYD, MolecularPhotoRates, RYDBERG_ERG, SAHA, attachment_cooling,
    h2_small_photo_rates, h2_small_thermal_exchange,
    molecular_continuum_coefficients, molecular_photo_rates,
    molecular_reaction_balance,
)
from .hydrogen_nlte import _bound_ratios
from .hydrogen_rates import BOLTZMANN_CGS, LIGHT_SPEED_CGS, PLANCK_CGS
from .metal_network import ATOMS, MetalNetwork, metal_frequency_edges
from .nlte_forward import HydrogenNLTEForwardModel
from .nonlinear import implicit_newton_krylov, newton_krylov
from .parker import PROTON_MASS_CGS
from .reactive_flow import _newton

# Explicit retained subset of C23 default.abn, not a metallicity calibration.
SOLAR_ABUNDANCE = {6: 2.45e-4, 7: 8.51e-5, 8: 4.90e-4, 12: 3.47e-5,
                   13: 2.95e-6, 14: 3.47e-5, 20: 2.29e-6, 25: 2.88e-7, 26: 2.82e-5}
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
                 linear_solver="krylov"):
        if linear_solver not in ("dense", "krylov"):
            raise ValueError("Choose dense or krylov Newton linear solves.")
        self.linear_solver = linear_solver
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
            line_optical_depth_scale=line_optical_depth_scale,
        )
        h = self.hydrogen
        self.grid, self.widths_cm, self.cells = h.grid, h.widths_cm, h.cells
        self.radius_cm = (h.edges_cm[:-1]+h.edges_cm[1:])/2
        self.metal = MetalNetwork(self.grid.frequency_hz, self.grid.weight_hz, self.widths_cm,
                                 atoms=atoms, line_optical_depth_scale=line_optical_depth_scale)
        self.helium = GroundHelium(self.grid.frequency_hz, self.grid.weight_hz)
        self.free_free = FreeFreeContinuum()
        self.abundance = jnp.array([SOLAR_ABUNDANCE[z] for z in self.metal.atomic_number])
        self.mass_per_h = ATOMIC_MASS_G*(1.00794+4.002602*HELIUM_ABUNDANCE
                                        +sum(m*SOLAR_ABUNDANCE[z] for z, m in self.metal.elements))
        self.hydrogen_size = len(h.atom.statistical_weights)
        self.helium_slice = slice(self.hydrogen_size+4, self.hydrogen_size+6)
        self.metal_slice = slice(self.hydrogen_size+6, None)
        self.state_size = self.hydrogen_size+6+self.metal.state_size
        self.molecule_nuclei = jnp.array([1., 2., 2., 2.])

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

    def evaluate(self, state, density_cm3, temperature_k, incident_mean_intensity, *,
                 inner_intensity=0., hydrogen_line_intensity=None, metal_line_intensity=None):
        """Return the coupled fixed point, all species and a non-duplicated heat ledger."""
        h, m, grid = self.hydrogen, self.metal, self.grid
        t, density = jnp.asarray(temperature_k), jnp.asarray(density_cm3)
        physical = self.unpack(state, density)
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
        radiation = h.population_state(t, bound, proton, ne, incident_mean_intensity,
            additional_opacity=extra_chi, additional_emission=extra_eta,
            additional_line_opacity=hline_extra, inner_intensity=inner_intensity,
            incident_line_mean_intensity=hydrogen_line_intensity)
        mline_chi, _ = self.continuum_coefficients_at(t, physical, m.line_frequency_hz)
        diffuse = h._forward_diffuse(radiation["transport_extinction_cm1"], radiation["continuum_emission"])
        mline_diffuse = jax.vmap(lambda row: jnp.interp(m.line_frequency_hz, grid.frequency_hz, row))(diffuse)
        chemical_rec = jnp.zeros((self.cells, len(m.elements), 2))
        chemical_rec = chemical_rec.at[:, :, 0].set((molecules[:, 0]*4e-6/jnp.sqrt(t))[:, None])
        metal = m.evaluate(t, state[:, self.metal_slice], physical["element_density_cm3"],
            ne, bound[:, 0], proton, helium[:, 0], helium[:, 1], radiation, h.atom.line_frequency_hz,
            additional_recombination_s=chemical_rec, line_incident_mean_intensity=metal_line_intensity,
            line_continuum_opacity_cm1=mline_chi, line_diffuse_mean_intensity=mline_diffuse,
            line_scattering_cm1=h.scattering_coefficients_at(bound, ne, m.line_frequency_hz),
            hydrogen_neutral_cm3=neutral,
            additional_collider_density_cm3={"H2": molecules[:, 2]+molecules[:, 3]})
        ots_frequency = jnp.concatenate((h.atom.line_frequency_hz, m.line_frequency_hz))
        ots_flux = jnp.concatenate((radiation["ots_photon_flux_cm2_s"], metal["ots_photon_flux_cm2_s"]), axis=1)
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
        sigma = atom_cross_sections(h.atom, m.line_frequency_hz)
        inverse = jax.vmap(lambda temp: atom_milne_population(h.atom, temp, m.line_frequency_hz))(t)
        metal_flux = metal["ots_photon_flux_cm2_s"]
        photo_h = metal_flux@sigma.T
        alpha_h = jnp.einsum("cf,lf,clf->cl", metal_flux, sigma, inverse)
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
        # Extra OTS photons are allocated to each absorbing species exactly once.
        excess = jnp.maximum(PLANCK_CGS*m.line_frequency_hz[None, :]-h.atom.ionization_energy_erg[:, None], 0.)
        h_metal_heat = jnp.sum(metal_flux[:, None, :]*sigma*(bound[:, :, None]-(ne*proton)[:, None, None]*inverse)*excess, axis=(1, 2))
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
        ledger = dict(hydrogen=radiation["thermal_net_erg_cm3_s"]+h_metal_heat,
                      metals=metal["energy_erg_cm3_s"]["net_kinetic_heat"],
                      helium=he["thermal_net"], free_free=ff_net,
                      molecular=molecule_net+molecular_h2)
        net = sum(ledger.values())
        # Scale by real channels, not the much larger trapped radiative line cycles.
        scale = radiation["thermal_heating_erg_cm3_s"]+radiation["thermal_cooling_erg_cm3_s"]
        scale += sum(jnp.abs(value) for value in ledger.values())
        valid = (radiation["converged"] & metal["valid"] & he["valid"]
                 & jnp.all(ne > 0) & jnp.all((t >= 4000) & (t <= 14000))
                 & jnp.all(jnp.isfinite(target)))
        return target, dict(**physical, temperature_k=t, n_h_cm3=density,
            thermal_net=net, thermal_scale=scale, thermal_ledger=ledger,
            radiation=radiation, metal=metal, helium=he, molecule_balance=molecular,
            valid=valid)

    @partial(jax.jit, static_argnums=0, static_argnames=("iterations",))
    def solve(self, density_cm3, temperature_k, incident_mean_intensity, initial_state, *,
              inner_intensity=0., hydrogen_line_intensity=None, metal_line_intensity=None,
              iterations=30):
        """Fixed-profile coupled root with implicit derivatives of its residual."""
        shape = (self.cells, self.state_size)
        if initial_state.shape != shape:
            raise ValueError("initial_state must match the composition grid.")
        def evaluate(flat):
            return self.evaluate(flat.reshape(shape), density_cm3, temperature_k, incident_mean_intensity,
                inner_intensity=inner_intensity, hydrogen_line_intensity=hydrogen_line_intensity,
                metal_line_intensity=metal_line_intensity)
        def residual(flat):
            target, physical = evaluate(flat)
            return jnp.where(physical["valid"], (flat.reshape(shape)-target).ravel(), jnp.nan)
        initial = jax.lax.stop_gradient(_relax_composition(residual, initial_state.ravel()))
        if self.linear_solver == "krylov":
            root = implicit_newton_krylov(residual, initial, iterations=iterations)
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
    boundary. Results are ordered inner-to-outer, unlike the observer arrays.
    """

    def __init__(self, radius_edges_rp, *, planet_mass_mj=PLANET_MASS_MJ,
                 tidal_coefficient_s2=0., wind_ngrid=129, observer_options=None,
                 **composition_options):
        edges = np.asarray(radius_edges_rp, dtype=float)
        if edges.ndim != 1 or len(edges) < 4 or edges[0] < 1.:
            raise ValueError("Use at least three atmospheric cells outside one planet radius.")
        super().__init__(edges*PLANET_RADIUS_CM, **composition_options)
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
        wind, info = wind_from_profiles(self.radius_cm, temperature, mu, mass_loss_g_s,
            self.planet_mass_mj, base_radius_cm=PLANET_RADIUS_CM,
            outer_radius_cm=32*PLANET_RADIUS_CM, tidal_coefficient_s2=self.tidal_coefficient_s2,
            ngrid=self.wind_ngrid, return_info=True)
        return wind, info

    def evaluate_atmosphere(self, flat, reference_t, mass_loss_g_s, incident_mean_intensity, *,
                            inner_intensity=0., hydrogen_line_intensity=None,
                            metal_line_intensity=None, thermal_blend=1., flow_energy=True):
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
            metal_line_intensity=metal_line_intensity)
        terms = thermal_flow_terms(self.radius_cm, wind.density_g_cm3, temperature,
                                    physical["mean_molecular_weight"], wind.velocity_kms*1e5)
        flow = terms.advection+terms.expansion if flow_energy else jnp.zeros(self.cells)
        scale = physical["thermal_scale"]+jnp.abs(terms.advection)+jnp.abs(terms.expansion)
        # Cooling minus heating has the same sign as a positive temperature
        # displacement on a stable thermal branch; this avoids reversing the
        # continuation Jacobian merely because of a residual sign convention.
        thermal = -thermal_blend*(physical["thermal_net"]+flow)/scale+(1-thermal_blend)*values[:, 0]
        thermal = thermal.at[0].set(values[0, 0])
        residual = jnp.column_stack((thermal, composition-target)).ravel()
        valid = physical["valid"] & wind_info["valid"] & jnp.all(jnp.isfinite(thermal))
        result = dict(**physical, radial_velocity_kms=wind.velocity_kms,
                      thermal_advection=terms.advection, thermal_expansion=terms.expansion,
                      wind_info=wind_info, flow_energy_included=flow_energy)
        return jnp.where(valid, residual, jnp.nan), result


    @partial(jax.jit, static_argnums=0, static_argnames=("iterations", "continuation_steps", "flow_energy"))
    def solve_atmosphere(self, mass_loss_g_s, temperature_initial_k,
                         incident_mean_intensity, initial_state, *, inner_intensity=0.,
                         hydrogen_line_intensity=None, metal_line_intensity=None,
                         iterations=30, continuation_steps=8, flow_energy=True):
        """Solve temperatures and all populations, then implicitly differentiate.

        initial_state is a composition seed on this same radial grid. The
        supplied T profile initializes the branch; only its bottom value is
        fixed. flow_energy=False selects radiative equilibrium as a declared
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
                flow_energy=flow_energy)

        residual = lambda flat: evaluate(flat)[0]

        def solve_root(function, initial):
            del function
            def primal(fn, value):
                if self.linear_solver == "krylov":
                    return newton_krylov(fn, value, iterations=iterations).value
                return _newton(fn, value, iterations)
            fixed_temperature = lambda flat: evaluate(flat, 0.)[0]
            initial = _relax_composition(fixed_temperature, initial)
            initial = primal(fixed_temperature, initial)
            def step(index, current):
                return primal(lambda flat: evaluate(flat, (index+1)/continuation_steps)[0], current)
            return jax.lax.fori_loop(0, continuation_steps, step, initial)

        if self.linear_solver == "krylov":
            warm = jax.lax.stop_gradient(solve_root(residual, guess))
            root = implicit_newton_krylov(residual, warm, iterations=iterations)
        else:
            root = jax.lax.custom_root(residual, guess, solve_root,
                lambda linear, rhs: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs))
        remainder, physical = evaluate(root)
        norm = jnp.max(jnp.abs(remainder))
        return dict(**physical, state=root.reshape(shape)[:, 1:], atmosphere_state=root.reshape(shape),
                    residual_norm=norm, converged=physical["valid"] & (norm < 1e-8))

    @partial(jax.jit, static_argnums=0)
    def spectrum_from_solution(self, solution, *, rotation_kms=0., zonal_kms=0.):
        """Observe the actual solved species with their own continuum coefficients."""
        frequency = jnp.asarray(self.observer.wavenumber_cm)*LIGHT_SPEED_CGS
        chi, eta = self.continuum_coefficients_at(solution["temperature_k"], solution, frequency)
        contrast = self.observer.spectrum_from_profiles(
            solution["bound_cm3"][::-1], solution["temperature_k"][::-1],
            solution["radial_velocity_kms"][::-1], rotation_kms=rotation_kms, zonal_kms=zonal_kms,
            continuum_extinction=chi[::-1], continuum_emission=eta[::-1]*LIGHT_SPEED_CGS)
        return jnp.where(solution["converged"], contrast, jnp.nan)


__all__ = ["SolarCompositionModel", "SolarAtmosphereModel", "SOLAR_ABUNDANCE", "HELIUM_ABUNDANCE"]
