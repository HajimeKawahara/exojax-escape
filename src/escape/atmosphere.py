"""Pure-H radiation/SE, thermal balance and transonic pressure feedback.

The selected closure follows local statistical equilibrium (no population
advection), with thermal advection and expansion. It is a reduced native
model; Cloudy composition and resonance redistribution require comparisons.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .atmosphere_wind import thermal_flow_terms, wind_from_profiles
from .coupled_forward import CoupledHydrogenForwardModel
from .hydrogen_atom import atom_saha_factors
from .hydrogen_forward import PLANET_RADIUS_CM
from .hydrogen_nlte import _bound_ratios
from .parker import PROTON_MASS_CGS
from .reactive_flow import _newton


class HydrogenAtmosphereModel(CoupledHydrogenForwardModel):
    """One implicitly differentiated atmosphere-to-observer entry point.

    Mass loss sets density through a transonic tidal wind, whose pressure
    uses the solved T/mu. Local SE and internal radiation set the pure-H
    ionization and mu=1/(1+x). Temperature solves kinetic heating/cooling,
    thermal advection and expansion, except in the bottom cell, where the
    supplied base temperature is a boundary condition. The finite atom's
    temperature validity limits remain enforced; there is no clipped root.

    The computational gas base is not the paper's unprovided dense 4500 K
    extrapolation. Constant continuation of T/mu beyond the radiation grid
    selects the wind critical branch; its effect needs its own convergence
    check. Spherical continuity is combined with plane-parallel irradiation.
    """

    def __init__(self, *, photosphere_layers=0, **kwargs):
        # The explicit C23 low-temperature law extends the finite atom to the
        # 4000 K ionization-fit boundary; it is extrapolation, not new data.
        kwargs.setdefault("low_temperature_extrapolation", "cloudy")
        kwargs.setdefault("stellar_beam", True)
        super().__init__(**kwargs)
        if (not isinstance(photosphere_layers, int) or isinstance(photosphere_layers, bool)
                or not 0 <= photosphere_layers < len(self.radius_rp) - 1):
            raise ValueError("photosphere_layers must leave at least two atmospheric cells.")
        self.photosphere_layers = photosphere_layers

    @partial(jax.jit, static_argnums=0, static_argnames=("iterations", "continuation_steps"))
    def solve(
        self, mass_loss_g_s, temperature_initial_k, inner_intensity, outer_intensity,
        *, iterations=32, continuation_steps=8, initial_state=None,
    ):
        """Return atmosphere arrays in observer order (outer to inner).

        temperature_initial_k is an array in that same order. It initializes
        the root, except for its bottom value which fixes the thermal boundary.
        If photosphere_layers>0, those bottom cells retain their supplied T
        and Saha/Boltzmann populations. This explicitly prescribed dense LTE
        patch does not solve their thermal balance; its join must be validated.
        The layer count is static, not a moving pressure threshold during AD.
        Changing the initial interior profile must not change a converged root
        on the same branch. All incoming spectra use the radiation grid and
        I_nu units documented by CoupledHydrogenForwardModel.
        """
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (iterations, continuation_steps)):
            raise ValueError("Iteration counts must be positive static integers.")
        initial_temperature = jnp.asarray(temperature_initial_k)[::-1]
        if initial_temperature.shape != self.radius_rp.shape or len(self.radius_rp) < 3:
            raise ValueError("Use a temperature profile on at least three atmospheric layers.")
        radius = self.radius_rp[::-1] * PLANET_RADIUS_CM
        radiation_model = self.radiation

        def wind(temperature, ionized):
            return wind_from_profiles(
                radius, temperature, 1 / (1 + ionized), mass_loss_g_s,
                self.observer.planet_mass_mj,
                base_radius_cm=PLANET_RADIUS_CM, outer_radius_cm=32 * PLANET_RADIUS_CM,
                tidal_coefficient_s2=self.observer.tidal_coefficient_s2,
                ngrid=self.observer.wind_ngrid, return_info=True,
            )

        shape = (len(radius), len(radiation_model.atom.statistical_weights) + 1)
        if initial_state is not None:
            if jnp.shape(initial_state) != shape:
                raise ValueError("initial_state must match the atmosphere log-coordinate grid.")
            initial = jnp.asarray(initial_state)
        else:
            initial_wind, _ = wind(initial_temperature, jnp.ones_like(radius))
            initial_density = initial_wind.density_g_cm3 / PROTON_MASS_CGS
            initial_f = radiation_model._initial_fractions(
                initial_density, initial_temperature, inner_intensity, outer_intensity,
            )
            if self.photosphere_layers:
                saha = jax.vmap(lambda t: atom_saha_factors(radiation_model.atom, t))(initial_temperature)
                ionized_lte = 2 / (1 + jnp.sqrt(1 + 4 * initial_density * saha.sum(1)))
                lte_f = initial_density[:, None] * ionized_lte[:, None]**2 * saha
                initial_f = jnp.where(jnp.arange(len(radius))[:, None] < self.photosphere_layers,
                                      lte_f, initial_f)
            initial = jnp.concatenate((jnp.zeros((len(radius), 1)),
                                       jnp.log(initial_f / (1 - initial_f.sum(1))[:, None])), axis=1)

        def evaluate(flat, blend=1., thermal_blend=1.):
            values = flat.reshape(shape)
            temperature = initial_temperature * jnp.exp(values[:, 0])
            fractions = jax.nn.softmax(jnp.concatenate(
                (values[:, 1:], jnp.zeros((len(radius), 1))), axis=1,
            ), axis=1)
            bound_f, ionized = fractions[:, :-1], fractions[:, -1]
            flow, wind_info = wind(temperature, ionized)
            density = flow.density_g_cm3 / PROTON_MASS_CGS
            electron = density * ionized
            radiation = radiation_model.radiation_state(
                density, temperature, bound_f, inner_intensity, outer_intensity,
                coupling=blend, ionized_fraction=ionized,
            )
            target = jax.vmap(_bound_ratios)(electron, radiation.rates)
            balance = values[:, 1:] - jnp.log(target)
            if self.photosphere_layers:
                saha = jax.vmap(lambda t: atom_saha_factors(radiation_model.atom, t))(temperature)
                lte_balance = values[:, 1:] - jnp.log(electron[:, None] * saha)
                balance = jnp.where(jnp.arange(len(radius))[:, None] < self.photosphere_layers,
                                     lte_balance, balance)
            terms = thermal_flow_terms(radius, flow.density_g_cm3, temperature,
                                        1 / (1 + ionized), flow.velocity_kms * 1e5)
            heating, cooling = (radiation.thermal_heating_erg_cm3_s,
                                radiation.thermal_cooling_erg_cm3_s)
            scale = heating + cooling + jnp.abs(terms.advection) + jnp.abs(terms.expansion)
            thermal = (radiation.thermal_net_erg_cm3_s + terms.advection + terms.expansion) / scale
            thermal = thermal_blend * thermal + (1 - thermal_blend) * values[:, 0]
            thermal = jnp.where(jnp.arange(len(radius)) < max(self.photosphere_layers, 1),
                                 values[:, 0], thermal)
            residual = jnp.concatenate((thermal[:, None], balance), axis=1).ravel()
            return residual, temperature, density, bound_f, ionized, flow, wind_info, radiation, terms

        residual = lambda state: evaluate(state)[0]

        def solve_root(function, guess):
            del function
            if continuation_steps == 1:
                return _newton(residual, guess, iterations)

            def radiation_step(index, values):
                opacity = 10.**(-8 * (1 - (index + 1) / continuation_steps))
                return _newton(lambda state: evaluate(state, opacity, 0.)[0], values, iterations)

            def thermal_step(index, values):
                blend = (index + 1) / continuation_steps
                return _newton(lambda state: evaluate(state, 1., blend)[0], values, iterations)

            initial_radiation = jax.lax.fori_loop(0, continuation_steps, radiation_step, guess)
            return jax.lax.fori_loop(0, continuation_steps, thermal_step, initial_radiation)

        root = jax.lax.custom_root(
            residual, initial.ravel(), solve_root,
            lambda linear, rhs: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs),
        )
        result, temperature, density, fractions, ionized, flow, wind_info, radiation, terms = evaluate(root)
        norm = jnp.max(jnp.abs(result))
        converged = (jnp.isfinite(norm) & (norm < 1e-8) & radiation.converged
                     & jnp.all(jnp.isfinite(flow.velocity_kms)) & (mass_loss_g_s > 0))
        profile = lambda value: jnp.where(converged, value[::-1], jnp.nan)
        bound = density[:, None] * fractions
        return {
            "temperature_k": profile(temperature), "n_h_cm3": profile(density),
            "bound_cm3": profile(bound), "electron_cm3": profile(density * ionized),
            "proton_cm3": profile(density * ionized), "ionized_fraction": profile(ionized),
            "mean_molecular_weight": profile(1 / (1 + ionized)),
            "pressure_bar": profile(density * (1 + ionized) * 1.380649e-16 * temperature / 1e6),
            "photosphere_mask": jnp.arange(len(radius))[::-1] < self.photosphere_layers,
            "radial_velocity_kms": profile(flow.velocity_kms),
            "n2_cm3": profile(bound[:, 1:3].sum(1)), "n3_cm3": profile(bound[:, 3:6].sum(1)),
            "mean_intensity": profile(radiation.mean_intensity),
            "thermal_heating": radiation.thermal_heating_erg_cm3_s[::-1],
            "thermal_cooling": radiation.thermal_cooling_erg_cm3_s[::-1],
            "thermal_advection": terms.advection[::-1], "expansion": terms.expansion[::-1],
            "radiation_balance_error": radiation.radiation_balance_error,
            "residual_norm": norm, "converged": converged, "wind_info": wind_info,
            "state": root.reshape(shape),
        }

    @partial(jax.jit, static_argnums=0, static_argnames=(
        "iterations", "continuation_steps", "include_continuum", "include_hminus",
    ))
    def spectrum(
        self, mass_loss_g_s, temperature_initial_k, inner_intensity, outer_intensity,
        *, rotation_kms=0., zonal_kms=0., iterations=32, continuation_steps=8,
        include_continuum=True, include_hminus=None, initial_state=None,
    ):
        """Generate H-alpha from the joint atmospheric root, with implicit AD.

        H-minus is selected at construction for both internal thermal/radiation
        balance and the observer. A different observer-only setting is rejected.
        """
        if include_hminus is None:
            include_hminus = self.radiation.include_hminus
        elif include_hminus != self.radiation.include_hminus:
            raise ValueError("Select H-minus at construction for both atmosphere and observer.")
        solution = self.solve(mass_loss_g_s, temperature_initial_k, inner_intensity,
                               outer_intensity, iterations=iterations, continuation_steps=continuation_steps,
                               initial_state=initial_state)
        contrast = self.observer.spectrum_from_profiles(
            solution["bound_cm3"], solution["temperature_k"], solution["radial_velocity_kms"],
            electron_cm3=solution["electron_cm3"], proton_cm3=solution["proton_cm3"],
            rotation_kms=rotation_kms, zonal_kms=zonal_kms,
            include_continuum=include_continuum, include_hminus=include_hminus,
        )
        return jnp.where(solution["converged"], contrast, jnp.nan)


__all__ = ["HydrogenAtmosphereModel"]
