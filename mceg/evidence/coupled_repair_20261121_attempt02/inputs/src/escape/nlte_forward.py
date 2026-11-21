"""Prescribed-field multilevel hydrogen populations connected to H-alpha transfer.

This opt-in model solves absolute populations and ionization locally. Temperature,
irradiation and the constant wind molecular weight remain independent inputs;
neither line trapping, continuum attenuation nor thermal balance is solved.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity.lpf.line_coefficients import line_coefficients

from .hydrogen_forward import HydrogenForwardModel, PLANET_RADIUS_CM
from .hydrogen_nlte import solve_statistical_equilibrium
from .hydrogen_populations import hydrogen_nuclei_density
from .hydrogen_rates import (
    LINE_EINSTEIN_A_S, LINE_LOWER, LINE_UPPER, MAX_TEMPERATURE_K,
    MIN_TEMPERATURE_K, STATISTICAL_WEIGHTS, blackbody_hydrogen_rates,
)


def _nlte_temperature(radius_cm, parameters):
    base, top = parameters
    return base + (top - base) * (
        1 - jnp.exp(-(radius_cm / PLANET_RADIUS_CM - 1) / 0.12)
    )


def halpha_coefficients(bound_cm3, line_center_cm):
    """Sum 3s->2p, 3p->2s and 3d->2p with their actual populations.

    Returns integrated extinction and emissivity, each (..., 1), using the
    ExoJAX wavenumber convention. A shared NIST center and normalized profile
    approximate the unresolved branches. Summed shell n2/n3 are not used to
    assign statistical sublevel populations. The omitted branch-center span
    is about 5.3 km/s; precise centroids and damping wings are outside scope.
    """
    lower, upper = LINE_LOWER[2:], LINE_UPPER[2:]
    bound = jnp.asarray(bound_cm3)
    return line_coefficients(
        jnp.ones(bound.shape[:-1] + (3, 1)), jnp.full((3,), line_center_cm),
        jnp.asarray(LINE_EINSTEIN_A_S[2:]),
        jnp.asarray(STATISTICAL_WEIGHTS[lower]),
        jnp.asarray(STATISTICAL_WEIGHTS[upper]), bound[..., lower], bound[..., upper],
    )


class HydrogenNLTEForwardModel(HydrogenForwardModel):
    """Six-level local SE on a prescribed Parker atmosphere, with implicit AD.

    The default gas base temperature is 6000 K to stay within the atomic-data
    domain (5802--14000 K). The continuum lower boundary remains 4500 K. The
    nonisothermal wind uses this gas profile with externally fixed mu; inferred
    ionization does not feed back into wind pressure. The default irradiation
    is an unattenuated diluted 10000 K blackbody. No excited or ionized fraction
    is fitted. Require positive mass loss and dilution, and float64 JAX.
    Only the NIST center is supported; the legacy Rydberg experiment does not
    adjust this atom's branch rates. The shared legacy line width omits full
    upper/lower lifetimes and pressure broadening.
    """

    def __init__(
        self, *, temperature_base_k=6000.0, temperature_top_k=10000.0,
        continuum_points=128, wind_model="nonisothermal", **kwargs,
    ):
        if kwargs.get("wavelength_model", "nist") != "nist":
            raise ValueError("The NLTE atom supports only wavelength_model='nist'.")
        if not all(np.isfinite(t) and MIN_TEMPERATURE_K <= t <= MAX_TEMPERATURE_K
                   for t in (temperature_base_k, temperature_top_k)):
            raise ValueError("NLTE gas temperatures must lie within the atomic-data domain.")
        if (not isinstance(continuum_points, int) or isinstance(continuum_points, bool)
                or continuum_points < 2):
            raise ValueError("continuum_points must be an integer of at least two.")
        self.temperature_base_k = float(temperature_base_k)
        self.continuum_points = continuum_points
        super().__init__(temperature_top_k=temperature_top_k, wind_model=wind_model, **kwargs)

    def temperature_profile(self, temperature_top_k=None):
        top = self.temperature_top_k if temperature_top_k is None else temperature_top_k
        return _nlte_temperature(
            self.radius_rp * PLANET_RADIUS_CM, (self.temperature_base_k, top),
        )

    def _wind_temperature_law(self, temperature_top_k):
        top = self.temperature_top_k if temperature_top_k is None else temperature_top_k
        return _nlte_temperature, (self.temperature_base_k, top)

    @partial(jax.jit, static_argnums=0)
    def population_profiles(
        self, *, mass_loss_g_s=1e13, radiation_dilution=0.025,
        radiation_temperature_k=10000.0, mean_molecular_weight=0.5,
        wind_temperature_k=10000.0, temperature_top_k=None,
    ):
        """Return absolute populations, charge balance and local diagnostics.

        All densities are cm^-3. mean_molecular_weight controls only the wind;
        mean_molecular_weight_nlte is the diagnostic value 1/(1+np/nH).
        A failed/invalid local solve remains visible through NaNs and flags.
        """
        temperature = self.temperature_profile(temperature_top_k)
        velocity = self.radial_velocity(
            wind_temperature_k, mean_molecular_weight,
            temperature_top_k=temperature_top_k,
        )
        radius = self.radius_rp * PLANET_RADIUS_CM
        density = hydrogen_nuclei_density(radius, velocity, mass_loss_g_s)
        rates = jax.vmap(lambda t: blackbody_hydrogen_rates(
            t, radiation_temperature_k, radiation_dilution,
            quadrature_points=self.continuum_points,
        ))(temperature)
        solution = jax.vmap(solve_statistical_equilibrium)(density, rates)
        bound = solution.bound_cm3
        n2, n3 = jnp.sum(bound[:, 1:3], axis=-1), jnp.sum(bound[:, 3:], axis=-1)
        return {
            "n_h_cm3": density, "bound_cm3": bound,
            "proton_density_cm3": solution.proton_cm3,
            "electron_density_cm3": solution.electron_cm3,
            "ionized_fraction": solution.ionized_fraction,
            "mean_molecular_weight_nlte": 1 / (1 + solution.ionized_fraction),
            "n2_cm3": n2, "n3_cm3": n3, "ratio": n3 / n2,
            "radial_velocity_kms": velocity, "temperature_k": temperature,
            "residual_norm": solution.residual_norm,
            "conservation_error": solution.conservation_error,
            "rate_residual_norm": solution.rate_residual_norm,
            "converged": solution.converged, "valid": solution.valid,
        }

    @partial(jax.jit, static_argnums=0)
    def spectrum(
        self, *, mass_loss_g_s=1e13, radiation_dilution=0.025,
        radiation_temperature_k=10000.0, mean_molecular_weight=0.5,
        wind_temperature_k=10000.0, temperature_top_k=None,
        rotation_kms=0.0, zonal_kms=0.0,
    ):
        """Return H-alpha contrast from branch-weighted NLTE populations.

        The current formal solver is used for strictly positive net extinction.
        Invalid SE or a zero/negative integrated extinction returns NaN; no
        population, temperature, opacity or derivative is silently clipped.
        """
        profiles = self.population_profiles(
            mass_loss_g_s=mass_loss_g_s, radiation_dilution=radiation_dilution,
            radiation_temperature_k=radiation_temperature_k,
            mean_molecular_weight=mean_molecular_weight,
            wind_temperature_k=wind_temperature_k, temperature_top_k=temperature_top_k,
        )
        alpha, emission = halpha_coefficients(profiles["bound_cm3"], self.line_center_cm)
        positive = jnp.isfinite(alpha) & (alpha > 0)
        source = emission / jnp.where(positive, alpha, 1.0)
        contrast = self._transfer_coefficients(
            alpha, source, profiles["radial_velocity_kms"], rotation_kms,
            zonal_kms, profiles["temperature_k"],
        )
        return jnp.where(jnp.all(profiles["converged"]) & jnp.all(positive), contrast, jnp.nan)
