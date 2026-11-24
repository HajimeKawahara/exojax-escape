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

from exojax.database.core.broadening import doppler_sigma
from exojax.opacity.lpf.line_coefficients import line_coefficients
from exojax.opacity.lpf.lpf import vvoigt
from exojax.utils.constants import c

from .hydrogen_atom import (
    DEFAULT_PRINCIPAL_N, atom_natural_width_hz, make_hydrogen_atom,
    solve_atom_populations,
)
from .hydrogen_collisions import validate_shell_collision_model
from .hydrogen_continuum import continuum_coefficients
from .hydrogen_forward import HydrogenForwardModel, PLANET_RADIUS_CM
from .hydrogen_populations import hydrogen_nuclei_density
from .hydrogen_rates import (
    EV_CGS, LIGHT_SPEED_CGS, LINE_EINSTEIN_A_S, LINE_LOWER, LINE_UPPER,
    MAX_TEMPERATURE_K, MIN_TEMPERATURE_K, PLANCK_CGS, STATISTICAL_WEIGHTS,
    planck_occupation,
)


def _nlte_temperature(radius_cm, parameters):
    base, top = parameters
    return base + (top - base) * (
        1 - jnp.exp(-(radius_cm / PLANET_RADIUS_CM - 1) / 0.12)
    )


def halpha_coefficients(bound_cm3, line_center_cm, *, sum_branches=True):
    """Sum 3s->2p, 3p->2s and 3d->2p with their actual populations.

    Returns integrated extinction and emissivity, each (..., 1), using the
    ExoJAX wavenumber convention. sum_branches=False returns separate weights
    with shape (..., 3), for use with distinct branch profiles. A shared NIST
    center approximates the unresolved branches. Summed shell n2/n3 are not
    used to assign statistical sublevel populations. The omitted branch-center
    span is about 5.3 km/s; precise centroids are outside scope.
    """
    lower, upper = LINE_LOWER[2:], LINE_UPPER[2:]
    bound = jnp.asarray(bound_cm3)
    profiles = jnp.ones((3, 1)) if sum_branches else jnp.eye(3)
    return line_coefficients(
        profiles, jnp.full((3,), line_center_cm),
        jnp.asarray(LINE_EINSTEIN_A_S[2:]),
        jnp.asarray(STATISTICAL_WEIGHTS[lower]),
        jnp.asarray(STATISTICAL_WEIGHTS[upper]), bound[..., lower], bound[..., upper],
    )


class HydrogenNLTEForwardModel(HydrogenForwardModel):
    """Extended local SE on a prescribed Parker atmosphere, with implicit AD.

    The default atom resolves 1s, 2s, 2p, 3s, 3p and 3d, and adds statistical
    shells through nmax=15. nmax=3..20 permits finite-atom comparisons; the
    small atoms are not assumed to be converged. Electron-density feedback in
    orbital mixing remains inside the implicitly differentiated charge root.
    continuum_points is the Gauss-Legendre order in each interval between
    ionization thresholds, with an upper photon energy of 100 eV. Its default
    of 48 follows the extended atom's continuum convergence checks. Atomic
    data and quadrature arrays are constructed once, outside traced calls.
    Shell collisions default to Anderson data for n=4,5 and VS80 above n=5;
    shell_collision_model="vs80" retains the earlier all-VS80 shell choice.

    The default gas base temperature is 6000 K to stay within the atomic-data
    domain (5802--14000 K). The continuum lower boundary remains 4500 K. The
    nonisothermal wind uses this gas profile with externally fixed mu; inferred
    ionization does not feed back into wind pressure. The default irradiation
    is an unattenuated diluted 10000 K blackbody. No excited or ionized fraction
    is fitted. Require positive mass loss and dilution, and float64 JAX.
    Only the NIST center is supported; the legacy Rydberg experiment does not
    adjust this atom's branch rates. Each H-alpha branch has its own natural
    width from the full upper and lower lifetimes. Stark and other pressure
    broadening are omitted; these remain a separate physical limitation.
    """

    def __init__(
        self, *, nlayer=128, temperature_base_k=6000.0, temperature_top_k=10000.0,
        nmax=DEFAULT_PRINCIPAL_N, resolved_nmax=3, continuum_points=48,
        wind_model="nonisothermal", shell_collision_model="anderson_n45", **kwargs,
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
        self.atom = make_hydrogen_atom(nmax, resolved_nmax=resolved_nmax)
        self.nmax = nmax
        validate_shell_collision_model(shell_collision_model, resolved_nmax=self.atom.resolved_nmax)
        self.shell_collision_model = shell_collision_model
        self.continuum_points = continuum_points
        thresholds = np.asarray(self.atom.ionization_energy_erg) / PLANCK_CGS
        edges = np.unique(np.r_[thresholds, 100 * EV_CGS / PLANCK_CGS])
        nodes, weights = np.polynomial.legendre.leggauss(continuum_points)
        widths = np.diff(np.log(edges))[:, None]
        frequency = edges[:-1, None] * np.exp(widths * (nodes + 1) / 2)
        self.continuum_frequency_hz = jnp.asarray(frequency.ravel())
        self.continuum_weight_hz = jnp.asarray((frequency * widths * weights / 2).ravel())
        super().__init__(nlayer=nlayer, temperature_top_k=temperature_top_k,
                         wind_model=wind_model, **kwargs)
        self.halpha_natural_width_cm = atom_natural_width_hz(self.atom)[2:5] / LIGHT_SPEED_CGS
        self.halpha_centers_cm = jnp.full((3,), self.line_center_cm)
        self.halpha_rest_detuning_cm = jnp.asarray(self.wavenumber_cm - self.line_center_cm)

    def _line_profiles(self, temperature_k, velocity_kms):
        """Return three thermal/lifetime Voigt profiles in the ExoJAX convention.

        Positive velocity is receding. Profiles integrate to one over infinite
        wavenumber; a finite observer window does not renormalize their wings.
        """
        sigma = doppler_sigma(self.halpha_centers_cm, temperature_k,
                              self.adb.line_masses[0])
        detuning = (self.halpha_rest_detuning_cm[None, :]
                    + self.halpha_centers_cm[:, None] * velocity_kms / c)
        return vvoigt(detuning, sigma, self.halpha_natural_width_cm)

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
        line_occupation=None, continuum_occupation=None,
    ):
        """Return absolute populations, charge balance and local diagnostics.

        All densities are cm^-3. mean_molecular_weight controls only the wind;
        mean_molecular_weight_nlte is the diagnostic value 1/(1+np/nH).
        n3_cm3 includes only 3s, 3p and 3d, not the added higher shells.
        The H-alpha extinction integral has units cm^-2; its sign is reported
        separately from convergence of the population solve. positive_extinction
        describes the integrated weight; transfer checks the frequency-dependent
        sum of the three branch profiles independently.
        A failed/invalid local solve remains visible through NaNs and flags.

        Supply both occupation arrays to replace the diluted blackbody. They
        are dimensionless photon occupations, already diluted, on
        atom.line_frequency_hz and continuum_frequency_hz respectively. Each
        must be one-dimensional; the same field illuminates every layer.
        radiation_temperature_k and radiation_dilution are then unused.
        Spectral interpolation and any line-profile averaging belong to the
        caller; no internal attenuation or diffuse field is computed here.
        """
        if (line_occupation is None) != (continuum_occupation is None):
            raise ValueError("Supply both line_occupation and continuum_occupation.")
        if line_occupation is None:
            line_occupation = planck_occupation(
                self.atom.line_frequency_hz, radiation_temperature_k, radiation_dilution,
            )
            continuum_occupation = planck_occupation(
                self.continuum_frequency_hz, radiation_temperature_k, radiation_dilution,
            )
        else:
            line_occupation = jnp.asarray(line_occupation)
            continuum_occupation = jnp.asarray(continuum_occupation)
            if line_occupation.shape != self.atom.line_frequency_hz.shape:
                raise ValueError("line_occupation must match atom.line_frequency_hz.")
            if continuum_occupation.shape != self.continuum_frequency_hz.shape:
                raise ValueError("continuum_occupation must match continuum_frequency_hz.")
        temperature = self.temperature_profile(temperature_top_k)
        velocity = self.radial_velocity(
            wind_temperature_k, mean_molecular_weight,
            temperature_top_k=temperature_top_k,
        )
        radius = self.radius_rp * PLANET_RADIUS_CM
        density = hydrogen_nuclei_density(radius, velocity, mass_loss_g_s)
        solution = jax.vmap(lambda nh, t: solve_atom_populations(
            nh, self.atom, t, line_occupation, continuum_occupation,
            self.continuum_frequency_hz, self.continuum_weight_hz,
            shell_collision_model=self.shell_collision_model,
        ))(density, temperature)
        bound = solution.bound_cm3
        n2, n3 = jnp.sum(bound[:, 1:3], axis=-1), jnp.sum(bound[:, 3:6], axis=-1)
        alpha, emission = halpha_coefficients(bound, self.line_center_cm)
        branch_alpha, branch_emission = halpha_coefficients(
            bound, self.line_center_cm, sum_branches=False,
        )
        positive_extinction = jnp.isfinite(alpha[:, 0]) & (alpha[:, 0] > 0)
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
            "halpha_extinction_integral": alpha,
            "halpha_emissivity_integral": emission,
            "halpha_branch_extinction_integral": branch_alpha,
            "halpha_branch_emissivity_integral": branch_emission,
            "positive_extinction": positive_extinction,
        }

    @partial(jax.jit, static_argnums=0)
    def spectrum(
        self, *, mass_loss_g_s=1e13, radiation_dilution=0.025,
        radiation_temperature_k=10000.0, mean_molecular_weight=0.5,
        wind_temperature_k=10000.0, temperature_top_k=None,
        rotation_kms=0.0, zonal_kms=0.0,
        line_occupation=None, continuum_occupation=None,
    ):
        """Return H-alpha contrast from branch-weighted NLTE populations.

        Each branch uses its thermal and full-lifetime width. The existing
        formal solver is applied to their summed frequency-dependent extinction
        and emissivity. Invalid SE or unsupported transfer coefficients return
        NaN; no population, opacity or derivative is silently clipped.
        Optional occupation arrays follow population_profiles: both are
        required and replace the blackbody inputs without further dilution.
        """
        profiles = self.population_profiles(
            mass_loss_g_s=mass_loss_g_s, radiation_dilution=radiation_dilution,
            radiation_temperature_k=radiation_temperature_k,
            mean_molecular_weight=mean_molecular_weight,
            wind_temperature_k=wind_temperature_k, temperature_top_k=temperature_top_k,
            line_occupation=line_occupation, continuum_occupation=continuum_occupation,
        )
        contrast = self._transfer_line_coefficients(
            profiles["halpha_branch_extinction_integral"],
            profiles["halpha_branch_emissivity_integral"],
            profiles["radial_velocity_kms"], rotation_kms,
            zonal_kms, profiles["temperature_k"],
        )
        return jnp.where(jnp.all(profiles["converged"]), contrast, jnp.nan)

    @partial(jax.jit, static_argnums=0, static_argnames=("include_continuum", "include_hminus"))
    def spectrum_from_profiles(
        self, bound_cm3, temperature_k, radial_velocity_kms, *,
        electron_cm3=None, proton_cm3=None, rotation_kms=0., zonal_kms=0.,
        include_continuum=False, include_hminus=False,
        continuum_extinction=None, continuum_emission=None,
    ):
        """Observe supplied absolute populations without solving local SE again.

        All profiles run from top to bottom on this instance's radial grid.
        bound_cm3 has shape (Nlayer, Nbound) for this atom; the remaining
        profiles have shape (Nlayer,). This is the common observer for coupled
        native and independent reference atmospheres. It does not establish
        that the supplied atmosphere is self-consistent.

        Optional continuum includes H bound-free/free-free and, explicitly,
        the existing checked LTE trace H-minus closure. Supply ne and np when
        enabling it. Continuum is evaluated at the observer frequencies;
        its Doppler shift across this narrow optical window is neglected.
        Emissivity transforms from per Hz to per wavenumber with a factor c.
        H-minus is not easychem chemistry and retains its temperature limits.
        Alternatively, supply both continuum_extinction [cm^-1] and
        continuum_emission [per wavenumber/sr] from the solved composition.
        This replaces the built-in continuum; include_continuum must be False.
        """
        bound = jnp.asarray(bound_cm3)
        temperature = jnp.asarray(temperature_k)
        velocity = jnp.asarray(radial_velocity_kms)
        count = len(self.radius_rp)
        if (bound.shape != (count, len(self.atom.statistical_weights))
                or temperature.shape != (count,) or velocity.shape != (count,)):
            raise ValueError("Atmosphere profiles must match the observer's layer and atom grids.")
        if include_hminus and not include_continuum:
            raise ValueError("H-minus requires include_continuum=True.")
        if (continuum_extinction is None) != (continuum_emission is None):
            raise ValueError("Supply both external continuum coefficients.")
        if continuum_extinction is not None and include_continuum:
            raise ValueError("Choose the external or built-in continuum to avoid double counting.")
        alpha, emission = halpha_coefficients(bound, self.line_center_cm, sum_branches=False)
        continuum = {}
        valid = (jnp.all(jnp.isfinite(bound) & (bound >= 0))
                 & jnp.all(jnp.isfinite(temperature) & (temperature > 0))
                 & jnp.all(jnp.isfinite(velocity)))
        if include_continuum:
            if electron_cm3 is None or proton_cm3 is None:
                raise ValueError("Continuum requires electron_cm3 and proton_cm3.")
            electron, proton = jnp.asarray(electron_cm3), jnp.asarray(proton_cm3)
            if electron.shape != (count,) or proton.shape != (count,):
                raise ValueError("Charge profiles must match the observer's layer grid.")
            valid &= jnp.all(jnp.isfinite(electron) & (electron >= 0))
            valid &= jnp.all(jnp.isfinite(proton) & (proton >= 0))
            chi, eta = jax.vmap(lambda t, n, ne, np_: continuum_coefficients(
                self.atom, t, n, ne, np_,
                jnp.asarray(self.wavenumber_cm) * LIGHT_SPEED_CGS,
                include_hminus=include_hminus,
            ))(temperature, bound, electron, proton)
            continuum = {"continuum_extinction": chi,
                         "continuum_emission": eta * LIGHT_SPEED_CGS}
        elif continuum_extinction is not None:
            continuum = {"continuum_extinction": jnp.asarray(continuum_extinction),
                         "continuum_emission": jnp.asarray(continuum_emission)}
        contrast = self._transfer_line_coefficients(
            alpha, emission, velocity, rotation_kms, zonal_kms, temperature,
            **continuum,
        )
        return jnp.where(valid, contrast, jnp.nan)
