"""Multilayer H-alpha emission using existing ExoJAX solvers.

Radial velocities and total density follow isothermal Parker flow and mass
continuity. An isolated two-level rate balance sets n3/n2; temperature, ionized
fraction, excited fraction and incident radiation remain prescribed. The
former empirical populations remain available as a comparison. Neither path
is a self-consistent hydrodynamic/multilevel NLTE atmosphere. Local columns are plane
parallel, with projected area fixed at Rp. The atmosphere has no continuum
opacity above a blackbody lower boundary; the stellar reference is also a
blackbody. Tides, nightside emission,
occultations, off-limb emission, and observational processing are omitted.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from exojax.database import AdbNist
from exojax.database.core.broadening import gamma_natural
from exojax.database.core_atom.line_strength import line_strength_atom
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.line_coefficients import line_coefficients
from exojax.postproc.response import ipgauss_ola
from exojax.rt.planck import piB
from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_intensity_surface
from exojax.utils.constants import RJ, c, hcperk
from hydrogen_populations import (
    COLLISION_STRENGTH_23, diluted_blackbody_occupation,
    hydrogen_nuclei_density, two_level_ratio,
)
from parker import parker_velocity

PLANET_RADIUS_CM = 1.9 * RJ
# Illustrative low-mass value quoted in Zhang et al. (2026), Section VI.
PLANET_MASS_MJ = 2.17
RADIUS_RATIO = 0.08
BOUNDARY_TEMPERATURE_K = 4500.0
STELLAR_TEMPERATURE_K = 10000.0
RESOLVING_POWER = 98000.0
REFERENCE_COLUMN_CM2 = 3.0e14
SYNCHRONOUS_ROTATION_KMS = 2 * np.pi * PLANET_RADIUS_CM / (1.48 * 86400) / 1e5


def _use_rydberg_center(adb):
    """Apply a local H-alpha comparison, keeping the NIST oscillator strength.

    This experiment belongs to the analysis, not the NIST database API. Run it
    only on the newly loaded single-line database before constructing OpaDirect.
    The energy gap, Einstein A and LTE strength follow the chosen center.
    """
    nu = np.full_like(adb.nu_lines, 109677.5834 * (1 / 2**2 - 1 / 3**2))
    adb._A = adb._A * (nu / adb.nu_lines)**2
    adb.nu_lines = nu
    adb._eupper = adb._elower + nu
    adb.Sij0 = line_strength_atom(
        adb._A, adb._gupper, nu, adb._elower, adb.QTref_284, adb._QTmask,
    )
    adb.generate_jnp_arrays()


def disk_quadrature(phase: float, nlon: int, nlat: int) -> tuple:
    """Return visible dayside mu, projected weight, and velocity geometry.

    Phase is in orbital cycles: 0 faces the nightside and 0.5 the dayside,
    without occultations. Weights are mu * dOmega, not normalized to unity.
    Latitude quadrature uses sin(latitude); longitude uses midpoint cells.
    """
    sin_lat, lat_weight = np.polynomial.legendre.leggauss(nlat)
    longitude = -np.pi / 2 + (np.arange(nlon) + 0.5) * np.pi / nlon
    lat, lon = np.meshgrid(np.arcsin(sin_lat), longitude, indexing="ij")
    normal = np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
        axis=-1,
    )
    tangent = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    observer = np.array([-np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase), 0])
    mu = normal @ observer
    weight = mu * lat_weight[:, None] * np.pi / nlon
    visible = mu > 0
    return tuple(jnp.asarray(a[visible]) for a in (
        mu, weight, tangent @ observer, np.cos(lat), np.sin(lon)
    ))


class HydrogenForwardModel:
    """Fixed grids with differentiable populations and Parker-wind inputs.

    Enable JAX float64 before construction. Do not mutate a constructed model;
    create another instance to change the grids or orbital phase. Wavenumbers
    are in cm^-1; velocity_kms is positive for recession relative to the
    selected H-alpha rest center, without an orbital/systemic velocity.
    temperature_top_k sets the asymptotic kinetic temperature for the instance.
    wavelength_model="nist" uses the NIST level-energy difference (default).
    "rydberg" is a local line-center experiment at fixed oscillator strength;
    it does not recreate every coefficient of the removed AdbHydrogen class.
    """

    def __init__(
        self, *, nlayer: int = 64, nlon: int = 24, nlat: int = 12,
        nnu: int = 1201, phase: float = 0.375, temperature_top_k: float = 9000.0,
        planet_mass_mj: float = PLANET_MASS_MJ, wavelength_model: str = "nist",
    ):
        if min(nlayer, nlon, nlat) < 1 or nnu < 3:
            raise ValueError("Use positive layer/disk counts and at least 3 spectral points.")
        if not np.isfinite(phase):
            raise ValueError("phase must be finite.")
        if not np.isfinite(temperature_top_k) or temperature_top_k <= 0:
            raise ValueError("temperature_top_k must be positive and finite.")
        if not np.isfinite(planet_mass_mj) or planet_mass_mj <= 0:
            raise ValueError("planet_mass_mj must be positive and finite.")
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before constructing the model.")

        if wavelength_model not in ("nist", "rydberg"):
            raise ValueError("wavelength_model must be 'nist' or 'rydberg'.")
        self.wavelength_model = wavelength_model
        self.adb = AdbNist("H_I", nurange=(15200.0, 15260.0))
        if len(self.adb.nu_lines) != 1 or not np.allclose(
            [self.adb.glower[0], self.adb.gupper[0]], [8.0, 18.0]
        ):
            raise ValueError("Expected one unresolved NIST H-alpha transition (g=8,18).")
        if wavelength_model == "rydberg":
            _use_rydberg_center(self.adb)
        self.line_center_cm = float(self.adb.nu_lines[0])
        self.wavenumber_cm = self.line_center_cm * np.exp(
            np.linspace(-120 / c, 120 / c, nnu)
        )
        self.velocity_kms = c * (self.line_center_cm / self.wavenumber_cm - 1)
        # Illustrative isolated-transition width, not a complete lifetime or
        # Stark-broadening model. NIST does not supply the total damping here.
        gamma = gamma_natural(self.adb.A)
        self.opa = OpaDirect(
            self.adb, self.wavenumber_cm, atomic_broadening=lambda T, P: gamma
        )

        edges = np.linspace(1.0, 3.0, nlayer + 1)
        # The existing RT solver expects layers ordered from top to bottom.
        self.radius_rp = jnp.asarray(((edges[:-1] + edges[1:]) / 2)[::-1].copy())
        self.thickness_cm = jnp.asarray(np.diff(edges)[::-1].copy() * PLANET_RADIUS_CM)
        self.temperature_top_k = float(temperature_top_k)
        self.planet_mass_mj = float(planet_mass_mj)
        self.temperature_k = 4500 + (self.temperature_top_k - 4500) * (
            1 - jnp.exp(-(self.radius_rp - 1) / 0.12)
        )
        shape = jnp.exp(-(self.radius_rp - 1) / 0.35)
        self.n2_reference = REFERENCE_COLUMN_CM2 * shape / jnp.sum(shape * self.thickness_cm)
        self.departure_transition = jax.nn.sigmoid((1.45 - self.radius_rp) / 0.08)
        self.geometry = disk_quadrature(phase, nlon, nlat)
        self.disk_weights = self.geometry[1]
        self.boundary_intensity = piB(BOUNDARY_TEMPERATURE_K, self.wavenumber_cm) / jnp.pi
        self.stellar_flux = piB(STELLAR_TEMPERATURE_K, self.wavenumber_cm)

        self.ip_sigma_kms = c / RESOLVING_POWER / np.sqrt(8 * np.log(2))
        velocity_step = 240 / (nnu - 1)
        half_kernel = max(1, int(np.ceil(6 * self.ip_sigma_kms / velocity_step)))
        self.kernel_velocity_kms = jnp.arange(-half_kernel, half_kernel + 1) * velocity_step

    def radial_velocity(self, wind_temperature_k=9000.0, mean_molecular_weight=0.5):
        """Return transonic Parker velocities [km/s], without rescaling.

        The constant wind temperature and mean molecular weight (in proton-mass
        units) are effective dynamics inputs, separate from the non-isothermal
        temperature used for line formation. Both must be positive. The default
        mean molecular weight represents fully ionized pure hydrogen.
        """
        return parker_velocity(
            self.radius_rp * PLANET_RADIUS_CM, wind_temperature_k,
            self.planet_mass_mj, mean_molecular_weight,
        )

    @partial(jax.jit, static_argnums=0, static_argnames=("radial_wind",))
    def prescribed_spectrum(
        self, n2_column=REFERENCE_COLUMN_CM2, departure_floor=0.08,
        wind_temperature_k=9000.0, mean_molecular_weight=0.5,
        rotation_kms=0.0, zonal_kms=0.0, *, radial_wind=True,
    ):
        """Return continuum-subtracted planet/star contrast as a fraction.

        n2_column is the nonnegative n=2 column in cm^-2, not total hydrogen.
        departure_floor is in [0, 1] and controls b3/b2 in the upper layers;
        1 gives the Boltzmann n3/n2 ratio. Positive wind_temperature_k [K] and
        mean_molecular_weight (in proton-mass units) set the Parker speed.
        radial_wind=False supplies a static comparison, not a scaled wind.
        rotation_kms is the signed equatorial speed
        at Rp (decaying as 1/r), and zonal_kms multiplies sin(longitude).
        These parameter bounds are the caller's responsibility; no clipping
        is performed, so JAX derivatives are preserved.
        """
        b_ratio = departure_floor + (1 - departure_floor) * self.departure_transition
        ratio = self.adb.gupper[0] / self.adb.glower[0] * jnp.exp(
            -hcperk * self.line_center_cm / self.temperature_k
        ) * b_ratio
        radial_velocity = (
            self.radial_velocity(wind_temperature_k, mean_molecular_weight)
            if radial_wind else jnp.zeros_like(self.radius_rp)
        )
        lower = self.n2_reference * (n2_column / REFERENCE_COLUMN_CM2)
        return self._transfer(lower, ratio, radial_velocity, rotation_kms, zonal_kms)

    @partial(jax.jit, static_argnums=0)
    def population_profiles(
        self, *, mass_loss_g_s=1e13, excited_fraction=1e-9,
        ionized_fraction=0.5, wind_temperature_k=9000.0,
        collision_strength=COLLISION_STRENGTH_23, radiation_dilution=0.025,
    ):
        """Return density, populations and velocity for the two-level closure.

        Pure hydrogen: ionized_fraction = np/nH = ne/nH sets both electron
        density and mu=1/(1+ionized_fraction). excited_fraction is the fraction
        of NEUTRAL hydrogen in the n=2 plus n=3 reservoir, not n2/nH.
        Both fractions are constant with radius, in [0,1]. Mass loss [g/s],
        collision strength and radiation dilution are nonnegative; Twind > 0.
        Callers must enforce bounds, without clipping JAX parameters.

        J = radiation_dilution * B(10000 K) is constant across the atmosphere.
        The default 0.025 approximates stellar geometric dilution at KELT-9b;
        no stellar H-alpha feature, attenuation or trapped line field is solved.
        The collision strength is frozen at kT=1 eV (see hydrogen_populations).
        Temperature remains prescribed and differs from effective Twind.
        """
        mean_molecular_weight = 1 / (1 + ionized_fraction)
        velocity = self.radial_velocity(wind_temperature_k, mean_molecular_weight)
        density = hydrogen_nuclei_density(
            self.radius_rp * PLANET_RADIUS_CM, velocity, mass_loss_g_s,
        )
        electron_density = ionized_fraction * density
        occupation = diluted_blackbody_occupation(
            self.line_center_cm, STELLAR_TEMPERATURE_K, radiation_dilution,
        )
        ratio = two_level_ratio(
            self.temperature_k, electron_density, self.line_center_cm,
            self.adb.A[0], self.adb.glower[0], self.adb.gupper[0],
            collision_strength, occupation,
        )
        excited_density = excited_fraction * (1 - ionized_fraction) * density
        n2 = excited_density / (1 + ratio)
        return {
            "n_h_cm3": density, "electron_density_cm3": electron_density,
            "n2_cm3": n2, "n3_cm3": n2 * ratio, "ratio": ratio,
            "radial_velocity_kms": velocity,
        }

    @partial(jax.jit, static_argnums=0)
    def spectrum(
        self, *, mass_loss_g_s=1e13, excited_fraction=1e-9,
        ionized_fraction=0.5, wind_temperature_k=9000.0,
        collision_strength=COLLISION_STRENGTH_23, radiation_dilution=0.025,
        rotation_kms=0.0, zonal_kms=0.0,
    ):
        """Return contrast using continuity and isolated two-level populations.

        See population_profiles for units, bounds and physical assumptions.
        Rotation and zonal velocities [km/s] have the same geometry as the
        prescribed_spectrum comparison. Mass loss and excited fraction can be
        zero; the spectrum and its parameter derivatives remain finite.
        """
        profiles = self.population_profiles(
            mass_loss_g_s=mass_loss_g_s, excited_fraction=excited_fraction,
            ionized_fraction=ionized_fraction, wind_temperature_k=wind_temperature_k,
            collision_strength=collision_strength, radiation_dilution=radiation_dilution,
        )
        return self._transfer(
            profiles["n2_cm3"], profiles["ratio"], profiles["radial_velocity_kms"],
            rotation_kms, zonal_kms,
        )

    def _transfer(self, lower_density, ratio, radial_velocity, rotation_kms, zonal_kms):
        """Share opacity, RT and disk integration between population closures."""
        # Per-lower-atom coefficients keep j/alpha finite even at zero density.
        unit_lower = jnp.ones((len(self.radius_rp), 1))
        alpha_weight, emission_weight = line_coefficients(
            jnp.ones((len(self.radius_rp), 1, 1)), self.adb.nu_lines, self.adb.A,
            self.adb.glower, self.adb.gupper, unit_lower, ratio[:, None],
        )
        source = emission_weight / alpha_weight

        def add_patch(total, patch):
            mu, weight, tangent_los, cos_lat, sin_lon = patch
            velocity = -mu * radial_velocity - tangent_los * (
                rotation_kms * cos_lat / self.radius_rp + zonal_kms * sin_lon
            )
            profiles = jax.vmap(
                lambda T, v: self.opa.line_profiles(T, velocity=v)
            )(self.temperature_k, velocity)[:, 0, :]
            dtau = (alpha_weight * profiles * self.thickness_cm[:, None]
                    * lower_density[:, None])
            intensity = rtrun_emis_pureabs_ibased_intensity_surface(
                dtau, jnp.broadcast_to(source, profiles.shape), self.boundary_intensity,
                jnp.atleast_1d(mu),
            )[0]
            return total + weight * (intensity - self.boundary_intensity), None

        excess, _ = jax.lax.scan(
            add_patch, jnp.zeros_like(self.boundary_intensity), self.geometry
        )
        # The existing OLA response also supports a single segment and handles
        # both odd/even grid lengths. The spectral window provides padding.
        broadened = ipgauss_ola(excess[None, :], self.kernel_velocity_kms, self.ip_sigma_kms)
        return RADIUS_RATIO**2 * broadened / self.stellar_flux
