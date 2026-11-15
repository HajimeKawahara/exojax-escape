"""Prescribed multilayer H-alpha emission using existing ExoJAX solvers.

Radial velocities follow an isothermal transonic Parker solution. Temperature
and level populations remain prescribed: this is not a self-consistent
hydrodynamic/statistical-equilibrium atmosphere. Local columns are plane
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

from exojax.database import AdbHydrogen
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.line_coefficients import line_coefficients
from exojax.postproc.response import ipgauss_ola
from exojax.rt.planck import piB
from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_intensity_surface
from exojax.utils.constants import RJ, c, hcperk
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
    AdbHydrogen H-alpha rest center, without an orbital/systemic velocity.
    temperature_top_k sets the asymptotic kinetic temperature for the instance.
    """

    def __init__(
        self, *, nlayer: int = 64, nlon: int = 24, nlat: int = 12,
        nnu: int = 1201, phase: float = 0.375, temperature_top_k: float = 9000.0,
        planet_mass_mj: float = PLANET_MASS_MJ,
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

        self.adb = AdbHydrogen(n_upper_max=3)
        self.line_center_cm = float(self.adb.nu_lines[0])
        self.wavenumber_cm = self.line_center_cm * np.exp(
            np.linspace(-120 / c, 120 / c, nnu)
        )
        self.velocity_kms = c * (self.line_center_cm / self.wavenumber_cm - 1)
        self.opa = OpaDirect(self.adb, self.wavenumber_cm)

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
    def spectrum(
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
        lower = self.n2_reference[:, None]
        upper = lower * ratio[:, None]
        # A single line shares its absorption/emission profile. Evaluate the
        # profile-independent coefficients once with unit profile values.
        alpha_weight, emission_weight = line_coefficients(
            jnp.ones((len(self.radius_rp), 1, 1)), self.adb.nu_lines, self.adb.A,
            self.adb.glower, self.adb.gupper, lower, upper,
        )
        # Keep S finite at zero requested column: scale dtau, not j/alpha.
        source = emission_weight / alpha_weight
        column_scale = n2_column / REFERENCE_COLUMN_CM2
        radial_velocity = (
            self.radial_velocity(wind_temperature_k, mean_molecular_weight)
            if radial_wind else jnp.zeros_like(self.radius_rp)
        )

        def add_patch(total, patch):
            mu, weight, tangent_los, cos_lat, sin_lon = patch
            velocity = -mu * radial_velocity - tangent_los * (
                rotation_kms * cos_lat / self.radius_rp + zonal_kms * sin_lon
            )
            profiles = jax.vmap(self.opa.line_profiles)(self.temperature_k, velocity)[:, 0, :]
            dtau = alpha_weight * profiles * self.thickness_cm[:, None] * column_scale
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
