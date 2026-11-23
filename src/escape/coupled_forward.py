"""Internal radiation and multilevel populations on the actual observer path.

This controlled model keeps the atmosphere's density, temperature and velocity
explicit. Its static, plane-parallel CRD closure is not Cloudy's PRD/escape
approximation. Thermal and hydrodynamic adequacy require separate evidence.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .coupled_hydrogen import CoupledHydrogenModel
from .hydrogen_forward import PLANET_RADIUS_CM
from .nlte_forward import HydrogenNLTEForwardModel


class CoupledHydrogenForwardModel:
    """Solve the internal mean field before observing the same populations.

    Supplied atmospheric arrays follow the observer's top-to-bottom ordering.
    Incoming intensities are I_nu on radiation.grid.frequency_hz, in cgs per Hz
    per steradian. The default outer boundary illuminates the inward hemisphere.
    With stellar_beam=True, outer/2 instead specifies the mean intensity of a
    normally incident beam. Both conventions give J_nu = W B_nu for outer=2 W B_nu
    in a transparent slab with a dark bottom; their incident fluxes differ.
    The default boundary continuum of the observer is still B(4500 K).
    Pass the same lower radiation boundary when its feedback is intended.

    Temperature/density are not fit-independent excited-population knobs:
    absolute populations and ionization are solved jointly with radiation.
    No MCEG import or external executable enters the forward computation.
    """

    def __init__(self, *, nlayer=32, quadrature_order=4, angle_order=4,
                 transfer="formal", include_hminus=False,
                 low_temperature_extrapolation="reject", stellar_beam=False,
                 line_wing_offsets=(30., 100., 300., 1000.), **kwargs):
        self.observer = HydrogenNLTEForwardModel(nlayer=nlayer, **kwargs)
        self.radius_rp = self.observer.radius_rp
        self.velocity_kms = self.observer.velocity_kms
        self.radiation = CoupledHydrogenModel(
            np.linspace(1., 3., nlayer + 1) * PLANET_RADIUS_CM,
            nmax=self.observer.nmax, quadrature_order=quadrature_order,
            spherical=False, transfer=transfer, angle_order=angle_order,
            include_hminus=include_hminus,
            low_temperature_extrapolation=low_temperature_extrapolation,
            stellar_beam=stellar_beam,
            line_wing_offsets=line_wing_offsets,
        )

    @partial(jax.jit, static_argnums=0, static_argnames=("thermal", "iterations", "continuation_steps"))
    def population_profiles(
        self, n_h_cm3, temperature_k, inner_intensity, outer_intensity,
        *, thermal=False, iterations=32, continuation_steps=8,
    ):
        """Return the joint solution, reordered to the observer's radial grid.

        thermal=True solves local radiative equilibrium at prescribed density;
        it does not include expansion or material advection. The default
        thermal=False is the matched-profile radiation-feedback experiment.
        Failed roots are visible, never replaced by prescribed-field results.
        """
        shape = self.radius_rp.shape
        density, temperature = jnp.asarray(n_h_cm3), jnp.asarray(temperature_k)
        if density.shape != shape or temperature.shape != shape:
            raise ValueError("Density and temperature must match the observer's radial grid.")
        solution = self.radiation.solve_equilibrium(
            density[::-1], temperature[::-1], inner_intensity, outer_intensity,
            thermal=thermal, iterations=iterations, continuation_steps=continuation_steps,
            continuation_start=1e-8,
        )
        return {
            "bound_cm3": solution.bound_cm3[::-1],
            "electron_cm3": solution.electron_cm3[::-1],
            "proton_cm3": solution.electron_cm3[::-1],
            "temperature_k": solution.temperature_k[::-1],
            "ionized_fraction": solution.ionized_fraction[::-1],
            "mean_molecular_weight": 1 / (1 + solution.ionized_fraction[::-1]),
            "mean_intensity": solution.radiation.mean_intensity[::-1],
            "n2_cm3": jnp.sum(solution.bound_cm3[::-1, 1:3], axis=1),
            "n3_cm3": jnp.sum(solution.bound_cm3[::-1, 3:6], axis=1),
            "residual_norm": solution.residual_norm,
            "radiation_balance_error": solution.radiation.radiation_balance_error,
            "converged": solution.converged,
        }

    @partial(jax.jit, static_argnums=0, static_argnames=(
        "thermal", "iterations", "continuation_steps", "include_continuum", "include_hminus",
    ))
    def spectrum(
        self, n_h_cm3, temperature_k, radial_velocity_kms,
        inner_intensity, outer_intensity, *, rotation_kms=0., zonal_kms=0.,
        thermal=False, iterations=32, continuation_steps=8,
        include_continuum=True, include_hminus=None,
    ):
        """Differentiate through the same RT/SE root used to form the spectrum."""
        if include_hminus is None:
            include_hminus = self.radiation.include_hminus
        elif include_hminus != self.radiation.include_hminus:
            raise ValueError("Select H-minus at construction for both atmosphere and observer.")
        solution = self.population_profiles(
            n_h_cm3, temperature_k, inner_intensity, outer_intensity,
            thermal=thermal, iterations=iterations, continuation_steps=continuation_steps,
        )
        contrast = self.observer.spectrum_from_profiles(
            solution["bound_cm3"], solution["temperature_k"], radial_velocity_kms,
            electron_cm3=solution["electron_cm3"], proton_cm3=solution["proton_cm3"],
            rotation_kms=rotation_kms, zonal_kms=zonal_kms,
            include_continuum=include_continuum, include_hminus=include_hminus,
        )
        return jnp.where(solution["converged"], contrast, jnp.nan)


__all__ = ["CoupledHydrogenForwardModel"]
