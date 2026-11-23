"""Ground-state helium charge ladder for the cool irradiated atmosphere.

This reduction has He I, He II and He III, ground photoionization, electron
ionization, total radiative recombination and H charge exchange. Excited He I,
metastable photoionization, dielectronic capture and line cooling are absent;
their small contribution must be checked in the selected stellar/T domain.
The rates are source-backed C23 fits, not reference atmosphere interpolants.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .metal_rates import (
    EV_ERG, KB, PLANCK, charge_exchange, collisional_ionization,
    ground_recombination_continuum, load_metal_rates,
    photoionization_cross_sections, radiative_recombination,
)


class GroundHelium:
    """Variable-temperature three-charge-state closure on a common continuum."""

    def __init__(self, frequency_hz, weight_hz):
        self.frequency_hz, self.weight_hz = jnp.asarray(frequency_hz), jnp.asarray(weight_hz)
        self.data = load_metal_rates()
        self.photo = tuple(self.data["photo"][2, q] for q in range(2))
        self.threshold = jnp.array([fit.threshold_ev[fit.outer_shell-1]*EV_ERG for fit in self.photo])

    @staticmethod
    def unpack(state, helium_density_cm3):
        fractions = jax.nn.softmax(jnp.column_stack((jnp.zeros(len(state)), state)), axis=1)
        return jnp.asarray(helium_density_cm3)[:, None]*fractions

    def cross_sections(self, frequency_hz):
        energy = jnp.asarray(frequency_hz)*PLANCK/EV_ERG
        return jnp.stack([photoionization_cross_sections(fit, energy).sum(-1) for fit in self.photo])

    def continuum_coefficients_at(self, temperature_k, ion_cm3, electron_cm3, frequency_hz):
        absorption = jnp.asarray(ion_cm3)[:, :2]@self.cross_sections(frequency_hz)
        emission = jnp.zeros_like(absorption)
        for q in range(2):
            weights = self.data["ground_weight"][2, q]
            emission += ground_recombination_continuum(
                self.photo[q], weights[0]/weights[1], temperature_k,
                electron_cm3, ion_cm3[:, q+1], frequency_hz,
            )
        return absorption, emission

    def evaluate(self, temperature_k, ion_cm3, electron_cm3, ground_h_cm3,
                 proton_cm3, mean_intensity, *, ots_frequency_hz=None,
                 ots_flux_cm2_s=None, extra_ionization_s=0., extra_recombination_s=0.):
        """Return charge-ratio target, H backreaction, and a kinetic heat ledger.

        OTS photons use the same cross sections as continuum absorption.
        RR cooling uses kT per capture as in the low-charge approximation;
        the emitted ground continuum is computed separately. Charge-transfer
        heating here is zero, consistent with C23's disabled He-CT heat term.
        """
        t, ne = jnp.asarray(temperature_k), jnp.asarray(electron_cm3)
        ion = jnp.asarray(ion_cm3)
        frequency = self.frequency_hz
        flux = 4*jnp.pi*mean_intensity*self.weight_hz/(PLANCK*frequency)

        def photo_terms(nu, photons):
            cross = self.cross_sections(nu)
            excess = jnp.maximum(PLANCK*nu[None, :]-self.threshold[:, None], 0.)
            return photons@cross.T, photons@(cross*excess).T

        photo, heat = photo_terms(frequency, flux)
        if ots_frequency_hz is not None:
            extra_photo, extra_heat = photo_terms(ots_frequency_hz, ots_flux_cm2_s)
            photo, heat = photo+extra_photo, heat+extra_heat
        ci = jnp.stack([collisional_ionization(t, 2, q, self.data) for q in range(2)], axis=1)
        rr = jnp.stack([radiative_recombination(t, 2, q, self.data) for q in (1, 2)], axis=1)
        ct = [charge_exchange(t, 2, q, self.data) for q in (1, 2)]
        h_ion = jnp.stack([item[0] for item in ct], axis=1)
        h_rec = jnp.stack([item[1] for item in ct], axis=1)
        upward = photo+ne[:, None]*ci+proton_cm3[:, None]*h_ion+extra_ionization_s
        downward = ne[:, None]*rr+ground_h_cm3[:, None]*h_rec+extra_recombination_s
        # A zero source has a zero physical rate. Only its logarithmic state
        # representation uses machine-tiny, below any retained population.
        tiny = jnp.finfo(t.dtype).tiny
        log_ratio = jnp.log(jnp.maximum(upward, tiny))-jnp.log(jnp.maximum(downward, tiny))
        target = jnp.stack((log_ratio[:, 0], log_ratio.sum(1)), axis=1)
        photo_heat = jnp.sum(ion[:, :2]*heat, axis=1)
        collision_cool = jnp.sum(ne[:, None]*ion[:, :2]*ci*self.threshold, axis=1)
        rr_cool = jnp.sum(ne[:, None]*ion[:, 1:]*rr, axis=1)*KB*t
        return dict(
            target_state=target,
            hydrogen_ct_ionization_s=jnp.sum(ion[:, 1:]*h_rec, axis=1),
            hydrogen_ct_recombination_s=jnp.sum(ion[:, :2]*h_ion, axis=1),
            thermal_heating=photo_heat, thermal_cooling=collision_cool+rr_cool,
            thermal_net=photo_heat-collision_cool-rr_cool,
            valid=jnp.all(jnp.isfinite(target)),
        )


__all__ = ["GroundHelium"]
