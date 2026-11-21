"""Continuum coefficients and total photon energy exchange for the finite atom.

Bound-free opacity and emissivity use exactly the same Kramers/Milne kernels
as the population rates. Free-free emission uses Maxwellian H+ bremsstrahlung
with Gaunt factor one, as in Garcia Munoz & Schneider (2019), Appendix FF:
https://arxiv.org/html/1910.00267v1 . The optional H-minus opacity reuses
ExoJAX's John (1988) implementation; it is an LTE trace-species closure, not
an H-minus reaction network. Its temperature range is 5040/3.6--10080 K.

Energy exchange is FULL photon energy, including ionization and excitation.
It belongs with a material energy that includes chemical/level energies.
Do not additionally add collisional excitation or ionization cooling to that
energy equation: those processes redistribute material energy internally.
"""

from __future__ import annotations

import jax.numpy as jnp

from exojax.database.hminus import bound_free_absorption, free_free_absorption

from .hydrogen_atom import (
    ELECTRON_CHARGE_ESU, atom_cross_sections, atom_milne_population,
)
from .hydrogen_rates import (
    BOLTZMANN_CGS, ELECTRON_MASS_CGS, LIGHT_SPEED_CGS, PLANCK_CGS,
    TWO_PHOTON_A_S, planck_occupation,
)

HMINUS_MIN_TEMPERATURE_K = 5040. / 3.6
HMINUS_MAX_TEMPERATURE_K = 10080.
HMINUS_TRACE_LIMIT = 1e-3
HMINUS_THRESHOLD_CM = 1.6419e-4


def planck_intensity_hz(frequency_hz, temperature_k):
    """Return B_nu [erg s^-1 cm^-2 sr^-1 Hz^-1]."""
    frequency = jnp.asarray(frequency_hz)
    return 2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * frequency**3 * planck_occupation(frequency, temperature_k)


def hydrogen_free_free_coefficients(temperature_k, electron_cm3, proton_cm3, frequency_hz):
    """Return H+ free-free extinction [cm^-1] and emissivity [per Hz/sr].

    Includes stimulated emission. The Maxwellian electrons and protons are
    at the same prescribed temperature; the frequency-dependent Gaunt factor
    is approximated by one. Kirchhoff's law is enforced by construction.
    """
    frequency = jnp.asarray(frequency_hz)
    prefactor = (4 / 3 * jnp.sqrt(2 * jnp.pi / (3 * ELECTRON_MASS_CGS * BOLTZMANN_CGS))
                 * ELECTRON_CHARGE_ESU**6 / (ELECTRON_MASS_CGS * PLANCK_CGS * LIGHT_SPEED_CGS))
    x = PLANCK_CGS * frequency / (BOLTZMANN_CGS * temperature_k)
    extinction = (prefactor * electron_cm3 * proton_cm3 / jnp.sqrt(temperature_k)
                  / frequency**3 * (-jnp.expm1(-x)))
    return extinction, extinction * planck_intensity_hz(frequency, temperature_k)


def hminus_lte_trace_density(temperature_k, ground_h_cm3, electron_cm3):
    """Return the H-minus population implied by the John opacity normalization.

    This Saha-like trace closure is relative to ground-state H plus electrons,
    even when neutral H is not in Saha equilibrium with protons. The adopted
    threshold and rounded normalization match ExoJAX's bound-free formula.
    Its charge and binding energy are not added to the principal H atom.
    """
    attachment_energy = PLANCK_CGS * LIGHT_SPEED_CGS / HMINUS_THRESHOLD_CM
    factor = .750 * BOLTZMANN_CGS * temperature_k**-1.5 * jnp.exp(
        attachment_energy / (BOLTZMANN_CGS * temperature_k)
    )
    return factor * ground_h_cm3 * electron_cm3


def hminus_trace_fraction(temperature_k, ground_h_cm3, electron_cm3, hydrogen_nuclei_cm3):
    """Return n(H-)/min(ne,nH), the diagnostic for neglecting its charge."""
    negative_ion = hminus_lte_trace_density(temperature_k, ground_h_cm3, electron_cm3)
    reference = jnp.minimum(electron_cm3, hydrogen_nuclei_cm3)
    return jnp.where(reference > 0, negative_ion / jnp.where(reference > 0, reference, 1.),
                     jnp.where(negative_ion == 0, 0., jnp.inf))


def hminus_lte_coefficients(temperature_k, ground_h_cm3, electron_cm3, hydrogen_nuclei_cm3,
                             frequency_hz):
    """Return optional John H-minus continuum with a checked LTE trace closure.

    ExoJAX's wavelength fits are reused without altering their coefficients.
    Their explicit wavelength support is 0.125--1.6419 micron (bound-free)
    and >=0.1823 micron (free-free). They contribute zero outside those fits;
    this does not assert a physically zero short-wavelength cross section.
    A temperature outside the fit range, or a trace fraction >=1e-3, gives NaN.
    """
    frequency = jnp.asarray(frequency_hz)
    wavelength_um = 1e4 * LIGHT_SPEED_CGS / frequency
    coefficient = (bound_free_absorption(wavelength_um, temperature_k)
                   + free_free_absorption(wavelength_um, temperature_k))
    extinction = coefficient * electron_cm3 * BOLTZMANN_CGS * temperature_k * ground_h_cm3
    trace = hminus_trace_fraction(temperature_k, ground_h_cm3, electron_cm3, hydrogen_nuclei_cm3)
    valid = ((temperature_k >= HMINUS_MIN_TEMPERATURE_K)
             & (temperature_k <= HMINUS_MAX_TEMPERATURE_K)
             & (ground_h_cm3 >= 0) & (electron_cm3 >= 0) & (hydrogen_nuclei_cm3 > 0)
             & jnp.isfinite(trace) & (trace < HMINUS_TRACE_LIMIT))
    extinction = jnp.where(valid, extinction, jnp.nan)
    return extinction, extinction * planck_intensity_hz(frequency, temperature_k)


def continuum_coefficients(atom, temperature_k, bound_cm3, electron_cm3, proton_cm3,
                           frequency_hz, *, include_free_free=True, include_hminus=False):
    """Return total continuum extinction [cm^-1], emissivity [per Hz/sr].

    Arrays have shape (Nfrequency,). Bound-free extinction is signed and
    contains stimulated recombination. No opacity or populations are clipped.
    H-minus is optional and omits trace charge/energy feedback explicitly.
    """
    frequency = jnp.asarray(frequency_hz)
    sigma = atom_cross_sections(atom, frequency)
    inverse_population = electron_cm3 * proton_cm3 * atom_milne_population(atom, temperature_k, frequency)
    extinction = jnp.sum(sigma * (jnp.asarray(bound_cm3)[:, None] - inverse_population), axis=0)
    emissivity = (2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * frequency**3
                  * jnp.sum(sigma * inverse_population, axis=0))
    if include_free_free:
        alpha_ff, j_ff = hydrogen_free_free_coefficients(temperature_k, electron_cm3, proton_cm3, frequency)
        extinction, emissivity = extinction + alpha_ff, emissivity + j_ff
    if include_hminus:
        alpha_minus, j_minus = hminus_lte_coefficients(
            temperature_k, bound_cm3[0], electron_cm3, jnp.sum(bound_cm3) + proton_cm3, frequency,
        )
        extinction, emissivity = extinction + alpha_minus, emissivity + j_minus
    return extinction, emissivity


def total_radiative_energy(atom, temperature_k, bound_cm3, electron_cm3, proton_cm3,
                           line_occupation, continuum_occupation, frequency_hz, weight_hz,
                           *, include_free_free=True, include_hminus=False):
    """Net matter energy gain [erg cm^-3 s^-1] from the SAME photon processes.

    Positive values heat total matter energy. The line occupation should be
    profile-averaged; continuum occupations and quadrature weights are on one
    common frequency grid. The spontaneous two-photon energy is an escaping
    channel; inverse/induced two-photon radiation is not included in this model.
    """
    upper, lower = atom.line_upper, atom.line_lower
    bound = jnp.asarray(bound_cm3)
    g = atom.statistical_weights
    line_net_absorption = atom.einstein_a_s * (
        (g[upper] / g[lower] * bound[lower] - bound[upper]) * line_occupation - bound[upper]
    )
    line_energy = jnp.sum(PLANCK_CGS * atom.line_frequency_hz * line_net_absorption)
    two_photon_energy = -bound[1] * TWO_PHOTON_A_S * (atom.excitation_energy_erg[1] - atom.excitation_energy_erg[0])
    extinction, emissivity = continuum_coefficients(
        atom, temperature_k, bound, electron_cm3, proton_cm3, frequency_hz,
        include_free_free=include_free_free, include_hminus=include_hminus,
    )
    frequency = jnp.asarray(frequency_hz)
    intensity = 2 * PLANCK_CGS / LIGHT_SPEED_CGS**2 * frequency**3 * continuum_occupation
    continuum_energy = 4 * jnp.pi * jnp.sum((extinction * intensity - emissivity) * weight_hz)
    return line_energy + two_photon_energy + continuum_energy
