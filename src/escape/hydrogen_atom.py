"""Hydrogen atom with resolved low levels and optional collapsed n=4,5 shells.

This extends, rather than changes, the frozen six-state model. Higher-shell
populations are statistical over l; the first six populations remain resolved.
Dipole integrals use exact nonrelativistic hydrogen radial polynomials. See
Prozesky & Smits (2018), Appendix A, Eqs. 26 and 45,
https://arxiv.org/html/1805.02440v1 . Higher-shell collisions use the VS80
prescription evaluated in Cloudy C23 ``hydro_vs_deexcit``; this is a shell
approximation, not new l-resolved collision data. The public formula source is
Vriens & Smeets (1980), https://doi.org/10.1103/PhysRevA.22.940 .
The VS80 evaluation is an altered Python adaptation of Cloudy C23, not the
original Cloudy routine; its retained notice is in escape/_licenses/Cloudy.txt.

All included recombination/cascade paths are explicit. No Case-B coefficient
or unmodelled high-n recombination is added at the truncation boundary. The
nmax=3,4,5 comparison measures sensitivity, not convergence to an infinite atom.
The retained low-level data restrict gas temperatures to 5802.26--14000 K.
"""

from __future__ import annotations

from math import comb, factorial, sqrt
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .hydrogen_nlte import RateCoefficients
from .hydrogen_rates import (
    BOLTZMANN_CGS, ELECTRON_MASS_CGS, EV_CGS, EXCITATION_ENERGY_ERG,
    IONIZATION_ENERGY_ERG, LIGHT_SPEED_CGS, LINE_EINSTEIN_A_S, LINE_LOWER,
    LINE_UPPER, MAX_TEMPERATURE_K, MIN_TEMPERATURE_K, PLANCK_CGS,
    PRINCIPAL_QUANTUM_NUMBERS, STATISTICAL_WEIGHTS, TWO_PHOTON_A_S,
    collision_rates,
)

ELECTRON_CHARGE_ESU = 4.803204712570263e-10
PROTON_MASS_CGS = 1.67262192369e-24
HYDROGEN_BOHR_RADIUS_CM = 5.29177210544e-9 * (1 + ELECTRON_MASS_CGS / PROTON_MASS_CGS)
_IONIZATION_LIMIT_ERG = 13.5984345 * EV_CGS
_THERMAL_VOLUME = (PLANCK_CGS**2 / (2 * np.pi * ELECTRON_MASS_CGS * BOLTZMANN_CGS))**1.5


class HydrogenAtom(NamedTuple):
    """Static atomic arrays; ordinary frequencies in Hz, energies in erg."""

    principal_n: jnp.ndarray
    statistical_weights: jnp.ndarray
    excitation_energy_erg: jnp.ndarray
    ionization_energy_erg: jnp.ndarray
    line_upper: jnp.ndarray
    line_lower: jnp.ndarray
    einstein_a_s: jnp.ndarray
    line_frequency_hz: jnp.ndarray


def hydrogenic_radial_dipole(n_upper, l_upper, n_lower, l_lower):
    """Return the radial integral of r between normalized orbitals, in cm.

    Finite Laguerre polynomials are integrated analytically term by term;
    no numerical radial grid or fitted A coefficient enters this calculation.
    Only n <= 5 is supported here to avoid large-n cancellation.
    """
    for n, ell in ((n_upper, l_upper), (n_lower, l_lower)):
        if not (isinstance(n, int) and isinstance(ell, int) and 1 <= n <= 5 and 0 <= ell < n):
            raise ValueError("Use integer 1 <= n <= 5 and 0 <= l < n.")

    def polynomial(n, ell):
        order = n - ell - 1
        norm = 2 / n**2 * sqrt(factorial(order) / factorial(n + ell))
        return [norm * (-1)**k * comb(n + ell, order - k) / factorial(k)
                * (2 / n)**(ell + k) for k in range(order + 1)]

    upper, lower = polynomial(n_upper, l_upper), polynomial(n_lower, l_lower)
    inverse_scale = 1 / n_upper + 1 / n_lower
    integral = 0.0
    for i, a in enumerate(upper):
        for j, b in enumerate(lower):
            power = l_upper + l_lower + i + j + 3
            integral += a * b * factorial(power) / inverse_scale**(power + 1)
    return integral * HYDROGEN_BOHR_RADIUS_CM


def hydrogenic_einstein_a(n_upper, l_upper, n_lower, l_lower, frequency_hz=None):
    """Return an electric-dipole A [s^-1], averaged over the upper m states."""
    if n_upper <= n_lower:
        raise ValueError("The upper principal quantum number must be larger.")
    if abs(l_upper - l_lower) != 1:
        return 0.0
    if frequency_hz is None:
        frequency_hz = _IONIZATION_LIMIT_ERG / PLANCK_CGS * (1 / n_lower**2 - 1 / n_upper**2)
    radial = hydrogenic_radial_dipole(n_upper, l_upper, n_lower, l_lower)
    angular = max(l_upper, l_lower) / (2 * l_upper + 1)
    return (64 * np.pi**4 / (3 * PLANCK_CGS * LIGHT_SPEED_CGS**3)
            * frequency_hz**3 * ELECTRON_CHARGE_ESU**2 * radial**2 * angular)


def make_hydrogen_atom(nmax=5):
    """Construct six resolved states plus zero, one or two statistical shells.

    The resolved states retain the NIST-derived energies and A values of the
    previous model. Added n=4,5 energies use the hydrogenic ionization limit;
    their A values are summed with g_nl/g_n over their upper l sublevels.
    """
    if not isinstance(nmax, int) or isinstance(nmax, bool) or nmax not in (3, 4, 5):
        raise ValueError("nmax must be 3, 4 or 5.")
    principal = list(PRINCIPAL_QUANTUM_NUMBERS)
    weights = list(STATISTICAL_WEIGHTS)
    energy = list(EXCITATION_ENERGY_ERG)
    binding = list(IONIZATION_ENERGY_ERG)
    members = [[(1, 0)], [(2, 0)], [(2, 1)], [(3, 0)], [(3, 1)], [(3, 2)]]
    upper, lower, a_values = list(LINE_UPPER), list(LINE_LOWER), list(LINE_EINSTEIN_A_S)
    for n in range(4, nmax + 1):
        principal.append(n)
        weights.append(2. * n**2)
        energy.append(_IONIZATION_LIMIT_ERG * (1 - 1 / n**2))
        binding.append(_IONIZATION_LIMIT_ERG / n**2)
        members.append([(n, ell) for ell in range(n)])
        high = len(principal) - 1
        for low in range(high):
            frequency = (energy[high] - energy[low]) / PLANCK_CGS
            a_group = sum(
                2 * (2 * lh + 1) / weights[high]
                * hydrogenic_einstein_a(nh, lh, nl, ll, frequency)
                for nh, lh in members[high] for nl, ll in members[low]
            )
            if a_group > 0:
                upper.append(high)
                lower.append(low)
                a_values.append(a_group)
    energy = np.asarray(energy)
    return HydrogenAtom(
        jnp.asarray(principal), jnp.asarray(weights), jnp.asarray(energy),
        jnp.asarray(binding), jnp.asarray(upper), jnp.asarray(lower),
        jnp.asarray(a_values),
        jnp.asarray((energy[upper] - energy[lower]) / PLANCK_CGS),
    )


def atom_saha_factors(atom, temperature_k):
    """Return n_i/(n_e n_p) [cm^3] for this finite atom."""
    return (atom.statistical_weights / 2 * _THERMAL_VOLUME / temperature_k**1.5
            * jnp.exp(atom.ionization_energy_erg / (BOLTZMANN_CGS * temperature_k)))


def atom_cross_sections(atom, frequency_hz):
    """Kramers g_bf=1 cross sections [cm^2], shape (Nlevel, Nfrequency)."""
    frequency = jnp.asarray(frequency_hz)[None, :]
    sigma = 2.815e29 / atom.principal_n[:, None]**5 / frequency**3
    return jnp.where(PLANCK_CGS * frequency >= atom.ionization_energy_erg[:, None], sigma, 0.0)


def atom_milne_population(atom, temperature_k, frequency_hz):
    """Return the per-ne-np inverse-continuum population factor [cm^3].

    Multiplication by ne*np gives the upper-continuum term in bound-free
    extinction and spontaneous recombination emissivity. Writing the excess
    photon energy in one exponential avoids a product of large/small factors.
    """
    excess = PLANCK_CGS * jnp.asarray(frequency_hz)[None, :] - atom.ionization_energy_erg[:, None]
    return (atom.statistical_weights[:, None] / 2 * _THERMAL_VOLUME / temperature_k**1.5
            * jnp.exp(-excess / (BOLTZMANN_CGS * temperature_k)))


def atom_bound_free_rates(atom, temperature_k, occupation, frequency_hz, weight_hz):
    """Return photoionization [s^-1], Milne recombination [cm^3/s] vectors."""
    frequency = jnp.asarray(frequency_hz)
    kernel = 8 * jnp.pi / LIGHT_SPEED_CGS**2 * frequency[None, :]**2 * atom_cross_sections(atom, frequency)
    gamma = jnp.sum(kernel * occupation * weight_hz, axis=-1)
    alpha = jnp.sum(kernel * atom_milne_population(atom, temperature_k, frequency)
                    * (1 + occupation) * weight_hz, axis=-1)
    return gamma, alpha


def _vs80_shell_deexcitation(temperature_k, n_upper, n_lower, binding_upper_ev,
                             energy_gap_ev, shell_a_s):
    """VS80 Eq. 24 shell rate, using the Cloudy C23 evaluation convention.

    In this convention the binding energy in B_np is that of the upper state,
    as in Cloudy's hydro_vs_deexcit. This adopted approximation is independently
    attributed; agreement with every original cross-section fit is not asserted.
    """
    kt_ev = BOLTZMANN_CGS * temperature_k / EV_CGS
    ryd_ev = _IONIZATION_LIMIT_ERG / EV_CGS
    gap = energy_gap_ev
    frequency = gap * EV_CGS / PLANCK_CGS
    oscillator = (ELECTRON_MASS_CGS * LIGHT_SPEED_CGS**3 / (8 * jnp.pi**2 * ELECTRON_CHARGE_ESU**2)
                  * n_upper**2 / n_lower**2 * shell_a_s / frequency**2)
    a_term = 2 * ryd_ev / gap * oscillator
    b_lower = (1.4 * jnp.log(n_lower) - .7 - .51 / n_lower
               + 1.16 / n_lower**2 - .55 / n_lower**3) / n_lower
    b_term = 4 * ryd_ev**2 / (n_upper**3 * gap**2) * (
        1 + binding_upper_ev / gap * (4 / 3 + b_lower * binding_upper_ev / gap)
    )
    separation = n_upper - n_lower
    delta = jnp.exp(-b_term / a_term) + .06 * separation**2 / (n_upper * n_lower**2)
    gamma = (ryd_ev * jnp.log1p(n_lower**3 * kt_ev / ryd_ev)
             * (3 + 11 * separation**2 / n_lower**2) * separation**2
             / (separation**2 * (6 + 1.6 * n_upper * separation) + .3
                + .8 * (n_upper * separation)**1.5 * jnp.abs(separation - .6)))
    return (1.6e-7 * jnp.sqrt(kt_ev) * n_lower**2 / n_upper**2 / (kt_ev + gamma)
            * (a_term * jnp.log(.3 * kt_ev / ryd_ev + delta) + b_term))


def atom_collision_rates(atom, temperature_k):
    """Return Qbb [cm^3/s], qion [cm^3/s], q3 [cm^6/s] for this atom.

    Existing low-level rates are unchanged. Added shell de-excitation is
    distributed among resolved lower states in proportion to branch A values;
    all inverse rates use the actual group energies and statistical weights.
    """
    low_bb, low_ion, _ = collision_rates(temperature_k)
    size = atom.principal_n.shape[0]
    bound_bound = jnp.zeros((size, size)).at[:6, :6].set(low_bb)
    # Only added-shell transitions use the approximate shell collision formula.
    upper, lower = atom.line_upper[5:], atom.line_lower[5:]
    a_values = atom.einstein_a_s[5:]
    n_upper, n_lower = atom.principal_n[upper], atom.principal_n[lower]
    matching = (upper[:, None] == upper[None, :]) & (n_lower[:, None] == n_lower[None, :])
    shell_a = jnp.sum(jnp.where(matching, a_values[None, :], 0.), axis=1)
    lower_members = atom.principal_n[None, :] == n_lower[:, None]
    shell_lower_energy = jnp.sum(jnp.where(
        lower_members, atom.statistical_weights * atom.excitation_energy_erg, 0.,
    ), axis=1) / (2 * n_lower**2)
    gap_ev = (atom.excitation_energy_erg[upper] - shell_lower_energy) / EV_CGS
    shell_down = _vs80_shell_deexcitation(
        temperature_k, n_upper, n_lower, atom.ionization_energy_erg[upper] / EV_CGS,
        gap_ev, shell_a,
    )
    down = shell_down * a_values / shell_a
    up = down * atom.statistical_weights[upper] / atom.statistical_weights[lower] * jnp.exp(
        -(atom.excitation_energy_erg[upper] - atom.excitation_energy_erg[lower])
        / (BOLTZMANN_CGS * temperature_k)
    )
    bound_bound = bound_bound.at[upper, lower].set(down).at[lower, upper].set(up)
    # Garcia Munoz & Schneider (2019), Appendix Table 2, shell n=4,5 fits.
    a = jnp.array([1.27e-6, 5.39e-6])[:size - 6]
    b = jnp.array([.053, -.023])[:size - 6]
    c = jnp.array([-1.09e4, -7.38e3])[:size - 6]
    ionization = jnp.concatenate((low_ion, a * temperature_k**b * jnp.exp(c / temperature_k)))
    three_body = ionization * atom_saha_factors(atom, temperature_k)
    valid = ((temperature_k >= MIN_TEMPERATURE_K) & (temperature_k <= MAX_TEMPERATURE_K)
             & jnp.all(jnp.isfinite(bound_bound) & (bound_bound >= 0)))
    return tuple(jnp.where(valid, item, jnp.nan) for item in (bound_bound, ionization, three_body))


def atom_rate_coefficients(atom, temperature_k, line_occupation, continuum_occupation,
                           frequency_hz, weight_hz):
    """Assemble local rates using a common 1D continuum grid and line occupations."""
    if jnp.shape(line_occupation) != jnp.shape(atom.einstein_a_s):
        raise ValueError("Supply one occupation per atomic line.")
    shape = jnp.shape(frequency_hz)
    if len(shape) != 1 or shape[0] < 2 or jnp.shape(weight_hz) != shape or jnp.shape(continuum_occupation) != shape:
        raise ValueError("Continuum frequency, occupation and weights must share shape (N,), N >= 2.")
    upper, lower = atom.line_upper, atom.line_lower
    g = atom.statistical_weights
    radiation = jnp.zeros((g.size, g.size))
    radiation = radiation.at[upper, lower].set(atom.einstein_a_s * (1 + line_occupation))
    radiation = radiation.at[lower, upper].set(atom.einstein_a_s * line_occupation * g[upper] / g[lower])
    radiation = radiation.at[1, 0].set(TWO_PHOTON_A_S)
    collisions, ionization, three_body = atom_collision_rates(atom, temperature_k)
    photo, recombination = atom_bound_free_rates(
        atom, temperature_k, continuum_occupation, frequency_hz, weight_hz,
    )
    valid = (jnp.all(jnp.isfinite(line_occupation) & (line_occupation >= 0))
             & jnp.all(jnp.isfinite(continuum_occupation) & (continuum_occupation >= 0))
             & jnp.all(jnp.isfinite(frequency_hz) & (frequency_hz > 0))
             & jnp.all(jnp.isfinite(weight_hz) & (weight_hz > 0))
             & (temperature_k >= MIN_TEMPERATURE_K) & (temperature_k <= MAX_TEMPERATURE_K))
    rates = RateCoefficients(radiation, collisions, photo, recombination, ionization, three_body)
    return RateCoefficients(*(jnp.where(valid, item, jnp.nan) for item in rates))
