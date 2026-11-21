"""Hydrogenic bound-free dipoles resolved in orbital angular momentum.

This is an altered small-n JAX adaptation of the Brocklehurst recurrence in
Cloudy C23.01 hydro_bauman.cpp (Robert Bauman's implementation). Its license
is retained in escape/_licenses/Cloudy.txt. The defining reference is
Brocklehurst (1971), MNRAS 153, 471, equations 3.13 and 3.17--3.24:
https://doi.org/10.1093/mnras/153.4.471 .

Only n<=5 is supported: ordinary float64 arithmetic suffices in this domain.
These are Coulomb-wave dipoles, not fits to the compared level populations.
Collapsed shells average over statistical orbital populations. The same
cross section must be used for photoionization and inverse Milne emission.
"""

from __future__ import annotations

from math import factorial, pi, sqrt

import jax.numpy as jnp

from .hydrogen_rates import PLANCK_CGS

_BOHR_RADIUS_CM = 5.29177210903e-9
_FINE_STRUCTURE = 1 / 137.035999084
_RESOLVED_NL = ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2))


def hydrogenic_cross_section(n, ell, relative_energy):
    """Return sigma_nl [cm^2] at photon energy divided by its threshold.

    Quantum numbers are static integers. Below threshold the result is zero;
    the right-hand threshold value is finite. A small-energy series avoids
    a singular intermediate atan(t)/t and retains its derivative.
    """
    if (not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 5
            or not isinstance(ell, int) or isinstance(ell, bool) or not 0 <= ell < n):
        raise ValueError("Use integer quantum numbers 1 <= n <= 5 and 0 <= ell < n.")
    energy = jnp.asarray(relative_energy)
    t2 = jnp.maximum(energy - 1, 0.)
    small = t2 < 1e-6
    safe_t = jnp.sqrt(jnp.where(small, 1., t2))
    atan_over_t = jnp.where(small, 1 - t2 / 3 + t2**2 / 5 - t2**3 / 7,
                           jnp.arctan(safe_t) / safe_t)
    k2 = t2 / n**2
    safe_k = jnp.sqrt(jnp.maximum(k2, 1e-30))
    g0 = sqrt(pi / 2) * 8 * n * (4 * n)**n / factorial(2 * n - 1)
    gk = (g0 * jnp.exp(-2 * n * atan_over_t)
          / jnp.sqrt(-jnp.expm1(-2 * jnp.pi / safe_k))
          / (1 + t2)**(n + 2))
    # G(n,l;K,l+1), descending in bound-state l.
    plus = {n - 1: gk}
    if n > 1:
        plus[n - 2] = (2 * n - 1) * (1 + t2) * n * gk
    for q in range(n - 3, -1, -1):
        plus[q] = ((4 * n**2 - 4 * (q + 2)**2 + (q + 2) * (2 * q + 3) * (1 + t2))
                   * plus[q + 1] - 4 * n**2 * (n**2 - (q + 2)**2)
                   * (1 + (q + 3)**2 * k2) * plus[q + 2])
    # G(n,l;K,l-1).
    minus = {n - 1: (1 + t2) / (2 * n) * gk}
    if n > 2:
        minus[n - 2] = (2 * n - 1) * (4 + (n - 1) * (1 + t2)) * minus[n - 1]
    for q in range(n - 3, 0, -1):
        minus[q] = ((4 * n**2 - 4 * (q + 1)**2 + (q + 1) * (2 * q + 3) * (1 + t2))
                    * minus[q + 1] - 4 * n**2 * (n**2 - (q + 2)**2)
                    * (1 + (q + 1)**2 * k2) * minus[q + 2])
    total = jnp.zeros_like(energy)
    channels = [(ell + 1, plus)] + ([(ell - 1, minus)] if ell > 0 else [])
    for lp, recurrence in channels:
        product = jnp.ones_like(energy)
        for s in range(1, lp + 1):
            product = product * (1 + s**2 * k2)
        radial2 = (factorial(n + ell) / factorial(n - ell - 1) * product
                   * (2 * n)**(2 * (ell - n)) * recurrence[ell]**2)
        total = total + max(ell, lp) / (2 * ell + 1) * (1 + t2) * radial2
    sigma = 4 / 3 * pi * _FINE_STRUCTURE * _BOHR_RADIUS_CM**2 * n**2 * total
    return jnp.where(jnp.isfinite(energy) & (energy > 0),
                     jnp.where(energy >= 1, sigma, 0.), jnp.nan)


def resolved_atom_cross_sections(atom, frequency_hz):
    """Return (Nlevel,Nfrequency) cross sections for the finite escape atom."""
    frequency = jnp.asarray(frequency_hz)
    count = atom.principal_n.shape[0]
    if count not in (6, 7, 8) or frequency.ndim != 1:
        raise ValueError("Use the six-state atom with optional n=4,5 shells and a 1D grid.")
    rows = []
    for i, (n, ell) in enumerate(_RESOLVED_NL):
        rows.append(hydrogenic_cross_section(n, ell, PLANCK_CGS * frequency
                                             / atom.ionization_energy_erg[i]))
    for i in range(6, count):
        n = i - 2
        relative = PLANCK_CGS * frequency / atom.ionization_energy_erg[i]
        rows.append(sum((2 * ell + 1) / n**2 * hydrogenic_cross_section(n, ell, relative)
                        for ell in range(n)))
    return jnp.stack(rows)


__all__ = ["hydrogenic_cross_section", "resolved_atom_cross_sections"]
