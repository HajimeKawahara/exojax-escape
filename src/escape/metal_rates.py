"""Low-charge metal coefficients and thermal primitives, adapted from C23.01.

The packaged tables have explicit source hashes and Cloudy's license. These
are altered JAX routines, not a replacement for the full Cloudy chemistry.
The Cota domain is 4000--15000 K and ne >= 1e5 cm^-3; its inverse is implicitly
differentiated. Ground PI excludes Auger branching and secondary electrons.
"""

from __future__ import annotations

from importlib.resources import files
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erfc
import numpy as np

EV_ERG = 1.602176634e-12
KB = 1.380649e-16
PLANCK = 6.62607015e-27
C = 2.99792458e10
ME = 9.1093837015e-28


class PhotoFit(NamedTuple):
    angular_momentum: jax.Array
    fit95: jax.Array
    fit96: jax.Array
    threshold_ev: jax.Array
    valid: jax.Array
    inner_shell: jax.Array
    outer_shell: jax.Array
    inner_fit_threshold_ev: jax.Array


@jax.jit
def photoionization_cross_sections(fit, photon_energy_ev):
    """Partial ground-state cross sections [cm2], final axis = 7 shells.

    Input energies are positive. Threshold discontinuities are physical; grids
    used for rate integrals should explicitly resolve every retained threshold.
    """
    energy = jnp.asarray(photon_energy_ev)[..., None]
    evaluation_energy = jnp.maximum(energy, fit.threshold_ev)
    p = fit.fit95
    y = evaluation_energy / p[:, 1]
    width95 = (p[:, 5].astype(jnp.float32) ** 2).astype(p.dtype)
    old = (
        p[:, 2]
        * ((y - 1) ** 2 + width95)
        * y ** (0.5 * p[:, 4] - fit.angular_momentum - 5.5)
    )
    old *= (1 + jnp.sqrt(y / p[:, 3])) ** (-p[:, 4])
    p = fit.fit96
    x = evaluation_energy / p[0] - p[5]
    width96 = (p[4].astype(jnp.float32) ** 2).astype(p.dtype)
    smooth96 = (p[6].astype(jnp.float32) ** 2).astype(p.dtype)
    y = jnp.sqrt(x * x + smooth96)
    modern = (
        p[1]
        * ((x - 1) ** 2 + width96)
        * y ** (0.5 * p[3] - 5.5)
        * (1 + jnp.sqrt(y / p[2])) ** (-p[3])
    )
    shell = jnp.arange(1, 8)
    use_old = (shell <= fit.inner_shell) | (energy >= fit.inner_fit_threshold_ev)
    shell_gap = (shell < fit.outer_shell) & (shell > fit.inner_shell) & ~use_old
    cross = jnp.where(use_old, old, modern)
    return jnp.where(
        fit.valid & (energy >= fit.threshold_ev) & ~shell_gap, cross * 1e-18, 0.0
    )


@jax.jit
def photoionization_rates(fit, photon_energy_ev, photon_flux_per_ev):
    """Per-ion shell rates [s-1] and primary excess energy [erg/s].

    The supplied flux is the angle-integrated photon intensity (4*pi*J_E),
    including any direct beam, in photons/cm2/s/eV. It is not a fitted Cloudy
    rate. Auger branching, secondary electrons, excited-state photoionization,
    diffuse radiation, and feedback on the flux are separate missing closures.
    """
    energy = jnp.asarray(photon_energy_ev)
    integrand = (
        photoionization_cross_sections(fit, energy) * photon_flux_per_ev[..., None]
    )
    ionization = jnp.trapezoid(integrand, energy, axis=-2)
    excess = jnp.maximum(energy[..., None] - fit.threshold_ev, 0.0) * EV_ERG
    heating = jnp.trapezoid(integrand * excess, energy, axis=-2)
    return ionization, heating


def radiative_recombination(temperature_k, z, charge, data):
    """Ground-parent q+ -> (q-1)+ RR coefficient [cm^3/s], q=1..3."""
    if not 1 <= charge <= min(z, 3):
        raise ValueError("Only low-charge stages 1..3 are supported.")
    temperature = jnp.asarray(temperature_k)
    key = z, z - charge
    if key in data["rr"]:
        p = list(data["rr"][key])
        p += [0.0] * (6 - len(p))
        a, b, t0, t1, c, t2 = p
        b = b + c * jnp.exp(-t2 / temperature)
        x, y = jnp.sqrt(temperature / t0), jnp.sqrt(temperature / t1)
        value = a / (x * (1 + x) ** (1 - b) * (1 + y) ** (1 + b))
    else:
        n_after = z - charge + 1
        if (
            n_after <= 3
            or n_after == 11
            or z in (6, 7, 8, 10)
            or (z == 26 and n_after > 11)
        ):
            a, b, t0, t1 = data["new_rr"][n_after - 1, z - 1]
            x, y = jnp.sqrt(temperature / t0), jnp.sqrt(temperature / t1)
            value = a / (x * (1 + x) ** (1 - b) * (1 + y) ** (1 + b))
        elif z == 26 and n_after <= 13:
            a, b, c = data["iron_rr"][n_after - 1]
            scaled = temperature / 1e4
            value = a / scaled ** (b + c * jnp.log10(scaled))
        else:
            a, b = data["old_rr"][n_after - 1, z - 1]
            value = a / (temperature / 1e4) ** b
    return jnp.where(temperature > 0, value, jnp.nan)


def raw_dielectronic_recombination(temperature_k, z, charge, data):
    """Pre-suppression DR; NOT Cloudy's final density-dependent coefficient."""
    if not 1 <= charge <= min(z, 3):
        raise ValueError("Only low-charge stages 1..3 are supported.")
    temperature = jnp.asarray(temperature_k)
    key = z, z - charge
    if key in data["dr_c"]:
        c, e = jnp.array(data["dr_c"][key]), jnp.array(data["dr_e"][key])
        value = (
            jnp.sum(c * jnp.exp(-e / temperature[..., None]), axis=-1)
            / temperature**1.5
        )
    else:
        value = jnp.zeros_like(temperature) + (3.78e-13, 1.70e-12, 8.14e-12)[charge - 1]
        if z == 26:
            # Arnaud & Raymond 1992, as C23 atmdat_dielrec_fe.cpp; low q.
            e, c = (
                ([5.12, 12.9], [2.20e-4, 1.00e-4]),
                ([16.7, 31.4], [2.30e-3, 2.70e-3]),
                ([28.6, 52.1], [1.50e-2, 4.70e-3]),
            )[charge - 1]
            value += (
                jnp.sum(
                    jnp.array(c)
                    * jnp.exp(
                        -jnp.array(e) / (temperature[..., None] * 8.617333262145e-5)
                    ),
                    axis=-1,
                )
                / temperature**1.5
            )
    return jnp.where(temperature > 0, value, jnp.nan)


def hydrogen_charge_exchange(temperature_k, z, charge, data):
    """Return ionization by H+ and recombination by H0 [cm^3/s].

    These are the baseline Kingdon--Ferland tables. Element-specific later
    overrides in ChargTranEval and molecular chemistry are not folded in.
    The selected C, Mg, Si, Ca, Fe low stages need a separate override audit.
    """
    if not 1 <= charge <= 3 or z < charge:
        raise ValueError("Use a bound low-charge metal stage 1..3.")
    temperature = jnp.asarray(temperature_k)
    rates = []
    for name in ("ct_ion", "ct_rec"):
        p = data[name][z - 1, charge - 1]
        a, b, c, d, tlo, thi = p[:6]
        scaled = jnp.maximum(jnp.clip(temperature, tlo, thi) * 1e-4, 1e-100)
        rate = a * 1e-9 * scaled**b * (1 + c * jnp.exp(d * scaled))
        if name == "ct_ion":
            rate *= jnp.exp(-p[6] * 1e4 / temperature)
        rates.append(jnp.where(temperature > 0, rate, jnp.nan))
    return tuple(rates)


def charge_exchange(temperature_k, z, charge, data):
    """Selected default-C23 H/He CT coefficients, in cm^3/s.

    Return H-ionization, H-recombination, He-ionization, He-recombination.
    charge=q labels (q-1)+ -> q+ for ionization and its reverse. Collider
    densities must be supplied separately, using ground H0/He0 for capture.
    The optional UMIST/Leiden override and molecular/grain reactions are absent.
    No temperature-range clipping is added beyond the original table formulas.
    """
    if z not in (2, 6, 7, 8, 12, 13, 14, 20, 25, 26) or charge not in (1, 2, 3):
        raise ValueError("Use a packaged element and a bound charge stage 1..3.")
    temperature = jnp.asarray(temperature_k)
    hi, hr = hydrogen_charge_exchange(temperature, z, charge, data)
    if charge == 1:
        if z == 26:
            hi = jnp.zeros_like(temperature) + 5.4e-9
        elif z in (13, 25):
            hi = jnp.zeros_like(temperature) + 3e-9
        elif z in (12, 14):
            a, b, c, d = (
                (9.76e-12, 3.14, 55.54, 1.12)
                if z == 12
                else (0.92e-12, 1.15, 0.80, 0.24)
            )
            hi = jnp.where(
                temperature < 1e5,
                a
                * (temperature / 1e4) ** b
                * (1 + c * jnp.exp(-d * temperature / 1e4)),
                0.0,
            )
    if z == 2 and charge == 1:
        hr += (
            7.47e-15
            * (temperature / 1e4) ** 2.06
            * (1 + 9.93 * jnp.exp(-3.89e-4 * temperature))
        )
    elif z == 7:
        logt = jnp.log(temperature)
        if charge == 1:
            hi = (
                1.64e-16 * temperature
                - 8.76e-17 * temperature * logt
                + 2.41e-20 * temperature**2
                + 9.83e-13 * logt
            ) * jnp.exp(-10985 / temperature)
            hr = (
                1.56e-15 * temperature
                - 1.79e-16 * temperature * logt
                + 1.15e-20 * temperature**2
                + 1.08e-13 * logt
            )
            hr += (
                6.83e-16 * temperature
                - 7.40e-17 * temperature * logt
                + 3.73e-21 * temperature**2
                + 1.75e-15 * logt
            ) * jnp.exp(-16680 / temperature)
        elif charge == 2:
            high = (
                1.0101e-9
                + 1.4589e-9 * jnp.maximum(jnp.log10(temperature / 20000), 0.0) ** 2.06
            )
            hr = jnp.where(
                temperature <= 1500,
                0.8692e-9 * (temperature / 1500) ** 0.17,
                jnp.where(
                    temperature <= 20000,
                    0.9703e-9 * (temperature / 10000) ** 0.058,
                    high,
                ),
            )
    elif z == 8:
        logt = jnp.log(temperature)
        if charge == 1:
            hr = jnp.polyval(
                jnp.array(
                    [
                        1.1963502e-13,
                        -2.8577012e-12,
                        2.9979994e-11,
                        -1.3146803e-10,
                        2.3651505e-10,
                        2.3344302e-10,
                    ]
                ),
                logt,
            )
            hi = jnp.polyval(
                jnp.array(
                    [
                        1.1580844e-14,
                        -2.6139493e-13,
                        2.0699463e-12,
                        -3.6606214e-12,
                        -1.488594e-12,
                        -3.7282001e-13,
                        -7.6767404e-14,
                    ]
                ),
                logt,
            )
            hi = jnp.where(
                temperature <= 200, 2.18733e-12 * (temperature - 190) + 1.85823e-10, hi
            )
            hi = jnp.where(
                temperature <= 190,
                jnp.exp(
                    -21.134531 - 242.06831 / temperature + 84.761441 / temperature**2
                ),
                hi,
            )
            hi = jnp.where(temperature <= 10, 4.749e-20, hi)
            hr = jnp.where(temperature <= 10, 3.744e-10, hr)
        elif charge == 2:
            hr = jnp.where(
                temperature <= 1500,
                0.5337e-9 * (temperature / 100) ** (-0.076),
                0.4344e-9
                + 0.6340e-9 * jnp.maximum(jnp.log10(temperature / 1500), 0.0) ** 2.06,
            )
    hei, her = jnp.zeros_like(temperature), jnp.zeros_like(temperature)
    tev = temperature / 11604.518121550082
    if z == 6:
        if charge == 1:
            hei = 6.3e-15 * (temperature / 300) ** 0.75
        elif charge == 2:
            hei = 5e-20 * temperature**2 * jnp.exp(-0.07e-4 * temperature - 6.29 / tev)
        elif charge == 3:
            her = 4.6e-19 * temperature**2
    elif z == 7:
        if charge == 2:
            her = jnp.zeros_like(temperature) + 0.8e-10
            hei = (
                3.7e-20 * temperature**2 * jnp.exp(-0.063e-4 * temperature - 1.44 / tev)
            )
        elif charge == 3:
            her = jnp.zeros_like(temperature) + 1.5e-10
    elif z == 8:
        if charge == 1:
            hei = (
                4.991e-15
                * (temperature / 1e4) ** 0.3794
                * jnp.exp(-temperature / 1.121e6)
            )
            hei += (
                2.780e-15
                * (temperature / 1e4) ** (-0.2163)
                * jnp.exp(jnp.minimum(1e7, temperature) / 8.158e5)
            )
        elif charge == 2:
            her = 3.2e-14 * temperature**0.95
        elif charge == 3:
            her = jnp.zeros_like(temperature) + 1e-9
    elif z == 12 and charge == 3:
        her = jnp.zeros_like(temperature) + 7.5e-10
    elif z == 14:
        if charge == 1:
            hei = jnp.zeros_like(temperature) + 3.3e-9
        elif charge == 2:
            hei = 1.5e-11 * temperature**0.25 * jnp.exp(-6.91 / tev)
        elif charge == 3:
            hei = 1.15e-11 * jnp.sqrt(temperature) * jnp.exp(-8.88 / tev)
            her = 1.3 * 1.5e-12 * temperature**0.7
    return tuple(jnp.where(temperature > 0, x, jnp.nan) for x in (hi, hr, hei, her))


def dielectronic_suppression(temperature_k, electron_cm3, z, charge):
    """Return DR suppression for the packaged metals, q=1..3."""
    if z not in (6, 7, 8, 12, 13, 14, 20, 25, 26) or charge not in (1, 2, 3):
        raise ValueError("Only packaged metals and q=1..3 are supported.")
    temperature = jnp.asarray(temperature_k)
    density = jnp.asarray(electron_cm3)
    sequence, q = z - charge, float(charge)
    n1, n2 = (3, 10) if sequence <= 10 else ((11, 18) if sequence <= 18 else (19, 36))
    an = 12 + 10 * n1 + (10 * n1 - 2 * n2) * (sequence - n1) / (n1 - n2)
    c = q / 10
    ec = 0.0
    if sequence == 3:
        ec = 1.96274 + c * (
            20.30014 + c * (-0.97103 + c * (0.85453 + c * (0.13547 + 0.02401 * c)))
        )
        an = 66.0
        p = (
            4.55441 + 0.08 * q**1.11864,
            0.3 + 2 * q**-2 * jnp.exp(-q / 67.36368),
            -0.4 + 0.38 * q**1.62248 * jnp.exp(-q / 2.78841),
            4.00192 + 0.58 * q**0.93519 * jnp.exp(-q / 21.28094),
            0.00198 + 0.32 * q**0.84436 * jnp.exp(-q / 9.73494),
            0.55031 - 0.32251 * q**0.75493 * jnp.exp(-q / 19.89169),
        )
    elif sequence == 4:
        ec = 5.78908 + c * (
            34.08270 + c * (1.51729 + c * (-1.21227 + c * (0.77559 - 0.00410 * c)))
        )
        an = 66.0
        p = (
            2.79861 + q**0.82983 * jnp.exp(-q / 18.05422),
            -0.01897 + 0.05 * q**1.34569 * jnp.exp(-q / 10.82096),
            -0.56934 + 0.68 * q**0.78839 * jnp.exp(-q / 2.77582),
            4.07101 + q**0.7175 * jnp.exp(-q / 25.89966),
            0.44352 + 0.05 * q**3.54877 * jnp.exp(-q / 0.94416),
            -0.57838 + 0.68 * q**0.08484 * jnp.exp(-q / 6.70076),
        )
    elif sequence == 7:
        ec = 11.37092 + c * (
            36.22053 + c * (7.08448 + c * (-5.16840 + c * (2.45056 - 0.16961 * c)))
        )
    elif sequence == 11:
        ec = 2.24809 + c * (
            22.27768 + c * (-1.12285 + c * (0.90267 + c * (-0.03860 + 0.01468 * c)))
        )
    elif sequence == 12:
        ec = 2.74508 + c * (
            19.18623 + c * (-0.54317 + c * (0.78685 + c * (-0.04249 + 0.01357 * c)))
        )
    logt = jnp.log10(temperature)

    def adjustment(p):
        p1, p2, p3, p4, p5, p6 = p
        return (
            1
            + p3 * jnp.exp(-0.5 * ((logt - p1) / p2) ** 2)
            + p6 * jnp.exp(-0.5 * ((logt - p4) / p5) ** 2)
        )

    if sequence in (3, 4):
        an *= 2 / (1 + jnp.exp(-jnp.sqrt(25000 * q * q / temperature))) * adjustment(p)
    if sequence == 5:
        # C23 explicitly resets A_N here, discarding the preceding Psi factor.
        an = 52.0
        g = (
            6.91078 - 1.6385 * q**2.18197 * jnp.exp(-q / 1.45091),
            0.4959 - 0.08348 * q**1.24745 * jnp.exp(-q / 8.55397),
            -0.27525 + 0.132 * q**1.15443 * jnp.exp(-q / 3.79949),
            7.45975 - 2.6722 * q**1.7423 * jnp.exp(-q / 1.19649),
            0.51285 - 0.60987 * q**5.15431 * jnp.exp(-q / 0.49095),
            -0.24818 + 0.125 * q**0.59971 * jnp.exp(-q / 8.34052),
        )
        an *= adjustment(g)
    elif sequence == 6:
        g = (
            5.90184 - 1.2997 * q**1.32018 * jnp.exp(-q / 2.10442),
            0.12606 + 0.009 * q**8.33887 * jnp.exp(-q / 0.44742),
            -0.28222 + 0.018 * q**2.50307 * jnp.exp(-q / 3.83303),
            6.96615 - 0.41775 * q**2.75045 * jnp.exp(-q / 1.32394),
            0.55843 + 0.45 * jnp.exp(-q / 2.06664),
            -0.17208 - 0.17353 * jnp.exp(-q / 2.57406),
        )
        an *= adjustment(g)
    elif sequence == 13:
        g = (
            6.59628 - 3.03115 * jnp.exp(-q / 10.519821),
            1.20824 - 0.85509 * q**0.21258 * jnp.exp(-q / 25.56),
            -0.34292 - 0.06013 * q**4.09344 * jnp.exp(-q / 0.90604),
            7.92025 - 3.38912 * jnp.exp(-q / 10.02741),
            0.06976 + 0.6453 * q**0.24827 * jnp.exp(-q / 20.94907),
            -0.34108 - 0.17353 * jnp.exp(-q / 6.0384),
        )
        an *= adjustment(g)
    q0 = an * (1 - 0.816497 / jnp.sqrt(q)) / jnp.sqrt(q)
    activation = 10.1821 + jnp.log10(
        (q / q0) ** 7 * jnp.sqrt(temperature / (50000 * q0 * q0))
    )
    logn = jnp.log10(jnp.maximum(density, 1e-300))
    ordinary = jnp.exp(
        -jnp.log(2.0) * (jnp.maximum(logn - activation, 0) / 5.64586) ** 2
    )
    energy_ev = 20 * erfc(2 * (logn - 10.1821)) if sequence == 10 else ec
    value = 1 + (ordinary - 1) * jnp.exp(
        -energy_ev * 11604.518121550082 / (10 * temperature)
    )
    return jnp.where((temperature > 0) & (density >= 0), value, jnp.nan)


def _parabola(x, y, query):
    x1, x2, x3 = x
    y1, y2, y3 = y
    d13, d12 = x1 - x3, x1 - x2
    s13, s12 = (x1 + x3) * d13, (x1 + x2) * d12
    b = ((y1 - y2) * s13 - (y1 - y3) * s12) / (d12 * s13 - d13 * s12)
    a = ((y1 - y3) - d13 * b) / s13
    c = y1 - a * x1 * x1 - b * x1
    return a * query * query + b * query + c


def three_body_recombination(temperature_k, electron_cm3, charge, data):
    """C23 effective Cota coefficient for q=1..3, 4000–15000 K, ne>=1e5.

    Uses the original parabolic fits continuously, omitting the tiny
    exact-node shortcut. Solves the inverse mapping to machine tolerance,
    instead of C23's relative 1e-4 stop, and differentiates it implicitly.
    The restricted domain avoids Cota's low-temperature discontinuities.
    """
    if charge not in (1, 2, 3):
        raise ValueError("Supported ionic charges are 1..3.")
    temperature, density = jnp.broadcast_arrays(
        jnp.asarray(temperature_k), jnp.asarray(electron_cm3)
    )
    valid = (
        (temperature >= 4000)
        & (temperature <= 15000)
        & (density >= 1e5)
        & jnp.isfinite(density)
    )
    zlog2, zlog7 = data["zlog2"][charge - 1], data["zlog7"][charge - 1]
    logt = jnp.log10(temperature) - zlog2
    logn = 5 - zlog7
    grid = jnp.asarray(data["tz"])
    nearest = jnp.argmin(abs(logt[..., None] - grid), axis=-1)
    indices = nearest[..., None] + jnp.array([-1, 0, 1])
    abscissa = jnp.moveaxis(grid[indices], -1, 0)

    def interpolate(name, offset=0):
        values = jnp.asarray(data[name])[indices - offset]
        return _parabola(abscissa, jnp.moveaxis(values, -1, 0), logt)

    suffix, offset = ("2", 20) if charge == 1 else ("1", 0)
    c, d, u = (interpolate(name + suffix, offset) for name in ("a", "b", "x"))
    v = 9.4 if charge == 1 else 8.9

    def residual(logx):
        return c + d * 10 ** (u * logx) - v * logx - logn

    def solve(function, guess):
        def step(_, value):
            derivative = d * 10 ** (u * value) * u * jnp.log(10.0) - v
            return value - jnp.clip(function(value) / derivative, -0.09, 0.09)

        return jax.lax.fori_loop(0, 20, step, guess)

    logx = jax.lax.custom_root(
        residual,
        (c - logn) / v,
        solve,
        lambda linear, rhs: rhs / linear(jnp.ones_like(rhs)),
    )
    a, b, y = (interpolate(name + "0") for name in ("a", "b", "x"))
    coefficient = (
        charge
        * 10 ** (a - y * logx + b * 10 ** (-2.45 * logx))
        * (density / 1e5) ** 0.25
    )
    valid &= abs(residual(logx)) < 1e-10
    return jnp.where(valid, coefficient, jnp.nan)


def load_metal_rates():
    """Read compact selected coefficients without importing a reference solver."""
    data = {
        name: {}
        for name in (
            "rr",
            "dr_c",
            "dr_e",
            "old_rr",
            "new_rr",
            "iron_rr",
            "photo",
            "cota",
            "coll_ion",
            "ground_weight",
        )
    }
    resource = files("escape").joinpath("data/metal_rates.npz")
    with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as arrays:
        photo = {}
        for key in arrays.files:
            parts = key.split("__")
            value = arrays[key].copy()
            if parts[0] == "photo":
                ion = int(parts[1]), int(parts[2])
                photo.setdefault(ion, {})[parts[3]] = jnp.asarray(value)
            elif parts[0] == "cota":
                data["cota"][parts[1]] = jnp.asarray(value)
            elif len(parts) > 1:
                index = tuple(map(int, parts[1:]))
                if parts[0] == "iron_rr":
                    index = index[0]
                data[parts[0]][index] = value
            else:
                data[key] = value
        data["photo"] = {
            key: PhotoFit(*(values[name] for name in PhotoFit._fields))
            for key, values in photo.items()
        }
    return data


def collisional_ionization(temperature_k, atomic_number, charge, data):
    """C23 default HYBRID ground-ionization coefficient [cm3/s].

    Voronov (1997) formula times the Dere (2007) ratio used by C23.01.
    It excludes excited-ion and secondary-electron ionization.
    """
    temperature = jnp.asarray(temperature_k)
    threshold, p, amplitude, x, k, ratio = data["coll_ion"][atomic_number, charge]
    u = threshold / (temperature * KB / EV_ERG)
    coefficient = amplitude * (1 + p * jnp.sqrt(u)) / (x + u) * u**k * jnp.exp(-u)
    return jnp.where(
        temperature > 0, jnp.where(u > 80, 0.0, coefficient * ratio), jnp.nan
    )


def ground_recombination_continuum(
    fit,
    ground_weight_ratio,
    temperature_k,
    electron_cm3,
    recombining_ion_cm3,
    frequency_hz,
    *,
    valence_cross_section_cm2=None,
):
    """C23 ground-shell Milne emissivity [erg/s/cm3/sr/Hz].

    This is the ground capture continuum, not total RR/DR. The ionic ladder
    uses independently fitted total coefficients. No total-rate renormalization
    is applied, following rt_diffuse.cpp's heavy-ion approximation.
    """
    frequency = jnp.asarray(frequency_hz)
    temperature = jnp.asarray(temperature_k)[..., None]
    cross = photoionization_cross_sections(fit, PLANCK * frequency / EV_ERG)[
        ..., fit.outer_shell - 1
    ]
    if valence_cross_section_cm2 is not None:
        cross = jnp.asarray(valence_cross_section_cm2)
    threshold = fit.threshold_ev[fit.outer_shell - 1] * EV_ERG
    inverse = (
        ground_weight_ratio
        / 2
        * (PLANCK**2 / (2 * jnp.pi * ME * KB * temperature)) ** 1.5
    )
    inverse *= jnp.exp(
        -jnp.maximum(PLANCK * frequency - threshold, 0) / (KB * temperature)
    )
    return (
        (jnp.asarray(electron_cm3) * jnp.asarray(recombining_ion_cm3))[..., None]
        * inverse
        * cross
        * 2
        * PLANCK
        * frequency**3
        / C**2
    )


def oxygen_ground_valence_cross_section(photon_energy_ev):
    """C23 ``ofit`` six-channel O I valence opacity [cm2].

    This is the source-defined replacement of the valence entry, including
    its multi-electron channels above 50 eV. The reduced three-charge ladder
    does not include Auger branching; incident hard photons need a scope check.
    Original intermediate float32 rounding is omitted for differentiability.
    """
    coefficients = jnp.array(
        [
            [8.915, 3995.0, 3.242, 10.44, 0.0],
            [11.31, 1498.0, 5.27, 7.319, 0.0],
            [10.5, 1.059e5, 1.263, 13.04, 0.0],
            [19.49, 48.47, 8.806, 5.983, 0.0],
            [50.0, 4.244e4, 0.1913, 7.012, 0.04454],
            [110.5, 0.1588, 148.3, -3.38, 0.03589],
        ]
    )
    threshold = jnp.array([13.62, 16.94, 18.79, 28.48, 50.0, 110.5])
    angular = jnp.array([1, 1, 1, 0, 1, 1])
    energy = jnp.asarray(photon_energy_ev)[..., None]
    x = jnp.maximum(energy, threshold) / coefficients[:, 0]
    exponent = 5.5 - 0.5 * coefficients[:, 3] + angular
    values = coefficients[:, 1] * ((x - 1) ** 2 + coefficients[:, 4] ** 2) / x**exponent
    values /= (1 + jnp.sqrt(x / coefficients[:, 2])) ** coefficients[:, 3]
    return jnp.sum(jnp.where(energy >= threshold, values, 0.0), axis=-1) * 1e-18


def excited_photoionization_cross_section(species, frequency_hz):
    """Selected C23 excited-ion fits [cm2], with their finite energy domains.

    N I 2D, O I 1D and Mg II 3p fits from opacity_createall.cpp. C23's
    continuum-bin edges are replaced by the declared physical endpoints.
    """
    rydberg_hz = C * 109737.3156816
    energy = jnp.asarray(frequency_hz) / rydberg_hz
    if species == "N I":
        threshold, upper = 0.893, 2.0
        value = 9e-18 * (jnp.maximum(energy, threshold) / threshold) ** (-1.75)
    elif species == "O I":
        threshold, upper = 0.856, 2.0
        value = jnp.zeros_like(energy) + 4.64e-18
    elif species == "Mg II":
        threshold, upper = 0.779, 1.0
        value = (
            0.2602325880970085
            + 445.8558249365131 * jnp.exp(-energy / 0.1009243952792674)
        ) * 1e-18
    else:
        raise ValueError(
            "Excited-ion cross sections are available for N I, O I and Mg II."
        )
    return jnp.where((energy >= threshold) & (energy <= upper), value, 0.0)
