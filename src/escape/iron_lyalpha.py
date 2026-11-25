"""Local H/Fe II exchange using the C23.01 PRD-width/trapezoid approximation.

Altered source derived from Cloudy (C) 1978--2023 Gary J. Ferland and others;
see _licenses/Cloudy.txt. Sources: species2.cpp:setXtraRatesFe2 and
rt_escprob.cpp:RT_LineWidth. This is NOT the original Cloudy implementation.

This empirical closure is neither a full PRD transfer solution nor an LTE
detailed-balance model. The source's hard Fe optical-depth selection is
retained explicitly. Direct Fe exchange must not also enter continuum OTS.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .line_escape_probability import prd_escape

_WIDTH_TO_WAVENUMBER = 1.37194e-6 * 2 / 3
_FLOAT_EPSILON = 1.1920928955078125e-7


def _width_geometry(tau_in, tau_total, damping, doppler_width_cm_s, electron_cm3):
    """Static C23 geometry; evaluate PRD escape before its thermalization cap."""
    nearest = jnp.minimum(tau_in, tau_total - tau_in)
    escape = prd_escape(jnp.maximum(nearest, 0), damping)
    tau = jnp.minimum(nearest, 5.3e16 / jnp.maximum(electron_cm3, 1e-15))
    log_small = jnp.where(tau > 1e-3, jnp.log(jnp.maximum(tau, 1e-3)), -6.907755)
    small_a = 4.8 + 5.2 * tau + (4 * tau - 1) * log_small
    small_b = 6.5 * tau - log_small
    safe = jnp.maximum(tau, 1e-4)
    log_tau = jnp.log(safe)
    large_a = (
        1
        + 2 * log_tau / (1 + 0.3 * damping * safe) ** 0.6667
        + (6.5 * damping * safe) ** 0.333
    )
    large_b = 1.6 + 1.5 / (1 + 0.20 * damping * safe)
    geometry = (
        2
        * doppler_width_cm_s
        * 0.8862
        * jnp.where(tau <= 20, small_a / small_b, large_a / large_b)
    )
    active = (nearest >= 1e-3) & (tau / 100 >= _FLOAT_EPSILON)
    return jnp.where(active, geometry, 0), escape


def static_lyalpha_width(
    tau_in,
    tau_total,
    damping,
    doppler_width_cm_s,
    electron_cm3,
    electron_escape_probability,
    destruction_probability,
):
    """C23 static full line width [cm/s], including its finite-depth guards.

    This uses the PRD width prescription, not the Hummer--Kunasz Ly-alpha
    escape probability used in the hydrogen level balance. All arguments
    broadcast; the ballistic-wind branch of Cloudy is not included.
    """
    geometry, escape = _width_geometry(
        tau_in, tau_total, damping, doppler_width_cm_s, electron_cm3
    )
    loss = escape + electron_escape_probability + destruction_probability
    width = geometry * (1 - jnp.minimum(1, loss))
    return jnp.where(loss >= 1 - 100 * _FLOAT_EPSILON, 0, width)


def _trapezoid(width_cm_s, wavenumber_cm):
    half_width = width_cm_s[:, None] * _WIDTH_TO_WAVENUMBER
    offset = jnp.abs(wavenumber_cm[None, :] - 82259.0)
    value = jnp.clip(2 - offset / jnp.where(half_width > 0, half_width, 1), 0, 1)
    return jnp.where(half_width > 0, value, 0)


def _extra_rates(width_cm_s, occupation, aa, ratio, wn, column):
    profile = _trapezoid(width_cm_s, wn) * (column > 1)
    stimulated = aa * occupation[:, None] * profile
    return stimulated * ratio, stimulated, profile


def _prefix_probability(width, offsets, cumulative, moments):
    """Approximate a signed trapezoid sum using prepared prefix moments."""
    half = width * _WIDTH_TO_WAVENUMBER
    first = jnp.searchsorted(offsets, half, side="right")
    second = jnp.searchsorted(offsets, 2 * half, side="right")

    def at(values, index):
        return jnp.take_along_axis(values, index[:, None], axis=1)[:, 0]

    c1, c2 = at(cumulative, first), at(cumulative, second)
    d1, d2 = at(moments, first), at(moments, second)
    value = 2 * c2 - c1 - (d2 - d1) / jnp.where(half > 0, half, 1)
    return jnp.where(half > 0, value, 0)


def _certified_bisection(function, approximate, initial, *, allow_certificate=True):
    """Certify a monotone residual's dyadic bracket, or use the original solve."""
    def candidate_step(_, bracket):
        left, right = bracket
        middle = (left + right) / 2
        negative = approximate(middle) < 0
        return jnp.where(negative, middle, left), jnp.where(negative, right, middle)

    left, right = jax.lax.fori_loop(
        0, 48, candidate_step, (jnp.zeros_like(initial), jnp.ones_like(initial))
    )
    # Prefix cancellation can move the candidate. Check its adjacent original
    # grid intervals, using the original sum and its strict-negative convention.
    interval = right - left
    outer_left = jnp.maximum(0, left - interval)
    outer_right = jnp.minimum(1, right + interval)
    f0, f1 = function(outer_left), function(left)
    f2, f3 = function(right), function(outer_right)
    below, above = (f1 >= 0) & (left > 0), (f2 < 0) & (right < 1)
    lo = jnp.where(below, outer_left, jnp.where(above, right, left))
    hi = jnp.where(below, left, jnp.where(above, outer_right, right))
    flo = jnp.where(below, f0, jnp.where(above, f2, f1))
    fhi = jnp.where(below, f1, jnp.where(above, f3, f2))
    certified = jnp.all(
        jnp.asarray(allow_certificate)
        & ((flo < 0) | ((lo == 0) & (flo == 0))) & (fhi >= 0)
    )
    start = (
        jnp.where(certified, 48, 0),
        jnp.where(certified, lo, jnp.zeros_like(lo)),
        jnp.where(certified, hi, jnp.ones_like(hi)),
    )

    def fallback(values):
        count, lo, hi = values
        middle = (lo + hi) / 2
        negative = function(middle) < 0
        return count + 1, jnp.where(negative, middle, lo), jnp.where(negative, hi, middle)

    # Unlike a vmapped cond, this loop skips the old 48 sums when every node
    # is certified. Failed candidates retain the original bounded solve.
    _, lo, hi = jax.lax.while_loop(lambda values: values[0] < 48, fallback, start)
    return (lo + hi) / 2


class IronLyAlphaExchange(NamedTuple):
    upward_s: object
    stimulated_downward_s: object
    hydrogen_upward_s: object
    hydrogen_downward_s: object
    net_photons_cm3_s: object
    redistribution_heat_erg_cm3_s: object
    width_cm_s: object
    width_residual: object
    fe_destruction_probability: object
    selection_distance: object
    valid: object


def iron_lyalpha_exchange(
    *,
    lower_population_cm3,
    upper_population_cm3,
    einstein_a_s,
    weight_ratio,
    wavenumber_cm,
    transition_energy_erg,
    total_optical_depth,
    hydrogen_ground_cm3,
    hydrogen_2p_cm3,
    hydrogen_einstein_a_s,
    hydrogen_transition_energy_erg,
    hydrogen_tau_in,
    hydrogen_tau_total,
    hydrogen_damping,
    hydrogen_doppler_width_cm_s,
    electron_cm3,
    hydrogen_escape_probability,
    electron_escape_probability,
    continuum_destruction_probability,
    wavenumber_order=None,
):
    """Return coupled local rates and a separate photon/energy exchange ledger.

    Fe metadata have shape [line], populations/columns [cell,line], and H
    quantities [cell]. Columns must follow the selected atmosphere's geometry.
    The C23 ``TauTot > 1`` selection is discontinuous; ``selection_distance``
    reports proximity for lines inside the maximum possible profile (infinity
    if none are present) and is not a global NUTS guarantee.

    Width and signed Fe destruction are solved together by a scalar implicit
    root in each cell. Stimulated reverse exchange is retained. The explicit
    domain gate requires nonnegative total destruction and a positive slope
    bound; unsupported states return NaN rates. With signed coefficients the
    discontinuous thermalization cap can still admit multiple roots. Such
    states retain the original bounded branch selection and require separate
    physical appraisal. No negative exchange is clipped to zero.

    Add the returned H and Fe rates to their respective level matrices. Do
    not add ``net_photons`` to the H continuum OTS field. The redistribution
    term balances the supplied atomic level gaps; it is a declared local
    energy closure, not a derivation of recoil/PRD thermalization.

    Optional ``wavenumber_order`` is a host-prepared permutation sorting
    abs(wavenumber_cm - 82259). It accelerates only a certified primal
    candidate; the implicit derivative and all ledgers use the original sum.
    """
    nl, nu = jnp.asarray(lower_population_cm3), jnp.asarray(upper_population_cm3)
    aa, ratio = jnp.asarray(einstein_a_s), jnp.asarray(weight_ratio)
    wn, energy = jnp.asarray(wavenumber_cm), jnp.asarray(transition_energy_erg)
    column = jnp.asarray(total_optical_depth)
    n1, n2 = jnp.asarray(hydrogen_ground_cm3), jnp.asarray(hydrogen_2p_cm3)
    denominator = 3 * n1 - n2
    # Algebraic N_H/n2 has a finite limit at n2=0; no photon floor is added.
    occupation_per_2p = jnp.maximum(
        0, 1 - hydrogen_escape_probability - electron_escape_probability
    ) / jnp.where(denominator > 0, denominator, 1)
    occupation = occupation_per_2p * n2
    selected = column > 1
    coefficient = aa * (nl * ratio - nu) * occupation_per_2p[:, None]
    coefficient = jnp.where(selected, coefficient / hydrogen_einstein_a_s, 0)
    geometry, escape_prd = _width_geometry(
        hydrogen_tau_in,
        hydrogen_tau_total,
        hydrogen_damping,
        hydrogen_doppler_width_cm_s,
        electron_cm3,
    )

    def width_from_destruction(destruction):
        loss = escape_prd + electron_escape_probability + destruction
        return jnp.where(
            loss >= 1 - 100 * _FLOAT_EPSILON,
            0,
            geometry * (1 - jnp.minimum(1, loss)),
        )

    maximum = width_from_destruction(jnp.zeros_like(n1))
    scale = jnp.where(maximum > 0, maximum, 1)
    offset = jnp.abs(wn - 82259.0)
    possible = offset < 2 * _WIDTH_TO_WAVENUMBER * maximum[:, None]
    # A signed contribution can cross the source's discontinuous width-zero
    # cap in reverse. The existing slope guard does not prove global uniqueness
    # across that jump. Preserve the old branch selection in every such case.
    monotone = jnp.all(~possible | (coefficient >= 0), axis=1)
    order = (jnp.argsort(offset, stable=True)
             if wavenumber_order is None else jnp.asarray(wavenumber_order))
    if order.shape != wn.shape:
        raise ValueError("The prepared wavenumber order must match the line array.")
    offsets = jnp.abs(wn[order] - 82259.0)
    # Node-dependent occupation factors stay outside these shared cell sums.
    base = jnp.where(selected, aa * (nl * ratio - nu) / hydrogen_einstein_a_s, 0)
    sorted_base = base[:, order]
    cumulative = jnp.pad(jnp.cumsum(sorted_base, axis=1), ((0, 0), (1, 0)))
    moments = jnp.pad(jnp.cumsum(sorted_base * offsets, axis=1), ((0, 0), (1, 0)))

    def approximate_residual(fraction):
        probability = occupation_per_2p * _prefix_probability(
            fraction * scale, offsets, cumulative, moments
        )
        return fraction - width_from_destruction(
            continuum_destruction_probability + probability
        ) / scale

    def probability(fraction):
        return jnp.sum(coefficient * _trapezoid(fraction * scale, wn), axis=1)

    def residual(fraction):
        return (
            fraction
            - width_from_destruction(
                continuum_destruction_probability + probability(fraction)
            )
            / scale
        )

    def solve(function, _):
        fraction = _certified_bisection(
            function, approximate_residual, n1, allow_certificate=monotone
        )
        return jnp.where(maximum > 0, fraction, 0)

    fraction = jax.lax.custom_root(
        residual,
        jnp.zeros_like(n1),
        solve,
        lambda linear, value: value / linear(jnp.ones_like(value)),
    )
    width = fraction * scale
    upward, stimulated, profile = _extra_rates(width, occupation, aa, ratio, wn, column)
    absorbed, emitted = nl * upward, nu * stimulated
    net = absorbed - emitted
    hydrogen_down = jnp.sum(nl * aa * ratio * profile, axis=1) * occupation_per_2p
    hydrogen_up = jnp.sum(emitted, axis=1) / jnp.where(n1 > 0, n1, 1)
    destruction = probability(fraction)
    error = residual(fraction)
    # A negative line coefficient can broaden the profile. Bound its most
    # negative slope over all widths, rather than dropping stimulated gain.
    negative_slope = jnp.sum(
        jnp.where(possible, jnp.minimum(coefficient, 0), 0)
        * 4
        * _WIDTH_TO_WAVENUMBER
        / jnp.where(offset > 0, offset, 1),
        axis=1,
    )
    unique = 1 + geometry * negative_slope > 0
    valid = (
        (n1 > 0)
        & (n2 >= 0)
        & (denominator > 0)
        & (electron_cm3 >= 0)
        & jnp.isfinite(
            n1
            + n2
            + electron_cm3
            + hydrogen_einstein_a_s
            + hydrogen_transition_energy_erg
        )
        & (hydrogen_tau_in >= 0)
        & (hydrogen_tau_total >= hydrogen_tau_in)
        & (hydrogen_damping > 0)
        & (hydrogen_doppler_width_cm_s > 0)
        & (hydrogen_escape_probability >= 0)
        & (electron_escape_probability >= 0)
        & (continuum_destruction_probability >= 0)
        & (continuum_destruction_probability + destruction >= 0)
        & jnp.all((nl >= 0) & (nu >= 0), axis=1)
        & jnp.all(jnp.isfinite(nl) & jnp.isfinite(nu) & jnp.isfinite(column), axis=1)
        & jnp.all((aa >= 0) & (ratio > 0) & (wn > 0) & (energy > 0))
        & jnp.all(
            jnp.isfinite(aa)
            & jnp.isfinite(ratio)
            & jnp.isfinite(wn)
            & jnp.isfinite(energy)
        )
        & (hydrogen_einstein_a_s > 0)
        & (hydrogen_transition_energy_erg > 0)
        & jnp.all((offset > 0) | (coefficient == 0), axis=1)
        & (residual(jnp.ones_like(n1)) >= 0)
        & unique
        & jnp.isfinite(error)
        & (jnp.abs(error) < 1e-10)
    )
    def mask(value):
        return jnp.where(valid, value, jnp.nan)

    return IronLyAlphaExchange(
        jnp.where(valid[:, None], upward, jnp.nan),
        jnp.where(valid[:, None], stimulated, jnp.nan),
        mask(hydrogen_up),
        mask(hydrogen_down),
        jnp.where(valid[:, None], net, jnp.nan),
        mask(jnp.sum((hydrogen_transition_energy_erg - energy) * net, axis=1)),
        mask(width),
        error,
        destruction,
        jnp.min(jnp.where(possible, jnp.abs(column - 1), jnp.inf), axis=1),
        valid,
    )
