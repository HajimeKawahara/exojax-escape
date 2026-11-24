"""Differentiable adaptation of Cloudy C23.01 line operators.

Altered source derived from Cloudy (C) 1978--2023 Gary J. Ferland and others;
see _licenses/Cloudy.txt. This is NOT the original Cloudy implementation.
Sources: rt_escprob.cpp, rt_continuum_shield_fcn.cpp, rt_line_one.cpp,
iso_level.cpp, rt_stark.cpp, opacity_createall.cpp and rt_ots.cpp in C23.01.
Static plane slabs and the original unsaturated gain branch (-30<tau<0)
only; no Sobolev or continuum-transport solve is supplied here. Columns and
multiplet overlap opacities are external. External continuum occupations
must exclude the trapped local line field.

Cloudy tau is the coefficient multiplying H(a,x), with x=(nu-nu0)/Delta_nu_D
and Delta_nu_D=nu0*v_D/c. Thus Hummer's integrated tau is sqrt(pi)*tau.
Internal damping uses the UPPER lifetime, unlike the separate physical
observer's upper+lower lifetime. All rates are per second; all opacities
are inverse cm. Each kernel is piecewise differentiable away from fit joins.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

SQRTPI = math.sqrt(math.pi)
SIGMA_THOMSON = 6.6524587321e-25
PLANCK = 6.62607015e-27
LYA, CORE, WING, PRD = 0, 1, 2, 3


def maser_escape(tau):
    """C23 escmase on -30<tau<=0; no saturated-maser model is implied."""
    safe = jnp.minimum(tau, -0.1)
    value = jnp.where(tau > -0.1, 1 - tau * (0.5 - tau / 6), -jnp.expm1(-safe) / safe)
    return jnp.where((tau > -30) & (tau <= 0), value, jnp.nan)


def doppler_core_escape(tau, *, core_escape_model="cloudy"):
    """Doppler K2 escape, with an explicit continuous positive-depth control.

    ``cloudy`` retains C23's piecewise Hummer (1981) approximation exactly.
    ``continuous`` extends its accurate tau*log(tau) expression to small
    positive depths and joins the moderate/large fits with a C1 blend on
    10 < sqrt(pi)*tau < 12. The maser prescription is unchanged. This does
    not regularize the physical logarithmic derivative at tau=0: the value
    there is one, with a constant-branch tangent for a fixed zero column.
    """
    if core_escape_model not in ("cloudy", "continuous"):
        raise ValueError("Choose cloudy or continuous Doppler core escape.")
    tau = jnp.asarray(tau)
    t = (jnp.maximum(tau * SQRTPI, 0.01) if core_escape_model == "cloudy"
         else jnp.where(tau > 0, jnp.minimum(tau * SQRTPI, 12.0), 1.0))
    a = (1.0, -0.1117897, -0.1249099917, -0.009136358767, -0.0003370280896)
    b = (
        1.0,
        0.1566124168,
        0.009013261660,
        0.0001908481163,
        -1.547417750e-7,
        -6.657439727e-9,
    )
    c = (1.0, 19.15049608, 100.7986843, 129.5307533, -31.43372468)
    d = (1.0, 19.68910391, 110.2576321, 169.4911399, -16.69969409, -36.664480000)
    moderate = t / 2.5066283 * jnp.log(t / SQRTPI) + jnp.polyval(
        jnp.array(a[::-1]), t
    ) / jnp.polyval(jnp.array(b[::-1]), t)
    large_t = (jnp.maximum(t, 11.0) if core_escape_model == "cloudy"
               else jnp.maximum(tau * SQRTPI, 10.0))
    log_t = jnp.log(large_t / SQRTPI)
    inv = 1 / log_t
    large = (
        jnp.polyval(jnp.array(c[::-1]), inv)
        / jnp.polyval(jnp.array(d[::-1]), inv)
        / (2 * large_t * jnp.sqrt(log_t))
    )
    if core_escape_model == "cloudy":
        value = jnp.where(
            tau * SQRTPI < 0.01, 1 - 2 * tau * SQRTPI,
            jnp.where(t <= 11.0, moderate, large),
        )
    else:
        fraction = jnp.clip((tau * SQRTPI - 10.0) / 2.0, 0.0, 1.0)
        blend = fraction**2 * (3 - 2 * fraction)
        value = jnp.where(tau > 0, (1 - blend) * moderate + blend * large, 1.0)
    return jnp.where(tau >= 0, value, maser_escape(tau))


def crd_wing_escape(tau, damping, *, core_escape_model="cloudy"):
    """C23 esc_CRDwing_1side finite-damping fit (source estimates +/-20%)."""
    a = jnp.asarray(damping)
    positive = jnp.maximum(tau, 0.0)
    scale = a * (1 + a + positive) / ((1 + a) ** 2 + a * positive)
    wing = scale * jnp.sqrt(a) / jnp.sqrt(a + 2.25 * SQRTPI * positive)
    value = doppler_core_escape(positive, core_escape_model=core_escape_model) * (1 - wing) + wing
    return jnp.where(a > 0, jnp.where(tau >= 0, value, maser_escape(tau)), jnp.nan)


def prd_escape(tau, damping):
    """C23 esc_PRD_1side; not the special Ly-alpha fit."""
    a = jnp.asarray(damping)
    at = jnp.maximum(a * tau, 0.0)
    root = jnp.sqrt(jnp.maximum(at, 1e-300))
    b = 1.6 + 3 * (2 * a) ** (-0.12) * jnp.where(
        at > 1, 1 / (1 + at), root / (1 + root)
    )
    value = 1 / (1 + jnp.minimum(6.0, b) * tau)
    return jnp.where(a > 0, jnp.where(tau >= 0, value, maser_escape(tau)), jnp.nan)


def lya_one_side(tau, beta, albedo=0.0):
    """C23 Hummer--Kunasz (1980) fit; beta is kappa_c/(sqrt(pi)*kappa_l+kappa_c)."""
    t = tau * SQRTPI
    escape0 = jnp.clip(1 / ((0.6451 + t) * (0.47 + 1.08 / (1 + 7.3e-6 * t))), 0.0, 1.0)
    log_t = jnp.log10(jnp.clip(t, 1e-300, 1e8))
    log_t = jnp.where(t > 0, log_t, 0.0)
    tau_c = jnp.minimum(2.0, beta * t)
    factor = jnp.minimum(
        1.0,
        10 ** (-(-1.25 + 0.475 * log_t) * tau_c + (-0.485 + 0.1615 * log_t) * tau_c**2),
    )
    factor = jnp.where(tau_c > 1e-3, factor, 1.0)
    escape = escape0 * factor
    destruction = beta / (0.30972 - jnp.minimum(0.28972, 0.03541667 * log_t))
    destruction = jnp.clip(destruction, 0.0, 1 - escape)
    # DEST0=SMALLFLOAT contributes only a numerical floor in scattering-only
    # gas; omit that ~1e-38 floor while retaining the physical albedo factor.
    destruction *= 1 - albedo
    valid = (tau >= 0) & (beta >= 0) & (beta <= 1) & (albedo >= 0) & (albedo <= 1)
    return jnp.where(valid, escape, jnp.nan), jnp.where(valid, destruction, jnp.nan)


def _federman_core_coefficients(shielding_model):
    if shielding_model == "continuous_core":
        c1 = math.exp(-2 * 0.66666) * 2**1.25
        c2 = c1 * 10 ** (1.149999976158142 - 1.25)
        c3 = c2 * 100 ** (1.0666999816894531 - 1.149999976158142)
        return c1, c2, c3
    if shielding_model == "cloudy":
        return 0.638, 0.505, 0.344
    raise ValueError("Choose cloudy or continuous_core shielding.")


def _federman_cap_depths(damping, shielding_model):
    """Locate unity caps on each smooth branch, with implicit derivatives.

    The original fit can cross unity twice around a discontinuous join.
    Inactive roots duplicate existing segment boundaries. Individual root
    slots can jump at activation; only the resulting integral is meaningful
    there. No source fit or cap is smoothed.
    """
    a = jnp.asarray(damping)
    c1, c2, c3 = _federman_core_coefficients(shielding_model)
    left = jnp.broadcast_to(
        jnp.array([0., 2., 10., 100.]).reshape((4,) + (1,) * a.ndim),
        (4,) + a.shape,
    )
    right = jnp.stack([jnp.full_like(a, value) for value in (2., 10., 100.)]
                      + [jnp.maximum(100., 4 * a / jnp.sqrt(jnp.pi))])
    span = right - left
    wing_scale = 3.02 * (a * 1e3)**(-.064)

    def excess(tau):
        core_minus_one = jnp.stack([
            jnp.expm1(-.66666 * tau[0]),
            c1 * tau[1]**(-1.25) - 1,
            c2 * tau[2]**(-1.149999976158142) - 1,
            c3 * tau[3]**(-1.0666999816894531) - 1,
        ])
        wings = a / wing_scale / jnp.sqrt(
            .78540 + tau * jnp.sqrt(jnp.pi) * a / wing_scale**2)
        return core_minus_one + wings

    active = (excess(left) > 0) & (excess(right) < 0) & (span > 0)

    def residual(fraction):
        return jnp.where(active, excess(left + span * fraction), fraction)

    def solve(function, initial):
        def step(_, bounds):
            lo, hi = bounds
            middle = (lo + hi) / 2
            positive = function(middle) > 0
            return jnp.where(positive, middle, lo), jnp.where(positive, hi, middle)

        lo, hi = jax.lax.fori_loop(
            0, 64, step, (jnp.zeros_like(initial), jnp.ones_like(initial)))
        return jnp.where(active, (lo + hi) / 2, 0.)

    fraction = jax.lax.custom_root(
        residual, jnp.zeros_like(left), solve,
        lambda linear, rhs: rhs / linear(jnp.ones_like(rhs)))
    return left + span * fraction


def federman_shield(tau, damping, delta_tau=0.0, *, average_zone=True,
                    shielding_model="cloudy"):
    """Corrected C23 Federman et al. (1979) shielding and default AVG_DEPTH.

    tau is Cloudy's TauCon (line self-shielding column), NOT the separately
    attenuated continuum. delta_tau includes line plus continuum absorption
    across the zone. Historical buggy wing factor is intentionally absent.
    For tau<-1 and positive zone depth, C23's AVG_DEPTH branch returns zero
    numerically; retain that reference approximation. At tau=-1 use its
    continuous zero limit, without claiming a differentiable physical maser
    model at this branch boundary.

    ``continuous_core`` is a declared numerical variant: each positive-tau
    power law meets the preceding branch at tau=2,10,100. It changes the
    source core by at most 1.731734 percent, without altering its exponents
    or wings. Kinks and the separate source jump at tau=0 remain; this is
    not a globally smooth shielding model or exact C23 reproduction.
    """
    c1, c2, c3 = _federman_core_coefficients(shielding_model)
    # Power-law branches are only used for tau>=2. Keeping their inactive
    # arguments in that domain avoids overflowing derivatives at tau<=0.
    t = jnp.maximum(tau, 2.0)
    # realnum exponents in C23 are single precision constants.
    core = jnp.where(
        tau < 2.0,
        jnp.exp(-tau * 0.66666),
        jnp.where(
            tau < 10.0,
            c1 * t ** (-1.25),
            jnp.where(
                tau < 100.0,
                c2 * t ** float(-1.149999976158142),
                c3 * t ** float(-1.0666999816894531),
            ),
        ),
    )
    t1 = 3.02 * (damping * 1e3) ** (-0.064)
    u1sq = jnp.maximum(tau, 0.0) * SQRTPI * damping / t1**2
    wings = damping / t1 / jnp.sqrt(0.78540 + u1sq)
    point = jnp.where(tau >= 0, jnp.minimum(1.0, core + wings), core + wings)
    # C23 rt_continuum_shield_fcn.cpp:176-180 handles negative dTauRel
    # explicitly: MAX2(dTau/(1+tau), SMALLFLOAT). Its log(1+SMALLFLOAT)
    # rounds to zero in double precision. Without this branch an inverted
    # metal line can produce log1p(relative<-1), blocking the coupled root.
    positive_denominator = tau > -1
    relative = delta_tau / jnp.where(positive_denominator, 1 + tau, 1.0)
    safe = jnp.where(jnp.abs(relative) < 1e-8, 1e-8, relative)
    correction = jnp.where(
        jnp.abs(relative) < 1e-8,
        1 - relative / 2 + relative**2 / 3,
        jnp.log1p(safe) / safe,
    )
    correction = jnp.where(positive_denominator, correction, 0.0)
    value = point * jnp.where(average_zone & (delta_tau > 1e-3), correction, 1.0)
    return jnp.where(
        (tau > -30) & (damping > 0) & jnp.isfinite(delta_tau), value, jnp.nan
    )


def hydrogen_rayleigh_cross_section(energy_ryd):
    """C23 rayleh (DampOnFac=1), cm² per ground-state neutral H.

    opacity_createall.cpp:936, citing Gavrila (1967), Phys. Rev. 163, 147,
    and Mihalas damping. This is the C23 approximation, not a new fit.
    Its resonant cap is deliberately preserved for operator comparison.
    """
    energy = jnp.asarray(energy_ryd)
    low = 8.41e-25 * energy**4 + 3.37e-24 * energy**6
    frequency_ryd = 3.2898419602508e15
    resonance = jnp.maximum(
        1e-24,
        1.788e5 / (frequency_ryd * jnp.maximum(0.001, jnp.abs(0.74959 - energy))) ** 2,
    )
    sigma = jnp.where(
        energy < 0.05,
        low,
        jnp.where(
            energy < 0.646,
            low + 4.71e-22 * energy**14,
            jnp.where(energy < 1.0, resonance, 0.0),
        ),
    )
    return jnp.where(energy >= 0, sigma, jnp.nan)


class LineOperators(NamedTuple):
    escape_probability: object
    destruction_probability: object
    electron_escape_probability: object
    pump_up_s: object
    stimulated_down_s: object
    radiative_down_s: object
    destroyed_photons_cm3_s: object


def hydrogen_stark_escape(n_upper, n_lower, electron_cm3, temperature_k, tau_in):
    """C23 rt_stark.cpp Puetter (1981) internal escape increment.

    Ly-alpha is explicitly disabled in C23. Other Lyman terms have factor
    1/2, subordinate lines do not, and every increment is capped at .01.
    Unlike Cloudy's once-per-zone cache this pure function follows current ne.
    It is not an observer Stark profile or a validated pressure-line shape.
    """
    ah = 6.9e-15 * electron_cm3 / temperature_k**0.75
    factor = 0.264 * ah**0.4 * (n_lower * n_upper) ** 1.2000000476837158
    probability = (
        factor * jnp.maximum(1.0, tau_in) ** (-0.75) * jnp.where(n_lower == 1, 0.5, 1.0)
    )
    probability = jnp.where(
        (n_lower == 1) & (n_upper == 2), 0.0, jnp.minimum(0.01, probability)
    )
    return jnp.where(
        (electron_cm3 >= 0) & (temperature_k > 0) & (tau_in > -30), probability, jnp.nan
    )


def line_operators(
    kind,
    *,
    tau_in,
    tau_total,
    tau_pump,
    damping,
    line_extinction_cm1,
    lower_absorption_cm1,
    continuum_absorption_cm1,
    electron_cm3,
    einstein_a_s,
    lower_weight,
    upper_weight,
    incident_occupation,
    outward_continuum_occupation,
    zone_width_cm,
    upper_population_cm3,
    albedo=0.0,
    multiplet_extinction_cm1=0.0,
    stark_escape_probability=0.0,
    include_electron_scattering=True,
    average_zone=True,
    shielding_model="cloudy",
    core_escape_model="cloudy",
):
    """Return C23-like effective line SE coefficients for a supplied local state.

    ``line_extinction`` contains n_l-n_u*g_l/g_u; ``lower_absorption`` contains
    n_l alone for electron scattering. For resolved nonoverlapping lines,
    kappa_l=sqrt(pi)*e^2/(m_e*c)*f*lambda*(n_l-n_u*g_l/g_u)/v_D.
    Supplied columns must be self-consistent with that state. Inverted
    transitions use the original bounded gain branch, requiring tau>-30.
    n_l*up - n_u*down enters line statistical equilibrium. Do NOT also add
    A times the trapped local line occupation: escape accounts for it here.
    Destroyed photons require absorber-specific OTS return, not immediate heat.
    """
    tau_out = jnp.abs(tau_total - tau_in)
    tau_out = jnp.where(
        (kind == CORE) & ((tau_total < 0) | (tau_in < 0)),
        jnp.minimum(tau_total, tau_in),
        tau_out,
    )
    c = continuum_absorption_cm1
    beta_lya = c / (
        SQRTPI * jnp.where(line_extinction_cm1 > 0, line_extinction_cm1, 1.0) + c
    )
    effective = jnp.where(
        multiplet_extinction_cm1 > 0, multiplet_extinction_cm1, line_extinction_cm1
    )
    beta = c / (SQRTPI * jnp.where(effective > 0, effective, 1.0) + c)
    lya_in, dest_in = lya_one_side(jnp.maximum(tau_in, 0.0), beta_lya, albedo)
    lya_out, dest_out = lya_one_side(jnp.maximum(tau_out, 0.0), beta_lya, albedo)

    def one(t):
        return jnp.where(
            kind == CORE,
            doppler_core_escape(t, core_escape_model=core_escape_model),
            jnp.where(
                kind == WING,
                crd_wing_escape(t, damping, core_escape_model=core_escape_model),
                prd_escape(t, damping),
            ),
        )

    escape = jnp.where(
        kind == LYA, 0.5 * (lya_in + lya_out), 0.5 * (one(tau_in) + one(tau_out))
    )
    lya_dest = jnp.minimum(0.5 * (dest_in + dest_out), 1 - escape)
    escape = jnp.where(
        escape < 1, jnp.minimum(1.0, escape + stark_escape_probability), escape
    )
    fit = jnp.minimum(1e-3, 8.5 * beta)
    destruction = jnp.where(
        kind == LYA,
        lya_dest,
        jnp.where(
            (line_extinction_cm1 > 0) & (escape < 1),
            fit / (1 + fit) * (1 - escape),
            0.0,
        ),
    )
    electron_opacity = electron_cm3 * SIGMA_THOMSON
    lower_effective = jnp.where(
        multiplet_extinction_cm1 > 0, multiplet_extinction_cm1, lower_absorption_cm1
    )
    electron_escape = (
        electron_opacity
        / (electron_opacity + lower_effective)
        * jnp.maximum(0.0, 1 - escape - destruction)
    )
    electron_escape = jnp.where(include_electron_scattering, electron_escape, 0.0)
    shielding = federman_shield(
        tau_pump,
        damping,
        (line_extinction_cm1 + c) * zone_width_cm,
        average_zone=average_zone,
        shielding_model=shielding_model,
    )
    pump = (
        einstein_a_s
        * upper_weight
        / lower_weight
        * shielding
        * (incident_occupation + outward_continuum_occupation)
    )
    stimulated = pump * lower_weight / upper_weight
    down = einstein_a_s * (escape + destruction + electron_escape) + stimulated
    destroyed = upper_population_cm3 * einstein_a_s * destruction
    valid = (
        (kind >= LYA)
        & (kind <= PRD)
        & (tau_in > -30)
        & (tau_out > -30)
        & (
            (kind != LYA)
            | ((line_extinction_cm1 > 0) & (tau_in >= 0) & (tau_total >= tau_in))
        )
        & (stark_escape_probability >= 0)
        & (lower_absorption_cm1 > 0)
        & (c >= 0)
        & (electron_cm3 >= 0)
        & (zone_width_cm >= 0)
        & (incident_occupation >= 0)
        & (outward_continuum_occupation >= 0)
        & (einstein_a_s > 0)
        & (lower_weight > 0)
        & (upper_weight > 0)
        & (upper_population_cm3 >= 0)
    )
    return LineOperators(
        *(
            jnp.where(valid, x, jnp.nan)
            for x in (
                escape,
                destruction,
                electron_escape,
                pump,
                stimulated,
                down,
                destroyed,
            )
        )
    )


def line_cell_quadrature(order):
    """Return Gauss positions and weights on a cell's illuminated-to-dark axis."""
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("Line quadrature order must be a positive integer.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return jnp.asarray((nodes + 1) / 2), jnp.asarray(weights / 2)


def _optical_depth_cell_segments(tau_front, tau_total, delta_tau,
                                 continuum_depth, beta_lya, escape_join_depth,
                                 shielding_join_depth=None):
    """Shared segment geometry; metal scans avoid materializing all nodes."""
    front, total, depth, continuum, beta = jnp.broadcast_arrays(
        tau_front, tau_total, delta_tau, continuum_depth, beta_lya)
    if front.ndim != 1:
        raise ValueError("Optical-depth quadrature requires one value per cell.")
    safe_depth = jnp.where(depth != 0, depth, 1.)
    critical = jnp.where(beta > 0, 1e-3/(jnp.sqrt(jnp.pi)*jnp.where(beta > 0, beta, 1.)), -1.)
    critical = (critical[None, :] if escape_join_depth is None else
                jnp.asarray(escape_join_depth))
    if critical.ndim != 2 or critical.shape[1] != len(front):
        raise ValueError("Escape joins must have shape [join,cell].")
    # Shielding depends only on the illuminated column; Ly-alpha escape
    # contains both faces. Duplicate clipped endpoints have zero measure.
    joins = jnp.array([2., 10., 100.])[:, None]
    boundaries = [
        jnp.zeros((1, len(front))),
        jnp.clip((joins-front)/safe_depth, 0., 1.),
        jnp.clip((critical-front)/safe_depth, 0., 1.),
        jnp.clip((total-critical-front)/safe_depth, 0., 1.),
        jnp.ones((1, len(front))),
    ]
    if shielding_join_depth is not None:
        boundaries.append(jnp.clip((shielding_join_depth-front)/safe_depth, 0., 1.))
    breaks = jnp.sort(jnp.concatenate(boundaries), axis=0)
    left, span = breaks[:-1], jnp.diff(breaks, axis=0)
    local_depth = span*depth
    near, end = front+left*depth, front+(left+span)*depth
    near_squared = jnp.minimum(near**2, end**2)
    far_squared = jnp.minimum((total-front-left*depth)**2,
                              (total-front-(left+span)*depth)**2)
    scale = jnp.sqrt(1+local_depth**2/(1+near_squared)
                     +local_depth**2/(1+far_squared)+(span*continuum)**2)
    return left, span, scale


def optical_depth_cell_quadrature(tau_front, tau_total, delta_tau,
                                  continuum_depth, beta_lya, order=8, *,
                                  escape_join_depth=None):
    """Return cellwise nodes/weights resolving both optical-depth faces.

    Arrays have shape [cell]; output has shape [node,cell]. Split at the
    source shielding joins and the Ly-alpha beta*sqrt(pi)*tau=.001 switch.
    Moving endpoints, nodes and Jacobians remain differentiable: omitting
    their derivatives loses the integral's moving-discontinuity term.
    Each segment uses a symmetric log1p/expm1 map. Positive weights are
    normalized; the third result exposes the unnormalized constant error.
    Optional ``escape_join_depth`` [join,cell] replaces the Ly-alpha switch
    by supplied escape-kernel joins, also split on both faces.
    ``continuum_depth`` includes the smallest angular direction if needed.
    This is a quadrature rule, not an accuracy certificate for an atmosphere.
    """
    x, w = line_cell_quadrature(order)
    left, span, scale = _optical_depth_cell_segments(
        tau_front, tau_total, delta_tau, continuum_depth, beta_lya, escape_join_depth)
    cells = left.shape[1]
    length = jnp.log1p(scale)
    local = .5*jnp.expm1(x[:, None, None]*length)/scale
    jacobian = .5*length*jnp.exp(x[:, None, None]*length)/scale
    positions = jnp.concatenate((left+span*local, left+span*(1-local)), axis=0)
    half = w[:, None, None]*span*jacobian
    weights = jnp.concatenate((half, half), axis=0)
    positions = positions.reshape(-1, cells)
    weights = weights.reshape(-1, cells)
    mass = weights.sum(axis=0)
    return positions, weights/mass, mass-1


def cell_averaged_line_operators(
    kind,
    *,
    line_optical_depth_increment,
    quadrature_order=4,
    positions=None,
    weights=None,
    node_incident_occupation=None,
    node_outward_continuum_occupation=None,
    node_stark_escape_probability=None,
    shielding_model="cloudy",
    core_escape_model="cloudy",
    return_nodes=False,
    **arguments,
):
    """Average the supplied line closure consistently inside a uniform cell.

    This is an explicit spatial discretization, not Cloudy's front-face
    approximation. Columns at fraction u are tau_front + u*delta_tau. The
    point shielding function is averaged with the escape, destruction and
    stimulated terms; C23 AVG_DEPTH is not applied a second time. Constant
    populations make averaged destruction exactly n_upper*A*mean(Pdest).

    Optional positions/weights have a leading node axis and broadcast to
    the line array; supply [node,cell,1] for shared cellwise line nodes.
    Optional node fields have the same leading axis. Without
    them the supplied continuum occupation/Stark term is held constant.
    Callers with appreciable continuum depth must supply the spatially
    varying occupation, rather than multiply independent field averages.
    The original -30 gain domain is required at both cell endpoints as well
    as every quadrature node. No saturated-maser extension is introduced.
    ``return_nodes=True`` also returns the guarded point operators, for a
    coupled process that must use the same quadrature before averaging.
    """
    if "average_zone" in arguments:
        raise ValueError("Cell quadrature uses point shielding, not AVG_DEPTH.")
    if (positions is None) != (weights is None):
        raise ValueError("Supply both line positions and weights.")
    supplied = positions is not None
    if not supplied:
        positions, weights = line_cell_quadrature(quadrature_order)
        gauss_weights = weights
    else:
        positions, weights = jnp.asarray(positions), jnp.asarray(weights)
        if positions.ndim < 1 or positions.shape != weights.shape:
            raise ValueError("Line positions and weights need identical node shapes.")
    node_count = positions.shape[0]
    sampled = {
        "incident_occupation": node_incident_occupation,
        "outward_continuum_occupation": node_outward_continuum_occupation,
        "stark_escape_probability": node_stark_escape_probability,
    }
    for values in sampled.values():
        if values is not None and (
            jnp.ndim(values) == 0 or jnp.shape(values)[0] != node_count
        ):
            raise ValueError("Node fields must have one leading entry per Gauss node.")
    front, pump_front, increment = jnp.broadcast_arrays(
        arguments["tau_in"], arguments["tau_pump"], line_optical_depth_increment)
    if not supplied:
        positions = positions.reshape((node_count,) + (1,) * front.ndim)
        weights = weights.reshape((node_count,) + (1,) * front.ndim)
    positions = jnp.broadcast_to(positions, (node_count,) + front.shape)
    weights = jnp.broadcast_to(weights, positions.shape)
    # One vectorized operator graph avoids copying the entire atomic closure
    # for every Gauss node during whole-atmosphere differentiation.
    values = {name: jnp.asarray(value)[None, ...]
              for name, value in arguments.items()}
    values["tau_in"] = front[None, ...] + positions * increment[None, ...]
    values["tau_pump"] = pump_front[None, ...] + positions * increment[None, ...]
    values.update({name: array for name, array in sampled.items() if array is not None})
    result = line_operators(jnp.asarray(kind)[None, ...], average_zone=False,
                           shielding_model=shielding_model,
                           core_escape_model=core_escape_model, **values)
    endpoint_valid = (
        jnp.all((positions >= 0) & (positions <= 1) & (weights >= 0), axis=0)
        & jnp.isclose(weights.sum(axis=0), 1., rtol=1e-12, atol=1e-14)
        &
        (jnp.minimum(front, front + increment) > -30)
        & (jnp.minimum(pump_front, pump_front + increment) > -30)
        & ((kind != CORE) | (arguments["tau_total"] > -30))
        & ((kind != LYA) | (
            (jnp.minimum(front, front + increment) >= 0)
            & (arguments["tau_total"] >= jnp.maximum(front, front + increment))
        ))
    )
    averaged = LineOperators(*(
        jnp.where(endpoint_valid,
                  (jnp.sum(weights*value, axis=0) if supplied else
                   jnp.tensordot(gauss_weights, value, axes=1)), jnp.nan)
        for value in result
    ))
    if return_nodes:
        nodes = LineOperators(*(jnp.where(endpoint_valid[None, ...], value, jnp.nan)
                                for value in result))
        return averaged, nodes
    return averaged


class OTSReturn(NamedTuple):
    absorbed_photons_cm3_s: object
    photoionization_s: object
    photoelectron_heating_erg_cm3_s: object
    unresolved_photons_cm3_s: object


def on_the_spot_return(
    destroyed_photons_cm3_s,
    frequency_hz,
    absorber_opacity_cm1,
    absorber_density_cm3,
    threshold_erg,
    total_absorption_cm1,
):
    """Partition locally destroyed photons by actual supplied absorption opacity.

    The last axis indexes absorbers; preceding axes index lines/layers. This
    realizes rt_ots.cpp's flux D/kappa_total before photoionization integrals.
    Only supplied bound-free absorbers count as ionizations. Missing opacity
    is returned as unresolved photons, never relabeled as heat. At zero total
    opacity destruction must be zero. Heating uses h*nu-chi, not h*nu.
    """
    total = jnp.asarray(total_absorption_cm1)
    opacity = jnp.asarray(absorber_opacity_cm1)
    density = jnp.asarray(absorber_density_cm3)
    threshold = jnp.asarray(threshold_erg)
    photon = PLANCK * jnp.asarray(frequency_hz)
    eligible = (photon[..., None] >= threshold) & (density > 0)
    flux = destroyed_photons_cm3_s / jnp.where(total > 0, total, 1.0)
    absorbed = flux[..., None] * jnp.where(eligible, opacity, 0.0)
    ionization = absorbed / jnp.where(density > 0, density, 1.0)
    heat = absorbed * jnp.maximum(photon[..., None] - threshold, 0.0)
    unresolved = destroyed_photons_cm3_s - absorbed.sum(axis=-1)
    valid = (
        (total >= 0)
        & (destroyed_photons_cm3_s >= 0)
        & jnp.all((opacity >= 0) & (density >= 0) & (threshold >= 0), axis=-1)
        & (opacity.sum(axis=-1) <= total * (1 + 1e-12))
        & ((total > 0) | (destroyed_photons_cm3_s == 0))
    )
    return OTSReturn(
        jnp.where(valid[..., None], absorbed, jnp.nan),
        jnp.where(valid[..., None], ionization, jnp.nan),
        jnp.where(valid[..., None], heat, jnp.nan),
        jnp.where(valid, unresolved, jnp.nan),
    )
