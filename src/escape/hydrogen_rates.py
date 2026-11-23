"""Rates for a truncated, local six-level hydrogen atom (cgs units).

The levels are 1s, 2s, 2p, 3s, 3p and 3d, with statistical populations only
within each fine-structure multiplet. The supported gas-temperature interval
is 0.5 eV / k_B to 14000 K by default. An explicit Cloudy-style low-T
extrapolation extends the approximation to the ionization-fit floor of 4000 K;
this is not additional validated collision data.

Atomic data: Lehmann et al. (2020), Appendix B, Tables 5--7 (CC0),
https://arxiv.org/html/2010.01042v1 . Electron ionization and the Kramers
bound-free approximation follow Garcia Munoz & Schneider (2019), Appendix,
https://arxiv.org/html/1910.00267v1 . Temperature-dependent electron/proton
2s--2p mixing follows Huang et al. (2023), Eqs. 12--13,
https://arxiv.org/html/2304.07352v1 . Reverse collision and bound-free rates
are set by detailed balance, not independently fitted.

This is an explicit n <= 3 closure: no recombination into higher levels,
no Case-B cascade correction, no n=3 l-mixing, and no radiation feedback.
Kramers cross sections omit Gaunt factors and l dependence; the ionization
fit for each principal shell is also shared by its l sublevels. Only electric
dipole bound-bound lines and spontaneous 2s two-photon decay are included;
induced/inverse two-photon processes are absent, so the complete atom does
not have exact blackbody LTE at arbitrary density.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .hydrogen_nlte import RateCoefficients

PLANCK_CGS = 6.62607015e-27
BOLTZMANN_CGS = 1.380649e-16
LIGHT_SPEED_CGS = 2.99792458e10
ELECTRON_MASS_CGS = 9.1093837139e-28
EV_CGS = 1.602176634e-12
_THERMAL_VOLUME_COEFFICIENT = (
    PLANCK_CGS**2 / (2 * np.pi * ELECTRON_MASS_CGS * BOLTZMANN_CGS)
) ** 1.5
LEVEL_NAMES = ("1s", "2s", "2p", "3s", "3p", "3d")
PRINCIPAL_QUANTUM_NUMBERS = np.array([1, 2, 2, 3, 3, 3])
STATISTICAL_WEIGHTS = np.array([2., 2., 6., 2., 6., 10.])
MIN_TEMPERATURE_K = 0.5 * EV_CGS / BOLTZMANN_CGS
MAX_TEMPERATURE_K = 14000.0
MIN_EXTRAPOLATED_TEMPERATURE_K = 4000.0

# NIST fine-structure data as tabulated in Lehmann et al., Table 5.
_FINE_G = np.array([2., 2., 2., 4., 2., 2., 4., 4., 6.])
_FINE_E_K = np.array([
    0., 118352.24454327, 118352.29531193, 118352.77097467,
    140269.55551305, 140269.57062165, 140269.71123621,
    140269.71149231, 140269.76322354,
])
_FINE_GROUP = np.array([0, 2, 1, 2, 4, 3, 5, 4, 5])
EXCITATION_ENERGY_ERG = np.bincount(
    _FINE_GROUP, weights=_FINE_G * _FINE_E_K
) / STATISTICAL_WEIGHTS * BOLTZMANN_CGS
IONIZATION_ENERGY_ERG = 13.5984345 * EV_CGS - EXCITATION_ENERGY_ERG

# Order: Ly alpha, Ly beta, and the three allowed H alpha nl branches.
LINE_UPPER = np.array([2, 4, 3, 4, 5])
LINE_LOWER = np.array([0, 0, 2, 1, 2])
LINE_EINSTEIN_A_S = np.array([
    (2 * 6.2649e8 + 4 * 6.2648e8) / 6,
    1.6725e8,
    4.2097e6 + 2.1046e6,
    (2 * 2.2449e7 + 4 * 2.2448e7) / 6,
    (4 * (1.0775e7 + 5.3877e7) + 6 * 6.4651e7) / 10,
])
LINE_FREQUENCY_HZ = (
    EXCITATION_ENERGY_ERG[LINE_UPPER] - EXCITATION_ENERGY_ERG[LINE_LOWER]
) / PLANCK_CGS
TWO_PHOTON_A_S = 8.2

# First three temperature columns of Table 7 suffice for the stated domain.
# Entries: upper fine-level index, lower index, Upsilon at kT=0.5,1,3 eV.
_FINE_UPSILON = np.array([
    [2, 1, .143, .176, .284],
    [3, 1, .260, .296, .326],
    [4, 1, .286, .353, .569],
    [5, 1, .0373, .0420, .0620],
    [5, 2, .807, .880, 1.19],
    [5, 3, .820, 1.02, 1.76],
    [5, 4, 1.61, 1.76, 2.38],
    [6, 1, .0651, .0696, .0776],
    [6, 2, .680, .740, .760],
    [6, 3, 1.38, 1.45, 2.28],
    [6, 4, 1.36, 1.48, 1.52],
    [7, 1, .0248, .0263, .0313],
    [7, 2, 1.76, 2.37, 4.92],
    [7, 3, .836, 1.23, 2.62],
    [7, 4, 3.52, 4.75, 9.84],
    [8, 1, .0747, .0840, .124],
    [8, 2, 1.61, 1.76, 2.38],
    [8, 3, 1.64, 2.03, 3.52],
    [8, 4, 3.23, 3.52, 4.76],
    [9, 1, .0373, .0395, .0469],
    [9, 2, 2.64, 3.56, 7.38],
    [9, 3, 1.25, 1.85, 3.94],
    [9, 4, 5.28, 7.12, 14.8],
])
_UPSILON = np.zeros((6, 6, 3))
for _entry in _FINE_UPSILON:
    _upper, _lower = _FINE_GROUP[_entry[:2].astype(int) - 1]
    _UPSILON[_upper, _lower] += _entry[2:]



def saha_factors(temperature_k):
    """Return n_i/(n_e n_p) in cm^3 for each level in Saha equilibrium."""
    thermal_volume = _THERMAL_VOLUME_COEFFICIENT / temperature_k**1.5
    return (jnp.asarray(STATISTICAL_WEIGHTS) / 2 * thermal_volume
            * jnp.exp(jnp.asarray(IONIZATION_ENERGY_ERG)
                      / (BOLTZMANN_CGS * temperature_k)))


def planck_occupation(frequency_hz, temperature_k, dilution=1.0):
    """Return eta = c^2 J_nu/(2 h nu^3) for J_nu = dilution B_nu(T)."""
    x = PLANCK_CGS * jnp.asarray(frequency_hz) / (BOLTZMANN_CGS * temperature_k)
    # This form avoids exp(+x) overflow in the Wien tail and its derivative.
    return dilution * jnp.exp(-x) / (-jnp.expm1(-x))


def continuum_frequency_grid(points=128, maximum_energy_ev=100.0):
    """Return (frequencies, weights), shape (6, points), for integral dnu.

    Gauss-Legendre quadrature uses log frequency from each level's threshold
    to a fixed upper photon energy. The default upper limit resolves thermal
    and stellar blackbodies near 10^4 K; harder spectra need a larger limit.
    Node count and upper limit are static configuration, not fit parameters.
    """
    if points < 2 or maximum_energy_ev <= 13.5984345:
        raise ValueError("Use at least two points and an upper energy above 13.6 eV.")
    nodes, weights = np.polynomial.legendre.leggauss(points)
    lower = IONIZATION_ENERGY_ERG / PLANCK_CGS
    upper = maximum_energy_ev * EV_CGS / PLANCK_CGS
    width = np.log(upper / lower)
    frequency = lower[:, None] * np.exp(width[:, None] * (nodes + 1) / 2)
    weight = frequency * width[:, None] * weights / 2
    return jnp.asarray(frequency), jnp.asarray(weight)


def photoionization_cross_section(frequency_hz):
    """Return Kramers g_bf=1 cross sections [cm^2], shape (6, frequencies).

    This shell approximation applies the same cross section to every l at
    fixed n. It is not a quantum-mechanical, l-resolved photoionization model.
    """
    frequency = jnp.asarray(frequency_hz)
    n = jnp.asarray(PRINCIPAL_QUANTUM_NUMBERS)[:, None]
    threshold = jnp.asarray(IONIZATION_ENERGY_ERG)[:, None] / PLANCK_CGS
    sigma = 2.815e29 / n**5 / frequency**3
    return jnp.where(frequency >= threshold, sigma, 0.0)


def bound_free_rates(temperature_k, occupation, frequency_hz, weight_hz):
    """Return photoionization [s^-1] and Milne recombination [cm^3/s].

    The same quadrature and cross sections are used for both directions.
    Recombination includes spontaneous and stimulated terms (1 + eta),
    giving pointwise Saha balance when eta is Planckian at the gas T.
    """
    frequency = jnp.asarray(frequency_hz)
    occupation = jnp.asarray(occupation)
    sigma = photoionization_cross_section(frequency)
    kernel = 8 * jnp.pi / LIGHT_SPEED_CGS**2 * frequency**2 * sigma
    photoionization = jnp.sum(kernel * occupation * weight_hz, axis=-1)
    excess = PLANCK_CGS * frequency - jnp.asarray(IONIZATION_ENERGY_ERG)[:, None]
    thermal_volume = _THERMAL_VOLUME_COEFFICIENT / temperature_k**1.5
    recombination = jnp.asarray(STATISTICAL_WEIGHTS) / 2 * thermal_volume * jnp.sum(
        kernel * jnp.exp(-excess / (BOLTZMANN_CGS * temperature_k))
        * (1 + occupation) * weight_hz, axis=-1,
    )
    return photoionization, recombination


def bound_bound_radiative_rates(occupation):
    """Return Einstein rates [s^-1] for five lines plus spontaneous 2s decay."""
    eta = jnp.asarray(occupation)
    weights = jnp.asarray(STATISTICAL_WEIGHTS)
    a = jnp.asarray(LINE_EINSTEIN_A_S)
    rates = jnp.zeros((6, 6), dtype=jnp.result_type(eta, float))
    rates = rates.at[LINE_UPPER, LINE_LOWER].set(a * (1 + eta))
    rates = rates.at[LINE_LOWER, LINE_UPPER].set(
        a * eta * weights[LINE_UPPER] / weights[LINE_LOWER]
    )
    return rates.at[1, 0].set(TWO_PHOTON_A_S)


def minimum_rate_temperature(low_temperature_extrapolation="reject"):
    """Return the declared lower bound of the selected collision approximation."""
    if low_temperature_extrapolation not in ("reject", "cloudy"):
        raise ValueError("low_temperature_extrapolation must be 'reject' or 'cloudy'.")
    return (MIN_EXTRAPOLATED_TEMPERATURE_K if low_temperature_extrapolation == "cloudy"
            else MIN_TEMPERATURE_K)


def collision_rates(temperature_k, *, low_temperature_extrapolation="reject"):
    """Return pure-H bound-bound, ionization and three-body coefficients.

    Upsilon is interpolated linearly in log(T), log(Upsilon); reverse rates
    use the model energies and degeneracies. Ionization fits are only
    documented to 20 percent accuracy on 4000--14000 K. Mixing rates are
    sums of electron and proton coefficients because n_e = n_p here.

    ``low_temperature_extrapolation="cloudy"`` adopts C23.01 HCSAR_interp:
    below the first tabulated temperature T0, Upsilon(T)=Upsilon(T0)*T/T0.
    It is a model extrapolation, not calculated low-temperature atomic data.
    We retain the existing Lehmann table and exact T0=0.5 eV/kB; Cloudy uses
    Anderson et al. (2000, corrected 2002) nl values and rounded T0=5802 K.
    Rates above T0 are unchanged. Below 4000 K the ionization fits still fail.
    This altered adaptation retains the Cloudy notice in _licenses/Cloudy.txt.
    """
    minimum_temperature = minimum_rate_temperature(low_temperature_extrapolation)
    temperature = jnp.asarray(temperature_k)
    log_kt_ev = jnp.log(BOLTZMANN_CGS * temperature / EV_CGS)
    log_nodes = jnp.log(jnp.array([.5, 1., 3.]))
    interval = jnp.minimum(jnp.searchsorted(log_nodes, log_kt_ev) - 1, 1)
    interval = jnp.maximum(interval, 0)
    fraction = ((log_kt_ev - log_nodes[interval])
                / (log_nodes[interval + 1] - log_nodes[interval]))
    table = jnp.asarray(_UPSILON)
    active = table[..., 0] > 0
    log_table = jnp.log(jnp.where(active[..., None], table, 1.0))
    upsilon = jnp.where(active, jnp.exp(
        (1 - fraction) * log_table[..., interval]
        + fraction * log_table[..., interval + 1]
    ), 0.0)
    if low_temperature_extrapolation == "cloudy":
        upsilon = jnp.where(temperature < MIN_TEMPERATURE_K,
                            table[..., 0] * temperature / MIN_TEMPERATURE_K, upsilon)
    g = jnp.asarray(STATISTICAL_WEIGHTS)
    energies = jnp.asarray(EXCITATION_ENERGY_ERG)
    down = 8.629e-6 * upsilon / (g[:, None] * jnp.sqrt(temperature))
    # The table contains only transitions from higher to lower n.
    up = down * (g[:, None] / g[None, :]) * jnp.exp(
        -(energies[:, None] - energies[None, :]) / (BOLTZMANN_CGS * temperature)
    )
    bound_bound = down + up.T
    mixing = (
        1.394e-3 * temperature**-.4055 * jnp.exp(-13.35 / temperature)
        + 8.723e-4 * temperature**-.3803 * jnp.exp(-23.97 / temperature)
        + 3.706e-3 * temperature**-.2509 * jnp.exp(-188.6 / temperature)
        + 3.654e-4 * temperature**-.07835 * jnp.exp(-1080 / temperature)
    )
    bound_bound = bound_bound.at[1, 2].set(mixing)
    bound_bound = bound_bound.at[2, 1].set(
        mixing * g[1] / g[2]
        * jnp.exp((energies[2] - energies[1]) / (BOLTZMANN_CGS * temperature))
    )
    a = jnp.array([1.51, 1.35e-7, 1.35e-7, 1.96e-7, 1.96e-7, 1.96e-7])
    b = jnp.array([-2.09, -.033, -.033, .15, .15, .15])
    c = jnp.array([-1.68e5, -3.89e4, -3.89e4, -1.85e4, -1.85e4, -1.85e4])
    ionization = a * temperature**b * jnp.exp(c / temperature)
    three_body = ionization * saha_factors(temperature)
    valid = (temperature >= minimum_temperature) & (temperature <= MAX_TEMPERATURE_K)
    return tuple(jnp.where(valid, x, jnp.nan)
                 for x in (bound_bound, ionization, three_body))


def hydrogen_rate_coefficients(
    temperature_k, bound_bound_occupation, bound_free_occupation,
    continuum_frequency_hz, continuum_weight_hz, *, low_temperature_extrapolation="reject",
):
    """Assemble rates from prescribed photon occupations; invalid inputs yield NaN.

    Bound-bound occupation has shape (5,), in LINE_UPPER/LINE_LOWER order;
    bound-free occupation, quadrature frequencies and weights have shape
    (6, N). The radiation is external and fixed during the population solve.
    Use float64 for accurate statistical equilibrium and its derivatives.
    """
    if jnp.shape(bound_bound_occupation) != (5,):
        raise ValueError("Bound-bound occupations must have shape (5,).")
    shape = jnp.shape(continuum_frequency_hz)
    if len(shape) != 2 or shape[0] != 6 or shape[1] < 2:
        raise ValueError("Continuum arrays must have shape (6, N), with N >= 2.")
    if (jnp.shape(bound_free_occupation) != shape
            or jnp.shape(continuum_weight_hz) != shape):
        raise ValueError("Continuum occupations, frequencies and weights must match.")
    collisional, ionization, three_body = collision_rates(
        temperature_k, low_temperature_extrapolation=low_temperature_extrapolation,
    )
    photoionization, recombination = bound_free_rates(
        temperature_k, bound_free_occupation, continuum_frequency_hz,
        continuum_weight_hz,
    )
    rates = RateCoefficients(
        bound_bound_radiative_rates(bound_bound_occupation), collisional,
        photoionization, recombination, ionization, three_body,
    )
    valid = (
        (temperature_k >= minimum_rate_temperature(low_temperature_extrapolation))
        & (temperature_k <= MAX_TEMPERATURE_K)
        & jnp.all(jnp.isfinite(jnp.asarray(bound_bound_occupation)))
        & jnp.all(jnp.asarray(bound_bound_occupation) >= 0)
        & jnp.all(jnp.isfinite(jnp.asarray(bound_free_occupation)))
        & jnp.all(jnp.asarray(bound_free_occupation) >= 0)
        & jnp.all(jnp.isfinite(jnp.asarray(continuum_frequency_hz)))
        & jnp.all(jnp.asarray(continuum_frequency_hz) > 0)
        & jnp.all(jnp.isfinite(jnp.asarray(continuum_weight_hz)))
        & jnp.all(jnp.asarray(continuum_weight_hz) > 0)
    )
    return RateCoefficients(*(jnp.where(valid, item, jnp.nan) for item in rates))


def blackbody_hydrogen_rates(
    temperature_k, radiation_temperature_k, dilution, *, quadrature_points=128,
    maximum_energy_ev=100.0, low_temperature_extrapolation="reject",
):
    """Convenience closure for prescribed J_nu = dilution B_nu(T_rad)."""
    frequency, weight = continuum_frequency_grid(quadrature_points, maximum_energy_ev)
    rates = hydrogen_rate_coefficients(
        temperature_k,
        planck_occupation(LINE_FREQUENCY_HZ, radiation_temperature_k, dilution),
        planck_occupation(frequency, radiation_temperature_k, dilution),
        frequency, weight, low_temperature_extrapolation=low_temperature_extrapolation,
    )
    valid = (jnp.isfinite(radiation_temperature_k)
             & (radiation_temperature_k > 0) & jnp.isfinite(dilution)
             & (dilution >= 0))
    return RateCoefficients(*(jnp.where(valid, item, jnp.nan) for item in rates))
