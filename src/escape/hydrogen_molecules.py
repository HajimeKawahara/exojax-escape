"""H-minus and H2-plus microphysics for the explicit C23 closure.

Altered JAX adaptation of Cloudy C23.01 mole_reactions.cpp, cool_eval.cpp,
opacity_createall.cpp and rt_diffuse.cpp; see _licenses/Cloudy.txt. These
functions are components of a chemical network, not an LTE abundance model.
The empirical chemical cooling prescription is intentionally distinguished
from the transported continuum and is not a detailed-balance identity.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .hydrogen_rates import BOLTZMANN_CGS, LIGHT_SPEED_CGS, PLANCK_CGS

RYDBERG_ERG = 2.1798723611035e-11
SAHA = 4.1413302848114741e-16
HMINUS_THRESHOLD_RYD = .055432956
HMINUS_UPPER_RYD = 1.807139


class MolecularPhotoRates(NamedTuple):
    """Per-particle photo rates and coefficients, with no density factors."""

    hminus_detachment_s: object
    hminus_induced_attachment_cm3_s: object
    hminus_photo_heat_erg_s: object
    hminus_induced_cooling_erg_cm3_s: object
    h2plus_dissociation_s: object
    h2plus_photo_heat_erg_s: object


def hminus_attachment(temperature_k):
    """Spontaneous radiative attachment H + e -> H- + photon [cm3/s]."""
    t = jnp.asarray(temperature_k)
    value = jnp.select(
        [t < 31.62, t < 90, t < 1200, t < 3800, t <= 14000],
        [8.934e-18*t**.505, 5.159e-18*t**.664, 2.042e-18*t**.87,
         8.861e-18*t**.663, 8.204e-17*t**.393], default=5.70e-16*t**.19,
    )
    return jnp.where(t > 0, value, jnp.nan)


def hminus_equilibrium_factor(temperature_k):
    """C23 n(H-)/(ne n(H I)) factor [cm3], used only for inverse rates."""
    t = jnp.asarray(temperature_k)
    return SAHA / (4*t**1.5) * jnp.exp(8745/t)


def hminus_electron_detachment(temperature_k):
    """H- + e -> H + e + e [cm3/s]; the inverse is this times S_H-."""
    t = jnp.asarray(temperature_k)
    return jnp.where(t < 3074, 1.46e-32*t**6.5*jnp.exp(-8745/t),
                     jnp.where(t < 30000, 5.9e-19*t**2.55, 1.54e-7))


def hminus_associative_detachment(temperature_k):
    """H- + H -> H2 + e [cm3/s], summed over the two H2 superstates."""
    return 1/(545969508.1323510 + jnp.clip(temperature_k, 10., 1e4)*71239.23653059864)


def hminus_mutual_neutralization(temperature_k):
    """H- + H+ -> H + H [cm3/s]."""
    t = jnp.asarray(temperature_k)
    return jnp.where(t < 14125, 1.4e-7*(t/300)**-.487*jnp.exp(t/29300),
                     3.4738192887404660e-8)


def hminus_proton_association(temperature_k):
    """H- + H+ -> H2+ + e [cm3/s]."""
    t = jnp.asarray(temperature_k)
    return jnp.where(t <= 7891, 6.9e-9*t**-.35, 9.6e-7*t**-.9)


def h2plus_attachment(temperature_k):
    """Spontaneous radiative association H + H+ -> H2+ + photon [cm3/s]."""
    return jnp.maximum(2.325*jnp.minimum(5000., temperature_k)-1375., 0.)*1e-20


def attachment_cooling(temperature_k, neutral_h_cm3, proton_cm3, electron_cm3):
    """Return separate C23 H-minus and H2-plus cooling [erg/cm3/s].

    H2+ cooling uses a different empirical fit from its formation rate,
    including a different intercept and no 5000 K cap. Do not add a second
    Milne or generic reaction-enthalpy cooling term for these same channels.
    """
    t = jnp.asarray(temperature_k)
    negative = hminus_attachment(t)*RYDBERG_ERG*t*1.15e-5*electron_cm3*neutral_h_cm3
    positive = jnp.maximum((2.325*t-1875)*1e-20, 0.)*neutral_h_cm3*proton_cm3*1.66e-11
    return negative, positive


def h2_small_thermal_exchange(density_cm3, neutral_h_cm3, draine_face,
                              molecule_cm3, photo_rates_s,
                              collision_exchange_erg_cm3_s):
    """C23 Elwert two-superstate heating in a plane slab, without secondaries.

    The empirical density/UV factors are part of the selected small-H2
    prescription, not an independently derived reaction enthalpy. Both
    photo and collisional inputs use the same solved H2 populations.
    """
    density = jnp.asarray(density_cm3)
    log_density = jnp.clip(jnp.log10(density), 0., 7.)
    log_field = jnp.log10(jnp.clip(draine_face, 1., 1e7))
    f4 = jnp.where(density > 1.,
        (1.25-.25*log_field)**2*10**(2.2211*log_density-29.8506), 0.)
    collision = (jnp.maximum(.1, 24.25-4.75*log_density)*collision_exchange_erg_cm3_s
                 +f4*neutral_h_cm3/density)
    collision *= jnp.where((log_field == 0.) & (density > 1.), .001/density, 1.)
    photo = (.25*1.602176634e-12*(.75+.15*log_density)
             *jnp.sum(molecule_cm3*photo_rates_s[..., :2], axis=-1))
    return photo+collision


def _natural_spline(x, y):
    """Build the fixed natural cubic table without a runtime SciPy dependency."""
    step = np.diff(x)
    matrix = np.diag(2*(step[:-1]+step[1:]))
    matrix += np.diag(step[1:-1], 1) + np.diag(step[1:-1], -1)
    second = np.concatenate(([0.], np.linalg.solve(matrix, 6*np.diff(np.diff(y)/step)), [0.]))
    return np.array([(second[1:]-second[:-1])/(6*step), second[:-1]/2,
                     np.diff(y)/step-step*(2*second[:-1]+second[1:])/6, y[:-1]])


_HMINUS_ENERGY = np.array([
    0, .001459, .003296, .005256, .007351, .009595, .01201, .01460,
    .01741, .02044, .02375, .02735, .03129, .03563, .04043, .04576,
    .05171, .05841, .06601, .07469, .08470, .09638, .1102, .1268,
    .1470, .1723, .2049, .2483, .3090, .4001, .5520, .8557, 1.7669,
])
_HMINUS_CROSS = np.array([
    0, .124, .398, .708, 1.054, 1.437, 1.805, 2.176, 2.518, 2.842,
    3.126, 3.377, 3.580, 3.741, 3.851, 3.913, 3.925, 3.887, 3.805,
    3.676, 3.511, 3.306, 3.071, 2.810, 2.523, 2.219, 1.898, 1.567,
    1.233, .912, .629, .39, .19,
])*1e-17
_HMINUS_SPLINE = _natural_spline(_HMINUS_ENERGY, _HMINUS_CROSS)


def hminus_cross_section(photon_energy_ryd):
    """Wishart/Doughty natural-cubic photodetachment cross section [cm2].

    The explicit C23 continuum stops at the He I ionization threshold.
    Outside this support the model contributes zero, rather than extrapolating
    a photodetachment cross section to arbitrarily high energies.
    """
    photon = jnp.asarray(photon_energy_ryd)
    energy = photon-HMINUS_THRESHOLD_RYD
    knots = jnp.asarray(_HMINUS_ENERGY)
    index = jnp.clip(jnp.searchsorted(knots, energy, side="right")-1, 0, len(knots)-2)
    delta = energy-knots[index]
    c = jnp.asarray(_HMINUS_SPLINE)[:, index]
    cross = ((c[0]*delta+c[1])*delta+c[2])*delta+c[3]
    return jnp.where((energy >= 0) & (photon <= HMINUS_UPPER_RYD), cross, 0.)


def h2plus_cross_sections(photon_energy_ryd):
    """Ground/excited-superstate photodissociation cross sections [cm2].

    Buckingham et al. (1952) and Dunn (1968) tables as used by C23. Its
    historical single-precision eV/13.6 conversion is retained explicitly.
    """
    photon = jnp.asarray(photon_energy_ryd)

    def interpolate(energy, cross, threshold):
        x = jnp.asarray(np.asarray(energy, np.float32)/np.float32(13.6))
        y = jnp.asarray(np.asarray(cross, np.float32)*np.float32(1e-18))
        index = jnp.clip(jnp.searchsorted(x, photon, side="right")-1, 0, len(x)-2)
        # C23 stores the interpolation slopes as realnum (float32).
        slopes = ((y[1:]-y[:-1])/(x[1:]-x[:-1])).astype(jnp.float32)
        value = y[index]+(photon-x[index])*slopes[index]
        return jnp.where((photon >= threshold) & (photon <= 1.03), value, 0.)

    ground = interpolate([6.75, 8.68, 10.54, 12.46, 14.28], [.24, 2.5, 7.1, 6., 2.7], .502)
    excited = interpolate([.69, .83, .95, 1.03, 1.24, 1.38, 1.77, 2.48, 14.28],
                          [1e-5, 1e-4, .01, .08, 2., 10., 20., 8., 1.], .052)
    return ground, excited


def molecular_continuum_coefficients(
    temperature_k, neutral_h_cm3, electron_cm3, hminus_cm3, h2plus_cm3, frequency_hz,
):
    """Return absorption [cm-1] and H-minus emissivity [erg/s/cm3/sr/Hz].

    C23 clips negative net H-minus opacity; this explicit escape-probability
    closure is unsuitable for a maser. H2+ contributes absorption only in
    this prescription. Electron-neutral free-free must be added separately.
    """
    frequency = jnp.asarray(frequency_hz)
    photon = PLANCK_CGS*frequency/RYDBERG_ERG
    sigma = hminus_cross_section(photon)
    inverse = (neutral_h_cm3*electron_cm3*hminus_equilibrium_factor(temperature_k)
               *jnp.exp(-PLANCK_CGS*frequency/(BOLTZMANN_CGS*temperature_k)))
    negative_opacity = sigma*jnp.maximum(hminus_cm3-inverse, 0.)
    emission = sigma*inverse*2*PLANCK_CGS*frequency**3/LIGHT_SPEED_CGS**2
    ground, excited = h2plus_cross_sections(photon)
    fraction = jnp.exp(-10000/temperature_k)
    positive_opacity = h2plus_cm3*((1-fraction)*ground+fraction*excited)
    return negative_opacity+positive_opacity, emission


def molecular_photo_rates(temperature_k, frequency_hz, photon_flux_cm2_s):
    """Integrate continuum bins or discrete OTS events with the same kernels.

    Each input flux already includes its bin width, 4pi and 1/(h nu). OTS
    inputs instead contain integrated photon flux per discrete line. This
    allows all absorbers to use the same photons without interpolating edges.
    Photoelectron energies follow C23's two H2+ superstate thresholds.
    """
    photon = PLANCK_CGS*jnp.asarray(frequency_hz)
    flux = jnp.asarray(photon_flux_cm2_s)
    sigma = hminus_cross_section(photon/RYDBERG_ERG)
    absorption = flux*sigma
    inverse = (hminus_equilibrium_factor(temperature_k)
               *jnp.exp(-photon/(BOLTZMANN_CGS*temperature_k)))
    excess = jnp.maximum(photon-HMINUS_THRESHOLD_RYD*RYDBERG_ERG, 0.)
    ground, excited = h2plus_cross_sections(photon/RYDBERG_ERG)
    fraction = jnp.exp(-10000/temperature_k)
    ground_rate, excited_rate = flux*(1-fraction)*ground, flux*fraction*excited
    return MolecularPhotoRates(
        jnp.sum(absorption), jnp.sum(absorption*inverse), jnp.sum(absorption*excess),
        jnp.sum(absorption*inverse*excess), jnp.sum(ground_rate+excited_rate),
        jnp.sum(ground_rate*jnp.maximum(photon-.502*RYDBERG_ERG, 0.)
                +excited_rate*jnp.maximum(photon-.052*RYDBERG_ERG, 0.)),
    )


def h2_small_photo_rates(temperature_k, ground_column_cm2, excited_column_cm2,
                        draine_field_face, draine_field_local):
    """C23 Elwert two-superstate FUV closure, without grains/cosmic rays.

    Draine fields use the 912--1110 A photon flux divided by 1.232e7*1.71.
    The face field and local continuum attenuation are supplied by the same
    transfer solve; H2 self-shielding uses the columns from the star-facing
    boundary. Return ground dissociation, excited dissociation, and ground
    to excited pumping, all in s^-1. This is not a resolved H2 line model.
    """
    log_field = jnp.clip(jnp.log10(jnp.maximum(draine_field_face, 1.)), 0., 7.)
    b = jnp.sqrt(BOLTZMANN_CGS*temperature_k/(1.00794*1.6605390666e-24))/1e5
    ground = ground_column_cm2/4e14/b
    excited = excited_column_cm2/9e15/b
    shield_g = .965/(1+ground)**(2.25-.05*log_field)
    shield_g += (.032-.004*log_field)/(1+ground)**(.625-.005*log_field)
    shield_s = .965/(1+excited)**(2.625-.125*log_field)
    local = jnp.maximum(draine_field_local, 0.)
    solomon_g = (1.32+.06*log_field)*4.6e-11*shield_g*local
    solomon_s = (2.125+.375*log_field)*4.6e-11*shield_s*(
        local+jnp.maximum(.1, 2.375*log_field-1.875)*local**.2)
    return jnp.array([solomon_g+local*1e-21, solomon_s+local*1e-11,
                      jnp.maximum(1., 11.25-1.75*log_field)*solomon_g])


def _h2_reverse_detachment(temperature_k):
    """C23 chemical partition-ratio inverse, distinct from Saha factors.

    The small chemical network uses statistical weights one and tabulated
    formation enthalpies H=216, H-=143.2, H2=0 kJ/mol; H2* begins at
    4100 cm^-1. This historical prescription differs from its effective
    30172 K excitation energy and is kept explicit for reference comparison.
    """
    mass_h = float(np.float32(np.float32(1.00794)*np.float32(1.6605390666e-24)))
    mass_e = float(np.float32(9.1093837015e-28))
    hminus_enthalpy = float(np.float32(143.2))
    excited_enthalpy = float(np.float32(4100*PLANCK_CGS*LIGHT_SPEED_CGS*6.02214076e23/1e10))
    energies = jnp.array([216+hminus_enthalpy, 216+hminus_enthalpy-excited_enthalpy])*1e10/(6.02214076e23*BOLTZMANN_CGS)
    ratio = (mass_h/(2*mass_e))**1.5*jnp.exp(-energies/temperature_k)
    return hminus_associative_detachment(temperature_k)*jnp.array([4.938e-6, 1-4.938e-6])*ratio


class MolecularBalance(NamedTuple):
    production_cm3_s: object
    destruction_cm3_s: object
    hydrogen_production_cm3_s: object
    hydrogen_destruction_cm3_s: object
    proton_production_cm3_s: object
    proton_destruction_cm3_s: object
    metal_neutralization_cm3_s: object
    h2_collision_exchange_erg_cm3_s: object


def molecular_reaction_balance(
    temperature_k, neutral_h_cm3, ground_h_cm3, n2_cm3, n3_cm3,
    proton_cm3, electron_cm3, molecules_cm3, photo_rates, h2_photo_s,
    *, neutral_he_cm3=0., singly_charged_metals_cm3=0.,
):
    """Four-species chemical source/sink ledger for H-, H2+, H2, H2*.

    H nuclei and charge are carried by every retained reaction. Neutral H
    and H+ exchange with the atomic SE system; electron density is determined
    by the caller's full charge constraint. Heavy hydrides, H3+, HeH+, grains,
    cosmic rays, and H2 ionization above 15.4 eV are absent from this reduced
    network and require a target-domain contribution check. Do not infer
    adequacy merely from small molecular abundance or a small root residual.
    """
    t, h, p, e = temperature_k, neutral_h_cm3, proton_cm3, electron_cm3
    hm, hp, h2, hs = molecules_cm3
    formation = hminus_attachment(t)+photo_rates.hminus_induced_attachment_cm3_s
    detach = hminus_electron_detachment(t)
    association = hminus_associative_detachment(t)
    inverse = _h2_reverse_detachment(t)
    correction = 10**(jnp.clip(14.44-3.08*jnp.log10(t), 0., 6.)*h/(h+1.6e4))
    diss_g = 1.55e-8/jnp.sqrt(t)*jnp.exp(-65107/t)*correction
    diss_s = 4.67e-7*(t/300)**-1*jnp.exp(-55000/t)
    saha_h2 = SAHA*3.634e-5/(4*t**1.5)*jnp.exp(51950/t)
    collision_down = h*1e-12*jnp.sqrt(t)*jnp.exp(-1000/t)/6
    collision_down += h2*1.4e-12*jnp.sqrt(t)*jnp.exp(-18100/(t+1200))/6
    collision_up = collision_down*jnp.exp(-30172/t)
    neutralization = hminus_mutual_neutralization(t)
    # Species order: H, H+, electron, H-, H2+, H2, H2*.
    events = []

    def reaction(rate, reactants, products):
        events.append((rate, reactants, products))

    reaction(formation*h*e, (1, 0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0))
    reaction(photo_rates.hminus_detachment_s*hm, (0, 0, 0, 1, 0, 0, 0), (1, 0, 1, 0, 0, 0, 0))
    reaction(detach*hm*e, (0, 0, 1, 1, 0, 0, 0), (1, 0, 2, 0, 0, 0, 0))
    reaction(detach*hminus_equilibrium_factor(t)*h*e**2,
             (1, 0, 2, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 0))
    reaction(neutralization*hm*p, (0, 1, 0, 1, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0))
    # C23's enthalpy-ratio back reaction, with its n=3 population factor.
    backward = neutralization*jnp.exp(-(1528+float(np.float32(143.2))-2*216)*1e10/(6.02214076e23*BOLTZMANN_CGS*t))
    reaction(backward*h*n3_cm3, (2, 0, 0, 0, 0, 0, 0), (0, 1, 0, 1, 0, 0, 0))
    for index, fraction in ((5, 4.938e-6), (6, 1-4.938e-6)):
        product = [0]*7
        product[index] = 1
        reaction(association*fraction*hm*h, (1, 0, 0, 1, 0, 0, 0), tuple(np.array(product)+[0, 0, 1, 0, 0, 0, 0]))
        reactant = product.copy()
        reactant[2] = 1
        reaction(inverse[index-5]*(h2 if index == 5 else hs)*e, tuple(reactant), (1, 0, 0, 1, 0, 0, 0))
    reaction(hminus_proton_association(t)*hm*p, (0, 1, 0, 1, 0, 0, 0), (0, 0, 1, 0, 1, 0, 0))
    reaction(h2plus_attachment(t)*h*p, (1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0))
    reaction(3.27e-10*(t/300)**-.35*jnp.exp(-17829/t)*ground_h_cm3*n2_cm3,
             (2, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 1, 0, 0))
    reaction(3e-14*ground_h_cm3*n2_cm3, (2, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0))
    reaction(photo_rates.h2plus_dissociation_s*hp, (0, 0, 0, 0, 1, 0, 0), (1, 1, 0, 0, 0, 0, 0))
    reaction(6.4e-10*h*hp, (1, 0, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0, 1))
    reaction(4.37e-8*(t/300)**-.37*e*hp, (0, 0, 1, 0, 1, 0, 0), (2, 0, 0, 0, 0, 0, 0))
    reaction(2e-7*jnp.exp(-30720/t)*e*hp, (0, 0, 1, 0, 1, 0, 0), (1, 1, 1, 0, 0, 0, 0))
    reaction(2e-7*SAHA*2*3.634e-5*t**-1.5*h*p*e,
             (1, 1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 1, 0, 0))
    reaction(2.45e-10*jnp.exp(-21237.15/t)*p*h2, (0, 1, 0, 0, 0, 1, 0), (1, 0, 0, 0, 1, 0, 0))
    reaction(diss_g*h*h2, (1, 0, 0, 0, 0, 1, 0), (3, 0, 0, 0, 0, 0, 0))
    reaction(diss_g*saha_h2*h**3, (3, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 1, 0))
    reaction(diss_s*h*hs, (1, 0, 0, 0, 0, 0, 1), (3, 0, 0, 0, 0, 0, 0))
    reaction((collision_down+2e-7)*hs, (0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1, 0))
    reaction((collision_up+h2_photo_s[2])*h2, (0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1))
    reaction(h2_photo_s[0]*h2, (0, 0, 0, 0, 0, 1, 0), (2, 0, 0, 0, 0, 0, 0))
    reaction(h2_photo_s[1]*hs, (0, 0, 0, 0, 0, 0, 1), (2, 0, 0, 0, 0, 0, 0))
    for index, molecule, threshold in ((5, h2, 51963.), (6, hs, 22953.)):
        reactant = [0]*7
        reactant[2], reactant[index] = 1, 1
        reaction(1e-14*jnp.exp(-threshold/t)*e*molecule, tuple(reactant), (2, 0, 1, 0, 0, 0, 0))
    rates = jnp.stack([item[0] for item in events])
    reactants = jnp.asarray([item[1] for item in events])
    products = jnp.asarray([item[2] for item in events])
    # Cancel catalysts before forming gross source/sink terms.
    difference = products-reactants
    production = rates@jnp.maximum(difference, 0)
    destruction = rates@jnp.maximum(-difference, 0)
    metal_events = hm*singly_charged_metals_cm3*4e-6/jnp.sqrt(t)
    helium_detach = hm*neutral_he_cm3*3.7e-12*(t/300)**2*jnp.exp(-19870/t)
    production = production.at[0].add(metal_events+helium_detach)
    destruction = destruction.at[3].add(metal_events+helium_detach)
    return MolecularBalance(production[3:], destruction[3:], production[0], destruction[0],
                            production[1], destruction[1], metal_events,
                            (collision_down*hs-collision_up*h2)*4.17e-12)


__all__ = [
    "MolecularPhotoRates", "hminus_attachment", "hminus_equilibrium_factor",
    "hminus_electron_detachment", "hminus_associative_detachment",
    "hminus_mutual_neutralization", "hminus_proton_association", "h2plus_attachment",
    "attachment_cooling", "hminus_cross_section", "h2plus_cross_sections",
    "molecular_continuum_coefficients", "molecular_photo_rates",
    "h2_small_photo_rates", "MolecularBalance", "molecular_reaction_balance",
]
