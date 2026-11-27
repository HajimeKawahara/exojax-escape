"""Conditional confinement scales of the saved PR13 state; no new wind solve.

Replay from the repository root with PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src.
The trusted local physical pickle supplies pressure, composition and velocity.
"""
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
from scipy.optimize import brentq

from escape.hydrogen_forward import PLANET_MASS_MJ, PLANET_RADIUS_CM
from escape.parker import PROTON_MASS_CGS
from escape.hydrogen_rates import BOLTZMANN_CGS as kB
from exojax.utils.constants import G, MJ

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def run():
    physical_path = Path('/tmp/escape-solar-molecular-continuation/current8_e_mdot_refined_trust/complete_physical.pkl')
    fixed_path = Path('/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz')
    recipe_path = Path('/tmp/escape-corrected-observer-full-f/recipe.json')
    driver_path = Path('/tmp/escape-corrected-observer-full-f/audit_corrected_observer_full_f.py')
    free_path = ROOT / 'mceg/evidence/halpha_causes_20261127/sources/free_shape.json'
    physical = pickle.loads(physical_path.read_bytes())
    fixed = dict(np.load(fixed_path))
    for key in ('temperature_k', 'bound_cm3', 'radial_velocity_kms'):
        np.testing.assert_array_equal(physical[key], fixed[key])
    free = json.loads(free_path.read_text())['free']['original']
    recipe = json.loads(recipe_path.read_text())
    edges = np.asarray(recipe['edges_rp'])
    radius = (edges[1:] + edges[:-1]) / 2
    r, width = radius * PLANET_RADIUS_CM, np.diff(edges) * PLANET_RADIUS_CM
    pressure = np.asarray(physical['pressure_dyn_cm2'])
    temperature = np.asarray(physical['temperature_k'])
    mu = np.asarray(physical['mean_molecular_weight'])
    velocity = np.asarray(physical['radial_velocity_kms'])
    rho = pressure * mu * PROTON_MASS_CGS / (kB * temperature)
    mdot = 4 * np.pi * r**2 * rho * velocity * 1e5
    np.testing.assert_allclose(mdot, np.full(8, 1e13 * np.exp(.05)), rtol=1e-12)
    sound = np.sqrt(pressure / rho)
    ram = rho * (velocity * 1e5)**2
    magnetic_gas = np.sqrt(8 * np.pi * pressure)
    magnetic_total = np.sqrt(8 * np.pi * (pressure + ram))
    n2 = np.asarray(physical['bound_cm3'])[:, 1:3].sum(axis=1)
    nH = np.asarray(physical['n_h_cm3'])
    proton = np.asarray(physical['proton_cm3'])
    neutral = np.asarray(physical['bound_cm3']).sum(axis=1)
    # Trammell et al. (2011), Appendix B, Table 2: an approximate H-p
    # charge-exchange momentum-transfer frequency, not an n=2 lifetime.
    nu_Hp = 1.2 * (proton / 1e8) * np.sqrt(temperature / 1e4)
    thermal_H = np.sqrt(kB * temperature / PROTON_MASS_CGS)
    omega = 2 * np.pi / (1.48 * 86400)
    tidal = 3 * omega**2
    hill = (G * MJ * PLANET_MASS_MJ / tidal)**(1 / 3)
    # Hill potential: -GM/r - 3/2 Omega^2 x^2 + 1/2 Omega^2 z^2.
    # These are critical-surface intercepts, NOT a volume-equivalent radius.
    transverse = (2 / 3) * hill
    polar = brentq(lambda t: t**3 + 9 * t - 6, .1, 1.) * hill
    phi = np.array([.3761495834419293, .3932225607463806])
    direction_cosine = -np.cos(2 * np.pi * phi)
    sight_radius = np.array([brentq(lambda t: u*u*t**3-3*t+2, .1, 1.) for u in direction_cosine]) * hill
    barrier = G * MJ * PLANET_MASS_MJ / PLANET_RADIUS_CM * (1 - 1.5 * PLANET_RADIUS_CM / hill + .5 * (PLANET_RADIUS_CM / hill)**3)
    np.testing.assert_allclose(G * MJ * PLANET_MASS_MJ / hill**2, tidal * hill, rtol=1e-14)
    scales = {}
    for D in (1., 3., 10., 20.19, 100.):
        # Same T and composition only: these are alternative bookkeeping paths,
        # not states satisfying a momentum/energy/population equation.
        scales[f'{D:g}'] = dict(
            gas_density_multiplier=D, pressure_dyn_cm2=(D * pressure).tolist(),
            speed_same_spherical_mdot_kms=(velocity / D).tolist(),
            radial_crossing_same_mdot_hours=(width / (velocity * 1e5 / D) / 3600).tolist(),
            mdot_same_speed_g_s=(D * mdot).tolist(),
            gas_pressure_local_field_G=(np.sqrt(D) * magnetic_gas).tolist(),
            gas_pressure_surface_equatorial_dipole_G=(np.sqrt(D) * magnetic_gas * radius**3).tolist(),
            gas_plus_radial_ram_surface_dipole_G_same_mdot=(np.sqrt(8*np.pi*(D*pressure+ram/D))*radius**3).tolist(),
            gas_plus_radial_ram_surface_dipole_G_same_speed=(np.sqrt(D)*magnetic_total*radius**3).tolist(),
            external_pressure_to_stagnate_same_mdot_dyn_cm2=(D * pressure + ram / D).tolist(),
            external_proton_density_if_ram_only_400kms_cm3=((D * pressure + ram / D) / (PROTON_MASS_CGS * (4e7)**2)).tolist(),
            isobaric_temperature_if_same_mu_k=(temperature / D).tolist(),
            microclump_volume_fraction_if_same_mean_mass=1/D,
            microclump_recombination_multiplier_if_fixed_ionization=D,
            microclump_linear_column_multiplier_if_fixed_n2_fraction=1.)
    physical_outer_column = float(np.sum(n2[5:] * width[5:]))
    inputs = [Path(__file__), physical_path, fixed_path, recipe_path, driver_path, free_path,
              *[ROOT/'src/escape'/name for name in ('solar.py','atmosphere_wind.py','hydrogen_forward.py','hydrogen_rates.py','parker.py')]]
    result = dict(scope=__doc__.strip(),
        constants=dict(planet_mass_mj=float(PLANET_MASS_MJ),planet_radius_cm=float(PLANET_RADIUS_CM),period_days=1.48,
            gravitational_cgs=float(G),jupiter_mass_g=float(MJ),boltzmann_cgs=float(kB),proton_mass_g=float(PROTON_MASS_CGS)),
        radius_rp=radius.tolist(), baseline=dict(temperature_k=temperature.tolist(),pressure_dyn_cm2=pressure.tolist(),
            density_g_cm3=rho.tolist(),radial_velocity_kms=velocity.tolist(),sound_speed_kms=(sound/1e5).tolist(),
            radial_ram_pressure_dyn_cm2=ram.tolist(),mass_flux_g_s=mdot.tolist(),
            hydrogen_ion_fraction=(proton/nH).tolist(),n2_fraction_of_H_nuclei=(n2/nH).tolist(),
            neutral_H_fraction=(neutral/nH).tolist(),
            local_gas_pressure_field_G=magnetic_gas.tolist(),surface_equatorial_gas_pressure_field_G=(magnetic_gas*radius**3).tolist(),
            local_gas_plus_ram_field_G=magnetic_total.tolist(),surface_equatorial_gas_plus_ram_field_G=(magnetic_total*radius**3).tolist(),
            neutral_proton_momentum_frequency_s=nu_Hp.tolist(),neutral_proton_momentum_time_s=(1/nu_Hp).tolist(),
            H_thermal_mean_free_path_cm=(thermal_H/nu_Hp).tolist(),mean_free_path_over_cell_width=(thermal_H/nu_Hp/width).tolist()),
        original_free_comparison=dict(n2_local_ratio=(np.asarray(free['at_physical_centers']['n2_cm3'])/n2).tolist(),
            source_local_ratio=free['at_physical_centers']['source_ratio_to_physical'],
            same_mdot_density_ratio_from_free_speed_alone=(velocity/np.asarray(free['at_physical_centers']['radial_velocity_kms'])).tolist(),
            n2_column_above_1p725_cm2=dict(physical=physical_outer_column,free=free['n2_column_above_radius_cm2']['1.725'],
                ratio=free['n2_column_above_radius_cm2']['1.725']/physical_outer_column)),
        density_scalings=scales,
        tidal_geometry=dict(already_included_1D_tidal_coefficient_s2=tidal,hill_L1_approximation_rp=hill/PLANET_RADIUS_CM,
            critical_surface_transverse_y_rp=transverse/PLANET_RADIUS_CM,critical_surface_polar_z_rp=polar/PLANET_RADIUS_CM,
            target_phase=phi.tolist(),critical_surface_radial_intercept_toward_observer_rp=(sight_radius/PLANET_RADIUS_CM).tolist(),
            saved_sonic_radius_rp=float(physical['wind_info']['sonic_radius_cm'])/PLANET_RADIUS_CM,
            free_static_nominal_tau1_radius_rp=free['static_center_tau_one_radius_rp'],
            volume_equivalent_roche_radius=None,
            note='Circular, synchronous Hill approximation with the actual model mass/period. Directional intercepts are not a volume radius, an observer formation radius or a magnetized confinement boundary. Exact binary Roche geometry is not evaluated.'),
        energy_bookkeeping=dict(specific_potential_barrier_base_to_L1_erg_g=barrier,
            baseline_mdot_times_barrier_erg_s=float(mdot.mean()*barrier),
            caveat='Minimum potential-work scale only, not an energy-limited upper bound: base enthalpy, heating, radiative losses, geometry and terminal kinetic energy are not solved.'),
        checks=dict(saved_state_identity=True,spherical_mass_flux_relative_spread=float(np.ptp(mdot)/mdot.mean()),
            matches_recipe_mdot_times_exp_0p05=True,hill_force_balance=True),
        limitations=[
            'Density multiplier D is independent of the required n2 multiplier. No n2 proportional-to-density law is assumed.',
            'B=sqrt(8 pi P) gives beta=1; a pressure comparison alone does not prove a closed stable field or support along it. Polar surface field is twice the stated equatorial field for a centered dipole.',
            'No stellar or planetary magnetic-field detection, stellar-wind boundary or magnetopause is assumed.',
            'Slowdown at fixed mass flux is continuity bookkeeping, not a replacement transonic wind solution. A closed sector need not carry that global mass flux.',
            'Momentum coupling estimates describe bulk H and protons at the frozen ionization state, not metastable trapping or long-term ambipolar/reconnection stability.',
            'Microclumping estimates assume fixed temperature/ionization and optically thin rate scaling. Volume filling factor is not projected covering fraction.',
            'All density changes require new charge, radiative, thermal and momentum balances. No native observation score is evaluated.'],
        source_and_input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (HERE/'scales.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cell5={k:v[5] for k,v in result['baseline'].items()},tidal_geometry=result['tidal_geometry'],
                         comparison=result['original_free_comparison']),indent=2))


if __name__=='__main__':
    run()
