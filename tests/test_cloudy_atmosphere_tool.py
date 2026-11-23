"""Check the offline reference's orientation, units and energy convention."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location("cloudy_atmosphere", TOOLS / "cloudy_atmosphere.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
finally:
    sys.path.pop(0)


def test_depth_law_illuminates_outer_boundary_without_surface_log_interpolation_bias():
    radius = np.geomspace(1e10, 3e10, 32)
    density = 1e10 * (radius[0]/radius)**4
    lines = reference.depth_law(radius, density, "dlaw")
    table = np.array([[float(value) for value in line.split()] for line in lines[1:-1]])
    # At one centimeter, density must still equal the top value, not a sizable
    # interpolation toward the next radial node across 35 decades of depth.
    surface = 10**np.interp(0., table[:, 0], table[:, 1])
    assert surface == pytest.approx(density[-1], rel=1e-7)
    assert 10**table[-2, 1] == pytest.approx(density[0])


def test_thermal_balance_has_the_adiabatic_spherical_limit():
    radius = np.geomspace(1e10, 4e10, 1001)
    density = 1e-14 * (radius/radius[0])**-2
    temperature = 1e4 * (radius/radius[0])**(-4/3)
    profile = dict(radius_cm=radius, density_g_cm3=density, temperature_k=temperature,
                   mean_molecular_weight=np.ones_like(radius),
                   heating_erg_cm3_s=np.full_like(radius, 1e-20),
                   cooling_erg_cm3_s=np.full_like(radius, 1e-20))
    adv, expansion, ratio = reference.thermal_balance(profile, np.full_like(radius, 1e5))
    assert np.all(adv > 0) and np.all(expansion < 0)
    assert np.max(abs(ratio[1:-1]-1)) < 1e-5
    projected = dict(profile, density_g_cm3=2*density, wind_density_g_cm3=density)
    flow = reference.thermal_balance(projected, np.full_like(radius, 1e5))
    np.testing.assert_allclose(flow[0], adv)
    np.testing.assert_allclose(flow[1], expansion)


def test_input_rejects_reversed_or_nonfinite_physical_profiles():
    profile = dict(radius_cm=np.arange(1., 5.), temperature_k=np.full(4, 8000.),
                   n_h_cm3=np.full(4, 1e10), velocity_cm_s=np.full(4, 1e5))
    reference.validate_profiles(profile)
    with pytest.raises(ValueError, match="outward"):
        reference.validate_profiles(dict(profile, radius_cm=profile["radius_cm"][::-1]))
    with pytest.raises(ValueError, match="finite"):
        reference.validate_profiles(dict(profile, temperature_k=np.full(4, np.nan)))


def test_reference_atom_overrides_are_explicit_and_require_solar_composition():
    profile = dict(radius_cm=np.linspace(1e10, 3e10, 4),
                   temperature_k=np.full(4, 8000.), n_h_cm3=np.full(4, 1e10))
    text = reference.column_input(profile, [], composition="solar",
                                  iron_levels=716, sulfur_levels=72)
    assert 'species "Fe+" levels=716' in text
    assert 'species "S+" levels=72' in text
    default = reference.column_input(profile, [], composition="solar")
    assert 'species "S+"' not in default
    with pytest.raises(ValueError, match="S II level override"):
        reference.column_input(profile, [], composition="pure-h", sulfur_levels=72)


def test_damped_wind_preserves_proposed_mass_flow():
    radius = np.geomspace(1e10, 3e10, 8)
    old = dict(radius_cm=radius, velocity_cm_s=np.full(8, 1e4))
    proposed = dict(radius_cm=radius, velocity_cm_s=np.full(8, 1e6),
                    n_h_cm3=1e9*(radius[0]/radius)**2)
    relaxed = reference.relax_wind(old, proposed, .5)
    flux = radius**2*relaxed["n_h_cm3"]*relaxed["velocity_cm_s"]
    np.testing.assert_allclose(flux, radius**2*proposed["n_h_cm3"]*proposed["velocity_cm_s"])
    assert np.all(relaxed["velocity_cm_s"] > old["velocity_cm_s"])
    assert np.all(relaxed["velocity_cm_s"] < proposed["velocity_cm_s"])


def test_independent_wind_matches_isothermal_solution_and_sonic_refinement():
    from scipy.special import lambertw
    from exojax.utils.constants import G, MJ

    radius = np.geomspace(1e10, 3e11, 40)
    temperature, mu = np.full(40, 9000.), np.full(40, 1.2)
    sound_squared = reference.KB*temperature[0]/(reference.MH*mu[0])
    sonic_radius = G*MJ/(2*sound_squared)
    argument = -(sonic_radius/radius)**4*np.exp(3-4*sonic_radius/radius)
    branch = np.where(radius < sonic_radius, 0, -1)
    expected = np.sqrt(-sound_squared*lambertw(argument, branch).real)
    wind, info = reference.adaptive_wind(radius, temperature, mu, 1e13, 1., 3.2e11)
    refined, _ = reference.adaptive_wind(radius, temperature, mu, 1e13, 1., 3.2e11,
                                        sonic_offset=1e-7)
    np.testing.assert_allclose(wind["velocity_cm_s"], expected, rtol=3e-6)
    np.testing.assert_allclose(wind["velocity_cm_s"], refined["velocity_cm_s"], rtol=3e-6)
    np.testing.assert_allclose(4*np.pi*radius**2*wind["density_g_cm3"]*wind["velocity_cm_s"], 1e13)
    assert info["sonic_radius_cm"] == pytest.approx(sonic_radius, rel=1e-10)
    expected_gradient = (2*sound_squared/radius-G*MJ/radius**2)/(expected**2-sound_squared)
    np.testing.assert_allclose(wind["log_velocity_gradient_cm1"], expected_gradient, rtol=2e-5)


def test_fixed_wind_flow_uses_the_latest_temperature_and_composition():
    radius = np.geomspace(1e10, 2e10, 9)
    density = 1e-14*(radius/radius[0])**-2
    profile = dict(radius_cm=radius, density_g_cm3=density,
                   temperature_k=np.full(9, 8000.), mean_molecular_weight=np.ones(9),
                   log_velocity_gradient_cm1=np.zeros(9),
                   heating_erg_cm3_s=np.full(9, 1e-6), cooling_erg_cm3_s=np.full(9, 1e-6))
    velocity = np.full(9, 1e5)
    old = reference.thermal_balance(profile, velocity)
    np.testing.assert_allclose(old[0], 0., atol=1e-25)
    mu = (radius/radius[0])**.1
    changed = dict(profile, mean_molecular_weight=mu,
                   wind_temperature_k=8000.*mu*(radius/radius[0])**(-4/3))
    advection, expansion, _ = reference.thermal_balance(changed, velocity)
    assert np.all(advection[1:-1] > 0)
    np.testing.assert_allclose(advection[1:-1], -expansion[1:-1], rtol=1e-12)
    np.testing.assert_array_equal(changed["log_velocity_gradient_cm1"], np.zeros(9))


def test_logarithmic_wind_damping_preserves_the_velocity_derivative():
    radius = np.geomspace(1e10, 3e10, 20)
    current = dict(radius_cm=radius, velocity_cm_s=np.full(20, 1e4),
                   log_velocity_gradient_cm1=np.zeros(20))
    velocity = 1e6*(radius/radius[0])**.3
    proposal = dict(radius_cm=radius, velocity_cm_s=velocity,
                    n_h_cm3=1e15*(radius/radius[0])**-2/velocity,
                    log_velocity_gradient_cm1=.3/radius)
    relaxed = reference.relax_wind(current, proposal, .25)
    derivative = np.gradient(np.log(relaxed["velocity_cm_s"]), np.log(radius))/radius
    np.testing.assert_allclose(relaxed["log_velocity_gradient_cm1"], derivative, rtol=1e-12)
    np.testing.assert_allclose(radius**2*relaxed["n_h_cm3"]*relaxed["velocity_cm_s"],
                               radius**2*proposal["n_h_cm3"]*proposal["velocity_cm_s"])


def test_matching_velocity_samples_do_not_hide_a_flow_gradient_error():
    radius = np.geomspace(1e10, 2e10, 9)
    density = 1e-14*(radius/radius[0])**-2
    velocity = np.full(9, 1e5)
    result = dict(radius_cm=radius, density_g_cm3=density,
                  temperature_k=np.full(9, 8000.), mean_molecular_weight=np.ones(9),
                  log_velocity_gradient_cm1=np.zeros(9),
                  heating_erg_cm3_s=np.zeros(9), cooling_erg_cm3_s=np.zeros(9))
    advection, expansion, _ = reference.thermal_balance(result, velocity)
    result.update(advection_erg_cm3_s=advection, expansion_erg_cm3_s=expansion)
    # Identical nodal values can have different interpolant derivatives.
    proposal = dict(n_h_cm3=density/reference.MH, velocity_cm_s=velocity,
                    log_velocity_gradient_cm1=.03/radius)
    error = reference.wind_flow_proposal_error(result, proposal, reference.MH)
    assert error == pytest.approx(.015)
    assert error > .01


def test_successful_cloudy_exit_does_not_hide_failed_optical_depth_convergence(tmp_path):
    (tmp_path / "model.out").write_text(
        "C-Iterate to convergence did not converge in 20 iterations.\nCloudy exited OK\n")
    with pytest.raises(ValueError, match="without optical-depth convergence"):
        reference.parse_column(tmp_path, 3e10, 25, 10)


@pytest.mark.parametrize("history", [
    "iteration 1 of 4\n",
    "conv_itercheck exits converged\nconv_itercheck exits NOT converged\n",
])
def test_explicit_convergence_history_must_confirm_the_final_column(tmp_path, history):
    # C23 can exit successfully without the usual output warning at an explicit
    # iteration cap; its SAVE convergence result remains authoritative.
    (tmp_path / "model.out").write_text("Cloudy exited OK\n")
    (tmp_path / "model.convergence").write_text(history)
    with pytest.raises(ValueError, match="final optical-depth convergence"):
        reference.parse_column(tmp_path, 3e10, 25, 10)


def test_convergence_history_uses_the_last_completed_result(tmp_path, monkeypatch):
    (tmp_path / "model.out").write_text("Cloudy exited OK\n")
    (tmp_path / "model.heating").write_text("# accepted ionization\n")
    (tmp_path / "model.convergence").write_text(
        "conv_itercheck exits NOT converged\nconv_itercheck exits converged\n")
    expected = {"checked": True}
    monkeypatch.setattr(reference, "read_zone_profiles", lambda *args: expected)
    assert reference.parse_column(tmp_path, 3e10, 25, 10) is expected


@pytest.mark.parametrize("resolved_levels", [3, 4])
def test_mean_molecular_weight_and_resolved_hydrogen_states_are_read_from_cloudy(tmp_path, resolved_levels):
    (tmp_path / "model.out").write_text("Cloudy exited OK\n")
    depth = np.array([1., 2., 3.])
    temperature, hydrogen, electrons = 8000., 1e10, 3e9
    mass_density = 1.4 * reference.MH * hydrogen
    expected_mu = 1.4 / 1.4  # 1.1 nuclei/H plus 0.3 electrons/H.
    pgas = mass_density * reference.KB * temperature / (reference.MH * expected_mu)
    np.savetxt(tmp_path / "model.conditions", np.column_stack((depth, np.full((3, 3), [temperature, hydrogen, electrons]),
                                                              np.tile([.7, .3, 0, 0, 0, 0], (3, 1)))))
    np.savetxt(tmp_path / "model.heating", np.column_stack((depth, np.full((3, 4), [temperature, 2e-6, 1e-6, 0]))))
    np.savetxt(tmp_path / "model.pressure", np.column_stack((depth, np.zeros((3, 4)), np.full(3, pgas))))
    np.savetxt(tmp_path / "model.mass", np.column_stack((np.zeros((3, 5)), np.full(3, mass_density), np.full(3, hydrogen))))
    shell_four = [2] if resolved_levels == 3 else [.2, .4, .6, .8]
    np.savetxt(tmp_path / "model.levels", np.column_stack((depth, np.tile([1e9, 20, 30, 4, 5, 6, *shell_four], (3, 1)))))
    np.savetxt(tmp_path / "model.overview", np.column_stack((depth, np.zeros((3, 10)))))
    result = reference.parse_column(tmp_path, 10., 4, resolved_levels)
    np.testing.assert_allclose(result["mean_molecular_weight"], expected_mu)
    np.testing.assert_allclose(result["n2_cm3"], 50.)
    np.testing.assert_allclose(result["n3_cm3"], 15.)
    np.testing.assert_allclose(result["bound_cm3"][:, 6], 2.)
    np.testing.assert_allclose(result["radius_cm"], [7, 8, 9])
    # A completed iteration can be inspected while its enclosing run failed;
    # diagnostic reading must not make that run an accepted reference.
    (tmp_path / "model.out").write_text("Cloudy exited after an interrupted iteration.\n")
    with pytest.raises(ValueError, match="did not complete"):
        reference.parse_column(tmp_path, 10., 4, resolved_levels)
    diagnostic = reference.read_zone_profiles(tmp_path, 10., 4, resolved_levels)
    np.testing.assert_array_equal(diagnostic["n3_cm3"], result["n3_cm3"])


def test_dense_patch_has_lte_excitation_charge_eos_and_marks_unsolved_heat():
    radius = np.linspace(1e10, 2e10, 8)
    nh = np.geomspace(1e16, 1e12, 8)
    temperature = np.linspace(4500., 8000., 8)
    ionized = np.full(8, .1)
    profiles = dict(radius_cm=radius, temperature_k=temperature.copy(), n_h_cm3=nh,
                    velocity_cm_s=np.full(8, 1e4))
    result = dict(radius_cm=radius, temperature_k=temperature.copy(), n_h_cm3=nh.copy(),
                  density_g_cm3=nh*reference.MH, electron_cm3=nh*.1,
                  mean_molecular_weight=1/(1+ionized), ionized_fraction=ionized.copy(),
                  gas_pressure_dyn_cm2=nh*1.1*reference.KB*temperature,
                  bound_cm3=np.ones((8,18)), n2_cm3=np.ones(8), n3_cm3=np.ones(8),
                  hminus_fraction=np.zeros(8), heating_erg_cm3_s=np.ones(8),
                  cooling_erg_cm3_s=np.ones(8))
    reference.dense_lower_patch(profiles, result, 3, 4500.)
    assert result["temperature_k"][0] == 4500.
    assert np.isnan(result["heating_erg_cm3_s"][:3]).all()
    np.testing.assert_allclose(result["electron_cm3"][:3]/nh[:3], result["ionized_fraction"][:3])
    np.testing.assert_allclose(result["mean_molecular_weight"][:3], 1/(1+result["ionized_fraction"][:3]))
    ratio = result["n2_cm3"][:3]/result["bound_cm3"][:3,0]
    expected = 4*np.exp(-13.5984346*.75*1.602176634e-12/(reference.KB*result["temperature_k"][:3]))
    np.testing.assert_allclose(ratio, expected)
