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


def test_successful_cloudy_exit_does_not_hide_failed_optical_depth_convergence(tmp_path):
    (tmp_path / "model.out").write_text(
        "C-Iterate to convergence did not converge in 20 iterations.\nCloudy exited OK\n")
    with pytest.raises(ValueError, match="without optical-depth convergence"):
        reference.parse_column(tmp_path, 3e10, 25, 10)


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
