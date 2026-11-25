"""Population-to-spectrum conservation and derivatives with an offline line."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_atom import solve_atom_populations
from escape.hydrogen_forward import PLANET_RADIUS_CM
from escape.hydrogen_rates import LIGHT_SPEED_CGS, PLANCK_CGS, planck_occupation
from escape.nlte_forward import HydrogenNLTEForwardModel, halpha_coefficients
from escape.parker import PROTON_MASS_CGS
from test_hydrogen_forward import offline_nist  # noqa: F401

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def model():
    return HydrogenNLTEForwardModel(
        nlayer=8, nlon=4, nlat=2, nnu=101, wind_ngrid=65,
        continuum_points=48,
    )


def test_halpha_uses_individual_sublevel_populations():
    # Independent branch constants; shell totals alone cannot fix these sums.
    populations = np.array([1e9, 2e4, 7e4, 120., 210., 450.])
    a_values = np.array([6.3143e6, (2.2448e7 + 2 * 2.24485e7) / 3,
                         (4 * 6.4651e7 + 6 * 6.465166666666667e7) / 10])
    lower = populations[[2, 1, 2]]
    upper = populations[[3, 4, 5]]
    ratio = np.array([2 / 6, 6 / 2, 10 / 6])
    center = 15233.146
    alpha, emission = halpha_coefficients(populations, center)
    expected_alpha = np.sum(a_values * (ratio * lower - upper)) / (
        8 * np.pi * LIGHT_SPEED_CGS * center**2
    )
    expected_emission = PLANCK_CGS * LIGHT_SPEED_CGS * center * np.sum(
        a_values * upper
    ) / (4 * np.pi)
    np.testing.assert_allclose(alpha, expected_alpha, rtol=2e-6)
    np.testing.assert_allclose(emission, expected_emission, rtol=2e-6)
    redistributed = populations.copy()
    redistributed[1:3] = populations[[2, 1]]
    new_alpha, new_emission = halpha_coefficients(redistributed, center)
    assert not np.allclose(alpha, new_alpha, rtol=1e-2, atol=0)
    np.testing.assert_allclose(emission, new_emission)
    extended = np.r_[populations, np.full(12, 1e8)]
    extended_alpha, extended_emission = halpha_coefficients(extended, center)
    np.testing.assert_allclose(extended_alpha, alpha)
    np.testing.assert_allclose(extended_emission, emission)
    branch_alpha, branch_emission = halpha_coefficients(
        populations, center, sum_branches=False,
    )
    assert branch_alpha.shape == branch_emission.shape == (3,)
    np.testing.assert_allclose(branch_alpha.sum(), alpha[0])
    np.testing.assert_allclose(branch_emission.sum(), emission[0])


def test_profiles_conserve_charge_nuclei_and_mass_flux(model):
    profiles = model.population_profiles()
    assert np.all(profiles["converged"])
    density = np.asarray(profiles["n_h_cm3"])
    bound = np.asarray(profiles["bound_cm3"])
    assert model.nmax == 15
    assert bound.shape == (8, 18)
    electrons = np.asarray(profiles["electron_density_cm3"])
    assert np.all(bound >= 0)
    np.testing.assert_allclose(electrons, profiles["proton_density_cm3"], rtol=0)
    np.testing.assert_allclose(bound.sum(axis=1) + electrons, density, rtol=1e-10)
    flux = (4 * np.pi * (np.asarray(model.radius_rp) * PLANET_RADIUS_CM)**2
            * PROTON_MASS_CGS * density * profiles["radial_velocity_kms"] * 1e5)
    np.testing.assert_allclose(flux, 1e13, rtol=1e-13)
    np.testing.assert_allclose(profiles["temperature_k"], model.temperature_profile())
    assert np.all(np.asarray(profiles["temperature_k"]) > 6000)
    np.testing.assert_allclose(profiles["n2_cm3"], bound[:, 1:3].sum(axis=1))
    np.testing.assert_allclose(profiles["n3_cm3"], bound[:, 3:6].sum(axis=1))
    assert np.all(bound[:, 6:].sum(axis=1) > 0)
    assert np.all(profiles["n3_cm3"] < bound[:, 3:].sum(axis=1))
    assert np.all(profiles["positive_extinction"])


def test_layer_populations_match_direct_extended_atom(model):
    parameters = dict(mass_loss_g_s=1.2e13, radiation_dilution=0.03,
                      radiation_temperature_k=11000., temperature_top_k=9500.)
    profiles = model.population_profiles(**parameters)
    line_occupation = planck_occupation(
        model.atom.line_frequency_hz, parameters["radiation_temperature_k"],
        parameters["radiation_dilution"],
    )
    continuum_occupation = planck_occupation(
        model.continuum_frequency_hz, parameters["radiation_temperature_k"],
        parameters["radiation_dilution"],
    )
    # Separate scalar solves check both ends of the density/temperature profile.
    for layer in (0, -1):
        direct = solve_atom_populations(
            profiles["n_h_cm3"][layer], model.atom, profiles["temperature_k"][layer],
            line_occupation, continuum_occupation,
            model.continuum_frequency_hz, model.continuum_weight_hz,
        )
        assert direct.converged
        np.testing.assert_allclose(profiles["bound_cm3"][layer], direct.bound_cm3,
                                   rtol=1e-10)
        np.testing.assert_allclose(profiles["electron_density_cm3"][layer],
                                   direct.electron_cm3, rtol=1e-10)


def test_supplied_blackbody_reproduces_default_populations_and_spectrum(model):
    radiation = dict(radiation_temperature_k=11000., radiation_dilution=0.03)
    supplied = dict(
        line_occupation=planck_occupation(model.atom.line_frequency_hz, 11000., 0.03),
        continuum_occupation=planck_occupation(model.continuum_frequency_hz, 11000., 0.03),
        # These blackbody-only inputs must not alter an explicitly supplied field.
        radiation_temperature_k=-1., radiation_dilution=-1.,
    )
    expected = model.population_profiles(**radiation)
    actual = model.population_profiles(**supplied)
    assert np.all(actual["valid"] & actual["converged"])
    for key in ("bound_cm3", "electron_density_cm3", "halpha_branch_extinction_integral",
                "halpha_branch_emissivity_integral"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=0)
    np.testing.assert_allclose(model.spectrum(**supplied), model.spectrum(**radiation),
                               rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("field", ["line_occupation", "continuum_occupation"])
def test_supplied_field_requires_both_arrays(model, field):
    with pytest.raises(ValueError, match="Supply both"):
        model.spectrum(**{field: jnp.ones(1)})


@pytest.mark.parametrize("field", ["line_occupation", "continuum_occupation"])
@pytest.mark.parametrize("wrong_shape", ["short", "two_dimensional"])
def test_supplied_field_requires_model_frequency_grids(model, field, wrong_shape):
    supplied = dict(
        line_occupation=jnp.ones_like(model.atom.line_frequency_hz),
        continuum_occupation=jnp.ones_like(model.continuum_frequency_hz),
    )
    supplied[field] = (supplied[field][:-1] if wrong_shape == "short"
                       else supplied[field][None, :])
    with pytest.raises(ValueError, match=field + " must match"):
        model.spectrum(**supplied)


@pytest.mark.parametrize("field", ["line_occupation", "continuum_occupation"])
@pytest.mark.parametrize("invalid", [-1., np.nan])
def test_invalid_supplied_radiation_remains_visible(model, field, invalid):
    supplied = dict(
        line_occupation=planck_occupation(model.atom.line_frequency_hz, 10000., 0.025),
        continuum_occupation=planck_occupation(model.continuum_frequency_hz, 10000., 0.025),
    )
    supplied[field] = supplied[field].at[0].set(invalid)
    profiles = model.population_profiles(**supplied)
    assert not np.any(profiles["valid"] | profiles["converged"])
    assert np.isnan(np.asarray(profiles["bound_cm3"])).all()
    assert np.isnan(np.asarray(model.spectrum(**supplied))).all()


def test_supplied_field_spectrum_derivatives_match_finite_differences(model):
    line = planck_occupation(model.atom.line_frequency_hz, 10000., 0.025)
    continuum = planck_occupation(model.continuum_frequency_hz, 10000., 0.025)
    weights = jnp.exp(-0.5 * ((model.velocity_kms - 10) / 35)**2)

    @jax.jit
    def observable(parameters):
        return jnp.mean(weights * model.spectrum(
            mass_loss_g_s=1e13 * parameters[0],
            line_occupation=line * parameters[1],
            continuum_occupation=continuum * parameters[2],
            temperature_top_k=10000 * parameters[3],
        ))

    parameters = np.ones(4)
    automatic = np.asarray(jax.jit(jax.grad(observable))(parameters))
    step = 1e-4
    finite = np.array([
        (observable(parameters + step * direction)
         - observable(parameters - step * direction)) / (2 * step)
        for direction in np.eye(4)
    ])
    assert np.isfinite(automatic).all()
    assert np.all(np.abs(automatic) > 1e-12)
    np.testing.assert_allclose(automatic, finite, rtol=3e-3, atol=1e-12)


def test_branch_profiles_preserve_lifetime_widths_and_velocity(model):
    profiles = np.asarray(model._line_profiles(10000., 0.))
    assert profiles.shape == (3, model.wavenumber_cm.size)
    assert np.isfinite(profiles).all()
    assert np.all(profiles > 0)
    assert not np.allclose(profiles[0], profiles[1], rtol=1e-4, atol=0)
    # Infinite-domain normalization retains the missing finite-window wings.
    integrals = np.trapezoid(profiles, model.wavenumber_cm, axis=1)
    assert np.all((integrals > 0.99) & (integrals < 1.))
    shifted = np.asarray(model._line_profiles(10000., 20.))
    peak_velocity = np.asarray(model.velocity_kms)[np.argmax(shifted, axis=1)]
    # Keep the NIST plotting origin; each atom branch has its own rest center.
    centers = np.array([15233.060432647531, 15233.330845097202,
                        15233.179844462315])
    np.testing.assert_allclose(model.halpha_centers_cm, centers, rtol=2e-15)
    c_kms = LIGHT_SPEED_CGS / 1e5
    expected_velocity = c_kms * (
        model.line_center_cm / (centers * (1 - 20. / c_kms)) - 1
    )
    np.testing.assert_allclose(peak_velocity, expected_velocity, rtol=0, atol=1.3)


def test_spectrum_derivatives_include_rates_populations_and_wind(model):
    weights = jnp.exp(-0.5 * ((model.velocity_kms - 10) / 35)**2)

    @jax.jit
    def observable(parameters):
        return jnp.mean(weights * model.spectrum(
            mass_loss_g_s=1e13 * parameters[0],
            radiation_dilution=0.025 * parameters[1],
            temperature_top_k=10000 * parameters[2],
        ))

    parameters = np.ones(3)
    assert np.isfinite(np.asarray(model.spectrum())).all()
    automatic = np.asarray(jax.jit(jax.grad(observable))(parameters))
    step = 1e-4
    finite = np.array([
        (observable(parameters + step * direction)
         - observable(parameters - step * direction)) / (2 * step)
        for direction in np.eye(3)
    ])
    assert np.isfinite(automatic).all()
    assert np.all(np.abs(automatic) > 1e-12)
    np.testing.assert_allclose(automatic, finite, rtol=3e-3, atol=1e-12)


def test_invalid_profiles_remain_visible(model):
    profiles = model.population_profiles(mass_loss_g_s=-1)
    assert not np.any(profiles["converged"])
    assert not np.any(profiles["positive_extinction"])
    assert np.isnan(np.asarray(model.spectrum(radiation_dilution=0))).all()
    assert np.isnan(np.asarray(model.spectrum(temperature_top_k=15000))).all()
    with pytest.raises(ValueError, match="atomic-data domain"):
        HydrogenNLTEForwardModel(temperature_base_k=4500)
    with pytest.raises(ValueError, match="only wavelength_model='nist'"):
        HydrogenNLTEForwardModel(wavelength_model="rydberg")


@pytest.mark.parametrize("nmax, states", [(3, 6), (20, 23), (25, 28)])
def test_finite_atom_comparison_configuration(nmax, states):
    model = HydrogenNLTEForwardModel(
        nmax=nmax, nlayer=2, nlon=2, nlat=2, nnu=101, wind_ngrid=33,
    )
    assert len(model.atom.principal_n) == states
    assert model.nmax == nmax


@pytest.mark.parametrize("nmax", [2, 26, True, 4.5])
def test_invalid_atom_size_is_rejected(nmax):
    with pytest.raises(ValueError, match="nmax must be an integer"):
        HydrogenNLTEForwardModel(nmax=nmax)


def test_nonuniform_observer_accepts_the_solved_continuum_once():
    from escape.hydrogen_continuum import continuum_coefficients
    from escape.hydrogen_forward import PLANET_RADIUS_CM
    from escape.hydrogen_rates import LIGHT_SPEED_CGS

    model = HydrogenNLTEForwardModel(
        radius_edges_rp=[1., 1.05, 1.3, 3.], nmax=3,
        nlon=2, nlat=2, nnu=101, wind_ngrid=33,
    )
    np.testing.assert_allclose(model.thickness_cm.sum(), 2*PLANET_RADIUS_CM)
    temperature = jnp.array([8000., 7000., 6000.])
    electron = jnp.array([1e9, 1e10, 1e11])
    bound = jnp.array([1., 10., 100.])[:, None]*jnp.array([1e10, 1e3, 3e3, 10., 20., 30.])
    frequency = jnp.asarray(model.wavenumber_cm)*LIGHT_SPEED_CGS
    chi, eta = jax.vmap(lambda t, b, e: continuum_coefficients(
        model.atom, t, b, e, e, frequency))(temperature, bound, electron)
    supplied = model.spectrum_from_profiles(bound, temperature, jnp.ones(3),
        continuum_extinction=chi, continuum_emission=eta*LIGHT_SPEED_CGS)
    builtin = model.spectrum_from_profiles(bound, temperature, jnp.ones(3),
        electron_cm3=electron, proton_cm3=electron, include_continuum=True)
    assert np.isfinite(supplied).all()
    np.testing.assert_allclose(supplied, builtin, rtol=2e-12, atol=1e-15)
    with pytest.raises(ValueError, match="avoid double counting"):
        model.spectrum_from_profiles(bound, temperature, jnp.ones(3),
            include_continuum=True, continuum_extinction=chi, continuum_emission=eta)


def test_halpha_coefficients_use_each_branch_center():
    from escape.hydrogen_rates import LINE_EINSTEIN_A_S

    populations = jnp.array([1e9, 2e4, 7e4, 120., 210., 450.])
    centers = jnp.array([15233.060432647531, 15233.330845097202,
                         15233.179844462315])
    a = np.asarray(LINE_EINSTEIN_A_S[2:])
    lower, upper = np.asarray(populations)[[2, 1, 2]], np.asarray(populations)[[3, 4, 5]]
    expected_alpha = a * (np.array([1/3, 3., 5/3]) * lower - upper) / (
        8 * np.pi * LIGHT_SPEED_CGS * np.asarray(centers)**2
    )
    expected_emission = PLANCK_CGS * LIGHT_SPEED_CGS * centers * a * upper / (4*np.pi)
    alpha, emission = halpha_coefficients(populations, centers, sum_branches=False)
    np.testing.assert_allclose(alpha, expected_alpha, rtol=3e-15)
    np.testing.assert_allclose(emission, expected_emission, rtol=3e-15)
    common = halpha_coefficients(populations, 15233.146)
    explicit_common = halpha_coefficients(populations, jnp.full((3,), 15233.146))
    for actual, expected in zip(common, explicit_common):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="scalar or have shape"):
        halpha_coefficients(populations, jnp.ones(2))


def test_prescribed_observer_uses_branch_coefficient_centers(model):
    bound = jnp.tile(jnp.array([1e10, 1e3, 3e3, 10., 20., 30.] + [0.]*12), (8, 1))
    temperature = jnp.full((8,), 8000.)
    velocity = jnp.linspace(1., 4., 8)
    alpha, emission = halpha_coefficients(bound, model.atom.line_frequency_hz[2:5]/LIGHT_SPEED_CGS,
                                          sum_branches=False)
    expected = model._transfer_line_coefficients(alpha, emission, velocity, 5., 10., temperature)
    actual = model.spectrum_from_profiles(bound, temperature, velocity,
                                          rotation_kms=5., zonal_kms=10.)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)
