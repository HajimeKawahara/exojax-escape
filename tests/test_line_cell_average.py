"""Spatial escape integration, distinct from source-kernel parity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
from escape.line_escape_probability import (
    CORE, WING, cell_averaged_line_operators, line_cell_quadrature, line_operators,
)

jax.config.update("jax_enable_x64", True)


def _arguments(**changes):
    values = dict(tau_in=100., tau_total=1200., tau_pump=100., damping=1e-3,
        line_extinction_cm1=1e-8, lower_absorption_cm1=1.2e-8,
        continuum_absorption_cm1=1e-12, electron_cm3=1e10,
        einstein_a_s=1e7, lower_weight=2., upper_weight=6.,
        incident_occupation=.1, outward_continuum_occupation=0.,
        zone_width_cm=4e10, upper_population_cm3=2e4)
    values.update(changes)
    return values


def test_weak_gain_integral_and_consistent_photon_budget():
    front, delta, total = -.01, -.02, -.05
    args = _arguments(tau_in=front, tau_total=total, tau_pump=front,
        line_extinction_cm1=-1e-8, continuum_absorption_cm1=0., electron_cm3=0.)
    result = cell_averaged_line_operators(
        CORE, line_optical_depth_increment=delta, quadrature_order=8, **args)
    # Source escmase is exactly quadratic for -0.1<tau<0; CORE's outward
    # column is the smaller total column in this explicitly inverted case.
    inward = 1-(front+delta/2)/2+(front**2+front*delta+delta**2/3)/6
    outward = 1-total/2+total**2/6
    np.testing.assert_allclose(result.escape_probability, (inward+outward)/2, rtol=2e-15)
    np.testing.assert_allclose(result.stimulated_down_s,
        result.pump_up_s*args["lower_weight"]/args["upper_weight"], rtol=2e-15)
    np.testing.assert_allclose(result.radiative_down_s,
        args["einstein_a_s"]*result.escape_probability+result.stimulated_down_s,
        rtol=2e-15)
    assert result.destroyed_photons_cm3_s == 0.

    positive = cell_averaged_line_operators(
        WING, line_optical_depth_increment=400., **_arguments())
    assert positive.destruction_probability > 0
    np.testing.assert_allclose(positive.destroyed_photons_cm3_s,
        2e4*1e7*positive.destruction_probability, rtol=3e-15)
    assert (positive.escape_probability+positive.destruction_probability
            +positive.electron_escape_probability <= 1.)


def test_quadrature_and_cell_refinement_share_the_same_integral():
    # Compare one cell with independent fine-cell midpoint integration. All
    # cells share the supplied populations and continuum point occupation.
    args = _arguments()
    coarse = cell_averaged_line_operators(
        WING, line_optical_depth_increment=400., quadrature_order=16, **args)
    count = 4096
    fraction = (np.arange(count)+.5)/count
    args["tau_in"] = args["tau_pump"] = 100+400*fraction
    fine = line_operators(WING, average_zone=False, **args)
    np.testing.assert_allclose(np.asarray(coarse), np.asarray(fine).mean(axis=1),
        rtol=3e-7, atol=1e-20)
    # The front approximation is recovered as the cell shrinks, including
    # its AVG_DEPTH branch (which is inactive at vanishing zone depth).
    args = _arguments(zone_width_cm=1e-10)
    tiny = cell_averaged_line_operators(
        WING, line_optical_depth_increment=1e-10, **args)
    np.testing.assert_allclose(tiny, line_operators(WING, **args), rtol=2e-11)


def test_variable_node_field_and_derivative():
    positions, weights = line_cell_quadrature(8)
    def evaluate(log_column):
        scale = jnp.exp(log_column)
        field = .1*jnp.exp(-positions*scale)
        result = cell_averaged_line_operators(WING,
            line_optical_depth_increment=400*scale, quadrature_order=8,
            node_incident_occupation=field,
            **_arguments(tau_in=100*scale, tau_total=1200*scale,
                         tau_pump=100*scale))
        return jnp.array([result.pump_up_s, result.radiative_down_s,
                          result.destroyed_photons_cm3_s])
    point, step = .2, 1e-5
    derivative = jax.jacfwd(evaluate)(point)
    finite = (evaluate(point+step)-evaluate(point-step))/(2*step)
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, finite, rtol=2e-6)
    # A boundary crossing hidden between quadrature nodes is still rejected.
    bad = cell_averaged_line_operators(CORE, line_optical_depth_increment=-30.,
        quadrature_order=1, **_arguments(tau_in=0.,tau_pump=0.,tau_total=-30.))
    assert np.isnan(bad).all()
    with pytest.raises(ValueError, match="point shielding"):
        cell_averaged_line_operators(WING,line_optical_depth_increment=1.,
                                    average_zone=True,**_arguments())


def test_diffuse_point_field_integrates_to_existing_formal_cell_mean():
    model = HydrogenEscapeProbabilityModel([0.,1.,3.], nmax=3, angle_order=4)
    chi = jnp.array([[.1,2.],[.5,1.]])
    emission = jnp.array([[.4,3.],[.7,.2]])
    positions, weights = line_cell_quadrature(32)
    points = jax.vmap(lambda u:model.forward_diffuse_at_fraction(chi,emission,u))(positions)
    # The old helper initializes from its normal frequency grid; use an
    # explicit analytic streaming integral for this two-frequency test.
    angles, angular_weights = np.polynomial.legendre.leggauss(4)
    expected = np.zeros((2,2))
    for mu,w in zip((angles+1)/2,angular_weights/4):
        incoming = np.zeros(2)
        for cell in (1,0):
            tau = np.asarray(chi[cell])*float(model.widths_cm[cell])/mu
            source = np.asarray(emission[cell])/np.asarray(chi[cell])
            average = -np.expm1(-tau)/tau
            expected[cell] += w*(incoming*average+source*(1-average))
            incoming = incoming*np.exp(-tau)+source*(-np.expm1(-tau))
    np.testing.assert_allclose(jnp.tensordot(weights,points,axes=1),expected,rtol=1e-12)


def test_actual_hydrogen_path_recovers_front_limit_and_has_finite_ad():
    from escape.hydrogen_atom import atom_saha_factors
    from escape.hydrogen_continuum import planck_intensity_hz

    options = dict(nmax=3,quadrature_order=2,angle_order=2)
    front = HydrogenEscapeProbabilityModel([0.,1e-6],**options)
    average = HydrogenEscapeProbabilityModel([0.,1e-6],line_sampling="average",
        line_quadrature_order=4,**options)
    temperature, electron = 8000., 1e10
    bound = (electron**2*atom_saha_factors(front.atom,temperature))[None,:]
    boundary = .025*planck_intensity_hz(front.grid.frequency_hz,10000.)
    def population(model,temp):
        return model.population_state(temp,bound,jnp.array([electron]),
                                      jnp.array([electron]),boundary)
    before,after = population(front,temperature),population(average,temperature)
    assert before["converged"] and after["converged"]
    np.testing.assert_allclose(after["line_operators"],before["line_operators"],
                               rtol=2e-5,atol=1e-30)
    derivative = jax.jacfwd(lambda temp:population(average,temp)[
        "line_operators"].pump_up_s)(temperature)
    assert np.isfinite(derivative).all()


def test_solar_supplies_hydrogen_and_metal_point_fields_in_thin_cell_limit():
    from escape.solar import SolarCompositionModel
    from escape.hydrogen_continuum import planck_intensity_hz

    options = dict(nmax=3,quadrature_order=2,angle_order=2,atoms=(("C I",6,0),))
    front = SolarCompositionModel([0.,1e-6,3e-6],**options)
    average = SolarCompositionModel([0.,1e-6,3e-6],line_sampling="average",
                                    line_quadrature_order=4,**options)
    density,temperature = jnp.array([1e10,2e10]),jnp.array([7000.,8000.])
    state = front.initialize(density,temperature)
    incident = .025*planck_intensity_hz(front.grid.frequency_hz,10000.)
    def evaluate(model):
        target,physical = model.evaluate(state,density,temperature,incident)
        return target,physical["valid"]
    before,valid_before = jax.jit(lambda:evaluate(front))()
    after,valid_after = jax.jit(lambda:evaluate(average))()
    assert valid_before and valid_after
    np.testing.assert_allclose(after,before,rtol=2e-8,atol=2e-8)
