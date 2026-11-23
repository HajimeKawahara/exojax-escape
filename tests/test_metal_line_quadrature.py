"""Optical-depth line integration against independent adaptive quadrature."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq

from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
from escape.line_escape_probability import PRD, line_operators
from escape.metal_network import _depth_averaged_metal_lines
from escape.radiation import _cell_weights

jax.config.update("jax_enable_x64", True)


def case(log_column=0., increment=1e4, front=3., total=2e4):
    column = jnp.exp(log_column)
    frequency = jnp.array([1e15])
    field = dict(frequency_hz=jnp.array([.9e15, 1.1e15]),
                 incoming=jnp.array([[[1e-5, 1.2e-5]]]),
                 optical_depth=jnp.array([[[.8, 1.6]]]),
                 emitted=jnp.array([[[2e-6, 3e-6]]]),
                 angular_weight=jnp.array([.5]))
    arguments = dict(tau_in=jnp.asarray(front).reshape(1, 1)*column,
        tau_total=jnp.asarray(total).reshape(1)*column,
        tau_pump=jnp.asarray(front).reshape(1, 1)*column, damping=jnp.array([[.003]]),
        line_extinction_cm1=jnp.array([[1e-6]]), lower_absorption_cm1=jnp.array([[1e-6]]),
        continuum_absorption_cm1=jnp.array([[1e-10]]), electron_cm3=jnp.array([[1e8]]),
        einstein_a_s=jnp.array([1e8]), lower_weight=jnp.array([2.]), upper_weight=jnp.array([4.]),
        incident_occupation=jnp.array([[0.]]), outward_continuum_occupation=0.,
        zone_width_cm=jnp.array([[1e8]]), upper_population_cm3=jnp.array([[1e4]]))
    return frequency, field, arguments, jnp.array([[increment]])*column


def integrated(log_column=0., increment=1e4, order=16, front=3., total=2e4,
               return_normalization_error=False):
    nu, field, args, depth = case(log_column, increment, front, total)
    return _depth_averaged_metal_lines(jnp.array([PRD]), depth, nu, jnp.array([2e-5]),
        jnp.array([[.2]]), jnp.array([[.7]]), field, order,
        "continuous_core", "continuous",
        return_normalization_error=return_normalization_error, **args)


def adaptive(log_column=0., increment=1e4, front=3., total=2e4):
    nu, field, args, depth = case(log_column, increment, front, total)

    @jax.jit
    def point(position):
        diffuse = .25*jnp.sum(field["incoming"]*jnp.exp(-position*field["optical_depth"])
            +position*field["emitted"]*_cell_weights(position*field["optical_depth"])[0])
        intensity = 2e-5*jnp.exp(-.2-.7*position)+diffuse
        from escape.metal_atom import PLANCK, C
        return jnp.array(line_operators(PRD, average_zone=False,
            shielding_model="continuous_core", core_escape_model="continuous",
            **dict(args, tau_in=args["tau_in"]+position*depth,
                   tau_pump=args["tau_pump"]+position*depth,
                   incident_occupation=intensity/(2*PLANCK*nu**3/C**2))))[:, 0, 0]

    scale = np.maximum(np.abs(np.asarray(point(0.))), np.abs(np.asarray(point(1.))))
    # Independent log-coordinate integration; no production nodes or weights.
    stretch = 1e4
    damping = .003
    wing_scale = 3.02*(damping*1e3)**(-.064)
    # Locate the source's continuous positive-depth cap independently. This
    # splits only the adaptive reference, not the production quadrature.
    cap = brentq(lambda tau: np.expm1(-.66666*tau)+damping/wing_scale
        /np.sqrt(.78540+tau*np.sqrt(np.pi)*damping/wing_scale**2), 0., 2., xtol=1e-15)
    crossings = [(tau-front)/increment for tau in (0., cap)]
    points = [np.log1p(stretch*u) for u in crossings if 0 < u < 1]
    value, _ = quad_vec(lambda y: np.asarray(point(np.expm1(y)/stretch))
                       *np.exp(y)/stretch/scale, 0., np.log1p(stretch),
                       epsabs=1e-12, epsrel=1e-11, points=points)
    return value*scale


def test_resolves_prd_skin_and_preserves_destroyed_photon_budget():
    operators, mass_error = integrated(return_normalization_error=True)
    assert np.max(np.abs(mass_error)) < 1e-12
    result = np.asarray(operators)[:, 0, 0]
    np.testing.assert_allclose(result, adaptive(), rtol=3e-6)
    np.testing.assert_allclose(result[6], 1e4*1e8*result[1], rtol=3e-14)


def test_moving_prd_joins_differentiate_the_integral():
    derivative = np.asarray(jax.jacfwd(lambda value: jnp.asarray(integrated(value)))(0.))[:, 0, 0]
    reverse = np.asarray(jax.jacrev(lambda value: jnp.asarray(integrated(value)))(0.))[:, 0, 0]
    np.testing.assert_allclose(reverse, derivative, rtol=3e-12, atol=1e-13)
    finite = (adaptive(2e-5)-adaptive(-2e-5))/4e-5
    np.testing.assert_allclose(derivative, finite, rtol=8e-5, atol=1e-13)


def test_negative_increment_follows_prd_join_with_positive_endpoints():
    options = dict(increment=-999., front=1000., total=2000.)
    result = np.asarray(integrated(**options))[:, 0, 0]
    np.testing.assert_allclose(result, adaptive(**options), rtol=3e-6)
    derivative = np.asarray(jax.jacfwd(
        lambda theta: jnp.asarray(integrated(theta, **options)))(0.))[:, 0, 0]
    finite = (adaptive(2e-5, **options)-adaptive(-2e-5, **options))/4e-5
    np.testing.assert_allclose(derivative, finite, rtol=8e-5, atol=1e-13)


def test_moving_zero_depth_jump_retains_unsaturated_maser_source():
    def value(theta):
        # This artificial interval also contains the continuous shielding-cap
        # kink. Resolve its derivative by refinement, without changing the fit.
        return jnp.asarray(integrated(increment=-.02, front=.01+.02*theta,
                                      total=100., order=1024))[:, 0, 0]
    def reference(theta):
        return adaptive(increment=-.02, front=.01+.02*theta, total=100.)
    np.testing.assert_allclose(value(0.), reference(0.), rtol=3e-6)
    forward = jax.jacfwd(value)(0.)
    reverse = jax.jacrev(value)(0.)
    finite = (reference(2e-5)-reference(-2e-5))/4e-5
    np.testing.assert_allclose(forward, reverse, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(forward, finite, rtol=8e-5, atol=1e-13)


def test_zero_depth_and_invalid_negative_columns_retain_guards():
    assert np.all(np.isfinite(np.asarray(integrated(increment=0.))))
    assert np.all(np.isfinite(np.asarray(integrated(increment=1e-12))))
    nu, field, args, depth = case()
    args.update(tau_in=jnp.array([[-31.]]), tau_pump=jnp.array([[-31.]]))
    result = _depth_averaged_metal_lines(jnp.array([PRD]), depth, nu, jnp.array([2e-5]),
        jnp.array([[.2]]), jnp.array([[.7]]), field, 8,
        "continuous_core", "continuous", **args)
    assert np.all(np.isnan(np.asarray(result)))


def test_prepared_diffuse_rays_match_existing_point_transfer():
    model = HydrogenEscapeProbabilityModel([0., 1e6, 3e6], nmax=3,
                                           quadrature_order=2, angle_order=2)
    extinction = jnp.ones((2, len(model.grid.frequency_hz)))*1e-6
    emission = extinction*jnp.geomspace(1e-8, 1e-5, len(model.grid.frequency_hz))[None, :]
    field = model.forward_diffuse_field(extinction, emission)
    position = jnp.array([.1, .8])[None, :, None]
    point = (field["incoming"]*jnp.exp(-position*field["optical_depth"])
             +position*field["emitted"]*_cell_weights(position*field["optical_depth"])[0])
    expected = jnp.einsum("a,acf->cf", field["angular_weight"], point)
    np.testing.assert_allclose(expected, model.forward_diffuse_at_fraction(
        extinction, emission, position[0, :, 0]), rtol=3e-14)
