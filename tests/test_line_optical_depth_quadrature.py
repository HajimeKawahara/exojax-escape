"""Skin resolution and moving source-fit boundaries in line integrals."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel
from escape.line_escape_probability import (
    CORE, cell_averaged_line_operators, federman_shield, lya_one_side,
    optical_depth_cell_quadrature,
)

jax.config.update("jax_enable_x64", True)


def test_illuminated_skin_matches_independent_optical_depth_integral():
    depth, damping = 138149.6, .000550008
    nodes, weights, error = optical_depth_cell_quadrature(
        jnp.array([0.]), 2*depth, jnp.array([depth]), jnp.array([.03]),
        jnp.array([1.54e-11]), order=16)
    point = jax.jit(lambda tau: federman_shield(
        tau, damping, average_zone=False, shielding_model="continuous_core"))
    actual = jnp.sum(weights[:, 0]*point(depth*nodes[:, 0]))
    breaks = [0., np.log1p(2), np.log1p(10), np.log1p(100), np.log1p(depth)]
    expected = sum(quad(lambda y: float(point(np.expm1(y)))*np.exp(y)/depth,
                        lo, hi, epsabs=1e-15, epsrel=1e-10)[0]
                   for lo, hi in zip(breaks[:-1], breaks[1:]))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    assert np.max(abs(error)) < 1e-10
    assert np.min(weights) >= 0
    np.testing.assert_allclose(weights.sum(0), 1., rtol=2e-15)


def test_moving_lya_fit_boundary_derivative_matches_adaptive_integral():
    front, depth, total, beta = 138149.6, 3.10056173e8, 7.1e12, 2.37898665e-11
    critical = .001/(beta*np.sqrt(np.pi))

    def evaluate(log_column):
        factor = jnp.exp(log_column)
        positions, weights, _ = optical_depth_cell_quadrature(
            jnp.array([front*factor]), total*factor,
            jnp.array([depth*factor]), jnp.array([.01]), jnp.array([beta]), 16)
        inward = (front+positions[:, 0]*depth)*factor
        escape = .5*(lya_one_side(inward, beta)[0]
                      +lya_one_side(total*factor-inward, beta)[0])
        return jnp.sum(weights[:, 0]*escape)

    point = jax.jit(lambda u, factor: .5*(
        lya_one_side((front+u*depth)*factor, beta)[0]
        +lya_one_side((total-front-u*depth)*factor, beta)[0]))

    def independent(log_column):
        factor = np.exp(log_column)
        boundary = (critical/factor-front)/depth
        return sum(quad(lambda u: float(point(u, factor)), lo, hi,
                        epsabs=1e-20, epsrel=1e-10)[0]
                   for lo, hi in zip([0., boundary], [boundary, 1.]))

    step = 1e-4
    derivative = jax.grad(evaluate)(0.)
    finite = (independent(step)-independent(-step))/(2*step)
    np.testing.assert_allclose(derivative, finite, rtol=3e-5)


def test_cellwise_diffuse_fraction_uses_each_cells_own_position():
    model = HydrogenEscapeProbabilityModel([0., 1., 3.], nmax=3, angle_order=4)
    chi, eta = jnp.array([[.1, 2.], [.5, 1.]]), jnp.array([[.4, 3.], [.7, .2]])
    fractions = jnp.array([.01, .9])
    actual = model.forward_diffuse_at_fraction(chi, eta, fractions)
    first = model.forward_diffuse_at_fraction(chi, eta, fractions[0])
    second = model.forward_diffuse_at_fraction(chi, eta, fractions[1])
    np.testing.assert_allclose(actual[0], first[0], rtol=2e-15)
    np.testing.assert_allclose(actual[1], second[1], rtol=2e-15)


def test_depth_rule_is_explicit_and_requires_cell_averaging():
    import pytest

    default = HydrogenEscapeProbabilityModel([0., 1.], nmax=3)
    assert default.line_quadrature == "gauss"
    with pytest.raises(ValueError, match="average sampling"):
        HydrogenEscapeProbabilityModel([0., 1.], nmax=3, line_quadrature="optical_depth")


def test_zero_depth_clipped_segments_and_node_broadcast_validation():
    import pytest
    from test_line_cell_average import _arguments

    nodes, weights, error = optical_depth_cell_quadrature(
        jnp.array([0., 10.]), jnp.array([0., 20.]), jnp.zeros(2),
        jnp.zeros(2), jnp.zeros(2), order=8)
    assert np.isfinite(nodes).all() and np.isfinite(weights).all()
    assert np.all((nodes >= 0) & (nodes <= 1))
    np.testing.assert_allclose(weights.sum(0), 1., rtol=2e-15)
    assert np.max(abs(error)) < 1e-12
    args = _arguments(tau_in=jnp.zeros((2, 1)),tau_pump=jnp.zeros((2, 1)),
                      tau_total=jnp.zeros((2, 1)))
    value = cell_averaged_line_operators(CORE,line_optical_depth_increment=0.,
        positions=nodes[..., None],weights=weights[..., None],**args)
    assert value.escape_probability.shape == (2, 1)
    np.testing.assert_allclose(value.escape_probability, 1., rtol=2e-15)
    with pytest.raises(ValueError, match="both"):
        cell_averaged_line_operators(CORE,line_optical_depth_increment=0.,positions=nodes,**args)
    with pytest.raises(ValueError, match="identical"):
        cell_averaged_line_operators(CORE,line_optical_depth_increment=0.,
            positions=nodes,weights=weights[:, :1],**args)
