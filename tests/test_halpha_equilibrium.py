"""Actual-residual and rejection checks for the diagnostic continuation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples"))
from halpha_equilibrium import solve_broyden, solve_subspace


def test_inverse_secants_solve_coupled_nonlinear_root_with_actual_decrease():
    def residual(x):
        return np.array([x[0]**2+.2*x[1]-1.2, .3*x[0]+np.exp(x[1])-np.e-.3])
    seed = np.array([.8, .7])
    jacobian = np.array([[1.6, .2], [.3, np.exp(.7)]])
    inverse = np.linalg.inv(jacobian)
    accepted = []
    result = solve_broyden(lambda x, _: (residual(x), True), seed, residual(seed),
        inverse-np.eye(2), np.eye(2), domain=lambda x: np.all(x > 0),
        save=lambda x, f, *_: accepted.append(np.linalg.norm(f)), event=lambda *_: None)
    np.testing.assert_allclose(result[0], [1., 1.], atol=1e-9)
    assert result[-1]["converged"]
    assert np.all(np.diff(accepted) < 0)
    assert np.max(np.abs(residual(result[0]))) < 1e-9


def test_smaller_but_physically_invalid_residual_is_not_accepted():
    seed = np.array([1.])
    result = solve_broyden(lambda x, _: (x, False), seed, seed,
        np.empty((1, 0)), np.empty((0, 1)), domain=lambda x: True,
        save=lambda *_: None, event=lambda *_: None)
    assert not result[-1]["converged"]
    assert result[-1]["accepted"] == 0
    np.testing.assert_array_equal(result[0], seed)


def test_wrong_inverse_cannot_hide_a_failed_residual_step():
    seed = np.array([1.])
    result = solve_broyden(lambda x, _: (x, True), seed, seed,
        np.array([[-2.]]), np.ones((1, 1)), domain=lambda x: True,
        save=lambda *_: None, event=lambda *_: None)
    assert not result[-1]["converged"]
    assert result[-1]["accepted"] == 0


def test_current_measurement_repairs_a_wrong_inverse():
    matrix = np.array([[2., .4], [.3, 1.]])
    residual = lambda x: matrix@(x-np.ones(2))
    seed = np.array([.8, .7])
    kinds, accepted = [], []
    def evaluate(x, kind):
        kinds.append(kind)
        return residual(x), True
    result = solve_subspace(evaluate, seed, residual(seed), -2*np.eye(2), np.eye(2),
        domain=lambda x: True, save=lambda x, f, *_: accepted.append(np.linalg.norm(f)),
        event=lambda *_: None, forcing=.01)
    np.testing.assert_allclose(result[0], [1., 1.], atol=1e-9)
    assert result[-1]["converged"]
    assert kinds[:2] == ["basis", "basis"]
    assert np.all(np.diff(accepted) <= 0)


def test_invalid_finite_difference_probe_is_rejected():
    seed = np.ones(2)
    result = solve_subspace(lambda x, _: (x, False), seed, seed,
        np.empty((2, 0)), np.empty((0, 2)), domain=lambda x: True,
        save=lambda *_: None, event=lambda *_: None)
    assert not result[-1]["converged"]
    assert result[-1]["reason"] == "probe_physical"
    np.testing.assert_array_equal(result[0], seed)


def test_current_subspace_solves_a_coupled_nonlinear_root():
    residual = lambda x: np.array([x[0]**2+.2*x[1]-1.2, .3*x[0]+np.exp(x[1])-np.e-.3])
    seed = np.array([.8, .7])
    result = solve_subspace(lambda x, _: (residual(x), True), seed, residual(seed),
        np.empty((2, 0)), np.empty((0, 2)), domain=lambda x: np.all(x > 0),
        save=lambda *_: None, event=lambda *_: None)
    assert result[-1]["converged"]
    np.testing.assert_allclose(result[0], [1., 1.], atol=1e-9)
