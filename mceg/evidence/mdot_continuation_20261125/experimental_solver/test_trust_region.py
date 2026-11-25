import numpy as np

from trust_region import trust_step


def test_constrained_solution_satisfies_kkt_conditions():
    S = np.array([[2., .1], [.2, 3.], [0., 1.]])
    J = np.array([[2., .2, .3], [.1, .001, .02], [.3, .1, 1.]])
    F = np.array([1., 2., -1.])
    result = trust_step(S, J @ S, F, .3)
    d = result['step']
    assert result['damping'] > 0
    np.testing.assert_allclose(np.linalg.norm(d), .3, rtol=1e-12)
    gradient = S.T @ (J.T @ (F + J @ d) + result['damping'] * d)
    np.testing.assert_allclose(gradient, 0., atol=1e-12)


def test_column_scaling_does_not_change_physical_step():
    S = np.array([[2., .1], [.2, 3.], [0., 1.]])
    Y = np.array([[3., .2], [.3, .5], [1., -.1]])
    F = np.array([1., 2., 3.])
    original = trust_step(S, Y, F, .5)
    changed = trust_step(S * [1e-3, 1e3], Y * [1e-3, 1e3], F, .5)
    np.testing.assert_allclose(original['step'], changed['step'], atol=1e-12)


def test_trust_step_improves_on_uniformly_shrunk_ill_conditioned_step():
    S = np.eye(3)[:, :2]
    Y = S * [1., .001]
    F = np.array([1., 1., 0.])
    full = trust_step(S, Y, F, np.inf)['step']
    shrunk = full / np.linalg.norm(full)
    result = trust_step(S, Y, F, 1.)
    assert np.linalg.norm(result['predicted_residual']) < .8 * np.linalg.norm(F + np.diag([1., .001, 1.]) @ shrunk)
    assert result['norm'] <= 1. + 1e-12


def test_unbounded_step_matches_exact_linear_root():
    S = np.array([[1., 0.], [0., 1.], [0., 0.]])
    Y = S * [2., 3.]
    F = np.array([4., 9., 0.])
    result = trust_step(S, Y, F, np.inf)
    np.testing.assert_allclose(result['step'], [-2., -3., 0.], atol=1e-13)
    np.testing.assert_allclose(result['predicted_residual'], 0., atol=1e-13)
