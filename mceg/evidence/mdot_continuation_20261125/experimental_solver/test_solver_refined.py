import numpy as np

from solver_refined import solve


def run(fun, initial, **options):
    events = []
    x = np.asarray(initial, float)
    value, residual, result = solve(lambda x, kind: (fun(x), {}), x, fun(x),
        valid=lambda f, p: bool(np.isfinite(f).all()), domain=lambda x: bool(np.isfinite(x).all()),
        checkpoint=lambda *args: None, event=lambda kind, d: events.append((kind, d)), **options)
    return value, residual, result, events


def test_rosenbrock_root_through_nonlinear_valley():
    def fun(x):
        return np.array([10*(x[1]-x[0]**2), 1-x[0], x[2]])
    x, f, result, events = run(fun, [-1.2, 1., 0.], radius=.5, max_basis=2)
    assert result['converged'] and max(abs(f)) < 1e-9
    np.testing.assert_allclose(x, [1., 1., 0.], atol=1e-9)
    assert any(kind == 'trust_trial' and not d['accepted'] for kind, d in events)


def test_keeps_large_well_constrained_correction_with_small_radius():
    J = np.diag([1., .001, 1.])
    S = np.eye(3)[:, :2]
    x, f, result, events = run(lambda x: J @ x, [1., 1000., 0.],
        S=S, Y=J @ S, radius=1., max_steps=1)
    assert result['accepted'] == 1 and np.linalg.norm(f) < 1.01
    assert np.linalg.norm(x - [1., 1000., 0.]) <= 1. + 1e-12
    assert not result['converged']


def test_invalid_trial_is_rejected_even_when_its_residual_vanishes():
    initial = np.ones(3)
    calls = []
    def evaluate(x, kind):
        calls.append(kind)
        return (np.zeros(3), {'valid': False}) if kind == 'trial' else (x, {'valid': True})
    x, f, result = solve(evaluate, initial, initial,
        valid=lambda f, p: p['valid'], domain=lambda x: True,
        checkpoint=lambda *args: None, event=lambda *args: None, max_steps=1)
    assert calls.count('trial') > 0 and result['accepted'] == 0 and not result['converged']
    np.testing.assert_array_equal(x, initial)


def test_inaccurate_linear_model_is_not_used_for_a_trial():
    initial = np.array([1., 1., 0.])
    directions = np.eye(3)[:, :2]
    x, f, result, events = run(lambda x: x, initial,
        S=directions, Y=2*directions, radius=2.)
    assert result['reason'] == 'direction_accuracy'
    assert not result['converged'] and result['accepted'] == 0
    assert not any(kind == 'trust_trial' for kind, d in events)
    np.testing.assert_array_equal(x, initial)
