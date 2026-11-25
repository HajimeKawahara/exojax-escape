"""Current-FD subspace solve with an actual-residual trust-region test.

The radius controls numerical updates only. It changes neither the physical
equations nor the final maximum-residual criterion.
"""
from __future__ import annotations

import numpy as np

from inverse import guarded_update, make_apply
from linear import append, fit, next_direction
from solver import forcing


def fd_state(x, d, norm):
    delta = min(1e-5, max(3e-8, 1e-3*np.sqrt(norm)))
    h = delta / np.max(abs(d))
    return x+h*d, float(h)
from trust_region import trust_step


def solve(evaluate, x, f, *, valid, domain, checkpoint, event,
          S=None, Y=None, L=None, R=None, eta=.05, radius=None,
          max_steps=64, max_basis=32, tolerance=1e-9):
    x, f = np.array(x, copy=True), np.array(f, copy=True)
    n = len(x)
    if L is None:
        L, R = np.empty((n, 0)), np.empty((0, n))
    if S is None:
        S = Y = np.empty((n, 0))
    else:
        S, Y = np.array(S, copy=True), np.array(Y, copy=True)
    accepted = 0

    def save():
        checkpoint(x, f, L, R, S, Y, x, f, eta, radius)

    save()
    for iteration in range(max_steps):
        if np.max(abs(f)) < tolerance:
            return x, f, dict(converged=True, accepted=accepted, reason='root_candidate')
        norm = float(np.linalg.norm(f))
        apply, guard = make_apply(L, R)
        event('outer', dict(iteration=iteration, norm=norm, infinity=float(np.max(abs(f))),
              forcing_target=eta, carried_current_columns=S.shape[1], radius=radius,
              preconditioner=guard))
        proposal = fit(S, Y, f) if S.shape[1] else None
        for k in range(max_basis):
            if proposal is not None and proposal['predicted_relative_residual'] <= eta:
                break
            if S.shape[1]:
                direction = next_direction(S, proposal['leftover'], apply)
            else:
                direction = apply(-f)
                direction /= np.max(abs(direction))
            probe, h = fd_state(x, direction, norm)
            if not domain(probe):
                return x, f, dict(converged=False, accepted=accepted, reason='fd_domain')
            fp, physical = evaluate(probe, 'basis')
            if not valid(fp, physical):
                return x, f, dict(converged=False, accepted=accepted, reason='fd_physical')
            product = (fp - f) / h
            if S.shape[1]:
                S, Y = append(S, Y, direction, product)
            else:
                S, Y = direction[:, None], product[:, None]
            proposal = fit(S, Y, f)
            event('basis', dict(iteration=iteration, new_basis=k+1, columns=S.shape[1],
                  eta=proposal['predicted_relative_residual'], h=h,
                  condition=proposal['condition']))
            save()
        if proposal is None or proposal['predicted_relative_residual'] >= 1.:
            return x, f, dict(converged=False, accepted=accepted, reason='no_descent_model')
        step = proposal['step']
        probe, h = fd_state(x, step, norm)
        if not domain(probe):
            return x, f, dict(converged=False, accepted=accepted, reason='combined_domain')
        fp, physical = evaluate(probe, 'combined')
        if not valid(fp, physical):
            return x, f, dict(converged=False, accepted=accepted, reason='combined_physical')
        measured = (fp-f)/h
        measured_eta = float(np.linalg.norm(f+measured)/norm)
        event('measured_direction', dict(iteration=iteration, h=h,
              model_forcing=proposal['predicted_relative_residual'], forcing=measured_eta,
              disagreement=float(np.linalg.norm(measured-proposal['predicted'])/norm)))
        if not np.isfinite(measured_eta) or measured_eta > .1:
            save()
            return x, f, dict(converged=False, accepted=accepted, reason='direction_accuracy')
        if radius is None:
            radius = float(np.linalg.norm(proposal['step']))
        found = False
        for trial_index in range(16):
            model = trust_step(S, Y, f, radius)
            step, predicted = model['step'], model['predicted_reduction']
            trial = x + step
            if predicted <= 0. or np.array_equal(trial, x):
                return x, f, dict(converged=False, accepted=accepted, reason='stationary_subspace')
            if not domain(trial):
                event('domain_rejected', dict(iteration=iteration, trial=trial_index, radius=radius))
                radius = .25 * model['norm']
                continue
            ft, physical = evaluate(trial, 'trial')
            physical_valid = valid(ft, physical)
            actual = float((f @ f - ft @ ft) / 2.) if physical_valid else None
            ratio = actual / predicted if actual is not None else None
            ok = bool(physical_valid and actual > 0. and ratio > .1)
            old_radius = radius
            if ratio is None or ratio < .25:
                radius = .25 * model['norm']
            elif ratio > .75 and model['norm'] >= .95 * radius:
                radius *= 2.
            event('trust_trial', dict(iteration=iteration, trial=trial_index, accepted=ok,
                  radius=old_radius, next_radius=radius, norm=model['norm'],
                  damping=model['damping'], predicted_reduction=predicted,
                  actual_reduction=actual, ratio=ratio,
                  l2=float(np.linalg.norm(ft)) if np.isfinite(ft).all() else None,
                  infinity=float(np.max(abs(ft))) if np.isfinite(ft).all() else None))
            if ok:
                x, f = trial, ft
                found = True
                accepted += 1
                break
        if not found:
            save()
            return x, f, dict(converged=False, accepted=accepted, reason='trust_trials_failed')
        eta = min(.05, forcing(eta, float(np.linalg.norm(f) / norm)))
        L, R, update = guarded_update(L, R, S, Y)
        event('inverse_update', update)
        S = Y = np.empty((n, 0))
        save()
    return x, f, dict(converged=bool(np.max(abs(f)) < tolerance), accepted=accepted, reason='iteration_limit')
