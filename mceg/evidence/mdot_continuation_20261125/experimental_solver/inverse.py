"""Guarded retained low-rank right inverse; no current-derivative claim."""
from __future__ import annotations

import numpy as np


def identity(n):
    assert isinstance(n, int) and n > 0
    return np.empty((n, 0)), np.empty((0, n))


def _factors(L, R):
    L, R = np.asarray(L, dtype=float), np.asarray(R, dtype=float)
    assert L.ndim == R.ndim == 2 and L.shape[0] > 0
    assert R.shape == (L.shape[1], L.shape[0])
    assert np.isfinite(L).all() and np.isfinite(R).all()
    return L, R


def assess(L, R):
    """Bound P=I+LR on span(L,R.T), including discarded-span remainder."""
    report = dict(passed=False, maximum_norm=1e4, maximum_condition=1e8,
                  current_derivative_claim=False)
    try:
        L, R = _factors(L, R)
        norms = np.r_[np.linalg.norm(L, axis=0), np.linalg.norm(R, axis=1)]
        assert np.isfinite(norms).all()
        span = np.column_stack((L, R.T))
        active = norms > 0
        if not np.any(active):
            report.update(passed=True, span_rank=0, norm_bound=1.,
                          minimum_singular_bound=1., condition_bound=1.,
                          reconstruction_remainder_bound=0., reason='identity')
            return report
        U, singular, _ = np.linalg.svd(span[:, active] / norms[active], full_matrices=False)
        rank = int(np.sum(singular > singular[0] * np.finfo(float).eps * max(span.shape)))
        U = U[:, :rank]
        LU, RU = U.T @ L, R @ U
        values = np.linalg.svd(np.eye(rank) + LU @ RU, compute_uv=False)
        remainder = float(np.linalg.norm(L-U@LU) * np.linalg.norm(R, 2)
                          + np.linalg.norm(LU, 2) * np.linalg.norm(R-RU@U.T))
        upper = float(max(1., values[0]) + remainder)
        lower = float(min(1., values[-1]) - remainder)
        assert np.isfinite([upper, lower, remainder]).all() and lower > 0
        condition = upper / lower
        report.update(span_rank=rank, norm_bound=upper, minimum_singular_bound=lower,
                      condition_bound=condition, reconstruction_remainder_bound=remainder)
        assert upper < 1e4 and condition < 1e8
        report.update(passed=True, reason='compact_norm_and_invertibility_passed')
    except (AssertionError, ValueError, np.linalg.LinAlgError) as error:
        report['reason'] = 'guard_rejected_' + type(error).__name__
    return report


def make_apply(L, R):
    """Freeze a safe right map for one inner solve."""
    L, R = _factors(L, R)
    report = assess(L, R)
    assert report['passed']
    L, R = L.copy(), R.copy()
    L.flags.writeable = R.flags.writeable = False

    def apply(v):
        v = np.asarray(v, dtype=float)
        assert v.shape == (L.shape[0],) and np.isfinite(v).all()
        result = v + L @ (R @ v)
        assert np.isfinite(result).all()
        assert np.linalg.norm(result) <= report['norm_bound'] * np.linalg.norm(v) * (1+1e-10) + 1e-300
        return result
    return apply, report


def guarded_update(L, R, S, Y):
    """Pnew=P+(S-PY)Y+; unsafe proposals retain safe P, otherwise identity.

    S/Y must contain only independently measured directions at ONE anchor.
    The caller owns that provenance and excludes the dependent combined FD.
    """
    L, R = np.asarray(L, dtype=float), np.asarray(R, dtype=float)
    assert L.ndim == 2 and L.shape[0] > 0
    n = L.shape[0]
    previous = assess(L, R)
    if previous['passed']:
        safe_L, safe_R = L.copy(), R.copy()
        fallback = 'last_safe'
    else:
        safe_L, safe_R = identity(n)
        fallback = 'identity'
    report = dict(accepted=False, fallback=fallback, previous_guard=previous,
                  current_derivative_claim=False)
    try:
        S, Y = np.asarray(S, dtype=float), np.asarray(Y, dtype=float)
        assert S.ndim == Y.ndim == 2 and S.shape == Y.shape
        assert S.shape[0] == n and 0 < S.shape[1] < n
        assert np.isfinite(S).all() and np.isfinite(Y).all()
        sn, yn = np.linalg.norm(S, axis=0), np.linalg.norm(Y, axis=0)
        assert np.isfinite(sn).all() and np.isfinite(yn).all() and np.all(sn > 0) and np.all(yn > 0)
        for matrix in (S/sn, Y/yn):
            singular = np.linalg.svd(matrix, compute_uv=False)
            assert singular[-1] > singular[0] * np.finfo(float).eps * max(matrix.shape)
            assert singular[0] / singular[-1] < 1e8
        A = Y/yn
        U, singular, Vh = np.linalg.svd(A, full_matrices=False)
        right = (Vh.T/singular) @ U.T
        correction = (S - (Y + safe_L @ (safe_R @ Y))) / yn
        candidate_L = np.column_stack((safe_L, correction))
        candidate_R = np.vstack((safe_R, right))
        guard = assess(candidate_L, candidate_R)
        report.update(candidate_guard=guard, added_columns=S.shape[1])
        assert guard['passed']
        reconstructed = Y + candidate_L @ (candidate_R @ Y)
        error = float(np.linalg.norm(reconstructed-S) / np.linalg.norm(S))
        assert np.isfinite(error) and error < 1e-10
        report.update(accepted=True, fallback=None, selected_guard=guard,
                      secant_relative_error=error, reason='same_anchor_multisecant_update')
        return candidate_L, candidate_R, report
    except (AssertionError, ValueError, np.linalg.LinAlgError) as error:
        report.update(reason='update_rejected_'+type(error).__name__,
                      selected_guard=assess(safe_L, safe_R))
        return safe_L, safe_R, report
