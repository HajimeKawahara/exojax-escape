"""Least-squares trust step in a measured subspace, without a dense Jacobian.

Minimize ||F + Y c|| subject to ||S c|| <= radius. The QR transform makes
the constraint depend on the actual state update, not the arbitrary scaling
of the finite-difference basis. This is primal search algebra only.
"""
from __future__ import annotations

import numpy as np


def trust_step(directions, products, residual, radius):
    S, Y, F = map(np.asarray, (directions, products, residual))
    if S.ndim != 2 or S.shape != Y.shape or F.shape != (len(S),):
        raise ValueError('Incompatible subspace or residual shape.')
    if not all(np.isfinite(a).all() for a in (S, Y, F)) or not radius > 0:
        raise ValueError('Finite arrays and a positive radius are required.')
    Q, R = np.linalg.qr(S, mode='reduced')
    A = np.linalg.solve(R.T, Y.T).T
    U, singular, Vh = np.linalg.svd(A, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps * max(A.shape) * singular[0]:
        raise ValueError('The measured subspace is rank deficient.')
    projected = U.T @ F

    def coordinates(damping):
        return -Vh.T @ (singular * projected / (singular**2 + damping))

    damping = 0.
    z = coordinates(damping)
    if np.linalg.norm(z) > radius:
        lower, upper = 0., np.linalg.norm(A.T @ F) / radius
        for _ in range(80):
            middle = (lower + upper) / 2.
            if np.linalg.norm(coordinates(middle)) > radius:
                lower = middle
            else:
                upper = middle
        damping = upper
        z = coordinates(damping)
    step, prediction = Q @ z, F + A @ z
    return dict(step=step, predicted_residual=prediction,
                predicted_reduction=float((F @ F - prediction @ prediction) / 2.),
                norm=float(np.linalg.norm(step)), damping=float(damping), radius=radius)
