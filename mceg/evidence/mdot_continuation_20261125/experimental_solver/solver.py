"""Host inexact Newton continuation with actual-residual backtracking.

The forcing target controls effort, not root acceptance. Products from older
states are used only in the right preconditioner. Final root checks are external.
"""
from __future__ import annotations

import numpy as np
from linear import fit, append, next_direction
from inverse import guarded_update, make_apply


def forcing(previous, ratio):
    """Safeguarded Eisenstat-Walker choice 2, with a practical lower floor."""
    eta = .9 * ratio**1.5
    safeguard = .9 * previous**1.5
    if safeguard > .1:
        eta = max(eta, safeguard)
    return float(np.clip(eta, .01, .7))


def fd_state(x, d, norm):
    delta = min(1e-4, max(3e-8, 1e-3*np.sqrt(norm)))
    h = delta / np.max(abs(d))
    return x+h*d, float(h)


def armijo(f, previous, slope, fraction):
    if not np.isfinite(f).all() or not np.isfinite(slope) or slope >= 0:
        return False
    return bool(f@f/2 <= previous@previous/2 + 1e-4*fraction*slope)


def solve(evaluate, x, f, *, valid, domain, checkpoint, event,
          S=None, Y=None, max_steps=32, max_basis=32, tolerance=1e-9):
    x, f = np.array(x, copy=True), np.array(f, copy=True)
    n=len(x)
    L, R = np.empty((n,0)), np.empty((0,n))
    if S is None:
        S=Y=np.empty((n,0))
    else:
        S,Y=np.array(S,copy=True),np.array(Y,copy=True)
    eta=.5
    accepted=0
    checkpoint(x,f,L,R,S,Y,x,f,eta)
    for iteration in range(max_steps):
        if np.max(abs(f)) < tolerance:
            return x,f,dict(converged=True,accepted=accepted,reason='root_candidate')
        norm=float(np.linalg.norm(f)); anchor_x=x.copy(); anchor_f=f.copy()
        apply, guard=make_apply(L,R)
        event('outer',dict(iteration=iteration,norm=norm,infinity=float(np.max(abs(f))),
                           forcing_target=eta,carried_current_columns=S.shape[1],preconditioner=guard))
        proposal=fit(S,Y,f) if S.shape[1] else None
        for k in range(max_basis):
            if proposal is not None and proposal['predicted_relative_residual']<=eta:
                break
            if S.shape[1]:
                d=next_direction(S,proposal['leftover'],apply)
            else:
                d=apply(-f);d/=np.max(abs(d))
            probe,h=fd_state(x,d,norm)
            if not domain(probe):
                return x,f,dict(converged=False,accepted=accepted,reason='fd_domain')
            fp,physical=evaluate(probe,'basis')
            if not valid(fp,physical):
                return x,f,dict(converged=False,accepted=accepted,reason='fd_physical')
            y=(fp-f)/h
            if S.shape[1]: S,Y=append(S,Y,d,y)
            else: S,Y=d[:,None],y[:,None]
            proposal=fit(S,Y,f)
            event('basis',dict(iteration=iteration,new_basis=k+1,columns=S.shape[1],
                               eta=proposal['predicted_relative_residual'],h=h,
                               condition=proposal['condition']))
            checkpoint(x,f,L,R,S,Y,anchor_x,anchor_f,eta)
        # The desired forcing is an effort target. Any measured eta<.9 must
        # still show actual Armijo decrease; no smaller final-root tolerance.
        if proposal is None or proposal['predicted_relative_residual']>=.9:
            return x,f,dict(converged=False,accepted=accepted,reason='no_descent_proposal')
        step=proposal['step']; probe,h=fd_state(x,step,norm)
        if not domain(probe):
            return x,f,dict(converged=False,accepted=accepted,reason='combined_domain')
        fp,physical=evaluate(probe,'combined')
        if not valid(fp,physical):
            return x,f,dict(converged=False,accepted=accepted,reason='combined_physical')
        measured=(fp-f)/h
        achieved=float(np.linalg.norm(f+measured)/norm);slope=float(f@measured)
        event('measured_direction',dict(iteration=iteration,forcing=achieved,slope=slope,
              target=eta,model_forcing=proposal['predicted_relative_residual'],
              model_disagreement=float(np.linalg.norm(measured-proposal['predicted'])/np.linalg.norm(measured))))
        if not np.isfinite([achieved,slope]).all() or achieved>=.9 or slope>=0:
            return x,f,dict(converged=False,accepted=accepted,reason='measured_direction_rejected')
        found=False
        for backtrack in range(12):
            fraction=2.**(-backtrack);trial=x+fraction*step
            if not domain(trial):continue
            ft,physical=evaluate(trial,'trial')
            ok=valid(ft,physical) and armijo(ft,f,slope,fraction)
            event('trial',dict(iteration=iteration,fraction=fraction,accepted=ok,
                  l2=float(np.linalg.norm(ft)),infinity=float(np.max(abs(ft)))))
            if ok:
                x,f=trial,ft;found=True;accepted+=1;break
        if not found:
            return x,f,dict(converged=False,accepted=accepted,reason='backtracking_failed')
        eta=forcing(eta,float(np.linalg.norm(f)/norm))
        # This approximate inverse is only a right map at the next state.
        L,R,update=guarded_update(L,R,S,Y)
        event('inverse_update',update)
        S=Y=np.empty((n,0))
        checkpoint(x,f,L,R,S,Y,x,f,eta)
    return x,f,dict(converged=bool(np.max(abs(f))<tolerance),accepted=accepted,reason='iteration_limit')
