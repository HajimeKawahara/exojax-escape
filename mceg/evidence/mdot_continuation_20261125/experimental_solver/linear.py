"""Host algebra for a primal, fixed-current-point finite-difference subspace."""
from __future__ import annotations

import numpy as np


def _bank(s,y):
    s=np.asarray(s,dtype=float);y=np.asarray(y,dtype=float)
    assert s.ndim==2 and s.shape==y.shape and s.shape[0]>s.shape[1]>0
    assert np.isfinite(s).all() and np.isfinite(y).all()
    return s,y


def fit(S,Y,F):
    """Minimize the unscaled original residual; products must be current FD rows."""
    S,Y=_bank(S,Y);F=np.asarray(F,dtype=float)
    assert F.shape==(len(S),) and np.isfinite(F).all() and np.linalg.norm(F)>0
    scales=np.linalg.norm(Y,axis=0)
    assert np.isfinite(scales).all() and np.all(scales>0)
    c,_,rank,singular=np.linalg.lstsq(Y/scales,-F,rcond=np.finfo(float).eps*max(Y.shape))
    condition=float(singular[0]/singular[-1]);amplification=float(np.sum(abs(c))/np.linalg.norm(F))
    assert rank==Y.shape[1] and np.isfinite([condition,amplification]).all()
    assert condition<1e8 and amplification<1e4
    coefficients=c/scales;step=S@coefficients;predicted=Y@coefficients;leftover=F+predicted
    assert np.isfinite(step).all() and np.isfinite(leftover).all()
    return dict(step=step,predicted=predicted,leftover=leftover,coefficients=coefficients,
                rank=int(rank),condition=condition,amplification=amplification,
                predicted_relative_residual=float(np.linalg.norm(leftover)/np.linalg.norm(F)))


def make_preconditioner(oldS,oldY):
    """Fixed P=I+(S0-Y0)Y0^+, with measured small-block bounds or identity.

    On the orthogonal complement of span(S0,Y0), P is identity. Old products
    supply only this right map; they never enter the current FD product bank.
    """
    S,Y=_bank(oldS,oldY)
    report=dict(kind='identity',fixed=True,current_derivative_claim=False,
                maximum_norm=1e4,maximum_condition=1e8)
    def identity(v):
        v=np.asarray(v,dtype=float)
        assert v.shape==(len(S),) and np.isfinite(v).all()
        return v.copy()
    try:
        scales=np.linalg.norm(Y,axis=0);snorm=np.linalg.norm(S,axis=0)
        assert np.isfinite(scales).all() and np.all(scales>0) and np.all(snorm>0)
        A=Y/scales;L=(S-Y)/scales
        q,sy,vh=np.linalg.svd(A,full_matrices=False)
        assert sy[-1]>np.finfo(float).eps*max(A.shape)*sy[0]
        assert sy[0]/sy[-1]<1e8
        R=(vh.T/sy)@q.T
        span=np.column_stack((S/snorm,A))
        u,ss,_=np.linalg.svd(span,full_matrices=False)
        rank=int(np.sum(ss>ss[0]*np.finfo(float).eps*max(span.shape)))
        U=u[:,:rank]
        LU=U.T@L;RU=R@U
        block=np.eye(rank)+LU@RU
        singular=np.linalg.svd(block,compute_uv=False)
        # Conservative floating reconstruction allowance for discarded numerical
        # span components, using low-rank operator norm inequalities.
        remainder=float(np.linalg.norm(L-U@LU)*np.linalg.norm(R,2)
                        +np.linalg.norm(LU,2)*np.linalg.norm(R-RU@U.T))
        upper=float(max(1.,singular[0])+remainder)
        lower=float(min(1.,singular[-1])-remainder)
        assert np.isfinite([upper,lower,remainder]).all() and lower>0
        condition=float(upper/lower)
        report.update(span_rank=rank,old_output_condition=float(sy[0]/sy[-1]),
                      compact_norm=float(max(1.,singular[0])),compact_minimum_singular=float(min(1.,singular[-1])),
                      reconstruction_remainder_bound=remainder,norm_bound=upper,
                      minimum_singular_bound=lower,condition_bound=condition)
        assert upper<1e4 and condition<1e8
        report.update(kind='old13_right_map',reason='finite_small_span_norm_and_invertibility_checks_passed')
    except (AssertionError,np.linalg.LinAlgError,ValueError) as error:
        report.update(reason='identity_fallback_'+type(error).__name__,selected_norm_bound=1.,selected_condition_bound=1.)
        return identity,report
    report.update(selected_norm_bound=upper,selected_condition_bound=condition)
    def apply(v):
        v=np.asarray(v,dtype=float)
        assert v.shape==(len(S),) and np.isfinite(v).all()
        result=v+L@(R@v)
        assert np.isfinite(result).all()
        assert np.linalg.norm(result)<=upper*np.linalg.norm(v)*(1+1e-10)+1e-300
        return result
    return apply,report


def _state_span(S):
    S=np.asarray(S,dtype=float)
    assert S.ndim==2 and S.shape[0]>S.shape[1]>0 and np.isfinite(S).all()
    scales=np.linalg.norm(S,axis=0)
    assert np.isfinite(scales).all() and np.all(scales>0)
    U,singular,_=np.linalg.svd(S/scales,full_matrices=False)
    assert singular[-1]>np.finfo(float).eps*max(S.shape)*singular[0]
    assert singular[0]/singular[-1]<1e8
    return U


def next_direction(S,leftover,apply_fn):
    """Fixed right map of remaining residual, independent of existing states."""
    U=_state_span(S);leftover=np.asarray(leftover,dtype=float)
    assert leftover.shape==(len(S),) and np.isfinite(leftover).all()
    raw=np.asarray(apply_fn(-leftover),dtype=float);direction=raw.copy()
    assert direction.shape==leftover.shape and np.isfinite(direction).all()
    for _ in range(2):direction-=U@(U.T@direction)
    norm=float(np.linalg.norm(direction))
    assert np.isfinite(norm) and norm>1e-10*max(float(np.linalg.norm(raw)),1e-300)
    direction/=np.max(abs(direction))
    assert np.isfinite(direction).all() and max(abs(direction))==1.
    return direction


def append(S,Y,d,product):
    """Append only a saved current-point FD product, after physical guards."""
    S,Y=_bank(S,Y);d=np.asarray(d,dtype=float);product=np.asarray(product,dtype=float)
    assert d.shape==product.shape==(len(S),) and np.isfinite(d).all() and np.isfinite(product).all()
    U=_state_span(S);independent=d-U@(U.T@d)
    assert np.linalg.norm(independent)>1e-10*max(np.linalg.norm(d),1e-300)
    return np.column_stack((S,d)),np.column_stack((Y,product))


def fd_trial(x,d,max_coordinate_step=1e-4):
    x=np.asarray(x,dtype=float);d=np.asarray(d,dtype=float)
    assert x.ndim==1 and x.shape==d.shape and np.isfinite(x).all() and np.isfinite(d).all()
    assert max_coordinate_step==1e-4 and max(abs(d))>0
    h=float(max_coordinate_step/max(abs(d)));trial=x+h*d
    assert np.isfinite(trial).all()
    return trial,h


def fresh_check(F,product,model_product):
    """Measured combined finite-difference forcing; no exact-linearity test."""
    F=np.asarray(F,dtype=float);product=np.asarray(product,dtype=float);model=np.asarray(model_product,dtype=float)
    assert F.ndim==1 and F.shape==product.shape==model.shape
    finite=bool(np.isfinite(F).all() and np.isfinite(product).all() and np.isfinite(model).all() and np.linalg.norm(F)>0)
    if not finite:return dict(passed=False,finite=False,forcing=None,model_disagreement_relative=None,measured_fd_slope=None,current_exact_derivative_claim=False)
    forcing=float(np.linalg.norm(F+product)/np.linalg.norm(F))
    disagreement=float(np.linalg.norm(product-model)/max(np.linalg.norm(product),1e-300))
    slope=float(np.dot(F.astype(np.longdouble),product.astype(np.longdouble)))
    return dict(passed=bool(np.isfinite([forcing,disagreement,slope]).all() and forcing<=.2 and slope<0),
                finite=True,forcing=forcing,model_disagreement_relative=disagreement,
                measured_fd_slope=slope,current_exact_derivative_claim=False)


def progress_guard(before_eta,after_eta,rank_before,rank_after):
    return bool(np.isfinite([before_eta,after_eta]).all() and before_eta>0
                and after_eta<=.9*before_eta and rank_after>rank_before)
