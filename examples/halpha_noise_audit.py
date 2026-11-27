"""Audit the archived H-alpha noise proxy without solving an atmosphere.

Correlations and simulation envelopes are discrepancy diagnostics, not calibrated
significance tests. Posterior covariance draws are a sensitivity analysis of a
fixed spectrum, not a new posterior for atmospheric physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from gp_halpha import load_gp_data, make_gp_problem
from gp_controls import hyper_fit, subset
from halpha_geometry_fit import WINDOWS
from phase_halpha import REST_ANGSTROM
from escape.observation import C_KMS
from escape.time_gp import prepare_time_gp, time_gp_statistics, time_gp_posterior


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covariance(state, group, noise, index):
    """Return prior, posterior, and coefficient-estimation matrices."""
    scale = np.exp(noise["log_white"][index])
    length = np.exp(noise["log_length"][index])
    amp = np.exp(noise["log_gp"][index])
    distance = np.sqrt(3) * abs(group["time_min"][:, None] - group["time_min"][None]) / length
    correlation = (1 + distance) * np.exp(-distance) + 1e-10 * np.eye(len(distance))
    factor = la.block_diag(*[a * la.cholesky(correlation, lower=True) for a in amp])
    system = np.eye(len(factor)) + factor.T @ np.asarray(state.gram) @ factor / scale**2
    posterior = factor @ la.solve(system, factor.T, assume_a="pos")
    return dict(prior=factor @ factor.T, factor=factor, posterior=posterior,
                estimator=posterior / scale**2, correction=posterior / scale**4, scale=scale)


def centered(group, data):
    weight = group["weight"]
    total = weight.sum(axis=0)
    common = (weight * np.where(weight > 0, data, 0.)).sum(axis=0) / np.where(total > 0,total,1.)
    return np.where(weight > 0, data - common, 0.)


def coefficient_mean(group, state, cov, data):
    product = np.einsum("ep,ep,mp->me", group["weight"], centered(group, data), state.basis)
    return (cov["estimator"] @ product.ravel()).reshape(state.basis.shape[0], -1)


def residual(group, state, cov, data):
    mean = coefficient_mean(group, state, cov, data)
    return centered(group, data - mean.T @ np.asarray(state.basis)) * np.sqrt(group["weight"]) / cov["scale"]


def correlation(left, right, valid):
    a, b = np.asarray(left)[valid], np.asarray(right)[valid]
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def correlation_statistics(group, value):
    """Use exact native spectral lags and consecutive retained exposure lags."""
    weight, pixels = group["weight"], group["pixel_index"]
    result = {}
    for lag in (1, 2, 4, 8):
        right = np.searchsorted(pixels, pixels + lag)
        left = np.flatnonzero(right < len(pixels))
        left = left[pixels[right[left]] == pixels[left] + lag]
        right = right[left]
        if len(left):
            valid = (weight[:, left] > 0) & (weight[:, right] > 0)
            result[f"spectral_native_lag{lag}"] = correlation(value[:, left], value[:, right], valid)
    cadence = np.median(np.diff(group["time_min"]))
    for lag in (1, 2, 3):
        gap = group["time_min"][lag:] - group["time_min"][:-lag]
        valid = (weight[lag:] > 0) & (weight[:-lag] > 0) & (gap[:, None] < 1.6 * lag * cadence)
        result[f"time_exposure_lag{lag}"] = correlation(value[:-lag], value[lag:], valid)
    # A quadratic spectral coefficient orthogonal to grey and slope at each time.
    x = (group["stellar_velocity"] - 250) / 140
    basis = np.stack((np.ones_like(x), x), axis=1)
    quadratic = .5 * (3 * x*x - 1)
    z = []
    for row, w in zip(value, weight):
        q = quadratic - basis @ la.solve(basis.T @ (w[:, None] * basis), basis.T @ (w * quadratic), assume_a="pos")
        z.append(np.sum(q * np.sqrt(w) * row) / np.sqrt(np.sum(w*q*q)))
    result["orthogonal_quadratic_coefficient_mean_square"] = float(np.mean(np.array(z)**2))
    return result


def simulate(group, state, cov, rng):
    modes, exposures = state.basis.shape[0], len(group["phase"])
    coefficients = (cov["factor"] @ rng.normal(size=modes*exposures)).reshape(modes, exposures)
    sigma = np.divide(cov["scale"], np.sqrt(group["weight"]), out=np.zeros_like(group["weight"]), where=group["weight"] > 0)
    return coefficients.T @ state.basis + sigma * rng.normal(size=sigma.shape)


def archive_errors(paths):
    records = []
    for night, path in enumerate(paths):
        with h5py.File(path) as archive:
            flux, error, wave = (archive[key][:] for key in ("flux", "err", "wave"))
        for order in range(len(wave)):
            velocity = C_KMS * (wave[order] / REST_ANGSTROM - 1)
            keep = (velocity >= 110) & (velocity <= 390)
            values, errors = flux[:, order, keep], error[:, order, keep]
            valid = np.isfinite(values).all(axis=0) & np.isfinite(errors).all(axis=0) & (errors > 0).all(axis=0)
            values, errors = values[:, valid], errors[:, valid]
            median = np.median(errors, axis=0)
            scale = np.median(errors / median, axis=1)
            records.append(dict(night=night, order=order, complete_native_pixels=int(valid.sum()),
                err_over_full_time_std_quantiles05_50_95=np.quantile(errors/np.std(values, axis=0), [.05,.5,.95]).tolist(),
                empirical_exposure_scale_range=[float(scale.min()),float(scale.max())],
                separable_error_max_relative_remainder=float(np.max(abs(errors / (scale[:, None] * median) - 1))),
                log_weight_to_constant_denominator_quantiles=np.quantile((1+values)**2,[0,.05,.5,.95,1]).tolist(),
                half_variance_log_transform_scale_ppm=np.quantile(.5*(errors/(1+values))**2*1e6,[.05,.5,.95]).tolist()))
    return records


def fit_factory(problem):
    """Compile once, passing covariance matrices as dynamic numeric inputs."""
    @jax.jit
    def objective(parameters, matrices, scales):
        models = problem["native_log_models"](parameters)
        values = []
        for group, state, model, matrix, scale in zip(problem["groups"],problem["states"],models,matrices,scales):
            norm, product = time_gp_statistics(group["log_data"] - model, state)
            values.append(norm / scale**2 - product @ matrix @ product)
        return jnp.sum(jnp.stack(values))
    gradient = jax.jit(jax.value_and_grad(objective))

    def fit(covs, seed=(0.,0.)):
        matrices = tuple(jnp.asarray(cov["correction"]) for cov in covs)
        scales = jnp.array([cov["scale"] for cov in covs])
        run = minimize(lambda p: gradient(p,matrices,scales),seed,jac=True,method="L-BFGS-B",
                       bounds=[(-15,15),(-3,3)],options=dict(maxiter=120,ftol=1e-12,gtol=1e-6))
        return dict(parameters=run.x.tolist(),Q=float(run.fun),success=bool(run.success),message=str(run.message))
    return fit


def band_maps(groups, covs, edges):
    """Exactly the original 8 km/s display bins, then white-weighted band means."""
    centers = (edges[1:]+edges[:-1])/2
    maps, raw, total_weight = [], [], np.zeros(len(centers))
    for group in groups:
        orbital = 244.8*np.sin(2*np.pi*group["phase"])
        velocity = C_KMS*((1+group["stellar_velocity"][None]/C_KMS)/(1+orbital[:,None]/C_KMS)-1)
        index = np.searchsorted(edges,velocity,side="right")-1
        w = group["weight"]
        coefficient = np.array([w*group["science"][:,None]*(index == k) for k in range(len(centers))])
        total_weight += coefficient.sum(axis=(1,2))
        raw.append(coefficient)
        maps.append(coefficient - w[None]/w.sum(axis=0)[None,None]*coefficient.sum(axis=1)[:,None])
    norm = np.where(total_weight>0,total_weight,1.)
    maps = [x/norm[:,None,None] for x in maps]
    raw = [x/norm[:,None,None] for x in raw]
    bin_cov = sum(cov["scale"]**2*np.einsum("bep,cep,ep->bc",m,m,np.divide(1.,g["weight"],out=np.zeros_like(g["weight"]),where=g["weight"]>0)) for m,g,cov in zip(maps,groups,covs))
    transform = []
    for low,high in WINDOWS.values():
        weight = np.where((centers>=low)&(centers<high)&(total_weight>0),1/np.maximum(np.diag(bin_cov),1e-300),0.)
        transform.append(weight/weight.sum())
    transform = np.array(transform)
    return [np.einsum("ab,bep->aep",transform,m) for m in maps],dict(
        names=list(WINDOWS),white_covariance_ppm2=(transform@bin_cov@transform.T*1e12).tolist(),
        white_diagonal_bins_only_covariance_ppm2=(transform@np.diag(np.diag(bin_cov))@transform.T*1e12).tolist(),
        maximum_common_mode_leakage=max(float(np.max(abs(m.sum(axis=1)))) for m in maps))


def band_record(problem, maps, covs, parameters, include_covariance=False):
    data, planet, correction = (np.zeros(3) for _ in range(3))
    white, prior_gp, conditional_gp, fitted_residual_cov = (np.zeros((3,3)) for _ in range(4))
    models = problem["native_log_models"](jnp.array(parameters))
    for group,state,cov,model,mapping in zip(problem["groups"],problem["states"],covs,models,maps):
        model = np.asarray(model)
        mean = coefficient_mean(group,state,cov,group["log_data"]-model)
        gp_map = np.einsum("bep,mp->bme",mapping,state.basis).reshape(3,-1)
        data += np.einsum("bep,ep->b",mapping,group["log_data"])
        planet += np.einsum("bep,ep->b",mapping,model)
        correction += gp_map@mean.ravel()
        if include_covariance:
            w = group["weight"]
            inverse = np.divide(cov["scale"]**2,w,out=np.zeros_like(w),where=w>0)
            white += np.einsum("bep,cep,ep->bc",mapping,mapping,inverse)
            prior_gp += gp_map@cov["prior"]@gp_map.T
            conditional_gp += gp_map@cov["posterior"]@gp_map.T
            basis = np.asarray(state.basis)
            # The GP estimator and common projection are applied to every simulated datum.
            product_map = (np.einsum("mp,ep,ef->mefp",basis,w,np.eye(len(w)))
                           - np.einsum("mp,ep,fp,p->mefp",basis,w,w,1/w.sum(axis=0)))
            product_map = product_map.reshape(len(cov["estimator"]),*w.shape)
            residual_map = mapping - np.einsum("bi,ij,jep->bep",gp_map,cov["estimator"],product_map)
            residual_gp_map = np.einsum("bep,mp->bme",residual_map,basis).reshape(3,-1)
            fitted_residual_cov += (np.einsum("bep,cep,ep->bc",residual_map,residual_map,inverse)
                                   + residual_gp_map@cov["prior"]@residual_gp_map.T)
    result = dict(observed_ppm=(data*1e6).tolist(),planet_ppm=(planet*1e6).tolist(),gp_mean_ppm=(correction*1e6).tolist(),
                  residual_ppm=((data-planet-correction)*1e6).tolist(),blue_minus_core_residual_ppm=float((data-planet-correction)@[1,-1,0]*1e6))
    if include_covariance:
        result.update(white_covariance_ppm2=(white*1e12).tolist(),prior_gp_covariance_ppm2=(prior_gp*1e12).tolist(),
                      conditional_gp_covariance_ppm2=(conditional_gp*1e12).tolist(),fixed_mean_fitted_gp_residual_covariance_ppm2=(fitted_residual_cov*1e12).tolist())
    return result


def block_sensitivity(problem, maps, covs, parameters):
    """Empirical time-block multiplier covariance, with local nuisance refitting.

    One multiplier acts on every wavelength and both orders in each time block.
    This preserves arbitrary within-block spectral/order covariance. Between-
    block independence and residual representativeness remain assumptions; the
    observed mean-model mismatch itself contributes to this noise estimate.
    """
    groups,states=problem["groups"],problem["states"]
    models=problem["native_log_models"](jnp.array(parameters))
    derivatives=jax.jacfwd(problem["native_log_models"])(jnp.array(parameters))
    residual_maps,precisions,innovations=[],[],[]
    information=np.zeros((2,2));band_derivative=np.zeros((3,2))
    for group,state,cov,model,jac,mapping in zip(groups,states,covs,models,derivatives,maps):
        w=group["weight"];basis=np.asarray(state.basis);jac=np.asarray(jac)
        product_map=(np.einsum("mp,ep,ef->mefp",basis,w,np.eye(len(w)))
                     -np.einsum("mp,ep,fp,p->mefp",basis,w,w,1/w.sum(axis=0))).reshape(len(cov["estimator"]),*w.shape)
        gp_map=np.einsum("bep,mp->bme",mapping,basis).reshape(3,-1)
        response=mapping-np.einsum("bi,ij,jep->bep",gp_map,cov["estimator"],product_map)
        product_jac=np.einsum("iep,epk->ik",product_map,jac)
        precision_jac=(np.stack([centered(group,jac[:,:,k]) for k in range(2)],axis=-1)*w[:,:,None]/cov["scale"]**2
                       -np.einsum("iep,ij,jk->epk",product_map,cov["correction"],product_jac))
        information+=np.einsum("epk,epl->kl",jac,precision_jac)
        band_derivative+=np.einsum("bep,epk->bk",response,jac)
        residual_maps.append(response);precisions.append(precision_jac)
        mean=coefficient_mean(group,state,cov,group["log_data"]-model)
        innovations.append(centered(group,group["log_data"]-model-mean.T@basis))
    inverse=la.inv(information)
    adjusted=[r-np.einsum("bk,kl,epl->bep",band_derivative,inverse,p) for r,p in zip(residual_maps,precisions)]
    leakage=sum(np.einsum("bep,epk->bk",r,j) for r,j in zip(adjusted,derivatives))
    model_cov=np.zeros((3,3))
    for group,state,cov,mapping in zip(groups,states,covs,adjusted):
        inverse_weight=np.divide(cov["scale"]**2,group["weight"],out=np.zeros_like(group["weight"]),where=group["weight"]>0)
        gp_map=np.einsum("bep,mp->bme",mapping,state.basis).reshape(3,-1)
        model_cov+=np.einsum("bep,cep,ep->bc",mapping,mapping,inverse_weight)+gp_map@cov["prior"]@gp_map.T
    exposure_influence=[np.einsum("bep,ep->eb",mapping,error) for mapping,error in zip(adjusted,innovations)]
    rows=[];contrast=np.array([1.,-1.,0.])
    for block_size in (1,2,4,8):
        for offset in range(block_size):
            contributions=[]
            for night in (0,1):
                a,b=2*night,2*night+1
                if not np.array_equal(groups[a]["phase"],groups[b]["phase"]):
                    raise ValueError("Matched simultaneous order cadences required")
                values=exposure_influence[a]+exposure_influence[b]
                time=groups[a]["time_min"]
                starts=np.r_[0,np.flatnonzero(np.diff(time)>1.6*np.median(np.diff(time)))+1,len(time)]
                for low,high in zip(starts[:-1],starts[1:]):
                    labels=(np.arange(high-low)+offset)//block_size
                    contributions.extend(values[low:high][labels==k].sum(axis=0) for k in np.unique(labels))
            contributions=np.array(contributions);covariance=contributions.T@contributions
            scalar=contributions@contrast
            rows.append(dict(block_exposures=block_size,offset=offset,blocks=len(scalar),
                covariance_ppm2=(covariance*1e12).tolist(),blue_minus_core_sigma_ppm=float(np.sqrt(contrast@covariance@contrast)*1e6),
                contrast_variance_effective_blocks=float(np.sum(scalar**2)**2/np.sum(scalar**4))))
    return dict(scope=block_sensitivity.__doc__,local_nuisance_information=information.tolist(),
        nuisance_derivative_leakage_max=float(np.max(abs(leakage))),
        model_family_covariance_after_local_RV_amplitude_refit_ppm2=(model_cov*1e12).tolist(),
        model_family_blue_minus_core_sigma_ppm=float(np.sqrt(contrast@model_cov@contrast)*1e6),results=rows,
        limitation="No calibrated p-value: blocks are short and few influential blocks support the blue band; longer systematic modes and the original extraction are not recovered. Local linear nuisance refitting is not nonlinear bootstrap refitting.")


def line_order_check(problem,covs,parameters,simulations):
    """Resolve cross-order comparisons into the moving line and exposure bands."""
    groups,states=problem["groups"],problem["states"]
    models=problem["native_log_models"](jnp.array(parameters))
    records=[];rng=np.random.default_rng(20261130)
    for night in (0,1):
        pair=[]
        for k in (2*night,2*night+1):
            g,s,c=groups[k],states[k],covs[k]
            orbital=244.8*np.sin(2*np.pi*g["phase"])
            velocity=C_KMS*((1+g["stellar_velocity"][None]/C_KMS)/(1+orbital[:,None]/C_KMS)-1)
            mean=coefficient_mean(g,s,c,g["log_data"]-models[k])
            fields=dict(observed=centered(g,g["log_data"]),planet=centered(g,models[k]),
                gp=centered(g,mean.T@s.basis),residual=centered(g,g["log_data"]-models[k]-mean.T@s.basis))
            band_maps=[]
            for low,high in (WINDOWS["blue"],WINDOWS["core"]):
                w=g["weight"]*(velocity>=low)*(velocity<high)
                norm=w.sum(axis=1)
                band_maps.append(np.divide(w,norm[:,None],out=np.zeros_like(w),where=norm[:,None]>0))
            available=g["science"]&(band_maps[0].sum(axis=1)>0)&(band_maps[1].sum(axis=1)>0)
            mapper=band_maps[0]-band_maps[1]
            curves={name:np.einsum("ep,ep->e",mapper,field) for name,field in fields.items()}
            null=[]
            for _ in range(simulations):
                fake=simulate(g,s,c,rng)
                fake_mean=coefficient_mean(g,s,c,fake)
                null.append(np.einsum("ep,ep->e",mapper,centered(g,fake-fake_mean.T@s.basis)))
            pair.append((g,velocity,fields,available,curves,np.array(null)))
        a,b=pair;valid=a[3]&b[3]
        curve_corr={name:correlation(a[4][name],b[4][name],valid) for name in a[4]} if valid.sum()>=3 else {}
        null_corr=[correlation(x,y,valid) for x,y in zip(a[5],b[5])] if valid.sum()>=3 else []
        distance=abs(a[0]["stellar_velocity"][:,None]-b[0]["stellar_velocity"][None])
        right=np.argmin(distance,axis=1);left=np.arange(len(right))
        mutual=np.argmin(distance,axis=0)[right]==left;left,right=left[mutual],right[mutual]
        native=[]
        for name,(low,high) in dict(all_retained=(-np.inf,np.inf),line=(-40.,40.),blue=WINDOWS["blue"],core=WINDOWS["core"],red=WINDOWS["red"]).items():
            keep=(a[0]["science"][:,None]&(a[0]["weight"][:,left]>0)&(b[0]["weight"][:,right]>0)
                  &(a[1][:,left]>=low)&(a[1][:,left]<high)&(b[1][:,right]>=low)&(b[1][:,right]<high))
            native.append(dict(window=name,paired_samples=int(keep.sum()),
                centered_log_correlations={field:correlation(a[2][field][:,left],b[2][field][:,right],keep) for field in ("observed","residual")},
                standardized_residual_correlation=correlation(a[2]["residual"][:,left]*np.sqrt(a[0]["weight"][:,left])/covs[2*night]["scale"],
                    b[2]["residual"][:,right]*np.sqrt(b[0]["weight"][:,right])/covs[2*night+1]["scale"],keep)))
        records.append(dict(night=night,science_exposures_with_both_bands=int(valid.sum()),
            phase_range=a[0]["phase"][valid][[0,-1]].tolist(),per_exposure_blue_minus_core_correlations=curve_corr,
            independent_order_fixed_noise_residual_correlation_quantiles025_975=np.quantile(null_corr,[.025,.975]).tolist() if null_corr else None,
            per_exposure=dict(phase=a[0]["phase"][valid].tolist(),**{f"order{k}_{name}_ppm":(item[4][name][valid]*1e6).tolist() for k,item in enumerate(pair) for name in item[4]}),
            native_pixel_correlations=native))
    return dict(scope="Common-reference and GP use all selected exposures. Per-exposure bands use direct native velocity cuts, matching the phase-support diagnostic, not display-bin center cuts. Moving line cross-order correlations can include the common planet and stellar signal; they are not automatically noise correlations. Only seven late-night exposures have any simultaneous blue/core support, including partial bands; their correlation is poorly constrained.",records=records)


def diagnostics(groups, models, noise, simulations, seed):
    rng = np.random.default_rng(seed)
    records, standardized = [], {}
    for group,model in zip(groups,models):
        k = group["night"]*2+group["order"]
        for selection in ("science", "eclipse"):
            take = group[selection]
            if take.sum()<4:
                continue
            local = subset(group,take)
            x = (local["stellar_velocity"]-250)/140
            state = prepare_time_gp(local["weight"],np.array([np.ones_like(x),x]))
            local["weight"] = np.asarray(state.weight)
            local_noise = {key: value.copy() for key,value in noise.items()}
            if selection == "eclipse":
                hyper,hyper_review = hyper_fit(local,True)
                amp,length,scale = map(np.asarray,hyper)
                local_noise["log_gp"][k] = np.log(amp)
                local_noise["log_length"][k] = np.log(length)
                local_noise["log_white"][k] = np.log(scale)
            else:
                hyper_review = dict(scope="Full-science conditional posterior median, not a noise-only fit")
            cov = covariance(state,local,local_noise,k)
            actual = residual(local,state,cov,local["log_data"]-np.asarray(model)[take])
            actual_stats = correlation_statistics(local,actual)
            draws, residual_draws = [], []
            for _ in range(simulations):
                z = residual(local,state,cov,simulate(local,state,cov,rng))
                draws.append(list(correlation_statistics(local,z).values()))
                residual_draws.append(z.astype(np.float32))
            standardized[(group["night"],group["order"],selection)] = (local,actual,np.array(residual_draws))
            records.append(dict(label=group["label"],selection=selection,exposures=int(take.sum()),pixels=len(x),
                statistics=actual_stats,conditional_simulation_quantiles025_975=dict(zip(actual_stats,np.quantile(draws,[.025,.975],axis=0).T.tolist())),
                hyperparameter_fit=hyper_review))
            print(json.dumps(dict(progress="correlations",label=group["label"],selection=selection,statistics=actual_stats)),flush=True)
    cross = []
    for night in (0,1):
        for selection in ("science","eclipse"):
            if (night,0,selection) not in standardized:continue
            ga,za,sa = standardized[(night,0,selection)]
            gb,zb,sb = standardized[(night,1,selection)]
            # One-to-one nearest matching in stellar wavelength; no resampling.
            distance = abs(ga["stellar_velocity"][:,None]-gb["stellar_velocity"][None])
            right = np.argmin(distance,axis=1)
            left = np.arange(len(right))
            mutual = np.argmin(distance,axis=0)[right] == left
            left,right = left[mutual],right[mutual]
            valid = (ga["weight"][:,left]>0)&(gb["weight"][:,right]>0)
            value = correlation(za[:,left],zb[:,right],valid)
            draws = [correlation(a[:,left],b[:,right],valid) for a,b in zip(sa,sb)]
            cross.append(dict(night=night,selection=selection,matched_pixels=len(left),
                wavelength_matching_max_delta_kms=float(distance[left,right].max()),
                residual_correlation=value,conditional_independent_order_quantiles025_975=np.quantile(draws,[.025,.975]).tolist()))
    return dict(groups=records,cross_order=cross,simulation_count=simulations,
                scope="Fixed-hyperparameter conditional null simulations refit latent GP means and common spectra. Science residuals also contain possible physical mean-model errors. Marginal envelopes are not calibrated or simultaneous hypothesis tests.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--simulations",type=int,default=128)
    parser.add_argument("--posterior-draws",type=int,default=64)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08","20")]
    model_path=Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz")
    sample_path=Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    surface_path=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    bins_path=Path("/tmp/escape-gp-combined-20261125/combined_arrays.npz")
    repo=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),*[repo/"examples"/name for name in ("gp_halpha.py","gp_controls.py","halpha_geometry_fit.py","phase_halpha.py","cloudy_halpha.py")],
             repo/"src/escape/time_gp.py",repo/"src/escape/time_contrast.py"]
    hashes={str(p):digest(p) for p in paths+[model_path,sample_path,surface_path,bins_path]+sources}
    started=time.monotonic()
    groups=load_gp_data(paths)
    model=dict(np.load(model_path));surface=dict(np.load(surface_path));samples=dict(np.load(sample_path))
    noise={key:np.median(samples[key],axis=(0,1)) for key in ("log_gp","log_length","log_white")}
    problem=make_gp_problem(groups,model,surface)
    covs=[covariance(state,g,noise,k) for k,(state,g) in enumerate(zip(problem["states"],groups))]
    fit=fit_factory(problem)
    fits=[fit(covs,[rv,0.]) for rv in (0.,3.)]
    primary=min(fits,key=lambda row:row["Q"])
    edges=np.load(bins_path)["edges"]
    maps,map_review=band_maps(groups,covs,edges)
    primary["bands"]=band_record(problem,maps,covs,primary["parameters"],True)
    primary["projection"]=map_review
    primary["rank"]=sum(int(state.rank) for state in problem["states"])
    blocks=block_sensitivity(problem,maps,covs,primary["parameters"])
    line_order=line_order_check(problem,covs,primary["parameters"],args.simulations)
    print(json.dumps(dict(progress="primary",result=primary)),flush=True)
    flat={key:value.reshape((-1,)+value.shape[2:]) for key,value in samples.items()}
    rng=np.random.default_rng(20261127)
    indices=np.sort(rng.choice(len(flat["log_white"]),min(args.posterior_draws,len(flat["log_white"])),replace=False))
    posterior=[]
    for i in indices:
        draw_noise={key:flat[key][i] for key in noise}
        draw_cov=[covariance(s,g,draw_noise,k) for k,(s,g) in enumerate(zip(problem["states"],groups))]
        row=fit(draw_cov,primary["parameters"])
        row.update(draw_index=int(i),bands=band_record(problem,maps,draw_cov,row["parameters"]))
        posterior.append(row)
    altered=[dict(g,weight=g["weight"]*np.exp(-2*g["log_data"])) for g in groups]
    alternative=make_gp_problem(altered,model,surface)
    alt_cov=[covariance(s,g,noise,k) for k,(s,g) in enumerate(zip(alternative["states"],altered))]
    alt_fit=fit_factory(alternative)(alt_cov,primary["parameters"])
    alt_maps,_=band_maps(altered,alt_cov,edges)
    alt_fit["bands"]=band_record(alternative,alt_maps,alt_cov,alt_fit["parameters"])
    alt_fit["scope"]="Sensitivity only: replace ratio-dependent log-error weights by 1/archive_sigma^2; no statistical claim from Q differences across changed measures."
    extra={key:value.copy() for key,value in noise.items()};extra["log_gp"][:]=-100.
    white_cov=[covariance(s,g,extra,k) for k,(s,g) in enumerate(zip(problem["states"],groups))]
    white_fit=fit(white_cov,primary["parameters"])
    white_fit["bands"]=band_record(problem,maps,white_cov,white_fit["parameters"])
    models=problem["native_log_models"](jnp.array(primary["parameters"]))
    coefficient_errors=[]
    for k,(group,state,cov,model_value) in enumerate(zip(groups,problem["states"],covs,models)):
        original=time_gp_posterior(group["log_data"]-model_value,state,group["time_min"],
            np.exp(noise["log_gp"][k]),np.exp(noise["log_length"][k]),np.exp(noise["log_white"][k]))
        coefficient_errors.append(dict(mean_max_abs=float(np.max(abs(np.asarray(original.mean)-coefficient_mean(group,state,cov,group["log_data"]-model_value)))),
            covariance_max_abs=float(np.max(abs(np.asarray(original.covariance)-cov["posterior"])))))
    stats=diagnostics(groups,models,noise,args.simulations,20261128)
    # Retain all native pixels only for a separate eclipse covariance control.
    full=load_gp_data(paths,pixel_stride=1)
    full_problem=make_gp_problem(full,model,surface)
    full_models=full_problem["native_log_models"](jnp.array(primary["parameters"]))
    full_stats=diagnostics([g for g in full if g["night"]==1],full_models[2:],noise,args.simulations,20261129)
    receipt=dict(schema="escape.halpha_noise_audit.v1",scope=__doc__,source_and_input_sha256=hashes,
        runtime=dict(jax=jax.__version__,x64=bool(jax.config.jax_enable_x64),devices=[str(x) for x in jax.devices()]),
        archive_errors=archive_errors(paths),primary=primary,posterior_covariance_sensitivity=posterior,
        empirical_time_block_covariance=blocks,
        moving_line_cross_order=line_order,
        numerical_checks=dict(coefficient_correspondence=coefficient_errors,
            previous_baseline_Q_absolute_difference=abs(primary["Q"]-38326.64894003179),
            previous_baseline_blue_core_residual_max_absolute_difference_ppm=float(np.max(abs(np.array(primary["bands"]["residual_ppm"][:2])-[409.0567079,-235.2446886])))),
        posterior_covariance_scope="64 reproducibly selected archived joint noise draws; RV and amplitude are reprofiled for each. Band maps stay fixed at the primary median-covariance display definition. Ranges are sensitivity summaries, not new posterior credible intervals or atmospheric evidence.",
        constant_denominator_log_weights=alt_fit,white_only_same_white_scales=white_fit,
        primary_correlations=stats,full_native_night1_correlations=full_stats,
        limitations=["No original extraction covariance, reference weights or empirical-error construction is recovered.",
                     "No narrow spectral systematics, cross-order or non-GP residual correlation is included in quoted model covariance.",
                     "Conditional band covariance holds planet parameters fixed; the mean was fitted to these data. It is not a calibrated residual significance.",
                     "Posterior hyperparameters came from one fixed atmosphere; their sensitivity does not integrate over alternative noise families."],
        unchanged_source_and_input_hashes=all(digest(p)==h for p,h in hashes.items()),elapsed_seconds=time.monotonic()-started)
    (args.output/"review.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(progress="complete",output=str(args.output),elapsed_seconds=receipt["elapsed_seconds"])),flush=True)


if __name__=="__main__":
    main()
