"""Infer a fixed H-alpha atmosphere's contrast/RV with eclipse-constrained GPs.

Native log ratios retain independent night/order groups. A common spectrum is
integrated out, and grey/slope time GPs are analytically marginalized with the
same projection. Ingress is excluded; fully eclipsed planet flux is zero.
The empirical archive errors have inferred white scales, not assumed photon
noise. This is conditional inference for a fixed atmosphere, not NLTE NUTS.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import minimize
from scipy.special import ndtr

from escape.observation import C_KMS
from escape.time_gp import (
    prepare_time_gp,
    time_gp_loglike,
    time_gp_posterior,
    time_gp_predict,
)
from phase_halpha import REST_ANGSTROM
from cloudy_halpha import stellar_reference

PERIOD_MINUTES = 1.48 * 24 * 60


def load_gp_data(
    paths,
    science_max=0.435,
    eclipse_min=0.47,
    science_only=False,
    pixel_stride=2,
    pixel_offset=0,
):
    """Use native positive ratios, retaining each night/order separately."""
    if pixel_stride < 1 or not 0 <= pixel_offset < pixel_stride:
        raise ValueError("Require positive pixel stride and an offset smaller than it.")
    groups = []
    for night, path in enumerate(paths):
        with h5py.File(path) as source:
            phase, wave, residual, error = (
                source[name][:] for name in ("phase", "wave", "flux", "err")
            )
        science = (phase >= 0.30) & (phase < science_max)
        eclipse = (phase >= eclipse_min) & (phase <= 0.5)
        keep = science if science_only else science | eclipse
        for order in range(len(wave)):
            coordinate = C_KMS * (wave[order] / REST_ANGSTROM - 1)
            original_pixel = np.arange(len(coordinate))
            pixels = (
                (coordinate >= 110)
                & (coordinate <= 390)
                & (original_pixel % pixel_stride == pixel_offset)
            )
            ratio = 1 + residual[keep, order][:, pixels]
            sigma = error[keep, order][:, pixels]
            valid = np.isfinite(ratio) & (ratio > 0) & np.isfinite(sigma) & (sigma > 0)
            support = valid.sum(axis=0) >= 2
            ratio, sigma, valid = (x[:, support] for x in (ratio, sigma, valid))
            selected_phase = phase[keep]
            groups.append(
                dict(
                    night=night,
                    order=order,
                    label=f"night{night}_order{order}",
                    phase=selected_phase,
                    science=science[keep],
                    eclipse=eclipse[keep],
                    time_min=(selected_phase - selected_phase.min()) * PERIOD_MINUTES,
                    stellar_velocity=coordinate[pixels][support],
                    pixel_index=original_pixel[pixels][support],
                    log_data=np.log(np.where(valid, ratio, 1.0)),
                    weight=np.where(
                        valid, (ratio / np.where(valid, sigma, 1.0)) ** 2, 0.0
                    ),
                    input_sha256=sha256(Path(path).read_bytes()).hexdigest(),
                )
            )
    return groups


def make_native_flux_fast(model_velocity_kms, surface, group):
    """Exact existing two-stage interpolation with fixed queries prepared once."""
    velocity_np = np.asarray(model_velocity_kms)
    velocity = jnp.asarray(velocity_np)
    coordinate = group["stellar_velocity"]
    native_star = jnp.asarray(stellar_reference(coordinate, surface, 110.0))
    normalization_grid = np.linspace(-600.0, 600.0, 2401)
    normalization_grid = normalization_grid[abs(normalization_grid) > 400]
    normalization_star = jnp.asarray(
        stellar_reference(normalization_grid, surface, 110.0)
    )
    orbital = 244.8 * np.sin(2 * np.pi * group["phase"])

    def sampler(coordinate):
        query = C_KMS * (
            (1 + coordinate[None, :] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1
        )
        index = np.clip(
            np.searchsorted(velocity_np, query, side="right") - 1,
            0,
            len(velocity_np) - 2,
        )
        fraction = np.clip(
            (query - velocity_np[index])
            / (velocity_np[index + 1] - velocity_np[index]),
            0.0,
            1.0,
        )
        index, fraction = jnp.asarray(index), jnp.asarray(fraction)

        def sample(rows):
            left = jnp.take_along_axis(rows, index, axis=1)
            return left + fraction * (
                jnp.take_along_axis(rows, index + 1, axis=1) - left
            )

        return sample

    native, continuum = sampler(coordinate), sampler(normalization_grid)

    @jax.jit
    def flux(spectra, offset):
        query = velocity - offset
        index = jnp.clip(
            jnp.searchsorted(velocity, query, side="right") - 1, 0, len(velocity_np) - 2
        )
        fraction = (query - velocity[index]) / (velocity[index + 1] - velocity[index])
        shifted = spectra[:, index] + fraction[None, :] * (
            spectra[:, index + 1] - spectra[:, index]
        )
        shifted = jnp.where(query[None, :] < velocity[0], spectra[:, 0, None], shifted)
        shifted = jnp.where(
            query[None, :] > velocity[-1], spectra[:, -1, None], shifted
        )
        normalization = (normalization_star[None, :] + continuum(shifted)).mean(axis=1)
        return (native_star[None, :] + native(shifted)) / normalization[:, None]

    return flux


def make_gp_problem(groups, model, surface):
    """Prepare fixed kernels and native log-flux callbacks for [RV, log A]."""
    phase_grid = jnp.asarray(model["phase"])
    spectra = jnp.asarray(model["fine_intrinsic_phases"])
    fluxes, intrinsic, states = [], [], []
    for group in groups:
        science_phase = group["phase"][group["science"]]
        if len(science_phase) and (
            science_phase.min() < phase_grid.min()
            or science_phase.max() > phase_grid.max()
        ):
            raise ValueError("Saved spectra must bracket every science exposure.")
        sampled = jax.vmap(
            lambda column: jnp.interp(group["phase"], phase_grid, column),
            in_axes=1,
            out_axes=1,
        )(spectra)
        # Eclipsed values never use the interpolated planet: zero it exactly.
        intrinsic.append(
            jnp.where(jnp.asarray(group["science"])[:, None], sampled, 0.0)
        )
        fluxes.append(
            make_native_flux_fast(model["model_velocity_kms"], surface, group)
        )
        coordinate = (group["stellar_velocity"] - 250) / 140
        states.append(
            prepare_time_gp(
                group["weight"], np.array([np.ones_like(coordinate), coordinate])
            )
        )

    @jax.jit
    def native_log_models(parameters):
        rv, log_amplitude = parameters
        return tuple(
            jnp.log(flux(jnp.exp(log_amplitude) * spectrum, rv))
            for flux, spectrum in zip(fluxes, intrinsic)
        )

    return dict(groups=groups, states=states, native_log_models=native_log_models)


def log_likelihood(problem, planet, amplitudes, length, white_scale):
    """Exact projected GP likelihood, up to parameter-independent constants."""
    models = problem["native_log_models"](planet)
    return jnp.sum(
        jnp.stack(
            [
                time_gp_loglike(
                    group["log_data"] - model, state, group["time_min"], amp, ell, scale
                )
                for group, state, model, amp, ell, scale in zip(
                    problem["groups"],
                    problem["states"],
                    models,
                    amplitudes,
                    length,
                    white_scale,
                )
            ]
        )
    )


def make_model(problem, use_gp=True, amplitude_prior=300e-6, length_prior=20.0):
    """Return a NumPyro model with GP paths integrated out, not sampled."""
    groups = len(problem["groups"])

    def model():
        rv = numpyro.sample("rv_kms", dist.Uniform(-15.0, 15.0))
        log_amplitude = numpyro.sample("log_contrast", dist.Normal(0.0, 0.7))
        log_white = numpyro.sample(
            "log_white", dist.Normal(0.0, 0.35).expand((groups,)).to_event(1)
        )
        if use_gp:
            log_gp = numpyro.sample(
                "log_gp",
                dist.Normal(np.log(amplitude_prior), 1.0)
                .expand((groups, 2))
                .to_event(2),
            )
            log_length = numpyro.sample(
                "log_length",
                dist.Normal(np.log(length_prior), 1.0).expand((groups,)).to_event(1),
            )
            amplitudes, length = jnp.exp(log_gp), jnp.exp(log_length)
        else:
            amplitudes, length = jnp.zeros((groups, 2)), jnp.full(groups, length_prior)
        numpyro.factor(
            "native_time",
            log_likelihood(
                problem,
                jnp.array([rv, log_amplitude]),
                amplitudes,
                length,
                jnp.exp(log_white),
            ),
        )

    return model


def map_initialization(problem, use_gp, amplitude_prior, length_prior):
    """Find a reproducible finite starting point using JAX derivatives."""
    count = len(problem["groups"])
    initial = np.r_[3.0, 0.0, np.full(count, np.log(0.9))]
    if use_gp:
        initial = np.r_[
            initial,
            np.full(count * 2, np.log(amplitude_prior)),
            np.full(count, np.log(length_prior)),
        ]

    def unpack(point):
        planet, white = point[:2], jnp.exp(point[2 : 2 + count])
        if use_gp:
            amp = jnp.exp(point[2 + count : 2 + 3 * count].reshape(count, 2))
            length = jnp.exp(point[2 + 3 * count :])
        else:
            amp, length = jnp.zeros((count, 2)), jnp.full(count, length_prior)
        return planet, amp, length, white

    @jax.jit
    def objective(point):
        prior = 0.5 * (point[1] / 0.7) ** 2 + 0.5 * jnp.sum(
            (point[2 : 2 + count] / 0.35) ** 2
        )
        if use_gp:
            prior += 0.5 * jnp.sum(
                (point[2 + count : 2 + 3 * count] - np.log(amplitude_prior)) ** 2
            )
            prior += 0.5 * jnp.sum((point[2 + 3 * count :] - np.log(length_prior)) ** 2)
        return -log_likelihood(problem, *unpack(point)) + prior

    value_gradient = jax.jit(jax.value_and_grad(objective))
    begin = time.monotonic()
    first = value_gradient(jnp.asarray(initial))
    first[0].block_until_ready()
    compile_seconds = time.monotonic() - begin
    begin = time.monotonic()
    for _ in range(10):
        first = value_gradient(jnp.asarray(initial))
        first[0].block_until_ready()
    gradient_seconds = (time.monotonic() - begin) / 10
    bounds = [(-14.99, 14.99), (-4.0, 3.0)] + [(-2.0, 1.0)] * count
    if use_gp:
        bounds += [(np.log(1e-7), np.log(0.02))] * (2 * count) + [
            (np.log(0.1), np.log(2000))
        ] * count
    fit = minimize(
        value_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=250, ftol=1e-11, gtol=1e-5),
    )
    init = dict(rv_kms=fit.x[0], log_contrast=fit.x[1], log_white=fit.x[2 : 2 + count])
    if use_gp:
        init.update(
            log_gp=fit.x[2 + count : 2 + 3 * count].reshape(count, 2),
            log_length=fit.x[2 + 3 * count :],
        )
    record = dict(
        success=bool(fit.success),
        message=str(fit.message),
        objective=float(fit.fun),
        iterations=int(fit.nit),
        initial={k: np.asarray(v).tolist() for k, v in init.items()},
        compile_seconds=compile_seconds,
        gradient_seconds=gradient_seconds,
        prior_scope="MAP in the explicitly sampled log parameters; finite bounds initialize NUTS only.",
    )
    return init, record


def diagnostic_summary(samples, extra):
    """Report chains before judging posterior convergence or precision."""
    diagnostic = summary(samples, prob=0.9, group_by_chain=True)
    packed = {
        name: {key: np.asarray(value).tolist() for key, value in row.items()}
        for name, row in diagnostic.items()
    }
    rhat = max(float(np.max(row["r_hat"])) for row in diagnostic.values())
    ess = min(float(np.min(row["n_eff"])) for row in diagnostic.values())
    divergence = int(np.sum(extra["diverging"]))
    import arviz as az

    inference = az.from_dict(posterior={k: np.asarray(v) for k, v in samples.items()})
    rank_rhat = max(
        float(np.max(v.values))
        for v in az.rhat(inference, method="rank").data_vars.values()
    )
    bulk_ess = min(
        float(np.min(v.values))
        for v in az.ess(inference, method="bulk").data_vars.values()
    )
    tail_ess = min(
        float(np.min(v.values))
        for v in az.ess(inference, method="tail").data_vars.values()
    )
    energy = np.asarray(extra["energy"])
    bfmi = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(energy, axis=1)
    return dict(
        parameters=packed,
        max_rhat=rhat,
        min_effective_samples=ess,
        max_rank_rhat=rank_rhat,
        min_bulk_ess=bulk_ess,
        min_tail_ess=tail_ess,
        energy_bfmi=bfmi.tolist(),
        divergences=divergence,
        maximum_steps=int(np.max(extra["num_steps"])),
        mean_accept_probability=float(np.mean(extra["accept_prob"])),
        scope="NumPyro split-Rhat plus ArviZ rank-normalized Rhat and bulk/tail ESS; not an observational validation.",
    )


def run(args):
    jax.config.update("jax_enable_x64", True)
    source_at_start = sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = load_gp_data(
        args.data,
        science_only=args.science_only,
        pixel_stride=args.pixel_stride,
        pixel_offset=args.pixel_offset,
    )
    problem = make_gp_problem(
        groups, dict(np.load(args.model)), dict(np.load(args.stellar_surface))
    )
    use_gp = args.mode == "gp"
    if args.predict_only:
        samples = dict(np.load(args.output / "samples.npz"))
        save_predictions(problem, samples, args.output, use_gp)
        if use_gp:
            save_gap_predictions(problem, samples, args.output)
        if args.white_results:
            plot_results(
                args.output,
                args.white_results,
                args.output / "figures",
                args.alternate_results,
                args.science_only_results,
            )
        return
    plan = dict(
        driver_sha256_at_start=source_at_start,
        mode=args.mode,
        science_only=args.science_only,
        pixel_stride=args.pixel_stride,
        pixel_offset=args.pixel_offset,
        seed=args.seed,
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        priors=dict(
            rv_kms="Uniform(-15,15)",
            log_contrast="Normal(0,0.7)",
            log_white="Normal(0,0.35)",
            log_gp=f"Normal(log({args.amplitude_prior}),1)",
            log_length=f"Normal(log({args.length_prior}),1)",
        ),
        groups=[
            dict(
                label=g["label"],
                exposures=len(g["phase"]),
                science=int(g["science"].sum()),
                eclipse=int(g["eclipse"].sum()),
                pixels=len(g["stellar_velocity"]),
            )
            for g in groups
        ],
        limits=[
            "No atmospheric solve or physical source-map inference.",
            "Archive errors are empirical; white scales are fitted jointly with correlated noise.",
            "Independent night/order GP realizations; unresolved cross-order and detailed wavelength covariance.",
            "Eclipse membership assumes the adopted circular geometry and no emission outside 3Rp.",
        ],
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.initialization is None:
        initial, start = map_initialization(
            problem, use_gp, args.amplitude_prior, args.length_prior
        )
    else:
        start = json.loads(args.initialization.read_text())
        initial = {k: jnp.asarray(v) for k, v in start["initial"].items()}
    (args.output / "initialization.json").write_text(json.dumps(start, indent=2) + "\n")
    print(json.dumps(start), flush=True)
    if args.initialize_only:
        return
    model = make_model(problem, use_gp, args.amplitude_prior, args.length_prior)
    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=initial),
        dense_mass=True,
        target_accept_prob=args.target_accept,
        max_tree_depth=9,
    )
    sampler = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=True,
    )
    begin = time.monotonic()
    sampler.run(
        jax.random.key(args.seed),
        extra_fields=("diverging", "num_steps", "accept_prob", "energy"),
    )
    samples, extra = (
        sampler.get_samples(group_by_chain=True),
        sampler.get_extra_fields(group_by_chain=True),
    )
    diagnostics = diagnostic_summary(samples, extra)
    report = dict(
        plan=plan,
        initialization=start,
        diagnostics=diagnostics,
        sampling_seconds=time.monotonic() - begin,
        sources={
            str(p): sha256(p.read_bytes()).hexdigest()
            for p in [*args.data, args.model, args.stellar_surface, Path(__file__)]
        },
    )
    np.savez_compressed(
        args.output / "samples.npz", **{k: np.asarray(v) for k, v in samples.items()}
    )
    np.savez_compressed(
        args.output / "sampler.npz", **{k: np.asarray(v) for k, v in extra.items()}
    )
    (args.output / "review.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            dict(
                sampling_seconds=report["sampling_seconds"],
                max_rhat=diagnostics["max_rhat"],
                min_ess=diagnostics["min_effective_samples"],
                divergences=diagnostics["divergences"],
            )
        ),
        flush=True,
    )
    if not args.no_predictions:
        save_predictions(problem, samples, args.output, use_gp)
        if use_gp:
            save_gap_predictions(problem, samples, args.output)


def save_predictions(problem, samples, output, use_gp, maximum_draws=128):
    """Save projected predictive means and conditional GP covariance locally."""
    flat = {k: np.asarray(v).reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
    indices = np.linspace(
        0, len(flat["rv_kms"]) - 1, min(maximum_draws, len(flat["rv_kms"]))
    ).astype(int)
    arrays = {}
    for i, g in enumerate(problem["groups"]):
        for k in (
            "phase",
            "time_min",
            "science",
            "eclipse",
            "stellar_velocity",
            "pixel_index",
            "log_data",
            "weight",
        ):
            arrays[f"group{i}_{k}"] = g[k]
    arrays["white_scale"] = np.exp(flat["log_white"][indices])
    posterior_at_data = jax.jit(time_gp_posterior)
    # Posterior GP functions are conditional on planet and noise parameters.
    # Store coefficient covariance so later plots need not assume a known path.
    for sample in indices:
        planet = jnp.array([flat["rv_kms"][sample], flat["log_contrast"][sample]])
        models = problem["native_log_models"](planet)
        for i, (g, state, model) in enumerate(
            zip(problem["groups"], problem["states"], models)
        ):
            prefix = f"group{i}"
            if prefix + "_planet" not in arrays:
                arrays[prefix + "_planet"] = []
                arrays[prefix + "_gp_mean"] = []
                arrays[prefix + "_gp_covariance"] = []
            amp = jnp.exp(flat["log_gp"][sample, i]) if use_gp else jnp.zeros(2)
            ell = jnp.exp(flat["log_length"][sample, i]) if use_gp else 20.0
            posterior = posterior_at_data(
                g["log_data"] - model,
                state,
                g["time_min"],
                amp,
                ell,
                jnp.exp(flat["log_white"][sample, i]),
            )
            arrays[prefix + "_planet"].append(np.asarray(model))
            arrays[prefix + "_gp_mean"].append(np.asarray(posterior.mean))
            arrays[prefix + "_gp_covariance"].append(np.asarray(posterior.covariance))
    np.savez_compressed(output / "predictions.npz", **arrays)


def normal_mixture_quantiles(mean, variance):
    """Pointwise 5/50/95% quantiles of equally weighted conditional Gaussians."""
    sigma = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1e-15)
    lower, upper = np.min(mean - 10 * sigma, axis=0), np.max(mean + 10 * sigma, axis=0)
    quantiles = []
    for probability in (0.05, 0.5, 0.95):
        left, right = lower.copy(), upper.copy()
        for _ in range(48):
            middle = (left + right) / 2
            cdf = ndtr((middle[None] - mean) / sigma).mean(axis=0)
            left, right = (
                np.where(cdf < probability, middle, left),
                np.where(cdf >= probability, middle, right),
            )
        quantiles.append((left + right) / 2)
    return np.asarray(quantiles)


def spectral_display(predictions, night, low_phase, high_phase):
    """Bin the fitted native contrasts, propagating the marginalized GP covariance.

    Binning is only for display. The inference uses every selected native pixel.
    A linear operator applies the same weighted common-spectrum removal to data,
    planet, and GP. Bands include conditional GP uncertainty and hyperparameters.
    """
    edges = np.arange(-168.0, 177.0, 8.0)
    centers = (edges[:-1] + edges[1:]) / 2
    nbin = len(centers)
    count = len(predictions["group0_planet"])
    total_weight = np.zeros(nbin)
    observed = np.zeros(nbin)
    noise = np.zeros((count, nbin))
    planet = np.zeros((count, nbin))
    total_mean = np.zeros((count, nbin))
    gp_variance = np.zeros((count, nbin))
    for i in (2 * night, 2 * night + 1):
        prefix = f"group{i}_"
        phase = predictions[prefix + "phase"]
        weight = predictions[prefix + "weight"]
        wave = predictions[prefix + "stellar_velocity"]
        orbital = 244.8 * np.sin(2 * np.pi * phase)
        rest = C_KMS * (
            (1 + wave[None, :] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1
        )
        phase_keep = (phase >= low_phase) & (phase < high_phase)
        bins = np.searchsorted(edges, rest, side="right") - 1
        coefficient = np.array(
            [weight * (bins == b) * phase_keep[:, None] for b in range(nbin)]
        )
        total_weight += coefficient.sum(axis=(1, 2))
        common = weight / np.maximum(weight.sum(axis=0), 1.0)[None, :]
        projected = (
            coefficient - common[None, :, :] * coefficient.sum(axis=1)[:, None, :]
        )
        observed += np.einsum("bep,ep->b", projected, predictions[prefix + "log_data"])
        local_planet = np.einsum(
            "bep,dep->db", projected, predictions[prefix + "planet"]
        )
        planet += local_planet
        basis = np.array([np.ones_like(wave), (wave - 250) / 140])
        gp_map = np.einsum("bep,mp->bme", projected, basis).reshape(nbin, -1)
        gp_mean = predictions[prefix + "gp_mean"].reshape(count, -1)
        total_mean += local_planet + gp_mean @ gp_map.T
        gp_variance += np.einsum(
            "bi,dij,bj->db", gp_map, predictions[prefix + "gp_covariance"], gp_map
        )
        variance = np.sum(
            projected**2 / np.where(weight > 0, weight, np.inf), axis=(1, 2)
        )
        noise += predictions["white_scale"][:, i, None] ** 2 * variance[None, :]
    valid = total_weight > 0
    normalizer = np.where(valid, total_weight, 1.0)
    observed /= normalizer
    planet /= normalizer
    total_mean /= normalizer
    gp_variance /= normalizer**2
    noise /= normalizer**2
    return dict(
        velocity=centers[valid],
        observed=observed[valid],
        error=np.sqrt(noise.mean(axis=0))[valid],
        planet=np.quantile(planet, [0.05, 0.5, 0.95], axis=0)[:, valid],
        total=normal_mixture_quantiles(total_mean, gp_variance)[:, valid],
    )


def save_gap_predictions(problem, samples, output, maximum_draws=128):
    """Predict joint-fit GP differences from the observed eclipse average.

    The same linear map acts on mean and full joint grid/reference covariance.
    These paths use science AND eclipse; they are not independent predictions
    made from eclipse alone. No absolute common GP level is identifiable.
    """
    flat = {k: np.asarray(v).reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
    indices = np.linspace(
        0, len(flat["rv_kms"]) - 1, min(maximum_draws, len(flat["rv_kms"]))
    ).astype(int)
    predict = jax.jit(time_gp_predict)
    arrays = {}
    for i, (group, state) in enumerate(zip(problem["groups"], problem["states"])):
        if not np.any(group["eclipse"]):
            continue
        grid = np.linspace(group["time_min"].min(), group["time_min"].max(), 161)
        reference = group["time_min"][group["eclipse"]]
        query = np.r_[grid, reference]
        width = len(query)
        transform = np.zeros((2 * len(grid), 2 * width))
        for mode in range(2):
            rows = mode * len(grid) + np.arange(len(grid))
            transform[rows, mode * width + np.arange(len(grid))] = 1
            transform[
                rows[:, None],
                mode * width + len(grid) + np.arange(len(reference))[None, :],
            ] = -1 / len(reference)
        means, variances = [], []
        for draw in indices:
            model = problem["native_log_models"](
                jnp.array([flat["rv_kms"][draw], flat["log_contrast"][draw]])
            )[i]
            posterior = predict(
                group["log_data"] - model,
                state,
                group["time_min"],
                jnp.exp(flat["log_gp"][draw, i]),
                jnp.exp(flat["log_length"][draw, i]),
                jnp.exp(flat["log_white"][draw, i]),
                query,
            )
            means.append(
                (transform @ np.asarray(posterior.mean).ravel()).reshape(2, -1)
            )
            covariance = np.asarray(posterior.covariance)
            variances.append(
                np.einsum("ij,jk,ik->i", transform, covariance, transform).reshape(
                    2, -1
                )
            )
        means, variances = np.asarray(means), np.asarray(variances)
        arrays[f"group{i}_time"] = grid
        arrays[f"group{i}_band"] = normal_mixture_quantiles(means, variances)
        arrays[f"group{i}_science_time"] = group["time_min"][group["science"]]
        arrays[f"group{i}_eclipse_time"] = reference
    np.savez_compressed(output / "gap_predictions.npz", **arrays)


def plot_results(primary, white, output, alternate=None, science_only=None):
    """Reduce local posteriors to a small, independently plottable bundle."""
    output.mkdir(parents=True, exist_ok=True)
    predictions = dict(np.load(primary / "predictions.npz"))
    white_predictions = dict(np.load(white / "predictions.npz"))
    arrays = dict(np.load(primary / "gap_predictions.npz"))
    for k, (night, low, high) in enumerate(
        ((0, 0.30, 0.37), (0, 0.37, 0.435), (1, 0.30, 0.435))
    ):
        for name, value in spectral_display(predictions, night, low, high).items():
            arrays[f"panel{k}_{name}"] = value
        arrays[f"panel{k}_white_planet"] = spectral_display(
            white_predictions, night, low, high
        )["planet"]
    comparison = {}
    for name, folder in (
        ("white", white),
        ("primary", primary),
        ("alternate", alternate),
        ("science_only", science_only),
    ):
        if folder is None:
            continue
        chain = dict(np.load(folder / "samples.npz"))
        rv, contrast = chain["rv_kms"].ravel(), np.exp(chain["log_contrast"].ravel())
        index = np.linspace(0, len(rv) - 1, min(400, len(rv))).astype(int)
        arrays[name + "_points"] = np.array([rv[index], contrast[index]])
        arrays[name + "_rv_contrast"] = np.array(
            [
                np.quantile(rv, [0.05, 0.5, 0.95]),
                np.quantile(contrast, [0.05, 0.5, 0.95]),
            ]
        )
        physical = {
            "rv_kms": chain["rv_kms"],
            "contrast": np.exp(chain["log_contrast"]),
            "white_scale": np.exp(chain["log_white"]),
        }
        if "log_gp" in chain:
            physical.update(
                gp_amplitude_ppm=1e6 * np.exp(chain["log_gp"]),
                length_minutes=np.exp(chain["log_length"]),
            )
        comparison[name] = {
            key: np.quantile(
                values.reshape((-1,) + values.shape[2:]), [0.05, 0.5, 0.95], axis=0
            ).tolist()
            for key, values in physical.items()
        }
    np.savez_compressed(output / "plot_arrays.npz", **arrays)
    (output / "posterior_summary.json").write_text(
        json.dumps(
            {
                "interval": "equal-tailed 5%,50%,95%; fixed atmosphere and declared noise model",
                "parameters": comparison,
                "limits": [
                    "Positive amplitude priors do not test GP absence.",
                    "Alternate native-pixel sets are not independent posterior chains.",
                    "Bands are pointwise and conditional on the same fitted data.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    plot_saved_arrays(arrays, output)


def plot_saved_arrays(arrays, output):
    """Render the public figure bundle without stellar or observational inputs."""
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    panels = [
        (0, 0.30, 0.37, "August 8: earlier phases"),
        (0, 0.37, 0.435, "August 8: later phases"),
        (1, 0.30, 0.435, "August 20: before ingress"),
    ]
    for k, (night, low, high, title) in enumerate(panels):
        ax = axes.flat[k]
        gp = {
            name: arrays[f"panel{k}_{name}"]
            for name in ("velocity", "observed", "error", "planet", "total")
        }
        base = {"planet": arrays[f"panel{k}_white_planet"]}
        visible = (gp["velocity"] >= -80) & (gp["velocity"] <= 140)
        gp = {name: value[..., visible] for name, value in gp.items()}
        base["planet"] = base["planet"][:, visible]
        velocity = gp["velocity"]
        ax.errorbar(
            velocity,
            1e6 * gp["observed"],
            1e6 * gp["error"],
            fmt=".",
            color="0.55",
            alpha=0.7,
            label="Observed contrast (white 1σ)",
        )
        ax.fill_between(
            velocity, 1e6 * gp["total"][0], 1e6 * gp["total"][2], color="C0", alpha=0.22
        )
        ax.plot(
            velocity,
            1e6 * gp["total"][1],
            color="C0",
            label="Planet + GP, 90% pointwise band",
        )
        ax.plot(
            velocity,
            1e6 * gp["planet"][1],
            color="C1",
            label="Planet contribution with GP",
        )
        ax.plot(
            velocity,
            1e6 * base["planet"][1],
            "--",
            color="C3",
            label="White-only planet",
        )
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set(
            title=title,
            xlabel="Planet-frame velocity (km/s)",
            ylabel="Projected log-ratio (ppm)",
            xlim=(-80, 140),
        )
    axes.flat[0].legend(fontsize=8)
    ax = axes.flat[3]
    for name, label, color in [
        ("white", "White only", "C3"),
        ("primary", "GP: primary pixels", "C0"),
        ("alternate", "GP: alternate pixels", "C2"),
        ("science_only", "GP: no eclipse", "C4"),
    ]:
        if name + "_points" not in arrays:
            continue
        rv, contrast = arrays[name + "_points"]
        rq, aq = arrays[name + "_rv_contrast"]
        ax.scatter(rv, contrast, s=5, alpha=0.10, color=color)
        ax.errorbar(
            rq[1],
            aq[1],
            xerr=[[rq[1] - rq[0]], [rq[2] - rq[1]]],
            yerr=[[aq[1] - aq[0]], [aq[2] - aq[1]]],
            fmt="o",
            color=color,
            label=label,
        )
    ax.set(
        xlabel="Shared RV offset (km/s)",
        ylabel="Fixed-atmosphere contrast A",
        title="Conditional NUTS: marginal 90% intervals",
    )
    ax.legend(fontsize=8)
    fig.suptitle(
        "Fixed atmosphere + native time GP; independent orders, stride-2 pixels",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "Common spectra are projected out. Error bars show fitted white noise; bands show the latent joint fit, not independent validation. Alternate pixels are a sensitivity check.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(output / "gp_spectra.png", dpi=170)
    plt.close(fig)
    gap = arrays
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    for order, group in enumerate((2, 3)):
        time_grid, band = gap[f"group{group}_time"], gap[f"group{group}_band"]
        science, eclipse = (
            gap[f"group{group}_science_time"],
            gap[f"group{group}_eclipse_time"],
        )
        for mode, mode_label in enumerate(
            ("Grey coefficient", "Wavelength-slope coefficient")
        ):
            ax = axes[mode, order]
            ax.axvspan(
                science.max(),
                eclipse.min(),
                color="0.9",
                label="Excluded ingress / gap",
            )
            ax.fill_between(
                time_grid,
                1e6 * band[0, mode],
                1e6 * band[2, mode],
                color=f"C{order}",
                alpha=0.25,
            )
            ax.plot(time_grid, 1e6 * band[1, mode], color=f"C{order}")
            ax.axhline(0, color="0.6", lw=0.7)
            ax.plot(
                science,
                np.zeros_like(science),
                "|",
                color="0.35",
                transform=ax.get_xaxis_transform(),
            )
            ax.plot(
                eclipse,
                np.zeros_like(eclipse),
                "|",
                color="C2",
                transform=ax.get_xaxis_transform(),
            )
            ax.set(
                title=f"August 20, order {order + 1}: {mode_label}",
                ylabel="Relative to mean eclipse (ppm)",
                xlabel="Minutes after first selected exposure",
            )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Joint science + eclipse GP: 90% pointwise intervals across the gap",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "Bands propagate GP paths and hyperparameters, including covariance with the eclipse anchor. An absolute common GP level is not identified.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(output / "gp_gap.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs=2, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stellar-surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("gp", "white"), default="gp")
    parser.add_argument("--science-only", action="store_true")
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--pixel-offset", type=int, default=0)
    parser.add_argument("--amplitude-prior", type=float, default=300e-6)
    parser.add_argument("--length-prior", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--white-results", type=Path)
    parser.add_argument("--alternate-results", type=Path)
    parser.add_argument("--science-only-results", type=Path)
    run(parser.parse_args())
