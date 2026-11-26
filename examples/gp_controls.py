"""Validate conditional time-GP predictions and native planetary injections.

Eclipse blocks are held out chronologically; GP/white hyperparameters are fit
only to retained exposures. Predictive gains compare identical masks and common
intercept conventions, so fixed contrast-measure constants cancel. Injection
coverage conditions on known noise parameters and is not a calibration of the
full hyperparameter posterior or the original observational covariance.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest, chi2

from escape.time_gp import prepare_time_gp, time_gp_loglike, time_gp_posterior


def subset(group, times, pixels=None):
    """Retain the declared native wavelength columns for a temporal split."""
    result = dict(group)
    for key in ("phase", "time_min", "science", "eclipse"):
        if key in group:
            result[key] = np.asarray(group[key])[times]
    for key in ("log_data", "weight"):
        result[key] = np.asarray(group[key])[times]
        if pixels is not None:
            result[key] = result[key][:, pixels]
    if pixels is not None:
        result["stellar_velocity"] = np.asarray(group["stellar_velocity"])[pixels]
        if "pixel_index" in group:
            result["pixel_index"] = np.asarray(group["pixel_index"])[pixels]
    return result


def prepare(group):
    coordinate = (np.asarray(group["stellar_velocity"]) - 250.0) / 140.0
    basis = np.stack((np.ones_like(coordinate), coordinate))
    return prepare_time_gp(group["weight"], basis)


def hyper_fit(group, use_gp):
    """Fit the declared log-parameter priors on this training set only."""
    state = prepare(group)
    data, time = jnp.asarray(group["log_data"]), jnp.asarray(group["time_min"])

    def unpack(parameters):
        if use_gp:
            return (
                jnp.exp(parameters[:2]),
                jnp.exp(parameters[2]),
                jnp.exp(parameters[3]),
            )
        return jnp.zeros(2), jnp.asarray(20.0), jnp.exp(parameters[0])

    def objective(parameters):
        amplitudes, length, scale = unpack(parameters)
        value = time_gp_loglike(data, state, time, amplitudes, length, scale)
        penalty = 0.5 * (parameters[-1] / 0.35) ** 2
        if use_gp:
            penalty += 0.5 * jnp.sum((parameters[:2] - jnp.log(300e-6)) ** 2)
            penalty += 0.5 * (parameters[2] - jnp.log(20.0)) ** 2
        return -value + penalty

    evaluate = jax.jit(jax.value_and_grad(objective))
    if use_gp:
        seeds = [
            [np.log(300e-6), np.log(300e-6), np.log(20.0), 0.0],
            [np.log(100e-6), np.log(100e-6), np.log(5.0), 0.0],
        ]
        bounds = [
            (-18.0, -3.0),
            (-18.0, -3.0),
            (np.log(0.3), np.log(600.0)),
            (-2.0, 1.0),
        ]
    else:
        seeds, bounds = [[0.0]], [(-2.0, 1.0)]
    fits = [
        minimize(
            evaluate,
            seed,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-6},
        )
        for seed in seeds
    ]
    best = min(fits, key=lambda fit: fit.fun)
    amplitudes, length, scale = unpack(best.x)
    record = dict(
        use_gp=use_gp,
        log_parameters=best.x.tolist(),
        amplitudes_ppm=(np.asarray(amplitudes) * 1e6).tolist(),
        length_minutes=float(length),
        white_scale=float(scale),
        success=bool(best.success),
        message=str(best.message),
        log_posterior_without_constants=float(-best.fun),
        boundary_hit=bool(
            any(
                abs(value - low) < 1e-4 or abs(value - high) < 1e-4
                for value, (low, high) in zip(best.x, bounds)
            )
        ),
    )
    return (amplitudes, length, scale), record


def eclipse_controls(groups):
    """Compare held-out conditional GP/white scores without fitting test data."""
    records = []
    for original in groups:
        if not np.any(~np.asarray(original["science"])):
            continue
        eclipse = subset(original, ~np.asarray(original["science"]))
        # Identical complete native columns in every temporal split make the
        # omitted fixed-measure constants exactly cancel between GP and white.
        pixels = np.all(np.asarray(eclipse["weight"]) > 0, axis=0)
        eclipse = subset(eclipse, np.arange(len(eclipse["phase"])), pixels)
        full_state = prepare(eclipse)
        for index, held in enumerate(
            np.array_split(np.arange(len(eclipse["phase"])), 3)
        ):
            take = np.ones(len(eclipse["phase"]), dtype=bool)
            take[held] = False
            training = subset(eclipse, take)
            training_state = prepare(training)
            comparisons = []
            for use_gp in (False, True):
                hyper, fit = hyper_fit(training, use_gp)
                all_ll = time_gp_loglike(
                    eclipse["log_data"], full_state, eclipse["time_min"], *hyper
                )
                training_ll = time_gp_loglike(
                    training["log_data"], training_state, training["time_min"], *hyper
                )
                fit["conditional_log_score_up_to_common_measure_constant"] = float(
                    all_ll - training_ll
                )
                comparisons.append(fit)
            record = dict(
                night=int(original["night"]),
                order=int(original["order"]),
                block=index,
                native_pixels=int(pixels.sum()),
                training_exposures=int(take.sum()),
                heldout_exposures=len(held),
                heldout_phase_range=eclipse["phase"][held[[0, -1]]].tolist(),
                fits=comparisons,
                gp_minus_white_log_predictive_score=comparisons[1][
                    "conditional_log_score_up_to_common_measure_constant"
                ]
                - comparisons[0]["conditional_log_score_up_to_common_measure_constant"],
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    return records


def spectral_controls(groups, pixel_stride, simulations=64):
    """Compare eclipse residual spectral correlations with conditional draws.

    Lags refer to original native pixel indices, including gaps. Hyperparameters
    use all eclipse exposures; each simulated draw refits the latent GP mean and
    common spectrum but keeps hyperparameters fixed. The envelopes diagnose
    covariance discrepancies; they are not calibrated significance intervals.
    """
    records = []
    for original in groups:
        if not np.any(original["eclipse"]):
            continue
        group = subset(original, original["eclipse"])
        state = prepare(group)
        weight = np.asarray(state.weight)
        indices = np.asarray(group["pixel_index"])
        if np.any(np.diff(indices) <= 0):
            raise ValueError("Original native pixel indices must increase.")
        hyper, fit = hyper_fit(group, True)
        amplitudes, length, scale = map(np.asarray, hyper)
        time = np.asarray(group["time_min"])

        @jax.jit
        def standardized(data):
            posterior = time_gp_posterior(data, state, time, *hyper)
            residual = data - posterior.mean.T @ state.basis
            common = (state.weight * residual).sum(axis=0) / state.total_weight
            return (residual - common) * jnp.sqrt(state.weight) / scale

        lags = pixel_stride * np.arange(1, 6)
        pairs = []
        for lag in lags:
            left = np.arange(len(indices))
            right = np.searchsorted(indices, indices + lag)
            valid = right < len(indices)
            left, right = left[valid], right[valid]
            valid = indices[right] == indices[left] + lag
            left, right = left[valid], right[valid]
            mask = (weight[:, left] > 0) & (weight[:, right] > 0)
            pairs.append((left, right, mask))

        def correlations(residual):
            result = []
            for left, right, mask in pairs:
                a, b = residual[:, left][mask], residual[:, right][mask]
                a, b = a - a.mean(), b - b.mean()
                result.append(float(a @ b / np.sqrt((a @ a) * (b @ b))))
            return result

        actual = np.asarray(standardized(jnp.asarray(group["log_data"])))
        distance = np.sqrt(3) * abs(time[:, None] - time[None, :]) / length
        factor = np.linalg.cholesky(
            (1 + distance) * np.exp(-distance) + 1e-10 * np.eye(len(time))
        )
        rng = np.random.default_rng(9248 + group["order"])
        draws = []
        for _ in range(simulations):
            coefficients = (
                amplitudes[:, None] * (factor @ rng.normal(size=(len(time), 2))).T
            )
            noise = np.divide(
                scale, np.sqrt(weight), out=np.zeros_like(weight), where=weight > 0
            ) * rng.normal(size=weight.shape)
            data = coefficients.T @ np.asarray(state.basis) + noise
            draws.append(correlations(np.asarray(standardized(jnp.asarray(data)))))
        record = dict(
            night=int(group["night"]),
            order=int(group["order"]),
            exposures=len(time),
            pixels=len(indices),
            pixel_index=indices.tolist(),
            native_lags=lags.tolist(),
            valid_pair_counts=[int(mask.sum()) for _, _, mask in pairs],
            hyper=fit,
            residual_correlation=correlations(actual),
            conditional_simulated_95percent_envelope=np.quantile(
                draws, [0.025, 0.975], axis=0
            ).tolist(),
            simulation_count=simulations,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


def injection_controls(problem, repetitions=32):
    """Check conditional RV/log-amplitude coverage with known covariance.

    One arbitrary common spectral master is added to every realization. GP
    coefficients are drawn anew for each repetition and night/order. The same
    realization is fitted with the correct and omitted-GP likelihoods. This is
    a finite local coverage check at one atmosphere/cadence, not a general
    posterior-calibration claim or proof that real stellar variations have the
    chosen grey/slope form.
    """
    groups, states = problem["groups"], problem["states"]
    native = problem["native_log_models"]
    truth = np.array([2.7, np.log(1.1)])
    expected = tuple(np.asarray(value) for value in native(jnp.asarray(truth)))
    amplitudes = np.array([350e-6, 250e-6])
    length, white = 20.0, 0.8
    gp_factors, masters = [], []
    for group in groups:
        time = np.asarray(group["time_min"])
        distance = np.sqrt(3.0) * abs(time[:, None] - time[None, :]) / length
        gp_factors.append(
            np.linalg.cholesky(
                (1 + distance) * np.exp(-distance) + 1e-10 * np.eye(len(time))
            )
        )
        velocity = np.asarray(group["stellar_velocity"])
        masters.append(
            0.003 * np.sin(velocity / 35.0) + 0.002 * np.cos(velocity / 91.0)
        )

    functions = {}
    for use_gp in (False, True):
        amp = jnp.asarray(amplitudes if use_gp else np.zeros(2))

        def negative(parameters, data, amp=amp):
            return -sum(
                time_gp_loglike(y - model, state, group["time_min"], amp, length, white)
                for y, model, state, group in zip(
                    data, native(parameters), states, groups
                )
            )

        functions[use_gp] = (
            jax.jit(jax.value_and_grad(negative)),
            jax.jit(jax.hessian(negative)),
        )
    records = []
    bounds = [(-15.0, 15.0), (np.log(0.05), np.log(4.0))]
    for truth_gp, fitted_gp in [(False, False), (True, True), (True, False)]:
        # Reset this seed so correct/omitted GP see identical GP injections.
        local_rng = np.random.default_rng(90422 if truth_gp else 91534)
        evaluate, hessian = functions[fitted_gp]
        trials = []
        for repetition in range(repetitions):
            data = []
            for group, state, model, factor, master in zip(
                groups, states, expected, gp_factors, masters
            ):
                weight = np.asarray(state.weight)
                standard_error = np.divide(
                    1, np.sqrt(weight), out=np.zeros_like(weight), where=weight > 0
                )
                noise = white * standard_error * local_rng.normal(size=weight.shape)
                coefficients = (
                    amplitudes[:, None]
                    * (factor @ local_rng.normal(size=(len(factor), 2))).T
                )
                source = coefficients.T @ np.asarray(state.basis) if truth_gp else 0.0
                data.append(jnp.asarray(model + master[None, :] + source + noise))
            data = tuple(data)
            fit = minimize(
                lambda theta: evaluate(theta, data),
                truth,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 80, "ftol": 1e-11, "gtol": 1e-6},
            )
            information = np.asarray(hessian(jnp.asarray(fit.x), data))
            valid = bool(np.linalg.eigvalsh(information).min() > 0)
            covariance = (
                np.linalg.inv(information) if valid else np.full((2, 2), np.nan)
            )
            difference = fit.x - truth
            mahalanobis = float(difference @ information @ difference)
            trials.append(
                dict(
                    repetition=repetition,
                    parameters=fit.x.tolist(),
                    success=bool(fit.success),
                    positive_information=valid,
                    standard_error=np.sqrt(np.diag(covariance)).tolist(),
                    joint95_contains_truth=bool(
                        valid and mahalanobis <= chi2.ppf(0.95, 2)
                    ),
                    mahalanobis_squared=mahalanobis,
                )
            )
        coverage = sum(row["joint95_contains_truth"] for row in trials)
        records.append(
            dict(
                gp_injected=truth_gp,
                gp_fitted=fitted_gp,
                repetitions=repetitions,
                joint95_coverage_count=coverage,
                joint95_coverage_fraction=coverage / repetitions,
                binomial_pvalue_against_nominal95=float(
                    binomtest(coverage, repetitions, 0.95).pvalue
                ),
                all_optimizer_success=all(row["success"] for row in trials),
                mean_parameter_error=(
                    np.mean([row["parameters"] for row in trials], axis=0) - truth
                ).tolist(),
                trials=trials,
            )
        )
        print(
            json.dumps({k: v for k, v in records[-1].items() if k != "trials"}),
            flush=True,
        )
    # A common master leaves the objective unchanged at a fixed model point.
    plain = tuple(jnp.asarray(value) for value in expected)
    shifted = tuple(
        jnp.asarray(value + master) for value, master in zip(expected, masters)
    )
    evaluate = functions[True][0]
    master_difference = float(evaluate(truth, shifted)[0] - evaluate(truth, plain)[0])
    return dict(
        truth=truth.tolist(),
        parameter_names=["rv_kms", "log_amplitude"],
        noise=dict(
            amplitudes_ppm=(amplitudes * 1e6).tolist(),
            length_minutes=length,
            white_scale=white,
        ),
        arbitrary_master_loglike_difference=master_difference,
        uncertainty_scope="Local observed-information Gaussian ellipses conditional on known covariance, with independent repeated synthetic realizations.",
        results=records,
    )


def plot_summary(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    controls = report["eclipse"]
    labels = [f'O{row["order"]+1}, block {row["block"]+1}' for row in controls]
    gains = [row["gp_minus_white_log_predictive_score"] for row in controls]
    axes[0].bar(
        labels, gains, color=["#0072b2" if value >= 0 else "#d55e00" for value in gains]
    )
    axes[0].axhline(0, color=".4", lw=0.8)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set(
        title="Eclipse blocks: unseen time prediction",
        ylabel="GP minus white log predictive score",
    )
    injections = report["injection"]["results"]
    labels = ["White truth\nwhite fit", "GP truth\nGP fit", "GP truth\nwhite fit"]
    axes[1].bar(
        labels,
        [row["joint95_coverage_fraction"] for row in injections],
        color=[".6", "#0072b2", "#d55e00"],
    )
    axes[1].axhline(0.95, color=".3", ls="--", label="Nominal 95%")
    for i, row in enumerate(injections):
        axes[1].text(
            i,
            row["joint95_coverage_fraction"] + 0.02,
            f'{row["joint95_coverage_count"]}/{row["repetitions"]}',
            ha="center",
        )
    axes[1].set(
        ylim=(0, 1.12),
        title="Conditional injection coverage",
        ylabel="RV/log-amplitude joint ellipse coverage",
    )
    axes[1].legend(frameon=False)
    fig.suptitle(
        "GP controls: predictive utility and preservation of the planetary signal"
    )
    fig.text(
        0.5,
        0.015,
        "Eclipse hyperparameters use training exposures only. Injection uncertainties condition on known noise parameters.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output / "gp_controls.png", dpi=180)
    plt.close(fig)


def plot_review_receipt(receipt, output):
    """Plot compact public control summaries without loading observation arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ["#0072b2", "#d55e00"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    for row in receipt["covariance_table"]:
        selection = row["selection"]
        position = {"all_native": 0, "stride2_offset0": 1, "stride2_offset1": 2}[
            selection
        ]
        order = row["order"]
        x = position + (order - 0.5) * 0.22
        low, high = row["nearest_conditional_95percent_envelope"]
        axes[0].plot(
            [x, x],
            [low, high],
            color=colors[order],
            alpha=0.28,
            lw=9,
            solid_capstyle="butt",
        )
        axes[0].plot(x, row["nearest_residual_correlation"], "o", color=colors[order])
    axes[0].axhline(0, color=".6", lw=0.8)
    axes[0].set(
        xticks=[0, 1, 2],
        xticklabels=[
            "All pixels\noriginal lag 1",
            "Even pixels\noriginal lag 2",
            "Odd pixels\noriginal lag 2",
        ],
        ylabel="Residual correlation",
        title="A. Eclipse residuals\nAdjacent-pixel correlation is reduced",
        ylim=(-0.08, 0.28),
    )
    axes[0].legend(
        handles=[
            Line2D([], [], color=color, marker="o", ls="", label=f"Order {i+1}")
            for i, color in enumerate(colors)
        ],
        frameon=False,
        loc="upper right",
    )
    matched = "primary_heldout_table" in receipt
    heldout = receipt.get("primary_heldout_table", receipt["heldout_table"])
    for order, color in enumerate(colors):
        rows = [row for row in heldout if row["order"] == order]
        axes[1].bar(
            [row["block"] + 1 + (order - 0.5) * 0.32 for row in rows],
            [row["gp_minus_white_log_predictive_score"] for row in rows],
            width=0.32,
            color=color,
        )
    axes[1].axhline(0, color=".4", lw=0.8)
    axes[1].set(
        xticks=[1, 2, 3],
        xlabel="Held-out eclipse block",
        ylabel="GP minus white log predictive score",
        title=(
            "B. Prediction: primary even-pixel sample\nNo clear predictive gain in eclipse data"
            if matched
            else "B. Prediction: historical all-pixel check\nBenefit varies between blocks and orders"
        ),
    )
    rows = receipt["injection_table"]
    axes[2].bar(
        [0, 1, 2],
        [row["joint95_coverage_count"] / row["repetitions"] for row in rows],
        color=[".6", colors[0], colors[1]],
    )
    axes[2].axhline(0.95, color=".3", ls="--", lw=1)
    for i, row in enumerate(rows):
        axes[2].text(
            i,
            1.04,
            f'{row["joint95_coverage_count"]}/{row["repetitions"]}',
            ha="center",
        )
    axes[2].set(
        xticks=[0, 1, 2],
        xticklabels=[
            "White truth\nwhite fit",
            "GP truth\nGP fit",
            "GP truth\nwhite fit",
        ],
        ylim=(0, 1.17),
        ylabel="Fraction containing injected RV / log contrast",
        title="C. Injections: historical all-pixel check\nKnown noise; nominal joint coverage 95%",
    )
    fig.suptitle(
        "GP controls: eclipse data expose a covariance limitation", fontsize=14
    )
    fig.text(
        0.5,
        0.075,
        "A: dots = data; translucent bars = conditional 95% ranges from 64 simulations. Remaining longer-lag discrepancies are not shown.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.035,
        "B: hyperparameters fit to training exposures only; plug-in scores are not Bayes factors. C: known-noise tests do not calibrate real-data NUTS uncertainties.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(output / "gp_controls_review.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs=2, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--stellar-surface", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--pixel-offset", type=int, default=0)
    parser.add_argument("--spectral-only", action="store_true")
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.review_receipt is not None:
        plot_review_receipt(json.loads(args.review_receipt.read_text()), args.output)
        return
    if args.plot_only:
        plot_summary(json.loads((args.output / "review.json").read_text()), args.output)
        return
    if not args.data:
        parser.error("Supply both observation paths.")
    if not args.spectral_only and (args.model is None or args.stellar_surface is None):
        parser.error("Supply data, model and stellar surface for the calculation.")
    if args.repetitions < 2:
        parser.error("Use at least two independent injection realizations.")
    from gp_halpha import load_gp_data, make_gp_problem

    jax.config.update("jax_enable_x64", True)
    groups = load_gp_data(
        args.data, pixel_stride=args.pixel_stride, pixel_offset=args.pixel_offset
    )
    selection = dict(stride=args.pixel_stride, offset=args.pixel_offset)
    if args.spectral_only:
        report = dict(
            pixel_selection=selection,
            scope="Eclipse-only covariance discrepancy control; conditional simulation envelopes fix fitted hyperparameters and refit latent means and common spectra. These are not calibrated significance intervals or proof of model adequacy.",
            results=spectral_controls(groups, args.pixel_stride),
            source_sha256={
                str(path): sha256(path.read_bytes()).hexdigest()
                for path in [
                    *args.data,
                    Path(__file__),
                    Path(__file__).with_name("gp_halpha.py"),
                ]
            },
        )
        (args.output / "spectral_correlation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        return
    problem = make_gp_problem(
        groups, dict(np.load(args.model)), dict(np.load(args.stellar_surface))
    )
    eclipse = eclipse_controls(groups)
    (args.output / "eclipse.json").write_text(
        json.dumps(eclipse, indent=2, allow_nan=False) + "\n"
    )
    injection = injection_controls(problem, args.repetitions)
    report = dict(
        pixel_selection=selection,
        eclipse=eclipse,
        injection=injection,
        scope="Finite controls for the native common-master-marginalized GP. GP test functions are marginalized; empirical supplied errors are scaled jointly.",
        prior=dict(
            log_amplitude_mean=np.log(300e-6),
            log_amplitude_sigma=1.0,
            log_length_mean=np.log(20.0),
            log_length_sigma=1.0,
            log_white_sigma=0.35,
        ),
        limitations=[
            "Eclipse predictive scores are plug-in training-MAP hyperparameter comparisons, not hyperparameter-marginalized Bayesian evidence.",
            "The fixed contrast-measure constants cancel in each GP-minus-white gain; individual displayed conditional scores are not absolute densities.",
            "The original extraction and cross-order covariance are not reconstructed.",
            "Injection coverage is local and conditional on known noise. It does not validate the full atmospheric model, arbitrary systematics or joint hyperparameter calibration.",
        ],
        source_sha256={
            str(path): sha256(path.read_bytes()).hexdigest()
            for path in [*args.data, args.model, args.stellar_surface, Path(__file__)]
        },
    )
    (args.output / "review.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot_summary(report, args.output)


if __name__ == "__main__":
    main()
