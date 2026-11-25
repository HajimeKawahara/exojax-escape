"""Fit one fixed-opacity longitude map and test it on the other night.

The shared parameters are RV, global source contrast, cosine and sine map
coefficients. This uses native time contrasts, not nightly coadded spectra.
The two map coefficients redistribute local source terms while preserving
opacity. Their local covariance is a diagnostic, not a posterior uncertainty.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from escape.longitude_map import longitude_spectrum
from time_halpha import load_time_data, make_time_objective, select_times


def load_kernel_bank(directory, coarse_nodes=False):
    bank = [dict(np.load(path)) for path in sorted(Path(directory).glob("phase_*.npz"))]
    bank.sort(key=lambda row: float(row["phase"]))
    if len(bank) < 2:
        raise ValueError("At least two phase kernels are required.")
    if any(
        not np.array_equal(row["velocity_kms"], bank[0]["velocity_kms"])
        or not np.array_equal(row["longitude"], bank[0]["longitude"])
        for row in bank
    ):
        raise ValueError("All phase kernels must share spectral and longitude nodes.")
    if np.any(np.diff([row["phase"] for row in bank]) <= 0):
        raise ValueError("Phase kernels must have unique, strictly increasing phases.")
    nodes = bank[0]["longitude"][::2] if coarse_nodes else bank[0]["longitude"]
    return dict(
        phase=np.array([row["phase"] for row in bank]),
        longitude=nodes,
        velocity=bank[0]["velocity_kms"],
        kernel=np.array(
            [row["coarse_kernel" if coarse_nodes else "kernel"] for row in bank]
        ),
    )


def map_intrinsic(bank, uniform=False):
    phases, kernels, nodes = map(
        jnp.asarray, (bank["phase"], bank["kernel"], bank["longitude"])
    )

    def intrinsic(parameters, exposure_phase):
        a, b = (0.0, 0.0) if uniform else (parameters[2], parameters[3])
        spectra = longitude_spectrum(kernels, nodes, a, b)
        sampled = jax.vmap(
            lambda column: jnp.interp(exposure_phase, phases, column),
            in_axes=1,
            out_axes=1,
        )(spectra)
        return sampled * parameters[1], parameters[0]

    return intrinsic


def fit_map(objective, uniform=False, map_bound=3.0):
    jacobian = jax.jit(jax.jacfwd(objective))
    bounds = (
        [-15.0, 0.0] if uniform else [-15.0, 0.0, -map_bound, -map_bound],
        [15.0, 4.0] if uniform else [15.0, 4.0, map_bound, map_bound],
    )
    seeds = (
        [[2.0, 1.0], [0.0, 1.5]]
        if uniform
        else [[2.0, 1.0, 0.0, 0.0], [0.0, 1.5, -1.0, 1.0], [4.0, 0.7, 1.0, -1.0]]
    )
    fits = [
        least_squares(
            objective,
            seed,
            jac=jacobian,
            bounds=bounds,
            x_scale="jac",
            max_nfev=140,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-7,
        )
        for seed in seeds
    ]
    fit = min(fits, key=lambda result: result.fun @ result.fun)
    point = fit.x.copy()
    point[0] += 0.017
    analytic = np.asarray(jacobian(point))
    step = 1e-4
    finite = np.column_stack(
        [
            (
                np.asarray(objective(point + step * e))
                - np.asarray(objective(point - step * e))
            )
            / (2 * step)
            for e in np.eye(len(point))
        ]
    )
    jac = np.asarray(jacobian(fit.x))
    singular = np.linalg.svd(jac, compute_uv=False)
    covariance = np.linalg.pinv(jac.T @ jac)
    record = dict(
        parameters=fit.x.tolist(),
        chi2=float(fit.fun @ fit.fun),
        success=bool(fit.success),
        message=str(fit.message),
        optimality=float(fit.optimality),
        bounds=[list(x) for x in bounds],
        boundary_parameters=np.flatnonzero(
            (fit.x - bounds[0] < 0.02) | (np.asarray(bounds[1]) - fit.x < 0.02)
        ).tolist(),
        derivative_max_scaled_error=float(
            np.max(np.abs(analytic - finite) / (1 + np.abs(finite)))
        ),
        jacobian_singular_values=singular.tolist(),
        local_diagonal_error=np.sqrt(np.diag(covariance)).tolist(),
        all_start_chi2=[float(result.fun @ result.fun) for result in fits],
    )
    return record


def score(objective, parameters):
    residual = np.asarray(objective(jnp.asarray(parameters)))
    return float(residual @ residual)


def run(data, stellar_surface, kernels, output, phase_max=0.435):
    jax.config.update("jax_enable_x64", True)
    output.mkdir(exist_ok=True, parents=True)
    groups, surface = (
        load_time_data(data, phase_range=(0.30, phase_max)),
        dict(np.load(stellar_surface)),
    )
    bank = load_kernel_bank(kernels)
    if any(
        g["phase"].min() < bank["phase"].min() or g["phase"].max() > bank["phase"].max()
        for g in groups
    ):
        raise ValueError("Kernels must bracket every retained exposure.")
    records, arrays = [], {}
    for selection, selected in [
        ("pooled", groups),
        ("night0", [groups[0]]),
        ("night1", [groups[1]]),
    ]:
        for uniform in (True, False):
            for drift in [False, True] if selection == "pooled" else [False]:
                intrinsic = map_intrinsic(bank, uniform)
                objective, native = make_time_objective(
                    selected, bank["velocity"], surface, intrinsic, drift=drift
                )
                result = fit_map(objective, uniform)
                mode = "uniform" if uniform else "map"
                result.update(selection=selection, mode=mode, drift=drift)
                name = f"{selection}_{mode}" + ("_drift" if drift else "")
                for group, flux in zip(
                    selected, native(jnp.asarray(result["parameters"]))
                ):
                    arrays[f"{name}_night{group['night']}_log_model"] = np.log(
                        np.asarray(flux)
                    )
                if selection != "pooled":
                    train, other = selected[0], groups[1 - selected[0]["night"]]
                    low = max(train["phase"].min(), other["phase"].min())
                    high = min(train["phase"].max(), other["phase"].max())
                    test = select_times(
                        other, (other["phase"] >= low) & (other["phase"] <= high)
                    )
                    held, _ = make_time_objective(
                        [test], bank["velocity"], surface, intrinsic
                    )
                    result["held_out"] = dict(
                        night=test["night"],
                        exposures=len(test["phase"]),
                        phase_range=[float(low), float(high)],
                        chi2=score(held, result["parameters"]),
                        rank=int(np.maximum((test["weight"] > 0).sum(0) - 1, 0).sum()),
                    )
                    full, full_native = make_time_objective(
                        [other], bank["velocity"], surface, intrinsic
                    )
                    result["held_out_full_phase_diagnostic"] = dict(
                        chi2=score(full, result["parameters"]),
                        scope="Includes unobserved training phases; extrapolation diagnostic only.",
                    )
                    arrays[f"{name}_prediction_night{other['night']}_log_model"] = (
                        np.log(
                            np.asarray(
                                full_native(jnp.asarray(result["parameters"]))[0]
                            )
                        )
                    )
                records.append(result)
                (output / "progress.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                print(json.dumps(result), flush=True)
    for group in groups:
        for key in ("phase", "stellar_velocity", "order", "log_data", "weight"):
            arrays[f"night{group['night']}_{key}"] = group[key]
    np.savez_compressed(output / "arrays.npz", **arrays)
    report = dict(
        results=records,
        parameter_names=["rv_kms", "contrast", "cosine", "sine"],
        kernel_directory=str(kernels),
        phase_range=[0.30, phase_max],
        normalization="Uniform dayside surface area mean.",
        scope=__doc__,
        held_out_scope="Freeze planet parameters. Profile only a common native log-spectrum on the test night. Primary score uses common phase coverage.",
        input_sha256={
            str(path): sha256(Path(path).read_bytes()).hexdigest()
            for path in [
                *data,
                stellar_surface,
                *sorted(kernels.glob("phase_*.npz")),
                Path(__file__),
            ]
        },
        limitations=[
            "Local source redistribution at fixed opacity, NLTE populations and dynamics; no new thermal equilibrium.",
            "The supplied diagonal archive errors do not recover extraction or reference covariance.",
            "A local Jacobian error bar is not a calibrated posterior; boundary fits and phase extrapolation need particular caution.",
            "The sin-longitude flow and radial wind remain fixed at their previously saved values.",
        ],
    )
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")


def plot_summary(output):
    """Plot identical native time projections for data and fitted models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ROOT = DIR = Path(output)
    a = dict(np.load(DIR / "arrays.npz"))
    report = json.loads((DIR / "review.json").read_text())
    records = report["results"]
    phase_max = report["phase_range"][1]

    def record(selection, mode, drift=False):
        return next(
            r
            for r in records
            if r["selection"] == selection and r["mode"] == mode and r["drift"] == drift
        )

    def project(value, weight):
        return value - np.sum(weight * value, axis=0, keepdims=True) / np.sum(
            weight, axis=0, keepdims=True
        )

    def bins(night, low, high, mode):
        phase = a[f"night{night}_phase"]
        coord = a[f"night{night}_stellar_velocity"]
        weight = a[f"night{night}_weight"]
        data = (
            a[f"night{night}_log_data"]
            if mode == "data"
            else a[f"pooled_{mode}_night{night}_log_model"]
        )
        projected = project(data, weight)
        orb = 244.8 * np.sin(2 * np.pi * phase)
        velocity = 299792.458 * (
            (1 + coord[None] / 299792.458) / (1 + orb[:, None] / 299792.458) - 1
        )
        centers = np.arange(-80.0, 81.0, 8.0)
        values = []
        errors = []
        for center in centers:
            keep = (
                (phase[:, None] >= low)
                & (phase[:, None] < high)
                & (velocity >= center - 4)
                & (velocity < center + 4)
            )
            local = weight * keep
            total = local.sum()
            if total == 0:
                values.append(np.nan)
                errors.append(np.nan)
                continue
            value = np.sum(local * projected) / total
            # Same-column exposure covariance introduced by the weighted common-mode
            # projection is retained under the declared diagonal input-error proxy.
            variance = (total - np.sum(local.sum(0) ** 2 / weight.sum(0))) / total**2
            values.append(value)
            errors.append(np.sqrt(max(0, variance)))
        return centers, np.array(values) * 1e6, np.array(errors) * 1e6

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
    colors = ["#0072B2", "#D55E00"]
    for panel, (low, high) in enumerate(
        [(0.30, 0.35), (0.35, 0.40), (0.40, phase_max)]
    ):
        ax = axs[0, panel]
        for n, color in enumerate(colors):
            x, d, e = bins(n, low, high, "data")
            if not np.isfinite(d).any():
                continue
            ax.errorbar(
                x,
                d,
                yerr=e,
                fmt="o",
                ms=3,
                color=color,
                alpha=0.65,
                lw=0.7,
                label=["Aug 8", "Aug 20"][n] + " data",
            )
            _, uniform, _ = bins(n, low, high, "uniform")
            _, mapped, _ = bins(n, low, high, "map")
            ax.plot(
                x,
                uniform,
                color=color,
                ls="--",
                lw=1.3,
                label=["Aug 8", "Aug 20"][n] + " uniform",
            )
            ax.plot(
                x,
                mapped,
                color=color,
                lw=2,
                label=["Aug 8", "Aug 20"][n] + " shared map",
            )
        ax.axhline(0, color=".7", lw=0.6)
        ax.set(
            title=f"{low:.2f} ≤ phase < {high:g}",
            xlabel="Planet-frame velocity [km/s]",
            ylabel="Within-night log contrast [ppm]",
        )
        ax.legend(fontsize=7, ncol=2)
    ax = axs[1, 0]
    lon = np.linspace(-np.pi / 2, np.pi / 2, 300)
    gn, gw = np.polynomial.legendre.leggauss(128)
    for selection, drift, color, style, label in [
        ("pooled", False, "#CC79A7", "-", "Pooled: source map"),
        ("pooled", True, "#009E73", "-", "Pooled + grey/slope drift"),
        ("night0", False, colors[0], "--", "Fit Aug 8 only"),
        ("night1", False, colors[1], ":", "Fit Aug 20 only"),
    ]:
        p = record(selection, "map", drift)["parameters"]
        norm = (
            np.sum(
                gw
                * np.exp(p[2] * np.cos(gn * np.pi / 2) + p[3] * np.sin(gn * np.pi / 2))
            )
            / 2
        )
        w = np.exp(p[2] * np.cos(lon) + p[3] * np.sin(lon)) / norm
        ax.plot(lon * 180 / np.pi, w, color=color, ls=style, lw=2, label=label)
    ax.axhline(1, color=".5", ls=":", lw=1)
    ax.set(
        xlabel="Longitude from substellar meridian [deg]",
        ylabel="Local source weight W",
        title="One map shared by every phase",
    )
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    labels = ["Pooled", "Pooled + drift", "Aug 8 → Aug 20", "Aug 20 → Aug 8"]
    gain = []
    for sel, drift, held in [
        ("pooled", False, False),
        ("pooled", True, False),
        ("night0", False, True),
        ("night1", False, True),
    ]:
        u, m = record(sel, "uniform", drift), record(sel, "map", drift)
        gain.append(
            (u["held_out"]["chi2"] - m["held_out"]["chi2"])
            if held
            else u["chi2"] - m["chi2"]
        )
    ax.barh(np.arange(4), gain, color=["#CC79A7", "#009E73", colors[0], colors[1]])
    ax.axvline(0, color=".2", lw=0.8)
    ax.set(
        yticks=np.arange(4),
        yticklabels=labels,
        xlabel="Q(uniform) − Q(map); positive favors map",
        title="Fit gain versus withheld-night prediction",
    )
    ax.invert_yaxis()
    ax.set_xlim(min(-1.0, min(gain) * 1.1), max(1.0, max(gain) * 1.25))
    for i, value in enumerate(gain):
        ax.text(
            max(0.0, value) + 0.3,
            i,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax = axs[1, 2]
    ax.axis("off")
    p = record("pooled", "map")["parameters"]
    pd = record("pooled", "map", True)["parameters"]
    text = (
        "What this test changes\n\n"
        + r"$j_\nu(\lambda)\to W(\lambda)j_\nu(\lambda)$"
        + "\n"
        + r"$W\propto\exp(a\cos\lambda+b\sin\lambda)$"
        + "\nOpacity, populations and wind stay fixed.\nEach source keeps its own Doppler velocity.\n\n"
        + f"Pooled a={p[2]:.2f}, b={p[3]:.2f} (bound ±3)\nWith drift a={pd[2]:.2f}, b={pd[3]:.2f}\n\n"
        + "A shared map is not established:\ncontinuum drift changes the inferred map;\nheld-out predictions test repeatability.\n\n"
        + "Top: the same per-native-pixel temporal\nprojection is applied to data and models.\nAbsolute continuum/mean spectrum is lost.\nMissing rest-frame bins are left blank."
    )
    ax.text(0, 1, text, va="top", fontsize=10, linespacing=1.35)
    fig.suptitle(
        "Testing a shared longitude source map with withheld-night prediction",
        fontsize=14,
    )
    fig.text(
        0.02,
        0.015,
        "KELT-9b, two archived Keck/KPF nights. Error bars retain projection covariance only under the supplied diagonal-error proxy; Q is not a calibrated likelihood.\nTraining parameters are frozen for held-out scores; each test night profiles only its common native log spectrum, on common phase coverage.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.065, 1, 0.95])
    fig.savefig(ROOT / "longitude_map.png", dpi=180)
    return ROOT / "longitude_map.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs=2, type=Path)
    parser.add_argument("--stellar-surface", type=Path)
    parser.add_argument("--kernels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase-max",
        type=float,
        default=0.435,
        help="Upper science phase cut; default excludes atmospheric ingress.",
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_summary(args.output)
        parser.exit()
    if args.data is None or args.stellar_surface is None or args.kernels is None:
        parser.error("Fitting requires --data, --stellar-surface and --kernels.")
    run(args.data, args.stellar_surface, args.kernels, args.output, args.phase_max)

    plot_summary(args.output)
