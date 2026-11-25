"""Screen a night-dependent background using channels outside the planet line.

The archive contains residual spectra: do not divide them by another master.
The model NPZ supplies post-master stellar-frame arrays for every exposure:
phase, night, planet_velocity_kms, stellar_velocity_kms, reference_residuals,
and perturbed_residuals. The last array is the explicit phase-amplitude control,
not a best fit. Preserve the complete nightly exposure list when producing it.
Optional fixed_master_residuals uses the perturbed normalized spectrum divided
by the reference master, isolating the direct change from master feedback.

Example::

    python examples/night_residuals.py --data night1.h5 night2.h5 \
        --model processed_exposures.npz --output results/night_residuals

Generated coadds support --plot-only without the archive or stellar input.
This diagnostic does not adopt a correction or identify atmospheric variability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_KMS = 299792.458
REST_ANGSTROM = 1e8 / 15233.146


def run(data_paths: list[Path], model_path: Path, output: Path) -> None:
    """Fit only off-line channels, then predict held-out continuum and line."""
    output.mkdir(parents=True, exist_ok=True)
    with np.load(model_path) as archive:
        model = dict(archive)
    phase, night = model["phase"], model["night"]
    vp = model["planet_velocity_kms"]
    fixed_master = "fixed_master_residuals" in model
    if phase.ndim != 1 or night.shape != phase.shape or vp.shape != phase.shape:
        raise ValueError("Phase, night and orbital velocity must match exposures.")
    if len(data_paths) != 2 or set(np.unique(night)) != {0, 1}:
        raise ValueError("This matched-night example requires exactly two nights.")
    shape = (len(phase), len(model["stellar_velocity_kms"]))
    fields = ["reference_residuals", "perturbed_residuals"]
    if fixed_master:
        fields.append("fixed_master_residuals")
    for name in fields:
        if model[name].shape != shape or not np.all(np.isfinite(model[name])):
            raise ValueError(f"{name} must be a finite exposure by velocity array.")
    if np.any(np.diff(model["stellar_velocity_kms"]) <= 0):
        raise ValueError("The stellar velocity grid must be strictly increasing.")
    if (
        model["stellar_velocity_kms"][0] > 110
        or model["stellar_velocity_kms"][-1] < 390
    ):
        raise ValueError(
            "The model must cover the complete 110..390 km/s audit domain."
        )
    overlap = [
        max(phase[night == n].min() for n in (0, 1)),
        min(phase[night == n].max() for n in (0, 1)),
    ]
    matched = (phase >= overlap[0]) & (phase <= overlap[1])
    C, REST = C_KMS, REST_ANGSTROM
    center = np.arange(-99.0, 251.0, 2.0)
    edges = np.r_[center - 1.0, center[-1] + 1.0]
    count = len(center)
    den = np.zeros((2, 2, count))
    num = np.zeros((4 if fixed_master else 3, 2, 2, count))
    exposure_count = np.zeros((2, 2, count), int)
    offset = 0
    for n, path in enumerate(data_paths):
        with h5py.File(path) as h:
            ph, wave, data, error = [h[k][:] for k in ("phase", "wave", "flux", "err")]
        np.testing.assert_array_equal(ph, phase[offset : offset + len(ph)])
        if np.any(night[offset : offset + len(ph)] != n):
            raise ValueError("Model and data night order do not match.")
        for i in range(len(ph)):
            e = offset + i
            if not matched[e]:
                continue
            for order in (0, 1):
                vs = C * (wave[order] / REST - 1)
                vr = C * ((1 + vs / C) / (1 + vp[e] / C) - 1)
                bi = np.searchsorted(edges, vr, side="right") - 1
                keep = (bi >= 0) & (bi < count) & (vs >= 110) & (vs <= 390)
                keep &= (
                    np.isfinite(data[i, order])
                    & np.isfinite(error[i, order])
                    & (error[i, order] > 0)
                )
                index = bi[keep]
                weight = 1 / error[i, order, keep] ** 2
                m = np.interp(
                    vs[keep],
                    model["stellar_velocity_kms"],
                    model["reference_residuals"][e],
                )
                changed = np.interp(
                    vs[keep],
                    model["stellar_velocity_kms"],
                    model["perturbed_residuals"][e],
                )
                den[n, order] += np.bincount(index, weights=weight, minlength=count)
                values_to_bin = [data[i, order, keep], m, changed - m]
                if fixed_master:
                    direct = np.interp(
                        vs[keep],
                        model["stellar_velocity_kms"],
                        model["fixed_master_residuals"][e],
                    )
                    values_to_bin.append(direct - m)
                for target, values in enumerate(values_to_bin):
                    num[target, n, order] += np.bincount(
                        index, weights=weight * values, minlength=count
                    )
                exposure_count[n, order, np.unique(index)] += 1
        offset += len(ph)
    if offset != len(phase):
        raise ValueError("Data and model must retain the same complete exposure list.")

    def coadd(orders):
        weight = den[:, orders, :].sum(1)
        numerator = num[:, :, orders, :].sum(2)
        value = np.divide(
            numerator,
            weight[None],
            out=np.full_like(numerator, np.nan),
            where=weight[None] > 0,
        )
        variance = np.divide(
            1.0, weight, out=np.full_like(weight, np.nan), where=weight > 0
        ).sum(0)
        difference = value[:, 0] - value[:, 1]
        result = {
            "data": difference[0],
            "model": difference[1],
            "residual": difference[0] - difference[1],
            "leakage": difference[2],
            "variance": variance,
        }
        if fixed_master:
            result["direct_leakage"] = difference[3]
            result["master_leakage"] = difference[2] - difference[3]
        return result

    all_coadds = {"both": coadd([0, 1]), "order0": coadd([0]), "order1": coadd([1])}

    def fit_template(c, low, high, degree, field="residual"):
        take = (
            np.isfinite(c[field])
            & np.isfinite(c["variance"])
            & (center >= low)
            & (center <= high)
        )
        x = np.column_stack([((center - 150) / 100) ** k for k in range(degree + 1)])
        weighted = x[take] / np.sqrt(c["variance"][take])[:, None]
        covariance = np.linalg.inv(weighted.T @ weighted)
        coefficients = covariance @ (
            weighted.T @ (c[field][take] / np.sqrt(c["variance"][take]))
        )
        return x @ coefficients, coefficients, covariance, take

    def region(c, low, high, correction=0.0):
        y = c["residual"] - correction
        take = (
            np.isfinite(y)
            & np.isfinite(c["variance"])
            & (center >= low)
            & (center <= high)
        )
        w = 1 / c["variance"][take]
        mean = np.sum(w * y[take]) / w.sum()
        return {
            "bins": int(take.sum()),
            "mean_ppm": float(mean * 1e6),
            "formal_error_ppm": float(1e6 / np.sqrt(w.sum())),
            "chi2": float(np.sum(w * y[take] ** 2)),
        }

    def prediction_error(c, low, high, covariance):
        take = np.isfinite(c["variance"]) & (center >= low) & (center <= high)
        w = 1 / c["variance"][take]
        design = np.column_stack(
            [((center[take] - 150) / 100) ** k for k in range(len(covariance))]
        )
        average = np.sum(design * w[:, None], axis=0) / w.sum()
        return float(np.sqrt(average @ covariance @ average) * 1e6)

    records, predictions = [], {}
    for orders, c in all_coadds.items():
        for low, high in [(80, 200), (120, 200), (150, 230)]:
            for degree in [0, 1]:
                pred, coef, cov, training = fit_template(c, low, high, degree)
                key = f"{orders}_{low}_{high}_degree{degree}"
                predictions[key] = pred
                row = {
                    "key": key,
                    "training_domain_kms": [low, high],
                    "degree": degree,
                    "training_bins": int(training.sum()),
                    "coefficients": coef.tolist(),
                    "coefficient_covariance_formal": cov.tolist(),
                    "regions_before": {},
                    "regions_after": {},
                }
                for name, a, b in [
                    ("red_peak", 15, 45),
                    ("red_wing", 50, 70),
                    ("off_line_near", 80, 120),
                    ("off_line_far", 150, 200),
                ]:
                    row["regions_before"][name] = region(c, a, b)
                    row["regions_after"][name] = region(c, a, b, pred)
                row["red_peak_template_formal_error_ppm"] = prediction_error(
                    c, 15, 45, cov
                )
                row["red_peak_residual_combined_formal_error_ppm"] = float(
                    np.hypot(
                        row["regions_before"]["red_peak"]["formal_error_ppm"],
                        row["red_peak_template_formal_error_ppm"],
                    )
                )
                # Training deliberately excludes the line. A full-master planet
                # perturbation can nevertheless contaminate the sidebands.
                leakage, _, _, _ = fit_template(c, low, high, degree, "leakage")
                predictions[key + "_leakage"] = leakage
                peak = (center >= 15) & (center <= 45)
                row["master_signal_leakage_peak_prediction_ppm"] = float(
                    np.mean(leakage[peak]) * 1e6
                )
                if fixed_master:
                    direct, _, _, _ = fit_template(
                        c, low, high, degree, "direct_leakage"
                    )
                    predictions[key + "_direct_leakage"] = direct
                    row["fixed_master_peak_prediction_ppm"] = float(
                        np.mean(direct[peak]) * 1e6
                    )
                    row["master_feedback_peak_prediction_ppm"] = float(
                        np.mean(leakage[peak] - direct[peak]) * 1e6
                    )
                records.append(row)

    # Independent-order prediction uses the other order's off-line-only fit.
    cross = []
    for training_order, validation_order in [
        ("order0", "order1"),
        ("order1", "order0"),
    ]:
        for degree in [0, 1]:
            p, _, covariance, _ = fit_template(
                all_coadds[training_order], 120, 200, degree
            )
            c = all_coadds[validation_order]
            cross.append(
                {
                    "training_order": training_order,
                    "validation_order": validation_order,
                    "degree": degree,
                    "holdout_80_120": region(c, 80, 120, p),
                    "line_15_45": region(c, 15, 45, p),
                    "holdout_template_formal_error_ppm": prediction_error(
                        c, 80, 120, covariance
                    ),
                    "line_template_formal_error_ppm": prediction_error(
                        c, 15, 45, covariance
                    ),
                }
            )

    master_control = {"available": fixed_master}
    if fixed_master:
        master_control["definition"] = (
            "Full=N_A/M_A-N_0/M_0; direct=N_A/M_0-N_0/M_0; master feedback=full-direct. "
            "The frozen-master array is a control, not the observation model."
        )
        c = all_coadds["both"]
        band = (center >= 120) & (center <= 200) & np.isfinite(c["variance"])
        weight = 1 / c["variance"][band]
        for field in ["leakage", "direct_leakage", "master_leakage"]:
            master_control[field] = {
                "sideband_weighted_mean_ppm": float(
                    np.sum(c[field][band] * weight) / weight.sum() * 1e6
                ),
                "sideband_max_abs_ppm": float(np.max(np.abs(c[field][band])) * 1e6),
            }
        row = next(row for row in records if row["key"] == "both_120_200_degree1")
        master_control["full_extrapolated_peak_ppm"] = row[
            "master_signal_leakage_peak_prediction_ppm"
        ]
        master_control["direct_extrapolated_peak_ppm"] = row[
            "fixed_master_peak_prediction_ppm"
        ]
        master_control["master_feedback_extrapolated_peak_ppm"] = row[
            "master_feedback_peak_prediction_ppm"
        ]
    sources = [Path(__file__), model_path, *data_paths]
    report = {
        "scope": "Matched-phase off-line-only diagnostic, fixed atmosphere; no correction is adopted.",
        "overlap_phase": overlap,
        "stellar_velocity_domain_kms": [110, 390],
        "line_exclusion": "All nuisance fits exclude planetary velocities below +80 km/s; red peak 15..45 is never fitted.",
        "method": "Subtract the modeled full-night-master residual, fit constant/linear night difference only in red off-line channels, then predict held-out channels and line. Both orders use the same native-bin rule.",
        "records": records,
        "independent_order_prediction": cross,
        "fixed_master_control": master_control,
        "limitations": [
            "All line predictions are extrapolations; smoothness under the line is not identified by off-line data.",
            "Archived errors omit shared-master and extraction covariance. Formal errors are not calibrated significance.",
            "The fixed planetary model is itself conditional; off-line channels retain its master self-subtraction.",
            "Phase distribution within the shared interval and native wavelength grids differ between nights.",
            "No nightly correction or physical variability claim follows from this sensitivity screen.",
        ],
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    (output / "review.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    saved = {"velocity_kms": center, "exposure_count": exposure_count}
    for k, c in all_coadds.items():
        saved.update({f"{k}_{name}": v for name, v in c.items()})
    saved.update(predictions)
    np.savez_compressed(output / "arrays.npz", **saved)
    plot(output)
    print(
        json.dumps({"completed": True, "records": len(records), "output": str(output)})
    )


def plot(output: Path) -> None:
    """Replot only derived binned arrays; no input spectra are required."""
    with np.load(output / "arrays.npz") as archive:
        a = dict(archive)
    center = a["velocity_kms"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].errorbar(
        center,
        a["both_residual"] * 1e6,
        yerr=np.sqrt(a["both_variance"]) * 1e6,
        fmt=".",
        color="0.45",
        ms=3,
        elinewidth=0.5,
        label="Night difference minus fixed processed model",
    )
    for degree, color in [(0, "C0"), (1, "C1")]:
        axes[0].plot(
            center,
            a[f"both_120_200_degree{degree}"] * 1e6,
            color=color,
            label=f"Degree {degree}: trained at +120..200 km/s",
        )
    axes[1].plot(
        center,
        a["both_leakage"] * 1e6,
        color="C3",
        label="Processed difference from test amplitudes [1.5, 1.0, 0.8]",
    )
    if "both_direct_leakage" in a:
        axes[1].plot(
            center,
            a["both_direct_leakage"] * 1e6,
            color="C2",
            ls=":",
            label="Same contrast change, reference master held fixed",
        )
    axes[1].plot(
        center,
        a["both_120_200_degree1_leakage"] * 1e6,
        color="C1",
        ls="--",
        label="Linear extrapolation of this signal from training band",
    )
    for ax in axes:
        ax.axvspan(
            -60, 60, color="0.7", alpha=0.2, label="Line: excluded from training"
        )
        ax.axvspan(120, 200, color="C2", alpha=0.1, label="Training band")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(-30, 230)
        ax.grid(alpha=0.2)
        ax.set_ylabel("Aug 8 - Aug 20 (ppm)")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylim(-2500, 1800)
    axes[-1].set_xlabel("Planet-rest velocity under fixed circular reduction (km/s)")
    axes[0].set_title("An off-line extrapolation is not an identified correction")
    axes[1].set_title(
        "Changing intrinsic contrast also changes off-line channels through the master"
    )
    fig.tight_layout()
    fig.savefig(output / "off_line_diagnostic.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs=2, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
    else:
        if args.data is None or args.model is None:
            parser.error("--data and --model are required unless --plot-only is used")
        run(args.data, args.model, args.output)


if __name__ == "__main__":
    main()
