"""Compare hand-selected parameters and a constant residual baseline.

No optimizer or posterior sampler is used. The baseline is the weighted mean
of observed residuals at |v| > 50 km/s, assumed to be outside the line. The
weighted residual RMS is a diagnostic, not a calibrated goodness-of-fit test:
the supplied uncertainties may describe correlated, processed data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_spectra import load_observations
from hydrogen_forward import HydrogenForwardModel, SYNCHRONOUS_ROTATION_KMS

ORIGINAL_PARAMETERS = {
    "n2_column": 3e14, "departure_floor": 0.08, "outflow_kms": 10.0,
    "rotation_kms": SYNCHRONOUS_ROTATION_KMS, "zonal_kms": 11.6,
}
MANUAL_PARAMETERS = {
    "n2_column": 1.5e14, "departure_floor": 0.03, "outflow_kms": 8.0,
    "rotation_kms": SYNCHRONOUS_ROTATION_KMS, "zonal_kms": 6.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/manual"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    original_model = HydrogenForwardModel()
    manual_model = HydrogenForwardModel(temperature_top_k=12000.0)
    velocity, observed, error = load_observations(original_model.line_center_cm)
    baseline_checks = {}
    for cutoff in (45, 50, 55, 60, 65):
        outside = np.abs(velocity) > cutoff
        weights = 1 / error[outside]**2
        baseline_checks[str(cutoff)] = float(np.average(observed[outside], weights=weights))
    baseline = baseline_checks["50"]
    original = np.asarray(original_model.spectrum(**ORIGINAL_PARAMETERS))
    adjusted = np.asarray(manual_model.spectrum(**MANUAL_PARAMETERS))
    spectra = {
        "Original example": original,
        "Original + baseline": original + baseline,
        "Manual + baseline": adjusted + baseline,
    }
    residuals = {
        name: (observed - np.interp(
            velocity, original_model.velocity_kms[::-1], spectrum[::-1]
        )) / error for name, spectrum in spectra.items()
    }
    report = {
        "model_phase": 0.375, "observation_phase_range": [0.30, 0.45],
        "original": {"temperature_top_k": original_model.temperature_top_k, **ORIGINAL_PARAMETERS},
        "manual": {"temperature_top_k": manual_model.temperature_top_k, **MANUAL_PARAMETERS},
        "baseline_fraction": baseline, "baseline_cutoff_kms": 50,
        "baseline_by_cutoff_fraction": baseline_checks,
        "weighted_residual_rms": {
            name: float(np.sqrt(np.mean(residual**2))) for name, residual in residuals.items()
        },
        "note": "Hand-selected atmosphere; diagnostic RMS only, no optimizer or posterior sampling.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["velocity_kms", *spectra])
        writer.writerows(zip(original_model.velocity_kms, *spectra.values()))

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].errorbar(velocity, 100 * observed, yerr=100 * error, fmt=".",
                     color="black", markersize=4, elinewidth=0.7, alpha=0.7,
                     label="Keck/KPF observation", zorder=4)
    styles = [("C0", "-"), ("C1", "--"), ("C3", "-")]
    for (name, spectrum), (color, linestyle) in zip(spectra.items(), styles):
        axes[0].plot(original_model.velocity_kms, 100 * spectrum,
                     color=color, ls=linestyle, label=name)
        axes[1].plot(velocity, residuals[name], color=color, ls=linestyle,
                     marker=".", markersize=3, alpha=0.8)
    axes[0].axhline(100 * baseline, color="0.5", ls=":", lw=1)
    for ax in axes:
        ax.axvspan(-105, -50, color="0.5", alpha=0.06)
        ax.axvspan(50, 105, color="0.5", alpha=0.06)
    axes[0].set(ylabel="Line contrast [%]", title=(
        f"Model phase 0.375; data phases 0.30-0.45; constant baseline {100 * baseline:.4f}%"
    ))
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].axhline(0, color="0.5", ls=":")
    axes[1].axhspan(-1, 1, color="0.5", alpha=0.1)
    axes[1].set(xlim=(-105, 105), ylabel="Residual / error",
                xlabel="Velocity from H-alpha rest center [km/s]")
    fig.suptitle("Manual comparison with a constant residual baseline")
    fig.supxlabel(
        "Data: Yapeng Zhang (2026), doi:10.5281/zenodo.20707255; "
        "creativecommons.org/licenses/by/4.0 (CC BY 4.0)", fontsize=8,
    )
    fig.savefig(args.output / "manual_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f"Wrote spectra.csv, parameters.json, and manual_comparison.png to {args.output}")


if __name__ == "__main__":
    main()
