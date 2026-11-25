"""Plot a compact, source-qualified fit export without rerunning the model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    data_path = args.manifest.parent / config["data"]
    with np.load(data_path) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    velocity = data["observed_velocity_kms"]
    observed, error = data["observed"], data["supplied_error"]
    grid = data["model_velocity_kms"]
    assert len(velocity) == len(observed) == len(error)
    assert np.all(error > 0) and np.all(np.diff(grid) > 0)
    assert grid[0] <= velocity.min() <= velocity.max() <= grid[-1]
    assert all(np.isfinite(value).all() for value in data.values())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, (axis, residual) = plt.subplots(
        2, 1, figsize=(9.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    axis.errorbar(velocity, observed * 1e6, yerr=error * 1e6, fmt=".",
                  color="0.25", ecolor="0.7", elinewidth=0.8, markersize=5,
                  label="Zenodo observations (supplied errors)", zorder=4)
    metrics = {}
    for curve in config["curves"]:
        model = data[curve["key"]]
        assert model.shape == grid.shape
        axis.plot(grid, model * 1e6, color=curve["color"],
                  linestyle=curve.get("linestyle", "-"),
                  linewidth=curve.get("linewidth", 1.9), label=curve["label"])
        difference = (np.interp(velocity, grid, model) - observed) / error
        metrics[curve["key"]] = {
            "chi2_diagonal_supplied_errors": float(difference @ difference),
            "weighted_rms": float(np.sqrt(np.mean(difference**2))),
        }
        if curve["key"] == config["residual_curve"]:
            residual.plot(velocity, difference, ".", color=curve["color"], markersize=4)
    axis.axhline(0, color="0.8", linewidth=0.8, zorder=0)
    residual.axhspan(-1, 1, color="0.94", zorder=0)
    residual.axhline(0, color="0.55", linewidth=0.8)
    axis.set_ylabel("Processed residual flux [ppm]")
    residual.set_ylabel("Model − data\n/ supplied error")
    residual.set_xlabel("Planet-frame velocity [km s$^{-1}$]")
    residual.set_xlim(velocity.min() - 2, velocity.max() + 2)
    axis.set_title(config["title"], loc="left", fontsize=13)
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.13)
    if config.get("note"):
        fig.text(0.12, 0.063, config["note"], fontsize=8, color="0.35")
    fig.text(0.12, 0.025,
             "Observations: Zhang (2026), Zenodo 20707255, CC BY 4.0. "
             "Models use the declared phase-processing proxy.", fontsize=8, color="0.35")
    fig.savefig(args.output / "best_effort_spectrum.png", dpi=170)
    fig.savefig(args.output / "best_effort_spectrum.pdf")
    plt.close(fig)
    (args.output / "plotted_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
