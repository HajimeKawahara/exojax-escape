"""Plot the actual cell values used by a spectrum, from a compact export."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    radius = data["radius_rp"]
    edges = data["edges_rp"]
    prescribed = data["prescribed_lower_mask"]
    assert radius.size == 8 and edges.size == 9
    assert prescribed.dtype == bool and np.all(np.diff(radius) > 0)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharex=True)
    panels = [
        ([("temperature_k", "Temperature", "#c75b12")], "Temperature [K]", False),
        ([("radial_velocity_kms", "Tidal nonisothermal Parker wind", "#0072b2")],
         "Radial velocity [km s$^{-1}$]", True),
        ([("n_h_cm3", "H nuclei", "0.25"), ("electron_cm3", "Electrons", "#009e73")],
         "Number density [cm$^{-3}$]", True),
        ([("n2_cm3", "$n=2$ total", "#0072b2"), ("n3_cm3", "$n=3$ total", "#cc79a7")],
         "H level population [cm$^{-3}$]", True),
    ]
    for axis, (curves, label, log) in zip(axes.flat, panels):
        for index in np.flatnonzero(prescribed):
            axis.axvspan(edges[index], edges[index + 1], color="0.9", zorder=0)
        for key, name, color in curves:
            values = data[key]
            assert values.shape == radius.shape and np.isfinite(values).all()
            if log:
                assert np.all(values > 0)
            axis.plot(radius, values, "o-", color=color, markersize=4,
                      linewidth=1.4, label=name)
        if log:
            axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.legend(frameon=False, fontsize=9)
        axis.set_xlim(edges[0], edges[-1])
    for axis in axes[1]:
        axis.set_xlabel("Radius / planet radius")
    fig.suptitle("Atmosphere used by the current conditional spectrum fits", x=0.11,
                 ha="left", fontsize=13)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.89,
                        hspace=0.18, wspace=0.28)
    fig.text(0.11, 0.035, "Eight model cells; connecting lines are guides. "
             "Gray: prescribed lower atmosphere.\n"
             "Solar-composition proxy; mass-loss rate 10$^{13}$ g s$^{-1}$, "
             "planet mass 2.17 $M_{\\rm J}$.", fontsize=8, color="0.35")
    fig.savefig(args.output / "atmosphere_profiles.png", dpi=170)
    fig.savefig(args.output / "atmosphere_profiles.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
