"""Redraw the recorded boundary-control figure from safe derived plot arrays.

Run `python plot.py`; requires NumPy and Matplotlib. This does not re-solve the
wind or atmosphere. provenance.json identifies the original calculations.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output = Path(__file__).resolve().parent
saved = np.load(output / "plot_arrays.npz")
radius = saved["radius_plot_rp"]
labels = ["Flat beyond last cell center", "Pressure extended to the 3 Rp edge"]
colors = ["#425D91", "#C15B32"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), layout="constrained")
for curve, label, color, advection in zip(saved["effective_temperature_curves_k"], labels, colors, saved["advection_erg_cm3_s"]):
    axes[0].plot(radius, np.asarray(curve)/1e4, label=label, color=color, lw=2)
    axes[1].plot(saved["cell_radius_rp"], advection/1e-7,
                 "o-", color=color, lw=1.8, ms=5)
axes[0].scatter(saved["sample_radius_rp"], saved["sample_effective_temperature_k"]/1e4,
                color="black", s=20, zorder=4, label="Same input values (base + 5 cells)")
axes[0].scatter([3.], [saved["pressure_edge_effective_temperature_k"]/1e4], facecolors="none", edgecolors=colors[1],
                s=60, zorder=4)
axes[0].axvline(2.15, color="0.65", ls=":", lw=1)
axes[0].axvline(3., color="0.65", ls=":", lw=1)
axes[0].set(xlabel="Radius / Rp", ylabel="Pressure variable T / mu  [10,000 K]")
axes[0].legend(fontsize=8, loc="upper left")
axes[1].axhline(0., color="0.55", lw=1)
axes[1].set(xlabel="Radius / Rp", ylabel=r"Advective heating  [$10^{-7}$ erg cm$^{-3}$ s$^{-1}$]")
axes[1].annotate("Endpoint rule forces zero", xy=(2.15, 0.), xytext=(1.45, .2),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color=colors[0]))
axes[1].annotate("Pressure slope allows cooling", xy=(2.15, saved["advection_erg_cm3_s"][1, -1]/1e-7),
                 xytext=(1.25, -.9), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=colors[1]))
fig.suptitle("Outer pressure boundary: conditional wind sensitivity", fontsize=13)
fig.supxlabel("Five-cell temperatures and composition held fixed; energy balance is not re-solved.",
              fontsize=9)
for ax in axes:
    ax.grid(alpha=.16)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(output/"outer_pressure_control.png", dpi=180)
fig.savefig(output/"outer_pressure_control.pdf")
