"""Plot a measured infrared rate discontinuity and a local transfer control."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parent
a = np.load(root / "plot_arrays.npz")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
old, new = "#cc6677", "#4477aa"
for section in (slice(0, 3), slice(3, 6)):
    axes[0].plot(a["offset"][section], a["original"][section], "o-", color=old)
axes[0].plot(a["offset"], a["formal"], "o-", color=new, label="Three-line transfer control")
axes[0].plot([], [], "o-", color=old, label="Inherited escape closure")
axes[0].axvline(0, color=".6", lw=.8, linestyle=":")
axes[0].set_xscale("symlog", linthresh=1e-8)
axes[0].set_xticks([-1e-4, -1e-6, -1e-8, 0, 1e-8, 1e-6, 1e-4])
axes[0].set_xlim(-2e-4, 2e-4)
axes[0].set(xlabel=r"Step fraction from the opacity crossing, $\alpha-\alpha_*$",
            ylabel=r"Outer-cell $n=14$ equation residual",
            title="A rate switch creates a finite jump")
axes[0].legend(fontsize=8, loc="lower left")
axes[1].loglog(a["separation"], a["original_difference"], "o-", color=old,
               label="Inherited closure: jump remains")
axes[1].loglog(a["separation"], a["formal_difference"], "o-", color=new,
               label="Transfer control: difference shrinks")
axes[1].invert_xaxis()
axes[1].set(xlabel=r"Separation of paired step fractions, $\Delta\alpha$",
            ylabel="Maximum change in the upper H residuals",
            title="Bring the two input states closer")
axes[1].legend(fontsize=8, loc="upper right", bbox_to_anchor=(.99, .82))
for ax in axes:
    ax.grid(alpha=.15)
fig.supxlabel(
    "Same saved eight-cell Newton direction; only three fixed IR-line rates are replaced.\n"
    "All 18 H equations retained in each upper cell; prescribed lower cells excluded.\n"
    "Fixed-state continuity test, not a solved thermal atmosphere or a validated scattering closure.",
    fontsize=8)
fig.savefig(root / "infrared_continuity.png", dpi=180)
fig.savefig(root / "infrared_continuity.pdf")
