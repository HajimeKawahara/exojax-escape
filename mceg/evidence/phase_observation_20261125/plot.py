"""Replot only the compact derived evidence arrays; no raw inputs required."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PHASE_EDGES=np.array([.30,.35,.40,.45])

def plot_arrays(path, output):
    """Plot compact derived arrays without the source spectra or HDF5 files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(path) as source:
        arrays = dict(source)
    velocity, weights = arrays["velocity_kms"], arrays["weight"]
    total = weights.sum(axis=1)

    def combined(value):
        return np.divide(np.sum(np.nan_to_num(value) * weights, axis=1), total,
                         out=np.full_like(total, np.nan), where=total > 0)

    data = combined(arrays["data"])
    error = np.divide(1., np.sqrt(total), out=np.full_like(total, np.nan), where=total > 0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [2, 1]})
    for phase_bin in range(3):
        ax = axes[0, phase_bin]
        ax.errorbar(velocity, 1e6 * data[phase_bin], yerr=1e6 * error[phase_bin], fmt=".",
                    color="0.4", ms=3, elinewidth=.5, label="Re-binned archived residuals")
        for name, style, color, label in [
            ("common_rv", "--", "0.5", "One RV, fixed strength"),
            ("phase_rv", "-", "C0", "Per-phase RV"),
            ("phase_rv_amplitude", "-", "C1", "Per-phase RV + strength"),
        ]:
            ax.plot(velocity, 1e6 * combined(arrays[name])[phase_bin], ls=style, color=color, label=label)
        ax.set_title(f"Phase {PHASE_EDGES[phase_bin]:.2f}-{PHASE_EDGES[phase_bin + 1]:.2f}")
        for night in range(weights.shape[1]):
            take = weights[phase_bin, night] > 0
            delta = arrays["data"][phase_bin, night] - arrays["phase_rv_amplitude"][phase_bin, night]
            axes[1, phase_bin].errorbar(velocity[take], 1e6 * delta[take],
                                       yerr=1e6 / np.sqrt(weights[phase_bin, night, take]),
                                       fmt=".", ms=3, elinewidth=.5, label=f"Night {night + 1}")
        for row in axes[:, phase_bin]:
            row.axhline(0, color="k", lw=.4)
            row.grid(alpha=.2)
            row.set_xlim(-95, 95)
        axes[1, phase_bin].set_xlabel("Planet-rest velocity (km/s)")
    axes[0, 0].set_ylabel("Residual contrast (ppm)")
    axes[1, 0].set_ylabel("Data - per-phase fit\n(ppm)")
    axes[0, 0].legend(fontsize=7)
    axes[1, -1].legend(fontsize=8)
    fig.suptitle("Fixed atmosphere; fits retain night information and all exposures in each master")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def plot(output: Path) -> None:
    """Replot only derived binned arrays; no input spectra are required."""
    with np.load(output / "night_arrays.npz") as archive:
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

if __name__ == "__main__":
    here=Path(__file__).resolve().parent
    plot_arrays(here/"phase_arrays.npz", here/"phase_fit.png")
    plot(here)
