"""Plot saved physical and original free-model radial profiles; no new fit."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges_from_centers(radius, widths):
    left, right = radius - widths / 2, radius + widths / 2
    np.testing.assert_allclose(left[1:], right[:-1], rtol=0, atol=1e-14)
    return np.r_[left[0], right]


def main():
    folder = Path(__file__).resolve().parent
    source, path = Path(__file__).resolve(), folder / "sources/free_shape.json"
    before = {str(p): digest(p) for p in (source, path)}
    data = json.loads(path.read_text())
    physical, free = data["physical"], data["free"]["original"]
    radius = np.asarray(physical["radius_rp"])
    physical_edges = edges_from_centers(radius, np.asarray(physical["thickness_rp"]))
    free_radius = np.asarray(free["full_radius_rp"])
    free_edges = edges_from_centers(free_radius, np.full(len(free_radius), free["layer_thickness_rp"]))
    np.testing.assert_allclose(physical_edges[[0, -1]], [1., 3.], rtol=0, atol=1e-14)
    np.testing.assert_allclose(free_edges[[0, -1]], [1., 3.], rtol=0, atol=1e-14)
    values = ((physical["n2_cm3"], free["full_n2_cm3"]),
              (physical["rest_center_source"], free["full_source"]))
    assert all(np.all(np.isfinite(x)) and np.all(np.asarray(x) > 0) for pair in values for x in pair)
    index = int(np.argmin(abs(radius - 1.94)))
    free_n2 = float(np.interp(radius[index], free_radius, free["full_n2_cm3"]))
    np.testing.assert_allclose(free_n2, free["at_physical_centers"]["n2_cm3"][index], rtol=1e-13)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.65), sharex=True)
    for ax, (current, original) in zip(axes, values):
        ax.axvspan(1.725, 3., color="0.94", zorder=0)
        ax.stairs(current, physical_edges, baseline=None, color="C0", lw=2.1,
                  label="Physical atmosphere (8 cells)")
        ax.stairs(original, free_edges, baseline=None, color="C1", lw=1.8,
                  label="Original free model #1 (64 layers)")
        ax.axvline(1.725, color="0.4", lw=1., ls="--")
        ax.set_yscale("log")
        ax.set_xlim(1., 3.)
        ax.set_xlabel(r"Radius $r/R_p$")
        ax.grid(axis="y", which="major", color="0.88", lw=.5)
        ax.text(.975, .97, r"Outer region: $r>1.725R_p$", ha="right", va="top",
                transform=ax.transAxes, fontsize=9, color="0.3")
    axes[0].set_ylabel(r"Excited hydrogen $n_2$ (cm$^{-3}$)")
    axes[0].set_ylim(.06, 1.2e6)
    axes[1].set_ylabel(r"Local H$\alpha$ source (cgs per cm$^{-1}$)")
    axes[1].set_ylim(6e4, 2e7)
    axes[0].text(.05, .12,
        rf"At $r={radius[index]:.2f}R_p$:" + "\n" +
        rf"$n_2 \simeq {physical['n2_cm3'][index]:.0f}$ vs. ${free_n2:.0f}$ cm$^{{-3}}$" + "\n" +
        "(physical vs. free)", transform=axes[0].transAxes, fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=.95, pad=5))
    axes[1].legend(loc="upper right", bbox_to_anchor=(.99, .88), fontsize=8.5,
                   framealpha=.95)
    fig.suptitle("Radial profiles: physical atmosphere and original free model #1", fontsize=13, y=.98)
    fig.text(.5, .025,
        "Stored local source functions before global amplitude scaling; no new fit.\n"
        "Transfer geometries and atomic descriptions differ: this comparison does not isolate a cause.",
        ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94), w_pad=2.0)
    files = [folder / ("radial_structure." + extension) for extension in ("png", "pdf")]
    for output in files:
        fig.savefig(output, dpi=180, metadata={"Creator": "plot_structure.py"})
    plt.close(fig)
    unchanged = all(digest(Path(p)) == value for p, value in before.items())
    assert unchanged
    receipt = dict(scope=__doc__, source_and_input_sha256=before,
        output_sha256={p.name: digest(p) for p in files},
        unchanged_source_inputs=unchanged,
        checks=dict(contiguous_cell_edges=True, shared_radius_limits=[1., 3.],
            finite_positive_log_axes=True, physical_cells=len(radius), free_layers=len(free_radius),
            interpolated_free_annotation_matches_saved_value=True),
        annotation=dict(radius_rp=float(radius[index]), physical_n2_cm3=physical["n2_cm3"][index],
                        interpolated_free_n2_cm3=free_n2),
        source_definition="Physical source is summed branch eta divided by summed branch alpha at the nominal rest H-alpha center; free source is its unresolved transition. Both use intensity per wavenumber before the global fitted amplitude.",
        limitations=["This figure compares prescribed or solved local coefficients, not emergent fluxes in matched geometry.",
            "The 64-layer free density at the annotated physical center is interpolated; the plotted profiles themselves retain their cell edges.",
            "A radial-structure difference is not causal proof or a new equilibrium solution."],
        replay="MPLCONFIGDIR=/tmp/escape-halpha-structure-mpl python mceg/evidence/halpha_causes_20261127/plot_structure.py")
    (folder / "radial_structure_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
