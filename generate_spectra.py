"""Generate the four illustrative H-alpha spectra and optional convergence checks."""

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

from exojax.utils.constants import hcperk
from hydrogen_forward import HydrogenForwardModel, SYNCHRONOUS_ROTATION_KMS

CASES = {
    "Boltzmann n3/n2, static": {"departure_floor": 1.0, "outflow_kms": 0.0},
    "Reduced upper population, static": {"outflow_kms": 0.0},
    "Reduced upper population + outflow": {"outflow_kms": 10.0},
    "Outflow + rotation + zonal wind": {
        "outflow_kms": 10.0, "rotation_kms": SYNCHRONOUS_ROTATION_KMS, "zonal_kms": 11.6,
    },
}


def plot_spectra(model, spectra, phase, destination):
    """Plot prescribed temperatures and the resulting line excess."""
    b_ratio = 0.08 + 0.92 * np.asarray(model.departure_transition)
    energy_k = hcperk * model.line_center_cm
    excitation_k = energy_k / (energy_k / np.asarray(model.temperature_k) - np.log(b_ratio))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(model.radius_rp, model.temperature_k, label="Kinetic temperature")
    axes[0].plot(model.radius_rp, excitation_k, label="Excitation temperature")
    axes[0].axhline(4500, color="0.5", ls=":", label="Lower boundary")
    axes[0].set(xlabel="Radius / Rp", ylabel="Temperature [K]",
                title=f"Prescribed atmosphere ({len(model.radius_rp)} layers)")
    axes[0].legend(fontsize=8)
    for label, spectrum in spectra.items():
        axes[1].plot(model.velocity_kms, 100 * spectrum, label=label)
    axes[1].axhline(0, color="0.5", ls=":")
    axes[1].set(xlim=(-70, 70), xlabel="Velocity from H-alpha rest center [km/s]",
                ylabel="Planet line excess / blackbody star [%]",
                title=f"Dayside model at phase {phase:g}")
    axes[1].legend(fontsize=7)
    fig.suptitle("Prescribed hydrogen escape model; no fit to observations")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    parser.add_argument("--phase", type=float, default=0.375)
    parser.add_argument("--nlayer", type=int, default=64)
    parser.add_argument("--check-convergence", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    model = HydrogenForwardModel(nlayer=args.nlayer, phase=args.phase)
    spectra = {name: np.asarray(model.spectrum(**parameters)) for name, parameters in CASES.items()}
    if not all(np.isfinite(spectrum).all() for spectrum in spectra.values()):
        raise RuntimeError("The generated spectra contain non-finite values.")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["velocity_kms", *spectra])
        writer.writerows(zip(model.velocity_kms, *spectra.values()))
    report = {
        "phase": args.phase, "nlayer": args.nlayer, "nlon": 24, "nlat": 12,
        "nnu": len(model.wavenumber_cm), "n2_column_cm-2": 3e14,
        "departure_floor": 0.08, "rest_wavelength_angstrom": 1e8 / model.line_center_cm,
        "cases": CASES,
    }
    if args.check_convergence:
        reference = spectra["Reduced upper population + outflow"]
        finer_layers = HydrogenForwardModel(nlayer=2 * args.nlayer, phase=args.phase)
        finer_disk = HydrogenForwardModel(nlayer=args.nlayer, nlon=48, nlat=24, phase=args.phase)
        report["max_absolute_contrast_change"] = {
            "double_layers": float(np.max(np.abs(np.asarray(finer_layers.spectrum()) - reference))),
            "double_disk_resolution": float(np.max(np.abs(np.asarray(finer_disk.spectrum()) - reference))),
        }
    (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_spectra(model, spectra, args.phase, args.output / "model.png")
    print(json.dumps(report, indent=2))
    print(f"Wrote spectra.csv, parameters.json, and model.png to {args.output}")


if __name__ == "__main__":
    main()
