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

from exojax.utils.constants import c, hcperk
from hydrogen_forward import HydrogenForwardModel, PLANET_RADIUS_CM, SYNCHRONOUS_ROTATION_KMS
from parker import sonic_radius_cm

CASES = {
    "Boltzmann n3/n2, static": {"departure_floor": 1.0, "radial_wind": False},
    "Reduced upper population, static": {"radial_wind": False},
    "Reduced upper population + Parker wind": {"wind_temperature_k": 9000.0},
    "Parker wind + rotation + zonal wind": {
        "wind_temperature_k": 9000.0,
        "rotation_kms": SYNCHRONOUS_ROTATION_KMS, "zonal_kms": 11.6,
    },
}
OBSERVATION_PATH = (
    Path(__file__).resolve().parent / "data" / "zenodo_20707255"
    / "KELT9b_w6565_avg_p0.30-0.45.dat"
)


def load_observations(line_center_cm):
    """Read wavelength, fractional residual, and supplied uncertainty.

    The headerless column interpretation follows the data values and paper;
    see data/zenodo_20707255/NOTICE.txt. Use the model's rest center without
    an additional fitted shift, scaling, or rebinning.
    """
    wavelength, residual, uncertainty = np.loadtxt(OBSERVATION_PATH, unpack=True)
    velocity = c * (wavelength / (1e8 / line_center_cm) - 1)
    return velocity, residual, uncertainty


def plot_spectra(model, spectra, phase, destination):
    """Plot prescribed temperatures, line excess, and observed residuals."""
    b_ratio = 0.08 + 0.92 * np.asarray(model.departure_transition)
    energy_k = hcperk * model.line_center_cm
    excitation_k = energy_k / (energy_k / np.asarray(model.temperature_k) - np.log(b_ratio))
    fig = plt.figure(figsize=(11, 6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.3])
    temperature_ax = fig.add_subplot(grid[0, 0])
    velocity_ax = fig.add_subplot(grid[1, 0])
    spectrum_ax = fig.add_subplot(grid[:, 1])
    temperature_ax.plot(model.radius_rp, model.temperature_k, label="Kinetic temperature")
    temperature_ax.plot(model.radius_rp, excitation_k, label="Excitation temperature")
    temperature_ax.axhline(4500, color="0.5", ls=":", label="Lower boundary")
    temperature_ax.set(xlabel="Radius / Rp", ylabel="Temperature [K]",
                title=f"Prescribed atmosphere ({len(model.radius_rp)} layers)")
    temperature_ax.legend(fontsize=8)
    for wind_temperature in (9000.0, 12000.0):
        sonic_rp = float(sonic_radius_cm(wind_temperature, model.planet_mass_mj, 0.5)) / PLANET_RADIUS_CM
        velocity_ax.plot(
            model.radius_rp, model.radial_velocity(wind_temperature),
            label=f"Twind={wind_temperature:g} K; rs/Rp={sonic_rp:.2f}",
        )
    velocity_ax.set(xlabel="Radius / Rp", ylabel="Radial velocity [km/s]",
                    title=f"Parker wind: M={model.planet_mass_mj:g} Mj, mu=0.5")
    velocity_ax.legend(fontsize=8)
    for label, spectrum in spectra.items():
        spectrum_ax.plot(model.velocity_kms, 100 * spectrum, label=label)
    velocity, residual, uncertainty = load_observations(model.line_center_cm)
    spectrum_ax.errorbar(
        velocity, 100 * residual, yerr=100 * uncertainty, fmt=".", color="black",
        markersize=3, elinewidth=0.7, alpha=0.75, zorder=4,
        label="Keck/KPF data (phases 0.30-0.45)",
    )
    spectrum_ax.axhline(0, color="0.5", ls=":")
    spectrum_ax.set(xlim=(-105, 105), xlabel="Velocity from H-alpha rest center [km/s]",
                    ylabel="Line contrast [%]")
    spectrum_ax.set_title(
        f"Spectra: Twind=9000 K, model phase {phase:g}\nData phases 0.30-0.45", fontsize=11,
    )
    spectrum_ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Prescribed line-forming atmosphere with an isothermal Parker velocity")
    fig.supxlabel(
        "Data: Yapeng Zhang (2026), doi:10.5281/zenodo.20707255; "
        "creativecommons.org/licenses/by/4.0 (CC BY 4.0)", fontsize=8,
    )
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    parser.add_argument("--phase", type=float, default=0.375)
    parser.add_argument("--nlayer", type=int, default=64)
    parser.add_argument("--check-convergence", action="store_true")
    parser.add_argument("--wavelength-model", choices=("nist", "rydberg"), default="nist")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    model = HydrogenForwardModel(nlayer=args.nlayer, phase=args.phase, wavelength_model=args.wavelength_model)
    spectra = {name: np.asarray(model.spectrum(**parameters)) for name, parameters in CASES.items()}
    if not all(np.isfinite(spectrum).all() for spectrum in spectra.values()):
        raise RuntimeError("The generated spectra contain non-finite values.")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["velocity_kms", *spectra])
        writer.writerows(zip(model.velocity_kms, *spectra.values()))
    report = {
        "wavelength_model": args.wavelength_model,
        "phase": args.phase, "nlayer": args.nlayer, "nlon": 24, "nlat": 12,
        "nnu": len(model.wavenumber_cm), "n2_column_cm-2": 3e14,
        "departure_floor": 0.08, "rest_wavelength_angstrom": 1e8 / model.line_center_cm,
        "velocity_model": "isothermal transonic Parker wind (no tides)",
        "planet_mass_mj": model.planet_mass_mj, "mean_molecular_weight": 0.5,
        "temperature_top_k": model.temperature_top_k,
        "cases": CASES,
        "observation": {
            "doi": "10.5281/zenodo.20707255", "file": OBSERVATION_PATH.name,
            "license": "CC-BY-4.0", "phase_range": [0.30, 0.45],
        },
    }
    if args.check_convergence:
        reference = spectra["Reduced upper population + Parker wind"]
        finer_layers = HydrogenForwardModel(nlayer=2 * args.nlayer, phase=args.phase, wavelength_model=args.wavelength_model)
        finer_disk = HydrogenForwardModel(nlayer=args.nlayer, nlon=48, nlat=24, phase=args.phase, wavelength_model=args.wavelength_model)
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
