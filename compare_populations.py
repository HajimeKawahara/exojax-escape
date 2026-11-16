"""Compare prescribed and two-level populations at the same n=2 column.

The two rate-model cases are illustrative, not refits. Their constant neutral
excited fractions are chosen only to match the previous reference column;
neither those fractions nor the supplied mass-loss rates are measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.utils.constants import hcperk
from generate_spectra import load_observations
from hydrogen_forward import (
    HydrogenForwardModel, REFERENCE_COLUMN_CM2, SYNCHRONOUS_ROTATION_KMS,
)
from hydrogen_populations import COLLISION_STRENGTH_23

TEMPERATURE_TOP_K = 15000.0
BASELINE = -0.000147797378605
REFERENCE = {
    "n2_column": 8.654280631524622e13, "departure_floor": 0.005573349747,
    "wind_temperature_k": 13580.399904, "mean_molecular_weight": 0.5,
    "rotation_kms": SYNCHRONOUS_ROTATION_KMS, "zonal_kms": 19.625210378,
}
STYLE = {
    "prescribed": ("Prescribed reference", "C0", "--"),
    "rates_1e13": ("Rates: 1e13 g/s", "C1", "-"),
    "rates_1e14": ("Rates: 1e14 g/s", "C3", "-"),
}


def plot_comparison(model, profiles, spectra, data, residuals, destination):
    """Show the density/excitation changes together with observed spectra."""
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.45], "height_ratios": [2, 1.25]},
    )
    temperature_ax, spectrum_ax = axes[0]
    population_ax, residual_ax = axes[1]
    temperature_ax.plot(model.radius_rp, model.temperature_k, color="black",
                        lw=1.5, label="Kinetic temperature (all cases)")
    temperature_ax.axhline(4500, color="0.5", ls=":", label="Lower boundary")
    for name, profile in profiles.items():
        label, color, style = STYLE[name]
        excitation = hcperk * model.line_center_cm / np.log(
            float(model.adb.gupper[0] / model.adb.glower[0]) / profile["ratio"]
        )
        temperature_ax.plot(model.radius_rp, excitation, color=color, ls=style,
                            label=label + " excitation")
        population_ax.semilogy(model.radius_rp, profile["n2_cm3"], color=color,
                              ls=style, label=label)
        population_ax.semilogy(model.radius_rp, profile["n3_cm3"], color=color, ls=":")
        spectrum_ax.plot(model.velocity_kms, 100 * spectra[name], color=color,
                         ls=style, label=label)
        residual_ax.plot(data[0], residuals[name], color=color, ls=style, lw=1)
    temperature_ax.set(xlabel="Radius / Rp", ylabel="Temperature [K]",
                       title="Prescribed kinetic T and resulting excitation T")
    temperature_ax.legend(fontsize=7, loc="upper right")
    population_ax.set(xlabel="Radius / Rp", ylabel="Number density [cm$^{-3}$]",
                      title="n=2 curves; dotted curves show n=3")
    population_ax.legend(fontsize=7, loc="upper right")
    velocity, observed, error = data
    spectrum_ax.errorbar(velocity, 100 * observed, yerr=100 * error, fmt=".",
                         color="black", markersize=4, elinewidth=0.7, alpha=0.65,
                         label="Keck/KPF observations", zorder=5)
    spectrum_ax.axhline(100 * BASELINE, color="0.5", ls=":", lw=1)
    spectrum_ax.set(xlim=(-105, 105), ylabel="Line contrast [%]",
                    title="Same integrated n=2 column; rate cases are not refits")
    spectrum_ax.legend(fontsize=8, loc="upper left")
    residual_ax.axhline(0, color="0.5", lw=1)
    residual_ax.axhspan(-1, 1, color="0.5", alpha=0.1)
    residual_ax.set(xlim=(-105, 105), xlabel="Velocity from NIST H-alpha [km/s]",
                    ylabel="(Model - data) / error")
    fig.suptitle("Continuity and isolated two-level excitation: illustrative comparison", fontsize=14)
    fig.supxlabel(
        "Rates: xe=0.5, Twind=9,000 K, Upsilon=35.52, dilution=0.025; "
        "prescribed reference: Twind=13,580 K, mu=0.5.\n"
        "Model phase 0.375; data phases 0.30-0.45. "
        "Data: Yapeng Zhang (2026), doi:10.5281/zenodo.20707255; CC BY 4.0.",
        fontsize=8,
    )
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/populations"))
    parser.add_argument("--check-convergence", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    model = HydrogenForwardModel(temperature_top_k=TEMPERATURE_TOP_K)
    data = load_observations(model.line_center_cm)
    b = REFERENCE["departure_floor"] + (1 - REFERENCE["departure_floor"]) * model.departure_transition
    ratio = (model.adb.gupper[0] / model.adb.glower[0]
             * np.exp(-hcperk * model.line_center_cm / model.temperature_k) * b)
    lower = model.n2_reference * REFERENCE["n2_column"] / REFERENCE_COLUMN_CM2
    profiles = {"prescribed": {
        "n2_cm3": np.asarray(lower), "n3_cm3": np.asarray(lower * ratio),
        "ratio": np.asarray(ratio), "radial_velocity_kms": np.asarray(
            model.radial_velocity(REFERENCE["wind_temperature_k"], REFERENCE["mean_molecular_weight"])),
    }}
    spectra = {"prescribed": np.asarray(model.prescribed_spectrum(**REFERENCE)) + BASELINE}
    parameters = {"prescribed": dict(REFERENCE)}
    for mass_loss in (1e13, 1e14):
        name = f"rates_{mass_loss:.0e}".replace("+", "")
        inputs = dict(mass_loss_g_s=mass_loss, ionized_fraction=0.5,
                      wind_temperature_k=9000.0, collision_strength=COLLISION_STRENGTH_23,
                      radiation_dilution=0.025)
        unit = model.population_profiles(**inputs, excited_fraction=1.0)
        fraction = REFERENCE["n2_column"] / float(np.sum(unit["n2_cm3"] * model.thickness_cm))
        if not 0 <= fraction <= 1:
            raise ValueError("The requested reference column exceeds the neutral reservoir.")
        inputs["excited_fraction"] = fraction
        profiles[name] = {key: np.asarray(value) for key, value in model.population_profiles(**inputs).items()}
        inputs.update(rotation_kms=REFERENCE["rotation_kms"], zonal_kms=REFERENCE["zonal_kms"])
        parameters[name] = inputs
        spectra[name] = np.asarray(model.spectrum(**inputs)) + BASELINE
    residuals = {
        name: (np.interp(data[0], model.velocity_kms[::-1], flux[::-1]) - data[1]) / data[2]
        for name, flux in spectra.items()
    }
    report = {
        "note": "Illustrative rate cases, not refits or mass-loss/excitation measurements; RMS is diagnostic only.",
        "normalization": "Constant f23 selected separately for each rate case to match the prescribed N2 column.",
        "temperature_top_k": TEMPERATURE_TOP_K, "model_phase": 0.375,
        "observation_phase_range": [0.30, 0.45], "baseline_fraction": BASELINE,
        "grid": {"nlayer": 64, "nlon": 24, "nlat": 12, "nnu": 1201},
        "planet_mass_mj": model.planet_mass_mj,
        "wavelength_model": model.wavelength_model,
        "rest_wavelength_angstrom": 1e8 / model.line_center_cm,
        "parameters": parameters,
        "n2_column_cm2": {name: float(np.sum(p["n2_cm3"] * model.thickness_cm)) for name, p in profiles.items()},
        "residual_rms": {name: float(np.sqrt(np.mean(r**2))) for name, r in residuals.items()},
    }
    if args.check_convergence:
        report["convergence_note"] = "Both rate cases retain their f23; no column renormalization"
        report["max_absolute_contrast_change"] = {name: {} for name in ("rates_1e13", "rates_1e14")}
        for label, grids in (("double_layers", {"nlayer": 128}), ("double_disk", {"nlon": 48, "nlat": 24})):
            finer = HydrogenForwardModel(temperature_top_k=TEMPERATURE_TOP_K, **grids)
            for name in report["max_absolute_contrast_change"]:
                flux = np.asarray(finer.spectrum(**parameters[name])) + BASELINE
                report["max_absolute_contrast_change"][name][label] = float(np.max(np.abs(flux - spectra[name])))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(args.output / "spectra.csv", np.column_stack([model.velocity_kms, *spectra.values()]),
               delimiter=",", header="velocity_kms," + ",".join(spectra), comments="")
    columns = {"radius_rp": np.asarray(model.radius_rp), "temperature_k": np.asarray(model.temperature_k)}
    for name, profile in profiles.items():
        columns.update({name + "_" + key: value for key, value in profile.items()})
    np.savetxt(args.output / "profiles.csv", np.column_stack(list(columns.values())),
               delimiter=",", header=",".join(columns), comments="")
    plot_comparison(model, profiles, spectra, data, residuals, args.output / "compare_populations.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
