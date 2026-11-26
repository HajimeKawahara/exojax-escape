"""Independent H-alpha quadrature checks at one saved coupled atmosphere.

Only the selected radiation group and its receiving continuum events are
re-evaluated. These controls do not solve a second atmospheric equilibrium.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import exojax

from escape.hydrogen_rates import LIGHT_SPEED_CGS as C, PLANCK_CGS as H
from halpha_equilibrium import build_model, digest, write_json

jax.config.update("jax_enable_x64", True)


def diagnose(model, boundary, physical):
    """Keep all background populations, temperatures and incident flux fixed."""
    group, atom = model.halpha_group, model.hydrogen.atom
    indices = model.halpha_indices
    common = float(jnp.mean(group.line_frequency_hz))*(1+np.linspace(-120., 120., 1201)/(C/1e5))
    state = {key: physical[key] for key in (
        "bound_cm3", "proton_cm3", "electron_cm3", "metal_ion_cm3", "metal_levels_cm3",
        "molecule_cm3", "helium_cm3", "atomic_lte_mask")}

    @jax.jit
    def call(temperature, state, global_mean):
        result = model._halpha_state(temperature, state, boundary["incident_mean_intensity"],
                                    0., boundary["halpha_intensity"])
        frequency, weight = group.frequency_hz, group.weight_hz
        background = jax.vmap(lambda row: jnp.interp(frequency, model.grid.frequency_hz, row))(global_mean)
        photon_weight = 4*jnp.pi*weight/(H*frequency)
        line_packet = result["additional_photon_flux_cm2_s"]
        background_packet = (result["continuum_transfer"].mean_intensity-background)*photon_weight
        line_mean = result["transfer"].mean_intensity
        direct_line_power = 4*jnp.pi*jnp.sum((result["line_opacity_cm1"]*line_mean[:, None, :]
                                             -result["line_emission"])*weight, axis=(1, 2))
        net = result["net_excitation_cm3_s"]
        thermal = result["redistribution_heat_erg_cm3_s"]
        gap = atom.excitation_energy_erg[atom.line_upper[indices]]-atom.excitation_energy_erg[atom.line_lower[indices]]
        excitation_power = jnp.sum(net*gap, axis=1)+thermal
        emission_power = 4*jnp.pi*jnp.sum(result["line_emission"]*weight, axis=(1, 2))
        lower = state["bound_cm3"][:, atom.line_lower[indices]]
        upper = state["bound_cm3"][:, atom.line_upper[indices]]
        metrics = dict(upward_s=result["pump_up_s"], downward_s=result["radiative_down_s"],
            net_excitation_cm3_s=net, n3_net_excitation_cm3_s=net.sum(axis=1),
            effective_transition_scale_cm3_s=lower*result["pump_up_s"]+upper*result["radiative_down_s"],
            redistribution_heat_erg_cm3_s=thermal, line_absorbed_power_erg_cm3_s=excitation_power,
            direct_line_power_erg_cm3_s=direct_line_power,
            line_energy_defect_over_emission=(direct_line_power-excitation_power)/emission_power,
            mean_on_common_frequency=jax.vmap(lambda row: jnp.interp(common, frequency, row))(line_mean),
            transfer_balance_error=result["transfer"].balance_error, converged=result["converged"])
        neutral = state["bound_cm3"].sum(axis=1)
        ff, _ = jax.vmap(lambda t, e, ions, n: model.free_free.coefficients(t, e, ions, n, frequency))(
            temperature, state["electron_cm3"], model._ion_charge_density(state), neutral)
        for label, packet in (("line", line_packet), ("background", background_packet)):
            received = model.hydrogen.photoionization_from_photon_flux(temperature,
                state["bound_cm3"], state["proton_cm3"], state["electron_cm3"], frequency, packet)
            metrics[label+"_hydrogen_photoionization_s"] = received["photoionization_s"]
            metrics[label+"_hydrogen_recombination_cm3_s"] = received["recombination_cm3_s"]
            metrics[label+"_hydrogen_kinetic_heat_erg_cm3_s"] = received["net_erg_cm3_s"]
            metrics[label+"_free_free_heat_erg_cm3_s"] = jnp.sum(ff*packet*H*frequency, axis=1)
        return metrics

    result = jax.tree.map(np.asarray, call(jnp.asarray(physical["temperature_k"]),
        jax.tree.map(jnp.asarray, state), jnp.asarray(physical["radiation"]["mean_intensity"])))
    result["frequency_edges_hz"] = np.asarray(group.frequency_edges_hz)
    result["common_frequency_hz"] = common
    return result


def compare(reference, refined):
    """Expose cancellation separately from positive-rate quadrature errors."""
    comparison = {}
    for key in ("upward_s", "downward_s", "mean_on_common_frequency"):
        relative = (refined[key]-reference[key])/reference[key]
        comparison[key+"_max_relative_change"] = float(np.max(np.abs(relative)))
    for key in ("net_excitation_cm3_s", "n3_net_excitation_cm3_s",
                "redistribution_heat_erg_cm3_s", "line_absorbed_power_erg_cm3_s",
                "line_hydrogen_kinetic_heat_erg_cm3_s", "background_hydrogen_kinetic_heat_erg_cm3_s",
                "line_free_free_heat_erg_cm3_s", "background_free_free_heat_erg_cm3_s"):
        delta = refined[key]-reference[key]
        relative = np.divide(delta, np.abs(reference[key]), out=np.zeros_like(delta), where=reference[key] != 0)
        comparison[key] = dict(absolute_change=delta.tolist(),
            relative_to_absolute_reference=relative.tolist(),
            zero_reference=(reference[key] == 0).tolist())
    comparison["net_change_over_effective_transition_scale"] = (
        (refined["net_excitation_cm3_s"]-reference["net_excitation_cm3_s"])
        /reference["effective_transition_scale_cm3_s"]).tolist()
    return comparison


def run(args):
    if jax.default_backend() != "cpu":
        raise ValueError("Run these independent controls with JAX_PLATFORMS=cpu.")
    root = Path(__file__).resolve().parents[1]
    summary_path = args.physical.with_name("summary.json")
    configuration_path = args.physical.with_name("configuration.json")
    summary = json.loads(summary_path.read_text())
    configuration = json.loads(configuration_path.read_text())
    if not (summary.get("converged") and summary.get("inputs_unchanged")
            and summary["checks"]["passed"] and summary["stages"][-1]["blend"] == 1.
            and summary["residual_infinity"] < 1e-9
            and summary["repeated_residual_difference"] < 1e-10
            and summary["baseline_residual_difference"] < 1e-10):
        raise ValueError("The saved complete-root receipt must be converged with unchanged inputs.")
    if configuration["geometry"] != "spherical" or configuration["halpha_band"] != dict(
            wing_extent=30., angle_order=32, quadrature_order=8):
        raise ValueError("These controls require the declared spherical primary quadrature.")
    pins = dict(configuration["source_and_input_sha256"])
    for path in (args.recipe, args.stellar, args.nist):
        if str(path.resolve()) not in pins:
            raise ValueError(f"The requested model input is absent from the source run: {path}")
    dependency_path = root/"mceg/evidence/halpha_equilibrium_20261126/dependency_snapshot.json"
    dependency = json.loads(dependency_path.read_text())
    external = dependency["runtime_exojax"]
    if not Path(exojax.__file__).resolve().is_relative_to(Path(external["path"]).resolve()):
        raise ValueError("Use the archived ExoJAX source tree recorded by the source run.")
    pins.update({str(Path(external["path"])/name): value
                 for name, value in external["python_source_sha256"].items()})
    pins.update({str(root/name): value for name, value in dependency["atomic_tables"].items()})
    if any(digest(path) != value for path, value in pins.items()):
        raise ValueError("The saved root's source/input hashes no longer match.")
    with args.physical.open("rb") as stream:
        physical = pickle.load(stream)
    if not bool(physical["valid"]) or float(physical["halpha"]["blend"]) != 1.:
        raise ValueError("Supply a valid atmosphere with complete H-alpha replacement.")
    paths = [args.physical, summary_path, configuration_path, dependency_path, args.recipe, args.stellar, args.nist, Path(__file__).resolve(),
             root/"examples/halpha_equilibrium.py", root/"src/escape/solar.py",
             root/"src/escape/hydrogen_escape_probability.py", root/"src/escape/line_overlap.py",
             root/"src/escape/spherical_radiation.py"]
    pins.update({str(path): digest(path) for path in paths})
    cases, elapsed = {}, {}
    settings = dict(primary=(32, 8, 30.), angle=(64, 8, 30.),
                    frequency=(32, 16, 30.), wings=(32, 8, 60.))
    for label, (angle, frequency, wing) in settings.items():
        started = time.monotonic()
        print(f"Evaluating {label}: angle={angle}, frequency={frequency}, wings={wing}", flush=True)
        model, boundary, _ = build_model(args.recipe, args.stellar, args.nist, "spherical", wing, angle, frequency)
        cases[label] = diagnose(model, boundary, physical)
        elapsed[label] = time.monotonic()-started
        print(f"Finished {label}: {elapsed[label]:.2f} s", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output/"arrays.npz"
    np.savez_compressed(raw_path, **{f"{label}_{key}": value for label, values in cases.items()
                                    for key, value in values.items()})
    scalar_fields = ("upward_s", "downward_s", "net_excitation_cm3_s", "n3_net_excitation_cm3_s",
        "redistribution_heat_erg_cm3_s", "line_absorbed_power_erg_cm3_s", "line_energy_defect_over_emission",
        "line_hydrogen_kinetic_heat_erg_cm3_s", "background_hydrogen_kinetic_heat_erg_cm3_s",
        "line_free_free_heat_erg_cm3_s", "background_free_free_heat_erg_cm3_s")
    repeated = {key: float(np.max(np.abs(cases["primary"][key]-np.asarray(physical["halpha"][key]))
                    /np.maximum(np.abs(np.asarray(physical["halpha"][key])), np.finfo(float).tiny)))
                for key in ("net_excitation_cm3_s", "redistribution_heat_erg_cm3_s")}
    repeated_absolute = {key: float(np.max(np.abs(
        cases["primary"][key]-np.asarray(physical["halpha"][key])))) for key in repeated}
    comparisons = {key: compare(cases["primary"], value) for key, value in cases.items() if key != "primary"}
    review = dict(scope="Independent selected-group quadrature controls at fixed converged populations and temperature; no second SE/thermal root.",
        backend=jax.default_backend(), x64=jax.config.x64_enabled,
        jax_version=jax.__version__, exojax_path=exojax.__file__,
        source_and_input_sha256=pins, elapsed_seconds=elapsed,
        settings={key: dict(angle_order=value[0], quadrature_order=value[1], wing_extent=value[2])
                  for key, value in settings.items()},
        cases={label: {key: value[key].tolist() for key in scalar_fields} for label, value in cases.items()},
        primary_reproduction_relative_error=repeated,
        primary_reproduction_absolute_error=repeated_absolute, comparisons=comparisons,
        all_transfer_valid=bool(all(value["converged"] for value in cases.values())),
        maximum_line_energy_defect_over_emission=float(max(np.max(np.abs(value["line_energy_defect_over_emission"]))
                                                          for value in cases.values())),
        transfer_balance_errors={key: float(value["transfer_balance_error"]) for key, value in cases.items()},
        input_unchanged=all(digest(path) == value for path, value in pins.items()),
        raw_arrays=dict(path=str(raw_path), sha256=digest(raw_path)),
        limitations=["Integrated positive rates, net rates and kinetic heat have different conditioning; each is reported separately.",
            "Upward and downward rates have the local diagonal radiation response analytically eliminated. Their sum defines an effective SE transition scale, not the gross count of physical absorption/emission events.",
            "Common-frequency J uses linear interpolation and is a display comparison, not the integrated-rate convergence criterion.",
            "The background replacement includes spherical geometry, exact stellar samples versus continuum-bin reconstruction, and finite-grid interpolation/quadrature differences; it is not a pure geometry measurement.",
            "Doubling the H-alpha window also changes the extent of this background replacement; its heating contribution is separated.",
            "Hydrogen and free-free receiving channels are exposed here; the full model distributes the same packet to all retained species.",
            "A significant refined residual requires a newly converged full atmosphere before an observational comparison can claim that refinement."])
    args.review.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.review, review)
    print(json.dumps(dict(all_transfer_valid=review["all_transfer_valid"],
        maximum_line_energy_defect_over_emission=review["maximum_line_energy_defect_over_emission"],
        rate_changes={key: {quantity: value[quantity] for quantity in (
            "upward_s_max_relative_change", "downward_s_max_relative_change")} for key, value in comparisons.items()})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=Path, default=Path("/tmp/escape-halpha-equilibrium-final/complete_physical.pkl"))
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--stellar", type=Path, default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--nist", type=Path, default=Path("/tmp/escape-five-cell-root-evidence/nist_row.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/escape-halpha-equilibrium-controls"))
    parser.add_argument("--review", type=Path, default=Path("mceg/evidence/halpha_equilibrium_20261126/quadrature_review.json"))
    run(parser.parse_args())
