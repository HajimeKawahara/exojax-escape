"""Collect the finite reservoir probes and verify their retained provenance."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(name):
    return json.loads((HERE / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    atomic = read("atomic/review.json")
    radiation = read("radiation/review.json")
    controls = read("radiation/controls.json")
    dynamics = read("dynamics/scales.json")
    dynamic_review = read("dynamics/review.json")
    geometry = read("geometry_budget.json")
    pins = {}
    for document, key in [
        (atomic, "source_input_sha256"),
        (radiation, "source_and_input_sha256"),
        (dynamics, "source_and_input_sha256"),
        (dynamic_review, "artifacts"),
        (geometry, "input_sha256"),
    ]:
        for path, expected in document[key].items():
            assert sha(path) == expected, path
            assert path not in pins or pins[path] == expected, path
            pins[path] = expected
    for path, expected in [
        (HERE / "geometry_budget.py", geometry["source_sha256"]),
        (HERE / "atomic/receiving_and_rates.npz", atomic["raw_array_sha256"]),
    ]:
        assert sha(path) == expected, path
        pins[str(path)] = expected

    local = atomic["fixed_bath_sensitivities"]["5"]
    keys = ["n2_cm3", "n3_cm3", "n2_factor", "n3_n2", "density_factor",
            "rest_center_source_ratio", "static_center_opacity_ratio",
            "halpha_common_up_down_scale"]
    extract = lambda d: {k: d[k] for k in keys}
    summary = {
        "scope": "Fixed five-level baths, opacity integrals and dimensional scales; no new full atmosphere or observational fit.",
        "density_matches": [
            dict(radius_rp=c["radius_rp"], **extract(
                atomic["fixed_bath_sensitivities"][str(c["cell"])]["density_matching_free_n2"]))
            for c in atomic["cells"]
        ],
        "cell5_baseline": extract(local["baseline"]),
        "cell5_halpha_sensitivities": [extract(x) for x in local["density_target_halpha_escape_sensitivity"]],
        "cell5_pump_only": {k: local["pump_matching_free_n2"][k] for k in
                            ["effective_lya_pump_factor", "ionization_budget_warning"]},
        "cell5_sink_probes": {
            "zero_selected_PI_n2_factor": local["suppressed_photoionization_bound"][-1]["n2_factor"],
            "zero_Lya_spontaneous_escape_n2_factor": local["suppressed_lya_escape"][-1]["n2_factor"],
        },
        "cell5_D20p19_dynamics": {
            k: v[5] for k, v in dynamics["density_scalings"]["20.19"].items()
            if k in ["pressure_dyn_cm2", "speed_same_spherical_mdot_kms",
                     "radial_crossing_same_mdot_hours", "mdot_same_speed_g_s",
                     "gas_pressure_local_field_G", "gas_pressure_surface_equatorial_dipole_G"]
        },
        "shielding": radiation["pure_n2_screen"],
        "geometry_counterexample": {k: geometry[k] for k in
            ["projected_area_ratio", "source_ratio", "approximate_area_times_source_ratio",
             "thick_full_shell_dim_condition_S_over_Iinner_lt"]},
        "ranked_hypothesis": "A force-supported dense slow sector with continuous recombination and self-consistent non-LTE radiation is motivated, not established.",
        "physical_support": "unknown",
        "observed_residual_improvement": None,
    }
    report = (ROOT / "doc/halpha_outer_reservoir.tex").read_text()
    checks = []
    for row in summary["density_matches"]:
        expected = (f'{row["radius_rp"]:.4f} & {row["n2_factor"]:.2f} & '
                    f'{row["density_factor"]:.2f} & {row["rest_center_source_ratio"]:.3f}')
        assert expected in report, expected
        checks.append(expected)
    for value, precision in [
        (summary["cell5_halpha_sensitivities"][0]["rest_center_source_ratio"], 3),
        (summary["cell5_halpha_sensitivities"][1]["rest_center_source_ratio"], 3),
        (summary["cell5_D20p19_dynamics"]["speed_same_spherical_mdot_kms"], 3),
        (summary["cell5_D20p19_dynamics"]["gas_pressure_surface_equatorial_dipole_G"], 2),
        *[(v, 3) for v in summary["geometry_counterexample"].values()],
    ]:
        expected = f"{value:.{precision}f}"
        assert expected in report, expected
        checks.append(expected)
    finite_checks = {
        "saved_state_identity": atomic["checks"]["native_state_identity"],
        "source_input_identity": atomic["unchanged_source_inputs"] and radiation["unchanged_inputs"],
        "receiving_balance_below_1e_10": atomic["checks"]["receiving_corrected_max_level_balance"] < 1e-10,
        "collision_parity_below_1e_12": atomic["checks"]["collision_rebuild_max_relative_error"] < 1e-12,
        "radiation_controls": all(controls["checks"].values()),
        "mass_flux_parity_below_1e_12": dynamic_review["checks"]["mass_flux_relative_spread"] < 1e-12,
        "report_table_and_key_values": bool(checks),
    }
    assert all(finite_checks.values()), finite_checks
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    validation = {
        "scope": "Input identity and finite diagnostic controls only; no physical or observational validation.",
        "verified_pins": pins,
        "pin_count": len(pins),
        "checks": finite_checks,
        "atomic_checks": atomic["checks"],
        "report_value_checks": checks,
        "report_source_sha256": sha(ROOT / "doc/halpha_outer_reservoir.tex"),
        "summary_sha256": sha(HERE / "summary.json"),
        "assembler_sha256": sha(__file__),
    }
    (HERE / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"pins": len(pins), "finite_checks": finite_checks,
                      "physical_support": "unknown"}))


if __name__ == "__main__":
    main()
