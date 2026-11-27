"""Collect existing observer receipts without rerendering or refitting anything."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bands(record, scope="all"):
    target = record["target"][scope]
    return None if target is None else [target[b]["residual_ppm"] for b in ("blue", "core", "red")]


def field_bands(record, field):
    return [record["target"]["all"][band][field + "_ppm"] for band in ("blue", "core", "red")]


def changes(before, after):
    return [None if a is None or b is None else a - b for a, b in zip(before, after)]


def collateral(reference, candidate):
    result = {}
    for product in ("common_complete", "band_complete"):
        before = reference["phase"][product]["residual_BCR_ppm"]
        after = candidate["phase"][product]["residual_BCR_ppm"]
        change = [changes(a, b) for a, b in zip(before, after)]
        outside = [abs(value) for index, row in enumerate(change) if index != 3
                   for value in row if value is not None]
        result[product] = dict(model_plus_GP_change_BCR_ppm_by_phase=change,
                               maximum_absolute_change_outside_target_phase_ppm=max(outside))
    return result


def main():
    root = Path(__file__).resolve().parent
    names = ["D1", "D1.2", "D2", "D5", "D10", "D20"]
    selection_paths = [root / name for name in ("column_selection.json", "upper_column_bracket_selection.json")]
    for selection_path in selection_paths:
        if selection_path.exists():
            selected = json.loads(selection_path.read_text()).get("selected_factor")
            if selected is not None:
                names.append(f"D{selected:g}")
    paths = [root / name / "review.json" for name in sorted(set(names), key=lambda name: float(name[1:]))]
    reference = json.loads(paths[0].read_text())
    rows, inputs, status = [], {str(Path(__file__)): digest(__file__)}, {}
    for selection_path in selection_paths:
        if selection_path.exists():
            inputs[str(selection_path)] = digest(selection_path)
    table = ["| Case | Pixels | RV/A | Q | Q gain vs PR13 | Q gain vs D1 | Blue | Core | Red | Blue − core |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for path in paths:
        name = path.parent.name
        if not path.exists():
            status[name] = "not_yet_evaluated"
            continue
        inputs[str(path)] = digest(path)
        report = json.loads(path.read_text())
        assessment_path = path.parent / "stage_assessment.json"
        assessment = json.loads(assessment_path.read_text()) if assessment_path.exists() else None
        classification = (assessment or {}).get("case_classification", "Previously accepted D1 control" if name == "D1"
            else "Predeclared observer density factor" if name in ("D2", "D5", "D10", "D20") else "Additional exploratory case")
        status[name] = assessment["status"] if assessment else (
            "previous_accepted_D1_control" if name == "D1" else "awaiting_stage_assessment")
        if assessment:
            inputs[str(assessment_path)] = digest(assessment_path)
            if assessment["observer_review_sha256"] != digest(path):
                raise ValueError(f"Observer receipt changed after assessment: {name}")
        for selection, cases in report["selections"].items():
            for mode, candidate in cases["candidate"].items():
                pr13 = cases["baseline"][mode]
                d1 = reference["selections"][selection]["candidate"][mode]
                residual = bands(candidate)
                row = dict(case=name, selection=selection, nuisance_mode=mode,
                    case_classification=classification,
                    status=status[name], Q_all_native_data=candidate["Q"],
                    Q_gain_vs_PR13=pr13["Q"]-candidate["Q"], Q_gain_vs_D1=d1["Q"]-candidate["Q"],
                    residual_BCR_ppm=residual, residual_blue_minus_core_ppm=residual[0]-residual[1],
                    observed_BCR_ppm=field_bands(candidate, "observed"),
                    planet_BCR_ppm=field_bands(candidate, "planet"), GP_BCR_ppm=field_bands(candidate, "gp"),
                    planet_change_vs_D1_BCR_ppm=changes(field_bands(candidate, "planet"), field_bands(d1, "planet")),
                    GP_change_vs_D1_BCR_ppm=changes(field_bands(candidate, "gp"), field_bands(d1, "gp")),
                    both_blue_core_absolute_residuals_reduce_vs_PR13=all(abs(residual[i]) < abs(bands(pr13)[i]) for i in (0, 1)),
                    both_blue_core_absolute_residuals_reduce_vs_D1=all(abs(residual[i]) < abs(bands(d1)[i]) for i in (0, 1)),
                    absolute_blue_minus_core_residual_reduction_vs_PR13_ppm=abs(bands(pr13)[0]-bands(pr13)[1])-abs(residual[0]-residual[1]),
                    absolute_blue_minus_core_residual_reduction_vs_D1_ppm=abs(bands(d1)[0]-bands(d1)[1])-abs(residual[0]-residual[1]),
                    model_plus_GP_change_vs_PR13_BCR_ppm=changes(bands(pr13), residual),
                    model_plus_GP_change_vs_D1_BCR_ppm=changes(bands(d1), residual),
                    group_Q_gain_vs_PR13=changes(pr13["group_Q"], candidate["group_Q"]),
                    group_Q_gain_vs_D1=changes(d1["group_Q"], candidate["group_Q"]),
                    target_by_order_residual_BCR_ppm={scope: bands(candidate, scope) for scope in
                        ("night0_order0", "night0_order1", "night1_order0", "night1_order1")},
                    collateral_vs_PR13=collateral(pr13, candidate), collateral_vs_D1=collateral(d1, candidate),
                    parameters=candidate["parameters"], nuisance_success=candidate["nuisance_success"])
                if mode == "profiled" and assessment:
                    fit = assessment["selections"][selection]
                    row["nuisance_bound_distances"] = fit["distance_to_lower_and_upper_bounds"]
                    row["nuisance_within_one_percent_of_bound"] = fit["within_one_percent_of_any_bound"]
                rows.append(row)
                marker = "*" if "exploratory" in row["case_classification"].lower() else ""
                table.append(f"| {name}{marker} | {selection} | {mode} | {candidate['Q']:.3f} | "
                    f"{row['Q_gain_vs_PR13']:+.3f} | {row['Q_gain_vs_D1']:+.3f} | "
                    + " | ".join(f"{value:+.1f}" for value in residual + [row['residual_blue_minus_core_ppm']]) + " |")
    result = dict(scope="Same 12 complete August-8 exposures, phase 0.3761495834–0.3932225607; fixed GP covariance. No new fitting is done by this collector.",
        case_status=status, rows=rows, inputs=inputs,
        phase_edges=[.30, .325, .35, .375, .40, .425, .435], target_phase_index=3,
        bands_kms=dict(blue=[-40, -16], core=[-8, 8], red=[16, 40]),
        sign_convention="Positive Q gain means lower Q. Positive model-plus-GP change means brighter. Residual=data−planet−conditional GP.",
        group_order=["night0_order0", "night0_order1", "night1_order0", "night1_order1"],
        limitations=["Q includes both nights, both orders and all retained phases; target residuals use the same complete exposures in all B/C/R bands.",
            "Other-phase band_complete rows may use different exposures between bands. Missing values remain null.",
            "Provisional stages must not be used in a final physical assessment. D1.2 and the column-selected or upper-column-bracket extra cases are exploratory; D2/5/10/20 were predeclared.",
            "Saved observer spectra combine disk and halo. Their separate contributions cannot be inferred from these saved outputs."])
    (root / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (root / "comparison_table.md").write_text("Q gains are descriptive; residuals are ppm. An asterisk marks an additional exploratory case. Provisional status is recorded in comparison.json.\n\n" + "\n".join(table) + "\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
