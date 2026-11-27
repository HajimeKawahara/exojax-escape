"""Select one exploratory column match without reading any observer score."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=root / "column_selection.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the recorded selection; replay to a different output path")
    protocol_path = root / "column_selection_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    target = protocol["free_model_1_target_cm2"]
    structure_path = root.parent / "review/structure.json"
    structure_bytes = structure_path.read_bytes()
    structure = json.loads(structure_bytes)
    np.testing.assert_allclose(structure["free_model1_reference"]["n2_column_above_radius_cm2"]["1.725"], target, rtol=1e-12)
    recipe_path = Path("/tmp/escape-corrected-observer-full-f/recipe.json")
    edges = np.asarray(json.loads(recipe_path.read_text())["edges_rp"])
    np.testing.assert_array_equal(edges, structure["edges_rp"])
    width = np.maximum(edges[1:] - np.maximum(edges[:-1], 1.725), 0) * (1.9 * 7.1492e9)
    inputs = {str(path): digest(path) for path in (Path(__file__), protocol_path, recipe_path)}
    rows = []
    for factor in [protocol["reference_factor"], *protocol["eligible_factors"]]:
        row = next(row for row in structure["candidates"] if abs(row["factor"]-factor) < 1e-12)
        source = Path(row["directory"])
        state_path, summary_path, checkpoint_path = [source / name for name in ("fixed_inputs.npz", "summary.json", "checkpoint.npz")]
        summary = json.loads(summary_path.read_text())
        assert summary["converged"] and summary["accepted_for_observation"]
        assert digest(state_path) == summary["fixed_inputs_sha256"]
        assert digest(checkpoint_path) == summary["checkpoint_sha256"]
        cpu_paths = list(args.cpu_directory.glob(f"*_D{factor:g}.json"))
        assert len(cpu_paths) == 1, f"Missing or ambiguous independent CPU receipt for D{factor:g}"
        cpu_path = cpu_paths[0]
        cpu = json.loads(cpu_path.read_text())
        assert cpu["passed"] and cpu["checks"]["passed"] and Path(cpu["source_directory"]) == source
        cpu_arrays = Path(cpu["output"]["path"])
        assert digest(cpu_arrays) == cpu["output"]["sha256"]
        state, physical, checkpoint = np.load(state_path), np.load(cpu_arrays), np.load(checkpoint_path)
        np.testing.assert_array_equal(physical["theta"], checkpoint["theta"])
        np.testing.assert_array_equal(physical["log_density_factor"], checkpoint["log_density_factor"])
        np.testing.assert_allclose(physical["bound_cm3"], state["bound_cm3"], rtol=1e-10, atol=1e-25)
        column = row["n2_column_above_radius_cm2"]["1.725"]
        np.testing.assert_allclose(np.sum(state["bound_cm3"][:, 1:3].sum(axis=1) * width), column, rtol=1e-12)
        rows.append(dict(factor=factor, source_directory=str(source), cpu_review=str(cpu_path),
            n2_column_cm2=column, ratio_to_free=column/target, absolute_log_ratio=abs(math.log(column/target)),
            n2_column_above_2p15_cm2=row["n2_column_above_radius_cm2"]["2.15"],
            static_center_tau_one_radius_rp=row["static_center_tau_one_radius_rp"]))
        inputs.update({str(path): digest(path) for path in (state_path, summary_path, checkpoint_path, cpu_path, cpu_arrays)})
    best = min(rows[1:], key=lambda row: (row["absolute_log_ratio"], row["factor"]))
    selected = best if best["absolute_log_ratio"] < rows[0]["absolute_log_ratio"] else None
    result = dict(scope=__doc__, classification="Additional exploratory observer comparison selected by the empirical free-model column reference, not by Q or residuals.",
        target_n2_column_cm2=target, column_interval_rp=[1.725, 3.0], candidates=rows,
        selected_factor=None if selected is None else selected["factor"], selected=selected,
        free_n2_column_above_2p15_cm2=structure["free_model1_reference"]["n2_column_above_radius_cm2"]["2.15"],
        free_static_center_tau_one_radius_rp=structure["free_model1_reference"]["static_center_tau_one_radius_rp"],
        structure_readout=dict(path=str(structure_path), historical_sha256=hashlib.sha256(structure_bytes).hexdigest(),
            note="The live structure receipt may later expand. These copied columns were independently verified against the immutable exported populations."),
        inputs=inputs, observer_scores_used_in_selection=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(selected_factor=result["selected_factor"], candidates=rows)))


if __name__ == "__main__":
    main()
