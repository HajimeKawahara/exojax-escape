"""Audit saved observer artifacts and comparison identities without new fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    collection_path = directory / "comparison.json"
    collection = json.loads(collection_path.read_text())
    inputs = {str(Path(__file__)): sha256(__file__), str(collection_path): sha256(collection_path)}
    checked = 0

    def check(condition):
        nonlocal checked
        if not condition:
            raise AssertionError(f"Observer artifact audit failed at check {checked + 1}")
        checked += 1

    def close(left, right, atol=1e-7):
        check(np.allclose(left, right, rtol=0, atol=atol))

    for path, digest in collection["inputs"].items():
        check(sha256(path) == digest)
        inputs[path] = digest
    control = json.loads((directory / "D1/review.json").read_text())
    for name, status in collection["case_status"].items():
        if status == "not_yet_evaluated":
            continue
        check(status in ("previous_accepted_D1_control", "independent_CPU_verified"))
        case_path = directory / name / "review.json"
        case = json.loads(case_path.read_text())
        for path, digest in case["inputs"].items():
            check(sha256(path) == digest)
        for filename, digest in case["figure_sha256"].items():
            check(sha256(case_path.parent / filename) == digest)
        check(sha256(case["raw_arrays"]["path"]) == case["raw_arrays"]["sha256"])
        with np.load(case["raw_arrays"]["path"]) as raw:
            check(np.isfinite(raw["candidate"]).all())
        for selection, records in case["selections"].items():
            check(case["target_support"][selection] == control["target_support"][selection])
            for mode, record in records["candidate"].items():
                baseline, d1 = records["baseline"][mode], control["selections"][selection]["candidate"][mode]
                close(baseline["Q"], control["selections"][selection]["baseline"][mode]["Q"])
                if mode == "profiled":
                    check(record["nuisance_success"])
                close(record["Q"], sum(record["group_Q"]))
                for scope, bands in record["target"].items():
                    if scope in ("night1", "night1_order0", "night1_order1"):
                        check(bands is None)
                        continue
                    for band, value in bands.items():
                        check(value["supported"])
                        close(value["observed_ppm"], value["planet_ppm"] + value["gp_ppm"] + value["residual_ppm"])
                        close(value["observed_ppm"], baseline["target"][scope][band]["observed_ppm"])
                row = next(row for row in collection["rows"] if
                    (row["case"], row["selection"], row["nuisance_mode"]) == (name, selection, mode))
                close(row["Q_gain_vs_PR13"], baseline["Q"] - record["Q"])
                close(row["Q_gain_vs_D1"], d1["Q"] - record["Q"])
                residual = [record["target"]["all"][band]["residual_ppm"] for band in ("blue", "core", "red")]
                close(row["residual_BCR_ppm"], residual)
                close(row["residual_blue_minus_core_ppm"], residual[0] - residual[1])
                close(np.array(row["planet_change_vs_D1_BCR_ppm"]) + row["GP_change_vs_D1_BCR_ppm"],
                      row["model_plus_GP_change_vs_D1_BCR_ppm"])
    refinement_path = directory / "D20/target_ray_refinement.json"
    if refinement_path.exists():
        refinement = json.loads(refinement_path.read_text())
        inputs[str(refinement_path)] = sha256(refinement_path)
        for path, digest in refinement["inputs"].items():
            check(sha256(path) == digest)
        check(sha256(refinement["raw_arrays"]["path"]) == refinement["raw_arrays"]["sha256"])
        with np.load(refinement["raw_arrays"]["path"]) as raw:
            check(np.array_equal(raw["original"][[0, 1, 2, 5, 6]], raw["refined"][[0, 1, 2, 5, 6]]))
            case = json.loads((directory / "D20/review.json").read_text())
            with np.load(case["raw_arrays"]["path"]) as original:
                check(np.array_equal(raw["original"], original["candidate"]))
        for versions in refinement["nuisance_multistart"].values():
            for control in versions.values():
                check(all(fit["success"] for fit in control["additional_starts"]))
                check(control["two_start_Q_agrees_with_best_tried"])
    result = dict(passed=True, checks=checked, inputs=inputs, case_status=collection["case_status"],
        scope="Saved-input and raw/figure hashes; common support; matching nuisance references; native group Q sum; data=planet+GP+residual; explicit unsupported rows; comparison arithmetic; saved target-refinement array identity and multistart success.",
        limitations="No new equilibrium, observer integration, optimizer or statistical-significance test is performed by this artifact audit.")
    (directory / "artifact_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=True, checks=checked)))


if __name__ == "__main__":
    main()
