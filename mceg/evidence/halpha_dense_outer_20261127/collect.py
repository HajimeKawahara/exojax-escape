"""Archive compact accepted states and exact CPU receipts after a closed run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--cpu", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-stage", type=Path, required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    run_reviews = [json.loads((p / "summary.json").read_text()) for p in args.run]
    cpu_reviews = [json.loads((p / "review.json").read_text()) for p in args.cpu]
    assert all(r["inputs_unchanged"] for r in run_reviews)
    assert all(r["inputs_unchanged"] for r in cpu_reviews)
    records, pins, excluded = [], {}, []

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        pins[str(source)] = dict(sha256=digest(source),
                                archived=str(destination.relative_to(output)))

    receipts = [p for folder in args.cpu for p in
                [folder / "baseline_D1.json", *sorted(folder.glob("case_*.json"))]]
    seen = {}
    for receipt in receipts:
        case = json.loads(receipt.read_text())
        if not case["passed"]:
            destination = output / "attempts" / ("cpu_"+receipt.parent.name)
            copy(receipt, destination / ("excluded_"+receipt.name))
            arrays = Path(case["output"]["path"])
            assert digest(arrays) == case["output"]["sha256"]
            copy(arrays, destination / ("excluded_"+arrays.name))
            excluded.append(dict(receipt=str(receipt), factor=case["factor"],
                reason="Failed CPU comparison; excluded unless a separately qualified receipt exists."))
            continue
        assert case["residual_infinity"] < 1e-9
        source = Path(case["source_directory"]) if case["source_directory"] else args.baseline_stage
        accepted = json.loads((source / "summary.json").read_text())
        assert accepted["converged"] and accepted["accepted_for_observation"]
        assert np.isclose(case["factor"], accepted["factor"], rtol=1e-13, atol=0.)
        label = source.name
        if label in seen:
            with np.load(case["output"]["path"]) as new, np.load(seen[label]) as old:
                np.testing.assert_array_equal(new["theta"], old["theta"])
                np.testing.assert_array_equal(new["log_density_factor"], old["log_density_factor"])
            continue
        seen[label] = case["output"]["path"]
        destination = output / "states" / label
        for filename, key in (("checkpoint.npz", "checkpoint_sha256"),
                              ("physical_summary.npz", "physical_summary_sha256"),
                              ("fixed_inputs.npz", "fixed_inputs_sha256")):
            assert digest(source / filename) == accepted[key]
        for filename in ("summary.json", "physical_summary.npz", "fixed_inputs.npz"):
            copy(source / filename, destination / filename)
        copy(receipt, destination / "cpu_review.json")
        # Omit large inverse accelerators; retain exact accepted coordinates.
        with np.load(source / "checkpoint.npz") as checkpoint:
            np.savez_compressed(destination / "coordinates.npz", **{
                key: checkpoint[key] for key in
                ("theta", "residual", "log_density_factor")})
        cpu_arrays = Path(case["output"]["path"])
        assert digest(cpu_arrays) == case["output"]["sha256"]
        copy(cpu_arrays, destination / "cpu_arrays.npz")
        with np.load(cpu_arrays) as cpu, np.load(destination / "coordinates.npz") as saved:
            np.testing.assert_array_equal(cpu["theta"], saved["theta"])
            np.testing.assert_array_equal(cpu["log_density_factor"], saved["log_density_factor"])
        if case["source_directory"] is None:
            # The CPU baseline checks the original spherical physical state.
            # Tie the dense-driver D1 observer export to those same coordinates
            # and coefficients; its required-force diagnostics have separate
            # CPU algebra checks (a zero GPU acceleration can round at 1e-14).
            with np.load(cpu_arrays) as cpu, np.load(source / "fixed_inputs.npz") as physical:
                for key in physical.files:
                    if key != "wavenumber_cm":
                        np.testing.assert_allclose(cpu[key], physical[key], rtol=1e-10, atol=1e-25)
        parent = source.parent
        copy(parent / "configuration.json", output / "attempts" / parent.name / "configuration.json")
        archived_driver = parent / "driver_source.py"
        if archived_driver.exists():
            copy(archived_driver, output / "attempts" / parent.name / "driver_source.py")
        records.append(dict(factor=accepted["factor"], state=f"states/{label}",
            gpu_residual_infinity=accepted["residual_infinity"],
            cpu_residual_infinity=case["residual_infinity"],
            cpu_gpu_difference=case["residual_difference_infinity"],
            raw_source=str(source), raw_checkpoint_sha256=accepted["checkpoint_sha256"]))
    for folder in args.run:
        for filename in ("configuration.json", "summary.json", "events.jsonl"):
            copy(folder / filename, output / "attempts" / folder.name / filename)
    cpu_attempts = []
    for folder, review in zip(args.cpu, cpu_reviews):
        destination = output / "attempts" / ("cpu_"+folder.name) / "review.json"
        copy(folder / "review.json", destination)
        cpu_attempts.append(dict(path=str(destination), sha256=digest(destination),
            passed=review["passed"], all_requested_stages_checked=review["all_requested_stages_checked"],
            unavailable_stages=review["unavailable_stages"]))
    (output / "cpu_review.json").write_text(json.dumps(dict(
        scope="Only individually passed CPU cases enter the accepted archive. Historical unavailable requests and excluded failures remain explicit.",
        passed=bool(records), all_historical_runs_passed=all(r["passed"] for r in cpu_reviews),
        attempts=cpu_attempts, excluded_cases=excluded), indent=2)+"\n")
    result = dict(scope="Conditional full population/thermal roots with prescribed momentum support.",
        accepted=sorted(records, key=lambda item: item["factor"]),
        final_cpu_attempt_all_requested_stages_checked=cpu_reviews[-1]["all_requested_stages_checked"],
        cpu_attempts=cpu_attempts,
        excluded_cpu_checks=excluded,
        planned_observer_factors_available=all(any(np.isclose(r["factor"], d) for r in records)
                                               for d in (1., 2., 5., 10., 20.)),
        archived_inputs=pins,
        omitted_raw_artifacts="Inverse accelerators and full radiation pickle remain in hashed raw outputs.",
        physical_support="unknown; no magnetic geometry or momentum closure solved")
    (output / "numerical_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(accepted=[r["factor"] for r in result["accepted"]],
                         final_cpu_all_requested=result["final_cpu_attempt_all_requested_stages_checked"],
                         planned_observer_factors_available=result["planned_observer_factors_available"])))


if __name__ == "__main__":
    main()
