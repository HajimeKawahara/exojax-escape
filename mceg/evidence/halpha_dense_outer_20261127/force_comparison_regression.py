"""Check cancellation-aware diagnostic comparison on the saved D8 state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from halpha_dense_crosscheck import compare


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("/tmp/escape-halpha-dense-cpu-crosscheck/attempt05"))
    parser.add_argument("--new", type=Path, default=Path("/tmp/escape-halpha-dense-cpu-crosscheck/attempt06"))
    parser.add_argument("--gpu", type=Path, default=Path("/tmp/escape-halpha-dense-outer/attempt05/D8"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_path, new_path = args.old/"case_3_D8.npz", args.new/"case_1_D8.npz"
    receipt_path = args.new/"case_1_D8.json"
    receipt = json.loads(receipt_path.read_text())
    if not receipt["passed"]:
        raise ValueError("The new complete CPU check must pass first.")
    old, new = np.load(old_path), np.load(new_path)
    gpu_path = args.gpu/"physical_summary.npz"
    gpu = np.load(gpu_path)
    np.testing.assert_array_equal(old["theta"], new["theta"])
    key = "wind__reference_momentum_defect_cm_s2"
    a, b = old[key], gpu[key]
    scale = np.asarray(receipt["force_and_flow"]["reference_force_term_scale_cm_s2"])
    original, corrected = compare(a, b), compare(a, b, 1e-12*scale)
    perturbed = a.copy()
    perturbed[1] += 1e-6*scale[1]
    injected = compare(perturbed, b, 1e-12*scale)
    checks = dict(original_diagnostic_rejects_roundoff=not original["passed"],
        corrected_diagnostic_accepts_roundoff=corrected["passed"],
        injected_force_error_rejected=not injected["passed"],
        full_residual_and_physical_checks_pass=receipt["passed"])
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).resolve(), root/"examples/halpha_dense_crosscheck.py",
             args.old/"crosscheck_source.py", old_path, new_path, receipt_path, gpu_path]
    output = dict(scope=__doc__, passed=all(checks.values()), checks=checks,
        original=original, corrected=corrected, injected=injected,
        maximum_difference_over_force_terms=float(np.max(abs(a-b)/scale)),
        injected_cell=1, injected_acceleration_cm_s2=float(1e-6*scale[1]),
        tolerance_scope="Only the cancellation-prone reference momentum diagnostic uses a 1e-12 uncancelled-force floor. Full-F, thermal, conservation, population and continuum criteria are unchanged.",
        source_and_input_sha256={str(path): digest(path) for path in paths})
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    if not output["passed"]:
        raise RuntimeError("Saved-array diagnostic regression failed.")


if __name__ == "__main__":
    main()
