"""Watch selected factors after a numerical restart without replacing old cases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from watch_candidates import assess, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--factors", nargs="+", type=float, default=[10., 20.])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    control = json.loads((args.output / "D1/review.json").read_text())
    for factor in args.factors:
        name = f"D{factor:g}"
        source = args.attempt / name
        summary_path = source / "summary.json"
        print(f"Waiting for accepted {name} in {args.attempt}", flush=True)
        while not summary_path.exists():
            if (args.attempt / "summary.json").exists():
                print(f"Closed numerical attempt before {name}; no unaccepted state rendered", flush=True)
                return
            time.sleep(30)
        summary = json.loads(summary_path.read_text())
        if not summary.get("accepted_for_observation") or not summary.get("converged"):
            print(f"Stage {name} is unaccepted; remaining observer cases unevaluated", flush=True)
            return
        state = source / "fixed_inputs.npz"
        if digest(state) != summary["fixed_inputs_sha256"]:
            raise ValueError("Physical export hash mismatch")
        configuration = json.loads((args.attempt / "configuration.json").read_text())
        for path, expected in configuration["source_and_input_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"Solver input/source changed: {path}")
        destination = args.output / name
        if (destination / "review.json").exists():
            saved = json.loads((destination / "review.json").read_text())
            if saved["inputs"].get(str(state)) != digest(state):
                raise ValueError(f"Existing observer case {name} uses a different state")
        else:
            print(f"Rendering provisional {name}", flush=True)
            subprocess.run([sys.executable, "examples/halpha_dense_observe.py", "--state-npz", str(state),
                "--state-review", str(summary_path), "--raw-output", str(args.raw / name),
                "--output", str(destination)], check=True)
        assess(destination, dict(path=str(summary_path), sha256=digest(summary_path)), control)
        print(f"Observer {name} complete; CPU qualification remains separate", flush=True)


if __name__ == "__main__":
    main()
