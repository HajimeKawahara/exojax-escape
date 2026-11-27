"""Check retained input pins and compact-state identity without re-solving."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    campaign = Path(__file__).resolve().parent
    root = campaign.parents[2]
    numerical = json.loads((campaign / "numerical_summary.json").read_text())
    verified, historical, missing = {}, [], []
    archives = {}
    for pattern, original in (("driver_source.py", "halpha_dense_outer.py"),
                              ("crosscheck_source.py", "halpha_dense_crosscheck.py")):
        for p in campaign.glob("attempts/*/"+pattern):
            archives[(str(root / "examples" / original), digest(p))] = p

    def pin(path, expected, receipt):
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        if p.exists() and digest(p) == expected:
            verified[str(p)] = expected
        elif (str(p), expected) in archives:
            historical.append(dict(original=str(p), sha256=expected,
                archived=str(archives[(str(p), expected)].relative_to(root)), receipt=str(receipt),
                scope="Historical source snapshot; not a current-source equality claim."))
        else:
            missing.append(dict(path=str(p), expected=expected, receipt=str(receipt),
                actual=digest(p) if p.is_file() else None))

    def walk(value, receipt):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                pin(value["path"], value["sha256"], receipt)
            for key, item in value.items():
                if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item):
                    if key.startswith("/") or (root / key).is_file():
                        pin(key, item, receipt)
                walk(item, receipt)
        elif isinstance(value, list):
            for item in value:
                walk(item, receipt)

    # The early review missed tuple serialization and is superseded explicitly.
    superseded = campaign / "review/driver_review.json"
    # Historical attempt receipts intentionally describe their own revisions.
    for p in sorted(campaign.rglob("*.json")):
        historical_attempt = ("attempts" in p.relative_to(campaign).parts
                              and not (p.parent.name.startswith("cpu_") and p.name == "review.json"))
        if p == superseded or p.name == "validation.json" or historical_attempt:
            continue
        walk(json.loads(p.read_text()), p.relative_to(root))
    for raw, item in numerical["archived_inputs"].items():
        pin(raw, item["sha256"], "numerical_summary.json")
        assert digest(campaign / item["archived"]) == item["sha256"]
    cases = []
    for record in numerical["accepted"]:
        folder = campaign / record["state"]
        gpu = json.loads((folder / "summary.json").read_text())
        cpu = json.loads((folder / "cpu_review.json").read_text())
        assert gpu["converged"] and cpu["passed"]
        assert gpu["residual_infinity"] < 1e-9 and cpu["residual_infinity"] < 1e-9
        with np.load(folder / "coordinates.npz") as saved, np.load(folder / "cpu_arrays.npz") as check:
            np.testing.assert_array_equal(saved["theta"], check["theta"])
            np.testing.assert_array_equal(saved["log_density_factor"], check["log_density_factor"])
        for name, key in (("physical_summary.npz", "physical_summary_sha256"),
                          ("fixed_inputs.npz", "fixed_inputs_sha256")):
            assert digest(folder / name) == gpu[key]
        cases.append(record["factor"])
    result = dict(passed=not missing, accepted_exact_coordinate_cases=cases,
        verified_pins=verified, historical_driver_pins=historical, unresolved_pins=missing,
        superseded_review=dict(path=str(superseded.relative_to(root)), sha256=digest(superseded),
            reason="Historical pre-export review; its old mutable README pin is not revalidated. Actual tuple controls and current CPU checks supersede it."),
        scope="Artifact identity and accepted finite-grid roots; physical support and transfer adequacy remain separate.",
        source_sha256=digest(__file__))
    (campaign / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=result["passed"], cases=cases, pins=len(verified),
                         historical=len(historical), unresolved=missing), indent=2))
    assert result["passed"]


if __name__ == "__main__":
    main()
