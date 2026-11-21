#!/usr/bin/env python3
"""Run all registered Cloudy cases and report investigation readiness.

This provider tool uses retained numerical receipts and native MCEG navigation.
It never changes scientific Assessments or interprets code agreement as physical
validation. Cloudy and MCEG remain external to the escape package.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from cloudy_reference import (ROOT, artifact_sha256, has_artifact, run, sha256, write_json)

NUMERICAL_GATES = (
    "atomic_balance", "radiation_conservation", "reactive_flow_balance",
    "rt_se_balance", "thermal_balance", "coupled_flow_balance",
    "discretization_convergence", "coupled_gradients",
    "legacy_population_regression", "verification_tooling",
)


def case_configuration(case, resolution, thickness_cm):
    return dict(nmax=case["nmax"], temperature_k=case["temperature_k"],
                density_cm3=case["hydrogen_nuclei_cm3"],
                radiation_temperature_k=case["radiation_temperature_k"],
                dilution=case["radiation_dilution"], resolution=resolution,
                thickness_cm=thickness_cm)


def manifest_cases(manifest):
    cases = manifest["cases"]
    identifiers = [case["id"] for case in cases]
    if (not cases or len(set(identifiers)) != len(cases)
            or any(not re.fullmatch(r"[a-z][a-z0-9_]*", key) for key in identifiers)):
        raise ValueError("Use nonempty, unique, safe case IDs.")
    return cases


def run_cases(executable, manifest_path, destination, *, resolution=.05, thickness_cm=1e-4,
              reference_batch=None):
    manifest_path, destination = Path(manifest_path), Path(destination).resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = manifest_cases(manifest)
    protocol = ROOT / manifest["protocol"]
    if sha256(protocol) != manifest["protocol_sha256"]:
        raise ValueError("The registered protocol hash does not match.")
    declared = json.loads(protocol.read_text())["acceptance"]["reference_relative_population_error_maximum"]
    if manifest["relative_population_tolerance"] != declared:
        raise ValueError("The validation tolerance differs from the registered protocol.")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "cases.json").write_bytes(manifest_path.read_bytes())
    (destination / "protocol.json").write_bytes(protocol.read_bytes())
    summary = dict(batch_version=1, started_utc=datetime.now(timezone.utc).isoformat(),
                   manifest_sha256=sha256(manifest_path), protocol_sha256=sha256(protocol),
                   resolution=resolution, thickness_cm=thickness_cm, cases={}, status="running")
    write_json(destination / "batch.json", summary)
    for case in cases:
        # Every declared case is attempted, including cases after a disagreement.
        receipt = destination / case["id"] / "receipt.json"
        try:
            run(executable, receipt.parent,
                case_configuration(case, resolution, thickness_cm), protocol=protocol,
                compare_escape=True, relative_tolerance=declared,
                reference_directory=(Path(reference_batch) / case["id"] if reference_batch else None))
        except Exception as error:
            # Preserve an unexpected tool failure and still attempt later cases.
            receipt.parent.mkdir(parents=True, exist_ok=True)
            if receipt.exists():
                saved = json.loads(receipt.read_text())
            else:
                saved = {"configuration": case_configuration(case, resolution, thickness_cm)}
            saved.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(receipt, saved)
        summary["cases"][case["id"]] = {"receipt": f'{case["id"]}/receipt.json',
                                        "sha256": sha256(receipt)}
        write_json(destination / "batch.json", summary)
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary.update(evaluate_batch(destination, summary))
    write_json(destination / "batch.json", summary)
    return summary


def evaluate_batch(directory, summary=None):
    """Recompute coverage, integrity and accuracy; never trust saved PASS labels."""
    directory = Path(directory).resolve()
    if summary is None:
        summary = json.loads((directory / "batch.json").read_text())
    manifest = json.loads((directory / "cases.json").read_text())
    cases = manifest_cases(manifest)
    reasons, metrics, source_maps, executables = [], {}, [], []
    if sha256(directory / "cases.json") != summary["manifest_sha256"]:
        reasons.append("Case manifest changed.")
    if sha256(directory / "protocol.json") != summary["protocol_sha256"]:
        reasons.append("Protocol changed.")
    if set(summary["cases"]) != {case["id"] for case in cases}:
        reasons.append("Missing or unexpected declared cases.")
    for case in cases:
        item = summary["cases"].get(case["id"])
        if item is None:
            continue
        path = (directory / item["receipt"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != item["sha256"]:
            reasons.append(f'{case["id"]}: missing or changed receipt.')
            continue
        receipt = json.loads(path.read_text())
        expected = case_configuration(case, summary["resolution"], summary["thickness_cm"])
        if receipt["configuration"] != expected:
            reasons.append(f'{case["id"]}: input configuration differs.')
        if receipt.get("status") != "completed" or receipt.get("changed_sources"):
            reasons.append(f'{case["id"]}: failed execution or changed source.')
        if receipt.get("protocol", {}).get("sha256") != summary["protocol_sha256"]:
            reasons.append(f'{case["id"]}: protocol was not retained.')
        if receipt.get("acceptance", {}).get("relative_tolerance") != manifest["relative_population_tolerance"]:
            reasons.append(f'{case["id"]}: acceptance threshold differs.')
        for name, expected_hash in receipt.get("artifacts", {}).items():
            artifact = (path.parent / name).resolve()
            if not artifact.is_relative_to(path.parent) or not has_artifact(artifact) or artifact_sha256(artifact) != expected_hash:
                reasons.append(f'{case["id"]}: changed artifact {name}.')
        sources = receipt.get("source_sha256_before", {})
        if not sources or sources != receipt.get("source_sha256_after"):
            reasons.append(f'{case["id"]}: no unchanged source identity.')
        source_maps.append(sources)
        executables.append(receipt.get("executable_sha256"))
        errors = {}
        for observable in manifest["observables"]:
            name = observable + "_cm3"
            actual, reference = receipt.get("escape", {}).get(name), receipt.get("cloudy", {}).get(name)
            valid = isinstance(actual, (int, float)) and isinstance(reference, (int, float))
            valid = valid and math.isfinite(actual) and math.isfinite(reference) and actual > 0 and reference > 0
            error = abs(actual / reference - 1) if valid else None
            errors[observable] = error
            if error is None or error > manifest["relative_population_tolerance"]:
                reasons.append(f'{case["id"]}: {observable} misses the declared comparison criterion.')
        metrics[case["id"]] = errors
    if source_maps and any(source != source_maps[0] for source in source_maps):
        reasons.append("Different implementation identities were combined across cases.")
    if not executables or None in executables or len(set(executables)) != 1:
        reasons.append("A single recorded Cloudy executable identity is required across cases.")
    return dict(status="passed" if not reasons else "failed", reasons=reasons,
                relative_errors=metrics, source_sha256=source_maps[0] if source_maps else {},
                scope="Only the declared finite-atom code comparison; no scientific Assessment is changed.")


def evaluate_reference_refinement(fine_directory, coarse_directory):
    """Check every matched Cloudy control, independently of its escape result."""
    fine_directory, coarse_directory = Path(fine_directory).resolve(), Path(coarse_directory).resolve()
    fine = json.loads((fine_directory / "batch.json").read_text())
    coarse = json.loads((coarse_directory / "batch.json").read_text())
    manifest = json.loads((fine_directory / "cases.json").read_text())
    protocol = json.loads((fine_directory / "protocol.json").read_text())
    tolerance = protocol["acceptance"]["reference_refinement_relative_change_maximum"]
    expected_ids = {case["id"] for case in manifest_cases(manifest)}
    reasons, changes = [], {}
    for root, batch in ((fine_directory, fine), (coarse_directory, coarse)):
        if (sha256(root / "cases.json") != fine["manifest_sha256"]
                or batch["manifest_sha256"] != fine["manifest_sha256"]
                or sha256(root / "protocol.json") != fine["protocol_sha256"]
                or batch["protocol_sha256"] != fine["protocol_sha256"]
                or set(batch["cases"]) != expected_ids):
            reasons.append("Reference controls need the identical protocol and complete case manifest.")
    if not fine["resolution"] < coarse["resolution"]:
        reasons.append("The designated fine continuum resolution must be smaller.")
    for key in sorted(expected_ids):
        receipts = []
        for root, batch in ((fine_directory, fine), (coarse_directory, coarse)):
            item = batch["cases"].get(key, {})
            path = (root / item.get("receipt", "missing.json")).resolve()
            if (not path.is_relative_to(root) or not path.is_file()
                    or sha256(path) != item.get("sha256")):
                reasons.append(f"{key}: missing or changed reference receipt.")
                break
            receipt = json.loads(path.read_text())
            if receipt.get("status") != "completed" or not receipt.get("artifacts"):
                reasons.append(f"{key}: reference run did not complete with retained output.")
            for name, expected in receipt.get("artifacts", {}).items():
                artifact = (path.parent / name).resolve()
                if (not artifact.is_relative_to(path.parent) or not has_artifact(artifact)
                        or artifact_sha256(artifact) != expected):
                    reasons.append(f"{key}: missing or changed reference artifact {name}.")
            receipts.append(receipt)
        if len(receipts) != 2:
            continue
        left, right = receipts
        a, b = (dict(item["configuration"]) for item in receipts)
        a.pop("resolution", None); b.pop("resolution", None)
        if a != b or left.get("executable_sha256") != right.get("executable_sha256"):
            reasons.append(f"{key}: reference controls changed physical inputs or executable.")
        errors = {}
        for observable in manifest["observables"]:
            values = [item.get("cloudy", {}).get(observable + "_cm3") for item in receipts]
            valid = all(isinstance(value, (float, int)) and math.isfinite(value) and value > 0
                        for value in values)
            error = abs(values[1] / values[0] - 1) if valid else None
            errors[observable] = error
            if error is None or error > tolerance:
                reasons.append(f"{key}: {observable} misses the declared reference-refinement criterion.")
        changes[key] = errors
    return dict(status="passed" if not reasons else "failed", reasons=reasons,
                relative_changes=changes, relative_tolerance=tolerance,
                fine_batch_sha256=sha256(fine_directory / "batch.json"),
                coarse_batch_sha256=sha256(coarse_directory / "batch.json"))


def readiness(comparison, verification, investigation, current_sources, refinement=None):
    """Only explicit evidence and resolved actions can make a candidate reviewable."""
    if comparison["status"] != "passed":
        return dict(state="continue_diagnosis", reasons=comparison["reasons"])
    if refinement is None or refinement["status"] != "passed":
        return dict(state="continue_verification", reasons=["Every declared reference case needs a passing refinement control."])
    gates = {gate["id"]: gate["status"] for gate in verification.get("gates", [])}
    missing = [name for name in NUMERICAL_GATES if gates.get(name) != "passed"]
    core = {name: digest for name, digest in comparison["source_sha256"].items()
            if name.startswith("src/escape/")}
    matched = bool(core) and all(
        verification.get("selected_inputs_before", {}).get(name) == digest
        and current_sources.get(name) == digest for name, digest in core.items())
    if missing or not matched or not verification.get("review_ready") or verification.get("changed_inputs"):
        return dict(state="continue_verification", reasons=["Required numerical gates or current source identity are incomplete."],
                    missing_gates=missing, source_identity_matches=matched)
    bundles = [entry.get("investigation") for entry in investigation.get("contracts", [])
               if entry.get("model") == "escape.cloudy_investigation"]
    if len(bundles) != 1 or bundles[0] is None:
        return dict(state="pending_followups", reasons=["A current native MCEG investigation view is required."])
    pending = [task["id"] for task in bundles[0]["pending_follow_ups"]]
    unresolved = [name for name in pending if name != "review_completion"]
    if unresolved:
        return dict(state="pending_followups", reasons=["Recorded diagnostic or repair actions remain unresolved."],
                    pending_follow_ups=unresolved)
    return dict(state="ready_for_review", reasons=["All declared comparison cases and numerical gates pass at the same source identity; required investigation actions have recorded resolutions."],
                pending_follow_ups=pending,
                limitation="Readiness does not create or approve a scientific Assessment.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    execute = sub.add_parser("run", help="Attempt every case from a frozen manifest.")
    source = execute.add_mutually_exclusive_group(required=True)
    source.add_argument("--executable", type=Path)
    source.add_argument("--reference-batch", type=Path, help="Replay verified retained Cloudy outputs at current escape source.")
    execute.add_argument("--cases", type=Path, default=ROOT / "mceg/investigations/cloudy_validation_cases.json")
    execute.add_argument("--record", type=Path, required=True)
    execute.add_argument("--resolution", type=float, default=.05)
    execute.add_argument("--thickness-cm", type=float, default=1e-4)
    check = sub.add_parser("readiness", help="Check evidence and fresh MCEG pending actions.")
    check.add_argument("--batch", type=Path, required=True)
    check.add_argument("--verification", type=Path, required=True)
    check.add_argument("--reference-control", type=Path, required=True, help="Matched coarser Cloudy campaign for every case.")
    check.add_argument("--contract", type=Path, default=ROOT / "mceg/contracts/escape.cloudy_investigation.yaml")
    args = parser.parse_args()
    if args.action == "run":
        result = run_cases(args.executable, args.cases, args.record,
                           resolution=args.resolution, thickness_cm=args.thickness_cm,
                           reference_batch=args.reference_batch)
        successful = result["status"] == "passed"
    else:
        comparison = evaluate_batch(args.batch)
        verification = json.loads(args.verification.read_text())
        before = sha256(args.contract)
        command = [sys.executable, "-m", "mceg", "explain", str(args.contract),
                   "--claim", "repaired_reference_agreement", "--investigation", "--json"]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        if sha256(args.contract) != before:
            raise ValueError("The MCEG contract changed during readiness review.")
        for gate in verification.get("gates", []):
            for name, expected in gate.get("artifacts", {}).items():
                artifact = args.verification.parent / name
                if not artifact.is_file() or sha256(artifact) != expected:
                    raise ValueError(f"Changed numerical verification artifact: {name}")
        current_sources = {name: sha256(ROOT / name) if (ROOT / name).is_file() else None
                           for name in comparison["source_sha256"]}
        refinement = evaluate_reference_refinement(args.batch, args.reference_control)
        result = readiness(comparison, verification, json.loads(process.stdout), current_sources, refinement)
        result["reference_refinement"] = refinement
        result.update(contract_sha256=before, native_mceg_command=command,
                      batch_sha256=sha256(args.batch / "batch.json"),
                      verification_sha256=sha256(args.verification))
        successful = result["state"] == "ready_for_review"
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
