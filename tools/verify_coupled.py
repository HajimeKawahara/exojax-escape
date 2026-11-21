#!/usr/bin/env python3
"""Execute provider-owned gates and retain evidence; never update MCEG claims."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git_state(root):
    def read(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    return {"root": str(root), "revision": read("rev-parse", "HEAD"),
            "dirty": bool(read("status", "--porcelain"))}


def selected_gates(workflow, requested):
    gates = {gate["id"]: gate for gate in workflow["gates"]}
    if len(gates) != len(workflow["gates"]):
        raise ValueError("duplicate gate id")
    ordered, visiting = [], set()

    def visit(name):
        if name not in gates:
            raise ValueError(f"unknown gate: {name}")
        if name in visiting:
            raise ValueError(f"cyclic prerequisite: {name}")
        if name in ordered:
            return
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"unsafe gate id: {name}")
        visiting.add(name)
        for dependency in gates[name].get("requires", []):
            visit(dependency)
        visiting.remove(name)
        ordered.append(name)

    for name in requested or gates:
        visit(name)
    return [gates[name] for name in ordered]


def input_files(root, patterns, *, require_matches=True):
    paths = set()
    for pattern in patterns:
        matches = glob.glob(str(root / pattern), recursive=True)
        if not matches and require_matches:
            raise ValueError(f"input pattern has no files: {pattern}")
        for name in matches:
            path = Path(name)
            if path.is_file():
                resolved = path.resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError(f"input outside repository: {path}")
                paths.add(resolved.relative_to(root.resolve()).as_posix())
    return sorted(paths)


def read_junit(path):
    if not path.exists():
        return {"cases": [], "metrics": {}, "counts": {}, "error": "JUnit missing"}
    try:
        tree = ET.parse(path)
    except ET.ParseError as error:
        return {"cases": [], "metrics": {}, "counts": {}, "error": str(error)}
    cases, metrics = [], {}
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for case in tree.iter("testcase"):
        status = next((key for key, tag in (("failed", "failure"), ("error", "error"),
                                            ("skipped", "skipped"))
                       if case.find(tag) is not None), "passed")
        counts[status] += 1
        identity = f"{case.get('classname', '')}::{case.get('name', '')}"
        properties = []
        for prop in case.findall("properties/property"):
            name, raw = prop.get("name", ""), prop.get("value", "")
            properties.append({"name": name, "value": raw})
            metrics.setdefault(name, []).append({"case": identity, "value": raw})
        cases.append({"id": identity, "status": status, "properties": properties})
    return {"cases": cases, "metrics": metrics, "counts": counts}


def evaluate_gate(report, returncode, criteria):
    checks = []
    for criterion in criteria:
        values = report["metrics"].get(criterion["name"], [])
        check = {"criterion": criterion, "values": values, "status": "passed"}
        if len(values) < criterion.get("minimum_count", 1):
            check["status"] = "incomplete"
        for item in values:
            try:
                value = float(item["value"])
                passed = (math.isfinite(value)
                          and value >= criterion.get("minimum", -math.inf)
                          and value <= criterion.get("maximum", math.inf))
            except ValueError:
                passed = False
            if not passed:
                check["status"] = "failed"
        checks.append(check)
    counts = report["counts"]
    if returncode != 0 or counts.get("failed", 0) or counts.get("error", 0):
        return "failed", checks
    if any(check["status"] == "failed" for check in checks):
        return "failed", checks
    if (report.get("error") or not counts.get("passed", 0)
            or counts.get("skipped", 0)
            or any(check["status"] == "incomplete" for check in checks)):
        return "incomplete", checks
    return "passed", checks


def run(workflow_path, requested, output, extra_pythonpath, *, root=ROOT):
    workflow = json.loads(workflow_path.read_text())
    if workflow["workflow_version"] != "1":
        raise ValueError("unsupported workflow version")
    gates = selected_gates(workflow, requested)
    if output is None:
        print(json.dumps({"workflow": str(workflow_path), "gates": gates,
                          "action": "plan only; supply --record NEW_DIRECTORY to execute"}, indent=2))
        return 0
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = input_files(root, workflow["inputs"])
    before = {name: digest(root / name) for name in paths}
    for name in paths:
        snapshot = output / "inputs" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, snapshot)
    environment = os.environ.copy()
    environment.update(workflow.get("environment", {}))
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), *(str(Path(path).resolve()) for path in extra_pythonpath)])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    probe_command = [sys.executable, "-c", (
        "import importlib,json,sys\n"
        "out={}\n"
        "for name in json.loads(sys.argv[1]):\n"
        " module=importlib.import_module(name)\n"
        " out[name]={'file':getattr(module,'__file__',None),"
        "'version':getattr(module,'__version__',None)}\n"
        "print(json.dumps(out))\n"
    ), json.dumps(workflow.get("import_modules", []))]
    probe = subprocess.run(probe_command, cwd=root, env=environment,
                           capture_output=True, text=True, check=False)
    versions = {}
    for package in ("jax", "jaxlib", "numpy", "scipy", "pytest", "exojax"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    receipt = {
        "receipt_version": "1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": str(workflow_path.relative_to(root)), "workflow_sha256": digest(workflow_path),
        "repository": git_state(root), "python": sys.version, "executable": sys.executable,
        "package_metadata_versions": versions,
        "dependency_probe": {"command": probe_command, "exit_code": probe.returncode,
                             "stdout": probe.stdout, "stderr": probe.stderr},
        "external_source_roots": [git_state(Path(path).resolve()) for path in extra_pythonpath],
        "environment": {key: environment[key] for key in
                        (*workflow.get("environment", {}), "PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
        "selected_inputs_before": before, "gates": [], "status": "running",
        "limitations": [
            "Gate outcomes do not update or establish scientific Assessments.",
            "Package metadata plus source roots are not a complete environment lock.",
            "Only declared repository inputs are snapshotted; external dependencies are not vendored.",
        ],
    }
    write_json(output / "receipt.json", receipt)
    statuses = {}
    for gate in gates:
        entry = {"id": gate["id"], "scope": gate["scope"],
                 "what_not_proved": gate.get("what_not_proved", [])}
        blocked = [name for name in gate.get("requires", []) if statuses[name] != "passed"]
        if blocked:
            entry.update(status="blocked", prerequisites=blocked)
        elif not gate.get("tests"):
            entry.update(status="incomplete", reason="No retained executable reference is configured.")
        else:
            junit = output / f"{gate['id']}.xml"
            log = output / f"{gate['id']}.log"
            command = [sys.executable, "-m", "pytest", "-q", "-o", "junit_family=legacy",
                       *gate["tests"], f"--junitxml={junit}"]
            entry.update(command=command, cwd=str(root))
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=root, env=environment, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            report = read_junit(junit)
            status, checks = evaluate_gate(report, result.returncode, gate.get("metrics", []))
            entry.update(status=status, exit_code=result.returncode, junit=report,
                         metric_checks=checks, artifacts={
                             path.name: digest(path) for path in (log, junit) if path.exists()})
        statuses[gate["id"]] = entry["status"]
        receipt["gates"].append(entry)
        write_json(output / "receipt.json", receipt)
        print(f"{gate['id']}: {entry['status']}", flush=True)
    after_paths = sorted(set(paths) | set(input_files(root, workflow["inputs"], require_matches=False)))
    after = {name: digest(root / name) if (root / name).exists() else None for name in after_paths}
    changed = [name for name in after_paths if before.get(name) != after[name]]
    receipt.update(selected_inputs_after=after, changed_inputs=changed,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    receipt["status"] = ("failed" if changed or probe.returncode != 0 or "failed" in statuses.values()
                         else "incomplete" if any(status != "passed" for status in statuses.values())
                         else "passed")
    receipt["review_ready"] = not changed and probe.returncode == 0
    write_json(output / "receipt.json", receipt)
    return 0 if receipt["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=ROOT / "mceg/workflows/coupled.json")
    parser.add_argument("--gate", action="append", default=[], help="Includes prerequisites; repeatable.")
    parser.add_argument("--record", type=Path, help="New directory only; never overwrites a prior attempt.")
    parser.add_argument("--pythonpath", action="append", default=[], help="Explicit external source directory.")
    args = parser.parse_args()
    try:
        return run(args.workflow.resolve(), args.gate, args.record, args.pythonpath)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
