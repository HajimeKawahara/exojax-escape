"""Audit execution failures and source identity without importing MCEG."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "verify_coupled", Path(__file__).parents[1] / "tools/verify_coupled.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_gate_dependencies_and_cycles():
    workflow = {"gates": [{"id": "atom"}, {"id": "flow", "requires": ["atom"]}]}
    assert [gate["id"] for gate in runner.selected_gates(workflow, ["flow"])] == ["atom", "flow"]
    workflow["gates"][0]["requires"] = ["flow"]
    with pytest.raises(ValueError, match="cyclic"):
        runner.selected_gates(workflow, ["flow"])


def test_missing_nonfinite_and_failed_metrics_are_not_passes(tmp_path):
    junit = tmp_path / "test.xml"
    junit.write_text('<testsuite><testcase classname="test" name="case"><properties>'
                     '<property name="error" value="nan"/></properties></testcase></testsuite>')
    report = runner.read_junit(junit)
    assert runner.evaluate_gate(report, 0, [{"name": "absent"}])[0] == "incomplete"
    assert runner.evaluate_gate(report, 0, [{"name": "error", "maximum": .05}])[0] == "failed"
    report["metrics"]["error"][0]["value"] = ".11"
    assert runner.evaluate_gate(report, 0, [{"name": "error", "maximum": .05}])[0] == "failed"


def test_real_attempt_keeps_failed_accuracy_and_changed_source(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original\n")
    (tmp_path / "test_sample.py").write_text(
        'from pathlib import Path\n'
        'def test_measurement(record_property):\n'
        '    record_property("fractional_change", .11)\n'
        '    Path("source.txt").write_text("changed\\n")\n'
    )
    workflow = {"workflow_version": "1", "inputs": ["source.txt", "test_sample.py", "workflow.json"],
                "environment": {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, "gates": [
                    {"id": "accuracy", "scope": "Synthetic runner regression, not physics.",
                     "tests": ["test_sample.py"],
                     "metrics": [{"name": "fractional_change", "maximum": .05}]},
                    {"id": "dependent", "scope": "Must be blocked.", "requires": ["accuracy"],
                     "tests": ["test_sample.py"]}]}
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(json.dumps(workflow))
    destination = tmp_path / "attempt"
    assert runner.run(workflow_path, [], destination, [], root=tmp_path) == 1
    receipt = json.loads((destination / "receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["review_ready"]
    assert receipt["changed_inputs"] == ["source.txt"]
    assert receipt["gates"][0]["junit"]["counts"]["passed"] == 1
    assert [gate["status"] for gate in receipt["gates"]] == ["failed", "blocked"]
    assert (destination / "inputs/source.txt").read_text() == "original\n"
    assert (destination / "accuracy.log").exists()
    assert workflow_path.read_text() == json.dumps(workflow)
    with pytest.raises(FileExistsError):
        runner.run(workflow_path, [], destination, [], root=tmp_path)
