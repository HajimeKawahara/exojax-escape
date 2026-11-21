"""Exercise campaign completeness and continuation policy without Cloudy or MCEG."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("verify_cloudy_cases", TOOLS / "verify_cloudy_cases.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
sys.path.pop(0)


def campaign(tmp_path, monkeypatch, *, first_error=.0, raises=False):
    protocol = tmp_path / "protocol.json"
    workflow.write_json(protocol, {"acceptance": {"reference_relative_population_error_maximum": .05, "reference_refinement_relative_change_maximum": .01}})
    cases = [dict(id=name, nmax=5, temperature_k=9000., hydrogen_nuclei_cm3=1e12,
                  radiation_temperature_k=10000., radiation_dilution=.025)
             for name in ("first", "second")]
    manifest = tmp_path / "cases.json"
    workflow.write_json(manifest, dict(cases=cases, protocol="protocol.json",
                                      protocol_sha256=workflow.sha256(protocol),
                                      relative_population_tolerance=.05, observables=["n2", "n3"]))
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    calls = []

    def stub(executable, destination, config, **options):
        calls.append(destination.name)
        if raises and destination.name == "first":
            raise RuntimeError("Synthetic external failure")
        destination.mkdir()
        raw = destination / "raw.txt"
        raw.write_text("Synthetic fixture, not a Cloudy execution.\n")
        error = first_error if destination.name == "first" else 0.
        receipt = dict(configuration=config, status="completed", agreement="passed", executable_sha256="cloudy",
                       changed_sources=[], source_sha256_before={"src/escape/model.py": "abc"},
                       source_sha256_after={"src/escape/model.py": "abc"},
                       protocol={"sha256": workflow.sha256(protocol)},
                       acceptance={"relative_tolerance": .05},
                       artifacts={"raw.txt": workflow.sha256(raw)},
                       escape={"n2_cm3": 1., "n3_cm3": 1. + error},
                       cloudy={"n2_cm3": 1., "n3_cm3": 1.})
        workflow.write_json(destination / "receipt.json", receipt)
        return receipt

    monkeypatch.setattr(workflow, "run", stub)
    target = tmp_path / "attempt"
    summary = workflow.run_cases("unused", manifest, target)
    return target, summary, calls


def test_failed_comparison_does_not_drop_later_cases(tmp_path, monkeypatch):
    target, result, calls = campaign(tmp_path, monkeypatch, first_error=.3)
    assert calls == ["first", "second"]
    assert result["status"] == "failed"
    assert result["relative_errors"]["first"]["n3"] == pytest.approx(.3)
    assert result["relative_errors"]["second"]["n3"] == 0
    with pytest.raises(FileExistsError):
        workflow.run_cases("unused", tmp_path / "cases.json", target)


def test_unexpected_external_failure_is_retained_and_next_case_runs(tmp_path, monkeypatch):
    target, result, calls = campaign(tmp_path, monkeypatch, raises=True)
    assert calls == ["first", "second"] and result["status"] == "failed"
    first = json.loads((target / "first/receipt.json").read_text())
    assert first["status"] == "failed" and "Synthetic external failure" in first["error"]


def test_missing_case_and_changed_evidence_cannot_pass(tmp_path, monkeypatch):
    target, result, _ = campaign(tmp_path, monkeypatch)
    assert result["status"] == "passed"
    missing = json.loads((target / "batch.json").read_text())
    del missing["cases"]["second"]
    assert workflow.evaluate_batch(target, missing)["status"] == "failed"
    (target / "first/raw.txt").write_text("Changed\n")
    assert workflow.evaluate_batch(target)["status"] == "failed"


def test_readiness_requires_numerics_matching_source_and_resolved_actions():
    comparison = dict(status="passed", reasons=[], source_sha256={"src/escape/model.py": "abc"})
    verification = dict(review_ready=True, changed_inputs=[],
                        selected_inputs_before={"src/escape/model.py": "abc"},
                        gates=[dict(id=name, status="passed") for name in workflow.NUMERICAL_GATES])
    sources = {"src/escape/model.py": "abc"}
    investigation = {"contracts": [{"model": "escape.cloudy_investigation",
                                    "investigation": {"pending_follow_ups": [{"id": "test_hypotheses"}]}}]}
    assert workflow.readiness(comparison, verification, investigation, sources, {"status": "passed"})["state"] == "pending_followups"
    investigation["contracts"][0]["investigation"]["pending_follow_ups"] = [{"id": "review_completion"}]
    assert workflow.readiness(comparison, verification, investigation, sources, {"status": "passed"})["state"] == "ready_for_review"
    assert workflow.readiness(comparison, verification, investigation, {}, {"status": "passed"})["state"] == "continue_verification"
    verification["gates"].pop()
    assert workflow.readiness(comparison, verification, investigation, sources, {"status": "passed"})["state"] == "continue_verification"
    comparison.update(status="failed", reasons=["n3 discrepancy"])
    assert workflow.readiness(comparison, verification, investigation, sources, {"status": "passed"})["state"] == "continue_diagnosis"


def test_reference_refinement_keeps_physics_and_artifact_identity(tmp_path, monkeypatch):
    import shutil
    coarse, _, _ = campaign(tmp_path, monkeypatch)
    fine = tmp_path / "fine"
    shutil.copytree(coarse, fine)
    batch = json.loads((fine / "batch.json").read_text())
    batch["resolution"] = .025
    for item in batch["cases"].values():
        path = fine / item["receipt"]
        receipt = json.loads(path.read_text())
        receipt["configuration"]["resolution"] = .025
        workflow.write_json(path, receipt)
        item["sha256"] = workflow.sha256(path)
    workflow.write_json(fine / "batch.json", batch)
    assert workflow.evaluate_reference_refinement(fine, coarse)["status"] == "passed"
    path = fine / "first/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["cloudy"]["n3_cm3"] = 1.03
    workflow.write_json(path, receipt)
    batch["cases"]["first"]["sha256"] = workflow.sha256(path)
    workflow.write_json(fine / "batch.json", batch)
    assert workflow.evaluate_reference_refinement(fine, coarse)["status"] == "failed"
