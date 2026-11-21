"""Check retained external outputs and failed execution bookkeeping offline."""

from __future__ import annotations

import importlib.util
import gzip
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("cloudy_reference", ROOT / "tools/cloudy_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
EVIDENCE = ROOT / "mceg/evidence/cloudy_reference_20261121"
CONFIG = {"nmax": 5, "temperature_k": 9000., "density_cm3": 1e12,
          "radiation_temperature_k": 10000., "dilution": .025,
          "thickness_cm": 1e-4, "resolution": .05}


def test_parse_actual_reference_preserves_charge_output_precision():
    answer = reference.parse_cloudy(EVIDENCE, 5, "thin_nmax5_fine3")
    assert answer["n2_cm3"] == pytest.approx(131816.7)
    assert answer["n3_cm3"] == pytest.approx(20091.05)
    assert answer["electron_cm3"] == 8.22e11
    assert answer["proton_cm3_from_nucleus_conservation"] == pytest.approx(821550821619.55)
    assert "rounded" in answer["charge_source"]
    with pytest.raises(ValueError, match="finite atom"):
        reference.parse_cloudy(EVIDENCE, 4, "thin_nmax5_fine3")


def test_upstream_failure_is_not_a_reference():
    with pytest.raises(ValueError, match="successful completion"):
        reference.parse_cloudy(EVIDENCE, 5, "unmodified_nmax5")


def test_one_zone_reader_rejects_nonfinite_and_multiple_rows(tmp_path):
    saved = tmp_path / "saved.levels"
    saved.write_text("#depth n1\n1.0 nan\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        reference.one_row(saved)
    saved.write_text("#depth n1\n1.0 2.0\n2.0 3.0\n")
    with pytest.raises(ValueError, match="one final zone"):
        reference.one_row(saved)


def test_input_records_each_ablation_and_rejects_implicit_multiline_commands():
    text = reference.hydrogen_input(CONFIG, ["database H-like collisions l-mixing off"])
    assert "levels resolved 3" in text and "levels collapsed 2" in text
    assert text.endswith("database H-like collisions l-mixing off\n")
    assert '"e-" last' in text and '"H+" last' in text
    assert 'save line optical depths "model.line_tau" last' in text
    assert "model.continuum_tau" not in text
    with pytest.raises(ValueError, match="one line"):
        reference.hydrogen_input(CONFIG, ["command\nsecond command"])
    with pytest.raises(ValueError, match="collapsed shell"):
        reference.hydrogen_input({**CONFIG, "nmax": 3})


def test_failed_subprocess_is_retained_and_never_overwritten(tmp_path):
    executable = tmp_path / "cloudy-test"
    executable.write_text("#!/bin/sh\nprintf 'deliberate test failure\\n'\nexit 3\n")
    executable.chmod(0o755)
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"predeclared": true}\n')
    destination = tmp_path / "attempt"
    report = reference.run(executable, destination, CONFIG, protocol=protocol)
    assert report["status"] == "failed" and report["exit_code"] == 3
    assert report["command"] == [str(executable), "-r", "model"]
    assert report["changed_sources"] == []
    assert report["protocol"]["sha256"] == reference.sha256(protocol)
    assert (destination / "protocol.json").read_bytes() == protocol.read_bytes()
    assert "deliberate test failure" in (destination / "stdout.log").read_text()
    retained = (destination / "receipt.json").read_bytes()
    assert json.loads(retained)["artifacts"]["model.in"] == reference.sha256(destination / "model.in")
    with pytest.raises(FileExistsError):
        reference.run(executable, destination, CONFIG)
    assert (destination / "receipt.json").read_bytes() == retained


def test_replay_verifies_raw_bytes_and_does_not_reuse_old_escape_support(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    for suffix in ("out", "levels", "conditions"):
        data = (EVIDENCE / f"thin_nmax5_fine3.{suffix}").read_bytes()
        if suffix == "out":
            (old / "model.out.gz").write_bytes(gzip.compress(data, mtime=0))
        else:
            (old / f"model.{suffix}").write_bytes(data)
    old_receipt = {"status": "completed", "exit_code": 0, "configuration": CONFIG,
                   "executable_sha256": "a" * 64, "extra_cloudy_commands": [],
                   "cloudy": reference.parse_cloudy(old, 5),
                   "changed_sources": ["source_changed_during_old_escape_calculation.py"],
                   "review_ready": False,
                   "artifacts": {f"model.{suffix}": reference.artifact_sha256(old / f"model.{suffix}")
                                 for suffix in ("out", "levels", "conditions")}}
    reference.write_json(old / "receipt.json", old_receipt)
    original = (old / "receipt.json").read_bytes()
    result = reference.run(None, tmp_path / "new", CONFIG, reference_directory=old)
    assert result["status"] == "completed" and result["review_ready"]
    assert result["execution_mode"] == "retained_reference" and result["command"] is None
    assert result["reference_escape_reused"] is False and "escape" not in result
    assert (tmp_path / "new/reference/receipt.json").read_bytes() == original
    assert (old / "receipt.json").read_bytes() == original
    (old / "model.levels").write_text("corrupted\n")
    failed = reference.run(None, tmp_path / "bad", CONFIG, reference_directory=old)
    assert failed["status"] == "failed"
    assert "Reference artifact changed" in failed["error"]


def test_replay_rejects_a_different_physical_configuration(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    reference.write_json(old / "receipt.json", {
        "status": "completed", "exit_code": 0, "configuration": {**CONFIG, "temperature_k": 6000},
        "cloudy": {"n3_cm3": 1.},
    })
    result = reference.run(None, tmp_path / "new", CONFIG, reference_directory=old)
    assert result["status"] == "failed"
    assert "configuration" in result["error"]
