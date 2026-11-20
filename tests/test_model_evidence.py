"""Check saved numerical evidence without importing MCEG or parsing YAML."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def test_population_evidence_matches_sources_and_recorded_outcomes():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "mceg/evidence/hydrogen_populations_20261117"
    receipt = json.loads((evidence / "receipt.json").read_text())

    for path, expected in {
        **receipt["selected_files"], **receipt["artifacts"],
    }.items():
        assert sha256((root / path).read_bytes()).hexdigest() == expected, (
            f"Evidence changed: {path}. Review and renew the recorded run; "
            "do not refresh hashes alone."
        )

    cases = list(ET.parse(evidence / "tests.xml").iter("testcase"))
    outcomes = {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }
    assert outcomes == receipt["outcomes"]
    assert outcomes == {"tests": 6, "failures": 0, "errors": 0, "skipped": 0}
    assert receipt["exit_code"] == 0
    assert receipt["selected_files_match_git_revision"]
    assert receipt["selected_files_unchanged_during_run"]
    assert receipt["mceg_installed_in_test_environment"] is False
    test_ids = [
        f"{case.attrib['classname'].replace('.', '/')}.py::{case.attrib['name']}"
        for case in cases
    ]
    assert test_ids == receipt["test_ids"]
