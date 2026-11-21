"""Prevent numerical evidence from accepting invalid or incomplete comparisons."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "verify_nlte_spectrum", Path(__file__).resolve().parents[1] / "tools/verify_nlte_spectrum.py"
)
verification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verification)


def test_nonfinite_predictions_and_invalid_errors_cannot_pass():
    for candidate in (np.array([np.nan]), np.array([np.inf])):
        assert not verification.uncertainty_scaled_difference([0.0], candidate, [1.0])["passed"]
    for uncertainty in ([0.0], [-1.0], [np.nan]):
        with pytest.raises(ValueError, match="finite and positive"):
            verification.uncertainty_scaled_difference([0.0], [0.0], uncertainty)
    with pytest.raises(ValueError, match="identical shapes"):
        verification.uncertainty_scaled_difference([0.0], [0.0, 1.0], [1.0])


def test_observation_pixels_cannot_be_silently_extrapolated():
    verification.require_observer_coverage([2.0, 1.0, 0.0], [0.0, 2.0])
    for grid, pixels in (([2.0, 1.0, 0.0], [-0.1]), ([0.0, 1.0, 2.0], [1.0])):
        with pytest.raises(ValueError, match="cover every observation pixel"):
            verification.require_observer_coverage(grid, pixels)


@pytest.mark.parametrize("worker_exit,changed,expected", [(9, False, 9), (0, True, 1)])
def test_execution_keeps_worker_failure_and_source_changes(tmp_path, monkeypatch,
                                                         worker_exit, changed, expected):
    record = tmp_path / "evidence"
    snapshots = iter([{"source": "before"}, {"source": "after" if changed else "before"}])
    monkeypatch.setattr(verification, "source_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(verification.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=worker_exit))
    monkeypatch.setattr(verification.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(verification.sys, "argv", ["verify_nlte_spectrum.py", "--record", str(record)])
    assert verification.main() == expected
    execution = json.loads((record / "execution.json").read_text())
    assert execution["worker_exit_code"] == worker_exit
    assert execution["source_unchanged"] is not changed
    assert not execution["passed"]
    with pytest.raises(FileExistsError):
        verification.main()
