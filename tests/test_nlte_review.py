"""The external review must not hide a missing reader or an unhandled question."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "review_nlte_spectrum",
    Path(__file__).resolve().parents[1] / "tools/review_nlte_spectrum.py",
)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


@pytest.mark.parametrize("exit_codes", [(0, 0), (1, 0), (0, 1), (2, 2)])
def test_scoped_review_preserves_failures_and_checks_both_questions(
    tmp_path, monkeypatch, capsys, exit_codes,
):
    calls = []
    results = iter(exit_codes)

    def run(command, *, check):
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=next(results))

    monkeypatch.setattr(review.subprocess, "run", run)
    result = review.main(["--root", str(tmp_path), "--python", "external-python"])
    assert result == int(any(exit_codes))
    assert len(calls) == 2
    for command, assessment in zip(calls, review.ASSESSMENTS):
        assert command[:4] == ["external-python", "-m", "mceg", "check"]
        assert command[4] == str(tmp_path / "mceg/contracts/escape.nlte_spectrum.yaml")
        assert command[command.index("--assessment") + 1] == assessment
        assert command[command.index("--root") + 1] == str(tmp_path)
        assert "--require-handling" in command
    assert "Unknown scientific support remains unknown" in capsys.readouterr().out


def test_missing_external_interpreter_is_an_error(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("MCEG environment missing")

    monkeypatch.setattr(review.subprocess, "run", missing)
    with pytest.raises(SystemExit) as error:
        review.main(["--root", str(tmp_path)])
    assert error.value.code == 2
