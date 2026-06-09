"""`ultan doctor` must run cleanly in any environment state — it is the
diagnostic of last resort, so it can't itself crash on missing pieces."""

from pathlib import Path

import pytest

from ultan import _doctor


def test_doctor_runs_against_empty_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "agent-mem"
    home.mkdir()
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))

    rc = _doctor.run()
    out = capsys.readouterr().out

    # Dev venv has the retrieval stack installed, so an empty home is not
    # "broken" — just idle with nothing captured yet.
    assert rc == 0
    assert "daemon process: not running" in out
    assert "events.jsonl missing" in out
    assert "verdict: IDLE" in out


def test_doctor_reports_warming_when_pid_alive_but_no_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    home = tmp_path / "agent-mem"
    home.mkdir()
    (home / "daemon.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))

    rc = _doctor.run()
    out = capsys.readouterr().out

    assert rc == 0
    assert "WARMING UP" in out
    assert "verdict: WARMING" in out
