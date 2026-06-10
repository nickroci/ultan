"""`ultan doctor` must run cleanly in any environment state — it is the
diagnostic of last resort, so it can't itself crash on missing pieces.

The runtime-deps section (`_report_runtime`) is stubbed to "not broken" in
the state-classification tests: whether the [retrieval] extra is installed
depends on which env runs the suite (full dev venv vs the root CI job's thin
env), and these tests pin the daemon-state verdicts, not the dep probe.
"""

from pathlib import Path

import pytest

from ultan import _doctor


@pytest.fixture()
def _runtime_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_doctor, "_report_runtime", lambda home: False)


def test_doctor_runs_against_empty_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _runtime_ok: None,
) -> None:
    home = tmp_path / "agent-mem"
    home.mkdir()
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))

    rc = _doctor.run()
    out = capsys.readouterr().out

    assert rc == 0
    assert "daemon process: not running" in out
    assert "events.jsonl missing" in out
    assert "verdict: IDLE" in out


def test_doctor_reports_warming_when_pid_alive_but_no_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _runtime_ok: None,
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


def test_doctor_reports_broken_when_runtime_deps_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(_doctor, "_report_runtime", lambda home: True)
    home = tmp_path / "agent-mem"
    home.mkdir()
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))

    rc = _doctor.run()
    out = capsys.readouterr().out

    assert rc == 1
    assert "verdict: BROKEN" in out
