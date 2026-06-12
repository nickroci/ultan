"""A daemon left running OLD code after an `ultan` tool update must be detected
and restarted — a long-running process keeps the code it loaded at spawn, so
`uv tool install` swapping the venv underneath it doesn't take effect until the
process is replaced. The version-aware restart lives in the session-start path
(NOT the per-turn hot path) and the mismatch is surfaced by `ultan doctor`.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ultan import _daemon, _doctor, _hooks


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "agent-mem"
    h.mkdir()
    monkeypatch.setenv("AGENT_MEM_HOME", str(h))
    return h


def _write_state(home: Path, *, pid: int, version: Any) -> None:
    state: dict[str, Any] = {"phase": "ready", "pid": pid, "since": "x"}
    if version is not None:
        state["version"] = version
    (home / "daemon.state").write_text(json.dumps(state), encoding="utf-8")


# ── restart_if_stale: only restarts on a genuine version mismatch ──────


def test_no_restart_when_no_state(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_daemon, "_installed_daemon_version", lambda: "0.3.0")
    assert _daemon.restart_if_stale() is False


def test_no_restart_when_pid_dead(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_state(home, pid=99999999, version="0.2.0")
    monkeypatch.setattr(_daemon, "_installed_daemon_version", lambda: "0.3.0")
    assert _daemon.restart_if_stale() is False


def test_no_restart_when_versions_match(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_state(home, pid=os.getpid(), version="0.3.0")
    monkeypatch.setattr(_daemon, "_installed_daemon_version", lambda: "0.3.0")
    stopped: list[int] = []
    monkeypatch.setattr(_daemon, "_stop_daemon", lambda pid, **k: stopped.append(pid))
    assert _daemon.restart_if_stale() is False
    assert stopped == []


def test_no_restart_when_installed_version_unknown(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_state(home, pid=os.getpid(), version="0.2.0")
    monkeypatch.setattr(_daemon, "_installed_daemon_version", lambda: None)
    stopped: list[int] = []
    monkeypatch.setattr(_daemon, "_stop_daemon", lambda pid, **k: stopped.append(pid))
    assert _daemon.restart_if_stale() is False
    assert stopped == []


def test_no_restart_when_running_version_missing(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A legacy daemon (pre-this-feature) wrote no version — never restart blindly.
    _write_state(home, pid=os.getpid(), version=None)
    monkeypatch.setattr(_daemon, "_installed_daemon_version", lambda: "0.3.0")
    assert _daemon.restart_if_stale() is False


def test_restart_when_stale_stops_and_clears_backoff(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_state(home, pid=os.getpid(), version="0.2.0")
    (home / ".daemon-spawn-attempt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(_daemon, "_installed_daemon_version", lambda: "0.3.0")
    stopped: list[int] = []
    monkeypatch.setattr(_daemon, "_stop_daemon", lambda pid, **k: stopped.append(pid))

    assert _daemon.restart_if_stale() is True
    assert stopped == [os.getpid()]  # stopped the stale daemon
    # backoff stamp cleared so ensure_running() respawns immediately, not throttled
    assert not (home / ".daemon-spawn-attempt").exists()


# ── _installed_daemon_version ─────────────────────────────────────────


def test_installed_version_matches_metadata_or_none() -> None:
    import importlib.metadata as md

    # Env-agnostic: in a full `ultan[retrieval]` install the daemon dist resolves;
    # in the thin root/CI venv (no extras) it doesn't and the helper returns None.
    # Either way it must agree with importlib.metadata — never invent a version.
    try:
        expected: str | None = md.version("agent-mem-daemon")
    except md.PackageNotFoundError:
        expected = None
    assert _daemon._installed_daemon_version() == expected


def test_installed_version_none_when_dist_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as md

    def _raise(_name: str) -> str:
        raise md.PackageNotFoundError

    monkeypatch.setattr(md, "version", _raise)
    assert _daemon._installed_daemon_version() is None


# ── session-start wires the restart in (before warming) ───────────────


def test_session_start_calls_restart_if_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        _hooks._daemon, "restart_if_stale", lambda: order.append("restart") or False
    )
    monkeypatch.setattr(_hooks._daemon, "ensure_running", lambda: order.append("ensure") or True)
    monkeypatch.setattr(_hooks._events, "append_event", lambda *a, **k: None)

    rc = _hooks.dispatch("session-start", {"session_id": "s", "source": "startup"})

    assert rc == 0
    assert order == ["restart", "ensure"]  # stale-check BEFORE warming


# ── doctor surfaces the mismatch even before the restart lands ─────────


def test_doctor_flags_stale_daemon(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_doctor, "_report_runtime", lambda home: False)
    (home / "daemon.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_state(home, pid=os.getpid(), version="0.2.0")
    monkeypatch.setattr(_daemon, "_installed_daemon_version", lambda: "0.3.0")

    _doctor.run()
    out = capsys.readouterr().out

    assert "daemon code: 0.2.0 — STALE (installed 0.3.0)" in out


def test_doctor_no_stale_line_when_versions_match(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_doctor, "_report_runtime", lambda home: False)
    (home / "daemon.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_state(home, pid=os.getpid(), version="0.3.0")
    monkeypatch.setattr(_daemon, "_installed_daemon_version", lambda: "0.3.0")

    _doctor.run()
    out = capsys.readouterr().out

    assert "daemon code: 0.3.0" in out
    assert "STALE" not in out
