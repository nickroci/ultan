"""During daemon warmup the fallback must keep working WITH a note — the
agent should know priming/recall results are provisional, and `_daemon.status`
is the single source of that truth for every consumer."""

import json
import os
import socket
from pathlib import Path

import pytest

from ultan import _daemon, _hooks


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "agent-mem"
    h.mkdir()
    monkeypatch.setenv("AGENT_MEM_HOME", str(h))
    return h


# ── _daemon.status() ──────────────────────────────────────────────────


def test_status_down_with_no_signals(home: Path) -> None:
    assert _daemon.status() == "down"


def test_status_warming_from_state_flag_with_live_pid(home: Path) -> None:
    (home / "daemon.state").write_text(
        json.dumps({"phase": "warming", "pid": os.getpid(), "since": "x"}),
        encoding="utf-8",
    )
    assert _daemon.status() == "warming"


def test_status_ignores_stale_flag_with_dead_pid(home: Path) -> None:
    (home / "daemon.state").write_text(
        json.dumps({"phase": "warming", "pid": 99999999, "since": "x"}),
        encoding="utf-8",
    )
    assert _daemon.status() == "down"


def test_status_warming_from_fresh_spawn_stamp(home: Path) -> None:
    (home / ".daemon-spawn-attempt").write_text("x", encoding="utf-8")
    assert _daemon.status() == "warming"


def test_status_ready_when_socket_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    # AF_UNIX paths cap at ~104 bytes on macOS; pytest tmp dirs are too deep,
    # so bind in a short mkdtemp under /tmp instead.
    import shutil
    import tempfile

    short_home = Path(tempfile.mkdtemp(prefix="ultan-t-", dir="/tmp"))
    monkeypatch.setenv("AGENT_MEM_HOME", str(short_home))
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(str(short_home / "priming.sock"))
        srv.listen(1)
        assert _daemon.status() == "ready"
    finally:
        srv.close()
        shutil.rmtree(short_home, ignore_errors=True)


# ── hook priming carries the warming note ─────────────────────────────


def _dispatch_priming(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> str:
    monkeypatch.setattr(_hooks._daemon, "ensure_running", lambda: True)
    monkeypatch.setattr(_hooks._daemon, "status", lambda: status)
    monkeypatch.setattr(_hooks._priming, "get_priming", lambda *a, **k: "## bullets\n")
    rc = _hooks.dispatch("user-prompt-submit", {"session_id": "s", "prompt": "q"})
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    context: str = data["hookSpecificOutput"]["additionalContext"]
    return context


def test_priming_notes_lexical_fallback_while_warming(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _dispatch_priming(monkeypatch, capsys, "warming")
    assert "## bullets" in context  # the fallback still delivers
    assert "warming up" in context  # ...but says so


def test_priming_carries_no_note_when_ready(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _dispatch_priming(monkeypatch, capsys, "ready")
    assert "## bullets" in context
    assert "warming" not in context
