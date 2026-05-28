"""Tests for the Stop hook — emit one turn-boundary event and exit."""

from __future__ import annotations

import importlib
import json
import sys
from io import StringIO
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def _fresh_stop(monkeypatch, home: Path):
    """Reload ``stop`` with a fresh AGENT_MEM_HOME so the event path
    resolves under ``home``."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    return importlib.import_module("stop")


def _drive_stop(monkeypatch, stop_mod, payload: dict) -> None:
    """Feed ``payload`` to ``stop.main()`` via a fake stdin."""
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(payload)))
    stop_mod.main()


def test_stop_appends_event(tmp_path: Path, monkeypatch):
    stop_mod = _fresh_stop(monkeypatch, tmp_path)
    _drive_stop(monkeypatch, stop_mod, {"session_id": "s1", "cwd": "/tmp"})
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 1
    rec = json.loads(events[0])
    assert rec["type"] == "Stop"
    assert rec["session_id"] == "s1"
    assert rec["payload"] == {}


def test_stop_recursion_guard(tmp_path: Path, monkeypatch):
    stop_mod = _fresh_stop(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "scholar")
    _drive_stop(monkeypatch, stop_mod, {"session_id": "s1"})
    assert not (tmp_path / "events.jsonl").exists()


def test_stop_handles_malformed_stdin(tmp_path: Path, monkeypatch):
    stop_mod = _fresh_stop(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("not json"))
    # Should not raise; should not write anything.
    stop_mod.main()
    assert not (tmp_path / "events.jsonl").exists()


def test_stop_handles_windows_backslash_input(tmp_path: Path, monkeypatch):
    """Lone backslashes in file paths — the re.sub fallback should
    rescue parsing."""
    stop_mod = _fresh_stop(monkeypatch, tmp_path)
    raw = '{"session_id": "s1", "cwd": "C:\\Users\\x"}'
    monkeypatch.setattr(sys, "stdin", StringIO(raw))
    stop_mod.main()
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["session_id"] == "s1"


def test_stop_rejects_non_dict_stdin(tmp_path: Path, monkeypatch):
    """A bare JSON list is parseable but not a dict — drop silently."""
    stop_mod = _fresh_stop(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("[1, 2, 3]"))
    stop_mod.main()
    assert not (tmp_path / "events.jsonl").exists()


def test_stop_subprocess_end_to_end(isolated_home: Path, hook_runner):
    """End-to-end via subprocess — the real entrypoint Claude Code uses."""
    res = hook_runner(
        "stop.py",
        {"session_id": "s1", "cwd": "/tmp"},
        env={"AGENT_MEM_HOME": str(isolated_home)},
    )
    assert res.returncode == 0, res.stderr
    assert (isolated_home / "events.jsonl").exists()
