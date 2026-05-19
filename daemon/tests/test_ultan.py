"""Tests for the ``/ultan`` event-injection flow.

The script lives at tools/ultan/remember.py and uses stdlib only. We
exec it as a subprocess so the tests pin the actual CLI contract
(argument parsing, stdout shape, side effect on events.jsonl).

Coverage:
  - Default invocation writes a UserPromptSubmit + Stop pair to
    events.jsonl with ``user_asserted=True``.
  - --global flag suppresses the project slug.
  - --scope explicitly overrides.
  - Empty text exits with code 2.
  - Multiple invocations append (don't truncate).
  - The daemon's librarian flatten_buffer picks up the
    ``user_asserted=true`` flag from events the script wrote.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ULTAN_SCRIPT = (
    Path(__file__).resolve().parents[2] / "tools" / "ultan" / "remember.py"
)


def _run_ultan(args, *, env_home: Path, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["AGENT_MEM_HOME"] = str(env_home)
    return subprocess.run(
        [sys.executable, str(ULTAN_SCRIPT), *args],
        env=env,
        cwd=str(cwd or env_home),
        text=True,
        capture_output=True,
    )


def _read_events(env_home: Path) -> list[dict]:
    path = env_home / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_ultan_writes_user_asserted_event(tmp_path: Path):
    proc = _run_ultan(
        ["always wrap upstream errors at the boundary"],
        env_home=tmp_path,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "queued for librarian" in proc.stdout
    events = _read_events(tmp_path)
    # One UserPromptSubmit + one Stop.
    assert len(events) == 2
    ups = events[0]
    stop = events[1]
    assert ups["type"] == "UserPromptSubmit"
    assert stop["type"] == "Stop"
    assert ups["session_id"] == stop["session_id"]
    assert ups["payload"]["user_asserted"] is True
    assert ups["payload"]["text"] == "always wrap upstream errors at the boundary"
    assert ups["payload"]["role"] == "user"
    assert stop["payload"]["user_asserted"] is True


def test_ultan_global_flag_clears_scope(tmp_path: Path):
    proc = _run_ultan(
        ["--global", "always wrap errors"],
        env_home=tmp_path,
        cwd=tmp_path,
    )
    assert proc.returncode == 0
    events = _read_events(tmp_path)
    ups = events[0]
    assert ups["payload"]["project_slug"] is None
    assert "[global]" in proc.stdout


def test_ultan_explicit_scope(tmp_path: Path):
    proc = _run_ultan(
        ["--scope", "my-cool-project", "always wrap errors"],
        env_home=tmp_path,
        cwd=tmp_path,
    )
    assert proc.returncode == 0
    events = _read_events(tmp_path)
    ups = events[0]
    assert ups["payload"]["project_slug"] == "my-cool-project"
    assert "[project:my-cool-project]" in proc.stdout


def test_ultan_empty_text_returns_2(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, str(ULTAN_SCRIPT), "   "],
        env={**os.environ, "AGENT_MEM_HOME": str(tmp_path)},
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
        input="",
    )
    assert proc.returncode == 2


def test_ultan_appends_does_not_truncate(tmp_path: Path):
    _run_ultan(["first memory"], env_home=tmp_path, cwd=tmp_path)
    _run_ultan(["second memory"], env_home=tmp_path, cwd=tmp_path)
    events = _read_events(tmp_path)
    # Two invocations × 2 events each.
    assert len(events) == 4
    user_texts = [ev["payload"]["text"] for ev in events if ev["type"] == "UserPromptSubmit"]
    assert user_texts == ["first memory", "second memory"]


def test_ultan_event_flows_through_librarian_flatten(tmp_path: Path):
    """End-to-end: the events written by /ultan are picked up correctly
    by the daemon's buffer-flattening path with user_asserted=True."""
    _run_ultan(["always use uv to install deps"], env_home=tmp_path, cwd=tmp_path)
    events = _read_events(tmp_path)

    # Synthesise a snapshot the same way the daemon's RollingBuffer
    # would: one turn containing the UserPromptSubmit + Stop.
    snap = {
        "session_id": events[0]["session_id"],
        "cwd": events[0].get("cwd"),
        "turns": [{
            "started_at": 1.0, "sealed_at": 2.0,
            "events": [
                {"ts": 1.0, "type": ev["type"], "cwd": ev.get("cwd"),
                 "payload": ev["payload"]}
                for ev in events
            ],
        }],
    }
    from agent_mem_daemon import librarian_prompt as lp
    flat = lp.flatten_buffer(snap)
    assert len(flat) == 1
    _tid, role, text, user_asserted = flat[0]
    assert role == "user"
    assert text == "always use uv to install deps"
    assert user_asserted is True
    rendered = lp.format_rolling_buffer(flat)
    assert "[USER-ASSERTED]" in rendered
