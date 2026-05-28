"""Behavioural regression tests for ``scripts/flush.py``.

Covers the deterministic, non-SDK units: flush-state round-trip, daily-log
append, and the time-gated compile trigger. ``run_flush`` / ``main`` are
not exercised — they call the Claude Agent SDK.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import flush

# Importing flush sets ``CLAUDE_INVOKED_BY`` at module top (its recursion
# guard for when it runs as a real subprocess). That env mutation is
# process-global and would otherwise leak into *other* test modules — e.g.
# hooks/_events.append_event short-circuits on it — so undo it here, right
# after import, before any test runs.
os.environ.pop("CLAUDE_INVOKED_BY", None)


# ── flush-state round-trip ───────────────────────────────────────────


def test_load_flush_state_empty_when_absent(store_home: Path) -> None:
    assert flush.load_flush_state() == {}


def test_save_then_load_flush_state_round_trips(store_home: Path) -> None:
    flush.save_flush_state({"session_id": "sess-1", "timestamp": 1234.5})
    assert (store_home / "state" / "last-flush.json").exists()
    assert flush.load_flush_state() == {"session_id": "sess-1", "timestamp": 1234.5}


def test_load_flush_state_tolerates_corrupt_json(store_home: Path) -> None:
    state_file = store_home / "state" / "last-flush.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("{not json", encoding="utf-8")
    assert flush.load_flush_state() == {}


# ── append_to_daily_log ──────────────────────────────────────────────


def _today_log(home: Path) -> Path:
    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    return home / "daily" / f"{today}.md"


def test_append_creates_today_log_with_scaffold(store_home: Path) -> None:
    flush.append_to_daily_log("first body", "Session", "myproj")
    log = _today_log(store_home)
    assert log.exists()
    text = log.read_text(encoding="utf-8")
    assert text.startswith("# Daily Log:")
    assert "## Sessions" in text
    assert "### Session" in text
    assert "project:myproj" in text
    assert "first body" in text


def test_append_appends_to_existing_log(store_home: Path) -> None:
    flush.append_to_daily_log("body one", "Session", "proj")
    flush.append_to_daily_log("body two", "Memory Flush", "proj")
    text = _today_log(store_home).read_text(encoding="utf-8")
    # Scaffold written exactly once; both tagged sections present.
    assert text.count("# Daily Log:") == 1
    assert "body one" in text and "body two" in text
    assert "### Session" in text and "### Memory Flush" in text


# ── maybe_trigger_compilation ────────────────────────────────────────


class _FrozenDatetime(datetime):
    """datetime whose ``now()`` returns a fixed instant."""

    _frozen: datetime

    @classmethod
    def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
        return cls._frozen


def _freeze_hour(monkeypatch, hour: int) -> None:
    # Build the instant in the *local* zone so ``.astimezone().hour`` (what
    # maybe_trigger_compilation reads) is exactly ``hour`` regardless of the
    # machine's UTC offset.
    local_tz = datetime.now(timezone.utc).astimezone().tzinfo
    frozen = datetime(2026, 5, 28, hour, 0, 0, tzinfo=local_tz)
    _FrozenDatetime._frozen = frozen
    monkeypatch.setattr(flush, "datetime", _FrozenDatetime)


def test_maybe_trigger_noop_before_compile_hour(store_home: Path, monkeypatch) -> None:
    _freeze_hour(monkeypatch, flush.COMPILE_AFTER_HOUR - 1)

    spawned: list[object] = []
    monkeypatch.setattr(flush.subprocess, "Popen", lambda *a, **k: spawned.append((a, k)))
    flush.maybe_trigger_compilation()
    assert spawned == []


def test_maybe_trigger_spawns_after_compile_hour(store_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(flush, "COMPILE_AFTER_HOUR", 0)
    _freeze_hour(monkeypatch, 1)  # past the (lowered) compile hour

    spawned: list[tuple] = []

    def _fake_popen(cmd, *args, **kwargs):
        spawned.append((cmd, kwargs))

    monkeypatch.setattr(flush.subprocess, "Popen", _fake_popen)
    flush.maybe_trigger_compilation()

    assert len(spawned) == 1
    cmd = spawned[0][0]
    assert cmd[0] == "uv" and "python" in cmd
    assert str(flush.COMPILE_SCRIPT) in cmd


def test_maybe_trigger_skips_when_today_log_unchanged(store_home: Path, monkeypatch) -> None:
    import hashlib
    import json

    monkeypatch.setattr(flush, "COMPILE_AFTER_HOUR", 0)
    _freeze_hour(monkeypatch, 1)

    today = flush.get_config()
    # _freeze_hour pins the clock to 2026-05-28, so that's the log name the
    # function will look for.
    log_name = "2026-05-28.md"
    today.daily_dir.mkdir(parents=True, exist_ok=True)
    log_path = today.daily_dir / log_name
    log_path.write_text("compiled already\n", encoding="utf-8")
    digest = hashlib.sha256(log_path.read_bytes()).hexdigest()[:16]

    today.state_dir.mkdir(parents=True, exist_ok=True)
    today.state_file.write_text(
        json.dumps({"ingested": {log_name: {"hash": digest, "compiled_at": "t", "cost_usd": 0.0}}}),
        encoding="utf-8",
    )

    spawned: list[object] = []
    monkeypatch.setattr(flush.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    flush.maybe_trigger_compilation()
    assert spawned == []
