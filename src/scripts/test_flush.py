"""Behavioural regression tests for ``scripts/flush.py``.

Covers the deterministic, non-SDK units: flush-state round-trip and daily-log
append. ``run_flush`` / ``main`` are not exercised — they call the Claude
Agent SDK.
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
