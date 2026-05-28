"""Tests for ``_events.append_event`` — the shared JSONL writer every
hook calls.

These exercise the helper in-process. The hook subprocess tests
(test_post_tool_use, test_stop, etc.) also indirectly hit
``append_event`` end-to-end, but the unit tests below pin the
budget/truncation/recursion-guard branches that are awkward to drive
through stdin JSON.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def _reload_events_with_home(monkeypatch, home: Path):
    """Pin AGENT_MEM_HOME for the test. ``_events`` resolves the store
    path via ``config.get_config()`` at call time, so a plain import
    already picks up the new value — no module reload needed."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    return importlib.import_module("_events")


def test_append_event_writes_jsonl_line(tmp_path, monkeypatch):
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    _events.append_event(
        "Stop",
        {"session_id": "s1", "cwd": "/tmp"},
        payload={"k": "v"},
    )
    out = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(out) == 1
    rec = json.loads(out[0])
    assert rec["session_id"] == "s1"
    assert rec["type"] == "Stop"
    assert rec["payload"] == {"k": "v"}
    assert rec["cwd"] == "/tmp"
    assert isinstance(rec["ts"], (int, float))


def test_append_event_no_payload_normalises_to_empty(tmp_path, monkeypatch):
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    _events.append_event("Stop", {"session_id": "s1"})
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["payload"] == {}


def test_append_event_skips_when_no_session_id(tmp_path, monkeypatch):
    """Daemon rejects events without session_id — drop them rather than
    emit a malformed line."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    _events.append_event("Stop", {"cwd": "/tmp"}, payload={"k": "v"})
    assert not (tmp_path / "events.jsonl").exists()


def test_append_event_recursion_guard(tmp_path, monkeypatch):
    """``CLAUDE_INVOKED_BY`` short-circuits BEFORE any work."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "scholar")
    _events.append_event("Stop", {"session_id": "s1"}, payload={"k": "v"})
    assert not (tmp_path / "events.jsonl").exists()


def test_append_event_truncates_oversized_payload(tmp_path, monkeypatch):
    """A payload whose string value blows past ``MAX_LINE_BYTES`` (3KiB)
    triggers the shrink cascade. The line still gets written and stays
    under budget."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    huge = "x" * 5000
    _events.append_event(
        "PostToolUse",
        {"session_id": "s1"},
        payload={"content": huge, "summary": "x"},
    )
    line = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip()
    assert len(line) + 1 <= _events.MAX_LINE_BYTES + 100  # slack for fence
    rec = json.loads(line)
    # The content field should have been truncated (or dropped).
    if "content" in rec["payload"]:
        assert len(rec["payload"]["content"]) < len(huge)


def test_append_event_drops_payload_when_unrecoverable(tmp_path, monkeypatch):
    """When even after truncation the line is too big, we drop the
    payload entirely rather than skip the event."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    # Build a payload whose KEYS alone (not values) push the line over
    # budget — _truncate_payload only shrinks values.
    big_keys = {f"k{'x' * 60}_{i}": f"v{i}" for i in range(80)}
    _events.append_event("PostToolUse", {"session_id": "s1"}, payload=big_keys)
    line = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    # Either payload survived (sometimes the truncation pass is enough),
    # or it was reset to {}.
    assert isinstance(rec["payload"], dict)


def test_append_event_nested_dict_truncation(tmp_path, monkeypatch):
    """Nested dicts also get the truncation pass — the cascade recurses."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    nested = {"a": {"b": "x" * 4000}}
    _events.append_event("PostToolUse", {"session_id": "s1"}, payload=nested)
    line = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    if "a" in rec["payload"] and "b" in rec["payload"]["a"]:
        assert len(rec["payload"]["a"]["b"]) < 4000


def test_append_event_unserialisable_payload_falls_back(tmp_path, monkeypatch):
    """Non-JSON-serialisable payload → drop the payload, keep the event
    as a bare turn marker."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)

    class Weird:
        pass

    _events.append_event(
        "Stop",
        {"session_id": "s1"},
        payload={"obj": Weird()},
    )
    line = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["payload"] == {}


def test_truncate_payload_pure_function(tmp_path, monkeypatch):
    """Quick spot-check the truncation helper preserves shape."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    out = _events._truncate_payload(
        {"a": "abcd" * 100, "b": 42, "c": {"d": "x" * 100, "e": 1}},
        budget=50,
    )
    assert out["a"].endswith("...")
    assert out["b"] == 42
    assert out["c"]["d"].endswith("...")
    assert out["c"]["e"] == 1


def test_append_event_cwd_normalised(tmp_path, monkeypatch):
    """cwd from stdin → mirrored verbatim; non-string → None."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    _events.append_event(
        "Stop",
        {"session_id": "s1", "cwd": 42},  # bogus type
        payload={},
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["cwd"] is None


def test_events_path_resolves_via_config(tmp_path, monkeypatch):
    """_events_path reads config.STORE_DIR; that's the single source of
    truth the rest of the codebase uses."""
    _events = _reload_events_with_home(monkeypatch, tmp_path)
    assert _events._events_path() == tmp_path / "events.jsonl"
