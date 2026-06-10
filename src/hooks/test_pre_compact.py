"""Tests for the PreCompact hook (snapshot transcript before
auto-compaction)."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

from conftest import drive_stdin, fresh_hook


def _write_transcript(path: Path, n: int = 6) -> None:
    lines = [
        json.dumps({"message": {"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i}"}})
        for i in range(n)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_pre_compact_emits_session_end_event_with_source(tmp_path: Path, monkeypatch):
    """PreCompact reuses the SessionEnd event type but tags payload
    source=pre-compact so the daemon can distinguish."""
    pc = fresh_hook(monkeypatch, tmp_path, "pre_compact")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript)
    drive_stdin(
        monkeypatch,
        pc,
        {"session_id": "s1", "cwd": "/tmp", "transcript_path": str(transcript)},
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["type"] == "SessionEnd"
    assert rec["payload"]["source"] == "pre-compact"


def test_pre_compact_recursion_guard(tmp_path: Path, monkeypatch):
    pc = fresh_hook(monkeypatch, tmp_path, "pre_compact")
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "x")
    drive_stdin(monkeypatch, pc, {"session_id": "s1"})
    assert not (tmp_path / "events.jsonl").exists()


def test_pre_compact_skips_below_min_turns(tmp_path: Path, monkeypatch):
    """PreCompact min-turns threshold is 5 — a short transcript writes
    NO snapshot (but still emits the event)."""
    pc = fresh_hook(monkeypatch, tmp_path, "pre_compact")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, n=2)
    drive_stdin(
        monkeypatch,
        pc,
        {"session_id": "s1", "cwd": "/tmp", "transcript_path": str(transcript)},
    )
    # Event fires; no snapshot.
    assert (tmp_path / "events.jsonl").exists()
    state_files = list((tmp_path / "state").glob("flush-context-*.md"))
    assert state_files == []


def test_pre_compact_writes_snapshot_above_min_turns(tmp_path: Path, monkeypatch):
    pc = fresh_hook(monkeypatch, tmp_path, "pre_compact")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, n=10)
    drive_stdin(
        monkeypatch,
        pc,
        {"session_id": "s1", "cwd": "/tmp", "transcript_path": str(transcript)},
    )
    state_files = list((tmp_path / "state").glob("flush-context-*.md"))
    assert len(state_files) == 1


def test_pre_compact_handles_malformed_stdin(tmp_path: Path, monkeypatch):
    pc = fresh_hook(monkeypatch, tmp_path, "pre_compact")
    monkeypatch.setattr(sys, "stdin", StringIO("xxx"))
    pc.main()


def test_pre_compact_handles_windows_backslash(tmp_path: Path, monkeypatch):
    pc = fresh_hook(monkeypatch, tmp_path, "pre_compact")
    raw = '{"session_id": "s1", "cwd": "C:\\Users\\x"}'
    monkeypatch.setattr(sys, "stdin", StringIO(raw))
    pc.main()
    assert (tmp_path / "events.jsonl").exists()


def test_pre_compact_handles_non_dict_stdin(tmp_path: Path, monkeypatch):
    pc = fresh_hook(monkeypatch, tmp_path, "pre_compact")
    monkeypatch.setattr(sys, "stdin", StringIO("42"))
    pc.main()
    assert not (tmp_path / "events.jsonl").exists()


def test_pre_compact_subprocess_end_to_end(isolated_home: Path, hook_runner, tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, n=10)
    res = hook_runner(
        "pre-compact.py",
        {"session_id": "s1", "cwd": "/tmp", "transcript_path": str(transcript)},
        env={"AGENT_MEM_HOME": str(isolated_home)},
    )
    assert res.returncode == 0, res.stderr
    assert (isolated_home / "events.jsonl").exists()
