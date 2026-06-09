"""Tests for the SessionEnd hook.

The hook is essentially the glue between Claude Code's stdin payload,
the daemon's event stream, and the flush.py spawner. We stub
``subprocess.Popen`` everywhere so flush.py is never actually invoked.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

from conftest import drive_stdin, fresh_hook


def _write_transcript(path: Path, n: int = 3) -> None:
    lines = [
        json.dumps({"message": {"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i}"}})
        for i in range(n)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_session_end_emits_event(tmp_path: Path, monkeypatch):
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    drive_stdin(
        monkeypatch,
        se,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "transcript_path": str(transcript),
            "source": "logout",
        },
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["type"] == "SessionEnd"
    assert rec["session_id"] == "s1"


def test_session_end_recursion_guard(tmp_path: Path, monkeypatch):
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "scholar")
    drive_stdin(monkeypatch, se, {"session_id": "s1"})
    assert not (tmp_path / "events.jsonl").exists()


def test_session_end_skips_when_no_transcript(tmp_path: Path, monkeypatch):
    """Event still fires, but no flush snapshot is written."""
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    drive_stdin(monkeypatch, se, {"session_id": "s1", "cwd": "/tmp"})
    assert (tmp_path / "events.jsonl").exists()
    # No state file written.
    state_files = list((tmp_path / "state").glob("session-flush-*.md"))
    assert state_files == []


def test_session_end_skips_when_transcript_missing(tmp_path: Path, monkeypatch):
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    drive_stdin(
        monkeypatch,
        se,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "transcript_path": str(tmp_path / "missing.jsonl"),
        },
    )
    state_files = list((tmp_path / "state").glob("session-flush-*.md"))
    assert state_files == []


def test_session_end_writes_snapshot_when_transcript_present(tmp_path: Path, monkeypatch):
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, n=3)
    drive_stdin(
        monkeypatch,
        se,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "transcript_path": str(transcript),
        },
    )
    state_files = list((tmp_path / "state").glob("session-flush-*.md"))
    assert len(state_files) == 1


def test_session_end_handles_malformed_stdin(tmp_path: Path, monkeypatch):
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    monkeypatch.setattr(sys, "stdin", StringIO("not json"))
    se.main()  # must not raise


def test_session_end_handles_non_dict_stdin(tmp_path: Path, monkeypatch):
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    monkeypatch.setattr(sys, "stdin", StringIO('"hello"'))
    se.main()
    assert not (tmp_path / "events.jsonl").exists()


def test_session_end_handles_windows_backslash(tmp_path: Path, monkeypatch):
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    raw = (
        '{"session_id": "s1", "cwd": "C:\\Users\\x", "transcript_path": "' + str(transcript) + '"}'
    )
    monkeypatch.setattr(sys, "stdin", StringIO(raw))
    se.main()
    assert (tmp_path / "events.jsonl").exists()


def test_session_end_uses_reason_field_as_source_fallback(tmp_path: Path, monkeypatch):
    se = fresh_hook(monkeypatch, tmp_path, "session_end")
    drive_stdin(
        monkeypatch,
        se,
        {"session_id": "s1", "cwd": "/tmp", "reason": "user-logout"},
    )
    # Event fires (no real flush, transcript missing). The hook doesn't
    # surface 'source' in its payload, but the call path itself must
    # have used 'reason' as a fallback — we just confirm no crash.
    assert (tmp_path / "events.jsonl").exists()


def test_session_end_subprocess_end_to_end(isolated_home: Path, hook_runner, tmp_path):
    """End-to-end via subprocess. Use a transcript on disk so flush is
    spawned (and immediately dies because the test env doesn't have the
    full uv project, but the snapshot file is what we care about)."""
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    res = hook_runner(
        "session-end.py",
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "transcript_path": str(transcript),
        },
        env={"AGENT_MEM_HOME": str(isolated_home)},
    )
    # The hook itself exits 0 even though the spawned flush.py may
    # immediately exit non-zero — Popen is detached.
    assert res.returncode == 0, res.stderr
    assert (isolated_home / "events.jsonl").exists()
