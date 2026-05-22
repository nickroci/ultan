"""Tests for the SessionEnd hook.

The hook is essentially the glue between Claude Code's stdin payload,
the daemon's event stream, and the flush.py spawner. We stub
``subprocess.Popen`` everywhere so flush.py is never actually invoked.
"""

from __future__ import annotations

import importlib
import json
import sys
from io import StringIO
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` calls inside ``_flush_spawn``.

    Real ``subprocess.run`` (used by ``scope._git_remote_slug`` to read
    the git remote) goes through ``Popen`` too, so we can't replace the
    module-level symbol. Instead the test fixture swaps the ``Popen``
    binding on the ``_flush_spawn`` module specifically — its
    ``snapshot_and_spawn_flush`` resolves the name through that
    module's globals."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.returncode = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fresh_session_end(monkeypatch, home: Path):
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    for mod in ("config", "_events", "_flush_spawn", "session_end"):
        if mod in sys.modules:
            del sys.modules[mod]
    se = importlib.import_module("session_end")
    # Stub the Popen binding on the helper module only.
    import _flush_spawn

    # Wrap the subprocess module so only Popen is overridden, real
    # subprocess.run still works for the git-remote probe upstream.
    class _ShimSubprocess:
        Popen = staticmethod(_FakePopen)
        DEVNULL = _flush_spawn.subprocess.DEVNULL
        CREATE_NO_WINDOW = getattr(_flush_spawn.subprocess, "CREATE_NO_WINDOW", 0)

    monkeypatch.setattr(_flush_spawn, "subprocess", _ShimSubprocess)
    return se


def _drive(monkeypatch, mod, payload: dict) -> None:
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(payload)))
    mod.main()


def _write_transcript(path: Path, n: int = 3) -> None:
    lines = [
        json.dumps({"message": {"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i}"}})
        for i in range(n)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_session_end_emits_event(tmp_path: Path, monkeypatch):
    se = _fresh_session_end(monkeypatch, tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    _drive(
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
    se = _fresh_session_end(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "scholar")
    _drive(monkeypatch, se, {"session_id": "s1"})
    assert not (tmp_path / "events.jsonl").exists()


def test_session_end_skips_when_no_transcript(tmp_path: Path, monkeypatch):
    """Event still fires, but no flush snapshot is written."""
    se = _fresh_session_end(monkeypatch, tmp_path)
    _drive(monkeypatch, se, {"session_id": "s1", "cwd": "/tmp"})
    assert (tmp_path / "events.jsonl").exists()
    # No state file written.
    state_files = list((tmp_path / "state").glob("session-flush-*.md"))
    assert state_files == []


def test_session_end_skips_when_transcript_missing(tmp_path: Path, monkeypatch):
    se = _fresh_session_end(monkeypatch, tmp_path)
    _drive(
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
    se = _fresh_session_end(monkeypatch, tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, n=3)
    _drive(
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
    se = _fresh_session_end(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("not json"))
    se.main()  # must not raise


def test_session_end_handles_non_dict_stdin(tmp_path: Path, monkeypatch):
    se = _fresh_session_end(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO('"hello"'))
    se.main()
    assert not (tmp_path / "events.jsonl").exists()


def test_session_end_handles_windows_backslash(tmp_path: Path, monkeypatch):
    se = _fresh_session_end(monkeypatch, tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    raw = (
        '{"session_id": "s1", "cwd": "C:\\Users\\x", "transcript_path": "' + str(transcript) + '"}'
    )
    monkeypatch.setattr(sys, "stdin", StringIO(raw))
    se.main()
    assert (tmp_path / "events.jsonl").exists()


def test_session_end_uses_reason_field_as_source_fallback(tmp_path: Path, monkeypatch):
    se = _fresh_session_end(monkeypatch, tmp_path)
    _drive(
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
