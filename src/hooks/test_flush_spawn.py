"""Tests for the shared flush-spawn helper used by SessionEnd + PreCompact.

The helper does two things:

1. Walks a Claude Code JSONL transcript and pulls the last N
   conversation turns as Markdown.
2. Writes the snapshot to ``state_dir`` and spawns
   ``scripts/flush.py`` as a detached subprocess.

We stub out ``subprocess.Popen`` in every test so no flush.py is
actually invoked — that script needs the SDK and Anthropic API access.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import _flush_spawn  # noqa: E402


class _FakePopen:
    """Context-managerable stand-in for ``subprocess.Popen`` invoked by
    ``snapshot_and_spawn_flush``."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.returncode = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _shim_popen(monkeypatch):
    """Replace ``_flush_spawn.subprocess.Popen`` via a shim module —
    leaving the global ``subprocess`` intact so other Popen-based
    calls (subprocess.run, etc.) elsewhere keep working."""

    class _ShimSubprocess:
        Popen = staticmethod(_FakePopen)
        DEVNULL = _flush_spawn.subprocess.DEVNULL
        CREATE_NO_WINDOW = getattr(_flush_spawn.subprocess, "CREATE_NO_WINDOW", 0)

    monkeypatch.setattr(_flush_spawn, "subprocess", _ShimSubprocess)


def _write_transcript(path: Path, turns: list[dict]) -> None:
    """Write a JSONL transcript with the given turn dicts."""
    lines = [json.dumps(t) for t in turns]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_extract_picks_user_assistant_turns(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": "hi"}},
            {"message": {"role": "assistant", "content": "hello"}},
            {"message": {"role": "system", "content": "ignored"}},
            {"message": {"role": "user", "content": "bye"}},
        ],
    )
    md, count = _flush_spawn.extract_conversation_context(transcript)
    assert count == 3
    assert "**User:** hi" in md
    assert "**Assistant:** hello" in md
    assert "**User:** bye" in md
    assert "ignored" not in md


def test_extract_handles_content_blocks(tmp_path: Path):
    """Claude Code transcripts sometimes nest text into typed blocks."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "block-a"},
                        {"type": "image", "source": "..."},
                        "raw-string-block",
                    ],
                }
            },
        ],
    )
    md, _ = _flush_spawn.extract_conversation_context(transcript)
    assert "block-a" in md
    assert "raw-string-block" in md


def test_extract_handles_top_level_role(tmp_path: Path):
    """When ``message`` is present but not a dict (e.g. None), the
    helper falls back to top-level ``role``/``content`` fields."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": None, "role": "user", "content": "top-level"}],
    )
    md, count = _flush_spawn.extract_conversation_context(transcript)
    assert count == 1
    assert "top-level" in md


def test_extract_skips_malformed_lines(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                "not json",
                "",
                json.dumps({"message": {"role": "user", "content": "ok"}}),
            ]
        ),
        encoding="utf-8",
    )
    md, count = _flush_spawn.extract_conversation_context(transcript)
    assert count == 1
    assert "ok" in md


def test_extract_caps_at_max_turns(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"turn-{i}"}} for i in range(60)],
    )
    _, count = _flush_spawn.extract_conversation_context(transcript)
    assert count == _flush_spawn.MAX_TURNS


def test_extract_caps_at_max_chars(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    # One big turn that blows past MAX_CONTEXT_CHARS.
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": "x" * 30_000}}],
    )
    md, _ = _flush_spawn.extract_conversation_context(transcript)
    assert len(md) <= _flush_spawn.MAX_CONTEXT_CHARS


def test_spawn_skips_when_no_transcript_path(tmp_path: Path, monkeypatch):
    _shim_popen(monkeypatch)
    out = _flush_spawn.snapshot_and_spawn_flush(
        "",
        "s1",
        "proj",
        state_dir=tmp_path / "state",
        code_root=tmp_path,
        min_turns=1,
        file_prefix="x",
        log_tag="t",
    )
    assert out is None


def test_spawn_skips_when_transcript_missing(tmp_path: Path, monkeypatch):
    _shim_popen(monkeypatch)
    out = _flush_spawn.snapshot_and_spawn_flush(
        str(tmp_path / "nope.jsonl"),
        "s1",
        "proj",
        state_dir=tmp_path / "state",
        code_root=tmp_path,
        min_turns=1,
        file_prefix="x",
        log_tag="t",
    )
    assert out is None


def test_spawn_skips_when_below_min_turns(tmp_path: Path, monkeypatch):
    _shim_popen(monkeypatch)
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": "single"}}],
    )
    out = _flush_spawn.snapshot_and_spawn_flush(
        str(transcript),
        "s1",
        "proj",
        state_dir=tmp_path / "state",
        code_root=tmp_path,
        min_turns=5,
        file_prefix="x",
        log_tag="t",
    )
    assert out is None


def test_spawn_skips_when_context_empty(tmp_path: Path, monkeypatch):
    _shim_popen(monkeypatch)
    transcript = tmp_path / "t.jsonl"
    # All blank turns => empty context.
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": "   "}}],
    )
    out = _flush_spawn.snapshot_and_spawn_flush(
        str(transcript),
        "s1",
        "proj",
        state_dir=tmp_path / "state",
        code_root=tmp_path,
        min_turns=1,
        file_prefix="x",
        log_tag="t",
    )
    assert out is None


def test_spawn_writes_snapshot_and_calls_popen(tmp_path: Path, monkeypatch):
    calls: list[list] = []

    class _CapturingPopen(_FakePopen):
        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
            super().__init__(cmd, **kwargs)

    class _ShimSubprocess:
        Popen = staticmethod(_CapturingPopen)
        DEVNULL = _flush_spawn.subprocess.DEVNULL
        CREATE_NO_WINDOW = getattr(_flush_spawn.subprocess, "CREATE_NO_WINDOW", 0)

    monkeypatch.setattr(_flush_spawn, "subprocess", _ShimSubprocess)
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": "hi"}},
            {"message": {"role": "assistant", "content": "hello"}},
        ],
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    out = _flush_spawn.snapshot_and_spawn_flush(
        str(transcript),
        "s1",
        "myproj",
        state_dir=state_dir,
        code_root=tmp_path,
        min_turns=1,
        file_prefix="session-flush",
        log_tag="t",
    )
    assert out is not None
    assert out.exists()
    assert out.read_text(encoding="utf-8")  # non-empty
    # The cmd includes the context file as the first positional arg to flush.py.
    assert len(calls) == 1
    cmd = calls[0]
    assert str(out) in cmd
    assert "s1" in cmd
    assert "myproj" in cmd


def test_spawn_logs_when_extraction_raises(tmp_path: Path, monkeypatch, caplog):
    """A transcript that raises during extraction => the helper logs
    and returns None, never re-raises."""

    def boom(_path):
        raise RuntimeError("whoops")

    monkeypatch.setattr(_flush_spawn, "extract_conversation_context", boom)
    _shim_popen(monkeypatch)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("dummy", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        out = _flush_spawn.snapshot_and_spawn_flush(
            str(transcript),
            "s1",
            "proj",
            state_dir=tmp_path / "state",
            code_root=tmp_path,
            min_turns=1,
            file_prefix="x",
            log_tag="t",
        )
    assert out is None


def test_spawn_logs_when_popen_raises(tmp_path: Path, monkeypatch, caplog):
    def boom(*a, **kw):
        raise OSError("no exec")

    class _ShimSubprocess:
        Popen = staticmethod(boom)
        DEVNULL = _flush_spawn.subprocess.DEVNULL
        CREATE_NO_WINDOW = getattr(_flush_spawn.subprocess, "CREATE_NO_WINDOW", 0)

    monkeypatch.setattr(_flush_spawn, "subprocess", _ShimSubprocess)
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": "hi"}}],
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    with caplog.at_level(logging.ERROR):
        out = _flush_spawn.snapshot_and_spawn_flush(
            str(transcript),
            "s1",
            "proj",
            state_dir=state_dir,
            code_root=tmp_path,
            min_turns=1,
            file_prefix="x",
            log_tag="t",
        )
    # Snapshot file was written before Popen blew up, but the
    # helper still returns None — the spawn was the whole point.
    assert out is None


def test_spawn_logs_when_snapshot_write_fails(tmp_path: Path, monkeypatch):
    """A read-only state dir => snapshot write fails, helper returns None."""

    _shim_popen(monkeypatch)
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": "hi"}}],
    )

    # Make a state_dir Path whose write_text fails. Monkeypatch the
    # Path's write_text via a subclass-ish trick: replace
    # Path.write_text just for this call.
    orig = Path.write_text

    def failing_write(self, *args, **kwargs):
        if "state" in str(self):
            raise OSError("disk full")
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    out = _flush_spawn.snapshot_and_spawn_flush(
        str(transcript),
        "s1",
        "proj",
        state_dir=state_dir,
        code_root=tmp_path,
        min_turns=1,
        file_prefix="x",
        log_tag="t",
    )
    assert out is None
