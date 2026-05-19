"""JSONL tail behaviour: append, rotation, truncation, partial lines."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from agent_mem_daemon.buffer import Event
from agent_mem_daemon.ingest import JsonlTailer, parse_event_line


# ---- parser -------------------------------------------------------


def test_parse_full_event():
    line = json.dumps({
        "ts": 1234567890.0,
        "session_id": "s1",
        "type": "PostToolUse",
        "cwd": "/repo",
        "payload": {"tool": "Read"},
    })
    ev = parse_event_line(line)
    assert ev is not None
    assert ev.session_id == "s1"
    assert ev.type == "PostToolUse"
    assert ev.cwd == "/repo"
    assert ev.payload == {"tool": "Read"}
    assert ev.ts == 1234567890.0


def test_parse_iso_timestamp():
    line = json.dumps({
        "ts": "2026-05-19T10:30:00Z",
        "session_id": "s1",
        "type": "Stop",
    })
    ev = parse_event_line(line)
    assert ev is not None
    assert ev.ts > 0  # parsed something


def test_parse_missing_required_returns_none(caplog):
    # No session_id.
    line = json.dumps({"ts": 1.0, "type": "Stop"})
    assert parse_event_line(line) is None
    # No type.
    line = json.dumps({"ts": 1.0, "session_id": "s1"})
    assert parse_event_line(line) is None


def test_parse_missing_ts_falls_back_to_receipt(caplog):
    line = json.dumps({"session_id": "s1", "type": "Stop"})
    ev = parse_event_line(line, now_fn=lambda: 42.0)
    assert ev is not None
    assert ev.ts == 42.0


def test_parse_bad_json_returns_none():
    assert parse_event_line("not json") is None
    assert parse_event_line("") is None
    assert parse_event_line("   ") is None


def test_parse_non_object_returns_none():
    assert parse_event_line(json.dumps([1, 2, 3])) is None
    assert parse_event_line(json.dumps("just a string")) is None


# ---- tail --------------------------------------------------------


@pytest.fixture
def events_file(tmp_path) -> Path:
    return tmp_path / "events.jsonl"


def _append(path: Path, obj: dict) -> None:
    """Append one JSONL line. Open/close per call so the tailer sees
    durable bytes — mirrors what the hook will do."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _run_one_poll(tailer: JsonlTailer) -> list[Event]:
    seen: list[Event] = []
    tailer.on_event = lambda e: seen.append(e)
    tailer.poll_once()
    return seen


def test_tail_picks_up_appended_lines(events_file):
    # Pre-create empty file. start_from_end=False so we don't skip.
    events_file.touch()
    seen: list[Event] = []
    tailer = JsonlTailer(events_file, seen.append, start_from_end=False)
    tailer.poll_once()  # attach
    assert seen == []

    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "PostToolUse"})
    _append(events_file, {"ts": 2.0, "session_id": "s1", "type": "Stop"})

    tailer.poll_once()
    assert [e.type for e in seen] == ["PostToolUse", "Stop"]
    assert all(e.session_id == "s1" for e in seen)


def test_tail_picks_up_lines_added_after_attach(events_file):
    """Even if we attach with start_from_end=True (the daemon default),
    we still see lines written after attach."""
    events_file.touch()
    seen: list[Event] = []
    tailer = JsonlTailer(events_file, seen.append, start_from_end=True)
    tailer.poll_once()  # attach at EOF

    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "Stop"})
    tailer.poll_once()
    assert len(seen) == 1
    assert seen[0].type == "Stop"


def test_tail_start_from_end_skips_history(events_file):
    """If the file already has content at attach time and we asked to
    start from the end, the pre-existing content must not be replayed."""
    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "PostToolUse"})
    _append(events_file, {"ts": 2.0, "session_id": "s1", "type": "Stop"})

    seen: list[Event] = []
    tailer = JsonlTailer(events_file, seen.append, start_from_end=True)
    tailer.poll_once()
    assert seen == []  # we attached at EOF; nothing to see

    _append(events_file, {"ts": 3.0, "session_id": "s2", "type": "Stop"})
    tailer.poll_once()
    assert [e.session_id for e in seen] == ["s2"]


def test_tail_default_reads_existing_file_from_start(events_file):
    """Regression for the cold-start race: a fresh daemon attaching to
    an events.jsonl that *already has events* must process them, not
    skip to EOF. This is the bug that ate the user's /ultan events."""
    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "UserPromptSubmit"})
    _append(events_file, {"ts": 2.0, "session_id": "s1", "type": "Stop"})
    _append(events_file, {"ts": 3.0, "session_id": "s2", "type": "Stop"})

    seen: list[Event] = []
    # Default (no offset_state_path, start_from_end defaults to False).
    tailer = JsonlTailer(events_file, seen.append)
    tailer.poll_once()
    assert [e.session_id for e in seen] == ["s1", "s1", "s2"], (
        f"expected all 3 pre-existing events, got {[e.session_id for e in seen]}"
    )


def test_tail_resumes_from_persisted_offset(tmp_path: Path, events_file):
    """A daemon restart with a persisted offset must resume from there,
    not re-process old events and not skip new ones."""
    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "Stop"})
    _append(events_file, {"ts": 2.0, "session_id": "s2", "type": "Stop"})

    offset_state = tmp_path / "daemon.offset.json"

    # First daemon: reads both events and persists offset at EOF.
    seen1: list[Event] = []
    t1 = JsonlTailer(events_file, seen1.append, offset_state_path=offset_state)
    t1.poll_once()
    assert [e.session_id for e in seen1] == ["s1", "s2"]
    assert offset_state.exists(), "offset file should be written"

    # Append a new event, then start a fresh daemon. It must resume
    # from the persisted offset — neither replay old events nor miss
    # the new one.
    _append(events_file, {"ts": 3.0, "session_id": "s3", "type": "Stop"})
    seen2: list[Event] = []
    t2 = JsonlTailer(events_file, seen2.append, offset_state_path=offset_state)
    t2.poll_once()
    assert [e.session_id for e in seen2] == ["s3"], (
        f"fresh daemon with persisted offset should resume; got {[e.session_id for e in seen2]}"
    )


def test_tail_ignores_stale_offset_state(tmp_path: Path, events_file):
    """If the persisted offset is past the current file size (truncation,
    rotation, manual edit), fall back to reading from the start."""
    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "Stop"})

    # Plant an obviously stale offset (points past EOF).
    offset_state = tmp_path / "daemon.offset.json"
    offset_state.write_text(
        '{"path": "x", "inode": 99999, "dev": 99999, "offset": 999999, '
        '"last_size": 999999, "last_mtime_ns": 0}',
        encoding="utf-8",
    )

    seen: list[Event] = []
    t = JsonlTailer(events_file, seen.append, offset_state_path=offset_state)
    t.poll_once()
    assert [e.session_id for e in seen] == ["s1"]


def test_tail_handles_file_not_existing_yet(events_file):
    """The daemon may come up before the first hook fires. The tailer
    should poll without crashing and pick up the file when it appears."""
    seen: list[Event] = []
    tailer = JsonlTailer(events_file, seen.append, start_from_end=False)
    # File doesn't exist.
    assert not events_file.exists()
    tailer.poll_once()  # must not raise
    assert seen == []

    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "Stop"})
    tailer.poll_once()
    assert len(seen) == 1


def test_tail_handles_rotation(events_file, tmp_path):
    """File renamed away + recreated => tailer must re-open from start
    and pick up new content. Detected via inode change."""
    events_file.touch()
    seen: list[Event] = []
    tailer = JsonlTailer(events_file, seen.append, start_from_end=False)
    tailer.poll_once()
    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "Stop"})
    tailer.poll_once()
    assert len(seen) == 1

    # Rotate: move the file aside and recreate.
    rotated = tmp_path / "events.jsonl.1"
    shutil.move(str(events_file), str(rotated))
    events_file.touch()
    _append(events_file, {"ts": 2.0, "session_id": "s2", "type": "Stop"})

    tailer.poll_once()
    # The newly-written event in the rotated-in file should be seen.
    assert any(e.session_id == "s2" for e in seen)


def test_tail_handles_truncation(events_file):
    """File truncated in place (size < our offset) => re-open from
    start. The hook author might do this if they use a `>` redirect to
    reset the log."""
    events_file.touch()
    seen: list[Event] = []
    tailer = JsonlTailer(events_file, seen.append, start_from_end=False)
    tailer.poll_once()
    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "Stop"})
    tailer.poll_once()
    assert len(seen) == 1
    seen.clear()

    # Truncate.
    with open(events_file, "w") as f:
        f.truncate(0)

    # New content after truncation.
    _append(events_file, {"ts": 2.0, "session_id": "s2", "type": "Stop"})
    tailer.poll_once()
    assert [e.session_id for e in seen] == ["s2"]


def test_tail_handles_partial_line(events_file):
    """Half a line on one poll, the rest on the next => one event."""
    events_file.touch()
    seen: list[Event] = []
    tailer = JsonlTailer(events_file, seen.append, start_from_end=False)
    tailer.poll_once()  # attach

    obj = {"ts": 1.0, "session_id": "s1", "type": "Stop"}
    line = json.dumps(obj)
    # Write the first half, no newline.
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(line[:5])
    tailer.poll_once()
    assert seen == []  # partial — buffered

    with open(events_file, "a", encoding="utf-8") as f:
        f.write(line[5:] + "\n")
    tailer.poll_once()
    assert len(seen) == 1
    assert seen[0].session_id == "s1"


def test_tail_skips_bad_json_without_crashing(events_file):
    events_file.touch()
    seen: list[Event] = []
    tailer = JsonlTailer(events_file, seen.append, start_from_end=False)
    tailer.poll_once()

    with open(events_file, "a", encoding="utf-8") as f:
        f.write("this is not json\n")
        f.write(json.dumps({"ts": 1.0, "session_id": "s1", "type": "Stop"}) + "\n")
        f.write('{"incomplete":\n')   # malformed JSON on its own line
        f.write(json.dumps({"ts": 2.0, "session_id": "s2", "type": "Stop"}) + "\n")

    tailer.poll_once()
    assert [e.session_id for e in seen] == ["s1", "s2"]


def test_tail_callback_exception_does_not_break_loop(events_file):
    """A buggy on_event must not poison subsequent events."""
    events_file.touch()
    seen: list[Event] = []
    n_calls = {"v": 0}

    def cb(ev):
        n_calls["v"] += 1
        if n_calls["v"] == 1:
            raise RuntimeError("first call boom")
        seen.append(ev)

    tailer = JsonlTailer(events_file, cb, start_from_end=False)
    tailer.poll_once()
    _append(events_file, {"ts": 1.0, "session_id": "s1", "type": "Stop"})
    _append(events_file, {"ts": 2.0, "session_id": "s2", "type": "Stop"})

    tailer.poll_once()
    assert n_calls["v"] == 2
    assert [e.session_id for e in seen] == ["s2"]
