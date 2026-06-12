"""Assistant-prose capture: transcript parsing + buffer injection.

Two layers:

1. Unit tests for ``transcript.py`` — the faithful port of the legacy
   ``_flush_spawn`` markdown-aware text extraction, plus the incremental
   per-session dedup marker and the fail-soft / cap guards.
2. An integration test driving ``Scheduler.on_event`` with a synthetic
   Stop event that carries ``transcript_path``, proving the assistant's
   prose reaches the flattened buffer the Librarian sees (rendered with
   the distinct ``assistant-prose`` role).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent_mem_daemon import transcript as tr
from agent_mem_daemon.buffer import Event, RollingBuffer
from agent_mem_daemon.librarian_prompt import flatten_buffer
from agent_mem_daemon.scheduler import ASSISTANT_PROSE_ROLE, Scheduler

# ── fixtures / helpers ───────────────────────────────────────────────


def _line(role: str, content: Any, *, uuid: str) -> str:
    """One Claude Code transcript JSONL line (wrapped-message shape)."""
    return json.dumps({"uuid": uuid, "message": {"role": role, "content": content}})


def _write_transcript(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_use_block(name: str) -> Dict[str, Any]:
    return {"type": "tool_use", "name": name, "input": {"file": "x.py"}}


# ── transcript parsing ───────────────────────────────────────────────


def test_iter_assistant_prose_keeps_only_assistant_text(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    _write_transcript(
        t,
        [
            _line("user", "please use uv", uuid="u1"),
            _line(
                "assistant",
                [
                    _text_block("I'll use uv as per your convention."),
                    _tool_use_block("Bash"),
                ],
                uuid="a1",
            ),
            # tool_result lines come back as role=user with list content —
            # must be dropped (user prompts arrive via UserPromptSubmit).
            _line("user", [{"type": "tool_result", "content": "ok"}], uuid="u2"),
        ],
    )
    turns = tr.iter_assistant_prose(t)
    assert [x.text for x in turns] == ["I'll use uv as per your convention."]
    assert turns[0].uuid == "a1"


def test_content_string_and_blocks_both_flatten(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    _write_transcript(
        t,
        [
            _line("assistant", "plain string content", uuid="a1"),
            _line(
                "assistant",
                [_text_block("first"), _tool_use_block("Read"), _text_block("second")],
                uuid="a2",
            ),
        ],
    )
    turns = tr.iter_assistant_prose(t)
    assert [x.text for x in turns] == ["plain string content", "first\nsecond"]


def test_flat_top_level_message_shape(tmp_path: Path):
    """Flat-shape fallback (faithful port of _flush_spawn): role/content are
    read off the top level only when ``message`` is present but NOT a dict
    (e.g. a null/string placeholder). A line with no ``message`` key at all
    takes the wrapped branch's empty default and yields nothing — exactly
    the legacy behaviour."""
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        json.dumps({"uuid": "a1", "message": None, "role": "assistant", "content": "flat prose"})
        + "\n",
        encoding="utf-8",
    )
    turns = tr.iter_assistant_prose(t)
    assert [x.text for x in turns] == ["flat prose"]


def test_empty_and_whitespace_text_dropped(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    _write_transcript(
        t,
        [
            _line("assistant", [_text_block("   ")], uuid="a1"),
            _line("assistant", [_tool_use_block("Bash")], uuid="a2"),
            _line("assistant", "real", uuid="a3"),
        ],
    )
    turns = tr.iter_assistant_prose(t)
    assert [x.text for x in turns] == ["real"]


def test_bad_json_lines_skipped(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        "\n".join(
            [
                "not json at all",
                json.dumps([1, 2, 3]),  # not an object
                _line("assistant", "survives", uuid="a1"),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    turns = tr.iter_assistant_prose(t)
    assert [x.text for x in turns] == ["survives"]


def test_missing_transcript_returns_empty(tmp_path: Path):
    assert tr.iter_assistant_prose(tmp_path / "nope.jsonl") == []
    # A directory is not a file — must also degrade, not raise.
    assert tr.iter_assistant_prose(tmp_path) == []


def test_per_turn_char_cap(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    big = "x" * (tr.MAX_PROSE_CHARS + 500)
    _write_transcript(t, [_line("assistant", big, uuid="a1")])
    turns = tr.iter_assistant_prose(t)
    assert len(turns) == 1
    assert turns[0].text.endswith("... (truncated)")
    assert len(turns[0].text) < tr.MAX_PROSE_CHARS + 100


def test_missing_uuid_falls_back_to_line_number(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        json.dumps({"message": {"role": "assistant", "content": "no uuid here"}}) + "\n",
        encoding="utf-8",
    )
    turns = tr.iter_assistant_prose(t)
    assert turns[0].uuid == "line:1"


# ── incremental marker / dedup ───────────────────────────────────────


def test_marker_dedups_across_reads(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    marker = tmp_path / "marker.json"
    _write_transcript(t, [_line("assistant", "turn one", uuid="a1")])

    first = tr.read_new_assistant_prose(t, session_id="s1", marker_path=marker)
    assert [x.text for x in first] == ["turn one"]

    # Same transcript, second Stop — nothing new.
    second = tr.read_new_assistant_prose(t, session_id="s1", marker_path=marker)
    assert second == []

    # Transcript grows; only the new turn comes back.
    _write_transcript(
        t,
        [_line("assistant", "turn one", uuid="a1"), _line("assistant", "turn two", uuid="a2")],
    )
    third = tr.read_new_assistant_prose(t, session_id="s1", marker_path=marker)
    assert [x.text for x in third] == ["turn two"]


def test_marker_is_per_session(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    marker = tmp_path / "marker.json"
    _write_transcript(t, [_line("assistant", "shared", uuid="a1")])

    tr.read_new_assistant_prose(t, session_id="s1", marker_path=marker)
    # A different session has its own high-water mark — sees the turn fresh.
    other = tr.read_new_assistant_prose(t, session_id="s2", marker_path=marker)
    assert [x.text for x in other] == ["shared"]


def test_no_marker_returns_all_capped(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    lines = [_line("assistant", f"t{i}", uuid=f"a{i}") for i in range(tr.MAX_PROSE_TURNS + 5)]
    _write_transcript(t, lines)
    out = tr.read_new_assistant_prose(t, session_id="s1", marker_path=None)
    # Capped to the newest MAX_PROSE_TURNS.
    assert len(out) == tr.MAX_PROSE_TURNS
    assert out[-1].text == f"t{tr.MAX_PROSE_TURNS + 4}"


def test_corrupt_marker_tolerated(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    marker = tmp_path / "marker.json"
    marker.write_text("{ not valid json", encoding="utf-8")
    _write_transcript(t, [_line("assistant", "still works", uuid="a1")])
    out = tr.read_new_assistant_prose(t, session_id="s1", marker_path=marker)
    assert [x.text for x in out] == ["still works"]


def test_string_block_in_content_list(tmp_path: Path):
    """Faithful port: a bare string inside the content list is kept as text
    (legacy ``_content_to_text`` appends str blocks)."""
    t = tmp_path / "transcript.jsonl"
    _write_transcript(t, [_line("assistant", ["a bare string block"], uuid="a1")])
    assert [x.text for x in tr.iter_assistant_prose(t)] == ["a bare string block"]


def test_non_list_non_str_content_yields_nothing(tmp_path: Path):
    t = tmp_path / "transcript.jsonl"
    _write_transcript(t, [_line("assistant", {"unexpected": "dict"}, uuid="a1")])
    assert tr.iter_assistant_prose(t) == []


def test_marker_save_failure_is_fail_soft(tmp_path: Path):
    """If the marker can't be persisted, the read still returns the prose
    (dedup degrades to potential re-reads, never a crash)."""
    t = tmp_path / "transcript.jsonl"
    _write_transcript(t, [_line("assistant", "prose", uuid="a1")])
    # Point the marker at a path whose parent is a FILE — mkdir/rename fail.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    marker = blocker / "marker.json"
    out = tr.read_new_assistant_prose(t, session_id="s1", marker_path=marker)
    assert [x.text for x in out] == ["prose"]


def test_seen_uuid_set_is_bounded(tmp_path: Path, monkeypatch):
    """The per-session seen-uuid list is trimmed so a marathon session can't
    grow the marker file without bound."""
    monkeypatch.setattr(tr, "MAX_SEEN_UUIDS_PER_SESSION", 3)
    t = tmp_path / "transcript.jsonl"
    marker = tmp_path / "marker.json"
    _write_transcript(t, [_line("assistant", f"t{i}", uuid=f"a{i}") for i in range(6)])
    tr.read_new_assistant_prose(t, session_id="s1", marker_path=marker)
    state = json.loads(marker.read_text(encoding="utf-8"))
    assert len(state["s1"]) == 3
    # The kept uuids are the most recent ones.
    assert state["s1"] == ["a3", "a4", "a5"]


# ── end-to-end: scheduler folds prose into the buffer ────────────────


def _noop_librarian(snap: Dict[str, Any]):
    return {"session_id": snap["session_id"], "proposals": [], "interrupts": []}


def _noop_scholar(batch: List[Any]):
    return None


def test_assistant_prose_reaches_flattened_buffer(tmp_path: Path):
    """The headline guarantee: a Stop carrying transcript_path causes the
    assistant's prose to land in the buffer snapshot the Librarian sees,
    rendered with the distinct ``assistant-prose`` role."""
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            _line("user", "set up the project", uuid="u1"),
            _line(
                "assistant",
                [
                    _text_block("I'll use uv as per your convention, not pip."),
                    _tool_use_block("Bash"),
                ],
                uuid="a1",
            ),
        ],
    )

    buf = RollingBuffer()
    sched = Scheduler(
        buffer=buf,
        librarian=_noop_librarian,
        scholar=_noop_scholar,
        transcript_marker=tmp_path / "marker.json",
    )

    # A normal tool-use event opens the turn...
    sched.on_event(
        Event(
            ts=1.0,
            session_id="s1",
            type="PostToolUse",
            cwd="/repo",
            payload={"tool": "Bash"},
        )
    )
    # ...then a Stop carrying the transcript_path seals it, folding in prose.
    sched.on_event(
        Event(
            ts=2.0,
            session_id="s1",
            type="Stop",
            cwd="/repo",
            payload={"transcript_path": str(transcript)},
        )
    )

    snap = buf.snapshot("s1")
    assert snap is not None
    flat = flatten_buffer(snap)
    roles_and_text = [(role, text) for _tid, _seq, role, text, _ua in flat]

    assert (
        ASSISTANT_PROSE_ROLE,
        "I'll use uv as per your convention, not pip.",
    ) in roles_and_text
    # The tool-I/O event is still there too (prose augments, never replaces).
    assert any(role != ASSISTANT_PROSE_ROLE for role, _ in roles_and_text)


def test_stop_without_transcript_path_is_noop(tmp_path: Path):
    buf = RollingBuffer()
    sched = Scheduler(
        buffer=buf,
        librarian=_noop_librarian,
        scholar=_noop_scholar,
        transcript_marker=tmp_path / "marker.json",
    )
    sched.on_event(
        Event(ts=1.0, session_id="s1", type="PostToolUse", cwd="/r", payload={"tool": "Read"})
    )
    sched.on_event(Event(ts=2.0, session_id="s1", type="Stop", cwd="/r", payload={}))

    snap = buf.snapshot("s1")
    assert snap is not None
    flat = flatten_buffer(snap)
    assert all(role != ASSISTANT_PROSE_ROLE for _t, _s, role, _txt, _u in flat)


def test_unreadable_transcript_does_not_break_seal(tmp_path: Path):
    buf = RollingBuffer()
    sched = Scheduler(
        buffer=buf,
        librarian=_noop_librarian,
        scholar=_noop_scholar,
        transcript_marker=tmp_path / "marker.json",
    )
    sched.on_event(
        Event(ts=1.0, session_id="s1", type="PostToolUse", cwd="/r", payload={"tool": "Read"})
    )
    # transcript_path points at a nonexistent file — seal must still happen.
    sched.on_event(
        Event(
            ts=2.0,
            session_id="s1",
            type="Stop",
            cwd="/r",
            payload={"transcript_path": str(tmp_path / "ghost.jsonl")},
        )
    )
    snap = buf.snapshot("s1")
    assert snap is not None
    # The turn sealed (one turn in the deque) despite the missing transcript.
    assert len(snap["turns"]) == 1


def test_transcript_path_on_raw_event_also_works(tmp_path: Path):
    """The hook may put transcript_path on the raw line rather than nested
    in payload — the extractor checks both."""
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, [_line("assistant", "raw-path prose", uuid="a1")])

    buf = RollingBuffer()
    sched = Scheduler(
        buffer=buf,
        librarian=_noop_librarian,
        scholar=_noop_scholar,
        transcript_marker=tmp_path / "marker.json",
    )
    ev = Event(ts=1.0, session_id="s1", type="Stop", cwd="/r", payload={})
    ev.raw = {"transcript_path": str(transcript)}
    sched.on_event(ev)

    snap = buf.snapshot("s1")
    assert snap is not None
    flat = flatten_buffer(snap)
    assert any(
        role == ASSISTANT_PROSE_ROLE and text == "raw-path prose" for _t, _s, role, text, _u in flat
    )
