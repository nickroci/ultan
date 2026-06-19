"""Regression tests for the plugin's event-capture path.

These pin the bug that shipped in the first plugin cut: the hooks recalled
memory but never *captured* it. ``post-tool-use`` / ``stop`` / ``session-end``
were no-ops, so nothing wrote ``events.jsonl`` and the daemon — whose only
input is that file — tailed an empty stream and learned nothing.

The contract under test (the daemon's ``agent_mem_daemon/ingest.parse_event_line``):
a JSON line ``{ts: float, session_id, type, cwd?, payload?}`` with ``session_id``
and ``type`` required. ``Stop`` and ``SessionEnd`` seal turns and drive the
Librarian (``agent_mem_daemon/buffer.py``), so they must be written even with an
empty payload.

Hermetic: every test points ``AGENT_MEM_HOME`` at a tmp dir and clears the
``CLAUDE_INVOKED_BY`` recursion guard, so nothing touches the real store and no
daemon is spawned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultan import _events, _hooks


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    return tmp_path


def _events_file(home: Path) -> Path:
    return home / "events.jsonl"


def _read_events(home: Path) -> list[dict]:
    path = _events_file(home)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ── the schema every captured event must satisfy ────────────────────────────


def _assert_valid_event(ev: dict, *, expected_type: str) -> None:
    assert isinstance(ev["ts"], float)  # daemon accepts unix float
    assert ev["session_id"]  # required — daemon drops events without it
    assert ev["type"] == expected_type
    assert "cwd" in ev  # optional value, but the key is always present
    assert isinstance(ev["payload"], dict)


# ── post-tool-use → PostToolUse (turn content) ──────────────────────────────


def test_post_tool_use_writes_event(_isolated_store: Path) -> None:
    rc = _hooks.dispatch(
        "post-tool-use",
        {
            "session_id": "sess-1",
            "cwd": "/work/proj",
            "tool_name": "Edit",
            "tool_input": {"file_path": "/work/proj/a.py", "new_string": "x = 1"},
            "tool_response": "ok",
        },
    )
    assert rc == 0
    events = _read_events(_isolated_store)
    assert len(events) == 1
    ev = events[0]
    _assert_valid_event(ev, expected_type="PostToolUse")
    assert ev["cwd"] == "/work/proj"
    payload = ev["payload"]
    assert payload["tool"] == "Edit"
    assert payload["ok"] is True
    assert payload["content"] == "x = 1"  # the Librarian reads `content`
    assert "Edit on /work/proj/a.py ok" == payload["summary"]


def test_post_tool_use_marks_failure(_isolated_store: Path) -> None:
    _hooks.dispatch(
        "post-tool-use",
        {"session_id": "s", "tool_name": "Bash", "error": "boom"},
    )
    (ev,) = _read_events(_isolated_store)
    assert ev["payload"]["ok"] is False
    assert ev["payload"]["summary"].endswith("FAILED")


# ── user-prompt-submit → Surfaced (what the instant triggers showed) ─────────


def test_extract_surfaced_links_dedups_and_handles_empty() -> None:
    from ultan import _priming

    md = "## Ultan says\n- [[projects/x/foo]] ★ — a\n- [[global/y/bar]] — b\n- [[projects/x/foo]] dup\n"
    assert _priming.extract_surfaced_links(md) == ["projects/x/foo", "global/y/bar"]
    assert _priming.extract_surfaced_links("") == []
    assert _priming.extract_surfaced_links("no links here") == []


def _stub_priming(monkeypatch: pytest.MonkeyPatch, md: str) -> None:
    # Don't spawn a daemon or hit the socket; pin the priming output.
    monkeypatch.setattr(_hooks._daemon, "ensure_running", lambda: None)
    monkeypatch.setattr(_hooks._daemon, "status", lambda: "ready")
    monkeypatch.setattr(_hooks._priming, "get_priming", lambda *a, **k: md)


def test_user_prompt_submit_emits_surfaced_event(
    _isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_priming(
        monkeypatch,
        "## Ultan says\n- [[projects/x/crypto-no-profit]] — you can't profit from crypto\n",
    )
    rc = _hooks.dispatch(
        "user-prompt-submit",
        {"session_id": "s1", "cwd": "/w", "prompt": "how do I make money from BTC"},
    )
    assert rc == 0
    by_type = {e["type"]: e for e in _read_events(_isolated_store)}
    assert "UserPromptSubmit" in by_type  # the prompt itself still captured
    assert "Surfaced" in by_type  # and what the instant triggers surfaced
    surfaced = by_type["Surfaced"]
    _assert_valid_event(surfaced, expected_type="Surfaced")
    assert surfaced["payload"]["role"] == "recall"
    assert "[[projects/x/crypto-no-profit]]" in surfaced["payload"]["content"]


def test_user_prompt_submit_no_surfaced_event_when_nothing_primed(
    _isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_priming(monkeypatch, "")  # daemon found nothing → no Surfaced line
    _hooks.dispatch("user-prompt-submit", {"session_id": "s1", "prompt": "hi"})
    types = {e["type"] for e in _read_events(_isolated_store)}
    assert "Surfaced" not in types


# ── stop / session-end → turn-sealing events (the Librarian triggers) ────────


def test_stop_writes_sealing_event(_isolated_store: Path) -> None:
    # G1: Stop now carries transcript_path so the daemon's scheduler can read the
    # turn's assistant prose (scheduler._transcript_path_of reads
    # ev.payload["transcript_path"]).
    _hooks.dispatch(
        "stop",
        {"session_id": "sess-1", "cwd": "/work/proj", "transcript_path": "/tmp/t.jsonl"},
    )
    (ev,) = _read_events(_isolated_store)
    _assert_valid_event(ev, expected_type="Stop")
    assert ev["payload"] == {"transcript_path": "/tmp/t.jsonl"}


def test_stop_forwards_none_transcript_when_absent(_isolated_store: Path) -> None:
    # Absent transcript_path in stdin → None in payload; harmless (the daemon
    # degrades to no prose) and Stop still seals the turn.
    _hooks.dispatch("stop", {"session_id": "sess-1", "cwd": "/work/proj"})
    (ev,) = _read_events(_isolated_store)
    _assert_valid_event(ev, expected_type="Stop")
    assert ev["payload"] == {"transcript_path": None}


def test_session_end_writes_event(_isolated_store: Path) -> None:
    _hooks.dispatch("session-end", {"session_id": "sess-1"})
    (ev,) = _read_events(_isolated_store)
    _assert_valid_event(ev, expected_type="SessionEnd")


# ── recursion guard: the daemon's own SDK agents must not capture ────────────


def test_recursion_guard_suppresses_capture(
    _isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "agent_mem_daemon")
    _hooks.dispatch(
        "post-tool-use",
        {"session_id": "s", "tool_name": "Write", "tool_input": {"content": "ghost"}},
    )
    _hooks.dispatch("stop", {"session_id": "s"})
    assert _read_events(_isolated_store) == []  # nothing leaked back to the daemon


# ── robustness: never crash the host, never emit a malformed line ────────────


def test_missing_session_id_drops_event(_isolated_store: Path) -> None:
    _events.append_event("PostToolUse", {"tool_name": "Read"}, payload={"tool": "Read"})
    assert _read_events(_isolated_store) == []  # daemon requires session_id → drop


def test_oversize_payload_stays_within_line_budget(_isolated_store: Path) -> None:
    huge = "y = 2  # " + ("z" * 10_000)
    _hooks.dispatch(
        "post-tool-use",
        {"session_id": "s", "tool_name": "Write", "tool_input": {"content": huge}},
    )
    raw = _events_file(_isolated_store).read_bytes().rstrip(b"\n")
    assert len(raw) + 1 <= _events.MAX_LINE_BYTES
    # Still a valid, parseable line the daemon can ingest.
    ev = json.loads(raw)
    _assert_valid_event(ev, expected_type="PostToolUse")


# ── dispatch routing: user-prompt-submit captures; the rest stay no-ops ──────


def test_user_prompt_submit_writes_event(
    _isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stub the two side-effecting collaborators so the test exercises ONLY the
    # capture write — not the lazy daemon spawn or the priming socket call.
    monkeypatch.setattr(_hooks._daemon, "ensure_running", lambda: True)
    monkeypatch.setattr(_hooks._priming, "get_priming", lambda *a, **k: "")

    rc = _hooks.dispatch(
        "user-prompt-submit",
        {"session_id": "s", "cwd": "/w", "prompt": "always use uv, never pip"},
    )
    assert rc == 0
    (ev,) = [e for e in _read_events(_isolated_store) if e["type"] == "UserPromptSubmit"]
    _assert_valid_event(ev, expected_type="UserPromptSubmit")
    assert ev["payload"]["role"] == "user"
    assert ev["payload"]["content"] == "always use uv, never pip"


def test_pre_tool_use_writes_pre_check_event(_isolated_store: Path) -> None:
    # G2: PreToolUse is now wired (Tier 3 blocker check). With no blocker rules
    # in the (empty) store nothing is denied, but we ALWAYS append a PreToolUse
    # event so the daemon sees the call — PostToolUse won't fire for a denied
    # call, so this is the only record of a blocked one.
    rc = _hooks.dispatch(
        "pre-tool-use", {"session_id": "s", "tool_name": "Read", "tool_input": {"file_path": "/x"}}
    )
    assert rc == 0
    (ev,) = _read_events(_isolated_store)
    _assert_valid_event(ev, expected_type="PreToolUse")
    assert ev["payload"]["blocked"] is False
    assert ev["payload"]["tool"] == "Read"
    assert ev["payload"]["summary"] == "Read pre-check ok"


def test_pre_compact_seals_turn_with_session_end(_isolated_store: Path) -> None:
    # Legacy parity (src/hooks/pre_compact.py): compaction discards transcript
    # detail, so the open turn must be sealed (SessionEnd → Librarian) FIRST.
    rc = _hooks.dispatch(
        "pre-compact", {"session_id": "s", "cwd": "/w", "transcript_path": "/tmp/t.jsonl"}
    )
    assert rc == 0
    (ev,) = _read_events(_isolated_store)
    _assert_valid_event(ev, expected_type="SessionEnd")
    # G1: pre-compact carries BOTH the source tag AND transcript_path, so the
    # pre-compaction prose is captured before compaction discards it.
    assert ev["payload"] == {"source": "pre-compact", "transcript_path": "/tmp/t.jsonl"}


def test_session_start_writes_boundary_event(
    _isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_hooks._daemon, "ensure_running", lambda: True)
    rc = _hooks.dispatch("session-start", {"session_id": "s", "cwd": "/w", "source": "resume"})
    assert rc == 0
    (ev,) = _read_events(_isolated_store)
    _assert_valid_event(ev, expected_type="SessionStart")
    assert ev["payload"] == {"source": "resume"}


def test_unknown_event_is_rejected(_isolated_store: Path) -> None:
    rc = _hooks.dispatch("bogus-event", {"session_id": "s"})
    # Exit 1, NOT 2: in the hook protocol exit 2 is the BLOCKING signal
    # (denies tools / erases prompts) — a misconfigured event name must be
    # visible (non-zero) without blocking anything.
    assert rc == 1
    assert _read_events(_isolated_store) == []
