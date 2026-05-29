"""Buffer aggregation and eviction tests."""

from __future__ import annotations

from agent_mem_daemon.buffer import (
    DEFAULT_INACTIVITY_SECONDS,
    Event,
    RollingBuffer,
)


def _ev(ts: float, sid: str, typ: str, **payload):
    return Event(ts=ts, session_id=sid, type=typ, cwd="/repo", payload=payload)


def test_turn_aggregates_post_tool_use_between_stops():
    """A turn is everything from the previous Stop up to (and including)
    the next Stop. PostToolUse events between two Stops belong to the
    same turn."""
    buf = RollingBuffer(max_turns=20)
    buf.ingest(_ev(1.0, "s1", "PostToolUse", tool="Read"))
    buf.ingest(_ev(2.0, "s1", "PostToolUse", tool="Edit"))
    sealed = buf.ingest(_ev(3.0, "s1", "Stop"))

    assert sealed is not None
    sess = buf.session("s1")
    assert sess is not None
    assert len(sess.turns) == 1
    turn = sess.turns[0]
    # Two PostToolUse events + the Stop itself.
    assert len(turn.events) == 3
    assert [e.type for e in turn.events] == ["PostToolUse", "PostToolUse", "Stop"]
    assert turn.started_at == 1.0
    assert turn.sealed_at == 3.0
    # Open turn is reset.
    assert sess.open_turn.is_empty()
    # Stop should have flipped needs_librarian.
    assert sess.needs_librarian is True


def test_multiple_turns_in_same_session():
    """Sequential Stop events produce sequential turns."""
    buf = RollingBuffer(max_turns=20)
    buf.ingest(_ev(1.0, "s1", "PostToolUse"))
    buf.ingest(_ev(2.0, "s1", "Stop"))
    buf.ingest(_ev(3.0, "s1", "PostToolUse"))
    buf.ingest(_ev(4.0, "s1", "PostToolUse"))
    buf.ingest(_ev(5.0, "s1", "Stop"))

    sess = buf.session("s1")
    assert sess is not None
    assert len(sess.turns) == 2
    # First turn: PostToolUse + Stop
    assert [e.type for e in sess.turns[0].events] == ["PostToolUse", "Stop"]
    # Second turn: PostToolUse + PostToolUse + Stop
    assert [e.type for e in sess.turns[1].events] == ["PostToolUse", "PostToolUse", "Stop"]


def test_max_turns_enforced():
    """Beyond max_turns, the oldest turn is dropped."""
    buf = RollingBuffer(max_turns=3)
    for i in range(5):
        buf.ingest(_ev(float(i), "s1", "PostToolUse"))
        buf.ingest(_ev(float(i) + 0.5, "s1", "Stop"))
    sess = buf.session("s1")
    assert sess is not None
    assert len(sess.turns) == 3
    # First two turns should have been dropped; the surviving turns
    # have Stop at ts=2.5, 3.5, 4.5.
    sealed_ats = [t.sealed_at for t in sess.turns]
    assert sealed_ats == [2.5, 3.5, 4.5]


def test_sessions_isolated():
    """Two session_ids don't bleed into each other."""
    buf = RollingBuffer()
    buf.ingest(_ev(1.0, "s1", "PostToolUse"))
    buf.ingest(_ev(2.0, "s2", "PostToolUse"))
    buf.ingest(_ev(3.0, "s1", "Stop"))

    s1 = buf.session("s1")
    s2 = buf.session("s2")
    assert s1 is not None and s2 is not None
    assert len(s1.turns) == 1
    assert len(s2.turns) == 0
    # s2 still has its open turn populated.
    assert len(s2.open_turn.events) == 1


def test_session_end_seals_open_turn_and_marks_ended():
    buf = RollingBuffer()
    buf.ingest(_ev(1.0, "s1", "PostToolUse"))
    buf.ingest(_ev(2.0, "s1", "SessionEnd"))

    sess = buf.session("s1")
    assert sess is not None
    assert sess.ended is True
    assert len(sess.turns) == 1
    assert sess.needs_librarian is True


def test_session_end_with_empty_open_turn_still_seals():
    """SessionEnd after a Stop still produces a (1-event) final turn —
    we record the SessionEnd itself."""
    buf = RollingBuffer()
    buf.ingest(_ev(1.0, "s1", "PostToolUse"))
    buf.ingest(_ev(2.0, "s1", "Stop"))
    buf.ingest(_ev(3.0, "s1", "SessionEnd"))

    sess = buf.session("s1")
    assert sess is not None
    assert sess.ended is True
    assert len(sess.turns) == 2
    assert sess.turns[-1].events[-1].type == "SessionEnd"


def test_sweep_drops_old_sessions():
    """Sessions idle longer than inactivity_seconds are evicted."""
    buf = RollingBuffer(inactivity_seconds=60.0)
    # Inject an event with a synthetic old ts.
    buf.ingest(_ev(0.0, "old", "PostToolUse"))
    buf.ingest(_ev(1000.0, "new", "PostToolUse"))

    dropped = buf.sweep(now=1000.0)
    assert dropped == ["old"]
    assert buf.session("old") is None
    assert buf.session("new") is not None


def test_sweep_default_one_hour():
    """The default inactivity window is one hour, per scope."""
    buf = RollingBuffer()
    assert buf.inactivity_seconds == DEFAULT_INACTIVITY_SECONDS
    assert DEFAULT_INACTIVITY_SECONDS == 3600


def test_snapshot_shape():
    """The snapshot the scheduler hands to the Librarian doesn't leak
    internal dataclasses."""
    buf = RollingBuffer()
    buf.ingest(_ev(1.0, "s1", "PostToolUse", tool="Read"))
    buf.ingest(_ev(2.0, "s1", "Stop"))

    snap = buf.snapshot("s1")
    assert snap is not None
    assert snap["session_id"] == "s1"
    assert snap["cwd"] == "/repo"
    assert isinstance(snap["turns"], list)
    assert len(snap["turns"]) == 1
    t = snap["turns"][0]
    assert "events" in t
    # Stable per-session turn id rides in the snapshot for fired-helpful dedup.
    assert t["turn_seq"] == 1
    assert all(isinstance(e, dict) for e in t["events"])
    # No SessionState / Turn dataclasses in the snapshot.
    assert all(not hasattr(e, "__dataclass_fields__") for e in t["events"])


def test_turn_seq_monotonic_per_session():
    """Each sealed turn gets a strictly-increasing per-session turn_seq."""
    buf = RollingBuffer(max_turns=20)
    for i in range(3):
        buf.ingest(_ev(float(i), "s1", "PostToolUse"))
        buf.ingest(_ev(float(i) + 0.5, "s1", "Stop"))
    sess = buf.session("s1")
    assert sess is not None
    assert [t.turn_seq for t in sess.turns] == [1, 2, 3]


def test_turn_seq_stable_across_eviction():
    """The double-count fix hinges on turn_seq NOT shifting when older
    turns age out of the deque. After eviction, the surviving turns keep
    their original ids — the allocator never resets."""
    buf = RollingBuffer(max_turns=3)
    for i in range(5):
        buf.ingest(_ev(float(i), "s1", "PostToolUse"))
        buf.ingest(_ev(float(i) + 0.5, "s1", "Stop"))
    sess = buf.session("s1")
    assert sess is not None
    # 5 turns sealed, only the last 3 retained — but their ids are the
    # original 3, 4, 5 (NOT renumbered 1, 2, 3).
    assert [t.turn_seq for t in sess.turns] == [3, 4, 5]


def test_turn_seq_independent_per_session():
    """turn_seq allocators are per-session and do not bleed across sessions."""
    buf = RollingBuffer(max_turns=20)
    buf.ingest(_ev(1.0, "s1", "PostToolUse"))
    buf.ingest(_ev(1.5, "s1", "Stop"))
    buf.ingest(_ev(2.0, "s2", "PostToolUse"))
    buf.ingest(_ev(2.5, "s2", "Stop"))
    buf.ingest(_ev(3.0, "s1", "PostToolUse"))
    buf.ingest(_ev(3.5, "s1", "Stop"))
    s1 = buf.session("s1")
    s2 = buf.session("s2")
    assert s1 is not None and s2 is not None
    assert [t.turn_seq for t in s1.turns] == [1, 2]
    assert [t.turn_seq for t in s2.turns] == [1]


def test_cwd_back_fills_from_first_event_that_has_it():
    """If the first event lacks cwd, a later event can populate it."""
    buf = RollingBuffer()
    buf.ingest(Event(ts=1.0, session_id="s1", type="PostToolUse", cwd=None))
    buf.ingest(Event(ts=2.0, session_id="s1", type="PostToolUse", cwd="/repo"))
    sess = buf.session("s1")
    assert sess is not None
    assert sess.cwd == "/repo"
