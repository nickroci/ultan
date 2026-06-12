"""Tests for the plugin-path pending-nudges consumer (``ultan/_nudges.py``).

The Scholar writes ``pending-nudges.md``; this module reads + clears + budgets
+ filters + renders. These pin the consumer's contract:

- read + clear → renames to ``.consumed`` (atomic clear, last batch kept)
- budget caps: 1/turn, 3/session, with the counter persisted across calls
- cross-project filter: a nudge scoped to another project is filtered out and
  RE-QUEUED to disk; a global (or unrecognised-bucket) nudge delivers to anyone
- the file is read+cleared even when the budget is exhausted (no infinite pile)
- empty / missing file → ""

Hermetic: every test points ``AGENT_MEM_HOME`` at a tmp dir so nothing touches
the real store. ``project-aliases.json`` is written into that home where a test
exercises slug→bucket resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultan import _nudges


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    return tmp_path


# ── helpers ─────────────────────────────────────────────────────────────────


def _block(nudge_id: str, lesson: str, text: str, created: str = "2026-06-12T00:00:00Z") -> str:
    return f"---\nid: {nudge_id}\ncreated: {created}\nlesson: {lesson}\n---\n{text}\n"


def _write_nudges(home: Path, *blocks: str) -> Path:
    path = home / "pending-nudges.md"
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _consumed_count(home: Path, session_id: str) -> int:
    path = home / "state" / f"nudge-budget-{session_id}.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8"))["consumed"])


# ── empty / missing file ─────────────────────────────────────────────────────


def test_missing_file_returns_empty(_isolated_home: Path) -> None:
    assert _nudges.take_and_render("sess-1", None) == ""


def test_empty_file_returns_empty_and_is_cleaned(_isolated_home: Path) -> None:
    path = _isolated_home / "pending-nudges.md"
    path.write_text("", encoding="utf-8")
    assert _nudges.take_and_render("sess-1", None) == ""
    # Empty file is unlinked, not renamed aside.
    assert not path.exists()


def test_empty_session_id_returns_empty_and_leaves_file(_isolated_home: Path) -> None:
    path = _write_nudges(_isolated_home, _block("a", "global/foo", "do the thing"))
    # No session key → no budget bookkeeping possible → skip without consuming.
    assert _nudges.take_and_render("", None) == ""
    assert path.exists()


# ── parsing + rendering ──────────────────────────────────────────────────────


def test_parse_nudges_roundtrip() -> None:
    body = (
        _block("id1", "global/lesson-a", "first lesson")
        + "\n"
        + _block("id2", "projects/agent-mem/lesson-b", "second lesson")
    )
    parsed = _nudges.parse_nudges(body)
    assert [n.id for n in parsed] == ["id1", "id2"]
    assert parsed[0].lesson == "global/lesson-a"
    assert parsed[1].text == "second lesson"


def test_render_context_shape() -> None:
    nudges = [_nudges.Nudge(id="x", created="c", lesson="global/foo", text="apply X")]
    rendered = _nudges.render_context(nudges)
    assert "Memory has 1 relevant lesson(s)." in rendered
    assert "The user has not been asked." in rendered
    assert "- [[global/foo]]: apply X" in rendered


def test_render_context_empty() -> None:
    assert _nudges.render_context([]) == ""


# ── read + clear → .consumed ─────────────────────────────────────────────────


def test_read_and_clear_renames_to_consumed(_isolated_home: Path) -> None:
    path = _write_nudges(_isolated_home, _block("a", "global/foo", "lesson body"))
    out = _nudges.take_and_render("sess-1", None)
    assert "lesson body" in out
    # Original cleared by rename; .consumed sibling holds the last batch.
    assert not path.exists()
    consumed_file = _isolated_home / "pending-nudges.md.consumed"
    assert consumed_file.exists()
    assert "lesson body" in consumed_file.read_text(encoding="utf-8")


# ── budget: 1 per turn ───────────────────────────────────────────────────────


def test_per_turn_cap_is_one(_isolated_home: Path) -> None:
    _write_nudges(
        _isolated_home,
        _block("a", "global/x", "first"),
        _block("b", "global/y", "second"),
        _block("c", "global/z", "third"),
    )
    selected, consumed = _nudges.take_nudges("sess-1", current_project_slug=None)
    assert len(selected) == 1
    assert selected[0].id == "a"  # FIFO: first queued wins the single slot
    assert consumed == 1
    # The other two were NOT re-queued — only cross-project nudges roll over.
    assert not (_isolated_home / "pending-nudges.md").exists()


# ── budget: 3 per session, persisted across calls ────────────────────────────


def test_per_session_cap_persists_across_turns(_isolated_home: Path) -> None:
    session = "sess-budget"
    # Three separate turns, each with a fresh nudge, all global so all match.
    for i in range(3):
        _write_nudges(_isolated_home, _block(f"n{i}", "global/x", f"lesson {i}"))
        out = _nudges.take_and_render(session, None)
        assert f"lesson {i}" in out
        assert _consumed_count(_isolated_home, session) == i + 1

    # Fourth turn: budget exhausted → nothing delivered, counter unchanged.
    path = _write_nudges(_isolated_home, _block("n3", "global/x", "lesson 3"))
    out = _nudges.take_and_render(session, None)
    assert out == ""
    assert _consumed_count(_isolated_home, session) == 3
    # ...but the file was STILL cleared so it doesn't pile up unread.
    assert not path.exists()
    assert (_isolated_home / "pending-nudges.md.consumed").exists()


def test_exhausted_budget_still_clears_file(_isolated_home: Path) -> None:
    session = "sess-exhausted"
    # Pre-seed the counter at the session cap.
    state = _isolated_home / "state"
    state.mkdir(parents=True)
    (state / f"nudge-budget-{session}.json").write_text(
        json.dumps({"consumed": 3}), encoding="utf-8"
    )
    path = _write_nudges(_isolated_home, _block("a", "global/x", "should not appear"))
    out = _nudges.take_and_render(session, None)
    assert out == ""
    # File cleared regardless of budget — the whole point of this branch.
    assert not path.exists()
    assert (_isolated_home / "pending-nudges.md.consumed").exists()
    # Counter unchanged (nothing was consumed).
    assert _consumed_count(_isolated_home, session) == 3


# ── cross-project filter + re-queue ──────────────────────────────────────────


def test_global_nudge_delivers_to_any_project(_isolated_home: Path) -> None:
    _write_nudges(_isolated_home, _block("a", "global/foo", "global lesson"))
    out = _nudges.take_and_render("sess-1", "some-unrelated-slug")
    assert "global lesson" in out


def test_unrecognised_bucket_delivers_to_anyone(_isolated_home: Path) -> None:
    # Path layout we don't recognise (not projects/ or global/) → universal.
    _write_nudges(_isolated_home, _block("a", "notes/freeform", "loose lesson"))
    out = _nudges.take_and_render("sess-1", "any-slug")
    assert "loose lesson" in out


def test_cross_project_nudge_is_filtered_and_requeued(_isolated_home: Path) -> None:
    # Current session is project "alpha"; nudge is scoped to project "beta".
    path = _write_nudges(_isolated_home, _block("b", "projects/beta/lesson", "beta-only lesson"))
    out = _nudges.take_and_render("sess-1", "alpha")
    # Filtered out of this session's delivery...
    assert out == ""
    # ...and re-queued to disk for a future session in project beta.
    assert path.exists()
    requeued = _nudges.parse_nudges(path.read_text(encoding="utf-8"))
    assert len(requeued) == 1
    assert requeued[0].id == "b"
    assert requeued[0].lesson == "projects/beta/lesson"


def test_matching_project_nudge_delivers(_isolated_home: Path) -> None:
    _write_nudges(_isolated_home, _block("a", "projects/alpha/lesson", "alpha lesson"))
    out = _nudges.take_and_render("sess-1", "alpha")
    assert "alpha lesson" in out


def test_alias_resolves_bucket_to_canonical_slug(_isolated_home: Path) -> None:
    # Folder bucket "agent-mem" aliases to a git-remote-derived session slug.
    (_isolated_home / "project-aliases.json").write_text(
        json.dumps({"agent-mem": "github.com-nickroci-ultan"}), encoding="utf-8"
    )
    _write_nudges(_isolated_home, _block("a", "projects/agent-mem/lesson", "aliased lesson"))
    out = _nudges.take_and_render("sess-1", "github.com-nickroci-ultan")
    assert "aliased lesson" in out


def test_no_project_context_delivers_everything(_isolated_home: Path) -> None:
    # project_slug=None → permissive: even a project-scoped nudge delivers,
    # because no future session could claim it either.
    _write_nudges(_isolated_home, _block("a", "projects/beta/lesson", "orphan lesson"))
    out = _nudges.take_and_render("sess-1", None)
    assert "orphan lesson" in out


def test_mixed_batch_delivers_match_requeues_foreign(_isolated_home: Path) -> None:
    # One matching + one foreign nudge. Match delivers (within the 1/turn cap),
    # foreign re-queues.
    path = _write_nudges(
        _isolated_home,
        _block("match", "projects/alpha/lesson", "alpha lesson"),
        _block("foreign", "projects/beta/lesson", "beta lesson"),
    )
    out = _nudges.take_and_render("sess-1", "alpha")
    assert "alpha lesson" in out
    assert "beta lesson" not in out
    # Foreign nudge survives on disk for a future beta session.
    requeued = _nudges.parse_nudges(path.read_text(encoding="utf-8"))
    assert [n.id for n in requeued] == ["foreign"]
