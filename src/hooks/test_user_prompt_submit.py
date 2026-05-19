"""Tests for the pending-nudges read+budget logic that
``user-prompt-submit.py`` depends on, plus an end-to-end test of the
hook itself (driven via subprocess so the stdin/stdout protocol path
is exercised verbatim).

Run from the ``src/`` directory with::

    uv run python -m pytest hooks/ -q

The hook script uses a hyphenated filename (``user-prompt-submit.py``)
because that matches what Claude Code's settings.json conventions
expect; we exercise it via subprocess rather than importing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make ``_nudges`` importable. pytest runs from src/, so hooks/ is the
# obvious place to add to sys.path. We do it once at module load.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import _nudges  # noqa: E402


HOOK_SCRIPT = _THIS_DIR / "user-prompt-submit.py"


# ── pure-Python tests of _nudges (fast, no subprocess) ───────────────


def test_parse_nudges_empty():
    assert _nudges.parse_nudges("") == []
    assert _nudges.parse_nudges("\n\n   \n") == []


def test_parse_nudges_three_blocks():
    body = (
        "---\nid: a1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/a\n---\nfirst\n"
        "---\nid: b2\ncreated: 2026-05-19T00:00:01+00:00\nlesson: l/b\n---\nsecond\n"
        "---\nid: c3\ncreated: 2026-05-19T00:00:02+00:00\nlesson: l/c\n---\nthird\n"
    )
    out = _nudges.parse_nudges(body)
    assert len(out) == 3
    assert out[0].id == "a1" and out[0].text == "first"
    assert out[2].lesson == "l/c"


def test_take_nudges_empty_file_returns_nothing(tmp_path: Path):
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    selected, consumed = _nudges.take_nudges(
        "s1",
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    assert selected == []
    assert consumed == 0
    # No state file should have been written for a no-op.
    assert not state_path.exists()


def test_take_nudges_missing_file_returns_nothing(tmp_path: Path):
    nudges_path = tmp_path / "does-not-exist.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    selected, consumed = _nudges.take_nudges(
        "s1",
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    assert selected == []
    assert consumed == 0


def test_take_nudges_budget_three_per_session(tmp_path: Path):
    """Three nudges queued, budget 1/turn, 3/session — across four turns
    we should see 1+1+1+0."""
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"

    # Write three nudges all at once. The hook clears on first read, so
    # we rewrite the file before each successive turn to simulate the
    # daemon producing more — except for the third+ turn where we test
    # the per-session cap kicks in.
    three_blocks = (
        "---\nid: a1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/a\n---\nA text\n"
        "---\nid: b2\ncreated: 2026-05-19T00:00:01+00:00\nlesson: l/b\n---\nB text\n"
        "---\nid: c3\ncreated: 2026-05-19T00:00:02+00:00\nlesson: l/c\n---\nC text\n"
    )
    nudges_path.write_text(three_blocks, encoding="utf-8")
    # First turn: only 1 emitted (per-turn budget), file is consumed.
    sel1, consumed1 = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert len(sel1) == 1
    assert sel1[0].id == "a1"
    assert consumed1 == 1
    # The file is cleared (renamed to .consumed).
    assert not nudges_path.exists()
    assert (tmp_path / "pending-nudges.md.consumed").exists()

    # Daemon writes more nudges before turn 2.
    nudges_path.write_text(
        "---\nid: d4\ncreated: 2026-05-19T00:00:03+00:00\nlesson: l/d\n---\nD text\n"
        "---\nid: e5\ncreated: 2026-05-19T00:00:04+00:00\nlesson: l/e\n---\nE text\n",
        encoding="utf-8",
    )
    sel2, consumed2 = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert len(sel2) == 1
    assert sel2[0].id == "d4"
    assert consumed2 == 2

    # Turn 3: one more queued, budget allows.
    nudges_path.write_text(
        "---\nid: f6\ncreated: 2026-05-19T00:00:05+00:00\nlesson: l/f\n---\nF text\n",
        encoding="utf-8",
    )
    sel3, consumed3 = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert len(sel3) == 1
    assert sel3[0].id == "f6"
    assert consumed3 == 3

    # Turn 4: more queued, but the per-session cap is exhausted.
    nudges_path.write_text(
        "---\nid: g7\ncreated: 2026-05-19T00:00:06+00:00\nlesson: l/g\n---\nG text\n",
        encoding="utf-8",
    )
    sel4, consumed4 = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert sel4 == []
    assert consumed4 == 3  # unchanged
    # The file is still cleared — we don't let a backlog accumulate.
    assert not nudges_path.exists()


def test_take_nudges_clears_file_even_when_over_budget(tmp_path: Path):
    """A session past its budget should still drain the file so unrelated
    sessions don't see stale content. (Today's design: one nudge file
    shared by all sessions — first-come first-serve.)"""
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    # Prime state to "already at budget".
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"consumed": 3, "updated": 0}), encoding="utf-8")
    nudges_path.write_text(
        "---\nid: x1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/x\n---\nover-budget text\n",
        encoding="utf-8",
    )
    selected, consumed = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert selected == []
    assert consumed == 3
    assert not nudges_path.exists()  # cleared


def test_render_context_includes_count_and_paths():
    nudges = [
        _nudges.Nudge(id="a", created="t", lesson="l/a", text="alpha text"),
        _nudges.Nudge(id="b", created="t", lesson="l/b", text="beta text"),
    ]
    rendered = _nudges.render_context(nudges)
    assert "2 relevant" in rendered
    assert "alpha text" in rendered
    assert "beta text" in rendered
    assert "[[l/a]]" in rendered
    assert "[[l/b]]" in rendered
    assert "user has not been asked" in rendered


def test_render_context_empty_returns_empty():
    assert _nudges.render_context([]) == ""


def test_per_turn_budget_overridable(tmp_path: Path):
    """If a caller passes per_turn_budget=3 they get up to 3 at once
    (still capped by per-session budget)."""
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    nudges_path.write_text(
        "---\nid: a1\ncreated: t\nlesson: l/a\n---\nA\n"
        "---\nid: b2\ncreated: t\nlesson: l/b\n---\nB\n"
        "---\nid: c3\ncreated: t\nlesson: l/c\n---\nC\n",
        encoding="utf-8",
    )
    sel, consumed = _nudges.take_nudges(
        "s1",
        per_turn_budget=10,
        per_session_budget=3,
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    assert len(sel) == 3
    assert consumed == 3


# ── End-to-end via subprocess (hook script) ───────────────────────────


def _run_hook(stdin_payload: dict, env_overrides: dict) -> subprocess.CompletedProcess:
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        # Belt-and-braces: ensure recursion guard is OFF for the test.
        # If a previous test happened to set it in the parent env we'd
        # short-circuit the hook entirely.
    }
    # Inherit just enough so Python can find its stdlib.
    import os
    for k in ("HOME", "PYTHONPATH", "VIRTUAL_ENV", "PATH", "PYTHONHOME"):
        if k in os.environ:
            env[k] = os.environ[k]
    env.update(env_overrides)
    env.pop("CLAUDE_INVOKED_BY", None)
    return subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def test_hook_emits_nothing_when_no_nudges(tmp_path: Path):
    res = _run_hook(
        {"session_id": "s1", "cwd": "/tmp", "prompt": "hello"},
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    # No additionalContext should be printed.
    assert res.stdout.strip() == "" or "additionalContext" not in res.stdout


def test_hook_emits_one_nudge_then_consumes(tmp_path: Path):
    nudges_path = tmp_path / "pending-nudges.md"
    nudges_path.write_text(
        "---\nid: a1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/a\n---\nfirst nudge text\n"
        "---\nid: b2\ncreated: 2026-05-19T00:00:01+00:00\nlesson: l/b\n---\nsecond nudge text\n",
        encoding="utf-8",
    )

    res = _run_hook(
        {"session_id": "s1", "cwd": "/tmp", "prompt": "do thing"},
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    stdout = res.stdout.strip()
    assert stdout, "hook should emit additionalContext when nudges queued"
    output = json.loads(stdout)
    ctx = output["hookSpecificOutput"]["additionalContext"]
    # Only ONE nudge per turn.
    assert "first nudge text" in ctx
    assert "second nudge text" not in ctx
    # Nudges file should be cleared (renamed to .consumed).
    assert not nudges_path.exists()
    assert (tmp_path / "pending-nudges.md.consumed").exists()
    # Budget state file written.
    budget = tmp_path / "state" / "nudge-budget-s1.json"
    assert budget.exists()
    data = json.loads(budget.read_text(encoding="utf-8"))
    assert data["consumed"] == 1


def test_hook_skips_nudges_when_no_session_id(tmp_path: Path):
    nudges_path = tmp_path / "pending-nudges.md"
    nudges_path.write_text(
        "---\nid: a1\ncreated: t\nlesson: l/a\n---\norphan\n",
        encoding="utf-8",
    )
    res = _run_hook(
        {"cwd": "/tmp", "prompt": "hi"},  # no session_id
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0
    assert "additionalContext" not in res.stdout
    # Without session_id we don't drain — leave the file intact.
    assert nudges_path.exists()
