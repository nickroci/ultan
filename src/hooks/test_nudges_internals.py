"""Cover the lower-level helpers in ``_nudges`` that the public
``take_nudges`` tests don't reach: the budget reader/writer (used
when no test seam is passed), the read-and-clear edge cases, and the
project-bucket filter classifier."""

from __future__ import annotations

import json
from pathlib import Path

import _nudges


def test_lesson_project_bucket_recognises_projects_path():
    assert _nudges._lesson_project_bucket("projects/foo/x.md") == "foo"


def test_lesson_project_bucket_recognises_global():
    assert _nudges._lesson_project_bucket("global/x.md") == "__global__"


def test_lesson_project_bucket_handles_missing():
    assert _nudges._lesson_project_bucket("") is None
    assert _nudges._lesson_project_bucket("daily/2026-05-19.md") is None


def test_lesson_project_bucket_strips_leading_slash():
    assert _nudges._lesson_project_bucket("/projects/foo/x.md") == "foo"


def test_load_budget_with_no_file_returns_zero(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    out = _nudges._load_budget("session-1")
    assert out == {"consumed": 0}


def test_load_budget_with_existing_file_returns_count(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    state = tmp_path / "state"
    state.mkdir()
    (state / "nudge-budget-session-1.json").write_text(
        json.dumps({"consumed": 2, "updated": 1.0}), encoding="utf-8"
    )
    out = _nudges._load_budget("session-1")
    assert out["consumed"] == 2


def test_load_budget_handles_corrupted_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    state = tmp_path / "state"
    state.mkdir()
    (state / "nudge-budget-session-1.json").write_text("not json", encoding="utf-8")
    out = _nudges._load_budget("session-1")
    assert out == {"consumed": 0}


def test_save_budget_writes_atomically(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    _nudges._save_budget("session-1", 2)
    state_file = tmp_path / "state" / "nudge-budget-session-1.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["consumed"] == 2


def test_save_budget_no_test_seam_via_take_nudges(tmp_path: Path, monkeypatch):
    """When ``_budget_state_path`` isn't passed, take_nudges goes
    through the production ``_save_budget`` path."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    nudges_path = tmp_path / "pending-nudges.md"
    nudges_path.write_text(
        "---\nid: a\ncreated: t\nlesson: global/x\n---\nbody\n",
        encoding="utf-8",
    )
    sel, consumed = _nudges.take_nudges(
        "session-prod-1",
        _nudges_path=nudges_path,
    )
    assert len(sel) == 1
    assert consumed == 1
    state_file = tmp_path / "state" / "nudge-budget-session-prod-1.json"
    assert state_file.exists()


def test_save_budget_swallows_disk_error(tmp_path: Path, monkeypatch):
    """If the parent dir mkdir fails, _save_budget returns silently."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))

    # Make mkdir raise.
    orig_mkdir = Path.mkdir

    def boom(self, *args, **kwargs):
        if "state" in str(self):
            raise OSError("nope")
        return orig_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", boom)
    _nudges._save_budget("s1", 1)  # must not raise


def test_save_budget_handles_mkstemp_failure(tmp_path: Path, monkeypatch):
    """tempfile failure → returns silently."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    (tmp_path / "state").mkdir()

    def boom(*a, **kw):
        raise OSError("no temp")

    monkeypatch.setattr(_nudges.tempfile, "mkstemp", boom)
    _nudges._save_budget("s1", 1)  # must not raise


def test_read_and_clear_returns_empty_for_missing(tmp_path: Path):
    out = _nudges._read_and_clear_nudges_file(tmp_path / "nope.md")
    assert out == ""


def test_read_and_clear_handles_empty_file(tmp_path: Path):
    f = tmp_path / "empty.md"
    f.write_text("", encoding="utf-8")
    out = _nudges._read_and_clear_nudges_file(f)
    assert out == ""
    # Empty file gets cleaned up.
    assert not f.exists()


def test_budget_state_path_sanitises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    p = _nudges.budget_state_path("session/with/slashes")
    assert "/" not in p.name
    assert "session-with-slashes" in p.name


def test_budget_state_path_empty_session_defaults_to_unknown(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    p = _nudges.budget_state_path("")
    assert p.name == "nudge-budget-unknown.json"


def test_pending_nudges_path_uses_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    assert _nudges.pending_nudges_path() == tmp_path / "pending-nudges.md"


def test_pending_nudges_path_default(monkeypatch):
    monkeypatch.delenv("AGENT_MEM_HOME", raising=False)
    assert _nudges.pending_nudges_path() == Path.home() / ".agent-mem" / "pending-nudges.md"


def test_take_nudges_corrupted_budget_state_resets(tmp_path: Path, monkeypatch):
    """A corrupted state file shouldn't crash take_nudges; treated as
    fresh session (consumed=0)."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    nudges_path = tmp_path / "pending-nudges.md"
    nudges_path.write_text(
        "---\nid: a\ncreated: t\nlesson: global/x\n---\nbody\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    state_path.parent.mkdir()
    state_path.write_text("garbage", encoding="utf-8")
    sel, consumed = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert len(sel) == 1
    assert consumed == 1


def test_requeue_nudges_empty_list_is_noop(tmp_path: Path):
    """Empty requeue list → file shouldn't appear."""
    _nudges._requeue_nudges(tmp_path / "should-not-exist.md", [])
    assert not (tmp_path / "should-not-exist.md").exists()


def test_render_context_handles_unknown_lesson():
    """A nudge with no lesson path falls back to a placeholder."""
    n = _nudges.Nudge(id="x", created="t", lesson="", text="body")
    out = _nudges.render_context([n])
    assert "(unknown lesson)" in out


def test_nudge_matches_project_no_bucket_treated_universal():
    """A lesson path with no recognised bucket structure delivers
    everywhere (over-deliver > silently drop)."""
    n = _nudges.Nudge(id="a", created="t", lesson="weird/random/path.md", text="x")
    assert _nudges._nudge_matches_project(n, "any-project", {}) is True


def test_nudge_matches_project_loads_aliases_on_demand(tmp_path: Path, monkeypatch):
    """When aliases=None, _nudge_matches_project loads the alias file."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    (tmp_path / "project-aliases.json").write_text('{"alpha": "beta"}', encoding="utf-8")
    n = _nudges.Nudge(id="a", created="t", lesson="projects/alpha/x.md", text="x")
    # No aliases passed → loads from disk → "alpha" canonicalises to "beta".
    assert _nudges._nudge_matches_project(n, "beta") is True
    assert _nudges._nudge_matches_project(n, "alpha") is False
