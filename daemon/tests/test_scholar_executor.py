"""Tests for the deterministic Scholar executor: each action type applied
to a fixture library, plus index.md / log.md maintenance and the
failure-tolerant batch behaviour."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent_mem_daemon import scholar_executor
from agent_mem_daemon._schemas import ScholarDecisions

from .conftest import scholar_entry_body, seed_scholar_tree

NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


def _body(id_: str, *, scope: str = "global", extra: str = "") -> str:
    return scholar_entry_body(id_, scope=scope, extra=extra)


def _seed(tmp_path: Path) -> Path:
    return seed_scholar_tree(tmp_path)


def _decisions(*actions) -> ScholarDecisions:
    return ScholarDecisions.model_validate({"actions": list(actions), "interrupts_processed": []})


def _apply(k: Path, *actions) -> scholar_executor.ExecResult:
    return scholar_executor.apply_decisions(_decisions(*actions), k, session_id="s1", now=NOW)


# ── write_entry ───────────────────────────────────────────────────────


def test_write_entry_creates_file_and_index_and_log(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new"),
            "reasoning": "r",
        },
    )
    assert res.counts["actions_applied"] == 1
    assert res.counts["write_entry"] == 1
    assert (k / "global" / "python" / "new.md").exists()
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert "[[global/python/new]]" in index_md
    assert "| global |" in index_md  # scope column derived from frontmatter
    log_md = (k / "log.md").read_text(encoding="utf-8")
    assert "write_entry | global/python/new.md" in log_md


def test_write_entry_replaces_existing_index_row(tmp_path: Path):
    k = _seed(tmp_path)
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new"),
            "reasoning": "r",
        },
    )
    # A second write to the same path must replace, not duplicate, the row.
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new", extra=" updated"),
            "reasoning": "r",
        },
    )
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert index_md.count("[[global/python/new]]") == 1


# ── update_entry ──────────────────────────────────────────────────────


def test_update_entry_overwrites_body(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "update_entry",
            "path": "global/python/use-uv.md",
            "new_body": _body("use-uv", extra=" Now with a new clause."),
            "reasoning": "r",
        },
    )
    assert res.counts["update_entry"] == 1
    assert "new clause" in (k / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")


# ── merge_entries ─────────────────────────────────────────────────────


def test_merge_entries_writes_target_and_archives_sources(tmp_path: Path):
    k = _seed(tmp_path)
    (k / "global" / "python" / "a.md").write_text(_body("a"), encoding="utf-8")
    (k / "global" / "python" / "b.md").write_text(_body("b"), encoding="utf-8")
    res = _apply(
        k,
        {
            "action": "merge_entries",
            "source_paths": ["global/python/a.md", "global/python/b.md"],
            "target_path": "global/python/merged.md",
            "target_body": _body("merged"),
            "reasoning": "r",
        },
    )
    assert res.counts["merge_entries"] == 1
    assert (k / "global" / "python" / "merged.md").exists()
    # Sources archived (moved out), not left in place.
    assert not (k / "global" / "python" / "a.md").exists()
    assert (k / "_archive" / "global" / "python" / "a.md").exists()
    archived = (k / "_archive" / "global" / "python" / "a.md").read_text(encoding="utf-8")
    assert "status: stale" in archived


# ── move_entry ────────────────────────────────────────────────────────


def test_move_entry_relocates_and_rewrites_index(tmp_path: Path):
    k = _seed(tmp_path)
    # Seed an index row for the entry being moved.
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/use-uv.md",
            "body": _body("use-uv"),
            "reasoning": "seed index",
        },
    )
    res = _apply(
        k,
        {
            "action": "move_entry",
            "from_path": "global/python/use-uv.md",
            "to_path": "global/tooling/use-uv.md",
        },
    )
    assert res.counts["move_entry"] == 1
    assert not (k / "global" / "python" / "use-uv.md").exists()
    assert (k / "global" / "tooling" / "use-uv.md").exists()
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert "[[global/tooling/use-uv]]" in index_md
    assert "[[global/python/use-uv]]" not in index_md


def test_move_entry_failure_recorded(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "move_entry",
            "from_path": "global/python/does-not-exist.md",
            "to_path": "global/tooling/x.md",
        },
    )
    assert res.counts.get("actions_failed") == 1


# ── archive_entry ─────────────────────────────────────────────────────


def test_archive_entry_moves_and_stamps(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k, {"action": "archive_entry", "path": "global/python/use-uv.md", "reasoning": "r"}
    )
    assert res.counts["archive_entry"] == 1
    assert not (k / "global" / "python" / "use-uv.md").exists()
    archived = (k / "_archive" / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")
    assert "status: stale" in archived
    assert "2026-05-19" in archived
    assert "archived:" in archived


def test_archive_missing_entry_recorded_as_failure(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(k, {"action": "archive_entry", "path": "global/python/ghost.md", "reasoning": "r"})
    assert res.counts.get("actions_failed") == 1


# ── deprecate_entry ───────────────────────────────────────────────────


def test_deprecate_entry_marks_in_place(tmp_path: Path):
    k = _seed(tmp_path)
    (k / "global" / "python" / "newer.md").write_text(_body("newer"), encoding="utf-8")
    res = _apply(
        k,
        {
            "action": "deprecate_entry",
            "path": "global/python/use-uv.md",
            "superseded_by": "global/python/newer.md",
            "reasoning": "r",
        },
    )
    assert res.counts["deprecate_entry"] == 1
    # File kept in place so inbound wikilinks still resolve.
    text = (k / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")
    assert "status: deprecated" in text
    assert "superseded_by: global/python/newer.md" in text
    assert "Superseded by [[global/python/newer]]" in text


def test_deprecate_missing_entry_recorded_as_failure(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "deprecate_entry",
            "path": "global/python/ghost.md",
            "superseded_by": "global/python/use-uv.md",
            "reasoning": "r",
        },
    )
    assert res.counts.get("actions_failed") == 1


# ── batch behaviour ───────────────────────────────────────────────────


def test_one_bad_action_does_not_abort_batch(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        # good
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new"),
            "reasoning": "r",
        },
        # bad (archive of a non-existent file)
        {"action": "archive_entry", "path": "global/python/ghost.md", "reasoning": "r"},
    )
    assert res.counts.get("write_entry") == 1
    assert res.counts.get("actions_failed") == 1
    assert (k / "global" / "python" / "new.md").exists()


def test_empty_decisions_no_op(tmp_path: Path):
    k = _seed(tmp_path)
    res = scholar_executor.apply_decisions(_decisions(), k, session_id="s1", now=NOW)
    assert res.counts == {}
    assert res.notes == []


def test_index_created_when_absent(tmp_path: Path):
    k = tmp_path / "knowledge"
    k.mkdir()
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/x.md",
            "body": _body("x"),
            "reasoning": "r",
        },
    )
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert "Knowledge Index" in index_md
    assert "[[global/x]]" in index_md


def test_dispatch_handler_exception_recorded(tmp_path: Path, monkeypatch):
    """If a handler raises unexpectedly, apply_decisions records a failure
    rather than propagating."""
    k = _seed(tmp_path)

    def _boom(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(scholar_executor, "_do_write", _boom)
    res = _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new"),
            "reasoning": "r",
        },
    )
    assert res.counts.get("actions_failed") == 1
