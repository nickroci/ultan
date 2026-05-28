"""Tests for the Scholar action-model boundary validators (frontmatter
parses, required fields present, id matches the filename slug, scope agrees
with the path) and the shared ``_validation`` helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_mem_daemon import _validation
from agent_mem_daemon._schemas import (
    ScholarArchiveEntry,
    ScholarDecisions,
    ScholarDeprecateEntry,
    ScholarMergeEntries,
    ScholarMoveEntry,
    ScholarUpdateEntry,
    ScholarWriteEntry,
)

from .conftest import scholar_entry_body


def _body(id_: str, *, scope: str = "global", extra: str = "") -> str:
    return scholar_entry_body(id_, scope=scope, extra=extra)


# ── ScholarWriteEntry ────────────────────────────────────────────────


def test_write_entry_valid():
    a = ScholarWriteEntry(path="global/python/use-uv.md", body=_body("use-uv"), reasoning="r")
    assert a.action == "write_entry"


def test_write_entry_rejects_missing_body():
    with pytest.raises(ValidationError):
        ScholarWriteEntry(path="global/python/x.md", body="", reasoning="r")


def test_write_entry_rejects_unparseable_frontmatter():
    with pytest.raises(ValidationError) as exc:
        ScholarWriteEntry(path="global/python/x.md", body="# no frontmatter\n\nbody", reasoning="r")
    assert "frontmatter" in str(exc.value)


def test_write_entry_rejects_missing_required_fields():
    body = "---\nid: x\nscope: global\n---\n\n# x\n\nbody.\n"
    with pytest.raises(ValidationError) as exc:
        ScholarWriteEntry(path="global/python/x.md", body=body, reasoning="r")
    assert "missing required fields" in str(exc.value)


def test_write_entry_rejects_id_slug_mismatch():
    with pytest.raises(ValidationError) as exc:
        ScholarWriteEntry(path="global/python/use-uv.md", body=_body("WRONG"), reasoning="r")
    assert "does not match the filename slug" in str(exc.value)


def test_write_entry_rejects_scope_path_mismatch():
    # scope=global but path under projects/ → mismatch.
    with pytest.raises(ValidationError) as exc:
        ScholarWriteEntry(path="projects/foo/x.md", body=_body("x", scope="global"), reasoning="r")
    assert "scope/path mismatch" in str(exc.value)


def test_write_entry_project_scope_ok():
    a = ScholarWriteEntry(
        path="projects/foo/x.md",
        body=_body("x", scope="project:foo"),
        reasoning="r",
    )
    assert a.path == "projects/foo/x.md"


# ── other action models ──────────────────────────────────────────────


def test_update_entry_validates_new_body():
    with pytest.raises(ValidationError):
        ScholarUpdateEntry(path="global/x.md", new_body="", reasoning="r")
    ok = ScholarUpdateEntry(path="global/x.md", new_body=_body("x"), reasoning="r")
    assert ok.action == "update_entry"


def test_merge_entries_requires_sources_and_body():
    with pytest.raises(ValidationError):
        ScholarMergeEntries(
            source_paths=[], target_path="global/m.md", target_body=_body("m"), reasoning="r"
        )
    ok = ScholarMergeEntries(
        source_paths=["global/a.md"],
        target_path="global/m.md",
        target_body=_body("m"),
        reasoning="r",
    )
    assert ok.action == "merge_entries"


def test_move_entry_requires_both_paths():
    with pytest.raises(ValidationError):
        ScholarMoveEntry(from_path="global/a.md", to_path="", reasoning="r")
    ok = ScholarMoveEntry(from_path="global/a.md", to_path="global/b/a.md", reasoning="r")
    assert ok.action == "move_entry"


def test_archive_entry_requires_path():
    with pytest.raises(ValidationError):
        ScholarArchiveEntry(path="", reasoning="r")
    assert ScholarArchiveEntry(path="global/a.md", reasoning="r").action == "archive_entry"


def test_deprecate_entry_requires_both():
    with pytest.raises(ValidationError):
        ScholarDeprecateEntry(path="global/a.md", superseded_by="", reasoning="r")
    ok = ScholarDeprecateEntry(path="global/a.md", superseded_by="global/b.md", reasoning="r")
    assert ok.action == "deprecate_entry"


# ── ScholarDecisions union discrimination ────────────────────────────


def test_decisions_discriminates_actions():
    decisions = ScholarDecisions.model_validate(
        {
            "actions": [
                {"action": "archive_entry", "path": "global/a.md", "reasoning": "r"},
                {"action": "move_entry", "from_path": "global/a.md", "to_path": "global/b/a.md"},
            ],
            "interrupts_processed": [],
        }
    )
    assert isinstance(decisions.actions[0], ScholarArchiveEntry)
    assert isinstance(decisions.actions[1], ScholarMoveEntry)


# ── _validation helper edges ─────────────────────────────────────────


def test_parse_frontmatter_unparseable_yaml_returns_empty():
    assert _validation.parse_frontmatter("---\n: : : bad\n---\nbody") == {}


def test_parse_frontmatter_non_mapping_returns_empty():
    assert _validation.parse_frontmatter("---\n- just\n- a\n- list\n---\nbody") == {}


def test_strip_frontmatter_no_block_returns_text():
    assert _validation.strip_frontmatter("no block here") == "no block here"


def test_path_slug_strips_md():
    assert _validation.path_slug("global/python/use-uv.md") == "use-uv"
    assert _validation.path_slug("flat") == "flat"


def test_scope_path_violation_project_slug_mismatch():
    msg = _validation.scope_path_violation(Path("projects/bar/x.md"), "project:foo")
    assert msg is not None and "projects/foo" in msg


def test_scope_path_violation_empty_parts():
    assert _validation.scope_path_violation(Path(""), "global") is None


def test_wikilink_resolves_archive_and_daily():
    root = Path("/nonexistent-root")
    assert _validation.wikilink_resolves("_archive/foo", root, root)
    assert _validation.wikilink_resolves("daily/2026-05-19", root, root)
