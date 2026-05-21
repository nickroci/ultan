"""Tests for the Scholar's pre-write wikilink guard.

The guard inspects every Write/Edit tool call and either:
  - lets it through unchanged (all links resolve),
  - auto-repairs broken links whose target has exactly one same-leaf
    match elsewhere in the tree, OR
  - denies the call when a broken link can't be auto-resolved.

Tests target the pure helpers in llm.py so they don't need the Claude
Agent SDK to actually run.
"""
from __future__ import annotations

from pathlib import Path

from agent_mem_daemon.llm import (
    _check_and_repair_writes,
    _find_unique_leaf,
    _resolve_wikilink,
    _rewrite_link_in_text,
)


def _seed(tmp_path: Path) -> Path:
    """Knowledge tree with a couple of entries to resolve against."""
    root = tmp_path / "knowledge"
    (root / "global" / "user" / "profile").mkdir(parents=True)
    (root / "global" / "user" / "profile" / "foo.md").write_text("foo")
    (root / "global" / "user" / "profile" / "README.md").write_text("# Profile")
    (root / "projects" / "x").mkdir(parents=True)
    (root / "projects" / "x" / "bar.md").write_text("bar")
    (root / "projects" / "x" / "README.md").write_text("# X")
    # Duplicate leaf name to test ambiguity guard.
    (root / "projects" / "y").mkdir(parents=True)
    (root / "projects" / "y" / "dup.md").write_text("dup-y")
    (root / "projects" / "z").mkdir(parents=True)
    (root / "projects" / "z" / "dup.md").write_text("dup-z")
    return root


def test_resolve_full_path_link(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    assert _resolve_wikilink("global/user/profile/foo", root, root / "index.md")


def test_resolve_archive_and_daily_links_always_ok(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    assert _resolve_wikilink("_archive/old/thing", root, root / "x.md")
    assert _resolve_wikilink("daily/2026-05-21", root, root / "x.md")


def test_resolve_folder_shaped_link(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    assert _resolve_wikilink(
        "global/user/profile/", root, root / "index.md",
    )


def test_resolve_sibling_fallback(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    # The README in projects/x can reference its sibling as just [[bar]].
    assert _resolve_wikilink(
        "bar", root, root / "projects" / "x" / "README.md",
    )


def test_resolve_returns_false_for_truly_broken(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    assert not _resolve_wikilink("nope/missing", root, root / "x.md")


def test_unique_leaf_lookup_finds_canonical(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    assert _find_unique_leaf("foo", root) == "global/user/profile/foo"
    assert _find_unique_leaf("bar", root) == "projects/x/bar"


def test_unique_leaf_lookup_returns_none_for_ambiguous(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    assert _find_unique_leaf("dup", root) is None


def test_unique_leaf_lookup_returns_none_for_missing(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    assert _find_unique_leaf("nope", root) is None


def test_unique_leaf_skips_archive_matches(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    (root / "_archive" / "global" / "user").mkdir(parents=True)
    (root / "_archive" / "global" / "user" / "foo.md").write_text("old foo")
    # Still finds the live one only (the _archive copy is excluded).
    assert _find_unique_leaf("foo", root) == "global/user/profile/foo"


def test_rewrite_preserves_alias() -> None:
    text = "See [[old/path]] and [[old/path|nice name]]."
    out = _rewrite_link_in_text(text, "old/path", "new/spot")
    assert "[[new/spot]]" in out
    assert "[[new/spot|nice name]]" in out


def test_rewrite_handles_md_suffix() -> None:
    text = "Link: [[old/path.md]]"
    out = _rewrite_link_in_text(text, "old/path", "new/spot")
    # The rewrite normalises away the .md suffix on the target.
    assert "[[new/spot]]" in out


# ── _check_and_repair_writes ─────────────────────────────────────────


def _write_input(root: Path, content: str) -> dict:
    return {"file_path": str(root / "index.md"), "content": content}


def _edit_input(root: Path, new_string: str) -> dict:
    return {
        "file_path": str(root / "index.md"),
        "old_string": "PLACEHOLDER",
        "new_string": new_string,
    }


def test_write_with_all_good_links_passes_through(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    status, _, _ = _check_and_repair_writes(
        "Write",
        _write_input(root, "See [[global/user/profile/foo]]"),
        root,
    )
    assert status == "allow"


def test_write_auto_repairs_broken_link_with_unique_leaf(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    status, new_input, summary = _check_and_repair_writes(
        "Write",
        _write_input(root, "See [[global/user/foo]] — pre-split path"),
        root,
    )
    assert status == "allow_with_repair"
    assert new_input is not None
    assert "[[global/user/profile/foo]]" in new_input["content"]
    assert "[[global/user/foo]] —" not in new_input["content"]
    assert "global/user/foo" in summary


def test_edit_auto_repairs_in_new_string(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    status, new_input, _ = _check_and_repair_writes(
        "Edit",
        _edit_input(root, "Related: [[global/user/foo]]"),
        root,
    )
    assert status == "allow_with_repair"
    assert new_input is not None
    assert "[[global/user/profile/foo]]" in new_input["new_string"]


def test_write_with_unresolvable_link_is_denied(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    status, payload, info = _check_and_repair_writes(
        "Write",
        _write_input(root, "See [[totally/missing/entry]]"),
        root,
    )
    assert status == "deny"
    assert payload is None
    assert "totally/missing/entry" in info
    assert "do not resolve" in info or "does not" in info


def test_write_with_ambiguous_leaf_is_denied(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    # `dup` exists in projects/y AND projects/z — can't auto-resolve.
    status, _, _ = _check_and_repair_writes(
        "Write",
        _write_input(root, "Ambiguous: [[some/dup]]"),
        root,
    )
    assert status == "deny"


def test_log_md_writes_are_exempt(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    # log.md often quotes vetoed paths that don't resolve — by design.
    status, _, _ = _check_and_repair_writes(
        "Write",
        {
            "file_path": str(root / "log.md"),
            "content": "vetoed: [[totally/missing/entry]] — reason: bad path",
        },
        root,
    )
    assert status == "allow"


def test_write_with_code_span_link_is_ignored(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    # `[[in/code]]` inside backticks shouldn't trigger the guard at all.
    status, _, _ = _check_and_repair_writes(
        "Write",
        _write_input(root, "Syntax: `[[some/missing]]` is a placeholder"),
        root,
    )
    assert status == "allow"


def test_write_with_empty_content_passes_through(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    status, _, _ = _check_and_repair_writes(
        "Write",
        {"file_path": str(root / "index.md"), "content": ""},
        root,
    )
    assert status == "allow"


def test_mixed_repairable_and_unresolvable_denies_with_note(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    status, _, msg = _check_and_repair_writes(
        "Write",
        _write_input(
            root,
            "Good but moved: [[global/user/foo]] "
            "Bad: [[totally/missing/entry]]",
        ),
        root,
    )
    assert status == "deny"
    # The unresolvable one is mentioned.
    assert "totally/missing/entry" in msg
    # And the note mentions the auto-resolvable one too.
    assert "would auto-resolve" in msg
