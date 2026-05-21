"""Tests for the in-process ``move_entries`` MCP tool.

The tool runs deterministic Python — no LLM in the loop — so we can
test the implementation function directly without spinning up the
Claude Agent SDK.
"""
from __future__ import annotations

from pathlib import Path

from agent_mem_daemon.library_tools import (
    _move_entries_impl,
    _rewrite_wikilinks_in_text,
    _path_to_wikilink,
)


def _text(node: dict) -> str:
    return node["content"][0]["text"]


def _seed_library(tmp_path: Path) -> Path:
    """Create a minimal knowledge dir with two entries and a README that
    references them. Returns the knowledge root."""
    root = tmp_path / "knowledge"
    user_dir = root / "global" / "user"
    user_dir.mkdir(parents=True)

    (user_dir / "foo.md").write_text(
        "---\nname: foo\n---\n\n# Foo\n\nBody.\n",
        encoding="utf-8",
    )
    (user_dir / "bar.md").write_text(
        "---\nname: bar\n---\n\n# Bar\n\nRelated: [[global/user/foo]]\n",
        encoding="utf-8",
    )
    # Index references both entries by full path.
    (root / "index.md").write_text(
        "# Index\n"
        "- [[global/user/foo]] — foo entry\n"
        "- [[global/user/foo|the foo entry]] — with alias\n"
        "- [[global/user/bar]] — bar entry\n",
        encoding="utf-8",
    )
    return root


def test_move_one_file_rewrites_inbound_links(tmp_path: Path) -> None:
    root = _seed_library(tmp_path)

    result = _move_entries_impl(
        root,
        {
            "to_folder": "global/user/profile",
            "files": ["global/user/foo.md"],
            "readme": "# Profile\n\nUser profile entries.\n",
        },
    )
    text = _text(result)

    assert "move_entries: ok" in text
    assert "rewrote 3 inbound wikilink(s)" in text

    # File moved.
    assert not (root / "global/user/foo.md").exists()
    assert (root / "global/user/profile/foo.md").exists()

    # README created.
    readme = (root / "global/user/profile/README.md").read_text()
    assert readme.startswith("# Profile")

    # Inbound wikilinks rewritten in index.md (preserving alias).
    index = (root / "index.md").read_text()
    assert "[[global/user/profile/foo]]" in index
    assert "[[global/user/profile/foo|the foo entry]]" in index
    assert "[[global/user/foo]]" not in index
    assert "[[global/user/foo|" not in index

    # Inbound link in bar.md also rewritten.
    bar = (root / "global/user/bar.md").read_text()
    assert "[[global/user/profile/foo]]" in bar


def test_split_folder_moves_two_files_with_one_call(tmp_path: Path) -> None:
    root = _seed_library(tmp_path)

    result = _move_entries_impl(
        root,
        {
            "to_folder": "global/user/profile",
            "files": ["global/user/foo.md", "global/user/bar.md"],
            "readme": "# Profile\n",
        },
    )
    text = _text(result)
    assert "move_entries: ok" in text

    assert (root / "global/user/profile/foo.md").exists()
    assert (root / "global/user/profile/bar.md").exists()
    assert not (root / "global/user/foo.md").exists()
    assert not (root / "global/user/bar.md").exists()

    index = (root / "index.md").read_text()
    assert "[[global/user/profile/foo]]" in index
    assert "[[global/user/profile/bar]]" in index
    assert "[[global/user/foo]]" not in index
    assert "[[global/user/bar]]" not in index


def test_existing_readme_is_left_untouched(tmp_path: Path) -> None:
    root = _seed_library(tmp_path)
    (root / "global/user/profile").mkdir(parents=True)
    (root / "global/user/profile/README.md").write_text(
        "# Pre-existing\n",
        encoding="utf-8",
    )

    result = _move_entries_impl(
        root,
        {
            "to_folder": "global/user/profile",
            "files": ["global/user/foo.md"],
            "readme": "# Replacement (should be ignored)\n",
        },
    )
    text = _text(result)
    assert "left existing" in text
    # README content was NOT replaced.
    assert (
        (root / "global/user/profile/README.md").read_text()
        == "# Pre-existing\n"
    )


def test_rejects_path_escape(tmp_path: Path) -> None:
    root = _seed_library(tmp_path)

    result = _move_entries_impl(
        root,
        {
            "to_folder": "../escape",
            "files": ["global/user/foo.md"],
            "readme": "",
        },
    )
    assert "outside the knowledge root" in _text(result)
    # Nothing actually moved.
    assert (root / "global/user/foo.md").exists()


def test_rejects_missing_source(tmp_path: Path) -> None:
    root = _seed_library(tmp_path)
    result = _move_entries_impl(
        root,
        {
            "to_folder": "global/user/profile",
            "files": ["global/user/nope.md"],
            "readme": "",
        },
    )
    assert "does not exist" in _text(result)


def test_rejects_overwrite(tmp_path: Path) -> None:
    root = _seed_library(tmp_path)
    (root / "global/user/profile").mkdir(parents=True)
    (root / "global/user/profile/foo.md").write_text(
        "preexisting", encoding="utf-8",
    )

    result = _move_entries_impl(
        root,
        {
            "to_folder": "global/user/profile",
            "files": ["global/user/foo.md"],
            "readme": "",
        },
    )
    assert "already exists" in _text(result)
    # Both source and destination untouched.
    assert (root / "global/user/foo.md").exists()
    assert (
        (root / "global/user/profile/foo.md").read_text() == "preexisting"
    )


def test_rewrite_helper_preserves_aliases_and_md_suffix() -> None:
    text = (
        "[[a/foo]] and [[a/foo|nice]] and [[a/foo.md]] and "
        "[[a/foo|alias with spaces]] and [[other]]"
    )
    out, n = _rewrite_wikilinks_in_text(text, {"a/foo": "b/foo"})
    assert n == 4
    assert "[[b/foo]]" in out
    assert "[[b/foo|nice]]" in out
    assert "[[b/foo|alias with spaces]]" in out
    # Unmapped link untouched.
    assert "[[other]]" in out
    # The .md-suffixed form should also get rewritten.
    # (We strip .md when normalising, so it maps and the rewrite drops the suffix.)
    assert "[[a/foo.md]]" not in out


def test_path_to_wikilink_strips_md_suffix(tmp_path: Path) -> None:
    root = tmp_path / "k"
    (root / "global" / "user").mkdir(parents=True)
    p = root / "global" / "user" / "foo.md"
    p.write_text("hi")
    assert _path_to_wikilink(p, root.resolve()) == "global/user/foo"
