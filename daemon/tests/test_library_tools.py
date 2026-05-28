"""Tests for ``library_tools.py``.

Coverage focus:
  - ``_move_entries_impl``: real move + wikilink rewrite against a tmp
    knowledge tree, error paths (escape, missing source, wrong suffix,
    destination collision, partial-read failure).
  - ``_rewrite_wikilinks_in_text``: alias preservation, .md normalisation.
  - ``_path_to_wikilink`` edge cases (README folder-shape).
  - ``_normalize_link``, ``_safe_inside``.

The bm25 / embedding search runners (``run_bm25_search`` /
``run_embedding_search``) are pure-Python in-process entry points the
curator agents call directly; their branches are covered below.
"""

from __future__ import annotations

from pathlib import Path

from agent_mem_daemon import library_tools


def _entry_text(id_: str, body: str = "") -> str:
    """Tiny entry skeleton — we only need wikilink-bearing markdown for
    most tests, not a full frontmatter."""
    return f"""---
id: {id_}
type: lesson
scope: global
status: provisional
confidence: 0.7
applies-when: |
  any
keywords: [demo]
title: "{id_}"
created: 2026-05-19
updated: 2026-05-19
fired: 0
fired-helpful: 0
sources:
  - manual
---

# {id_}

{body}
"""


# ── Tiny pure helpers ────────────────────────────────────────────────────


def test_normalize_link_strips_md_suffix() -> None:
    assert library_tools._normalize_link("global/foo.md") == "global/foo"
    assert library_tools._normalize_link("  global/bar  ") == "global/bar"
    assert library_tools._normalize_link("plain") == "plain"


def test_safe_inside_accepts_inside(tmp_path: Path) -> None:
    inside = tmp_path / "inside.md"
    inside.write_text("x")
    assert library_tools._safe_inside(tmp_path, inside) is True


def test_safe_inside_rejects_outside(tmp_path: Path) -> None:
    other = tmp_path.parent / "elsewhere.md"
    assert library_tools._safe_inside(tmp_path, other) is False


def test_path_to_wikilink_strips_md_suffix(tmp_path: Path) -> None:
    (tmp_path / "global").mkdir()
    p = tmp_path / "global" / "x.md"
    p.write_text("x")
    assert library_tools._path_to_wikilink(p, tmp_path) == "global/x"


def test_path_to_wikilink_keeps_readme_intact(tmp_path: Path) -> None:
    """READMEs are folder-shaped wikilinks; the helper leaves the file
    name in place so we don't accidentally rewrite a folder reference."""
    (tmp_path / "global").mkdir()
    readme = tmp_path / "global" / "README.md"
    readme.write_text("# global")
    link = library_tools._path_to_wikilink(readme, tmp_path)
    # The README path is returned WITH .md (the comment in the source
    # says "callers shouldn't pass them" — but we still pin behaviour).
    assert link == "global/README.md"


# ── _rewrite_wikilinks_in_text ───────────────────────────────────────────


def test_rewrite_wikilinks_changes_matching_targets() -> None:
    text = "See [[old/foo]] and unrelated [[old/bar]]."
    new, n = library_tools._rewrite_wikilinks_in_text(text, {"old/foo": "new/foo"})
    assert "[[new/foo]]" in new
    assert "[[old/bar]]" in new
    assert n == 1


def test_rewrite_wikilinks_preserves_alias() -> None:
    text = "See [[old/foo|cool name]]."
    new, n = library_tools._rewrite_wikilinks_in_text(text, {"old/foo": "new/foo"})
    assert "[[new/foo|cool name]]" in new
    assert n == 1


def test_rewrite_wikilinks_handles_md_suffix() -> None:
    text = "Reference: [[old/foo.md]]."
    new, n = library_tools._rewrite_wikilinks_in_text(text, {"old/foo": "new/foo"})
    assert "[[new/foo]]" in new
    assert n == 1


def test_rewrite_wikilinks_no_matches() -> None:
    text = "[[stay/here]] [[stay/there]]"
    new, n = library_tools._rewrite_wikilinks_in_text(text, {"old/foo": "new/foo"})
    assert new == text
    assert n == 0


# ── _move_entries_impl: happy paths ─────────────────────────────────────


def test_move_entries_moves_files_and_rewrites_inbound_links(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "global" / "foo").mkdir(parents=True)
    src = root / "global" / "foo" / "old-name.md"
    src.write_text(_entry_text("old-name", body="content"))
    # Sibling entry that wikilinks to the soon-to-be-moved one.
    citer = root / "global" / "citer.md"
    citer.write_text("References: [[global/foo/old-name]].")
    # Another citer with an alias.
    citer2 = root / "global" / "citer-alias.md"
    citer2.write_text("With alias: [[global/foo/old-name|the rule]].")

    out = library_tools._move_entries_impl(
        root,
        {"to_folder": "global/bar", "files": ["global/foo/old-name.md"], "readme": ""},
    )
    text = out["content"][0]["text"]
    assert text.startswith("move_entries: ok")
    # File physically moved.
    assert not src.exists()
    assert (root / "global" / "bar" / "old-name.md").exists()
    # Citer rewritten to point at the new path.
    assert "[[global/bar/old-name]]" in citer.read_text(encoding="utf-8")
    # Alias preserved.
    assert "[[global/bar/old-name|the rule]]" in citer2.read_text(encoding="utf-8")


def test_move_entries_writes_readme_when_provided(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    src = root / "src" / "thing.md"
    src.write_text(_entry_text("thing"))
    out = library_tools._move_entries_impl(
        root,
        {
            "to_folder": "dst",
            "files": ["src/thing.md"],
            "readme": "# Destination\n\nNew folder.",
        },
    )
    text = out["content"][0]["text"]
    assert "wrote dst/README.md" in text
    assert (root / "dst" / "README.md").read_text(encoding="utf-8").startswith("# Destination")


def test_move_entries_leaves_existing_readme_alone(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    (root / "src" / "thing.md").write_text(_entry_text("thing"))
    (root / "dst").mkdir(parents=True)
    existing = root / "dst" / "README.md"
    existing.write_text("# Pre-existing")
    out = library_tools._move_entries_impl(
        root,
        {"to_folder": "dst", "files": ["src/thing.md"], "readme": "# new content"},
    )
    text = out["content"][0]["text"]
    assert "left existing" in text
    assert existing.read_text(encoding="utf-8") == "# Pre-existing"


def test_move_entries_idempotent_when_source_at_destination(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "stay").mkdir(parents=True)
    p = root / "stay" / "here.md"
    p.write_text(_entry_text("here"))
    out = library_tools._move_entries_impl(
        root, {"to_folder": "stay", "files": ["stay/here.md"], "readme": ""}
    )
    text = out["content"][0]["text"]
    assert "already at destination" in text
    assert p.exists()


# ── _move_entries_impl: error paths ─────────────────────────────────────


def test_move_entries_rejects_missing_to_folder(tmp_path: Path) -> None:
    out = library_tools._move_entries_impl(tmp_path, {"files": ["x.md"]})
    assert "missing required arg 'to_folder'" in out["content"][0]["text"]


def test_move_entries_rejects_empty_files(tmp_path: Path) -> None:
    out = library_tools._move_entries_impl(tmp_path, {"to_folder": "dst", "files": []})
    assert "non-empty list" in out["content"][0]["text"]


def test_move_entries_rejects_non_list_files(tmp_path: Path) -> None:
    out = library_tools._move_entries_impl(tmp_path, {"to_folder": "dst", "files": "x.md"})
    assert "non-empty list" in out["content"][0]["text"]


def test_move_entries_rejects_destination_escape(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    out = library_tools._move_entries_impl(
        root, {"to_folder": "../escape", "files": ["x.md"], "readme": ""}
    )
    assert "outside the knowledge root" in out["content"][0]["text"]


def test_move_entries_rejects_source_escape(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    out = library_tools._move_entries_impl(
        root, {"to_folder": "dst", "files": ["../outside.md"], "readme": ""}
    )
    assert "outside the knowledge root" in out["content"][0]["text"]


def test_move_entries_rejects_blank_filename(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    out = library_tools._move_entries_impl(root, {"to_folder": "dst", "files": [""], "readme": ""})
    assert "non-empty string" in out["content"][0]["text"]


def test_move_entries_rejects_non_string_filename(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    out = library_tools._move_entries_impl(root, {"to_folder": "dst", "files": [42], "readme": ""})
    assert "non-empty string" in out["content"][0]["text"]


def test_move_entries_rejects_missing_source(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    out = library_tools._move_entries_impl(
        root, {"to_folder": "dst", "files": ["src/nope.md"], "readme": ""}
    )
    assert "does not exist" in out["content"][0]["text"]


def test_move_entries_rejects_non_file_source(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    # Source path points at a directory, not a regular file.
    (root / "src" / "is-dir").mkdir()
    out = library_tools._move_entries_impl(
        root, {"to_folder": "dst", "files": ["src/is-dir"], "readme": ""}
    )
    assert "not a regular file" in out["content"][0]["text"]


def test_move_entries_rejects_non_markdown_source(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    (root / "src" / "thing.txt").write_text("not md")
    out = library_tools._move_entries_impl(
        root, {"to_folder": "dst", "files": ["src/thing.txt"], "readme": ""}
    )
    assert "not a .md file" in out["content"][0]["text"]


def test_move_entries_rejects_collision_with_existing_dest(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    (root / "dst").mkdir(parents=True)
    (root / "src" / "x.md").write_text(_entry_text("x"))
    (root / "dst" / "x.md").write_text(_entry_text("dst-x"))
    out = library_tools._move_entries_impl(
        root, {"to_folder": "dst", "files": ["src/x.md"], "readme": ""}
    )
    assert "already exists" in out["content"][0]["text"]


def test_move_entries_summary_lists_inbound_link_count(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    (root / "src" / "x.md").write_text(_entry_text("x"))
    (root / "a.md").write_text("[[src/x]] is moving")
    (root / "b.md").write_text("twice in one file: [[src/x]] and [[src/x|aliased]]")
    out = library_tools._move_entries_impl(
        root, {"to_folder": "dst", "files": ["src/x.md"], "readme": ""}
    )
    text = out["content"][0]["text"]
    # 3 rewrites total: a.md (1) + b.md (2)
    assert "rewrote 3 inbound wikilink(s)" in text


# ── bm25 / embedding runners (in-process, no SDK) ───────────────────────
#
# These are the public entry points the curator agents call directly
# (via ``_agent_common``). The old Claude-Agent-SDK MCP-server wrapper is
# gone, so we exercise the branch coverage straight through the runners.


def _bm25_text(args: dict, root: Path) -> str:
    return library_tools.unwrap_text_response(library_tools.run_bm25_search(args, root))


def _embedding_text(args: dict, root: Path) -> str:
    return library_tools.unwrap_text_response(library_tools.run_embedding_search(args, root))


def test_run_bm25_search_returns_hits(seed_library) -> None:
    root = seed_library().resolve()
    text = _bm25_text({"query": "python uv install", "k": 3}, root)
    assert "score=" in text
    assert "use-uv-not-pip" in text


def test_run_bm25_search_empty_query(tmp_path: Path) -> None:
    assert "empty query" in _bm25_text({"query": "", "k": 3}, tmp_path)


def test_run_bm25_search_clamps_k_to_max_20(seed_library) -> None:
    """``_parse_search_args`` clamps k to ``min(20, k)``. Asking for 100
    still works and never returns more than 20 result lines."""
    root = seed_library().resolve()
    text = _bm25_text({"query": "python", "k": 100}, root)
    assert "score=" in text
    lines = [ln for ln in text.splitlines() if "score=" in ln]
    assert len(lines) <= 20


def test_run_bm25_search_missing_dir(tmp_path: Path) -> None:
    """Library dir does not exist -> the "library is empty" branch fires."""
    text = _bm25_text({"query": "anything", "k": 3}, tmp_path / "no-such-dir")
    assert "library is empty" in text


def test_run_bm25_search_no_results(seed_library) -> None:
    root = seed_library().resolve()
    text = _bm25_text({"query": "zzzzqqqqxxxx", "k": 3}, root)
    assert "no results" in text or "score=" not in text


def test_run_bm25_search_handles_load_failure(tmp_path: Path, monkeypatch) -> None:
    """If bm25.load_or_build raises a generic exception, the runner logs
    and returns an error-shaped content block."""
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "x.md").write_text("# x")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(library_tools, "load_or_build", _boom)
    assert "bm25 backend error" in _bm25_text({"query": "python", "k": 3}, k)


def test_run_bm25_search_handles_filenotfound(tmp_path: Path, monkeypatch) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()

    def _missing(*args, **kwargs):
        raise FileNotFoundError("no index")

    monkeypatch.setattr(library_tools, "load_or_build", _missing)
    assert "no entries yet" in _bm25_text({"query": "python", "k": 3}, k)


def test_run_bm25_search_clamps_bad_k(tmp_path: Path) -> None:
    """A non-int k falls back to the default rather than raising."""
    assert "empty query" in _bm25_text({"query": "", "k": "lots"}, tmp_path)


def test_run_embedding_search_empty_query(tmp_path: Path) -> None:
    assert "empty query" in _embedding_text({"query": "", "k": 3}, tmp_path)


def test_run_embedding_search_missing_dir(tmp_path: Path) -> None:
    text = _embedding_text({"query": "anything", "k": 3}, tmp_path / "no-such-dir")
    assert "library is empty" in text


def test_run_embedding_search_handles_filenotfound(tmp_path: Path, monkeypatch) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()

    def _missing(*args, **kwargs):
        raise FileNotFoundError("no index")

    monkeypatch.setattr(library_tools, "embeddings_load_or_build", _missing)
    assert "no entries yet" in _embedding_text({"query": "python", "k": 3}, k)


def test_run_embedding_search_handles_backend_error(tmp_path: Path, monkeypatch) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "x.md").write_text("# x")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated embedding failure")

    monkeypatch.setattr(library_tools, "embeddings_load_or_build", _boom)
    assert "embedding backend error" in _embedding_text({"query": "python", "k": 3}, k)


def test_unwrap_text_response_handles_unexpected_shape() -> None:
    assert library_tools.unwrap_text_response({"content": []}) == ""
    assert library_tools.unwrap_text_response({}) == ""


def test_move_entries_public_entry_point(tmp_path: Path) -> None:
    """``move_entries`` is called directly by the Scholar executor now."""
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    (root / "src" / "x.md").write_text(_entry_text("x"))
    out = library_tools.move_entries(
        root, {"to_folder": "dst", "files": ["src/x.md"], "readme": ""}
    )
    text = out["content"][0]["text"]
    assert text.startswith("move_entries: ok")
    assert (root / "dst" / "x.md").exists()


def test_move_entries_unreadable_source_returns_error(tmp_path: Path, monkeypatch) -> None:
    """If reading the source raises OSError mid-move, the helper returns
    a structured error rather than crashing."""
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    src = root / "src" / "x.md"
    src.write_text(_entry_text("x"))

    real_read = Path.read_text

    def _boom(self, *args, **kwargs):
        if self == src:
            raise OSError("simulated read failure")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _boom)
    out = library_tools._move_entries_impl(
        root, {"to_folder": "dst", "files": ["src/x.md"], "readme": ""}
    )
    assert "could not read" in out["content"][0]["text"]


def test_rewrite_skips_unreadable_md_files(tmp_path: Path, monkeypatch, caplog) -> None:
    """If an .md file can't be read during the wikilink rewrite scan, the
    move continues — the file is skipped silently."""
    root = tmp_path / "knowledge"
    (root / "src").mkdir(parents=True)
    src = root / "src" / "x.md"
    src.write_text(_entry_text("x"))
    # Another .md file that will be unreadable during the rewrite scan.
    bad = root / "unreadable.md"
    bad.write_text("[[src/x]]")

    real_read = Path.read_text

    def _selective_boom(self, *args, **kwargs):
        if self == bad:
            raise OSError("simulated unreadable file")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _selective_boom)
    out = library_tools._move_entries_impl(
        root, {"to_folder": "dst", "files": ["src/x.md"], "readme": ""}
    )
    # Move still succeeds — the unreadable file is skipped, not raised.
    text = out["content"][0]["text"]
    assert "move_entries: ok" in text
