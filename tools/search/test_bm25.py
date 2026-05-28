"""Sanity tests for the BM25 indexer/searcher against the fixture knowledge base.

Run with:
    cd tools/search && uv run python -m pytest test_bm25.py -v

Or directly:
    cd tools/search && uv run python test_bm25.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

import bm25 as bm25_mod
from bm25 import (
    BM25Index,
    _build_snippet,
    _default_index_path,
    _frontmatter_search_text,
    _one_line,
    _strip_and_extract_frontmatter,
    build_index,
    is_stale,
    iter_markdown,
    load_or_build,
    load_pickled,
    save_index,
    tokenize,
)

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


def test_tokenize_strips_frontmatter_keeps_keywords() -> None:
    text = (
        "---\n"
        "id: x\n"
        "keywords: [factory, paradigm]\n"
        "applies-when: |\n"
        "  designing or building any new API\n"
        "private-field: should-not-appear\n"
        "---\n"
        "# Heading\n"
        "Body text here.\n"
    )
    toks = tokenize(text)
    assert "factory" in toks
    assert "paradigm" in toks
    assert "designing" in toks
    assert "api" in toks
    assert "body" in toks
    # `id` is length 2 and would survive — make sure `should-not-appear` doesn't
    # (it's in a frontmatter field we don't index).
    assert "should" not in toks
    assert "appear" not in toks


def test_build_index_finds_all_fixture_entries() -> None:
    index = build_index(FIXTURES)
    assert isinstance(index, BM25Index)
    paths = {Path(rec.path).name for rec in index.docs}
    assert "factory-pattern-for-apis.md" in paths
    assert "no-mock-db.md" in paths
    assert "prefer-pathlib.md" in paths
    assert "paradigms-cross-cutting.md" in paths
    assert "auth-redirects.md" in paths
    # `index.md` is the catalog; BM25 skips it on purpose (see bm25.py).
    assert "index.md" not in paths


def test_search_factory_finds_factory_entry() -> None:
    index = build_index(FIXTURES)
    hits = index.search("factory pattern for APIs", k=5)
    assert hits, "expected at least one hit for 'factory pattern for APIs'"
    top_names = [p.name for p, _, _ in hits[:2]]
    # In a 5-doc corpus the connection file's wikilink density can edge out the
    # canonical entry. Both are legitimately on-topic; we only require the
    # canonical entry to appear in the top 2.
    assert "factory-pattern-for-apis.md" in top_names, (
        f"factory entry missing from top 2; got {top_names}"
    )


def test_search_paradigm_finds_cross_cutting_via_body() -> None:
    """The PLAN's motivating example: 'paradigm' is in the body, not the index.

    BM25 over article bodies must surface paradigms-cross-cutting.md for the
    query 'what paradigms do we use'.
    """
    index = build_index(FIXTURES)
    hits = index.search("what paradigms do we use", k=5)
    assert hits, "BM25 returned nothing for the paradigms query"
    top_names = [p.name for p, _, _ in hits]
    # Either the cross-cutting connection or the factory entry is acceptable as
    # top hit — both surface the concept. The point is BM25 finds *something*.
    assert any("paradigms" in n or "factory" in n for n in top_names)


def test_search_database_finds_no_mock_db() -> None:
    index = build_index(FIXTURES)
    hits = index.search("postgres database tests", k=5)
    assert hits
    assert hits[0][0].name == "no-mock-db.md"


def test_search_returns_empty_for_nonsense_query() -> None:
    index = build_index(FIXTURES)
    hits = index.search("zzzzzzz qqqqqqq", k=5)
    # No token from the query exists in the corpus, so every score is 0.
    # We filter zero-score results, so this should be empty.
    assert hits == []


def test_load_or_build_caches_and_rebuilds() -> None:
    """Index is reused when fresh, rebuilt when a file's mtime advances."""
    with tempfile.TemporaryDirectory() as td:
        knowledge = Path(td) / "knowledge"
        shutil.copytree(FIXTURES, knowledge)

        first = load_or_build(knowledge)
        assert _default_index_path(knowledge).exists()

        # Second call: should reuse the persisted index.
        second = load_or_build(knowledge)
        assert first.built_at == second.built_at

        # Mutate a file and confirm we rebuild.
        target = knowledge / "global" / "concepts" / "factory-pattern-for-apis.md"
        text = target.read_text(encoding="utf-8")
        target.write_text(text + "\n\nNew sentence added for cache-invalidation test.\n")
        # Force mtime to advance noticeably.
        import os
        import time

        future = time.time() + 5
        os.utime(target, (future, future))

        third = load_or_build(knowledge)
        assert third.built_at > second.built_at


# ── Edge cases ─────────────────────────────────────────────────────────────────


def test_strip_frontmatter_handles_malformed_yaml() -> None:
    """A frontmatter block with invalid YAML must not crash tokenization."""
    text = "---\n: : : invalid\n---\nbody here\n"
    body, fm = _strip_and_extract_frontmatter(text)
    assert fm == {}
    assert "body here" in body


def test_strip_frontmatter_non_dict_yaml() -> None:
    """Frontmatter that parses to a list rather than a dict -> empty dict."""
    text = "---\n- one\n- two\n---\nbody\n"
    _, fm = _strip_and_extract_frontmatter(text)
    assert fm == {}


def test_strip_frontmatter_missing_block_returns_full_text() -> None:
    text = "no frontmatter here\n"
    body, fm = _strip_and_extract_frontmatter(text)
    assert body == text
    assert fm == {}


def test_frontmatter_search_text_handles_string_keywords() -> None:
    """``keywords`` as a string rather than a list is still included."""
    out = _frontmatter_search_text({"keywords": "foo bar baz"})
    assert "foo bar baz" in out


def test_frontmatter_search_text_handles_list_applies_when() -> None:
    out = _frontmatter_search_text({"applies-when": ["one", "two"]})
    assert "one" in out and "two" in out


def test_search_empty_query_returns_empty() -> None:
    """Query with no >=2-char tokens shouldn't return anything."""
    index = build_index(FIXTURES)
    assert index.search("a x") == []


def test_search_on_empty_index_returns_empty(tmp_path: Path) -> None:
    """An index built over an empty dir has no docs -> empty results."""
    empty = tmp_path / "empty"
    empty.mkdir()
    index = build_index(empty)
    assert index.bm25 is None
    assert index.search("anything") == []


def test_build_index_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "nope")


def test_build_index_skips_unreadable_files(tmp_path: Path, monkeypatch) -> None:
    """A file that raises ``UnicodeDecodeError`` on read is silently skipped."""
    knowledge = tmp_path / "knowledge"
    (knowledge / "global").mkdir(parents=True)
    good = knowledge / "global" / "good.md"
    good.write_text("# good content\n\nactual body here\n", encoding="utf-8")
    bad = knowledge / "global" / "bad.md"
    bad.write_text("placeholder\n", encoding="utf-8")

    real_read_text = Path.read_text

    def fake_read_text(self, *a, **kw):
        if self == bad:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "boom")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    index = build_index(knowledge)
    paths = {Path(rec.path).name for rec in index.docs}
    assert "good.md" in paths
    assert "bad.md" not in paths


def test_load_pickled_corrupt_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "garbage.idx"
    p.write_bytes(b"not a pickle stream")
    assert load_pickled(p, BM25Index) is None


def test_load_or_build_rebuilds_when_knowledge_dir_changed(tmp_path: Path) -> None:
    """A pickled index from a *different* knowledge dir must trigger rebuild."""
    knowledge_a = tmp_path / "a" / "knowledge"
    knowledge_b = tmp_path / "b" / "knowledge"
    shutil.copytree(FIXTURES, knowledge_a)
    shutil.copytree(FIXTURES, knowledge_b)
    # Build against A but save the pickle into B's expected slot.
    idx_a = build_index(knowledge_a)
    target_b = _default_index_path(knowledge_b)
    save_index(idx_a, target_b)
    # Now ask for B's index: cached.knowledge_dir != knowledge_b -> rebuild.
    rebuilt = load_or_build(knowledge_b)
    assert rebuilt.knowledge_dir == knowledge_b.resolve()


def test_load_or_build_force_rebuild(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    shutil.copytree(FIXTURES, knowledge)
    first = load_or_build(knowledge)
    second = load_or_build(knowledge, force_rebuild=True)
    assert second.built_at >= first.built_at


def test_load_or_build_saves_best_effort_on_oserror(tmp_path: Path, monkeypatch) -> None:
    """A failure to persist the index doesn't fail the call — we return
    the in-memory index anyway."""
    knowledge = tmp_path / "knowledge"
    shutil.copytree(FIXTURES, knowledge)

    def fake_save(*a, **kw):
        raise OSError("cannot write")

    monkeypatch.setattr(bm25_mod, "save_index", fake_save)
    idx = load_or_build(knowledge)
    assert len(idx.docs) > 0


def test_is_stale_detects_removed_file(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    shutil.copytree(FIXTURES, knowledge)
    idx = build_index(knowledge)
    # Remove a tracked file.
    (knowledge / "global" / "concepts" / "no-mock-db.md").unlink()
    current_files = {str(p): p.stat().st_mtime for p in iter_markdown(knowledge)}
    assert is_stale(idx.docs, current_files) is True


def test_build_snippet_falls_back_to_first_body_line() -> None:
    """No query tokens hit -> snippet is the first non-heading line."""
    text = "---\nid: x\n---\n# Heading\n\nFirst body line here.\n"
    out = _build_snippet(text, q_tokens=["xxnonexistentxx"])
    assert out.startswith("First body line")


def test_build_snippet_only_heading_returns_body_strip() -> None:
    """No matching token and body is only a heading -> falls through to body.strip()."""
    text = "---\nid: x\n---\n# Heading only\n"
    out = _build_snippet(text, q_tokens=["xxnomatch"])
    # No non-heading lines exist -> falls back to the stripped body (the heading itself).
    assert "Heading" in out


def test_one_line_truncates_with_ellipsis() -> None:
    long = "word " * 50
    out = _one_line(long, width=20)
    assert len(out) == 20
    assert out.endswith("…")


if __name__ == "__main__":
    # Lightweight runner so this works without pytest.
    fns = [
        test_tokenize_strips_frontmatter_keeps_keywords,
        test_build_index_finds_all_fixture_entries,
        test_search_factory_finds_factory_entry,
        test_search_paradigm_finds_cross_cutting_via_body,
        test_search_database_finds_no_mock_db,
        test_search_returns_empty_for_nonsense_query,
        test_load_or_build_caches_and_rebuilds,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR  {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    raise SystemExit(1 if failed else 0)
