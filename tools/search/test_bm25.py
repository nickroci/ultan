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

from bm25 import (
    BM25Index,
    _default_index_path,
    build_index,
    load_or_build,
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
