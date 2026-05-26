"""Sanity tests for the cross-encoder reranker.

First run downloads ``cross-encoder/ms-marco-MiniLM-L-12-v2`` (~130 MB)
from HuggingFace; subsequent runs are fast. The model is module-cached
inside ``reranker._MODEL_CACHE`` so multiple tests in the same process
share one load.

Run with:
    cd tools/search && uv run pytest test_reranker.py -v
"""

from __future__ import annotations

from pathlib import Path

from reranker import rerank


def _make_candidates(tmp_path: Path) -> list[tuple[Path, str]]:
    """Three throwaway markdown bodies with clear topical separation.

    The first body is the obvious answer to the query in the tests; the
    other two are distractors that should rank below it.
    """
    a = tmp_path / "uv.md"
    a.write_text(
        "# Always use uv for Python\n\n"
        "All Python tooling, virtualenvs, dependency resolution, and script "
        "execution goes through uv. Never pip install directly.\n"
    )
    b = tmp_path / "submarines.md"
    b.write_text(
        "# Submarine periscope navigation\n\n"
        "Periscope-based attack runs rely on stadimeter range estimation "
        "and bearing rate.\n"
    )
    c = tmp_path / "frontmatter.md"
    c.write_text(
        "# Frontmatter conventions\n\n"
        "Every entry begins with a YAML frontmatter block delimited by "
        "triple-dash fences.\n"
    )
    return [(a, a.read_text()), (b, b.read_text()), (c, c.read_text())]


def test_rerank_orders_by_relevance(tmp_path: Path) -> None:
    """Most-relevant candidate should land first after reranking.

    The query is a paraphrase of the uv entry's intent — neither the
    submarines doc nor the frontmatter doc covers Python dependency
    management, so the uv doc must rank above both.
    """
    candidates = _make_candidates(tmp_path)
    result = rerank("how should I manage python dependencies?", candidates)
    assert result is not None, "reranker returned None on a well-formed call"
    assert len(result) == 3
    top_path, _ = result[0]
    assert top_path.name == "uv.md", (
        f"expected uv.md to rank first; got {[p.name for p, _ in result]}"
    )


def test_rerank_scores_are_descending(tmp_path: Path) -> None:
    candidates = _make_candidates(tmp_path)
    result = rerank("python package manager", candidates)
    assert result is not None
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True), f"scores not sorted desc: {scores}"


def test_rerank_empty_query_returns_none(tmp_path: Path) -> None:
    candidates = _make_candidates(tmp_path)
    assert rerank("", candidates) is None
    assert rerank("   ", candidates) is None


def test_rerank_empty_candidates_returns_none() -> None:
    assert rerank("python dependencies", []) is None
