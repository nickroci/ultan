"""Sanity tests for the sentence-transformer embedding indexer.

First run downloads ``sentence-transformers/all-MiniLM-L6-v2`` (~80 MB) from
HuggingFace; subsequent runs are fast. The tests share a session-scoped
fixture so the model is loaded exactly once across the whole module.

Run with:
    cd tools/search && uv run pytest test_embeddings.py -v
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from embeddings import (
    EmbeddingHit,
    EmbeddingIndex,
    _default_index_path,
    build_index,
    load_or_build,
    save_index,
)

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


# ── Fixture: a temp knowledge dir we own (copy of fixtures + a uv entry) ───────


UV_ENTRY = """\
---
id: always-use-uv
type: lesson
scope: global
status: confirmed
confidence: 0.9
applies-when: |
  installing or running Python tools and managing project dependencies
keywords: [uv, python, dependencies, package-manager, virtualenv, pip]
created: 2026-05-19
updated: 2026-05-19
fired: 0
fired-helpful: 0
---

# Always use uv for Python

**Rule:** All Python tooling, virtualenvs, dependency resolution, and script
execution goes through `uv`. Never `pip install` directly; never `python -m venv`
by hand; never `pip-tools` or `poetry`.

**Why:** `uv` is dramatically faster, has a single coherent lockfile, and gives
us deterministic environments across machines without per-project ceremony.

**How to apply:** `uv sync` to install a project's dependencies. `uv run <cmd>`
to execute anything inside the project env. `uv add <pkg>` to add a dependency.
`uv venv` only when you genuinely need a bare virtualenv outside a project.
"""


@pytest.fixture(scope="module")
def knowledge_with_uv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy the BM25 fixtures, add a uv entry, return the knowledge dir.

    Module-scoped so we pay the embed cost once across all tests that don't
    need to mutate the corpus.
    """
    root = tmp_path_factory.mktemp("kb-with-uv") / "knowledge"
    shutil.copytree(FIXTURES, root)
    (root / "global" / "concepts" / "always-use-uv.md").write_text(UV_ENTRY)
    return root


@pytest.fixture(scope="module")
def built_index(knowledge_with_uv: Path) -> EmbeddingIndex:
    """One built index reused across the read-only tests."""
    return build_index(knowledge_with_uv)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_build_index_finds_all_fixture_entries(built_index: EmbeddingIndex) -> None:
    # Original 5 BM25 fixtures + the uv entry we injected.
    assert isinstance(built_index, EmbeddingIndex)
    paths = {Path(rec.path).name for rec in built_index.docs}
    expected = {
        "factory-pattern-for-apis.md",
        "no-mock-db.md",
        "prefer-pathlib.md",
        "paradigms-cross-cutting.md",
        "auth-redirects.md",
        "always-use-uv.md",
    }
    assert expected.issubset(paths), f"missing entries; got {paths}"
    # Catalog is skipped, same as BM25.
    assert "index.md" not in paths
    # Embedding matrix shape matches docs count.
    assert built_index.embeddings.shape[0] == len(built_index.docs)
    assert built_index.embeddings.shape[1] > 0  # some embedding dim


def test_search_returns_semantically_related_entry(built_index: EmbeddingIndex) -> None:
    """The motivating case: query has zero token overlap with the uv entry's
    rule text, but semantically asks the same question. BM25 would miss this;
    embeddings should surface it."""
    hits = built_index.search("managing python dependencies", k=5)
    assert hits, "expected at least one hit"
    names = [h.path.name for h in hits]
    assert "always-use-uv.md" in names, f"uv entry missing from results; got {names}"
    uv_score = next(h.score for h in hits if h.path.name == "always-use-uv.md")
    assert uv_score > 0.3, f"uv entry should score > 0.3; got {uv_score:.3f}"


def test_search_returns_empty_for_unrelated_query(built_index: EmbeddingIndex) -> None:
    """A truly off-topic query should not surface meaningful matches.

    We tolerate either: empty results, or all results below a low threshold.
    Embedding models always produce *some* cosine signal — the question is
    whether it's above noise.
    """
    hits = built_index.search("submarine periscope navigation", k=10)
    # Either no hits, or every hit is weak.
    if hits:
        top = max(h.score for h in hits)
        assert top < 0.35, (
            f"unrelated query produced a strong hit (score={top:.3f}); "
            f"top results: {[(h.path.name, round(h.score, 3)) for h in hits[:3]]}"
        )


def test_search_score_in_zero_to_one_range(built_index: EmbeddingIndex) -> None:
    hits = built_index.search("python dependencies", k=20)
    assert hits
    for h in hits:
        assert isinstance(h, EmbeddingHit)
        assert 0.0 <= h.score <= 1.0, f"score out of range: {h.score} for {h.path.name}"
    # And they're in descending order.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_archive_subtree_excluded(tmp_path: Path) -> None:
    """Files under `_archive/` must not be indexed (same rule as BM25)."""
    knowledge = tmp_path / "knowledge"
    (knowledge / "global" / "concepts").mkdir(parents=True)
    (knowledge / "_archive").mkdir()
    (knowledge / "global" / "concepts" / "live.md").write_text(
        "# Live entry\n\nThis should be indexed.\n"
    )
    (knowledge / "_archive" / "old.md").write_text("# Archived\n\nThis should NOT be indexed.\n")

    index = build_index(knowledge)
    names = {Path(rec.path).name for rec in index.docs}
    assert "live.md" in names
    assert "old.md" not in names


def test_load_or_build_caches_and_rebuilds(tmp_path: Path) -> None:
    """Index is reused when fresh, rebuilt when a file's mtime advances."""
    knowledge = tmp_path / "knowledge"
    shutil.copytree(FIXTURES, knowledge)

    first = load_or_build(knowledge)
    assert _default_index_path(knowledge).exists()
    assert first.embeddings.shape[0] == len(first.docs) == 5

    # Second call: should reuse the persisted index (same built_at).
    second = load_or_build(knowledge)
    assert first.built_at == second.built_at
    assert second.embeddings.shape == first.embeddings.shape

    # Touch a file's mtime far enough into the future to defeat any rounding,
    # then confirm we rebuild.
    target = knowledge / "global" / "concepts" / "factory-pattern-for-apis.md"
    future = time.time() + 5
    os.utime(target, (future, future))

    third = load_or_build(knowledge)
    assert third.built_at > second.built_at
    assert third.embeddings.shape[0] == len(third.docs) == 5


def test_save_and_reload_roundtrip(built_index: EmbeddingIndex, tmp_path: Path) -> None:
    """Persisting and reloading produces an equivalent index."""
    target = tmp_path / ".embeddings.idx"
    written = save_index(built_index, target)
    assert written == target
    assert target.exists()

    reloaded = load_or_build(
        built_index.knowledge_dir,
        index_path=target,
    )
    assert reloaded.embeddings.shape == built_index.embeddings.shape
    assert len(reloaded.docs) == len(built_index.docs)
    # Same doc paths in the same order.
    assert [d.path for d in reloaded.docs] == [d.path for d in built_index.docs]
