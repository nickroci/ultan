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

import numpy as np
import pytest

import embeddings
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

    Threshold note: nomic-embed-text-v1.5 produces tighter cosine ranges than
    older models (MiniLM sat ~0.2 for unrelated; nomic sits ~0.4-0.5).
    Genuine paraphrase matches land at 0.65+, so 0.55 separates the two
    populations with margin. Re-tune if the default embedder changes again.
    """
    hits = built_index.search("submarine periscope navigation", k=10)
    # Either no hits, or every hit is weak.
    if hits:
        top = max(h.score for h in hits)
        assert top < 0.55, (
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


def test_load_or_build_incremental_reencodes_only_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale-but-compatible index re-encodes ONLY the changed entries, not the
    whole corpus — the fix for the priming-hot-path full rebuild after every
    Scholar write. On the old code this re-encoded everything (batches [3, 3]).
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    for name in ("a", "b", "c"):
        (knowledge / f"{name}.md").write_text(f"# {name}\nbody about {name}\n", encoding="utf-8")

    batches: list[int] = []

    class _FakeModel:
        def encode(self, texts: list[str], **_kwargs: object) -> object:
            batches.append(len(texts))
            arr = np.zeros((len(texts), 4), dtype=np.float32)
            for i, text in enumerate(texts):
                arr[i, len(text) % 4] = 1.0
            return arr

    fake = _FakeModel()
    monkeypatch.setattr(embeddings, "_load_model", lambda *_a, **_k: fake)

    first = load_or_build(knowledge)
    assert batches == [3]  # full build encoded all 3 entries
    assert len(first.docs) == 3

    # Modify ONE entry and push its mtime into the future so the index is stale.
    (knowledge / "b.md").write_text("# b\ntotally different body now\n", encoding="utf-8")
    future = time.time() + 5
    os.utime(knowledge / "b.md", (future, future))

    second = load_or_build(knowledge)
    assert batches == [3, 1]  # incremental encoded ONLY the changed entry
    assert len(second.docs) == 3

    # Unchanged entries kept their exact cached embedding rows (reused, not re-encoded).
    def _row(idx: EmbeddingIndex, name: str) -> object:
        pos = next(i for i, d in enumerate(idx.docs) if d.path.endswith(name))
        return idx.embeddings[pos]

    assert np.array_equal(_row(first, "a.md"), _row(second, "a.md"))
    assert np.array_equal(_row(first, "c.md"), _row(second, "c.md"))


# ── Edge cases that don't need the model loaded ──────────────────────────────


def test_search_on_empty_index_returns_empty(tmp_path: Path) -> None:
    """Built against an empty dir -> structurally empty index, search returns []."""
    import numpy as np

    from embeddings import EmbeddingIndex

    idx = EmbeddingIndex(
        knowledge_dir=tmp_path,
        docs=[],
        embeddings=np.zeros((0, 0), dtype=np.float32),
    )
    assert idx.search("anything") == []


def test_search_empty_query_returns_empty(built_index: EmbeddingIndex) -> None:
    """Blank or whitespace-only query short-circuits before model load."""
    assert built_index.search("") == []
    assert built_index.search("   ") == []


def test_search_top_k_zero_returns_empty(built_index: EmbeddingIndex) -> None:
    """``k=0`` short-circuits the result loop."""
    assert built_index.search("python dependencies", k=0) == []


def test_build_index_raises_for_missing_dir(tmp_path: Path) -> None:
    import pytest

    from embeddings import build_index

    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "nope")


def test_build_index_empty_corpus_returns_empty(tmp_path: Path) -> None:
    """Knowledge dir exists but has no markdown -> structurally empty index."""
    from embeddings import build_index

    empty = tmp_path / "knowledge"
    empty.mkdir()
    idx = build_index(empty)
    assert len(idx.docs) == 0
    assert idx.embeddings.shape == (0, 0)


def test_build_index_skips_unreadable_and_empty(tmp_path: Path, monkeypatch) -> None:
    """A file that raises UnicodeDecodeError on read is dropped, and a file
    whose embedded text is whitespace-only is also dropped."""
    from embeddings import build_index

    knowledge = tmp_path / "knowledge"
    (knowledge / "global").mkdir(parents=True)
    bad = knowledge / "global" / "bad.md"
    bad.write_text("# bad\n", encoding="utf-8")
    blank = knowledge / "global" / "blank.md"
    blank.write_text("---\nid: blank\n---\n   \n", encoding="utf-8")
    good = knowledge / "global" / "good.md"
    good.write_text("# good\n\nReal body here.\n", encoding="utf-8")

    real_read_text = Path.read_text

    def fake_read_text(self, *a, **kw):
        if self == bad:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "boom")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    idx = build_index(knowledge)
    paths = {Path(rec.path).name for rec in idx.docs}
    assert "good.md" in paths
    assert "bad.md" not in paths
    assert "blank.md" not in paths


def test_load_pickled_corrupt_file_returns_none(tmp_path: Path) -> None:
    from bm25 import load_pickled

    p = tmp_path / "garbage.idx"
    p.write_bytes(b"not a pickle stream")
    assert load_pickled(p, EmbeddingIndex) is None


def test_is_stale_detects_new_file(built_index: EmbeddingIndex, tmp_path: Path) -> None:
    """A file appearing after the index was built is detected as stale."""
    import shutil as shu

    from bm25 import is_stale, iter_markdown
    from embeddings import build_index

    # Build a fresh index against a known corpus.
    knowledge = tmp_path / "kb"
    shu.copytree(FIXTURES, knowledge)
    idx = build_index(knowledge)
    # Drop in a new file and confirm staleness.
    new = knowledge / "global" / "concepts" / "extra.md"
    new.write_text("# extra\n\nNew content.\n", encoding="utf-8")
    current_files = {str(p): p.stat().st_mtime for p in iter_markdown(knowledge)}
    assert is_stale(idx.docs, current_files) is True


def test_load_or_build_save_failure_returns_in_memory_index(tmp_path: Path, monkeypatch) -> None:
    """If persisting the index fails, the in-memory index is still returned."""
    import embeddings as emb

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "x.md").write_text(
        "# X\n\nA reasonably long body that will be embedded.\n", encoding="utf-8"
    )

    def fake_save(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(emb, "save_index", fake_save)
    idx = emb.load_or_build(knowledge)
    assert len(idx.docs) == 1


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
