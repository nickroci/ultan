"""Sentence-transformer embedding index for the agent-mem knowledge store.

Mirrors the shape of ``bm25.BM25Index`` so the integration layer can treat the
two retrieval modes symmetrically. This module is infrastructure only — it is
not wired into the existing CLI search subcommands. BM25 continues to be the
sole engine behind those; an integrator will fuse the two later.

File selection rules (same as BM25):
  - skip ``_archive/`` subtrees
  - skip the top-level ``index.md`` and ``log.md`` catalogs
  - YAML frontmatter is stripped from the embedded text, except ``keywords:``
    and ``applies-when:`` which are preserved (PLAN section 2)

Model:
  - Default ``sentence-transformers/all-MiniLM-L6-v2``: 80 MB, 384-dim, CPU-fast.
  - Loaded lazily on first build/search call; cached at module level keyed on
    model name so multiple ``EmbeddingIndex`` instances share one model.

Persistence:
  - Pickled ``EmbeddingIndex`` at ``<knowledge_dir>/../.embeddings.idx`` by
    default. Same caveat as ``bm25.py``: pickle is not portable across Python
    versions; delete the file after an interpreter upgrade. ``load_or_build``
    swallows unpickle errors and rebuilds.

Threading:
  - Reads (``search``) are safe to call concurrently.
  - Builds and saves are single-threaded; do not call ``build_index`` /
    ``save_index`` from multiple threads against the same path.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# Reuse BM25's file-selection + frontmatter discipline. They're underscore-prefixed
# but this is a single internal package; pragmatic over copy-paste.
from bm25 import (
    _build_snippet,
    _frontmatter_search_text,
    _iter_markdown,
    _strip_and_extract_frontmatter,
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB, fast on CPU

# Module-level model cache so multiple indices share one loaded model.
_MODEL_CACHE: dict[str, Any] = {}


# ── Model loading ──────────────────────────────────────────────────────────────


def _load_model(model_name: str) -> Any:
    """Lazy-load and cache a SentenceTransformer.

    Tries offline-first (zero network) so cached models load with no
    HuggingFace HEAD requests. Every subprocess that uses embeddings
    (advisor, daemon worker restart, manual scripts) would otherwise
    ping HF on every load — adds latency, leaks usage info, breaks
    offline.

    Falls back to online (downloads) only when the model isn't cached
    locally yet. First-ever load needs network; everything after is
    offline.
    """
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    # Try offline first. `local_files_only=True` tells HuggingFace
    # transformers to refuse network and load from disk cache only.
    # If the model isn't cached, this raises — we catch and retry online.
    try:
        model = SentenceTransformer(
            model_name,
            device="cpu",
            local_files_only=True,
        )
    except Exception:
        # Not cached yet — fall back to download. Subsequent loads in
        # other processes will hit the cache and go offline.
        model = SentenceTransformer(model_name, device="cpu")

    _MODEL_CACHE[model_name] = model
    return model


# ── Records / hit dataclass ────────────────────────────────────────────────────


@dataclass
class _DocRecord:
    """Minimal per-doc record; parallel to a row of ``EmbeddingIndex.embeddings``."""

    path: str  # absolute path string
    mtime: float
    raw_text: str  # full file text, kept for snippet generation


@dataclass
class EmbeddingHit:
    """A single search result."""

    path: Path
    score: float  # cosine similarity in [0, 1] (clamped on negatives)
    snippet: str  # one-line snippet, same style as bm25


# ── Index dataclass ────────────────────────────────────────────────────────────


@dataclass
class EmbeddingIndex:
    """Sentence-transformer embedding index over markdown bodies."""

    knowledge_dir: Path
    model_name: str = DEFAULT_MODEL
    docs: list[_DocRecord] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    built_at: float = 0.0

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> list[EmbeddingHit]:
        """Embed the query, cosine against all docs, return top-k descending.

        Filters out non-positive cosine scores so results are "actually
        semantically related" rather than padded.
        """
        if not self.docs or self.embeddings.size == 0:
            return []
        if not query or not query.strip():
            return []

        model = _load_model(self.model_name)
        q_vec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        # Embeddings are stored already L2-normalized, so cosine == dot product.
        scores = self.embeddings @ q_vec  # shape (n_docs,)
        # Clamp tiny negatives (numerical noise) to 0 so callers can assume [0,1].
        scores = np.clip(scores, 0.0, 1.0)

        # Argpartition + sort the top-k window — faster than full sort for big n,
        # equivalent for small n.
        n = scores.shape[0]
        top_k = min(k, n)
        if top_k <= 0:
            return []
        # Use simple sort: corpora are small (dozens-to-hundreds of entries).
        order = np.argsort(-scores)
        # Build snippets using the same helper BM25 uses, with the query lowercased
        # and tokenized loosely (semantic matches won't necessarily contain query
        # tokens — the snippet just falls back to the first body line in that case).
        q_tokens = [t for t in query.lower().split() if t]
        hits: list[EmbeddingHit] = []
        for idx in order[:top_k]:
            score = float(scores[idx])
            if score <= 0:
                continue
            rec = self.docs[int(idx)]
            snippet = _build_snippet(rec.raw_text, q_tokens)
            hits.append(EmbeddingHit(path=Path(rec.path), score=score, snippet=snippet))
        return hits


# ── Text extraction (matches bm25's body-vs-frontmatter discipline) ────────────


def _embedding_text(raw: str) -> str:
    """The text we feed to the encoder for a given .md file.

    Same discipline as ``bm25.tokenize`` — frontmatter stripped except for
    ``keywords:`` and ``applies-when:``, which are prepended to the body.
    Unlike BM25 we keep the original casing: sentence-transformers handle case.
    """
    body, fm = _strip_and_extract_frontmatter(raw)
    fm_text = _frontmatter_search_text(fm)
    if fm_text:
        return fm_text + "\n" + body
    return body


# ── Index build / persist / load ───────────────────────────────────────────────


def _default_index_path(knowledge_dir: Path) -> Path:
    """``~/.agent-mem/.embeddings.idx`` when knowledge_dir is ``~/.agent-mem/knowledge``."""
    return knowledge_dir.parent / ".embeddings.idx"


def build_index(
    knowledge_dir: Path,
    model_name: str = DEFAULT_MODEL,
) -> EmbeddingIndex:
    """Walk ``knowledge_dir``, embed every entry's body, return a fresh index."""
    knowledge_dir = knowledge_dir.expanduser().resolve()
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"knowledge_dir does not exist: {knowledge_dir}")

    docs: list[_DocRecord] = []
    texts: list[str] = []
    for md in _iter_markdown(knowledge_dir):
        try:
            raw = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        text = _embedding_text(raw).strip()
        if not text:
            continue
        docs.append(_DocRecord(path=str(md), mtime=md.stat().st_mtime, raw_text=raw))
        texts.append(text)

    if not docs:
        # Empty corpus — return a structurally valid empty index.
        return EmbeddingIndex(
            knowledge_dir=knowledge_dir,
            model_name=model_name,
            docs=[],
            embeddings=np.zeros((0, 0), dtype=np.float32),
            built_at=time.time(),
        )

    model = _load_model(model_name)
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    return EmbeddingIndex(
        knowledge_dir=knowledge_dir,
        model_name=model_name,
        docs=docs,
        embeddings=vecs,
        built_at=time.time(),
    )


def save_index(index: EmbeddingIndex, path: Path | None = None) -> Path:
    """Pickle the index. Returns the path written.

    Atomic-ish: write to a temp sibling then rename.
    """
    target = path or _default_index_path(index.knowledge_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)
    return target


def _load_pickled(index_path: Path) -> EmbeddingIndex | None:
    try:
        with index_path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, EmbeddingIndex):
            return obj
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError):
        return None
    return None


def _is_stale(index: EmbeddingIndex, knowledge_dir: Path) -> bool:
    """True if any tracked file moved/disappeared, or any .md is newer than the index."""
    current_files = {str(p): p.stat().st_mtime for p in _iter_markdown(knowledge_dir)}
    tracked = {rec.path: rec.mtime for rec in index.docs}

    if set(current_files) != set(tracked):
        return True
    for path, mtime in current_files.items():
        if mtime > tracked[path] + 1e-6:
            return True
    return False


def load_or_build(
    knowledge_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    force_rebuild: bool = False,
    index_path: Path | None = None,
) -> EmbeddingIndex:
    """Load the persisted index if fresh, otherwise rebuild and save.

    "Fresh" means the index was built against the same ``knowledge_dir`` and
    ``model_name``, every file we tracked is still present with the same mtime,
    and no new ``.md`` files have appeared.
    """
    knowledge_dir = knowledge_dir.expanduser().resolve()
    target = index_path or _default_index_path(knowledge_dir)

    if not force_rebuild and target.exists():
        cached = _load_pickled(target)
        if (
            cached is not None
            and cached.knowledge_dir == knowledge_dir
            and cached.model_name == model_name
            and not _is_stale(cached, knowledge_dir)
        ):
            return cached

    index = build_index(knowledge_dir, model_name=model_name)
    try:
        save_index(index, target)
    except OSError:
        # Persistence is best-effort; in-memory index is still usable.
        pass
    return index
