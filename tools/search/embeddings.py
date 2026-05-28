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
  - ``save_index`` is concurrency-safe (see ``bm25.save_pickled``): racing
    saves to the same path each write a unique temp file then atomically
    replace, so they never interleave into a corrupt file. Last writer wins.
    ``build_index`` has no shared state, so parallel rebuilds are wasteful
    (duplicated encode work) but correct.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

# Reuse BM25's file-selection + frontmatter discipline. ``bm25.py`` re-exports
# these helpers under non-underscore names specifically so we can import them
# here without triggering pyright's reportPrivateUsage check.
from _device import select_device
from bm25 import build_snippet as _build_snippet
from bm25 import frontmatter_search_text as _frontmatter_search_text
from bm25 import is_stale, load_pickled, save_pickled
from bm25 import iter_markdown as _iter_markdown
from bm25 import strip_and_extract_frontmatter as _strip_and_extract_frontmatter

# A 2-D float32 array — the embedding matrix shape we store on disk. Numpy
# only began shipping precise generic annotations recently; npt.NDArray gives
# us "array of float32" without committing to dim-typing.
_FloatArray = npt.NDArray[np.float32]

# Default embedder. Picked for the combination of:
#   - asymmetric retrieval prompts (search_query: / search_document:) —
#     genuine precision lift on a query-vs-document pipeline like priming
#   - Matryoshka dims 64/128/256/512/768 (can truncate later without reindex)
#   - 8192-token input (older embedders truncate long entries silently)
#   - ~270 MB resident, well under the daemon's RAM ceiling
#
# TODO(future): google/embeddinggemma-300m is the quality ceiling — ~1.2 GB
# resident, task-aware prompts, 2048-token input. Swap when that footprint
# is acceptable. The encode contract here is the same; only the prefix
# helpers below would need updating to its `task: ... | query: ...` shape.
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Module-level model cache so multiple indices share one loaded model.
_MODEL_CACHE: dict[str, Any] = {}


# ── Model loading ──────────────────────────────────────────────────────────────


def _needs_trust_remote_code(model_name: str) -> bool:
    """Nomic's embedders ship custom modeling code on the Hub.

    Loading them through HuggingFace requires opting in to
    ``trust_remote_code=True`` (which executes Python from the model
    repo). We do this for nomic models specifically; other embedders
    (MiniLM, BGE, GTE) use stock transformers archs and don't need it.
    """
    return "nomic-embed-text" in model_name


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

    device = select_device()
    extra: dict[str, Any] = {}
    if _needs_trust_remote_code(model_name):
        extra["trust_remote_code"] = True

    # Try offline first. `local_files_only=True` tells HuggingFace
    # transformers to refuse network and load from disk cache only.
    # If the model isn't cached, this raises — we catch and retry online.
    try:
        model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
            **extra,
        )
    except Exception:
        # Not cached yet — fall back to download. Subsequent loads in
        # other processes will hit the cache and go offline.
        model = SentenceTransformer(model_name, device=device, **extra)

    # Cap sequence length to bound the attention-matrix memory cost.
    # nomic-embed-text-v1.5 ships with max_seq_length=2048 which, on MPS,
    # exceeds the per-tensor allocation ceiling at any non-trivial batch
    # size (RuntimeError: Invalid buffer size). Our longest library entries
    # are ~3000 tokens — capping at 1024 truncates a handful of long
    # entries to their first half (which contains the title, frontmatter
    # keywords, and lede — the load-bearing semantic content). Faster
    # build, MPS-stable, no quality loss in practice.
    if hasattr(model, "max_seq_length") and getattr(model, "max_seq_length", 0) > 1024:
        model.max_seq_length = 1024

    _MODEL_CACHE[model_name] = model
    return model


# ── Task-prompt formatting (model-specific) ────────────────────────────────────


# Nomic embed v1.5 was trained with explicit task prefixes. Using the
# right one at the right time is what unlocks the asymmetric-retrieval
# quality lift vs models that don't make this distinction (MiniLM/BGE).
_NOMIC_QUERY_PREFIX = "search_query: "
_NOMIC_DOC_PREFIX = "search_document: "


def _format_query(query: str, model_name: str) -> str:
    """Prepend the model's retrieval-query prefix when one applies."""
    if "nomic-embed-text" in model_name:
        return _NOMIC_QUERY_PREFIX + query
    return query


def _format_doc(text: str, model_name: str) -> str:
    """Prepend the model's retrieval-document prefix when one applies."""
    if "nomic-embed-text" in model_name:
        return _NOMIC_DOC_PREFIX + text
    return text


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
    docs: list[_DocRecord] = field(default_factory=list[_DocRecord])
    embeddings: _FloatArray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
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
            [_format_query(query, self.model_name)],
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
    # Small batch size for MPS stability — a 1024-seq attention matrix at
    # batch_size=32 still hits MPS's per-tensor allocation ceiling. 8 is
    # safe on M-series Macs and barely slower than 32 for our corpus size
    # since the bottleneck is the per-layer matmul, not batching overhead.
    vecs = model.encode(
        [_format_doc(t, model_name) for t in texts],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=8,
    ).astype(np.float32)

    return EmbeddingIndex(
        knowledge_dir=knowledge_dir,
        model_name=model_name,
        docs=docs,
        embeddings=vecs,
        built_at=time.time(),
    )


def save_index(index: EmbeddingIndex, path: Path | None = None) -> Path:
    """Pickle the index atomically. Returns the path written.

    Delegates to ``bm25.save_pickled`` for a concurrency-safe write (unique
    temp file per writer + atomic ``os.replace``), so racing rebuilds from the
    daemon's parallel Librarian threads can't interleave into a corrupt index.
    """
    target = path or _default_index_path(index.knowledge_dir)
    return save_pickled(index, target)


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
        cached = load_pickled(target, EmbeddingIndex)
        if (
            cached is not None
            and cached.knowledge_dir == knowledge_dir
            and cached.model_name == model_name
            and not is_stale(
                cached.docs,
                {str(p): p.stat().st_mtime for p in _iter_markdown(knowledge_dir)},
            )
        ):
            return cached

    index = build_index(knowledge_dir, model_name=model_name)
    try:
        save_index(index, target)
    except OSError:
        # Persistence is best-effort; in-memory index is still usable.
        pass
    return index
