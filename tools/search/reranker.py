"""Cross-encoder reranker on top of hybrid retrieval.

After BM25 + embedding + RRF surfaces topical candidates, this stage asks
the different question: "does this entry actually apply to *this* query?"
A cross-encoder co-attends over the (query, document) pair and outputs a
single relevance logit — a signal orthogonal to keyword/semantic overlap.

Latency budget: ~10 ms to score 20 candidates on CPU with the L-12 MiniLM
model, ~3-5 ms on Apple MPS. The daemon keeps the model warm in memory so
per-request cost stays inside the 200 ms hook budget.

Future upgrade path: ``BAAI/bge-reranker-base`` (278M params, ~1.1 GB
resident) is the quality ceiling at the next size class. Swap when that
footprint is acceptable; the public ``rerank`` contract here is the same.

Failure semantics: every public entry point returns ``None`` (or an
unchanged result) on any internal error. Callers fall back to the
upstream RRF ordering. The daemon must never crash because the
cross-encoder hiccupped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from sentence_transformers import CrossEncoder

from _device import select_device

log = logging.getLogger("agent_mem_daemon.reranker")

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Module-level cache so the daemon pays the load cost exactly once.
_MODEL_CACHE: dict[str, Any] = {}


def _load_model(model_name: str) -> Any:
    """Lazy-load and cache the cross-encoder. Raises on failure.

    Originally this returned ``None`` on failure and the rerank path
    silently degraded to no-rerank. That hides a real problem: if the
    cross-encoder can't load (no network on first run, model gone from
    HuggingFace, disk full, etc.) the operator wants to know loudly, not
    discover the rerank stage has been quietly disabled for weeks. So
    load failures now propagate out — the caller decides whether to
    crash the daemon or surface a clear error to the hook.
    """
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    device = select_device()
    model = CrossEncoder(model_name, device=device)
    _MODEL_CACHE[model_name] = model
    return model


def ensure_model_loaded(model_name: str = DEFAULT_MODEL) -> None:
    """Eagerly load the cross-encoder; raise if unavailable.

    Call this from daemon startup so a missing/unreachable model crashes
    the daemon visibly at boot rather than at the first priming request.
    """
    _load_model(model_name)


def rerank(
    query: str,
    candidates: List[Tuple[Path, str]],
    *,
    model_name: str = DEFAULT_MODEL,
) -> Optional[List[Tuple[Path, float]]]:
    """Score (query, body) pairs with the cross-encoder; return sorted desc.

    Args:
        query: the user prompt / search text.
        candidates: list of ``(path, body_text)`` tuples. ``body_text`` is
            what the model co-attends with the query — full entry body
            (capped by the model's max-seq-len internally).
        model_name: HuggingFace cross-encoder identifier.

    Returns:
        ``[(path, score), ...]`` sorted by relevance score desc on
        success — ``None`` on any failure (model missing, predict raised,
        empty input). Callers treat ``None`` as "skip rerank, use the
        upstream ordering verbatim".
    """
    if not query or not query.strip():
        return None
    if not candidates:
        return None

    # Load failures propagate — the daemon should not silently degrade to
    # no-rerank if the model is unreachable. See ``_load_model``'s
    # docstring for the rationale.
    model = _load_model(model_name)

    pairs = [(query, body) for _, body in candidates]
    try:
        scores = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        log.exception("reranker: predict raised on %d candidates", len(pairs))
        return None

    paths = [p for p, _ in candidates]
    scored: List[Tuple[Path, float]] = list(zip(paths, (float(s) for s in scores)))
    scored.sort(key=lambda t: (-t[1], str(t[0])))
    return scored
