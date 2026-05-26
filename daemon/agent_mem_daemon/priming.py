"""Tier 1 — Ambient Priming.

The expensive way to consult the library is ``/ultan-advisor`` (an LLM
call costing tens of cents and tens of seconds). That cost makes the
agent reluctant to consult memory at all, so memory gets ignored. This
module is the cheap alternative: a passive priming layer the daemon
refreshes whenever the Scholar finishes a batch.

The cognitive analog is "familiarity" / spreading activation: after a
batch the daemon already knows what the session has been chewing on, so
it BM25-searches the library for the top-K most-relevant entries, boosts
by each entry's ``reinforced`` counter (the user has reasserted these,
so weight them higher), trims to a small char budget, and writes the
result to ``~/.agent-mem/hot-context.md``. On the next ``UserPromptSubmit``
hook the file is injected as ``additionalContext``, putting those entries
"in mind" for the next turn without an LLM call.

Output format (≤ ``char_budget`` chars total, wikilinks resolve via the
agent's Read tool):

    ## Library priming (you may know more about this than you think)

    - [[global/python/use-uv-not-pip]] (×4) — always uv, never pip
    - [[projects/x/conventions/no-deploy-without-approval]] (×2) — explicit OK before prod
    - [[global/preferences/no-end-summaries]] — don't summarise the diff at the end

    *Tip-of-the-tongue? `/ultan-advisor <q>` pulls the full entry.*

Design notes:
  - Never raises; logs and returns on every failure path. The Scholar's
    post-batch flow swallows any exception anyway, but defensive coding
    in here keeps the log readable.
  - Atomic write via temp-file + ``os.replace`` — the hook reads this on
    every turn, so a partial write would surface as a corrupted nudge.
  - Idempotent: writes the SAME bytes given the same inputs (we don't
    embed timestamps), so re-runs with an unchanged buffer don't churn
    the file's mtime in a way that matters. Tests pin on byte equality.
  - The BM25 index belongs to ``agent-mem-search`` (path dep). We import
    lazily inside the function so a missing/broken bm25 install never
    breaks daemon startup — same defensive style as ``library_tools.py``.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, cast

import yaml
from bm25 import load_or_build as bm25_load_or_build
from embeddings import load_or_build as embeddings_load_or_build
from reranker import rerank as _cross_encoder_rerank

from .paths import home as _agent_mem_home

log = logging.getLogger("agent_mem_daemon.priming")


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


_HEADER = "## Ultan — your library says (cite or follow when applicable)"
_FOOTER = (
    "*Wikilinks resolve to real entries. Use the `ultan-search` skill to read one "
    "(returns content + sibling entries + subfolders + parent README so you can traverse), "
    "or `/ultan-advisor <question>` to have Sonnet + Opus intelligently "
    "synthesise across multiple entries.*"
)


# ── Frontmatter helpers (kept here so we don't depend on scholar_prompt,
# which would create an import cycle — scholar.py imports priming) ────


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """Return parsed YAML frontmatter or ``{}`` on any failure.

    Permissive: a malformed entry just yields ``{}`` rather than killing
    the priming refresh.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    try:
        fm: object = yaml.safe_load(m.group(1)) or {}
        if isinstance(fm, dict):
            # yaml.safe_load yields a dict[Any, Any]; the rest of this module
            # treats keys as strings, so coerce them on the way out.
            return {str(k): v for k, v in cast("dict[object, Any]", fm).items()}
    except yaml.YAMLError:
        pass
    return {}


def _first_line(value: object) -> str:
    """First non-empty line of a frontmatter value, normalised.

    ``applies-when`` is typically a block scalar with multiple lines; we
    want the first situation listed so the user gets a tight one-liner
    next to the wikilink.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        items = cast("list[object]", value)
        for item in items:
            s = str(item).strip()
            if s:
                return _shorten(s)
        return ""
    text = str(value)
    for line in text.splitlines():
        s = line.strip()
        if s:
            return _shorten(s)
    return ""


def _shorten(text: str, *, max_chars: int = 80) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _reinforced_count(fm: Dict[str, Any]) -> int:
    raw = fm.get("reinforced")
    if raw is None:
        return 0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _entry_title(fm: Dict[str, Any], path: Path) -> str:
    """Punchy human-readable title for the entry. Falls back to the
    slug (de-kebabed) if no ``title:`` frontmatter is present."""
    title = fm.get("title")
    if isinstance(title, str) and title.strip():
        return _shorten(title.strip())
    return path.stem.replace("-", " ")


def _entry_applies_when(fm: Dict[str, Any]) -> str:
    """First line of the applies-when block, shortened. May be empty
    if the frontmatter has no applies-when at all."""
    return _first_line(fm.get("applies-when") or fm.get("applies_when"))


def _wikilink_path(entry: Path, knowledge_dir: Path) -> str:
    """Relative path from knowledge_dir with no ``.md`` suffix.

    Matches the wikilink convention the rest of the library uses (see
    scholar_prompt's wikilink-resolution rules)."""
    try:
        rel = entry.resolve().relative_to(knowledge_dir.resolve())
    except ValueError:
        # Fall back to the absolute string — better than crashing.
        rel = entry
    s = str(rel).replace(os.sep, "/")
    if s.endswith(".md"):
        s = s[:-3]
    return s


# ── Buffer-text extraction (the input for BM25) ──────────────────────


def extract_buffer_text(packets: Iterable[Mapping[str, Any]]) -> str:
    """Flatten the Scholar's incoming packets back into plain text for
    BM25 to chew on.

    Packets carry ``proposals`` and ``interrupts``, not the raw rolling
    buffer events (the Librarian summarises before handing off). We pull
    every string-valued field out of every proposal and interrupt — this
    captures the Librarian's reasoning, the proposed entry bodies, the
    cited evidence quotes, and the matching-applies-when phrases. That
    set is a faithful proxy for "what the session has been talking
    about" because the Librarian only emits proposals for material it
    found worth flagging.

    Newline-joined. Empty string if no usable text is present (caller
    short-circuits in that case).
    """
    parts: List[str] = []
    for packet in packets or []:
        if not isinstance(packet, dict):
            continue
        proposals: object = packet.get("proposals") or []
        if isinstance(proposals, list):
            for prop in cast("list[object]", proposals):
                parts.extend(_collect_strings(prop))
        interrupts: object = packet.get("interrupts") or []
        if isinstance(interrupts, list):
            for itr in cast("list[object]", interrupts):
                parts.extend(_collect_strings(itr))
    # Strip empties and dedupe to keep the BM25 query reasonable.
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return "\n".join(out)


def _collect_strings(obj: object) -> List[str]:
    """Recursively pull every string value out of a nested dict/list.

    Skips keys that are pure structural ids (no semantic content) so we
    don't dilute the BM25 query with UUIDs and integers cast to str.
    """
    skip_keys = {
        "_action_index",
        "action",
        "action_index",
        "session_id",
        "lesson_id",
        "lesson_path",
        "match_score",
        "librarian_confidence",
        "turn_id",
    }
    out: List[str] = []
    if isinstance(obj, dict):
        items = cast("dict[object, object]", obj)
        for k, v in items.items():
            if k in skip_keys:
                continue
            out.extend(_collect_strings(v))
    elif isinstance(obj, list):
        list_items = cast("list[object]", obj)
        for item in list_items:
            out.extend(_collect_strings(item))
    elif isinstance(obj, str):
        s = obj.strip()
        # Strip path-only strings to keep BM25 tokens topical.
        if s:
            out.append(s)
    # ints / floats / None / bools deliberately excluded.
    return out


# ── Hybrid retrieval (BM25 + semantic embeddings via RRF) ────────────


def _bm25_search(
    knowledge_dir: Path,
    query: str,
    *,
    k: int,
) -> List[Tuple[Path, float]]:
    """Run BM25 over the library. Returns ``(path, score)`` pairs.

    Tolerant: returns ``[]`` when the bm25 backend isn't installed, when
    the knowledge dir is missing, or when the index can't be built.
    """
    if not query:
        return []

    if not knowledge_dir.exists():
        return []

    try:
        index = bm25_load_or_build(knowledge_dir)
    except FileNotFoundError:
        return []
    except Exception:
        log.exception("priming: bm25 index load/build failed")
        return []

    try:
        hits = index.search(query, k=max(1, k))
    except Exception:
        log.exception("priming: bm25 search raised")
        return []

    return [(Path(p), float(score)) for p, score, _snippet in hits]


def _embedding_search(
    knowledge_dir: Path,
    query: str,
    *,
    k: int,
) -> List[Tuple[Path, float]]:
    """Run semantic search via sentence-transformer embeddings.

    Tolerant in exactly the same way as ``_bm25_search``: returns ``[]``
    when the ``embeddings`` package, model, or index isn't available.
    Catches paraphrasing BM25 misses (e.g. "deploy to prod" vs
    "ship to production").
    """
    if not query:
        return []

    if not knowledge_dir.exists():
        return []

    try:
        index = embeddings_load_or_build(knowledge_dir)
    except FileNotFoundError:
        return []
    except Exception:
        log.exception("priming: embedding index load/build failed")
        return []

    try:
        hits = index.search(query, k=max(1, k))
    except Exception:
        log.exception("priming: embedding search raised")
        return []

    # Filter out very weak semantic matches before they reach RRF.
    # Calibrated for nomic-embed-text-v1.5 — its cosine distribution
    # sits noticeably higher than older models (unrelated ~0.40-0.50,
    # genuine paraphrase matches ~0.65+). A floor at 0.50 separates the
    # populations: nonsense queries against unrelated entries fall under
    # it (so gibberish in == gibberish-free out, the "no real match
    # clears stale priming" semantic), while real paraphrase matches
    # comfortably clear it. Re-tune if the embedder swaps again — see
    # test_search_returns_empty_for_unrelated_query in tools/search for
    # the empirical handle.
    EMBEDDING_NOISE_FLOOR = 0.50
    return [(Path(h.path), float(h.score)) for h in hits if float(h.score) >= EMBEDDING_NOISE_FLOOR]


def _rrf_merge(
    rankings: List[List[Tuple[Path, float]]],
    *,
    k_top: int,
    rrf_k: int = 60,
) -> List[Tuple[Path, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    For each path appearing in any input ranking, score = sum over
    rankings of ``1 / (rrf_k + rank_in_that_ranking)``. The constant
    ``rrf_k`` dampens the contribution from low ranks; 60 is the value
    from the original RRF paper (Cormack et al. '09).

    The RRF score is rank-based, not magnitude-based, so it's robust
    against BM25 and cosine-similarity living in different score ranges.
    Returns the top ``k_top`` paths with their RRF scores.
    """
    rrf_scores: Dict[Path, float] = {}
    for ranking in rankings:
        if not ranking:
            continue
        for rank, (path, _score) in enumerate(ranking):
            rrf_scores[path] = rrf_scores.get(path, 0.0) + 1.0 / (rrf_k + rank + 1)
    ordered = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ordered[:k_top]


# Minimum cross-encoder logit for a candidate to count as a real match.
# ms-marco-MiniLM-L-12-v2 outputs unnormalized relevance logits — strong
# matches sit at +5 to +10, weak-but-real around -2 to +2, definite
# non-matches at -5 and below. The "no real match" filter is mostly
# handled upstream by ``EMBEDDING_NOISE_FLOOR``; this floor is the
# rerank's own guard against an outright non-match slipping through
# the embedding lane on lexical-only signal. A permissive value lets
# weak-but-real matches through (a tightly-scoped lesson on a less
# common topic can rerank around 0). Tighten toward 0 if noise leaks
# through; loosen toward -5 if valid matches get dropped.
_RERANK_SCORE_FLOOR = -3.0


def _rerank_candidates(
    query: str,
    candidates: List[Tuple[Path, float]],
    *,
    k: int,
) -> List[Tuple[Path, float]]:
    """Cross-encoder rerank pass on ``candidates``; degrade to input on failure.

    Reads each candidate's body once, hands the ``(query, body)`` pairs to
    the cross-encoder, and returns the top-``k`` by relevance score with
    the cross-encoder score in place of the RRF score. Candidates scoring
    below ``_RERANK_SCORE_FLOOR`` are dropped — see the constant's docstring
    for why. The reranker returns ``None`` on any internal failure (model
    missing, predict raised, empty input) — in that case we hand back the
    upstream order truncated to ``k`` so retrieval still works without
    rerank.
    """
    if not candidates:
        return []

    bodies: List[Tuple[Path, str]] = []
    for path, _ in candidates:
        try:
            bodies.append((path, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            # Skip unreadable candidates rather than fail the whole rerank.
            continue
    if not bodies:
        return candidates[:k]

    reranked = _cross_encoder_rerank(query, bodies)
    if reranked is None:
        return candidates[:k]
    above_floor = [(p, s) for p, s in reranked if s >= _RERANK_SCORE_FLOOR]
    return above_floor[:k]


def _hybrid_search(
    knowledge_dir: Path,
    query: str,
    *,
    k: int,
) -> List[Tuple[Path, float]]:
    """Run BM25 + embedding search, fuse with RRF, then cross-encoder rerank.

    Three-stage retrieval, low precision → high precision:

    1. **Recall**: BM25 and embedding lanes each pull ``k * 4`` candidates.
       BM25 catches exact-token matches; embeddings catch paraphrase.
    2. **Fusion**: RRF merges the two ranked lists into ``k * 4``
       topically-relevant candidates (rank-based, robust to score scale
       differences between BM25 and cosine).
    3. **Rerank**: a cross-encoder co-attends over each ``(query, body)``
       pair and scores actual applicability — orthogonal signal to the
       overlap-based lanes upstream. Returns the top ``k`` reranked.

    Graceful degradation: if the embedding lane is unavailable, we
    fall back to BM25-only feed into the reranker. If the reranker is
    unavailable, we fall back to the RRF order verbatim. The hot path
    must keep working when any single component breaks.
    """
    # Pull 2× k from each lane. Wider over-fetch (e.g. 4×) would give
    # the reranker more variety, but cross-encoder predict on MPS scales
    # nonlinearly — measured ~60ms @ 5 candidates, ~170ms @ 10, ~600ms @
    # 20 (graph-cache effects above the small-tensor threshold). At 4×
    # the rerank stage alone overruns the 200ms hook budget. Stick to 2×
    # so end-to-end fits.
    per_lane_k = max(k * 2, k + 5)
    bm25_hits = _bm25_search(knowledge_dir, query, k=per_lane_k)
    emb_hits = _embedding_search(knowledge_dir, query, k=per_lane_k)
    if not bm25_hits and not emb_hits:
        return []

    if not emb_hits:
        fused = bm25_hits
    elif not bm25_hits:
        fused = emb_hits
    else:
        fused = _rrf_merge([bm25_hits, emb_hits], k_top=per_lane_k)

    return _rerank_candidates(query, fused, k=k)


# Scope bonuses are calibrated against the RRF top-rank score (~0.016 for
# rank 1, ~0.033 with both lanes hitting the top). Current-project +0.020
# is roughly a 4-rank bump; global +0.005 is roughly a 1-rank bump; cross-
# project entries get no bonus. Designed to break near-ties and gently
# reorder, not to override a strong topical match. Tune here if needed.
_SCOPE_BONUS_CURRENT = 0.020
_SCOPE_BONUS_GLOBAL = 0.005


# Project-scope alias resolution lives in ``tools/search/aliases.py``
# (shared with the hook side) — see that module's docstring for shape
# and contract.
from aliases import bucket_canonical_slug, load_aliases  # noqa: E402


def _path_project_bucket(path: Path, knowledge_dir: Path) -> Optional[str]:
    """Return the project bucket name for entries under ``projects/<bucket>/``,
    the sentinel ``"__global__"`` for entries under ``global/``, or ``None``
    for root-level entries (READMEs etc. that belong to no specific scope).
    """
    try:
        rel = path.resolve().relative_to(knowledge_dir.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "projects":
        return parts[1]
    if parts and parts[0] == "global":
        return "__global__"
    return None


def _scope_bonus(
    bucket: Optional[str],
    current_project_slug: Optional[str],
    aliases: Mapping[str, str],
) -> float:
    """Score adjustment for project-scope preference.

    - Entries under the user's current project: ``+_SCOPE_BONUS_CURRENT``
    - Global entries: ``+_SCOPE_BONUS_GLOBAL``
    - Other-project entries (and root entries with no bucket): ``0.0``

    Matching is done by translating the bucket name to its canonical
    slug (via ``_bucket_canonical_slug``) and comparing against the
    session's slug — buckets not listed in the alias map default to
    slug == bucket name.
    """
    if bucket is None:
        return 0.0
    if bucket == "__global__":
        return _SCOPE_BONUS_GLOBAL
    if current_project_slug and bucket_canonical_slug(bucket, aliases) == current_project_slug:
        return _SCOPE_BONUS_CURRENT
    return 0.0


def _boost_with_reinforcement(
    hits: List[Tuple[Path, float]],
    *,
    knowledge_dir: Optional[Path] = None,
    current_project_slug: Optional[str] = None,
) -> List[Tuple[Path, float, int]]:
    """Re-rank by ``score + reinforced * 0.5 + scope_bonus``.

    The reinforcement multiplier is gentle on purpose: reinforced is an
    integer counter the user has driven up by repetition, and BM25 scores
    on a small corpus typically sit in the 1.0–4.0 range, so half a point
    per reinforcement reliably nudges a reasserted entry up without
    overwhelming a genuinely better BM25 match.

    When ``knowledge_dir`` is supplied, also applies the scope bonus
    (see ``_scope_bonus``). ``current_project_slug`` may be ``None``;
    in that case only the global bonus fires (no current-project boost).
    Each bucket's canonical slug is read from
    ``~/.agent-mem/project-aliases.json`` (see ``aliases.load_aliases``);
    buckets absent from that file default to ``slug == bucket name``.

    Returns ``(path, ranked_score, reinforced_count)`` sorted desc by
    the BOOSTED score. Stable on tie via path string so output is
    deterministic across runs.
    """
    # Load the alias map once per call. The lookup is cheap (one small
    # JSON read) and skipping it on the no-scope path keeps that path
    # zero-cost.
    aliases: Mapping[str, str] = {}
    if knowledge_dir is not None:
        aliases = load_aliases(_agent_mem_home())

    enriched: List[Tuple[Path, float, int, float]] = []
    for path, score in hits:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            enriched.append((path, score, 0, score))
            continue
        fm = _parse_frontmatter(text)
        reinforced = _reinforced_count(fm)
        scope = 0.0
        if knowledge_dir is not None:
            scope = _scope_bonus(
                _path_project_bucket(path, knowledge_dir),
                current_project_slug,
                aliases,
            )
        boosted = score + reinforced * 0.5 + scope
        enriched.append((path, boosted, reinforced, score))
    enriched.sort(key=lambda t: (-t[1], str(t[0])))
    return [(p, ranked, reinforced) for p, ranked, reinforced, _orig in enriched]


# ── Rendering ─────────────────────────────────────────────────────────


def _render_bullet(
    path: Path,
    reinforced: int,
    knowledge_dir: Path,
    *,
    title_only: bool = False,
) -> str:
    """Render one bullet line. ``title_only`` drops the applies-when
    hook (used as a fallback when the budget is tight) while keeping
    the title so the agent always has something semantic to anchor on.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    fm = _parse_frontmatter(text)
    link = _wikilink_path(path, knowledge_dir)
    count_suffix = f" (×{reinforced})" if reinforced > 0 else ""
    title = _entry_title(fm, path)
    if title_only:
        if not title:
            return f"- [[{link}]]{count_suffix}"
        return f"- [[{link}]]{count_suffix} — {title}"
    hook = _entry_applies_when(fm)
    if title and hook:
        return f"- [[{link}]]{count_suffix} — {title} — {_shorten(hook, max_chars=70)}"
    if title:
        return f"- [[{link}]]{count_suffix} — {title}"
    if hook:
        return f"- [[{link}]]{count_suffix} — {_shorten(hook)}"
    return f"- [[{link}]]{count_suffix}"


def _assemble_output(
    ranked: List[Tuple[Path, float, int]],
    knowledge_dir: Path,
    *,
    top_k: int,
    char_budget: int,
) -> str:
    """Compose the hot-context body, respecting the char budget.

    Strategy:
      - Reserve room for header + footer + the two blank-line separators.
      - Try full bullets first; if we blow the budget, fall back to
        title-only bullets; if still over, drop trailing bullets until
        we fit.
    """
    if not ranked:
        return ""

    sliced = ranked[:top_k]

    def _join(bullets: List[str]) -> str:
        return f"{_HEADER}\n\n" + "\n".join(bullets) + f"\n\n{_FOOTER}\n"

    # Attempt 1: full bullets, all top_k.
    bullets = [b for b in (_render_bullet(p, r, knowledge_dir) for p, _score, r in sliced) if b]
    if not bullets:
        return ""
    full = _join(bullets)
    if len(full) <= char_budget:
        return full

    # Attempt 2: title-only bullets.
    bullets_terse = [
        b
        for b in (_render_bullet(p, r, knowledge_dir, title_only=True) for p, _score, r in sliced)
        if b
    ]
    terse = _join(bullets_terse)
    if len(terse) <= char_budget:
        return terse

    # Attempt 3: drop trailing bullets until we fit. Always keep at
    # least one — an empty list defeats the purpose.
    while len(bullets_terse) > 1:
        bullets_terse.pop()
        candidate = _join(bullets_terse)
        if len(candidate) <= char_budget:
            return candidate

    # Final fallback: the single best bullet only, even if over budget.
    return _join(bullets_terse)


# ── Public API ────────────────────────────────────────────────────────


def refresh_hot_context(
    knowledge_dir: Path,
    rolling_buffer_text: str,
    out_path: Path,
    *,
    top_k: int = 5,
    char_budget: int = 1500,
    current_project_slug: Optional[str] = None,
) -> Optional[Path]:
    """Recompute the hot-context file from the recent buffer.

    Args:
        knowledge_dir: root of the library to search.
        rolling_buffer_text: plain-text proxy of recent session activity.
            Empty string is OK — the function clears the hot-context
            file in that case (no stale priming).
        out_path: where to write the new hot-context body.
        top_k: how many entries to include before budget trimming.
        char_budget: hard upper bound on output length.

    Returns:
        The path written, or ``None`` if nothing was written (empty
        buffer with no existing file, or every step failed).

    Never raises. All failures are logged at WARN/EXCEPTION and the
    function returns; the daemon must not crash because BM25 hiccupped.
    """
    try:
        # Empty buffer: clear the file if it exists so we don't leave
        # stale priming around. Returning silently is fine if it never
        # existed in the first place.
        if not rolling_buffer_text.strip():
            if out_path.exists():
                _atomic_write(out_path, "")
                return out_path
            return None

        hits = _hybrid_search(
            knowledge_dir,
            rolling_buffer_text,
            # Pull a few extras so the reinforcement boost can promote a
            # low-BM25-but-heavily-reinforced entry into the top_k.
            k=max(top_k * 2, top_k + 3),
        )
        if not hits:
            # Buffer had content but nothing matched — wipe any stale
            # priming so the hook doesn't inject irrelevant entries.
            if out_path.exists():
                _atomic_write(out_path, "")
                return out_path
            return None

        ranked = _boost_with_reinforcement(
            hits,
            knowledge_dir=knowledge_dir,
            current_project_slug=current_project_slug,
        )
        body = _assemble_output(
            ranked,
            knowledge_dir,
            top_k=top_k,
            char_budget=char_budget,
        )
        if not body:
            return None

        # Idempotence: skip the write (and the mtime bump) if the
        # content is byte-identical to what's already on disk.
        try:
            current = out_path.read_text(encoding="utf-8")
            if current == body:
                return out_path
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            pass

        _atomic_write(out_path, body)
        return out_path
    except Exception:
        log.exception("priming.refresh_hot_context: unexpected failure")
        return None


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via tmp+rename.

    Used so the hook never sees a half-written file. Raises on disk
    error — callers wrap with try/except to keep the daemon alive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
