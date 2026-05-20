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
from typing import Any, Dict, Iterable, List, Optional, Tuple


log = logging.getLogger("agent_mem_daemon.priming")


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


_HEADER = "## Ultan — your library says (cite or follow when applicable)"
_FOOTER = (
    "*Wikilinks resolve to real files in `~/.agent-mem/knowledge/<wikilink>.md` — "
    "Read one for the full entry, or call `/ultan-advisor <question>` for synthesis.*"
)


# ── Frontmatter helpers (kept here so we don't depend on scholar_prompt,
# which would create an import cycle — scholar.py imports priming) ────


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """Return parsed YAML frontmatter or ``{}`` on any failure.

    Permissive: a malformed entry just yields ``{}`` rather than killing
    the priming refresh.
    """
    import yaml

    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if isinstance(fm, dict):
            return fm
    except yaml.YAMLError:
        pass
    return {}


def _first_line(value: Any) -> str:
    """First non-empty line of a frontmatter value, normalised.

    ``applies-when`` is typically a block scalar with multiple lines; we
    want the first situation listed so the user gets a tight one-liner
    next to the wikilink.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        for item in value:
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
    try:
        n = int(raw)
        return n if n > 0 else 0
    except (TypeError, ValueError):
        return 0


def _entry_summary(fm: Dict[str, Any], path: Path) -> str:
    """One-line summary: prefer applies-when's first line, else title,
    else the slug. Always non-empty."""
    s = _first_line(fm.get("applies-when") or fm.get("applies_when"))
    if s:
        return s
    title = fm.get("title")
    if isinstance(title, str) and title.strip():
        return _shorten(title.strip())
    return path.stem.replace("-", " ")


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


def extract_buffer_text(packets: Iterable[Dict[str, Any]]) -> str:
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
        for prop in packet.get("proposals") or []:
            parts.extend(_collect_strings(prop))
        for itr in packet.get("interrupts") or []:
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


def _collect_strings(obj: Any) -> List[str]:
    """Recursively pull every string value out of a nested dict/list.

    Skips keys that are pure structural ids (no semantic content) so we
    don't dilute the BM25 query with UUIDs and integers cast to str.
    """
    skip_keys = {
        "_action_index", "action", "action_index", "session_id",
        "lesson_id", "lesson_path", "match_score", "librarian_confidence",
        "turn_id",
    }
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            out.extend(_collect_strings(v))
    elif isinstance(obj, list):
        for item in obj:
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
    try:
        from bm25 import load_or_build  # provided by agent-mem-search
    except ImportError as e:
        log.warning("priming: bm25 backend not importable (%s); skipping", e)
        return []

    if not knowledge_dir.exists():
        return []

    try:
        index = load_or_build(knowledge_dir)
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
    try:
        from embeddings import load_or_build  # provided by agent-mem-search
    except ImportError:
        # No noisy warning — embeddings are an optional retrieval lane.
        # BM25 still runs. Log once at DEBUG so the operator can spot it.
        log.debug("priming: embeddings backend not importable; bm25-only mode")
        return []

    if not knowledge_dir.exists():
        return []

    try:
        index = load_or_build(knowledge_dir)
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

    # Filter out very weak semantic matches — cosine below ~0.25 is
    # noise from this model and would poison RRF with irrelevant
    # paths. Empirical: gibberish queries against unrelated entries
    # produce scores in the 0.02-0.05 range; genuine paraphrase
    # matches sit at 0.4+.
    EMBEDDING_NOISE_FLOOR = 0.25
    return [
        (Path(h.path), float(h.score))
        for h in hits
        if float(h.score) >= EMBEDDING_NOISE_FLOOR
    ]


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


def _hybrid_search(
    knowledge_dir: Path,
    query: str,
    *,
    k: int,
) -> List[Tuple[Path, float]]:
    """Run BM25 and embedding search in parallel, merge via RRF.

    Each lane pulls a few more than ``k`` so RRF has overlap to work
    with. If the embedding lane is unavailable (no model installed,
    fresh corpus, transient failure), gracefully degrades to BM25-only.
    """
    # Pull 2× k from each lane so RRF has a chance to surface entries
    # that one lane ranks low but the other ranks high.
    per_lane_k = max(k * 2, k + 5)
    bm25_hits = _bm25_search(knowledge_dir, query, k=per_lane_k)
    emb_hits = _embedding_search(knowledge_dir, query, k=per_lane_k)
    if not bm25_hits and not emb_hits:
        return []
    if not emb_hits:
        # Embedding lane unavailable — fall back to BM25 verbatim.
        return bm25_hits[:k]
    if not bm25_hits:
        return emb_hits[:k]
    return _rrf_merge([bm25_hits, emb_hits], k_top=k)


def _boost_with_reinforcement(
    hits: List[Tuple[Path, float]],
) -> List[Tuple[Path, float, int]]:
    """Re-rank by ``score + reinforced * 0.5``.

    The multiplier is gentle on purpose: reinforced is an integer
    counter the user has driven up by repetition, and BM25 scores on a
    small corpus typically sit in the 1.0–4.0 range, so half a point per
    reinforcement reliably nudges a reasserted entry up without
    overwhelming a genuinely better BM25 match.

    Returns ``(path, ranked_score, reinforced_count)`` sorted desc by
    the BOOSTED score. Stable on tie via path string so output is
    deterministic across runs.
    """
    enriched: List[Tuple[Path, float, int, float]] = []
    for path, score in hits:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            enriched.append((path, score, 0, score))
            continue
        fm = _parse_frontmatter(text)
        reinforced = _reinforced_count(fm)
        boosted = score + reinforced * 0.5
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
    """Render one bullet line. ``title_only`` drops the summary suffix
    (used as a fallback when the budget is tight)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    fm = _parse_frontmatter(text)
    link = _wikilink_path(path, knowledge_dir)
    count_suffix = f" (×{reinforced})" if reinforced > 0 else ""
    if title_only:
        return f"- [[{link}]]{count_suffix}"
    summary = _entry_summary(fm, path)
    if not summary:
        return f"- [[{link}]]{count_suffix}"
    return f"- [[{link}]]{count_suffix} — {summary}"


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

    # Budget accounting: header + blank + bullets + blank + footer.
    overhead = len(_HEADER) + 2 + 2 + len(_FOOTER)
    body_budget = max(0, char_budget - overhead)

    def _join(bullets: List[str]) -> str:
        return f"{_HEADER}\n\n" + "\n".join(bullets) + f"\n\n{_FOOTER}\n"

    # Attempt 1: full bullets, all top_k.
    bullets = [
        b for b in (
            _render_bullet(p, r, knowledge_dir) for p, _score, r in sliced
        ) if b
    ]
    if not bullets:
        return ""
    full = _join(bullets)
    if len(full) <= char_budget:
        return full

    # Attempt 2: title-only bullets.
    bullets_terse = [
        b for b in (
            _render_bullet(p, r, knowledge_dir, title_only=True)
            for p, _score, r in sliced
        ) if b
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
        if not isinstance(rolling_buffer_text, str):
            rolling_buffer_text = str(rolling_buffer_text or "")

        # Empty buffer: clear the file if it exists so we don't leave
        # stale priming around. Returning silently is fine if it never
        # existed in the first place.
        if not rolling_buffer_text.strip():
            if out_path.exists():
                _atomic_write(out_path, "")
                return out_path
            return None

        hits = _hybrid_search(
            knowledge_dir, rolling_buffer_text,
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

        ranked = _boost_with_reinforcement(hits)
        body = _assemble_output(
            ranked, knowledge_dir,
            top_k=top_k, char_budget=char_budget,
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
