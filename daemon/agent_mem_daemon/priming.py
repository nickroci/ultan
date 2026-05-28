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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, cast

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
    "synthesise across multiple entries.*\n\n"
    "*Entries are living — the curator updates them on new info. "
    "You may cite the current text; no need to maintain them yourself.*"
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


# ── Hybrid retrieval (BM25 + semantic embeddings, weighted fusion) ───

# Minimum cosine for an embedding hit to count as a real semantic match.
# Calibrated for nomic-embed-text-v1.5 — its cosine distribution sits
# noticeably higher than older models (unrelated ~0.40-0.50, genuine
# paraphrase matches ~0.65+). A floor at 0.50 separates the populations:
# nonsense queries against unrelated entries fall under it (so gibberish
# in == gibberish-free out, the "no real match clears stale priming"
# semantic), while real paraphrase matches comfortably clear it. Re-tune
# if the embedder swaps again — see test_search_returns_empty_for_unrelated_query
# in tools/search for the empirical handle.
EMBEDDING_NOISE_FLOOR = 0.50


def _is_readme(path: Path) -> bool:
    """True for folder-overview README files.

    READMEs are navigation/overview text, not stored lessons. They're
    generic enough to act as "universal donors" in the embedding lane — a
    vague or proper-noun query matches their overview text more strongly
    than any specific entry — so they crowd real lessons out of priming.
    Excluded from retrieval candidates entirely (see ``_hybrid_search``).
    """
    return path.name.lower() == "readme.md"


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

    # Filter out very weak semantic matches (see EMBEDDING_NOISE_FLOOR).
    return [(Path(h.path), float(h.score)) for h in hits if float(h.score) >= EMBEDDING_NOISE_FLOOR]


# Fusion weights. Lexical-leaning on purpose: BM25 carries rare-token /
# proper-noun matches (high-IDF terms the embedder has no representation
# for — e.g. a coined project name), while the embedding lane adds
# paraphrase recall. 0.6 / 0.4 favours lexical without silencing
# semantics. Rationale + measurements: a contentless or proper-noun query
# leaves the embedding lane near-flat (cosines clustered in a narrow band,
# no discrimination), so a rank-only fusion let that flat lane out-vote a
# strong, specific BM25 hit. Weighting by normalised magnitude lets the
# confident lane win.
_FUSION_W_BM25 = 0.6
_FUSION_W_EMB = 0.4


def _weighted_merge(
    bm25_hits: List[Tuple[Path, float]],
    emb_hits: List[Tuple[Path, float]],
    *,
    k_top: int,
    w_bm25: float = _FUSION_W_BM25,
    w_emb: float = _FUSION_W_EMB,
) -> List[Tuple[Path, float]]:
    """Convex score fusion of the BM25 and embedding lanes, leaning lexical.

    Replaces rank-only RRF. RRF discards score *magnitude*, so a strong
    high-IDF BM25 hit (a rare token the embedder can't place) was flattened
    to "just rank N" and out-voted by a flat embedding lane. Here each lane
    is normalised to ``[0, 1]`` and combined ``w_bm25 / w_emb``.

    Normalisation is fixed-reference, NOT per-query min-max — min-max
    amplifies a flat lane into full-range noise, and the embedding lane's
    cosines cluster in a narrow band on this corpus:

    - BM25: divided by the batch max (BM25 has no fixed ceiling).
    - Embeddings: ``(cos - floor) / (1 - floor)`` against
      :data:`EMBEDDING_NOISE_FLOOR` — hits are already floored there, so
      this maps ``[floor, 1] -> [0, 1]``.

    When a lane is empty its term drops out, so a query with no semantic
    matches above the floor degrades to pure lexical order — exactly what
    we want for rare-token / proper-noun queries. Returns the top
    ``k_top`` ``(path, fused_score)`` pairs, descending, stable on tie.
    """
    bm25_norm: Dict[Path, float] = {}
    if bm25_hits:
        max_bm25 = max(s for _, s in bm25_hits)
        if max_bm25 > 0:
            bm25_norm = {p: s / max_bm25 for p, s in bm25_hits}

    span = 1.0 - EMBEDDING_NOISE_FLOOR
    emb_norm: Dict[Path, float] = {}
    if emb_hits and span > 0:
        emb_norm = {p: max(0.0, min(1.0, (s - EMBEDDING_NOISE_FLOOR) / span)) for p, s in emb_hits}

    fused: Dict[Path, float] = {}
    for path in set(bm25_norm) | set(emb_norm):
        fused[path] = w_bm25 * bm25_norm.get(path, 0.0) + w_emb * emb_norm.get(path, 0.0)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], str(kv[0])))
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
    """Run BM25 + embedding search, fuse by weighted score, then rerank.

    Three-stage retrieval, low precision → high precision:

    1. **Recall**: BM25 and embedding lanes each pull ``2 * k`` candidates.
       BM25 catches exact-token / rare-term matches; embeddings catch
       paraphrase. README files are dropped here (see :func:`_is_readme`):
       they're navigation, and their generic overview text dominates the
       embedding lane for vague queries.
    2. **Fusion**: :func:`_weighted_merge` combines the lanes by normalised
       score, leaning lexical, so a confident BM25 hit on a rare token
       isn't out-voted by a flat embedding lane (rank-only RRF was).
    3. **Rerank**: a cross-encoder co-attends over each ``(query, body)``
       pair and scores actual applicability — orthogonal signal to the
       overlap-based lanes upstream. Returns the top ``k`` reranked.

    Graceful degradation: an empty lane simply drops out of the weighted
    merge. If the reranker is unavailable, we fall back to the fused order
    verbatim. The hot path must keep working when any component breaks.
    """
    # Pull 2× k from each lane. Wider over-fetch (e.g. 4×) would give
    # the reranker more variety, but cross-encoder predict on MPS scales
    # nonlinearly — measured ~60ms @ 5 candidates, ~170ms @ 10, ~600ms @
    # 20 (graph-cache effects above the small-tensor threshold). At 4×
    # the rerank stage alone overruns the 200ms hook budget. Stick to 2×
    # so end-to-end fits.
    per_lane_k = max(k * 2, k + 5)
    bm25_hits = [
        h for h in _bm25_search(knowledge_dir, query, k=per_lane_k) if not _is_readme(h[0])
    ]
    emb_hits = [
        h for h in _embedding_search(knowledge_dir, query, k=per_lane_k) if not _is_readme(h[0])
    ]
    if not bm25_hits and not emb_hits:
        return []

    fused = _weighted_merge(bm25_hits, emb_hits, k_top=per_lane_k)
    return _rerank_candidates(query, fused, k=k)


# Scope bonuses (and one penalty) nudge the ranked score in
# ``_boost_with_reinforcement``: prefer the current project, then global,
# and gently penalise other projects so cross-context entries only surface
# when nothing scoped beats them (the agent's complaint: vol-predictor
# entries surfacing during agent-mem work is noise).
#
# NOTE: these magnitudes (±0.01-0.02) are small relative to the rerank
# logits (~ -3..+10) and the reinforcement term (reinforced * 0.5) they're
# added to, so on the post-rerank scale the scope penalty is close to a
# no-op. Re-calibrating the boost formula's three terms onto a common
# scale is tracked separately — out of scope for the fusion change.
_SCOPE_BONUS_CURRENT = 0.020
_SCOPE_BONUS_GLOBAL = 0.005
_SCOPE_PENALTY_CROSS_PROJECT = -0.010


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
    - Other-project entries: ``_SCOPE_PENALTY_CROSS_PROJECT`` (negative)
    - Root entries with no resolvable bucket: ``0.0``

    Matching is done by translating the bucket name to its canonical
    slug (via ``_bucket_canonical_slug``) and comparing against the
    session's slug — buckets not listed in the alias map default to
    slug == bucket name. When ``current_project_slug`` is ``None`` we
    can't tell current-project from cross-project, so non-global
    project entries collapse to zero (no penalty without a baseline
    to compare against).
    """
    if bucket is None:
        return 0.0
    if bucket == "__global__":
        return _SCOPE_BONUS_GLOBAL
    if not current_project_slug:
        return 0.0
    if bucket_canonical_slug(bucket, aliases) == current_project_slug:
        return _SCOPE_BONUS_CURRENT
    return _SCOPE_PENALTY_CROSS_PROJECT


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


# Body excerpts inline the first ~250 chars of the top entries' bodies so
# the agent can cite the actual rule rather than guessing from the title.
# Per agent feedback: title-only priming caused at least one case where
# the agent reinvented a weaker version of an entry it had been "told" via
# wikilink. Cap chosen to fit ~3 bullets-with-bodies under the 1500-char
# total budget. See ``_extract_body_excerpt``.
_BODY_EXCERPT_MAX_CHARS = 250

# Freshness marker: entries with frontmatter ``updated`` (or ``created``
# as a fallback) within this many days get a star prefix so the agent can
# prioritise recent thinking. 7d is the user's working-week window.
_FRESHNESS_WINDOW_DAYS = 7
_FRESHNESS_MARKER = "★"


def _extract_body_excerpt(text: str, *, max_chars: int = _BODY_EXCERPT_MAX_CHARS) -> str:
    """Pull the first paragraph of body content as a short excerpt.

    Skips the YAML frontmatter and the entry's leading ``# Title``
    heading (the title is already rendered above the excerpt from
    frontmatter, so repeating it wastes the budget). Takes everything
    up to the next blank line, collapses whitespace, and truncates at
    ``max_chars`` on a word boundary with a trailing ellipsis if needed.

    Returns "" if the body is empty or unreadable.
    """
    # Strip frontmatter if present.
    fm_match = _FRONTMATTER_RE.match(text)
    body = text[fm_match.end() :] if fm_match else text

    # Walk lines, skipping blanks and any leading "# Heading" lines.
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        break
    if i >= len(lines):
        return ""

    # Collect until the next blank line — first paragraph only.
    paragraph: List[str] = []
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            break
        paragraph.append(stripped)
        i += 1
    if not paragraph:
        return ""

    text_one_line = " ".join(paragraph)
    text_one_line = " ".join(text_one_line.split())  # collapse internal whitespace
    if len(text_one_line) <= max_chars:
        return text_one_line

    # Truncate on a word boundary.
    cut = text_one_line[: max_chars - 1]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def _parse_iso_date(value: object) -> Optional[date]:
    """Best-effort parse of an ISO date (YYYY-MM-DD) from frontmatter."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        return None


def _is_fresh(fm: Dict[str, Any], *, today: Optional[date] = None) -> bool:
    """True if frontmatter ``updated`` (or ``created``) is within the
    freshness window. Today is overridable for deterministic tests."""
    when = _parse_iso_date(fm.get("updated")) or _parse_iso_date(fm.get("created"))
    if when is None:
        return False
    now = today or date.today()
    return (now - when) <= timedelta(days=_FRESHNESS_WINDOW_DAYS)


# Path-derived kind classification. We don't have a ``kind:`` frontmatter
# field yet, so this is best-effort heuristic from the directory layout
# the Scholar tends to use. Falls back to "" (no marker) when nothing
# obviously matches — better silent than wrongly classified.
_KIND_SEGMENTS: Tuple[Tuple[str, str], ...] = (
    ("conventions", "C"),  # convention — must follow
    ("preferences", "P"),  # preference — style/taste
    ("user/", "P"),  # user/* tends to be preferences
    ("findings", "F"),  # finding — informational
    ("research", "F"),  # research/* same
)


def _kind_marker(path: Path, knowledge_dir: Path, fm: Dict[str, Any]) -> str:
    """One-char kind marker (C/P/F/W) derived from path + frontmatter.

    Order matters — ``severity: block`` in frontmatter wins over a path
    classification because a warn is the most actionable category.
    """
    if str(fm.get("severity", "")).strip() == "block":
        return "W"
    try:
        rel = path.resolve().relative_to(knowledge_dir.resolve())
    except ValueError:
        return ""
    rel_str = str(rel).replace(os.sep, "/").lower()
    for segment, marker in _KIND_SEGMENTS:
        if segment in rel_str:
            return marker
    return ""


def _render_bullet(
    path: Path,
    reinforced: int,
    knowledge_dir: Path,
    *,
    title_only: bool = False,
    include_body: bool = False,
    today: Optional[date] = None,
) -> str:
    """Render one bullet line. ``title_only`` drops the applies-when
    hook and any body excerpt (used as a fallback when the budget is
    tight) while keeping the title so the agent always has something
    semantic to anchor on. ``include_body`` inlines a short body
    excerpt under the bullet — used for top entries that haven't been
    surfaced this session yet.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    fm = _parse_frontmatter(text)
    link = _wikilink_path(path, knowledge_dir)

    count_suffix = f" (×{reinforced})" if reinforced > 0 else ""
    fresh_marker = f" {_FRESHNESS_MARKER}" if _is_fresh(fm, today=today) else ""
    kind = _kind_marker(path, knowledge_dir, fm)
    kind_marker = f" [{kind}]" if kind else ""
    markers = f"{count_suffix}{fresh_marker}{kind_marker}"
    title = _entry_title(fm, path)

    if title_only:
        if not title:
            return f"- [[{link}]]{markers}"
        return f"- [[{link}]]{markers} — {title}"
    hook = _entry_applies_when(fm)
    if title and hook:
        bullet = f"- [[{link}]]{markers} — {title} — {_shorten(hook, max_chars=70)}"
    elif title:
        bullet = f"- [[{link}]]{markers} — {title}"
    elif hook:
        bullet = f"- [[{link}]]{markers} — {_shorten(hook)}"
    else:
        bullet = f"- [[{link}]]{markers}"

    if include_body:
        excerpt = _extract_body_excerpt(text)
        if excerpt:
            bullet += f"\n  > {excerpt}"
    return bullet


def _filter_already_sent(
    ranked: List[Tuple[Path, float, int]],
    knowledge_dir: Path,
    already_sent: Set[str],
) -> List[Tuple[Path, float, int]]:
    """Drop entries already surfaced to this session. Preserves rank order."""
    out: List[Tuple[Path, float, int]] = []
    for path, score, reinforced in ranked:
        if _wikilink_path(path, knowledge_dir) in already_sent:
            continue
        out.append((path, score, reinforced))
    return out


def _render_at_verbosity(
    sliced: List[Tuple[Path, float, int]],
    knowledge_dir: Path,
    *,
    include_body: bool,
    title_only: bool,
    today: Optional[date],
) -> List[str]:
    """Render the chosen entries at the given verbosity. Drops empty bullets."""
    return [
        b
        for b in (
            _render_bullet(
                p,
                r,
                knowledge_dir,
                include_body=include_body,
                title_only=title_only,
                today=today,
            )
            for p, _score, r in sliced
        )
        if b
    ]


def _wrap_bullets(bullets: List[str]) -> str:
    """Standard header + bullets + footer envelope."""
    return f"{_HEADER}\n\n" + "\n".join(bullets) + f"\n\n{_FOOTER}\n"


def _assemble_output(
    ranked: List[Tuple[Path, float, int]],
    knowledge_dir: Path,
    *,
    top_k: int,
    char_budget: int,
    already_sent: Optional[Set[str]] = None,
    today: Optional[date] = None,
) -> Tuple[str, List[str]]:
    """Compose the hot-context body, respecting the char budget.

    Args:
      ranked: scored candidates from the rerank+boost pipeline.
      knowledge_dir: library root (for wikilink rendering).
      top_k: max entries to render (3 in production — the agent
        explicitly asked for fewer-and-richer over more-and-shallower).
      char_budget: hard upper bound on output length.
      already_sent: wikilink paths the daemon has surfaced to this
        session before. Filtered out so the agent doesn't see the same
        priming over and over. Pass an empty set to disable dedup.
      today: override for the freshness-marker date check; tests pin it.

    Returns:
      ``(rendered_markdown, newly_sent_wikilinks)``. The markdown is
      "" when every candidate was already-sent (no new content to
      surface — the caller treats this as "nothing to inject"). The
      second element is the list of wikilinks the caller should add
      to the session's seen-set after this turn.

    Strategy: pick a verbosity level that fits the char budget, in
    decreasing order — full (with body excerpts) → hooked (titles +
    applies-when) → terse (titles only) → terse-truncated. Each
    verbosity is rendered by ``_render_at_verbosity``; the first one
    that fits wins.
    """
    if not ranked:
        return "", []

    filtered = _filter_already_sent(ranked, knowledge_dir, already_sent or set())
    if not filtered:
        # Every top candidate was already shown to this session.
        return "", []

    sliced = filtered[:top_k]
    newly_sent: List[str] = [_wikilink_path(p, knowledge_dir) for p, _score, _r in sliced]

    # Render at each verbosity in decreasing richness; return the first
    # that fits the char budget.
    verbosity_attempts = (
        (True, False),  # full bodies — the new richer shape
        (False, False),  # drop bodies, keep applies-when hooks
        (False, True),  # title-only
    )
    last_bullets: List[str] = []
    for include_body, title_only in verbosity_attempts:
        bullets = _render_at_verbosity(
            sliced,
            knowledge_dir,
            include_body=include_body,
            title_only=title_only,
            today=today,
        )
        if not bullets:
            continue
        last_bullets = bullets
        candidate = _wrap_bullets(bullets)
        if len(candidate) <= char_budget:
            return candidate, newly_sent

    if not last_bullets:
        return "", []

    # Last resort: drop trailing bullets from the terse render until we
    # fit. Keep at least one — an empty list defeats the purpose.
    truncated_sent = list(newly_sent)
    while len(last_bullets) > 1:
        last_bullets.pop()
        truncated_sent.pop()
        candidate = _wrap_bullets(last_bullets)
        if len(candidate) <= char_budget:
            return candidate, truncated_sent

    # Final fallback: the single best bullet only, even if over budget.
    return _wrap_bullets(last_bullets), truncated_sent


# ── Public API ────────────────────────────────────────────────────────


def refresh_hot_context(
    knowledge_dir: Path,
    rolling_buffer_text: str,
    out_path: Path,
    *,
    top_k: int = 3,
    char_budget: int = 1500,
    current_project_slug: Optional[str] = None,
    already_sent: Optional[Set[str]] = None,
) -> Optional[Path]:
    """Recompute the hot-context file from the recent buffer.

    Args:
        knowledge_dir: root of the library to search.
        rolling_buffer_text: plain-text proxy of recent session activity.
            Empty string is OK — the function clears the hot-context
            file in that case (no stale priming).
        out_path: where to write the new hot-context body.
        top_k: how many entries to include before budget trimming
            (default 3 — fewer-and-richer, with body excerpts).
        char_budget: hard upper bound on output length.
        already_sent: wikilinks already surfaced to this session. Used
            for dedup so the agent doesn't see the same priming over
            and over. Pass ``None`` or an empty set to disable.

    Returns:
        The path written, or ``None`` if nothing was written (empty
        buffer with no existing file, or every step failed).

    Never raises. All failures are logged at WARN/EXCEPTION and the
    function returns; the daemon must not crash because BM25 hiccupped.

    Note: this function discards the ``newly_sent`` list from
    ``_assemble_output`` — it predates the dedup feature and has no
    session context. The RPC handler (``priming_rpc._handle_priming``)
    is the path that actually maintains session state. Keep this
    function in sync with the RPC's call shape so the file-write
    fallback (when the daemon isn't fronting an RPC call) produces
    the same content the RPC would.
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
        body, _newly_sent = _assemble_output(
            ranked,
            knowledge_dir,
            top_k=top_k,
            char_budget=char_budget,
            already_sent=already_sent,
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
