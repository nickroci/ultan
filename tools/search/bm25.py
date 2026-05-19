"""BM25 indexer + searcher for the agent-mem knowledge store.

Tokenization rules (v1, see README for rationale):
  - YAML frontmatter is stripped entirely BEFORE tokenization, except:
      * `keywords:` values (list or inline) are appended to the body text.
      * `applies-when:` values (block scalar or inline) are appended to the body.
    Those two fields are explicitly search-relevant per PLAN section 2.
  - Code fences are kept verbatim. Code identifiers are valuable search terms.
  - Markdown is otherwise treated as plain text: we don't strip headings,
    wikilinks, list markers, etc. (BM25 handles the noise; over-stripping
    risks dropping signal.)
  - Lowercase everything.
  - Split on non-alphanumeric (regex `[^a-z0-9]+`).
  - Keep tokens of length >= 2.

Index persistence:
  - Pickled `BM25Index` at `<knowledge_dir>/../.bm25.idx` by default
    (i.e. `~/.agent-mem/.bm25.idx` when knowledge_dir is `~/.agent-mem/knowledge`).
  - Rebuild when any source `.md` file has mtime > index file mtime, or when
    a tracked file has been removed, or when a new file has appeared.
"""

from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml
from rank_bm25 import BM25Okapi

# ── Tokenization ───────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _strip_and_extract_frontmatter(text: str) -> tuple[str, dict]:
    """Return (body_without_frontmatter, parsed_frontmatter_dict)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text, {}
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    body = text[m.end():]
    return body, fm


def _frontmatter_search_text(fm: dict) -> str:
    """Pull out the search-relevant frontmatter fields per PLAN section 2."""
    parts: list[str] = []

    keywords = fm.get("keywords")
    if isinstance(keywords, list):
        parts.extend(str(k) for k in keywords)
    elif isinstance(keywords, str):
        parts.append(keywords)

    applies_when = fm.get("applies-when") or fm.get("applies_when")
    if isinstance(applies_when, list):
        parts.extend(str(a) for a in applies_when)
    elif isinstance(applies_when, str):
        parts.append(applies_when)

    return "\n".join(parts)


def tokenize(text: str) -> list[str]:
    """Tokenize a full markdown document for BM25.

    Strips YAML frontmatter from the body, but pulls `keywords` and
    `applies-when` into the searchable text first.
    """
    body, fm = _strip_and_extract_frontmatter(text)
    fm_text = _frontmatter_search_text(fm)
    combined = (fm_text + "\n" + body).lower()
    return [tok for tok in _TOKEN_SPLIT_RE.split(combined) if len(tok) >= 2]


# ── Index dataclass ────────────────────────────────────────────────────────────


@dataclass
class _DocRecord:
    """Minimal record we persist for each indexed file."""

    path: str  # absolute path string
    mtime: float
    raw_text: str  # full file text, kept for snippet generation


@dataclass
class BM25Index:
    """BM25 index over markdown bodies."""

    knowledge_dir: Path
    docs: list[_DocRecord] = field(default_factory=list)
    tokenized: list[list[str]] = field(default_factory=list)
    bm25: BM25Okapi | None = None
    built_at: float = 0.0

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[Path, float, str]]:
        """Top-k hits as (path, score, one-line snippet).

        Returns at most k results, ordered by descending score. Filters out
        zero-score hits — BM25Okapi otherwise pads results with score=0 docs.
        """
        if self.bm25 is None or not self.docs:
            return []
        q_tokens = [tok for tok in _TOKEN_SPLIT_RE.split(query.lower()) if len(tok) >= 2]
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda iv: iv[1],
            reverse=True,
        )
        results: list[tuple[Path, float, str]] = []
        for idx, score in ranked[:k]:
            if score <= 0:
                continue
            rec = self.docs[idx]
            snippet = _build_snippet(rec.raw_text, q_tokens)
            results.append((Path(rec.path), float(score), snippet))
        return results


# ── Snippet ────────────────────────────────────────────────────────────────────


def _build_snippet(raw_text: str, q_tokens: Iterable[str], width: int = 140) -> str:
    """Return a one-line snippet showing query context, or the first body line.

    Strips frontmatter from the snippet source so we don't surface YAML.
    """
    body, _fm = _strip_and_extract_frontmatter(raw_text)
    lowered = body.lower()
    hit_pos = -1
    for tok in q_tokens:
        pos = lowered.find(tok)
        if pos != -1:
            hit_pos = pos
            break
    if hit_pos == -1:
        # Fall back to the first non-empty, non-heading line.
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return _one_line(stripped, width)
        return _one_line(body.strip(), width)

    start = max(0, hit_pos - width // 3)
    end = min(len(body), start + width)
    snippet = body[start:end]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(body) else ""
    return _one_line(prefix + snippet + suffix, width + 4)


def _one_line(s: str, width: int) -> str:
    s = " ".join(s.split())
    if len(s) > width:
        s = s[: width - 1] + "…"
    return s


# ── Index build / load ─────────────────────────────────────────────────────────


def _iter_markdown(knowledge_dir: Path) -> list[Path]:
    """All .md files under knowledge_dir that count as content entries.

    Skipped:
      - `_archive/` subtrees (PLAN section 6).
      - The top-level `index.md` and `log.md` — they are catalogs, not entries.
        Including them inflates term frequency for every concept name in the
        catalog, which collapses IDF for those terms (BM25Okapi clamps any
        term with df >= N/2 to zero IDF). The catalog is what `--index` mode
        consumes directly; BM25 should not see it.
    """
    files: list[Path] = []
    catalog_names = {"index.md", "log.md"}
    for p in sorted(knowledge_dir.rglob("*.md")):
        if "_archive" in p.parts:
            continue
        if p.parent == knowledge_dir and p.name in catalog_names:
            continue
        files.append(p)
    return files


def _default_index_path(knowledge_dir: Path) -> Path:
    """`~/.agent-mem/.bm25.idx` when knowledge_dir is `~/.agent-mem/knowledge`."""
    return knowledge_dir.parent / ".bm25.idx"


def build_index(knowledge_dir: Path) -> BM25Index:
    """Walk knowledge_dir, tokenize every .md file, build a fresh BM25 index."""
    knowledge_dir = knowledge_dir.expanduser().resolve()
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"knowledge_dir does not exist: {knowledge_dir}")

    docs: list[_DocRecord] = []
    tokenized: list[list[str]] = []
    for md in _iter_markdown(knowledge_dir):
        try:
            raw = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        toks = tokenize(raw)
        if not toks:
            continue
        docs.append(_DocRecord(path=str(md), mtime=md.stat().st_mtime, raw_text=raw))
        tokenized.append(toks)

    bm25 = BM25Okapi(tokenized) if tokenized else None
    return BM25Index(
        knowledge_dir=knowledge_dir,
        docs=docs,
        tokenized=tokenized,
        bm25=bm25,
        built_at=time.time(),
    )


def save_index(index: BM25Index, index_path: Path | None = None) -> Path:
    """Pickle the index. Returns the path it was written to."""
    target = index_path or _default_index_path(index.knowledge_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename — atomic enough for v1.
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)
    return target


def _load_pickled(index_path: Path) -> BM25Index | None:
    try:
        with index_path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, BM25Index):
            return obj
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError):
        return None
    return None


def _is_stale(index: BM25Index, knowledge_dir: Path) -> bool:
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
    index_path: Path | None = None,
    force_rebuild: bool = False,
) -> BM25Index:
    """Load the persisted index if fresh, otherwise rebuild and save."""
    knowledge_dir = knowledge_dir.expanduser().resolve()
    target = index_path or _default_index_path(knowledge_dir)

    if not force_rebuild and target.exists():
        cached = _load_pickled(target)
        if cached is not None and cached.knowledge_dir == knowledge_dir:
            if not _is_stale(cached, knowledge_dir):
                return cached

    index = build_index(knowledge_dir)
    try:
        save_index(index, target)
    except OSError:
        # Persisting is best-effort; an in-memory index is still useful.
        pass
    return index
