"""BM25 indexer + searcher for the agent-mem knowledge store.

Tokenization rules (v1, see README for rationale):
  - YAML frontmatter is stripped entirely BEFORE tokenization, except:
      * `title:` is appended to the body text (high-signal; some titles phrase
        the claim differently from the body `# H1`).
      * `keywords:` values (list or inline) are appended to the body text.
      * `applies-when:` values (block scalar or inline) are appended to the body.
    `keywords`/`applies-when` are search-relevant per PLAN section 2; `title`
    was added later (see `_frontmatter_search_text`).
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

import hashlib
import os
import pickle
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Protocol, Sequence, TypeVar, cast

import yaml
from rank_bm25 import BM25Okapi

# ── Type model ─────────────────────────────────────────────────────────────────

# Frontmatter values can be anything YAML emits (str, int, list, dict, None);
# ``object`` is the only honest annotation at the boundary. Callers narrow
# with isinstance.
FrontmatterDict = dict[str, object]


class _BM25Backend(Protocol):
    """Minimal slice of ``rank_bm25.BM25Okapi`` we actually use.

    The library has no py.typed marker, so pyright sees its methods as
    Unknown. We pin the shape locally and cast at the boundary.
    """

    def get_scores(self, query: Sequence[str]) -> "Sequence[float]": ...


# ── Tokenization ───────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _strip_and_extract_frontmatter(text: str) -> tuple[str, FrontmatterDict]:
    """Return (body_without_frontmatter, parsed_frontmatter_dict)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text, {}
    try:
        loaded: object = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return text[m.end() :], {}
    if not isinstance(loaded, dict):
        return text[m.end() :], {}
    fm: FrontmatterDict = {str(k): v for k, v in loaded.items()}  # type: ignore[misc]
    body = text[m.end() :]
    return body, fm


def _frontmatter_search_text(fm: Mapping[str, object]) -> str:
    """Pull out the search-relevant frontmatter fields: ``title``, ``keywords``
    and ``applies-when``.

    ``title`` is included (beyond PLAN section 2's original keywords/applies-when)
    so titles are first-class search signal: most titles are mirrored by the body
    ``# H1`` and would be found anyway, but a real minority phrase the claim
    differently in the title than the H1, and those terms are only searchable if
    the title is indexed. A side effect makes the contract clean: because this is
    the single text source shared by BM25, the embedding index AND
    :func:`index_content_hash`, indexing the title also means a re-title changes
    the content hash and correctly triggers a reindex."""
    parts: list[str] = []

    title = fm.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title)

    keywords = fm.get("keywords")
    if isinstance(keywords, list):
        keywords_list = cast(list[object], keywords)
        parts.extend(str(k) for k in keywords_list)
    elif isinstance(keywords, str):
        parts.append(keywords)

    applies_when = fm.get("applies-when") or fm.get("applies_when")
    if isinstance(applies_when, list):
        applies_when_list = cast(list[object], applies_when)
        parts.extend(str(a) for a in applies_when_list)
    elif isinstance(applies_when, str):
        parts.append(applies_when)

    return "\n".join(parts)


def tokenize(text: str) -> list[str]:
    """Tokenize a full markdown document for BM25.

    Strips YAML frontmatter from the body, but pulls `title`, `keywords` and
    `applies-when` into the searchable text first.
    """
    body, fm = _strip_and_extract_frontmatter(text)
    fm_text = _frontmatter_search_text(fm)
    combined = (fm_text + "\n" + body).lower()
    return [tok for tok in _TOKEN_SPLIT_RE.split(combined) if len(tok) >= 2]


def index_content_hash(raw: str) -> str:
    """Stable hash of the text that actually feeds the indexes.

    Both BM25 (:func:`tokenize`) and the embedding index
    (``embeddings._embedding_text``) index the body plus the ``title``,
    ``keywords`` and ``applies-when`` frontmatter — and nothing else (all three
    derive their text from :func:`_frontmatter_search_text`, so this hash covers
    exactly what is searched, no more). Mutable bookkeeping fields (``fired``,
    ``last_surfaced``, ``reconsolidated``, …) are stripped before indexing, so a
    write that only touches them leaves every token and every embedding
    byte-identical. Keying staleness on this hash instead of the file mtime means
    such writes no longer trigger a needless re-tokenise / re-encode on the
    priming hot path — which is what made every priming surface (it bumps
    ``fired``/``last_surfaced``) re-index the corpus. A re-title, by contrast,
    DOES change this hash and so reindexes — correct, since the title is searched.
    """
    body, fm = _strip_and_extract_frontmatter(raw)
    indexed_text = _frontmatter_search_text(fm) + "\n" + body
    return hashlib.blake2b(indexed_text.encode("utf-8"), digest_size=16).hexdigest()


# ── Public aliases for cross-module use ────────────────────────────────────────
# ``embeddings.py`` reuses the same file-selection and frontmatter discipline.
# We expose non-private aliases so it can import them without tripping
# ``reportPrivateUsage``. Tests continue to import the underscore-prefixed
# originals (back-compat).
# (Defined after the helpers themselves further down so they reference final
# objects.)


# ── Index dataclass ────────────────────────────────────────────────────────────


@dataclass
class _DocRecord:
    """Minimal record we persist for each indexed file."""

    path: str  # absolute path string
    mtime: float
    raw_text: str  # full file text, kept for snippet generation
    # blake2b of the indexed text (body + title + keywords + applies-when). Lets
    # the staleness check tell a real content edit from a bookkeeping-only mtime
    # bump. Defaults to "" so indexes pickled before this field existed still
    # unpickle — a missing hash never matches, so the first staleness check
    # after an upgrade falls through to a rebuild that repopulates it.
    content_hash: str = ""


class BM25Hit(NamedTuple):
    """A single BM25 search result.

    Destructures as ``(path, score, snippet)`` for callers that already
    unpack tuples (the daemon's priming code does this). Adding a typed
    NamedTuple gives static guarantees without breaking those call sites.
    """

    path: Path
    score: float
    snippet: str


@dataclass
class BM25Index:
    """BM25 index over markdown bodies."""

    knowledge_dir: Path
    docs: list[_DocRecord] = field(default_factory=list[_DocRecord])
    tokenized: list[list[str]] = field(default_factory=list[list[str]])
    bm25: BM25Okapi | None = None
    built_at: float = 0.0

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[BM25Hit]:
        """Top-k hits as (path, score, one-line snippet).

        Returns at most k results, ordered by descending score. Filters out
        zero-score hits — BM25Okapi otherwise pads results with score=0 docs.
        """
        if self.bm25 is None or not self.docs:
            return []
        q_tokens = [tok for tok in _TOKEN_SPLIT_RE.split(query.lower()) if len(tok) >= 2]
        if not q_tokens:
            return []
        backend = cast(_BM25Backend, self.bm25)
        raw_scores = backend.get_scores(q_tokens)
        scores: list[float] = [float(s) for s in raw_scores]
        ranked: list[tuple[int, float]] = sorted(
            enumerate(scores),
            key=lambda iv: iv[1],
            reverse=True,
        )
        results: list[BM25Hit] = []
        for idx, score in ranked[:k]:
            if score <= 0:
                continue
            rec = self.docs[idx]
            snippet = _build_snippet(rec.raw_text, q_tokens)
            results.append(BM25Hit(path=Path(rec.path), score=float(score), snippet=snippet))
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


# Public aliases so ``embeddings.py`` (a sibling module in this package) can
# reuse the helpers without tripping pyright's reportPrivateUsage warning.
# Tests still import the underscore-prefixed originals.
build_snippet = _build_snippet
frontmatter_search_text = _frontmatter_search_text
iter_markdown = _iter_markdown
strip_and_extract_frontmatter = _strip_and_extract_frontmatter


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
        docs.append(
            _DocRecord(
                path=str(md),
                mtime=md.stat().st_mtime,
                raw_text=raw,
                content_hash=index_content_hash(raw),
            )
        )
        tokenized.append(toks)

    bm25 = BM25Okapi(tokenized) if tokenized else None
    return BM25Index(
        knowledge_dir=knowledge_dir,
        docs=docs,
        tokenized=tokenized,
        bm25=bm25,
        built_at=time.time(),
    )


def save_pickled(obj: object, target: Path) -> Path:
    """Atomically pickle ``obj`` to ``target``. Returns the path written.

    Concurrency-safe: each writer pickles to its OWN uniquely-named temp file
    (``tempfile.mkstemp`` in the destination directory) before a single
    ``os.replace``. Two threads or processes saving the same index therefore
    never interleave their byte streams into one shared ``*.tmp`` file — the
    failure mode that produced a half-written, corrupt index when the daemon's
    parallel Librarian threads rebuilt a stale index at once. ``os.replace``
    is atomic, so readers see either the old file or a complete new one, never
    a partial write. Last writer wins; every file on disk is internally
    complete. Shared by the BM25 and embedding indices.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, target)
    except BaseException:
        # Never leave an orphaned temp file behind on a failed write.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def save_index(index: BM25Index, index_path: Path | None = None) -> Path:
    """Pickle the index. Returns the path it was written to."""
    target = index_path or _default_index_path(index.knowledge_dir)
    return save_pickled(index, target)


_PickledT = TypeVar("_PickledT")


def load_pickled(index_path: Path, expected_type: type[_PickledT]) -> _PickledT | None:
    """Unpickle ``index_path`` and return it only if it is ``expected_type``.

    Any failure to read the cache yields None so the caller rebuilds from
    scratch. The ``except`` is deliberately broad: a persisted index is a
    disposable cache, and corruption surfaces in many guises beyond
    ``pickle.UnpicklingError`` — a mangled stream misread as a string raises
    ``UnicodeDecodeError``/``ValueError``, a bogus length prefix raises
    ``MemoryError``, a truncated file raises ``EOFError``, schema drift raises
    ``AttributeError``/``ModuleNotFoundError``. The only correct response to an
    unreadable cache is to discard and rebuild it, so we catch ``Exception``
    rather than enumerate an inevitably-incomplete list. ``BaseException``
    (``KeyboardInterrupt``, ``SystemExit``) still propagates. Shared by the
    BM25 and embedding indices.
    """
    try:
        with index_path.open("rb") as f:
            obj: object = pickle.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, expected_type) else None


class _TimestampedDoc(Protocol):
    path: str
    mtime: float


def is_stale(docs: Sequence[_TimestampedDoc], current_files: Mapping[str, float]) -> bool:
    """True if the tracked doc set differs from what's on disk, or any current
    file is newer than when it was indexed.

    ``current_files`` maps absolute-path string → mtime; the caller builds it
    from its own markdown walk. Shared by the BM25 and embedding indices.
    """
    tracked = {rec.path: rec.mtime for rec in docs}
    if set(current_files) != set(tracked):
        return True
    for path, mtime in current_files.items():
        if mtime > tracked[path] + 1e-6:
            return True
    return False


def only_bookkeeping_changed(
    docs: Sequence[_TimestampedDoc],
    current_files: Mapping[str, float],
) -> bool:
    """True iff every change since the index was built is confined to non-indexed
    (bookkeeping/title) frontmatter — so a rebuild would be byte-identical and the
    cached index can be reused as-is.

    Call only after :func:`is_stale` returned True. Returns False (→ caller must
    rebuild) if any file was added or removed, if a changed file can't be read,
    or if a changed file's indexed-content hash differs from its cached record.
    Only files whose mtime advanced are read; the unchanged majority is skipped
    via the same cheap mtime comparison ``is_stale`` uses. A cached record with no
    stored ``content_hash`` (index pickled before that field existed) never
    matches, so the first post-upgrade staleness check falls through to a rebuild
    that repopulates the hashes. Shared by the BM25 and embedding indices.
    """
    tracked: dict[str, _TimestampedDoc] = {rec.path: rec for rec in docs}
    if set(current_files) != set(tracked):
        return False
    for path, mtime in current_files.items():
        rec = tracked[path]
        if mtime <= rec.mtime + 1e-6:
            continue  # unchanged — no need to read the file
        cached_hash = getattr(rec, "content_hash", "")
        if not cached_hash:
            return False  # pre-upgrade record (or genuinely empty) → rebuild
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        if index_content_hash(raw) != cached_hash:
            return False  # indexed content really changed
    return True


def load_or_build(
    knowledge_dir: Path,
    index_path: Path | None = None,
    force_rebuild: bool = False,
) -> BM25Index:
    """Load the persisted index if fresh, otherwise rebuild and save."""
    knowledge_dir = knowledge_dir.expanduser().resolve()
    target = index_path or _default_index_path(knowledge_dir)

    if not force_rebuild and target.exists():
        cached = load_pickled(target, BM25Index)
        if cached is not None and cached.knowledge_dir == knowledge_dir:
            current_files = {str(p): p.stat().st_mtime for p in _iter_markdown(knowledge_dir)}
            if not is_stale(cached.docs, current_files):
                return cached
            # Stale by mtime, but if the only writes were bookkeeping/title
            # frontmatter the indexed text is unchanged — reuse without a
            # rebuild. (We don't re-save: stored mtimes stay behind, so the
            # next request re-reads the same handful and confirms again. That
            # read is sub-millisecond; a full re-tokenise + BM25Okapi rebuild
            # on every priming surface is what we're avoiding.)
            if only_bookkeeping_changed(cached.docs, current_files):
                return cached

    index = build_index(knowledge_dir)
    try:
        save_index(index, target)
    except OSError:
        # Persisting is best-effort; an in-memory index is still useful.
        pass
    return index
