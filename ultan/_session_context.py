"""SessionStart boot-context for the `ultan` plugin (G5).

Claude Code's SessionStart hook can inject an ``additionalContext`` block that
the agent reads once at the top of every session. The legacy
``src/hooks/session_start.py`` used this to "boot" the agent's memory: today's
date + current project, the knowledge ``index.md``, and the last lines of the
day's daily log.

The daemon model dropped the ``daily/`` log entirely (it was the legacy flush's
output). So a literal port is impossible. This module rebuilds the *equivalent*
against the daemon's ``knowledge/`` layout:

  1. **Date + project.** Today's date, the cwd-derived project slug, and the
     library bucket the session maps to (so the agent knows which
     ``knowledge/projects/<bucket>/`` lessons apply most directly).
  2. **Knowledge Base Index excerpt.** The HEAD of ``knowledge/index.md`` — the
     curator's catalog of what the library knows. The live index is ~225 KB, so
     we take a bounded head, not the whole file.
  3. **Recent activity.** The newest entries of ``knowledge/log.md`` (the
     curator's "Scholar Action Log", written newest-first). This is the daemon
     equivalent of "what happened lately" now that ``daily/`` is gone — it shows
     which entries were just compiled / updated / vetoed. When ``log.md`` is
     absent we degrade to the most-recently-modified entries under
     ``knowledge/`` so a freshly-seeded store still shows signs of life.

Design constraints (this lands at EVERY session start):

* **Stdlib-only and self-contained.** The thin ``ultan`` wheel ships only the
  ``ultan/`` package — ``aliases`` / ``scope`` (in ``tools/search`` and
  ``src/``) are NOT on its import path, and ``ultan/_priming.py`` already sets
  the precedent of mirroring cross-package logic locally rather than importing
  across the packaging boundary. So the slug + bucket derivation here MIRRORS
  ``scope.current_project_slug`` (git-remote → basename → slugify) and
  ``aliases.session_bucket`` (steps 1-2: existing-bucket match, else
  repo-root/basename candidate) rather than calling them.
* **Read-only.** Unlike ``aliases.session_bucket``, this never writes the alias
  file. SessionStart should observe, not mutate; Phase 2's ``_session_start``
  already restarts/ensures the daemon and the daemon owns alias bootstrap.
* **Capped.** The whole block is hard-capped (``MAX_CONTEXT_CHARS``) and each
  section is independently bounded so one runaway file can't crowd out the
  others.

The public entry point is :func:`build_session_start_context`. It never raises
— a SessionStart hook must not crash the host — and returns ``""`` when the
store is empty / unreadable, which the caller treats as "nothing to inject".
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ._aliases import load_aliases
from ._daemon import _home  # pyright: ignore[reportPrivateUsage]  # intra-package

# ── Caps ─────────────────────────────────────────────────────────────────────

# Hard ceiling for the whole injected block. Mirrors the legacy
# session_start.py budget. This is paid at every session start, so keep it lean.
MAX_CONTEXT_CHARS = 20_000

# Index excerpt: the live index.md is ~225 KB / ~400 rows. We want the catalog
# header + the freshest rows, not the whole table — the agent can search the
# library for the rest. Bounded by both lines and chars so a pathological row
# can't blow the budget.
_MAX_INDEX_LINES = 60
_MAX_INDEX_CHARS = 8_000

# Recent-activity excerpt from log.md (newest-first). A handful of the newest
# Scholar entries is enough to show "what the library just learned".
_MAX_LOG_ENTRIES = 5
_MAX_LOG_CHARS = 6_000

# Fallback recent-activity scan (no log.md): how many recently-modified entries
# to list, and how deep to walk before giving up on a huge tree.
_MAX_RECENT_ENTRIES = 8
_MAX_SCAN_FILES = 2_000

# Skip these catalog/non-entry files when listing "recent entries" — same
# spirit as priming's _iter_markdown (index.md / log.md / READMEs are catalogs,
# not lessons).
_CATALOG_NAMES = {"index.md", "log.md", "README.md"}

# ── Slug derivation (mirrors src/scripts/scope.py) ───────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
_GIT_URL_RE = re.compile(
    r"""^
    (?:[a-z]+://)?                  # optional scheme
    (?:[^@/]+@)?                    # optional user@
    (?P<host>[^/:]+)                # host
    [/:]                           # separator
    (?P<path>[^\s]+?)               # path (non-greedy)
    (?:\.git)?$                     # optional .git
    """,
    re.VERBOSE,
)


def _slugify(text: str) -> str:
    """Lowercase, collapse non-slug runs to a dash. Mirrors ``scope._slugify``."""
    text = text.strip().lower()
    text = _SLUG_RE.sub("-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def _git_remote_slug(cwd: Path) -> Optional[str]:
    """``host/owner/repo`` slug from the git remote, or ``None``. Silent on
    every failure (not a repo, no remote, git missing, timeout)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    if not url:
        return None
    m = _GIT_URL_RE.match(url)
    if not m:
        return _slugify(url)
    return _slugify(f"{m.group('host')}/{m.group('path')}")


def _project_slug(cwd: Path) -> str:
    """Best-effort project slug for ``cwd``. Mirrors ``scope.current_project_slug``:
    git remote → cwd basename → ``"unknown"``."""
    slug = _git_remote_slug(cwd)
    if slug:
        return slug
    base = cwd.name
    return _slugify(base) if base else "unknown"


# ── Bucket resolution (read-only mirror of aliases.session_bucket) ───────────


def _find_repo_root(cwd: Path) -> Optional[Path]:
    """Walk up from ``cwd`` to the ``.git`` root. Mirrors ``aliases._find_repo_root``."""
    try:
        cur = cwd.resolve()
    except (OSError, RuntimeError):
        return None
    while True:
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def _session_bucket(home: Path, cwd: Path, slug: str) -> Optional[str]:
    """Which ``knowledge/projects/<bucket>/`` does this session map to?

    Read-only mirror of ``aliases.session_bucket`` steps 1-2 (the resolution
    half) — it deliberately OMITS step 3 (auto-writing the alias file), because
    SessionStart should observe, not mutate. Returns the bucket name or
    ``None`` when ``slug`` is empty.
    """
    if not slug:
        return None

    aliases = load_aliases(home)
    projects_dir = home / "knowledge" / "projects"

    # Step 1: an existing bucket whose canonical slug already resolves to ours.
    if projects_dir.is_dir():
        try:
            entries = sorted(projects_dir.iterdir())
        except OSError:
            entries = []
        for bucket_dir in entries:
            if not bucket_dir.is_dir():
                continue
            if aliases.get(bucket_dir.name, bucket_dir.name) == slug:
                return bucket_dir.name

    # Step 2: derive a candidate from cwd — repo root if in a git tree, else the
    # cwd basename (matches the no-git slug == basename rule).
    repo_root = _find_repo_root(cwd)
    candidate_path = repo_root if repo_root is not None else cwd
    try:
        candidate = candidate_path.resolve().name
    except (OSError, RuntimeError):
        candidate = candidate_path.name
    return candidate or None


# ── Section builders ─────────────────────────────────────────────────────────


def _cap(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` with a visible marker."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n…(truncated)"


def _header_section(cwd: Path, home: Path) -> str:
    """Date + project slug + library bucket."""
    today = datetime.now(timezone.utc).astimezone()
    slug = _project_slug(cwd)
    bucket = _session_bucket(home, cwd, slug)

    lines = [
        "## Today",
        today.strftime("%A, %B %d, %Y"),
        "",
        "## Current project",
        f"`{slug}`",
    ]
    if bucket:
        lines.append(
            f"Lessons under `knowledge/projects/{bucket}/` apply most directly "
            "(plus everything under `knowledge/global/`)."
        )
    else:
        lines.append("No project bucket resolved; `knowledge/global/` lessons still apply.")
    return "\n".join(lines)


def _index_section(home: Path) -> Optional[str]:
    """HEAD excerpt of ``knowledge/index.md`` (the curator's catalog)."""
    index_file = home / "knowledge" / "index.md"
    try:
        text = index_file.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    lines = text.splitlines()
    head = "\n".join(lines[:_MAX_INDEX_LINES])
    excerpt = _cap(head, _MAX_INDEX_CHARS)
    if not excerpt.strip():
        return None
    truncated = len(lines) > _MAX_INDEX_LINES
    note = (
        "\n\n*(index truncated — search the library for entries beyond the head)*"
        if truncated
        else ""
    )
    return f"## Knowledge Base Index\n\n{excerpt}{note}"


def _recent_log_entries(text: str) -> str:
    """Take the newest ``_MAX_LOG_ENTRIES`` ``## [..]`` blocks from log.md.

    The Scholar Action Log is written newest-first, so the freshest entries are
    at the HEAD. We slice on the ``## [`` block boundary rather than by line
    count so each entry stays whole.
    """
    lines = text.splitlines()
    # Find the start of the (N+1)th entry; everything before it is the newest N.
    seen = 0
    cut = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("## ["):
            seen += 1
            if seen > _MAX_LOG_ENTRIES:
                cut = i
                break
    return "\n".join(lines[:cut]).rstrip()


def _activity_section(home: Path) -> Optional[str]:
    """Recent-activity recap. Prefer ``log.md`` (curator-maintained, newest-first);
    fall back to recently-modified ``knowledge/`` entries when it's absent."""
    log_file = home / "knowledge" / "log.md"
    try:
        text = log_file.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        text = ""

    if text.strip():
        excerpt = _cap(_recent_log_entries(text), _MAX_LOG_CHARS)
        if excerpt.strip():
            return f"## Recent Library Activity\n\n{excerpt}"

    # Fallback: no log.md (e.g. a freshly-seeded store) — list the most
    # recently-modified entries so the agent still sees signs of life.
    recent = _recent_entries(home)
    if recent:
        body = "\n".join(f"- `{rel}`" for rel in recent)
        return f"## Recent Library Activity\n\nMost recently updated entries:\n{body}"
    return None


def _recent_entries(home: Path) -> List[str]:
    """The most-recently-modified knowledge entries, as paths relative to
    ``knowledge/``. Skips catalogs (index/log/README) and ``_archive``."""
    kdir = home / "knowledge"
    if not kdir.is_dir():
        return []
    scored: List[tuple[float, str]] = []
    scanned = 0
    try:
        for p in kdir.rglob("*.md"):
            scanned += 1
            if scanned > _MAX_SCAN_FILES:
                break
            if "_archive" in p.parts or p.name in _CATALOG_NAMES:
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            rel = p.relative_to(kdir).as_posix()
            scored.append((mtime, rel))
    except OSError:
        return []
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [rel for _mtime, rel in scored[:_MAX_RECENT_ENTRIES]]


# ── Public API ───────────────────────────────────────────────────────────────


def build_session_start_context(cwd: Optional[str]) -> str:
    """Build the SessionStart ``additionalContext`` markdown for ``cwd``.

    Returns a capped (``MAX_CONTEXT_CHARS``) markdown block with three sections —
    date+project, knowledge-index head, recent library activity — or ``""`` when
    the store is empty / unreadable (nothing worth injecting).

    ``cwd`` is the agent's working directory from the SessionStart hook payload;
    ``None`` falls back to the process cwd (the hook should always pass it, but
    we never want to crash if it doesn't). Never raises.
    """
    try:
        home = _home()
        cwd_path = Path(cwd).expanduser() if cwd else Path.cwd()

        sections: List[str] = [_header_section(cwd_path, home)]

        index = _index_section(home)
        activity = _activity_section(home)

        # If the store has neither an index nor any activity, there's nothing
        # substantive to inject — the bare date/project header alone isn't worth
        # the context budget, so degrade to "".
        if index is None and activity is None:
            return ""

        if index is not None:
            sections.append(index)
        if activity is not None:
            sections.append(activity)

        context = "\n\n---\n\n".join(sections)
        return _cap(context, MAX_CONTEXT_CHARS)
    except Exception:
        # A SessionStart hook must NEVER crash the host. Any unforeseen error
        # degrades to "no context injected".
        return ""
