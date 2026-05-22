"""Project scoping — derive a stable slug from a working directory.

The slug is what tags lessons with the project they came from. PLAN.md §7.4
pins down the rule: prefer the git remote URL (normalised), fall back to the
basename of the working tree. Stable across machines for the same repo,
distinct across forks/clones, never empty.

The hook passes its `cwd` field straight in. Most other call sites should
use :func:`current_project_slug` with no args, which reads `os.getcwd()`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Union

# ── Slug normaliser ────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def _slugify(text: str) -> str:
    """Lowercase, collapse runs of non-slug chars to a single dash.

    Kept loose on purpose: ``github.com/foo/bar`` should survive as
    ``github.com-foo-bar`` rather than being flattened to nothing.
    """
    text = text.strip().lower()
    text = _SLUG_RE.sub("-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


# ── Git remote extraction ──────────────────────────────────────────────

# Matches the host+path part of a git remote URL, regardless of transport:
#   git@github.com:user/repo.git    -> github.com/user/repo
#   https://github.com/user/repo    -> github.com/user/repo
#   ssh://git@github.com/user/repo  -> github.com/user/repo
_GIT_URL_RE = re.compile(
    r"""^
    (?:[a-z]+://)?                  # optional scheme
    (?:[^@/]+@)?                    # optional user@
    (?P<host>[^/:]+)                # host
    [/:]                            # separator
    (?P<path>[^\s]+?)               # path (non-greedy)
    (?:\.git)?$                     # optional .git
    """,
    re.VERBOSE,
)


def _git_remote_slug(cwd: Path) -> Optional[str]:
    """Return ``host/owner/repo``-shaped slug from the git remote, or None.

    Silent on every failure: not-a-repo, no remote configured, git missing,
    timeout. Falls back to the basename path instead.
    """
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

    raw = f"{m.group('host')}/{m.group('path')}"
    return _slugify(raw)


# ── Public API ─────────────────────────────────────────────────────────


def current_project_slug(cwd: Optional[Union[str, "os.PathLike[str]"]] = None) -> str:
    """Best-effort project slug for the given (or current) working directory.

    Order of preference:

    1. Git remote URL (``remote.origin.url``), normalised to ``host-owner-repo``.
    2. Basename of the working directory.
    3. ``"unknown"`` as a last resort (e.g. cwd is the filesystem root).

    Hooks receive the agent's cwd via stdin (`hook_input["cwd"]`) and should
    pass it in explicitly — the hook process's own cwd is unrelated.
    """
    path = Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()

    slug = _git_remote_slug(path)
    if slug:
        return slug

    base = path.name
    if base:
        return _slugify(base)
    return "unknown"
