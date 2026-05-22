"""Project-bucket → canonical slug alias resolution.

Single source of truth for both the daemon's priming scope-boost and
the user-prompt-submit hook's cross-project nudge filter. The file
lives at ``<agent-mem-home>/project-aliases.json`` with the shape:

    {"<bucket>": "<canonical-slug>", ...}

Buckets not present in the file default to ``slug == bucket name``, so
a project where ``current_project_slug()`` (cwd basename when no git
remote) already matches the on-disk bucket needs no entry.

Pure-functional and stateless — callers pass their own home-dir Path
so daemon and hook can resolve ``AGENT_MEM_HOME`` independently.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, cast

ALIASES_FILENAME = "project-aliases.json"


def aliases_path(home: Path) -> Path:
    """Resolve the alias-file path beneath an agent-mem home dir."""
    return home / ALIASES_FILENAME


def load_aliases(home: Path) -> Dict[str, str]:
    """Read the alias map for the given home dir.

    Returns ``{}`` when the file is missing, unreadable, or doesn't
    decode as a JSON object with string-coercible values — alias
    resolution must never crash the caller. Loaded fresh on each call
    so users can edit the file without restarting anything.
    """
    try:
        text = aliases_path(home).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    try:
        data: object = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # ``json.loads`` types ``data`` as ``Any``; once narrowed via isinstance
    # to ``dict`` pyright drops to ``dict[Unknown, Unknown]``. Cast to a
    # concrete object-valued mapping so the comprehension below stays typed.
    items = cast(dict[object, object], data)
    out: Dict[str, str] = {}
    for k, v in items.items():
        if isinstance(v, (str, int)):
            out[str(k)] = str(v)
    return out


def bucket_canonical_slug(
    bucket: Optional[str],
    aliases: Mapping[str, str],
) -> Optional[str]:
    """Return the canonical project slug for a bucket directory.

    Looks up the bucket in the alias map; falls back to the bucket name
    itself when absent. ``None`` in, ``None`` out — keeps callers from
    having to guard their `bucket is None` cases.
    """
    if not bucket:
        return None
    return aliases.get(bucket, bucket)


def _find_repo_root(cwd: Path) -> Optional[Path]:
    """Walk up from ``cwd`` looking for a ``.git`` entry. Returns the
    repo root or ``None`` if ``cwd`` isn't inside a git repo. Lets a
    session started in a subfolder still resolve to the right bucket
    (the subfolder basename would otherwise mislead).
    """
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


def _atomic_write_aliases(home: Path, data: Mapping[str, str]) -> None:
    """Write the alias map atomically (tmp file + ``os.replace``).
    Errors propagate — callers are expected to swallow them so auto-
    bootstrap never breaks a session.
    """
    path = aliases_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(dict(data), tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def session_bucket(
    home: Path,
    cwd: Path,
    slug: Optional[str],
) -> Optional[str]:
    """Single source of truth: which library bucket does this session
    belong to?

    Used by every component that needs to answer "is this entry / nudge
    / write target part of the current project?" — session-start, the
    nudge filter, the priming scope-boost, and (in time) the scholar's
    write-path computation. Centralising the rule keeps the four call
    sites from drifting.

    Resolution order:

    1. **Existing bucket whose canonical slug matches the session slug.**
       Iterate ``knowledge/projects/*``; return the first bucket dir
       whose ``bucket_canonical_slug`` resolves to ``slug``. Covers the
       steady-state case where the alias file is already set up.

    2. **Derive a bucket-name candidate from cwd.** Walk up to the git
       repo root if there is one (``agent-mem/daemon`` → ``agent-mem``);
       otherwise fall back to the immediate cwd basename. Non-git
       projects naturally get slug == basename, so this branch usually
       only matters once git is added.

    3. **Auto-add the alias when there's evidence of a real bucket.**
       If a bucket dir already exists with the candidate name and has
       no alias entry yet, persist ``{candidate: slug}`` atomically so
       future priming + nudge filtering picks it up. Never overwrites
       an existing alias.

    4. **Always return the candidate.** Even when the bucket doesn't
       exist yet — that's the name the scholar will use when it creates
       the first entry for this project.

    Returns ``None`` only when ``slug`` is empty (no session context).
    """
    if not slug:
        return None

    aliases = load_aliases(home)
    projects_dir = home / "knowledge" / "projects"

    # Step 1: existing bucket already resolves to this slug.
    if projects_dir.exists():
        for bucket_dir in projects_dir.iterdir():
            if not bucket_dir.is_dir():
                continue
            if bucket_canonical_slug(bucket_dir.name, aliases) == slug:
                return bucket_dir.name

    # Step 2: derive candidate. Repo root if we're in a git tree, else
    # the cwd basename (which matches the no-git slug-derivation rule).
    repo_root = _find_repo_root(cwd)
    candidate_path = repo_root if repo_root is not None else cwd
    try:
        candidate = candidate_path.resolve().name
    except (OSError, RuntimeError):
        candidate = candidate_path.name
    if not candidate:
        return None

    # Step 3: if a real bucket exists with this name, auto-bootstrap
    # the alias so other call sites can find it from this session on.
    # No-op when candidate == slug (no translation needed) or when an
    # alias is already in place (don't overwrite — user might have
    # set it explicitly).
    bucket_exists = projects_dir.exists() and (projects_dir / candidate).is_dir()
    if bucket_exists and candidate != slug and candidate not in aliases:
        try:
            _atomic_write_aliases(home, {**aliases, candidate: slug})
        except OSError:
            pass

    return candidate


def ensure_alias_for_cwd(
    home: Path,
    cwd: Path,
    slug: Optional[str],
) -> Optional[Tuple[str, str]]:
    """Back-compat shim around :func:`session_bucket`. Returns the
    ``(bucket, slug)`` pair if a new alias entry was persisted, else
    ``None``. Prefer ``session_bucket`` in new code — it answers the
    more useful question ("which bucket?") directly.
    """
    aliases_before = load_aliases(home)
    bucket = session_bucket(home, cwd, slug)
    if bucket is None or slug is None:
        return None
    aliases_after = load_aliases(home)
    if aliases_after.get(bucket) == slug and aliases_before.get(bucket) != slug:
        return bucket, slug
    return None
