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
from typing import Dict, Mapping, Optional, Tuple

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
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, (str, int))}


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


def ensure_alias_for_cwd(
    home: Path,
    cwd: Path,
    slug: Optional[str],
) -> Optional[Tuple[str, str]]:
    """Auto-detect and persist a ``{bucket: slug}`` alias on session start.

    Scenario this handles: a repo cloned into folder ``X`` whose git
    remote produces a slug different from ``X`` (e.g. ``X = agent-mem``
    but slug = ``github.com-nickroci-ultan``). If a library bucket
    named ``X`` already exists, write the mapping automatically so the
    user doesn't have to hand-edit the config.

    No-op (returns ``None``) when any of:
    - ``slug`` is empty
    - some existing bucket already resolves (directly or via alias) to
      ``slug`` — nothing to add
    - ``cwd`` isn't inside a git repo — the slug then came from cwd
      basename already, no alias needed (and would be wrong if added)
    - no library bucket exists with the repo-root basename
    - that bucket already has an explicit alias entry — never overwrite

    Returns the ``(bucket, slug)`` pair that was added, otherwise ``None``.
    """
    if not slug:
        return None

    aliases = load_aliases(home)
    projects_dir = home / "knowledge" / "projects"
    if not projects_dir.exists():
        return None

    # Already covered? Check every bucket's canonical slug against the
    # session slug; if anything matches, no bootstrap needed.
    for bucket_dir in projects_dir.iterdir():
        if not bucket_dir.is_dir():
            continue
        if bucket_canonical_slug(bucket_dir.name, aliases) == slug:
            return None

    # No git -> slug came from cwd basename and either already matches
    # a bucket (handled above) or there's no bucket yet. Either way we
    # have nothing to write.
    repo_root = _find_repo_root(cwd)
    if repo_root is None:
        return None

    bucket_name = repo_root.name
    if not bucket_name:
        return None
    if bucket_name in aliases:
        return None  # never overwrite
    if not (projects_dir / bucket_name).is_dir():
        return None

    new_aliases = dict(aliases)
    new_aliases[bucket_name] = slug
    try:
        _atomic_write_aliases(home, new_aliases)
    except OSError:
        return None
    return bucket_name, slug
