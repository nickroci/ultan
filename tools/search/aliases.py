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
from pathlib import Path
from typing import Dict, Mapping, Optional

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
