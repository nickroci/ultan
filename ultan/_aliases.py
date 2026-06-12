"""Local stdlib mirror of the project-bucket alias helpers from
``tools/search/aliases.py``.

The thin ``ultan`` install (no ``[retrieval]`` extra) doesn't ship the
agent-mem-search package, so ``from aliases import …`` would break the base
install and its CI job (`uv sync --locked` with no extra → ModuleNotFoundError);
``_priming.py`` mirrors its priming client for the same reason. These two
helpers are shared by ``_nudges`` (cross-project nudge filter) and
``_session_context`` (SessionStart bucket resolution) — kept here, once, so the
mirror has a single source of truth. Keep byte-faithful to the canonical
``aliases`` module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, cast

_ALIASES_FILENAME = "project-aliases.json"


def load_aliases(home: Path) -> Dict[str, str]:
    """Read ``project-aliases.json`` → ``{bucket: slug}``. Returns ``{}`` when the
    file is missing, unreadable, or not a JSON object — alias resolution must
    never crash the caller. Mirror of ``aliases.load_aliases``."""
    try:
        text = (home / _ALIASES_FILENAME).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    try:
        data: object = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    items = cast("dict[object, object]", data)
    out: Dict[str, str] = {}
    for k, v in items.items():
        if isinstance(v, (str, int)):
            out[str(k)] = str(v)
    return out


def bucket_canonical_slug(bucket: Optional[str], aliases: Mapping[str, str]) -> Optional[str]:
    """Canonical project slug for a bucket directory: alias-map lookup, falling
    back to the bucket name itself when absent. ``None`` in → ``None`` out.
    Mirror of ``aliases.bucket_canonical_slug``."""
    if not bucket:
        return None
    return aliases.get(bucket, bucket)
