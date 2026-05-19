"""Frontmatter read/write utilities for `agent-mem` lifecycle commands.

Used by `promote`, `demote`, `forget`, and `review` to mutate entry
frontmatter without disturbing the body. Atomic writes (tmp + rename),
key ordering preserved where possible, no YAML reformatting surprises.

The API is intentionally small:

    read(path)            -> (fm_dict, body_str)
    write(path, fm, body) -> None              (atomic)
    set_status(path, s)   -> None              (also bumps `updated`)
    bump_updated(path)    -> None
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Tuple

import yaml

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# The canonical order of fields documented in src/AGENTS.md §2. Anything not
# in this list is appended at the end in original insertion order, which
# yaml.safe_load preserves on dict (PyYAML returns a regular dict and Python
# dicts preserve insertion order since 3.7).
_CANONICAL_ORDER: tuple[str, ...] = (
    "id",
    "type",
    "scope",
    "status",
    "confidence",
    "applies-when",
    "keywords",
    "title",
    "aliases",
    "tags",
    "created",
    "updated",
    "archived",
    "fired",
    "fired-helpful",
    "sources",
    "related",
)


class FrontmatterError(ValueError):
    """Raised when a file has no frontmatter or malformed frontmatter."""


def read(path: Path) -> Tuple[dict, str]:
    """Read a markdown file's YAML frontmatter and body.

    Returns ``({}, full_text)`` if there is no frontmatter block — callers
    that *require* frontmatter should check for the empty dict.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    try:
        fm = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise FrontmatterError(f"malformed YAML frontmatter in {path}: {e}") from e
    if not isinstance(fm, dict):
        raise FrontmatterError(
            f"frontmatter in {path} is not a mapping (got {type(fm).__name__})"
        )
    body = text[m.end():]
    return fm, body


def _ordered(fm: dict) -> dict:
    """Return a new dict with canonical fields first, others trailing."""
    out: dict = {}
    for key in _CANONICAL_ORDER:
        if key in fm:
            out[key] = fm[key]
    for key, value in fm.items():
        if key not in out:
            out[key] = value
    return out


def _dump_yaml(fm: dict) -> str:
    """Dump frontmatter as YAML.

    - Preserves the multi-line block style of ``applies-when`` when it is a
      literal block scalar (``|``) — we coerce strings with embedded newlines
      to that style explicitly so they round-trip predictably.
    - sort_keys=False so our ordering wins.
    - allow_unicode=True so non-ASCII content stays readable.
    """
    # Coerce: any string value with a newline gets block-scalar style.
    class _Dumper(yaml.SafeDumper):
        pass

    def _str_representer(dumper: yaml.SafeDumper, data: str):
        style = "|" if "\n" in data else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    _Dumper.add_representer(str, _str_representer)

    return yaml.dump(
        fm,
        Dumper=_Dumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=None,
    )


def write(path: Path, fm: dict, body: str) -> None:
    """Atomically write frontmatter + body to ``path``.

    Uses a tmp file in the same directory + ``os.replace`` so callers never
    see a half-written file. Preserves canonical field ordering where the
    fields are known; unknown fields trail.
    """
    path = Path(path)
    ordered = _ordered(fm)
    rendered = "---\n" + _dump_yaml(ordered) + "---\n" + body

    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile with delete=False so we can rename it.
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(rendered)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the tmp file if anything failed before replace.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _today() -> str:
    return date.today().isoformat()


def bump_updated(path: Path, today: str | None = None) -> None:
    """Set ``updated:`` to today's date (or the supplied string)."""
    fm, body = read(path)
    fm["updated"] = today or _today()
    write(path, fm, body)


def set_status(path: Path, status: str, *, today: str | None = None) -> None:
    """Set ``status:`` and bump ``updated:`` in one atomic write."""
    if status not in {"provisional", "confirmed", "stale"}:
        raise ValueError(
            f"invalid status {status!r}; expected provisional|confirmed|stale"
        )
    fm, body = read(path)
    fm["status"] = status
    fm["updated"] = today or _today()
    write(path, fm, body)


def get_status(fm_or_path: Any) -> str | None:
    """Best-effort status lookup. Accepts a dict (parsed fm) or a path."""
    if isinstance(fm_or_path, dict):
        v = fm_or_path.get("status")
    else:
        fm, _ = read(Path(fm_or_path))
        v = fm.get("status")
    return str(v) if v is not None else None
