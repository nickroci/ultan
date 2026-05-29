"""Pure, dependency-light validation helpers shared by the Scholar's two
validation layers so they can never drift apart:

  - **Boundary validation** (``_schemas`` Pydantic model/field validators and
    the Pydantic AI ``output_validator``): runs on the typed actions the
    Scholar emits BEFORE the daemon touches disk, so a malformed action is
    bounced back to the model via ``ModelRetry``.
  - **Post-write safety net** (``scholar_prompt.check_invariants``): runs on
    the on-disk tree AFTER the executor applies actions, to catch anything
    the boundary missed.

Both layers ask the same questions — does the frontmatter parse, are the
required fields present, does ``id`` match the filename slug, does ``scope``
agree with the path, does every wikilink resolve — so the question-asking
code lives here, once.

This module imports only ``yaml`` + the daemon's ``markdown_utils`` (itself
dependency-light). It must NOT import ``_schemas`` / ``scholar_prompt`` /
``scholar_agent`` so it can be imported from any of them without a cycle.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, cast

import yaml

from . import markdown_utils

# ── Frontmatter parsing ──────────────────────────────────────────────────

# Leading YAML frontmatter block (``---\n...\n---``) at the very start of an
# entry body. Non-greedy so a ``---`` rule later in the body isn't swallowed.
FRONTMATTER_HEAD_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# Frontmatter fields the schema requires on every entry. Single source of
# truth — both the model validators and ``check_invariants`` read this tuple.
REQUIRED_FRONTMATTER_FIELDS = (
    "id",
    "type",
    "scope",
    "status",
    "confidence",
    "applies-when",
    "keywords",
    "title",
    "created",
    "updated",
    "fired",
    "fired-helpful",
    "sources",
)

# How many entry .md files per directory we permit before flagging.
MAX_FLAT_DIR_ENTRIES = 5


def parse_frontmatter(text: str) -> Dict[str, Any]:
    """Return the parsed YAML frontmatter mapping, or ``{}`` when the block
    is missing, unparseable, or not a mapping."""
    m = FRONTMATTER_HEAD_RE.match(text)
    if not m:
        return {}
    try:
        loaded: object = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}
    if isinstance(loaded, dict):
        return cast(Dict[str, Any], loaded)
    return {}


def strip_frontmatter(text: str) -> str:
    """Return ``text`` with a leading YAML frontmatter block removed."""
    m = FRONTMATTER_HEAD_RE.match(text)
    if not m:
        return text
    return text[m.end() :]


def missing_frontmatter_fields(fm: Dict[str, Any]) -> List[str]:
    """Required-field names absent from a parsed frontmatter mapping."""
    return [f for f in REQUIRED_FRONTMATTER_FIELDS if f not in fm]


# ── id ↔ filename-slug agreement ─────────────────────────────────────────


def path_slug(path: str) -> str:
    """The expected ``id`` for an entry at ``path`` — the filename stem with
    any ``.md`` suffix stripped (e.g. ``global/python/use-uv.md`` → ``use-uv``).
    """
    leaf = path.rsplit("/", 1)[-1]
    if leaf.endswith(".md"):
        leaf = leaf[:-3]
    return leaf


# ── scope ↔ path agreement ───────────────────────────────────────────────


def scope_path_violation(rel: Path, scope: str) -> str | None:
    """``scope: global`` must live under ``global/``; ``scope: project:<slug>``
    under ``projects/<slug>/``. Returns a one-line violation string or
    ``None`` when the path agrees with the scope (or the scope is unknown,
    which other checks handle)."""
    parts = rel.parts
    if not parts:
        return None
    if scope == "global":
        if parts[0] != "global":
            return f"scope/path mismatch in {rel}: scope=global but path is not under global/"
        return None
    if scope.startswith("project:"):
        slug = scope.split(":", 1)[1].strip()
        if parts[0] != "projects" or (len(parts) > 1 and parts[1] != slug):
            return (
                f"scope/path mismatch in {rel}: scope={scope!r} but "
                f"path is not under projects/{slug}/"
            )
    return None


# ── wikilink resolution ──────────────────────────────────────────────────


def wikilink_resolves(link: str, parent_dir: Path, knowledge_root: Path) -> bool:
    """True iff a wikilink target resolves to an existing entry/folder.

    Resolution rules (must match the post-write checker in
    ``scholar_prompt._wikilink_resolves``):
      - ``_archive/...`` and ``daily/...`` targets are always treated as
        resolvable (archive is intentionally out of the active graph; daily
        notes live outside the knowledge tree).
      - A trailing ``/`` denotes a folder link → resolves to ``<link>/README.md``.
      - Otherwise the target is an entry → resolves to ``<link>.md``.
      - Root-relative first, then a sibling fallback relative to ``parent_dir``.
    """
    if link.startswith("_archive/") or "/_archive/" in link:
        return True
    if link.startswith("daily/"):
        return True
    if link.endswith("/"):
        target = knowledge_root / link / "README.md"
    else:
        target = knowledge_root / (link if link.endswith(".md") else f"{link}.md")
    if target.exists():
        return True
    if link.endswith("/"):
        sibling = parent_dir / link / "README.md"
    else:
        sibling = parent_dir / (link if link.endswith(".md") else f"{link}.md")
    return sibling.exists()


def body_wikilinks(text: str) -> List[str]:
    """Every prose wikilink target in ``text`` (code spans / fences /
    frontmatter excluded by ``markdown_utils``). Empty targets dropped."""
    return [hit.target for hit in markdown_utils.extract_wikilinks(text) if hit.target]
