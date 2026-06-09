"""Shared utilities for agent-mem.

Most helpers are unchanged from upstream (claude-memory-compiler). Two
things move with the new layout:

- ``list_wiki_articles`` / ``read_all_wiki_content`` recurse into both
  ``knowledge/global/`` and ``knowledge/projects/<slug>/`` because articles
  now live under either tier.
- ``wiki_article_exists`` / ``count_inbound_links`` accept paths relative to
  ``knowledge/`` (e.g. ``global/concepts/foo`` or
  ``projects/<slug>/concepts/bar``), matching how wikilinks should be
  written under the new schema. The legacy short form (``concepts/foo``)
  is still accepted as a fallback so existing articles don't immediately
  go red.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict

from config import get_config

# ── State shape ───────────────────────────────────────────────────────


class IngestedEntry(TypedDict):
    """One row in ``state["ingested"]`` — written by the removed compile.py;
    kept so legacy state files on disk still parse."""

    hash: str
    compiled_at: str
    cost_usd: float


class State(TypedDict, total=False):
    """Persistent state stored in ``~/.agent-mem/state/state.json``.

    ``total=False`` because legacy state files on disk may be missing keys; the
    loader fills in defaults, but downstream code should treat every key as
    optional and use ``.get(...)`` with a default.
    """

    ingested: dict[str, IngestedEntry]
    query_count: int
    last_lint: str | None
    total_cost: float


# ── State management ──────────────────────────────────────────────────


def load_state() -> State:
    """Load persistent state from state.json."""
    state_file = get_config().state_file
    if state_file.exists():
        # json.loads returns Any; we trust the schema we wrote ourselves.
        loaded: State = json.loads(state_file.read_text(encoding="utf-8"))
        return loaded
    return {"ingested": {}, "query_count": 0, "last_lint": None, "total_cost": 0.0}


def save_state(state: State) -> None:
    """Save state to state.json."""
    cfg = get_config()
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── File hashing ──────────────────────────────────────────────────────


def file_hash(path: Path) -> str:
    """SHA-256 hash of a file (first 16 hex chars)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ── Slug / naming ─────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


# ── Wikilink helpers ──────────────────────────────────────────────────


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from markdown content."""
    return re.findall(r"\[\[([^\]]+)\]\]", content)


def wiki_article_exists(link: str) -> bool:
    """Check if a wikilinked article exists on disk.

    Wikilinks under the new layout are written relative to ``knowledge/``,
    e.g. ``global/concepts/factory-pattern`` or
    ``projects/github.com-foo-bar/concepts/x``. As a courtesy we also accept
    the legacy ``concepts/x`` form by trying ``global/concepts/x`` next.
    """
    knowledge_dir = get_config().knowledge_dir
    path = knowledge_dir / f"{link}.md"
    if path.exists():
        return True

    # Legacy form: ``concepts/x`` -> ``global/concepts/x``
    if not link.startswith(("global/", "projects/")):
        legacy = knowledge_dir / "global" / f"{link}.md"
        if legacy.exists():
            return True
    return False


# ── Wiki content helpers ──────────────────────────────────────────────


def read_wiki_index() -> str:
    """Read the knowledge base index file."""
    index_file = get_config().index_file
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return (
        "# Knowledge Base Index\n\n"
        "| Article | Summary | Compiled From | Updated |\n"
        "|---------|---------|---------------|---------|"
    )


def _iter_article_dirs() -> list[Path]:
    """Every directory we should scan for compiled articles.

    Both tiers — the global one and every per-project subtree — are
    fair game. Order: global first (most lessons), then projects
    alphabetically for stable output.
    """
    cfg = get_config()
    dirs: list[Path] = []
    for sub in ("concepts", "connections", "qa"):
        d = cfg.global_dir / sub
        if d.exists():
            dirs.append(d)

    if cfg.projects_dir.exists():
        for proj in sorted(p for p in cfg.projects_dir.iterdir() if p.is_dir()):
            for sub in ("concepts", "connections", "qa"):
                d = proj / sub
                if d.exists():
                    dirs.append(d)
    return dirs


def read_all_wiki_content() -> str:
    """Read index + all wiki articles into a single string for context."""
    parts = [f"## INDEX\n\n{read_wiki_index()}"]

    knowledge_dir = get_config().knowledge_dir
    for subdir in _iter_article_dirs():
        for md_file in sorted(subdir.glob("*.md")):
            rel = md_file.relative_to(knowledge_dir)
            content = md_file.read_text(encoding="utf-8")
            parts.append(f"## {rel}\n\n{content}")

    return "\n\n---\n\n".join(parts)


def list_wiki_articles() -> list[Path]:
    """List all wiki article files across global and per-project tiers."""
    articles: list[Path] = []
    for subdir in _iter_article_dirs():
        articles.extend(sorted(subdir.glob("*.md")))
    return articles


def list_raw_files() -> list[Path]:
    """List all daily log files."""
    daily_dir = get_config().daily_dir
    if not daily_dir.exists():
        return []
    return sorted(daily_dir.glob("*.md"))


# ── Index helpers ─────────────────────────────────────────────────────


def count_inbound_links(target: str, exclude_file: Path | None = None) -> int:
    """Count how many wiki articles link to a given target.

    Tolerant of legacy vs new wikilink forms: counts both
    ``[[concepts/foo]]`` and ``[[global/concepts/foo]]`` as inbound to the
    same article so migration doesn't blow up the orphan-page check.
    """
    alt: str | None = None
    if target.startswith("global/"):
        alt = target[len("global/") :]
    elif "/" in target and not target.startswith("projects/"):
        alt = f"global/{target}"

    count = 0
    for article in list_wiki_articles():
        if article == exclude_file:
            continue
        content = article.read_text(encoding="utf-8")
        if f"[[{target}]]" in content:
            count += 1
        elif alt and f"[[{alt}]]" in content:
            count += 1
    return count


def get_article_word_count(path: Path) -> int:
    """Count words in an article, excluding YAML frontmatter."""
    content = path.read_text(encoding="utf-8")
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3 :]
    return len(content.split())


def build_index_entry(rel_path: str, summary: str, sources: str, updated: str) -> str:
    """Build a single index table row."""
    link = rel_path.replace(".md", "")
    return f"| [[{link}]] | {summary} | {sources} | {updated} |"
