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

from config import (
    DAILY_DIR,
    GLOBAL_DIR,
    INDEX_FILE,
    KNOWLEDGE_DIR,
    PROJECTS_DIR,
    STATE_DIR,
    STATE_FILE,
)

# ── State management ──────────────────────────────────────────────────


def load_state() -> dict:
    """Load persistent state from state.json."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"ingested": {}, "query_count": 0, "last_lint": None, "total_cost": 0.0}


def save_state(state: dict) -> None:
    """Save state to state.json."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


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
    path = KNOWLEDGE_DIR / f"{link}.md"
    if path.exists():
        return True

    # Legacy form: ``concepts/x`` -> ``global/concepts/x``
    if not link.startswith(("global/", "projects/")):
        legacy = KNOWLEDGE_DIR / "global" / f"{link}.md"
        if legacy.exists():
            return True
    return False


# ── Wiki content helpers ──────────────────────────────────────────────


def read_wiki_index() -> str:
    """Read the knowledge base index file."""
    if INDEX_FILE.exists():
        return INDEX_FILE.read_text(encoding="utf-8")
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
    dirs: list[Path] = []
    for sub in ("concepts", "connections", "qa"):
        d = GLOBAL_DIR / sub
        if d.exists():
            dirs.append(d)

    if PROJECTS_DIR.exists():
        for proj in sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir()):
            for sub in ("concepts", "connections", "qa"):
                d = proj / sub
                if d.exists():
                    dirs.append(d)
    return dirs


def read_all_wiki_content() -> str:
    """Read index + all wiki articles into a single string for context."""
    parts = [f"## INDEX\n\n{read_wiki_index()}"]

    for subdir in _iter_article_dirs():
        for md_file in sorted(subdir.glob("*.md")):
            rel = md_file.relative_to(KNOWLEDGE_DIR)
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
    if not DAILY_DIR.exists():
        return []
    return sorted(DAILY_DIR.glob("*.md"))


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
