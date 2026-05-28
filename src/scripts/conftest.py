"""Shared pytest fixtures for the ``scripts/`` regression tests.

The scripts (``compile.py``, ``lint.py``, ``flush.py``, ``query.py`` and
their shared ``utils.py``) resolve every storage path through
``config.get_config()``, which reads ``AGENT_MEM_HOME`` *at call time*.
So test isolation is just::

    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))

— no module reloading, no ``del sys.modules`` dance. The fixtures here
build the scaffolding once so individual test files don't copy-paste
library-seeding blocks (the repo has an 8-line duplicate-code gate).

Mirrors ``hooks/conftest.py`` in putting ``scripts/`` on ``sys.path`` so
``import config`` / ``import utils`` resolve regardless of cwd.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

# Append (not insert) so we never shadow ``hooks/conftest.py`` when both
# test trees run together: the hooks tests do a plain ``from conftest
# import ...`` that must keep resolving to the hooks dir (which its own
# conftest puts at sys.path[0]). ``config`` / ``utils`` only live under
# scripts/, so appending is enough for our own ``import config`` etc.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))


# ── Home isolation ───────────────────────────────────────────────────


@pytest.fixture
def store_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin ``AGENT_MEM_HOME`` at ``tmp_path`` and return it.

    Because ``get_config()`` re-reads the env var on every call, this is
    the whole isolation story — nothing to reload, nothing to reset.
    """
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    return tmp_path


# ── Markdown article seeding ─────────────────────────────────────────


def write_article(
    path: Path,
    *,
    title: str,
    body: str,
    links: list[str] | None = None,
) -> Path:
    """Write a minimal compiled article (frontmatter + body) at ``path``.

    Mirrors the shape ``utils`` / ``lint`` parse: a YAML frontmatter block
    delimited by ``---`` (so ``get_article_word_count`` can strip it) plus
    a markdown body that may embed ``[[wikilinks]]``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    link_lines = "\n".join(f"- [[{link}]]" for link in (links or []))
    text = (
        "---\n"
        f"id: {path.stem}\n"
        "type: concept\n"
        "scope: global\n"
        f'title: "{title}"\n'
        "---\n\n"
        f"# {title}\n\n"
        f"{body}\n\n"
        "## Related\n\n"
        f"{link_lines}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def article_writer() -> Callable[..., Path]:
    """Fixture wrapper around :func:`write_article`."""
    return write_article
