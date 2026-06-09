"""Shared pytest fixtures for the ``scripts/`` regression tests.

The scripts (``compile.py``, ``lint.py``, ``flush.py``, ``query.py`` and
their shared ``utils.py``) resolve every storage path through
``config.get_config()``, which reads ``AGENT_MEM_HOME`` *at call time*.
So test isolation is just::

    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))

— no module reloading, no ``del sys.modules`` dance. The fixtures here
build the scaffolding once so individual test files don't copy-paste
library-seeding blocks (the repo has an 8-line duplicate-code gate).

``config`` / ``utils`` import as top-level names because ``src`` is
installed editable into the workspace venv (its flat ``scripts/``
modules flatten to top-level), so no ``sys.path`` tweaking is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

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
