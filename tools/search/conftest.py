"""Shared pytest fixtures for the tools/search test suite.

Centralises the patterns the individual test files used to copy-paste:
  - a fresh ``tmp_path / knowledge`` copy of the fixture corpus,
  - a paired ``home`` + ``knowledge`` layout (with ``AGENT_MEM_HOME`` env
    pointed at it) for doctor / lifecycle scenarios,
  - small helpers for materialising provisional entries inline.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Callable

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """A fresh copy of the fixture corpus under ``tmp_path / knowledge``.

    No provisional entries injected — use ``knowledge_dir_with_provisional``
    when you need lifecycle scenarios.
    """
    dst = tmp_path / "knowledge"
    shutil.copytree(FIXTURES, dst)
    return dst


@pytest.fixture
def home_and_knowledge(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(home, knowledge)`` with the fixture corpus copied under
    ``home/knowledge``. Mirrors the on-disk layout the daemon and CLI
    expect (knowledge dir is parent's child of home).
    """
    home = tmp_path / "home"
    knowledge = home / "knowledge"
    shutil.copytree(FIXTURES, knowledge)
    return home, knowledge


@pytest.fixture
def make_provisional() -> Callable[..., Path]:
    """Factory: drop a provisional entry into a knowledge dir.

    Usage::

        make_provisional(knowledge_dir / "global" / "concepts" / "x.md", ident="x")
    """

    def _factory(path: Path, *, ident: str, scope: str = "global") -> Path:
        body = textwrap.dedent(
            f"""\
            ---
            id: {ident}
            type: lesson
            scope: {scope}
            status: provisional
            confidence: 0.6
            applies-when: |
              writing CLI lifecycle tests
            keywords: [test, lifecycle, {ident}]
            created: 2026-05-19
            updated: 2026-05-19
            fired: 0
            fired-helpful: 0
            ---

            # {ident}

            A provisional fixture entry.
            """
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    return _factory
