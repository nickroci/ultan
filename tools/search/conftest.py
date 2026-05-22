"""Shared pytest fixtures for the tools/search test suite.

Centralises the patterns the individual test files used to copy-paste:
  - a fresh ``tmp_path / knowledge`` copy of the fixture corpus,
  - a paired ``home`` + ``knowledge`` layout (with ``AGENT_MEM_HOME`` env
    pointed at it) for doctor / lifecycle scenarios,
  - small helpers for materialising provisional entries inline,
  - the ``fake_sdk`` patcher for ``cli``'s claude-agent-sdk bindings.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Any, Callable

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeAssistantMessage:
    def __init__(self, content: list[FakeTextBlock]) -> None:
        self.content = content


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the cli module's SDK bindings with configurable fakes.

    cli.py imports AssistantMessage / ClaudeAgentOptions / TextBlock / query
    at module top, so sys.modules patches no longer reach the bound names.
    This fixture uses ``monkeypatch.setattr`` on the cli module directly.

    Mutate the returned dict before driving the code under test:

        fake_sdk["chunks"] = ["text the fake assistant emits"]
        fake_sdk["raise_exc"] = RuntimeError("boom")
    """
    import cli

    config: dict[str, Any] = {"chunks": [], "raise_exc": None}

    async def _query(**_kw: Any) -> Any:
        if config["raise_exc"] is not None:
            raise config["raise_exc"]
        for c in config["chunks"]:
            yield FakeAssistantMessage([FakeTextBlock(c)])

    monkeypatch.setattr(cli, "query", _query)
    monkeypatch.setattr(cli, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(cli, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(cli, "ClaudeAgentOptions", FakeClaudeAgentOptions)
    return config


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
