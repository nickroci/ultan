"""Shared fixtures.

The fixtures here are used by multiple test modules:

  - ``_isolate_agent_mem_home``: an autouse fixture that points the
    daemon's ``home()`` at a temp directory for every test, so we never
    touch the user's real ``~/.agent-mem``.
  - ``seed_library``: builds a small but realistic library tree used by
    several integration tests (priming, rpc, library_tools).
  - ``library_entry``: factory producing a single valid library entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest


def _entry_text(
    *,
    id_: str,
    title: str,
    applies_when: str,
    keywords: List[str],
    reinforced: Optional[int] = None,
    body: str = "",
    scope: str = "global",
) -> str:
    """Render a minimal valid library entry (mirrors the inline helper
    already in test_priming.py / test_priming_rpc.py; centralised so new
    tests stop reinventing it)."""
    lines = [
        "---",
        f"id: {id_}",
        "type: lesson",
        f"scope: {scope}",
        "status: provisional",
        "confidence: 0.7",
        "applies-when: |",
    ]
    for line in applies_when.splitlines():
        lines.append(f"  {line}")
    lines.append("keywords: [" + ", ".join(keywords) + "]")
    lines.append(f'title: "{title}"')
    lines.append("created: 2026-05-19")
    lines.append("updated: 2026-05-19")
    lines.append("fired: 0")
    lines.append("fired-helpful: 0")
    if reinforced is not None:
        lines.append(f"reinforced: {reinforced}")
    lines.append("sources:")
    lines.append("  - manual")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(body or f"Body for {id_}. {applies_when}.")
    lines.append("")
    return "\n".join(lines)


@pytest.fixture
def library_entry():
    """Factory fixture so tests can ask for entries on demand without
    re-importing the helper."""
    return _entry_text


@pytest.fixture
def seed_library(tmp_path):
    """Build a small library under ``tmp_path/knowledge`` with enough
    entries that BM25 IDF doesn't collapse to zero on common tokens.

    Returns the knowledge root path."""

    def _write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _seed(root: Optional[Path] = None) -> Path:
        base = root or tmp_path
        k = base / "knowledge"
        _write(k / "README.md", "# knowledge root\n")
        _write(k / "global" / "README.md", "# global\n")
        _write(k / "global" / "python" / "README.md", "# python\n")
        _write(k / "global" / "git" / "README.md", "# git\n")
        _write(
            k / "global" / "python" / "use-uv-not-pip.md",
            _entry_text(
                id_="use-uv-not-pip",
                title="Always use uv for python",
                applies_when="installing python deps or running scripts",
                keywords=["python", "uv", "pip", "packaging"],
                body="Always use uv for python package management. Never pip.",
            ),
        )
        _write(
            k / "global" / "python" / "ruff-format.md",
            _entry_text(
                id_="ruff-format",
                title="Format python with ruff",
                applies_when="formatting python files",
                keywords=["python", "ruff", "format"],
            ),
        )
        _write(
            k / "global" / "git" / "no-force-push.md",
            _entry_text(
                id_="no-force-push",
                title="Never force-push to main",
                applies_when="pushing to git remotes",
                keywords=["git", "push", "remote"],
            ),
        )
        _write(
            k / "global" / "git" / "small-commits.md",
            _entry_text(
                id_="small-commits",
                title="Prefer small commits",
                applies_when="committing changes",
                keywords=["git", "commits", "history"],
            ),
        )
        _write(
            k / "global" / "git" / "branch-naming.md",
            _entry_text(
                id_="branch-naming",
                title="Use kebab-case branches",
                applies_when="creating new git branches",
                keywords=["git", "branches", "naming"],
            ),
        )
        _write(
            k / "global" / "git" / "rebase-not-merge.md",
            _entry_text(
                id_="rebase-not-merge",
                title="Prefer rebase over merge",
                applies_when="updating feature branches",
                keywords=["git", "rebase", "merge"],
            ),
        )
        _write(
            k / "global" / "git" / "signed-commits.md",
            _entry_text(
                id_="signed-commits",
                title="Sign all commits",
                applies_when="committing changes",
                keywords=["git", "gpg", "sign"],
            ),
        )
        return k

    return _seed


@pytest.fixture
def agent_mem_home(tmp_path, monkeypatch):
    """Point ``AGENT_MEM_HOME`` at a per-test tmp directory.

    Not autouse — opt-in. Tests that don't touch ``home()`` paths don't
    need this overhead.
    """
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    return tmp_path
