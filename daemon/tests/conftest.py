"""Shared fixtures.

The fixtures here are used by multiple test modules:

  - ``_isolate_agent_mem_home``: an autouse fixture that points the
    daemon's ``home()`` at a temp directory for every test, so we never
    touch the user's real ``~/.agent-mem``.
  - ``seed_library``: builds a small but realistic library tree used by
    several integration tests (priming, rpc, library_tools).
  - ``library_entry``: factory producing a single valid library entry.

The module-level helpers (``render_entry``, ``write_entry``,
``build_library``) are imported directly by ``test_priming`` and
``test_priming_rpc`` so the seed/entry scaffolding lives in exactly one
place rather than being copy-pasted per module.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

# Default body template shared by most callers. ``test_priming`` overrides
# it with a slightly different phrasing (see that module) — both are passed
# through ``render_entry`` so there is a single renderer implementation.
DEFAULT_BODY_TEMPLATE = "Body for {id_}. {applies_when}."


def render_entry(
    *,
    id_: str,
    title: str,
    applies_when: str,
    keywords: List[str],
    reinforced: Optional[int] = None,
    body: str = "",
    scope: str = "global",
    default_body_template: str = DEFAULT_BODY_TEMPLATE,
) -> str:
    """Render a valid library entry as a YAML-frontmattered markdown
    string. Matches the schema enforced by
    ``scholar_prompt._REQUIRED_FRONTMATTER_FIELDS``.

    When ``body`` is the empty string the body is synthesised from
    ``default_body_template`` (which sees ``id_`` and ``applies_when``)."""
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
    lines.append(body or default_body_template.format(id_=id_, applies_when=applies_when))
    lines.append("")
    return "\n".join(lines)


# Backwards-compatible alias: a couple of modules referenced ``_entry_text``.
_entry_text = render_entry


def write_entry(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# Default short body for the python/uv entry. ``test_priming`` passes a
# longer multi-line variant so its rerank/BM25 fixtures stay byte-identical.
DEFAULT_UV_BODY = "Always use uv for python package management. Never pip."


def build_library(
    root: Path,
    *,
    include_type_hints: bool = False,
    uv_body: str = DEFAULT_UV_BODY,
    default_body_template: str = DEFAULT_BODY_TEMPLATE,
) -> Path:
    """Build a small library under ``root/knowledge`` with enough entries
    that BM25 IDF doesn't collapse to zero on common tokens.

    ``include_type_hints`` adds an extra python entry (priming's fixture
    seeds 8 leaf entries; the rpc fixture seeds 7). ``uv_body`` and
    ``default_body_template`` let callers reproduce their exact prior
    bytes — both BM25 and the cross-encoder rerank are body-sensitive.

    Returns the knowledge root path."""
    k = root / "knowledge"
    write_entry(k / "README.md", "# knowledge root\n")
    write_entry(k / "global" / "README.md", "# global\n")
    write_entry(k / "global" / "python" / "README.md", "# python\n")
    write_entry(k / "global" / "git" / "README.md", "# git\n")

    def _entry(**kwargs: object) -> str:
        return render_entry(default_body_template=default_body_template, **kwargs)  # type: ignore[arg-type]

    write_entry(
        k / "global" / "python" / "use-uv-not-pip.md",
        _entry(
            id_="use-uv-not-pip",
            title="Always use uv for python",
            applies_when="installing python deps or running scripts",
            keywords=["python", "uv", "pip", "packaging"],
            body=uv_body,
        ),
    )
    write_entry(
        k / "global" / "python" / "ruff-format.md",
        _entry(
            id_="ruff-format",
            title="Format python with ruff",
            applies_when="formatting python files",
            keywords=["python", "ruff", "format"],
        ),
    )
    if include_type_hints:
        write_entry(
            k / "global" / "python" / "type-hints.md",
            _entry(
                id_="type-hints",
                title="Always use type hints",
                applies_when="writing python functions",
                keywords=["python", "types", "mypy"],
            ),
        )
    write_entry(
        k / "global" / "git" / "no-force-push.md",
        _entry(
            id_="no-force-push",
            title="Never force-push to main",
            applies_when="pushing to git remotes",
            keywords=["git", "push", "remote"],
        ),
    )
    write_entry(
        k / "global" / "git" / "small-commits.md",
        _entry(
            id_="small-commits",
            title="Prefer small commits",
            applies_when="committing changes",
            keywords=["git", "commits", "history"],
        ),
    )
    write_entry(
        k / "global" / "git" / "branch-naming.md",
        _entry(
            id_="branch-naming",
            title="Use kebab-case branches",
            applies_when="creating new git branches",
            keywords=["git", "branches", "naming"],
        ),
    )
    write_entry(
        k / "global" / "git" / "rebase-not-merge.md",
        _entry(
            id_="rebase-not-merge",
            title="Prefer rebase over merge",
            applies_when="updating feature branches",
            keywords=["git", "rebase", "merge"],
        ),
    )
    write_entry(
        k / "global" / "git" / "signed-commits.md",
        _entry(
            id_="signed-commits",
            title="Sign all commits",
            applies_when="committing changes",
            keywords=["git", "gpg", "sign"],
        ),
    )
    return k


@pytest.fixture
def library_entry():
    """Factory fixture so tests can ask for entries on demand without
    re-importing the helper."""
    return render_entry


@pytest.fixture
def seed_library(tmp_path):
    """Build a small library under ``tmp_path/knowledge``.

    Returns a callable so tests can seed into an explicit root (or default
    to ``tmp_path``). Mirrors the previous fixture contract."""

    def _seed(root: Optional[Path] = None) -> Path:
        return build_library(root or tmp_path)

    return _seed


@pytest.fixture
def agent_mem_home(tmp_path, monkeypatch):
    """Point ``AGENT_MEM_HOME`` at a per-test tmp directory.

    Not autouse — opt-in. Tests that don't touch ``home()`` paths don't
    need this overhead.
    """
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    return tmp_path
