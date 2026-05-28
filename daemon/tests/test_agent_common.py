"""Tests for the shared curator-agent building blocks in ``_agent_common``.

Covers the pure helpers both agents reuse: path-containment, grep, the
search-runner unwrap (including the empty-content sentinel), and the
best-effort cost estimator's failure path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agent_mem_daemon import _agent_common

from .conftest import seed_scholar_tree


def test_inside_guard() -> None:
    root = Path("/a/b")
    assert _agent_common.inside(root, Path("/a/b/c"))
    assert not _agent_common.inside(root, Path("/a/x"))


def test_grep_library_finds_matches(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    out = _agent_common.grep_library(k, "real body", "")
    assert "use-uv.md" in out


def test_grep_library_empty_pattern(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "empty pattern" in _agent_common.grep_library(k, "  ", "")


def test_grep_library_missing_scope(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "not found" in _agent_common.grep_library(k, "x", "global/nonexistent")


def test_grep_library_no_matches(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "no matches" in _agent_common.grep_library(k, "zzz-not-present-zzz", "")


def test_grep_library_skips_archive(tmp_path: Path) -> None:
    k = tmp_path / "knowledge"
    (k / "_archive").mkdir(parents=True)
    (k / "live.md").write_text("the needle here\n", encoding="utf-8")
    (k / "_archive" / "old.md").write_text("the needle too\n", encoding="utf-8")
    out = _agent_common.grep_library(k, "needle", "")
    assert "live.md" in out
    assert "_archive" not in out


def test_grep_library_truncates(tmp_path: Path) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "big.md").write_text("\n".join(f"line {i} needle" for i in range(80)), encoding="utf-8")
    assert "truncated at 40 matches" in _agent_common.grep_library(k, "needle", "")


def test_search_text_unwraps_response(tmp_path: Path) -> None:
    def _runner(args: Dict[str, Any], root: Path) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": f"hit for {args['query']}"}]}

    assert _agent_common.search_text(_runner, tmp_path, "uv", 5) == "hit for uv"


def test_search_text_handles_empty(tmp_path: Path) -> None:
    def _runner(args: Dict[str, Any], root: Path) -> Dict[str, Any]:
        return {"content": []}

    assert "no content" in _agent_common.search_text(_runner, tmp_path, "q", 5)


def test_estimate_cost_best_effort_on_bad_input() -> None:
    # Passing nonsense usage must not raise; returns 0.0 on failure.
    assert _agent_common.estimate_cost("anthropic:claude-sonnet-4-6", object()) == 0.0
