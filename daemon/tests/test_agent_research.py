"""Tests for the read-only research tools the curator roles hand the
typed-agent shim (``_agent_research``).

The pure helpers (inside / read_entry / grep_library / search_text / _coerce_k)
are tested directly; the SDK server/tool-ref builders are smoke-tested. No model
calls — these are the in-process tools the model would invoke.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agent_mem_daemon import _agent_research
from agent_mem_daemon._agent_research import SERVER_NAME

from .conftest import seed_scholar_tree

# ── inside (path-escape guard) ───────────────────────────────────────────────


def test_inside_true_for_child() -> None:
    assert _agent_research.inside(Path("/a/b"), Path("/a/b/c"))


def test_inside_false_for_escape() -> None:
    assert not _agent_research.inside(Path("/a/b"), Path("/a/x"))


# ── read_entry ───────────────────────────────────────────────────────────────


def test_read_entry_returns_contents(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "A real body sentence." in _agent_research.read_entry(k, "global/python/use-uv.md")


def test_read_entry_not_found(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "(not found" in _agent_research.read_entry(k, "global/python/ghost.md")


def test_read_entry_refuses_path_escape(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "outside the knowledge store" in _agent_research.read_entry(k, "../../etc/passwd")


# ── grep_library ─────────────────────────────────────────────────────────────


def test_grep_library_finds_matches(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "use-uv.md" in _agent_research.grep_library(k, "real body", "")


def test_grep_library_empty_pattern(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "empty pattern" in _agent_research.grep_library(k, "  ", "")


def test_grep_library_missing_scope(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "not found" in _agent_research.grep_library(k, "x", "global/nonexistent")


def test_grep_library_no_matches(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert "no matches" in _agent_research.grep_library(k, "zzz-absent-zzz", "")


def test_grep_library_truncates_at_40(tmp_path: Path) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "big.md").write_text("\n".join(f"line {i} needle" for i in range(80)), encoding="utf-8")
    assert "truncated at 40 matches" in _agent_research.grep_library(k, "needle", "")


# ── search_text (unwraps an injected runner) ─────────────────────────────────


def test_search_text_unwraps_runner_response(tmp_path: Path) -> None:
    def runner(args: Dict[str, Any], root: Path) -> Dict[str, Any]:  # noqa: ARG001
        return {"content": [{"type": "text", "text": f"hit:{args['query']}"}]}

    assert _agent_research.search_text(runner, tmp_path, "uv", 5) == "hit:uv"


def test_search_text_empty_response_sentinel(tmp_path: Path) -> None:
    def runner(args: Dict[str, Any], root: Path) -> Dict[str, Any]:  # noqa: ARG001
        return {"content": []}

    assert "no content" in _agent_research.search_text(runner, tmp_path, "q", 5)


# ── _coerce_k ────────────────────────────────────────────────────────────────


def test_coerce_k_default_for_non_numeric() -> None:
    assert _agent_research._coerce_k("abc") == 6
    assert _agent_research._coerce_k(None) == 6
    assert _agent_research._coerce_k(True) == 6  # bool is rejected, not treated as 1


def test_coerce_k_clamps_to_range() -> None:
    assert _agent_research._coerce_k(0) == 1
    assert _agent_research._coerce_k(99) == 20
    assert _agent_research._coerce_k("8") == 8


# ── server / tool-ref wiring ─────────────────────────────────────────────────


def test_research_tool_refs_are_namespaced() -> None:
    refs = _agent_research.research_tool_refs()
    assert len(refs) == 4
    assert all(r.startswith(f"mcp__{SERVER_NAME}__") for r in refs)
    assert any(r.endswith("read_entry") for r in refs)
    assert any(r.endswith("bm25_search") for r in refs)


def test_research_server_and_tools_shape(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    servers, allowed = _agent_research.research_server_and_tools(k)
    assert set(servers) == {SERVER_NAME}
    assert allowed == _agent_research.research_tool_refs()


def test_make_research_mcp_server_builds(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    assert _agent_research.make_research_mcp_server(k) is not None
