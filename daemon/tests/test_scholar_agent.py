"""Tests for the Pydantic-AI Scholar agent: read-only tools, the
boundary output-validator (wikilink resolution + flat-dir cap), the
ModelRetry feedback loop, and the run wrapper.

No real Anthropic calls — we drive the agent with Pydantic AI's
``FunctionModel`` (a scripted model) via ``agent.override(model=...)``,
and exercise the pure helpers directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_mem_daemon import scholar_agent
from agent_mem_daemon._schemas import ScholarDecisions
from agent_mem_daemon.scholar_agent import SCHOLAR_AGENT, ScholarDeps

from .conftest import scholar_entry_body, seed_scholar_tree


def _valid_body(id_: str, *, scope: str = "global", body_extra: str = "") -> str:
    return scholar_entry_body(id_, scope=scope, extra=body_extra)


def _seed_clean_tree(tmp_path: Path) -> Path:
    return seed_scholar_tree(tmp_path)


# ── Read-only tool helpers ───────────────────────────────────────────


def test_grep_library_finds_matches(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    out = scholar_agent._grep_library(k, "real body", "")
    assert "use-uv.md" in out


def test_grep_library_empty_pattern(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    assert "empty pattern" in scholar_agent._grep_library(k, "  ", "")


def test_grep_library_missing_scope(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    assert "not found" in scholar_agent._grep_library(k, "x", "global/nonexistent")


def test_grep_library_no_matches(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    assert "no matches" in scholar_agent._grep_library(k, "zzz-not-present-zzz", "")


def test_grep_library_truncates(tmp_path: Path):
    k = tmp_path / "knowledge"
    k.mkdir()
    big = "\n".join(f"line {i} needle" for i in range(80))
    (k / "big.md").write_text(big, encoding="utf-8")
    out = scholar_agent._grep_library(k, "needle", "")
    assert "truncated at 40 matches" in out


def test_search_text_unwraps_response(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)

    def _runner(args: Dict[str, Any], root: Path) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": f"hit for {args['query']}"}]}

    assert scholar_agent._search_text(_runner, k, "uv", 5) == "hit for uv"


def test_search_text_handles_empty(tmp_path: Path):
    def _runner(args: Dict[str, Any], root: Path) -> Dict[str, Any]:
        return {"content": []}

    assert "no content" in scholar_agent._search_text(_runner, tmp_path, "q", 5)


def test_inside_guard():
    root = Path("/a/b")
    assert scholar_agent._inside(root, Path("/a/b/c"))
    assert not scholar_agent._inside(root, Path("/a/x"))


# ── output_validator: pure helpers ───────────────────────────────────


def _decisions(actions: List[Dict[str, Any]]) -> ScholarDecisions:
    return ScholarDecisions.model_validate({"actions": actions, "interrupts_processed": []})


def test_validate_wikilinks_passes_for_resolvable(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    decisions = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": _valid_body("new", body_extra=" See [[global/python/use-uv]]."),
                "reasoning": "r",
            }
        ]
    )
    # Should not raise.
    scholar_agent._validate_wikilinks(decisions, k.resolve())


def test_validate_wikilinks_allows_batch_created_target(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    decisions = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/a.md",
                "body": _valid_body("a", body_extra=" Links to [[global/python/b]]."),
                "reasoning": "r",
            },
            {
                "action": "write_entry",
                "path": "global/python/b.md",
                "body": _valid_body("b"),
                "reasoning": "r",
            },
        ]
    )
    scholar_agent._validate_wikilinks(decisions, k.resolve())  # no raise


def test_validate_wikilinks_raises_modelretry_for_unresolvable(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    decisions = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/a.md",
                "body": _valid_body("a", body_extra=" Links to [[global/ghost/nope]]."),
                "reasoning": "r",
            }
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        scholar_agent._validate_wikilinks(decisions, k.resolve())
    assert "global/ghost/nope" in str(exc.value)


def test_validate_flat_dir_cap_passes_under_cap(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    decisions = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": _valid_body("new"),
                "reasoning": "r",
            }
        ]
    )
    scholar_agent._validate_flat_dir_caps(decisions, k.resolve())  # no raise


def test_validate_flat_dir_cap_raises_when_over(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    # use-uv already exists (1). Add 5 more writes → 6 > cap (5).
    actions = [
        {
            "action": "write_entry",
            "path": f"global/python/e{i}.md",
            "body": _valid_body(f"e{i}"),
            "reasoning": "r",
        }
        for i in range(5)
    ]
    with pytest.raises(ModelRetry) as exc:
        scholar_agent._validate_flat_dir_caps(_decisions(actions), k.resolve())
    assert "global/python" in str(exc.value)
    assert "cap is 5" in str(exc.value)


def test_flat_dir_cap_move_relieves_source(tmp_path: Path):
    """A move OUT of an over-cap dir into a fresh dir should not itself trip
    the cap (delta arithmetic decrements the source)."""
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    for i in range(5):
        (k / "global" / "python" / f"e{i}.md").write_text(_valid_body(f"e{i}"), encoding="utf-8")
    decisions = _decisions(
        [{"action": "move_entry", "from_path": "global/python/e0.md", "to_path": "global/db/e0.md"}]
    )
    scholar_agent._validate_flat_dir_caps(decisions, k.resolve())  # no raise


# ── FunctionModel-driven end-to-end through the agent ────────────────


def _output_tool_call(info: AgentInfo, decisions: Dict[str, Any]) -> ModelResponse:
    name = info.output_tools[0].name
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=decisions)])


def _good_write() -> Dict[str, Any]:
    return {
        "actions": [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": _valid_body("new"),
                "reasoning": "r",
            }
        ],
        "interrupts_processed": [],
    }


def test_agent_retries_then_succeeds_on_unresolvable_wikilink(tmp_path: Path):
    """The model first returns an action with an unresolvable wikilink; the
    output validator raises ModelRetry; the model is re-prompted and emits a
    clean action the second time."""
    k = _seed_clean_tree(tmp_path)
    calls: List[int] = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            bad = {
                "actions": [
                    {
                        "action": "write_entry",
                        "path": "global/python/new.md",
                        "body": _valid_body("new", body_extra=" [[global/ghost/x]]"),
                        "reasoning": "r",
                    }
                ],
                "interrupts_processed": [],
            }
            return _output_tool_call(info, bad)
        return _output_tool_call(info, _good_write())

    with SCHOLAR_AGENT.override(model=FunctionModel(fn)):
        result = SCHOLAR_AGENT.run_sync("go", deps=ScholarDeps(knowledge_dir=k.resolve()))
    assert len(calls) == 2  # one retry
    assert len(result.output.actions) == 1


def test_agent_retries_then_succeeds_on_bad_frontmatter(tmp_path: Path):
    """A body whose frontmatter is missing required fields is rejected by the
    per-action Pydantic validator → ModelRetry → fixed on the next emit."""
    k = _seed_clean_tree(tmp_path)
    calls: List[int] = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            bad = {
                "actions": [
                    {
                        "action": "write_entry",
                        "path": "global/python/new.md",
                        "body": "---\nid: new\nscope: global\n---\n\n# new\n\nbody here.\n",
                        "reasoning": "r",
                    }
                ],
                "interrupts_processed": [],
            }
            return _output_tool_call(info, bad)
        return _output_tool_call(info, _good_write())

    with SCHOLAR_AGENT.override(model=FunctionModel(fn)):
        result = SCHOLAR_AGENT.run_sync("go", deps=ScholarDeps(knowledge_dir=k.resolve()))
    assert len(calls) == 2
    assert len(result.output.actions) == 1


def test_agent_retries_then_succeeds_on_over_cap(tmp_path: Path):
    """Five writes into a dir already holding one entry trips the flat-dir
    cap; the model is re-prompted and trims to a single write."""
    k = _seed_clean_tree(tmp_path)
    calls: List[int] = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            over = {
                "actions": [
                    {
                        "action": "write_entry",
                        "path": f"global/python/e{i}.md",
                        "body": _valid_body(f"e{i}"),
                        "reasoning": "r",
                    }
                    for i in range(5)
                ],
                "interrupts_processed": [],
            }
            return _output_tool_call(info, over)
        return _output_tool_call(info, _good_write())

    with SCHOLAR_AGENT.override(model=FunctionModel(fn)):
        result = SCHOLAR_AGENT.run_sync("go", deps=ScholarDeps(knowledge_dir=k.resolve()))
    assert len(calls) == 2
    assert len(result.output.actions) == 1


def test_agent_gives_up_after_retry_budget(tmp_path: Path):
    """If the model never fixes the issue, Pydantic AI exhausts the output
    retry budget and raises — the daemon treats this as a dropped batch."""
    k = _seed_clean_tree(tmp_path)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        bad = {
            "actions": [
                {
                    "action": "write_entry",
                    "path": "global/python/new.md",
                    "body": _valid_body("new", body_extra=" [[global/ghost/x]]"),
                    "reasoning": "r",
                }
            ],
            "interrupts_processed": [],
        }
        return _output_tool_call(info, bad)

    with SCHOLAR_AGENT.override(model=FunctionModel(fn)):
        with pytest.raises(UnexpectedModelBehavior):
            SCHOLAR_AGENT.run_sync("go", deps=ScholarDeps(knowledge_dir=k.resolve()))


def test_agent_tools_are_callable_via_function_model(tmp_path: Path):
    """Drive the read-only tools through the agent so the registered tool
    bodies (read_entry / grep_library / bm25_search / embedding_search) run
    in-process at least once."""
    k = _seed_clean_tree(tmp_path)
    state = {"step": 0}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        state["step"] += 1
        if state["step"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="read_entry", args={"path": "global/python/use-uv.md"}),
                    ToolCallPart(tool_name="grep_library", args={"pattern": "real body"}),
                    ToolCallPart(tool_name="bm25_search", args={"query": "uv", "k": 3}),
                    ToolCallPart(tool_name="embedding_search", args={"query": "uv", "k": 3}),
                    ToolCallPart(tool_name="read_entry", args={"path": "../escape.md"}),
                ]
            )
        return _output_tool_call(info, {"actions": [], "interrupts_processed": []})

    with SCHOLAR_AGENT.override(model=FunctionModel(fn)):
        result = SCHOLAR_AGENT.run_sync("go", deps=ScholarDeps(knowledge_dir=k.resolve()))
    assert result.output.actions == []


# ── run wrapper + cost estimation ────────────────────────────────────


def test_run_scholar_agent_returns_decisions_and_cost(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        return _output_tool_call(info, _good_write())

    with SCHOLAR_AGENT.override(model=FunctionModel(fn)):
        decisions, cost = scholar_agent.run_scholar_agent("go", k.resolve(), timeout_s=30)
    assert len(decisions.actions) == 1
    assert isinstance(cost, float)
    assert cost >= 0.0


def test_estimate_cost_best_effort_on_bad_input():
    # Passing nonsense usage must not raise; returns 0.0 on failure.
    assert scholar_agent._estimate_cost("anthropic:claude-opus-4-7", object()) == 0.0


# ── extra tool-body coverage via FunctionModel ───────────────────────


def test_read_entry_handles_missing_and_outside(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    state = {"step": 0}
    seen: List[str] = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        state["step"] += 1
        if state["step"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="read_entry", args={"path": "global/python/ghost.md"}),
                    ToolCallPart(tool_name="read_entry", args={"path": "../../etc/passwd"}),
                ]
            )
        # Capture the tool returns from the second turn's message history.
        for m in messages:
            for part in getattr(m, "parts", []):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    seen.append(content)
        return _output_tool_call(info, {"actions": [], "interrupts_processed": []})

    with SCHOLAR_AGENT.override(model=FunctionModel(fn)):
        SCHOLAR_AGENT.run_sync("go", deps=ScholarDeps(knowledge_dir=k.resolve()))
    joined = " ".join(seen)
    assert "not found" in joined
    assert "outside the knowledge store" in joined


def test_grep_skips_archive(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "_archive").mkdir(parents=True)
    (k / "live.md").write_text("the needle here\n", encoding="utf-8")
    (k / "_archive" / "old.md").write_text("the needle here too\n", encoding="utf-8")
    out = scholar_agent._grep_library(k, "needle", "")
    assert "live.md" in out
    assert "_archive" not in out


# ── _apply_count_deltas (merge branch) ───────────────────────────────


def test_apply_count_deltas_merge(tmp_path: Path):
    k = _seed_clean_tree(tmp_path)
    decisions = _decisions(
        [
            {
                "action": "merge_entries",
                "source_paths": ["global/python/use-uv.md", "global/python/other.md"],
                "target_path": "global/python/merged.md",
                "target_body": _valid_body("merged"),
                "reasoning": "r",
            }
        ]
    )
    counts = scholar_agent._current_dir_counts(k.resolve())
    scholar_agent._apply_count_deltas(counts, decisions)
    # use-uv (1 existing) - 2 sources + 1 target → never negative, no over-cap.
    assert counts["global/python"] >= 0


# ── run wrapper timeout ──────────────────────────────────────────────


def test_run_scholar_agent_timeout(tmp_path: Path, monkeypatch):
    import asyncio

    from agent_mem_daemon.llm import LLMTimeout

    k = _seed_clean_tree(tmp_path)

    async def _slow(*a, **kw):
        await asyncio.sleep(10)

    monkeypatch.setattr(SCHOLAR_AGENT, "run", _slow)
    with pytest.raises(LLMTimeout):
        scholar_agent.run_scholar_agent("go", k.resolve(), timeout_s=0.05)
