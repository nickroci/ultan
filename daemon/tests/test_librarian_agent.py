"""Tests for the Pydantic-AI Librarian agent: typed proposal output, the
boundary output-validator (well-formed paths + parseable bodies), the
ModelRetry feedback loop, the shared read-only tools, and the run wrapper.

No real Anthropic calls — we drive the agent with Pydantic AI's
``FunctionModel`` (a scripted model) via ``agent.override(model=...)``,
and exercise the pure validator helpers directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_mem_daemon import librarian_agent
from agent_mem_daemon._agent_common import ResearchDeps
from agent_mem_daemon._schemas import LibrarianProposal
from agent_mem_daemon.librarian_agent import LIBRARIAN_AGENT, LibrarianDeps

from .conftest import scholar_entry_body, seed_scholar_tree


def _proposal(obj: Dict[str, Any]) -> LibrarianProposal:
    return LibrarianProposal.model_validate(obj)


# ── output_validator: pure helpers ───────────────────────────────────


def _one(proposal: Dict[str, Any]) -> LibrarianProposal:
    return _proposal({"proposals": [proposal], "interrupts": []})


def test_validate_paths_passes_for_wellformed(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    out = _one(
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": scholar_entry_body("new"),
            "reasoning": "r",
        }
    )
    # No raise.
    for p in out.proposals:
        librarian_agent._validate_proposal_paths(p, k.resolve())
        librarian_agent._validate_proposal_body(p)


def test_validate_paths_rejects_absolute(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    out = _one({"action": "write_entry", "path": "/etc/passwd.md", "body": "", "reasoning": "r"})
    with pytest.raises(ModelRetry) as exc:
        librarian_agent._validate_proposal_paths(out.proposals[0], k.resolve())
    assert "RELATIVE" in str(exc.value)


def test_validate_paths_rejects_escape(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    out = _one({"action": "archive_entry", "path": "../../secrets.md", "reasoning": "r"})
    with pytest.raises(ModelRetry) as exc:
        librarian_agent._validate_proposal_paths(out.proposals[0], k.resolve())
    assert "OUTSIDE" in str(exc.value)


def test_validate_paths_rejects_non_md(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    out = _one({"action": "write_entry", "path": "global/python/new", "body": "", "reasoning": "r"})
    with pytest.raises(ModelRetry) as exc:
        librarian_agent._validate_proposal_paths(out.proposals[0], k.resolve())
    assert ".md" in str(exc.value)


def test_validate_paths_checks_move_endpoints(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    out = _one(
        {
            "action": "move_entry",
            "from_path": "global/a.md",
            "to_path": "/abs/b.md",
            "reasoning": "r",
        }
    )
    with pytest.raises(ModelRetry) as exc:
        librarian_agent._validate_proposal_paths(out.proposals[0], k.resolve())
    assert "to_path" in str(exc.value)


def test_validate_body_rejects_unparseable_frontmatter(tmp_path: Path) -> None:
    out = _one(
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": "no frontmatter here, just prose",
            "reasoning": "r",
        }
    )
    with pytest.raises(ModelRetry) as exc:
        librarian_agent._validate_proposal_body(out.proposals[0])
    assert "frontmatter" in str(exc.value)


def test_validate_body_allows_empty_body(tmp_path: Path) -> None:
    """A thin write with no body is left to the Scholar — only a NON-empty
    unparseable body is a boundary defect."""
    out = _one(
        {"action": "write_entry", "path": "global/python/new.md", "body": "", "reasoning": "r"}
    )
    librarian_agent._validate_proposal_body(out.proposals[0])  # no raise


def test_validate_body_merge_uses_target_body(tmp_path: Path) -> None:
    out = _one(
        {
            "action": "merge_entries",
            "source_paths": ["global/a.md"],
            "target_path": "global/m.md",
            "target_body": "garbage with no fm",
            "reasoning": "r",
        }
    )
    with pytest.raises(ModelRetry):
        librarian_agent._validate_proposal_body(out.proposals[0])


def test_validate_body_skips_non_body_actions(tmp_path: Path) -> None:
    out = _one({"action": "archive_entry", "path": "global/a.md", "reasoning": "r"})
    librarian_agent._validate_proposal_body(out.proposals[0])  # no raise


def test_entry_target_paths_covers_every_action_kind(tmp_path: Path) -> None:
    """Each action kind contributes its entry path(s) to the well-formedness
    sweep (merge source_paths, deprecate's superseded_by, add_wikilink
    endpoints). All well-formed here → no raise."""
    k = seed_scholar_tree(tmp_path).resolve()
    cases = [
        {"action": "update_entry", "path": "global/python/u.md", "new_body": "", "reasoning": "r"},
        {
            "action": "merge_entries",
            "source_paths": ["global/python/a.md", "global/python/b.md"],
            "target_path": "global/python/m.md",
            "target_body": "",
            "reasoning": "r",
        },
        {
            "action": "deprecate_entry",
            "path": "global/python/old.md",
            "superseded_by": "global/python/new.md",
            "reasoning": "r",
        },
        {
            "action": "add_wikilink",
            "from_path": "global/python/x.md",
            "to_path": "global/python/y.md",
            "context": "see also",
            "reasoning": "r",
        },
    ]
    for case in cases:
        prop = _one(case).proposals[0]
        # Every path is well-formed, so the sweep returns at least one pair
        # and raises nothing.
        assert librarian_agent._entry_target_paths(prop)
        librarian_agent._validate_proposal_paths(prop, k)


def test_entry_target_paths_skips_folder_only_actions(tmp_path: Path) -> None:
    """update_readme / split_folder carry folder paths, not entry .md
    targets, so the entry-path sweep yields nothing for them."""
    for case in (
        {
            "action": "update_readme",
            "folder_path": "global/python",
            "new_body": "x",
            "reasoning": "r",
        },
        {
            "action": "split_folder",
            "folder_path": "global/python",
            "into": {"sub": ["global/python/a.md"]},
            "reasoning": "r",
        },
    ):
        prop = _one(case).proposals[0]
        assert librarian_agent._entry_target_paths(prop) == []


def test_validate_body_update_entry_uses_new_body(tmp_path: Path) -> None:
    out = _one(
        {
            "action": "update_entry",
            "path": "global/python/u.md",
            "new_body": "prose, no frontmatter",
            "reasoning": "r",
        }
    )
    with pytest.raises(ModelRetry):
        librarian_agent._validate_proposal_body(out.proposals[0])


def test_validate_paths_rejects_merge_source(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path).resolve()
    out = _one(
        {
            "action": "merge_entries",
            "source_paths": ["/abs/a.md"],
            "target_path": "global/python/m.md",
            "target_body": "",
            "reasoning": "r",
        }
    )
    with pytest.raises(ModelRetry) as exc:
        librarian_agent._validate_proposal_paths(out.proposals[0], k)
    assert "source_paths" in str(exc.value)


# ── FunctionModel-driven end-to-end through the agent ────────────────


def _output_tool_call(info: AgentInfo, proposal: Dict[str, Any]) -> ModelResponse:
    name = info.output_tools[0].name
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=proposal)])


def _good_proposal() -> Dict[str, Any]:
    return {
        "proposals": [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": scholar_entry_body("new"),
                "reasoning": "buffer turn [1] stated a preference",
            }
        ],
        "interrupts": [],
    }


def test_agent_returns_typed_proposal(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        return _output_tool_call(info, _good_proposal())

    with LIBRARIAN_AGENT.override(model=FunctionModel(fn)):
        result = LIBRARIAN_AGENT.run_sync("go", deps=LibrarianDeps(knowledge_dir=k.resolve()))
    assert isinstance(result.output, LibrarianProposal)
    assert len(result.output.proposals) == 1
    assert result.output.proposals[0].action == "write_entry"


def test_agent_retries_then_succeeds_on_bad_path(tmp_path: Path) -> None:
    """First emit has an absolute path → the output validator raises
    ModelRetry → the model is re-prompted and emits a clean proposal."""
    k = seed_scholar_tree(tmp_path)
    calls: List[int] = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            bad = {
                "proposals": [
                    {
                        "action": "write_entry",
                        "path": "/abs/new.md",
                        "body": scholar_entry_body("new"),
                        "reasoning": "r",
                    }
                ],
                "interrupts": [],
            }
            return _output_tool_call(info, bad)
        return _output_tool_call(info, _good_proposal())

    with LIBRARIAN_AGENT.override(model=FunctionModel(fn)):
        result = LIBRARIAN_AGENT.run_sync("go", deps=LibrarianDeps(knowledge_dir=k.resolve()))
    assert len(calls) == 2
    assert len(result.output.proposals) == 1


def test_agent_retries_then_succeeds_on_unparseable_body(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    calls: List[int] = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            bad = {
                "proposals": [
                    {
                        "action": "write_entry",
                        "path": "global/python/new.md",
                        "body": "prose with no frontmatter block",
                        "reasoning": "r",
                    }
                ],
                "interrupts": [],
            }
            return _output_tool_call(info, bad)
        return _output_tool_call(info, _good_proposal())

    with LIBRARIAN_AGENT.override(model=FunctionModel(fn)):
        result = LIBRARIAN_AGENT.run_sync("go", deps=LibrarianDeps(knowledge_dir=k.resolve()))
    assert len(calls) == 2
    assert len(result.output.proposals) == 1


def test_agent_gives_up_after_retry_budget(tmp_path: Path) -> None:
    """If the model never fixes the malformed path, Pydantic AI exhausts the
    output retry budget and raises — the daemon treats this as an empty
    packet (caught by ``librarian.scan``)."""
    k = seed_scholar_tree(tmp_path)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        bad = {
            "proposals": [
                {"action": "write_entry", "path": "/abs/new.md", "body": "", "reasoning": "r"}
            ],
            "interrupts": [],
        }
        return _output_tool_call(info, bad)

    with LIBRARIAN_AGENT.override(model=FunctionModel(fn)):
        with pytest.raises(UnexpectedModelBehavior):
            LIBRARIAN_AGENT.run_sync("go", deps=LibrarianDeps(knowledge_dir=k.resolve()))


def test_agent_emits_empty_proposal(tmp_path: Path) -> None:
    """Empty proposals + interrupts is a valid output (recall layer found
    nothing worth filing)."""
    k = seed_scholar_tree(tmp_path)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        return _output_tool_call(info, {"proposals": [], "interrupts": []})

    with LIBRARIAN_AGENT.override(model=FunctionModel(fn)):
        result = LIBRARIAN_AGENT.run_sync("go", deps=LibrarianDeps(knowledge_dir=k.resolve()))
    assert result.output.proposals == []
    assert result.output.interrupts == []


def test_librarian_agent_registers_the_research_tools() -> None:
    """The Librarian shares the read-only tool surface with the Scholar.
    Confirm all four are registered on the agent (the shared registration in
    ``_agent_common``; the tool bodies themselves are exercised in
    ``test_agent_common`` and through the Scholar's FunctionModel test)."""
    names = set(LIBRARIAN_AGENT._function_toolset.tools)  # noqa: SLF001 — introspection
    assert {"read_entry", "grep_library", "bm25_search", "embedding_search"} <= names


def test_agent_proposes_repair_when_prompted(tmp_path: Path) -> None:
    """The integrity-repair flow: given a repair task in the prompt, the
    Librarian proposes an EXISTING action (here update_entry). Drive it via
    FunctionModel so the typed repair proposal round-trips."""
    k = seed_scholar_tree(tmp_path)
    repair_body = scholar_entry_body("use-uv", extra=" See [[global/python/use-uv]].")

    def fn(messages, info: AgentInfo) -> ModelResponse:
        return _output_tool_call(
            info,
            {
                "proposals": [
                    {
                        "action": "update_entry",
                        "path": "global/python/use-uv.md",
                        "new_body": repair_body,
                        "reasoning": "repair task: rewrite broken [[ghost]] link",
                        "salience_signal": None,
                    }
                ],
                "interrupts": [],
            },
        )

    with LIBRARIAN_AGENT.override(model=FunctionModel(fn)):
        result = LIBRARIAN_AGENT.run_sync(
            "repair the link", deps=LibrarianDeps(knowledge_dir=k.resolve())
        )
    assert len(result.output.proposals) == 1
    assert result.output.proposals[0].action == "update_entry"


# ── deps type is the shared ResearchDeps ─────────────────────────────


def test_librarian_deps_is_research_deps() -> None:
    assert LibrarianDeps is ResearchDeps


# ── run wrapper + cost + timeout ─────────────────────────────────────


def test_run_librarian_agent_returns_proposal_and_cost(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        return _output_tool_call(info, _good_proposal())

    with LIBRARIAN_AGENT.override(model=FunctionModel(fn)):
        proposal, cost = librarian_agent.run_librarian_agent("go", k.resolve(), timeout_s=30)
    assert len(proposal.proposals) == 1
    assert isinstance(cost, float)
    assert cost >= 0.0


def test_run_librarian_agent_timeout(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    from agent_mem_daemon.llm import LLMTimeout

    k = seed_scholar_tree(tmp_path)

    async def _slow(*a, **kw):
        await asyncio.sleep(10)

    monkeypatch.setattr(LIBRARIAN_AGENT, "run", _slow)
    with pytest.raises(LLMTimeout):
        librarian_agent.run_librarian_agent("go", k.resolve(), timeout_s=0.05)
