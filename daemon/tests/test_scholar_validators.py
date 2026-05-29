"""Boundary-validator + run-wrapper tests for the typed Scholar (``scholar.py``).

Adapted from the reverted Pydantic-AI migration's ``test_scholar_agent.py``: the
wikilink / flat-dir-cap helpers now live in ``scholar`` and the validators raise
:class:`typed_agent.ModelRetry`. ``run_scholar_agent`` is tested by stubbing
``scholar.run_typed`` — no model call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent_mem_daemon import scholar
from agent_mem_daemon._schemas import ScholarDecisions
from agent_mem_daemon.scholar import ScholarDeps
from agent_mem_daemon.typed_agent import ModelRetry, TypedResult

from .conftest import scholar_entry_body, seed_scholar_tree


def _decisions(actions: List[Dict[str, Any]]) -> ScholarDecisions:
    return ScholarDecisions.model_validate({"actions": actions, "interrupts_processed": []})


def _body(id_: str, *, extra: str = "") -> str:
    return scholar_entry_body(id_, extra=extra)


# ── _validate_wikilinks ──────────────────────────────────────────────────────


def test_wikilinks_pass_for_resolvable(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    d = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": _body("new", extra=" See [[global/python/use-uv]]."),
                "reasoning": "r",
            }
        ]
    )
    scholar._validate_wikilinks(d, k.resolve())  # no raise


def test_wikilinks_allow_batch_created_target(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    d = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/a.md",
                "body": _body("a", extra=" Links to [[global/python/b]]."),
                "reasoning": "r",
            },
            {
                "action": "write_entry",
                "path": "global/python/b.md",
                "body": _body("b"),
                "reasoning": "r",
            },
        ]
    )
    scholar._validate_wikilinks(d, k.resolve())  # no raise


def test_wikilinks_raise_for_unresolvable(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    d = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/a.md",
                "body": _body("a", extra=" Links to [[global/ghost/nope]]."),
                "reasoning": "r",
            }
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        scholar._validate_wikilinks(d, k.resolve())
    assert "global/ghost/nope" in str(exc.value)


# ── _validate_flat_dir_caps ──────────────────────────────────────────────────


def test_flat_dir_cap_passes_under(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    d = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": _body("new"),
                "reasoning": "r",
            }
        ]
    )
    scholar._validate_flat_dir_caps(d, k.resolve())  # no raise


def test_flat_dir_cap_raises_when_over(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)  # use-uv.md already there (1); +5 → 6 > cap (5)
    actions = [
        {
            "action": "write_entry",
            "path": f"global/python/e{i}.md",
            "body": _body(f"e{i}"),
            "reasoning": "r",
        }
        for i in range(5)
    ]
    with pytest.raises(ModelRetry) as exc:
        scholar._validate_flat_dir_caps(_decisions(actions), k.resolve())
    assert "global/python" in str(exc.value)


def test_flat_dir_cap_move_relieves_source(tmp_path: Path) -> None:
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    for i in range(5):
        (k / "global" / "python" / f"e{i}.md").write_text(_body(f"e{i}"), encoding="utf-8")
    d = _decisions(
        [{"action": "move_entry", "from_path": "global/python/e0.md", "to_path": "global/db/e0.md"}]
    )
    scholar._validate_flat_dir_caps(d, k.resolve())  # no raise — source decremented


# ── validate_decisions (public whole-batch validator) ────────────────────────


def test_validate_decisions_returns_output_when_clean(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    d = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": _body("new"),
                "reasoning": "r",
            }
        ]
    )
    assert scholar.validate_decisions(ScholarDeps(knowledge_dir=k), d) is d


def test_validate_decisions_raises_on_bad_wikilink(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    d = _decisions(
        [
            {
                "action": "write_entry",
                "path": "global/python/a.md",
                "body": _body("a", extra=" [[global/ghost/x]]"),
                "reasoning": "r",
            }
        ]
    )
    with pytest.raises(ModelRetry):
        scholar.validate_decisions(ScholarDeps(knowledge_dir=k), d)


# ── run_scholar_agent (stub run_typed; no model call) ────────────────────────


def test_run_scholar_agent_returns_decisions_and_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    k = seed_scholar_tree(tmp_path)
    decisions = _decisions([])

    async def fake_run_typed(*_a: Any, **_kw: Any) -> TypedResult:
        return TypedResult(output=decisions, cost_usd=0.05, attempts=1)

    monkeypatch.setattr(scholar, "run_typed", fake_run_typed)
    out, cost = scholar.run_scholar_agent("prompt", k, timeout_s=5.0)
    assert out is decisions
    assert cost == 0.05


def test_run_scholar_agent_timeout_maps_to_llmtimeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    k = seed_scholar_tree(tmp_path)

    async def slow_run_typed(*_a: Any, **_kw: Any) -> TypedResult:
        await asyncio.sleep(10.0)
        raise AssertionError("should have timed out")  # pragma: no cover

    monkeypatch.setattr(scholar, "run_typed", slow_run_typed)
    with pytest.raises(scholar.LLMTimeout):
        scholar.run_scholar_agent("prompt", k, timeout_s=0.05)


# ── all action kinds (body/path extraction, created-paths, count deltas) ─────


def test_validators_exercise_all_action_kinds(tmp_path: Path) -> None:
    """update / merge / move / archive over distinct entries exercise the
    update_entry / merge_entries branches of ``_action_body_and_path`` +
    ``_created_paths`` and the move / merge / archive branches of
    ``_apply_count_deltas`` — all under cap with resolvable links, so neither
    validator raises."""
    k = seed_scholar_tree(tmp_path)
    for name in ("m1", "m2", "mv", "ar", "up"):
        (k / "global" / "python" / f"{name}.md").write_text(_body(name), encoding="utf-8")
    d = _decisions(
        [
            {
                "action": "update_entry",
                "path": "global/python/up.md",
                "new_body": _body("up", extra=" See [[global/python/use-uv]]."),
                "reasoning": "r",
            },
            {
                "action": "merge_entries",
                "source_paths": ["global/python/m1.md", "global/python/m2.md"],
                "target_path": "global/python/merged.md",
                "target_body": _body("merged"),
                "reasoning": "r",
            },
            {
                "action": "move_entry",
                "from_path": "global/python/mv.md",
                "to_path": "global/db/mv.md",
                "reasoning": "r",
            },
            {"action": "archive_entry", "path": "global/python/ar.md", "reasoning": "r"},
        ]
    )
    scholar.validate_decisions(ScholarDeps(knowledge_dir=k), d)  # no raise


def test_run_scholar_agent_sets_recursion_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every curator SDK call MUST carry CLAUDE_INVOKED_BY in its env, so the
    Claude process the SDK spawns sees it and its hooks bail — otherwise the
    daemon ingests its own model calls and loops. Regression guard for that."""
    k = seed_scholar_tree(tmp_path)
    captured: dict[str, Any] = {}

    async def fake_run_typed(*_a: Any, **kw: Any) -> TypedResult:
        captured.update(kw)
        return TypedResult(output=_decisions([]), cost_usd=0.0, attempts=1)

    monkeypatch.setattr(scholar, "run_typed", fake_run_typed)
    scholar.run_scholar_agent("p", k, timeout_s=5.0)
    assert captured.get("env", {}).get("CLAUDE_INVOKED_BY") == "agent_mem_daemon"
