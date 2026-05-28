"""Behavioural regression test for ``scripts/query.py``.

SDK-heavy, so this is a single happy-path test: the imported async
``query`` symbol is monkeypatched to an async generator yielding real
``AssistantMessage`` / ``ResultMessage`` instances, and we assert that
``run_query`` returns the synthesized text and persists the state bump
(query_count + total_cost) via ``load_state`` / ``save_state``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

import query as query_mod
import utils
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock


def _fake_query_factory(answer: str, cost: float):
    """Build a stand-in for the SDK ``query`` async generator.

    Yields one ``AssistantMessage`` carrying ``answer`` then one
    ``ResultMessage`` carrying ``cost`` — the only two message shapes
    ``run_query`` inspects.
    """

    async def _fake_query(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        yield AssistantMessage(content=[TextBlock(text=answer)], model="fake")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sess",
            total_cost_usd=cost,
        )

    return _fake_query


def test_run_query_returns_answer_and_bumps_state(store_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(query_mod, "query", _fake_query_factory("the answer", 0.25))

    # Seed prior state so we can assert the increments rather than absolute values.
    utils.save_state({"ingested": {}, "query_count": 4, "last_lint": None, "total_cost": 1.0})

    answer = asyncio.run(query_mod.run_query("how do I X?"))

    assert answer == "the answer"
    state = utils.load_state()
    assert state["query_count"] == 5
    assert state["total_cost"] == 1.25


def test_run_query_starts_state_from_scratch(store_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(query_mod, "query", _fake_query_factory("hello", 0.0))
    answer = asyncio.run(query_mod.run_query("first ever question"))
    assert answer == "hello"
    state = utils.load_state()
    assert state["query_count"] == 1
