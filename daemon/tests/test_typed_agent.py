"""Tests for the typed-agent shim (``typed_agent.py``).

Two layers:
  * ``evaluate_submission`` — the validate→accept/retry core — is tested
    directly; it is pure and SDK-free.
  * ``run_typed`` — the agentic loop — is tested with an injected fake
    ``_query`` plus a shared ``_state``. The fake drives ``evaluate_submission``
    exactly as the real ``submit_result`` handler would, so we exercise the loop
    (cost extraction, submitted-detection, success vs. ``TypedAgentError``)
    without ever calling the subscription-billed model.

Async tests run under pytest-asyncio's auto mode (``asyncio_mode = "auto"``).
"""

from __future__ import annotations

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, ToolUseBlock
from pydantic import BaseModel, field_validator

from agent_mem_daemon.typed_agent import (
    ModelRetry,
    TypedAgentError,
    _RunState,
    _single_user_message,
    evaluate_submission,
    run_typed,
    submit_tool_ref,
)


class Decision(BaseModel):
    n: int
    label: str

    @field_validator("n")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v


# ── evaluate_submission: the pure validate→accept/retry core ─────────────────


def test_valid_submission_stashes_typed_result() -> None:
    st: _RunState[Decision] = _RunState()
    out = evaluate_submission({"n": 3, "label": "x"}, Decision, None, [], st, 4)
    assert "is_error" not in out
    assert st.result == Decision(n=3, label="x")
    assert st.attempts == 1


def test_structural_error_returns_is_error_naming_the_field() -> None:
    st: _RunState[Decision] = _RunState()
    out = evaluate_submission({"n": -1, "label": "x"}, Decision, None, [], st, 4)
    assert out["is_error"] is True
    assert "n" in out["content"][0]["text"]
    assert st.result is None


def test_missing_required_field_is_rejected() -> None:
    st: _RunState[Decision] = _RunState()
    out = evaluate_submission({"n": 1}, Decision, None, [], st, 4)
    assert out["is_error"] is True
    assert "label" in out["content"][0]["text"]


def test_model_retry_from_validator_feeds_its_message_back() -> None:
    def even_only(_deps: object, o: Decision) -> Decision:
        if o.n % 2:
            raise ModelRetry("n must be even")
        return o

    st: _RunState[Decision] = _RunState()
    out = evaluate_submission({"n": 3, "label": "x"}, Decision, None, [even_only], st, 4)
    assert out["is_error"] is True
    assert "even" in out["content"][0]["text"]
    assert st.result is None


def test_validator_may_normalise_the_output() -> None:
    def upper(_deps: object, o: Decision) -> Decision:
        return o.model_copy(update={"label": o.label.upper()})

    st: _RunState[Decision] = _RunState()
    evaluate_submission({"n": 1, "label": "x"}, Decision, None, [upper], st, 4)
    assert st.result is not None
    assert st.result.label == "X"


def test_exhausted_budget_tells_model_to_stop() -> None:
    st: _RunState[Decision] = _RunState()
    st.attempts = 4  # this call becomes attempt 5, beyond max_retries=4
    out = evaluate_submission({"n": -1, "label": "x"}, Decision, None, [], st, 4)
    assert out["is_error"] is True
    assert "exhausted" in out["content"][0]["text"].lower()


# ── run_typed: the agentic loop (fake _query drives the shared state) ─────────


def _assistant_calling_submit() -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id="t1", name=submit_tool_ref(), input={})],
        model="test",
    )


def _result(cost: float) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        total_cost_usd=cost,
    )


async def test_run_typed_happy_path_returns_typed_result_and_cost() -> None:
    shared: _RunState[Decision] = _RunState()

    async def fake_query(*, prompt: object, options: object):  # noqa: ARG001
        evaluate_submission({"n": 7, "label": "ok"}, Decision, None, [], shared, 4)
        yield _assistant_calling_submit()
        yield _result(0.02)

    res = await run_typed(
        "p",
        Decision,
        deps=None,
        system_prompt="s",
        model="m",
        mcp_servers={},
        allowed_tools=[],
        _query=fake_query,
        _state=shared,
    )
    assert res.output == Decision(n=7, label="ok")
    assert res.cost_usd == 0.02
    assert res.attempts == 1


async def test_run_typed_never_submitted_raises_typed_agent_error() -> None:
    shared: _RunState[Decision] = _RunState()

    async def fake_query(*, prompt: object, options: object):  # noqa: ARG001
        yield _result(0.0)  # model produced a text answer; never called submit

    with pytest.raises(TypedAgentError) as exc:
        await run_typed(
            "p",
            Decision,
            deps=None,
            system_prompt="s",
            model="m",
            mcp_servers={},
            allowed_tools=[],
            _query=fake_query,
            _state=shared,
        )
    assert "never called submit_result" in str(exc.value)
    assert exc.value.attempts == 0


async def test_run_typed_submitted_but_invalid_reports_last_error() -> None:
    shared: _RunState[Decision] = _RunState()

    async def fake_query(*, prompt: object, options: object):  # noqa: ARG001
        evaluate_submission({"n": -1, "label": "x"}, Decision, None, [], shared, 4)
        yield _assistant_calling_submit()
        yield _result(0.0)

    with pytest.raises(TypedAgentError) as exc:
        await run_typed(
            "p",
            Decision,
            deps=None,
            system_prompt="s",
            model="m",
            mcp_servers={},
            allowed_tools=[],
            _query=fake_query,
            _state=shared,
        )
    assert exc.value.last_error is not None
    assert "never called" not in str(exc.value)  # it DID submit, just invalid


async def test_run_typed_retry_then_succeed() -> None:
    shared: _RunState[Decision] = _RunState()

    async def fake_query(*, prompt: object, options: object):  # noqa: ARG001
        evaluate_submission({"n": -1, "label": "x"}, Decision, None, [], shared, 4)  # rejected
        evaluate_submission({"n": 1, "label": "x"}, Decision, None, [], shared, 4)  # accepted
        yield _assistant_calling_submit()
        yield _result(0.01)

    res = await run_typed(
        "p",
        Decision,
        deps=None,
        system_prompt="s",
        model="m",
        mcp_servers={},
        allowed_tools=[],
        _query=fake_query,
        _state=shared,
    )
    assert res.output == Decision(n=1, label="x")
    assert res.attempts == 2


def test_submit_tool_ref_is_namespaced() -> None:
    assert submit_tool_ref() == "mcp__agent_mem_output__submit_result"


async def test_single_user_message_yields_one_streaming_user_message() -> None:
    msgs = [m async for m in _single_user_message("hi")]
    assert msgs == [{"type": "user", "message": {"role": "user", "content": "hi"}}]
