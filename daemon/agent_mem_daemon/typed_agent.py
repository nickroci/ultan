"""Typed, validated agent output over ``claude_agent_sdk`` — a tiny shim that
gives the daemon's curator roles the one Pydantic-AI behaviour we actually want
(typed output + model self-correction) WITHOUT leaving the Claude Code
subscription for the metered Anthropic API.

Why this exists
---------------
The daemon must authenticate via ``claude-agent-sdk`` (subscription), never the
metered API. Pydantic-AI's Anthropic provider only speaks the metered API, so
we can't use it. But we still want its boundary discipline: the model returns a
*typed* result, a validator rejects bad content, and the model re-emits — rather
than us scraping JSON out of free text and repairing it after the fact.

How it works (mirrors how Pydantic-AI talks to Anthropic)
---------------------------------------------------------
Pydantic-AI sends the output model as a tool whose ``input_schema`` is literally
``OutputModel.model_json_schema()`` (see its ``models/anthropic.py``
``_map_tool_definition``). The Claude Agent SDK accepts the same thing: an MCP
``@tool`` takes a raw JSON-Schema dict. So:

  1. We register an in-process MCP server with a ``submit_result`` tool whose
     ``input_schema`` is ``output_type.model_json_schema()`` — alongside the
     caller's read-only research tools.
  2. The system prompt instructs the model to verify with the research tools,
     then call ``submit_result`` exactly once with its final answer.
  3. The handler runs ``output_type.model_validate(args)`` then the supplied
     validators (which may raise :class:`ModelRetry`). On success it stashes the
     typed object; on failure it returns the error AS the tool result
     (``is_error``), so the agentic loop continues and the model self-corrects
     in-band. Bad data never escapes the boundary.
  4. After the run we return the validated object, or raise
     :class:`TypedAgentError` if the model never produced a valid result within
     the retry budget — the caller then drops the batch and the lessons recur.

We cannot *force* the ``submit_result`` call: Anthropic can't force a specific
tool while still allowing free research-tool use, and forcing is incompatible
with extended thinking — Pydantic-AI falls back to instruct-and-retry the same
way. So "never submitted" is just a terminal validation failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from claude_agent_sdk.types import McpServerConfig

log = logging.getLogger("agent_mem_daemon.typed_agent")

OutputT = TypeVar("OutputT", bound=BaseModel)
DepsT = TypeVar("DepsT")

# A validator runs after structural ``model_validate``. It returns the (possibly
# normalised) object or raises ``ModelRetry`` with a corrective message that is
# fed back to the model. Mirrors a Pydantic-AI ``@output_validator``.
Validator = Callable[[DepsT, OutputT], OutputT]

# The output-submission tool lives in its own MCP server so it never collides
# with a role's research-tool server.
SUBMIT_SERVER_NAME = "agent_mem_output"
SUBMIT_TOOL_NAME = "submit_result"


def submit_tool_ref() -> str:
    """Fully-qualified ``allowed_tools`` name for the submit tool."""
    return f"mcp__{SUBMIT_SERVER_NAME}__{SUBMIT_TOOL_NAME}"


# ── Exceptions ───────────────────────────────────────────────────────────────


class ModelRetry(Exception):  # noqa: N818  (matches pydantic_ai.ModelRetry's name)
    """Raised by a validator to bounce the result back to the model with a
    corrective message, exactly like ``pydantic_ai.ModelRetry``."""


class TypedAgentError(Exception):
    """The model never produced a result that passed validation within the
    retry budget (or never called ``submit_result``). The caller should drop
    the batch — there is deliberately no partial/repaired result."""

    def __init__(self, message: str, *, attempts: int, last_error: str | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# ── Result + internal run state ──────────────────────────────────────────────


@dataclass
class TypedResult(Generic[OutputT]):
    """A successful run: the validated output plus run metadata."""

    output: OutputT
    cost_usd: float
    attempts: int


@dataclass
class _RunState(Generic[OutputT]):
    """Mutable state shared between the submit-tool handler and the run loop."""

    result: OutputT | None = None
    attempts: int = 0
    last_error: str | None = None
    cost_usd: float = 0.0


# ── The validate→accept-or-retry core (pure; unit-tested without the SDK) ─────


def _format_validation_error(exc: ValidationError) -> str:
    """One actionable line per failing field — concise enough to fit in a tool
    result, specific enough for the model to fix."""
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        lines.append(f"  - {loc}: {err.get('msg', 'invalid')}")
    return "field validation failed:\n" + "\n".join(lines)


def evaluate_submission(
    args: dict[str, Any],
    output_type: type[OutputT],
    deps: DepsT,
    validators: Sequence[Validator[DepsT, OutputT]],
    state: _RunState[OutputT],
    max_retries: int,
) -> dict[str, Any]:
    """Validate one ``submit_result`` payload and return the MCP tool result.

    On success: stashes the typed object on ``state`` and returns an accept
    message. On failure: records the error and returns an ``is_error`` result
    that tells the model how to fix it (or that the budget is exhausted). This
    is the whole retry mechanism, kept SDK-free so it can be tested directly.
    """
    state.attempts += 1
    try:
        obj = output_type.model_validate(args)
        for validator in validators:
            obj = validator(deps, obj)
    except ValidationError as exc:
        state.last_error = _format_validation_error(exc)
        return _retry_result(state, max_retries)
    except ModelRetry as exc:
        state.last_error = str(exc)
        return _retry_result(state, max_retries)

    state.result = obj
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    "Accepted — your result was recorded and validated. "
                    "You are done; do not call any more tools."
                ),
            }
        ]
    }


def _retry_result(state: _RunState[Any], max_retries: int) -> dict[str, Any]:
    """The ``is_error`` tool result sent back when a submission is rejected."""
    if state.attempts > max_retries:
        text = (
            f"Submission rejected and the retry budget ({max_retries}) is "
            f"exhausted. Stop now — do not call submit_result again.\n"
            f"Last error: {state.last_error}"
        )
    else:
        text = (
            f"Your submission was rejected:\n{state.last_error}\n\n"
            f"Fix exactly that and call {SUBMIT_TOOL_NAME} again with corrected "
            f"input. (attempt {state.attempts} of {max_retries + 1})"
        )
    return {"content": [{"type": "text", "text": text}], "is_error": True}


# ── SDK wiring ───────────────────────────────────────────────────────────────


def _build_submit_server(
    output_type: type[OutputT],
    deps: DepsT,
    validators: Sequence[Validator[DepsT, OutputT]],
    state: _RunState[OutputT],
    max_retries: int,
) -> "McpServerConfig":
    """In-process MCP server exposing only ``submit_result``, whose input schema
    is the output model's JSON schema (the model thus sees the exact typed
    shape, the same way Pydantic-AI hands Anthropic the schema)."""
    from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: PLC0415

    schema = output_type.model_json_schema()

    @tool(
        SUBMIT_TOOL_NAME,
        (
            "Submit your final answer. Call this EXACTLY ONCE, at the end, with "
            "the complete result matching the schema. If the result is rejected "
            "you will be told why — fix it and call this tool again."
        ),
        schema,
    )
    async def submit_result(args: dict[str, Any]) -> dict[str, Any]:
        return evaluate_submission(args, output_type, deps, validators, state, max_retries)

    return create_sdk_mcp_server(name=SUBMIT_SERVER_NAME, tools=[submit_result])


async def _single_user_message(prompt: str) -> AsyncIterator[dict[str, Any]]:
    """Yield the prompt as a one-message stream. The SDK requires streaming-mode
    input for tool-enabled runs; a single yielded message is equivalent to
    one-shot mode. Mirrors ``llm._drain_query``."""
    yield {"type": "user", "message": {"role": "user", "content": prompt}}


# Injection seam: the default is the real SDK ``query``; tests pass a fake that
# drives the registered ``submit_result`` handler and yields a ``ResultMessage``.
_QueryFn = Callable[..., Any]


async def run_typed(
    prompt: str,
    output_type: type[OutputT],
    *,
    deps: DepsT,
    system_prompt: str,
    model: str,
    mcp_servers: dict[str, "McpServerConfig"],
    allowed_tools: Sequence[str],
    validators: Sequence[Validator[DepsT, OutputT]] = (),
    max_retries: int = 4,
    max_turns: int = 20,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    _query: _QueryFn | None = None,
    _state: "_RunState[OutputT] | None" = None,
) -> TypedResult[OutputT]:
    """Run a typed agent turn and return the validated ``output_type`` instance.

    ``mcp_servers`` / ``allowed_tools`` carry the role's read-only research tools;
    this function adds the ``submit_result`` server/tool. The model writes
    nothing to disk — it returns a typed result that a deterministic executor
    applies — so no file-write tools or path guards are wired here.

    Raises :class:`TypedAgentError` if no valid result is produced within
    ``max_retries`` (or the model never calls ``submit_result``).
    """
    from claude_agent_sdk import (  # noqa: PLC0415
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        ToolUseBlock,
        query,
    )

    # ``_state`` is a test seam: callers never pass it, but tests inject a shared
    # state so a fake ``_query`` can drive ``evaluate_submission`` (what the real
    # handler does) against the same object this loop inspects.
    state: _RunState[OutputT] = _state if _state is not None else _RunState()
    submit_server = _build_submit_server(output_type, deps, validators, state, max_retries)

    servers: dict[str, McpServerConfig] = {**mcp_servers, SUBMIT_SERVER_NAME: submit_server}
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        mcp_servers=servers,
        allowed_tools=[*allowed_tools, submit_tool_ref()],
        max_turns=max_turns,
        cwd=str(cwd) if cwd is not None else None,
        env=env or {},
        # SDK isolation mode: load NO filesystem settings. Without this the SDK
        # defaults to loading ~/.claude + project + local settings.json, so the
        # curator would inherit the user's permissions, hooks, and project MCP
        # servers — none of which a background memory curator should have. With
        # ``[]`` it gets ONLY the read-only research tools + submit_result we
        # pass explicitly here. Applies to both roles (both go through run_typed).
        # Also a second layer against the recursion loop: the user's event-
        # appending hooks aren't even loaded into the curator's own session.
        setting_sources=[],
    )

    run = _query or query
    submitted = False
    stream = run(prompt=_single_user_message(prompt), options=options)
    try:
        async for message in stream:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.name.endswith(SUBMIT_TOOL_NAME):
                        submitted = True
            elif isinstance(message, ResultMessage):
                state.cost_usd = float(message.total_cost_usd or 0.0)
    except asyncio.CancelledError:
        # A wait_for timeout (or daemon shutdown) cancelled us mid-stream.
        # Close the SDK generator NOW: its close path terminates the spawned
        # CLI subprocess (SIGTERM, then SIGKILL after grace). Abandoning the
        # generator instead leaks a live `claude` process — the cancelled
        # task still references it mid-iteration, so not even asyncio.run's
        # shutdown_asyncgens reaches it. shield() keeps the close itself from
        # being re-cancelled; the wait_for bounds a wedged transport so the
        # timeout path can't hang.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(aclose()), timeout=15.0)
        raise

    if state.result is None:
        detail = state.last_error if submitted else "the model never called submit_result"
        raise TypedAgentError(
            f"no valid result after {state.attempts} attempt(s): {detail}",
            attempts=state.attempts,
            last_error=state.last_error,
        )

    return TypedResult(output=state.result, cost_usd=state.cost_usd, attempts=state.attempts)
