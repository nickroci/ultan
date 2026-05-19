"""Claude Agent SDK invocation wrappers shared by the Librarian and Scholar.

Both roles call ``claude_agent_sdk.query(...)`` but with different option
shapes. This module owns the SDK-touching code so the role modules stay
focused on prompt assembly and decision logic.

Two entry points:

- ``run_librarian_call(prompt, *, timeout_s)`` — text-only call, no tools,
  Haiku tier. Returns the concatenated text response and cost.

- ``run_scholar_call(prompt, *, cwd, timeout_s)`` — file-tool-enabled call,
  Opus tier, ``permission_mode="acceptEdits"``. Returns the concatenated
  text response and cost.

Both wrappers also set ``CLAUDE_INVOKED_BY`` in the spawned SDK process's
environment (via the ``env=`` option) so any user-installed hook can detect
the recursion-blocked context and bail. Same pattern as `flush.py`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Tuple

log = logging.getLogger("agent_mem_daemon.llm")


# Models. Originally Haiku for the Librarian, but Haiku-tier was
# under-extracting even textbook user preferences in live testing —
# the recall tier needs a stronger model. Sonnet is the right balance:
# cheaper than Opus, much more reliable than Haiku at "is this a
# preference worth capturing" judgement.
LIBRARIAN_MODEL = "claude-sonnet-4-6"
SCHOLAR_MODEL = "claude-opus-4-7"

# Both calls run on the daemon's background loop — never on the hot path
# of a user-facing agent turn. Generous timeouts are fine; we'd rather
# wait than drop a packet to a transient Haiku slow-call.
LIBRARIAN_TIMEOUT_S = 600.0
SCHOLAR_TIMEOUT_S = 600.0


class LLMTimeout(RuntimeError):
    pass


def _make_path_guard(boundary: Path, *, allow_writes: bool):
    """Return a ``can_use_tool`` callback that rejects any tool call
    referencing a path outside ``boundary``.

    This is INFRA-level enforcement — we do NOT trust the prompt to
    constrain paths. The Scholar previously read the user's real
    ``~/.agent-mem/`` even when given ``cwd=/tmp/ulttest`` because the
    LLM produced absolute paths. This callback prevents that class of
    bug entirely: any tool input whose path resolves outside the
    knowledge root is denied with a clear error.

    Args:
        boundary: the directory the role is allowed to touch. Any path
            that does not resolve inside this directory is denied.
        allow_writes: if False, Write/Edit are denied outright (read-only
            role like the Librarian). Read/Glob/Grep are always
            path-constrained.
    """
    root = boundary.expanduser().resolve()

    # tool name → list of input keys that contain a filesystem path.
    PATH_KEYS = {
        "Read": ["file_path"],
        "Write": ["file_path"],
        "Edit": ["file_path"],
        "Glob": ["path"],
        "Grep": ["path"],
        "NotebookEdit": ["notebook_path"],
    }

    WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}

    # In-process MCP tools that don't take filesystem paths from the
    # model. They're safe by construction because the daemon pinned
    # their scope when it registered the server. Allow without path
    # validation.
    PATH_FREE_TOOLS = {
        "mcp__agent_mem_library__bm25_search",
    }

    async def _can_use_tool(tool_name, tool_input, context):  # noqa: ARG001
        from claude_agent_sdk.types import (
            PermissionResultAllow,
            PermissionResultDeny,
        )

        if not allow_writes and tool_name in WRITE_TOOLS:
            return PermissionResultDeny(
                message=f"tool {tool_name!r} is not allowed in this role",
            )

        if tool_name in PATH_FREE_TOOLS:
            return PermissionResultAllow(updated_input=tool_input)

        if tool_name not in PATH_KEYS:
            # Other tools (Bash, WebFetch, etc.) — deny by default. The
            # whole point of this guard is to keep the role inside the
            # library. If a future role legitimately needs another tool,
            # add it to PATH_KEYS (with appropriate path validation) or
            # to an explicit safe-list.
            return PermissionResultDeny(
                message=f"tool {tool_name!r} is not in the allow-list for this role",
            )

        for key in PATH_KEYS[tool_name]:
            raw = tool_input.get(key)
            if raw is None:
                # Optional path key (e.g. Glob without an explicit path
                # defaults to cwd, which we've already set inside root).
                continue
            try:
                resolved = Path(str(raw)).expanduser().resolve()
                resolved.relative_to(root)
            except (ValueError, OSError):
                return PermissionResultDeny(
                    message=(
                        f"path {raw!r} is outside the knowledge dir "
                        f"({root}); the {tool_name} call has been denied. "
                        f"Use a path under {root} instead."
                    ),
                )

        return PermissionResultAllow(updated_input=tool_input)

    return _can_use_tool


def _recursion_guard_env() -> dict:
    """Marker env vars passed into every SDK call.

    The hook layer is meant to check ``CLAUDE_INVOKED_BY`` and skip its
    work if set. Matches `src/scripts/flush.py`. We also inherit our own
    environment, which is what `asyncio.create_subprocess_*` does by
    default — but the SDK takes an explicit ``env=`` dict, so we pass it.
    """
    env = dict(os.environ)
    env["CLAUDE_INVOKED_BY"] = "agent_mem_daemon"
    return env


async def _drain_query(
    prompt: str,
    options,
) -> Tuple[str, float]:
    """Run one ``query(...)`` and return (full_text, cost_usd).

    The prompt is always passed as a streaming AsyncIterable rather
    than a bare string because the SDK requires streaming mode whenever
    a ``can_use_tool`` callback is set (see SDK ``client.py``: bare
    strings raise ``ValueError`` once a permission callback is wired
    up). Streaming with a single yielded message is functionally
    identical to one-shot mode but unlocks the permission-guard path.
    """
    # Lazy import so a missing SDK doesn't kill the daemon on startup —
    # only the actual Librarian/Scholar paths need it.
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
        query,
    )

    async def _prompt_stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": prompt},
        }

    full = ""
    cost = 0.0
    async for message in query(prompt=_prompt_stream(), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    full += block.text
                elif isinstance(block, ToolUseBlock):
                    # Inline a marker so the transcript shows what tools
                    # were called. Critical diagnostic — the Scholar's
                    # writes happen via tools, not in the text stream.
                    input_repr = str(block.input)[:300]
                    full += f"\n[tool: {block.name}({input_repr})]\n"
        elif isinstance(message, ResultMessage):
            # ResultMessage carries the cost the SDK reports for the run.
            cost = float(message.total_cost_usd or 0.0)
    return full, cost


def _run_with_timeout(coro, timeout_s: float):
    """Wrap a coroutine in ``asyncio.run`` with a timeout. Raises
    LLMTimeout on expiry."""
    async def _wrapped():
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError as e:
            raise LLMTimeout(f"SDK call exceeded {timeout_s}s") from e

    return asyncio.run(_wrapped())


def run_librarian_call(
    prompt: str,
    *,
    cwd: Path | None = None,
    timeout_s: float = LIBRARIAN_TIMEOUT_S,
    model: str = LIBRARIAN_MODEL,
) -> Tuple[str, float]:
    """Run the Librarian Haiku call with read-only tools.

    The Librarian needs Read+Glob so it can inspect specific entries
    before proposing actions — the snapshot in the prompt is a teaser,
    not the full corpus. Tools are READ-ONLY: no Write, no Edit. The
    Scholar is the only writer.

    Args:
        prompt: full assembled prompt.
        cwd: working directory for the SDK — should be the knowledge
            store root so Read/Glob land in the right tree. Required when
            the Librarian is allowed to inspect files.
        timeout_s: SDK timeout.
        model: override the default Haiku model.

    Returns (response_text, cost_usd). Raises:
      - LLMTimeout if the call exceeded ``timeout_s``.
      - Any other exception the SDK raises (caller should catch and log).
    """
    # Lazy import so daemon startup doesn't require the SDK to be present.
    from claude_agent_sdk import ClaudeAgentOptions
    from . import library_tools

    librarian_tools = ["Read", "Glob", "Grep"]
    mcp_servers: dict = {}

    if cwd is not None:
        # Register the in-process BM25 search tool so the Librarian can
        # find semantically related entries without the daemon doing a
        # regex pre-pass. Complements Glob (filename) and Grep (literal).
        mcp_servers[library_tools._SERVER_NAME] = library_tools.make_library_mcp_server(cwd)
        librarian_tools.append(library_tools.fully_qualified_tool_name())

    opts: dict = dict(
        model=model,
        allowed_tools=librarian_tools,
        mcp_servers=mcp_servers,
        # 10 turns: ~6 Read/Glob/bm25 calls + the JSON emission. Plenty of
        # slack without letting Haiku wander.
        max_turns=10,
        env=_recursion_guard_env(),
        # The claude_code preset is what lets Read/Glob work end-to-end.
        # Without it the SDK exposes the tools but the model's harness
        # doesn't know how to act on results.
        system_prompt={"type": "preset", "preset": "claude_code"},
    )
    if cwd is not None:
        opts["cwd"] = str(cwd)
        # Infra-level path guard: deny any Read/Glob/Grep outside the
        # knowledge dir. Without this, the LLM can pass absolute paths
        # and bypass cwd. The bm25_search tool is path-safe by
        # construction (knowledge_dir captured in closure), so it
        # doesn't need guard coverage.
        opts["can_use_tool"] = _make_path_guard(cwd, allow_writes=False)
    options = ClaudeAgentOptions(**opts)
    log.debug(
        "librarian SDK call: model=%s cwd=%s prompt_chars=%d",
        model, cwd, len(prompt),
    )
    return _run_with_timeout(_drain_query(prompt, options), timeout_s)


def run_scholar_call(
    prompt: str,
    *,
    cwd: Path,
    timeout_s: float = SCHOLAR_TIMEOUT_S,
    model: str = SCHOLAR_MODEL,
) -> Tuple[str, float]:
    """Run the Scholar's file-tool-enabled Opus call.

    Args:
        prompt: full assembled prompt (template + packets).
        cwd: working directory for the SDK — should be the store dir
             (``~/.agent-mem/``), so Read/Glob/Grep land in the right
             tree. The Scholar writes via Write/Edit relative to this.

    Returns (response_text, cost_usd). Raises LLMTimeout on timeout; the
    caller logs and skips the batch.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    # Scholar's cwd is the agent-mem home (so prompt-relative paths like
    # ``knowledge/index.md`` resolve correctly). The guard's boundary is
    # the narrower ``knowledge/`` subdir — Scholar must not touch
    # events.jsonl, daemon.log, runs/, cost.json, etc.
    boundary = cwd / "knowledge"
    options = ClaudeAgentOptions(
        model=model,
        cwd=str(cwd),
        system_prompt={"type": "preset", "preset": "claude_code"},
        allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
        permission_mode="acceptEdits",
        max_turns=30,
        env=_recursion_guard_env(),
        # Infra-level path guard: every Read/Write/Edit/Glob/Grep is
        # rejected if its path resolves outside the knowledge dir.
        # Stops the "Scholar edits the wrong store" class of bug at
        # the SDK layer, independent of whatever the prompt says.
        can_use_tool=_make_path_guard(boundary, allow_writes=True),
    )
    log.debug(
        "scholar SDK call: model=%s cwd=%s prompt_chars=%d",
        model, cwd, len(prompt),
    )
    return _run_with_timeout(_drain_query(prompt, options), timeout_s)
