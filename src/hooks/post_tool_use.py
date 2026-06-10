"""PostToolUse hook logic — append a turn-event to the daemon's input stream.

Fires after every tool call. We emit one tiny event line and exit. The
full tool input/output is intentionally NOT included; it's already in
the transcript and would blow the per-line PIPE_BUF budget on big Read
or Bash payloads. The daemon's Librarian reads the transcript when it
needs more context.

Payload shape (frozen)::

    {"tool_name": "<name>", "ok": true|false}

``ok`` is best-effort: Claude Code's PostToolUse hook fires for both
success and failure (there's also a separate PostToolUseFailure event,
but configuring both would mean the daemon sees every failure twice).
We infer success from presence of ``tool_response`` and absence of
``error``.

Latency target: < 5 ms. No SDK calls, no subprocesses, no logging.

This module holds the testable logic; ``post-tool-use.py`` is the thin
shim Claude Code invokes via settings.json.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, cast

from _events import EventPayload, HookPayload, append_event
from _hookutil import parse_stdin


def _first_str_field(d: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty string field from ``d`` keyed by ``keys``."""
    for key in keys:
        v: Any = d.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _response_text(tool_response: object) -> str:
    """Flatten the ``tool_response`` field to a plain string."""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        resp = cast("Mapping[str, Any]", tool_response)
        return str(resp.get("content") or resp.get("output") or "")
    return ""


def _build_payload(hook_input: HookPayload) -> EventPayload:
    """Assemble the small payload dict that ships with the PostToolUse event.

    Field names align with what the daemon's Librarian extracts (see
    ``librarian_prompt._TEXT_KEYS``): ``content`` gets the actual code
    / command / new_string when present; ``summary`` gets a one-line
    synthesized description as a fallback.
    """
    raw_name: Any = hook_input.get("tool_name") or "unknown"
    name = str(raw_name) if not isinstance(raw_name, str) else raw_name
    # Success heuristic: PostToolUse carries ``tool_response``; the
    # corresponding failure event carries ``error`` instead. Default to
    # True so we don't falsely flag every event when the shape is odd.
    ok = not hook_input.get("error")

    raw_tool_input: Any = hook_input.get("tool_input") or {}
    target = ""
    content = ""
    if isinstance(raw_tool_input, dict):
        tool_input = cast("Mapping[str, Any]", raw_tool_input)
        target = _first_str_field(tool_input, ("file_path", "path", "url", "pattern", "query"))
        content = _first_str_field(tool_input, ("new_string", "content", "command", "code"))

    bits = [name]
    if target:
        bits.append(f"on {target}")
    bits.append("ok" if ok else "FAILED")

    return {
        "role": "assistant",
        "tool": name,
        "ok": ok,
        "summary": " ".join(bits),
        "content": content,
        "response": _response_text(hook_input.get("tool_response")),
    }


def main() -> None:
    # Recursion guard. If we're inside an SDK call spawned by flush.py,
    # emitting a PostToolUse event for every Read/Edit the SDK does
    # would flood the daemon with ghost events.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    hook_input = parse_stdin()
    if hook_input is None:
        return

    append_event("PostToolUse", hook_input, payload=_build_payload(hook_input))


if __name__ == "__main__":  # pragma: no cover
    main()
