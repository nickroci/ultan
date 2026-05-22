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

import json
import os
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _events import append_event  # noqa: E402


def _parse_stdin() -> dict | None:
    """Parse the JSON hook_input from stdin; return None on any failure.

    Same Windows-backslash workaround as the other hooks — Claude Code
    on Windows sometimes emits paths with lone backslashes that aren't
    valid JSON escapes.
    """
    try:
        raw = sys.stdin.read()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            fixed = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw)
            parsed = json.loads(fixed)
    except (json.JSONDecodeError, ValueError, EOFError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _first_str_field(d: dict, keys: tuple[str, ...]) -> str:
    """Return the first non-empty string field from ``d`` keyed by ``keys``."""
    for key in keys:
        v = d.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _response_text(tool_response: object) -> str:
    """Flatten the ``tool_response`` field to a plain string."""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        return str(tool_response.get("content") or tool_response.get("output") or "")
    return ""


def _build_payload(hook_input: dict) -> dict:
    """Assemble the small payload dict that ships with the PostToolUse event.

    Field names align with what the daemon's Librarian extracts (see
    ``librarian_prompt._TEXT_KEYS``): ``content`` gets the actual code
    / command / new_string when present; ``summary`` gets a one-line
    synthesized description as a fallback.
    """
    name = str(hook_input.get("tool_name") or "unknown")
    # Success heuristic: PostToolUse carries ``tool_response``; the
    # corresponding failure event carries ``error`` instead. Default to
    # True so we don't falsely flag every event when the shape is odd.
    ok = not (hook_input.get("error"))

    tool_input = hook_input.get("tool_input") or {}
    target = ""
    content = ""
    if isinstance(tool_input, dict):
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

    hook_input = _parse_stdin()
    if hook_input is None:
        return

    append_event("PostToolUse", hook_input, payload=_build_payload(hook_input))


if __name__ == "__main__":  # pragma: no cover
    main()
